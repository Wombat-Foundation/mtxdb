use super::*;
use std::collections::HashMap;

fn test_hash(byte: u8) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0] = byte;
    h
}

/// Deterministic 16-byte hashes (splitmix64), so tag collisions occur
/// among a few thousand of them.
fn spread_hash(seq: u64) -> [u8; 16] {
    let mut state = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut hash = [0u8; 16];
    hash[..8].copy_from_slice(&next().to_be_bytes());
    hash[8..].copy_from_slice(&next().to_be_bytes());
    hash
}

/// Every occupied bucket carries a hydrated identity (the tail's known
/// marker) and a home from which the entry is reachable by linear probing
/// without crossing an empty slot: what a rehash from `homes`/`tails`
/// needs to be able to trust.
fn identities_complete(index: &LossyIndex) -> bool {
    let homes = index.homes.lock();
    let tails = index.tails.lock();
    (0..index.capacity as usize).all(|bucket| {
        if IndexEntry(index.entry_at(bucket)).is_empty() {
            return true;
        }
        if tails[bucket] >> 63 == 0 {
            return false;
        }
        let mut probe = index.bucket_for_home(homes[bucket]);
        while probe != bucket {
            if IndexEntry(index.entry_at(probe)).is_empty() {
                return false;
            }
            probe = probe.wrapping_add(1) & index.mask as usize;
        }
        true
    })
}

fn candidates(index: &LossyIndex, hash: &[u8; 16]) -> Vec<(u16, u64)> {
    let mut found: Vec<(u16, u64)> = index.lookup_all(hash).collect();
    found.sort_unstable();
    found
}

/// Whether a checkpoint-loaded index grows from its persisted `homes` and
/// `tails`: build a live index, persist and load it, and check (a) a fully
/// hydrated load has complete
/// identities, and growing it with the existing rehash gives the same keys,
/// locators, length, per-pack counts and collision candidates; (b) a load
/// whose identity side tables lost entries is detected as incomplete, does
/// not grow, and the pack-based growth (`grow_by_recovering_hashes`) gives the
/// same result.
#[test]
fn a_hydrated_checkpoint_loaded_index_can_grow_from_its_identity_tables() {
    let config = IndexConfig {
        seed: 0x5EED_1234_ABCD_0042,
        ..IndexConfig::default()
    };
    let live = LossyIndex::with_config(4096, config);
    let mut locators: HashMap<(u16, u64), [u8; 16]> = HashMap::new();
    let mut expected: Vec<([u8; 16], (u16, u64))> = Vec::new();
    for seq in 0..3000u64 {
        let hash = spread_hash(seq);
        let locator = (u16::try_from(seq % 8).unwrap(), seq * 64);
        live.insert(&hash, locator.0, locator.1).unwrap();
        locators.insert(locator, hash);
        expected.push((hash, locator));
    }
    let collisions = expected
        .iter()
        .filter(|(hash, _)| live.lookup_all(hash).count() > 1)
        .count();
    assert!(
        (0..3000u64).any(|seq| {
            let tag = live.tag_for_hash(&spread_hash(seq));
            (seq + 1..3000).any(|other| live.tag_for_hash(&spread_hash(other)) == tag)
        }),
        "the workload must contain tag collisions to exercise them"
    );
    let _ = collisions;

    // (a) fully hydrated checkpoint load.
    let blob = live.serialize();
    let loaded = LossyIndex::deserialize_with_config(&blob, config).unwrap();
    assert!(
        identities_complete(&loaded),
        "v6 blobs persist full identities"
    );
    let grown = loaded
        .grow()
        .expect("a complete hydrated load grows from its side tables");
    assert_eq!(grown.capacity(), live.capacity() * 2);
    assert_eq!(grown.len(), live.len());
    assert_eq!(grown.slot_counts(), live.slot_counts());
    for (hash, locator) in &expected {
        assert_eq!(grown.lookup(hash), Some(*locator));
        assert_eq!(candidates(&grown, hash), candidates(&live, hash));
    }
    assert!(
        identities_complete(&grown),
        "growth keeps identities complete"
    );

    // (b) identity side tables that lost entries.
    let slots_region = 8 + live.capacity() as usize * 8;
    let tails_region = slots_region + live.capacity() as usize * 8;
    let mut damaged = blob.clone();
    let mut zeroed = 0;
    for bucket in 0..live.capacity() as usize {
        if !IndexEntry(live.entry_at(bucket)).is_empty() && zeroed < 10 {
            let at = tails_region + bucket * 8;
            damaged[at..at + 8].fill(0);
            zeroed += 1;
        }
    }
    let partial = LossyIndex::deserialize_with_config(&damaged, config).unwrap();
    assert!(
        !identities_complete(&partial),
        "zeroed tails must be detected, not trusted"
    );
    assert!(partial.grow().is_none(), "and it must not grow from them");
    let recovered = partial
        .grow_by_recovering_hashes(|slot, offset, _tag| {
            locators.get(&(slot, offset)).copied().ok_or(())
        })
        .unwrap()
        .expect("pack-based growth still works");
    assert_eq!(recovered.len(), live.len());
    assert_eq!(recovered.slot_counts(), live.slot_counts());
    for (hash, locator) in &expected {
        assert_eq!(recovered.lookup(hash), Some(*locator));
    }

    // A blob with no identity tables at all (pre-v4) is incomplete too.
    let bare = LossyIndex::deserialize_with_config(&blob[..slots_region], config).unwrap();
    assert!(!identities_complete(&bare));
}

/// The write path after a checkpoint reopen: the index is loaded over the
/// mapping (with its persisted identity tables), then cloned into owned
/// storage on the first write. If that clone drops the tables, every entry
/// loses its identity and the *next* checkpoint persists zeros, so a
/// hydrated-only growth gate would fail after any reopen.
#[test]
fn a_clone_of_a_checkpoint_mapped_index_keeps_its_identity_tables() {
    let config = IndexConfig {
        seed: 0x5EED_1234_ABCD_0042,
        ..IndexConfig::default()
    };
    let live = LossyIndex::with_config(1024, config);
    for seq in 0..600u64 {
        live.insert(&spread_hash(seq), u16::try_from(seq % 4).unwrap(), seq * 64)
            .unwrap();
    }
    assert!(identities_complete(&live));
    let path = std::env::temp_dir().join(format!(
        "mtxdb-index-identity-{}-{}.bin",
        std::process::id(),
        live.capacity()
    ));
    std::fs::write(&path, live.serialize()).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = Arc::new(crate::packfile::map_pack(&file).unwrap());
    let capacity = live.capacity() as usize;
    let mapped = LossyIndex::from_mmap_slots_with_homes_tails(
        mmap,
        8,
        8 + capacity * 8,
        8 + capacity * 16,
        live.capacity(),
        u32::try_from(live.len()).unwrap(),
        config,
    );
    assert!(mapped.is_mmap_backed());
    assert!(
        identities_complete(&mapped),
        "the mapped load holds the persisted identities"
    );
    let owned = mapped.clone();
    let complete_after_clone = identities_complete(&owned);
    let _ = std::fs::remove_file(&path);
    assert!(
        complete_after_clone,
        "cloning a mapped index into owned storage dropped its identity tables"
    );
}

/// The whole lifecycle a real store goes through: load a checkpoint mapped
/// with its identity tables, clone on the first write, overwrite an existing
/// identity and add a new one, checkpoint again, reload, then grow. Every
/// entry must keep a complete identity throughout, and the grown index must
/// answer as the pre-growth one did, including tag collisions.
#[test]
fn identities_survive_reopen_write_checkpoint_reopen_and_growth() {
    let config = IndexConfig {
        seed: 0x5EED_1234_ABCD_0042,
        ..IndexConfig::default()
    };
    let live = LossyIndex::with_config(4096, config);
    let mut locators: HashMap<(u16, u64), [u8; 16]> = HashMap::new();
    let mut expected: HashMap<[u8; 16], (u16, u64)> = HashMap::new();
    for seq in 0..2500u64 {
        let hash = spread_hash(seq);
        let locator = (u16::try_from(seq % 4).unwrap(), seq * 64);
        live.insert(&hash, locator.0, locator.1).unwrap();
        locators.insert(locator, hash);
        expected.insert(hash, locator);
    }
    let dir = std::env::temp_dir().join(format!("mtxdb-index-lifecycle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let load_mapped = |index: &LossyIndex, name: &str| {
        let path = dir.join(name);
        std::fs::write(&path, index.serialize()).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let capacity = index.capacity() as usize;
        LossyIndex::from_mmap_slots_with_homes_tails(
            Arc::new(crate::packfile::map_pack(&file).unwrap()),
            8,
            8 + capacity * 8,
            8 + capacity * 16,
            index.capacity(),
            u32::try_from(index.len()).unwrap(),
            config,
        )
    };

    // Reopen 1: mapped, then cloned by the first write.
    let mapped = load_mapped(&live, "first.bin");
    assert!(identities_complete(&mapped));
    let owned = mapped.clone();
    assert!(identities_complete(&owned), "the clone keeps identities");
    // Overwrite an existing identity with a new locator, and add a new one.
    let overwritten = spread_hash(7);
    owned.insert(&overwritten, 2, 999_936).unwrap();
    locators.insert((2, 999_936), overwritten);
    expected.insert(overwritten, (2, 999_936));
    let added = spread_hash(90_000);
    owned.insert(&added, 3, 1_000_000).unwrap();
    locators.insert((3, 1_000_000), added);
    expected.insert(added, (3, 1_000_000));
    assert!(
        identities_complete(&owned),
        "writes keep identities complete"
    );
    assert_eq!(owned.len(), live.len() + 1);

    // Checkpoint, reopen 2, and grow the reloaded index from its tables.
    let reopened = load_mapped(&owned, "second.bin").clone();
    assert!(
        identities_complete(&reopened),
        "the second reopen is complete too"
    );
    let grown = reopened.grow().expect("grows from persisted identities");
    assert_eq!(grown.len(), owned.len());
    assert_eq!(grown.slot_counts(), owned.slot_counts());
    for (hash, locator) in &expected {
        assert_eq!(grown.lookup(hash), Some(*locator), "locator after growth");
        assert_eq!(candidates(&grown, hash), candidates(&owned, hash));
    }
    assert!(identities_complete(&grown));

    // Memory: the clone holds owned slots plus both tables, 24 bytes per slot
    // (what `memory_usage` charges a live index); the mapped source held the
    // two tables in memory already, with its slots in the page cache.
    assert_eq!(
        owned.homes.lock().len() + owned.tails.lock().len(),
        2 * owned.capacity() as usize
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Delta replay stores slot values straight into the table, with no
/// identity. An index that was complete when it was loaded is therefore
/// incomplete once a delta has been replayed onto it, and growing it from
/// its identity tables would place the replayed entries by zeros and lose
/// them. Growth must re-check completeness when it is used, not trust the
/// load-time answer (a power-cut recovery test lost a synced record to this).
#[test]
fn a_replayed_delta_makes_a_loaded_index_incomplete_so_it_must_not_grow() {
    let config = IndexConfig {
        seed: 0x5EED_1234_ABCD_0042,
        ..IndexConfig::default()
    };
    let live = LossyIndex::with_config(256, config);
    for seq in 0..100u64 {
        live.insert(&spread_hash(seq), 1, seq * 64).unwrap();
    }
    let loaded = LossyIndex::deserialize_with_config(&live.serialize(), config).unwrap();
    assert!(loaded.can_grow, "a complete load enables growth");
    assert!(loaded.grow().is_some(), "and grows from its tables");

    let empty_bucket = (0..loaded.capacity() as usize)
        .find(|bucket| IndexEntry(loaded.entry_at(*bucket)).is_empty())
        .unwrap();
    let frame = DeltaFrame {
        collection_id: [9; 16],
        bucket: u32::try_from(empty_bucket).unwrap(),
        generation: 0,
        slot: IndexEntry::new(0x1234, 2, 4096).0,
    };
    loaded.replay_frames(&[frame]).unwrap();
    assert!(
        !loaded.identities_now_complete(),
        "the replayed entry has no identity"
    );
    assert!(
        loaded.grow().is_none(),
        "so growth falls back to the pack-based path instead of losing it"
    );
}

#[test]
fn test_slot_packing() {
    let entry = IndexEntry::new(0xCDEF, 42, 0x0FFF_FFF0);
    assert_eq!(entry.tag(), 0xCDEF);
    assert_eq!(entry.slot(), 42);
    assert_eq!(entry.offset(), 0x0FFF_FFF0);
    assert!(!entry.is_empty());
}

#[test]
fn new_accepts_the_highest_live_slot_and_rejects_the_next() {
    let max_live = u16::try_from(crate::shard::MAX_SHARDS - 1).unwrap();
    let entry = IndexEntry::new(0, max_live, 0);
    assert_eq!(entry.slot(), max_live);

    let out_of_range = u16::try_from(crate::shard::MAX_SHARDS).unwrap();
    let result = std::panic::catch_unwind(|| IndexEntry::new(0, out_of_range, 0));
    assert!(
        result.is_err(),
        "a slot at MAX_SHARDS is not a live shard and must be rejected"
    );
}

#[test]
fn test_slot_empty() {
    let entry = IndexEntry::empty();
    assert!(entry.is_empty());
    assert_eq!(entry.tag(), 0);
}

#[test]
fn test_insert_and_lookup() {
    let index = LossyIndex::new(128);
    let h1 = test_hash(0x01);
    let h2 = test_hash(0x02);
    let h3 = test_hash(0xFF);

    index.insert(&h1, 0, 100).unwrap();
    index.insert(&h2, 1, 200).unwrap();
    index.insert(&h3, 0, 999).unwrap();

    assert_eq!(index.len(), 3);
    assert_eq!(index.lookup(&h1), Some((0, 100)));
    assert_eq!(index.lookup(&h2), Some((1, 200)));
    assert_eq!(index.lookup(&h3), Some((0, 999)));
    assert_eq!(index.lookup(&[0xFE; 16]), None);
}

#[test]
fn grow_by_recovering_hashes_rehashes_only_occupied_locations() {
    let index = LossyIndex::new(16);
    let mut hashes = HashMap::new();
    for offset in 0..12u64 {
        let mut hash = [0u8; 16];
        hash[..8].copy_from_slice(&(offset.wrapping_mul(17).wrapping_add(1)).to_be_bytes());
        hash[8] = u8::try_from(offset.wrapping_add(1)).unwrap();
        index.insert(&hash, 3, offset).unwrap();
        hashes.insert((3, offset), hash);
    }

    let mut recovered = 0usize;
    let grown = index
        .grow_by_recovering_hashes(|shard, offset, _tag| {
            recovered = recovered.saturating_add(1);
            Ok::<_, ()>(hashes[&(shard, offset)])
        })
        .unwrap()
        .expect("a 16-entry table can double");

    assert_eq!(recovered, 12, "recover exactly one hash per occupied entry");
    assert_eq!(grown.capacity, 32);
    for ((shard, offset), hash) in hashes {
        assert_eq!(grown.lookup(&hash), Some((shard, offset)));
    }
}

#[test]
fn test_linear_probing() {
    let index = LossyIndex::new(16); // small table
    for i in 0..10u8 {
        let mut h = [0u8; 16];
        h[0] = i;
        h[1] = 0xFF; // different second byte to avoid tag collisions
        index.insert(&h, 0, u64::from(i) * 100).unwrap();
    }
    for i in 0..10u8 {
        let mut h = [0u8; 16];
        h[0] = i;
        h[1] = 0xFF;
        assert!(index.lookup(&h).is_some());
    }
}

#[test]
fn test_empty_terminates_probe() {
    let index = LossyIndex::new(16);
    let h = test_hash(0x42);
    assert_eq!(index.lookup(&h), None);
}

#[test]
fn test_table_full_returns_error() {
    let index = LossyIndex::new(16); // capacity 16, threshold at 75% = 12
                                     // Insert hashes until we get 12 occupied slots.
    let mut inserted = 0u64;
    for i in 0..1000u64 {
        let h = splitmix_hash(i);
        if index.insert(&h, 0, i).is_ok() {
            inserted += 1;
            if inserted >= 12 {
                break;
            }
        }
    }
    assert_eq!(index.len(), 12);
    let h = splitmix_hash(9999);
    assert!(matches!(
        index.insert(&h, 0, 999),
        Err(InsertError::TableFull)
    ));
}

#[test]
fn test_is_empty_and_memory_usage() {
    let index = LossyIndex::new(128);
    assert!(index.is_empty());
    assert_eq!(
        index.memory_usage(),
        128 * LIVE_SLOT_BYTES + std::mem::size_of::<LossyIndex>()
    );

    let index = LossyIndex::new(128);
    index.insert(&test_hash(0x01), 0, 1).unwrap();
    assert!(!index.is_empty());
}

#[test]
fn max_probe_len_tracks_the_longest_insert_chain() {
    let index = LossyIndex::new(16);
    assert_eq!(
        index.max_probe_len(),
        0,
        "a fresh index has walked no probes"
    );
    // Five hashes sharing identical bytes 0..8 (home) land in the same
    // bucket regardless of seed/mix, forcing a deterministic 5-entry
    // linear-probe chain: the Nth insert walks N-1 occupied slots.
    for i in 0..5u8 {
        let mut h = [0u8; 16];
        h[9] = i; // varies tag (bytes 8..12) so entries stay distinct
        h[15] = i;
        index.insert(&h, 0, u64::from(i)).unwrap();
    }
    assert_eq!(
        index.max_probe_len(),
        4,
        "the 5th colliding insert walks past the 4 entries ahead of it"
    );
}

#[test]
fn max_probe_len_tracks_lookup_chains_independently_of_insert() {
    let source = LossyIndex::new(16);
    let mut hashes = Vec::new();
    for i in 0..5u8 {
        let mut h = [0u8; 16];
        h[9] = i;
        h[15] = i;
        source.insert(&h, 0, u64::from(i)).unwrap();
        hashes.push(h);
    }
    // Deserializing does not itself walk any probes.
    let restored = LossyIndex::deserialize(&source.serialize()).unwrap();
    assert_eq!(restored.max_probe_len(), 0);

    let last = *hashes.last().expect("5 hashes inserted");
    assert!(restored.lookup(&last).is_some());
    assert!(
        restored.max_probe_len() >= 4,
        "looking up the deepest-chained entry must walk at least as \
             far as its insert-time chain length, got {}",
        restored.max_probe_len()
    );
}

#[test]
fn test_grow_preserves_existing_lookups() {
    let index = LossyIndex::new(16);
    let hashes: Vec<_> = (0..12).map(splitmix_hash).collect();
    for (offset, hash) in hashes.iter().enumerate() {
        index.insert(hash, 0, offset as u64).unwrap();
    }
    let index = index.grow().expect("live index can grow");
    assert_eq!(index.len(), hashes.len());
    for (offset, hash) in hashes.iter().enumerate() {
        assert_eq!(index.lookup(hash), Some((0, offset as u64)));
    }
    index.insert(&splitmix_hash(12), 0, 12).unwrap();
}

#[test]
fn test_rollback_slot_undoes_fresh_insert() {
    let index = LossyIndex::new(128);
    let h1 = test_hash(0x01);
    let h2 = test_hash(0x02);
    index.insert(&h1, 0, 100).unwrap();
    assert_eq!(index.len(), 1);

    let (_, _, undo) = index.insert_undoable(&h2, 1, 200).unwrap();
    assert_eq!(index.len(), 2);
    assert_eq!(index.lookup(&h2), Some((1, 200)));

    index.rollback_slot(&undo);
    assert_eq!(
        index.len(),
        1,
        "rollback of a fresh-entry insert must restore len"
    );
    assert_eq!(
        index.lookup(&h2),
        None,
        "rolled-back entry must not be findable"
    );
    assert_eq!(
        index.lookup(&h1),
        Some((0, 100)),
        "rollback must not disturb an unrelated entry"
    );
}

#[test]
fn test_rollback_slot_undoes_overwrite() {
    let index = LossyIndex::new(128);
    let h = test_hash(0x01);
    index.insert(&h, 0, 100).unwrap();
    assert_eq!(index.len(), 1);

    let (_, _, undo) = index.insert_undoable(&h, 1, 200).unwrap();
    assert_eq!(index.lookup(&h), Some((1, 200)));

    index.rollback_slot(&undo);
    assert_eq!(
        index.len(),
        1,
        "rollback of an overwrite must not change len"
    );
    assert_eq!(
        index.lookup(&h),
        Some((0, 100)),
        "rollback of an overwrite must restore the prior location"
    );
}

#[test]
fn test_overwrite_same_hash() {
    let index = LossyIndex::new(128);
    let h = test_hash(0x01);
    index.insert(&h, 0, 100).unwrap();
    index.insert(&h, 1, 200).unwrap(); // overwrite
    assert_eq!(index.len(), 1);
    assert_eq!(index.lookup(&h), Some((1, 200)));
}

#[test]
fn same_tag_and_home_but_distinct_hashes_both_remain_indexed() {
    let index = LossyIndex::new(16);
    let mut first = [0u8; 16];
    first[8] = 0x42;
    first[15] = 1;
    let mut second = first;
    second[15] = 2;

    index.insert(&first, 0, 100).unwrap();
    index.insert(&second, 0, 200).unwrap();

    assert_eq!(index.len(), 2, "a tag is not an overwrite proof");
    // Both coexist and are individually findable via their exact hash.
    assert_eq!(index.lookup(&first), Some((0, 100)));
    assert_eq!(index.lookup(&second), Some((0, 200)));
}

#[test]
fn replay_preserves_a_distinct_same_tag_slot() {
    let mut first = [0u8; 16];
    first[8] = 0x42;
    first[15] = 1;
    let mut second = first;
    second[15] = 2;

    // Model the checkpoint before the second write. Serialization omits
    // identities, as the compact checkpoint format intentionally does.
    let checkpoint_source = LossyIndex::new(16);
    checkpoint_source.insert(&first, 0, 100).unwrap();
    let checkpoint = LossyIndex::deserialize(&checkpoint_source.serialize()).unwrap();

    // The live writer records the exact bucket/entry placement for the
    // second colliding identity. Replay must preserve that placement; it
    // must not re-run tag-only insertion and drop either record.
    let live = LossyIndex::new(16);
    live.insert(&first, 0, 100).unwrap();
    let (bucket, entry) = live.insert_tracked(&second, 0, 200).unwrap();
    checkpoint
        .replay_frames(&[DeltaFrame {
            collection_id: [0; 16],
            bucket,
            generation: 0,
            slot: entry,
        }])
        .unwrap();

    assert_eq!(checkpoint.len(), 2);
    assert_eq!(
        checkpoint.lookup_all(&first).collect::<Vec<_>>(),
        vec![(0, 100), (0, 200)]
    );
}

#[test]
fn replay_rejects_a_frame_carrying_the_empty_sentinel() {
    let mut hash = [0u8; 16];
    hash[15] = 1;

    let checkpoint_source = LossyIndex::new(16);
    let (bucket, _slot) = checkpoint_source.insert_tracked(&hash, 0, 100).unwrap();
    let checkpoint = LossyIndex::deserialize(&checkpoint_source.serialize()).unwrap();
    let occupied_before = checkpoint.len();

    // A structurally-framed but invalid delta frame claiming the empty
    // sentinel for an already-occupied bucket must be rejected wholesale,
    // not silently applied — applying it would erase the live entry and
    // truncate the probe chain past it.
    let err = checkpoint
        .replay_frames(&[DeltaFrame {
            collection_id: [0; 16],
            bucket,
            generation: 0,
            slot: 0,
        }])
        .unwrap_err();
    assert!(matches!(err, DeltaReplayError::EmptySlot { bucket: b } if b == bucket));

    // The index must be left untouched: an owned clone is made before
    // replay, so this checks the caller's rescan fallback has a live
    // checkpoint copy to fall back to, not a partially-erased one.
    assert_eq!(checkpoint.len(), occupied_before);
    assert_eq!(checkpoint.lookup(&hash), Some((0, 100)));
}

#[test]
fn replay_rejects_a_frame_naming_a_slot_above_max_shards() {
    let mut hash = [0u8; 16];
    hash[15] = 1;

    let checkpoint_source = LossyIndex::new(16);
    let (bucket, _slot) = checkpoint_source.insert_tracked(&hash, 0, 100).unwrap();
    let checkpoint = LossyIndex::deserialize(&checkpoint_source.serialize()).unwrap();
    let occupied_before = checkpoint.len();

    // A frame whose packed entry decodes to a slot at MAX_SHARDS names no
    // live shard; a writer can never emit one (`IndexEntry::new` rejects
    // it), so replay must reject the whole log instead of installing an
    // entry no shard scan can resolve.
    let out_of_range = u16::try_from(crate::shard::MAX_SHARDS).unwrap();
    let bad_entry = IndexEntry::new(0, 0, 0).0 | (u64::from(out_of_range) << 32);
    let err = checkpoint
        .replay_frames(&[DeltaFrame {
            collection_id: [0; 16],
            bucket,
            generation: 0,
            slot: bad_entry,
        }])
        .unwrap_err();
    assert!(
        matches!(err, DeltaReplayError::SlotOutOfRange { bucket: b, slot } if b == bucket && slot == out_of_range)
    );
    assert_eq!(checkpoint.len(), occupied_before);
    assert_eq!(checkpoint.lookup(&hash), Some((0, 100)));
}

#[test]
fn test_serialize_roundtrip() {
    let index = LossyIndex::new(128);
    for i in 0..50u16 {
        let mut h = [0u8; 16];
        h[0] = (i & 0xFF) as u8;
        h[8] = (i & 0xFF) as u8; // distinct tag bytes
        h[9] = (i >> 8) as u8;
        index.insert(&h, i % 3, u64::from(i) * 1000 + 1).unwrap();
    }

    let bytes = index.serialize();
    let restored = LossyIndex::deserialize(&bytes).unwrap();

    assert_eq!(restored.len(), index.len());
    for i in 0..50u16 {
        let mut h = [0u8; 16];
        h[0] = (i & 0xFF) as u8;
        h[8] = (i & 0xFF) as u8;
        h[9] = (i >> 8) as u8;
        assert_eq!(restored.lookup(&h), index.lookup(&h));
    }
}

#[test]
fn test_power_of_two_capacity() {
    let index = LossyIndex::new(100);
    assert_eq!(index.capacity, 128);
    let index = LossyIndex::new(128);
    assert_eq!(index.capacity, 128);
    let index = LossyIndex::new(129);
    assert_eq!(index.capacity, 256);
}

// jscpd:ignore-start
// False-positive match against storage.rs's NodeRef::data() — token-shape
// coincidence (both short, similar brace/operator density), not related
// logic. Nothing to extract: one's a hash mixer, one's an enum accessor.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
// jscpd:ignore-end

fn splitmix_hash(i: u64) -> [u8; 16] {
    let a = splitmix64(i);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    let mut h = [0u8; 16];
    h[..8].copy_from_slice(&a.to_le_bytes());
    h[8..].copy_from_slice(&b.to_le_bytes());
    h
}

/// Regression test for a shift-formula bug: with `capacity` a `u32`,
/// `64 - capacity.leading_zeros()` does not equal `64 - log2(capacity)`,
/// so at larger capacities the bucket function only used a fraction of
/// its intended hash bits, collapsing most of the table into a handful
/// of buckets and causing O(n) probes per insert. `trailing_zeros` gives
/// the correct shift for any power-of-two capacity.
#[test]
fn test_shift_uses_full_bucket_range() {
    for &capacity in &[16u32, 1024, 65536, 131_072, 262_144] {
        let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());
        assert_eq!(shift, 64 - capacity.ilog2());
    }
}

/// Regression test: at scale, insert must not silently drop distinct
/// entries via spurious tag collisions caused by the tag and bucket
/// being derived from overlapping hash bits.
#[test]
fn test_insert_preserves_distinct_entries_at_scale() {
    let n = 50_000usize;
    let index = LossyIndex::new(n * 2);
    for i in 0..n {
        let h = splitmix_hash(i as u64);
        index.insert(&h, 0, i as u64).unwrap();
    }
    // No distinct entry should have been silently overwritten.
    assert_eq!(index.len(), n);
    for i in 0..n {
        let h = splitmix_hash(i as u64);
        assert_eq!(index.lookup(&h), Some((0, i as u64)));
    }
}

#[test]
fn test_deserialize_errors() {
    // TooShort: data shorter than 8 bytes
    assert!(matches!(
        LossyIndex::deserialize(&[0u8; 4]),
        Err(DeserializationError::TooShort)
    ));
    // TooShort: capacity header present but slots truncated
    let mut buf = vec![0u8; 16];
    buf[..8].copy_from_slice(&16u64.to_le_bytes());
    assert!(matches!(
        LossyIndex::deserialize(&buf),
        Err(DeserializationError::TooShort)
    ));
    // InvalidCapacity: capacity < 16
    let mut buf = vec![0u8; 8 + 8 * 8];
    buf[..8].copy_from_slice(&8u64.to_le_bytes());
    assert!(matches!(
        LossyIndex::deserialize(&buf),
        Err(DeserializationError::InvalidCapacity)
    ));
    // InvalidCapacity: capacity not power of two
    let mut buf = vec![0u8; 8 + 20 * 8];
    buf[..8].copy_from_slice(&20u64.to_le_bytes());
    assert!(matches!(
        LossyIndex::deserialize(&buf),
        Err(DeserializationError::InvalidCapacity)
    ));
}

#[test]
fn test_error_display() {
    assert_eq!(InsertError::TableFull.to_string(), "index table too full");
    assert_eq!(DeserializationError::TooShort.to_string(), "data too short");
    assert_eq!(
        DeserializationError::InvalidCapacity.to_string(),
        "capacity must be a power of two and >= 16"
    );
}

#[test]
fn test_lookup_iter_exhausts_remaining() {
    // Build a 16-entry index with all 16 slots occupied (no empty
    // terminator) by constructing the serialized form directly.
    let capacity: u64 = 16;
    let mut bytes = Vec::with_capacity(8 + 16 * 8);
    bytes.extend_from_slice(&capacity.to_le_bytes());
    let tmp = LossyIndex::new(16);
    for i in 0..16u64 {
        let h = splitmix_hash(i + 5000);
        let tag = tmp.tag(&h);
        let entry = IndexEntry::new(tag, 0, i);
        bytes.extend_from_slice(&entry.0.to_le_bytes());
    }
    let restored = LossyIndex::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 16);
    // Query a hash whose tag does NOT match any entry — the iterator
    // must probe all 16 slots and return None.
    let mut query = [0xFFu8; 16];
    query[8..12].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
    assert_eq!(restored.lookup(&query), None);
}
