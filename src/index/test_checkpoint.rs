use super::*;
use crate::index::IndexConfig;
use crate::index::LossyIndex;

fn hash_for(seed: u16, i: usize) -> [u8; 16] {
    let mut hash = [0u8; 16];
    hash[..2].copy_from_slice(&seed.wrapping_add(u16::try_from(i).unwrap()).to_le_bytes());
    hash
}

fn index_with_entries(seed: u16, count: usize) -> LossyIndex {
    let index = LossyIndex::new(count.max(16).saturating_mul(2));
    for i in 0..count {
        let shard = u16::try_from(i % 64).unwrap();
        let _ = index.insert(&hash_for(seed, i), shard, (i as u64).wrapping_mul(128));
    }
    index
}

#[test]
fn fingerprint_is_deterministic_but_sensitive() {
    let packs = [(3, 100), (1, 50), (2, 75)];
    assert_eq!(pack_fingerprint(&packs), pack_fingerprint(&packs));
    assert_eq!(
        pack_fingerprint(&packs),
        pack_fingerprint(&[(1, 50), (2, 75), (3, 100)])
    );
    // A length change (append/rotation) must change the fingerprint.
    assert_ne!(
        pack_fingerprint(&packs),
        pack_fingerprint(&[(1, 51), (2, 75), (3, 100)])
    );
    // A pack-id change (repack/retire) must change it too.
    assert_ne!(
        pack_fingerprint(&packs),
        pack_fingerprint(&[(1, 50), (2, 75), (4, 100)])
    );
}

#[test]
fn checkpoint_round_trips_index_slots() {
    let cases: Vec<([u8; 16], u16, usize)> =
        vec![([1u8; 16], 1, 40), ([2u8; 16], 2, 200), ([3u8; 16], 3, 5)];
    let indexes: Vec<LossyIndex> = cases
        .iter()
        .map(|(_, seed, count)| index_with_entries(*seed, *count))
        .collect();
    let blobs: Vec<([u8; 16], Vec<u8>)> = cases
        .iter()
        .zip(&indexes)
        .map(|((id, _, _), index)| (*id, index.serialize()))
        .collect();
    let dir = std::env::temp_dir().join(format!("mtxdb_checkpoint_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    let fingerprint = pack_fingerprint(&[(7, 12345)]);
    write_checkpoint(
        &path,
        fingerprint,
        0,
        0,
        &blobs
            .iter()
            .map(|(id, blob)| (*id, 0, blob.as_slice()))
            .collect::<Vec<_>>(),
        &[],
    )
    .unwrap();

    let loaded = read_checkpoint(&path).unwrap_or_else(|| {
        panic!(
            "valid checkpoint should read; file has {} bytes",
            std::fs::read(&path).map_or(0, |b| b.len())
        )
    });
    assert_eq!(loaded.fingerprint, fingerprint);
    assert_eq!(
        loaded.collections.len(),
        3,
        "checkpoint must preserve directory order and count"
    );
    let mmap = Arc::clone(&loaded.mmap);
    for ((case, loaded), (_blob_id, expected_blob)) in
        cases.iter().zip(&loaded.collections).zip(&blobs)
    {
        let (expected_id, seed, count) = case;
        assert_eq!(loaded.collection_id, *expected_id);
        let slots_len = (loaded.capacity as usize).saturating_mul(8);
        assert_eq!(
            &mmap[loaded.slots_offset..loaded.slots_offset.saturating_add(slots_len)],
            &expected_blob[8..8 + slots_len],
            "slots must round-trip verbatim"
        );

        // The mmap-backed index must find every hash the live index had.
        let loaded_index = if loaded.has_homes_tails {
            LossyIndex::from_mmap_slots_with_homes_tails(
                Arc::clone(&mmap),
                loaded.slots_offset,
                loaded.homes_offset,
                loaded.tails_offset,
                loaded.capacity,
                loaded.slot_count,
                IndexConfig::default(),
            )
        } else {
            LossyIndex::from_mmap_slots(
                Arc::clone(&mmap),
                loaded.slots_offset,
                loaded.capacity,
                loaded.slot_count,
            )
        };
        let original_index = LossyIndex::deserialize(expected_blob).unwrap();
        for i in 0..*count {
            let hash = hash_for(*seed, i);
            assert_eq!(
                loaded_index.lookup(&hash),
                original_index.lookup(&hash),
                "loaded index must agree with the source index on spot lookups"
            );
            assert!(
                loaded_index.lookup(&hash).is_some(),
                "round-tripped index lost a record"
            );
        }
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn empty_checkpoint_round_trips() {
    let dir = std::env::temp_dir().join(format!("mtxdb_checkpoint_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);
    write_checkpoint(&path, pack_fingerprint(&[]), 0, 0, &[], &[]).unwrap();
    let loaded = read_checkpoint(&path).expect("empty checkpoint is still a valid file");
    assert_eq!(loaded.fingerprint, pack_fingerprint(&[]));
    assert!(loaded.collections.is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn corrupt_checkpoints_read_as_none() {
    let dir = std::env::temp_dir().join(format!("mtxdb_checkpoint_bad_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    // Missing file.
    assert!(read_checkpoint(&path).is_none());

    // Truncated (valid header, cut slot section).
    let blobs = [([1u8; 16], index_with_entries(1, 40).serialize())];
    write_checkpoint(
        &path,
        0,
        0,
        0,
        &blobs
            .iter()
            .map(|(id, b)| (*id, 0, b.as_slice()))
            .collect::<Vec<_>>(),
        &[],
    )
    .unwrap();
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len().saturating_sub(10)]).unwrap();
    assert!(read_checkpoint(&path).is_none());

    // Wrong magic.
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[0xEE; 64]).unwrap();
    assert!(read_checkpoint(&path).is_none());

    // Wrong version byte in an otherwise valid file.
    write_checkpoint(
        &path,
        0,
        0,
        0,
        &blobs
            .iter()
            .map(|(id, b)| (*id, 0, b.as_slice()))
            .collect::<Vec<_>>(),
        &[],
    )
    .unwrap();
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, {
        let mut v = full.clone();
        v[8..12].copy_from_slice(&99u32.to_le_bytes());
        v
    })
    .unwrap();
    assert!(read_checkpoint(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn content_crc32_catches_value_preserving_slot_corruption() {
    // The pre-CRC occupancy walk only ever compared a *count* of
    // non-empty slots against `slot_count` — a corruption that flips
    // bits within an already-occupied slot (changing which shard/offset
    // it names, without zeroing it) preserves that count and would have
    // silently passed. The CRC covers the exact bytes, so this same
    // corruption must now be rejected.
    let dir = std::env::temp_dir().join(format!(
        "mtxdb_checkpoint_slot_corruption_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    let blobs = [([7u8; 16], index_with_entries(3, 40).serialize())];
    write_checkpoint(
        &path,
        0,
        0,
        0,
        &blobs
            .iter()
            .map(|(id, b)| (*id, 0, b.as_slice()))
            .collect::<Vec<_>>(),
        &[],
    )
    .unwrap();
    assert!(
        read_checkpoint(&path).is_some(),
        "sanity check: the unmodified checkpoint must read back cleanly"
    );

    let mut bytes = std::fs::read(&path).unwrap();
    // Flip one bit inside the slots section (well past the header +
    // single directory entry) in a byte that is currently non-zero, so
    // the slot stays non-empty — only its value changes.
    let slots_start = CHECKPOINT_HEADER_LEN + COLLECTION_DIR_ENTRY_LEN;
    let corrupt_at = bytes[slots_start..]
        .iter()
        .position(|&b| b != 0)
        .map(|offset| slots_start + offset)
        .expect("at least one non-zero slot byte exists");
    bytes[corrupt_at] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    assert!(
        read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_none(),
        "a value-preserving bit flip inside an occupied slot must be rejected"
    );
    assert!(
        read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::WriteOnly).is_some(),
        "write-only mode deliberately trusts a structurally valid checkpoint"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_pack_fingerprint_missing_file() {
    let dir = std::env::temp_dir().join(format!("mtxdb_ckpt_fp_missing_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);
    assert!(
        matches!(read_pack_fingerprint(&path), Ok(None)),
        "missing file must return Ok(None)"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_pack_fingerprint_truncated() {
    let dir = std::env::temp_dir().join(format!("mtxdb_ckpt_fp_trunc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);
    std::fs::write(&path, [0u8; 32]).unwrap();
    assert!(
        read_pack_fingerprint(&path).is_err(),
        "truncated header must return an error"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_pack_fingerprint_invalid_magic() {
    let dir = std::env::temp_dir().join(format!("mtxdb_ckpt_fp_badmagic_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);
    let mut buf = [0u8; CHECKPOINT_HEADER_LEN];
    buf[..8].copy_from_slice(b"BADMAGIC");
    std::fs::write(&path, buf).unwrap();
    assert!(
        read_pack_fingerprint(&path).is_err(),
        "bad magic must return an error"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_pack_fingerprint_valid_roundtrip() {
    let dir = std::env::temp_dir().join(format!("mtxdb_ckpt_fp_valid_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    let header = CheckpointHeader {
        magic: CHECKPOINT_MAGIC,
        version: CHECKPOINT_VERSION,
        collection_count: 0,
        directory_bytes: 0,
        slots_bytes: 0,
        pack_fingerprint: 0xDEAD_BEEF_CAFE_BABE,
        content_crc32: 0,
        homes_bytes: 0,
        tails_bytes: 0,
        covered_lsn: 0,
        pack_table_count: 0,
        pack_table_bytes: 0,
        base_delta_seq: 0,
    };
    std::fs::write(&path, header.encode()).unwrap();

    let fp = read_pack_fingerprint(&path).expect("valid checkpoint");
    assert_eq!(fp, Some(0xDEAD_BEEF_CAFE_BABE));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_durable_fingerprint_no_checkpoint() {
    let dir = std::env::temp_dir().join(format!("mtxdb_durable_no_ckpt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    assert!(
        matches!(read_durable_fingerprint(&dir), Ok(None)),
        "missing checkpoint must return Ok(None)"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_durable_fingerprint_checkpoint_only() {
    let dir = std::env::temp_dir().join(format!("mtxdb_durable_ckpt_only_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    let header = CheckpointHeader {
        magic: CHECKPOINT_MAGIC,
        version: CHECKPOINT_VERSION,
        collection_count: 0,
        directory_bytes: 0,
        slots_bytes: 0,
        pack_fingerprint: 0x42,
        content_crc32: 0,
        homes_bytes: 0,
        tails_bytes: 0,
        covered_lsn: 0,
        pack_table_count: 0,
        pack_table_bytes: 0,
        base_delta_seq: 0,
    };
    std::fs::write(&path, header.encode()).unwrap();

    let fp = read_durable_fingerprint(&dir).expect("read should succeed");
    assert_eq!(
        fp.map(|durable| durable.fingerprint),
        Some(0x42),
        "without a delta, durable = checkpoint fp"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_pack_fingerprint_malformed_returns_err() {
    let dir = std::env::temp_dir().join(format!("mtxdb_malformed_ckpt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);

    // Write a corrupted checkpoint (bad magic).
    std::fs::write(&path, vec![0xFF; CHECKPOINT_HEADER_LEN]).unwrap();

    assert!(
        read_pack_fingerprint(&path).is_err(),
        "malformed checkpoint must return Err"
    );
}

/// The base sequence a checkpoint incorporates survives the header round trip
/// and is visible without reading the body.
#[test]
fn base_delta_seq_round_trips_through_the_header() {
    let dir = std::env::temp_dir().join(format!("mtxdb-base-seq-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_CHECKPOINT_FILE);
    let blobs = [([3u8; 16], index_with_entries(1, 20).serialize())];
    write_checkpoint(
        &path,
        pack_fingerprint(&[(1, 10)]),
        5,
        0xDEAD_BEEF_0042,
        &blobs
            .iter()
            .map(|(id, b)| (*id, 1, b.as_slice()))
            .collect::<Vec<_>>(),
        &[(0, 1)],
    )
    .unwrap();
    assert_eq!(
        read_checkpoint(&path).unwrap().base_delta_seq,
        0xDEAD_BEEF_0042
    );
    let summary = read_checkpoint_summary(&path).unwrap().unwrap();
    assert_eq!(summary.base_delta_seq, 0xDEAD_BEEF_0042);
    assert_eq!(summary.covered_lsn, 5);
    std::fs::remove_dir_all(&dir).unwrap();
}
