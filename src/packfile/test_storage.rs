use super::*;
use std::time::Duration;

const TEST_COLLECTION: [u8; 16] = [0x01; 16];
const SECOND_COLLECTION: [u8; 16] = [0x02; 16];

/// A node id distinct across both the bucket bytes (0..8) and tag bytes
/// (8..12) the lossy index actually reads — an id that only varies byte
/// 0 collapses every entry onto the same 24-bit tag, which is not a
/// realistic content hash and defeats the index in ways unrelated to
/// whatever a test using it is meant to check.
fn distinct_id(byte: u8) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[0] = byte;
    id[9] = byte.wrapping_mul(37).wrapping_add(11);
    id[15] = byte.wrapping_mul(7);
    id
}
const OTHER_COLLECTION: [u8; 16] = [0x02; 16];

fn test_dir(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mdb_test_pfs_{name}_{}_{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_sees_a_committed_but_unflushed_group() {
    let dir = test_dir("read_committed_unflushed");
    let wal = dir.join("wal.bin");
    let collection = [0x42u8; 16];
    let node = [0x07u8; 16];

    // A read-only open needs at least one shard on disk, so seed a
    // durable record first.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x99u8; 16],
        &[0x99u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    // A complete group (trailer present) with no fsync: committed, but
    // invisible to the durable fingerprint gate.
    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"committed-unflushed".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
    assert_eq!(
        read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"committed-unflushed"[..])
    );

    // The durable API still hides the unflushed group.
    let durable = store.get_many_with_refresh(&collection, &[node]).unwrap();
    assert!(
        durable[0].is_none(),
        "durable read must not observe an unflushed journal group"
    );
}

/// Journal state for a writer-restart scenario: LSN 1 is durable and LSN 2
/// is visible but never reached the disk. Returns the segment length that
/// covers only LSN 1, so a test can cut the file back to what a crash would
/// have kept.
#[cfg(feature = "multi-reader")]
fn journal_with_a_visible_but_undurable_group(
    wal: &std::path::Path,
    collection: [u8; 16],
    stable: [u8; 16],
    reused: [u8; 16],
) -> u64 {
    let (mut journal, _) = Journal::open(wal).unwrap();
    journal
        .commit_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: stable,
            payload: b"stable".to_vec(),
        }])
        .unwrap();
    let durable_len = std::fs::metadata(wal).unwrap().len();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: reused,
            payload: b"stale-".to_vec(),
        }])
        .unwrap();
    durable_len
}

/// A read-only store with a live reader journal that has already applied a
/// visible-but-undurable group (`stale-` at the `reused` id, collection
/// `0x42`), for the restarted-writer tests. Returns the
/// journal path, the store and the journal's durable length.
#[cfg(feature = "multi-reader")]
fn reader_over_an_undurable_group(name: &str) -> (PathBuf, PackfileStorage, u64) {
    let dir = test_dir(name);
    let wal = dir.join("wal.bin");
    let collection = [0x42u8; 16];
    let stable = [0x01u8; 16];
    let reused = [0x02u8; 16];
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x99u8; 16],
        &[0x99u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);
    let durable_len = journal_with_a_visible_but_undurable_group(&wal, collection, stable, reused);
    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    let before_restart = store.get_read_committed(&collection, &[reused]).unwrap();
    assert_eq!(
        before_restart[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"stale-"[..])
    );
    (wal, store, durable_len)
}

/// A writer that crashes loses its visible-but-undurable tail, and the
/// restarted writer reuses those LSNs. A live reader that already applied
/// the lost group must not keep serving it. The reissued group has the same
/// length, so the file is exactly as long as when the reader last scanned.
#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_drops_a_group_the_restarted_writer_discarded() {
    let (wal, store, durable_len) =
        reader_over_an_undurable_group("read_committed_writer_restart_same_len");
    let collection = [0x42u8; 16];
    let reused = [0x02u8; 16];

    // The crash: LSN 2 never reached the disk. The restarted writer
    // recovers LSN 1 and publishes a different LSN 2 of the same length.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .set_len(durable_len)
        .unwrap();
    let (mut journal, scan) = Journal::open(&wal).unwrap();
    assert_eq!(scan.groups.len(), 1);
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: reused,
            payload: b"fresh-".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let after = store.get_read_committed(&collection, &[reused]).unwrap();
    assert_eq!(
        after[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"fresh-"[..]),
        "the reader must not keep the group the restarted writer discarded"
    );
}

/// Same restart, but the reissued LSN 2 is followed by an LSN 3, so the file
/// has grown and the reader takes its incremental-scan path.
#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_drops_a_discarded_group_when_the_file_grew() {
    let (wal, store, durable_len) =
        reader_over_an_undurable_group("read_committed_writer_restart_grew");
    let collection = [0x42u8; 16];
    let reused = [0x02u8; 16];
    let later = [0x03u8; 16];

    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .set_len(durable_len)
        .unwrap();
    let (mut journal, _) = Journal::open(&wal).unwrap();
    for (node, payload) in [(reused, &b"fresh-"[..]), (later, &b"after-"[..])] {
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: payload.to_vec(),
            }])
            .unwrap();
    }
    drop(journal);

    let after = store
        .get_read_committed(&collection, &[reused, later])
        .unwrap();
    assert_eq!(
        after[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"fresh-"[..]),
        "the reissued LSN must replace the discarded group's content"
    );
    assert_eq!(
        after[1].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"after-"[..])
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_shadows_durable_with_a_committed_delete() {
    let dir = test_dir("read_committed_delete");
    let wal = dir.join("wal.bin");
    let collection = [0x43u8; 16];
    let node = [0x08u8; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"live")),
        )
        .unwrap();
    writer.sync().unwrap();
    drop(writer);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::DeleteCollection {
            collection_id: collection,
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
    assert!(
        read_committed[0].is_none(),
        "a committed delete must shadow the durable record"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_preserves_a_delete_boundary_across_recreate() {
    let dir = test_dir("read_committed_recreate");
    let wal = dir.join("wal.bin");
    let collection = [0x45u8; 16];
    let old = [0x0au8; 16];
    let new = [0x0bu8; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer
        .put(
            &collection,
            &old,
            &NodeData::new(bytes::Bytes::from_static(b"old")),
        )
        .unwrap();
    writer.sync().unwrap();
    drop(writer);

    // Delete the collection, then recreate it with a different record. The
    // pre-delete record must not be resurrected by the durable fallback.
    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::DeleteCollection {
            collection_id: collection,
        }])
        .unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: new,
            payload: b"new".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    let read_committed = store.get_read_committed(&collection, &[old, new]).unwrap();
    assert!(
        read_committed[0].is_none(),
        "a pre-delete record must stay deleted after the collection is recreated"
    );
    assert_eq!(
        read_committed[1].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"new"[..])
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_ignores_entries_the_checkpoint_covers() {
    let dir = test_dir("read_committed_covered");
    let wal = dir.join("wal.bin");
    let collection = [0x44u8; 16];
    let node = [0x09u8; 16];

    // A real writer with a journal: its sync checkpoints the put and embeds
    // the covered LSN atomically, so the node is both durable and covered
    // rather than the test merely claiming coverage the index never had.
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"covered")),
        )
        .unwrap();
    writer.sync().unwrap();
    drop(writer);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    {
        let overlay = store.read_journal.lock();
        let overlay = overlay.as_ref().expect("journal overlay enabled");
        assert!(
            overlay.puts.is_empty(),
            "covered puts must not stay in the overlay"
        );
        assert!(
            overlay.delete_lsn.is_empty(),
            "covered deletes must not stay in the overlay"
        );
    }
    let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
    assert_eq!(
        read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"covered"[..]),
        "an entry the checkpoint covers must be served from the durable index"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_keeps_entries_a_newer_checkpoint_covers() {
    let dir = test_dir("read_committed_stale_index");
    let wal = dir.join("wal.bin");
    let collection = [0x47u8; 16];
    let node = [0x0eu8; 16];

    // Durable value V0, loaded by the reader's index at open.
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"v0")),
        )
        .unwrap();
    writer.sync().unwrap();
    drop(writer);

    // V1 committed to the journal but not checkpointed into the reader's
    // already-loaded index.
    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"v1".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    assert_eq!(
        store.get_read_committed(&collection, &[node]).unwrap()[0]
            .as_ref()
            .map(|data| data.bytes.as_ref()),
        Some(&b"v1"[..])
    );

    // A concurrent checkpoint advances coverage past V1's LSN, but this
    // handle's in-memory index still holds only V0. Pruning against that
    // detached coverage would drop V1 from the overlay and resurrect the
    // stale V0 from the index.
    write_journal_lsn_file(&store.base_dir, &store.durable_coverage, 1).unwrap();
    assert_eq!(
        store.get_read_committed(&collection, &[node]).unwrap()[0]
            .as_ref()
            .map(|data| data.bytes.as_ref()),
        Some(&b"v1"[..]),
        "a checkpoint this handle has not loaded must not evict overlay entries"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_fails_closed_when_reclaim_skips_its_covered_lsn() {
    let dir = test_dir("read_committed_reclaim_gap");
    let wal = dir.join("wal.bin");
    let collection = [0x48u8; 16];
    let node = [0x0fu8; 16];

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x95u8; 16],
        &[0x95u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"first".to_vec(),
        }])
        .unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: [0x10u8; 16],
            payload: b"second".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    // Reclaim drops LSN 1, so the segment now begins at LSN 2 while the
    // reader's index only incorporated coverage 0. Those records are in
    // neither source, so the read must fail closed rather than serve a gap.
    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal.reclaim_through(1).unwrap();
    drop(journal);

    assert!(
        store.get_read_committed(&collection, &[node]).is_err(),
        "a reclaimed segment base beyond the reader's covered LSN must error"
    );
}

/// A genuine coverage gap that reloads cannot close — the checkpoint
/// exists and matches, but never covers the reclaimed LSNs — is a
/// persistent corruption, not a transient condition.
#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_gap_with_matching_checkpoint_is_corrupt() {
    let dir = test_dir("read_committed_gap_corrupt");
    let wal = dir.join("wal.bin");
    let collection = [0x4Cu8; 16];
    let node = [0x0fu8; 16];

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x96u8; 16],
        &[0x96u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"first".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal.reclaim_through(1).unwrap();
    drop(journal);

    let error = store
        .get_read_committed(&collection, &[node])
        .expect_err("the unrecoverable gap must fail");
    assert!(
        matches!(error, StorageError::Corrupt(_)),
        "a matching checkpoint that never covers the gap is corrupt, got {error:?}"
    );
}

/// The same gap, but with no usable checkpoint to reload: every reload
/// attempt fails, so the read is retryable once the writer publishes a
/// checkpoint instead of being reported as permanent corruption.
#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_gap_without_checkpoint_is_retryable() {
    let dir = test_dir("read_committed_gap_retry");
    let wal = dir.join("wal.bin");
    let collection = [0x4Du8; 16];
    let node = [0x0fu8; 16];

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x97u8; 16],
        &[0x97u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"first".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    // Remove the checkpoint the reader would reload, so the reload path
    // fails closed but retryably.
    fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal.reclaim_through(1).unwrap();
    drop(journal);

    let error = store
        .get_read_committed(&collection, &[node])
        .expect_err("the unrecoverable gap must fail");
    assert!(
        error.is_would_block(),
        "a missing checkpoint must be reported as retryable, got {error:?}"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_reloads_after_reclaim_advances_coverage() {
    let dir = test_dir("read_committed_reclaim_reload");
    let wal = dir.join("wal.bin");
    let collection = [0x49u8; 16];
    let first = [0x11u8; 16];
    let second = [0x12u8; 16];

    // Durable seed, then an empty journal segment the reader can attach to.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x94u8; 16],
        &[0x94u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);
    let (journal, _) = Journal::open(&wal).unwrap();
    drop(journal);

    // Reader attaches with the pre-advance index (coverage 0).
    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    // Remove the seed checkpoint so the writer's sync must write a full
    // checkpoint instead of appending to its existing delta log. A delta
    // append extends the loaded index but does not advance checkpoint
    // coverage; this test specifically needs a checkpoint that covers LSN 1.
    fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();

    // A real writer checkpoints `first` and reclaims through its LSN, so the
    // checkpoint embeds coverage 1 and its index contains `first`.
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &first,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();

    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    assert_eq!(checkpoint.covered_lsn, 1);
    let after_checkpoint = Journal::scan_read_only(&wal).unwrap();
    assert!(after_checkpoint.base_lsn > checkpoint.covered_lsn);

    // Append a new record after the checkpoint, then sync the writer. The
    // sync commits its journal group and extends the checkpoint's delta
    // log without rewriting the checkpoint. Reload must accept that delta
    // tail despite the live pack having grown.
    writer
        .put(
            &collection,
            &second,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    writer.sync().unwrap();

    let read_committed = store
        .get_read_committed(&collection, &[first, second])
        .unwrap();
    assert_eq!(store.read_covered_lsn.load(Ordering::Acquire), 1);
    assert!(
        store.stats().read_reloads >= 1,
        "the reclaim must have driven at least one checkpoint-bound reload"
    );
    assert_eq!(
        read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"first"[..]),
        "the reloaded checkpoint index must serve the covered record, not a hole"
    );
    assert_eq!(
        read_committed[1].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"second"[..]),
        "the post-checkpoint group must be served from the overlay"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_reload_skips_the_exact_pack_gate() {
    let dir = test_dir("read_committed_reload_skip_gate");
    let wal = dir.join("wal.bin");
    let collection = [0x4Bu8; 16];
    let first = [0x31u8; 16];
    let second = [0x32u8; 16];
    let third = [0x33u8; 16];

    // Durable seed, then an empty segment the reader can attach to.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x97u8; 16],
        &[0x97u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);
    let (journal, _) = Journal::open(&wal).unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    // Force the writer's next sync to write a full checkpoint so coverage
    // advances and the segment is reclaimed through it.
    fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &first,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();
    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    assert_eq!(checkpoint.covered_lsn, 1);

    // A synced second record extends the delta log.
    writer
        .put(
            &collection,
            &second,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    writer.sync().unwrap();

    // A third record is appended (eagerly) but never synced, so the live
    // packs grow past the delta log's tail. An exact-pack gate would reject
    // the checkpoint here and fail the read closed; the read-journal reload
    // must skip that gate and let the overlay supply the suffix.
    writer
        .put(
            &collection,
            &third,
            &NodeData::new(bytes::Bytes::from_static(b"third")),
        )
        .unwrap();

    // The relaxation is scoped to the read-journal reload: a normal open
    // runs `ReloadMode::Strict` and must still reject the checkpoint the
    // eager append has outgrown, falling back to a full rescan rather than
    // trusting a stale fingerprint.
    let strict_reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    assert_eq!(
        strict_reader.open_timings().unwrap().path,
        OpenPath::FullScan,
        "a strict open must not accept a checkpoint the live packs have outgrown"
    );

    let read_committed = store
        .get_read_committed(&collection, &[first, second, third])
        .unwrap();
    assert!(
        store.stats().read_reloads >= 1,
        "the reclaim must have driven the read-journal reload path"
    );
    for (value, expected) in read_committed
        .iter()
        .zip([&b"first"[..], b"second", b"third"])
    {
        assert_eq!(
            value.as_ref().map(|data| data.bytes.as_ref()),
            Some(expected),
            "every committed record must be visible after a gated reload"
        );
    }
}

#[cfg(feature = "multi-reader")]
#[test]
fn repack_persists_checkpoint_so_read_journal_reload_survives_retirement() {
    // Repack moves a collection's records into a fresh destination shard
    // and retires the old one once nothing else references it. A reader
    // reloading from the checkpoint afterward must see the new shard
    // layout — otherwise it reloads a checkpoint whose index still names
    // a shard repack already unlinked, and the read fails closed.
    let dir = test_dir("repack_persists_checkpoint");
    let wal = dir.join("wal.bin");
    let collection = [0x51u8; 16];
    let node = [0x61u8; 16];

    // `collection` is the only collection in this store, so its shard is
    // never shared with anything else: repack's retirement of the
    // now-empty source shard actually unlinks the file, rather than
    // leaving it alive because another collection still references it.
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();
    let before =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    assert_eq!(before.covered_lsn, 1);

    // Repack `collection`: it moves `node` into a fresh non-source shard
    // and, since no other collection references the old one, retires
    // (unlinks) it.
    writer
        .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
        .unwrap();

    let after =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    assert_eq!(
        after.covered_lsn, before.covered_lsn,
        "repack must carry the checkpoint's covered_lsn forward unchanged, not reset it"
    );

    // Open a reader only now, after retirement already unlinked the
    // source shard: it must never hold an open handle to the retired
    // file, or the read would trivially keep succeeding through the
    // stale-but-still-open fd (Linux keeps an unlinked file's bytes
    // readable through any fd opened before the unlink) and the test
    // would not actually exercise the checkpoint staleness at all.
    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    // Drive the checkpoint reload directly rather than through
    // get_read_committed: the journal overlay can serve a still-covered
    // record straight from its segment without ever touching the index,
    // which would leave this test unable to distinguish a correct reload
    // from a mismatched one. reload_index_from_checkpoint is what
    // discovers the live shard set and must succeed against the
    // checkpoint repack just wrote.
    assert!(
        store.reload_index_from_checkpoint(),
        "reload must succeed against the checkpoint repack persisted, not a stale one \
         naming a shard repack already unlinked"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn repack_persisted_checkpoint_reload_resolves_record_by_post_repack_offset() {
    let dir = test_dir("repack_persists_checkpoint_read");
    let wal = dir.join("wal.bin");
    let collection = [0x51u8; 16];
    let node = [0x61u8; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();
    writer
        .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
        .unwrap();

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    assert!(store.reload_index_from_checkpoint());

    let got = store.get(&collection, &node).unwrap();
    assert_eq!(
        got.map(|data| data.bytes.to_vec()),
        Some(b"first".to_vec()),
        "the reloaded index must resolve the record through its post-repack shard offset"
    );
}

/// Coverage for a post-repack delta epoch over *existing* pack identities:
/// a delta written after repack's fresh checkpoint must replay for a cold
/// reader and resolve through the checkpoint's v6 pack table, not the
/// reader's own shard numbering.
///
/// `repack_collection_reachable` persists a checkpoint (C1) naming the
/// post-retirement pack set, so the pre-retirement delta epoch is retired
/// with it. A later `sync_all` appends a fresh epoch continuing C1; this
/// test pins that a cold open *replays* that epoch (nonzero `delta_replay`,
/// not a checkpoint rewrite that already included `third`) and returns all
/// three records.
///
/// `third` must be durably synced, not merely shard-flushed: only a sync
/// persists the index. An uncommitted record is invisible to checkpoint
/// replay and recoverable only by a fallback full scan — a different
/// property, and the one the previous version of this test accidentally
/// measured.
///
/// Out of scope: a delta frame referencing a pack created *after* C1 (a
/// pack absent from C1's pack table), and any claim that a delta-side
/// `SlotBinding` is unnecessary.
#[cfg(feature = "multi-reader")]
#[test]
fn post_repack_delta_epoch_replays_for_a_cold_reader() {
    let dir = test_dir("delta_across_retirement");
    let wal = dir.join("wal.bin");
    let collection = [0x71u8; 16];
    let first = [0x81u8; 16];
    let second = [0x82u8; 16];
    let third = [0x83u8; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();

    // Commit enough that a checkpoint lands and rotates the delta epoch.
    writer
        .put(
            &collection,
            &first,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();

    // A post-checkpoint write leaves a delta epoch whose base is the
    // current checkpoint and whose tail names the current pack set.
    writer
        .put(
            &collection,
            &second,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    writer.sync_all().unwrap();

    // Repack moves the collection's records into a fresh destination shard
    // and unlinks the source. The writer's slot table now has a hole, so a
    // fresh reader's `discover_shards` will not reproduce the writer's
    // numbering. Repack persists a checkpoint for the new layout.
    writer
        .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
        .unwrap();

    // Commit `third` into the delta epoch continuing the post-repack
    // checkpoint. A shard flush alone would not persist the index.
    writer
        .put(
            &collection,
            &third,
            &NodeData::new(bytes::Bytes::from_static(b"third")),
        )
        .unwrap();
    writer.sync_all().unwrap();
    drop(writer);

    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    reader.enable_read_journal(&wal).unwrap();

    let timings = reader
        .stats()
        .last_open_timings
        .expect("open must record timings");
    // The post-repack checkpoint and its delta epoch must both validate, so
    // the cold open takes the checkpoint fast path rather than a full scan.
    assert_eq!(
        timings.path,
        OpenPath::Checkpoint,
        "the post-repack checkpoint and the epoch continuing it must be usable"
    );
    // Prove the records below come from a gated delta replay, not from a
    // checkpoint rewrite that already included `third`. This is a
    // deterministic count, unlike the `delta_replay` duration.
    assert!(
        timings.delta_replay_operations > 0,
        "cold open must have replayed the post-repack delta epoch: {timings:?}"
    );
    assert_eq!(
        timings.full_scan,
        Duration::ZERO,
        "cold open must not have fallen back to a full scan"
    );

    // All three records must resolve to their exact bytes through the
    // replayed delta — never to a wrong shard's bytes.
    for (id, expected) in [
        (first, &b"first"[..]),
        (second, &b"second"[..]),
        (third, &b"third"[..]),
    ] {
        let data = reader
            .get(&collection, &id)
            .unwrap()
            .expect("record must exist");
        assert_eq!(data.bytes.as_ref(), expected);
    }
}

#[test]
fn repack_persists_checkpoint_so_fresh_cold_open_survives_retirement() {
    // Same failure shape as the read-journal test above, but through the
    // *plain* open() path and no journal at all. After repack, the
    // checkpoint's pack set is exactly the live pack set (one pack), so
    // a fresh open's fingerprint matches exactly and the exact-pack gate
    // takes no action (replay_needed is false) — the checkpoint's raw
    // index slots are trusted directly. A brand-new reader's own
    // discover_shards, seeing only that one surviving pack from an
    // empty slot table, assigns it local slot 0; the checkpoint's index
    // still names the writer's slot 1 (assigned before the original
    // slot-0 shard was retired). Without the pack-table remap, get()
    // resolves against the wrong (nonexistent, for this reader) slot.
    let dir = test_dir("repack_persists_checkpoint_cold_open");
    let collection = [0x52u8; 16];
    let node = [0x62u8; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer
        .put(
            &collection,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    writer.sync().unwrap();

    writer
        .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
        .unwrap();
    drop(writer);

    // Fresh process-equivalent open: no history, no shard ever open
    // before this point, so this reader's own discover_shards renumbers
    // from scratch.
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    let got = reopened.get(&collection, &node).unwrap();
    assert_eq!(
        got.map(|data| data.bytes.to_vec()),
        Some(b"first".to_vec()),
        "a fresh cold open after repack must resolve the record through \
         the checkpoint's pack table, not the writer's raw (unstable) slot number"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn open_read_committed_serves_the_overlay_in_one_call() {
    let dir = test_dir("open_read_committed");
    let wal = dir.join("wal.bin");
    let collection = [0x4Au8; 16];
    let node = [0x21u8; 16];

    // A shard must exist before a read-only open will succeed.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x96u8; 16],
        &[0x96u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    // A committed group in the writer's segment that was never synced into
    // the packs: the durable API cannot see it, only the overlay can.
    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: node,
            payload: b"committed".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let plain = PackfileStorage::open_read_only(dir.clone()).unwrap();
    assert!(
        plain.get(&collection, &node).unwrap().is_none(),
        "a plain read-only handle must not see the unsynced committed group"
    );
    drop(plain);

    let store = PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap();
    assert_eq!(
        store.get_read_committed(&collection, &[node]).unwrap()[0]
            .as_ref()
            .map(|data| data.bytes.as_ref()),
        Some(&b"committed"[..]),
        "the one-call handle must serve the committed group from the overlay"
    );
}

#[test]
fn force_index_checkpoint_rewrites_with_and_without_pending_delta() {
    let dir = test_dir("force_index_checkpoint");
    let wal = dir.join("wal.bin");
    let checkpoint_path = PackfileStorage::index_checkpoint_path(&dir);
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();

    // Empty delta log: no mutations yet. A normal sync would be a no-op;
    // forcing still writes a checkpoint with no committed coverage.
    store.force_index_checkpoint().unwrap();
    let empty = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();

    // Non-empty delta: the put commits a journal group, and sync_all takes
    // the delta fast path, which does not advance checkpoint coverage.
    store
        .put(
            &TEST_COLLECTION,
            &[0x11; 16],
            &NodeData::new(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();
    store.sync_all().unwrap();
    let after_sync = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();
    assert_eq!(
        after_sync.covered_lsn, empty.covered_lsn,
        "a delta append must not advance checkpoint coverage"
    );

    // Forcing must take the full-rewrite path and record the committed
    // journal tail. Read the tail from the journal rather than hardcoding
    // an LSN, so this survives any change to how the first mutation is
    // numbered.
    let committed = store.journal().expect("journal enabled").committed_lsn();
    assert!(
        committed > after_sync.covered_lsn,
        "the put must have committed past the loaded coverage"
    );
    store.force_index_checkpoint().unwrap();
    let forced = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();
    assert_eq!(
        forced.covered_lsn, committed,
        "force must advance coverage to the committed journal tail"
    );
}

#[test]
fn detached_force_checkpoint_counts_like_the_synchronous_fallback() {
    let dir = test_dir("detached_force_checkpoint_metric");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(dir.join("wal.bin")).unwrap();

    // Background off: the detached call falls back to the synchronous path,
    // which counts the write it performed.
    store
        .put(
            &TEST_COLLECTION,
            &[0x22; 16],
            &NodeData::new(bytes::Bytes::from_static(b"y")),
        )
        .unwrap();
    let before = store.stats().checkpoint_writes;
    store.force_index_checkpoint_detached().unwrap();
    assert_eq!(store.stats().checkpoint_writes - before, 1);

    // Background on: the tail runs detached, but the write must still be
    // counted so the reported total does not depend on that setting.
    store
        .put(
            &TEST_COLLECTION,
            &[0x23; 16],
            &NodeData::new(bytes::Bytes::from_static(b"z")),
        )
        .unwrap();
    store.set_background_checkpoint(true);
    let before = store.stats().checkpoint_writes;
    store.force_index_checkpoint_detached().unwrap();
    store.wait_for_checkpoint();
    assert_eq!(
        store.stats().checkpoint_writes - before,
        1,
        "a detached forced checkpoint must count like the synchronous one"
    );
}

#[test]
fn persist_index_checkpoint_reports_whether_it_wrote() {
    let dir = test_dir("persist_checkpoint_reports_write");
    let store = PackfileStorage::open(dir).unwrap();

    // A fresh store opens structurally invalidated, so the first checkpoint
    // writes; a second with nothing dirty must report that no write
    // happened (a concurrent sync may have consumed the dirty flag first).
    assert!(store.persist_index_checkpoint().unwrap());
    assert!(!store.persist_index_checkpoint().unwrap());

    store
        .put(
            &TEST_COLLECTION,
            &[0x24; 16],
            &NodeData::new(bytes::Bytes::from_static(b"w")),
        )
        .unwrap();
    assert!(store.persist_index_checkpoint().unwrap());
    assert!(!store.persist_index_checkpoint().unwrap());
}

#[cfg(feature = "multi-reader")]
#[test]
fn reset_has_coverage_gap_is_a_pure_base_jump() {
    // A fully reclaimed segment carries no groups, only a moved base LSN.
    // Keying the gap check off the first group would miss it entirely.
    let header_only = crate::journal::Scan {
        groups: Vec::new(),
        valid_len: 0,
        truncated_tail: false,
        base_lsn: 3,
        consumed_tail: Vec::new(),
        group_ends: Vec::new(),
    };
    let per_pool = |covered: u64| ReadJournal::empty(std::path::PathBuf::new(), covered, None);
    let shared = |covered: u64| {
        ReadJournal::empty(
            std::path::PathBuf::new(),
            covered,
            Some(crate::layout::ShardType::State),
        )
    };
    // The base jumped from covered + 1 = 2 to 3, so the reclaimed prefix may
    // hold a frame this reader lacks. Both layouts must ask for a reload;
    // whether the jump is a real gap for this pool is decided after the
    // reload, not by inspecting the surviving groups. A previous check that
    // looked for this pool's frames in those groups both missed reclaimed
    // frames (silent staleness) and rejected valid frames when another pool
    // opened the gap.
    assert!(per_pool(1).reset_has_coverage_gap(&header_only));
    assert!(shared(1).reset_has_coverage_gap(&header_only));
    // covered + 1 reaches the base, so nothing at or below covered moved.
    assert!(!per_pool(2).reset_has_coverage_gap(&header_only));
    assert!(!shared(2).reset_has_coverage_gap(&header_only));
    assert!(
        !per_pool(0).reset_has_coverage_gap(&crate::journal::Scan::empty()),
        "a missing/short segment reports base 0 and is not a gap"
    );
}

#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_overlay_picks_up_groups_appended_after_the_first_scan() {
    let dir = test_dir("read_committed_tail");
    let wal = dir.join("wal.bin");
    let collection = [0x46u8; 16];
    let first = [0x0cu8; 16];
    let second = [0x0du8; 16];

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x97u8; 16],
        &[0x97u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: first,
            payload: b"first".to_vec(),
        }])
        .unwrap();

    // Open the overlay now, so its first scan stops at the first group.
    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();
    assert!(store.get_read_committed(&collection, &[first]).unwrap()[0].is_some());

    // A later append must be picked up by the incremental tail scan.
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: second,
            payload: b"second".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let read_committed = store.get_read_committed(&collection, &[second]).unwrap();
    assert_eq!(
        read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"second"[..])
    );
}

/// The publish-signal gate: a worker whose writer published nothing since
/// the last refresh must skip the `fs::metadata` refresh entirely, and must
/// still observe the next published group. This is the zero-staleness fast
/// path that replaces the per-call stat.
#[cfg(feature = "multi-reader")]
#[test]
fn read_committed_publish_signal_skips_the_stat_until_a_group_is_published() {
    let dir = test_dir("read_committed_publish_signal");
    let wal = dir.join("wal.bin");
    let collection = [0x51u8; 16];
    let first = [0x11u8; 16];
    let second = [0x12u8; 16];

    // Seed a durable shard so the read-only handle can open.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x98u8; 16],
        &[0x98u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    // The writer opens the journal, which creates the publish signal, then
    // publishes one group through the coordinator (the real write path).
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &first,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();

    let store = PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap();
    assert!(store.read_journal_publish_signal_active());
    assert!(store.get_read_committed(&collection, &[first]).unwrap()[0].is_some());
    let baseline = store.read_journal_stat_checks();

    // No publish since the last refresh: the gate must serve the overlay
    // without another stat.
    assert!(store.get_read_committed(&collection, &[first]).unwrap()[0].is_some());
    assert_eq!(
        store.read_journal_stat_checks(),
        baseline,
        "an unchanged publish generation must skip the stat"
    );

    // A new group advances the generation, so the next read must rescan and
    // observe it.
    writer
        .put(
            &collection,
            &second,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    assert!(store.get_read_committed(&collection, &[second]).unwrap()[0].is_some());
    assert!(store.stats().read_refresh_bytes > 0);
    assert_eq!(
        store.read_journal_stat_checks(),
        baseline + 1,
        "a published group must force exactly one stat refresh"
    );
}

fn ten_record_fixture() -> Vec<(NodeId, NodeData)> {
    (0..10u8)
        .map(|i| {
            let mut id = [0u8; 16];
            id[0] = i;
            id[8..12].copy_from_slice(&u32::from(i).saturating_add(1).to_le_bytes());
            (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
        })
        .collect()
}

#[test]
fn test_put_and_get() {
    let dir = test_dir("putget");
    let store = PackfileStorage::open(dir).unwrap();

    let id = [0x42u8; 16];
    let data = NodeData::new(bytes::Bytes::from_static(b"hello world"));

    store.put(&TEST_COLLECTION, &id, &data).unwrap();
    let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
    assert_eq!(got.bytes, data.bytes);
}

#[test]
fn reopened_index_keeps_distinct_same_tag_entries() {
    let dir = test_dir("reopen_same_tag");
    let mut first = [0u8; 16];
    first[8] = 0x42;
    first[15] = 1;
    let mut second = first;
    second[15] = 2;

    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &first,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    store.sync_all().unwrap();
    drop(store);

    // The checkpoint has only slots, no retained full identities. The
    // second write must hydrate the same-tag candidate and continue its
    // probe, rather than replacing the first record's only location.
    let reopened = PackfileStorage::open(dir).unwrap();
    reopened
        .put(
            &TEST_COLLECTION,
            &second,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    assert!(reopened.get(&TEST_COLLECTION, &first).unwrap().is_some());
    assert!(reopened.get(&TEST_COLLECTION, &second).unwrap().is_some());
}

#[test]
fn checkpoint_index_growth_recovers_hashes_from_indexed_frames() {
    // A batch large enough to be a "real" collection. The checkpoint
    // this reopen loads stores locations but not home hashes, so
    // growing it (via the `put_many` call below) used to rescan every
    // pack for this collection instead of recovering hashes from the
    // indexed delta frames.
    const ENTRIES: usize = 3_072;
    let dir = test_dir("checkpoint_growth_recovery");
    let store = PackfileStorage::open(dir.clone())
        .unwrap()
        .with_append_policy(crate::shard::AppendPolicy::buffered());
    let entries: Vec<_> = (0..ENTRIES)
        .map(|i| {
            let mut id = [0u8; 16];
            let mixed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            id[..8].copy_from_slice(&mixed.to_be_bytes());
            id[8] = u8::try_from((i >> 16) & 0xFF).unwrap();
            id[9] = u8::try_from((i >> 8) & 0xFF).unwrap();
            id[10] = u8::try_from(i & 0xFF).unwrap();
            (id, NodeData::new(bytes::Bytes::from_static(b"payload")))
        })
        .collect();
    store.put_many(&TEST_COLLECTION, &entries).unwrap();
    store.sync_all().unwrap();
    drop(store);

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    let checkpoint_index = reopened
        .generation(&TEST_COLLECTION)
        .expect("checkpoint collection exists");
    assert!(checkpoint_index.index.is_mmap_backed());
    let grown = reopened
        .grow_checkpoint_index(&TEST_COLLECTION, &checkpoint_index.index)
        .unwrap()
        .expect("4K checkpoint index can grow");
    assert_eq!(grown.len(), ENTRIES);
    for (id, _) in &entries {
        assert!(grown.lookup(id).is_some(), "recovered hash remains indexed");
    }
    drop(checkpoint_index);

    // Exercise the actual put_many fallback too. `insert_index` fails
    // closed once `len >= capacity * 3 / 4`, and the reopened generation
    // is mmap-backed (so `LossyIndex::grow` — which needs the original
    // homes — can't apply): crossing that threshold therefore always
    // routes through `grow_checkpoint_index`, never a silent plain grow
    // or a full pack rescan. Compute the threshold from the checkpoint's
    // actual on-disk capacity and add enough unique post-reopen entries
    // to be certain we cross it, instead of relying on one extra record
    // that may land comfortably under it.
    let checkpoint_capacity = u64::from(
        reopened
            .generation(&TEST_COLLECTION)
            .expect("checkpoint collection exists")
            .index
            .capacity(),
    );
    let threshold = checkpoint_capacity * 3 / 4;
    // Add enough unique entries to certainly cross the threshold,
    // regardless of exactly how much headroom the checkpoint's
    // persisted capacity happens to have.
    let needed = threshold.saturating_sub(u64::try_from(ENTRIES).unwrap_or(0)) + 64;
    let extra_entries: Vec<_> = (0..needed)
        .map(|i| {
            let mut id = [0xA5; 16];
            id[..8].copy_from_slice(&i.to_be_bytes());
            (id, NodeData::new(bytes::Bytes::from_static(b"extra")))
        })
        .collect();
    reopened.put_many(&TEST_COLLECTION, &extra_entries).unwrap();
    let grown_capacity = reopened
        .generation(&TEST_COLLECTION)
        .expect("collection still exists")
        .index
        .capacity();
    assert!(
        u64::from(grown_capacity) > checkpoint_capacity,
        "put_many must have grown the index past its checkpoint capacity"
    );
    for (id, _) in &extra_entries {
        assert!(reopened.get(&TEST_COLLECTION, id).unwrap().is_some());
    }
    assert!(reopened
        .get(&TEST_COLLECTION, &entries[0].0)
        .unwrap()
        .is_some());
}

/// A writable open must not silently skip a corrupt pack and publish a
/// partial index. The caller needs an error so it can repair or restore
/// the pack before accepting writes.
#[test]
fn test_writable_open_rejects_corrupt_pack_instead_of_skipping_it() {
    let dir = test_dir("open_corrupt_pack");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let id = distinct_id(0x42);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"payload")),
        )
        .unwrap();
    // Commit the buffered frame so the tamper below actually overwrites
    // on-disk bytes (the record is otherwise only in the append buffer).
    store.sync_all().unwrap();
    let path = store.shards.active_shard().path.clone();
    drop(store);

    let mut bytes = fs::read(&path).unwrap();
    let payload_byte = crate::packfile::HEADER_LEN + 4 + crate::packfile::FRAME_FIXED_LEN as usize;
    bytes[payload_byte] ^= 0xff;
    fs::write(path, bytes).unwrap();

    assert!(PackfileStorage::open(dir).is_err());
}

/// Full and `WriteOnly` keep writing real CRCs, so a reopen scan still
/// verifies them: a physically tampered pack must fail to reopen.
/// `Disabled` frames carry no checksum, so a tampered pack reopens and
/// the engine serves the corrupted bytes as data.
#[test]
fn test_checksum_policy_gates_verification_on_reopen_scan() {
    for (policy, expect_reopen_ok) in [
        (crate::packfile::ChecksumPolicy::Full, false),
        (crate::packfile::ChecksumPolicy::WriteOnly, false),
        (crate::packfile::ChecksumPolicy::Disabled, true),
    ] {
        let dir = test_dir(&format!("checksum_policy_reopen_{policy:?}"));
        {
            let store = PackfileStorage::open_with_policies(dir.clone(), true, policy).unwrap();
            let id = distinct_id(0x52);
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(b"tamper target")),
                )
                .unwrap();
            let (slot, offset) = store
                .generation(&TEST_COLLECTION)
                .unwrap()
                .index
                .lookup(&id)
                .expect("just-written record must be indexed");
            // Tampering happens on-disk: commit the buffered frame so the
            // tamper site computed from `offset` actually lands in the file.
            store.sync_all().unwrap();
            let path = store.shards.get_shard(slot).unwrap().path.clone();
            drop(store);

            let mut bytes = fs::read(&path).unwrap();
            let payload_byte = usize::try_from(offset)
                .expect("u64 offset fits usize on any supported host")
                .saturating_add(4)
                .saturating_add(crate::packfile::FRAME_FIXED_LEN as usize);
            assert!(payload_byte < bytes.len(), "tamper site must be in-file");
            bytes[payload_byte] ^= 0xff;
            bytes[payload_byte + 1] ^= 0x0f;
            fs::write(path, bytes).unwrap();
        }

        if expect_reopen_ok {
            let store = PackfileStorage::open_with_policies(dir, true, policy).unwrap();
            let got = store
                .get(&TEST_COLLECTION, &distinct_id(0x52))
                .unwrap()
                .expect("Disabled reopen must serve the record");
            assert_eq!(
                got.bytes.as_ref().len(),
                b"tamper target".len(),
                "payload frame must still decode"
            );
        } else {
            let reopened = PackfileStorage::open_with_policies(dir.clone(), true, policy);
            match reopened {
                Ok(_) => panic!("reopen scan must reject a tampered pack"),
                Err(err) => {
                    assert!(
                        matches!(err, std::io::Error { .. }),
                        "expected I/O error from scan_and_recover, got {err:?}"
                    );
                }
            }
            // Remove the trusted checkpoint to exercise the read-only
            // full-scan fallback. A valid checkpoint deliberately avoids
            // scanning frames and verifies them lazily on lookup.
            fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
            let read_only = PackfileStorage::open_read_only_with_policies(dir, policy);
            assert!(
                read_only.is_err(),
                "read-only recovery must fail closed instead of omitting a corrupt shard"
            );
        }
    }
}

#[test]
fn test_read_survives_append_after_mmap_established() {
    let dir = test_dir("stale_mmap_regression");
    let store = PackfileStorage::open(dir).unwrap();

    let a = [0xAAu8; 16];
    let b = [0xBBu8; 16];
    let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
    let data_b = NodeData::new(bytes::Bytes::from_static(b"bbbb"));

    store.put(&TEST_COLLECTION, &a, &data_a).unwrap();

    if let Some(gen) = store.generation(&TEST_COLLECTION) {
        gen.cache.clear();
    }
    let got_a = store.get(&TEST_COLLECTION, &a).unwrap();
    assert_eq!(got_a.unwrap().bytes, data_a.bytes);

    store.put(&TEST_COLLECTION, &b, &data_b).unwrap();

    if let Some(gen) = store.generation(&TEST_COLLECTION) {
        gen.cache.clear();
    }
    let got_b = store.get(&TEST_COLLECTION, &b).unwrap();
    assert_eq!(got_b.expect("B must be found").bytes, data_b.bytes);
}

#[test]
fn test_get_not_found() {
    let dir = test_dir("notfound");
    let store = PackfileStorage::open(dir).unwrap();
    assert!(store.get(&TEST_COLLECTION, &[0x00; 16]).unwrap().is_none());
}

#[test]
fn test_cache_hit() {
    let dir = test_dir("cachehit");
    let store = PackfileStorage::open(dir).unwrap();

    let id = [0x01u8; 16];
    let data = NodeData::new(bytes::Bytes::from_static(b"cached"));

    store.put(&TEST_COLLECTION, &id, &data).unwrap();

    let _ = store.get(&TEST_COLLECTION, &id).unwrap();
    let gen = store.generation(&TEST_COLLECTION).unwrap();
    assert_eq!(gen.cache.hits(), 1);

    let _ = store.get(&TEST_COLLECTION, &id).unwrap();
    assert_eq!(gen.cache.hits(), 2);
}

#[test]
fn test_cold_get_does_not_populate_lru() {
    let dir = test_dir("cold_get_no_lru_insert");
    let store = PackfileStorage::open(dir).unwrap();
    let id = [0x01u8; 16];
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"mmap-backed")),
        )
        .unwrap();

    let gen = store.generation(&TEST_COLLECTION).unwrap();
    gen.cache.clear();
    assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
    assert_eq!(gen.cache.len(), 0);
}

#[test]
fn test_delete_collection() {
    let dir = test_dir("delete");
    let store = PackfileStorage::open(dir).unwrap();

    let id = [0x01u8; 16];
    let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

    store.put(&OTHER_COLLECTION, &id, &data).unwrap();

    store.delete_collection(&OTHER_COLLECTION).unwrap();
    assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
    assert!(store.generation(&OTHER_COLLECTION).is_none());
}

#[test]
fn test_delete_collection_does_not_resurrect_on_reopen() {
    let dir = test_dir("delete_no_resurrect");

    let id = [0x01u8; 16];
    let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.put(&OTHER_COLLECTION, &id, &data).unwrap();
        store.delete_collection(&OTHER_COLLECTION).unwrap();
        store.sync_all().unwrap();
    }

    // Reopen from scratch: the on-disk deleted.collections marker must make
    // the startup scan skip OTHER_COLLECTION's leftover packfile records,
    // rather than resurrecting them into a fresh index.
    let store = PackfileStorage::open(dir).unwrap();
    assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
    assert!(store.generation(&OTHER_COLLECTION).is_none());
}

#[test]
fn test_delete_collection_clears_live_roots() {
    let dir = test_dir("delete_live_roots");
    let store = PackfileStorage::open(dir).unwrap();

    let id = [0x01u8; 16];
    store
        .put(
            &OTHER_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();
    store.set_live_roots(&OTHER_COLLECTION, vec![id]);
    assert!(store.live_roots.read().contains_key(&OTHER_COLLECTION));

    store.delete_collection(&OTHER_COLLECTION).unwrap();
    assert!(!store.live_roots.read().contains_key(&OTHER_COLLECTION));
}

#[test]
fn test_batch_put_get() {
    let dir = test_dir("batch");
    let store = PackfileStorage::open(dir).unwrap();

    let entries = ten_record_fixture();

    assert_eq!(
        store.put_many(&TEST_COLLECTION, &entries).unwrap(),
        10,
        "put_many reports every committed entry"
    );
    assert_eq!(
        store.put_many(&TEST_COLLECTION, &[]).unwrap(),
        0,
        "an empty batch commits nothing"
    );

    let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
    let results = store.get_many(&TEST_COLLECTION, &ids).unwrap();
    assert_eq!(results.len(), 10);
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().bytes, entries[i].1.bytes);
    }
}

#[test]
fn test_get_many_preserves_caller_order_despite_sorted_reads() {
    let dir = test_dir("batch_order");
    let store = PackfileStorage::open(dir).unwrap();

    let entries = ten_record_fixture();
    store.put_many(&TEST_COLLECTION, &entries).unwrap();
    if let Some(gen) = store.generation(&TEST_COLLECTION) {
        gen.cache.clear();
    }

    let mut reversed_ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
    reversed_ids.reverse();

    let results = store.get_many(&TEST_COLLECTION, &reversed_ids).unwrap();
    assert_eq!(results.len(), 10);
    for (i, id) in reversed_ids.iter().enumerate() {
        let expected = &entries.iter().find(|(eid, _)| eid == id).unwrap().1;
        assert_eq!(
            results[i].as_ref().expect("record must be found").bytes,
            expected.bytes,
            "result at position {i} must match the id requested at that position"
        );
    }
}

#[test]
fn test_walk_ancestors_stops_at_boundary() {
    let dir = test_dir("walk_basic");
    let store = PackfileStorage::open(dir).unwrap();

    // A -> B -> C -> D -> E, linear chain, edges point to predecessors.
    let ids: Vec<NodeId> = (0..5u8).map(distinct_id).collect();
    for (i, id) in ids.iter().enumerate() {
        let byte = u8::try_from(i).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                id,
                &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
            )
            .unwrap();
    }
    let edges = std::collections::HashMap::from([
        (ids[1], vec![ids[0]]),
        (ids[2], vec![ids[1]]),
        (ids[3], vec![ids[2]]),
        (ids[4], vec![ids[3]]),
    ]);
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

    // frontier=[D], stop_at=[B]: should yield C, D in ancestor-first
    // order — stopping at and excluding B, including the D frontier.
    let walked: Vec<(NodeId, NodeData)> = store
        .walk_ancestors(
            &TEST_COLLECTION,
            &[ids[3]],
            &[ids[1]],
            extract,
            WalkLimits::default(),
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
    assert_eq!(
        bytes,
        vec![b'C', b'D'],
        "walk must be ancestor-first, excluding the stop_at boundary"
    );

    // Empty stop_at: walks all the way back to the collection's true roots.
    let full: Vec<(NodeId, NodeData)> = store
        .walk_ancestors(
            &TEST_COLLECTION,
            &[ids[4]],
            &[],
            extract,
            WalkLimits::default(),
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let full_bytes: Vec<u8> = full.iter().map(|(_, d)| d.bytes[0]).collect();
    assert_eq!(full_bytes, vec![b'A', b'B', b'C', b'D', b'E']);
}

#[test]
fn test_walk_ancestors_branching_dag() {
    let dir = test_dir("walk_branch");
    let store = PackfileStorage::open(dir).unwrap();

    //   A
    //  / \
    // B   C
    //  \ /
    //   D
    let ids: Vec<NodeId> = (0..4u8).map(distinct_id).collect();
    for (i, id) in ids.iter().enumerate() {
        let byte = u8::try_from(i).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                id,
                &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
            )
            .unwrap();
    }
    let edges = std::collections::HashMap::from([
        (ids[1], vec![ids[0]]),
        (ids[2], vec![ids[0]]),
        (ids[3], vec![ids[1], ids[2]]),
    ]);
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

    // frontier=[D], stop_at=[A]: should yield exactly {B, C, D} — A
    // excluded as the boundary, both branches merge back in correctly.
    let walked: Vec<(NodeId, NodeData)> = store
        .walk_ancestors(
            &TEST_COLLECTION,
            &[ids[3]],
            &[ids[0]],
            extract,
            WalkLimits::default(),
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
    // D must come last (ancestor-first order); B/C order between them
    // is unconstrained since they're siblings.
    assert_eq!(*bytes.last().unwrap(), b'D');
    bytes.sort_unstable();
    assert_eq!(bytes, vec![b'B', b'C', b'D']);
}

#[test]
fn test_walk_ancestors_never_reaching_stop_at_walks_to_roots() {
    // A fork that never crosses the supplied stop_at boundary must not
    // be silently truncated there — it's an ancestor walk with stop
    // markers, not a from/to span, so it walks to the collection's true
    // roots instead.
    let dir = test_dir("walk_fork_misses_boundary");
    let store = PackfileStorage::open(dir).unwrap();

    //   ROOT
    //   /  \
    // MAIN  FORK
    //  |
    // TIP
    let root = distinct_id(0);
    let main = distinct_id(1);
    let fork = distinct_id(2);
    let tip = distinct_id(3);
    for (id, byte) in [(root, b'R'), (main, b'M'), (fork, b'F'), (tip, b'T')] {
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from(vec![byte])),
            )
            .unwrap();
    }
    let edges = std::collections::HashMap::from([
        (main, vec![root]),
        (fork, vec![root]),
        (tip, vec![main]),
    ]);
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

    // stop_at names `fork`, which TIP's history never crosses — the
    // walk from TIP must still reach ROOT rather than stopping short.
    let walked: Vec<(NodeId, NodeData)> = store
        .walk_ancestors(
            &TEST_COLLECTION,
            &[tip],
            &[fork],
            extract,
            WalkLimits::default(),
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
    bytes.sort_unstable();
    assert_eq!(bytes, vec![b'M', b'R', b'T']);
}

#[test]
fn test_walk_ancestors_max_nodes_bounds_a_runaway_walk() {
    let dir = test_dir("walk_max_nodes");
    let store = PackfileStorage::open(dir).unwrap();

    // A long chain with a stop_at that never gets hit — max_nodes is
    // the only thing that keeps this walk from covering the chain.
    let ids: Vec<NodeId> = (0..20u8).map(distinct_id).collect();
    for (i, id) in ids.iter().enumerate() {
        let byte = u8::try_from(i % 26).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                id,
                &NodeData::new(bytes::Bytes::from(vec![b'a' + byte])),
            )
            .unwrap();
    }
    let mut edges = std::collections::HashMap::new();
    for i in 1..ids.len() {
        edges.insert(ids[i], vec![ids[i - 1]]);
    }
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

    let walked: Vec<(NodeId, NodeData)> = store
        .walk_ancestors(
            &TEST_COLLECTION,
            &[ids[19]],
            &[],
            extract,
            WalkLimits { max_nodes: Some(5) },
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        walked.len() <= 5,
        "max_nodes must bound the walk even though stop_at was never reached"
    );
}

#[test]
fn test_walk_ancestors_survives_concurrent_repack_gc() {
    // Regression test for the exact concurrency bug flagged in review:
    // an earlier draft resolved each node lazily, at drain time,
    // against whatever generation happened to be live *then* — so a
    // repack that GC'd a node between building the walk and draining
    // it would silently drop that node from the results. Resolving
    // eagerly against one frozen generation snapshot up front (what
    // walk_ancestors does now) must not exhibit that.
    let dir = test_dir("walk_survives_repack");
    let store = PackfileStorage::open(dir).unwrap();

    let a = distinct_id(0);
    let b = distinct_id(1);
    store
        .put(
            &TEST_COLLECTION,
            &a,
            &NodeData::new(bytes::Bytes::from_static(b"A")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &b,
            &NodeData::new(bytes::Bytes::from_static(b"B")),
        )
        .unwrap();
    let edges = std::collections::HashMap::from([(b, vec![a])]);
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

    // Build the walk (resolves and captures A and B eagerly)...
    let walk = store
        .walk_ancestors(&TEST_COLLECTION, &[b], &[], extract, WalkLimits::default())
        .unwrap();

    // ...then, before draining it, run a repack that only keeps B as
    // a live root — A becomes unreachable and gets GC'd out of the
    // collection's index/cache entirely.
    store.set_live_roots(&TEST_COLLECTION, vec![b]);
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    assert_eq!((kept, dropped), (1, 1), "repack must have GC'd A");

    // The already-built walk must still yield both A and B: it
    // captured its data before the repack ran, so it isn't affected
    // by the generation swap that just happened.
    let results: Vec<(NodeId, NodeData)> = walk.collect::<Result<_, _>>().unwrap();
    let mut bytes: Vec<u8> = results.iter().map(|(_, d)| d.bytes[0]).collect();
    bytes.sort_unstable();
    assert_eq!(
        bytes,
        vec![b'A', b'B'],
        "a walk built before a concurrent repack must not lose nodes that repack GC'd afterward"
    );
}

fn setup_repack_mid_lookup(
    test_name: &str,
    id: NodeId,
) -> (
    Arc<PackfileStorage>,
    u16,
    Arc<AtomicBool>,
    TestBeforePinGuard,
) {
    // Put the repack in the exact window between candidate collection and
    // shard pinning, and force its output onto a new shard so the original
    // candidate shard is actually retired.
    let dir = test_dir(test_name);
    let store = Arc::new(PackfileStorage::open(dir).unwrap());
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"payload")),
        )
        .unwrap();
    store.sync().unwrap();
    store.generation(&TEST_COLLECTION).unwrap().cache.clear();
    let old_shard = store
        .generation(&TEST_COLLECTION)
        .unwrap()
        .index
        .lookup_all(&id)
        .next()
        .expect("id is indexed")
        .0;

    let fired = Arc::new(AtomicBool::new(false));
    let hook_fired = Arc::clone(&fired);
    let hook_store = Arc::clone(&store);
    let guard = PackfileStorage::install_test_before_pin(Box::new(move || {
        if !hook_fired.swap(true, Ordering::Relaxed) {
            hook_store.shards.active_shard().file_len.store(
                shard::MAX_SHARD_BYTES - 10,
                std::sync::atomic::Ordering::Release,
            );
            hook_store.set_live_roots(&TEST_COLLECTION, vec![id]);
            hook_store
                .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
                .unwrap();
        }
    }));
    (store, old_shard, fired, guard)
}

#[test]
fn test_get_retries_when_a_repack_retires_the_shard_mid_lookup() {
    let id = distinct_id(0x77);
    let (store, old_shard, fired, _guard) = setup_repack_mid_lookup("get_retry_mid_repack", id);
    let result = store.get(&TEST_COLLECTION, &id).unwrap();

    assert!(fired.load(Ordering::Relaxed), "the test hook must have run");
    assert!(
        store.shards.get_shard(old_shard).is_none(),
        "the repack must have retired the looked-up shard, or this test proves nothing"
    );
    assert!(
        result.is_some(),
        "get must retry and still find a record whose shard a repack retired mid-lookup"
    );
}

#[test]
fn test_get_many_retries_when_a_repack_retires_the_shard_mid_lookup() {
    let id = distinct_id(0x78);
    let (store, old_shard, fired, _guard) =
        setup_repack_mid_lookup("get_many_retry_mid_repack", id);
    let results = store.get_many(&TEST_COLLECTION, &[id]).unwrap();

    assert!(fired.load(Ordering::Relaxed), "the test hook must have run");
    assert!(
        store.shards.get_shard(old_shard).is_none(),
        "the repack must have retired the looked-up shard, or this test proves nothing"
    );
    assert!(
        results[0].is_some(),
        "get_many must retry and still find a record whose shard a repack retired mid-lookup"
    );
}

#[test]
fn test_get_location_returns_the_winning_frame() {
    let dir = test_dir("get_location_winner");
    let store = PackfileStorage::open(dir).unwrap();
    let id = distinct_id(0x60);
    let payload = bytes::Bytes::from_static(b"winner payload");
    store
        .put(&TEST_COLLECTION, &id, &NodeData::new(payload.clone()))
        .unwrap();
    store.sync().unwrap();

    let (pack_id, offset) = store
        .get_location(&TEST_COLLECTION, &id)
        .unwrap()
        .expect("a just-written record is live");
    let shard = store
        .shards
        .all_shards()
        .into_iter()
        .find_map(|(_, shard)| (shard.pack_id == pack_id).then_some(shard))
        .expect("the returned pack id is an open shard");
    let record = store
        .read_at(
            &shard,
            offset,
            store.shards.checksum_policy().verifies_reads(),
        )
        .unwrap();
    assert_eq!(record.hash, id);
    assert_eq!(record.data.as_ref(), payload.as_ref());

    assert_eq!(
        store
            .get_location(&TEST_COLLECTION, &distinct_id(0x61))
            .unwrap(),
        None,
        "an unknown id resolves to no location"
    );
}

#[test]
fn test_get_location_retries_when_a_repack_retires_the_shard_mid_lookup() {
    let id = distinct_id(0x62);
    let (store, old_shard, fired, _guard) =
        setup_repack_mid_lookup("get_location_retry_mid_repack", id);
    let result = store.get_location(&TEST_COLLECTION, &id).unwrap();

    assert!(fired.load(Ordering::Relaxed), "the test hook must have run");
    assert!(
        store.shards.get_shard(old_shard).is_none(),
        "the repack must have retired the looked-up shard, or this test proves nothing"
    );
    assert!(
        result.is_some(),
        "get_location must retry and still find a record whose shard a repack retired mid-lookup"
    );
}

#[test]
fn test_get_location_errors_on_a_corrupt_candidate_instead_of_reporting_missing() {
    let dir = test_dir("get_location_corrupt");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let id = distinct_id(0x63);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"corrupt me")),
        )
        .unwrap();
    let (slot, offset) = store
        .generation(&TEST_COLLECTION)
        .unwrap()
        .index
        .lookup(&id)
        .expect("just-written record is indexed");
    store.sync_all().unwrap();
    let path = store.shards.get_shard(slot).unwrap().path.clone();
    drop(store);

    // Tamper the on-disk payload while no handle holds the pack open, so
    // the mutation is seen identically on every platform.
    let mut bytes = fs::read(&path).unwrap();
    let payload_byte = usize::try_from(offset)
        .expect("u64 offset fits usize")
        .saturating_add(4)
        .saturating_add(crate::packfile::FRAME_FIXED_LEN as usize);
    assert!(payload_byte < bytes.len(), "tamper site must be in-file");
    bytes[payload_byte] ^= 0xff;
    bytes[payload_byte + 1] ^= 0x0f;
    fs::write(&path, bytes).unwrap();

    // A read-only open validates framing at scan time, but the read that
    // `get_location` performs is lazy, so it observes the tamper above.
    let store = PackfileStorage::open_read_only(dir).unwrap();
    let result = store.get_location(&TEST_COLLECTION, &id);
    assert!(
        result.is_err(),
        "a corrupt candidate must surface an error, not a silent None"
    );
}

#[test]
fn test_get_location_rejects_an_active_transaction_overlay() {
    let dir = test_dir("get_location_overlay");
    let store = PackfileStorage::open(dir).unwrap();
    let id = distinct_id(0x64);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"overlay")),
        )
        .unwrap();
    store.sync().unwrap();

    // Simulate an active transaction overlay: `get` would consult it, but
    // `get_location` can only speak about durable pack state and must say so
    // rather than silently reporting a stale location.
    store
        .transaction_overlay_users
        .fetch_add(1, Ordering::AcqRel);
    let result = store.get_location(&TEST_COLLECTION, &id);
    store
        .transaction_overlay_users
        .fetch_sub(1, Ordering::AcqRel);
    assert!(
        matches!(result, Err(StorageError::Internal(_))),
        "get_location must refuse to resolve while an overlay is active, got {result:?}"
    );
}

/// The shard→collection sidecar stamped with the previous version byte (v6)
/// must be rejected by the version gate before attempting to parse records.
/// (Patching only the version byte is sufficient: the reader rejects the file
/// before looking at the body, so a real v6-width body need not be constructed.)
/// The sidecar is a rebuildable acceleration, so rejection routes open to
/// scanning the collection directory, and a subsequent persist rewrites a
/// valid v7 sidecar.
#[test]
fn previous_version_shard_collections_sidecar_is_rejected_and_rebuilt() {
    let dir = test_dir("shard_collections_prev_version");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let a = distinct_id(0xA0);
    store
        .put(
            &TEST_COLLECTION,
            &a,
            &NodeData::new(bytes::Bytes::from_static(b"data")),
        )
        .unwrap();
    store.sync_all().unwrap();
    store.persist_shard_collections().unwrap();
    drop(store);

    let sidecar_path = PackfileStorage::shard_collections_path(&dir);
    let mut buf = std::fs::read(&sidecar_path).unwrap();
    assert_eq!(&buf[0..4], SHARD_ROOMS_MAGIC);
    assert_eq!(buf[4], SHARD_ROOMS_VERSION);

    // Patch only the version byte to the previous version; the body stays
    // at the current width (the gate rejects before parsing it).
    buf[4] = SHARD_ROOMS_VERSION - 1;
    std::fs::write(&sidecar_path, &buf).unwrap();

    assert!(
        read_persisted_shard_collections(&dir).is_none(),
        "a v{} shard-collections sidecar must be rejected by the v{SHARD_ROOMS_VERSION} reader",
        SHARD_ROOMS_VERSION - 1
    );

    // Reopen: must fall back to scanning collections without error
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert!(reopened.collection_index_info(&TEST_COLLECTION).is_some());

    // Rebuilding sidecar:
    reopened.persist_shard_collections().unwrap();
    drop(reopened);

    let rebuilt = std::fs::read(&sidecar_path).unwrap();
    assert_eq!(&rebuilt[0..4], SHARD_ROOMS_MAGIC);
    assert_eq!(rebuilt[4], SHARD_ROOMS_VERSION);
}

#[test]
fn test_walk_ancestors_pins_shards_against_a_mid_walk_repack() {
    // The walk freezes one generation, but a concurrent repack can still
    // retire the shards that generation's index points at. `extract_edges`
    // runs between node resolutions, so triggering the repack from it
    // deterministically reproduces a mid-walk swap; every node must still
    // resolve.
    let dir = test_dir("walk_pins_mid_repack");
    let store = PackfileStorage::open(dir).unwrap();

    let a = distinct_id(0xA0);
    let b = distinct_id(0xA1);
    let c = distinct_id(0xA2);
    for (id, byte) in [(&a, b'A'), (&b, b'B'), (&c, b'C')] {
        store
            .put(
                &TEST_COLLECTION,
                id,
                &NodeData::new(bytes::Bytes::copy_from_slice(&[byte])),
            )
            .unwrap();
    }
    store.sync().unwrap();
    let old_shard = store
        .generation(&TEST_COLLECTION)
        .unwrap()
        .index
        .lookup_all(&c)
        .next()
        .expect("c is indexed")
        .0;

    let edges = std::collections::HashMap::from([(c, vec![b]), (b, vec![a])]);
    let fired = std::cell::Cell::new(false);
    let extract = |hash: &[u8; 16], _data: &[u8]| {
        if !fired.replace(true) {
            // Force the repack's output onto a new shard so the snapshot's
            // shard is retired rather than reused as the live write shard.
            store.shards.active_shard().file_len.store(
                shard::MAX_SHARD_BYTES - 10,
                std::sync::atomic::Ordering::Release,
            );
            store.set_live_roots(&TEST_COLLECTION, vec![c]);
            let (kept, dropped) = store
                .repack_collection_reachable(&TEST_COLLECTION, |h, _d| {
                    edges.get(h).cloned().unwrap_or_default()
                })
                .unwrap();
            assert_eq!((kept, dropped), (3, 0), "all three nodes stay reachable");
        }
        edges.get(hash).cloned().unwrap_or_default()
    };

    let walk = store
        .walk_ancestors(&TEST_COLLECTION, &[c], &[], extract, WalkLimits::default())
        .unwrap();
    let results: Vec<(NodeId, NodeData)> = walk.collect::<Result<_, _>>().unwrap();

    assert!(
        store.shards.get_shard(old_shard).is_none(),
        "the repack must have retired the snapshot's shard, or this test proves nothing"
    );
    assert_eq!(
        results.len(),
        3,
        "the walk must resolve every node even though a repack retired the snapshot's shards mid-walk"
    );
}

#[test]
fn test_multiple_records_same_collection() {
    let dir = test_dir("multi");
    let store = PackfileStorage::open(dir).unwrap();

    for i in 0..5u8 {
        let mut id = [0u8; 16];
        id[0] = i;
        let data = NodeData::new(bytes::Bytes::from(format!("record {i}")));
        store.put(&TEST_COLLECTION, &id, &data).unwrap();
    }

    for i in 0..5u8 {
        let mut id = [0u8; 16];
        id[0] = i;
        let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from(format!("record {i}")));
    }
}

#[test]
fn test_collection_isolation() {
    let dir = test_dir("isolation");
    let store = PackfileStorage::open(dir).unwrap();

    let id_a = [0x42u8; 16];
    let id_b = [0x43u8; 16];
    let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
    let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

    store.put(&TEST_COLLECTION, &id_a, &data_a).unwrap();
    store.put(&OTHER_COLLECTION, &id_b, &data_b).unwrap();

    let got_a = store.get(&TEST_COLLECTION, &id_a).unwrap().unwrap();
    let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
    assert_eq!(got_a.bytes, data_a.bytes);
    assert_eq!(got_b.bytes, data_b.bytes);

    assert!(store.get(&OTHER_COLLECTION, &id_a).unwrap().is_none());
    assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_none());

    store.delete_collection(&TEST_COLLECTION).unwrap();
    assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_none());
    let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
    assert_eq!(got_b.bytes, data_b.bytes);
}

#[test]
fn test_concurrent_federation_swarm() {
    use std::sync::Mutex;
    use std::thread;

    const NUM_WRITERS: usize = 8;
    const EVENTS_PER_WRITER: usize = 500;
    const NUM_READERS: usize = 8;
    const READS_PER_READER: usize = 2000;

    let dir = test_dir("concurrent_federation");
    let store = PackfileStorage::open(dir).unwrap();

    let collection = [0x77u8; 16];
    let written: Mutex<Vec<(NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());
    let read_ok = std::sync::atomic::AtomicUsize::new(0);
    let read_not_found = std::sync::atomic::AtomicUsize::new(0);

    thread::scope(|scope| {
        for w in 0..NUM_WRITERS {
            let store = &store;
            let written = &written;
            scope.spawn(move || {
                for i in 0..EVENTS_PER_WRITER {
                    let mut id = [0u8; 16];
                    id[0] = u8::try_from(w).unwrap();
                    id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
                    let bytes = bytes::Bytes::from(format!("writer {w} event {i}"));
                    let data = NodeData::new(bytes.clone());
                    store.put(&collection, &id, &data).unwrap();
                    written.lock().unwrap().push((id, bytes));
                }
            });
        }

        for _ in 0..NUM_READERS {
            let store = &store;
            let written = &written;
            let read_ok = &read_ok;
            let read_not_found = &read_not_found;
            scope.spawn(move || {
                for i in 0..READS_PER_READER {
                    let snapshot_len = written.lock().unwrap().len();
                    if snapshot_len == 0 {
                        continue;
                    }
                    let idx = i % snapshot_len;
                    let (id, expected_bytes) = written.lock().unwrap()[idx].clone();
                    match store.get(&collection, &id).unwrap() {
                        Some(data) => {
                            assert_eq!(data.bytes, expected_bytes);
                            read_ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        None => {
                            read_not_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    let written = written.into_inner().unwrap();
    assert_eq!(written.len(), NUM_WRITERS * EVENTS_PER_WRITER);

    let mut lost = Vec::new();
    for (id, expected_bytes) in &written {
        match store.get(&collection, id).unwrap() {
            Some(data) => assert_eq!(data.bytes, *expected_bytes),
            None => lost.push(*id),
        }
    }

    eprintln!(
        "concurrent federation swarm: {} writes, {} reads ok, {} reads not-found, {} lost",
        written.len(),
        read_ok.load(std::sync::atomic::Ordering::Relaxed),
        read_not_found.load(std::sync::atomic::Ordering::Relaxed),
        lost.len()
    );

    assert!(
        lost.is_empty(),
        "{} of {} records lost: {lost:?}",
        lost.len(),
        written.len()
    );
}

/// Regression for the epoch-handoff protocol in `persist_index_checkpoint`:
/// full checkpoint rewrites now hold every collection's `put_mutex` only
/// across the fingerprint/index snapshot and delta-epoch rotation, not
/// across the (unlocked) serialize + fsync + rename. Concurrent `put`s
/// landing in that unlocked window must all still be durable after the
/// next sync and a reopen — none may be silently dropped by the epoch
/// rotation, and none may be lost to a torn straddle between the pack
/// fingerprint and the index snapshot.
#[test]
fn test_concurrent_put_survives_checkpoint_rewrites() {
    use std::sync::Mutex;
    use std::thread;

    const NUM_WRITERS: usize = 6;
    const PUTS_PER_WRITER: usize = 300;
    // Force many full rewrites during the run: each rewrite thread
    // iteration invalidates the delta log (via a collection delete on an
    // otherwise-untouched collection) then rewrites, so persistence keeps
    // taking the epoch-rotation path instead of the cheap delta append.
    const REWRITES: usize = 40;

    let dir = test_dir("concurrent_put_checkpoint_rewrite");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    // Seed one throwaway collection per rewrite so each rewrite iteration
    // has a fresh structural invalidation to force (deleting an
    // already-deleted collection is a no-op and wouldn't force a rewrite).
    for r in 0..REWRITES {
        let mut collection = [0u8; 16];
        collection[0] = 0xEE;
        collection[1..9].copy_from_slice(&(r as u64).to_le_bytes());
        let mut id = [0u8; 16];
        id[15] = 1;
        store
            .put(&collection, &id, &NodeData::new(bytes::Bytes::from("seed")))
            .unwrap();
    }
    store.sync_all().unwrap();

    let written: Mutex<Vec<([u8; 16], NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for w in 0..NUM_WRITERS {
            let store = &store;
            let written = &written;
            scope.spawn(move || {
                let mut collection = [0u8; 16];
                collection[0] = 0xAA;
                collection[1] = u8::try_from(w).unwrap();
                for i in 0..PUTS_PER_WRITER {
                    let mut id = [0u8; 16];
                    id[0] = u8::try_from(w).unwrap();
                    id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
                    let bytes = bytes::Bytes::from(format!("writer {w} put {i}"));
                    let data = NodeData::new(bytes.clone());
                    store.put(&collection, &id, &data).unwrap();
                    written.lock().unwrap().push((collection, id, bytes));
                }
            });
        }

        scope.spawn(|| {
            for r in 0..REWRITES {
                let mut throwaway = [0u8; 16];
                throwaway[0] = 0xEE;
                throwaway[1..9].copy_from_slice(&(r as u64).to_le_bytes());
                store.delete_collection(&throwaway).unwrap();
                store.persist_index_checkpoint_best_effort();
            }
        });
    });

    store.sync_all().unwrap();
    drop(store);

    let reopened = PackfileStorage::open(dir).unwrap();
    let written = written.into_inner().unwrap();
    assert_eq!(written.len(), NUM_WRITERS * PUTS_PER_WRITER);

    let mut lost = Vec::new();
    for (collection, id, expected_bytes) in &written {
        match reopened.get(collection, id).unwrap() {
            Some(data) => assert_eq!(data.bytes, *expected_bytes),
            None => lost.push(*id),
        }
    }
    assert!(
        lost.is_empty(),
        "{} of {} records lost to a concurrent checkpoint rewrite: {lost:?}",
        lost.len(),
        written.len()
    );
}

/// Deterministic, fast regression guard for the epoch-handoff protocol's
/// actual point: a concurrent `put()` must not block on the checkpoint
/// rewrite's serialize/write/rename, no matter how long that takes.
/// Rather than inferring "not blocked" from a wall-clock race against
/// real disk I/O (slow, and only as reliable as the dataset is large
/// enough to make a real rewrite slow relative to test-machine noise —
/// see the git history on this function for the timing-based version
/// this replaced), this injects an artificial, arbitrarily long delay
/// into the unlocked window via a test-only hook and asserts a `put()`
/// issued after the rewrite has entered that window returns in a small
/// fraction of it. A proper wall-clock benchmark of this same property
/// lives in `benches/storage.rs` (`cargo bench`), where it belongs.
#[test]
fn test_put_does_not_block_on_slow_checkpoint_rewrite() {
    use std::sync::atomic::AtomicBool;
    use std::thread;
    use std::time::{Duration, Instant};

    const ARTIFICIAL_DELAY: Duration = Duration::from_millis(300);

    let dir = test_dir("put_not_blocked_by_slow_rewrite");
    let store = PackfileStorage::open(dir).unwrap();

    // Establish a first checkpoint so the forced rewrite below has
    // something real to serialize and rewrite.
    let cid = [0x33u8; 16];
    store
        .put(&cid, &[0u8; 16], &NodeData::new(bytes::Bytes::from("seed")))
        .unwrap();
    store.sync_all().unwrap();

    // Force the next rewrite via a structural invalidation.
    let throwaway = [0x44u8; 16];
    store
        .put(
            &throwaway,
            &[0u8; 16],
            &NodeData::new(bytes::Bytes::from("x")),
        )
        .unwrap();
    store.delete_collection(&throwaway).unwrap();

    let entered_unlocked_window = std::sync::Arc::new(AtomicBool::new(false));
    let put_elapsed = std::sync::Mutex::new(None::<Duration>);

    thread::scope(|scope| {
        {
            let store = &store;
            let hook_flag = std::sync::Arc::clone(&entered_unlocked_window);
            scope.spawn(move || {
                store
                    .test_persist_index_checkpoint_with_delay(ARTIFICIAL_DELAY, &hook_flag)
                    .unwrap();
            });
        }
        // Wait for the hook to confirm it has released every lock and is
        // now sleeping inside the artificial delay — a real signal from
        // the rewrite thread, not a fixed-sleep guess that depends on
        // test-machine timing. The put issued below then provably runs
        // concurrently with the unlocked rewrite window.
        while !entered_unlocked_window.load(Ordering::Acquire) {
            thread::yield_now();
        }

        let started = Instant::now();
        store
            .put(
                &cid,
                &[1u8; 16],
                &NodeData::new(bytes::Bytes::from("concurrent")),
            )
            .unwrap();
        *put_elapsed.lock().unwrap() = Some(started.elapsed());
    });

    let put_elapsed = put_elapsed.into_inner().unwrap().unwrap();
    assert!(
        put_elapsed < ARTIFICIAL_DELAY / 3,
        "a put() issued while a rewrite was sleeping in its unlocked window took \
         {put_elapsed:?} — it should return almost immediately, not wait anywhere near the \
         artificial {ARTIFICIAL_DELAY:?} delay; did the lock scope regress to holding \
         put_mutex across the serialize/write/rename again?"
    );
}

/// Crash-boundary: a crash between the D0→D1 epoch rotation and C1's
/// checkpoint rename. D1 exists only in the (now-lost) in-memory state;
/// nothing on disk names its fingerprint. The pack bytes for whatever
/// prompted the rotation are already physically flushed (rotation always
/// flushes first), so the reopen's local fingerprint has already moved
/// past D0's sealed tail — the trusted-log check fails and this must fall
/// back to a full rescan (packfiles stay authoritative), never silently
/// lose the data.
#[test]
fn test_crash_between_rotation_and_checkpoint_rename_falls_back_to_rescan() {
    let dir = test_dir("crash_between_rotation_and_rename");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let cid = [0x11u8; 16];
    let mut written = Vec::new();

    // Round 1: establish C0 via a full (first-ever) rewrite.
    for i in 0..50u64 {
        let mut id = [0u8; 16];
        id[1..9].copy_from_slice(&i.to_le_bytes());
        let bytes = bytes::Bytes::from(format!("round1 {i}"));
        store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
        written.push((id, bytes));
    }
    store.sync_all().unwrap();

    // Round 2: a plain write's sync appends a delta batch continuing C0
    // (D0), never rewriting the checkpoint.
    for i in 50..100u64 {
        let mut id = [0u8; 16];
        id[1..9].copy_from_slice(&i.to_le_bytes());
        let bytes = bytes::Bytes::from(format!("round2 {i}"));
        store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
        written.push((id, bytes));
    }
    store.sync_all().unwrap();

    // Round 3: more writes, then simulate a crash exactly between the
    // epoch rotation (which flushes these bytes and discards their
    // not-yet-appended delta frames) and the checkpoint that would have
    // validated the new epoch.
    for i in 100..150u64 {
        let mut id = [0u8; 16];
        id[1..9].copy_from_slice(&i.to_le_bytes());
        let bytes = bytes::Bytes::from(format!("round3 {i}"));
        store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
        written.push((id, bytes));
    }
    let _ = store.test_rotate_epoch_without_checkpoint();
    drop(store); // simulated crash: no checkpoint, no retirement.

    let reopened = PackfileStorage::open(dir).unwrap();
    let open = reopened.open_timings().expect("open must record timings");
    assert!(open.total > Duration::ZERO);
    assert!(open.packfile_recovery_calls >= 1);
    assert!(open.packfile_open_calls >= 1);
    assert!(open.packfile_recovery + open.packfile_open <= open.shard_open);
    assert_eq!(
        open.path,
        OpenPath::FullScan,
        "D0's sealed tail can no longer match the post-rotation pack state, so the \
         stale-but-still-present log must be rejected wholesale, not partially replayed"
    );
    for (id, expected_bytes) in &written {
        let got = reopened
            .get(&cid, id)
            .unwrap()
            .expect("every write, including the un-appended round 3, survives via the packs");
        assert_eq!(got.bytes.as_ref(), expected_bytes.as_ref());
    }
}

/// Crash-boundary: a crash between C1's checkpoint rename becoming
/// durable and D0's retirement. Both C1 (+ the fresh, still-empty D1) and
/// the now-stale D0 exist on disk at once; the reopen must select C1 and
/// ignore D0 (base fingerprint no longer matches), and the orphaned D0
/// file must be swept away by the sweep-on-open cleanup.
#[test]
fn test_crash_between_checkpoint_rename_and_retirement_uses_new_checkpoint() {
    let dir = test_dir("crash_between_rename_and_retire");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let cid = [0x22u8; 16];
    let mut written = Vec::new();

    // Stay well under `NEW_COLLECTION_INDEX_FLOOR`'s 75%-load grow
    // threshold (48 entries at the floor of 64): this test exercises
    // checkpoint/epoch continuity across a simulated crash, not the
    // index's grow() behavior, and a grow mid-round would invalidate
    // the delta log this test is asserting still exists on disk.
    for i in 0..10u64 {
        let mut id = [0u8; 16];
        id[1..9].copy_from_slice(&i.to_le_bytes());
        let bytes = bytes::Bytes::from(format!("round1 {i}"));
        store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
        written.push((id, bytes));
    }
    store.sync_all().unwrap(); // C0, D0 exists once round 2 appends below.

    for i in 10..20u64 {
        let mut id = [0u8; 16];
        id[1..9].copy_from_slice(&i.to_le_bytes());
        let bytes = bytes::Bytes::from(format!("round2 {i}"));
        store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
        written.push((id, bytes));
    }
    store.sync_all().unwrap(); // D0 now has a committed batch continuing C0.

    let old_d0_fingerprint = {
        let (_new_fingerprint, old_base_fingerprint) =
            store.test_write_checkpoint_without_retire().unwrap();
        old_base_fingerprint.expect("this session had a prior epoch (C0/D0) to retire")
    };
    let d0_path = PackfileStorage::delta_path(&dir, old_d0_fingerprint);
    assert!(
        d0_path.exists(),
        "D0 must still be on disk immediately after the simulated crash point"
    );
    drop(store); // simulated crash: retire_delta_epoch never ran.

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    let open = reopened.open_timings().expect("open must record timings");
    assert_eq!(
        open.path,
        OpenPath::Checkpoint,
        "C1 is durable and matches the current pack state exactly (no writes happened \
         after the snapshot), so the reopen must take the fast path"
    );
    assert!(
        !d0_path.exists(),
        "the orphaned, now-unreachable D0 must be swept away by the writable open's cleanup"
    );
    for (id, expected_bytes) in &written {
        let got = reopened
            .get(&cid, id)
            .unwrap()
            .expect("every write survives a reopen onto the new checkpoint");
        assert_eq!(got.bytes.as_ref(), expected_bytes.as_ref());
    }
}

/// Collection creation and growth become per-collection v3 snapshots, so
/// neither mutation requires a store-wide checkpoint rewrite.
#[test]
fn delta_invalidation_triggers_are_new_collections_and_growth() {
    const ROUNDS: u8 = 5;

    // Arm A: one new collection per barrier -- the Complement room-churn
    // shape. Each create adds a collection snapshot to the v3 log.
    let store_a = PackfileStorage::open(test_dir("delta_triggers_new_collections")).unwrap();
    let before_a = store_a.stats();
    for round in 0..ROUNDS {
        let mut cid = [0u8; 16];
        cid[0] = 0xA0;
        cid[1] = round;
        for i in 0..3u8 {
            store_a
                .put(
                    &cid,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(format!("round {round} node {i}"))),
                )
                .unwrap();
        }
        store_a.sync_all().unwrap();
    }
    let after_a = store_a.stats();
    assert_eq!(
        after_a.delta_invalidations - before_a.delta_invalidations,
        u64::from(ROUNDS),
        "one is_new invalidation per collection created between barriers"
    );
    assert_eq!(
        after_a.checkpoint_writes - before_a.checkpoint_writes,
        1,
        "only the initial empty-store baseline needs a checkpoint"
    );
    assert_eq!(
        after_a.delta_appends - before_a.delta_appends,
        u64::from(ROUNDS - 1),
        "new collections after the baseline append snapshots"
    );

    // Arm B: one collection that never grows. Its first sync establishes
    // the baseline; subsequent writes append incrementals.
    let store_b = PackfileStorage::open(test_dir("delta_triggers_single_collection")).unwrap();
    let before_b = store_b.stats();
    let mut next = 0u8;
    for _ in 0..ROUNDS {
        for _ in 0..3u8 {
            store_b
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(next),
                    &NodeData::new(bytes::Bytes::from(format!("node {next}"))),
                )
                .unwrap();
            next = next.wrapping_add(1);
        }
        store_b.sync_all().unwrap();
    }
    let after_b = store_b.stats();
    assert_eq!(
        after_b.delta_invalidations - before_b.delta_invalidations,
        1,
        "only collection creation is structural"
    );
    assert_eq!(
        after_b.checkpoint_writes - before_b.checkpoint_writes,
        1,
        "one rewrite re-bases the log, then it is continuable"
    );
    assert_eq!(
        after_b.delta_appends - before_b.delta_appends,
        u64::from(ROUNDS - 1),
        "every barrier after the re-base appends the delta"
    );

    // Arm C: force that same collection past the index's 75%-load grow
    // threshold with a batch (`put_many` is the path that sizes and grows
    // an index; the single-record `put` grow path is not what
    // `index_grow_count` tracks). Growth is logical: it emits redo records,
    // not a replacement snapshot.
    let before_c = store_b.stats();
    let entries: Vec<(NodeId, NodeData)> = (next..80u8)
        .map(|i| {
            (
                distinct_id(i),
                NodeData::new(bytes::Bytes::from(format!("node {i}"))),
            )
        })
        .collect();
    store_b.put_many(&TEST_COLLECTION, &entries).unwrap();
    store_b.sync_all().unwrap();
    let after_c = store_b.stats();
    let grows = after_c.index_grow_count - before_c.index_grow_count;
    assert!(
        grows >= 1,
        "crossing the load threshold must grow the index"
    );
    assert_eq!(
        after_c.delta_invalidations - before_c.delta_invalidations,
        0,
        "growth does not invalidate the log: replay grows the table itself"
    );
    assert_eq!(
        after_c.delta_appends - before_c.delta_appends,
        1,
        "a growing collection appends its redo records in one batch"
    );
    assert_log_has_redo_and_no_snapshot(&store_b, entries.len());
    assert_eq!(
        after_c.checkpoint_writes - before_c.checkpoint_writes,
        0,
        "growth no longer rewrites the whole checkpoint"
    );
}

/// The store's current delta log holds at least `min_redo` redo records and
/// no whole-index snapshot.
fn assert_log_has_redo_and_no_snapshot(store: &PackfileStorage, min_redo: usize) {
    let base = store.delta_state.lock().base_fingerprint.unwrap();
    let log =
        delta::read_delta_log_v3(&PackfileStorage::delta_path(&store.base_dir, base)).unwrap();
    assert!(
        log.operations
            .iter()
            .all(|operation| !matches!(operation, DeltaOperation::CollectionSnapshot { .. })),
        "no whole-index snapshot for a growth"
    );
    let redo_ops = log
        .operations
        .iter()
        .filter(|operation| matches!(operation, DeltaOperation::Redo(_)))
        .count();
    assert!(redo_ops >= min_redo, "the batch's records are logged");
}

fn batch_node(index: u32) -> NodeId {
    let mut id = [0u8; 16];
    id[..4].copy_from_slice(&index.to_le_bytes());
    id[15] = 0xB7;
    id
}

/// A batch that does not fit under the cap rotates the log, though the log is
/// still well below the rotation length: the batch (which can carry
/// whole-index snapshots) is bigger than the room left. The rotation is
/// planned and reported as such, and nothing is lost.
#[test]
fn a_batch_that_crosses_the_cap_below_the_rotation_length_rotates_the_log() {
    let dir = test_dir("v3_delta_cap_batch");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0xD3; 16];
    store
        .put(
            &collection,
            &distinct_id(90),
            &NodeData::from_slice(b"base"),
        )
        .unwrap();
    store.sync_all().unwrap();

    store.delta_log_cap_override.store(8192, Ordering::Relaxed);
    store.delta_state.lock().log_bytes = 100;
    assert!(
        100 < store.delta_log_rotate_bytes(),
        "below the rotation length"
    );
    // Enough frames that the batch alone is larger than the room under the
    // (test) cap.
    for id in 0..400u32 {
        store
            .put(&collection, &batch_node(id), &NodeData::from_slice(b"v"))
            .unwrap();
    }
    let before = store.stats();
    store.sync_all().unwrap();
    let after = store.stats();
    assert_eq!(after.delta_appends - before.delta_appends, 0);
    assert_eq!(after.checkpoint_writes - before.checkpoint_writes, 1);
    assert!(
        store.delta_state.lock().log_bytes < 100,
        "the rotation starts a fresh log"
    );
    drop(store);

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    for id in 0..400u32 {
        assert!(reopened
            .get(&collection, &batch_node(id))
            .unwrap()
            .is_some());
    }
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// A pending whole-index snapshot that cannot fit is refused from its
/// projected length while the batch is being built, before the index is
/// serialized, and the projection is the length the batch would have had.
#[test]
fn a_snapshot_that_cannot_fit_is_refused_before_it_is_serialized() {
    let dir = test_dir("v3_snapshot_preflight");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0xD6; 16];
    store
        .put(&collection, &batch_node(1), &NodeData::from_slice(b"one"))
        .unwrap();
    store.sync_all().unwrap();
    store.delta_log_cap_override.store(1024, Ordering::Relaxed);
    store.invalidate_delta_log(&collection);

    let state = store.delta_state.lock().clone();
    let Err(error) = store.build_v3_pending_batch(&state, false) else {
        panic!("the snapshot must not fit under a 1 KiB cap");
    };
    let detail = PackfileStorage::delta_batch_too_large(&error).expect("a too-large batch");
    let blob_len = store
        .collections_read()
        .get(&collection)
        .map(|room| room.load_full().index.serialized_len())
        .unwrap();
    let expected = delta::v3_empty_batch_len() + delta::v3_snapshot_frame_len(blob_len).unwrap();
    assert_eq!(detail.batch_bytes, expected);
    assert_eq!(detail.cap, 1024);

    // With the real cap the same pending snapshot builds.
    store.delta_log_cap_override.store(0, Ordering::Relaxed);
    assert!(store.build_v3_pending_batch(&state, false).is_ok());
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}

/// A delta error that is not "does not fit" stays a failure, not a rotation.
#[test]
fn only_a_batch_that_does_not_fit_counts_as_a_planned_rotation() {
    let full = StorageError::Io(std::io::Error::other(DeltaBatchTooLarge {
        batch_bytes: 10,
        log_bytes: 5,
        cap: 12,
    }));
    let other = StorageError::Io(std::io::Error::other("disk on fire"));
    assert!(PackfileStorage::delta_batch_too_large(&full).is_some());
    assert!(PackfileStorage::delta_batch_too_large(&other).is_none());
}

/// A log that has grown to the rotation length is replaced by a checkpoint
/// at the next sync, on purpose and before the cap; below it, the append
/// path is untouched.
#[test]
fn a_delta_log_is_rotated_before_it_reaches_the_cap() {
    let dir = test_dir("delta_log_rotation");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0xD4; 16];
    store
        .put(&collection, &distinct_id(1), &NodeData::from_slice(b"base"))
        .unwrap();
    store.sync_all().unwrap();

    // Well under the rotation length: an ordinary append.
    store
        .put(
            &collection,
            &distinct_id(2),
            &NodeData::from_slice(b"below"),
        )
        .unwrap();
    let before = store.stats();
    store.sync_all().unwrap();
    let after = store.stats();
    assert_eq!(after.delta_appends - before.delta_appends, 1);
    assert_eq!(after.checkpoint_writes - before.checkpoint_writes, 0);

    // At the rotation length: a checkpoint, with no failed append first.
    let at = store.delta_log_rotate_bytes();
    store.delta_state.lock().log_bytes = at;
    store
        .put(&collection, &distinct_id(3), &NodeData::from_slice(b"at"))
        .unwrap();
    let before = store.stats();
    store.sync_all().unwrap();
    let after = store.stats();
    assert_eq!(after.delta_appends - before.delta_appends, 0);
    assert_eq!(after.checkpoint_writes - before.checkpoint_writes, 1);
    assert!(
        store.delta_state.lock().log_bytes < store.delta_log_rotate_bytes(),
        "the rotation starts a fresh log"
    );
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    for id in 1..=3 {
        assert!(reopened
            .get(&collection, &distinct_id(id))
            .unwrap()
            .is_some());
    }
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// A full checkpoint records where its time went. The phases are disjoint
/// stretches of one call, so they never add up to more than it; what they do
/// not cover is reported by `unaccounted`.
#[test]
fn a_checkpoint_records_a_phase_breakdown() {
    let dir = test_dir("checkpoint_breakdown");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    assert!(store.checkpoint_breakdown().is_none());
    store
        .put(&[0xD5; 16], &distinct_id(1), &NodeData::from_slice(b"one"))
        .unwrap();
    store.sync_all().unwrap();
    let breakdown = store.checkpoint_breakdown().expect("a checkpoint ran");
    assert!(breakdown.total > Duration::ZERO);
    assert!(
        breakdown.unaccounted() <= breakdown.total,
        "the residual cannot exceed the whole"
    );
    assert!(breakdown.write > Duration::ZERO, "the write fsyncs");
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_puts_during_sync_keep_using_the_v3_append_path() {
    let dir = test_dir("concurrent_put_sync_delta");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"base")),
        )
        .unwrap();
    store.sync_all().unwrap();
    let before = store.stats();

    let writers: Vec<_> = (2..18u8)
        .map(|seed| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                store
                    .put(
                        &TEST_COLLECTION,
                        &distinct_id(seed),
                        &NodeData::new(bytes::Bytes::from(vec![seed; 128])),
                    )
                    .unwrap();
            })
        })
        .collect();
    for _ in 0..32 {
        store.sync_all().unwrap();
        std::thread::yield_now();
    }
    for writer in writers {
        writer.join().unwrap();
    }
    store.sync_all().unwrap();

    let after = store.stats();
    assert_eq!(
        after.checkpoint_writes - before.checkpoint_writes,
        0,
        "concurrent puts must not make the pre-lock fingerprint stale and force rewrites"
    );
    assert!(after.delta_appends - before.delta_appends > 0);
    drop(store);

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    for seed in 2..18u8 {
        assert!(reopened
            .get(&TEST_COLLECTION, &distinct_id(seed))
            .unwrap()
            .is_some());
    }
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn checkpoint_rewrite_budget_defers_invalidated_rewrites() {
    const ROUNDS: u8 = 5;
    let dir = test_dir("checkpoint_rewrite_budget");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    // Once a rewrite has run, defer the next for an hour; no size budget.
    store.set_checkpoint_rewrite_budget(Duration::from_secs(3600), 0);
    let before = store.stats();
    let mut written: Vec<([u8; 16], [u8; 16], Vec<u8>)> = Vec::new();
    for round in 0..ROUNDS {
        let mut cid = [0u8; 16];
        cid[0] = 0xB0;
        cid[1] = round;
        for i in 0..3u8 {
            let id = distinct_id(i);
            let value = format!("budget round {round} node {i}").into_bytes();
            store
                .put(&cid, &id, &NodeData::new(bytes::Bytes::from(value.clone())))
                .unwrap();
            written.push((cid, id, value));
        }
        store.sync_all().unwrap();
    }
    let after = store.stats();
    let checkpoints = after.checkpoint_writes - before.checkpoint_writes;
    let skips = after.checkpoint_skips - before.checkpoint_skips;
    eprintln!(
        "rewrite-budget impact over {ROUNDS} barriers: checkpoint_writes={checkpoints} \
         checkpoint_skips={skips} delta_appends={}",
        after.delta_appends - before.delta_appends
    );
    assert_eq!(
        checkpoints, 1,
        "one baseline rewrite, then the budget defers"
    );
    assert_eq!(
        skips, 0,
        "v3 snapshots avoid checkpoint skips for ordinary structural changes"
    );
    assert_eq!(
        after.delta_appends - before.delta_appends,
        u64::from(ROUNDS - 1)
    );
    drop(store);

    // The v3 log replays the collection snapshots on reopen.
    let reopened = PackfileStorage::open(dir).unwrap();
    assert_eq!(
        reopened
            .open_timings()
            .expect("open must record timings")
            .path,
        OpenPath::Checkpoint,
        "the checkpoint plus v3 log should avoid a full scan"
    );
    for (cid, id, expected) in &written {
        let got = reopened
            .get(cid, id)
            .unwrap()
            .expect("record survives the rescan");
        assert_eq!(got.bytes.as_ref(), expected.as_slice());
    }
}

/// With a journal enabled, `sync` makes the WAL group commit the
/// durability point rather than the per-shard pack fsyncs.
#[test]
fn journal_sync_routes_durability_through_wal() {
    let dir = test_dir("journal_sync_routing");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(dir.join("wal.bin")).unwrap();
    let id = distinct_id(7);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"wal")),
        )
        .unwrap();
    store.sync_all().unwrap();
    let timings = store.sync_timings().expect("sync must record timings");
    assert!(
        timings.wal > Duration::ZERO,
        "the journal group commit is the durability point"
    );
    assert_eq!(
        timings.pack_fsync,
        Duration::ZERO,
        "the barrier path must not fsync pack shards with a WAL"
    );
    assert_eq!(timings.journal_sync_calls, 1);
    assert!(timings.journal_records >= 1);
    assert!(timings.journal_fsync > Duration::ZERO);
    assert!(
        timings.journal_lock_wait + timings.journal_fsync <= timings.wal,
        "reported journal phases cannot exceed the WAL phase"
    );
    let stats = store.stats();
    assert_eq!(stats.publish_calls, 1);
    assert_eq!(stats.sync_diagnostics.worst_syncs.len(), 1);
    assert_eq!(stats.sync_diagnostics.peak_journal_in_flight, 1);
    assert_eq!(
        stats
            .sync_diagnostics
            .fsync_latency
            .buckets
            .iter()
            .sum::<u64>(),
        1
    );
    assert_eq!(
        stats
            .sync_diagnostics
            .lock_wait_latency
            .buckets
            .iter()
            .sum::<u64>(),
        1
    );
    let interval = store.take_sync_diagnostics();
    assert_eq!(interval.peak_journal_in_flight, 1);
    assert!(interval.max_journal_fsync > Duration::ZERO);
    assert_eq!(interval.max_journal_lock_wait, timings.journal_lock_wait);
    assert_eq!(interval.worst_syncs.len(), 1);
    assert_eq!(
        interval.fsync_latency.buckets.iter().sum::<u64>(),
        1,
        "taking diagnostics must return the completed interval"
    );
    assert_eq!(
        store.take_sync_diagnostics(),
        SyncDiagnosticsSnapshot::default(),
        "taking diagnostics must reset only the diagnostics interval"
    );
    assert_eq!(
        store.stats().sync_totals.calls,
        1,
        "taking diagnostics must preserve cumulative sync totals"
    );
    assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
    assert!(store.journal().expect("journal enabled").committed_lsn() >= 1);
}

/// A mutation committed to the journal after the last checkpoint is
/// re-applied by `replay_journal` on a fresh open.
#[test]
fn journal_replays_post_checkpoint_mutations_on_reopen() {
    let dir = test_dir("journal_replay");
    let journal_path = dir.join("wal.bin");
    let id = distinct_id(9);
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&journal_path).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(8),
                &NodeData::new(bytes::Bytes::from_static(b"checkpointed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        // A second, non-structural put: the next sync's delta path does not
        // write a checkpoint, so this mutation stays journal-only.
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"wal-only")),
            )
            .unwrap();
        store.sync_all().unwrap();
    }
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.enable_journal(&journal_path).unwrap();
    let replayed = reopened.replay_journal().unwrap();
    assert!(
        replayed >= 1,
        "the post-checkpoint mutation must be replayed"
    );
    assert!(reopened.get(&TEST_COLLECTION, &id).unwrap().is_some());
}

/// A single-pool journal may reclaim an older committed group while a
/// later group remains in the segment. Reopening that same store must
/// still replay the surviving suffix.
#[test]
fn single_pool_reclaim_then_replay_surviving_group() {
    let dir = test_dir("single_pool_reclaim_replay");
    let journal_path = dir.join("wal.bin");
    let reclaimed_id = distinct_id(10);
    let surviving_id = distinct_id(11);
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&journal_path).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(8),
                &NodeData::new(bytes::Bytes::from_static(b"checkpointed")),
            )
            .unwrap();
        store.sync_all().unwrap();

        store
            .put(
                &TEST_COLLECTION,
                &reclaimed_id,
                &NodeData::new(bytes::Bytes::from_static(b"reclaimed")),
            )
            .unwrap();
        let reclaimed_lsn = store.journal().unwrap().published_lsn();
        store.sync_all().unwrap();

        store
            .put(
                &TEST_COLLECTION,
                &surviving_id,
                &NodeData::new(bytes::Bytes::from_static(b"surviving")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let journal = store.journal().unwrap();
        let surviving_lsn = journal.published_lsn();
        assert!(surviving_lsn > reclaimed_lsn);
        journal.reclaim_through(reclaimed_lsn).unwrap();
    }

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.enable_journal(&journal_path).unwrap();
    assert_eq!(reopened.replay_journal().unwrap(), 1);
    assert!(reopened
        .get(&TEST_COLLECTION, &surviving_id)
        .unwrap()
        .is_some());
    assert!(reopened
        .get(&TEST_COLLECTION, &reclaimed_id)
        .unwrap()
        .is_some());
}

/// A checkpoint must bound journal replay by the LSN that is actually
/// *committed*, never the published one. A concurrent put can publish
/// above the last WAL commit, and LSNs above `committed_lsn` are discarded
/// and reused after a crash -- recording one as covered would make a
/// reopen skip a future mutation that reuses it. Drive the full-checkpoint
/// path directly so a pending (published, uncommitted) put is live without
/// the barrier commit that a real `sync` would have done first.
#[test]
fn checkpoint_records_the_committed_journal_lsn() {
    let dir = test_dir("journal_checkpoint_committed_lsn");
    let journal_path = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&journal_path).unwrap();

    // Commit one mutation so the journal has a non-zero durable prefix.
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(41),
            &NodeData::new(bytes::Bytes::from_static(b"committed")),
        )
        .unwrap();
    store.sync_all().unwrap();
    let journal = store.journal().expect("journal enabled");
    let committed = journal.committed_lsn();
    assert_eq!(
        journal.published_lsn(),
        committed,
        "test setup: a completed sync leaves nothing published-but-uncommitted"
    );
    assert!(
        committed > 0,
        "test setup: the first sync must commit an LSN"
    );

    // Publish a second mutation into a new collection WITHOUT committing:
    // this is the state a concurrent `put` leaves behind after a barrier.
    store
        .put(
            &SECOND_COLLECTION,
            &distinct_id(42),
            &NodeData::new(bytes::Bytes::from_static(b"pending")),
        )
        .unwrap();
    let journal = store.journal().expect("journal enabled");
    let published = journal.published_lsn();
    assert!(
        published > journal.committed_lsn(),
        "test setup: the second put must be published but uncommitted"
    );

    // Run the full-checkpoint path as a sync would, but without the
    // barrier that would advance `committed_lsn` past the pending put.
    assert!(
        store
            .persist_index_checkpoint()
            .expect("checkpoint must succeed"),
        "a dirty store must write a checkpoint"
    );

    let covered = PackfileStorage::read_journal_lsn(&dir);
    assert_eq!(
        covered, committed,
        "the checkpoint must record the committed LSN, not a published-only one"
    );
    assert!(
        covered < published,
        "recording the published LSN would let a reopen skip a reused LSN"
    );
}

/// Hold the next background checkpoint tail until the returned sender is
/// used or dropped; with `fail` the tail then errors instead of installing.
fn hold_next_checkpoint_tail(store: &PackfileStorage, fail: bool) -> std::sync::mpsc::Sender<()> {
    store.hold_next_checkpoint_tail(fail)
}

fn put_bytes(store: &PackfileStorage, byte: u8) {
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(byte),
            &NodeData::new(bytes::Bytes::from(vec![byte; 8])),
        )
        .unwrap();
}

fn assert_all_present(store: &PackfileStorage, bytes: &[u8]) {
    for &byte in bytes {
        assert!(
            store
                .get(&TEST_COLLECTION, &distinct_id(byte))
                .unwrap()
                .is_some(),
            "record {byte} must be readable"
        );
    }
}

/// Copy every file of `from` into a fresh directory `name`, the disk a crash
/// at this instant would leave (files here are all fsynced or rebuildable).
fn copy_store_dir(from: &Path, name: &str) -> PathBuf {
    let to = test_dir(name);
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            // The live writer lock is not database state in a crash image.
            if entry.file_name() == ".mtxdb.lock" {
                continue;
            }
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    to
}

/// The first sync of a store that opened without a checkpoint takes the full
/// path. While its tail is held the WAL is neither covered nor reclaimed, a
/// second sync starts no second checkpoint, everything synced in the window
/// survives a crash at that instant, and releasing the tail installs the
/// image and advances coverage.
#[test]
fn a_background_checkpoint_holds_the_wal_until_its_image_is_durable() {
    let dir = test_dir("bg_checkpoint_holds_wal");
    let journal_path = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&journal_path).unwrap();
    store.set_background_checkpoint(true);
    let covered_before = store.durable_coverage();
    put_bytes(&store, 1);
    let release = hold_next_checkpoint_tail(&store, false);
    store.sync().unwrap();
    assert!(store.checkpoint_in_flight(), "the tail runs on a worker");
    assert_eq!(store.durable_coverage(), covered_before);

    put_bytes(&store, 2);
    store.sync().unwrap();
    assert!(store.checkpoint_in_flight(), "one worker, not two");
    assert_eq!(
        store.durable_coverage(),
        covered_before,
        "no coverage before the image is durable"
    );

    // A crash now finds no new image, the old (absent) one, and the WAL.
    let crashed = copy_store_dir(&dir, "bg_checkpoint_holds_wal_crash");
    let after_crash = PackfileStorage::open(crashed.clone()).unwrap();
    after_crash.enable_journal(crashed.join("wal.bin")).unwrap();
    after_crash.replay_journal().unwrap();
    assert_all_present(&after_crash, &[1, 2]);
    drop(after_crash);

    drop(release);
    store.wait_for_checkpoint();
    assert!(!store.checkpoint_in_flight());
    assert!(
        store.durable_coverage() > covered_before,
        "the installed image advances coverage"
    );
    assert!(store.checkpoint_breakdown().is_some());
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.enable_journal(&journal_path).unwrap();
    reopened.replay_journal().unwrap();
    assert_all_present(&reopened, &[1, 2]);
}

/// A collection created after the handoff, while the tail is held, is in
/// neither the captured image nor a log the image names. It must survive a
/// crash before the tail finishes and a reopen after it: the creation lock
/// is released at the handoff on the strength of this.
#[test]
fn a_collection_created_during_a_background_checkpoint_is_not_lost() {
    let dir = test_dir("bg_checkpoint_new_collection");
    let journal_path = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&journal_path).unwrap();
    store.set_background_checkpoint(true);
    put_bytes(&store, 1);
    let release = hold_next_checkpoint_tail(&store, false);
    store.sync().unwrap();
    assert!(store.checkpoint_in_flight());
    store
        .put(
            &SECOND_COLLECTION,
            &distinct_id(9),
            &NodeData::new(bytes::Bytes::from_static(b"late")),
        )
        .unwrap();
    store.sync().unwrap();
    let crashed = copy_store_dir(&dir, "bg_checkpoint_new_collection_crash");
    let after_crash = PackfileStorage::open(crashed.clone()).unwrap();
    after_crash.enable_journal(crashed.join("wal.bin")).unwrap();
    after_crash.replay_journal().unwrap();
    assert!(after_crash
        .get(&SECOND_COLLECTION, &distinct_id(9))
        .unwrap()
        .is_some());
    drop(after_crash);
    drop(release);
    store.wait_for_checkpoint();
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.enable_journal(&journal_path).unwrap();
    reopened.replay_journal().unwrap();
    assert_all_present(&reopened, &[1]);
    assert!(reopened
        .get(&SECOND_COLLECTION, &distinct_id(9))
        .unwrap()
        .is_some());
}

/// A failed tail leaves the old state valid, reports, and the next sync
/// rewrites the checkpoint instead of appending to the orphaned epoch.
#[test]
fn a_failed_background_checkpoint_keeps_the_wal_and_is_retried() {
    let dir = test_dir("bg_checkpoint_fails");
    let journal_path = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&journal_path).unwrap();
    store.set_background_checkpoint(true);
    put_bytes(&store, 1);
    let release = hold_next_checkpoint_tail(&store, true);
    store.sync().unwrap();
    assert!(store.checkpoint_in_flight());
    drop(release);
    store.wait_for_checkpoint();
    assert!(!store.checkpoint_in_flight());
    assert_eq!(store.durable_coverage(), 0, "a failed tail claims nothing");
    assert!(
        store.delta_state.lock().base_fingerprint.is_none(),
        "the orphaned epoch is forgotten"
    );
    put_bytes(&store, 2);
    store.sync().unwrap();
    store.wait_for_checkpoint();
    assert!(store.durable_coverage() > 0, "the retry installed an image");
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.enable_journal(&journal_path).unwrap();
    reopened.replay_journal().unwrap();
    assert_all_present(&reopened, &[1, 2]);
}

#[cfg(feature = "multi-reader")]
/// Near the WAL cap a sync waits for the tail instead of letting it run on
/// with reclaim suppressed.
#[test]
fn a_sync_in_the_wal_emergency_zone_waits_for_the_tail() {
    let dir = test_dir("bg_checkpoint_emergency");
    let journal_path = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&journal_path).unwrap();
    store.set_background_checkpoint(true);
    put_bytes(&store, 1);
    let release = hold_next_checkpoint_tail(&store, false);
    store.sync().unwrap();
    assert!(store.checkpoint_in_flight());
    let journal = store.journal().expect("journal enabled");
    journal.set_segment_cap(journal.segment_len() + journal.segment_len() / 4);
    assert!(journal.in_emergency_zone());
    put_bytes(&store, 2);
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        drop(release);
    });
    store.sync().unwrap();
    releaser.join().unwrap();
    assert!(
        !store.checkpoint_in_flight(),
        "the sync had to wait for the tail"
    );
    assert!(store.durable_coverage() > 0);
}

#[test]
fn test_swizzle_callback() {
    static SWIZZLE_CALLS: AtomicU64 = AtomicU64::new(0);
    static CACHED_CHILDREN_FOUND: AtomicU64 = AtomicU64::new(0);

    fn test_swizzle(
        data: &NodeData,
        children: &[NodeId],
        cached: &[Option<Arc<NodeData>>],
    ) -> NodeData {
        SWIZZLE_CALLS.fetch_add(1, Ordering::Relaxed);
        assert_eq!(children.len(), 2);
        for entry in cached {
            if entry.is_some() {
                CACHED_CHILDREN_FOUND.fetch_add(1, Ordering::Relaxed);
            }
        }
        data.clone()
    }

    let dir = test_dir("swizzle");
    let store = PackfileStorage::open_with_swizzle(dir, 100, test_swizzle).unwrap();

    let parent_id = [0x10u8; 16];
    let child_a = [0x20u8; 16];
    let child_b = [0x30u8; 16];

    let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
    let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
    let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));

    store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
    store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();
    store
        .put(&TEST_COLLECTION, &parent_id, &parent_data)
        .unwrap();

    if let Some(gen) = store.generation(&TEST_COLLECTION) {
        gen.cache.clear();
    }
    store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
    store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();

    let extract = |_data: &NodeData| -> Vec<NodeId> { vec![child_a, child_b] };

    let result = store
        .get_swizzled(&TEST_COLLECTION, &parent_id, extract)
        .unwrap();
    assert!(result.is_some());
    let node_ref = result.unwrap();
    assert!(node_ref.is_resolved());
    assert_eq!(node_ref.structural_hash(), &parent_id);

    assert_eq!(SWIZZLE_CALLS.load(Ordering::Relaxed), 1);
    assert_eq!(CACHED_CHILDREN_FOUND.load(Ordering::Relaxed), 2);
}

#[test]
fn test_sync() {
    let dir = test_dir("sync");
    let store = PackfileStorage::open(dir).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &[0xAA; 16],
            &NodeData::new(bytes::Bytes::from_static(b"hi")),
        )
        .unwrap();
    store.sync().unwrap();
}

#[test]
fn test_probe_reopen_first_append_sync_path_no_growth() {
    // Discriminates reopen-materialization cost from real capacity-growth
    // cost: build to a load well under the 75% grow threshold (so the
    // post-reopen append batch cannot trigger `index_grow_count`), then
    // reopen and append. Reopen should continue the v3 epoch directly.
    let dir = test_dir("probe_reopen_sync_no_growth");
    let total = 2000u32; // 2032 / 4096 is well under the 75% grow threshold.
    build_reopen_probe_checkpoint(&dir, total);
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let before = store.stats();
        reopen_probe_put_range(&store, total..(total + 32), 32, |_| {
            NodeData::new(bytes::Bytes::from_static(b"y"))
        });
        let after_put = store.stats();
        let d = store.delta_state.lock();
        let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
        eprintln!(
            "AFTER REOPEN PUT: cap={} len={} load={:.3} clones+={} grows+={} invalids+={} | pending={:?}",
            room.index.capacity(),
            room.index.len(),
            reopen_probe_load_factor(&room),
            after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
            after_put.index_grow_count - before.index_grow_count,
            after_put.delta_invalidations - before.delta_invalidations,
            d.pending.keys().collect::<Vec<_>>(),
        );
        assert_eq!(
            after_put.index_grow_count - before.index_grow_count,
            0,
            "test setup invariant violated: this batch must not trigger real growth"
        );
        assert_eq!(
            after_put.delta_invalidations - before.delta_invalidations,
            0
        );
        assert!(matches!(
            d.pending.get(&REOPEN_PROBE_COLLECTION),
            Some(PendingDelta::Redo(_))
        ));
        drop(d);
        store.sync().unwrap();
        let st = store.stats();
        let ts = store.sync_timings().unwrap();
        eprintln!(
            "REOPEN SYNC TIMINGS (no growth): checkpoint={:?} delta={:?} total={:?}",
            ts.checkpoint, ts.delta_log, ts.total
        );
        eprintln!(
            "REOPEN SYNC STATS (no growth): ckpt_writes+={} delta_appends+={}",
            st.checkpoint_writes - before.checkpoint_writes,
            st.delta_appends - before.delta_appends,
        );
        assert_eq!(st.checkpoint_writes - before.checkpoint_writes, 0);
        assert_eq!(st.delta_appends - before.delta_appends, 1);
        assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
        assert!(ts.delta_log > std::time::Duration::ZERO);
    }
}

const REOPEN_PROBE_COLLECTION: [u8; 16] = [0x11; 16];

fn reopen_probe_id(i: u32) -> NodeId {
    let mut id = [0u8; 16];
    id[0..4].copy_from_slice(&i.to_le_bytes());
    id[8..12].copy_from_slice(&(i ^ 0x9E37_79B9).to_le_bytes());
    id
}

fn reopen_probe_payload(i: u32) -> NodeData {
    let mut payload = vec![0u8; 1024];
    let seed = u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF;
    payload[..2].copy_from_slice(&seed.to_le_bytes()[..2]);
    NodeData::new(bytes::Bytes::from(payload))
}

fn reopen_probe_put_range(
    store: &PackfileStorage,
    range: std::ops::Range<u32>,
    batch_size: usize,
    payload: impl Fn(u32) -> NodeData,
) {
    let mut batch = Vec::with_capacity(batch_size);
    for i in range {
        batch.push((reopen_probe_id(i), payload(i)));
        if batch.len() == batch_size {
            store.put_many(&REOPEN_PROBE_COLLECTION, &batch).unwrap();
            batch.clear();
        }
    }
    if !batch.is_empty() {
        store.put_many(&REOPEN_PROBE_COLLECTION, &batch).unwrap();
    }
}

fn reopen_probe_load_factor(room: &RoomGeneration) -> f64 {
    f64::from(u32::try_from(room.index.len()).expect("probe index length fits in u32"))
        / f64::from(room.index.capacity())
}

fn build_reopen_probe_checkpoint(dir: &std::path::Path, total: u32) {
    let store = PackfileStorage::open_with_cache_and_policies(
        dir.to_path_buf(),
        0,
        true,
        packfile::ChecksumPolicy::Full,
    )
    .unwrap()
    .with_append_policy(shard::AppendPolicy::buffered());
    reopen_probe_put_range(&store, 0..total, 256, reopen_probe_payload);
    store.sync_all().unwrap();
    let stats = store.stats();
    let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
    eprintln!(
        "BUILD done: cap={} len={} load={:.3} gen={} | clones={} grows={} ckpt_writes={} delta_appends={}",
        room.index.capacity(), room.index.len(), reopen_probe_load_factor(&room), room.generation,
        stats.put_many_clone_path_calls, stats.index_grow_count, stats.checkpoint_writes,
        stats.delta_appends,
    );
}

#[test]
fn test_probe_reopen_first_append_sync_path() {
    let dir = test_dir("probe_reopen_sync");
    let total = 3051u32;
    build_reopen_probe_checkpoint(&dir, total);
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let d = store.delta_state.lock();
        eprintln!(
            "REOPEN delta_state: base_fingerprint={:?} pending={} log_bytes={}",
            d.base_fingerprint,
            d.pending.len(),
            d.log_bytes
        );
        let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
        eprintln!(
            "REOPEN room: cap={} len={} load={:.3} gen={} base_gen={:?}",
            room.index.capacity(),
            room.index.len(),
            reopen_probe_load_factor(&room),
            room.generation,
            d.base_generations.get(&REOPEN_PROBE_COLLECTION),
        );
        drop(d);
        drop(room);

        let before = store.stats();
        reopen_probe_put_range(&store, total..(total + 32), 32, |_| {
            NodeData::new(bytes::Bytes::from_static(b"y"))
        });
        let after_put = store.stats();
        let d = store.delta_state.lock();
        eprintln!(
            "AFTER REOPEN PUT: clones+={} grows+={} invalids+={} | pending={:?}",
            after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
            after_put.index_grow_count - before.index_grow_count,
            after_put.delta_invalidations - before.delta_invalidations,
            d.pending.keys().collect::<Vec<_>>(),
        );
        assert_eq!(
            after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
            1
        );
        assert_eq!(after_put.index_grow_count - before.index_grow_count, 1);
        assert_eq!(
            after_put.delta_invalidations - before.delta_invalidations,
            0,
            "growth of a reopened index is logical, not a snapshot"
        );
        assert!(matches!(
            d.pending.get(&REOPEN_PROBE_COLLECTION),
            Some(PendingDelta::Redo(_))
        ));
        drop(d);
        store.sync().unwrap();
        let st = store.stats();
        let ts = store.sync_timings().unwrap();
        eprintln!(
            "REOPEN SYNC TIMINGS: flush={:?} fsync={:?} checkpoint={:?} delta={:?} total={:?}",
            ts.pack_flush, ts.pack_fsync, ts.checkpoint, ts.delta_log, ts.total
        );
        eprintln!(
            "REOPEN SYNC STATS: ckpt_writes+={} delta_appends+={}",
            st.checkpoint_writes - before.checkpoint_writes,
            st.delta_appends - before.delta_appends,
        );
        assert_eq!(st.checkpoint_writes - before.checkpoint_writes, 0);
        assert_eq!(st.delta_appends - before.delta_appends, 1);
        assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
        assert!(ts.delta_log > std::time::Duration::ZERO);

        reopen_probe_put_range(&store, (total + 32)..(total + 288), 256, |_| {
            NodeData::new(bytes::Bytes::from_static(b"z"))
        });
        store.sync().unwrap();
        let ts = store.sync_timings().unwrap();
        eprintln!(
            "SECOND SYNC TIMINGS: checkpoint={:?} delta={:?} total={:?}",
            ts.checkpoint, ts.delta_log, ts.total
        );
        assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
        assert!(ts.delta_log > std::time::Duration::ZERO);
    }
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    for i in total..(total + 288) {
        let got = reopened
            .get(&REOPEN_PROBE_COLLECTION, &reopen_probe_id(i))
            .unwrap()
            .expect("snapshot and following incrementals survive reopen");
        assert_eq!(got.bytes.as_ref(), if i < total + 32 { b"y" } else { b"z" });
    }
}

#[test]
fn test_rebuild_index_triggers_on_full_table() {
    let dir = test_dir("rebuild_index_full");
    let store = PackfileStorage::open(dir).unwrap();

    let threshold = 3073u32;
    for i in 0..threshold {
        let mut id = [0u8; 16];
        id[0..4].copy_from_slice(&i.to_le_bytes());
        id[8..12].copy_from_slice(&(i + 1).to_le_bytes());
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
    }

    let mut extra = [0u8; 16];
    extra[0..4].copy_from_slice(&threshold.to_le_bytes());
    extra[8..12].copy_from_slice(&(threshold + 1).to_le_bytes());
    store
        .put(
            &TEST_COLLECTION,
            &extra,
            &NodeData::new(bytes::Bytes::from_static(b"y")),
        )
        .unwrap();

    let got = store.get(&TEST_COLLECTION, &extra).unwrap().unwrap();
    assert_eq!(got.bytes, bytes::Bytes::from_static(b"y"));
}

#[test]
fn test_put_many_publishes_successive_batches() {
    // Regression coverage for successful copy-on-write batch publication:
    // every record remains readable across several batches that need no
    // index growth.
    let dir = test_dir("put_many_successive_batches");
    let store = PackfileStorage::open(dir).unwrap();

    for batch in 0..8u32 {
        let entries: Vec<_> = (0..20u32)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0..4].copy_from_slice(&batch.to_le_bytes());
                id[4..8].copy_from_slice(&i.to_le_bytes());
                (
                    id,
                    NodeData::new(bytes::Bytes::from(format!("v{batch}-{i}"))),
                )
            })
            .collect();
        store.put_many(&TEST_COLLECTION, &entries).unwrap();
    }

    for batch in 0..8u32 {
        for i in 0..20u32 {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&batch.to_le_bytes());
            id[4..8].copy_from_slice(&i.to_le_bytes());
            let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
            assert_eq!(got.bytes, bytes::Bytes::from(format!("v{batch}-{i}")));
        }
    }
}

#[test]
fn test_runtime_stats_counters_round_trip() {
    let dir = test_dir("runtime_stats_counters");
    let store = PackfileStorage::open(dir).unwrap();

    // Fresh store: every counter starts at zero (open_count counts the
    // one store assembled, not work).
    let snapshot = store.stats();
    assert_eq!(snapshot.open_count, 1);
    assert_eq!(snapshot.put_calls, 0);
    assert_eq!(snapshot.put_many_calls, 0);
    assert_eq!(snapshot.sync_calls, 0);
    assert_eq!(snapshot.get_calls, 0);

    // Read counters are opt-in: with stats disabled a get is invisible.
    store
        .put(
            &TEST_COLLECTION,
            &[0u8; 16],
            &NodeData::new(bytes::Bytes::from("solo")),
        )
        .unwrap();
    assert_eq!(store.stats().put_calls, 1);
    assert!(store.get(&TEST_COLLECTION, &[0u8; 16]).unwrap().is_some());
    assert_eq!(store.stats().get_calls, 0);
    assert_eq!(store.stats().get_misses, 0);

    // Enable read tracking, then a hit and a miss both count.
    store.set_stats_enabled(true);
    assert!(store.get(&TEST_COLLECTION, &[0u8; 16]).unwrap().is_some());
    assert!(store
        .get(&TEST_COLLECTION, &[0xFFu8; 16])
        .unwrap()
        .is_none());
    let snapshot = store.stats();
    assert_eq!(snapshot.get_calls, 2);
    assert_eq!(snapshot.get_misses, 1);
    assert_eq!(snapshot.get_latency.calls, 2);
    assert!(snapshot.get_latency.total > std::time::Duration::ZERO);
    assert!(snapshot.get_latency.max > std::time::Duration::ZERO);
    assert!(snapshot.get_latency.max >= snapshot.get_latency.total / 2);
    assert!(snapshot.get_latency.max <= snapshot.get_latency.total);
    assert_eq!(snapshot.get_latency.buckets.iter().sum::<u64>(), 2);

    // Batched writes against an already-materialized (non-mmap) index
    // mutate it in place and roll back via an undo log on failure,
    // rather than cloning -- so only the very first batch (which
    // creates the collection) pays the clone/materialize cost.
    let batches: Vec<Vec<(NodeId, NodeData)>> = (0..8u32)
        .map(|batch| {
            (0..8u32)
                .map(|i| {
                    let mut id = [2u8; 16];
                    id[0..4].copy_from_slice(&batch.to_le_bytes());
                    id[4..8].copy_from_slice(&i.to_le_bytes());
                    (
                        id,
                        NodeData::new(bytes::Bytes::from(format!("b{batch}-{i}"))),
                    )
                })
                .collect()
        })
        .collect();
    for entries in &batches {
        store.put_many(&SECOND_COLLECTION, entries).unwrap();
    }
    let snapshot = store.stats();
    assert_eq!(snapshot.put_many_calls, 8);
    assert_eq!(snapshot.put_many_records, 64);
    assert!(snapshot.put_many_bytes > 0);
    // Only the first batch creates `SECOND_COLLECTION` (the `None` arm,
    // which always materializes); every later batch finds an
    // already-owned, non-mmap index and takes the in-place fast path.
    assert_eq!(snapshot.put_many_clone_path_calls, 1);
    assert_eq!(snapshot.put_many_fast_path_calls, 7);
    assert_eq!(snapshot.put_many_latency.calls, 8);
    assert_eq!(snapshot.put_many_latency.buckets.iter().sum::<u64>(), 8);
    // 64 distinct records into a floor-sized 64-slot index must cross the
    // grow threshold at least once.
    assert!(snapshot.index_grow_count >= 1);
    assert!(snapshot.index_clone_time > std::time::Duration::ZERO);

    // Every put_many byte is the sum of entry payloads.
    let expected_bytes: u64 = batches
        .iter()
        .flatten()
        .map(|(_, data)| data.bytes.len() as u64)
        .sum();
    assert_eq!(snapshot.put_many_bytes, expected_bytes);

    // reset_stats zeroes the runtime counters but not open_count.
    store.read_refreshes.store(7, Ordering::Relaxed);
    store.read_refresh_bytes.store(123, Ordering::Relaxed);
    store.reset_stats();
    let snapshot = store.stats();
    assert_eq!(snapshot.open_count, 1);
    assert_eq!(snapshot.put_many_calls, 0);
    assert_eq!(snapshot.get_calls, 0);
    assert_eq!(snapshot.get_misses, 0);
    assert_eq!(snapshot.read_refreshes, 0);
    assert_eq!(snapshot.read_refresh_bytes, 0);
    assert_eq!(snapshot.get_latency.calls, 0);
    assert_eq!(snapshot.get_latency.max, std::time::Duration::ZERO);
    assert_eq!(snapshot.put_many_latency.calls, 0);

    // Sync accounting: the first (structurally invalidated) sync rewrites
    // the checkpoint; a later clean batch sync appends the delta log.
    store.sync().unwrap();
    assert_eq!(store.stats().sync_calls, 1);
    assert_eq!(store.stats().checkpoint_writes, 1);
    assert_eq!(store.stats().delta_appends, 0);
    store.put_many(&SECOND_COLLECTION, &batches[0]).unwrap();
    store.sync().unwrap();
    let snapshot = store.stats();
    assert_eq!(snapshot.sync_calls, 2);
    assert_eq!(snapshot.delta_appends, 1);
}

#[test]
fn stats_surfaces_max_index_probe_len_and_dirty_lock_wait() {
    let dir = test_dir("stats_probe_len_and_lock_wait");
    let store = PackfileStorage::open(dir).unwrap();

    // Fresh store, no writes: neither counter has anything to report.
    let snapshot = store.stats();
    assert_eq!(snapshot.max_index_probe_len, 0);
    assert_eq!(snapshot.dirty_lock_wait, std::time::Duration::ZERO);

    for i in 0..20u8 {
        let mut id = [0u8; 16];
        id[0] = i;
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    // A real sync exercises the dirty-set lock at least once; the exact
    // wait duration is unmeasurable deterministically (uncontended in a
    // single-threaded test), but the field must be wired through from
    // `ShardPool::dirty_lock_wait` and never decrease.
    store.sync().unwrap();
    let after_sync = store.stats();
    assert!(after_sync.dirty_lock_wait >= snapshot.dirty_lock_wait);
    // 20 inserts into a real index will very likely walk at least one
    // non-trivial probe chain, but this is observability, not a
    // guarantee -- assert the field is wired through and sane
    // (bounded by the collection's own capacity) rather than a specific
    // value.
    assert!(after_sync.max_index_probe_len < 1000);
}

#[test]
fn test_read_amplification_counters_cover_batch_candidates() {
    let dir = test_dir("read_amplification_counters");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let entries: Vec<_> = (0..4u8)
        .map(|value| {
            let mut id = [0u8; 16];
            id[0] = value;
            (id, NodeData::new(bytes::Bytes::from(vec![value; 8])))
        })
        .collect();
    writer.put_many(&TEST_COLLECTION, &entries).unwrap();
    writer.sync_all().unwrap();
    drop(writer);

    let reader = PackfileStorage::open_read_only(dir).unwrap();
    reader.set_stats_enabled(true);
    let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
    let results = reader.get_many(&TEST_COLLECTION, &ids).unwrap();
    assert_eq!(
        results.iter().filter(|value| value.is_some()).count(),
        ids.len()
    );

    let stats = reader.stats();
    assert_eq!(stats.get_many_calls, 1);
    assert_eq!(stats.get_many_records, ids.len() as u64);
    assert!(stats.index_candidates >= ids.len() as u64);
    assert!(stats.candidate_reads >= ids.len() as u64);
    assert!(stats.get_many_shards_touched >= 1);
}

#[test]
fn test_read_scatter_counters_track_runs_span_and_bytes() {
    let dir = test_dir("read_scatter_counters");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let entries: Vec<_> = (0..8u8)
        .map(|value| {
            let mut id = [0u8; 16];
            id[0] = value;
            id[9] = value.wrapping_mul(37).wrapping_add(11);
            (id, NodeData::new(bytes::Bytes::from(vec![value; 16])))
        })
        .collect();
    writer.put_many(&TEST_COLLECTION, &entries).unwrap();
    writer.sync_all().unwrap();
    drop(writer);

    let reader = PackfileStorage::open_read_only(dir).unwrap();
    reader.set_stats_enabled(true);

    // One batch read of all eight records: the plan covers one shard,
    // its offsets are a single sequential run, and span/bytes record the
    // extent the batch drags in.
    let before = reader.stats();
    let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
    let results = reader.get_many(&TEST_COLLECTION, &ids).unwrap();
    assert_eq!(
        results.iter().filter(|value| value.is_some()).count(),
        entries.len()
    );
    let after = reader.stats();
    assert_eq!(
        after.get_many_shards_touched - before.get_many_shards_touched,
        1
    );
    assert_eq!(after.read_many_runs - before.read_many_runs, 1);
    assert!(after.read_many_span_bytes > before.read_many_span_bytes);
    assert!(after.candidate_frame_bytes > before.candidate_frame_bytes);

    // A single `get` adds candidate frame bytes but is not a batch, so
    // the batch-scatter shape counters (`runs`, `span`) must not move.
    let single_before = reader.stats();
    reader
        .get(&TEST_COLLECTION, &entries[0].0)
        .unwrap()
        .unwrap();
    let single_after = reader.stats();
    assert!(single_after.candidate_frame_bytes > single_before.candidate_frame_bytes);
    assert_eq!(single_after.read_many_runs, single_before.read_many_runs);
    assert_eq!(
        single_after.read_many_span_bytes,
        single_before.read_many_span_bytes
    );
}

#[test]
fn test_plan_read_extents_melds_gaps_and_splits_on_cap() {
    let mut per_shard: HashMap<u16, Vec<u64>> = HashMap::new();

    // Meld runs across gaps within the threshold, split across larger ones.
    per_shard.insert(3, vec![0, 50_000, 400_000, 450_000, 10_000_000]);
    let extents = plan_read_extents(
        &per_shard,
        ReadPlanPolicy {
            merge_gap_bytes: 100_000,
            max_extent_bytes: 10_000_000,
            min_batch_candidates: 1,
            random_advice: false,
        },
    );
    assert_eq!(
        extents,
        vec![
            ReadExtent {
                slot: 3,
                start: 0,
                end: 50_000 + MAX_FRAME_DISK_LEN,
            },
            ReadExtent {
                slot: 3,
                start: 400_000,
                end: 450_000 + MAX_FRAME_DISK_LEN,
            },
            ReadExtent {
                slot: 3,
                start: 10_000_000,
                end: 10_000_000 + MAX_FRAME_DISK_LEN,
            },
        ]
    );

    // A generous gap threshold still splits once the run would exceed the
    // cap: 0 and 100_000 fit under 200_000, but adding 150_000 would not.
    per_shard.clear();
    per_shard.insert(7, vec![0, 100_000, 150_000]);
    let capped = plan_read_extents(
        &per_shard,
        ReadPlanPolicy {
            merge_gap_bytes: 10_000_000,
            max_extent_bytes: 200_000,
            min_batch_candidates: 1,
            random_advice: false,
        },
    );
    assert_eq!(
        capped,
        vec![
            ReadExtent {
                slot: 7,
                start: 0,
                end: 100_000 + MAX_FRAME_DISK_LEN,
            },
            ReadExtent {
                slot: 7,
                start: 150_000,
                end: 150_000 + MAX_FRAME_DISK_LEN,
            },
        ]
    );
}

#[test]
fn test_plan_read_extents_orders_by_slot() {
    let mut per_shard: HashMap<u16, Vec<u64>> = HashMap::new();
    per_shard.insert(9, vec![0, 10]);
    per_shard.insert(2, vec![0, 10]);
    per_shard.insert(5, vec![0, 10]);
    let extents = plan_read_extents(
        &per_shard,
        ReadPlanPolicy {
            merge_gap_bytes: 100,
            max_extent_bytes: 8 * 1024 * 1024,
            min_batch_candidates: 1,
            random_advice: false,
        },
    );
    let slots: Vec<u16> = extents.iter().map(|extent| extent.slot).collect();
    assert_eq!(slots, vec![2, 5, 9]);
}

#[test]
fn test_read_plan_policy_wants_gates_on_batch_size_and_cap() {
    let policy = ReadPlanPolicy {
        merge_gap_bytes: 0,
        max_extent_bytes: 8 * 1024 * 1024,
        min_batch_candidates: 16,
        random_advice: false,
    };
    assert!(!policy.wants(15));
    assert!(policy.wants(16));
    assert!(!ReadPlanPolicy::disabled().wants(1_000_000));
}

#[test]
fn test_random_advice_preset_plans_nothing() {
    // The random-advice preset suppresses readahead but builds no
    // extents: it must never satisfy `wants`, whatever the batch size.
    let policy = ReadPlanPolicy::random_advice();
    assert!(policy.random_advice);
    assert!(!policy.wants(1_000_000));
    assert!(!ReadPlanPolicy::disabled().random_advice);
    assert!(!ReadPlanPolicy::prefetch().random_advice);
    assert!(ReadPlanPolicy::prefetch().wants(16));
}

#[test]
#[cfg(unix)]
fn test_get_many_prefetch_plan_counters() {
    let dir = test_dir("read_plan_counters");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let entries: Vec<_> = (0..8u8)
        .map(|value| {
            let mut id = [0u8; 16];
            id[0] = value;
            id[9] = value.wrapping_mul(37).wrapping_add(11);
            (id, NodeData::new(bytes::Bytes::from(vec![value; 16])))
        })
        .collect();
    writer.put_many(&TEST_COLLECTION, &entries).unwrap();
    writer.sync_all().unwrap();
    drop(writer);

    let reader = PackfileStorage::open_read_only(dir).unwrap();
    reader.set_stats_enabled(true);
    reader.set_read_plan_policy(ReadPlanPolicy {
        merge_gap_bytes: 1024 * 1024,
        max_extent_bytes: 8 * 1024 * 1024,
        min_batch_candidates: 1,
        random_advice: false,
    });

    let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
    let before = reader.stats();
    let results = reader.get_many(&TEST_COLLECTION, &ids).unwrap();
    assert_eq!(
        results.iter().filter(|value| value.is_some()).count(),
        entries.len()
    );
    let after = reader.stats();
    assert!(after.read_plan_extents > before.read_plan_extents);
    assert!(after.read_plan_prefetch_bytes > before.read_plan_prefetch_bytes);
    // Every candidate is committed to disk, so every planned extent is
    // prefetchable and none is skipped.
    assert_eq!(after.read_plan_skipped_extents, 0);

    // A disabled policy prefetches nothing, even for the same batch.
    reader.set_read_plan_policy(ReadPlanPolicy::disabled());
    let disabled_before = reader.stats();
    reader.get_many(&TEST_COLLECTION, &ids).unwrap();
    let disabled_after = reader.stats();
    assert_eq!(
        disabled_after.read_plan_extents,
        disabled_before.read_plan_extents
    );
    assert_eq!(
        disabled_after.read_plan_prefetch_bytes,
        disabled_before.read_plan_prefetch_bytes
    );
}

#[test]
fn test_walk_ancestors_counts_candidate_probes_reads_and_frame_bytes() {
    let dir = test_dir("walk_counter_tracking");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let ids: Vec<NodeId> = (0..3u8).map(distinct_id).collect();
    for (i, id) in ids.iter().enumerate() {
        let byte = b'A' + u8::try_from(i).unwrap();
        writer
            .put(
                &TEST_COLLECTION,
                id,
                &NodeData::new(bytes::Bytes::from(vec![byte])),
            )
            .unwrap();
    }
    writer.sync_all().unwrap();
    drop(writer);

    // A fresh read-only store has an empty decoded-node cache, so a walk
    // must resolve every hash through the lossy index and the packfiles —
    // and the counters for that traverse path (`resolve_pinned`, the
    // ancestor/frontier hook) must move, not just `get`/`get_many`.
    let reader = PackfileStorage::open_read_only(dir).unwrap();
    reader.set_stats_enabled(true);
    let edges = std::collections::HashMap::from([(ids[1], vec![ids[0]]), (ids[2], vec![ids[1]])]);
    let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();
    let walked: Vec<(NodeId, NodeData)> = reader
        .walk_ancestors(
            &TEST_COLLECTION,
            &[ids[2]],
            &[],
            extract,
            WalkLimits::default(),
        )
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(walked.len(), 3);

    let stats = reader.stats();
    assert!(stats.index_candidates >= 3);
    assert!(stats.candidate_reads >= 3);
    assert!(stats.candidate_frame_bytes >= stats.candidate_reads);
    assert!(stats.cache.misses >= 3);
    // `walk_ancestors` is a traversal, not a `get_many` batch: its
    // scatter-shape counters must not be the ones that moved.
    assert_eq!(stats.read_many_runs, 0);
}

#[test]
fn test_put_many_growth_mid_batch_keeps_records_inserted_before_the_grow() {
    // Exercises the fast path's trickiest transition: some records in
    // the batch insert via the live, shared index (no clone), then a
    // later record in the *same* batch forces a grow — which must
    // materialize an owned index that still contains everything the
    // live path already applied, not just what comes after the grow.
    let dir = test_dir("put_many_growth_mid_batch");
    let store = PackfileStorage::open(dir).unwrap();

    // Get the collection's live index past its NEW_COLLECTION_INDEX_FLOOR
    // starting capacity via ordinary put_many calls (fast path), then
    // issue one big batch that must cross the 75%-load grow threshold
    // partway through.
    let seed: Vec<_> = (0..40u32)
        .map(|i| {
            let mut id = [1u8; 16];
            id[4..8].copy_from_slice(&i.to_le_bytes());
            (id, NodeData::new(bytes::Bytes::from(format!("seed{i}"))))
        })
        .collect();
    store.put_many(&TEST_COLLECTION, &seed).unwrap();

    let growth_batch: Vec<_> = (0..200u32)
        .map(|i| {
            let mut id = [2u8; 16];
            id[4..8].copy_from_slice(&i.to_le_bytes());
            (id, NodeData::new(bytes::Bytes::from(format!("grow{i}"))))
        })
        .collect();
    store.put_many(&TEST_COLLECTION, &growth_batch).unwrap();

    for (id, data) in seed.iter().chain(&growth_batch) {
        let got = store.get(&TEST_COLLECTION, id).unwrap().unwrap();
        assert_eq!(
            &got.bytes, &data.bytes,
            "record must survive the mid-batch grow"
        );
    }
}

#[test]
fn test_index_offset_gate_rejects_unrepresentable_offsets() {
    let hash = [0x5A; 16];
    assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT).is_ok());
    assert!(check_index_offset(0, &hash, (1u64 << 28) + 1).is_ok());
    assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT + 1).is_err());
    assert!(check_index_offset(0, &hash, u64::MAX).is_err());
}

#[test]
fn test_get_many_with_refresh_coalesces_negative_lookups() {
    let dir = test_dir("get_many_with_refresh_negative");
    let store = PackfileStorage::open(dir).unwrap();
    // Use a collection that was NOT pre-seeded (not in collection_order)
    // so the first miss has no stored fingerprint → refresh required.
    let missing = [[0xF0u8; 16], [0xF1u8; 16]];

    assert!(store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap()
        .iter()
        .all(Option::is_none));
    assert!(store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap()
        .iter()
        .all(Option::is_none));

    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert_eq!(stats.miss_refresh_recovered, 0);
    assert_eq!(stats.miss_refresh_skips, 2);
    assert_eq!(stats.miss_refresh_retry_ids, 0);
}

#[test]
fn test_get_many_with_refresh_disabled_is_authoritative() {
    let dir = test_dir("refresh_disabled_authoritative");
    let store = Arc::new(PackfileStorage::open(dir).unwrap());
    // A single writer's in-memory index is authoritative for every key it
    // has written, so a negative lookup must bypass the refresh lock and
    // the durable-fingerprint probe entirely rather than paying either.
    store.set_refresh_on_miss(false);

    let missing = [[0xF0u8; 16], [0xF1u8; 16]];
    let refresh_lock = store.refresh_lock(&TEST_COLLECTION);
    let refresh_guard = refresh_lock.lock();
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader_store = Arc::clone(&store);
    let reader = std::thread::spawn(move || {
        let result = reader_store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap()
            .iter()
            .all(Option::is_none);
        sender.send(result).unwrap();
    });

    // Keep the refresh lock held while the lookup runs. The writer path
    // must return from the in-memory index without waiting for it. This is
    // a deadlock guard, not a latency assertion: the timeout is generous
    // enough that a merely loaded CI worker cannot trip it. The
    // timing-independent proof that no durable fingerprint was probed is
    // the `miss_refreshes == 0` counter below.
    assert!(receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap());
    drop(refresh_guard);
    reader.join().unwrap();

    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert_eq!(stats.miss_refresh_skips, 0);
    assert_eq!(stats.miss_refresh_retry_ids, 0);
}

#[test]
fn test_get_many_with_refresh_flag_gates_the_reader_path() {
    let dir = test_dir("refresh_flag_gates");
    let store = PackfileStorage::open(dir).unwrap();
    let missing = [[0xF2u8; 16]];

    store.set_refresh_on_miss(false);
    let _ = store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap();
    assert_eq!(store.stats().miss_refresh_skips, 0);

    // Re-enabling restores the multi-process-reader behavior: the miss is
    // rate-limited by the durable fingerprint rather than bypassed.
    store.set_refresh_on_miss(true);
    let _ = store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap();
    assert_eq!(store.stats().miss_refresh_skips, 1);
}

#[test]
fn test_get_many_with_refresh_all_hit_skips_refresh() {
    let dir = test_dir("refresh_all_hit");
    let store = PackfileStorage::open(dir).unwrap();

    let id_a = distinct_id(0xA0);
    let id_b = distinct_id(0xA1);
    let data = NodeData::new(bytes::Bytes::from_static(b"payload"));
    store.put(&TEST_COLLECTION, &id_a, &data).unwrap();
    store.put(&TEST_COLLECTION, &id_b, &data).unwrap();
    store.sync().unwrap();

    store.reset_stats();
    let result = store
        .get_many_with_refresh(&TEST_COLLECTION, &[id_a, id_b])
        .unwrap();
    assert!(result.iter().all(Option::is_some));

    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert_eq!(stats.miss_refresh_skips, 0);
}

#[test]
fn test_get_many_with_refresh_mixed_batch_retries_only_misses() {
    let dir = test_dir("refresh_mixed_batch");
    let store = PackfileStorage::open(dir).unwrap();

    let present = distinct_id(0xB0);
    let missing = distinct_id(0xBF);
    let data = NodeData::new(bytes::Bytes::from_static(b"here"));
    store.put(&TEST_COLLECTION, &present, &data).unwrap();
    store.sync().unwrap();

    store.reset_stats();
    let result = store
        .get_many_with_refresh(&TEST_COLLECTION, &[present, missing])
        .unwrap();
    assert!(result[0].is_some());
    assert!(result[1].is_none());

    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 1);
    assert_eq!(stats.miss_refresh_recovered, 0);
    // Exactly the 1 missing ID was retried.
    assert_eq!(stats.miss_refresh_retry_ids, 1);
}

#[test]
fn test_get_many_with_refresh_unsynced_append_invisible() {
    let dir = test_dir("refresh_unsynced");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

    let id = distinct_id(0xC0);
    let data = NodeData::new(bytes::Bytes::from_static(b"unsynced"));

    // Writer appends but does NOT sync.
    writer.put(&TEST_COLLECTION, &id, &data).unwrap();

    // Reader sees durable fingerprint unchanged (writer didn't sync)
    // → dominated → confirmed negative, no refresh.
    reader.reset_stats();
    let result = reader
        .get_many_with_refresh(&TEST_COLLECTION, &[id])
        .unwrap();
    assert!(result[0].is_none(), "unsynced append must remain invisible");

    let stats = reader.stats();
    assert_eq!(
        stats.miss_refreshes, 0,
        "no refresh should occur for unsynced data"
    );
    assert_eq!(stats.miss_refresh_skips, 1);
}

#[test]
fn test_get_many_with_refresh_synced_append_detected() {
    let dir = test_dir("refresh_synced_append");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

    let id = distinct_id(0xD0);
    let data = NodeData::new(bytes::Bytes::from_static(b"synced"));

    // Writer appends and syncs → durable fingerprint changes.
    writer.put(&TEST_COLLECTION, &id, &data).unwrap();
    writer.sync().unwrap();

    // Existing reader: stored fp != durable fp → refresh → recovered.
    reader.reset_stats();
    let result = reader
        .get_many_with_refresh(&TEST_COLLECTION, &[id])
        .unwrap();
    assert_eq!(
        result[0].as_ref().map(|d| &d.bytes),
        Some(&bytes::Bytes::from_static(b"synced")),
        "synced append must be recovered by the existing reader handle"
    );

    let stats = reader.stats();
    assert_eq!(stats.miss_refreshes, 1);
    assert_eq!(stats.miss_refresh_recovered, 1);
    assert_eq!(stats.miss_refresh_retry_ids, 1);
}

/// Regression for the deferred-checkpoint reader window.
///
/// A writer that owes a *structurally needed* full checkpoint rewrite
/// (`pending` is empty while the index is dirty — the shape a
/// `refresh_collection` that pulled in another process's appends leaves
/// behind) will skip that rewrite under
/// [`PackfileStorage::set_checkpoint_rewrite_budget`]. The packs are still
/// flushed and fsynced, so the record is durable; but with no checkpoint
/// rewrite and no delta frame, the durable fingerprint does not advance.
/// A reader handle that opened at the previous fingerprint therefore treats
/// its negative as confirmed and never refreshes: a *synced* record stays
/// invisible to [`PackfileStorage::get_many_with_refresh`] for as long as
/// the budget defers the rewrite.
///
/// Ignored because WAL-off cross-process reads are no longer a supported
/// shape. The journal's [`PackfileStorage::get_read_committed`] overlay is
/// the authoritative cross-process visibility path and serves the committed
/// record before the checkpoint advances, independent of the rewrite
/// budget. This pins the unsupported window so a future change that makes
/// WAL-off reads appear to work by accident (rather than by the overlay) is
/// noticed.
///
/// Known issue, not exercised here: `post_refresh_fingerprint` records the
/// *post*-refresh durable fingerprint as the collection's refresh baseline,
/// so a sync racing between `refresh_collection` and that read can store a
/// baseline the index never actually incorporated — over-claiming coverage
/// and suppressing a refresh that was needed. A fix must capture the
/// fingerprint the refresh itself observed rather than re-reading it.
#[test]
#[ignore = "WAL-off cross-process reads are unsupported; the journal overlay is authoritative (see test docs)"]
fn deferred_checkpoint_rewrite_leaves_synced_write_invisible() {
    let dir = test_dir("deferred_checkpoint_reader_window");
    let writer = PackfileStorage::open(dir.clone()).unwrap();

    // Baseline checkpoint, so the writer has a prior rewrite timestamp for
    // the budget's time headroom to measure against.
    let seed = distinct_id(0xC0);
    writer
        .put(
            &TEST_COLLECTION,
            &seed,
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    writer.sync().unwrap();

    // Reader opens bound to the baseline checkpoint's fingerprint.
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

    // Once a rewrite has run, defer the next for an hour; no size budget.
    writer.set_checkpoint_rewrite_budget(Duration::from_secs(3600), 0);

    let id = distinct_id(0xC1);
    writer
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"deferred")),
        )
        .unwrap();
    // The record is eagerly in the pack, but leave no local delta frame:
    // this is the `pending.is_empty()` + dirty shape that makes the next
    // sync structurally need a full rewrite (as a refresh of external
    // appends does).
    writer.delta_state.lock().pending.clear();
    writer.index_checkpoint_dirty.store(true, Ordering::Relaxed);

    let before = writer.stats();
    writer.sync().unwrap();
    let after = writer.stats();
    assert_eq!(
        after.checkpoint_writes - before.checkpoint_writes,
        0,
        "the structurally-needed rewrite must be deferred, not written"
    );
    assert_eq!(
        after.checkpoint_skips - before.checkpoint_skips,
        1,
        "the sync must have taken the deferred branch"
    );

    // The pack bytes are durable, but the durable fingerprint is unchanged,
    // so the reader's gate confirms the negative without refreshing.
    reader.reset_stats();
    let result = reader
        .get_many_with_refresh(&TEST_COLLECTION, &[id])
        .unwrap();
    assert!(
        result[0].is_none(),
        "a synced record stays invisible while the checkpoint rewrite is deferred"
    );
    let reader_stats = reader.stats();
    assert_eq!(
        reader_stats.miss_refreshes, 0,
        "an unchanged durable fingerprint suppresses the refresh entirely"
    );
    assert_eq!(reader_stats.miss_refresh_skips, 1);
}

#[test]
fn test_get_many_with_refresh_repeated_negative_suppressed() {
    let dir = test_dir("refresh_repeated_negative");
    let store = PackfileStorage::open(dir).unwrap();

    let missing = [distinct_id(0xE0), distinct_id(0xE1)];

    // First call: no stored fingerprint matches durable fp (both 0),
    // so dominated = true → skip refresh.
    store.reset_stats();
    let _ = store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap();
    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert_eq!(stats.miss_refresh_skips, 1);

    // Second call: same durable fingerprint → still dominated.
    let _ = store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap();
    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert!(stats.miss_refresh_skips >= 2);
}

#[test]
fn test_get_many_with_refresh_preserves_result_order() {
    let dir = test_dir("refresh_ordering");
    let store = PackfileStorage::open(dir).unwrap();

    let present_a = distinct_id(0xF0);
    let present_b = distinct_id(0xF1);
    let missing = distinct_id(0xFF);
    let data = NodeData::new(bytes::Bytes::from_static(b"data"));

    store.put(&TEST_COLLECTION, &present_a, &data).unwrap();
    store.put(&TEST_COLLECTION, &present_b, &data).unwrap();
    store.sync().unwrap();

    let batch = [missing, present_a, present_b, missing];
    let result = store
        .get_many_with_refresh(&TEST_COLLECTION, &batch)
        .unwrap();

    assert!(result[0].is_none(), "first missing stays at index 0");
    assert!(result[1].is_some(), "present_a stays at index 1");
    assert!(result[2].is_some(), "present_b stays at index 2");
    assert!(result[3].is_none(), "second missing stays at index 3");

    let stats = store.stats();
    assert_eq!(stats.miss_refreshes, 1);
    // Exactly 2 missing IDs were retried.
    assert_eq!(stats.miss_refresh_retry_ids, 2);
}

#[test]
fn test_get_many_with_refresh_concurrent_misses_coalesce() {
    use std::sync::Arc;

    let dir = test_dir("refresh_concurrent");
    let store = Arc::new(PackfileStorage::open(dir).unwrap());

    // Seed a record so TEST_COLLECTION exists and has a shard on disk.
    let seed_id = distinct_id(0xC0);
    store
        .put(
            &TEST_COLLECTION,
            &seed_id,
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    store.sync().unwrap();

    let missing = [distinct_id(0xC1), distinct_id(0xC2)];

    // Pre-seed: first call populates last_refresh_fingerprint for
    // TEST_COLLECTION. Durable fp is the same as stored → skip.
    store.reset_stats();
    let _ = store
        .get_many_with_refresh(&TEST_COLLECTION, &missing)
        .unwrap();

    // Append and sync to change the durable fingerprint.
    let id = distinct_id(0xC3);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"synced")),
        )
        .unwrap();
    store.sync().unwrap();

    // Stored fp is stale → refresh needed. Spawn concurrent readers.
    store.reset_stats();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            let _ = store.get_many_with_refresh(&TEST_COLLECTION, &missing);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Only one refresh should have actually occurred (coalesced).
    let stats = store.stats();
    assert_eq!(
        stats.miss_refreshes, 1,
        "concurrent misses must coalesce into one refresh"
    );
}

#[test]
fn test_get_many_with_refresh_negative_then_recovered() {
    let dir = test_dir("refresh_negative_then_recovered");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

    let missing_id = distinct_id(0xD1);

    // First call: reader sees no durable change → confirmed negative.
    reader.reset_stats();
    let result = reader
        .get_many_with_refresh(&TEST_COLLECTION, &[missing_id])
        .unwrap();
    assert!(result[0].is_none());
    let stats = reader.stats();
    assert_eq!(stats.miss_refreshes, 0);
    assert_eq!(stats.miss_refresh_skips, 1);

    // Writer appends and syncs → durable fingerprint changes.
    let data = NodeData::new(bytes::Bytes::from_static(b"now_here"));
    writer.put(&TEST_COLLECTION, &missing_id, &data).unwrap();
    writer.sync().unwrap();

    // Second call: stored fp != durable fp → refresh → recovered.
    let result = reader
        .get_many_with_refresh(&TEST_COLLECTION, &[missing_id])
        .unwrap();
    assert_eq!(
        result[0].as_ref().map(|d| &d.bytes),
        Some(&bytes::Bytes::from_static(b"now_here")),
        "record written after first negative must be recovered"
    );
    let stats = reader.stats();
    assert_eq!(stats.miss_refreshes, 1);
    assert_eq!(stats.miss_refresh_recovered, 1);
    assert_eq!(stats.miss_refresh_retry_ids, 1);
}

#[test]
fn test_refresh_collection_multi_worker_visibility() {
    let dir = test_dir("refresh_collection_multi_worker");
    let writer = PackfileStorage::open(dir.clone()).unwrap();

    // Open a read-only instance BEFORE the writer writes the data.
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

    let id = [0xAA; 16];
    writer
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"multi_worker_test")),
        )
        .unwrap();
    writer.sync().unwrap();

    // Reader's in-memory index should be completely unaware of the new write.
    assert!(reader.get(&TEST_COLLECTION, &id).unwrap().is_none());

    // Trigger the external refresh, simulating a cache invalidation signal.
    reader.refresh_collection(&TEST_COLLECTION).unwrap();

    // Reader should now have correctly loaded the delta and rebuilt its index.
    let got = reader.get(&TEST_COLLECTION, &id).unwrap().unwrap();
    assert_eq!(got.bytes, bytes::Bytes::from_static(b"multi_worker_test"));
    assert_eq!(
        reader
            .collection_index_info(&TEST_COLLECTION)
            .expect("refreshed collection exists")
            .2,
        u32::try_from(NEW_COLLECTION_INDEX_FLOOR).unwrap(),
        "a refresh rebuild keeps the same minimum capacity as a new collection"
    );
}

#[test]
fn test_scan_existing_skips_malformed_filenames() {
    let dir = test_dir("scan_existing_junk");
    std::fs::write(dir.join("nounderscore.pack"), b"").unwrap();
    std::fs::write(dir.join("aabb_00.pack"), b"").unwrap();
    std::fs::write(dir.join("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_00.pack"), b"").unwrap();
    std::fs::write(dir.join("00000000000000000000000000000000_gg.pack"), b"").unwrap();
    // A filename that isn't valid UTF-8 is only creatable on filesystems
    // that accept arbitrary bytes (Linux/BSD). APFS on macOS rejects it
    // with EILSEQ (os error 92), and Windows has no `std::os::unix`; the
    // ASCII-only malformed names above still cover the skip path there.
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let path = dir.join(OsStr::from_bytes(b"\xff\xfe.pack"));
        std::fs::write(path, b"").unwrap();
    }

    let mut valid_bytes = [0u8; crate::packfile::PACK_ID_LEN];
    valid_bytes[crate::packfile::PACK_ID_LEN - 1] = 1;
    let pack_id = PackId(valid_bytes);
    let valid_path = dir.join(pack_id.filename());
    let mut buf = Vec::new();
    packfile::write_header(&mut buf, &pack_id).unwrap();
    packfile::write_record(
        &mut buf,
        &packfile::Record {
            collection_id: [0x01; 16],
            hash: [0xAA; 16],
            data: bytes::Bytes::from_static(b"hello"),
            metadata: None,
        },
    )
    .unwrap();
    std::fs::write(&valid_path, &buf).unwrap();

    let store = PackfileStorage::open(dir).unwrap();
    let gen = store.generation(&[0x01u8; 16]).unwrap();
    assert_eq!(gen.index.len(), 1);
}

#[test]
fn test_delete_collection_preserves_other_collection_cache() {
    let dir = test_dir("delete_collection_cache");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let collection_a = TEST_COLLECTION;
    let collection_b = OTHER_COLLECTION;
    let id_a = [0x10u8; 16];
    let id_b = [0x20u8; 16];
    let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
    let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

    store.put(&collection_a, &id_a, &data_a).unwrap();
    store.put(&collection_b, &id_b, &data_b).unwrap();

    assert!(store.get(&collection_a, &id_a).unwrap().is_some());
    assert!(store.get(&collection_b, &id_b).unwrap().is_some());

    let gen_b_before = store.generation(&collection_b).unwrap();
    let hits_b_before = gen_b_before.cache.hits();

    store.delete_collection(&collection_a).unwrap();

    assert!(store.get(&collection_a, &id_a).unwrap().is_none());
    assert!(store.generation(&collection_a).is_none());
    assert!(store.generation(&collection_b).is_some());
    assert!(store.get(&collection_b, &id_b).unwrap().is_some());

    let gen_b_after = store.generation(&collection_b).unwrap();
    assert!(gen_b_after.cache.hits() > hits_b_before);
}

#[test]
fn test_repack_collection_reachable_no_roots_preserves_diamond_dag_in_topo_order() {
    let dir = test_dir("repack_topo");
    let store = PackfileStorage::open(dir).unwrap();

    let mut id_a = [0u8; 16];
    id_a[0] = 1;
    let mut id_b = [0u8; 16];
    id_b[0] = 2;
    let mut id_c = [0u8; 16];
    id_c[0] = 3;
    let mut id_d = [0u8; 16];
    id_d[0] = 4;

    store
        .put(
            &TEST_COLLECTION,
            &id_a,
            &NodeData::new(bytes::Bytes::from_static(b"A")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &id_b,
            &NodeData::new(bytes::Bytes::from_static(b"B")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &id_c,
            &NodeData::new(bytes::Bytes::from_static(b"C")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &id_d,
            &NodeData::new(bytes::Bytes::from_static(b"D")),
        )
        .unwrap();

    let edges = std::collections::HashMap::from([
        (id_b, vec![id_a]),
        (id_c, vec![id_a]),
        (id_d, vec![id_b, id_c]),
    ]);

    // No live roots configured for TEST_COLLECTION: preserves everything,
    // still deduplicated and topologically ordered.
    let result = store.repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
        edges.get(hash).cloned().unwrap_or_default()
    });
    let (kept, dropped) = result.unwrap();
    assert_eq!(kept, 4);
    assert_eq!(dropped, 0);

    for (id, expected) in [
        (id_a, b"A".as_slice()),
        (id_b, b"B".as_slice()),
        (id_c, b"C".as_slice()),
        (id_d, b"D".as_slice()),
    ] {
        let got = store
            .get(&TEST_COLLECTION, &id)
            .unwrap()
            .expect("record missing after topo repack");
        assert_eq!(got.bytes.as_ref(), expected);
    }
}

#[test]
fn test_repack_collection_reachable_drops_unreachable_records() {
    let dir = test_dir("repack_reachable_gc");
    let store = PackfileStorage::open(dir).unwrap();

    // Live chain: root -> p1 -> p0 (p0 has no further dependencies).
    let root = distinct_id(1);
    let p1 = distinct_id(2);
    let p0 = distinct_id(3);

    // Garbage island, unreachable from root: garbage -> garbage_dep.
    let garbage = distinct_id(4);
    let garbage_dep = distinct_id(5);

    for (id, bytes) in [
        (root, b"root".as_slice()),
        (p1, b"p1".as_slice()),
        (p0, b"p0".as_slice()),
        (garbage, b"garbage".as_slice()),
        (garbage_dep, b"garbage_dep".as_slice()),
    ] {
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(bytes)),
            )
            .unwrap();
    }

    let edges = std::collections::HashMap::from([
        (root, vec![p1]),
        (p1, vec![p0]),
        (garbage, vec![garbage_dep]),
    ]);

    store.set_live_roots(&TEST_COLLECTION, vec![root]);

    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        })
        .unwrap();

    assert_eq!(kept, 3, "root, p1, p0 should survive");
    assert_eq!(dropped, 2, "garbage and garbage_dep should be collected");

    for id in [root, p1, p0] {
        assert!(
            store.get(&TEST_COLLECTION, &id).unwrap().is_some(),
            "live record missing after reachable repack"
        );
    }
    for id in [garbage, garbage_dep] {
        assert!(
            store.get(&TEST_COLLECTION, &id).unwrap().is_none(),
            "garbage record survived reachable repack"
        );
    }

    let stats = store.repack_stats();
    assert_eq!(stats.repack_count, 1);
    assert_eq!(stats.kept_total, 3);
    assert_eq!(stats.dropped_total, 2);
    assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 1);

    // A second repack uses the incremental path: it clones the
    // previous live_map (3 entries from the first repack's output)
    // and scans only newly-appended bytes (none here).  The garbage
    // entries in the old shard are never re-scanned — the live_map
    // already excludes them, so dropped is 0.  This is the O(n)
    // behavior: each repack does bounded work proportional to new
    // bytes, not proportional to total shard size.
    let (kept2, dropped2) = store
        .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        })
        .unwrap();
    assert_eq!((kept2, dropped2), (3, 0));

    let stats2 = store.repack_stats();
    assert_eq!(stats2.repack_count, 2);
    assert_eq!(stats2.kept_total, stats.kept_total + kept2 as u64);
    assert_eq!(stats2.dropped_total, stats.dropped_total + dropped2 as u64);
    assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 2);
}

/// Verify the root-removal fallback: if a live root is removed between
/// repacks, nodes exclusively reachable through it must not stay in the
/// cached live set indefinitely. Without the fallback, the incremental
/// adjacency cache only grows and those nodes would never be collected.
#[test]
fn test_repack_incremental_root_removal_forces_full_sweep() {
    let dir = test_dir("repack_root_removal_fallback");
    let store = PackfileStorage::open(dir).unwrap();

    // Graph:
    //   root_a -> shared -> base
    //   root_b -> shared -> base
    //   orphan_a -> orphan_dep    (reachable only via root_a)
    //
    // First repack: both root_a and root_b are live. Everything survives.
    // Second repack: only root_b is live. orphan_a and orphan_dep must be
    // collected. Without the root-removal fallback they would remain in the
    // incremental live set permanently.
    let root_a = distinct_id(1);
    let root_b = distinct_id(2);
    let shared = distinct_id(3);
    let base = distinct_id(4);
    let orphan_a = distinct_id(5);
    let orphan_dep = distinct_id(6);

    for (id, bytes) in [
        (root_a, b"root_a".as_slice()),
        (root_b, b"root_b".as_slice()),
        (shared, b"shared".as_slice()),
        (base, b"base".as_slice()),
        (orphan_a, b"orphan_a".as_slice()),
        (orphan_dep, b"orphan_dep".as_slice()),
    ] {
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(bytes)),
            )
            .unwrap();
    }

    let edges = std::collections::HashMap::from([
        (root_a, vec![shared, orphan_a]),
        (root_b, vec![shared]),
        (shared, vec![base]),
        (orphan_a, vec![orphan_dep]),
    ]);
    let extract = |hash: &[u8; 16], _data: &[u8]| -> Vec<[u8; 16]> {
        edges.get(hash).cloned().unwrap_or_default()
    };

    // First repack: both roots live — everything reachable from either root
    // survives and the incremental adjacency cache is populated.
    store.set_live_roots(&TEST_COLLECTION, vec![root_a, root_b]);
    let (kept1, dropped1) = store
        .repack_collection_reachable(&TEST_COLLECTION, extract)
        .unwrap();
    assert_eq!(kept1, 6, "all nodes should survive with both roots live");
    assert_eq!(dropped1, 0);

    // Now remove root_a — switch to root_b only.
    // orphan_a and orphan_dep are now exclusively reachable through root_a,
    // which was removed. The root-removal fallback must trigger a full BFS
    // sweep so they are actually collected, rather than remaining in the
    // stale incremental live set forever.
    store.set_live_roots(&TEST_COLLECTION, vec![root_b]);
    let (kept2, dropped2) = store
        .repack_collection_reachable(&TEST_COLLECTION, extract)
        .unwrap();
    assert_eq!(
        kept2, 3,
        "root_b, shared, base should survive; root_a, orphan_a, orphan_dep must be GC'd"
    );
    assert_eq!(
        dropped2, 3,
        "root_a, orphan_a, and orphan_dep must be collected once root_a is removed from live roots"
    );

    assert!(
        store.get(&TEST_COLLECTION, &orphan_a).unwrap().is_none(),
        "orphan_a must be gone after root_a was removed"
    );
    assert!(
        store.get(&TEST_COLLECTION, &orphan_dep).unwrap().is_none(),
        "orphan_dep must be gone after root_a was removed"
    );
    assert!(
        store.get(&TEST_COLLECTION, &root_a).unwrap().is_none(),
        "root_a itself has no incoming edges so it is unreachable from root_b"
    );
    assert!(
        store.get(&TEST_COLLECTION, &root_b).unwrap().is_some(),
        "root_b must still be present"
    );
    assert!(
        store.get(&TEST_COLLECTION, &shared).unwrap().is_some(),
        "shared must still be present"
    );
    assert!(
        store.get(&TEST_COLLECTION, &base).unwrap().is_some(),
        "base must still be present"
    );
}

/// Repack drops unreachable *records* from the index (`dropped` count),
/// but today nothing ever retires the *shard file* they lived in: no
/// code path sets `Shard::is_current` to `false` or clears the pool's
/// slot for it (see `Shard::drop`'s doc and
/// `test_drop_deletes_retired_shard_only_after_last_reference` in
/// shard.rs, which has to simulate retirement manually because nothing
/// production triggers it). `rotate()`'s error message says "repack to
/// reclaim", but repack currently reclaims nothing at the shard level.
///
/// This fills a small fixed number of pool slots (4), repacks away
/// everything in the non-active ones (100% garbage, nothing live left),
/// and then expects one more rotation to succeed by reusing a
/// now-empty slot.
#[test]
fn test_repack_reclaims_shard_slots_for_rotation() {
    // Use a fixed small count rather than MAX_SHARDS — with 4096 slots
    // and 256MB each that would be a 1TB test.
    const TEST_SHARDS: usize = 4;

    let dir = test_dir("repack_reclaims_slots");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let mut root = [0u8; 16];
    root[0] = 0xFF;
    store
        .put(
            &TEST_COLLECTION,
            &root,
            &NodeData::new(bytes::Bytes::from_static(b"root")),
        )
        .unwrap();

    // `root` already occupies the first slot; force rotation through
    // the remaining TEST_SHARDS - 1 slots, dumping garbage into each so
    // every shard but the last ends up fully unreachable once we
    // repack with `root` as the only live node.
    for i in 0..TEST_SHARDS - 1 {
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = u8::try_from(i + 1).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();
    }

    // Every slot should now be occupied.
    let occupied = (0..u16::try_from(TEST_SHARDS).unwrap())
        .filter(|&id| store.shards.get_shard(id).is_some())
        .count();
    assert_eq!(
        occupied, TEST_SHARDS,
        "test setup should have filled every shard slot"
    );

    store.set_live_roots(&TEST_COLLECTION, vec![root]);
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    assert_eq!(kept, 1, "only root should survive");
    assert!(dropped > 0, "the garbage records should have been dropped");
    assert!(
        store.shards_retired() > 0,
        "retire_empty_shards should have retired at least one now-garbage-only shard"
    );

    // The collection occupied several packs, and repack retired the
    // garbage-only source packs. The persisted byte total must follow the
    // surviving physical layout rather than retaining bytes for retired
    // packs.
    store.sync_all().unwrap();
    let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
    let expected = physical
        .collections
        .get(&TEST_COLLECTION)
        .expect("repacked collection remains on disk")
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "retired packs must not remain in collection byte accounting"
    );

    // Every shard except the current active one held nothing but
    // garbage, and that garbage is now unreachable -- those slots
    // should be reclaimable. Forcing one more rotation should reuse
    // one of them rather than failing.
    store.shards.active_shard().file_len.store(
        shard::MAX_SHARD_BYTES - 10,
        std::sync::atomic::Ordering::Release,
    );
    let mut one_more = [0u8; 16];
    one_more[0] = 0xEE;
    store
        .put(
            &TEST_COLLECTION,
            &one_more,
            &NodeData::new(bytes::Bytes::from_static(b"one more")),
        )
        .expect(
            "rotation should reclaim a garbage-only shard slot after repack, \
             not report the pool as permanently full",
        );
}

#[test]
fn test_repack_moves_live_data_out_of_its_source_shard() {
    let dir = test_dir("repack_moves_from_source");
    let store = PackfileStorage::open(dir).unwrap();
    let node = [0xA5; 16];
    store
        .put(
            &TEST_COLLECTION,
            &node,
            &NodeData::new(bytes::Bytes::from_static(b"live")),
        )
        .unwrap();

    let source = store
        .collection_referenced_shards(&TEST_COLLECTION)
        .into_iter()
        .next()
        .unwrap();
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();

    assert_eq!((kept, dropped), (1, 0));
    let destination = store
        .collection_referenced_shards(&TEST_COLLECTION)
        .into_iter()
        .next()
        .unwrap();
    assert_ne!(destination, source);
    assert!(store.shards.get_shard(source).is_none());
    assert_eq!(
        store
            .get(&TEST_COLLECTION, &node)
            .unwrap()
            .unwrap()
            .bytes
            .as_ref(),
        b"live"
    );
}

/// Repack must not retire a shard that another collection's index still
/// references.  This test puts live records from two different collections
/// into the same shard, then repacks only collection A — the shared shard
/// must survive because collection B's index still points into it.
#[test]
fn test_repack_does_not_retire_shard_referenced_by_other_collection() {
    let dir = test_dir("repack_cross_collection_safety");
    let store = PackfileStorage::open(dir).unwrap();

    // Put a record for collection A — lands on shard 0.
    let mut root_a = [0u8; 16];
    root_a[0] = 0xAA;
    store
        .put(
            &TEST_COLLECTION,
            &root_a,
            &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
        )
        .unwrap();

    // Force a rotation so the next write goes to a different shard.
    store.shards.active_shard().file_len.store(
        shard::MAX_SHARD_BYTES - 10,
        std::sync::atomic::Ordering::Release,
    );

    // Put a record for collection B — lands on shard 1.
    let mut root_b = [0u8; 16];
    root_b[0] = 0xBB;
    store
        .put(
            &OTHER_COLLECTION,
            &root_b,
            &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
        )
        .unwrap();

    // Force another rotation and put more collection A garbage on shard 2.
    store.shards.active_shard().file_len.store(
        shard::MAX_SHARD_BYTES - 10,
        std::sync::atomic::Ordering::Release,
    );
    let mut garbage = [0u8; 16];
    garbage[0] = 0xCC;
    store
        .put(
            &TEST_COLLECTION,
            &garbage,
            &NodeData::new(bytes::Bytes::from_static(b"garbage")),
        )
        .unwrap();

    // Repack collection A with only root_a as live.  The garbage record
    // should be dropped, but shard 1 (which holds collection B's root)
    // must NOT be retired.
    store.set_live_roots(&TEST_COLLECTION, vec![root_a]);
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    assert_eq!(kept, 1);
    assert!(dropped > 0);

    // Collection B's root must still be retrievable — shard 1 was not retired.
    let got = store.get(&OTHER_COLLECTION, &root_b).unwrap();
    assert!(
        got.is_some(),
        "collection B root must survive repack of collection A — shared shard must not be retired",
    );
}

/// `open_read_only` must actually coexist with a live writer end to
/// end — not just take no lock, but also not touch/truncate the
/// writer's files via the collection-index rebuild scan (the real risk this
/// whole design exists to avoid). Also confirms it sees the writer's
/// already-durable data and that the writer is unaffected afterward.
#[test]
fn test_open_read_only_coexists_with_active_writer() {
    let dir = test_dir("open_read_only_coexist");
    let writer = PackfileStorage::open(dir.clone()).unwrap();

    let mut id = [0u8; 16];
    id[0] = 0xAA;
    writer
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"hello")),
        )
        .unwrap();
    // Reads through a second (read-only) process open against the same
    // directory only observe committed bytes, so make the write durable
    // before opening the reader — buffered bytes are RAM-only.
    writer.sync_all().unwrap();

    // Open read-only while the writer is still alive — must succeed
    // (no lock conflict) and see the write above.
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    assert_eq!(
        reader.get(&TEST_COLLECTION, &id).unwrap().map(|d| d.bytes),
        Some(bytes::Bytes::from_static(b"hello"))
    );
    drop(reader);

    // The writer must be completely unaffected — still open, still
    // able to write more data afterward.
    let mut id2 = [0u8; 16];
    id2[0] = 0xBB;
    writer
        .put(
            &TEST_COLLECTION,
            &id2,
            &NodeData::new(bytes::Bytes::from_static(b"world")),
        )
        .unwrap();
    assert_eq!(
        writer.get(&TEST_COLLECTION, &id2).unwrap().map(|d| d.bytes),
        Some(bytes::Bytes::from_static(b"world"))
    );
}

/// `repack_shard` must find and repack every collection sharing a shard —
/// the targeted way to reclaim one shard without waiting for each
/// collection to independently cross its own repack threshold.
#[test]
fn test_repack_shard_repacks_every_referencing_collection() {
    let dir = test_dir("repack_shard");
    let store = PackfileStorage::open(dir).unwrap();

    let mut root_a = [0u8; 16];
    root_a[0] = 0xAA;
    let mut garbage_a = [0u8; 16];
    garbage_a[0] = 0xA1;
    store
        .put(
            &TEST_COLLECTION,
            &root_a,
            &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &garbage_a,
            &NodeData::new(bytes::Bytes::from_static(b"collection A garbage")),
        )
        .unwrap();
    store.set_live_roots(&TEST_COLLECTION, vec![root_a]);

    let mut root_b = [0u8; 16];
    root_b[0] = 0xBB;
    store
        .put(
            &OTHER_COLLECTION,
            &root_b,
            &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
        )
        .unwrap();
    // No set_live_roots for collection B: everything must survive its repack.

    // Fresh pool, both collections' first writes land on shard 0 (shared).
    let referencing = store.collections_referencing_shard(0).unwrap();
    assert_eq!(referencing.len(), 2);
    assert!(referencing.contains(&TEST_COLLECTION));
    assert!(referencing.contains(&OTHER_COLLECTION));

    let mut results = store.repack_shard(0, |_hash, _data| Vec::new()).unwrap();
    results.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);

    let mut expected = vec![
        (TEST_COLLECTION, 1usize, 1usize),
        (OTHER_COLLECTION, 1usize, 0usize),
    ];
    expected.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);
    assert_eq!(
        results, expected,
        "repack_shard must repack both collections sharing shard 0"
    );

    assert!(store.get(&TEST_COLLECTION, &root_a).unwrap().is_some());
    assert!(store.get(&TEST_COLLECTION, &garbage_a).unwrap().is_none());
    assert!(store.get(&OTHER_COLLECTION, &root_b).unwrap().is_some());
}

/// `collections_referencing_shard` must filter the shard-scan's candidate
/// set against each collection's *current* index, not just report every collection
/// whose bytes ever physically touched the shard. A collection that has
/// since repacked its live data onto a different shard leaves its old
/// bytes sitting there untouched (shards are append-only, never
/// rewritten in place) — that collection must NOT show up as still
/// referencing the shard its data moved away from.
#[test]
fn test_collections_referencing_shard_excludes_collections_already_repacked_away() {
    let dir = test_dir("collections_referencing_shard_stale");
    let store = PackfileStorage::open(dir).unwrap();

    let mut root = [0u8; 16];
    root[0] = 0xAA;
    store
        .put(
            &TEST_COLLECTION,
            &root,
            &NodeData::new(bytes::Bytes::from_static(b"root")),
        )
        .unwrap();

    // OTHER_COLLECTION also lands on shard 0 (shared, still the pool's
    // active shard) and stays live there — this is what keeps shard 0
    // from being retired outright once TEST_COLLECTION moves off it, so the
    // "stale reference" case below is actually reachable to query.
    let mut other_root = [0u8; 16];
    other_root[0] = 0xBB;
    store
        .put(
            &OTHER_COLLECTION,
            &other_root,
            &NodeData::new(bytes::Bytes::from_static(b"other collection root")),
        )
        .unwrap();

    // Force TEST_COLLECTION's home (shard 0) to look full, so its next write
    // rotates *its own* home to shard 1 — root stays physically on
    // shard 0, but TEST_COLLECTION's home (and this next record, `garbage`)
    // moves to shard 1.
    store.shards.active_shard().file_len.store(
        shard::MAX_SHARD_BYTES - 10,
        std::sync::atomic::Ordering::Release,
    );
    let mut garbage = [0u8; 16];
    garbage[0] = 0xA1;
    store
        .put(
            &TEST_COLLECTION,
            &garbage,
            &NodeData::new(bytes::Bytes::from_static(b"garbage")),
        )
        .unwrap();

    // Repack with only root live: TEST_COLLECTION's records now span *two*
    // source shards (root on 0, garbage on 1), and a repack must never
    // rewrite live data back into one of its own sources (see
    // `ShardPool::prepare_collection_repack`'s doc) — so the kept copy lands
    // on a *third* shard (2), not shard 1. Shard 1, left holding only
    // the now-dropped `garbage` record with nothing live pointing at
    // it, gets retired and closed during this same repack.
    store.set_live_roots(&TEST_COLLECTION, vec![root]);
    store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();

    // Shard 0's raw bytes still physically contain TEST_COLLECTION's
    // original root record — a naive scan-only approach would still
    // report it. The current index must exclude it.
    assert!(
        !store.collections_referencing_shard(0).unwrap().contains(&TEST_COLLECTION),
        "a collection whose live data has moved off a shard must not be reported as still referencing it"
    );

    // The real invariant here isn't "the destination is shard 2" —
    // that's just where a deterministic, empty-pool allocator happens
    // to land today. What must hold is: exactly one destination, and
    // it's neither of the two source shards (0 and 1) a repack must
    // never write live data back into.
    let destinations = store.collection_referenced_shards(&TEST_COLLECTION);
    assert_eq!(
        destinations.len(),
        1,
        "TEST_COLLECTION's live data must land on exactly one shard after repack"
    );
    let destination = destinations[0];
    assert!(
        ![0, 1].contains(&destination),
        "the repack destination must not be one of its own source shards (0: root, 1: garbage)"
    );
    assert!(
        store
            .collections_referencing_shard(destination)
            .unwrap()
            .contains(&TEST_COLLECTION),
        "the collection's current shard must still be reported"
    );
}

#[test]
fn test_collection_directory_from_disk_matches_collection_summaries_after_sync() {
    let dir = test_dir("collection_directory_roundtrip");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    for i in 0..5u8 {
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(i),
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    for i in 0..3u8 {
        store
            .put(
                &OTHER_COLLECTION,
                &distinct_id(100 + i),
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    store.sync_all().unwrap();

    let mut from_disk = PackfileStorage::collection_directory_from_disk(&dir);
    from_disk.sort_unstable_by_key(|(collection_id, _)| *collection_id);

    let mut expected: Vec<([u8; 16], u64)> = store
        .collection_summaries()
        .into_iter()
        .map(|(collection_id, count, _mem, _cap)| (collection_id, count as u64))
        .collect();
    expected.sort_unstable_by_key(|(collection_id, _)| *collection_id);

    assert_eq!(
        from_disk, expected,
        "the persisted directory's per-collection totals must match the live index's own counts"
    );
    let disk_bytes = PackfileStorage::collection_disk_bytes_from_disk(&dir)
        .expect("v5 sidecar physical metrics");
    assert!(disk_bytes.get(&TEST_COLLECTION).copied().unwrap_or(0) > 0);
    assert!(disk_bytes.get(&OTHER_COLLECTION).copied().unwrap_or(0) > 0);
    assert!(PackfileStorage::collection_directory_persisted_at(&dir).is_some());
}

#[test]
fn put_many_overwrites_keep_one_live_node_and_count_every_frame() {
    let dir = test_dir("put_many_overwrites_accounting");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let id = distinct_id(0);
    let entries = vec![
        (id, NodeData::new(bytes::Bytes::from_static(b"first"))),
        (id, NodeData::new(bytes::Bytes::from_static(b"second"))),
    ];
    assert_eq!(store.put_many(&TEST_COLLECTION, &entries).unwrap(), 2);
    store.sync_all().unwrap();

    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 1)],
        "an overwrite inside one batch must not inflate live-node counts"
    );
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "batch accounting must include both appended frames"
    );
}

#[test]
fn put_verified_and_empty_records_are_accounted_as_physical_frames() {
    let dir = test_dir("verified_and_empty_accounting");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let verified_id = distinct_id(0);
    let verified = bytes::Bytes::from_static(b"verified");
    let digest = DigestAlgorithm::Sha256.digest(&verified);
    store
        .put_verified(
            &TEST_COLLECTION,
            &verified_id,
            &NodeData::new(verified),
            &digest,
            DigestAlgorithm::Sha256,
            &digest,
            None,
        )
        .unwrap();
    let empty_id = distinct_id(1);
    store
        .put(
            &TEST_COLLECTION,
            &empty_id,
            &NodeData::new(bytes::Bytes::new()),
        )
        .unwrap();
    store.sync_all().unwrap();

    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 2)],
        "empty records still occupy live index slots"
    );
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "verified and empty records must use their encoded frame lengths"
    );
}

#[test]
fn failed_sidecar_recovery_is_retried_and_never_publishes_zero_metrics() {
    let dir = test_dir("sidecar_recovery_failure");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"data")),
        )
        .unwrap();
    store.sync_all().unwrap();
    let sidecar = PackfileStorage::shard_collections_path(&dir);
    let before = fs::read(&sidecar).unwrap();
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;

    // Model an open whose recovery scan failed: byte totals unknown, flag set.
    store.index_tables.write().collection_disk_bytes.clear();
    store
        .shard_collections_recovery_failed
        .store(true, Ordering::Release);

    // While the scan still fails (an unreadable pack), nothing is written.
    let junk = dir.join("pack_00000000000000ff.pack");
    fs::write(&junk, b"not a packfile").unwrap();
    assert!(store.persist_shard_collections().is_err());
    assert!(store
        .shard_collections_recovery_failed
        .load(Ordering::Acquire));
    assert_eq!(fs::read(&sidecar).unwrap(), before, "no zero-byte sidecar");

    // Once the scan can succeed, the next persist retries it and recovers.
    fs::remove_file(&junk).unwrap();
    store.persist_shard_collections().unwrap();
    assert!(!store
        .shard_collections_recovery_failed
        .load(Ordering::Acquire));
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected)
    );
}

/// Deterministic version of the recovery/append race: hold recovery between
/// its scan and the swap of its totals and force a writer into that window.
/// The writer must be blocked by the collection mutexes; without them its
/// frame would be appended after the scan and then be lost by the swap.
#[test]
fn a_writer_cannot_slip_into_the_recovery_scan_to_swap_window() {
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::time::Duration;

    let wait = Duration::from_secs(10);
    let dir = test_dir("sidecar_recovery_window");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    store.sync_all().unwrap();
    store.index_tables.write().collection_disk_bytes.clear();
    store
        .shard_collections_recovery_failed
        .store(true, Ordering::Release);

    let (scanned_tx, scanned_rx) = channel::<()>();
    let (resume_tx, resume_rx) = channel::<()>();
    let resume_rx = parking_lot::Mutex::new(resume_rx);
    *store.recovery_pause_hook.lock() = Some(Arc::new(move || {
        scanned_tx.send(()).unwrap();
        resume_rx
            .lock()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
    }));

    let recovery = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || store.persist_shard_collections().unwrap())
    };
    // Recovery has scanned and is paused, still holding its locks.
    scanned_rx.recv_timeout(wait).unwrap();

    let (started_tx, started_rx) = channel::<()>();
    let (done_tx, done_rx) = channel::<()>();
    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(1),
                    &NodeData::new(bytes::Bytes::from_static(b"written during recovery")),
                )
                .unwrap();
            done_tx.send(()).unwrap();
        })
    };
    // Only start the clock once the writer is running, so the timeout
    // measures lock blocking rather than thread scheduling.
    started_rx.recv_timeout(wait).unwrap();
    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)),
        Err(RecvTimeoutError::Timeout),
        "a writer must be blocked while recovery holds the collection locks"
    );

    resume_tx.send(()).unwrap();
    recovery.join().unwrap();
    done_rx
        .recv_timeout(wait)
        .expect("writer must finish once recovery ends");
    writer.join().unwrap();

    // `sync_all` defers the sidecar on a delta-only barrier; flush it
    // explicitly so this checks the totals rather than the deferral.
    store.sync_all().unwrap();
    // The sidecar is rebuilt from the shards' in-memory lengths, while
    // `physical_layout` reads the pack files. Force the writer's frame out
    // of any staging buffer first so the two sources cannot disagree just
    // because the frame was appended after the recovery thread's own
    // `flush_all` — the very window this test drives.
    store.shards.flush_all().unwrap();
    store.persist_shard_collections().unwrap();
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "the frame written around recovery must be in the persisted totals"
    );
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// A put whose insert triggers index growth takes a different code path;
/// it must still account for its frame's bytes.
#[test]
fn puts_and_batches_that_grow_the_index_are_byte_accounted() {
    let dir = test_dir("growth_byte_accounting");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x43u8; 16];
    for i in 0..200u8 {
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(i),
                &NodeData::new(bytes::Bytes::from(vec![b'x'; 64])),
            )
            .unwrap();
    }
    let batch: Vec<_> = (0..200u8)
        .map(|i| {
            (
                distinct_id(i),
                NodeData::new(bytes::Bytes::from(vec![b'y'; 64])),
            )
        })
        .collect();
    assert_eq!(store.put_many(&collection, &batch).unwrap(), 200);

    // A batch into an EXISTING small collection is what grows its index
    // mid-batch (a fresh collection is pre-sized for the batch).
    let grown_collection = [0x44u8; 16];
    store
        .put(
            &grown_collection,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    let (_, _, capacity_before) = store.collection_index_info(&grown_collection).unwrap();
    let batch: Vec<_> = (1..250u8)
        .map(|i| {
            (
                distinct_id(i),
                NodeData::new(bytes::Bytes::from(vec![b'z'; 64])),
            )
        })
        .collect();
    assert_eq!(store.put_many(&grown_collection, &batch).unwrap(), 249);
    let (_, _, capacity_after) = store.collection_index_info(&grown_collection).unwrap();
    assert!(
        capacity_after > capacity_before,
        "the batch must grow the index ({capacity_before} -> {capacity_after})"
    );
    store.sync_all().unwrap();

    let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
    let sidecar = PackfileStorage::collection_disk_bytes_from_disk(&dir).unwrap();
    for id in [TEST_COLLECTION, collection, grown_collection] {
        assert_eq!(
            sidecar.get(&id),
            Some(&physical.collections[&id].disk_bytes),
            "collection {id:02x?}: growth must not drop a frame's bytes"
        );
    }
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// Recovery must be atomic with appends: frames written while it runs may
/// not be left out of (or double counted in) the totals it publishes.
#[test]
fn sidecar_recovery_is_consistent_with_concurrent_appends() {
    let dir = test_dir("sidecar_recovery_concurrent");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    store.sync_all().unwrap();

    store.index_tables.write().collection_disk_bytes.clear();
    store
        .shard_collections_recovery_failed
        .store(true, Ordering::Release);

    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 1..200u8 {
                store
                    .put(
                        &TEST_COLLECTION,
                        &distinct_id(i),
                        &NodeData::new(bytes::Bytes::from(vec![b'x'; 64])),
                    )
                    .unwrap();
            }
        })
    };
    // Recover while the writer is appending.
    while store
        .shard_collections_recovery_failed
        .load(Ordering::Acquire)
    {
        let _ = store.persist_shard_collections();
    }
    writer.join().unwrap();
    // `sync_all` now defers the sidecar on a delta-only barrier, so flush
    // it explicitly before checking the totals recovery published.
    store.sync_all().unwrap();
    store.persist_shard_collections().unwrap();

    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "totals published by recovery must match the frames on disk"
    );
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// A checkpoint that survives without its shard→collection sidecar must not
/// leave the sidecar missing (no write dirties the store to regenerate it)
/// nor persist zero byte totals for data that is already on disk.
#[test]
fn missing_sidecar_is_rewritten_with_exact_bytes_when_the_checkpoint_survives() {
    let dir = test_dir("sidecar_missing_checkpoint_present");
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let id = distinct_id(0);
        for payload in [&b"first"[..], &b"second!!"[..]] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::copy_from_slice(payload)),
                )
                .unwrap();
        }
        store.sync_all().unwrap();
    }
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    let sidecar = PackfileStorage::shard_collections_path(&dir);
    fs::remove_file(&sidecar).unwrap();
    assert!(PackfileStorage::index_checkpoint_path(&dir).exists());

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.sync_all().unwrap();
    assert!(sidecar.exists(), "sync must regenerate the lost sidecar");
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 1)]
    );
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "regenerated bytes must equal the physical layout, not zero"
    );
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

/// A delta-only sync must not rewrite the sidecar: it records staleness and
/// waits for an explicit flush. The first sync (no checkpoint base) is a
/// full rewrite, so it is still an anchor.
#[test]
fn delta_barrier_defers_sidecar_until_an_explicit_flush() {
    let dir = test_dir("sidecar_deferred_delta_barrier");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    store.sync_all().unwrap();
    assert_eq!(
        store.stats().sidecar_writes,
        1,
        "the checkpoint anchor writes the sidecar"
    );
    let sidecar = PackfileStorage::shard_collections_path(&dir);
    let after_checkpoint = fs::read(&sidecar).unwrap();

    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"more")),
        )
        .unwrap();
    store.sync_all().unwrap();
    assert_eq!(
        store.stats().sidecar_writes,
        1,
        "a delta-only barrier must not rewrite the sidecar"
    );
    assert!(store.shard_collections_stale.load(Ordering::Relaxed));
    assert_eq!(
        fs::read(&sidecar).unwrap(),
        after_checkpoint,
        "the on-disk sidecar is left stale, not rewritten"
    );

    // An explicit flush is the checkpoint/repack/shutdown anchor.
    store.persist_shard_collections().unwrap();
    assert_eq!(store.stats().sidecar_writes, 2);
    assert!(!store.shard_collections_stale.load(Ordering::Relaxed));
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 2)]
    );

    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// A sidecar left stale by a deferred delta barrier must not be trusted: a
/// concurrent read-only open falls back to the slot walk. Dropping the
/// writer flushes it, so a later open trusts it again.
#[test]
fn stale_sidecar_forces_slot_scan_and_drop_repins_it() {
    let dir = test_dir("sidecar_stale_then_drop");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    for i in 0..3u8 {
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(i),
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    store.sync_all().unwrap(); // checkpoint anchor: sidecar written

    for i in 3..6u8 {
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(i),
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    store.sync_all().unwrap(); // delta barrier: sidecar deferred

    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    assert_eq!(
        reader.open_timings().unwrap().bookkeeping_source,
        BookkeepingSource::SlotScan,
        "a stale sidecar must not be trusted"
    );
    assert_eq!(
        reader
            .collection_summaries()
            .into_iter()
            .find(|(id, _, _, _)| *id == TEST_COLLECTION)
            .unwrap()
            .1,
        6,
        "the slot walk must still see every live record"
    );
    drop(reader);

    // Dropping the writer flushes the deferred sidecar, re-pinned to the
    // live pack set.
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened.open_timings().unwrap().bookkeeping_source,
        BookkeepingSource::Sidecar,
        "Drop must have re-pinned the sidecar to the live packs"
    );
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 6)]
    );
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

/// A torn (partially written) sidecar must be treated as absent: open
/// rebuilds from the slot walk, and the next clean barrier regenerates it
/// with exact byte totals.
#[test]
fn torn_sidecar_is_ignored_and_regenerated() {
    let dir = test_dir("sidecar_torn");
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.sync_all().unwrap();
    }
    let sidecar = PackfileStorage::shard_collections_path(&dir);
    let full = fs::read(&sidecar).unwrap();
    assert!(full.len() > 20);
    fs::write(&sidecar, &full[..full.len() / 2]).unwrap();

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened.open_timings().unwrap().bookkeeping_source,
        BookkeepingSource::SlotScan,
        "a torn sidecar must not be trusted"
    );
    // The open observed a missing sidecar, so the clean barrier regenerates
    // it without needing any write.
    reopened.sync_all().unwrap();
    assert!(reopened.stats().sidecar_writes >= 1);
    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "regeneration must use exact byte totals, not zeros"
    );
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

/// The rate-limited flush must stay consistent while writers append.
#[test]
fn maybe_persist_shard_collections_is_consistent_under_concurrent_writes() {
    let dir = test_dir("sidecar_maybe_persist_concurrent");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 0..500u32 {
                let mut id = [0u8; 16];
                id[..4].copy_from_slice(&i.to_le_bytes());
                id[9] = u8::try_from(i & 0xff).unwrap().wrapping_mul(37);
                store
                    .put(
                        &TEST_COLLECTION,
                        &id,
                        &NodeData::new(bytes::Bytes::from(vec![b'x'; 32])),
                    )
                    .unwrap();
            }
        })
    };
    while !writer.is_finished() {
        store.persist_shard_collections().unwrap();
    }
    writer.join().unwrap();
    store.sync_all().unwrap();
    store.persist_shard_collections().unwrap();

    let expected = crate::packfile::layout::physical_layout(&dir)
        .unwrap()
        .collections[&TEST_COLLECTION]
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "totals flushed under concurrent writes must match the frames on disk"
    );
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 500)]
    );
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// A write that is never synced is deliberately not flushed at shutdown:
/// it already advances a pack past the checkpoint, so the next open cannot
/// take the checkpoint fast path and never consults the sidecar. Correctness
/// must hold anyway, because the full rescan rebuilds everything.
#[test]
fn unsynced_write_then_drop_reopens_by_full_scan_with_correct_data() {
    let dir = test_dir("sidecar_unsynced_drop");
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"synced")),
            )
            .unwrap();
        store.sync_all().unwrap();
        // A later write with no sync must not mark the sidecar stale, and
        // `Drop` must not need to flush it.
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(1),
                &NodeData::new(bytes::Bytes::from_static(b"unsynced")),
            )
            .unwrap();
        assert!(!store.is_shard_collections_stale());
    }

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened.open_timings().unwrap().path,
        OpenPath::FullScan,
        "an unsynced write invalidates the checkpoint fast path"
    );
    assert_eq!(
        reopened
            .collection_summaries()
            .into_iter()
            .find(|(id, _, _, _)| *id == TEST_COLLECTION)
            .unwrap()
            .1,
        2,
        "the full rescan must still recover every record"
    );
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_collection_disk_bytes_track_overwrite_reopen_and_repack() {
    let dir = test_dir("collection_disk_bytes_semantics");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let id = distinct_id(0);
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    store.sync_all().unwrap();

    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 1)],
        "overwriting a key must not inflate the live-node count"
    );
    let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
    let expected = physical
        .collections
        .get(&TEST_COLLECTION)
        .expect("collection in physical layout")
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "sidecar bytes must include both the original and overwrite frames"
    );

    drop(store);
    let checkpoint_reopened = PackfileStorage::open(dir.clone()).unwrap();
    checkpoint_reopened.sync_all().unwrap();
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "checkpoint-backed reopen must preserve physical byte accounting"
    );
    drop(checkpoint_reopened);
    fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
    fs::remove_file(PackfileStorage::shard_collections_path(&dir)).unwrap();
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    reopened.sync_all().unwrap();
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        vec![(TEST_COLLECTION, 1)],
        "a full rescan must preserve the live-node count"
    );
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&expected),
        "reopen must preserve physical byte accounting"
    );
    reopened
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    reopened.sync_all().unwrap();
    let repacked_physical = crate::packfile::layout::physical_layout(&dir).unwrap();
    let repacked_expected = repacked_physical
        .collections
        .get(&TEST_COLLECTION)
        .expect("collection after repack")
        .disk_bytes;
    assert_eq!(
        PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .unwrap()
            .get(&TEST_COLLECTION),
        Some(&repacked_expected),
        "repack must update physical byte accounting"
    );
}

#[test]
fn test_collection_summaries_from_disk_capacity_matches_new_collection_floor() {
    // A one-node collection's live index starts at
    // `NEW_COLLECTION_INDEX_FLOOR` (64), never `LossyIndex::new`'s own
    // generic 16-slot floor. The disk-only estimate (no packfiles
    // opened) must report the same capacity the live index actually
    // has, or a load-factor figure derived from it is wrong.
    let dir = test_dir("collection_summaries_from_disk_capacity_floor");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();
    store.sync_all().unwrap();

    let live_capacity = store
        .collection_index_info(&TEST_COLLECTION)
        .expect("collection exists")
        .2;
    assert_eq!(
        live_capacity, 64,
        "sanity check: a fresh one-node collection's live index capacity \
         is NEW_COLLECTION_INDEX_FLOOR"
    );

    // Force `open` down its packfile-scan path rather than letting it use
    // the serialized checkpoint, then ensure the reconstructed index has
    // the same floor as the original collection.
    drop(store);
    fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened
            .collection_index_info(&TEST_COLLECTION)
            .expect("scanned collection exists")
            .2,
        u32::try_from(NEW_COLLECTION_INDEX_FLOOR).unwrap(),
        "an open-time scan keeps the new-collection index floor"
    );

    let from_disk = PackfileStorage::collection_summaries_from_disk(&dir)
        .expect("directory sidecar was persisted by sync_all");
    let (_, disk_nodes, _, disk_capacity) = from_disk
        .into_iter()
        .find(|(id, _, _, _)| *id == TEST_COLLECTION)
        .expect("collection present in disk summary");
    assert_eq!(disk_nodes, 1);
    assert_eq!(
        disk_capacity, live_capacity,
        "disk-only capacity estimate must match the live index, not undershoot it"
    );
}

#[test]
fn test_collection_directory_from_disk_empty_when_never_persisted() {
    let dir = test_dir("collection_directory_never_persisted");
    let _store = PackfileStorage::open(dir.clone()).unwrap();
    // No sync_all call: nothing has been persisted yet.
    assert_eq!(
        PackfileStorage::collection_directory_from_disk(&dir),
        Vec::new()
    );
    assert_eq!(
        PackfileStorage::collection_directory_persisted_at(&dir),
        None
    );
}

#[test]
fn test_collection_directory_reflects_delete_and_repack() {
    let dir = test_dir("collection_directory_delete_repack");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let a = distinct_id(0);
    let b = distinct_id(1);
    store
        .put(
            &TEST_COLLECTION,
            &a,
            &NodeData::new(bytes::Bytes::from_static(b"a")),
        )
        .unwrap();
    store
        .put(
            &OTHER_COLLECTION,
            &b,
            &NodeData::new(bytes::Bytes::from_static(b"b")),
        )
        .unwrap();
    store.sync_all().unwrap();

    let directory = PackfileStorage::collection_directory_from_disk(&dir);
    assert_eq!(
        directory.len(),
        2,
        "both collections must appear before deletion"
    );

    store.delete_collection(&TEST_COLLECTION).unwrap();
    // A delta-only sync defers the sidecar, so flush it to exercise the
    // persisted directory rather than the deferral.
    store.sync_all().unwrap();
    store.persist_shard_collections().unwrap();

    let directory = PackfileStorage::collection_directory_from_disk(&dir);
    assert_eq!(
        directory,
        vec![(OTHER_COLLECTION, 1)],
        "a deleted collection must not linger in the persisted directory"
    );
}

#[test]
fn test_plan_collection_repack_matches_real_repack_without_mutating() {
    let dir = test_dir("plan_matches_real");
    let store = PackfileStorage::open(dir).unwrap();

    for i in 0..5u8 {
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(i),
                &NodeData::new(bytes::Bytes::from(vec![i])),
            )
            .unwrap();
    }
    // Duplicate a hash's payload under a fresh id to create something
    // dedup would drop... actually LossyIndex already dedups by hash
    // on insert, so instead exercise the "no live roots: keep
    // everything" path, which is what --root-less usage always hits.
    let plan = store
        .plan_collection_repack(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    assert_eq!(plan.kept, 5);
    assert_eq!(plan.dropped, 0);
    assert!(plan.kept_bytes > 0);
    assert_eq!(plan.shards_touched, vec![0]);

    // The dry run must not have mutated anything: a real repack run
    // right after must see the exact same collection state and produce the
    // exact same result.
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();
    assert_eq!(kept, plan.kept);
    assert_eq!(dropped, plan.dropped);
}

#[test]
fn test_repack_closure_single_collection_single_shard() {
    let dir = test_dir("closure_trivial");
    let store = PackfileStorage::open(dir).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();

    let (collections, shards) = store.repack_closure(0).unwrap();
    assert_eq!(collections, vec![TEST_COLLECTION]);
    assert_eq!(shards, vec![0]);
}

#[test]
fn test_repack_closure_pulls_in_collections_second_shard_transitively() {
    // Collection A lives on shards {0, 1} (spans a rotation). Collection B lives
    // only on shard 1. Starting the closure from shard 0 must still
    // discover shard 1 (because collection A references it) and, through
    // shard 1, collection B — even though collection B never touched shard 0 at
    // all.
    let dir = test_dir("closure_transitive");
    let store = PackfileStorage::open(dir).unwrap();

    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-0")),
        )
        .unwrap();

    // Force rotation so TEST_COLLECTION's next write lands on a new shard.
    store.shards.active_shard().file_len.store(
        shard::MAX_SHARD_BYTES - 10,
        std::sync::atomic::Ordering::Release,
    );
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-1")),
        )
        .unwrap();

    store
        .put(
            &OTHER_COLLECTION,
            &distinct_id(2),
            &NodeData::new(bytes::Bytes::from_static(b"b-on-shard-1")),
        )
        .unwrap();

    let (mut collections, mut shards) = store.repack_closure(0).unwrap();
    collections.sort_unstable();
    shards.sort_unstable();
    let mut expected_collections = vec![TEST_COLLECTION, OTHER_COLLECTION];
    expected_collections.sort_unstable();
    assert_eq!(collections, expected_collections);
    assert_eq!(shards, vec![0, 1]);
}

#[test]
fn test_collection_referenced_shards_empty_for_unknown_collection() {
    let dir = test_dir("referenced_shards_unknown");
    let store = PackfileStorage::open(dir).unwrap();
    assert_eq!(
        store.collection_referenced_shards(&TEST_COLLECTION),
        Vec::<u16>::new()
    );
}

#[test]
fn test_repack_collection_reachable_without_live_roots_preserves_everything() {
    let dir = test_dir("repack_reachable_no_roots");
    let store = PackfileStorage::open(dir).unwrap();

    let mut id_a = [0u8; 16];
    id_a[0] = 1;
    let mut id_b = [0u8; 16];
    id_b[0] = 2;

    store
        .put(
            &TEST_COLLECTION,
            &id_a,
            &NodeData::new(bytes::Bytes::from_static(b"A")),
        )
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &id_b,
            &NodeData::new(bytes::Bytes::from_static(b"B")),
        )
        .unwrap();

    // No set_live_roots call for this collection: nothing is known to be
    // garbage, so everything must survive.
    let (kept, dropped) = store
        .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
        .unwrap();

    assert_eq!(kept, 2);
    assert_eq!(dropped, 0);
    assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_some());
    assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_some());
}

/// `pin_shards` must return a distinct `Arc<Shard>` per unique id, and
/// the returned handle must still resolve real, previously-written
/// data — i.e. it's the live pool's own shard, not a stand-in.
/// The deeper guarantee (pinning survives a shard being retired and
/// its slot recycled) is `Shard::drop`'s contract, tested directly in
/// `shard.rs` where `is_current` and the pool's internals are
/// accessible without going through `ShardPool::rotate`, which no
/// longer has any path to actually recycle a slot (see its doc).
#[test]
fn test_pin_shards_returns_readable_deduped_handles() {
    let dir = test_dir("pin_shards_basic");
    let store = PackfileStorage::open(dir).unwrap();

    let id = [0x77u8; 16];
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"pin me")),
        )
        .unwrap();
    let (slot, offset) = store
        .generation(&TEST_COLLECTION)
        .unwrap()
        .index
        .lookup(&id)
        .expect("just-written record must be indexed");

    // Duplicate ids in the input must collapse to one pinned entry.
    let pinned = store.pin_shards([slot, slot, slot].into_iter());
    assert_eq!(pinned.len(), 1);

    let record = store
        .read_at(&pinned[&slot], offset, true)
        .expect("pinned shard must resolve the real on-disk record");
    assert_eq!(record.data.as_ref(), b"pin me");
}

/// NOTE: this test asserts a stronger property than the concurrent
/// put+reachable-repack contract provides: every `put` that has returned
/// remains present in a later repack output. A failure implies no
/// data-loss or buffering bug — the durability/checkpoint machinery is
/// not exercised here.
///
/// An earlier revision of this comment documented a specific race (a put
/// landing between the repack's scan boundary and its generation swap)
/// and labelled the test KNOWN-FLAKY. That mechanism does not hold: both
/// `put` and `repack_collection_reachable` hold the same
/// `put_mutex(collection_id)` for their entire body, so a put cannot
/// observe an in-progress repack at all. Re-investigation (55 runs,
/// plain and under artificial CPU load) reproduced zero failures, and no
/// panic output survives from the original reports, so the flakiness
/// claim, its mechanism, and the "known-flaky" label are retracted as
/// unconfirmed. If this test ever fails, the cause is the unchecked
/// linearizability assumption above — investigate from that assertion,
/// not from a presumed put/repack interleaving.
#[test]
fn test_concurrent_put_repack_reachable_no_lost_writes() {
    use std::thread;

    let dir = test_dir("concurrent_put_repack_reachable");
    let store = PackfileStorage::open(dir).unwrap();

    let collection = [0x66u8; 16];
    let written_count = std::sync::atomic::AtomicU32::new(0);

    // No live roots configured, so the reachable repacker must fall
    // back to "preserve everything" — this proves that fallback holds
    // even when interleaved with concurrent writes.
    thread::scope(|scope| {
        let writer = scope.spawn(|| {
            for i in 0..200u32 {
                let mut id = [0u8; 16];
                id[0..4].copy_from_slice(&i.to_le_bytes());
                let data = NodeData::new(bytes::Bytes::from(format!("entry {i}")));
                store.put(&collection, &id, &data).unwrap();
                written_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });

        let repacker = scope.spawn(|| {
            for _ in 0..5 {
                std::thread::sleep(std::time::Duration::from_micros(50));
                let _ = store.repack_collection_reachable(&collection, |_hash, _data| Vec::new());
            }
        });

        writer.join().unwrap();
        repacker.join().unwrap();
    });

    let total = written_count.load(std::sync::atomic::Ordering::Relaxed);
    let mut found = 0u32;
    for i in 0..total {
        let mut id = [0u8; 16];
        id[0..4].copy_from_slice(&i.to_le_bytes());
        if store.get(&collection, &id).unwrap().is_some() {
            found += 1;
        }
    }

    assert_eq!(
        found, total,
        "lost writes during concurrent put+reachable-repack"
    );
}

/// Collect the metadata of every record in every pack file under `dir`.
fn all_record_metadata(dir: &std::path::Path) -> Vec<FrameMetadata> {
    use std::io::{Seek, SeekFrom};

    let mut found = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "pack") {
            let mut file = fs::File::open(&path).unwrap();
            file.seek(SeekFrom::Start(packfile::HEADER_LEN as u64))
                .unwrap();
            while let Some(record) = packfile::read_record_metadata(&mut file).unwrap() {
                if let Some(metadata) = record.metadata {
                    found.push(metadata);
                }
            }
        }
    }
    found
}

/// `put_verified` must reject a mismatched digest before writing anything,
/// and accept a matching digest while attaching the full logical id,
/// content digest, algorithm, and role to the on-disk frame.
#[test]
fn put_verified_checks_digest_and_attaches_metadata() {
    let dir = test_dir("put_verified_metadata");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x31; 16];
    let id = [0x32; 16];
    let payload = b"canonical event bytes";
    let data = NodeData::new(bytes::Bytes::from_static(payload));
    let logical_id = [0x77; 32];
    let digest = crate::storage::content_digest(DigestAlgorithm::Sha256, payload);

    // A wrong expected digest is rejected and leaves no record behind.
    let mut wrong = digest;
    wrong[0] ^= 0xff;
    let error = store
        .put_verified(
            &collection,
            &id,
            &data,
            &logical_id,
            DigestAlgorithm::Sha256,
            &wrong,
            Some(b"event"),
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Corrupt(_)), "got {error:?}");
    assert!(store.get(&collection, &id).unwrap().is_none());
    assert_eq!(all_record_metadata(&dir), Vec::new());

    // The matching digest succeeds.
    store
        .put_verified(
            &collection,
            &id,
            &data,
            &logical_id,
            DigestAlgorithm::Sha256,
            &digest,
            Some(b"event"),
        )
        .unwrap();
    store.sync().unwrap();

    assert_eq!(
        store.get(&collection, &id).unwrap().map(|d| d.bytes),
        Some(bytes::Bytes::from_static(payload))
    );

    let metadatas = all_record_metadata(&dir);
    assert_eq!(metadatas.len(), 1);
    let metadata = &metadatas[0];
    assert_eq!(metadata.logical_id, Some(logical_id));
    assert_eq!(metadata.content_digest, Some(digest));
    assert_eq!(metadata.digest_algorithm, DigestAlgorithm::Sha256);
    assert_eq!(metadata.role.as_deref(), Some(&b"event"[..]));

    // Read path reconstructs metadata too (via `read_record_metadata`).
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened.get(&collection, &id).unwrap().map(|d| d.bytes),
        Some(bytes::Bytes::from_static(payload))
    );
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

/// A generic `put` (metadata `None`) must not grow frames or set the
/// metadata flag, so existing workloads see byte-identical records.
#[test]
fn plain_put_writes_no_metadata() {
    let dir = test_dir("plain_put_no_metadata");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x41; 16];
    let id = [0x42; 16];
    store
        .put(
            &collection,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"plain")),
        )
        .unwrap();
    store.sync().unwrap();

    assert_eq!(all_record_metadata(&dir), Vec::new());
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// A writer that opens without a usable checkpoint (e.g. after an aborted
/// import that never synced) must persist one on its next sync even though
/// it wrote nothing itself; otherwise every later open rescans every pack.
#[test]
fn rescan_open_persists_a_checkpoint_on_the_next_sync() {
    let dir = test_dir("rescan_open_checkpoint");
    {
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &[0x11u8; 16],
                &NodeData::new(bytes::Bytes::from_static(b"payload")),
            )
            .unwrap();
        store.sync_all().unwrap();
    }
    let checkpoint = PackfileStorage::index_checkpoint_path(&dir);
    fs::remove_file(&checkpoint).unwrap();

    let store = PackfileStorage::open(dir.clone()).unwrap();
    assert!(!checkpoint.exists(), "the rescan open alone writes none");
    store.sync_all().unwrap();
    assert!(
        checkpoint.exists(),
        "a sync after a rescan open must persist the checkpoint"
    );
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn collection_len_counts_genesis_and_distinct_ids() {
    let dir = test_dir("collection_len");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    crate::storage::assert_collection_len_counts_genesis_and_distinct_ids(&store);
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// An empty batch must be a no-op across every backend: no collection is
/// created, so `collection_exists`/`collection_len` report absence. This
/// mirrors the in-memory-backend test of the same name; the two backends
/// previously disagreed (`InMemoryStorage` left an empty entry behind).
#[test]
fn empty_put_many_is_a_noop_and_does_not_create_the_collection() {
    let dir = test_dir("empty_put_many_noop");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x7Du8; 16];
    assert_eq!(store.put_many(&collection, &[]).unwrap(), 0);
    assert!(!store.collection_exists(&collection));
    assert_eq!(store.collection_len(&collection).unwrap(), None);
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// Concurrent genesis establishment on a real packfile-backed store must
/// serialize on the collection `put_mutex`: without it, two writers can
/// both observe an absent metadata record and append conflicting genesis
/// frames. `InMemoryStorage`'s equivalent is trivially serialized by its
/// single map write lock; this exercises the locking path that is not.
///
/// Scope: intra-process only. The `put_mutex` map is per-instance, so it
/// says nothing about writers in separate processes. Cross-process writers
/// are excluded by the shard pool's exclusive `.mtxdb.lock` writer lock
/// (`ShardPool::acquire_writer_lock`), which permits one writer process per
/// store; this test pins the thread-level race.
#[test]
fn ensure_collection_metadata_is_atomic_under_concurrency() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };

    let dir = test_dir("ensure_metadata_concurrent");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    let metadata = Arc::new(CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: None,
        schema: None,
    });
    let collection =
        derive_collection_id(metadata.member_namespace, &metadata.collection_canonical_id);

    // Release every thread at once so the lookup/append windows overlap.
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            let metadata = Arc::clone(&metadata);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                store
                    .ensure_collection_metadata(&collection, &metadata)
                    .expect("concurrent genesis establishment must be idempotent");
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(
        store.get_collection_metadata(&collection).unwrap(),
        Some((*metadata).clone())
    );
    assert_eq!(store.collection_len(&collection).unwrap(), Some(1));
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// Writers racing with *different* genesis metadata must not both win:
/// exactly one establishes the collection and every other writer sees the
/// mismatch. Without the collection's `put_mutex` held across the lookup
/// and append, several writers could each append their own genesis frame.
#[test]
fn ensure_collection_metadata_conflicting_writers_have_one_winner() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };

    let dir = test_dir("ensure_metadata_conflict");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    let collection = derive_collection_id(Some(*b"EVNT"), b"!room:matrix.org");
    let candidates: Vec<CollectionMetadata> = (0..8)
        .map(|i| CollectionMetadata {
            member_namespace: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some(format!("role_{i}")),
            schema: None,
        })
        .collect();

    let barrier = Arc::new(std::sync::Barrier::new(candidates.len()));
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .iter()
            .map(|metadata| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    store.ensure_collection_metadata(&collection, metadata)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(
        results.iter().filter(|r| r.is_ok()).count(),
        1,
        "exactly one conflicting genesis writer may succeed: {results:?}"
    );
    for result in results.iter().filter(|r| r.is_err()) {
        assert!(
            matches!(result, Err(StorageError::Internal(msg)) if msg.contains("mismatch")),
            "losers must report a metadata mismatch, got {result:?}"
        );
    }
    let stored = store
        .get_collection_metadata(&collection)
        .unwrap()
        .expect("the winner's genesis record is stored");
    assert!(candidates.contains(&stored));
    assert_eq!(store.collection_len(&collection).unwrap(), Some(1));
    drop(store);
    fs::remove_dir_all(&dir).ok();
}

/// Checkpoint-backed hash recovery must handle frames that carry metadata
/// (as `put_verified` writes do). `record_identity_at` previously rejected
/// `FLAG_METADATA` frames, so `grow_checkpoint_index` failed closed to a
/// full rescan instead of recovering their identity.
#[test]
fn checkpoint_growth_recovers_metadata_bearing_frames() {
    let dir = test_dir("checkpoint_growth_metadata");
    let collection = TEST_COLLECTION;
    let id = [0x33u8; 16];
    let payload = bytes::Bytes::from_static(b"metadata payload");
    let digest = DigestAlgorithm::Sha256.digest(&payload);

    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put_verified(
            &collection,
            &id,
            &NodeData::new(payload),
            &digest,
            DigestAlgorithm::Sha256,
            &digest,
            None,
        )
        .unwrap();
    store.sync_all().unwrap();
    drop(store);

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    let checkpoint_index = reopened
        .generation(&collection)
        .expect("checkpoint collection exists");
    assert!(checkpoint_index.index.is_mmap_backed());
    let grown = reopened
        .grow_checkpoint_index(&collection, &checkpoint_index.index)
        .unwrap()
        .expect("checkpoint index can grow");
    assert!(
        grown.lookup(&id).is_some(),
        "a metadata-bearing frame's identity must be recoverable"
    );
    drop(checkpoint_index);
    drop(reopened);
    fs::remove_dir_all(&dir).ok();
}

/// Two stores attached to one shared coordinator: a single pool's sync
/// commits both pools' published mutations in one durability barrier, and
/// a later reopen replays only each pool's own tagged frames.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_journal_fences_all_pools_and_routes_replay_by_pool() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_journal_root");
    let dir_a = test_dir("shared_journal_a");
    let dir_b = test_dir("shared_journal_b");
    let wal = root.join("wal.bin");

    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });

    let a = PackfileStorage::open(dir_a.clone()).unwrap();
    let b = PackfileStorage::open(dir_b.clone()).unwrap();
    a.enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    b.enable_shared_journal(Arc::clone(&coordinator), ShardType::EventDag)
        .unwrap();

    a.put(
        &TEST_COLLECTION,
        &distinct_id(0),
        &NodeData::new(bytes::Bytes::from_static(b"state")),
    )
    .unwrap();
    b.put(
        &OTHER_COLLECTION,
        &distinct_id(1),
        &NodeData::new(bytes::Bytes::from_static(b"event")),
    )
    .unwrap();

    // One durability barrier fences the other pool's published mutation
    // too, without checkpointing either pool and triggering reclaim.
    coordinator.sync().unwrap();
    assert_eq!(
        coordinator.committed_lsn(),
        coordinator.published_lsn(),
        "one barrier must commit every pool's published mutation"
    );
    // A fresh session over the shared segment: each autocommit write is
    // its own complete group; one fsync covered both.
    let (journal, scan) = Journal::open_shared(&wal).unwrap();
    assert_eq!(scan.groups.len(), 2, "one group per autocommit write");
    assert_eq!(scan.groups[0].entries.len(), 1);
    assert_eq!(scan.groups[1].entries.len(), 1);
    let recovered = Arc::new(JournalCoordinator::new(journal, &scan));

    let fresh_a = PackfileStorage::open(test_dir("shared_journal_fresh_a")).unwrap();
    fresh_a
        .enable_shared_journal(Arc::clone(&recovered), ShardType::State)
        .unwrap();
    assert_eq!(
        fresh_a.replay_journal().unwrap(),
        1,
        "the state store replays only its own pool's frame"
    );
    assert!(fresh_a
        .get(&TEST_COLLECTION, &distinct_id(0))
        .unwrap()
        .is_some());
    assert!(
        fresh_a
            .get(&OTHER_COLLECTION, &distinct_id(1))
            .unwrap()
            .is_none(),
        "a pool must not replay another pool's frames"
    );

    let fresh_b = PackfileStorage::open(test_dir("shared_journal_fresh_b")).unwrap();
    fresh_b
        .enable_shared_journal(Arc::clone(&recovered), ShardType::EventDag)
        .unwrap();
    assert_eq!(fresh_b.replay_journal().unwrap(), 1);
    assert!(fresh_b
        .get(&OTHER_COLLECTION, &distinct_id(1))
        .unwrap()
        .is_some());
    assert!(fresh_b
        .get(&TEST_COLLECTION, &distinct_id(0))
        .unwrap()
        .is_none());
    drop(fresh_a);
    drop(fresh_b);
    drop(a);
    drop(b);
}

/// A single-pool autocommit group can be reclaimed on its own once its
/// pool checkpoints: a reopen replays only the surviving frames, and LSNs
/// continue past the reclaimed prefix without being reused.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_reclaim_of_a_single_pool_group_then_reopen_replays_the_rest() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_reclaim_replay_root");
    let wal = root.join("wal.bin");
    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });
    let a = PackfileStorage::open(test_dir("shared_reclaim_replay_a")).unwrap();
    let b = PackfileStorage::open(test_dir("shared_reclaim_replay_b")).unwrap();
    a.enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    b.enable_shared_journal(Arc::clone(&coordinator), ShardType::EventDag)
        .unwrap();

    // LSN 1 is State's; LSNs 2 and 3 are EventDag's. Each autocommit
    // write is its own group.
    a.put(
        &TEST_COLLECTION,
        &distinct_id(0),
        &NodeData::new(bytes::Bytes::from_static(b"state")),
    )
    .unwrap();
    for id in [1u8, 2] {
        b.put(
            &OTHER_COLLECTION,
            &distinct_id(id),
            &NodeData::new(bytes::Bytes::from_static(b"event")),
        )
        .unwrap();
    }
    coordinator.sync().unwrap();
    assert_eq!(coordinator.committed_lsn(), 3);

    // State checkpoints through its own frame; EventDag has not. Only the
    // covered single-pool group may be reclaimed.
    coordinator.report_pool_coverage(ShardType::State, 1);
    assert!(coordinator.reclaim_shared().unwrap().is_some());
    drop(a);
    drop(b);
    drop(coordinator);

    let (journal, scan) = Journal::open_shared(&wal).unwrap();
    assert_eq!(scan.base_lsn, 2, "the state group is reclaimed");
    assert_eq!(scan.groups.len(), 2, "both uncovered event groups survive");
    assert_eq!(scan.groups[0].first_lsn, 2);
    let recovered = Arc::new(JournalCoordinator::new(journal, &scan));

    // The reclaimed state frame is gone from the journal by design: its
    // pool's checkpoint owns it, so a state store has nothing to replay.
    let fresh_a = PackfileStorage::open(test_dir("shared_reclaim_replay_fresh_a")).unwrap();
    fresh_a
        .enable_shared_journal(Arc::clone(&recovered), ShardType::State)
        .unwrap();
    assert_eq!(fresh_a.replay_journal().unwrap(), 0);

    // The event store replays exactly the frames that survived.
    let fresh_b = PackfileStorage::open(test_dir("shared_reclaim_replay_fresh_b")).unwrap();
    fresh_b
        .enable_shared_journal(Arc::clone(&recovered), ShardType::EventDag)
        .unwrap();
    assert_eq!(fresh_b.replay_journal().unwrap(), 2);
    for id in [1u8, 2] {
        assert!(fresh_b
            .get(&OTHER_COLLECTION, &distinct_id(id))
            .unwrap()
            .is_some());
    }

    // Numbering continues past the reclaimed prefix: no LSN is reused.
    fresh_b
        .put(
            &OTHER_COLLECTION,
            &distinct_id(3),
            &NodeData::new(bytes::Bytes::from_static(b"event")),
        )
        .unwrap();
    assert_eq!(recovered.published_lsn(), 4);
    drop(fresh_a);
    drop(fresh_b);
}

/// Collection logical versions remain stable after their WAL group is covered,
/// reclaimed, and the writer is reopened from its pool checkpoint.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_checkpoint_preserves_collection_versions_after_reclaim() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let wal_dir = test_dir("version_checkpoint_wal");
    let store_dir = test_dir("version_checkpoint_store");
    let wal = wal_dir.join("shared.wal");
    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });
    let store = PackfileStorage::open(store_dir.clone()).unwrap();
    store
        .enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(91),
            &NodeData::new(bytes::Bytes::from_static(b"versioned")),
        )
        .unwrap();
    coordinator.sync().unwrap();
    let expected = coordinator.collection_version(ShardType::State, &TEST_COLLECTION);
    store.sync_all().unwrap();

    let checkpoint = crate::index::checkpoint::read_checkpoint(
        &PackfileStorage::index_checkpoint_path(&store_dir),
    )
    .expect("sync_all writes a valid checkpoint");
    assert_eq!(
        checkpoint
            .logical_versions
            .iter()
            .find(|(id, _)| id == &TEST_COLLECTION)
            .map(|(_, version)| *version),
        Some(expected)
    );
    drop(store);
    drop(coordinator);

    let (journal, scan) = Journal::open_shared(&wal).unwrap();
    assert!(
        scan.groups.is_empty(),
        "checkpoint coverage reclaimed the WAL group"
    );
    let recovered = Arc::new(JournalCoordinator::new(journal, &scan));
    let reopened = PackfileStorage::open(store_dir.clone()).unwrap();
    reopened
        .enable_shared_journal(Arc::clone(&recovered), ShardType::State)
        .unwrap();
    assert_eq!(
        recovered.collection_version(ShardType::State, &TEST_COLLECTION),
        expected,
        "the checkpoint restores an idle collection's logical version"
    );
    drop(reopened);
}

/// A shared segment may only be reclaimed up to the minimum durable
/// coverage across every pool that has frames in it.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_reclaim_waits_for_every_pool_then_truncates() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_reclaim_root");
    let dir_a = test_dir("shared_reclaim_a");
    let dir_b = test_dir("shared_reclaim_b");
    let wal = root.join("wal.bin");

    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });

    let a = PackfileStorage::open(dir_a.clone()).unwrap();
    let b = PackfileStorage::open(dir_b.clone()).unwrap();
    a.enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    b.enable_shared_journal(Arc::clone(&coordinator), ShardType::EventDag)
        .unwrap();

    a.put(
        &TEST_COLLECTION,
        &distinct_id(0),
        &NodeData::new(bytes::Bytes::from_static(b"a")),
    )
    .unwrap();
    b.put(
        &OTHER_COLLECTION,
        &distinct_id(1),
        &NodeData::new(bytes::Bytes::from_static(b"b")),
    )
    .unwrap();

    // Only A checkpoints; B still has an un-covered frame in the segment,
    // so the shared prefix must not be reclaimed yet.
    a.sync_all().unwrap();
    assert_eq!(
        Journal::scan_read_only(&wal).unwrap().groups.len(),
        1,
        "shared reclaim must wait for every pool's coverage"
    );

    // B checkpoints too; now the covered prefix can be dropped.
    b.sync_all().unwrap();
    let scan = Journal::scan_read_only(&wal).unwrap();
    assert_eq!(scan.groups.len(), 0, "every pool covered: prefix reclaimed");
    assert!(
        scan.base_lsn > 1,
        "the segment base advanced past the reclaimed group"
    );
}

/// End-to-end: three interleaved pool groups, independent per-pool
/// checkpoints, shared reclaim, then a read-only reopen. The state reader's
/// index boundary is state's own watermark, so a base advanced by the
/// event-DAG pool must not be mistaken for a lost state frame.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_interleaved_checkpoints_reclaim_then_read_only_reopen() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_e2e_root");
    let dir_state = test_dir("shared_e2e_state");
    let dir_event = test_dir("shared_e2e_event");
    let wal = root.join("wal.bin");

    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });

    let state = PackfileStorage::open(dir_state.clone()).unwrap();
    let event = PackfileStorage::open(dir_event.clone()).unwrap();
    state
        .enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    event
        .enable_shared_journal(Arc::clone(&coordinator), ShardType::EventDag)
        .unwrap();

    // Interleaved commits: each sync checkpoints only its own pool.
    state
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"state-1")),
        )
        .unwrap();
    state.sync_all().unwrap();
    event
        .put(
            &OTHER_COLLECTION,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"event-1")),
        )
        .unwrap();
    event.sync_all().unwrap();
    state
        .put(
            &TEST_COLLECTION,
            &distinct_id(2),
            &NodeData::new(bytes::Bytes::from_static(b"state-2")),
        )
        .unwrap();
    state.sync_all().unwrap();
    // Force a full rewrite so this test exercises reclaim deterministically
    // (an ordinary sync may append a delta instead).
    state.force_index_checkpoint().unwrap();

    // The two pools have distinct watermarks. Each pool's group is
    // reclaimed as soon as that pool has checkpointed it: state's later
    // group does not wait on event-DAG, whose frames never reached it.
    let state_lsn = coordinator.committed_lsn_for_pool(ShardType::State);
    let event_lsn = coordinator.committed_lsn_for_pool(ShardType::EventDag);
    assert!(state_lsn > event_lsn, "state committed a later group");
    let scan = Journal::scan_read_only(&wal).unwrap();
    assert!(
        scan.groups.is_empty(),
        "each pool's own group is reclaimed once that pool covers it"
    );
    assert_eq!(
        scan.base_lsn,
        state_lsn + 1,
        "an idle pool must not pin a later group its frames never reached"
    );

    drop(state);
    drop(event);

    // Read-only reopen of the state pool: its checkpoint covered exactly
    // its own watermark, so the reclaimed prefix is not a coverage gap and
    // the durable records remain readable.
    let reader = PackfileStorage::open_read_only(dir_state.clone()).unwrap();
    reader
        .enable_read_journal_shared(&wal, ShardType::State)
        .unwrap();
    let got = reader
        .get_read_committed(&TEST_COLLECTION, &[distinct_id(0), distinct_id(2)])
        .unwrap();
    assert_eq!(
        got[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"state-1"[..])
    );
    assert_eq!(
        got[1].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"state-2"[..])
    );
}

/// A shared reader whose own frame was committed by another pool's sync
/// (so its checkpoint has not advanced) must tolerate a base LSN gap opened
/// entirely by the other pools and read its own frame, instead of failing
/// closed on the reclaim.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_read_committed_accepts_another_pools_reclaimed_prefix() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_gap_root");
    let dir_state = test_dir("shared_gap_state");
    let dir_event = test_dir("shared_gap_event");
    let wal = root.join("wal.bin");

    let coordinator = Arc::new({
        let (journal, scan) = Journal::open_shared(&wal).unwrap();
        JournalCoordinator::new(journal, &scan)
    });
    let state = PackfileStorage::open(dir_state.clone()).unwrap();
    let event = PackfileStorage::open(dir_event.clone()).unwrap();
    state
        .enable_shared_journal(Arc::clone(&coordinator), ShardType::State)
        .unwrap();
    event
        .enable_shared_journal(Arc::clone(&coordinator), ShardType::EventDag)
        .unwrap();

    // State checkpoints one frame, then the reader opens bound to it.
    state
        .put(
            &TEST_COLLECTION,
            &distinct_id(0),
            &NodeData::new(bytes::Bytes::from_static(b"state-1")),
        )
        .unwrap();
    state.sync_all().unwrap();
    state.force_index_checkpoint().unwrap();
    let reader = PackfileStorage::open_read_only(dir_state.clone()).unwrap();
    reader
        .enable_read_journal_shared(&wal, ShardType::State)
        .unwrap();

    // Event-DAG reclaims its own later groups, advancing the segment base
    // well past the reader's coverage without ever moving state's.
    for id in 1..4u8 {
        event
            .put(
                &OTHER_COLLECTION,
                &distinct_id(id),
                &NodeData::new(bytes::Bytes::from_static(b"event")),
            )
            .unwrap();
        event.sync_all().unwrap();
        event.force_index_checkpoint().unwrap();
    }

    // State publishes a second frame; event's sync commits it, but state
    // never checkpoints, so state's durable coverage stays at the first
    // frame while the frame now sits above the advanced base.
    state
        .put(
            &TEST_COLLECTION,
            &distinct_id(9),
            &NodeData::new(bytes::Bytes::from_static(b"state-2")),
        )
        .unwrap();
    event.sync_all().unwrap();

    let got = reader
        .get_read_committed(&TEST_COLLECTION, &[distinct_id(9)])
        .expect("a base gap made of other pools' frames must not fail closed");
    assert_eq!(
        got[0].as_ref().map(|data| data.bytes.as_ref()),
        Some(&b"state-2"[..])
    );
    assert_eq!(
        reader
            .get_read_committed(&TEST_COLLECTION, &[distinct_id(0)])
            .unwrap()[0]
            .as_ref()
            .map(|data| data.bytes.as_ref()),
        Some(&b"state-1"[..]),
        "the reader's own checkpointed frame must remain readable"
    );
}

/// A read-only worker on a shared WAL must observe only its own pool's
/// tagged frames, not the other pools' interleaved in the same segment.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_read_committed_filters_by_pool() {
    use crate::journal::{Journal, JournalCoordinator};
    use crate::layout::ShardType;

    let root = test_dir("shared_read_committed_root");
    let wal = root.join("wal.bin");
    let (journal, scan) = Journal::open_shared(&wal).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    coordinator
        .publish_group_tagged(
            ShardType::State,
            &[JournalMutation::Put {
                collection_id: TEST_COLLECTION,
                node_id: distinct_id(0),
                payload: b"state".to_vec(),
            }],
        )
        .unwrap();
    coordinator
        .publish_group_tagged(
            ShardType::EventDag,
            &[JournalMutation::Put {
                collection_id: OTHER_COLLECTION,
                node_id: distinct_id(1),
                payload: b"event".to_vec(),
            }],
        )
        .unwrap();
    coordinator.sync().unwrap();
    drop(coordinator);

    // A read-only open needs at least one shard, so seed each reader's own
    // pool directory with a durable record in an unrelated collection.
    let seed_dir = |name: &str| {
        let dir = test_dir(name);
        {
            let seed = PackfileStorage::open(dir.clone()).unwrap();
            seed.put(
                &[0x77; 16],
                &distinct_id(9),
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
            seed.sync_all().unwrap();
        }
        dir
    };

    let state_reader = PackfileStorage::open_read_committed_shared(
        seed_dir("shared_read_committed_state"),
        &wal,
        ShardType::State,
    )
    .unwrap();
    assert!(
        state_reader
            .get_read_committed(&TEST_COLLECTION, &[distinct_id(0)])
            .unwrap()[0]
            .is_some(),
        "the state worker must see its own pool's frame"
    );
    assert!(
        state_reader
            .get_read_committed(&OTHER_COLLECTION, &[distinct_id(1)])
            .unwrap()[0]
            .is_none(),
        "the state worker must not see the event-DAG pool's frame"
    );

    let event_reader = PackfileStorage::open_read_committed_shared(
        seed_dir("shared_read_committed_event"),
        &wal,
        ShardType::EventDag,
    )
    .unwrap();
    assert!(event_reader
        .get_read_committed(&OTHER_COLLECTION, &[distinct_id(1)])
        .unwrap()[0]
        .is_some());
    assert!(event_reader
        .get_read_committed(&TEST_COLLECTION, &[distinct_id(0)])
        .unwrap()[0]
        .is_none());
}

/// The root shared-WAL lock is exclusive while held and reacquirable after
/// the holder drops it.
#[test]
#[cfg(feature = "multi-reader")]
fn shared_wal_lock_is_exclusive() {
    let root = test_dir("shared_wal_lock_exclusive");
    // A shared-WAL lock is gated on a shared-layout database descriptor.
    let _layout = crate::layout::DatabaseLayout::open(root.clone()).unwrap();
    let first = crate::journal::SharedWalLock::acquire(&root).unwrap();
    let second = crate::journal::SharedWalLock::acquire(&root);
    assert!(
        matches!(second, Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock),
        "a second writer must be refused while the root lock is held"
    );
    drop(first);
    assert!(
        crate::journal::SharedWalLock::acquire(&root).is_ok(),
        "a released root lock must be reacquirable"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn packfile_storage_create_or_upsert_established_validated_contract() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        MEMBER_NAMESPACE_INTL,
    };

    let dir = test_dir("create_or_upsert_contract");
    let store = PackfileStorage::open(dir.clone()).unwrap();

    let canonical_id = b"sys:packfile-upsert";
    let col_id = derive_collection_id(Some(MEMBER_NAMESPACE_INTL), canonical_id);

    let valid_meta = CollectionMetadata {
        member_namespace: Some(MEMBER_NAMESPACE_INTL),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("system_auxiliary".to_owned()),
        schema: None,
    };

    let node_id = [0x55; 16];
    let data1 = NodeData::new(bytes::Bytes::from_static(b"val1"));

    // 1. Validation failure on absent collection leaves collection absent
    let err = store
        .create_or_upsert_established_validated(&col_id, &valid_meta, &node_id, &data1, &mut |_| {
            Err(StorageError::Internal("pre-check failed".into()))
        })
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));
    assert!(!store.collection_exists(&col_id));
    assert!(store.get_collection_metadata(&col_id).unwrap().is_none());
    assert!(store.get(&col_id, &node_id).unwrap().is_none());

    // 2. Successful establishment commits metadata and record atomically
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data1,
            &mut |existing| {
                assert!(existing.is_none());
                Ok(())
            },
        )
        .unwrap();
    assert!(store.collection_exists(&col_id));
    assert_eq!(
        store.get_collection_metadata(&col_id).unwrap().unwrap(),
        valid_meta
    );
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"val1"[..]
    );

    // 3. Idempotent retry: same key + same payload -> Ok(())
    let mut ran = false;
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data1,
            &mut |existing| {
                ran = true;
                assert_eq!(existing.unwrap().bytes, b"val1"[..]);
                Ok(())
            },
        )
        .unwrap();
    assert!(ran);

    // 4. Same key replacement succeeds with new value
    let data2 = NodeData::new(bytes::Bytes::from_static(b"val2_updated"));
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data2,
            &mut |existing| {
                assert_eq!(existing.unwrap().bytes, b"val1"[..]);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"val2_updated"[..]
    );

    // 5. Collision detected by validator aborts without mutating
    let data3 = NodeData::new(bytes::Bytes::from_static(b"val3_conflict"));
    let err = store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data3,
            &mut |existing| {
                assert_eq!(existing.unwrap().bytes, b"val2_updated"[..]);
                Err(StorageError::Collision("logical key collision".into()))
            },
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Collision(_)));
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"val2_updated"[..]
    );

    // 6. Persistence across reopen
    drop(store);
    let reopened = PackfileStorage::open(dir).unwrap();
    assert_eq!(
        reopened.get_collection_metadata(&col_id).unwrap().unwrap(),
        valid_meta
    );
    assert_eq!(
        reopened.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"val2_updated"[..]
    );

    // 7. Corrupted stored metadata detection: stored canonical id does not reproduce collection id
    let meta_colliding = CollectionMetadata {
        collection_canonical_id: b"sys:packfile-other".to_vec(),
        ..valid_meta.clone()
    };
    let target_col_id = derive_collection_id(
        meta_colliding.member_namespace,
        &meta_colliding.collection_canonical_id,
    );
    let sim_dir = test_dir("create_or_upsert_collision");
    let sim_store = PackfileStorage::open(sim_dir).unwrap();
    let seed = [(
        COLLECTION_METADATA_RECORD_ID,
        NodeData::new(valid_meta.encode().into()),
    )];
    sim_store
        .put_many_internal_locked(&target_col_id, &seed, None)
        .unwrap();

    let err = sim_store
        .create_or_upsert_established_validated(
            &target_col_id,
            &meta_colliding,
            &node_id,
            &data1,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));

    // 8. Unknown member namespace rejected on establishment
    let meta_unknown = CollectionMetadata {
        member_namespace: Some(*b"EDGE"),
        ..valid_meta.clone()
    };
    let err_unknown = sim_store
        .create_or_upsert_established_validated(
            &target_col_id,
            &meta_unknown,
            &node_id,
            &data1,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(err_unknown, StorageError::Internal(_)));
}

#[test]
fn packfile_storage_create_or_upsert_concurrent_writers() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        MEMBER_NAMESPACE_INTL,
    };

    let dir = test_dir("create_or_upsert_concurrency");
    let store = Arc::new(PackfileStorage::open(dir).unwrap());

    let canonical_id = b"sys:concurrent-upsert";
    let col_id = derive_collection_id(Some(MEMBER_NAMESPACE_INTL), canonical_id);

    let meta = CollectionMetadata {
        member_namespace: Some(MEMBER_NAMESPACE_INTL),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("system_auxiliary".to_owned()),
        schema: None,
    };

    let node_id = [0x77; 16];
    let barrier = Arc::new(std::sync::Barrier::new(2));

    // Two threads race to write distinct keys sharing the same physical node_id
    let t1 = {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        let meta = meta.clone();
        std::thread::spawn(move || {
            barrier.wait();
            let key_tag = b"key_alpha:";
            let data = NodeData::new(bytes::Bytes::from_static(b"key_alpha:payload_alpha"));
            store.create_or_upsert_established_validated(
                &col_id,
                &meta,
                &node_id,
                &data,
                &mut |existing| {
                    if let Some(existing) = existing {
                        if !existing.bytes.starts_with(key_tag) {
                            return Err(StorageError::Collision("key mismatch".into()));
                        }
                    }
                    Ok(())
                },
            )
        })
    };

    let t2 = {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        let meta = meta.clone();
        std::thread::spawn(move || {
            barrier.wait();
            let key_tag = b"key_beta:";
            let data = NodeData::new(bytes::Bytes::from_static(b"key_beta:payload_beta"));
            store.create_or_upsert_established_validated(
                &col_id,
                &meta,
                &node_id,
                &data,
                &mut |existing| {
                    if let Some(existing) = existing {
                        if !existing.bytes.starts_with(key_tag) {
                            return Err(StorageError::Collision("key mismatch".into()));
                        }
                    }
                    Ok(())
                },
            )
        })
    };

    let res1 = t1.join().unwrap();
    let res2 = t2.join().unwrap();

    // Exactly one thread must win, and the other must get Collision
    let (winner, _loser) = match (res1, res2) {
        (Ok(()), Err(StorageError::Collision(_))) => ("alpha", "beta"),
        (Err(StorageError::Collision(_)), Ok(())) => ("beta", "alpha"),
        (r1, r2) => panic!("unexpected outcome: r1={r1:?}, r2={r2:?}"),
    };

    let final_data = store.get(&col_id, &node_id).unwrap().unwrap();
    if winner == "alpha" {
        assert!(final_data.bytes.starts_with(b"key_alpha:"));
    } else {
        assert!(final_data.bytes.starts_with(b"key_beta:"));
    }
}
/// With a journal a sync only has to fsync the WAL, so pack data would pile
/// up until the checkpoint. Once enough is unsynced a sync also fsyncs the
/// packs, in slices; the budget can be turned off.
#[test]
fn a_journal_sync_fsyncs_packs_once_a_budget_of_them_is_unsynced() {
    for (budget, expect_fsync) in [(0_u64, false), (1024, true)] {
        let dir = test_dir(&format!("pack_budget_{budget}"));
        let wal = dir.join("budget.wal");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&wal).unwrap();
        let collection = [0x11u8; 16];
        // The first sync is the store's first checkpoint, which fsyncs every
        // pack whatever the budget; take it before measuring the delta path.
        store
            .put(
                &collection,
                &[0xFE; 16],
                &NodeData::new(bytes::Bytes::from_static(b"base")),
            )
            .unwrap();
        store.sync().unwrap();
        store.set_pack_fsync_budget(budget);
        for index in 0..64u8 {
            store
                .put(
                    &collection,
                    &[index; 16],
                    &NodeData::new(bytes::Bytes::from(vec![index; 200])),
                )
                .unwrap();
        }
        store.sync().unwrap();
        let timings = store.sync_timings().unwrap();
        assert_eq!(
            store.shards.unsynced_bytes() == 0,
            expect_fsync,
            "budget {budget}: packs synced or not"
        );
        assert_eq!(
            !timings.pack_fsync.is_zero(),
            expect_fsync,
            "budget {budget}"
        );
        drop(store);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// A checkpoint records journal coverage, so it must leave every pack byte
/// durable, whatever the incremental budget did before it.
#[test]
fn a_checkpoint_leaves_no_pack_bytes_unsynced_whatever_the_budget() {
    for budget in [0_u64, 1 << 30] {
        let dir = test_dir(&format!("checkpoint_syncs_packs_{budget}"));
        let wal = dir.join("checkpoint.wal");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&wal).unwrap();
        store.set_pack_fsync_budget(budget);
        let collection = [0x12u8; 16];
        for index in 0..32u8 {
            store
                .put(
                    &collection,
                    &[index; 16],
                    &NodeData::new(bytes::Bytes::from(vec![index; 300])),
                )
                .unwrap();
        }
        store.force_index_checkpoint().unwrap();
        assert_eq!(store.shards.unsynced_bytes(), 0, "budget {budget}");
        drop(store);
        let _ = fs::remove_dir_all(&dir);
    }
}
/// After the writer reclaims its journal on the strength of a delta coverage
/// batch (no new checkpoint), a reader that attaches sees every record that
/// is now only covered there: the reader loads the checkpoint, applies the
/// delta operations the coverage claim describes, and binds its coverage to
/// the claim, so nothing falls into the gap the reclaim left.
#[cfg(feature = "multi-reader")]
#[test]
fn a_reader_sees_records_covered_only_by_a_delta_coverage_batch() {
    let dir = test_dir("reader_delta_coverage");
    let wal = dir.join("wal.bin");
    let collection = [0x4Au8; 16];
    let key = |index: u8| [index; 16];

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    // Force the coverage step on every sync.
    writer.journal().unwrap().set_reclaim_trigger_len(1);
    writer
        .put(
            &collection,
            &key(1),
            &NodeData::new(bytes::Bytes::from_static(b"one")),
        )
        .unwrap();
    writer.sync().unwrap();
    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();

    for index in 2..=6u8 {
        writer
            .put(
                &collection,
                &key(index),
                &NodeData::new(bytes::Bytes::from(vec![index; 8])),
            )
            .unwrap();
        writer.sync().unwrap();
        let timings = writer.sync_timings().unwrap();
        assert!(
            timings.checkpoint.is_zero(),
            "sync {index} rewrote the checkpoint"
        );
        assert!(
            !timings.delta_log.is_zero(),
            "sync {index} wrote no coverage batch"
        );
    }
    assert!(
        writer.durable_coverage() > checkpoint.covered_lsn,
        "coverage must have advanced past the checkpoint's"
    );
    let base_lsn = Journal::scan_read_only(&wal).unwrap().base_lsn;
    assert!(
        base_lsn > checkpoint.covered_lsn.saturating_add(1),
        "the journal was reclaimed past the checkpoint's coverage"
    );

    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    reader.enable_read_journal(&wal).unwrap();
    let ids: Vec<NodeId> = (1..=6u8).map(key).collect();
    let values = reader.get_read_committed(&collection, &ids).unwrap();
    for (index, value) in values.iter().enumerate() {
        assert!(
            value.is_some(),
            "record {} is invisible to the reader",
            index + 1
        );
    }
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}
/// A reader that attached before the writer advanced coverage by delta
/// batch must reload correctly when the reclaim moves the journal past its
/// index: the reload applies the delta operations up to the last coverage
/// claim, records after it still come from the overlay, and a record the
/// writer overwrote in a covered batch is served at its newest value.
#[cfg(feature = "multi-reader")]
#[test]
fn a_reader_attached_earlier_reloads_through_the_delta_coverage_prefix() {
    let dir = test_dir("reader_delta_coverage_reload");
    let wal = dir.join("wal.bin");
    let collection = [0x4Cu8; 16];
    let key = |index: u8| [index; 16];
    let value = |bytes: &'static [u8]| NodeData::new(bytes::Bytes::from_static(bytes));

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer.journal().unwrap().set_reclaim_trigger_len(1);
    writer.put(&collection, &key(1), &value(b"one")).unwrap();
    writer.sync().unwrap();

    // The reader attaches now: its index is the first checkpoint.
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    reader.enable_read_journal(&wal).unwrap();
    assert_eq!(reader.read_covered_lsn.load(Ordering::Acquire), 1);

    // Coverage moves ahead by delta batches, overwriting key 1 on the way.
    writer.put(&collection, &key(2), &value(b"two")).unwrap();
    writer.put(&collection, &key(1), &value(b"one-v2")).unwrap();
    writer.sync().unwrap();
    writer.put(&collection, &key(3), &value(b"three")).unwrap();
    writer.sync().unwrap();
    assert!(writer.sync_timings().unwrap().checkpoint.is_zero());
    // A record after the last coverage batch, journalled and synced only in
    // the WAL: no coverage step is possible without new uncovered frames, so
    // stop the writer from taking one for it.
    writer.journal().unwrap().set_reclaim_trigger_len(u64::MAX);
    writer.put(&collection, &key(4), &value(b"four")).unwrap();
    writer.sync().unwrap();
    assert!(writer.sync_timings().unwrap().delta_log.as_nanos() > 0);

    let ids: Vec<NodeId> = (1..=4u8).map(key).collect();
    let values = reader.get_read_committed(&collection, &ids).unwrap();
    let payloads: Vec<Option<&[u8]>> = values
        .iter()
        .map(|value| value.as_ref().map(|data| data.bytes.as_ref()))
        .collect();
    assert_eq!(
        payloads,
        vec![
            Some(&b"one-v2"[..]),
            Some(&b"two"[..]),
            Some(&b"three"[..]),
            Some(&b"four"[..]),
        ]
    );
    assert!(
        reader.stats().read_reloads >= 1,
        "the reclaim must have forced a reload"
    );
    assert!(
        reader.read_covered_lsn.load(Ordering::Acquire) > 1,
        "the reload must have bound coverage to the delta claim"
    );
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}
/// A reader attached before a burst that grows the collection's table
/// several times reloads through the delta coverage prefix: it replays the
/// redo records with the same rules as an open, growing its own copy of the
/// table, and finds every record. The burst reaches the log as redo records
/// only, no snapshot.
#[cfg(feature = "multi-reader")]
#[test]
fn a_reader_reloads_through_a_burst_that_grows_the_table() {
    let dir = test_dir("reader_reload_growth");
    let wal = dir.join("wal.bin");
    let collection = [0x4D; 16];
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer.journal().unwrap().set_reclaim_trigger_len(1);
    writer
        .put(
            &collection,
            &batch_node(u32::MAX),
            &NodeData::from_slice(b"seed"),
        )
        .unwrap();
    writer.sync().unwrap();
    let capacity_before = writer.generation(&collection).unwrap().index.capacity();

    // The reader attaches now: its index is the small first checkpoint.
    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    reader.enable_read_journal(&wal).unwrap();

    let entries: Vec<(NodeId, NodeData)> = (0..700u32)
        .map(|i| (batch_node(i), NodeData::from_slice(&i.to_le_bytes())))
        .collect();
    writer.put_many(&collection, &entries).unwrap();
    writer.sync().unwrap();
    assert!(
        writer.generation(&collection).unwrap().index.capacity() > capacity_before * 4,
        "the burst must grow the table more than once"
    );
    assert!(
        writer.sync_timings().unwrap().checkpoint.is_zero(),
        "the burst is a delta step, not a checkpoint"
    );
    assert_log_has_redo_and_no_snapshot(&writer, entries.len());

    // One more record that stays only in the WAL (no coverage step for it):
    // the reader then sees a journal that starts past its index and must
    // reload through the delta prefix, so the burst's records are served
    // from a replayed (grown) table, not by refreshing on a miss.
    writer.journal().unwrap().set_reclaim_trigger_len(u64::MAX);
    writer
        .put(
            &collection,
            &batch_node(9_999),
            &NodeData::from_slice(b"tail"),
        )
        .unwrap();
    writer.sync().unwrap();
    let seed = reader
        .get_read_committed(&collection, &[batch_node(u32::MAX)])
        .unwrap();
    assert_eq!(seed[0].as_ref().unwrap().bytes.as_ref(), b"seed");
    let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
    let found = reader.get_read_committed(&collection, &ids).unwrap();
    for (index, ((_, expected), value)) in entries.iter().zip(found.iter()).enumerate() {
        let data = value
            .as_ref()
            .unwrap_or_else(|| panic!("record {index} of the burst is missing for the reader"));
        assert_eq!(data.bytes, expected.bytes);
    }
    let tail = reader
        .get_read_committed(&collection, &[batch_node(9_999)])
        .unwrap();
    assert_eq!(tail[0].as_ref().unwrap().bytes.as_ref(), b"tail");
    assert!(
        reader.stats().read_reloads >= 1,
        "the reclaim must have forced a reload"
    );
    assert_eq!(reader.stats().read_reload_failures, 0);
    assert_eq!(
        reader.stats().miss_refreshes,
        0,
        "every record came from the replayed table, none by refreshing on a miss"
    );
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}

/// A writer with a journal that syncs under the size trigger: every sync
/// past the first is a delta coverage step. Returns the store and the keys.
#[cfg(feature = "multi-reader")]
fn writer_with_coverage_steps(
    dir: &std::path::Path,
    wal: &std::path::Path,
    records: u8,
) -> PackfileStorage {
    let writer = PackfileStorage::open(dir.to_path_buf()).unwrap();
    writer.enable_journal(wal).unwrap();
    writer.journal().unwrap().set_reclaim_trigger_len(1);
    for index in 1..=records {
        writer
            .put(
                &[0x4Du8; 16],
                &[index; 16],
                &NodeData::new(bytes::Bytes::from(vec![index; 16])),
            )
            .unwrap();
        writer.sync().unwrap();
    }
    writer
}

/// Reopen a store with a journal and check every record is served.
#[cfg(feature = "multi-reader")]
fn reopen_and_read_all(
    dir: &std::path::Path,
    wal: &std::path::Path,
    records: u8,
) -> PackfileStorage {
    let reopened = PackfileStorage::open(dir.to_path_buf()).unwrap();
    reopened.enable_journal(wal).unwrap();
    reopened.replay_journal().unwrap();
    for index in 1..=records {
        let value = reopened.get(&[0x4Du8; 16], &[index; 16]).unwrap();
        assert_eq!(
            value.map(|data| data.bytes.to_vec()),
            Some(vec![index; 16]),
            "record {index} lost across the reopen"
        );
    }
    reopened
}

/// Crash after coverage steps and a reclaim that went past the checkpoint:
/// the reopened writer takes its coverage from the delta claim, replays only
/// what is uncovered, and loses nothing.
#[cfg(feature = "multi-reader")]
#[test]
fn a_writer_recovers_from_a_reclaim_licensed_by_a_delta_coverage_batch() {
    let dir = test_dir("recover_delta_coverage");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 6);
    let checkpoint_coverage =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap()
            .covered_lsn;
    let claimed = writer.durable_coverage();
    assert!(
        claimed > checkpoint_coverage,
        "the delta path must have advanced coverage"
    );
    assert!(
        Journal::scan_read_only(&wal).unwrap().base_lsn > checkpoint_coverage.saturating_add(1),
        "the journal must have been reclaimed past the checkpoint"
    );
    drop(writer); // no clean shutdown work is relied on: everything below is on disk

    assert_eq!(
        PackfileStorage::read_journal_lsn(&dir),
        claimed,
        "the disk must say what the writer claimed"
    );
    let reopened = reopen_and_read_all(&dir, &wal, 6);
    assert_eq!(reopened.durable_coverage(), claimed);
    drop(reopened);
    let _ = fs::remove_dir_all(&dir);
}

/// A coverage batch that is torn, or whose bytes are damaged, claims
/// nothing. The journal was reclaimed on the strength of it and cannot help,
/// but the packs it described are durable, so the reopen rebuilds the index
/// from them and every record is still there.
#[cfg(feature = "multi-reader")]
#[test]
fn a_torn_or_damaged_coverage_batch_loses_no_records() {
    let dir = test_dir("torn_delta_coverage");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 6);
    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    drop(writer);
    let delta_path = PackfileStorage::delta_path(&dir, checkpoint.fingerprint);
    let intact = fs::read(&delta_path).unwrap();
    assert!(
        delta::read_delta_log_v3(&delta_path)
            .unwrap()
            .coverage
            .is_some(),
        "the log must carry a coverage claim to lose"
    );

    // Cut the log inside its last batch, and separately flip a bit in it.
    let cut = intact.len() - 3;
    let mut damaged = intact.clone();
    let last = damaged.len() - 20;
    damaged[last] ^= 0x40;
    for (label, bytes) in [("torn", intact[..cut].to_vec()), ("bit flip", damaged)] {
        fs::write(&delta_path, &bytes).unwrap();
        let claimed = PackfileStorage::read_journal_lsn(&dir);
        let last_claim = delta::read_delta_log_v3(&delta_path)
            .and_then(|log| log.coverage)
            .unwrap_or(0);
        assert!(
            claimed == checkpoint.covered_lsn.max(last_claim),
            "{label}: coverage must come only from what still validates"
        );
        let reopened = reopen_and_read_all(&dir, &wal, 6);
        drop(reopened);
        // A reopen must not have kept the damaged log going.
        fs::write(&delta_path, &intact).unwrap();
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Without a coverage step (the journal is not over its trigger) nothing is
/// claimed beyond the checkpoint, the journal keeps its groups, and a crash
/// replays them: the behaviour before coverage batches existed.
#[cfg(feature = "multi-reader")]
#[test]
fn a_sync_under_the_trigger_claims_no_coverage() {
    let dir = test_dir("no_coverage_under_trigger");
    let wal = dir.join("wal.bin");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &[0x4Du8; 16],
            &[1; 16],
            &NodeData::new(bytes::Bytes::from(vec![1; 16])),
        )
        .unwrap();
    writer.sync().unwrap();
    let after_checkpoint = writer.durable_coverage();
    for index in 2..=4u8 {
        writer
            .put(
                &[0x4Du8; 16],
                &[index; 16],
                &NodeData::new(bytes::Bytes::from(vec![index; 16])),
            )
            .unwrap();
        writer.sync().unwrap();
        let timings = writer.sync_timings().unwrap();
        assert!(timings.checkpoint.is_zero());
        assert!(!timings.delta_log.is_zero(), "an ordinary delta batch");
    }
    assert_eq!(writer.durable_coverage(), after_checkpoint);
    let groups = Journal::scan_read_only(&wal).unwrap().groups.len();
    assert!(
        groups >= 3,
        "the journal keeps the uncovered groups ({groups})"
    );
    drop(writer);
    drop(reopen_and_read_all(&dir, &wal, 4));
    let _ = fs::remove_dir_all(&dir);
}

/// The one-pass durable state gives what the two separate readers gave: the
/// coverage `read_journal_lsn` reports and the fingerprint
/// `read_durable_fingerprint` reports, with and without a coverage claim in
/// the delta log.
#[cfg(feature = "multi-reader")]
#[test]
fn the_one_pass_durable_state_matches_the_separate_readers() {
    let dir = test_dir("one_pass_durable_state");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 5);
    drop(writer);
    let state = PackfileStorage::durable_index_state(&dir);
    let fingerprint = crate::index::checkpoint::read_durable_fingerprint(&dir)
        .unwrap()
        .expect("a durable fingerprint")
        .fingerprint;
    assert_eq!(state.fingerprint, fingerprint);
    assert!(state.covered_lsn > 0, "the log must carry a claim");
    assert_eq!(
        PackfileStorage::read_journal_lsn(&dir),
        state
            .covered_lsn
            .max(PackfileStorage::journal_lsn_with(&dir, 0)),
    );
    // No log for the checkpoint: the fingerprint is the checkpoint's own.
    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    fs::remove_file(PackfileStorage::delta_path(&dir, checkpoint.fingerprint)).unwrap();
    let bare = PackfileStorage::durable_index_state(&dir);
    assert_eq!(bare.fingerprint, checkpoint.fingerprint);
    assert_eq!(bare.covered_lsn, checkpoint.covered_lsn);
    let _ = fs::remove_dir_all(&dir);
}

/// The one-pass state also matches the separate readers when the log names
/// another checkpoint, and when its tail is torn.
#[cfg(feature = "multi-reader")]
#[test]
fn the_one_pass_durable_state_matches_for_a_foreign_or_torn_log() {
    let dir = test_dir("one_pass_foreign_torn");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 5);
    drop(writer);
    let checkpoint =
        crate::index::checkpoint::read_checkpoint(&PackfileStorage::index_checkpoint_path(&dir))
            .unwrap();
    let log_path = PackfileStorage::delta_path(&dir, checkpoint.fingerprint);
    let intact = fs::read(&log_path).unwrap();

    // Torn: cut inside the last batch. The fingerprint is whatever the last
    // intact batch ended at, exactly as `read_durable_fingerprint` says.
    fs::write(&log_path, &intact[..intact.len() - 3]).unwrap();
    let torn = PackfileStorage::durable_index_state(&dir);
    let reference = crate::index::checkpoint::read_durable_fingerprint(&dir)
        .unwrap()
        .expect("a fingerprint");
    assert!(reference.torn_tail);
    assert_eq!(torn.fingerprint, reference.fingerprint);
    let tail = delta::read_delta_tail_fingerprint(&log_path)
        .unwrap()
        .unwrap();
    assert_eq!(
        torn.covered_lsn,
        checkpoint.covered_lsn.max(tail.coverage.unwrap_or(0))
    );

    // Foreign: a log that continues a different checkpoint is inert.
    fs::remove_file(&log_path).unwrap();
    delta::append_v3_batch_with_durability(
        &log_path,
        true,
        checkpoint.fingerprint ^ 1,
        &[delta::DeltaOperation::Coverage {
            covered_lsn: u64::MAX,
        }],
        0x99,
        false,
    )
    .unwrap();
    let foreign = PackfileStorage::durable_index_state(&dir);
    assert_eq!(foreign.fingerprint, checkpoint.fingerprint);
    assert_eq!(foreign.covered_lsn, checkpoint.covered_lsn);
    let _ = fs::remove_dir_all(&dir);
}

/// The redo ordering key survives a checkpoint and a reopen: the counter
/// resumes at the base sequence the checkpoint recorded.
#[test]
fn the_delta_sequence_resumes_from_the_checkpoints_base_after_a_reopen() {
    let dir = test_dir("delta_seq_reopen");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(&[0xD7; 16], &batch_node(1), &NodeData::from_slice(b"one"))
        .unwrap();
    store.delta_state.lock().delta_seq_high = 17;
    store.sync_all().unwrap();
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(reopened.delta_state.lock().delta_seq_high, 17);
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// Puts after a checkpoint are recorded as logical redo records with
/// increasing sequence numbers, and a reopen replays them in order: an
/// overwrite of an identity converges to its last locator, and the counter
/// resumes above the last sequence in the log.
#[test]
fn logical_redo_replays_in_order_and_the_counter_resumes_above_the_log() {
    let dir = test_dir("logical_redo_order");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0xD8; 16];
    store
        .put(&collection, &batch_node(1), &NodeData::from_slice(b"first"))
        .unwrap();
    store.sync_all().unwrap();
    store
        .put(
            &collection,
            &batch_node(1),
            &NodeData::from_slice(b"second"),
        )
        .unwrap();
    store
        .put(&collection, &batch_node(2), &NodeData::from_slice(b"other"))
        .unwrap();
    let seqs: Vec<u64> = match &store.delta_state.lock().pending[&collection] {
        PendingDelta::Redo(records) => records.iter().map(|record| record.delta_seq).collect(),
        _ => panic!("puts after a checkpoint are recorded as redo records"),
    };
    assert_eq!(seqs.len(), 2);
    assert!(seqs[0] < seqs[1], "sequence order is append order");
    store.sync_all().unwrap();
    drop(store);

    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(reopened.open_timings().unwrap().path, OpenPath::Checkpoint);
    assert_eq!(
        reopened
            .get(&collection, &batch_node(1))
            .unwrap()
            .unwrap()
            .bytes,
        bytes::Bytes::from_static(b"second"),
        "the overwrite converges to its last locator"
    );
    assert!(reopened.get(&collection, &batch_node(2)).unwrap().is_some());
    assert!(reopened.delta_state.lock().delta_seq_high >= seqs[1]);
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// A redo log that names an unknown pack, a locator outside its pack, or a
/// sequence that does not increase is rejected, and the open falls back to a
/// rescan that still finds every record. The unmodified log is trusted.
#[test]
fn a_bad_redo_log_is_rejected_and_the_open_rescans() {
    fn open_after(mutate: impl Fn(&mut RedoRecord)) -> (OpenPath, usize) {
        let dir = test_dir("bad_redo_log");
        let collection = [0xD9; 16];
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(&collection, &batch_node(0), &NodeData::from_slice(b"base"))
            .unwrap();
        store.sync_all().unwrap();
        for id in 1..=5 {
            store
                .put(&collection, &batch_node(id), &NodeData::from_slice(b"redo"))
                .unwrap();
        }
        store.sync_all().unwrap();
        let base = store.delta_state.lock().base_fingerprint.unwrap();
        drop(store);

        let path = PackfileStorage::delta_path(&dir, base);
        let log = delta::read_delta_log_v3(&path).unwrap();
        let mut operations = log.operations;
        let mut mutated = 0;
        for operation in &mut operations {
            if let DeltaOperation::Redo(record) = operation {
                mutate(record);
                mutated += 1;
            }
        }
        assert!(mutated > 0, "the log must hold redo records");
        fs::remove_file(&path).unwrap();
        delta::append_v3_batch_with_durability(
            &path,
            true,
            log.base_fingerprint,
            &operations,
            log.tail_fingerprint,
            false,
        )
        .unwrap();

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let path_taken = reopened.open_timings().unwrap().path;
        let found = (0..=5)
            .filter(|id| {
                reopened
                    .get(&collection, &batch_node(*id))
                    .unwrap()
                    .is_some()
            })
            .count();
        drop(reopened);
        fs::remove_dir_all(dir).unwrap();
        (path_taken, found)
    }

    assert_eq!(open_after(|_| {}), (OpenPath::Checkpoint, 6), "control");
    let unknown_pack = |record: &mut RedoRecord| {
        if let RedoOp::Set { pack_id, .. } = &mut record.op {
            *pack_id = PackId([0xEE; crate::packfile::PACK_ID_LEN]);
        }
    };
    let out_of_extent = |record: &mut RedoRecord| {
        if let RedoOp::Set { offset, .. } = &mut record.op {
            *offset = crate::index::IndexEntry::MAX_OFFSET;
        }
    };
    let flat_sequence = |record: &mut RedoRecord| record.delta_seq = 0;
    assert_eq!(open_after(unknown_pack), (OpenPath::FullScan, 6));
    assert_eq!(open_after(out_of_extent), (OpenPath::FullScan, 6));
    assert_eq!(open_after(flat_sequence), (OpenPath::FullScan, 6));
}

/// Differential: a collection driven through many batches, overwrites and
/// several capacity growths after its checkpoint, synced only as deltas, must
/// reopen to exactly what the live session held: every identity at its last
/// value and the same number of entries. Growth is logical, so the log holds
/// redo records only. Repeated across two reopens so a log continues a log.
#[test]
fn logical_replay_across_growth_matches_the_live_index() {
    let dir = test_dir("logical_replay_differential");
    let collection = [0xDA; 16];
    let mut expected: HashMap<NodeId, Vec<u8>> = HashMap::new();
    let mut next = 0u32;
    let mut round_value = 0u8;
    let mut drive = |store: &PackfileStorage, expected: &mut HashMap<NodeId, Vec<u8>>| {
        round_value = round_value.wrapping_add(1);
        let mut batch = Vec::new();
        // New identities (enough over the rounds to cross several growths).
        for _ in 0..400 {
            batch.push((batch_node(next), vec![round_value; 8]));
            next += 1;
        }
        // Overwrites of earlier identities with a new value.
        for old in (0..next.saturating_sub(400)).step_by(37) {
            batch.push((batch_node(old), vec![round_value; 8]));
        }
        let entries: Vec<(NodeId, NodeData)> = batch
            .iter()
            .map(|(id, value)| (*id, NodeData::from_slice(value)))
            .collect();
        store.put_many(&collection, &entries).unwrap();
        for (id, value) in batch {
            expected.insert(id, value);
        }
        // A few single puts too (the other recording path).
        for _ in 0..5 {
            store
                .put(
                    &collection,
                    &batch_node(next),
                    &NodeData::from_slice(&[round_value; 8]),
                )
                .unwrap();
            expected.insert(batch_node(next), vec![round_value; 8]);
            next += 1;
        }
        store.sync_all().unwrap();
    };
    let verify = |store: &PackfileStorage, expected: &HashMap<NodeId, Vec<u8>>| {
        for (id, value) in expected {
            assert_eq!(
                store.get(&collection, id).unwrap().unwrap().bytes.as_ref(),
                value.as_slice(),
                "a record diverged from the live session"
            );
        }
        assert_eq!(
            store.generation(&collection).unwrap().index.len(),
            expected.len(),
            "the index holds a different number of entries"
        );
    };

    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(
            &collection,
            &batch_node(u32::MAX),
            &NodeData::from_slice(b"seed"),
        )
        .unwrap();
    expected.insert(batch_node(u32::MAX), b"seed".to_vec());
    store.sync_all().unwrap(); // the checkpoint every later batch continues
    let grows_before = store.stats().index_grow_count;
    for _ in 0..8 {
        drive(&store, &mut expected);
    }
    assert!(
        store.stats().index_grow_count > grows_before,
        "the workload must cross capacity growths after the checkpoint"
    );
    drop(store);

    // Reopen 1: replays a log of redo records, then keeps appending to it.
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(reopened.open_timings().unwrap().path, OpenPath::Checkpoint);
    verify(&reopened, &expected);
    assert_log_has_redo_and_no_snapshot(&reopened, 1);
    for _ in 0..4 {
        drive(&reopened, &mut expected);
    }
    drop(reopened);

    // Reopen 2: a log that continued a replayed log.
    let again = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(again.open_timings().unwrap().path, OpenPath::Checkpoint);
    verify(&again, &expected);
    drop(again);
    fs::remove_dir_all(dir).unwrap();
}

/// Applying redo sets is idempotent and order-convergent: replaying the whole
/// sequence twice, or over an image that already holds a prefix of it, gives
/// the same identities and locators as applying it once, with the last write
/// of an identity winning.
#[test]
fn applying_redo_sets_is_idempotent_and_order_convergent() {
    let config = crate::index::IndexConfig {
        seed: 0x1D3A,
        ..crate::index::IndexConfig::default()
    };
    let pack_lookup: HashMap<PackId, (u16, u64)> =
        HashMap::from([(PackId([7; crate::packfile::PACK_ID_LEN]), (2, 1 << 20))]);
    let hash = |seq: u32| {
        let mut hash = [0u8; 16];
        hash[..4].copy_from_slice(&seq.to_le_bytes());
        hash[4..8].copy_from_slice(&seq.wrapping_mul(0x9E37_79B1).to_le_bytes());
        hash[8..12].copy_from_slice(&seq.wrapping_mul(0x85EB_CA6B).to_le_bytes());
        hash[12..].copy_from_slice(&seq.wrapping_mul(0xC2B2_AE35).to_le_bytes());
        hash
    };
    // 300 distinct identities, then overwrites of every third to new locators.
    let mut records = Vec::new();
    let mut seq = 0u64;
    for id in 0..300u32 {
        seq += 1;
        records.push(RedoRecord {
            collection_id: [1; 16],
            op: RedoOp::Set {
                full_hash: hash(id),
                pack_id: PackId([7; crate::packfile::PACK_ID_LEN]),
                offset: 1_000 + u64::from(id) * 64,
                record_len: 61,
            },
            delta_seq: seq,
            base_generation: 1,
        });
    }
    for id in (0..300u32).step_by(3) {
        seq += 1;
        records.push(RedoRecord {
            collection_id: [1; 16],
            op: RedoOp::Set {
                full_hash: hash(id),
                pack_id: PackId([7; crate::packfile::PACK_ID_LEN]),
                offset: 500_000 + u64::from(id) * 64,
                record_len: 61,
            },
            delta_seq: seq,
            base_generation: 1,
        });
    }
    let fresh = || LossyIndex::with_config(16, config);
    let once = PackfileStorage::apply_redo_sets(fresh(), &records, &pack_lookup).unwrap();
    let twice = PackfileStorage::apply_redo_sets(once.clone(), &records, &pack_lookup).unwrap();
    let over_prefix = {
        let prefix =
            PackfileStorage::apply_redo_sets(fresh(), &records[..350], &pack_lookup).unwrap();
        PackfileStorage::apply_redo_sets(prefix, &records, &pack_lookup).unwrap()
    };
    assert_eq!(once.len(), 300, "300 distinct identities");
    for candidate in [&twice, &over_prefix] {
        assert_eq!(candidate.len(), once.len());
        assert_eq!(candidate.slot_counts(), once.slot_counts());
    }
    for id in 0..300u32 {
        let expected_offset = if id % 3 == 0 {
            500_000 + u64::from(id) * 64
        } else {
            1_000 + u64::from(id) * 64
        };
        for index in [&once, &twice, &over_prefix] {
            assert_eq!(
                index.lookup(&hash(id)),
                Some((2, expected_offset)),
                "identity {id}: the last write wins"
            );
        }
    }
}

/// A replayed record whose locator is wrong but in bounds (it points at
/// another valid record of the same pack) would install and then read as a
/// missing key. Replay verifies each locator's frame, so the log is rejected
/// and the open rescans, finding every record.
#[test]
fn a_wrong_but_in_bounds_locator_rejects_the_log_and_the_open_rescans() {
    let dir = test_dir("wrong_locator");
    let collection = [0xDB; 16];
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store
        .put(&collection, &batch_node(0), &NodeData::from_slice(b"base"))
        .unwrap();
    store.sync_all().unwrap();
    for id in 1..=5 {
        store
            .put(&collection, &batch_node(id), &NodeData::from_slice(&[7; 8]))
            .unwrap();
    }
    store.sync_all().unwrap();
    let base = store.delta_state.lock().base_fingerprint.unwrap();
    drop(store);
    let path = PackfileStorage::delta_path(&dir, base);
    let log = delta::read_delta_log_v3(&path).unwrap();
    let mut operations = log.operations;
    let (donor_offset, donor_len) = operations
        .iter()
        .find_map(|operation| match operation {
            DeltaOperation::Redo(RedoRecord {
                op: RedoOp::Set {
                    offset, record_len, ..
                },
                ..
            }) => Some((*offset, *record_len)),
            _ => None,
        })
        .unwrap();
    // Point the last record (id 5) at the first record's frame.
    if let Some(DeltaOperation::Redo(record)) = operations
        .iter_mut()
        .filter(|operation| matches!(operation, DeltaOperation::Redo(_)))
        .last()
    {
        if let RedoOp::Set {
            offset, record_len, ..
        } = &mut record.op
        {
            *offset = donor_offset;
            *record_len = donor_len;
        }
    }
    fs::remove_file(&path).unwrap();
    delta::append_v3_batch_with_durability(
        &path,
        true,
        log.base_fingerprint,
        &operations,
        log.tail_fingerprint,
        false,
    )
    .unwrap();
    let reopened = PackfileStorage::open(dir.clone()).unwrap();
    assert_eq!(
        reopened.open_timings().unwrap().path,
        OpenPath::FullScan,
        "the wrong locator must be caught at open, not at the first read"
    );
    for id in 0..=5 {
        assert!(
            reopened
                .get(&collection, &batch_node(id))
                .unwrap()
                .is_some(),
            "record {id} must still be found after the rescan"
        );
    }
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// Counts and pack identity across several packs: with small packs, growth,
/// overwrites and new packs created after the checkpoint (which the redo log
/// names by pack id), a reopen holds exactly the live session's records and
/// per-collection entry counts, on the checkpoint path.
#[test]
fn logical_replay_across_several_packs_keeps_counts_and_records() {
    let dir = test_dir("logical_replay_packs");
    let collection = [0xDC; 16];
    let open = || PackfileStorage::open_with_max_shard_bytes(dir.clone(), 48 * 1024).unwrap();
    let store = open();
    store
        .put(
            &collection,
            &batch_node(u32::MAX),
            &NodeData::from_slice(b"seed"),
        )
        .unwrap();
    store.sync_all().unwrap();
    let mut expected: HashMap<NodeId, Vec<u8>> = HashMap::new();
    expected.insert(batch_node(u32::MAX), b"seed".to_vec());
    for round in 1..=6u8 {
        let entries: Vec<(NodeId, NodeData)> = (0..300u32)
            .map(|i| {
                let id = batch_node(u32::from(round) * 1000 + i);
                (id, NodeData::from_slice(&[round; 40]))
            })
            .chain((0..50u32).map(|i| (batch_node(i * 3), NodeData::from_slice(&[round; 40]))))
            .collect();
        for (id, data) in &entries {
            expected.insert(*id, data.bytes.to_vec());
        }
        store.put_many(&collection, &entries).unwrap();
        store.sync_all().unwrap();
    }
    let live_summary = store.collection_summaries();
    let live_packs = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "pack"))
        .count();
    assert!(
        live_packs > 2,
        "the workload must span several packs ({live_packs})"
    );
    drop(store);

    let reopened = open();
    assert_eq!(reopened.open_timings().unwrap().path, OpenPath::Checkpoint);
    assert_eq!(
        reopened
            .collection_summaries()
            .iter()
            .map(|summary| (summary.0, summary.1))
            .collect::<Vec<_>>(),
        live_summary
            .iter()
            .map(|summary| (summary.0, summary.1))
            .collect::<Vec<_>>(),
        "per-collection entry counts"
    );
    for (id, value) in &expected {
        assert_eq!(
            reopened
                .get(&collection, id)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            value.as_slice()
        );
    }
    drop(reopened);
    fs::remove_dir_all(dir).unwrap();
}

/// The coverage step obeys the same rotation: with the log at the rotation
/// length, a sync over the trigger takes the checkpoint rather than a batch.
#[cfg(feature = "multi-reader")]
#[test]
fn a_coverage_step_is_replaced_by_a_checkpoint_at_the_rotation_length() {
    let dir = test_dir("coverage_rotation");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 3);
    assert!(!writer.sync_timings().unwrap().delta_log.is_zero());
    let at = writer.delta_log_rotate_bytes();
    writer.delta_state.lock().log_bytes = at;
    writer
        .put(
            &[0x4Du8; 16],
            &[9; 16],
            &NodeData::new(bytes::Bytes::from(vec![9; 16])),
        )
        .unwrap();
    writer.sync().unwrap();
    let timings = writer.sync_timings().unwrap();
    assert!(!timings.checkpoint.is_zero(), "a full checkpoint ran");
    assert!(timings.delta_log.is_zero(), "no batch was appended");
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}

/// The coverage step is only taken while every live pack is in the base
/// checkpoint's pack table, since a reader translates the delta operations'
/// slots through it. A pack created since forces a full checkpoint, which
/// records the new table and starts a new epoch.
#[cfg(feature = "multi-reader")]
#[test]
fn a_pack_missing_from_the_checkpoint_table_forces_a_full_checkpoint() {
    let dir = test_dir("coverage_pack_gate");
    let wal = dir.join("wal.bin");
    let writer = writer_with_coverage_steps(&dir, &wal, 3);
    // With the table intact the step is a delta batch.
    writer
        .put(
            &[0x4Du8; 16],
            &[9; 16],
            &NodeData::new(bytes::Bytes::from(vec![9; 16])),
        )
        .unwrap();
    writer.sync().unwrap();
    assert!(writer.sync_timings().unwrap().checkpoint.is_zero());
    assert!(writer.packs_match_checkpoint_table());
    writer.set_checkpoint_rewrite_budget(std::time::Duration::from_secs(3600), 0);
    // Forget the table, as if a pack had been created since the checkpoint.
    writer.delta_state.lock().base_pack_ids.clear();
    assert!(!writer.packs_match_checkpoint_table());
    writer
        .put(
            &[0x4Du8; 16],
            &[10; 16],
            &NodeData::new(bytes::Bytes::from(vec![10; 16])),
        )
        .unwrap();
    writer.sync().unwrap();
    let timings = writer.sync_timings().unwrap();
    assert!(
        !timings.checkpoint.is_zero(),
        "a full checkpoint re-bases the epoch"
    );
    assert!(
        writer.packs_match_checkpoint_table(),
        "and records the new table"
    );
    drop(writer);
    drop(reopen_and_read_all(&dir, &wal, 3));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_yields_each_live_key_once() {
    let dir = test_dir("scan_collection_basic");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x5au8; 16];
    for value in 0..32u8 {
        store
            .put(
                &collection,
                &distinct_id(value),
                &NodeData::new(bytes::Bytes::from(vec![value])),
            )
            .unwrap();
    }
    store.sync().unwrap();

    let mut scanned: Vec<(NodeId, u8)> = store
        .scan_collection(&collection)
        .unwrap()
        .map(|entry| {
            let (id, data) = entry.unwrap();
            (id, data.bytes[0])
        })
        .collect();
    scanned.sort_unstable_by_key(|(id, _)| *id);
    assert_eq!(scanned.len(), 32);
    for (position, (id, value)) in scanned.iter().enumerate() {
        let expected = u8::try_from(position).expect("32 records fit in u8");
        assert_eq!(*id, distinct_id(expected));
        assert_eq!(*value, expected);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_skips_superseded_versions() {
    let dir = test_dir("scan_collection_superseded");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x6bu8; 16];
    let key = distinct_id(7);
    store
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"old")),
        )
        .unwrap();
    store.sync().unwrap();
    store
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"new")),
        )
        .unwrap();
    store.sync().unwrap();

    let scanned: Vec<(NodeId, bytes::Bytes)> = store
        .scan_collection(&collection)
        .unwrap()
        .map(|entry| {
            let (id, data) = entry.unwrap();
            (id, data.bytes)
        })
        .collect();
    assert_eq!(scanned.len(), 1, "each key is yielded once");
    assert_eq!(scanned[0].0, key);
    assert_eq!(scanned[0].1.as_ref(), b"new", "the live version wins");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_isolates_the_requested_collection() {
    let dir = test_dir("scan_collection_isolated");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let wanted = [0x81u8; 16];
    let other = [0x82u8; 16];
    store
        .put(
            &wanted,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"a")),
        )
        .unwrap();
    store
        .put(
            &other,
            &distinct_id(2),
            &NodeData::new(bytes::Bytes::from_static(b"b")),
        )
        .unwrap();
    store.sync().unwrap();

    let scanned: Vec<NodeId> = store
        .scan_collection(&wanted)
        .unwrap()
        .map(|entry| entry.unwrap().0)
        .collect();
    assert_eq!(scanned, vec![distinct_id(1)]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_is_empty_after_delete() {
    let dir = test_dir("scan_collection_deleted");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x7cu8; 16];
    store
        .put(
            &collection,
            &distinct_id(1),
            &NodeData::new(bytes::Bytes::from_static(b"x")),
        )
        .unwrap();
    store.sync().unwrap();
    store.delete_collection(&collection).unwrap();
    store.sync().unwrap();

    assert_eq!(
        store.scan_collection(&collection).unwrap().count(),
        0,
        "a deleted collection scans empty"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_excludes_a_post_snapshot_append() {
    let dir = test_dir("scan_collection_boundary");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x91u8; 16];
    let before = distinct_id(1);
    let after = distinct_id(2);
    store
        .put(
            &collection,
            &before,
            &NodeData::new(bytes::Bytes::from_static(b"before")),
        )
        .unwrap();
    store.sync().unwrap();

    // The boundary is fixed when `scan_collection` is called, so a record
    // appended and synced before the iterator is drained must not appear.
    let scan = store.scan_collection(&collection).unwrap();
    store
        .put(
            &collection,
            &after,
            &NodeData::new(bytes::Bytes::from_static(b"after")),
        )
        .unwrap();
    store.sync().unwrap();

    let scanned: Vec<NodeId> = scan.map(|entry| entry.unwrap().0).collect();
    assert_eq!(
        scanned,
        vec![before],
        "the post-boundary append is excluded"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_keeps_a_pre_snapshot_overwrite() {
    let dir = test_dir("scan_collection_overwrite_boundary");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0x94u8; 16];
    let key = distinct_id(1);
    store
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"before")),
        )
        .unwrap();
    store.sync().unwrap();

    // The index snapshot is captured while scan_collection holds the put
    // lock. An overwrite after it returns must not change the lazy result.
    let scan = store.scan_collection(&collection).unwrap();
    store
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"after")),
        )
        .unwrap();
    store.sync().unwrap();

    let scanned: Vec<(NodeId, bytes::Bytes)> = scan
        .map(|entry| {
            let (id, data) = entry.unwrap();
            (id, data.bytes)
        })
        .collect();
    assert_eq!(scanned.len(), 1);
    assert_eq!(scanned[0].0, key);
    assert_eq!(scanned[0].1.as_ref(), b"before");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_at_snapshot_replays_later_durable_groups() {
    let dir = test_dir("scan_collection_replay_boundary");
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();
    let collection = [0x92u8; 16];
    let before = distinct_id(1);
    let after = distinct_id(2);

    store
        .put(
            &collection,
            &before,
            &NodeData::new(bytes::Bytes::from_static(b"before")),
        )
        .unwrap();
    let (scan, cursor, _lease) = store.scan_collection_at_snapshot(&collection).unwrap();

    store
        .put(
            &collection,
            &after,
            &NodeData::new(bytes::Bytes::from_static(b"after")),
        )
        .unwrap();
    let journal = store.journal().unwrap();
    journal.sync().unwrap();

    let scanned: Vec<NodeId> = scan.map(|entry| entry.unwrap().0).collect();
    assert_eq!(scanned, vec![before], "scan is fixed at its cursor");

    let changes = journal.changes_since(&cursor, 16).unwrap();
    let replayed: Vec<NodeId> = changes
        .groups
        .iter()
        .flat_map(|group| group.entries.iter())
        .filter_map(|entry| match &entry.mutation {
            JournalMutation::Put {
                collection_id,
                node_id,
                ..
            } if *collection_id == collection => Some(*node_id),
            _ => None,
        })
        .collect();
    assert_eq!(replayed, vec![after]);
    assert!(!changes.has_more);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_at_snapshot_requires_a_journal() {
    let dir = test_dir("scan_collection_replay_without_wal");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let Err(error) = store.scan_collection_at_snapshot(&[0x93u8; 16]) else {
        panic!("replayable scans require a journal");
    };
    assert!(error.is_unsupported(), "expected Unsupported, got {error}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collections_at_snapshot_binds_one_boundary() {
    let dir = test_dir("scan_collections_one_boundary");
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();

    let collection_a = [0xa1u8; 16];
    let collection_b = [0xb2u8; 16];
    let a_before = distinct_id(0x11);
    let b_before = distinct_id(0x22);
    let a_after = distinct_id(0x33);
    let b_after = distinct_id(0x44);

    store
        .put(
            &collection_a,
            &a_before,
            &NodeData::new(bytes::Bytes::from_static(b"a-before")),
        )
        .unwrap();
    store
        .put(
            &collection_b,
            &b_before,
            &NodeData::new(bytes::Bytes::from_static(b"b-before")),
        )
        .unwrap();

    // Passed in reverse order: the result is ordered by collection id, and
    // each scan is paired with the collection it came from.
    let (scans, cursor, _lease) = store
        .scan_collections_at_snapshot(&[collection_b, collection_a])
        .unwrap();
    assert_eq!(scans.len(), 2);
    assert_eq!(scans[0].0, collection_a);
    assert_eq!(scans[1].0, collection_b);

    store
        .put(
            &collection_a,
            &a_after,
            &NodeData::new(bytes::Bytes::from_static(b"a-after")),
        )
        .unwrap();
    store
        .put(
            &collection_b,
            &b_after,
            &NodeData::new(bytes::Bytes::from_static(b"b-after")),
        )
        .unwrap();
    let journal = store.journal().unwrap();
    journal.sync().unwrap();

    let mut scans = scans.into_iter();
    let (_, scan_a) = scans.next().unwrap();
    let (_, scan_b) = scans.next().unwrap();
    let scanned_a: Vec<NodeId> = scan_a.map(|entry| entry.unwrap().0).collect();
    let scanned_b: Vec<NodeId> = scan_b.map(|entry| entry.unwrap().0).collect();
    assert_eq!(
        scanned_a,
        vec![a_before],
        "every scan is fixed at the one shared cursor"
    );
    assert_eq!(scanned_b, vec![b_before]);

    let changes = journal.changes_since(&cursor, 16).unwrap();
    let replayed: Vec<NodeId> = changes
        .groups
        .iter()
        .flat_map(|group| group.entries.iter())
        .filter_map(|entry| match &entry.mutation {
            JournalMutation::Put { node_id, .. } => Some(*node_id),
            JournalMutation::DeleteCollection { .. } => None,
        })
        .collect();
    assert_eq!(replayed, vec![a_after, b_after]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collections_at_snapshot_deduplicates_collections() {
    let dir = test_dir("scan_collections_dedup");
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();

    let collection = [0xc3u8; 16];
    let (scans, _cursor, _lease) = store
        .scan_collections_at_snapshot(&[collection, collection, collection])
        .unwrap();
    assert_eq!(scans.len(), 1, "duplicate ids collapse to one scan");
    assert_eq!(scans[0].0, collection);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collections_at_snapshot_requires_a_journal() {
    let dir = test_dir("scan_collections_without_wal");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let Err(error) = store.scan_collections_at_snapshot(&[[0x93u8; 16]]) else {
        panic!("replayable scans require a journal");
    };
    assert!(error.is_unsupported(), "expected Unsupported, got {error}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn scan_collection_at_snapshot_excludes_the_read_journal_overlay() {
    let dir = test_dir("scan_collection_replay_with_reader_overlay");
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0x96),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    store.sync().unwrap();

    // This is normally only installed on a read-only worker. The replayable
    // scan is pack-only, so an overlay mutation cannot be returned both here
    // and by changes_since.
    store.enable_read_journal(&wal).unwrap();
    let later = distinct_id(0x97);
    store
        .put(
            &TEST_COLLECTION,
            &later,
            &NodeData::new(bytes::Bytes::from_static(b"later")),
        )
        .unwrap();

    let (scan, cursor, _lease) = store.scan_collection_at_snapshot(&TEST_COLLECTION).unwrap();
    let scanned: Vec<NodeId> = scan.map(|entry| entry.unwrap().0).collect();
    assert_eq!(scanned, vec![distinct_id(0x96), later]);
    let changes = store.journal().unwrap().changes_since(&cursor, 8).unwrap();
    assert!(
        changes.groups.is_empty(),
        "scanned records are not replayed"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "multi-reader")]
#[test]
fn replayable_scan_does_not_hold_transaction_lifecycle_lock_during_pack_walk() {
    use crate::layout::ShardType;
    use std::sync::mpsc;
    use std::sync::Arc;

    let dir = test_dir("scan_collection_lifecycle_window");
    let wal = dir.join("wal.bin");
    let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    store.enable_journal(&wal).unwrap();
    store
        .put(
            &TEST_COLLECTION,
            &distinct_id(0x98),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();

    let (parked_tx, parked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = std::sync::Mutex::new(release_rx);
    *store.replay_snapshot_hook.lock() = Some(Arc::new(move || {
        parked_tx.send(()).ok();
        release_rx.lock().unwrap().recv().ok();
    }));

    let scan_store = Arc::clone(&store);
    let scanner = std::thread::spawn(move || {
        let (scan, cursor, lease) = scan_store
            .scan_collection_at_snapshot(&TEST_COLLECTION)
            .unwrap();
        (
            scan.map(|entry| entry.unwrap().0).collect::<Vec<_>>(),
            cursor,
            lease,
        )
    });
    parked_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("snapshot should reach WAL sync");

    let (activated_tx, activated_rx) = mpsc::channel();
    let activate_store = Arc::clone(&store);
    let post_boundary_id = distinct_id(0x99);
    let activator = std::thread::spawn(move || {
        if activate_store
            .activate_transaction_overlay(&wal, ShardType::State)
            .is_err()
        {
            activated_tx.send(false).unwrap();
            return;
        }
        activated_tx.send(true).unwrap();
        activate_store
            .put(
                &TEST_COLLECTION,
                &post_boundary_id,
                &NodeData::new(bytes::Bytes::from_static(b"after boundary")),
            )
            .unwrap();
        activate_store.deactivate_transaction_overlay();
    });
    let activated = activated_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap_or(false);

    release_tx.send(()).unwrap();
    let (scan_results, cursor, _lease) = scanner.join().unwrap();
    activator.join().unwrap();
    assert!(
        activated,
        "transaction activation must not wait for the pack walk"
    );
    assert_eq!(scan_results, vec![distinct_id(0x98)]);
    let journal = store.journal().unwrap();
    journal.sync().unwrap();
    let replay = journal.changes_since(&cursor, 8).unwrap();
    let replayed_ids: Vec<NodeId> = replay
        .groups
        .iter()
        .flat_map(|group| group.entries.iter())
        .filter_map(|entry| match &entry.mutation {
            JournalMutation::Put {
                collection_id,
                node_id,
                ..
            } if *collection_id == TEST_COLLECTION => Some(*node_id),
            _ => None,
        })
        .collect();
    assert_eq!(replayed_ids, vec![post_boundary_id]);
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "multi-reader")]
#[test]
fn scan_collection_at_snapshot_rejects_an_active_transaction_overlay() {
    let dir = test_dir("scan_collection_replay_transaction");
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(&wal).unwrap();
    store
        .transaction_overlay_users
        .fetch_add(1, Ordering::AcqRel);

    let Err(error) = store.scan_collection_at_snapshot(&[0x94u8; 16]) else {
        panic!("snapshot cannot cross transaction materialization");
    };
    store
        .transaction_overlay_users
        .fetch_sub(1, Ordering::AcqRel);
    assert!(error.is_would_block(), "expected WouldBlock, got {error}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[cfg(feature = "multi-reader")]
fn read_only_handles_cannot_take_replayable_snapshots() {
    let dir = test_dir("scan_collection_replay_read_only");
    let wal = dir.join("wal.bin");
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &TEST_COLLECTION,
            &distinct_id(0x95),
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
    writer.sync().unwrap();
    drop(writer);

    let reader = PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap();
    let Err(error) = reader.scan_collection_at_snapshot(&TEST_COLLECTION) else {
        panic!("read-only handles cannot create replay snapshots");
    };
    assert!(error.is_unsupported(), "expected Unsupported, got {error}");
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "multi-reader")]
#[test]
fn scan_collection_merges_the_read_committed_overlay() {
    let dir = test_dir("scan_collection_overlay");
    let wal = dir.join("wal.bin");
    let collection = [0x42u8; 16];
    let durable_only = distinct_id(1);
    let superseded = distinct_id(2);
    let overlay_only = distinct_id(3);

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &collection,
        &durable_only,
        &NodeData::new(bytes::Bytes::from_static(b"d")),
    )
    .unwrap();
    seed.put(
        &collection,
        &superseded,
        &NodeData::new(bytes::Bytes::from_static(b"old")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[
            JournalMutation::Put {
                collection_id: collection,
                node_id: superseded,
                payload: b"overlay".to_vec(),
            },
            JournalMutation::Put {
                collection_id: collection,
                node_id: overlay_only,
                payload: b"new".to_vec(),
            },
        ])
        .unwrap();
    drop(journal);

    let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
    store.enable_read_journal(&wal).unwrap();

    let mut scanned: Vec<(NodeId, bytes::Bytes)> = store
        .scan_collection(&collection)
        .unwrap()
        .map(|entry| {
            let (id, data) = entry.unwrap();
            (id, data.bytes)
        })
        .collect();
    scanned.sort_unstable_by_key(|(id, _)| *id);
    assert_eq!(scanned.len(), 3, "durable-only, superseded, overlay-only");
    let find = |id: NodeId| {
        scanned
            .iter()
            .find(|(entry_id, _)| *entry_id == id)
            .map(|(_, data)| data.clone())
    };
    assert_eq!(find(durable_only).as_deref(), Some(&b"d"[..]));
    assert_eq!(
        find(superseded).as_deref(),
        Some(&b"overlay"[..]),
        "the overlay value wins over the durable one"
    );
    assert_eq!(find(overlay_only).as_deref(), Some(&b"new"[..]));
    let _ = fs::remove_dir_all(&dir);
}

/// A snapshot pins one boundary across several collections: a publication
/// that lands after capture is invisible to it, and a fresh read sees it.
#[cfg(feature = "multi-reader")]
#[test]
fn read_snapshot_pins_a_boundary_across_collections() {
    let dir = test_dir("read_snapshot_pins");
    let wal = dir.join("wal.bin");
    let collection = [0x61u8; 16];
    let other = [0x62u8; 16];
    let key = distinct_id(0x01);
    let other_key = distinct_id(0x02);

    // Seed a durable shard so a read-only handle can open.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x97u8; 16],
        &[0x97u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[
            JournalMutation::Put {
                collection_id: collection,
                node_id: key,
                payload: b"old".to_vec(),
            },
            JournalMutation::Put {
                collection_id: other,
                node_id: other_key,
                payload: b"other-old".to_vec(),
            },
        ])
        .unwrap();

    let store = std::sync::Arc::new(PackfileStorage::open_read_only(dir.clone()).unwrap());
    store.enable_read_journal(&wal).unwrap();

    let snapshot = store.read_snapshot().unwrap();

    // A later publication to the same key must not be observed by the pinned
    // snapshot, even though it lands while the snapshot is held.
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: key,
            payload: b"new".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let read = snapshot
        .get_many(&[(&collection, &[key]), (&other, &[other_key])])
        .unwrap();
    assert_eq!(read[0][0].as_ref().unwrap().bytes.as_ref(), b"old");
    assert_eq!(read[1][0].as_ref().unwrap().bytes.as_ref(), b"other-old");
    drop(snapshot);

    // A fresh read refreshes and observes the new value.
    let after = store.get_read_committed(&collection, &[key]).unwrap();
    assert_eq!(after[0].as_ref().unwrap().bytes.as_ref(), b"new");
    let _ = fs::remove_dir_all(&dir);
}

/// A snapshot must not resurrect records a later collection delete removes.
#[cfg(feature = "multi-reader")]
#[test]
fn read_snapshot_does_not_resurrect_a_later_delete() {
    let dir = test_dir("read_snapshot_delete");
    let wal = dir.join("wal.bin");
    let collection = [0x63u8; 16];
    let key = distinct_id(0x03);

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x95u8; 16],
        &[0x95u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let (mut journal, _) = Journal::open(&wal).unwrap();
    journal
        .append_group(&[JournalMutation::Put {
            collection_id: collection,
            node_id: key,
            payload: b"before".to_vec(),
        }])
        .unwrap();

    let store = std::sync::Arc::new(PackfileStorage::open_read_only(dir.clone()).unwrap());
    store.enable_read_journal(&wal).unwrap();

    let snapshot = store.read_snapshot().unwrap();
    assert_eq!(
        snapshot.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"before"
    );

    // Delete the whole collection after capture: the pinned snapshot keeps its
    // pre-delete view.
    journal
        .append_group(&[JournalMutation::DeleteCollection {
            collection_id: collection,
        }])
        .unwrap();
    drop(journal);

    assert_eq!(
        snapshot.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"before",
        "the pinned snapshot must not observe the later delete"
    );
    drop(snapshot);

    assert!(
        store.get_read_committed(&collection, &[key]).unwrap()[0].is_none(),
        "a fresh read observes the delete"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A writer restart changes the publish-signal epoch; a held snapshot must
/// fail closed rather than serve a stale view, and a fresh snapshot must
/// resync.
#[cfg(feature = "multi-reader")]
#[test]
fn read_snapshot_rejects_a_writer_restart() {
    let dir = test_dir("read_snapshot_incarnation");
    let wal = dir.join("wal.bin");
    let collection = [0x64u8; 16];
    let key = distinct_id(0x04);

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x94u8; 16],
        &[0x94u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();

    let store =
        std::sync::Arc::new(PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap());
    let snapshot = store.read_snapshot().unwrap();
    assert!(
        snapshot.incarnation().is_some(),
        "the writer's publish signal must be mapped"
    );
    assert!(snapshot.get(&collection, &[key]).unwrap()[0].is_some());

    // Restart the writer while the snapshot is held: a fresh journal installs
    // a new epoch, so the pinned snapshot must fail closed rather than serve a
    // stale view.
    drop(writer);
    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();

    match snapshot.get(&collection, &[key]) {
        Err(crate::storage::StorageError::WouldBlock(_)) => {}
        other => panic!("expected WouldBlock after a writer restart, got {other:?}"),
    }
    drop(snapshot);

    // A fresh snapshot resyncs to the new incarnation and sees the new value.
    let fresh = store.read_snapshot().unwrap();
    assert_eq!(
        fresh.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"second"
    );
    drop(fresh);
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}

/// A snapshot taken on a writer handle with no read overlay reads the live
/// index.
#[cfg(feature = "multi-reader")]
#[test]
fn read_snapshot_reads_the_live_index_without_an_overlay() {
    let dir = test_dir("read_snapshot_no_overlay");
    let collection = [0x65u8; 16];
    let key = distinct_id(0x05);

    let store = std::sync::Arc::new(PackfileStorage::open(dir.clone()).unwrap());
    store
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"live")),
        )
        .unwrap();

    let snapshot = store.read_snapshot().unwrap();
    assert!(snapshot.incarnation().is_none());
    assert_eq!(
        snapshot.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"live"
    );
    drop(snapshot);
    drop(store);
    let _ = fs::remove_dir_all(&dir);
}

/// A pinned snapshot survives reclamation of the journal groups it applied:
/// the overlay entry is in memory, so materializing and checkpointing the
/// group underneath it must not change what the snapshot reads.
#[cfg(feature = "multi-reader")]
#[test]
fn read_snapshot_survives_reclaiming_the_group_it_applied() {
    let dir = test_dir("read_snapshot_reclaim");
    let wal = dir.join("wal.bin");
    let collection = [0x66u8; 16];
    let key = distinct_id(0x06);

    let seed = PackfileStorage::open(dir.clone()).unwrap();
    seed.put(
        &[0x93u8; 16],
        &[0x93u8; 16],
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.sync().unwrap();
    drop(seed);

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();
    writer
        .put(
            &collection,
            &key,
            &NodeData::new(bytes::Bytes::from_static(b"kept")),
        )
        .unwrap();

    let store =
        std::sync::Arc::new(PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap());
    let snapshot = store.read_snapshot().unwrap();
    assert_eq!(
        snapshot.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"kept"
    );

    // Materialize and checkpoint the group underneath the held snapshot.
    writer.sync_all().unwrap();

    assert_eq!(
        snapshot.get(&collection, &[key]).unwrap()[0]
            .as_ref()
            .unwrap()
            .bytes
            .as_ref(),
        b"kept",
        "a pinned snapshot must survive reclamation of the groups it applied"
    );
    drop(snapshot);
    drop(writer);
    let _ = fs::remove_dir_all(&dir);
}
