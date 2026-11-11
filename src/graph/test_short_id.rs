use super::short_id::*;
use crate::database::Database;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::ServerInfo;
const SCOPE: [u8; 16] = [0x53; 16];

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-short-id-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn index() -> ShortIdIndex {
    ShortIdIndex::new(POOL, SCOPE)
}

/// Retry the engine's retryable outcomes (`WouldBlock` and a stale read), as a
/// caller must when another writer is mid-commit. The stress tests below
/// deliberately create that contention; the retry budget inside
/// `get_or_create`/`record_events` is finite, so exhaustion is not a test
/// failure.
fn retry_contended<T>(mut operation: impl FnMut() -> Result<T, StorageError>) -> T {
    for _ in 0..1_000 {
        match operation() {
            Ok(value) => return value,
            Err(error) if error.is_would_block() || error.is_stale_read() => {
                std::thread::yield_now();
            }
            Err(error) => panic!("{error:?}"),
        }
    }
    panic!("the operation kept reporting a retryable contention error");
}

#[test]
fn allocation_is_dense_stable_and_idempotent() {
    let root = test_root("alloc");
    let db = Database::open(root.clone()).unwrap();
    let ids = index()
        .get_or_create(&db, &[b"$a", b"$b", b"$a", b"$c"])
        .unwrap();
    assert_eq!(ids, vec![1, 2, 1, 3]);
    assert_eq!(
        index().get_or_create(&db, &[b"$c", b"$d"]).unwrap(),
        vec![3, 4]
    );
    let back = index().resolve(&db, &[1, 4, 99]).unwrap();
    assert_eq!(back, vec![Some(b"$a".to_vec()), Some(b"$d".to_vec()), None]);
    assert!(index().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn previous_short_id_wire_version_is_rejected() {
    let root = test_root("old-format");
    let db = Database::open(root.clone()).unwrap();
    let txn = db.begin_transaction();
    let mut old_counter = header(*b"SIDC");
    old_counter[4] = SHORT_ID_FORMAT_VERSION - 1;
    old_counter.extend_from_slice(&1_u32.to_be_bytes());
    old_counter.extend_from_slice(&1_u32.to_be_bytes());
    old_counter.extend_from_slice(&0_u32.to_be_bytes());
    old_counter.extend_from_slice(&1_u32.to_be_bytes());
    old_counter.extend_from_slice(&0_u32.to_be_bytes());
    txn.put(
        POOL,
        SCOPE,
        COUNTER_ID,
        &crate::storage::NodeData::new(bytes::Bytes::from(old_counter)),
    )
    .unwrap();
    txn.commit().unwrap();
    assert!(matches!(
        index().counter(&db),
        Err(StorageError::Corrupt(_))
    ));
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reopen_preserves_ids_and_purge_resets_the_scope() {
    let root = test_root("reopen");
    {
        let db = Database::open(root.clone()).unwrap();
        index().get_or_create(&db, &[b"$x", b"$y"]).unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(
        index().get_or_create(&db, &[b"$y", b"$x", b"$z"]).unwrap(),
        vec![2, 1, 3]
    );
    index().purge(&db).unwrap();
    assert_eq!(index().resolve(&db, &[1]).unwrap(), vec![None]);
    assert_eq!(index().get_or_create(&db, &[b"$new"]).unwrap(), vec![1]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn concurrent_allocators_never_share_or_duplicate_ids() {
    let root = test_root("race");
    let db = Database::open(root.clone()).unwrap();
    let results: Vec<Vec<(String, u32)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
            .map(|thread| {
                let db = &db;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for round in 0..20 {
                        // Overlapping keys across threads.
                        let key = format!("$k{}", (round + thread) % 25);
                        let id = retry_contended(|| {
                            index()
                                .get_or_create(db, &[key.as_bytes()])
                                .map(|ids| ids[0])
                        });
                        out.push((key, id));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut by_key = std::collections::HashMap::new();
    for (key, id) in results.into_iter().flatten() {
        assert_eq!(*by_key.entry(key).or_insert(id), id, "one key, one id");
    }
    let mut ids: Vec<u32> = by_key.values().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), by_key.len(), "no two keys share an id");
    assert!(index().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn exhaustion_is_a_hard_error_and_publishes_nothing() {
    let root = test_root("overflow");
    let db = Database::open(root.clone()).unwrap();
    index().set_counter_for_test(&db, SHORT_ID_MAX).unwrap();
    // The last valid id is allocatable...
    assert_eq!(
        index().get_or_create(&db, &[b"$last"]).unwrap(),
        vec![SHORT_ID_MAX]
    );
    // ...the next allocation fails, including mid-batch, and aborts whole.
    let error = index()
        .get_or_create(&db, &[b"$over", b"$last", b"$also-new"])
        .unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(
        index().resolve(&db, &[SHORT_ID_MAX]).unwrap(),
        vec![Some(b"$last".to_vec())]
    );
    // Nothing from the failed batch is visible: re-asking still fails the same
    // way rather than finding a partial mapping.
    assert!(index().get_or_create(&db, &[b"$over"]).is_err());
    // Existing ids keep resolving.
    assert_eq!(
        index().get_or_create(&db, &[b"$last"]).unwrap(),
        vec![SHORT_ID_MAX]
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn corrupt_counter_is_reported_not_repaired() {
    let root = test_root("corrupt");
    let db = Database::open(root.clone()).unwrap();
    index().get_or_create(&db, &[b"$a"]).unwrap();
    {
        use crate::storage::NodeData;
        let txn = db.begin_transaction();
        txn.put(
            ShardType::ServerInfo,
            SCOPE,
            *b"MTXD-SID-CNTR-v1",
            &NodeData::new(bytes::Bytes::from_static(b"junk")),
        )
        .unwrap();
        txn.commit().unwrap();
    }
    let error = index().get_or_create(&db, &[b"$b"]).unwrap_err();
    assert!(matches!(error, StorageError::Corrupt(_)), "{error}");
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn lookup_never_allocates() {
    let root = test_root("lookup");
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(index().lookup(&db, b"$a").unwrap(), None);
    // A miss left no trace: the first real allocation still gets id 1.
    assert_eq!(index().get_or_create(&db, &[b"$a"]).unwrap(), vec![1]);
    assert_eq!(index().lookup(&db, b"$a").unwrap(), Some(1));
    assert_eq!(index().lookup(&db, b"$b").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_lowered_max_id_refuses_inside_the_transaction() {
    let root = test_root("max-id");
    let db = Database::open(root.clone()).unwrap();
    let small = index().with_max_id(3);
    assert_eq!(
        small.get_or_create(&db, &[b"$a", b"$b", b"$c"]).unwrap(),
        vec![1, 2, 3]
    );
    // A batch that would cross the limit is refused whole: the new key past the
    // limit is not published, and neither is the key before it in the batch.
    let error = small.get_or_create(&db, &[b"$d", b"$e"]).unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(small.lookup(&db, b"$d").unwrap(), None);
    assert_eq!(small.lookup(&db, b"$e").unwrap(), None);
    // Existing ids still resolve and re-asking for them still works.
    assert_eq!(small.get_or_create(&db, &[b"$c"]).unwrap(), vec![3]);
    // The limit cannot be raised above the hard ceiling.
    assert_eq!(
        index()
            .with_max_id(u32::MAX)
            .get_or_create(&db, &[b"$z"])
            .unwrap(),
        vec![4]
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A batch allocates once and either publishes whole or not at all.
#[test]
fn a_batch_allocates_once_and_a_failed_batch_publishes_nothing() {
    let root = test_root("batch");
    let db = Database::open(root.clone()).unwrap();
    let event = |owner| BatchEvent {
        owner,
        owner_payload: None,
    };
    let ids = index()
        .record_events(&db, &[event(b"$a"), event(b"$b"), event(b"$a")])
        .unwrap();
    assert_eq!(ids, vec![1, 2, 1]);
    assert_eq!(index().record_events(&db, &[]).unwrap(), Vec::<u32>::new());

    // Running out of ids part-way through publishes none of the batch.
    index()
        .set_counter_for_test(&db, SHORT_ID_MAX.saturating_sub(1))
        .unwrap();
    let error = index()
        .record_events(&db, &[event(b"$e1"), event(b"$e2"), event(b"$e3")])
        .unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(index().lookup(&db, b"$e1").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The engine's hash index buckets a record by id bytes 0..8 and tags it with
/// bytes 8..12, assuming ids are spread. Every record id this layer builds must
/// vary in both fields across sequential short ids; a constant prefix there put
/// every reverse record into one bucket.
#[test]
fn record_ids_vary_in_the_bytes_the_index_uses() {
    const IDS: u32 = 10_000;
    let mut buckets = std::collections::HashSet::new();
    let mut tags = std::collections::HashSet::new();
    for short_id in 1..=IDS {
        let id = reverse_record_id_for_test(short_id);
        buckets.insert(<[u8; 8]>::try_from(&id[..8]).unwrap());
        tags.insert(<[u8; 4]>::try_from(&id[8..12]).unwrap());
    }
    let total = usize::try_from(IDS).unwrap();
    assert!(
        buckets.len() >= total - 3,
        "the first 8 bytes (the index bucket) collide: {} distinct of {total}",
        buckets.len()
    );
    assert!(
        tags.len() >= total - 3,
        "bytes 8..12 (the index tag) collide: {} distinct of {total}",
        tags.len()
    );
}

fn owned<'a>(owner: &'a [u8], payload: &'a [u8]) -> BatchEvent<'a> {
    BatchEvent {
        owner,
        owner_payload: Some(payload),
    }
}

fn log(db: &Database) -> Vec<(u32, u32, Vec<u8>)> {
    let counters = index().counters(db).unwrap();
    index()
        .owner_log(
            db,
            counters.log_epoch,
            counters.log_epoch_start_seq,
            counters.next_owner_seq,
        )
        .unwrap()
        .into_iter()
        .map(|e| (e.seq, e.short_id, e.payload))
        .collect()
}

#[test]
fn recording_an_owner_twice_logs_it_once() {
    let root = test_root("owner-twice");
    let db = Database::open(root.clone()).unwrap();
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    let after_first = index().counters(&db).unwrap();
    // A replay, a re-fetch, and the same owner twice in one batch.
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    index()
        .record_events(&db, &[owned(b"$b", b"pb"), owned(b"$b", b"pb")])
        .unwrap();
    index().record_events(&db, &[owned(b"$b", b"pb")]).unwrap();
    assert_eq!(index().counters(&db).unwrap().next_owner_seq, 3);
    assert_eq!(
        log(&db),
        vec![(1, 1, b"pa".to_vec()), (2, 2, b"pb".to_vec())]
    );
    assert_eq!(after_first.next_owner_seq, 2);
    let report = index().verify(&db).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.owners_checked, 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_allocated_key_that_becomes_an_owner_is_logged_once() {
    let root = test_root("owner-after-target");
    let db = Database::open(root.clone()).unwrap();
    // `$parent` is allocated an id without being an owner.
    index().get_or_create(&db, &[b"$parent"]).unwrap();
    assert_eq!(index().is_owner(&db, b"$parent").unwrap(), Some(false));
    assert_eq!(log(&db).len(), 0, "an allocated key is not an owner");

    // Now the parent arrives: its bit flips and it is logged, once.
    index()
        .record_events(&db, &[owned(b"$parent", b"parent")])
        .unwrap();
    index()
        .record_events(&db, &[owned(b"$parent", b"parent")])
        .unwrap();
    assert_eq!(index().is_owner(&db, b"$parent").unwrap(), Some(true));
    assert_eq!(log(&db).len(), 1);
    assert!(index().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn conflicting_owner_payloads_in_one_batch_collide() {
    let root = test_root("owner-conflict");
    let db = Database::open(root.clone()).unwrap();
    let error = index()
        .record_events(&db, &[owned(b"$a", b"one"), owned(b"$a", b"two")])
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error:?}");
    assert_eq!(index().counters(&db).unwrap().next_owner_seq, 1);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Several writers record the same owners at once: every owner is logged
/// exactly once and the log stays dense.
#[test]
fn racing_writers_log_each_owner_once() {
    let root = test_root("owner-race");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    let keys: Vec<Vec<u8>> = (0..24).map(|i| format!("$e{i}").into_bytes()).collect();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let db = db.clone();
            let keys = keys.clone();
            std::thread::spawn(move || {
                for key in &keys {
                    retry_contended(|| index().record_events(&db, &[owned(key, key)]));
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let entries = log(&db);
    assert_eq!(entries.len(), keys.len());
    let mut payloads: Vec<_> = entries.iter().map(|e| e.2.clone()).collect();
    payloads.sort();
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(payloads, expected);
    assert!(entries
        .iter()
        .enumerate()
        .all(|(i, e)| e.0 as usize == i + 1));
    let report = index().verify(&db).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn purge_removes_the_owner_log_too() {
    let root = test_root("owner-purge");
    let db = Database::open(root.clone()).unwrap();
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    index().purge(&db).unwrap();
    assert_eq!(index().counters(&db).unwrap(), ScopeCounters::default());
    assert_eq!(index().is_owner(&db, b"$a").unwrap(), None);
    // The generation collection is gone: a fresh owner starts a fresh log.
    index().record_events(&db, &[owned(b"$b", b"pb")]).unwrap();
    assert_eq!(log(&db), vec![(1, 1, b"pb".to_vec())]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_checks_each_log_entry_against_its_forward_record() {
    let root = test_root("verify-log");
    let db = Database::open(root.clone()).unwrap();
    index()
        .record_events(&db, &[owned(b"$a", b"pa"), owned(b"$b", b"pb")])
        .unwrap();
    assert!(index().verify(&db).unwrap().is_consistent());
    // Point entry 2 at id 1: the bit count still matches, the entry is wrong.
    let counters = index().counters(&db).unwrap();
    let txn = db.begin_transaction();
    let mut bad = b"OWNL".to_vec();
    bad.push(SHORT_ID_FORMAT_VERSION);
    bad.extend_from_slice(&1_u32.to_be_bytes());
    bad.extend_from_slice(b"pb");
    txn.put(
        POOL,
        index().owner_log_collection(counters.log_epoch),
        super::short_id::owner_log_record_id_for_test(2),
        &crate::storage::NodeData::new(bytes::Bytes::from(bad)),
    )
    .unwrap();
    txn.commit().unwrap();
    let report = index().verify(&db).unwrap();
    assert!(
        report.problems.iter().any(|p| p.contains("repeats id 1")),
        "{:?}",
        report.problems
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// `verify` racing a writer must never report a problem on a healthy store:
/// a pass that overlapped a commit is discarded and rerun.
#[test]
fn verify_racing_a_writer_reports_no_false_problems() {
    let root = test_root("verify-race");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    // Ids that already exist as non-owners: the writer flips their owner bits
    // under a pass that has already fixed its counters.
    let keys: Vec<Vec<u8>> = (0..600).map(|i| format!("$e{i}").into_bytes()).collect();
    let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
    index().get_or_create(&db, &refs).unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
    let writer = {
        let db = db.clone();
        std::thread::spawn(move || {
            for (position, key) in keys.iter().enumerate() {
                let key = key.clone();
                index().record_events(&db, &[owned(&key, &key)]).unwrap();
                if position == 0 {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                }
            }
        })
    };
    started_rx.recv().unwrap();
    let report = index().verify(&db).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    release_tx.send(()).unwrap();
    writer.join().unwrap();
    let report = index().verify(&db).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.owners_checked, 600);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
