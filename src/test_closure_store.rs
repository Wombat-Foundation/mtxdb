use super::closure_store::*;
use super::short_id::{EdgeFamily, ShortIdIndex};
use crate::database::SharedDatabase;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const SCOPE: [u8; 16] = [0xC1; 16];

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-closure-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn store() -> ClosureStore {
    ClosureStore::new(POOL, SCOPE)
}

fn build(db: &SharedDatabase, tag: u8, ids: std::ops::Range<u32>) -> ClosureHead {
    let store = store();
    let mut builder = store.begin(db).unwrap();
    let blobs: Vec<(u32, Vec<u8>)> = ids.clone().map(|id| (id, vec![tag, id as u8])).collect();
    let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
    builder.add(db, &refs).unwrap();
    builder.publish(db, ids.end).unwrap()
}

#[test]
fn unpublished_generation_is_invisible_then_published_atomically() {
    let root = test_root("publish");
    let db = SharedDatabase::open(root.clone()).unwrap();
    assert_eq!(store().head(&db).unwrap(), None);
    assert_eq!(store().get(&db, 1).unwrap(), None);

    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"one".as_slice())]).unwrap();
    assert_eq!(
        store().get(&db, 1).unwrap(),
        None,
        "not visible before publish"
    );
    let head = builder.publish(&db, 2).unwrap();
    assert_eq!((head.generation, head.previous, head.count), (1, 0, 1));
    assert_eq!(store().get(&db, 1).unwrap(), Some(b"one".to_vec()));
    assert!(store().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn republish_swaps_generation_and_retirement_keeps_the_predecessor() {
    let root = test_root("swap");
    let db = SharedDatabase::open(root.clone()).unwrap();
    build(&db, 1, 1..4);
    let second = build(&db, 2, 1..6);
    assert_eq!((second.generation, second.previous), (2, 1));
    assert_eq!(store().get(&db, 5).unwrap(), Some(vec![2, 5]));
    assert_eq!(store().get(&db, 2).unwrap(), Some(vec![2, 2]));

    // Generations 1 and 2 are live (current + predecessor): nothing to retire.
    assert_eq!(store().retire_superseded(&db).unwrap(), 0);
    let third = build(&db, 3, 1..6);
    assert_eq!((third.generation, third.previous), (3, 2));
    assert_eq!(
        store().retire_superseded(&db).unwrap(),
        1,
        "generation 1 retired"
    );
    assert_eq!(store().get(&db, 4).unwrap(), Some(vec![3, 4]));
    assert!(store().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn stale_builder_loses_the_race_and_can_abandon() {
    let root = test_root("race");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let store = store();
    let mut slow = store.begin(&db).unwrap();
    slow.add(&db, &[(1, b"slow".as_slice())]).unwrap();
    // A faster builder publishes first.
    build(&db, 9, 1..3);
    let error = slow.publish(&db, 2).unwrap_err();
    assert!(matches!(error, StorageError::StaleRead { .. }), "{error}");
    assert_eq!(
        store.get(&db, 1).unwrap(),
        Some(vec![9, 1]),
        "winner stays published"
    );

    let another = store.begin(&db).unwrap();
    assert!(
        another.generation() > 2,
        "generation numbers are never reused"
    );
    another.abandon(&db).unwrap();
    assert!(
        store.retire_superseded(&db).unwrap() >= 1,
        "orphan from the loser reclaimed"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reopen_preserves_head_and_orphans_are_reclaimed() {
    let root = test_root("reopen");
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        build(&db, 4, 1..4);
        // A crash mid-build: generation written, head never swapped.
        let mut crashed = store().begin(&db).unwrap();
        crashed.add(&db, &[(1, b"partial".as_slice())]).unwrap();
        std::mem::forget(crashed);
    }
    let db = SharedDatabase::open(root.clone()).unwrap();
    let head = store().head(&db).unwrap().unwrap();
    assert_eq!((head.generation, head.count, head.source_next), (1, 3, 4));
    assert_eq!(store().get(&db, 3).unwrap(), Some(vec![4, 3]));
    assert!(store().verify(&db).unwrap().is_consistent());
    assert_eq!(
        store().retire_superseded(&db).unwrap(),
        1,
        "orphan generation 2"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_reports_gaps_and_empty_blobs_are_rejected() {
    let root = test_root("verify");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let mut builder = store().begin(&db).unwrap();
    assert!(builder.add(&db, &[(1, b"".as_slice())]).is_err());
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    // Claims to cover ids 1..4 but only one closure was written.
    builder.publish(&db, 4).unwrap();
    let report = store().verify(&db).unwrap();
    assert!(!report.is_consistent());
    assert_eq!(report.records_checked, 1);
    assert!(report.problems.len() >= 2, "{:?}", report.problems);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn scope_purge_is_atomic_with_the_short_id_scope() {
    let root = test_root("purge");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let ids = ShortIdIndex::new(POOL, SCOPE);
    ids.record_edges(&db, b"$a", EdgeFamily::plain(1), &[b"$b"])
        .unwrap();
    build(&db, 1, 1..3);
    build(&db, 2, 1..3);

    let txn = db.begin_transaction();
    store().stage_purge(&txn).unwrap();
    ids.stage_purge(&txn).unwrap();
    txn.commit().unwrap();

    assert_eq!(store().head(&db).unwrap(), None);
    assert_eq!(store().get(&db, 1).unwrap(), None);
    assert_eq!(ids.resolve(&db, &[1, 2]).unwrap(), vec![None, None]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------
// Crash injection: a real child process is aborted at an exact step (no
// destructors, no flush), and the parent reopens the store. The invariant in
// every case: readers see the old complete generation or the new complete one,
// never a partial or mixed generation.
// ---------------------------------------------------------------------------

const CRASH_IDS: u32 = 7;

/// Blob for `id` in `generation`: lets a reader detect a mixed generation.
fn blob_for(generation: u64, id: u32) -> Vec<u8> {
    vec![generation as u8, id as u8]
}

fn publish_full(db: &SharedDatabase) -> ClosureHead {
    let mut builder = store().begin(db).unwrap();
    let generation = builder.generation();
    for batch in [1..4u32, 4..CRASH_IDS] {
        let blobs: Vec<(u32, Vec<u8>)> = batch.map(|id| (id, blob_for(generation, id))).collect();
        let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
        builder.add(db, &refs).unwrap();
    }
    builder.publish(db, CRASH_IDS).unwrap()
}

/// Child-process body. A no-op unless the parent set `MTXDB_CRASH_SCENARIO`.
#[test]
fn crash_child() {
    let Ok(scenario) = std::env::var("MTXDB_CRASH_SCENARIO") else {
        return;
    };
    let root = PathBuf::from(std::env::var("MTXDB_CRASH_ROOT").unwrap());
    let db = SharedDatabase::open(root).unwrap();
    match scenario.as_str() {
        "rebuild" => {
            publish_full(&db);
        }
        "retire" => {
            store().retire_superseded(&db).unwrap();
        }
        other => panic!("unknown scenario {other}"),
    }
}

/// Run `scenario` in a child that aborts at `crash_at`; returns whether it died
/// of SIGABRT (the injected crash), as opposed to a panic or a clean exit.
fn run_child(root: &PathBuf, scenario: &str, crash_at: &str) -> bool {
    use std::os::unix::process::ExitStatusExt;
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "test_closure_store::crash_child",
            "--test-threads=1",
        ])
        .env("MTXDB_CRASH_SCENARIO", scenario)
        .env("MTXDB_CRASH_ROOT", root)
        .env("MTXDB_CRASH_AT", crash_at)
        .output()
        .unwrap()
        .status;
    status.signal() == Some(6)
}

/// Control: with no matching crash point the child completes normally, so the
/// crash tests cannot pass just because the child always fails.
#[test]
fn crash_child_without_a_crash_point_succeeds() {
    let root = test_root("crash-control");
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        publish_full(&db);
    }
    assert!(!run_child(&root, "rebuild", "no-such-point"));
    let db = SharedDatabase::open(root.clone()).unwrap();
    assert_eq!(assert_complete_snapshot(&db).generation, 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The published generation is complete, uniform and self-consistent.
fn assert_complete_snapshot(db: &SharedDatabase) -> ClosureHead {
    let head = store()
        .head(db)
        .unwrap()
        .expect("a generation is published");
    let ids: Vec<u32> = (1..head.source_next).collect();
    let (generation, blobs) = store().get_many(db, &ids).unwrap();
    assert_eq!(generation, Some(head.generation));
    for (id, blob) in ids.iter().zip(blobs) {
        assert_eq!(
            blob,
            Some(blob_for(head.generation, *id)),
            "id {id} must come from generation {}",
            head.generation
        );
    }
    assert!(store().verify(db).unwrap().is_consistent());
    head
}

fn crash_rebuild_keeps_old_generation(name: &str, crash_at: &str, written: &[u32]) {
    let root = test_root(name);
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        publish_full(&db);
    }
    assert!(
        run_child(&root, "rebuild", crash_at),
        "child must crash at {crash_at}"
    );
    let db = SharedDatabase::open(root.clone()).unwrap();
    // The crash left exactly the expected partial orphan under generation 2.
    for id in 1..CRASH_IDS {
        assert_eq!(
            store().raw_generation_record(&db, 2, id).is_some(),
            written.contains(&id),
            "orphan generation 2, id {id}, after a crash at {crash_at}"
        );
    }
    let head = assert_complete_snapshot(&db);
    assert_eq!(
        head.generation, 1,
        "an unpublished rebuild must not become visible"
    );
    // The orphan left by the dead builder is reclaimed, and the store still works.
    store().retire_superseded(&db).unwrap();
    assert_complete_snapshot(&db);
    let next = publish_full(&db);
    assert!(
        next.generation > 2,
        "reserved generation numbers are never reused"
    );
    assert_complete_snapshot(&db);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn crash_after_generation_records_are_written() {
    crash_rebuild_keeps_old_generation("crash-after-add", "after-add", &[1, 2, 3]);
}

#[test]
fn crash_before_head_publication() {
    crash_rebuild_keeps_old_generation(
        "crash-before-publish",
        "before-publish",
        &[1, 2, 3, 4, 5, 6],
    );
}

#[test]
fn crash_staged_head_swap_before_commit() {
    crash_rebuild_keeps_old_generation(
        "crash-before-commit",
        "publish-before-commit",
        &[1, 2, 3, 4, 5, 6],
    );
}

#[test]
fn crash_after_head_publication_before_sync() {
    let root = test_root("crash-after-commit");
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        publish_full(&db);
    }
    assert!(run_child(&root, "rebuild", "publish-after-commit"));
    let db = SharedDatabase::open(root.clone()).unwrap();
    // A killed process keeps its page-cache writes, so the swap is normally
    // durable; a power loss could lose it. Either outcome is a complete
    // generation, which is the invariant under test.
    let head = assert_complete_snapshot(&db);
    assert!(
        matches!(head.generation, 1 | 2),
        "generation {}",
        head.generation
    );
    if head.generation == 2 {
        assert_eq!(head.previous, 1);
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

fn crash_retire(name: &str, crash_at: &str, generation_one_survives: bool) {
    let root = test_root(name);
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        for _ in 0..3 {
            publish_full(&db);
        }
    }
    assert!(
        run_child(&root, "retire", crash_at),
        "child must crash at {crash_at}"
    );
    let db = SharedDatabase::open(root.clone()).unwrap();
    assert_eq!(
        store().raw_generation_record(&db, 1, 1).is_some(),
        generation_one_survives,
        "superseded generation 1 after a crash at {crash_at}"
    );
    assert!(
        store().raw_generation_record(&db, 2, 1).is_some(),
        "the predecessor is never retired"
    );
    let head = assert_complete_snapshot(&db);
    assert_eq!(
        head.generation, 3,
        "cleanup must never touch the published generation"
    );
    // Cleanup is idempotent and the predecessor stays readable throughout.
    store().retire_superseded(&db).unwrap();
    store().retire_superseded(&db).unwrap();
    assert_complete_snapshot(&db);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn crash_during_orphan_cleanup_before_commit() {
    crash_retire("crash-retire-before", "retire-before-commit", true);
}

#[test]
fn crash_during_orphan_cleanup_after_commit() {
    crash_retire("crash-retire-after", "retire-after-commit", false);
}

/// A live reader never observes a partial or mixed generation while builders
/// write batches and publish, and cleanup retires superseded generations.
#[test]
fn concurrent_readers_never_see_a_mixed_generation() {
    let root = test_root("readers");
    let db = SharedDatabase::open(root.clone()).unwrap();
    publish_full(&db);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..3)
            .map(|_| {
                scope.spawn(|| {
                    let ids: Vec<u32> = (1..CRASH_IDS).collect();
                    let mut reads = 0u32;
                    while !done.load(std::sync::atomic::Ordering::Acquire) {
                        let (generation, blobs) = store().get_many(&db, &ids).unwrap();
                        let generation = generation.expect("a generation is always published");
                        for (id, blob) in ids.iter().zip(&blobs) {
                            assert_eq!(
                                blob.as_deref(),
                                Some(blob_for(generation, *id).as_slice()),
                                "mixed or partial generation for id {id}"
                            );
                        }
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();
        for _ in 0..8 {
            publish_full(&db);
            store().retire_superseded(&db).unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Release);
        for reader in readers {
            assert!(reader.join().unwrap() > 0);
        }
    });
    assert_complete_snapshot(&db);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
