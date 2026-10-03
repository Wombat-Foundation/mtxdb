use super::population::*;
use super::population::{run_chunk_id_for_test, BEFORE_PUBLISH, WRITTEN_RUNS};
use crate::database::Database;
use crate::layout::ShardType;
use crate::short_id::{BatchEvent, ScopeCounters, ShortIdIndex};
use crate::storage::StorageError;
use rezzy_recon::{
    build_bucket_nodes, BucketRequest, ElementHash, Population, ResidentKernel, SortedPopulation,
};
use std::path::PathBuf;

const SCOPE: [u8; 16] = [0x61; 16];

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-population-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn index() -> ShortIdIndex {
    ShortIdIndex::new(ShardType::Edges, SCOPE)
}

/// A spread-out hash for element `i`: distinct digests, varied top bits.
fn hash(i: u32) -> ElementHash {
    let mut digest = [0_u8; 32];
    let mut x = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for chunk in digest.chunks_exact_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_be_bytes());
    }
    ElementHash::from_digest32(digest)
}

/// `record_events`, retried while the engine reports a retryable
/// `WouldBlock` (record versions advancing under a read), as a caller would.
fn record_retrying(scope: &ShortIdIndex, db: &Database, events: &[BatchEvent<'_>]) {
    for _ in 0..1000 {
        match scope.record_events(db, events) {
            Ok(_) => return,
            Err(error) if error.is_would_block() => std::thread::yield_now(),
            Err(error) => panic!("{error:?}"),
        }
    }
    panic!("record_events kept blocking");
}

fn record(db: &Database, range: std::ops::Range<u32>) {
    for i in range {
        let key = format!("$e{i}").into_bytes();
        let payload = encode_owner_payload(hash(i));
        record_retrying(
            &index(),
            db,
            &[BatchEvent {
                owner: &key,
                families: &[],
                owner_payload: Some(&payload),
            }],
        );
    }
}

/// The nodes a round touches: four disjoint depth-2 buckets.
fn nodes() -> Vec<BucketRequest> {
    (0..4)
        .map(|prefix| BucketRequest::new(2, prefix, 16))
        .collect()
}

#[test]
fn an_empty_store_has_an_empty_population() {
    let root = test_root("empty");
    let db = Database::open(root.clone()).unwrap();
    let snapshot = index().population_snapshot(&db).unwrap();
    assert!(snapshot.is_empty());
    assert_eq!(snapshot.manifest_version(), 0);
    assert_eq!(snapshot.owner_seq_ceiling(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_snapshot_serves_exactly_the_logged_owners() {
    let root = test_root("serves");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..40);
    // An id allocated only because something referenced it is not an owner.
    index().get_or_create(&db, &[b"$referenced"]).unwrap();
    let snapshot = index().population_snapshot(&db).unwrap();
    let expected = SortedPopulation::new((0..40).map(hash).collect());
    assert_eq!(snapshot.len(), 40);
    assert_eq!(snapshot.owner_seq_ceiling(), 41);
    assert_eq!(
        build_bucket_nodes(&snapshot, &nodes()).unwrap(),
        build_bucket_nodes(&expected, &nodes()).unwrap()
    );
    let mut got = Vec::new();
    snapshot.candidates_into(hash(7).h64, &mut got);
    assert_eq!(got, vec![hash(7).h128]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Writers keep appending between the rounds of an exchange; a snapshot taken
/// before them answers every round from the same population.
#[test]
fn a_pinned_snapshot_ignores_inserts_between_rounds() {
    let root = test_root("pinned");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..30);
    let pinned = index().population_snapshot(&db).unwrap();
    let original = SortedPopulation::new((0..30).map(hash).collect());
    let want = build_bucket_nodes(&original, &nodes()).unwrap();

    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    record(&db, 30..45);
    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    record(&db, 45..60);
    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    assert_eq!(pinned.len(), 30);
    assert_eq!(pinned.owner_seq_ceiling(), 31);

    let fresh = index().population_snapshot(&db).unwrap();
    assert_eq!(fresh.len(), 60);
    assert_eq!(fresh.owner_seq_ceiling(), 61);
    assert_ne!(build_bucket_nodes(&fresh, &nodes()).unwrap(), want);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_malformed_payload_is_corruption() {
    let root = test_root("malformed");
    let db = Database::open(root.clone()).unwrap();
    index()
        .record_events(
            &db,
            &[BatchEvent {
                owner: b"$a",
                families: &[],
                owner_payload: Some(b"too short"),
            }],
        )
        .unwrap();
    assert!(matches!(
        index().population_snapshot(&db),
        Err(StorageError::Corrupt(_))
    ));
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

const GRACE_MS: u64 = 1_000;

/// The nodes and kernel a population of `range` must serve, built directly.
fn expected(
    range: std::ops::Range<u32>,
) -> (
    Vec<(
        rezzy_recon::SyndromeSketch,
        rezzy_recon::triage::NodeSummary,
    )>,
    ResidentKernel,
) {
    let hashes: Vec<ElementHash> = range.map(hash).collect();
    let mut kernel = ResidentKernel::new();
    for &h in &hashes {
        kernel.insert(h).unwrap();
    }
    let nodes = build_bucket_nodes(&SortedPopulation::new(hashes), &nodes()).unwrap();
    (nodes, kernel)
}

fn assert_serves(snapshot: &super::population::PopulationSnapshot, range: std::ops::Range<u32>) {
    let (nodes_want, kernel_want) = expected(range.clone());
    assert_eq!(snapshot.len(), range.len());
    assert_eq!(build_bucket_nodes(snapshot, &nodes()).unwrap(), nodes_want);
    assert_eq!(snapshot.kernel().unwrap(), kernel_want);
}

#[test]
fn compaction_folds_the_log_without_changing_the_population() {
    let root = test_root("compact");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..50);
    let report = index().compact(&db, 0, GRACE_MS, true).unwrap().unwrap();
    assert_eq!(
        (
            report.manifest_version,
            report.folded_entries,
            report.run_entries
        ),
        (1, 50, 50)
    );
    let counters = index().counters(&db).unwrap();
    assert_eq!(
        (
            counters.manifest_version,
            counters.log_epoch,
            counters.log_epoch_start_seq
        ),
        (1, 1, 51)
    );
    let snapshot = index().population_snapshot(&db).unwrap();
    assert_eq!(snapshot.manifest_version(), 1);
    assert_eq!(snapshot.owner_seq_ceiling(), 51);
    assert_serves(&snapshot, 0..50);
    // New owners land in the new generation and join as the tail.
    record(&db, 50..60);
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..60);
    assert!(index().verify(&db, &[]).unwrap().is_consistent());
    // A tail below the trigger is left alone unless forced.
    assert_eq!(index().compact(&db, 0, GRACE_MS, false).unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn comparable_runs_merge_and_a_small_run_stands_alone() {
    let root = test_root("tiering");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..30);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    record(&db, 30..60);
    let merged = index().compact(&db, 0, GRACE_MS, true).unwrap().unwrap();
    assert_eq!((merged.folded_entries, merged.run_entries), (30, 60));
    record(&db, 60..70);
    let small = index().compact(&db, 0, GRACE_MS, true).unwrap().unwrap();
    assert_eq!((small.folded_entries, small.run_entries), (10, 10));
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..70);
    assert!(index().verify(&db, &[]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The straddle: a pin read before a compaction must still build the
/// population it named after the compaction publishes. Manifest, watermark and
/// tail generation come from the pin's one counter read; reading any of them
/// live would pair the old tail with the new runs and lose what was folded.
#[test]
fn a_pin_survives_a_compaction_that_publishes_before_it_materializes() {
    let root = test_root("straddle");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..40);
    let pin = index().pin_population(&db).unwrap();
    record(&db, 40..55);
    index().compact(&db, 0, GRACE_MS, true).unwrap().unwrap();
    record(&db, 55..60);
    let snapshot = index().materialize_population(&db, &pin, None).unwrap();
    assert_eq!(snapshot.manifest_version(), 0);
    assert_eq!(snapshot.owner_seq_ceiling(), 41);
    assert_serves(&snapshot, 0..40);
    // A pin taken after the compaction sees everything.
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..60);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_dropped_generation_expires_the_pin_only_after_the_grace_period() {
    let root = test_root("gc");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..20);
    let pin = index().pin_population(&db).unwrap();
    index().compact(&db, 100, GRACE_MS, true).unwrap().unwrap();

    // Inside the grace period nothing is dropped and the pin still resolves.
    let early = index().collect_garbage(&db, 100 + GRACE_MS - 1).unwrap();
    assert_eq!((early.dropped, early.pending), (0, 1));
    assert_serves(
        &index().materialize_population(&db, &pin, None).unwrap(),
        0..20,
    );

    let late = index().collect_garbage(&db, 100 + GRACE_MS).unwrap();
    assert_eq!((late.dropped, late.pending), (1, 0));
    // The dropped generation reports a missing entry, not corruption.
    assert!(matches!(
        index().owner_log(&db, 0, 1, 21),
        Err(StorageError::NotFound(_))
    ));
    let expired = index().materialize_population(&db, &pin, None);
    assert!(
        matches!(expired, Err(StorageError::StaleGeneration { .. })),
        "{:?}",
        expired.map(|s| s.len())
    );
    // A fresh pin is unaffected.
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..20);
    assert_eq!(
        index().collect_garbage(&db, u64::MAX).unwrap(),
        GcReport::default()
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Writers appending while compaction runs: no owner is lost or logged twice.
#[test]
fn writers_racing_compaction_lose_nothing() {
    let root = test_root("race");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    let writer = {
        let db = db.clone();
        std::thread::spawn(move || {
            // A writer with idle gaps, as a room's traffic has.
            for i in 0..400 {
                record(&db, i..i + 1);
                std::thread::sleep(std::time::Duration::from_micros(150));
            }
        })
    };
    let mut compactions = 0;
    while !writer.is_finished() {
        if index().compact(&db, 0, GRACE_MS, true).unwrap().is_some() {
            compactions += 1;
        }
    }
    writer.join().unwrap();
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    assert!(compactions >= 1, "compaction never ran against the writer");
    let report = index().verify(&db, &[]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.owners_checked, 400);
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..400);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_cache_reuses_a_base_until_the_manifest_moves() {
    let root = test_root("cache");
    let db = Database::open(root.clone()).unwrap();
    let cache = PopulationCache::new(2);
    record(&db, 0..20);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    let first = index().population_snapshot_cached(&db, &cache).unwrap();
    record(&db, 20..25);
    let second = index().population_snapshot_cached(&db, &cache).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!((first.len(), second.len()), (20, 25));
    assert_serves(&second, 0..25);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    assert_serves(
        &index().population_snapshot_cached(&db, &cache).unwrap(),
        0..25,
    );
    assert_eq!(cache.len(), 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn purge_after_compaction_drops_every_run_and_queued_generation() {
    let root = test_root("purge");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..30);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    record(&db, 30..60);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    let before = index().run_collections_for_test(&db).unwrap();
    // The live run holds chunks; the queued entries are folded generations,
    // which hold log entries instead.
    assert!(before.iter().any(|(_, present)| *present));
    index().purge(&db).unwrap();
    assert_eq!(index().counters(&db).unwrap(), ScopeCounters::default());
    // The old run collections are empty, not just unreferenced.
    let txn = db.begin_transaction();
    for (collection, _) in &before {
        let (records, _) = txn
            .get_with_record_versions(ShardType::Edges, collection, &[run_chunk_id_for_test(0)])
            .unwrap();
        // Only runs have chunk records; a queued generation reads empty here.
        assert!(records[0].is_none());
    }
    for epoch in 0..3 {
        let (records, _) = txn
            .get_with_record_versions(
                ShardType::Edges,
                &index().owner_log_collection(epoch),
                &[crate::short_id::owner_log_record_id_for_test(1)],
            )
            .unwrap();
        assert!(
            records[0].is_none(),
            "generation {epoch} survived the purge"
        );
    }
    // The scope starts over cleanly.
    record(&db, 0..5);
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..5);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_flags_runs_that_disagree_with_the_log() {
    let root = test_root("verify-runs");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..10);
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    assert!(index().verify(&db, &[]).unwrap().is_consistent());
    // Make the counter claim one more folded owner than the runs hold.
    let counters = index().counters(&db).unwrap();
    let txn = db.begin_transaction();
    txn.put(
        ShardType::Edges,
        SCOPE,
        super::short_id::COUNTER_ID,
        &crate::storage::NodeData::new(bytes::Bytes::from(super::short_id::encode_counter(
            ScopeCounters {
                log_epoch_start_seq: counters.log_epoch_start_seq + 1,
                next_owner_seq: counters.next_owner_seq + 1,
                ..counters
            },
        ))),
    )
    .unwrap();
    txn.commit().unwrap();
    let report = index().verify(&db, &[]).unwrap();
    assert!(!report.is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Every run a compaction attempt started writing is either live or queued for
/// deletion, however many attempts raced and were abandoned: two compactors
/// and a writer on one scope.
#[test]
fn abandoned_compaction_attempts_leave_no_unqueued_runs() {
    let scope = ShortIdIndex::new(ShardType::Edges, [0x6f; 16]);
    let root = test_root("leak");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    let writer = {
        let db = db.clone();
        std::thread::spawn(move || {
            for i in 0..300_u32 {
                let key = format!("$e{i}").into_bytes();
                let payload = encode_owner_payload(hash(i));
                record_retrying(
                    &scope,
                    &db,
                    &[BatchEvent {
                        owner: &key,
                        families: &[],
                        owner_payload: Some(&payload),
                    }],
                );
                std::thread::sleep(std::time::Duration::from_micros(150));
            }
        })
    };
    let compactors: Vec<_> = (0..2)
        .map(|_| {
            let db = db.clone();
            let writer_done = std::sync::Arc::new(());
            std::thread::spawn(move || {
                let _ = writer_done;
                for _ in 0..40 {
                    // A starved attempt may report a stale read; leaks are
                    // what is under test, not progress.
                    let _ = scope.compact(&db, 0, GRACE_MS, true);
                }
            })
        })
        .collect();
    writer.join().unwrap();
    for compactor in compactors {
        compactor.join().unwrap();
    }
    scope.compact(&db, 0, GRACE_MS, true).unwrap();
    let known: Vec<[u8; 16]> = scope
        .run_collections_for_test(&db)
        .unwrap()
        .into_iter()
        .map(|(collection, _)| collection)
        .collect();
    let written: Vec<[u8; 16]> = WRITTEN_RUNS
        .lock()
        .iter()
        .filter(|(owner, _)| *owner == [0x6f; 16])
        .map(|(_, run)| *run)
        .collect();
    assert!(!written.is_empty());
    for run in &written {
        assert!(
            known.contains(run),
            "a written run is neither live nor queued"
        );
    }
    let report = scope.verify(&db, &[]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Garbage collection takes the run queued for an in-progress compaction (its
/// grace period is already over): the publish must notice and abandon that
/// attempt, not publish a manifest naming a deleted run.
#[test]
fn a_run_collected_before_the_publish_is_never_published() {
    let scope = ShortIdIndex::new(ShardType::Edges, [0x70; 16]);
    let root = test_root("collected-run");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    for i in 0..20_u32 {
        let key = format!("$e{i}").into_bytes();
        let payload = encode_owner_payload(hash(i));
        record_retrying(
            &scope,
            &db,
            &[BatchEvent {
                owner: &key,
                families: &[],
                owner_payload: Some(&payload),
            }],
        );
    }
    let hook_db = db.clone();
    BEFORE_PUBLISH.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            // grace_ms = 0 below, so the queued run is already due.
            let report = scope.collect_garbage(&hook_db, u64::MAX).unwrap();
            assert_eq!(report.dropped, 1, "the in-progress run was not queued");
        }));
    });
    let report = scope.compact(&db, 0, 0, true).unwrap().unwrap();
    // The first attempt was abandoned; the second wrote and published a new run.
    let runs: Vec<_> = WRITTEN_RUNS
        .lock()
        .iter()
        .filter(|(owner, _)| *owner == [0x70; 16])
        .map(|(_, run)| *run)
        .collect();
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert_eq!(report.manifest_version, 1);
    // A delete reaches reads only after a reopen (engine bug), so check there:
    // the published manifest's run must still exist.
    drop(db);
    let db = Database::open(root.clone()).unwrap();
    let snapshot = scope.population_snapshot(&db).unwrap();
    assert_eq!(snapshot.len(), 20);
    assert!(scope.verify(&db, &[]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The scope's records stay in one pool while the owner log and the runs live
/// in another: counter, manifests and garbage list in `Edges`; generations and
/// runs in `ServerInfo`.
#[test]
fn the_log_and_runs_can_live_in_their_own_pool() {
    let scope =
        ShortIdIndex::new(ShardType::Edges, [0x75; 16]).with_log_pool(ShardType::ServerInfo);
    let root = test_root("log-pool");
    let db = Database::open(root.clone()).unwrap();
    let put = |range: std::ops::Range<u32>| {
        for i in range {
            let key = format!("$e{i}").into_bytes();
            let payload = encode_owner_payload(hash(i));
            record_retrying(
                &scope,
                &db,
                &[BatchEvent {
                    owner: &key,
                    families: &[],
                    owner_payload: Some(&payload),
                }],
            );
        }
    };
    let has = |pool: ShardType, collection: [u8; 16], id| {
        db.begin_transaction()
            .get_with_record_versions(pool, &collection, &[id])
            .unwrap()
            .0[0]
            .is_some()
    };
    put(0..30);
    let generation0 = scope.owner_log_collection(0);
    let entry1 = crate::short_id::owner_log_record_id_for_test(1);
    assert!(has(ShardType::ServerInfo, generation0, entry1));
    assert!(!has(ShardType::Edges, generation0, entry1));

    scope.compact(&db, 0, 0, true).unwrap().unwrap();
    put(30..40);
    let runs = scope.run_collections_for_test(&db).unwrap();
    assert!(runs.iter().any(|(_, present)| *present));
    for (collection, _) in &runs {
        assert!(!has(
            ShardType::Edges,
            *collection,
            run_chunk_id_for_test(0)
        ));
    }
    let snapshot = scope.population_snapshot(&db).unwrap();
    assert_eq!(snapshot.len(), 40);
    assert!(scope.verify(&db, &[]).unwrap().is_consistent());

    assert_eq!(scope.collect_garbage(&db, u64::MAX).unwrap().dropped, 1);
    assert!(!has(ShardType::ServerInfo, generation0, entry1));
    assert_eq!(scope.repack_live_collections(&db).unwrap(), 2);
    assert_eq!(scope.population_snapshot(&db).unwrap().len(), 40);

    scope.purge(&db).unwrap();
    for (collection, _) in &runs {
        assert!(!has(
            ShardType::ServerInfo,
            *collection,
            run_chunk_id_for_test(0)
        ));
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Backfill: a writer commits back to back with no idle gap. Compaction must
/// still land every time it is called. This held without any per-scope lock
/// once refresh-on-miss was off (the O(store) rescans had stretched every
/// publish window); it guards against that regressing.
#[test]
fn a_back_to_back_writer_cannot_starve_compaction() {
    let root = test_root("starve");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    let writer = {
        let db = db.clone();
        std::thread::spawn(move || record(&db, 0..2000))
    };
    let mut landed = 0;
    while !writer.is_finished() {
        match index().compact(&db, 0, GRACE_MS, true) {
            Ok(Some(_)) => landed += 1,
            Ok(None) => {}
            Err(error) => panic!("compaction starved: {error:?}"),
        }
    }
    writer.join().unwrap();
    index().compact(&db, 0, GRACE_MS, true).unwrap();
    assert!(landed >= 1, "compaction never ran against the writer");
    let report = index().verify(&db, &[]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.owners_checked, 2000);
    assert_serves(&index().population_snapshot(&db).unwrap(), 0..2000);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
