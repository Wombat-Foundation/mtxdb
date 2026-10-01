#![cfg(test)]

//! Persisted `AuthClosure` semantics: per-event partial publishing, the
//! coverage tri-state a reader observes, and verify agreeing with both.

use super::auth_closure::{AuthClosure, ClosureOutcome};
use super::closure_store::{self, ClosureCoverage};
use super::database::SharedDatabase;
use super::layout::ShardType;
use super::matrix_adjacency::MatrixAdjacency;
use super::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const ROOM: &str = "!closure-room:example.org";

fn test_root(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("mtxdb-auth-closure-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// A room whose events form a chain off `$create`. `$orphan` references a parent
/// that was never recorded, so its walk is incomplete and cascades nowhere else
/// because it is the only event depending on it.
fn record_room(db: &SharedDatabase) -> MatrixAdjacency {
    let adjacency = MatrixAdjacency::new(POOL, ROOM);
    adjacency
        .record_event(db, "$create", &[], &[], None)
        .unwrap();
    adjacency
        .record_event(db, "$m1", &[], &["$create"], None)
        .unwrap();
    adjacency
        .record_event(db, "$orphan", &[], &["$never-recorded"], None)
        .unwrap();
    adjacency
        .record_event(db, "$m2", &[], &["$create"], None)
        .unwrap();
    adjacency
}

#[test]
fn rebuild_publishes_per_event_and_reports_the_skipped_one() {
    let root = test_root("partial");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let adjacency = record_room(&db);
    let closure = AuthClosure::from_adjacency(adjacency);

    // Recording an event also assigns its `auth` targets a short id, so the
    // never-recorded parent is covered too: it has no `auth` adjacency of its own
    // and therefore cannot be complete either.
    let report = closure.rebuild(&db).unwrap();
    assert_eq!(
        report.source_next, 6,
        "five assigned short ids, so the generation covers 1..6"
    );
    assert_eq!(
        report.count, 3,
        "$create, $m1 and $m2 are complete; the gap and its dependent are not"
    );
    assert!(
        !report.is_complete(),
        "a generation with skipped events is explicitly not complete"
    );
    assert_eq!(report.skipped_count, 2);
    assert_eq!(
        u64::try_from(report.skipped_ids().count()).unwrap(),
        u64::from(report.skipped_count),
        "the expanded ids agree with the count"
    );
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    assert_eq!(
        report.missing,
        vec!["$never-recorded".to_owned()],
        "the root cause is reported separately from the events it skips"
    );
    // The generation is published and internally consistent despite the gap.
    let snapshot = closure.snapshot(&db).unwrap().unwrap();
    assert_eq!(snapshot.head.count, 3);
    assert_eq!(snapshot.head.skipped_count, 2);
    assert!(!snapshot.is_complete());
    let orphan = adjacency
        .short_id(&db, "$orphan")
        .unwrap()
        .expect("recorded event has a short id");
    let never_recorded = adjacency
        .short_id(&db, "$never-recorded")
        .unwrap()
        .expect("an auth target has a short id");
    let (first, last) = if orphan < never_recorded {
        (orphan, never_recorded)
    } else {
        (never_recorded, orphan)
    };
    assert_eq!(
        report.runs,
        vec![(first, last)],
        "the two adjacent skipped ids collapse into one coverage run"
    );
    assert_eq!(
        snapshot.skipped.runs(),
        report.runs,
        "the report's runs are the ones the generation actually stored"
    );
    assert_eq!(
        report.skipped_ids().collect::<Vec<_>>(),
        vec![first, last],
        "skipped_ids walks the runs in ascending order"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn readers_distinguish_complete_incomplete_and_absent() {
    let root = test_root("tri-state");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let adjacency = record_room(&db);
    let closure = AuthClosure::from_adjacency(adjacency);
    closure.rebuild(&db).unwrap();
    let snapshot = closure.snapshot(&db).unwrap().unwrap();

    let short_id = |event_id: &str| {
        adjacency
            .short_id(&db, event_id)
            .unwrap()
            .expect("recorded event has a short id")
    };
    let orphan = short_id("$orphan");
    let create = short_id("$create");
    let missing_parent = short_id("$never-recorded");

    assert_eq!(snapshot.coverage(create), ClosureCoverage::Complete);
    assert_eq!(
        snapshot.coverage(missing_parent),
        ClosureCoverage::Incomplete,
        "the parent that was never recorded is itself incomplete"
    );
    assert_eq!(snapshot.coverage(orphan), ClosureCoverage::Incomplete);
    assert_eq!(
        snapshot.coverage(snapshot.head.source_next),
        ClosureCoverage::Absent,
        "an id past source_next is not covered"
    );

    assert!(
        closure.get(&db, "$orphan").unwrap().is_none(),
        "an incomplete event has no closure record to read"
    );
    assert!(
        closure.get(&db, "$m1").unwrap().is_some(),
        "a complete event still reads back"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_accepts_the_gap_as_intentional() {
    let root = test_root("verify-gap");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let closure = AuthClosure::from_adjacency(record_room(&db));
    closure.rebuild(&db).unwrap();

    let report = closure.verify(&db).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.closures_checked, 3);
    assert_eq!(
        report.skipped_checked, 2,
        "the gap and its dependent were confirmed, not ignored"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_head_that_omits_a_real_gap_is_reported() {
    let root = test_root("verify-omission");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let closure = AuthClosure::from_adjacency(record_room(&db));
    closure.rebuild(&db).unwrap();

    // Republish claiming full coverage while the walk is still incomplete. This
    // needs the store directly, since `rebuild` always records what it skipped.
    let store = closure.store_for_test();
    let mut builder = store.begin(&db).unwrap();
    let graph = closure.graph(&db).unwrap();
    let mut batch: Vec<(u32, Vec<u8>)> = Vec::new();
    for short_id in 1..5 {
        if let super::auth_closure::ClosureOutcome::Complete(set) =
            graph.compute_at(short_id).unwrap()
        {
            batch.push((short_id, set.encode().unwrap()));
        }
    }
    let refs: Vec<(u32, &[u8])> = batch.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    builder.add(&db, &refs).unwrap();
    builder.publish(&db, 5, &[]).unwrap();

    let report = closure.verify(&db).unwrap();
    assert!(!report.is_consistent(), "{:?}", report.problems);
    assert_eq!(
        report.skipped_checked, 0,
        "nothing was claimed skipped, so nothing was confirmed"
    );
    assert!(
        report
            .problems
            .iter()
            .any(|p| p.contains("does not mark it skipped")),
        "{:?}",
        report.problems
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn no_generation_yet_is_not_a_failure() {
    let root = test_root("unpublished");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let closure = AuthClosure::from_adjacency(record_room(&db));
    assert!(closure.head(&db).unwrap().is_none());
    assert!(closure.verify(&db).unwrap().is_consistent());
    assert!(closure.get(&db, "$m1").unwrap().is_none());
    let computed = closure.compute(&db, "$m1").unwrap();
    assert!(
        matches!(computed, ClosureOutcome::Complete(_)),
        "an unpublished store still computes on demand"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_auth_cycle_is_corruption_not_a_gap() {
    let root = test_root("cycle");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let adjacency = MatrixAdjacency::new(POOL, ROOM);
    adjacency
        .record_event(&db, "$a", &[], &["$b"], None)
        .unwrap();
    adjacency
        .record_event(&db, "$b", &[], &["$a"], None)
        .unwrap();
    let closure = AuthClosure::from_adjacency(adjacency);

    let error = closure.rebuild(&db).unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(msg) if msg.contains("cycle")),
        "{error}"
    );
    // Nothing was published, so no reader sees a half-built generation, and the
    // head still decodes.
    assert!(closure.head(&db).unwrap().is_none());
    assert!(closure.verify(&db).unwrap().is_consistent());

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_cycle_abandons_its_generation_instead_of_leaving_an_orphan() {
    let root = test_root("cycle-orphan");
    let db = SharedDatabase::open(root.clone()).unwrap();
    // A good generation first, so the orphan left by the cycle is distinguishable
    // from the retained generations.
    let good = AuthClosure::from_adjacency(record_room(&db));
    let published = good.rebuild(&db).unwrap();
    let retained_generation = published.generation;

    let cyclic = MatrixAdjacency::new(POOL, ROOM);
    cyclic.record_event(&db, "$x", &[], &["$y"], None).unwrap();
    cyclic.record_event(&db, "$y", &[], &["$x"], None).unwrap();
    let closure = AuthClosure::from_adjacency(cyclic);
    assert!(closure
        .rebuild(&db)
        .unwrap_err()
        .to_string()
        .contains("cycle"));

    // The old head is untouched and still decodes.
    let head = closure.head(&db).unwrap().unwrap();
    assert_eq!(head.generation, published.generation);
    assert_eq!(head.count, published.count);
    // The cycle's reserved generation was abandoned, not left holding the
    // records staged before the walk failed.
    let ids: Vec<u32> = (1..=4).collect();
    assert!(
        !closure_store::test::generation_exists(
            &closure.store_for_test(),
            &db,
            head.generation + 1,
            &ids
        )
        .unwrap(),
        "the failed rebuild reclaimed its reserved generation"
    );
    assert!(
        closure_store::test::generation_exists(
            &closure.store_for_test(),
            &db,
            retained_generation,
            &ids
        )
        .unwrap(),
        "the retained generation it superseded is still readable"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
