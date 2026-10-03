use bytes::Bytes;

use super::closure_store;
use super::closure_store::*;
use super::short_id::{EdgeFamily, ShortIdIndex};
use crate::database::Database;
use crate::layout::ShardType;
use crate::storage::{NodeData, StorageError};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

fn build(db: &Database, tag: u8, ids: std::ops::Range<u32>) -> ClosureHead {
    let store = store();
    let mut builder = store.begin(db).unwrap();
    let blobs: Vec<(u32, Vec<u8>)> = ids
        .clone()
        .map(|id| {
            (
                id,
                vec![
                    tag,
                    u8::try_from(id).expect("test closure id must fit in a byte"),
                ],
            )
        })
        .collect();
    let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
    builder.add(db, &refs).unwrap();
    builder.publish(db, ids.end, &[]).unwrap()
}

#[test]
fn unpublished_generation_is_invisible_then_published_atomically() {
    let root = test_root("publish");
    let db = open_db(&root);
    assert_eq!(store().head(&db).unwrap(), None);
    assert_eq!(store().get(&db, 1).unwrap(), None);

    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"one".as_slice())]).unwrap();
    assert_eq!(
        store().get(&db, 1).unwrap(),
        None,
        "not visible before publish"
    );
    let head = builder.publish(&db, 2, &[]).unwrap();
    assert_eq!((head.generation, head.previous, head.count), (1, 0, 1));
    assert_eq!(store().get(&db, 1).unwrap(), Some(b"one".to_vec()));
    assert!(store().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn republish_swaps_generation_and_retirement_keeps_the_predecessor() {
    let root = test_root("swap");
    let db = open_db(&root);
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
    let db = open_db(&root);
    let store = store();
    let mut slow = store.begin(&db).unwrap();
    slow.add(&db, &[(1, b"slow".as_slice())]).unwrap();
    // A faster builder publishes first.
    build(&db, 9, 1..3);
    let error = slow.publish(&db, 2, &[]).unwrap_err();
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
        let db = open_db(&root);
        build(&db, 4, 1..4);
        // A crash mid-build: generation written, head never swapped. The
        // builder has no destructor; ending this scope models abandoning it.
        {
            let mut crashed = store().begin(&db).unwrap();
            crashed.add(&db, &[(1, b"partial".as_slice())]).unwrap();
        }
    }
    let db = open_db(&root);
    let head = store().head(&db).unwrap().unwrap();
    assert_eq!((head.generation, head.count, head.source_next), (1, 3, 4));
    assert_eq!(store().get(&db, 3).unwrap(), Some(vec![4, 3]));
    assert!(store().verify(&db).unwrap().is_consistent());
    // Generation 2 sits above the head, so it is indistinguishable from a live
    // builder's reservation and is left alone.
    assert_eq!(store().retire_superseded(&db).unwrap(), 0);
    // Once a later publish moves the head past it, it is a plain orphan.
    build(&db, 5, 1..4);
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
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    assert!(builder.add(&db, &[(1, b"".as_slice())]).is_err());
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    // Claims to cover ids 1..4 but only one closure was written.
    builder.publish(&db, 4, &[]).unwrap();
    let report = store().verify(&db).unwrap();
    assert!(!report.is_consistent());
    assert_eq!(report.records_checked, 1);
    assert!(report.problems.len() >= 2, "{:?}", report.problems);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn skipped_coverage_round_trips_and_separates_the_three_states() {
    let root = test_root("coverage");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    // Cover 1..8, store closures for all but 3 and 4.
    let blobs: Vec<(u32, Vec<u8>)> = [1u32, 2, 5, 6, 7]
        .iter()
        .map(|id| {
            let byte = u8::try_from(*id).expect("test closure id must fit in a byte");
            (*id, vec![byte])
        })
        .collect();
    let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    builder.add(&db, &refs).unwrap();
    builder.publish(&db, 8, &[4, 3]).unwrap();

    let snapshot = store().snapshot(&db).unwrap().unwrap();
    assert_eq!(
        snapshot.skipped.runs(),
        &[(3, 4)],
        "consecutive ids collapse into one run"
    );
    assert_eq!(snapshot.skipped.len(), 2);
    assert_eq!(snapshot.head.skipped_count, 2);
    assert_eq!(
        snapshot.coverage(2),
        ClosureCoverage::Complete,
        "covered with a record"
    );
    assert_eq!(
        snapshot.coverage(3),
        ClosureCoverage::Incomplete,
        "covered but deliberately has no record"
    );
    assert_eq!(
        snapshot.coverage(8),
        ClosureCoverage::Absent,
        "outside 1..source_next"
    );
    assert_eq!(
        snapshot.coverage(0),
        ClosureCoverage::Absent,
        "id 0 is never covered"
    );
    // The head itself carries only the count, so it cannot scale with the room.
    let head = store().head(&db).unwrap().unwrap();
    assert!(!head.is_complete());
    assert!(store().verify(&db).unwrap().is_consistent());
    assert_eq!(
        store().get(&db, 3).unwrap(),
        None,
        "skipped ids have no record"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn publish_rejects_an_id_that_is_both_stored_and_skipped() {
    let root = test_root("coverage-stray");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    // Id 1 is both stored and claimed skipped. Coverage resolves first, so a
    // stored record there would be unreachable; publish rejects the overlap
    // rather than producing a generation `verify` would later flag.
    let error = builder.publish(&db, 2, &[1]).unwrap_err();
    assert!(
        matches!(&error, StorageError::Internal(msg) if msg.contains("stored and claimed as skipped")),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn publish_rejects_skipped_ids_outside_the_covered_range() {
    let root = test_root("coverage-range");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    for bad in [0u32, 4, 9] {
        let error = builder.publish(&db, 4, &[bad]).unwrap_err();
        assert!(matches!(error, StorageError::Internal(_)), "{error}");
        // A lost generation is unusable, so re-begin for each attempt.
        builder = store().begin(&db).unwrap();
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn get_many_serves_skipped_and_absent_without_retrying() {
    let root = test_root("get-many-coverage");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    // Cover 1..6, store 1, 3 and 5, skip 2 and 4.
    let blobs: Vec<(u32, Vec<u8>)> = [1u32, 3, 5]
        .iter()
        .map(|id| {
            let byte = u8::try_from(*id).expect("test closure id must fit in a byte");
            (*id, vec![byte])
        })
        .collect();
    let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    builder.add(&db, &refs).unwrap();
    builder.publish(&db, 6, &[4, 2]).unwrap();

    // 2 and 4 are skipped, 6 is absent, 0 is absent; 1, 3, 5 are complete.
    let (generation, out) = store().get_many(&db, &[1, 2, 3, 4, 5, 6, 0]).unwrap();
    assert_eq!(generation, Some(1));
    assert_eq!(out.len(), 7);
    assert_eq!(out[0], Some(vec![1]));
    assert_eq!(out[1], None, "skipped");
    assert_eq!(out[2], Some(vec![3]));
    assert_eq!(out[3], None, "skipped");
    assert_eq!(out[4], Some(vec![5]));
    assert_eq!(out[5], None, "absent");
    assert_eq!(out[6], None, "id 0 is absent");

    // get_many and the pinned read agree, including on the None cases.
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    let pinned = store().get_many_pinned(&db, &snapshot, &[1, 2, 6]).unwrap();
    assert_eq!(pinned[0], Some(vec![1]));
    assert_eq!(pinned[1], None);
    assert_eq!(pinned[2], None);

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn get_many_all_skipped_needs_no_record_read() {
    let root = test_root("get-many-all-skipped");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    builder.publish(&db, 4, &[2, 3]).unwrap();
    // Nothing here is complete, so the answer comes from coverage alone.
    let (generation, out) = store().get_many(&db, &[2, 3]).unwrap();
    assert_eq!(generation, Some(1));
    assert_eq!(out, vec![None, None]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn get_many_reports_a_missing_record_for_a_complete_id() {
    let root = test_root("get-many-corrupt");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    builder
        .add(&db, &[(1, b"a".as_slice()), (2, b"b".as_slice())])
        .unwrap();
    // Claims full coverage of 1..4 but only wrote two records, so the head lies
    // about id 3 rather than recording it skipped.
    builder.publish(&db, 4, &[]).unwrap();

    let (generation, out) = store().get_many(&db, &[1, 2]).unwrap();
    assert_eq!(generation, Some(1), "the two stored ids still read back");
    assert_eq!(out, vec![Some(vec![0x61]), Some(vec![0x62])]);

    // Id 3 is covered and complete per the head, but has no record: that is
    // corruption, and it must surface as an error rather than a silent None.
    let error = store().get_many(&db, &[3]).unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(msg) if msg.contains("as complete")),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn get_many_retries_when_the_head_moves_under_it() {
    let root = test_root("get-many-moved");
    let db = open_db(&root);
    build(&db, 7, 1..3);

    // Republish *between* the head read and the record read. Without the hook
    // this test would pass even if `get_many` mixed generations, because the
    // publish would simply win the race before the read even started.
    // Publish a new generation from inside the read, after it resolved the old
    // head. Without this the test would pass even if `get_many` mixed
    // generations, because the publish would win the race before the read began.
    // Publishing once is not enough: generation 1 is still retained as the
    // predecessor, so its records read back fine and `get_many` is entitled to
    // return generation 1 as a stable snapshot. Publishing twice and retiring
    // deletes generation 1's collection, which is what forces the retry.
    let published = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&published);
    closure_store::test::arm_snapshot_hook(move |db| {
        for tag in [9u8, 10] {
            let mut builder = store().begin(db).unwrap();
            let blobs: Vec<(u32, Vec<u8>)> = (1..4u32)
                .map(|id| (id, vec![tag, u8::try_from(id).unwrap()]))
                .collect();
            let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
            builder.add(db, &refs).unwrap();
            let head = builder.publish(db, 4, &[]).unwrap();
            *slot.lock().unwrap() = Some(head);
        }
        store().retire_superseded(db).unwrap();
    });
    let (generation, out) = store().get_many(&db, &[1, 2, 3]).unwrap();
    closure_store::test::disarm_snapshot_hook();
    assert_eq!(
        generation,
        Some(published.lock().unwrap().unwrap().generation),
        "retried against the surviving generation"
    );
    assert_eq!(
        out,
        vec![Some(vec![10, 1]), Some(vec![10, 2]), Some(vec![10, 3])],
        "answered entirely from the new generation, never a mix"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn pinned_reads_skip_the_head_reresolve_loop() {
    let root = test_root("coverage-pinned");
    let db = open_db(&root);
    build(&db, 7, 1..4);
    // 4 is not covered at all, so it is absent without any head retry.
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    let _ = closure_store::test::take_head_reads();
    let blobs = store().get_many_pinned(&db, &snapshot, &[1, 2, 4]).unwrap();
    assert_eq!(
        closure_store::test::take_head_reads(),
        0,
        "a pinned read performs no head read of its own"
    );
    assert_eq!(blobs.len(), 3);
    assert!(blobs[0].is_some() && blobs[1].is_some());
    assert!(blobs[2].is_none());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_pinned_read_of_a_retired_generation_is_stale_not_corrupt() {
    let root = test_root("pinned-retired");
    let db = open_db(&root);
    build(&db, 7, 1..4);
    let pinned = store().snapshot(&db).unwrap().unwrap();
    assert_eq!(pinned.head.generation, 1);

    // Two more publishes push the pinned generation out of the retained pair.
    build(&db, 8, 1..4);
    build(&db, 9, 1..4);
    store().retire_superseded(&db).unwrap();

    // Id 1 is Complete in the pinned snapshot, but its generation is gone. That
    // is a stale reader, not corruption: the head moved.
    let error = store().get_many_pinned(&db, &pinned, &[1]).unwrap_err();
    assert!(
        matches!(
            &error,
            StorageError::StaleGeneration {
                generation,
                current: Some(3)
            } if *generation == 1
        ),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_pinned_read_of_an_unmoved_generation_with_no_record_is_corrupt() {
    let root = test_root("pinned-corrupt");
    let db = open_db(&root);
    // Claims full coverage of 1..4 but stores only 1 and 2.
    let mut builder = store().begin(&db).unwrap();
    builder
        .add(&db, &[(1, b"a".as_slice()), (2, b"b".as_slice())])
        .unwrap();
    builder.publish(&db, 4, &[]).unwrap();
    let snapshot = store().snapshot(&db).unwrap().unwrap();

    // Id 3 is Complete and the head has not moved, so this is corruption.
    let error = store().get_many_pinned(&db, &snapshot, &[3]).unwrap_err();
    assert!(
        matches!(&error, StorageError::Corrupt(msg) if msg.contains("as complete")),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn publish_rejects_a_stored_id_outside_the_covered_range() {
    let root = test_root("stored-range");
    let db = open_db(&root);
    let first = build(&db, 7, 1..4);

    // Id 10 is stored but the generation claims to cover only 1..5, so the
    // record would be unreachable: its count fits, so the head decodes, and only
    // verify would notice. Reject at publish instead.
    let mut builder = store().begin(&db).unwrap();
    let reserved = builder.generation();
    builder.add(&db, &[(10, b"j".as_slice())]).unwrap();
    let error = builder.publish(&db, 5, &[]).unwrap_err();
    assert!(
        matches!(&error, StorageError::Internal(msg) if msg.contains("lie outside")),
        "{error}"
    );

    // The refused publish left the previous head readable and untouched.
    let head = store().head(&db).unwrap().unwrap();
    assert_eq!(head.generation, first.generation);
    assert_eq!(store().get(&db, 1).unwrap(), Some(vec![7, 1]));
    assert!(
        !closure_store::test::generation_exists(&store(), &db, reserved, &[10]).unwrap(),
        "the rejected generation discarded the record it had already staged, \
         rather than leaving an orphan no head names"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn every_published_head_partitions_its_covered_range() {
    let root = test_root("partition");
    let db = open_db(&root);
    let first = build(&db, 7, 1..4);

    // Stored ids outside the range and stored/skipped overlap are both rejected,
    // and together they make the count-partition check unreachable through
    // `publish`. So assert the invariant the check defends: for every generation
    // `publish` accepts, stored plus skipped covers exactly `1..source_next`,
    // which is what `decode_head` requires and what `verify` counts against.
    for (source_next, stored, skipped) in [
        (4u32, vec![1u32, 2, 3], Vec::new()),
        (8, vec![1, 6], vec![2, 3, 4, 5, 7]),
        (5, vec![1, 2], vec![3, 4]),
        (3, vec![1, 2], Vec::new()),
    ] {
        let mut builder = store().begin(&db).unwrap();
        let blobs: Vec<(u32, Vec<u8>)> = stored
            .iter()
            .map(|&id| {
                let byte = u8::try_from(id).expect("test closure id must fit in a byte");
                (id, vec![byte])
            })
            .collect();
        let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(id, b)| (*id, b.as_slice())).collect();
        builder.add(&db, &refs).unwrap();
        let head = builder.publish(&db, source_next, &skipped).unwrap();
        assert_eq!(
            u64::from(head.count).saturating_add(u64::from(head.skipped_count)),
            u64::from(source_next).saturating_sub(1),
            "stored plus skipped must partition 1..{source_next}"
        );
        assert!(store().verify(&db).unwrap().is_consistent());
    }

    // The refused out-of-range publish earlier in this file left the first head
    // intact; nothing above disturbed it beyond superseding it.
    assert!(store().head(&db).unwrap().unwrap().generation > first.generation);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_coverage_record_round_trips_and_a_zero_count_writes_none() {
    let root = test_root("coverage-record");
    let db = open_db(&root);

    // No skipped ids: publish must not write a coverage record at all, and the
    // snapshot must still resolve to the empty set.
    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    let complete = builder.publish(&db, 2, &[]).unwrap();
    assert_eq!(complete.skipped_count, 0);
    assert!(complete.is_complete());
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    assert!(snapshot.skipped.is_empty());
    assert_eq!(snapshot.skipped.runs(), &[]);
    assert!(store()
        .raw_coverage_record(&db, complete.generation)
        .is_none());

    // Now a real gap: the runs round trip through the record. Stored and skipped
    // must partition 1..8 exactly, so id 5 is skipped too.
    let mut builder = store().begin(&db).unwrap();
    builder
        .add(&db, &[(1, b"a".as_slice()), (6, b"f".as_slice())])
        .unwrap();
    let gapped = builder.publish(&db, 8, &[2, 3, 4, 5, 7]).unwrap();
    assert_eq!(gapped.skipped_count, 5);
    assert!(store()
        .raw_coverage_record(&db, gapped.generation)
        .is_some());
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    assert_eq!(snapshot.skipped.runs(), &[(2, 5), (7, 7)]);
    assert_eq!(snapshot.skipped.len(), 5);
    assert_eq!(snapshot.head.skipped_count, 5);
    assert!(store().verify(&db).unwrap().is_consistent());

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_reports_a_head_count_that_disagrees_with_the_coverage_record() {
    let root = test_root("coverage-mismatch");
    let db = open_db(&root);
    let mut builder = store().begin(&db).unwrap();
    builder.add(&db, &[(1, b"a".as_slice())]).unwrap();
    let head = builder.publish(&db, 5, &[2, 3]).unwrap();
    assert_eq!(head.skipped_count, 2);

    // Rewrite the coverage record so it no longer matches the head's count. This
    // is exactly what a partially applied publish would leave behind.
    let txn = db.begin_transaction();
    txn.put(
        POOL,
        closure_store::test::generation_collection(&store(), head.generation),
        closure_store::test::coverage_id(),
        &NodeData::new(Bytes::from(closure_store::test::encode_coverage_for_test(
            &ClosureCoverageSet::from_ids(&[2]),
        ))),
    )
    .unwrap();
    txn.commit().unwrap();

    // verify reports the mismatch as a problem; it does not abort.
    let report = store().verify(&db).unwrap();
    assert!(!report.is_consistent(), "{:?}", report.problems);
    assert!(
        report.problems.iter().any(|p| p.contains("snapshot")),
        "{:?}",
        report.problems
    );

    // A reader cannot build a snapshot at all, which is the right answer: the
    // generation's own accounting is inconsistent.
    let error = store().snapshot(&db).unwrap_err();
    assert!(matches!(&error, StorageError::Corrupt(_)), "{error}");

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn snapshot_retries_when_the_head_moves_under_it() {
    let root = test_root("snapshot-race");
    let db = open_db(&root);

    // Generation 1 must have skipped ids, so `snapshot` actually reads a coverage
    // record. With `skipped_count == 0` there is no coverage read to race, and a
    // superseded-but-consistent generation is a legitimate answer, not a retry.
    let mut first = store().begin(&db).unwrap();
    first
        .add(&db, &[(1, b"a".as_slice()), (4, b"d".as_slice())])
        .unwrap();
    let first_head = first.publish(&db, 5, &[2, 3]).unwrap();
    assert_eq!(first_head.skipped_count, 2);

    // Publish two replacements and retire from inside the snapshot, after it read
    // the head and the coverage record. Publishing once is not enough: generation
    // 1 would survive as the predecessor and stay readable. Retiring is what
    // deletes its coverage record and produces the count mismatch.
    let published = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&published);
    closure_store::test::arm_snapshot_hook(move |db| {
        for tag in [9u8, 10] {
            let mut builder = store().begin(db).unwrap();
            let blobs: Vec<(u32, Vec<u8>)> = (1..4u32)
                .map(|id| (id, vec![tag, u8::try_from(id).unwrap()]))
                .collect();
            let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
            builder.add(db, &refs).unwrap();
            *slot.lock().unwrap() = Some(builder.publish(db, 4, &[]).unwrap());
        }
        store().retire_superseded(db).unwrap();
    });
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    closure_store::test::disarm_snapshot_hook();

    // Before this changed, a generation retired between the head read and the
    // coverage read produced a false Corrupt. Now the count mismatch is
    // recognized as a move and the snapshot comes from the new generation.
    let fresh = published.lock().unwrap().unwrap();
    assert_eq!(snapshot.head.generation, fresh.generation);
    assert_eq!(snapshot.head.count, 3);
    assert!(snapshot.skipped.is_empty());

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_complete_generation_costs_one_head_read_and_no_coverage_read() {
    let root = test_root("read-count");
    let db = open_db(&root);
    build(&db, 7, 1..4);

    let _ = closure_store::test::take_head_reads();
    let snapshot = store().snapshot(&db).unwrap().unwrap();
    assert_eq!(
        closure_store::test::take_head_reads(),
        1,
        "a complete generation reads the head once and skips the coverage record"
    );
    assert!(snapshot.skipped.is_empty());

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn scope_purge_is_atomic_with_the_short_id_scope() {
    let root = test_root("purge");
    let db = open_db(&root);
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
// Crash injection: a real child process is stopped at an exact step (no
// destructors, no flush), and the parent reopens the store. The invariant in
// every case: readers see the old complete generation or the new complete one,
// never a partial or mixed generation.
// ---------------------------------------------------------------------------

const CRASH_IDS: u32 = 7;

/// Blob for `id` in `generation`: lets a reader detect a mixed generation.
fn blob_for(generation: u64, id: u32) -> Vec<u8> {
    vec![
        u8::try_from(generation).expect("test generation must fit in a byte"),
        u8::try_from(id).expect("test closure id must fit in a byte"),
    ]
}

fn publish_full(db: &Database) -> ClosureHead {
    let mut builder = store().begin(db).unwrap();
    let generation = builder.generation();
    for batch in [1..4u32, 4..CRASH_IDS] {
        let blobs: Vec<(u32, Vec<u8>)> = batch.map(|id| (id, blob_for(generation, id))).collect();
        let refs: Vec<(u32, &[u8])> = blobs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
        builder.add(db, &refs).unwrap();
    }
    builder.publish(db, CRASH_IDS, &[]).unwrap()
}

/// Child-process body. A no-op unless the parent set `MTXDB_CRASH_SCENARIO`.
#[test]
fn crash_child() {
    let Ok(scenario) = std::env::var("MTXDB_CRASH_SCENARIO") else {
        return;
    };
    let root = PathBuf::from(std::env::var("MTXDB_CRASH_ROOT").unwrap());
    let db = open_db(&root);
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

/// Pause between `open_db` attempts while the writer lock is still held.
const OPEN_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);

/// Attempts `open_db` makes before giving up: `OPEN_RETRY_ATTEMPTS` x
/// `OPEN_RETRY_INTERVAL` is a ~10 s ceiling. It is only a ceiling; the lock
/// clears within milliseconds in practice, so it just turns a lock that is
/// genuinely held (a leaked `Database`) into a panic instead of a hang.
const OPEN_RETRY_ATTEMPTS: u32 = 2000;

/// Open `root`, retrying while its writer lock is briefly still held.
///
/// The lock is an `flock` on an open file description. This module spawns
/// child processes from parallel test threads, and a child inherits every open
/// fd, including another test's lock file, until it reaches `exec`. So a lock
/// can outlive its owner's `drop(db)` by a moment even with nothing else
/// running on that root, and a plain open fails with `WouldBlock`. A lock that
/// is really held stays held, so the retry is bounded.
fn open_db(root: &Path) -> Database {
    let mut last = None;
    for _ in 0..OPEN_RETRY_ATTEMPTS {
        match Database::open(root.to_path_buf()) {
            Ok(db) => return db,
            Err(error)
                if error.is_would_block()
                    || matches!(&error, StorageError::Io(io) if io.kind() == std::io::ErrorKind::WouldBlock) =>
            {
                last = Some(error);
                std::thread::sleep(OPEN_RETRY_INTERVAL);
            }
            Err(error) => panic!("open {}: {error}", root.display()),
        }
    }
    panic!("open {}: {:?}", root.display(), last);
}

/// Run `scenario` in a child that stops at `crash_at`; returns whether it exited
/// with `CRASH_EXIT_CODE` (the injected crash), as opposed to a panic or a clean
/// exit.
fn run_child(root: &PathBuf, scenario: &str, crash_at: &str) -> bool {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "test_closure_store::crash_child",
            "--test-threads=1",
        ])
        .env("MTXDB_CRASH_SCENARIO", scenario)
        .env("MTXDB_CRASH_ROOT", root)
        .env("MTXDB_CRASH_AT", crash_at)
        .output()
        .unwrap();
    let crashed = output.status.code() == Some(CRASH_EXIT_CODE);
    if !crashed {
        eprintln!(
            "child for {scenario}/{crash_at} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    crashed
}

/// Control: with no matching crash point the child completes normally, so the
/// crash tests cannot pass just because the child always fails.
#[test]
fn crash_child_without_a_crash_point_succeeds() {
    let root = test_root("crash-control");
    {
        let db = open_db(&root);
        publish_full(&db);
    }
    assert!(!run_child(&root, "rebuild", "no-such-point"));
    let db = open_db(&root);
    assert_eq!(assert_complete_snapshot(&db).generation, 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The published generation is complete, uniform and self-consistent.
fn assert_complete_snapshot(db: &Database) -> ClosureHead {
    let head = store()
        .head(db)
        .unwrap()
        .expect("a generation is published");
    let ids: Vec<u32> = (1..head.source_next).collect();
    let (generation, blobs) = store().get_many(db, &ids).unwrap();
    assert_eq!(
        generation,
        Some(head.generation),
        "all closure records must come from the published generation"
    );
    for (id, blob) in ids.iter().zip(blobs) {
        assert_eq!(
            blob,
            Some(blob_for(head.generation, *id)),
            "id {id} must come from generation {}",
            head.generation
        );
    }
    assert!(
        store().verify(db).unwrap().is_consistent(),
        "published closure generation must verify"
    );
    head
}

fn crash_rebuild_keeps_old_generation(name: &str, crash_at: &str, written: &[u32]) {
    let root = test_root(name);
    {
        let db = open_db(&root);
        publish_full(&db);
    }
    assert!(
        run_child(&root, "rebuild", crash_at),
        "child must crash at {crash_at}"
    );
    let db = open_db(&root);
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
        let db = open_db(&root);
        publish_full(&db);
    }
    assert!(run_child(&root, "rebuild", "publish-after-commit"));
    let db = open_db(&root);
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
        let db = open_db(&root);
        for _ in 0..3 {
            publish_full(&db);
        }
    }
    assert!(
        run_child(&root, "retire", crash_at),
        "child must crash at {crash_at}"
    );
    let db = open_db(&root);
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
    let db = open_db(&root);
    publish_full(&db);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..3)
            .map(|_| {
                scope.spawn(|| {
                    let ids: Vec<u32> = (1..CRASH_IDS).collect();
                    let mut reads = 0u32;
                    // Read at least once even if the writer finishes before this
                    // thread is scheduled, so a loaded machine cannot fail the
                    // `reads > 0` check below.
                    loop {
                        // `WouldBlock` is the documented "versions kept advancing,
                        // retry" answer, which a busy machine can produce when the
                        // writer outpaces a read's bounded attempts.
                        let (generation, blobs) = match store().get_many(&db, &ids) {
                            Ok(read) => read,
                            Err(error) if error.is_would_block() => continue,
                            Err(error) => panic!("get_many: {error}"),
                        };
                        let generation = generation.expect("a generation is always published");
                        for (id, blob) in ids.iter().zip(&blobs) {
                            assert_eq!(
                                blob.as_deref(),
                                Some(blob_for(generation, *id).as_slice()),
                                "mixed or partial generation for id {id}"
                            );
                        }
                        reads += 1;
                        if done.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
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

/// Closure record ids must be spread in the bytes the engine's hash index uses
/// (see `short_id`'s test of the same property): a constant prefix put every
/// closure of a room into one bucket.
#[test]
fn closure_record_ids_vary_in_the_bytes_the_index_uses() {
    const IDS: u32 = 10_000;
    let mut buckets = std::collections::HashSet::new();
    let mut tags = std::collections::HashSet::new();
    for short_id in 1..=IDS {
        let id = record_id_for_test(short_id);
        buckets.insert(<[u8; 8]>::try_from(&id[..8]).unwrap());
        tags.insert(<[u8; 4]>::try_from(&id[8..12]).unwrap());
    }
    let total = usize::try_from(IDS).unwrap();
    assert!(
        buckets.len() >= total - 1,
        "bucket bytes collide: {}",
        buckets.len()
    );
    assert!(tags.len() >= total - 1, "tag bytes collide: {}", tags.len());
}
