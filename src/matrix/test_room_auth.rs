#![cfg(test)]

//! The room auth façade: ingest, lazy and persisted auth chains, pinned
//! snapshots, rebuild/retire/verify/purge, and the error contract.

use crate::database::Database;
use crate::layout::ShardType;
use crate::matrix_adjacency::{AlwaysVisible, RelationRef};
use crate::room_auth::tests::take_edge_reads;
use crate::room_auth::{NewEvent, RoomAuth, RoomAuthError};
use crate::short_id::SHORT_ID_MAX;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const ROOM: &str = "!facade:example.org";

fn root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-room-auth-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn event<'a>(event_id: &'a str, auth: &'a [&'a str]) -> NewEvent<'a> {
    NewEvent {
        event_id,
        prev: &[],
        auth,
        relation: None,
    }
}

/// A diamond: `$create` is the parent of `$member` and `$power`, and `$msg`
/// authorizes against all three.
fn record_diamond(room: &RoomAuth, db: &Database) {
    for (id, auth) in [
        ("$create", &[][..]),
        ("$member", &["$create"][..]),
        ("$power", &["$create"][..]),
        ("$msg", &["$create", "$member", "$power"][..]),
    ] {
        room.record_event(db, &event(id, auth)).unwrap();
    }
}

#[test]
fn a_chain_is_ancestors_only_before_any_generation_exists() {
    let dir = root("lazy");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);

    assert_eq!(room.snapshot(&db).unwrap().generation(), None);
    let mut chain = room.auth_chain(&db, "$msg").unwrap();
    chain.sort();
    assert_eq!(chain, vec!["$create", "$member", "$power"]);
    assert_eq!(
        room.auth_chain(&db, "$create").unwrap(),
        Vec::<String>::new()
    );

    assert!(room.is_in_auth_chain(&db, "$create", "$msg").unwrap());
    assert!(!room.is_in_auth_chain(&db, "$msg", "$msg").unwrap());
    assert!(
        !room.is_in_auth_chain(&db, "$msg", "$create").unwrap(),
        "a descendant is not an ancestor"
    );
    assert!(
        !room.is_in_auth_chain(&db, "$never-seen", "$msg").unwrap(),
        "an ancestor the room never saw is in no chain"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persisted_and_lazy_chains_agree() {
    let dir = root("agree");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);
    let lazy: Vec<_> = ["$create", "$member", "$power", "$msg"]
        .iter()
        .map(|id| room.auth_chain_ids(&db, id).unwrap())
        .collect();

    let report = room.rebuild(&db).unwrap();
    assert!(report.is_complete());
    let snapshot = room.snapshot(&db).unwrap();
    assert_eq!(snapshot.generation(), Some(report.generation));
    for (id, expected) in ["$create", "$member", "$power", "$msg"].iter().zip(&lazy) {
        assert!(snapshot.is_covered(&db, id).unwrap());
        assert_eq!(&snapshot.auth_chain_ids(&db, id).unwrap(), expected, "{id}");
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_query_for_a_new_event_never_walks_the_covered_history() {
    let dir = root("bounded");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    // A long covered chain: e0 <- e1 <- ... <- e49.
    let ids: Vec<String> = (0..50).map(|i| format!("$e{i}")).collect();
    for (index, id) in ids.iter().enumerate() {
        let auth: Vec<&str> = if index == 0 {
            Vec::new()
        } else {
            vec![ids[index - 1].as_str()]
        };
        room.record_event(&db, &event(id, &auth)).unwrap();
    }
    room.rebuild(&db).unwrap();

    // One event recorded after the generation, authorized by the chain's tip.
    room.record_event(&db, &event("$new", &["$e49"])).unwrap();

    let _ = take_edge_reads();
    let chain = room.auth_chain(&db, "$new").unwrap();
    let reads = take_edge_reads();
    assert_eq!(chain.len(), 50, "the whole covered chain is in the answer");
    assert!(
        reads <= 2,
        "only the uncovered frontier is walked, not the {} covered events: {reads} edge reads",
        ids.len()
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn absent_events_are_told_apart() {
    let dir = root("absent");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    // `$leaf` is recorded with an empty auth list; `$ghost` is only referenced.
    room.record_event(&db, &event("$leaf", &[])).unwrap();
    room.record_event(&db, &event("$child", &["$ghost"]))
        .unwrap();

    assert_eq!(room.auth_edges(&db, "$leaf").unwrap(), Vec::<String>::new());
    assert_eq!(room.auth_edges(&db, "$child").unwrap(), vec!["$ghost"]);
    assert!(matches!(
        room.auth_edges(&db, "$ghost").unwrap_err(),
        RoomAuthError::EventNotRecorded { .. }
    ));
    assert!(matches!(
        room.auth_edges(&db, "$nobody").unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));

    // A chain through an unrecorded parent names it; it is not an empty chain.
    let error = room.auth_chain(&db, "$child").unwrap_err();
    assert!(
        matches!(&error, RoomAuthError::MissingParent { event_id } if event_id == "$ghost"),
        "{error}"
    );
    assert!(matches!(
        room.auth_chain(&db, "$ghost").unwrap_err(),
        RoomAuthError::EventNotRecorded { .. }
    ));
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_auth_cycle_is_corruption() {
    let dir = root("cycle");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    room.record_event(&db, &event("$a", &["$b"])).unwrap();
    room.record_event(&db, &event("$b", &["$a"])).unwrap();
    let error = room.auth_chain(&db, "$a").unwrap_err();
    assert!(
        matches!(&error, RoomAuthError::Corruption(message) if message.contains("cycle")),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn visibility_filters_relations_but_never_the_auth_chain() {
    let dir = root("visibility");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);
    room.record_event(
        &db,
        &NewEvent {
            event_id: "$edit",
            prev: &[],
            auth: &["$create"],
            relation: Some(RelationRef {
                target: "$msg",
                rel_type: "m.replace",
            }),
        },
    )
    .unwrap();

    let hide_everything = |_: &str| false;
    assert!(room
        .relation_of(&db, "$edit", &AlwaysVisible)
        .unwrap()
        .is_some());
    assert!(room
        .relation_of(&db, "$edit", &hide_everything)
        .unwrap()
        .is_none());

    // Disposition changes what a relationship query shows. It cannot change
    // `auth_events`, so the auth answers are the same either way.
    assert_eq!(
        room.auth_edges(&db, "$edit").unwrap(),
        vec!["$create"],
        "a redacted or rejected event keeps its auth_events"
    );
    assert_eq!(room.auth_chain(&db, "$edit").unwrap(), vec!["$create"]);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pinned_snapshot_goes_stale_when_its_generation_is_retired() {
    let dir = root("stale");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);
    let first = room.rebuild(&db).unwrap();
    let pinned = room.snapshot(&db).unwrap();
    assert_eq!(pinned.generation(), Some(first.generation));
    assert_eq!(pinned.auth_chain(&db, "$msg").unwrap().len(), 3);

    room.rebuild(&db).unwrap();
    let third = room.rebuild(&db).unwrap();
    assert_eq!(room.retire_old_generations(&db).unwrap(), 1);

    let error = pinned.auth_chain(&db, "$msg").unwrap_err();
    assert!(
        matches!(
            &error,
            RoomAuthError::StaleGeneration { pinned, current }
                if *pinned == first.generation && *current == Some(third.generation)
        ),
        "{error}"
    );
    // A fresh query is unaffected: it pins the current generation.
    assert_eq!(room.auth_chain(&db, "$msg").unwrap().len(), 3);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn purge_removes_the_room_and_expires_held_snapshots() {
    let dir = root("purge");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    let other = RoomAuth::new(POOL, "!other:example.org");
    record_diamond(&room, &db);
    record_diamond(&other, &db);
    let generation = room.rebuild(&db).unwrap().generation;
    other.rebuild(&db).unwrap();
    let pinned = room.snapshot(&db).unwrap();

    room.purge(&db).unwrap();

    assert!(matches!(
        room.auth_edges(&db, "$msg").unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));
    assert_eq!(room.snapshot(&db).unwrap().generation(), None);
    // With the adjacency gone the event is unknown, so a held snapshot answers
    // that, never stale data.
    let error = pinned.auth_chain(&db, "$msg").unwrap_err();
    assert!(
        matches!(&error, RoomAuthError::UnknownEvent { .. }),
        "{error}"
    );
    assert!(generation >= 1);
    // Another room is untouched.
    assert_eq!(other.auth_chain(&db, "$msg").unwrap().len(), 3);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_exhausted_id_space_is_typed() {
    let dir = root("exhausted");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    room.record_event(&db, &event("$first", &[])).unwrap();
    room.closure_for_test()
        .adjacency()
        .events_index()
        .set_counter_for_test(&db, SHORT_ID_MAX)
        .unwrap();
    // One id remains, so an event with two new ids cannot be recorded.
    let error = room
        .record_event(&db, &event("$over", &["$a-new-parent"]))
        .unwrap_err();
    assert!(matches!(error, RoomAuthError::OrdinalExhausted), "{error}");
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_lost_publish_race_is_a_rebuild_conflict() {
    let lost = crate::room_auth::rebuild_error_for_test(StorageError::StaleRead {
        pool: POOL,
        collection_id: [0; 16],
        expected: 1,
        actual: 2,
    });
    assert!(matches!(lost, RoomAuthError::RebuildConflict), "{lost}");
    let other = crate::room_auth::rebuild_error_for_test(StorageError::Corrupt("x".into()));
    assert!(matches!(other, RoomAuthError::Corruption(_)), "{other}");
}

#[test]
fn a_lost_publish_race_leaves_the_winner_readable_and_the_loser_reclaimable() {
    let dir = root("rebuild-race");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    room.record_event(&db, &event("$root", &[])).unwrap();

    let store = room.closure_for_test().store();
    let first = store.begin(&db).unwrap();
    let second = store.begin(&db).unwrap();
    let first_generation = first.generation();
    let second_generation = second.generation();

    // Leave the only event skipped so the manually published generation is
    // structurally valid without duplicating the rebuild walk here.
    let winner = first.publish(&db, 2, &[1]).unwrap();
    assert_eq!(winner.generation, first_generation);
    let error = second.publish(&db, 2, &[1]).unwrap_err();
    assert!(
        matches!(
            &error,
            StorageError::StaleRead {
                pool: actual_pool,
                expected,
                actual,
                ..
            } if *actual_pool == POOL && expected < actual
        ),
        "stale publish must report the winning version transition: {error}"
    );

    let head = room.snapshot(&db).unwrap();
    assert_eq!(head.generation(), Some(first_generation));
    assert_eq!(room.auth_chain(&db, "$root").unwrap(), Vec::<String>::new());
    assert!(
        !crate::closure_store::test::generation_exists(
            room.closure_for_test().store(),
            &db,
            second_generation,
            &[],
        )
        .unwrap(),
        "the losing generation must be discarded on a stale publish"
    );

    assert!(crate::closure_store::test::generation_exists(
        room.closure_for_test().store(),
        &db,
        first_generation,
        &[],
    )
    .unwrap());
    room.retire_old_generations(&db).unwrap();
    assert!(
        crate::closure_store::test::generation_exists(
            room.closure_for_test().store(),
            &db,
            first_generation,
            &[],
        )
        .unwrap(),
        "retirement must preserve the winning generation"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_batch_is_one_transaction_and_a_failed_one_records_nothing() {
    let dir = root("batch");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    room.record_event(&db, &event("$x", &["$p1"])).unwrap();

    // Events in one batch may reference each other.
    let good = [
        event("$a", &[]),
        event("$b", &["$a"]),
        event("$c", &["$b", "$a"]),
    ];
    let recorded = room.record_events(&db, &good).unwrap();
    assert_eq!(recorded.len(), 3);
    assert_eq!(room.auth_chain(&db, "$c").unwrap().len(), 2);

    // A batch with a conflicting event (same id, different auth) records none of
    // it, including the events before it.
    let bad = [
        event("$d", &["$c"]),
        event("$x", &["$p2"]),
        event("$e", &["$d"]),
    ];
    let error = room.record_events(&db, &bad).unwrap_err();
    assert!(
        matches!(&error, RoomAuthError::Storage(StorageError::Collision(_))),
        "{error}"
    );
    assert!(matches!(
        room.auth_edges(&db, "$d").unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));
    assert!(matches!(
        room.auth_edges(&db, "$e").unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));

    // Re-running a batch that already succeeded is a no-op.
    let again = room.record_events(&db, &good).unwrap();
    assert_eq!(again, recorded);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_oversized_batch_is_refused_whole() {
    let dir = root("batch-limit");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    let ids: Vec<String> = (0..=crate::room_auth::MAX_BATCH_EVENTS)
        .map(|index| format!("$e{index}"))
        .collect();
    let events: Vec<NewEvent<'_>> = ids.iter().map(|id| event(id, &[])).collect();
    let error = room.record_events(&db, &events).unwrap_err();
    assert!(
        matches!(&error, RoomAuthError::BatchTooLarge { limit } if *limit == crate::room_auth::MAX_BATCH_EVENTS),
        "{error}"
    );
    assert!(matches!(
        room.auth_edges(&db, "$e0").unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_reports_state_and_gaps_without_calling_them_corruption() {
    let dir = root("verify");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);
    room.record_event(&db, &event("$orphan", &["$ghost"]))
        .unwrap();

    let before = room.verify(&db).unwrap();
    assert!(before.is_consistent(), "{before:?}");
    assert_eq!(before.generation, None);
    assert!(before.needs_rebuild());
    assert_eq!(before.missing_parents, vec!["$ghost"]);

    room.rebuild(&db).unwrap();
    let after = room.verify(&db).unwrap();
    assert!(after.is_consistent(), "{after:?}");
    assert!(after.generation.is_some());
    assert!(!after.needs_rebuild());
    assert_eq!(
        after.closures_checked, 4,
        "the four events with full history"
    );
    assert_eq!(
        after.incomplete_events, 2,
        "$orphan and its never-recorded parent are skipped by design"
    );
    assert_eq!(after.missing_parents, vec!["$ghost"]);

    // An event recorded afterwards needs a rebuild, and is not a problem.
    room.record_event(&db, &event("$later", &["$msg"])).unwrap();
    let stale = room.verify(&db).unwrap();
    assert!(stale.is_consistent(), "{stale:?}");
    assert_eq!(stale.uncovered_events, 1);
    assert!(stale.needs_rebuild());
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_held_snapshot_reports_no_current_generation_after_the_closures_are_purged() {
    let dir = root("stale-none");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    record_diamond(&room, &db);
    let generation = room.rebuild(&db).unwrap().generation;
    let pinned = room.snapshot(&db).unwrap();

    // Purge only the closure generations, so the adjacency still resolves and
    // the read must reach the (now absent) generation.
    let txn = db.begin_transaction();
    room.closure_for_test().stage_purge(&txn).unwrap();
    txn.commit().unwrap();

    let error = pinned.auth_chain(&db, "$msg").unwrap_err();
    assert!(
        matches!(
            &error,
            RoomAuthError::StaleGeneration { pinned, current: None } if *pinned == generation
        ),
        "{error}"
    );
    // The convenience query pins afresh, finds no generation, and walks lazily.
    assert_eq!(room.auth_chain(&db, "$msg").unwrap().len(), 3);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A power cut at any point leaves the room either as it was at the last fsync
/// or with later whole batches, never with a half-recorded batch or a closure
/// generation that disagrees with the adjacency.
///
/// After every batch and rebuild the database directory is imaged twice: with
/// the packs cut to what an fsync had covered (the existing `crash_image`), and
/// additionally with the WAL cut to its durable mark, which loses every frame
/// written since the last fsync.
#[test]
fn a_power_cut_never_leaves_a_half_published_room() {
    use crate::database::tests::crash_image;
    use crate::journal::cut_segment_image_to_durable_mark;

    const BATCHES: usize = 12;
    const PER_BATCH: usize = 8;
    let dir = root("power-cut");
    let image = root("power-cut-image");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);

    let id = |index: usize| format!("$p{index}");
    // The last global event index (+1) known durable, and its generation.
    let mut durable_events = 0usize;
    let mut durable_generation = 0u64;
    // How many cut images lost the newest batch. The cut must actually remove
    // unsynced data, or this test would prove nothing.
    let mut batches_lost_to_the_cut = 0usize;

    for batch in 0..BATCHES {
        let first = batch * PER_BATCH;
        let ids: Vec<String> = (first..first + PER_BATCH).map(id).collect();
        let auth: Vec<Vec<String>> = (first..first + PER_BATCH)
            .map(|index| {
                if index == 0 {
                    Vec::new()
                } else {
                    vec![id(index - 1)]
                }
            })
            .collect();
        let auth_refs: Vec<Vec<&str>> = auth
            .iter()
            .map(|list| list.iter().map(String::as_str).collect())
            .collect();
        let events: Vec<NewEvent<'_>> = ids
            .iter()
            .zip(&auth_refs)
            .map(|(event_id, auth)| NewEvent {
                event_id,
                prev: &[],
                auth,
                relation: None,
            })
            .collect();
        room.record_events(&db, &events).unwrap();
        let generation = room.rebuild(&db).unwrap().generation;
        if batch % 3 == 0 {
            // Make everything so far durable: every pool, which also syncs the WAL.
            for pool in ShardType::ALL {
                db.pool(pool).sync_all().unwrap();
            }
            durable_events = first + PER_BATCH;
            durable_generation = generation;
        }

        for cut_wal in [false, true] {
            crash_image(&db, &dir, &image);
            if cut_wal {
                cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
            }
            let recovered = Database::open(image.clone()).unwrap();
            let after = RoomAuth::new(POOL, ROOM);
            let label = format!("batch {batch}, wal cut: {cut_wal}");

            // Nothing stored may disagree with the adjacency.
            let verify = after.verify(&recovered).unwrap();
            assert!(verify.is_consistent(), "{label}: {verify:?}");

            // Everything made durable survived, with its closures.
            if durable_events > 0 {
                let tip = id(durable_events - 1);
                assert_eq!(
                    after.auth_chain(&recovered, &tip).unwrap().len(),
                    durable_events - 1,
                    "{label}: a durable event lost part of its chain"
                );
                let held = after.snapshot(&recovered).unwrap().generation();
                assert!(
                    held.is_some_and(|held| held >= durable_generation),
                    "{label}: generation {held:?} is older than the durable {durable_generation}"
                );
            }

            // A batch is all or nothing: its first and last event are both
            // present or both absent.
            for later in 0..=batch {
                let start = later * PER_BATCH;
                let present = |index: usize| after.auth_edges(&recovered, &id(index)).is_ok();
                assert_eq!(
                    present(start),
                    present(start + PER_BATCH - 1),
                    "{label}: batch {later} was left half recorded"
                );
            }
            if cut_wal
                && after
                    .auth_edges(&recovered, &id(batch * PER_BATCH))
                    .is_err()
            {
                batches_lost_to_the_cut += 1;
            }
            drop(recovered);
        }
    }
    assert!(
        batches_lost_to_the_cut > 0,
        "cutting the WAL to its durable mark never lost a batch, so the cut is not exercised"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&image);
}

/// The engine's hash index buckets a record by the first 8 bytes of its id and
/// assumes ids are spread. Records whose ids share a long constant prefix (the
/// short-id reverse and edges records, and closure records, were once laid out
/// that way) all land in one bucket and make every insert scan the whole chain:
/// 60,000 events took about a minute. Ingest and rebuild enough events to expose
/// that, and check the longest probe chain any index saw stayed short.
#[test]
fn ingesting_and_rebuilding_many_events_keeps_index_probe_chains_short() {
    const EVENTS: usize = 3000;
    const BATCH: usize = 500;
    let dir = root("probe-chains");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);

    let ids: Vec<String> = (0..EVENTS).map(|index| format!("$e{index:0>40}")).collect();
    let auth: Vec<Vec<&str>> = (0..EVENTS)
        .map(|index| {
            let mut list = Vec::new();
            if index > 0 {
                list.push(ids[index - 1].as_str());
            }
            if index > 1 {
                list.push(ids[index / 2].as_str());
            }
            list
        })
        .collect();
    let events: Vec<NewEvent<'_>> = ids
        .iter()
        .zip(&auth)
        .map(|(event_id, auth)| NewEvent {
            event_id,
            prev: &[],
            auth,
            relation: None,
        })
        .collect();
    for chunk in events.chunks(BATCH) {
        room.record_events(&db, chunk).unwrap();
    }
    room.rebuild(&db).unwrap();

    // Measured for 3000 events: 92 with spread ids, 13,185 when the ids shared a
    // constant prefix. Linear probing on a well-spread, fairly full table reaches
    // double digits, so the bound only has to separate those two by a wide margin.
    let probe = db.pool(POOL).stats().max_index_probe_len;
    assert!(
        probe <= 600,
        "an index probe chain reached {probe} for {EVENTS} events: record ids are clustering"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A full batch of realistic events must commit as one transaction. Matrix
/// events carry up to about ten `auth_events` and a handful of `prev_events` by
/// 44-character ids, and some carry a relation, so this is the fan-out one
/// transaction has to stage at the batch limit.
#[test]
fn a_full_batch_of_realistic_events_fits_one_transaction() {
    let dir = root("full-batch");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    let count = crate::room_auth::MAX_BATCH_EVENTS;

    let ids: Vec<String> = (0..count).map(|index| format!("${index:0>43}")).collect();
    let auth: Vec<Vec<&str>> = (0..count)
        .map(|index| {
            (1..=10)
                .filter_map(|back| index.checked_sub(back))
                .map(|i| ids[i].as_str())
                .collect()
        })
        .collect();
    let prev: Vec<Vec<&str>> = (0..count)
        .map(|index| {
            (1..=3)
                .filter_map(|back| index.checked_sub(back))
                .map(|i| ids[i].as_str())
                .collect()
        })
        .collect();
    let events: Vec<NewEvent<'_>> = (0..count)
        .map(|index| NewEvent {
            event_id: &ids[index],
            prev: &prev[index],
            auth: &auth[index],
            relation: (index % 5 == 4).then(|| RelationRef {
                target: ids[index - 1].as_str(),
                rel_type: "m.annotation",
            }),
        })
        .collect();

    let recorded = room.record_events(&db, &events).unwrap();
    assert_eq!(recorded.len(), count, "the whole batch is one commit");
    let last = &ids[count - 1];
    assert_eq!(room.auth_edges(&db, last).unwrap().len(), 10);
    assert!(room.verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Events with absurd fan-out can exceed what one transaction may stage even
/// under the event-count limit. That must be a typed, recoverable error, and a
/// failed batch records nothing.
#[test]
fn a_batch_over_the_transaction_stage_limit_is_a_typed_error_and_records_nothing() {
    const TEST_TXN_STAGE_LIMIT_BYTES: usize = 2 << 20;
    let dir = root("stage-limit");
    let db = Database::open_with_txn_stage_limit(dir.clone(), TEST_TXN_STAGE_LIMIT_BYTES).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    // Staged size scales with key length, and every new key is stored twice (its
    // forward and reverse records), so long ids cross the test ceiling while a
    // single event still fits beneath it.
    let count = 8usize;
    let padding = "x".repeat(900);
    let refs: Vec<Vec<String>> = (0..count)
        .map(|event| {
            (0..1000)
                .map(|r| format!("${event:0>20}-{r:0>22}{padding}"))
                .collect()
        })
        .collect();
    let auth: Vec<Vec<&str>> = refs
        .iter()
        .map(|list| list.iter().map(String::as_str).collect())
        .collect();
    let ids: Vec<String> = (0..count).map(|index| format!("$ev{index:0>41}")).collect();
    let events: Vec<NewEvent<'_>> = (0..count)
        .map(|index| NewEvent {
            event_id: &ids[index],
            prev: &[],
            auth: &auth[index],
            relation: None,
        })
        .collect();

    let error = room.record_events(&db, &events).unwrap_err();
    assert!(
        matches!(
            &error,
            RoomAuthError::BatchExceedsTransactionLimit {
                staged_bytes,
                limit_bytes,
            } if staged_bytes > limit_bytes && *limit_bytes == TEST_TXN_STAGE_LIMIT_BYTES
        ),
        "{error}"
    );
    assert!(matches!(
        room.auth_edges(&db, &ids[0]).unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));
    // A sub-limit retry remains usable after the rejected batch.
    room.record_events(&db, &events[..1]).unwrap();
    assert_eq!(room.auth_edges(&db, &ids[0]).unwrap().len(), 1000);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
