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
    assert!(room.auth_chain(&db, "$create").unwrap().is_empty());

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
fn a_failed_batch_reports_what_was_recorded_and_is_safe_to_retry() {
    let dir = root("batch");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    room.record_event(&db, &event("$x", &["$p1"])).unwrap();
    let events = [
        event("$a", &[]),
        event("$b", &["$a"]),
        // Same id, different auth: refused, because an event never changes.
        event("$x", &["$p2"]),
        event("$c", &["$b"]),
    ];
    let failure = room.record_events(&db, &events).unwrap_err();
    assert_eq!(failure.recorded.len(), 2, "$a and $b committed before $x");
    assert!(failure.to_string().contains("2 recorded"));

    // The committed events are there, and re-running the prefix is a no-op.
    assert_eq!(room.auth_edges(&db, "$b").unwrap(), vec!["$a"]);
    let again = room.record_events(&db, &events[..2]).unwrap();
    assert_eq!(again, failure.recorded);
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
