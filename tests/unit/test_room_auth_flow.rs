#![allow(clippy::tests_outside_test_module)]

//! One caller-level flow over the public façade only: import, auth query,
//! rebuild, reopen, repack, verify, with a failed rebuild and a stale snapshot
//! along the way. Nothing here touches the short-id index, closure store, bitmap
//! sets or logical heads directly.

use std::path::PathBuf;

use mtxdb::room_auth::{NewEvent, RoomAuth, RoomAuthError};
use mtxdb::{Database, ShardType};

const POOL: ShardType = ShardType::Edges;
const ROOM: &str = "!flow:example.org";
const CYCLIC_ROOM: &str = "!cyclic:example.org";
const EVENTS: usize = 40;
const LAST: usize = EVENTS.saturating_sub(1);

fn root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-room-flow-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn id(index: usize) -> String {
    format!("$e{index}")
}

/// `$e0` is the create event. Every later event authorizes against the create
/// event and its predecessor, so chains overlap and the graph has diamonds.
fn auth_of(index: usize) -> Vec<String> {
    match index {
        0 => Vec::new(),
        1 => vec![id(0)],
        _ => vec![id(0), id(index.saturating_sub(1))],
    }
}

/// The ancestors of event `index`, which is every earlier event.
fn expected_chain(index: usize) -> Vec<String> {
    (0..index).map(id).collect()
}

fn import(room: &RoomAuth, db: &Database) {
    for index in 0..EVENTS {
        let event_id = id(index);
        let auth = auth_of(index);
        let auth_refs: Vec<&str> = auth.iter().map(String::as_str).collect();
        room.record_event(
            db,
            &NewEvent {
                event_id: &event_id,
                prev: &[],
                auth: &auth_refs,
                relation: None,
            },
        )
        .unwrap();
    }
}

fn sorted(mut ids: Vec<String>) -> Vec<String> {
    ids.sort_by_key(|event_id| event_id[2..].parse::<usize>().unwrap());
    ids
}

fn assert_chains(room: &RoomAuth, db: &Database, when: &str) {
    for index in [0, 1, 2, 17, LAST] {
        let chain = sorted(room.auth_chain(db, &id(index)).unwrap());
        assert_eq!(
            chain,
            expected_chain(index),
            "chain of {} {when}",
            id(index)
        );
    }
    assert!(
        room.is_in_auth_chain(db, &id(0), &id(LAST)).unwrap(),
        "the create event is an ancestor of the last event {when}"
    );
    assert!(
        !room.is_in_auth_chain(db, &id(LAST), &id(0)).unwrap(),
        "a descendant is not an ancestor {when}"
    );
}

/// Rewrite every collection of the edges pool in place, keeping everything.
fn repack(db: &Database) {
    let storage = db.edges();
    for collection in storage.collection_ids() {
        let (_kept, dropped) = storage
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        assert_eq!(dropped, 0, "repack with no live roots must drop nothing");
    }
}

#[test]
fn import_query_rebuild_reopen_repack_verify() {
    let dir = root("flow");
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);

    // Import, then query before any closure exists: correct through the lazy walk.
    import(&room, &db);
    assert!(room.verify(&db).unwrap().needs_rebuild());
    assert_chains(&room, &db, "before any rebuild");

    // Rebuild: covers everything, complete, and the answers do not change.
    let report = room.rebuild(&db).unwrap();
    assert!(report.is_complete());
    assert_eq!(report.count as usize, EVENTS);
    assert_chains(&room, &db, "after the first rebuild");
    let verify = room.verify(&db).unwrap();
    assert!(verify.is_consistent(), "{verify:?}");
    assert!(!verify.needs_rebuild());

    // Pin the first generation, then publish two more and retire: the pinned
    // handle is stale, a fresh query is not.
    let pinned = room.snapshot(&db).unwrap();
    assert_eq!(pinned.generation(), Some(report.generation));
    room.rebuild(&db).unwrap();
    let latest = room.rebuild(&db).unwrap();
    assert_eq!(room.retire_old_generations(&db).unwrap(), 1);
    let stale = pinned.auth_chain(&db, &id(LAST)).unwrap_err();
    assert!(
        matches!(
            &stale,
            RoomAuthError::StaleGeneration { pinned, current }
                if *pinned == report.generation && *current == Some(latest.generation)
        ),
        "{stale}"
    );
    assert_chains(&room, &db, "after retiring the pinned generation");

    // A rebuild that fails leaves the room exactly as it was. The cyclic room
    // is a separate room, so the first room is the control for "untouched".
    let cyclic = RoomAuth::new(POOL, CYCLIC_ROOM);
    cyclic
        .record_event(
            &db,
            &NewEvent {
                event_id: "$x",
                prev: &[],
                auth: &["$y"],
                relation: None,
            },
        )
        .unwrap();
    cyclic
        .record_event(
            &db,
            &NewEvent {
                event_id: "$y",
                prev: &[],
                auth: &["$x"],
                relation: None,
            },
        )
        .unwrap();
    let failed = cyclic.rebuild(&db).unwrap_err();
    assert!(
        matches!(&failed, RoomAuthError::Corruption(message) if message.contains("cycle")),
        "{failed}"
    );
    assert_eq!(
        cyclic.snapshot(&db).unwrap().generation(),
        None,
        "a failed rebuild publishes nothing"
    );
    assert_eq!(
        room.snapshot(&db).unwrap().generation(),
        Some(latest.generation),
        "the other room's head is untouched"
    );

    // Reopen: the same generation resolves, answers and verify are unchanged.
    drop(db);
    let db = Database::open(dir.clone()).unwrap();
    let room = RoomAuth::new(POOL, ROOM);
    assert_eq!(
        room.snapshot(&db).unwrap().generation(),
        Some(latest.generation)
    );
    assert_chains(&room, &db, "after reopen");
    assert!(room.verify(&db).unwrap().is_consistent());

    // Repack every collection, then the same again, and once more after a
    // second reopen, when only the rewritten checkpoint and offsets remain.
    repack(&db);
    assert_chains(&room, &db, "after repack");
    let verify = room.verify(&db).unwrap();
    assert!(verify.is_consistent(), "{verify:?}");

    drop(db);
    let db = Database::open(dir.clone()).unwrap();
    assert_eq!(
        room.snapshot(&db).unwrap().generation(),
        Some(latest.generation)
    );
    assert_chains(&room, &db, "after repack and reopen");
    assert!(room.verify(&db).unwrap().is_consistent());

    // Purge removes the room and leaves the other one.
    room.purge(&db).unwrap();
    assert!(matches!(
        room.auth_edges(&db, &id(0)).unwrap_err(),
        RoomAuthError::UnknownEvent { .. }
    ));
    assert!(cyclic.auth_edges(&db, "$x").is_ok());

    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
