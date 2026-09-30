#![allow(clippy::tests_outside_test_module)]

//! Repack check for the persisted short-ID / auth-closure stack.
//!
//! A collection rewrite must preserve, not merely keep a similar physical
//! record count:
//!
//! - the short-ID mapping of every event id;
//! - every recorded `auth` adjacency (including present-but-empty leaves);
//! - the published closure generation's head (`generation`, `source_next`,
//!   `count`) and every closure bitmap;
//! - the same view again after a close/reopen, when the rewritten checkpoint
//!   and record offsets are all that remain.

use std::path::PathBuf;

use mtxdb::auth_closure::{AuthClosure, RebuildOutcome};
use mtxdb::matrix_adjacency::MatrixAdjacency;
use mtxdb::{BitmapSet, ShardType, SharedDatabase};

const POOL: ShardType = ShardType::Edges;
const ROOM: &str = "!repack:example.org";

/// `(event id, auth events)` in dependency order: a referenced target is
/// recorded before its referrer, so every walk is complete.
const EVENTS: &[(&str, &[&str])] = &[
    ("$create", &[]),
    ("$member", &["$create"]),
    ("$power", &["$create"]),
    ("$leaf", &[]),
    ("$child", &["$leaf"]),
    ("$message", &["$create", "$member", "$power"]),
];

fn root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-repack-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// Everything the repack must leave untouched, read through the public API.
#[derive(Debug, PartialEq)]
struct Observed {
    ids: Vec<(String, Option<u32>)>,
    auth: Vec<(String, Option<Vec<String>>)>,
    head: Option<(u64, u32, u32)>,
    closures: Vec<(String, Option<BitmapSet>)>,
    per_collection: Vec<([u8; 16], usize)>,
}

fn observe(db: &SharedDatabase, adjacency: &MatrixAdjacency, closure: &AuthClosure) -> Observed {
    let ids = EVENTS
        .iter()
        .map(|&(id, _)| (id.to_owned(), adjacency.short_id(db, id).unwrap()))
        .collect();
    let auth = EVENTS
        .iter()
        .map(|&(id, _)| (id.to_owned(), adjacency.auth_of(db, id).unwrap()))
        .collect();
    let head = closure
        .head(db)
        .unwrap()
        .map(|head| (head.generation, head.source_next, head.count));
    let closures = EVENTS
        .iter()
        .map(|&(id, _)| (id.to_owned(), closure.get(db, id).unwrap()))
        .collect();
    let mut per_collection: Vec<([u8; 16], usize)> = db
        .edges()
        .collection_summaries()
        .into_iter()
        .map(|(id, entries, _memory, _capacity)| (id, entries))
        .collect();
    per_collection.sort_unstable();
    Observed {
        ids,
        auth,
        head,
        closures,
        per_collection,
    }
}

/// Rewrite every collection in the edges pool in place. No live roots are
/// configured, so repack must GC nothing and keep every live record.
fn repack_pool(db: &SharedDatabase) -> usize {
    let storage = db.edges();
    let collections = storage.collection_ids();
    assert!(
        collections.len() >= 2,
        "expected short-id and closure collections, found {}",
        collections.len()
    );
    let mut kept_total: usize = 0;
    for collection in &collections {
        let (kept, dropped) = storage
            .repack_collection_reachable(collection, |_, _| Vec::new())
            .unwrap();
        assert_eq!(dropped, 0, "repack with no live roots must GC nothing");
        kept_total = kept_total.saturating_add(kept);
    }
    kept_total
}

#[test]
fn repack_preserves_short_ids_adjacency_and_published_closures() {
    let dir = root("preserve");
    let db = SharedDatabase::open(dir.clone()).unwrap();
    let adjacency = MatrixAdjacency::new(POOL, ROOM);
    for &(id, auth) in EVENTS {
        adjacency.record_event(&db, id, &[], auth, None).unwrap();
    }

    let closure = AuthClosure::new(POOL, ROOM);
    match closure.rebuild(&db).unwrap() {
        RebuildOutcome::Published(report) => {
            assert_eq!(
                report.count as usize,
                EVENTS.len(),
                "one closure record per recorded event"
            );
        }
        RebuildOutcome::Incomplete { missing } => {
            panic!("every event is recorded, but rebuild was incomplete: {missing:?}");
        }
    }
    assert!(
        closure.verify(&db).unwrap().is_consistent(),
        "closures verify against direct auth edges before repack"
    );

    let before = observe(&db, &adjacency, &closure);
    assert!(
        before.ids.iter().all(|(_, id)| id.is_some()),
        "every event has a short id before repack"
    );
    assert!(
        before.auth.iter().all(|(_, auth)| auth.is_some()),
        "every recorded event has an auth record before repack"
    );
    assert!(
        before.closures.iter().all(|(_, set)| set.is_some()),
        "every event has a persisted closure before repack"
    );
    assert!(
        before
            .per_collection
            .iter()
            .any(|(_, entries)| *entries == EVENTS.len()),
        "the closure generation collection (one record per covered id) is present to be repacked"
    );

    let kept = repack_pool(&db);
    assert!(kept > 0, "repack should have copied records");

    assert_eq!(
        observe(&db, &adjacency, &closure),
        before,
        "live reads unchanged by repack"
    );
    assert!(
        closure.verify(&db).unwrap().is_consistent(),
        "closures still verify after repack"
    );

    drop(db);
    let db = SharedDatabase::open(dir.clone()).unwrap();
    assert_eq!(
        observe(&db, &adjacency, &closure),
        before,
        "reopen after repack sees the same ids, adjacency, head and closures"
    );
    assert!(
        closure.verify(&db).unwrap().is_consistent(),
        "closures verify after reopen"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
