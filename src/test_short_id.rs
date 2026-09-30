use super::short_id::*;
use crate::database::SharedDatabase;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const SCOPE: [u8; 16] = [0x53; 16];
const PLAIN: EdgeFamily = EdgeFamily::plain(1);
const TYPED: EdgeFamily = EdgeFamily::typed(3);

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-short-id-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn index() -> ShortIdIndex {
    ShortIdIndex::new(POOL, SCOPE)
}

#[test]
fn allocation_is_dense_stable_and_idempotent() {
    let root = test_root("alloc");
    let db = SharedDatabase::open(root.clone()).unwrap();
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
    assert!(index()
        .verify(&db, &[PLAIN, TYPED])
        .unwrap()
        .is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn record_edges_is_atomic_immutable_and_leaf_safe() {
    let root = test_root("edges");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let (create, none) = index().record_edges(&db, b"$create", PLAIN, &[]).unwrap();
    assert!(none.is_empty());
    // A leaf has a present, empty edge list, distinct from an unknown id.
    assert_eq!(index().edges(&db, create, PLAIN).unwrap(), Some(vec![]));
    assert_eq!(index().edges(&db, 500, PLAIN).unwrap(), None);

    let (child, targets) = index()
        .record_edges(&db, b"$child", PLAIN, &[b"$create", b"$power", b"$create"])
        .unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(
        index()
            .edges(&db, child, PLAIN)
            .unwrap()
            .map(|e| e.iter().map(|x| x.target).collect::<Vec<_>>()),
        Some(targets.clone())
    );
    // Same set again is a no-op; a different set is a collision.
    index()
        .record_edges(&db, b"$child", PLAIN, &[b"$power", b"$create"])
        .unwrap();
    let error = index()
        .record_edges(&db, b"$child", PLAIN, &[b"$create"])
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");

    let report = index().verify(&db, &[PLAIN, TYPED]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.edge_lists_checked, 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reopen_preserves_ids_and_purge_resets_the_scope() {
    let root = test_root("reopen");
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        index().record_edges(&db, b"$x", PLAIN, &[b"$y"]).unwrap();
    }
    let db = SharedDatabase::open(root.clone()).unwrap();
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
    let db = SharedDatabase::open(root.clone()).unwrap();
    let results: Vec<Vec<(String, u32)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
            .map(|thread| {
                let db = &db;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for round in 0..20 {
                        // Overlapping keys across threads.
                        let key = format!("$k{}", (round + thread) % 25);
                        let id = index().get_or_create(db, &[key.as_bytes()]).unwrap()[0];
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
    assert!(index()
        .verify(&db, &[PLAIN, TYPED])
        .unwrap()
        .is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn exhaustion_is_a_hard_error_and_publishes_nothing() {
    let root = test_root("overflow");
    let db = SharedDatabase::open(root.clone()).unwrap();
    index().set_counter_for_test(&db, SHORT_ID_MAX).unwrap();
    // The last valid id is allocatable...
    assert_eq!(
        index().get_or_create(&db, &[b"$last"]).unwrap(),
        vec![SHORT_ID_MAX]
    );
    // ...the next allocation fails, including mid-batch, and aborts whole.
    let error = index()
        .record_edges(&db, b"$over", PLAIN, &[b"$last", b"$also-new"])
        .unwrap_err();
    assert!(matches!(error, StorageError::Internal(_)), "{error}");
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
    let db = SharedDatabase::open(root.clone()).unwrap();
    index().get_or_create(&db, &[b"$a"]).unwrap();
    {
        use crate::storage::NodeData;
        let txn = db.begin_transaction();
        txn.put(
            ShardType::Edges,
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

fn typed_edges(db: &SharedDatabase, id: u32) -> Vec<(u32, u16)> {
    index()
        .edges(db, id, TYPED)
        .unwrap()
        .unwrap()
        .iter()
        .map(|edge| (edge.target, edge.kind))
        .collect()
}

#[test]
fn typed_family_keeps_kinds_sorted_and_rejects_changes() {
    let root = test_root("typed");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let edge = |target: &'static [u8], kind| EdgeKey { target, kind };
    // Two edges to one target that differ only by kind are both kept.
    let first = index()
        .record_event(
            &db,
            b"$src",
            &[FamilyEdges {
                family: TYPED,
                edges: &[edge(b"$orig", 9), edge(b"$orig", 7), edge(b"$orig", 9)],
            }],
        )
        .unwrap();
    let orig = index().get_or_create(&db, &[b"$orig"]).unwrap()[0];
    assert_eq!(typed_edges(&db, first.id), vec![(orig, 7), (orig, 9)]);
    // The same list again is a no-op (idempotent re-import).
    index()
        .record_event(
            &db,
            b"$src",
            &[FamilyEdges {
                family: TYPED,
                edges: &[edge(b"$orig", 7), edge(b"$orig", 9)],
            }],
        )
        .unwrap();
    // A different list, including a changed kind, is a collision, never a merge.
    for changed in [
        vec![edge(b"$orig", 7)],
        vec![edge(b"$orig", 7), edge(b"$orig", 9), edge(b"$other", 7)],
        vec![edge(b"$orig", 7), edge(b"$orig", 10)],
    ] {
        let error = index()
            .record_event(
                &db,
                b"$src",
                &[FamilyEdges {
                    family: TYPED,
                    edges: &changed,
                }],
            )
            .unwrap_err();
        assert!(matches!(error, StorageError::Collision(_)), "{error}");
    }
    assert_eq!(typed_edges(&db, first.id), vec![(orig, 7), (orig, 9)]);
    assert!(index()
        .verify(&db, &[PLAIN, TYPED])
        .unwrap()
        .is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn families_are_independent_per_owner_in_one_transaction() {
    let root = test_root("families");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let recorded = index()
        .record_event(
            &db,
            b"$e",
            &[
                FamilyEdges {
                    family: PLAIN,
                    edges: &[EdgeKey::plain(b"$create")],
                },
                FamilyEdges {
                    family: TYPED,
                    edges: &[EdgeKey {
                        target: b"$create",
                        kind: 2,
                    }],
                },
            ],
        )
        .unwrap();
    assert_eq!(recorded.edges.len(), 2);
    assert_eq!(recorded.edges[0][0].kind, 0, "untyped family has kind 0");
    assert_eq!(recorded.edges[1][0].kind, 2);
    // A conflicting list in one family fails the whole transaction: the other
    // family's (valid) write is not applied either.
    let other_owner = index().get_or_create(&db, &[b"$f"]).unwrap()[0];
    let error = index()
        .record_event(
            &db,
            b"$e",
            &[
                FamilyEdges {
                    family: PLAIN,
                    edges: &[EdgeKey::plain(b"$different")],
                },
                FamilyEdges {
                    family: EdgeFamily::plain(9),
                    edges: &[EdgeKey::plain(b"$new-family")],
                },
            ],
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
    let e = index().get_or_create(&db, &[b"$e"]).unwrap()[0];
    assert_eq!(index().edges(&db, e, EdgeFamily::plain(9)).unwrap(), None);
    assert_eq!(index().edges(&db, other_owner, PLAIN).unwrap(), None);
    assert_eq!(index().resolve(&db, &[e + 100]).unwrap(), vec![None]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_family_cannot_be_reinterpreted_or_given_kinds_when_untyped() {
    let root = test_root("flags");
    let db = SharedDatabase::open(root.clone()).unwrap();
    index()
        .record_event(
            &db,
            b"$e",
            &[FamilyEdges {
                family: TYPED,
                edges: &[EdgeKey {
                    target: b"$t",
                    kind: 1,
                }],
            }],
        )
        .unwrap();
    // Same family number, read as untyped: refused, not misparsed.
    let id = index().get_or_create(&db, &[b"$e"]).unwrap()[0];
    assert!(matches!(
        index().edges(&db, id, EdgeFamily::plain(3)),
        Err(StorageError::Collision(_))
    ));
    // A kind on an untyped family is a caller error.
    let error = index()
        .record_event(
            &db,
            b"$x",
            &[FamilyEdges {
                family: PLAIN,
                edges: &[EdgeKey {
                    target: b"$t",
                    kind: 5,
                }],
            }],
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Internal(_)), "{error}");
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Identical concurrent writers all succeed (idempotent); a writer with a
/// different list loses with a collision and never overwrites the winner.
#[test]
fn concurrent_writers_are_idempotent_and_never_overwrite() {
    let root = test_root("race-idempotent");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let outcomes: Vec<Result<(), StorageError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8usize)
            .map(|thread| {
                let db = &db;
                scope.spawn(move || {
                    // Even threads agree on one list; odd threads propose another.
                    let target: &[u8] = if thread % 2 == 0 { b"$a" } else { b"$b" };
                    index()
                        .record_event(
                            db,
                            b"$owner",
                            &[FamilyEdges {
                                family: TYPED,
                                edges: &[EdgeKey { target, kind: 1 }],
                            }],
                        )
                        .map(|_| ())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wins = outcomes.iter().filter(|o| o.is_ok()).count();
    assert!(
        (4..=8).contains(&wins),
        "one side's list is stored, all its writers agree"
    );
    assert!(outcomes
        .iter()
        .filter_map(|o| o.as_ref().err())
        .all(|e| matches!(e, StorageError::Collision(_))));
    let owner = index().get_or_create(&db, &[b"$owner"]).unwrap()[0];
    assert_eq!(
        typed_edges(&db, owner).len(),
        1,
        "exactly one list survived"
    );
    assert!(index().verify(&db, &[TYPED]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
