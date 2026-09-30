use super::short_id::*;
use crate::database::SharedDatabase;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const SCOPE: [u8; 16] = [0x53; 16];
const AUTH: EdgeFamily = EdgeFamily::immutable(1);
const RELATIONS: EdgeFamily = EdgeFamily::typed_union(3);

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
        .verify(&db, &[AUTH, RELATIONS])
        .unwrap()
        .is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn record_edges_is_atomic_immutable_and_leaf_safe() {
    let root = test_root("edges");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let (create, none) = index().record_edges(&db, b"$create", AUTH, &[]).unwrap();
    assert!(none.is_empty());
    // A leaf has a present, empty edge list, distinct from an unknown id.
    assert_eq!(index().edges(&db, create, AUTH).unwrap(), Some(vec![]));
    assert_eq!(index().edges(&db, 500, AUTH).unwrap(), None);

    let (child, targets) = index()
        .record_edges(&db, b"$child", AUTH, &[b"$create", b"$power", b"$create"])
        .unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(
        index()
            .edges(&db, child, AUTH)
            .unwrap()
            .map(|e| e.iter().map(|x| x.target).collect::<Vec<_>>()),
        Some(targets.clone())
    );
    // Same set again is a no-op; a different set is a collision.
    index()
        .record_edges(&db, b"$child", AUTH, &[b"$power", b"$create"])
        .unwrap();
    let error = index()
        .record_edges(&db, b"$child", AUTH, &[b"$create"])
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");

    let report = index().verify(&db, &[AUTH, RELATIONS]).unwrap();
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
        index().record_edges(&db, b"$x", AUTH, &[b"$y"]).unwrap();
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
        .verify(&db, &[AUTH, RELATIONS])
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
        .record_edges(&db, b"$over", AUTH, &[b"$last", b"$also-new"])
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

fn kinds(db: &SharedDatabase, id: u32) -> Vec<(u32, u16)> {
    index()
        .edges(db, id, RELATIONS)
        .unwrap()
        .unwrap()
        .iter()
        .map(|edge| (edge.target, edge.kind))
        .collect()
}

#[test]
fn union_family_merges_typed_edges_across_writes() {
    let root = test_root("union");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let edge = |target: &'static [u8], kind| EdgeKey { target, kind };
    // A reply and a reaction to the same target differ by kind, not target.
    let first = index()
        .record_event(
            &db,
            b"$reply",
            &[FamilyEdges {
                family: RELATIONS,
                edges: &[edge(b"$orig", 7), edge(b"$orig", 9)],
            }],
        )
        .unwrap();
    assert_eq!(first.edges[0].len(), 2);
    // A later import adds a new edge and repeats an old one: union, not conflict.
    index()
        .record_event(
            &db,
            b"$reply",
            &[FamilyEdges {
                family: RELATIONS,
                edges: &[edge(b"$orig", 9), edge(b"$other", 7)],
            }],
        )
        .unwrap();
    let orig = index().get_or_create(&db, &[b"$orig"]).unwrap()[0];
    let other = index().get_or_create(&db, &[b"$other"]).unwrap()[0];
    let reply = first.id;
    assert_eq!(
        kinds(&db, reply),
        vec![(orig, 7), (orig, 9), (other, 7)],
        "sorted by (target, kind), de-duplicated"
    );
    assert!(index()
        .verify(&db, &[AUTH, RELATIONS])
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
                    family: AUTH,
                    edges: &[EdgeKey::plain(b"$create")],
                },
                FamilyEdges {
                    family: RELATIONS,
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
    // The immutable family rejects a change while the union family still grows.
    let error = index()
        .record_event(
            &db,
            b"$e",
            &[FamilyEdges {
                family: AUTH,
                edges: &[EdgeKey::plain(b"$other-auth")],
            }],
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
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
                family: RELATIONS,
                edges: &[EdgeKey {
                    target: b"$t",
                    kind: 1,
                }],
            }],
        )
        .unwrap();
    // Same family number, read as immutable/untyped: refused, not misparsed.
    let wrong = EdgeFamily::immutable(3);
    let id = index().get_or_create(&db, &[b"$e"]).unwrap()[0];
    assert!(matches!(
        index().edges(&db, id, wrong),
        Err(StorageError::Collision(_))
    ));
    // A kind on an untyped family is a caller error.
    let error = index()
        .record_event(
            &db,
            b"$x",
            &[FamilyEdges {
                family: AUTH,
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

#[test]
fn concurrent_union_writers_lose_no_edges() {
    let root = test_root("union-race");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let targets: Vec<String> = (0..24).map(|i| format!("$t{i}")).collect();
    std::thread::scope(|scope| {
        for thread in 0..6usize {
            let db = &db;
            let targets = &targets;
            scope.spawn(move || {
                for round in 0..8usize {
                    let target = &targets[(thread * 8 + round) % targets.len()];
                    index()
                        .record_event(
                            db,
                            b"$owner",
                            &[FamilyEdges {
                                family: RELATIONS,
                                edges: &[EdgeKey {
                                    target: target.as_bytes(),
                                    kind: 4,
                                }],
                            }],
                        )
                        .unwrap();
                }
            });
        }
    });
    let owner = index().get_or_create(&db, &[b"$owner"]).unwrap()[0];
    let stored = kinds(&db, owner);
    assert_eq!(stored.len(), 24, "every concurrent union write survived");
    assert!(index().verify(&db, &[RELATIONS]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
