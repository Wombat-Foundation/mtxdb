use super::short_id::*;
use crate::database::Database;
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
    let (create, none) = index().record_edges(&db, b"$create", PLAIN, &[]).unwrap();
    assert!(
        none.is_empty(),
        "recording only a leaf edge must collect no touched neighbours"
    );
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
        let db = Database::open(root.clone()).unwrap();
        index().record_edges(&db, b"$x", PLAIN, &[b"$y"]).unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    assert!(error.is_exhausted(), "{error}");
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
    let db = Database::open(root.clone()).unwrap();
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

fn typed_edges(db: &Database, id: u32) -> Vec<(u32, u16)> {
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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

#[test]
fn lookup_never_allocates() {
    let root = test_root("lookup");
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(index().lookup(&db, b"$a").unwrap(), None);
    // A miss left no trace: the first real allocation still gets id 1.
    assert_eq!(index().get_or_create(&db, &[b"$a"]).unwrap(), vec![1]);
    assert_eq!(index().lookup(&db, b"$a").unwrap(), Some(1));
    assert_eq!(index().lookup(&db, b"$b").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_lowered_max_id_refuses_inside_the_transaction() {
    let root = test_root("max-id");
    let db = Database::open(root.clone()).unwrap();
    let small = index().with_max_id(3);
    assert_eq!(
        small.get_or_create(&db, &[b"$a", b"$b", b"$c"]).unwrap(),
        vec![1, 2, 3]
    );
    // A batch that would cross the limit is refused whole: the new key past the
    // limit is not published, and neither is the key before it in the batch.
    let error = small.get_or_create(&db, &[b"$d", b"$e"]).unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(small.lookup(&db, b"$d").unwrap(), None);
    assert_eq!(small.lookup(&db, b"$e").unwrap(), None);
    // Existing ids still resolve and re-asking for them still works.
    assert_eq!(small.get_or_create(&db, &[b"$c"]).unwrap(), vec![3]);
    // The limit cannot be raised above the hard ceiling.
    assert_eq!(
        index()
            .with_max_id(u32::MAX)
            .get_or_create(&db, &[b"$z"])
            .unwrap(),
        vec![4]
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

fn plain_edges<'a>(targets: &[&'a [u8]]) -> Vec<EdgeKey<'a>> {
    targets
        .iter()
        .map(|target| EdgeKey::plain(target))
        .collect()
}

/// Events that reference each other share ids, and the whole batch is one
/// transaction.
#[test]
fn a_batch_allocates_once_and_events_may_reference_each_other() {
    let root = test_root("batch-shared");
    let db = Database::open(root.clone()).unwrap();
    let a_edges = plain_edges(&[b"$b"]);
    let b_edges = plain_edges(&[b"$c"]);
    let a_families = [FamilyEdges {
        family: PLAIN,
        edges: &a_edges,
    }];
    let b_families = [FamilyEdges {
        family: PLAIN,
        edges: &b_edges,
    }];
    let recorded = index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner_payload: None,
                    owner: b"$a",
                    families: &a_families,
                },
                BatchEvent {
                    owner_payload: None,
                    owner: b"$b",
                    families: &b_families,
                },
            ],
        )
        .unwrap();
    // First-appearance order: $a, $b (a's target), $c (b's target).
    assert_eq!(recorded[0].id, 1);
    assert_eq!(recorded[1].id, 2);
    assert_eq!(index().counter(&db).unwrap(), 4);
    let first = index().edges(&db, 1, PLAIN).unwrap().unwrap();
    let second = index().edges(&db, 2, PLAIN).unwrap().unwrap();
    assert_eq!(first.iter().map(|e| e.target).collect::<Vec<_>>(), vec![2]);
    assert_eq!(second.iter().map(|e| e.target).collect::<Vec<_>>(), vec![3]);
    assert!(
        index().edges(&db, 3, PLAIN).unwrap().is_none(),
        "$c was only referenced, so it has no edge record"
    );
    assert!(index().verify(&db, &[PLAIN]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A batch whose later event conflicts with stored data publishes nothing, not
/// even the earlier events that were fine.
#[test]
fn a_failed_batch_publishes_nothing() {
    let root = test_root("batch-atomic");
    let db = Database::open(root.clone()).unwrap();
    index()
        .record_edges(&db, b"$stored", PLAIN, &[b"$p"])
        .unwrap();
    let counter = index().counter(&db).unwrap();

    let ok_edges = plain_edges(&[b"$x"]);
    let conflict_edges = plain_edges(&[b"$different"]);
    let ok = [FamilyEdges {
        family: PLAIN,
        edges: &ok_edges,
    }];
    let conflict = [FamilyEdges {
        family: PLAIN,
        edges: &conflict_edges,
    }];
    let error = index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner_payload: None,
                    owner: b"$fine",
                    families: &ok,
                },
                BatchEvent {
                    owner_payload: None,
                    owner: b"$stored",
                    families: &conflict,
                },
            ],
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
    assert_eq!(
        index().counter(&db).unwrap(),
        counter,
        "no id was allocated"
    );
    assert_eq!(index().lookup(&db, b"$fine").unwrap(), None);
    assert_eq!(index().lookup(&db, b"$x").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The same owner twice in a batch matches or collides; it never stages two
/// writes of one record.
#[test]
fn a_duplicate_owner_in_a_batch_matches_or_collides() {
    let root = test_root("batch-duplicate");
    let db = Database::open(root.clone()).unwrap();
    let edges = plain_edges(&[b"$p"]);
    let other = plain_edges(&[b"$q"]);
    let same = [FamilyEdges {
        family: PLAIN,
        edges: &edges,
    }];
    let different = [FamilyEdges {
        family: PLAIN,
        edges: &other,
    }];
    let twice = index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner_payload: None,
                    owner: b"$e",
                    families: &same,
                },
                BatchEvent {
                    owner_payload: None,
                    owner: b"$e",
                    families: &same,
                },
            ],
        )
        .unwrap();
    assert_eq!(twice[0].id, twice[1].id);

    let error = index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner_payload: None,
                    owner: b"$fresh",
                    families: &same,
                },
                BatchEvent {
                    owner_payload: None,
                    owner: b"$fresh",
                    families: &different,
                },
            ],
        )
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");
    assert_eq!(index().lookup(&db, b"$fresh").unwrap(), None);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Running out of ids part-way through a batch publishes none of it.
#[test]
fn exhaustion_mid_batch_publishes_nothing() {
    let root = test_root("batch-exhausted");
    let db = Database::open(root.clone()).unwrap();
    index().get_or_create(&db, &[b"$seed"]).unwrap();
    index()
        .set_counter_for_test(&db, SHORT_ID_MAX.saturating_sub(1))
        .unwrap();

    let first_edges = plain_edges(&[b"$t1"]);
    let second_edges = plain_edges(&[b"$t2"]);
    let first = [FamilyEdges {
        family: PLAIN,
        edges: &first_edges,
    }];
    let second = [FamilyEdges {
        family: PLAIN,
        edges: &second_edges,
    }];
    // Two ids remain; the batch needs four ($e1, $t1, $e2, $t2).
    let error = index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner_payload: None,
                    owner: b"$e1",
                    families: &first,
                },
                BatchEvent {
                    owner_payload: None,
                    owner: b"$e2",
                    families: &second,
                },
            ],
        )
        .unwrap_err();
    assert!(error.is_exhausted(), "{error}");
    assert_eq!(
        index().lookup(&db, b"$e1").unwrap(),
        None,
        "nothing published"
    );
    assert_eq!(index().lookup(&db, b"$t1").unwrap(), None);
    assert!(index().record_events(&db, &[]).unwrap().is_empty());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The engine's hash index buckets a record by id bytes 0..8 and tags it with
/// bytes 8..12, assuming ids are spread. Every record id this layer builds must
/// vary in both fields across sequential short ids; a constant prefix there put
/// every reverse and edges record into one bucket.
#[test]
fn record_ids_vary_in_the_bytes_the_index_uses() {
    const IDS: u32 = 10_000;
    let mut buckets = std::collections::HashSet::new();
    let mut tags = std::collections::HashSet::new();
    for short_id in 1..=IDS {
        for id in [
            reverse_record_id_for_test(short_id),
            edges_record_id_for_test(short_id, 1),
            edges_record_id_for_test(short_id, 2),
        ] {
            buckets.insert(<[u8; 8]>::try_from(&id[..8]).unwrap());
            tags.insert(<[u8; 4]>::try_from(&id[8..12]).unwrap());
        }
    }
    let total = usize::try_from(IDS).unwrap() * 3;
    assert!(
        buckets.len() >= total - 3,
        "the first 8 bytes (the index bucket) collide: {} distinct of {total}",
        buckets.len()
    );
    assert!(
        tags.len() >= total - 3,
        "bytes 8..12 (the index tag) collide: {} distinct of {total}",
        tags.len()
    );
}

fn owned<'a>(owner: &'a [u8], payload: &'a [u8]) -> BatchEvent<'a> {
    BatchEvent {
        owner,
        families: &[],
        owner_payload: Some(payload),
    }
}

fn log(db: &Database) -> Vec<(u32, u32, Vec<u8>)> {
    let counters = index().counters(db).unwrap();
    index()
        .owner_log(
            db,
            counters.log_epoch,
            counters.log_epoch_start_seq,
            counters.next_owner_seq,
        )
        .unwrap()
        .into_iter()
        .map(|e| (e.seq, e.short_id, e.payload))
        .collect()
}

#[test]
fn recording_an_owner_twice_logs_it_once() {
    let root = test_root("owner-twice");
    let db = Database::open(root.clone()).unwrap();
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    let after_first = index().counters(&db).unwrap();
    // A replay, a re-fetch, and the same owner twice in one batch.
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    index()
        .record_events(&db, &[owned(b"$b", b"pb"), owned(b"$b", b"pb")])
        .unwrap();
    index().record_events(&db, &[owned(b"$b", b"pb")]).unwrap();
    assert_eq!(index().counters(&db).unwrap().next_owner_seq, 3);
    assert_eq!(
        log(&db),
        vec![(1, 1, b"pa".to_vec()), (2, 2, b"pb".to_vec())]
    );
    assert_eq!(after_first.next_owner_seq, 2);
    let report = index().verify(&db, &[PLAIN]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    assert_eq!(report.owners_checked, 2);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_edge_target_that_becomes_an_owner_is_logged_once() {
    let root = test_root("owner-after-target");
    let db = Database::open(root.clone()).unwrap();
    // `$parent` is allocated an id only because `$child` names it.
    let edges = [EdgeKey::plain(b"$parent")];
    let families = [FamilyEdges {
        family: PLAIN,
        edges: &edges,
    }];
    index()
        .record_events(
            &db,
            &[BatchEvent {
                owner: b"$child",
                families: &families,
                owner_payload: Some(b"child"),
            }],
        )
        .unwrap();
    assert_eq!(index().is_owner(&db, b"$parent").unwrap(), Some(false));
    assert_eq!(log(&db).len(), 1, "a target is not an owner");

    // Now the parent arrives: its bit flips and it is logged, once.
    index()
        .record_events(&db, &[owned(b"$parent", b"parent")])
        .unwrap();
    index()
        .record_events(&db, &[owned(b"$parent", b"parent")])
        .unwrap();
    assert_eq!(index().is_owner(&db, b"$parent").unwrap(), Some(true));
    assert_eq!(log(&db).len(), 2);
    // An owner that is also a target in the same batch flips once.
    index()
        .record_events(
            &db,
            &[
                BatchEvent {
                    owner: b"$grand",
                    families: &[],
                    owner_payload: None,
                },
                owned(b"$child2", b"c2"),
                BatchEvent {
                    owner: b"$x",
                    families: &[FamilyEdges {
                        family: PLAIN,
                        edges: &[EdgeKey::plain(b"$child2")],
                    }],
                    owner_payload: Some(b"x"),
                },
            ],
        )
        .unwrap();
    assert_eq!(index().is_owner(&db, b"$grand").unwrap(), Some(false));
    assert_eq!(log(&db).len(), 4);
    assert!(index().verify(&db, &[PLAIN]).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn conflicting_owner_payloads_in_one_batch_collide() {
    let root = test_root("owner-conflict");
    let db = Database::open(root.clone()).unwrap();
    let error = index()
        .record_events(&db, &[owned(b"$a", b"one"), owned(b"$a", b"two")])
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error:?}");
    assert_eq!(index().counters(&db).unwrap().next_owner_seq, 1);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Several writers record the same owners at once: every owner is logged
/// exactly once and the log stays dense.
#[test]
fn racing_writers_log_each_owner_once() {
    let root = test_root("owner-race");
    let db = std::sync::Arc::new(Database::open(root.clone()).unwrap());
    let keys: Vec<Vec<u8>> = (0..24).map(|i| format!("$e{i}").into_bytes()).collect();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let db = db.clone();
            let keys = keys.clone();
            std::thread::spawn(move || {
                for key in &keys {
                    index().record_events(&db, &[owned(key, key)]).unwrap();
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let entries = log(&db);
    assert_eq!(entries.len(), keys.len());
    let mut payloads: Vec<_> = entries.iter().map(|e| e.2.clone()).collect();
    payloads.sort();
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(payloads, expected);
    assert!(entries
        .iter()
        .enumerate()
        .all(|(i, e)| e.0 as usize == i + 1));
    let report = index().verify(&db, &[PLAIN]).unwrap();
    assert!(report.is_consistent(), "{:?}", report.problems);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn purge_removes_the_owner_log_too() {
    let root = test_root("owner-purge");
    let db = Database::open(root.clone()).unwrap();
    index().record_events(&db, &[owned(b"$a", b"pa")]).unwrap();
    index().purge(&db).unwrap();
    assert_eq!(index().counters(&db).unwrap(), ScopeCounters::default());
    assert_eq!(index().is_owner(&db, b"$a").unwrap(), None);
    // The generation collection is gone: a fresh owner starts a fresh log.
    index().record_events(&db, &[owned(b"$b", b"pb")]).unwrap();
    assert_eq!(log(&db), vec![(1, 1, b"pb".to_vec())]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_old_format_store_is_reported_as_unsupported() {
    let root = test_root("old-format");
    let db = Database::open(root.clone()).unwrap();
    index().get_or_create(&db, &[b"$a"]).unwrap();
    // Rewrite the counter as a v3 record (3, not the current version).
    let txn = db.begin_transaction();
    txn.put(
        POOL,
        SCOPE,
        *b"MTXD-SID-CNTR-v1",
        &crate::storage::NodeData::new(bytes::Bytes::from(vec![
            b'S', b'I', b'D', b'C', 3, 0, 0, 0, 2,
        ])),
    )
    .unwrap();
    txn.commit().unwrap();
    let error = index().counters(&db).unwrap_err();
    assert!(
        error.to_string().contains("unsupported short-id format v3"),
        "{error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verify_checks_each_log_entry_against_its_forward_record() {
    let root = test_root("verify-log");
    let db = Database::open(root.clone()).unwrap();
    index()
        .record_events(&db, &[owned(b"$a", b"pa"), owned(b"$b", b"pb")])
        .unwrap();
    assert!(index().verify(&db, &[PLAIN]).unwrap().is_consistent());
    // Point entry 2 at id 1: the bit count still matches, the entry is wrong.
    let counters = index().counters(&db).unwrap();
    let txn = db.begin_transaction();
    let mut bad = b"OWNL\x04".to_vec();
    bad.extend_from_slice(&1_u32.to_be_bytes());
    bad.extend_from_slice(b"pb");
    txn.put(
        POOL,
        index().owner_log_collection(counters.log_epoch),
        super::short_id::owner_log_record_id_for_test(2),
        &crate::storage::NodeData::new(bytes::Bytes::from(bad)),
    )
    .unwrap();
    txn.commit().unwrap();
    let report = index().verify(&db, &[PLAIN]).unwrap();
    assert!(
        report.problems.iter().any(|p| p.contains("repeats id 1")),
        "{:?}",
        report.problems
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
