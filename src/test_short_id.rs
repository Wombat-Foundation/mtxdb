use super::short_id::*;
use crate::database::SharedDatabase;
use crate::layout::ShardType;
use crate::storage::StorageError;
use std::path::PathBuf;

const POOL: ShardType = ShardType::Edges;
const SCOPE: [u8; 16] = [0x53; 16];

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
    assert!(index().verify(&db).unwrap().is_consistent());
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn record_edges_is_atomic_immutable_and_leaf_safe() {
    let root = test_root("edges");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let (create, none) = index().record_edges(&db, b"$create", &[]).unwrap();
    assert!(none.is_empty());
    // A leaf has a present, empty edge list, distinct from an unknown id.
    assert_eq!(index().edges(&db, create).unwrap(), Some(vec![]));
    assert_eq!(index().edges(&db, 500).unwrap(), None);

    let (child, targets) = index()
        .record_edges(&db, b"$child", &[b"$create", b"$power", b"$create"])
        .unwrap();
    assert_eq!(targets.len(), 2);
    assert_eq!(index().edges(&db, child).unwrap(), Some(targets.clone()));
    // Same set again is a no-op; a different set is a collision.
    index()
        .record_edges(&db, b"$child", &[b"$power", b"$create"])
        .unwrap();
    let error = index()
        .record_edges(&db, b"$child", &[b"$create"])
        .unwrap_err();
    assert!(matches!(error, StorageError::Collision(_)), "{error}");

    let report = index().verify(&db).unwrap();
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
        index().record_edges(&db, b"$x", &[b"$y"]).unwrap();
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
    assert!(index().verify(&db).unwrap().is_consistent());
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
        .record_edges(&db, b"$over", &[b"$last", b"$also-new"])
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
