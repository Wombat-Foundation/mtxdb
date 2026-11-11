//! The generic HAMT store with non-Matrix key and value types.

use std::path::PathBuf;

use crate::database::Database;
use crate::hamt::{build_hamt, diff_hamt_nodes, HamtNode};
use crate::hamt_store::{encode_nodes, node_edges, put_nodes, HamtCollection};
use crate::layout::ShardType;

const POOL: ShardType = ShardType::State;
const COLLECTION: [u8; 16] = [0x4B; 16];
const KEY: &[u8] = b"test-hamt-store";

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-hamt-store-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn entries(n: u64) -> Vec<(u64, Vec<u8>)> {
    (0..n).map(|i| (i, i.to_le_bytes().repeat(3))).collect()
}

fn publish(db: &Database, nodes: &[([u8; 32], Vec<u8>)]) {
    let txn = db.begin_transaction();
    put_nodes(&txn, POOL, COLLECTION, nodes).unwrap();
    txn.commit().unwrap();
}

#[test]
fn round_trips_point_reads_and_scans_for_any_codec_types() {
    let root_dir = test_root("round-trip");
    let db = Database::open(root_dir.clone()).unwrap();
    let input = entries(500);
    let root = build_hamt::<u64, Vec<u8>, _>(KEY, input.clone()).unwrap();
    let nodes = encode_nodes(&root);
    assert!(nodes.len() > 1, "500 entries span more than one node");
    publish(&db, &nodes);

    let store = HamtCollection::new(db.pool(POOL), COLLECTION, KEY);
    for (key, value) in input.iter().step_by(37) {
        assert_eq!(
            store
                .get::<u64, Vec<u8>>(root.structural_hash, key)
                .unwrap(),
            Some(value.clone())
        );
    }
    assert_eq!(
        store
            .get::<u64, Vec<u8>>(root.structural_hash, &9_999)
            .unwrap(),
        None
    );

    let mut scanned = store
        .collect_entries::<u64, Vec<u8>>(root.structural_hash)
        .unwrap();
    scanned.sort();
    assert_eq!(scanned, input);
    drop(db);
    let _ = std::fs::remove_dir_all(root_dir);
}

#[test]
fn encoded_nodes_list_children_first_and_edges_name_stored_ids() {
    let root = build_hamt::<u64, Vec<u8>, _>(KEY, entries(300)).unwrap();
    let nodes = encode_nodes(&root);
    assert_eq!(
        nodes.last().unwrap().0,
        root.structural_hash,
        "root is last"
    );
    let ids: std::collections::HashSet<[u8; 16]> = nodes
        .iter()
        .map(|(hash, _)| crate::state_hamt_node_id(hash))
        .collect();
    for (_, bytes) in &nodes {
        for edge in node_edges::<u64, Vec<u8>>(bytes).unwrap() {
            assert!(ids.contains(&edge), "every edge names a stored node");
        }
    }
    assert_eq!(node_edges::<u64, Vec<u8>>(b"not a node"), None);
}

#[test]
fn a_tree_from_another_structural_key_fails_verification() {
    let root_dir = test_root("wrong-key");
    let db = Database::open(root_dir.clone()).unwrap();
    let root = build_hamt::<u64, Vec<u8>, _>(KEY, entries(200)).unwrap();
    publish(&db, &encode_nodes(&root));

    let wrong = HamtCollection::new(db.pool(POOL), COLLECTION, b"another-key".as_slice());
    assert!(wrong
        .collect_entries::<u64, Vec<u8>>(root.structural_hash)
        .is_err());
    drop(db);
    let _ = std::fs::remove_dir_all(root_dir);
}

#[test]
fn persisted_trees_diff_like_in_memory_ones() {
    let a = build_hamt::<u64, Vec<u8>, _>(KEY, entries(400)).unwrap();
    let mut changed = entries(400);
    changed[7].1 = b"changed".to_vec();
    changed.push((1_000, b"new".to_vec()));
    let b = build_hamt::<u64, Vec<u8>, _>(KEY, changed).unwrap();
    let mut unreachable = |_: &[u8; 32]| -> Result<std::sync::Arc<HamtNode<u64, Vec<u8>>>, ()> {
        unreachable!("trees are fully resolved")
    };
    let (first, second) = diff_hamt_nodes(&a, &b, &mut unreachable).unwrap();
    let mut all: Vec<(u64, Vec<u8>)> = first.into_iter().chain(second).collect();
    all.sort();
    assert_eq!(
        all,
        vec![
            (7, 7_u64.to_le_bytes().repeat(3)),
            (7, b"changed".to_vec()),
            (1_000, b"new".to_vec()),
        ]
    );
}
