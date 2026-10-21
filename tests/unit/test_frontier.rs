#![allow(clippy::tests_outside_test_module)]

use mtxdb::frontier::*;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};

#[test]
fn test_frontier_batch() {
    let hashes = vec![[1u8; 16], [2u8; 16], [3u8; 16]];
    let batch = FrontierBatch::new(hashes.clone());
    assert_eq!(batch.len(), 3);
    assert!(!batch.is_empty());
    assert_eq!(batch.hashes, hashes);
}

#[test]
fn test_bfs_layer_dedup() {
    let mut layer = BfsLayer::new();
    let h1 = [1u8; 16];
    let h2 = [2u8; 16];
    let target = [0xFFu8; 16];

    layer.push(h1, target);
    layer.push(h1, [0xEE; 16]); // same hash, different dependent
    layer.push(h2, target);

    assert_eq!(layer.len(), 2);
    assert_eq!(layer.dependents()[0].len(), 2); // h1 has 2 dependents
    assert_eq!(layer.dependents()[1].len(), 1); // h2 has 1 dependent
}

#[test]
fn test_bfs_layer_hashes_and_is_empty() {
    let layer = BfsLayer::new();
    assert!(layer.is_empty());
    let empty: &[NodeId] = &[];
    assert_eq!(layer.hashes(), empty);

    let mut layer = BfsLayer::new();
    layer.push([1u8; 16], [0xFF; 16]);
    layer.push([2u8; 16], [0xFF; 16]);
    assert!(!layer.is_empty());
    assert_eq!(layer.hashes().len(), 2);
    assert_eq!(layer.hashes()[0], [1u8; 16]);
    assert_eq!(layer.hashes()[1], [2u8; 16]);
}

#[test]
fn test_bfs_layer_default() {
    let layer = BfsLayer::default();
    assert!(layer.is_empty());
    assert_eq!(layer.len(), 0);
}

#[test]
fn test_fetch_frontier_sequential() {
    use mtxdb::storage::InMemoryStorage;

    let engine = InMemoryStorage::new();
    let collection = [0x01; 16];
    let id1 = [1u8; 16];
    let id2 = [2u8; 16];

    engine
        .put(
            &collection,
            &id1,
            &NodeData::new(bytes::Bytes::from_static(b"node1")),
        )
        .unwrap();
    engine
        .put(
            &collection,
            &id2,
            &NodeData::new(bytes::Bytes::from_static(b"node2")),
        )
        .unwrap();

    let batch = FrontierBatch::new(vec![id1, id2, [3u8; 16]]); // id3 not in store
    let results = fetch_frontier_sequential(&engine, &collection, &batch).unwrap();

    assert_eq!(results.len(), 3);
    assert!(results[0].1.is_some());
    assert!(results[1].1.is_some());
    assert!(results[2].1.is_none());
}
