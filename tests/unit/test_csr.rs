#![allow(clippy::tests_outside_test_module)]

use mtxdb::csr::*;
use std::collections::HashMap;

fn h(byte: u8) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[0] = byte;
    id
}

#[test]
fn test_linear_chain() {
    let nodes = vec![h(1), h(2), h(3), h(4)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2)]);
    adj.insert(h(2), vec![h(3)]);
    adj.insert(h(3), vec![h(4)]);

    let csr = Csr::build_from_edges(&nodes, &adj);
    assert_eq!(csr.node_count(), 4);
    assert_eq!(csr.edge_count(), 3);

    let order = csr.topo_order();
    assert_eq!(order.len(), 4);
    let order_hashes: Vec<[u8; 16]> = order.iter().map(|&l| *csr.hash_of(l).unwrap()).collect();
    assert_eq!(order_hashes, vec![h(1), h(2), h(3), h(4)]);
}

#[test]
fn test_diamond_dag() {
    let nodes = vec![h(1), h(2), h(3), h(4)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2), h(3)]);
    adj.insert(h(2), vec![h(4)]);
    adj.insert(h(3), vec![h(4)]);

    let csr = Csr::build_from_edges(&nodes, &adj);
    let order = csr.topo_order();
    assert_eq!(order.len(), 4);

    let pos = |hash: u8| {
        order
            .iter()
            .position(|&l| *csr.hash_of(l).unwrap() == h(hash))
            .unwrap()
    };
    assert!(pos(1) < pos(2));
    assert!(pos(1) < pos(3));
    assert!(pos(2) < pos(4));
    assert!(pos(3) < pos(4));
}

#[test]
fn test_disconnected_nodes() {
    let nodes = vec![h(1), h(2), h(3)];
    let adj = HashMap::new();

    let csr = Csr::build_from_edges(&nodes, &adj);
    let order = csr.topo_order();
    assert_eq!(order.len(), 3);
}

#[test]
fn test_serialize_roundtrip() {
    let nodes = vec![h(1), h(2), h(3)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2)]);
    adj.insert(h(2), vec![h(3)]);

    let csr = Csr::build_from_edges(&nodes, &adj);
    let bytes = csr.serialize();
    let restored = Csr::deserialize(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);
    assert_eq!(restored.edge_count(), 2);
    assert_eq!(restored.neighbors(0), &[1]);
    assert_eq!(restored.neighbors(1), &[2]);
    assert_eq!(restored.neighbors(2), &[]);
}

#[test]
fn test_local_id_lookup() {
    let nodes = vec![h(10), h(20), h(30)];
    let adj = HashMap::new();

    let csr = Csr::build_from_edges(&nodes, &adj);
    assert_eq!(csr.local_id(&h(10)), Some(0));
    assert_eq!(csr.local_id(&h(20)), Some(1));
    assert_eq!(csr.local_id(&h(30)), Some(2));
    assert_eq!(csr.local_id(&h(99)), None);
}

#[test]
fn test_csr_error_display() {
    assert_eq!(CsrError::TooShort.to_string(), "csr data too short");
    assert_eq!(
        CsrError::UnsupportedVersion(99).to_string(),
        "unsupported csr version: 99"
    );
}

#[test]
fn test_empty_graph() {
    let csr = Csr::build_from_edges(&[], &HashMap::new());
    assert_eq!(csr.node_count(), 0);
    assert_eq!(csr.edge_count(), 0);
    assert_eq!(csr.topo_order().len(), 0);
}

#[test]
fn test_topo_order_ready_queue_is_deterministic() {
    // Initial ready nodes are queued by local ID, then newly ready nodes
    // follow the FIFO queue in CSR adjacency order.
    let nodes = vec![h(0), h(1), h(2), h(3)];
    let mut adj = HashMap::new();
    adj.insert(h(0), vec![h(3)]);
    adj.insert(h(3), vec![h(1), h(2)]);

    let csr = Csr::build_from_edges(&nodes, &adj);
    let order = csr.topo_order();
    assert_eq!(order, vec![0, 3, 1, 2]);
}

#[test]
fn test_topo_order_uses_fifo_for_newly_ready_nodes() {
    // A min-heap would choose 2 before 3 after processing node 1. The
    // linear-time FIFO traversal preserves the order in which nodes
    // become ready: 3 is discovered before 2.
    let nodes = vec![h(0), h(1), h(2), h(3)];
    let mut adj = HashMap::new();
    adj.insert(h(0), vec![h(3)]);
    adj.insert(h(1), vec![h(2)]);

    let csr = Csr::build_from_edges(&nodes, &adj);
    assert_eq!(csr.topo_order(), vec![0, 1, 3, 2]);
}

#[test]
fn test_deserialize_malformed_first_offset_not_zero() {
    let nodes = vec![h(1), h(2)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let mut bytes = csr.serialize();
    // Corrupt offsets[0]: change first offset byte from 0 to 1
    bytes[9] = 1;
    assert!(matches!(Csr::deserialize(&bytes), Err(CsrError::Malformed)));
}

#[test]
fn test_deserialize_malformed_offsets_not_monotonic() {
    let nodes = vec![h(1), h(2), h(3)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2), h(3)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let mut bytes = csr.serialize();
    // offsets = [0, 2, 2, 2]; corrupt offsets[3] from 2 to 1 → [0, 2, 2, 1]
    // offsets[3] < offsets[2] → decreasing
    bytes[21] = 1;
    assert!(matches!(Csr::deserialize(&bytes), Err(CsrError::Malformed)));
}

#[test]
fn test_deserialize_malformed_final_offset_wrong() {
    let nodes = vec![h(1), h(2)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let mut bytes = csr.serialize();
    // Corrupt offsets[2]: should be 1 (edge count), change to 2
    bytes[17] = 2;
    assert!(matches!(Csr::deserialize(&bytes), Err(CsrError::Malformed)));
}

#[test]
fn test_deserialize_malformed_target_out_of_range() {
    let nodes = vec![h(1), h(2)];
    let mut adj = HashMap::new();
    adj.insert(h(1), vec![h(2)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let mut bytes = csr.serialize();
    // Corrupt targets[0]: should be 1 (local id for h(2)), change to 99
    let targets_offset = 9 + 3 * 4; // version + (n+1)*4 offsets
    bytes[targets_offset] = 99;
    assert!(matches!(Csr::deserialize(&bytes), Err(CsrError::Malformed)));
}
