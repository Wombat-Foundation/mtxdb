#![allow(clippy::tests_outside_test_module)]

use mtxdb::dag::*;

#[test]
fn test_graph_edge_packing() {
    let resident = GraphEdge::resident(42);
    assert!(resident.is_resident());
    assert_eq!(resident.arena_index(), 42);

    let disk = GraphEdge::disk(12345);
    assert!(!disk.is_resident());
    assert_eq!(disk.local_id(), 12345);
}

#[test]
fn test_insert_event_basic() {
    let mut frontier = ActiveRoomFrontier::new();

    let idx_a = frontier.insert_event(100, &[], &[]);
    assert_eq!(idx_a, 0);
    assert_eq!(frontier.len(), 1);

    let idx_b = frontier.insert_event(101, &[100], &[100]);
    assert_eq!(idx_b, 1);
    assert_eq!(frontier.len(), 2);

    let prev = frontier.prev_edges(idx_b);
    assert_eq!(prev.len(), 1);
    assert!(prev[0].is_resident());
    assert_eq!(prev[0].arena_index(), 0);
}

#[test]
fn test_insert_event_with_disk_parent() {
    let mut frontier = ActiveRoomFrontier::new();

    let idx_a = frontier.insert_event(100, &[999], &[]);
    assert_eq!(idx_a, 0);

    let prev = frontier.prev_edges(idx_a);
    assert_eq!(prev.len(), 1);
    assert!(!prev[0].is_resident());
}

#[test]
fn test_rebind_resident_edges_after_out_of_order_insert() {
    let mut frontier = ActiveRoomFrontier::new();
    let child = frontier.insert_event(101, &[100, 999], &[]);
    let parent = frontier.insert_event(100, &[], &[]);

    // Deliberately pins the pre-rebind state (a tripwire, not a bug): if a
    // future fix makes the late parent resident on insert, this fails and
    // the explicit post-batch rebind below can be removed.
    assert!(!frontier.prev_edges(child)[0].is_resident());
    frontier.rebind_resident_edges();

    let prev = frontier.prev_edges(child);
    assert!(prev[0].is_resident());
    assert_eq!(prev[0].arena_index(), parent);
    assert!(!prev[1].is_resident());
}

#[test]
fn test_clear_is_o1() {
    let mut frontier = ActiveRoomFrontier::new();
    for i in 0..1000 {
        frontier.insert_event(i, &[], &[]);
    }
    assert_eq!(frontier.len(), 1000);
    frontier.clear();
    assert_eq!(frontier.len(), 0);
    assert!(frontier.is_empty());
}

#[test]
fn test_auth_edges_resident_and_disk() {
    let mut frontier = ActiveRoomFrontier::new();
    let a = frontier.insert_event(100, &[], &[]);
    let b = frontier.insert_event(101, &[], &[100]);
    let c = frontier.insert_event(102, &[], &[100, 999]);

    assert_eq!(frontier.auth_edges(a), &[]);

    let auth_b = frontier.auth_edges(b);
    assert_eq!(auth_b.len(), 1);
    assert!(auth_b[0].is_resident());
    assert_eq!(auth_b[0].arena_index(), a);

    let auth_c = frontier.auth_edges(c);
    assert_eq!(auth_c.len(), 2);
    assert!(auth_c[0].is_resident());
    assert_eq!(auth_c[0].arena_index(), a);
    assert!(!auth_c[1].is_resident());
}

#[test]
fn test_active_collection_frontier_default() {
    let frontier = ActiveRoomFrontier::default();
    assert!(frontier.is_empty());
    assert_eq!(frontier.len(), 0);
}

#[test]
fn test_id_remap() {
    let mut frontier = ActiveRoomFrontier::new();
    let local_a = frontier.register_id(100);
    let local_b = frontier.register_id(200);
    let local_a2 = frontier.register_id(100);

    assert_eq!(local_a, 0);
    assert_eq!(local_b, 1);
    assert_eq!(local_a2, 0);
}
