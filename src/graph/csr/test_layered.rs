use super::Csr;
use std::collections::HashMap;

fn id(n: u8) -> [u8; 16] {
    [n; 16]
}

#[test]
fn layered_order_puts_dependencies_first_and_breaks_ties_by_id() {
    // 5 -> 3, 4 ; 3 -> 1 ; 4 -> 1 ; 1 -> 0 ; 2 is isolated.
    // Listed in scrambled input order on purpose.
    let nodes = [id(5), id(2), id(4), id(1), id(3), id(0)];
    let mut adj: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
    adj.insert(id(5), vec![id(3), id(4)]);
    adj.insert(id(3), vec![id(1)]);
    adj.insert(id(4), vec![id(1)]);
    adj.insert(id(1), vec![id(0)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let order: Vec<[u8; 16]> = csr
        .layered_order()
        .expect("acyclic")
        .into_iter()
        .map(|l| *csr.hash_of(l).unwrap())
        .collect();
    // level 0: {0, 2}; 1: {1}; 2: {3, 4}; 3: {5}
    assert_eq!(order, vec![id(0), id(2), id(1), id(3), id(4), id(5)]);
}

#[test]
fn layered_order_rejects_cycles() {
    let nodes = [id(1), id(2)];
    let mut adj: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
    adj.insert(id(1), vec![id(2)]);
    adj.insert(id(2), vec![id(1)]);
    assert!(Csr::build_from_edges(&nodes, &adj).layered_order().is_none());
}
