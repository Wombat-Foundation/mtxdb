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
    assert!(Csr::build_from_edges(&nodes, &adj)
        .layered_order()
        .is_none());
}

#[test]
fn a_cycle_falls_back_to_id_order_instead_of_failing() {
    let nodes = [id(3), id(1), id(2)];
    let mut adj: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
    adj.insert(id(1), vec![id(2)]);
    adj.insert(id(2), vec![id(1)]);
    let csr = Csr::build_from_edges(&nodes, &adj);
    let (order, cyclic) = csr.layered_or_id_order();
    assert!(cyclic);
    let ids: Vec<[u8; 16]> = order
        .into_iter()
        .map(|l| *csr.hash_of(l).unwrap())
        .collect();
    assert_eq!(ids, vec![id(1), id(2), id(3)]);
}

fn rows(pairs: &[(u8, &[u8])]) -> HashMap<[u8; 16], Vec<[u8; 16]>> {
    pairs
        .iter()
        .map(|(source, targets)| (id(*source), targets.iter().map(|t| id(*t)).collect()))
        .collect()
}

fn sample_families() -> super::CsrFamilies {
    // Node 9 is referenced but has no event body; node 4 has an empty prev row.
    let prev = rows(&[(3, &[1, 9]), (4, &[])]);
    let auth = rows(&[(3, &[1])]);
    let relations = rows(&[(3, &[4])]);
    let types = HashMap::from([(id(3), vec![7u16])]);
    super::CsrFamilies::build(&[id(1), id(3), id(4)], &prev, &auth, &relations, types)
}

#[test]
fn csr_round_trips_and_reports_shape() {
    let adj = rows(&[(2, &[1]), (3, &[1, 2])]);
    let csr = Csr::build_from_edges(&[id(1), id(2), id(3)], &adj);
    assert_eq!(csr.edge_count(), 3);
    assert!(csr.memory_usage() > 0);
    assert!(csr.local_id(&id(2)).is_some());
    assert!(csr.local_id(&id(77)).is_none());
    let decoded = Csr::deserialize(&csr.serialize()).unwrap();
    assert_eq!(decoded.node_count(), 3);
    assert_eq!(decoded.edge_count(), 3);
    assert!(decoded.memory_usage() > 0);
}

#[test]
fn csr_deserialize_rejects_bad_buffers() {
    use super::CsrError;
    let adj = rows(&[(2, &[1])]);
    let good = Csr::build_from_edges(&[id(1), id(2)], &adj).serialize();
    assert!(matches!(Csr::deserialize(&[0; 3]), Err(CsrError::TooShort)));
    let mut bad_version = good.clone();
    bad_version[0] = 99;
    assert!(matches!(
        Csr::deserialize(&bad_version),
        Err(CsrError::UnsupportedVersion(99))
    ));
    assert!(matches!(
        Csr::deserialize(&good[..good.len() - 1]),
        Err(CsrError::TooShort)
    ));
    let mut bad_target = good.clone();
    let last = bad_target.len() - 4;
    bad_target[last..].copy_from_slice(&50u32.to_le_bytes());
    assert!(matches!(
        Csr::deserialize(&bad_target),
        Err(CsrError::Malformed)
    ));
    let mut bad_first = good.clone();
    bad_first[9..13].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        Csr::deserialize(&bad_first),
        Err(CsrError::Malformed)
    ));
    for error in [
        CsrError::TooShort,
        CsrError::UnsupportedVersion(3),
        CsrError::Malformed,
    ] {
        assert_ne!(error.to_string(), "");
    }
}

#[test]
fn csr_families_round_trip_preserves_rows_and_relation_types() {
    let families = sample_families();
    assert!(families.nodes().contains(&id(9)));
    assert!(families.has_prev_row(&id(4)));
    assert!(!families.has_prev_row(&id(9)));
    assert!(families.has_auth_row(&id(3)));
    assert!(!families.has_auth_row(&id(4)));
    assert!(families.has_relation_row(&id(3)));
    assert_eq!(families.relation_types(&id(3)), &[7]);
    assert_eq!(families.relation_types(&id(1)), []);

    let decoded = super::CsrFamilies::deserialize(&families.serialize().unwrap()).unwrap();
    assert_eq!(decoded.nodes(), families.nodes());
    assert!(decoded.has_prev_row(&id(4)));
    assert!(!decoded.has_auth_row(&id(4)));
    assert!(decoded.has_relation_row(&id(3)));
    assert_eq!(decoded.relation_types(&id(3)), &[7]);
    assert_eq!(decoded.prev.edge_count(), families.prev.edge_count());
}

#[test]
fn csr_families_serialize_rejects_misaligned_relation_types() {
    let relations = rows(&[(3, &[4])]);
    let families = super::CsrFamilies::build(
        &[id(3), id(4)],
        &HashMap::new(),
        &HashMap::new(),
        &relations,
        HashMap::new(),
    );
    assert!(families.serialize().is_err());
}

#[test]
fn csr_families_deserialize_rejects_damage() {
    use super::CsrError;
    let good = sample_families().serialize().unwrap();
    assert!(matches!(
        super::CsrFamilies::deserialize(&good[..8]),
        Err(CsrError::TooShort)
    ));
    let recrc = |mut bytes: Vec<u8>| {
        let body = bytes.len() - 4;
        let crc = crc32fast::hash(&bytes[..body]);
        bytes[body..].copy_from_slice(&crc.to_le_bytes());
        bytes
    };
    // Flipped payload byte without fixing the checksum.
    let mut corrupt = good.clone();
    corrupt[10] ^= 0xff;
    assert!(super::CsrFamilies::deserialize(&corrupt).is_err());
    // Wrong magic / version with a valid checksum.
    let mut magic = good.clone();
    magic[0] ^= 0xff;
    assert!(super::CsrFamilies::deserialize(&recrc(magic)).is_err());
    let mut version = good.clone();
    version[4] = 200;
    assert!(matches!(
        super::CsrFamilies::deserialize(&recrc(version)),
        Err(CsrError::UnsupportedVersion(200))
    ));
    // Unsorted node directory.
    let mut unsorted = good.clone();
    let (a, b) = (9usize, 25usize);
    for i in 0..16 {
        unsorted.swap(a + i, b + i);
    }
    assert!(super::CsrFamilies::deserialize(&recrc(unsorted)).is_err());
    // Row flag outside {0,1}.
    let mut flag = good.clone();
    let node_count = u32::from_le_bytes(good[5..9].try_into().unwrap()) as usize;
    flag[9 + node_count * 16] = 2;
    assert!(super::CsrFamilies::deserialize(&recrc(flag)).is_err());
    // Trailing garbage before the checksum.
    let mut trailing = good[..good.len() - 4].to_vec();
    trailing.extend_from_slice(&[0, 0]);
    trailing.extend_from_slice(&[0; 4]);
    assert!(super::CsrFamilies::deserialize(&recrc(trailing)).is_err());
}
