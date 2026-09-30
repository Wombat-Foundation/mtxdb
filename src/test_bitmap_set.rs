#![cfg(test)]

use super::bitmap_set::*;
use crate::layout::ShardType;
use crate::storage::StorageError;

const A: DomainTag = DomainTag::new([0x11; 16]);
const B: DomainTag = DomainTag::new([0x22; 16]);

fn values(set: &BitmapSet) -> Vec<u32> {
    set.iter().collect()
}

#[test]
fn empty_set_is_empty_and_has_no_members() {
    let set = BitmapSet::new(A);
    assert!(set.is_empty(), "new set must be empty");
    assert_eq!(set.len(), 0, "new set must have length zero");
    assert!(!set.contains(1), "new set must contain nothing");
    assert_eq!(set.domain(), A, "the domain must be retained");
    assert_eq!(values(&set), Vec::<u32>::new(), "iteration must be empty");
}

#[test]
fn insertion_and_removal_report_membership_changes() {
    let mut set = BitmapSet::from_values(A, [3, 1, 2, 3]);
    assert_eq!(values(&set), vec![1, 2, 3], "values sort and de-duplicate");
    assert!(set.contains(2), "inserted id must be present");
    assert!(!set.insert(2), "re-inserting an id is not a change");
    assert!(set.insert(4), "a new id is a change");
    assert!(set.remove(1), "removing a present id is a change");
    assert!(!set.remove(1), "removing an absent id is not a change");
    assert_eq!(values(&set), vec![2, 3, 4], "removal must take effect");
}

#[test]
fn algebra_union_intersection_and_difference() {
    let left = BitmapSet::from_values(A, [1, 2, 3, 5]);
    let right = BitmapSet::from_values(A, [3, 4, 5]);
    assert_eq!(
        values(&left.union(&right).unwrap()),
        vec![1, 2, 3, 4, 5],
        "union must combine both sides"
    );
    assert_eq!(
        values(&left.intersection(&right).unwrap()),
        vec![3, 5],
        "intersection must keep shared ids"
    );
    assert_eq!(
        values(&left.difference(&right).unwrap()),
        vec![1, 2],
        "difference must keep left-only ids"
    );
}

#[test]
fn subset_and_disjoint_predicates() {
    let small = BitmapSet::from_values(A, [2, 4]);
    let big = BitmapSet::from_values(A, [1, 2, 3, 4]);
    let other = BitmapSet::from_values(A, [7, 8]);
    assert!(
        small.is_subset(&big).unwrap(),
        "a set must be a subset of its superset"
    );
    assert!(
        !big.is_subset(&small).unwrap(),
        "a superset must not be a subset of a smaller set"
    );
    assert!(
        small.is_disjoint(&other).unwrap(),
        "sets with no members in common are disjoint"
    );
    assert!(
        !big.is_disjoint(&small).unwrap(),
        "overlapping sets are not disjoint"
    );
}

#[test]
fn round_trip_preserves_domain_and_members() {
    let set = BitmapSet::from_values(A, [0, 1, 7, 70_000, 4_294_967_295]);
    let bytes = set.encode().unwrap();
    let decoded = BitmapSet::decode(&bytes).unwrap();
    assert_eq!(decoded, set, "a round trip must preserve the whole set");
    assert_eq!(decoded.domain(), A, "a round trip must preserve the domain");
    assert_eq!(
        BitmapSet::decode_in_domain(&bytes, A).unwrap(),
        set,
        "decode_in_domain must accept the stored domain"
    );
}

#[test]
fn empty_set_round_trips() {
    let set = BitmapSet::new(A);
    let bytes = set.encode().unwrap();
    let decoded = BitmapSet::decode(&bytes).unwrap();
    assert!(decoded.is_empty(), "an empty set must stay empty");
    assert_eq!(decoded.domain(), A, "an empty set keeps its domain");
}

#[test]
fn large_containers_round_trip() {
    let dense = BitmapSet::from_values(A, 0..100_000u32);
    assert_eq!(dense.len(), 100_000, "every dense id must be stored");
    let decoded = BitmapSet::decode(&dense.encode().unwrap()).unwrap();
    assert_eq!(decoded, dense, "dense containers must round-trip");
}

#[test]
fn malformed_input_is_rejected() {
    let set = BitmapSet::from_values(A, [1, 2, 3]);
    let good = set.encode().unwrap();

    assert!(
        BitmapSet::decode(&[]).is_err(),
        "an empty buffer is not a set"
    );

    let mut bad_magic = good.clone();
    bad_magic[0] = bad_magic[0].wrapping_add(1);
    assert!(
        BitmapSet::decode(&bad_magic).is_err(),
        "a bad magic must be rejected"
    );

    let mut bad_version = good.clone();
    bad_version[4] = bad_version[4].wrapping_add(1);
    assert!(
        BitmapSet::decode(&bad_version).is_err(),
        "an unknown version must be rejected"
    );

    let truncated = &good[..good.len().saturating_sub(1)];
    assert!(
        BitmapSet::decode(truncated).is_err(),
        "a truncated payload must be rejected"
    );

    let mut trailing = good.clone();
    trailing.push(0);
    assert!(
        BitmapSet::decode(&trailing).is_err(),
        "trailing bytes must be rejected"
    );
}

#[test]
fn domain_mismatch_is_rejected_everywhere() {
    let left = BitmapSet::from_values(A, [1, 2]);
    let right = BitmapSet::from_values(B, [2, 3]);

    assert!(
        matches!(left.union(&right), Err(StorageError::Collision(_))),
        "union across domains must be a collision"
    );
    assert!(
        matches!(left.intersection(&right), Err(StorageError::Collision(_))),
        "intersection across domains must be a collision"
    );
    assert!(
        matches!(left.difference(&right), Err(StorageError::Collision(_))),
        "difference across domains must be a collision"
    );
    assert!(
        matches!(left.is_subset(&right), Err(StorageError::Collision(_))),
        "subset across domains must be a collision"
    );
    assert!(
        matches!(left.is_disjoint(&right), Err(StorageError::Collision(_))),
        "disjoint across domains must be a collision"
    );

    let bytes = left.encode().unwrap();
    assert!(
        matches!(
            BitmapSet::decode_in_domain(&bytes, B),
            Err(StorageError::Collision(_))
        ),
        "decoding for the wrong domain must be a collision"
    );
    assert_eq!(
        BitmapSet::decode(&bytes).unwrap().domain(),
        A,
        "plain decode must still report the stored domain"
    );
}

#[test]
fn derived_domain_tags_are_stable_and_distinct() {
    assert_eq!(
        DomainTag::derive(b"room-a"),
        DomainTag::derive(b"room-a"),
        "the same label must derive the same tag"
    );
    assert_ne!(
        DomainTag::derive(b"room-a"),
        DomainTag::derive(b"room-b"),
        "different labels must derive different tags"
    );
    assert_ne!(
        DomainTag::for_scope(ShardType::Edges, &[0; 16]),
        DomainTag::for_scope(ShardType::State, &[0; 16]),
        "the same collection id in different pools must not share a domain"
    );
    assert_ne!(
        DomainTag::for_scope(ShardType::Edges, &[0; 16]),
        DomainTag::for_scope(ShardType::Edges, &[1; 16]),
        "different collections must not share a domain"
    );
    assert_eq!(
        DomainTag::for_scope(ShardType::Edges, &[7; 16]),
        DomainTag::for_scope(ShardType::Edges, &[7; 16]),
        "a scope must derive a stable tag"
    );
}
