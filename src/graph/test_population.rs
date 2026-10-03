use super::population::*;
use crate::database::Database;
use crate::layout::ShardType;
use crate::short_id::{BatchEvent, ShortIdIndex};
use crate::storage::StorageError;
use rezzy_recon::{build_bucket_nodes, BucketRequest, ElementHash, Population, SortedPopulation};
use std::path::PathBuf;

const SCOPE: [u8; 16] = [0x61; 16];

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-population-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn index() -> ShortIdIndex {
    ShortIdIndex::new(ShardType::Edges, SCOPE)
}

/// A spread-out hash for element `i`: distinct digests, varied top bits.
fn hash(i: u32) -> ElementHash {
    let mut digest = [0_u8; 32];
    let mut x = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for chunk in digest.chunks_exact_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_be_bytes());
    }
    ElementHash::from_digest32(digest)
}

fn record(db: &Database, range: std::ops::Range<u32>) {
    for i in range {
        let key = format!("$e{i}").into_bytes();
        let payload = encode_owner_payload(hash(i));
        index()
            .record_events(
                db,
                &[BatchEvent {
                    owner: &key,
                    families: &[],
                    owner_payload: Some(&payload),
                }],
            )
            .unwrap();
    }
}

/// The nodes a round touches: four disjoint depth-2 buckets.
fn nodes() -> Vec<BucketRequest> {
    (0..4)
        .map(|prefix| BucketRequest::new(2, prefix, 16))
        .collect()
}

#[test]
fn an_empty_store_has_an_empty_population() {
    let root = test_root("empty");
    let db = Database::open(root.clone()).unwrap();
    let snapshot = index().population_snapshot(&db).unwrap();
    assert!(snapshot.is_empty());
    assert_eq!(snapshot.manifest_version(), 0);
    assert_eq!(snapshot.owner_seq_ceiling(), 1);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_snapshot_serves_exactly_the_logged_owners() {
    let root = test_root("serves");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..40);
    // An id allocated only because something referenced it is not an owner.
    index().get_or_create(&db, &[b"$referenced"]).unwrap();
    let snapshot = index().population_snapshot(&db).unwrap();
    let expected = SortedPopulation::new((0..40).map(hash).collect());
    assert_eq!(snapshot.len(), 40);
    assert_eq!(snapshot.owner_seq_ceiling(), 41);
    assert_eq!(
        build_bucket_nodes(&snapshot, &nodes()).unwrap(),
        build_bucket_nodes(&expected, &nodes()).unwrap()
    );
    let mut got = Vec::new();
    snapshot.candidates_into(hash(7).h64, &mut got);
    assert_eq!(got, vec![hash(7).h128]);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Writers keep appending between the rounds of an exchange; a snapshot taken
/// before them answers every round from the same population.
#[test]
fn a_pinned_snapshot_ignores_inserts_between_rounds() {
    let root = test_root("pinned");
    let db = Database::open(root.clone()).unwrap();
    record(&db, 0..30);
    let pinned = index().population_snapshot(&db).unwrap();
    let original = SortedPopulation::new((0..30).map(hash).collect());
    let want = build_bucket_nodes(&original, &nodes()).unwrap();

    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    record(&db, 30..45);
    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    record(&db, 45..60);
    assert_eq!(build_bucket_nodes(&pinned, &nodes()).unwrap(), want);
    assert_eq!(pinned.len(), 30);
    assert_eq!(pinned.owner_seq_ceiling(), 31);

    let fresh = index().population_snapshot(&db).unwrap();
    assert_eq!(fresh.len(), 60);
    assert_eq!(fresh.owner_seq_ceiling(), 61);
    assert_ne!(build_bucket_nodes(&fresh, &nodes()).unwrap(), want);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_malformed_payload_is_corruption() {
    let root = test_root("malformed");
    let db = Database::open(root.clone()).unwrap();
    index()
        .record_events(
            &db,
            &[BatchEvent {
                owner: b"$a",
                families: &[],
                owner_payload: Some(b"too short"),
            }],
        )
        .unwrap();
    assert!(matches!(
        index().population_snapshot(&db),
        Err(StorageError::Corrupt(_))
    ));
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
