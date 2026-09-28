use super::*;
use crate::storage::{InMemoryStorage, StorageEngine};
use crate::template::MEMBER_NAMESPACE_INTL;

#[test]
fn named_indexes_share_storage_but_have_distinct_collections() {
    let engine = InMemoryStorage::new();
    let first = AuxiliaryIndex::open(&engine, "first");
    let second = AuxiliaryIndex::open(&engine, "second");
    assert_ne!(first.collection_id(), second.collection_id());
    first.put(b"key", b"value").unwrap();
    assert_eq!(first.get(b"key").unwrap(), Some(b"value".to_vec()));
    assert_eq!(second.get(b"key").unwrap(), None);
}

#[test]
fn get_many_preserves_order_and_distinguishes_missing_and_empty_values() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "batch");
    index.put(b"first", b"one").unwrap();
    index.put(b"empty", b"").unwrap();

    let values = index.get_many(&[b"empty", b"missing", b"first"]).unwrap();
    assert_eq!(values[0], Some(Vec::new()));
    assert_eq!(values[1], None);
    assert_eq!(values[2], Some(b"one".to_vec()));
}

#[test]
fn empty_put_many_establishes_metadata_for_a_fresh_index() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "empty_batch");
    assert_eq!(index.put_many(&[]).unwrap(), 0);
    assert_eq!(
        engine
            .get_collection_metadata(&index.collection_id())
            .unwrap(),
        Some(index.metadata())
    );
    index.put(b"key", b"value").unwrap();
    assert_eq!(index.get(b"key").unwrap(), Some(b"value".to_vec()));
}

#[test]
fn get_many_rejects_a_corrupt_envelope() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "corrupt");
    let digest = auxiliary_key_digest(b"bad");
    let node_id = physical_id(&digest);
    engine
        .put(
            &index.collection_id(),
            &node_id,
            &NodeData::new(VALUE_MAGIC.to_vec().into()),
        )
        .unwrap();

    assert!(index.get_many(&[b"bad"]).is_err());
}

#[test]
fn envelope_retains_full_key_digest() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "test");
    index.put(b"key", b"value").unwrap();
    let digest = auxiliary_key_digest(b"key");
    let node_id: NodeId = digest[..16].try_into().unwrap();
    let raw = engine
        .get(&index.collection_id(), &node_id)
        .unwrap()
        .unwrap();
    assert_eq!(&raw.bytes[..4], VALUE_MAGIC);
    assert_eq!(&raw.bytes[4..36], &digest);
}

#[test]
fn legacy_values_remain_readable() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "legacy");
    let digest = auxiliary_key_digest(b"key");
    let node_id: NodeId = digest[..16].try_into().unwrap();
    engine
        .put(
            &index.collection_id(),
            &node_id,
            &NodeData::new(bytes::Bytes::from_static(b"old")),
        )
        .unwrap();
    assert_eq!(index.get(b"key").unwrap(), Some(b"old".to_vec()));
}

#[test]
fn auxiliary_index_establishes_metadata_on_put() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "sys:matrix-state-groups");
    index.put(b"event_1", b"state_group_1").unwrap();
    let meta = engine
        .get_collection_metadata(&index.collection_id())
        .unwrap()
        .expect("metadata must be established on put");
    assert_eq!(meta.collection_canonical_id, b"sys:matrix-state-groups");
    assert_eq!(meta.member_namespace, Some(MEMBER_NAMESPACE_INTL));
    assert_eq!(meta.role.as_deref(), Some("system_auxiliary"));
    assert_eq!(meta.record_id_rule.policy, FrameIdPolicy::Key);
    assert!(meta.verify_collection_id(&index.collection_id()));

    // Idempotent retry succeeds
    index.put(b"event_1", b"state_group_1").unwrap();
    assert_eq!(
        index.get(b"event_1").unwrap(),
        Some(b"state_group_1".to_vec())
    );

    // Same-key replacement succeeds and updates value
    index.put(b"event_1", b"different_group").unwrap();
    assert_eq!(
        index.get(b"event_1").unwrap(),
        Some(b"different_group".to_vec())
    );
}

#[test]
fn auxiliary_index_rejects_truncated_id_collision_without_corrupting() {
    let engine = InMemoryStorage::new();
    let index = AuxiliaryIndex::open(&engine, "sys:test-collisions");
    index.put(b"key_1", b"val_1").unwrap();
    assert_eq!(index.get(b"key_1").unwrap(), Some(b"val_1".to_vec()));

    let digest1 = auxiliary_key_digest(b"key_1");
    let node_id1 = physical_id(&digest1);

    // Fabricate a colliding key that has a different full digest but targets the same node_id
    let mut colliding_digest = digest1;
    colliding_digest[31] ^= 0xFF; // Different 32-byte digest, same first 16 bytes!
    assert_eq!(&colliding_digest[..16], &node_id1[..]);

    let mut encoded = Vec::new();
    encoded.extend_from_slice(VALUE_MAGIC);
    encoded.extend_from_slice(&colliding_digest);
    encoded.extend_from_slice(b"colliding_val");
    let colliding_data = NodeData::new(bytes::Bytes::from(encoded));

    let mut validate = |existing: Option<&NodeData>| -> Result<(), StorageError> {
        if let Some(existing) = existing {
            if !existing.bytes.is_empty() {
                decode_value(&colliding_digest, &existing.bytes)?;
            }
        }
        Ok(())
    };

    let err = engine
        .create_or_upsert_established_validated(
            &index.collection_id(),
            &index.metadata(),
            &node_id1,
            &colliding_data,
            &mut validate,
        )
        .unwrap_err();

    assert!(matches!(err, StorageError::Collision(_)));

    // Original key remains intact and uncorrupted:
    assert_eq!(index.get(b"key_1").unwrap(), Some(b"val_1".to_vec()));
}
