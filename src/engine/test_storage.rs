use super::*;
use std::error::Error;

const TEST_COLLECTION: [u8; 16] = [0x01; 16];

#[test]
fn digest_algorithm_ids_roundtrip_and_hash() {
    // The default id is stable and round-trips.
    assert_eq!(DigestAlgorithm::Sha256.id(), 0x01);
    assert_eq!(DigestAlgorithm::from_id(0x01), DigestAlgorithm::Sha256);
    assert_eq!(DigestAlgorithm::Blake3.id(), 0x02);
    assert_eq!(DigestAlgorithm::from_id(0x02), DigestAlgorithm::Blake3);

    // An unknown id is preserved rather than reinterpreted.
    assert_eq!(
        DigestAlgorithm::from_id(0x7f),
        DigestAlgorithm::Unknown(0x7f)
    );
    assert_eq!(DigestAlgorithm::from_id(0x7f).id(), 0x7f);

    // SHA-256 matches the reference digest for a known vector.
    let digest = content_digest(DigestAlgorithm::Sha256, b"abc");
    assert_eq!(
        digest,
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ]
    );

    assert_eq!(
        content_digest(DigestAlgorithm::Blake3, b"abc"),
        *blake3::hash(b"abc").as_bytes()
    );
}

#[test]
fn test_in_memory_roundtrip() {
    let store = InMemoryStorage::new();
    let id = [0x42u8; 16];
    let data = NodeData::new(bytes::Bytes::from_static(b"test node data"));

    store.put(&TEST_COLLECTION, &id, &data).unwrap();
    let fetched = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
    assert_eq!(fetched.bytes, data.bytes);
}

#[test]
fn test_in_memory_not_found() {
    let store = InMemoryStorage::new();
    assert!(store.get(&TEST_COLLECTION, &[0x00; 16]).unwrap().is_none());
}

#[test]
fn collection_metadata_roundtrips_and_verifies() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };

    let store = InMemoryStorage::new();
    let metadata = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: Some(br#"{"ext":"matrix.room","fmt":1,"room_version":"10"}"#.to_vec()),
        role: None,
        schema: None,
    };
    let collection =
        derive_collection_id(metadata.member_namespace, &metadata.collection_canonical_id);
    assert_eq!(
        store.get_collection_metadata(&collection).unwrap(),
        None,
        "a fresh collection has no genesis metadata"
    );

    store
        .ensure_collection_metadata(&collection, &metadata)
        .unwrap();
    assert_eq!(
        store.get_collection_metadata(&collection).unwrap(),
        Some(metadata.clone())
    );

    // Idempotent: repeating the same metadata is a no-op.
    store
        .ensure_collection_metadata(&collection, &metadata)
        .unwrap();

    // A different record is rejected rather than silently ignored.
    let conflicting = CollectionMetadata {
        collection_canonical_id: b"!other:matrix.org".to_vec(),
        ..metadata
    };
    assert!(matches!(
        store.ensure_collection_metadata(&collection, &conflicting),
        Err(StorageError::Internal(_))
    ));
}

/// Genesis metadata must precede the collection's first application
/// record: a late genesis write is rejected, so a crash can never leave a
/// collection with records but no metadata.
#[test]
fn genesis_metadata_must_precede_the_first_record() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };

    let metadata = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: None,
        schema: None,
    };
    let collection =
        derive_collection_id(metadata.member_namespace, &metadata.collection_canonical_id);

    // Correct order: metadata first, then records.
    let ordered = InMemoryStorage::new();
    ordered
        .ensure_collection_metadata(&collection, &metadata)
        .unwrap();
    ordered
        .put(
            &collection,
            &[0x01; 16],
            &NodeData::new(bytes::Bytes::from_static(b"e")),
        )
        .unwrap();
    assert_eq!(
        ordered.get_collection_metadata(&collection).unwrap(),
        Some(metadata.clone())
    );

    // Wrong order: a record first, then genesis metadata is rejected, and
    // the collection keeps no metadata rather than silently gaining a late
    // genesis record.
    let late = InMemoryStorage::new();
    late.put(
        &collection,
        &[0x02; 16],
        &NodeData::new(bytes::Bytes::from_static(b"e")),
    )
    .unwrap();
    assert!(matches!(
        late.ensure_collection_metadata(&collection, &metadata),
        Err(StorageError::Internal(_))
    ));
    assert_eq!(late.get_collection_metadata(&collection).unwrap(), None);
}

#[test]
fn empty_collection_entry_does_not_block_genesis_metadata() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };

    // An empty collection entry holds no record, so it must not be mistaken
    // for a non-empty collection when genesis metadata is established.
    // `put_many` no longer creates such an entry (an empty batch is a
    // no-op), so build it directly to keep the guard covered.
    let store = InMemoryStorage::new();
    let metadata = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: None,
        schema: None,
    };
    let collection =
        derive_collection_id(metadata.member_namespace, &metadata.collection_canonical_id);
    store.collections.write().entry(collection).or_default();
    assert!(store.collection_exists(&collection).unwrap());

    store
        .ensure_collection_metadata(&collection, &metadata)
        .unwrap();
    assert_eq!(
        store.get_collection_metadata(&collection).unwrap(),
        Some(metadata)
    );
}

#[test]
fn collection_len_counts_genesis_and_distinct_ids() {
    assert_collection_len_counts_genesis_and_distinct_ids(&InMemoryStorage::new());
}

/// An empty batch must be a no-op across every backend: no collection is
/// created, so `collection_exists`/`collection_len` report absence. This
/// mirrors the packfile-backend test of the same name; the two backends
/// previously disagreed (`InMemoryStorage` left an empty entry behind).
#[test]
fn empty_put_many_is_a_noop_and_does_not_create_the_collection() {
    let store = InMemoryStorage::new();
    let collection = [0x7Du8; 16];
    assert_eq!(store.put_many(&collection, &[]).unwrap(), 0);
    assert!(!store.collection_exists(&collection).unwrap());
    assert_eq!(store.collection_len(&collection).unwrap(), None);
}

#[test]
fn ensure_collection_metadata_is_atomic_under_concurrency() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    };
    use std::sync::Arc;

    let store = Arc::new(InMemoryStorage::new());
    let metadata = Arc::new(CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: None,
        schema: None,
    });
    let collection =
        derive_collection_id(metadata.member_namespace, &metadata.collection_canonical_id);

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            let metadata = Arc::clone(&metadata);
            std::thread::spawn(move || {
                store
                    .ensure_collection_metadata(&collection, &metadata)
                    .expect("concurrent genesis establishment must be idempotent");
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(
        store.get_collection_metadata(&collection).unwrap(),
        Some((*metadata).clone())
    );
}

#[test]
fn test_in_memory_batch() {
    let store = InMemoryStorage::new();
    let entries: Vec<(NodeId, NodeData)> = (0..10)
        .map(|i| {
            let mut id = [0u8; 16];
            id[0] = i;
            (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
        })
        .collect();

    let committed = store.put_many(&TEST_COLLECTION, &entries).unwrap();
    assert_eq!(committed, 10, "put_many reports every committed entry");
    assert_eq!(
        store.put_many(&TEST_COLLECTION, &[]).unwrap(),
        0,
        "an empty batch commits nothing"
    );

    let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
    let results = store.get_many(&TEST_COLLECTION, &ids).unwrap();
    assert_eq!(results.len(), 10);
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().bytes, entries[i].1.bytes);
    }
}

#[test]
fn test_node_ref_resolved_carries_hash() {
    let id = [0xAAu8; 16];
    let data = Arc::new(NodeData::new(bytes::Bytes::from_static(b"hello")));
    let r#ref = NodeRef::Resolved(id, data.clone());

    assert!(r#ref.is_resolved());
    assert_eq!(r#ref.structural_hash(), &id);
    assert_eq!(r#ref.data().unwrap().bytes, data.bytes);
}

#[test]
fn test_node_ref_lazy() {
    let id = [0xBBu8; 16];
    let r#ref = NodeRef::Lazy(id);

    assert!(!r#ref.is_resolved());
    assert_eq!(r#ref.structural_hash(), &id);
    assert!(r#ref.data().is_none());
}

#[test]
fn test_storage_error_display() {
    let io_err = StorageError::Io(std::io::Error::other("boom"));
    assert_eq!(io_err.to_string(), "I/O error: boom");

    let id = [0x01; 16];
    assert_eq!(
        StorageError::NotFound(id).to_string(),
        format!("node not found: {id:?}")
    );
    assert_eq!(
        StorageError::VerificationFailed(id).to_string(),
        format!("verification failed for node {id:?}")
    );
    assert_eq!(
        StorageError::Corrupt("bad".into()).to_string(),
        "corrupt data: bad"
    );
    assert_eq!(
        StorageError::Internal("oops".into()).to_string(),
        "internal error: oops"
    );
    assert_eq!(
        StorageError::Collision("conflict".into()).to_string(),
        "record collision: conflict"
    );
}

#[test]
fn test_storage_error_source() {
    let io = std::io::Error::other("x");
    assert!(StorageError::Io(io).source().is_some());
    assert!(StorageError::NotFound([0; 16]).source().is_none());
    assert!(StorageError::Corrupt(String::new()).source().is_none());
    assert!(StorageError::Internal(String::new()).source().is_none());
    assert!(StorageError::Collision(String::new()).source().is_none());
}

#[test]
fn test_storage_error_from_io() {
    let io = std::io::Error::other("y");
    let e: StorageError = io.into();
    assert!(matches!(e, StorageError::Io(_)));
}

#[test]
fn test_io_error_from_storage_error_preserves_kinds() {
    let inner = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pb");
    let e: std::io::Error = StorageError::Io(inner).into();
    assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe);

    let corrupted: std::io::Error = StorageError::Corrupt("mbz".into()).into();
    assert_eq!(corrupted.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(corrupted.to_string(), "mbz");

    let not_found: std::io::Error = StorageError::NotFound([0; 16]).into();
    assert_eq!(not_found.kind(), std::io::ErrorKind::NotFound);

    let verification_failed: std::io::Error = StorageError::VerificationFailed([0; 16]).into();
    assert_eq!(verification_failed.kind(), std::io::ErrorKind::InvalidData);

    let internal: std::io::Error = StorageError::Internal("mbz".into()).into();
    assert_eq!(internal.kind(), std::io::ErrorKind::Other);

    let collision: std::io::Error = StorageError::Collision("dup".into()).into();
    assert_eq!(collision.kind(), std::io::ErrorKind::AlreadyExists);

    // An expired pinned snapshot is neither "absent" nor "retry", and keeps its
    // typed error so a caller can still recover the generations.
    let stale: std::io::Error = StorageError::StaleGeneration {
        generation: 1,
        current: Some(3),
    }
    .into();
    assert_eq!(stale.kind(), std::io::ErrorKind::Other);
    assert!(matches!(
        stale
            .get_ref()
            .and_then(|source| source.downcast_ref::<StorageError>()),
        Some(StorageError::StaleGeneration {
            generation: 1,
            current: Some(3)
        })
    ));
    assert!(StorageError::StaleGeneration {
        generation: 1,
        current: None
    }
    .is_stale_generation());
}

#[test]
fn test_in_memory_default() {
    let store = InMemoryStorage::default();
    assert!(store.get(&TEST_COLLECTION, &[0; 16]).unwrap().is_none());
}

#[test]
fn test_in_memory_delete_and_sync() {
    let store = InMemoryStorage::new();
    let id = [0x42u8; 16];
    store
        .put(
            &TEST_COLLECTION,
            &id,
            &NodeData::new(bytes::Bytes::from_static(b"d")),
        )
        .unwrap();
    assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
    store.delete_collection(&TEST_COLLECTION).unwrap();
    assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_none());
    store.sync().unwrap();
}

#[test]
#[allow(clippy::too_many_lines)]
fn established_collection_lifecycle_and_validation() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        MEMBER_NAMESPACE_INTL,
    };

    let store = InMemoryStorage::new();
    let canonical_id = b"sys:test-col";
    let col_id = derive_collection_id(Some(MEMBER_NAMESPACE_INTL), canonical_id);

    let valid_meta = CollectionMetadata {
        member_namespace: Some(MEMBER_NAMESPACE_INTL),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("system_auxiliary".to_owned()),
        schema: None,
    };

    // 1. Rejects metadata if it does not reproduce collection id
    let wrong_col_id = [0x99; 16];
    assert!(store
        .create_or_put_established(&wrong_col_id, &valid_meta, &[])
        .is_err());

    // 2. Rejects metadata if canonical id is empty
    let empty_canonical_meta = CollectionMetadata {
        collection_canonical_id: Vec::new(),
        ..valid_meta.clone()
    };
    assert!(store
        .create_or_put_established(&col_id, &empty_canonical_meta, &[])
        .is_err());

    // 3. Rejects application records containing COLLECTION_METADATA_RECORD_ID
    let bad_rec = (
        COLLECTION_METADATA_RECORD_ID,
        NodeData::new(bytes::Bytes::from_static(b"bad")),
    );
    assert!(store
        .create_or_put_established(&col_id, &valid_meta, &[bad_rec])
        .is_err());

    // 4. Rejects duplicate node IDs within a single batch
    let node_a = [0x01; 16];
    let dup_rec1 = (node_a, NodeData::new(bytes::Bytes::from_static(b"data1")));
    let dup_rec2 = (node_a, NodeData::new(bytes::Bytes::from_static(b"data2")));
    assert!(store
        .create_or_put_established(&col_id, &valid_meta, &[dup_rec1.clone(), dup_rec2])
        .is_err());

    // 5. Successful atomic establishment with records
    let node_b = [0x02; 16];
    let rec_b = (node_b, NodeData::new(bytes::Bytes::from_static(b"datab")));
    store
        .create_or_put_established(&col_id, &valid_meta, &[dup_rec1.clone(), rec_b.clone()])
        .unwrap();

    // Stored metadata matches
    let stored_meta = store.get_collection_metadata(&col_id).unwrap().unwrap();
    assert_eq!(stored_meta, valid_meta);
    assert_eq!(
        store.get(&col_id, &node_a).unwrap().unwrap().bytes,
        b"data1"[..]
    );
    assert_eq!(
        store.get(&col_id, &node_b).unwrap().unwrap().bytes,
        b"datab"[..]
    );

    // 6. Conflicting metadata on existing collection fails closed
    let conflicting_meta = CollectionMetadata {
        role: Some("other_role".to_owned()),
        ..valid_meta.clone()
    };
    assert!(store
        .create_or_put_established(&col_id, &conflicting_meta, &[])
        .is_err());

    // 7. Idempotent success on matching record
    store
        .create_or_put_established(&col_id, &valid_meta, &[dup_rec1])
        .unwrap();

    // 8. Collision error on different payload for existing key
    let collision_rec = (
        node_a,
        NodeData::new(bytes::Bytes::from_static(b"collision")),
    );
    let err = store
        .create_or_put_established(&col_id, &valid_meta, std::slice::from_ref(&collision_rec))
        .unwrap_err();
    assert!(matches!(err, StorageError::Collision(_)));
    let batch_new_id = [0x04; 16];
    let batch_new = (
        batch_new_id,
        NodeData::new(bytes::Bytes::from_static(b"new")),
    );
    assert!(matches!(
        store.create_or_put_established(
            &col_id,
            &valid_meta,
            &[batch_new.clone(), collision_rec.clone()]
        ),
        Err(StorageError::Collision(_))
    ));
    assert!(store.get(&col_id, &batch_new_id).unwrap().is_none());

    // 9. put_many_established on existing collection
    let node_c = [0x03; 16];
    let rec_c = (node_c, NodeData::new(bytes::Bytes::from_static(b"datac")));
    store.put_many_established(&col_id, &[rec_c]).unwrap();
    assert_eq!(
        store.get(&col_id, &node_c).unwrap().unwrap().bytes,
        b"datac"[..]
    );

    // 10. put_many_established fails on payload collision
    let err = store
        .put_many_established(&col_id, std::slice::from_ref(&collision_rec))
        .unwrap_err();
    assert!(matches!(err, StorageError::Collision(_)));
    let batch_new_id = [0x05; 16];
    let batch_new = (
        batch_new_id,
        NodeData::new(bytes::Bytes::from_static(b"new")),
    );
    assert!(matches!(
        store.put_many_established(&col_id, &[batch_new, collision_rec]),
        Err(StorageError::Collision(_))
    ));
    assert!(store.get(&col_id, &batch_new_id).unwrap().is_none());

    // 11. put_many_established fails on non-existent collection
    assert!(store
        .put_many_established(
            &[0xee; 16],
            &[(node_a, NodeData::new(bytes::Bytes::from_static(b"data")))]
        )
        .is_err());

    // 12. Corrupted stored metadata detection: stored canonical id does not reproduce collection id
    let meta_colliding = CollectionMetadata {
        collection_canonical_id: b"sys:other-room".to_vec(),
        ..valid_meta.clone()
    };
    let target_col_id = derive_collection_id(
        meta_colliding.member_namespace,
        &meta_colliding.collection_canonical_id,
    );
    let sim_store = InMemoryStorage::new();
    // Existing collection was established with valid_meta (which does not reproduce target_col_id)
    sim_store
        .collections
        .write()
        .entry(target_col_id)
        .or_default()
        .insert(
            COLLECTION_METADATA_RECORD_ID,
            NodeData::new(valid_meta.encode().into()),
        );
    let err = sim_store
        .create_or_put_established(&target_col_id, &meta_colliding, &[])
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));

    // 13. Unknown member namespace rejected on establishment
    let meta_unknown = CollectionMetadata {
        member_namespace: Some(*b"EDGE"),
        ..valid_meta.clone()
    };
    let err_unknown = sim_store
        .create_or_put_established(&target_col_id, &meta_unknown, &[])
        .unwrap_err();
    assert!(matches!(err_unknown, StorageError::Internal(_)));
}

#[test]
#[allow(clippy::too_many_lines)]
fn create_or_upsert_established_validated_contract() {
    use crate::template::{
        derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        MEMBER_NAMESPACE_INTL,
    };

    let store = InMemoryStorage::new();
    let canonical_id = b"sys:validated-upsert";
    let col_id = derive_collection_id(Some(MEMBER_NAMESPACE_INTL), canonical_id);

    let valid_meta = CollectionMetadata {
        member_namespace: Some(MEMBER_NAMESPACE_INTL),
        collection_canonical_id: canonical_id.to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("system_auxiliary".to_owned()),
        schema: None,
    };

    let node_id = [0x11; 16];
    let data1 = NodeData::new(bytes::Bytes::from_static(b"value_1"));

    // 1. Rejects policy != Key
    let mut non_key_meta = valid_meta.clone();
    non_key_meta.record_id_rule.policy = FrameIdPolicy::Pointer {
        pointer: "/event_id".into(),
    };
    let err = store
        .create_or_upsert_established_validated(
            &col_id,
            &non_key_meta,
            &node_id,
            &data1,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));

    // 2. Rejects reserved node ID
    let err = store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &COLLECTION_METADATA_RECORD_ID,
            &data1,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));

    // 3. Validation failure on absent collection leaves collection absent!
    let err = store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data1,
            &mut |existing| {
                assert!(existing.is_none());
                Err(StorageError::Internal("aborted by validator".into()))
            },
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));
    assert_eq!(store.get_collection_metadata(&col_id).unwrap(), None);
    assert!(store.get(&col_id, &node_id).unwrap().is_none());
    assert!(!store.collection_exists(&col_id).unwrap());

    // 4. First successful write establishes collection and writes record atomically
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data1,
            &mut |existing| {
                assert!(existing.is_none());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        store.get_collection_metadata(&col_id).unwrap().unwrap(),
        valid_meta
    );
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"value_1"[..]
    );

    // 5. Idempotent retry: same key + same payload -> Ok(())
    let mut callback_ran = false;
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data1,
            &mut |existing| {
                callback_ran = true;
                assert_eq!(existing.unwrap().bytes, b"value_1"[..]);
                Ok(())
            },
        )
        .unwrap();
    assert!(callback_ran);

    // 6. Same key replacement succeeds and updates value
    let data2 = NodeData::new(bytes::Bytes::from_static(b"value_2_updated"));
    store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data2,
            &mut |existing| {
                assert_eq!(existing.unwrap().bytes, b"value_1"[..]);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"value_2_updated"[..]
    );

    // 7. Validation failure on existing record aborts without updating
    let data3 = NodeData::new(bytes::Bytes::from_static(b"value_3_collision"));
    let err = store
        .create_or_upsert_established_validated(
            &col_id,
            &valid_meta,
            &node_id,
            &data3,
            &mut |existing| {
                assert_eq!(existing.unwrap().bytes, b"value_2_updated"[..]);
                Err(StorageError::Collision("key digest mismatch".into()))
            },
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Collision(_)));
    // Value remains uncorrupted:
    assert_eq!(
        store.get(&col_id, &node_id).unwrap().unwrap().bytes,
        b"value_2_updated"[..]
    );

    // 8. Corrupted stored metadata on identity: existing collection metadata does not reproduce collection id
    let meta_colliding = CollectionMetadata {
        collection_canonical_id: b"sys:validated-other".to_vec(),
        ..valid_meta.clone()
    };
    let target_col_id = derive_collection_id(
        meta_colliding.member_namespace,
        &meta_colliding.collection_canonical_id,
    );
    let sim_store = InMemoryStorage::new();
    sim_store
        .collections
        .write()
        .entry(target_col_id)
        .or_default()
        .insert(
            COLLECTION_METADATA_RECORD_ID,
            NodeData::new(valid_meta.encode().into()),
        );
    let err = sim_store
        .create_or_upsert_established_validated(
            &target_col_id,
            &meta_colliding,
            &node_id,
            &data1,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(err, StorageError::Internal(_)));
}

#[test]
fn storage_engine_trait_object_safety() {
    let store = InMemoryStorage::new();
    let trait_ref: &dyn StorageEngine = &store;
    let arc_trait: Arc<dyn StorageEngine> = Arc::new(InMemoryStorage::new());
    assert!(!trait_ref.collection_exists(&[0u8; 16]).unwrap());
    assert!(!arc_trait.collection_exists(&[0u8; 16]).unwrap());
}
