//! Named auxiliary indexes for application-owned lookup tables.
//!
//! Auxiliary indexes are logical stores, not additional packfile pools. They
//! use the normal [`StorageEngine`] so callers do not create one directory or
//! file per room, namespace, or index name.
//!
//! Identity note: auxiliary key digests and collection ids are intentionally
//! BLAKE3 (via [`DigestAlgorithm::Blake3`] and [`derive_collection_id`]). These
//! indexes are core-internal and unreleased, so there is no persisted SHA-256
//! data to remain compatible with and no versioned dual-read or migration is
//! required. Reverting these derivations to SHA-256 would be a regression, not
//! a compatibility fix.

use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    MEMBER_NAMESPACE_INTL,
};

const VALUE_MAGIC: &[u8; 4] = b"AUX1";
const DIGEST_LEN: usize = 32;

/// The canonical identity of an auxiliary-index key.
pub type AuxiliaryKeyDigest = [u8; DIGEST_LEN];

/// Derive the canonical full digest for an auxiliary-index key.
#[must_use]
pub fn auxiliary_key_digest(key: &[u8]) -> AuxiliaryKeyDigest {
    DigestAlgorithm::Blake3.digest(key)
}

/// Derive the logical collection identity for a named auxiliary index.
///
/// Auxiliary indexes are core-internal collections, so they use the shared
/// [`derive_collection_id`] with the core-internal pool namespace rather than a
/// private domain string. The returned 16-byte value is the routing id; key
/// identities remain the full 32-byte digests stored in each envelope.
#[must_use]
pub fn auxiliary_collection_id(name: &str) -> [u8; 16] {
    derive_collection_id(Some(MEMBER_NAMESPACE_INTL), name.as_bytes())
}

fn physical_id(digest: &AuxiliaryKeyDigest) -> NodeId {
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// A named logical auxiliary index backed by an existing storage engine.
pub struct AuxiliaryIndex<'a, S: StorageEngine + ?Sized> {
    engine: &'a S,
    name: String,
    collection_id: [u8; 16],
}

impl<'a, S: StorageEngine + ?Sized> AuxiliaryIndex<'a, S> {
    /// Open a named auxiliary index in an existing storage engine.
    #[must_use]
    pub fn open(engine: &'a S, name: &str) -> Self {
        Self {
            engine,
            name: name.to_owned(),
            collection_id: auxiliary_collection_id(name),
        }
    }

    /// Return this index's canonical name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return this index's collection metadata.
    #[must_use]
    pub fn metadata(&self) -> CollectionMetadata {
        CollectionMetadata {
            member_namespace: Some(MEMBER_NAMESPACE_INTL),
            collection_canonical_id: self.name.as_bytes().to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("system_auxiliary".to_owned()),
            schema: None,
        }
    }

    /// Ensure that collection metadata is established for this auxiliary index.
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] if existing metadata conflicts, or propagates backend error.
    pub fn ensure_metadata(&self) -> Result<(), StorageError> {
        self.engine
            .create_or_put_established(&self.collection_id, &self.metadata(), &[])
    }

    /// Return this index's compatibility collection ID.
    #[must_use]
    pub fn collection_id(&self) -> [u8; 16] {
        self.collection_id
    }

    /// Read a value by its logical key.
    ///
    /// Legacy unwrapped values are returned unchanged. New values verify the
    /// full 32-byte digest before returning the application payload.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] for a malformed or mismatched
    /// envelope, or propagates a backend error.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        let digest = auxiliary_key_digest(key);
        let node_id = physical_id(&digest);
        let Some(data) = self.engine.get(&self.collection_id, &node_id)? else {
            return Ok(None);
        };
        if data.bytes.is_empty() {
            return Ok(None);
        }
        decode_value(&digest, &data.bytes).map(Some)
    }

    /// Insert or update a value by its logical key.
    ///
    /// Atomically establishes the collection metadata on the first write
    /// and appends the key-value envelope. Retains idempotency (same key and value succeeds),
    /// permits same-key replacement with updated values, and rejects truncated-ID collisions
    /// (distinct key with the same 16-byte physical node ID) with [`StorageError::Collision`].
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] on physical ID key collisions,
    /// [`StorageError::Corrupt`] on corrupted existing records, or propagates backend error.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), StorageError> {
        let digest = auxiliary_key_digest(key);
        let node_id = physical_id(&digest);
        let mut encoded = Vec::with_capacity(VALUE_MAGIC.len() + DIGEST_LEN + value.len());
        encoded.extend_from_slice(VALUE_MAGIC);
        encoded.extend_from_slice(&digest);
        encoded.extend_from_slice(value);
        let data = NodeData::new(bytes::Bytes::from(encoded));

        let mut validate = |existing: Option<&NodeData>| -> Result<(), StorageError> {
            if let Some(existing) = existing {
                if !existing.bytes.is_empty() {
                    decode_value(&digest, &existing.bytes)?;
                }
            }
            Ok(())
        };

        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            &node_id,
            &data,
            &mut validate,
        )
    }
}

fn decode_value(
    expected_digest: &AuxiliaryKeyDigest,
    encoded: &[u8],
) -> Result<Vec<u8>, StorageError> {
    if !encoded.starts_with(VALUE_MAGIC) {
        return Ok(encoded.to_vec());
    }
    let digest_start = VALUE_MAGIC.len();
    let value_start = digest_start.checked_add(DIGEST_LEN).ok_or_else(|| {
        StorageError::Corrupt("auxiliary-index envelope length overflow".to_owned())
    })?;
    if encoded.len() < value_start {
        return Err(StorageError::Corrupt(
            "truncated auxiliary-index value envelope".to_owned(),
        ));
    }
    if encoded[digest_start..value_start] != expected_digest[..] {
        return Err(StorageError::Collision(
            "auxiliary-index key digest collision".to_owned(),
        ));
    }
    Ok(encoded[value_start..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
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
}
