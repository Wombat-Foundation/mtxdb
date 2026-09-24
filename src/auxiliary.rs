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

use std::collections::HashMap;

use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{derive_collection_id, POOL_DST_INTERNAL};

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
    derive_collection_id(Some(POOL_DST_INTERNAL), name.as_bytes())
}

fn physical_id(digest: &AuxiliaryKeyDigest) -> NodeId {
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// A named logical auxiliary index backed by an existing storage engine.
pub struct AuxiliaryIndex<'a, S: StorageEngine + ?Sized> {
    engine: &'a S,
    collection_id: [u8; 16],
}

impl<'a, S: StorageEngine + ?Sized> AuxiliaryIndex<'a, S> {
    /// Open a named auxiliary index in an existing storage engine.
    #[must_use]
    pub fn open(engine: &'a S, name: &str) -> Self {
        Self {
            engine,
            collection_id: auxiliary_collection_id(name),
        }
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

    /// Insert or replace a value by its logical key.
    ///
    /// A collision in the current 16-byte physical compatibility ID is
    /// rejected instead of silently overwriting the existing key. Full
    /// 32-byte identity is retained in the versioned value envelope.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] if the physical compatibility ID is
    /// occupied by another key, or propagates a backend error.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), StorageError> {
        let digest = auxiliary_key_digest(key);
        let node_id = physical_id(&digest);
        if let Some(existing) = self.engine.get(&self.collection_id, &node_id)? {
            if !existing.bytes.is_empty() {
                decode_value(&digest, &existing.bytes)?;
            }
        }
        let capacity = VALUE_MAGIC
            .len()
            .checked_add(DIGEST_LEN)
            .and_then(|length| length.checked_add(value.len()))
            .ok_or_else(|| StorageError::Corrupt("auxiliary value length overflow".to_owned()))?;
        let mut encoded = Vec::with_capacity(capacity);
        encoded.extend_from_slice(VALUE_MAGIC);
        encoded.extend_from_slice(&digest);
        encoded.extend_from_slice(value);
        self.engine.put(
            &self.collection_id,
            &node_id,
            &NodeData::new(bytes::Bytes::from(encoded)),
        )
    }

    /// Insert or replace several values with one read and one write batch.
    ///
    /// Existing records are still checked for full-digest collisions, but
    /// callers avoid one storage round trip per auxiliary entry.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] for malformed existing envelopes or
    /// digest collisions, and propagates storage errors.
    pub fn put_many(&self, entries: &[(&[u8], &[u8])]) -> Result<usize, StorageError> {
        let mut physical: Vec<(NodeId, NodeData)> = Vec::with_capacity(entries.len());
        let mut seen: HashMap<NodeId, AuxiliaryKeyDigest> = HashMap::with_capacity(entries.len());
        for (key, value) in entries {
            let digest = auxiliary_key_digest(key);
            let node_id = physical_id(&digest);
            if let Some(previous) = seen.insert(node_id, digest) {
                if previous != digest {
                    return Err(StorageError::Corrupt(
                        "auxiliary-index key digest collision".to_owned(),
                    ));
                }
            }
            let capacity = VALUE_MAGIC
                .len()
                .checked_add(DIGEST_LEN)
                .and_then(|length| length.checked_add(value.len()))
                .ok_or_else(|| {
                    StorageError::Corrupt("auxiliary value length overflow".to_owned())
                })?;
            let mut encoded = Vec::with_capacity(capacity);
            encoded.extend_from_slice(VALUE_MAGIC);
            encoded.extend_from_slice(&digest);
            encoded.extend_from_slice(value);
            physical.push((node_id, NodeData::new(bytes::Bytes::from(encoded))));
        }

        let ids: Vec<NodeId> = physical.iter().map(|(id, _)| *id).collect();
        let existing = self.engine.get_many(&self.collection_id, &ids)?;
        for ((_, data), existing) in physical.iter().zip(existing) {
            if let Some(existing) = existing {
                // The full digest is retained in the envelope. Validate it
                // before allowing the batch to replace the record.
                let digest_start = VALUE_MAGIC.len();
                let digest_end = digest_start.checked_add(DIGEST_LEN).ok_or_else(|| {
                    StorageError::Corrupt("auxiliary digest length overflow".to_owned())
                })?;
                let digest: AuxiliaryKeyDigest = data
                    .bytes
                    .get(digest_start..digest_end)
                    .and_then(|bytes| bytes.try_into().ok())
                    .ok_or_else(|| {
                        StorageError::Corrupt(
                            "new auxiliary value has an incomplete digest envelope".to_owned(),
                        )
                    })?;
                decode_value(&digest, &existing.bytes)?;
            }
        }
        self.engine.put_many(&self.collection_id, &physical)
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
        return Err(StorageError::Corrupt(
            "auxiliary-index key digest collision".to_owned(),
        ));
    }
    Ok(encoded[value_start..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

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
}
