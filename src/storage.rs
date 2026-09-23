use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::template::{CollectionMetadata, COLLECTION_METADATA_RECORD_ID};

/// A 128-bit lookup identity for a record within a collection.
///
/// A `NodeId` is an opaque key chosen by the caller — it is not necessarily a
/// hash of [`NodeData::bytes`], and not all `NodeId` values are derived from
/// a digest algorithm. Some are fixed constants (e.g. auxiliary-index
/// sentinels), protocol-level identities (e.g. Matrix event IDs hashed by
/// Synapse's own derivation), or application-defined keys.
///
/// # 256-bit payload digest
///
/// The current `NodeId` is 128-bit and cannot double as a full payload digest.
/// A 256-bit payload digest belongs in pack-record metadata and should be
/// computed by the storage layer on write, not supplied by application callers.
/// Until that metadata field exists, reads verify only that the stored 128-bit
/// lookup ID matches the requested ID.
pub type NodeId = [u8; 16];

/// A 256-bit content digest.
///
/// This is the general-purpose digest type used for full logical identities
/// and content hashes. It is deliberately algorithm-agnostic: which function
/// produced the bytes is recorded separately (see [`DigestAlgorithm`]), so the
/// storage layer is not pinned to SHA-256 and a template or configuration can
/// select SHA-512, BLAKE3, or another 256-bit output without changing record
/// layout.
pub type Digest32 = [u8; 32];

/// The hash function used to produce a [`Digest32`].
///
/// Only [`DigestAlgorithm::Sha256`] is implemented today; the enum exists so
/// the on-disk metadata can name the algorithm per record and so adding a new
/// one is additive rather than a format break. Unknown algorithm ids decoded
/// from disk are preserved as [`DigestAlgorithm::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DigestAlgorithm {
    /// SHA-256 (FIPS 180-4). The default and currently the only implemented
    /// algorithm.
    #[default]
    Sha256,
    /// An algorithm this build does not recognize, preserved so a newer
    /// writer's digests round-trip without being reinterpreted.
    Unknown(u8),
}

impl DigestAlgorithm {
    /// The stable on-disk identifier for this algorithm.
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            Self::Sha256 => 0x01,
            Self::Unknown(id) => id,
        }
    }

    /// Reconstruct an algorithm from its on-disk identifier.
    #[must_use]
    pub fn from_id(id: u8) -> Self {
        match id {
            0x01 => Self::Sha256,
            other => Self::Unknown(other),
        }
    }

    /// Begin a streaming hash.
    ///
    /// Lets callers feed input in parts — a domain separator, a fixed field, a
    /// variable key — without concatenating into a temporary buffer first.
    #[must_use]
    pub fn hasher(self) -> DigestHasher {
        match self {
            Self::Sha256 => {
                use sha2::Digest as _;
                DigestHasher::Sha256(sha2::Sha256::new())
            }
            Self::Unknown(id) => DigestHasher::Unknown(id),
        }
    }

    /// Hash `data`, producing a [`Digest32`].
    ///
    /// # Panics
    /// Panics if called on [`DigestAlgorithm::Unknown`], which has no
    /// implementation in this build.
    #[must_use]
    pub fn digest(self, data: &[u8]) -> Digest32 {
        let mut hasher = self.hasher();
        hasher.update(data);
        hasher.finalize()
    }
}

/// A streaming hasher for a [`DigestAlgorithm`].
///
/// Obtained from [`DigestAlgorithm::hasher`]; feed input with [`Self::update`]
/// and finish with [`Self::finalize`].
pub enum DigestHasher {
    /// SHA-256 state.
    Sha256(sha2::Sha256),
    /// An algorithm this build cannot hash; every operation panics.
    Unknown(u8),
}

impl DigestHasher {
    /// Feed `data` into the hash state.
    ///
    /// # Panics
    /// Panics if this hasher is [`DigestHasher::Unknown`].
    pub fn update(&mut self, data: &[u8]) {
        match self {
            Self::Sha256(hasher) => {
                use sha2::Digest as _;
                hasher.update(data);
            }
            Self::Unknown(id) => {
                panic!("cannot hash with unimplemented digest algorithm id {id:#04x}")
            }
        }
    }

    /// Finish the hash, producing a [`Digest32`].
    ///
    /// # Panics
    /// Panics if this hasher is [`DigestHasher::Unknown`].
    #[must_use]
    pub fn finalize(self) -> Digest32 {
        match self {
            Self::Sha256(hasher) => {
                use sha2::Digest as _;
                hasher.finalize().into()
            }
            Self::Unknown(id) => {
                panic!("cannot hash with unimplemented digest algorithm id {id:#04x}")
            }
        }
    }
}

/// Compute the content digest of `data` under `algorithm`.
///
/// # Panics
/// Panics if `algorithm` is [`DigestAlgorithm::Unknown`].
#[must_use]
pub fn content_digest(algorithm: DigestAlgorithm, data: &[u8]) -> Digest32 {
    algorithm.digest(data)
}

/// Opaque node data as raw bytes (the encoded HAMT node or PDU).
#[derive(Debug, Clone)]
pub struct NodeData {
    /// The raw encoded node bytes.
    pub bytes: bytes::Bytes,
    /// Child references, if this node has been decoded/swizzled.
    pub children: Vec<NodeRef>,
}

impl NodeData {
    /// Wrap raw bytes as node data with no resolved children.
    pub fn new(bytes: bytes::Bytes) -> Self {
        Self {
            bytes,
            children: Vec::new(),
        }
    }
}

/// A reference to a node that may be resident in cache or need disk fetch.
///
/// This is the swizzling enum inspired by `LeanStore`:
/// - `Lazy(id)`: the node is on disk, identified by its hash.
/// - `Resolved(hash, data)`: the node is in memory, ready for use,
///   with its structural hash stored alongside.
#[derive(Debug, Clone)]
pub enum NodeRef {
    /// The node lives on disk, identified by its hash.
    Lazy(NodeId),
    /// The node is resolved in memory, alongside its structural hash.
    Resolved(NodeId, Arc<NodeData>),
}

impl NodeRef {
    /// The structural hash of the referenced node, regardless of residency.
    #[must_use]
    pub fn structural_hash(&self) -> &NodeId {
        match self {
            Self::Lazy(id) | Self::Resolved(id, _) => id,
        }
    }

    // jscpd:ignore-start
    // False-positive match against index/mod.rs's splitmix64 (a test-only
    // hash mixer) — token-shape coincidence, not related logic. See the
    // "why can't it be fixed" discussion: nothing to extract here.
    /// The resolved node data, if this reference is already in memory.
    #[must_use]
    pub fn data(&self) -> Option<&Arc<NodeData>> {
        match self {
            Self::Lazy(_) => None,
            Self::Resolved(_, data) => Some(data),
        }
    }
    // jscpd:ignore-end

    /// Returns `true` if this reference is already resolved in memory.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved(..))
    }
}

/// The storage engine trait. Abstracts the backend so the packfile,
/// index, cache, and frontier code don't depend on a specific engine.
///
/// Every operation is scoped to a single collection. The caller always knows
/// which collection a node belongs to; the engine uses this to select the
/// correct per-collection index and packfile, keeping each collection's active index
/// at ~8KB (100 active collections < 1MB total).
///
/// Implementations:
/// - `PackfileStorage`: the custom append-only packfile with lossy index.
/// - `InMemoryStorage`: for tests.
pub trait StorageEngine: Send + Sync {
    /// Fetch a single node by its structural hash within a collection.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError>;

    /// Fetch multiple nodes by their structural hashes within a collection.
    /// Returns results in the same order as the input keys.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get_many(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError>;

    /// Store a new node within a collection. The caller must ensure the node
    /// is not already present (content-addressed: identical data produces
    /// identical hash).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
    ) -> Result<(), StorageError>;

    /// Store multiple new nodes in a single batch within a collection.
    ///
    /// Returns the number of entries committed. A batch is **all-or-nothing**:
    /// implementations roll back the index changes of a partially-appended
    /// batch, so on error nothing is logically committed and there is no
    /// partial count to report — callers retry the whole batch, which is safe
    /// because the writes are idempotent.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<usize, StorageError>;

    /// Delete all nodes for a given collection (range delete).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn delete_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError>;

    /// Sync to disk (fsync).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn sync(&self) -> Result<(), StorageError>;

    /// Force a re-scan of a collection's shards from disk and atomically
    /// swap in the freshly-built index, picking up records another process
    /// wrote after this one last loaded (or never loaded) the collection.
    ///
    /// Every implementation's in-memory index is private to the process
    /// that built it (see `PackfileStorage`'s per-process `LossyIndex`) --
    /// there is no ambient cross-process invalidation, so a multi-worker
    /// deployment must call this explicitly on a suspected miss to pull in
    /// another worker's writes. Default no-op: engines with no on-disk
    /// shard files of their own to rescan (e.g. `InMemoryStorage`, used
    /// only in single-process tests) have nothing to refresh.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure re-reading shard files.
    fn refresh_collection(&self, _collection_id: &[u8; 16]) -> Result<(), StorageError> {
        Ok(())
    }

    /// Whether `collection_id` currently holds any record (including the
    /// genesis metadata record).
    ///
    /// Defaults to `false`; engines that can answer cheaply override it. Used
    /// by [`Self::ensure_collection_metadata`] to enforce that a protocol
    /// collection's genesis metadata precedes its first application record.
    /// Auxiliary/internal collections that carry no metadata are unaffected,
    /// because they never call `ensure_collection_metadata`.
    fn collection_exists(&self, _collection_id: &[u8; 16]) -> bool {
        false
    }

    /// Write a collection's genesis metadata record if it has none, or verify
    /// that an existing record matches `metadata`. Idempotent: a caller may
    /// invoke it before every batch.
    ///
    /// The record is stored as an ordinary frame under the reserved
    /// [`COLLECTION_METADATA_RECORD_ID`] key, so it rides the same append,
    /// checkpoint, and crash-recovery paths as application records. Callers
    /// must invoke it before the collection's first application record, so the
    /// genesis frame precedes (and is durable with or before) any application
    /// record. The engine enforces this: writing genesis metadata for a
    /// collection that already holds records is rejected.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure, [`StorageError::Corrupt`]
    /// if an existing record cannot be decoded, or [`StorageError::Internal`]
    /// if an existing record decodes but differs from `metadata`, or if the
    /// collection already holds records (genesis metadata written too late).
    fn ensure_collection_metadata(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
    ) -> Result<(), StorageError> {
        if let Some(existing) = self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? {
            let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }
            return Ok(());
        }
        if self.collection_exists(collection_id) {
            return Err(StorageError::Internal(
                "genesis metadata must be written before the collection's first record".to_owned(),
            ));
        }
        self.put(
            collection_id,
            &COLLECTION_METADATA_RECORD_ID,
            &NodeData::new(metadata.encode().into()),
        )
    }

    /// Fetch and decode a collection's genesis metadata record, if present.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure or
    /// [`StorageError::Corrupt`] if a present record cannot be decoded.
    fn get_collection_metadata(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Option<CollectionMetadata>, StorageError> {
        match self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? {
            Some(data) => CollectionMetadata::decode(&data.bytes)
                .map(Some)
                .ok_or_else(|| {
                    StorageError::Corrupt("malformed collection metadata record".to_owned())
                }),
            None => Ok(None),
        }
    }
}

/// Errors returned by [`StorageEngine`] operations.
#[derive(Debug)]
pub enum StorageError {
    /// An underlying I/O operation failed.
    Io(std::io::Error),
    /// The requested node was not found.
    NotFound(NodeId),
    /// The node's content did not match its requested hash.
    VerificationFailed(NodeId),
    /// The stored data is malformed or fails an integrity check.
    Corrupt(String),
    /// An internal invariant was violated.
    Internal(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::NotFound(id) => write!(f, "node not found: {id:?}"),
            Self::VerificationFailed(id) => write!(f, "verification failed for node {id:?}"),
            Self::Corrupt(msg) => write!(f, "corrupt data: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<StorageError> for std::io::Error {
    fn from(error: StorageError) -> Self {
        match error {
            StorageError::Io(error) => error,
            StorageError::NotFound(_) => {
                std::io::Error::new(std::io::ErrorKind::NotFound, error.to_string())
            }
            StorageError::VerificationFailed(_) => {
                std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
            }
            StorageError::Corrupt(message) => {
                std::io::Error::new(std::io::ErrorKind::InvalidData, message)
            }
            StorageError::Internal(message) => std::io::Error::other(message),
        }
    }
}

/// In-memory storage engine for tests.
///
/// Partitions nodes by collection. Each collection's nodes are tracked in a
/// per-collection `HashMap`, enabling correct `delete_collection` behavior.
pub struct InMemoryStorage {
    collections: RwLock<HashMap<[u8; 16], HashMap<NodeId, NodeData>>>,
}

impl InMemoryStorage {
    /// Create an empty in-memory storage engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageEngine for InMemoryStorage {
    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let collections = self.collections.read();
        Ok(collections
            .get(collection_id)
            .and_then(|r| r.get(id).cloned()))
    }

    fn collection_exists(&self, collection_id: &[u8; 16]) -> bool {
        self.collections.read().contains_key(collection_id)
    }

    fn get_many(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let collections = self.collections.read();
        let collection = collections.get(collection_id);
        Ok(ids
            .iter()
            .map(|id| collection.and_then(|r| r.get(id).cloned()))
            .collect())
    }

    fn put(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
    ) -> Result<(), StorageError> {
        self.collections
            .write()
            .entry(*collection_id)
            .or_default()
            .insert(*id, data.clone());
        Ok(())
    }

    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<usize, StorageError> {
        let mut collections = self.collections.write();
        let collection = collections.entry(*collection_id).or_default();
        for (id, data) in entries {
            collection.insert(*id, data.clone());
        }
        Ok(entries.len())
    }

    fn delete_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        self.collections.write().remove(collection_id);
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::error::Error;

    const TEST_COLLECTION: [u8; 16] = [0x01; 16];

    #[test]
    fn digest_algorithm_ids_roundtrip_and_hash() {
        // The default id is stable and round-trips.
        assert_eq!(DigestAlgorithm::Sha256.id(), 0x01);
        assert_eq!(DigestAlgorithm::from_id(0x01), DigestAlgorithm::Sha256);

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
            CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        };

        let store = InMemoryStorage::new();
        let collection = [0x5Au8; 16];
        assert_eq!(
            store.get_collection_metadata(&collection).unwrap(),
            None,
            "a fresh collection has no genesis metadata"
        );

        let metadata = CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: Some(br#"{"ext":"matrix.room","fmt":1,"room_version":"10"}"#.to_vec()),
        };
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
            CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        };

        let metadata = CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
        };

        // Correct order: metadata first, then records.
        let ordered = InMemoryStorage::new();
        let collection = [0x6Bu8; 16];
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
    }

    #[test]
    fn test_storage_error_source() {
        let io = std::io::Error::other("x");
        assert!(StorageError::Io(io).source().is_some());
        assert!(StorageError::NotFound([0; 16]).source().is_none());
        assert!(StorageError::Corrupt(String::new()).source().is_none());
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
}
