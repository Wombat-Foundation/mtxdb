use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::layout::ShardType;
use crate::template::{CollectionMetadata, FrameIdPolicy, COLLECTION_METADATA_RECORD_ID};

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
/// A 256-bit payload digest belongs in pack-record metadata: `put_verified`
/// computes one and checks it on write, but no read path recomputes it, so
/// reads today match only the stored 128-bit lookup ID.
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
/// The on-disk metadata names the algorithm per record. Unknown algorithm ids
/// decoded from disk are preserved as [`DigestAlgorithm::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DigestAlgorithm {
    /// SHA-256 (FIPS 180-4), the default compatibility algorithm.
    #[default]
    Sha256,
    /// BLAKE3 with its standard 256-bit output.
    Blake3,
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
            Self::Blake3 => 0x02,
            Self::Unknown(id) => id,
        }
    }

    /// Reconstruct an algorithm from its on-disk identifier.
    #[must_use]
    pub fn from_id(id: u8) -> Self {
        match id {
            0x01 => Self::Sha256,
            0x02 => Self::Blake3,
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
            Self::Blake3 => DigestHasher::Blake3(blake3::Hasher::new()),
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
///
/// Intended to be used transiently — created, fed, finalized, and dropped
/// within one call. It is sized for the largest algorithm's state (BLAKE3,
/// ~1.9 KiB), so storing many in a `Vec` or a long-lived struct wastes memory;
/// prefer `DigestAlgorithm::digest` or `content_digest` for one-shot hashing.
#[allow(
    clippy::large_enum_variant,
    reason = "the ~1.9 KiB BLAKE3 state is only ever created and consumed \
              transiently on the stack by `digest`/`derive_collection_id`; \
              unboxing trades that single stack bump for not heap-allocating \
              a hasher on every content hash"
)]
pub enum DigestHasher {
    /// SHA-256 state.
    Sha256(sha2::Sha256),
    /// BLAKE3 state.
    Blake3(blake3::Hasher),
    /// An algorithm this build cannot hash; every operation panics.
    Unknown(u8),
}

const _: () = assert!(
    std::mem::size_of::<DigestHasher>() <= 2048,
    "DigestHasher outgrew its documented ~1.9 KiB bound; re-check the \
     `large_enum_variant` allowance above and the stack cost of transient use"
);

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
            Self::Blake3(hasher) => {
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
            Self::Blake3(hasher) => *hasher.finalize().as_bytes(),
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

    /// Wrap a byte slice as node data, copying it.
    ///
    /// Convenience for callers (including integration tests and language
    /// bindings) that have a `&[u8]` rather than a `bytes::Bytes`.
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self::new(bytes::Bytes::copy_from_slice(bytes))
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

    /// Fetch multiple node payloads without retaining the [`NodeData`] wrapper.
    ///
    /// The returned [`bytes::Bytes`] values share the payload allocation from
    /// the underlying read; extracting them does not copy the node bytes.
    /// Implementations inherit this adapter from [`Self::get_many`], so the
    /// storage-specific read and visibility semantics remain identical.
    ///
    /// # Errors
    ///
    /// Propagates the errors returned by [`Self::get_many`].
    fn get_many_bytes(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<bytes::Bytes>>, StorageError> {
        self.get_many(collection_id, ids).map(|records| {
            records
                .into_iter()
                .map(|record| record.map(|data| data.bytes))
                .collect()
        })
    }

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
    /// An empty batch commits nothing and is a no-op: it does not create the
    /// collection, so [`Self::collection_exists`] stays `false` and
    /// [`Self::collection_len`] stays `None` afterward.
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
    /// Defaults to `Ok(false)`; engines that can answer cheaply override it. Used
    /// by [`Self::ensure_collection_metadata`] to enforce that a protocol
    /// collection's genesis metadata precedes its first application record.
    /// Auxiliary/internal collections that carry no metadata are unaffected,
    /// because they never call `ensure_collection_metadata`.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on storage or journal-overlay failure.
    fn collection_exists(&self, _collection_id: &[u8; 16]) -> Result<bool, StorageError> {
        Ok(false)
    }

    /// Number of records currently indexed for `collection_id`, or `None` if
    /// the collection does not exist.
    ///
    /// This counts distinct record ids, not physical frames: overwriting an id
    /// does not change the count. It includes the genesis metadata record
    /// (`COLLECTION_METADATA_RECORD_ID`) written by
    /// [`Self::ensure_collection_metadata`], so a collection with N application
    /// records reports N + 1, and it includes tombstones (a deleted record's
    /// empty payload is still an indexed record). The count reflects what this
    /// handle can see, so a read-only or checkpoint-backed handle can lag the
    /// writer until its index refreshes.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] on I/O failure.
    fn collection_len(&self, collection_id: &[u8; 16]) -> Result<Option<usize>, StorageError>;

    /// Atomically establish a collection or append to it if already established.
    ///
    /// Under the collection's put-mutex:
    /// - Validates common batch inputs:
    ///   - Rejects [`COLLECTION_METADATA_RECORD_ID`] in `records`.
    ///   - Rejects intra-batch duplicate node IDs in `records`.
    ///   - Rejects metadata whose canonical ID is empty.
    ///   - Validates that `metadata.verify_collection_id(collection_id)` is `true`.
    /// - If the collection does not yet exist:
    ///   - Prepends the genesis metadata record ([`COLLECTION_METADATA_RECORD_ID`])
    ///     with encoded `metadata`.
    ///   - Appends `records` in a single atomic batch.
    /// - If the collection already exists:
    ///   - Verifies that stored genesis metadata matches `metadata` (`found == *metadata`).
    ///     Fails closed with [`StorageError::Internal`] if metadata conflicts.
    ///   - Enforces idempotency and collision semantics on `records`:
    ///     - Same node ID + same payload => idempotent success (skipped).
    ///     - Same node ID + different payload => [`StorageError::Collision`] collision failure.
    ///   - Appends any new records in `records`.
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] if inputs are invalid or if metadata
    /// fails verification.
    /// Returns [`StorageError::Collision`] if a record payload collision occurs.
    /// Returns [`StorageError::Corrupt`] if existing stored metadata cannot be decoded.
    /// Returns [`StorageError::Io`] on I/O failure.
    fn create_or_put_established(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError>;

    /// Append records to an already-established collection.
    ///
    /// Locks the collection under its put-mutex, verifies that the collection
    /// exists and has an established genesis metadata record whose derivation reproduces
    /// `collection_id`, and appends the records.
    ///
    /// If an input record is already present:
    /// - If existing payload matches, it is treated as an idempotent retry.
    /// - If existing payload differs, a [`StorageError::Collision`] collision error is returned.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if the collection does not exist or has no metadata record.
    /// Returns [`StorageError::Internal`] if `records` contains [`COLLECTION_METADATA_RECORD_ID`]
    /// or duplicate keys within the batch.
    /// Returns [`StorageError::Collision`] if a record payload collision occurs.
    /// Returns [`StorageError::Corrupt`] if stored genesis metadata is corrupted or fails derivation.
    /// Returns [`StorageError::Io`] on I/O failure.
    fn put_many_established(
        &self,
        collection_id: &[u8; 16],
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError>;

    /// Establish a collection (if absent) or upsert a record (if present),
    /// running a validation closure against any existing record under the engine's
    /// per-collection lock before mutating.
    ///
    /// # Policy
    /// This method requires `metadata.record_id_rule.policy == FrameIdPolicy::Key`.
    /// Immutable content-addressed collections must use [`Self::create_or_put_established`].
    ///
    /// # Safety and Re-entrancy
    /// The `validate` callback runs synchronously while holding the internal
    /// collection mutex. It is inspection-only and **must not** call back into the
    /// engine or block, or deadlock will result.
    ///
    /// # Lifecycle under `put_mutex(collection_id)`:
    /// 1. Verifies `metadata.verify_collection_id(collection_id)`.
    /// 2. Verifies `metadata.record_id_rule.policy == FrameIdPolicy::Key`.
    /// 3. Rejects `node_id == &COLLECTION_METADATA_RECORD_ID`.
    /// 4. If collection is unestablished:
    ///    - Inspects `existing` (which is `None`).
    ///    - Invokes `validate(None)`.
    ///    - If validation fails, aborts immediately; the collection remains absent.
    ///    - Atomically writes genesis metadata and the record in a single journal group.
    /// 5. If collection is established:
    ///    - Verifies `metadata` matches stored genesis record.
    ///    - Fetches `existing` record under `node_id`.
    ///    - Invokes `validate(existing.as_ref())`.
    ///    - If validation fails, aborts with error; no changes are made.
    ///    - Idempotency check: if `existing.bytes == data.bytes`, returns `Ok(())` without appending a frame.
    ///    - Otherwise, atomically appends the replacement record and updates the index.
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] on derivation mismatch, non-Key policy,
    /// or reserved node ID inclusion.
    /// Returns [`StorageError::Collision`] if validation or key identity detects a collision.
    /// Returns [`StorageError::Corrupt`] if existing metadata or framing is corrupt.
    /// Returns [`StorageError::Io`] on write failure.
    fn create_or_upsert_established_validated(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        node_id: &NodeId,
        data: &NodeData,
        validate: &mut dyn FnMut(Option<&NodeData>) -> Result<(), StorageError>,
    ) -> Result<(), StorageError>;

    /// Convenience alias for [`Self::create_or_put_established`] to establish a collection
    /// with an initial batch of application records (or empty).
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Collision`] on truncation collision or record collision,
    /// [`StorageError::Internal`] if existing genesis metadata differs or on derivation mismatch,
    /// and [`StorageError::Io`] on write failures.
    fn create_collection_with_records(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        initial_records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        self.create_or_put_established(collection_id, metadata, initial_records)
    }

    /// Write a collection's genesis metadata record if it has none, or verify
    /// that an existing record matches `metadata`. Idempotent: a caller may
    /// invoke it before every batch.
    ///
    /// The record is stored as an ordinary frame under the reserved
    /// [`COLLECTION_METADATA_RECORD_ID`] key, so it rides the same append,
    /// checkpoint, and crash-recovery paths as application records.
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
        self.create_or_put_established(collection_id, metadata, &[])
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
    /// The operation was blocked by current storage activity and can be retried.
    WouldBlock(String),
    /// The requested operation is not supported by this storage configuration.
    Unsupported(String),
    /// The requested node was not found.
    NotFound(NodeId),
    /// The node's content did not match its requested hash.
    VerificationFailed(NodeId),
    /// The stored data is malformed or fails an integrity check.
    Corrupt(String),
    /// An internal invariant was violated.
    Internal(String),
    /// A record collision occurred: an existing record conflicts with the requested key or content identity.
    Collision(String),
    /// A conditional commit was rejected because a collection's logical version
    /// no longer matches the value the caller staged. Retryable: re-read the
    /// collection through a versioned read and re-apply the update.
    StaleRead {
        /// Pool holding the collection.
        pool: ShardType,
        /// Collection whose logical version did not match.
        collection_id: [u8; 16],
        /// Logical version the caller expected.
        expected: u64,
        /// Logical version observed when the commit was validated.
        actual: u64,
    },
}

impl StorageError {
    /// Whether this error represents a retryable operation blocked by current
    /// storage activity.
    #[must_use]
    pub fn is_would_block(&self) -> bool {
        matches!(self, Self::WouldBlock(_))
    }

    /// Whether this storage configuration does not support the requested
    /// operation.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    /// Whether this error is a retryable collection-version conflict from a
    /// conditional-commit precondition.
    #[must_use]
    pub fn is_stale_read(&self) -> bool {
        matches!(self, Self::StaleRead { .. })
    }
}

pub(crate) fn collect_missing_established_records<F>(
    records: &[(NodeId, NodeData)],
    mut lookup: F,
) -> Result<Vec<(NodeId, NodeData)>, StorageError>
where
    F: FnMut(&NodeId) -> Result<Option<NodeData>, StorageError>,
{
    let mut missing = Vec::with_capacity(records.len());
    for (id, data) in records {
        if let Some(existing) = lookup(id)? {
            if existing.bytes != data.bytes {
                return Err(StorageError::Collision(format!(
                    "record collision on node {}",
                    hex16(id)
                )));
            }
        } else {
            missing.push((*id, data.clone()));
        }
    }
    Ok(missing)
}

pub(crate) fn validate_established_upsert_inputs(
    collection_id: &NodeId,
    metadata: &CollectionMetadata,
    node_id: &NodeId,
) -> Result<(), StorageError> {
    if node_id == &COLLECTION_METADATA_RECORD_ID {
        return Err(StorageError::Internal(
            "cannot write application record to reserved collection metadata node id".to_owned(),
        ));
    }
    if !metadata.verify_collection_id(collection_id) {
        return Err(StorageError::Internal(
            "collection metadata does not reproduce collection id".to_owned(),
        ));
    }
    if metadata.collection_canonical_id.is_empty() {
        return Err(StorageError::Internal(
            "collection metadata has empty canonical id".to_owned(),
        ));
    }
    if metadata.record_id_rule.policy != FrameIdPolicy::Key {
        return Err(StorageError::Internal(
            "create_or_upsert_established_validated requires FrameIdPolicy::Key".to_owned(),
        ));
    }
    Ok(())
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::WouldBlock(msg) => write!(f, "operation blocked: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::NotFound(id) => write!(f, "node not found: {id:?}"),
            Self::VerificationFailed(id) => write!(f, "verification failed for node {id:?}"),
            Self::Corrupt(msg) => write!(f, "corrupt data: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
            Self::Collision(msg) => write!(f, "record collision: {msg}"),
            Self::StaleRead {
                pool,
                collection_id,
                expected,
                actual,
            } => write!(
                f,
                "stale read on {pool:?} collection {collection_id:?}: expected version {expected}, found {actual}"
            ),
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
            StorageError::WouldBlock(message) => {
                std::io::Error::new(std::io::ErrorKind::WouldBlock, message)
            }
            StorageError::Unsupported(message) => {
                std::io::Error::new(std::io::ErrorKind::Unsupported, message)
            }
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
            StorageError::Collision(message) => {
                std::io::Error::new(std::io::ErrorKind::AlreadyExists, message)
            }
            StorageError::StaleRead { .. } => {
                std::io::Error::new(std::io::ErrorKind::WouldBlock, error.to_string())
            }
        }
    }
}

/// In-memory storage engine for tests.
///
pub(crate) fn hex16(bytes: &[u8; 16]) -> String {
    let mut out = String::with_capacity(32);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

pub(crate) fn validate_established_batch_inputs(
    records: &[(NodeId, NodeData)],
) -> Result<(), StorageError> {
    if records.len() > 1 {
        let mut seen = HashSet::with_capacity(records.len());
        for (id, _) in records {
            if *id == COLLECTION_METADATA_RECORD_ID {
                return Err(StorageError::Internal(
                    "cannot write reserved metadata record id as application record".to_owned(),
                ));
            }
            if !seen.insert(*id) {
                return Err(StorageError::Internal(
                    "duplicate node ID within batch".to_owned(),
                ));
            }
        }
    } else if let Some((id, _)) = records.first() {
        if *id == COLLECTION_METADATA_RECORD_ID {
            return Err(StorageError::Internal(
                "cannot write reserved metadata record id as application record".to_owned(),
            ));
        }
    }
    Ok(())
}

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

    fn collection_exists(&self, collection_id: &[u8; 16]) -> Result<bool, StorageError> {
        Ok(self.collections.read().contains_key(collection_id))
    }

    fn collection_len(&self, collection_id: &[u8; 16]) -> Result<Option<usize>, StorageError> {
        Ok(self
            .collections
            .read()
            .get(collection_id)
            .map(std::collections::HashMap::len))
    }

    fn create_or_put_established(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        validate_established_batch_inputs(records)?;
        if !metadata.verify_collection_id(collection_id) {
            return Err(StorageError::Internal(
                "collection metadata does not reproduce collection id".to_owned(),
            ));
        }
        if metadata.collection_canonical_id.is_empty() {
            return Err(StorageError::Internal(
                "collection metadata has empty canonical id".to_owned(),
            ));
        }

        let mut collections = self.collections.write();
        let collection = collections.entry(*collection_id).or_default();
        if let Some(existing) = collection.get(&COLLECTION_METADATA_RECORD_ID) {
            let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                found.validate_identity_collision(metadata, collection_id)?;
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }
            for (id, data) in records {
                if collection
                    .get(id)
                    .is_some_and(|existing_rec| existing_rec.bytes != data.bytes)
                {
                    return Err(StorageError::Collision(format!(
                        "record collision on node {}",
                        hex16(id)
                    )));
                }
            }
            for (id, data) in records {
                collection.entry(*id).or_insert_with(|| data.clone());
            }
        } else {
            if !collection.is_empty() {
                return Err(StorageError::Internal(
                    "genesis metadata must be written before the collection's first record"
                        .to_owned(),
                ));
            }
            collection.insert(
                COLLECTION_METADATA_RECORD_ID,
                NodeData::new(metadata.encode().into()),
            );
            for (id, data) in records {
                collection.insert(*id, data.clone());
            }
        }
        Ok(())
    }

    fn put_many_established(
        &self,
        collection_id: &[u8; 16],
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        validate_established_batch_inputs(records)?;
        if records.is_empty() {
            return Ok(());
        }
        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(collection_id)
            .ok_or(StorageError::NotFound(*collection_id))?;
        let existing = collection
            .get(&COLLECTION_METADATA_RECORD_ID)
            .ok_or(StorageError::NotFound(*collection_id))?;
        let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
            StorageError::Corrupt("malformed collection metadata record".to_owned())
        })?;
        if !found.verify_collection_id(collection_id) {
            return Err(StorageError::Corrupt(
                "stored collection metadata does not reproduce collection id".to_owned(),
            ));
        }

        for (id, data) in
            collect_missing_established_records(records, |id| Ok(collection.get(id).cloned()))?
        {
            collection.insert(id, data);
        }
        Ok(())
    }

    fn create_or_upsert_established_validated(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        node_id: &NodeId,
        data: &NodeData,
        validate: &mut dyn FnMut(Option<&NodeData>) -> Result<(), StorageError>,
    ) -> Result<(), StorageError> {
        validate_established_upsert_inputs(collection_id, metadata, node_id)?;

        let mut collections = self.collections.write();
        let exists = collections
            .get(collection_id)
            .and_then(|c| c.get(&COLLECTION_METADATA_RECORD_ID))
            .is_some();

        if exists {
            let collection = collections.get_mut(collection_id).expect("checked above");
            let existing_meta_rec = collection
                .get(&COLLECTION_METADATA_RECORD_ID)
                .expect("checked above");
            let found = CollectionMetadata::decode(&existing_meta_rec.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                found.validate_identity_collision(metadata, collection_id)?;
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }

            let existing = collection.get(node_id);
            validate(existing)?;

            if let Some(existing) = existing {
                if existing.bytes == data.bytes {
                    return Ok(());
                }
            }
            collection.insert(*node_id, data.clone());
        } else {
            if let Some(c) = collections.get(collection_id) {
                if !c.is_empty() {
                    return Err(StorageError::Internal(
                        "genesis metadata must be written before the collection's first record"
                            .to_owned(),
                    ));
                }
            }

            validate(None)?;

            let collection = collections.entry(*collection_id).or_default();
            collection.insert(
                COLLECTION_METADATA_RECORD_ID,
                NodeData::new(metadata.encode().into()),
            );
            collection.insert(*node_id, data.clone());
        }

        Ok(())
    }

    fn ensure_collection_metadata(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
    ) -> Result<(), StorageError> {
        self.create_or_put_established(collection_id, metadata, &[])
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
        // An empty batch writes nothing, so it must not create an empty
        // collection entry: `collection_exists`/`collection_len` must report
        // the collection as absent, matching `PackfileStorage`.
        if entries.is_empty() {
            return Ok(0);
        }
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
/// Shared body of the `collection_len_counts_genesis_and_distinct_ids` test
/// that every backend runs. Both the in-memory and the packfile suite call
/// this against their own store, so the assertions live in one place instead
/// of drifting copies.
pub(crate) fn assert_collection_len_counts_genesis_and_distinct_ids(store: &dyn StorageEngine) {
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
    assert_eq!(store.collection_len(&collection).unwrap(), None);

    store
        .ensure_collection_metadata(&collection, &metadata)
        .unwrap();
    assert_eq!(store.collection_len(&collection).unwrap(), Some(1));

    let a = [0xA1u8; 16];
    let b = [0xB2u8; 16];
    store
        .put(
            &collection,
            &a,
            &NodeData::new(bytes::Bytes::from_static(b"one")),
        )
        .unwrap();
    store
        .put(
            &collection,
            &b,
            &NodeData::new(bytes::Bytes::from_static(b"two")),
        )
        .unwrap();
    assert_eq!(store.collection_len(&collection).unwrap(), Some(3));

    // Overwriting an id does not add a record.
    store
        .put(
            &collection,
            &a,
            &NodeData::new(bytes::Bytes::from_static(b"uno")),
        )
        .unwrap();
    assert_eq!(store.collection_len(&collection).unwrap(), Some(3));
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_storage.rs"]
mod tests;
