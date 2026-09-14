use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// A node ID is a 16-byte structural hash.
pub type NodeId = [u8; 16];

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
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError>;

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
    ) -> Result<(), StorageError> {
        let mut collections = self.collections.write();
        let collection = collections.entry(*collection_id).or_default();
        for (id, data) in entries {
            collection.insert(*id, data.clone());
        }
        Ok(())
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
    fn test_in_memory_batch() {
        let store = InMemoryStorage::new();
        let entries: Vec<(NodeId, NodeData)> = (0..10)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
            })
            .collect();

        store.put_many(&TEST_COLLECTION, &entries).unwrap();

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
