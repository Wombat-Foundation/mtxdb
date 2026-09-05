use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// A node ID is a 16-byte structural hash.
pub type NodeId = [u8; 16];

/// Opaque node data as raw bytes (the encoded HAMT node or PDU).
#[derive(Debug, Clone)]
pub struct NodeData {
    pub bytes: bytes::Bytes,
}

/// A reference to a node that may be resident in cache or need disk fetch.
///
/// This is the swizzling enum inspired by `LeanStore`:
/// - `Lazy(id)`: the node is on disk, identified by its hash.
/// - `Resolved(data)`: the node is in memory, ready for use.
#[derive(Debug, Clone)]
pub enum NodeRef {
    Lazy(NodeId),
    Resolved(Arc<NodeData>),
}

impl NodeRef {
    #[must_use]
    pub fn structural_hash(&self) -> &NodeId {
        match self {
            NodeRef::Lazy(id) => id,
            NodeRef::Resolved(_data) => {
                // For resolved nodes, the hash is derived from the data.
                // In practice, the caller already knows the hash.
                // This is a placeholder — the real system would store
                // the hash alongside the data.
                unresolved_hash()
            }
        }
    }
}

fn unresolved_hash() -> &'static NodeId {
    static ZERO: NodeId = [0u8; 16];
    &ZERO
}

/// The storage engine trait. Abstracts the backend so the packfile,
/// index, cache, and frontier code don't depend on a specific engine.
///
/// Implementations:
/// - `PackfileStorage`: the custom append-only packfile with lossy index.
/// - `InMemoryStorage`: for tests.
pub trait StorageEngine: Send + Sync {
    /// Fetch a single node by its structural hash.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get(&self, id: &NodeId) -> Result<Option<NodeData>, StorageError>;

    /// Fetch multiple nodes by their structural hashes.
    /// Returns results in the same order as the input keys.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get_many(&self, ids: &[NodeId]) -> Result<Vec<Option<NodeData>>, StorageError>;

    /// Store a new node. The caller must ensure the node is not already
    /// present (content-addressed: identical data produces identical hash).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put(&self, id: &NodeId, data: &NodeData) -> Result<(), StorageError>;

    /// Store multiple new nodes in a single batch.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put_many(&self, entries: &[(NodeId, NodeData)]) -> Result<(), StorageError>;

    /// Delete all nodes for a given room (range delete).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn delete_room(&self, room_id: &[u8; 16]) -> Result<(), StorageError>;

    /// Sync to disk (fsync).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn sync(&self) -> Result<(), StorageError>;
}

#[derive(Debug)]
pub enum StorageError {
    Io(std::io::Error),
    NotFound(NodeId),
    VerificationFailed(NodeId),
    Corrupt(String),
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

/// In-memory storage engine for tests.
pub struct InMemoryStorage {
    nodes: RwLock<HashMap<NodeId, NodeData>>,
}

impl InMemoryStorage {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageEngine for InMemoryStorage {
    fn get(&self, id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        Ok(self.nodes.read().get(id).cloned())
    }

    fn get_many(&self, ids: &[NodeId]) -> Result<Vec<Option<NodeData>>, StorageError> {
        let map = self.nodes.read();
        Ok(ids.iter().map(|id| map.get(id).cloned()).collect())
    }

    fn put(&self, id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        self.nodes.write().insert(*id, data.clone());
        Ok(())
    }

    fn put_many(&self, entries: &[(NodeId, NodeData)]) -> Result<(), StorageError> {
        let mut map = self.nodes.write();
        for (id, data) in entries {
            map.insert(*id, data.clone());
        }
        Ok(())
    }

    fn delete_room(&self, _room_id: &[u8; 16]) -> Result<(), StorageError> {
        // In-memory storage doesn't track rooms; no-op.
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_roundtrip() {
        let store = InMemoryStorage::new();
        let id = [0x42u8; 16];
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"test node data"),
        };

        store.put(&id, &data).unwrap();
        let fetched = store.get(&id).unwrap().unwrap();
        assert_eq!(fetched.bytes, data.bytes);
    }

    #[test]
    fn test_in_memory_not_found() {
        let store = InMemoryStorage::new();
        assert!(store.get(&[0x00; 16]).unwrap().is_none());
    }

    #[test]
    fn test_in_memory_batch() {
        let store = InMemoryStorage::new();
        let entries: Vec<(NodeId, NodeData)> = (0..10)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                (
                    id,
                    NodeData {
                        bytes: bytes::Bytes::from(format!("node {i}")),
                    },
                )
            })
            .collect();

        store.put_many(&entries).unwrap();

        let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        let results = store.get_many(&ids).unwrap();
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_some());
            assert_eq!(result.as_ref().unwrap().bytes, entries[i].1.bytes);
        }
    }
}
