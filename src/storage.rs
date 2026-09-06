use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

/// A node ID is a 16-byte structural hash.
pub type NodeId = [u8; 16];

/// Opaque node data as raw bytes (the encoded HAMT node or PDU).
#[derive(Debug, Clone)]
pub struct NodeData {
    pub bytes: bytes::Bytes,
    pub children: Vec<NodeRef>,
}

impl NodeData {
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
    Lazy(NodeId),
    Resolved(NodeId, Arc<NodeData>),
}

impl NodeRef {
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
    #[must_use]
    pub fn data(&self) -> Option<&Arc<NodeData>> {
        match self {
            Self::Lazy(_) => None,
            Self::Resolved(_, data) => Some(data),
        }
    }
    // jscpd:ignore-end

    #[must_use]
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved(..))
    }
}

/// The storage engine trait. Abstracts the backend so the packfile,
/// index, cache, and frontier code don't depend on a specific engine.
///
/// Every operation is scoped to a single room. The caller always knows
/// which room a node belongs to; the engine uses this to select the
/// correct per-room index and packfile, keeping each room's active index
/// at ~8KB (100 active rooms < 1MB total).
///
/// Implementations:
/// - `PackfileStorage`: the custom append-only packfile with lossy index.
/// - `InMemoryStorage`: for tests.
pub trait StorageEngine: Send + Sync {
    /// Fetch a single node by its structural hash within a room.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError>;

    /// Fetch multiple nodes by their structural hashes within a room.
    /// Returns results in the same order as the input keys.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError>;

    /// Store a new node within a room. The caller must ensure the node
    /// is not already present (content-addressed: identical data produces
    /// identical hash).
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put(&self, room_id: &[u8; 16], id: &NodeId, data: &NodeData) -> Result<(), StorageError>;

    /// Store multiple new nodes in a single batch within a room.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
    fn put_many(
        &self,
        room_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError>;

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
///
/// Partitions nodes by room. Each room's nodes are tracked in a
/// per-room `HashMap`, enabling correct `delete_room` behavior.
pub struct InMemoryStorage {
    rooms: RwLock<HashMap<[u8; 16], HashMap<NodeId, NodeData>>>,
}

impl InMemoryStorage {
    #[must_use]
    pub fn new() -> Self {
        Self {
            rooms: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageEngine for InMemoryStorage {
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let rooms = self.rooms.read();
        Ok(rooms.get(room_id).and_then(|r| r.get(id).cloned()))
    }

    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let rooms = self.rooms.read();
        let room = rooms.get(room_id);
        Ok(ids
            .iter()
            .map(|id| room.and_then(|r| r.get(id).cloned()))
            .collect())
    }

    fn put(&self, room_id: &[u8; 16], id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        self.rooms
            .write()
            .entry(*room_id)
            .or_default()
            .insert(*id, data.clone());
        Ok(())
    }

    fn put_many(
        &self,
        room_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        let mut rooms = self.rooms.write();
        let room = rooms.entry(*room_id).or_default();
        for (id, data) in entries {
            room.insert(*id, data.clone());
        }
        Ok(())
    }

    fn delete_room(&self, room_id: &[u8; 16]) -> Result<(), StorageError> {
        self.rooms.write().remove(room_id);
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;

    const TEST_ROOM: [u8; 16] = [0x01; 16];

    #[test]
    fn test_in_memory_roundtrip() {
        let store = InMemoryStorage::new();
        let id = [0x42u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"test node data"));

        store.put(&TEST_ROOM, &id, &data).unwrap();
        let fetched = store.get(&TEST_ROOM, &id).unwrap().unwrap();
        assert_eq!(fetched.bytes, data.bytes);
    }

    #[test]
    fn test_in_memory_not_found() {
        let store = InMemoryStorage::new();
        assert!(store.get(&TEST_ROOM, &[0x00; 16]).unwrap().is_none());
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

        store.put_many(&TEST_ROOM, &entries).unwrap();

        let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        let results = store.get_many(&TEST_ROOM, &ids).unwrap();
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
}
