use std::collections::{HashMap, HashSet};

use crate::shard::ShardPool;
use crate::storage::{NodeData, NodeId};

/// A function that resolves a node hash to its data and child hashes.
pub type ResolverFn = dyn Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)>;

/// Result of a repack: `(room_id, hash, shard_id, offset)`.
pub type RepackEntry = ([u8; 16], NodeId, u8, u64);

/// Manages per-room repack lifecycle in the global shard pool model.
///
/// Repack walks the DAG reachable from registered roots and writes
/// live records to the current shard. The caller (`PackfileStorage`)
/// rebuilds its per-room index from the returned entries.
pub struct RepackManager {
    /// Serializes the full repack lifecycle per room.
    room_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
}

use std::sync::Arc;

impl Default for RepackManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RepackManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            room_locks: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a per-room mutex that serializes repack and purge.
    fn room_mutex(&self, room_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.room_locks.lock();
        locks.entry(*room_id).or_default().clone()
    }

    /// Perform a reachability-order repack for a room.
    ///
    /// Walks the DAG reachable from `roots` using the provided resolver,
    /// and writes all reachable nodes into the current shard via
    /// `shard_pool.put_record()`.
    ///
    /// # Arguments
    /// * `room_id` - The room to repack.
    /// * `roots` - Every node that must remain reachable after this repack.
    /// * `resolver` - Function to fetch a node's children given its hash.
    /// * `shard_pool` - The global shard pool to write records into.
    ///
    /// # Returns
    /// The list of `(room_id, hash, shard_id, offset)` entries written,
    /// which the caller can use to rebuild the per-room index.
    ///
    /// # Errors
    /// Returns `RepackError::Io` on I/O failure during shard write.
    pub fn repack_room(
        &self,
        room_id: [u8; 16],
        roots: impl IntoIterator<Item = NodeId>,
        resolver: impl Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)>,
        shard_pool: &ShardPool,
    ) -> Result<Vec<RepackEntry>, RepackError> {
        // Serialize the entire repack lifecycle per room.
        let room_arc = self.room_mutex(&room_id);
        let _room_guard = room_arc.lock();

        // BFS traversal in reachability order, seeded from every root.
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut entries = Vec::new();
        let mut initial_roots = HashSet::new();

        for root in roots {
            if visited.insert(root) {
                queue.push_back(root);
                initial_roots.insert(root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            if let Some((data, children)) = resolver(&hash) {
                let record = crate::packfile::Record {
                    room_id,
                    hash,
                    data: data.bytes.clone(),
                };
                let (shard_id, offset) = shard_pool.put_record(&record).map_err(RepackError::Io)?;
                entries.push((room_id, hash, shard_id, offset));
                for child in children {
                    if visited.insert(child) {
                        queue.push_back(child);
                    }
                }
            } else if initial_roots.contains(&hash) {
                return Err(RepackError::RootNotFound);
            }
        }

        Ok(entries)
    }

    /// Remove a room's lock entry (room purge).
    ///
    /// In the global shard model, records are not deleted individually.
    /// The caller (`PackfileStorage`) clears the room's index, and dead
    /// records are reclaimed by the next repack that rewrites live data.
    pub fn purge_room(&self, room_id: &[u8; 16]) {
        self.room_locks.lock().remove(room_id);
    }
}

#[derive(Debug)]
pub enum RepackError {
    Io(std::io::Error),
    RootNotFound,
    RoomNotFound,
}

impl std::fmt::Display for RepackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::RootNotFound => write!(f, "resolver returned no data for root hash"),
            Self::RoomNotFound => write!(f, "room not found"),
        }
    }
}

impl std::error::Error for RepackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RepackError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::error::Error;
    use std::path::PathBuf;

    fn make_node(id: u8) -> (NodeId, NodeData, Vec<NodeId>) {
        let mut hash = [0u8; 16];
        hash[0] = id;
        let data = NodeData::new(bytes::Bytes::from(format!("node {id}")));
        let children: Vec<NodeId> = (1..=3)
            .filter(|&i| i < id)
            .map(|i| {
                let mut h = [0u8; 16];
                h[0] = i;
                h
            })
            .collect();
        (hash, data, children)
    }

    fn test_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_repack_{name}_{id}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_resolver(
        nodes: HashMap<NodeId, (NodeData, Vec<NodeId>)>,
    ) -> impl Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)> {
        move |hash: &NodeId| nodes.get(hash).map(|(d, c)| (d.clone(), c.clone()))
    }

    #[test]
    fn test_repack_creates_entries() {
        let dir = test_dir("repack");
        let pool = ShardPool::open(dir).unwrap();
        let manager = RepackManager::new();

        let mut root_id = [0u8; 16];
        root_id[0] = 3;
        let mut nodes = HashMap::new();
        for i in 1..=3u8 {
            let (id, data, children) = make_node(i);
            nodes.insert(id, (data, children));
        }

        let entries = manager
            .repack_room([0xAA; 16], [root_id], make_resolver(nodes), &pool)
            .unwrap();
        assert_eq!(entries.len(), 3);

        // Verify entries contain the expected room_id
        for (room, _, _, _) in &entries {
            assert_eq!(*room, [0xAA; 16]);
        }
    }

    #[test]
    fn test_repack_preserves_all_roots() {
        let dir = test_dir("multiroot");
        let pool = ShardPool::open(dir).unwrap();
        let manager = RepackManager::new();

        let mut nodes = HashMap::new();
        for i in 1..=3u8 {
            let (id, data, children) = make_node(i);
            nodes.insert(id, (data, children));
        }
        let mut state_root = [0u8; 16];
        state_root[0] = 3;

        let mut timeline_root = [0u8; 16];
        timeline_root[1] = 200;
        nodes.insert(
            timeline_root,
            (
                NodeData::new(bytes::Bytes::from_static(b"ancient message")),
                vec![],
            ),
        );

        let entries = manager
            .repack_room(
                [0xCC; 16],
                [state_root, timeline_root],
                make_resolver(nodes),
                &pool,
            )
            .unwrap();

        let hashes: HashSet<NodeId> = entries.into_iter().map(|(_, h, _, _)| h).collect();
        assert!(hashes.contains(&state_root));
        assert!(hashes.contains(&timeline_root));
    }

    #[test]
    fn test_purge_removes_room_lock() {
        let manager = RepackManager::new();
        // Create a lock entry
        let _lock = manager.room_mutex(&[0xBB; 16]);
        assert!(manager.room_locks.lock().contains_key(&[0xBB; 16]));

        manager.purge_room(&[0xBB; 16]);
        assert!(!manager.room_locks.lock().contains_key(&[0xBB; 16]));
    }

    #[test]
    fn test_repack_error_display_and_source() {
        let io = std::io::Error::other("disk");
        let e = RepackError::Io(io);
        assert_eq!(e.to_string(), "I/O error: disk");
        assert!(e.source().is_some());

        let e = RepackError::RootNotFound;
        assert_eq!(e.to_string(), "resolver returned no data for root hash");
        assert!(e.source().is_none());

        let e = RepackError::RoomNotFound;
        assert_eq!(e.to_string(), "room not found");
        assert!(e.source().is_none());
    }

    #[test]
    fn test_repack_error_from_io() {
        let io = std::io::Error::other("z");
        let e: RepackError = io.into();
        assert!(matches!(e, RepackError::Io(_)));
    }
}
