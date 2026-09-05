use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::packfile::{self, PackGeneration, Record};
use crate::storage::{NodeData, NodeId};

/// A function that resolves a node hash to its data and child hashes.
pub type ResolverFn = dyn Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)>;

/// A per-room repacked packfile.
///
/// During idle periods, a background thread walks the current root of
/// a room and rewrites its reachable closure into a new pack in
/// traversal order. This is `git gc` for Matrix state nodes.
///
/// Benefits:
/// 1. Full materialization becomes a sequential scan instead of N random seeks.
/// 2. GC for free: unreachable nodes are dropped by omission.
/// 3. No refcount table, no branching hazard.
///
/// Safety:
/// - Pack files are immutable once written.
/// - Readers hold `Arc<PackGeneration>` for the duration of a traversal.
/// - The repacker writes a new pack, fsyncs, renames atomically, then
///   swaps the room's pointer via `ArcSwap`.
/// - The old pack is unlinked when the last reader releases its Arc.
pub struct RepackManager {
    /// Per-room pack generations. Room ID → current pack generation.
    pub(crate) packs: RwLock<HashMap<[u8; 16], Arc<PackGeneration>>>,
    /// Base directory for pack files.
    base_dir: PathBuf,
}

impl RepackManager {
    #[must_use]
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            packs: RwLock::new(HashMap::new()),
            base_dir,
        }
    }

    /// Get the current pack generation for a room.
    pub fn get_pack(&self, room_id: &[u8; 16]) -> Option<Arc<PackGeneration>> {
        self.packs.read().get(room_id).cloned()
    }

    /// Swap in a new pack generation for a room.
    /// Returns the old generation (readers holding it keep the old file alive).
    pub fn swap_pack(
        &self,
        room_id: [u8; 16],
        new_gen: Arc<PackGeneration>,
    ) -> Option<Arc<PackGeneration>> {
        self.packs.write().insert(room_id, new_gen)
    }

    /// Remove a room's pack from the index (room purge).
    pub fn remove_room(&self, room_id: &[u8; 16]) -> Option<Arc<PackGeneration>> {
        self.packs.write().remove(room_id)
    }

    /// Perform a reachability-order repack for a room.
    ///
    /// This walks the reachable DAG from `root_hash` using the provided
    /// resolver, and writes all reachable nodes into a new packfile
    /// in BFS traversal order.
    ///
    /// # Arguments
    /// * `room_id` - The room to repack.
    /// * `root_hash` - The root node to start the walk from.
    /// * `resolver` - Function to fetch a node's children given its hash.
    ///   Returns `(node_data, child_hashes)`.
    ///
    /// # Errors
    /// Returns `RepackError::Io` on I/O failure during packfile write.
    pub fn repack_room(
        &self,
        room_id: [u8; 16],
        root_hash: NodeId,
        resolver: impl Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)>,
    ) -> Result<Arc<PackGeneration>, RepackError> {
        let pack_id = self.next_pack_id(&room_id);
        let pack_path = self.pack_path(&room_id, pack_id);

        // BFS traversal in reachability order
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut records = Vec::new();

        queue.push_back(root_hash);
        visited.insert(root_hash);

        while let Some(hash) = queue.pop_front() {
            if let Some((data, children)) = resolver(&hash) {
                records.push(Record {
                    hash,
                    data: data.bytes.clone(),
                });
                for child in children {
                    if visited.insert(child) {
                        queue.push_back(child);
                    }
                }
            }
        }

        // Write the new packfile atomically: .tmp → fsync → rename
        let tmp_path = pack_path.with_extension("pack.tmp");
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            packfile::write_header(&mut writer)?;
            for record in &records {
                packfile::write_record(&mut writer, record)?;
            }
            writer.flush()?;
            let file = writer.into_inner().map_err(std::io::Error::other)?;
            file.sync_all()?;
        }

        // fsync parent directory
        if let Some(parent) = tmp_path.parent() {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }

        // Atomic rename
        fs::rename(&tmp_path, &pack_path)?;

        // Open the new pack
        let file = packfile::open_packfile(&pack_path, false)?;

        let generation = Arc::new(PackGeneration {
            room_id,
            pack_id,
            file,
            path: pack_path,
            mmap: parking_lot::RwLock::new(None),
        });

        self.swap_pack(room_id, generation.clone());
        Ok(generation)
    }

    /// Delete all data for a room.
    ///
    /// # Errors
    /// Always returns `Ok(())`. Provided for API consistency.
    pub fn purge_room(&self, room_id: &[u8; 16]) -> Result<(), RepackError> {
        if let Some(gen) = self.remove_room(room_id) {
            // Drop the Arc — if readers are still active, the file
            // will be unlinked when the last one releases.
            drop(gen);
        }
        Ok(())
    }

    fn next_pack_id(&self, room_id: &[u8; 16]) -> u8 {
        let packs = self.packs.read();
        packs.get(room_id).map_or(0, |g| g.pack_id.wrapping_add(1))
    }

    fn pack_path(&self, room_id: &[u8; 16], pack_id: u8) -> PathBuf {
        let hex: String = room_id
            .iter()
            .fold(String::with_capacity(32), |mut acc, &b| {
                let _ = write!(acc, "{b:02x}");
                acc
            });
        self.base_dir.join(format!("{hex}_{pack_id:02x}.pack"))
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
mod tests {
    use super::*;

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
        let dir = std::env::temp_dir().join(format!("mdb_test_{name}_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_repack_creates_file() {
        let dir = test_dir("repack");
        let manager = RepackManager::new(dir);

        let root_id = [3u8; 16];
        let mut nodes = HashMap::new();
        for i in 1..=3u8 {
            let (id, data, children) = make_node(i);
            nodes.insert(id, (data, children));
        }

        let resolver = move |hash: &NodeId| -> Option<(NodeData, Vec<NodeId>)> {
            nodes.get(hash).map(|(d, c)| (d.clone(), c.clone()))
        };

        let gen = manager.repack_room([0xAA; 16], root_id, resolver).unwrap();
        assert!(gen.path.exists());
        assert!(manager.get_pack(&[0xAA; 16]).is_some());
    }

    #[test]
    fn test_purge_removes_room() {
        let dir = test_dir("purge");
        let manager = RepackManager::new(dir.clone());

        // Create a dummy pack generation
        let path = dir.join("test.pack");
        let file = packfile::open_packfile(&path, true).unwrap();
        let gen = Arc::new(PackGeneration {
            room_id: [0xBB; 16],
            pack_id: 0,
            file,
            path,
            mmap: parking_lot::RwLock::new(None),
        });
        manager.swap_pack([0xBB; 16], gen);

        assert!(manager.get_pack(&[0xBB; 16]).is_some());
        manager.purge_room(&[0xBB; 16]).unwrap();
        assert!(manager.get_pack(&[0xBB; 16]).is_none());
    }
}
