use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
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
    /// Serializes the full repack and purge lifecycle per room.
    /// A `repack_room` holds this lock across generation selection,
    /// new-pack creation, and swap, preventing concurrent repacks from
    /// reusing the same `pack_id` or path, and preventing a `purge_room`
    /// from deleting a generation that a concurrent repack just published.
    room_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
}

impl RepackManager {
    #[must_use]
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            packs: RwLock::new(HashMap::new()),
            base_dir,
            room_locks: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// Get the current pack generation for a room.
    pub fn get_pack(&self, room_id: &[u8; 16]) -> Option<Arc<PackGeneration>> {
        self.packs.read().get(room_id).cloned()
    }

    /// Swap in a new pack generation for a room.
    /// Returns the old generation (readers holding it keep the old file alive).
    /// Marks the old generation as no longer current so its Drop won't
    /// delete the file (preventing `pack_id` wraparound from unlinking a
    /// newer generation).
    pub fn swap_pack(
        &self,
        room_id: [u8; 16],
        new_gen: Arc<PackGeneration>,
    ) -> Option<Arc<PackGeneration>> {
        let mut packs = self.packs.write();
        let old = packs.insert(room_id, new_gen);
        if let Some(ref old_gen) = old {
            old_gen.is_current.store(false, Ordering::Release);
        }
        old
    }

    /// Remove a room's pack from the index (room purge).
    pub fn remove_room(&self, room_id: &[u8; 16]) -> Option<Arc<PackGeneration>> {
        self.packs.write().remove(room_id)
    }

    /// Get or create a per-room mutex that serializes repack and purge.
    fn room_mutex(&self, room_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.room_locks.lock();
        locks.entry(*room_id).or_default().clone()
    }

    /// Perform a reachability-order repack for a room.
    ///
    /// This walks the DAG reachable from `roots` using the provided
    /// resolver, and writes all reachable nodes into a new packfile in BFS
    /// traversal order.
    ///
    /// # Multiple roots, not one
    ///
    /// A room has more than one thing that must stay reachable: the current
    /// HAMT state root, *and* every current forward-extremity (tip) of the
    /// `prev_events` timeline DAG. Matrix requires a full chronological
    /// ledger — an old message, reaction, or read receipt that isn't bound
    /// into the active state trie is not garbage, it's history that
    /// federation, backfill, and permalinks still need to resolve. A single
    /// root here would mean any repack silently drops everything not
    /// reachable from just the state trie, destroying the timeline on the
    /// first background repack. The caller is responsible for supplying
    /// every root that must survive — this function does not know or care
    /// whether a given root is a state root or a timeline tip, it just
    /// preserves the union of everything reachable from all of them.
    ///
    /// # Arguments
    /// * `room_id` - The room to repack.
    /// * `roots` - Every node that must remain reachable after this repack
    ///   (the current state root plus every timeline forward-extremity, at
    ///   minimum).
    /// * `resolver` - Function to fetch a node's children given its hash.
    ///   Returns `(node_data, child_hashes)`.
    ///
    /// # Errors
    /// Returns `RepackError::Io` on I/O failure during packfile write.
    pub fn repack_room(
        &self,
        room_id: [u8; 16],
        roots: impl IntoIterator<Item = NodeId>,
        resolver: impl Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)>,
    ) -> Result<Arc<PackGeneration>, RepackError> {
        // Serialize the entire repack lifecycle per room: from pack_id
        // selection through swap, no concurrent repack or purge can run.
        let room_arc = self.room_mutex(&room_id);
        let _room_guard = room_arc.lock();

        let pack_id = self.next_pack_id(&room_id);
        let pack_path = self.pack_path(&room_id, pack_id);

        // BFS traversal in reachability order, seeded from every root.
        // Roots are load-bearing — a root the caller explicitly named
        // (the state root, a timeline tip) must resolve, or this repack
        // would silently write a pack missing that entire subtree. A child
        // discovered mid-walk failing to resolve is different: that's the
        // ordinary shape of an incomplete local DAG, and packing what we
        // have is the existing, correct behavior for that case.
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut records = Vec::new();
        let mut initial_roots = HashSet::new();

        for root in roots {
            if visited.insert(root) {
                queue.push_back(root);
                initial_roots.insert(root);
            }
        }

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
            } else if initial_roots.contains(&hash) {
                return Err(RepackError::RootNotFound);
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

        // Atomic rename
        fs::rename(&tmp_path, &pack_path)?;

        // fsync parent directory AFTER rename to ensure the new
        // directory entry is durable before we start using the file.
        if let Some(parent) = pack_path.parent() {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }

        // Open the new pack
        let file = packfile::open_packfile(&pack_path, false)?;

        let generation = Arc::new(PackGeneration {
            room_id,
            pack_id,
            file,
            path: pack_path,
            mmap: parking_lot::RwLock::new(None),
            append_lock: parking_lot::Mutex::new(()),
            is_current: std::sync::atomic::AtomicBool::new(true),
        });

        self.swap_pack(room_id, generation.clone());
        Ok(generation)
    }

    /// Delete all data for a room.
    ///
    /// # Errors
    /// Always returns `Ok(())`. Provided for API consistency.
    pub fn purge_room(&self, room_id: &[u8; 16]) -> Result<(), RepackError> {
        // Serialize with repack_room so we don't delete a generation
        // that a concurrent repack just published.
        let room_arc = self.room_mutex(room_id);
        let _room_guard = room_arc.lock();
        if let Some(gen) = self.remove_room(room_id) {
            drop(gen);
        }
        Ok(())
    }

    fn next_pack_id(&self, room_id: &[u8; 16]) -> u8 {
        let packs = self.packs.read();
        let current = packs.get(room_id).map_or(0, |g| g.pack_id);
        // Skip to the next ID, but check if the target path already exists
        // to avoid overwriting a file from a previous generation that is
        // still referenced by active readers.
        let mut next = current.wrapping_add(1);
        for _ in 0..256 {
            let path = packfile::pack_path(&self.base_dir, room_id, next);
            if !path.exists() {
                return next;
            }
            next = next.wrapping_add(1);
        }
        // All 256 slots occupied — return the current + 1 and let the
        // caller handle the overwrite (this is an extremely unlikely edge
        // case: 256 concurrent repacks for the same room).
        current.wrapping_add(1)
    }

    fn pack_path(&self, room_id: &[u8; 16], pack_id: u8) -> PathBuf {
        packfile::pack_path(&self.base_dir, room_id, pack_id)
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

// jscpd:ignore-start
// False-positive match against benches/storage.rs's DagGenerator::node_data
// — token-shape coincidence, not related logic.
impl std::error::Error for RepackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
// jscpd:ignore-end

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
    use std::path::Path;

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

    /// A resolver over a fixed, pre-built node map — the shape every
    /// `repack_room` test needs, differing only in which map they pass.
    fn make_resolver(
        nodes: HashMap<NodeId, (NodeData, Vec<NodeId>)>,
    ) -> impl Fn(&NodeId) -> Option<(NodeData, Vec<NodeId>)> {
        move |hash: &NodeId| nodes.get(hash).map(|(d, c)| (d.clone(), c.clone()))
    }

    /// Assert that nodes with hash `[i, 0, ..., 0]` for `i` in `1..=3` are
    /// present in the packfile at `path` — the reachable closure every
    /// `make_node`-based test expects to survive a repack.
    fn assert_pack_contains_make_node_ids(path: &Path, context: &str) {
        let entries = packfile::scan_packfile(path).unwrap();
        let hashes: HashSet<NodeId> = entries.into_iter().map(|(h, _)| h).collect();
        for i in 1..=3u8 {
            let mut h = [0u8; 16];
            h[0] = i;
            assert!(
                hashes.contains(&h),
                "{context}: node {i} must be in the repacked pack"
            );
        }
    }

    #[test]
    fn test_repack_creates_file() {
        let dir = test_dir("repack");
        let manager = RepackManager::new(dir);

        // Must match make_node's hash shape (hash[0] = id) — not [3u8; 16],
        // which is a distinct value that never actually resolves via the
        // resolver below. That mismatch was latent before this test started
        // checking record contents: the old repack_room silently wrote an
        // empty pack on an unresolvable root instead of failing.
        let mut root_id = [0u8; 16];
        root_id[0] = 3;
        let mut nodes = HashMap::new();
        for i in 1..=3u8 {
            let (id, data, children) = make_node(i);
            nodes.insert(id, (data, children));
        }

        let gen = manager
            .repack_room([0xAA; 16], [root_id], make_resolver(nodes))
            .unwrap();
        assert!(gen.path.exists());
        assert!(manager.get_pack(&[0xAA; 16]).is_some());

        // Now that the root actually resolves, verify the pack really
        // contains the reachable closure {1, 2, 3} — not just that some
        // file got created.
        assert_pack_contains_make_node_ids(&gen.path, "test_repack_creates_file");
    }

    #[test]
    fn test_repack_preserves_all_roots_not_just_the_first() {
        // Regression test: repack_room must preserve everything reachable
        // from EVERY supplied root, not just the first. Two disjoint trees
        // stand in for "the current HAMT state root" and "an old timeline
        // forward-extremity" — a single-root walk from the state root alone
        // would silently drop the timeline node, exactly the bug that would
        // shred a room's chronological ledger on its first repack.
        let dir = test_dir("repack_multiroot");
        let manager = RepackManager::new(dir);

        let mut nodes = HashMap::new();
        for i in 1..=3u8 {
            let (id, data, children) = make_node(i);
            nodes.insert(id, (data, children));
        }
        let mut state_root = [0u8; 16];
        state_root[0] = 3;

        // A disjoint node — not reachable from state_root by any path —
        // standing in for an old message with no living reference from the
        // active state trie.
        let mut timeline_root = [0u8; 16];
        timeline_root[1] = 200;
        nodes.insert(
            timeline_root,
            (
                NodeData::new(bytes::Bytes::from_static(b"ancient message")),
                vec![],
            ),
        );

        let gen = manager
            .repack_room(
                [0xCC; 16],
                [state_root, timeline_root],
                make_resolver(nodes),
            )
            .unwrap();

        assert_pack_contains_make_node_ids(
            &gen.path,
            "test_repack_preserves_all_roots_not_just_the_first",
        );

        let entries = packfile::scan_packfile(&gen.path).unwrap();
        let hashes: HashSet<NodeId> = entries.into_iter().map(|(h, _)| h).collect();
        assert!(
            hashes.contains(&timeline_root),
            "timeline root disjoint from the state trie must survive repack \
             — a single-root walk would have silently dropped it"
        );
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
            append_lock: parking_lot::Mutex::new(()),
            is_current: std::sync::atomic::AtomicBool::new(true),
        });
        manager.swap_pack([0xBB; 16], gen);

        assert!(manager.get_pack(&[0xBB; 16]).is_some());
        manager.purge_room(&[0xBB; 16]).unwrap();
        assert!(manager.get_pack(&[0xBB; 16]).is_none());
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
