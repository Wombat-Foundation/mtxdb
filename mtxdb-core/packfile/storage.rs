use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::csr::Csr;
use crate::index::LossyIndex;
use crate::packfile::{self, Record};
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{NodeData, NodeId, NodeRef, StorageEngine, StorageError};

/// Callback that rewrites a node's child references given resolved child data,
/// used to inline already-cached children in place of lazy hash pointers.
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[Option<Arc<NodeData>>]) -> NodeData;

/// A live hash set plus the adjacency discovered while determining it, as
/// returned by the reachability repacker's live-set computation.
type AdjacencyResult = (Vec<[u8; 16]>, HashMap<[u8; 16], Vec<[u8; 16]>>);

/// One shard's scanned `(hash, offset)` entries for a single room, as
/// produced by a repack's initial shard scan.
type ScannedShard = (u8, Vec<([u8; 16], u64)>);

/// Immutable snapshot of a room's in-memory state.
///
/// Index and cache are bundled so readers see a consistent triple
/// via a single `ArcSwap::load()` — no three-lock coordination.
#[derive(Clone)]
struct RoomGeneration {
    index: LossyIndex,
    cache: Arc<NodeCache>,
}

/// A content-addressed packfile storage engine backed by a global shard pool.
///
/// All rooms share a small pool of shard files (~4, each ~2GB), keeping
/// file descriptor usage constant regardless of room count.
///
/// Per-room state (index + cache) is bundled in an immutable
/// `RoomGeneration` and swapped atomically via `ArcSwap`:
///
/// - **Reads**: `load_full()` once, see a consistent snapshot.
/// - **Writes**: build new generation under put lock, swap atomically.
/// - **Delete**: drop the generation — cache disappears with it.
pub struct PackfileStorage {
    shards: ShardPool,
    rooms: RwLock<HashMap<[u8; 16], ArcSwap<RoomGeneration>>>,
    pinned: PinnedNodes,
    base_dir: PathBuf,
    swizzle: Option<SwizzleFn>,
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    repack_threshold_entries: AtomicU64,
    cache_capacity: usize,
}

const DEFAULT_REPACK_THRESHOLD_ENTRIES: u64 = 2048;
const DEFAULT_CACHE_CAPACITY: usize = 100_000;

impl PackfileStorage {
    /// Open a packfile storage with default settings.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, DEFAULT_CACHE_CAPACITY, None)
    }

    /// Open a packfile storage with a custom per-room cache capacity.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_cache(
        base_dir: PathBuf,
        cache_capacity: usize,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, cache_capacity, None)
    }

    /// Open a packfile storage with a swizzle callback for in-cache pointer resolution.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_swizzle(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: SwizzleFn,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, cache_capacity, Some(swizzle))
    }

    fn open_with_options(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: Option<SwizzleFn>,
    ) -> Result<Self, std::io::Error> {
        type ShardRecord = (u8, [u8; 16], u64);
        fs::create_dir_all(&base_dir)?;

        let shards = ShardPool::open(base_dir.clone())?;
        let mut rooms: HashMap<[u8; 16], ArcSwap<RoomGeneration>> = HashMap::new();

        // Phase 1: accumulate all records per room across every shard so
        // the index can be sized once for the true total.
        let mut room_entries: HashMap<[u8; 16], Vec<ShardRecord>> = HashMap::new();
        for shard_id in 0..shard::MAX_SHARDS_U8 {
            let path = ShardPool::shard_path(&base_dir, shard_id);
            if !path.exists() {
                continue;
            }
            let entries = match packfile::scan_and_recover_packfile(&path) {
                Ok(e) => e,
                Err(recovery_err) => {
                    eprintln!(
                        "warning: recovery scan failed for shard {shard_id:02x}: {recovery_err}"
                    );
                    match packfile::scan_packfile(&path) {
                        Ok(partial) => {
                            eprintln!("warning: partial scan recovered {} records from shard {shard_id:02x}", partial.len());
                            partial
                        }
                        Err(scan_err) => {
                            eprintln!("warning: shard {shard_id:02x} skipped entirely: {scan_err}");
                            continue;
                        }
                    }
                }
            };
            for (room_id, hash, offset) in entries {
                room_entries
                    .entry(room_id)
                    .or_default()
                    .push((shard_id, hash, offset));
            }
        }

        // P1: load deleted rooms set (persisted to disk)
        let deleted_rooms = Self::load_deleted_rooms(&base_dir);

        // Phase 2: build per-room indexes sized to the true total.
        for (room_id, records) in &room_entries {
            if deleted_rooms.contains(room_id) {
                continue;
            }
            let mut index = LossyIndex::new(records.len().saturating_mul(2).max(16));
            for (shard_id, hash, offset) in records {
                let _ = index.insert(hash, *shard_id, *offset);
            }
            rooms.insert(
                *room_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::new(cache_capacity)),
                }),
            );
        }

        Ok(Self {
            shards,
            rooms: RwLock::new(rooms),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
        })
    }

    fn generation(&self, room_id: &[u8; 16]) -> Option<arc_swap::Guard<Arc<RoomGeneration>>> {
        self.rooms
            .read()
            .get(room_id)
            .map(arc_swap::ArcSwapAny::load)
    }

    /// Room IDs currently known to this engine, sorted for deterministic output.
    pub fn room_ids(&self) -> Vec<[u8; 16]> {
        let mut ids: Vec<[u8; 16]> = self.rooms.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// A room's `(entry count, memory usage in bytes)`, if the room exists.
    pub fn room_index_info(&self, room_id: &[u8; 16]) -> Option<(usize, usize)> {
        self.rooms.read().get(room_id).map(|gen| {
            let g = gen.load();
            (g.index.len(), g.index.memory_usage())
        })
    }

    /// `(room_id, entry count, memory usage in bytes)` for every known room,
    /// sorted by room ID, in a single pass over the room map.
    pub fn room_summaries(&self) -> Vec<([u8; 16], usize, usize)> {
        let mut out: Vec<([u8; 16], usize, usize)> = self
            .rooms
            .read()
            .iter()
            .map(|(id, gen)| {
                let g = gen.load();
                (*id, g.index.len(), g.index.memory_usage())
            })
            .collect();
        out.sort_unstable_by_key(|(id, _, _)| *id);
        out
    }

    /// Set the live roots to preserve for a room on its next repack.
    pub fn set_live_roots(&self, room_id: &[u8; 16], roots: Vec<NodeId>) {
        self.live_roots.write().insert(*room_id, roots);
    }

    /// Set the repack trigger threshold directly, in index entries.
    pub fn set_repack_threshold_entries(&self, entries: u64) {
        self.repack_threshold_entries
            .store(entries, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the repack trigger threshold from an approximate byte budget
    /// (assumes ~200 bytes/entry).
    pub fn set_repack_threshold_bytes(&self, bytes: u64) {
        let entries = bytes / 200;
        self.set_repack_threshold_entries(entries.max(1));
    }

    /// Returns `true` if a room's index has reached the configured repack
    /// threshold.
    ///
    /// Repacking is entirely caller-driven — nothing in this engine polls
    /// this on its own. A background GC worker is expected to call this
    /// periodically and issue `repack_room_rewrite`/`repack_room_topo`
    /// itself; nothing in `mtxdb-core` currently does so.
    #[must_use]
    pub fn needs_repack(&self, room_id: &[u8; 16]) -> bool {
        let count = self
            .room_index_info(room_id)
            .map_or(0, |(len, _)| len as u64);
        count
            >= self
                .repack_threshold_entries
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The room-scoped node cache for `room_id`, or a fresh empty one if the room is unknown.
    pub fn room_cache(&self, room_id: &[u8; 16]) -> Arc<NodeCache> {
        self.generation(room_id).map_or_else(
            || Arc::new(NodeCache::new(self.cache_capacity)),
            |g| g.cache.clone(),
        )
    }

    fn put_mutex(&self, room_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.put_locks.lock();
        locks.entry(*room_id).or_default().clone()
    }

    fn read_at(shard: &Shard, offset: u64) -> Result<Record, StorageError> {
        ShardPool::read_at(shard, offset)
    }

    fn scan_room_records(&self, room_id: &[u8; 16]) -> Vec<ScannedShard> {
        let mut scanned: Vec<ScannedShard> = Vec::new();
        for shard_id in 0..shard::MAX_SHARDS_U8 {
            let path = ShardPool::shard_path(&self.base_dir, shard_id);
            if !path.exists() {
                continue;
            }
            match packfile::scan_packfile(&path) {
                Ok(entries) => {
                    let room_entries: Vec<([u8; 16], u64)> = entries
                        .into_iter()
                        .filter(|(rid, _, _)| rid == room_id)
                        .map(|(_, hash, offset)| (hash, offset))
                        .collect();
                    if !room_entries.is_empty() {
                        scanned.push((shard_id, room_entries));
                    }
                }
                Err(e) => {
                    eprintln!("warning: scan_packfile failed for shard {shard_id:02x}: {e}");
                }
            }
        }
        scanned
    }

    fn build_index(offsets: &[([u8; 16], u8, u64)]) -> LossyIndex {
        let mut index = LossyIndex::new(offsets.len().saturating_mul(2).max(16));
        for (hash, shard_id, offset) in offsets {
            let _ = index.insert(hash, *shard_id, *offset);
        }
        index
    }

    fn deleted_rooms_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("deleted.rooms")
    }

    fn load_deleted_rooms(base_dir: &std::path::Path) -> HashSet<[u8; 16]> {
        let path = Self::deleted_rooms_path(base_dir);
        let Ok(bytes) = fs::read(&path) else {
            return HashSet::new();
        };
        bytes
            .chunks_exact(16)
            .map(|chunk| {
                let mut id = [0u8; 16];
                id.copy_from_slice(chunk);
                id
            })
            .collect()
    }

    fn persist_deleted_room(&self, room_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_rooms_path(&self.base_dir);
        let mut set = Self::load_deleted_rooms(&self.base_dir);
        set.insert(*room_id);
        let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
        fs::write(&path, &bytes).map_err(StorageError::Io)?;
        Ok(())
    }

        fn swap_generation(&self, room_id: &[u8; 16], index: LossyIndex) {
        let cache = self.generation(room_id).map_or_else(
            || Arc::new(NodeCache::new(self.cache_capacity)),
            |gen| gen.cache.clone(),
        );
        let new_gen = Arc::new(RoomGeneration { index, cache });
        self.rooms
            .write()
            .entry(*room_id)
            .or_insert_with(|| {
                ArcSwap::from_pointee(RoomGeneration {
                    index: LossyIndex::new(0),
                    cache: Arc::new(NodeCache::new(self.cache_capacity)),
                })
            })
            .store(new_gen);
    }

    fn rebuild_index(&self, room_id: &[u8; 16]) -> LossyIndex {
        let scanned = self.scan_room_records(room_id);
        let total: usize = scanned.iter().map(|(_, e)| e.len()).sum();
        let mut index = LossyIndex::new(total.saturating_mul(2).max(16));
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                let _ = index.insert(&hash, shard_id, offset);
            }
        }
        index
    }

    /// Fetch a node and return it as a `NodeRef` with swizzled children.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure or CRC mismatch.
    pub fn get_swizzled(
        &self,
        room_id: &[u8; 16],
        id: &NodeId,
        extract_children: impl Fn(&NodeData) -> Vec<NodeId>,
    ) -> Result<Option<NodeRef>, StorageError> {
        let Some(data) = self.get(room_id, id)? else {
            return Ok(None);
        };

        if let Some(swizzle_fn) = self.swizzle {
            let children = extract_children(&data);
            if !children.is_empty() {
                let cached = if let Some(gen) = self.generation(room_id) {
                    gen.cache.resolve_hashes(&children)
                } else {
                    vec![None; children.len()]
                };
                let swizzled = swizzle_fn(&data, &children, &cached);
                if let Some(gen) = self.generation(room_id) {
                    gen.cache.insert(*id, Arc::new(swizzled.clone()));
                }
                return Ok(Some(NodeRef::Resolved(*id, Arc::new(swizzled))));
            }
        }

        Ok(Some(NodeRef::Resolved(*id, Arc::new(data))))
    }

    fn resolve_from_candidates(
        &self,
        gen: Option<&Arc<RoomGeneration>>,
        id: &NodeId,
        candidates: &[(u8, u64)],
    ) -> Result<Option<NodeData>, StorageError> {
        let mut last_err: Option<StorageError> = None;
        for (shard_id, offset) in candidates {
            let Some(shard) = self.shards.get_shard(*shard_id) else {
                continue;
            };

            match Self::read_at(&shard, *offset) {
                Ok(record) => {
                    if record.hash != *id {
                        continue;
                    }

                    let data = NodeData {
                        bytes: record.data,
                        children: Vec::new(),
                    };
                    if let Some(g) = gen {
                        g.cache.insert(*id, Arc::new(data.clone()));
                    }
                    return Ok(Some(data));
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        if let Some(err) = last_err {
            return Err(err);
        }
        Ok(None)
    }

    /// Rewrite every record for a room by scanning all shards.
    /// Full compaction: every record is copied to the active shard.
    /// This does NOT perform GC (orphaned nodes are preserved).
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    pub fn repack_room_rewrite(&self, room_id: &[u8; 16]) -> Result<(), StorageError> {
        let room_arc = self.put_mutex(room_id);
        let _room_guard = room_arc.lock();

        let scanned = self.scan_room_records(room_id);

        let mut new_offsets: Vec<([u8; 16], u8, u64)> = Vec::new();
        for (old_shard_id, entries) in &scanned {
            for (hash, offset) in entries {
                let Some(old_shard) = self.shards.get_shard(*old_shard_id) else {
                    continue;
                };
                let record = Self::read_at(&old_shard, *offset)?;
                let (new_shard_id, new_offset) = self.shards.put_record(&Record {
                    room_id: *room_id,
                    hash: *hash,
                    data: record.data,
                })?;
                new_offsets.push((*hash, new_shard_id, new_offset));
            }
        }

        let index = Self::build_index(&new_offsets);
        self.swap_generation(room_id, index);

        Ok(())
    }

    /// Rewrite every record for a room in topological order.
    ///
    /// `extract_edges` receives a record's hash and raw bytes, and returns the
    /// hashes of this record's outgoing-edge targets (e.g. `prev_events`). The
    /// storage engine stays content-agnostic; the caller owns parsing.
    ///
    /// If `extract_edges` returns no edges for any record, that record
    /// is treated as a root (zero in-degree) in the topological sort.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    pub fn repack_room_topo(
        &self,
        room_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<(), StorageError> {
        let room_arc = self.put_mutex(room_id);
        let _room_guard = room_arc.lock();

        let scanned = self.scan_room_records(room_id);

        let mut all_hashes: Vec<[u8; 16]> = Vec::new();
        let mut hash_to_shard_offset: HashMap<[u8; 16], (u8, u64)> = HashMap::new();
        for (shard_id, entries) in &scanned {
            for (hash, offset) in entries {
                all_hashes.push(*hash);
                hash_to_shard_offset.insert(*hash, (*shard_id, *offset));
            }
        }

        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        for (shard_id, entries) in &scanned {
            for (hash, offset) in entries {
                if let Some(old_shard) = self.shards.get_shard(*shard_id) {
                    let record = Self::read_at(&old_shard, *offset)?;
                    adjacency.insert(*hash, extract_edges(hash, &record.data));
                }
            }
        }

        let csr = Csr::build_from_edges(&all_hashes, &adjacency);
        let topo = csr.topo_order();

        let mut new_offsets: Vec<([u8; 16], u8, u64)> = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let (old_shard_id, old_offset) = hash_to_shard_offset
                .get(hash)
                .expect("all hashes in CSR exist in shard map");
            let Some(old_shard) = self.shards.get_shard(*old_shard_id) else {
                continue;
            };
            let record = Self::read_at(&old_shard, *old_offset)?;
            let (new_shard_id, new_offset) = self.shards.put_record(&Record {
                room_id: *room_id,
                hash: *hash,
                data: record.data,
            })?;
            new_offsets.push((*hash, new_shard_id, new_offset));
        }

        let index = Self::build_index(&new_offsets);
        self.swap_generation(room_id, index);

        Ok(())
    }

    /// BFS outward from `roots` over `extract_edges`, reading only records
    /// actually reached. Returns the sorted, deduplicated live hash set and
    /// the adjacency discovered along the way.
    fn bfs_live_set(
        &self,
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u8, u64)>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut visited: HashSet<[u8; 16]> = HashSet::new();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(&(shard_id, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = self.shards.get_shard(shard_id) else {
                continue;
            };
            let record = Self::read_at(&old_shard, offset)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        let mut live_hashes: Vec<[u8; 16]> = visited.into_iter().collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// Read every scanned record's edges, with no reachability filtering.
    /// Used when a room has no configured live roots, so nothing is known
    /// to be garbage.
    fn scan_full_adjacency(
        &self,
        scanned: &[ScannedShard],
        hash_to_shard_offset: &HashMap<[u8; 16], (u8, u64)>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                if let Some(old_shard) = self.shards.get_shard(*shard_id) {
                    let record = Self::read_at(&old_shard, *offset)?;
                    let edges = extract_edges(hash, &record.data);
                    adjacency.insert(*hash, edges);
                }
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Rewrite only the nodes reachable from a room's live roots, in
    /// topological order — this is the one repack path that actually
    /// performs garbage collection.
    ///
    /// Unlike [`Self::repack_room_rewrite`] and [`Self::repack_room_topo`],
    /// which preserve every record ever written for the room (including
    /// orphaned/unreachable ones — see their docs), this traverses outward
    /// from the room's configured live roots (see [`Self::set_live_roots`])
    /// via `extract_edges`, and drops anything not reached. Only reachable
    /// records are read from disk during the traversal — unreachable data
    /// is never even fetched, not just excluded from the output.
    ///
    /// If no live roots are configured for this room (`set_live_roots` was
    /// never called, or was called with an empty list), nothing is known to
    /// be garbage, so this falls back to preserving every scanned record —
    /// same as `repack_room_topo` — rather than risk deleting live data on
    /// the assumption that "no roots" means "nothing is live".
    ///
    /// Returns `(kept, dropped)`: the number of records written to the new
    /// generation, and the number of scanned records found unreachable and
    /// discarded.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    pub fn repack_room_reachable(
        &self,
        room_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<(usize, usize), StorageError> {
        let room_arc = self.put_mutex(room_id);
        let _room_guard = room_arc.lock();

        let scanned = self.scan_room_records(room_id);

        let scanned_count: usize = scanned.iter().map(|(_, entries)| entries.len()).sum();

        let mut hash_to_shard_offset: HashMap<[u8; 16], (u8, u64)> = HashMap::new();
        for (shard_id, entries) in &scanned {
            for (hash, offset) in entries {
                hash_to_shard_offset.insert(*hash, (*shard_id, *offset));
            }
        }

        let roots = self.live_roots.read().get(room_id).cloned();

        let (live_hashes, adjacency) = match roots {
            Some(roots) if !roots.is_empty() => {
                self.bfs_live_set(&roots, &hash_to_shard_offset, &extract_edges)?
            }
            // No live roots configured: we don't know what's garbage,
            // so preserve everything (same as repack_room_topo).
            _ => self.scan_full_adjacency(&scanned, &hash_to_shard_offset, &extract_edges)?,
        };

        let dropped = scanned_count.saturating_sub(live_hashes.len());

        // Evict garbage-collected hashes from the room's cache. Repack
        // carries the same cache forward into the new generation (below);
        // without this, a stale cache hit could still serve bytes for a
        // hash this pass just decided is unreachable, defeating GC.
        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(room_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();

        let mut new_offsets: Vec<([u8; 16], u8, u64)> = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            let Some(old_shard) = self.shards.get_shard(old_shard_id) else {
                continue;
            };
            let record = Self::read_at(&old_shard, old_offset)?;
            let (new_shard_id, new_offset) = self.shards.put_record(&Record {
                room_id: *room_id,
                hash: *hash,
                data: record.data,
            })?;
            new_offsets.push((*hash, new_shard_id, new_offset));
        }

        let kept = new_offsets.len();

        let mut index = LossyIndex::new(new_offsets.len().saturating_mul(2).max(16));
        for (hash, shard_id, offset) in &new_offsets {
            let _ = index.insert(hash, *shard_id, *offset);
        }

        let cache = self.generation(room_id).map_or_else(
            || Arc::new(NodeCache::new(self.cache_capacity)),
            |gen| gen.cache.clone(),
        );
        let new_gen = Arc::new(RoomGeneration { index, cache });

        self.rooms
            .write()
            .entry(*room_id)
            .or_insert_with(|| {
                ArcSwap::from_pointee(RoomGeneration {
                    index: LossyIndex::new(0),
                    cache: Arc::new(NodeCache::new(self.cache_capacity)),
                })
            })
            .store(new_gen);

        Ok((kept, dropped))
    }
}

impl StorageEngine for PackfileStorage {
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let gen_guard = self.generation(room_id);
        let gen = gen_guard.as_deref();

        let candidates: Vec<(u8, u64)> = match gen {
            Some(g) => g.index.lookup_all(id).collect(),
            None => return Ok(None),
        };

        if candidates.is_empty() {
            return Ok(None);
        }

        if let Some(g) = gen {
            if let Some(data) = g.cache.get(id) {
                return Ok(Some((*data).clone()));
            }
        }

        self.resolve_from_candidates(gen, id, &candidates)
    }

    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

        let gen_guard = self.generation(room_id);
        let gen = gen_guard.as_deref();

        let mut to_fetch: Vec<(usize, Vec<(u8, u64)>)> = Vec::new();
        if let Some(g) = gen {
            for (i, id) in ids.iter().enumerate() {
                let candidates: Vec<(u8, u64)> = g.index.lookup_all(id).collect();
                if !candidates.is_empty() {
                    if let Some(data) = g.cache.get(id) {
                        results[i] = Some((*data).clone());
                    } else {
                        to_fetch.push((i, candidates));
                    }
                }
            }
        }

        to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

        for (i, candidates) in &to_fetch {
            results[*i] = self.resolve_from_candidates(gen, &ids[*i], candidates)?;
        }

        Ok(results)
    }

    fn put(&self, room_id: &[u8; 16], id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        let room_arc = self.put_mutex(room_id);
        let _room_guard = room_arc.lock();

        let record = Record {
            room_id: *room_id,
            hash: *id,
            data: data.bytes.clone(),
        };

        let (shard_id, offset) = self.shards.put_record(&record)?;

        let new_gen = {
            let old_gen = self.generation(room_id);
            let mut index = match &old_gen {
                Some(g) => g.index.clone(),
                None => LossyIndex::new(4096),
            };
            let index_full = index.insert(id, shard_id, offset).is_err();
            if index_full {
                index = self.rebuild_index(room_id);
                let _ = index.insert(id, shard_id, offset);
            }
            let cache = match &old_gen {
                Some(g) => g.cache.clone(),
                None => Arc::new(NodeCache::new(self.cache_capacity)),
            };

            let mut data_to_cache = data.clone();
            for child in &mut data_to_cache.children {
                if let NodeRef::Lazy(child_id) = child {
                    if let Some(child_data) = self.pinned.get(child_id) {
                        *child = NodeRef::Resolved(*child_id, child_data);
                    }
                }
            }
            cache.insert(*id, Arc::new(data_to_cache));

            Arc::new(RoomGeneration { index, cache })
        };

        self.rooms
            .write()
            .entry(*room_id)
            .or_insert_with(|| {
                ArcSwap::from_pointee(RoomGeneration {
                    index: LossyIndex::new(0),
                    cache: Arc::new(NodeCache::new(self.cache_capacity)),
                })
            })
            .store(new_gen);

        Ok(())
    }

    fn put_many(
        &self,
        room_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        for (id, data) in entries {
            self.put(room_id, id, data)?;
        }
        Ok(())
    }

    fn delete_room(&self, room_id: &[u8; 16]) -> Result<(), StorageError> {
        self.rooms.write().remove(room_id);
        self.live_roots.write().remove(room_id);
        self.put_locks.lock().remove(room_id);
        self.persist_deleted_room(room_id)?;
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        Ok(self.shards.sync_all()?)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const TEST_ROOM: [u8; 16] = [0x01; 16];

    /// A node id distinct across both the bucket bytes (0..8) and tag bytes
    /// (8..12) the lossy index actually reads — an id that only varies byte
    /// 0 collapses every entry onto the same 24-bit tag, which is not a
    /// realistic content hash and defeats the index in ways unrelated to
    /// whatever a test using it is meant to check.
    fn distinct_id(byte: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[0] = byte;
        id[9] = byte.wrapping_mul(37).wrapping_add(11);
        id[15] = byte.wrapping_mul(7);
        id
    }
    const OTHER_ROOM: [u8; 16] = [0x02; 16];

    fn test_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_pfs_{name}_{id}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ten_record_fixture() -> Vec<(NodeId, NodeData)> {
        (0..10u8)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                id[8..12].copy_from_slice(&(u32::from(i) + 1).to_le_bytes());
                (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
            })
            .collect()
    }

    #[test]
    fn test_put_and_get() {
        let dir = test_dir("putget");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x42u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"hello world"));

        store.put(&TEST_ROOM, &id, &data).unwrap();
        let got = store.get(&TEST_ROOM, &id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);
    }

    #[test]
    fn test_read_survives_append_after_mmap_established() {
        let dir = test_dir("stale_mmap_regression");
        let store = PackfileStorage::open(dir).unwrap();

        let a = [0xAAu8; 16];
        let b = [0xBBu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"bbbb"));

        store.put(&TEST_ROOM, &a, &data_a).unwrap();

        if let Some(gen) = store.generation(&TEST_ROOM) {
            gen.cache.clear();
        }
        let got_a = store.get(&TEST_ROOM, &a).unwrap();
        assert_eq!(got_a.unwrap().bytes, data_a.bytes);

        store.put(&TEST_ROOM, &b, &data_b).unwrap();

        if let Some(gen) = store.generation(&TEST_ROOM) {
            gen.cache.clear();
        }
        let got_b = store.get(&TEST_ROOM, &b).unwrap();
        assert_eq!(got_b.expect("B must be found").bytes, data_b.bytes);
    }

    #[test]
    fn test_get_not_found() {
        let dir = test_dir("notfound");
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&TEST_ROOM, &[0x00; 16]).unwrap().is_none());
    }

    #[test]
    fn test_cache_hit() {
        let dir = test_dir("cachehit");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"cached"));

        store.put(&TEST_ROOM, &id, &data).unwrap();

        let _ = store.get(&TEST_ROOM, &id).unwrap();
        let gen = store.generation(&TEST_ROOM).unwrap();
        assert_eq!(gen.cache.hits(), 1);

        let _ = store.get(&TEST_ROOM, &id).unwrap();
        assert_eq!(gen.cache.hits(), 2);
    }

    #[test]
    fn test_delete_room() {
        let dir = test_dir("delete");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"room data"));

        store.put(&OTHER_ROOM, &id, &data).unwrap();

        store.delete_room(&OTHER_ROOM).unwrap();
        assert!(store.get(&OTHER_ROOM, &id).unwrap().is_none());
        assert!(store.generation(&OTHER_ROOM).is_none());
    }

    #[test]
    fn test_delete_room_clears_live_roots() {
        let dir = test_dir("delete_live_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        store
            .put(
                &OTHER_ROOM,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.set_live_roots(&OTHER_ROOM, vec![id]);
        assert!(store.live_roots.read().contains_key(&OTHER_ROOM));

        store.delete_room(&OTHER_ROOM).unwrap();
        assert!(!store.live_roots.read().contains_key(&OTHER_ROOM));
    }

    #[test]
    fn test_batch_put_get() {
        let dir = test_dir("batch");
        let store = PackfileStorage::open(dir).unwrap();

        let entries = ten_record_fixture();

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
    fn test_get_many_preserves_caller_order_despite_sorted_reads() {
        let dir = test_dir("batch_order");
        let store = PackfileStorage::open(dir).unwrap();

        let entries = ten_record_fixture();
        store.put_many(&TEST_ROOM, &entries).unwrap();
        if let Some(gen) = store.generation(&TEST_ROOM) {
            gen.cache.clear();
        }

        let mut reversed_ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        reversed_ids.reverse();

        let results = store.get_many(&TEST_ROOM, &reversed_ids).unwrap();
        assert_eq!(results.len(), 10);
        for (i, id) in reversed_ids.iter().enumerate() {
            let expected = &entries.iter().find(|(eid, _)| eid == id).unwrap().1;
            assert_eq!(
                results[i].as_ref().expect("record must be found").bytes,
                expected.bytes,
                "result at position {i} must match the id requested at that position"
            );
        }
    }

    #[test]
    fn test_multiple_records_same_room() {
        let dir = test_dir("multi");
        let store = PackfileStorage::open(dir).unwrap();

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let data = NodeData::new(bytes::Bytes::from(format!("record {i}")));
            store.put(&TEST_ROOM, &id, &data).unwrap();
        }

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let got = store.get(&TEST_ROOM, &id).unwrap().unwrap();
            assert_eq!(got.bytes, bytes::Bytes::from(format!("record {i}")));
        }
    }

    #[test]
    fn test_room_isolation() {
        let dir = test_dir("isolation");
        let store = PackfileStorage::open(dir).unwrap();

        let id_a = [0x42u8; 16];
        let id_b = [0x43u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"room A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"room B data"));

        store.put(&TEST_ROOM, &id_a, &data_a).unwrap();
        store.put(&OTHER_ROOM, &id_b, &data_b).unwrap();

        let got_a = store.get(&TEST_ROOM, &id_a).unwrap().unwrap();
        let got_b = store.get(&OTHER_ROOM, &id_b).unwrap().unwrap();
        assert_eq!(got_a.bytes, data_a.bytes);
        assert_eq!(got_b.bytes, data_b.bytes);

        assert!(store.get(&OTHER_ROOM, &id_a).unwrap().is_none());
        assert!(store.get(&TEST_ROOM, &id_b).unwrap().is_none());

        store.delete_room(&TEST_ROOM).unwrap();
        assert!(store.get(&TEST_ROOM, &id_a).unwrap().is_none());
        let got_b = store.get(&OTHER_ROOM, &id_b).unwrap().unwrap();
        assert_eq!(got_b.bytes, data_b.bytes);
    }

    #[test]
    fn test_concurrent_federation_swarm() {
        use std::sync::Mutex;
        use std::thread;

        const NUM_WRITERS: usize = 8;
        const EVENTS_PER_WRITER: usize = 500;
        const NUM_READERS: usize = 8;
        const READS_PER_READER: usize = 2000;

        let dir = test_dir("concurrent_federation");
        let store = PackfileStorage::open(dir).unwrap();

        let room = [0x77u8; 16];
        let written: Mutex<Vec<(NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());
        let read_ok = std::sync::atomic::AtomicUsize::new(0);
        let read_not_found = std::sync::atomic::AtomicUsize::new(0);

        thread::scope(|scope| {
            for w in 0..NUM_WRITERS {
                let store = &store;
                let written = &written;
                scope.spawn(move || {
                    for i in 0..EVENTS_PER_WRITER {
                        let mut id = [0u8; 16];
                        id[0] = u8::try_from(w).unwrap();
                        id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
                        let bytes = bytes::Bytes::from(format!("writer {w} event {i}"));
                        let data = NodeData::new(bytes.clone());
                        store.put(&room, &id, &data).unwrap();
                        written.lock().unwrap().push((id, bytes));
                    }
                });
            }

            for _ in 0..NUM_READERS {
                let store = &store;
                let written = &written;
                let read_ok = &read_ok;
                let read_not_found = &read_not_found;
                scope.spawn(move || {
                    for i in 0..READS_PER_READER {
                        let snapshot_len = written.lock().unwrap().len();
                        if snapshot_len == 0 {
                            continue;
                        }
                        let idx = i % snapshot_len;
                        let (id, expected_bytes) = written.lock().unwrap()[idx].clone();
                        match store.get(&room, &id).unwrap() {
                            Some(data) => {
                                assert_eq!(data.bytes, expected_bytes);
                                read_ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            None => {
                                read_not_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });

        let written = written.into_inner().unwrap();
        assert_eq!(written.len(), NUM_WRITERS * EVENTS_PER_WRITER);

        let mut lost = Vec::new();
        for (id, expected_bytes) in &written {
            match store.get(&room, id).unwrap() {
                Some(data) => assert_eq!(data.bytes, *expected_bytes),
                None => lost.push(*id),
            }
        }

        eprintln!(
            "concurrent federation swarm: {} writes, {} reads ok, {} reads not-found, {} lost",
            written.len(),
            read_ok.load(std::sync::atomic::Ordering::Relaxed),
            read_not_found.load(std::sync::atomic::Ordering::Relaxed),
            lost.len()
        );

        assert!(
            lost.is_empty(),
            "{} of {} records lost: {lost:?}",
            lost.len(),
            written.len()
        );
    }

    #[test]
    fn test_swizzle_callback() {
        use std::sync::atomic::Ordering;

        static SWIZZLE_CALLS: AtomicU64 = AtomicU64::new(0);
        static CACHED_CHILDREN_FOUND: AtomicU64 = AtomicU64::new(0);

        fn test_swizzle(
            data: &NodeData,
            children: &[NodeId],
            cached: &[Option<Arc<NodeData>>],
        ) -> NodeData {
            SWIZZLE_CALLS.fetch_add(1, Ordering::Relaxed);
            assert_eq!(children.len(), 2);
            for entry in cached {
                if entry.is_some() {
                    CACHED_CHILDREN_FOUND.fetch_add(1, Ordering::Relaxed);
                }
            }
            data.clone()
        }

        let dir = test_dir("swizzle");
        let store = PackfileStorage::open_with_swizzle(dir, 100, test_swizzle).unwrap();

        let parent_id = [0x10u8; 16];
        let child_a = [0x20u8; 16];
        let child_b = [0x30u8; 16];

        let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
        let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));

        store.put(&TEST_ROOM, &child_a, &data_a).unwrap();
        store.put(&TEST_ROOM, &child_b, &data_b).unwrap();
        store.put(&TEST_ROOM, &parent_id, &parent_data).unwrap();

        if let Some(gen) = store.generation(&TEST_ROOM) {
            gen.cache.clear();
        }
        store.put(&TEST_ROOM, &child_a, &data_a).unwrap();
        store.put(&TEST_ROOM, &child_b, &data_b).unwrap();

        let extract = |_data: &NodeData| -> Vec<NodeId> { vec![child_a, child_b] };

        let result = store.get_swizzled(&TEST_ROOM, &parent_id, extract).unwrap();
        assert!(result.is_some());
        let node_ref = result.unwrap();
        assert!(node_ref.is_resolved());
        assert_eq!(node_ref.structural_hash(), &parent_id);

        assert_eq!(SWIZZLE_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(CACHED_CHILDREN_FOUND.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_sync() {
        let dir = test_dir("sync");
        let store = PackfileStorage::open(dir).unwrap();
        store
            .put(
                &TEST_ROOM,
                &[0xAA; 16],
                &NodeData::new(bytes::Bytes::from_static(b"hi")),
            )
            .unwrap();
        store.sync().unwrap();
    }

    #[test]
    fn test_rebuild_index_triggers_on_full_table() {
        let dir = test_dir("rebuild_index_full");
        let store = PackfileStorage::open(dir).unwrap();

        let threshold = 3073u32;
        for i in 0..threshold {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            id[8..12].copy_from_slice(&(i + 1).to_le_bytes());
            store
                .put(
                    &TEST_ROOM,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(b"x")),
                )
                .unwrap();
        }

        let mut extra = [0u8; 16];
        extra[0..4].copy_from_slice(&threshold.to_le_bytes());
        extra[8..12].copy_from_slice(&(threshold + 1).to_le_bytes());
        store
            .put(
                &TEST_ROOM,
                &extra,
                &NodeData::new(bytes::Bytes::from_static(b"y")),
            )
            .unwrap();

        let got = store.get(&TEST_ROOM, &extra).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"y"));
    }

    #[test]
    fn test_scan_existing_skips_malformed_filenames() {
        let dir = test_dir("scan_existing_junk");
        std::fs::write(dir.join("nounderscore.pack"), b"").unwrap();
        std::fs::write(dir.join("aabb_00.pack"), b"").unwrap();
        std::fs::write(dir.join("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_00.pack"), b"").unwrap();
        std::fs::write(dir.join("00000000000000000000000000000000_gg.pack"), b"").unwrap();
        {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt;
            let path = dir.join(OsStr::from_bytes(b"\xff\xfe.pack"));
            std::fs::write(path, b"").unwrap();
        }

        let valid_path = dir.join("shard_00.pack");
        let mut buf = Vec::new();
        packfile::write_header(&mut buf).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                room_id: [0x01; 16],
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"hello"),
            },
        )
        .unwrap();
        std::fs::write(&valid_path, &buf).unwrap();

        let store = PackfileStorage::open(dir).unwrap();
        let gen = store.generation(&[0x01u8; 16]).unwrap();
        assert_eq!(gen.index.len(), 1);
    }

    #[test]
    fn test_repack_room_rewrite_preserves_all_records() {
        let dir = test_dir("repack_rewrite");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let root = [0xAAu8; 16];
        let child1 = [0x01u8; 16];
        let child2 = [0x02u8; 16];
        let child3 = [0x03u8; 16];

        let entries: Vec<(NodeId, NodeData)> = vec![
            (root, NodeData::new(bytes::Bytes::from_static(b"root"))),
            (child1, NodeData::new(bytes::Bytes::from_static(b"child1"))),
            (child2, NodeData::new(bytes::Bytes::from_static(b"child2"))),
            (child3, NodeData::new(bytes::Bytes::from_static(b"child3"))),
        ];
        store.put_many(&TEST_ROOM, &entries).unwrap();

        assert_eq!(store.generation(&TEST_ROOM).unwrap().index.len(), 4);

        store.repack_room_rewrite(&TEST_ROOM).unwrap();

        assert_eq!(store.generation(&TEST_ROOM).unwrap().index.len(), 4);
        assert!(store.get(&TEST_ROOM, &root).unwrap().is_some());
        assert!(store.get(&TEST_ROOM, &child1).unwrap().is_some());
        assert!(store.get(&TEST_ROOM, &child2).unwrap().is_some());
        assert!(store.get(&TEST_ROOM, &child3).unwrap().is_some());
    }

    #[test]
    fn test_delete_room_preserves_other_room_cache() {
        let dir = test_dir("delete_room_cache");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let room_a = TEST_ROOM;
        let room_b = OTHER_ROOM;
        let id_a = [0x10u8; 16];
        let id_b = [0x20u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"room A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"room B data"));

        store.put(&room_a, &id_a, &data_a).unwrap();
        store.put(&room_b, &id_b, &data_b).unwrap();

        assert!(store.get(&room_a, &id_a).unwrap().is_some());
        assert!(store.get(&room_b, &id_b).unwrap().is_some());

        let gen_b_before = store.generation(&room_b).unwrap();
        let hits_b_before = gen_b_before.cache.hits();

        store.delete_room(&room_a).unwrap();

        assert!(store.get(&room_a, &id_a).unwrap().is_none());
        assert!(store.generation(&room_a).is_none());
        assert!(store.generation(&room_b).is_some());
        assert!(store.get(&room_b, &id_b).unwrap().is_some());

        let gen_b_after = store.generation(&room_b).unwrap();
        assert!(gen_b_after.cache.hits() > hits_b_before);
    }

    #[test]
    fn test_concurrent_put_repack_no_lost_writes() {
        use std::thread;

        let dir = test_dir("concurrent_put_repack");
        let store = PackfileStorage::open(dir).unwrap();

        let room = [0x55u8; 16];
        let written_count = std::sync::atomic::AtomicU32::new(0);

        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for i in 0..200u32 {
                    let mut id = [0u8; 16];
                    id[0..4].copy_from_slice(&i.to_le_bytes());
                    let data = NodeData::new(bytes::Bytes::from(format!("entry {i}")));
                    store.put(&room, &id, &data).unwrap();
                    written_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

            let repacker = scope.spawn(|| {
                for _ in 0..5 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    let _ = store.repack_room_rewrite(&room);
                }
            });

            writer.join().unwrap();
            repacker.join().unwrap();
        });

        let total = written_count.load(std::sync::atomic::Ordering::Relaxed);
        let mut found = 0u32;
        for i in 0..total {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            if store.get(&room, &id).unwrap().is_some() {
                found += 1;
            }
        }

        assert_eq!(found, total, "lost writes during concurrent put+repack");
    }

    #[test]
    fn test_repack_room_topo_preserves_all_records() {
        let dir = test_dir("repack_topo");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;
        let mut id_c = [0u8; 16];
        id_c[0] = 3;
        let mut id_d = [0u8; 16];
        id_d[0] = 4;

        store
            .put(
                &TEST_ROOM,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_ROOM,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();
        store
            .put(
                &TEST_ROOM,
                &id_c,
                &NodeData::new(bytes::Bytes::from_static(b"C")),
            )
            .unwrap();
        store
            .put(
                &TEST_ROOM,
                &id_d,
                &NodeData::new(bytes::Bytes::from_static(b"D")),
            )
            .unwrap();

        let edges = std::collections::HashMap::from([
            (id_b, vec![id_a]),
            (id_c, vec![id_a]),
            (id_d, vec![id_b, id_c]),
        ]);

        let result = store.repack_room_topo(&TEST_ROOM, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        });
        result.unwrap();

        for (id, expected) in [
            (id_a, b"A".as_slice()),
            (id_b, b"B".as_slice()),
            (id_c, b"C".as_slice()),
            (id_d, b"D".as_slice()),
        ] {
            let got = store
                .get(&TEST_ROOM, &id)
                .unwrap()
                .expect("record missing after topo repack");
            assert_eq!(got.bytes.as_ref(), expected);
        }
    }

    #[test]
    fn test_repack_room_reachable_drops_unreachable_records() {
        let dir = test_dir("repack_reachable_gc");
        let store = PackfileStorage::open(dir).unwrap();

        // Live chain: root -> p1 -> p0 (p0 has no further dependencies).
        let root = distinct_id(1);
        let p1 = distinct_id(2);
        let p0 = distinct_id(3);

        // Garbage island, unreachable from root: garbage -> garbage_dep.
        let garbage = distinct_id(4);
        let garbage_dep = distinct_id(5);

        for (id, bytes) in [
            (root, b"root".as_slice()),
            (p1, b"p1".as_slice()),
            (p0, b"p0".as_slice()),
            (garbage, b"garbage".as_slice()),
            (garbage_dep, b"garbage_dep".as_slice()),
        ] {
            store
                .put(
                    &TEST_ROOM,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(bytes)),
                )
                .unwrap();
        }

        let edges = std::collections::HashMap::from([
            (root, vec![p1]),
            (p1, vec![p0]),
            (garbage, vec![garbage_dep]),
        ]);

        store.set_live_roots(&TEST_ROOM, vec![root]);

        let (kept, dropped) = store
            .repack_room_reachable(&TEST_ROOM, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();

        assert_eq!(kept, 3, "root, p1, p0 should survive");
        assert_eq!(dropped, 2, "garbage and garbage_dep should be collected");

        for id in [root, p1, p0] {
            assert!(
                store.get(&TEST_ROOM, &id).unwrap().is_some(),
                "live record missing after reachable repack"
            );
        }
        for id in [garbage, garbage_dep] {
            assert!(
                store.get(&TEST_ROOM, &id).unwrap().is_none(),
                "garbage record survived reachable repack"
            );
        }
    }

    #[test]
    fn test_repack_room_reachable_without_live_roots_preserves_everything() {
        let dir = test_dir("repack_reachable_no_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;

        store
            .put(
                &TEST_ROOM,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_ROOM,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();

        // No set_live_roots call for this room: nothing is known to be
        // garbage, so everything must survive.
        let (kept, dropped) = store
            .repack_room_reachable(&TEST_ROOM, |_hash, _data| Vec::new())
            .unwrap();

        assert_eq!(kept, 2);
        assert_eq!(dropped, 0);
        assert!(store.get(&TEST_ROOM, &id_a).unwrap().is_some());
        assert!(store.get(&TEST_ROOM, &id_b).unwrap().is_some());
    }

    #[test]
    fn test_concurrent_put_repack_reachable_no_lost_writes() {
        use std::thread;

        let dir = test_dir("concurrent_put_repack_reachable");
        let store = PackfileStorage::open(dir).unwrap();

        let room = [0x66u8; 16];
        let written_count = std::sync::atomic::AtomicU32::new(0);

        // No live roots configured, so the reachable repacker must fall
        // back to "preserve everything" — this proves that fallback holds
        // even when interleaved with concurrent writes.
        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for i in 0..200u32 {
                    let mut id = [0u8; 16];
                    id[0..4].copy_from_slice(&i.to_le_bytes());
                    let data = NodeData::new(bytes::Bytes::from(format!("entry {i}")));
                    store.put(&room, &id, &data).unwrap();
                    written_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

            let repacker = scope.spawn(|| {
                for _ in 0..5 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    let _ = store.repack_room_reachable(&room, |_hash, _data| Vec::new());
                }
            });

            writer.join().unwrap();
            repacker.join().unwrap();
        });

        let total = written_count.load(std::sync::atomic::Ordering::Relaxed);
        let mut found = 0u32;
        for i in 0..total {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            if store.get(&room, &id).unwrap().is_some() {
                found += 1;
            }
        }

        assert_eq!(
            found, total,
            "lost writes during concurrent put+reachable-repack"
        );
    }
}
