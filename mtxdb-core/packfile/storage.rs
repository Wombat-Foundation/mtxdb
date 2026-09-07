use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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
type ScannedShard = (u16, Vec<([u8; 16], u64)>);

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
    room_order: RwLock<Vec<[u8; 16]>>,
    pinned: PinnedNodes,
    base_dir: PathBuf,
    swizzle: Option<SwizzleFn>,
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    repack_threshold_entries: AtomicU64,
    cache_capacity: usize,
    /// Total number of `repack_room_reachable` calls across all rooms.
    repack_count: AtomicU64,
    /// Total records kept (rewritten into the new generation) across all repacks.
    repack_kept_total: AtomicU64,
    /// Total records dropped (found unreachable) across all repacks.
    repack_dropped_total: AtomicU64,
    /// Per-room repack counts, so a hot room's churn is visible individually.
    repack_counts_by_room: RwLock<HashMap<[u8; 16], u64>>,
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
        type ShardRecord = (u16, [u8; 16], u64);
        fs::create_dir_all(&base_dir)?;

        let shards = ShardPool::open(base_dir.clone())?;
        let mut rooms: HashMap<[u8; 16], ArcSwap<RoomGeneration>> = HashMap::new();

        // Phase 1: accumulate all records per room across every shard so
        // the index can be sized once for the true total.
        let mut room_entries: HashMap<[u8; 16], Vec<ShardRecord>> = HashMap::new();
        let mut room_order: Vec<[u8; 16]> = Vec::new();

        // Collect open shard info (slot, path) before the scan loop so we
        // don't hold the shards read-lock across the I/O-heavy scan.
        let open_shards: Vec<(u16, PathBuf)> = shards
            .all_shards()
            .into_iter()
            .map(|(id, shard)| (id, shard.path.clone()))
            .collect();

        for (shard_id, path) in open_shards {
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
                    .or_insert_with(|| {
                        room_order.push(room_id);
                        Vec::new()
                    })
                    .push((shard_id, hash, offset));
            }
        }

        // P1: load deleted rooms set (persisted to disk)
        let deleted_rooms = Self::load_deleted_rooms(&base_dir);

        // Phase 2: build per-room indexes sized to the true total.
        for room_id in &room_order {
            let records = &room_entries[room_id];
            if deleted_rooms.contains(room_id) {
                continue;
            }
            // Seed each room's home shard from the scan: the shard of its
            // last-scanned record is a best-effort proxy for "most recent"
            // (shards are scanned in ascending ID order, and IDs generally
            // increase over time via rotation) — not exact chronology
            // across shards, but enough to keep a resumed room's writes
            // landing near its existing data instead of restarting at
            // whatever the pool's active shard happens to be.
            if let Some(&(last_shard_id, _, _)) = records.last() {
                shards.set_room_home(room_id, last_shard_id);
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
            room_order: RwLock::new(room_order),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_room: RwLock::new(HashMap::new()),
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

    /// Room IDs whose live index currently references at least one record
    /// physically stored in `shard_id`.
    ///
    /// Shards are shared, so this is normally more than one room; it's the
    /// set `repack_shard` needs to touch before that shard can retire.
    #[must_use]
    pub fn rooms_referencing_shard(&self, shard_id: u16) -> Vec<[u8; 16]> {
        let rooms = self.rooms.read();
        rooms
            .iter()
            .filter_map(|(room_id, gen_swap)| {
                let gen = gen_swap.load();
                gen.index
                    .referenced_shard_ids()
                    .get(shard_id as usize)
                    .copied()
                    .unwrap_or(false)
                    .then_some(*room_id)
            })
            .collect()
    }

    /// Repack every room that still references `shard_id`.
    ///
    /// A shard only ever retires once *every* room referencing it has
    /// repacked past its data there (see `retire_empty_shards`) — there's
    /// no per-shard compaction primitive, since reachability (what's
    /// live vs. garbage) is inherently a per-room concept, not a shard
    /// one. This is the targeted way to reclaim one specific shard: find
    /// every room still pinning it live and repack each of them, instead
    /// of waiting for each room to independently cross its own repack
    /// threshold. Live records get moved to whatever's currently that
    /// room's home shard (`ShardPool::room_home`) — not necessarily the
    /// pool's single "main" shard, since homes are per-room, not global.
    ///
    /// Returns `(room_id, kept, dropped)` for each room repacked, in the
    /// order `rooms_referencing_shard` returned them. Does not itself
    /// guarantee the shard retires — a room with `set_live_roots` never
    /// called preserves everything (nothing is provably garbage) and its
    /// repack is a no-op for this purpose.
    ///
    /// # Errors
    /// Returns `StorageError` if any room's repack fails; already-repacked
    /// rooms in this call are not rolled back.
    pub fn repack_shard(
        &self,
        shard_id: u16,
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let rooms = self.rooms_referencing_shard(shard_id);
        let mut results = Vec::with_capacity(rooms.len());
        for room_id in rooms {
            let (kept, dropped) =
                self.repack_room_reachable(&room_id, |hash, data| extract_edges(hash, data))?;
            results.push((room_id, kept, dropped));
        }
        Ok(results)
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
        let rooms = self.rooms.read();
        self.room_order
            .read()
            .iter()
            .filter_map(|id| {
                rooms.get(id).map(|gen| {
                    let g = gen.load();
                    (*id, g.index.len(), g.index.memory_usage())
                })
            })
            .collect()
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

    /// Returns `true` if a room's index has reached the configured repack
    /// threshold.
    ///
    /// Repacking is entirely caller-driven — nothing in this engine polls
    /// this on its own. A background GC worker is expected to call this
    /// periodically and issue `repack_room_reachable` itself; nothing in
    /// `mtxdb-core` currently does so.
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

    fn scan_room_records(&self, room_id: &[u8; 16]) -> Result<Vec<ScannedShard>, StorageError> {
        let mut scanned: Vec<ScannedShard> = Vec::new();
        for (shard_id, shard) in self.shards.all_shards() {
            match packfile::scan_packfile(&shard.path) {
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
                    return Err(StorageError::Io(e));
                }
            }
        }
        Ok(scanned)
    }

    fn build_index(offsets: &[([u8; 16], u16, u64)]) -> LossyIndex {
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

    fn clear_deleted_room(&self, room_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_rooms_path(&self.base_dir);
        let mut set = Self::load_deleted_rooms(&self.base_dir);
        if set.remove(room_id) {
            let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
            fs::write(&path, &bytes).map_err(StorageError::Io)?;
        }
        Ok(())
    }

    /// Pin one `Arc<Shard>` per distinct shard id, keeping each shard's file
    /// handle (and thus its `Drop`-triggered deletion, see `Shard::drop`)
    /// alive for as long as the returned map is held — even if a later
    /// write in the same operation triggers `ShardPool::rotate` and retires
    /// that shard id's pool slot.
    ///
    /// This matters specifically for a scan-then-rewrite repack: without
    /// it, re-querying `ShardPool::get_shard` on every loop iteration can
    /// observe a shard that a *rotation triggered by the same repack's own
    /// writes* just retired-and-recreated at the same path, turning a
    /// stale-but-valid offset into a read into unrelated, freshly-written
    /// bytes.
    fn pin_shards(&self, shard_ids: impl Iterator<Item = u16>) -> HashMap<u16, Arc<Shard>> {
        let unique: HashSet<u16> = shard_ids.collect();
        unique
            .into_iter()
            .filter_map(|id| self.shards.get_shard(id).map(|shard| (id, shard)))
            .collect()
    }

    /// Copy a record from an old (pinned) shard to the active shard.
    /// Returns the new `(hash, shard_id, offset)` or None if `old_shard_id`
    /// isn't in `pinned`.
    fn copy_record_to_shard(
        &self,
        room_id: &[u8; 16],
        pinned: &HashMap<u16, Arc<Shard>>,
        old_shard_id: u16,
        old_offset: u64,
    ) -> Result<Option<([u8; 16], u16, u64)>, StorageError> {
        let Some(old_shard) = pinned.get(&old_shard_id) else {
            return Ok(None);
        };
        let record = Self::read_at(old_shard, old_offset)?;
        let (new_shard_id, new_offset) = self.shards.put_record(&Record {
            room_id: *room_id,
            hash: record.hash,
            data: record.data,
        })?;
        Ok(Some((record.hash, new_shard_id, new_offset)))
    }
    fn swap_generation(&self, room_id: &[u8; 16], index: LossyIndex) {
        let cache = self.generation(room_id).map(|g| g.cache.clone());
        self.store_generation(room_id, index, cache);
    }

    /// Store a new generation for a room, reusing the existing cache if present.
    ///
    /// If the room is new (not yet in the `rooms` map), it is added to
    /// `room_order` so that [`Self::room_summaries`] will include it, and
    /// any prior tombstone in `deleted.rooms` is cleared so the room
    /// survives a restart.
    fn store_generation(
        &self,
        room_id: &[u8; 16],
        index: LossyIndex,
        cache: Option<Arc<NodeCache>>,
    ) {
        let cache = cache.unwrap_or_else(|| Arc::new(NodeCache::new(self.cache_capacity)));
        let new_gen = Arc::new(RoomGeneration { index, cache });

        let is_new = self.rooms.read().get(room_id).is_none();

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

        if is_new {
            self.room_order.write().push(*room_id);
            let _ = self.clear_deleted_room(room_id);
        }
    }

    fn rebuild_index(&self, room_id: &[u8; 16]) -> Result<LossyIndex, StorageError> {
        let scanned = self.scan_room_records(room_id)?;
        let total: usize = scanned.iter().map(|(_, e)| e.len()).sum();
        let mut index = LossyIndex::new(total.saturating_mul(2).max(16));
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                let _ = index.insert(&hash, shard_id, offset);
            }
        }
        Ok(index)
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
        candidates: &[(u16, u64)],
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

    // repack_room_rewrite and repack_room_topo were retired: both scanned
    // for records without deduplicating physical duplicates (each prior
    // repack's rewritten copies included), giving an exponential blowup on
    // repeated calls against a growing room, and both re-fetched
    // Arc<Shard> per loop iteration rather than pinning shards for the
    // call's duration — vulnerable to the rotation race documented on
    // `pin_shards`. `repack_room_reachable`'s no-live-roots fallback
    // subsumes both: same topological ordering, correct dedup via
    // `hash_to_shard_offset`, and pinned shards throughout.

    /// BFS outward from `roots` over `extract_edges`, reading only records
    /// actually reached. Returns the sorted, deduplicated live hash set and
    /// the adjacency discovered along the way.
    fn bfs_live_set(
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
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
            let Some(old_shard) = pinned.get(&shard_id) else {
                continue;
            };
            let record = Self::read_at(old_shard, offset)?;
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
        scanned: &[ScannedShard],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                if let Some(old_shard) = pinned.get(shard_id) {
                    let record = Self::read_at(old_shard, *offset)?;
                    let edges = extract_edges(hash, &record.data);
                    adjacency.insert(*hash, edges);
                }
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Rewrite a room's records in topological order, optionally performing
    /// garbage collection — this is the engine's one repack entry point.
    ///
    /// If live roots are configured for this room (see
    /// [`Self::set_live_roots`]), traverses outward from them via
    /// `extract_edges` and drops anything not reached: only reachable
    /// records are read from disk during the traversal, so unreachable
    /// data is never even fetched, not just excluded from the output.
    ///
    /// If no live roots are configured (`set_live_roots` was never called,
    /// or was called with an empty list), nothing is known to be garbage,
    /// so every scanned record is preserved, still deduplicated and
    /// rewritten in topological order — rather than risk deleting live
    /// data on the assumption that "no roots" means "nothing is live".
    ///
    /// All shards this call will read from are pinned (held via an
    /// `Arc<Shard>` for the whole call) before any writes happen, so a
    /// rotation triggered by this call's own writes can never retire a
    /// shard this
    /// call still needs to read stale data from.
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

        let scanned = self.scan_room_records(room_id)?;

        let scanned_count: usize = scanned.iter().map(|(_, entries)| entries.len()).sum();

        let mut hash_to_shard_offset: HashMap<[u8; 16], (u16, u64)> = HashMap::new();
        for (shard_id, entries) in &scanned {
            for (hash, offset) in entries {
                hash_to_shard_offset.insert(*hash, (*shard_id, *offset));
            }
        }

        // Pin every shard this call could possibly read from before any
        // writes happen — see pin_shards' doc for why this must come
        // first.
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(room_id).cloned();

        let (live_hashes, adjacency) = match roots {
            Some(roots) if !roots.is_empty() => {
                Self::bfs_live_set(&roots, &hash_to_shard_offset, &pinned, &extract_edges)?
            }
            // No live roots configured: we don't know what's garbage, so
            // preserve everything.
            _ => {
                Self::scan_full_adjacency(&scanned, &hash_to_shard_offset, &pinned, &extract_edges)?
            }
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
                        let mut hex = String::with_capacity(32);
                        for b in hash {
                            let _ = write!(hex, "{b:02x}");
                        }
                        eprintln!("  dropped: {hex}");
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "repack: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        let mut new_offsets: Vec<([u8; 16], u16, u64)> = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some(entry) =
                self.copy_record_to_shard(room_id, &pinned, old_shard_id, old_offset)?
            {
                new_offsets.push(entry);
            }
        }

        let kept = new_offsets.len();

        let index = Self::build_index(&new_offsets);
        self.swap_generation(room_id, index);
        self.retire_empty_shards(room_id);

        self.repack_count.fetch_add(1, Ordering::Relaxed);
        self.repack_kept_total
            .fetch_add(kept as u64, Ordering::Relaxed);
        self.repack_dropped_total
            .fetch_add(dropped as u64, Ordering::Relaxed);
        self.repack_counts_by_room
            .write()
            .entry(*room_id)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);

        // Fsync the shards this repack just wrote into and persist the
        // updated stats snapshot as part of finishing the repack, rather
        // than leaving both to whatever the next unrelated sync_dirty()
        // call happens to be — a crash right after a repack should not
        // lose durability for data this repack itself just wrote, nor
        // leave the persisted stats stale relative to what's on disk.
        self.shards.sync_dirty()?;

        Ok((kept, dropped))
    }

    /// After a repack, scan all rooms' indexes and retire any shard that
    /// no room references.
    ///
    /// Acquires every room's `put_mutex` (in sorted order, skipping the
    /// caller's already-held lock) to prevent a concurrent `put()` from
    /// landing a write in a shard whose index entry hasn't been inserted
    /// yet — without that, the scan could see zero references to a shard
    /// that a writer just committed bytes to but hasn't index-updated yet,
    /// causing a live shard to be retired under it.
    fn retire_empty_shards(&self, held_room: &[u8; 16]) {
        let rooms = self.rooms.read();

        // Collect room IDs in sorted order for deadlock-free lock acquisition.
        // Skip the room whose put_mutex the caller already holds.
        let mut rooms_to_lock: Vec<[u8; 16]> = rooms
            .keys()
            .filter(|id| *id != held_room)
            .copied()
            .collect();
        rooms_to_lock.sort_unstable();

        // Acquire all other rooms' put_mutexes. The sorted order prevents
        // deadlocks; the held room is skipped (parking_lot is non-reentrant).
        let mutexes: Vec<_> = rooms_to_lock.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();

        // Build the union of shard IDs referenced across all rooms.
        let mut referenced = [false; shard::MAX_SHARDS];
        for gen_swap in rooms.values() {
            let gen = gen_swap.load();
            let ids = gen.index.referenced_shard_ids();
            for (i, &has_refs) in ids.iter().enumerate() {
                referenced[i] |= has_refs;
            }
        }

        // Retire any shard not in the union.
        for id in 0..u16::try_from(shard::MAX_SHARDS).unwrap() {
            if !referenced[id as usize] {
                self.shards.retire_slot(id);
            }
        }
    }
}

impl StorageEngine for PackfileStorage {
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let gen_guard = self.generation(room_id);
        let gen = gen_guard.as_deref();

        let candidates: Vec<(u16, u64)> = match gen {
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

        let mut to_fetch: Vec<(usize, Vec<(u16, u64)>)> = Vec::new();
        if let Some(g) = gen {
            for (i, id) in ids.iter().enumerate() {
                let candidates: Vec<(u16, u64)> = g.index.lookup_all(id).collect();
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

        let (index, cache) = {
            let old_gen = self.generation(room_id);
            let mut index = match &old_gen {
                Some(g) => g.index.clone(),
                None => LossyIndex::new(4096),
            };
            let index_full = index.insert(id, shard_id, offset).is_err();
            if index_full {
                index = self.rebuild_index(room_id)?;
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

            (index, cache)
        };

        self.store_generation(room_id, index, Some(cache));

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
        // Acquire the room's put mutex to serialize with any in-flight put,
        // preventing a concurrent put from resurrecting the room after we
        // remove it from the generation map.
        let room_arc = self.put_mutex(room_id);
        let _room_guard = room_arc.lock();

        self.rooms.write().remove(room_id);
        self.live_roots.write().remove(room_id);
        self.put_locks.lock().remove(room_id);
        self.persist_deleted_room(room_id)?;
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        Ok(self.shards.sync_dirty()?)
    }
}

impl PackfileStorage {
    /// Sync all open shards to disk (full pool, not just dirty).
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure.
    pub fn sync_all(&self) -> Result<(), StorageError> {
        Ok(self.shards.sync_all()?)
    }

    /// Snapshot IO/sync stats for every currently-open shard.
    ///
    /// Shards are shared across rooms, so this is per-shard, not per-room.
    #[must_use]
    pub fn shard_stats(&self) -> Vec<(u16, crate::shard::ShardStats)> {
        self.shards.all_stats()
    }

    /// Get IO/sync stats for a single shard by ID.
    #[must_use]
    pub fn shard_stats_for(&self, shard_id: u16) -> Option<crate::shard::ShardStats> {
        self.shards.stats(shard_id)
    }

    /// Total number of shards retired (garbage-collected after a repack)
    /// over this storage's lifetime.
    #[must_use]
    pub fn shards_retired(&self) -> u64 {
        self.shards.retired_count()
    }

    /// List every currently-open shard with basic size, generation, and
    /// IO/sync stats — the data behind a `shards` CLI listing.
    #[must_use]
    pub fn shard_summaries(&self) -> Vec<ShardSummary> {
        self.shards
            .all_shards()
            .into_iter()
            .map(|(shard_id, shard)| ShardSummary {
                shard_id,
                generation: shard.generation,
                file_bytes: shard.file_len(),
                stats: shard.stats(),
            })
            .collect()
    }

    /// Snapshot global repack stats across all rooms.
    #[must_use]
    pub fn repack_stats(&self) -> RepackStats {
        RepackStats {
            repack_count: self.repack_count.load(Ordering::Relaxed),
            kept_total: self.repack_kept_total.load(Ordering::Relaxed),
            dropped_total: self.repack_dropped_total.load(Ordering::Relaxed),
        }
    }

    /// Number of times a specific room has been repacked. 0 if it has
    /// never been repacked (or doesn't exist).
    #[must_use]
    pub fn repack_count_for_room(&self, room_id: &[u8; 16]) -> u64 {
        self.repack_counts_by_room
            .read()
            .get(room_id)
            .copied()
            .unwrap_or(0)
    }

    /// Cache hit/miss stats for a room's decoded-node cache, if the room
    /// currently has one loaded.
    #[must_use]
    pub fn cache_stats_for(&self, room_id: &[u8; 16]) -> Option<CacheStats> {
        let gen = self.generation(room_id)?;
        Some(CacheStats {
            hits: gen.cache.hits(),
            misses: gen.cache.misses(),
            hit_rate: gen.cache.hit_rate(),
        })
    }
}

/// Basic size, generation, and IO/sync info for one open shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSummary {
    /// Shard ID within the pool.
    pub shard_id: u16,
    /// Monotonically increasing generation counter for this shard's slot.
    pub generation: u64,
    /// Current on-disk file length in bytes.
    pub file_bytes: u64,
    /// IO/sync counters for this shard.
    pub stats: shard::ShardStats,
}

/// Snapshot of global repack activity across all rooms.
#[derive(Debug, Clone, Copy, Default)]
pub struct RepackStats {
    /// Total number of `repack_room_reachable` calls across all rooms.
    pub repack_count: u64,
    /// Total records kept (rewritten into a new generation) across all repacks.
    pub kept_total: u64,
    /// Total records dropped (found unreachable) across all repacks.
    pub dropped_total: u64,
}

/// Snapshot of a room's decoded-node cache hit/miss stats.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub hit_rate: f64,
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
    fn test_delete_room_does_not_resurrect_on_reopen() {
        let dir = test_dir("delete_no_resurrect");

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"room data"));

        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            store.put(&OTHER_ROOM, &id, &data).unwrap();
            store.delete_room(&OTHER_ROOM).unwrap();
            store.sync_all().unwrap();
        }

        // Reopen from scratch: the on-disk deleted.rooms marker must make
        // the startup scan skip OTHER_ROOM's leftover packfile records,
        // rather than resurrecting them into a fresh index.
        let store = PackfileStorage::open(dir).unwrap();
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
    fn test_repack_room_reachable_no_roots_preserves_diamond_dag_in_topo_order() {
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

        // No live roots configured for TEST_ROOM: preserves everything,
        // still deduplicated and topologically ordered.
        let result = store.repack_room_reachable(&TEST_ROOM, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        });
        let (kept, dropped) = result.unwrap();
        assert_eq!(kept, 4);
        assert_eq!(dropped, 0);

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

        let stats = store.repack_stats();
        assert_eq!(stats.repack_count, 1);
        assert_eq!(stats.kept_total, 3);
        assert_eq!(stats.dropped_total, 2);
        assert_eq!(store.repack_count_for_room(&TEST_ROOM), 1);

        // A second repack rescans every shard still holding this room's
        // bytes from disk (repack rewrites live records into a fresh
        // generation but doesn't erase the old shard's bytes unless that
        // shard gets retired), so it now sees both the freshly-written
        // generation (3 live) and the original shard's full 5 records
        // (3 live + 2 garbage) again, dropping the garbage a second time.
        // Stats should accumulate across both calls rather than reset.
        let (kept2, dropped2) = store
            .repack_room_reachable(&TEST_ROOM, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();
        assert_eq!((kept2, dropped2), (3, 5));

        let stats2 = store.repack_stats();
        assert_eq!(stats2.repack_count, 2);
        assert_eq!(stats2.kept_total, stats.kept_total + kept2 as u64);
        assert_eq!(stats2.dropped_total, stats.dropped_total + dropped2 as u64);
        assert_eq!(store.repack_count_for_room(&TEST_ROOM), 2);
    }

    /// Repack drops unreachable *records* from the index (`dropped` count),
    /// but today nothing ever retires the *shard file* they lived in: no
    /// code path sets `Shard::is_current` to `false` or clears the pool's
    /// slot for it (see `Shard::drop`'s doc and
    /// `test_drop_deletes_retired_shard_only_after_last_reference` in
    /// shard.rs, which has to simulate retirement manually because nothing
    /// production triggers it). `rotate()`'s error message says "repack to
    /// reclaim", but repack currently reclaims nothing at the shard level.
    ///
    /// This fills a small fixed number of pool slots (4), repacks away
    /// everything in the non-active ones (100% garbage, nothing live left),
    /// and then expects one more rotation to succeed by reusing a
    /// now-empty slot.
    #[test]
    fn test_repack_reclaims_shard_slots_for_rotation() {
        // Use a fixed small count rather than MAX_SHARDS — with 4096 slots
        // and 256MB each that would be a 1TB test.
        const TEST_SHARDS: usize = 4;

        let dir = test_dir("repack_reclaims_slots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root = [0u8; 16];
        root[0] = 0xFF;
        store
            .put(
                &TEST_ROOM,
                &root,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();

        // `root` already occupies the first slot; force rotation through
        // the remaining TEST_SHARDS - 1 slots, dumping garbage into each so
        // every shard but the last ends up fully unreachable once we
        // repack with `root` as the only live node.
        for i in 0..TEST_SHARDS - 1 {
            store.shards.active_shard().file_len.store(
                shard::MAX_SHARD_BYTES - 10,
                std::sync::atomic::Ordering::Release,
            );
            let mut garbage = [0u8; 16];
            garbage[0] = u8::try_from(i + 1).unwrap();
            store
                .put(
                    &TEST_ROOM,
                    &garbage,
                    &NodeData::new(bytes::Bytes::from_static(b"garbage")),
                )
                .unwrap();
        }

        // Every slot should now be occupied.
        let occupied = (0..u16::try_from(TEST_SHARDS).unwrap())
            .filter(|&id| store.shards.get_shard(id).is_some())
            .count();
        assert_eq!(
            occupied, TEST_SHARDS,
            "test setup should have filled every shard slot"
        );

        store.set_live_roots(&TEST_ROOM, vec![root]);
        let (kept, dropped) = store
            .repack_room_reachable(&TEST_ROOM, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1, "only root should survive");
        assert!(dropped > 0, "the garbage records should have been dropped");
        assert!(
            store.shards_retired() > 0,
            "retire_empty_shards should have retired at least one now-garbage-only shard"
        );

        // Every shard except the current active one held nothing but
        // garbage, and that garbage is now unreachable -- those slots
        // should be reclaimable. Forcing one more rotation should reuse
        // one of them rather than failing.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut one_more = [0u8; 16];
        one_more[0] = 0xEE;
        store
            .put(
                &TEST_ROOM,
                &one_more,
                &NodeData::new(bytes::Bytes::from_static(b"one more")),
            )
            .expect(
                "rotation should reclaim a garbage-only shard slot after repack, \
                 not report the pool as permanently full",
            );
    }

    /// Repack must not retire a shard that another room's index still
    /// references.  This test puts live records from two different rooms
    /// into the same shard, then repacks only room A — the shared shard
    /// must survive because room B's index still points into it.
    #[test]
    fn test_repack_does_not_retire_shard_referenced_by_other_room() {
        let dir = test_dir("repack_cross_room_safety");
        let store = PackfileStorage::open(dir).unwrap();

        // Put a record for room A — lands on shard 0.
        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        store
            .put(
                &TEST_ROOM,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"room A root")),
            )
            .unwrap();

        // Force a rotation so the next write goes to a different shard.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );

        // Put a record for room B — lands on shard 1.
        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_ROOM,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"room B root")),
            )
            .unwrap();

        // Force another rotation and put more room A garbage on shard 2.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = 0xCC;
        store
            .put(
                &TEST_ROOM,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();

        // Repack room A with only root_a as live.  The garbage record
        // should be dropped, but shard 1 (which holds room B's root)
        // must NOT be retired.
        store.set_live_roots(&TEST_ROOM, vec![root_a]);
        let (kept, dropped) = store
            .repack_room_reachable(&TEST_ROOM, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1);
        assert!(dropped > 0);

        // Room B's root must still be retrievable — shard 1 was not retired.
        let got = store.get(&OTHER_ROOM, &root_b).unwrap();
        assert!(
            got.is_some(),
            "room B root must survive repack of room A — shared shard must not be retired",
        );
    }

    /// `repack_shard` must find and repack every room sharing a shard —
    /// the targeted way to reclaim one shard without waiting for each
    /// room to independently cross its own repack threshold.
    #[test]
    fn test_repack_shard_repacks_every_referencing_room() {
        let dir = test_dir("repack_shard");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        let mut garbage_a = [0u8; 16];
        garbage_a[0] = 0xA1;
        store
            .put(
                &TEST_ROOM,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"room A root")),
            )
            .unwrap();
        store
            .put(
                &TEST_ROOM,
                &garbage_a,
                &NodeData::new(bytes::Bytes::from_static(b"room A garbage")),
            )
            .unwrap();
        store.set_live_roots(&TEST_ROOM, vec![root_a]);

        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_ROOM,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"room B root")),
            )
            .unwrap();
        // No set_live_roots for room B: everything must survive its repack.

        // Fresh pool, both rooms' first writes land on shard 0 (shared).
        let referencing = store.rooms_referencing_shard(0);
        assert_eq!(referencing.len(), 2);
        assert!(referencing.contains(&TEST_ROOM));
        assert!(referencing.contains(&OTHER_ROOM));

        let mut results = store.repack_shard(0, |_hash, _data| Vec::new()).unwrap();
        results.sort_unstable_by_key(|(room_id, _, _)| *room_id);

        let mut expected = vec![(TEST_ROOM, 1usize, 1usize), (OTHER_ROOM, 1usize, 0usize)];
        expected.sort_unstable_by_key(|(room_id, _, _)| *room_id);
        assert_eq!(
            results, expected,
            "repack_shard must repack both rooms sharing shard 0"
        );

        assert!(store.get(&TEST_ROOM, &root_a).unwrap().is_some());
        assert!(store.get(&TEST_ROOM, &garbage_a).unwrap().is_none());
        assert!(store.get(&OTHER_ROOM, &root_b).unwrap().is_some());
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

    /// `pin_shards` must return a distinct `Arc<Shard>` per unique id, and
    /// the returned handle must still resolve real, previously-written
    /// data — i.e. it's the live pool's own shard, not a stand-in.
    /// The deeper guarantee (pinning survives a shard being retired and
    /// its slot recycled) is `Shard::drop`'s contract, tested directly in
    /// `shard.rs` where `is_current` and the pool's internals are
    /// accessible without going through `ShardPool::rotate`, which no
    /// longer has any path to actually recycle a slot (see its doc).
    #[test]
    fn test_pin_shards_returns_readable_deduped_handles() {
        let dir = test_dir("pin_shards_basic");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x77u8; 16];
        store
            .put(
                &TEST_ROOM,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"pin me")),
            )
            .unwrap();
        let (shard_id, offset) = store
            .generation(&TEST_ROOM)
            .unwrap()
            .index
            .lookup(&id)
            .expect("just-written record must be indexed");

        // Duplicate ids in the input must collapse to one pinned entry.
        let pinned = store.pin_shards([shard_id, shard_id, shard_id].into_iter());
        assert_eq!(pinned.len(), 1);

        let record = PackfileStorage::read_at(&pinned[&shard_id], offset)
            .expect("pinned shard must resolve the real on-disk record");
        assert_eq!(record.data.as_ref(), b"pin me");
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
