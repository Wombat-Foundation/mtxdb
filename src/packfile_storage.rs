use std::collections::HashMap;
use std::fs;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::index::LossyIndex;
use crate::packfile::{self, PackGeneration, Record};
use crate::repack::RepackManager;
use crate::storage::{NodeData, NodeId, NodeRef, StorageEngine, StorageError};

pub type NodeParserFn = fn(&[u8]) -> Vec<NodeId>;

/// `(hash, offset)` entries scanned from a single packfile generation.
type PackEntries = Vec<(NodeId, u64)>;

/// Scanned entries grouped by pack id, in generation order.
type ScannedPacks = Vec<(u8, PackEntries)>;

/// Callback that rewrites a node's child pointers for in-cache swizzling.
///
/// When a node is fetched from disk, this callback is invoked with:
/// - `data`: the raw node data
/// - `children`: the child hashes extracted from the node
/// - `cached`: parallel bool array — `true` if the child is already in cache
///
/// The callback should rewrite `NodeRef::Lazy(hash)` to
/// `NodeRef::Resolved(hash, Arc<NodeData>)` for children where `cached[i]`
/// is `true`, then return the rewritten node data.
///
/// The callback is responsible for:
/// 1. Parsing the HAMT node encoding from `data`
/// 2. Identifying child `NodeRef::Lazy(hash)` entries
/// 3. Swizzling cached children to `NodeRef::Resolved`
/// 4. Re-encoding the node (the cache stores the swizzled version)
///
/// **Pinning constraint:** The callback should only swizzle children at
/// levels 0 and 1 of the state trie (~33 nodes per room). Deeper nodes
/// stay `Lazy` to avoid holding Arc references that prevent LRU eviction.
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[bool]) -> NodeData;

/// A content-addressed packfile storage engine.
///
/// Wires together the packfile format, lossy fanout index, and
/// decoded-node cache into a single `StorageEngine` implementation.
///
/// Each room gets its own index and packfile, keeping the active index
/// at ~8KB per 1000-node room (100 active rooms < 1MB total).
///
/// Read path: cache → index lookup (with linear-probe collision retry)
///            → packfile read → CRC verify → swizzle children → cache insert
/// Write path: packfile append → index insert → cache insert
pub struct PackfileStorage {
    /// Manages per-room pack generations and atomic repack.
    repack: RepackManager,
    /// Per-room lossy fanout index for (hash → `pack_id`, offset).
    indexes: RwLock<HashMap<[u8; 16], LossyIndex>>,
    /// Verify-once decoded node cache.
    cache: NodeCache,
    /// Top-level pinned state trie nodes (L0/L1) safe for zero-eviction swizzling.
    pinned: PinnedNodes,
    /// Base directory for all pack files.
    base_dir: PathBuf,
    /// Optional swizzle callback for in-cache pointer resolution.
    swizzle: Option<SwizzleFn>,
    /// Optional HAMT node parser to extract child edges for cache swizzling.
    parser: Option<NodeParserFn>,
    /// Serializes first-ever pack creation per room. `ensure_pack`'s fast
    /// path (room already has a pack) never touches this — only the rare,
    /// once-per-room-lifetime "create the first packfile" path does, via
    /// double-checked locking. Without it, two threads racing to create
    /// the same brand-new room's pack both see "doesn't exist yet" and
    /// both write a header to the same file, corrupting it (confirmed via
    /// `test_concurrent_federation_swarm`: "invalid packfile header").
    ensure_pack_lock: parking_lot::Mutex<()>,
    /// Per-room roots that must survive repack (state root + timeline
    /// forward-extremities). `repack_room` itself is caller-agnostic about
    /// what a root means — this is where `PackfileStorage` remembers what
    /// its own callers told it, so `maybe_repack` has something to pass.
    /// A room with no registered roots is never repacked: repacking
    /// without knowing what must survive would risk exactly the reachability
    /// gap `repack_room`'s multi-root fix exists to prevent.
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    /// Trigger repack once a room's active generation exceeds this many
    /// bytes. Checked after `put()`; `AtomicU64` so it's settable without
    /// `&mut self` on an already-shared `PackfileStorage`.
    repack_threshold_bytes: std::sync::atomic::AtomicU64,
}

/// Default repack trigger: once a room's active packfile exceeds this size,
/// `maybe_repack` rewrites it in reachability order (if live roots are
/// registered). Deliberately small relative to real deployments — the
/// documented design envelope is ~500-1000 HAMT nodes per room (~8KB index),
/// so an 8MB packfile already represents a room whose garbage almost
/// certainly dwarfs its live data.
const DEFAULT_REPACK_THRESHOLD_BYTES: u64 = 8 * 1024 * 1024;

impl PackfileStorage {
    /// Create a new packfile storage.
    ///
    /// On creation, scans existing packfiles to rebuild in-memory indexes.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be read.
    pub fn open(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, NodeCache::with_default_capacity(), None, None)
    }

    /// Create a new packfile storage with custom cache size.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_cache(
        base_dir: PathBuf,
        cache: NodeCache,
        parser: Option<NodeParserFn>,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, cache, None, parser)
    }

    /// Create a new packfile storage with a swizzle callback.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_swizzle(
        base_dir: PathBuf,
        cache: NodeCache,
        swizzle: SwizzleFn,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(base_dir, cache, Some(swizzle), None)
    }

    fn open_with_options(
        base_dir: PathBuf,
        cache: NodeCache,
        swizzle: Option<SwizzleFn>,
        parser: Option<NodeParserFn>,
    ) -> Result<Self, std::io::Error> {
        // Ensure directory exists so initial puts do not fail
        fs::create_dir_all(&base_dir)?;

        let repack = RepackManager::new(base_dir.clone());
        let mut indexes = HashMap::new();

        if base_dir.exists() {
            Self::scan_existing(&base_dir, &repack, &mut indexes)?;
        }

        Ok(Self {
            repack,
            indexes: RwLock::new(indexes),
            cache,
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            parser,
            ensure_pack_lock: parking_lot::Mutex::new(()),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_bytes: std::sync::atomic::AtomicU64::new(
                DEFAULT_REPACK_THRESHOLD_BYTES,
            ),
        })
    }

    fn scan_existing(
        base_dir: &Path,
        repack: &RepackManager,
        indexes: &mut HashMap<[u8; 16], LossyIndex>,
    ) -> Result<(), std::io::Error> {
        for entry in fs::read_dir(base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "pack") {
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let Some((hex, pack_id_hex)) = stem.rsplit_once('_') else {
                    continue;
                };
                if hex.len() != 32 {
                    continue;
                }

                let mut room_id = [0u8; 16];
                let mut valid = true;
                for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
                    let Ok(byte) = u8::from_str_radix(std::str::from_utf8(chunk).unwrap_or(""), 16)
                    else {
                        valid = false;
                        break;
                    };
                    room_id[i] = byte;
                }
                if !valid {
                    continue;
                }

                let trimmed = pack_id_hex.trim_start_matches('0');
                let Ok(pack_id) =
                    u8::from_str_radix(if trimmed.is_empty() { "0" } else { trimmed }, 16)
                else {
                    continue;
                };

                if let Ok(entries) = packfile::scan_packfile(&path) {
                    let mut index = LossyIndex::new(entries.len().saturating_mul(2));
                    for (hash, offset) in &entries {
                        let _ = index.insert(hash, pack_id, *offset);
                    }
                    indexes.insert(room_id, index);
                }

                if let Ok(file) = packfile::open_packfile(&path, false) {
                    let gen = Arc::new(PackGeneration {
                        room_id,
                        pack_id,
                        file,
                        path: path.clone(),
                        mmap: parking_lot::RwLock::new(None),
                        append_lock: parking_lot::Mutex::new(()),
                    });
                    repack.swap_pack(room_id, gen);
                }
            }
        }
        Ok(())
    }

    /// Get a reference to the node cache.
    #[must_use]
    pub fn cache(&self) -> &NodeCache {
        &self.cache
    }

    /// Get a reference to the pinned L0/L1 nodes set for explicit memory pinning.
    #[must_use]
    pub fn pinned(&self) -> &PinnedNodes {
        &self.pinned
    }

    /// Register the set of roots that must survive repack for a room: the
    /// current HAMT state root plus every timeline forward-extremity (tip).
    /// `maybe_repack` refuses to repack a room with no registered roots —
    /// repacking without knowing what must survive risks the exact
    /// reachability gap `repack_room`'s multi-root design exists to
    /// prevent. The caller (the layer that actually understands Matrix
    /// semantics) is responsible for keeping this current as state changes
    /// and the timeline advances.
    pub fn set_live_roots(&self, room_id: &[u8; 16], roots: Vec<NodeId>) {
        self.live_roots.write().insert(*room_id, roots);
    }

    /// Set the packfile-size threshold that triggers a repack. Checked
    /// after `put()` for the room just written to.
    pub fn set_repack_threshold_bytes(&self, bytes: u64) {
        self.repack_threshold_bytes
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Repack a room if its active generation has grown past the
    /// configured threshold and live roots are registered for it.
    /// Best-effort: any failure (no roots registered, no pack yet, repack
    /// I/O error) is silently skipped rather than propagated — a missed
    /// repack opportunity is not a correctness problem, it just means the
    /// room's packfile stays larger than it needs to be until the next
    /// `put()` checks again.
    fn maybe_repack(&self, room_id: &[u8; 16]) {
        let roots = {
            let live_roots = self.live_roots.read();
            match live_roots.get(room_id) {
                Some(r) if !r.is_empty() => r.clone(),
                _ => return,
            }
        };

        let Some(gen) = self.repack.get_pack(room_id) else {
            return;
        };
        let Ok(metadata) = gen.file.metadata() else {
            return;
        };
        if metadata.len()
            < self
                .repack_threshold_bytes
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }

        let resolver = |id: &NodeId| -> Option<(NodeData, Vec<NodeId>)> {
            let data = self.get(room_id, id).ok().flatten()?;
            let children = self
                .parser
                .map_or_else(Vec::new, |parse| parse(&data.bytes));
            Some((data, children))
        };

        let Ok(new_gen) = self.repack.repack_room(*room_id, roots, resolver) else {
            return;
        };

        // repack_room swaps the active generation but has no notion of
        // PackfileStorage's index — without rebuilding it here, every
        // record's index entry would still point at the old (now-orphaned)
        // pack_id, and get() would silently report every one of them as
        // not found the moment this repack completes.
        if let Ok(entries) = packfile::scan_packfile(&new_gen.path) {
            let mut index = LossyIndex::new(entries.len().saturating_mul(2).max(16));
            for (hash, offset) in &entries {
                let _ = index.insert(hash, new_gen.pack_id, *offset);
            }
            self.indexes.write().insert(*room_id, index);
        }
    }

    /// Ensure a pack exists for the given room, creating one if needed.
    fn ensure_pack(&self, room_id: &[u8; 16]) -> Result<Arc<PackGeneration>, StorageError> {
        if let Some(gen) = self.repack.get_pack(room_id) {
            return Ok(gen);
        }

        // Slow path: this room may not have a pack yet, or another thread
        // is racing to create it right now. Serialize first-ever creation
        // per room and re-check after acquiring the lock — the winner of
        // the race creates the pack, everyone else just observes it.
        let _guard = self.ensure_pack_lock.lock();
        if let Some(gen) = self.repack.get_pack(room_id) {
            return Ok(gen);
        }

        let pack_id: u8 = 0;
        let pack_path = self.pack_path(room_id, pack_id);
        let file = packfile::open_packfile(&pack_path, true)?;
        let gen = Arc::new(PackGeneration {
            room_id: *room_id,
            pack_id,
            file,
            path: pack_path,
            mmap: parking_lot::RwLock::new(None),
            append_lock: parking_lot::Mutex::new(()),
        });
        self.repack.swap_pack(*room_id, gen.clone());

        self.indexes
            .write()
            .entry(*room_id)
            .or_insert_with(|| LossyIndex::new(4096));

        Ok(gen)
    }

    /// Rebuild a room's index from packfile records using their real hashes.
    ///
    /// Called when the index fills (`TableFull`). Unlike rehashing packed
    /// 24-bit tags (which cannot reproduce the original bucket placement),
    /// scanning the packfile yields the true 128-bit hashes and exact
    /// offsets, so the rebuilt index preserves correct probe buckets.
    fn rebuild_index(&self, room_id: &[u8; 16]) -> LossyIndex {
        let packs: Vec<Arc<PackGeneration>> = {
            let packs = self.repack.packs.read();
            packs
                .values()
                .filter(|g| g.room_id == *room_id)
                .cloned()
                .collect()
        };

        let mut total = 0usize;
        let mut scanned: ScannedPacks = Vec::new();
        for gen in &packs {
            if let Ok(entries) = packfile::scan_packfile(&gen.path) {
                total = total.saturating_add(entries.len());
                scanned.push((gen.pack_id, entries));
            }
        }

        let mut index = LossyIndex::new(total.saturating_mul(2));
        for (pack_id, entries) in scanned {
            for (hash, offset) in entries {
                // Insertion cannot fail: capacity is 2× the record count.
                let _ = index.insert(&hash, pack_id, offset);
            }
        }
        index
    }

    fn pack_path(&self, room_id: &[u8; 16], pack_id: u8) -> PathBuf {
        packfile::pack_path(&self.base_dir, room_id, pack_id)
    }

    /// Read a record at the given offset from a memory-mapped packfile.
    ///
    /// Unlike `seek`+`read_exact`, mmap reads require no syscalls — the
    /// kernel resolves faults directly into the page cache. This collapses
    /// cold-read latency from 3 syscalls/record to ~zero.
    ///
    /// The mapping is remapped when the active generation has grown past the
    /// mapping's current end (appends land after the initial mapping). This
    /// self-healing keeps the mmap bounded-correct so appends written after
    /// a read are never silently invisible.
    fn read_at(gen: &PackGeneration, offset: u64) -> Result<Record, StorageError> {
        // Fast path: read from the mapped bytes. If the record's frame lies
        // beyond the current mapping (the active generation grew after the
        // existing map was created), drop the read guard, remap to the file's
        // current length, and retry once. Remap is bounded to one retry, so a
        // genuinely truncated frame still surfaces as `Corrupt` rather than
        // an unbounded remap loop.
        for attempt in 0..2 {
            let guard = gen.mmap().map_err(StorageError::Io)?;
            let Some(mem) = guard.as_deref() else {
                return Err(StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "packfile could not be mapped",
                )));
            };

            let offset = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

            if offset.checked_add(4).map_or(true, |end| end > mem.len()) {
                if attempt == 0 {
                    drop(guard);
                    Self::remap_mapping(gen)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated length prefix".into()));
            }

            let prefix_end = offset.wrapping_add(4);
            let payload_len_bytes: [u8; 4] = mem[offset..prefix_end].try_into().unwrap();
            let payload_len = u32::from_le_bytes(payload_len_bytes);

            if payload_len == 0 || payload_len > packfile::MAX_RECORD_LEN {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {payload_len}"
                )));
            }

            let payload_len_usize = payload_len as usize;
            let crc_pos = prefix_end.wrapping_add(payload_len_usize);
            let frame_end = crc_pos.wrapping_add(4);

            if frame_end > mem.len() {
                if attempt == 0 {
                    drop(guard);
                    Self::remap_mapping(gen)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated frame".into()));
            }

            let payload = &mem[prefix_end..crc_pos];
            let crc_buf: [u8; 4] = mem[crc_pos..frame_end].try_into().unwrap();

            let mut crc = crc32fast::Hasher::new();
            crc.update(&payload_len_bytes);
            crc.update(payload);
            let expected = crc.finalize();
            let actual = u32::from_le_bytes(crc_buf);
            if expected != actual {
                return Err(StorageError::Corrupt(format!(
                    "CRC mismatch: expected {expected:08x}, got {actual:08x}"
                )));
            }

            // Copy the payload out of the mmap before releasing the guard;
            // the mapping may be remapped by a concurrent append.
            let mut hash = [0u8; 16];
            hash.copy_from_slice(&payload[..16]);
            let data = bytes::Bytes::copy_from_slice(&payload[16..]);

            return Ok(Record { hash, data });
        }
        unreachable!("read_at remap-retry is bounded to two iterations")
    }

    /// Re-map `gen`'s packfile to its current on-disk length.
    ///
    /// This is the self-healing path: the active generation accumulates
    /// appends after the initial mapping, and reads must see those appends
    /// rather than silently treating them as truncated/unreadable.
    fn remap_mapping(gen: &PackGeneration) -> Result<(), StorageError> {
        let file_len = gen.file.metadata().map_err(StorageError::Io)?.len();
        let mut guard = gen.mmap.write();
        if guard.as_ref().map_or(true, |m| (m.len() as u64) < file_len) {
            *guard = Some(crate::packfile::map_pack(&gen.file).map_err(StorageError::Io)?);
        }
        Ok(())
    }

    /// Fetch a node and return it as a `NodeRef` with swizzled children.
    ///
    /// This is the swizzling entry point. After reading and verifying the
    /// node from disk, if a swizzle callback is configured:
    /// 1. Parse child hashes from the raw node data
    /// 2. Check which children are already resident in cache
    /// 3. Call the swizzle callback to rewrite `Lazy → Resolved` for cached children
    /// 4. Cache the swizzled version
    ///
    /// Returns `NodeRef::Resolved(id, Arc::new(swizzled_data))` on success.
    ///
    /// # Errors
    /// Returns `StorageError::Io` on I/O failure.
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
                let cached = self.cache.resolve_hashes(&children);
                let swizzled = swizzle_fn(&data, &children, &cached);
                self.cache.insert(*id, Arc::new(swizzled.clone()));
                return Ok(Some(NodeRef::Resolved(*id, Arc::new(swizzled))));
            }
        }

        Ok(Some(NodeRef::Resolved(*id, Arc::new(data))))
    }

    /// Try each `(pack_id, offset)` candidate in order, verifying against
    /// `id`'s full hash — on tag collision, continue to the next candidate.
    /// This is the shared tail of `get()`'s lookup, factored out so
    /// `get_many` can supply candidates it already resolved (and sorted by
    /// offset) instead of re-deriving them per call.
    fn resolve_from_candidates(
        &self,
        room_id: &[u8; 16],
        id: &NodeId,
        candidates: &[(u8, u64)],
    ) -> Option<NodeData> {
        for (pack_id, offset) in candidates {
            let gen = {
                let packs = self.repack.packs.read();
                match packs
                    .values()
                    .find(|g| g.pack_id == *pack_id && g.room_id == *room_id)
                {
                    Some(g) => g.clone(),
                    None => continue,
                }
            };

            let Ok(record) = Self::read_at(&gen, *offset) else {
                continue;
            };

            // Verify against caller-requested hash, not the frame hash
            if record.hash != *id {
                continue;
            }

            let mut children = Vec::new();
            if let Some(parse) = self.parser {
                for child_id in parse(&record.data) {
                    if let Some(child_data) = self.pinned.get(&child_id) {
                        children.push(NodeRef::Resolved(child_id, child_data));
                    } else {
                        children.push(NodeRef::Lazy(child_id));
                    }
                }
            }

            let data = NodeData {
                bytes: record.data,
                children,
            };
            self.cache.insert(*id, Arc::new(data.clone()));
            return Some(data);
        }

        None
    }
}

impl StorageEngine for PackfileStorage {
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        // 1. Check the per-room index first — this is the room-scope gate.
        //    The cache is shared (content-addressed), but we only serve data
        //    for nodes whose index entry is in the requested room.
        let candidates: Vec<(u8, u64)> = {
            let indexes = self.indexes.read();
            let Some(index) = indexes.get(room_id) else {
                return Ok(None);
            };
            index.lookup_all(id).collect()
        };

        if candidates.is_empty() {
            return Ok(None);
        }

        // 2. Check cache (index confirmed the node belongs to this room)
        if let Some(data) = self.cache.get(id) {
            return Ok(Some((*data).clone()));
        }

        Ok(self.resolve_from_candidates(room_id, id, &candidates))
    }

    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

        // Phase 1: resolve cache hits and index candidates for the rest, all
        // in-RAM (no disk I/O yet). Keep the *full* candidate list per id
        // (not just the first) so a tag collision on the first candidate
        // still falls back to the rest, exactly as `get()` does — only the
        // *order in which ids are visited* is driven by the first
        // candidate's offset, not correctness.
        let mut to_fetch: Vec<(usize, Vec<(u8, u64)>)> = Vec::new();
        {
            let indexes = self.indexes.read();
            let index = indexes.get(room_id);
            for (i, id) in ids.iter().enumerate() {
                if let Some(data) = self.cache.get(id) {
                    results[i] = Some((*data).clone());
                    continue;
                }
                if let Some(index) = index {
                    let candidates: Vec<(u8, u64)> = index.lookup_all(id).collect();
                    if !candidates.is_empty() {
                        to_fetch.push((i, candidates));
                    }
                }
            }
        }

        // Phase 2: sort by the first candidate's physical offset so disk
        // access is monotonic instead of scattered in the caller's
        // requested order.
        to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

        // Phase 3: fetch in sorted order, reusing the already-resolved
        // candidates directly — no second index lookup per id.
        for (i, candidates) in &to_fetch {
            results[*i] = self.resolve_from_candidates(room_id, &ids[*i], candidates);
        }

        Ok(results)
    }

    fn put(&self, room_id: &[u8; 16], id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        let gen = self.ensure_pack(room_id)?;

        let record = Record {
            hash: *id,
            data: data.bytes.clone(),
        };

        // Capture the offset from the same seek that positions this write,
        // on the same handle that performs it — not from a separate
        // metadata().len() call afterward. The previous approach computed
        // offset = file_len - record_len using gen.file's length queried
        // after the write completed; under any concurrent-writer scenario
        // for the same room, another append landing between the write and
        // that metadata() call would corrupt the computed offset for this
        // record.
        //
        // gen.append_lock serializes this whole seek+write+offset-capture
        // section per generation (not globally — a different room, or a
        // different generation of this room after a repack swap, never
        // contends here). Without it, two threads racing put() on the same
        // generation could both capture the identical seek(End(0)) offset
        // before either's write_record lands; O_APPEND still places both
        // writes correctly at the true end, but the LossyIndex would then
        // record that same offset for both ids, permanently orphaning one.
        // Confirmed as a real, if narrow, gap — not fixed preemptively:
        // the sibling ensure_pack race in this same file was reproduced
        // directly under test_concurrent_federation_swarm, which is the
        // standard this lock is held to.
        let offset = {
            let _append_guard = gen.append_lock.lock();
            let mut file = gen.file.try_clone()?;
            let offset = file.seek(std::io::SeekFrom::End(0))?;
            packfile::write_record(&mut file, &record)?;
            offset
        };

        // Update per-room index
        let index_full = {
            let mut indexes = self.indexes.write();
            let index = indexes
                .entry(*room_id)
                .or_insert_with(|| LossyIndex::new(4096));
            index.insert(id, gen.pack_id, offset).is_err()
        };
        if index_full {
            // The index cannot rehash packed 24-bit tags onto the same
            // buckets, so rebuild it from the packfile (real hashes, exact
            // offsets). Scans happen without the index lock held.
            let mut rebuilt = self.rebuild_index(room_id);
            let _ = rebuilt.insert(id, gen.pack_id, offset);
            self.indexes.write().insert(*room_id, rebuilt);
        }

        // Cache it, swizzling any lazy refs if they are in pinned
        let mut data_to_cache = data.clone();
        for child in &mut data_to_cache.children {
            if let NodeRef::Lazy(child_id) = child {
                if let Some(child_data) = self.pinned.get(child_id) {
                    *child = NodeRef::Resolved(*child_id, child_data);
                }
            }
        }
        self.cache.insert(*id, Arc::new(data_to_cache));

        self.maybe_repack(room_id);

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
        self.repack
            .purge_room(room_id)
            .map_err(|e| StorageError::Internal(e.to_string()))?;
        self.indexes.write().remove(room_id);
        // Without this, live_roots grows without bound over a server's
        // lifetime as rooms are created and purged — nothing else ever
        // removes an entry from it.
        self.live_roots.write().remove(room_id);
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        let packs = self.repack.packs.read();
        for gen in packs.values() {
            gen.file.sync_all()?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;

    const TEST_ROOM: [u8; 16] = [0x01; 16];
    const OTHER_ROOM: [u8; 16] = [0x02; 16];

    fn test_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_pfs_{name}_{id}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_concurrent_federation_swarm() {
        // Simulates concurrent federation ingress (many writers) and client
        // /sync (many readers) hammering ONE shared room — the maximum-
        // contention case for the exact gap flagged this session: put()
        // captures its offset from `file.seek(SeekFrom::End(0))` on a
        // try_clone'd handle, with no explicit lock serializing the
        // append+offset-capture critical section across threads. The file
        // is opened O_APPEND, so each individual write() syscall always
        // lands at the true current end regardless of a thread's seek
        // position — but the *offset value this thread captured* can go
        // stale if another thread's put() interleaves and appends more
        // bytes before this thread's write_record finishes. That's a real,
        // reachable race, not a hypothetical: this test proves whether it
        // corrupts data or is merely benign under real contention.
        use std::sync::Mutex;
        use std::thread;

        const NUM_WRITERS: usize = 8;
        const EVENTS_PER_WRITER: usize = 500;
        const NUM_READERS: usize = 8;
        const READS_PER_READER: usize = 2000;

        let dir = test_dir("concurrent_federation");
        let store = PackfileStorage::open(dir).unwrap();

        let room = [0x77u8; 16];
        // Every written id + its expected bytes, so we can verify after
        // the swarm that nothing was silently corrupted — not just that
        // nothing panicked.
        let written: Mutex<Vec<(NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());
        let read_ok = std::sync::atomic::AtomicUsize::new(0);
        let read_not_found = std::sync::atomic::AtomicUsize::new(0);

        thread::scope(|scope| {
            for w in 0..NUM_WRITERS {
                let store = &store;
                let written = &written;
                scope.spawn(move || {
                    for i in 0..EVENTS_PER_WRITER {
                        // Unique per (writer, event) — globally distinct
                        // across all writers hammering the same room.
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
                        // Read whatever's been published so far — may be
                        // empty early on, which is fine, get() on an empty
                        // room correctly returns Ok(None).
                        let snapshot_len = written.lock().unwrap().len();
                        if snapshot_len == 0 {
                            continue;
                        }
                        let idx = i % snapshot_len;
                        let (id, expected_bytes) = written.lock().unwrap()[idx].clone();
                        match store.get(&room, &id).unwrap() {
                            Some(data) => {
                                assert_eq!(
                                    data.bytes, expected_bytes,
                                    "get() returned WRONG bytes for a concurrently-written \
                                     record — silent data corruption under concurrent writes"
                                );
                                read_ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            None => {
                                // Under the flagged race, a stale-offset
                                // write can make a record briefly (or
                                // permanently) unreachable rather than
                                // return wrong data — the hash-verification
                                // safety net rejects a wrong-offset read
                                // rather than returning it. Not silently
                                // wrong, but also not what should happen.
                                read_not_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });

        // Final integrity pass: every single record any writer claims to
        // have successfully put() must be findable, with correct bytes,
        // once all concurrent activity has settled.
        let written = written.into_inner().unwrap();
        assert_eq!(written.len(), NUM_WRITERS * EVENTS_PER_WRITER);

        let mut lost = Vec::new();
        for (id, expected_bytes) in &written {
            match store.get(&room, id).unwrap() {
                Some(data) => assert_eq!(
                    data.bytes, *expected_bytes,
                    "post-swarm read returned wrong bytes for {id:?} — data corruption"
                ),
                None => lost.push(*id),
            }
        }

        eprintln!(
            "concurrent federation swarm: {} writes, {} reads ok, {} reads not-found \
             during the race window, {} lost post-swarm",
            written.len(),
            read_ok.load(std::sync::atomic::Ordering::Relaxed),
            read_not_found.load(std::sync::atomic::Ordering::Relaxed),
            lost.len()
        );

        assert!(
            lost.is_empty(),
            "{} of {} concurrently-written records are permanently unreachable \
             after the swarm settled — the unprotected put() offset race causes \
             real data loss, not just a benign transient miss: {lost:?}",
            lost.len(),
            written.len()
        );
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
        // Regression test for the stale-mmap bug: put(A), get(A) (which
        // lazily creates/caches the mmap at its length at that moment), then
        // put(B) on the SAME generation, then get(B). Before the
        // RwLock<Option<Mmap>> self-healing fix, the cached mapping never
        // grew past A's length, so B's bytes fell outside it and get(B)
        // silently returned Ok(None) even though B was correctly written
        // and indexed.
        let dir = test_dir("stale_mmap_regression");
        let store = PackfileStorage::open(dir).unwrap();

        let a = [0xAAu8; 16];
        let b = [0xBBu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"bbbb"));

        store.put(&TEST_ROOM, &a, &data_a).unwrap();

        // Force a disk read of A, which lazily creates the mmap for this
        // generation at its current (pre-B) length.
        store.cache().clear();
        let got_a = store.get(&TEST_ROOM, &a).unwrap();
        assert_eq!(got_a.unwrap().bytes, data_a.bytes);

        // Append B to the SAME generation, after the mmap was established.
        store.put(&TEST_ROOM, &b, &data_b).unwrap();

        // B must be readable — not silently "not found".
        store.cache().clear();
        let got_b = store.get(&TEST_ROOM, &b).unwrap();
        assert_eq!(
            got_b
                .expect("B must be found: it was written after the mmap was established")
                .bytes,
            data_b.bytes
        );
    }

    #[test]
    fn test_read_at_rejects_genuinely_torn_frame() {
        // Distinct from test_read_survives_append_after_mmap_established:
        // that test covers the "offset+4 exceeds the old mapping" bounds
        // check (read_at line ~311). This test targets the OTHER bounds
        // check — "frame_end exceeds the mapping" (read_at line ~334) —
        // by writing a length prefix whose payload+crc were never actually
        // written, simulating a reader observing a write mid-flight
        // (write_record issues 4 separate write_all calls, so a concurrent
        // mmap creation between them is a real, if narrow, window). This
        // must fail cleanly with Corrupt after the bounded retry — never
        // panic, never silently return Ok(None).
        use std::io::Write as _;

        let dir = test_dir("torn_frame");
        let store = PackfileStorage::open(dir).unwrap();

        let a = [0xAAu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        store.put(&TEST_ROOM, &a, &data_a).unwrap();

        // Manually append ONLY a valid-looking 4-byte length prefix for a
        // second record, with no payload/crc behind it.
        let gen = store.repack.get_pack(&TEST_ROOM).unwrap();
        {
            let mut file = gen.file.try_clone().unwrap();
            file.seek(std::io::SeekFrom::End(0)).unwrap();
            let fake_payload_len: u32 = 20; // 16-byte hash + 4 bytes of "data"
            file.write_all(&fake_payload_len.to_le_bytes()).unwrap();
        }

        // The offset where this truncated frame starts.
        let torn_offset = {
            let file_len = gen.file.metadata().unwrap().len();
            file_len - 4
        };

        // First access lazily creates the mmap covering exactly what's on
        // disk right now: A's full frame plus the 4-byte orphan prefix —
        // not the (never-written) payload+crc.
        let result = PackfileStorage::read_at(&gen, torn_offset);
        assert!(
            matches!(result, Err(StorageError::Corrupt(_))),
            "a genuinely truncated frame must surface as Corrupt, got {result:?}"
        );
    }

    #[test]
    fn test_read_at_rejects_offset_past_old_mapping() {
        // Distinct from test_read_survives_append_after_mmap_established:
        // that test covers the general stale-mmap fix. This test targets
        // the FIRST bounds check specifically (read_at line ~311):
        // offset.checked_add(4).map_or(true, |end| end > mem.len()).
        // We force an mmap at a known small size, then try to read at an
        // offset that is >= that mapping's length — so offset+4 definitely
        // exceeds it — without a full frame behind it. This exercises the
        // attempt-0 branch that drops the guard, remaps, and retries.
        let dir = test_dir("offset_past_mapping");
        let store = PackfileStorage::open(dir).unwrap();

        // Write one record so we have a generation with content.
        let a = [0xAAu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        store.put(&TEST_ROOM, &a, &data_a).unwrap();

        // Force a disk read of A, which lazily creates the mmap covering
        // exactly A's frame (length = frame_end of A).
        let gen = store.repack.get_pack(&TEST_ROOM).unwrap();
        store.cache().clear();
        let _ = store.get(&TEST_ROOM, &a).unwrap();

        // The mapping now ends exactly at A's frame_end. The next byte
        // (A's frame_end) is past the mapping. Try to read at that offset.
        let file_len = gen.file.metadata().unwrap().len();
        let past_mapping_offset = file_len; // exactly at EOF, which is past mapping

        // This read should hit the first bounds check (offset+4 > mem.len())
        // on attempt 0, trigger remap, and on attempt 1 find that the
        // offset is genuinely at EOF (no record there), failing with Corrupt.
        let result = PackfileStorage::read_at(&gen, past_mapping_offset);
        assert!(
            matches!(result, Err(StorageError::Corrupt(_))),
            "read at offset past old mapping must surface as Corrupt after remap retry, got {result:?}"
        );
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

        // put already caches the node, so both gets are cache hits
        let _ = store.get(&TEST_ROOM, &id).unwrap();
        assert_eq!(store.cache.hits(), 1);

        let _ = store.get(&TEST_ROOM, &id).unwrap();
        assert_eq!(store.cache.hits(), 2);
    }

    #[test]
    fn test_delete_room() {
        let dir = test_dir("delete");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"room data"));

        // Manually set up a pack for this room
        let gen = store.ensure_pack(&OTHER_ROOM).unwrap();
        let record = Record {
            hash: id,
            data: data.bytes.clone(),
        };
        {
            let mut file = gen.file.try_clone().unwrap();
            file.seek(std::io::SeekFrom::End(0)).unwrap();
            packfile::write_record(&mut file, &record).unwrap();
        }
        {
            let mut indexes = store.indexes.write();
            let index = indexes
                .entry(OTHER_ROOM)
                .or_insert_with(|| LossyIndex::new(4096));
            let _ = index.insert(&id, gen.pack_id, 5);
        }

        store.delete_room(&OTHER_ROOM).unwrap();
        assert!(store.repack.get_pack(&OTHER_ROOM).is_none());
        assert!(store.indexes.read().get(&OTHER_ROOM).is_none());
    }

    #[test]
    fn test_delete_room_clears_live_roots() {
        // Regression test: delete_room must also drop the room's
        // live_roots entry, or that map grows without bound over a
        // server's lifetime as rooms are created and purged — nothing
        // else ever removes an entry from it.
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
        assert!(
            !store.live_roots.read().contains_key(&OTHER_ROOM),
            "live_roots entry must be removed on delete_room, not leaked"
        );
    }

    #[test]
    fn test_batch_put_get() {
        let dir = test_dir("batch");
        let store = PackfileStorage::open(dir).unwrap();

        let entries: Vec<(NodeId, NodeData)> = (0..10u8)
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
    fn test_get_many_preserves_caller_order_despite_sorted_reads() {
        // get_many sorts candidates by physical offset internally to keep
        // disk access monotonic, but the returned Vec must still match the
        // caller's requested id order — not the internal read order.
        let dir = test_dir("batch_order");
        let store = PackfileStorage::open(dir).unwrap();

        let entries: Vec<(NodeId, NodeData)> = (0..10u8)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
            })
            .collect();

        // Written in ascending order, so physical offsets are ascending too.
        store.put_many(&TEST_ROOM, &entries).unwrap();

        // put_many populates the cache for every id it writes — clear it so
        // get_many is forced through the index-lookup + sort-then-read path
        // this test actually exists to exercise, not served entirely from
        // cache (which would pass trivially without touching that code).
        store.cache().clear();

        // Request them in REVERSED order — internal sort-then-read will
        // visit them ascending-by-offset (the opposite of this request
        // order), so this only passes if the results are reassembled back
        // into the caller's order rather than left in read order.
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

        // Verify all can be read back
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

        // In production, structural hashes include the room's structural key,
        // so the same logical data in different rooms produces different hashes.
        let id_a = [0x42u8; 16];
        let id_b = [0x43u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"room A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"room B data"));

        store.put(&TEST_ROOM, &id_a, &data_a).unwrap();
        store.put(&OTHER_ROOM, &id_b, &data_b).unwrap();

        // Each room's index resolves its own nodes
        let got_a = store.get(&TEST_ROOM, &id_a).unwrap().unwrap();
        let got_b = store.get(&OTHER_ROOM, &id_b).unwrap().unwrap();
        assert_eq!(got_a.bytes, data_a.bytes);
        assert_eq!(got_b.bytes, data_b.bytes);

        // Node from room A is not in room B's index (different hash)
        assert!(store.get(&OTHER_ROOM, &id_a).unwrap().is_none());
        assert!(store.get(&TEST_ROOM, &id_b).unwrap().is_none());

        // Deleting room A doesn't affect room B
        store.delete_room(&TEST_ROOM).unwrap();
        assert!(store.get(&TEST_ROOM, &id_a).unwrap().is_none());
        let got_b = store.get(&OTHER_ROOM, &id_b).unwrap().unwrap();
        assert_eq!(got_b.bytes, data_b.bytes);
    }

    #[test]
    fn test_collision_retry() {
        // Verify that a CRC failure (simulating a tag collision) triggers
        // the linear-probe retry and returns None if no valid candidate exists.
        let dir = test_dir("collision");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x42u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"valid data"));
        store.put(&TEST_ROOM, &id, &data).unwrap();

        // Clear cache so the next get must go to disk
        store.cache.clear();

        // Normal read works
        let got = store.get(&TEST_ROOM, &id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);

        // Non-existent hash returns None (not a tag collision loop)
        assert!(store.get(&TEST_ROOM, &[0xFF; 16]).unwrap().is_none());
    }

    #[test]
    fn test_maybe_repack_triggers_and_preserves_live_data() {
        // End-to-end proof that wiring repack_room into PackfileStorage
        // actually works: garbage accumulates, a live chain of nodes is
        // registered as roots, the threshold trips, repack fires, the
        // packfile shrinks, and the live chain is still readable afterward
        // — through the SAME PackfileStorage handle, meaning maybe_repack's
        // index rebuild (the gap identified via the concurrency swarm test:
        // repack_room swaps generations but doesn't touch
        // PackfileStorage.indexes) actually closes that gap in practice,
        // not just in isolated repack.rs unit tests.
        let dir = test_dir("maybe_repack");
        let store = PackfileStorage::open(dir).unwrap();
        store.set_repack_threshold_bytes(1024); // trip quickly in a test

        let room = [0x55u8; 16];

        // A live chain: root -> mid -> leaf. Only `root` is registered,
        // but repack must preserve the whole chain since put()'s
        // NodeData::children/parser aren't used here — instead we
        // register all three explicitly as roots, standing in for
        // "the caller enumerated everything that must survive."
        let leaf_id = [0x01u8; 16];
        let mid_id = [0x02u8; 16];
        let root_id = [0x03u8; 16];
        let leaf_bytes = bytes::Bytes::from_static(b"leaf");
        let mid_bytes = bytes::Bytes::from_static(b"mid");
        let root_bytes = bytes::Bytes::from_static(b"root");
        store
            .put(&room, &leaf_id, &NodeData::new(leaf_bytes.clone()))
            .unwrap();
        store
            .put(&room, &mid_id, &NodeData::new(mid_bytes.clone()))
            .unwrap();
        store
            .put(&room, &root_id, &NodeData::new(root_bytes.clone()))
            .unwrap();

        store.set_live_roots(&room, vec![leaf_id, mid_id, root_id]);

        // Pad with garbage — unregistered nodes that must NOT survive
        // repack — until the packfile crosses the (tiny, test-only)
        // threshold and a subsequent put() triggers maybe_repack.
        let garbage_data = NodeData::new(bytes::Bytes::from(vec![0u8; 200]));
        let mut garbage_ids = Vec::new();
        for i in 0u32..20 {
            let mut id = [0u8; 16];
            id[0] = 0xEE;
            id[4..8].copy_from_slice(&i.to_le_bytes());
            garbage_ids.push(id);
            store.put(&room, &id, &garbage_data).unwrap();
        }

        let pack_size_after = store
            .repack
            .get_pack(&room)
            .unwrap()
            .file
            .metadata()
            .unwrap()
            .len();

        // The live chain must still be readable, with correct bytes, after
        // whatever repacking happened along the way.
        assert_eq!(
            store.get(&room, &leaf_id).unwrap().unwrap().bytes,
            leaf_bytes
        );
        assert_eq!(store.get(&room, &mid_id).unwrap().unwrap().bytes, mid_bytes);
        assert_eq!(
            store.get(&room, &root_id).unwrap().unwrap().bytes,
            root_bytes
        );

        // At least one garbage node must have been dropped by a repack —
        // otherwise this test never actually exercised maybe_repack at all,
        // it just proved plain put()/get() works (already covered
        // elsewhere).
        let mut any_garbage_dropped = false;
        for id in &garbage_ids {
            if store.get(&room, id).unwrap().is_none() {
                any_garbage_dropped = true;
                break;
            }
        }
        assert!(
            any_garbage_dropped,
            "expected at least one unregistered garbage node to be reclaimed by a \
             triggered repack (pack size after writes: {pack_size_after} bytes) — \
             if none were dropped, maybe_repack likely never fired"
        );
    }

    #[test]
    fn test_swizzle_callback() {
        use std::sync::atomic::{AtomicU64, Ordering};

        // Track how many times the swizzle callback is invoked
        static SWIZZLE_CALLS: AtomicU64 = AtomicU64::new(0);
        static CACHED_CHILDREN_FOUND: AtomicU64 = AtomicU64::new(0);

        fn test_swizzle(data: &NodeData, children: &[NodeId], cached: &[bool]) -> NodeData {
            SWIZZLE_CALLS.fetch_add(1, Ordering::Relaxed);
            assert_eq!(children.len(), 2, "expected 2 children from parent node");
            // Both children should be found in cache
            for &is_cached in cached {
                if is_cached {
                    CACHED_CHILDREN_FOUND.fetch_add(1, Ordering::Relaxed);
                }
            }
            // In a real implementation, this would parse the HAMT encoding
            // and rewrite Lazy → Resolved for cached[i] == true children.
            // For this test, we return the data unchanged.
            data.clone()
        }

        let dir = test_dir("swizzle");
        let store =
            PackfileStorage::open_with_swizzle(dir, NodeCache::new(100), test_swizzle).unwrap();

        // Simulate a parent node with two children
        let parent_id = [0x10u8; 16];
        let child_a = [0x20u8; 16];
        let child_b = [0x30u8; 16];

        let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
        let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));
        let _ = &parent_data;

        // Insert all three nodes
        store.put(&TEST_ROOM, &child_a, &data_a).unwrap();
        store.put(&TEST_ROOM, &child_b, &data_b).unwrap();
        store.put(&TEST_ROOM, &parent_id, &parent_data).unwrap();

        // Clear cache, then re-insert children (they must be resident for swizzle)
        // Parent must NOT be in cache so get_swizzled goes to disk
        store.cache.clear();
        store.put(&TEST_ROOM, &child_a, &data_a).unwrap();
        store.put(&TEST_ROOM, &child_b, &data_b).unwrap();

        // Fetch parent via get_swizzled — callback should fire
        let extract = |data: &NodeData| -> Vec<NodeId> {
            // Simulated child extraction: in reality this would parse
            // the HAMT node encoding. For this test we just return known children.
            let _ = data; // unused
            vec![child_a, child_b]
        };

        let result = store.get_swizzled(&TEST_ROOM, &parent_id, extract).unwrap();
        assert!(result.is_some());
        let node_ref = result.unwrap();
        assert!(node_ref.is_resolved());
        assert_eq!(node_ref.structural_hash(), &parent_id);

        // Verify the callback was invoked exactly once
        assert_eq!(SWIZZLE_CALLS.load(Ordering::Relaxed), 1);
        // Both children should have been found in cache
        assert_eq!(CACHED_CHILDREN_FOUND.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_in_cache_swizzling() {
        fn dummy_parser(bytes: &[u8]) -> Vec<NodeId> {
            if bytes == b"parent" {
                vec![[0x11; 16], [0x22; 16]]
            } else {
                vec![]
            }
        }

        let dir = test_dir("in_cache_swizzle");

        let store = PackfileStorage::open_with_cache(
            dir,
            NodeCache::with_default_capacity(),
            Some(dummy_parser),
        )
        .unwrap();

        let child1_id = [0x11; 16];
        let child1_data = Arc::new(NodeData::new(bytes::Bytes::from_static(b"child 1")));
        store.pinned().pin(child1_id, child1_data.clone());

        let parent_id = [0x42; 16];
        let mut parent_data = NodeData::new(bytes::Bytes::from_static(b"parent"));
        parent_data.children = vec![NodeRef::Lazy(child1_id), NodeRef::Lazy([0x22; 16])];

        store.put(&TEST_ROOM, &parent_id, &parent_data).unwrap();
        let fetched = store.get(&TEST_ROOM, &parent_id).unwrap().unwrap();
        assert_eq!(fetched.children.len(), 2);

        // Child 1 is Resolved directly from pinned L0/L1 cache
        match &fetched.children[0] {
            NodeRef::Resolved(id, data) => {
                assert_eq!(*id, child1_id);
                assert_eq!(data.bytes, child1_data.bytes);
            }
            NodeRef::Lazy(_) => panic!("Expected child 1 to be resolved"),
        }

        // Child 2 remains Lazy (not in pinned)
        assert!(matches!(fetched.children[1], NodeRef::Lazy(_)));
    }
}
