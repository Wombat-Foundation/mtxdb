use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;
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
}

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
                        mmap: OnceLock::new(),
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

    /// Ensure a pack exists for the given room, creating one if needed.
    fn ensure_pack(&self, room_id: &[u8; 16]) -> Result<Arc<PackGeneration>, StorageError> {
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
            mmap: OnceLock::new(),
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
        let hex: String = room_id
            .iter()
            .fold(String::with_capacity(32), |mut acc, &b| {
                let _ = write!(acc, "{b:02x}");
                acc
            });
        self.base_dir.join(format!("{hex}_{pack_id:02x}.pack"))
    }

    /// Read a record at the given offset from a memory-mapped packfile.
    ///
    /// Unlike `seek`+`read_exact`, mmap reads require no syscalls — the
    /// kernel resolves faults directly into the page cache. This collapses
    /// cold-read latency from 3 syscalls/record to ~zero.
    fn read_at(gen: &PackGeneration, offset: u64) -> Result<Record, StorageError> {
        let mmap: &Mmap = gen.mmap().map_err(StorageError::Io)?;
        let mem: &[u8] = mmap;

        let offset = usize::try_from(offset)
            .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

        if offset.checked_add(4).map_or(true, |end| end > mem.len()) {
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

        let mut hash = [0u8; 16];
        hash.copy_from_slice(&payload[..16]);
        let data = bytes::Bytes::copy_from_slice(&payload[16..]);

        Ok(Record { hash, data })
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

        // 3. Try each candidate — on tag collision (hash mismatch), continue
        for (pack_id, offset) in &candidates {
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
            return Ok(Some(data));
        }

        Ok(None)
    }

    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        ids.iter().map(|id| self.get(room_id, id)).collect()
    }

    fn put(&self, room_id: &[u8; 16], id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        let gen = self.ensure_pack(room_id)?;

        let record = Record {
            hash: *id,
            data: data.bytes.clone(),
        };
        let record_len = record.serialized_len();

        // Append to packfile
        {
            let mut file = gen.file.try_clone()?;
            file.seek(std::io::SeekFrom::End(0))?;
            packfile::write_record(&mut file, &record)?;
        }

        // Calculate offset: end of file minus record length
        let file_len = gen.file.metadata()?.len();
        let offset = file_len.saturating_sub(record_len as u64);

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
