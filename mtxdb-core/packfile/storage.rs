use std::collections::HashMap;
use std::fs;
use std::io::Seek;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::index::LossyIndex;
use crate::packfile::{self, Record};
use crate::repack::RepackManager;
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{NodeData, NodeId, NodeRef, StorageEngine, StorageError};

pub type NodeParserFn = fn(&[u8]) -> Vec<NodeId>;

/// Callback that rewrites a node's child pointers for in-cache swizzling.
///
/// The callback is responsible for:
/// 1. Parsing the HAMT node encoding from `data`
/// 2. Identifying child `NodeRef::Lazy(hash)` entries
/// 3. Swizzling cached children to `NodeRef::Resolved`
/// 4. Re-encoding the node (the cache stores the swizzled version)
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[Option<Arc<NodeData>>]) -> NodeData;

/// A content-addressed packfile storage engine backed by a global shard pool.
///
/// All rooms share a small pool of shard files (~4, each ~2GB), keeping
/// file descriptor usage constant regardless of room count.
///
/// Read path: cache → index lookup (with linear-probe collision retry)
///            → shard read → CRC verify → swizzle children → cache insert
/// Write path: shard append → index insert → cache insert
pub struct PackfileStorage {
    /// Global shard pool shared across all rooms.
    shards: ShardPool,
    /// Per-room lossy fanout index for (hash → `shard_id`, offset).
    indexes: RwLock<HashMap<[u8; 16], LossyIndex>>,
    /// Verify-once decoded node cache.
    cache: NodeCache,
    /// Top-level pinned state trie nodes (L0/L1) safe for zero-eviction swizzling.
    pinned: PinnedNodes,
    /// Base directory for all shard files.
    base_dir: PathBuf,
    /// Optional swizzle callback for in-cache pointer resolution.
    swizzle: Option<SwizzleFn>,
    /// Optional HAMT node parser to extract child edges for cache swizzling.
    parser: Option<NodeParserFn>,
    /// Manages per-room repack lifecycle.
    repack: RepackManager,
    /// Serializes `put` + `maybe_repack` per room.
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    /// Per-room roots that must survive repack.
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    /// Trigger repack once a room's index exceeds this many entries.
    repack_threshold_entries: AtomicU64,
}

/// Default repack trigger: once a room's index exceeds this many entries,
/// `maybe_repack` rewrites it in reachability order (if live roots are
/// registered). ~2x the expected steady-state entry count for a room.
const DEFAULT_REPACK_THRESHOLD_ENTRIES: u64 = 2048;

impl PackfileStorage {
    /// Create a new packfile storage.
    ///
    /// On creation, scans existing shard files to rebuild in-memory indexes.
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
        fs::create_dir_all(&base_dir)?;

        let shards = ShardPool::open(base_dir.clone())?;
        let mut indexes = HashMap::new();

        // Scan all shard files, building per-room indexes
        for shard_id in 0..shard::MAX_SHARDS as u8 {
            let path = ShardPool::shard_path(&base_dir, shard_id);
            if !path.exists() {
                continue;
            }
            let entries = match packfile::scan_and_recover_packfile(&path) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("warning: recovery scan failed for shard {shard_id:02x}, falling back to partial scan: {e}");
                    packfile::scan_packfile(&path)?
                }
            };
            let mut room_entries: HashMap<[u8; 16], Vec<([u8; 16], u64)>> = HashMap::new();
            for (room_id, hash, offset) in &entries {
                room_entries
                    .entry(*room_id)
                    .or_default()
                    .push((*hash, *offset));
            }
            for (room_id, records) in room_entries {
                let index = indexes
                    .entry(room_id)
                    .or_insert_with(|| LossyIndex::new(records.len().saturating_mul(2).max(16)));
                for (hash, offset) in &records {
                    let _ = index.insert(hash, shard_id, *offset);
                }
            }
        }

        Ok(Self {
            shards,
            indexes: RwLock::new(indexes),
            cache,
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            parser,
            repack: RepackManager::new(),
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
        })
    }

    /// Get a reference to the node cache.
    #[must_use]
    pub fn cache(&self) -> &NodeCache {
        &self.cache
    }

    /// Get a reference to the pinned L0/L1 nodes set.
    #[must_use]
    pub fn pinned(&self) -> &PinnedNodes {
        &self.pinned
    }

    /// Get a reference to the per-room indexes.
    #[must_use]
    pub fn indexes(&self) -> &RwLock<HashMap<[u8; 16], LossyIndex>> {
        &self.indexes
    }

    /// Register the set of roots that must survive repack for a room.
    pub fn set_live_roots(&self, room_id: &[u8; 16], roots: Vec<NodeId>) {
        self.live_roots.write().insert(*room_id, roots);
    }

    /// Set the index entry count threshold that triggers a repack.
    pub fn set_repack_threshold_entries(&self, entries: u64) {
        self.repack_threshold_entries
            .store(entries, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the packfile-size threshold (legacy API, now maps to entries).
    pub fn set_repack_threshold_bytes(&self, bytes: u64) {
        // Approximate: 1 entry ~ 200 bytes on average
        let entries = bytes / 200;
        self.set_repack_threshold_entries(entries.max(1));
    }

    /// Repack a room if its index has grown past the configured threshold
    /// and live roots are registered.
    fn maybe_repack(&self, room_id: &[u8; 16]) {
        let roots = {
            let live_roots = self.live_roots.read();
            match live_roots.get(room_id) {
                Some(r) if !r.is_empty() => r.clone(),
                _ => return,
            }
        };

        let index_len = {
            let indexes = self.indexes.read();
            match indexes.get(room_id) {
                Some(idx) => idx.len() as u64,
                _ => return,
            }
        };

        if index_len
            < self
                .repack_threshold_entries
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

        let Ok(new_entries) = self
            .repack
            .repack_room(*room_id, roots, resolver, &self.shards)
        else {
            return;
        };

        // Rebuild index from repacked entries
        let mut index = LossyIndex::new(new_entries.len().saturating_mul(2).max(16));
        for (_room, hash, shard_id, offset) in &new_entries {
            let _ = index.insert(hash, *shard_id, *offset);
        }
        self.indexes.write().insert(*room_id, index);
    }

    /// Get or create a per-room mutex for serializing `put` and `maybe_repack`.
    fn put_mutex(&self, room_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.put_locks.lock();
        locks.entry(*room_id).or_default().clone()
    }

    /// Read a record at the given offset from a memory-mapped shard.
    fn read_at(shard: &Shard, offset: u64) -> Result<Record, StorageError> {
        ShardPool::read_at(shard, offset)
    }

    /// Rebuild a room's index from shard records.
    fn rebuild_index(&self, room_id: &[u8; 16]) -> LossyIndex {
        let mut total = 0usize;
        let mut scanned: Vec<(u8, Vec<([u8; 16], u64)>)> = Vec::new();

        for shard_id in 0..shard::MAX_SHARDS as u8 {
            let path = ShardPool::shard_path(&self.base_dir, shard_id);
            if !path.exists() {
                continue;
            }
            if let Ok(entries) = packfile::scan_packfile(&path) {
                let room_entries: Vec<_> = entries
                    .into_iter()
                    .filter(|(rid, _, _)| rid == room_id)
                    .map(|(_, hash, offset)| (hash, offset))
                    .collect();
                total = total.saturating_add(room_entries.len());
                scanned.push((shard_id, room_entries));
            }
        }

        let mut index = LossyIndex::new(total.saturating_mul(2));
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                let _ = index.insert(&hash, shard_id, offset);
            }
        }
        index
    }

    /// Fetch a node and return it as a `NodeRef` with swizzled children.
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

    /// Try each `(shard_id, offset)` candidate in order, verifying against
    /// `id`'s full hash — on tag collision, continue to the next candidate.
    fn resolve_from_candidates(
        &self,
        _room_id: &[u8; 16],
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
}

impl StorageEngine for PackfileStorage {
    fn get(&self, room_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        // 1. Check the per-room index first
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

        // 2. Check cache
        if let Some(data) = self.cache.get(id) {
            return Ok(Some((*data).clone()));
        }

        self.resolve_from_candidates(room_id, id, &candidates)
    }

    fn get_many(
        &self,
        room_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

        let mut to_fetch: Vec<(usize, Vec<(u8, u64)>)> = Vec::new();
        {
            let indexes = self.indexes.read();
            let index = indexes.get(room_id);
            for (i, id) in ids.iter().enumerate() {
                if let Some(index) = index {
                    let candidates: Vec<(u8, u64)> = index.lookup_all(id).collect();
                    if !candidates.is_empty() {
                        if let Some(data) = self.cache.get(id) {
                            results[i] = Some((*data).clone());
                        } else {
                            to_fetch.push((i, candidates));
                        }
                    }
                }
            }
        }

        to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

        for (i, candidates) in &to_fetch {
            results[*i] = self.resolve_from_candidates(room_id, &ids[*i], candidates)?;
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

        let (shard_id, offset) = {
            let shard = self.shards.active_shard();
            let _append_guard = shard.append_lock.lock();
            let mut file = shard.file.try_clone()?;
            let offset = file.seek(std::io::SeekFrom::End(0))?;
            packfile::write_record(&mut file, &record)?;
            shard.file_len.store(
                offset + record.serialized_len() as u64,
                std::sync::atomic::Ordering::Release,
            );
            (shard.shard_id, offset)
        };

        // Update per-room index
        let index_full = {
            let mut indexes = self.indexes.write();
            let index = indexes
                .entry(*room_id)
                .or_insert_with(|| LossyIndex::new(4096));
            index.insert(id, shard_id, offset).is_err()
        };
        if index_full {
            let mut rebuilt = self.rebuild_index(room_id);
            let _ = rebuilt.insert(id, shard_id, offset);
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
        self.repack.purge_room(room_id);
        self.indexes.write().remove(room_id);
        self.live_roots.write().remove(room_id);
        self.put_locks.lock().remove(room_id);
        self.cache.clear();
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

        store.cache().clear();
        let got_a = store.get(&TEST_ROOM, &a).unwrap();
        assert_eq!(got_a.unwrap().bytes, data_a.bytes);

        store.put(&TEST_ROOM, &b, &data_b).unwrap();

        store.cache().clear();
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

        store.put(&OTHER_ROOM, &id, &data).unwrap();

        store.delete_room(&OTHER_ROOM).unwrap();
        assert!(store.get(&OTHER_ROOM, &id).unwrap().is_none());
        assert!(store.indexes.read().get(&OTHER_ROOM).is_none());
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
        store.cache().clear();

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
        let store =
            PackfileStorage::open_with_swizzle(dir, NodeCache::new(100), test_swizzle).unwrap();

        let parent_id = [0x10u8; 16];
        let child_a = [0x20u8; 16];
        let child_b = [0x30u8; 16];

        let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
        let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));

        store.put(&TEST_ROOM, &child_a, &data_a).unwrap();
        store.put(&TEST_ROOM, &child_b, &data_b).unwrap();
        store.put(&TEST_ROOM, &parent_id, &parent_data).unwrap();

        store.cache.clear();
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

        match &fetched.children[0] {
            NodeRef::Resolved(id, data) => {
                assert_eq!(*id, child1_id);
                assert_eq!(data.bytes, child1_data.bytes);
            }
            NodeRef::Lazy(_) => panic!("Expected child 1 to be resolved"),
        }

        assert!(matches!(fetched.children[1], NodeRef::Lazy(_)));
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
    fn test_resolve_from_candidates_parser_children() {
        fn parser(data: &[u8]) -> Vec<NodeId> {
            if data == b"parent" {
                vec![[0x11; 16], [0x22; 16]]
            } else {
                vec![]
            }
        }

        let dir = test_dir("resolve_parser");
        let store =
            PackfileStorage::open_with_cache(dir, NodeCache::with_default_capacity(), Some(parser))
                .unwrap();

        let child1_id = [0x11; 16];
        let child1_data = NodeData::new(bytes::Bytes::from_static(b"child 1"));
        store.put(&TEST_ROOM, &child1_id, &child1_data).unwrap();

        let parent_id = [0x42; 16];
        store
            .put(
                &TEST_ROOM,
                &parent_id,
                &NodeData::new(bytes::Bytes::from_static(b"parent")),
            )
            .unwrap();

        store.cache().clear();

        let fetched = store.get(&TEST_ROOM, &parent_id).unwrap().unwrap();
        assert_eq!(fetched.children.len(), 2);
        assert!(matches!(fetched.children[0], NodeRef::Lazy(id) if id == child1_id));
        assert!(matches!(fetched.children[1], NodeRef::Lazy(id) if id == [0x22; 16]));
    }

    #[test]
    fn test_resolve_from_candidates_pinned_child() {
        fn parser(data: &[u8]) -> Vec<NodeId> {
            if data == b"parent" {
                vec![[0x11; 16]]
            } else {
                vec![]
            }
        }

        let dir = test_dir("resolve_pinned");
        let store =
            PackfileStorage::open_with_cache(dir, NodeCache::with_default_capacity(), Some(parser))
                .unwrap();

        let child_id = [0x11; 16];
        let child_data = Arc::new(NodeData::new(bytes::Bytes::from_static(b"child")));
        store.pinned().pin(child_id, child_data.clone());

        store
            .put(
                &TEST_ROOM,
                &[0x42; 16],
                &NodeData::new(bytes::Bytes::from_static(b"parent")),
            )
            .unwrap();

        store.cache().clear();

        let fetched = store.get(&TEST_ROOM, &[0x42; 16]).unwrap().unwrap();
        assert_eq!(fetched.children.len(), 1);
        match &fetched.children[0] {
            NodeRef::Resolved(id, data) => {
                assert_eq!(*id, child_id);
                assert_eq!(data.bytes, child_data.bytes);
            }
            NodeRef::Lazy(_) => panic!("Expected child to be Resolved from pinned"),
        }
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

        // A valid shard file
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
        assert_eq!(
            store
                .indexes
                .read()
                .get(&[0x01u8; 16])
                .map_or(0, crate::index::LossyIndex::len),
            1
        );
    }
}
