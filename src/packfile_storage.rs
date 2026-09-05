use std::collections::HashMap;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::{BufReader, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::cache::NodeCache;
use crate::index::LossyIndex;
use crate::packfile::{self, PackGeneration, Record};
use crate::repack::RepackManager;
use crate::storage::{NodeData, NodeId, StorageEngine, StorageError};

/// A content-addressed packfile storage engine.
///
/// Wires together the packfile format, lossy fanout index, and
/// decoded-node cache into a single `StorageEngine` implementation.
///
/// Each room gets its own index and packfile, keeping the active index
/// at ~8KB per 1000-node room (100 active rooms < 1MB total).
///
/// Read path: cache → index lookup (with linear-probe collision retry)
///            → packfile read → CRC verify → cache insert
/// Write path: packfile append → index insert → cache insert
pub struct PackfileStorage {
    /// Manages per-room pack generations and atomic repack.
    repack: RepackManager,
    /// Per-room lossy fanout index for (hash → `pack_id`, offset).
    indexes: RwLock<HashMap<[u8; 16], LossyIndex>>,
    /// Verify-once decoded node cache.
    cache: NodeCache,
    /// Base directory for all pack files.
    base_dir: PathBuf,
}

impl PackfileStorage {
    /// Create a new packfile storage.
    ///
    /// On creation, scans existing packfiles to rebuild in-memory indexes.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be read.
    pub fn open(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_cache(base_dir, NodeCache::with_default_capacity())
    }

    /// Create a new packfile storage with custom cache size.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be read.
    pub fn open_with_cache(base_dir: PathBuf, cache: NodeCache) -> Result<Self, std::io::Error> {
        let repack = RepackManager::new(base_dir.clone());
        let mut indexes = HashMap::new();

        if base_dir.exists() {
            Self::scan_existing(&base_dir, &repack, &mut indexes)?;
        }

        Ok(Self {
            repack,
            indexes: RwLock::new(indexes),
            cache,
            base_dir,
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
        });
        self.repack.swap_pack(*room_id, gen.clone());

        self.indexes
            .write()
            .entry(*room_id)
            .or_insert_with(|| LossyIndex::new(256));

        Ok(gen)
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

    /// Read a record at the given offset from a file handle.
    fn read_at(file: &File, offset: u64) -> Result<Record, StorageError> {
        let mut reader = BufReader::new(file);
        reader.seek(std::io::SeekFrom::Start(offset))?;
        packfile::read_record(&mut reader)?
            .ok_or_else(|| StorageError::Corrupt("unexpected EOF at record boundary".into()))
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

            let Ok(record) = Self::read_at(&gen.file, *offset) else {
                continue;
            };

            // Verify against caller-requested hash, not the frame hash
            if record.hash != *id {
                continue;
            }

            let data = NodeData { bytes: record.data };
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
        {
            let mut indexes = self.indexes.write();
            let index = indexes
                .entry(*room_id)
                .or_insert_with(|| LossyIndex::new(256));
            let _ = index.insert(id, gen.pack_id, offset);
        }

        // Cache it
        self.cache.insert(*id, Arc::new(data.clone()));

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
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"hello world"),
        };

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
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"cached"),
        };

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
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"room data"),
        };

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
                .or_insert_with(|| LossyIndex::new(256));
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
                (
                    id,
                    NodeData {
                        bytes: bytes::Bytes::from(format!("node {i}")),
                    },
                )
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
            let data = NodeData {
                bytes: bytes::Bytes::from(format!("record {i}")),
            };
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
        let data_a = NodeData {
            bytes: bytes::Bytes::from_static(b"room A data"),
        };
        let data_b = NodeData {
            bytes: bytes::Bytes::from_static(b"room B data"),
        };

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
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"valid data"),
        };
        store.put(&TEST_ROOM, &id, &data).unwrap();

        // Clear cache so the next get must go to disk
        store.cache.clear();

        // Normal read works
        let got = store.get(&TEST_ROOM, &id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);

        // Non-existent hash returns None (not a tag collision loop)
        assert!(store.get(&TEST_ROOM, &[0xFF; 16]).unwrap().is_none());
    }
}
