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
/// Read path: cache → index lookup → packfile read → cache insert
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

                // Register the pack with the repack manager
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

        // Create empty index for this room
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
    fn get(&self, id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        // 1. Check cache
        if let Some(data) = self.cache.get(id) {
            return Ok(Some((*data).clone()));
        }

        // 2. Find record offset via index (try all rooms)
        let (pack_id, offset) = {
            let indexes = self.indexes.read();
            match indexes.values().find_map(|idx| idx.lookup(id)) {
                Some(v) => v,
                None => return Ok(None),
            }
        };

        // 3. Find the pack generation with matching pack_id
        let gen = {
            let packs = self.repack.packs.read();
            match packs.values().find(|g| g.pack_id == pack_id) {
                Some(g) => g.clone(),
                None => return Ok(None),
            }
        };

        // 4. Read and verify
        let record = Self::read_at(&gen.file, offset)?;
        if record.hash != *id {
            return Err(StorageError::VerificationFailed(*id));
        }

        let data = NodeData { bytes: record.data };

        // 5. Cache it
        self.cache.insert(*id, Arc::new(data.clone()));

        Ok(Some(data))
    }

    fn get_many(&self, ids: &[NodeId]) -> Result<Vec<Option<NodeData>>, StorageError> {
        ids.iter().map(|id| self.get(id)).collect()
    }

    fn put(&self, id: &NodeId, data: &NodeData) -> Result<(), StorageError> {
        // Use default room; real implementation extracts room from content.
        let room_id = [0u8; 16];

        let gen = self.ensure_pack(&room_id)?;

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

        // Update index
        {
            let mut indexes = self.indexes.write();
            let index = indexes
                .entry(room_id)
                .or_insert_with(|| LossyIndex::new(256));
            let _ = index.insert(id, gen.pack_id, offset);
        }

        // Cache it
        self.cache.insert(*id, Arc::new(data.clone()));

        Ok(())
    }

    fn put_many(&self, entries: &[(NodeId, NodeData)]) -> Result<(), StorageError> {
        for (id, data) in entries {
            self.put(id, data)?;
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

        store.put(&id, &data).unwrap();
        let got = store.get(&id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);
    }

    #[test]
    fn test_get_not_found() {
        let dir = test_dir("notfound");
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&[0x00; 16]).unwrap().is_none());
    }

    #[test]
    fn test_cache_hit() {
        let dir = test_dir("cachehit");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"cached"),
        };

        store.put(&id, &data).unwrap();

        // put already caches the node, so both gets are cache hits
        let _ = store.get(&id).unwrap();
        assert_eq!(store.cache.hits(), 1);

        let _ = store.get(&id).unwrap();
        assert_eq!(store.cache.hits(), 2);
    }

    #[test]
    fn test_delete_room() {
        let dir = test_dir("delete");
        let store = PackfileStorage::open(dir).unwrap();

        let room = [0xAA; 16];
        let id = [0x01u8; 16];
        let data = NodeData {
            bytes: bytes::Bytes::from_static(b"room data"),
        };

        // Manually set up a pack for this room
        let gen = store.ensure_pack(&room).unwrap();
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
            let index = indexes.entry(room).or_insert_with(|| LossyIndex::new(256));
            let _ = index.insert(&id, gen.pack_id, 5);
        }

        store.delete_room(&room).unwrap();
        assert!(store.repack.get_pack(&room).is_none());
        assert!(store.indexes.read().get(&room).is_none());
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

        store.put_many(&entries).unwrap();

        let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        let results = store.get_many(&ids).unwrap();
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
            store.put(&id, &data).unwrap();
        }

        // Verify all can be read back
        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let got = store.get(&id).unwrap().unwrap();
            assert_eq!(got.bytes, bytes::Bytes::from(format!("record {i}")));
        }
    }
}
