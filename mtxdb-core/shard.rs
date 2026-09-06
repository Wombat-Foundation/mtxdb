use std::fs::{self, File};
use std::io::{self, BufReader, Seek};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::packfile::{self, Record};

/// Maximum number of shards in the pool.
pub const MAX_SHARDS: usize = 4;

/// Maximum shard size before rotation (2 GB).
pub const MAX_SHARD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// A single global shard file shared across all rooms.
pub struct Shard {
    pub shard_id: u8,
    pub file: File,
    pub path: PathBuf,
    /// Lazily-created mmap. Remapped when the file grows.
    pub(crate) mmap: RwLock<Option<Mmap>>,
    /// Serializes appends to this shard.
    pub(crate) append_lock: parking_lot::Mutex<()>,
    /// Whether this shard is still in active use. Set to `false` when
    /// retired by the pool (all records reclaimed by repack). Drop
    /// deletes the file only when retired.
    pub(crate) is_current: AtomicBool,
    /// Current file length, tracked atomically for rotation decisions
    /// without a `metadata()` syscall on every put.
    pub(crate) file_len: AtomicU64,
}

impl Shard {
    /// Get the memory-mapped view, creating it if absent.
    ///
    /// Remaps when the file grows past the current mapping.
    pub fn mmap(&self) -> io::Result<parking_lot::RwLockReadGuard<'_, Option<Mmap>>> {
        let guard = self.mmap.read();
        if guard.is_some() {
            return Ok(guard);
        }
        drop(guard);
        let mut guard = self.mmap.write();
        if guard.is_none() {
            *guard = Some(packfile::map_pack(&self.file)?);
        }
        drop(guard);
        Ok(self.mmap.read())
    }
}

impl Drop for Shard {
    fn drop(&mut self) {
        if !self.is_current.load(Ordering::Acquire) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Pool of global shard files shared across all rooms.
///
/// Only `MAX_SHARDS` files are open at any time, capping file descriptor
/// usage regardless of room count. The active write shard rotates when it
/// exceeds `MAX_SHARD_BYTES`.
pub struct ShardPool {
    /// Fixed-size array of shard slots. `None` means unused.
    shards: RwLock<Vec<Option<Arc<Shard>>>>,
    /// Index of the shard currently accepting writes.
    active_write: parking_lot::Mutex<u8>,
    /// Serializes shard rotation (finding/creating the next shard).
    rotation_lock: parking_lot::Mutex<()>,
    base_dir: PathBuf,
}

impl ShardPool {
    /// Open or create a shard pool, scanning for existing shard files.
    pub fn open(base_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&base_dir)?;

        let mut shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS as u8).map(|_| None).collect();
        let mut highest_active: u8 = 0;

        // Scan for existing shard files
        for entry in fs::read_dir(&base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "pack") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(id_str) = stem.strip_prefix("shard_") {
                        if let Ok(id) = u8::from_str_radix(id_str, 16) {
                            if (id as usize) < MAX_SHARDS {
                                let file = packfile::open_packfile(&path, false)?;
                                let file_len = file.metadata()?.len();
                                let shard = Arc::new(Shard {
                                    shard_id: id,
                                    file,
                                    path,
                                    mmap: RwLock::new(None),
                                    append_lock: parking_lot::Mutex::new(()),
                                    is_current: AtomicBool::new(true),
                                    file_len: AtomicU64::new(file_len),
                                });
                                shards[id as usize] = Some(shard);
                                if id > highest_active {
                                    highest_active = id;
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no shards exist, create the initial shard 0
        if shards.iter().all(std::option::Option::is_none) {
            let path = Self::shard_path(&base_dir, 0);
            let file = packfile::open_packfile(&path, true)?;
            let file_len = file.metadata()?.len();
            shards[0] = Some(Arc::new(Shard {
                shard_id: 0,
                file,
                path,
                mmap: RwLock::new(None),
                append_lock: parking_lot::Mutex::new(()),
                is_current: AtomicBool::new(true),
                file_len: AtomicU64::new(file_len),
            }));
        }

        Ok(Self {
            shards: RwLock::new(shards),
            active_write: parking_lot::Mutex::new(highest_active),
            rotation_lock: parking_lot::Mutex::new(()),
            base_dir,
        })
    }

    /// On-disk path for a shard file.
    #[must_use]
    pub fn shard_path(base_dir: &Path, shard_id: u8) -> PathBuf {
        base_dir.join(format!("shard_{shard_id:02x}.pack"))
    }

    /// Get a reference to a shard by ID.
    pub fn get_shard(&self, shard_id: u8) -> Option<Arc<Shard>> {
        self.shards.read().get(shard_id as usize)?.clone()
    }

    /// Get the current active write shard.
    pub fn active_shard(&self) -> Arc<Shard> {
        let id = *self.active_write.lock();
        self.shards
            .read()
            .get(id as usize)
            .and_then(std::clone::Clone::clone)
            .expect("active write shard must exist")
    }

    /// Append a record to the active shard. Returns `(shard_id, offset)`.
    /// Rotates to a new shard if the current one is full.
    pub fn put_record(&self, record: &Record) -> io::Result<(u8, u64)> {
        loop {
            let shard = self.active_shard();
            let record_len = record.serialized_len() as u64;
            let current_len = shard.file_len.load(Ordering::Acquire);

            // Check if this record would exceed the shard capacity
            if current_len + record_len > MAX_SHARD_BYTES && current_len > 5 {
                // Don't rotate if the shard is nearly empty (just header)
                self.rotate()?;
                continue;
            }

            let offset = {
                let _guard = shard.append_lock.lock();
                let mut file = shard.file.try_clone()?;
                let offset = file.seek(io::SeekFrom::End(0))?;
                packfile::write_record(&mut file, record)?;
                shard.file_len.store(offset + record_len, Ordering::Release);
                offset
            };

            return Ok((shard.shard_id, offset));
        }
    }

    /// Read a record from a specific shard at the given offset.
    pub fn read_at(shard: &Shard, offset: u64) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;

        for attempt in 0..2 {
            let guard = shard.mmap().map_err(StorageError::Io)?;
            let Some(mem) = guard.as_deref() else {
                return Err(StorageError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "shard could not be mapped",
                )));
            };

            let offset = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

            if offset.checked_add(4).map_or(true, |end| end > mem.len()) {
                if attempt == 0 {
                    drop(guard);
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated length prefix".into()));
            }

            let prefix_end = offset + 4;
            let payload_len_bytes: [u8; 4] = mem[offset..prefix_end].try_into().unwrap();
            let payload_len = u32::from_le_bytes(payload_len_bytes);

            if !(32..=packfile::MAX_RECORD_LEN).contains(&payload_len) {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {payload_len}"
                )));
            }

            let payload_len_usize = payload_len as usize;
            let crc_pos = prefix_end + payload_len_usize;
            let frame_end = crc_pos + 4;

            if frame_end > mem.len() {
                if attempt == 0 {
                    drop(guard);
                    Self::remap_shard(shard)?;
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

            let mut room_id = [0u8; 16];
            room_id.copy_from_slice(&payload[..16]);
            let mut hash = [0u8; 16];
            hash.copy_from_slice(&payload[16..32]);
            let data = bytes::Bytes::copy_from_slice(&payload[32..]);

            return Ok(Record {
                room_id,
                hash,
                data,
            });
        }
        unreachable!("read_at remap-retry is bounded to two iterations")
    }

    /// Remap a shard to its current on-disk length.
    fn remap_shard(shard: &Shard) -> Result<(), crate::storage::StorageError> {
        let file_len = shard
            .file
            .metadata()
            .map_err(crate::storage::StorageError::Io)?
            .len();
        let mut guard = shard.mmap.write();
        if guard.as_ref().map_or(true, |m| (m.len() as u64) < file_len) {
            *guard =
                Some(packfile::map_pack(&shard.file).map_err(crate::storage::StorageError::Io)?);
        }
        Ok(())
    }

    /// Rotate to the next shard. Reuses a retired shard slot if available,
    /// otherwise creates a new shard file.
    fn rotate(&self) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        let current = *self.active_write.lock();

        let mut shards = self.shards.write();

        // Find the next available slot (skip current, prefer retired/empty)
        for offset in 1..=MAX_SHARDS as u8 {
            let candidate = current.wrapping_add(offset) % MAX_SHARDS as u8;
            if shards[candidate as usize].is_none() {
                // Empty slot — create new shard here
                let path = Self::shard_path(&self.base_dir, candidate);
                let file = packfile::open_packfile(&path, true)?;
                let file_len = file.metadata()?.len();
                let shard = Arc::new(Shard {
                    shard_id: candidate,
                    file,
                    path,
                    mmap: RwLock::new(None),
                    append_lock: parking_lot::Mutex::new(()),
                    is_current: AtomicBool::new(true),
                    file_len: AtomicU64::new(file_len),
                });
                shards[candidate as usize] = Some(shard);
                drop(shards);
                *self.active_write.lock() = candidate;
                return Ok(());
            }
        }

        // All slots occupied — pick the next one and overwrite it.
        // The old shard's Drop will delete its file when the last Arc goes away.
        let candidate = current.wrapping_add(1) % MAX_SHARDS as u8;
        if let Some(ref old) = shards[candidate as usize] {
            old.is_current.store(false, Ordering::Release);
        }
        let path = Self::shard_path(&self.base_dir, candidate);
        let file = packfile::open_packfile(&path, true)?;
        let file_len = file.metadata()?.len();
        let shard = Arc::new(Shard {
            shard_id: candidate,
            file,
            path,
            mmap: RwLock::new(None),
            append_lock: parking_lot::Mutex::new(()),
            is_current: AtomicBool::new(true),
            file_len: AtomicU64::new(file_len),
        });
        shards[candidate as usize] = Some(shard);
        drop(shards);
        *self.active_write.lock() = candidate;
        Ok(())
    }

    /// Sync all shards to disk.
    pub fn sync_all(&self) -> io::Result<()> {
        let shards = self.shards.read();
        for shard in shards.iter().flatten() {
            shard.file.sync_all()?;
        }
        Ok(())
    }

    /// Scan a shard file and return `(room_id, hash, offset)` entries.
    /// Used during startup to rebuild per-room indexes.
    pub fn scan_shard(path: &Path) -> io::Result<Vec<([u8; 16], [u8; 16], u64)>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        if !packfile::read_header(&mut reader)? {
            return Ok(entries);
        }

        loop {
            let offset = reader.stream_position()?;
            match packfile::read_record(&mut reader) {
                Ok(Some(record)) => {
                    entries.push((record.room_id, record.hash, offset));
                }
                Ok(None) => break,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("warning: shard scan stopped at offset {offset}: {e}");
                    break;
                }
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        use std::sync::atomic::AtomicU64;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_shard_{name}_{id}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_record(room: u8, hash_byte: u8, data: &[u8]) -> Record {
        let mut room_id = [0u8; 16];
        room_id[0] = room;
        let mut hash = [0u8; 16];
        hash[0] = hash_byte;
        Record {
            room_id,
            hash,
            data: bytes::Bytes::copy_from_slice(data),
        }
    }

    #[test]
    fn test_shard_pool_create_and_write() {
        let dir = test_dir("pool_create");
        let pool = ShardPool::open(dir).unwrap();

        let record = test_record(0x01, 0xAA, b"hello shard");
        let (shard_id, offset) = pool.put_record(&record).unwrap();
        assert_eq!(shard_id, 0);
        assert!(offset > 0);

        let shard = pool.get_shard(shard_id).unwrap();
        let read = ShardPool::read_at(&shard, offset).unwrap();
        assert_eq!(read.room_id[0], 0x01);
        assert_eq!(read.hash[0], 0xAA);
        assert_eq!(read.data.as_ref(), b"hello shard");
    }

    #[test]
    fn test_shard_pool_rotation() {
        let dir = test_dir("pool_rotate");
        let pool = ShardPool::open(dir).unwrap();
        pool.active_shard()
            .file_len
            .store(MAX_SHARD_BYTES - 10, Ordering::Release);

        let record = test_record(0x01, 0xBB, b"trigger rotation");
        let (shard_id, _offset) = pool.put_record(&record).unwrap();
        assert_eq!(shard_id, 1);
    }

    #[test]
    fn test_shard_pool_scan() {
        let dir = test_dir("pool_scan");
        let pool = ShardPool::open(dir.clone()).unwrap();

        let r1 = test_record(0x01, 0x10, b"room1 msg1");
        let r2 = test_record(0x02, 0x20, b"room2 msg1");
        let r3 = test_record(0x01, 0x11, b"room1 msg2");
        pool.put_record(&r1).unwrap();
        pool.put_record(&r2).unwrap();
        pool.put_record(&r3).unwrap();

        let entries = ShardPool::scan_shard(&ShardPool::shard_path(&dir, 0)).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0[0], 0x01);
        assert_eq!(entries[0].1[0], 0x10);
        assert_eq!(entries[2].0[0], 0x01);
        assert_eq!(entries[2].1[0], 0x11);
    }
}
