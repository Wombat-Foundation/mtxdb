use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::packfile::{self, Record};

/// Maximum number of shards in the pool.
pub const MAX_SHARDS: usize = 4096;

/// Maximum number of shards as `u16`. Primary constant for shard IDs
/// and modular arithmetic.
pub(crate) const MAX_SHARDS_U16: u16 = 4096;

/// Maximum shard size before rotation (256 MB).
pub const MAX_SHARD_BYTES: u64 = 256 * 1024 * 1024;

/// Scanned `(room_id, hash, offset)` entry from a shard file.
pub type ShardEntry = ([u8; 16], [u8; 16], u64);

/// A single global shard file shared across all rooms.
pub struct Shard {
    /// Identifier for this shard within the pool.
    pub shard_id: u16,
    /// Monotonically increasing generation counter, distinct from the
    /// slot index. Ensures a reused slot never collides on-disk with a
    /// still-referenced old shard at the same slot.
    pub generation: u64,
    /// The open file handle backing this shard.
    pub file: File,
    /// Filesystem path to this shard's file.
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
    /// Number of records appended to this shard.
    write_count: AtomicU64,
    /// Total payload bytes appended to this shard (serialized record length).
    bytes_written: AtomicU64,
    /// Number of times this shard's file has been fsynced.
    sync_count: AtomicU64,
}

/// Point-in-time snapshot of a shard's IO/sync counters.
///
/// Loaded with `Ordering::Relaxed` — cheap to take, not synchronized
/// against concurrent activity on the shard.
///
/// Deliberately write/sync-only, not read-tracking: `read_at` goes through
/// an mmap, so a "read" often just touches an already page-cache-resident
/// page — no disk I/O at all — while counting it would still cost a real
/// cache-line-contending atomic write on every call, on the hottest path
/// in the engine, for a number that doesn't answer any disk-I/O question.
/// `write_count`/`bytes_written`/`sync_count` are the ones tied to actual fsync
/// calls, the only genuinely disk-relevant signal here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShardStats {
    /// Number of records written to this shard.
    pub write_count: u64,
    /// Total bytes written to this shard.
    pub bytes_written: u64,
    /// Number of fsync calls on this shard.
    pub sync_count: u64,
}

/// Filename for the persisted stats snapshot, stored alongside shard files.
const STATS_FILENAME: &str = "shard_stats.bin";

/// Magic bytes + version identifying the stats file format.
const STATS_MAGIC: &[u8; 4] = b"MSTA";
const STATS_VERSION: u8 = 2;

/// On-disk size of one stats record: `shard_id`(2) + generation(8) + 3×counter(8) = 34 bytes.
const STATS_RECORD_LEN: usize = 2 + 8 + 8 * 3;

impl ShardStats {
    fn encode(self, shard_id: u16, generation: u64, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&shard_id.to_le_bytes());
        buf.extend_from_slice(&generation.to_le_bytes());
        buf.extend_from_slice(&self.write_count.to_le_bytes());
        buf.extend_from_slice(&self.bytes_written.to_le_bytes());
        buf.extend_from_slice(&self.sync_count.to_le_bytes());
    }

    fn decode(rec: &[u8; STATS_RECORD_LEN]) -> (u16, u64, Self) {
        let shard_id = u16::from_le_bytes(rec[0..2].try_into().unwrap());
        let generation = u64::from_le_bytes(rec[2..10].try_into().unwrap());
        let write_count = u64::from_le_bytes(rec[10..18].try_into().unwrap());
        let bytes_written = u64::from_le_bytes(rec[18..26].try_into().unwrap());
        let sync_count = u64::from_le_bytes(rec[26..34].try_into().unwrap());
        (
            shard_id,
            generation,
            Self {
                write_count,
                bytes_written,
                sync_count,
            },
        )
    }
}

impl Shard {
    fn new(shard_id: u16, generation: u64, file: File, path: PathBuf, file_len: u64) -> Self {
        Self {
            shard_id,
            generation,
            file,
            path,
            mmap: RwLock::new(None),
            append_lock: parking_lot::Mutex::new(()),
            is_current: AtomicBool::new(true),
            file_len: AtomicU64::new(file_len),
            write_count: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            sync_count: AtomicU64::new(0),
        }
    }

    /// Snapshot this shard's IO/sync counters.
    #[must_use]
    pub fn stats(&self) -> ShardStats {
        ShardStats {
            write_count: self.write_count.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            sync_count: self.sync_count.load(Ordering::Relaxed),
        }
    }

    /// Current on-disk file length, as tracked without a `metadata()` syscall.
    #[must_use]
    pub fn file_len(&self) -> u64 {
        self.file_len.load(Ordering::Acquire)
    }

    /// Restore counters from a persisted snapshot. Only called at startup,
    /// before the shard is shared, so `Relaxed` stores are fine.
    fn restore_stats(&self, stats: ShardStats) {
        self.write_count.store(stats.write_count, Ordering::Relaxed);
        self.bytes_written
            .store(stats.bytes_written, Ordering::Relaxed);
        self.sync_count.store(stats.sync_count, Ordering::Relaxed);
    }
}

impl Shard {
    /// Get the memory-mapped view, creating it if absent.
    ///
    /// Remaps when the file grows past the current mapping.
    ///
    /// # Errors
    /// Returns `io::Error` if the packfile cannot be mapped.
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
    active_write: parking_lot::Mutex<u16>,
    /// Serializes shard rotation (finding/creating the next shard).
    rotation_lock: parking_lot::Mutex<()>,
    base_dir: PathBuf,
    /// Shard IDs written to since the last sync, for scoped fsync.
    dirty: parking_lot::Mutex<HashSet<u16>>,
    /// Monotonically increasing generation counter for shard filenames.
    /// Each newly created shard file gets a unique generation, so a
    /// reused slot never collides on-disk with a still-referenced old
    /// shard at the same slot.
    next_generation: AtomicU64,
    /// Total number of shards retired (garbage-collected after a repack)
    /// over the pool's lifetime.
    retired_count: AtomicU64,
    /// Each room's current "home" shard: writes for a room are routed here
    /// instead of always following the pool-wide active-write cursor, so a
    /// room's records stay contiguous within a shard rather than
    /// interleaving with whichever other rooms happen to write around the
    /// same time. Shards remain shared — multiple rooms' homes can and do
    /// coincide, especially low-traffic rooms — a room just sticks to its
    /// home until that shard fills, rather than following the global
    /// cursor wherever unrelated rooms have since moved it.
    room_home: RwLock<HashMap<[u8; 16], u16>>,
}

impl ShardPool {
    /// Open or create a shard pool, scanning for existing shard files.
    ///
    /// # Errors
    /// Returns `io::Error` on directory read failure or packfile open failure.
    pub fn open(base_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&base_dir)?;

        let mut shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS).map(|_| None).collect();
        let mut highest_active: u16 = 0;
        let mut max_generation: u64 = 0;

        // Track the highest generation seen per slot so we can reject
        // stale files left behind by a crash between rotate() creating a
        // new generation and the old generation's Drop deleting it.
        let mut best_generation: Vec<Option<u64>> = vec![None; MAX_SHARDS];

        // Scan for existing shard files.  Supports three filename formats:
        //   shard_XX.pack                      — ancient (no generation, implied gen 0)
        //   shard_XX_YYYYYYYYYYYYYYYY.pack      — 2-digit slot, generation-tracked
        //   shard_XXXX_YYYYYYYYYYYYYYYY.pack    — 4-digit slot, generation-tracked
        for entry in fs::read_dir(&base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.extension().is_some_and(|e| e == "pack") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(id_hex) = stem.strip_prefix("shard_") else {
                continue;
            };

            // Three-way parse: detect by position of underscore.
            // - No underscore → ancient format "XX" (gen 0)
            // - Underscore after 2 chars → 2-digit slot "XX_YYYY..."
            // - Underscore after 4 chars → 4-digit slot "XXXX_YYYY..."
            let (slot_hex, generation) = match id_hex.split_once('_') {
                Some((slot, gen_hex)) => {
                    let gen = u64::from_str_radix(gen_hex, 16).unwrap_or(0);
                    (slot, gen)
                }
                None => (id_hex, 0),
            };

            if let Ok(id) = u16::from_str_radix(slot_hex, 16) {
                if id < MAX_SHARDS_U16 {
                    let id_usize = id as usize;

                    // If we already have a generation for this slot and the
                    // current file is not newer, it's stale — delete it.
                    if let Some(prev) = best_generation[id_usize] {
                        if prev >= generation {
                            let _ = fs::remove_file(&path);
                            continue;
                        }
                    }

                    // We have a strictly newer generation — drop the old one
                    // and its file before installing the new shard.
                    if let Some(old) = shards[id_usize].take() {
                        let _ = fs::remove_file(&old.path);
                    }

                    let file = packfile::open_packfile(&path, false)?;
                    let file_len = file.metadata()?.len();
                    let shard = Arc::new(Shard::new(id, generation, file, path, file_len));
                    best_generation[id_usize] = Some(generation);
                    shards[id_usize] = Some(shard);
                    if id > highest_active {
                        highest_active = id;
                    }
                    if generation >= max_generation {
                        max_generation = generation.saturating_add(1);
                    }
                }
            }
        }

        // If no shards exist, create the initial shard 0
        if shards.iter().all(std::option::Option::is_none) {
            let path = Self::shard_path(&base_dir, 0, 0);
            let file = packfile::open_packfile(&path, true)?;
            let file_len = file.metadata()?.len();
            shards[0] = Some(Arc::new(Shard::new(0, 0, file, path, file_len)));
            max_generation = 1;
        }

        Self::restore_persisted_stats(&base_dir, &shards);

        Ok(Self {
            shards: RwLock::new(shards),
            active_write: parking_lot::Mutex::new(highest_active),
            rotation_lock: parking_lot::Mutex::new(()),
            base_dir,
            dirty: parking_lot::Mutex::new(HashSet::new()),
            next_generation: AtomicU64::new(max_generation),
            retired_count: AtomicU64::new(0),
            room_home: RwLock::new(HashMap::new()),
        })
    }

    /// Seed a room's home shard — used at startup to approximate where a
    /// room's most recent data already lives (from a content scan elsewhere,
    /// since `ShardPool::open` itself only discovers shard *files*, not
    /// their room contents). Normal routing updates the home automatically
    /// from then on via `put_record`.
    pub(crate) fn set_room_home(&self, room_id: &[u8; 16], shard_id: u16) {
        self.room_home.write().insert(*room_id, shard_id);
    }

    /// The shard a room's writes should go to: its remembered home if one
    /// exists and the slot is still occupied, otherwise a freshly assigned
    /// home (the pool's current active shard, the same fallback every room
    /// used before per-room routing existed).
    fn shard_for_room(&self, room_id: &[u8; 16]) -> Arc<Shard> {
        if let Some(id) = self.room_home.read().get(room_id).copied() {
            if let Some(shard) = self.get_shard(id) {
                return shard;
            }
        }
        let shard = self.active_shard();
        self.room_home.write().insert(*room_id, shard.shard_id);
        shard
    }

    /// A room's home shard just filled up: rotate the pool forward (unless
    /// another room already did, in which case just adopt whatever's now
    /// active) and point the room at the result.
    fn rotate_room_full_home(
        &self,
        room_id: &[u8; 16],
        full_shard_id: u16,
    ) -> io::Result<Arc<Shard>> {
        {
            let active = self.active_write.lock();
            if *active == full_shard_id {
                drop(active);
                self.rotate()?;
            }
        }
        let shard = self.active_shard();
        self.room_home.write().insert(*room_id, shard.shard_id);
        Ok(shard)
    }

    /// Path to the persisted stats snapshot for a base directory.
    fn stats_path(base_dir: &Path) -> PathBuf {
        base_dir.join(STATS_FILENAME)
    }

    /// Load a persisted stats snapshot, if one exists, and restore each
    /// shard's counters when its generation still matches — a stale
    /// snapshot entry (from a slot since retired and reused) is silently
    /// skipped rather than misapplied.
    ///
    /// Best-effort: a missing, truncated, or corrupt file just means no
    /// stats are restored — never a startup failure over stats alone.
    fn restore_persisted_stats(base_dir: &Path, shards: &[Option<Arc<Shard>>]) {
        let path = Self::stats_path(base_dir);
        let Ok(buf) = fs::read(&path) else {
            return;
        };
        if buf.len() < 5 || &buf[0..4] != STATS_MAGIC || buf[4] != STATS_VERSION {
            return;
        }
        let body = &buf[5..];
        for chunk in body.chunks(STATS_RECORD_LEN) {
            let Ok(rec) = <&[u8; STATS_RECORD_LEN]>::try_from(chunk) else {
                break;
            };
            let (shard_id, generation, stats) = ShardStats::decode(rec);
            if let Some(Some(shard)) = shards.get(shard_id as usize) {
                if shard.generation == generation {
                    shard.restore_stats(stats);
                }
            }
        }
    }

    /// Persist every currently-open shard's IO/sync counters to disk,
    /// keyed by `(shard_id, generation)` so a retired/reused slot's stale
    /// numbers are never mistakenly restored onto a new shard.
    ///
    /// Writes to a temp file and renames into place, so a crash mid-write
    /// leaves the previous snapshot (or none) rather than a torn file.
    ///
    /// # Errors
    /// Returns `io::Error` on write or rename failure.
    fn persist_stats(&self) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(STATS_MAGIC);
        buf.push(STATS_VERSION);
        for (shard_id, shard) in self.all_shards() {
            shard.stats().encode(shard_id, shard.generation, &mut buf);
        }

        let tmp_path = Self::stats_path(&self.base_dir).with_extension("bin.tmp");
        let final_path = Self::stats_path(&self.base_dir);
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&buf)?;
            tmp.sync_all()?;
        }
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// On-disk path for a shard file.
    #[must_use]
    pub fn shard_path(base_dir: &Path, shard_id: u16, generation: u64) -> PathBuf {
        base_dir.join(format!("shard_{shard_id:04x}_{generation:016x}.pack"))
    }

    /// Get a reference to a shard by ID.
    #[must_use]
    pub fn get_shard(&self, shard_id: u16) -> Option<Arc<Shard>> {
        self.shards.read().get(shard_id as usize)?.clone()
    }

    /// Get IO/sync stats for a single shard by ID.
    #[must_use]
    pub fn stats(&self, shard_id: u16) -> Option<ShardStats> {
        self.shards
            .read()
            .get(shard_id as usize)?
            .as_ref()
            .map(|shard| shard.stats())
    }

    /// Snapshot IO/sync stats for every currently-open shard, as
    /// `(shard_id, ShardStats)` pairs.
    #[must_use]
    pub fn all_stats(&self) -> Vec<(u16, ShardStats)> {
        self.shards
            .read()
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .map(|shard| (u16::try_from(i).unwrap_or(u16::MAX), shard.stats()))
            })
            .collect()
    }

    /// Return all currently-open shards as `(slot_id, Arc<Shard>)` pairs.
    #[must_use]
    pub fn all_shards(&self) -> Vec<(u16, Arc<Shard>)> {
        self.shards
            .read()
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .map(|shard| (u16::try_from(i).unwrap_or(u16::MAX), Arc::clone(shard)))
            })
            .collect()
    }

    /// Get the current active write shard.
    ///
    /// # Panics
    /// Panics if the active write shard slot is `None` (invariant:
    /// `open()` always ensures at least shard 0 exists).
    #[must_use]
    pub fn active_shard(&self) -> Arc<Shard> {
        let id = *self.active_write.lock();
        self.shards
            .read()
            .get(id as usize)
            .and_then(std::clone::Clone::clone)
            .expect("active write shard must exist")
    }

    /// Append a record to its room's home shard (see `room_home`).
    /// Returns `(shard_id, offset)`. Rotates the room to a new home shard
    /// if its current one is full.
    ///
    /// # Errors
    /// Returns `io::Error` on write or rotation failure.
    pub fn put_record(&self, record: &Record) -> io::Result<(u16, u64)> {
        let mut shard = self.shard_for_room(&record.room_id);
        loop {
            let record_len = record.serialized_len() as u64;

            let offset = {
                let guard = shard.append_lock.lock();
                let mut file = shard.file.try_clone()?;
                let offset = file.seek(io::SeekFrom::End(0))?;

                // Check capacity while holding the append lock and after
                // seeking to the true end — avoids TOCTOU race where two
                // threads both pass the check then one exceeds the limit.
                let current_len = shard.file_len.load(Ordering::Acquire);
                let fits = current_len
                    .checked_add(record_len)
                    .is_some_and(|sum| sum <= MAX_SHARD_BYTES);
                if !fits && current_len > 5 {
                    drop(guard);
                    drop(file);
                    shard = self.rotate_room_full_home(&record.room_id, shard.shard_id)?;
                    continue;
                }

                packfile::write_record(&mut file, record)?;
                let new_len = offset.checked_add(record_len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "offset + record_len overflow")
                })?;
                shard.file_len.store(new_len, Ordering::Release);
                offset
            };

            shard.write_count.fetch_add(1, Ordering::Relaxed);
            shard.bytes_written.fetch_add(record_len, Ordering::Relaxed);
            self.dirty.lock().insert(shard.shard_id);
            return Ok((shard.shard_id, offset));
        }
    }

    /// Read a record from a specific shard at the given offset.
    ///
    /// # Errors
    /// Returns `StorageError::Corrupt` on CRC mismatch or truncated frame,
    /// `StorageError::Io` on I/O failure.
    ///
    /// # Panics
    /// Panics only on internal invariant violation (unreachable path).
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

            let prefix_end = offset.checked_add(4).expect("checked above");
            let payload_len_bytes: [u8; 4] = mem[offset..prefix_end].try_into().unwrap();
            let payload_len = u32::from_le_bytes(payload_len_bytes);

            if !(32..=packfile::MAX_RECORD_LEN).contains(&payload_len) {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {payload_len}"
                )));
            }

            let payload_len_usize = payload_len as usize;
            let crc_pos = prefix_end
                .checked_add(payload_len_usize)
                .ok_or_else(|| StorageError::Corrupt("prefix_end + payload_len overflow".into()))?;
            let frame_end = crc_pos
                .checked_add(4)
                .ok_or_else(|| StorageError::Corrupt("crc_pos + 4 overflow".into()))?;

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
    ///
    /// # Errors
    /// Returns `io::Error` on shard file creation failure.
    fn rotate(&self) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        let current = *self.active_write.lock();

        let mut shards = self.shards.write();

        // Find the next available slot (skip current, prefer retired/empty)
        for offset in 1..=MAX_SHARDS_U16 {
            let candidate = current.wrapping_add(offset).wrapping_rem(MAX_SHARDS_U16);
            if shards[candidate as usize].is_none() {
                let gen = self.next_generation.fetch_add(1, Ordering::Relaxed);
                let path = Self::shard_path(&self.base_dir, candidate, gen);
                let file = packfile::open_packfile(&path, true)?;
                let file_len = file.metadata()?.len();
                shards[candidate as usize] =
                    Some(Arc::new(Shard::new(candidate, gen, file, path, file_len)));
                drop(shards);
                *self.active_write.lock() = candidate;
                return Ok(());
            }
        }

        // All slots occupied and none retired — fail rather than silently
        // overwriting a shard that may still be referenced by room indexes.
        // The caller must repack to reclaim retired shard slots before
        // rotating again.
        Err(io::Error::other(
            "shard pool full: all slots occupied by active shards; repack to reclaim",
        ))
    }

    /// Best-effort stats snapshot write, same contract as the `Drop` impl:
    /// the persisted stats file is pure observability, not correctness, so
    /// a failure here (e.g. the base directory momentarily gone during test
    /// teardown) is logged and swallowed -- it must never turn a durable
    /// data sync that actually succeeded into a hard error for the caller.
    fn persist_stats_best_effort(&self) {
        if let Err(e) = self.persist_stats() {
            eprintln!("mtxdb: failed to persist shard IO/sync stats: {e}");
        }
    }

    /// Sync all shards to disk.
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_all(&self) -> io::Result<()> {
        {
            let shards = self.shards.read();
            for shard in shards.iter().flatten() {
                shard.file.sync_all()?;
                shard.sync_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.persist_stats_best_effort();
        Ok(())
    }

    /// Sync only shards written to since the last sync.
    ///
    /// Each shard's dirty bit is cleared only after a successful fsync,
    /// so a partial failure leaves the untried shards marked dirty for
    /// the next call.
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_dirty(&self) -> io::Result<()> {
        let dirty: Vec<u16> = { self.dirty.lock().iter().copied().collect() };
        if dirty.is_empty() {
            return Ok(());
        }
        {
            let shards = self.shards.read();
            for &id in &dirty {
                if let Some(shard) = shards.get(id as usize).and_then(|s| s.as_ref()) {
                    shard.file.sync_all()?;
                    shard.sync_count.fetch_add(1, Ordering::Relaxed);
                    self.dirty.lock().remove(&id);
                }
            }
        }
        self.persist_stats_best_effort();
        Ok(())
    }

    /// Retire a shard: mark it for deletion and free its pool slot.
    ///
    /// The shard file is deleted when the last `Arc<Shard>` reference drops
    /// (via `Shard::drop` when `is_current == false`). Setting the slot to
    /// `None` allows `rotate()` to reuse it.
    ///
    /// Does nothing if the slot is already empty or is the active write shard.
    pub fn retire_slot(&self, shard_id: u16) {
        let mut shards = self.shards.write();
        if *self.active_write.lock() == shard_id {
            return;
        }
        if let Some(slot) = shards.get_mut(shard_id as usize) {
            if let Some(shard) = slot.take() {
                shard.is_current.store(false, Ordering::Release);
                self.dirty.lock().remove(&shard_id);
                self.retired_count.fetch_add(1, Ordering::Relaxed);
                drop(shards);
                // Any room whose home was this slot must re-home on its
                // next write — otherwise, once the slot is reused, that
                // room's writes would silently land in an unrelated
                // shard that happens to have been assigned the same slot.
                self.room_home.write().retain(|_, home| *home != shard_id);
            }
        }
    }

    /// Total number of shards retired over the pool's lifetime.
    #[must_use]
    pub fn retired_count(&self) -> u64 {
        self.retired_count.load(Ordering::Relaxed)
    }

    /// Scan a shard file and return `(room_id, hash, offset)` entries.
    /// Used during startup to rebuild per-room indexes.
    ///
    /// # Errors
    /// Returns `io::Error` on file open or header read failure.
    pub fn scan_shard(path: &Path) -> io::Result<Vec<ShardEntry>> {
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
                Err(e) => return Err(e),
            }
        }

        Ok(entries)
    }
}

impl Drop for ShardPool {
    /// Best-effort final flush of IO/sync stats. Errors are swallowed —
    /// a failed stats write on shutdown must never panic or mask the
    /// original error path the caller was already on.
    fn drop(&mut self) {
        self.persist_stats_best_effort();
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
    fn test_sync_dirty_noop_when_nothing_written() {
        let dir = test_dir("sync_dirty_noop");
        let pool = ShardPool::open(dir).unwrap();
        assert!(pool.dirty.lock().is_empty());
        pool.sync_dirty().unwrap();
        assert!(pool.dirty.lock().is_empty());
    }

    #[test]
    fn test_sync_dirty_clears_only_written_shards() {
        let dir = test_dir("sync_dirty_clears");
        let pool = ShardPool::open(dir).unwrap();

        let record = test_record(0x01, 0xCC, b"needs sync");
        let (shard_id, _offset) = pool.put_record(&record).unwrap();
        assert!(pool.dirty.lock().contains(&shard_id));

        pool.sync_dirty().unwrap();
        assert!(
            pool.dirty.lock().is_empty(),
            "dirty bit must clear after a successful sync"
        );

        // A second sync with nothing new written is a no-op, not an error.
        pool.sync_dirty().unwrap();
    }

    /// A real fsync succeeding must never be turned into a hard error by a
    /// failure in the best-effort stats snapshot write -- e.g. the base
    /// directory getting removed out from under a live pool (test teardown,
    /// or any other external interference) must not make `sync_all`/
    /// `sync_dirty` (and by extension every `maybe_sync(DURABLE)` caller
    /// upstream) return `Err` when the actual shard data is safely synced.
    #[test]
    fn test_sync_survives_persist_stats_failure() {
        let dir = test_dir("sync_survives_stats_failure");
        let pool = ShardPool::open(dir.clone()).unwrap();

        let record = test_record(0x01, 0xDD, b"data that must stay durable");
        pool.put_record(&record).unwrap();

        // Remove the base directory itself, so persist_stats' File::create
        // for its temp file fails with ENOENT -- while the shard's already-
        // open file descriptor (and thus its real fsync) is unaffected.
        fs::remove_dir_all(&dir).unwrap();

        pool.sync_all()
            .expect("sync_all must succeed even if stats persistence fails");
        pool.put_record(&test_record(0x01, 0xEE, b"more data"))
            .unwrap();
        pool.sync_dirty()
            .expect("sync_dirty must succeed even if stats persistence fails");
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

    /// Core fix: a room's writes must stay on its own home shard even
    /// after *unrelated* activity rotates the pool's global active-write
    /// cursor far ahead. Before per-room home routing, every room simply
    /// followed that single pool-wide cursor, so any other room's churn
    /// (with nothing to do with room A, and no capacity reason for room
    /// A specifically to move) would silently redirect room A's next
    /// write too — destroying locality room A never had a reason to lose.
    #[test]
    fn test_room_stays_on_home_shard_despite_unrelated_pool_rotation() {
        let dir = test_dir("room_locality");
        let pool = ShardPool::open(dir).unwrap();

        // Room A's first write establishes its home on shard 0.
        let room_a = test_record(0x01, 0x01, b"room A first");
        let (shard_a1, _) = pool.put_record(&room_a).unwrap();
        assert_eq!(shard_a1, 0);

        // Simulate unrelated churn (other rooms' own rotations) dragging
        // the pool-wide cursor far ahead — room A is not involved at all,
        // and shard 0 still has essentially all its capacity free.
        for _ in 0..5 {
            pool.rotate().unwrap();
        }

        // Room A writes again: it must still land on shard 0, its own
        // home — not wherever unrelated rotations left the pool cursor.
        let room_a2 = test_record(0x01, 0x03, b"room A second");
        let (shard_a2, _) = pool.put_record(&room_a2).unwrap();
        assert_eq!(
            shard_a2, 0,
            "room A must stay on its own home shard, unaffected by unrelated pool rotation"
        );
    }

    /// Once a room's own home shard actually fills up, that room (and
    /// only that room) rotates to a new home — independent of whatever
    /// the pool-wide cursor is doing for other rooms.
    #[test]
    fn test_room_rotates_its_own_home_when_full() {
        let dir = test_dir("room_locality_own_rotation");
        let pool = ShardPool::open(dir).unwrap();

        let room_a = test_record(0x01, 0x01, b"room A first");
        let (shard_a1, _) = pool.put_record(&room_a).unwrap();
        assert_eq!(shard_a1, 0);

        // Fill room A's own home shard (shard 0) — not some other shard —
        // and confirm room A itself rotates off it.
        pool.get_shard(0)
            .unwrap()
            .file_len
            .store(MAX_SHARD_BYTES - 10, Ordering::Release);
        let room_a2 = test_record(0x01, 0x02, b"room A triggers its own rotation");
        let (shard_a2, _) = pool.put_record(&room_a2).unwrap();
        assert_eq!(shard_a2, 1);

        // And room A stays on its new home from then on.
        let room_a3 = test_record(0x01, 0x03, b"room A third");
        let (shard_a3, _) = pool.put_record(&room_a3).unwrap();
        assert_eq!(shard_a3, 1);
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

        let entries = ShardPool::scan_shard(&pool.get_shard(0).unwrap().path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0[0], 0x01);
        assert_eq!(entries[0].1[0], 0x10);
        assert_eq!(entries[2].0[0], 0x01);
        assert_eq!(entries[2].1[0], 0x11);
    }

    /// A retired shard's file must survive as long as any `Arc<Shard>`
    /// reference is held (e.g. via `PackfileStorage::pin_shards`, so a
    /// repack reading stale offsets from it can't be undercut), and must
    /// be deleted once the last reference actually drops. This is the
    /// invariant `pin_shards` relies on to fix the shard-rotation race:
    /// pinning a shard for a repack's duration keeps this `Drop` from
    /// firing early, regardless of what else happens to the pool's own
    /// slot for that shard in the meantime.
    #[test]
    fn test_drop_deletes_retired_shard_only_after_last_reference() {
        let dir = test_dir("drop_retire");
        let pool = ShardPool::open(dir).unwrap();

        let record = test_record(0x01, 0xAA, b"payload");
        let (shard_id, _offset) = pool.put_record(&record).unwrap();

        let pinned = pool.get_shard(shard_id).unwrap();
        let path = pinned.path.clone();
        assert!(path.exists());

        // Simulate a future repack-driven retirement (nothing currently
        // does this — see rotate()'s doc — but pin_shards must protect
        // against it regardless of how it eventually gets triggered), then
        // drop the pool's own reference the way replacing a recycled slot
        // would.
        pinned.is_current.store(false, Ordering::Release);
        drop(pool);

        // The pinned clone is still held, so the file must survive.
        assert!(
            path.exists(),
            "file deleted while a pinned Arc<Shard> was still held"
        );

        drop(pinned);
        assert!(
            !path.exists(),
            "retired shard's file should be deleted once its last reference drops"
        );
    }

    /// Crash-recovery: if two generation files exist for the same slot
    /// (e.g. a crash landed between `rotate()` creating a new generation
    /// and the old generation's `Drop` deleting its file), the scan must
    /// keep the higher generation and delete the stale one.
    #[test]
    fn test_scan_keeps_highest_generation_per_slot() {
        let dir = test_dir("scan_generation_dedup");
        let pool = ShardPool::open(dir.clone()).unwrap();

        // Write a record so shard 0 exists (generation 0).
        let record = test_record(0x01, 0xAA, b"live data");
        pool.put_record(&record).unwrap();
        drop(pool);

        // Promote the live file to generation 99 (simulating that rotate()
        // created a newer generation before a crash left a stale gen-0 file
        // behind).
        let live_path = ShardPool::shard_path(&dir, 0, 99);
        let old_path = ShardPool::shard_path(&dir, 0, 0);
        std::fs::rename(&old_path, &live_path).unwrap();

        // Create a stale gen-0 file (the crash leftover).
        let mut buf = Vec::new();
        packfile::write_header(&mut buf).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                room_id: [0xFF; 16],
                hash: [0xBB; 16],
                data: bytes::Bytes::from_static(b"stale leftover"),
            },
        )
        .unwrap();
        std::fs::write(&old_path, &buf).unwrap();

        // Both files exist on disk.
        assert!(live_path.exists(), "live gen-99 file missing");
        assert!(old_path.exists(), "stale gen-0 file missing");

        // Reopen the pool — the scan must keep gen 99, delete gen 0.
        let pool = ShardPool::open(dir.clone()).unwrap();
        let shard = pool.get_shard(0).unwrap();
        assert_eq!(
            shard.generation, 99,
            "scan should have kept the higher generation"
        );
        assert_eq!(shard.path, live_path);
        drop(pool);

        // The stale gen-0 file must have been cleaned up by the scan.
        assert!(
            !old_path.exists(),
            "stale generation file should be deleted during scan"
        );
    }

    /// `stats()` must reflect actual write/read/sync activity, so callers
    /// can distinguish "many small fsyncs" (seek-bound, noisy) from
    /// "few large writes, batched syncs" (quiet) without guessing from
    /// disk sound. Deliberately no read counters — see `ShardStats`' doc
    /// for why counting reads isn't worth the cache-line contention on
    /// that path.
    #[test]
    fn test_shard_stats_track_write_sync() {
        let dir = test_dir("stats_tracking");
        let pool = ShardPool::open(dir).unwrap();

        let record = test_record(0x01, 0xAA, b"payload for stats");
        let (shard_id, offset) = pool.put_record(&record).unwrap();

        let shard = pool.get_shard(shard_id).unwrap();
        let stats = shard.stats();
        assert_eq!(stats.write_count, 1);
        assert_eq!(stats.bytes_written, record.serialized_len() as u64);
        assert_eq!(stats.sync_count, 0);

        ShardPool::read_at(&shard, offset).unwrap();

        pool.sync_dirty().unwrap();
        let stats = shard.stats();
        assert_eq!(stats.sync_count, 1, "sync_dirty must bump sync_count");

        // A second sync_dirty with nothing new written must not double-count.
        pool.sync_dirty().unwrap();
        assert_eq!(shard.stats().sync_count, 1);

        let all = pool.all_stats();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, shard_id);
        assert_eq!(all[0].1, shard.stats());

        assert_eq!(pool.stats(shard_id), Some(shard.stats()));
        assert_eq!(pool.stats(shard_id.wrapping_add(1)), None);
    }

    /// Backward compatibility: the startup scan must correctly parse all
    /// three filename formats that can exist on disk:
    ///   - `shard_XX.pack` — ancient (2-digit, no generation, implied gen 0)
    ///   - `shard_XX_YYYYYYYYYYYYYYYY.pack` — current (2-digit + generation)
    ///   - `shard_XXXX_YYYYYYYYYYYYYYYY.pack` — new (4-digit + generation)
    #[test]
    fn test_scan_parses_three_filename_formats() {
        let dir = test_dir("scan_three_formats");

        // Manually create one file in each format.
        let make_pack = |room_byte: u8| -> Vec<u8> {
            let mut buf = Vec::new();
            packfile::write_header(&mut buf).unwrap();
            packfile::write_record(
                &mut buf,
                &packfile::Record {
                    room_id: {
                        let mut r = [0u8; 16];
                        r[0] = room_byte;
                        r
                    },
                    hash: [0xAA; 16],
                    data: bytes::Bytes::from_static(b"payload"),
                },
            )
            .unwrap();
            buf
        };

        // Slot 0: legacy format "shard_00.pack" → gen 0
        std::fs::write(dir.join("shard_00.pack"), make_pack(0x01)).unwrap();

        // Slot 1: 2-digit with generation "shard_01_0000000000000005.pack" → gen 5
        std::fs::write(dir.join("shard_01_0000000000000005.pack"), make_pack(0x02)).unwrap();

        // Slot 2: 4-digit with generation "shard_0002_0000000000000003.pack" → gen 3
        std::fs::write(
            dir.join("shard_0002_0000000000000003.pack"),
            make_pack(0x03),
        )
        .unwrap();

        let pool = ShardPool::open(dir).unwrap();

        let s0 = pool.get_shard(0).unwrap();
        assert_eq!(s0.generation, 0, "legacy shard_00.pack → gen 0");
        assert_eq!(s0.shard_id, 0);

        let s1 = pool.get_shard(1).unwrap();
        assert_eq!(s1.generation, 5, "shard_01_...0005.pack → gen 5");
        assert_eq!(s1.shard_id, 1);

        let s2 = pool.get_shard(2).unwrap();
        assert_eq!(s2.generation, 3, "shard_0002_...0003.pack → gen 3");
        assert_eq!(s2.shard_id, 2);

        // Slot 3 was never created; pool should have no shard there.
        assert!(pool.get_shard(3).is_none());
    }
}
