use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::packfile::{self, Record};

/// Maximum number of shards in the pool.
pub const MAX_SHARDS: usize = 4096;

/// Maximum number of shards as `u16`. Primary constant for shard IDs
/// and modular arithmetic.
pub(crate) const MAX_SHARDS_U16: u16 = 4096;

/// Maximum shard size before rotation: `2^28 - 1` bytes (~256 MB), the
/// largest value for which every offset a shard can ever produce still
/// fits `IndexSlot`'s 28-bit offset field (which reserves its all-zero
/// encoding as the empty-slot sentinel, capping the max representable
/// offset at `2^28 - 2` — see `index::IndexSlot`). Do not round this up
/// to a clean `256 * 1024 * 1024`: that's one byte over the ceiling and
/// lets a shard produce an offset `IndexSlot::new` panics on.
pub const MAX_SHARD_BYTES: u64 = (1u64 << 28) - 1;

/// Scanned `(room_id, hash, offset)` entry from a shard file.
pub type ShardEntry = ([u8; 16], [u8; 16], u64);

/// A single global shard file shared across all rooms.
pub struct Shard {
    /// Pool-local allocator slot (index into the open-shard table).
    /// Ephemeral, recycled on retire. Never shown to operators.
    pub slot: u16,
    /// Globally monotonic file epoch, distinct from the slot index.
    /// Ensures a reused slot never collides on-disk with a still-referenced
    /// older file at that slot.
    pub epoch: u64,
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

/// Basic size, epoch, and IO/sync info for one open shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSummary {
    /// Pool-local allocator slot (ephemeral).
    pub slot: u16,
    /// Globally monotonic file epoch for this shard incarnation.
    pub epoch: u64,
    /// Current on-disk file length in bytes.
    pub file_bytes: u64,
    /// IO/sync counters for this shard.
    pub stats: ShardStats,
}

/// Filename for the persisted stats snapshot, stored alongside shard files.
const STATS_FILENAME: &str = "shard_stats.bin";

/// Magic bytes + version identifying the stats file format.
const STATS_MAGIC: &[u8; 4] = b"MSTA";
/// v3 adds an 8-byte persisted-at unix-seconds timestamp right after the
/// version byte, so a reader (e.g. `mtxdb shards`) can tell how stale a
/// snapshot is instead of just trusting whatever numbers happen to be on
/// disk. A v2 file is simply not restored — best-effort, same as any
/// other unreadable snapshot — rather than migrated in place.
const STATS_VERSION: u8 = 3;

/// On-disk size of one stats record: `slot_id`(2) + epoch(8) + 3×counter(8) = 34 bytes.
const STATS_RECORD_LEN: usize = 2 + 8 + 8 * 3;

/// Header size: magic(4) + version(1) + `persisted_at`(8).
const STATS_HEADER_LEN: usize = 4 + 1 + 8;

/// Disambiguates concurrent `persist_stats` tmp filenames within this
/// process (paired with the process id, which disambiguates across
/// processes sharing the same `base_dir`).
static STATS_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl ShardStats {
    fn encode(self, shard_id: u16, epoch: u64, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&shard_id.to_le_bytes());
        buf.extend_from_slice(&epoch.to_le_bytes());
        buf.extend_from_slice(&self.write_count.to_le_bytes());
        buf.extend_from_slice(&self.bytes_written.to_le_bytes());
        buf.extend_from_slice(&self.sync_count.to_le_bytes());
    }

    fn decode(rec: &[u8; STATS_RECORD_LEN]) -> (u16, u64, Self) {
        let shard_id = u16::from_le_bytes(rec[0..2].try_into().unwrap());
        let epoch = u64::from_le_bytes(rec[2..10].try_into().unwrap());
        let write_count = u64::from_le_bytes(rec[10..18].try_into().unwrap());
        let bytes_written = u64::from_le_bytes(rec[18..26].try_into().unwrap());
        let sync_count = u64::from_le_bytes(rec[26..34].try_into().unwrap());
        (
            shard_id,
            epoch,
            Self {
                write_count,
                bytes_written,
                sync_count,
            },
        )
    }
}

impl Shard {
    fn new(slot: u16, epoch: u64, file: File, path: PathBuf, file_len: u64) -> Self {
        Self {
            slot,
            epoch,
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

/// Holds the writer's exclusive claim on a `base_dir` (see
/// `ShardPool::acquire_writer_lock`). Removing the marker file on drop is
/// what makes a clean shutdown release the lock instantly, same as real
/// `flock` releasing on fd close — a crash instead leaves it for the next
/// opener's staleness check (`lock_holder_is_dead`) to reclaim.
#[cfg(not(target_arch = "wasm32"))]
struct WriterLock {
    path: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WriterLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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
    /// Globally monotonic epoch counter for shard filenames.
    /// Each newly created shard file gets a unique epoch, so a
    /// reused slot never collides on-disk with a still-referenced old
    /// shard at the same slot.
    next_epoch: AtomicU64,
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
    /// Unix-seconds timestamp of the currently-persisted stats snapshot,
    /// if one has ever been restored or written by this pool — restored
    /// at open time from an existing `shard_stats.bin`, and updated on
    /// every successful `persist_stats`. Lets a caller (e.g. `mtxdb
    /// shards`) show how stale the counters it's displaying are, since a
    /// read-only pool only ever sees whatever the real writer last
    /// flushed, not live in-process counters.
    stats_persisted_at: RwLock<Option<u64>>,
    /// Wall-clock instant of the last `maybe_persist_stats` flush (or
    /// `None` if it's never been called), used to rate-limit that
    /// timer-driven path — separate from `stats_persisted_at`, which is
    /// the on-disk snapshot's own unix-seconds timestamp.
    last_stats_flush: RwLock<Option<Instant>>,
    /// Whether this pool holds the writer lock on `base_dir` (see `open`
    /// vs `open_read_only`). Gates `persist_stats`: a read-only pool
    /// never writes anything, including its own (always-zero) stats
    /// snapshot, so it can't race the real writer's.
    writable: bool,
    /// Present only for a writable pool — `open_read_only` takes no lock
    /// at all, since it never writes or truncates anything (that risk
    /// lives one layer up, in `PackfileStorage`'s room-index rebuild, not
    /// here), so there's nothing for a reader to need exclusivity from.
    ///
    /// Never read: this is an RAII guard held purely for its `Drop` side
    /// effect (removing the `.mtxdb.lock` marker file when the pool
    /// itself drops). Its value is deliberately never inspected — only
    /// its lifetime matters — so `#[allow(dead_code)]` here is correct,
    /// not a lint dodge: the field genuinely has no read access by
    /// design, the same way a `MutexGuard` binding is never "used" either.
    /// `None` on wasm32, where there's no cross-process model to guard.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    writer_lock: Option<WriterLock>,
}

impl ShardPool {
    /// Open or create a shard pool as its exclusive writer, scanning for
    /// existing shard files.
    ///
    /// Acquires an exclusive lock on `base_dir` (a `.mtxdb.lock` marker
    /// file within it) for the pool's lifetime. A second writer opening
    /// the same directory while this one is alive fails fast here instead
    /// of silently risking corruption — nothing else in this engine
    /// coordinates concurrent writers across processes; `active_write`,
    /// `file_len`, and `append_lock` are all purely in-process state, so
    /// two processes racing on the same shard file would interleave
    /// writes into it with no protection at all.
    ///
    /// # Errors
    /// Returns `io::Error` on directory read failure, packfile open
    /// failure, or if another process already holds the writer lock.
    pub fn open(base_dir: PathBuf) -> io::Result<Self> {
        Self::open_internal(base_dir, true)
    }

    /// Open a shard pool as a read-only observer, coexisting with a
    /// concurrent writer on the same directory (or none at all).
    ///
    /// Takes no lock at all: unlike `open`, this never writes or
    /// truncates anything — no shard creation on an empty directory
    /// (errors instead: a read-only open of a store that doesn't exist
    /// yet makes no sense), no persisted-stats snapshot (its own
    /// counters, on a pool that never writes, would always be zero) —
    /// so there's nothing here for a writer to need protecting from.
    ///
    /// # Errors
    /// Returns `io::Error` on directory read failure, or if the directory
    /// has no shards yet.
    pub fn open_read_only(base_dir: PathBuf) -> io::Result<Self> {
        Self::open_internal(base_dir, false)
    }

    /// Discovers existing `shard_*.pack` files in `base_dir` and parses
    /// each one's `(slot_id, epoch, path)` from its filename.
    /// Supports three filename formats:
    ///   `shard_XX.pack`                      — legacy (no epoch, implied epoch 0)
    ///   `shard_XX_YYYYYYYYYYYYYYYY.pack`      — 2-digit slot, epoch-suffixed
    ///   `shard_XXXX_YYYYYYYYYYYYYYYY.pack`    — 4-digit slot, epoch-suffixed
    fn discover_shard_files(base_dir: &Path) -> io::Result<Vec<(u16, u64, PathBuf)>> {
        let mut shard_files = Vec::new();
        for entry in fs::read_dir(base_dir)? {
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
            let (slot_hex, epoch) = match id_hex.split_once('_') {
                Some((slot, gen_hex)) => {
                    let gen = u64::from_str_radix(gen_hex, 16).unwrap_or(0);
                    (slot, gen)
                }
                None => (id_hex, 0),
            };

            if let Ok(id) = u16::from_str_radix(slot_hex, 16) {
                if id < MAX_SHARDS_U16 {
                    shard_files.push((id, epoch, path));
                }
            }
        }
        Ok(shard_files)
    }

    fn open_internal(base_dir: PathBuf, writable: bool) -> io::Result<Self> {
        fs::create_dir_all(&base_dir)?;

        #[cfg(not(target_arch = "wasm32"))]
        let writer_lock = writable
            .then(|| Self::acquire_writer_lock(&base_dir))
            .transpose()?;

        let mut shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS).map(|_| None).collect();
        let mut highest_active: u16 = 0;
        let mut max_epoch: u64 = 0;

        // Track the highest epoch seen per slot so we can reject
        // stale files left behind by a crash between rotate() creating a
        // new epoch and the old epoch's Drop deleting it.
        let mut best_epoch: Vec<Option<u64>> = vec![None; MAX_SHARDS];

        let mut shard_files = Self::discover_shard_files(&base_dir)?;

        // Process older epochs first. Besides making recovery
        // deterministic, this means a malformed newer candidate can fall
        // back to the already-validated predecessor.
        shard_files.sort_unstable_by_key(|(id, epoch, _)| (*id, *epoch));

        for (id, epoch, path) in shard_files {
            let id_usize = id as usize;

            // If we already have an epoch for this slot and the current
            // file is not newer, it's stale — skip it (delete only when
            // writable; a read-only open must never mutate files).
            if let Some(prev) = best_epoch[id_usize] {
                if prev >= epoch {
                    if writable {
                        let _ = fs::remove_file(&path);
                    }
                    continue;
                }
            }

            // Validate a newer candidate before retiring the older
            // epoch. A torn header in a just-created file must
            // not destroy the last known-good recovery fallback.
            let file = match packfile::open_packfile(&path, false, id, epoch) {
                Ok(file) => file,
                Err(error) if shards[id_usize].is_some() => {
                    eprintln!(
                        "warning: ignoring invalid newer shard {} while retaining its valid predecessor: {error}",
                        path.display()
                    );
                    continue;
                }
                Err(error) => return Err(error),
            };

            // The candidate is valid, so it is now safe to retire
            // the older epoch for this slot (only when writable).
            if let Some(old) = shards[id_usize].take() {
                if writable {
                    let _ = fs::remove_file(&old.path);
                }
            }
            let file_len = file.metadata()?.len();
            let shard = Arc::new(Shard::new(id, epoch, file, path, file_len));
            best_epoch[id_usize] = Some(epoch);
            shards[id_usize] = Some(shard);
            if id > highest_active {
                highest_active = id;
            }
            if epoch >= max_epoch {
                max_epoch = epoch.saturating_add(1);
            }
        }

        // If no shards exist, create the initial shard 0 — but a read-only
        // open of a store that doesn't exist yet makes no sense; error
        // instead of a read-only pool silently creating on-disk state.
        if shards.iter().all(std::option::Option::is_none) {
            if !writable {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "no shards found in {} (nothing to read)",
                        base_dir.display()
                    ),
                ));
            }
            let path = Self::shard_path(&base_dir, 0, 0);
            let file = packfile::open_packfile(&path, true, 0, 0)?;
            let file_len = file.metadata()?.len();
            shards[0] = Some(Arc::new(Shard::new(0, 0, file, path, file_len)));
            max_epoch = 1;
        }

        // Restoring is a pure read of shard_stats.bin applied to our own
        // in-memory Shard objects — unconditional regardless of writable.
        // A read-only pool must never *write* a new snapshot (its own
        // counters are always zero, since it never writes or syncs), but
        // it absolutely should show whatever the real writer already
        // persisted — that's the entire point of a `shards`-style
        // inspection tool being able to see real numbers at all.
        let stats_persisted_at = Self::restore_persisted_stats(&base_dir, &shards);

        Ok(Self {
            shards: RwLock::new(shards),
            active_write: parking_lot::Mutex::new(highest_active),
            rotation_lock: parking_lot::Mutex::new(()),
            base_dir,
            dirty: parking_lot::Mutex::new(HashSet::new()),
            next_epoch: AtomicU64::new(max_epoch),
            retired_count: AtomicU64::new(0),
            room_home: RwLock::new(HashMap::new()),
            stats_persisted_at: RwLock::new(stats_persisted_at),
            last_stats_flush: RwLock::new(None),
            writable,
            #[cfg(not(target_arch = "wasm32"))]
            writer_lock,
        })
    }

    /// Claim the writer lock on `base_dir`: atomically create a
    /// `.mtxdb.lock` marker file, which fails with `AlreadyExists` if
    /// another writer already holds it. Pure `std::fs` — no OS-level
    /// advisory-lock API and no dependency, so it needs one thing real
    /// `flock` gives for free: recovery if the previous holder crashed
    /// without cleaning up (`WriterLock`'s `Drop` handles the clean-exit
    /// case). We store our PID in the file and, if creation fails because
    /// it already exists, check whether that PID is still alive
    /// (`/proc/<pid>` on Linux) before concluding the lock is genuinely
    /// held — a stale file from a killed process is removed and retried
    /// once rather than wrongly blocking forever.
    #[cfg(not(target_arch = "wasm32"))]
    fn acquire_writer_lock(base_dir: &Path) -> io::Result<WriterLock> {
        let lock_path = base_dir.join(".mtxdb.lock");
        match Self::try_create_lock_file(&lock_path) {
            Ok(()) => Ok(WriterLock { path: lock_path }),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                if Self::lock_holder_is_dead(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    Self::try_create_lock_file(&lock_path)?;
                    return Ok(WriterLock { path: lock_path });
                }
                Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!(
                        "{} is already locked by another writer process",
                        base_dir.display()
                    ),
                ))
            }
            Err(e) => Err(e),
        }
    }

    /// Atomically create the lock file and write our PID into it. No
    /// `sync_all` here: the PID is advisory only, and `lock_holder_is_dead`
    /// already fails closed (treats an unparsable file as "might be
    /// alive") on a torn write from a crash mid-write — there's no
    /// correctness reason to pay an fsync on every lock acquisition to
    /// protect against that.
    #[cfg(not(target_arch = "wasm32"))]
    fn try_create_lock_file(lock_path: &Path) -> io::Result<()> {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(lock_path)?;
        write!(file, "{}", std::process::id())
    }

    /// Liveness check for whoever wrote `lock_path`'s PID. Only verifies
    /// anything on Linux (`/proc/<pid>`); everywhere else this
    /// conservatively assumes the holder might still be alive.
    ///
    /// A time-based fallback ("reclaim if the lock file is older than N")
    /// was considered and rejected: a long-lived, mostly-idle writer (a
    /// Synapse process sitting quiet for hours is normal, not an edge
    /// case) looks indistinguishable from an abandoned lock once it
    /// crosses any fixed age threshold, since nothing refreshes the
    /// timestamp while the lock is held — that trades a rare stuck-lock
    /// annoyance (fixable by deleting the file) for an occasional silent
    /// second writer and packfile corruption, the wrong side of that
    /// trade. A real fix needs a refreshed heartbeat/lease or a real OS
    /// advisory lock per platform, not a bare creation timestamp; until
    /// one of those lands, non-Linux platforms fail closed and require
    /// manual recovery from a hard crash.
    #[cfg(not(target_arch = "wasm32"))]
    fn lock_holder_is_dead(lock_path: &Path) -> bool {
        #[cfg(target_os = "linux")]
        {
            let Ok(contents) = fs::read_to_string(lock_path) else {
                return false;
            };
            let Ok(pid) = contents.trim().parse::<u32>() else {
                return false;
            };
            !Path::new(&format!("/proc/{pid}")).exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = lock_path;
            false
        }
    }

    /// Seed a room's home shard — used at startup to approximate where a
    /// room's most recent data already lives (from a content scan elsewhere,
    /// since `ShardPool::open` itself only discovers shard *files*, not
    /// their room contents). Normal routing updates the home automatically
    /// from then on via `put_record`.
    pub(crate) fn set_room_home(&self, room_id: &[u8; 16], slot: u16) {
        self.room_home.write().insert(*room_id, slot);
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
        self.room_home.write().insert(*room_id, shard.slot);
        shard
    }

    /// A room's home shard just filled up: rotate the pool forward (unless
    /// another room already did, in which case just adopt whatever's now
    /// active) and point the room at the result.
    fn rotate_room_full_home(&self, room_id: &[u8; 16], full_slot: u16) -> io::Result<Arc<Shard>> {
        {
            let active = self.active_write.lock();
            if *active == full_slot {
                drop(active);
                self.rotate()?;
            }
        }
        let shard = self.active_shard();
        self.room_home.write().insert(*room_id, shard.slot);
        Ok(shard)
    }

    /// Path to the persisted stats snapshot for a base directory.
    fn stats_path(base_dir: &Path) -> PathBuf {
        base_dir.join(STATS_FILENAME)
    }

    /// Load a persisted stats snapshot, if one exists, and restore each
    /// shard's counters when its epoch still matches — a stale
    /// snapshot entry (from a slot since retired and reused) is silently
    /// skipped rather than misapplied. Returns the snapshot's persisted-at
    /// unix timestamp, if the file was readable and current-format.
    ///
    /// Best-effort: a missing, truncated, or corrupt file just means no
    /// stats are restored — never a startup failure over stats alone.
    fn restore_persisted_stats(base_dir: &Path, shards: &[Option<Arc<Shard>>]) -> Option<u64> {
        let path = Self::stats_path(base_dir);
        let buf = fs::read(&path).ok()?;
        if buf.len() < STATS_HEADER_LEN || &buf[0..4] != STATS_MAGIC || buf[4] != STATS_VERSION {
            return None;
        }
        let persisted_at = u64::from_le_bytes(buf[5..13].try_into().ok()?);
        let body = &buf[STATS_HEADER_LEN..];
        for chunk in body.chunks(STATS_RECORD_LEN) {
            let Ok(rec) = <&[u8; STATS_RECORD_LEN]>::try_from(chunk) else {
                break;
            };
            let (shard_id, epoch, stats) = ShardStats::decode(rec);
            if let Some(Some(shard)) = shards.get(shard_id as usize) {
                if shard.epoch == epoch {
                    shard.restore_stats(stats);
                }
            }
        }
        Some(persisted_at)
    }

    /// Persist every currently-open shard's IO/sync counters to disk,
    /// keyed by `(slot_id, epoch)` so a retired/reused slot's stale
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
        let persisted_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        buf.extend_from_slice(&persisted_at.to_le_bytes());
        for (slot, shard) in self.all_shards() {
            shard.stats().encode(slot, shard.epoch, &mut buf);
        }

        // Unique per (process, call) — the same base_dir can be opened by
        // more than one process at once (e.g. a long-running embedder plus
        // a short-lived `mtxdb shards` CLI invocation), and a fixed tmp
        // filename shared across them races: one process's rename() can
        // consume the shared tmp path out from under another's, which
        // then sees ENOENT on its own rename despite having written its
        // tmp file successfully moments earlier.
        let unique = STATS_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = Self::stats_path(&self.base_dir)
            .with_extension(format!("bin.tmp.{}.{unique}", std::process::id()));
        let final_path = Self::stats_path(&self.base_dir);
        let write_result = (|| -> io::Result<()> {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&buf)?;
            tmp.sync_all()
        })();
        if let Err(e) = write_result {
            // Don't leave a half-written tmp file behind under its
            // now-unique name — best-effort, the write already failed.
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        fs::rename(&tmp_path, &final_path)?;
        *self.stats_persisted_at.write() = Some(persisted_at);
        Ok(())
    }

    /// On-disk path for a shard file.
    #[must_use]
    pub fn shard_path(base_dir: &Path, shard_id: u16, epoch: u64) -> PathBuf {
        base_dir.join(format!("shard_{shard_id:04x}_{epoch:016x}.pack"))
    }

    /// Get a reference to a shard by ID.
    #[must_use]
    pub fn get_shard(&self, slot: u16) -> Option<Arc<Shard>> {
        self.shards.read().get(slot as usize)?.clone()
    }

    /// Get IO/sync stats for a single shard by ID.
    #[must_use]
    pub fn stats(&self, slot: u16) -> Option<ShardStats> {
        self.shards
            .read()
            .get(slot as usize)?
            .as_ref()
            .map(|shard| shard.stats())
    }

    /// Snapshot IO/sync stats for every currently-open shard, as
    /// `(slot, ShardStats)` pairs.
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

    /// List every currently-open shard with basic size, epoch, and
    /// IO/sync stats. Needs no room data at all — the shard-only path a
    /// `mtxdb shards`-style inspection tool should use directly (via
    /// `open_read_only`) rather than opening a full `PackfileStorage`,
    /// which rebuilds every room's index and thus needs the writer lock.
    #[must_use]
    pub fn summaries(&self) -> Vec<ShardSummary> {
        self.all_shards()
            .into_iter()
            .map(|(slot, shard)| ShardSummary {
                slot,
                epoch: shard.epoch,
                file_bytes: shard.file_len(),
                stats: shard.stats(),
            })
            .collect()
    }

    /// Return all currently-open shards as `(slot, Arc<Shard>)` pairs.
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
    /// Returns `(slot, offset)`. Rotates the room to a new home shard
    /// if its current one is full.
    ///
    /// # Errors
    /// Returns `io::Error` on write or rotation failure.
    pub fn put_record(&self, record: &Record) -> io::Result<(u16, u64)> {
        let mut shard = self.shard_for_room(&record.room_id);
        loop {
            // Uncompressed upper bound, used only for the pre-write
            // capacity check below — `write_record` may compress the
            // payload and write fewer bytes than this, but never more,
            // so checking against this bound never lets a shard overflow
            // MAX_SHARD_BYTES; it may just rotate a little earlier than
            // strictly necessary when compression would have made it fit.
            let max_record_len = record.serialized_len() as u64;

            let (offset, record_len) = {
                let guard = shard.append_lock.lock();
                let mut file = shard.file.try_clone()?;
                let offset = file.seek(io::SeekFrom::End(0))?;

                // Check capacity while holding the append lock and after
                // seeking to the true end — avoids TOCTOU race where two
                // threads both pass the check then one exceeds the limit.
                let current_len = shard.file_len.load(Ordering::Acquire);
                let fits = current_len
                    .checked_add(max_record_len)
                    .is_some_and(|sum| sum <= MAX_SHARD_BYTES);
                if !fits && current_len > packfile::HEADER_LEN as u64 {
                    drop(guard);
                    drop(file);
                    shard = self.rotate_room_full_home(&record.room_id, shard.slot)?;
                    continue;
                }

                // The actual on-disk length — may be smaller than
                // `max_record_len` when the payload compressed.
                let record_len = packfile::write_record(&mut file, record)?;
                let new_len = offset.checked_add(record_len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "offset + record_len overflow")
                })?;
                shard.file_len.store(new_len, Ordering::Release);
                (offset, record_len)
            };

            shard.write_count.fetch_add(1, Ordering::Relaxed);
            shard.bytes_written.fetch_add(record_len, Ordering::Relaxed);
            self.dirty.lock().insert(shard.slot);
            return Ok((shard.slot, offset));
        }
    }

    /// The actual on-disk byte length of the frame at `offset` — length
    /// prefix (4) + frame body + CRC (4). Unlike [`serialized_len`]
    /// (an uncompressed upper bound computed from a decoded `Record`),
    /// this reflects compression: it's read straight off the length
    /// prefix, without decompressing (or even reading) the node bytes.
    /// Used for disk-usage accounting (e.g. repack preflight), where an
    /// uncompressed estimate would overstate what a repack actually
    /// reclaims.
    ///
    /// [`serialized_len`]: crate::packfile::Record::serialized_len
    ///
    /// # Errors
    /// Returns `StorageError::Corrupt` on a truncated/invalid length
    /// prefix, `StorageError::Io` on I/O failure.
    ///
    /// # Panics
    /// Panics only on internal invariant violation (unreachable path).
    pub fn record_disk_len_at(
        shard: &Shard,
        offset: u64,
    ) -> Result<u64, crate::storage::StorageError> {
        use crate::storage::StorageError;

        for attempt in 0..2 {
            let guard = shard.mmap().map_err(StorageError::Io)?;
            let Some(mem) = guard.as_deref() else {
                return Err(StorageError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "shard could not be mapped",
                )));
            };

            let offset_usize = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

            if offset_usize
                .checked_add(4)
                .map_or(true, |end| end > mem.len())
            {
                if attempt == 0 {
                    drop(guard);
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated length prefix".into()));
            }

            let prefix_end = offset_usize.checked_add(4).expect("checked above");
            let frame_len_bytes: [u8; 4] = mem[offset_usize..prefix_end].try_into().unwrap();
            let frame_len = u32::from_le_bytes(frame_len_bytes);

            if !(packfile::FRAME_FIXED_LEN..=packfile::MAX_RECORD_LEN).contains(&frame_len) {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {frame_len}"
                )));
            }

            return Ok(4_u64.wrapping_add(u64::from(frame_len)).wrapping_add(4));
        }
        unreachable!("record_disk_len_at remap-retry is bounded to two iterations")
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
            let frame_len_bytes: [u8; 4] = mem[offset..prefix_end].try_into().unwrap();
            let frame_len = u32::from_le_bytes(frame_len_bytes);

            if !(packfile::FRAME_FIXED_LEN..=packfile::MAX_RECORD_LEN).contains(&frame_len) {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {frame_len}"
                )));
            }

            let frame_len_usize = frame_len as usize;
            let crc_pos = prefix_end
                .checked_add(frame_len_usize)
                .ok_or_else(|| StorageError::Corrupt("prefix_end + frame_len overflow".into()))?;
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

            return Self::decode_record_frame(
                frame_len_bytes,
                &mem[prefix_end..crc_pos],
                mem[crc_pos..frame_end].try_into().unwrap(),
            );
        }
        unreachable!("read_at remap-retry is bounded to two iterations")
    }

    /// Verify and decode one complete v3 frame already bounded within an mmap.
    fn decode_record_frame(
        frame_len: [u8; 4],
        payload: &[u8],
        checksum: [u8; 4],
    ) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;

        let mut crc = crc32fast::Hasher::new();
        crc.update(&frame_len);
        crc.update(payload);
        let expected = crc.finalize();
        let actual = u32::from_le_bytes(checksum);
        if expected != actual {
            return Err(StorageError::Corrupt(format!(
                "CRC mismatch: expected {expected:08x}, got {actual:08x}"
            )));
        }

        let flags = payload[0];
        if flags & !packfile::FLAG_COMPRESSED != 0 {
            return Err(StorageError::Corrupt(format!(
                "unsupported record flags: {flags:#04x}"
            )));
        }
        let uncompressed_len = u32::from_le_bytes(payload[1..5].try_into().unwrap());
        let mut room_id = [0u8; 16];
        room_id.copy_from_slice(&payload[5..21]);
        let mut hash = [0u8; 16];
        hash.copy_from_slice(&payload[21..37]);
        let data = Self::decode_node_bytes(flags, uncompressed_len, &payload[37..])?;

        Ok(Record {
            room_id,
            hash,
            data,
        })
    }

    /// Decode the node portion after a frame's CRC and structural fields passed.
    fn decode_node_bytes(
        flags: u8,
        uncompressed_len: u32,
        node_bytes: &[u8],
    ) -> Result<bytes::Bytes, crate::storage::StorageError> {
        use crate::storage::StorageError;

        let expected_len = usize::try_from(uncompressed_len).expect("u32 always fits in usize");
        if flags & packfile::FLAG_COMPRESSED == 0 {
            return (expected_len == node_bytes.len())
                .then(|| bytes::Bytes::copy_from_slice(node_bytes))
                .ok_or_else(|| {
                    StorageError::Corrupt(
                        "raw node length differs from framed uncompressed_len".into(),
                    )
                });
        }
        if uncompressed_len > packfile::MAX_DATA_LEN {
            return Err(StorageError::Corrupt(format!(
                "framed uncompressed_len too large: {uncompressed_len} > {}",
                packfile::MAX_DATA_LEN
            )));
        }
        let decompressed = zstd::bulk::decompress(node_bytes, expected_len)
            .map_err(|e| StorageError::Corrupt(format!("zstd decompress failed: {e}")))?;
        if decompressed.len() != expected_len {
            return Err(StorageError::Corrupt(format!(
                "decompressed length {} != framed uncompressed_len {uncompressed_len}",
                decompressed.len()
            )));
        }
        Ok(bytes::Bytes::from(decompressed))
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
        self.rotate_locked()
    }

    /// Rotate while `rotation_lock` is held by the caller.
    fn rotate_locked(&self) -> io::Result<()> {
        let current = *self.active_write.lock();

        let mut shards = self.shards.write();

        // Find the next available slot (skip current, prefer retired/empty)
        for offset in 1..=MAX_SHARDS_U16 {
            let candidate = current.wrapping_add(offset).wrapping_rem(MAX_SHARDS_U16);
            if shards[candidate as usize].is_none() {
                let epoch = self.next_epoch.fetch_add(1, Ordering::Relaxed);
                let path = Self::shard_path(&self.base_dir, candidate, epoch);
                let file = packfile::open_packfile(&path, true, candidate, epoch)?;
                let file_len = file.metadata()?.len();
                shards[candidate as usize] =
                    Some(Arc::new(Shard::new(candidate, epoch, file, path, file_len)));
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

    /// Assign a non-source shard as `room_id`'s write home before a repack.
    /// Repack must never copy live entries back into one of its source
    /// shards: doing so leaves the source still referenced after the
    /// generation swap, preventing its retirement and turning compaction
    /// into unbounded append growth. Rooms being rewritten from the same
    /// sources share the current destination shard.
    ///
    /// # Errors
    /// Returns an error if no free shard slot is available for the temporary
    /// rewrite destination.
    pub(crate) fn prepare_room_repack(
        &self,
        room_id: &[u8; 16],
        source_shards: &HashSet<u16>,
    ) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        if source_shards.contains(&*self.active_write.lock()) {
            self.rotate_locked()?;
        }
        let slot = *self.active_write.lock();
        self.room_home.write().insert(*room_id, slot);
        Ok(())
    }

    /// Route a whole repack batch through one shared, non-source write
    /// stream.  Individual rooms still retain their own indexes, but their
    /// replacement frames fill the same succession of destination shards
    /// instead of stranding one partially-filled shard per room.
    pub(crate) fn prepare_rooms_repack(
        &self,
        room_ids: &[[u8; 16]],
        source_shards: &HashSet<u16>,
    ) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        if source_shards.contains(&*self.active_write.lock()) {
            self.rotate_locked()?;
        }
        let slot = *self.active_write.lock();
        let mut homes = self.room_home.write();
        for room_id in room_ids {
            homes.insert(*room_id, slot);
        }
        Ok(())
    }

    /// Best-effort stats snapshot write, same contract as the `Drop` impl:
    /// the persisted stats file is pure observability, not correctness, so
    /// a failure here (e.g. the base directory momentarily gone during test
    /// teardown) is logged and swallowed -- it must never turn a durable
    /// data sync that actually succeeded into a hard error for the caller.
    ///
    /// No-op for a read-only pool: it never writes anything, so its own
    /// counters are always zero and would only ever overwrite the real
    /// writer's snapshot with nothing of value.
    fn persist_stats_best_effort(&self) {
        if !self.writable {
            return;
        }
        if let Err(e) = self.persist_stats() {
            eprintln!("mtxdb: failed to persist shard IO/sync stats: {e}");
        }
    }

    /// Best-effort, rate-limited stats flush for a writer's own periodic
    /// tick (e.g. a ~1s flush loop), so a `mtxdb shards`-style reader in
    /// another process sees reasonably fresh counters between real
    /// `sync_all`/`sync_dirty` calls — those already persist stats as a
    /// side effect, but a write-heavy, rarely-syncing process could
    /// otherwise leave a live writer's snapshot stale indefinitely.
    ///
    /// No-op on a read-only pool, and a no-op if called again before
    /// `min_interval` has passed since the last flush from here — the
    /// caller can tick this on every write without it turning into
    /// `persist_stats`' full snapshot-write-and-rename on every call.
    /// Never fsyncs anything beyond what `persist_stats` itself does for
    /// the snapshot file's own durability — this is observability data,
    /// not something worth slowing writes down to protect.
    pub fn maybe_persist_stats(&self, min_interval: Duration) {
        if !self.writable {
            return;
        }
        let now = Instant::now();
        {
            let mut last = self.last_stats_flush.write();
            if last.is_some_and(|prev| now.duration_since(prev) < min_interval) {
                return;
            }
            *last = Some(now);
        }
        self.persist_stats_best_effort();
    }

    /// Unix-seconds timestamp of the currently-known stats snapshot: the
    /// most recent of what this pool has itself persisted and whatever it
    /// restored from disk at open time. `None` if no snapshot has ever
    /// existed for this `base_dir`.
    #[must_use]
    pub fn stats_persisted_at(&self) -> Option<u64> {
        *self.stats_persisted_at.read()
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
        // Keep this lock through the fsync and removal. An append that
        // completes during the fsync blocks before marking itself dirty, so
        // it cannot be accidentally cleared by this generation of the sync.
        let mut dirty_set = self.dirty.lock();
        let dirty: Vec<u16> = dirty_set.iter().copied().collect();
        if dirty.is_empty() {
            return Ok(());
        }
        {
            let shards = self.shards.read();
            for &id in &dirty {
                if let Some(shard) = shards.get(id as usize).and_then(|s| s.as_ref()) {
                    shard.file.sync_all()?;
                    shard.sync_count.fetch_add(1, Ordering::Relaxed);
                    dirty_set.remove(&id);
                }
            }
        }
        drop(dirty_set);
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
    pub fn retire_slot(&self, slot: u16) {
        // Acquire dirty then shards to maintain lock order: dirty → shards
        // (matching sync_dirty). Hold both through the retirement so the
        // dirty bit is only cleared when the shard actually leaves the pool.
        let mut dirty = self.dirty.lock();
        let mut shards = self.shards.write();

        if *self.active_write.lock() == slot {
            return; // leave dirty unchanged — sync_dirty must fsync later
        }
        if let Some(slot_entry) = shards.get_mut(slot as usize) {
            if let Some(shard) = slot_entry.take() {
                dirty.remove(&slot);
                shard.is_current.store(false, Ordering::Release);
                self.retired_count.fetch_add(1, Ordering::Relaxed);
                drop(shards);
                self.room_home.write().retain(|_, home| *home != slot);
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

        if packfile::read_header(&mut reader)?.is_none() {
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
        let (slot, offset) = pool.put_record(&record).unwrap();
        assert_eq!(slot, 0);
        assert!(offset > 0);

        let shard = pool.get_shard(slot).unwrap();
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
    fn test_stats_persisted_at_none_until_first_flush() {
        let dir = test_dir("stats_persisted_at_none");
        let pool = ShardPool::open(dir).unwrap();
        assert_eq!(pool.stats_persisted_at(), None);
    }

    #[test]
    fn test_stats_persisted_at_set_after_sync_and_survives_reopen() {
        let dir = test_dir("stats_persisted_at_roundtrip");
        let pool = ShardPool::open(dir.clone()).unwrap();
        let record = test_record(0x01, 0xAA, b"hello");
        pool.put_record(&record).unwrap();
        pool.sync_all().unwrap();

        let persisted_at = pool
            .stats_persisted_at()
            .expect("sync_all must persist stats");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            now.saturating_sub(persisted_at) < 5,
            "persisted_at must be a recent timestamp, not zero or garbage"
        );
        drop(pool);

        // A fresh pool reading the same base_dir must restore the
        // timestamp along with the counters, not just the counters —
        // otherwise a read-only `mtxdb shards` invocation could show a
        // real snapshot's numbers next to a `None`/unknown age.
        let reopened = ShardPool::open(dir).unwrap();
        assert_eq!(reopened.stats_persisted_at(), Some(persisted_at));
    }

    #[test]
    fn test_maybe_persist_stats_rate_limited() {
        let dir = test_dir("maybe_persist_rate_limited");
        let pool = ShardPool::open(dir).unwrap();
        assert_eq!(pool.stats_persisted_at(), None);

        // First call always flushes (nothing to rate-limit against yet).
        pool.maybe_persist_stats(Duration::from_secs(3600));
        let first = pool
            .stats_persisted_at()
            .expect("first maybe_persist_stats call must flush");

        // A second call inside the interval must not re-flush. There's no
        // observable-from-here difference if it did (the timestamp is in
        // whole seconds), so this mainly documents the contract; the real
        // guard is that it doesn't pay a write+rename every call.
        pool.maybe_persist_stats(Duration::from_secs(3600));
        assert_eq!(pool.stats_persisted_at(), Some(first));

        // Zero interval must always flush.
        pool.maybe_persist_stats(Duration::ZERO);
        assert!(pool.stats_persisted_at().is_some());
    }

    #[test]
    fn test_maybe_persist_stats_noop_on_read_only_pool() {
        let dir = test_dir("maybe_persist_read_only");
        // Keep the writer open (not dropped) with no explicit sync, so no
        // shard_stats.bin exists yet — Drop's own best-effort persist
        // would otherwise write one and confound what this test checks.
        let writer = ShardPool::open(dir.clone()).unwrap();
        writer.put_record(&test_record(0x01, 0xAA, b"x")).unwrap();

        let reader = ShardPool::open_read_only(dir).unwrap();
        assert_eq!(reader.stats_persisted_at(), None);
        reader.maybe_persist_stats(Duration::ZERO);
        // A read-only pool must never write a stats snapshot of its own
        // (always-zero) counters — it must still show nothing persisted,
        // not conjure a snapshot the real writer never flushed.
        assert_eq!(reader.stats_persisted_at(), None);
        drop(writer);
    }

    #[test]
    fn test_sync_dirty_clears_only_written_shards() {
        let dir = test_dir("sync_dirty_clears");
        let pool = ShardPool::open(dir).unwrap();

        let record = test_record(0x01, 0xCC, b"needs sync");
        let (slot, _offset) = pool.put_record(&record).unwrap();
        assert!(pool.dirty.lock().contains(&slot));

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
        let (slot, _offset) = pool.put_record(&record).unwrap();
        assert_eq!(slot, 1);
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

    /// `persist_stats` used a fixed tmp filename, so two `ShardPool`s
    /// pointed at the same `base_dir` (e.g. a long-running embedder and a
    /// short-lived `mtxdb shards` CLI invocation) could race: one's
    /// `rename()` consumes the shared tmp path out from under the
    /// other's, which then fails its own `rename()` with ENOENT despite
    /// having written its tmp file successfully. Each pool now uses a
    /// tmp filename unique to its own process id and an internal counter,
    /// so concurrent persists from separate pools never collide.
    #[test]
    fn test_concurrent_persist_stats_does_not_race() {
        // A single writable pool (only one can ever exist per directory
        // now — see the writer-lock tests below), shared across threads
        // within this one process — the realistic scenario for the
        // tmp-filename-uniqueness fix, since cross-process contention on
        // the same directory is now prevented entirely by the writer lock
        // rather than needing to be tolerated here.
        let dir = test_dir("persist_stats_race");
        let pool = Arc::new(ShardPool::open(dir.clone()).unwrap());

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..25 {
                        pool.persist_stats().unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // The final file must be well-formed, not torn by a partial
        // overlapping write.
        let path = ShardPool::stats_path(&dir);
        let buf = fs::read(&path).unwrap();
        assert_eq!(&buf[0..4], STATS_MAGIC);
        assert_eq!(buf[4], STATS_VERSION);

        // No leftover tmp files from a failed/interrupted attempt.
        let leftover_tmp = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".tmp."));
        assert!(!leftover_tmp, "a persist_stats tmp file was left behind");
    }

    /// Core invariant of the writer lock: at most one writer per
    /// `base_dir`. A second `open` while the first is still alive must
    /// fail fast rather than silently risking the interleaved-append
    /// corruption this lock exists to prevent.
    #[test]
    fn test_second_writer_fails_while_first_is_open() {
        let dir = test_dir("writer_lock_exclusive");
        let _first = ShardPool::open(dir.clone()).unwrap();

        let second = ShardPool::open(dir.clone());
        assert!(
            second.is_err(),
            "a second writer must not be able to open the same base_dir concurrently"
        );
    }

    /// Dropping the writer releases its lock immediately (via
    /// `WriterLock`'s `Drop` removing the marker file), so a subsequent
    /// open — not concurrent with the first — must succeed normally.
    #[test]
    fn test_writer_lock_releases_on_drop() {
        let dir = test_dir("writer_lock_release");
        let first = ShardPool::open(dir.clone()).unwrap();
        drop(first);

        let second = ShardPool::open(dir.clone());
        assert!(
            second.is_ok(),
            "a new writer must be able to open once the previous one has dropped"
        );
    }

    /// A read-only open takes no lock at all (it never writes or
    /// truncates anything), so it must succeed even while a writer is
    /// actively holding the directory — this is the actual scenario a
    /// `mtxdb shards`-style inspection tool needs to work at all.
    #[test]
    fn test_read_only_coexists_with_active_writer() {
        let dir = test_dir("writer_lock_reader_coexist");
        let writer = ShardPool::open(dir.clone()).unwrap();

        let reader = ShardPool::open_read_only(dir.clone());
        assert!(
            reader.is_ok(),
            "a read-only open must coexist with an active writer, not be excluded by its lock"
        );

        drop(writer);
    }

    /// A read-only pool must never write a stats snapshot (its own
    /// counters are always zero), but it absolutely must *restore* the
    /// real writer's already-persisted one — otherwise a `shards`-style
    /// inspection tool built on `open_read_only` would always show
    /// `write_count`/`bytes_written`/`sync_count` as 0 regardless of how much
    /// real activity the writer has persisted, while file size (read
    /// live off disk, independent of the stats file) correctly grows —
    /// exactly the confusing "bytes climbing, everything else frozen at
    /// zero" symptom this test guards against.
    #[test]
    fn test_read_only_open_restores_persisted_stats() {
        let dir = test_dir("read_only_restores_stats");

        let writer = ShardPool::open(dir.clone()).unwrap();
        let record = test_record(0x01, 0xAA, b"payload");
        writer.put_record(&record).unwrap();
        writer.sync_dirty().unwrap();
        let persisted = writer.get_shard(0).unwrap().stats();
        assert!(
            persisted.write_count > 0,
            "test setup: writer must have written something"
        );
        drop(writer);

        let reader = ShardPool::open_read_only(dir).unwrap();
        let restored = reader.get_shard(0).unwrap().stats();
        assert_eq!(
            restored, persisted,
            "a read-only open must restore the real writer's persisted stats, not start at zero"
        );
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

    /// End-to-end: real data survives a real close-and-reopen through the
    /// actual `ShardPool::open` bootstrap, not just a raw byte-level
    /// `read_header` call — proves the v2 header roundtrips through the
    /// path a real process restart actually takes.
    #[test]
    fn test_shard_pool_reopen_survives_v2_header_roundtrip() {
        let dir = test_dir("pool_reopen_v2");
        let pool = ShardPool::open(dir.clone()).unwrap();
        let record = test_record(0x01, 0xAA, b"survives reopen");
        let (slot, offset) = pool.put_record(&record).unwrap();
        drop(pool);

        let pool = ShardPool::open(dir).unwrap();
        let shard = pool.get_shard(slot).unwrap();
        let read = ShardPool::read_at(&shard, offset).unwrap();
        assert_eq!(read.data.as_ref(), b"survives reopen");
    }

    /// End-to-end: `ShardPool::open` itself refuses a shard file whose
    /// embedded identity doesn't match its filename — not just
    /// `open_packfile` called directly in isolation.
    #[test]
    fn test_shard_pool_open_rejects_identity_mismatch() {
        let dir = test_dir("pool_open_identity_mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        // A valid v2 header claiming slot 0, epoch 0, filed under a
        // filename that claims slot 0, epoch 7.
        let path = ShardPool::shard_path(&dir, 0, 7);
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0, 0).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let Err(err) = ShardPool::open(dir) else {
            panic!("expected ShardPool::open to reject the identity mismatch");
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("epoch 7"));
    }

    /// End-to-end: `ShardPool::open` refuses a shard file with a CRC-
    /// corrupted header, not just `read_header` called directly.
    #[test]
    fn test_shard_pool_open_rejects_corrupt_header_crc() {
        let dir = test_dir("pool_open_bad_crc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = ShardPool::shard_path(&dir, 0, 0);
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0, 0).unwrap();
        buf[12] ^= 0xFF; // corrupt a byte inside the CRC-covered region
        std::fs::write(&path, &buf).unwrap();

        let Err(err) = ShardPool::open(dir) else {
            panic!("expected ShardPool::open to reject the corrupt header");
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// End-to-end: `ShardPool::open` refuses a real pre-cutover v1 store
    /// with the specific version-mismatch error, rather than treating it
    /// as an empty/fresh directory and silently starting a new v2 store
    /// alongside old data it can no longer read.
    #[test]
    fn test_shard_pool_open_refuses_v1_store() {
        let dir = test_dir("pool_open_v1_store");
        std::fs::create_dir_all(&dir).unwrap();
        let path = ShardPool::shard_path(&dir, 0, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&packfile::MAGIC);
        buf.push(0x01);
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                room_id: [0x22; 16],
                hash: [0x33; 16],
                data: bytes::Bytes::from_static(b"pre-cutover data"),
            },
        )
        .unwrap();
        std::fs::write(&path, &buf).unwrap();

        let Err(err) = ShardPool::open(dir) else {
            panic!("expected ShardPool::open to refuse the v1 store");
        };
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("0x01"));
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
        let (slot, _offset) = pool.put_record(&record).unwrap();

        let pinned = pool.get_shard(slot).unwrap();
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

    /// Crash-recovery: if two epoch files exist for the same slot, the scan
    /// keeps the higher epoch and deletes the stale one.
    #[test]
    fn test_scan_keeps_highest_epoch_per_slot() {
        let dir = test_dir("scan_epoch_dedup");
        let pool = ShardPool::open(dir.clone()).unwrap();

        // Write a record so slot 0 exists (epoch 0), then discard it —
        // this test hand-constructs both on-disk files below instead, since
        // each one's embedded header must genuinely match its own filename
        // (renaming a real gen-0 file to a gen-99 path wouldn't update the
        // header baked in at that file's actual creation, which is exactly
        // the "copied/renamed inconsistently" case open_packfile now
        // rejects — correctly, just not what this test is trying to model).
        let record = test_record(0x01, 0xAA, b"live data");
        pool.put_record(&record).unwrap();
        drop(pool);
        let old_path = ShardPool::shard_path(&dir, 0, 0);
        std::fs::remove_file(&old_path).unwrap();

        // Simulate that rotate() created a newer epoch before a crash
        // left a stale epoch-0 file behind: a genuine epoch-99 file whose header
        // matches its filename.
        let live_path = ShardPool::shard_path(&dir, 0, 99);
        let mut live_buf = Vec::new();
        packfile::write_header(&mut live_buf, 0, 99).unwrap();
        packfile::write_record(
            &mut live_buf,
            &packfile::Record {
                room_id: [0x01; 16],
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"live data"),
            },
        )
        .unwrap();
        std::fs::write(&live_path, &live_buf).unwrap();

        // Create a stale epoch-0 file (the crash leftover).
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0, 0).unwrap();
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
        assert!(live_path.exists(), "live epoch-99 file missing");
        assert!(old_path.exists(), "stale epoch-0 file missing");

        // Reopen the pool — the scan must keep epoch 99, delete epoch 0.
        let pool = ShardPool::open(dir.clone()).unwrap();
        let shard = pool.get_shard(0).unwrap();
        assert_eq!(shard.epoch, 99, "scan should have kept the higher epoch");
        assert_eq!(shard.path, live_path);
        drop(pool);

        // The stale gen-0 file must have been cleaned up by the scan.
        assert!(
            !old_path.exists(),
            "stale epoch file should be deleted during scan"
        );
    }

    #[test]
    fn test_scan_retains_valid_epoch_when_newer_header_is_torn() {
        let dir = test_dir("scan_torn_newer_epoch");
        let old_path = ShardPool::shard_path(&dir, 0, 1);
        let mut old = Vec::new();
        packfile::write_header(&mut old, 0, 1).unwrap();
        std::fs::write(&old_path, old).unwrap();

        // A higher-epoch filename alone is not enough to supersede a
        // valid shard: its header must validate first.
        let torn_path = ShardPool::shard_path(&dir, 0, 2);
        std::fs::write(&torn_path, b"MTX").unwrap();

        let pool = ShardPool::open(dir).unwrap();
        assert_eq!(pool.get_shard(0).unwrap().epoch, 1);
        assert!(old_path.exists());
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
        let (slot, offset) = pool.put_record(&record).unwrap();

        let shard = pool.get_shard(slot).unwrap();
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
        assert_eq!(all[0].0, slot);
        assert_eq!(all[0].1, shard.stats());

        assert_eq!(pool.stats(slot), Some(shard.stats()));
        assert_eq!(pool.stats(slot.wrapping_add(1)), None);
    }

    /// Backward compatibility: the startup scan must correctly parse all
    /// three filename formats that can exist on disk:
    ///   - `shard_XX.pack` — legacy (2-digit, no epoch, implied epoch 0)
    ///   - `shard_XX_YYYYYYYYYYYYYYYY.pack` — current (2-digit + epoch)
    ///   - `shard_XXXX_YYYYYYYYYYYYYYYY.pack` — new (4-digit + epoch)
    #[test]
    fn test_scan_parses_three_filename_formats() {
        let dir = test_dir("scan_three_formats");

        // Manually create one file in each format. Each file's header must
        // genuinely match the (slot_id, epoch) implied by its own
        // filename — that's exactly what open_packfile's identity check
        // verifies now, so this fixture has to be honest about it too.
        let make_pack = |room_byte: u8, shard_id: u16, epoch: u64| -> Vec<u8> {
            let mut buf = Vec::new();
            packfile::write_header(&mut buf, shard_id, epoch).unwrap();
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

        // Slot 0: legacy format "shard_00.pack" → epoch 0
        std::fs::write(dir.join("shard_00.pack"), make_pack(0x01, 0, 0)).unwrap();

        // Slot 1: 2-digit with epoch "shard_01_0000000000000005.pack" → epoch 5
        std::fs::write(
            dir.join("shard_01_0000000000000005.pack"),
            make_pack(0x02, 1, 5),
        )
        .unwrap();

        // Slot 2: 4-digit with epoch "shard_0002_0000000000000003.pack" → epoch 3
        std::fs::write(
            dir.join("shard_0002_0000000000000003.pack"),
            make_pack(0x03, 2, 3),
        )
        .unwrap();

        let pool = ShardPool::open(dir).unwrap();

        let s0 = pool.get_shard(0).unwrap();
        assert_eq!(s0.epoch, 0, "legacy shard_00.pack → epoch 0");
        assert_eq!(s0.slot, 0);

        let s1 = pool.get_shard(1).unwrap();
        assert_eq!(s1.epoch, 5, "shard_01_...0005.pack → epoch 5");
        assert_eq!(s1.slot, 1);

        let s2 = pool.get_shard(2).unwrap();
        assert_eq!(s2.epoch, 3, "shard_0002_...0003.pack → epoch 3");
        assert_eq!(s2.slot, 2);

        // Slot 3 was never created; pool should have no shard there.
        assert!(pool.get_shard(3).is_none());
    }

    /// Regression: `MAX_SHARD_BYTES` was 2^28 exactly, but `IndexSlot` stores
    /// `offset + 1` in 28 bits, so offset 2^28 - 1 would overflow. The
    /// shard cap must be ≤ (1u64 << 28) - 1 to guarantee no offset
    /// reaches the sentinel boundary.
    #[test]
    fn max_shard_bytes_fits_index_slot() {
        // The largest offset a shard can ever present must survive
        // IndexSlot::new without panicking.
        let max_offset = MAX_SHARD_BYTES - 1;
        let slot = crate::index::IndexSlot::new(0, 0, max_offset);
        assert_eq!(slot.offset(), max_offset);
    }

    /// Regression: `retire_slot` must not clear the active write shard's
    /// dirty bit. If it does, `sync_dirty()` sees no dirty shards and
    /// skips the fsync — a repack can report success without persisting
    /// its copied data.
    #[test]
    fn retire_slot_preserves_active_dirty_bit() {
        let dir = test_dir("retire_preserves_dirty");
        let pool = ShardPool::open(dir).unwrap();

        // Put a record so the active shard is dirty.
        let record = test_record(1, 1, b"payload");
        let (slot, _offset) = pool.put_record(&record).unwrap();
        assert!(
            pool.dirty.lock().contains(&slot),
            "active shard should be dirty after put"
        );

        // Attempting to retire the active shard should be a no-op.
        pool.retire_slot(slot);
        assert!(
            pool.dirty.lock().contains(&slot),
            "active shard dirty bit must survive retire_slot"
        );

        // sync_dirty must still find and fsync the shard.
        pool.sync_dirty().unwrap();
        assert!(
            pool.dirty.lock().is_empty(),
            "sync_dirty should clear dirty after fsync"
        );
    }
}
