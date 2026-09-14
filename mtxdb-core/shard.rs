use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
use parking_lot::{Mutex, RwLock};

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

/// Default in-memory threshold before a shard's buffered frames are written
/// to disk as one positioned write. Keeps bulk writes from paying a
/// `pwrite` (plus the per-record `try_clone` and file-length store) for
/// every single record — the whole remaining bulk-write gap vs the bench's
/// mdbx (which writes a memory-mapped in-memory transaction and pays
/// persistence once at commit). The default [`AppendPolicy`] is
/// [`AppendPolicy::Eager`], so this threshold only applies to pools opened
/// with [`AppendPolicy::Buffered`].
pub(crate) const PENDING_FLUSH_BYTES: usize = 1 << 20;

/// When a shard's appended frames are written to its underlying pack file.
///
/// The default is [`AppendPolicy::Eager`]: every `put_record` commits its
/// own frame with one positioned write, matching the engine's historical
/// behavior. An unsynced put is therefore immediately visible to other
/// processes reading the pack files, and survives this process exiting (a
/// crash or power loss still requires an explicit `sync_*` to be durable).
///
/// [`AppendPolicy::Buffered`] defers those writes into one positioned write
/// per shard once `max_pending_bytes` of frames have accumulated (or at the
/// rotation/sync/read/repack/checkpoint boundaries that flush first). This
/// is far faster for bulk writes — one `pwrite` per ~1 MiB instead of one
/// per record — but it changes the durability boundary:
///
/// **Records since the last flush are lost on process crash.**
///
/// Unflushed bytes live only in this process's memory, are invisible to
/// other processes, and are gone if this process exits — there is
/// deliberately no recovery path for them. (Recovery machinery here only
/// covers bytes already committed to a pack file: checkpoint writes flush
/// first, so the persisted fingerprint and index describe only committed
/// lengths, never buffered ones.) A crash therefore leaves on-disk state
/// self-consistent but drops everything still in the buffer. Buffered is a
/// pure throughput option trading crash-loss of the unflushed tail; opt
/// into it only where the caller already flushes or syncs on its own
/// durability schedule (e.g. a batch import that ends in `sync_all`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendPolicy {
    /// Commit each record's frame with its own positioned write, exactly as
    /// mtxdb always did before buffering existed.
    Eager,
    /// Accumulate frames in memory and write them out once
    /// `max_pending_bytes` have been buffered for a shard.
    Buffered {
        /// Per-shard in-memory flush threshold, in buffered frame bytes.
        max_pending_bytes: usize,
    },
}

impl AppendPolicy {
    /// [`AppendPolicy::Buffered`] at the default flush threshold (~1 MiB).
    #[must_use]
    pub fn buffered() -> Self {
        Self::Buffered {
            max_pending_bytes: PENDING_FLUSH_BYTES,
        }
    }
}

/// Scanned `(collection_id, hash, offset)` entry from a shard file.
pub type ShardEntry = ([u8; 16], [u8; 16], u64);

/// A single global shard file shared across all collections.
pub struct Shard {
    /// Pool-local allocator slot (index into the open-shard table).
    /// Ephemeral, recycled on retire. Never shown to operators.
    pub slot: u16,
    /// Pool-local `pack_id`, distinct from the slot index.
    /// Ensures a reused slot never collides on-disk with a still-referenced
    /// older file at that slot.
    pub pack_id: u64,
    /// The open file handle backing this shard.
    pub file: File,
    /// Filesystem path to this shard's file.
    pub path: PathBuf,
    /// Lazily-created mmap. Remapped when the file grows.
    // Keep mappings reference-counted: a raw, uncompressed `Bytes` returned
    // from `read_at` owns an `Arc` to the mapping it points into.  A later
    // remap can therefore replace this slot without invalidating an already
    // returned node.
    pub(crate) mmap: RwLock<Option<Arc<Mmap>>>,
    /// Serializes appends to this shard.
    pub(crate) append_lock: parking_lot::Mutex<()>,
    /// Whether this shard is still in active use. Set to `false` when
    /// retired by the pool (all records reclaimed by repack). Drop
    /// deletes the file only when retired.
    pub(crate) is_current: AtomicBool,
    /// Current file length, tracked atomically for rotation decisions
    /// without a `metadata()` syscall on every put. This is the *committed*
    /// on-disk length — buffered records occupy `[file_len, file_len +
    /// pending.len())` and are not yet on disk (see [`PENDING_FLUSH_BYTES`]).
    pub(crate) file_len: AtomicU64,
    /// Encoded frame bytes buffered since the last flush of this shard,
    /// waiting to be written as one positioned `pwrite`. Guarded by
    /// `append_lock` (writers) up to the inner `Mutex` that provides the
    /// interior mutability everything else accesses via `Arc<Shard>`.
    /// Index offsets returned by `put_record` point into the virtual
    /// `[file_len, file_len + pending.len())` region.
    pub(crate) pending: Mutex<Vec<u8>>,
    /// Number of records represented by the buffered frames (every frame
    /// carries a variable-length body, so this cannot be derived from
    /// `pending.len()`). Guarded by `append_lock` and the `pending` mutex;
    /// credited to `write_count` at flush time.
    pub(crate) pending_records: AtomicU64,
    /// Number of records appended to this shard.
    write_count: AtomicU64,
    /// Total payload bytes appended to this shard (serialized record length).
    bytes_written: AtomicU64,
    /// Number of times this shard's file has been fsynced.
    sync_count: AtomicU64,
}

/// A byte range which keeps the mmap that backs it alive.  `Bytes::from_owner`
/// turns this into a cheap, cloneable `Bytes` without copying the range.
///
/// Packfiles are append-only and repack is copy-on-write, so replacing the
/// pool's current mapping never changes bytes in an existing mapping. A
/// recovery scan may truncate a torn tail while an old range is still alive;
/// that range remains memory-safe but can only observe the old, stale bytes.
/// No surviving index entry refers to a recovered-away tail, so it cannot be
/// reached by a later lookup.
struct MmapRange {
    mmap: Arc<Mmap>,
    start: usize,
    end: usize,
}

impl AsRef<[u8]> for MmapRange {
    fn as_ref(&self) -> &[u8] {
        &self.mmap[self.start..self.end]
    }
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

/// Basic size, `pack_id`, and IO/sync info for one open shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSummary {
    /// Pool-local allocator slot (ephemeral).
    pub slot: u16,
    /// Pool-local `pack_id` for this shard incarnation.
    pub pack_id: u64,
    /// Current on-disk file length in bytes.
    pub file_bytes: u64,
    /// IO/sync counters for this shard.
    pub stats: ShardStats,
}

/// Filename for the persisted stats snapshot, stored alongside shard files.
const STATS_FILENAME: &str = "shard_stats.bin";

/// Magic bytes + version identifying the stats file format.
const STATS_MAGIC: &[u8; 4] = b"MSTA";
/// v4 replaces the v3 `slot_id`(2) + epoch(8) key with a single `pack_id`(8)
/// key, cutting the per-record overhead by 2 bytes and eliminating the
/// ephemeral slot-vs-epoch disambiguation. A v3 file is read as a
/// legacy fallback; a v2 file is simply not restored.
const STATS_VERSION: u8 = 4;

/// On-disk size of one v4 stats record: `pack_id`(8) + 3×counter(8) = 32 bytes.
const STATS_RECORD_LEN: usize = 8 + 8 * 3;

/// Header size: magic(4) + version(1) + `persisted_at`(8).
const STATS_HEADER_LEN: usize = 4 + 1 + 8;

/// Pool metadata filename — persists the `next_pack_id` high-water mark
/// so a restarted process never reuses a `pack_id` that was already
/// assigned, even if all shards from that range have been retired and
/// deleted.
const POOL_META_FILENAME: &str = "pool.meta";

/// Pool metadata format version.
const POOL_META_VERSION: u8 = 1;

/// Disambiguates concurrent `persist_stats` tmp filenames within this
/// process (paired with the process id, which disambiguates across
/// processes sharing the same `base_dir`).
static STATS_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Disambiguates pack-creation tmp filenames; paired with the process id
/// like [`STATS_TMP_COUNTER`].
static PACK_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl ShardStats {
    fn encode(self, pack_id: u64, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&pack_id.to_le_bytes());
        buf.extend_from_slice(&self.write_count.to_le_bytes());
        buf.extend_from_slice(&self.bytes_written.to_le_bytes());
        buf.extend_from_slice(&self.sync_count.to_le_bytes());
    }

    fn decode(rec: &[u8; STATS_RECORD_LEN]) -> (u64, Self) {
        let pack_id = u64::from_le_bytes(rec[0..8].try_into().unwrap());
        let write_count = u64::from_le_bytes(rec[8..16].try_into().unwrap());
        let bytes_written = u64::from_le_bytes(rec[16..24].try_into().unwrap());
        let sync_count = u64::from_le_bytes(rec[24..32].try_into().unwrap());
        (
            pack_id,
            Self {
                write_count,
                bytes_written,
                sync_count,
            },
        )
    }
}

impl Shard {
    fn new(slot: u16, pack_id: u64, file: File, path: PathBuf, file_len: u64) -> Self {
        Self {
            slot,
            pack_id,
            file,
            path,
            mmap: RwLock::new(None),
            append_lock: parking_lot::Mutex::new(()),
            is_current: AtomicBool::new(true),
            file_len: AtomicU64::new(file_len),
            pending: Mutex::new(Vec::new()),
            pending_records: AtomicU64::new(0),
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
    pub fn mmap(&self) -> io::Result<parking_lot::RwLockReadGuard<'_, Option<Arc<Mmap>>>> {
        let guard = self.mmap.read();
        if guard.is_some() {
            return Ok(guard);
        }
        drop(guard);
        let mut guard = self.mmap.write();
        if guard.is_none() {
            *guard = Some(Arc::new(packfile::map_pack(&self.file)?));
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

/// Pool of global shard files shared across all collections.
///
/// Only `MAX_SHARDS` files are open at any time, capping file descriptor
/// usage regardless of collection count. The active write shard rotates when it
/// exceeds `MAX_SHARD_BYTES`.
pub struct ShardPool {
    /// Fixed-size array of shard slots. `None` means unused.
    shards: RwLock<Vec<Option<Arc<Shard>>>>,
    /// Index of the shard currently accepting writes.
    active_write: parking_lot::Mutex<u16>,
    /// Serializes shard rotation (finding/creating the next shard).
    rotation_lock: parking_lot::Mutex<()>,
    base_dir: PathBuf,
    /// Per-pool rotation threshold, in bytes. Defaults to `MAX_SHARD_BYTES`
    /// but may be set lower (e.g. by a benchmark that wants many small
    /// packs to exercise repack/locality behavior) — never higher, since
    /// `MAX_SHARD_BYTES` is a hard ceiling imposed by `IndexSlot`'s 28-bit
    /// offset field.
    max_shard_bytes: u64,
    /// Shard IDs written to since the last sync, for scoped fsync.
    dirty: parking_lot::Mutex<HashSet<u16>>,
    /// Monotonically increasing `pack_id` counter for pack filenames.
    /// Each newly created shard file gets a unique `pack_id`, so a
    /// reused slot never collides on-disk with a still-referenced old
    /// shard at the same slot.
    next_pack_id: AtomicU64,
    /// Total number of shards retired (garbage-collected after a repack)
    /// over the pool's lifetime.
    retired_count: AtomicU64,
    /// Each collection's current "home" shard: writes for a collection are routed here
    /// instead of always following the pool-wide active-write cursor, so a
    /// collection's records stay contiguous within a shard rather than
    /// interleaving with whichever other collections happen to write around the
    /// same time. Shards remain shared — multiple collections' homes can and do
    /// coincide, especially low-traffic collections — a collection just sticks to its
    /// home until that shard fills, rather than following the global
    /// cursor wherever unrelated collections have since moved it.
    collection_home: RwLock<HashMap<[u8; 16], u16>>,
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
    /// (flush, fsync) wall-clock split of the last full `sync_all`, used to
    /// attribute sync cost between buffered frame write-out and the fsync
    /// calls themselves. `None` until the first sync.
    last_sync_split: Mutex<Option<(Duration, Duration)>>,
    /// Whether this pool holds the writer lock on `base_dir` (see `open`
    /// vs `open_read_only`). Gates `persist_stats`: a read-only pool
    /// never writes anything, including its own (always-zero) stats
    /// snapshot, so it can't race the real writer's.
    writable: bool,
    /// Whether records written through this pool are zstd-attempted (see
    /// [`crate::packfile::write_record_with_options`]). `false` for a pool
    /// whose payloads never shrink under compression (e.g. HAMT nodes),
    /// to skip paying the compressor's cost on every write for no benefit.
    compress: bool,
    /// How much of the per-frame CRC32 this pool writes and verifies (see
    /// [`packfile::ChecksumPolicy`]). Controls both what
    /// [`Self::put_record`] hashes when appending and what
    /// [`Self::read_at`] verifies on point lookups.
    checksum_policy: packfile::ChecksumPolicy,
    /// When a shard's appended frames are written to its pack file (see
    /// [`AppendPolicy`]). Set at open; defaults to
    /// [`AppendPolicy::Eager`], the historical per-put behavior. [`Self`]
    /// owns it so `put_record`'s flush decision needs no extra indirection.
    append_policy: AppendPolicy,
    /// Present only for a writable pool — `open_read_only` takes no lock
    /// at all, since it never writes or truncates anything (that risk
    /// lives one layer up, in `PackfileStorage`'s collection-index rebuild, not
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
        Self::open_internal(
            base_dir,
            true,
            MAX_SHARD_BYTES,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open or create a writable shard pool with explicit compression and
    /// checksum policies — the advanced durability/performance entry point.
    /// Pass `Some` for `max_shard_bytes` to rotate shards at a custom
    /// threshold instead of [`MAX_SHARD_BYTES`].
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidInput` if `max_shard_bytes`
    /// is zero or exceeds [`MAX_SHARD_BYTES`].
    pub fn open_with_policies(
        base_dir: PathBuf,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
        max_shard_bytes: Option<u64>,
    ) -> io::Result<Self> {
        let max_shard_bytes = match max_shard_bytes {
            Some(threshold) => Self::validate_shard_size(threshold)?,
            None => MAX_SHARD_BYTES,
        };
        Self::open_internal(base_dir, true, max_shard_bytes, compress, checksum_policy)
    }

    /// Open or create a shard pool as its exclusive writer, with `compress`
    /// controlling whether records are zstd-attempted on write (see
    /// [`crate::packfile::write_record_with_options`]) — pass `false` for a
    /// pool whose payloads (e.g. HAMT nodes/roots) never benefit, to skip
    /// paying the compressor's cost on every put. Checksum policy stays
    /// [`packfile::ChecksumPolicy::Full`].
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_compression(base_dir: PathBuf, compress: bool) -> io::Result<Self> {
        Self::open_internal(
            base_dir,
            true,
            MAX_SHARD_BYTES,
            compress,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open or create a shard pool as its exclusive writer, rotating
    /// shards at `max_shard_bytes` instead of the default
    /// [`MAX_SHARD_BYTES`]. Intended for benchmarks and tests that want
    /// many small packs without writing gigabytes of data to trigger it.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidInput` if `max_shard_bytes`
    /// is zero or exceeds [`MAX_SHARD_BYTES`].
    pub fn open_with_max_shard_bytes(base_dir: PathBuf, max_shard_bytes: u64) -> io::Result<Self> {
        Self::open_internal(
            base_dir,
            true,
            Self::validate_shard_size(max_shard_bytes)?,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Reject rotation thresholds that can't ever hold one complete record.
    fn validate_shard_size(max_shard_bytes: u64) -> io::Result<u64> {
        let min_required = (packfile::HEADER_LEN as u64)
            .saturating_add(4)
            .saturating_add(u64::from(packfile::FRAME_FIXED_LEN))
            .saturating_add(4);
        if max_shard_bytes < min_required || max_shard_bytes > MAX_SHARD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "max_shard_bytes must be in {min_required}..={MAX_SHARD_BYTES}, got {max_shard_bytes}"
                ),
            ));
        }
        Ok(max_shard_bytes)
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
        // A read-only pool never writes, so the rotation threshold and
        // compression policy are never consulted — pass the defaults for
        // consistency.
        Self::open_internal(
            base_dir,
            false,
            MAX_SHARD_BYTES,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a shard pool as a read-only observer with an explicit checksum
    /// policy, gating per-lookup CRC verification in
    /// [`Self::read_at`]. See [`packfile::ChecksumPolicy`].
    ///
    /// # Errors
    /// Same as [`Self::open_read_only`].
    pub fn open_read_only_with_policies(
        base_dir: PathBuf,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> io::Result<Self> {
        // A read-only pool never writes, so the rotation threshold and
        // compression policy are never consulted — pass the defaults.
        Self::open_internal(base_dir, false, MAX_SHARD_BYTES, true, checksum_policy)
    }

    /// Discovers packfiles in the pool directory.
    ///
    /// Only v4 filenames are accepted: `pack_XXXXXXXXXXXXXXXX.pack`
    /// where the 16 lowercase hex digits encode the `pack_id`. Uppercase
    /// hex is rejected to prevent case-insensitive collisions on
    /// case-insensitive filesystems and to enforce a single canonical
    /// spelling per ID.
    ///
    /// A `pack_*.pack` file that doesn't match the v4 format, or a legacy
    /// `shard_*.pack` file, is rejected with `Unsupported` — this is a hard
    /// cutover, not a silent skip. Other applications' `.pack` files are
    /// unrelated and ignored. Duplicate `pack_id`s are also rejected.
    fn discover_pack_files(base_dir: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
        let mut pack_files = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in fs::read_dir(base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.extension().is_some_and(|e| e == "pack") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let id_hex = match stem.strip_prefix("pack_") {
                Some(id_hex) => id_hex,
                None if stem.starts_with("shard_") => {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "found pre-v4 shard file {}; \
                             use a fresh pool or explicitly migrate/reset it",
                            path.display()
                        ),
                    ));
                }
                None => continue,
            };

            // v4 format: exactly 16 lowercase hex digits for pack_id.
            if id_hex.len() != 16 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "found pre-v4 pack file {}; \
                         use a fresh pool or explicitly migrate/reset it",
                        path.display()
                    ),
                ));
            }
            if !id_hex.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "pack filename {} contains non-hex characters",
                        path.display()
                    ),
                ));
            }
            if id_hex != id_hex.to_ascii_lowercase() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("pack filename {} must use lowercase hex", path.display()),
                ));
            }
            let pack_id = u64::from_str_radix(id_hex, 16).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{id_hex}: {e}"))
            })?;
            if !seen.insert(pack_id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "duplicate pack_id {pack_id:#018x} found in {}",
                        path.display()
                    ),
                ));
            }
            pack_files.push((pack_id, path));
        }
        Ok(pack_files)
    }

    #[allow(clippy::too_many_lines)]
    fn open_internal(
        base_dir: PathBuf,
        writable: bool,
        max_shard_bytes: u64,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> io::Result<Self> {
        if writable {
            fs::create_dir_all(&base_dir)?;
        } else if !base_dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "shard pool directory does not exist: {}",
                    base_dir.display()
                ),
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        let writer_lock = writable
            .then(|| Self::acquire_writer_lock(&base_dir))
            .transpose()?;

        let mut shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS).map(|_| None).collect();
        let mut next_slot: u16 = 0;
        let mut max_pack_id: u64 = 0;

        let mut pack_files = Self::discover_pack_files(&base_dir)?;

        // Enforce capacity: more pack files than available slots is an
        // error — we can't open them all, and silently ignoring extras
        // would lose data.
        if pack_files.len() > MAX_SHARDS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "too many pack files: {} found, but the pool can hold at most {}",
                    pack_files.len(),
                    MAX_SHARDS
                ),
            ));
        }

        // Process in pack_id order for deterministic recovery.
        pack_files.sort_unstable_by_key(|(pack_id, _)| *pack_id);

        for (pack_id, path) in pack_files {
            if writable {
                let _ = packfile::scan_and_recover_packfile(&path).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt pack {}; failed to scan and recover: {error}",
                            path.display()
                        ),
                    )
                })?;
            }

            // Validate the file before accepting it. A valid v4 file
            // must have the right magic, version, and CRC. If it
            // doesn't, the pool is corrupt — fail open rather than
            // silently deleting data.
            let file = packfile::open_packfile(&path, writable, pack_id).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("corrupt pack {}; failed to open: {error}", path.display()),
                )
            })?;

            let file_len = file.metadata()?.len();
            let slot = next_slot;
            next_slot = next_slot.saturating_add(1);
            let shard = Arc::new(Shard::new(slot, pack_id, file, path, file_len));
            shards[slot as usize] = Some(shard);
            if pack_id >= max_pack_id {
                max_pack_id = pack_id.saturating_add(1);
            }
        }

        let highest_active = next_slot.saturating_sub(1);

        // Restore next_pack_id from pool.meta if available, falling back
        // to max_pack_id computed from discovered files. The meta file
        // survives retired-and-deleted shards, so it's strictly more
        // conservative than the file scan. A corrupt pool.meta is a hard
        // error — it means pack_ids could be reused, which is data corruption.
        let mut next_pack_id = Self::restore_pool_meta(&base_dir)?
            .map_or(max_pack_id, |persisted| persisted.max(max_pack_id));

        // If no shards exist, create the initial shard — but a read-only
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
            let pack_id = next_pack_id;
            // Persist the high-water mark BEFORE creating the pack file.
            // A crash after pack creation but before next persist would
            // leave a pack_id in use with no pool.meta reservation — so
            // we write pool.meta first, guaranteeing the ID is reserved.
            Self::persist_pool_meta_at(
                &base_dir,
                pack_id.checked_add(1).expect("pack_id overflow"),
            )?;
            let (file, path) = Self::create_packfile_atomically(&base_dir, pack_id)?;
            let file_len = file.metadata()?.len();
            shards[0] = Some(Arc::new(Shard::new(0, pack_id, file, path, file_len)));
            next_pack_id = pack_id.checked_add(1).expect("pack_id overflow");
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
            max_shard_bytes,
            dirty: parking_lot::Mutex::new(HashSet::new()),
            next_pack_id: AtomicU64::new(next_pack_id),
            retired_count: AtomicU64::new(0),
            collection_home: RwLock::new(HashMap::new()),
            stats_persisted_at: RwLock::new(stats_persisted_at),
            last_stats_flush: RwLock::new(None),
            last_sync_split: Mutex::new(None),
            writable,
            compress,
            checksum_policy,
            append_policy: AppendPolicy::Eager,
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
    /// case). We store our `{pid, starttime}` in the file and, if creation
    /// fails because it already exists, check whether that exact process
    /// (not just that PID number) is still alive before concluding the
    /// lock is genuinely held — a stale file from a killed process is
    /// removed and retried once rather than wrongly blocking forever. The
    /// starttime is what makes this safe across PID reuse (see
    /// `lock_holder_is_dead`) — a bare PID is not enough on its own.
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

    /// Atomically create the lock file and write our `{pid, starttime}`
    /// into it (Linux) or just our PID (elsewhere, where starttime can't be
    /// read and `lock_holder_is_dead` never trusts a bare PID anyway). No
    /// `sync_all` here: this is advisory only, and `lock_holder_is_dead`
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
        let pid = std::process::id();
        #[cfg(target_os = "linux")]
        {
            if let Some(start_time) = Self::proc_start_time("self") {
                return write!(file, "{pid} {start_time}");
            }
        }
        write!(file, "{pid}")
    }

    /// Reads field 22 (`starttime`, clock ticks since boot) out of
    /// `/proc/<pid_or_self>/stat`. Parses from the *last* `)` rather than
    /// splitting on whitespace from the start: the second field (`comm`,
    /// the executable name in parens) can itself contain spaces and
    /// parens, which would otherwise misalign every field after it — a
    /// classic `/proc/stat` parsing bug. Field 22 is the 20th
    /// whitespace-separated token after that closing paren (field 3 is the
    /// first token after it).
    #[cfg(target_os = "linux")]
    fn proc_start_time(pid_or_self: &str) -> Option<u64> {
        let contents = fs::read_to_string(format!("/proc/{pid_or_self}/stat")).ok()?;
        let after_comm = contents.rsplit_once(')')?.1;
        after_comm.split_whitespace().nth(19)?.parse::<u64>().ok()
    }

    /// Liveness check for whoever wrote `lock_path`. Only verifies
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
    /// trade.
    ///
    /// A bare `/proc/<pid>` existence check is *not* enough on its own,
    /// either: PIDs get recycled by the OS, and a container commonly has
    /// long-lived sibling processes (postgres, redis, nginx, ...) that
    /// outlive many respawns of a crash-looping writer. Once the OS
    /// happens to reuse a dead writer's old PID number for one of those
    /// unrelated long-lived processes, a plain PID check reports "alive"
    /// forever — not because the original holder is still running, but
    /// because something else now coincidentally has that PID — poisoning
    /// recovery permanently. Comparing the recorded `starttime` (written
    /// at lock-acquisition time) against the current holder of that PID
    /// closes this: a reused PID almost certainly has a different
    /// starttime, so a mismatch means the original writer is gone even
    /// though its old PID number is (again) in use. A lock file written by
    /// an older binary (bare PID, no starttime) has nothing to compare
    /// against and fails closed exactly as before, same as any other
    /// unparsable content.
    #[cfg(not(target_arch = "wasm32"))]
    fn lock_holder_is_dead(lock_path: &Path) -> bool {
        #[cfg(target_os = "linux")]
        {
            let Ok(contents) = fs::read_to_string(lock_path) else {
                return false;
            };
            let mut fields = contents.split_whitespace();
            let Some(pid) = fields.next().and_then(|s| s.parse::<u32>().ok()) else {
                return false;
            };
            if !Path::new(&format!("/proc/{pid}")).exists() {
                return true;
            }
            // The PID exists as a live process, but that alone doesn't mean
            // it's still our original writer (see doc comment above) —
            // disambiguate via starttime when we have one to compare.
            match fields.next().and_then(|s| s.parse::<u64>().ok()) {
                Some(recorded_start) => match Self::proc_start_time(&pid.to_string()) {
                    Some(current_start) => current_start != recorded_start,
                    // Couldn't read the current holder's stat (raced with
                    // its own exit, permissions, ...) — fail closed.
                    None => false,
                },
                // Old-format lock file, or starttime collection failed at
                // creation time — nothing to disambiguate a reuse with.
                None => false,
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = lock_path;
            false
        }
    }

    /// Discovers new pack files on disk and adds them to the pool.
    ///
    /// # Errors
    /// Returns `io::Error` if reading the directory, opening a discovered
    /// pack, reading its metadata, or allocating a shard slot fails.
    pub fn discover_shards(&self) -> io::Result<()> {
        let pack_files = Self::discover_pack_files(&self.base_dir)?;
        let mut shards = self.shards.write();
        let existing: std::collections::HashSet<u64> =
            shards.iter().flatten().map(|s| s.pack_id).collect();
        let mut sorted_files = pack_files;
        sorted_files.sort_unstable_by_key(|(id, _)| *id);
        for (pack_id, path) in sorted_files {
            if existing.contains(&pack_id) {
                continue;
            }
            let slot = shards.iter().position(Option::is_none).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "shard pool full while discovering pack files",
                )
            })?;
            let file = crate::packfile::open_packfile(&path, false, pack_id)?;
            let file_len = file.metadata()?.len();
            let slot = u16::try_from(slot)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid shard slot"))?;
            shards[usize::from(slot)] = Some(std::sync::Arc::new(Shard::new(
                slot, pack_id, file, path, file_len,
            )));
        }
        Ok(())
    }

    pub(crate) fn set_collection_home(&self, collection_id: &[u8; 16], slot: u16) {
        self.collection_home.write().insert(*collection_id, slot);
    }

    /// The shard a collection's writes should go to: its remembered home if one
    /// exists and the slot is still occupied, otherwise a freshly assigned
    /// home (the pool's current active shard, the same fallback every collection
    /// used before per-collection routing existed).
    fn shard_for_collection(&self, collection_id: &[u8; 16]) -> Arc<Shard> {
        if let Some(id) = self.collection_home.read().get(collection_id).copied() {
            if let Some(shard) = self.get_shard(id) {
                return shard;
            }
        }
        let shard = self.active_shard();
        self.collection_home
            .write()
            .insert(*collection_id, shard.slot);
        shard
    }

    /// A collection's home shard just filled up: rotate the pool forward (unless
    /// another collection already did, in which case just adopt whatever's now
    /// active) and point the collection at the result.
    fn rotate_collection_full_home(
        &self,
        collection_id: &[u8; 16],
        full_slot: u16,
    ) -> io::Result<Arc<Shard>> {
        {
            let active = self.active_write.lock();
            if *active == full_slot {
                drop(active);
                self.rotate()?;
            }
        }
        let shard = self.active_shard();
        self.collection_home
            .write()
            .insert(*collection_id, shard.slot);
        Ok(shard)
    }

    /// Path to the persisted stats snapshot for a base directory.
    fn stats_path(base_dir: &Path) -> PathBuf {
        base_dir.join(STATS_FILENAME)
    }

    /// On-disk path for the pool metadata file.
    fn pool_meta_path(base_dir: &Path) -> PathBuf {
        base_dir.join(POOL_META_FILENAME)
    }

    /// Persist `next_pack_id` to `pool.meta` using an atomic
    /// tmp+rename + dir-fsync pattern. The high-water mark is written
    /// *before* the pack file it protects is created, so a crash at any
    /// point never leaves a `pack_id` that could be reused.
    ///
    /// The `next` argument is the value to persist — the first `pack_id`
    /// that has *not* yet been allocated. The caller must ensure this
    /// is written before creating any pack file that uses IDs below it.
    fn persist_pool_meta_at(base_dir: &Path, next: u64) -> io::Result<()> {
        let mut buf = Vec::with_capacity(14);
        buf.extend_from_slice(b"PMeta");
        buf.push(POOL_META_VERSION);
        buf.extend_from_slice(&next.to_le_bytes());

        let final_path = Self::pool_meta_path(base_dir);
        let tmp_path = final_path.with_extension(format!("meta.tmp.{}", std::process::id()));
        let write_result = (|| -> io::Result<()> {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&buf)?;
            tmp.sync_all()?;
            drop(tmp);
            fs::rename(&tmp_path, &final_path)?;
            // Fsync the containing directory so the rename is durable
            // across power loss — without this, a crash could leave the
            // old pool.meta (or no file) in place, allowing pack_id reuse.
            let dir = File::open(base_dir)?;
            dir.sync_all()?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        Ok(())
    }

    /// Restore `next_pack_id` from `pool.meta`. Returns an error if
    /// the file exists but is corrupt, truncated, or has an unknown
    /// version — a corrupt pool.meta means `pack_id`s could be reused,
    /// which is unrecoverable data corruption. Returns `Ok(None)` if
    /// the file does not exist (fresh pool).
    fn restore_pool_meta(base_dir: &Path) -> io::Result<Option<u64>> {
        let path = Self::pool_meta_path(base_dir);
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        if data.len() < 14 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "pool.meta is truncated ({} bytes, expected >= 14)",
                    data.len()
                ),
            ));
        }
        if &data[0..5] != b"PMeta" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pool.meta has invalid magic",
            ));
        }
        if data[5] != POOL_META_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "pool.meta has unsupported version {} (expected {})",
                    data[5], POOL_META_VERSION
                ),
            ));
        }
        let bytes: [u8; 8] = data[6..14].try_into().unwrap();
        Ok(Some(u64::from_le_bytes(bytes)))
    }

    /// Load a persisted stats snapshot, if one exists, and restore each
    /// shard's counters when its `pack_id` matches — a stale snapshot entry
    /// (from a shard since retired and deleted) is silently skipped
    /// rather than misapplied. Returns the snapshot's persisted-at
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
            let (pack_id, stats) = ShardStats::decode(rec);
            for shard in shards.iter().flatten() {
                if shard.pack_id == pack_id {
                    shard.restore_stats(stats);
                    break;
                }
            }
        }
        Some(persisted_at)
    }

    /// Persist every currently-open shard's IO/sync counters to disk,
    /// keyed by `pack_id` so a retired/reused slot's stale
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
        for (_slot, shard) in self.all_shards() {
            shard.stats().encode(shard.pack_id, &mut buf);
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

    /// On-disk path for a pack file.
    #[must_use]
    pub fn pack_path(base_dir: &Path, pack_id: u64) -> PathBuf {
        base_dir.join(format!("pack_{pack_id:016x}.pack"))
    }

    /// Initialize a pack under a non-discoverable temporary name, then rename
    /// it into place. Directory scanners consequently observe either no pack
    /// or a fully written header, never a partially initialized canonical file.
    fn create_packfile_atomically(base_dir: &Path, pack_id: u64) -> io::Result<(File, PathBuf)> {
        let path = Self::pack_path(base_dir, pack_id);
        let unique = PACK_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("tmp.{}.{unique}", std::process::id()));
        let file = match packfile::open_packfile(&tmp_path, true, pack_id) {
            Ok(file) => file,
            Err(error) => {
                let _ = fs::remove_file(&tmp_path);
                return Err(error);
            }
        };
        if let Err(error) = fs::rename(&tmp_path, &path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
        Ok((file, path))
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
    /// IO/sync stats. Needs no collection data at all — the shard-only path a
    /// `mtxdb shards`-style inspection tool should use directly (via
    /// `open_read_only`) rather than opening a full `PackfileStorage`,
    /// which rebuilds every collection's index and thus needs the writer lock.
    #[must_use]
    pub fn summaries(&self) -> Vec<ShardSummary> {
        self.all_shards()
            .into_iter()
            .map(|(slot, shard)| ShardSummary {
                slot,
                pack_id: shard.pack_id,
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

    /// The pool's checksum policy (see [`packfile::ChecksumPolicy`]) —
    /// which parts of the per-frame CRC32 this pool writes and verifies.
    #[must_use]
    pub fn checksum_policy(&self) -> packfile::ChecksumPolicy {
        self.checksum_policy
    }

    /// The pool's append policy (see [`AppendPolicy`]) — whether frames go
    /// to disk per put (the default [`AppendPolicy::Eager`]) or accumulate
    /// for one positioned write per flush ([`AppendPolicy::Buffered`]).
    #[must_use]
    pub fn append_policy(&self) -> AppendPolicy {
        self.append_policy
    }

    /// Replace the append policy. Only affects *future* `put_record` calls,
    /// so call it before the pool is shared or before the first concurrent
    /// puts — the policy is read unlocked by writers. See [`AppendPolicy`]
    /// for the durability/visibility trade each choice makes.
    ///
    /// Consuming form of [`Self::set_append_policy`], for chaining:
    /// `ShardPool::open(dir)?.with_append_policy(AppendPolicy::buffered())`.
    #[must_use]
    pub fn with_append_policy(mut self, policy: AppendPolicy) -> Self {
        self.append_policy = policy;
        self
    }

    /// In-place form of [`Self::with_append_policy`].
    pub fn set_append_policy(&mut self, policy: AppendPolicy) {
        self.append_policy = policy;
    }

    /// Append a record to its collection's home shard (see `collection_home`).
    /// Returns `(slot, offset)`. Rotates the collection to a new home shard
    /// if its current one is full.
    ///
    /// Under the default [`AppendPolicy::Eager`] the frame is written to the
    /// page cache immediately — one positioned write per put, the historical
    /// behavior (unsynced bytes survive this process exiting; durability on
    /// crash/power loss still requires a `sync_*`). Under
    /// [`AppendPolicy::Buffered`] the frame is appended to the shard's
    /// in-memory pending region instead: offsets are handed out into the
    /// virtual `[file_len, file_len + pending)` range and committed by a
    /// later flush. Reads of a virtual offset first flush (see
    /// [`Self::read_at`]), so callers observing the returned offset always
    /// see the record either way.
    ///
    /// # Errors
    /// Returns `io::Error` on write, flush, or rotation failure.
    pub fn put_record(&self, record: &Record) -> io::Result<(u16, u64)> {
        let mut shard = self.shard_for_collection(&record.collection_id);
        loop {
            // Uncompressed upper bound, used only for the pre-append
            // capacity check below — `write_record` may compress the
            // payload and write fewer bytes than this, but never more,
            // so checking against this bound never lets a shard overflow
            // MAX_SHARD_BYTES; it may just rotate a little earlier than
            // strictly necessary when compression would have made it fit.
            let max_record_len = record.serialized_len() as u64;

            let offset = {
                let guard = shard.append_lock.lock();
                // `append_lock` serializes writers and `file_len` is updated
                // at flush time, making the virtual end (committed length
                // plus already-buffered frames) the authoritative next
                // offset. No syscalls happen per append.
                let committed = shard.file_len.load(Ordering::Acquire);
                let buffered = shard.pending.lock().len() as u64;
                let virtual_end = committed.checked_add(buffered).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "file_len + pending overflow")
                })?;

                // Check capacity while holding the append lock and against
                // the virtual end — avoids a TOCTOU race where two threads
                // both pass the check then one exceeds the limit.
                let fits = virtual_end
                    .checked_add(max_record_len)
                    .is_some_and(|sum| sum <= self.max_shard_bytes);
                if !fits && committed > packfile::HEADER_LEN as u64 {
                    drop(guard);
                    // First make room: push the buffered bytes to disk so
                    // the capacity check is measured against real length.
                    self.flush_shard(&shard)?;
                    shard = self.rotate_collection_full_home(&record.collection_id, shard.slot)?;
                    continue;
                }

                let frame = packfile::encode_record_with_options(
                    record,
                    self.compress,
                    self.checksum_policy.computes_checksum(),
                )?;
                shard.pending.lock().extend_from_slice(&frame);
                shard.pending_records.fetch_add(1, Ordering::Relaxed);
                virtual_end
            };

            match self.append_policy {
                AppendPolicy::Eager => self.flush_shard(&shard)?,
                AppendPolicy::Buffered { max_pending_bytes } => {
                    if shard.pending.lock().len() >= max_pending_bytes {
                        self.flush_shard(&shard)?;
                    }
                }
            }
            return Ok((shard.slot, offset));
        }
    }

    /// Commit one shard's buffered frames to disk as a single positioned
    /// write. No-op if the shard has nothing buffered. Updates `file_len`,
    /// IO/sync counters, and the dirty set only after a successful write,
    /// so a partial failure defers rather than loses the accounting.
    ///
    /// Safe to call concurrently with writers and other flushers: the
    /// additional buffered bytes are simply rolled into this (or the next)
    /// write, keeping `file_len` monotonic and offsets stable.
    ///
    /// Note: the shard→collection count bookkeeping is deliberately NOT
    /// here — it stays per-record in `PackfileStorage::put`/`put_many`
    /// (via `record_new_shard_collection`), entirely unchanged by
    /// buffering, so no flush-time aggregation or recompute can drift from
    /// its current semantics.
    ///
    /// # Errors
    /// Returns `io::Error` on flush failure.
    pub(crate) fn flush_shard(&self, shard: &Arc<Shard>) -> io::Result<()> {
        {
            let guard = shard.append_lock.lock();
            let mut pending_guard = shard.pending.lock();
            if pending_guard.is_empty() {
                // Nothing buffered: nothing to commit or dirty.
                return Ok(());
            }
            let committed = shard.file_len.load(Ordering::Acquire);
            let bytes = std::mem::take(&mut *pending_guard);
            drop(pending_guard);
            let file = shard.file.try_clone()?;
            #[cfg(unix)]
            file.write_all_at(&bytes, committed)?;
            #[cfg(not(unix))]
            {
                let mut file = file;
                file.seek(io::SeekFrom::Start(committed))?;
                file.write_all(&bytes)?;
            }
            let new_len = bytes.len() as u64;
            let file_len = committed.checked_add(new_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "committed + buffered len overflow",
                )
            })?;
            shard.file_len.store(file_len, Ordering::Release);
            shard.write_count.fetch_add(
                shard.pending_records.swap(0, Ordering::Relaxed),
                Ordering::Relaxed,
            );
            shard.bytes_written.fetch_add(new_len, Ordering::Relaxed);
            // Hold `append_lock` for the whole commit (snapshot → write →
            // length/accounting update) so no put can observe a file_len
            // that hasn't caught up with bytes already handed out.
            drop(guard);
        }
        self.dirty.lock().insert(shard.slot);
        Ok(())
    }

    /// Commit every open shard's buffered frames to the page cache. No-op when
    /// nothing is buffered (always the case under the default
    /// [`AppendPolicy::Eager`], where each put already wrote its own frame).
    /// Under [`AppendPolicy::Buffered`] this is the durability boundary the
    /// append buffer introduces: a put's bytes live only in process memory
    /// until a flush, and only in the page cache until a sync. Callers that
    /// must observe a shard's bytes exactly match the index (sync,
    /// checkpoint, repack, delete, scan) flush by this.
    ///
    /// # Errors
    /// Returns `io::Error` if any shard's flush fails.
    pub fn flush_all(&self) -> io::Result<()> {
        for (_, shard) in self.all_shards() {
            self.flush_shard(&shard)?;
        }
        Ok(())
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
    /// Returns `StorageError::Corrupt` on a truncated or invalid frame,
    /// `StorageError::Io` on I/O failure.
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
            let Some(mapping) = guard.as_ref().cloned() else {
                return Err(StorageError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "shard could not be mapped",
                )));
            };
            drop(guard);
            let mem = mapping.as_ref();

            let offset_usize = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

            let file_len_usize = usize::try_from(shard.file_len()).unwrap_or(usize::MAX);

            if offset_usize
                .checked_add(4)
                .map_or(true, |end| end > file_len_usize || end > mem.len())
            {
                if attempt == 0 {
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

            let frame_end = prefix_end
                .checked_add(frame_len as usize)
                .and_then(|end| end.checked_add(4));
            if frame_end.map_or(true, |end| end > file_len_usize || end > mem.len()) {
                if attempt == 0 {
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated frame body or CRC".into()));
            }

            return Ok(u64::from(frame_len) + 8);
        }
        unreachable!("record_disk_len_at remap-retry is bounded to two iterations")
    }

    /// Read a record from a specific shard at the given offset.
    ///
    /// If `offset` lies in the shard's buffered-but-unflushed region
    /// (`>= file_len`), the shard is flushed first (one positioned write)
    /// so the record is always observable — a buffered record is indexed
    /// and cache-resident, but not on disk until then.
    ///
    /// `verify` controls whether the frame's CRC32 is checked: pass the
    /// pool's [`packfile::ChecksumPolicy::verifies_reads`] so `WriteOnly` and
    /// Disabled stores skip the hashing pass on point lookups. Frames written
    /// with [`packfile::FLAG_CRC_DISABLED`] are never verified regardless,
    /// and each call still validates the frame's structural lengths.
    ///
    /// # Errors
    /// Returns `StorageError::Corrupt` on CRC mismatch (when `verify` and the
    /// frame carries a checksum) or truncated frame, `StorageError::Io` on
    /// I/O failure (including flush failure for a pending offset).
    ///
    /// # Panics
    /// Panics only on internal invariant violation (unreachable path).
    pub fn read_at(
        &self,
        shard: &Arc<Shard>,
        offset: u64,
        verify: bool,
    ) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;
        if offset >= shard.file_len() {
            // The offset is virtual — still inside the append buffer. Write
            // the pending frames so the mmap can serve it.
            self.flush_shard(shard).map_err(StorageError::Io)?;
        }
        Self::read_at_committed(shard, offset, verify)
    }

    /// Read only a record's collection and content hash from its frame.
    ///
    /// This is the bounded recovery primitive for a checkpoint-backed index
    /// that needs to grow: its slots retain locations but not full hashes.
    /// It validates the frame's structural bounds but intentionally neither
    /// copies nor decompresses the payload (nor walks it to calculate CRC).
    ///
    /// # Errors
    /// Returns [`crate::storage::StorageError::Corrupt`] for malformed frames
    /// and [`crate::storage::StorageError::Io`] if a pending frame cannot be
    /// flushed or mapped.
    ///
    /// # Panics
    /// Panics only on an internal invariant violation after frame bounds have
    /// been validated.
    pub fn record_identity_at(
        &self,
        shard: &Arc<Shard>,
        offset: u64,
    ) -> Result<([u8; 16], [u8; 16]), crate::storage::StorageError> {
        use crate::storage::StorageError;
        if offset >= shard.file_len() {
            self.flush_shard(shard).map_err(StorageError::Io)?;
        }

        for attempt in 0..2 {
            let guard = shard.mmap().map_err(StorageError::Io)?;
            let Some(mapping) = guard.as_ref().cloned() else {
                return Err(StorageError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "shard could not be mapped",
                )));
            };
            drop(guard);
            let mem = mapping.as_ref();
            let offset = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;
            let file_len = usize::try_from(shard.file_len()).unwrap_or(usize::MAX);
            let Some(prefix_end) = offset.checked_add(4) else {
                return Err(StorageError::Corrupt("record offset overflow".into()));
            };
            if prefix_end > file_len || prefix_end > mem.len() {
                if attempt == 0 {
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated length prefix".into()));
            }
            let frame_len = u32::from_le_bytes(
                mem[offset..prefix_end]
                    .try_into()
                    .expect("validated length prefix range"),
            );
            if !(packfile::FRAME_FIXED_LEN..=packfile::MAX_RECORD_LEN).contains(&frame_len) {
                return Err(StorageError::Corrupt(format!(
                    "invalid record length: {frame_len}"
                )));
            }
            let Some(frame_end) = prefix_end
                .checked_add(frame_len as usize)
                .and_then(|end| end.checked_add(4))
            else {
                return Err(StorageError::Corrupt("record length overflow".into()));
            };
            let Some(metadata_end) = prefix_end.checked_add(37) else {
                return Err(StorageError::Corrupt("record metadata overflow".into()));
            };
            if frame_end > file_len || frame_end > mem.len() || metadata_end > frame_end {
                if attempt == 0 {
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated record metadata".into()));
            }
            let flags = mem[prefix_end];
            if flags & !(packfile::FLAG_COMPRESSED | packfile::FLAG_CRC_DISABLED) != 0 {
                return Err(StorageError::Corrupt(format!(
                    "unsupported record flags: {flags:#04x}"
                )));
            }
            let metadata_start = prefix_end
                .checked_add(5)
                .expect("validated record metadata prefix");
            let collection_end = metadata_start
                .checked_add(16)
                .expect("validated record collection id");
            let mut collection_id = [0; 16];
            collection_id.copy_from_slice(&mem[metadata_start..collection_end]);
            let mut hash = [0; 16];
            hash.copy_from_slice(&mem[collection_end..metadata_end]);
            return Ok((collection_id, hash));
        }
        unreachable!("record_identity_at remap-retry is bounded to two iterations")
    }

    /// Read a record whose offset is already committed to disk (e.g. from
    /// `scan_packfile`, whose offsets are by construction within the file).
    /// Unlike [`Self::read_at`] this does not flush anything: a virtual
    /// (buffered) offset would fail the bounds checks. Intended for the
    /// offline inspection paths that only ever touch committed frames.
    ///
    /// # Errors
    /// Returns `StorageError::Corrupt` on CRC mismatch (when `verify` and the
    /// frame carries a checksum) or truncated frame, `StorageError::Io` on
    /// I/O failure.
    ///
    /// # Panics
    /// Panics only on internal invariant violation (unreachable path).
    pub fn read_at_committed(
        shard: &Arc<Shard>,
        offset: u64,
        verify: bool,
    ) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;

        for attempt in 0..2 {
            let guard = shard.mmap().map_err(StorageError::Io)?;
            let Some(mapping) = guard.as_ref().cloned() else {
                return Err(StorageError::Io(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "shard could not be mapped",
                )));
            };
            drop(guard);
            let mem = mapping.as_ref();

            let offset = usize::try_from(offset)
                .map_err(|_| StorageError::Corrupt(format!("offset too large: {offset}")))?;

            let file_len_usize = usize::try_from(shard.file_len()).unwrap_or(usize::MAX);

            if offset
                .checked_add(4)
                .map_or(true, |end| end > file_len_usize || end > mem.len())
            {
                if attempt == 0 {
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

            if frame_end > file_len_usize || frame_end > mem.len() {
                if attempt == 0 {
                    Self::remap_shard(shard)?;
                    continue;
                }
                return Err(StorageError::Corrupt("truncated frame".into()));
            }

            return Self::decode_record_frame(
                frame_len_bytes,
                &mem[prefix_end..crc_pos],
                mem[crc_pos..frame_end].try_into().unwrap(),
                verify,
                MmapRange {
                    mmap: Arc::clone(&mapping),
                    // `frame_len >= FRAME_FIXED_LEN` was checked above, so
                    // the fixed 37-byte metadata prefix fits in the frame.
                    start: prefix_end
                        .checked_add(37)
                        .expect("validated frame metadata prefix fits in usize"),
                    end: crc_pos,
                },
            );
        }
        unreachable!("read_at remap-retry is bounded to two iterations")
    }

    /// Verify and decode one complete v3 frame already bounded within an mmap.
    /// When `verify` is false — or the frame carries
    /// [`packfile::FLAG_CRC_DISABLED`] — the checksum is skipped, not
    /// compared; structural validation (lengths, flags, node bytes) still runs.
    fn decode_record_frame(
        frame_len: [u8; 4],
        payload: &[u8],
        checksum: [u8; 4],
        verify: bool,
        node_bytes_owner: MmapRange,
    ) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;

        let flags = payload[0];
        if flags & !(packfile::FLAG_COMPRESSED | packfile::FLAG_CRC_DISABLED) != 0 {
            return Err(StorageError::Corrupt(format!(
                "unsupported record flags: {flags:#04x}"
            )));
        }

        if verify && flags & packfile::FLAG_CRC_DISABLED == 0 {
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
        }

        let uncompressed_len = u32::from_le_bytes(payload[1..5].try_into().unwrap());
        let mut collection_id = [0u8; 16];
        collection_id.copy_from_slice(&payload[5..21]);
        let mut hash = [0u8; 16];
        hash.copy_from_slice(&payload[21..37]);
        let data =
            Self::decode_node_bytes(flags, uncompressed_len, &payload[37..], node_bytes_owner)?;

        Ok(Record {
            collection_id,
            hash,
            data,
        })
    }

    /// Decode the node portion after a frame's CRC and structural fields passed.
    fn decode_node_bytes(
        flags: u8,
        uncompressed_len: u32,
        node_bytes: &[u8],
        node_bytes_owner: MmapRange,
    ) -> Result<bytes::Bytes, crate::storage::StorageError> {
        use crate::storage::StorageError;

        let expected_len = usize::try_from(uncompressed_len).expect("u32 always fits in usize");
        if flags & packfile::FLAG_COMPRESSED == 0 {
            return (expected_len == node_bytes.len())
                .then(|| bytes::Bytes::from_owner(node_bytes_owner))
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
            *guard = Some(Arc::new(
                packfile::map_pack(&shard.file).map_err(crate::storage::StorageError::Io)?,
            ));
        }
        // A read-only pool can observe an append made by another process. Its
        // local atomic is only a steady-state fast-path cache, so publish the
        // length verified while remapping before retrying the read.
        shard.file_len.store(file_len, Ordering::Release);
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
                let pack_id = self.next_pack_id.load(Ordering::Relaxed);
                let next = pack_id.checked_add(1).expect("pack_id overflow");

                // Persist the high-water mark BEFORE creating the pack file.
                // A crash after pack creation but before the next persist
                // would leave a pack_id in use with no pool.meta reservation.
                Self::persist_pool_meta_at(&self.base_dir, next)?;

                let (file, path) = Self::create_packfile_atomically(&self.base_dir, pack_id)?;
                let file_len = file.metadata()?.len();
                shards[candidate as usize] = Some(Arc::new(Shard::new(
                    candidate, pack_id, file, path, file_len,
                )));
                // Advance the in-memory counter only after successful creation.
                self.next_pack_id.store(next, Ordering::Relaxed);
                drop(shards);
                *self.active_write.lock() = candidate;
                return Ok(());
            }
        }

        // All slots occupied and none retired — fail rather than silently
        // overwriting a shard that may still be referenced by collection indexes.
        // The caller must repack to reclaim retired shard slots before
        // rotating again.
        Err(io::Error::other(
            "shard pool full: all slots occupied by active shards; repack to reclaim",
        ))
    }

    /// Assign a non-source shard as `collection_id`'s write home before a repack.
    /// Repack must never copy live entries back into one of its source
    /// shards: doing so leaves the source still referenced after the
    /// generation swap, preventing its retirement and turning compaction
    /// into unbounded append growth. Collections being rewritten from the same
    /// sources share the current destination shard.
    ///
    /// # Errors
    /// Returns an error if no free shard slot is available for the temporary
    /// rewrite destination.
    pub(crate) fn prepare_collection_repack(
        &self,
        collection_id: &[u8; 16],
        source_shards: &HashSet<u16>,
    ) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        if source_shards.contains(&*self.active_write.lock()) {
            self.rotate_locked()?;
        }
        let slot = *self.active_write.lock();
        self.collection_home.write().insert(*collection_id, slot);
        Ok(())
    }

    /// Route a whole repack batch through one shared, non-source write
    /// stream.  Individual collections still retain their own indexes, but their
    /// replacement frames fill the same succession of destination shards
    /// instead of stranding one partially-filled shard per collection.
    pub(crate) fn prepare_collections_repack(
        &self,
        collection_ids: &[[u8; 16]],
        source_shards: &HashSet<u16>,
    ) -> io::Result<()> {
        let _guard = self.rotation_lock.lock();
        if source_shards.contains(&*self.active_write.lock()) {
            self.rotate_locked()?;
        }
        let slot = *self.active_write.lock();
        let mut homes = self.collection_home.write();
        for collection_id in collection_ids {
            homes.insert(*collection_id, slot);
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
    /// Commits any buffered frames first, so fsync covers everything a
    /// caller believes it has put.
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_all(&self) -> io::Result<()> {
        let flush_started = Instant::now();
        self.flush_all()?;
        let flush_elapsed = flush_started.elapsed();
        let fsync_started = Instant::now();
        {
            let shards = self.shards.read();
            for shard in shards.iter().flatten() {
                shard.file.sync_all()?;
                shard.sync_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        let fsync_elapsed = fsync_started.elapsed();
        *self.last_sync_split.lock() = Some((flush_elapsed, fsync_elapsed));
        self.persist_stats_best_effort();
        Ok(())
    }

    /// (flush, fsync) wall-clock split of the last sync — `sync_all` or the
    /// dirty-scoped `sync_dirty` — if any.
    #[must_use]
    pub fn last_sync_split(&self) -> Option<(Duration, Duration)> {
        *self.last_sync_split.lock()
    }

    /// Sync only shards written to since the last sync.
    ///
    /// Each shard's dirty bit is cleared only after a successful fsync,
    /// so a partial failure leaves the untried shards marked dirty for
    /// the next call.
    ///
    /// Commits any buffered frames first, so the fsync covers everything a
    /// caller believes it has put.
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_dirty(&self) -> io::Result<()> {
        let flush_started = Instant::now();
        self.flush_all()?;
        let flush_elapsed = flush_started.elapsed();
        let fsync_started = Instant::now();
        let had_dirty;
        {
            let shards = self.shards.read();
            let mut dirty_set = self.dirty.lock();
            let dirty: Vec<u16> = dirty_set.iter().copied().collect();
            had_dirty = !dirty.is_empty();
            for &id in &dirty {
                if let Some(shard) = shards.get(id as usize).and_then(|s| s.as_ref()) {
                    shard.file.sync_all()?;
                    shard.sync_count.fetch_add(1, Ordering::Relaxed);
                    dirty_set.remove(&id);
                }
            }
        }
        let fsync_elapsed = fsync_started.elapsed();
        *self.last_sync_split.lock() = Some((flush_elapsed, fsync_elapsed));
        if had_dirty {
            self.persist_stats_best_effort();
        }
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
        // Commit any buffered frames first: records are only ever retired
        // after their bytes are on disk, so a retire can never strand data
        // that an index still references.
        if let Some(shard) = self.get_shard(slot) {
            let _ = self.flush_shard(&shard);
        }
        let mut shards = self.shards.write();
        let mut dirty = self.dirty.lock();

        if *self.active_write.lock() == slot {
            return; // leave dirty unchanged — sync_dirty must fsync later
        }
        if let Some(slot_entry) = shards.get_mut(slot as usize) {
            if let Some(shard) = slot_entry.take() {
                dirty.remove(&slot);
                shard.is_current.store(false, Ordering::Release);
                self.retired_count.fetch_add(1, Ordering::Relaxed);
                drop(shards);
                self.collection_home.write().retain(|_, home| *home != slot);
            }
        }
    }

    /// Total number of shards retired over the pool's lifetime.
    #[must_use]
    pub fn retired_count(&self) -> u64 {
        self.retired_count.load(Ordering::Relaxed)
    }

    /// Scan a shard file and return `(collection_id, hash, offset)` entries.
    /// Used during startup to rebuild per-collection indexes.
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
                    entries.push((record.collection_id, record.hash, offset));
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

    fn test_record(collection: u8, hash_byte: u8, data: &[u8]) -> Record {
        let mut collection_id = [0u8; 16];
        collection_id[0] = collection;
        let mut hash = [0u8; 16];
        hash[0] = hash_byte;
        Record {
            collection_id,
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
        let read = pool.read_at(&shard, offset, true).unwrap();
        assert_eq!(read.collection_id[0], 0x01);
        assert_eq!(read.hash[0], 0xAA);
        assert_eq!(read.data.as_ref(), b"hello shard");
    }

    #[test]
    fn raw_reads_borrow_the_mmap_and_survive_a_remap() {
        let dir = test_dir("raw_read_mmap_owner");
        let pool = ShardPool::open(dir).unwrap();
        let first = test_record(1, 1, b"first raw payload");
        let (slot, first_offset) = pool.put_record(&first).unwrap();
        let shard = pool.get_shard(slot).unwrap();
        // Buffered records are not on disk until flushed; the pointer
        // invariant below is about the mmap backing, which only exists for
        // committed bytes, so commit before establishing the mapping.
        pool.flush_all().unwrap();

        // Establish the initial mapping and retain it only for checking the
        // returned Bytes pointer. The read itself must hold its own owner.
        let initial_mapping = shard.mmap().unwrap().as_ref().unwrap().clone();
        let read_first = pool.read_at(&shard, first_offset, true).unwrap();
        let first_offset = usize::try_from(first_offset).unwrap();
        let node_offset = first_offset
            .checked_add(4)
            .and_then(|offset| offset.checked_add(37))
            .unwrap();
        assert_eq!(
            read_first.data.as_ptr(),
            initial_mapping[node_offset..].as_ptr(),
            "raw payload must be backed directly by the mmap"
        );

        // Grow the file, then read the new record. This replaces the pool's
        // current mapping; `read_first` must retain the old one safely.
        let second = test_record(1, 2, b"second raw payload");
        let (_, second_offset) = pool.put_record(&second).unwrap();
        let read_second = pool.read_at(&shard, second_offset, true).unwrap();
        assert_eq!(read_first.data.as_ref(), b"first raw payload");
        assert_eq!(read_second.data.as_ref(), b"second raw payload");
    }

    #[test]
    fn record_disk_len_rejects_truncated_frame_body() {
        let dir = test_dir("disk_len_truncated_frame");
        let pool = ShardPool::open(dir).unwrap();
        let (slot, offset) = pool
            .put_record(&test_record(0x01, 0xAA, b"truncated frame"))
            .unwrap();
        let shard = pool.get_shard(slot).unwrap();
        pool.flush_all().unwrap();

        let len = shard.file.metadata().unwrap().len();
        shard.file.set_len(len - 1).unwrap();
        *shard.mmap.write() = None;

        assert!(matches!(
            ShardPool::record_disk_len_at(&shard, offset),
            Err(crate::storage::StorageError::Corrupt(_))
        ));
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
        // `ShardPool`'s `Drop` impl does its own best-effort final stats
        // flush (`persist_stats_best_effort`) unconditionally, even
        // though nothing changed since the `sync_all` above — so
        // dropping `pool` legitimately bumps `shard_stats.bin`'s
        // timestamp to whatever `SystemTime::now()` reads *at drop
        // time*, which can differ from `persisted_at` by a second if a
        // wall-clock boundary falls between the two. Asserting exact
        // equality with the pre-drop value below would be pinning an
        // implementation-timing coincidence, not a real invariant —
        // hence the `>=`-and-recent checks instead, further down.
        drop(pool);

        // A fresh pool reading the same base_dir must restore *a*
        // recent timestamp along with the counters, not just the
        // counters — otherwise a read-only `mtxdb shards` invocation
        // could show a real snapshot's numbers next to a `None`/unknown
        // age.
        let reopened = ShardPool::open(dir).unwrap();
        let reopened_persisted_at = reopened
            .stats_persisted_at()
            .expect("stats must survive a reopen, not come back None");
        assert!(
            reopened_persisted_at >= persisted_at,
            "reopened persisted_at ({reopened_persisted_at}) must not be older than the \
             pre-drop snapshot ({persisted_at})"
        );
        let now_after_reopen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            now_after_reopen.saturating_sub(reopened_persisted_at) < 5,
            "reopened persisted_at must still be a recent timestamp"
        );
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
        // Default policy is eager: the put is committed immediately, so the
        // shard is already marked dirty and sync_dirty must clear it.
        assert!(
            pool.dirty.lock().contains(&slot),
            "eager put must mark the shard dirty"
        );

        pool.sync_dirty().unwrap();
        assert!(
            pool.dirty.lock().is_empty(),
            "dirty bit must clear after a successful sync"
        );

        // A second sync with nothing new written is a no-op, not an error.
        pool.sync_dirty().unwrap();
    }

    /// Compatibility contract for the default eager policy: an unsynced put
    /// is committed to the page cache immediately, so a fresh open of the
    /// same directory sees the record even though `flush_all`/`sync` were
    /// never called. This is the historical behavior buffering must not
    /// silently change — callers that don't opt into buffering keep it.
    #[test]
    fn eager_put_is_visible_to_fresh_open_without_flush_or_sync() {
        let dir = test_dir("eager_put_fresh_open");
        let record = test_record(0x01, 0xAA, b"eager visibility");

        let (offset, slot) = {
            let pool = ShardPool::open(dir.clone()).unwrap();
            let (slot, offset) = pool.put_record(&record).unwrap();
            assert!(
                pool.dirty.lock().contains(&slot),
                "eager put must mark the shard dirty before any sync"
            );
            (offset, slot)
        };

        // Fresh process-equivalent open, no flush/sync anywhere.
        let pool = ShardPool::open(dir).unwrap();
        let read = pool
            .read_at(&pool.get_shard(slot).unwrap(), offset, true)
            .unwrap();
        assert_eq!(read.data.as_ref(), b"eager visibility");
    }

    /// The buffered policy's known visibility boundary: an unflushed put
    /// occupies only a virtual offset past the committed file length, so a
    /// fresh open — which only sees committed bytes — must not be able to
    /// read it back. This is opt-in behavior; eager callers never hit it.
    #[test]
    fn buffered_put_is_hidden_from_fresh_open_until_flush() {
        let record = test_record(0x01, 0xBB, b"buffered invisibility");

        // No flush before drop: the byte is only in the writer's memory, so
        // a fresh open must not be able to read it back.
        let dir = test_dir("buffered_put_fresh_open");
        let (offset, slot) = {
            let pool = ShardPool::open(dir.clone())
                .unwrap()
                .with_append_policy(AppendPolicy::buffered());
            let (slot, offset) = pool.put_record(&record).unwrap();
            assert!(
                pool.dirty.lock().is_empty(),
                "a buffered put alone must not dirty the shard"
            );
            (offset, slot)
        };
        let pool = ShardPool::open(dir).unwrap();
        let shard = pool.get_shard(slot).unwrap();
        assert!(
            shard.file_len.load(Ordering::Acquire) <= offset,
            "buffered byte must not have been committed to the file"
        );
        assert!(
            pool.read_at(&shard, offset, true).is_err(),
            "a fresh open must not see an unflushed buffered byte"
        );

        // flush_all before drop puts the frame on disk; a fresh open then
        // reads the same offset back — the boundary is flush, not drop.
        let dir = test_dir("buffered_put_fresh_open_flushed");
        let (slot, offset) = {
            let pool = ShardPool::open(dir.clone())
                .unwrap()
                .with_append_policy(AppendPolicy::buffered());
            let (slot, offset) = pool.put_record(&record).unwrap();
            pool.flush_all().unwrap();
            (slot, offset)
        };
        let pool = ShardPool::open(dir).unwrap();
        let read = pool
            .read_at(&pool.get_shard(slot).unwrap(), offset, true)
            .unwrap();
        assert_eq!(read.data.as_ref(), b"buffered invisibility");
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

    /// Core fix: a collection's writes must stay on its own home shard even
    /// after *unrelated* activity rotates the pool's global active-write
    /// cursor far ahead. Before per-collection home routing, every collection simply
    /// followed that single pool-wide cursor, so any other collection's churn
    /// (with nothing to do with collection A, and no capacity reason for collection
    /// A specifically to move) would silently redirect collection A's next
    /// write too — destroying locality collection A never had a reason to lose.
    #[test]
    fn test_collection_stays_on_home_shard_despite_unrelated_pool_rotation() {
        let dir = test_dir("collection_locality");
        let pool = ShardPool::open(dir).unwrap();

        // Collection A's first write establishes its home on shard 0.
        let collection_a = test_record(0x01, 0x01, b"collection A first");
        let (shard_a1, _) = pool.put_record(&collection_a).unwrap();
        assert_eq!(shard_a1, 0);

        // Simulate unrelated churn (other collections' own rotations) dragging
        // the pool-wide cursor far ahead — collection A is not involved at all,
        // and shard 0 still has essentially all its capacity free.
        for _ in 0..5 {
            pool.rotate().unwrap();
        }

        // Collection A writes again: it must still land on shard 0, its own
        // home — not wherever unrelated rotations left the pool cursor.
        let collection_a2 = test_record(0x01, 0x03, b"collection A second");
        let (shard_a2, _) = pool.put_record(&collection_a2).unwrap();
        assert_eq!(
            shard_a2, 0,
            "collection A must stay on its own home shard, unaffected by unrelated pool rotation"
        );
    }

    /// Once a collection's own home shard actually fills up, that collection (and
    /// only that collection) rotates to a new home — independent of whatever
    /// the pool-wide cursor is doing for other collections.
    #[test]
    fn test_collection_rotates_its_own_home_when_full() {
        let dir = test_dir("collection_locality_own_rotation");
        let pool = ShardPool::open(dir).unwrap();

        let collection_a = test_record(0x01, 0x01, b"collection A first");
        let (shard_a1, _) = pool.put_record(&collection_a).unwrap();
        assert_eq!(shard_a1, 0);

        // Fill collection A's own home shard (shard 0) — not some other shard —
        // and confirm collection A itself rotates off it.
        pool.get_shard(0)
            .unwrap()
            .file_len
            .store(MAX_SHARD_BYTES - 10, Ordering::Release);
        let collection_a2 = test_record(0x01, 0x02, b"collection A triggers its own rotation");
        let (shard_a2, _) = pool.put_record(&collection_a2).unwrap();
        assert_eq!(shard_a2, 1);

        // And collection A stays on its new home from then on.
        let collection_a3 = test_record(0x01, 0x03, b"collection A third");
        let (shard_a3, _) = pool.put_record(&collection_a3).unwrap();
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

    /// A lock file recording our own actual `{pid, starttime}` (exactly
    /// what a live writer's own lock file looks like) must correctly be
    /// treated as held, not stale — otherwise a legitimately-running
    /// writer could have its own lock reclaimed out from under it.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_lock_with_matching_starttime_is_not_reclaimed() {
        let dir = test_dir("lock_matching_starttime_alive");
        std::fs::create_dir_all(&dir).unwrap();
        let lock_path = dir.join(".mtxdb.lock");
        let pid = std::process::id();
        let start = ShardPool::proc_start_time("self").expect("must read our own starttime");
        std::fs::write(&lock_path, format!("{pid} {start}")).unwrap();

        assert!(
            !ShardPool::lock_holder_is_dead(&lock_path),
            "a lock file matching our own live pid+starttime must not be reclaimable"
        );
    }

    /// The actual regression this fix exists for: a lock file whose PID
    /// has been recycled onto a *different* still-running process (a
    /// long-lived sidecar taking over a dead writer's old PID number, in
    /// the original bug's terms) must be reclaimed, not treated as held
    /// forever. We simulate "different process" the same way the real
    /// check would distinguish one: same PID, mismatched starttime.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_lock_with_stale_starttime_is_reclaimed_despite_live_pid() {
        let dir = test_dir("lock_stale_starttime_pid_reused");
        std::fs::create_dir_all(&dir).unwrap();
        let lock_path = dir.join(".mtxdb.lock");
        let pid = std::process::id();
        let real_start = ShardPool::proc_start_time("self").expect("must read our own starttime");
        // A starttime that cannot possibly be ours: any recorded starttime
        // must exactly match `/proc/<pid>/stat`'s live value, so simply
        // perturbing it stands in for "this pid now belongs to someone
        // else" without needing to actually fork/reap a process to force a
        // real-world PID recycle.
        let impostor_start = real_start.wrapping_add(1);
        std::fs::write(&lock_path, format!("{pid} {impostor_start}")).unwrap();

        assert!(
            ShardPool::lock_holder_is_dead(&lock_path),
            "a starttime mismatch on a live pid must be treated as a recycled-PID impostor, not the original holder"
        );
    }

    /// An old-format lock file (bare PID, no starttime — what a pre-fix
    /// binary writes) has nothing to disambiguate a PID reuse with, so it
    /// must keep failing closed exactly as before: a live PID is always
    /// treated as held, never reclaimed just because we can't check
    /// further. This is a no-regression guarantee for the previous format.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_old_format_lock_file_with_live_pid_still_fails_closed() {
        let dir = test_dir("lock_old_format_live_pid");
        std::fs::create_dir_all(&dir).unwrap();
        let lock_path = dir.join(".mtxdb.lock");
        std::fs::write(&lock_path, format!("{}", std::process::id())).unwrap();

        assert!(
            !ShardPool::lock_holder_is_dead(&lock_path),
            "a bare-PID (pre-fix) lock file for a still-live pid must fail closed, not be reclaimed"
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

        let r1 = test_record(0x01, 0x10, b"collection1 msg1");
        let r2 = test_record(0x02, 0x20, b"collection2 msg1");
        let r3 = test_record(0x01, 0x11, b"collection1 msg2");
        pool.put_record(&r1).unwrap();
        pool.put_record(&r2).unwrap();
        pool.put_record(&r3).unwrap();
        // The file scan only sees committed bytes, so flush before scanning.
        pool.flush_all().unwrap();

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
        // Durability is the sync boundary: buffered records are RAM-only
        // until flushed, so make this reopen test's record actually durable
        // before the pool drops.
        pool.sync_all().unwrap();
        drop(pool);

        let pool = ShardPool::open(dir).unwrap();
        let shard = pool.get_shard(slot).unwrap();
        let read = pool.read_at(&shard, offset, true).unwrap();
        assert_eq!(read.data.as_ref(), b"survives reopen");

        // Recovered shards must retain append access. Opening an existing
        // pack read-only here makes the next real import fail with EBADF.
        let appended = test_record(0x01, 0xBB, b"appends after reopen");
        let (_, appended_offset) = pool.put_record(&appended).unwrap();
        let read = pool.read_at(&shard, appended_offset, true).unwrap();
        assert_eq!(read.data.as_ref(), b"appends after reopen");
    }

    /// End-to-end: `ShardPool::open` skips a shard file whose
    /// embedded identity doesn't match its filename and creates a
    /// fresh shard 0 when no valid files remain.
    #[test]
    fn test_shard_pool_open_fails_on_identity_mismatch() {
        let dir = test_dir("pool_open_identity_mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        // A valid header claiming pack_id 0, filed under a
        // filename that claims pack_id 7.
        let path = ShardPool::pack_path(&dir, 7);
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0).unwrap();
        std::fs::write(&path, &buf).unwrap();
        // The pool fails to open — identity mismatch is corruption.
        match ShardPool::open(dir) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
            Ok(_) => panic!("expected InvalidData for identity mismatch"),
        }
    }

    /// End-to-end: `ShardPool::open` fails when a shard file has a
    /// corrupted CRC — this is pool corruption, not a skip.
    #[test]
    fn test_shard_pool_open_fails_on_corrupt_header_crc() {
        let dir = test_dir("pool_open_bad_crc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = ShardPool::pack_path(&dir, 0);
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0).unwrap();
        buf[12] ^= 0xFF; // corrupt a byte inside the CRC-covered region
        std::fs::write(&path, &buf).unwrap();

        // The pool fails to open — corrupt header is corruption.
        match ShardPool::open(dir) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
            Ok(_) => panic!("expected InvalidData for corrupt CRC"),
        }
    }

    /// End-to-end: `ShardPool::open` fails when a shard file has a
    /// v4 filename but contains v1 content — this is a pre-v4
    /// remnant, not a valid v4 pack.
    #[test]
    fn test_shard_pool_open_fails_on_v1_with_v4_filename() {
        let dir = test_dir("pool_open_v1_store");
        std::fs::create_dir_all(&dir).unwrap();
        let path = ShardPool::pack_path(&dir, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&packfile::MAGIC);
        buf.push(0x01);
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                collection_id: [0x22; 16],
                hash: [0x33; 16],
                data: bytes::Bytes::from_static(b"pre-cutover data"),
            },
        )
        .unwrap();
        std::fs::write(&path, &buf).unwrap();

        // The pool fails to open — v1 content in a v4 filename slot
        // is corruption.
        match ShardPool::open(dir) {
            Err(e) => {
                assert!(
                    e.kind() == io::ErrorKind::Unsupported
                        || e.kind() == io::ErrorKind::InvalidData,
                    "expected Unsupported or InvalidData, got: {:?}",
                    e.kind()
                );
            }
            Ok(_) => panic!("expected error for v1 content in v4 filename"),
        }
    }

    /// `discover_pack_files` rejects v3 filenames (e.g.
    /// `pack_0004_0000000000000004.pack`) with `Unsupported` rather
    /// than silently skipping them.
    #[test]
    fn test_discover_rejects_v3_filenames() {
        let dir = test_dir("discover_v3_reject");
        std::fs::create_dir_all(&dir).unwrap();
        // Create a v3-format filename — 4-digit slot + underscore + 16-digit epoch.
        let path = dir.join("shard_0004_0000000000000004.pack");
        std::fs::write(&path, b"fake").unwrap();
        match ShardPool::discover_pack_files(&dir) {
            Err(e) => {
                assert_eq!(e.kind(), io::ErrorKind::Unsupported);
                assert!(
                    e.to_string().contains("pre-v4"),
                    "error should mention pre-v4, got: {e}"
                );
            }
            Ok(_) => panic!("expected Unsupported error for v3 filename"),
        }
    }

    /// `pool.meta` persists `next_pack_id` across restarts so a
    /// reopened pool never reuses a `pack_id` that was already assigned,
    /// even if all shards from that range have been retired and deleted.
    #[test]
    fn test_pool_meta_persists_next_pack_id_across_restarts() {
        let dir = test_dir("pool_meta_persist");
        let pool = ShardPool::open(dir.clone()).unwrap();

        // pool.meta is written before the initial pack file, so it
        // exists from the very first open with next_pack_id = 1.
        assert!(
            ShardPool::pool_meta_path(&dir).exists(),
            "pool.meta must exist after initial pack creation"
        );
        let restored = ShardPool::restore_pool_meta(&dir).unwrap();
        assert_eq!(
            restored,
            Some(1),
            "pool.meta should contain next_pack_id = 1 after initial shard 0"
        );

        // Create a record so shard 0 has data, then force rotation.
        let record = test_record(0x01, 0xAA, b"first");
        pool.put_record(&record).unwrap();

        // Force a rotation — this creates pack_id 1 and persists pool.meta
        // with next_pack_id = 2.
        pool.rotate().unwrap();
        pool.put_record(&test_record(0x02, 0xBB, b"second"))
            .unwrap();

        // pool.meta should now have next_pack_id = 2.
        assert!(ShardPool::pool_meta_path(&dir).exists());
        let restored = ShardPool::restore_pool_meta(&dir).unwrap();
        assert_eq!(
            restored,
            Some(2),
            "pool.meta should contain next_pack_id = 2"
        );

        // Delete all shard files to simulate full retirement.
        drop(pool);
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "pack") {
                std::fs::remove_file(&path).unwrap();
            }
        }

        // Reopen: pool.meta survives, so next_pack_id must be >= 2
        // even though no shard files remain.
        let pool2 = ShardPool::open(dir.clone()).unwrap();
        assert!(
            pool2.next_pack_id.load(Ordering::Relaxed) >= 2,
            "pool.meta must prevent pack_id reuse after all shards are deleted"
        );

        // Creating a new shard should get pack_id 2 (not 0).
        let record = test_record(0x03, 0xCC, b"after restart");
        let (_, offset) = pool2.put_record(&record).unwrap();
        assert!(offset > 0, "record should have been written");

        // Verify pack_id 2 was assigned to the new shard.
        let summaries = pool2.summaries();
        assert!(
            summaries.iter().any(|s| s.pack_id == 2),
            "new shard should have pack_id 2, got: {:?}",
            summaries.iter().map(|s| s.pack_id).collect::<Vec<_>>()
        );
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

    /// Crash-recovery: multiple valid shard files are discovered and
    /// assigned to sequential slots in `pack_id` order.
    #[test]
    fn test_scan_discovers_valid_shards_in_pack_id_order() {
        let dir = test_dir("scan_pack_id_order");
        let pool = ShardPool::open(dir.clone()).unwrap();

        // Write a record so slot 0 exists (pack_id 0), then discard it —
        // this test hand-constructs on-disk files below instead, since
        // each one's embedded header must genuinely match its own filename.
        let record = test_record(0x01, 0xAA, b"live data");
        pool.put_record(&record).unwrap();
        drop(pool);
        let old_path = ShardPool::pack_path(&dir, 0);
        std::fs::remove_file(&old_path).unwrap();

        // Create a valid pack_id 99 file.
        let live_path = ShardPool::pack_path(&dir, 99);
        let mut live_buf = Vec::new();
        packfile::write_header(&mut live_buf, 99).unwrap();
        packfile::write_record(
            &mut live_buf,
            &packfile::Record {
                collection_id: [0x01; 16],
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"live data"),
            },
        )
        .unwrap();
        std::fs::write(&live_path, &live_buf).unwrap();

        // Create a valid pack_id 0 file (the crash leftover).
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                collection_id: [0xFF; 16],
                hash: [0xBB; 16],
                data: bytes::Bytes::from_static(b"stale leftover"),
            },
        )
        .unwrap();
        std::fs::write(&old_path, &buf).unwrap();

        // Both files exist on disk.
        assert!(live_path.exists(), "pack_id 99 file missing");
        assert!(old_path.exists(), "pack_id 0 file missing");

        // Reopen the pool — the scan must discover both, assign them
        // to slots 0 and 1 in pack_id order.
        let pool = ShardPool::open(dir.clone()).unwrap();
        let shard0 = pool.get_shard(0).unwrap();
        assert_eq!(shard0.pack_id, 0, "pack_id 0 should be assigned to slot 0");
        assert_eq!(shard0.path, old_path);

        let shard1 = pool.get_shard(1).unwrap();
        assert_eq!(
            shard1.pack_id, 99,
            "pack_id 99 should be assigned to slot 1"
        );
        assert_eq!(shard1.path, live_path);
        drop(pool);

        // Both files should still exist (no dedup in v4 — each pack_id is unique).
        assert!(old_path.exists(), "pack_id 0 file should still exist");
        assert!(live_path.exists(), "pack_id 99 file should still exist");
    }

    /// A torn header in a v4-named shard file causes pool open to fail.
    #[test]
    fn test_scan_retains_valid_shard_when_newer_header_is_torn() {
        let dir = test_dir("scan_torn_newer_pack");
        let old_path = ShardPool::pack_path(&dir, 1);
        let mut old = Vec::new();
        packfile::write_header(&mut old, 1).unwrap();
        std::fs::write(&old_path, old).unwrap();

        // A higher pack_id filename with a torn header — pool open
        // must fail because this corrupt shard is a valid v4 filename
        // that can't be read.
        let torn_path = ShardPool::pack_path(&dir, 2);
        std::fs::write(&torn_path, b"MTX").unwrap();

        match ShardPool::open(dir) {
            Err(e) => {
                assert!(
                    e.kind() == io::ErrorKind::InvalidData
                        || e.kind() == io::ErrorKind::UnexpectedEof,
                    "expected InvalidData or UnexpectedEof for torn header, got: {:?}",
                    e.kind()
                );
            }
            Ok(_) => panic!("expected error for torn shard header"),
        }
    }

    #[test]
    fn discover_shards_reports_an_unopenable_new_pack() {
        let dir = test_dir("discover_unopenable_pack");
        let pool = ShardPool::open(dir.clone()).unwrap();
        let path = ShardPool::pack_path(&dir, 1);
        std::fs::write(path, b"MTX").unwrap();

        let error = pool.discover_shards().unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
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
        // Buffered policy so the flush (triggered by the read of the still-
        // virtual offset below) is what credits the write counters.
        let pool = ShardPool::open(dir)
            .unwrap()
            .with_append_policy(AppendPolicy::buffered());

        let record = test_record(0x01, 0xAA, b"payload for stats");
        let (slot, offset) = pool.put_record(&record).unwrap();

        let shard = pool.get_shard(slot).unwrap();
        // Buffered records are counted at flush time, not append time.
        assert_eq!(shard.stats(), ShardStats::default());

        // The read of the still-buffered offset flushes the shard, which is
        // what credits the write counters.
        pool.read_at(&shard, offset, true).unwrap();
        let stats = shard.stats();
        assert_eq!(stats.write_count, 1);
        assert_eq!(stats.bytes_written, record.serialized_len() as u64);
        assert_eq!(stats.sync_count, 0);

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

    /// Backward compatibility: the startup scan must correctly parse
    /// v4 filename format and assign shards to sequential slots.
    #[test]
    fn test_scan_parses_v4_filename_format() {
        let dir = test_dir("scan_v4_format");

        // Manually create files in v4 format. Each file's header must
        // genuinely match the pack_id implied by its own filename.
        let make_pack = |collection_byte: u8, pack_id: u64| -> Vec<u8> {
            let mut buf = Vec::new();
            packfile::write_header(&mut buf, pack_id).unwrap();
            packfile::write_record(
                &mut buf,
                &packfile::Record {
                    collection_id: {
                        let mut r = [0u8; 16];
                        r[0] = collection_byte;
                        r
                    },
                    hash: [0xAA; 16],
                    data: bytes::Bytes::from_static(b"payload"),
                },
            )
            .unwrap();
            buf
        };

        // Pack ID 0: "pack_0000000000000000.pack"
        std::fs::write(dir.join("pack_0000000000000000.pack"), make_pack(0x01, 0)).unwrap();

        // Pack ID 5: "pack_0000000000000005.pack"
        std::fs::write(dir.join("pack_0000000000000005.pack"), make_pack(0x02, 5)).unwrap();

        // Pack ID 3: "pack_0000000000000003.pack"
        std::fs::write(dir.join("pack_0000000000000003.pack"), make_pack(0x03, 3)).unwrap();

        let pool = ShardPool::open(dir).unwrap();

        // Files are assigned to slots in pack_id order (0, 3, 5).
        let s0 = pool.get_shard(0).unwrap();
        assert_eq!(s0.pack_id, 0, "pack_0000000000000000.pack → pack_id 0");
        assert_eq!(s0.slot, 0);

        let s1 = pool.get_shard(1).unwrap();
        assert_eq!(s1.pack_id, 3, "pack_0000000000000003.pack → pack_id 3");
        assert_eq!(s1.slot, 1);

        let s2 = pool.get_shard(2).unwrap();
        assert_eq!(s2.pack_id, 5, "pack_0000000000000005.pack → pack_id 5");
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

        // Put a record so the active shard is dirty. Default policy is eager, so
        // the put alone commits it and marks the shard dirty.
        let record = test_record(1, 1, b"payload");
        let (slot, _offset) = pool.put_record(&record).unwrap();
        assert!(
            pool.dirty.lock().contains(&slot),
            "active shard should be dirty after writing data"
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
