use fs2::FileExt as _;
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

// Linux `sync_file_range(2)`, declared directly rather than adding a `libc`
// dependency for a single probe call.
#[cfg(target_os = "linux")]
extern "C" {
    fn sync_file_range(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i32;
}

/// Diagnostic information recovered from a lock marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockHolderInfo {
    /// PID recorded by the writer.
    pub pid: u32,
    /// Whether the recorded process identity is currently alive.
    pub running: bool,
    /// Whether the marker includes a Linux process starttime.
    pub has_starttime: bool,
}

/// Probe: start writeback for a just-flushed range before the next flush, to
/// test whether keeping the dirty frontier ahead of the writer shrinks the
/// `folio_wait_bit` stall. Enabled by `MTXDB_SYNC_FILE_RANGE=1`. A pure hint:
/// failure is ignored, since the bytes are already in the page cache.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn kick_writeback(file: &File, offset: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    const SYNC_FILE_RANGE_WRITE: u32 = 2;
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("MTXDB_SYNC_FILE_RANGE").is_ok_and(|v| v != "0")) {
        return;
    }
    // SAFETY: `file` holds a valid fd for the duration of the call, and the
    // syscall only advances writeback for the given range — it cannot affect
    // correctness.
    let _ = unsafe {
        sync_file_range(
            file.as_raw_fd(),
            i64::try_from(offset).unwrap_or(i64::MAX),
            i64::try_from(len).unwrap_or(i64::MAX),
            SYNC_FILE_RANGE_WRITE,
        )
    };
}

/// Flush a directory's entries to stable storage.
///
/// Unix can fsync a directory descriptor directly. Windows has no std API for
/// it, and only lets you open a directory handle at all with
/// `FILE_FLAG_BACKUP_SEMANTICS`; `sync_all` then maps to `FlushFileBuffers`,
/// which persists the directory's entries — the same approach SQLite's Win32
/// VFS uses. `custom_flags` is safe, so this needs no FFI or new dependency.
pub(crate) fn sync_directory(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(dir)?
            .sync_all()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = dir;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "directory sync is unsupported on this platform",
        ))
    }
}

/// Maximum number of shards in the pool.
///
/// This is a runtime policy cap, not a format limit: the shard slot is a
/// `u16` that can represent 65,536 values, and nothing about the on-disk
/// encoding forbids more than this count. The cap deliberately bounds one
/// process's live shard objects — their pack-file descriptors, their
/// (lazily created) mappings, and their buffered append memory — under a
/// no-eviction lifecycle: once live, a shard is retired only when a repack
/// reclaims it, never by closing and reopening its file.
///
/// It is also not a complete resource limit on its own. A process's usable
/// shard count is bounded by its file-descriptor and VMA budgets, which are
/// separate and often tighter (see [`ShardPool`]); a single
/// `MAX_SHARDS`-sized pool can still reach `EMFILE` or exhaust
/// `vm.max_map_count` first.
pub const MAX_SHARDS: usize = 4096;

/// Maximum number of shards as `u16`. Primary constant for shard IDs
/// and modular arithmetic.
pub(crate) const MAX_SHARDS_U16: u16 = 4096;

/// Maximum shard size before rotation: `2^32 - 2` bytes (~4 GiB), the
/// largest value for which every offset a shard can ever produce still
/// fits `IndexEntry`'s 32-bit offset field (which reserves its all-zero
/// encoding as the empty-slot sentinel, capping the max representable
/// offset at `2^32 - 2` — see `index::IndexEntry`). Do not round this up
/// to a clean `4 * 1024 * 1024 * 1024`: that's one byte over the ceiling and
/// lets a shard produce an offset `IndexEntry::new` panics on.
pub const MAX_SHARD_BYTES: u64 = crate::index::IndexEntry::MAX_OFFSET;

/// Default in-memory threshold before a shard's buffered frames are written
/// to disk as one positioned write. Keeps bulk writes from paying a
/// `pwrite` (plus the per-record `try_clone` and file-length store) for
/// every single record — the whole remaining bulk-write gap vs the bench's
/// mdbx (which writes a memory-mapped in-memory transaction and pays
/// persistence once at commit). The default [`AppendPolicy`] is
/// [`AppendPolicy::Eager`], so this threshold only applies to pools opened
/// with [`AppendPolicy::Buffered`].
///
/// The buffer is per shard, so the pool's buffered-memory commitment is this
/// threshold times the number of shards holding unflushed data. That product
/// is a worst-case upper bound, not a standing footprint: only a shard whose
/// buffer is currently near full contributes this much, and `MAX_SHARDS`
/// bounds how many can do so at once.
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
    /// Serializes fsyncs of this shard across concurrent `sync_dirty`/
    /// `sync_all` callers. A second caller waits here for an in-flight
    /// fsync, then finds the dirty bit already cleared and skips its own —
    /// so N concurrent sync callers for one shard still cost one physical
    /// fsync, without any of them holding the pool-wide `dirty` lock across
    /// the fsync. Distinct from `append_lock`, which serializes writes.
    pub(crate) sync_lock: parking_lot::Mutex<()>,
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
    /// Set when a failed flush could not be rolled back (the tail-truncating
    /// `set_len` itself failed), leaving torn, undeclared bytes past
    /// `file_len` on disk. Once set, every further append/flush on this
    /// shard is refused rather than risk writing on top of, or scanning
    /// past, that corrupt physical tail.
    poisoned: AtomicBool,
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

/// Minimum interval between implicit stats-snapshot writes from the hot
/// dirty-sync path ([`ShardPool::sync_dirty`]). The snapshot is
/// observability data rebuilt from live counters, so a dirty sync persists
/// it at most this often; `sync_all` (when it had dirty shards) and `Drop`
/// still persist it.
const STATS_FLUSH_MIN_INTERVAL: Duration = Duration::from_secs(5);

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
const POOL_META_VERSION: u8 = 2;

/// Parsed pool metadata from `pool.meta`.
#[derive(Debug, Clone, Copy)]
struct PoolMeta {
    /// First `pack_id` not yet allocated.
    next_pack_id: u64,
    /// Per-pool seed mixed into index bucket/tag derivation.
    bucket_seed: u64,
}

/// Filename for the one-time store-creation marker: records which
/// `mtxdb` version created this store. Written once, when the very
/// first shard is created, and never rewritten — unlike `pool.meta`, it
/// has no in-place-updated field, so it needs no version-preservation
/// dance across later opens.
const STORE_META_FILENAME: &str = "store.meta";

/// Store metadata format version.
const STORE_META_VERSION: u8 = 1;

/// Write the one-time store-creation marker, recording the `mtxdb`
/// version (`CARGO_PKG_VERSION`) that created this store. Best-effort: a
/// failure here doesn't fail store creation, since this is diagnostic
/// metadata, not data the engine depends on to operate correctly.
fn persist_store_meta(base_dir: &Path) {
    let version = env!("CARGO_PKG_VERSION").as_bytes();
    let Ok(version_len) = u8::try_from(version.len()) else {
        return; // never true for a real semver string; just don't write garbage
    };
    let mut buf = Vec::with_capacity(6usize.saturating_add(version.len()));
    buf.extend_from_slice(b"MTXS");
    buf.push(STORE_META_VERSION);
    buf.push(version_len);
    buf.extend_from_slice(version);

    let final_path = base_dir.join(STORE_META_FILENAME);
    let tmp_path = final_path.with_extension(format!("meta.tmp.{}", std::process::id()));
    let result = (|| -> io::Result<()> {
        let mut tmp = File::create(&tmp_path)?;
        tmp.write_all(&buf)?;
        drop(tmp);
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
}

/// Read back the `mtxdb` version that created this store, if the
/// store was created by a build new enough to record it (`store.meta`
/// predates this feature, so an older store — or one with an unreadable
/// or corrupt marker — returns `None` rather than erroring: this is
/// diagnostic-only information).
#[must_use]
pub fn store_created_by_version(base_dir: &Path) -> Option<String> {
    let data = fs::read(base_dir.join(STORE_META_FILENAME)).ok()?;
    if data.len() < 6 || &data[0..4] != b"MTXS" || data[4] != STORE_META_VERSION {
        return None;
    }
    let version_len = usize::from(data[5]);
    let version_end = 6usize.checked_add(version_len)?;
    let version_bytes = data.get(6..version_end)?;
    String::from_utf8(version_bytes.to_vec()).ok()
}

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
            sync_lock: parking_lot::Mutex::new(()),
            is_current: AtomicBool::new(true),
            file_len: AtomicU64::new(file_len),
            pending: Mutex::new(Vec::new()),
            pending_records: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            sync_count: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Whether this shard has been poisoned by an unrecoverable rollback
    /// failure and must no longer be appended to or flushed.
    #[must_use]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
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

/// Holds the writer's exclusive claim on a pool or database-root lock path.
/// On Unix, the open file descriptor owns a kernel advisory lock for this
/// value's lifetime. The marker contents are diagnostic only.
pub(crate) struct WriterLock {
    _file: File,
}

/// Pool of global shard files shared across all collections.
///
/// At most `MAX_SHARDS` shard files are live at any time, which bounds the
/// pool's own resource use regardless of collection count. That bound is a
/// policy cap over live shard *objects*, not a promise about how many the
/// process can actually hold; the real ceilings are the surrounding system
/// budgets:
///
/// - File descriptors are per-process (`RLIMIT_NOFILE`): each live shard
///   holds a persistent file descriptor, plus transient ones during rotation
///   or recovery, so a process reaches `EMFILE` once
///   `(limit - baseline - headroom) / fds_per_live_shard` shards are live.
///   Under multiple readers on one host these descriptors also aggregate
///   against the system-wide `fs.file-max` pool (`ENFILE`).
/// - Mappings consume virtual address space and VMAs against the *per-process*
///   `vm.max_map_count`. A mapping is created lazily, and because mappings are
///   reference-counted a remap can leave more than one live while previously
///   returned ranges still reference an older mapping — so an active shard
///   accounts for one or more VMAs, not exactly one. VMA counts do not
///   aggregate across processes.
///
/// Neither budget is fixed here: `fds_per_live_shard` and the per-shard VMA
/// count both vary with workload and implementation, so the formulas stay
/// symbolic rather than being reduced to constants.
///
/// The active write shard rotates when it exceeds `MAX_SHARD_BYTES`.
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
    /// `MAX_SHARD_BYTES` is a hard ceiling imposed by `IndexEntry`'s 32-bit
    /// offset field.
    max_shard_bytes: u64,
    /// Per-pool seed mixed into index bucket/tag derivation. Generated once
    /// at first pool creation, persisted in `pool.meta`, immutable after
    /// construction.
    bucket_seed: u64,
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
    /// Number of `shard_stats.bin` snapshots renamed into place over the
    /// pool's lifetime (counted only after the rename succeeds; the snapshot
    /// is best-effort observability and is not fsynced). The snapshot is
    /// observability data whose rewrites should be gated to dirty syncs;
    /// exposing the count lets callers and tests confirm a steady-state
    /// clean sync pays no stats metadata write.
    stats_snapshots: AtomicU64,
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
    /// Cumulative wall-clock time (microseconds) spent waiting to acquire
    /// `dirty` inside `sync_dirty` and `sync_all`. Pure observability —
    /// never resets, monotonically increasing. Lets operators confirm
    /// whether the coarse dirty-set lock is actually a contention point
    /// under concurrent load, before replacing it with a finer-grained
    /// mechanism.
    dirty_lock_wait: AtomicU64,
    last_open_timings: Mutex<Option<ShardOpenTimings>>,
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
    #[allow(dead_code)]
    writer_lock: Option<WriterLock>,
}

/// Wall-clock breakdown of opening a shard pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShardOpenTimings {
    /// Time spent enumerating packfiles in the pool directory.
    pub discovery: Duration,
    /// Time spent acquiring the exclusive writer lock.
    pub writer_lock: Duration,
    /// Cumulative time spent scanning and recovering writable packfiles.
    pub packfile_recovery: Duration,
    /// Number of writable packfiles passed through recovery.
    pub packfile_recovery_calls: u64,
    /// Cumulative time spent opening packfiles and validating their headers.
    pub packfile_open: Duration,
    /// Number of packfiles successfully opened.
    pub packfile_open_calls: u64,
    /// Time spent restoring pool metadata and persisted shard statistics.
    pub metadata_restore: Duration,
    /// Restoring pool.meta header and reading next pack ID.
    pub pool_meta_restore: Duration,
    /// Reading and restoring persisted snapshot counters from `shard_stats.bin`.
    pub persisted_stats_restore: Duration,
    /// Writing store.meta version marker via best-effort atomic write (fresh pool only; ZERO on existing pool).
    pub store_meta_write: Duration,
    /// Persisting pool.meta reservation and syncing file contents (fresh pool only; ZERO on existing pool).
    pub pool_meta_persist: Duration,
    /// Creating the initial packfile atomically, writing its header, syncing,
    /// renaming, and performing the final directory sync (fresh pool only;
    /// ZERO on existing pool).
    pub initial_pack_create: Duration,
    /// Unattributed time inside the `metadata_restore` span.
    pub metadata_unattributed: Duration,
    /// Time not covered by the named phases (sorting, allocation, and other
    /// small bookkeeping). This makes the phase accounting explicit instead
    /// of inviting callers to assume the named fields sum to `total`.
    pub unattributed: Duration,
    /// Total wall-clock time spent in the shard-pool open path.
    pub total: Duration,
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

    /// Discover and sort pack files in `base_dir`, enforcing the
    /// `MAX_SHARDS` capacity bound.
    fn discover_pack_files_sorted(base_dir: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
        let mut pack_files = Self::discover_pack_files(base_dir)?;
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
        pack_files.sort_unstable_by_key(|(pack_id, _)| *pack_id);
        Ok(pack_files)
    }

    /// Recover (if writable) and open every pack file, slotting them into
    /// the shard vector. Consumes `pack_files` so each `PathBuf` moves into
    /// its `Shard` rather than being cloned. Returns `(recovery_time,
    /// recovery_calls, packfile_open_time, packfile_open_calls)`.
    fn recover_and_open_packs(
        writable: bool,
        pack_files: Vec<(u64, PathBuf)>,
        shards: &mut [Option<Arc<Shard>>],
        next_slot: &mut u16,
        max_pack_id: &mut u64,
    ) -> io::Result<(Duration, u64, Duration, u64)> {
        let mut recovery_time = Duration::ZERO;
        let mut recovery_calls: u64 = 0;
        let mut packfile_open_time = Duration::ZERO;
        let mut packfile_open_calls: u64 = 0;
        for (pack_id, path) in pack_files {
            if writable {
                let recovery_started = Instant::now();
                let _ = packfile::scan_and_recover_packfile(&path).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt pack {}; failed to scan and recover: {error}",
                            path.display()
                        ),
                    )
                })?;
                recovery_time = recovery_time.saturating_add(recovery_started.elapsed());
                recovery_calls = recovery_calls.saturating_add(1);
            }

            let packfile_open_started = Instant::now();
            let file = packfile::open_packfile(&path, writable, pack_id).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("corrupt pack {}; failed to open: {error}", path.display()),
                )
            })?;
            packfile_open_time = packfile_open_time.saturating_add(packfile_open_started.elapsed());
            packfile_open_calls = packfile_open_calls.saturating_add(1);

            let file_len = file.metadata()?.len();
            let slot = *next_slot;
            *next_slot = next_slot.saturating_add(1);
            let shard = Arc::new(Shard::new(slot, pack_id, file, path, file_len));
            shards[slot as usize] = Some(shard);
            if pack_id >= *max_pack_id {
                *max_pack_id = pack_id.saturating_add(1);
            }
        }
        Ok((
            recovery_time,
            recovery_calls,
            packfile_open_time,
            packfile_open_calls,
        ))
    }

    /// Bootstrap a brand-new pool: create the first pack file, persist
    /// `pool.meta` (with a fresh seed if needed), and write `store.meta`.
    /// Returns `(next_pack_id, bucket_seed, store_meta_write_time,
    /// pool_meta_persist_time, initial_pack_create_time)`.
    ///
    /// # Errors
    /// Returns `io::Error` if the pool is empty and not writable.
    #[allow(clippy::too_many_arguments)]
    fn initialize_empty_pool(
        base_dir: &Path,
        shards: &mut [Option<Arc<Shard>>],
        mut next_pack_id: u64,
        mut bucket_seed: u64,
        writable: bool,
    ) -> io::Result<(u64, u64, Duration, Duration, Duration)> {
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

        let t_store_meta = Instant::now();
        if !base_dir.join(STORE_META_FILENAME).exists() {
            persist_store_meta(base_dir);
        }
        let store_meta_write_time = t_store_meta.elapsed();

        if bucket_seed == 0 {
            use std::collections::hash_map::RandomState;
            use std::hash::{BuildHasher, Hasher};
            bucket_seed = RandomState::new().build_hasher().finish();
        }

        let t_pool_meta_persist = Instant::now();
        Self::persist_pool_meta_at_sync_dir(
            base_dir,
            pack_id.checked_add(1).expect("pack_id overflow"),
            bucket_seed,
            false,
        )?;
        let pool_meta_persist_time = t_pool_meta_persist.elapsed();

        let t_initial_pack = Instant::now();
        let (file, path) = Self::create_packfile_atomically(base_dir, pack_id)?;
        let file_len = file.metadata()?.len();
        shards[0] = Some(Arc::new(Shard::new(0, pack_id, file, path, file_len)));
        next_pack_id = pack_id.checked_add(1).expect("pack_id overflow");

        sync_directory(base_dir)?;
        let initial_pack_create_time = t_initial_pack.elapsed();

        Ok((
            next_pack_id,
            bucket_seed,
            store_meta_write_time,
            pool_meta_persist_time,
            initial_pack_create_time,
        ))
    }

    /// Restore `next_pack_id` and `bucket_seed` from `pool.meta`, falling
    /// back to `max_pack_id` from discovered files. Returns
    /// `(next_pack_id, bucket_seed, pool_meta_restore_time)`.
    fn restore_pool_meta_state(
        base_dir: &Path,
        max_pack_id: u64,
    ) -> io::Result<(u64, u64, Duration)> {
        let started = Instant::now();
        let restored_meta = Self::restore_pool_meta(base_dir)?;
        let next_pack_id =
            restored_meta.map_or(max_pack_id, |meta| meta.next_pack_id.max(max_pack_id));
        let bucket_seed = restored_meta.map_or(0u64, |meta| meta.bucket_seed);
        Ok((next_pack_id, bucket_seed, started.elapsed()))
    }

    /// Restore persisted stats from `shard_stats.bin`. Returns
    /// `(stats_persisted_at, persisted_stats_restore_time)`.
    fn restore_stats_phase(
        base_dir: &Path,
        shards: &[Option<Arc<Shard>>],
    ) -> (Option<u64>, Duration) {
        let started = Instant::now();
        let stats_persisted_at = Self::restore_persisted_stats(base_dir, shards);
        (stats_persisted_at, started.elapsed())
    }

    /// Assemble a `ShardPool` from its discovered parts. Separated from
    /// `open_internal` to keep the latter under the clippy line limit; the
    /// raw timing values travel as one [`ShardOpenTimings`] rather than a
    /// dozen positional `Duration`s, and the two derived fields
    /// (`metadata_unattributed`, `unattributed`) are computed here.
    #[allow(clippy::too_many_arguments)]
    fn build_pool(
        shards: Vec<Option<Arc<Shard>>>,
        base_dir: PathBuf,
        max_shard_bytes: u64,
        bucket_seed: u64,
        next_pack_id: u64,
        highest_active: u16,
        writable: bool,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
        stats_persisted_at: Option<u64>,
        mut timings: ShardOpenTimings,
        writer_lock: Option<WriterLock>,
    ) -> Self {
        let metadata_subphases_sum = timings
            .pool_meta_restore
            .saturating_add(timings.store_meta_write)
            .saturating_add(timings.pool_meta_persist)
            .saturating_add(timings.initial_pack_create)
            .saturating_add(timings.persisted_stats_restore);
        timings.metadata_unattributed = timings
            .metadata_restore
            .saturating_sub(metadata_subphases_sum);

        let named_phases = timings
            .discovery
            .saturating_add(timings.writer_lock)
            .saturating_add(timings.packfile_recovery)
            .saturating_add(timings.packfile_open)
            .saturating_add(timings.metadata_restore);
        timings.unattributed = timings.total.saturating_sub(named_phases);

        Self {
            shards: RwLock::new(shards),
            active_write: parking_lot::Mutex::new(highest_active),
            rotation_lock: parking_lot::Mutex::new(()),
            base_dir,
            max_shard_bytes,
            bucket_seed,
            dirty: parking_lot::Mutex::new(HashSet::new()),
            next_pack_id: AtomicU64::new(next_pack_id),
            retired_count: AtomicU64::new(0),
            stats_snapshots: AtomicU64::new(0),
            collection_home: RwLock::new(HashMap::new()),
            stats_persisted_at: RwLock::new(stats_persisted_at),
            last_stats_flush: RwLock::new(None),
            last_sync_split: Mutex::new(None),
            dirty_lock_wait: AtomicU64::new(0),
            last_open_timings: Mutex::new(Some(timings)),
            writable,
            compress,
            checksum_policy,
            append_policy: AppendPolicy::Eager,
            writer_lock,
        }
    }

    fn open_internal(
        base_dir: PathBuf,
        writable: bool,
        max_shard_bytes: u64,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> io::Result<Self> {
        let open_started = Instant::now();
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

        let writer_lock_started = Instant::now();
        let writer_lock = writable
            .then(|| Self::acquire_writer_lock(&base_dir))
            .transpose()?;
        let writer_lock_time = writer_lock_started.elapsed();

        let mut shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS).map(|_| None).collect();
        let mut next_slot: u16 = 0;
        let mut max_pack_id: u64 = 0;

        let discovery_started = Instant::now();
        let pack_files = Self::discover_pack_files_sorted(&base_dir)?;
        let discovery_time = discovery_started.elapsed();

        let (recovery_time, recovery_calls, packfile_open_time, packfile_open_calls) =
            Self::recover_and_open_packs(
                writable,
                pack_files,
                &mut shards,
                &mut next_slot,
                &mut max_pack_id,
            )?;

        let highest_active = next_slot.saturating_sub(1);

        // Restore next_pack_id and bucket_seed from pool.meta if available,
        // falling back to max_pack_id computed from discovered files. The meta
        // file survives retired-and-deleted shards, so it's strictly more
        // conservative than the file scan. A corrupt pool.meta is a hard
        // error — it means pack_ids could be reused, which is data corruption.
        let metadata_started = Instant::now();
        let (next_pack_id, bucket_seed, pool_meta_restore_time) =
            Self::restore_pool_meta_state(&base_dir, max_pack_id)?;

        let (
            store_meta_write_time,
            pool_meta_persist_time,
            initial_pack_create_time,
            shards,
            next_pack_id,
            bucket_seed,
        ) = if shards.iter().all(std::option::Option::is_none) {
            let (npi, bs, smw, pmp, ipc) = Self::initialize_empty_pool(
                &base_dir,
                &mut shards,
                next_pack_id,
                bucket_seed,
                writable,
            )?;
            (smw, pmp, ipc, shards, npi, bs)
        } else {
            (
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                shards,
                next_pack_id,
                bucket_seed,
            )
        };

        // Restoring is a pure read of shard_stats.bin applied to our own
        // in-memory Shard objects — unconditional regardless of writable.
        let (stats_persisted_at, persisted_stats_restore_time) =
            Self::restore_stats_phase(&base_dir, &shards);
        let metadata_restore_time = metadata_started.elapsed();

        let total = open_started.elapsed();

        Ok(Self::build_pool(
            shards,
            base_dir,
            max_shard_bytes,
            bucket_seed,
            next_pack_id,
            highest_active,
            writable,
            compress,
            checksum_policy,
            stats_persisted_at,
            ShardOpenTimings {
                discovery: discovery_time,
                writer_lock: writer_lock_time,
                packfile_recovery: recovery_time,
                packfile_recovery_calls: recovery_calls,
                packfile_open: packfile_open_time,
                packfile_open_calls,
                metadata_restore: metadata_restore_time,
                pool_meta_restore: pool_meta_restore_time,
                persisted_stats_restore: persisted_stats_restore_time,
                store_meta_write: store_meta_write_time,
                pool_meta_persist: pool_meta_persist_time,
                initial_pack_create: initial_pack_create_time,
                total,
                ..Default::default()
            },
            writer_lock,
        ))
    }

    /// Return the timing breakdown from the most recent pool open.
    #[must_use]
    pub fn open_timings(&self) -> Option<ShardOpenTimings> {
        *self.last_open_timings.lock()
    }

    /// Return the per-pool seed mixed into index bucket/tag derivation.
    #[must_use]
    pub(crate) fn bucket_seed(&self) -> u64 {
        self.bucket_seed
    }

    /// Whether this pool attempts zstd compression on written records.
    #[must_use]
    pub fn is_compression_enabled(&self) -> bool {
        self.compress
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
    fn acquire_writer_lock(base_dir: &Path) -> io::Result<WriterLock> {
        Self::acquire_lock_path(&base_dir.join(".mtxdb.lock"))
    }

    /// Acquire the same `{pid, starttime}`-guarded exclusive lock at an
    /// arbitrary path. Shared by the per-pool writer lock (`.mtxdb.lock`) and
    /// the database-root shared-WAL lock (`.mtxdb.wal.lock`); see
    /// `acquire_writer_lock`'s doc for the staleness contract.
    pub(crate) fn acquire_lock_path(lock_path: &Path) -> io::Result<WriterLock> {
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;

        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == io::ErrorKind::WouldBlock {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!(
                        "{} is already locked by another writer",
                        lock_path.display()
                    ),
                )
            } else {
                error
            }
        })?;

        file.set_len(0)?;
        #[cfg(target_os = "linux")]
        let marker = Self::proc_start_time("self").map_or_else(
            || format!("{}\n", std::process::id()),
            |start| format!("{} {start}\n", std::process::id()),
        );
        #[cfg(not(target_os = "linux"))]
        let marker = format!("{}\n", std::process::id());
        file.write_all(marker.as_bytes())?;
        file.sync_all()?;

        Ok(WriterLock { _file: file })
    }

    /// Atomically create the lock file and write our `{pid, starttime}`
    /// into it (Linux) or just our PID (elsewhere, where starttime can't be
    /// read and `lock_holder_is_dead` never trusts a bare PID anyway). No
    /// `sync_all` here: this is advisory only, and `lock_holder_is_dead`
    /// already fails closed (treats an unparsable file as "might be
    /// alive") on a torn write from a crash mid-write — there's no
    /// correctness reason to pay an fsync on every lock acquisition to
    /// protect against that.
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
    #[must_use]
    pub fn lock_holder_is_dead(lock_path: &Path) -> bool {
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

    /// Read the PID recorded in a lock marker and classify its holder.
    ///
    /// Returns `None` for a missing or unparsable marker. The liveness result
    /// is advisory and never acquires, removes, or modifies the lock.
    #[must_use]
    pub fn lock_holder_status(lock_path: &Path) -> Option<(u32, bool)> {
        let info = Self::lock_holder_info(lock_path)?;
        Some((info.pid, info.running))
    }

    /// Read the lock marker with confidence information for diagnostics.
    #[must_use]
    pub fn lock_holder_info(lock_path: &Path) -> Option<LockHolderInfo> {
        let contents = fs::read_to_string(lock_path).ok()?;
        let pid = contents.split_whitespace().next()?.parse().ok()?;
        Some(LockHolderInfo {
            pid,
            running: !Self::lock_holder_is_dead(lock_path),
            has_starttime: contents
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u64>().ok())
                .is_some(),
        })
    }

    /// Probe whether an exclusive writer currently holds `lock_path`.
    ///
    /// This opens the marker read-only, takes a shared lock only for the
    /// duration of the probe, and never changes the file. A stale marker with
    /// no active advisory lock therefore reports `Some(false)`.
    ///
    /// This is a best-effort instantaneous probe: a writer can acquire or
    /// release the exclusive lock immediately before or after this call. The
    /// result is diagnostic evidence, not a synchronization guarantee.
    /// Because the writer uses a non-retrying exclusive try-lock, a probe that
    /// races its acquisition can also make that writer report a transient
    /// "already locked" failure. Callers should use this only for inspection,
    /// never as a coordination protocol.
    #[must_use]
    pub fn lock_contended(lock_path: &Path) -> Option<bool> {
        let Ok(file) = File::options().read(true).open(lock_path) else {
            return None;
        };
        match fs2::FileExt::try_lock_shared(&file) {
            Ok(()) => {
                let _ = fs2::FileExt::unlock(&file);
                Some(false)
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Some(true),
            Err(_) => None,
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

    /// Persist `next_pack_id` and `bucket_seed` to `pool.meta` using an atomic
    /// tmp+rename + dir-fsync pattern. The high-water mark is written
    /// *before* the pack file it protects is created, so a crash at any
    /// point never leaves a `pack_id` that could be reused.
    fn persist_pool_meta_at(base_dir: &Path, next: u64, bucket_seed: u64) -> io::Result<()> {
        Self::persist_pool_meta_at_sync_dir(base_dir, next, bucket_seed, true)
    }

    /// Internal form of [`Self::persist_pool_meta_at`] allowing callers that
    /// create multiple files (such as fresh pool open) to defer directory
    /// synchronization until all renames have completed.
    fn persist_pool_meta_at_sync_dir(
        base_dir: &Path,
        next: u64,
        bucket_seed: u64,
        sync_dir: bool,
    ) -> io::Result<()> {
        let mut buf = Vec::with_capacity(21);
        buf.extend_from_slice(b"MTXP");
        buf.push(POOL_META_VERSION);
        buf.extend_from_slice(&next.to_le_bytes());
        buf.extend_from_slice(&bucket_seed.to_le_bytes());

        let final_path = Self::pool_meta_path(base_dir);
        let tmp_path = final_path.with_extension(format!("meta.tmp.{}", std::process::id()));
        let write_result = (|| -> io::Result<()> {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&buf)?;
            tmp.sync_all()?;
            drop(tmp);
            fs::rename(&tmp_path, &final_path)?;
            if sync_dir {
                // Fsync the containing directory so the rename is durable
                // across power loss — without this, a crash could leave the
                // old pool.meta (or no file) in place, allowing pack_id reuse.
                sync_directory(base_dir)?;
            }
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        Ok(())
    }

    /// Restore pool metadata from `pool.meta`. Returns an error if
    /// the file exists but is corrupt, truncated, or has an unknown
    /// version — a corrupt pool.meta means `pack_id`s could be reused,
    /// which is unrecoverable data corruption. Returns `Ok(None)` if
    /// the file does not exist (fresh pool).
    fn restore_pool_meta(base_dir: &Path) -> io::Result<Option<PoolMeta>> {
        let path = Self::pool_meta_path(base_dir);
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        if data.len() < 21 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "pool.meta is truncated ({} bytes, expected >= 21)",
                    data.len()
                ),
            ));
        }
        if &data[0..4] != b"MTXP" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "pool.meta has invalid magic",
            ));
        }
        if data[4] != POOL_META_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "pool.meta has unsupported version {} (expected {})",
                    data[4], POOL_META_VERSION
                ),
            ));
        }
        let next_pack_id = u64::from_le_bytes(data[5..13].try_into().unwrap());
        let bucket_seed = u64::from_le_bytes(data[13..21].try_into().unwrap());
        Ok(Some(PoolMeta {
            next_pack_id,
            bucket_seed,
        }))
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
        // Deliberately no fsync here. The snapshot is observability data
        // (rebuilt from live counters; open tolerates a missing/old/short
        // file), and the rename below is never followed by a directory
        // fsync -- so the old `tmp.sync_all()` paid an fsync per persist for
        // a power-loss guarantee the rename didn't actually provide.
        let write_result = (|| -> io::Result<()> {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&buf)?;
            Ok(())
        })();
        if let Err(e) = write_result {
            // Don't leave a half-written tmp file behind under its
            // now-unique name — best-effort, the write already failed.
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        fs::rename(&tmp_path, &final_path)?;
        self.stats_snapshots.fetch_add(1, Ordering::Relaxed);
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

    /// Number of currently-open shard slots (O(1), no allocation).
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.read().iter().filter(|s| s.is_some()).count()
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
        self.put_record_with_len(record)
            .map(|(slot, offset, _)| (slot, offset))
    }

    /// Append a record and return its exact encoded frame size as well as its
    /// location. The extra size is used by incremental physical accounting.
    ///
    /// # Errors
    /// Returns an I/O or encoding error if the record cannot be appended.
    pub fn put_record_with_len(&self, record: &Record) -> io::Result<(u16, u64, u64)> {
        let mut shard = self.shard_for_collection(&record.collection_id);
        loop {
            if shard.is_poisoned() {
                return Err(io::Error::other(format!(
                    "shard {} is poisoned after an unrecoverable rollback failure",
                    shard.slot
                )));
            }
            // Uncompressed upper bound, used only for the pre-append
            // capacity check below — `write_record` may compress the
            // payload and write fewer bytes than this, but never more,
            // so checking against this bound never lets a shard overflow
            // MAX_SHARD_BYTES; it may just rotate a little earlier than
            // strictly necessary when compression would have made it fit.
            let max_record_len = record.serialized_len() as u64;

            // `append_lock` serializes writers *and* flushers, and is held
            // from the capacity check through the append, the automatic
            // flush, and (on failure) the rollback. That is what makes the
            // rollback safe: a shard is shared by many collections (its
            // `collection_home` maps several collections onto the same
            // slot), so per-collection put mutexes DON'T serialize the
            // appenders — only this lock does. While it is held, no other
            // put, of any collection, can append to `pending`, so our frame
            // is provably the tail when the failed flush needs to retract
            // it.
            let append_guard = shard.append_lock.lock();
            // Re-check poison after acquiring the lock: the check above is
            // only a fast-path skip for the common case. A concurrent
            // put/flush can poison the shard between that check and here —
            // `append_lock` is what actually excludes further appends, so
            // it's the only place this check is load-bearing.
            if shard.is_poisoned() {
                return Err(io::Error::other(format!(
                    "shard {} is poisoned after an unrecoverable rollback failure",
                    shard.slot
                )));
            }
            // `file_len` is updated at flush time, making the virtual end
            // (committed length plus already-buffered frames) the
            // authoritative next offset. No syscalls happen per append.
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
                drop(append_guard);
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
            let frame_len = frame.len();
            let pending_start = {
                let mut pending = shard.pending.lock();
                let start = pending.len();
                pending.extend_from_slice(&frame);
                shard.pending_records.fetch_add(1, Ordering::Relaxed);
                start
            };

            let flush_result = match self.append_policy {
                AppendPolicy::Eager => self.flush_shard_with_guard(&shard, &append_guard),
                AppendPolicy::Buffered { max_pending_bytes } => {
                    if shard.pending.lock().len() >= max_pending_bytes {
                        self.flush_shard_with_guard(&shard, &append_guard)
                    } else {
                        Ok(())
                    }
                }
            };
            if let Err(e) = flush_result {
                // The caller sees `Err` and has not indexed this record, yet
                // its frame is still in `pending` — a later flush would
                // silently make a "failed" put durable, surfacing it as
                // stored data on restart's full scan. Retract it: a failed
                // flush leaves `pending` untouched, and `append_guard` is
                // still held from the append so nothing appended after our
                // frame, making the tail truncation exact. A torn write's
                // partially-written prefix is also reined back (`set_len` to
                // the still-current committed frontier) so leftover bytes
                // never read back as phantom records on a rebuild.
                let mut pending = shard.pending.lock();
                let end = pending_start.saturating_add(frame_len);
                if pending.len() >= end {
                    // Only discard the frame once the tail truncation has
                    // actually succeeded — otherwise the torn bytes stay on
                    // disk past `committed` with no frame left to account
                    // for them, and a restart's full scan (which reads real
                    // file length, not our tracked `file_len`) can surface
                    // them as phantom data. If truncation itself fails,
                    // leave `pending` untouched and poison the shard so
                    // nothing else appends past, or scans over, the corrupt
                    // tail.
                    match shard.file.set_len(committed) {
                        Ok(()) => {
                            pending.truncate(pending_start);
                            shard.pending_records.fetch_sub(1, Ordering::Relaxed);
                        }
                        Err(rollback_err) => {
                            drop(pending);
                            shard.poisoned.store(true, Ordering::Release);
                            return Err(io::Error::other(format!(
                                "flush failed ({e}) and rollback truncation also failed \
                                 ({rollback_err}); shard {} poisoned",
                                shard.slot
                            )));
                        }
                    }
                }
                return Err(e);
            }
            let disk_bytes = u64::try_from(frame_len).unwrap_or(u64::MAX);
            return Ok((shard.slot, virtual_end, disk_bytes));
        }
    }

    /// Validate a record's plaintext size against the packfile frame limit.
    ///
    /// This intentionally does not compress or allocate the frame. The append
    /// path performs the actual encoding exactly once.
    ///
    /// # Errors
    /// Returns `io::Error` if the record cannot fit within the frame limits.
    pub(crate) fn validate_record(record: &Record) -> io::Result<()> {
        let payload_len = u32::try_from(record.data.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "record payload exceeds u32::MAX",
            )
        })?;
        let frame_len = packfile::FRAME_FIXED_LEN
            .checked_add(payload_len)
            .expect("fixed frame length plus u32 payload cannot overflow u32");
        if frame_len > packfile::MAX_RECORD_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "record payload too large: {frame_len} > {}",
                    packfile::MAX_RECORD_LEN
                ),
            ));
        }
        Ok(())
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
        if shard.is_poisoned() {
            return Err(io::Error::other(format!(
                "shard {} is poisoned after an unrecoverable rollback failure",
                shard.slot
            )));
        }
        let guard = shard.append_lock.lock();
        self.flush_shard_with_guard(shard, &guard)
    }

    /// Body of `flush_shard` under a caller-supplied `append_lock` guard, so
    /// a writer that already holds the lock (see `put_record`, which must
    /// keep it across append → flush → rollback) can flush without
    /// re-acquiring it.
    fn flush_shard_with_guard(
        &self,
        shard: &Arc<Shard>,
        _guard: &parking_lot::MutexGuard<'_, ()>,
    ) -> io::Result<()> {
        // The guard proves `append_lock` is held, so this is the single
        // point every flush path — direct, via `flush_shard`, or inline
        // from `put_record`'s own automatic flush — actually funnels
        // through under lock. Enforcing the poison check here, not just at
        // each call site, means correctness never depends on every current
        // and future caller remembering to check first.
        if shard.is_poisoned() {
            return Err(io::Error::other(format!(
                "shard {} is poisoned after an unrecoverable rollback failure",
                shard.slot
            )));
        }
        {
            let mut pending_guard = shard.pending.lock();
            if pending_guard.is_empty() {
                // Nothing buffered: nothing to commit or dirty.
                return Ok(());
            }
            let committed = shard.file_len.load(Ordering::Acquire);
            // Write directly from the still-buffered bytes rather than
            // draining them first: `append_lock` is held for this whole
            // block, so nothing else can append to `pending` meanwhile,
            // and keeping the bytes in place until the write actually
            // succeeds means a failed write leaves them buffered for the
            // next flush attempt instead of silently discarding already
            // "successful" puts whose offsets the index has already
            // handed out.
            let file = shard.file.try_clone()?;
            #[cfg(unix)]
            file.write_all_at(&pending_guard, committed)?;
            #[cfg(not(unix))]
            {
                let mut file = file;
                file.seek(io::SeekFrom::Start(committed))?;
                file.write_all(&pending_guard)?;
            }
            #[cfg(target_os = "linux")]
            kick_writeback(&file, committed, pending_guard.len() as u64);
            let new_len = pending_guard.len() as u64;
            let file_len = committed.checked_add(new_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "committed + buffered len overflow",
                )
            })?;
            pending_guard.clear();
            drop(pending_guard);
            shard.file_len.store(file_len, Ordering::Release);
            shard.write_count.fetch_add(
                shard.pending_records.swap(0, Ordering::Relaxed),
                Ordering::Relaxed,
            );
            shard.bytes_written.fetch_add(new_len, Ordering::Relaxed);
            // Hold `append_lock` for the whole commit (snapshot → write →
            // length/accounting update) so no put can observe a file_len
            // that hasn't caught up with bytes already handed out.
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

            return u64::from(frame_len)
                .checked_add(8)
                .ok_or_else(|| StorageError::Corrupt("record length overflow".into()));
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
            if flags
                & !(packfile::FLAG_COMPRESSED
                    | packfile::FLAG_CRC_DISABLED
                    | packfile::FLAG_METADATA)
                != 0
            {
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
                Arc::clone(&mapping),
                prefix_end,
            );
        }
        unreachable!("read_at remap-retry is bounded to two iterations")
    }

    /// Verify and decode one complete v3 frame already bounded within an mmap.
    /// When `verify` is false — or the frame carries
    /// [`packfile::FLAG_CRC_DISABLED`] — the checksum is skipped, not
    /// compared; structural validation (lengths, flags, node bytes) still runs.
    ///
    /// `frame_base` is the absolute mmap offset of `payload[0]` (the flags
    /// byte), used to build the zero-copy `MmapRange` for the node bytes after
    /// any metadata block has been located.
    fn decode_record_frame(
        frame_len: [u8; 4],
        payload: &[u8],
        checksum: [u8; 4],
        verify: bool,
        mmap: Arc<Mmap>,
        frame_base: usize,
    ) -> Result<Record, crate::storage::StorageError> {
        use crate::storage::StorageError;

        let flags = payload[0];
        if flags
            & !(packfile::FLAG_COMPRESSED | packfile::FLAG_CRC_DISABLED | packfile::FLAG_METADATA)
            != 0
        {
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

        let rest = &payload[37..];
        let (metadata, node_offset, node_bytes) = if flags & packfile::FLAG_METADATA != 0 {
            let (metadata, consumed) = packfile::FrameMetadata::decode(rest)
                .map_err(|e| StorageError::Corrupt(format!("frame metadata: {e}")))?;
            let node_bytes = rest.get(consumed..).ok_or_else(|| {
                StorageError::Corrupt("frame metadata consumes past payload".into())
            })?;
            let node_offset = 37usize
                .checked_add(consumed)
                .ok_or_else(|| StorageError::Corrupt("frame node offset overflow".into()))?;
            (Some(metadata), node_offset, node_bytes)
        } else {
            (None, 37, rest)
        };

        let data = Self::decode_node_bytes(
            flags,
            uncompressed_len,
            node_bytes,
            MmapRange {
                mmap,
                start: frame_base
                    .checked_add(node_offset)
                    .ok_or_else(|| StorageError::Corrupt("frame node offset overflow".into()))?,
                end: frame_base
                    .checked_add(payload.len())
                    .ok_or_else(|| StorageError::Corrupt("frame end overflow".into()))?,
            },
        )?;

        Ok(Record {
            collection_id,
            hash,
            data,
            metadata,
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
        #[cfg(feature = "zstd")]
        {
            let decompressed = packfile::zstd_decompress(node_bytes, expected_len)
                .map_err(|e| StorageError::Corrupt(e.to_string()))?;
            Ok(bytes::Bytes::from(decompressed))
        }
        #[cfg(not(feature = "zstd"))]
        {
            Err(StorageError::Corrupt(
                "compressed node but this build was compiled without the `zstd` feature".into(),
            ))
        }
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
                Self::persist_pool_meta_at(&self.base_dir, next, self.bucket_seed)?;

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
    /// `persist_stats` issues no fsync (the snapshot is best-effort
    /// observability), so this path never forces durability work onto a
    /// write.
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

    /// Number of `shard_stats.bin` snapshot rewrites that durably landed over
    /// this pool's lifetime (see [`Self::stats_snapshots`]).
    #[must_use]
    pub fn stats_snapshots(&self) -> u64 {
        self.stats_snapshots.load(Ordering::Relaxed)
    }

    /// Claim this shard's dirty bit, returning whether it was set. The
    /// pool-wide `dirty` lock is held only for the check-and-remove, never
    /// across an fsync, so concurrent sync callers don't serialize on it.
    /// The wait is folded into `dirty_lock_wait`.
    fn take_dirty(&self, id: u16) -> bool {
        let lock_wait = Instant::now();
        let mut dirty = self.dirty.lock();
        self.dirty_lock_wait.fetch_add(
            u64::try_from(lock_wait.elapsed().as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        dirty.remove(&id)
    }

    /// Re-mark a shard dirty after a failed fsync so the next sync retries
    /// it. Callers hold the shard's `sync_lock`, so no concurrent syncer can
    /// observe the transiently-clean state.
    fn restore_dirty(&self, id: u16) {
        self.dirty.lock().insert(id);
    }

    /// Snapshot the ids currently marked dirty without holding the lock
    /// across any fsync.
    fn dirty_snapshot(&self) -> Vec<u16> {
        let lock_wait = Instant::now();
        let dirty = self.dirty.lock();
        self.dirty_lock_wait.fetch_add(
            u64::try_from(lock_wait.elapsed().as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        dirty.iter().copied().collect()
    }

    /// Sync all shards to disk.
    ///
    /// Commits any buffered frames first, so fsync covers everything a
    /// caller believes it has put.
    ///
    /// **Deliberately does not skip-based-coalesce.** Unlike [`Self::sync_dirty`],
    /// which may skip a shard once another caller has fsynced it, this fsyncs
    /// every shard unconditionally: its contract is that everything put before
    /// the call is durable, and "someone else fsynced this shard at some point"
    /// does not prove that. The per-shard `sync_lock` here therefore provides
    /// mutual exclusion only, not the coalescing win `sync_dirty` gets — two
    /// concurrent `sync_all` callers can each issue their own fsync, which is
    /// correct, just not deduplicated.
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_all(&self) -> io::Result<()> {
        let flush_started = Instant::now();
        self.flush_all()?;
        let flush_elapsed = flush_started.elapsed();
        let fsync_started = Instant::now();
        let mut had_dirty = false;
        for (id, shard) in self.all_shards() {
            // Full-pool sync fsyncs every shard unconditionally (that is its
            // contract), so unlike `sync_dirty` it can't skip a clean shard —
            // but the per-shard lock still serializes it against a concurrent
            // `sync_dirty` fsyncing the same file, and the dirty bit is
            // claimed before the fsync rather than under the pool lock
            // across it.
            let _sync_guard = shard.sync_lock.lock();
            if self.take_dirty(id) {
                had_dirty = true;
            }
            if let Err(error) = shard.file.sync_all() {
                self.restore_dirty(id);
                return Err(error);
            }
            shard.sync_count.fetch_add(1, Ordering::Relaxed);
        }
        let fsync_elapsed = fsync_started.elapsed();
        *self.last_sync_split.lock() = Some((flush_elapsed, fsync_elapsed));
        // The stats snapshot is observability, not state: only rewrite it when
        // a sync actually moved or committed data (parallel to `sync_dirty`),
        // so a steady-state writer that syncs between writes pays no metadata
        // write+rename for a file nothing changed. Periodic freshness is
        // `maybe_persist_stats`' job.
        if had_dirty {
            self.persist_stats_best_effort();
        }
        Ok(())
    }

    /// (flush, fsync) wall-clock split of the last sync — `sync_all` or the
    /// dirty-scoped `sync_dirty` — if any.
    #[must_use]
    pub fn last_sync_split(&self) -> Option<(Duration, Duration)> {
        *self.last_sync_split.lock()
    }

    /// Cumulative wall-clock time spent waiting to acquire the dirty-set
    /// lock inside `sync_dirty` and `sync_all`. Monotonically increasing,
    /// never reset. Compare against total sync wall time to gauge whether
    /// the coarse lock is a real contention point under concurrent load.
    #[must_use]
    pub fn dirty_lock_wait(&self) -> Duration {
        Duration::from_micros(self.dirty_lock_wait.load(Ordering::Relaxed))
    }

    /// Sync only shards written to since the last sync.
    ///
    /// Each shard's dirty bit is claimed before its fsync and restored if the
    /// fsync fails, so a partial failure leaves the untried shards marked
    /// dirty for the next call.
    ///
    /// Concurrent callers coalesce per shard: the first to claim a shard's
    /// dirty bit fsyncs it; any other caller wanting the same shard waits on
    /// that shard's `sync_lock`, then finds the bit already cleared and skips
    /// its own fsync. The pool-wide `dirty` lock is never held across an
    /// fsync.
    ///
    /// Skipping is safe because a shard's dirty bit is only cleared when an
    /// fsync that *covers that shard's current bytes* has completed: every
    /// flush writes its bytes (`flush_shard_with_guard`) before marking the
    /// shard dirty, and each caller flushes before it claims — so any byte
    /// whose dirty bit this caller observes was written before the fsync that
    /// clears it, and an fsync covers all writes that completed before it.
    /// A write that lands after the claim re-marks the shard dirty and is
    /// caught by the next sync.
    ///
    /// Commits any buffered frames first, so the fsync covers everything a
    /// caller believes it has put.
    ///
    /// # Cross-process visibility
    ///
    /// This makes the dirty shard bytes durable, but it is not the
    /// transaction-level durability or publication barrier: it does not commit
    /// the journal/LSN boundary and does not advance the LSN a reader in
    /// another process observes, so nothing about another process's view
    /// follows from it returning `Ok`.
    ///
    /// * With a [journal](crate::journal) enabled, the transaction-level
    ///   durability barrier is
    ///   [`JournalCoordinator::sync_through`](crate::journal::JournalCoordinator::sync_through)
    ///   (reached via the packfile storage sync path), and shard fsyncs are
    ///   acceleration recovered from the journal on reopen. A separate process
    ///   observes committed data through the read-committed overlay
    ///   (`PackfileStorage::open_read_committed` / `get_read_committed`, the
    ///   `multi-reader` feature), whose boundary is the journal's committed
    ///   LSN. That boundary can advance *before* this call, and a
    ///   visible-but-uncommitted group is not crash-durable.
    /// * Without a journal, a peer sees shard bytes only after an explicit
    ///   [`StorageEngine::refresh_collection`](crate::storage::StorageEngine::refresh_collection)
    ///   re-scan; the engine has no ambient cross-process invalidation. A
    ///   successful `sync_dirty` then guarantees those bytes survive a crash,
    ///   not that a peer has picked them up.
    ///
    /// Same-process readers go through the live index; cross-process readers
    /// go through the journal overlay (or an explicit rescan).
    ///
    /// # Errors
    /// Returns `io::Error` on sync failure.
    pub fn sync_dirty(&self) -> io::Result<()> {
        let flush_started = Instant::now();
        self.flush_all()?;
        let flush_elapsed = flush_started.elapsed();
        let fsync_started = Instant::now();
        let mut synced_any = false;
        for id in self.dirty_snapshot() {
            let Some(shard) = self.get_shard(id) else {
                // Slot was retired between snapshot and lookup; leave any
                // remaining dirty bit for the next pass to resolve.
                continue;
            };
            let _sync_guard = shard.sync_lock.lock();
            if !self.take_dirty(id) {
                // A concurrent syncer we just waited for already fsynced
                // this shard — nothing left to do.
                continue;
            }
            // `sync_data` (fdatasync): the packfile is append-only and its
            // frame framing/size live in the data stream, so flushing data
            // plus the size metadata makes appended bytes readable after a
            // crash; the mode/timestamp metadata `sync_all` would also write
            // is not needed. The full-barrier `sync_all` above keeps
            // `sync_all`.
            if let Err(error) = shard.file.sync_data() {
                self.restore_dirty(id);
                return Err(error);
            }
            shard.sync_count.fetch_add(1, Ordering::Relaxed);
            synced_any = true;
        }
        let fsync_elapsed = fsync_started.elapsed();
        *self.last_sync_split.lock() = Some((flush_elapsed, fsync_elapsed));
        if synced_any {
            // Rate-limit the snapshot on the hot dirty path: it is
            // observability data, and `sync_all` (on dirty shards) / `Drop`
            // persist it.
            self.maybe_persist_stats(STATS_FLUSH_MIN_INTERVAL);
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
    ///
    /// Also does nothing if the shard has buffered frames that fail to
    /// flush: retiring anyway would mark the shard for deletion (via
    /// `Shard::drop`) while data an index still references is only in
    /// memory, destroying it. Leaving the slot occupied means the next
    /// flush attempt (or another retire) gets another chance.
    pub fn retire_slot(&self, slot: u16) {
        // Commit any buffered frames first: records are only ever retired
        // after their bytes are on disk, so a retire can never strand data
        // that an index still references.
        if let Some(shard) = self.get_shard(slot) {
            if self.flush_shard(&shard).is_err() {
                return;
            }
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
    use fs2::FileExt as FileExtTrait;

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
            metadata: None,
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
    fn poisoned_shard_refuses_append_and_flush() {
        let dir = test_dir("poisoned_shard_refuses_writes");
        let pool = ShardPool::open(dir).unwrap();
        let record = test_record(0x01, 0xAA, b"before poison");
        pool.put_record(&record).unwrap();
        let shard = pool.get_shard(0).unwrap();

        // Simulate the state left behind by an unrecoverable rollback
        // failure (see `put_record`'s failed-flush path) directly, rather
        // than engineering an actual `set_len` failure.
        shard.poisoned.store(true, Ordering::Release);

        let after = test_record(0x02, 0xBB, b"after poison");
        let put_err = pool
            .put_record(&after)
            .expect_err("a poisoned shard must refuse further appends");
        assert!(put_err.to_string().contains("poisoned"));

        let flush_err = pool
            .flush_shard(&shard)
            .expect_err("a poisoned shard must refuse flush too");
        assert!(flush_err.to_string().contains("poisoned"));
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
    fn dirty_lock_wait_starts_at_zero_and_accrues_on_sync() {
        let dir = test_dir("dirty_lock_wait");
        let pool = ShardPool::open(dir).unwrap();
        assert_eq!(
            pool.dirty_lock_wait(),
            Duration::ZERO,
            "a fresh pool has never contended for the dirty-set lock"
        );

        let record = test_record(0x01, 0xAA, b"hello");
        pool.put_record(&record).unwrap();
        pool.sync_dirty().unwrap();
        // A single-threaded sync still acquires the lock (uncontended), so
        // this only asserts the counter is wired up and monotonic, not that
        // it measures anything close to real contention. Since the per-shard
        // sync coalescing landed, the pool-wide lock is only held for the
        // brief dirty-bit check/claim, so real contention shows up here only
        // if many syncers collide on that short critical section.
        let after_one_sync = pool.dirty_lock_wait();

        pool.put_record(&test_record(0x01, 0xAB, b"world")).unwrap();
        pool.sync_dirty().unwrap();
        assert!(
            pool.dirty_lock_wait() >= after_one_sync,
            "dirty_lock_wait must never decrease"
        );
    }

    /// N concurrent `sync_dirty` callers for the same shard must coalesce
    /// into one physical fsync: the first to claim the dirty bit fsyncs, the
    /// rest wait on the shard's `sync_lock` for that fsync, then find the bit
    /// already cleared and skip theirs.
    #[test]
    fn concurrent_sync_dirty_coalesces_to_a_single_fsync() {
        use std::thread;

        let dir = test_dir("concurrent_sync_dirty_coalesce");
        let pool = Arc::new(ShardPool::open(dir).unwrap());
        let record = test_record(0x01, 0xAA, b"coalesce me");
        let (slot, _offset) = pool.put_record(&record).unwrap();
        let shard = pool.get_shard(slot).unwrap();
        assert_eq!(shard.stats().sync_count, 0);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || pool.sync_dirty())
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        assert_eq!(
            shard.stats().sync_count,
            1,
            "8 concurrent sync_dirty callers must coalesce into one fsync"
        );
        assert!(
            pool.dirty.lock().is_empty(),
            "sync_dirty must clear the dirty bit it claimed"
        );
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
    fn test_clean_sync_all_does_not_rewrite_stats_snapshot() {
        let dir = test_dir("clean_sync_stats");
        let pool = ShardPool::open(dir).unwrap();
        let before = pool.stats_snapshots();
        pool.sync_all().unwrap();
        assert_eq!(
            pool.stats_snapshots(),
            before,
            "a clean sync must not rewrite the stats snapshot"
        );
        assert_eq!(
            pool.stats_persisted_at(),
            None,
            "a clean sync must not persist any stats snapshot"
        );
    }

    #[test]
    fn test_dirty_sync_all_rewrites_stats_snapshot_once() {
        let dir = test_dir("dirty_sync_stats_once");
        let pool = ShardPool::open(dir).unwrap();
        pool.put_record(&test_record(0x02, 0xBB, b"payload"))
            .unwrap();
        let before = pool.stats_snapshots();
        pool.sync_all().unwrap();
        assert_eq!(
            pool.stats_snapshots(),
            before + 1,
            "a dirty sync must persist the stats snapshot exactly once"
        );
        assert!(
            pool.stats_persisted_at().is_some(),
            "the dirty sync must leave a stats snapshot timestamp"
        );
    }

    #[test]
    fn test_maybe_persist_stats_still_works_without_data_mutation() {
        let dir = test_dir("periodic_stats_no_mutation");
        let pool = ShardPool::open(dir).unwrap();
        assert_eq!(pool.stats_persisted_at(), None);
        let before = pool.stats_snapshots();
        pool.maybe_persist_stats(Duration::ZERO);
        assert_eq!(
            pool.stats_snapshots(),
            before + 1,
            "the periodic flush must write a snapshot even with no dirty shards"
        );
        assert!(
            pool.stats_persisted_at().is_some(),
            "the periodic flush must leave a stats snapshot timestamp"
        );
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

    #[test]
    fn test_lock_contention_probe_distinguishes_held_marker() {
        let dir = test_dir("writer_lock_contention_probe");
        std::fs::create_dir_all(&dir).unwrap();
        let lock_path = dir.join(".mtxdb.lock");
        #[cfg(target_os = "linux")]
        let marker = format!(
            "{} {}\n",
            std::process::id(),
            ShardPool::proc_start_time("self").unwrap()
        );
        #[cfg(not(target_os = "linux"))]
        let marker = format!("{}\n", std::process::id());
        std::fs::write(&lock_path, marker).unwrap();
        let file = File::options()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        FileExtTrait::try_lock_exclusive(&file).unwrap();
        assert_eq!(ShardPool::lock_contended(&lock_path), Some(true));
        FileExtTrait::unlock(&file).unwrap();
        assert_eq!(ShardPool::lock_contended(&lock_path), Some(false));
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
                metadata: None,
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
            restored.map(|m| m.next_pack_id),
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
            restored.map(|m| m.next_pack_id),
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

    /// Simulated intermediate recovery states during fresh pool initialization.
    /// Exercises recovery from synthetic intermediate directory states representing
    /// crashes or interruptions before or between filesystem operations, verifying that:
    /// 1. Stale temporary files from an aborted open are ignored safely.
    /// 2. If pool.meta was renamed (`next_pack_id` = 1) but the packfile creation
    ///    aborted before pack rename, reopen discovers 0 shards, respects the
    ///    persisted pool.meta high-water mark, and allocates `pack_id` = 1,
    ///    never reusing `pack_id` = 0.
    /// 3. Normal clean initialization produces durable pool.meta = 1 and pack 0.
    #[test]
    fn test_fresh_pool_recovery_states() {
        // State 1: interruption before any rename (only tmp files exist)
        {
            let dir = test_dir("interruption_state_1");
            // Place fake orphaned .tmp files
            fs::write(
                dir.join("pool.meta.tmp.1234"),
                b"MTXP\x01\x01\x00\x00\x00\x00\x00\x00\x00",
            )
            .unwrap();
            fs::write(dir.join("pack_0000000000000000.tmp.1234.0"), b"PACK\x02...").unwrap();

            let pool = ShardPool::open(dir.clone()).unwrap();
            assert_eq!(pool.all_shards().len(), 1);
            let summaries = pool.summaries();
            assert_eq!(
                summaries[0].pack_id, 0,
                "fresh allocation after tmp crash gets pack_id 0"
            );
            let restored = ShardPool::restore_pool_meta(&dir).unwrap();
            assert_eq!(
                restored.map(|m| m.next_pack_id),
                Some(1),
                "pool.meta must have next_pack_id = 1"
            );
        }

        // State 2: Crash after pool.meta rename, but before initial pack rename
        {
            let dir = test_dir("interruption_state_2");
            // Simulate pool.meta was renamed with next_pack_id = 1, but
            // before the deferred directory sync.
            ShardPool::persist_pool_meta_at_sync_dir(&dir, 1, 0xDEAD_BEEF, false).unwrap();
            // Stale pack tmp left behind
            fs::write(
                dir.join("pack_0000000000000000.tmp.1234.0"),
                b"fake pack tmp",
            )
            .unwrap();

            // Reopen must succeed, see 0 canonical shards, read pool.meta (next_pack_id = 1),
            // and allocate pack_id = 1 (preventing reuse of pack_id 0).
            let pool = ShardPool::open(dir.clone()).unwrap();
            let summaries = pool.summaries();
            assert_eq!(summaries.len(), 1);
            assert_eq!(
                summaries[0].pack_id, 1,
                "must assign pack_id 1 to prevent reuse of 0"
            );
            let restored = ShardPool::restore_pool_meta(&dir).unwrap();
            assert_eq!(
                restored.map(|m| m.next_pack_id),
                Some(2),
                "pool.meta must now be advanced to 2"
            );
        }

        // State 3: Clean fresh initialization
        {
            let dir = test_dir("interruption_state_3");
            let pool = ShardPool::open(dir.clone()).unwrap();
            let summaries = pool.summaries();
            assert_eq!(summaries.len(), 1);
            assert_eq!(summaries[0].pack_id, 0);
            assert_eq!(
                ShardPool::restore_pool_meta(&dir)
                    .unwrap()
                    .map(|m| m.next_pack_id),
                Some(1)
            );
            drop(pool);

            // Reopen sees existing pack 0
            let pool2 = ShardPool::open(dir.clone()).unwrap();
            let summaries2 = pool2.summaries();
            assert_eq!(summaries2.len(), 1);
            assert_eq!(summaries2[0].pack_id, 0);
            assert_eq!(
                ShardPool::restore_pool_meta(&dir)
                    .unwrap()
                    .map(|m| m.next_pack_id),
                Some(1)
            );
        }
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
                metadata: None,
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
                metadata: None,
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
                    metadata: None,
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

    /// Regression: `MAX_SHARD_BYTES` must stay within the offset field that
    /// `IndexEntry` stores as `offset + 1`, so the maximum valid offset must
    /// never reach the empty-slot sentinel boundary.
    #[test]
    fn max_shard_bytes_fits_index_slot() {
        // The largest offset a shard can ever present must survive
        // IndexEntry::new without panicking.
        let max_offset = MAX_SHARD_BYTES - 1;
        let slot = crate::index::IndexEntry::new(0, 0, max_offset);
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
