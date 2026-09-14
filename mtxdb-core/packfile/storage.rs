use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::csr::Csr;
use crate::index::delta::{self, DELTA_LOG_HEADER_LEN, INDEX_DELTA_FILE};
use crate::index::format::DeltaFrame;
use crate::index::{InsertError, LossyIndex};
use crate::packfile::{self, Record};
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{NodeData, NodeId, NodeRef, StorageEngine, StorageError};

/// Callback that rewrites a node's child references given resolved child data,
/// used to inline already-cached children in place of lazy hash pointers.
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[Option<Arc<NodeData>>]) -> NodeData;

/// Result of [`PackfileStorage::plan_collection_repack`] — what a real repack
/// of this collection would do, computed exactly (same scan + reachability
/// pass) but without performing any writes.
#[derive(Debug, Clone, Default)]
pub struct RepackPlan {
    /// Records that would survive (be kept) by this repack.
    pub kept: usize,
    /// Records that would be dropped (found unreachable) by this repack.
    pub dropped: usize,
    /// Total on-disk frame bytes of the records that would be kept.
    pub kept_bytes: u64,
    /// Total on-disk frame bytes of records that would be pruned.
    pub dropped_bytes: u64,
    /// Every shard id at least one kept record currently lives in.
    pub shards_touched: Vec<u16>,
}

/// Which index-loading path a `PackfileStorage` open took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenPath {
    /// A persisted index checkpoint validated against the current pack set's
    /// fingerprint was loaded in place of a rescan.
    Checkpoint,
    /// No usable checkpoint: every pack's records were scanned and the
    /// per-collection indexes rebuilt from scratch.
    FullScan,
}

/// Where a checkpoint-path open sourced each collection's per-shard counts
/// and home-shard assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookkeepingSource {
    /// Reconstructed from the fingerprint-gated `shard_collections.bin`
    /// records — no slot walk across any collection index.
    Sidecar,
    /// Computed by walking every slot of every collection index.
    SlotScan,
}

/// Wall-clock breakdown of one `PackfileStorage` open, by phase. Measured
/// with a handful of `Instant::now()` deltas on the open path only, so
/// recording these doesn't perturb the numbers they report.
///
/// The per-phase fields partition the synchronous open work; see the
/// struct-level timing doc for what each phase covers. On the checkpoint
/// path `full_scan` is zero; on the fallback path `checkpoint_decode` and
/// `fingerprint` still reflect the attempt that failed validation (e.g. a
/// missing or stale checkpoint) before the scan had to run.
#[derive(Debug, Clone, Copy)]
pub struct OpenTimings {
    /// Directory scan + shard file open/validation, plus writer-lock take.
    pub shard_open: std::time::Duration,
    /// Reading the collection-order and deleted-collections sidecars.
    pub metadata_load: std::time::Duration,
    /// Reading + decoding the persisted index checkpoint.
    pub checkpoint_decode: std::time::Duration,
    /// Recomputing the live `(pack_id, file_len)` set's fingerprint.
    pub fingerprint: std::time::Duration,
    /// Deserializing checkpoint index blobs into live collections (checkpoint
    /// path) or building per-collection indexes from scanned records
    /// (fallback path). On the checkpoint path this includes replaying any
    /// fingerprint/generation-gated delta-log frames on top of the raw slots.
    pub index_materialization: std::time::Duration,
    /// Reading the delta log, validating its gates, and applying frames to
    /// each affected collection (checkpoint path only; ZERO when there is no
    /// committed log to replay).
    pub delta_replay: std::time::Duration,
    /// The full packfile scan + torn-tail recovery pass (fallback only).
    pub full_scan: std::time::Duration,
    /// Total synchronous wall time of the open.
    pub total: std::time::Duration,
    /// Which index-loading path was taken.
    pub path: OpenPath,
    /// Where per-shard counts and home-shard assignment came from (checkpoint
    /// path only; the fallback path builds them from the scan).
    pub bookkeeping_source: BookkeepingSource,
}

impl Default for OpenTimings {
    fn default() -> Self {
        Self {
            shard_open: std::time::Duration::ZERO,
            metadata_load: std::time::Duration::ZERO,
            checkpoint_decode: std::time::Duration::ZERO,
            fingerprint: std::time::Duration::ZERO,
            index_materialization: std::time::Duration::ZERO,
            delta_replay: std::time::Duration::ZERO,
            full_scan: std::time::Duration::ZERO,
            total: std::time::Duration::ZERO,
            path: OpenPath::FullScan,
            bookkeeping_source: BookkeepingSource::SlotScan,
        }
    }
}

/// Wall-clock breakdown of one sync — the append-path `sync()` (dirty-scoped
/// flush/fsync) or the full `sync_all` — by phase. The pack phases come from
/// `ShardPool::last_sync_split` (which measures the flush and fsync legs
/// separately) and the metadata phases from the sync method itself.
#[derive(Debug, Clone, Copy)]
pub struct SyncTimings {
    /// Writing buffered frames out to the pack files.
    pub pack_flush: std::time::Duration,
    /// fsync'ing the pack files.
    pub pack_fsync: std::time::Duration,
    /// Writing the shard→collection metadata sidecar.
    pub sidecar: std::time::Duration,
    /// Persisting the index checkpoint (complete rewrite) or appending the
    /// incremental delta log instead (ZERO on the path not taken; the appended
    /// case is measured when a dirty `sync()` can continue an existing log).
    pub delta_log: std::time::Duration,
    /// Persisting the index checkpoint.
    pub checkpoint: std::time::Duration,
    /// Total wall time of the `sync_all` call.
    pub total: std::time::Duration,
}

impl Default for SyncTimings {
    fn default() -> Self {
        Self {
            pack_flush: std::time::Duration::ZERO,
            pack_fsync: std::time::Duration::ZERO,
            sidecar: std::time::Duration::ZERO,
            delta_log: std::time::Duration::ZERO,
            checkpoint: std::time::Duration::ZERO,
            total: std::time::Duration::ZERO,
        }
    }
}

/// Bounds on a [`PackfileStorage::walk_ancestors`] call.
///
/// Without a cap, a walk whose `stop_at` set is never reached (e.g. the
/// caller passed a stale or wrong boundary) silently degrades into "walk
/// the entire history of this branch" — `max_nodes` is the safety valve
/// for that case.
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkLimits {
    /// Stop discovering new ancestors once this many nodes have been
    /// visited. `None` means unbounded (walk until `stop_at` or the
    /// collection's true roots are reached).
    pub max_nodes: Option<usize>,
}

/// Lazy, ancestor-first iterator over the result of
/// [`PackfileStorage::walk_ancestors`].
///
/// Every node was already resolved (hash-verified, same as [`PackfileStorage::get`])
/// against one frozen generation snapshot while [`PackfileStorage::walk_ancestors`]
/// built the walk — this iterator just replays that in ancestor-first
/// order. No further store access happens here, so a repack that runs
/// after the walk was built can't cause a node to go missing partway
/// through iteration.
pub struct DagWalk {
    order: std::vec::IntoIter<[u8; 16]>,
    resolved: HashMap<[u8; 16], NodeData>,
}

impl Iterator for DagWalk {
    type Item = Result<(NodeId, NodeData), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        let hash = self.order.next()?;
        let data = self
            .resolved
            .remove(&hash)
            .expect("order only contains hashes this walk itself resolved");
        Some(Ok((hash, data)))
    }
}

/// A live hash set plus the adjacency discovered while determining it, as
/// returned by the reachability repacker's live-set computation.
type AdjacencyResult = (Vec<[u8; 16]>, HashMap<[u8; 16], Vec<[u8; 16]>>);

/// One shard's scanned `(hash, offset)` entries for a single collection, as
/// produced by a repack's initial shard scan.
type ScannedShard = (u16, Vec<([u8; 16], u64)>);

/// Deduplicated live-location map for one collection during a repack scan.
type RepackRecordMap = HashMap<[u8; 16], (u16, u64)>;
/// Replacement index entries produced while copying one repack collection.
type RepackOffsets = Vec<([u8; 16], u16, u64)>;
/// Return type of [`PackfileStorage::repack_scan_incremental`]: the merged
/// live-location map and (on the incremental path) the previous cursor state
/// that the adjacency helpers need to skip re-reading already-seen nodes.
type RepackScanResult = (RepackRecordMap, Option<RepackIncrementalState>);

/// One scanned record's `(shard_id, hash, offset)`, as accumulated per
/// collection during `PackfileStorage::open_with_options`'s initial scan.
type ShardRecord = (u16, [u8; 16], u64, u64);
/// `(collection_id, entry count, index memory usage in bytes, index capacity)`.
pub type CollectionSummary = ([u8; 16], usize, usize, u32);

/// Accumulator for `PackfileStorage::open_with_options`'s phase 2 —
/// bundles the three maps `init_collection_from_scan` fills in per collection, so
/// that function takes one out-parameter instead of three.
#[derive(Default)]
struct RoomScanOutput {
    collections: HashMap<[u8; 16], ArcSwap<RoomGeneration>>,
    shard_collections: HashMap<u64, HashMap<[u8; 16], u64>>,
    collection_shards: HashMap<[u8; 16], HashSet<u64>>,
}

/// Magic bytes + version identifying the persisted shard→collection directory
/// format (see `PackfileStorage::persist_shard_collections`).
///
/// Pre-release, no compatibility fallback: an unrecognized version is
/// treated exactly like a missing/corrupt file (see
/// `read_persisted_shard_collections`) — reset or let the next sync
/// regenerate it, not a format this reader tries to still understand.
/// This sidecar has changed shape three times already (adding insertion
/// order, switching `shard_id` for `pack_id`, then v4 adding the
/// pack-fingerprint gate); none of that carries forward, since nothing
/// depends on reading a store from before the current format existed.
const SHARD_ROOMS_MAGIC: &[u8; 4] = b"MSRM";
/// v4 pins the reduced bookkeeping to the exact `(pack_id, file_len)` set by
/// carrying the same `pack_fingerprint` as the index checkpoint.
const SHARD_ROOMS_VERSION: u8 = 4;
/// Header size: magic(4) + version(1) + `pack_fingerprint(8)` + `persisted_at(8)`.
const SHARD_ROOMS_HEADER_LEN: usize = 4 + 1 + 8 + 8;
/// One entry: `pack_id`(8) + `collection_id`(16) + count(8) + the
/// collection's stable insertion ordinal(8).
const SHARD_ROOMS_RECORD_LEN: usize = 8 + 16 + 8 + 8;

/// The largest record offset `IndexSlot` can represent: its 28-bit offset
/// field stores `offset + 1`, reserving the all-zeros encoding for the empty
/// sentinel. Offsets beyond this can only come from legacy or externally
/// created oversized packs — this engine's own writes rotate long before
/// reaching it (`MAX_SHARD_BYTES` caps each shard's file size).
const PACK_INDEX_OFFSET_LIMIT: u64 = (1u64 << 28) - 2;

/// Byte ceiling for the incremental index delta log. Once a session's
/// accumulated frames exceed this, the next `sync()` stops appending and does
/// a full checkpoint rewrite (which truncates the log), keeping replay cost
/// and the log file bounded. Frames are 36 bytes each, so this is on the order
/// of ~230k appends between full rewrites.
const DELTA_LOG_CAP_BYTES: u64 = 8 * 1024 * 1024;

/// Reject a record whose in-shard offset the 28-bit `IndexSlot` field cannot
/// represent, so the caller surfaces a `StorageError::Corrupt` instead of
/// silently dropping the record (or panicking in `IndexSlot::new`).
fn check_index_offset(shard_id: u16, hash: &[u8; 16], offset: u64) -> Result<(), StorageError> {
    if offset > PACK_INDEX_OFFSET_LIMIT {
        return Err(StorageError::Corrupt(format!(
            "shard {shard_id} holds record {hash:?} at offset {offset}, beyond the index offset limit {PACK_INDEX_OFFSET_LIMIT}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PersistedShardRoom {
    pack_id: u64,
    collection_id: [u8; 16],
    count: u64,
    insertion_order: u64,
}

/// A decoded shard→collection directory: the pack set it was written
/// against, when it was written, and the per-(pack, collection) records.
#[derive(Clone)]
struct PersistedShardDirectory {
    /// The `pack_fingerprint` of the `(pack_id, file_len)` set the counts
    /// apply to. Only a directory whose fingerprint equals the index
    /// checkpoint's may serve open's per-shard bookkeeping without re-walking
    /// every slot — any other directory is stale (or corrupt) and must be
    /// ignored in favor of the slot walk.
    fingerprint: u64,
    /// Unix-seconds timestamp of when the directory was persisted.
    persisted_at: u64,
    /// One record per (pack, collection) the store contains.
    records: Vec<PersistedShardRoom>,
}

/// Decode the small inspection sidecar. This is deliberately shared by all
/// read-only CLI summary helpers so they agree on validation and format
/// compatibility.
fn read_persisted_shard_collections(base_dir: &std::path::Path) -> Option<PersistedShardDirectory> {
    let buf = fs::read(base_dir.join("shard_collections.bin")).ok()?;
    if buf.len() < SHARD_ROOMS_HEADER_LEN
        || &buf[0..4] != SHARD_ROOMS_MAGIC
        || buf[4] != SHARD_ROOMS_VERSION
    {
        return None;
    }
    let body = &buf[SHARD_ROOMS_HEADER_LEN..];
    if body.len().checked_rem(SHARD_ROOMS_RECORD_LEN) != Some(0) {
        return None;
    }
    let fingerprint = u64::from_le_bytes(buf[5..13].try_into().ok()?);
    let persisted_at = u64::from_le_bytes(buf[13..21].try_into().ok()?);
    let records = body
        .chunks_exact(SHARD_ROOMS_RECORD_LEN)
        .map(|chunk| {
            let pack_id = u64::from_le_bytes(chunk[0..8].try_into().ok()?);
            let mut collection_id = [0u8; 16];
            collection_id.copy_from_slice(&chunk[8..24]);
            let count = u64::from_le_bytes(chunk[24..32].try_into().ok()?);
            let insertion_order = u64::from_le_bytes(chunk[32..40].try_into().ok()?);
            Some(PersistedShardRoom {
                pack_id,
                collection_id,
                count,
                insertion_order,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(PersistedShardDirectory {
        fingerprint,
        persisted_at,
        records,
    })
}

/// The durable insertion order supplied by the inspection directory.
fn persisted_collection_order(base_dir: &std::path::Path) -> Option<Vec<[u8; 16]>> {
    let records = read_persisted_shard_collections(base_dir)?.records;
    let mut orders: HashMap<[u8; 16], u64> = HashMap::new();
    for record in records {
        let order = record.insertion_order;
        orders
            .entry(record.collection_id)
            .and_modify(|current| *current = (*current).min(order))
            .or_insert(order);
    }
    let mut collections: Vec<([u8; 16], u64)> = orders.into_iter().collect();
    collections.sort_unstable_by_key(|(collection_id, order)| (*order, *collection_id));
    Some(
        collections
            .into_iter()
            .map(|(collection_id, _)| collection_id)
            .collect(),
    )
}

/// Disambiguates concurrent `persist_shard_collections` tmp filenames within
/// this process, paired with the process id for uniqueness across
/// processes — same rationale as `shard::STATS_TMP_COUNTER`.
static SHARD_ROOMS_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Immutable snapshot of a collection's in-memory state.
///
/// Index and cache are bundled so readers see a consistent triple
/// via a single `ArcSwap::load()` — no three-lock coordination.
///
/// `generation` is the collection index's structural generation: it
/// increments on every capacity-changing or shape-changing rebuild (growth,
/// repack, refresh) and is preserved across plain COW writes. The delta log
/// stamps every frame with the generation it was recorded under, so a log
/// never replay-applies frames for a pre-resize table onto a resized one.
#[derive(Clone)]
struct RoomGeneration {
    index: LossyIndex,
    cache: Arc<NodeCache>,
    generation: u64,
}

/// Session state for the incremental index delta log (`index.delta`).
///
/// A fresh checkpoint rewrite re-bases this structure; between rewrites, live
/// index mutations are recorded as [`crate::index::format::DeltaFrame`]s in
/// `pending` and appended as a single framed batch by the next `sync()`, so a
/// dirty sync pays a few-dozen-byte append instead of a multi-megabyte
/// checkpoint rewrite. A structurally unsafe transition sets `invalid` (see
/// below), which forces the next sync back to a full rewrite.
#[derive(Default)]
struct DeltaLogState {
    /// Fingerprint of the checkpoint this session continues. `None` when the
    /// store opened via a full rescan and hasn't rewritten a checkpoint yet —
    /// every such open forces the next dirty sync to a full rewrite.
    base_fingerprint: Option<u64>,
    /// Generation of each collection at the last full checkpoint rewrite. A
    /// frame is only recorded when the collection's live generation still
    /// equals this value; bumping a generation (structural change) therefore
    /// silently stops recording for that collection until the next rewrite.
    base_generations: HashMap<[u8; 16], u64>,
    /// Frames accumulated since the last persist (checkpoint rewrite or delta
    /// append). Cleared on every successful persist.
    pending: Vec<DeltaFrame>,
    /// Bytes already committed to the on-disk log (upper bound that a fresh
    /// header may still need to be written). Resets to zero on rewrite.
    log_bytes: u64,
    /// When set, the pending log can no longer be continued correctly —
    /// capacity growth, full rebuild, repack, refresh, and collection
    /// create/delete all invalidate it because replay couldn't reconstruct the
    /// index from the last checkpoint alone. Forces the next full rewrite.
    invalid: bool,
}

/// A content-addressed packfile storage engine backed by a global shard pool.
///
/// All collections share a small pool of shard files (~4, each ~2GB), keeping
/// file descriptor usage constant regardless of collection count.
///
/// Per-collection state (index + cache) is bundled in an immutable
/// `RoomGeneration` and swapped atomically via `ArcSwap`:
///
/// - **Reads**: `load_full()` once, see a consistent snapshot.
/// - **Writes**: build new generation under put lock, swap atomically.
/// - **Delete**: drop the generation — cache disappears with it.
///
/// # Timing instrumentation
///
/// Every open records a wall-clock breakdown of its phases (and which
/// index-loading path it took) into [`OpenTimings`], surfaced via
/// [`PackfileStorage::open_timings`]; every `sync_all` records a per-phase
/// breakdown into [`SyncTimings`], surfaced via
/// [`PackfileStorage::sync_timings`]. These exist to decide whether startup
/// is dominated by deserialization (justifying mmap-ready persisted
/// indexes) or by validation I/O, and whether sync is dominated by the
/// checkpoint rewrite — before either format change is built. The cost is a
/// handful of `Instant::now()` deltas on the open/sync paths only, never
/// per record.
///
/// The phases measured:
/// - **shard directory discovery/open**: directory scan plus each pack
///   file's open and (for the writer) writer-lock take.
/// - **metadata load**: the collection-order and deleted-collections
///   sidecars.
/// - **checkpoint decode**: reading + decoding the persisted index
///   checkpoint from disk.
/// - **fingerprint**: recomputing the live set of `(pack_id, file_len)`
///   tuples and hashing them for validation against the checkpoint.
/// - **index materialization**: deserializing checkpoint index blobs, or
///   (on the fallback path) building per-collection indexes from a scan.
/// - **full scan**: the fallback path's packfile scan + torn-tail recovery
///   (zero on the checkpoint path).
pub struct PackfileStorage {
    shards: ShardPool,
    collections: RwLock<HashMap<[u8; 16], ArcSwap<RoomGeneration>>>,
    collection_order: RwLock<Vec<[u8; 16]>>,
    pinned: PinnedNodes,
    base_dir: PathBuf,
    swizzle: Option<SwizzleFn>,
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    /// Serializes the *publication* of a brand-new collection against a
    /// checkpoint rewrite (see `persist_index_checkpoint`). Creates take this
    /// lock shared for the whole put; the checkpoint holds it exclusive for
    /// the fingerprint→snapshot window, so a record whose bytes land on disk
    /// mid-rewrite can never be fingerprinted without its collection being
    /// snapshotted in the same checkpoint. Existing-collection puts never
    /// touch it (their `put_mutex` already gates them), so the shrink-latency
    /// property the epoch-handoff window exists to protect is untouched.
    collection_creation: parking_lot::RwLock<()>,
    /// In-memory cache of the `deleted.collections` file, seeded once at
    /// `open()` from disk. Guards both the set and the read-modify-write of
    /// the backing file, so a plain membership check (the common case: most
    /// collections were never deleted) never touches disk. Collection locks
    /// protect a collection's lifecycle, but distinct collections may be
    /// recreated or deleted concurrently, hence the separate lock here.
    deleted_collections: parking_lot::Mutex<HashSet<[u8; 16]>>,
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    repack_threshold_entries: AtomicU64,
    cache_capacity: usize,
    /// Total number of `repack_collection_reachable` calls across all collections.
    repack_count: AtomicU64,
    /// Total records kept (rewritten into the new generation) across all repacks.
    repack_kept_total: AtomicU64,
    /// Total records dropped (found unreachable) across all repacks.
    repack_dropped_total: AtomicU64,
    /// Per-collection repack counts, so a hot collection's churn is visible individually.
    repack_counts_by_collection: RwLock<HashMap<[u8; 16], u64>>,
    /// Per-collection incremental repack state. Each entry tracks the byte
    /// offset up to which each shard has been scanned for this collection and
    /// the deduplicated hash → (`shard_id`, offset) map from the last repack.
    /// On the next repack, only bytes after these offsets are scanned and
    /// merged into the existing map — turning O(n²) full rescan into
    /// O(n) total work across all repack calls.
    repack_incremental: RwLock<HashMap<[u8; 16], RepackIncrementalState>>,
    /// Persisted directory: which collections have live records in each shard,
    /// and how many. Maintained incrementally (a plain `put` just
    /// increments one counter; a full index rebuild/repack/initial scan
    /// replaces one collection's contribution wholesale via
    /// `LossyIndex::shard_counts`) rather than ever re-derived by
    /// scanning a shard file, which is what made `collections_referencing_shard`
    /// and a `collections`-style listing expensive before this existed.
    shard_collections: RwLock<HashMap<u64, HashMap<[u8; 16], u64>>>,
    /// Reverse index of `shard_collections`: which shards a given collection currently
    /// contributes a nonzero count to. Lets a collection's full-index-rebuild
    /// path (`replace_collection_shard_counts`) clear exactly the shard entries
    /// it used to occupy without scanning every shard in `shard_collections`.
    collection_shards: RwLock<HashMap<[u8; 16], HashSet<u64>>>,
    /// Wall-clock instant of the last `maybe_persist_shard_collections` flush,
    /// used to rate-limit that timer-driven path.
    last_shard_collections_flush: RwLock<Option<std::time::Instant>>,
    /// Wall-clock breakdown of the most recent `open` — which index-loading
    /// path was taken and how long each phase took — set at construction.
    last_open_timings: parking_lot::Mutex<Option<OpenTimings>>,
    /// Wall-clock breakdown of the most recent `sync_all`, by phase.
    last_sync_timings: parking_lot::Mutex<Option<SyncTimings>>,
    /// Gates the logical read-path counters (`get`/`get_many` below). Off by
    /// default so a store doing no reads-of-record pays one relaxed load per
    /// logical read at most, and the write/batch/sync counters are the only
    /// always-on instrumentation (each is a single batch-granular `fetch_add`
    /// on a write-locked or per-call path, never per-record on the hot read
    /// path). See [`Self::set_stats_enabled`].
    stats_enabled: AtomicBool,
    /// Number of times this class of store was assembled in this process — 1
    /// for a fresh open. Not reset by `reset_stats` (it counts stores, not work).
    open_count: AtomicU64,

    // Logical read-path counters. Only incremented while `stats_enabled` is
    // true; otherwise the read path touches none of these.
    /// Logical `get` calls (including cache hits and any collection's absence).
    get_calls: AtomicU64,
    /// Logical `get` calls that resolved to no record (unknown collection,
    /// empty index probe, or unresolved candidate).
    get_misses: AtomicU64,
    /// Logical `get_many` calls.
    get_many_calls: AtomicU64,
    /// Records requested across `get_many` calls.
    get_many_records: AtomicU64,
    /// `get_many` results that resolved to no record.
    get_many_misses: AtomicU64,

    // Always-on write-path counters (per-record `fetch_add` only on the
    // single-record `put`, whose hot cost is dominated by the write itself).
    /// Single-record `put` attempts.
    put_calls: AtomicU64,
    /// Bytes accepted by single-record `put` attempts.
    put_bytes: AtomicU64,
    /// `put_many` calls (batches).
    put_many_calls: AtomicU64,
    /// Records written across `put_many` calls.
    put_many_records: AtomicU64,
    /// Bytes written across `put_many` calls.
    put_many_bytes: AtomicU64,
    /// `put_many` calls that started on the owned, no-clone fast path.
    put_many_fast_path_calls: AtomicU64,
    /// `put_many` calls that had to materialize an owned index up front
    /// (mmap-backed first write, or a brand-new collection).
    put_many_clone_path_calls: AtomicU64,
    /// Cumulative nanoseconds spent materializing or growing an owned index in
    /// `put_many` (batch-granular; drives the steady-append scaler's
    /// resume-from-checkpoint cost).
    index_clone_time_ns: AtomicU64,
    /// Index grow/rebuild events in `put_many` (each invalidates the delta log).
    index_grow_count: AtomicU64,
    /// Fallback `rebuild_index` calls (full pack scan).
    index_rebuild_count: AtomicU64,
    /// `invalidate_delta_log` calls — structural changes that force the next
    /// dirty sync onto a full checkpoint rewrite + log re-base.
    delta_invalidations: AtomicU64,
    /// `sync`/`sync_all` calls.
    sync_calls: AtomicU64,
    /// Syncs that rewrote the index checkpoint in full.
    checkpoint_writes: AtomicU64,
    /// Syncs that appended the incremental delta log instead.
    delta_appends: AtomicU64,
    /// Whether any collection data changed since the last `index.checkpoint`
    /// write. Set by every generation swap (`put`/`put_many`/`repack`/`refresh`);
    /// cleared only by a successful [`Self::persist_index_checkpoint`], so a
    /// failed write is retried on the next sync. Lets `sync()`/`sync_all()`
    /// skip rewriting the checkpoint when nothing has changed since the last
    /// one — a steady-state writer that syncs between writes pays no checkpoint
    /// cost, while a crash left a stale checkpoint is still always resolved by
    /// the fingerprint → rescan fallback.
    index_checkpoint_dirty: AtomicBool,
    /// Session state for the incremental index delta log: the base checkpoint
    /// fingerprint + generations the log continues, and the frames accumulated
    /// since the last persist. See [`DeltaLogState`].
    delta_state: parking_lot::Mutex<DeltaLogState>,
}

/// Per-collection state for incremental repack.
#[derive(Clone)]
struct RepackIncrementalState {
    /// Per-shard byte offset: next scan starts here (file length at end
    /// of last scan). Shards not in this map haven't been scanned yet.
    // A slot can be retired and reused for a different pack.  Keep the
    // pack_id beside the cursor so an offset from the old incarnation is
    // never applied to the replacement file.
    scan_offsets: HashMap<u16, (u64, u64)>,
    /// The deduplicated hash → (`shard_id`, offset) map from the last repack.
    /// The next repack merges newly-scanned entries into this map.
    live_map: HashMap<[u8; 16], (u16, u64)>,
    /// Cached edge lists for every node that survived the last repack.
    /// Lets the BFS and full-scan adjacency walks skip re-reading disk for
    /// nodes that were already seen — turning O(total) disk reads per repack
    /// call into O(delta) reads for only nodes added since the last repack.
    adjacency: HashMap<[u8; 16], Vec<[u8; 16]>>,
    /// The live-roots snapshot from the last repack. Used to identify which
    /// roots are genuinely new so the incremental BFS can seed only those,
    /// rather than re-walking the entire reachable set from scratch.
    prev_roots: Vec<[u8; 16]>,
}

const DEFAULT_REPACK_THRESHOLD_ENTRIES: u64 = 2048;
const DEFAULT_CACHE_CAPACITY: usize = 100_000;

/// Minimum `LossyIndex` capacity for a brand-new collection's first
/// insert (`put` and `put_many`), in slots (8 bytes each).
///
/// Not the index's own 16-slot hard floor: at the ~75% load factor
/// `grow()` targets, 16 slots admits only ~12 entries before the next
/// insert forces a grow -- and every `grow()` invalidates the delta log
/// (see its doc comment), forcing the next sync to do a full checkpoint
/// rewrite instead of a cheap delta append. A workload that lands entries
/// one at a time into a new collection (e.g. events arriving
/// individually) would trade checkpoint bloat for far more frequent full
/// rewrites. 64 slots (512 B) covers a "handful of records" collection
/// without ever growing, while remaining ~64x smaller than the old flat
/// 4096-slot (32 KiB) floor this replaced.
const NEW_COLLECTION_INDEX_FLOOR: usize = 64;

impl PackfileStorage {
    /// Open a packfile storage with default settings.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage with explicit compression and checksum
    /// policies — the advanced durability/performance entry point. See
    /// [`crate::packfile::ChecksumPolicy`] for what giving up read-time
    /// verification costs.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_policies(
        base_dir: PathBuf,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            compress,
            checksum_policy,
        )
    }

    /// Open a writable packfile storage with explicit cache, compression, and
    /// checksum policies.
    ///
    /// `cache_capacity` applies independently to each collection; pass zero
    /// to disable decoded-node caching entirely.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_cache_and_policies(
        base_dir: PathBuf,
        cache_capacity: usize,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            None,
            true,
            None,
            compress,
            checksum_policy,
        )
    }

    /// Open a packfile storage with `compress` controlling whether records
    /// are zstd-attempted on write (see
    /// [`crate::packfile::write_record_with_options`]) — pass `false` for a
    /// pool whose payloads (e.g. HAMT nodes/roots) never benefit, to skip
    /// paying the compressor's cost on every put.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_compression(
        base_dir: PathBuf,
        compress: bool,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            compress,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage that rotates shards at `max_shard_bytes`
    /// instead of the default `~256 MB` ceiling. Intended for benchmarks
    /// and tests that want many small packs without writing gigabytes of
    /// data to trigger rotation — see
    /// [`crate::shard::ShardPool::open_with_max_shard_bytes`].
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or
    /// read, or if `max_shard_bytes` is zero or exceeds
    /// [`crate::shard::MAX_SHARD_BYTES`].
    pub fn open_with_max_shard_bytes(
        base_dir: PathBuf,
        max_shard_bytes: u64,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            Some(max_shard_bytes),
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage as a read-only observer, coexisting with a
    /// concurrent writer on the same directory (see
    /// `ShardPool::open_read_only`'s doc for the locking contract).
    ///
    /// Intended for inspection tooling (e.g. a `shards`/`collections`/`info` CLI
    /// command) that needs to run alongside a live writer process without
    /// either racing its on-disk state or being mistaken for a second
    /// writer and rejected.
    ///
    /// # Errors
    /// Returns `io::Error` if the directory can't be read, has no shards
    /// yet, or a writer already holds the exclusive lock.
    pub fn open_read_only(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            false,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage as a read-only observer with an explicit
    /// checksum policy, gating per-lookup CRC verification in `read_at`.
    /// See [`crate::packfile::ChecksumPolicy`]. Matches a writer opened with
    /// the same policy.
    ///
    /// # Errors
    /// Same as [`Self::open_read_only`].
    pub fn open_read_only_with_policies(
        base_dir: PathBuf,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            false,
            None,
            true,
            checksum_policy,
        )
    }

    /// Open a read-only observer with explicit decoded-node cache and
    /// checksum policies.
    ///
    /// `cache_capacity` applies independently to each collection; pass zero
    /// to disable decoded-node caching entirely.
    ///
    /// # Errors
    /// Same as [`Self::open_read_only`].
    pub fn open_read_only_with_cache_and_policies(
        base_dir: PathBuf,
        cache_capacity: usize,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            None,
            false,
            None,
            true,
            checksum_policy,
        )
    }

    /// Open a packfile storage with a custom per-collection cache capacity.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_cache(
        base_dir: PathBuf,
        cache_capacity: usize,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            None,
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage with a swizzle callback for in-cache pointer resolution.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_swizzle(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: SwizzleFn,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            Some(swizzle),
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn open_with_options(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: Option<SwizzleFn>,
        writable: bool,
        max_shard_bytes: Option<u64>,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        fs::create_dir_all(&base_dir)?;

        let started = std::time::Instant::now();
        let mut timings = OpenTimings::default();

        let shard_open_started = std::time::Instant::now();
        let shards = if writable {
            match max_shard_bytes {
                // Custom rotation thresholds are only used by benchmarks/
                // tests today, none of which also need to disable
                // compression, so this combination just keeps the default.
                Some(max_shard_bytes) => ShardPool::open_with_policies(
                    base_dir.clone(),
                    compress,
                    checksum_policy,
                    Some(max_shard_bytes),
                )?,
                None => ShardPool::open_with_policies(
                    base_dir.clone(),
                    compress,
                    checksum_policy,
                    None,
                )?,
            }
        } else {
            ShardPool::open_read_only_with_policies(base_dir.clone(), checksum_policy)?
        };
        timings.shard_open = shard_open_started.elapsed();

        // Phase 1: accumulate all records per collection across every shard so
        // the index can be sized once for the true total.
        let mut collection_entries: HashMap<[u8; 16], Vec<ShardRecord>> = HashMap::new();
        let metadata_started = std::time::Instant::now();
        // Seed logical insertion order before scanning. A newly written collection
        // cannot appear in the sidecar until the next flush, so append it at
        // first physical sight below. Legacy v1 sidecars yield no order and
        // therefore use the same first-seen fallback for every collection.
        let mut collection_order = persisted_collection_order(&base_dir).unwrap_or_default();
        let mut known_collections: HashSet<[u8; 16]> = collection_order.iter().copied().collect();

        // Collect open shard info (slot, pack_id, path, length) before the scan
        // loop so we don't hold the shards read-lock across the I/O-heavy scan.
        // The lengths feed the checkpoint fingerprint (see `index::checkpoint`).
        let open_shards: Vec<(u16, u64, PathBuf, u64)> = shards
            .all_shards()
            .into_iter()
            .map(|(id, shard)| (id, shard.pack_id, shard.path.clone(), shard.file_len()))
            .collect();

        // A checkpoint from the previous session's last sync lets us skip the
        // rescan and index rebuild entirely. It is only trusted when every
        // pack on disk still matches the fingerprint it was written against.
        let deleted_collections = Self::load_deleted_collections(&base_dir);
        timings.metadata_load = metadata_started.elapsed();
        if let Some((scan_out, collection_order, delta_state)) = Self::checkpoint_scan_out(
            &base_dir,
            cache_capacity,
            &shards,
            &open_shards,
            &deleted_collections,
            writable,
            &mut timings,
        ) {
            timings.path = OpenPath::Checkpoint;
            let store = Self::assemble(
                shards,
                scan_out,
                collection_order,
                deleted_collections,
                base_dir,
                swizzle,
                cache_capacity,
                delta_state,
            );
            timings.total = started.elapsed();
            *store.last_open_timings.lock() = Some(timings);
            return Ok(store);
        }
        timings.path = OpenPath::FullScan;
        // The checkpoint/log gates rejected the persisted index state; never
        // carry forward a stale log into the rescanned session — a writable
        // open can delete it outright (read-only opens leave the file alone
        // but will re-rescan, since the damaged bases can never trust it).
        if writable {
            Self::sweep_orphan_delta_logs(&base_dir, None);
        }

        let scan_started = std::time::Instant::now();
        for (shard_id, shard_pack_id, path, _file_len) in open_shards {
            // A read-only open must never touch the file at all — the
            // truncating recovery scan below is only safe when we're the
            // sole writer (guaranteed by the writer lock); a read-only
            // opener can run concurrently with an active writer (no lock
            // taken at all — see ShardPool::open_read_only), so it must
            // use the non-mutating scan_packfile, which just stops
            // cleanly at a torn tail (indistinguishable from a writer's
            // in-flight append) instead of truncating it away.
            let entries = if writable {
                // Recovery can safely truncate only a torn final frame. Any
                // other failure (CRC mismatch, invalid framing, permissions)
                // means this pack cannot be indexed faithfully; fail open
                // rather than publish a store that silently omitted it.
                packfile::scan_and_recover_packfile(&path).map_err(|error| {
                    std::io::Error::new(
                        error.kind(),
                        format!("recovery scan failed for pack {}: {error}", path.display()),
                    )
                })?
            } else {
                match packfile::scan_packfile(&path) {
                    Ok(e) => e,
                    Err(scan_err) => {
                        eprintln!(
                            "warning: read-only scan of shard {shard_id:02x} skipped: {scan_err}"
                        );
                        continue;
                    }
                }
            };
            for (collection_id, hash, offset) in entries {
                if known_collections.insert(collection_id) {
                    collection_order.push(collection_id);
                }
                collection_entries.entry(collection_id).or_default().push((
                    shard_id,
                    hash,
                    offset,
                    shard_pack_id,
                ));
            }
        }
        collection_order.retain(|collection_id| collection_entries.contains_key(collection_id));
        timings.full_scan = scan_started.elapsed();

        let mut scan_out = RoomScanOutput::default();

        // Phase 2: build per-collection indexes sized to the true total.
        let materialization_started = std::time::Instant::now();
        for collection_id in &collection_order {
            if deleted_collections.contains(collection_id) {
                continue;
            }
            Self::init_collection_from_scan(
                collection_id,
                &collection_entries[collection_id],
                &shards,
                cache_capacity,
                &mut scan_out,
            )?;
        }
        timings.index_materialization = materialization_started.elapsed();

        let store = Self::assemble(
            shards,
            scan_out,
            collection_order,
            deleted_collections,
            base_dir,
            swizzle,
            cache_capacity,
            // A full rescan re-cooks every index from scratch, so no delta log
            // can continue anything the checkpoint recorded; force the next
            // dirty sync into a full checkpoint rewrite that re-bases the log.
            DeltaLogState {
                invalid: true,
                ..DeltaLogState::default()
            },
        );
        timings.total = started.elapsed();
        *store.last_open_timings.lock() = Some(timings);
        Ok(store)
    }

    /// Builds one collection's index, cache, and shard-directory contribution
    /// from its scanned records, and inserts all three into `out` — the
    /// per-collection body of `open_with_options`'s phase 2, factored out to
    /// keep that function under the line-count lint rather than
    /// suppressing it.
    fn init_collection_from_scan(
        collection_id: &[u8; 16],
        records: &[ShardRecord],
        shards: &ShardPool,
        cache_capacity: usize,
        out: &mut RoomScanOutput,
    ) -> Result<(), StorageError> {
        // Seed each collection's home shard from the scan: the shard of its
        // last-scanned record is a best-effort proxy for "most recent"
        // (shards are scanned in ascending ID order, and IDs generally
        // increase over time via rotation) — not exact chronology across
        // shards, but enough to keep a resumed collection's writes landing near
        // its existing data instead of restarting at whatever the pool's
        // active shard happens to be.
        if let Some(&(last_shard_id, _, _, _)) = records.last() {
            shards.set_collection_home(collection_id, last_shard_id);
        }
        let index = LossyIndex::new(
            records
                .len()
                .saturating_mul(2)
                .max(NEW_COLLECTION_INDEX_FLOOR),
        );
        for (shard_id, hash, offset, _pack_id) in records {
            check_index_offset(*shard_id, hash, *offset)?;
            let _ = index.insert(hash, *shard_id, *offset);
        }
        let counts = index.shard_counts();

        // Build slot→pack_id lookup from the records for this collection.
        let slot_to_pack_id: HashMap<u16, u64> = records
            .iter()
            .map(|(slot, _, _, pack_id)| (*slot, *pack_id))
            .collect();

        for (&shard_id, &count) in &counts {
            let pack_id = slot_to_pack_id
                .get(&shard_id)
                .copied()
                .unwrap_or(u64::from(shard_id));
            out.shard_collections
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, count);
            out.collection_shards
                .entry(*collection_id)
                .or_default()
                .insert(pack_id);
        }
        out.collections.insert(
            *collection_id,
            ArcSwap::from_pointee(RoomGeneration {
                index,
                cache: Arc::new(NodeCache::new(cache_capacity)),
                generation: 1,
            }),
        );
        Ok(())
    }

    /// Assemble a fully constructed store from the per-collection state both
    /// the rescan path and the checkpoint fast path produce, so the two share
    /// one field-for-field constructor.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        shards: ShardPool,
        scan_out: RoomScanOutput,
        collection_order: Vec<[u8; 16]>,
        deleted_collections: HashSet<[u8; 16]>,
        base_dir: PathBuf,
        swizzle: Option<SwizzleFn>,
        cache_capacity: usize,
        delta_state: DeltaLogState,
    ) -> Self {
        Self {
            shards,
            collections: RwLock::new(scan_out.collections),
            collection_order: RwLock::new(collection_order),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            collection_creation: parking_lot::RwLock::new(()),
            deleted_collections: parking_lot::Mutex::new(deleted_collections),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_collection: RwLock::new(HashMap::new()),
            repack_incremental: RwLock::new(HashMap::new()),
            shard_collections: RwLock::new(scan_out.shard_collections),
            collection_shards: RwLock::new(scan_out.collection_shards),
            last_shard_collections_flush: RwLock::new(None),
            index_checkpoint_dirty: AtomicBool::new(false),
            last_open_timings: parking_lot::Mutex::new(None),
            last_sync_timings: parking_lot::Mutex::new(None),
            stats_enabled: AtomicBool::new(false),
            open_count: AtomicU64::new(1),
            get_calls: AtomicU64::new(0),
            get_misses: AtomicU64::new(0),
            get_many_calls: AtomicU64::new(0),
            get_many_records: AtomicU64::new(0),
            get_many_misses: AtomicU64::new(0),
            put_calls: AtomicU64::new(0),
            put_bytes: AtomicU64::new(0),
            put_many_calls: AtomicU64::new(0),
            put_many_records: AtomicU64::new(0),
            put_many_bytes: AtomicU64::new(0),
            put_many_fast_path_calls: AtomicU64::new(0),
            put_many_clone_path_calls: AtomicU64::new(0),
            index_clone_time_ns: AtomicU64::new(0),
            index_grow_count: AtomicU64::new(0),
            index_rebuild_count: AtomicU64::new(0),
            delta_invalidations: AtomicU64::new(0),
            sync_calls: AtomicU64::new(0),
            checkpoint_writes: AtomicU64::new(0),
            delta_appends: AtomicU64::new(0),
            delta_state: parking_lot::Mutex::new(delta_state),
        }
    }

    /// Path of this store's persisted-index checkpoint.
    fn index_checkpoint_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join(crate::index::checkpoint::INDEX_CHECKPOINT_FILE)
    }

    /// Path of one *epoch* of this store's incremental index delta log.
    ///
    /// Named by the checkpoint fingerprint it continues (`index.delta.<hex
    /// fingerprint>`) rather than a single fixed name: the epoch-handoff
    /// protocol (see `persist_index_checkpoint`) needs the about-to-be-retired
    /// epoch (D0, continuing the old checkpoint) and the freshly rotated one
    /// (D1, continuing the new one) to coexist on disk as distinct files
    /// while the new checkpoint is serialized and fsynced unlocked. Naming by
    /// fingerprint also *is* the reopen gate: a loader only ever looks at the
    /// epoch whose name matches the checkpoint's own `pack_fingerprint`, so
    /// a leftover D0 (crash before retirement) or an orphaned D1 (crash
    /// before the checkpoint that would have validated it) are both
    /// automatically ignored without any extra bookkeeping.
    fn delta_path(base_dir: &std::path::Path, fingerprint: u64) -> PathBuf {
        base_dir.join(format!("{INDEX_DELTA_FILE}.{fingerprint:016x}"))
    }

    /// Remove every on-disk delta-log epoch file except `keep` (pass `None`
    /// to remove all of them, e.g. after a full rescan with no checkpoint to
    /// continue). Best-effort and silent on failure — an orphaned epoch file
    /// left behind is always safe: it is either identical to the kept one
    /// (redundant) or has a fingerprint that can never match the current
    /// checkpoint again (inert). Also opportunistically removes the single
    /// fixed-name log from before epoch-named logs existed; it is never read
    /// by this version, so leaving it around would just be clutter.
    fn sweep_orphan_delta_logs(base_dir: &std::path::Path, keep: Option<&std::path::Path>) {
        let _ = fs::remove_file(base_dir.join(INDEX_DELTA_FILE));
        let Ok(read_dir) = fs::read_dir(base_dir) else {
            return;
        };
        let prefix = format!("{INDEX_DELTA_FILE}.");
        for entry in read_dir.flatten() {
            let path = entry.path();
            if keep.is_some_and(|keep| keep == path) {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.starts_with(&prefix) {
                let _ = fs::remove_file(&path);
            }
        }
    }

    /// Attempt to recover the per-shard/collection bookkeeping from the
    /// inspection sidecar, gated on the checkpoint's own pack fingerprint.
    ///
    /// Returns `BookkeepingSource::Sidecar` plus the slotted counts (for home
    /// shard selection) and pack-keyed entries (for the shard directory) only
    /// when the sidecar is gated to this exact pack set and describes every
    /// non-deleted collection completely — per-pack counts must sum to the
    /// checkpoint's own slot count. Stale, corrupt, or absent sidecars (or
    /// one referencing packs that aren't live open shards) fall through to
    /// `BookkeepingSource::SlotScan` with empty maps: correctness never
    /// depends on the sidecar, it only accelerates.
    #[allow(clippy::type_complexity)]
    fn gated_sidecar_bookkeeping(
        base_dir: &std::path::Path,
        local_fingerprint: u64,
        current_len: &HashMap<[u8; 16], u32>,
        open_shards: &[(u16, u64, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
    ) -> (BookkeepingSource, HashMap<[u8; 16], HashMap<u16, u64>>) {
        let mut counts: HashMap<[u8; 16], HashMap<u16, u64>> = HashMap::new();
        let Some(directory) = read_persisted_shard_collections(base_dir) else {
            return (BookkeepingSource::SlotScan, counts);
        };
        if directory.fingerprint != local_fingerprint {
            return (BookkeepingSource::SlotScan, counts);
        }
        let pack_to_slot: HashMap<u64, u16> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*pack_id, *slot))
            .collect();
        for record in &directory.records {
            if deleted_collections.contains(&record.collection_id) {
                continue;
            }
            let Some(slot) = pack_to_slot.get(&record.pack_id) else {
                return (BookkeepingSource::SlotScan, counts);
            };
            counts
                .entry(record.collection_id)
                .or_default()
                .insert(*slot, record.count);
        }
        for (collection_id, len) in current_len {
            if deleted_collections.contains(collection_id) {
                continue;
            }
            let Some(slot_counts) = counts.get(collection_id) else {
                return (BookkeepingSource::SlotScan, counts);
            };
            if slot_counts.values().copied().sum::<u64>() != u64::from(*len) {
                return (BookkeepingSource::SlotScan, counts);
            }
        }
        (BookkeepingSource::Sidecar, counts)
    }

    /// Fast-path index loading for `open_with_options`: build the
    /// per-collection state directly from the persisted-index checkpoint
    /// instead of rescanning every packfile.
    ///
    /// Returns `None` — falling through to the full rescan — for a missing,
    /// malformed, or stale checkpoint, one that doesn't describe the pack set
    /// currently on disk, or a delta log that can't be replayed onto the
    /// checkpoint. The fast path deliberately skips the torn-tail recovery
    /// scan, which is safe: a fingerprint match means the packs are the exact
    /// state the previous session's last sync fsynced, so there can be no torn
    /// tail to recover.
    ///
    /// On success, also returns the fresh [`DeltaLogState`] for the session: the
    /// delta log's base fingerprint (the checkpoint's) and each collection's
    /// checkpoint generation, with `invalid` set only if a structural change
    /// (e.g. a grow) forced replay to bail into a rescan — in which case this
    /// function returns `None`, and the rescan path constructs the
    /// invalidated state instead.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn checkpoint_scan_out(
        base_dir: &std::path::Path,
        cache_capacity: usize,
        shards: &ShardPool,
        open_shards: &[(u16, u64, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
        writable: bool,
        timings: &mut OpenTimings,
    ) -> Option<(RoomScanOutput, Vec<[u8; 16]>, DeltaLogState)> {
        let decode_started = std::time::Instant::now();
        let checkpoint =
            crate::index::checkpoint::read_checkpoint(&Self::index_checkpoint_path(base_dir));
        timings.checkpoint_decode = decode_started.elapsed();
        let checkpoint = checkpoint?;
        let fingerprint_started = std::time::Instant::now();
        let packs: Vec<(u64, u64)> = open_shards
            .iter()
            .map(|(_, pack_id, _, file_len)| (*pack_id, *file_len))
            .collect();
        let local_fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);
        timings.fingerprint = fingerprint_started.elapsed();

        // Generations at the checkpoint, per collection — used both to seed the
        // new session's `DeltaLogState` and to gate whether any delta frames may be
        // applied (a frame's stamp must equal the checkpoint generation it
        // claims to continue, which also rules out the never-recorded ancestors
        // of collections created or structurally changed since the checkpoint).
        let ckpt_generations: HashMap<[u8; 16], u64> = checkpoint
            .collections
            .iter()
            .filter(|loaded| !deleted_collections.contains(&loaded.collection_id))
            .map(|loaded| (loaded.collection_id, loaded.generation))
            .collect();

        let delta_path = Self::delta_path(base_dir, checkpoint.fingerprint);
        // Only the epoch named for this exact checkpoint can ever be trusted
        // (see `delta_path`); anything else on disk — a stale predecessor
        // epoch left by a crash between checkpoint rename and retirement, or
        // an orphaned successor epoch from a rewrite that never completed —
        // is inert by construction, but sweep it now on a writable open so
        // it doesn't accumulate across sessions.
        if writable {
            Self::sweep_orphan_delta_logs(base_dir, Some(&delta_path));
        }
        // A delta log can only continue a checkpoint whose packs have advanced
        // (every append accompanies a flush that grows a pack, changing the
        // fingerprint). When the packs still match the checkpoint exactly,
        // any leftover log is stale — likely a crashed full rewrite that had
        // already committed the new checkpoint but not yet deleted its
        // predecessor log. Never replayed; removed on a writable open.
        let replay_needed = local_fingerprint != checkpoint.fingerprint;
        if !replay_needed && writable {
            let _ = std::fs::remove_file(&delta_path);
        }
        // The new session continues whatever log is genuinely on disk — its
        // committed byte length must be inherited, not zeroed: a later append
        // decides whether to write a fresh base header from `log_bytes == 0`,
        // and re-headering an existing log would corrupt the batch framing the
        // next reader depends on. Zero only when the open removed the file.
        let mut log_bytes_on_disk = 0u64;

        // Gate the log (case B only): it must start at this checkpoint and end
        // at the exact pack set now on disk, and every committed frame must
        // target a checkpointed collection at its recorded checkpoint
        // generation. Any failure rejects the log wholesale and falls through
        // to the rescan — never a partial replay.
        let mut replay_frames: Vec<DeltaFrame> = Vec::new();
        if replay_needed {
            let delta_started = std::time::Instant::now();
            let log = delta::read_delta_log(&delta_path);
            let trusted = log.as_ref().is_some_and(|log| {
                log.base_fingerprint == checkpoint.fingerprint
                    && log.tail_fingerprint == local_fingerprint
                    && log.frames.iter().all(|frame| {
                        ckpt_generations.get(&frame.collection_id) == Some(&frame.generation)
                    })
            });
            if trusted {
                // The byte frontier travels with the decode (no re-stat):
                // `file_len` is where the last committed trailer ends, which is
                // exactly where the next append must continue.
                let trusted_log = log.expect("trusted implies a decoded log");
                log_bytes_on_disk = trusted_log.file_len;
                replay_frames = trusted_log.frames;
            } else {
                timings.delta_replay = delta_started.elapsed();
                if writable {
                    let _ = std::fs::remove_file(&delta_path);
                }
                return None;
            }
            timings.delta_replay = delta_started.elapsed();
        }

        // Group the (gated) frames by target collection once, so the
        // materialization loop below applies each collection's frames by hash
        // lookup rather than re-scanning the whole frame list per collection.
        let mut frames_by_collection: HashMap<[u8; 16], Vec<DeltaFrame>> = HashMap::new();
        for frame in &replay_frames {
            frames_by_collection
                .entry(frame.collection_id)
                .or_default()
                .push(*frame);
        }

        let slot_to_pack_id: HashMap<u16, u64> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*slot, *pack_id))
            .collect();

        // The inspection sidecar is the same reduced per-(pack, collection)
        // bookkeeping the loop below would recover by walking every slot of
        // every collection index. Accept it only when it is gated to the pack
        // set now on disk and describes every non-deleted collection completely
        // (see `gated_sidecar_bookkeeping`); any mismatch falls through to the
        // faithful slot walk, because the sidecar is an acceleration and
        // correctness never depends on it.
        //
        // Gating needs each collection's post-replay length (the checkpoint
        // occupancy plus whatever fresh-slot writes the replayed frames add),
        // so the sidecar check runs after materialization below.
        let mut scan_out = RoomScanOutput::default();
        let mut collection_order = Vec::with_capacity(checkpoint.collections.len());
        let mut current_len: HashMap<[u8; 16], u32> =
            HashMap::with_capacity(checkpoint.collections.len());
        let materialization_started = std::time::Instant::now();
        for loaded in &checkpoint.collections {
            if deleted_collections.contains(&loaded.collection_id) {
                // The checkpoint may predate the deletion marker; the
                // logical-delete set is authoritative.
                continue;
            }
            // The checkpoint reader has already validated this range. Keep it
            // mmap-backed through the read-only fast path; the first writer
            // copy-on-writes it into the normal atomic slot array.
            let mmap_index = LossyIndex::from_mmap_slots(
                Arc::clone(&checkpoint.mmap),
                loaded.slots_offset,
                loaded.capacity,
                loaded.slot_count,
            );
            // Collections with replayed frames must be materialized (owned) so
            // the frames can be applied on top of the raw checkpoint slots;
            // everything else stays an O(1) mmap view.
            let index = if let Some(frames) = frames_by_collection.get(&loaded.collection_id) {
                let owned = mmap_index.clone();
                if owned.replay_frames(frames).is_err() {
                    // Structurally inconsistent with the checkpoint despite
                    // passing the fingerprint/generation gates (e.g. a frame
                    // bucket past this checkpoint's capacity). It can't be
                    // replayed; fall through to the rescan.
                    if writable {
                        let _ = std::fs::remove_file(&delta_path);
                    }
                    return None;
                }
                owned
            } else {
                mmap_index
            };
            current_len.insert(loaded.collection_id, u32::try_from(index.len()).ok()?);
            scan_out.collections.insert(
                loaded.collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::new(cache_capacity)),
                    generation: loaded.generation,
                }),
            );
            collection_order.push(loaded.collection_id);
        }
        timings.index_materialization = materialization_started.elapsed();

        let (bookkeeping_source, sidecar_counts) = Self::gated_sidecar_bookkeeping(
            base_dir,
            local_fingerprint,
            &current_len,
            open_shards,
            deleted_collections,
        );
        timings.bookkeeping_source = bookkeeping_source;

        // Bookkeeping (home-shard seeding + shard directory) from the sidecar
        // where that passed its gates, otherwise a slot walk of the — possibly
        // replayed — live indexes.
        let counts_lookup: HashMap<[u8; 16], HashMap<u16, u64>> = match bookkeeping_source {
            BookkeepingSource::Sidecar => sidecar_counts,
            BookkeepingSource::SlotScan => {
                let mut out = HashMap::with_capacity(collection_order.len());
                for collection_id in &collection_order {
                    if let Some(gen) = scan_out.collections.get(collection_id) {
                        out.insert(*collection_id, gen.load().index.shard_counts());
                    }
                }
                out
            }
        };
        for collection_id in &collection_order {
            let counts = &counts_lookup[collection_id];
            // Seed the home shard from the highest shard a slot references —
            // the nearest proxy for the scan path's "shard of the last
            // record", since slots were appended in shard-id order.
            if let Some(&home_shard) = counts.keys().max() {
                shards.set_collection_home(collection_id, home_shard);
            }
            for (&shard_id, &count) in counts {
                let pack_id = slot_to_pack_id
                    .get(&shard_id)
                    .copied()
                    .unwrap_or(u64::from(shard_id));
                scan_out
                    .shard_collections
                    .entry(pack_id)
                    .or_default()
                    .insert(*collection_id, count);
                scan_out
                    .collection_shards
                    .entry(*collection_id)
                    .or_default()
                    .insert(pack_id);
            }
        }
        // `SlotScan` derives counts straight from the live indexes; the
        // sidecar's `counts` are consumed above either way.

        let delta_state = DeltaLogState {
            base_fingerprint: Some(checkpoint.fingerprint),
            base_generations: ckpt_generations,
            log_bytes: log_bytes_on_disk,
            invalid: false,
            ..DeltaLogState::default()
        };

        Some((scan_out, collection_order, delta_state))
    }

    fn generation(&self, collection_id: &[u8; 16]) -> Option<arc_swap::Guard<Arc<RoomGeneration>>> {
        self.collections
            .read()
            .get(collection_id)
            .map(arc_swap::ArcSwapAny::load)
    }

    /// Collection IDs currently known to this engine, sorted for deterministic output.
    pub fn collection_ids(&self) -> Vec<[u8; 16]> {
        let mut ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Collection IDs whose live index currently references at least one record
    /// physically stored in `shard_id`.
    ///
    /// Shards are shared, so this is normally more than one collection; it's the
    /// set `repack_shard` needs to touch before that shard can retire.
    ///
    /// Sweeps the shard's own file content (bounded by `MAX_SHARD_BYTES`,
    /// not by how many unrelated collections happen to be loaded) rather than
    /// scanning every collection's index to ask "does this touch shard N" — a
    /// candidate set from the raw scan is then filtered against each
    /// candidate collection's *current* live index, since the scan alone can't
    /// tell a still-live reference from a collection that already repacked past
    /// this shard (its old bytes just haven't been overwritten — they
    /// never are, shards are append-only).
    ///
    /// # Errors
    /// Returns `StorageError` if the shard's file can't be read, or if
    /// `shard_id` doesn't correspond to a currently-open shard.
    pub fn collections_referencing_shard(
        &self,
        shard_id: u16,
    ) -> Result<Vec<[u8; 16]>, StorageError> {
        let shard = self.shards.get_shard(shard_id).ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no open shard with id {shard_id}"),
            ))
        })?;
        // O(1) against the incrementally-maintained shard→collection directory
        // instead of scanning the shard file — this used to be the
        // dominant cost of shard retirement/evacuation-style operations
        // on a large shard.
        Ok(self
            .shard_collections
            .read()
            .get(&shard.pack_id)
            .map(|collections| collections.keys().copied().collect())
            .unwrap_or_default())
    }

    /// Repack every collection that still references `shard_id`.
    ///
    /// A shard only ever retires once *every* collection referencing it has
    /// repacked past its data there (see `retire_empty_shards`) — there's
    /// no per-shard compaction primitive, since reachability (what's
    /// live vs. garbage) is inherently a per-collection concept, not a shard
    /// one. This is the targeted way to reclaim one specific shard: find
    /// every collection still pinning it live and repack each of them, instead
    /// of waiting for each collection to independently cross its own repack
    /// threshold. Live records get moved to whatever's currently that
    /// collection's home shard (`ShardPool::collection_home`) — not necessarily the
    /// pool's single "main" shard, since homes are per-collection, not global.
    ///
    /// Returns `(collection_id, kept, dropped)` for each collection repacked, in the
    /// order `collections_referencing_shard` returned them. Does not itself
    /// guarantee the shard retires — a collection with `set_live_roots` never
    /// called preserves everything (nothing is provably garbage) and its
    /// repack is a no-op for this purpose.
    ///
    /// # Errors
    /// Returns `StorageError` if any collection's repack fails; already-repacked
    /// collections in this call are not rolled back.
    pub fn repack_shard(
        &self,
        shard_id: u16,
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let collections = self.collections_referencing_shard(shard_id)?;
        let mut results = Vec::with_capacity(collections.len());
        for collection_id in collections {
            let (kept, dropped) = self
                .repack_collection_reachable(&collection_id, |hash, data| {
                    extract_edges(hash, data)
                })?;
            results.push((collection_id, kept, dropped));
        }
        Ok(results)
    }

    /// A collection's `(entry count, memory usage in bytes, index capacity)`,
    /// if the collection exists. `entry count / capacity` is the index's
    /// load factor.
    pub fn collection_index_info(&self, collection_id: &[u8; 16]) -> Option<(usize, usize, u32)> {
        self.collections.read().get(collection_id).map(|gen| {
            let g = gen.load();
            (g.index.len(), g.index.memory_usage(), g.index.capacity())
        })
    }

    /// `(collection_id, entry count, memory usage in bytes, index capacity)`
    /// for every known collection, sorted by collection ID, in a single pass
    /// over the collection map.
    pub fn collection_summaries(&self) -> Vec<CollectionSummary> {
        let collections = self.collections.read();
        self.collection_order
            .read()
            .iter()
            .filter_map(|id| {
                collections.get(id).map(|gen| {
                    let g = gen.load();
                    (
                        *id,
                        g.index.len(),
                        g.index.memory_usage(),
                        g.index.capacity(),
                    )
                })
            })
            .collect()
    }

    /// Set the live roots to preserve for a collection on its next repack.
    pub fn set_live_roots(&self, collection_id: &[u8; 16], roots: Vec<NodeId>) {
        self.live_roots.write().insert(*collection_id, roots);
    }

    /// Set the repack trigger threshold directly, in index entries.
    pub fn set_repack_threshold_entries(&self, entries: u64) {
        self.repack_threshold_entries
            .store(entries, std::sync::atomic::Ordering::Relaxed);
    }

    /// Returns `true` if a collection's index has reached the configured repack
    /// threshold.
    ///
    /// Repacking is entirely caller-driven — nothing in this engine polls
    /// this on its own. A background GC worker is expected to call this
    /// periodically and issue `repack_collection_reachable` itself; nothing in
    /// `mtxdb-core` currently does so.
    #[must_use]
    pub fn needs_repack(&self, collection_id: &[u8; 16]) -> bool {
        let count = self
            .collection_index_info(collection_id)
            .map_or(0, |(len, _, _)| len as u64);
        count
            >= self
                .repack_threshold_entries
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The collection-scoped node cache for `collection_id`, or a fresh empty one if the collection is unknown.
    pub fn collection_cache(&self, collection_id: &[u8; 16]) -> Arc<NodeCache> {
        self.generation(collection_id).map_or_else(
            || Arc::new(NodeCache::new(self.cache_capacity)),
            |g| g.cache.clone(),
        )
    }

    fn put_mutex(&self, collection_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.put_locks.lock();
        locks.entry(*collection_id).or_default().clone()
    }

    fn read_at(
        &self,
        shard: &Arc<Shard>,
        offset: u64,
        verify: bool,
    ) -> Result<Record, StorageError> {
        self.shards.read_at(shard, offset, verify)
    }

    /// The record's actual on-disk byte length (see
    /// [`ShardPool::record_disk_len_at`]) — the true disk-usage figure,
    /// unlike `Record::serialized_len()` which is an uncompressed upper
    /// bound.
    fn record_disk_len_at(shard: &Shard, offset: u64) -> Result<u64, StorageError> {
        ShardPool::record_disk_len_at(shard, offset)
    }

    fn scan_collection_records(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Vec<ScannedShard>, StorageError> {
        let mut scanned: Vec<ScannedShard> = Vec::new();
        for (shard_id, shard) in self.shards.all_shards() {
            match packfile::scan_packfile(&shard.path) {
                Ok(entries) => {
                    let collection_entries: Vec<([u8; 16], u64)> = entries
                        .into_iter()
                        .filter(|(rid, _, _)| rid == collection_id)
                        .map(|(_, hash, offset)| (hash, offset))
                        .collect();
                    if !collection_entries.is_empty() {
                        scanned.push((shard_id, collection_entries));
                    }
                }
                Err(e) => {
                    return Err(StorageError::Io(e));
                }
            }
        }
        Ok(scanned)
    }

    /// Scan every open shard once and build deduplicated record maps for the
    /// requested collections. This is the batch counterpart to `scan_collection_records`:
    /// a shard-compaction preflight must not reread the same packfile once per
    /// collection in its closure.
    fn scan_collection_record_maps(
        &self,
        collection_ids: &[[u8; 16]],
    ) -> Result<HashMap<[u8; 16], RepackRecordMap>, StorageError> {
        let wanted: HashSet<[u8; 16]> = collection_ids.iter().copied().collect();
        let mut maps: HashMap<[u8; 16], RepackRecordMap> = wanted
            .iter()
            .copied()
            .map(|collection_id| (collection_id, HashMap::new()))
            .collect();
        for (shard_id, shard) in self.shards.all_shards() {
            for (collection_id, hash, offset) in packfile::scan_packfile(&shard.path)? {
                if wanted.contains(&collection_id) {
                    maps.entry(collection_id)
                        .or_default()
                        .insert(hash, (shard_id, offset));
                }
            }
        }
        Ok(maps)
    }

    fn build_index(offsets: &[([u8; 16], u16, u64)]) -> Result<LossyIndex, StorageError> {
        let index = LossyIndex::new(
            offsets
                .len()
                .saturating_mul(2)
                .max(NEW_COLLECTION_INDEX_FLOOR),
        );
        for (hash, shard_id, offset) in offsets {
            check_index_offset(*shard_id, hash, *offset)?;
            let _ = index.insert(hash, *shard_id, *offset);
        }
        Ok(index)
    }

    /// Convert a slot-keyed `shard_counts` from `LossyIndex` to a pack_id-keyed
    /// `HashMap<u64, u64>` for use with `replace_collection_shard_counts`.
    fn slot_counts_to_pack_id_counts(&self, counts: &HashMap<u16, u64>) -> HashMap<u64, u64> {
        let shards = self.shards.all_shards();
        let slot_to_pack_id: HashMap<u16, u64> = shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        counts
            .iter()
            .map(|(&slot, &count)| {
                let pack_id = slot_to_pack_id
                    .get(&slot)
                    .copied()
                    .unwrap_or(u64::from(slot));
                (pack_id, count)
            })
            .collect()
    }

    fn deleted_collections_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("deleted.collections")
    }

    fn load_deleted_collections(base_dir: &std::path::Path) -> HashSet<[u8; 16]> {
        let path = Self::deleted_collections_path(base_dir);
        let Ok(bytes) = fs::read(&path) else {
            return HashSet::new();
        };
        bytes
            .chunks_exact(16)
            .map(|chunk| {
                let mut id = [0u8; 16];
                id.copy_from_slice(chunk);
                id
            })
            .collect()
    }

    fn persist_deleted_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_collections_path(&self.base_dir);
        let mut set = self.deleted_collections.lock();
        set.insert(*collection_id);
        let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
        fs::write(&path, &bytes).map_err(StorageError::Io)?;
        Ok(())
    }

    /// Fast path for `store_generation`'s new-collection case: most
    /// collections were never deleted, so check the in-memory cache before
    /// paying for a lock + no-op write. Avoids re-reading `deleted.collections`
    /// from disk on every first write to a new collection — previously this
    /// reloaded the whole file from disk unconditionally, which under a cold
    /// page cache turned a batch of first-writes across many collections into
    /// one serialized disk read per collection.
    fn is_deleted_collection(&self, collection_id: &[u8; 16]) -> bool {
        self.deleted_collections.lock().contains(collection_id)
    }

    fn clear_deleted_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_collections_path(&self.base_dir);
        let mut set = self.deleted_collections.lock();
        if set.remove(collection_id) {
            let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
            fs::write(&path, &bytes).map_err(StorageError::Io)?;
        }
        Ok(())
    }

    /// Records one new record landing in `shard_id` for `collection_id` — the
    /// cheap, O(1) path for a plain `put` that only appends, never moves
    /// or drops anything. Kept separate from
    /// [`Self::replace_collection_shard_counts`], which pays for a full
    /// per-shard recount and is reserved for the cases that actually
    /// change a collection's existing distribution (an index rebuild or a
    /// repack), so a normal write never regresses to O(collection size).
    fn record_new_shard_collection(&self, pack_id: u64, collection_id: &[u8; 16]) {
        let mut shard_collections = self.shard_collections.write();
        let count = shard_collections
            .entry(pack_id)
            .or_default()
            .entry(*collection_id)
            .or_insert(0);
        *count = count.saturating_add(1);
        drop(shard_collections);
        self.collection_shards
            .write()
            .entry(*collection_id)
            .or_default()
            .insert(pack_id);
    }

    /// Replaces `collection_id`'s entire contribution to `shard_collections` with
    /// `counts` (typically `LossyIndex::shard_counts()` converted to `pack_id` keys)
    /// — clears it out of any shard it no longer occupies and installs the fresh
    /// per-shard counts. Used wherever a collection's index is replaced wholesale
    /// rather than incrementally appended to, since only then can its distribution
    /// across shards actually change.
    fn replace_collection_shard_counts(
        &self,
        collection_id: &[u8; 16],
        counts: &HashMap<u64, u64>,
    ) {
        let old_shards = self
            .collection_shards
            .write()
            .insert(*collection_id, counts.keys().copied().collect());
        let mut shard_collections = self.shard_collections.write();
        if let Some(old_shards) = old_shards {
            for pack_id in &old_shards {
                if !counts.contains_key(pack_id) {
                    if let Some(m) = shard_collections.get_mut(pack_id) {
                        m.remove(collection_id);
                        if m.is_empty() {
                            shard_collections.remove(pack_id);
                        }
                    }
                }
            }
        }
        for (&pack_id, &count) in counts {
            shard_collections
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, count);
        }
    }

    /// Removes `collection_id` from `shard_collections`/`collection_shards` entirely —
    /// used on collection deletion, where nothing of the collection survives in any
    /// shard.
    fn remove_collection_shard_counts(&self, collection_id: &[u8; 16]) {
        let Some(old_shards) = self.collection_shards.write().remove(collection_id) else {
            return;
        };
        let mut shard_collections = self.shard_collections.write();
        for pack_id in old_shards {
            if let Some(m) = shard_collections.get_mut(&pack_id) {
                m.remove(collection_id);
                if m.is_empty() {
                    shard_collections.remove(&pack_id);
                }
            }
        }
    }

    /// Path to the persisted shard→collection directory for a base directory.
    fn shard_collections_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("shard_collections.bin")
    }

    /// Persist the current `shard_collections` directory to disk: which collections
    /// have live records in each shard, and how many. Read by
    /// [`Self::collection_directory_from_disk`] — a free function a separate,
    /// short-lived process (a `collections`-style CLI listing) can call without
    /// opening a full `PackfileStorage` (which would otherwise mean
    /// scanning and index-building every shard just to answer "what collections
    /// exist and how big are they").
    ///
    /// Writes to a temp file and renames into place, same crash-safety
    /// pattern as `ShardPool::persist_stats`.
    ///
    /// The header pins the directory to the `pack_fingerprint` of the
    /// current committed pack set. Only the caller writing right after
    /// flush+fsync (the index-checkpoint rewrite) is guaranteed to produce a
    /// fingerprint matching the checkpoint's, which is what lets open rebuild
    /// per-shard counts from these records instead of walking every slot; any
    /// other write leaves a directory that open either accepts (pack set
    /// unchanged) or safely falls back from.
    ///
    /// # Errors
    /// Returns `StorageError` on write or rename failure.
    pub fn persist_shard_collections(&self) -> Result<(), StorageError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(SHARD_ROOMS_MAGIC);
        buf.push(SHARD_ROOMS_VERSION);
        let packs: Vec<(u64, u64)> = self
            .shards
            .all_shards()
            .into_iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        let fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);
        buf.extend_from_slice(&fingerprint.to_le_bytes());
        let persisted_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        buf.extend_from_slice(&persisted_at.to_le_bytes());
        let collection_order: HashMap<[u8; 16], u64> = self
            .collection_order
            .read()
            .iter()
            .enumerate()
            .map(|(index, collection_id)| {
                u64::try_from(index)
                    .map(|index| (*collection_id, index))
                    .map_err(|_| {
                        StorageError::Io(std::io::Error::other("collection order exceeds u64"))
                    })
            })
            .collect::<Result<_, _>>()?;
        for (pack_id, collections) in self.shard_collections.read().iter() {
            for (collection_id, count) in collections {
                buf.extend_from_slice(&pack_id.to_le_bytes());
                buf.extend_from_slice(collection_id);
                buf.extend_from_slice(&count.to_le_bytes());
                let order = collection_order
                    .get(collection_id)
                    .copied()
                    .unwrap_or(u64::MAX);
                buf.extend_from_slice(&order.to_le_bytes());
            }
        }

        let unique = SHARD_ROOMS_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = Self::shard_collections_path(&self.base_dir)
            .with_extension(format!("bin.tmp.{}.{unique}", std::process::id()));
        let final_path = Self::shard_collections_path(&self.base_dir);
        let write_result = (|| -> std::io::Result<()> {
            let mut tmp = fs::File::create(&tmp_path)?;
            std::io::Write::write_all(&mut tmp, &buf)?;
            tmp.sync_all()
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(StorageError::Io(e));
        }
        fs::rename(&tmp_path, &final_path).map_err(StorageError::Io)?;
        Ok(())
    }

    /// Best-effort wrapper around [`Self::persist_shard_collections`] — logs and
    /// swallows a failure rather than turning it into a hard error, same
    /// contract as `ShardPool`'s stats persistence: this is observability
    /// data, not something worth failing an otherwise-successful sync
    /// over.
    fn persist_shard_collections_best_effort(&self) {
        if let Err(e) = self.persist_shard_collections() {
            eprintln!("mtxdb: failed to persist shard→collection directory: {e}");
        }
    }

    /// Mark that the pending delta log can no longer be continued: a
    /// structural change (capacity growth, full rebuild, repack, refresh, or a
    /// collection create/delete) means replay onto the last checkpoint can't
    /// reconstruct the new state, so the next sync must do a full checkpoint
    /// rewrite (which re-bases the log) instead of an append.
    fn invalidate_delta_log(&self) {
        self.delta_invalidations.fetch_add(1, Ordering::Relaxed);
        self.delta_state.lock().invalid = true;
    }

    /// Record a successful index mutation as a delta frame, if the delta log
    /// can legitimately continue through it.
    ///
    /// Recording is suppressed whenever continuing the log is unsafe: after a
    /// structural invalidation, or when the frame's collection isn't part of
    /// the log's base checkpoint or has since changed shape (its live
    /// generation no longer equals the base's). A suppressed frame simply
    /// leaves the log unable to cover the write; the next sync notices the
    /// gap (missing frames for some collection ⇒ base generation mismatch
    /// recorded at checkpoint-decode time forces a full rewrite) and re-bases.
    fn record_delta(&self, collection_id: &[u8; 16], generation: u64, bucket: u32, slot: u64) {
        let mut state = self.delta_state.lock();
        if state.invalid {
            return;
        }
        if state.base_generations.get(collection_id) != Some(&generation) {
            return;
        }
        state.pending.push(DeltaFrame {
            collection_id: *collection_id,
            bucket,
            generation,
            slot,
        });
    }

    /// Persist the full per-collection index state to `index.checkpoint`, so
    /// the next open can load it instead of rescanning every packfile.
    ///
    /// Called after every sync barrier — packfile data first, checkpoint
    /// second. A crash in between leaves a fingerprint mismatch that the next
    /// open resolves with a rescan; a crash after leaves a valid checkpoint.
    ///
    /// The rewrite hands off the delta log across two epochs: the session
    /// keeps appending to the log that continues the *old* checkpoint (D0)
    /// right up to the rotation below, after which every `put` is recorded
    /// against a fresh epoch (D1) named for the new checkpoint's fingerprint.
    /// D0's file is untouched through the rotation and retired only once the
    /// new checkpoint (C1) is durably renamed, so a crash at any point pairs
    /// the loader with whichever log legitimately continues the checkpoint on
    /// disk: before C1's rename C0 + D0 is authoritative, after it C1 + D1.
    /// The log's fingerprint-based name is itself the safety gate — a stale
    /// predecessor (D0) or an orphaned successor (D1) can never match the
    /// checkpoint a reopen loads (see [`Self::delta_path`]).
    ///
    /// Cheap no-op when no collection data has changed since the last write
    /// (`index_checkpoint_dirty` cleared on success), so a writer that syncs
    /// repeatedly without writes doesn't rewrite the acceleration file. The
    /// flag is cleared only when nothing after this rewrite's snapshot is
    /// left unpersisted (see the lock re-acquisition below), so a transient
    /// failure or a racing write defers rather than drops the update.
    fn persist_index_checkpoint(&self) -> Result<(), StorageError> {
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Epoch-handoff protocol: hold every collection's put_mutex only
        // across the snapshot + epoch rotation below (same sorted-lock
        // pattern as `retire_empty_shards_after_batch`), then release it
        // before the comparatively slow serialize + fsync + rename. This is
        // safe — where releasing right after the fingerprint snapshot alone
        // was *not* (see git history on this function) — because rotation
        // switches every future `put`'s delta frames onto a fresh epoch
        // (named for the fingerprint captured here) before the locks drop,
        // rather than leaving them targeting an epoch that's about to be
        // silently discarded:
        //
        //   - Every put after this point is recorded, if at all, against the
        //     *new* epoch (D1), never the old one (D0) — `rotate_delta_epoch`
        //     runs inside the same locked section as the fingerprint/index
        //     snapshot, so there is no window where a put's frame could still
        //     land on D0 after D0's corresponding checkpoint state has
        //     already been captured for D1's own checkpoint.
        //   - D0's on-disk file is left completely untouched by the rotation
        //     — only retired (deleted) after the new checkpoint (C1) is
        //     durably renamed, below. So a crash before C1 exists leaves the
        //     old checkpoint (C0) still correctly paired with D0.
        //   - D1 is only ever trusted by a reopen if it names the exact
        //     fingerprint of the checkpoint that reopen loads (see
        //     `delta_path`) — so a crash after rotation but before C1 is
        //     durable leaves D1 orphaned (no checkpoint names its
        //     fingerprint) and correctly ignored; the reopen uses C0 + D0.
        //
        // `RoomGeneration` is immutable once published (COW, swapped
        // atomically), so an owned `Arc` clone taken under the lock is as
        // good a snapshot as the live one for serialization purposes.
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        // The guards borrow these `Arc`s, so the handles must outlive the
        // guards; this vector is also re-used for the end-of-function
        // dirty-check re-lock so late-added collections stay pinned too.
        let mut lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();

        // Exclude new-collection publication from the fingerprint→snapshot
        // window. A put *to* an existing collection is already gated by that
        // collection's `put_mutex`, but a put that is about to *create* a
        // collection holds no mutex this function acquired, so its record
        // frame could be Eager-flushed into a pack between our `flush_all` and
        // our fingerprint read — the fingerprint would then name a record
        // whose collection never made it into the snapshot, and a reopen that
        // trusts C1 would silently drop it. Publishing puts hold this lock
        // shared across the entire put; acquiring it exclusive here (before
        // the fingerprint, held through the rotation) means no such put can
        // interleave its flush with our capture: it either finished
        // beforehand (its collection is in the snapshot) or runs entirely
        // afterwards (its bytes land post-fingerprint, and the fingerprint
        // mismatch on reopen routes to the full rescan).
        let create_guard = self.collection_creation.write();
        // A collection created between the initial scan above and this lock
        // wasn't in the locked set and must be now — otherwise a put to one
        // serially dispatched by the engine could still squeeze under the
        // fingerprint while its publication was snapshotted.
        let ids_now: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        for id in ids_now.iter().filter(|id| !collection_ids.contains(id)) {
            lock_arcs.push(self.put_mutex(id));
        }
        // Now build the lock guards: the arcs must outlive them (collected
        // in `lock_arcs` above and reused below for the dirty-check).
        let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();

        // Commit any buffered frames first: the fingerprint below pins each
        // shard to its committed on-disk length, and the serialized index
        // offsets must be readable against exactly that length on the next
        // open. Writing the checkpoint while records are still buffered
        // would persist virtual offsets the fingerprint can't describe.
        self.shards.flush_all()?;
        let packs: Vec<(u64, u64)> = self
            .shards
            .all_shards()
            .into_iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        let fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);

        let order = self.collection_order.read().clone();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let collections = self.collections.read();
            order
                .iter()
                .filter_map(|collection_id| {
                    collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };

        // Rotate the delta epoch while still locked: `old_base_fingerprint`
        // (D0's name, if any) is retired below only after `fingerprint`'s
        // checkpoint (C1) is durable; every collection's writers already
        // target the new epoch (D1) by the time the locks drop next.
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        drop(guards);
        // `create_guard` is deliberately NOT dropped here, unlike the
        // per-collection put mutexes above. An existing collection's
        // concurrent put after this point is safe to let through: its frame
        // lands after `fingerprint` and is recovered via the new delta epoch
        // (D1) on reopen — see `rotate_delta_epoch`. A *newly discovered*
        // collection (via `put`/`put_many` creating one, or `refresh_collection`
        // publishing one from external writes) has no such fallback: it
        // doesn't go through delta tracking, so if it publishes into
        // `self.collections` during the unlocked serialize+write+rename
        // below, its data can already be covered by `fingerprint` (the
        // packs it lives in were flushed above) while being absent from
        // `entries`/`blobs` (snapshotted just above, before it existed) —
        // a crash after the rename makes reopen trust a checkpoint that
        // silently omits it. Keep new-collection publication excluded until
        // the checkpoint is durably renamed.

        let entries: Vec<([u8; 16], u64, Vec<u8>)> = snapshots
            .iter()
            .map(|(collection_id, generation, room)| {
                (*collection_id, *generation, room.index.serialize())
            })
            .collect();
        let blobs: Vec<([u8; 16], u64, &[u8])> = entries
            .iter()
            .map(|(collection_id, generation, blob)| (*collection_id, *generation, blob.as_slice()))
            .collect();
        crate::index::checkpoint::write_checkpoint(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            &blobs,
        )
        .map_err(StorageError::Io)?;
        // `write_checkpoint` fsyncs the new checkpoint's own bytes before
        // renaming it into place, but the rename itself — the directory
        // entry now pointing at the new inode — is only durable once the
        // directory's own metadata is synced. Do that explicitly here rather
        // than relying on `retire_delta_epoch`'s fsync (below) to cover it
        // incidentally: that one only runs, and only after this rename, when
        // there was a prior epoch to retire (`old_base_fingerprint.is_some()`
        // and its `remove_file` succeeds) — a first-ever checkpoint, or a
        // failed unlink, would otherwise leave the rename's durability
        // resting on a later, unrelated sync happening to occur. Best-effort:
        // a failure here is still safe, since a reopen after a crash before
        // this fsync commits either observes the rename (fingerprint gate
        // passes) or doesn't (falls back to the still-valid predecessor
        // checkpoint) — never a torn or partially-visible rename.
        let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        // The checkpoint naming `fingerprint` is now durably in place, so a
        // new collection publishing from here on lands after it — same
        // recovery story a reopen already has for any other post-checkpoint
        // write (full rescan on fingerprint mismatch, or a fresh delta
        // epoch). Safe to let new-collection publication through now.
        drop(create_guard);
        self.retire_delta_epoch(old_base_fingerprint);
        // The unlocked serialize/write window above let concurrent puts land
        // after the rotation. C1 was snapshotted before them, so such a put
        // is durable only as a pending delta frame — if any frames, or a new
        // invalidation, are still unapplied the dirty flag must survive so
        // the next sync appends them to D1 instead of stranding them in
        // memory. Briefly re-acquire every put_mutex (no I/O under them) so
        // this decision can't interleave with a put's own frame-push +
        // dirty-set (see `put_many`; the frame goes in before the flag).
        // Re-lock the *extended* set (`lock_arcs` includes anything published
        // during the initial scan), so a put to a collection that entered the
        // locked window late is still pinned for this check.
        let guards = lock_arcs.iter().map(|arc| arc.lock()).collect::<Vec<_>>();
        let has_unfinished_work = {
            let state = self.delta_state.lock();
            !state.pending.is_empty() || state.invalid
        };
        if !has_unfinished_work {
            self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        }
        drop(guards);
        // Keep the inspection directory gated to the same pack set the
        // checkpoint just became: ride the checkpoint rewrite (best-effort)
        // so the next open can rebuild per-shard counts from records instead
        // of walking every slot. Deliberately not wired to bare sync_all —
        // after a dirty sync() the checkpoint advances to a new fingerprint
        // while a sync_all-only sidecar would stay stale, silently regressing
        // the very reopen this sidecar exists to speed up. A failure here
        // leaves the previous directory stale; the fingerprint gate then
        // falls back to the slot walk until the next rewrite.
        self.persist_shard_collections_best_effort();
        Ok(())
    }

    /// Switch the live delta-log state onto a fresh epoch named for
    /// `fingerprint`, the checkpoint about to be written. Must be called
    /// while every collection's `put_mutex` is held (see
    /// `persist_index_checkpoint`), so every `put` from here on is recorded,
    /// if at all, against the new epoch — never the one this call is
    /// replacing. Returns the *previous* base fingerprint (the epoch to
    /// retire once the new checkpoint is durable), or `None` if this session
    /// had no prior epoch (opened via a full rescan).
    fn rotate_delta_epoch(
        &self,
        fingerprint: u64,
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
    ) -> Option<u64> {
        let mut state = self.delta_state.lock();
        let old_base_fingerprint = state.base_fingerprint;
        state.base_fingerprint = Some(fingerprint);
        state.base_generations = snapshots
            .iter()
            .map(|(collection_id, generation, _)| (*collection_id, *generation))
            .collect();
        state.pending.clear();
        state.log_bytes = 0;
        state.invalid = false;
        old_base_fingerprint
    }

    /// Test-only: a byte-for-byte mirror of `persist_index_checkpoint`,
    /// except it sleeps for `delay` right after releasing the locks (in
    /// place of, not in addition to, the production function's body — this
    /// is never called from non-test code and the production function is
    /// untouched). Lets a test make the unlocked window arbitrarily long
    /// without depending on a dataset large enough to make a real
    /// serialize+fsync slow relative to test-machine noise. See
    /// `test_put_does_not_block_on_slow_checkpoint_rewrite`.
    #[cfg(test)]
    fn test_persist_index_checkpoint_with_delay(
        &self,
        delay: std::time::Duration,
        entered_unlocked_window: &AtomicBool,
    ) -> Result<(), StorageError> {
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            entered_unlocked_window.store(true, Ordering::Release);
            return Ok(());
        }
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mut lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let create_guard = self.collection_creation.write();
        let ids_now: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        for id in ids_now.iter().filter(|id| !collection_ids.contains(id)) {
            lock_arcs.push(self.put_mutex(id));
        }
        let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let order = self.collection_order.read().clone();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let collections = self.collections.read();
            order
                .iter()
                .filter_map(|collection_id| {
                    collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        drop(guards);
        drop(create_guard);

        // The test waits on this: it now *knows* the rewrite is sleeping in
        // its unlocked window rather than guessing via a fixed sleep, so a
        // put issued from here provably overlaps the delay.
        entered_unlocked_window.store(true, Ordering::Release);

        std::thread::sleep(delay);

        let entries: Vec<([u8; 16], u64, Vec<u8>)> = snapshots
            .iter()
            .map(|(collection_id, generation, room)| {
                (*collection_id, *generation, room.index.serialize())
            })
            .collect();
        let blobs: Vec<([u8; 16], u64, &[u8])> = entries
            .iter()
            .map(|(collection_id, generation, blob)| (*collection_id, *generation, blob.as_slice()))
            .collect();
        crate::index::checkpoint::write_checkpoint(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            &blobs,
        )
        .map_err(StorageError::Io)?;
        let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        self.retire_delta_epoch(old_base_fingerprint);
        let guards = lock_arcs.iter().map(|m| m.lock()).collect::<Vec<_>>();
        let has_unfinished_work = {
            let state = self.delta_state.lock();
            !state.pending.is_empty() || state.invalid
        };
        if !has_unfinished_work {
            self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        }
        drop(guards);
        self.persist_shard_collections_best_effort();
        Ok(())
    }

    /// Test-only: perform exactly the locked prefix of
    /// `persist_index_checkpoint` (flush, fingerprint, index snapshot, epoch
    /// rotation) and then stop, *never* writing or renaming a checkpoint.
    /// Simulates a crash between the D0→D1 rotation and C1's rename — the
    /// window where D1 exists only in memory and no on-disk checkpoint names
    /// its fingerprint yet.
    #[cfg(test)]
    fn test_rotate_epoch_without_checkpoint(&self) -> (u64, Option<u64>) {
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all().unwrap();
        let fingerprint = self.current_pack_fingerprint();
        let order = self.collection_order.read().clone();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let collections = self.collections.read();
            order
                .iter()
                .filter_map(|collection_id| {
                    collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        (fingerprint, old_base_fingerprint)
    }

    /// Test-only: perform everything `persist_index_checkpoint` does up
    /// through the checkpoint rename and its directory fsync, but stop
    /// *before* `retire_delta_epoch`. Simulates a crash between C1 becoming
    /// durable and D0's retirement — the window where both C1+D1 (current)
    /// and C0's now-stale D0 (orphaned) exist on disk simultaneously.
    #[cfg(test)]
    fn test_write_checkpoint_without_retire(&self) -> Result<(u64, Option<u64>), StorageError> {
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let order = self.collection_order.read().clone();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let collections = self.collections.read();
            order
                .iter()
                .filter_map(|collection_id| {
                    collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        drop(guards);
        let entries: Vec<([u8; 16], u64, Vec<u8>)> = snapshots
            .iter()
            .map(|(collection_id, generation, room)| {
                (*collection_id, *generation, room.index.serialize())
            })
            .collect();
        let blobs: Vec<([u8; 16], u64, &[u8])> = entries
            .iter()
            .map(|(collection_id, generation, blob)| (*collection_id, *generation, blob.as_slice()))
            .collect();
        crate::index::checkpoint::write_checkpoint(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            &blobs,
        )
        .map_err(StorageError::Io)?;
        let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        Ok((fingerprint, old_base_fingerprint))
    }

    /// Remove the now-superseded delta-log epoch, once the checkpoint that
    /// makes it safe to discard is durably renamed. A no-op if this session
    /// had no prior epoch. Best-effort: a failure here leaves the retired
    /// epoch's file on disk, which is always safe (see `delta_path`) — it
    /// simply names a fingerprint no future checkpoint will ever carry again.
    fn retire_delta_epoch(&self, old_base_fingerprint: Option<u64>) {
        let Some(old_base_fingerprint) = old_base_fingerprint else {
            return;
        };
        let path = Self::delta_path(&self.base_dir, old_base_fingerprint);
        if fs::remove_file(&path).is_ok() {
            // The checkpoint rename's own durability is already covered by
            // the caller's fsync right after `write_checkpoint` returns; this
            // one covers the unlink instead, so the retired epoch's file
            // doesn't linger past a crash — harmless either way (see the
            // doc comment above), but tidier.
            let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        }
    }

    /// Append the pending delta frames as one framed, fsynced batch and clear
    /// the dirty flag. `tail_fingerprint` is the fingerprint of the pack set
    /// the frames were recorded against — computed by the caller after the
    /// flush that made those bytes durable. The on-disk log, if any, is
    /// continued; otherwise a fresh header pins `base_fingerprint` (the
    /// checkpoint the frames extend) for the reopen replay gate.
    fn append_index_delta(&self, tail_fingerprint: u64) -> Result<(), StorageError> {
        let mut state = self.delta_state.lock();
        let Some(base_fingerprint) = state.base_fingerprint else {
            return Err(StorageError::Io(std::io::Error::other(
                "delta append with no base fingerprint",
            )));
        };
        if state.invalid {
            return Err(StorageError::Io(std::io::Error::other(
                "delta append while the log is invalidated",
            )));
        }
        if state.pending.is_empty() {
            return Err(StorageError::Io(std::io::Error::other(
                "delta append with no pending frames",
            )));
        }
        let path = Self::delta_path(&self.base_dir, base_fingerprint);
        // A decoder deliberately stops at a torn or corrupt suffix and gives
        // us the sealed frontier. Before extending the log, discard that
        // suffix: appending after it would leave every new, valid batch behind
        // bytes a future decoder correctly refuses to cross.
        let on_disk_len = fs::metadata(&path)
            .map_or_else(
                |error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Ok(0)
                    } else {
                        Err(error)
                    }
                },
                |metadata| Ok(metadata.len()),
            )
            .map_err(StorageError::Io)?;
        if on_disk_len < state.log_bytes {
            return Err(StorageError::Io(std::io::Error::other(
                "delta log shrank below its sealed frontier",
            )));
        }
        if on_disk_len > state.log_bytes {
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(state.log_bytes))
                .map_err(StorageError::Io)?;
        }
        let write_header = state.log_bytes == 0;
        // Do not charge a continuation for another header: the exact next
        // physical write is either one batch, or a header plus one batch.
        let batch_bytes = delta::batch_len(state.pending.len()).ok_or_else(|| {
            StorageError::Io(std::io::Error::other("delta batch length overflow"))
        })?;
        let next_len = state
            .log_bytes
            .saturating_add(u64::try_from(batch_bytes).unwrap_or(u64::MAX))
            .saturating_add(if write_header {
                DELTA_LOG_HEADER_LEN as u64
            } else {
                0
            });
        if next_len > DELTA_LOG_CAP_BYTES {
            return Err(StorageError::Io(std::io::Error::other(
                "delta log cap reached",
            )));
        }
        let pending = std::mem::take(&mut state.pending);
        let appended = delta::append_batch(
            &path,
            write_header,
            base_fingerprint,
            &pending,
            tail_fingerprint,
        )
        .map_err(StorageError::Io)?;
        state.log_bytes = state.log_bytes.saturating_add(appended as u64);
        // Clear the dirty flag *under* the delta-state lock: a racing put
        // pushes its frame (`record_delta`) before setting the flag, so
        // clearing after `drop(state)` could clobber a frame that was pushed
        // between the `mem::take` above and this store, leaving it stranded in
        // memory with nothing to force its append.
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        drop(state);
        self.persist_shard_collections_best_effort();
        Ok(())
    }

    /// Best-effort wrapper around [`Self::persist_index_checkpoint`] — logs
    /// and swallows a failure rather than turning it into a hard error. The
    /// checkpoint is an acceleration structure; packfiles remain
    /// authoritative, so a failed persist only costs the next open a rescan.
    fn persist_index_checkpoint_best_effort(&self) {
        if let Err(e) = self.persist_index_checkpoint() {
            eprintln!("mtxdb: failed to persist index checkpoint: {e}");
        }
    }

    /// Reads a persisted shard→collection directory directly off disk, with no
    /// `ShardPool`/`PackfileStorage` construction at all — the fast path
    /// for a `collections`-style CLI listing. Returns each collection's total record
    /// count summed across every shard it appears in, in insertion order.
    ///
    /// Best-effort: a missing, truncated, or corrupt file just yields an
    /// empty result rather than an error — the caller decides whether to
    /// fall back to a full scan (e.g. because this store predates the
    /// feature, or a writer hasn't flushed it yet).
    #[must_use]
    pub fn collection_directory_from_disk(base_dir: &std::path::Path) -> Vec<([u8; 16], u64)> {
        let Some(records) = read_persisted_shard_collections(base_dir).map(|d| d.records) else {
            return Vec::new();
        };
        let mut totals: HashMap<[u8; 16], (u64, u64)> = HashMap::new();
        for record in records {
            let entry = totals
                .entry(record.collection_id)
                .or_insert((0, record.insertion_order));
            entry.0 = entry.0.saturating_add(record.count);
            entry.1 = entry.1.min(record.insertion_order);
        }
        let mut result: Vec<([u8; 16], u64, u64)> = totals
            .into_iter()
            .map(|(collection_id, (count, order))| (collection_id, count, order))
            .collect();
        result.sort_unstable_by_key(|(collection_id, _, order)| (*order, *collection_id));

        result
            .into_iter()
            .map(|(collection_id, count, _)| (collection_id, count))
            .collect()
    }

    /// Reads collection summaries from the persisted shard→collection directory without
    /// opening packfiles. The node count and index-RAM figure match the
    /// index rebuilt by [`Self::open`], provided the sidecar is current.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn collection_summaries_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<Vec<CollectionSummary>> {
        Self::collection_directory_persisted_at(base_dir)?;
        Self::collection_directory_from_disk(base_dir)
            .into_iter()
            .map(|(collection_id, nodes)| {
                let nodes = usize::try_from(nodes).ok()?;
                // A live collection always starts at `NEW_COLLECTION_INDEX_FLOOR`
                // (64), not `LossyIndex::new`'s own generic 16-slot floor — use
                // the same starting point here or a small collection's
                // capacity/memory estimate undershoots its real index.
                let capacity = u32::try_from(LossyIndex::capacity_for_entries(
                    nodes,
                    NEW_COLLECTION_INDEX_FLOOR,
                ))
                .ok()?;
                Some((
                    collection_id,
                    nodes,
                    LossyIndex::memory_usage_for_entries(nodes, NEW_COLLECTION_INDEX_FLOOR),
                    capacity,
                ))
            })
            .collect()
    }

    /// Current live shard `pack_id`s per collection from the persisted shard→collection
    /// directory, without opening packfiles or rebuilding collection indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn collection_shards_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<HashMap<[u8; 16], Vec<u64>>> {
        let records = read_persisted_shard_collections(base_dir)?.records;
        let mut shards: HashMap<[u8; 16], Vec<u64>> = HashMap::new();
        for record in records {
            shards
                .entry(record.collection_id)
                .or_default()
                .push(record.pack_id);
        }
        for collection_shards in shards.values_mut() {
            collection_shards.sort_unstable();
            collection_shards.dedup();
        }
        Some(shards)
    }

    /// Current live-node count per shard from the persisted shard→collection
    /// directory, without opening packfiles or rebuilding collection indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed. Callers that need an authoritative fresh count can then
    /// explicitly open the store and rebuild its indexes instead.
    #[must_use]
    pub fn shard_node_counts_from_disk(base_dir: &std::path::Path) -> Option<HashMap<u64, u64>> {
        let records = read_persisted_shard_collections(base_dir)?.records;
        let mut totals = HashMap::new();
        for record in records {
            let total = totals.entry(record.pack_id).or_insert(0_u64);
            *total = total.saturating_add(record.count);
        }
        Some(totals)
    }

    /// Current number of collections with live nodes per shard from the persisted
    /// shard→collection directory, without opening packfiles or rebuilding collection
    /// indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn shard_collection_counts_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<HashMap<u64, u64>> {
        let records = read_persisted_shard_collections(base_dir)?.records;
        let mut totals = HashMap::new();
        for record in records {
            let count = totals.entry(record.pack_id).or_insert(0_u64);
            *count = count.saturating_add(1);
        }
        Some(totals)
    }

    /// Unix-seconds timestamp of the persisted shard→collection directory, if
    /// one exists — lets a reader (e.g. a `collections` CLI listing) label how
    /// stale the counts it's showing are, the same way
    /// `ShardPool::stats_persisted_at` does for shard IO stats.
    #[must_use]
    pub fn collection_directory_persisted_at(base_dir: &std::path::Path) -> Option<u64> {
        read_persisted_shard_collections(base_dir).map(|d| d.persisted_at)
    }

    /// Pin one `Arc<Shard>` per distinct shard id, keeping each shard's file
    /// handle (and thus its `Drop`-triggered deletion, see `Shard::drop`)
    /// alive for as long as the returned map is held — even if a later
    /// write in the same operation triggers `ShardPool::rotate` and retires
    /// that shard id's pool slot.
    ///
    /// This matters specifically for a scan-then-rewrite repack: without
    /// it, re-querying `ShardPool::get_shard` on every loop iteration can
    /// observe a shard that a *rotation triggered by the same repack's own
    /// writes* just retired-and-recreated at the same path, turning a
    /// stale-but-valid offset into a read into unrelated, freshly-written
    /// bytes.
    fn pin_shards(&self, shard_ids: impl Iterator<Item = u16>) -> HashMap<u16, Arc<Shard>> {
        let unique: HashSet<u16> = shard_ids.collect();
        unique
            .into_iter()
            .filter_map(|id| self.shards.get_shard(id).map(|shard| (id, shard)))
            .collect()
    }

    /// Copy a record from an old (pinned) shard to the active shard.
    /// Returns the new `(hash, shard_id, offset)` or None if `old_shard_id`
    /// isn't in `pinned`.
    fn copy_record_to_shard(
        &self,
        collection_id: &[u8; 16],
        pinned: &HashMap<u16, Arc<Shard>>,
        old_shard_id: u16,
        old_offset: u64,
    ) -> Result<Option<([u8; 16], u16, u64)>, StorageError> {
        let Some(old_shard) = pinned.get(&old_shard_id) else {
            return Ok(None);
        };
        let record = self.read_at(old_shard, old_offset, true)?;
        let (new_shard_id, new_offset) = self.shards.put_record(&Record {
            collection_id: *collection_id,
            hash: record.hash,
            data: record.data,
        })?;
        Ok(Some((record.hash, new_shard_id, new_offset)))
    }
    fn swap_generation(
        &self,
        collection_id: &[u8; 16],
        index: LossyIndex,
    ) -> Result<(), StorageError> {
        let cache = self.generation(collection_id).map(|g| g.cache.clone());
        self.store_generation(collection_id, index, cache, true)
    }

    /// Store a new generation for a collection, reusing the existing cache if present.
    ///
    /// If the collection is new (not yet in the `collections` map), it is added to
    /// `collection_order` so that [`Self::collection_summaries`] will include it, and
    /// any prior tombstone in `deleted.collections` is cleared so the collection
    /// survives a restart.
    ///
    /// `bump_generation` marks a *shape* change to the index (capacity growth,
    /// a from-scratch rebuild, repack, refresh) versus a plain COW update that
    /// preserves the previous generation. Shape changes invalidate the pending
    /// delta log — its frames were recorded against the old table, so they
    /// can't be replayed onto the checkpoint's — and the next sync rewrites the
    /// checkpoint instead of appending. Creating a brand-new collection also
    /// invalidates the log: a log that predates the collection can't express
    /// its records, so replaying it would silently omit every one.
    fn store_generation(
        &self,
        collection_id: &[u8; 16],
        index: LossyIndex,
        cache: Option<Arc<NodeCache>>,
        bump_generation: bool,
    ) -> Result<(), StorageError> {
        let cache = cache.unwrap_or_else(|| Arc::new(NodeCache::new(self.cache_capacity)));
        let is_new = self.collections.read().get(collection_id).is_none();
        let structural = bump_generation || is_new;
        if structural {
            self.invalidate_delta_log();
        }
        let generation = self.generation(collection_id).map_or(1, |gen| {
            gen.generation.saturating_add(u64::from(bump_generation))
        });
        let new_gen = Arc::new(RoomGeneration {
            index,
            cache,
            generation,
        });

        // Clear a prior deletion marker before publishing the recreated
        // generation. A failure must fail the write: otherwise it appears to
        // succeed but disappears on the next startup scan.
        if is_new {
            if self.is_deleted_collection(collection_id) {
                self.clear_deleted_collection(collection_id)?;
            }
            self.collections
                .write()
                .entry(*collection_id)
                .or_insert_with(|| {
                    ArcSwap::from_pointee(RoomGeneration {
                        index: LossyIndex::new(0),
                        cache: Arc::new(NodeCache::new(self.cache_capacity)),
                        generation: 1,
                    })
                })
                .store(new_gen);

            if !self.collection_order.read().contains(collection_id) {
                self.collection_order.write().push(*collection_id);
            }
        } else {
            // Fast path: just update the ArcSwap using a read lock on the map.
            // This avoids a global write lock on every single put() call.
            let read_guard = self.collections.read();
            if let Some(arc_swap) = read_guard.get(collection_id) {
                arc_swap.store(new_gen);
            } else {
                // Fallback in case of a race condition with a deletion
                drop(read_guard);
                self.collections
                    .write()
                    .entry(*collection_id)
                    .or_insert_with(|| {
                        ArcSwap::from_pointee(RoomGeneration {
                            index: LossyIndex::new(0),
                            cache: Arc::new(NodeCache::new(self.cache_capacity)),
                            generation: 1,
                        })
                    })
                    .store(new_gen);
            }
        }
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }
    /// Forces a re-scan of a collection's shards from disk and atomically swaps in the new index.
    /// This is designed for multi-worker environments to pull in external appends on demand.
    ///
    /// # Errors
    /// Returns `StorageError` if reading or parsing the underlying shards fails.
    pub fn refresh_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        // A refresh that publishes a *newly discovered* collection must be
        // excluded from a concurrent checkpoint's fingerprint→snapshot
        // window, exactly like a new-collection `put`/`put_many` (see the
        // comment on `put`). Otherwise its `rebuild_index` flush could land
        // the refreshed records in a pack covered by the checkpoint's
        // fingerprint while the collection itself is absent from the
        // snapshot, and a reopen trusting that checkpoint would lose them.
        let _create_guard = if self.generation(collection_id).is_none() {
            Some(self.collection_creation.read())
        } else {
            None
        };
        self.shards.discover_shards()?;
        let new_index = self.rebuild_index(collection_id)?;
        let existing_cache = self.generation(collection_id).map(|g| g.cache.clone());
        self.store_generation(collection_id, new_index, existing_cache, true)
    }

    fn rebuild_index(&self, collection_id: &[u8; 16]) -> Result<LossyIndex, StorageError> {
        // Full-fallback path: a batch that exhausted growth re-scans the
        // packfiles. Tracked so a heavy rebuild is visible in `stats()`.
        self.index_rebuild_count.fetch_add(1, Ordering::Relaxed);
        // `scan_collection_records` traverses packfiles; any buffered frames
        // (this collection's or any other's) are invisible to it, so a fresh
        // flush guarantees the rebuilt index reflects every record that has
        // actually been put. Callers only reach this path while holding the
        // collection's put mutex, so nothing new can buffer for this
        // collection between the flush and the scan.
        self.shards.flush_all()?;
        let scanned = self.scan_collection_records(collection_id)?;
        let total: usize = scanned.iter().map(|(_, e)| e.len()).sum();
        let index = LossyIndex::new(total.saturating_mul(2).max(NEW_COLLECTION_INDEX_FLOOR));
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                // `IndexSlot` can only represent offsets up to
                // `PACK_INDEX_OFFSET_LIMIT`. Offsets a legacy or externally
                // created oversized shard can no longer fit are rejected
                // rather than silently skipped: writes are already capped at
                // `MAX_SHARD_BYTES`, so a record beyond the limit means the
                // shard did not come from this engine, and silently indexing
                // around it would make its data unreachable while reporting a
                // successful rebuild.
                check_index_offset(shard_id, &hash, offset)?;
                let _ = index.insert(&hash, shard_id, offset);
            }
        }
        Ok(index)
    }

    /// Rehash a checkpoint-derived index without rescanning unrelated packs.
    ///
    /// A compact checkpoint keeps the packed `(shard, offset)` slots but not
    /// the 64-bit home hashes needed by [`LossyIndex::grow`].  Each occupied
    /// slot still names an authoritative frame whose fixed metadata prefix
    /// contains the full hash, so recover those identities and rehash only
    /// this collection's entries.  A missing shard, malformed frame, or a
    /// frame from another collection fails closed to the caller's existing
    /// whole-collection scan fallback.
    fn grow_checkpoint_index(
        &self,
        collection_id: &[u8; 16],
        index: &LossyIndex,
    ) -> Result<Option<LossyIndex>, StorageError> {
        index.grow_by_recovering_hashes(|shard_id, offset, slot_tag| {
            let shard = self.shards.get_shard(shard_id).ok_or_else(|| {
                StorageError::Corrupt(format!(
                    "checkpoint index refers to missing shard {shard_id}"
                ))
            })?;
            let (found_collection, hash) = self.shards.record_identity_at(&shard, offset)?;
            if &found_collection != collection_id {
                return Err(StorageError::Corrupt(format!(
                    "checkpoint index offset {offset} in shard {shard_id} belongs to another collection"
                )));
            }
            if LossyIndex::tag_for_hash(&hash) != slot_tag {
                return Err(StorageError::Corrupt(format!(
                    "checkpoint index tag does not match frame at offset {offset} in shard {shard_id}"
                )));
            }
            Ok(hash)
        })
    }

    /// Insert with exact identity checks for checkpoint-derived slots.
    ///
    /// A compact checkpoint retains a slot's tag and location but not the
    /// remaining hash bits. On the rare same-tag probe, recover that frame's
    /// metadata, hydrate the in-memory slot, and retry. Active indexes carry
    /// their identities already, so their ordinary inserts take no I/O path.
    fn insert_index(
        &self,
        collection_id: &[u8; 16],
        index: &LossyIndex,
        hash: &NodeId,
        shard_id: u16,
        offset: u64,
    ) -> Result<Result<(u32, u64), InsertError>, StorageError> {
        loop {
            match index.insert_tracked(hash, shard_id, offset) {
                Ok(written) => return Ok(Ok(written)),
                Err(InsertError::TableFull) => return Ok(Err(InsertError::TableFull)),
                Err(InsertError::NeedsIdentity {
                    bucket,
                    shard_id,
                    offset,
                }) => {
                    let shard = self.shards.get_shard(shard_id).ok_or_else(|| {
                        StorageError::Corrupt(format!(
                            "index refers to missing shard {shard_id} while resolving a tag collision"
                        ))
                    })?;
                    let (found_collection, found_hash) =
                        self.shards.record_identity_at(&shard, offset)?;
                    if &found_collection != collection_id
                        || !index.hydrate_slot_identity(bucket, &found_hash)
                    {
                        return Err(StorageError::Corrupt(format!(
                            "index tag candidate at offset {offset} in shard {shard_id} is inconsistent"
                        )));
                    }
                }
            }
        }
    }

    /// Fetch a node and return it as a `NodeRef` with swizzled children.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure or CRC mismatch.
    pub fn get_swizzled(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        extract_children: impl Fn(&NodeData) -> Vec<NodeId>,
    ) -> Result<Option<NodeRef>, StorageError> {
        let Some(data) = self.get(collection_id, id)? else {
            return Ok(None);
        };

        if let Some(swizzle_fn) = self.swizzle {
            let children = extract_children(&data);
            if !children.is_empty() {
                let cached = if let Some(gen) = self.generation(collection_id) {
                    gen.cache.resolve_hashes(&children)
                } else {
                    vec![None; children.len()]
                };
                let swizzled = swizzle_fn(&data, &children, &cached);
                if let Some(gen) = self.generation(collection_id) {
                    gen.cache.insert(*id, Arc::new(swizzled.clone()));
                }
                return Ok(Some(NodeRef::Resolved(*id, Arc::new(swizzled))));
            }
        }

        Ok(Some(NodeRef::Resolved(*id, Arc::new(data))))
    }

    fn resolve_from_candidates(
        &self,
        id: &NodeId,
        candidates: impl IntoIterator<Item = (u16, u64)>,
    ) -> Result<Option<NodeData>, StorageError> {
        let mut last_err: Option<StorageError> = None;
        for (shard_id, offset) in candidates {
            let Some(shard) = self.shards.get_shard(shard_id) else {
                continue;
            };

            match self.read_at(
                &shard,
                offset,
                self.shards.checksum_policy().verifies_reads(),
            ) {
                Ok(record) => {
                    if record.hash != *id {
                        continue;
                    }

                    let data = NodeData {
                        bytes: record.data,
                        children: Vec::new(),
                    };
                    // A raw mmap-backed `Bytes` is already cheap to retain
                    // in the caller. Do not turn every cold lookup into an
                    // LRU write (hash probe, lock, and eviction bookkeeping).
                    // The cache remains populated by writes and swizzling,
                    // where it stores work that cannot be recovered by a
                    // simple mmap range.
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

    // repack_collection_rewrite and repack_collection_topo were retired: both scanned
    // for records without deduplicating physical duplicates (each prior
    // repack's rewritten copies included), giving an exponential blowup on
    // repeated calls against a growing collection, and both re-fetched
    // Arc<Shard> per loop iteration rather than pinning shards for the
    // call's duration — vulnerable to the rotation race documented on
    // `pin_shards`. `repack_collection_reachable`'s no-live-roots fallback
    // subsumes both: same topological ordering, correct dedup via
    // `hash_to_shard_offset`, and pinned shards throughout.

    /// BFS outward from `roots` over `extract_edges`, reading only records
    /// actually reached. Returns the sorted, deduplicated live hash set and
    /// the adjacency discovered along the way.
    fn bfs_live_set(
        pool: &ShardPool,
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut visited: HashSet<[u8; 16]> = HashSet::new();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(&(shard_id, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&shard_id) else {
                continue;
            };
            let record = pool.read_at(old_shard, offset, true)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        let mut live_hashes: Vec<[u8; 16]> = visited.into_iter().collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// BFS backward (toward ancestors) from `frontier` over `extract_edges`,
    /// stopping expansion (and excluding from the result) anything in
    /// `stop_at`. Unlike [`Self::bfs_live_set`]/[`Self::bfs_bounded`]'s
    /// repack-side sibling, this resolves each node through `gen`'s index
    /// — a single frozen generation snapshot taken once before the walk
    /// starts — one hash lookup at a time, rather than pre-scanning every
    /// shard for the whole collection up front. Cost is proportional to the
    /// number of nodes actually walked, not collection size.
    ///
    /// Pinning `gen` for the whole walk (instead of re-resolving through
    /// `self.get()` per node against whatever generation happens to be
    /// live at call time) is also what makes this concurrency-safe: a
    /// repack that runs mid-walk and GCs a hash this walk already needs
    /// swaps in a *new* generation rather than mutating this one, so a
    /// node this walk has already committed to visiting can't vanish out
    /// from under it and get silently skipped.
    ///
    /// Returns the resolved `NodeData` for each visited node alongside the
    /// adjacency, so the caller has everything it needs without a second
    /// per-node fetch.
    ///
    /// This is an **ancestor walk with stop markers**, not a "span between
    /// two frontiers": a branch whose history never crosses any `stop_at`
    /// node is walked all the way back to the collection's true roots, not
    /// truncated at some implied boundary. A real from/to span would need
    /// `ancestors(frontier) ∩ descendants(stop_at)`, which requires
    /// reverse adjacency this call doesn't have — out of scope here.
    ///
    /// `limits.max_nodes`, if set, caps how many nodes this walk will visit
    /// before giving up on expanding further (existing queued nodes still
    /// finish resolving their own edges into `adjacency`, but no new nodes
    /// are enqueued past the cap) — the safety valve for exactly the
    /// runaway case above, where `stop_at` never gets hit.
    fn bfs_ancestors(
        &self,
        gen: &Arc<RoomGeneration>,
        frontier: &[[u8; 16]],
        stop_at: &[[u8; 16]],
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        limits: WalkLimits,
    ) -> Result<(AdjacencyResult, HashMap<[u8; 16], NodeData>), StorageError> {
        let boundary: HashSet<[u8; 16]> = stop_at.iter().copied().collect();
        let mut visited: HashSet<[u8; 16]> = HashSet::new();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        let mut resolved: HashMap<[u8; 16], NodeData> = HashMap::new();
        let at_cap =
            |visited: &HashSet<[u8; 16]>| limits.max_nodes.is_some_and(|max| visited.len() >= max);

        for root in frontier {
            if !boundary.contains(root) && !at_cap(&visited) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(data) = self.resolve_pinned(gen, &hash)? else {
                // Not actually present under this generation — nothing to
                // expand from and nothing to yield.
                continue;
            };
            let edges = extract_edges(&hash, &data.bytes);
            for edge in &edges {
                if !boundary.contains(edge) && !at_cap(&visited) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
            resolved.insert(hash, data);
        }

        // live_hashes tracks only nodes actually resolved, not everything
        // `visited` touched — a hash that turned out absent under `gen`
        // was marked visited to prevent re-queueing, but never belongs in
        // the CSR or the output.
        let mut live_hashes: Vec<[u8; 16]> = resolved.keys().copied().collect();
        live_hashes.sort_unstable();
        Ok(((live_hashes, adjacency), resolved))
    }

    /// Resolves `id` within a specific, already-loaded generation snapshot
    /// rather than whatever generation happens to be live when called —
    /// the building block [`Self::bfs_ancestors`] uses to keep an entire
    /// walk consistent against one point-in-time view of the collection, immune
    /// to a repack swapping in a new generation partway through.
    fn resolve_pinned(
        &self,
        gen: &Arc<RoomGeneration>,
        id: &NodeId,
    ) -> Result<Option<NodeData>, StorageError> {
        if let Some(data) = gen.cache.get(id) {
            return Ok(Some((*data).clone()));
        }
        self.resolve_from_candidates(id, gen.index.lookup_all(id))
    }

    /// Read every record's edges, with no reachability filtering.
    /// Used when a collection has no configured live roots, so nothing is known
    /// to be garbage. Iterates `hash_to_shard_offset` directly rather
    /// than the raw scan output, so it works for both full and incremental
    /// repack (where the map was built by merging new entries into a
    /// previous state rather than scanning every packfile from byte zero).
    fn scan_full_adjacency(
        pool: &ShardPool,
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&shard_id) {
                let record = pool.read_at(shard, offset, true)?;
                let edges = extract_edges(hash, &record.data);
                adjacency.insert(*hash, edges);
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Rewrite a collection's records in topological order, optionally performing
    /// garbage collection — this is the engine's one repack entry point.
    ///
    /// If live roots are configured for this collection (see
    /// [`Self::set_live_roots`]), traverses outward from them via
    /// `extract_edges` and drops anything not reached: only reachable
    /// records are read from disk during the traversal, so unreachable
    /// data is never even fetched, not just excluded from the output.
    ///
    /// If no live roots are configured (`set_live_roots` was never called,
    /// or was called with an empty list), nothing is known to be garbage,
    /// so every scanned record is preserved, still deduplicated and
    /// rewritten in topological order — rather than risk deleting live
    /// data on the assumption that "no roots" means "nothing is live".
    ///
    /// All shards this call will read from are pinned (held via an
    /// `Arc<Shard>` for the whole call) before any writes happen, so a
    /// rotation triggered by this call's own writes can never retire a
    /// shard this
    /// call still needs to read stale data from.
    ///
    /// Returns `(kept, dropped)`: the number of records written to the new
    /// generation, and the number of scanned records found unreachable and
    /// discarded.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    /// Builds this repack's deduped `hash → (shard_id, offset)` map,
    /// incrementally where possible, and returns the previous
    /// [`RepackIncrementalState`] (moved out, not cloned) so the caller can
    /// reuse its `adjacency` cache when deciding which nodes need fresh disk
    /// reads.
    ///
    /// On the incremental path the previous `live_map` is extended in-place
    /// with only the newly-appended entries from each shard (O(delta) scan
    /// work per repack call rather than O(total)). On the cold-start path
    /// (first repack for this collection) every shard is scanned from byte
    /// zero.
    ///
    /// Returns `(hash_to_shard_offset, Some(prev))` on the incremental path
    /// and `(hash_to_shard_offset, None)` on the cold-start path. The caller
    /// must pass `prev` to the adjacency helpers to unlock the incremental
    /// BFS optimisation; when it is `None` the caller uses the full-scan
    /// adjacency helpers instead.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure scanning a shard.
    fn repack_scan_incremental(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<RepackScanResult, StorageError> {
        // Check for existing state under a read lock, then move it out under a
        // write lock if present — O(1) ownership transfer, no clone.
        let has_prev = self.repack_incremental.read().contains_key(collection_id);
        if has_prev {
            let prev = self
                .repack_incremental
                .write()
                .remove(collection_id)
                .expect("entry was present under read lock and collection mutex is held");

            let shards = self.shards.all_shards();
            let current_pack_ids: HashMap<u16, u64> = shards
                .iter()
                .map(|(shard_id, shard)| (*shard_id, shard.pack_id))
                .collect();
            // Slot IDs are recyclable.  A cursor (and every entry in the
            // carried live_map) is meaningful only for the pack incarnation
            // that produced it.  On any replacement or disappearance, start
            // over rather than mixing old offsets with the new pack.
            let pack_changed = prev
                .scan_offsets
                .iter()
                .any(|(shard_id, (pack_id, _))| current_pack_ids.get(shard_id) != Some(pack_id));
            if pack_changed {
                let scanned = self.scan_collection_records(collection_id)?;
                let mut map = HashMap::new();
                for (shard_id, entries) in scanned {
                    for (hash, offset) in entries {
                        map.insert(hash, (shard_id, offset));
                    }
                }
                return Ok((map, None));
            }

            // Scan only bytes appended since the last repack, merging into
            // the stolen live_map in-place — O(delta) work, not O(total).
            let mut map = prev.live_map.clone();
            for (shard_id, shard) in shards {
                let start = prev
                    .scan_offsets
                    .get(&shard_id)
                    .filter(|(pack_id, _)| *pack_id == shard.pack_id)
                    .map_or(0, |(_, offset)| *offset);
                let entries =
                    packfile::scan_packfile_from(&shard.path, start).map_err(StorageError::Io)?;
                for (rid, hash, offset) in entries {
                    if rid == *collection_id {
                        map.insert(hash, (shard_id, offset));
                    }
                }
            }
            Ok((map, Some(prev)))
        } else {
            // First repack: full scan of every shard.
            let scanned = self.scan_collection_records(collection_id)?;
            let mut map = HashMap::new();
            for (shard_id, entries) in &scanned {
                for (hash, offset) in entries {
                    map.insert(*hash, (*shard_id, *offset));
                }
            }
            Ok((map, None))
        }
    }

    /// Incremental BFS reachability walk.
    ///
    /// Uses the adjacency cache from `prev` to avoid re-reading disk for nodes
    /// that were already known-live after the last repack. Only newly-added
    /// nodes (those absent from `prev.adjacency`) require a disk read.
    ///
    /// **Root-removal invariant (critical):** This function must NOT be called
    /// when the current root set is a strict subset of `prev.prev_roots` (i.e.
    /// any root was removed since the last repack). In that case nodes
    /// exclusively reachable through the removed root would remain in the live
    /// set indefinitely, because `visited` in `prev.adjacency` only grows —
    /// nothing in this incremental path forces a re-check of reachability.
    /// The caller ([`Self::repack_collection_reachable`]) detects root removal
    /// and falls back to [`Self::bfs_live_set`] (cold-start, full BFS) for
    /// that repack, then saves a fresh cursor so subsequent repacks can be
    /// incremental again.
    ///
    /// **Memory:** `prev.adjacency` holds `O(live_nodes` × `avg_fanout`) hashes
    /// (≈ `live_nodes × avg_fanout × 16` bytes) in RAM per collection. This
    /// is the cost of eliminating the O(live) disk reads per repack call —
    /// callers should be aware for large collections.
    fn bfs_live_set_incremental(
        pool: &ShardPool,
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        // Seed with roots that are not already in the cached live set.
        // Anything in prev.adjacency was reachable last time and — given no
        // root was removed — is still reachable now; no need to re-expand it.
        let mut visited: HashSet<[u8; 16]> = prev.adjacency.keys().copied().collect();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        // Retain the cached edges for already-live nodes.  Initialising these
        // entries with empty vectors loses prior edges when they are not put
        // back on the queue during an incremental rooted repack.
        let mut adjacency = prev.adjacency.clone();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            // Fast path: edge list already cached — no disk read needed.
            if let Some(edges) = prev.adjacency.get(&hash) {
                for edge in edges {
                    if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                        queue.push_back(*edge);
                    }
                }
                adjacency.insert(hash, edges.clone());
                continue;
            }

            // Slow path: new node, must read from disk.
            let Some(&(shard_id, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&shard_id) else {
                continue;
            };
            let record = pool.read_at(old_shard, offset, true)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        // Restrict adjacency to the actual visited set — entries inherited
        // from prev that are no longer in hash_to_shard_offset (e.g. they
        // were in a shard that was later dropped) must not survive.
        adjacency.retain(|h, _| visited.contains(h) && hash_to_shard_offset.contains_key(h));

        let mut live_hashes: Vec<[u8; 16]> = visited
            .into_iter()
            .filter(|h| hash_to_shard_offset.contains_key(h))
            .collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// Incremental full-adjacency scan (no-roots / preserve-everything path).
    ///
    /// For each hash in `hash_to_shard_offset`, reuses the cached edge list
    /// from `prev.adjacency` when available — avoiding a disk read for any
    /// node that survived the last repack. Only genuinely new hashes (not in
    /// the cache) require a disk read.
    ///
    /// All hashes in `hash_to_shard_offset` are live (no GC in this path),
    /// so the result is always the full map.
    ///
    /// **Memory:** same as [`Self::bfs_live_set_incremental`] — see that doc.
    fn scan_full_adjacency_incremental(
        pool: &ShardPool,
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> =
            HashMap::with_capacity(hash_to_shard_offset.len());

        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            // Fast path: node was live last time, edge list is cached.
            if let Some(edges) = prev.adjacency.get(hash) {
                adjacency.insert(*hash, edges.clone());
                continue;
            }
            // Slow path: new node, must read from disk.
            if let Some(shard) = pinned.get(&shard_id) {
                let record = pool.read_at(shard, offset, true)?;
                let edges = extract_edges(hash, &record.data);
                adjacency.insert(*hash, edges);
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Saves the cursor state a future call to [`Self::repack_scan_incremental`]
    /// needs: each shard's current byte length (the next repack's scan start
    /// point), the deduped live map restricted to hashes that survived into
    /// `new_offsets`, the adjacency cache for those survivors (so the next
    /// repack's BFS/scan can skip re-reading disk for already-seen nodes), and
    /// the roots snapshot (so the next repack can detect whether any root was
    /// removed and fall back to a full BFS sweep if so).
    ///
    /// Anything this repack dropped must not resurface via the carried-forward
    /// `live_map` or adjacency cache next time.
    fn repack_save_incremental_state(
        &self,
        collection_id: &[u8; 16],
        new_offsets: &[([u8; 16], u16, u64)],
        adjacency: HashMap<[u8; 16], Vec<[u8; 16]>>,
        current_roots: &[[u8; 16]],
    ) {
        let surviving: HashSet<[u8; 16]> = new_offsets.iter().map(|&(h, _, _)| h).collect();

        let mut scan_offsets = HashMap::new();
        for (shard_id, shard) in self.shards.all_shards() {
            scan_offsets.insert(shard_id, (shard.pack_id, shard.file_len()));
        }
        let next_live_map: HashMap<[u8; 16], (u16, u64)> = new_offsets
            .iter()
            .map(|&(hash, shard_id, offset)| (hash, (shard_id, offset)))
            .collect();
        // Evict dropped nodes from the adjacency cache so they can't
        // re-enter the live set on a future incremental BFS pass.
        let next_adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = adjacency
            .into_iter()
            .filter(|(h, _)| surviving.contains(h))
            .collect();
        self.repack_incremental.write().insert(
            *collection_id,
            RepackIncrementalState {
                scan_offsets,
                live_map: next_live_map,
                adjacency: next_adjacency,
                prev_roots: current_roots.to_vec(),
            },
        );
    }

    /// Non-mutating preview of what [`Self::repack_collection_reachable`] would
    /// do for `collection_id`: runs the exact same scan + reachability
    /// computation (so kept/dropped counts are exact, not estimates), but
    /// performs no writes, no index swap, and no incremental-repack
    /// cursor update. Used to preview a repack's effect before committing
    /// to it — e.g. a shard-compaction preflight that needs "how many
    /// bytes would survive, and which shards do they currently live in"
    /// without actually rewriting anything.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    pub fn plan_collection_repack(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        // `repack_scan_incremental` moves the cursor out to avoid cloning on
        // real repacks.  A plan must leave that state untouched, and must not
        // race a real repack while temporarily doing so.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        // Make buffered frames visible to the packfile scan so the plan
        // reflects records that have actually been put (not yet flushed).
        self.shards.flush_all()?;
        let saved_state = self.repack_incremental.read().get(collection_id).cloned();
        let result = self
            .repack_scan_incremental(collection_id)
            .and_then(|(map, _)| {
                self.plan_collection_repack_from_map(collection_id, &map, &extract_edges)
            });
        let mut states = self.repack_incremental.write();
        if let Some(state) = saved_state {
            states.insert(*collection_id, state);
        } else {
            states.remove(collection_id);
        }
        result
    }

    fn plan_collection_repack_from_map(
        &self,
        collection_id: &[u8; 16],
        hash_to_shard_offset: &RepackRecordMap,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, _adjacency) = match roots {
            Some(roots) if !roots.is_empty() => Self::bfs_live_set(
                &self.shards,
                &roots,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
            )?,
            _ => Self::scan_full_adjacency(
                &self.shards,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
            )?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
        let mut kept_bytes: u64 = 0;
        let mut dropped_bytes: u64 = 0;
        let mut shards_touched: HashSet<u16> = HashSet::new();
        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&shard_id) {
                // Actual on-disk bytes, not the uncompressed upper bound —
                // a repack preflight should report what will really be
                // reclaimed/kept, which is smaller than plaintext size for
                // any frame that compressed.
                if let Ok(bytes) = Self::record_disk_len_at(shard, offset) {
                    if live_set.contains(hash) {
                        shards_touched.insert(shard_id);
                        kept_bytes = kept_bytes.saturating_add(bytes);
                    } else {
                        dropped_bytes = dropped_bytes.saturating_add(bytes);
                    }
                }
            }
        }

        let mut shards_touched: Vec<u16> = shards_touched.into_iter().collect();
        shards_touched.sort_unstable();
        Ok(RepackPlan {
            kept: live_hashes.len(),
            dropped,
            kept_bytes,
            dropped_bytes,
            shards_touched,
        })
    }

    /// Plan several collection repacks from one physical scan of their shard
    /// closure. Unlike repeatedly calling [`Self::plan_collection_repack`], this
    /// is O(bytes scanned) rather than O(collections × bytes scanned).
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption while scanning or
    /// resolving a requested collection's records.
    pub fn plan_collections_repack(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<RepackPlan>, StorageError> {
        // Planning reads packfiles directly; flush first so buffered records
        // are part of the plan.
        self.shards.flush_all()?;
        let maps = self.scan_collection_record_maps(collection_ids)?;
        collection_ids
            .iter()
            .map(|collection_id| {
                let map = maps.get(collection_id).ok_or_else(|| {
                    StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
                })?;
                self.plan_collection_repack_from_map(collection_id, map, &extract_edges)
            })
            .collect()
    }

    /// Finds every collection and shard transitively reachable from
    /// `start_shard` via the collection↔shard reference relation: collections
    /// referencing this shard, the other shards those collections reference,
    /// the collections referencing *those* shards, and so on until nothing new
    /// is found. In practice this converges immediately in almost every
    /// case — a collection typically has a live footprint on only one or two
    /// shards (sticky home routing) — but the closure has to be computed
    /// rather than assumed, since a collection-scoped repack can't correctly
    /// judge liveness by looking at only one shard's slice of that collection's
    /// data (see [`Self::collections_referencing_shard`]'s doc).
    ///
    /// This is the basis for a shard-compaction preflight: you can't
    /// truthfully say "this touches N collections across K shards" without
    /// first finding the full closure, not just the one shard the
    /// operation was originally pointed at.
    ///
    /// # Errors
    /// Returns `StorageError` if `start_shard` isn't an open shard.
    pub fn repack_closure(
        &self,
        start_shard: u16,
    ) -> Result<(Vec<[u8; 16]>, Vec<u16>), StorageError> {
        let mut collections: HashSet<[u8; 16]> = HashSet::new();
        let mut shards: HashSet<u16> = HashSet::new();
        let mut shard_queue = vec![start_shard];
        let mut collection_queue: Vec<[u8; 16]> = Vec::new();

        loop {
            if let Some(shard_id) = shard_queue.pop() {
                if shards.insert(shard_id) {
                    for collection_id in self.collections_referencing_shard(shard_id)? {
                        if collections.insert(collection_id) {
                            collection_queue.push(collection_id);
                        }
                    }
                }
            } else if let Some(collection_id) = collection_queue.pop() {
                for shard_id in self.collection_referenced_shards(&collection_id) {
                    if !shards.contains(&shard_id) {
                        shard_queue.push(shard_id);
                    }
                }
            } else {
                break;
            }
        }

        let mut collections: Vec<[u8; 16]> = collections.into_iter().collect();
        collections.sort_unstable();
        let mut shards: Vec<u16> = shards.into_iter().collect();
        shards.sort_unstable();
        Ok((collections, shards))
    }

    /// Every shard id `collection_id`'s current index has at least one live
    /// entry pointing into. Empty if the collection doesn't exist.
    #[must_use]
    pub fn collection_referenced_shards(&self, collection_id: &[u8; 16]) -> Vec<u16> {
        let Some(gen) = self.generation(collection_id) else {
            return Vec::new();
        };
        gen.index
            .referenced_shard_ids()
            .iter()
            .enumerate()
            .filter_map(|(i, &referenced)| referenced.then(|| u16::try_from(i).ok()).flatten())
            .collect()
    }

    /// Like `collection_referenced_shards`, but returns the `pack_id`
    /// for each shard rather than the runtime slot index.
    pub fn collection_referenced_pack_ids(&self, collection_id: &[u8; 16]) -> Vec<u64> {
        self.collection_referenced_shards(collection_id)
            .iter()
            .filter_map(|&slot| self.shards.get_shard(slot).map(|s| s.pack_id))
            .collect()
    }

    /// Current indexed-node count for every shard.
    ///
    /// Unlike a physical packfile scan, these counts include only index
    /// entries that are presently live and resolve through a collection's current
    /// generation. Rewritten or superseded append entries are excluded.
    #[must_use]
    pub fn shard_node_counts(&self) -> HashMap<u64, u64> {
        self.shard_collections
            .read()
            .iter()
            .map(|(&shard_id, collections)| {
                let count = collections
                    .values()
                    .copied()
                    .fold(0u64, u64::saturating_add);
                (shard_id, count)
            })
            .collect()
    }

    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    // Four dispatch arms (roots/no-roots × incremental/cold-start) plus root-removal
    // detection make this function long; splitting it would just move the complexity
    // without reducing it.
    #[allow(clippy::too_many_lines)]
    pub fn repack_collection_reachable(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<(usize, usize), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        // The scan below reads packfiles directly, so any records still
        // sitting in a shard's append buffer would be invisible to it and
        // get silently dropped by this repack. Flush first so the whole
        // collection's committed bytes are scan-visible. Holding this
        // collection's put mutex means nothing can buffer behind it for this
        // collection between the flush and the scan.
        self.shards.flush_all()?;

        let (hash_to_shard_offset, prev_state) = self.repack_scan_incremental(collection_id)?;

        // Pin every shard this call could possibly read from before any
        // writes happen — see pin_shards' doc for why this must come
        // first.
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(collection_id).cloned();

        // Root-removal detection: if any root present in the previous repack's
        // snapshot is absent from the current root set, nodes exclusively
        // reachable through that removed root may be stuck in the cached live
        // set indefinitely (the incremental adjacency cache only grows — it has
        // no mechanism to re-derive reachability). Fall back to a full cold-start
        // BFS for this repack so those nodes can actually be collected. Once the
        // full sweep completes, the saved cursor is fresh and subsequent repacks
        // can be incremental again.
        let roots_shrunk = prev_state.as_ref().is_some_and(|prev| {
            let current_root_set: HashSet<[u8; 16]> =
                roots.as_deref().unwrap_or(&[]).iter().copied().collect();
            prev.prev_roots
                .iter()
                .any(|r| !current_root_set.contains(r))
        });
        // Use incremental adjacency only when prev state exists AND no root was removed.
        let use_incremental = prev_state.is_some() && !roots_shrunk;

        let (live_hashes, adjacency) = match (roots.as_deref(), use_incremental) {
            (Some(roots), true) if !roots.is_empty() => {
                // Incremental BFS: only expand from roots not already in the
                // cached live set; reuse cached edge lists for everything else.
                Self::bfs_live_set_incremental(
                    &self.shards,
                    roots,
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            (Some(roots), false) if !roots.is_empty() => {
                // Cold-start BFS: root set shrank or first repack — full walk.
                Self::bfs_live_set(
                    &self.shards,
                    roots,
                    &hash_to_shard_offset,
                    &pinned,
                    &extract_edges,
                )?
            }
            (_, true) => {
                // No live roots, incremental path: only read disk for new nodes.
                Self::scan_full_adjacency_incremental(
                    &self.shards,
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            _ => {
                // No live roots, cold-start: preserve everything, read all.
                Self::scan_full_adjacency(
                    &self.shards,
                    &hash_to_shard_offset,
                    &pinned,
                    &extract_edges,
                )?
            }
        };

        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        // Evict garbage-collected hashes from the collection's cache. Repack
        // carries the same cache forward into the new generation (below);
        // without this, a stale cache hit could still serve bytes for a
        // hash this pass just decided is unreachable, defeating GC.
        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        let mut hex = String::with_capacity(32);
                        for b in hash {
                            let _ = write!(hex, "{b:02x}");
                        }
                        eprintln!("  dropped: {hex}");
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "repack: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        // Copy into a non-source destination shard, never back into the
        // collection's current generation. Once the new index is published below,
        // this lets old source shards retire when no other collection references
        // them.
        let source_shards: HashSet<u16> = hash_to_shard_offset
            .values()
            .map(|&(shard_id, _)| shard_id)
            .collect();
        self.shards
            .prepare_collection_repack(collection_id, &source_shards)?;

        let mut new_offsets: Vec<([u8; 16], u16, u64)> = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some(entry) =
                self.copy_record_to_shard(collection_id, &pinned, old_shard_id, old_offset)?
            {
                new_offsets.push(entry);
            }
        }

        let kept = new_offsets.len();

        let index = Self::build_index(&new_offsets)?;
        self.replace_collection_shard_counts(
            collection_id,
            &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
        );
        self.swap_generation(collection_id, index)?;
        self.retire_empty_shards(collection_id);

        self.repack_count.fetch_add(1, Ordering::Relaxed);
        self.repack_kept_total
            .fetch_add(kept as u64, Ordering::Relaxed);
        self.repack_dropped_total
            .fetch_add(dropped as u64, Ordering::Relaxed);
        self.repack_counts_by_collection
            .write()
            .entry(*collection_id)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);

        let current_roots: Vec<[u8; 16]> = roots.as_deref().unwrap_or(&[]).to_vec();
        self.repack_save_incremental_state(collection_id, &new_offsets, adjacency, &current_roots);

        // Fsync the shards this repack just wrote into and persist the
        // updated stats snapshot as part of finishing the repack, rather
        // than leaving both to whatever the next unrelated sync_dirty()
        // call happens to be — a crash right after a repack should not
        // lose durability for data this repack itself just wrote, nor
        // leave the persisted stats stale relative to what's on disk.
        self.shards.sync_dirty()?;

        Ok((kept, dropped))
    }

    /// Repack several collections as one physical shard-compaction batch.
    ///
    /// The input is scanned once, then all replacement records are written
    /// through one shared destination stream. This is O(bytes + nodes), not
    /// O(collections × bytes), and avoids leaving one partially-filled destination
    /// shard behind for each collection.
    ///
    /// # Errors
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        self.repack_collections_reachable_with_progress(
            collection_ids,
            extract_edges,
            |_collection, _from, _to, _nodes, _complete| {},
        )
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// This lets an interactive caller show meaningful physical-compaction
    /// progress without turning the shared shard batch back into a
    /// collection-at-a-time operation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    fn report_repack_output_rotation(
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        active_output_shard: &mut u16,
        current_active_shard: u16,
        collection_id: &[u8; 16],
        copied_nodes: usize,
    ) {
        if current_active_shard != *active_output_shard {
            output_progress(
                Some(*collection_id),
                *active_output_shard,
                current_active_shard,
                copied_nodes,
                false,
            );
            *active_output_shard = current_active_shard;
        }
    }

    fn copy_repack_batch_collection(
        &self,
        collection_id: &[u8; 16],
        hash_to_shard_offset: &RepackRecordMap,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        output_state: &mut (u16, usize),
    ) -> Result<(RepackOffsets, usize), StorageError> {
        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, adjacency) = match roots {
            Some(roots) if !roots.is_empty() => Self::bfs_live_set(
                &self.shards,
                &roots,
                hash_to_shard_offset,
                pinned,
                extract_edges,
            )?,
            _ => Self::scan_full_adjacency(
                &self.shards,
                hash_to_shard_offset,
                pinned,
                extract_edges,
            )?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();
        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "repack: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        let mut new_offsets = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some(entry) =
                self.copy_record_to_shard(collection_id, pinned, old_shard_id, old_offset)?
            {
                output_state.1 = output_state.1.saturating_add(1);
                let current_active_shard = self.shards.active_shard().slot;
                Self::report_repack_output_rotation(
                    output_progress,
                    &mut output_state.0,
                    current_active_shard,
                    collection_id,
                    output_state.1,
                );
                new_offsets.push(entry);
            }
        }

        Ok((new_offsets, dropped))
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable_with_progress(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        mut output_progress: impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let mut collection_ids = collection_ids.to_vec();
        collection_ids.sort_unstable();
        collection_ids.dedup();
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        // A batch is one coherent generation change. Hold every selected
        // collection's writer mutex in a stable order so a concurrent put cannot
        // be missed by the scan or revive an old source after the swap.
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let collection_guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();

        // Same scan visibility requirement as the single-collection repack:
        // every selected collection's buffered frames must be on disk before
        // `scan_collection_record_maps` reads packfiles, or those records
        // would be dropped as if never written.
        self.shards.flush_all()?;

        let maps = self.scan_collection_record_maps(&collection_ids)?;
        let source_shards: HashSet<u16> = maps
            .values()
            .flat_map(|map| map.values().map(|&(shard_id, _)| shard_id))
            .collect();
        let pinned = self.pin_shards(source_shards.iter().copied());

        // Establish one common destination before copying the first frame.
        // `put_record` moves every collection to the same active successor when a
        // destination fills, so output remains densely packed across collections.
        self.shards
            .prepare_collections_repack(&collection_ids, &source_shards)?;

        let mut results = Vec::with_capacity(collection_ids.len());
        let mut output_state = (self.shards.active_shard().slot, 0usize);
        for collection_id in &collection_ids {
            let hash_to_shard_offset = maps.get(collection_id).ok_or_else(|| {
                StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
            })?;
            let (new_offsets, dropped) = self.copy_repack_batch_collection(
                collection_id,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
                &mut output_progress,
                &mut output_state,
            )?;
            let kept = new_offsets.len();
            let index = Self::build_index(&new_offsets)?;
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
            );
            self.swap_generation(collection_id, index)?;
            // The batch path does a full cold-start scan (scan_collection_record_maps),
            // so we have no incremental adjacency to carry forward. Pass an empty
            // map so the next single-collection repack starts with a fresh cache
            // (it will populate it from the full BFS on that first pass).
            let batch_roots: Vec<[u8; 16]> = self
                .live_roots
                .read()
                .get(collection_id)
                .cloned()
                .unwrap_or_default();
            self.repack_save_incremental_state(
                collection_id,
                &new_offsets,
                HashMap::new(),
                &batch_roots,
            );

            self.repack_count.fetch_add(1, Ordering::Relaxed);
            self.repack_kept_total
                .fetch_add(kept as u64, Ordering::Relaxed);
            self.repack_dropped_total
                .fetch_add(dropped as u64, Ordering::Relaxed);
            self.repack_counts_by_collection
                .write()
                .entry(*collection_id)
                .and_modify(|count| *count = count.saturating_add(1))
                .or_insert(1);
            results.push((*collection_id, kept, dropped));
        }

        output_progress(None, output_state.0, output_state.0, output_state.1, true);

        // All target collection locks remain held while source references are
        // replaced. Drop them before the general retirement protocol, which
        // obtains every collection lock itself.
        drop(collection_guards);
        self.retire_empty_shards_after_batch();
        self.shards.sync_dirty()?;
        Ok(results)
    }

    /// Fetches the ancestors of `frontier`, walking `extract_edges`
    /// backward and stopping expansion at (and excluding) anything in
    /// `stop_at` — a bounded ancestor walk, e.g. for auth-chain or
    /// prev-event traversal from a caller-supplied frontier back to
    /// caller-supplied already-known boundary nodes.
    ///
    /// This is **not** a from/to range or span: if a branch's history
    /// never crosses any `stop_at` node (a fork off the main line, say),
    /// that branch is walked all the way back to the collection's true roots,
    /// not truncated at an implied boundary. A genuine "everything between
    /// these two frontiers" query is `ancestors(frontier) ∩
    /// descendants(stop_at)`, which needs reverse adjacency this doesn't
    /// build — `NodeId`s are content hashes with no key order to slice a
    /// real range query over, and this walk only covers the ancestor half.
    /// `limits.max_nodes` bounds the runaway case above.
    ///
    /// Returns nodes lazily, ancestor-first (parents before children,
    /// topologically), as a [`DagWalk`] iterator over data already
    /// resolved during the walk itself (no second per-node fetch against
    /// whatever generation happens to be live when the iterator is
    /// drained) — every record was matched by content hash, not just
    /// index tag, so an index-tag collision can't surface the wrong one.
    ///
    /// Read-only: resolves against one frozen generation snapshot via its
    /// internal breadth-first traversal and takes no `put_mutex`. Cost is
    /// proportional to the number of ancestors actually walked — one
    /// index lookup per node — not collection size, since it doesn't pre-scan
    /// every shard the way `repack_collection_reachable` does.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption, or if the walk finds a
    /// cycle (an event graph must be acyclic; a cycle means the extracted
    /// edges are wrong or the data is corrupt).
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    pub fn walk_ancestors(
        &self,
        collection_id: &[u8; 16],
        frontier: &[[u8; 16]],
        stop_at: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        limits: WalkLimits,
    ) -> Result<DagWalk, StorageError> {
        let Some(gen) = self.generation(collection_id).map(|g| Arc::clone(&g)) else {
            return Ok(DagWalk {
                order: Vec::new().into_iter(),
                resolved: HashMap::new(),
            });
        };

        let ((live_hashes, adjacency), resolved) =
            self.bfs_ancestors(&gen, frontier, stop_at, &extract_edges, limits)?;

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let mut topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "walk_ancestors: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        // `extract_edges` points each node at its parents, so Kahn's
        // algorithm (which surfaces zero-incoming nodes first) naturally
        // yields children before parents. Reverse it: callers of an
        // ancestor walk want parents-before-children, the order data
        // could actually be replayed in.
        topo.reverse();

        let order: Vec<[u8; 16]> = topo
            .into_iter()
            .map(|local| {
                *csr.hash_of(local)
                    .expect("topo order contains valid local IDs")
            })
            .collect();

        Ok(DagWalk {
            order: order.into_iter(),
            resolved,
        })
    }

    /// After a repack, scan all collections' indexes and retire any shard that
    /// no collection references.
    ///
    /// Acquires every collection's `put_mutex` (in sorted order, skipping the
    /// caller's already-held lock) to prevent a concurrent `put()` from
    /// landing a write in a shard whose index entry hasn't been inserted
    /// yet — without that, the scan could see zero references to a shard
    /// that a writer just committed bytes to but hasn't index-updated yet,
    /// causing a live shard to be retired under it.
    fn retire_empty_shards(&self, held_collection: &[u8; 16]) {
        // Collect collection IDs in sorted order for deadlock-free lock acquisition.
        // Skip the collection whose put_mutex the caller already holds.
        let mut collections_to_lock: Vec<[u8; 16]> = self
            .collections
            .read()
            .keys()
            .filter(|id| *id != held_collection)
            .copied()
            .collect();
        collections_to_lock.sort_unstable();

        // Acquire all other collections' put_mutexes. The sorted order prevents
        // deadlocks; the held collection is skipped (parking_lot is non-reentrant).
        let mutexes: Vec<_> = collections_to_lock
            .iter()
            .map(|id| self.put_mutex(id))
            .collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();

        self.retire_empty_shards_locked();
    }

    /// Retire unreferenced shards after a batch has released all collection locks.
    /// This takes every collection lock itself, unlike [`Self::retire_empty_shards`]
    /// which is called while one particular collection lock is already held.
    fn retire_empty_shards_after_batch(&self) {
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();
        self.retire_empty_shards_locked();
    }

    /// Every collection writer lock is held by the caller.
    fn retire_empty_shards_locked(&self) {
        // Do not retain the map guard while acquiring collection locks: another
        // operation may need the map lock while it holds a collection lock.
        let collections = self.collections.read();

        // Build the union of shard IDs referenced across all collections.
        let mut referenced = [false; shard::MAX_SHARDS];
        for gen_swap in collections.values() {
            let gen = gen_swap.load();
            let ids = gen.index.referenced_shard_ids();
            for (i, &has_refs) in ids.iter().enumerate() {
                referenced[i] |= has_refs;
            }
        }

        // Retire any shard not in the union.
        for id in 0..u16::try_from(shard::MAX_SHARDS).unwrap() {
            if !referenced[id as usize] {
                self.shards.retire_slot(id);
            }
        }
    }
}

impl StorageEngine for PackfileStorage {
    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let track = self.stats_enabled.load(Ordering::Relaxed);
        let gen_guard = self.generation(collection_id);
        let gen = gen_guard.as_deref();

        let Some(gen) = gen else {
            if track {
                self.get_calls.fetch_add(1, Ordering::Relaxed);
                self.get_misses.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(None);
        };
        if let Some(data) = gen.cache.get(id) {
            if track {
                self.get_calls.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(Some((*data).clone()));
        }
        let result = self.resolve_from_candidates(id, gen.index.lookup_all(id));
        if track {
            self.get_calls.fetch_add(1, Ordering::Relaxed);
            if result.as_ref().is_ok_and(Option::is_none) {
                self.get_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }

    fn get_many(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let track = self.stats_enabled.load(Ordering::Relaxed);
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

        let gen_guard = self.generation(collection_id);
        let gen = gen_guard.as_deref();

        let mut to_fetch: Vec<(usize, Vec<(u16, u64)>)> = Vec::new();
        if let Some(g) = gen {
            for (i, id) in ids.iter().enumerate() {
                // The generation-local cache only contains records from this
                // generation, so it is authoritative for hits. Avoid an
                // otherwise redundant index probe for the common warm case.
                if let Some(data) = g.cache.get(id) {
                    results[i] = Some((*data).clone());
                    continue;
                }
                let candidates: Vec<(u16, u64)> = g.index.lookup_all(id).collect();
                if !candidates.is_empty() {
                    to_fetch.push((i, candidates));
                }
            }
        }

        to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

        for (i, candidates) in &to_fetch {
            results[*i] = self.resolve_from_candidates(&ids[*i], candidates.iter().copied())?;
        }

        if track {
            self.get_many_calls.fetch_add(1, Ordering::Relaxed);
            self.get_many_records
                .fetch_add(ids.len() as u64, Ordering::Relaxed);
            self.get_many_misses.fetch_add(
                results.iter().filter(|r| r.is_none()).count() as u64,
                Ordering::Relaxed,
            );
        }

        Ok(results)
    }

    fn put(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
    ) -> Result<(), StorageError> {
        self.put_calls.fetch_add(1, Ordering::Relaxed);
        self.put_bytes
            .fetch_add(data.bytes.len() as u64, Ordering::Relaxed);
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        // New-collection puts must not interleave their record flush with a
        // concurrent checkpoint's fingerprint→snapshot window (see
        // `persist_index_checkpoint`). Existing collections are already gated
        // by their `put_mutex`; a brand-new one isn't in that set yet, so we
        // hold the publication lock shared for the whole put instead — the
        // checkpoint holds it exclusive, excluding this put from the window.
        let _create_guard = if self.generation(collection_id).is_none() {
            Some(self.collection_creation.read())
        } else {
            None
        };

        let record = Record {
            collection_id: *collection_id,
            hash: *id,
            data: data.bytes.clone(),
        };
        let (shard_id, offset) = self.shards.put_record(&record)?;
        if let Some(gen) = self.generation(collection_id) {
            if !gen.index.is_mmap_backed() {
                if let Ok((bucket, slot)) =
                    self.insert_index(collection_id, &gen.index, id, shard_id, offset)?
                {
                    self.record_delta(collection_id, gen.generation, bucket, slot);
                    let pack_id = self
                        .shards
                        .get_shard(shard_id)
                        .map_or(u64::from(shard_id), |shard| shard.pack_id);
                    self.record_new_shard_collection(pack_id, collection_id);

                    let mut data_to_cache = data.clone();
                    for child in &mut data_to_cache.children {
                        if let NodeRef::Lazy(child_id) = child {
                            if let Some(child_data) = self.pinned.get(child_id) {
                                *child = NodeRef::Resolved(*child_id, child_data);
                            }
                        }
                    }
                    gen.cache.insert(*id, Arc::new(data_to_cache));
                    self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
                    return Ok(());
                }
            }
        }
        let (index, cache) = {
            let old_gen = self.generation(collection_id);
            let mut index = match &old_gen {
                Some(g) => g.index.clone(),
                // A brand-new collection's first insert: no count hint to
                // size from (unlike `put_many`, which knows `entries.len()`),
                // so start at a small fixed floor and let `grow()` size it
                // up as real entries land. Previously this was a flat
                // 4096-slot (32 KiB) floor per collection regardless of how
                // many records it would ever hold -- workloads with many
                // small/near-empty collections (e.g. per-room collections
                // holding a handful of records each) paid that 32 KiB up
                // front per collection, dwarfing the actual data by orders
                // of magnitude.
                //
                // See `NEW_COLLECTION_INDEX_FLOOR` for why this isn't the
                // index's own smaller hard floor.
                None => LossyIndex::new(NEW_COLLECTION_INDEX_FLOOR),
            };
            let inserted = self.insert_index(collection_id, &index, id, shard_id, offset)?;
            if let Ok((bucket, slot)) = inserted {
                let generation = old_gen.as_ref().map_or(1, |g| g.generation);
                self.record_delta(collection_id, generation, bucket, slot);
                let pack_id = self
                    .shards
                    .get_shard(shard_id)
                    .map_or(u64::from(shard_id), |s| s.pack_id);
                self.record_new_shard_collection(pack_id, collection_id);
            } else {
                // The insert was rejected (table full). The collection's shape
                // is about to change, so any delta frames would no longer be
                // replayable against a checkpoint at this generation.
                self.invalidate_delta_log();
                if let Some(grown) = index.grow() {
                    // The failed insert did not mutate the table, so retry it
                    // after the in-memory rehash. This is the normal capacity
                    // path and must not turn into a full-pack scan.
                    let _ = grown.insert(id, shard_id, offset);
                    index = grown;
                } else if let Ok(Some(grown)) = self.grow_checkpoint_index(collection_id, &index) {
                    // A checkpoint-backed index has locations but not homes.
                    // Recovering the hashes from those locations is bounded
                    // by this collection, unlike `rebuild_index`'s pack scan.
                    let _ = grown.insert(id, shard_id, offset);
                    index = grown;
                } else {
                    index = self.rebuild_index(collection_id)?;
                    let _ = index.insert(id, shard_id, offset);
                }
                // The rebuild re-derived the collection's entire live set from
                // scratch, so its shard distribution needs a full
                // recompute too, not just crediting this one record.
                self.replace_collection_shard_counts(
                    collection_id,
                    &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
                );
            }
            let cache = match &old_gen {
                Some(g) => g.cache.clone(),
                None => Arc::new(NodeCache::new(self.cache_capacity)),
            };

            let mut data_to_cache = data.clone();
            for child in &mut data_to_cache.children {
                if let NodeRef::Lazy(child_id) = child {
                    if let Some(child_data) = self.pinned.get(child_id) {
                        *child = NodeRef::Resolved(*child_id, child_data);
                    }
                }
            }
            cache.insert(*id, Arc::new(data_to_cache));

            (index, cache)
        };

        self.store_generation(collection_id, index, Some(cache), false)?;

        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the live-index fast path and its clone-on-demand fallback read clearest \
                  kept together rather than split across helpers that would each need most \
                  of the same state (old_gen, owned_index, structural_change) threaded through"
    )]
    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        if entries.is_empty() {
            return Ok(());
        }

        self.put_many_calls.fetch_add(1, Ordering::Relaxed);
        self.put_many_records
            .fetch_add(entries.len() as u64, Ordering::Relaxed);
        self.put_many_bytes.fetch_add(
            entries
                .iter()
                .map(|(_, data)| data.bytes.len() as u64)
                .sum::<u64>(),
            Ordering::Relaxed,
        );

        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        // Same comment as in `put`: a brand-new collection must be excluded
        // from a concurrent checkpoint's fingerprint→snapshot window.
        let _create_guard = if self.generation(collection_id).is_none() {
            Some(self.collection_creation.read())
        } else {
            None
        };

        let old_gen = self.generation(collection_id);
        let generation = old_gen.as_ref().map_or(1, |g| g.generation);
        let cache = match &old_gen {
            Some(g) => g.cache.clone(),
            None => Arc::new(NodeCache::new(self.cache_capacity)),
        };

        // Fast path, mirroring `put`'s single-record optimization (see its
        // comment): as long as the collection already has an owned,
        // materialized index — not the very first write since a checkpoint
        // reopen — every record in this batch that fits without growth
        // mutates that live, already-published index directly. `owned_index`
        // stays `None` for as long as that holds, so the O(capacity) clone
        // below never runs and no new generation ever gets published —
        // exactly `put`'s in-place success path, just looped over the
        // batch. It is materialized, once, only when a structural change is
        // actually required: growth, the first write since a checkpoint
        // reopen (mmap-backed), or a brand-new collection. A clone taken
        // partway through the batch (a mid-batch grow) still captures every
        // record already applied via the live path, since those mutations
        // landed on the very object being cloned.
        let materialize_started = std::time::Instant::now();
        let mut owned_index: Option<LossyIndex> = match &old_gen {
            Some(g) if !g.index.is_mmap_backed() => None,
            Some(g) => Some(g.index.clone()),
            // Unlike `put`'s single-record path, we know exactly how many
            // records this brand-new collection is about to receive -- size
            // from that instead of the old flat 4096-slot (32 KiB) floor,
            // matching the `records.len().saturating_mul(2).max(16)` pattern
            // used elsewhere in this file (e.g. checkpoint/pack rebuild).
            // Floor is 64, not 16 -- see the matching comment on `put`'s
            // `None` arm above: a small batch (e.g. a single-event
            // `put_many` call) would otherwise still hit `grow()` almost
            // immediately, invalidating the delta log and forcing a full
            // checkpoint rewrite on the next sync.
            None => Some(LossyIndex::new(
                entries
                    .len()
                    .saturating_mul(2)
                    .max(NEW_COLLECTION_INDEX_FLOOR),
            )),
        };
        // Batch-granular clone accounting: a call that materialized an owned
        // index up front (first write since a checkpoint reopen, or a
        // brand-new collection) pays the O(capacity) copy here; one that
        // starts on the live no-clone path does not. `stats()` reports both
        // counts and the cumulative `index_clone_time`.
        if owned_index.is_some() {
            self.put_many_clone_path_calls
                .fetch_add(1, Ordering::Relaxed);
            self.index_clone_time_ns.fetch_add(
                u64::try_from(materialize_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        } else {
            self.put_many_fast_path_calls
                .fetch_add(1, Ordering::Relaxed);
        }
        // Whether this call must publish a new generation at all. Starts
        // true exactly when `owned_index` already had to be materialized
        // above; flips true the moment a mid-batch grow/rebuild happens.
        let mut structural_change = owned_index.is_some();
        let mut index_needs_rebuild = false;

        for (id, data) in entries {
            let record = Record {
                collection_id: *collection_id,
                hash: *id,
                data: data.bytes.clone(),
            };

            let (shard_id, offset) = self.shards.put_record(&record)?;

            if index_needs_rebuild {
                continue;
            }

            // Whichever index is authoritative for this insert: the owned
            // copy once one exists, else the collection's live, still-shared
            // index — only reachable here when it's neither mmap-backed nor
            // absent, both of which already forced `owned_index` above.
            let live: &LossyIndex = match owned_index.as_ref() {
                Some(index) => index,
                None => {
                    &old_gen
                        .as_ref()
                        .expect("live path implies an existing generation")
                        .index
                }
            };

            let inserted = if let Ok((bucket, slot)) =
                self.insert_index(collection_id, live, id, shard_id, offset)?
            {
                self.record_delta(collection_id, generation, bucket, slot);
                true
            } else {
                // Growth (either flavor) materializes a fresh owned index; the
                // O(n) copy is exactly what the steady-append scaler tracks,
                // so bag its time alongside the up-front materialization.
                let grow_started = std::time::Instant::now();
                let grown = if let Some(grown) = live.grow() {
                    // `insert_tracked` leaves the table unchanged on
                    // TableFull, so retrying with the record's
                    // still-available location is sufficient; no pack scan is
                    // needed for pure growth. The grow changes the
                    // collection's shape, so the delta log is invalidated and
                    // no frame is recorded for this (or any later)
                    // overwrite in the batch.
                    self.invalidate_delta_log();
                    structural_change = true;
                    grown
                } else if let Ok(Some(grown)) = self.grow_checkpoint_index(collection_id, live) {
                    // The checkpoint does not retain homes, but its slots
                    // retain locations. Recover only those identities and
                    // retry; do not scan every pack in the store.
                    self.invalidate_delta_log();
                    structural_change = true;
                    grown
                } else {
                    index_needs_rebuild = true;
                    structural_change = true;
                    self.invalidate_delta_log();
                    continue;
                };
                self.index_grow_count.fetch_add(1, Ordering::Relaxed);
                self.index_clone_time_ns.fetch_add(
                    u64::try_from(grow_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                let inserted = grown.insert(id, shard_id, offset).is_ok();
                owned_index = Some(grown);
                inserted
            };
            if inserted {
                let pack_id = self
                    .shards
                    .get_shard(shard_id)
                    .map_or(u64::from(shard_id), |s| s.pack_id);
                self.record_new_shard_collection(pack_id, collection_id);
            }
        }

        if index_needs_rebuild {
            let rebuilt = self.rebuild_index(collection_id)?;
            // rebuild_index automatically discovers all the records we just appended
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&rebuilt.shard_counts()),
            );
            owned_index = Some(rebuilt);
        }

        // Apply cache mutations only after all disk writes succeed, so a
        // failed batch does not leak partial state into the shared cache.
        // Resolve and insert one entry at a time: retaining a prepared clone
        // of the entire batch here would defeat the cache's size bound.
        for (id, data) in entries {
            let mut data_to_cache = data.clone();
            for child in &mut data_to_cache.children {
                if let NodeRef::Lazy(child_id) = child {
                    if let Some(child_data) = self.pinned.get(child_id) {
                        *child = NodeRef::Resolved(*child_id, child_data);
                    }
                }
            }
            cache.insert(*id, Arc::new(data_to_cache));
        }

        if structural_change {
            let index = owned_index
                .expect("structural_change is only set once owned_index is materialized");
            self.store_generation(collection_id, index, Some(cache), false)?;
        } else {
            // Every record landed on the live, already-published index in
            // place -- no new generation to publish, matching `put`'s
            // in-place success path.
            self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }

        Ok(())
    }

    fn delete_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        // Acquire the collection's put mutex to serialize with any in-flight put,
        // preventing a concurrent put from resurrecting the collection after we
        // remove it from the generation map.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        self.collections.write().remove(collection_id);
        self.live_roots.write().remove(collection_id);
        // A deletion changes the collection directory the same way a refill
        // does, so any pending delta frames are no longer replayable against
        // the checkpoint they were recorded against.
        self.invalidate_delta_log();
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        // Keep lock entries for the storage lifetime. Removing an entry while
        // a caller still owns its Arc permits a later put to obtain a second
        // mutex and bypass this deletion's serialization.
        self.remove_collection_shard_counts(collection_id);
        self.persist_deleted_collection(collection_id)?;
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        let started = std::time::Instant::now();
        let mut timings = SyncTimings::default();
        self.shards.sync_dirty()?;
        if let Some((flush, fsync)) = self.shards.last_sync_split() {
            timings.pack_flush = flush;
            timings.pack_fsync = fsync;
        }
        self.persist_index_checkpoint_or_delta(&mut timings);
        timings.total = started.elapsed();
        self.count_sync_persistence(&timings);
        *self.last_sync_timings.lock() = Some(timings);
        Ok(())
    }

    fn refresh_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        Self::refresh_collection(self, collection_id)
    }
}

impl PackfileStorage {
    /// Sync all open shards to disk (full pool, not just dirty).
    ///
    /// Also persists the shard→collection directory as a side effect, same as
    /// `ShardPool::sync_all` persists shard IO stats — an explicit sync is
    /// a natural point to flush this observability data too.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure.
    pub fn sync_all(&self) -> Result<(), StorageError> {
        let started = std::time::Instant::now();
        let mut timings = SyncTimings::default();
        self.shards.sync_all()?;
        if let Some((flush, fsync)) = self.shards.last_sync_split() {
            timings.pack_flush = flush;
            timings.pack_fsync = fsync;
        }
        let sidecar_started = std::time::Instant::now();
        self.persist_shard_collections_best_effort();
        timings.sidecar = sidecar_started.elapsed();
        self.persist_index_checkpoint_or_delta(&mut timings);
        timings.total = started.elapsed();
        self.count_sync_persistence(&timings);
        *self.last_sync_timings.lock() = Some(timings);
        Ok(())
    }

    /// Batch-granular sync accounting: every sync counts once, and the
    /// checkpoint-vs-delta discriminator comes from which phase
    /// `persist_index_checkpoint_or_delta` actually ran (a non-dirty sync runs
    /// neither). `sync` (dirty-scoped) and `sync_all` both funnel through here.
    fn count_sync_persistence(&self, timings: &SyncTimings) {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        if !timings.checkpoint.is_zero() {
            self.checkpoint_writes.fetch_add(1, Ordering::Relaxed);
        } else if !timings.delta_log.is_zero() {
            self.delta_appends.fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Whether the pending mutation set must be persisted as a full index
    /// checkpoint rather than a delta append. A delta append can only cover a
    /// log that has been continued continuously from the last checkpoint: any
    /// structural invalidation, a missing log base, an empty frame set, or a
    /// log grown past `DELTA_LOG_CAP_BYTES` breaks that chain — the delta
    /// path would either replay garbage or never terminate (grow forever).
    fn delta_state_needs_full_rewrite(&self) -> bool {
        let state = self.delta_state.lock();
        if state.invalid || state.base_fingerprint.is_none() || state.pending.is_empty() {
            return true;
        }
        let Some(batch_bytes) = delta::batch_len(state.pending.len()) else {
            return true;
        };
        state
            .log_bytes
            .saturating_add(u64::try_from(batch_bytes).unwrap_or(u64::MAX))
            .saturating_add(if state.log_bytes == 0 {
                DELTA_LOG_HEADER_LEN as u64
            } else {
                0
            })
            > DELTA_LOG_CAP_BYTES
    }

    /// Persist the dirty index state for a sync barrier — a delta append when
    /// the log can be continued, otherwise a full checkpoint rewrite — and
    /// record which path ran in `timings`. No-op when nothing is dirty.
    fn persist_index_checkpoint_or_delta(&self, timings: &mut SyncTimings) {
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            return;
        }
        if self.delta_state_needs_full_rewrite() {
            let checkpoint_started = std::time::Instant::now();
            self.persist_index_checkpoint_best_effort();
            timings.checkpoint = checkpoint_started.elapsed();
            return;
        }
        // The frames were recorded against the pack set the preceding flush
        // made durable, so the tail fingerprint is computed after that flush.
        let tail_fingerprint = self.current_pack_fingerprint();
        let delta_started = std::time::Instant::now();
        if let Err(error) = self.append_index_delta(tail_fingerprint) {
            // An append failure took the frames with it, so the pending state
            // no longer reflects the live indexes. Fall back to a full
            // rewrite rather than leaving the acceleration files stale until
            // the next sync notices the gap.
            eprintln!("mtxdb: delta log append failed, rewriting checkpoint: {error}");
            let checkpoint_started = std::time::Instant::now();
            self.persist_index_checkpoint_best_effort();
            timings.checkpoint = checkpoint_started.elapsed();
        } else {
            timings.delta_log = delta_started.elapsed();
        }
    }

    /// Fingerprint of the current on-disk pack set, computed from each
    /// shard's committed length. Must be called after the flush that made the
    /// bytes corresponding to outstanding index frames durable, so the file
    /// lengths reflect exactly what the indexes' offsets point at.
    fn current_pack_fingerprint(&self) -> u64 {
        let packs: Vec<(u64, u64)> = self
            .shards
            .all_shards()
            .into_iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        crate::index::checkpoint::pack_fingerprint(&packs)
    }

    /// Wall-clock breakdown of this store's most recent `open`, including
    /// which index-loading path was taken. Set at construction; every store
    /// returned by a `PackfileStorage::open*` constructor carries it.
    #[must_use]
    pub fn open_timings(&self) -> Option<OpenTimings> {
        *self.last_open_timings.lock()
    }

    /// Wall-clock breakdown of the most recent sync — `sync()`, `sync_all`,
    /// or the bench's internal sync calls — by phase.
    #[must_use]
    pub fn sync_timings(&self) -> Option<SyncTimings> {
        *self.last_sync_timings.lock()
    }

    /// Commit every open shard's buffered frames to the page cache without
    /// fsyncing (see [`ShardPool::flush_all`]). No-op under the default
    /// [`crate::shard::AppendPolicy::Eager`], where every put already wrote
    /// its own frame.
    ///
    /// Under [`crate::shard::AppendPolicy::Buffered`], a put's bytes live
    /// only in process memory until a flush, and only in the page cache
    /// until a `sync_all`. This is the explicit way to advance that boundary
    /// for all shards. A flushed store also means a subsequent
    /// `PackfileStorage::open`/`open_read_only` against the same directory
    /// sees the data.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure.
    pub fn flush_all(&self) -> Result<(), StorageError> {
        Ok(self.shards.flush_all()?)
    }

    /// Replace this store's append policy — whether frames go to disk per
    /// put (the default [`crate::shard::AppendPolicy::Eager`], historical
    /// behavior) or accumulate for one positioned write per flush
    /// ([`crate::shard::AppendPolicy::Buffered`]). See
    /// [`crate::shard::AppendPolicy`] for the visibility/durability trade.
    ///
    /// Only affects future puts, so call it before the store starts handling
    /// concurrent writes. For a batch import that ends in `sync_all` — or a
    /// benchmark that syncs explicitly — `Buffered` usually wins:
    ///
    /// ```
    /// use mtxdb_core::shard::AppendPolicy;
    /// use mtxdb_core::PackfileStorage;
    /// # let dir = std::env::temp_dir().join("mtxdb-doc-with-append-policy");
    /// let store = PackfileStorage::open(dir.clone())
    ///     .unwrap()
    ///     .with_append_policy(AppendPolicy::buffered());
    /// # std::fs::remove_dir_all(dir).ok();
    /// ```
    #[must_use]
    pub fn with_append_policy(mut self, policy: shard::AppendPolicy) -> Self {
        self.shards.set_append_policy(policy);
        self
    }

    /// Best-effort, rate-limited flush of the shard→collection directory for a
    /// writer's own periodic tick — same contract as
    /// `ShardPool::maybe_persist_stats`: a no-op if called again before
    /// `min_interval` has passed since the last flush from here.
    pub fn maybe_persist_shard_collections(&self, min_interval: std::time::Duration) {
        let now = std::time::Instant::now();
        {
            let mut last = self.last_shard_collections_flush.write();
            if last.is_some_and(|prev| now.duration_since(prev) < min_interval) {
                return;
            }
            *last = Some(now);
        }
        self.persist_shard_collections_best_effort();
    }

    /// Snapshot IO/sync stats for every currently-open shard.
    ///
    /// Shards are shared across collections, so this is per-shard, not per-collection.
    #[must_use]
    pub fn shard_stats(&self) -> Vec<(u16, crate::shard::ShardStats)> {
        self.shards.all_stats()
    }

    /// Get IO/sync stats for a single shard by ID.
    #[must_use]
    pub fn shard_stats_for(&self, shard_id: u16) -> Option<crate::shard::ShardStats> {
        self.shards.stats(shard_id)
    }

    /// Total number of shards retired (garbage-collected after a repack)
    /// over this storage's lifetime.
    #[must_use]
    pub fn shards_retired(&self) -> u64 {
        self.shards.retired_count()
    }

    /// List every currently-open shard with basic size, generation, and
    /// IO/sync stats — the data behind a `shards` CLI listing.
    ///
    /// This needs no collection data at all; a caller that only wants shard-level
    /// info (e.g. a `shards`-only inspection tool that must coexist with a
    /// live writer) should call `ShardPool::summaries` directly instead of
    /// opening a full `PackfileStorage`, which rebuilds every collection's index.
    #[must_use]
    pub fn shard_summaries(&self) -> Vec<shard::ShardSummary> {
        self.shards.summaries()
    }

    /// Snapshot global repack stats across all collections.
    #[must_use]
    pub fn repack_stats(&self) -> RepackStats {
        RepackStats {
            repack_count: self.repack_count.load(Ordering::Relaxed),
            kept_total: self.repack_kept_total.load(Ordering::Relaxed),
            dropped_total: self.repack_dropped_total.load(Ordering::Relaxed),
        }
    }

    /// Opt in to the logical read-path counters (`get`/`get_many`). Read
    /// counters are off by default so the read hot path pays a single relaxed
    /// load per logical read at most (never a per-record `fetch_add`); write,
    /// batch, and sync counters are always on regardless.
    ///
    /// Changing the flag is safe at any point and affects only subsequent
    /// reads; counters are monotone, so reads disabled by a false-to-true
    /// transition are reflected as fewer `get_*` calls, not as zeros.
    pub fn set_stats_enabled(&self, enabled: bool) {
        self.stats_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Point-in-time snapshot of this store's runtime counters, plus the
    /// always-persisted pool stats embedded alongside them ([`RepackStats`],
    /// aggregate [`CacheStats`], per-shard `ShardStats`, index bytes).
    ///
    /// The logical read counters reflect only reads performed while stats were
    /// enabled ([`Self::set_stats_enabled`]); all other counters are lifetime
    /// totals since the store was assembled. See [`Self::reset_stats`] for the
    /// reset semantics — `shards`/`cache`/`repack` reflect persisted state and
    /// are deliberately excluded from the reset.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "a hit-rate is inherently an approximate floating-point presentation; counters retain their exact u64 values"
    )]
    pub fn stats(&self) -> RuntimeStats {
        let mut cache = CacheStats::default();
        let mut index_bytes: u64 = 0;
        let summaries = self.collection_summaries();
        for summary in &summaries {
            index_bytes = index_bytes.saturating_add(u64::try_from(summary.2).unwrap_or(u64::MAX));
            if let Some(stats) = self.cache_stats_for(&summary.0) {
                cache.hits = cache.hits.saturating_add(stats.hits);
                cache.misses = cache.misses.saturating_add(stats.misses);
            }
        }
        let cache_accesses = cache.hits.saturating_add(cache.misses);
        cache.hit_rate = if cache_accesses > 0 {
            cache.hits as f64 / cache_accesses as f64
        } else {
            0.0
        };
        RuntimeStats {
            open_count: self.open_count.load(Ordering::Relaxed),
            get_calls: self.get_calls.load(Ordering::Relaxed),
            get_misses: self.get_misses.load(Ordering::Relaxed),
            get_many_calls: self.get_many_calls.load(Ordering::Relaxed),
            get_many_records: self.get_many_records.load(Ordering::Relaxed),
            get_many_misses: self.get_many_misses.load(Ordering::Relaxed),
            put_calls: self.put_calls.load(Ordering::Relaxed),
            put_bytes: self.put_bytes.load(Ordering::Relaxed),
            put_many_calls: self.put_many_calls.load(Ordering::Relaxed),
            put_many_records: self.put_many_records.load(Ordering::Relaxed),
            put_many_bytes: self.put_many_bytes.load(Ordering::Relaxed),
            put_many_fast_path_calls: self.put_many_fast_path_calls.load(Ordering::Relaxed),
            put_many_clone_path_calls: self.put_many_clone_path_calls.load(Ordering::Relaxed),
            index_clone_time: std::time::Duration::from_nanos(
                self.index_clone_time_ns.load(Ordering::Relaxed),
            ),
            index_grow_count: self.index_grow_count.load(Ordering::Relaxed),
            index_rebuild_count: self.index_rebuild_count.load(Ordering::Relaxed),
            delta_invalidations: self.delta_invalidations.load(Ordering::Relaxed),
            checkpoint_writes: self.checkpoint_writes.load(Ordering::Relaxed),
            delta_appends: self.delta_appends.load(Ordering::Relaxed),
            sync_calls: self.sync_calls.load(Ordering::Relaxed),
            last_open_timings: self.open_timings(),
            last_sync_timings: self.sync_timings(),
            repack: self.repack_stats(),
            cache,
            shards: self.shard_stats(),
            index_bytes,
            collection_count: summaries.len(),
        }
    }

    /// Zero every runtime counter except `open_count` (counts stores
    /// assembled, not work) and the persisted pool stats — `shards`,
    /// `cache`, `repack`, `index_bytes`, and the collection count reflect
    /// on-disk state plus this process's decoded-cache and repack history and
    /// are kept as-is so a reset means "restart the runtime accounting," not
    /// "lie about the database."
    ///
    /// Also clears the retained `last_open_timings`/`last_sync_timings`
    /// breakdowns and leaves the `stats_enabled` flag untouched.
    pub fn reset_stats(&self) {
        for counter in [
            &self.get_calls,
            &self.get_misses,
            &self.get_many_calls,
            &self.get_many_records,
            &self.get_many_misses,
            &self.put_calls,
            &self.put_bytes,
            &self.put_many_calls,
            &self.put_many_records,
            &self.put_many_bytes,
            &self.put_many_fast_path_calls,
            &self.put_many_clone_path_calls,
            &self.index_clone_time_ns,
            &self.index_grow_count,
            &self.index_rebuild_count,
            &self.delta_invalidations,
            &self.checkpoint_writes,
            &self.delta_appends,
            &self.sync_calls,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        *self.last_open_timings.lock() = None;
        *self.last_sync_timings.lock() = None;
    }

    /// Number of times a specific collection has been repacked. 0 if it has
    /// never been repacked (or doesn't exist).
    #[must_use]
    pub fn repack_count_for_collection(&self, collection_id: &[u8; 16]) -> u64 {
        self.repack_counts_by_collection
            .read()
            .get(collection_id)
            .copied()
            .unwrap_or(0)
    }

    /// Cache hit/miss stats for a collection's decoded-node cache, if the collection
    /// currently has one loaded.
    #[must_use]
    pub fn cache_stats_for(&self, collection_id: &[u8; 16]) -> Option<CacheStats> {
        let gen = self.generation(collection_id)?;
        Some(CacheStats {
            hits: gen.cache.hits(),
            misses: gen.cache.misses(),
            hit_rate: gen.cache.hit_rate(),
        })
    }
}

/// Snapshot of global repack activity across all collections.
#[derive(Debug, Clone, Copy, Default)]
pub struct RepackStats {
    /// Total number of `repack_collection_reachable` calls across all collections.
    pub repack_count: u64,
    /// Total records kept (rewritten into a new generation) across all repacks.
    pub kept_total: u64,
    /// Total records dropped (found unreachable) across all repacks.
    pub dropped_total: u64,
}

/// Snapshot of a collection's decoded-node cache hit/miss stats.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub hit_rate: f64,
}

/// Point-in-time snapshot of a store's runtime counters and the persisted
/// pool stats embedded alongside them — see [`PackfileStorage::stats`].
///
/// All fields are relaxed reads of independent atomics (no shared snapshot
/// lock), so a snapshot is a consistent-enough picture, not a single instant
/// in time. The read-path counters (`get_*`) are only meaningful while
/// [`PackfileStorage::set_stats_enabled`] was true.
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Stores assembled in this process against this directory (1 for a fresh
    /// open; not reset by `reset_stats`).
    pub open_count: u64,
    /// Logical `get` calls (while stats enabled).
    pub get_calls: u64,
    /// Logical `get` calls that resolved to no record.
    pub get_misses: u64,
    /// Logical `get_many` calls (while stats enabled).
    pub get_many_calls: u64,
    /// Records requested across `get_many` calls.
    pub get_many_records: u64,
    /// `get_many` results that resolved to no record.
    pub get_many_misses: u64,
    /// Single-record `put` attempts.
    pub put_calls: u64,
    /// Bytes accepted across single-record `put` attempts.
    pub put_bytes: u64,
    /// `put_many` calls (batches).
    pub put_many_calls: u64,
    /// Records written across `put_many` calls.
    pub put_many_records: u64,
    /// Bytes written across `put_many` calls.
    pub put_many_bytes: u64,
    /// `put_many` calls on the owned no-clone fast path.
    pub put_many_fast_path_calls: u64,
    /// `put_many` calls that materialized an owned index up front.
    pub put_many_clone_path_calls: u64,
    /// Cumulative time spent materializing/growing owned indexes in `put_many`.
    pub index_clone_time: std::time::Duration,
    /// Index grow events in `put_many`.
    pub index_grow_count: u64,
    /// Fallback full-scan `rebuild_index` calls.
    pub index_rebuild_count: u64,
    /// Structural `invalidate_delta_log` calls.
    pub delta_invalidations: u64,
    /// Syncs that rewrote the index checkpoint in full.
    pub checkpoint_writes: u64,
    /// Syncs that appended the incremental delta log instead.
    pub delta_appends: u64,
    /// `sync`/`sync_all` calls.
    pub sync_calls: u64,
    /// Per-phase breakdown of the most recent open.
    pub last_open_timings: Option<OpenTimings>,
    /// Per-phase breakdown of the most recent sync.
    pub last_sync_timings: Option<SyncTimings>,
    /// Cumulative repack activity (persisted across opens).
    pub repack: RepackStats,
    /// Aggregate decoded-node cache hit/miss across loaded collections.
    pub cache: CacheStats,
    /// Persisted per-shard write/sync counters.
    pub shards: Vec<(u16, crate::shard::ShardStats)>,
    /// Total live index bytes across collections.
    pub index_bytes: u64,
    /// Number of live collections.
    pub collection_count: usize,
}

impl Default for RuntimeStats {
    fn default() -> Self {
        Self {
            open_count: 0,
            get_calls: 0,
            get_misses: 0,
            get_many_calls: 0,
            get_many_records: 0,
            get_many_misses: 0,
            put_calls: 0,
            put_bytes: 0,
            put_many_calls: 0,
            put_many_records: 0,
            put_many_bytes: 0,
            put_many_fast_path_calls: 0,
            put_many_clone_path_calls: 0,
            index_clone_time: std::time::Duration::ZERO,
            index_grow_count: 0,
            index_rebuild_count: 0,
            delta_invalidations: 0,
            checkpoint_writes: 0,
            delta_appends: 0,
            sync_calls: 0,
            last_open_timings: None,
            last_sync_timings: None,
            repack: RepackStats::default(),
            cache: CacheStats::default(),
            shards: Vec::new(),
            index_bytes: 0,
            collection_count: 0,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const TEST_COLLECTION: [u8; 16] = [0x01; 16];
    const SECOND_COLLECTION: [u8; 16] = [0x02; 16];

    /// A node id distinct across both the bucket bytes (0..8) and tag bytes
    /// (8..12) the lossy index actually reads — an id that only varies byte
    /// 0 collapses every entry onto the same 24-bit tag, which is not a
    /// realistic content hash and defeats the index in ways unrelated to
    /// whatever a test using it is meant to check.
    fn distinct_id(byte: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[0] = byte;
        id[9] = byte.wrapping_mul(37).wrapping_add(11);
        id[15] = byte.wrapping_mul(7);
        id
    }
    const OTHER_COLLECTION: [u8; 16] = [0x02; 16];

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

        store.put(&TEST_COLLECTION, &id, &data).unwrap();
        let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);
    }

    #[test]
    fn reopened_index_keeps_distinct_same_tag_entries() {
        let dir = test_dir("reopen_same_tag");
        let mut first = [0u8; 16];
        first[8] = 0x42;
        first[15] = 1;
        let mut second = first;
        second[15] = 2;

        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &first,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // The checkpoint has only slots, no retained full identities. The
        // second write must hydrate the same-tag candidate and continue its
        // probe, rather than replacing the first record's only location.
        let reopened = PackfileStorage::open(dir).unwrap();
        reopened
            .put(
                &TEST_COLLECTION,
                &second,
                &NodeData::new(bytes::Bytes::from_static(b"second")),
            )
            .unwrap();
        assert!(reopened.get(&TEST_COLLECTION, &first).unwrap().is_some());
        assert!(reopened.get(&TEST_COLLECTION, &second).unwrap().is_some());
    }

    #[test]
    fn checkpoint_index_growth_recovers_hashes_from_indexed_frames() {
        // A batch large enough to be a "real" collection. The checkpoint
        // this reopen loads stores locations but not home hashes, so
        // growing it (via the `put_many` call below) used to rescan every
        // pack for this collection instead of recovering hashes from the
        // indexed delta frames.
        const ENTRIES: usize = 3_072;
        let dir = test_dir("checkpoint_growth_recovery");
        let store = PackfileStorage::open(dir.clone())
            .unwrap()
            .with_append_policy(crate::shard::AppendPolicy::buffered());
        let entries: Vec<_> = (0..ENTRIES)
            .map(|i| {
                let mut id = [0u8; 16];
                let mixed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                id[..8].copy_from_slice(&mixed.to_be_bytes());
                id[8] = u8::try_from((i >> 16) & 0xFF).unwrap();
                id[9] = u8::try_from((i >> 8) & 0xFF).unwrap();
                id[10] = u8::try_from(i & 0xFF).unwrap();
                (id, NodeData::new(bytes::Bytes::from_static(b"payload")))
            })
            .collect();
        store.put_many(&TEST_COLLECTION, &entries).unwrap();
        store.sync_all().unwrap();
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let checkpoint_index = reopened
            .generation(&TEST_COLLECTION)
            .expect("checkpoint collection exists");
        assert!(checkpoint_index.index.is_mmap_backed());
        let grown = reopened
            .grow_checkpoint_index(&TEST_COLLECTION, &checkpoint_index.index)
            .unwrap()
            .expect("4K checkpoint index can grow");
        assert_eq!(grown.len(), ENTRIES);
        for (id, _) in &entries {
            assert!(grown.lookup(id).is_some(), "recovered hash remains indexed");
        }
        drop(checkpoint_index);

        // Exercise the actual put_many fallback too. `insert_index` fails
        // closed once `len >= capacity * 3 / 4`, and the reopened generation
        // is mmap-backed (so `LossyIndex::grow` — which needs the original
        // homes — can't apply): crossing that threshold therefore always
        // routes through `grow_checkpoint_index`, never a silent plain grow
        // or a full pack rescan. Compute the threshold from the checkpoint's
        // actual on-disk capacity and add enough unique post-reopen entries
        // to be certain we cross it, instead of relying on one extra record
        // that may land comfortably under it.
        let checkpoint_capacity = u64::from(
            reopened
                .generation(&TEST_COLLECTION)
                .expect("checkpoint collection exists")
                .index
                .capacity(),
        );
        let threshold = checkpoint_capacity * 3 / 4;
        // Add enough unique entries to certainly cross the threshold,
        // regardless of exactly how much headroom the checkpoint's
        // persisted capacity happens to have.
        let needed = threshold.saturating_sub(u64::try_from(ENTRIES).unwrap_or(0)) + 64;
        let extra_entries: Vec<_> = (0..needed)
            .map(|i| {
                let mut id = [0xA5; 16];
                id[..8].copy_from_slice(&i.to_be_bytes());
                (id, NodeData::new(bytes::Bytes::from_static(b"extra")))
            })
            .collect();
        reopened.put_many(&TEST_COLLECTION, &extra_entries).unwrap();
        let grown_capacity = reopened
            .generation(&TEST_COLLECTION)
            .expect("collection still exists")
            .index
            .capacity();
        assert!(
            u64::from(grown_capacity) > checkpoint_capacity,
            "put_many must have grown the index past its checkpoint capacity"
        );
        for (id, _) in &extra_entries {
            assert!(reopened.get(&TEST_COLLECTION, id).unwrap().is_some());
        }
        assert!(reopened
            .get(&TEST_COLLECTION, &entries[0].0)
            .unwrap()
            .is_some());
    }

    /// A writable open must not silently skip a corrupt pack and publish a
    /// partial index. The caller needs an error so it can repair or restore
    /// the pack before accepting writes.
    #[test]
    fn test_writable_open_rejects_corrupt_pack_instead_of_skipping_it() {
        let dir = test_dir("open_corrupt_pack");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let id = distinct_id(0x42);
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"payload")),
            )
            .unwrap();
        // Commit the buffered frame so the tamper below actually overwrites
        // on-disk bytes (the record is otherwise only in the append buffer).
        store.sync_all().unwrap();
        let path = store.shards.active_shard().path.clone();
        drop(store);

        let mut bytes = fs::read(&path).unwrap();
        let payload_byte =
            crate::packfile::HEADER_LEN + 4 + crate::packfile::FRAME_FIXED_LEN as usize;
        bytes[payload_byte] ^= 0xff;
        fs::write(path, bytes).unwrap();

        assert!(PackfileStorage::open(dir).is_err());
    }

    /// Full and `WriteOnly` keep writing real CRCs, so a reopen scan still
    /// verifies them: a physically tampered pack must fail to reopen.
    /// `Disabled` frames carry no checksum, so a tampered pack reopens and
    /// the engine serves the corrupted bytes as data.
    #[test]
    fn test_checksum_policy_gates_verification_on_reopen_scan() {
        for (policy, expect_reopen_ok) in [
            (crate::packfile::ChecksumPolicy::Full, false),
            (crate::packfile::ChecksumPolicy::WriteOnly, false),
            (crate::packfile::ChecksumPolicy::Disabled, true),
        ] {
            let dir = test_dir(&format!("checksum_policy_reopen_{policy:?}"));
            {
                let store = PackfileStorage::open_with_policies(dir.clone(), true, policy).unwrap();
                let id = distinct_id(0x52);
                store
                    .put(
                        &TEST_COLLECTION,
                        &id,
                        &NodeData::new(bytes::Bytes::from_static(b"tamper target")),
                    )
                    .unwrap();
                let (shard_id, offset) = store
                    .generation(&TEST_COLLECTION)
                    .unwrap()
                    .index
                    .lookup(&id)
                    .expect("just-written record must be indexed");
                // Tampering happens on-disk: commit the buffered frame so the
                // tamper site computed from `offset` actually lands in the file.
                store.sync_all().unwrap();
                let path = store.shards.get_shard(shard_id).unwrap().path.clone();
                drop(store);

                let mut bytes = fs::read(&path).unwrap();
                let payload_byte = usize::try_from(offset)
                    .expect("u64 offset fits usize on any supported host")
                    .saturating_add(4)
                    .saturating_add(crate::packfile::FRAME_FIXED_LEN as usize);
                assert!(payload_byte < bytes.len(), "tamper site must be in-file");
                bytes[payload_byte] ^= 0xff;
                bytes[payload_byte + 1] ^= 0x0f;
                fs::write(path, bytes).unwrap();
            }

            if expect_reopen_ok {
                let store = PackfileStorage::open_with_policies(dir, true, policy).unwrap();
                let got = store
                    .get(&TEST_COLLECTION, &distinct_id(0x52))
                    .unwrap()
                    .expect("Disabled reopen must serve the record");
                assert_eq!(
                    got.bytes.as_ref().len(),
                    b"tamper target".len(),
                    "payload frame must still decode"
                );
            } else {
                let reopened = PackfileStorage::open_with_policies(dir, true, policy);
                match reopened {
                    Ok(_) => panic!("reopen scan must reject a tampered pack"),
                    Err(err) => {
                        assert!(
                            matches!(err, std::io::Error { .. }),
                            "expected I/O error from scan_and_recover, got {err:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_read_survives_append_after_mmap_established() {
        let dir = test_dir("stale_mmap_regression");
        let store = PackfileStorage::open(dir).unwrap();

        let a = [0xAAu8; 16];
        let b = [0xBBu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"bbbb"));

        store.put(&TEST_COLLECTION, &a, &data_a).unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        let got_a = store.get(&TEST_COLLECTION, &a).unwrap();
        assert_eq!(got_a.unwrap().bytes, data_a.bytes);

        store.put(&TEST_COLLECTION, &b, &data_b).unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        let got_b = store.get(&TEST_COLLECTION, &b).unwrap();
        assert_eq!(got_b.expect("B must be found").bytes, data_b.bytes);
    }

    #[test]
    fn test_get_not_found() {
        let dir = test_dir("notfound");
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&TEST_COLLECTION, &[0x00; 16]).unwrap().is_none());
    }

    #[test]
    fn test_cache_hit() {
        let dir = test_dir("cachehit");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"cached"));

        store.put(&TEST_COLLECTION, &id, &data).unwrap();

        let _ = store.get(&TEST_COLLECTION, &id).unwrap();
        let gen = store.generation(&TEST_COLLECTION).unwrap();
        assert_eq!(gen.cache.hits(), 1);

        let _ = store.get(&TEST_COLLECTION, &id).unwrap();
        assert_eq!(gen.cache.hits(), 2);
    }

    #[test]
    fn test_cold_get_does_not_populate_lru() {
        let dir = test_dir("cold_get_no_lru_insert");
        let store = PackfileStorage::open(dir).unwrap();
        let id = [0x01u8; 16];
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"mmap-backed")),
            )
            .unwrap();

        let gen = store.generation(&TEST_COLLECTION).unwrap();
        gen.cache.clear();
        assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
        assert_eq!(gen.cache.len(), 0);
    }

    #[test]
    fn test_delete_collection() {
        let dir = test_dir("delete");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

        store.put(&OTHER_COLLECTION, &id, &data).unwrap();

        store.delete_collection(&OTHER_COLLECTION).unwrap();
        assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
        assert!(store.generation(&OTHER_COLLECTION).is_none());
    }

    #[test]
    fn test_delete_collection_does_not_resurrect_on_reopen() {
        let dir = test_dir("delete_no_resurrect");

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            store.put(&OTHER_COLLECTION, &id, &data).unwrap();
            store.delete_collection(&OTHER_COLLECTION).unwrap();
            store.sync_all().unwrap();
        }

        // Reopen from scratch: the on-disk deleted.collections marker must make
        // the startup scan skip OTHER_COLLECTION's leftover packfile records,
        // rather than resurrecting them into a fresh index.
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
        assert!(store.generation(&OTHER_COLLECTION).is_none());
    }

    #[test]
    fn test_delete_collection_clears_live_roots() {
        let dir = test_dir("delete_live_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        store
            .put(
                &OTHER_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.set_live_roots(&OTHER_COLLECTION, vec![id]);
        assert!(store.live_roots.read().contains_key(&OTHER_COLLECTION));

        store.delete_collection(&OTHER_COLLECTION).unwrap();
        assert!(!store.live_roots.read().contains_key(&OTHER_COLLECTION));
    }

    #[test]
    fn test_batch_put_get() {
        let dir = test_dir("batch");
        let store = PackfileStorage::open(dir).unwrap();

        let entries = ten_record_fixture();

        store.put_many(&TEST_COLLECTION, &entries).unwrap();

        let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        let results = store.get_many(&TEST_COLLECTION, &ids).unwrap();
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
        store.put_many(&TEST_COLLECTION, &entries).unwrap();
        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }

        let mut reversed_ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        reversed_ids.reverse();

        let results = store.get_many(&TEST_COLLECTION, &reversed_ids).unwrap();
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
    fn test_walk_ancestors_stops_at_boundary() {
        let dir = test_dir("walk_basic");
        let store = PackfileStorage::open(dir).unwrap();

        // A -> B -> C -> D -> E, linear chain, edges point to predecessors.
        let ids: Vec<NodeId> = (0..5u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (ids[1], vec![ids[0]]),
            (ids[2], vec![ids[1]]),
            (ids[3], vec![ids[2]]),
            (ids[4], vec![ids[3]]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // frontier=[D], stop_at=[B]: should yield C, D in ancestor-first
        // order — stopping at and excluding B, including the D frontier.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[3]],
                &[ids[1]],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        assert_eq!(
            bytes,
            vec![b'C', b'D'],
            "walk must be ancestor-first, excluding the stop_at boundary"
        );

        // Empty stop_at: walks all the way back to the collection's true roots.
        let full: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[4]],
                &[],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let full_bytes: Vec<u8> = full.iter().map(|(_, d)| d.bytes[0]).collect();
        assert_eq!(full_bytes, vec![b'A', b'B', b'C', b'D', b'E']);
    }

    #[test]
    fn test_walk_ancestors_branching_dag() {
        let dir = test_dir("walk_branch");
        let store = PackfileStorage::open(dir).unwrap();

        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let ids: Vec<NodeId> = (0..4u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (ids[1], vec![ids[0]]),
            (ids[2], vec![ids[0]]),
            (ids[3], vec![ids[1], ids[2]]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // frontier=[D], stop_at=[A]: should yield exactly {B, C, D} — A
        // excluded as the boundary, both branches merge back in correctly.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[3]],
                &[ids[0]],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        // D must come last (ancestor-first order); B/C order between them
        // is unconstrained since they're siblings.
        assert_eq!(*bytes.last().unwrap(), b'D');
        bytes.sort_unstable();
        assert_eq!(bytes, vec![b'B', b'C', b'D']);
    }

    #[test]
    fn test_walk_ancestors_never_reaching_stop_at_walks_to_roots() {
        // A fork that never crosses the supplied stop_at boundary must not
        // be silently truncated there — it's an ancestor walk with stop
        // markers, not a from/to span, so it walks to the collection's true
        // roots instead.
        let dir = test_dir("walk_fork_misses_boundary");
        let store = PackfileStorage::open(dir).unwrap();

        //   ROOT
        //   /  \
        // MAIN  FORK
        //  |
        // TIP
        let root = distinct_id(0);
        let main = distinct_id(1);
        let fork = distinct_id(2);
        let tip = distinct_id(3);
        for (id, byte) in [(root, b'R'), (main, b'M'), (fork, b'F'), (tip, b'T')] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from(vec![byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (main, vec![root]),
            (fork, vec![root]),
            (tip, vec![main]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // stop_at names `fork`, which TIP's history never crosses — the
        // walk from TIP must still reach ROOT rather than stopping short.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[tip],
                &[fork],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        bytes.sort_unstable();
        assert_eq!(bytes, vec![b'M', b'R', b'T']);
    }

    #[test]
    fn test_walk_ancestors_max_nodes_bounds_a_runaway_walk() {
        let dir = test_dir("walk_max_nodes");
        let store = PackfileStorage::open(dir).unwrap();

        // A long chain with a stop_at that never gets hit — max_nodes is
        // the only thing that keeps this walk from covering the chain.
        let ids: Vec<NodeId> = (0..20u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i % 26).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'a' + byte])),
                )
                .unwrap();
        }
        let mut edges = std::collections::HashMap::new();
        for i in 1..ids.len() {
            edges.insert(ids[i], vec![ids[i - 1]]);
        }
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[19]],
                &[],
                extract,
                WalkLimits { max_nodes: Some(5) },
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            walked.len() <= 5,
            "max_nodes must bound the walk even though stop_at was never reached"
        );
    }

    #[test]
    fn test_walk_ancestors_survives_concurrent_repack_gc() {
        // Regression test for the exact concurrency bug flagged in review:
        // an earlier draft resolved each node lazily, at drain time,
        // against whatever generation happened to be live *then* — so a
        // repack that GC'd a node between building the walk and draining
        // it would silently drop that node from the results. Resolving
        // eagerly against one frozen generation snapshot up front (what
        // walk_ancestors does now) must not exhibit that.
        let dir = test_dir("walk_survives_repack");
        let store = PackfileStorage::open(dir).unwrap();

        let a = distinct_id(0);
        let b = distinct_id(1);
        store
            .put(
                &TEST_COLLECTION,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();
        let edges = std::collections::HashMap::from([(b, vec![a])]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // Build the walk (resolves and captures A and B eagerly)...
        let walk = store
            .walk_ancestors(&TEST_COLLECTION, &[b], &[], extract, WalkLimits::default())
            .unwrap();

        // ...then, before draining it, run a repack that only keeps B as
        // a live root — A becomes unreachable and gets GC'd out of the
        // collection's index/cache entirely.
        store.set_live_roots(&TEST_COLLECTION, vec![b]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!((kept, dropped), (1, 1), "repack must have GC'd A");

        // The already-built walk must still yield both A and B: it
        // captured its data before the repack ran, so it isn't affected
        // by the generation swap that just happened.
        let results: Vec<(NodeId, NodeData)> = walk.collect::<Result<_, _>>().unwrap();
        let mut bytes: Vec<u8> = results.iter().map(|(_, d)| d.bytes[0]).collect();
        bytes.sort_unstable();
        assert_eq!(
            bytes,
            vec![b'A', b'B'],
            "a walk built before a concurrent repack must not lose nodes that repack GC'd afterward"
        );
    }

    #[test]
    fn test_multiple_records_same_collection() {
        let dir = test_dir("multi");
        let store = PackfileStorage::open(dir).unwrap();

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let data = NodeData::new(bytes::Bytes::from(format!("record {i}")));
            store.put(&TEST_COLLECTION, &id, &data).unwrap();
        }

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
            assert_eq!(got.bytes, bytes::Bytes::from(format!("record {i}")));
        }
    }

    #[test]
    fn test_collection_isolation() {
        let dir = test_dir("isolation");
        let store = PackfileStorage::open(dir).unwrap();

        let id_a = [0x42u8; 16];
        let id_b = [0x43u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

        store.put(&TEST_COLLECTION, &id_a, &data_a).unwrap();
        store.put(&OTHER_COLLECTION, &id_b, &data_b).unwrap();

        let got_a = store.get(&TEST_COLLECTION, &id_a).unwrap().unwrap();
        let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
        assert_eq!(got_a.bytes, data_a.bytes);
        assert_eq!(got_b.bytes, data_b.bytes);

        assert!(store.get(&OTHER_COLLECTION, &id_a).unwrap().is_none());
        assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_none());

        store.delete_collection(&TEST_COLLECTION).unwrap();
        assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_none());
        let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
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

        let collection = [0x77u8; 16];
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
                        store.put(&collection, &id, &data).unwrap();
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
                        match store.get(&collection, &id).unwrap() {
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
            match store.get(&collection, id).unwrap() {
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

    /// Regression for the epoch-handoff protocol in `persist_index_checkpoint`:
    /// full checkpoint rewrites now hold every collection's `put_mutex` only
    /// across the fingerprint/index snapshot and delta-epoch rotation, not
    /// across the (unlocked) serialize + fsync + rename. Concurrent `put`s
    /// landing in that unlocked window must all still be durable after the
    /// next sync and a reopen — none may be silently dropped by the epoch
    /// rotation, and none may be lost to a torn straddle between the pack
    /// fingerprint and the index snapshot.
    #[test]
    fn test_concurrent_put_survives_checkpoint_rewrites() {
        use std::sync::Mutex;
        use std::thread;

        const NUM_WRITERS: usize = 6;
        const PUTS_PER_WRITER: usize = 300;
        // Force many full rewrites during the run: each rewrite thread
        // iteration invalidates the delta log (via a collection delete on an
        // otherwise-untouched collection) then rewrites, so persistence keeps
        // taking the epoch-rotation path instead of the cheap delta append.
        const REWRITES: usize = 40;

        let dir = test_dir("concurrent_put_checkpoint_rewrite");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        // Seed one throwaway collection per rewrite so each rewrite iteration
        // has a fresh structural invalidation to force (deleting an
        // already-deleted collection is a no-op and wouldn't force a rewrite).
        for r in 0..REWRITES {
            let mut collection = [0u8; 16];
            collection[0] = 0xEE;
            collection[1..9].copy_from_slice(&(r as u64).to_le_bytes());
            let mut id = [0u8; 16];
            id[15] = 1;
            store
                .put(&collection, &id, &NodeData::new(bytes::Bytes::from("seed")))
                .unwrap();
        }
        store.sync_all().unwrap();

        let written: Mutex<Vec<([u8; 16], NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());

        thread::scope(|scope| {
            for w in 0..NUM_WRITERS {
                let store = &store;
                let written = &written;
                scope.spawn(move || {
                    let mut collection = [0u8; 16];
                    collection[0] = 0xAA;
                    collection[1] = u8::try_from(w).unwrap();
                    for i in 0..PUTS_PER_WRITER {
                        let mut id = [0u8; 16];
                        id[0] = u8::try_from(w).unwrap();
                        id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
                        let bytes = bytes::Bytes::from(format!("writer {w} put {i}"));
                        let data = NodeData::new(bytes.clone());
                        store.put(&collection, &id, &data).unwrap();
                        written.lock().unwrap().push((collection, id, bytes));
                    }
                });
            }

            scope.spawn(|| {
                for r in 0..REWRITES {
                    let mut throwaway = [0u8; 16];
                    throwaway[0] = 0xEE;
                    throwaway[1..9].copy_from_slice(&(r as u64).to_le_bytes());
                    store.delete_collection(&throwaway).unwrap();
                    store.persist_index_checkpoint_best_effort();
                }
            });
        });

        store.sync_all().unwrap();
        drop(store);

        let reopened = PackfileStorage::open(dir).unwrap();
        let written = written.into_inner().unwrap();
        assert_eq!(written.len(), NUM_WRITERS * PUTS_PER_WRITER);

        let mut lost = Vec::new();
        for (collection, id, expected_bytes) in &written {
            match reopened.get(collection, id).unwrap() {
                Some(data) => assert_eq!(data.bytes, *expected_bytes),
                None => lost.push(*id),
            }
        }
        assert!(
            lost.is_empty(),
            "{} of {} records lost to a concurrent checkpoint rewrite: {lost:?}",
            lost.len(),
            written.len()
        );
    }

    /// Deterministic, fast regression guard for the epoch-handoff protocol's
    /// actual point: a concurrent `put()` must not block on the checkpoint
    /// rewrite's serialize/write/rename, no matter how long that takes.
    /// Rather than inferring "not blocked" from a wall-clock race against
    /// real disk I/O (slow, and only as reliable as the dataset is large
    /// enough to make a real rewrite slow relative to test-machine noise —
    /// see the git history on this function for the timing-based version
    /// this replaced), this injects an artificial, arbitrarily long delay
    /// into the unlocked window via a test-only hook and asserts a `put()`
    /// issued after the rewrite has entered that window returns in a small
    /// fraction of it. A proper wall-clock benchmark of this same property
    /// lives in `benches/storage.rs` (`cargo bench`), where it belongs.
    #[test]
    fn test_put_does_not_block_on_slow_checkpoint_rewrite() {
        use std::sync::atomic::AtomicBool;
        use std::thread;
        use std::time::{Duration, Instant};

        const ARTIFICIAL_DELAY: Duration = Duration::from_millis(300);

        let dir = test_dir("put_not_blocked_by_slow_rewrite");
        let store = PackfileStorage::open(dir).unwrap();

        // Establish a first checkpoint so the forced rewrite below has
        // something real to serialize and rewrite.
        let cid = [0x33u8; 16];
        store
            .put(&cid, &[0u8; 16], &NodeData::new(bytes::Bytes::from("seed")))
            .unwrap();
        store.sync_all().unwrap();

        // Force the next rewrite via a structural invalidation.
        let throwaway = [0x44u8; 16];
        store
            .put(
                &throwaway,
                &[0u8; 16],
                &NodeData::new(bytes::Bytes::from("x")),
            )
            .unwrap();
        store.delete_collection(&throwaway).unwrap();

        let entered_unlocked_window = std::sync::Arc::new(AtomicBool::new(false));
        let put_elapsed = std::sync::Mutex::new(None::<Duration>);

        thread::scope(|scope| {
            {
                let store = &store;
                let hook_flag = std::sync::Arc::clone(&entered_unlocked_window);
                scope.spawn(move || {
                    store
                        .test_persist_index_checkpoint_with_delay(ARTIFICIAL_DELAY, &hook_flag)
                        .unwrap();
                });
            }
            // Wait for the hook to confirm it has released every lock and is
            // now sleeping inside the artificial delay — a real signal from
            // the rewrite thread, not a fixed-sleep guess that depends on
            // test-machine timing. The put issued below then provably runs
            // concurrently with the unlocked rewrite window.
            while !entered_unlocked_window.load(Ordering::Acquire) {
                thread::yield_now();
            }

            let started = Instant::now();
            store
                .put(
                    &cid,
                    &[1u8; 16],
                    &NodeData::new(bytes::Bytes::from("concurrent")),
                )
                .unwrap();
            *put_elapsed.lock().unwrap() = Some(started.elapsed());
        });

        let put_elapsed = put_elapsed.into_inner().unwrap().unwrap();
        assert!(
            put_elapsed < ARTIFICIAL_DELAY / 3,
            "a put() issued while a rewrite was sleeping in its unlocked window took \
             {put_elapsed:?} — it should return almost immediately, not wait anywhere near the \
             artificial {ARTIFICIAL_DELAY:?} delay; did the lock scope regress to holding \
             put_mutex across the serialize/write/rename again?"
        );
    }

    /// Crash-boundary: a crash between the D0→D1 epoch rotation and C1's
    /// checkpoint rename. D1 exists only in the (now-lost) in-memory state;
    /// nothing on disk names its fingerprint. The pack bytes for whatever
    /// prompted the rotation are already physically flushed (rotation always
    /// flushes first), so the reopen's local fingerprint has already moved
    /// past D0's sealed tail — the trusted-log check fails and this must fall
    /// back to a full rescan (packfiles stay authoritative), never silently
    /// lose the data.
    #[test]
    fn test_crash_between_rotation_and_checkpoint_rename_falls_back_to_rescan() {
        let dir = test_dir("crash_between_rotation_and_rename");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let cid = [0x11u8; 16];
        let mut written = Vec::new();

        // Round 1: establish C0 via a full (first-ever) rewrite.
        for i in 0..50u64 {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&i.to_le_bytes());
            let bytes = bytes::Bytes::from(format!("round1 {i}"));
            store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
            written.push((id, bytes));
        }
        store.sync_all().unwrap();

        // Round 2: a plain write's sync appends a delta batch continuing C0
        // (D0), never rewriting the checkpoint.
        for i in 50..100u64 {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&i.to_le_bytes());
            let bytes = bytes::Bytes::from(format!("round2 {i}"));
            store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
            written.push((id, bytes));
        }
        store.sync_all().unwrap();

        // Round 3: more writes, then simulate a crash exactly between the
        // epoch rotation (which flushes these bytes and discards their
        // not-yet-appended delta frames) and the checkpoint that would have
        // validated the new epoch.
        for i in 100..150u64 {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&i.to_le_bytes());
            let bytes = bytes::Bytes::from(format!("round3 {i}"));
            store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
            written.push((id, bytes));
        }
        let _ = store.test_rotate_epoch_without_checkpoint();
        drop(store); // simulated crash: no checkpoint, no retirement.

        let reopened = PackfileStorage::open(dir).unwrap();
        let open = reopened.open_timings().expect("open must record timings");
        assert_eq!(
            open.path,
            OpenPath::FullScan,
            "D0's sealed tail can no longer match the post-rotation pack state, so the \
             stale-but-still-present log must be rejected wholesale, not partially replayed"
        );
        for (id, expected_bytes) in &written {
            let got = reopened
                .get(&cid, id)
                .unwrap()
                .expect("every write, including the un-appended round 3, survives via the packs");
            assert_eq!(got.bytes.as_ref(), expected_bytes.as_ref());
        }
    }

    /// Crash-boundary: a crash between C1's checkpoint rename becoming
    /// durable and D0's retirement. Both C1 (+ the fresh, still-empty D1) and
    /// the now-stale D0 exist on disk at once; the reopen must select C1 and
    /// ignore D0 (base fingerprint no longer matches), and the orphaned D0
    /// file must be swept away by the sweep-on-open cleanup.
    #[test]
    fn test_crash_between_checkpoint_rename_and_retirement_uses_new_checkpoint() {
        let dir = test_dir("crash_between_rename_and_retire");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let cid = [0x22u8; 16];
        let mut written = Vec::new();

        // Stay well under `NEW_COLLECTION_INDEX_FLOOR`'s 75%-load grow
        // threshold (48 entries at the floor of 64): this test exercises
        // checkpoint/epoch continuity across a simulated crash, not the
        // index's grow() behavior, and a grow mid-round would invalidate
        // the delta log this test is asserting still exists on disk.
        for i in 0..10u64 {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&i.to_le_bytes());
            let bytes = bytes::Bytes::from(format!("round1 {i}"));
            store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
            written.push((id, bytes));
        }
        store.sync_all().unwrap(); // C0, D0 exists once round 2 appends below.

        for i in 10..20u64 {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&i.to_le_bytes());
            let bytes = bytes::Bytes::from(format!("round2 {i}"));
            store.put(&cid, &id, &NodeData::new(bytes.clone())).unwrap();
            written.push((id, bytes));
        }
        store.sync_all().unwrap(); // D0 now has a committed batch continuing C0.

        let old_d0_fingerprint = {
            let (_new_fingerprint, old_base_fingerprint) =
                store.test_write_checkpoint_without_retire().unwrap();
            old_base_fingerprint.expect("this session had a prior epoch (C0/D0) to retire")
        };
        let d0_path = PackfileStorage::delta_path(&dir, old_d0_fingerprint);
        assert!(
            d0_path.exists(),
            "D0 must still be on disk immediately after the simulated crash point"
        );
        drop(store); // simulated crash: retire_delta_epoch never ran.

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let open = reopened.open_timings().expect("open must record timings");
        assert_eq!(
            open.path,
            OpenPath::Checkpoint,
            "C1 is durable and matches the current pack state exactly (no writes happened \
             after the snapshot), so the reopen must take the fast path"
        );
        assert!(
            !d0_path.exists(),
            "the orphaned, now-unreachable D0 must be swept away by the writable open's cleanup"
        );
        for (id, expected_bytes) in &written {
            let got = reopened
                .get(&cid, id)
                .unwrap()
                .expect("every write survives a reopen onto the new checkpoint");
            assert_eq!(got.bytes.as_ref(), expected_bytes.as_ref());
        }
    }

    #[test]
    fn test_swizzle_callback() {
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
        let store = PackfileStorage::open_with_swizzle(dir, 100, test_swizzle).unwrap();

        let parent_id = [0x10u8; 16];
        let child_a = [0x20u8; 16];
        let child_b = [0x30u8; 16];

        let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
        let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));

        store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
        store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();
        store
            .put(&TEST_COLLECTION, &parent_id, &parent_data)
            .unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
        store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();

        let extract = |_data: &NodeData| -> Vec<NodeId> { vec![child_a, child_b] };

        let result = store
            .get_swizzled(&TEST_COLLECTION, &parent_id, extract)
            .unwrap();
        assert!(result.is_some());
        let node_ref = result.unwrap();
        assert!(node_ref.is_resolved());
        assert_eq!(node_ref.structural_hash(), &parent_id);

        assert_eq!(SWIZZLE_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(CACHED_CHILDREN_FOUND.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_sync() {
        let dir = test_dir("sync");
        let store = PackfileStorage::open(dir).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &[0xAA; 16],
                &NodeData::new(bytes::Bytes::from_static(b"hi")),
            )
            .unwrap();
        store.sync().unwrap();
    }

    #[test]
    fn test_probe_reopen_first_append_sync_path_no_growth() {
        // Discriminates reopen-materialization cost from real capacity-growth
        // cost: build to a load well under the 75% grow threshold (so the
        // post-reopen append batch cannot trigger `index_grow_count`), then
        // reopen and append. If sync still takes the checkpoint path here,
        // materialization alone invalidates the delta log and a re-base
        // frame (which only helps the growth case) buys nothing.
        let dir = test_dir("probe_reopen_sync_no_growth");
        let total = 2000u32; // 2032 / 4096 is well under the 75% grow threshold.
        build_reopen_probe_checkpoint(&dir, total);
        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            let before = store.stats();
            reopen_probe_put_range(&store, total..(total + 32), 32, |_| {
                NodeData::new(bytes::Bytes::from_static(b"y"))
            });
            let after_put = store.stats();
            let d = store.delta_state.lock();
            let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
            eprintln!(
                "AFTER REOPEN PUT: cap={} len={} load={:.3} clones+={} grows+={} invalids+={} | pending={} invalid={}",
                room.index.capacity(),
                room.index.len(),
                reopen_probe_load_factor(&room),
                after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
                after_put.index_grow_count - before.index_grow_count,
                after_put.delta_invalidations - before.delta_invalidations,
                d.pending.len(),
                d.invalid,
            );
            assert_eq!(
                after_put.index_grow_count - before.index_grow_count,
                0,
                "test setup invariant violated: this batch must not trigger real growth"
            );
            assert_eq!(
                after_put.delta_invalidations - before.delta_invalidations,
                0
            );
            assert!(!d.invalid);
            drop(d);
            store.sync().unwrap();
            let st = store.stats();
            let ts = store.sync_timings().unwrap();
            eprintln!(
                "REOPEN SYNC TIMINGS (no growth): checkpoint={:?} delta={:?} total={:?}",
                ts.checkpoint, ts.delta_log, ts.total
            );
            eprintln!(
                "REOPEN SYNC STATS (no growth): ckpt_writes+={} delta_appends+={}",
                st.checkpoint_writes - before.checkpoint_writes,
                st.delta_appends - before.delta_appends,
            );
            assert_eq!(st.checkpoint_writes - before.checkpoint_writes, 0);
            assert_eq!(st.delta_appends - before.delta_appends, 1);
            assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
            assert!(ts.delta_log > std::time::Duration::ZERO);
        }
    }

    const REOPEN_PROBE_COLLECTION: [u8; 16] = [0x11; 16];

    fn reopen_probe_id(i: u32) -> NodeId {
        let mut id = [0u8; 16];
        id[0..4].copy_from_slice(&i.to_le_bytes());
        id[8..12].copy_from_slice(&(i ^ 0x9E37_79B9).to_le_bytes());
        id
    }

    fn reopen_probe_payload(i: u32) -> NodeData {
        let mut payload = vec![0u8; 1024];
        let seed = u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF;
        payload[..2].copy_from_slice(&seed.to_le_bytes()[..2]);
        NodeData::new(bytes::Bytes::from(payload))
    }

    fn reopen_probe_put_range(
        store: &PackfileStorage,
        range: std::ops::Range<u32>,
        batch_size: usize,
        payload: impl Fn(u32) -> NodeData,
    ) {
        let mut batch = Vec::with_capacity(batch_size);
        for i in range {
            batch.push((reopen_probe_id(i), payload(i)));
            if batch.len() == batch_size {
                store.put_many(&REOPEN_PROBE_COLLECTION, &batch).unwrap();
                batch.clear();
            }
        }
        if !batch.is_empty() {
            store.put_many(&REOPEN_PROBE_COLLECTION, &batch).unwrap();
        }
    }

    fn reopen_probe_load_factor(room: &RoomGeneration) -> f64 {
        f64::from(u32::try_from(room.index.len()).expect("probe index length fits in u32"))
            / f64::from(room.index.capacity())
    }

    fn build_reopen_probe_checkpoint(dir: &std::path::Path, total: u32) {
        let store = PackfileStorage::open_with_cache_and_policies(
            dir.to_path_buf(),
            0,
            true,
            packfile::ChecksumPolicy::Full,
        )
        .unwrap()
        .with_append_policy(shard::AppendPolicy::buffered());
        reopen_probe_put_range(&store, 0..total, 256, reopen_probe_payload);
        store.sync_all().unwrap();
        let stats = store.stats();
        let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
        eprintln!(
            "BUILD done: cap={} len={} load={:.3} gen={} | clones={} grows={} ckpt_writes={} delta_appends={}",
            room.index.capacity(), room.index.len(), reopen_probe_load_factor(&room), room.generation,
            stats.put_many_clone_path_calls, stats.index_grow_count, stats.checkpoint_writes,
            stats.delta_appends,
        );
    }

    #[test]
    fn test_probe_reopen_first_append_sync_path() {
        let dir = test_dir("probe_reopen_sync");
        let total = 3051u32;
        build_reopen_probe_checkpoint(&dir, total);
        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            let d = store.delta_state.lock();
            eprintln!(
                "REOPEN delta_state: base_fingerprint={:?} invalid={} pending={} log_bytes={}",
                d.base_fingerprint,
                d.invalid,
                d.pending.len(),
                d.log_bytes
            );
            let room = store.generation(&REOPEN_PROBE_COLLECTION).unwrap();
            eprintln!(
                "REOPEN room: cap={} len={} load={:.3} gen={} base_gen={:?}",
                room.index.capacity(),
                room.index.len(),
                reopen_probe_load_factor(&room),
                room.generation,
                d.base_generations.get(&REOPEN_PROBE_COLLECTION),
            );
            drop(d);
            drop(room);

            let before = store.stats();
            reopen_probe_put_range(&store, total..(total + 32), 32, |_| {
                NodeData::new(bytes::Bytes::from_static(b"y"))
            });
            let after_put = store.stats();
            let d = store.delta_state.lock();
            eprintln!(
                "AFTER REOPEN PUT: clones+={} grows+={} invalids+={} | pending={} invalid={}",
                after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
                after_put.index_grow_count - before.index_grow_count,
                after_put.delta_invalidations - before.delta_invalidations,
                d.pending.len(),
                d.invalid,
            );
            assert_eq!(
                after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
                1
            );
            assert_eq!(after_put.index_grow_count - before.index_grow_count, 1);
            assert_eq!(
                after_put.delta_invalidations - before.delta_invalidations,
                1
            );
            assert!(d.invalid);
            drop(d);
            store.sync().unwrap();
            let st = store.stats();
            let ts = store.sync_timings().unwrap();
            eprintln!(
                "REOPEN SYNC TIMINGS: flush={:?} fsync={:?} checkpoint={:?} delta={:?} total={:?}",
                ts.pack_flush, ts.pack_fsync, ts.checkpoint, ts.delta_log, ts.total
            );
            eprintln!(
                "REOPEN SYNC STATS: ckpt_writes+={} delta_appends+={}",
                st.checkpoint_writes - before.checkpoint_writes,
                st.delta_appends - before.delta_appends,
            );
            assert_eq!(st.checkpoint_writes - before.checkpoint_writes, 1);
            assert_eq!(st.delta_appends - before.delta_appends, 0);
            assert!(ts.checkpoint > std::time::Duration::ZERO);
            assert_eq!(ts.delta_log, std::time::Duration::ZERO);

            reopen_probe_put_range(&store, (total + 32)..(total + 288), 256, |_| {
                NodeData::new(bytes::Bytes::from_static(b"z"))
            });
            store.sync().unwrap();
            let ts = store.sync_timings().unwrap();
            eprintln!(
                "SECOND SYNC TIMINGS: checkpoint={:?} delta={:?} total={:?}",
                ts.checkpoint, ts.delta_log, ts.total
            );
            assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
            assert!(ts.delta_log > std::time::Duration::ZERO);
        }
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
                    &TEST_COLLECTION,
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
                &TEST_COLLECTION,
                &extra,
                &NodeData::new(bytes::Bytes::from_static(b"y")),
            )
            .unwrap();

        let got = store.get(&TEST_COLLECTION, &extra).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"y"));
    }

    #[test]
    fn test_put_many_fast_path_mutates_live_index_without_cloning() {
        // Regression coverage for put_many's live-index fast path: prove it
        // through observable behavior (every record readable, across
        // several separate put_many calls to the same collection, none of
        // which needs to grow) rather than asserting the clone didn't
        // happen internally.
        let dir = test_dir("put_many_fast_path_live_mutation");
        let store = PackfileStorage::open(dir).unwrap();

        // First call materializes the collection (no live index to reuse
        // yet); every call after it should take the fast path.
        for batch in 0..8u32 {
            let entries: Vec<_> = (0..20u32)
                .map(|i| {
                    let mut id = [0u8; 16];
                    id[0..4].copy_from_slice(&batch.to_le_bytes());
                    id[4..8].copy_from_slice(&i.to_le_bytes());
                    (
                        id,
                        NodeData::new(bytes::Bytes::from(format!("v{batch}-{i}"))),
                    )
                })
                .collect();
            store.put_many(&TEST_COLLECTION, &entries).unwrap();
        }

        for batch in 0..8u32 {
            for i in 0..20u32 {
                let mut id = [0u8; 16];
                id[0..4].copy_from_slice(&batch.to_le_bytes());
                id[4..8].copy_from_slice(&i.to_le_bytes());
                let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
                assert_eq!(got.bytes, bytes::Bytes::from(format!("v{batch}-{i}")));
            }
        }
    }

    #[test]
    fn test_runtime_stats_counters_round_trip() {
        let dir = test_dir("runtime_stats_counters");
        let store = PackfileStorage::open(dir).unwrap();

        // Fresh store: every counter starts at zero (open_count counts the
        // one store assembled, not work).
        let snapshot = store.stats();
        assert_eq!(snapshot.open_count, 1);
        assert_eq!(snapshot.put_calls, 0);
        assert_eq!(snapshot.put_many_calls, 0);
        assert_eq!(snapshot.sync_calls, 0);
        assert_eq!(snapshot.get_calls, 0);

        // Read counters are opt-in: with stats disabled a get is invisible.
        store
            .put(
                &TEST_COLLECTION,
                &[0u8; 16],
                &NodeData::new(bytes::Bytes::from("solo")),
            )
            .unwrap();
        assert_eq!(store.stats().put_calls, 1);
        assert!(store.get(&TEST_COLLECTION, &[0u8; 16]).unwrap().is_some());
        assert_eq!(store.stats().get_calls, 0);
        assert_eq!(store.stats().get_misses, 0);

        // Enable read tracking, then a hit and a miss both count.
        store.set_stats_enabled(true);
        assert!(store.get(&TEST_COLLECTION, &[0u8; 16]).unwrap().is_some());
        assert!(store
            .get(&TEST_COLLECTION, &[0xFFu8; 16])
            .unwrap()
            .is_none());
        let snapshot = store.stats();
        assert_eq!(snapshot.get_calls, 2);
        assert_eq!(snapshot.get_misses, 1);

        // Batched writes: the first put_many for a brand-new second collection
        // materializes up front (clone path), later batches ride the owned
        // no-clone fast path until a grow forces a fresh materialization;
        // every call is classified as exactly one of the two.
        let batches: Vec<Vec<(NodeId, NodeData)>> = (0..8u32)
            .map(|batch| {
                (0..8u32)
                    .map(|i| {
                        let mut id = [2u8; 16];
                        id[0..4].copy_from_slice(&batch.to_le_bytes());
                        id[4..8].copy_from_slice(&i.to_le_bytes());
                        (
                            id,
                            NodeData::new(bytes::Bytes::from(format!("b{batch}-{i}"))),
                        )
                    })
                    .collect()
            })
            .collect();
        for entries in &batches {
            store.put_many(&SECOND_COLLECTION, entries).unwrap();
        }
        let snapshot = store.stats();
        assert_eq!(snapshot.put_many_calls, 8);
        assert_eq!(snapshot.put_many_records, 64);
        assert!(snapshot.put_many_bytes > 0);
        let classified = snapshot.put_many_fast_path_calls + snapshot.put_many_clone_path_calls;
        assert_eq!(classified, 8, "every put_many call is one of the two paths");
        assert!(snapshot.put_many_clone_path_calls >= 1);
        assert!(snapshot.put_many_fast_path_calls >= 1);
        // 64 distinct records into a floor-sized 64-slot index must cross the
        // grow threshold at least once.
        assert!(snapshot.index_grow_count >= 1);
        assert!(snapshot.index_clone_time > std::time::Duration::ZERO);

        // Every put_many byte is the sum of entry payloads.
        let expected_bytes: u64 = batches
            .iter()
            .flatten()
            .map(|(_, data)| data.bytes.len() as u64)
            .sum();
        assert_eq!(snapshot.put_many_bytes, expected_bytes);

        // reset_stats zeroes the runtime counters but not open_count.
        store.reset_stats();
        let snapshot = store.stats();
        assert_eq!(snapshot.open_count, 1);
        assert_eq!(snapshot.put_many_calls, 0);
        assert_eq!(snapshot.get_calls, 0);
        assert_eq!(snapshot.get_misses, 0);

        // Sync accounting: the first (structurally invalidated) sync rewrites
        // the checkpoint; a later clean batch sync appends the delta log.
        store.sync().unwrap();
        assert_eq!(store.stats().sync_calls, 1);
        assert_eq!(store.stats().checkpoint_writes, 1);
        assert_eq!(store.stats().delta_appends, 0);
        store.put_many(&SECOND_COLLECTION, &batches[0]).unwrap();
        store.sync().unwrap();
        let snapshot = store.stats();
        assert_eq!(snapshot.sync_calls, 2);
        assert_eq!(snapshot.delta_appends, 1);
    }

    #[test]
    fn test_put_many_growth_mid_batch_keeps_records_inserted_before_the_grow() {
        // Exercises the fast path's trickiest transition: some records in
        // the batch insert via the live, shared index (no clone), then a
        // later record in the *same* batch forces a grow — which must
        // materialize an owned index that still contains everything the
        // live path already applied, not just what comes after the grow.
        let dir = test_dir("put_many_growth_mid_batch");
        let store = PackfileStorage::open(dir).unwrap();

        // Get the collection's live index past its NEW_COLLECTION_INDEX_FLOOR
        // starting capacity via ordinary put_many calls (fast path), then
        // issue one big batch that must cross the 75%-load grow threshold
        // partway through.
        let seed: Vec<_> = (0..40u32)
            .map(|i| {
                let mut id = [1u8; 16];
                id[4..8].copy_from_slice(&i.to_le_bytes());
                (id, NodeData::new(bytes::Bytes::from(format!("seed{i}"))))
            })
            .collect();
        store.put_many(&TEST_COLLECTION, &seed).unwrap();

        let growth_batch: Vec<_> = (0..200u32)
            .map(|i| {
                let mut id = [2u8; 16];
                id[4..8].copy_from_slice(&i.to_le_bytes());
                (id, NodeData::new(bytes::Bytes::from(format!("grow{i}"))))
            })
            .collect();
        store.put_many(&TEST_COLLECTION, &growth_batch).unwrap();

        for (id, data) in seed.iter().chain(&growth_batch) {
            let got = store.get(&TEST_COLLECTION, id).unwrap().unwrap();
            assert_eq!(
                &got.bytes, &data.bytes,
                "record must survive the mid-batch grow"
            );
        }
    }

    #[test]
    fn test_index_offset_gate_rejects_unrepresentable_offsets() {
        let hash = [0x5A; 16];
        assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT).is_ok());
        assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT + 1).is_err());
        assert!(check_index_offset(0, &hash, u64::MAX).is_err());
    }

    #[test]
    fn test_refresh_collection_multi_worker_visibility() {
        let dir = test_dir("refresh_collection_multi_worker");
        let writer = PackfileStorage::open(dir.clone()).unwrap();

        // Open a read-only instance BEFORE the writer writes the data.
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        let id = [0xAA; 16];
        writer
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"multi_worker_test")),
            )
            .unwrap();
        writer.sync().unwrap();

        // Reader's in-memory index should be completely unaware of the new write.
        assert!(reader.get(&TEST_COLLECTION, &id).unwrap().is_none());

        // Trigger the external refresh, simulating a cache invalidation signal.
        reader.refresh_collection(&TEST_COLLECTION).unwrap();

        // Reader should now have correctly loaded the delta and rebuilt its index.
        let got = reader.get(&TEST_COLLECTION, &id).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"multi_worker_test"));
        assert_eq!(
            reader
                .collection_index_info(&TEST_COLLECTION)
                .expect("refreshed collection exists")
                .2,
            u32::try_from(NEW_COLLECTION_INDEX_FLOOR).unwrap(),
            "a refresh rebuild keeps the same minimum capacity as a new collection"
        );
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

        let valid_path = dir.join("pack_0000000000000000.pack");
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                collection_id: [0x01; 16],
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"hello"),
            },
        )
        .unwrap();
        std::fs::write(&valid_path, &buf).unwrap();

        let store = PackfileStorage::open(dir).unwrap();
        let gen = store.generation(&[0x01u8; 16]).unwrap();
        assert_eq!(gen.index.len(), 1);
    }

    #[test]
    fn test_delete_collection_preserves_other_collection_cache() {
        let dir = test_dir("delete_collection_cache");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let collection_a = TEST_COLLECTION;
        let collection_b = OTHER_COLLECTION;
        let id_a = [0x10u8; 16];
        let id_b = [0x20u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

        store.put(&collection_a, &id_a, &data_a).unwrap();
        store.put(&collection_b, &id_b, &data_b).unwrap();

        assert!(store.get(&collection_a, &id_a).unwrap().is_some());
        assert!(store.get(&collection_b, &id_b).unwrap().is_some());

        let gen_b_before = store.generation(&collection_b).unwrap();
        let hits_b_before = gen_b_before.cache.hits();

        store.delete_collection(&collection_a).unwrap();

        assert!(store.get(&collection_a, &id_a).unwrap().is_none());
        assert!(store.generation(&collection_a).is_none());
        assert!(store.generation(&collection_b).is_some());
        assert!(store.get(&collection_b, &id_b).unwrap().is_some());

        let gen_b_after = store.generation(&collection_b).unwrap();
        assert!(gen_b_after.cache.hits() > hits_b_before);
    }

    #[test]
    fn test_repack_collection_reachable_no_roots_preserves_diamond_dag_in_topo_order() {
        let dir = test_dir("repack_topo");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;
        let mut id_c = [0u8; 16];
        id_c[0] = 3;
        let mut id_d = [0u8; 16];
        id_d[0] = 4;

        store
            .put(
                &TEST_COLLECTION,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_c,
                &NodeData::new(bytes::Bytes::from_static(b"C")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_d,
                &NodeData::new(bytes::Bytes::from_static(b"D")),
            )
            .unwrap();

        let edges = std::collections::HashMap::from([
            (id_b, vec![id_a]),
            (id_c, vec![id_a]),
            (id_d, vec![id_b, id_c]),
        ]);

        // No live roots configured for TEST_COLLECTION: preserves everything,
        // still deduplicated and topologically ordered.
        let result = store.repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        });
        let (kept, dropped) = result.unwrap();
        assert_eq!(kept, 4);
        assert_eq!(dropped, 0);

        for (id, expected) in [
            (id_a, b"A".as_slice()),
            (id_b, b"B".as_slice()),
            (id_c, b"C".as_slice()),
            (id_d, b"D".as_slice()),
        ] {
            let got = store
                .get(&TEST_COLLECTION, &id)
                .unwrap()
                .expect("record missing after topo repack");
            assert_eq!(got.bytes.as_ref(), expected);
        }
    }

    #[test]
    fn test_repack_collection_reachable_drops_unreachable_records() {
        let dir = test_dir("repack_reachable_gc");
        let store = PackfileStorage::open(dir).unwrap();

        // Live chain: root -> p1 -> p0 (p0 has no further dependencies).
        let root = distinct_id(1);
        let p1 = distinct_id(2);
        let p0 = distinct_id(3);

        // Garbage island, unreachable from root: garbage -> garbage_dep.
        let garbage = distinct_id(4);
        let garbage_dep = distinct_id(5);

        for (id, bytes) in [
            (root, b"root".as_slice()),
            (p1, b"p1".as_slice()),
            (p0, b"p0".as_slice()),
            (garbage, b"garbage".as_slice()),
            (garbage_dep, b"garbage_dep".as_slice()),
        ] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(bytes)),
                )
                .unwrap();
        }

        let edges = std::collections::HashMap::from([
            (root, vec![p1]),
            (p1, vec![p0]),
            (garbage, vec![garbage_dep]),
        ]);

        store.set_live_roots(&TEST_COLLECTION, vec![root]);

        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();

        assert_eq!(kept, 3, "root, p1, p0 should survive");
        assert_eq!(dropped, 2, "garbage and garbage_dep should be collected");

        for id in [root, p1, p0] {
            assert!(
                store.get(&TEST_COLLECTION, &id).unwrap().is_some(),
                "live record missing after reachable repack"
            );
        }
        for id in [garbage, garbage_dep] {
            assert!(
                store.get(&TEST_COLLECTION, &id).unwrap().is_none(),
                "garbage record survived reachable repack"
            );
        }

        let stats = store.repack_stats();
        assert_eq!(stats.repack_count, 1);
        assert_eq!(stats.kept_total, 3);
        assert_eq!(stats.dropped_total, 2);
        assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 1);

        // A second repack uses the incremental path: it clones the
        // previous live_map (3 entries from the first repack's output)
        // and scans only newly-appended bytes (none here).  The garbage
        // entries in the old shard are never re-scanned — the live_map
        // already excludes them, so dropped is 0.  This is the O(n)
        // behavior: each repack does bounded work proportional to new
        // bytes, not proportional to total shard size.
        let (kept2, dropped2) = store
            .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();
        assert_eq!((kept2, dropped2), (3, 0));

        let stats2 = store.repack_stats();
        assert_eq!(stats2.repack_count, 2);
        assert_eq!(stats2.kept_total, stats.kept_total + kept2 as u64);
        assert_eq!(stats2.dropped_total, stats.dropped_total + dropped2 as u64);
        assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 2);
    }

    /// Verify the root-removal fallback: if a live root is removed between
    /// repacks, nodes exclusively reachable through it must not stay in the
    /// cached live set indefinitely. Without the fallback, the incremental
    /// adjacency cache only grows and those nodes would never be collected.
    #[test]
    fn test_repack_incremental_root_removal_forces_full_sweep() {
        let dir = test_dir("repack_root_removal_fallback");
        let store = PackfileStorage::open(dir).unwrap();

        // Graph:
        //   root_a -> shared -> base
        //   root_b -> shared -> base
        //   orphan_a -> orphan_dep    (reachable only via root_a)
        //
        // First repack: both root_a and root_b are live. Everything survives.
        // Second repack: only root_b is live. orphan_a and orphan_dep must be
        // collected. Without the root-removal fallback they would remain in the
        // incremental live set permanently.
        let root_a = distinct_id(1);
        let root_b = distinct_id(2);
        let shared = distinct_id(3);
        let base = distinct_id(4);
        let orphan_a = distinct_id(5);
        let orphan_dep = distinct_id(6);

        for (id, bytes) in [
            (root_a, b"root_a".as_slice()),
            (root_b, b"root_b".as_slice()),
            (shared, b"shared".as_slice()),
            (base, b"base".as_slice()),
            (orphan_a, b"orphan_a".as_slice()),
            (orphan_dep, b"orphan_dep".as_slice()),
        ] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(bytes)),
                )
                .unwrap();
        }

        let edges = std::collections::HashMap::from([
            (root_a, vec![shared, orphan_a]),
            (root_b, vec![shared]),
            (shared, vec![base]),
            (orphan_a, vec![orphan_dep]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| -> Vec<[u8; 16]> {
            edges.get(hash).cloned().unwrap_or_default()
        };

        // First repack: both roots live — everything reachable from either root
        // survives and the incremental adjacency cache is populated.
        store.set_live_roots(&TEST_COLLECTION, vec![root_a, root_b]);
        let (kept1, dropped1) = store
            .repack_collection_reachable(&TEST_COLLECTION, extract)
            .unwrap();
        assert_eq!(kept1, 6, "all nodes should survive with both roots live");
        assert_eq!(dropped1, 0);

        // Now remove root_a — switch to root_b only.
        // orphan_a and orphan_dep are now exclusively reachable through root_a,
        // which was removed. The root-removal fallback must trigger a full BFS
        // sweep so they are actually collected, rather than remaining in the
        // stale incremental live set forever.
        store.set_live_roots(&TEST_COLLECTION, vec![root_b]);
        let (kept2, dropped2) = store
            .repack_collection_reachable(&TEST_COLLECTION, extract)
            .unwrap();
        assert_eq!(
            kept2, 3,
            "root_b, shared, base should survive; root_a, orphan_a, orphan_dep must be GC'd"
        );
        assert_eq!(
            dropped2, 3,
            "root_a, orphan_a, and orphan_dep must be collected once root_a is removed from live roots"
        );

        assert!(
            store.get(&TEST_COLLECTION, &orphan_a).unwrap().is_none(),
            "orphan_a must be gone after root_a was removed"
        );
        assert!(
            store.get(&TEST_COLLECTION, &orphan_dep).unwrap().is_none(),
            "orphan_dep must be gone after root_a was removed"
        );
        assert!(
            store.get(&TEST_COLLECTION, &root_a).unwrap().is_none(),
            "root_a itself has no incoming edges so it is unreachable from root_b"
        );
        assert!(
            store.get(&TEST_COLLECTION, &root_b).unwrap().is_some(),
            "root_b must still be present"
        );
        assert!(
            store.get(&TEST_COLLECTION, &shared).unwrap().is_some(),
            "shared must still be present"
        );
        assert!(
            store.get(&TEST_COLLECTION, &base).unwrap().is_some(),
            "base must still be present"
        );
    }

    /// Repack drops unreachable *records* from the index (`dropped` count),
    /// but today nothing ever retires the *shard file* they lived in: no
    /// code path sets `Shard::is_current` to `false` or clears the pool's
    /// slot for it (see `Shard::drop`'s doc and
    /// `test_drop_deletes_retired_shard_only_after_last_reference` in
    /// shard.rs, which has to simulate retirement manually because nothing
    /// production triggers it). `rotate()`'s error message says "repack to
    /// reclaim", but repack currently reclaims nothing at the shard level.
    ///
    /// This fills a small fixed number of pool slots (4), repacks away
    /// everything in the non-active ones (100% garbage, nothing live left),
    /// and then expects one more rotation to succeed by reusing a
    /// now-empty slot.
    #[test]
    fn test_repack_reclaims_shard_slots_for_rotation() {
        // Use a fixed small count rather than MAX_SHARDS — with 4096 slots
        // and 256MB each that would be a 1TB test.
        const TEST_SHARDS: usize = 4;

        let dir = test_dir("repack_reclaims_slots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root = [0u8; 16];
        root[0] = 0xFF;
        store
            .put(
                &TEST_COLLECTION,
                &root,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();

        // `root` already occupies the first slot; force rotation through
        // the remaining TEST_SHARDS - 1 slots, dumping garbage into each so
        // every shard but the last ends up fully unreachable once we
        // repack with `root` as the only live node.
        for i in 0..TEST_SHARDS - 1 {
            store.shards.active_shard().file_len.store(
                shard::MAX_SHARD_BYTES - 10,
                std::sync::atomic::Ordering::Release,
            );
            let mut garbage = [0u8; 16];
            garbage[0] = u8::try_from(i + 1).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    &garbage,
                    &NodeData::new(bytes::Bytes::from_static(b"garbage")),
                )
                .unwrap();
        }

        // Every slot should now be occupied.
        let occupied = (0..u16::try_from(TEST_SHARDS).unwrap())
            .filter(|&id| store.shards.get_shard(id).is_some())
            .count();
        assert_eq!(
            occupied, TEST_SHARDS,
            "test setup should have filled every shard slot"
        );

        store.set_live_roots(&TEST_COLLECTION, vec![root]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1, "only root should survive");
        assert!(dropped > 0, "the garbage records should have been dropped");
        assert!(
            store.shards_retired() > 0,
            "retire_empty_shards should have retired at least one now-garbage-only shard"
        );

        // Every shard except the current active one held nothing but
        // garbage, and that garbage is now unreachable -- those slots
        // should be reclaimable. Forcing one more rotation should reuse
        // one of them rather than failing.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut one_more = [0u8; 16];
        one_more[0] = 0xEE;
        store
            .put(
                &TEST_COLLECTION,
                &one_more,
                &NodeData::new(bytes::Bytes::from_static(b"one more")),
            )
            .expect(
                "rotation should reclaim a garbage-only shard slot after repack, \
                 not report the pool as permanently full",
            );
    }

    #[test]
    fn test_repack_moves_live_data_out_of_its_source_shard() {
        let dir = test_dir("repack_moves_from_source");
        let store = PackfileStorage::open(dir).unwrap();
        let node = [0xA5; 16];
        store
            .put(
                &TEST_COLLECTION,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"live")),
            )
            .unwrap();

        let source = store
            .collection_referenced_shards(&TEST_COLLECTION)
            .into_iter()
            .next()
            .unwrap();
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        assert_eq!((kept, dropped), (1, 0));
        let destination = store
            .collection_referenced_shards(&TEST_COLLECTION)
            .into_iter()
            .next()
            .unwrap();
        assert_ne!(destination, source);
        assert!(store.shards.get_shard(source).is_none());
        assert_eq!(
            store
                .get(&TEST_COLLECTION, &node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"live"
        );
    }

    /// Repack must not retire a shard that another collection's index still
    /// references.  This test puts live records from two different collections
    /// into the same shard, then repacks only collection A — the shared shard
    /// must survive because collection B's index still points into it.
    #[test]
    fn test_repack_does_not_retire_shard_referenced_by_other_collection() {
        let dir = test_dir("repack_cross_collection_safety");
        let store = PackfileStorage::open(dir).unwrap();

        // Put a record for collection A — lands on shard 0.
        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        store
            .put(
                &TEST_COLLECTION,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
            )
            .unwrap();

        // Force a rotation so the next write goes to a different shard.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );

        // Put a record for collection B — lands on shard 1.
        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
            )
            .unwrap();

        // Force another rotation and put more collection A garbage on shard 2.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = 0xCC;
        store
            .put(
                &TEST_COLLECTION,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();

        // Repack collection A with only root_a as live.  The garbage record
        // should be dropped, but shard 1 (which holds collection B's root)
        // must NOT be retired.
        store.set_live_roots(&TEST_COLLECTION, vec![root_a]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1);
        assert!(dropped > 0);

        // Collection B's root must still be retrievable — shard 1 was not retired.
        let got = store.get(&OTHER_COLLECTION, &root_b).unwrap();
        assert!(
            got.is_some(),
            "collection B root must survive repack of collection A — shared shard must not be retired",
        );
    }

    /// `open_read_only` must actually coexist with a live writer end to
    /// end — not just take no lock, but also not touch/truncate the
    /// writer's files via the collection-index rebuild scan (the real risk this
    /// whole design exists to avoid). Also confirms it sees the writer's
    /// already-durable data and that the writer is unaffected afterward.
    #[test]
    fn test_open_read_only_coexists_with_active_writer() {
        let dir = test_dir("open_read_only_coexist");
        let writer = PackfileStorage::open(dir.clone()).unwrap();

        let mut id = [0u8; 16];
        id[0] = 0xAA;
        writer
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"hello")),
            )
            .unwrap();
        // Reads through a second (read-only) process open against the same
        // directory only observe committed bytes, so make the write durable
        // before opening the reader — buffered bytes are RAM-only.
        writer.sync_all().unwrap();

        // Open read-only while the writer is still alive — must succeed
        // (no lock conflict) and see the write above.
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
        assert_eq!(
            reader.get(&TEST_COLLECTION, &id).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(b"hello"))
        );
        drop(reader);

        // The writer must be completely unaffected — still open, still
        // able to write more data afterward.
        let mut id2 = [0u8; 16];
        id2[0] = 0xBB;
        writer
            .put(
                &TEST_COLLECTION,
                &id2,
                &NodeData::new(bytes::Bytes::from_static(b"world")),
            )
            .unwrap();
        assert_eq!(
            writer.get(&TEST_COLLECTION, &id2).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(b"world"))
        );
    }

    /// `repack_shard` must find and repack every collection sharing a shard —
    /// the targeted way to reclaim one shard without waiting for each
    /// collection to independently cross its own repack threshold.
    #[test]
    fn test_repack_shard_repacks_every_referencing_collection() {
        let dir = test_dir("repack_shard");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        let mut garbage_a = [0u8; 16];
        garbage_a[0] = 0xA1;
        store
            .put(
                &TEST_COLLECTION,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &garbage_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A garbage")),
            )
            .unwrap();
        store.set_live_roots(&TEST_COLLECTION, vec![root_a]);

        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
            )
            .unwrap();
        // No set_live_roots for collection B: everything must survive its repack.

        // Fresh pool, both collections' first writes land on shard 0 (shared).
        let referencing = store.collections_referencing_shard(0).unwrap();
        assert_eq!(referencing.len(), 2);
        assert!(referencing.contains(&TEST_COLLECTION));
        assert!(referencing.contains(&OTHER_COLLECTION));

        let mut results = store.repack_shard(0, |_hash, _data| Vec::new()).unwrap();
        results.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);

        let mut expected = vec![
            (TEST_COLLECTION, 1usize, 1usize),
            (OTHER_COLLECTION, 1usize, 0usize),
        ];
        expected.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);
        assert_eq!(
            results, expected,
            "repack_shard must repack both collections sharing shard 0"
        );

        assert!(store.get(&TEST_COLLECTION, &root_a).unwrap().is_some());
        assert!(store.get(&TEST_COLLECTION, &garbage_a).unwrap().is_none());
        assert!(store.get(&OTHER_COLLECTION, &root_b).unwrap().is_some());
    }

    /// `collections_referencing_shard` must filter the shard-scan's candidate
    /// set against each collection's *current* index, not just report every collection
    /// whose bytes ever physically touched the shard. A collection that has
    /// since repacked its live data onto a different shard leaves its old
    /// bytes sitting there untouched (shards are append-only, never
    /// rewritten in place) — that collection must NOT show up as still
    /// referencing the shard its data moved away from.
    #[test]
    fn test_collections_referencing_shard_excludes_collections_already_repacked_away() {
        let dir = test_dir("collections_referencing_shard_stale");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root = [0u8; 16];
        root[0] = 0xAA;
        store
            .put(
                &TEST_COLLECTION,
                &root,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();

        // OTHER_COLLECTION also lands on shard 0 (shared, still the pool's
        // active shard) and stays live there — this is what keeps shard 0
        // from being retired outright once TEST_COLLECTION moves off it, so the
        // "stale reference" case below is actually reachable to query.
        let mut other_root = [0u8; 16];
        other_root[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &other_root,
                &NodeData::new(bytes::Bytes::from_static(b"other collection root")),
            )
            .unwrap();

        // Force TEST_COLLECTION's home (shard 0) to look full, so its next write
        // rotates *its own* home to shard 1 — root stays physically on
        // shard 0, but TEST_COLLECTION's home (and this next record, `garbage`)
        // moves to shard 1.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = 0xA1;
        store
            .put(
                &TEST_COLLECTION,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();

        // Repack with only root live: TEST_COLLECTION's records now span *two*
        // source shards (root on 0, garbage on 1), and a repack must never
        // rewrite live data back into one of its own sources (see
        // `ShardPool::prepare_collection_repack`'s doc) — so the kept copy lands
        // on a *third* shard (2), not shard 1. Shard 1, left holding only
        // the now-dropped `garbage` record with nothing live pointing at
        // it, gets retired and closed during this same repack.
        store.set_live_roots(&TEST_COLLECTION, vec![root]);
        store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        // Shard 0's raw bytes still physically contain TEST_COLLECTION's
        // original root record — a naive scan-only approach would still
        // report it. The current index must exclude it.
        assert!(
            !store.collections_referencing_shard(0).unwrap().contains(&TEST_COLLECTION),
            "a collection whose live data has moved off a shard must not be reported as still referencing it"
        );

        // The real invariant here isn't "the destination is shard 2" —
        // that's just where a deterministic, empty-pool allocator happens
        // to land today. What must hold is: exactly one destination, and
        // it's neither of the two source shards (0 and 1) a repack must
        // never write live data back into.
        let destinations = store.collection_referenced_shards(&TEST_COLLECTION);
        assert_eq!(
            destinations.len(),
            1,
            "TEST_COLLECTION's live data must land on exactly one shard after repack"
        );
        let destination = destinations[0];
        assert!(
            ![0, 1].contains(&destination),
            "the repack destination must not be one of its own source shards (0: root, 1: garbage)"
        );
        assert!(
            store
                .collections_referencing_shard(destination)
                .unwrap()
                .contains(&TEST_COLLECTION),
            "the collection's current shard must still be reported"
        );
    }

    #[test]
    fn test_collection_directory_from_disk_matches_collection_summaries_after_sync() {
        let dir = test_dir("collection_directory_roundtrip");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        for i in 0..5u8 {
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        for i in 0..3u8 {
            store
                .put(
                    &OTHER_COLLECTION,
                    &distinct_id(100 + i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        store.sync_all().unwrap();

        let mut from_disk = PackfileStorage::collection_directory_from_disk(&dir);
        from_disk.sort_unstable_by_key(|(collection_id, _)| *collection_id);

        let mut expected: Vec<([u8; 16], u64)> = store
            .collection_summaries()
            .into_iter()
            .map(|(collection_id, count, _mem, _cap)| (collection_id, count as u64))
            .collect();
        expected.sort_unstable_by_key(|(collection_id, _)| *collection_id);

        assert_eq!(
            from_disk, expected,
            "the persisted directory's per-collection totals must match the live index's own counts"
        );
        assert!(PackfileStorage::collection_directory_persisted_at(&dir).is_some());
    }

    #[test]
    fn test_collection_summaries_from_disk_capacity_matches_new_collection_floor() {
        // A one-node collection's live index starts at
        // `NEW_COLLECTION_INDEX_FLOOR` (64), never `LossyIndex::new`'s own
        // generic 16-slot floor. The disk-only estimate (no packfiles
        // opened) must report the same capacity the live index actually
        // has, or a load-factor figure derived from it is wrong.
        let dir = test_dir("collection_summaries_from_disk_capacity_floor");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.sync_all().unwrap();

        let live_capacity = store
            .collection_index_info(&TEST_COLLECTION)
            .expect("collection exists")
            .2;
        assert_eq!(
            live_capacity, 64,
            "sanity check: a fresh one-node collection's live index capacity \
             is NEW_COLLECTION_INDEX_FLOOR"
        );

        // Force `open` down its packfile-scan path rather than letting it use
        // the serialized checkpoint, then ensure the reconstructed index has
        // the same floor as the original collection.
        drop(store);
        fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_eq!(
            reopened
                .collection_index_info(&TEST_COLLECTION)
                .expect("scanned collection exists")
                .2,
            u32::try_from(NEW_COLLECTION_INDEX_FLOOR).unwrap(),
            "an open-time scan keeps the new-collection index floor"
        );

        let from_disk = PackfileStorage::collection_summaries_from_disk(&dir)
            .expect("directory sidecar was persisted by sync_all");
        let (_, disk_nodes, _, disk_capacity) = from_disk
            .into_iter()
            .find(|(id, _, _, _)| *id == TEST_COLLECTION)
            .expect("collection present in disk summary");
        assert_eq!(disk_nodes, 1);
        assert_eq!(
            disk_capacity, live_capacity,
            "disk-only capacity estimate must match the live index, not undershoot it"
        );
    }

    #[test]
    fn test_collection_directory_from_disk_empty_when_never_persisted() {
        let dir = test_dir("collection_directory_never_persisted");
        let _store = PackfileStorage::open(dir.clone()).unwrap();
        // No sync_all call: nothing has been persisted yet.
        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            Vec::new()
        );
        assert_eq!(
            PackfileStorage::collection_directory_persisted_at(&dir),
            None
        );
    }

    #[test]
    fn test_collection_directory_reflects_delete_and_repack() {
        let dir = test_dir("collection_directory_delete_repack");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let a = distinct_id(0);
        let b = distinct_id(1);
        store
            .put(
                &TEST_COLLECTION,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"a")),
            )
            .unwrap();
        store
            .put(
                &OTHER_COLLECTION,
                &b,
                &NodeData::new(bytes::Bytes::from_static(b"b")),
            )
            .unwrap();
        store.sync_all().unwrap();

        let directory = PackfileStorage::collection_directory_from_disk(&dir);
        assert_eq!(
            directory.len(),
            2,
            "both collections must appear before deletion"
        );

        store.delete_collection(&TEST_COLLECTION).unwrap();
        store.sync_all().unwrap();

        let directory = PackfileStorage::collection_directory_from_disk(&dir);
        assert_eq!(
            directory,
            vec![(OTHER_COLLECTION, 1)],
            "a deleted collection must not linger in the persisted directory"
        );
    }

    #[test]
    fn test_plan_collection_repack_matches_real_repack_without_mutating() {
        let dir = test_dir("plan_matches_real");
        let store = PackfileStorage::open(dir).unwrap();

        for i in 0..5u8 {
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        // Duplicate a hash's payload under a fresh id to create something
        // dedup would drop... actually LossyIndex already dedups by hash
        // on insert, so instead exercise the "no live roots: keep
        // everything" path, which is what --root-less usage always hits.
        let plan = store
            .plan_collection_repack(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(plan.kept, 5);
        assert_eq!(plan.dropped, 0);
        assert!(plan.kept_bytes > 0);
        assert_eq!(plan.shards_touched, vec![0]);

        // The dry run must not have mutated anything: a real repack run
        // right after must see the exact same collection state and produce the
        // exact same result.
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, plan.kept);
        assert_eq!(dropped, plan.dropped);
    }

    #[test]
    fn test_repack_closure_single_collection_single_shard() {
        let dir = test_dir("closure_trivial");
        let store = PackfileStorage::open(dir).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();

        let (collections, shards) = store.repack_closure(0).unwrap();
        assert_eq!(collections, vec![TEST_COLLECTION]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_repack_closure_pulls_in_collections_second_shard_transitively() {
        // Collection A lives on shards {0, 1} (spans a rotation). Collection B lives
        // only on shard 1. Starting the closure from shard 0 must still
        // discover shard 1 (because collection A references it) and, through
        // shard 1, collection B — even though collection B never touched shard 0 at
        // all.
        let dir = test_dir("closure_transitive");
        let store = PackfileStorage::open(dir).unwrap();

        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-0")),
            )
            .unwrap();

        // Force rotation so TEST_COLLECTION's next write lands on a new shard.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(1),
                &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-1")),
            )
            .unwrap();

        store
            .put(
                &OTHER_COLLECTION,
                &distinct_id(2),
                &NodeData::new(bytes::Bytes::from_static(b"b-on-shard-1")),
            )
            .unwrap();

        let (mut collections, mut shards) = store.repack_closure(0).unwrap();
        collections.sort_unstable();
        shards.sort_unstable();
        let mut expected_collections = vec![TEST_COLLECTION, OTHER_COLLECTION];
        expected_collections.sort_unstable();
        assert_eq!(collections, expected_collections);
        assert_eq!(shards, vec![0, 1]);
    }

    #[test]
    fn test_collection_referenced_shards_empty_for_unknown_collection() {
        let dir = test_dir("referenced_shards_unknown");
        let store = PackfileStorage::open(dir).unwrap();
        assert_eq!(
            store.collection_referenced_shards(&TEST_COLLECTION),
            Vec::<u16>::new()
        );
    }

    #[test]
    fn test_repack_collection_reachable_without_live_roots_preserves_everything() {
        let dir = test_dir("repack_reachable_no_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;

        store
            .put(
                &TEST_COLLECTION,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();

        // No set_live_roots call for this collection: nothing is known to be
        // garbage, so everything must survive.
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        assert_eq!(kept, 2);
        assert_eq!(dropped, 0);
        assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_some());
        assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_some());
    }

    /// `pin_shards` must return a distinct `Arc<Shard>` per unique id, and
    /// the returned handle must still resolve real, previously-written
    /// data — i.e. it's the live pool's own shard, not a stand-in.
    /// The deeper guarantee (pinning survives a shard being retired and
    /// its slot recycled) is `Shard::drop`'s contract, tested directly in
    /// `shard.rs` where `is_current` and the pool's internals are
    /// accessible without going through `ShardPool::rotate`, which no
    /// longer has any path to actually recycle a slot (see its doc).
    #[test]
    fn test_pin_shards_returns_readable_deduped_handles() {
        let dir = test_dir("pin_shards_basic");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x77u8; 16];
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"pin me")),
            )
            .unwrap();
        let (shard_id, offset) = store
            .generation(&TEST_COLLECTION)
            .unwrap()
            .index
            .lookup(&id)
            .expect("just-written record must be indexed");

        // Duplicate ids in the input must collapse to one pinned entry.
        let pinned = store.pin_shards([shard_id, shard_id, shard_id].into_iter());
        assert_eq!(pinned.len(), 1);

        let record = store
            .read_at(&pinned[&shard_id], offset, true)
            .expect("pinned shard must resolve the real on-disk record");
        assert_eq!(record.data.as_ref(), b"pin me");
    }

    /// NOTE: this test asserts a stronger property than the concurrent
    /// put+reachable-repack contract provides: every `put` that has returned
    /// remains present in a later repack output. A failure implies no
    /// data-loss or buffering bug — the durability/checkpoint machinery is
    /// not exercised here.
    ///
    /// An earlier revision of this comment documented a specific race (a put
    /// landing between the repack's scan boundary and its generation swap)
    /// and labelled the test KNOWN-FLAKY. That mechanism does not hold: both
    /// `put` and `repack_collection_reachable` hold the same
    /// `put_mutex(collection_id)` for their entire body, so a put cannot
    /// observe an in-progress repack at all. Re-investigation (55 runs,
    /// plain and under artificial CPU load) reproduced zero failures, and no
    /// panic output survives from the original reports, so the flakiness
    /// claim, its mechanism, and the "known-flaky" label are retracted as
    /// unconfirmed. If this test ever fails, the cause is the unchecked
    /// linearizability assumption above — investigate from that assertion,
    /// not from a presumed put/repack interleaving.
    #[test]
    fn test_concurrent_put_repack_reachable_no_lost_writes() {
        use std::thread;

        let dir = test_dir("concurrent_put_repack_reachable");
        let store = PackfileStorage::open(dir).unwrap();

        let collection = [0x66u8; 16];
        let written_count = std::sync::atomic::AtomicU32::new(0);

        // No live roots configured, so the reachable repacker must fall
        // back to "preserve everything" — this proves that fallback holds
        // even when interleaved with concurrent writes.
        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for i in 0..200u32 {
                    let mut id = [0u8; 16];
                    id[0..4].copy_from_slice(&i.to_le_bytes());
                    let data = NodeData::new(bytes::Bytes::from(format!("entry {i}")));
                    store.put(&collection, &id, &data).unwrap();
                    written_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

            let repacker = scope.spawn(|| {
                for _ in 0..5 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    let _ =
                        store.repack_collection_reachable(&collection, |_hash, _data| Vec::new());
                }
            });

            writer.join().unwrap();
            repacker.join().unwrap();
        });

        let total = written_count.load(std::sync::atomic::Ordering::Relaxed);
        let mut found = 0u32;
        for i in 0..total {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            if store.get(&collection, &id).unwrap().is_some() {
                found += 1;
            }
        }

        assert_eq!(
            found, total,
            "lost writes during concurrent put+reachable-repack"
        );
    }
}
