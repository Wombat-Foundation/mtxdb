use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::csr::Csr;
use crate::index::delta::{self, DeltaOperation, DELTA_LOG_HEADER_LEN, INDEX_DELTA_FILE};
use crate::index::format::DeltaFrame;
use crate::index::{EntryUndo, InsertError, LossyIndex};
use crate::journal::{Journal, JournalCoordinator, Mutation as JournalMutation};
use crate::packfile::{self, FrameMetadata, Record};
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{
    Digest32, DigestAlgorithm, NodeData, NodeId, NodeRef, StorageEngine, StorageError,
};
use crate::template::{CollectionMetadata, COLLECTION_METADATA_RECORD_ID};

#[cfg(feature = "multi-reader")]
mod read_journal;
#[cfg(feature = "multi-reader")]
use read_journal::ReadJournal;

/// Callback that rewrites a node's child references given resolved child data,
/// used to inline already-cached children in place of lazy hash pointers.
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[Option<Arc<NodeData>>]) -> NodeData;

/// Lowercase hex rendering of a 32-byte digest, for error messages.
fn hex32(digest: &Digest32) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

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

/// How strictly `PackfileStorage::checkpoint_scan_out` validates the live pack
/// set against a persisted checkpoint.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReloadMode {
    /// The live packs must match the checkpoint's fingerprint exactly, or a
    /// delta log must bridge the checkpoint to them. Used by `open` and crash
    /// recovery, where a mismatch means the checkpoint is stale and the caller
    /// must rescan.
    Strict,
    /// Skip the exact-pack fingerprint gate and the delta-log replay; the
    /// read-committed overlay is authoritative for the post-checkpoint suffix.
    /// Used only by `reload_index_from_checkpoint`, where the writer is still
    /// appending and its packs have already grown past the checkpoint (a strict
    /// gate would fail the read closed).
    #[cfg(feature = "multi-reader")]
    JournalBound,
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
    /// Total time spent opening the shard pool, including discovery, locking,
    /// packfile recovery, and metadata restoration.
    pub shard_open: std::time::Duration,
    /// Directory and pack-file discovery.
    pub shard_discovery: std::time::Duration,
    /// Acquiring the writable pool lock.
    pub writer_lock: std::time::Duration,
    /// Cumulative packfile recovery/validation scan time in the shard pool.
    pub packfile_recovery: std::time::Duration,
    /// Number of packfiles passed through recovery/validation.
    pub packfile_recovery_calls: u64,
    /// Time spent opening packfiles and validating their headers.
    pub packfile_open: std::time::Duration,
    /// Number of packfiles successfully opened.
    pub packfile_open_calls: u64,
    /// Restoring pool metadata and persisted shard statistics.
    pub metadata_restore: std::time::Duration,
    /// Restoring pool.meta header and reading next pack ID.
    pub pool_meta_restore: std::time::Duration,
    /// Reading and restoring persisted snapshot counters from `shard_stats.bin`.
    pub persisted_stats_restore: std::time::Duration,
    /// Writing store.meta version marker via best-effort atomic write (fresh pool only; ZERO on existing pool).
    pub store_meta_write: std::time::Duration,
    /// Persisting pool.meta reservation and syncing file contents (fresh pool only; ZERO on existing pool).
    pub pool_meta_persist: std::time::Duration,
    /// Creating the initial packfile atomically, writing its header, syncing,
    /// renaming, and performing the final directory sync (fresh pool only;
    /// ZERO on existing pool).
    pub initial_pack_create: std::time::Duration,
    /// Unattributed time inside the `metadata_restore` span.
    pub metadata_unattributed: std::time::Duration,
    /// Shard-open time not covered by the named shard phases.
    pub shard_open_unattributed: std::time::Duration,
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
    /// Number of delta-log operations validated and applied during this open:
    /// v3 slot frames, collection snapshots, and tombstones, or v2 frames
    /// (checkpoint path only; zero when no committed log was replayed). A
    /// deterministic replay-path signal, unlike the `delta_replay` duration,
    /// which can round to zero on a fast machine.
    pub delta_replay_operations: u64,
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
            shard_discovery: std::time::Duration::ZERO,
            writer_lock: std::time::Duration::ZERO,
            packfile_recovery: std::time::Duration::ZERO,
            packfile_recovery_calls: 0,
            packfile_open: std::time::Duration::ZERO,
            packfile_open_calls: 0,
            metadata_restore: std::time::Duration::ZERO,
            pool_meta_restore: std::time::Duration::ZERO,
            persisted_stats_restore: std::time::Duration::ZERO,
            store_meta_write: std::time::Duration::ZERO,
            pool_meta_persist: std::time::Duration::ZERO,
            initial_pack_create: std::time::Duration::ZERO,
            metadata_unattributed: std::time::Duration::ZERO,
            shard_open_unattributed: std::time::Duration::ZERO,
            metadata_load: std::time::Duration::ZERO,
            checkpoint_decode: std::time::Duration::ZERO,
            fingerprint: std::time::Duration::ZERO,
            index_materialization: std::time::Duration::ZERO,
            delta_replay: std::time::Duration::ZERO,
            delta_replay_operations: 0,
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
    /// Whether this barrier returned an error after collecting its partial timings.
    pub failed: bool,
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
    /// Committing the pending write-ahead journal group (one sequential
    /// fsync). Zero when no journal is configured.
    pub wal: std::time::Duration,
    /// Time spent waiting for the journal's single-writer mutex.
    pub journal_lock_wait: std::time::Duration,
    /// Time spent waiting for the pending-mutation queue mutex.
    pub journal_pending_wait: std::time::Duration,
    /// Time spent encoding and appending the journal group, excluding fsync.
    pub journal_append: std::time::Duration,
    /// Time spent making the journal file durable.
    pub journal_fsync: std::time::Duration,
    /// Number of journal sync calls represented by this operation.
    pub journal_sync_calls: u64,
    /// Bytes appended to the journal by this operation, including framing.
    pub journal_bytes: u64,
    /// Records appended to the journal by this operation.
    pub journal_records: u64,
    /// Whether this operation waited for the journal mutex.
    pub journal_waiters: u64,
    /// Whether this operation was already covered by another sync.
    pub journal_coalesced: u64,
    /// Number of journal sync callers active when this operation entered.
    pub journal_in_flight: u64,
    /// Time spent waiting for the shard pool's dirty-set lock in this barrier.
    pub dirty_lock_wait: std::time::Duration,
    /// Age of the oldest unpublished write when this barrier began.
    pub pending_publish_age: std::time::Duration,
    /// Total wall time of the `sync_all` call.
    pub total: std::time::Duration,
}

impl Default for SyncTimings {
    fn default() -> Self {
        Self {
            failed: false,
            pack_flush: std::time::Duration::ZERO,
            pack_fsync: std::time::Duration::ZERO,
            sidecar: std::time::Duration::ZERO,
            delta_log: std::time::Duration::ZERO,
            checkpoint: std::time::Duration::ZERO,
            wal: std::time::Duration::ZERO,
            journal_lock_wait: std::time::Duration::ZERO,
            journal_pending_wait: std::time::Duration::ZERO,
            journal_append: std::time::Duration::ZERO,
            journal_fsync: std::time::Duration::ZERO,
            journal_sync_calls: 0,
            journal_bytes: 0,
            journal_records: 0,
            journal_waiters: 0,
            journal_coalesced: 0,
            journal_in_flight: 0,
            dirty_lock_wait: std::time::Duration::ZERO,
            pending_publish_age: std::time::Duration::ZERO,
            total: std::time::Duration::ZERO,
        }
    }
}

/// Store-internal lifetime accumulator of per-phase sync wall time.
///
/// Each field is a running total in nanoseconds, added once per
/// `sync()`/`sync_all` call in `count_sync_persistence`. This is the
/// cumulative counterpart to the most-recent-only [`SyncTimings`] retained in
/// `last_sync_timings`: without it, a run's scatter-vs-rewrite split can only
/// be read off one (possibly atypical) barrier. [`Self::snapshot`] converts the
/// atomics to a plain [`SyncTotalsSnapshot`] for [`RuntimeStats`].
#[derive(Default)]
struct SyncTotals {
    /// Every sync that ran these accumulations (dirty or not).
    calls: AtomicU64,
    /// Sum of [`SyncTimings::total`].
    total_ns: AtomicU64,
    /// Sum of [`SyncTimings::pack_flush`].
    pack_flush_ns: AtomicU64,
    /// Sum of [`SyncTimings::pack_fsync`].
    pack_fsync_ns: AtomicU64,
    /// Sum of [`SyncTimings::sidecar`].
    sidecar_ns: AtomicU64,
    /// Sum of [`SyncTimings::delta_log`].
    delta_log_ns: AtomicU64,
    /// Sum of [`SyncTimings::checkpoint`].
    checkpoint_ns: AtomicU64,
    /// Sum of [`SyncTimings::wal`].
    wal_ns: AtomicU64,
    journal_lock_wait_ns: AtomicU64,
    journal_pending_wait_ns: AtomicU64,
    journal_append_ns: AtomicU64,
    journal_fsync_ns: AtomicU64,
    journal_sync_calls: AtomicU64,
    journal_bytes: AtomicU64,
    journal_records: AtomicU64,
    journal_waiters: AtomicU64,
    journal_coalesced: AtomicU64,
    max_journal_lock_wait_ns: AtomicU64,
    max_journal_fsync_ns: AtomicU64,
    dirty_lock_wait_ns: AtomicU64,
    pending_publish_age_ns: AtomicU64,
}

impl SyncTotals {
    fn add_duration(counter: &AtomicU64, duration: std::time::Duration) {
        counter.fetch_add(
            u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Fold one barrier's phase breakdown into the lifetime totals.
    fn accumulate(&self, timings: &SyncTimings) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&self.total_ns, timings.total);
        Self::add_duration(&self.pack_flush_ns, timings.pack_flush);
        Self::add_duration(&self.pack_fsync_ns, timings.pack_fsync);
        Self::add_duration(&self.sidecar_ns, timings.sidecar);
        Self::add_duration(&self.delta_log_ns, timings.delta_log);
        Self::add_duration(&self.checkpoint_ns, timings.checkpoint);
        Self::add_duration(&self.wal_ns, timings.wal);
        Self::add_duration(&self.journal_lock_wait_ns, timings.journal_lock_wait);
        Self::add_duration(&self.journal_pending_wait_ns, timings.journal_pending_wait);
        Self::add_duration(&self.journal_append_ns, timings.journal_append);
        Self::add_duration(&self.journal_fsync_ns, timings.journal_fsync);
        self.journal_sync_calls
            .fetch_add(timings.journal_sync_calls, Ordering::Relaxed);
        self.journal_bytes
            .fetch_add(timings.journal_bytes, Ordering::Relaxed);
        self.journal_records
            .fetch_add(timings.journal_records, Ordering::Relaxed);
        self.journal_waiters
            .fetch_add(timings.journal_waiters, Ordering::Relaxed);
        self.journal_coalesced
            .fetch_add(timings.journal_coalesced, Ordering::Relaxed);
        self.max_journal_lock_wait_ns.fetch_max(
            u64::try_from(timings.journal_lock_wait.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.max_journal_fsync_ns.fetch_max(
            u64::try_from(timings.journal_fsync.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Self::add_duration(&self.dirty_lock_wait_ns, timings.dirty_lock_wait);
        Self::add_duration(&self.pending_publish_age_ns, timings.pending_publish_age);
    }

    fn snapshot(&self) -> SyncTotalsSnapshot {
        let duration =
            |counter: &AtomicU64| std::time::Duration::from_nanos(counter.load(Ordering::Relaxed));
        SyncTotalsSnapshot {
            calls: self.calls.load(Ordering::Relaxed),
            total: duration(&self.total_ns),
            pack_flush: duration(&self.pack_flush_ns),
            pack_fsync: duration(&self.pack_fsync_ns),
            sidecar: duration(&self.sidecar_ns),
            delta_log: duration(&self.delta_log_ns),
            checkpoint: duration(&self.checkpoint_ns),
            wal: duration(&self.wal_ns),
            journal_lock_wait: duration(&self.journal_lock_wait_ns),
            journal_pending_wait: duration(&self.journal_pending_wait_ns),
            journal_append: duration(&self.journal_append_ns),
            journal_fsync: duration(&self.journal_fsync_ns),
            journal_sync_calls: self.journal_sync_calls.load(Ordering::Relaxed),
            journal_bytes: self.journal_bytes.load(Ordering::Relaxed),
            journal_records: self.journal_records.load(Ordering::Relaxed),
            journal_waiters: self.journal_waiters.load(Ordering::Relaxed),
            journal_coalesced: self.journal_coalesced.load(Ordering::Relaxed),
            max_journal_lock_wait: duration(&self.max_journal_lock_wait_ns),
            max_journal_fsync: duration(&self.max_journal_fsync_ns),
            dirty_lock_wait: duration(&self.dirty_lock_wait_ns),
            pending_publish_age: duration(&self.pending_publish_age_ns),
        }
    }

    fn reset(&self) {
        for counter in [
            &self.calls,
            &self.total_ns,
            &self.pack_flush_ns,
            &self.pack_fsync_ns,
            &self.sidecar_ns,
            &self.delta_log_ns,
            &self.checkpoint_ns,
            &self.wal_ns,
            &self.journal_lock_wait_ns,
            &self.journal_pending_wait_ns,
            &self.journal_append_ns,
            &self.journal_fsync_ns,
            &self.journal_sync_calls,
            &self.journal_bytes,
            &self.journal_records,
            &self.journal_waiters,
            &self.journal_coalesced,
            &self.max_journal_lock_wait_ns,
            &self.max_journal_fsync_ns,
            &self.dirty_lock_wait_ns,
            &self.pending_publish_age_ns,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

/// Plain, copyable lifetime totals of per-phase sync wall time — the cumulative
/// counterpart to [`SyncTimings`]. `calls` counts every sync that accumulated,
/// so phase averages are `phase / calls` and shares are `phase / total`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncTotalsSnapshot {
    /// Number of syncs folded into these totals.
    pub calls: u64,
    /// Cumulative [`SyncTimings::total`].
    pub total: std::time::Duration,
    /// Cumulative [`SyncTimings::pack_flush`].
    pub pack_flush: std::time::Duration,
    /// Cumulative [`SyncTimings::pack_fsync`].
    pub pack_fsync: std::time::Duration,
    /// Cumulative [`SyncTimings::sidecar`].
    pub sidecar: std::time::Duration,
    /// Cumulative [`SyncTimings::delta_log`].
    pub delta_log: std::time::Duration,
    /// Cumulative [`SyncTimings::checkpoint`].
    pub checkpoint: std::time::Duration,
    /// Cumulative [`SyncTimings::wal`].
    pub wal: std::time::Duration,
    /// Cumulative time waiting for the journal mutex.
    pub journal_lock_wait: std::time::Duration,
    /// Cumulative time waiting for the pending-mutation queue mutex.
    pub journal_pending_wait: std::time::Duration,
    /// Cumulative journal append/encoding time, excluding fsync.
    pub journal_append: std::time::Duration,
    /// Cumulative journal fsync time.
    pub journal_fsync: std::time::Duration,
    /// Number of journal sync calls.
    pub journal_sync_calls: u64,
    /// Cumulative journal bytes appended, including framing.
    pub journal_bytes: u64,
    /// Cumulative journal records appended.
    pub journal_records: u64,
    /// Number of journal sync calls that waited for the journal mutex.
    pub journal_waiters: u64,
    /// Number of journal sync calls covered by another sync.
    pub journal_coalesced: u64,
    /// Largest single journal mutex wait observed.
    pub max_journal_lock_wait: std::time::Duration,
    /// Largest single journal fsync observed.
    pub max_journal_fsync: std::time::Duration,
    /// Cumulative dirty-set lock wait attributed to sync barriers.
    pub dirty_lock_wait: std::time::Duration,
    /// Cumulative age observed for pending writes at sync entry.
    pub pending_publish_age: std::time::Duration,
}

/// Fixed latency buckets used by sync diagnostics: `<1ms`, `<10ms`,
/// `<100ms`, `<1s`, and `>=1s`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct SyncLatencyHistogram {
    pub buckets: [u64; 5],
}

impl SyncLatencyHistogram {
    fn observe(&mut self, duration: std::time::Duration) {
        let micros = duration.as_micros();
        let bucket = if micros < 1_000 {
            0
        } else if micros < 10_000 {
            1
        } else if micros < 100_000 {
            2
        } else if micros < 1_000_000 {
            3
        } else {
            4
        };
        self.buckets[bucket] = self.buckets[bucket].saturating_add(1);
    }
}

/// One of the worst sync operations retained for post-run diagnosis.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct SyncDiagnosticSample {
    /// Unix timestamp in milliseconds when the sync completed.
    pub timestamp_ms: u128,
    pub process_id: u32,
    pub journal_path: Option<String>,
    pub total: std::time::Duration,
    pub failed: bool,
    pub pack_flush: std::time::Duration,
    pub pack_fsync: std::time::Duration,
    pub sidecar: std::time::Duration,
    pub delta_log: std::time::Duration,
    pub checkpoint: std::time::Duration,
    pub dirty_lock_wait: std::time::Duration,
    pub pending_publish_age: std::time::Duration,
    pub wal: std::time::Duration,
    pub journal_lock_wait: std::time::Duration,
    pub journal_pending_wait: std::time::Duration,
    pub journal_append: std::time::Duration,
    pub journal_fsync: std::time::Duration,
    pub journal_records: u64,
    pub journal_bytes: u64,
    pub journal_in_flight: u64,
    pub journal_waiters: u64,
    pub journal_coalesced: u64,
}

/// Runtime sync diagnostics retained after the operation that produced them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct SyncDiagnosticsSnapshot {
    pub fsync_latency: SyncLatencyHistogram,
    pub lock_wait_latency: SyncLatencyHistogram,
    pub peak_journal_in_flight: u64,
    pub worst_syncs: Vec<SyncDiagnosticSample>,
}

#[derive(Default)]
struct SyncDiagnostics {
    fsync_latency: SyncLatencyHistogram,
    lock_wait_latency: SyncLatencyHistogram,
    peak_journal_in_flight: u64,
    worst_syncs: Vec<SyncDiagnosticSample>,
}

impl SyncDiagnostics {
    const WORST_LIMIT: usize = 16;

    fn record(&mut self, sample: SyncDiagnosticSample) {
        if sample.journal_path.is_some() {
            if sample.journal_coalesced == 0 && !sample.journal_fsync.is_zero() {
                self.fsync_latency.observe(sample.journal_fsync);
            }
            if sample.journal_coalesced == 0 {
                self.lock_wait_latency.observe(sample.journal_lock_wait);
            }
            self.peak_journal_in_flight = self.peak_journal_in_flight.max(sample.journal_in_flight);
        }
        self.worst_syncs.push(sample);
        self.worst_syncs
            .sort_unstable_by_key(|sample| std::cmp::Reverse(sample.total));
        self.worst_syncs.truncate(Self::WORST_LIMIT);
    }

    fn snapshot(&self) -> SyncDiagnosticsSnapshot {
        SyncDiagnosticsSnapshot {
            fsync_latency: self.fsync_latency,
            lock_wait_latency: self.lock_wait_latency,
            peak_journal_in_flight: self.peak_journal_in_flight,
            worst_syncs: self.worst_syncs.clone(),
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
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
type RepackCopiedRecord = ([u8; 16], u16, u64, u64);
/// Return type of [`PackfileStorage::repack_scan_incremental`]: the merged
/// live-location map and (on the incremental path) the previous cursor state
/// that the adjacency helpers need to skip re-reading already-seen nodes.
type RepackScanResult = (RepackRecordMap, Option<RepackIncrementalState>);

/// One scanned record's `(slot, hash, offset)`, as accumulated per
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
    collection_disk_bytes: HashMap<u64, HashMap<[u8; 16], u64>>,
    /// A checkpoint-backed writable open found no valid shard→collection
    /// sidecar, so the next sync must write one even if no index is dirty.
    sidecar_missing: bool,
    /// The physical scan needed to rebuild the sidecar's byte metrics failed.
    /// Do not allow a sidecar containing zero-byte fallbacks to be persisted.
    sidecar_recovery_failed: bool,
}

/// Per-pack, per-collection on-disk bytes as `physical_layout` reports them
/// (every appended frame, superseded ones included).
fn disk_bytes_from_physical(
    physical: &crate::packfile::layout::PhysicalLayout,
) -> HashMap<u64, HashMap<[u8; 16], u64>> {
    let mut bytes: HashMap<u64, HashMap<[u8; 16], u64>> = HashMap::new();
    for (collection_id, layout) in &physical.collections {
        for (pack_id, pack_bytes) in &layout.pack_bytes {
            bytes
                .entry(*pack_id)
                .or_default()
                .insert(*collection_id, *pack_bytes);
        }
    }
    bytes
}

/// Magic bytes + version identifying the persisted shard→collection directory
/// format (see `PackfileStorage::persist_shard_collections`).
///
/// Pre-release, no compatibility fallback: an unrecognized version is
/// treated exactly like a missing/corrupt file (see
/// `read_persisted_shard_collections`) — reset or let the next sync
/// regenerate it, not a format this reader tries to still understand.
/// This sidecar has changed shape three times already (adding insertion
/// order, switching `slot` for `pack_id`, then v4 adding the
/// pack-fingerprint gate); none of that carries forward, since nothing
/// depends on reading a store from before the current format existed.
const SHARD_ROOMS_MAGIC: &[u8; 4] = b"MSRM";
/// v5 adds physical bytes to the reduced bookkeeping and pins it to the exact
/// `(pack_id, file_len)` set by carrying the same `pack_fingerprint` as the
/// index checkpoint.
const SHARD_ROOMS_VERSION: u8 = 5;
/// Header size: magic(4) + version(1) + `pack_fingerprint(8)` + `persisted_at(8)`.
const SHARD_ROOMS_HEADER_LEN: usize = 4 + 1 + 8 + 8;
/// One entry: `pack_id`(8) + `collection_id`(16) + count(8) + the
/// collection's stable insertion ordinal(8).
const SHARD_ROOMS_RECORD_LEN: usize = 8 + 16 + 8 + 8 + 8;

/// The largest record offset `IndexEntry` can represent: its 32-bit offset
/// field stores `offset + 1`, reserving the all-zeros encoding for the empty
/// sentinel. Offsets beyond this can only come from legacy or externally
/// created oversized packs — this engine's own writes rotate long before
/// reaching it (`MAX_SHARD_BYTES` caps each shard's file size).
const PACK_INDEX_OFFSET_LIMIT: u64 = crate::index::IndexEntry::MAX_OFFSET;

/// Byte ceiling for the incremental index delta log. Once a session's
/// accumulated frames exceed this, the next `sync()` stops appending and does
/// a full checkpoint rewrite (which truncates the log), keeping replay cost
/// and the log file bounded. Frames are 36 bytes each, so this is on the order
/// of ~230k appends between full rewrites.
const DELTA_LOG_CAP_BYTES: u64 = 256 * 1024 * 1024;

/// Candidate offsets closer than this on the same shard count as one
/// sequential read run when measuring a `get_many` batch's locality.
/// `record_disk_len` is `frame_len + 8` and frames are at least
/// `FRAME_FIXED_LEN` bytes, so a gap this small means no more than one
/// minimum-size interleaved foreign frame sits between two candidate frames —
/// reading both is effectively a single sequential range. It is a shape
/// heuristic for the optimizer's "one read plan per shard" target, not exact
/// adjacency (that would require a per-candidate frame-length probe).
const READ_RUN_GAP_BYTES: u64 = 128;

/// Reject a record whose in-shard offset the 32-bit `IndexEntry` field cannot
/// represent, so the caller surfaces a `StorageError::Corrupt` instead of
/// silently dropping the record (or panicking in `IndexEntry::new`).
fn check_index_offset(slot: u16, hash: &[u8; 16], offset: u64) -> Result<(), StorageError> {
    if offset > PACK_INDEX_OFFSET_LIMIT {
        return Err(StorageError::Corrupt(format!(
            "shard {slot} holds record {hash:?} at offset {offset}, beyond the index offset limit {PACK_INDEX_OFFSET_LIMIT}"
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
    disk_bytes: u64,
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
            let disk_bytes = u64::from_le_bytes(chunk[40..48].try_into().ok()?);
            Some(PersistedShardRoom {
                pack_id,
                collection_id,
                count,
                insertion_order,
                disk_bytes,
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

struct PutManyProgress {
    generation: u64,
    owned_index: Option<LossyIndex>,
    structural_change: bool,
    index_needs_rebuild: bool,
    pending_deltas: Vec<(u32, u64)>,
    pending_shard_collections: Vec<(u64, u64)>,
    pending_shard_collection_counts: Vec<u64>,
    invalidate_delta: bool,
    undo_log: Vec<EntryUndo>,
}

/// Index tables that must advance together when a reader reloads a checkpoint.
/// Individual table guards are mapped from this shared lock, so a reload's
/// replacement is indivisible with respect to every table lookup.
struct IndexTables {
    collections: HashMap<[u8; 16], ArcSwap<RoomGeneration>>,
    collection_order: Vec<[u8; 16]>,
    shard_collections: HashMap<u64, HashMap<[u8; 16], u64>>,
    collection_shards: HashMap<[u8; 16], HashSet<u64>>,
    collection_disk_bytes: HashMap<u64, HashMap<[u8; 16], u64>>,
}

/// Session state for the incremental index delta log (`index.delta`).
///
/// A fresh checkpoint rewrite re-bases this structure. Between rewrites, live
/// index mutations are coalesced per collection as v3 slot updates, whole-index
/// snapshots, or deletions, then appended by the next `sync()`.
#[derive(Clone, Default)]
struct DeltaLogState {
    /// Fingerprint of the checkpoint this session continues. `None` when the
    /// store opened via a full rescan and hasn't rewritten a checkpoint yet —
    /// every such open forces the next dirty sync to a full rewrite.
    base_fingerprint: Option<u64>,
    /// Generation of each collection at the last checkpoint or snapshot.
    base_generations: HashMap<[u8; 16], u64>,
    /// Stable order keys preserved across deletes and snapshots.
    base_order: HashMap<[u8; 16], u64>,
    /// Monotonic order key allocated to the next newly created collection.
    next_order_key: u64,
    /// Per-collection operations pending for the next v3 append. Snapshot and
    /// deletion entries subsume earlier slot updates for their collection.
    pending: HashMap<[u8; 16], PendingDelta>,
    /// Bytes already committed to the on-disk log (upper bound that a fresh
    /// header may still need to be written). Resets to zero on rewrite.
    log_bytes: u64,
    /// Wire version of the active on-disk epoch. A v2 epoch is read for
    /// compatibility but is rebased to v3 before any further writes.
    log_version: u8,
}

#[derive(Clone, PartialEq, Eq)]
enum PendingDelta {
    Slots(Vec<DeltaFrame>),
    Snapshot,
    Delete { generation: u64 },
}

struct V3PendingBatch {
    operations: Vec<DeltaOperation>,
    generation_updates: Vec<([u8; 16], u64)>,
    order_updates: Vec<([u8; 16], u64)>,
    next_order_key: u64,
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
    /// Collection indexes and their cross-table bookkeeping, replaced as one
    /// snapshot when a read-only worker reloads a checkpoint.
    index_tables: RwLock<IndexTables>,
    pinned: PinnedNodes,
    base_dir: PathBuf,
    swizzle: Option<SwizzleFn>,
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    /// Per-collection gates for read-miss refreshes. A reader that observes
    /// an index miss waits for an in-flight refresh of the same collection
    /// instead of starting another full scan.
    refresh_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    /// Pack fingerprint observed at the last read-miss refresh per collection.
    /// A subsequent miss with the same fingerprint skips the expensive rebuild.
    last_refresh_fingerprint: parking_lot::Mutex<HashMap<[u8; 16], u64>>,
    /// Durable pool fingerprint observed when this handle was opened. This is
    /// the baseline for collections that did not yet exist at open time.
    initial_durable_fingerprint: u64,
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
    /// Per-instance index config: seed from the pool's `pool.meta`, floor
    /// and load factor from defaults (or future tuning).
    index_config: crate::index::IndexConfig,
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
    /// the deduplicated hash → (`slot`, offset) map from the last repack.
    /// On the next repack, only bytes after these offsets are scanned and
    /// merged into the existing map — turning O(n²) full rescan into
    /// O(n) total work across all repack calls.
    repack_incremental: RwLock<HashMap<[u8; 16], RepackIncrementalState>>,
    /// Persisted directory: which collections have live records in each shard,
    /// and how many. Maintained incrementally (a plain `put` just
    /// increments one counter; a full index rebuild/repack/initial scan
    /// replaces one collection's contribution wholesale via
    /// `LossyIndex::slot_counts`) rather than ever re-derived by
    /// scanning a shard file, which is what made `collections_referencing_shard`
    /// and a `collections`-style listing expensive before this existed.
    /// Reverse index of `shard_collections`: which shards a given collection currently
    /// contributes a nonzero count to. Lets a collection's full-index-rebuild
    /// path (`replace_collection_shard_counts`) clear exactly the shard entries
    /// it used to occupy without scanning every shard in `shard_collections`.
    /// Wall-clock instant of the last `maybe_persist_shard_collections` flush,
    /// used to rate-limit that timer-driven path.
    last_shard_collections_flush: RwLock<Option<std::time::Instant>>,
    /// Wall-clock breakdown of the most recent `open` — which index-loading
    /// path was taken and how long each phase took — set at construction.
    last_open_timings: parking_lot::Mutex<Option<OpenTimings>>,
    /// Wall-clock breakdown of the most recent `sync_all`, by phase.
    last_sync_timings: parking_lot::Mutex<Option<SyncTimings>>,
    /// Lifetime per-phase sync totals, accumulated once per sync in
    /// `count_sync_persistence`, so the scatter-vs-rewrite split can be read
    /// across a whole run instead of only the most recent barrier (which
    /// `last_sync_timings` alone cannot answer).
    sync_totals: SyncTotals,
    /// Rolling sync diagnostics retained for post-run inspection. This is
    /// deliberately bounded and never participates in durability decisions.
    sync_diagnostics: parking_lot::Mutex<SyncDiagnostics>,
    /// Number and time spent publishing mutations to the optional journal.
    publish_calls: AtomicU64,
    publish_time_ns: AtomicU64,
    pending_publish_since: parking_lot::Mutex<Option<std::time::Instant>>,
    publish_generation: AtomicU64,
    /// Minimum wall-clock interval between full checkpoint rewrites needed
    /// because no usable delta base exists or a v3 append failed. Zero disables this half of the
    /// rewrite budget. See [`Self::set_checkpoint_rewrite_budget`].
    checkpoint_rewrite_min_interval_ns: AtomicU64,
    /// Maximum `put_bytes + put_many_bytes` accumulated since the last full
    /// checkpoint rewrite before one is forced even inside the time interval.
    /// Zero disables this half of the budget.
    checkpoint_rewrite_max_bytes: AtomicU64,
    /// Instant of the last full checkpoint rewrite this session — the time
    /// half of the rewrite budget.
    last_checkpoint_rewrite_at: parking_lot::Mutex<Option<std::time::Instant>>,
    /// `put_bytes + put_many_bytes` sampled at the last full checkpoint
    /// rewrite — the size half of the rewrite budget.
    checkpoint_bytes_at_last_rewrite: AtomicU64,
    /// Syncs that skipped a structurally-needed full checkpoint rewrite
    /// because the configured time/size budget still had headroom. The
    /// on-disk checkpoint stays stale by design; the next open then falls
    /// back to a rescan because the pack fingerprint advanced past it.
    checkpoint_skips: AtomicU64,
    /// Gates the logical read-path counters (`get`/`get_many` below). Off by
    /// default so a store doing no reads-of-record pays one relaxed load per
    /// logical read at most, and the write/batch/sync counters are the only
    /// always-on instrumentation (each is a single batch-granular `fetch_add`
    /// on a write-locked or per-call path, never per-record on the hot read
    /// path). See [`Self::set_stats_enabled`].
    stats_enabled: AtomicBool,
    /// Whether [`Self::get_many_with_refresh`] may rescan a collection when
    /// the caller's in-memory snapshot misses. On by default. A single-writer
    /// store sets this `false`: its in-memory index is authoritative for every
    /// key it has written -- the lossy index only yields false-positive
    /// candidate collisions, never false negatives -- so a negative lookup is
    /// a true miss and refreshing can only spend a durable-fingerprint probe
    /// -- and, after each checkpoint, a full rescan -- to rediscover nothing.
    /// Multi-process readers leave it on to observe records the writer process
    /// appends.
    refresh_on_miss: AtomicBool,
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
    /// Read-miss refreshes that actually rescanned a collection.
    miss_refreshes: AtomicU64,
    /// Read-miss refreshes skipped because another refresh was recent.
    miss_refresh_skips: AtomicU64,
    /// Missing keys recovered by a read-miss refresh.
    miss_refresh_recovered: AtomicU64,
    /// IDs submitted in the retry request after a refresh. Proves that only
    /// missing keys are retried: `miss_refresh_retry_ids == count(missing)`.
    miss_refresh_retry_ids: AtomicU64,
    /// Fingerprint read errors after refresh — used to rate-limit warnings
    /// and expose diagnostics.
    miss_refresh_fp_errors: AtomicU64,
    /// Candidate locations yielded by the lossy index.
    index_candidates: AtomicU64,
    /// Candidate locations actually read from packfiles.
    candidate_reads: AtomicU64,
    /// Candidate records rejected after full-hash verification.
    candidate_hash_mismatches: AtomicU64,
    /// Unique shards touched across `get_many` batches.
    get_many_shards_touched: AtomicU64,
    /// On-disk frame bytes touched by candidate-record reads (a prefix read
    /// per frame; gated on stats, and only covers the candidate-resolve path,
    /// not refresh rescan). This sums on-disk *frame lengths* — a logical
    /// extent estimate: an mmap read can pull in whole pages, readahead
    /// ranges, and merged extents, so this must not be read as a physical
    /// disk-byte count. The honest `disk-extent` complement to
    /// `candidate_reads`: `candidate_frame_bytes / candidate_reads` is the
    /// average compressed frame the cold lookups drag in.
    candidate_frame_bytes: AtomicU64,
    /// Offset runs a `get_many` batch collapses its candidate reads into.
    /// Offsets within [`READ_RUN_GAP_BYTES`] of the previous candidate on the
    /// same shard count as one sequential run; the frontier batch's "one read
    /// plan per shard" target shows up here as runs ≈ shards, not ≈
    /// candidates.
    read_many_runs: AtomicU64,
    /// Plan fan-in of a `get_many` batch: per shard, `last_offset −
    /// first_offset` summed across touched shards (byte-extent the batch
    /// scatters over, even if it only touches sparse frames). Drives the
    /// scattered-read signal independent of candidate count.
    read_many_span_bytes: AtomicU64,

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
    /// Index grow/rebuild events in `put_many` (each requests a v3 snapshot).
    index_grow_count: AtomicU64,
    /// Fallback `rebuild_index` calls (full pack scan).
    index_rebuild_count: AtomicU64,
    /// `invalidate_delta_log` calls — structural changes represented by a
    /// whole-collection snapshot in the v3 log.
    delta_invalidations: AtomicU64,
    /// `sync`/`sync_all` calls.
    sync_calls: AtomicU64,
    /// Syncs that rewrote the index checkpoint in full.
    checkpoint_writes: AtomicU64,
    /// Syncs that appended the incremental delta log instead.
    delta_appends: AtomicU64,
    /// Writes of the shard→collection inspection sidecar
    /// (`shard_collections.bin`). Counts every temp-write+rename, whichever
    /// caller triggered it.
    sidecar_writes: AtomicU64,
    /// Whether any collection data changed since the last `index.checkpoint`
    /// write. Set by every generation swap (`put`/`put_many`/`repack`/`refresh`);
    /// cleared only by a successful [`Self::persist_index_checkpoint`], so a
    /// failed write is retried on the next sync. Lets `sync()`/`sync_all()`
    /// skip rewriting the checkpoint when nothing has changed since the last
    /// one — a steady-state writer that syncs between writes pays no checkpoint
    /// cost, while a crash left a stale checkpoint is still always resolved by
    /// the fingerprint → rescan fallback.
    index_checkpoint_dirty: AtomicBool,
    /// The shard→collection sidecar needs writing even though no index is
    /// dirty (see `RoomScanOutput::sidecar_missing`).
    shard_collections_dirty: AtomicBool,
    /// A missing sidecar was observed, but its physical byte metrics could not
    /// be rebuilt. Keep retrying rather than publishing misleading zeros.
    shard_collections_recovery_failed: AtomicBool,
    /// A sidecar persist failure has already been logged; cleared on success so
    /// a persistent failure logs once per streak, not once per sync.
    shard_collections_failure_logged: AtomicBool,
    /// Test-only: runs after the recovery scan and before its totals are
    /// swapped in, so a test can hold recovery inside that window.
    #[cfg(test)]
    recovery_pause_hook: parking_lot::Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    /// Session state for the incremental index delta log: the base checkpoint
    /// fingerprint + generations the log continues, and the frames accumulated
    /// since the last persist. See [`DeltaLogState`].
    delta_state: parking_lot::Mutex<DeltaLogState>,
    /// Serializes checkpoint/delta persistence calls while allowing the
    /// delta-state mutex to be released during whole-index snapshot encoding.
    index_persist_lock: parking_lot::Mutex<()>,
    /// Optional write-ahead journal coordinator. When present, every mutation
    /// is published to it and `sync` commits a single durable group, so the
    /// per-shard pack fsyncs and the index checkpoint can be deferred without
    /// risking acknowledged data (see [`Self::enable_journal`]).
    journal: parking_lot::Mutex<Option<Arc<JournalCoordinator>>>,
    /// Committed groups recovered when the journal was opened, retained for
    /// [`Self::replay_journal`] to re-apply after a reopen.
    journal_recovery: parking_lot::Mutex<Vec<crate::journal::CommittedGroup>>,
    /// Set while [`Self::replay_journal`] re-applies recovered mutations, so
    /// those writes are not re-published to the journal.
    replaying: AtomicBool,
    /// Optional read-only journal overlay backing [`Self::get_read_committed`].
    /// Enabled on a read-only store so a worker can observe committed-but-
    /// unflushed journal groups that the durable fingerprint gate deliberately
    /// hides. See [`Self::enable_read_journal`].
    #[cfg(feature = "multi-reader")]
    read_journal: parking_lot::Mutex<Option<ReadJournal>>,
    /// Journal LSN covered by the durable index this handle actually loaded.
    ///
    /// Read once at open, when the index is built from the on-disk checkpoint
    /// (or a full scan). It is the *only* coverage the overlay may prune
    /// through: a fresher `journal.lsn` written by a concurrent checkpoint does
    /// not mean this handle's in-memory index contains those records. Advancing
    /// this pair requires loading the corresponding checkpoint, not a live
    /// packfile rescan. See [`Self::get_read_committed`].
    // Only read back by the (feature-gated) read-committed overlay; a
    // non-`multi-reader` build still writes it once at open so the two
    // build configurations share one open path, but never reads it back.
    #[cfg_attr(not(feature = "multi-reader"), allow(dead_code))]
    read_covered_lsn: AtomicU64,
    /// Successful checkpoint-bound index reloads triggered by the
    /// read-committed overlay, because a writer's reclaim outran the coverage
    /// this handle's index incorporated. A WAL-cell failure on the reload path
    /// is visible here instead of looking like an ordinary miss. See
    /// [`Self::refresh_read_journal`].
    read_reloads: AtomicU64,
    /// Reload attempts that could not load a checkpoint matching the current
    /// packs, so the read-committed overlay failed closed.
    read_reload_failures: AtomicU64,
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
    /// The deduplicated hash → (`slot`, offset) map from the last repack.
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
    fn refresh_lock(&self, collection_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.refresh_locks.lock();
        locks
            .entry(*collection_id)
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone()
    }

    fn post_refresh_fingerprint(&self) -> Option<u64> {
        match crate::index::checkpoint::read_durable_fingerprint(&self.base_dir) {
            Ok(Some(fp)) => Some(fp.fingerprint),
            Ok(None) => Some(0),
            Err(error) => {
                let previous = self.miss_refresh_fp_errors.fetch_add(1, Ordering::Relaxed);
                let count = previous.saturating_add(1);
                let periodic = count
                    .checked_rem(100)
                    .is_some_and(|remainder| remainder == 0);
                if previous == 0 || periodic {
                    eprintln!(
                        "warning: unable to observe post-refresh fingerprint (count={count}): {error}"
                    );
                }
                None
            }
        }
    }

    fn refresh_and_retry(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
        missing: Vec<usize>,
        results: &mut [Option<NodeData>],
    ) -> Result<(), StorageError> {
        self.refresh_collection(collection_id)?;
        // Reread the fingerprint after refresh to handle a concurrent sync.
        if let Some(fp) = self.post_refresh_fingerprint() {
            self.last_refresh_fingerprint
                .lock()
                .insert(*collection_id, fp);
        }
        self.miss_refreshes.fetch_add(1, Ordering::Relaxed);

        let retry_ids: Vec<NodeId> = missing.iter().map(|&index| ids[index]).collect();
        let retry_count = u64::try_from(retry_ids.len()).unwrap_or(u64::MAX);
        self.miss_refresh_retry_ids
            .fetch_add(retry_count, Ordering::Relaxed);
        let retry = self.get_many(collection_id, &retry_ids)?;
        let mut recovered = 0u64;
        for (index, value) in missing.into_iter().zip(retry) {
            if value.is_some() {
                recovered = recovered.saturating_add(1);
                results[index] = value;
            }
        }
        self.miss_refresh_recovered
            .fetch_add(recovered, Ordering::Relaxed);
        Ok(())
    }

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

    /// Open a read-only observer with the read-committed journal overlay
    /// enabled, in one call.
    #[cfg(feature = "multi-reader")]
    ///
    /// This is the intended constructor for an authoritative read-only worker
    /// handle. The durable read APIs ([`StorageEngine::get_many`],
    /// [`Self::get_many_with_refresh`]) deliberately hide a writer's
    /// unflushed and not-yet-checkpointed mutations, so a worker that must
    /// observe another process's committed writes has to read through
    /// [`Self::get_read_committed`] — and that only consults an overlay once
    /// [`Self::enable_read_journal`] has attached a writer's segment. Pairing
    /// the two here makes this the only supported way to obtain such a handle,
    /// so a caller cannot open one that silently falls back to the durable
    /// API.
    ///
    /// `wal_path` is the writer's journal segment. See
    /// [`Self::enable_read_journal`] for the read-only safety contract and
    /// [`Self::open_read_only`] for the writer-coexistence contract.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the store cannot be opened read-only (same
    /// as [`Self::open_read_only`]), or if the segment is unreadable, a
    /// committed group fails validation, or a coverage gap cannot be resolved
    /// by reloading the checkpoint (same as [`Self::enable_read_journal`]).
    pub fn open_read_committed(
        base_dir: PathBuf,
        wal_path: impl AsRef<Path>,
    ) -> Result<Self, StorageError> {
        let store = Self::open_read_only(base_dir)?;
        store.enable_read_journal(wal_path)?;
        Ok(store)
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
        let shard_open_time = shard_open_started.elapsed();
        timings.shard_open = shard_open_time;
        let index_config = crate::index::IndexConfig {
            seed: shards.bucket_seed(),
            ..Default::default()
        };
        if let Some(shard_timings) = shards.open_timings() {
            timings.shard_open = shard_timings.total;
            timings.shard_discovery = shard_timings.discovery;
            timings.writer_lock = shard_timings.writer_lock;
            timings.packfile_recovery = shard_timings.packfile_recovery;
            timings.packfile_recovery_calls = shard_timings.packfile_recovery_calls;
            timings.packfile_open = shard_timings.packfile_open;
            timings.packfile_open_calls = shard_timings.packfile_open_calls;
            timings.metadata_restore = shard_timings.metadata_restore;
            timings.pool_meta_restore = shard_timings.pool_meta_restore;
            timings.persisted_stats_restore = shard_timings.persisted_stats_restore;
            timings.store_meta_write = shard_timings.store_meta_write;
            timings.pool_meta_persist = shard_timings.pool_meta_persist;
            timings.initial_pack_create = shard_timings.initial_pack_create;
            timings.metadata_unattributed = shard_timings.metadata_unattributed;
            timings.shard_open_unattributed = shard_timings.unattributed;
        }

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
        // Capture the journal coverage *before* loading the index, so the bound
        // can never be newer than the index this handle builds. A writer writes
        // `journal.lsn` only after its covering checkpoint, so reading it first
        // always yields a bound <= the coverage of the checkpoint/scan below.
        let read_covered = Self::read_journal_lsn(&base_dir);
        timings.metadata_load = metadata_started.elapsed();
        if let Some((scan_out, collection_order, delta_state, checkpoint_covered)) =
            Self::checkpoint_scan_out(
                &base_dir,
                cache_capacity,
                &shards,
                &open_shards,
                &deleted_collections,
                writable,
                ReloadMode::Strict,
                index_config,
                &mut timings,
            )
        {
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
                checkpoint_covered,
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
        for (slot, shard_pack_id, path, _file_len) in open_shards {
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
                // Read-only mode must not truncate a possibly in-flight tail,
                // but it also cannot safely omit a shard.  Publishing a
                // partial index after an I/O or corruption failure makes real
                // records look deleted.  Surface the error and let the caller
                // retry after the writer has finished or repair the pack.
                packfile::scan_packfile(&path).map_err(|error| {
                    std::io::Error::new(
                        error.kind(),
                        format!(
                            "read-only scan failed for shard {slot:02x} ({}): {error}",
                            path.display()
                        ),
                    )
                })?
            };
            for (collection_id, hash, offset) in entries {
                if known_collections.insert(collection_id) {
                    collection_order.push(collection_id);
                }
                collection_entries.entry(collection_id).or_default().push((
                    slot,
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
                index_config,
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
            DeltaLogState::default(),
            read_covered,
        );
        // A writer that had to rescan has no checkpoint describing this pack
        // set. Mark it dirty so the next sync persists one; otherwise a store
        // opened and synced without further writes (e.g. after an aborted
        // import) would rescan every packfile on every open forever.
        if writable {
            store.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }
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
        index_config: crate::index::IndexConfig,
        out: &mut RoomScanOutput,
    ) -> Result<(), StorageError> {
        // Seed each collection's home shard from the scan: the shard of its
        // last-scanned record is a best-effort proxy for "most recent"
        // (shards are scanned in ascending ID order, and IDs generally
        // increase over time via rotation) — not exact chronology across
        // shards, but enough to keep a resumed collection's writes landing near
        // its existing data instead of restarting at whatever the pool's
        // active shard happens to be.
        if let Some(&(last_slot, _, _, _)) = records.last() {
            shards.set_collection_home(collection_id, last_slot);
        }
        let index = LossyIndex::with_config(
            records
                .len()
                .saturating_mul(2)
                .max(NEW_COLLECTION_INDEX_FLOOR),
            index_config,
        );
        for (slot, hash, offset, _pack_id) in records {
            check_index_offset(*slot, hash, *offset)?;
            let _ = index.insert(hash, *slot, *offset);
        }
        for (slot, _hash, offset, pack_id) in records {
            let shard = shards
                .get_shard(*slot)
                .ok_or_else(|| StorageError::Corrupt(format!("missing shard slot {slot}")))?;
            let bytes = ShardPool::record_disk_len_at(&shard, *offset)?;
            let total = out
                .collection_disk_bytes
                .entry(*pack_id)
                .or_default()
                .entry(*collection_id)
                .or_default();
            *total = total.saturating_add(bytes);
        }
        let counts = index.slot_counts();

        // Build slot→pack_id lookup from the records for this collection.
        let slot_to_pack_id: HashMap<u16, u64> = records
            .iter()
            .map(|(slot, _, _, pack_id)| (*slot, *pack_id))
            .collect();

        for (&slot, &count) in &counts {
            let pack_id = slot_to_pack_id
                .get(&slot)
                .copied()
                .unwrap_or(u64::from(slot));
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
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "one field-for-field constructor shared by both open paths"
    )]
    fn assemble(
        shards: ShardPool,
        scan_out: RoomScanOutput,
        collection_order: Vec<[u8; 16]>,
        deleted_collections: HashSet<[u8; 16]>,
        base_dir: PathBuf,
        swizzle: Option<SwizzleFn>,
        cache_capacity: usize,
        delta_state: DeltaLogState,
        read_covered: u64,
    ) -> Self {
        let index_config = crate::index::IndexConfig {
            seed: shards.bucket_seed(),
            ..Default::default()
        };
        // Seed the per-collection refresh fingerprint from the persisted
        // checkpoint/delta state so the first miss for each collection can
        // skip refresh when nothing durable changed.
        let initial_durable_fp = match crate::index::checkpoint::read_durable_fingerprint(&base_dir)
        {
            Ok(Some(dfp)) => dfp.fingerprint,
            Ok(None) | Err(_) => 0,
        };
        let initial_fingerprints: HashMap<[u8; 16], u64> = collection_order
            .iter()
            .map(|&cid| (cid, initial_durable_fp))
            .collect();
        // The checkpoint coverage this open's index corresponds to, captured
        // before the index was loaded. Bound to the loaded index, never
        // re-read from disk by the overlay.
        let read_covered_lsn = AtomicU64::new(read_covered);
        Self {
            shards,
            index_tables: RwLock::new(IndexTables {
                collections: scan_out.collections,
                collection_order,
                shard_collections: scan_out.shard_collections,
                collection_shards: scan_out.collection_shards,
                collection_disk_bytes: scan_out.collection_disk_bytes,
            }),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            refresh_locks: parking_lot::Mutex::new(HashMap::new()),
            last_refresh_fingerprint: parking_lot::Mutex::new(initial_fingerprints),
            initial_durable_fingerprint: initial_durable_fp,
            collection_creation: parking_lot::RwLock::new(()),
            deleted_collections: parking_lot::Mutex::new(deleted_collections),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
            index_config,
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_collection: RwLock::new(HashMap::new()),
            repack_incremental: RwLock::new(HashMap::new()),
            last_shard_collections_flush: RwLock::new(None),
            index_checkpoint_dirty: AtomicBool::new(false),
            shard_collections_dirty: AtomicBool::new(scan_out.sidecar_missing),
            shard_collections_recovery_failed: AtomicBool::new(scan_out.sidecar_recovery_failed),
            shard_collections_failure_logged: AtomicBool::new(false),
            #[cfg(test)]
            recovery_pause_hook: parking_lot::Mutex::new(None),
            journal: parking_lot::Mutex::new(None),
            journal_recovery: parking_lot::Mutex::new(Vec::new()),
            replaying: AtomicBool::new(false),
            #[cfg(feature = "multi-reader")]
            read_journal: parking_lot::Mutex::new(None),
            read_covered_lsn,
            read_reloads: AtomicU64::new(0),
            read_reload_failures: AtomicU64::new(0),
            last_open_timings: parking_lot::Mutex::new(None),
            last_sync_timings: parking_lot::Mutex::new(None),
            sync_totals: SyncTotals::default(),
            sync_diagnostics: parking_lot::Mutex::new(SyncDiagnostics::default()),
            publish_calls: AtomicU64::new(0),
            publish_time_ns: AtomicU64::new(0),
            pending_publish_since: parking_lot::Mutex::new(None),
            publish_generation: AtomicU64::new(0),
            checkpoint_rewrite_min_interval_ns: AtomicU64::new(0),
            checkpoint_rewrite_max_bytes: AtomicU64::new(0),
            last_checkpoint_rewrite_at: parking_lot::Mutex::new(None),
            checkpoint_bytes_at_last_rewrite: AtomicU64::new(0),
            checkpoint_skips: AtomicU64::new(0),
            stats_enabled: AtomicBool::new(false),
            refresh_on_miss: AtomicBool::new(true),
            open_count: AtomicU64::new(1),
            get_calls: AtomicU64::new(0),
            get_misses: AtomicU64::new(0),
            get_many_calls: AtomicU64::new(0),
            get_many_records: AtomicU64::new(0),
            get_many_misses: AtomicU64::new(0),
            miss_refreshes: AtomicU64::new(0),
            miss_refresh_skips: AtomicU64::new(0),
            miss_refresh_recovered: AtomicU64::new(0),
            miss_refresh_retry_ids: AtomicU64::new(0),
            miss_refresh_fp_errors: AtomicU64::new(0),
            index_candidates: AtomicU64::new(0),
            candidate_reads: AtomicU64::new(0),
            candidate_hash_mismatches: AtomicU64::new(0),
            get_many_shards_touched: AtomicU64::new(0),
            candidate_frame_bytes: AtomicU64::new(0),
            read_many_runs: AtomicU64::new(0),
            read_many_span_bytes: AtomicU64::new(0),
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
            sidecar_writes: AtomicU64::new(0),
            delta_state: parking_lot::Mutex::new(delta_state),
            index_persist_lock: parking_lot::Mutex::new(()),
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

    /// Path of the sidecar recording the journal LSN the on-disk index
    /// checkpoint covers. Written only when a journal is enabled; its contents
    /// bound [`Self::replay_journal`] to the post-checkpoint suffix.
    fn journal_lsn_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("journal.lsn")
    }

    /// Durably record the journal LSN the just-written checkpoint covers.
    fn write_journal_lsn(&self, lsn: u64) -> Result<(), StorageError> {
        let path = Self::journal_lsn_path(&self.base_dir);
        let tmp = path.with_extension("lsn.tmp");
        fs::write(&tmp, lsn.to_le_bytes()).map_err(StorageError::Io)?;
        fs::File::open(&tmp)
            .and_then(|file| file.sync_all())
            .map_err(StorageError::Io)?;
        fs::rename(&tmp, &path).map_err(StorageError::Io)?;
        let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        Ok(())
    }

    /// The journal LSN the on-disk checkpoint covers (0 when none is recorded).
    fn read_journal_lsn(base_dir: &std::path::Path) -> u64 {
        fs::read(Self::journal_lsn_path(base_dir))
            .ok()
            .and_then(|bytes| bytes.get(..8).map(<[u8; 8]>::try_from))
            .and_then(Result::ok)
            .map_or(0, u64::from_le_bytes)
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
    /// checkpoint generation. A malformed or semantically inconsistent log
    /// makes this function return `None`, selecting the full rescan path.
    ///
    /// `reload_mode` selects the caller's stance (see [`ReloadMode`]).
    /// `ReloadMode::JournalBound` is for the read-committed overlay's reload
    /// (see `reload_index_from_checkpoint`): the caller is a read-only worker
    /// advancing its index/coverage pair to a writer's checkpoint, not a
    /// session opening the store. In that mode the exact-pack fingerprint gate
    /// and the delta-log replay are skipped. A writer that is still appending
    /// has already grown the live packs past the checkpoint, and the delta log
    /// may not yet cover that growth, so the gate would reject the checkpoint
    /// and fail the read closed — the failure mode that otherwise looks like a
    /// persistent miss. The overlay is authoritative for the post-checkpoint
    /// suffix, so the durable index only needs the checkpoint's base, and its
    /// coverage binds to exactly that base. The checkpoint itself must still be
    /// valid; only the "packs match exactly / log bridges to live" checks are
    /// dropped.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn checkpoint_scan_out(
        base_dir: &std::path::Path,
        cache_capacity: usize,
        shards: &ShardPool,
        open_shards: &[(u16, u64, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
        writable: bool,
        reload_mode: ReloadMode,
        index_config: crate::index::IndexConfig,
        timings: &mut OpenTimings,
    ) -> Option<(RoomScanOutput, Vec<[u8; 16]>, DeltaLogState, u64)> {
        let decode_started = std::time::Instant::now();
        let checkpoint =
            crate::index::checkpoint::read_checkpoint(&Self::index_checkpoint_path(base_dir));
        timings.checkpoint_decode = decode_started.elapsed();
        let checkpoint = checkpoint?;
        let covered_lsn = checkpoint.covered_lsn;
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
        // A read-journal reload deliberately skips the exact-pack gate and the
        // delta replay (see this function's doc): the overlay supplies the
        // committed suffix, and requiring the live pack set to match would fail
        // the read closed while a writer is still appending.
        let replay_needed =
            reload_mode == ReloadMode::Strict && local_fingerprint != checkpoint.fingerprint;
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
        let mut replay_operations = Vec::new();
        // A checkpoint with no surviving delta log is a valid base for the
        // current v3 writer. Only an actual v2 log needs a one-time rebase.
        let mut replay_log_version = 3;
        if replay_needed {
            let delta_started = std::time::Instant::now();
            if let Some(log) = delta::read_delta_log_v3(&delta_path) {
                if log.base_fingerprint != checkpoint.fingerprint
                    || log.tail_fingerprint != local_fingerprint
                {
                    timings.delta_replay = delta_started.elapsed();
                    if writable {
                        let _ = std::fs::remove_file(&delta_path);
                    }
                    return None;
                }
                log_bytes_on_disk = log.file_len;
                replay_operations = log.operations;
                replay_log_version = 3;
            } else {
                let log = delta::read_delta_log(&delta_path);
                let trusted = log.as_ref().is_some_and(|log| {
                    log.base_fingerprint == checkpoint.fingerprint
                        && log.tail_fingerprint == local_fingerprint
                        && log.frames.iter().all(|frame| {
                            !deleted_collections.contains(&frame.collection_id)
                                && ckpt_generations.get(&frame.collection_id)
                                    == Some(&frame.generation)
                        })
                });
                if trusted {
                    let trusted_log = log.expect("trusted implies a decoded v2 log");
                    log_bytes_on_disk = trusted_log.file_len;
                    replay_frames = trusted_log.frames;
                    replay_log_version = 2;
                } else {
                    timings.delta_replay = delta_started.elapsed();
                    if writable {
                        let _ = std::fs::remove_file(&delta_path);
                    }
                    return None;
                }
            }
            timings.delta_replay = delta_started.elapsed();
            // Reachable only after a log validated against both fingerprints,
            // so a nonzero count means operations were actually replayed.
            timings.delta_replay_operations = u64::try_from(replay_operations.len())
                .unwrap_or(u64::MAX)
                .saturating_add(u64::try_from(replay_frames.len()).unwrap_or(u64::MAX));
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

        let mut live_generations: HashMap<[u8; 16], u64> = ckpt_generations
            .iter()
            .filter(|(collection_id, _)| !deleted_collections.contains(*collection_id))
            .map(|(collection_id, generation)| (*collection_id, *generation))
            .collect();
        let mut live_order: HashMap<[u8; 16], u64> = checkpoint
            .collections
            .iter()
            .filter(|loaded| !deleted_collections.contains(&loaded.collection_id))
            .enumerate()
            .map(|(order, loaded)| {
                (
                    loaded.collection_id,
                    u64::try_from(order).unwrap_or(u64::MAX),
                )
            })
            .collect();
        let mut snapshot_indexes: HashMap<[u8; 16], (u64, LossyIndex)> = HashMap::new();
        let mut v3_tombstones = HashSet::new();
        for operation in replay_operations {
            match operation {
                DeltaOperation::Incremental(frame) => {
                    if deleted_collections.contains(&frame.collection_id) {
                        continue;
                    }
                    if live_generations.get(&frame.collection_id) != Some(&frame.generation) {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    }
                    if let Some((_, index)) = snapshot_indexes.get(&frame.collection_id) {
                        if index.replay_frames(std::slice::from_ref(&frame)).is_err() {
                            if writable {
                                let _ = std::fs::remove_file(&delta_path);
                            }
                            return None;
                        }
                    } else {
                        frames_by_collection
                            .entry(frame.collection_id)
                            .or_default()
                            .push(frame);
                    }
                }
                DeltaOperation::CollectionSnapshot {
                    collection_id,
                    generation,
                    order_key,
                    index_blob,
                } => {
                    if deleted_collections.contains(&collection_id) {
                        continue;
                    }
                    let Some(capacity_bytes) = index_blob.get(..8) else {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    };
                    let capacity =
                        usize::try_from(u64::from_le_bytes(capacity_bytes.try_into().ok()?))
                            .ok()?;
                    let expected_blob_len = capacity.checked_mul(24)?.checked_add(8)?;
                    if expected_blob_len != index_blob.len() {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    }
                    let Ok(index) = LossyIndex::deserialize_with_config(&index_blob, index_config)
                    else {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    };
                    frames_by_collection.remove(&collection_id);
                    snapshot_indexes.insert(collection_id, (generation, index));
                    live_generations.insert(collection_id, generation);
                    live_order.insert(collection_id, order_key);
                    v3_tombstones.remove(&collection_id);
                }
                DeltaOperation::CollectionTombstone {
                    collection_id,
                    generation,
                } => {
                    if deleted_collections.contains(&collection_id) {
                        continue;
                    }
                    if live_generations.get(&collection_id) != Some(&generation) {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    }
                    live_generations.remove(&collection_id);
                    live_order.remove(&collection_id);
                    frames_by_collection.remove(&collection_id);
                    snapshot_indexes.remove(&collection_id);
                    v3_tombstones.insert(collection_id);
                }
            }
        }

        let slot_to_pack_id: HashMap<u16, u64> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*slot, *pack_id))
            .collect();

        // Translate every slot this checkpoint's index entries (and any
        // delta frame continuing it) encode — the writer's own local slot at
        // checkpoint-write time — to this reader's own local slot for the
        // same pack_id. Slot numbers are a process-local handle, not a
        // stable identity: a fresh reader's `discover_shards` reassigns
        // slots by first-free-in-pack_id-order, which does not reproduce
        // the writer's numbering once any shard has ever been retired (see
        // `CHECKPOINT_VERSION`'s v6 doc comment). A checkpoint_slot with no
        // entry here (its pack_id isn't among this reader's currently-open
        // packs) is left unmapped; any index entry that actually needs it
        // makes `remap_slots` fail closed below, forcing a full rescan
        // rather than serving an unresolvable slot.
        let pack_id_to_local_slot: HashMap<u64, u16> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*pack_id, *slot))
            .collect();
        let slot_remap: HashMap<u16, u16> = checkpoint
            .pack_table
            .iter()
            .filter_map(|&(checkpoint_slot, pack_id)| {
                pack_id_to_local_slot
                    .get(&pack_id)
                    .map(|&local_slot| (checkpoint_slot, local_slot))
            })
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
            if deleted_collections.contains(&loaded.collection_id)
                || v3_tombstones.contains(&loaded.collection_id)
            {
                // The checkpoint may predate the deletion marker; the
                // logical-delete set is authoritative.
                continue;
            }
            // The checkpoint reader has already validated this range. Keep it
            // mmap-backed through the read-only fast path; the first writer
            // copy-on-writes it into the normal atomic slot array.
            let (mut index, generation) = if let Some((generation, index)) =
                snapshot_indexes.remove(&loaded.collection_id)
            {
                (index, generation)
            } else {
                let mmap_index = if loaded.has_homes_tails {
                    LossyIndex::from_mmap_slots_with_homes_tails(
                        Arc::clone(&checkpoint.mmap),
                        loaded.slots_offset,
                        loaded.homes_offset,
                        loaded.tails_offset,
                        loaded.capacity,
                        loaded.slot_count,
                        index_config,
                    )
                } else {
                    LossyIndex::from_mmap_slots_with_config(
                        Arc::clone(&checkpoint.mmap),
                        loaded.slots_offset,
                        loaded.capacity,
                        loaded.slot_count,
                        index_config,
                    )
                };
                let index = if let Some(frames) = frames_by_collection.get(&loaded.collection_id) {
                    let owned = mmap_index.clone();
                    if owned.replay_frames(frames).is_err() {
                        if writable {
                            let _ = std::fs::remove_file(&delta_path);
                        }
                        return None;
                    }
                    owned
                } else {
                    mmap_index
                };
                (index, loaded.generation)
            };
            if !index.remap_slots(&slot_remap) {
                return None;
            }
            current_len.insert(loaded.collection_id, u32::try_from(index.len()).ok()?);
            scan_out.collections.insert(
                loaded.collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::new(cache_capacity)),
                    generation,
                }),
            );
            collection_order.push(loaded.collection_id);
        }
        for (collection_id, (generation, mut index)) in snapshot_indexes {
            if deleted_collections.contains(&collection_id) {
                continue;
            }
            if !index.remap_slots(&slot_remap) {
                return None;
            }
            current_len.insert(collection_id, u32::try_from(index.len()).ok()?);
            scan_out.collections.insert(
                collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::new(cache_capacity)),
                    generation,
                }),
            );
            collection_order.push(collection_id);
        }
        if replay_log_version == 3 {
            collection_order
                .sort_unstable_by_key(|id| (live_order.get(id).copied().unwrap_or(u64::MAX), *id));
        }
        timings.index_materialization = materialization_started.elapsed();

        let mut all_deleted_collections = deleted_collections.clone();
        all_deleted_collections.extend(v3_tombstones.iter().copied());
        let (bookkeeping_source, sidecar_counts) = Self::gated_sidecar_bookkeeping(
            base_dir,
            local_fingerprint,
            &current_len,
            open_shards,
            &all_deleted_collections,
        );
        timings.bookkeeping_source = bookkeeping_source;
        if bookkeeping_source == BookkeepingSource::Sidecar {
            if let Some(directory) = read_persisted_shard_collections(base_dir) {
                for record in directory.records {
                    let total = scan_out
                        .collection_disk_bytes
                        .entry(record.pack_id)
                        .or_default()
                        .entry(record.collection_id)
                        .or_default();
                    *total = total.saturating_add(record.disk_bytes);
                }
            }
        } else if writable {
            // No valid sidecar to seed the physical byte totals from. Rebuild
            // them exactly (every appended frame, as `physical_layout`
            // counts) rather than persisting zeros, and make the next sync
            // rewrite the sidecar. This is a recovery path, so one pack scan
            // is acceptable.
            scan_out.sidecar_missing = true;
            scan_out.sidecar_recovery_failed = true;
            if let Ok(physical) = crate::packfile::layout::physical_layout(base_dir) {
                scan_out.collection_disk_bytes = disk_bytes_from_physical(&physical);
                scan_out.sidecar_recovery_failed = false;
            }
        }

        // Bookkeeping (home-shard seeding + shard directory) from the sidecar
        // where that passed its gates, otherwise a slot walk of the — possibly
        // replayed — live indexes.
        let counts_lookup: HashMap<[u8; 16], HashMap<u16, u64>> = match bookkeeping_source {
            BookkeepingSource::Sidecar => sidecar_counts,
            BookkeepingSource::SlotScan => {
                let mut out = HashMap::with_capacity(collection_order.len());
                for collection_id in &collection_order {
                    if let Some(gen) = scan_out.collections.get(collection_id) {
                        out.insert(*collection_id, gen.load().index.slot_counts());
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
            for (&slot, &count) in counts {
                let pack_id = slot_to_pack_id
                    .get(&slot)
                    .copied()
                    .unwrap_or(u64::from(slot));
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
            base_generations: live_generations,
            next_order_key: live_order
                .values()
                .copied()
                .max()
                .map_or(0, |last| last.saturating_add(1)),
            base_order: live_order,
            log_bytes: log_bytes_on_disk,
            log_version: replay_log_version,
            ..DeltaLogState::default()
        };

        Some((scan_out, collection_order, delta_state, covered_lsn))
    }

    fn generation(&self, collection_id: &[u8; 16]) -> Option<arc_swap::Guard<Arc<RoomGeneration>>> {
        self.collections_read()
            .get(collection_id)
            .map(arc_swap::ArcSwapAny::load)
    }

    fn collections_read(
        &self,
    ) -> parking_lot::MappedRwLockReadGuard<'_, HashMap<[u8; 16], ArcSwap<RoomGeneration>>> {
        parking_lot::RwLockReadGuard::map(self.index_tables.read(), |tables| &tables.collections)
    }

    fn collections_write(
        &self,
    ) -> parking_lot::MappedRwLockWriteGuard<'_, HashMap<[u8; 16], ArcSwap<RoomGeneration>>> {
        parking_lot::RwLockWriteGuard::map(self.index_tables.write(), |tables| {
            &mut tables.collections
        })
    }

    fn shard_collections_read(
        &self,
    ) -> parking_lot::MappedRwLockReadGuard<'_, HashMap<u64, HashMap<[u8; 16], u64>>> {
        parking_lot::RwLockReadGuard::map(self.index_tables.read(), |tables| {
            &tables.shard_collections
        })
    }

    /// Collection IDs currently known to this engine, sorted for deterministic output.
    pub fn collection_ids(&self) -> Vec<[u8; 16]> {
        let mut ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Collection IDs whose live indexes currently reference records in
    /// `slot`. A collection repacked away from the slot is excluded even if
    /// its old bytes remain in the append-only pack. Order is unspecified.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] with `NotFound` if `slot` does not
    /// correspond to an open shard.
    pub fn collections_referencing_shard(&self, slot: u16) -> Result<Vec<[u8; 16]>, StorageError> {
        let shard = self.shards.get_shard(slot).ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no open shard with id {slot}"),
            ))
        })?;
        // O(1) against the incrementally-maintained shard→collection directory
        // instead of scanning the shard file — this used to be the
        // dominant cost of shard retirement/evacuation-style operations
        // on a large shard.
        Ok(self
            .shard_collections_read()
            .get(&shard.pack_id)
            .map(|collections| collections.keys().copied().collect())
            .unwrap_or_default())
    }

    /// Repack every collection that still references `slot`.
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
        slot: u16,
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let collections = self.collections_referencing_shard(slot)?;
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
        self.collections_read().get(collection_id).map(|gen| {
            let g = gen.load();
            (g.index.len(), g.index.memory_usage(), g.index.capacity())
        })
    }

    /// `(collection_id, entry count, memory usage in bytes, index capacity)`
    /// for every known collection, sorted by collection ID, in a single pass
    /// over the collection map.
    pub fn collection_summaries(&self) -> Vec<CollectionSummary> {
        let tables = self.index_tables.read();
        tables
            .collection_order
            .iter()
            .filter_map(|id| {
                tables.collections.get(id).map(|gen| {
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
    /// `mtxdb` currently does so.
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
        for (slot, shard) in self.shards.all_shards() {
            match packfile::scan_packfile(&shard.path) {
                Ok(entries) => {
                    let collection_entries: Vec<([u8; 16], u64)> = entries
                        .into_iter()
                        .filter(|(rid, _, _)| rid == collection_id)
                        .map(|(_, hash, offset)| (hash, offset))
                        .collect();
                    if !collection_entries.is_empty() {
                        scanned.push((slot, collection_entries));
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
        for (slot, shard) in self.shards.all_shards() {
            for (collection_id, hash, offset) in packfile::scan_packfile(&shard.path)? {
                if wanted.contains(&collection_id) {
                    maps.entry(collection_id)
                        .or_default()
                        .insert(hash, (slot, offset));
                }
            }
        }
        Ok(maps)
    }

    fn build_index(
        offsets: &[([u8; 16], u16, u64)],
        index_config: crate::index::IndexConfig,
    ) -> Result<LossyIndex, StorageError> {
        let index = LossyIndex::with_config(
            offsets
                .len()
                .saturating_mul(2)
                .max(NEW_COLLECTION_INDEX_FLOOR),
            index_config,
        );
        for (hash, slot, offset) in offsets {
            check_index_offset(*slot, hash, *offset)?;
            let _ = index.insert(hash, *slot, *offset);
        }
        Ok(index)
    }

    /// Convert a slot-keyed `slot_counts` from `LossyIndex` to a pack_id-keyed
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

    /// Records one new record landing in `slot` for `collection_id` — the
    /// cheap, O(1) path for a plain `put` that only appends, never moves
    /// or drops anything. Kept separate from
    /// [`Self::replace_collection_shard_counts`], which pays for a full
    /// per-shard recount and is reserved for the cases that actually
    /// change a collection's existing distribution (an index rebuild or a
    /// repack), so a normal write never regresses to O(collection size).
    fn record_new_shard_collection(&self, pack_id: u64, collection_id: &[u8; 16], disk_bytes: u64) {
        let mut tables = self.index_tables.write();
        let count = tables
            .shard_collections
            .entry(pack_id)
            .or_default()
            .entry(*collection_id)
            .or_insert(0);
        *count = count.saturating_add(1);
        tables
            .collection_shards
            .entry(*collection_id)
            .or_default()
            .insert(pack_id);
        let bytes = tables
            .collection_disk_bytes
            .entry(pack_id)
            .or_default()
            .entry(*collection_id)
            .or_default();
        *bytes = bytes.saturating_add(disk_bytes);
    }

    fn record_put_many_shard_collections(
        &self,
        collection_id: &[u8; 16],
        count_packs: &[u64],
        byte_packs: &[(u64, u64)],
    ) {
        let mut tables = self.index_tables.write();
        for &pack_id in count_packs {
            let count = tables
                .shard_collections
                .entry(pack_id)
                .or_default()
                .entry(*collection_id)
                .or_insert(0);
            *count = count.saturating_add(1);
            tables
                .collection_shards
                .entry(*collection_id)
                .or_default()
                .insert(pack_id);
        }
        for &(pack_id, disk_bytes) in byte_packs {
            let bytes = tables
                .collection_disk_bytes
                .entry(pack_id)
                .or_default()
                .entry(*collection_id)
                .or_default();
            *bytes = bytes.saturating_add(disk_bytes);
        }
    }

    fn record_disk_bytes(&self, pack_id: u64, collection_id: &[u8; 16], disk_bytes: u64) {
        let mut tables = self.index_tables.write();
        let bytes = tables
            .collection_disk_bytes
            .entry(pack_id)
            .or_default()
            .entry(*collection_id)
            .or_default();
        *bytes = bytes.saturating_add(disk_bytes);
    }

    fn replace_collection_disk_bytes(
        &self,
        collection_id: &[u8; 16],
        bytes_by_pack: &HashMap<u64, u64>,
    ) {
        let mut tables = self.index_tables.write();
        for collections in tables.collection_disk_bytes.values_mut() {
            collections.remove(collection_id);
        }
        for (&pack_id, &disk_bytes) in bytes_by_pack {
            tables
                .collection_disk_bytes
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, disk_bytes);
        }
        tables
            .collection_disk_bytes
            .retain(|_, collections| !collections.is_empty());
    }

    /// Replaces `collection_id`'s entire contribution to `shard_collections` with
    /// `counts` (typically `LossyIndex::slot_counts()` converted to `pack_id` keys)
    /// — clears it out of any shard it no longer occupies and installs the fresh
    /// per-shard counts. Used wherever a collection's index is replaced wholesale
    /// rather than incrementally appended to, since only then can its distribution
    /// across shards actually change.
    fn replace_collection_shard_counts(
        &self,
        collection_id: &[u8; 16],
        counts: &HashMap<u64, u64>,
    ) {
        let mut tables = self.index_tables.write();
        let old_shards = tables
            .collection_shards
            .insert(*collection_id, counts.keys().copied().collect());
        if let Some(old_shards) = old_shards {
            for pack_id in &old_shards {
                if !counts.contains_key(pack_id) {
                    if let Some(m) = tables.shard_collections.get_mut(pack_id) {
                        m.remove(collection_id);
                        if m.is_empty() {
                            tables.shard_collections.remove(pack_id);
                        }
                    }
                }
            }
        }
        for (&pack_id, &count) in counts {
            tables
                .shard_collections
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, count);
        }
    }

    /// Removes `collection_id` from `shard_collections`/`collection_shards` entirely —
    /// used on collection deletion, where nothing of the collection survives in any
    /// shard.
    fn remove_collection_shard_counts(&self, collection_id: &[u8; 16]) {
        let mut tables = self.index_tables.write();
        let Some(old_shards) = tables.collection_shards.remove(collection_id) else {
            return;
        };
        for pack_id in old_shards {
            if let Some(m) = tables.shard_collections.get_mut(&pack_id) {
                m.remove(collection_id);
                if m.is_empty() {
                    tables.shard_collections.remove(&pack_id);
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
        if self
            .shard_collections_recovery_failed
            .load(Ordering::Acquire)
        {
            // The open-time recovery scan failed, so the byte totals are not
            // trustworthy. Retry it now; until it succeeds nothing is written,
            // so zero-byte fallbacks are never published.
            //
            // The scan and the swap of the in-memory totals must not interleave
            // with an append: a frame written after the scan would be missing
            // from the totals while the persisted fingerprint already covers
            // it, and nothing would ever correct that. So hold every
            // collection's put mutex (the same sorted-lock pattern the
            // checkpoint writer uses; no caller holds one when it persists)
            // across the flush, the scan, and the swap. This only runs on the
            // rare recovery path, so briefly stalling writers is acceptable.
            // Take the creation lock first, then read the collection set: a
            // collection created earlier is in the set, and none can appear
            // while it is held. Sort so every mutex is acquired in one global
            // order, whatever else locks several collections at once.
            let create_guard = self.collection_creation.write();
            let mut collection_ids: Vec<[u8; 16]> =
                self.collections_read().keys().copied().collect();
            collection_ids.sort_unstable();
            collection_ids.dedup();
            let lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
            let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();
            self.shards.flush_all().map_err(StorageError::Io)?;
            let physical = crate::packfile::layout::physical_layout(&self.base_dir)
                .map_err(StorageError::Io)?;
            #[cfg(test)]
            {
                let hook = self.recovery_pause_hook.lock().clone();
                if let Some(hook) = hook {
                    hook();
                }
            }
            self.index_tables.write().collection_disk_bytes = disk_bytes_from_physical(&physical);
            drop(guards);
            drop(create_guard);
            self.shard_collections_recovery_failed
                .store(false, Ordering::Release);
        }
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
        let records = {
            let tables = self.index_tables.read();
            let collection_order: HashMap<[u8; 16], u64> = tables
                .collection_order
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
            let mut records = Vec::new();
            for (pack_id, collections) in &tables.shard_collections {
                for (collection_id, count) in collections {
                    let order = collection_order
                        .get(collection_id)
                        .copied()
                        .unwrap_or(u64::MAX);
                    let disk_bytes = tables
                        .collection_disk_bytes
                        .get(pack_id)
                        .and_then(|collections| collections.get(collection_id))
                        .copied()
                        .unwrap_or(0);
                    records.push((*pack_id, *collection_id, *count, order, disk_bytes));
                }
            }
            records
        };
        for (pack_id, collection_id, count, order, disk_bytes) in records {
            buf.extend_from_slice(&pack_id.to_le_bytes());
            buf.extend_from_slice(&collection_id);
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(&order.to_le_bytes());
            buf.extend_from_slice(&disk_bytes.to_le_bytes());
        }

        let unique = SHARD_ROOMS_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = Self::shard_collections_path(&self.base_dir)
            .with_extension(format!("bin.tmp.{}.{unique}", std::process::id()));
        let final_path = Self::shard_collections_path(&self.base_dir);
        let write_result = (|| -> std::io::Result<()> {
            let mut tmp = fs::File::create(&tmp_path)?;
            std::io::Write::write_all(&mut tmp, &buf)?;
            // See the matching comment in `checkpoint::write_checkpoint`:
            // this tmp file is renamed away immediately below and never
            // reopened by this name, so only its data need survive —
            // `sync_data` (fdatasync) skips the timestamp-only metadata
            // flush `sync_all` (fsync) would also perform.
            tmp.sync_data()
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(StorageError::Io(e));
        }
        fs::rename(&tmp_path, &final_path).map_err(StorageError::Io)?;
        // Counted only after the rename succeeds, so the metric reflects
        // completed sidecar writes, not attempts (no containing-directory
        // fsync is issued, so "completed" stops short of claiming power-loss
        // durability for the directory entry itself).
        self.sidecar_writes.fetch_add(1, Ordering::Relaxed);
        self.shard_collections_dirty.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Best-effort wrapper around [`Self::persist_shard_collections`] — logs and
    /// swallows a failure rather than turning it into a hard error, same
    /// contract as `ShardPool`'s stats persistence: this is observability
    /// data, not something worth failing an otherwise-successful sync
    /// over.
    fn persist_shard_collections_best_effort(&self) {
        match self.persist_shard_collections() {
            Ok(()) => self
                .shard_collections_failure_logged
                .store(false, Ordering::Relaxed),
            Err(e) => {
                // A persistent failure would otherwise log on every sync.
                if !self
                    .shard_collections_failure_logged
                    .swap(true, Ordering::Relaxed)
                {
                    eprintln!("mtxdb: failed to persist shard→collection directory: {e}");
                }
            }
        }
    }

    /// Replace this collection's pending slot deltas with a whole-index v3
    /// snapshot after a structural change (capacity growth, rebuild, repack,
    /// refresh, or collection creation). The counter records collection-level
    /// snapshot promotions; it no longer means a store-wide checkpoint rewrite.
    fn invalidate_delta_log(&self, collection_id: &[u8; 16]) {
        self.delta_invalidations.fetch_add(1, Ordering::Relaxed);
        self.delta_state
            .lock()
            .pending
            .insert(*collection_id, PendingDelta::Snapshot);
    }

    fn mark_v3_collection_deleted(&self, collection_id: &[u8; 16]) {
        let mut state = self.delta_state.lock();
        // The tombstone validates against the generation visible to replay
        // immediately before this operation, which may be an earlier
        // checkpoint/snapshot generation rather than the just-deleted live
        // generation (for example, if growth and deletion happened in one
        // unsynced interval).
        let generation = state
            .base_generations
            .get(collection_id)
            .copied()
            .unwrap_or(0);
        state
            .pending
            .insert(*collection_id, PendingDelta::Delete { generation });
    }

    /// Record a successful index mutation as a delta frame, if the delta log
    /// can legitimately continue through it.
    ///
    /// Recording stops once a snapshot or tombstone supersedes that
    /// collection's pending slot updates. A generation mismatch promotes the
    /// pending update to a snapshot instead of allowing replay against the
    /// wrong table shape.
    fn record_delta(&self, collection_id: &[u8; 16], generation: u64, bucket: u32, slot: u64) {
        let mut state = self.delta_state.lock();
        let base_generation_matches =
            state.base_generations.get(collection_id) == Some(&generation);
        match state.pending.entry(*collection_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                if base_generation_matches {
                    entry.insert(PendingDelta::Slots(vec![DeltaFrame {
                        collection_id: *collection_id,
                        bucket,
                        generation,
                        slot,
                    }]));
                } else {
                    entry.insert(PendingDelta::Snapshot);
                }
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if let PendingDelta::Slots(frames) = entry.get_mut() {
                    if frames.iter().all(|frame| frame.generation == generation) {
                        frames.push(DeltaFrame {
                            collection_id: *collection_id,
                            bucket,
                            generation,
                            slot,
                        });
                    } else {
                        entry.insert(PendingDelta::Snapshot);
                    }
                }
            }
        }
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
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
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
        let ids_now: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
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
        //
        // With a journal enabled the checkpoint is also the point at which the
        // packfiles themselves must become durable — the WAL only covers the
        // post-checkpoint suffix — so fsync every shard here, not just flush.
        if self.journal().is_some() {
            self.shards.sync_all()?;
        } else {
            self.shards.flush_all()?;
        }
        let live_shards = self.shards.all_shards();
        let packs: Vec<(u64, u64)> = live_shards
            .iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        let fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);
        // Every pack live right now, keyed by this writer's own local slot —
        // lets a reader translate this checkpoint's slot-encoded index
        // entries to its own local slot for the same pack_id, rather than
        // trusting the writer's raw slot number (see CHECKPOINT_VERSION's
        // doc comment on the v6 bump for why that's unsafe after a
        // retirement).
        let pack_table: Vec<(u16, u64)> = live_shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();

        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let tables = self.index_tables.read();
            tables
                .collection_order
                .iter()
                .filter_map(|collection_id| {
                    tables.collections.get(collection_id).map(|g| {
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
        // Every collection's put mutex is held here, so no `put` is mid-flight
        // and every mutation published so far has completed its index update.
        // The recorded LSN must be *committed*, not merely published: a
        // concurrent put can publish above the last WAL commit, and LSNs above
        // `committed_lsn` are discarded and reused after a crash. Recording one
        // as covered would make a reopen skip a future mutation that reuses
        // that LSN (the journal scan only knows the committed prefix). The
        // committed LSN is conservative -- the fsynced packs above may cover
        // more -- and replaying that suffix again is idempotent.
        let wal_lsn = self.journal().map(|journal| journal.committed_lsn());
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

        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            wal_lsn.unwrap_or(0),
            &snapshots,
            &pack_table,
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
        // The checkpoint naming `fingerprint` is durable; record the journal
        // LSN it covers so a reopen replays only mutations after it. Because
        // the journal branch above fsynced every shard before this point,
        // everything through `lsn` is now durable in the packfiles: the same
        // coverage lets the segment drop its pre-checkpoint groups, so the
        // journal holds only the post-checkpoint suffix instead of growing
        // without bound. Best-effort — a failed compaction costs disk, never
        // correctness (the checkpoint is already durable).
        if let Some(lsn) = wal_lsn {
            self.write_journal_lsn(lsn)?;
            if let Some(journal) = self.journal() {
                if let Err(error) = journal.reclaim_through(lsn) {
                    eprintln!("warning: journal reclaim through LSN {lsn} failed: {error}");
                }
            }
        }
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
            !state.pending.is_empty()
        };
        if !has_unfinished_work {
            self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        }
        drop(guards);
        Ok(())
    }

    /// Serialize one immutable generation snapshot and write its checkpoint.
    /// Shared by the production writer and test mirrors so they use identical
    /// collection, generation, and coverage encoding.
    fn write_checkpoint_snapshot(
        path: &Path,
        fingerprint: u64,
        covered_lsn: u64,
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
        pack_table: &[(u16, u64)],
    ) -> std::io::Result<()> {
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
            path,
            fingerprint,
            covered_lsn,
            &blobs,
            pack_table,
        )
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
        state.base_order = snapshots
            .iter()
            .enumerate()
            .map(|(order, (collection_id, _, _))| {
                (*collection_id, u64::try_from(order).unwrap_or(u64::MAX))
            })
            .collect();
        state.next_order_key = u64::try_from(snapshots.len()).unwrap_or(u64::MAX);
        state.pending.clear();
        state.log_bytes = 0;
        state.log_version = 3;
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
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mut lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let create_guard = self.collection_creation.write();
        let ids_now: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        for id in ids_now.iter().filter(|id| !collection_ids.contains(id)) {
            lock_arcs.push(self.put_mutex(id));
        }
        let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let tables = self.index_tables.read();
            tables
                .collection_order
                .iter()
                .filter_map(|collection_id| {
                    tables.collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.journal().map_or(0, |journal| journal.committed_lsn());
        drop(guards);
        drop(create_guard);

        // The test waits on this: it now *knows* the rewrite is sleeping in
        // its unlocked window rather than guessing via a fixed sleep, so a
        // put issued from here provably overlaps the delay.
        entered_unlocked_window.store(true, Ordering::Release);

        std::thread::sleep(delay);

        let pack_table: Vec<(u16, u64)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            &snapshots,
            &pack_table,
        )
        .map_err(StorageError::Io)?;
        let _ = fs::File::open(&self.base_dir).and_then(|dir| dir.sync_all());
        self.retire_delta_epoch(old_base_fingerprint);
        let guards = lock_arcs.iter().map(|m| m.lock()).collect::<Vec<_>>();
        let has_unfinished_work = {
            let state = self.delta_state.lock();
            !state.pending.is_empty()
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
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all().unwrap();
        let fingerprint = self.current_pack_fingerprint();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let tables = self.index_tables.read();
            tables
                .collection_order
                .iter()
                .filter_map(|collection_id| {
                    tables.collections.get(collection_id).map(|g| {
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
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let snapshots: Vec<([u8; 16], u64, Arc<RoomGeneration>)> = {
            let tables = self.index_tables.read();
            tables
                .collection_order
                .iter()
                .filter_map(|collection_id| {
                    tables.collections.get(collection_id).map(|g| {
                        let generation = arc_swap::ArcSwapAny::load_full(g);
                        (*collection_id, generation.generation, generation)
                    })
                })
                .collect()
        };
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.journal().map_or(0, |journal| journal.committed_lsn());
        drop(guards);
        let pack_table: Vec<(u16, u64)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            &snapshots,
            &pack_table,
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

    /// Append the pending delta operations and clear the dirty flag. The
    /// on-disk log, if any, is
    /// continued; otherwise a fresh header pins `base_fingerprint` (the
    /// checkpoint the frames extend) for the reopen replay gate.
    fn append_index_delta(&self) -> Result<(), StorageError> {
        self.append_index_delta_v3()
    }

    fn build_v3_pending_batch(
        &self,
        state: &DeltaLogState,
    ) -> Result<V3PendingBatch, StorageError> {
        let collections = self.collections_read();
        let mut operations = Vec::new();
        let mut pending_ids: Vec<_> = state.pending.keys().copied().collect();
        pending_ids.sort_unstable();
        let mut generation_updates = Vec::new();
        let mut order_updates = Vec::new();
        let mut next_order_key = state.next_order_key;
        for collection_id in pending_ids {
            let pending = state
                .pending
                .get(&collection_id)
                .expect("pending id exists");
            match pending {
                PendingDelta::Slots(frames) => {
                    let Some(base_generation) = state.base_generations.get(&collection_id) else {
                        return Err(StorageError::Io(std::io::Error::other(
                            "incremental v3 frames target a collection without a base",
                        )));
                    };
                    if frames
                        .iter()
                        .any(|frame| frame.generation != *base_generation)
                    {
                        return Err(StorageError::Io(std::io::Error::other(
                            "incremental v3 frame generation differs from its base",
                        )));
                    }
                    operations.extend(frames.iter().copied().map(DeltaOperation::Incremental));
                }
                PendingDelta::Snapshot => {
                    if let Some(room) = collections.get(&collection_id) {
                        let generation = arc_swap::ArcSwapAny::load_full(room);
                        let order_key =
                            if let Some(order_key) = state.base_order.get(&collection_id) {
                                *order_key
                            } else {
                                let order_key = next_order_key;
                                next_order_key = next_order_key.saturating_add(1);
                                order_key
                            };
                        operations.push(DeltaOperation::CollectionSnapshot {
                            collection_id,
                            generation: generation.generation,
                            order_key,
                            index_blob: generation.index.serialize(),
                        });
                        generation_updates.push((collection_id, generation.generation));
                        order_updates.push((collection_id, order_key));
                    } else if let Some(&generation) = state.base_generations.get(&collection_id) {
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
                PendingDelta::Delete { generation } => {
                    if state.base_generations.contains_key(&collection_id) {
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation: *generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
            }
        }
        Ok(V3PendingBatch {
            operations,
            generation_updates,
            order_updates,
            next_order_key,
        })
    }

    fn write_v3_delta_batch(
        &self,
        state: &mut DeltaLogState,
        base_fingerprint: u64,
        operations: &[DeltaOperation],
        tail_fingerprint: u64,
    ) -> Result<(), StorageError> {
        let path = Self::delta_path(&self.base_dir, base_fingerprint);
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
                "v3 delta log shrank below its sealed frontier",
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
        let batch_bytes = delta::v3_batch_len(operations)
            .ok_or_else(|| StorageError::Io(std::io::Error::other("v3 batch size overflow")))?;
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
                "v3 delta log cap reached",
            )));
        }
        let bytes_written = delta::append_v3_batch(
            &path,
            write_header,
            base_fingerprint,
            operations,
            tail_fingerprint,
        )
        .map_err(StorageError::Io)?;
        state.log_bytes = state
            .log_bytes
            .saturating_add(u64::try_from(bytes_written).unwrap_or(u64::MAX));
        Ok(())
    }

    /// Append v3 operations. Collection locks pin every live index
    /// and the collection-creation lock prevents an unrepresented collection
    /// from appearing between pack flush and snapshot capture.
    fn append_index_delta_v3(&self) -> Result<(), StorageError> {
        let create_guard = self.collection_creation.write();
        let mut ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        {
            let state = self.delta_state.lock();
            ids.extend(state.pending.keys().copied());
        }
        ids.sort_unstable();
        ids.dedup();
        let lock_arcs: Vec<_> = ids.iter().map(|id| self.put_mutex(id)).collect();
        let guards: Vec<_> = lock_arcs.iter().map(|lock| lock.lock()).collect();

        // A put may have completed after the caller's initial dirty flush but
        // before these locks were acquired. Flush again while the snapshot set
        // is stable so the tail fingerprint covers every serialized slot.
        self.shards.flush_all()?;
        let tail_fingerprint = self.current_pack_fingerprint();

        // Copy the small state-machine metadata, then release its mutex while
        // serializing whole-index snapshots. Collection put locks and the
        // creation lock keep this snapshot stable; the persistence lock keeps
        // another sync from advancing the on-disk frontier concurrently.
        let snapshot_state = self.delta_state.lock().clone();
        let Some(base_fingerprint) = snapshot_state.base_fingerprint else {
            return Err(StorageError::Io(std::io::Error::other(
                "v3 delta append with no base fingerprint",
            )));
        };
        if snapshot_state.log_version != 3 {
            return Err(StorageError::Io(std::io::Error::other(
                "v3 delta append without a clean v3 checkpoint base",
            )));
        }
        if snapshot_state.pending.is_empty() {
            return Err(StorageError::Io(std::io::Error::other(
                "v3 delta append with no pending operations",
            )));
        }

        let batch = self.build_v3_pending_batch(&snapshot_state)?;
        let V3PendingBatch {
            operations,
            generation_updates,
            order_updates,
            next_order_key,
        } = batch;

        let mut state = self.delta_state.lock();
        if state.pending != snapshot_state.pending
            || state.base_fingerprint != snapshot_state.base_fingerprint
            || state.log_bytes != snapshot_state.log_bytes
            || state.log_version != snapshot_state.log_version
        {
            return Err(StorageError::Io(std::io::Error::other(
                "v3 delta state changed while snapshot batch was prepared",
            )));
        }
        self.write_v3_delta_batch(&mut state, base_fingerprint, &operations, tail_fingerprint)?;
        for (collection_id, generation) in generation_updates {
            if generation == 0 {
                state.base_generations.remove(&collection_id);
                state.base_order.remove(&collection_id);
            } else {
                state.base_generations.insert(collection_id, generation);
            }
        }
        for (collection_id, order_key) in order_updates {
            state.base_order.insert(collection_id, order_key);
        }
        state.next_order_key = state.next_order_key.max(next_order_key);
        state.pending.clear();
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        drop(state);
        drop(guards);
        drop(create_guard);
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
                )?)
                .ok()?;
                Some((
                    collection_id,
                    nodes,
                    LossyIndex::memory_usage_for_entries(nodes, NEW_COLLECTION_INDEX_FLOOR)?,
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
            // A zero count means this (pack, collection) pair has no live
            // records anymore (e.g. the collection was deleted): the sidecar
            // still carries the entry, but reporting it here would give
            // callers like the CLI's "found in -t <other>" hint a false
            // positive pointing at a shard that no longer holds the data.
            if record.count == 0 {
                continue;
            }
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

    /// Current physical bytes per collection from the persisted directory.
    /// Values include superseded frames and are `None` for stores whose
    /// sidecar predates persisted physical metrics.
    #[must_use]
    pub fn collection_disk_bytes_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<HashMap<[u8; 16], u64>> {
        let records = read_persisted_shard_collections(base_dir)?.records;
        let mut bytes = HashMap::new();
        for record in records {
            let total = bytes.entry(record.collection_id).or_insert(0_u64);
            *total = total.saturating_add(record.disk_bytes);
        }
        Some(bytes)
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
    fn pin_shards(&self, slots: impl Iterator<Item = u16>) -> HashMap<u16, Arc<Shard>> {
        let unique: HashSet<u16> = slots.collect();
        unique
            .into_iter()
            .filter_map(|id| self.shards.get_shard(id).map(|shard| (id, shard)))
            .collect()
    }

    /// Whether `loaded` still refers to the collection's current generation.
    ///
    /// The read paths use this to detect a repack that swapped the generation
    /// (and so may have retired the shards `loaded`'s index points at) between
    /// an index lookup and pinning those shards. A stable generation means the
    /// pin happened before any retirement.
    fn same_generation(
        current: Option<&arc_swap::Guard<Arc<RoomGeneration>>>,
        loaded: &arc_swap::Guard<Arc<RoomGeneration>>,
    ) -> bool {
        current.is_some_and(|current| Arc::ptr_eq(&**current, &**loaded))
    }

    /// Copy a record from an old (pinned) shard to the active shard.
    /// Returns the new `(hash, slot, offset)` or None if `old_slot`
    /// isn't in `pinned`.
    fn copy_record_to_shard(
        &self,
        collection_id: &[u8; 16],
        pinned: &HashMap<u16, Arc<Shard>>,
        old_slot: u16,
        old_offset: u64,
    ) -> Result<Option<RepackCopiedRecord>, StorageError> {
        let Some(old_shard) = pinned.get(&old_slot) else {
            return Ok(None);
        };
        let record = self.read_at(old_shard, old_offset, true)?;
        let (new_slot, new_offset, disk_bytes) = self.shards.put_record_with_len(&Record {
            collection_id: *collection_id,
            hash: record.hash,
            data: record.data,
            metadata: record.metadata,
        })?;
        Ok(Some((record.hash, new_slot, new_offset, disk_bytes)))
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
    /// `bump_generation` marks a shape change (capacity growth, rebuild,
    /// repack, refresh) rather than a plain copy-on-write update. Shape changes
    /// and new collections are represented by whole-index snapshots in v3.
    fn store_generation(
        &self,
        collection_id: &[u8; 16],
        index: LossyIndex,
        cache: Option<Arc<NodeCache>>,
        bump_generation: bool,
    ) -> Result<(), StorageError> {
        let cache = cache.unwrap_or_else(|| Arc::new(NodeCache::new(self.cache_capacity)));
        let is_new = self.collections_read().get(collection_id).is_none();
        let structural = bump_generation || is_new;
        if structural {
            self.invalidate_delta_log(collection_id);
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
            let mut tables = self.index_tables.write();
            tables
                .collections
                .entry(*collection_id)
                .or_insert_with(|| {
                    ArcSwap::from_pointee(RoomGeneration {
                        index: LossyIndex::new(0),
                        cache: Arc::new(NodeCache::new(self.cache_capacity)),
                        generation: 1,
                    })
                })
                .store(new_gen);
            if !tables.collection_order.contains(collection_id) {
                tables.collection_order.push(*collection_id);
            }
        } else {
            // Fast path: just update the ArcSwap using a read lock on the map.
            // This avoids a global write lock on every single put() call.
            let read_guard = self.collections_read();
            if let Some(arc_swap) = read_guard.get(collection_id) {
                arc_swap.store(new_gen);
            } else {
                // Fallback in case of a race condition with a deletion
                drop(read_guard);
                self.collections_write()
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
        let index = LossyIndex::with_config(
            total.saturating_mul(2).max(NEW_COLLECTION_INDEX_FLOOR),
            self.index_config,
        );
        for (slot, entries) in scanned {
            for (hash, offset) in entries {
                // `IndexEntry` can only represent offsets up to
                // `PACK_INDEX_OFFSET_LIMIT`. Offsets a legacy or externally
                // created oversized shard can no longer fit are rejected
                // rather than silently skipped: writes are already capped at
                // `MAX_SHARD_BYTES`, so a record beyond the limit means the
                // shard did not come from this engine, and silently indexing
                // around it would make its data unreachable while reporting a
                // successful rebuild.
                check_index_offset(slot, &hash, offset)?;
                let _ = index.insert(&hash, slot, offset);
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
        index.grow_by_recovering_hashes(|slot, offset, slot_tag| {
            let shard = self.shards.get_shard(slot).ok_or_else(|| {
                StorageError::Corrupt(format!("checkpoint index refers to missing shard {slot}"))
            })?;
            let (found_collection, hash) = self.shards.record_identity_at(&shard, offset)?;
            if &found_collection != collection_id {
                return Err(StorageError::Corrupt(format!(
                    "checkpoint index offset {offset} in shard {slot} belongs to another collection"
                )));
            }
            if index.tag_for_hash(&hash) != slot_tag {
                return Err(StorageError::Corrupt(format!(
                    "checkpoint index tag does not match frame at offset {offset} in shard {slot}"
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
        slot: u16,
        offset: u64,
    ) -> Result<Result<(u32, u64, bool), InsertError>, StorageError> {
        Ok(self
            .insert_index_undoable(collection_id, index, hash, slot, offset)?
            .map(|(bucket, slot, undo)| (bucket, slot, undo.was_empty())))
    }

    /// Like [`Self::insert_index`], but also returns the [`EntryUndo`]
    /// needed to reverse this one write, for callers mutating a still-live
    /// (not cloned) index across a multi-record batch.
    fn insert_index_undoable(
        &self,
        collection_id: &[u8; 16],
        index: &LossyIndex,
        hash: &NodeId,
        slot: u16,
        offset: u64,
    ) -> Result<Result<(u32, u64, EntryUndo), InsertError>, StorageError> {
        loop {
            match index.insert_undoable(hash, slot, offset) {
                Ok(written) => return Ok(Ok(written)),
                Err(InsertError::TableFull) => return Ok(Err(InsertError::TableFull)),
                Err(InsertError::NeedsIdentity {
                    bucket,
                    slot,
                    offset,
                }) => {
                    let shard = self.shards.get_shard(slot).ok_or_else(|| {
                        StorageError::Corrupt(format!(
                            "index refers to missing shard {slot} while resolving a tag collision"
                        ))
                    })?;
                    let (found_collection, found_hash) =
                        self.shards.record_identity_at(&shard, offset)?;
                    if &found_collection != collection_id
                        || !index.hydrate_slot_identity(bucket, &found_hash)
                    {
                        return Err(StorageError::Corrupt(format!(
                            "index tag candidate at offset {offset} in shard {slot} is inconsistent"
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

    /// Resolve `id` against `candidates` through `pinned` shard handles.
    ///
    /// Holding the pinned `Arc<Shard>`s for the whole read keeps each shard's
    /// file handle alive, so a concurrent repack that swaps the generation and
    /// retires those shard ids cannot turn a live record into a miss while the
    /// lookup is in flight. See [`Self::pin_shards`].
    fn resolve_from_pinned(
        &self,
        id: &NodeId,
        candidates: &[(u16, u64)],
        pinned: &HashMap<u16, Arc<Shard>>,
        track: bool,
    ) -> Result<Option<NodeData>, StorageError> {
        let mut last_err: Option<StorageError> = None;
        for &(slot, offset) in candidates {
            let Some(shard) = pinned.get(&slot) else {
                continue;
            };

            if track {
                self.candidate_reads.fetch_add(1, Ordering::Relaxed);
                if let Ok(len) = Self::record_disk_len_at(shard.as_ref(), offset) {
                    self.candidate_frame_bytes.fetch_add(len, Ordering::Relaxed);
                }
            }

            match self.read_at(
                shard,
                offset,
                self.shards.checksum_policy().verifies_reads(),
            ) {
                Ok(record) => {
                    if record.hash != *id {
                        if track {
                            self.candidate_hash_mismatches
                                .fetch_add(1, Ordering::Relaxed);
                        }
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
            let Some(&(slot, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&slot) else {
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

        // Pin every shard this snapshot's index references before resolving
        // anything. The walk reads against a frozen generation, but a
        // concurrent repack can still retire the shard ids that generation's
        // index points at; without pinning, a node resolved after the swap
        // would miss even though it is still live. See `pin_shards`.
        let pinned = self.pin_shards(
            gen.index
                .referenced_slot_ids()
                .into_iter()
                .enumerate()
                .filter(|&(_, referenced)| referenced)
                .map(|(slot, _)| u16::try_from(slot).expect("shard slot fits u16")),
        );

        for root in frontier {
            if !boundary.contains(root) && !at_cap(&visited) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(data) = self.resolve_pinned(gen, &pinned, &hash)? else {
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
    /// walk consistent against one point-in-time view of the collection.
    ///
    /// `pinned` must hold every shard the snapshot's index references (see
    /// [`Self::pin_shards`]), so a repack that swaps in a new generation and
    /// retires those shards partway through the walk cannot make a later node
    /// miss.
    fn resolve_pinned(
        &self,
        gen: &Arc<RoomGeneration>,
        pinned: &HashMap<u16, Arc<Shard>>,
        id: &NodeId,
    ) -> Result<Option<NodeData>, StorageError> {
        if let Some(data) = gen.cache.get(id) {
            return Ok(Some((*data).clone()));
        }
        let track = self.stats_enabled.load(Ordering::Relaxed);
        let candidates: Vec<(u16, u64)> = gen.index.lookup_all(id).collect();
        if track {
            self.index_candidates
                .fetch_add(candidates.len() as u64, Ordering::Relaxed);
        }
        self.resolve_from_pinned(id, &candidates, pinned, track)
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
        for (hash, &(slot, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&slot) {
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
    /// Builds this repack's deduped `hash → (slot, offset)` map,
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
                .map(|(slot, shard)| (*slot, shard.pack_id))
                .collect();
            // Slot IDs are recyclable.  A cursor (and every entry in the
            // carried live_map) is meaningful only for the pack incarnation
            // that produced it.  On any replacement or disappearance, start
            // over rather than mixing old offsets with the new pack.
            let pack_changed = prev
                .scan_offsets
                .iter()
                .any(|(slot, (pack_id, _))| current_pack_ids.get(slot) != Some(pack_id));
            if pack_changed {
                let scanned = self.scan_collection_records(collection_id)?;
                let mut map = HashMap::new();
                for (slot, entries) in scanned {
                    for (hash, offset) in entries {
                        map.insert(hash, (slot, offset));
                    }
                }
                return Ok((map, None));
            }

            // Scan only bytes appended since the last repack, merging into
            // the stolen live_map in-place — O(delta) work, not O(total).
            let mut map = prev.live_map.clone();
            for (slot, shard) in shards {
                let start = prev
                    .scan_offsets
                    .get(&slot)
                    .filter(|(pack_id, _)| *pack_id == shard.pack_id)
                    .map_or(0, |(_, offset)| *offset);
                let entries =
                    packfile::scan_packfile_from(&shard.path, start).map_err(StorageError::Io)?;
                for (rid, hash, offset) in entries {
                    if rid == *collection_id {
                        map.insert(hash, (slot, offset));
                    }
                }
            }
            Ok((map, Some(prev)))
        } else {
            // First repack: full scan of every shard.
            let scanned = self.scan_collection_records(collection_id)?;
            let mut map = HashMap::new();
            for (slot, entries) in &scanned {
                for (hash, offset) in entries {
                    map.insert(*hash, (*slot, *offset));
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
            let Some(&(slot, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&slot) else {
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

        for (hash, &(slot, offset)) in hash_to_shard_offset {
            // Fast path: node was live last time, edge list is cached.
            if let Some(edges) = prev.adjacency.get(hash) {
                adjacency.insert(*hash, edges.clone());
                continue;
            }
            // Slow path: new node, must read from disk.
            if let Some(shard) = pinned.get(&slot) {
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
        for (slot, shard) in self.shards.all_shards() {
            scan_offsets.insert(slot, (shard.pack_id, shard.file_len()));
        }
        let next_live_map: HashMap<[u8; 16], (u16, u64)> = new_offsets
            .iter()
            .map(|&(hash, slot, offset)| (hash, (slot, offset)))
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
        for (hash, &(slot, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&slot) {
                // Actual on-disk bytes, not the uncompressed upper bound —
                // a repack preflight should report what will really be
                // reclaimed/kept, which is smaller than plaintext size for
                // any frame that compressed.
                if let Ok(bytes) = Self::record_disk_len_at(shard, offset) {
                    if live_set.contains(hash) {
                        shards_touched.insert(slot);
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
            if let Some(slot) = shard_queue.pop() {
                if shards.insert(slot) {
                    for collection_id in self.collections_referencing_shard(slot)? {
                        if collections.insert(collection_id) {
                            collection_queue.push(collection_id);
                        }
                    }
                }
            } else if let Some(collection_id) = collection_queue.pop() {
                for slot in self.collection_referenced_shards(&collection_id) {
                    if !shards.contains(&slot) {
                        shard_queue.push(slot);
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
            .referenced_slot_ids()
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
        self.index_tables
            .read()
            .shard_collections
            .iter()
            .map(|(&slot, collections)| {
                let count = collections
                    .values()
                    .copied()
                    .fold(0u64, u64::saturating_add);
                (slot, count)
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
        let collection_guard = collection_arc.lock();

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
            .map(|&(slot, _)| slot)
            .collect();
        self.shards
            .prepare_collection_repack(collection_id, &source_shards)?;

        let mut new_offsets: Vec<([u8; 16], u16, u64)> = Vec::with_capacity(topo.len());
        let mut new_disk_bytes = HashMap::new();
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_slot, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, &pinned, old_slot, old_offset)?
            {
                let pack_id = self
                    .shards
                    .get_shard(slot)
                    .map_or(u64::from(slot), |shard| shard.pack_id);
                let total = new_disk_bytes.entry(pack_id).or_insert(0_u64);
                *total = (*total).saturating_add(disk_bytes);
                new_offsets.push((hash, slot, offset));
            }
        }

        let kept = new_offsets.len();

        let index = Self::build_index(&new_offsets, self.index_config)?;
        self.replace_collection_disk_bytes(collection_id, &new_disk_bytes);
        self.replace_collection_shard_counts(
            collection_id,
            &self.slot_counts_to_pack_id_counts(&index.slot_counts()),
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

        // Release this collection's put_mutex before persisting the
        // checkpoint: persist_index_checkpoint acquires every collection's
        // put_mutex itself (parking_lot's Mutex is non-reentrant), and this
        // repack still holds this collection's.
        drop(collection_guard);

        // retire_empty_shards may have unlinked packs this collection's old
        // index entries pointed at. A reader that reloads from the
        // checkpoint after this point must see the new shard layout, not
        // the stale one — otherwise reload_index_from_checkpoint rebuilds a
        // fingerprint that can never match a checkpoint naming retired
        // packs, and the read-journal reload path fails closed. Mark the
        // checkpoint dirty (repack itself never does, since it never runs
        // through the put/sync paths that do) and persist it now so the
        // on-disk checkpoint matches what this repack just wrote.
        //
        // Best-effort: the repack itself already succeeded and its shard
        // writes are durable (fsynced above). A checkpoint-write failure
        // here shouldn't be reported as a repack failure — the packfiles
        // stay authoritative and the dirty flag (left set on failure, see
        // `persist_index_checkpoint`) means the next sync retries it.
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.persist_index_checkpoint_best_effort();
        self.persist_shard_collections_best_effort();

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
    ) -> Result<(RepackOffsets, usize, HashMap<u64, u64>), StorageError> {
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
        let mut new_disk_bytes = HashMap::new();
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_slot, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, pinned, old_slot, old_offset)?
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
                let pack_id = self
                    .shards
                    .get_shard(slot)
                    .map_or(u64::from(slot), |shard| shard.pack_id);
                let total = new_disk_bytes.entry(pack_id).or_insert(0_u64);
                *total = (*total).saturating_add(disk_bytes);
                new_offsets.push((hash, slot, offset));
            }
        }

        Ok((new_offsets, dropped, new_disk_bytes))
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
            .flat_map(|map| map.values().map(|&(slot, _)| slot))
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
            let (new_offsets, dropped, new_disk_bytes) = self.copy_repack_batch_collection(
                collection_id,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
                &mut output_progress,
                &mut output_state,
            )?;
            let kept = new_offsets.len();
            let index = Self::build_index(&new_offsets, self.index_config)?;
            self.replace_collection_disk_bytes(collection_id, &new_disk_bytes);
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&index.slot_counts()),
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

        // See the matching comment in repack_collection_reachable: retirement
        // above may have unlinked packs this batch's old index entries
        // pointed at, so the on-disk checkpoint must be refreshed or a
        // reader's checkpoint reload can never match the live shard set.
        // Best-effort for the same reason: the batch's shard writes are
        // already durable, so a checkpoint-write failure here costs the
        // next sync a retry, not correctness.
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.persist_index_checkpoint_best_effort();
        self.persist_shard_collections_best_effort();

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
            .collections_read()
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
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();
        self.retire_empty_shards_locked();
    }

    /// Every collection writer lock is held by the caller.
    fn retire_empty_shards_locked(&self) {
        // Do not retain the map guard while acquiring collection locks: another
        // operation may need the map lock while it holds a collection lock.
        let collections = self.collections_read();

        // Build the union of shard IDs referenced across all collections.
        let mut referenced = [false; shard::MAX_SHARDS];
        for gen_swap in collections.values() {
            let gen = gen_swap.load();
            let ids = gen.index.referenced_slot_ids();
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
        drop(collections);
        let mut tables = self.index_tables.write();
        let live_pack_ids: HashSet<u64> = tables
            .collection_shards
            .values()
            .flat_map(|packs| packs.iter().copied())
            .collect();
        tables
            .collection_disk_bytes
            .retain(|pack_id, _| live_pack_ids.contains(pack_id));
    }
}

impl PackfileStorage {
    /// Fetch records and, if necessary, refresh a stale read-only collection
    /// index once before retrying only the missing keys.
    ///
    /// This is deliberately separate from [`StorageEngine::get_many`], whose
    /// result is a snapshot of the caller's current in-memory index. The
    /// refresh path is intended for multi-process readers that need to observe
    /// records appended by another process. Refreshes are serialized per
    /// collection and successful refreshes are rate-limited briefly so a run
    /// of genuine negative lookups cannot cause a full pack scan per lookup.
    ///
    /// A store whose in-memory index is authoritative (a single writer, see
    /// [`Self::set_refresh_on_miss`]) can disable the refresh entirely; the
    /// call then degrades to a plain [`StorageEngine::get_many`] snapshot,
    /// skipping the refresh path's miss scan/allocation and lock rather than
    /// the underlying index lookups.
    ///
    /// # Errors
    /// Returns [`StorageError`] if reading the collection or refreshing its
    /// on-disk index fails.
    pub fn get_many_with_refresh(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        if !self.refresh_on_miss.load(Ordering::Relaxed) {
            return self.get_many(collection_id, ids);
        }
        let mut results = self.get_many(collection_id, ids)?;
        let mut missing: Vec<usize> = results
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_none().then_some(index))
            .collect();
        if missing.is_empty() {
            return Ok(results);
        }

        let refresh_lock = self.refresh_lock(collection_id);
        let _refresh_guard = refresh_lock.lock();

        // Another reader may have refreshed while this caller waited. Avoid a
        // second refresh and just recheck the keys against the new generation.
        let current_missing_ids: Vec<NodeId> = missing.iter().map(|&i| ids[i]).collect();
        let rechecked = self.get_many(collection_id, &current_missing_ids)?;
        for (index, value) in missing.iter().copied().zip(rechecked) {
            if value.is_some() {
                results[index] = value;
            }
        }
        missing.retain(|index| results[*index].is_none());
        if missing.is_empty() {
            return Ok(results);
        }

        let durable_fp = match crate::index::checkpoint::read_durable_fingerprint(&self.base_dir) {
            Ok(Some(dfp)) => dfp,
            Ok(None) => crate::index::checkpoint::DurableFingerprint {
                fingerprint: 0,
                torn_tail: false,
            },
            Err(e) => {
                return Err(StorageError::Corrupt(format!(
                    "failed to read durable fingerprint: {e}"
                )));
            }
        };

        let dominated = {
            let mut fingerprints = self.last_refresh_fingerprint.lock();
            let baseline = fingerprints
                .entry(*collection_id)
                .or_insert(self.initial_durable_fingerprint);
            *baseline == durable_fp.fingerprint
        };

        if dominated && !durable_fp.torn_tail {
            self.miss_refresh_skips.fetch_add(1, Ordering::Relaxed);
            return Ok(results);
        }

        self.refresh_and_retry(collection_id, ids, missing, &mut results)?;
        Ok(results)
    }
}

impl PackfileStorage {
    fn append_put_many_entry(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
        metadata: Option<FrameMetadata>,
        old_gen: Option<&RoomGeneration>,
        progress: &mut PutManyProgress,
    ) -> Result<(), StorageError> {
        let record = Record {
            collection_id: *collection_id,
            hash: *id,
            data: data.bytes.clone(),
            metadata,
        };
        let (slot, offset, disk_bytes) = self.shards.put_record_with_len(&record)?;
        self.publish_mutation(|| JournalMutation::Put {
            collection_id: *collection_id,
            node_id: *id,
            payload: data.bytes.to_vec(),
        })?;
        let pack_id = self
            .shards
            .get_shard(slot)
            .map_or(u64::from(slot), |shard| shard.pack_id);
        progress
            .pending_shard_collections
            .push((pack_id, disk_bytes));
        if progress.index_needs_rebuild {
            return Ok(());
        }
        self.index_put_many_entry(collection_id, id, old_gen, slot, offset, progress)
    }

    fn index_put_many_entry(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        old_gen: Option<&RoomGeneration>,
        slot: u16,
        offset: u64,
        progress: &mut PutManyProgress,
    ) -> Result<(), StorageError> {
        let live = match progress.owned_index.as_ref() {
            Some(index) => index,
            None => {
                &old_gen
                    .expect("live path implies an existing generation")
                    .index
            }
        };
        let insert_result = self.insert_index_undoable(collection_id, live, id, slot, offset)?;
        let inserted = if let Ok((bucket, slot, undo)) = insert_result {
            let was_empty = undo.was_empty();
            progress.pending_deltas.push((bucket, slot));
            if progress.owned_index.is_none() {
                progress.undo_log.push(undo);
            }
            was_empty
        } else {
            let grow_started = std::time::Instant::now();
            let grown = if let Some(grown) = live.grow() {
                Some(grown)
            } else {
                // Match the single-put fallback: if recovering the
                // checkpoint-backed hashes fails, rebuild this collection
                // from pack records below instead of aborting a valid batch.
                self.grow_checkpoint_index(collection_id, live)
                    .ok()
                    .flatten()
            };
            let Some(grown) = grown else {
                progress.index_needs_rebuild = true;
                progress.structural_change = true;
                progress.invalidate_delta = true;
                return Ok(());
            };
            progress.invalidate_delta = true;
            progress.structural_change = true;
            self.index_grow_count.fetch_add(1, Ordering::Relaxed);
            self.index_clone_time_ns.fetch_add(
                u64::try_from(grow_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let inserted = grown.insert(id, slot, offset).is_ok();
            progress.owned_index = Some(grown);
            inserted
        };
        if inserted {
            let pack_id = self
                .shards
                .get_shard(slot)
                .map_or(u64::from(slot), |shard| shard.pack_id);
            progress.pending_shard_collection_counts.push(pack_id);
        }
        Ok(())
    }

    fn append_put_record(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
        metadata: Option<FrameMetadata>,
    ) -> Result<(u16, u64, u64), StorageError> {
        let record = Record {
            collection_id: *collection_id,
            hash: *id,
            data: data.bytes.clone(),
            metadata,
        };
        let (slot, offset, disk_bytes) = self.shards.put_record_with_len(&record)?;
        self.publish_mutation(|| JournalMutation::Put {
            collection_id: *collection_id,
            node_id: *id,
            payload: data.bytes.to_vec(),
        })?;
        Ok((slot, offset, disk_bytes))
    }

    /// Store a record after verifying that its bytes hash to `expected_digest`,
    /// attaching versioned metadata (logical id, content digest, role) to the
    /// frame so readers can recover it without decoding a caller-supplied
    /// addressing scheme.
    ///
    /// This is the verify-on-write entry point for callers that already know
    /// the content digest they expect (e.g. a Matrix event's canonical
    /// content hash). The digest is recomputed over the exact bytes being
    /// stored; a mismatch is rejected *before* anything is appended, so a
    /// corrupt or mis-addressed write never becomes durable.
    ///
    /// `logical_id` is the full 256-bit logical identity (e.g. the digest of
    /// the event id) stored in the metadata; it is independent of the
    /// 16-byte `id` used as the index key, which survives redaction.
    ///
    /// The digest function is caller-selectable via `algorithm` so a template
    /// or configuration can choose SHA-256 today and BLAKE3/SHA-512 later
    /// without a format break; the chosen algorithm is recorded in the frame
    /// metadata.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] when the digest of `data` under
    /// `algorithm` does not equal `expected_digest`, and propagates any error
    /// from the underlying put.
    #[allow(clippy::too_many_arguments)]
    pub fn put_verified(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
        logical_id: &Digest32,
        algorithm: DigestAlgorithm,
        expected_digest: &Digest32,
        role: Option<&[u8]>,
    ) -> Result<(), StorageError> {
        let digest = crate::storage::content_digest(algorithm, &data.bytes);
        if &digest != expected_digest {
            return Err(StorageError::Corrupt(format!(
                "content digest mismatch: expected {}, got {}",
                hex32(expected_digest),
                hex32(&digest)
            )));
        }
        let metadata = FrameMetadata {
            logical_id: Some(*logical_id),
            content_digest: Some(digest),
            digest_algorithm: algorithm,
            role: role.map(<[u8]>::to_vec),
            unknown: Vec::new(),
        };
        self.put_internal(collection_id, id, data, Some(metadata))
    }

    fn prepare_put_many_progress(
        &self,
        old_gen: Option<&RoomGeneration>,
        entries_len: usize,
    ) -> PutManyProgress {
        let started = std::time::Instant::now();
        let owned_index = match old_gen {
            Some(generation) if !generation.index.is_mmap_backed() => None,
            Some(generation) => Some(generation.index.clone()),
            None => Some(LossyIndex::with_config(
                entries_len
                    .saturating_mul(2)
                    .max(NEW_COLLECTION_INDEX_FLOOR),
                self.index_config,
            )),
        };
        if owned_index.is_some() {
            self.put_many_clone_path_calls
                .fetch_add(1, Ordering::Relaxed);
            self.index_clone_time_ns.fetch_add(
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        } else {
            self.put_many_fast_path_calls
                .fetch_add(1, Ordering::Relaxed);
        }
        PutManyProgress {
            generation: old_gen.map_or(1, |generation| generation.generation),
            structural_change: owned_index.is_some(),
            owned_index,
            index_needs_rebuild: false,
            pending_deltas: Vec::with_capacity(entries_len),
            pending_shard_collections: Vec::with_capacity(entries_len),
            pending_shard_collection_counts: Vec::with_capacity(entries_len),
            invalidate_delta: false,
            undo_log: Vec::new(),
        }
    }

    fn validate_put_many_inputs(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<bool, StorageError> {
        if entries.is_empty() {
            return Ok(false);
        }
        for (id, data) in entries {
            ShardPool::validate_record(&Record {
                collection_id: *collection_id,
                hash: *id,
                data: data.bytes.clone(),
                metadata: None,
            })?;
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
        Ok(true)
    }

    fn put_internal(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
        metadata: Option<FrameMetadata>,
    ) -> Result<(), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        self.put_internal_locked(collection_id, id, data, metadata)
    }

    fn put_internal_locked(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
        metadata: Option<FrameMetadata>,
    ) -> Result<(), StorageError> {
        self.put_calls.fetch_add(1, Ordering::Relaxed);
        self.put_bytes
            .fetch_add(data.bytes.len() as u64, Ordering::Relaxed);
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

        let (slot, offset, disk_bytes) =
            self.append_put_record(collection_id, id, data, metadata)?;
        if let Some(gen) = self.generation(collection_id) {
            if !gen.index.is_mmap_backed() {
                if let Ok((bucket, entry, inserted)) =
                    self.insert_index(collection_id, &gen.index, id, slot, offset)?
                {
                    self.record_delta(collection_id, gen.generation, bucket, entry);
                    let pack_id = self
                        .shards
                        .get_shard(slot)
                        .map_or(u64::from(slot), |shard| shard.pack_id);
                    if inserted {
                        self.record_new_shard_collection(pack_id, collection_id, disk_bytes);
                    } else {
                        self.record_disk_bytes(pack_id, collection_id, disk_bytes);
                    }

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
            let generation = old_gen.as_ref().map_or(1, |g| g.generation);
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
                None => LossyIndex::with_config(NEW_COLLECTION_INDEX_FLOOR, self.index_config),
            };
            let inserted = self.insert_index(collection_id, &index, id, slot, offset)?;
            if let Ok((bucket, entry, is_new)) = inserted {
                self.record_delta(collection_id, generation, bucket, entry);
                let pack_id = self
                    .shards
                    .get_shard(slot)
                    .map_or(u64::from(slot), |s| s.pack_id);
                if is_new {
                    self.record_new_shard_collection(pack_id, collection_id, disk_bytes);
                } else {
                    self.record_disk_bytes(pack_id, collection_id, disk_bytes);
                }
            } else {
                // The insert was rejected (table full). The collection's shape
                // is about to change, so any delta frames would no longer be
                // replayable against a checkpoint at this generation.
                self.invalidate_delta_log(collection_id);
                if let Some(grown) = index.grow() {
                    // The failed insert did not mutate the table, so retry it
                    // after the in-memory rehash. This is the normal capacity
                    // path and must not turn into a full-pack scan.
                    let _ = grown.insert(id, slot, offset);
                    index = grown;
                } else if let Ok(Some(grown)) = self.grow_checkpoint_index(collection_id, &index) {
                    // A checkpoint-backed index has locations but not homes.
                    // Recovering the hashes from those locations is bounded
                    // by this collection, unlike `rebuild_index`'s pack scan.
                    let _ = grown.insert(id, slot, offset);
                    index = grown;
                } else {
                    index = self.rebuild_index(collection_id)?;
                    let _ = index.insert(id, slot, offset);
                }
                // The rebuild re-derived the collection's entire live set from
                // scratch, so its shard distribution needs a full
                // recompute too, not just crediting this one record.
                self.replace_collection_shard_counts(
                    collection_id,
                    &self.slot_counts_to_pack_id_counts(&index.slot_counts()),
                );
                // The recompute above covers live-node counts only. This frame
                // was appended either way, so its bytes still have to be added
                // (the branches above account for it themselves).
                let pack_id = self
                    .shards
                    .get_shard(slot)
                    .map_or(u64::from(slot), |s| s.pack_id);
                self.record_disk_bytes(pack_id, collection_id, disk_bytes);
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

    fn put_many_internal(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
        metadatas: Option<&[Option<FrameMetadata>]>,
    ) -> Result<usize, StorageError> {
        if let Some(metadatas) = metadatas {
            debug_assert_eq!(
                metadatas.len(),
                entries.len(),
                "per-entry metadata must align with entries"
            );
        }
        if !self.validate_put_many_inputs(collection_id, entries)? {
            return Ok(0);
        }
        let collection_arc = self.put_mutex(collection_id);
        let collection_guard = collection_arc.lock();

        // Same comment as in `put`: a brand-new collection must be excluded
        // from a concurrent checkpoint's fingerprint→snapshot window.
        let create_guard = if self.generation(collection_id).is_none() {
            Some(self.collection_creation.read())
        } else {
            None
        };

        let old_gen = self.generation(collection_id);
        let cache = match &old_gen {
            Some(g) => g.cache.clone(),
            None => Arc::new(NodeCache::new(self.cache_capacity)),
        };

        let mut progress = self.prepare_put_many_progress(
            old_gen.as_deref().map(|generation| &**generation),
            entries.len(),
        );
        // Undo entries for writes made directly to `old_gen`'s live index
        // (only populated while `owned_index` is still `None`). Replayed in
        // reverse on any later failure so the live index never ends up
        // observably holding a failed batch's partial prefix -- the
        // property the unconditional clone used to provide for free.
        macro_rules! rollback_and_fail {
            ($error:expr) => {{
                if let Some(g) = &old_gen {
                    for undo in progress.undo_log.iter().rev() {
                        g.index.rollback_slot(undo);
                    }
                }
                drop(create_guard);
                drop(collection_guard);
                return Err(self.persist_failed_batch_boundary($error));
            }};
        }

        for (index, (id, data)) in entries.iter().enumerate() {
            let metadata = metadatas.and_then(|m| m.get(index)).cloned().flatten();
            if let Err(error) = self.append_put_many_entry(
                collection_id,
                id,
                data,
                metadata,
                old_gen.as_deref().map(|generation| &**generation),
                &mut progress,
            ) {
                rollback_and_fail!(error);
            }
        }

        if progress.index_needs_rebuild {
            let rebuilt = match self.rebuild_index(collection_id) {
                Ok(index) => index,
                Err(error) => rollback_and_fail!(error),
            };
            // rebuild_index automatically discovers all the records we just appended
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&rebuilt.slot_counts()),
            );
            progress.owned_index = Some(rebuilt);
        }

        // All fallible work is complete. Only now make the batch visible to
        // the shared index, delta state, and shard bookkeeping.
        if progress.invalidate_delta {
            self.invalidate_delta_log(collection_id);
        } else {
            for (bucket, slot) in progress.pending_deltas {
                self.record_delta(collection_id, progress.generation, bucket, slot);
            }
        }
        self.record_put_many_shard_collections(
            collection_id,
            &progress.pending_shard_collection_counts,
            &progress.pending_shard_collections,
        );

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

        if progress.structural_change {
            let index = progress
                .owned_index
                .expect("structural_change is only set once owned_index is materialized");
            if let Err(error) = self.store_generation(collection_id, index, Some(cache), false) {
                rollback_and_fail!(error);
            }
        } else {
            // Every record landed on the live, already-published index in
            // place -- no new generation to publish, matching `put`'s
            // in-place success path.
            self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }

        Ok(entries.len())
    }
}

impl StorageEngine for PackfileStorage {
    fn collection_exists(&self, collection_id: &[u8; 16]) -> bool {
        self.generation(collection_id).is_some()
    }

    fn collection_len(&self, collection_id: &[u8; 16]) -> Result<Option<usize>, StorageError> {
        Ok(self
            .generation(collection_id)
            .map(|generation| generation.index.len()))
    }

    fn ensure_collection_metadata(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
    ) -> Result<(), StorageError> {
        // Hold the collection's put mutex across the metadata lookup, the
        // collection-existence check, and the append. Otherwise two concurrent
        // writers can both observe an absent genesis record and append
        // conflicting metadata, or a put can land between the existence check
        // and the append and make the genesis record non-first. The mutex is
        // per-instance, so this serializes threads within one process; a second
        // writer process cannot open the store at all (the shard pool holds an
        // exclusive `.mtxdb.lock`), so there is no cross-process race to close.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        if let Some(existing) = self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? {
            let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }
            return Ok(());
        }
        if self.collection_exists(collection_id) {
            return Err(StorageError::Internal(
                "genesis metadata must be written before the collection's first record".to_owned(),
            ));
        }
        self.put_internal_locked(
            collection_id,
            &COLLECTION_METADATA_RECORD_ID,
            &NodeData::new(metadata.encode().into()),
            None,
        )
    }

    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let track = self.stats_enabled.load(Ordering::Relaxed);
        // A concurrent repack can swap the generation and retire the shard ids
        // its index pointed at. Pin the candidate shards and confirm the
        // generation did not change between the index lookup and the pin
        // (retirement only follows a swap); retry against the new generation if
        // it did.
        loop {
            let gen_guard = self.generation(collection_id);
            let Some(gen) = gen_guard.as_deref() else {
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
            let candidates: Vec<(u16, u64)> = gen.index.lookup_all(id).collect();
            #[cfg(test)]
            Self::run_test_before_pin();
            let pinned = self.pin_shards(candidates.iter().map(|&(slot, _)| slot));
            let still_current = match gen_guard.as_ref() {
                Some(loaded) => {
                    let current = self.generation(collection_id);
                    Self::same_generation(current.as_ref(), loaded)
                }
                None => true,
            };
            if !still_current {
                continue;
            }
            if track {
                self.index_candidates
                    .fetch_add(candidates.len() as u64, Ordering::Relaxed);
            }
            let result = self.resolve_from_pinned(id, &candidates, &pinned, track);
            if track {
                self.get_calls.fetch_add(1, Ordering::Relaxed);
                if result.as_ref().is_ok_and(Option::is_none) {
                    self.get_misses.fetch_add(1, Ordering::Relaxed);
                }
            }
            return result;
        }
    }

    fn get_many(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let track = self.stats_enabled.load(Ordering::Relaxed);
        // As in `get`: pin the candidate shards and retry if a concurrent
        // repack swapped the generation (and so may have retired them) between
        // the index lookups and the pin.
        loop {
            let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

            let gen_guard = self.generation(collection_id);
            let mut to_fetch: Vec<(usize, Vec<(u16, u64)>)> = Vec::new();
            let mut candidates_seen: u64 = 0;
            if let Some(g) = gen_guard.as_deref() {
                for (i, id) in ids.iter().enumerate() {
                    // The generation-local cache only contains records from this
                    // generation, so it is authoritative for hits. Avoid an
                    // otherwise redundant index probe for the common warm case.
                    if let Some(data) = g.cache.get(id) {
                        results[i] = Some((*data).clone());
                        continue;
                    }
                    let candidates: Vec<(u16, u64)> = g.index.lookup_all(id).collect();
                    candidates_seen = candidates_seen.saturating_add(candidates.len() as u64);
                    if !candidates.is_empty() {
                        to_fetch.push((i, candidates));
                    }
                }
            }

            #[cfg(test)]
            Self::run_test_before_pin();
            let pinned = self.pin_shards(
                to_fetch
                    .iter()
                    .flat_map(|(_, candidates)| candidates.iter().map(|&(slot, _)| slot)),
            );
            let still_current = match gen_guard.as_ref() {
                Some(loaded) => {
                    let current = self.generation(collection_id);
                    Self::same_generation(current.as_ref(), loaded)
                }
                None => true,
            };
            if !still_current {
                continue;
            }

            to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

            if track {
                self.index_candidates
                    .fetch_add(candidates_seen, Ordering::Relaxed);
                let touched: HashSet<u16> = to_fetch
                    .iter()
                    .flat_map(|(_, candidates)| candidates.iter().map(|(slot, _)| *slot))
                    .collect();
                self.get_many_shards_touched
                    .fetch_add(touched.len() as u64, Ordering::Relaxed);

                let mut per_shard: HashMap<u16, Vec<u64>> = HashMap::new();
                for (_, candidates) in &to_fetch {
                    for (slot, offset) in candidates {
                        per_shard.entry(*slot).or_default().push(*offset);
                    }
                }
                let mut runs: u64 = 0;
                let mut span: u64 = 0;
                for offsets in per_shard.values_mut() {
                    offsets.sort_unstable();
                    let first = *offsets.first().expect("offsets non-empty by construction");
                    let last = *offsets.last().expect("offsets non-empty by construction");
                    runs = runs.saturating_add(1);
                    span = span.saturating_add(last.saturating_sub(first));
                    for pair in offsets.windows(2) {
                        if pair[1].saturating_sub(pair[0]) > READ_RUN_GAP_BYTES {
                            runs = runs.saturating_add(1);
                        }
                    }
                }
                self.read_many_runs.fetch_add(runs, Ordering::Relaxed);
                self.read_many_span_bytes.fetch_add(span, Ordering::Relaxed);
            }

            for (i, candidates) in &to_fetch {
                results[*i] = self.resolve_from_pinned(&ids[*i], candidates, &pinned, track)?;
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

            return Ok(results);
        }
    }

    fn put(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
    ) -> Result<(), StorageError> {
        self.put_internal(collection_id, id, data, None)
    }

    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<usize, StorageError> {
        self.put_many_internal(collection_id, entries, None)
    }

    fn delete_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        // Acquire the collection's put mutex to serialize with any in-flight put,
        // preventing a concurrent put from resurrecting the collection after we
        // remove it from the generation map.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        self.collections_write().remove(collection_id);
        self.live_roots.write().remove(collection_id);
        // A deletion changes the collection directory the same way a refill
        // does, so any pending delta frames are no longer replayable against
        // the checkpoint they were recorded against.
        self.delta_invalidations.fetch_add(1, Ordering::Relaxed);
        self.mark_v3_collection_deleted(collection_id);
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        // Keep lock entries for the storage lifetime. Removing an entry while
        // a caller still owns its Arc permits a later put to obtain a second
        // mutex and bypass this deletion's serialization.
        self.remove_collection_shard_counts(collection_id);
        self.persist_deleted_collection(collection_id)?;
        self.publish_mutation(|| JournalMutation::DeleteCollection {
            collection_id: *collection_id,
        })?;
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        let started = std::time::Instant::now();
        let mut timings = SyncTimings::default();
        let pending_generation = self.mark_sync_pending_age(&mut timings);
        let result = self
            .sync_durability(true, &mut timings)
            .map(|()| self.persist_index_checkpoint_or_delta(&mut timings));
        timings.failed = result.is_err();
        self.finish_sync_pending_age(pending_generation);
        timings.total = started.elapsed();
        self.record_sync_diagnostics(&timings);
        if result.is_ok() {
            self.count_sync_persistence(&timings);
        }
        *self.last_sync_timings.lock() = Some(timings);
        result
    }

    fn refresh_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        Self::refresh_collection(self, collection_id)
    }
}

impl PackfileStorage {
    /// Persist the pre-batch generation after a batch failed after physical
    /// appends. The append-only frames remain as unreachable bytes, but the
    /// checkpoint's matching pack fingerprint makes that exclusion durable
    /// across a crash before the caller can run a later sync.
    fn persist_failed_batch_boundary(&self, error: StorageError) -> StorageError {
        let _persist_guard = self.index_persist_lock.lock();
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        match self
            .shards
            .sync_dirty()
            .map_err(StorageError::Io)
            .and_then(|()| self.persist_index_checkpoint())
        {
            Ok(()) => error,
            Err(boundary_error) => boundary_error,
        }
    }

    /// Sync all open shards to disk (full pool, not just dirty).
    ///
    /// Also persists the shard→collection directory as a side effect — an
    /// explicit sync is a natural point to flush this observability data too.
    /// The sidecar is written exactly once, pinned to the same pack
    /// fingerprint the index checkpoint/delta advance to inside
    /// `persist_index_checkpoint_or_delta`; see its owner comment for why the
    /// write lives there rather than here.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure.
    pub fn sync_all(&self) -> Result<(), StorageError> {
        let started = std::time::Instant::now();
        let mut timings = SyncTimings::default();
        let pending_generation = self.mark_sync_pending_age(&mut timings);
        let result = self
            .sync_durability(false, &mut timings)
            .map(|()| self.persist_index_checkpoint_or_delta(&mut timings));
        timings.failed = result.is_err();
        self.finish_sync_pending_age(pending_generation);
        timings.total = started.elapsed();
        self.record_sync_diagnostics(&timings);
        if result.is_ok() {
            self.count_sync_persistence(&timings);
        }
        *self.last_sync_timings.lock() = Some(timings);
        result
    }

    /// Rewrite the on-disk index checkpoint now, bypassing the delta-append
    /// fast path a normal sync takes whenever the delta log can continue.
    ///
    /// [`Self::sync_all`] appends a delta and does not advance the checkpoint's
    /// covered LSN, so the read-committed reader-reload path is reached only
    /// when a full checkpoint runs — a delta-log cap rollover, or this call.
    /// Tests and benchmarks use this to exercise that path deterministically;
    /// production callers normally rely on `sync_all`.
    ///
    /// This hook is for reload-path testing, not a production durability
    /// barrier. It fsyncs dirty shard frames before taking `index_persist_lock`
    /// (so the fsync never runs under the lock a concurrent sync holds across
    /// its own checkpoint — the same order as the normal sync path,
    /// `sync_durability` before `persist_index_checkpoint_or_delta`). A
    /// concurrent put can still append between that fsync and the snapshot
    /// `persist_index_checkpoint` takes under its collection locks.
    ///
    /// In the no-journal path, concurrent writes may be flushed but not fsynced
    /// before the checkpoint snapshot, and the fingerprint gate is not
    /// guaranteed to reject the result (a torn write that preserves the
    /// recorded length can still match), so do not treat this call as a
    /// durability boundary. With a journal enabled, `persist_index_checkpoint`
    /// fsyncs the shards under its collection locks and the journal remains
    /// authoritative. Marking the index dirty first makes the rewrite run even
    /// immediately after a delta sync; the flag stays set if the fsync fails,
    /// so a later call retries rather than dropping it.
    ///
    /// Intentionally public so the standalone `mtxdb-benches` crate can reach
    /// it; a cargo feature-gated benchmark API would give stronger separation,
    /// at the cost of a feature that crate would have to enable.
    ///
    /// # Errors
    /// Propagates any shard-fsync or checkpoint-write failure.
    pub fn force_index_checkpoint(&self) -> Result<(), StorageError> {
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.shards.sync_dirty().map_err(StorageError::Io)?;
        let _persist_guard = self.index_persist_lock.lock();
        self.persist_index_checkpoint()
    }

    /// Bound how often a structurally-needed full checkpoint rewrite may run.
    ///
    /// Both budgets are unlimited at zero, and the pair is *disabled* when both
    /// are zero (the default). A rewrite is deferred only while both
    /// configured budgets still have headroom; exhausting either forces it.
    ///
    /// Deferring is safe for durability: packfiles stay authoritative and are
    /// still synced first, so this only costs the next open a rescan — the
    /// stale on-disk checkpoint (and the delta log, whose tail no longer matches
    /// the advanced packs) is rejected by the fingerprint gates. It is the
    /// write-neutral fallback when no checkpoint base exists, a legacy v2 log
    /// needs rebasing, or a v3 append fails (for example, when its byte cap is
    /// exceeded); see `delta_state_needs_full_rewrite` and `checkpoint_skips`.
    pub fn set_checkpoint_rewrite_budget(&self, min_interval: std::time::Duration, max_bytes: u64) {
        self.checkpoint_rewrite_min_interval_ns.store(
            u64::try_from(min_interval.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.checkpoint_rewrite_max_bytes
            .store(max_bytes, Ordering::Relaxed);
    }

    /// Route durability through a write-ahead journal at `path`.
    ///
    /// After this, every mutation (`put`/`put_many`/`delete_collection`) is
    /// published to the journal with a monotonic LSN, and `sync` durably
    /// commits the pending group as one sequential fsync. Packfile shard
    /// fsyncs and the index checkpoint then become acceleration-only: a crash
    /// that loses them replays the journal instead of rescanning every pack.
    ///
    /// This is opt-in and off by default (the store's historical behavior —
    /// packfiles are the sync point — is unchanged until it is called).
    ///
    /// # Errors
    /// Returns `StorageError` if the journal segment cannot be opened or its
    /// committed prefix cannot be validated.
    pub fn enable_journal(&self, path: impl AsRef<std::path::Path>) -> Result<(), StorageError> {
        let (journal, scan) = Journal::open(path).map_err(StorageError::Io)?;
        self.journal_recovery.lock().clone_from(&scan.groups);
        *self.journal.lock() = Some(Arc::new(JournalCoordinator::new(journal, &scan)));
        Ok(())
    }

    /// Route durability through a journal whose group sequence is drawn from a
    /// shared, cross-pool counter.
    ///
    /// Identical to [`Self::enable_journal`] except the coordinator orders its
    /// groups by `sequence`, so several pools' segments share one global order
    /// (see [`JournalCoordinator::with_shared_sequence`]). The caller
    /// initializes `sequence` above the maximum recovered
    /// [`Journal::next_sequence`] across the participating segments.
    ///
    /// # Errors
    /// Same as [`Self::enable_journal`].
    pub fn enable_journal_with_sequence(
        &self,
        path: impl AsRef<std::path::Path>,
        sequence: Arc<AtomicU64>,
    ) -> Result<(), StorageError> {
        let (journal, scan) = Journal::open(path).map_err(StorageError::Io)?;
        self.journal_recovery.lock().clone_from(&scan.groups);
        *self.journal.lock() = Some(Arc::new(JournalCoordinator::with_shared_sequence(
            journal, &scan, sequence,
        )));
        Ok(())
    }

    /// Re-apply the recovered journal's post-checkpoint mutations after a
    /// reopen, then dirty the index so the next sync checkpoints them.
    ///
    /// Only entries with `lsn >` the sidecar's covered LSN are re-applied, so a
    /// checkpoint bounds the work. Re-applying a mutation that was already
    /// durable in a packfile (because its shard fsync happened to survive) is
    /// idempotent on the append-only store: it writes a duplicate frame and
    /// repoints the index, and a later repack drops the dead copy.
    ///
    /// Returns the number of mutations re-applied. No-op (0) when no journal is
    /// enabled or nothing is uncovered.
    ///
    /// # Errors
    /// Propagates any write failure from re-applying a mutation.
    pub fn replay_journal(&self) -> Result<u64, StorageError> {
        if self.journal().is_none() {
            return Ok(0);
        }
        let covered = Self::read_journal_lsn(&self.base_dir);
        let groups = self.journal_recovery.lock().clone();
        self.replaying.store(true, Ordering::SeqCst);
        let result = (|| -> Result<u64, StorageError> {
            let mut replayed = 0u64;
            for group in &groups {
                for entry in &group.entries {
                    if entry.lsn <= covered {
                        continue;
                    }
                    match &entry.mutation {
                        JournalMutation::Put {
                            collection_id,
                            node_id,
                            payload,
                        } => {
                            let data = NodeData::new(bytes::Bytes::from(payload.clone()));
                            self.put(collection_id, node_id, &data)?;
                        }
                        JournalMutation::DeleteCollection { collection_id } => {
                            self.delete_collection(collection_id)?;
                        }
                    }
                    replayed = replayed.saturating_add(1);
                }
            }
            Ok(replayed)
        })();
        self.replaying.store(false, Ordering::SeqCst);
        let replayed = result?;
        if replayed > 0 {
            self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }
        Ok(replayed)
    }

    /// The journal coordinator, if [`Self::enable_journal`] was called.
    #[must_use]
    pub fn journal(&self) -> Option<Arc<JournalCoordinator>> {
        self.journal.lock().clone()
    }

    /// Publish one mutation to the journal, if enabled. Returns the assigned
    /// LSN, or `None` when no journal is configured.
    ///
    /// `mutation` is a closure so its payload (a full record copy) is only
    /// built when a journal is actually configured.
    fn publish_mutation(
        &self,
        mutation: impl FnOnce() -> JournalMutation,
    ) -> Result<Option<u64>, StorageError> {
        let started = std::time::Instant::now();
        let Some(journal) = self.journal() else {
            self.record_published_mutation(started);
            return Ok(None);
        };
        // Replayed mutations are already durable in the journal; never
        // re-publish them.
        if self.replaying.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // The live index is already updated synchronously on the write path,
        // so the overlay callback has nothing to publish.
        let result = journal
            .publish(mutation(), |_lsn| {})
            .map(Some)
            .map_err(StorageError::Io);
        if result.is_ok() {
            self.record_published_mutation(started);
        }
        result
    }

    fn record_published_mutation(&self, started: std::time::Instant) {
        self.publish_calls.fetch_add(1, Ordering::Relaxed);
        self.publish_time_ns.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let mut pending = self.pending_publish_since.lock();
        if pending.is_none() {
            *pending = Some(std::time::Instant::now());
        }
        self.publish_generation.fetch_add(1, Ordering::Release);
    }

    /// Advance one barrier's durability boundary.
    ///
    /// With a journal enabled, that boundary is the pending WAL group: flush
    /// buffered frames to the page cache, then commit the group as one
    /// sequential fsync. Packfile shard fsyncs and the index checkpoint are
    /// then acceleration-only and recovered from the journal on reopen.
    /// Without a journal, behaves as before (fsync the dirty shards, or every
    /// shard for `sync_all`).
    fn sync_durability(
        &self,
        dirty_only: bool,
        timings: &mut SyncTimings,
    ) -> Result<(), StorageError> {
        let dirty_lock_before = self.shards.dirty_lock_wait();
        if let Some(journal) = self.journal() {
            let flush_started = std::time::Instant::now();
            self.shards.flush_all()?;
            timings.pack_flush = flush_started.elapsed();
            let target = journal.capture_sync_target();
            let wal_started = std::time::Instant::now();
            let (_, journal_timings) = journal
                .sync_through_timed(target)
                .map_err(StorageError::Io)?;
            timings.wal = wal_started.elapsed();
            timings.journal_lock_wait = journal_timings.journal_lock_wait;
            timings.journal_pending_wait = journal_timings.journal_pending_wait;
            timings.journal_append = journal_timings.journal_append;
            timings.journal_fsync = journal_timings.journal_fsync;
            timings.journal_sync_calls = 1;
            timings.journal_bytes = journal_timings.journal_bytes;
            timings.journal_records = journal_timings.journal_records;
            timings.journal_waiters = u64::from(journal_timings.journal_waiter);
            timings.journal_coalesced = u64::from(journal_timings.journal_coalesced);
            timings.journal_in_flight = journal_timings.journal_in_flight;
        } else if dirty_only {
            self.shards.sync_dirty()?;
            if let Some((flush, fsync)) = self.shards.last_sync_split() {
                timings.pack_flush = flush;
                timings.pack_fsync = fsync;
            }
        } else {
            self.shards.sync_all()?;
            if let Some((flush, fsync)) = self.shards.last_sync_split() {
                timings.pack_flush = flush;
                timings.pack_fsync = fsync;
            }
        }
        timings.dirty_lock_wait = self
            .shards
            .dirty_lock_wait()
            .saturating_sub(dirty_lock_before);
        Ok(())
    }

    /// Batch-granular sync accounting: every sync counts once, and the
    /// checkpoint-vs-delta discriminator comes from which phase
    /// `persist_index_checkpoint_or_delta` actually ran (a non-dirty sync runs
    /// neither). `sync` (dirty-scoped) and `sync_all` both funnel through here,
    /// so this is also the single point that folds each barrier's phase
    /// breakdown into the lifetime `sync_totals`.
    fn count_sync_persistence(&self, timings: &SyncTimings) {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        self.sync_totals.accumulate(timings);
        if !timings.checkpoint.is_zero() {
            self.checkpoint_writes.fetch_add(1, Ordering::Relaxed);
        } else if !timings.delta_log.is_zero() {
            self.delta_appends.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_sync_diagnostics(&self, timings: &SyncTimings) {
        let journal_path = self
            .journal()
            .map(|journal| journal.path().display().to_string());
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        self.sync_diagnostics.lock().record(SyncDiagnosticSample {
            timestamp_ms,
            process_id: std::process::id(),
            journal_path,
            total: timings.total,
            failed: timings.failed,
            pack_flush: timings.pack_flush,
            pack_fsync: timings.pack_fsync,
            sidecar: timings.sidecar,
            delta_log: timings.delta_log,
            checkpoint: timings.checkpoint,
            dirty_lock_wait: timings.dirty_lock_wait,
            pending_publish_age: timings.pending_publish_age,
            wal: timings.wal,
            journal_lock_wait: timings.journal_lock_wait,
            journal_pending_wait: timings.journal_pending_wait,
            journal_append: timings.journal_append,
            journal_fsync: timings.journal_fsync,
            journal_records: timings.journal_records,
            journal_bytes: timings.journal_bytes,
            journal_in_flight: timings.journal_in_flight,
            journal_waiters: timings.journal_waiters,
            journal_coalesced: timings.journal_coalesced,
        });
    }

    fn mark_sync_pending_age(&self, timings: &mut SyncTimings) -> u64 {
        let generation = self.publish_generation.load(Ordering::Acquire);
        if let Some(since) = *self.pending_publish_since.lock() {
            timings.pending_publish_age = since.elapsed();
        }
        generation
    }

    fn finish_sync_pending_age(&self, generation: u64) {
        if self.publish_generation.load(Ordering::Acquire) == generation {
            *self.pending_publish_since.lock() = None;
        }
    }
    /// `put_bytes + put_many_bytes` written since the last full checkpoint
    /// rewrite — the size dimension of the rewrite budget.
    fn written_bytes_since_last_rewrite(&self) -> u64 {
        self.put_bytes
            .load(Ordering::Relaxed)
            .saturating_add(self.put_many_bytes.load(Ordering::Relaxed))
            .saturating_sub(
                self.checkpoint_bytes_at_last_rewrite
                    .load(Ordering::Relaxed),
            )
    }

    /// Whether a structurally-needed full checkpoint rewrite should be deferred
    /// under the configured time/size budget.
    ///
    /// Both budgets are unlimited at zero. A zero/zero budget (the default)
    /// never defers, preserving the historical behavior of rewriting on every
    /// invalidated dirty barrier. With one budget configured, deferral is gated
    /// on that one alone; with both, a rewrite runs as soon as either headroom
    /// runs out.
    fn should_defer_checkpoint_rewrite(&self) -> bool {
        let interval = std::time::Duration::from_nanos(
            self.checkpoint_rewrite_min_interval_ns
                .load(Ordering::Relaxed),
        );
        let max_bytes = self.checkpoint_rewrite_max_bytes.load(Ordering::Relaxed);
        if interval.is_zero() && max_bytes == 0 {
            return false;
        }
        let time_headroom = interval.is_zero()
            || self
                .last_checkpoint_rewrite_at
                .lock()
                .as_ref()
                .is_some_and(|last| last.elapsed() < interval);
        let size_headroom = max_bytes == 0 || self.written_bytes_since_last_rewrite() < max_bytes;
        time_headroom && size_headroom
    }

    /// Record that a full checkpoint rewrite just completed, resetting both
    /// budget baselines to now and the current write counter.
    fn note_checkpoint_rewrite(&self) {
        *self.last_checkpoint_rewrite_at.lock() = Some(std::time::Instant::now());
        self.checkpoint_bytes_at_last_rewrite.store(
            self.put_bytes
                .load(Ordering::Relaxed)
                .saturating_add(self.put_many_bytes.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }

    /// Whether pending state needs a full checkpoint rather than a v3 delta
    /// append. Missing bases, legacy v2 epochs, or no pending operations need
    /// a checkpoint; oversized v3 batches fall back from the append helper.
    fn delta_state_needs_full_rewrite(&self) -> bool {
        let state = self.delta_state.lock();
        if state.base_fingerprint.is_none() || state.log_version != 3 || state.pending.is_empty() {
            return true;
        }
        false
    }

    /// Persist the dirty index state for a sync barrier — a delta append when
    /// the log can be continued, otherwise a full checkpoint rewrite — and
    /// record which path ran in `timings`. No-op when nothing is dirty.
    ///
    /// The shard→collection inspection sidecar is owned here, written exactly
    /// once per dirty barrier whether the index persisted as a delta append or
    /// a full rewrite: it must stay gated to the same pack set the checkpoint
    /// just became, so the next open can rebuild per-shard counts from records
    /// instead of walking every slot. A clean barrier writes nothing (the
    /// sidecar can only have gone stale together with the index — the same
    /// mutations that move a record into a shard also dirty the index). A
    /// failure here leaves the previous directory stale; the fingerprint gate
    /// then falls back to the slot walk until the next rewrite.
    fn persist_index_checkpoint_or_delta(&self, timings: &mut SyncTimings) {
        let _persist_guard = self.index_persist_lock.lock();
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            // Nothing to checkpoint, but a sidecar lost while the checkpoint
            // survived still has to be regenerated.
            if self.shard_collections_dirty.load(Ordering::Relaxed) {
                self.persist_shard_collections_best_effort();
            }
            return;
        }
        if self.delta_state_needs_full_rewrite() {
            if self.should_defer_checkpoint_rewrite() {
                // Write-neutral stopgap: the caller already synced the
                // packfiles, so skipping the acceleration rewrite costs only
                // the next open a rescan — the stale on-disk checkpoint no
                // longer matches the advanced pack fingerprint, and the delta
                // log's tail does not either, so the opener rejects both. Leave
                // `index_checkpoint_dirty` set so a later barrier past the
                // budget still rewrites.
                self.checkpoint_skips.fetch_add(1, Ordering::Relaxed);
                return;
            }
            let checkpoint_started = std::time::Instant::now();
            self.persist_index_checkpoint_best_effort();
            self.note_checkpoint_rewrite();
            timings.checkpoint = checkpoint_started.elapsed();
        } else {
            let delta_started = std::time::Instant::now();
            if let Err(error) = self.append_index_delta() {
                // An append failure took the frames with it, so the pending state
                // no longer reflects the live indexes. Fall back to a full
                // rewrite rather than leaving the acceleration files stale until
                // the next sync notices the gap.
                eprintln!("mtxdb: delta log append failed, rewriting checkpoint: {error}");
                let checkpoint_started = std::time::Instant::now();
                self.persist_index_checkpoint_best_effort();
                self.note_checkpoint_rewrite();
                timings.checkpoint = checkpoint_started.elapsed();
            } else {
                timings.delta_log = delta_started.elapsed();
            }
        }
        let sidecar_started = std::time::Instant::now();
        self.persist_shard_collections_best_effort();
        timings.sidecar = sidecar_started.elapsed();
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
    /// use mtxdb::shard::AppendPolicy;
    /// use mtxdb::PackfileStorage;
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
    pub fn shard_stats_for(&self, slot: u16) -> Option<crate::shard::ShardStats> {
        self.shards.stats(slot)
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

    /// Set whether [`Self::get_many_with_refresh`] refreshes a collection's
    /// index when the caller's in-memory snapshot misses. Defaults to `true`
    /// (multi-process readers). A single writer -- whose in-memory index is
    /// authoritative for everything it has written -- sets this `false` so a
    /// negative lookup degrades to a plain [`StorageEngine::get_many`] and
    /// never touches the refresh lock or the durable fingerprint. See the
    /// `refresh_on_miss` field docs.
    pub fn set_refresh_on_miss(&self, enabled: bool) {
        self.refresh_on_miss.store(enabled, Ordering::Relaxed);
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
        // Longest linear-probe chain observed across every live collection's
        // index, for operator visibility into collision-chain growth (see
        // `LossyIndex::max_probe_len` — pure observability, no cap, no
        // effect on control flow).
        let max_index_probe_len = self
            .collections_read()
            .values()
            .map(|generation| generation.load().index.max_probe_len())
            .max()
            .unwrap_or(0);
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
            miss_refreshes: self.miss_refreshes.load(Ordering::Relaxed),
            miss_refresh_skips: self.miss_refresh_skips.load(Ordering::Relaxed),
            miss_refresh_recovered: self.miss_refresh_recovered.load(Ordering::Relaxed),
            miss_refresh_retry_ids: self.miss_refresh_retry_ids.load(Ordering::Relaxed),
            miss_refresh_fp_errors: self.miss_refresh_fp_errors.load(Ordering::Relaxed),
            index_candidates: self.index_candidates.load(Ordering::Relaxed),
            candidate_reads: self.candidate_reads.load(Ordering::Relaxed),
            candidate_hash_mismatches: self.candidate_hash_mismatches.load(Ordering::Relaxed),
            get_many_shards_touched: self.get_many_shards_touched.load(Ordering::Relaxed),
            candidate_frame_bytes: self.candidate_frame_bytes.load(Ordering::Relaxed),
            read_many_runs: self.read_many_runs.load(Ordering::Relaxed),
            read_many_span_bytes: self.read_many_span_bytes.load(Ordering::Relaxed),
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
            checkpoint_skips: self.checkpoint_skips.load(Ordering::Relaxed),
            delta_appends: self.delta_appends.load(Ordering::Relaxed),
            read_reloads: self.read_reloads.load(Ordering::Relaxed),
            read_reload_failures: self.read_reload_failures.load(Ordering::Relaxed),
            sidecar_writes: self.sidecar_writes.load(Ordering::Relaxed),
            sync_calls: self.sync_calls.load(Ordering::Relaxed),
            last_open_timings: self.open_timings(),
            last_sync_timings: self.sync_timings(),
            sync_totals: self.sync_totals.snapshot(),
            sync_diagnostics: self.sync_diagnostics.lock().snapshot(),
            publish_calls: self.publish_calls.load(Ordering::Relaxed),
            publish_time: std::time::Duration::from_nanos(
                self.publish_time_ns.load(Ordering::Relaxed),
            ),
            repack: self.repack_stats(),
            cache,
            shards: self.shard_stats(),
            index_bytes,
            collection_count: summaries.len(),
            max_index_probe_len,
            dirty_lock_wait: self.shards.dirty_lock_wait(),
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
    /// breakdowns and the lifetime `sync_totals` accumulator, and leaves the
    /// `stats_enabled` flag untouched.
    pub fn reset_stats(&self) {
        for counter in [
            &self.get_calls,
            &self.get_misses,
            &self.get_many_calls,
            &self.get_many_records,
            &self.get_many_misses,
            &self.miss_refreshes,
            &self.miss_refresh_skips,
            &self.miss_refresh_recovered,
            &self.miss_refresh_retry_ids,
            &self.miss_refresh_fp_errors,
            &self.index_candidates,
            &self.candidate_reads,
            &self.candidate_hash_mismatches,
            &self.get_many_shards_touched,
            &self.candidate_frame_bytes,
            &self.read_many_runs,
            &self.read_many_span_bytes,
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
            &self.checkpoint_skips,
            &self.delta_appends,
            &self.read_reloads,
            &self.read_reload_failures,
            &self.sidecar_writes,
            &self.sync_calls,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        *self.last_open_timings.lock() = None;
        *self.last_sync_timings.lock() = None;
        self.sync_totals.reset();
        self.sync_diagnostics.lock().reset();
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
    /// Read-miss refreshes that rescanned a collection.
    pub miss_refreshes: u64,
    /// Read-miss refreshes skipped because a recent refresh already occurred.
    pub miss_refresh_skips: u64,
    /// Missing keys recovered after a read-miss refresh.
    pub miss_refresh_recovered: u64,
    /// IDs submitted in retry requests after read-miss refreshes.
    pub miss_refresh_retry_ids: u64,
    /// Post-refresh fingerprint read errors (rate-limits warnings).
    pub miss_refresh_fp_errors: u64,
    /// Candidate locations yielded by the lossy index.
    pub index_candidates: u64,
    /// Candidate locations actually read from packfiles.
    pub candidate_reads: u64,
    /// Candidate records rejected after full-hash verification.
    pub candidate_hash_mismatches: u64,
    /// Sum of unique shards touched by each `get_many` batch.
    pub get_many_shards_touched: u64,
    /// Sum of on-disk frame lengths touched by candidate-record reads
    /// (stats-gated; the candidate-resolve path only, not refresh rescans).
    /// A logical extent estimate, not a physical disk-byte count — mmap may
    /// fetch whole pages, readahead ranges, or merged extents.
    pub candidate_frame_bytes: u64,
    /// Estimated/logical offset runs a `get_many` batch collapses candidate
    /// reads into (stats-gated; gap heuristic per `READ_RUN_GAP_BYTES`, not
    /// measured physical sequential reads).
    pub read_many_runs: u64,
    /// Per-shard `last − first` candidate-offset fan-in summed across a
    /// `get_many` batch (stats-gated; the logical byte-extent the batch
    /// scatters over, not physical disk bytes read).
    pub read_many_span_bytes: u64,
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
    /// Collection structural changes that promote pending slots to snapshot or
    /// tombstone operations; this does not imply a checkpoint rewrite.
    pub delta_invalidations: u64,
    /// Syncs that rewrote the index checkpoint in full.
    pub checkpoint_writes: u64,
    /// Syncs that deferred a structurally-needed full checkpoint rewrite under
    /// the configured time/size budget (see
    /// [`PackfileStorage::set_checkpoint_rewrite_budget`]). Non-zero only when
    /// a budget is configured.
    pub checkpoint_skips: u64,
    /// Syncs that appended the incremental delta log instead.
    pub delta_appends: u64,
    /// Successful checkpoint-bound index reloads by the read-committed overlay
    /// (a writer's reclaim outran this handle's incorporated coverage).
    pub read_reloads: u64,
    /// Read-committed overlay reload attempts that failed to load a checkpoint
    /// matching the current packs, so the read failed closed.
    pub read_reload_failures: u64,
    /// Writes of the shard→collection inspection sidecar (every one counts,
    /// whichever caller triggered it).
    pub sidecar_writes: u64,
    /// `sync`/`sync_all` calls.
    pub sync_calls: u64,
    /// Per-phase breakdown of the most recent open.
    pub last_open_timings: Option<OpenTimings>,
    /// Per-phase breakdown of the most recent sync.
    pub last_sync_timings: Option<SyncTimings>,
    /// Lifetime per-phase sync totals — the cumulative counterpart to
    /// `last_sync_timings`, and the only view that can answer a run-wide
    /// scatter-vs-rewrite split (the most-recent breakdown is one barrier).
    pub sync_totals: SyncTotalsSnapshot,
    /// Bounded worst-operation samples and latency histograms for the current
    /// process. Unlike cumulative totals, these preserve tail behavior.
    pub sync_diagnostics: SyncDiagnosticsSnapshot,
    /// Successful mutation publication calls and their cumulative time.
    pub publish_calls: u64,
    /// Cumulative time spent publishing mutations.
    pub publish_time: std::time::Duration,
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
    /// Longest linear-probe chain observed by any insert or lookup across
    /// every live collection's index, since each collection's index was
    /// constructed (never reset by `reset_stats`, matching `shards`/`cache`/
    /// `index_bytes`/`repack` — see `LossyIndex::max_probe_len`). Pure
    /// observability: no probe-length cap exists or is implied by this
    /// value; it exists so an operator can notice collision-chain growth
    /// (e.g. under adversarial content against a misconfigured/unseeded
    /// deployment) without any change in behavior.
    pub max_index_probe_len: u32,
    /// Cumulative wall-clock time any `sync`/`sync_all` caller has spent
    /// waiting to acquire the shard pool's dirty-set lock (never reset —
    /// see `ShardPool::dirty_lock_wait`). Exists to answer, before building
    /// a finer-grained per-shard sync coalescing mechanism, whether the
    /// current coarse lock is actually a measurable contention point under
    /// real concurrent-writer load: compare this against total sync wall
    /// time (`last_sync_timings`) to judge whether it's worth pursuing.
    pub dirty_lock_wait: std::time::Duration,
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
            miss_refreshes: 0,
            miss_refresh_skips: 0,
            miss_refresh_recovered: 0,
            miss_refresh_retry_ids: 0,
            miss_refresh_fp_errors: 0,
            index_candidates: 0,
            candidate_reads: 0,
            candidate_hash_mismatches: 0,
            get_many_shards_touched: 0,
            candidate_frame_bytes: 0,
            read_many_runs: 0,
            read_many_span_bytes: 0,
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
            checkpoint_skips: 0,
            delta_appends: 0,
            read_reloads: 0,
            read_reload_failures: 0,
            sidecar_writes: 0,
            sync_calls: 0,
            last_open_timings: None,
            last_sync_timings: None,
            sync_totals: SyncTotalsSnapshot::default(),
            sync_diagnostics: SyncDiagnosticsSnapshot::default(),
            publish_calls: 0,
            publish_time: std::time::Duration::ZERO,
            repack: RepackStats::default(),
            cache: CacheStats::default(),
            shards: Vec::new(),
            index_bytes: 0,
            collection_count: 0,
            max_index_probe_len: 0,
            dirty_lock_wait: std::time::Duration::ZERO,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only hook run by `get`/`get_many` after candidate shard ids are
    /// collected but before they are pinned, allowing deterministic repack
    /// interleavings in the read-path race tests.
    static TEST_BEFORE_PIN: std::cell::RefCell<Option<Box<dyn Fn()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
impl PackfileStorage {
    fn set_test_before_pin(hook: Option<Box<dyn Fn()>>) {
        TEST_BEFORE_PIN.with(|slot| *slot.borrow_mut() = hook);
    }

    fn run_test_before_pin() {
        TEST_BEFORE_PIN.with(|slot| {
            if let Some(hook) = slot.borrow().as_ref() {
                hook();
            }
        });
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::time::Duration;

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

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_sees_a_committed_but_unflushed_group() {
        let dir = test_dir("read_committed_unflushed");
        let wal = dir.join("wal.bin");
        let collection = [0x42u8; 16];
        let node = [0x07u8; 16];

        // A read-only open needs at least one shard on disk, so seed a
        // durable record first.
        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x99u8; 16],
            &[0x99u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        // A complete group (trailer present) with no fsync: committed, but
        // invisible to the durable fingerprint gate.
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"committed-unflushed".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
        assert_eq!(
            read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"committed-unflushed"[..])
        );

        // The durable API still hides the unflushed group.
        let durable = store.get_many_with_refresh(&collection, &[node]).unwrap();
        assert!(
            durable[0].is_none(),
            "durable read must not observe an unflushed journal group"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_shadows_durable_with_a_committed_delete() {
        let dir = test_dir("read_committed_delete");
        let wal = dir.join("wal.bin");
        let collection = [0x43u8; 16];
        let node = [0x08u8; 16];

        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"live")),
            )
            .unwrap();
        writer.sync().unwrap();
        drop(writer);

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::DeleteCollection {
                collection_id: collection,
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
        assert!(
            read_committed[0].is_none(),
            "a committed delete must shadow the durable record"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_preserves_a_delete_boundary_across_recreate() {
        let dir = test_dir("read_committed_recreate");
        let wal = dir.join("wal.bin");
        let collection = [0x45u8; 16];
        let old = [0x0au8; 16];
        let new = [0x0bu8; 16];

        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer
            .put(
                &collection,
                &old,
                &NodeData::new(bytes::Bytes::from_static(b"old")),
            )
            .unwrap();
        writer.sync().unwrap();
        drop(writer);

        // Delete the collection, then recreate it with a different record. The
        // pre-delete record must not be resurrected by the durable fallback.
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::DeleteCollection {
                collection_id: collection,
            }])
            .unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: new,
                payload: b"new".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        let read_committed = store.get_read_committed(&collection, &[old, new]).unwrap();
        assert!(
            read_committed[0].is_none(),
            "a pre-delete record must stay deleted after the collection is recreated"
        );
        assert_eq!(
            read_committed[1].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"new"[..])
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_ignores_entries_the_checkpoint_covers() {
        let dir = test_dir("read_committed_covered");
        let wal = dir.join("wal.bin");
        let collection = [0x44u8; 16];
        let node = [0x09u8; 16];

        // A real writer with a journal: its sync checkpoints the put and embeds
        // the covered LSN atomically, so the node is both durable and covered
        // rather than the test merely claiming coverage the index never had.
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"covered")),
            )
            .unwrap();
        writer.sync().unwrap();
        drop(writer);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        {
            let overlay = store.read_journal.lock();
            let overlay = overlay.as_ref().expect("journal overlay enabled");
            assert!(
                overlay.puts.is_empty(),
                "covered puts must not stay in the overlay"
            );
            assert!(
                overlay.delete_lsn.is_empty(),
                "covered deletes must not stay in the overlay"
            );
        }
        let read_committed = store.get_read_committed(&collection, &[node]).unwrap();
        assert_eq!(
            read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"covered"[..]),
            "an entry the checkpoint covers must be served from the durable index"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_keeps_entries_a_newer_checkpoint_covers() {
        let dir = test_dir("read_committed_stale_index");
        let wal = dir.join("wal.bin");
        let collection = [0x47u8; 16];
        let node = [0x0eu8; 16];

        // Durable value V0, loaded by the reader's index at open.
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"v0")),
            )
            .unwrap();
        writer.sync().unwrap();
        drop(writer);

        // V1 committed to the journal but not checkpointed into the reader's
        // already-loaded index.
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"v1".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        assert_eq!(
            store.get_read_committed(&collection, &[node]).unwrap()[0]
                .as_ref()
                .map(|data| data.bytes.as_ref()),
            Some(&b"v1"[..])
        );

        // A concurrent checkpoint advances coverage past V1's LSN, but this
        // handle's in-memory index still holds only V0. Pruning against that
        // detached coverage would drop V1 from the overlay and resurrect the
        // stale V0 from the index.
        store.write_journal_lsn(1).unwrap();
        assert_eq!(
            store.get_read_committed(&collection, &[node]).unwrap()[0]
                .as_ref()
                .map(|data| data.bytes.as_ref()),
            Some(&b"v1"[..]),
            "a checkpoint this handle has not loaded must not evict overlay entries"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_fails_closed_when_reclaim_skips_its_covered_lsn() {
        let dir = test_dir("read_committed_reclaim_gap");
        let wal = dir.join("wal.bin");
        let collection = [0x48u8; 16];
        let node = [0x0fu8; 16];

        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x95u8; 16],
            &[0x95u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"first".to_vec(),
            }])
            .unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: [0x10u8; 16],
                payload: b"second".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        // Reclaim drops LSN 1, so the segment now begins at LSN 2 while the
        // reader's index only incorporated coverage 0. Those records are in
        // neither source, so the read must fail closed rather than serve a gap.
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal.reclaim_through(1).unwrap();
        drop(journal);

        assert!(
            store.get_read_committed(&collection, &[node]).is_err(),
            "a reclaimed segment base beyond the reader's covered LSN must error"
        );
    }

    /// A genuine coverage gap that reloads cannot close — the checkpoint
    /// exists and matches, but never covers the reclaimed LSNs — is a
    /// persistent corruption, not a transient condition.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_gap_with_matching_checkpoint_is_corrupt() {
        let dir = test_dir("read_committed_gap_corrupt");
        let wal = dir.join("wal.bin");
        let collection = [0x4Cu8; 16];
        let node = [0x0fu8; 16];

        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x96u8; 16],
            &[0x96u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"first".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal.reclaim_through(1).unwrap();
        drop(journal);

        let error = store
            .get_read_committed(&collection, &[node])
            .expect_err("the unrecoverable gap must fail");
        assert!(
            matches!(error, StorageError::Corrupt(_)),
            "a matching checkpoint that never covers the gap is corrupt, got {error:?}"
        );
    }

    /// The same gap, but with no usable checkpoint to reload: every reload
    /// attempt fails, so the read is retryable once the writer publishes a
    /// checkpoint instead of being reported as permanent corruption.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_gap_without_checkpoint_is_retryable() {
        let dir = test_dir("read_committed_gap_retry");
        let wal = dir.join("wal.bin");
        let collection = [0x4Du8; 16];
        let node = [0x0fu8; 16];

        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x97u8; 16],
            &[0x97u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"first".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        // Remove the checkpoint the reader would reload, so the reload path
        // fails closed but retryably.
        fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal.reclaim_through(1).unwrap();
        drop(journal);

        let error = store
            .get_read_committed(&collection, &[node])
            .expect_err("the unrecoverable gap must fail");
        assert!(
            matches!(
                error,
                StorageError::Io(ref io) if io.kind() == std::io::ErrorKind::WouldBlock
            ),
            "a missing checkpoint must be reported as retryable, got {error:?}"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_reloads_after_reclaim_advances_coverage() {
        let dir = test_dir("read_committed_reclaim_reload");
        let wal = dir.join("wal.bin");
        let collection = [0x49u8; 16];
        let first = [0x11u8; 16];
        let second = [0x12u8; 16];

        // Durable seed, then an empty journal segment the reader can attach to.
        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x94u8; 16],
            &[0x94u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);
        let (journal, _) = Journal::open(&wal).unwrap();
        drop(journal);

        // Reader attaches with the pre-advance index (coverage 0).
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        // Remove the seed checkpoint so the writer's sync must write a full
        // checkpoint instead of appending to its existing delta log. A delta
        // append extends the loaded index but does not advance checkpoint
        // coverage; this test specifically needs a checkpoint that covers LSN 1.
        fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();

        // A real writer checkpoints `first` and reclaims through its LSN, so the
        // checkpoint embeds coverage 1 and its index contains `first`.
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();
        writer
            .put(
                &collection,
                &first,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();

        let checkpoint = crate::index::checkpoint::read_checkpoint(
            &PackfileStorage::index_checkpoint_path(&dir),
        )
        .unwrap();
        assert_eq!(checkpoint.covered_lsn, 1);
        let after_checkpoint = Journal::scan_read_only(&wal).unwrap();
        assert!(after_checkpoint.base_lsn > checkpoint.covered_lsn);

        // Append a new record after the checkpoint, then sync the writer. The
        // sync commits its journal group and extends the checkpoint's delta
        // log without rewriting the checkpoint. Reload must accept that delta
        // tail despite the live pack having grown.
        writer
            .put(
                &collection,
                &second,
                &NodeData::new(bytes::Bytes::from_static(b"second")),
            )
            .unwrap();
        writer.sync().unwrap();

        let read_committed = store
            .get_read_committed(&collection, &[first, second])
            .unwrap();
        assert_eq!(store.read_covered_lsn.load(Ordering::Acquire), 1);
        assert!(
            store.stats().read_reloads >= 1,
            "the reclaim must have driven at least one checkpoint-bound reload"
        );
        assert_eq!(
            read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"first"[..]),
            "the reloaded checkpoint index must serve the covered record, not a hole"
        );
        assert_eq!(
            read_committed[1].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"second"[..]),
            "the post-checkpoint group must be served from the overlay"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_reload_skips_the_exact_pack_gate() {
        let dir = test_dir("read_committed_reload_skip_gate");
        let wal = dir.join("wal.bin");
        let collection = [0x4Bu8; 16];
        let first = [0x31u8; 16];
        let second = [0x32u8; 16];
        let third = [0x33u8; 16];

        // Durable seed, then an empty segment the reader can attach to.
        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x97u8; 16],
            &[0x97u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);
        let (journal, _) = Journal::open(&wal).unwrap();
        drop(journal);

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        // Force the writer's next sync to write a full checkpoint so coverage
        // advances and the segment is reclaimed through it.
        fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();
        writer
            .put(
                &collection,
                &first,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();
        let checkpoint = crate::index::checkpoint::read_checkpoint(
            &PackfileStorage::index_checkpoint_path(&dir),
        )
        .unwrap();
        assert_eq!(checkpoint.covered_lsn, 1);

        // A synced second record extends the delta log.
        writer
            .put(
                &collection,
                &second,
                &NodeData::new(bytes::Bytes::from_static(b"second")),
            )
            .unwrap();
        writer.sync().unwrap();

        // A third record is appended (eagerly) but never synced, so the live
        // packs grow past the delta log's tail. An exact-pack gate would reject
        // the checkpoint here and fail the read closed; the read-journal reload
        // must skip that gate and let the overlay supply the suffix.
        writer
            .put(
                &collection,
                &third,
                &NodeData::new(bytes::Bytes::from_static(b"third")),
            )
            .unwrap();

        // The relaxation is scoped to the read-journal reload: a normal open
        // runs `ReloadMode::Strict` and must still reject the checkpoint the
        // eager append has outgrown, falling back to a full rescan rather than
        // trusting a stale fingerprint.
        let strict_reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
        assert_eq!(
            strict_reader.open_timings().unwrap().path,
            OpenPath::FullScan,
            "a strict open must not accept a checkpoint the live packs have outgrown"
        );

        let read_committed = store
            .get_read_committed(&collection, &[first, second, third])
            .unwrap();
        assert!(
            store.stats().read_reloads >= 1,
            "the reclaim must have driven the read-journal reload path"
        );
        for (value, expected) in read_committed
            .iter()
            .zip([&b"first"[..], b"second", b"third"])
        {
            assert_eq!(
                value.as_ref().map(|data| data.bytes.as_ref()),
                Some(expected),
                "every committed record must be visible after a gated reload"
            );
        }
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn repack_persists_checkpoint_so_read_journal_reload_survives_retirement() {
        // Repack moves a collection's records into a fresh destination shard
        // and retires the old one once nothing else references it. A reader
        // reloading from the checkpoint afterward must see the new shard
        // layout — otherwise it reloads a checkpoint whose index still names
        // a shard repack already unlinked, and the read fails closed.
        let dir = test_dir("repack_persists_checkpoint");
        let wal = dir.join("wal.bin");
        let collection = [0x51u8; 16];
        let node = [0x61u8; 16];

        // `collection` is the only collection in this store, so its shard is
        // never shared with anything else: repack's retirement of the
        // now-empty source shard actually unlinks the file, rather than
        // leaving it alive because another collection still references it.
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();
        let before = crate::index::checkpoint::read_checkpoint(
            &PackfileStorage::index_checkpoint_path(&dir),
        )
        .unwrap();
        assert_eq!(before.covered_lsn, 1);

        // Repack `collection`: it moves `node` into a fresh non-source shard
        // and, since no other collection references the old one, retires
        // (unlinks) it.
        writer
            .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
            .unwrap();

        let after = crate::index::checkpoint::read_checkpoint(
            &PackfileStorage::index_checkpoint_path(&dir),
        )
        .unwrap();
        assert_eq!(
            after.covered_lsn, before.covered_lsn,
            "repack must carry the checkpoint's covered_lsn forward unchanged, not reset it"
        );

        // Open a reader only now, after retirement already unlinked the
        // source shard: it must never hold an open handle to the retired
        // file, or the read would trivially keep succeeding through the
        // stale-but-still-open fd (Linux keeps an unlinked file's bytes
        // readable through any fd opened before the unlink) and the test
        // would not actually exercise the checkpoint staleness at all.
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();

        // Drive the checkpoint reload directly rather than through
        // get_read_committed: the journal overlay can serve a still-covered
        // record straight from its segment without ever touching the index,
        // which would leave this test unable to distinguish a correct reload
        // from a mismatched one. reload_index_from_checkpoint is what
        // discovers the live shard set and must succeed against the
        // checkpoint repack just wrote.
        assert!(
            store.reload_index_from_checkpoint(),
            "reload must succeed against the checkpoint repack persisted, not a stale one \
             naming a shard repack already unlinked"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn repack_persisted_checkpoint_reload_resolves_record_by_post_repack_offset() {
        let dir = test_dir("repack_persists_checkpoint_read");
        let wal = dir.join("wal.bin");
        let collection = [0x51u8; 16];
        let node = [0x61u8; 16];

        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();
        writer
            .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
            .unwrap();

        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        assert!(store.reload_index_from_checkpoint());

        let got = store.get(&collection, &node).unwrap();
        assert_eq!(
            got.map(|data| data.bytes.to_vec()),
            Some(b"first".to_vec()),
            "the reloaded index must resolve the record through its post-repack shard offset"
        );
    }

    /// Coverage for a post-repack delta epoch over *existing* pack identities:
    /// a delta written after repack's fresh checkpoint must replay for a cold
    /// reader and resolve through the checkpoint's v6 pack table, not the
    /// reader's own shard numbering.
    ///
    /// `repack_collection_reachable` persists a checkpoint (C1) naming the
    /// post-retirement pack set, so the pre-retirement delta epoch is retired
    /// with it. A later `sync_all` appends a fresh epoch continuing C1; this
    /// test pins that a cold open *replays* that epoch (nonzero `delta_replay`,
    /// not a checkpoint rewrite that already included `third`) and returns all
    /// three records.
    ///
    /// `third` must be durably synced, not merely shard-flushed: only a sync
    /// persists the index. An uncommitted record is invisible to checkpoint
    /// replay and recoverable only by a fallback full scan — a different
    /// property, and the one the previous version of this test accidentally
    /// measured.
    ///
    /// Out of scope: a delta frame referencing a pack created *after* C1 (a
    /// pack absent from C1's pack table), and any claim that a delta-side
    /// `SlotBinding` is unnecessary.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn post_repack_delta_epoch_replays_for_a_cold_reader() {
        let dir = test_dir("delta_across_retirement");
        let wal = dir.join("wal.bin");
        let collection = [0x71u8; 16];
        let first = [0x81u8; 16];
        let second = [0x82u8; 16];
        let third = [0x83u8; 16];

        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer.enable_journal(&wal).unwrap();

        // Commit enough that a checkpoint lands and rotates the delta epoch.
        writer
            .put(
                &collection,
                &first,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();

        // A post-checkpoint write leaves a delta epoch whose base is the
        // current checkpoint and whose tail names the current pack set.
        writer
            .put(
                &collection,
                &second,
                &NodeData::new(bytes::Bytes::from_static(b"second")),
            )
            .unwrap();
        writer.sync_all().unwrap();

        // Repack moves the collection's records into a fresh destination shard
        // and unlinks the source. The writer's slot table now has a hole, so a
        // fresh reader's `discover_shards` will not reproduce the writer's
        // numbering. Repack persists a checkpoint for the new layout.
        writer
            .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
            .unwrap();

        // Commit `third` into the delta epoch continuing the post-repack
        // checkpoint. A shard flush alone would not persist the index.
        writer
            .put(
                &collection,
                &third,
                &NodeData::new(bytes::Bytes::from_static(b"third")),
            )
            .unwrap();
        writer.sync_all().unwrap();
        drop(writer);

        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
        reader.enable_read_journal(&wal).unwrap();

        let timings = reader
            .stats()
            .last_open_timings
            .expect("open must record timings");
        // The post-repack checkpoint and its delta epoch must both validate, so
        // the cold open takes the checkpoint fast path rather than a full scan.
        assert_eq!(
            timings.path,
            OpenPath::Checkpoint,
            "the post-repack checkpoint and the epoch continuing it must be usable"
        );
        // Prove the records below come from a gated delta replay, not from a
        // checkpoint rewrite that already included `third`. This is a
        // deterministic count, unlike the `delta_replay` duration.
        assert!(
            timings.delta_replay_operations > 0,
            "cold open must have replayed the post-repack delta epoch: {timings:?}"
        );
        assert_eq!(
            timings.full_scan,
            Duration::ZERO,
            "cold open must not have fallen back to a full scan"
        );

        // All three records must resolve to their exact bytes through the
        // replayed delta — never to a wrong shard's bytes.
        for (id, expected) in [
            (first, &b"first"[..]),
            (second, &b"second"[..]),
            (third, &b"third"[..]),
        ] {
            let data = reader
                .get(&collection, &id)
                .unwrap()
                .expect("record must exist");
            assert_eq!(data.bytes.as_ref(), expected);
        }
    }

    #[test]
    fn repack_persists_checkpoint_so_fresh_cold_open_survives_retirement() {
        // Same failure shape as the read-journal test above, but through the
        // *plain* open() path and no journal at all. After repack, the
        // checkpoint's pack set is exactly the live pack set (one pack), so
        // a fresh open's fingerprint matches exactly and the exact-pack gate
        // takes no action (replay_needed is false) — the checkpoint's raw
        // index slots are trusted directly. A brand-new reader's own
        // discover_shards, seeing only that one surviving pack from an
        // empty slot table, assigns it local slot 0; the checkpoint's index
        // still names the writer's slot 1 (assigned before the original
        // slot-0 shard was retired). Without the pack-table remap, get()
        // resolves against the wrong (nonexistent, for this reader) slot.
        let dir = test_dir("repack_persists_checkpoint_cold_open");
        let collection = [0x52u8; 16];
        let node = [0x62u8; 16];

        let writer = PackfileStorage::open(dir.clone()).unwrap();
        writer
            .put(
                &collection,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        writer.sync().unwrap();

        writer
            .repack_collection_reachable(&collection, |_hash, _data| Vec::new())
            .unwrap();
        drop(writer);

        // Fresh process-equivalent open: no history, no shard ever open
        // before this point, so this reader's own discover_shards renumbers
        // from scratch.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let got = reopened.get(&collection, &node).unwrap();
        assert_eq!(
            got.map(|data| data.bytes.to_vec()),
            Some(b"first".to_vec()),
            "a fresh cold open after repack must resolve the record through \
             the checkpoint's pack table, not the writer's raw (unstable) slot number"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn open_read_committed_serves_the_overlay_in_one_call() {
        let dir = test_dir("open_read_committed");
        let wal = dir.join("wal.bin");
        let collection = [0x4Au8; 16];
        let node = [0x21u8; 16];

        // A shard must exist before a read-only open will succeed.
        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x96u8; 16],
            &[0x96u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        // A committed group in the writer's segment that was never synced into
        // the packs: the durable API cannot see it, only the overlay can.
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: node,
                payload: b"committed".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let plain = PackfileStorage::open_read_only(dir.clone()).unwrap();
        assert!(
            plain.get(&collection, &node).unwrap().is_none(),
            "a plain read-only handle must not see the unsynced committed group"
        );
        drop(plain);

        let store = PackfileStorage::open_read_committed(dir.clone(), &wal).unwrap();
        assert_eq!(
            store.get_read_committed(&collection, &[node]).unwrap()[0]
                .as_ref()
                .map(|data| data.bytes.as_ref()),
            Some(&b"committed"[..]),
            "the one-call handle must serve the committed group from the overlay"
        );
    }

    #[test]
    fn force_index_checkpoint_rewrites_with_and_without_pending_delta() {
        let dir = test_dir("force_index_checkpoint");
        let wal = dir.join("wal.bin");
        let checkpoint_path = PackfileStorage::index_checkpoint_path(&dir);
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&wal).unwrap();

        // Empty delta log: no mutations yet. A normal sync would be a no-op;
        // forcing still writes a checkpoint with no committed coverage.
        store.force_index_checkpoint().unwrap();
        let empty = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();

        // Non-empty delta: the put commits a journal group, and sync_all takes
        // the delta fast path, which does not advance checkpoint coverage.
        store
            .put(
                &TEST_COLLECTION,
                &[0x11; 16],
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let after_sync = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();
        assert_eq!(
            after_sync.covered_lsn, empty.covered_lsn,
            "a delta append must not advance checkpoint coverage"
        );

        // Forcing must take the full-rewrite path and record the committed
        // journal tail. Read the tail from the journal rather than hardcoding
        // an LSN, so this survives any change to how the first mutation is
        // numbered.
        let committed = store.journal().expect("journal enabled").committed_lsn();
        assert!(
            committed > after_sync.covered_lsn,
            "the put must have committed past the loaded coverage"
        );
        store.force_index_checkpoint().unwrap();
        let forced = crate::index::checkpoint::read_checkpoint(&checkpoint_path).unwrap();
        assert_eq!(
            forced.covered_lsn, committed,
            "force must advance coverage to the committed journal tail"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn reset_has_coverage_gap_flags_a_header_only_segment() {
        // A fully reclaimed segment carries no groups, only a moved base LSN.
        // Keying the gap check off the first group would miss it entirely.
        let header_only = crate::journal::Scan {
            groups: Vec::new(),
            valid_len: 0,
            truncated_tail: false,
            base_lsn: 3,
        };
        assert!(ReadJournal::reset_has_coverage_gap(&header_only, 1));
        assert!(!ReadJournal::reset_has_coverage_gap(&header_only, 2));
        assert!(
            !ReadJournal::reset_has_coverage_gap(&crate::journal::Scan::empty(), 0),
            "a missing/short segment reports base 0 and is not a gap"
        );
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn read_committed_overlay_picks_up_groups_appended_after_the_first_scan() {
        let dir = test_dir("read_committed_tail");
        let wal = dir.join("wal.bin");
        let collection = [0x46u8; 16];
        let first = [0x0cu8; 16];
        let second = [0x0du8; 16];

        let seed = PackfileStorage::open(dir.clone()).unwrap();
        seed.put(
            &[0x97u8; 16],
            &[0x97u8; 16],
            &NodeData::new(bytes::Bytes::from_static(b"seed")),
        )
        .unwrap();
        seed.sync().unwrap();
        drop(seed);

        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: first,
                payload: b"first".to_vec(),
            }])
            .unwrap();

        // Open the overlay now, so its first scan stops at the first group.
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        store.enable_read_journal(&wal).unwrap();
        assert!(store.get_read_committed(&collection, &[first]).unwrap()[0].is_some());

        // A later append must be picked up by the incremental tail scan.
        journal
            .append_group(&[JournalMutation::Put {
                collection_id: collection,
                node_id: second,
                payload: b"second".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let read_committed = store.get_read_committed(&collection, &[second]).unwrap();
        assert_eq!(
            read_committed[0].as_ref().map(|data| data.bytes.as_ref()),
            Some(&b"second"[..])
        );
    }

    fn ten_record_fixture() -> Vec<(NodeId, NodeData)> {
        (0..10u8)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                id[8..12].copy_from_slice(&u32::from(i).saturating_add(1).to_le_bytes());
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
                let (slot, offset) = store
                    .generation(&TEST_COLLECTION)
                    .unwrap()
                    .index
                    .lookup(&id)
                    .expect("just-written record must be indexed");
                // Tampering happens on-disk: commit the buffered frame so the
                // tamper site computed from `offset` actually lands in the file.
                store.sync_all().unwrap();
                let path = store.shards.get_shard(slot).unwrap().path.clone();
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
                let reopened = PackfileStorage::open_with_policies(dir.clone(), true, policy);
                match reopened {
                    Ok(_) => panic!("reopen scan must reject a tampered pack"),
                    Err(err) => {
                        assert!(
                            matches!(err, std::io::Error { .. }),
                            "expected I/O error from scan_and_recover, got {err:?}"
                        );
                    }
                }
                // Remove the trusted checkpoint to exercise the read-only
                // full-scan fallback. A valid checkpoint deliberately avoids
                // scanning frames and verifies them lazily on lookup.
                fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
                let read_only = PackfileStorage::open_read_only_with_policies(dir, policy);
                assert!(
                    read_only.is_err(),
                    "read-only recovery must fail closed instead of omitting a corrupt shard"
                );
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

        assert_eq!(
            store.put_many(&TEST_COLLECTION, &entries).unwrap(),
            10,
            "put_many reports every committed entry"
        );
        assert_eq!(
            store.put_many(&TEST_COLLECTION, &[]).unwrap(),
            0,
            "an empty batch commits nothing"
        );

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

    fn setup_repack_mid_lookup(
        test_name: &str,
        id: NodeId,
    ) -> (Arc<PackfileStorage>, u16, Arc<AtomicBool>) {
        // Put the repack in the exact window between candidate collection and
        // shard pinning, and force its output onto a new shard so the original
        // candidate shard is actually retired.
        let dir = test_dir(test_name);
        let store = Arc::new(PackfileStorage::open(dir).unwrap());
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"payload")),
            )
            .unwrap();
        store.sync().unwrap();
        store.generation(&TEST_COLLECTION).unwrap().cache.clear();
        let old_shard = store
            .generation(&TEST_COLLECTION)
            .unwrap()
            .index
            .lookup_all(&id)
            .next()
            .expect("id is indexed")
            .0;

        let fired = Arc::new(AtomicBool::new(false));
        let hook_fired = Arc::clone(&fired);
        let hook_store = Arc::clone(&store);
        PackfileStorage::set_test_before_pin(Some(Box::new(move || {
            if !hook_fired.swap(true, Ordering::Relaxed) {
                hook_store.shards.active_shard().file_len.store(
                    shard::MAX_SHARD_BYTES - 10,
                    std::sync::atomic::Ordering::Release,
                );
                hook_store.set_live_roots(&TEST_COLLECTION, vec![id]);
                hook_store
                    .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
                    .unwrap();
            }
        })));
        (store, old_shard, fired)
    }

    #[test]
    fn test_get_retries_when_a_repack_retires_the_shard_mid_lookup() {
        let id = distinct_id(0x77);
        let (store, old_shard, fired) = setup_repack_mid_lookup("get_retry_mid_repack", id);
        let result = store.get(&TEST_COLLECTION, &id).unwrap();
        PackfileStorage::set_test_before_pin(None);

        assert!(fired.load(Ordering::Relaxed), "the test hook must have run");
        assert!(
            store.shards.get_shard(old_shard).is_none(),
            "the repack must have retired the looked-up shard, or this test proves nothing"
        );
        assert!(
            result.is_some(),
            "get must retry and still find a record whose shard a repack retired mid-lookup"
        );
    }

    #[test]
    fn test_get_many_retries_when_a_repack_retires_the_shard_mid_lookup() {
        let id = distinct_id(0x78);
        let (store, old_shard, fired) = setup_repack_mid_lookup("get_many_retry_mid_repack", id);
        let results = store.get_many(&TEST_COLLECTION, &[id]).unwrap();
        PackfileStorage::set_test_before_pin(None);

        assert!(fired.load(Ordering::Relaxed), "the test hook must have run");
        assert!(
            store.shards.get_shard(old_shard).is_none(),
            "the repack must have retired the looked-up shard, or this test proves nothing"
        );
        assert!(
            results[0].is_some(),
            "get_many must retry and still find a record whose shard a repack retired mid-lookup"
        );
    }

    #[test]
    fn test_walk_ancestors_pins_shards_against_a_mid_walk_repack() {
        // The walk freezes one generation, but a concurrent repack can still
        // retire the shards that generation's index points at. `extract_edges`
        // runs between node resolutions, so triggering the repack from it
        // deterministically reproduces a mid-walk swap; every node must still
        // resolve.
        let dir = test_dir("walk_pins_mid_repack");
        let store = PackfileStorage::open(dir).unwrap();

        let a = distinct_id(0xA0);
        let b = distinct_id(0xA1);
        let c = distinct_id(0xA2);
        for (id, byte) in [(&a, b'A'), (&b, b'B'), (&c, b'C')] {
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::copy_from_slice(&[byte])),
                )
                .unwrap();
        }
        store.sync().unwrap();
        let old_shard = store
            .generation(&TEST_COLLECTION)
            .unwrap()
            .index
            .lookup_all(&c)
            .next()
            .expect("c is indexed")
            .0;

        let edges = std::collections::HashMap::from([(c, vec![b]), (b, vec![a])]);
        let fired = std::cell::Cell::new(false);
        let extract = |hash: &[u8; 16], _data: &[u8]| {
            if !fired.replace(true) {
                // Force the repack's output onto a new shard so the snapshot's
                // shard is retired rather than reused as the live write shard.
                store.shards.active_shard().file_len.store(
                    shard::MAX_SHARD_BYTES - 10,
                    std::sync::atomic::Ordering::Release,
                );
                store.set_live_roots(&TEST_COLLECTION, vec![c]);
                let (kept, dropped) = store
                    .repack_collection_reachable(&TEST_COLLECTION, |h, _d| {
                        edges.get(h).cloned().unwrap_or_default()
                    })
                    .unwrap();
                assert_eq!((kept, dropped), (3, 0), "all three nodes stay reachable");
            }
            edges.get(hash).cloned().unwrap_or_default()
        };

        let walk = store
            .walk_ancestors(&TEST_COLLECTION, &[c], &[], extract, WalkLimits::default())
            .unwrap();
        let results: Vec<(NodeId, NodeData)> = walk.collect::<Result<_, _>>().unwrap();

        assert!(
            store.shards.get_shard(old_shard).is_none(),
            "the repack must have retired the snapshot's shard, or this test proves nothing"
        );
        assert_eq!(
            results.len(),
            3,
            "the walk must resolve every node even though a repack retired the snapshot's shards mid-walk"
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
        assert!(open.total > Duration::ZERO);
        assert!(open.packfile_recovery_calls >= 1);
        assert!(open.packfile_open_calls >= 1);
        assert!(open.packfile_recovery + open.packfile_open <= open.shard_open);
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

    /// Collection creation and growth become per-collection v3 snapshots, so
    /// neither mutation requires a store-wide checkpoint rewrite.
    #[test]
    fn delta_invalidation_triggers_are_new_collections_and_growth() {
        const ROUNDS: u8 = 5;

        // Arm A: one new collection per barrier -- the Complement room-churn
        // shape. Each create adds a collection snapshot to the v3 log.
        let store_a = PackfileStorage::open(test_dir("delta_triggers_new_collections")).unwrap();
        let before_a = store_a.stats();
        for round in 0..ROUNDS {
            let mut cid = [0u8; 16];
            cid[0] = 0xA0;
            cid[1] = round;
            for i in 0..3u8 {
                store_a
                    .put(
                        &cid,
                        &distinct_id(i),
                        &NodeData::new(bytes::Bytes::from(format!("round {round} node {i}"))),
                    )
                    .unwrap();
            }
            store_a.sync_all().unwrap();
        }
        let after_a = store_a.stats();
        assert_eq!(
            after_a.delta_invalidations - before_a.delta_invalidations,
            u64::from(ROUNDS),
            "one is_new invalidation per collection created between barriers"
        );
        assert_eq!(
            after_a.checkpoint_writes - before_a.checkpoint_writes,
            1,
            "only the initial empty-store baseline needs a checkpoint"
        );
        assert_eq!(
            after_a.delta_appends - before_a.delta_appends,
            u64::from(ROUNDS - 1),
            "new collections after the baseline append snapshots"
        );

        // Arm B: one collection that never grows. Its first sync establishes
        // the baseline; subsequent writes append incrementals.
        let store_b = PackfileStorage::open(test_dir("delta_triggers_single_collection")).unwrap();
        let before_b = store_b.stats();
        let mut next = 0u8;
        for _ in 0..ROUNDS {
            for _ in 0..3u8 {
                store_b
                    .put(
                        &TEST_COLLECTION,
                        &distinct_id(next),
                        &NodeData::new(bytes::Bytes::from(format!("node {next}"))),
                    )
                    .unwrap();
                next = next.wrapping_add(1);
            }
            store_b.sync_all().unwrap();
        }
        let after_b = store_b.stats();
        assert_eq!(
            after_b.delta_invalidations - before_b.delta_invalidations,
            1,
            "only collection creation is structural"
        );
        assert_eq!(
            after_b.checkpoint_writes - before_b.checkpoint_writes,
            1,
            "one rewrite re-bases the log, then it is continuable"
        );
        assert_eq!(
            after_b.delta_appends - before_b.delta_appends,
            u64::from(ROUNDS - 1),
            "every barrier after the re-base appends the delta"
        );

        // Arm C: force that same collection past the index's 75%-load grow
        // threshold with a batch (`put_many` is the path that sizes and grows
        // an index; the single-record `put` grow path is not what
        // `index_grow_count` tracks). The grow emits a replacement snapshot.
        let before_c = store_b.stats();
        let entries: Vec<(NodeId, NodeData)> = (next..80u8)
            .map(|i| {
                (
                    distinct_id(i),
                    NodeData::new(bytes::Bytes::from(format!("node {i}"))),
                )
            })
            .collect();
        store_b.put_many(&TEST_COLLECTION, &entries).unwrap();
        store_b.sync_all().unwrap();
        let after_c = store_b.stats();
        let grows = after_c.index_grow_count - before_c.index_grow_count;
        assert!(
            grows >= 1,
            "crossing the load threshold must grow the index"
        );
        assert!(
            after_c.delta_invalidations - before_c.delta_invalidations >= grows,
            "each grow invalidates the pending log"
        );
        assert_eq!(
            after_c.delta_appends - before_c.delta_appends,
            1,
            "a growing collection appends one replacement snapshot"
        );
        assert_eq!(
            after_c.checkpoint_writes - before_c.checkpoint_writes,
            0,
            "growth no longer rewrites the whole checkpoint"
        );
    }

    #[test]
    fn oversized_v3_delta_falls_back_to_checkpoint_without_losing_data() {
        let dir = test_dir("v3_delta_cap_fallback");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(90),
                &NodeData::new(bytes::Bytes::from_static(b"checkpoint base")),
            )
            .unwrap();
        store.sync_all().unwrap();

        let state = store.delta_state.lock();
        let base = state.base_fingerprint.expect("baseline checkpoint exists");
        let delta_path = PackfileStorage::delta_path(&dir, base);
        drop(state);
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&delta_path)
            .unwrap();
        file.set_len(DELTA_LOG_CAP_BYTES.saturating_sub(1)).unwrap();
        store.delta_state.lock().log_bytes = DELTA_LOG_CAP_BYTES.saturating_sub(1);

        let collection = [0xD3; 16];
        let node = distinct_id(91);
        let value = bytes::Bytes::from_static(b"survives oversized snapshot fallback");
        store
            .put(&collection, &node, &NodeData::new(value.clone()))
            .unwrap();
        let before = store.stats();
        store.sync_all().unwrap();
        let after = store.stats();
        assert_eq!(after.delta_appends - before.delta_appends, 0);
        assert_eq!(after.checkpoint_writes - before.checkpoint_writes, 1);
        assert_eq!(store.get(&collection, &node).unwrap().unwrap().bytes, value);
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_eq!(
            reopened.get(&collection, &node).unwrap().unwrap().bytes,
            value
        );
        drop(reopened);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_puts_during_sync_keep_using_the_v3_append_path() {
        let dir = test_dir("concurrent_put_sync_delta");
        let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(1),
                &NodeData::new(bytes::Bytes::from_static(b"base")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let before = store.stats();

        let writers: Vec<_> = (2..18u8)
            .map(|seed| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    store
                        .put(
                            &TEST_COLLECTION,
                            &distinct_id(seed),
                            &NodeData::new(bytes::Bytes::from(vec![seed; 128])),
                        )
                        .unwrap();
                })
            })
            .collect();
        for _ in 0..32 {
            store.sync_all().unwrap();
            std::thread::yield_now();
        }
        for writer in writers {
            writer.join().unwrap();
        }
        store.sync_all().unwrap();

        let after = store.stats();
        assert_eq!(
            after.checkpoint_writes - before.checkpoint_writes,
            0,
            "concurrent puts must not make the pre-lock fingerprint stale and force rewrites"
        );
        assert!(after.delta_appends - before.delta_appends > 0);
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        for seed in 2..18u8 {
            assert!(reopened
                .get(&TEST_COLLECTION, &distinct_id(seed))
                .unwrap()
                .is_some());
        }
        drop(reopened);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn inherited_v2_delta_replays_then_rebases_to_checkpoint() {
        let dir = test_dir("v2_delta_rebase");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(20),
                &NodeData::new(bytes::Bytes::from_static(b"checkpointed")),
            )
            .unwrap();
        store.sync_all().unwrap();

        let replayed_id = distinct_id(21);
        let replayed_value = bytes::Bytes::from_static(b"inherited v2 delta");
        store
            .put(
                &TEST_COLLECTION,
                &replayed_id,
                &NodeData::new(replayed_value.clone()),
            )
            .unwrap();
        store.flush_all().unwrap();
        let current = store.delta_state.lock().clone();
        let base = current.base_fingerprint.expect("checkpoint base exists");
        let frames = match &current.pending[&TEST_COLLECTION] {
            PendingDelta::Slots(frames) => frames.clone(),
            _ => panic!("plain put records a v2-compatible incremental frame"),
        };
        let tail = store.current_pack_fingerprint();
        let path = PackfileStorage::delta_path(&dir, base);
        let bytes_written = crate::index::delta::append_batch(&path, true, base, &frames, tail)
            .expect("write legacy v2 log");
        {
            let mut state = store.delta_state.lock();
            state.log_version = 2;
            state.log_bytes = u64::try_from(bytes_written).unwrap();
        }
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_eq!(
            reopened
                .get(&TEST_COLLECTION, &replayed_id)
                .unwrap()
                .unwrap()
                .bytes,
            replayed_value
        );
        let next_id = distinct_id(22);
        reopened
            .put(
                &TEST_COLLECTION,
                &next_id,
                &NodeData::new(bytes::Bytes::from_static(b"post rebase")),
            )
            .unwrap();
        let before = reopened.stats();
        reopened.sync_all().unwrap();
        let after = reopened.stats();
        assert_eq!(after.checkpoint_writes - before.checkpoint_writes, 1);
        assert_eq!(after.delta_appends - before.delta_appends, 0);
        drop(reopened);

        let final_open = PackfileStorage::open(dir.clone()).unwrap();
        assert!(final_open
            .get(&TEST_COLLECTION, &replayed_id)
            .unwrap()
            .is_some());
        assert!(final_open
            .get(&TEST_COLLECTION, &next_id)
            .unwrap()
            .is_some());
        drop(final_open);
        fs::remove_dir_all(dir).unwrap();
    }

    /// The per-collection v3 snapshots make the rewrite budget irrelevant to
    /// ordinary room churn: after the initial baseline, each sync appends.
    #[test]
    fn checkpoint_rewrite_budget_defers_invalidated_rewrites() {
        const ROUNDS: u8 = 5;
        let dir = test_dir("checkpoint_rewrite_budget");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        // Once a rewrite has run, defer the next for an hour; no size budget.
        store.set_checkpoint_rewrite_budget(Duration::from_secs(3600), 0);
        let before = store.stats();
        let mut written: Vec<([u8; 16], [u8; 16], Vec<u8>)> = Vec::new();
        for round in 0..ROUNDS {
            let mut cid = [0u8; 16];
            cid[0] = 0xB0;
            cid[1] = round;
            for i in 0..3u8 {
                let id = distinct_id(i);
                let value = format!("budget round {round} node {i}").into_bytes();
                store
                    .put(&cid, &id, &NodeData::new(bytes::Bytes::from(value.clone())))
                    .unwrap();
                written.push((cid, id, value));
            }
            store.sync_all().unwrap();
        }
        let after = store.stats();
        let checkpoints = after.checkpoint_writes - before.checkpoint_writes;
        let skips = after.checkpoint_skips - before.checkpoint_skips;
        eprintln!(
            "rewrite-budget impact over {ROUNDS} barriers: checkpoint_writes={checkpoints} \
             checkpoint_skips={skips} delta_appends={}",
            after.delta_appends - before.delta_appends
        );
        assert_eq!(
            checkpoints, 1,
            "one baseline rewrite, then the budget defers"
        );
        assert_eq!(
            skips, 0,
            "v3 snapshots avoid checkpoint skips for ordinary structural changes"
        );
        assert_eq!(
            after.delta_appends - before.delta_appends,
            u64::from(ROUNDS - 1)
        );
        drop(store);

        // The v3 log replays the collection snapshots on reopen.
        let reopened = PackfileStorage::open(dir).unwrap();
        assert_eq!(
            reopened
                .open_timings()
                .expect("open must record timings")
                .path,
            OpenPath::Checkpoint,
            "the checkpoint plus v3 log should avoid a full scan"
        );
        for (cid, id, expected) in &written {
            let got = reopened
                .get(cid, id)
                .unwrap()
                .expect("record survives the rescan");
            assert_eq!(got.bytes.as_ref(), expected.as_slice());
        }
    }

    /// With a journal enabled, `sync` makes the WAL group commit the
    /// durability point rather than the per-shard pack fsyncs.
    #[test]
    fn journal_sync_routes_durability_through_wal() {
        let dir = test_dir("journal_sync_routing");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(dir.join("wal.bin")).unwrap();
        let id = distinct_id(7);
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"wal")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let timings = store.sync_timings().expect("sync must record timings");
        assert!(
            timings.wal > Duration::ZERO,
            "the journal group commit is the durability point"
        );
        assert_eq!(
            timings.pack_fsync,
            Duration::ZERO,
            "the barrier path must not fsync pack shards with a WAL"
        );
        assert_eq!(timings.journal_sync_calls, 1);
        assert!(timings.journal_records >= 1);
        assert!(timings.journal_bytes > 0);
        assert!(timings.journal_fsync > Duration::ZERO);
        assert!(
            timings.journal_lock_wait + timings.journal_append + timings.journal_fsync
                <= timings.wal,
            "reported journal phases cannot exceed the WAL phase"
        );
        let stats = store.stats();
        assert_eq!(stats.publish_calls, 1);
        assert_eq!(stats.sync_diagnostics.worst_syncs.len(), 1);
        assert_eq!(stats.sync_diagnostics.peak_journal_in_flight, 1);
        assert_eq!(
            stats
                .sync_diagnostics
                .fsync_latency
                .buckets
                .iter()
                .sum::<u64>(),
            1
        );
        assert_eq!(
            stats
                .sync_diagnostics
                .lock_wait_latency
                .buckets
                .iter()
                .sum::<u64>(),
            1
        );
        assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
        assert!(store.journal().expect("journal enabled").committed_lsn() >= 1);
    }

    /// A mutation committed to the journal after the last checkpoint is
    /// re-applied by `replay_journal` on a fresh open.
    #[test]
    fn journal_replays_post_checkpoint_mutations_on_reopen() {
        let dir = test_dir("journal_replay");
        let journal_path = dir.join("wal.bin");
        let id = distinct_id(9);
        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            store.enable_journal(&journal_path).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(8),
                    &NodeData::new(bytes::Bytes::from_static(b"checkpointed")),
                )
                .unwrap();
            store.sync_all().unwrap();
            // A second, non-structural put: the next sync's delta path does not
            // write a checkpoint, so this mutation stays journal-only.
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(b"wal-only")),
                )
                .unwrap();
            store.sync_all().unwrap();
        }
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        reopened.enable_journal(&journal_path).unwrap();
        let replayed = reopened.replay_journal().unwrap();
        assert!(
            replayed >= 1,
            "the post-checkpoint mutation must be replayed"
        );
        assert!(reopened.get(&TEST_COLLECTION, &id).unwrap().is_some());
    }

    /// A checkpoint must bound journal replay by the LSN that is actually
    /// *committed*, never the published one. A concurrent put can publish
    /// above the last WAL commit, and LSNs above `committed_lsn` are discarded
    /// and reused after a crash -- recording one as covered would make a
    /// reopen skip a future mutation that reuses it. Drive the full-checkpoint
    /// path directly so a pending (published, uncommitted) put is live without
    /// the barrier commit that a real `sync` would have done first.
    #[test]
    fn checkpoint_records_the_committed_journal_lsn() {
        let dir = test_dir("journal_checkpoint_committed_lsn");
        let journal_path = dir.join("wal.bin");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store.enable_journal(&journal_path).unwrap();

        // Commit one mutation so the journal has a non-zero durable prefix.
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(41),
                &NodeData::new(bytes::Bytes::from_static(b"committed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let journal = store.journal().expect("journal enabled");
        let committed = journal.committed_lsn();
        assert_eq!(
            journal.published_lsn(),
            committed,
            "test setup: a completed sync leaves nothing published-but-uncommitted"
        );
        assert!(
            committed > 0,
            "test setup: the first sync must commit an LSN"
        );

        // Publish a second mutation into a new collection WITHOUT committing:
        // this is the state a concurrent `put` leaves behind after a barrier.
        store
            .put(
                &SECOND_COLLECTION,
                &distinct_id(42),
                &NodeData::new(bytes::Bytes::from_static(b"pending")),
            )
            .unwrap();
        let journal = store.journal().expect("journal enabled");
        let published = journal.published_lsn();
        assert!(
            published > journal.committed_lsn(),
            "test setup: the second put must be published but uncommitted"
        );

        // Run the full-checkpoint path as a sync would, but without the
        // barrier that would advance `committed_lsn` past the pending put.
        store
            .persist_index_checkpoint()
            .expect("checkpoint must succeed");

        let covered = PackfileStorage::read_journal_lsn(&dir);
        assert_eq!(
            covered, committed,
            "the checkpoint must record the committed LSN, not a published-only one"
        );
        assert!(
            covered < published,
            "recording the published LSN would let a reopen skip a reused LSN"
        );
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
        // reopen and append. Reopen should continue the v3 epoch directly.
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
                "AFTER REOPEN PUT: cap={} len={} load={:.3} clones+={} grows+={} invalids+={} | pending={:?}",
                room.index.capacity(),
                room.index.len(),
                reopen_probe_load_factor(&room),
                after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
                after_put.index_grow_count - before.index_grow_count,
                after_put.delta_invalidations - before.delta_invalidations,
                d.pending.keys().collect::<Vec<_>>(),
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
            assert!(matches!(
                d.pending.get(&REOPEN_PROBE_COLLECTION),
                Some(PendingDelta::Slots(_))
            ));
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
                "REOPEN delta_state: base_fingerprint={:?} pending={} log_bytes={}",
                d.base_fingerprint,
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
                "AFTER REOPEN PUT: clones+={} grows+={} invalids+={} | pending={:?}",
                after_put.put_many_clone_path_calls - before.put_many_clone_path_calls,
                after_put.index_grow_count - before.index_grow_count,
                after_put.delta_invalidations - before.delta_invalidations,
                d.pending.keys().collect::<Vec<_>>(),
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
            assert!(matches!(
                d.pending.get(&REOPEN_PROBE_COLLECTION),
                Some(PendingDelta::Snapshot)
            ));
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
            assert_eq!(st.checkpoint_writes - before.checkpoint_writes, 0);
            assert_eq!(st.delta_appends - before.delta_appends, 1);
            assert_eq!(ts.checkpoint, std::time::Duration::ZERO);
            assert!(ts.delta_log > std::time::Duration::ZERO);

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
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        for i in total..(total + 288) {
            let got = reopened
                .get(&REOPEN_PROBE_COLLECTION, &reopen_probe_id(i))
                .unwrap()
                .expect("snapshot and following incrementals survive reopen");
            assert_eq!(got.bytes.as_ref(), if i < total + 32 { b"y" } else { b"z" });
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
    fn test_put_many_publishes_successive_batches() {
        // Regression coverage for successful copy-on-write batch publication:
        // every record remains readable across several batches that need no
        // index growth.
        let dir = test_dir("put_many_successive_batches");
        let store = PackfileStorage::open(dir).unwrap();

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

        // Batched writes against an already-materialized (non-mmap) index
        // mutate it in place and roll back via an undo log on failure,
        // rather than cloning -- so only the very first batch (which
        // creates the collection) pays the clone/materialize cost.
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
        // Only the first batch creates `SECOND_COLLECTION` (the `None` arm,
        // which always materializes); every later batch finds an
        // already-owned, non-mmap index and takes the in-place fast path.
        assert_eq!(snapshot.put_many_clone_path_calls, 1);
        assert_eq!(snapshot.put_many_fast_path_calls, 7);
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
    fn stats_surfaces_max_index_probe_len_and_dirty_lock_wait() {
        let dir = test_dir("stats_probe_len_and_lock_wait");
        let store = PackfileStorage::open(dir).unwrap();

        // Fresh store, no writes: neither counter has anything to report.
        let snapshot = store.stats();
        assert_eq!(snapshot.max_index_probe_len, 0);
        assert_eq!(snapshot.dirty_lock_wait, std::time::Duration::ZERO);

        for i in 0..20u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        // A real sync exercises the dirty-set lock at least once; the exact
        // wait duration is unmeasurable deterministically (uncontended in a
        // single-threaded test), but the field must be wired through from
        // `ShardPool::dirty_lock_wait` and never decrease.
        store.sync().unwrap();
        let after_sync = store.stats();
        assert!(after_sync.dirty_lock_wait >= snapshot.dirty_lock_wait);
        // 20 inserts into a real index will very likely walk at least one
        // non-trivial probe chain, but this is observability, not a
        // guarantee -- assert the field is wired through and sane
        // (bounded by the collection's own capacity) rather than a specific
        // value.
        assert!(after_sync.max_index_probe_len < 1000);
    }

    #[test]
    fn test_read_amplification_counters_cover_batch_candidates() {
        let dir = test_dir("read_amplification_counters");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let entries: Vec<_> = (0..4u8)
            .map(|value| {
                let mut id = [0u8; 16];
                id[0] = value;
                (id, NodeData::new(bytes::Bytes::from(vec![value; 8])))
            })
            .collect();
        writer.put_many(&TEST_COLLECTION, &entries).unwrap();
        writer.sync_all().unwrap();
        drop(writer);

        let reader = PackfileStorage::open_read_only(dir).unwrap();
        reader.set_stats_enabled(true);
        let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
        let results = reader.get_many(&TEST_COLLECTION, &ids).unwrap();
        assert_eq!(
            results.iter().filter(|value| value.is_some()).count(),
            ids.len()
        );

        let stats = reader.stats();
        assert_eq!(stats.get_many_calls, 1);
        assert_eq!(stats.get_many_records, ids.len() as u64);
        assert!(stats.index_candidates >= ids.len() as u64);
        assert!(stats.candidate_reads >= ids.len() as u64);
        assert!(stats.get_many_shards_touched >= 1);
    }

    #[test]
    fn test_read_scatter_counters_track_runs_span_and_bytes() {
        let dir = test_dir("read_scatter_counters");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let entries: Vec<_> = (0..8u8)
            .map(|value| {
                let mut id = [0u8; 16];
                id[0] = value;
                id[9] = value.wrapping_mul(37).wrapping_add(11);
                (id, NodeData::new(bytes::Bytes::from(vec![value; 16])))
            })
            .collect();
        writer.put_many(&TEST_COLLECTION, &entries).unwrap();
        writer.sync_all().unwrap();
        drop(writer);

        let reader = PackfileStorage::open_read_only(dir).unwrap();
        reader.set_stats_enabled(true);

        // One batch read of all eight records: the plan covers one shard,
        // its offsets are a single sequential run, and span/bytes record the
        // extent the batch drags in.
        let before = reader.stats();
        let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
        let results = reader.get_many(&TEST_COLLECTION, &ids).unwrap();
        assert_eq!(
            results.iter().filter(|value| value.is_some()).count(),
            entries.len()
        );
        let after = reader.stats();
        assert_eq!(
            after.get_many_shards_touched - before.get_many_shards_touched,
            1
        );
        assert_eq!(after.read_many_runs - before.read_many_runs, 1);
        assert!(after.read_many_span_bytes > before.read_many_span_bytes);
        assert!(after.candidate_frame_bytes > before.candidate_frame_bytes);

        // A single `get` adds candidate frame bytes but is not a batch, so
        // the batch-scatter shape counters (`runs`, `span`) must not move.
        let single_before = reader.stats();
        reader
            .get(&TEST_COLLECTION, &entries[0].0)
            .unwrap()
            .unwrap();
        let single_after = reader.stats();
        assert!(single_after.candidate_frame_bytes > single_before.candidate_frame_bytes);
        assert_eq!(single_after.read_many_runs, single_before.read_many_runs);
        assert_eq!(
            single_after.read_many_span_bytes,
            single_before.read_many_span_bytes
        );
    }

    #[test]
    fn test_walk_ancestors_counts_candidate_probes_reads_and_frame_bytes() {
        let dir = test_dir("walk_counter_tracking");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let ids: Vec<NodeId> = (0..3u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = b'A' + u8::try_from(i).unwrap();
            writer
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![byte])),
                )
                .unwrap();
        }
        writer.sync_all().unwrap();
        drop(writer);

        // A fresh read-only store has an empty decoded-node cache, so a walk
        // must resolve every hash through the lossy index and the packfiles —
        // and the counters for that traverse path (`resolve_pinned`, the
        // ancestor/frontier hook) must move, not just `get`/`get_many`.
        let reader = PackfileStorage::open_read_only(dir).unwrap();
        reader.set_stats_enabled(true);
        let edges =
            std::collections::HashMap::from([(ids[1], vec![ids[0]]), (ids[2], vec![ids[1]])]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();
        let walked: Vec<(NodeId, NodeData)> = reader
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[2]],
                &[],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(walked.len(), 3);

        let stats = reader.stats();
        assert!(stats.index_candidates >= 3);
        assert!(stats.candidate_reads >= 3);
        assert!(stats.candidate_frame_bytes >= stats.candidate_reads);
        assert!(stats.cache.misses >= 3);
        // `walk_ancestors` is a traversal, not a `get_many` batch: its
        // scatter-shape counters must not be the ones that moved.
        assert_eq!(stats.read_many_runs, 0);
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
        assert!(check_index_offset(0, &hash, (1u64 << 28) + 1).is_ok());
        assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT + 1).is_err());
        assert!(check_index_offset(0, &hash, u64::MAX).is_err());
    }

    #[test]
    fn test_get_many_with_refresh_coalesces_negative_lookups() {
        let dir = test_dir("get_many_with_refresh_negative");
        let store = PackfileStorage::open(dir).unwrap();
        // Use a collection that was NOT pre-seeded (not in collection_order)
        // so the first miss has no stored fingerprint → refresh required.
        let missing = [[0xF0u8; 16], [0xF1u8; 16]];

        assert!(store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap()
            .iter()
            .all(Option::is_none));
        assert!(store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap()
            .iter()
            .all(Option::is_none));

        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert_eq!(stats.miss_refresh_recovered, 0);
        assert_eq!(stats.miss_refresh_skips, 2);
        assert_eq!(stats.miss_refresh_retry_ids, 0);
    }

    #[test]
    fn test_get_many_with_refresh_disabled_is_authoritative() {
        let dir = test_dir("refresh_disabled_authoritative");
        let store = Arc::new(PackfileStorage::open(dir).unwrap());
        // A single writer's in-memory index is authoritative for every key it
        // has written, so a negative lookup must bypass the refresh lock and
        // the durable-fingerprint probe entirely rather than paying either.
        store.set_refresh_on_miss(false);

        let missing = [[0xF0u8; 16], [0xF1u8; 16]];
        let refresh_lock = store.refresh_lock(&TEST_COLLECTION);
        let refresh_guard = refresh_lock.lock();
        let (sender, receiver) = std::sync::mpsc::channel();
        let reader_store = Arc::clone(&store);
        let reader = std::thread::spawn(move || {
            let result = reader_store
                .get_many_with_refresh(&TEST_COLLECTION, &missing)
                .unwrap()
                .iter()
                .all(Option::is_none);
            sender.send(result).unwrap();
        });

        // Keep the refresh lock held while the lookup runs. The writer path
        // must return from the in-memory index without waiting for it. This is
        // a deadlock guard, not a latency assertion: the timeout is generous
        // enough that a merely loaded CI worker cannot trip it. The
        // timing-independent proof that no durable fingerprint was probed is
        // the `miss_refreshes == 0` counter below.
        assert!(receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap());
        drop(refresh_guard);
        reader.join().unwrap();

        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert_eq!(stats.miss_refresh_skips, 0);
        assert_eq!(stats.miss_refresh_retry_ids, 0);
    }

    #[test]
    fn test_get_many_with_refresh_flag_gates_the_reader_path() {
        let dir = test_dir("refresh_flag_gates");
        let store = PackfileStorage::open(dir).unwrap();
        let missing = [[0xF2u8; 16]];

        store.set_refresh_on_miss(false);
        let _ = store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap();
        assert_eq!(store.stats().miss_refresh_skips, 0);

        // Re-enabling restores the multi-process-reader behavior: the miss is
        // rate-limited by the durable fingerprint rather than bypassed.
        store.set_refresh_on_miss(true);
        let _ = store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap();
        assert_eq!(store.stats().miss_refresh_skips, 1);
    }

    #[test]
    fn test_get_many_with_refresh_all_hit_skips_refresh() {
        let dir = test_dir("refresh_all_hit");
        let store = PackfileStorage::open(dir).unwrap();

        let id_a = distinct_id(0xA0);
        let id_b = distinct_id(0xA1);
        let data = NodeData::new(bytes::Bytes::from_static(b"payload"));
        store.put(&TEST_COLLECTION, &id_a, &data).unwrap();
        store.put(&TEST_COLLECTION, &id_b, &data).unwrap();
        store.sync().unwrap();

        store.reset_stats();
        let result = store
            .get_many_with_refresh(&TEST_COLLECTION, &[id_a, id_b])
            .unwrap();
        assert!(result.iter().all(Option::is_some));

        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert_eq!(stats.miss_refresh_skips, 0);
    }

    #[test]
    fn test_get_many_with_refresh_mixed_batch_retries_only_misses() {
        let dir = test_dir("refresh_mixed_batch");
        let store = PackfileStorage::open(dir).unwrap();

        let present = distinct_id(0xB0);
        let missing = distinct_id(0xBF);
        let data = NodeData::new(bytes::Bytes::from_static(b"here"));
        store.put(&TEST_COLLECTION, &present, &data).unwrap();
        store.sync().unwrap();

        store.reset_stats();
        let result = store
            .get_many_with_refresh(&TEST_COLLECTION, &[present, missing])
            .unwrap();
        assert!(result[0].is_some());
        assert!(result[1].is_none());

        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 1);
        assert_eq!(stats.miss_refresh_recovered, 0);
        // Exactly the 1 missing ID was retried.
        assert_eq!(stats.miss_refresh_retry_ids, 1);
    }

    #[test]
    fn test_get_many_with_refresh_unsynced_append_invisible() {
        let dir = test_dir("refresh_unsynced");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        let id = distinct_id(0xC0);
        let data = NodeData::new(bytes::Bytes::from_static(b"unsynced"));

        // Writer appends but does NOT sync.
        writer.put(&TEST_COLLECTION, &id, &data).unwrap();

        // Reader sees durable fingerprint unchanged (writer didn't sync)
        // → dominated → confirmed negative, no refresh.
        reader.reset_stats();
        let result = reader
            .get_many_with_refresh(&TEST_COLLECTION, &[id])
            .unwrap();
        assert!(result[0].is_none(), "unsynced append must remain invisible");

        let stats = reader.stats();
        assert_eq!(
            stats.miss_refreshes, 0,
            "no refresh should occur for unsynced data"
        );
        assert_eq!(stats.miss_refresh_skips, 1);
    }

    #[test]
    fn test_get_many_with_refresh_synced_append_detected() {
        let dir = test_dir("refresh_synced_append");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        let id = distinct_id(0xD0);
        let data = NodeData::new(bytes::Bytes::from_static(b"synced"));

        // Writer appends and syncs → durable fingerprint changes.
        writer.put(&TEST_COLLECTION, &id, &data).unwrap();
        writer.sync().unwrap();

        // Existing reader: stored fp != durable fp → refresh → recovered.
        reader.reset_stats();
        let result = reader
            .get_many_with_refresh(&TEST_COLLECTION, &[id])
            .unwrap();
        assert_eq!(
            result[0].as_ref().map(|d| &d.bytes),
            Some(&bytes::Bytes::from_static(b"synced")),
            "synced append must be recovered by the existing reader handle"
        );

        let stats = reader.stats();
        assert_eq!(stats.miss_refreshes, 1);
        assert_eq!(stats.miss_refresh_recovered, 1);
        assert_eq!(stats.miss_refresh_retry_ids, 1);
    }

    /// Regression for the deferred-checkpoint reader window.
    ///
    /// A writer that owes a *structurally needed* full checkpoint rewrite
    /// (`pending` is empty while the index is dirty — the shape a
    /// `refresh_collection` that pulled in another process's appends leaves
    /// behind) will skip that rewrite under
    /// [`PackfileStorage::set_checkpoint_rewrite_budget`]. The packs are still
    /// flushed and fsynced, so the record is durable; but with no checkpoint
    /// rewrite and no delta frame, the durable fingerprint does not advance.
    /// A reader handle that opened at the previous fingerprint therefore treats
    /// its negative as confirmed and never refreshes: a *synced* record stays
    /// invisible to [`PackfileStorage::get_many_with_refresh`] for as long as
    /// the budget defers the rewrite.
    ///
    /// Ignored because WAL-off cross-process reads are no longer a supported
    /// shape. The journal's [`PackfileStorage::get_read_committed`] overlay is
    /// the authoritative cross-process visibility path and serves the committed
    /// record before the checkpoint advances, independent of the rewrite
    /// budget. This pins the unsupported window so a future change that makes
    /// WAL-off reads appear to work by accident (rather than by the overlay) is
    /// noticed.
    ///
    /// Known issue, not exercised here: `post_refresh_fingerprint` records the
    /// *post*-refresh durable fingerprint as the collection's refresh baseline,
    /// so a sync racing between `refresh_collection` and that read can store a
    /// baseline the index never actually incorporated — over-claiming coverage
    /// and suppressing a refresh that was needed. A fix must capture the
    /// fingerprint the refresh itself observed rather than re-reading it.
    #[test]
    #[ignore = "WAL-off cross-process reads are unsupported; the journal overlay is authoritative (see test docs)"]
    fn deferred_checkpoint_rewrite_leaves_synced_write_invisible() {
        let dir = test_dir("deferred_checkpoint_reader_window");
        let writer = PackfileStorage::open(dir.clone()).unwrap();

        // Baseline checkpoint, so the writer has a prior rewrite timestamp for
        // the budget's time headroom to measure against.
        let seed = distinct_id(0xC0);
        writer
            .put(
                &TEST_COLLECTION,
                &seed,
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
        writer.sync().unwrap();

        // Reader opens bound to the baseline checkpoint's fingerprint.
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        // Once a rewrite has run, defer the next for an hour; no size budget.
        writer.set_checkpoint_rewrite_budget(Duration::from_secs(3600), 0);

        let id = distinct_id(0xC1);
        writer
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"deferred")),
            )
            .unwrap();
        // The record is eagerly in the pack, but leave no local delta frame:
        // this is the `pending.is_empty()` + dirty shape that makes the next
        // sync structurally need a full rewrite (as a refresh of external
        // appends does).
        writer.delta_state.lock().pending.clear();
        writer.index_checkpoint_dirty.store(true, Ordering::Relaxed);

        let before = writer.stats();
        writer.sync().unwrap();
        let after = writer.stats();
        assert_eq!(
            after.checkpoint_writes - before.checkpoint_writes,
            0,
            "the structurally-needed rewrite must be deferred, not written"
        );
        assert_eq!(
            after.checkpoint_skips - before.checkpoint_skips,
            1,
            "the sync must have taken the deferred branch"
        );

        // The pack bytes are durable, but the durable fingerprint is unchanged,
        // so the reader's gate confirms the negative without refreshing.
        reader.reset_stats();
        let result = reader
            .get_many_with_refresh(&TEST_COLLECTION, &[id])
            .unwrap();
        assert!(
            result[0].is_none(),
            "a synced record stays invisible while the checkpoint rewrite is deferred"
        );
        let reader_stats = reader.stats();
        assert_eq!(
            reader_stats.miss_refreshes, 0,
            "an unchanged durable fingerprint suppresses the refresh entirely"
        );
        assert_eq!(reader_stats.miss_refresh_skips, 1);
    }

    #[test]
    fn test_get_many_with_refresh_repeated_negative_suppressed() {
        let dir = test_dir("refresh_repeated_negative");
        let store = PackfileStorage::open(dir).unwrap();

        let missing = [distinct_id(0xE0), distinct_id(0xE1)];

        // First call: no stored fingerprint matches durable fp (both 0),
        // so dominated = true → skip refresh.
        store.reset_stats();
        let _ = store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap();
        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert_eq!(stats.miss_refresh_skips, 1);

        // Second call: same durable fingerprint → still dominated.
        let _ = store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap();
        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert!(stats.miss_refresh_skips >= 2);
    }

    #[test]
    fn test_get_many_with_refresh_preserves_result_order() {
        let dir = test_dir("refresh_ordering");
        let store = PackfileStorage::open(dir).unwrap();

        let present_a = distinct_id(0xF0);
        let present_b = distinct_id(0xF1);
        let missing = distinct_id(0xFF);
        let data = NodeData::new(bytes::Bytes::from_static(b"data"));

        store.put(&TEST_COLLECTION, &present_a, &data).unwrap();
        store.put(&TEST_COLLECTION, &present_b, &data).unwrap();
        store.sync().unwrap();

        let batch = [missing, present_a, present_b, missing];
        let result = store
            .get_many_with_refresh(&TEST_COLLECTION, &batch)
            .unwrap();

        assert!(result[0].is_none(), "first missing stays at index 0");
        assert!(result[1].is_some(), "present_a stays at index 1");
        assert!(result[2].is_some(), "present_b stays at index 2");
        assert!(result[3].is_none(), "second missing stays at index 3");

        let stats = store.stats();
        assert_eq!(stats.miss_refreshes, 1);
        // Exactly 2 missing IDs were retried.
        assert_eq!(stats.miss_refresh_retry_ids, 2);
    }

    #[test]
    fn test_get_many_with_refresh_concurrent_misses_coalesce() {
        use std::sync::Arc;

        let dir = test_dir("refresh_concurrent");
        let store = Arc::new(PackfileStorage::open(dir).unwrap());

        // Seed a record so TEST_COLLECTION exists and has a shard on disk.
        let seed_id = distinct_id(0xC0);
        store
            .put(
                &TEST_COLLECTION,
                &seed_id,
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync().unwrap();

        let missing = [distinct_id(0xC1), distinct_id(0xC2)];

        // Pre-seed: first call populates last_refresh_fingerprint for
        // TEST_COLLECTION. Durable fp is the same as stored → skip.
        store.reset_stats();
        let _ = store
            .get_many_with_refresh(&TEST_COLLECTION, &missing)
            .unwrap();

        // Append and sync to change the durable fingerprint.
        let id = distinct_id(0xC3);
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"synced")),
            )
            .unwrap();
        store.sync().unwrap();

        // Stored fp is stale → refresh needed. Spawn concurrent readers.
        store.reset_stats();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let _ = store.get_many_with_refresh(&TEST_COLLECTION, &missing);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Only one refresh should have actually occurred (coalesced).
        let stats = store.stats();
        assert_eq!(
            stats.miss_refreshes, 1,
            "concurrent misses must coalesce into one refresh"
        );
    }

    #[test]
    fn test_get_many_with_refresh_negative_then_recovered() {
        let dir = test_dir("refresh_negative_then_recovered");
        let writer = PackfileStorage::open(dir.clone()).unwrap();
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        let missing_id = distinct_id(0xD1);

        // First call: reader sees no durable change → confirmed negative.
        reader.reset_stats();
        let result = reader
            .get_many_with_refresh(&TEST_COLLECTION, &[missing_id])
            .unwrap();
        assert!(result[0].is_none());
        let stats = reader.stats();
        assert_eq!(stats.miss_refreshes, 0);
        assert_eq!(stats.miss_refresh_skips, 1);

        // Writer appends and syncs → durable fingerprint changes.
        let data = NodeData::new(bytes::Bytes::from_static(b"now_here"));
        writer.put(&TEST_COLLECTION, &missing_id, &data).unwrap();
        writer.sync().unwrap();

        // Second call: stored fp != durable fp → refresh → recovered.
        let result = reader
            .get_many_with_refresh(&TEST_COLLECTION, &[missing_id])
            .unwrap();
        assert_eq!(
            result[0].as_ref().map(|d| &d.bytes),
            Some(&bytes::Bytes::from_static(b"now_here")),
            "record written after first negative must be recovered"
        );
        let stats = reader.stats();
        assert_eq!(stats.miss_refreshes, 1);
        assert_eq!(stats.miss_refresh_recovered, 1);
        assert_eq!(stats.miss_refresh_retry_ids, 1);
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
                metadata: None,
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
        let store = PackfileStorage::open(dir.clone()).unwrap();

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

        // The collection occupied several packs, and repack retired the
        // garbage-only source packs. The persisted byte total must follow the
        // surviving physical layout rather than retaining bytes for retired
        // packs.
        store.sync_all().unwrap();
        let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
        let expected = physical
            .collections
            .get(&TEST_COLLECTION)
            .expect("repacked collection remains on disk")
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "retired packs must not remain in collection byte accounting"
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
        let disk_bytes = PackfileStorage::collection_disk_bytes_from_disk(&dir)
            .expect("v5 sidecar physical metrics");
        assert!(disk_bytes.get(&TEST_COLLECTION).copied().unwrap_or(0) > 0);
        assert!(disk_bytes.get(&OTHER_COLLECTION).copied().unwrap_or(0) > 0);
        assert!(PackfileStorage::collection_directory_persisted_at(&dir).is_some());
    }

    #[test]
    fn put_many_overwrites_keep_one_live_node_and_count_every_frame() {
        let dir = test_dir("put_many_overwrites_accounting");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let id = distinct_id(0);
        let entries = vec![
            (id, NodeData::new(bytes::Bytes::from_static(b"first"))),
            (id, NodeData::new(bytes::Bytes::from_static(b"second"))),
        ];
        assert_eq!(store.put_many(&TEST_COLLECTION, &entries).unwrap(), 2);
        store.sync_all().unwrap();

        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            vec![(TEST_COLLECTION, 1)],
            "an overwrite inside one batch must not inflate live-node counts"
        );
        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "batch accounting must include both appended frames"
        );
    }

    #[test]
    fn put_verified_and_empty_records_are_accounted_as_physical_frames() {
        let dir = test_dir("verified_and_empty_accounting");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let verified_id = distinct_id(0);
        let verified = bytes::Bytes::from_static(b"verified");
        let digest = DigestAlgorithm::Sha256.digest(&verified);
        store
            .put_verified(
                &TEST_COLLECTION,
                &verified_id,
                &NodeData::new(verified),
                &digest,
                DigestAlgorithm::Sha256,
                &digest,
                None,
            )
            .unwrap();
        let empty_id = distinct_id(1);
        store
            .put(
                &TEST_COLLECTION,
                &empty_id,
                &NodeData::new(bytes::Bytes::new()),
            )
            .unwrap();
        store.sync_all().unwrap();

        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            vec![(TEST_COLLECTION, 2)],
            "empty records still occupy live index slots"
        );
        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "verified and empty records must use their encoded frame lengths"
        );
    }

    #[test]
    fn failed_sidecar_recovery_is_retried_and_never_publishes_zero_metrics() {
        let dir = test_dir("sidecar_recovery_failure");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"data")),
            )
            .unwrap();
        store.sync_all().unwrap();
        let sidecar = PackfileStorage::shard_collections_path(&dir);
        let before = fs::read(&sidecar).unwrap();
        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;

        // Model an open whose recovery scan failed: byte totals unknown, flag set.
        store.index_tables.write().collection_disk_bytes.clear();
        store
            .shard_collections_recovery_failed
            .store(true, Ordering::Release);

        // While the scan still fails (an unreadable pack), nothing is written.
        let junk = dir.join("pack_00000000000000ff.pack");
        fs::write(&junk, b"not a packfile").unwrap();
        assert!(store.persist_shard_collections().is_err());
        assert!(store
            .shard_collections_recovery_failed
            .load(Ordering::Acquire));
        assert_eq!(fs::read(&sidecar).unwrap(), before, "no zero-byte sidecar");

        // Once the scan can succeed, the next persist retries it and recovers.
        fs::remove_file(&junk).unwrap();
        store.persist_shard_collections().unwrap();
        assert!(!store
            .shard_collections_recovery_failed
            .load(Ordering::Acquire));
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected)
        );
    }

    /// Deterministic version of the recovery/append race: hold recovery between
    /// its scan and the swap of its totals and force a writer into that window.
    /// The writer must be blocked by the collection mutexes; without them its
    /// frame would be appended after the scan and then be lost by the swap.
    #[test]
    fn a_writer_cannot_slip_into_the_recovery_scan_to_swap_window() {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::Duration;

        let wait = Duration::from_secs(10);
        let dir = test_dir("sidecar_recovery_window");
        let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        store.index_tables.write().collection_disk_bytes.clear();
        store
            .shard_collections_recovery_failed
            .store(true, Ordering::Release);

        let (scanned_tx, scanned_rx) = channel::<()>();
        let (resume_tx, resume_rx) = channel::<()>();
        let resume_rx = parking_lot::Mutex::new(resume_rx);
        *store.recovery_pause_hook.lock() = Some(Arc::new(move || {
            scanned_tx.send(()).unwrap();
            resume_rx
                .lock()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }));

        let recovery = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || store.persist_shard_collections().unwrap())
        };
        // Recovery has scanned and is paused, still holding its locks.
        scanned_rx.recv_timeout(wait).unwrap();

        let (started_tx, started_rx) = channel::<()>();
        let (done_tx, done_rx) = channel::<()>();
        let writer = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                store
                    .put(
                        &TEST_COLLECTION,
                        &distinct_id(1),
                        &NodeData::new(bytes::Bytes::from_static(b"written during recovery")),
                    )
                    .unwrap();
                done_tx.send(()).unwrap();
            })
        };
        // Only start the clock once the writer is running, so the timeout
        // measures lock blocking rather than thread scheduling.
        started_rx.recv_timeout(wait).unwrap();
        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(1)),
            Err(RecvTimeoutError::Timeout),
            "a writer must be blocked while recovery holds the collection locks"
        );

        resume_tx.send(()).unwrap();
        recovery.join().unwrap();
        done_rx
            .recv_timeout(wait)
            .expect("writer must finish once recovery ends");
        writer.join().unwrap();

        store.sync_all().unwrap();
        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "the frame written around recovery must be in the persisted totals"
        );
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// A put whose insert triggers index growth takes a different code path;
    /// it must still account for its frame's bytes.
    #[test]
    fn puts_and_batches_that_grow_the_index_are_byte_accounted() {
        let dir = test_dir("growth_byte_accounting");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let collection = [0x43u8; 16];
        for i in 0..200u8 {
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(vec![b'x'; 64])),
                )
                .unwrap();
        }
        let batch: Vec<_> = (0..200u8)
            .map(|i| {
                (
                    distinct_id(i),
                    NodeData::new(bytes::Bytes::from(vec![b'y'; 64])),
                )
            })
            .collect();
        assert_eq!(store.put_many(&collection, &batch).unwrap(), 200);

        // A batch into an EXISTING small collection is what grows its index
        // mid-batch (a fresh collection is pre-sized for the batch).
        let grown_collection = [0x44u8; 16];
        store
            .put(
                &grown_collection,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
        let (_, _, capacity_before) = store.collection_index_info(&grown_collection).unwrap();
        let batch: Vec<_> = (1..250u8)
            .map(|i| {
                (
                    distinct_id(i),
                    NodeData::new(bytes::Bytes::from(vec![b'z'; 64])),
                )
            })
            .collect();
        assert_eq!(store.put_many(&grown_collection, &batch).unwrap(), 249);
        let (_, _, capacity_after) = store.collection_index_info(&grown_collection).unwrap();
        assert!(
            capacity_after > capacity_before,
            "the batch must grow the index ({capacity_before} -> {capacity_after})"
        );
        store.sync_all().unwrap();

        let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
        let sidecar = PackfileStorage::collection_disk_bytes_from_disk(&dir).unwrap();
        for id in [TEST_COLLECTION, collection, grown_collection] {
            assert_eq!(
                sidecar.get(&id),
                Some(&physical.collections[&id].disk_bytes),
                "collection {id:02x?}: growth must not drop a frame's bytes"
            );
        }
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// Recovery must be atomic with appends: frames written while it runs may
    /// not be left out of (or double counted in) the totals it publishes.
    #[test]
    fn sidecar_recovery_is_consistent_with_concurrent_appends() {
        let dir = test_dir("sidecar_recovery_concurrent");
        let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();

        store.index_tables.write().collection_disk_bytes.clear();
        store
            .shard_collections_recovery_failed
            .store(true, Ordering::Release);

        let writer = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for i in 1..200u8 {
                    store
                        .put(
                            &TEST_COLLECTION,
                            &distinct_id(i),
                            &NodeData::new(bytes::Bytes::from(vec![b'x'; 64])),
                        )
                        .unwrap();
                }
            })
        };
        // Recover while the writer is appending.
        while store
            .shard_collections_recovery_failed
            .load(Ordering::Acquire)
        {
            let _ = store.persist_shard_collections();
        }
        writer.join().unwrap();
        store.sync_all().unwrap();

        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "totals published by recovery must match the frames on disk"
        );
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint that survives without its shard→collection sidecar must not
    /// leave the sidecar missing (no write dirties the store to regenerate it)
    /// nor persist zero byte totals for data that is already on disk.
    #[test]
    fn missing_sidecar_is_rewritten_with_exact_bytes_when_the_checkpoint_survives() {
        let dir = test_dir("sidecar_missing_checkpoint_present");
        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            let id = distinct_id(0);
            for payload in [&b"first"[..], &b"second!!"[..]] {
                store
                    .put(
                        &TEST_COLLECTION,
                        &id,
                        &NodeData::new(bytes::Bytes::copy_from_slice(payload)),
                    )
                    .unwrap();
            }
            store.sync_all().unwrap();
        }
        let expected = crate::packfile::layout::physical_layout(&dir)
            .unwrap()
            .collections[&TEST_COLLECTION]
            .disk_bytes;
        let sidecar = PackfileStorage::shard_collections_path(&dir);
        fs::remove_file(&sidecar).unwrap();
        assert!(PackfileStorage::index_checkpoint_path(&dir).exists());

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        reopened.sync_all().unwrap();
        assert!(sidecar.exists(), "sync must regenerate the lost sidecar");
        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            vec![(TEST_COLLECTION, 1)]
        );
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "regenerated bytes must equal the physical layout, not zero"
        );
        drop(reopened);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collection_disk_bytes_track_overwrite_reopen_and_repack() {
        let dir = test_dir("collection_disk_bytes_semantics");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let id = distinct_id(0);
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"first")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"second")),
            )
            .unwrap();
        store.sync_all().unwrap();

        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            vec![(TEST_COLLECTION, 1)],
            "overwriting a key must not inflate the live-node count"
        );
        let physical = crate::packfile::layout::physical_layout(&dir).unwrap();
        let expected = physical
            .collections
            .get(&TEST_COLLECTION)
            .expect("collection in physical layout")
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "sidecar bytes must include both the original and overwrite frames"
        );

        drop(store);
        let checkpoint_reopened = PackfileStorage::open(dir.clone()).unwrap();
        checkpoint_reopened.sync_all().unwrap();
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "checkpoint-backed reopen must preserve physical byte accounting"
        );
        drop(checkpoint_reopened);
        fs::remove_file(PackfileStorage::index_checkpoint_path(&dir)).unwrap();
        fs::remove_file(PackfileStorage::shard_collections_path(&dir)).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        reopened.sync_all().unwrap();
        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            vec![(TEST_COLLECTION, 1)],
            "a full rescan must preserve the live-node count"
        );
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&expected),
            "reopen must preserve physical byte accounting"
        );
        reopened
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        reopened.sync_all().unwrap();
        let repacked_physical = crate::packfile::layout::physical_layout(&dir).unwrap();
        let repacked_expected = repacked_physical
            .collections
            .get(&TEST_COLLECTION)
            .expect("collection after repack")
            .disk_bytes;
        assert_eq!(
            PackfileStorage::collection_disk_bytes_from_disk(&dir)
                .unwrap()
                .get(&TEST_COLLECTION),
            Some(&repacked_expected),
            "repack must update physical byte accounting"
        );
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
        let (slot, offset) = store
            .generation(&TEST_COLLECTION)
            .unwrap()
            .index
            .lookup(&id)
            .expect("just-written record must be indexed");

        // Duplicate ids in the input must collapse to one pinned entry.
        let pinned = store.pin_shards([slot, slot, slot].into_iter());
        assert_eq!(pinned.len(), 1);

        let record = store
            .read_at(&pinned[&slot], offset, true)
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

    /// Collect the metadata of every record in every pack file under `dir`.
    fn all_record_metadata(dir: &std::path::Path) -> Vec<FrameMetadata> {
        use std::io::{Seek, SeekFrom};

        let mut found = Vec::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "pack") {
                let mut file = fs::File::open(&path).unwrap();
                file.seek(SeekFrom::Start(packfile::HEADER_LEN as u64))
                    .unwrap();
                while let Some(record) = packfile::read_record_metadata(&mut file).unwrap() {
                    if let Some(metadata) = record.metadata {
                        found.push(metadata);
                    }
                }
            }
        }
        found
    }

    /// `put_verified` must reject a mismatched digest before writing anything,
    /// and accept a matching digest while attaching the full logical id,
    /// content digest, algorithm, and role to the on-disk frame.
    #[test]
    fn put_verified_checks_digest_and_attaches_metadata() {
        let dir = test_dir("put_verified_metadata");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let collection = [0x31; 16];
        let id = [0x32; 16];
        let payload = b"canonical event bytes";
        let data = NodeData::new(bytes::Bytes::from_static(payload));
        let logical_id = [0x77; 32];
        let digest = crate::storage::content_digest(DigestAlgorithm::Sha256, payload);

        // A wrong expected digest is rejected and leaves no record behind.
        let mut wrong = digest;
        wrong[0] ^= 0xff;
        let error = store
            .put_verified(
                &collection,
                &id,
                &data,
                &logical_id,
                DigestAlgorithm::Sha256,
                &wrong,
                Some(b"event"),
            )
            .unwrap_err();
        assert!(matches!(error, StorageError::Corrupt(_)), "got {error:?}");
        assert!(store.get(&collection, &id).unwrap().is_none());
        assert_eq!(all_record_metadata(&dir), Vec::new());

        // The matching digest succeeds.
        store
            .put_verified(
                &collection,
                &id,
                &data,
                &logical_id,
                DigestAlgorithm::Sha256,
                &digest,
                Some(b"event"),
            )
            .unwrap();
        store.sync().unwrap();

        assert_eq!(
            store.get(&collection, &id).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(payload))
        );

        let metadatas = all_record_metadata(&dir);
        assert_eq!(metadatas.len(), 1);
        let metadata = &metadatas[0];
        assert_eq!(metadata.logical_id, Some(logical_id));
        assert_eq!(metadata.content_digest, Some(digest));
        assert_eq!(metadata.digest_algorithm, DigestAlgorithm::Sha256);
        assert_eq!(metadata.role.as_deref(), Some(&b"event"[..]));

        // Read path reconstructs metadata too (via `read_record_metadata`).
        drop(store);
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_eq!(
            reopened.get(&collection, &id).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(payload))
        );
        drop(reopened);
        fs::remove_dir_all(&dir).ok();
    }

    /// A generic `put` (metadata `None`) must not grow frames or set the
    /// metadata flag, so existing workloads see byte-identical records.
    #[test]
    fn plain_put_writes_no_metadata() {
        let dir = test_dir("plain_put_no_metadata");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let collection = [0x41; 16];
        let id = [0x42; 16];
        store
            .put(
                &collection,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"plain")),
            )
            .unwrap();
        store.sync().unwrap();

        assert_eq!(all_record_metadata(&dir), Vec::new());
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// A writer that opens without a usable checkpoint (e.g. after an aborted
    /// import that never synced) must persist one on its next sync even though
    /// it wrote nothing itself; otherwise every later open rescans every pack.
    #[test]
    fn rescan_open_persists_a_checkpoint_on_the_next_sync() {
        let dir = test_dir("rescan_open_checkpoint");
        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    &[0x11u8; 16],
                    &NodeData::new(bytes::Bytes::from_static(b"payload")),
                )
                .unwrap();
            store.sync_all().unwrap();
        }
        let checkpoint = PackfileStorage::index_checkpoint_path(&dir);
        fs::remove_file(&checkpoint).unwrap();

        let store = PackfileStorage::open(dir.clone()).unwrap();
        assert!(!checkpoint.exists(), "the rescan open alone writes none");
        store.sync_all().unwrap();
        assert!(
            checkpoint.exists(),
            "a sync after a rescan open must persist the checkpoint"
        );
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collection_len_counts_genesis_and_distinct_ids() {
        use crate::template::{
            CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        };

        let dir = test_dir("collection_len");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let collection = [0x7Au8; 16];
        assert_eq!(store.collection_len(&collection).unwrap(), None);

        let metadata = CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
        };
        store
            .ensure_collection_metadata(&collection, &metadata)
            .unwrap();
        assert_eq!(store.collection_len(&collection).unwrap(), Some(1));

        let a = [0xA1u8; 16];
        let b = [0xB2u8; 16];
        store
            .put(
                &collection,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"one")),
            )
            .unwrap();
        store
            .put(
                &collection,
                &b,
                &NodeData::new(bytes::Bytes::from_static(b"two")),
            )
            .unwrap();
        assert_eq!(store.collection_len(&collection).unwrap(), Some(3));

        // Overwriting an id appends a frame but does not add a record.
        store
            .put(
                &collection,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"uno")),
            )
            .unwrap();
        assert_eq!(store.collection_len(&collection).unwrap(), Some(3));
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// An empty batch must be a no-op across every backend: no collection is
    /// created, so `collection_exists`/`collection_len` report absence. This
    /// mirrors the in-memory-backend test of the same name; the two backends
    /// previously disagreed (`InMemoryStorage` left an empty entry behind).
    #[test]
    fn empty_put_many_is_a_noop_and_does_not_create_the_collection() {
        let dir = test_dir("empty_put_many_noop");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let collection = [0x7Du8; 16];
        assert_eq!(store.put_many(&collection, &[]).unwrap(), 0);
        assert!(!store.collection_exists(&collection));
        assert_eq!(store.collection_len(&collection).unwrap(), None);
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// Concurrent genesis establishment on a real packfile-backed store must
    /// serialize on the collection `put_mutex`: without it, two writers can
    /// both observe an absent metadata record and append conflicting genesis
    /// frames. `InMemoryStorage`'s equivalent is trivially serialized by its
    /// single map write lock; this exercises the locking path that is not.
    ///
    /// Scope: intra-process only. The `put_mutex` map is per-instance, so it
    /// says nothing about writers in separate processes. Cross-process writers
    /// are excluded by the shard pool's exclusive `.mtxdb.lock` writer lock
    /// (`ShardPool::acquire_writer_lock`), which permits one writer process per
    /// store; this test pins the thread-level race.
    #[test]
    fn ensure_collection_metadata_is_atomic_under_concurrency() {
        use crate::template::{
            CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        };

        let dir = test_dir("ensure_metadata_concurrent");
        let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
        let collection = [0x7Eu8; 16];
        let metadata = Arc::new(CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
        });

        // Release every thread at once so the lookup/append windows overlap.
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                let metadata = Arc::clone(&metadata);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store
                        .ensure_collection_metadata(&collection, &metadata)
                        .expect("concurrent genesis establishment must be idempotent");
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(
            store.get_collection_metadata(&collection).unwrap(),
            Some((*metadata).clone())
        );
        assert_eq!(store.collection_len(&collection).unwrap(), Some(1));
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// Writers racing with *different* genesis metadata must not both win:
    /// exactly one establishes the collection and every other writer sees the
    /// mismatch. Without the collection's `put_mutex` held across the lookup
    /// and append, several writers could each append their own genesis frame.
    #[test]
    fn ensure_collection_metadata_conflicting_writers_have_one_winner() {
        use crate::template::{
            CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
        };

        let dir = test_dir("ensure_metadata_conflict");
        let store = Arc::new(PackfileStorage::open(dir.clone()).unwrap());
        let collection = [0x7Fu8; 16];
        let candidates: Vec<CollectionMetadata> = (0..8)
            .map(|i| CollectionMetadata {
                pool_dst: Some(*b"EVNT"),
                collection_canonical_id: format!("!room{i}:matrix.org").into_bytes(),
                record_id_rule: RecordIdentityRule {
                    policy: FrameIdPolicy::Pointer {
                        pointer: "/event_id".into(),
                    },
                    digest_algorithm: DigestAlgorithm::Sha256,
                },
                payload: PayloadPolicy::Source,
                extension: None,
            })
            .collect();

        let barrier = Arc::new(std::sync::Barrier::new(candidates.len()));
        let handles: Vec<_> = candidates
            .iter()
            .cloned()
            .map(|metadata| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.ensure_collection_metadata(&collection, &metadata)
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(
            results.iter().filter(|r| r.is_ok()).count(),
            1,
            "exactly one conflicting genesis writer may succeed: {results:?}"
        );
        for result in results.iter().filter(|r| r.is_err()) {
            assert!(
                matches!(result, Err(StorageError::Internal(msg)) if msg.contains("mismatch")),
                "losers must report a metadata mismatch, got {result:?}"
            );
        }
        let stored = store
            .get_collection_metadata(&collection)
            .unwrap()
            .expect("the winner's genesis record is stored");
        assert!(candidates.contains(&stored));
        assert_eq!(store.collection_len(&collection).unwrap(), Some(1));
        drop(store);
        fs::remove_dir_all(&dir).ok();
    }

    /// Checkpoint-backed hash recovery must handle frames that carry metadata
    /// (as `put_verified` writes do). `record_identity_at` previously rejected
    /// `FLAG_METADATA` frames, so `grow_checkpoint_index` failed closed to a
    /// full rescan instead of recovering their identity.
    #[test]
    fn checkpoint_growth_recovers_metadata_bearing_frames() {
        let dir = test_dir("checkpoint_growth_metadata");
        let collection = TEST_COLLECTION;
        let id = [0x33u8; 16];
        let payload = bytes::Bytes::from_static(b"metadata payload");
        let digest = DigestAlgorithm::Sha256.digest(&payload);

        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put_verified(
                &collection,
                &id,
                &NodeData::new(payload),
                &digest,
                DigestAlgorithm::Sha256,
                &digest,
                None,
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let checkpoint_index = reopened
            .generation(&collection)
            .expect("checkpoint collection exists");
        assert!(checkpoint_index.index.is_mmap_backed());
        let grown = reopened
            .grow_checkpoint_index(&collection, &checkpoint_index.index)
            .unwrap()
            .expect("checkpoint index can grow");
        assert!(
            grown.lookup(&id).is_some(),
            "a metadata-bearing frame's identity must be recoverable"
        );
        drop(checkpoint_index);
        drop(reopened);
        fs::remove_dir_all(&dir).ok();
    }
}
