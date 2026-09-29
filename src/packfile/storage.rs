use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::{MutexGuard, RwLock};

use crate::cache::{NodeCache, PinnedNodes};
use crate::csr::Csr;
use crate::index::delta::{self, DeltaOperation, DELTA_LOG_HEADER_LEN, INDEX_DELTA_FILE};
use crate::index::format::DeltaFrame;
use crate::index::redo::{RedoOp, RedoRecord};
use crate::index::{EntryUndo, InsertError, LossyIndex};
#[cfg(feature = "multi-reader")]
use crate::journal::pool_tag;
use crate::journal::{
    pool_from_tag, DurabilityToken, GroupCommitConfig, Journal, JournalCoordinator,
    JournalReplayLease, Mutation as JournalMutation,
};
use crate::packfile::{self, FrameMetadata, PackId, Record};
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{
    collect_missing_established_records, hex16, validate_established_batch_inputs,
    validate_established_upsert_inputs, Digest32, DigestAlgorithm, NodeData, NodeId, NodeRef,
    StorageEngine, StorageError,
};
use crate::template::{CollectionMetadata, COLLECTION_METADATA_RECORD_ID};

mod read_journal;
use read_journal::ReadJournal;

thread_local! {
    /// Suppresses journal publication while a committed transaction applies
    /// its already-staged mutations to the live pack/index state. The outer
    /// transaction publishes the complete batch exactly once afterward.
    static JOURNAL_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
    /// Set while a read-committed lookup falls back to the live index. That
    /// fallback reads through `get_many`, which redirects to the overlay path
    /// while a transaction is active, so without this flag the two would call
    /// each other until the stack overflowed.
    static READ_COMMITTED_FALLBACK: Cell<bool> = const { Cell::new(false) };
}

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
    /// Grouping the replayed operations by collection and applying snapshot,
    /// tombstone and generation gates, before any index is materialized
    /// (checkpoint path only).
    pub replay_prepare: std::time::Duration,
    /// Seeding per-shard counts and home-shard assignment after the indexes are
    /// materialized: the sidecar gate, or a slot walk (checkpoint path only).
    pub bookkeeping: std::time::Duration,
    /// Assembling the `PackfileStorage` from the loaded state, after the index
    /// is built.
    pub assemble: std::time::Duration,
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
            replay_prepare: std::time::Duration::ZERO,
            bookkeeping: std::time::Duration::ZERO,
            assemble: std::time::Duration::ZERO,
            delta_replay_operations: 0,
            full_scan: std::time::Duration::ZERO,
            total: std::time::Duration::ZERO,
            path: OpenPath::FullScan,
            bookkeeping_source: BookkeepingSource::SlotScan,
        }
    }
}

/// Where the time of one full index checkpoint went, phase by phase, so a
/// slow rewrite (the delta-log rotation, a forced reclaim) can be attributed
/// before anything is redesigned. Read through
/// [`PackfileStorage::checkpoint_breakdown`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckpointBreakdown {
    /// The fsync of unsynced pack bytes done before any lock is taken.
    pub pre_sync: std::time::Duration,
    /// Waiting for every collection's put mutex and the creation lock.
    pub lock_wait: std::time::Duration,
    /// The pack flush and fsync under the locks (writers stalled).
    pub locked_sync: std::time::Duration,
    /// Fingerprint, snapshots, delta-epoch rotation and the coverage capture,
    /// under the locks.
    pub snapshot: std::time::Duration,
    /// Serializing every collection's index into the checkpoint blobs.
    pub serialize: std::time::Duration,
    /// Writing the checkpoint, its fsync and rename.
    pub write: std::time::Duration,
    /// The directory fsync that makes the rename durable.
    pub directory_sync: std::time::Duration,
    /// Writing `journal.lsn`.
    pub journal_lsn: std::time::Duration,
    /// Reporting coverage and reclaiming the WAL (its rewrite and fsyncs).
    pub reclaim: std::time::Duration,
    /// Retiring the previous delta epoch.
    pub retire: std::time::Duration,
    /// The whole call.
    pub total: std::time::Duration,
    /// Size of the checkpoint file this call wrote, in bytes (0 if it could not
    /// be read back).
    pub checkpoint_bytes: u64,
}

impl CheckpointBreakdown {
    /// Time in the call that belongs to no phase above: the final re-lock and
    /// the bookkeeping around the phases. A large value means the attribution
    /// is missing something.
    #[must_use]
    pub fn unaccounted(&self) -> std::time::Duration {
        [
            self.pre_sync,
            self.lock_wait,
            self.locked_sync,
            self.snapshot,
            self.serialize,
            self.write,
            self.directory_sync,
            self.journal_lsn,
            self.reclaim,
            self.retire,
        ]
        .into_iter()
        .fold(self.total, std::time::Duration::saturating_sub)
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
    /// Coverage reporting and WAL reclaim performed by a coverage-step delta
    /// batch (not by a checkpoint tail, whose phases are in
    /// [`CheckpointBreakdown`]). This is the shared-segment rewrite and fsync
    /// that shrinks the journal once every pool has reported.
    pub reclaim: std::time::Duration,
    /// Breakdown of [`Self::reclaim`]: deciding the shared-reclaim boundary
    /// (directory lookup, or a scan when the directory could not be trusted).
    pub reclaim_boundary: std::time::Duration,
    /// Breakdown of [`Self::reclaim`]: reading and re-encoding the retained
    /// suffix into the rebuilt segment.
    pub reclaim_copy: std::time::Duration,
    /// Breakdown of [`Self::reclaim`]: writing, fsyncing and renaming the
    /// rebuilt segment into place. The likely disk cost.
    pub reclaim_fsync: std::time::Duration,
    /// Bytes the last reclaim rewrite moved and fsynced: the retained suffix.
    /// This is what `reclaim_copy`/`reclaim_fsync` scale with.
    pub reclaim_retained_bytes: u64,
    /// The pool whose missing coverage stopped the last reclaim cut, so the
    /// suffix could not be dropped sooner. None when an untagged frame stopped
    /// it or the caller could not attribute it. Diagnostics only.
    pub reclaim_blocked_by: Option<crate::layout::ShardType>,
    /// Time spent in `remediate_lagging_pools`: in the non-background path this
    /// force-checkpoints a pool holding the shared WAL back, and the caller
    /// waits for it. Kept separate from [`Self::reclaim`] so a phase breakdown
    /// can tell a coverage-step reclaim from a remediation checkpoint (whose
    /// own journal wait would otherwise be miscounted as reclaim).
    pub remediation: std::time::Duration,
    /// Committing the pending write-ahead journal group (one sequential
    /// fsync). Zero when no journal is configured.
    pub wal: std::time::Duration,
    /// Time spent waiting for the journal's single-writer mutex.
    pub journal_lock_wait: std::time::Duration,
    /// Time spent making the journal file durable.
    pub journal_fsync: std::time::Duration,
    /// Number of journal sync calls represented by this operation.
    pub journal_sync_calls: u64,
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
            reclaim: std::time::Duration::ZERO,
            reclaim_boundary: std::time::Duration::ZERO,
            reclaim_copy: std::time::Duration::ZERO,
            reclaim_fsync: std::time::Duration::ZERO,
            reclaim_retained_bytes: 0,
            reclaim_blocked_by: None,
            remediation: std::time::Duration::ZERO,
            wal: std::time::Duration::ZERO,
            journal_lock_wait: std::time::Duration::ZERO,
            journal_fsync: std::time::Duration::ZERO,
            journal_sync_calls: 0,
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

impl SyncTimings {
    /// Copy a coverage-step reclaim's phase breakdown into this barrier's
    /// timings, so the `reclaim` total can be split into boundary, copy and
    /// fsync. A no-op when no reclaim ran.
    fn record_reclaim_phases(&mut self, reclaim: Option<&crate::journal::Reclaim>) {
        if let Some(reclaim) = reclaim {
            self.reclaim_boundary = reclaim.boundary;
            self.reclaim_copy = reclaim.copy;
            self.reclaim_fsync = reclaim.fsync;
            self.reclaim_retained_bytes = reclaim.retained_bytes;
            self.reclaim_blocked_by = reclaim.blocked_by;
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
    journal_fsync_ns: AtomicU64,
    journal_sync_calls: AtomicU64,
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
        Self::add_duration(&self.journal_fsync_ns, timings.journal_fsync);
        self.journal_sync_calls
            .fetch_add(timings.journal_sync_calls, Ordering::Relaxed);
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
            journal_fsync: duration(&self.journal_fsync_ns),
            journal_sync_calls: self.journal_sync_calls.load(Ordering::Relaxed),
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
            &self.journal_fsync_ns,
            &self.journal_sync_calls,
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
///
/// Marked `#[non_exhaustive]` so new cumulative phases can be added without
/// breaking downstream source builds. Read fields by name; construct test
/// fixtures with [`SyncTotalsSnapshot::default`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Cumulative journal fsync time.
    pub journal_fsync: std::time::Duration,
    /// Number of journal sync calls.
    pub journal_sync_calls: u64,
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
///
/// Buckets are **non-cumulative**: `buckets[i]` counts only the observations
/// that fell inside bucket `i`'s own range, so the total observation count is
/// the sum of all five entries, not the last one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncLatencyHistogram {
    /// Per-bucket observation counts, indexed to match the ranges above
    /// (`buckets[0]` = `<1ms` … `buckets[4]` = `>=1s`). Non-cumulative.
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
#[non_exhaustive]
pub struct SyncDiagnosticSample {
    /// Unix timestamp in milliseconds when the sync completed.
    pub timestamp_ms: u128,
    /// OS process ID that performed or coordinated the sync.
    pub process_id: u32,
    /// Path to the active journal segment file, if journal-backed.
    pub journal_path: Option<String>,
    /// Total duration of the sync operation.
    pub total: std::time::Duration,
    /// Whether the sync operation encountered an error or failed.
    pub failed: bool,
    /// Duration spent writing buffered packfile frames to disk.
    pub pack_flush: std::time::Duration,
    /// Duration spent issuing fsync on dirty packfiles.
    pub pack_fsync: std::time::Duration,
    /// Duration spent syncing collection sidecar state.
    pub sidecar: std::time::Duration,
    /// Duration spent appending to or flushing the index delta log.
    pub delta_log: std::time::Duration,
    /// Duration spent writing an index checkpoint.
    pub checkpoint: std::time::Duration,
    /// Duration spent waiting for the pool-wide dirty shard set lock.
    pub dirty_lock_wait: std::time::Duration,
    /// Age of the oldest pending uncommitted frame included in this sync.
    pub pending_publish_age: std::time::Duration,
    /// Duration spent on WAL sync (if WAL is enabled).
    pub wal: std::time::Duration,
    /// Duration spent waiting to acquire the journal lock.
    pub journal_lock_wait: std::time::Duration,
    /// Duration spent issuing fsync on the journal file.
    pub journal_fsync: std::time::Duration,
    /// Number of journal records committed by this sync barrier.
    pub journal_records: u64,
    /// Number of concurrent journal transactions in flight at sync time.
    pub journal_in_flight: u64,
    /// Number of concurrent callers waiting on this journal sync barrier.
    pub journal_waiters: u64,
    /// Number of sync callers whose sync was coalesced into this single physical fsync.
    pub journal_coalesced: u64,
}

/// Runtime sync diagnostics retained after the operation that produced them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SyncDiagnosticsSnapshot {
    /// Non-cumulative [`SyncLatencyHistogram`] of journal fsync latencies
    /// observed during the interval ending at the take. See
    /// [`SyncLatencyHistogram::buckets`] for bucket membership.
    pub fsync_latency: SyncLatencyHistogram,
    /// Non-cumulative [`SyncLatencyHistogram`] of journal lock-wait latencies
    /// observed during the interval ending at the take. See
    /// [`SyncLatencyHistogram::buckets`] for bucket membership.
    pub lock_wait_latency: SyncLatencyHistogram,
    /// Maximum journal in-flight count observed during this snapshot.
    /// Unlike the lifetime maxima in [`SyncTotalsSnapshot`], this value is
    /// cleared by [`PackfileStorage::take_sync_diagnostics`].
    pub peak_journal_in_flight: u64,
    /// Maximum journal lock wait observed during this snapshot.
    /// Unlike the lifetime maxima in [`SyncTotalsSnapshot`], this value is
    /// cleared by [`PackfileStorage::take_sync_diagnostics`].
    pub max_journal_lock_wait: std::time::Duration,
    /// Maximum journal fsync duration observed during this snapshot.
    /// Unlike the lifetime maxima in [`SyncTotalsSnapshot`], this value is
    /// cleared by [`PackfileStorage::take_sync_diagnostics`].
    pub max_journal_fsync: std::time::Duration,
    /// The up-to-16 slowest sync samples observed during this snapshot,
    /// ordered from slowest to fastest. This is interval-scoped and bounded,
    /// not a lifetime history.
    pub worst_syncs: Vec<SyncDiagnosticSample>,
}

#[derive(Default)]
struct SyncDiagnostics {
    fsync_latency: SyncLatencyHistogram,
    lock_wait_latency: SyncLatencyHistogram,
    peak_journal_in_flight: u64,
    max_journal_lock_wait: std::time::Duration,
    max_journal_fsync: std::time::Duration,
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
            self.max_journal_lock_wait = self.max_journal_lock_wait.max(sample.journal_lock_wait);
            self.max_journal_fsync = self.max_journal_fsync.max(sample.journal_fsync);
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
            max_journal_lock_wait: self.max_journal_lock_wait,
            max_journal_fsync: self.max_journal_fsync,
            worst_syncs: self.worst_syncs.clone(),
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Fixed latency buckets used by public operation diagnostics. Buckets are
/// non-cumulative: `<50us`, `<100us`, `<250us`, `<1ms`, `<10ms`, and `>=10ms`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OperationLatency {
    /// Number of timed operations.
    pub calls: u64,
    /// Sum of operation wall time.
    pub total: std::time::Duration,
    /// Largest observed operation wall time.
    pub max: std::time::Duration,
    /// Counts for `<50us`, `<100us`, `<250us`, `<1ms`, `<10ms`, and `>=10ms`.
    pub buckets: [u64; 6],
}

#[derive(Default)]
struct OperationLatencyTotals {
    calls: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
    buckets: [AtomicU64; 6],
}

impl OperationLatencyTotals {
    fn observe(&self, duration: std::time::Duration) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(
            u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.max_ns.fetch_max(
            u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let micros = duration.as_micros();
        let bucket = if micros < 50 {
            0
        } else if micros < 100 {
            1
        } else if micros < 250 {
            2
        } else if micros < 1_000 {
            3
        } else if micros < 10_000 {
            4
        } else {
            5
        };
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> OperationLatency {
        OperationLatency {
            calls: self.calls.load(Ordering::Relaxed),
            total: std::time::Duration::from_nanos(self.total_ns.load(Ordering::Relaxed)),
            max: std::time::Duration::from_nanos(self.max_ns.load(Ordering::Relaxed)),
            buckets: std::array::from_fn(|index| self.buckets[index].load(Ordering::Relaxed)),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.total_ns.store(0, Ordering::Relaxed);
        self.max_ns.store(0, Ordering::Relaxed);
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

#[derive(Default)]
struct OperationTimings {
    get: OperationLatencyTotals,
    get_many: OperationLatencyTotals,
    get_many_with_refresh: OperationLatencyTotals,
    put: OperationLatencyTotals,
    put_many: OperationLatencyTotals,
}

impl OperationTimings {
    fn reset(&self) {
        self.get.reset();
        self.get_many.reset();
        self.get_many_with_refresh.reset();
        self.put.reset();
        self.put_many.reset();
    }
}

struct OperationTimer<'a> {
    started: Option<std::time::Instant>,
    totals: Option<&'a OperationLatencyTotals>,
}

impl<'a> OperationTimer<'a> {
    fn disabled() -> Self {
        Self {
            started: None,
            totals: None,
        }
    }

    fn new(totals: &'a OperationLatencyTotals) -> Self {
        Self {
            started: Some(std::time::Instant::now()),
            totals: Some(totals),
        }
    }
}

impl Drop for OperationTimer<'_> {
    fn drop(&mut self) {
        if let (Some(totals), Some(started)) = (self.totals, self.started) {
            totals.observe(started.elapsed());
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
type RepackCopiedRecord = ([u8; 16], u16, u64, u64);
/// Return type of [`PackfileStorage::repack_scan_incremental`]: the merged
/// live-location map and (on the incremental path) the previous cursor state
/// that the adjacency helpers need to skip re-reading already-seen nodes.
type RepackScanResult = (RepackRecordMap, Option<RepackIncrementalState>);

/// One scanned record's `(slot, hash, offset, pack_id)`, as accumulated per
/// collection during `PackfileStorage::open_with_options`'s initial scan.
type ShardRecord = (u16, [u8; 16], u64, PackId);
/// `(collection_id, entry count, index memory usage in bytes, index capacity)`.
pub type CollectionSummary = ([u8; 16], usize, usize, u32);

/// Accumulator for `PackfileStorage::open_with_options`'s phase 2 —
/// bundles the three maps `init_collection_from_scan` fills in per collection, so
/// that function takes one out-parameter instead of three.
#[derive(Default)]
struct RoomScanOutput {
    collections: HashMap<[u8; 16], ArcSwap<RoomGeneration>>,
    shard_collections: HashMap<PackId, HashMap<[u8; 16], u64>>,
    collection_shards: HashMap<[u8; 16], HashSet<PackId>>,
    collection_disk_bytes: HashMap<PackId, HashMap<[u8; 16], u64>>,
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
) -> HashMap<PackId, HashMap<[u8; 16], u64>> {
    let mut bytes: HashMap<PackId, HashMap<[u8; 16], u64>> = HashMap::new();
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
/// index checkpoint. v7 uses the 16-byte [`PackId`] identity (v6 used the
/// 32-byte form); the record width changed, so the version must advance and a
/// v6 file is rejected and rebuilt.
const SHARD_ROOMS_VERSION: u8 = 7;
/// Header size: magic(4) + version(1) + `pack_fingerprint(8)` + `persisted_at(8)`.
const SHARD_ROOMS_HEADER_LEN: usize = 4 + 1 + 8 + 8;
/// One entry: `pack_id`(16) + `collection_id`(16) + count(8) + the
/// collection's stable insertion ordinal(8) + disk bytes(8).
const SHARD_ROOMS_RECORD_LEN: usize = crate::packfile::PACK_ID_LEN + 16 + 8 + 8 + 8;

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

/// What the disk records of a store's durable index state, read in one pass:
/// the newest journal coverage the checkpoint and its delta log claim, and the
/// fingerprint of the durable pack state they describe (0 if unknown).
#[derive(Debug, Default, Clone, Copy)]
struct DurableIndexState {
    covered_lsn: u64,
    fingerprint: u64,
}

/// A delta batch that does not fit under the delta-log cap.
#[derive(Debug)]
struct DeltaBatchTooLarge {
    batch_bytes: usize,
    log_bytes: u64,
    cap: u64,
}

impl std::fmt::Display for DeltaBatchTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the next delta batch of {} bytes does not fit: log {} bytes, cap {}",
            self.batch_bytes, self.log_bytes, self.cap
        )
    }
}

impl std::error::Error for DeltaBatchTooLarge {}

// The delta-log rotation length is three quarters of the cap, see
// `PackfileStorage::delta_log_rotate_bytes`. It cannot promise the next batch
// fits: a batch can carry whole-index snapshots, so a batch that does not is
// caught when it is built (`is_delta_batch_too_large`) and also rotates.

/// Candidate offsets closer than this on the same shard count as one
/// sequential read run when measuring a `get_many` batch's locality.
/// `record_disk_len` is `frame_len + 8` and frames are at least
/// `FRAME_FIXED_LEN` bytes, so a gap this small means no more than one
/// minimum-size interleaved foreign frame sits between two candidate frames —
/// reading both is effectively a single sequential range. It is a shape
/// heuristic for the optimizer's "one read plan per shard" target, not exact
/// adjacency (that would require a per-candidate frame-length probe).
const READ_RUN_GAP_BYTES: u64 = 128;

/// Largest on-disk size of a single frame: the 4-byte length prefix, a
/// [`packfile::MAX_RECORD_LEN`]-bounded body, and the 4-byte CRC. Used as the
/// per-candidate frame-length estimate when planning merged read extents: it
/// over-approximates so the plan never advises short of a frame's true end,
/// and it avoids a per-candidate length-prefix probe (which would fault in the
/// very cold page the plan exists to prefetch).
const MAX_FRAME_DISK_LEN: u64 = (packfile::MAX_RECORD_LEN + 8) as u64;

/// Tuning for `get_many`'s merged read plan.
///
/// `get_many` probes the lossy index once per requested id, so a large batch
/// yields many candidate `(shard, offset)` locations scattered across a few
/// shards. Without a plan each candidate is read independently; on rotational
/// media the per-candidate seeks dominate. When enabled, candidates on the same
/// shard whose offsets are within [`Self::merge_gap_bytes`] of each other are
/// melded into one contiguous extent, and each extent is prefetched with
/// `madvise(MADV_WILLNEED)` before resolution — the kernel then reads it as one
/// sequential run while the existing per-candidate decode path still verifies
/// every requested hash (lossy-index false positives included).
///
/// # Measured behaviour (cold-cache, 1M records × 1 KiB, careful HDD vs SSD)
///
/// This plan is **rotational-media (HDD) only** and must not be enabled by
/// default anywhere:
///
/// - **Wins on HDD** when target spacing is roughly 700 KiB or more: 1.9-3×
///   faster than the `MADV_RANDOM` baseline, and 2.6-5× faster than
///   independent reads. The win comes from prefetching target neighborhoods,
///   not from reading through gaps — any gap from 0 to ~256 KiB performed about
///   equally.
/// - **Neutral on HDD** by ~174 KiB spacing (all policies ~1.0×): the batch is
///   dense enough that independent reads already touch most pages.
/// - **Loses on SSD** (0.6-0.85× vs `MADV_RANDOM`): there is no seek to avoid,
///   so reading through gaps is pure waste, and the plan reads far more bytes
///   than the readahead-suppressing alternative.
///
/// [`Self::random_advice`] — suppressing kernel readahead with `MADV_RANDOM`
/// and no planning at all — is the portable win: 4.5-6.5× on SSD, 1.4-3.4× on
/// sparse HDD, and neutral on dense HDD. Prefer it unless the workload is a
/// sparse batch on rotational media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPlanPolicy {
    /// Candidate offsets at most this far apart (start to start, on the same
    /// shard) are melded into one prefetched extent. Reading through a gap can
    /// be cheaper than a seek, but a gap far larger than the target spacing
    /// makes the plan read through most of the file for no benefit — measured
    /// on a cold-cache HDD, every gap from 0 to ~256 KiB performed about
    /// equally, and a 2 MiB gap over-merged. [`ReadPlanPolicy::prefetch`] uses
    /// 64 KiB; `0` melds only adjacent offsets.
    pub merge_gap_bytes: u64,
    /// Hard ceiling on one merged extent. Bounds the bytes read through gaps
    /// and keeps a dense batch from collapsing into a single unbounded
    /// prefetch. `0` disables prefetching entirely.
    pub max_extent_bytes: u64,
    /// Batches with fewer candidate locations than this skip planning: a
    /// handful of reads gains nothing from a plan, and the grouping/sort is
    /// pure overhead.
    pub min_batch_candidates: usize,
    /// When set, advise each shard mapping `MADV_RANDOM` before reading, with
    /// no planning at all. A random-access hint suppresses the kernel's
    /// sequential readahead, so a scattered `get_many` reads only the pages
    /// it faults rather than having readahead amplify each one into a large
    /// block read.
    ///
    /// This is the **portable** win and the one to reach for first: measured
    /// at 4.5-6.5× faster than independent reads on SSD, 1.4-3.4× on sparse
    /// HDD, and neutral on dense HDD, with no device-specific tuning. Extent
    /// planning ([`Self::prefetch`]) only beats it on sparse rotational media.
    /// `false` on the presets that plan.
    ///
    /// # Warm and untested cases
    ///
    /// A warm-cache run showed no regression from suppression (random vs
    /// independent reads was 0.94-1.20× at 1-5 ms, i.e. within the spread), so
    /// the cold win does not reverse when data is resident. But only sparse
    /// *point* reads were tested: no warm dense case.
    ///
    /// # Persistent advice, and why this is not a default
    ///
    /// `madvise` is persistent mapping state, and repack/compaction read
    /// records through the *same* shard mmap this sets (`scan_full_adjacency`,
    /// `copy_record_to_shard`), so suppression can leak into them; nothing here
    /// resets it. A repack benchmark (fresh store per pass, 5 passes, 1M
    /// records) measured **no regression** — `REPACK_RANDOM_VS_PLAIN` = 0.998
    /// with fully overlapping ranges — but that store had **no edges**, so the
    /// mmap adjacency walk barely ran and the result mostly reflects
    /// `scan_packfile`, which uses its own file handle and is unaffected. The
    /// mmap walk path is therefore still unmeasured, and this null does not
    /// clear suppression for the default.
    ///
    /// Keep this opt-in. Making it safe by default needs the advice scoped to
    /// the point-read path — a separate mapping/fd for `get_many`, or an
    /// explicit concurrency-safe reset (`MADV_NORMAL`/`SEQUENTIAL`) before
    /// scan/compaction reads. Measurements are from one rotational HDD and one
    /// cheap SATA SSD; `NVMe` and other drives are untested.
    pub random_advice: bool,
}

impl Default for ReadPlanPolicy {
    /// Disabled: the defaults change no behaviour, so nothing regresses on
    /// existing callers. The portable improvement is
    /// [`ReadPlanPolicy::random_advice`] (readahead suppression); extent
    /// planning ([`ReadPlanPolicy::prefetch`]) is rotational-media only. A
    /// store that wants either opts in explicitly.
    fn default() -> Self {
        Self::disabled()
    }
}

impl ReadPlanPolicy {
    /// A plan that never prefetches: no extent is ever built.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            merge_gap_bytes: 0,
            max_extent_bytes: 0,
            min_batch_candidates: usize::MAX,
            random_advice: false,
        }
    }

    /// No extent planning, but advise each shard `MADV_RANDOM` before reading
    /// so kernel readahead does not amplify a scattered batch into large
    /// sequential reads. The cheap baseline to compare a real plan against:
    /// if this alone captures the win, the planner earns nothing.
    #[must_use]
    pub fn random_advice() -> Self {
        Self {
            merge_gap_bytes: 0,
            max_extent_bytes: 0,
            min_batch_candidates: usize::MAX,
            random_advice: true,
        }
    }

    /// A starting point for rotational (HDD) storage: meld candidates within
    /// 64 KiB, up to an 8 MiB extent, once a batch has at least 16 candidate
    /// locations. On a cold-cache HDD benchmark, any gap from 0 up to ~256 KiB
    /// performed about equally well, while the earlier 2 MiB guess over-merged
    /// and read through far more of the file than it saved in seeks. See the
    /// type docs for the measured HDD-only results and why readahead
    /// suppression ([`Self::random_advice`]) is the more portable win.
    #[must_use]
    pub fn prefetch() -> Self {
        Self {
            merge_gap_bytes: 64 * 1024,
            max_extent_bytes: 8 * 1024 * 1024,
            min_batch_candidates: 16,
            random_advice: false,
        }
    }

    /// Whether a batch with `candidate_count` candidate locations should be
    /// planned.
    fn wants(&self, candidate_count: usize) -> bool {
        self.max_extent_bytes > 0 && candidate_count >= self.min_batch_candidates
    }
}

/// One contiguous byte extent on a shard to prefetch as a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadExtent {
    /// Shard slot the extent lives on.
    slot: u16,
    /// Inclusive start offset (a candidate frame's length prefix).
    start: u64,
    /// Exclusive end offset.
    end: u64,
}

/// Meld a shard's sorted candidate offsets into prefetch extents.
///
/// `per_shard` maps each touched shard slot to its candidate offsets, each
/// vector sorted ascending. A run starts at the first offset and absorbs the
/// next while the start-to-start distance stays within
/// [`ReadPlanPolicy::merge_gap_bytes`] and the resulting span stays within
/// [`ReadPlanPolicy::max_extent_bytes`]. A run's end is its last offset plus
/// [`MAX_FRAME_DISK_LEN`] (see that constant for why the frame length is
/// estimated rather than probed).
fn plan_read_extents(
    per_shard: &HashMap<u16, Vec<u64>>,
    policy: ReadPlanPolicy,
) -> Vec<ReadExtent> {
    let mut extents = Vec::new();
    // A single candidate's extent is always `MAX_FRAME_DISK_LEN` long, so a
    // cap below that would split every candidate onto its own extent. Floor it.
    let cap = policy.max_extent_bytes.max(MAX_FRAME_DISK_LEN);
    for (&slot, offsets) in per_shard {
        let Some(&first) = offsets.first() else {
            continue;
        };
        let mut run_start = first;
        let mut run_last = first;
        for &offset in &offsets[1..] {
            let gap_ok = offset.saturating_sub(run_last) <= policy.merge_gap_bytes;
            let span_ok = offset
                .saturating_add(MAX_FRAME_DISK_LEN)
                .saturating_sub(run_start)
                <= cap;
            if gap_ok && span_ok {
                run_last = offset;
            } else {
                extents.push(ReadExtent {
                    slot,
                    start: run_start,
                    end: run_last.saturating_add(MAX_FRAME_DISK_LEN),
                });
                run_start = offset;
                run_last = offset;
            }
        }
        extents.push(ReadExtent {
            slot,
            start: run_start,
            end: run_last.saturating_add(MAX_FRAME_DISK_LEN),
        });
    }
    // `per_shard` is a `HashMap`, so iteration (and therefore extent order)
    // would otherwise be nondeterministic across runs. Prefetch order does not
    // affect correctness, but a stable order keeps plans reproducible and
    // testable.
    extents.sort_unstable_by_key(|extent| (extent.slot, extent.start));
    extents
}

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

/// One persisted `(pack, collection)` bookkeeping record.
#[derive(Clone, Copy)]
pub struct PersistedShardRoom {
    /// Packfile identity.
    pub pack_id: PackId,
    /// Collection identity.
    pub collection_id: [u8; 16],
    /// Number of records attributed to this collection in the pack.
    pub count: u64,
    /// Stable collection insertion order.
    pub insertion_order: u64,
    /// Physical bytes attributed to this collection.
    pub disk_bytes: u64,
}

/// A decoded shard→collection directory: the pack set it was written
/// against, when it was written, and the per-(pack, collection) records.
#[derive(Clone)]
pub struct PersistedShardDirectory {
    /// The `pack_fingerprint` of the `(pack_id, file_len)` set the counts
    /// apply to. Only a directory whose fingerprint equals the index
    /// checkpoint's may serve open's per-shard bookkeeping without re-walking
    /// every slot — any other directory is stale (or corrupt) and must be
    /// ignored in favor of the slot walk.
    pub fingerprint: u64,
    /// Unix-seconds timestamp of when the directory was persisted.
    pub persisted_at: u64,
    /// One record per (pack, collection) the store contains.
    pub records: Vec<PersistedShardRoom>,
}

/// Decode the small inspection sidecar. This is deliberately shared by all
/// read-only CLI summary helpers so they agree on validation and format
/// compatibility.
#[must_use]
pub fn read_persisted_shard_collections(
    base_dir: &std::path::Path,
) -> Option<PersistedShardDirectory> {
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
            // The record is fixed-width, so destructure by const offsets
            // rather than accumulating them: `SR_PACK_ID`(16) +
            // `SR_COLLECTION`(16) + `SR_COUNT`(8) + `SR_INSERTION`(8) +
            // `SR_DISK`(8).
            const SR_PACK_ID: usize = crate::packfile::PACK_ID_LEN;
            const SR_COLLECTION: usize = 16;
            const SR_COUNT: usize = 8;
            const SR_INSERTION: usize = 8;
            let (pack_id_bytes, rest) = chunk.split_at(SR_PACK_ID);
            let (collection_bytes, rest) = rest.split_at(SR_COLLECTION);
            let (count_bytes, rest) = rest.split_at(SR_COUNT);
            let (insertion_bytes, disk_bytes) = rest.split_at(SR_INSERTION);
            let mut pack_id = [0u8; crate::packfile::PACK_ID_LEN];
            pack_id.copy_from_slice(pack_id_bytes);
            let mut collection_id = [0u8; 16];
            collection_id.copy_from_slice(collection_bytes);
            Some(PersistedShardRoom {
                pack_id: PackId(pack_id),
                collection_id,
                count: u64::from_le_bytes(count_bytes.try_into().ok()?),
                insertion_order: u64::from_le_bytes(insertion_bytes.try_into().ok()?),
                disk_bytes: u64::from_le_bytes(disk_bytes.try_into().ok()?),
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

/// Where a just-appended record lives: its shard slot, byte offset, and on-disk
/// length.
#[derive(Clone, Copy)]
struct RecordLocator {
    slot: u16,
    offset: u64,
    len: u64,
}

struct PutManyProgress {
    generation: u64,
    owned_index: Option<LossyIndex>,
    structural_change: bool,
    index_needs_rebuild: bool,
    /// Redo inputs for the entries inserted so far: `(id, slot, offset, record_len)`.
    pending_deltas: Vec<(NodeId, u16, u64, u64)>,
    pending_shard_collections: Vec<(PackId, u64)>,
    pending_shard_collection_counts: Vec<PackId>,
    invalidate_delta: bool,
    undo_log: Vec<EntryUndo>,
}

/// A snapshot-consistent, lazily-read scan over one collection's live records.
///
/// Returned by [`PackfileStorage::scan_collection`]. Records are resolved
/// against a single pinned generation snapshot, so superseded versions that
/// are still physically present in earlier pack segments are never yielded,
/// and records appended after the snapshot boundary are excluded. Each
/// record's payload is read only as the iterator advances, so the whole
/// collection is never buffered.
pub struct CollectionScan<'a> {
    store: &'a PackfileStorage,
    pinned: HashMap<u16, Arc<Shard>>,
    pending: std::vec::IntoIter<ScanWork>,
}

/// One pending emission for a [`CollectionScan`]: either locators to resolve
/// lazily through the pinned generation, or an overlay payload the durable
/// index does not yet contain.
enum ScanWork {
    Locators(NodeId, Vec<(u16, u64)>),
    Data(NodeId, NodeData),
}

impl Iterator for CollectionScan<'_> {
    type Item = Result<(NodeId, NodeData), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.pending.next()? {
                ScanWork::Data(id, data) => return Some(Ok((id, data))),
                ScanWork::Locators(id, candidates) => {
                    match self
                        .store
                        .resolve_from_pinned(&id, &candidates, &self.pinned, false)
                    {
                        Ok(Some(data)) => return Some(Ok((id, data))),
                        // A candidate set with no hash match is a tag
                        // collision: the key's real locator was not in this
                        // snapshot, so skip it rather than fail the scan.
                        Ok(None) => {}
                        Err(error) => return Some(Err(error)),
                    }
                }
            }
        }
    }
}

/// Index tables that must advance together when a reader reloads a checkpoint.
/// Individual table guards are mapped from this shared lock, so a reload's
/// replacement is indivisible with respect to every table lookup.
struct IndexTables {
    collections: HashMap<[u8; 16], ArcSwap<RoomGeneration>>,
    collection_order: Vec<[u8; 16]>,
    shard_collections: HashMap<PackId, HashMap<[u8; 16], u64>>,
    collection_shards: HashMap<[u8; 16], HashSet<PackId>>,
    collection_disk_bytes: HashMap<PackId, HashMap<[u8; 16], u64>>,
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
    /// The `pack_id`s of the packs the checkpoint of this epoch recorded in its
    /// pack table. Every slot a delta operation of this epoch names resolves
    /// through that table, so only while every live pack is in it can a reader
    /// apply the log; a pack created since forces a full checkpoint instead of a
    /// delta coverage batch.
    base_pack_ids: HashSet<PackId>,
    /// Bytes of the on-disk log an fsync has covered. Ordinary delta batches are
    /// not fsynced (the log is rebuildable acceleration), so without this a
    /// coverage batch, which must be durable, would pay for every batch written
    /// since the epoch began in one fsync.
    log_synced_bytes: u64,
    /// The highest redo `delta_seq` assigned so far in this pool. It resumes above
    /// the checkpoint's base sequence at open and is written into the next
    /// checkpoint as its base, so the ordering key survives reopen and rotation.
    delta_seq_high: u64,
}

#[derive(Clone, PartialEq, Eq)]
enum PendingDelta {
    /// Logical redo records for the collection, in the order they were recorded
    /// (each carries its `delta_seq`).
    Redo(Vec<RedoRecord>),
    Snapshot,
    Delete {
        generation: u64,
    },
}

struct V3PendingBatch {
    operations: Vec<DeltaOperation>,
    generation_updates: Vec<([u8; 16], u64)>,
    order_updates: Vec<([u8; 16], u64)>,
    next_order_key: u64,
}

/// Written-but-unsynced pack bytes at which a sync with a journal enabled also
/// fsyncs the dirty packs. With a journal, a sync only has to fsync the WAL, so
/// pack data would otherwise pile up unsynced until the next checkpoint and be
/// flushed all at once there, under the collection locks. Fsyncing it in slices
/// as it accumulates keeps that burst small. Zero disables it.
pub const DEFAULT_PACK_FSYNC_BUDGET_BYTES: u64 = 8 << 20;

/// Unsynced delta-log bytes at which an ordinary delta batch also fsyncs the
/// log. A coverage batch must be durable, and its fsync covers everything
/// written to the log before it, so leaving the backlog to it would make the
/// batch that lets the journal be reclaimed pay for the whole epoch at once.
/// Zero disables it.
pub const DEFAULT_DELTA_FSYNC_BUDGET_BYTES: u64 = 1 << 20;

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
    /// Merged-read-plan tuning for `get_many` (see [`ReadPlanPolicy`]).
    /// Read once per batch; settable at runtime via
    /// [`Self::set_read_plan_policy`].
    read_plan: parking_lot::RwLock<ReadPlanPolicy>,
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
    /// Test-only: a smaller delta-log cap (0 = the real one), so a batch that
    /// does not fit can be produced without writing a 256 MiB log.
    #[cfg(test)]
    delta_log_cap_override: AtomicU64,
    /// Phase breakdown of the most recent full index checkpoint.
    last_checkpoint_breakdown: parking_lot::Mutex<Option<CheckpointBreakdown>>,
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
    /// Written-but-unsynced pack bytes at which a journal-mode sync also fsyncs
    /// the dirty packs (0 disables). See [`Self::set_pack_fsync_budget`].
    pack_fsync_budget_bytes: AtomicU64,
    /// Unsynced delta-log bytes at which a delta batch also fsyncs the log
    /// (0 disables). See [`DEFAULT_DELTA_FSYNC_BUDGET_BYTES`].
    delta_fsync_budget_bytes: AtomicU64,
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
    /// Opt-in wall-clock latency accounting for the public storage operations.
    /// The timer is created only while `stats_enabled` is true so normal
    /// deployments pay no `Instant` cost for this diagnostic.
    operation_timings: OperationTimings,
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
    /// Merged read extents prefetched with `madvise(MADV_WILLNEED)` across
    /// `get_many` batches. This is the physical plan the batch executed, as
    /// opposed to the [`READ_RUN_GAP_BYTES`] logical shape in
    /// [`Self::read_many_runs`].
    read_plan_extents: AtomicU64,
    /// Bytes covered by prefetched read extents (sum of extent lengths, not
    /// physical disk bytes — `madvise` is a hint the kernel may partially or
    /// fully ignore).
    read_plan_prefetch_bytes: AtomicU64,
    /// Planned extents that were not prefetched — offset beyond the current
    /// mapping (typically a buffered-but-unflushed frame), missing mapping, or
    /// a failed `madvise`. A persistently nonzero value means the plan is
    /// silently degrading to no prefetch for those extents.
    read_plan_skipped_extents: AtomicU64,

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
    /// Syncs and forced checkpoints that rewrote the index checkpoint in full.
    checkpoint_writes: AtomicU64,
    /// Background checkpoint tails started (one per detached checkpoint).
    checkpoint_tails_started: AtomicU64,
    /// Syncs that ran with a tail already in flight and appended to it rather
    /// than starting another checkpoint.
    syncs_with_tail_in_flight: AtomicU64,
    /// Syncs that waited for an in-flight tail (emergency zone or rotation).
    syncs_waited_for_tail: AtomicU64,
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
    /// The shard→collection sidecar is known missing or invalid and must be
    /// regenerated even though no index is dirty (see
    /// `RoomScanOutput::sidecar_missing`). Distinct from
    /// `shard_collections_stale`: that flag only says ordinary writes have
    /// advanced the in-memory bookkeeping past the on-disk copy, which is
    /// deferred to the next persistence anchor.
    shard_collections_dirty: AtomicBool,
    /// A completed sync barrier advanced the in-memory shard→collection
    /// bookkeeping past the on-disk sidecar by deferring its write. Set only on
    /// the delta-append and budget-deferred-checkpoint paths, never by the raw
    /// mutation helpers: bookkeeping changed by a write that was never synced
    /// already invalidates the checkpoint fast path, so the next open
    /// full-scans and has no use for a sidecar. The sidecar is rebuildable
    /// acceleration metadata, so this only schedules a flush at the next
    /// anchor: a checkpoint rewrite, a repack, or shutdown.
    shard_collections_stale: AtomicBool,
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
    /// Test-only pause after replay boundary/lease capture and before the
    /// pack walk, used to prove transaction activation is no longer serialized
    /// with the long scan phase.
    #[cfg(test)]
    replay_snapshot_hook: parking_lot::Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
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
    /// Pool tag this store publishes under on a shared journal, as a
    /// [`crate::journal::pool_tag`] code (`0` = untagged/per-pool). Set by
    /// [`Self::enable_shared_journal`]; read on the publish and replay hot
    /// paths, so it is a plain atomic rather than behind the journal mutex.
    journal_pool: std::sync::atomic::AtomicU8,
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
    read_journal: parking_lot::Mutex<Option<ReadJournal>>,
    /// Number of in-process transactions whose published groups are still
    /// being materialized. While non-zero, ordinary reads consult the journal
    /// overlay before the live index.
    transaction_overlay_users: AtomicU64,
    /// The transaction overlay while no transaction uses it. It is kept out of
    /// `read_journal` so read-committed reads on the writer skip the journal
    /// refresh, and kept at all so the next activation resumes its scan
    /// position instead of rescanning the segment.
    #[cfg(feature = "multi-reader")]
    parked_transaction_overlay: parking_lot::Mutex<Option<ReadJournal>>,
    /// Serializes overlay activation and deactivation, so the overlay moves
    /// between `read_journal` and the parking slot exactly once per 0 <-> 1
    /// transition of `transaction_overlay_users`.
    #[cfg(feature = "multi-reader")]
    transaction_overlay_lifecycle: parking_lot::Mutex<()>,
    /// Test-only total of segment bytes the transaction overlay has scanned,
    /// across activations, whether the overlay was retained or rebuilt.
    #[cfg(all(test, feature = "multi-reader"))]
    transaction_overlay_scanned: AtomicU64,
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
    /// The journal LSN this pool's packs are known to durably cover, so a reopen
    /// need not replay at or below it. Read from disk once at open
    /// ([`Self::read_journal_lsn`]) and advanced by whatever makes more of the
    /// journal safe to drop: a checkpoint's `journal.lsn` write, or a coverage
    /// batch appended to the delta log. It only ever moves forward.
    ///
    /// Shared (`Arc`) because a checkpoint tail on its own thread advances it
    /// after the image is durable and cannot borrow the store; the cost is one
    /// small allocation per pool.
    durable_coverage: Arc<AtomicU64>,
    /// The checkpoint tail running on its own thread, if one is (see
    /// [`CheckpointTail`]). At most one at a time; joined before a synchronous
    /// checkpoint, at close, and when the WAL nears its cap.
    checkpoint_worker: parking_lot::Mutex<Option<CheckpointWorker>>,
    /// Whether a sync that needs a full checkpoint hands its tail to a worker
    /// thread instead of running it inline.
    background_checkpoint: AtomicBool,
    /// Test-only: a hold and a failure the next checkpoint tail takes with it.
    #[cfg(test)]
    checkpoint_tail_hook: parking_lot::Mutex<Option<TailHook>>,
    /// Successful checkpoint-bound index reloads triggered by the
    /// read-committed overlay, because a writer's reclaim outran the coverage
    /// this handle's index incorporated. A WAL-cell failure on the reload path
    /// is visible here instead of looking like an ordinary miss. See
    /// [`Self::refresh_read_journal`].
    read_reloads: AtomicU64,
    /// Reload attempts that could not load a checkpoint matching the current
    /// packs, so the read-committed overlay failed closed.
    read_reload_failures: AtomicU64,
    /// Number of read-journal refresh attempts, including no-op refreshes.
    read_refreshes: AtomicU64,
    /// Journal bytes scanned by read-only worker refreshes.
    read_refresh_bytes: AtomicU64,
}

/// Per-collection state for incremental repack.
#[derive(Clone)]
struct RepackIncrementalState {
    /// Per-shard byte offset: next scan starts here (file length at end
    /// of last scan). Shards not in this map haven't been scanned yet.
    // A slot can be retired and reused for a different pack.  Keep the
    // pack_id beside the cursor so an offset from the old incarnation is
    // never applied to the replacement file.
    scan_offsets: HashMap<u16, (PackId, u64)>,
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

    /// Open a read-only worker on one pool of a **shared** WAL.
    ///
    /// Identical to [`Self::open_read_committed`] except the overlay applies
    /// only frames tagged with `pool`, so a worker reading the state pool does
    /// not observe event-DAG or edges mutations that share the same
    /// segment. `wal_path` is the database root's shared `wal.bin`, not a
    /// per-pool segment.
    ///
    /// # Errors
    /// Same as [`Self::open_read_committed`].
    #[cfg(feature = "multi-reader")]
    pub fn open_read_committed_shared(
        base_dir: PathBuf,
        wal_path: impl AsRef<Path>,
        pool: crate::layout::ShardType,
    ) -> Result<Self, StorageError> {
        let store = Self::open_read_only(base_dir)?;
        store.enable_read_journal_shared(wal_path, pool)?;
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
        let open_shards: Vec<(u16, PackId, PathBuf, u64)> = shards
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
        let durable_state = Self::durable_index_state(&base_dir);
        let read_covered = Self::journal_lsn_with(&base_dir, durable_state.covered_lsn);
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
            let assemble_started = std::time::Instant::now();
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
                read_covered,
                durable_state.fingerprint,
            );
            timings.assemble = assemble_started.elapsed();
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
            read_covered,
            durable_state.fingerprint,
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
        let slot_to_pack_id: HashMap<u16, PackId> = records
            .iter()
            .map(|(slot, _, _, pack_id)| (*slot, *pack_id))
            .collect();

        for (&slot, &count) in &counts {
            let Some(&pack_id) = slot_to_pack_id.get(&slot) else {
                continue;
            };
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
        durable_coverage: u64,
        initial_durable_fp: u64,
    ) -> Self {
        let index_config = crate::index::IndexConfig {
            seed: shards.bucket_seed(),
            ..Default::default()
        };
        // Seed the per-collection refresh fingerprint from the persisted
        // checkpoint/delta state so the first miss for each collection can
        // skip refresh when nothing durable changed.
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
            read_plan: RwLock::new(ReadPlanPolicy::default()),
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_collection: RwLock::new(HashMap::new()),
            repack_incremental: RwLock::new(HashMap::new()),
            last_shard_collections_flush: RwLock::new(None),
            index_checkpoint_dirty: AtomicBool::new(false),
            shard_collections_dirty: AtomicBool::new(scan_out.sidecar_missing),
            shard_collections_stale: AtomicBool::new(false),
            shard_collections_recovery_failed: AtomicBool::new(scan_out.sidecar_recovery_failed),
            shard_collections_failure_logged: AtomicBool::new(false),
            #[cfg(test)]
            recovery_pause_hook: parking_lot::Mutex::new(None),
            #[cfg(test)]
            replay_snapshot_hook: parking_lot::Mutex::new(None),
            journal: parking_lot::Mutex::new(None),
            journal_pool: std::sync::atomic::AtomicU8::new(0),
            journal_recovery: parking_lot::Mutex::new(Vec::new()),
            replaying: AtomicBool::new(false),
            read_journal: parking_lot::Mutex::new(None),
            #[cfg(feature = "multi-reader")]
            parked_transaction_overlay: parking_lot::Mutex::new(None),
            #[cfg(feature = "multi-reader")]
            transaction_overlay_lifecycle: parking_lot::Mutex::new(()),
            transaction_overlay_users: AtomicU64::new(0),
            #[cfg(all(test, feature = "multi-reader"))]
            transaction_overlay_scanned: AtomicU64::new(0),
            read_covered_lsn,
            durable_coverage: Arc::new(AtomicU64::new(durable_coverage)),
            checkpoint_worker: parking_lot::Mutex::new(None),
            background_checkpoint: AtomicBool::new(false),
            #[cfg(test)]
            checkpoint_tail_hook: parking_lot::Mutex::new(None),
            read_reloads: AtomicU64::new(0),
            read_reload_failures: AtomicU64::new(0),
            read_refreshes: AtomicU64::new(0),
            read_refresh_bytes: AtomicU64::new(0),
            last_open_timings: parking_lot::Mutex::new(None),
            last_sync_timings: parking_lot::Mutex::new(None),
            #[cfg(test)]
            delta_log_cap_override: AtomicU64::new(0),
            last_checkpoint_breakdown: parking_lot::Mutex::new(None),
            sync_totals: SyncTotals::default(),
            sync_diagnostics: parking_lot::Mutex::new(SyncDiagnostics::default()),
            publish_calls: AtomicU64::new(0),
            publish_time_ns: AtomicU64::new(0),
            pending_publish_since: parking_lot::Mutex::new(None),
            publish_generation: AtomicU64::new(0),
            checkpoint_rewrite_min_interval_ns: AtomicU64::new(0),
            pack_fsync_budget_bytes: AtomicU64::new(DEFAULT_PACK_FSYNC_BUDGET_BYTES),
            delta_fsync_budget_bytes: AtomicU64::new(DEFAULT_DELTA_FSYNC_BUDGET_BYTES),
            checkpoint_rewrite_max_bytes: AtomicU64::new(0),
            last_checkpoint_rewrite_at: parking_lot::Mutex::new(None),
            checkpoint_bytes_at_last_rewrite: AtomicU64::new(0),
            checkpoint_skips: AtomicU64::new(0),
            stats_enabled: AtomicBool::new(false),
            operation_timings: OperationTimings::default(),
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
            read_plan_extents: AtomicU64::new(0),
            read_plan_prefetch_bytes: AtomicU64::new(0),
            read_plan_skipped_extents: AtomicU64::new(0),
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
            checkpoint_tails_started: AtomicU64::new(0),
            syncs_with_tail_in_flight: AtomicU64::new(0),
            syncs_waited_for_tail: AtomicU64::new(0),
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

    /// The journal LSN this pool's durable state covers (0 when none is
    /// recorded): the newest of the `journal.lsn` a checkpoint wrote, the
    /// coverage the checkpoint's own header records, and what a committed batch
    /// of the delta log that continues the current checkpoint claims.
    ///
    /// A missing, torn or corrupt record of either kind claims nothing, so the
    /// result can only be lower than the truth, which makes recovery replay
    /// more, never less. It reads the disk (and scans the delta log), so a
    /// writer uses [`Self::durable_coverage`] instead.
    #[must_use]
    pub fn read_journal_lsn(base_dir: &std::path::Path) -> u64 {
        Self::journal_lsn_with(base_dir, Self::durable_index_state(base_dir).covered_lsn)
    }

    /// `read_journal_lsn` given the coverage a checkpoint and its delta log
    /// already yielded, so an open that also needs the log's fingerprint scans
    /// the log once.
    fn journal_lsn_with(base_dir: &std::path::Path, checkpoint_and_delta: u64) -> u64 {
        let recorded = fs::read(Self::journal_lsn_path(base_dir))
            .ok()
            .and_then(|bytes| bytes.get(..8).map(<[u8; 8]>::try_from))
            .and_then(Result::ok)
            .map_or(0, u64::from_le_bytes);
        recorded.max(checkpoint_and_delta)
    }

    /// What the disk records of the durable index state, in one pass over the
    /// checkpoint and its delta log: the coverage the checkpoint records or a
    /// committed batch of the log continuing it claims (whichever is newer; 0
    /// when there is neither), and the fingerprint of the durable pack state
    /// (0 when unknown). The log only counts if it names that checkpoint's
    /// fingerprint; a stale epoch left by a crash is inert.
    ///
    /// Reading the checkpoint's own record matters: a new checkpoint retires
    /// the delta epoch that held the newest claim, and `journal.lsn` is written
    /// only after the checkpoint is durable, so for a moment the checkpoint's
    /// header is the only place that coverage is written down.
    fn durable_index_state(base_dir: &std::path::Path) -> DurableIndexState {
        let Ok(Some(summary)) = crate::index::checkpoint::read_checkpoint_summary(
            &Self::index_checkpoint_path(base_dir),
        ) else {
            return DurableIndexState::default();
        };
        // One pass over the delta log yields both what it claims and where it
        // ends; the log is only trusted if it names this checkpoint.
        let tail =
            delta::read_delta_tail_fingerprint(&Self::delta_path(base_dir, summary.fingerprint));
        let (log_coverage, fingerprint) = match tail {
            Ok(Some(tail)) if tail.base_fingerprint == summary.fingerprint => {
                (tail.coverage.unwrap_or(0), tail.tail_fingerprint)
            }
            Ok(Some(_) | None) => (0, summary.fingerprint),
            Err(_) => (0, 0),
        };
        DurableIndexState {
            covered_lsn: summary.covered_lsn.max(log_coverage),
            fingerprint,
        }
    }

    /// The journal LSN this pool's packs are known to durably cover, held in
    /// memory. What [`Self::read_journal_lsn`] said at open, moved forward by
    /// every checkpoint and coverage batch this handle wrote since.
    #[must_use]
    pub fn durable_coverage(&self) -> u64 {
        self.durable_coverage.load(Ordering::Acquire)
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
        open_shards: &[(u16, PackId, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
    ) -> (BookkeepingSource, HashMap<[u8; 16], HashMap<u16, u64>>) {
        let mut counts: HashMap<[u8; 16], HashMap<u16, u64>> = HashMap::new();
        let Some(directory) = read_persisted_shard_collections(base_dir) else {
            return (BookkeepingSource::SlotScan, counts);
        };
        if directory.fingerprint != local_fingerprint {
            return (BookkeepingSource::SlotScan, counts);
        }
        let pack_to_slot: HashMap<PackId, u16> = open_shards
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

    /// Whether every pending redo set names a record that is really at its
    /// locator: the frame there belongs to the same collection and carries the
    /// same full hash. Any mismatch, or a frame that cannot be read, fails the
    /// whole log closed to the caller's rescan.
    fn redo_locators_verified(
        shards: &ShardPool,
        sets: &HashMap<[u8; 16], Vec<RedoRecord>>,
        pack_lookup: &HashMap<PackId, (u16, u64)>,
    ) -> bool {
        sets.values().flatten().all(|record| {
            let RedoOp::Set {
                full_hash,
                pack_id,
                offset,
                ..
            } = record.op
            else {
                return false;
            };
            let Some(shard) = pack_lookup
                .get(&pack_id)
                .and_then(|(slot, _)| shards.get_shard(*slot))
            else {
                return false;
            };
            matches!(
                shards.record_identity_at(&shard, offset),
                Ok((collection, hash)) if collection == record.collection_id && hash == full_hash
            )
        })
    }

    /// Apply logical redo sets, in `delta_seq` order, to a loaded index whose
    /// slots are already this process's. The index is pre-sized once for the whole
    /// batch and grown from its persisted identity tables when a growth boundary
    /// is reached; if it cannot grow (incomplete tables), the whole replay fails
    /// closed to the caller's rescan. Locators are checked for extent by the
    /// caller; identity is confirmed lazily by reads, which compare the record's
    /// hash.
    fn apply_redo_sets(
        index: LossyIndex,
        records: &[RedoRecord],
        pack_lookup: &HashMap<PackId, (u16, u64)>,
    ) -> Option<LossyIndex> {
        let mut index = if index.is_mmap_backed() {
            index.clone()
        } else {
            index
        };
        let target =
            LossyIndex::capacity_for_entries(index.len().saturating_add(records.len()), 0)?;
        while usize::try_from(index.capacity()).ok()? < target {
            index = index.grow()?;
        }
        for record in records {
            let RedoOp::Set {
                full_hash,
                pack_id,
                offset,
                ..
            } = record.op
            else {
                return None;
            };
            let (slot, _) = *pack_lookup.get(&pack_id)?;
            if index.insert(&full_hash, slot, offset).is_err() {
                index = index.grow()?;
                index.insert(&full_hash, slot, offset).ok()?;
            }
        }
        Some(index)
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
        open_shards: &[(u16, PackId, PathBuf, u64)],
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
        let packs: Vec<(PackId, u64)> = open_shards
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
        // The journal coverage the applied part of the delta log claims. A
        // coverage batch lets the journal be reclaimed past the checkpoint's own
        // coverage, so the index built here is only complete up to the claim if
        // the operations the claim describes were applied.
        let mut log_coverage: Option<u64> = None;
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
                log_coverage = log.coverage;
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
        } else if reload_mode == ReloadMode::JournalBound {
            // A read-journal reload skips the exact-pack gate (the writer keeps
            // appending), but it must still see the index changes a coverage
            // batch describes, or a reclaim past the checkpoint's coverage would
            // leave records that are in neither the index nor the journal. Take
            // the delta operations up to the last coverage claim. Every one of
            // them was written after the packs they name were durable, and the
            // writer only appends coverage while every live pack is in this
            // checkpoint's pack table, so their slots translate below; if one
            // does not, the load fails closed like any other bad slot.
            let delta_started = std::time::Instant::now();
            if let Some(mut log) = delta::read_delta_log_v3(&delta_path) {
                if log.base_fingerprint == checkpoint.fingerprint && log.coverage_prefix_ops > 0 {
                    log.operations.truncate(log.coverage_prefix_ops);
                    log_coverage = log.coverage;
                    replay_operations = log.operations;
                    timings.delta_replay_operations =
                        u64::try_from(replay_operations.len()).unwrap_or(u64::MAX);
                }
            }
            timings.delta_replay = delta_started.elapsed();
        }

        let prepare_started = std::time::Instant::now();
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
        // Logical redo: every pack this log may name, by its stable id, with its
        // current length. The log's tail fingerprint pins those lengths, so a
        // record whose extent lies outside its pack is corruption.
        let pack_lookup: HashMap<PackId, (u16, u64)> = open_shards
            .iter()
            .map(|(slot, pack_id, _, file_len)| (*pack_id, (*slot, *file_len)))
            .collect();
        let mut redo_by_collection: HashMap<[u8; 16], Vec<RedoRecord>> = HashMap::new();
        let mut last_delta_seq = checkpoint.base_delta_seq;
        let reject_log = || {
            if writable {
                let _ = std::fs::remove_file(&delta_path);
            }
        };
        for operation in replay_operations {
            match operation {
                DeltaOperation::Redo(record) => {
                    // Strictly increasing above the checkpoint's base sequence.
                    if record.delta_seq <= last_delta_seq {
                        reject_log();
                        return None;
                    }
                    last_delta_seq = record.delta_seq;
                    if deleted_collections.contains(&record.collection_id) {
                        continue;
                    }
                    let RedoOp::Set {
                        pack_id,
                        offset,
                        record_len,
                        ..
                    } = record.op
                    else {
                        reject_log();
                        return None;
                    };
                    let extent_ok = pack_lookup.get(&pack_id).is_some_and(|(_, pack_len)| {
                        offset
                            .checked_add(u64::from(record_len))
                            .is_some_and(|end| end <= *pack_len)
                    });
                    if live_generations.get(&record.collection_id) != Some(&record.base_generation)
                        || !extent_ok
                    {
                        reject_log();
                        return None;
                    }
                    redo_by_collection
                        .entry(record.collection_id)
                        .or_default()
                        .push(record);
                }
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
                    redo_by_collection.remove(&collection_id);
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
                    redo_by_collection.remove(&collection_id);
                    snapshot_indexes.remove(&collection_id);
                    v3_tombstones.insert(collection_id);
                }
                // A coverage claim changes no index state; it is read
                // separately (see `durable_coverage_from_disk`).
                DeltaOperation::Coverage { .. } => {}
            }
        }

        // Every redo record's locator must lead to the record it names. A locator
        // that is in bounds but wrong would otherwise be installed, and the key
        // would read as missing (the read compares the frame's hash and skips a
        // mismatch). Offsets ascend within a pack in log order, since each set
        // names a freshly appended record, so this is a sequential pass over
        // pages the pack recovery has just read.
        if !Self::redo_locators_verified(shards, &redo_by_collection, &pack_lookup) {
            reject_log();
            return None;
        }

        let slot_to_pack_id: HashMap<u16, PackId> = open_shards
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
        let pack_id_to_local_slot: HashMap<PackId, u16> = open_shards
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
        timings.replay_prepare = prepare_started.elapsed();
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
            let index = match redo_by_collection.remove(&loaded.collection_id) {
                Some(records) => Self::apply_redo_sets(index, &records, &pack_lookup)?,
                None => index,
            };
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
            let index = match redo_by_collection.remove(&collection_id) {
                Some(records) => Self::apply_redo_sets(index, &records, &pack_lookup)?,
                None => index,
            };
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
        let bookkeeping_started = std::time::Instant::now();

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
                let Some(&pack_id) = slot_to_pack_id.get(&slot) else {
                    continue;
                };
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

        timings.bookkeeping = bookkeeping_started.elapsed();

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
            // What was on disk at open is treated as durable; only what this
            // session appends is owed an fsync.
            log_synced_bytes: log_bytes_on_disk,
            delta_seq_high: last_delta_seq,
            log_version: replay_log_version,
            base_pack_ids: checkpoint
                .pack_table
                .iter()
                .map(|&(_, pack_id)| pack_id)
                .collect(),
            ..DeltaLogState::default()
        };

        // The index holds the checkpoint plus the operations of the log that were
        // applied, so it covers whatever those describe as well.
        let covered_lsn = covered_lsn.max(log_coverage.unwrap_or(0));
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
    ) -> parking_lot::MappedRwLockReadGuard<'_, HashMap<PackId, HashMap<[u8; 16], u64>>> {
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

    /// Set the merged-read-plan policy for `get_many` (see
    /// [`ReadPlanPolicy`]). Takes effect on the next batch.
    pub fn set_read_plan_policy(&self, policy: ReadPlanPolicy) {
        *self.read_plan.write() = policy;
    }

    /// The current merged-read-plan policy.
    #[must_use]
    pub fn read_plan_policy(&self) -> ReadPlanPolicy {
        *self.read_plan.read()
    }

    /// Whether this storage attempts zstd compression on written records.
    #[must_use]
    pub fn is_compression_enabled(&self) -> bool {
        self.shards.is_compression_enabled()
    }

    /// The checksum policy configured for this storage.
    #[must_use]
    pub fn checksum_policy(&self) -> packfile::ChecksumPolicy {
        self.shards.checksum_policy()
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

    /// Stream every live record of one collection from a single consistent
    /// snapshot.
    ///
    /// The scan captures a generation and pack-file lengths, resolving keys
    /// found within those lengths through the generation's index. Candidate
    /// locators and open shard handles are retained for lazy payload reads, so
    /// a concurrent repack may retire those shards while they remain readable.
    ///
    /// When a read-committed journal overlay is installed, its committed puts
    /// are merged in and win over the durable index (matching
    /// `get_read_committed`), and a collection delete the durable index does not
    /// yet reflect suppresses durable records entirely, so a live multi-reader
    /// store is scanned completely.
    ///
    /// The pack scan covers the captured file lengths without flushing or
    /// syncing them. Writes still buffered in this process are excluded unless
    /// supplied by the journal overlay; file-visible bytes need not be fsynced.
    /// Records are yielded as `(NodeId, NodeData)` with no guaranteed ordering.
    ///
    /// # Errors
    /// Propagates pack metadata-scan and journal-refresh errors, including
    /// [`StorageError::Io`] with `WouldBlock` when a read-journal reload must
    /// be retried. Payload-read errors are yielded by the returned iterator.
    pub fn scan_collection(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<CollectionScan<'_>, StorageError> {
        // Puts update a non-mmap index in place without replacing its
        // generation. Serialize snapshot construction with those updates so
        // every locator we resolve belongs to the captured pack-length
        // boundary. Shard handles are opened before the guard is released;
        // the full metadata scan and lazy payload reads do not block later puts.
        let put_lock = self.put_mutex(collection_id);
        let mut put_guard = Some(put_lock.lock());
        self.scan_collection_locked(collection_id, &mut put_guard)
    }

    /// Take a collection scan paired with a durable journal cursor.
    ///
    /// The collection lock spans the pack flush, WAL durability barrier, and
    /// snapshot construction. Thus every mutation for this collection through
    /// the returned cursor is represented by the scan; later mutations receive
    /// larger LSNs and are available from [`JournalCoordinator::changes_since`].
    /// The returned [`crate::journal::JournalReplayLease`] pins that history
    /// until dropped, preventing either per-pool or shared-WAL reclaim from
    /// expiring the cursor while the caller catches up.
    ///
    /// In multi-reader mode, a transaction publishes its journal group before
    /// materializing its pack records. The lifecycle lock covers the active
    /// transaction check and WAL-boundary capture, then is released before the
    /// pack walk. A transaction that starts later cannot materialize this
    /// collection while its put mutex is held, and this method scans packs only
    /// (never the read-journal overlay), so its mutations remain strictly in
    /// replay. The lock can span the WAL sync, but not the collection scan or
    /// the preceding pack flush.
    ///
    /// # Errors
    /// Returns `Unsupported` when no journal is enabled, `WouldBlock` while a
    /// transaction is already being materialized, or propagates pack/WAL
    /// errors. It also returns an expired-cursor error if reclaim wins the race
    /// before the replay lease is installed; in that case retry the snapshot.
    pub fn scan_collection_at_snapshot(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<
        (
            CollectionScan<'_>,
            crate::journal::JournalCursor,
            JournalReplayLease,
        ),
        StorageError,
    > {
        let put_lock = self.put_mutex(collection_id);
        let mut put_guard = Some(put_lock.lock());

        let journal = self.journal().ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "replayable collection scans require an enabled journal",
            ))
        })?;

        // Flush before taking the transaction lifecycle lock. A target-
        // collection put is excluded by `put_mutex`; unrelated writes may
        // proceed, but their records are irrelevant to this collection.
        self.shards.flush_all()?;

        let capture_boundary =
            || -> Result<(crate::journal::JournalCursor, JournalReplayLease), StorageError> {
                journal.sync().map_err(StorageError::Io)?;

                // This is the pool-specific materialized watermark (floored below
                // any published-but-not-yet-materialized transaction), also
                // floored by durable checkpoint coverage when the retained WAL
                // prefix is empty.
                let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
                let cursor = journal
                    .replay_cursor(covered_lsn)
                    .map_err(StorageError::Io)?;
                let lease = journal
                    .pin_replay_cursor(&cursor)
                    .map_err(StorageError::Io)?;
                Ok((cursor, lease))
            };

        #[cfg(feature = "multi-reader")]
        let (cursor, lease) = {
            let _transaction_lifecycle = self.transaction_overlay_lifecycle.lock();
            if self.transaction_overlay_users.load(Ordering::Acquire) != 0 {
                return Err(StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "cannot capture a replayable scan while a transaction is materializing",
                )));
            }
            capture_boundary()?
        };
        #[cfg(not(feature = "multi-reader"))]
        let (cursor, lease) = capture_boundary()?;

        #[cfg(test)]
        if let Some(hook) = self.replay_snapshot_hook.lock().take() {
            hook();
        }

        let scan = self.scan_collection_packs_locked(collection_id, &mut put_guard)?;
        Ok((scan, cursor, lease))
    }

    /// Build the pinned scan while the caller's collection put mutex remains
    /// held. The shared scan helper releases it after opening the snapshot's
    /// shard handles.
    fn scan_collection_locked(
        &self,
        collection_id: &[u8; 16],
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        // Snapshot the read-committed overlay once: committed-but-unflushed
        // puts, plus the per-collection delete boundary a reader's stale index
        // may not yet cover.
        let (overlay_puts, overlay_deletes) = {
            let guard = self.refresh_read_journal()?;
            match guard.as_ref() {
                Some(journal) => {
                    let deleted = journal.delete_lsn.contains_key(collection_id);
                    let puts: HashMap<NodeId, NodeData> = journal
                        .puts
                        .get(collection_id)
                        .map(|committed| {
                            committed
                                .iter()
                                .map(|(id, (payload, _lsn))| (*id, NodeData::new(payload.clone())))
                                .collect()
                        })
                        .unwrap_or_default();
                    (puts, deleted)
                }
                None => (HashMap::new(), false),
            }
        };

        self.scan_collection_with_overlay(collection_id, &overlay_puts, overlay_deletes, put_guard)
    }

    /// Snapshot only materialized pack records. The replayable scan uses this
    /// form so an overlay mutation beyond its cursor cannot be included both in
    /// the scan and in subsequent WAL replay.
    fn scan_collection_packs_locked(
        &self,
        collection_id: &[u8; 16],
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        let overlay_puts = HashMap::new();
        self.scan_collection_with_overlay(collection_id, &overlay_puts, false, put_guard)
    }

    /// Build lazy scan work, letting `overlay_puts` replace pack values.
    ///
    /// `overlay_deletes` suppresses all pack records for this collection. The
    /// caller holds its put mutex in `put_guard`; this releases the guard after
    /// capturing the generation, file lengths, and open handles. Pack-open and
    /// metadata-scan errors propagate.
    fn scan_collection_with_overlay(
        &self,
        collection_id: &[u8; 16],
        overlay_puts: &HashMap<NodeId, NodeData>,
        overlay_deletes: bool,
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        let generation = self.generation(collection_id);
        let shards = self.shards.all_shards();

        // A delete visible only in the overlay means the reader's durable
        // index may still hold pre-delete records, so it cannot be trusted
        // for this collection; the overlay puts are the whole answer.
        let scanners = if !overlay_deletes && generation.is_some() {
            Self::open_collection_scan_shards(&shards)?
        } else {
            Vec::new()
        };

        // The generation, file lengths, and open file handles now pin the
        // snapshot boundary. Releasing the collection lock here lets writers
        // proceed while the metadata scan walks every shard; retained handles
        // remain readable if repack unlinks a shard.
        drop(put_guard.take());

        let keys = Self::scan_collection_keys(scanners, collection_id)?;
        let mut pinned = HashMap::new();
        let mut work = Vec::new();
        if let Some(generation) = generation.as_deref() {
            Self::append_collection_locator_work(
                keys,
                &generation.index,
                overlay_puts,
                &shards,
                &mut pinned,
                &mut work,
            );
        }

        // Durable keys shadowed by the overlay were skipped above, so each
        // overlay put is appended once.
        work.extend(
            overlay_puts
                .iter()
                .map(|(id, data)| ScanWork::Data(*id, data.clone())),
        );

        Ok(CollectionScan {
            store: self,
            pinned,
            pending: work.into_iter(),
        })
    }

    /// Open metadata scanners paired with each shard's captured byte length.
    /// Open, metadata, and header-validation errors propagate.
    fn open_collection_scan_shards(
        shards: &[(u16, Arc<Shard>)],
    ) -> Result<Vec<(u64, packfile::PackfileScanner)>, StorageError> {
        shards
            .iter()
            .map(|(_, shard)| {
                let file = std::fs::File::open(&shard.path)?;
                let end = shard.file_len();
                let scanner = packfile::scan_packfile_iter_from_file(file, false)?;
                Ok((end, scanner))
            })
            .collect()
    }

    /// Collect distinct hashes for `collection_id` before each captured byte
    /// length, in first-seen order. Scanner errors propagate.
    fn scan_collection_keys(
        scanners: Vec<(u64, packfile::PackfileScanner)>,
        collection_id: &[u8; 16],
    ) -> Result<Vec<NodeId>, StorageError> {
        let mut keys = Vec::new();
        let mut seen = HashSet::new();
        for (end, scanner) in scanners {
            for entry in scanner {
                let (record_collection, hash, offset) = entry?;
                // The scanner streams in file order, so crossing the captured
                // length means every later entry is post-boundary.
                if offset >= end {
                    break;
                }
                if record_collection != *collection_id {
                    continue;
                }
                if seen.insert(hash) {
                    keys.push(hash);
                }
            }
        }
        Ok(keys)
    }

    /// Append indexed candidates for keys not shadowed by `overlay_puts`,
    /// pinning available shard handles. Keys without candidates are skipped.
    fn append_collection_locator_work(
        keys: Vec<NodeId>,
        index: &LossyIndex,
        overlay_puts: &HashMap<NodeId, NodeData>,
        shards: &[(u16, Arc<Shard>)],
        pinned: &mut HashMap<u16, Arc<Shard>>,
        work: &mut Vec<ScanWork>,
    ) {
        for id in keys {
            if overlay_puts.contains_key(&id) {
                // The overlay value is newer than the durable one.
                continue;
            }
            let candidates: Vec<(u16, u64)> = index.lookup_all(&id).collect();
            if candidates.is_empty() {
                continue;
            }
            for &(slot, _) in &candidates {
                if let Some((_, shard)) = shards.iter().find(|(s, _)| *s == slot) {
                    pinned.entry(slot).or_insert_with(|| Arc::clone(shard));
                }
            }
            work.push(ScanWork::Locators(id, candidates));
        }
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
    /// `HashMap<PackId, u64>` for use with `replace_collection_shard_counts`.
    fn slot_counts_to_pack_id_counts(&self, counts: &HashMap<u16, u64>) -> HashMap<PackId, u64> {
        let shards = self.shards.all_shards();
        let slot_to_pack_id: HashMap<u16, PackId> = shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        counts
            .iter()
            .filter_map(|(&slot, &count)| {
                slot_to_pack_id.get(&slot).map(|&pack_id| (pack_id, count))
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
    fn record_new_shard_collection(
        &self,
        pack_id: PackId,
        collection_id: &[u8; 16],
        disk_bytes: u64,
    ) {
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
        count_packs: &[PackId],
        byte_packs: &[(PackId, u64)],
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

    /// Resolve the pack address of an open shard slot. A slot written to just
    /// above always has a live shard; a missing one is corruption, not a
    /// reason to fabricate an identity.
    fn pack_id_for_slot(&self, slot: u16) -> Result<PackId, StorageError> {
        self.shards
            .get_shard(slot)
            .map(|shard| shard.pack_id)
            .ok_or_else(|| StorageError::Corrupt(format!("missing shard slot {slot}")))
    }

    fn record_disk_bytes(&self, pack_id: PackId, collection_id: &[u8; 16], disk_bytes: u64) {
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
        bytes_by_pack: &HashMap<PackId, u64>,
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
        counts: &HashMap<PackId, u64>,
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
        let packs: Vec<(PackId, u64)> = self
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
            buf.extend_from_slice(pack_id.as_bytes());
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
        self.shard_collections_stale.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// True when a completed sync barrier deferred the `shard_collections.bin`
    /// write to the next anchor (checkpoint rewrite, repack, or clean
    /// shutdown). Writes that have not been synced are not reported here: they
    /// invalidate the checkpoint fast path, so the next open full-scans and
    /// never consults the sidecar.
    pub fn is_shard_collections_stale(&self) -> bool {
        self.shard_collections_stale.load(Ordering::Relaxed)
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

    /// Record a successful index insertion as a logical redo record, if the delta
    /// log can legitimately continue through it.
    ///
    /// The record names the change by content identity (`hash`) and its
    /// locator (`pack_id`, `offset`, `record_len`), not by table position, and
    /// takes its `delta_seq` from this pool's counter under the delta-state lock,
    /// so sequence order is append order across collections. Recording stops
    /// once a snapshot or tombstone supersedes the collection's pending updates.
    /// A generation mismatch, or a locator a record cannot represent, promotes
    /// the collection to a snapshot instead.
    fn record_redo(
        &self,
        collection_id: &[u8; 16],
        generation: u64,
        hash: &[u8; 16],
        slot: u16,
        offset: u64,
        record_len: u64,
    ) {
        // A slot with no open shard has no stable pack id to name. Falling back
        // to the slot number would name a different pack that happens to carry
        // that number, so such a record is not representable: the collection is
        // snapshotted instead.
        let pack_id = self.shards.get_shard(slot).map(|shard| shard.pack_id);
        let record_len = u32::try_from(record_len).unwrap_or(0);
        let representable =
            pack_id.is_some() && record_len != 0 && offset <= crate::index::IndexEntry::MAX_OFFSET;
        let pack_id = pack_id.unwrap_or(PackId([0; crate::packfile::PACK_ID_LEN]));
        let mut state = self.delta_state.lock();
        let base_generation_matches =
            state.base_generations.get(collection_id) == Some(&generation);
        if !representable || !base_generation_matches {
            // Replaces pending records too: the snapshot taken at sync time
            // includes them.
            if !matches!(
                state.pending.get(collection_id),
                Some(PendingDelta::Delete { .. })
            ) {
                state.pending.insert(*collection_id, PendingDelta::Snapshot);
            }
            return;
        }
        if matches!(
            state.pending.get(collection_id),
            Some(PendingDelta::Snapshot | PendingDelta::Delete { .. })
        ) {
            return;
        }
        state.delta_seq_high = state.delta_seq_high.saturating_add(1);
        let record = RedoRecord {
            collection_id: *collection_id,
            op: RedoOp::Set {
                full_hash: *hash,
                pack_id,
                offset,
                record_len,
            },
            delta_seq: state.delta_seq_high,
            base_generation: generation,
        };
        match state.pending.entry(*collection_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(PendingDelta::Redo(vec![record]));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if let PendingDelta::Redo(records) = entry.get_mut() {
                    records.push(record);
                }
            }
        }
    }

    /// Snapshot every collection's live generation for a checkpoint, encoded as
    /// `(collection_id, generation, room)`. Callers hold all collection put
    /// mutexes, so the snapshot is a consistent point in the index.
    fn checkpoint_snapshots(&self) -> Vec<([u8; 16], u64, Arc<RoomGeneration>)> {
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
    }

    /// The journal LSN this store's checkpoint covers: its own pool's committed
    /// watermark on a shared segment (one group can interleave other pools'
    /// frames), or the global committed LSN for a per-pool segment. `None` when
    /// no journal is enabled.
    ///
    /// The value is floored by the durable coverage already recorded beside the
    /// checkpoint. A pool whose frames have all been reclaimed has no committed
    /// group left to derive a watermark from, so the fresh coordinator reports
    /// zero for it; writing that zero would erase the durable coverage and make
    /// the next replay start from a stale bound. Coverage only ever advances.
    fn checkpoint_covered_lsn(&self) -> Option<u64> {
        self.journal().map(|journal| {
            #[cfg(feature = "multi-reader")]
            let committed = match pool_from_tag(self.journal_pool.load(Ordering::Acquire)) {
                Some(pool) => journal.committed_lsn_for_pool(pool),
                None => journal.committed_lsn(),
            };
            #[cfg(not(feature = "multi-reader"))]
            let committed = journal.committed_lsn();
            // The shared-WAL coverage batch and the full checkpoint both run
            // `sync_all` after capturing this and before recording it, so a
            // frame at or below `committed` that has been applied is covered.
            // `a_power_cut_image_never_loses_a_record_a_claim_covered` checks
            // it for autocommit writes and shared-WAL transactions, on records
            // made durable by a full sync of both pools (an inflated claim
            // fails it). A pool synced alone while another stays silent, and
            // per-pool coordinators, are not exercised. `max` only keeps the
            // value from moving backwards.
            committed.max(self.durable_coverage())
        })
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
    ///
    /// Returns `true` when a checkpoint was written, `false` when nothing was
    /// dirty to persist — callers that count writes must gate on this, since a
    /// concurrent sync can consume the dirty flag first.
    fn persist_index_checkpoint(&self) -> Result<bool, StorageError> {
        // A synchronous checkpoint never overlaps a worker's.
        self.finish_checkpoint_worker(true);
        let Some(captured) = self.capture_checkpoint()? else {
            return Ok(false);
        };
        let CapturedCheckpoint { tail, breakdown } = captured;
        match tail.run() {
            Ok(done) => {
                self.record_checkpoint(breakdown, &done);
                Ok(true)
            }
            Err(error) => {
                self.abandon_checkpoint();
                Err(error)
            }
        }
    }

    /// The part of a checkpoint that needs the locks: everything up to the
    /// point where the image bytes and the delta-epoch rotation are captured.
    /// After it returns the returned tail owns everything it needs, so it can
    /// run on any thread. `None` when nothing is dirty.
    fn capture_checkpoint(&self) -> Result<Option<CapturedCheckpoint>, StorageError> {
        let started = std::time::Instant::now();
        let mut breakdown = CheckpointBreakdown::default();
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // With a journal, this checkpoint must make the packs durable before it
        // records coverage, and it does so below with every collection locked,
        // which stalls every writer for as long as the fsync takes. Do the bulk
        // of that here first, with no lock held: the fsync under the locks then
        // only covers what was written since.
        if self.journal().is_some() {
            self.shards.sync_dirty()?;
        }
        breakdown.pre_sync = started.elapsed();
        let lock_started = std::time::Instant::now();
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
        breakdown.lock_wait = lock_started.elapsed();
        let locked_started = std::time::Instant::now();

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
        breakdown.locked_sync = locked_started.elapsed();
        let snapshot_started = std::time::Instant::now();
        let live_shards = self.shards.all_shards();
        let packs: Vec<(PackId, u64)> = live_shards
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
        let pack_table: Vec<(u16, PackId)> = live_shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();

        let snapshots = self.checkpoint_snapshots();
        let serialize_started = std::time::Instant::now();

        // Rotate the delta epoch while still locked: `old_base_fingerprint`
        // (D0's name, if any) is retired below only after `fingerprint`'s
        // checkpoint (C1) is durable; every collection's writers already
        // target the new epoch (D1) by the time the locks drop next.
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        // Every put mutex is held, so no redo record is in flight: the sequence
        // reached so far is exactly what this checkpoint incorporates.
        let base_delta_seq = self.delta_state.lock().delta_seq_high;
        self.delta_state.lock().base_pack_ids =
            pack_table.iter().map(|&(_, pack_id)| pack_id).collect();
        // Every collection's put mutex is held here, so no `put` is mid-flight
        // and every mutation published so far has completed its index update.
        // The recorded LSN must be *committed*, not merely published: a
        // concurrent put can publish above the last WAL commit, and LSNs above
        // `committed_lsn` are discarded and reused after a crash. Recording one
        // as covered would make a reopen skip a future mutation that reuses
        // that LSN (the journal scan only knows the committed prefix). The
        // committed LSN is conservative -- the fsynced packs above may cover
        // more -- and replaying that suffix again is idempotent.
        //
        // On a shared segment the global committed LSN can include other pools'
        // frames, which this checkpoint does not materialize. Record this
        // pool's own committed watermark instead, so its coverage and the
        // reclaim floor both describe only frames this index holds.
        let wal_lsn = self.checkpoint_covered_lsn();
        // The image is captured here, under the locks, as bytes: the tail never
        // reads the live table. Pending frames were cleared by the rotation and
        // no put is mid-flight, so clearing the flag now loses nothing; a put
        // after the locks drop sets it again, and a failed tail sets it back.
        let blobs = Self::serialize_snapshots(&snapshots);
        breakdown.serialize = serialize_started.elapsed();
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        drop(guards);
        breakdown.snapshot = snapshot_started
            .elapsed()
            .saturating_sub(breakdown.serialize);
        // A collection published after this point lands after `fingerprint`
        // (its frames change a pack length), so a reopen that trusts C1 either
        // finds its frames in the new epoch or fails the tail-fingerprint gate
        // and rescans; it can never be silently omitted, because the image was
        // captured with publication excluded. The creation lock therefore drops
        // with the put mutexes rather than riding out the tail: every in-flight
        // sync appends to the new epoch through `append_index_delta_v3`, which
        // also takes this lock, so holding it across the tail would stall every
        // sync for the tail's duration and defeat the background checkpoint.
        drop(create_guard);
        let tail = CheckpointTail {
            base_dir: self.base_dir.clone(),
            fingerprint,
            covered_lsn: wal_lsn,
            base_delta_seq,
            blobs,
            pack_table,
            old_base_fingerprint,
            journal: self.journal(),
            pool_tag: self.journal_pool.load(Ordering::Acquire),
            durable_coverage: Arc::clone(&self.durable_coverage),
            #[cfg(test)]
            hook: self.checkpoint_tail_hook.lock().take(),
        };
        breakdown.total = started.elapsed();
        Ok(Some(CapturedCheckpoint { tail, breakdown }))
    }

    /// Serialize snapshots and write their checkpoint, for the test mirrors of
    /// the production capture and [`CheckpointTail::run`].
    #[cfg(test)]
    fn write_checkpoint_snapshot(
        path: &Path,
        fingerprint: u64,
        covered_lsn: u64,
        base_delta_seq: u64,
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
        pack_table: &[(u16, PackId)],
    ) -> std::io::Result<std::time::Duration> {
        let serialize_started = std::time::Instant::now();
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
        let serialized = serialize_started.elapsed();
        crate::index::checkpoint::write_checkpoint(
            path,
            fingerprint,
            covered_lsn,
            base_delta_seq,
            &blobs,
            pack_table,
        )?;
        Ok(serialized)
    }

    /// Serialize every snapshot's index into an owned checkpoint blob.
    fn serialize_snapshots(
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
    ) -> Vec<([u8; 16], u64, Vec<u8>)> {
        snapshots
            .iter()
            .map(|(collection_id, generation, room)| {
                (*collection_id, *generation, room.index.serialize())
            })
            .collect()
    }

    /// Fold a finished tail into the capture's breakdown and publish it.
    fn record_checkpoint(&self, mut breakdown: CheckpointBreakdown, done: &CheckpointBreakdown) {
        breakdown.write = done.write;
        breakdown.directory_sync = done.directory_sync;
        breakdown.journal_lsn = done.journal_lsn;
        breakdown.reclaim = done.reclaim;
        breakdown.retire = done.retire;
        breakdown.checkpoint_bytes = done.checkpoint_bytes;
        // The capture's own time plus the tail's, not the time until somebody
        // collected the worker.
        breakdown.total = breakdown.total.saturating_add(done.total);
        *self.last_checkpoint_breakdown.lock() = Some(breakdown);
    }

    /// A checkpoint whose tail failed did not become durable: the old
    /// checkpoint and its log still pair, but the epoch this session rotated onto
    /// names an image that does not exist. Forget it so the next sync rewrites
    /// the checkpoint rather than appending to an orphaned log.
    fn abandon_checkpoint(&self) {
        self.delta_state.lock().base_fingerprint = None;
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
    }

    /// Collect the background checkpoint tail. Returns whether none is left
    /// running. With `wait` it blocks until the tail ends; without, it only
    /// collects one that already has. A failed or panicked tail is reported and
    /// abandoned, and the old checkpoint and log stay valid.
    fn finish_checkpoint_worker(&self, wait: bool) -> bool {
        let mut slot = self.checkpoint_worker.lock();
        let Some(worker) = slot.take() else {
            return true;
        };
        if !wait && !worker.handle.is_finished() {
            *slot = Some(worker);
            return false;
        }
        drop(slot);
        match worker.handle.join() {
            Ok(Ok(done)) => self.record_checkpoint(worker.breakdown, &done),
            Ok(Err(error)) => {
                eprintln!("mtxdb: background checkpoint failed: {error}");
                self.abandon_checkpoint();
            }
            Err(_) => {
                eprintln!("mtxdb: background checkpoint panicked");
                self.abandon_checkpoint();
            }
        }
        true
    }

    /// Collect a finished checkpoint tail before a sync decides what to do.
    /// Returns `true` when none is still running.
    ///
    /// A tail still running holds the WAL: no coverage may be claimed and
    /// nothing reclaimed until its image is durable, and a second checkpoint
    /// must not start. It is waited for only when the WAL nears its cap or the
    /// delta log its rotation point (reclaim is never suppressed indefinitely).
    /// Otherwise this sync's operations go to the new epoch's log and the sync
    /// is done, which is why `false` means the caller returns.
    fn settle_checkpoint_worker(&self) -> bool {
        let must_wait = self
            .journal()
            .is_some_and(|journal| journal.in_emergency_zone())
            || self.delta_state.lock().log_bytes >= self.delta_log_rotate_bytes();
        let had_tail = self.checkpoint_worker.lock().is_some();
        if self.finish_checkpoint_worker(must_wait) {
            return true;
        }
        if had_tail {
            self.syncs_with_tail_in_flight
                .fetch_add(1, Ordering::Relaxed);
            if must_wait {
                self.syncs_waited_for_tail.fetch_add(1, Ordering::Relaxed);
            }
        }
        if self.index_checkpoint_dirty.load(Ordering::Relaxed)
            && !self.delta_state.lock().pending.is_empty()
        {
            match self.append_index_delta() {
                Ok(()) => self.shard_collections_stale.store(true, Ordering::Relaxed),
                Err(error) => {
                    eprintln!("mtxdb: delta append during a checkpoint failed: {error}");
                }
            }
        }
        false
    }

    /// Test-only: hold the next background checkpoint tail until the returned
    /// sender is used or dropped; with `fail` the tail then errors instead of
    /// installing.
    #[cfg(test)]
    pub(crate) fn hold_next_checkpoint_tail(&self, fail: bool) -> std::sync::mpsc::Sender<()> {
        let (release, gate) = std::sync::mpsc::channel();
        *self.checkpoint_tail_hook.lock() = Some(TailHook { gate, fail });
        release
    }

    /// Whether a checkpoint tail is running on its worker thread.
    #[must_use]
    pub fn checkpoint_in_flight(&self) -> bool {
        self.checkpoint_worker.lock().is_some()
    }

    /// Block until any background checkpoint has finished.
    pub fn wait_for_checkpoint(&self) {
        self.finish_checkpoint_worker(true);
    }

    /// Let a sync that needs a full checkpoint hand its tail (write, fsync,
    /// install, coverage, reclaim, retire) to a worker thread. Off by default.
    pub fn set_background_checkpoint(&self, enabled: bool) {
        self.background_checkpoint.store(enabled, Ordering::Relaxed);
    }

    /// Full checkpoint for a sync: capture under the locks, then run the tail on
    /// a worker when enabled, inline otherwise. Logs and swallows a failure like
    /// [`Self::persist_index_checkpoint_best_effort`].
    fn persist_index_checkpoint_for_sync(&self) {
        if !self.background_checkpoint.load(Ordering::Relaxed) {
            self.persist_index_checkpoint_best_effort();
            return;
        }
        if let Err(error) = self.start_checkpoint_worker() {
            eprintln!("mtxdb: failed to persist index checkpoint: {error}");
        }
    }

    /// Capture a checkpoint under the locks and hand its tail to a worker
    /// thread, returning once the capture is done. A failed spawn abandons the
    /// capture (the next sync rewrites the checkpoint).
    fn start_checkpoint_worker(&self) -> Result<(), StorageError> {
        let Some(CapturedCheckpoint { tail, breakdown }) = self.capture_checkpoint()? else {
            return Ok(());
        };
        let spawned = std::thread::Builder::new()
            .name("mtxdb-checkpoint".into())
            .spawn(move || tail.run());
        match spawned {
            Ok(handle) => {
                self.checkpoint_tails_started
                    .fetch_add(1, Ordering::Relaxed);
                *self.checkpoint_worker.lock() = Some(CheckpointWorker { handle, breakdown });
                Ok(())
            }
            Err(error) => {
                self.abandon_checkpoint();
                Err(StorageError::Io(error))
            }
        }
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
        state.log_synced_bytes = 0;
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
        let snapshots = self.checkpoint_snapshots();
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
        drop(guards);
        drop(create_guard);

        // The test waits on this: it now *knows* the rewrite is sleeping in
        // its unlocked window rather than guessing via a fixed sleep, so a
        // put issued from here provably overlaps the delay.
        entered_unlocked_window.store(true, Ordering::Release);

        std::thread::sleep(delay);

        let pack_table: Vec<(u16, PackId)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            self.delta_state.lock().delta_seq_high,
            &snapshots,
            &pack_table,
        )
        .map_err(StorageError::Io)?;
        let _ = crate::shard::sync_directory(&self.base_dir);
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

    /// Test-only: turn a copy of this pool's directory into the disk a power
    /// cut would leave, by cutting every pack back to the length an fsync has
    /// covered. Bytes written but not yet fsynced are what a crash may lose;
    /// bytes still buffered in memory are not in the copy at all.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn test_cut_packs_to_synced(&self, image_dir: &Path) {
        for (_, shard) in self.shards.all_shards() {
            let name = shard.path.file_name().expect("a pack has a file name");
            let file = fs::OpenOptions::new()
                .write(true)
                .open(image_dir.join(name))
                .expect("the image holds every pack");
            file.set_len(shard.synced_len().min(shard.file_len()))
                .expect("cut the pack");
        }
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
        let snapshots = self.checkpoint_snapshots();
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
        let snapshots = self.checkpoint_snapshots();
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
        drop(guards);
        let pack_table: Vec<(u16, PackId)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            self.delta_state.lock().delta_seq_high,
            &snapshots,
            &pack_table,
        )
        .map_err(StorageError::Io)?;
        let _ = crate::shard::sync_directory(&self.base_dir);
        Ok((fingerprint, old_base_fingerprint))
    }

    /// Remove the now-superseded delta-log epoch, once the checkpoint that
    /// makes it safe to discard is durably renamed. A no-op if this session
    /// had no prior epoch. Best-effort: a failure here leaves the retired
    /// epoch's file on disk, which is always safe (see `delta_path`) — it
    /// simply names a fingerprint no future checkpoint will ever carry again.
    #[cfg(test)]
    fn retire_delta_epoch(&self, old_base_fingerprint: Option<u64>) {
        retire_delta_epoch_file(&self.base_dir, old_base_fingerprint);
    }

    /// Append the pending delta operations and clear the dirty flag. The
    /// on-disk log, if any, is
    /// continued; otherwise a fresh header pins `base_fingerprint` (the
    /// checkpoint the frames extend) for the reopen replay gate.
    fn append_index_delta(&self) -> Result<(), StorageError> {
        self.append_index_delta_v3(false).map(|_| ())
    }

    /// The delta-log cap, the bytes still free under it for a batch, and the
    /// length of a batch before any operation is added to it.
    fn delta_batch_budget(&self, state: &DeltaLogState, with_coverage: bool) -> (u64, u64, usize) {
        let cap = self.delta_log_cap();
        let header = if state.log_bytes == 0 {
            DELTA_LOG_HEADER_LEN as u64
        } else {
            0
        };
        let budget = cap.saturating_sub(state.log_bytes).saturating_sub(header);
        let mut projected = delta::v3_empty_batch_len();
        if with_coverage {
            projected = projected.saturating_add(delta::v3_coverage_frame_len());
        }
        (cap, budget, projected)
    }

    fn build_v3_pending_batch(
        &self,
        state: &DeltaLogState,
        with_coverage: bool,
    ) -> Result<V3PendingBatch, StorageError> {
        // The batch's length is known from its parts without building them, so
        // one that cannot fit under the delta-log cap is refused before a
        // whole-index snapshot is serialized just to be thrown away.
        let (cap, budget, mut projected) = self.delta_batch_budget(state, with_coverage);
        let too_large = |projected: usize| {
            StorageError::Io(std::io::Error::other(DeltaBatchTooLarge {
                batch_bytes: projected,
                log_bytes: state.log_bytes,
                cap,
            }))
        };
        let collections = self.collections_read();
        let mut operations = Vec::new();
        let mut redo: Vec<RedoRecord> = Vec::new();
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
                PendingDelta::Redo(records) => {
                    let Some(base_generation) = state.base_generations.get(&collection_id) else {
                        return Err(StorageError::Io(std::io::Error::other(
                            "redo records target a collection without a base",
                        )));
                    };
                    if records
                        .iter()
                        .any(|record| record.base_generation != *base_generation)
                    {
                        return Err(StorageError::Io(std::io::Error::other(
                            "a redo record's generation differs from its base",
                        )));
                    }
                    projected = projected
                        .saturating_add(records.len().saturating_mul(delta::v3_redo_frame_len()));
                    redo.extend(records.iter().copied());
                }
                PendingDelta::Snapshot => {
                    if let Some(room) = collections.get(&collection_id) {
                        let generation = arc_swap::ArcSwapAny::load_full(room);
                        let frame_len =
                            delta::v3_snapshot_frame_len(generation.index.serialized_len())
                                .ok_or_else(|| {
                                    StorageError::Io(std::io::Error::other(
                                        "v3 batch size overflow",
                                    ))
                                })?;
                        projected = projected.saturating_add(frame_len);
                        if u64::try_from(projected).unwrap_or(u64::MAX) > budget {
                            return Err(too_large(projected));
                        }
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
                        projected = projected.saturating_add(delta::v3_tombstone_frame_len());
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
                PendingDelta::Delete { generation } => {
                    if state.base_generations.contains_key(&collection_id) {
                        projected = projected.saturating_add(delta::v3_tombstone_frame_len());
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation: *generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
            }
        }
        // Serialized in `delta_seq` order, whatever the collection order, so a
        // reader that validates strict monotonicity sees the append order.
        redo.sort_unstable_by_key(|record| record.delta_seq);
        operations.extend(redo.into_iter().map(DeltaOperation::Redo));
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
        durable: bool,
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
        let cap = self.delta_log_cap();
        if next_len > cap {
            return Err(StorageError::Io(std::io::Error::other(
                DeltaBatchTooLarge {
                    batch_bytes,
                    log_bytes: state.log_bytes,
                    cap,
                },
            )));
        }
        let bytes_written = delta::append_v3_batch_with_durability(
            &path,
            write_header,
            base_fingerprint,
            operations,
            tail_fingerprint,
            durable,
        )
        .map_err(StorageError::Io)?;
        state.log_bytes = state
            .log_bytes
            .saturating_add(u64::try_from(bytes_written).unwrap_or(u64::MAX));
        if durable {
            state.log_synced_bytes = state.log_bytes;
        } else {
            // Pay for the log's durability in slices as it grows. Best effort:
            // the log is acceleration, and a coverage batch fsyncs it again.
            let budget = self.delta_fsync_budget_bytes.load(Ordering::Relaxed);
            if budget != 0 && state.log_bytes.saturating_sub(state.log_synced_bytes) >= budget {
                match fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .and_then(|file| file.sync_data())
                {
                    Ok(()) => state.log_synced_bytes = state.log_bytes,
                    Err(error) => eprintln!("mtxdb: delta log fsync failed: {error}"),
                }
            }
        }
        Ok(())
    }

    /// Append v3 operations. Collection locks pin every live index
    /// and the collection-creation lock prevents an unrepresented collection
    /// from appearing between pack flush and snapshot capture.
    ///
    /// With `with_coverage` the batch also carries a claim that the packs
    /// durably cover this pool's journal frames through the returned LSN, so
    /// the journal can be reclaimed without a full checkpoint. The ordering that
    /// makes the claim true: with every collection locked (so no put is
    /// mid-flight and everything published for this pool has been applied) the
    /// LSN is captured and every pack is fsynced, and only then is the batch,
    /// claim last, appended and fsynced. A crash before the batch is durable
    /// leaves no claim; a torn batch fails its CRC and claims nothing.
    fn append_index_delta_v3(&self, with_coverage: bool) -> Result<Option<u64>, StorageError> {
        if with_coverage {
            // Make the bulk of the pack bytes durable before any lock is taken;
            // the fsync under the locks below then only covers what was written
            // since.
            self.shards.sync_dirty()?;
        }
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
        let covered = if with_coverage {
            let covered = self.checkpoint_covered_lsn();
            // Everything up to `covered` is applied and flushed; make it durable
            // before claiming it. `sync_all`, not `sync_dirty`: a claim must not
            // rest on a dirty bit some other syncer cleared.
            self.shards.sync_all()?;
            covered
        } else {
            None
        };
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
        if snapshot_state.pending.is_empty() && covered.is_none() {
            return Err(StorageError::Io(std::io::Error::other(
                "v3 delta append with no pending operations",
            )));
        }

        let batch = self.build_v3_pending_batch(&snapshot_state, with_coverage)?;
        let V3PendingBatch {
            mut operations,
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
        // The claim is the batch's last operation, so a reader that applies the
        // operations up to it holds exactly the index the claim describes.
        if let Some(covered_lsn) = covered {
            operations.push(DeltaOperation::Coverage { covered_lsn });
        }
        self.write_v3_delta_batch(
            &mut state,
            base_fingerprint,
            &operations,
            tail_fingerprint,
            covered.is_some(),
        )?;
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
        if let Some(covered_lsn) = covered {
            // The batch is durable, so the claim is too.
            self.durable_coverage
                .fetch_max(covered_lsn, Ordering::AcqRel);
        }
        Ok(covered)
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
    ) -> Option<HashMap<[u8; 16], Vec<PackId>>> {
        let records = read_persisted_shard_collections(base_dir)?.records;
        let mut shards: HashMap<[u8; 16], Vec<PackId>> = HashMap::new();
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
    pub fn shard_node_counts_from_disk(base_dir: &std::path::Path) -> Option<HashMap<PackId, u64>> {
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
    ) -> Option<HashMap<PackId, u64>> {
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

    /// Builds this repack's deduped `hash → (slot, offset)` map,
    /// incrementally where possible, and returns the previous
    /// [`RepackIncrementalState`] (moved out, not cloned) so the caller can
    /// reuse its `adjacency` cache when deciding which nodes need fresh disk
    /// reads.
    ///
    /// On the incremental path a copy of the previous `live_map` is extended
    /// with only the newly-appended entries from each shard (O(delta) scan
    /// work per repack call rather than O(total)). The first call, or a change
    /// in a previously scanned slot's pack identity, triggers a full scan.
    ///
    /// Returns `(hash_to_shard_offset, Some(prev))` on the incremental path
    /// and `(hash_to_shard_offset, None)` on the cold-start path. The caller
    /// must pass `prev` to the adjacency helpers to unlock the incremental
    /// BFS optimization; when it is `None` the caller uses the full-scan
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
            let current_pack_ids: HashMap<u16, PackId> = shards
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
    pub fn collection_referenced_pack_ids(&self, collection_id: &[u8; 16]) -> Vec<PackId> {
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
    pub fn shard_node_counts(&self) -> HashMap<PackId, u64> {
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

    /// Rewrite a collection's records in topological order, optionally performing
    /// garbage collection.
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
    /// Shard writes are synced before success. Checkpoint and inspection
    /// sidecar persistence are best-effort.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption, including a cycle in the
    /// live graph.
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
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
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
    ) -> Result<(RepackOffsets, usize, HashMap<PackId, u64>), StorageError> {
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
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
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
    /// `stop_at` — a bounded ancestor walk, e.g. for edges or
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
        let live_pack_ids: HashSet<PackId> = tables
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
        let track = self.stats_enabled.load(Ordering::Relaxed);
        let _latency = if track {
            OperationTimer::new(&self.operation_timings.get_many_with_refresh)
        } else {
            OperationTimer::disabled()
        };
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
        if let Some(shard) = self.shards.get_shard(slot) {
            progress
                .pending_shard_collections
                .push((shard.pack_id, disk_bytes));
        }
        if progress.index_needs_rebuild {
            return Ok(());
        }
        let locator = RecordLocator {
            slot,
            offset,
            len: disk_bytes,
        };
        self.index_put_many_entry(collection_id, id, old_gen, locator, progress)
    }

    fn index_put_many_entry(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        old_gen: Option<&RoomGeneration>,
        locator: RecordLocator,
        progress: &mut PutManyProgress,
    ) -> Result<(), StorageError> {
        let RecordLocator {
            slot,
            offset,
            len: record_len,
        } = locator;
        let live = match progress.owned_index.as_ref() {
            Some(index) => index,
            None => {
                &old_gen
                    .expect("live path implies an existing generation")
                    .index
            }
        };
        let insert_result = self.insert_index_undoable(collection_id, live, id, slot, offset)?;
        let inserted = if let Ok((_bucket, _entry, undo)) = insert_result {
            let was_empty = undo.was_empty();
            progress
                .pending_deltas
                .push((*id, slot, offset, record_len));
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
            // Growth is logical: the entry that hit the boundary is logged like
            // the others and replay grows the table, so no snapshot is needed.
            progress
                .pending_deltas
                .push((*id, slot, offset, record_len));
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
            if let Some(shard) = self.shards.get_shard(slot) {
                progress.pending_shard_collection_counts.push(shard.pack_id);
            }
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
                if let Ok((_bucket, _entry, inserted)) =
                    self.insert_index(collection_id, &gen.index, id, slot, offset)?
                {
                    self.record_redo(collection_id, gen.generation, id, slot, offset, disk_bytes);
                    let pack_id = self.pack_id_for_slot(slot)?;
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
            if let Ok((_bucket, _entry, is_new)) = inserted {
                self.record_redo(collection_id, generation, id, slot, offset, disk_bytes);
                let pack_id = self.pack_id_for_slot(slot)?;
                if is_new {
                    self.record_new_shard_collection(pack_id, collection_id, disk_bytes);
                } else {
                    self.record_disk_bytes(pack_id, collection_id, disk_bytes);
                }
            } else {
                // The insert was rejected (table full). Capacity growth changes
                // the table's shape but not what a logical redo record says, so
                // it does not invalidate the delta log: the record that hit the
                // boundary is logged like any other, and replay grows the table
                // itself. Only a rebuild re-derives the collection wholesale.
                if let Some(grown) = index.grow() {
                    // The failed insert did not mutate the table, so retry it
                    // after the in-memory rehash. This is the normal capacity
                    // path and must not turn into a full-pack scan.
                    let _ = grown.insert(id, slot, offset);
                    index = grown;
                    self.record_redo(collection_id, generation, id, slot, offset, disk_bytes);
                } else if let Ok(Some(grown)) = self.grow_checkpoint_index(collection_id, &index) {
                    // A checkpoint-backed index has locations but not homes.
                    // Recovering the hashes from those locations is bounded
                    // by this collection, unlike `rebuild_index`'s pack scan.
                    let _ = grown.insert(id, slot, offset);
                    index = grown;
                    self.record_redo(collection_id, generation, id, slot, offset, disk_bytes);
                } else {
                    self.invalidate_delta_log(collection_id);
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
                let pack_id = self.pack_id_for_slot(slot)?;
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
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        let written = self.put_many_internal_locked(collection_id, entries, metadatas)?;
        if written > 0 {
            self.put_many_calls.fetch_add(1, Ordering::Relaxed);
            self.put_many_records
                .fetch_add(written as u64, Ordering::Relaxed);
            self.put_many_bytes.fetch_add(
                entries
                    .iter()
                    .map(|(_, data)| data.bytes.len() as u64)
                    .sum::<u64>(),
                Ordering::Relaxed,
            );
        }
        Ok(written)
    }

    fn put_many_internal_locked(
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
        if !Self::validate_put_many_inputs(collection_id, entries)? {
            return Ok(0);
        }

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
            for (id, slot, offset, record_len) in progress.pending_deltas {
                self.record_redo(
                    collection_id,
                    progress.generation,
                    &id,
                    slot,
                    offset,
                    record_len,
                );
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

    /// The physical location `(pack address, byte offset)` of the durable frame
    /// `id` resolves to, using the same candidate selection and pin/retry path
    /// as [`StorageEngine::get`]. Returns `None` if `id` is not live.
    ///
    /// This deliberately returns only a **durable pack location**, not "the
    /// frame `get` would return": `get` can serve a generation-local cache hit
    /// or a transaction-overlay value, neither of which has a pack location.
    /// While a transaction overlay is active this returns an error rather than
    /// silently reporting durable state as if it were current. The overlay
    /// check is point-in-time; a transaction may become active immediately
    /// after, so callers must not treat a success as a globally frozen view.
    /// Callers that need the winning frame's identity (e.g. `export --format
    /// envelope`) use it to avoid re-deriving a winner from physical scan order,
    /// which is meaningless now that pack addresses are random.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a candidate frame cannot be read, or
    /// [`StorageError::Internal`] while a transaction overlay is active.
    pub fn get_location(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
    ) -> Result<Option<(PackId, u64)>, StorageError> {
        if self.overlay_reads_active() {
            return Err(StorageError::Internal(
                "get_location does not resolve transaction-overlay reads".to_owned(),
            ));
        }
        // As in `get`: a concurrent repack can swap the generation and retire
        // the shard slots its index pointed at between the lookup and the pin,
        // so confirm the generation is unchanged and retry if it moved.
        loop {
            let gen_guard = self.generation(collection_id);
            let Some(gen) = gen_guard.as_deref() else {
                return Ok(None);
            };
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
            return self.resolve_location_from_pinned(id, &candidates, &pinned);
        }
    }

    /// Resolve an `id` to its winning `(PackId, offset)` from already-pinned
    /// candidates, retaining the last read error rather than reporting a
    /// missing record. Mirrors [`Self::resolve_from_pinned`]'s candidate
    /// semantics so the two readers agree.
    fn resolve_location_from_pinned(
        &self,
        id: &NodeId,
        candidates: &[(u16, u64)],
        pinned: &HashMap<u16, Arc<Shard>>,
    ) -> Result<Option<(PackId, u64)>, StorageError> {
        let mut last_err: Option<StorageError> = None;
        for &(slot, offset) in candidates {
            let Some(shard) = pinned.get(&slot) else {
                continue;
            };
            match self.read_at(
                shard,
                offset,
                self.shards.checksum_policy().verifies_reads(),
            ) {
                Ok(record) => {
                    if record.hash == *id {
                        return Ok(Some((shard.pack_id, offset)));
                    }
                }
                Err(error) => last_err = Some(error),
            }
        }
        if let Some(error) = last_err {
            return Err(error);
        }
        Ok(None)
    }
}

impl StorageEngine for PackfileStorage {
    fn collection_exists(&self, collection_id: &[u8; 16]) -> Result<bool, StorageError> {
        self.try_collection_exists(collection_id)
    }

    fn collection_len(&self, collection_id: &[u8; 16]) -> Result<Option<usize>, StorageError> {
        self.try_collection_len(collection_id)
    }

    fn create_or_put_established(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        validate_established_batch_inputs(records)?;
        if !metadata.verify_collection_id(collection_id) {
            return Err(StorageError::Internal(
                "collection metadata does not reproduce collection id".to_owned(),
            ));
        }
        if metadata.collection_canonical_id.is_empty() {
            return Err(StorageError::Internal(
                "collection metadata has empty canonical id".to_owned(),
            ));
        }

        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        if let Some(existing) = self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? {
            let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                found.validate_identity_collision(metadata, collection_id)?;
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }
            let mut to_append = Vec::with_capacity(records.len());
            for (id, data) in records {
                if let Some(existing_rec) = self.get(collection_id, id)? {
                    if existing_rec.bytes != data.bytes {
                        return Err(StorageError::Collision(format!(
                            "record collision on node {}",
                            hex16(id)
                        )));
                    }
                } else {
                    to_append.push((*id, data.clone()));
                }
            }
            if to_append.len() == 1 {
                let (id, data) = &to_append[0];
                return self.put_internal_locked(collection_id, id, data, None);
            } else if !to_append.is_empty() {
                self.put_many_internal_locked(collection_id, &to_append, None)?;
            }
        } else {
            if self.collection_exists(collection_id) {
                return Err(StorageError::Internal(
                    "genesis metadata must be written before the collection's first record"
                        .to_owned(),
                ));
            }
            let mut batch = Vec::with_capacity(records.len().saturating_add(1));
            batch.push((
                COLLECTION_METADATA_RECORD_ID,
                NodeData::new(metadata.encode().into()),
            ));
            batch.extend_from_slice(records);
            self.put_many_internal_locked(collection_id, &batch, None)?;
        }
        Ok(())
    }

    fn put_many_established(
        &self,
        collection_id: &[u8; 16],
        records: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        validate_established_batch_inputs(records)?;
        if records.is_empty() {
            return Ok(());
        }

        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        let Some(existing) = self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? else {
            return Err(StorageError::NotFound(*collection_id));
        };
        let found = CollectionMetadata::decode(&existing.bytes).ok_or_else(|| {
            StorageError::Corrupt("malformed collection metadata record".to_owned())
        })?;
        if !found.verify_collection_id(collection_id) {
            return Err(StorageError::Corrupt(
                "stored collection metadata does not reproduce collection id".to_owned(),
            ));
        }

        let to_append =
            collect_missing_established_records(records, |id| self.get(collection_id, id))?;
        if !to_append.is_empty() {
            let written = self.put_many_internal_locked(collection_id, &to_append, None)?;
            if written > 0 {
                self.put_many_calls.fetch_add(1, Ordering::Relaxed);
                self.put_many_records
                    .fetch_add(written as u64, Ordering::Relaxed);
                self.put_many_bytes.fetch_add(
                    to_append
                        .iter()
                        .map(|(_, data)| data.bytes.len() as u64)
                        .sum::<u64>(),
                    Ordering::Relaxed,
                );
            }
        }
        Ok(())
    }

    fn create_or_upsert_established_validated(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
        node_id: &NodeId,
        data: &NodeData,
        validate: &mut dyn FnMut(Option<&NodeData>) -> Result<(), StorageError>,
    ) -> Result<(), StorageError> {
        validate_established_upsert_inputs(collection_id, metadata, node_id)?;

        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        if let Some(existing_meta_rec) = self.get(collection_id, &COLLECTION_METADATA_RECORD_ID)? {
            let found = CollectionMetadata::decode(&existing_meta_rec.bytes).ok_or_else(|| {
                StorageError::Corrupt("malformed collection metadata record".to_owned())
            })?;
            if found != *metadata {
                found.validate_identity_collision(metadata, collection_id)?;
                return Err(StorageError::Internal(
                    "collection metadata mismatch: existing genesis record differs".to_owned(),
                ));
            }

            let existing = self.get(collection_id, node_id)?;
            validate(existing.as_ref())?;

            if let Some(existing) = existing {
                if existing.bytes == data.bytes {
                    return Ok(());
                }
            }

            self.put_internal_locked(collection_id, node_id, data, None)?;
        } else {
            if self.collection_exists(collection_id) {
                return Err(StorageError::Internal(
                    "genesis metadata must be written before the collection's first record"
                        .to_owned(),
                ));
            }

            validate(None)?;

            let batch = [
                (
                    COLLECTION_METADATA_RECORD_ID,
                    NodeData::new(metadata.encode().into()),
                ),
                (*node_id, data.clone()),
            ];
            self.put_many_internal_locked(collection_id, &batch, None)?;
        }

        Ok(())
    }

    fn ensure_collection_metadata(
        &self,
        collection_id: &[u8; 16],
        metadata: &CollectionMetadata,
    ) -> Result<(), StorageError> {
        self.create_or_put_established(collection_id, metadata, &[])
    }

    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let track = self.stats_enabled.load(Ordering::Relaxed);
        let _latency = if track {
            OperationTimer::new(&self.operation_timings.get)
        } else {
            OperationTimer::disabled()
        };
        if self.overlay_reads_active() {
            return Ok(self
                .get_read_committed(collection_id, std::slice::from_ref(id))?
                .into_iter()
                .next()
                .flatten());
        }
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
        let _latency = if track {
            OperationTimer::new(&self.operation_timings.get_many)
        } else {
            OperationTimer::disabled()
        };
        if self.overlay_reads_active() {
            return self.get_read_committed(collection_id, ids);
        }
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

            let policy = self.read_plan_policy();
            let plan_wanted = policy.wants(usize::try_from(candidates_seen).unwrap_or(usize::MAX));

            // Suppress readahead for a scattered batch when asked. This is
            // independent of extent planning: it reads the same pages either
            // way, it just stops the kernel from over-reading first.
            if policy.random_advice && candidates_seen > 0 {
                Self::advise_random(&pinned);
            }

            // One grouping of candidate offsets per shard feeds both the
            // scatter counters and the merged read plan below.
            let mut per_shard: HashMap<u16, Vec<u64>> = HashMap::new();
            if track || plan_wanted {
                for (_, candidates) in &to_fetch {
                    for (slot, offset) in candidates {
                        per_shard.entry(*slot).or_default().push(*offset);
                    }
                }
                for offsets in per_shard.values_mut() {
                    offsets.sort_unstable();
                }
            }

            if track {
                self.index_candidates
                    .fetch_add(candidates_seen, Ordering::Relaxed);
                self.get_many_shards_touched
                    .fetch_add(per_shard.len() as u64, Ordering::Relaxed);

                let mut runs: u64 = 0;
                let mut span: u64 = 0;
                for offsets in per_shard.values() {
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

            // Meld nearby candidates into contiguous extents and hint the
            // kernel to read each as one sequential run, before the
            // per-candidate decode loop touches them one at a time.
            if plan_wanted {
                let extents = plan_read_extents(&per_shard, policy);
                self.prefetch_extents(&extents, &pinned, track);
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
        let _latency = if self.stats_enabled.load(Ordering::Relaxed) {
            OperationTimer::new(&self.operation_timings.put)
        } else {
            OperationTimer::disabled()
        };
        self.put_internal(collection_id, id, data, None)
    }

    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<usize, StorageError> {
        let _latency = if self.stats_enabled.load(Ordering::Relaxed) {
            OperationTimer::new(&self.operation_timings.put_many)
        } else {
            OperationTimer::disabled()
        };
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
        // Remediation runs after the journal barrier and is part of what the
        // caller waited for: a sync that unblocks a stalled shared WAL pays the
        // capture (or, without background checkpoints, the whole checkpoint).
        if result.is_ok() {
            let remediation_started = std::time::Instant::now();
            self.remediate_lagging_pools();
            timings.remediation = remediation_started.elapsed();
        }
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
    /// Check collection existence using the historical boolean API.
    ///
    /// Call [`Self::try_collection_exists`] when storage errors must be
    /// distinguished from a missing collection.
    #[must_use]
    pub fn collection_exists(&self, collection_id: &[u8; 16]) -> bool {
        self.try_collection_exists(collection_id).unwrap_or(false)
    }

    /// Check collection existence without hiding overlay/read failures.
    ///
    /// The trait's historical boolean API is retained for compatibility and
    /// fails closed on errors. New callers that need to distinguish absence
    /// from storage failure should use this method.
    ///
    /// # Errors
    /// Returns a storage or journal-overlay error when the authoritative
    /// read-committed lookup cannot be completed.
    pub fn try_collection_exists(&self, collection_id: &[u8; 16]) -> Result<bool, StorageError> {
        if self.transaction_overlay_users.load(Ordering::Acquire) != 0 {
            return self
                .get_read_committed(
                    collection_id,
                    std::slice::from_ref(&COLLECTION_METADATA_RECORD_ID),
                )
                .map(|values| values.first().is_some_and(Option::is_some));
        }
        Ok(self.generation(collection_id).is_some())
    }

    /// Return collection length at read-committed visibility.
    ///
    /// Overlay puts replace durable values with the same node ID and add to
    /// the durable count only for new IDs. A committed collection delete
    /// hides the durable generation until replacement puts are applied.
    ///
    /// # Errors
    /// Returns a storage or journal-overlay error when the authoritative
    /// read-committed view cannot be refreshed.
    pub fn try_collection_len(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Option<usize>, StorageError> {
        if self.transaction_overlay_users.load(Ordering::Acquire) == 0 {
            return Ok(self
                .generation(collection_id)
                .map(|generation| generation.index.len()));
        }

        let overlay_guard = self.refresh_read_journal()?;
        let Some(overlay) = overlay_guard.as_ref() else {
            return Ok(self
                .generation(collection_id)
                .map(|generation| generation.index.len()));
        };
        let puts = overlay.puts.get(collection_id);
        let deleted = overlay.delete_lsn.contains_key(collection_id);
        let generation = self.generation(collection_id);
        let durable_len = generation
            .as_ref()
            .map_or(0, |generation| generation.index.len());

        if deleted {
            return Ok(puts
                .filter(|puts| !puts.is_empty())
                .map(std::collections::HashMap::len));
        }

        let Some(puts) = puts else {
            return Ok(generation.map(|_| durable_len));
        };
        let additions = puts
            .keys()
            .filter(|node_id| {
                generation.as_ref().map_or(true, |generation| {
                    generation.index.lookup_all(node_id).next().is_none()
                })
            })
            .count();
        let length = durable_len.saturating_add(additions);
        Ok(if generation.is_some() || !puts.is_empty() {
            Some(length)
        } else {
            None
        })
    }

    /// Hint the kernel to read each planned extent as one sequential run.
    ///
    /// Reads go through the shard's mmap, so the physical coalescing is a
    /// `madvise(MADV_WILLNEED)` over the merged range rather than a `pread`
    /// into a buffer: the existing zero-copy decode path is unchanged, and the
    /// kernel is free to queue the readahead asynchronously while resolution
    /// proceeds. Best-effort: an extent that cannot be advised (missing
    /// mapping, offset beyond the current mapping, or a failed `madvise`) is
    /// counted in `read_plan_skipped_extents` rather than silently dropped.
    #[cfg(unix)]
    fn prefetch_extents(
        &self,
        extents: &[ReadExtent],
        pinned: &HashMap<u16, Arc<Shard>>,
        track: bool,
    ) {
        for extent in extents {
            match Self::prefetch_extent(extent, pinned) {
                Some(len) => {
                    if track {
                        self.read_plan_extents.fetch_add(1, Ordering::Relaxed);
                        self.read_plan_prefetch_bytes
                            .fetch_add(len, Ordering::Relaxed);
                    }
                }
                None => {
                    if track {
                        self.read_plan_skipped_extents
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Advise every pinned shard mapping `MADV_RANDOM` before a scattered
    /// batch read, so the kernel does not let readahead amplify each fault
    /// into a large sequential block read. This is the no-plan alternative to
    /// [`Self::prefetch_extents`]: it does not merge or prefetch anything, it
    /// only stops readahead from over-reading beyond the pages actually
    /// touched. A shard whose mapping is absent or whose `madvise` fails is
    /// simply left alone — the read still works, just with default advice.
    #[cfg(unix)]
    fn advise_random(pinned: &HashMap<u16, Arc<Shard>>) {
        for shard in pinned.values() {
            if let Ok(guard) = shard.mmap() {
                if let Some(mapping) = guard.as_ref() {
                    let _ = mapping.advise(memmap2::Advice::Random);
                }
            }
        }
    }

    /// Non-Unix builds have no `madvise`; readahead advice is a no-op.
    #[cfg(not(unix))]
    fn advise_random(_pinned: &HashMap<u16, Arc<Shard>>) {}

    /// `madvise(MADV_WILLNEED)` one extent, returning its byte length on
    /// success and `None` if it could not be advised.
    ///
    /// `advise_range` requires the offset and length to lie within the mapping;
    /// clamp rather than error, since a virtual (buffered-but-unflushed) offset
    /// is legitimately beyond the current mapping and simply cannot be
    /// prefetched until the shard flushes.
    #[cfg(unix)]
    fn prefetch_extent(extent: &ReadExtent, pinned: &HashMap<u16, Arc<Shard>>) -> Option<u64> {
        let shard = pinned.get(&extent.slot)?;
        let guard = shard.mmap().ok()?;
        let mapping = guard.as_ref()?;
        let start = usize::try_from(extent.start).ok()?;
        let end = usize::try_from(extent.end)
            .unwrap_or(usize::MAX)
            .min(mapping.len());
        if start >= end {
            return None;
        }
        let len = end.saturating_sub(start);
        mapping
            .advise_range(memmap2::Advice::WillNeed, start, len)
            .ok()?;
        Some(u64::try_from(len).unwrap_or(u64::MAX))
    }

    /// Non-Unix builds have no `madvise`; every planned extent is skipped.
    #[cfg(not(unix))]
    fn prefetch_extents(
        &self,
        extents: &[ReadExtent],
        _pinned: &HashMap<u16, Arc<Shard>>,
        track: bool,
    ) {
        if track {
            self.read_plan_skipped_extents
                .fetch_add(extents.len() as u64, Ordering::Relaxed);
        }
    }

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
            .and_then(|()| self.persist_index_checkpoint().map(|_| ()))
        {
            Ok(()) => error,
            Err(boundary_error) => boundary_error,
        }
    }

    /// Sync all open shards to disk (full pool, not just dirty).
    ///
    /// A full checkpoint rewrite also persists the shard→collection directory
    /// as a side effect — the rewrite is an anchor at which this observability
    /// sidecar is re-pinned to the same pack fingerprint the checkpoint just
    /// became. A delta-only barrier defers it; see
    /// `persist_index_checkpoint_or_delta` for the deferral policy.
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
        if result.is_ok() {
            let remediation_started = std::time::Instant::now();
            self.remediate_lagging_pools();
            timings.remediation = remediation_started.elapsed();
        }
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
        let wrote = self.persist_index_checkpoint()?;
        // A forced rewrite is a full checkpoint like a sync's, but it never
        // passes through `sync`/`sync_all`, so `count_sync_persistence` cannot
        // see it. Without this, a pool touched only by WAL-reclaim
        // remediation reports zero checkpoint writes despite rewriting its
        // checkpoint. A sync can win `index_persist_lock` between the dirty
        // store above and this lock and consume the flag, writing and counting
        // the checkpoint through its own accounting; count only when this call
        // actually wrote one, or the metric would double-count.
        if wrote {
            self.checkpoint_writes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Checkpoint to unblock WAL reclaim, detaching the tail when the
    /// background checkpoint is enabled.
    ///
    /// [`Self::force_index_checkpoint`] always joins and runs inline, so a
    /// shared-WAL remediation that forced it would pay the whole checkpoint on
    /// the syncing thread — exactly what the background path exists to avoid.
    /// This captures under the locks and spawns the tail, returning once the
    /// capture is done; the tail reports coverage when its image is durable.
    /// Falls back to the synchronous path when backgrounding is off or a
    /// capture cannot start.
    ///
    /// # Errors
    /// Propagates any shard-fsync failure from the synchronous fallback.
    pub fn force_index_checkpoint_detached(&self) -> Result<(), StorageError> {
        if !self.background_checkpoint.load(Ordering::Relaxed) {
            return self.force_index_checkpoint();
        }
        self.shards.sync_dirty().map_err(StorageError::Io)?;
        // Serialize with this pool's own syncs, and never start a second tail:
        // it would rotate the epoch again and race the first on the checkpoint
        // file. A tail already running is what remediation wanted; its coverage
        // lands when its image is durable, and a sync near the WAL cap waits for
        // it. Only mark the index dirty when a checkpoint really starts, or the
        // next sync would rewrite an image nothing needs.
        let _persist_guard = self.index_persist_lock.lock();
        if !self.finish_checkpoint_worker(false) {
            return Ok(());
        }
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.start_checkpoint_worker()?;
        // A forced rewrite is a full checkpoint like a sync's, but it never
        // passes through `sync`/`sync_all`, so `count_sync_persistence` cannot
        // see it. Count it at hand-off, matching that barrier accounting (which
        // counts a backgrounded sync checkpoint at capture, not at tail
        // completion). Without this, a pool touched only by WAL-reclaim
        // remediation reports zero checkpoint writes when background
        // checkpointing is on and one when it is off.
        self.checkpoint_writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
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

    /// Set how many written-but-unsynced pack bytes make a journal-mode sync also
    /// fsync the dirty packs; zero disables it and leaves pack fsyncs to the
    /// checkpoint. This only schedules fsyncs earlier: coverage still comes from
    /// the checkpoint's own fsync of every shard, so it never claims pack data
    /// the packs have not made durable.
    pub fn set_pack_fsync_budget(&self, bytes: u64) {
        self.pack_fsync_budget_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Set how many unsynced delta-log bytes make a delta batch also fsync the
    /// log; zero disables it. Like the pack budget it only schedules an fsync
    /// earlier, so a later coverage batch, which is always fsynced, pays for
    /// less.
    pub fn set_delta_fsync_budget(&self, bytes: u64) {
        self.delta_fsync_budget_bytes
            .store(bytes, Ordering::Relaxed);
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
    /// Returns [`StorageError::Internal`] for a pool inside a database root;
    /// such pools must use `enable_shared_journal` (with `multi-reader`).
    /// Propagates root-inspection, journal-open, recovery, and publish-signal
    /// setup errors, including OS entropy failures, as [`StorageError::Io`].
    pub fn enable_journal(&self, path: impl AsRef<std::path::Path>) -> Result<(), StorageError> {
        self.reject_per_pool_journal_in_root()?;
        let (journal, scan) = Journal::open(path).map_err(StorageError::Io)?;
        self.journal_recovery.lock().clone_from(&scan.groups);
        let coordinator = Arc::new(JournalCoordinator::new(journal, &scan));
        coordinator
            .enable_publish_signal()
            .map_err(StorageError::Io)?;
        *self.journal.lock() = Some(coordinator);
        Ok(())
    }

    /// Fail closed when a path-based per-pool journal would be attached to a
    /// store that lives inside a database root.
    ///
    /// Every root is driven by one root-level pool-tagged segment through one
    /// coordinator. Letting a caller hand such a pool a private per-pool
    /// `wal.bin` would split the durability fence and silently diverge from the
    /// root's WAL. A standalone store outside any root keeps the per-pool
    /// journal path.
    fn reject_per_pool_journal_in_root(&self) -> Result<(), StorageError> {
        match crate::layout::enclosing_root(&self.base_dir)? {
            Some(root) => Err(StorageError::Internal(format!(
                "{} is inside database root {}; attach it to the root coordinator \
                 with PackfileStorage::enable_shared_journal instead of a per-pool journal",
                self.base_dir.display(),
                root.display()
            ))),
            None => Ok(()),
        }
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
    #[cfg(feature = "multi-reader")]
    pub fn enable_journal_with_sequence(
        &self,
        path: impl AsRef<std::path::Path>,
        sequence: Arc<AtomicU64>,
    ) -> Result<(), StorageError> {
        self.reject_per_pool_journal_in_root()?;
        let (journal, scan) = Journal::open(path).map_err(StorageError::Io)?;
        self.journal_recovery.lock().clone_from(&scan.groups);
        let coordinator = Arc::new(JournalCoordinator::with_shared_sequence(
            journal, &scan, sequence,
        ));
        coordinator
            .enable_publish_signal()
            .map_err(StorageError::Io)?;
        *self.journal.lock() = Some(coordinator);
        Ok(())
    }

    /// Attach this store to a shared, multi-pool journal coordinator.
    ///
    /// Every mutation this store publishes is tagged with `pool`, so recovery
    /// routes it back to this pool. Because all pools share one coordinator
    /// (and therefore one segment), any one pool's `sync`/`sync_all` fences
    /// every pool's pending mutations in a single group and a single fsync —
    /// instead of each pool paying its own WAL fsync.
    ///
    /// The coordinator must be built from a pool-tagged segment (see
    /// [`Journal::open_shared`]) and the caller must hold the root writer lock
    /// ([`crate::journal::SharedWalLock`]). This is opt-in; a store that never
    /// calls it keeps the legacy per-pool journal path unchanged.
    ///
    /// The store's existing durable coverage (the `journal.lsn` written beside
    /// its last checkpoint) is reported to the coordinator on attach. Without
    /// it, a pool whose frames were reclaimed before a restart has no coverage
    /// in the freshly recovered segment, and reclaim could advance past frames
    /// this store has not materialized (or block on a pool that is in fact
    /// already covered).
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] if a journal is already enabled, or
    /// [`StorageError::Io`] if publish-signal setup fails, including OS entropy
    /// failures.
    #[cfg(feature = "multi-reader")]
    pub fn enable_shared_journal(
        &self,
        journal: Arc<JournalCoordinator>,
        pool: crate::layout::ShardType,
    ) -> Result<(), StorageError> {
        let mut slot = self.journal.lock();
        if slot.is_some() {
            return Err(StorageError::Internal(
                "a journal is already enabled on this store".into(),
            ));
        }
        self.journal_recovery
            .lock()
            .clone_from(&journal.recovered_groups());
        self.journal_pool.store(pool_tag(pool), Ordering::Release);
        journal.report_pool_coverage(pool, self.durable_coverage());
        journal.enable_publish_signal().map_err(StorageError::Io)?;
        *slot = Some(journal);
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
        let covered = self.durable_coverage();
        let groups = self.journal_recovery.lock().clone();
        self.replaying.store(true, Ordering::SeqCst);
        let result = (|| -> Result<u64, StorageError> {
            let mut replayed = 0u64;
            // On a shared journal, replay only this pool's frames; the other
            // pools' mutations are replayed by their own stores.
            let pool = pool_from_tag(self.journal_pool.load(Ordering::Acquire));
            for group in &groups {
                for entry in &group.entries {
                    if entry.lsn <= covered {
                        continue;
                    }
                    if pool.is_some_and(|pool| entry.pool != Some(pool)) {
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

    /// Start a bounded background WAL group committer on this store's journal.
    ///
    /// Additive and opt-in: without it the store keeps the historical blocking
    /// [`StorageEngine::sync`] behavior. The committer fsyncs the pending WAL
    /// group at most once per [`GroupCommitConfig::interval`], coalescing every
    /// mutation published in that window into one group, and flushes early once
    /// [`GroupCommitConfig::max_pending`] records are outstanding. It commits
    /// the journal only; packfile checkpointing and segment reclaim remain the
    /// explicit `sync`/`sync_all` path. No-op when no journal is enabled.
    ///
    /// # Errors
    /// Returns a storage error if the committer thread cannot be spawned.
    pub fn start_background_commit(&self, config: GroupCommitConfig) -> Result<(), StorageError> {
        match self.journal() {
            Some(journal) => journal
                .start_background_committer(config)
                .map_err(StorageError::Io),
            None => Ok(()),
        }
    }

    /// Stop the background committer (if any), join it, and flush the final
    /// pending WAL group so a quiet stream's last write is durable before this
    /// returns. No-op when no journal is enabled.
    ///
    /// # Errors
    /// Returns a storage error if the final flush fails.
    pub fn stop_background_commit(&self) -> Result<(), StorageError> {
        match self.journal() {
            Some(journal) => journal
                .stop_background_committer()
                .map_err(StorageError::Io),
            None => Ok(()),
        }
    }

    /// Register a non-blocking durability request through `target_lsn`,
    /// returning `None` when no journal is enabled.
    #[must_use]
    pub fn request_durable(&self, target_lsn: u64) -> Option<DurabilityToken> {
        self.journal()
            .map(|journal| journal.request_durable(target_lsn))
    }

    /// Block until the group covering `token` is durable. No-op when no journal
    /// is enabled. Pair with [`Self::request_durable`].
    ///
    /// # Errors
    /// Returns a storage error if the target was never published, the journal
    /// is poisoned, the background committer failed, or the commit fails.
    pub fn wait_durable(&self, token: DurabilityToken) -> Result<(), StorageError> {
        match self.journal() {
            Some(journal) => journal
                .wait_durable(token)
                .map(|_| ())
                .map_err(StorageError::Io),
            None => Ok(()),
        }
    }

    /// Perform one bounded WAL group commit over everything published so far.
    /// No-op when no journal is enabled.
    ///
    /// # Errors
    /// Returns a storage error if appending or fsyncing the group fails.
    pub fn flush_durable(&self) -> Result<(), StorageError> {
        match self.journal() {
            Some(journal) => journal
                .flush_durable()
                .map(|_| ())
                .map_err(StorageError::Io),
            None => Ok(()),
        }
    }

    /// Whether ordinary reads must go through the transaction overlay: a
    /// transaction is using it, and this read is not the overlay path's own
    /// fallback to the live index.
    fn overlay_reads_active(&self) -> bool {
        self.transaction_overlay_users.load(Ordering::Acquire) != 0
            && !READ_COMMITTED_FALLBACK.with(Cell::get)
    }

    /// Enable the in-process read overlay for a published transaction that is
    /// still being materialized. The overlay is shared with the existing
    /// read-committed implementation, but ordinary reads consult it only
    /// while at least one transaction is in this state.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn activate_transaction_overlay(
        &self,
        wal_path: &Path,
        pool: crate::layout::ShardType,
    ) -> Result<(), StorageError> {
        let _lifecycle = self.transaction_overlay_lifecycle.lock();
        let previous = self
            .transaction_overlay_users
            .fetch_add(1, Ordering::AcqRel);
        if previous == 0 {
            if let Err(error) = self.enable_transaction_read_journal(wal_path, pool) {
                self.transaction_overlay_users
                    .fetch_sub(1, Ordering::AcqRel);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Whether a journal overlay is installed on the read path.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn read_journal_installed(&self) -> bool {
        self.read_journal.lock().is_some()
    }

    /// Whether the read-journal overlay mapped the cross-process publish
    /// signal. `false` means reads remain correct but use the stat-based
    /// refresh path, either because no reader overlay is installed or the
    /// signal sidecar was unavailable when the overlay opened.
    #[must_use]
    pub fn read_journal_publish_signal_active(&self) -> bool {
        self.read_journal
            .lock()
            .as_ref()
            .is_some_and(ReadJournal::publish_signal_active)
    }

    /// Test-only: refreshes on this store's overlay that reached
    /// `fs::metadata`, i.e. were not skipped by the publish-signal gate.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn read_journal_stat_checks(&self) -> u64 {
        self.read_journal
            .lock()
            .as_ref()
            .map_or(0, |overlay| overlay.stat_checks)
    }

    /// Total segment bytes the retained transaction overlay has scanned.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn transaction_overlay_scanned_bytes(&self) -> u64 {
        self.transaction_overlay_scanned.load(Ordering::Relaxed)
    }

    /// Stop consulting the in-process transaction overlay after all staged
    /// mutations have been materialized or the transaction was abandoned
    /// before journal publication.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn deactivate_transaction_overlay(&self) {
        let _lifecycle = self.transaction_overlay_lifecycle.lock();
        let mut users = self.transaction_overlay_users.load(Ordering::Acquire);
        loop {
            let next = users
                .checked_sub(1)
                .expect("transaction overlay deactivated too often");
            match self.transaction_overlay_users.compare_exchange_weak(
                users,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => users = current,
            }
        }
        if users == 1 {
            self.park_transaction_overlay();
        }
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
        if JOURNAL_SUPPRESSED.with(Cell::get) {
            return Ok(None);
        }
        let Some(journal) = self.journal() else {
            self.record_published_mutation(started);
            return Ok(None);
        };
        // Replayed mutations are already durable in the journal; never
        // re-publish them.
        if self.replaying.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // The live index is already updated synchronously on the write path.
        // Append this autocommit mutation as its own complete, visible group;
        // durability remains separate so the background committer can batch
        // several such groups into one fsync. On a shared journal, tag the
        // frame with this store's pool so recovery can route it.
        #[cfg(feature = "multi-reader")]
        let result = match pool_from_tag(self.journal_pool.load(Ordering::Acquire)) {
            Some(pool) => journal
                .publish_group_tagged(pool, &[mutation()])
                .map(|receipt| receipt.last_lsn),
            None => journal
                .publish_group(&[mutation()])
                .map(|receipt| receipt.last_lsn),
        }
        .map(Some)
        .map_err(StorageError::Io);
        #[cfg(not(feature = "multi-reader"))]
        let result = journal
            .publish_group(&[mutation()])
            .map(|receipt| receipt.last_lsn)
            .map(Some)
            .map_err(StorageError::Io);
        if result.is_ok() {
            self.record_published_mutation(started);
        }
        result
    }

    /// Apply a mutation staged by a database transaction without publishing
    /// it to the legacy per-write journal queue. The transaction coordinator
    /// publishes the complete batch after every pool has applied successfully.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn apply_transaction_mutation(
        &self,
        mutation: &JournalMutation,
    ) -> Result<(), StorageError> {
        JOURNAL_SUPPRESSED.with(|suppressed| {
            let previous = suppressed.replace(true);
            let result = match mutation {
                JournalMutation::Put {
                    collection_id,
                    node_id,
                    payload,
                } => self.put(
                    collection_id,
                    node_id,
                    &NodeData::new(bytes::Bytes::from(payload.clone())),
                ),
                JournalMutation::DeleteCollection { collection_id } => {
                    self.delete_collection(collection_id)
                }
            };
            suppressed.set(previous);
            result
        })
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
            timings.journal_fsync = journal_timings.journal_fsync;
            timings.journal_sync_calls = 1;
            timings.journal_records = journal_timings.journal_records;
            timings.journal_waiters = u64::from(journal_timings.journal_waiter);
            timings.journal_coalesced = u64::from(journal_timings.journal_coalesced);
            timings.journal_in_flight = journal_timings.journal_in_flight;
            // The WAL is durable, so the packs are only owed for the checkpoint.
            // Pay that in slices, outside every collection lock, once enough has
            // accumulated, instead of all at once at the checkpoint.
            let budget = self.pack_fsync_budget_bytes.load(Ordering::Relaxed);
            if budget != 0 && self.shards.unsynced_bytes() >= budget {
                self.shards.sync_dirty()?;
                if let Some((_, fsync)) = self.shards.last_sync_split() {
                    timings.pack_fsync = fsync;
                }
            }
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
            journal_fsync: timings.journal_fsync,
            journal_records: timings.journal_records,
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
        state.base_fingerprint.is_none() || state.log_version != 3 || state.pending.is_empty()
    }

    /// Tell the journal this pool's packs durably cover its frames through
    /// `lsn`, and reclaim what that lets go of. Best-effort: a failed
    /// compaction costs disk, never correctness (the coverage is already
    /// durable).
    fn report_coverage_and_reclaim(&self, lsn: u64) -> Option<crate::journal::Reclaim> {
        let journal = self.journal()?;
        report_coverage_and_reclaim_via(&journal, self.journal_pool.load(Ordering::Acquire), lsn)
    }

    /// After a sync (no lock held): if a shared-WAL reclaim is stalled in the
    /// emergency zone, make the pools holding it back checkpoint. Which pools,
    /// and how, is decided by the coordinator and whoever owns the pools.
    fn remediate_lagging_pools(&self) {
        #[cfg(feature = "multi-reader")]
        if let Some(journal) = self.journal() {
            journal.remediate_blockers(pool_from_tag(self.journal_pool.load(Ordering::Acquire)));
        }
        #[cfg(not(feature = "multi-reader"))]
        let _ = self;
    }

    /// Whether this sync must take a full checkpoint so the journal can be
    /// reclaimed: the segment is large, and this pool has journal frames its
    /// last checkpoint does not yet cover. A pool with nothing new to cover
    /// gains nothing from a rewrite and never holds the segment back, since its
    /// earlier coverage is already reported.
    fn journal_needs_reclaim_checkpoint(&self) -> bool {
        let Some(journal) = self.journal() else {
            return false;
        };
        #[cfg(feature = "multi-reader")]
        let this_pool = pool_from_tag(self.journal_pool.load(Ordering::Acquire));
        #[cfg(not(feature = "multi-reader"))]
        let this_pool = None;
        if !journal.should_force_reclaim_checkpoint(this_pool) {
            return false;
        }
        #[cfg(feature = "multi-reader")]
        if let Some(pool) = pool_from_tag(self.journal_pool.load(Ordering::Acquire)) {
            return journal.committed_lsn_for_pool(pool) > self.durable_coverage();
        }
        true
    }

    /// Whether a delta log continues the checkpoint this session is based on, so
    /// a batch can be appended to it (pending operations or not).
    fn delta_base_is_usable(&self) -> bool {
        let state = self.delta_state.lock();
        state.base_fingerprint.is_some() && state.log_version == 3
    }

    /// The delta-log size limit: the real cap, or a test's smaller one.
    fn delta_log_cap(&self) -> u64 {
        #[cfg(test)]
        {
            let limit = self.delta_log_cap_override.load(Ordering::Relaxed);
            if limit != 0 {
                return limit;
            }
        }
        let _ = self;
        DELTA_LOG_CAP_BYTES
    }

    /// Log length at which a sync rewrites the checkpoint instead of appending:
    /// three quarters of [`Self::delta_log_cap`].
    fn delta_log_rotate_bytes(&self) -> u64 {
        self.delta_log_cap().saturating_div(4).saturating_mul(3)
    }

    /// The details of `error` if it is a delta batch that does not fit under the
    /// cap. That is a planned rotation, not a failure: the batch may carry
    /// whole-index snapshots, so it is refused by its projected length.
    fn delta_batch_too_large(error: &StorageError) -> Option<&DeltaBatchTooLarge> {
        match error {
            StorageError::Io(io) => io
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<DeltaBatchTooLarge>()),
            _ => None,
        }
    }

    /// Whether every live pack is one the base checkpoint's pack table names,
    /// so every slot a delta operation of this epoch carries resolves through
    /// that table for any reader. A pack created since forces a full
    /// checkpoint, which records the new table.
    fn packs_match_checkpoint_table(&self) -> bool {
        let state = self.delta_state.lock();
        self.shards
            .all_shards()
            .iter()
            .all(|(_, shard)| state.base_pack_ids.contains(&shard.pack_id))
    }

    /// Persist the dirty index state for a sync barrier — a delta append when
    /// the log can be continued, otherwise a full checkpoint rewrite — and
    /// record which path ran in `timings`. No-op when nothing is dirty.
    ///
    /// The shard→collection inspection sidecar is deferred away from the
    /// per-event durable barrier: a delta append and a deferred checkpoint
    /// rewrite only set `shard_collections_stale`, leaving the on-disk copy
    /// (and its pack fingerprint) behind the live pack set. That is safe
    /// because the sidecar is rebuildable acceleration metadata — the next open
    /// fails its fingerprint gate and falls back to the slot walk — so the hot
    /// barrier never pays its `sync_data` + rename. The sidecar is re-pinned
    /// only at anchors: a successful full checkpoint rewrite (to the same pack
    /// fingerprint the checkpoint just became), a repack, or shutdown (`Drop`).
    /// A known missing/invalid sidecar (`shard_collections_dirty`) is still
    /// regenerated even on an otherwise clean barrier, because that is a rare
    /// recovery path rather than a steady-state write.
    #[allow(clippy::too_many_lines)]
    fn persist_index_checkpoint_or_delta(&self, timings: &mut SyncTimings) {
        let _persist_guard = self.index_persist_lock.lock();
        if !self.settle_checkpoint_worker() {
            return;
        }
        // A delta append does not advance the journal coverage, so on its own it
        // never lets the segment shrink and the journal would fill to its hard
        // limit. Once the segment is large, a pool with journal frames its
        // checkpoint does not yet cover takes the full checkpoint (which records
        // coverage and reclaims), even if its index is clean; a shared segment is
        // only reclaimed once every such pool has reported.
        let journal_needs_reclaim = self.journal_needs_reclaim_checkpoint();
        let new_pack_requires_table_refresh = !self.packs_match_checkpoint_table();
        if new_pack_requires_table_refresh {
            self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }
        if journal_needs_reclaim {
            self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            // Nothing to checkpoint, but a sidecar lost while the checkpoint
            // survived still has to be regenerated. A sidecar that is merely
            // stale from deferred writes is left for the next anchor.
            if self.shard_collections_dirty.load(Ordering::Relaxed) {
                let sidecar_started = std::time::Instant::now();
                self.persist_shard_collections_best_effort();
                timings.sidecar = sidecar_started.elapsed();
            }
            return;
        }
        // Only a full checkpoint rewrite re-pins the sidecar to the pack
        // fingerprint the checkpoint just became. Every other dirty path
        // records staleness instead and writes nothing.
        let mut sidecar_anchor = false;
        // When the journal must be reclaimed and a delta log can continue the
        // checkpoint, record the coverage in a delta batch instead of rewriting
        // the whole index: the packs are made durable, then the batch carrying
        // the claim is, and only then is the journal reclaimed.
        let mut coverage_batch_written = false;
        let log_bytes = self.delta_state.lock().log_bytes;
        let rotating = log_bytes >= self.delta_log_rotate_bytes();
        if rotating {
            eprintln!(
                "mtxdb: delta log is at {log_bytes} of {} bytes, rewriting the checkpoint \
                 before it reaches the cap",
                self.delta_log_cap()
            );
        }
        if journal_needs_reclaim
            && !rotating
            && self.delta_base_is_usable()
            && self.packs_match_checkpoint_table()
        {
            let delta_started = std::time::Instant::now();
            match self.append_index_delta_v3(true) {
                Ok(Some(covered)) => {
                    timings.delta_log = delta_started.elapsed();
                    let reclaim_started = std::time::Instant::now();
                    let reclaim = self.report_coverage_and_reclaim(covered);
                    timings.reclaim = reclaim_started.elapsed();
                    timings.record_reclaim_phases(reclaim.as_ref());
                    coverage_batch_written = true;
                }
                Ok(None) => {}
                Err(error) if Self::delta_batch_too_large(&error).is_some() => {
                    if let Some(detail) = Self::delta_batch_too_large(&error) {
                        eprintln!(
                            "mtxdb: rotating the delta log: {detail}; rewriting the checkpoint"
                        );
                    }
                }
                Err(error) => {
                    eprintln!("mtxdb: delta coverage batch failed, rewriting checkpoint: {error}");
                }
            }
        }
        if coverage_batch_written {
            // Like any delta append: the sidecar is merely stale, unless it is
            // known missing.
            if self.shard_collections_dirty.load(Ordering::Relaxed) {
                sidecar_anchor = true;
            } else {
                self.shard_collections_stale.store(true, Ordering::Relaxed);
            }
        } else if journal_needs_reclaim
            || rotating
            || self.delta_state_needs_full_rewrite()
            || new_pack_requires_table_refresh
        {
            // A deferral budget postpones acceleration rewrites; it must not
            // postpone the one that lets the journal be reclaimed, or the segment
            // fills and commits fail, nor the one that keeps the delta log under
            // its cap.
            if !journal_needs_reclaim
                && !rotating
                && !new_pack_requires_table_refresh
                && self.should_defer_checkpoint_rewrite()
            {
                // Write-neutral stopgap: the caller already synced the
                // packfiles, so skipping the acceleration rewrite costs only
                // the next open a rescan — the stale on-disk checkpoint no
                // longer matches the advanced pack fingerprint, and the delta
                // log's tail does not either, so the opener rejects both. Leave
                // `index_checkpoint_dirty` set so a later barrier past the
                // budget still rewrites.
                self.checkpoint_skips.fetch_add(1, Ordering::Relaxed);
                self.shard_collections_stale.store(true, Ordering::Relaxed);
                return;
            }
            let checkpoint_started = std::time::Instant::now();
            self.persist_index_checkpoint_for_sync();
            self.note_checkpoint_rewrite();
            timings.checkpoint = checkpoint_started.elapsed();
            sidecar_anchor = true;
        } else {
            let delta_started = std::time::Instant::now();
            if let Err(error) = self.append_index_delta() {
                // An append failure took the frames with it, so the pending state
                // no longer reflects the live indexes. Fall back to a full
                // rewrite rather than leaving the acceleration files stale until
                // the next sync notices the gap.
                if let Some(detail) = Self::delta_batch_too_large(&error) {
                    eprintln!("mtxdb: rotating the delta log: {detail}; rewriting the checkpoint");
                } else {
                    eprintln!("mtxdb: delta log append failed, rewriting checkpoint: {error}");
                }
                let checkpoint_started = std::time::Instant::now();
                self.persist_index_checkpoint_for_sync();
                self.note_checkpoint_rewrite();
                timings.checkpoint = checkpoint_started.elapsed();
                sidecar_anchor = true;
            } else {
                timings.delta_log = delta_started.elapsed();
                // A missing/invalid sidecar must still be regenerated, but a
                // normal delta append defers it: the on-disk copy is merely
                // stale and the next open rebuilds it from the slot walk.
                if self.shard_collections_dirty.load(Ordering::Relaxed) {
                    sidecar_anchor = true;
                } else {
                    self.shard_collections_stale.store(true, Ordering::Relaxed);
                }
            }
        }
        if sidecar_anchor {
            let sidecar_started = std::time::Instant::now();
            self.persist_shard_collections_best_effort();
            timings.sidecar = sidecar_started.elapsed();
        }
    }

    /// Fingerprint of the current on-disk pack set, computed from each
    /// shard's committed length. Must be called after the flush that made the
    /// bytes corresponding to outstanding index frames durable, so the file
    /// lengths reflect exactly what the indexes' offsets point at.
    fn current_pack_fingerprint(&self) -> u64 {
        let packs: Vec<(PackId, u64)> = self
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

    /// Phase breakdown of the most recent full index checkpoint, if one ran.
    #[must_use]
    pub fn checkpoint_breakdown(&self) -> Option<CheckpointBreakdown> {
        *self.last_checkpoint_breakdown.lock()
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
    /// # let dir = std::env::temp_dir().join(format!("mtxdb-doc-with-append-policy-{}", std::process::id()));
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
    /// operations; logical read counters are monotone, so reads disabled by a
    /// false-to-true transition are reflected as fewer `get_*` calls, not as
    /// zeros.
    /// When enabled, it also records wall-clock totals and fixed latency
    /// buckets plus maxima for `get`, `get_many`, `get_many_with_refresh`,
    /// `put`, and `put_many` in [`Self::stats`].
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
            read_plan_extents: self.read_plan_extents.load(Ordering::Relaxed),
            read_plan_prefetch_bytes: self.read_plan_prefetch_bytes.load(Ordering::Relaxed),
            read_plan_skipped_extents: self.read_plan_skipped_extents.load(Ordering::Relaxed),
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
            checkpoint_tails_started: self.checkpoint_tails_started.load(Ordering::Relaxed),
            syncs_with_tail_in_flight: self.syncs_with_tail_in_flight.load(Ordering::Relaxed),
            syncs_waited_for_tail: self.syncs_waited_for_tail.load(Ordering::Relaxed),
            checkpoint_skips: self.checkpoint_skips.load(Ordering::Relaxed),
            delta_appends: self.delta_appends.load(Ordering::Relaxed),
            read_reloads: self.read_reloads.load(Ordering::Relaxed),
            read_reload_failures: self.read_reload_failures.load(Ordering::Relaxed),
            read_refreshes: self.read_refreshes.load(Ordering::Relaxed),
            read_refresh_bytes: self.read_refresh_bytes.load(Ordering::Relaxed),
            sidecar_writes: self.sidecar_writes.load(Ordering::Relaxed),
            sync_calls: self.sync_calls.load(Ordering::Relaxed),
            get_latency: self.operation_timings.get.snapshot(),
            get_many_latency: self.operation_timings.get_many.snapshot(),
            get_many_with_refresh_latency: self.operation_timings.get_many_with_refresh.snapshot(),
            put_latency: self.operation_timings.put.snapshot(),
            put_many_latency: self.operation_timings.put_many.snapshot(),
            last_open_timings: self.open_timings(),
            last_sync_timings: self.sync_timings(),
            sync_totals: self.sync_totals.snapshot(),
            sync_diagnostics: self.sync_diagnostics.lock().snapshot(),
            publish_calls: self.publish_calls.load(Ordering::Relaxed),
            publish_time: std::time::Duration::from_nanos(
                self.publish_time_ns.load(Ordering::Relaxed),
            ),
            background_commits: self
                .journal()
                .map_or(0, |journal| journal.background_commits()),
            background_coalesced: self
                .journal()
                .map_or(0, |journal| journal.background_coalesced()),
            durability: self
                .journal()
                .map_or_else(Default::default, |journal| journal.durability_stats()),
            repack: self.repack_stats(),
            cache,
            shards: self.shard_stats(),
            index_bytes,
            collection_count: summaries.len(),
            max_index_probe_len,
            dirty_lock_wait: self.shards.dirty_lock_wait(),
        }
    }

    /// Atomically take the current sync diagnostics and reset only those
    /// diagnostics for the next observation interval.
    ///
    /// Unlike [`Self::reset_stats`], this leaves all runtime counters and
    /// cumulative sync totals untouched. The returned histograms, peak
    /// in-flight count, interval maxima (`max_journal_lock_wait` and
    /// `max_journal_fsync`), and worst-operation samples therefore describe
    /// the interval ending at this call; the interval maxima reset to zero
    /// alongside the histograms. Syncs racing with the call are recorded in
    /// either the returned snapshot or the next interval, never partially in
    /// both.
    #[must_use]
    pub fn take_sync_diagnostics(&self) -> SyncDiagnosticsSnapshot {
        let mut diagnostics = self.sync_diagnostics.lock();
        let snapshot = diagnostics.snapshot();
        diagnostics.reset();
        snapshot
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
            &self.read_plan_extents,
            &self.read_plan_prefetch_bytes,
            &self.read_plan_skipped_extents,
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
            &self.read_refreshes,
            &self.read_refresh_bytes,
            &self.sidecar_writes,
            &self.sync_calls,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.operation_timings.reset();
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
    /// Merged read extents prefetched with `madvise(MADV_WILLNEED)` across
    /// `get_many` batches (the executed physical plan, as opposed to the
    /// [`Self::read_many_runs`] logical shape).
    pub read_plan_extents: u64,
    /// Bytes covered by prefetched read extents (sum of extent lengths; not
    /// physical disk bytes, since `madvise` is a hint).
    pub read_plan_prefetch_bytes: u64,
    /// Planned extents not prefetched (offset beyond the current mapping,
    /// missing mapping, or a failed `madvise`).
    pub read_plan_skipped_extents: u64,
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
    /// Syncs that rewrote the index checkpoint in full, plus forced ones
    /// (`force_index_checkpoint`, e.g. WAL-reclaim remediation on the BG=0
    /// path) that never pass through `sync`/`sync_all`.
    pub checkpoint_writes: u64,
    /// Background checkpoint tails started (one per detached checkpoint).
    pub checkpoint_tails_started: u64,
    /// Syncs that ran with a tail in flight and appended to it instead of
    /// starting another checkpoint.
    pub syncs_with_tail_in_flight: u64,
    /// Syncs that waited for an in-flight tail (WAL emergency zone or delta-log
    /// rotation), paying its duration on the foreground.
    pub syncs_waited_for_tail: u64,
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
    /// Read-journal refresh attempts by read-only workers, including no-op
    /// refreshes.
    pub read_refreshes: u64,
    /// Journal bytes scanned by read-only worker refreshes.
    pub read_refresh_bytes: u64,
    /// Writes of the shard→collection inspection sidecar (every one counts,
    /// whichever caller triggered it).
    pub sidecar_writes: u64,
    /// `sync`/`sync_all` calls.
    pub sync_calls: u64,
    /// Opt-in wall-clock latency for single-record reads.
    pub get_latency: OperationLatency,
    /// Opt-in wall-clock latency for batched reads.
    pub get_many_latency: OperationLatency,
    /// Opt-in inclusive wall-clock latency for refresh-aware batched reads,
    /// including any inner `get_many`, stale-index refresh, and retry work.
    /// This overlaps [`Self::get_many_latency`] when refresh is enabled and
    /// must not be added to it.
    pub get_many_with_refresh_latency: OperationLatency,
    /// Opt-in wall-clock latency for single-record writes.
    pub put_latency: OperationLatency,
    /// Opt-in wall-clock latency for batched writes.
    pub put_many_latency: OperationLatency,
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
    /// Background WAL group commits that appended and fsynced a group (see
    /// [`PackfileStorage::start_background_commit`]). Zero without a journal
    /// or when the committer was never started.
    pub background_commits: u64,
    /// Background commit attempts already covered by a durable group.
    pub background_coalesced: u64,
    /// Lifetime journal durability accounting (requests versus real fsyncs,
    /// records per fsync, blocked-wait latency). Default without a journal.
    pub durability: crate::journal::DurabilityStats,
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

/// Flush a dirty/stale shard→collection sidecar on shutdown so a clean exit
/// leaves the sidecar pinned to the last *durable* pack set.
///
/// This covers state that a completed `sync`/`sync_all` left deferred (a delta
/// append or a budget-deferred checkpoint rewrite set `shard_collections_stale`)
/// plus a sidecar lost at open (`shard_collections_dirty`). It deliberately
/// does **not** flush bookkeeping changed by writes that were never synced:
/// such a write grows a pack past the last checkpoint, so the next open cannot
/// take the checkpoint fast path at all and full-scans instead — it never
/// consults the sidecar. Writing one there would be unownable I/O, so the
/// narrower contract is the correct one.
///
/// Best-effort and gated: a read-only handle never sets either flag, and any
/// write error is swallowed by the best-effort wrapper. Even skipping the flush
/// entirely is safe — the sidecar is rebuildable acceleration metadata — so
/// correctness never depends on `Drop`.
/// [`PackfileStorage::write_journal_lsn`] on owned handles, for the tail.
fn write_journal_lsn_file(
    base_dir: &Path,
    durable_coverage: &AtomicU64,
    lsn: u64,
) -> Result<(), StorageError> {
    let path = PackfileStorage::journal_lsn_path(base_dir);
    let tmp = path.with_extension("lsn.tmp");
    fs::write(&tmp, lsn.to_le_bytes()).map_err(StorageError::Io)?;
    // Windows requires a write-capable handle for FlushFileBuffers,
    // which is what `sync_all` uses. A read-only handle works on Unix
    // but fails with ERROR_ACCESS_DENIED on Windows.
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&tmp)
        .and_then(|file| file.sync_all())
        .map_err(StorageError::Io)?;
    fs::rename(&tmp, &path).map_err(StorageError::Io)?;
    let _ = crate::shard::sync_directory(base_dir);
    durable_coverage.fetch_max(lsn, Ordering::AcqRel);
    Ok(())
}

/// Remove the superseded delta-log epoch once the checkpoint that makes it safe
/// to discard is durably renamed. Best-effort: a leftover file names a
/// fingerprint no future checkpoint carries, so it is inert.
fn retire_delta_epoch_file(base_dir: &Path, old_base_fingerprint: Option<u64>) {
    let Some(old_base_fingerprint) = old_base_fingerprint else {
        return;
    };
    let path = PackfileStorage::delta_path(base_dir, old_base_fingerprint);
    if fs::remove_file(&path).is_ok() {
        // The checkpoint rename's own durability is already covered by the
        // directory fsync after `write_checkpoint`; this one covers the unlink,
        // so the retired file does not linger past a crash.
        let _ = crate::shard::sync_directory(base_dir);
    }
}

/// Tell the journal a pool's packs durably cover its frames through `lsn`, and
/// reclaim what that lets go of. `tag` is the pool's journal tag.
/// Report this pool's durable coverage and reclaim the journal, returning the
/// reclaim's phase breakdown when one ran (for the sync-path timings).
fn report_coverage_and_reclaim_via(
    journal: &JournalCoordinator,
    tag: u8,
    lsn: u64,
) -> Option<crate::journal::Reclaim> {
    match pool_from_tag(tag) {
        // A per-pool segment holds only this pool's frames, so this coverage
        // means the segment can drop everything at or below `lsn`.
        None => match journal.reclaim_through(lsn) {
            Ok(reclaim) => Some(reclaim),
            Err(error) => {
                eprintln!("warning: journal reclaim through LSN {lsn} failed: {error}");
                None
            }
        },
        // A shared segment also holds other pools' frames. Record this pool's
        // coverage and reclaim only what every contributing pool has covered.
        #[cfg(feature = "multi-reader")]
        Some(pool) => {
            journal.report_pool_coverage(pool, lsn);
            match journal.reclaim_shared() {
                Ok(reclaim) => reclaim,
                Err(error) => {
                    eprintln!("warning: shared journal reclaim failed: {error}");
                    None
                }
            }
        }
        #[cfg(not(feature = "multi-reader"))]
        Some(_) => unreachable!("shared journal state requires multi-reader"),
    }
}

/// Everything a checkpoint needs after the handoff, owned so it can run on any
/// thread. Built under the locks by [`PackfileStorage::capture_checkpoint`]:
/// the image is the captured bytes, never the live table, and the delta epoch
/// has already been rotated. Running it writes and fsyncs the image, installs
/// it, records the coverage it carries, reclaims the WAL and retires the old
/// epoch, in that order; the WAL is not touched until the image is durable.
struct CheckpointTail {
    base_dir: PathBuf,
    fingerprint: u64,
    covered_lsn: Option<u64>,
    base_delta_seq: u64,
    blobs: Vec<([u8; 16], u64, Vec<u8>)>,
    pack_table: Vec<(u16, PackId)>,
    old_base_fingerprint: Option<u64>,
    journal: Option<Arc<JournalCoordinator>>,
    pool_tag: u8,
    durable_coverage: Arc<AtomicU64>,
    #[cfg(test)]
    hook: Option<TailHook>,
}

/// Test-only control over a checkpoint tail: it waits for `gate` to be
/// released (or dropped), then fails if `fail` is set.
#[cfg(test)]
struct TailHook {
    gate: std::sync::mpsc::Receiver<()>,
    fail: bool,
}

impl CheckpointTail {
    /// Run the tail; the returned breakdown holds only the phases it ran.
    fn run(self) -> Result<CheckpointBreakdown, StorageError> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            let _ = hook.gate.recv();
            if hook.fail {
                return Err(StorageError::Io(std::io::Error::other(
                    "injected checkpoint tail failure",
                )));
            }
        }
        let tail_started = std::time::Instant::now();
        let mut done = CheckpointBreakdown::default();
        let path = PackfileStorage::index_checkpoint_path(&self.base_dir);
        let write_started = std::time::Instant::now();
        let blobs: Vec<([u8; 16], u64, &[u8])> = self
            .blobs
            .iter()
            .map(|(collection_id, generation, blob)| (*collection_id, *generation, blob.as_slice()))
            .collect();
        crate::index::checkpoint::write_checkpoint(
            &path,
            self.fingerprint,
            self.covered_lsn.unwrap_or(0),
            self.base_delta_seq,
            &blobs,
            &self.pack_table,
        )
        .map_err(StorageError::Io)?;
        done.write = write_started.elapsed();
        done.checkpoint_bytes = fs::metadata(&path).map_or(0, |meta| meta.len());
        // The rename is only durable once the directory is synced. Best-effort:
        // a crash before it either observes the rename or falls back to the
        // still-valid predecessor checkpoint, never a torn one.
        let directory_started = std::time::Instant::now();
        let _ = crate::shard::sync_directory(&self.base_dir);
        done.directory_sync = directory_started.elapsed();
        // The image is durable; record the LSN it covers so a reopen replays only
        // what follows, and let the journal drop what the packs now cover. The
        // pack fsync happened before the handoff, so everything through the LSN
        // is durable in the packs. A failed compaction costs disk, not
        // correctness.
        if let (Some(journal), Some(lsn)) = (&self.journal, self.covered_lsn) {
            let lsn_started = std::time::Instant::now();
            write_journal_lsn_file(&self.base_dir, &self.durable_coverage, lsn)?;
            done.journal_lsn = lsn_started.elapsed();
            let reclaim_started = std::time::Instant::now();
            let _ = report_coverage_and_reclaim_via(journal, self.pool_tag, lsn);
            done.reclaim = reclaim_started.elapsed();
        }
        let retire_started = std::time::Instant::now();
        retire_delta_epoch_file(&self.base_dir, self.old_base_fingerprint);
        done.retire = retire_started.elapsed();
        done.total = tail_started.elapsed();
        Ok(done)
    }
}

/// A checkpoint captured under the locks, ready for its tail.
struct CapturedCheckpoint {
    tail: CheckpointTail,
    breakdown: CheckpointBreakdown,
}

/// A tail running on its own thread, with the capture's phase timings.
struct CheckpointWorker {
    handle: std::thread::JoinHandle<Result<CheckpointBreakdown, StorageError>>,
    breakdown: CheckpointBreakdown,
}

impl Drop for PackfileStorage {
    fn drop(&mut self) {
        self.finish_checkpoint_worker(true);
        if self.shard_collections_dirty.load(Ordering::Relaxed)
            || self.shard_collections_stale.load(Ordering::Relaxed)
        {
            self.persist_shard_collections_best_effort();
        }
    }
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
            read_plan_extents: 0,
            read_plan_prefetch_bytes: 0,
            read_plan_skipped_extents: 0,
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
            checkpoint_tails_started: 0,
            syncs_with_tail_in_flight: 0,
            syncs_waited_for_tail: 0,
            checkpoint_skips: 0,
            delta_appends: 0,
            read_reloads: 0,
            read_reload_failures: 0,
            read_refreshes: 0,
            read_refresh_bytes: 0,
            sidecar_writes: 0,
            sync_calls: 0,
            get_latency: OperationLatency::default(),
            get_many_latency: OperationLatency::default(),
            get_many_with_refresh_latency: OperationLatency::default(),
            put_latency: OperationLatency::default(),
            put_many_latency: OperationLatency::default(),
            last_open_timings: None,
            last_sync_timings: None,
            sync_totals: SyncTotalsSnapshot::default(),
            sync_diagnostics: SyncDiagnosticsSnapshot::default(),
            publish_calls: 0,
            publish_time: std::time::Duration::ZERO,
            background_commits: 0,
            background_coalesced: 0,
            durability: crate::journal::DurabilityStats::default(),
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

    /// Installs a `before_pin` hook that is removed when the guard drops,
    /// including on panic, so a failing test cannot leak the hook into later
    /// tests on the same thread.
    fn install_test_before_pin(hook: Box<dyn Fn()>) -> TestBeforePinGuard {
        Self::set_test_before_pin(Some(hook));
        TestBeforePinGuard
    }
}

#[cfg(test)]
struct TestBeforePinGuard;

#[cfg(test)]
impl Drop for TestBeforePinGuard {
    fn drop(&mut self) {
        PackfileStorage::set_test_before_pin(None);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_storage.rs"]
mod tests;
