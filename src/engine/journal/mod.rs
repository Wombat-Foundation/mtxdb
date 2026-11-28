//! Append-only, checksummed commit journal used by the packfile WAL path.
//!
//! This module owns the journal's disk framing and durability boundary. It
//! also provides [`JournalCoordinator`](crate::journal::JournalCoordinator),
//! which captures each sync caller's
//! target LSN and releases it only after a durable group covers that target.

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::layout::ShardType;
use crate::packfile::publish_signal::PublishSignal;

use crc32fast::Hasher;

mod coordinator;
mod segment;
mod txn_stage;
mod writer;
#[cfg(test)]
pub(crate) use segment::cut_segment_image_to_durable_mark;
#[cfg(test)]
use segment::{decode_mark_slot, decode_mutations};
use segment::{
    encode_group, encode_group_header, encode_group_trailer, encode_mark_slot, encode_mutation,
    file_header_bytes, invalid_data, newest_mark, read_durable_len_from_header, read_file_header,
    read_range_into, scan_bytes, scan_groups_from, sync_parent_dir, validate_file_header,
    write_file_header, MarkSlot,
};

const FILE_MAGIC: &[u8; 8] = b"MTXWAL01";
const GROUP_MAGIC: &[u8; 4] = b"MWG1";
const GROUP_COMMIT_MAGIC: &[u8; 4] = b"CMIT";

/// The sole on-disk journal format version. Every mutation carries its
/// physical pool tag, even when a journal currently serves only one pool. Its
/// header also records, per pool, the newest LSN of history a reclaim removed
/// ([`ReclaimedHistory`]).
const JOURNAL_VERSION: u32 = 1;

/// Per pool (indexed by [`ShardType::index`]), the newest LSN of a committed
/// group that touched the pool and that a reclaim rewrite removed from the
/// journal; 0 when none was. Only history actually dropped is recorded, never a
/// pool's reported coverage, and the marks only ever grow: each rewrite carries
/// the previous marks forward.
pub type ReclaimedHistory = [u64; ShardType::ALL.len()];

// File header field byte ranges. `FILE_HEADER_LEN` is the sum of the fields.
const FH_MAGIC: std::ops::Range<usize> = 0..8;
const FH_VERSION: std::ops::Range<usize> = 8..12;
const FH_BASE_SEQUENCE: std::ops::Range<usize> = 12..20;
const FH_BASE_LSN: std::ops::Range<usize> = 20..28;
// Version 1 keeps the base fields above, then holds the number of
// reclaimed-history slots (one per pool), the marks (`u64` each), and a CRC32
// over every header byte before it.
const FH_SLOTS: std::ops::Range<usize> = 28..32;
const FH_MARKS_AT: usize = 32;
const POOL_SLOTS: usize = ShardType::ALL.len();

/// Size of the unit a torn write can damage. 4096 matches common 4 KiB
/// physical sectors (advanced-format disks), which is the usual atomic write unit, though no filesystem or device guarantees it: with 512-byte
/// spacing both slots and the base header could share one such unit and a
/// single torn write could reach all three. It costs 12 KiB per segment.
const MARK_SECTOR_LEN: usize = 4096;
/// The durability mark lives in two alternating slots, each alone in its own
/// [`MARK_SECTOR_LEN`] unit after the one that holds the immutable base header. A torn write
/// of one slot can therefore damage only that slot: never the other slot, and
/// never the base header, which nothing rewrites after the file is created.
/// Each slot is `generation(8) | durable_len(8) | crc(4)`; an
/// all-zero slot has never been written and claims nothing.
const MARK_SLOT_OFFSETS: [usize; 2] = [MARK_SECTOR_LEN, 2 * MARK_SECTOR_LEN];
const MARK_SLOT_LEN: usize = 20;
/// Where journal groups begin: after the base header's sector and both mark
/// sectors. Everything that needs "the end of the header" uses this.
const FILE_HEADER_LEN: usize = 3 * MARK_SECTOR_LEN;

// Group header field byte ranges (`GROUP_HEADER_LEN` is the sum of the fields).
const GH_MAGIC: std::ops::Range<usize> = 0..4;
const GH_HEADER_LEN: std::ops::Range<usize> = 4..8;
const GH_SEQUENCE: std::ops::Range<usize> = 8..16;
const GH_FIRST_LSN: std::ops::Range<usize> = 16..24;
const GH_LAST_LSN: std::ops::Range<usize> = 24..32;
const GH_PAYLOAD_LEN: std::ops::Range<usize> = 32..40;
const GH_RECORD_COUNT: std::ops::Range<usize> = 40..44;
const GH_CRC: std::ops::Range<usize> = 44..48;
/// Group-header bytes covered by `GH_CRC` (everything before it).
const GH_CRC_COVERED: std::ops::Range<usize> = 0..GH_CRC.start;
const GROUP_HEADER_LEN: usize = GH_CRC.end;

// Group commit-trailer field byte ranges.
const GT_MAGIC: std::ops::Range<usize> = 0..4;
const GT_SEQUENCE: std::ops::Range<usize> = 4..12;
const GT_CRC: std::ops::Range<usize> = 12..16;
const GROUP_TRAILER_LEN: usize = GT_CRC.end;

// Mutation frame field byte ranges. A frame is `FRAME_FIXED_LEN` fixed bytes,
// then the payload, then `FRAME_TRAILER_LEN` CRC bytes.
const MF_KIND: std::ops::Range<usize> = 0..1;
/// Pool tag; always `1..=4`.
const MF_POOL: std::ops::Range<usize> = 1..2;
const MF_FLAGS: std::ops::Range<usize> = 2..4;
const MF_LSN: std::ops::Range<usize> = 4..12;
const MF_COLLECTION: std::ops::Range<usize> = 12..28;
const MF_NODE: std::ops::Range<usize> = 28..44;
const MF_PAYLOAD_LEN: std::ops::Range<usize> = 44..48;
const FRAME_FIXED_LEN: usize = MF_PAYLOAD_LEN.end;
const FRAME_TRAILER_LEN: usize = 4;
/// Smallest possible encoded mutation frame: fixed fields plus its CRC.
const MIN_FRAME_LEN: usize = FRAME_FIXED_LEN + FRAME_TRAILER_LEN;
const MAX_GROUP_LEN: u64 = 256 << 20;
const MAX_SEGMENT_LEN: u64 = (256 << 20) + FILE_HEADER_LEN as u64;

/// Segment size above which a sync forces a full index checkpoint, because only
/// a full checkpoint records journal coverage and lets the segment be
/// reclaimed. A quarter of [`MAX_SEGMENT_LEN`] leaves headroom for writes that
/// land while the checkpoint runs. It is a margin, not a guarantee: a writer
/// that appends more than the remaining space during one checkpoint still fills
/// the segment.
/// A coordinator starts with this and can be given another with
/// [`JournalCoordinator::set_reclaim_trigger_len`].
pub(crate) const RECLAIM_TRIGGER_LEN: u64 = MAX_SEGMENT_LEN / 4;

/// A reclaim that would copy at least this many suffix bytes is skipped when it
/// would drop under `1 / UNPROFITABLE_REWRITE_RATIO` of that suffix.
const UNPROFITABLE_REWRITE_MIN_COPY: u64 = 1 << 20;
/// See [`UNPROFITABLE_REWRITE_MIN_COPY`].
const UNPROFITABLE_REWRITE_RATIO: u64 = 8;

/// Segment length at which a stalled reclaim is treated as an emergency: three
/// quarters of `cap`. Validation and stall reporting share this calculation.
pub(crate) const fn emergency_len_for_cap(cap: u64) -> u64 {
    cap.saturating_sub(cap / 4)
}

/// Largest threshold that ensures forced reclaim starts by the emergency-zone
/// boundary of `cap`. The predicate fires only once `len > trigger`, hence the
/// minus 1.
pub(crate) const fn max_reclaim_trigger_for_cap(cap: u64) -> u64 {
    emergency_len_for_cap(cap).saturating_sub(1)
}

/// Largest threshold accepted from runtime configuration for the default
/// segment cap. [`JournalCoordinator::configure_reclaim_trigger_len`] further
/// limits it by the coordinator's live cap.
pub(crate) const MAX_RECLAIM_TRIGGER_LEN: u64 = max_reclaim_trigger_for_cap(MAX_SEGMENT_LEN);

/// A durable mutation represented in a journal group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mutation {
    /// Insert a content-addressed record.
    Put {
        /// Collection containing the record.
        collection_id: [u8; 16],
        /// Record identity.
        node_id: [u8; 16],
        /// Encoded record payload.
        payload: Vec<u8>,
    },
    /// Append a removed version for one record identity.
    RemoveRecord {
        /// Collection containing the record.
        collection_id: [u8; 16],
        /// Record identity being removed.
        node_id: [u8; 16],
    },
    /// Remove a collection. This is a journal operation because a replayed
    /// put must not resurrect a collection deleted by a later committed LSN.
    DeleteCollection {
        /// Collection to remove.
        collection_id: [u8; 16],
    },
}

/// Per-record write LSNs for one collection's live published suffix.
///
/// Authoritative for optimistic concurrency while the process runs, but
/// reconstructible rather than the sole source of truth: replayed and newly
/// published groups repopulate it, and a record absent here is seeded from its
/// frame's `last_write_lsn` tag before a conditional write validates. Each
/// record's version is the `last_lsn` of the journal group that last wrote it,
/// so a conditional write can name the exact version it observed.
///
/// `deleted_at` records the collection's last delete LSN. A record removed by
/// that delete has no entry, but still resolves to a version (the delete LSN)
/// so an "absent at version N" observation cannot be mistaken for a match
/// after a concurrent delete/recreate.
#[derive(Debug, Default)]
struct RecordVersions {
    nodes: HashMap<[u8; 16], u64>,
    deleted_at: Option<u64>,
}

/// Version trackers seeded from a retained WAL segment: each collection's
/// logical version and each record's write LSN.
type SeededVersions = (
    HashMap<(ShardType, [u8; 16]), u64>,
    HashMap<(ShardType, [u8; 16]), RecordVersions>,
);

/// A conditional write's expectation about one record: it observed `expected`
/// as the record's write LSN (or the collection's delete LSN when absent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordExpectation {
    /// Pool holding the record.
    pub pool: ShardType,
    /// Collection holding the record.
    pub collection_id: [u8; 16],
    /// Record identity.
    pub node_id: [u8; 16],
    /// The write LSN the caller observed for this record.
    pub expected: u64,
}

/// Upper bound for one SQL transaction's staged journal payloads.
///
/// Kept below the journal's 256 MiB group limit to leave room for framing and
/// other transactions sharing the process.
pub const MAX_TXN_STAGE_BYTES: usize = 64 << 20;
/// Bytes charged to a staged put for journal and mutation framing, in addition
/// to its payload. Kept public so bounded batch builders can size transactions
/// using the same accounting as [`TxnStage::stage_put`].
pub const TXN_STAGE_PUT_OVERHEAD_BYTES: usize = 64;

/// A transaction staged more than its configured stage limit.
///
/// Carried inside an [`io::Error`] of kind `InvalidInput`, so a caller can tell
/// "this transaction is too big, split it" from any other I/O failure without
/// matching on message text. See
/// [`crate::storage::StorageError::is_stage_too_large`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageTooLarge {
    /// Estimated staged bytes after the rejected mutation or batch.
    pub staged_bytes: usize,
    /// Maximum permitted staged bytes.
    pub limit_bytes: usize,
}

impl std::fmt::Display for StageTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "transaction journal stage exceeds its configured limit")
    }
}

impl std::error::Error for StageTooLarge {}

fn stage_too_large_with_limit(staged_bytes: usize, limit_bytes: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        StageTooLarge {
            staged_bytes,
            limit_bytes,
        },
    )
}

/// Lifecycle of a transaction's staged journal mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStageState {
    /// The SQL transaction attempt is active and may add mutations.
    Active,
    /// The published journal group is being materialized into pack/index
    /// storage. Per-mutation progress makes retries resumable.
    Materializing,
    /// The journal group is published; pack/index application is still
    /// pending or retrying.
    JournalPublished,
    /// The SQL attempt failed; retained callbacks must not publish its data.
    Discarded,
    /// Every per-pool group was appended successfully.
    Published,
}

/// Newest staged state per record for one pool, so a lookup costs the keys it
/// asks about rather than a pass over every staged mutation.
///
/// The index folds in the pool's mutations lazily, from `indexed` onward, so
/// the append paths need no bookkeeping of their own. Anything that removes or
/// reorders staged mutations must call [`Self::reset`].
///
/// `NodeId` is opaque and chosen by the caller (see `storage::NodeId`), so this
/// deliberately uses the standard randomized hasher rather than assuming the
/// ids are content hashes.
#[derive(Debug, Default)]
struct StageIndex {
    /// Number of leading mutations already folded in.
    indexed: usize,
    /// Position of the newest staged put or removal per `(collection, node)`.
    newest: HashMap<([u8; 16], [u8; 16]), usize>,
    /// Position of the newest staged delete per collection.
    last_delete: HashMap<[u8; 16], usize>,
}

impl StageIndex {
    fn reset(&mut self) {
        self.indexed = 0;
        self.newest.clear();
        self.last_delete.clear();
    }

    /// Fold in the mutations appended since the last call.
    fn sync(&mut self, mutations: &[Mutation]) {
        if self.indexed > mutations.len() {
            self.reset();
        }
        for (position, mutation) in mutations.iter().enumerate().skip(self.indexed) {
            match mutation {
                Mutation::Put {
                    collection_id,
                    node_id,
                    ..
                }
                | Mutation::RemoveRecord {
                    collection_id,
                    node_id,
                } => {
                    self.newest.insert((*collection_id, *node_id), position);
                }
                Mutation::DeleteCollection { collection_id } => {
                    self.last_delete.insert(*collection_id, position);
                }
            }
        }
        self.indexed = mutations.len();
    }
}

#[derive(Debug)]
struct TxnStageData {
    pools: [Vec<Mutation>; ShardType::ALL.len()],
    /// Lookup index over `pools`; see [`StageIndex`].
    index: [StageIndex; ShardType::ALL.len()],
    applied: [Vec<bool>; ShardType::ALL.len()],
    /// Collection-version preconditions validated at publication.
    expectations: Vec<CollectionExpectation>,
    /// Per-record version preconditions validated at publication.
    record_expectations: Vec<RecordExpectation>,
    /// Keys that already have a [`RecordExpectation`] above. A staged write is
    /// not enough: a blind put never validated the live value it replaced, so
    /// callers that need a read-modify-write precondition must still add one.
    record_expectation_keys: HashSet<(usize, [u8; 16], [u8; 16])>,
    bytes: usize,
    /// Successful pool appends. Retrying a callback after a partial error
    /// resumes at the failed pool instead of duplicating earlier groups.
    appended: [bool; ShardType::ALL.len()],
    /// Receipt identifying the published journal group owned by this stage.
    receipt: Option<CommitReceipt>,
}

/// Message prefix of the `InvalidData` error reported when reclaim has passed
/// a cursor's LSN. Match it through [`is_cursor_expired`].
const CURSOR_EXPIRED_PREFIX: &str = "journal cursor expired at LSN ";

/// Message of the `InvalidData` error reported when a reclaim rewrite lands
/// while a page is being read. Match it through [`is_segment_reclaimed`].
const SEGMENT_RECLAIMED_MESSAGE: &str =
    "journal segment was reclaimed while reading changes; retry from the cursor";

/// Whether `error` reports that reclaim already passed a replay cursor.
#[must_use]
pub fn is_cursor_expired(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::InvalidData
        && error.to_string().starts_with(CURSOR_EXPIRED_PREFIX)
}

/// Whether `error` reports a reclaim rewrite racing a page read; the read can
/// be retried from the same cursor.
#[must_use]
pub fn is_segment_reclaimed(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::InvalidData && error.to_string() == SEGMENT_RECLAIMED_MESSAGE
}

/// What a transaction's staged mutations say about one record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StagedLookup {
    /// The newest staged mutation for the record is a put of this payload.
    Put(Vec<u8>),
    /// A staged collection delete hides the record, and nothing newer puts it
    /// back.
    Deleted,
    /// Nothing staged names the record; the live store decides.
    Absent,
}

/// A collection-version precondition staged on a transaction.
///
/// The commit rejects with [`crate::storage::StorageError::StaleRead`] if the
/// collection's logical version at publication differs from `expected`. The
/// version is the global LSN of the last mutation touching the collection, so
/// it is only a valid compare-and-swap token when the caller read it together
/// with the data it is updating (see the versioned read path); a separate read
/// after the data can observe a version newer than the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectionExpectation {
    /// Pool holding the collection.
    pub pool: ShardType,
    /// Collection whose logical version must match.
    pub collection_id: [u8; 16],
    /// Logical version the caller read alongside the data.
    pub expected: u64,
}

/// A failed collection-version precondition, carried inside the publish path's
/// [`std::io::Error`] so the transaction layer can surface it as
/// [`crate::storage::StorageError::StaleRead`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StaleVersion {
    pub(crate) pool: ShardType,
    pub(crate) collection_id: [u8; 16],
    pub(crate) expected: u64,
    pub(crate) actual: u64,
}

impl std::fmt::Display for StaleVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stale read on {:?} collection {:?}: expected version {}, found {}",
            self.pool, self.collection_id, self.expected, self.actual
        )
    }
}

impl std::error::Error for StaleVersion {}

/// Transaction-local mutation buffer used by
/// [`crate::database::DatabaseTransaction`].
///
/// Journal entries for a SQL transaction are buffered here and published from
/// its post-commit callback. Call [`Self::discard`] from the transaction's
/// error callback; that is required because Synapse retains after-callbacks
/// across retry attempts.
///
/// # Storage boundary
///
/// [`Self::discard`] drops only buffered mutations. The database transaction
/// applies them to packs and indexes only after its caller declares commit,
/// then publishes the resulting shared-WAL group.
pub struct TxnStage {
    state: std::sync::atomic::AtomicU8,
    data: Mutex<TxnStageData>,
    limit_bytes: usize,
    #[cfg(test)]
    lookup_many_calls: std::sync::atomic::AtomicU64,
}

const fn pool_index(pool: ShardType) -> usize {
    match pool {
        ShardType::State => 0,
        ShardType::EventDag => 1,
        ShardType::ServerInfo => 2,
    }
}

/// Wire code for a pool tag in a mutation frame's flag byte.
///
/// Every frame carries a pool tag in `1..=3`; `0` is never written.
pub(crate) const fn pool_tag(pool: ShardType) -> u8 {
    match pool {
        ShardType::State => 1,
        ShardType::EventDag => 2,
        ShardType::ServerInfo => 3,
    }
}

/// Inverse of [`pool_tag`]. `0` and any other out-of-range code map to
/// `None`, which the frame decoder rejects.
pub(crate) const fn pool_from_tag(tag: u8) -> Option<ShardType> {
    match tag {
        1 => Some(ShardType::State),
        2 => Some(ShardType::EventDag),
        3 => Some(ShardType::ServerInfo),
        _ => None,
    }
}

/// One appended group's `last_lsn` and the distinct pools it carried, each
/// mapped to that same `last_lsn`. See [`JournalCoordinator::pending_promotions`].
type PendingPromotion = (u64, Vec<(ShardType, u64)>);

/// Distinct pools carried by one committed group, each mapped to the group's
/// `last_lsn`. A group is reclaimed atomically, so a pool's watermark must be
/// the whole group's end, not the LSN of its own frame within it: the group may
/// only be dropped once every pool in it has reported coverage through
/// `last_lsn`.
fn pool_extents(last_lsn: u64, pools: impl Iterator<Item = ShardType>) -> Vec<(ShardType, u64)> {
    let mut extents: Vec<(ShardType, u64)> = Vec::new();
    for pool in pools {
        if !extents.iter().any(|(existing, _)| *existing == pool) {
            extents.push((pool, last_lsn));
        }
    }
    extents
}

/// One decoded mutation frame, with the byte range it occupied in the segment
/// so a reader can re-read and re-validate the frame it has already trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalEntry {
    /// Monotonic LSN of this mutation.
    pub lsn: u64,
    /// Offset of the frame's first byte within the segment file.
    pub offset: u64,
    /// Encoded frame length in bytes (fixed fields + payload + CRC).
    pub frame_len: u64,
    /// Pool the frame belongs to. Every frame carries its pool tag.
    pub pool: ShardType,
    /// Decoded mutation.
    pub mutation: Mutation,
}

/// One recovered, complete, committed group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedGroup {
    /// Monotonic group sequence, starting at 1.
    pub sequence: u64,
    /// First mutation LSN in this group.
    pub first_lsn: u64,
    /// Last mutation LSN in this group.
    pub last_lsn: u64,
    /// Entries in LSN order.
    pub entries: Vec<JournalEntry>,
}

/// Receipt returned after a group and its commit trailer are appended.
///
/// A receipt is produced by [`Journal::append_group`]. The group is complete
/// and readable by a read-only scanner at this point, but not yet durable;
/// [`Journal::make_durable`] is what fsyncs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitReceipt {
    /// Monotonic group sequence.
    pub sequence: u64,
    /// First mutation LSN in the group.
    pub first_lsn: u64,
    /// Last mutation LSN in the group.
    pub last_lsn: u64,
    /// Bytes appended for this complete group, including header and trailer.
    pub bytes_written: u64,
}

/// Breakdown and counters for one journal durability request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JournalSyncTimings {
    /// Time spent waiting for the sync lock, which serializes fsyncs.
    pub journal_lock_wait: std::time::Duration,
    /// Time spent making the journal file durable.
    pub journal_fsync: std::time::Duration,
    /// Number of mutations this fsync made durable: the LSNs it advanced the
    /// durable boundary over, not a count of records it wrote (groups are
    /// appended when they are published, not when they are synced).
    pub journal_records: u64,
    /// Whether this request had to wait for the sync lock.
    pub journal_waiter: bool,
    /// Whether this request was already covered by another durable request.
    pub journal_coalesced: bool,
    /// Number of journal sync callers active when this request entered.
    pub journal_in_flight: u64,
}

/// Latency of blocked [`JournalCoordinator::wait_durable`] calls, in fixed
/// non-cumulative buckets: `<1ms`, `<10ms`, `<100ms`, `<1s`, and `>=1s`.
///
/// Sized for fsync-scale waits, unlike the sub-millisecond storage-operation
/// buckets. The total observation count is the sum of all five entries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DurableWaitLatency {
    /// Number of waits that had to block for a durable group.
    pub calls: u64,
    /// Sum of wait wall time.
    pub total: std::time::Duration,
    /// Largest single wait.
    pub max: std::time::Duration,
    /// Counts for `<1ms`, `<10ms`, `<100ms`, `<1s`, and `>=1s`.
    pub buckets: [u64; 5],
}

/// Lifetime durability accounting for one journal: how many requests shared
/// how many real fsyncs. Counters are monotone and never reset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DurabilityStats {
    /// [`JournalCoordinator::request_durable`] calls.
    pub durable_requests: u64,
    /// `wait_durable` calls already satisfied at entry (no blocking).
    pub durable_waits_already_durable: u64,
    /// Latency of `wait_durable` calls that had to block.
    pub durable_wait: DurableWaitLatency,
    /// Sync-barrier entries (`sync_through*`), including coalesced ones.
    pub sync_requests: u64,
    /// Sync-barrier entries that found the journal mutex occupied.
    pub sync_waiters: u64,
    /// Sync-barrier entries covered by another caller's fsync.
    pub sync_coalesced: u64,
    /// Real journal fsyncs that advanced the durable boundary.
    pub commits: u64,
    /// Mutations made durable across all commits (durable-LSN advance).
    pub commit_records: u64,
    /// Most mutations made durable by a single fsync.
    pub max_commit_records: u64,
}

impl DurabilityStats {
    /// Mean mutations made durable per real fsync, or `0.0` before any commit.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn records_per_commit(&self) -> f64 {
        if self.commits == 0 {
            0.0
        } else {
            self.commit_records as f64 / self.commits as f64
        }
    }
}

#[derive(Default)]
struct DurableWaitTotals {
    calls: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
    buckets: [AtomicU64; 5],
}

impl DurableWaitTotals {
    fn observe(&self, duration: std::time::Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_ns.fetch_max(nanos, Ordering::Relaxed);
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
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DurableWaitLatency {
        DurableWaitLatency {
            calls: self.calls.load(Ordering::Relaxed),
            total: std::time::Duration::from_nanos(self.total_ns.load(Ordering::Relaxed)),
            max: std::time::Duration::from_nanos(self.max_ns.load(Ordering::Relaxed)),
            buckets: std::array::from_fn(|i| self.buckets[i].load(Ordering::Relaxed)),
        }
    }
}

struct SyncInFlightGuard<'a>(&'a AtomicU64);

impl Drop for SyncInFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Test-only hook run between a sync's handle clone and its fsync; an `Err` it
/// returns stands in for the fsync failing.
#[cfg(test)]
type FsyncHook = Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

/// Work that must be durable before the WAL group covering a captured boundary
/// is: run by every commit path (explicit sync, background committer, close)
/// after the boundary is captured and before the fsync.
///
/// A derived structure that is written outside the WAL (the CSR adjacency
/// tails) uses it to make its writes for the group durable first, so a crash
/// can leave orphan derived rows but never a durable record whose derived rows
/// are missing. Publishers take no lock the hook needs, so they keep appending
/// while it runs; anything published after the capture is outside the group
/// being made durable and is covered by the next one. An `Err` fails the sync
/// without poisoning the journal: it says the hook's own work failed, not that
/// the kernel dropped WAL pages.
pub type PreSyncHook = Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

/// What a sync captured under the journal lock so it can flush without it.
struct SyncCapture {
    /// Duplicate of the segment file handle current at capture time.
    file: File,
    /// Newest visible LSN at capture time; the fsync makes everything at or
    /// below it durable.
    through_lsn: u64,
    /// Sequence of the newest appended group, for the receipt.
    sequence: u64,
    /// Segment path, for slow-fsync reporting.
    path: PathBuf,
    /// Length of the segment when the handle was cloned. Everything appended
    /// before that point is covered by the fsync, so this is what the
    /// durability mark may claim once the fsync returns.
    file_len: u64,
}

/// What the group directory says about reclaiming a shared segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedBoundary {
    /// The directory does not match the file, so it cannot decide.
    Untrusted,
    /// Even the first group is still needed (or the segment is empty).
    NothingCovered,
    /// Every group through this LSN may be dropped. The second field names the
    /// pool whose missing coverage stopped the cut (None when the whole segment
    /// is droppable).
    Through(u64, Option<ShardType>),
}

// Pool bits are `1 << index`, so they must fit in `GroupMark::pools`.
const _: () = assert!(
    ShardType::ALL.len() <= 8,
    "GroupMark::pools has one bit per pool"
);

/// The bit standing for `pool` in [`GroupMark::pools`].
fn pool_bit(pool: ShardType) -> u8 {
    1_u8 << pool.index()
}

/// What the in-memory directory remembers about one complete group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupMark {
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    /// Offset just after the group, so the bytes of every later group start
    /// here.
    end_offset: u64,
    /// Which pools have a frame in the group, as [`pool_bit`] bits.
    pools: u8,
}

/// A pool's reclaim coverage on a shared segment is stuck: the segment is over
/// the reclaim trigger and a reclaim could not shrink it. Remembered so forced
/// checkpoints back off until something can change the outcome.
#[derive(Clone, Copy, Debug)]
struct ReclaimStall {
    /// Segment length when the reclaim failed to shrink it.
    at_len: u64,
    /// `coverage_epoch` at that moment; a later value means a pool advanced its
    /// coverage, so a reclaim may now succeed.
    coverage_epoch: u64,
    /// Segment length when this run of failed reclaims began: the start of the
    /// grace period. Coverage moving does not restart it (the active pool's own
    /// reports would keep it from ever ending); only a reclaim that gets the
    /// segment back under the trigger does, by clearing the stall.
    grace_from: u64,
    /// Whether the stall has been counted and logged.
    reported: bool,
}

/// Something that makes a named pool checkpoint so it reports coverage, given
/// to a coordinator by whatever owns the pools.
type BlockerRemediation = Arc<dyn Fn(ShardType) + Send + Sync>;

thread_local! {
    /// Set while this thread is checkpointing lagging pools, so their syncs do
    /// not start another round.
    static REMEDIATING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What the in-memory group directory currently holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupDirectoryStats {
    /// Complete groups in the segment, one directory entry each.
    pub groups: u64,
    /// Bytes the directory's buffer has allocated (its capacity, an estimate
    /// of memory use, not a count of live entries).
    pub allocated_bytes: u64,
}

/// The live numbering of a journal segment: what reclaim has left, and what the
/// next append will take. Read from the in-memory header and group directory,
/// so this never scans the segment file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentPosition {
    /// Sequence of the oldest live group, or the sequence the next group would
    /// take when the segment holds none.
    pub base_sequence: u64,
    /// Sequence the next appended group will take.
    pub next_sequence: u64,
    /// LSN of the first mutation of the oldest live group, or the LSN the next
    /// group would take when the segment holds none.
    pub base_lsn: u64,
    /// LSN the next appended mutation will take.
    pub next_lsn: u64,
    /// Complete committed groups currently in the segment.
    pub groups: u64,
}

/// The smallest group the format allows: a header, one mutation frame and a
/// trailer.
const MIN_GROUP_LEN: u64 = (GROUP_HEADER_LEN + MIN_FRAME_LEN + GROUP_TRAILER_LEN) as u64;

/// Groups in a segment of the largest size made only of the smallest groups.
const WORST_CASE_DIRECTORY_GROUPS: u64 = MAX_SEGMENT_LEN / MIN_GROUP_LEN;

/// Bytes the directory needs at [`WORST_CASE_DIRECTORY_GROUPS`].
const WORST_CASE_DIRECTORY_BYTES: u64 =
    WORST_CASE_DIRECTORY_GROUPS * std::mem::size_of::<GroupMark>() as u64;

impl GroupDirectoryStats {
    /// The most entries a segment can ever hold, and so the most memory the
    /// directory can use for it: each group needs at least a header, one
    /// mutation frame and a trailer. It is only approached by a workload of
    /// single-record groups that keeps a nearly full segment unreclaimed, which
    /// the size trigger is there to prevent; the directory needs no cap of its
    /// own beyond it.
    #[must_use]
    pub const fn worst_case_groups() -> u64 {
        WORST_CASE_DIRECTORY_GROUPS
    }

    /// Bytes the directory uses at [`Self::worst_case_groups`].
    #[must_use]
    pub const fn worst_case_bytes() -> u64 {
        WORST_CASE_DIRECTORY_BYTES
    }
}

/// The directory entries for the groups a scan returned.
fn marks_from_scan(scan: &Scan) -> Vec<GroupMark> {
    scan.groups
        .iter()
        .zip(&scan.group_ends)
        .map(|(group, end_offset)| GroupMark {
            sequence: group.sequence,
            first_lsn: group.first_lsn,
            last_lsn: group.last_lsn,
            end_offset: *end_offset,
            pools: group
                .entries
                .iter()
                .fold(0, |mask, entry| mask | pool_bit(entry.pool)),
        })
        .collect()
}

/// The last group of the longest prefix of `groups` that a shared segment may
/// drop: every pool with a frame in each group has reported coverage through the
/// group's `last_lsn`. `None` when even the first group is still needed.
///
/// The one place this rule lives: the directory and the scan fallback both call
/// it, so they cannot disagree.
fn boundary_through(
    groups: &[GroupMark],
    covered: &HashMap<ShardType, u64>,
) -> Option<(u64, Option<ShardType>)> {
    let mut boundary = None;
    for group in groups {
        for (index, pool) in ShardType::ALL.into_iter().enumerate() {
            if group.pools & (1_u8 << index) != 0
                && covered.get(&pool).is_none_or(|lsn| *lsn < group.last_lsn)
            {
                // This pool's coverage stops the cut here; report it as the
                // blocker so reclaim can say why the suffix was retained.
                return boundary.map(|lsn| (lsn, Some(pool)));
            }
        }
        boundary = Some(group.last_lsn);
    }
    boundary.map(|lsn| (lsn, None))
}

/// Result of validating a journal file.
#[derive(Debug)]
pub struct Scan {
    /// Complete committed groups in sequence order.
    pub groups: Vec<CommittedGroup>,
    /// Byte offset immediately after the last complete committed group.
    pub valid_len: u64,
    /// True when bytes after `valid_len` were an incomplete final append.
    pub truncated_tail: bool,
    /// LSN the segment's first group would carry — the file header's base LSN.
    /// Lets a reader detect a reclaimed segment that is now header-only, where
    /// there are no groups to compare but the base still moved past what the
    /// reader's index incorporated.
    pub base_lsn: u64,
    /// Sequence the segment's first group would carry — the file header's base
    /// sequence. Like [`Self::base_lsn`], this survives a reclaim that removed
    /// every group.
    pub base_sequence: u64,
    /// The last bytes (up to [`CONSUMED_TAIL_LEN`]) of the complete groups this
    /// scan consumed, never reaching into the file header. A read-only overlay
    /// remembers them, appended to what it already remembered, so it later
    /// compares the disk against the bytes it actually built from instead of
    /// against a separate read that could race a rewrite.
    pub(crate) consumed_tail: Vec<u8>,
    /// Absolute offset just after each group in `groups`, in the same order.
    pub(crate) group_ends: Vec<u64>,
}

/// How many bytes ending at the last consumed group a read-only overlay
/// remembers to notice that the consumed prefix was rewritten.
pub(crate) const CONSUMED_TAIL_LEN: usize = 1024;

/// How long the remembered window is for a segment whose last complete group
/// ends at `valid_len`: the last [`CONSUMED_TAIL_LEN`] bytes, but never
/// reaching into the file header.
#[must_use]
pub(crate) fn consumed_tail_len(valid_len: u64) -> usize {
    let header_len = u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX);
    let available = valid_len.saturating_sub(header_len);
    usize::try_from(available).map_or(CONSUMED_TAIL_LEN, |len| len.min(CONSUMED_TAIL_LEN))
}

impl Scan {
    /// Empty result for a journal segment that does not exist or has not yet
    /// acquired a complete file header.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            groups: Vec::new(),
            valid_len: 0,
            truncated_tail: false,
            base_lsn: 0,
            base_sequence: 0,
            consumed_tail: Vec::new(),
            group_ends: Vec::new(),
        }
    }
}

/// Outcome of compacting a journal segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reclaim {
    /// Complete committed groups left in the segment because the checkpoint had
    /// not yet materialized them.
    pub retained_groups: u64,
    /// Bytes removed from the segment.
    pub reclaimed_bytes: u64,
    /// Size of the retained suffix, i.e. the rebuilt segment minus its header.
    /// This is what a rewrite would copy and fsync, and what makes `copy` /
    /// `fsync` scale with segment size, so it is the number to bound. On a
    /// skipped reclaim nothing is copied, but this still reports the suffix
    /// size that made the rewrite unprofitable.
    pub retained_bytes: u64,
    /// The pool whose missing coverage stopped the cut, so the suffix could not
    /// be dropped. None when the whole segment was droppable, or the caller
    /// could not say. Diagnostics only.
    pub blocked_by: Option<ShardType>,
    /// Time deciding the reclaim boundary (directory lookup, or a scan when the
    /// directory could not be trusted).
    pub boundary: std::time::Duration,
    /// Time reading and re-encoding the retained suffix into the rebuilt
    /// segment.
    pub copy: std::time::Duration,
    /// Time writing, fsyncing and renaming the rebuilt segment into place.
    pub fsync: std::time::Duration,
}

impl Default for Reclaim {
    fn default() -> Self {
        Self {
            retained_groups: 0,
            reclaimed_bytes: 0,
            retained_bytes: 0,
            blocked_by: None,
            boundary: std::time::Duration::ZERO,
            copy: std::time::Duration::ZERO,
            fsync: std::time::Duration::ZERO,
        }
    }
}

/// A single-writer journal file. [`Self::append_group`] writes a complete
/// group and its commit trailer; [`Self::make_durable`] fsyncs it;
/// [`Self::commit_group`] does both. Calls are serialized by the caller's
/// coordinator (or by `&mut self`).
pub struct Journal {
    path: PathBuf,
    file: File,
    /// Current on-disk length tracked after open, append, and reclaim. Keeping
    /// this in memory avoids an fstat on every group publication.
    file_len: u64,
    /// Base sequence and LSN from the file header.
    base_sequence: u64,
    base_lsn: u64,
    /// What reclaim rewrites removed, per pool, as recorded in the header.
    reclaimed_history: ReclaimedHistory,
    /// Generation of the newest durability mark written to this segment; the
    /// next mark uses the next generation, in the other slot.
    mark_generation: u64,
    next_sequence: u64,
    next_lsn: u64,
    poisoned: bool,
    /// Largest the segment may grow. Always [`MAX_SEGMENT_LEN`] outside tests.
    segment_cap: u64,
    /// In-memory directory of the complete groups in the segment, in order.
    ///
    /// It is a cache of what a scan of the file would find, built when the
    /// segment opens and extended on every append. Reclaim uses it to pick the
    /// boundary and to read only the retained suffix. It is trusted only while
    /// its last group ends exactly at `file_len`; otherwise reclaim falls back
    /// to scanning the file.
    ///
    /// This relies on this handle being the only writer of the segment: the
    /// root's writer lock keeps a second writer out, and readers never modify
    /// it. A file rewritten behind the handle to the same length would not be
    /// noticed by the length check. That is unsupported, but the retained suffix
    /// is still decoded and checked before it is copied, so a damaged group
    /// there is reported instead of being carried into the new segment.
    groups: Vec<GroupMark>,
    /// Test-only: records the intervals during which `install_rebuilt` is
    /// replacing the segment, so a test can check whether a read's own time
    /// interval intersected a rewrite.
    #[cfg(test)]
    rewrite_phase: Arc<RewritePhase>,
}

/// Test-only rewrite-window recorder: monotonic nanoseconds from one base.
#[cfg(test)]
pub(crate) struct RewritePhase {
    base: std::time::Instant,
    windows: Mutex<Vec<(u64, u64)>>,
}

#[cfg(test)]
impl RewritePhase {
    fn new() -> Self {
        Self {
            base: std::time::Instant::now(),
            windows: Mutex::new(Vec::new()),
        }
    }

    /// Current time in nanoseconds on this recorder's monotonic base.
    pub(crate) fn now_ns(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn record(&self, start: u64, end: u64) {
        self.windows.lock().push((start, end));
    }

    /// Completed rewrite windows as `(start, end)` nanosecond intervals.
    pub(crate) fn windows(&self) -> Vec<(u64, u64)> {
        self.windows.lock().clone()
    }
}

/// Marks the rewrite window for [`Journal::rewrite_phase`].
#[cfg(test)]
struct RewritePhaseGuard {
    phase: Arc<RewritePhase>,
    start: u64,
}

#[cfg(test)]
impl RewritePhaseGuard {
    fn enter(phase: &Arc<RewritePhase>) -> Self {
        Self {
            phase: Arc::clone(phase),
            start: phase.now_ns(),
        }
    }
}

#[cfg(test)]
impl Drop for RewritePhaseGuard {
    fn drop(&mut self) {
        let end = self.phase.now_ns();
        self.phase.record(self.start, end);
    }
}

/// An fsync at least this slow is reported on stderr when it happens.
const SLOW_FSYNC_WARN: std::time::Duration = std::time::Duration::from_secs(1);

/// Cross-pool coverage bookkeeping for reclaiming a shared segment.
#[derive(Default)]
struct CoverageState {
    /// Highest durable checkpoint coverage reported per pool. A group is
    /// reclaimable only once every pool that contributed a frame to it has
    /// reported coverage through the group's `last_lsn`.
    covered: HashMap<ShardType, u64>,
}

/// Maximum time [`JournalCoordinator::wait_durable`] sleeps before re-checking
/// the durable boundary, poison bit, and committer presence. Bounds how long a
/// waiter can lag a commit and how quickly it notices a stopped committer.
const DURABILITY_WAIT_POLL: Duration = Duration::from_millis(50);

/// Bounded group-commit policy for
/// [`JournalCoordinator::start_background_committer`].
///
/// The committer fsyncs at most once per `interval`, coalescing every mutation
/// published in that window into one group. It flushes early when the number of
/// published-but-uncommitted records reaches `max_pending`, so an unbroken
/// burst cannot grow the pending queue without bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupCommitConfig {
    /// Maximum time a published mutation waits before the committer fsyncs it.
    /// Also the crash-loss window when the process dies before a flush.
    pub interval: Duration,
    /// Published-but-uncommitted record count that forces an early flush.
    pub max_pending: u64,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            max_pending: 4096,
        }
    }
}

impl GroupCommitConfig {
    /// A policy that fsyncs under a fixed interval with the default
    /// `max_pending` bound.
    #[must_use]
    pub const fn with_interval(interval: Duration) -> Self {
        Self {
            interval,
            max_pending: 4096,
        }
    }
}

/// A handle to a durability request registered with
/// [`JournalCoordinator::request_durable`].
///
/// Holding a token is not itself durable: pass it to
/// [`JournalCoordinator::wait_durable`] to block until the group covering its
/// LSN has been fsynced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DurabilityToken {
    lsn: u64,
}

impl DurabilityToken {
    /// The LSN this token requests durability through.
    #[must_use]
    pub const fn lsn(self) -> u64 {
        self.lsn
    }

    /// Whether a durable boundary of `committed_lsn` already satisfies this
    /// token.
    #[must_use]
    pub const fn is_satisfied_by(self, committed_lsn: u64) -> bool {
        self.lsn <= committed_lsn
    }
}

/// Position in one journal from which a caller may replay later committed
/// groups. Cursors are created by a snapshot scan and are intentionally opaque
/// so they cannot be accidentally reused with a different journal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalCursor {
    path: PathBuf,
    incarnation: u64,
    lsn: u64,
    /// Segment base LSN when this cursor was created or last advanced. A reclaim
    /// rewrite raises it, which is how an empty incremental read is told apart
    /// from a legitimately idle one.
    base_lsn: u64,
    /// Absolute file offset of the first byte after the group at `lsn`, so the
    /// next [`JournalCoordinator::changes_since`] page reads only the appended
    /// tail instead of rescanning the retained prefix. A reclaim rewrite moves
    /// the surviving groups, invalidating this hint; the read detects that and
    /// rebuilds from the file header.
    resume_offset: u64,
}

impl JournalCursor {
    /// The last LSN represented by the snapshot.
    #[must_use]
    pub const fn lsn(&self) -> u64 {
        self.lsn
    }

    /// Whether reclaim had already passed this cursor when it was created.
    pub(crate) const fn is_behind_retained_history(&self) -> bool {
        self.lsn.saturating_add(1) < self.base_lsn
    }

    /// Newest LSN no longer retained, the lowest LSN a cursor can start at.
    pub(crate) const fn retained_floor_lsn(&self) -> u64 {
        self.base_lsn.saturating_sub(1)
    }
}

/// RAII pin that prevents journal reclaim from passing a replay cursor.
///
/// Keep this value alive while consuming pages from the cursor. Dropping it
/// releases the retained-history guarantee. Until [`Self::advance`] is called
/// it pins the original cursor rather than advancing with individual pages, so
/// an interrupted or partially applied replay can safely resume from its
/// starting snapshot.
#[must_use = "dropping the replay lease removes its pin and lets reclaim pass the cursor"]
pub struct JournalReplayLease {
    registry: Arc<ReplayLeaseRegistry>,
    id: u64,
}

/// A replay lease was revoked because the history it pinned exceeded the
/// configured retention cap.
///
/// This is a lease-state condition, not an I/O failure: the caller must discard
/// its partial rebuild and take a fresh snapshot. Reads through the lease's
/// cursor fail with the existing cursor-expired `InvalidData` error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayLeaseRevoked;

impl std::fmt::Display for ReplayLeaseRevoked {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("journal replay lease was revoked by the retention cap")
    }
}

impl std::error::Error for ReplayLeaseRevoked {}

impl JournalReplayLease {
    /// Raise this lease's floor to `through_lsn`, letting reclaim pass every
    /// group at or below it.
    ///
    /// Call this once the page through `through_lsn` has been durably applied:
    /// the history below it is then no longer needed, so a long rebuild does not
    /// pin the whole retained WAL. Advancing gives up the ability to resume from
    /// the original snapshot, so only advance past pages that are safely
    /// persisted. The floor is monotonic: a stale or out-of-order call can never
    /// lower it.
    ///
    /// # Errors
    /// Returns [`ReplayLeaseRevoked`] when the retention cap revoked this lease.
    /// A revoked lease cannot be advanced; the caller must discard its partial
    /// rebuild.
    pub fn advance(&self, through_lsn: u64) -> Result<(), ReplayLeaseRevoked> {
        let mut pins = self.registry.pins.lock();
        let Some(pin) = pins.get_mut(&self.id) else {
            return Ok(());
        };
        if pin.revoked {
            return Err(ReplayLeaseRevoked);
        }
        pin.lsn = pin.lsn.max(through_lsn);
        Ok(())
    }

    /// Whether the retention cap revoked this lease.
    #[must_use]
    pub fn is_revoked(&self) -> bool {
        self.registry
            .pins
            .lock()
            .get(&self.id)
            .is_some_and(|pin| pin.revoked)
    }
}

impl Drop for JournalReplayLease {
    fn drop(&mut self) {
        self.registry.pins.lock().remove(&self.id);
    }
}

#[derive(Default)]
struct ReplayLeaseRegistry {
    /// Lease ID to its pinned LSN and whether the retention cap revoked it.
    pins: Mutex<HashMap<u64, LeasePin>>,
}

struct LeasePin {
    /// Last LSN represented by the lease's snapshot cursor.
    lsn: u64,
    /// Set by the retention cap. A revoked pin no longer constrains reclaim.
    revoked: bool,
    /// When the lease was created, for the oldest-lease age stat.
    created: std::time::Instant,
}

/// One bounded page of complete journal groups after a [`JournalCursor`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalChangesPage {
    /// Complete groups in journal order. Groups are never split across pages.
    pub groups: Vec<CommittedGroup>,
    /// Cursor to pass to the next page; advances across all groups in this
    /// page, including groups whose mutations a caller later filters out.
    pub next_cursor: JournalCursor,
    /// Last LSN included by `next_cursor`.
    pub through_lsn: u64,
    /// Whether another currently durable group follows this page.
    pub has_more: bool,
    /// Durable high-water mark sampled *before* the WAL read. Every durable
    /// group at or below it is represented by this page's scan, so a caller
    /// can tell exactly which replay horizon the page covers. `has_more`
    /// refers to this horizon, not to groups that become durable later.
    pub horizon_lsn: u64,
}

/// Cumulative path counters for [`JournalCoordinator::changes_since`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayPageStats {
    /// Pages read from the cursor's resume offset without rebuilding the scan
    /// from the journal header.
    pub resumed_pages: u64,
    /// Pages that had to fall back to a full retained-segment scan because the
    /// cursor's resume offset could not be trusted.
    pub full_scan_fallbacks: u64,
}

#[cfg(test)]
thread_local! {
    /// A one-shot hook run inside [`JournalCoordinator::changes_since`] after
    /// the durable horizon is sampled but before the WAL is read, so a test can
    /// force a commit that lands inside the scan yet beyond the fixed horizon.
    static CHANGES_SINCE_BEFORE_READ: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

/// Install the [`CHANGES_SINCE_BEFORE_READ`] hook for the current thread.
#[cfg(test)]
pub(crate) fn set_changes_since_before_read_hook(hook: impl FnOnce() + 'static) {
    CHANGES_SINCE_BEFORE_READ.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(test)]
fn take_changes_since_before_read_hook() -> Option<Box<dyn FnOnce()>> {
    CHANGES_SINCE_BEFORE_READ.with(std::cell::RefCell::take)
}

#[cfg(test)]
thread_local! {
    /// When set, [`JournalCoordinator::changes_since`] treats the in-memory
    /// group directory as untrusted, so a test can exercise the incremental
    /// resume path that has no directory cap.
    static CHANGES_SINCE_FORCE_UNTRUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Force [`JournalCoordinator::changes_since`] to distrust the directory on the
/// current thread. Used by tests of the incremental resume path.
#[cfg(test)]
pub(crate) fn set_changes_since_force_untrusted(force: bool) {
    CHANGES_SINCE_FORCE_UNTRUSTED.with(|cell| cell.set(force));
}

#[cfg(test)]
fn changes_since_force_untrusted() -> bool {
    CHANGES_SINCE_FORCE_UNTRUSTED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn changes_since_force_untrusted() -> bool {
    false
}

#[cfg(test)]
thread_local! {
    /// Count of [`JournalCoordinator::changes_since`] calls that had to rebuild
    /// from the file header instead of resuming incrementally, so a test can
    /// prove a page stayed on the resume path.
    static CHANGES_SINCE_FULL_SCANS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_changes_since_full_scan() {
    CHANGES_SINCE_FULL_SCANS.with(|cell| cell.set(cell.get().saturating_add(1)));
}

#[cfg(test)]
fn changes_since_full_scans() -> u64 {
    CHANGES_SINCE_FULL_SCANS.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn note_changes_since_full_scan() {}

struct BackgroundCommitter {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Lifecycle of the background committer slot.
///
/// `Stopping` is held across the join so a concurrent
/// [`JournalCoordinator::start_background_committer`] cannot spawn a second
/// committer while the old thread is still winding down.
enum BackgroundState {
    Stopped,
    Running(BackgroundCommitter),
    Stopping,
}

/// A terminal failure of the background committer, recorded so `wait_durable`
/// and `stop_background_committer` can surface it instead of waiting forever on
/// a worker that is no longer running. A private snapshot of an [`io::Error`]
/// (`io::Error` is not `Clone`).
#[derive(Clone, Debug)]
struct BackgroundFailure {
    kind: io::ErrorKind,
    message: String,
}

impl BackgroundFailure {
    fn from_io(error: &io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
        }
    }

    fn into_io(self) -> io::Error {
        io::Error::new(self.kind, self.message)
    }
}

/// Serializes mutation publication and durable commits for one journal.
///
/// Every publication (an autocommit mutation or a transaction group) appends
/// one complete group under the `publication` lock, assigning its LSNs there,
/// and becomes visible without an fsync. A sync caller captures the published
/// LSN and fsyncs every visible group up to it in one call, so many small
/// groups share one fsync. Concurrent sync callers recheck the committed LSN
/// after taking the journal lock.
pub struct JournalCoordinator {
    journal: Mutex<Journal>,
    path: PathBuf,
    /// Serializes every group publication: autocommit mutations and
    /// transaction groups alike.
    publication: Mutex<()>,
    published_lsn: AtomicU64,
    /// Highest LSN whose group is complete (trailer appended) but not
    /// necessarily fsynced. Advanced between [`Journal::append_group`] and
    /// [`Journal::make_durable`] so a read-only overlay can observe a
    /// committed-but-unflushed group.
    visible_lsn: AtomicU64,
    committed_lsn: AtomicU64,
    /// Segment length above which a sync forces a reclaiming checkpoint; see
    /// [`RECLAIM_TRIGGER_LEN`].
    reclaim_trigger_len: AtomicU64,
    /// Optional cap, in bytes, on the history one replay lease may retain.
    /// [`u64::MAX`] means unlimited (the default). Retention is measured from
    /// the byte after the last complete group at or below the lease's pinned
    /// LSN through the current end of the segment; the file header is excluded.
    replay_lease_cap: AtomicU64,
    /// Replay leases revoked by the retention cap.
    replay_leases_revoked: AtomicU64,
    /// Set while a reclaim over the trigger cannot shrink the segment.
    reclaim_stall: Mutex<Option<ReclaimStall>>,
    /// Set when an active replay lease, rather than pool coverage, holds the
    /// reclaim boundary back. Kept separate from pool-stall remediation.
    replay_lease_stall: Mutex<Option<ReclaimStall>>,
    /// Advances whenever any pool's reported coverage does.
    coverage_epoch: AtomicU64,
    /// Times a reclaim over the trigger has been found unable to shrink the
    /// segment.
    reclaim_stalls: AtomicU64,
    /// Test-only: runs inside a reclaim, after the sync and journal locks are
    /// held and before the segment is rewritten.
    #[cfg(test)]
    reclaim_hook: Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>>,
    /// Checkpoints the pools that hold a stalled reclaim back.
    blocker_remediation: Mutex<Option<BlockerRemediation>>,
    /// `(segment length, coverage epoch)` at the last remediation attempt.
    last_remediation: Mutex<Option<(u64, u64)>>,
    /// Optional shared, cross-pool group-sequence allocator. When present,
    /// each committed group draws its sequence from here instead of the
    /// segment's own counter, so groups across several pool segments share one
    /// global order. See [`Self::with_shared_sequence`].
    sequence: Option<Arc<AtomicU64>>,
    /// Per-pool durable coverage, used to reclaim a shared segment only up to
    /// the point every pool present in it has materialized. See
    /// [`Self::report_pool_coverage`] and [`Self::reclaim_shared`].
    coverage: Mutex<CoverageState>,
    /// Highest committed group `last_lsn` that carried a frame for each pool.
    ///
    /// A pool's checkpoint may only record this, never the global
    /// [`Self::committed_lsn`]: one shared group can interleave several pools'
    /// frames, so the global value can include LSNs whose frames this pool's
    /// index has not materialized. Recording the global value would both claim
    /// coverage the checkpoint lacks and let reclaim drop another pool's
    /// uncovered frames. See [`Self::committed_lsn_for_pool`].
    pool_committed: Mutex<HashMap<ShardType, u64>>,
    /// Logical version of each collection: the global LSN of the last published
    /// mutation that touched it, keyed by `(pool, collection_id)`. Bumped in
    /// [`Self::publish_groups`] at the point the group becomes visible, so a
    /// conditional commit's validation and the version advance are one step
    /// under the publication lock. Seeded from the recovered segment on open;
    /// reclaim drops the WAL history a version was derived from, so a collection
    /// whose last write is reclaimed reads its checkpoint baseline after a
    /// reopen (see `version_floor`) until its next write.
    collection_versions: Mutex<HashMap<(ShardType, [u8; 16]), u64>>,
    /// Per-record write LSNs for the live published suffix, keyed by
    /// `(pool, collection)`. See [`RecordVersions`].
    record_versions: Mutex<HashMap<(ShardType, [u8; 16]), RecordVersions>>,
    /// Per-collection checkpoint baseline, fixed at open. A record that is
    /// absent with no live entry or tombstone resolves to this instead of `0`,
    /// so a delete whose WAL group was reclaimed cannot make a pre-delete "absent
    /// at 0" observation match again. Unlike `collection_versions` it does not
    /// advance with later writes, so an absent record's token stays stable.
    version_floor: Mutex<HashMap<(ShardType, [u8; 16]), u64>>,
    /// First LSNs of published groups, including autocommit groups, whose
    /// writes are not yet all in the packs. See [`Self::publish_groups`].
    unmaterialized: Mutex<std::collections::BTreeSet<u64>>,
    /// Wakes versioned reads blocked until a published group reaches the packs.
    /// A read that cannot certify its boundary waits here instead of polling;
    /// `transaction_materialized` and each pool's materialized watermark
    /// advance notify it.
    materialized_cv: Condvar,
    materialized_lock: Mutex<()>,
    /// Appended-but-not-yet-committed groups, with the highest LSN each carried
    /// per pool. Appending a transaction group makes it visible before it is
    /// fsynced; when a later durable commit passes it,
    /// [`Self::promote_pool_committed`] drains the entry so every pool's
    /// watermark still advances from a coalesced fsync.
    pending_promotions: Mutex<Vec<PendingPromotion>>,
    /// Committed groups recovered when this coordinator's segment was opened.
    /// Retained so a store attaching to a coordinator it did not construct
    /// (the shared-WAL path, where one coordinator serves every pool) can seed
    /// its replay set from the same scan.
    recovered: Vec<CommittedGroup>,
    /// Mirrors the journal's poison bit, so `publish` can reject without
    /// taking the `journal` mutex (which a sync holds across its fsync).
    poisoned: AtomicBool,
    /// Why a published group could not be applied to the packs and index, once
    /// one failed. Unlike `poisoned` this leaves the committer running, so
    /// groups already published still reach disk and a reopen replays them.
    apply_failure: std::sync::OnceLock<String>,
    /// Number of sync requests entering the coordinator.
    sync_calls: AtomicU64,
    /// Number of sync requests that found the journal mutex occupied.
    journal_waiters: AtomicU64,
    /// Number of requests covered without appending or fsyncing themselves.
    coalesced_syncs: AtomicU64,
    sync_in_flight: AtomicU64,
    /// Highest LSN a caller has asked to become durable through
    /// [`Self::request_durable`]. Purely a wake-up/observability bound: the
    /// background committer commits the whole published prefix, not this
    /// exact value.
    durable_requested: AtomicU64,
    /// Serializes fsyncs and segment replacement, and is held across the whole
    /// fsync. Publishers never take it, so they make progress while the device
    /// flushes. A sync caller takes it, rechecks `committed_lsn` (a concurrent
    /// fsync usually covered it), then briefly takes `journal` to clone the
    /// file handle and capture the newest visible LSN.
    ///
    /// Lock order: `sync_lock` -> `journal`. Reclaim takes `sync_lock` first so
    /// the segment cannot be replaced under an in-flight fsync. Nothing may
    /// take `sync_lock` while holding `journal`.
    sync_lock: Mutex<()>,
    /// Test-only hook run after the file handle is cloned and before the
    /// fsync, with `journal` released, so a test can park a sync mid-flight.
    /// An `Err` it returns is treated as the fsync failing.
    #[cfg(test)]
    fsync_hook: Mutex<Option<FsyncHook>>,
    /// Work to make durable before each WAL group's fsync. See
    /// [`PreSyncHook`].
    pre_sync_hook: Mutex<Option<PreSyncHook>>,
    /// Guards [`Self::durable_cv`]. The condvar carries no state of its own;
    /// waiters re-check [`Self::committed_lsn`] and the committer's presence
    /// after every wake, so this only provides the required mutex pairing.
    ///
    /// Lock order: `journal` may be held while acquiring `durable_lock` (the
    /// commit path wakes waiters after recording the commit). The reverse is
    /// forbidden: never acquire `journal` while holding `durable_lock`. Both
    /// the committer loop and `wait_durable` drop the guard before entering
    /// journal I/O.
    durable_lock: Mutex<()>,
    /// Signals `wait_durable` callers and the background committer when the
    /// durable boundary may have advanced or the committer has stopped.
    durable_cv: Condvar,
    /// Lifecycle of the background committer. Additive: a coordinator whose
    /// state is [`BackgroundState::Stopped`] keeps the historical
    /// blocking-barrier behavior for `wait_durable`/`sync_through`.
    background: Mutex<BackgroundState>,
    /// Terminal committer failure, surfaced by [`Self::wait_durable`] and
    /// [`Self::stop_background_committer`]. Cleared when a committer is
    /// (re)started.
    background_failure: Mutex<Option<BackgroundFailure>>,
    /// Published-but-uncommitted count at which `publish` wakes the committer
    /// for an early flush. Zero when no committer is running (no wakeups).
    committer_wake_threshold: AtomicU64,
    /// Monotonic commit generation, bumped whenever `committed_lsn` advances
    /// (background, explicit, or fallback commit). A new epoch re-arms the
    /// threshold wake without any explicit clear step.
    commit_epoch: AtomicU64,
    /// Commit epoch of the last threshold-crossing wake. Publishers wake only
    /// when it differs from the current [`Self::commit_epoch`], so a burst
    /// yields one wake. Initialised to `u64::MAX`, so it never matches epoch 0
    /// and the first crossing always wakes.
    threshold_notified_epoch: AtomicU64,
    /// Test-only readiness signal: incremented, under `durable_lock`, each time
    /// the committer is about to park. Lets tests wait for a parked worker
    /// instead of sleeping. Absent from non-test builds.
    #[cfg(test)]
    committer_parks: AtomicU64,
    /// Test-only: threshold wakes whose send path `publish` selected.
    /// Incremented inline, so a test can assert a burst selected (or
    /// suppressed) its wake without waiting for the committer to react.
    /// `notify_all` has no observable success/failure, so this counts decisions,
    /// not deliveries.
    #[cfg(test)]
    threshold_wakes: AtomicU64,
    /// Number of background group commits that appended and fsynced a group.
    background_commits: AtomicU64,
    /// Number of background commit attempts already covered by a concurrent
    /// durable group (never counted for idle timer ticks).
    background_coalesced: AtomicU64,
    /// `request_durable` calls.
    durable_requests: AtomicU64,
    /// `wait_durable` calls satisfied without blocking.
    durable_waits_already_durable: AtomicU64,
    /// Latency of `wait_durable` calls that blocked.
    durable_wait: DurableWaitTotals,
    /// Real fsyncs that advanced the durable boundary.
    commits: AtomicU64,
    /// Durable-LSN advance summed over all commits.
    commit_records: AtomicU64,
    /// Largest durable-LSN advance made by one fsync.
    max_commit_records: AtomicU64,
    /// Cross-process `(epoch, visible_lsn)` signal beside the segment. Lazily
    /// created by [`Self::enable_publish_signal`]; a coordinator with no
    /// signal cannot enable the worker fast path.
    publish_signal: std::sync::OnceLock<Result<Arc<PublishSignal>, (io::ErrorKind, String)>>,
    /// Active replay cursors pin the retained WAL prefix until their lease is
    /// dropped. Shared by leases so their Drop implementation can release a
    /// pin even after the coordinator borrow has ended.
    replay_leases: Arc<ReplayLeaseRegistry>,
    next_replay_lease: AtomicU64,
    /// Cumulative incremental-resume and fallback counts for replay paging.
    replay_page_stats: ReplayPageStatsAtomic,
}

#[derive(Default)]
struct ReplayPageStatsAtomic {
    resumed_pages: AtomicU64,
    full_scan_fallbacks: AtomicU64,
}

/// Directory-derived boundaries for one [`JournalCoordinator::changes_since`]
/// page: the base LSN to guard against a concurrent reclaim, the byte offset to
/// stop reading at (through at most `limit + 1` durable groups), and whether the
/// in-memory directory matched the file. The read starts at the cursor's own
/// resume offset, so the directory is only needed to cap the read.
struct ChangesWindow {
    base_lsn: u64,
    end: u64,
    expected_lsn: u64,
    has_durable_group: bool,
    directory_trusted: bool,
}

/// Exclusive writer claim on a database root's shared WAL.
///
/// One shared segment serves every pool, so a single writer process must own
/// it. This takes the same crash-safe `{pid, starttime}` lock the per-pool
/// writer lock uses, at `<db_root>/.mtxdb.wal.lock`, and releases it on drop
/// (see `ShardPool::acquire_writer_lock` for the staleness contract). The
/// holder opens the segment with [`Journal::open_shared`], builds one
/// [`JournalCoordinator`], and attaches each pool with
/// `PackfileStorage::enable_shared_journal`.
pub struct SharedWalLock {
    _lock: crate::shard::WriterLock,
}

impl SharedWalLock {
    /// Acquire the root shared-WAL writer lock.
    ///
    /// # Errors
    /// Returns `WouldBlock` if another live writer holds it, or an I/O error
    /// if the lock file cannot be created.
    pub fn acquire(db_root: impl AsRef<Path>) -> io::Result<Self> {
        let db_root = db_root.as_ref();
        // Require a database root.
        if !crate::layout::is_database_root(db_root)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} is not an mtxdb database root (missing {}); initialize it first",
                    db_root.display(),
                    crate::layout::DB_META_FILENAME
                ),
            ));
        }
        let path = db_root.join(crate::layout::WAL_LOCK_FILENAME);
        Ok(Self {
            _lock: crate::shard::ShardPool::acquire_lock_path(&path)?,
        })
    }
}

#[cfg(test)]
#[path = "test_journal.rs"]
mod tests;
