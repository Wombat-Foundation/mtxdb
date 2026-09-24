//! Append-only, checksummed commit journal used by the packfile WAL path.
//!
//! This module owns the journal's disk framing and durability boundary. It
//! also provides [`JournalCoordinator`](crate::journal::JournalCoordinator),
//! which captures each sync caller's
//! target LSN and releases it only after a durable group covers that target.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::layout::ShardType;

use crc32fast::Hasher;

const FILE_MAGIC: &[u8; 8] = b"MTXWAL01";
const GROUP_MAGIC: &[u8; 4] = b"MWG1";
const GROUP_COMMIT_MAGIC: &[u8; 4] = b"CMIT";

/// On-disk journal format version. The single place that decides frame
/// dialect: callers match on this rather than comparing raw version numbers,
/// so a future bump only has to add a variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JournalVersion {
    /// Per-pool segment. Mutation frames are untagged and the flag byte must be
    /// zero. Bumped to 2 when the header gained the base sequence/LSN a rotated
    /// segment needs to keep numbering global across a rewrite.
    V2,
    /// Shared multi-pool segment. Every mutation frame carries a mandatory pool
    /// tag in the flag byte, so one physical WAL can carry the state,
    /// event-DAG, and auth-chain pools and recovery can route each frame back
    /// to its pool.
    V3PoolTagged,
}

impl JournalVersion {
    /// Version a freshly created per-pool segment uses.
    const fn per_pool() -> Self {
        Self::V2
    }

    /// Version a freshly created shared (multi-pool) segment uses.
    const fn shared() -> Self {
        Self::V3PoolTagged
    }

    const fn as_u32(self) -> u32 {
        match self {
            Self::V2 => 2,
            Self::V3PoolTagged => 3,
        }
    }

    fn from_u32(value: u32) -> io::Result<Self> {
        match value {
            2 => Ok(Self::V2),
            3 => Ok(Self::V3PoolTagged),
            _ => Err(invalid_data("unsupported journal version")),
        }
    }

    /// Whether every mutation frame in this segment must carry a pool tag.
    const fn is_pool_tagged(self) -> bool {
        matches!(self, Self::V3PoolTagged)
    }
}

// File header field byte ranges. `FILE_HEADER_LEN` is the sum of the fields.
const FH_MAGIC: std::ops::Range<usize> = 0..8;
const FH_VERSION: std::ops::Range<usize> = 8..12;
const FH_BASE_SEQUENCE: std::ops::Range<usize> = 12..20;
const FH_BASE_LSN: std::ops::Range<usize> = 20..28;
const FH_CRC: std::ops::Range<usize> = 28..32;
/// Header bytes covered by the trailing CRC (everything before it).
const FH_CRC_COVERED: std::ops::Range<usize> = 0..FH_CRC.start;
const FILE_HEADER_LEN: usize = FH_CRC.end;

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
/// Pool tag (version 3) or reserved zero (version 2).
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
    /// Remove a collection. This is a journal operation because a replayed
    /// put must not resurrect a collection deleted by a later committed LSN.
    DeleteCollection {
        /// Collection to remove.
        collection_id: [u8; 16],
    },
}

/// Upper bound for one SQL transaction's staged journal payloads.
///
/// Kept below the journal's 256 MiB group limit to leave room for framing and
/// other transactions sharing the process.
pub const MAX_TXN_STAGE_BYTES: usize = 64 << 20;

/// Lifecycle of a transaction's staged journal mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStageState {
    /// The SQL transaction attempt is active and may add mutations.
    Active,
    /// The SQL attempt failed; retained callbacks must not publish its data.
    Discarded,
    /// Every per-pool group was appended successfully.
    Published,
}

#[derive(Debug)]
struct TxnStageData {
    pools: [Vec<Mutation>; 3],
    bytes: usize,
    /// Successful pool appends. Retrying a callback after a partial error
    /// resumes at the failed pool instead of duplicating earlier groups.
    appended: [bool; 3],
}

/// Transaction-local **journal** publication buffer -- not a storage
/// transaction.
///
/// Journal entries for a SQL transaction are buffered here and published from
/// its post-commit callback. Call [`Self::discard`] from the transaction's
/// error callback; that is required because Synapse retains after-callbacks
/// across retry attempts.
///
/// # Not a rollback
///
/// [`Self::discard`] drops only the buffered journal mutations. It does not
/// touch packs or the live index, because this buffer never writes them; the
/// deferred-publication redesign (prepare/commit) will route the pack/index
/// mutation through this buffer's commit path. Until that lands, no
/// production caller stages through here.
pub struct TxnStage {
    state: std::sync::atomic::AtomicU8,
    data: Mutex<TxnStageData>,
}

impl Default for TxnStage {
    fn default() -> Self {
        Self::new()
    }
}

impl TxnStage {
    const ACTIVE: u8 = 0;
    const DISCARDED: u8 = 1;
    const PUBLISHED: u8 = 2;

    /// Create an empty stage for one transaction attempt.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(Self::ACTIVE),
            data: Mutex::new(TxnStageData {
                pools: std::array::from_fn(|_| Vec::new()),
                bytes: 0,
                appended: [false; 3],
            }),
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> TxnStageState {
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED => TxnStageState::Discarded,
            Self::PUBLISHED => TxnStageState::Published,
            _ => TxnStageState::Active,
        }
    }

    /// Discard an active attempt. Safe to call more than once.
    ///
    /// See the "Not a rollback" section on [`TxnStage`]: this drops only the
    /// buffered journal mutations and never touches packs or the live index.
    pub fn discard(&self) {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) == Self::ACTIVE {
            data.pools.iter_mut().for_each(Vec::clear);
            data.bytes = 0;
            self.state.store(Self::DISCARDED, Ordering::Release);
        }
    }

    /// Ensure an estimated batch fits before buffering its journal mutations.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the stage is not active or the estimated
    /// total exceeds [`MAX_TXN_STAGE_BYTES`].
    pub fn ensure_capacity(&self, additional: usize) -> io::Result<()> {
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let data = self.data.lock();
        let total = data.bytes.checked_add(additional).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > MAX_TXN_STAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction journal stage exceeds its 64 MiB limit",
            ));
        }
        Ok(())
    }

    /// Add an immutable put to this attempt's pool-ordered journal batch.
    ///
    /// # Errors
    /// Returns `InvalidInput` if adding the payload would exceed the stage
    /// limit, or another error if the stage has already been discarded or
    /// published.
    pub fn stage_put(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        payload: Vec<u8>,
    ) -> io::Result<()> {
        let charge = payload.len().saturating_add(64);
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > MAX_TXN_STAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction journal stage exceeds its 64 MiB limit",
            ));
        }
        data.pools[pool_index(pool)].push(Mutation::Put {
            collection_id,
            node_id,
            payload,
        });
        data.bytes = total;
        Ok(())
    }

    /// Add a batch of immutable puts atomically.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the batch would exceed the stage limit, or
    /// another error if the stage has already been discarded or published.
    pub fn stage_puts(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        entries: &[([u8; 16], Vec<u8>)],
    ) -> io::Result<()> {
        let charge = entries
            .iter()
            .try_fold(0usize, |total, (_, payload)| {
                total.checked_add(payload.len().saturating_add(64))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "transaction stage size overflow",
                )
            })?;
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > MAX_TXN_STAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction journal stage exceeds its 64 MiB limit",
            ));
        }
        let pool_mutations = &mut data.pools[pool_index(pool)];
        pool_mutations.extend(entries.iter().map(|(node_id, payload)| Mutation::Put {
            collection_id,
            node_id: *node_id,
            payload: payload.clone(),
        }));
        data.bytes = total;
        Ok(())
    }

    /// Add a collection delete to this attempt. Callers must not eagerly
    /// mutate the live index for a delete that can still roll back.
    ///
    /// # Errors
    /// Returns `InvalidInput` if adding the mutation would exceed the stage
    /// limit, or another error if the stage has already been discarded or
    /// published.
    pub fn stage_delete_collection(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > MAX_TXN_STAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction journal stage exceeds its 64 MiB limit",
            ));
        }
        data.pools[pool_index(pool)].push(Mutation::DeleteCollection { collection_id });
        data.bytes = total;
        Ok(())
    }

    /// Append staged pool groups in dependency order: auth-chain, event-DAG,
    /// then state. Does not fsync; the ordinary coalesced sync remains the
    /// durability boundary. Repeated calls are safe, including after a
    /// partial error.
    ///
    /// # Errors
    /// Returns an error if a staged pool has no coordinator or its journal
    /// cannot append the group's complete framing and trailer.
    pub fn publish(
        &self,
        auth_chain: Option<&JournalCoordinator>,
        event_dag: Option<&JournalCoordinator>,
        state: Option<&JournalCoordinator>,
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED | Self::PUBLISHED => return Ok(()),
            _ => {}
        }
        let coordinators = [auth_chain, event_dag, state];
        if coordinators.iter().all(Option::is_none) {
            // Journaling is disabled process-wide, so there is no journal to
            // publish into.
            self.state.store(Self::PUBLISHED, Ordering::Release);
            return Ok(());
        }
        let pools = [ShardType::AuthChain, ShardType::EventDag, ShardType::State];
        for (ordered_index, pool) in pools.into_iter().enumerate() {
            let index = pool_index(pool);
            if data.appended[index] || data.pools[index].is_empty() {
                data.appended[index] = true;
                continue;
            }
            let coordinator = coordinators[ordered_index].ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotConnected, "staged pool has no journal")
            })?;
            coordinator.append_pending(&data.pools[index])?;
            data.appended[index] = true;
        }
        self.state.store(Self::PUBLISHED, Ordering::Release);
        Ok(())
    }
}

const fn pool_index(pool: ShardType) -> usize {
    match pool {
        ShardType::State => 0,
        ShardType::EventDag => 1,
        ShardType::AuthChain => 2,
    }
}

/// Wire code for a pool tag in a pool-tagged (`FILE_VERSION_POOL_TAGGED`)
/// mutation frame's previously reserved flag byte.
///
/// `0` is reserved for "untagged", which only a version-2 segment may contain;
/// a version-3 frame must carry `1..=3`.
const fn pool_tag(pool: ShardType) -> u8 {
    match pool {
        ShardType::State => 1,
        ShardType::EventDag => 2,
        ShardType::AuthChain => 3,
    }
}

/// Inverse of [`pool_tag`]. `0` (untagged) maps to `None`; any other
/// out-of-range code is rejected by the frame decoder.
const fn pool_from_tag(tag: u8) -> Option<ShardType> {
    match tag {
        1 => Some(ShardType::State),
        2 => Some(ShardType::EventDag),
        3 => Some(ShardType::AuthChain),
        _ => None,
    }
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
    /// Pool the frame belongs to. `None` for a version-2 (per-pool) segment;
    /// `Some` for every frame of a pool-tagged version-3 segment.
    pub pool: Option<ShardType>,
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
    /// Time spent waiting for the single-writer journal mutex.
    pub journal_lock_wait: std::time::Duration,
    /// Time spent waiting for the pending-mutation queue mutex.
    pub journal_pending_wait: std::time::Duration,
    /// Time spent appending and encoding the group, excluding fsync.
    pub journal_append: std::time::Duration,
    /// Time spent making the journal file durable.
    pub journal_fsync: std::time::Duration,
    /// Number of mutations included in a newly appended group.
    pub journal_records: u64,
    /// Bytes included in a newly appended group, including framing.
    pub journal_bytes: u64,
    /// Whether this request had to wait for the journal mutex.
    pub journal_waiter: bool,
    /// Whether this request was already covered by another durable request.
    pub journal_coalesced: bool,
    /// Number of journal sync callers active when this request entered.
    pub journal_in_flight: u64,
}

struct SyncInFlightGuard<'a>(&'a AtomicU64);

impl Drop for SyncInFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
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
}

/// A single-writer journal file. [`Self::append_group`] writes a complete
/// group and its commit trailer; [`Self::make_durable`] fsyncs it;
/// [`Self::commit_group`] does both. Calls are serialized by the caller's
/// coordinator (or by `&mut self`).
pub struct Journal {
    path: PathBuf,
    file: File,
    /// On-disk format version of this segment.
    version: JournalVersion,
    next_sequence: u64,
    next_lsn: u64,
    poisoned: bool,
}

/// An fsync at least this slow is reported on stderr when it happens.
const SLOW_FSYNC_WARN: std::time::Duration = std::time::Duration::from_secs(1);

/// Serializes mutation publication and durable commits for one journal.
///
/// Mutations are assigned LSNs under a short queue lock. A sync caller captures
/// the published LSN, detaches the covered mutations, then commits them while
/// holding the journal lock. Publishers can queue later mutations during the
/// fsync, and concurrent sync callers recheck the committed LSN after taking
/// the journal lock.
pub struct JournalCoordinator {
    journal: Mutex<Journal>,
    path: PathBuf,
    pending: Mutex<Vec<(u64, Option<ShardType>, Mutation)>>,
    /// Next LSN to assign. Advanced under `pending`, independently of journal
    /// I/O, so a publish never blocks behind a sync's fsync.
    next_lsn: AtomicU64,
    published_lsn: AtomicU64,
    /// Highest LSN whose group is complete (trailer appended) but not
    /// necessarily fsynced. Advanced between [`Journal::append_group`] and
    /// [`Journal::make_durable`] so a read-only overlay can observe a
    /// committed-but-unflushed group.
    visible_lsn: AtomicU64,
    committed_lsn: AtomicU64,
    /// Optional shared, cross-pool group-sequence allocator. When present,
    /// each committed group draws its sequence from here instead of the
    /// segment's own counter, so groups across several pool segments share one
    /// global order. See [`Self::with_shared_sequence`].
    sequence: Option<Arc<AtomicU64>>,
    /// Mirrors the journal's poison bit, so `publish` can reject without
    /// taking the `journal` mutex (which a sync holds across its fsync).
    poisoned: AtomicBool,
    /// Number of sync requests entering the coordinator.
    sync_calls: AtomicU64,
    /// Number of sync requests that found the journal mutex occupied.
    journal_waiters: AtomicU64,
    /// Number of requests covered without appending or fsyncing themselves.
    coalesced_syncs: AtomicU64,
    sync_in_flight: AtomicU64,
}

impl JournalCoordinator {
    /// Build a coordinator from an opened journal and its recovery scan.
    #[must_use]
    pub fn new(journal: Journal, scan: &Scan) -> Self {
        let committed_lsn = scan.groups.last().map_or(0, |group| group.last_lsn);
        let next_lsn = journal.next_lsn;
        let path = journal.path.clone();
        Self {
            journal: Mutex::new(journal),
            path,
            pending: Mutex::new(Vec::new()),
            next_lsn: AtomicU64::new(next_lsn),
            published_lsn: AtomicU64::new(committed_lsn),
            visible_lsn: AtomicU64::new(committed_lsn),
            committed_lsn: AtomicU64::new(committed_lsn),
            sequence: None,
            poisoned: AtomicBool::new(false),
            sync_calls: AtomicU64::new(0),
            journal_waiters: AtomicU64::new(0),
            coalesced_syncs: AtomicU64::new(0),
            sync_in_flight: AtomicU64::new(0),
        }
    }

    /// Build a coordinator whose groups draw their sequence from a shared,
    /// cross-pool counter instead of this segment's own numbering.
    ///
    /// The caller must initialize `sequence` above every participating
    /// segment's recovered next sequence (see [`Journal::next_sequence`]), so
    /// the first allocation is larger than any sequence already on disk. Gaps
    /// in a single segment's group sequence are then expected; recovery
    /// permits them.
    #[must_use]
    pub fn with_shared_sequence(journal: Journal, scan: &Scan, sequence: Arc<AtomicU64>) -> Self {
        let mut coordinator = Self::new(journal, scan);
        coordinator.sequence = Some(sequence);
        coordinator
    }

    /// Highest durably committed LSN. Everything at or below this is on disk.
    #[must_use]
    pub fn committed_lsn(&self) -> u64 {
        self.committed_lsn.load(Ordering::Acquire)
    }

    /// Path of the journal segment used by this coordinator.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// Highest LSN whose journal group is complete (commit trailer appended),
    /// whether or not it has been fsynced.
    ///
    /// A read-only overlay may observe everything at or below this. Durability
    /// is only promised by [`Self::committed_lsn`]: a crash before the fsync
    /// may lose a visible-but-uncommitted group.
    #[must_use]
    pub fn visible_lsn(&self) -> u64 {
        self.visible_lsn.load(Ordering::Acquire)
    }

    /// Highest published (assigned) LSN, committed or not.
    #[must_use]
    pub fn published_lsn(&self) -> u64 {
        self.published_lsn.load(Ordering::Acquire)
    }

    /// Assign an LSN, publish the mutation to the caller's live overlay, and
    /// queue it for the next covering durable group.
    ///
    /// The callback runs while publication is serialized and before the LSN
    /// becomes visible to sync callers. It must not call back into this
    /// coordinator. Keeping overlay publication in this critical section
    /// prevents a successful sync from racing ahead of a not-yet-visible put.
    ///
    /// # Errors
    /// Returns an error if the journal has been poisoned by a failed commit or
    /// if its LSN space is exhausted.
    pub fn publish(
        &self,
        mutation: Mutation,
        publish_overlay: impl FnOnce(u64),
    ) -> io::Result<u64> {
        self.publish_inner(None, mutation, publish_overlay)
    }

    /// Like [`Self::publish`], but records the frame's pool tag for a shared
    /// pool-tagged segment. The tag is encoded into the frame and used by
    /// recovery to route the mutation back to its pool.
    ///
    /// # Errors
    /// Same as [`Self::publish`].
    pub fn publish_tagged(
        &self,
        pool: ShardType,
        mutation: Mutation,
        publish_overlay: impl FnOnce(u64),
    ) -> io::Result<u64> {
        self.publish_inner(Some(pool), mutation, publish_overlay)
    }

    fn publish_inner(
        &self,
        pool: Option<ShardType>,
        mutation: Mutation,
        publish_overlay: impl FnOnce(u64),
    ) -> io::Result<u64> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        let mut pending = self.pending.lock();
        let lsn = self.next_lsn.load(Ordering::Relaxed);
        let next_lsn = lsn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
        publish_overlay(lsn);
        pending.push((lsn, pool, mutation));
        self.next_lsn.store(next_lsn, Ordering::Relaxed);
        self.published_lsn.store(lsn, Ordering::Release);
        Ok(lsn)
    }

    /// Capture the latest fully published mutation as this sync caller's
    /// acknowledgement boundary.
    #[must_use]
    pub fn capture_sync_target(&self) -> u64 {
        self.published_lsn.load(Ordering::Acquire)
    }

    /// Durably commit pending mutations through `target_lsn`.
    ///
    /// Calls with the same or an earlier target are covered by an already
    /// completed group. Later published mutations are left for a later group.
    /// The journal mutex serializes the disk commit; concurrent sync callers
    /// wait for it and then recheck the committed LSN before deciding whether
    /// to write.
    ///
    /// # Errors
    /// Returns an error if the target was never published, the journal is
    /// poisoned, or writing/syncing the covering group fails.
    pub fn sync_through(&self, target_lsn: u64) -> io::Result<Option<CommitReceipt>> {
        self.sync_through_timed(target_lsn)
            .map(|(receipt, _)| receipt)
    }

    /// Durably commit pending mutations and report where the time went.
    ///
    /// # Errors
    /// Returns an I/O error if the target is invalid, the journal is poisoned,
    /// or appending/fsyncing the group fails.
    pub fn sync_through_timed(
        &self,
        target_lsn: u64,
    ) -> io::Result<(Option<CommitReceipt>, JournalSyncTimings)> {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        let mut timings = JournalSyncTimings::default();
        let in_flight = self
            .sync_in_flight
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        timings.journal_in_flight = in_flight;
        let _in_flight_guard = SyncInFlightGuard(&self.sync_in_flight);
        if target_lsn == 0 {
            timings.journal_coalesced = true;
            self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
            return Ok((None, timings));
        }
        if target_lsn > self.published_lsn.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sync target has not been published",
            ));
        }
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            timings.journal_coalesced = true;
            self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
            return Ok((None, timings));
        }
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }

        let lock_started = std::time::Instant::now();
        let mut journal = if let Some(guard) = self.journal.try_lock() {
            guard
        } else {
            timings.journal_waiter = true;
            self.journal_waiters.fetch_add(1, Ordering::Relaxed);
            self.journal.lock()
        };
        timings.journal_lock_wait = lock_started.elapsed();
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            timings.journal_coalesced = true;
            self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
            return Ok((None, timings));
        }
        // Transaction-staged groups are already complete in the segment, so
        // the coalesced sync only needs to make their bytes durable.
        if target_lsn <= self.visible_lsn.load(Ordering::Acquire) {
            let fsync_started = std::time::Instant::now();
            if let Err(error) = journal.make_durable() {
                self.poisoned.store(true, Ordering::Release);
                return Err(error);
            }
            timings.journal_fsync = fsync_started.elapsed();
            self.warn_if_slow_fsync(journal.path.as_path(), target_lsn, &timings);
            self.committed_lsn.store(target_lsn, Ordering::Release);
            return Ok((None, timings));
        }
        let batch = {
            let pending_started = std::time::Instant::now();
            let mut pending = self.pending.lock();
            timings.journal_pending_wait = pending_started.elapsed();
            let covered_count = pending
                .iter()
                .take_while(|(lsn, _, _)| *lsn <= target_lsn)
                .count();
            if covered_count == 0 {
                // Another sync may have committed and drained this target
                // after our first check but before we acquired the journal
                // lock. In that case its durable group covers this caller.
                if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
                    timings.journal_coalesced = true;
                    self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
                    return Ok((None, timings));
                }
                return Err(io::Error::other(
                    "published sync target has no pending journal mutations",
                ));
            }
            let first_lsn = pending[0].0;
            let last_lsn = covered_count
                .checked_sub(1)
                .and_then(|last_index| pending.get(last_index))
                .map(|(lsn, _, _)| *lsn)
                .ok_or_else(|| io::Error::other("pending journal batch is incomplete"))?;
            if first_lsn != journal.next_lsn || last_lsn != target_lsn {
                return Err(io::Error::other(
                    "journal pending LSN sequence does not cover sync target",
                ));
            }
            pending.drain(..covered_count).collect::<Vec<_>>()
        };
        let record_count = u64::try_from(batch.len()).unwrap_or(u64::MAX);
        let result = self.append_and_sync_batch(&mut journal, batch, target_lsn, &mut timings);
        timings.journal_records = record_count;
        if let Ok(Some(receipt)) = &result {
            timings.journal_bytes = receipt.bytes_written;
        }
        result.map(|receipt| (receipt, timings))
    }

    /// Report an fsync slower than [`SLOW_FSYNC_WARN`] as it happens, with the
    /// lock wait and waiter count needed to tell contention from disk latency.
    fn warn_if_slow_fsync(&self, path: &Path, target_lsn: u64, timings: &JournalSyncTimings) {
        if timings.journal_fsync >= SLOW_FSYNC_WARN {
            eprintln!(
                "mtxdb: slow WAL fsync {}ms (through lsn {target_lsn}, lock wait {}ms, append {}ms, {} records, {} bytes, in-flight {}, waiters {}, {})",
                timings.journal_fsync.as_millis(),
                timings.journal_lock_wait.as_millis(),
                timings.journal_append.as_millis(),
                timings.journal_records,
                timings.journal_bytes,
                timings.journal_in_flight,
                self.journal_waiters.load(Ordering::Relaxed),
                path.display(),
            );
        }
    }

    fn append_and_sync_batch(
        &self,
        journal: &mut Journal,
        batch: Vec<(u64, Option<ShardType>, Mutation)>,
        target_lsn: u64,
        timings: &mut JournalSyncTimings,
    ) -> io::Result<Option<CommitReceipt>> {
        let mutations: Vec<(Option<ShardType>, Mutation)> = batch
            .iter()
            .map(|(_, pool, mutation)| (*pool, mutation.clone()))
            .collect();
        let sequence = self
            .sequence
            .as_ref()
            .map(|counter| counter.fetch_add(1, Ordering::Relaxed));
        let append_started = std::time::Instant::now();
        let receipt = match journal.append_group_tagged_with_sequence(&mutations, sequence) {
            Ok(receipt) => receipt,
            Err(error) => {
                if journal.poisoned {
                    self.poisoned.store(true, Ordering::Release);
                } else {
                    let mut pending = self.pending.lock();
                    pending.splice(0..0, batch);
                }
                return Err(error);
            }
        };
        timings.journal_append = append_started.elapsed();
        // The group is complete and readable by a read-only overlay, but not
        // yet durable. Publish the visibility boundary before the fsync so
        // workers can observe it. A crash before `make_durable` may lose it,
        // which is safe: an unfsynced group is never acknowledged.
        self.visible_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        let fsync_started = std::time::Instant::now();
        if let Err(error) = journal.make_durable() {
            self.poisoned.store(true, Ordering::Release);
            return Err(error);
        }
        timings.journal_fsync = fsync_started.elapsed();
        self.warn_if_slow_fsync(journal.path.as_path(), target_lsn, timings);
        // The group is durable, so record its extent before checking the
        // receipt. This prevents a retry from re-appending committed entries.
        self.committed_lsn
            .store(receipt.last_lsn, Ordering::Release);
        let committed_count = usize::try_from(
            receipt
                .last_lsn
                .saturating_sub(receipt.first_lsn)
                .saturating_add(1),
        )
        .unwrap_or(batch.len())
        .min(batch.len());
        if committed_count < batch.len() {
            let mut pending = self.pending.lock();
            pending.splice(0..0, batch.into_iter().skip(committed_count));
        }
        if receipt.last_lsn != target_lsn || committed_count != mutations.len() {
            return Err(io::Error::other(
                "durable journal group did not cover requested sync target",
            ));
        }
        Ok(Some(receipt))
    }

    /// Append a complete transaction group without fsyncing it.
    ///
    /// Unlike [`Self::publish`], this does not add mutations to the legacy
    /// queue: it appends queued legacy mutations first, followed by the staged
    /// transaction mutations, in one complete group and then advances the
    /// visible boundary. This preserves assigned LSN order and avoids making a
    /// post-commit callback fail merely because background/legacy writes are
    /// queued in the same pool.
    ///
    /// # Errors
    /// Returns an error if queued LSNs are inconsistent, the journal is
    /// poisoned, or the append fails. A partial append poisons the underlying
    /// journal; subsequent publication is rejected until reopen/recovery.
    pub fn append_pending(&self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        if mutations.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot append an empty transaction group",
            ));
        }
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed append",
            ));
        }
        let mut journal = self.journal.lock();
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed append",
            ));
        }
        // Hold the queue lock through the append so a concurrent legacy
        // publisher cannot assign an LSN between the queued and staged parts.
        let mut pending = self.pending.lock();
        let mut group_mutations: Vec<(Option<ShardType>, Mutation)> =
            Vec::with_capacity(pending.len().saturating_add(mutations.len()));
        for (offset, (lsn, pool, mutation)) in pending.iter().enumerate() {
            let expected = journal
                .next_lsn
                .checked_add(u64::try_from(offset).unwrap_or(u64::MAX))
                .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
            if *lsn != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "legacy journal queue is not contiguous with the journal tail",
                ));
            }
            group_mutations.push((*pool, mutation.clone()));
        }
        group_mutations.extend(mutations.iter().cloned().map(|mutation| (None, mutation)));
        let expected_first_lsn = journal.next_lsn;
        let expected_count = u64::try_from(group_mutations.len()).unwrap_or(u64::MAX);
        let sequence = self
            .sequence
            .as_ref()
            .map(|counter| counter.fetch_add(1, Ordering::Relaxed));
        let receipt = match journal.append_group_tagged_with_sequence(&group_mutations, sequence) {
            Ok(receipt) => receipt,
            Err(error) => {
                if journal.poisoned {
                    self.poisoned.store(true, Ordering::Release);
                }
                return Err(error);
            }
        };
        debug_assert_eq!(
            receipt.first_lsn, expected_first_lsn,
            "appended group must start at the journal tail"
        );
        debug_assert_eq!(
            receipt
                .last_lsn
                .saturating_sub(receipt.first_lsn)
                .saturating_add(1),
            expected_count,
            "appended group must cover every queued and staged mutation"
        );
        let next_lsn = receipt
            .last_lsn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
        pending.clear();
        self.next_lsn.store(next_lsn, Ordering::Relaxed);
        self.published_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        self.visible_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        drop(pending);
        Ok(receipt)
    }

    /// Capture the current boundary and wait for a durable group covering it.
    ///
    /// # Errors
    /// Returns an error if committing the captured boundary fails.
    pub fn sync(&self) -> io::Result<Option<CommitReceipt>> {
        let target = self.capture_sync_target();
        self.sync_through(target)
    }

    /// Compact the segment, dropping committed groups at or below
    /// `covered_lsn` while preserving any newer groups.
    ///
    /// Intended to run after a checkpoint durably records that LSN as applied,
    /// so it reclaims only data the packfiles already represent. Mutations that
    /// are published but not yet committed are untouched: they live in memory
    /// until the next [`Self::sync_through`]. Numbering is preserved across the
    /// rewrite.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned or the segment cannot be
    /// rewritten.
    pub fn reclaim_through(&self, covered_lsn: u64) -> io::Result<Reclaim> {
        let mut journal = self.journal.lock();
        journal.reclaim_through(covered_lsn)
    }
}

impl Journal {
    /// Scan an existing journal without opening it for writing or repairing a
    /// torn tail. A read-only worker uses this to observe only complete,
    /// committed groups while the writer may still be appending or reclaiming
    /// the segment.
    ///
    /// An incomplete final group is reported in [`Scan::truncated_tail`] and
    /// excluded from `groups`; the file is left byte-for-byte unchanged. A
    /// malformed committed group remains an error.
    ///
    /// A missing file or one shorter than the file header is treated as an
    /// empty segment. This permits a read-only worker to start before the
    /// writer has created the journal, without creating or repairing it.
    ///
    /// # Errors
    /// Returns `io::Error` if the file is unreadable, a complete header is
    /// invalid, the segment is oversized, or a committed group fails
    /// validation.
    pub fn scan_read_only(path: impl AsRef<Path>) -> io::Result<Scan> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Scan::empty()),
            Err(error) => return Err(error),
        };
        if bytes.len() < FILE_HEADER_LEN {
            return Ok(Scan::empty());
        }
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_SEGMENT_LEN {
            return Err(invalid_data("journal segment exceeds the 256 MiB limit"));
        }
        let (version, base_sequence, base_lsn) = validate_file_header(&bytes)?;
        scan_bytes(&bytes, base_sequence, base_lsn, version)
    }

    /// Scan only the committed groups appended at or after `start`, a group
    /// boundary returned by an earlier scan's `valid_len`.
    ///
    /// Reads only the tail bytes, so a reader that already applied the earlier
    /// groups does not re-decode them. `expected_lsn` is the next LSN the
    /// caller expects (its highest applied LSN plus one); a mismatch means the
    /// segment was replaced or rotated, and the caller should rebuild from a
    /// full [`Self::scan_read_only`]. Never repairs or creates the file.
    ///
    /// # Errors
    /// Returns `io::Error` if the tail is unreadable or a group fails
    /// validation. A missing file yields an empty scan.
    // This intentionally seeks to `start` and reads only the appended tail;
    // `fs::read` would decode the whole segment on every incremental refresh.
    #[allow(clippy::verbose_file_reads)]
    pub fn scan_read_only_from(
        path: impl AsRef<Path>,
        start: u64,
        expected_lsn: u64,
    ) -> io::Result<Scan> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Scan::empty()),
            Err(error) => return Err(error),
        };
        let len = file.metadata()?.len();
        if len < u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX) {
            return Ok(Scan::empty());
        }
        file.seek(SeekFrom::Start(0))?;
        let mut header = vec![0; FILE_HEADER_LEN];
        file.read_exact(&mut header)?;
        let (version, _, base_lsn) = validate_file_header(&header)?;
        if start >= len {
            return Ok(Scan {
                groups: Vec::new(),
                valid_len: start.min(len),
                truncated_tail: false,
                base_lsn,
            });
        }
        file.seek(SeekFrom::Start(start))?;
        let mut tail = Vec::new();
        file.read_to_end(&mut tail)?;
        scan_groups_from(&tail, start, 0, expected_lsn, base_lsn, version)
    }

    /// Open a journal and recover its committed groups.
    ///
    /// A segment shorter than the file header (a fresh file, or a crash while
    /// the header was being written) is reset and re-initialised: no group can
    /// exist without a fully synced header, so there is no acknowledged data
    /// to lose. An incomplete final group is truncated. A malformed
    /// *committed* group fails open and is never silently skipped.
    ///
    /// # Errors
    /// Returns `io::Error` if the segment cannot be created or read, if a
    /// data-bearing segment's header is invalid, if any committed group fails
    /// validation, or if the segment exceeds `MAX_SEGMENT_LEN`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<(Self, Scan)> {
        Self::open_versioned(path, JournalVersion::per_pool())
    }

    /// Open a shared (multi-pool) journal segment, or create one if the file is
    /// absent. The segment is created as [`JournalVersion::V3PoolTagged`], so
    /// every mutation frame appended through it must carry a pool tag.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidData` if an existing segment is not
    /// the pool-tagged version.
    pub fn open_shared(path: impl AsRef<Path>) -> io::Result<(Self, Scan)> {
        Self::open_versioned(path, JournalVersion::shared())
    }

    /// Open a journal whose on-disk version must be `version`, creating it with
    /// that version when the file is absent or shorter than the header.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidData` if an existing complete
    /// segment was written by a different version.
    fn open_versioned(path: impl AsRef<Path>, version: JournalVersion) -> io::Result<(Self, Scan)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        let len = file.metadata()?.len();
        if len < FILE_HEADER_LEN as u64 {
            // Empty (fresh) or a crash while writing the header: no committed
            // group can exist yet, so reset and re-initialise rather than
            // failing open forever on a partial header.
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            write_file_header(&mut file, version, 1, 1)?;
            file.sync_all()?;
            sync_parent_dir(&path)?;
        }

        let bytes = fs::read(&path)?;
        let (found_version, base_sequence, base_lsn) = validate_file_header(&bytes)?;
        if found_version != version {
            return Err(invalid_data(
                "journal segment version does not match open mode",
            ));
        }
        if bytes.len() as u64 > MAX_SEGMENT_LEN {
            return Err(invalid_data("journal segment exceeds the 256 MiB limit"));
        }
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, version)?;
        if scan.truncated_tail {
            file.set_len(scan.valid_len)?;
        }
        file.seek(SeekFrom::Start(scan.valid_len))?;
        let next_sequence = scan
            .groups
            .last()
            .map_or(base_sequence, |group| group.sequence.saturating_add(1));
        let next_lsn = scan
            .groups
            .last()
            .map_or(base_lsn, |group| group.last_lsn.saturating_add(1));

        Ok((
            Self {
                path,
                file,
                version,
                next_sequence,
                next_lsn,
                poisoned: false,
            },
            scan,
        ))
    }

    /// Append a non-empty group and its commit trailer, without fsyncing.
    ///
    /// Each mutation receives its own monotonic LSN. The group is complete and
    /// readable by [`Self::scan_read_only`] as soon as this returns, so a
    /// caller can publish a visibility boundary before [`Self::make_durable`]
    /// fsyncs it. If writing fails, this handle is poisoned: the caller must
    /// reopen and rescan before attempting another commit.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned, the group is empty or
    /// oversized, the LSN/sequence space is exhausted, the segment is full
    /// (`WouldBlock`; the caller must drain/rotate), or the write fails.
    pub fn append_group(&mut self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        self.append_group_with_sequence(mutations, None)
    }

    /// Append a group using `sequence` when given, or this segment's own next
    /// sequence otherwise.
    ///
    /// A shared sequence lets several pools' segments be ordered by one global
    /// group sequence; a gap between a segment's consecutive groups is then
    /// expected, and recovery permits it (integrity still comes from the group
    /// header CRC and commit trailer, which both cover the sequence).
    ///
    /// # Errors
    /// Same as [`Self::append_group`], plus `InvalidInput` if `sequence` is
    /// behind this segment's next sequence, or if this is a pool-tagged segment
    /// (which requires [`Self::append_group_tagged_with_sequence`]).
    pub fn append_group_with_sequence(
        &mut self,
        mutations: &[Mutation],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        let untagged = mutations
            .iter()
            .cloned()
            .map(|mutation| (None, mutation))
            .collect::<Vec<_>>();
        self.append_group_inner(&untagged, sequence)
    }

    /// Append a pool-tagged group to a shared, pool-tagged segment.
    ///
    /// # Errors
    /// Same as [`Self::append_group_with_sequence`], plus `InvalidInput` if
    /// this segment is not pool-tagged.
    pub fn append_group_tagged_with_sequence(
        &mut self,
        mutations: &[(Option<ShardType>, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        self.append_group_inner(mutations, sequence)
    }

    /// Shared implementation for a group of mutations, each paired with its
    /// optional pool tag. The tag must match the segment's dialect (see
    /// [`JournalVersion::is_pool_tagged`]).
    fn append_group_inner(
        &mut self,
        mutations: &[(Option<ShardType>, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        if mutations.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot commit an empty journal group",
            ));
        }

        let record_count = u32::try_from(mutations.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "too many journal mutations")
        })?;
        let first_lsn = self.next_lsn;
        let last_lsn = first_lsn
            .checked_add(u64::from(record_count).saturating_sub(1))
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
        let sequence = sequence.unwrap_or(self.next_sequence);
        if sequence < self.next_sequence {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal group sequence would move backwards",
            ));
        }

        let mut payload = Vec::new();
        for (index, (pool, mutation)) in mutations.iter().enumerate() {
            let lsn = first_lsn
                .checked_add(u64::try_from(index).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mutation index exceeds u64")
                })?)
                .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
            encode_mutation(lsn, *pool, self.version, mutation, &mut payload)?;
            if u64::try_from(payload.len()).unwrap_or(u64::MAX) > MAX_GROUP_LEN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "journal group exceeds the 256 MiB format limit",
                ));
            }
        }
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "group too large"))?;
        if payload_len > MAX_GROUP_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal group exceeds the 256 MiB segment limit",
            ));
        }

        let header = encode_group_header(sequence, first_lsn, last_lsn, payload_len, record_count);
        let mut group_checksum = Hasher::new();
        group_checksum.update(&header);
        group_checksum.update(&payload);
        let trailer = encode_group_trailer(sequence, group_checksum.finalize());

        let group_size = u64::try_from(
            header
                .len()
                .saturating_add(payload.len())
                .saturating_add(trailer.len()),
        )
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "group too large"))?;
        let segment_size = self.file.metadata()?.len();
        if segment_size.saturating_add(group_size) > MAX_SEGMENT_LEN {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "journal segment is full; drain/rotate before accepting more writes",
            ));
        }
        let next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal group sequence exhausted"))?;
        let next_lsn = last_lsn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;

        let write_result = (|| -> io::Result<()> {
            self.file.write_all(&header)?;
            self.file.write_all(&payload)?;
            self.file.write_all(&trailer)?;
            Ok(())
        })();
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(error);
        }

        self.next_sequence = next_sequence;
        self.next_lsn = next_lsn;

        Ok(CommitReceipt {
            sequence,
            first_lsn,
            last_lsn,
            bytes_written: group_size,
        })
    }

    /// Fsync the bytes [`Self::append_group`] wrote, making the group durable.
    ///
    /// Separate from [`Self::append_group`] so a caller can publish a
    /// visibility boundary after the group is complete but before the fsync.
    /// On failure this handle is poisoned: the appended group's on-disk state
    /// is uncertain, so the caller must reopen and rescan.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned or the sync fails.
    pub fn make_durable(&mut self) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        if let Err(error) = self.file.sync_all() {
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }

    /// Append a non-empty group and durably sync it.
    ///
    /// Convenience wrapper over [`Self::append_group`] followed by
    /// [`Self::make_durable`] for callers that do not need to publish a
    /// visibility boundary between the two.
    ///
    /// # Errors
    /// Propagates either step's failure. A failure leaves the handle poisoned.
    pub fn commit_group(&mut self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        let receipt = self.append_group(mutations)?;
        self.make_durable()?;
        Ok(receipt)
    }

    /// Path of this journal file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sequence the next group appended without an explicit sequence would use.
    ///
    /// Callers sharing one cross-pool sequence allocator initialize it above
    /// the maximum of these across segments. See
    /// [`JournalCoordinator::with_shared_sequence`].
    #[must_use]
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Compact this segment, dropping committed groups the checkpoint has
    /// already materialized.
    ///
    /// Every group with `last_lsn <= covered_lsn` is removed; the survivors are
    /// re-encoded with their original sequence and LSNs into a temporary file,
    /// durably synced, and atomically renamed over the active path, after which
    /// the parent directory is synced. A crash before the rename leaves the old
    /// segment intact and the temp file inert. Because the header records the
    /// surviving base numbering, the next [`Self::commit_group`] continues past
    /// the last committed LSN rather than restarting at 1.
    ///
    /// A checkpoint that has not advanced past the current segment (or has
    /// already reclaimed it) is a no-op.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned, the segment cannot be
    /// read or re-encoded, or the replacement cannot be synced or renamed.
    pub fn reclaim_through(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        if covered_lsn >= self.next_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot reclaim an LSN that has not been committed",
            ));
        }
        let bytes = fs::read(&self.path)?;
        let (version, base_sequence, base_lsn) = validate_file_header(&bytes)?;
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, version)?;
        let retained = scan
            .groups
            .iter()
            .filter(|group| group.last_lsn > covered_lsn)
            .collect::<Vec<_>>();
        if retained.len() == scan.groups.len() {
            return Ok(Reclaim {
                retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
            });
        }

        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let mut rebuilt = file_header_bytes(self.version, new_base_sequence, new_base_lsn);
        for group in &retained {
            encode_group(group, self.version, &mut rebuilt)?;
        }

        let temp_path = self.path.with_extension("rotate");
        let write_result = (|| -> io::Result<()> {
            let mut temp = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)?;
            temp.write_all(&rebuilt)?;
            temp.sync_all()?;
            drop(temp);
            fs::rename(&temp_path, &self.path)
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }
        // The rename has replaced the path, so from here onward any failure
        // must poison the handle: continuing to append through the old file
        // descriptor would write an unlinked inode and could acknowledge data
        // that is absent from the journal path.
        let replacement = match OpenOptions::new().read(true).write(true).open(&self.path) {
            Ok(file) => file,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        self.file = replacement;
        if let Err(error) = sync_parent_dir(&self.path) {
            self.poisoned = true;
            return Err(error);
        }
        if let Err(error) = self.file.seek(SeekFrom::End(0)) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(Reclaim {
            retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
            reclaimed_bytes: u64::try_from(bytes.len().saturating_sub(rebuilt.len()))
                .unwrap_or(u64::MAX),
        })
    }
}

fn file_header_bytes(version: JournalVersion, base_sequence: u64, base_lsn: u64) -> Vec<u8> {
    let mut header = vec![0_u8; FILE_HEADER_LEN];
    header[FH_MAGIC].copy_from_slice(FILE_MAGIC);
    header[FH_VERSION].copy_from_slice(&version.as_u32().to_le_bytes());
    header[FH_BASE_SEQUENCE].copy_from_slice(&base_sequence.to_le_bytes());
    header[FH_BASE_LSN].copy_from_slice(&base_lsn.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header[FH_CRC_COVERED]);
    header[FH_CRC].copy_from_slice(&crc.finalize().to_le_bytes());
    header
}

fn write_file_header(
    file: &mut File,
    version: JournalVersion,
    base_sequence: u64,
    base_lsn: u64,
) -> io::Result<()> {
    file.write_all(&file_header_bytes(version, base_sequence, base_lsn))
}

/// Validate the file header, returning the base `(sequence, lsn)` this segment
/// starts numbering at. A rotated segment records the first surviving group's
/// numbers here so scanning never has to assume the sequence begins at 1.
fn validate_file_header(bytes: &[u8]) -> io::Result<(JournalVersion, u64, u64)> {
    let Some(header) = bytes.get(..FILE_HEADER_LEN) else {
        return Err(invalid_data("truncated journal file header"));
    };
    if &header[FH_MAGIC] != FILE_MAGIC {
        return Err(invalid_data("invalid journal magic"));
    }
    let version = JournalVersion::from_u32(u32::from_le_bytes(
        header[FH_VERSION].try_into().expect("fixed slice"),
    ))?;
    let mut crc = Hasher::new();
    crc.update(&header[FH_CRC_COVERED]);
    if u32::from_le_bytes(header[FH_CRC].try_into().expect("fixed slice")) != crc.finalize() {
        return Err(invalid_data("journal header checksum mismatch"));
    }
    Ok((
        version,
        u64::from_le_bytes(header[FH_BASE_SEQUENCE].try_into().expect("fixed slice")),
        u64::from_le_bytes(header[FH_BASE_LSN].try_into().expect("fixed slice")),
    ))
}

fn encode_group_header(
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
) -> [u8; GROUP_HEADER_LEN] {
    let mut header = [0_u8; GROUP_HEADER_LEN];
    let header_len = u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN must fit in a u32");
    header[GH_MAGIC].copy_from_slice(GROUP_MAGIC);
    header[GH_HEADER_LEN].copy_from_slice(&header_len.to_le_bytes());
    header[GH_SEQUENCE].copy_from_slice(&sequence.to_le_bytes());
    header[GH_FIRST_LSN].copy_from_slice(&first_lsn.to_le_bytes());
    header[GH_LAST_LSN].copy_from_slice(&last_lsn.to_le_bytes());
    header[GH_PAYLOAD_LEN].copy_from_slice(&payload_len.to_le_bytes());
    header[GH_RECORD_COUNT].copy_from_slice(&record_count.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header[GH_CRC_COVERED]);
    header[GH_CRC].copy_from_slice(&crc.finalize().to_le_bytes());
    header
}

fn encode_group_trailer(sequence: u64, group_crc: u32) -> [u8; GROUP_TRAILER_LEN] {
    let mut trailer = [0_u8; GROUP_TRAILER_LEN];
    trailer[GT_MAGIC].copy_from_slice(GROUP_COMMIT_MAGIC);
    trailer[GT_SEQUENCE].copy_from_slice(&sequence.to_le_bytes());
    trailer[GT_CRC].copy_from_slice(&group_crc.to_le_bytes());
    trailer
}

/// Re-encode a committed group with its original sequence and LSNs, for a
/// segment rewrite. Mirrors exactly the framing `commit_group` writes.
fn encode_group(
    group: &CommittedGroup,
    version: JournalVersion,
    into: &mut Vec<u8>,
) -> io::Result<()> {
    let mut payload = Vec::new();
    for entry in &group.entries {
        encode_mutation(
            entry.lsn,
            entry.pool,
            version,
            &entry.mutation,
            &mut payload,
        )?;
    }
    let record_count = u32::try_from(group.entries.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many journal mutations"))?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "journal group too large"))?;
    let header = encode_group_header(
        group.sequence,
        group.first_lsn,
        group.last_lsn,
        payload_len,
        record_count,
    );
    let mut group_crc = Hasher::new();
    group_crc.update(&header);
    group_crc.update(&payload);
    let trailer = encode_group_trailer(group.sequence, group_crc.finalize());
    into.extend_from_slice(&header);
    into.extend_from_slice(&payload);
    into.extend_from_slice(&trailer);
    Ok(())
}

fn encode_mutation(
    lsn: u64,
    pool: Option<ShardType>,
    version: JournalVersion,
    mutation: &Mutation,
    into: &mut Vec<u8>,
) -> io::Result<()> {
    // The flag byte is the pool tag in a v3 segment and must be zero in a v2
    // segment; enforce the pairing rather than silently writing an ambiguous
    // frame.
    let pool_byte = match (version.is_pool_tagged(), pool) {
        (true, Some(pool)) => pool_tag(pool),
        (true, None) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a pool-tagged journal frame requires a pool tag",
            ))
        }
        (false, Some(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a per-pool journal frame must not carry a pool tag",
            ))
        }
        (false, None) => 0,
    };

    let start = into.len();
    // Zeroed fixed area; `MF_FLAGS` and the DeleteCollection-only fields stay
    // zero by construction. Index the fixed area by its named field ranges so
    // no raw offset arithmetic is needed.
    into.resize(start.saturating_add(FRAME_FIXED_LEN), 0);
    let payload = {
        let frame = &mut into[start..];
        frame[MF_KIND].copy_from_slice(&match mutation {
            Mutation::Put { .. } => [1],
            Mutation::DeleteCollection { .. } => [2],
        });
        frame[MF_POOL].copy_from_slice(&[pool_byte]);
        frame[MF_LSN].copy_from_slice(&lsn.to_le_bytes());
        match mutation {
            Mutation::Put {
                collection_id,
                node_id,
                payload,
            } => {
                frame[MF_COLLECTION].copy_from_slice(collection_id);
                frame[MF_NODE].copy_from_slice(node_id);
                let payload_len = u32::try_from(payload.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mutation payload exceeds u32")
                })?;
                frame[MF_PAYLOAD_LEN].copy_from_slice(&payload_len.to_le_bytes());
                payload.as_slice()
            }
            Mutation::DeleteCollection { collection_id } => {
                frame[MF_COLLECTION].copy_from_slice(collection_id);
                &[]
            }
        }
    };
    into.extend_from_slice(payload);
    let mut crc = Hasher::new();
    crc.update(&into[start..]);
    into.extend_from_slice(&crc.finalize().to_le_bytes());
    Ok(())
}

/// Validated fixed fields of one committed group header.
struct GroupHeader {
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
}

/// Validate a group header's magic, length, and CRC, returning its fields.
fn parse_group_header(header: &[u8]) -> io::Result<GroupHeader> {
    if &header[..4] != GROUP_MAGIC
        || u32::from_le_bytes(header[4..8].try_into().expect("fixed slice"))
            != u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN fits in u32")
    {
        return Err(invalid_data("invalid journal group header"));
    }
    let mut header_crc = Hasher::new();
    header_crc.update(&header[..44]);
    if u32::from_le_bytes(header[44..48].try_into().expect("fixed slice")) != header_crc.finalize()
    {
        return Err(invalid_data("journal group header checksum mismatch"));
    }
    Ok(GroupHeader {
        sequence: u64::from_le_bytes(header[8..16].try_into().expect("fixed slice")),
        first_lsn: u64::from_le_bytes(header[16..24].try_into().expect("fixed slice")),
        last_lsn: u64::from_le_bytes(header[24..32].try_into().expect("fixed slice")),
        payload_len: u64::from_le_bytes(header[32..40].try_into().expect("fixed slice")),
        record_count: u32::from_le_bytes(header[40..44].try_into().expect("fixed slice")),
    })
}

/// Verify a group's commit trailer and whole-group CRC, returning its payload.
fn verify_group_payload<'a>(
    bytes: &'a [u8],
    header: &[u8],
    group: &GroupHeader,
    payload_start: usize,
    payload_len: usize,
) -> io::Result<&'a [u8]> {
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer_end = payload_end
        .checked_add(GROUP_TRAILER_LEN)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer = bytes
        .get(payload_end..trailer_end)
        .ok_or_else(|| invalid_data("truncated journal commit trailer"))?;
    if &trailer[..4] != GROUP_COMMIT_MAGIC
        || u64::from_le_bytes(trailer[4..12].try_into().expect("fixed slice")) != group.sequence
    {
        return Err(invalid_data("invalid journal commit trailer"));
    }
    let payload = bytes
        .get(payload_start..payload_end)
        .ok_or_else(|| invalid_data("truncated journal group payload"))?;
    let mut group_crc = Hasher::new();
    group_crc.update(header);
    group_crc.update(payload);
    if u32::from_le_bytes(trailer[12..16].try_into().expect("fixed slice")) != group_crc.finalize()
    {
        return Err(invalid_data("journal committed group checksum mismatch"));
    }
    Ok(payload)
}

fn scan_bytes(
    bytes: &[u8],
    base_sequence: u64,
    base_lsn: u64,
    version: JournalVersion,
) -> io::Result<Scan> {
    if bytes.len() < FILE_HEADER_LEN {
        return Err(invalid_data("truncated journal file header"));
    }
    scan_groups_from(
        &bytes[FILE_HEADER_LEN..],
        u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX),
        base_sequence,
        base_lsn,
        base_lsn,
        version,
    )
}

/// Scan complete groups from `bytes`, which begins at absolute file offset
/// `base_offset`.
///
/// `expected_sequence` is a floor, not an exact match: a segment sharing a
/// global sequence with other pools may skip values. `expected_lsn` must match
/// the first group's `first_lsn` exactly, since LSNs stay contiguous within a
/// segment. `Scan::valid_len` is absolute in the file, not relative to `bytes`.
fn scan_groups_from(
    bytes: &[u8],
    base_offset: u64,
    mut expected_sequence: u64,
    mut expected_lsn: u64,
    segment_base_lsn: u64,
    version: JournalVersion,
) -> io::Result<Scan> {
    let mut cursor = 0usize;
    let mut valid_len = base_offset;
    let mut groups = Vec::new();
    let mut truncated_tail = false;

    while cursor < bytes.len() {
        let remaining = bytes.len().saturating_sub(cursor);
        if remaining < GROUP_HEADER_LEN {
            truncated_tail = true;
            break;
        }
        let header = bytes
            .get(cursor..cursor.saturating_add(GROUP_HEADER_LEN))
            .ok_or_else(|| invalid_data("truncated journal group header"))?;
        let group = parse_group_header(header)?;
        if group.sequence < expected_sequence
            || group.first_lsn != expected_lsn
            || group.last_lsn < group.first_lsn
            || group
                .last_lsn
                .saturating_sub(group.first_lsn)
                .saturating_add(1)
                != u64::from(group.record_count)
            || group.record_count == 0
            || group.payload_len > MAX_GROUP_LEN
            || u64::from(group.record_count).saturating_mul(MIN_FRAME_LEN as u64)
                > group.payload_len
        {
            return Err(invalid_data("invalid journal sequence or group bounds"));
        }
        let payload_len = usize::try_from(group.payload_len)
            .map_err(|_| invalid_data("journal group length exceeds address space"))?;
        let total_len = GROUP_HEADER_LEN
            .checked_add(payload_len)
            .and_then(|len| len.checked_add(GROUP_TRAILER_LEN))
            .ok_or_else(|| invalid_data("journal group length overflow"))?;
        if remaining < total_len {
            truncated_tail = true;
            break;
        }
        let payload_index = cursor.saturating_add(GROUP_HEADER_LEN);
        let payload = verify_group_payload(bytes, header, &group, payload_index, payload_len)?;
        let payload_offset = base_offset
            .saturating_add(u64::try_from(payload_index).unwrap_or(u64::MAX))
            .try_into()
            .unwrap_or(usize::MAX);
        let entries = decode_mutations(
            payload,
            group.first_lsn,
            group.record_count,
            payload_offset,
            version,
        )?;
        groups.push(CommittedGroup {
            sequence: group.sequence,
            first_lsn: group.first_lsn,
            last_lsn: group.last_lsn,
            entries,
        });
        cursor = cursor.saturating_add(total_len);
        valid_len = base_offset.saturating_add(u64::try_from(cursor).unwrap_or(u64::MAX));
        expected_sequence = group
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal group sequence overflow"))?;
        expected_lsn = group
            .last_lsn
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal LSN overflow"))?;
    }

    Ok(Scan {
        groups,
        valid_len,
        truncated_tail,
        base_lsn: segment_base_lsn,
    })
}

fn decode_mutations(
    payload: &[u8],
    first_lsn: u64,
    record_count: u32,
    payload_offset: usize,
    version: JournalVersion,
) -> io::Result<Vec<JournalEntry>> {
    let mut cursor = 0_usize;
    let mut entries = Vec::with_capacity(usize::try_from(record_count).unwrap_or(0));
    for index in 0..record_count {
        let start = cursor;
        if payload.len().saturating_sub(cursor) < MIN_FRAME_LEN {
            return Err(invalid_data("truncated journal mutation frame"));
        }
        let fixed_end = cursor
            .checked_add(FRAME_FIXED_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let frame = payload
            .get(cursor..fixed_end)
            .ok_or_else(|| invalid_data("truncated journal mutation frame"))?;
        let kind = frame[MF_KIND.start];
        let tag = frame[MF_POOL.start];
        if frame[MF_FLAGS] != [0, 0] {
            return Err(invalid_data("unsupported journal mutation flags"));
        }
        let pool = if version.is_pool_tagged() {
            if !matches!(tag, 1..=3) {
                return Err(invalid_data("pool-tagged frame has no valid pool tag"));
            }
            pool_from_tag(tag)
        } else {
            if tag != 0 {
                return Err(invalid_data("untagged frame has a nonzero flag byte"));
            }
            None
        };
        let lsn = u64::from_le_bytes(frame[MF_LSN].try_into().expect("fixed slice"));
        let expected = first_lsn
            .checked_add(u64::from(index))
            .ok_or_else(|| invalid_data("journal mutation LSN overflow"))?;
        if lsn != expected {
            return Err(invalid_data("journal mutation LSN out of order"));
        }
        let collection_id: [u8; 16] = frame[MF_COLLECTION].try_into().expect("fixed slice");
        let node_id: [u8; 16] = frame[MF_NODE].try_into().expect("fixed slice");
        let payload_len = usize::try_from(u32::from_le_bytes(
            frame[MF_PAYLOAD_LEN].try_into().expect("fixed slice"),
        ))
        .map_err(|_| invalid_data("journal mutation payload length exceeds address space"))?;
        let frame_end = fixed_end
            .checked_add(payload_len)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let crc_end = frame_end
            .checked_add(FRAME_TRAILER_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let data = payload
            .get(fixed_end..frame_end)
            .ok_or_else(|| invalid_data("journal mutation payload exceeds group"))?;
        let crc_bytes = payload
            .get(frame_end..crc_end)
            .ok_or_else(|| invalid_data("journal mutation checksum exceeds group"))?;
        let mut crc = Hasher::new();
        crc.update(
            payload
                .get(start..frame_end)
                .ok_or_else(|| invalid_data("journal mutation frame exceeds group"))?,
        );
        if u32::from_le_bytes(crc_bytes.try_into().expect("fixed slice")) != crc.finalize() {
            return Err(invalid_data("journal mutation checksum mismatch"));
        }
        let mutation = match kind {
            1 => Mutation::Put {
                collection_id,
                node_id,
                payload: data.to_vec(),
            },
            2 if payload_len == 0 && node_id == [0; 16] => {
                Mutation::DeleteCollection { collection_id }
            }
            _ => return Err(invalid_data("unknown or malformed journal mutation kind")),
        };
        entries.push(JournalEntry {
            lsn,
            offset: u64::try_from(payload_offset.saturating_add(start)).unwrap_or(u64::MAX),
            frame_len: u64::try_from(crc_end.saturating_sub(start)).unwrap_or(u64::MAX),
            pool,
            mutation,
        });
        cursor = crc_end;
    }
    if cursor != payload.len() {
        return Err(invalid_data("extra bytes after journal mutation frames"));
    }
    Ok(entries)
}

fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        // Rust's portable std API has no directory-handle durability
        // operation on non-Unix platforms. Refuse to enable the journal until
        // a platform-specific implementation can uphold create/rotation
        // durability; silently succeeding would weaken the contract.
        let _ = parent;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "durable journal creation requires platform directory sync support",
        ))
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{Journal, JournalCoordinator, Mutation};
    use std::fs;
    use std::io::Write as _;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mtxdb_journal_{label}_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn read_only_scan_treats_missing_and_short_segments_as_empty() {
        let missing = temp_path("read_only_missing");
        let _ = fs::remove_file(&missing);
        let scan = Journal::scan_read_only(&missing).unwrap();
        assert_eq!(scan.groups.len(), 0);
        assert_eq!(scan.valid_len, 0);
        assert!(!scan.truncated_tail);
        assert!(
            !missing.exists(),
            "read-only scan must not create a segment"
        );

        let short = temp_path("read_only_short");
        fs::write(&short, b"MTXWAL").unwrap();
        let before = fs::read(&short).unwrap();
        let scan = Journal::scan_read_only(&short).unwrap();
        assert_eq!(scan.groups.len(), 0);
        assert_eq!(scan.valid_len, 0);
        assert!(!scan.truncated_tail);
        assert_eq!(
            fs::read(&short).unwrap(),
            before,
            "scan must not repair bytes"
        );
        fs::remove_file(short).unwrap();
    }

    fn put(collection: u8, node: u8, payload: &[u8]) -> Mutation {
        Mutation::Put {
            collection_id: [collection; 16],
            node_id: [node; 16],
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn shared_segment_round_trips_pool_tags() {
        use crate::layout::ShardType;
        let path = temp_path("shared_pool_tags");
        let _ = fs::remove_file(&path);

        let (mut journal, scan) = Journal::open_shared(&path).unwrap();
        assert_eq!(scan.groups.len(), 0);
        let tagged = [
            (Some(ShardType::State), put(1, 1, b"state")),
            (Some(ShardType::EventDag), put(2, 2, b"event")),
            (Some(ShardType::AuthChain), put(3, 3, b"auth")),
        ];
        journal
            .append_group_tagged_with_sequence(&tagged, None)
            .unwrap();
        journal.make_durable().unwrap();

        let scan = Journal::scan_read_only(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        let pools: Vec<_> = scan.groups[0]
            .entries
            .iter()
            .map(|entry| entry.pool)
            .collect();
        assert_eq!(
            pools,
            vec![
                Some(ShardType::State),
                Some(ShardType::EventDag),
                Some(ShardType::AuthChain),
            ]
        );

        // Recovery must preserve the tags so each frame can be routed.
        let (_journal, scan) = Journal::open_shared(&path).unwrap();
        assert_eq!(scan.groups[0].entries[1].pool, Some(ShardType::EventDag));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn shared_segment_rejects_untagged_frames() {
        let path = temp_path("shared_untagged");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open_shared(&path).unwrap();
        assert!(
            journal.append_group(&[put(1, 1, b"x")]).is_err(),
            "a pool-tagged segment must reject an untagged frame"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn per_pool_segment_rejects_tagged_frames() {
        use crate::layout::ShardType;
        let path = temp_path("perpool_tagged");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        let tagged = [(Some(ShardType::State), put(1, 1, b"x"))];
        assert!(
            journal
                .append_group_tagged_with_sequence(&tagged, None)
                .is_err(),
            "a per-pool segment must reject a tagged frame"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn open_mode_must_match_segment_version() {
        let path = temp_path("version_mismatch");
        let _ = fs::remove_file(&path);
        Journal::open(&path).unwrap();
        assert!(
            Journal::open_shared(&path).is_err(),
            "a per-pool segment must not open as shared"
        );
        fs::remove_file(&path).unwrap();

        Journal::open_shared(&path).unwrap();
        assert!(
            Journal::open(&path).is_err(),
            "a shared segment must not open as per-pool"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coordinator_publishes_tagged_frames() {
        use crate::layout::ShardType;
        let path = temp_path("coordinator_tagged");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open_shared(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);
        coordinator
            .publish_tagged(ShardType::State, put(1, 1, b"a"), |_| {})
            .unwrap();
        coordinator
            .publish_tagged(ShardType::AuthChain, put(3, 3, b"c"), |_| {})
            .unwrap();
        coordinator.sync().unwrap();

        let scan = Journal::scan_read_only(&path).unwrap();
        let pools: Vec<_> = scan.groups[0]
            .entries
            .iter()
            .map(|entry| entry.pool)
            .collect();
        assert_eq!(
            pools,
            vec![Some(ShardType::State), Some(ShardType::AuthChain)]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn append_group_is_visible_before_make_durable() {
        let path = temp_path("append_visible");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        let receipt = journal.append_group(&[put(1, 1, b"pending")]).unwrap();
        assert_eq!((receipt.first_lsn, receipt.last_lsn), (1, 1));

        // The complete group, trailer included, is readable by a read-only
        // scanner before any fsync: this is the committed-but-unflushed state
        // the read overlay observes.
        let scan = Journal::scan_read_only(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].last_lsn, 1);
        assert!(!scan.truncated_tail);

        // `make_durable` is idempotent and must not duplicate the group.
        journal.make_durable().unwrap();
        journal.make_durable().unwrap();
        let scan = Journal::scan_read_only(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);

        drop(journal);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn commit_group_appends_then_syncs() {
        let path = temp_path("commit_appends_then_syncs");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        let receipt = journal.commit_group(&[put(1, 1, b"durable")]).unwrap();
        assert_eq!(receipt.last_lsn, 1);
        // The wrapper left the group complete and readable.
        let scan = Journal::scan_read_only(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        // A later explicit make_durable is a harmless no-op.
        journal.make_durable().unwrap();
        assert_eq!(Journal::scan_read_only(&path).unwrap().groups.len(), 1);
        drop(journal);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coordinator_visible_lsn_matches_committed_after_sync() {
        let path = temp_path("coordinator_visible");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);
        assert_eq!(coordinator.visible_lsn(), 0);
        let lsn = coordinator.publish(put(1, 1, b"first"), |_| {}).unwrap();
        coordinator.sync().unwrap();
        assert_eq!(coordinator.visible_lsn(), lsn);
        assert_eq!(coordinator.committed_lsn(), lsn);
        drop(coordinator);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn committed_groups_round_trip_with_contiguous_lsns() {
        let path = temp_path("round_trip");
        let _ = fs::remove_file(&path);
        let (mut journal, initial) = Journal::open(&path).unwrap();
        assert_eq!(initial.groups.len(), 0);
        let committed = journal
            .commit_group(&[
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [2; 16],
                    payload: b"alpha".to_vec(),
                },
                Mutation::DeleteCollection {
                    collection_id: [3; 16],
                },
            ])
            .unwrap();
        assert_eq!((committed.first_lsn, committed.last_lsn), (1, 2));
        drop(journal);

        let (reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].sequence, 1);
        assert_eq!(scan.groups[0].first_lsn, committed.first_lsn);
        assert_eq!(scan.groups[0].last_lsn, committed.last_lsn);
        let entries = &scan.groups[0].entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(
            entries[0].mutation,
            Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"alpha".to_vec(),
            }
        );
        assert_eq!(entries[1].lsn, 2);
        assert_eq!(
            entries[1].mutation,
            Mutation::DeleteCollection {
                collection_id: [3; 16]
            }
        );
        // The recovered byte ranges must let a reader re-read each frame from
        // the segment: the first frame starts right after the file + group
        // headers, and lengths are fixed + payload + CRC.
        let group_payload_start = super::FILE_HEADER_LEN + super::GROUP_HEADER_LEN;
        assert_eq!(entries[0].offset, group_payload_start as u64);
        assert_eq!(
            entries[0].frame_len,
            (super::FRAME_FIXED_LEN + 5 + super::FRAME_TRAILER_LEN) as u64
        );
        assert_eq!(
            entries[1].offset,
            entries[0].offset.saturating_add(entries[0].frame_len)
        );
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn incomplete_final_group_is_truncated_on_open() {
        let path = temp_path("torn_tail");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"complete".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let good_len = fs::metadata(&path).unwrap().len();
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"MWG1\x30\0")
            .unwrap();
        let (reopened, scan) = Journal::open(&path).unwrap();
        assert!(scan.truncated_tail);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(fs::metadata(&path).unwrap().len(), good_len);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_only_scan_keeps_complete_groups_and_leaves_torn_tail_untouched() {
        let path = temp_path("read_only_torn_tail");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal.commit_group(&[put(1, 2, b"complete")]).unwrap();
        drop(journal);

        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"MWG1\x30\0")
            .unwrap();
        let before = fs::read(&path).unwrap();
        let scan = Journal::scan_read_only(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert!(scan.truncated_tail);
        assert_eq!(fs::read(&path).unwrap(), before, "scan must not truncate");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn committed_group_corruption_fails_open() {
        let path = temp_path("corrupt");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"committed".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let mut bytes = fs::read(&path).unwrap();
        let payload_byte = super::FILE_HEADER_LEN
            .saturating_add(super::GROUP_HEADER_LEN)
            .saturating_add(super::FRAME_FIXED_LEN);
        bytes[payload_byte] ^= 0x80;
        fs::write(&path, bytes).unwrap();
        assert!(Journal::open(&path).is_err());
        fs::remove_file(path).unwrap();
    }

    /// A partial header is a crash mid-creation, not corruption: no group can
    /// exist without a synced header, so open resets the segment and it
    /// remains usable.
    #[test]
    fn partial_file_header_is_reset_on_open() {
        let path = temp_path("partial_header");
        let _ = fs::remove_file(&path);
        fs::write(&path, b"MTX").unwrap();

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 0);
        assert!(!scan.truncated_tail);
        let committed = journal
            .commit_group(&[Mutation::Put {
                collection_id: [4; 16],
                node_id: [5; 16],
                payload: b"after-reset".to_vec(),
            }])
            .unwrap();
        assert_eq!(committed.sequence, 1);
        assert_eq!(committed.first_lsn, 1);
        drop(journal);

        let (_reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].entries.len(), 1);
        fs::remove_file(path).unwrap();
    }

    /// Reopen after a torn tail, truncate it, and confirm numbering continues
    /// from the last committed group rather than restarting.
    #[test]
    fn recovery_after_torn_tail_continues_numbering() {
        let path = temp_path("continue");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [1; 16],
                payload: b"one".to_vec(),
            }])
            .unwrap();
        drop(journal);

        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"MWG1\x30\0\x30\0")
            .unwrap();

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert!(scan.truncated_tail);
        assert_eq!(scan.groups.len(), 1);
        let committed = journal
            .commit_group(&[
                Mutation::Put {
                    collection_id: [2; 16],
                    node_id: [2; 16],
                    payload: b"two".to_vec(),
                },
                Mutation::DeleteCollection {
                    collection_id: [3; 16],
                },
            ])
            .unwrap();
        assert_eq!(committed.sequence, 2);
        assert_eq!(committed.first_lsn, 2);
        assert_eq!(committed.last_lsn, 3);
        drop(journal);

        let (_reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups[0].last_lsn, 1);
        assert_eq!(scan.groups[1].first_lsn, 2);
        assert_eq!(scan.groups[1].last_lsn, 3);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coordinator_commits_only_through_the_captured_sync_boundary() {
        let path = temp_path("coordinator_boundary");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);

        let first_lsn = coordinator
            .publish(
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [1; 16],
                    payload: b"first".to_vec(),
                },
                |_| {},
            )
            .unwrap();
        let first_target = coordinator.capture_sync_target();
        assert_eq!(first_target, first_lsn);

        let second_lsn = coordinator
            .publish(
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [2; 16],
                    payload: b"second".to_vec(),
                },
                |_| {},
            )
            .unwrap();
        assert_eq!(second_lsn, first_lsn.saturating_add(1));

        let first_commit = coordinator.sync_through(first_target).unwrap().unwrap();
        assert_eq!(first_commit.first_lsn, first_lsn);
        assert_eq!(first_commit.last_lsn, first_target);
        assert!(coordinator.sync_through(first_target).unwrap().is_none());

        let second_commit = coordinator.sync().unwrap().unwrap();
        assert_eq!(second_commit.first_lsn, second_lsn);
        assert_eq!(second_commit.last_lsn, second_lsn);

        drop(coordinator);
        let (_journal, recovered) = Journal::open(&path).unwrap();
        assert_eq!(recovered.groups.len(), 2);
        assert_eq!(recovered.groups[0].last_lsn, first_target);
        assert_eq!(recovered.groups[1].first_lsn, second_lsn);
        fs::remove_file(path).unwrap();
    }

    /// A target that was never published is a caller bug, not a durable commit.
    #[test]
    fn sync_through_rejects_an_unpublished_target() {
        let path = temp_path("coordinator_unpublished");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);

        assert!(coordinator.sync_through(0).unwrap().is_none());
        let error = coordinator.sync_through(1).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        fs::remove_file(path).unwrap();
    }

    /// Many writers publishing and syncing concurrently must produce one
    /// contiguous, gap-free LSN sequence: a caller that loses the commit race
    /// is covered by another caller's group rather than acknowledged early.
    #[test]
    fn concurrent_callers_produce_contiguous_groups() {
        let path = temp_path("coordinator_concurrent");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = std::sync::Arc::new(JournalCoordinator::new(journal, &scan));

        let threads = 8_u8;
        let per_thread = 16_u8;
        let mut handles = Vec::new();
        for thread in 0..threads {
            let coordinator = std::sync::Arc::clone(&coordinator);
            handles.push(std::thread::spawn(move || {
                for item in 0..per_thread {
                    coordinator
                        .publish(
                            Mutation::Put {
                                collection_id: [thread; 16],
                                node_id: [item; 16],
                                payload: b"payload".to_vec(),
                            },
                            |_| {},
                        )
                        .unwrap();
                    // Either this caller commits the group or a concurrent
                    // caller already committed one covering this LSN.
                    coordinator.sync().unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let published = coordinator.capture_sync_target();
        assert_eq!(
            published,
            u64::from(u32::from(threads) * u32::from(per_thread))
        );
        coordinator.sync().unwrap();

        drop(coordinator);
        let (_journal, recovered) = Journal::open(&path).unwrap();
        let total: u64 = recovered
            .groups
            .iter()
            .map(|group| u64::try_from(group.entries.len()).unwrap())
            .sum();
        assert_eq!(total, published);
        let mut expected_first = 1;
        for group in &recovered.groups {
            assert_eq!(group.first_lsn, expected_first);
            expected_first = group.last_lsn.saturating_add(1);
        }
        fs::remove_file(path).unwrap();
    }

    /// Rotation drops covered groups but keeps the uncovered suffix, preserving
    /// each surviving group's original sequence and LSNs.
    #[test]
    fn reclaim_retains_uncovered_suffix_and_preserves_numbering() {
        let path = temp_path("reclaim_suffix");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        let first = journal.commit_group(&[put(1, 1, b"first")]).unwrap();
        let second = journal
            .commit_group(&[
                put(1, 2, b"second"),
                Mutation::DeleteCollection {
                    collection_id: [9; 16],
                },
            ])
            .unwrap();
        let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();

        let reclaim = journal.reclaim_through(first.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 2);
        assert!(reclaim.reclaimed_bytes > 0);
        assert!(!path.with_extension("rotate").exists());
        drop(journal);

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups[0].sequence, second.sequence);
        assert_eq!(scan.groups[0].first_lsn, second.first_lsn);
        assert_eq!(scan.groups[0].last_lsn, second.last_lsn);
        assert_eq!(scan.groups[1].sequence, third.sequence);
        assert_eq!(scan.groups[1].last_lsn, third.last_lsn);

        // Numbering continues from the global high-water mark, not the new base.
        let fourth = journal.commit_group(&[put(1, 4, b"fourth")]).unwrap();
        assert_eq!(fourth.sequence, third.sequence.saturating_add(1));
        assert_eq!(fourth.first_lsn, third.last_lsn.saturating_add(1));
        drop(journal);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 3);
        fs::remove_file(path).unwrap();
    }

    /// Reclaiming everything leaves an empty segment whose base still points
    /// past the last committed LSN, so numbering never restarts.
    #[test]
    fn reclaim_all_covered_keeps_numbering_for_the_next_group() {
        let path = temp_path("reclaim_all");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal.commit_group(&[put(1, 1, b"first")]).unwrap();
        let second = journal.commit_group(&[put(1, 2, b"second")]).unwrap();

        let reclaim = journal.reclaim_through(second.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 0);
        drop(journal);

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 0);
        let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();
        assert_eq!(third.sequence, second.sequence.saturating_add(1));
        assert_eq!(third.first_lsn, second.last_lsn.saturating_add(1));
        drop(journal);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].first_lsn, second.last_lsn.saturating_add(1));
        fs::remove_file(path).unwrap();
    }

    /// A covered LSN at or below the segment base reclaims nothing.
    #[test]
    fn reclaim_below_base_is_a_noop() {
        let path = temp_path("reclaim_noop");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal.commit_group(&[put(1, 1, b"only")]).unwrap();

        let reclaim = journal.reclaim_through(0).unwrap();
        assert_eq!(reclaim.retained_groups, 1);
        assert_eq!(reclaim.reclaimed_bytes, 0);
        drop(journal);
        fs::remove_file(path).unwrap();
    }

    /// The coordinator exposes reclamation without disturbing its LSN counters:
    /// a publish after reclaim still continues the sequence.
    #[test]
    fn coordinator_reclaims_without_reusing_lsns() {
        let path = temp_path("coordinator_reclaim");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);
        let lsn = coordinator.publish(put(1, 1, b"first"), |_| {}).unwrap();
        let receipt = coordinator.sync().unwrap().unwrap();
        assert_eq!(receipt.last_lsn, lsn);

        let reclaim = coordinator.reclaim_through(receipt.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 0);
        assert_eq!(coordinator.committed_lsn(), receipt.last_lsn);

        let next = coordinator.publish(put(1, 2, b"second"), |_| {}).unwrap();
        assert_eq!(next, lsn.saturating_add(1));
        coordinator.sync().unwrap();
        drop(coordinator);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].first_lsn, lsn.saturating_add(1));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn shared_sequence_orders_groups_across_segments_and_allows_gaps() {
        let dir = temp_path("shared_sequence");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let counter = Arc::new(AtomicU64::new(1));

        let path_a = dir.join("a.wal");
        let path_b = dir.join("b.wal");
        let (journal_a, scan_a) = Journal::open(&path_a).unwrap();
        let (journal_b, scan_b) = Journal::open(&path_b).unwrap();
        let a = JournalCoordinator::with_shared_sequence(journal_a, &scan_a, Arc::clone(&counter));
        let b = JournalCoordinator::with_shared_sequence(journal_b, &scan_b, Arc::clone(&counter));

        // Interleave commits; the shared counter hands out 1, 2, 3 so each
        // segment skips the values the other consumed.
        a.publish(put(1, 1, b"a1"), |_| {}).unwrap();
        a.sync().unwrap();
        b.publish(put(2, 1, b"b1"), |_| {}).unwrap();
        b.sync().unwrap();
        a.publish(put(1, 2, b"a2"), |_| {}).unwrap();
        a.sync().unwrap();

        drop(a);
        drop(b);
        // Segment A holds sequences 1 and 3; recovery must accept the gap.
        let (_journal, scan) = Journal::open(&path_a).unwrap();
        assert_eq!(
            scan.groups
                .iter()
                .map(|group| group.sequence)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        let (_journal, scan) = Journal::open(&path_b).unwrap();
        assert_eq!(
            scan.groups
                .iter()
                .map(|group| group.sequence)
                .collect::<Vec<_>>(),
            vec![2]
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
