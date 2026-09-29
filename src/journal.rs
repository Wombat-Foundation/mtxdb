//! Append-only, checksummed commit journal used by the packfile WAL path.
//!
//! This module owns the journal's disk framing and durability boundary. It
//! also provides [`JournalCoordinator`](crate::journal::JournalCoordinator),
//! which captures each sync caller's
//! target LSN and releases it only after a durable group covers that target.

use std::collections::HashMap;
#[cfg(feature = "multi-reader")]
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

const FILE_MAGIC: &[u8; 8] = b"MTXWAL01";
const GROUP_MAGIC: &[u8; 4] = b"MWG1";
const GROUP_COMMIT_MAGIC: &[u8; 4] = b"CMIT";

/// On-disk journal format version. The single place that decides frame
/// dialect: callers match on this rather than comparing raw version numbers,
/// so a future bump only has to add a variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JournalVersion {
    /// Per-pool segment. Mutation frames are untagged and the flag byte must be
    /// zero. Version 4 adds the embedded durability-mark header fields.
    V4,
    /// Shared multi-pool segment. Every mutation frame carries a mandatory pool
    /// tag in the flag byte, so one physical WAL can carry the state,
    /// event-DAG, and edges pools and recovery can route each frame back
    /// to its pool.
    V5PoolTagged,
}

impl JournalVersion {
    /// Version a freshly created per-pool segment uses.
    const fn per_pool() -> Self {
        Self::V4
    }

    /// Version a freshly created shared (multi-pool) segment uses.
    #[cfg(feature = "multi-reader")]
    const fn shared() -> Self {
        Self::V5PoolTagged
    }

    const fn as_u32(self) -> u32 {
        match self {
            Self::V4 => 4,
            Self::V5PoolTagged => 5,
        }
    }

    fn from_u32(value: u32) -> io::Result<Self> {
        match value {
            4 => Ok(Self::V4),
            5 => Ok(Self::V5PoolTagged),
            _ => Err(invalid_data("unsupported journal version")),
        }
    }

    /// Whether every mutation frame in this segment must carry a pool tag.
    const fn is_pool_tagged(self) -> bool {
        matches!(self, Self::V5PoolTagged)
    }
}

// File header field byte ranges. `FILE_HEADER_LEN` is the sum of the fields.
const FH_MAGIC: std::ops::Range<usize> = 0..8;
const FH_VERSION: std::ops::Range<usize> = 8..12;
const FH_BASE_SEQUENCE: std::ops::Range<usize> = 12..20;
const FH_BASE_LSN: std::ops::Range<usize> = 20..28;
const FH_BASE_CRC: std::ops::Range<usize> = 28..32;
/// Header bytes covered by the immutable base CRC.
const FH_BASE_CRC_COVERED: std::ops::Range<usize> = 0..FH_BASE_CRC.start;

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

/// Segment size above which a sync forces a full index checkpoint, because only
/// a full checkpoint records journal coverage and lets the segment be
/// reclaimed. A quarter of [`MAX_SEGMENT_LEN`] leaves headroom for writes that
/// land while the checkpoint runs. It is a margin, not a guarantee: a writer
/// that appends more than the remaining space during one checkpoint still fills
/// the segment.
/// A coordinator starts with this and can be given another with
/// [`JournalCoordinator::set_reclaim_trigger_len`].
pub(crate) const RECLAIM_TRIGGER_LEN: u64 = MAX_SEGMENT_LEN / 4;

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
#[cfg(feature = "multi-reader")]
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

#[derive(Debug)]
#[cfg(feature = "multi-reader")]
struct TxnStageData {
    pools: [Vec<Mutation>; ShardType::ALL.len()],
    applied: [Vec<bool>; ShardType::ALL.len()],
    bytes: usize,
    /// Successful pool appends. Retrying a callback after a partial error
    /// resumes at the failed pool instead of duplicating earlier groups.
    appended: [bool; ShardType::ALL.len()],
    /// Receipt identifying the published journal group owned by this stage.
    receipt: Option<CommitReceipt>,
}

/// What a transaction's staged mutations say about one record.
#[cfg(feature = "multi-reader")]
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
#[cfg(feature = "multi-reader")]
pub struct TxnStage {
    state: std::sync::atomic::AtomicU8,
    data: Mutex<TxnStageData>,
    #[cfg(test)]
    lookup_many_calls: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "multi-reader")]
impl Default for TxnStage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "multi-reader")]
impl TxnStage {
    const ACTIVE: u8 = 0;
    const DISCARDED: u8 = 1;
    const PUBLISHED: u8 = 2;
    const MATERIALIZING: u8 = 3;
    const JOURNAL_PUBLISHED: u8 = 4;

    /// Create an empty stage for one transaction attempt.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(Self::ACTIVE),
            data: Mutex::new(TxnStageData {
                pools: std::array::from_fn(|_| Vec::new()),
                applied: std::array::from_fn(|_| Vec::new()),
                bytes: 0,
                appended: [false; ShardType::ALL.len()],
                receipt: None,
            }),
            #[cfg(test)]
            lookup_many_calls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> TxnStageState {
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED => TxnStageState::Discarded,
            Self::PUBLISHED => TxnStageState::Published,
            Self::MATERIALIZING => TxnStageState::Materializing,
            Self::JOURNAL_PUBLISHED => TxnStageState::JournalPublished,
            _ => TxnStageState::Active,
        }
    }

    /// Look one record up in the mutations staged for `pool`. See
    /// [`Self::lookup_many`], which this calls.
    #[must_use]
    pub fn lookup(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_id: &[u8; 16],
    ) -> StagedLookup {
        self.lookup_many(pool, collection_id, std::slice::from_ref(node_id))
            .pop()
            .unwrap_or(StagedLookup::Absent)
    }

    /// Look several records up in the mutations staged for `pool`, in the order
    /// of `node_ids`, taking the stage lock once and scanning the staged
    /// mutations once however many records are asked for.
    ///
    /// This is what lets a transaction read its own uncommitted writes. The
    /// staged mutations are in append order, so the newest one that names a
    /// record wins. A staged `DeleteCollection` for the record's collection
    /// hides everything older, both earlier staged puts and whatever the live
    /// pool holds, while a put staged after it is newer and is seen. Only an
    /// active stage answers: once the stage is discarded, or publication has
    /// begun, every record is `Absent` and the caller reads the live store,
    /// which by then serves the group through the transaction overlay.
    ///
    /// The single pass keeps the newest put position for each wanted record
    /// and the position of the last delete for the collection, so the cost is
    /// the staged mutations plus the records asked for, not their product.
    /// Payloads are cloned only for records that resolve to a put.
    #[must_use]
    pub fn lookup_many(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[[u8; 16]],
    ) -> Vec<StagedLookup> {
        #[cfg(test)]
        self.lookup_many_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE || node_ids.is_empty() {
            return vec![StagedLookup::Absent; node_ids.len()];
        }
        let wanted: HashSet<[u8; 16]> = node_ids.iter().copied().collect();
        let mut newest_put: HashMap<[u8; 16], (usize, &Vec<u8>)> = HashMap::new();
        let mut last_delete: Option<usize> = None;
        for (position, mutation) in data.pools[pool_index(pool)].iter().enumerate() {
            match mutation {
                Mutation::Put {
                    collection_id: staged_collection,
                    node_id,
                    payload,
                } if staged_collection == collection_id && wanted.contains(node_id) => {
                    newest_put.insert(*node_id, (position, payload));
                }
                Mutation::DeleteCollection {
                    collection_id: staged_collection,
                } if staged_collection == collection_id => last_delete = Some(position),
                _ => {}
            }
        }
        node_ids
            .iter()
            .map(|node_id| match (newest_put.get(node_id), last_delete) {
                // A put older than the delete is hidden by it.
                (Some((position, _)), Some(delete)) if *position < delete => StagedLookup::Deleted,
                (Some((_, payload)), _) => StagedLookup::Put((*payload).clone()),
                (None, Some(_)) => StagedLookup::Deleted,
                (None, None) => StagedLookup::Absent,
            })
            .collect()
    }

    /// Number of batch lookup passes, for complexity tests only.
    #[cfg(test)]
    #[must_use]
    pub fn lookup_many_calls(&self) -> u64 {
        self.lookup_many_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Discard an active attempt. Safe to call more than once.
    ///
    /// See the "Not a rollback" section on [`TxnStage`]: this drops only the
    /// buffered journal mutations and never touches packs or the live index.
    pub fn discard(&self) {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) == Self::ACTIVE {
            data.pools.iter_mut().for_each(Vec::clear);
            data.applied.iter_mut().for_each(Vec::clear);
            data.bytes = 0;
            self.state.store(Self::DISCARDED, Ordering::Release);
        }
    }

    /// Snapshot staged mutations for application to storage at commit time.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn snapshot_mutations(&self) -> [Vec<Mutation>; ShardType::ALL.len()] {
        self.data.lock().pools.clone()
    }

    /// Receipt identifying this stage's published journal group.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn published_receipt(&self) -> Option<CommitReceipt> {
        self.data.lock().receipt
    }

    /// Which pools (in [`ShardType::ALL`] order) have at least one staged
    /// mutation.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn touched_pools(&self) -> [bool; ShardType::ALL.len()] {
        let data = self.data.lock();
        std::array::from_fn(|index| data.pools.get(index).is_some_and(|pool| !pool.is_empty()))
    }

    /// Whether this stage has no mutations to publish or materialize.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn is_empty(&self) -> bool {
        self.data.lock().pools.iter().all(Vec::is_empty)
    }

    /// Return whether one staged mutation has already been applied to storage.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn mutation_applied(&self, pool: ShardType, index: usize) -> bool {
        self.data.lock().applied[pool_index(pool)]
            .get(index)
            .copied()
            .unwrap_or(false)
    }

    /// Record successful application of one staged mutation. This makes a
    /// retry after a later mutation fails resume at the failed mutation.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn mark_mutation_applied(&self, pool: ShardType, index: usize) -> io::Result<()> {
        let mut data = self.data.lock();
        let applied = data.applied[pool_index(pool)]
            .get_mut(index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "staged mutation index out of bounds",
                )
            })?;
        *applied = true;
        Ok(())
    }

    /// Begin pack/index materialization after the journal group is published.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn begin_materialization(&self) -> io::Result<()> {
        let state = self.state.load(Ordering::Acquire);
        match state {
            Self::ACTIVE | Self::JOURNAL_PUBLISHED => {
                self.state.store(Self::MATERIALIZING, Ordering::Release);
                Ok(())
            }
            Self::MATERIALIZING | Self::PUBLISHED => Ok(()),
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other("invalid transaction stage state")),
        }
    }

    /// Mark the journal group published while storage application remains
    /// retryable.
    pub(crate) fn mark_journal_published(&self) -> io::Result<()> {
        match self.state.load(Ordering::Acquire) {
            Self::ACTIVE => {
                self.state.store(Self::JOURNAL_PUBLISHED, Ordering::Release);
                Ok(())
            }
            Self::JOURNAL_PUBLISHED | Self::MATERIALIZING | Self::PUBLISHED => Ok(()),
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other("invalid transaction stage state")),
        }
    }

    /// Complete an active no-op transaction without creating a journal group.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn mark_empty_published(&self) -> io::Result<()> {
        let data = self.data.lock();
        if data.pools.iter().any(|pool| !pool.is_empty()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot complete a transaction with staged mutations as empty",
            ));
        }
        match self.state.load(Ordering::Acquire) {
            Self::ACTIVE | Self::PUBLISHED => {
                self.state.store(Self::PUBLISHED, Ordering::Release);
                Ok(())
            }
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other(
                "transaction stage is not empty and active",
            )),
        }
    }

    /// Mark both journal publication and storage application complete.
    #[cfg(feature = "multi-reader")]
    pub(crate) fn mark_published(&self) -> io::Result<()> {
        match self.state.load(Ordering::Acquire) {
            Self::MATERIALIZING | Self::PUBLISHED => {
                self.state.store(Self::PUBLISHED, Ordering::Release);
                Ok(())
            }
            _ => Err(io::Error::other("transaction storage has not been applied")),
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
        data.applied[pool_index(pool)].push(false);
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
        data.applied[pool_index(pool)].extend(vec![false; entries.len()]);
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
        data.applied[pool_index(pool)].push(false);
        data.bytes = total;
        Ok(())
    }

    /// Append staged pool groups in dependency order: edges, event-DAG,
    /// then state. Does not fsync; the ordinary coalesced sync remains the
    /// durability boundary. Repeated calls are safe, including after a
    /// partial error.
    ///
    /// # Errors
    /// Returns an error if a staged pool has no coordinator or its journal
    /// cannot append the group's complete framing and trailer.
    pub fn publish(
        &self,
        coordinators: [Option<&JournalCoordinator>; ShardType::ALL.len()],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED | Self::JOURNAL_PUBLISHED | Self::MATERIALIZING | Self::PUBLISHED => {
                return Ok(())
            }
            _ => {}
        }
        if coordinators.iter().all(Option::is_none) {
            // Journaling is disabled process-wide, so there is no journal to
            // publish into.
            self.state.store(Self::PUBLISHED, Ordering::Release);
            return Ok(());
        }
        // Preserve the historical per-pool append order for the existing
        // pools; ServerInfo is appended after them without changing the WAL
        // ordering of legacy frames.
        let pools = [
            ShardType::Edges,
            ShardType::EventDag,
            ShardType::State,
            ShardType::ServerInfo,
        ];

        // A shared-WAL database gives every pool the same coordinator. Keep
        // the transaction as one journal group in that case, so readers never
        // observe only a prefix of a cross-pool commit. Standalone/per-pool
        // journals cannot provide that boundary and retain the ordered,
        // retry-safe fallback below.
        let active_pool_count = pools
            .iter()
            .filter(|pool| {
                let index = pool_index(**pool);
                !data.appended[index] && !data.pools[index].is_empty()
            })
            .count();
        if active_pool_count == 0 {
            self.state.store(Self::PUBLISHED, Ordering::Release);
            return Ok(());
        }
        let active_coordinators = pools
            .iter()
            .filter_map(|pool| {
                let index = pool_index(*pool);
                (!data.appended[index] && !data.pools[index].is_empty())
                    .then(|| coordinators[index])
            })
            .flatten()
            .collect::<Vec<_>>();
        if active_coordinators.len() == active_pool_count {
            if let Some(coordinator) = active_coordinators.first().copied() {
                if active_coordinators
                    .iter()
                    .all(|candidate| std::ptr::eq(*candidate, coordinator))
                {
                    let batches = pools
                        .iter()
                        .filter_map(|pool| {
                            let index = pool_index(*pool);
                            (!data.appended[index] && !data.pools[index].is_empty())
                                .then_some((*pool, data.pools[index].as_slice()))
                        })
                        .collect::<Vec<_>>();
                    let receipt = coordinator.publish_tagged_groups(&batches)?;
                    data.appended.fill(true);
                    data.receipt = Some(receipt);
                    self.mark_journal_published()?;
                    return Ok(());
                }
            }
        }

        if active_pool_count > 1 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-pool transactions require one shared journal coordinator",
            ));
        }

        for pool in pools {
            let index = pool_index(pool);
            if data.appended[index] || data.pools[index].is_empty() {
                data.appended[index] = true;
                continue;
            }
            let coordinator = coordinators[pool_index(pool)].ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotConnected, "staged pool has no journal")
            })?;
            let receipt = coordinator.publish_group_tagged(pool, &data.pools[index])?;
            data.appended[index] = true;
            data.receipt = Some(receipt);
        }
        self.mark_journal_published()?;
        Ok(())
    }
}

#[cfg(feature = "multi-reader")]
const fn pool_index(pool: ShardType) -> usize {
    match pool {
        ShardType::State => 0,
        ShardType::EventDag => 1,
        ShardType::Edges => 2,
        ShardType::ServerInfo => 3,
    }
}

/// Wire code for a pool tag in a pool-tagged (`FILE_VERSION_POOL_TAGGED`)
/// mutation frame's previously reserved flag byte.
///
/// `0` is reserved for "untagged", which only a version-2 segment may contain;
/// a version-3 frame must carry `1..=4`.
pub(crate) const fn pool_tag(pool: ShardType) -> u8 {
    match pool {
        ShardType::State => 1,
        ShardType::EventDag => 2,
        ShardType::Edges => 3,
        ShardType::ServerInfo => 4,
    }
}

/// Inverse of [`pool_tag`]. `0` (untagged) maps to `None`; any other
/// out-of-range code is rejected by the frame decoder.
pub(crate) const fn pool_from_tag(tag: u8) -> Option<ShardType> {
    match tag {
        1 => Some(ShardType::State),
        2 => Some(ShardType::EventDag),
        3 => Some(ShardType::Edges),
        4 => Some(ShardType::ServerInfo),
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
fn pool_extents(
    last_lsn: u64,
    pools: impl Iterator<Item = Option<ShardType>>,
) -> Vec<(ShardType, u64)> {
    let mut extents: Vec<(ShardType, u64)> = Vec::new();
    for pool in pools {
        let Some(pool) = pool else {
            continue;
        };
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

/// Bit in [`GroupMark::pools`] for a frame with no pool tag, which no pool's
/// coverage can account for.
const UNATTRIBUTED_POOL_BIT: u8 = 0x80;

/// What the group directory says about reclaiming a shared segment.
#[cfg(feature = "multi-reader")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedBoundary {
    /// The directory does not match the file, so it cannot decide.
    Untrusted,
    /// Even the first group is still needed (or the segment is empty).
    NothingCovered,
    /// Every group through this LSN may be dropped. The second field names the
    /// pool whose missing coverage stopped the cut (None when an untagged frame
    /// did, or when the whole segment is droppable).
    Through(u64, Option<ShardType>),
}

// Pool bits are `1 << index`, so they must stay below the unattributed bit.
const _: () = assert!(
    ShardType::ALL.len() < 7,
    "GroupMark::pools has one bit per pool below UNATTRIBUTED_POOL_BIT"
);

/// The bit standing for `pool` in [`GroupMark::pools`].
fn pool_bit(pool: Option<ShardType>) -> u8 {
    pool.and_then(|pool| ShardType::ALL.into_iter().position(|known| known == pool))
        .map_or(UNATTRIBUTED_POOL_BIT, |index| 1_u8 << index)
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
#[cfg(feature = "multi-reader")]
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
#[cfg(feature = "multi-reader")]
type BlockerRemediation = Arc<dyn Fn(ShardType) + Send + Sync>;

#[cfg(feature = "multi-reader")]
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
/// group's `last_lsn`, and the group holds no untagged frame (which no pool's
/// coverage can account for). `None` when even the first group is still needed.
///
/// The one place this rule lives: the directory and the scan fallback both call
/// it, so they cannot disagree.
#[cfg(feature = "multi-reader")]
fn boundary_through(
    groups: &[GroupMark],
    covered: &HashMap<ShardType, u64>,
) -> Option<(u64, Option<ShardType>)> {
    let mut boundary = None;
    for group in groups {
        for (index, pool) in ShardType::ALL.into_iter().enumerate() {
            if group.pools & (1_u8 << index) != 0
                && !covered.get(&pool).is_some_and(|lsn| *lsn >= group.last_lsn)
            {
                // This pool's coverage stops the cut here; report it as the
                // blocker so reclaim can say why the suffix was retained.
                return boundary.map(|lsn| (lsn, Some(pool)));
            }
        }
        if group.pools & UNATTRIBUTED_POOL_BIT != 0 {
            // No pool's coverage can account for an untagged frame: a real
            // blocker with no pool to name.
            return boundary.map(|lsn| (lsn, None));
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
    /// Bytes the rewrite moved and fsynced: the retained suffix, i.e. the
    /// rebuilt segment minus its header. This is what makes `copy`/`fsync`
    /// scale with segment size, so it is the number to bound.
    pub retained_bytes: u64,
    /// The pool whose missing coverage stopped the cut, so the suffix could not
    /// be dropped. None when an untagged frame stopped it, the whole segment
    /// was droppable, or the caller could not say. Diagnostics only.
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
    /// Generation of the newest durability mark written to this segment; the
    /// next mark uses the next generation, in the other slot.
    mark_generation: u64,
    /// On-disk format version of this segment.
    version: JournalVersion,
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
}

/// An fsync at least this slow is reported on stderr when it happens.
const SLOW_FSYNC_WARN: std::time::Duration = std::time::Duration::from_secs(1);

/// Cross-pool coverage bookkeeping for reclaiming a shared segment.
#[cfg(feature = "multi-reader")]
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
}

impl JournalCursor {
    /// The last LSN represented by the snapshot.
    #[must_use]
    pub const fn lsn(&self) -> u64 {
        self.lsn
    }
}

/// RAII pin that prevents journal reclaim from passing a replay cursor.
///
/// Keep this value alive while consuming pages from the cursor. Dropping it
/// releases the retained-history guarantee; it deliberately pins the original
/// cursor rather than advancing with individual pages, so an interrupted or
/// partially applied replay can safely resume from its starting snapshot.
#[must_use = "dropping the replay lease allows journal reclaim to expire its cursor"]
pub struct JournalReplayLease {
    registry: Arc<ReplayLeaseRegistry>,
    id: u64,
}

impl Drop for JournalReplayLease {
    fn drop(&mut self) {
        self.registry.pins.lock().remove(&self.id);
    }
}

#[derive(Default)]
struct ReplayLeaseRegistry {
    /// Lease ID to the last LSN represented by its snapshot cursor.
    pins: Mutex<HashMap<u64, u64>>,
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
    /// Set while a reclaim over the trigger cannot shrink the segment.
    #[cfg(feature = "multi-reader")]
    reclaim_stall: Mutex<Option<ReclaimStall>>,
    /// Advances whenever any pool's reported coverage does.
    #[cfg(feature = "multi-reader")]
    coverage_epoch: AtomicU64,
    /// Times a reclaim over the trigger has been found unable to shrink the
    /// segment.
    #[cfg(feature = "multi-reader")]
    reclaim_stalls: AtomicU64,
    /// Checkpoints the pools that hold a stalled reclaim back.
    #[cfg(feature = "multi-reader")]
    blocker_remediation: Mutex<Option<BlockerRemediation>>,
    /// `(segment length, coverage epoch)` at the last remediation attempt.
    #[cfg(feature = "multi-reader")]
    last_remediation: Mutex<Option<(u64, u64)>>,
    /// Optional shared, cross-pool group-sequence allocator. When present,
    /// each committed group draws its sequence from here instead of the
    /// segment's own counter, so groups across several pool segments share one
    /// global order. See [`Self::with_shared_sequence`].
    sequence: Option<Arc<AtomicU64>>,
    /// Per-pool durable coverage, used to reclaim a shared segment only up to
    /// the point every pool present in it has materialized. See
    /// [`Self::report_pool_coverage`] and [`Self::reclaim_shared`].
    #[cfg(feature = "multi-reader")]
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
    /// First LSNs of transaction groups that are published but whose writes are
    /// not yet all in the packs. See [`Self::publish_groups`].
    unmaterialized: Mutex<std::collections::BTreeSet<u64>>,
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
}

/// Directory-derived boundaries for one [`JournalCoordinator::changes_since`]
/// page: the base LSN to guard against a concurrent reclaim, the byte offset to
/// begin reading at, the offset to stop at (through at most `limit + 1` durable
/// groups), and whether the in-memory directory matched the file.
struct ChangesWindow {
    base_lsn: u64,
    start: u64,
    end: u64,
    directory_trusted: bool,
}

impl JournalCoordinator {
    /// Build a coordinator from an opened journal and its recovery scan.
    #[must_use]
    pub fn new(journal: Journal, scan: &Scan) -> Self {
        let committed_lsn = scan.groups.last().map_or(0, |group| group.last_lsn);
        let path = journal.path.clone();
        #[cfg(feature = "multi-reader")]
        let coverage = CoverageState::default();
        // Seed each pool's committed watermark from the recovered groups, so a
        // fresh coordinator over a pre-existing segment reports the same
        // coverage the segment already carries.
        let mut pool_committed: HashMap<ShardType, u64> = HashMap::new();
        for group in &scan.groups {
            for entry in &group.entries {
                if let Some(pool) = entry.pool {
                    let watermark = pool_committed.entry(pool).or_insert(0);
                    *watermark = (*watermark).max(group.last_lsn);
                }
            }
        }
        Self {
            journal: Mutex::new(journal),
            path,
            publication: Mutex::new(()),
            published_lsn: AtomicU64::new(committed_lsn),
            visible_lsn: AtomicU64::new(committed_lsn),
            committed_lsn: AtomicU64::new(committed_lsn),
            reclaim_trigger_len: AtomicU64::new(RECLAIM_TRIGGER_LEN),
            #[cfg(feature = "multi-reader")]
            reclaim_stall: Mutex::new(None),
            #[cfg(feature = "multi-reader")]
            coverage_epoch: AtomicU64::new(0),
            #[cfg(feature = "multi-reader")]
            reclaim_stalls: AtomicU64::new(0),
            #[cfg(feature = "multi-reader")]
            blocker_remediation: Mutex::new(None),
            #[cfg(feature = "multi-reader")]
            last_remediation: Mutex::new(None),
            sequence: None,
            #[cfg(feature = "multi-reader")]
            coverage: Mutex::new(coverage),
            pool_committed: Mutex::new(pool_committed),
            unmaterialized: Mutex::new(std::collections::BTreeSet::new()),
            pending_promotions: Mutex::new(Vec::new()),
            recovered: scan.groups.clone(),
            poisoned: AtomicBool::new(false),
            sync_calls: AtomicU64::new(0),
            journal_waiters: AtomicU64::new(0),
            coalesced_syncs: AtomicU64::new(0),
            sync_in_flight: AtomicU64::new(0),
            durable_requested: AtomicU64::new(committed_lsn),
            sync_lock: Mutex::new(()),
            #[cfg(test)]
            fsync_hook: Mutex::new(None),
            durable_lock: Mutex::new(()),
            durable_cv: Condvar::new(),
            background: Mutex::new(BackgroundState::Stopped),
            background_failure: Mutex::new(None),
            committer_wake_threshold: AtomicU64::new(0),
            commit_epoch: AtomicU64::new(0),
            threshold_notified_epoch: AtomicU64::new(u64::MAX),
            #[cfg(test)]
            committer_parks: AtomicU64::new(0),
            #[cfg(test)]
            threshold_wakes: AtomicU64::new(0),
            background_commits: AtomicU64::new(0),
            background_coalesced: AtomicU64::new(0),
            durable_requests: AtomicU64::new(0),
            durable_waits_already_durable: AtomicU64::new(0),
            durable_wait: DurableWaitTotals::default(),
            commits: AtomicU64::new(0),
            commit_records: AtomicU64::new(0),
            max_commit_records: AtomicU64::new(0),
            publish_signal: std::sync::OnceLock::new(),
            replay_leases: Arc::new(ReplayLeaseRegistry::default()),
            next_replay_lease: AtomicU64::new(1),
        }
    }

    /// The committed groups recovered when this coordinator's segment was
    /// opened, in sequence order. A store attaching to a coordinator it did
    /// not construct (the shared-WAL path) uses this to seed its replay set.
    #[must_use]
    pub fn recovered_groups(&self) -> Vec<CommittedGroup> {
        self.recovered.clone()
    }

    /// Create and map the cross-process publish signal beside this
    /// coordinator's segment. Idempotent; journal setup fails if the signal
    /// cannot be installed, since an old mapped signal could otherwise make
    /// workers skip refreshes after this writer starts publishing.
    ///
    /// The store that enables a journal calls this so read-only workers can
    /// sample `(epoch, visible_lsn)` with a plain atomic load instead of a
    /// per-call `fs::metadata`. A store that never calls it leaves the signal
    /// absent, and workers fall back to the stat-based refresh.
    pub(crate) fn enable_publish_signal(&self) -> io::Result<()> {
        let result = self.publish_signal.get_or_init(|| {
            PublishSignal::writer(&self.path)
                .map(Arc::new)
                .map_err(|error| (error.kind(), error.to_string()))
        });
        result
            .as_ref()
            .map(|_| ())
            .map_err(|(kind, message)| io::Error::new(*kind, message.clone()))
    }

    /// The publish signal, if [`Self::enable_publish_signal`] created one.
    #[must_use]
    pub(crate) fn publish_signal(&self) -> Option<&Arc<PublishSignal>> {
        self.publish_signal
            .get()
            .and_then(|result| result.as_ref().ok())
    }

    /// Advance the publish signal so a gated reader rescans. Called on a group
    /// publication and on a reclaim, the two events a reader must notice.
    fn bump_publish_signal(&self) {
        if let Some(signal) = self.publish_signal() {
            signal.bump();
        }
    }

    /// Record that `pool`'s durable checkpoint has materialized every frame it
    /// owns through `lsn`. Coverage only advances. A pool with no reported
    /// coverage never blocks a group that does not carry its frames.
    #[cfg(feature = "multi-reader")]
    pub fn report_pool_coverage(&self, pool: ShardType, lsn: u64) {
        let mut coverage = self.coverage.lock();
        let entry = coverage.covered.entry(pool).or_insert(0);
        if lsn > *entry {
            *entry = lsn;
            self.coverage_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Highest committed group `last_lsn` that carried at least one frame for
    /// `pool` and is fully in the packs, or 0 when there is none.
    ///
    /// A pool checkpoint must record this, not [`Self::committed_lsn`]. One
    /// shared group can interleave several pools' frames, so the global
    /// committed LSN can run ahead of what this pool's index has materialized;
    /// recording it would claim coverage the checkpoint does not have and let
    /// reclaim drop another pool's uncovered frames.
    #[must_use]
    #[cfg(feature = "multi-reader")]
    pub fn committed_lsn_for_pool(&self, pool: ShardType) -> u64 {
        let committed = self.pool_committed.lock().get(&pool).copied().unwrap_or(0);
        // A published transaction whose writes are not all in the packs yet
        // holds coverage below its first frame.
        match self.unmaterialized.lock().first() {
            Some(first_lsn) => committed.min(first_lsn.saturating_sub(1)),
            None => committed,
        }
    }

    /// Record that the transaction group whose first frame is `first_lsn` has
    /// been fully written to the packs, so coverage may pass it.
    #[cfg(feature = "multi-reader")]
    pub fn transaction_materialized(&self, first_lsn: u64) {
        self.unmaterialized.lock().remove(&first_lsn);
    }

    /// Record a durable commit and advance the generation that re-arms the
    /// background committer's threshold wake. Centralised so every commit path
    /// — background, explicit `sync_through`, and fallback — re-arms alike.
    fn note_commit(&self, through_lsn: u64) {
        let previous = self.committed_lsn.fetch_max(through_lsn, Ordering::AcqRel);
        if through_lsn > previous {
            let covered = through_lsn.saturating_sub(previous);
            self.commits.fetch_add(1, Ordering::Relaxed);
            self.commit_records.fetch_add(covered, Ordering::Relaxed);
            self.max_commit_records
                .fetch_max(covered, Ordering::Relaxed);
            self.commit_epoch.fetch_add(1, Ordering::AcqRel);
            self.wake_committer_if_backlogged();
        }
    }

    /// Advance per-pool committed watermarks for every appended group whose
    /// `last_lsn` is at or below `through_lsn`.
    fn promote_pool_committed(&self, through_lsn: u64) {
        let mut promotions = self.pending_promotions.lock();
        let mut watermarks = self.pool_committed.lock();
        let (ready, pending): (Vec<PendingPromotion>, Vec<PendingPromotion>) = promotions
            .drain(..)
            .partition(|(last_lsn, _)| *last_lsn <= through_lsn);
        *promotions = pending;
        for (_, extents) in ready {
            for (pool, lsn) in extents {
                let entry = watermarks.entry(pool).or_insert(0);
                *entry = (*entry).max(lsn);
            }
        }
    }

    /// Reclaim the longest prefix of the shared segment whose every group is
    /// durably covered by all of that group's own pools, returning the reclaim
    /// result or `None` when nothing is reclaimable yet.
    ///
    /// A group is reclaimed atomically, so it may only be dropped once every
    /// pool that contributed a frame to it has reported coverage through the
    /// group's `last_lsn`. The pools in one group are independent of the pools
    /// in another, so the reclaimable prefix is **not** a single minimum across
    /// every pool: an idle pool whose last frame sits in an early group must not
    /// pin later groups that its frames never reached. Walk the live groups in
    /// order and stop at the first one any contributing pool has not covered.
    ///
    /// # Errors
    /// Propagates a scan or segment-rewrite failure from [`Self::reclaim_through`].
    #[cfg(feature = "multi-reader")]
    pub fn reclaim_shared(&self) -> io::Result<Option<Reclaim>> {
        // The group directory answers this without reading the segment. The
        // lock order matches `reclaim_through`: the sync lock, then the journal,
        // and coverage last.
        {
            let _sync = self.sync_lock.lock();
            let mut journal = self.journal.lock();
            let boundary = {
                let coverage = self.coverage.lock();
                journal.shared_reclaim_boundary(&coverage.covered)
            };
            match boundary {
                SharedBoundary::Through(covered_lsn, blocked_by) => {
                    // Shared reclaim computes its own per-group/per-pool
                    // boundary, so apply the replay floor after that decision
                    // and before the segment rewrite. The lease is global to
                    // this segment and therefore protects groups from every
                    // pool, including groups whose pool coverage is complete.
                    let replay_limit = self.replay_reclaim_limit(covered_lsn);
                    let lease_limited = replay_limit < covered_lsn;
                    let blocked_by = (!lease_limited).then_some(blocked_by).flatten();
                    let reclaimed = journal.reclaim_through_with_blocker(replay_limit, blocked_by);
                    if reclaimed.is_ok() {
                        self.bump_publish_signal();
                    }
                    let reclaimed = reclaimed.map(Some);
                    if !lease_limited {
                        // A cut held back only by an active replay lease is not a
                        // pool-coverage stall. Recording one would drive forced
                        // checkpoints and warning logs for as long as a rebuild
                        // pins its own window; the next reclaim after the lease
                        // is dropped records the real outcome.
                        let coverage = self.coverage.lock();
                        self.record_reclaim_outcome(&journal, &coverage.covered);
                    }
                    return reclaimed;
                }
                SharedBoundary::NothingCovered => {
                    let coverage = self.coverage.lock();
                    self.record_reclaim_outcome(&journal, &coverage.covered);
                    return Ok(None);
                }
                SharedBoundary::Untrusted => {}
            }
        }
        // The directory did not match the file: decide from a scan of it.
        let scan = Journal::scan_read_only(&self.path)?;
        let marks = marks_from_scan(&scan);
        let boundary = {
            let coverage = self.coverage.lock();
            boundary_through(&marks, &coverage.covered)
        };
        match boundary {
            Some((covered_lsn, blocked_by)) => self
                .reclaim_through_with_blocker(covered_lsn, blocked_by)
                .map(Some),
            None => Ok(None),
        }
    }

    /// The pools whose missing coverage holds reclaim back right now, oldest
    /// needed group first. Empty when nothing is blocked or it cannot be told.
    #[cfg(feature = "multi-reader")]
    #[must_use]
    pub fn reclaim_blockers(&self) -> Vec<ShardType> {
        let journal = self.journal.lock();
        let coverage = self.coverage.lock();
        journal.blocking_pools(&coverage.covered)
    }

    /// Whether the last reclaim left the segment over the trigger.
    #[cfg(feature = "multi-reader")]
    #[must_use]
    pub fn is_reclaim_stalled(&self) -> bool {
        self.reclaim_stall.lock().is_some()
    }

    /// Times a reclaim has been found unable to shrink a segment over the
    /// trigger (each stall counts once, until a reclaim clears it).
    #[cfg(feature = "multi-reader")]
    #[must_use]
    pub fn reclaim_stalls(&self) -> u64 {
        self.reclaim_stalls.load(Ordering::Relaxed)
    }

    /// Give the coordinator a way to make one named pool checkpoint (and so
    /// report coverage). It is used only in the emergency zone, to un-stall a
    /// reclaim held back by a pool nobody is syncing.
    #[cfg(feature = "multi-reader")]
    pub fn set_blocker_remediation(&self, remediate: impl Fn(ShardType) + Send + Sync + 'static) {
        *self.blocker_remediation.lock() = Some(Arc::new(remediate));
    }

    /// Note how a reclaim ended, with the journal and sync locks held. A
    /// segment still over the trigger means the reclaim is stalled: remember
    /// it, and say so once, naming the pools it is waiting on.
    #[cfg(feature = "multi-reader")]
    fn record_reclaim_outcome(&self, journal: &Journal, covered: &HashMap<ShardType, u64>) {
        let len = journal.file_len;
        if len <= self.reclaim_trigger_len() {
            *self.reclaim_stall.lock() = None;
            *self.last_remediation.lock() = None;
            return;
        }
        let epoch = self.coverage_epoch.load(Ordering::Acquire);
        let mut stall = self.reclaim_stall.lock();
        let previous = *stall;
        let emergency = len >= journal.segment_cap.saturating_sub(journal.segment_cap / 4);
        // The first reclaim to leave the segment over the trigger is normal:
        // pools sync one after another, so the first to cross it cannot yet
        // reclaim what the others have not reported. It is only worth saying
        // once the segment has grown by an eighth of the trigger without a
        // reclaim getting it back under the trigger, or it reaches the
        // emergency zone. The
        // back-off and remediation below act on the stall from the start.
        let grace_from = previous.map_or(len, |old| old.grace_from);
        let already_reported = previous.is_some_and(|old| old.reported);
        let report = !already_reported
            && (emergency || len >= grace_from.saturating_add(self.reclaim_retry_growth()));
        *stall = Some(ReclaimStall {
            at_len: len,
            coverage_epoch: epoch,
            grace_from,
            reported: already_reported || report,
        });
        let entered_emergency = emergency
            && previous.is_some_and(|old| {
                old.at_len < journal.segment_cap.saturating_sub(journal.segment_cap / 4)
            });
        if report {
            self.reclaim_stalls.fetch_add(1, Ordering::Relaxed);
        }
        if report || (already_reported && entered_emergency) {
            eprintln!(
                "warning: shared WAL reclaim is stalled at {len} of {} bytes (trigger {}); \
                 waiting on pools {:?} to report coverage",
                journal.segment_cap,
                self.reclaim_trigger_len(),
                journal.blocking_pools(covered)
            );
        }
    }

    /// If a stalled reclaim has reached the emergency zone, make the pools
    /// holding it back checkpoint, except `caller`, which is mid-sync. Runs
    /// with no journal or persistence lock held, at most once per further
    /// eighth of the trigger of growth unless coverage moved, and never from
    /// inside another remediation.
    #[cfg(feature = "multi-reader")]
    pub fn remediate_blockers(&self, caller: Option<ShardType>) {
        if REMEDIATING.with(std::cell::Cell::get) {
            return;
        }
        let len = self.segment_len();
        if len < self.emergency_len() || !self.is_reclaim_stalled() {
            return;
        }
        let epoch = self.coverage_epoch.load(Ordering::Acquire);
        {
            let mut last = self.last_remediation.lock();
            if let Some((last_len, last_epoch)) = *last {
                if last_epoch == epoch && len < last_len.saturating_add(self.reclaim_retry_growth())
                {
                    return;
                }
            }
            *last = Some((len, epoch));
        }
        let Some(remediate) = self.blocker_remediation.lock().clone() else {
            return;
        };
        REMEDIATING.with(|flag| flag.set(true));
        for pool in self.reclaim_blockers() {
            if Some(pool) != caller {
                remediate(pool);
            }
        }
        REMEDIATING.with(|flag| flag.set(false));
    }

    /// The error for a commit refused because the segment is full, naming the
    /// pools reclaim is waiting on when that is known.
    fn explain_full_segment(&self, error: io::Error, journal: &Journal) -> io::Error {
        #[cfg(feature = "multi-reader")]
        if error.kind() == io::ErrorKind::WouldBlock {
            let blockers = journal.blocking_pools(&self.coverage.lock().covered);
            if !blockers.is_empty() {
                return io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("{error}; reclaim is waiting on pools {blockers:?} to report coverage"),
                );
            }
        }
        #[cfg(not(feature = "multi-reader"))]
        let _ = (self, journal);
        error
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
    #[cfg(feature = "multi-reader")]
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

    /// Create a replay cursor at a materialized LSN for this journal.
    ///
    /// The caller must establish that the snapshot contains the records through
    /// `lsn` before creating the cursor. The cursor is scoped to this journal
    /// path and writer incarnation. It must be reacquired after a writer
    /// restart; if reclaim has already removed required history,
    /// [`Self::changes_since`] reports an expired-cursor error.
    pub(crate) fn replay_cursor(&self, lsn: u64) -> io::Result<JournalCursor> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay cursors require an installed publish signal",
            )
        })?;
        Ok(JournalCursor {
            path: self.path.clone(),
            incarnation: signal.snapshot().0,
            lsn,
        })
    }

    /// Pin the journal history required by `cursor` until the returned lease
    /// is dropped. The pin and reclaim share the `sync_lock` -> `journal` lock
    /// order, so reclaim either observes this lease or completes first and
    /// causes an explicit expired-cursor error here.
    pub(crate) fn pin_replay_cursor(
        &self,
        cursor: &JournalCursor,
    ) -> io::Result<JournalReplayLease> {
        if cursor.path != self.path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal cursor belongs to a different segment",
            ));
        }
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay leases require an installed publish signal",
            )
        })?;

        let _sync = self.sync_lock.lock();
        let journal = self.journal.lock();
        if signal.snapshot().0 != cursor.incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal cursor belongs to a previous writer incarnation",
            ));
        }
        if cursor.lsn.saturating_add(1) < journal.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "journal cursor expired at LSN {}; retained history starts at {}",
                    cursor.lsn, journal.base_lsn
                ),
            ));
        }

        let mut next = self.next_replay_lease.load(Ordering::Relaxed);
        let id = loop {
            let following = next
                .checked_add(1)
                .ok_or_else(|| io::Error::other("journal replay lease IDs exhausted"))?;
            match self.next_replay_lease.compare_exchange_weak(
                next,
                following,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break next,
                Err(observed) => next = observed,
            }
        };
        self.replay_leases.pins.lock().insert(id, cursor.lsn);
        Ok(JournalReplayLease {
            registry: Arc::clone(&self.replay_leases),
            id,
        })
    }

    /// Highest LSN reclaim may pass without expiring an active replay cursor.
    fn replay_reclaim_limit(&self, covered_lsn: u64) -> u64 {
        self.replay_leases
            .pins
            .lock()
            .values()
            .copied()
            .min()
            .map_or(covered_lsn, |pin| covered_lsn.min(pin))
    }

    /// Read a bounded page of complete, durable groups after `cursor`.
    ///
    /// Pages preserve group boundaries and journal order. The durable horizon
    /// is sampled before the segment is read and returned as
    /// [`JournalChangesPage::horizon_lsn`]: `has_more` and the returned groups
    /// are complete with respect to that horizon, so a group that becomes
    /// durable after the sample is never silently skipped. Reclaim is allowed
    /// to remove history; if it has advanced past the cursor, this returns an
    /// `InvalidData` error so the caller can discard its partial rebuild and
    /// take a fresh snapshot. This method does not retain a lease across page
    /// calls. Hold a [`JournalReplayLease`] from the snapshot cursor while
    /// paging if reclaim must not expire it during a long replay.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a foreign cursor or zero page size, and
    /// `InvalidData` when the writer incarnation differs or the cursor's
    /// required history has been reclaimed.
    pub fn changes_since(
        &self,
        cursor: &JournalCursor,
        limit: usize,
    ) -> io::Result<JournalChangesPage> {
        self.validate_changes_request(cursor, limit)?;
        let incarnation = self.replay_incarnation(cursor)?;

        // Sample the durable high-water mark *before* reading the WAL. A group
        // at or below it was appended and fsynced before this read, so the
        // full-segment scan below cannot miss it. Sampling after the read could
        // include a group appended in between that the scan never saw, letting
        // the page report `has_more == false` with a durable group outstanding.
        let durable_lsn = self.committed_lsn();

        // Capture the directory boundary and the byte range needed for at most
        // `limit + 1` durable groups. The extra group determines `has_more`.
        // Holding the journal lock makes these offsets consistent with reclaim
        // and append; the actual I/O happens after releasing it.
        let window = self.changes_window(cursor, limit, durable_lsn)?;

        // A test can force a commit after the horizon and byte range are fixed
        // but before the WAL is read. That group belongs to a later pass.
        #[cfg(test)]
        if let Some(hook) = take_changes_since_before_read_hook() {
            hook();
        }

        let scan = self.scan_changes(cursor, &window)?;
        self.validate_changes_scan(cursor, incarnation, &window, &scan)?;
        Ok(self.changes_page(cursor, incarnation, durable_lsn, limit, scan))
    }

    /// Validate a [`Self::changes_since`] request's cursor and page size.
    fn validate_changes_request(&self, cursor: &JournalCursor, limit: usize) -> io::Result<()> {
        if cursor.path != self.path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal cursor belongs to a different segment",
            ));
        }
        if limit == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal changes page size must be nonzero",
            ));
        }
        Ok(())
    }

    /// The writer incarnation a [`Self::changes_since`] cursor must match, or an
    /// error when no publish signal is installed or the cursor is from a
    /// previous writer.
    fn replay_incarnation(&self, cursor: &JournalCursor) -> io::Result<u64> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay requires an installed publish signal",
            )
        })?;
        let incarnation = signal.snapshot().0;
        if cursor.incarnation != incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal cursor belongs to a previous writer incarnation",
            ));
        }
        Ok(incarnation)
    }

    /// Capture the [`ChangesWindow`] for a page: the directory-derived byte
    /// range through at most `limit + 1` durable groups, the base LSN to guard
    /// against a concurrent reclaim, and whether the in-memory directory
    /// matched the file.
    fn changes_window(
        &self,
        cursor: &JournalCursor,
        limit: usize,
        durable_lsn: u64,
    ) -> io::Result<ChangesWindow> {
        let journal = self.journal.lock();
        if cursor.lsn.saturating_add(1) < journal.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "journal cursor expired at LSN {}; retained history starts at {}",
                    cursor.lsn, journal.base_lsn
                ),
            ));
        }
        let first = journal
            .groups
            .partition_point(|group| group.last_lsn <= cursor.lsn);
        let start = first
            .checked_sub(1)
            .and_then(|index| journal.groups.get(index))
            .map_or(FILE_HEADER_LEN as u64, |group| group.end_offset);
        let mut durable_groups = journal
            .groups
            .iter()
            .skip(first)
            .take_while(|group| group.last_lsn <= durable_lsn);
        let end = durable_groups
            .by_ref()
            .take(limit.saturating_add(1))
            .last()
            .map_or(start, |group| group.end_offset);
        Ok(ChangesWindow {
            base_lsn: journal.base_lsn,
            start,
            end,
            directory_trusted: journal.directory_matches_file(),
        })
    }

    /// Read the segment region a [`Self::changes_since`] page needs: the bounded
    /// tail when the directory was trustworthy, else the whole segment.
    fn scan_changes(&self, cursor: &JournalCursor, window: &ChangesWindow) -> io::Result<Scan> {
        if window.directory_trusted {
            Journal::scan_read_only_range(
                &self.path,
                window.start,
                window.end,
                cursor.lsn.saturating_add(1),
            )
        } else {
            Journal::scan_read_only(&self.path)
        }
    }

    /// Re-check the writer incarnation and the reclaim guard after the WAL read,
    /// so a restart or in-place reclaim between capture and read is not mistaken
    /// for stable history.
    fn validate_changes_scan(
        &self,
        cursor: &JournalCursor,
        incarnation: u64,
        window: &ChangesWindow,
        scan: &Scan,
    ) -> io::Result<()> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay requires an installed publish signal",
            )
        })?;
        if signal.snapshot().0 != incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal writer restarted while reading changes",
            ));
        }
        if scan.base_lsn != window.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal segment was reclaimed while reading changes; retry from the cursor",
            ));
        }
        if cursor.lsn.saturating_add(1) < scan.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "journal cursor expired at LSN {}; retained history starts at {}",
                    cursor.lsn, scan.base_lsn
                ),
            ));
        }
        Ok(())
    }

    /// Select the durable groups after `cursor` into a page capped at `limit`,
    /// and derive the next cursor and `has_more` from one extra group.
    fn changes_page(
        &self,
        cursor: &JournalCursor,
        incarnation: u64,
        durable_lsn: u64,
        limit: usize,
        scan: Scan,
    ) -> JournalChangesPage {
        let mut following = scan
            .groups
            .into_iter()
            .filter(|group| group.last_lsn > cursor.lsn && group.last_lsn <= durable_lsn);
        let selected: Vec<CommittedGroup> = following.by_ref().take(limit).collect();
        let has_more = following.next().is_some();
        let through_lsn = selected.last().map_or(cursor.lsn, |group| group.last_lsn);
        JournalChangesPage {
            groups: selected,
            next_cursor: JournalCursor {
                path: self.path.clone(),
                incarnation,
                lsn: through_lsn,
            },
            through_lsn,
            has_more,
            horizon_lsn: durable_lsn,
        }
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
    /// `sync_lock` serializes the fsync (the journal mutex is held only briefly
    /// to capture the range, so publishers keep appending); concurrent sync
    /// callers wait for it and then recheck the committed LSN before deciding
    /// whether to flush.
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
        if target_lsn > self.published_lsn.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sync target has not been published",
            ));
        }
        if self.covered_by_prior_commit(target_lsn, &mut timings) {
            return Ok((None, timings));
        }
        self.reject_if_poisoned()?;

        let _sync = self.acquire_sync_lock(&mut timings);
        self.reject_if_poisoned()?;
        // A concurrent fsync that finished while we waited usually covered us.
        if self.covered_by_prior_commit(target_lsn, &mut timings) {
            return Ok((None, timings));
        }
        let previously_committed = self.committed_lsn.load(Ordering::Acquire);

        let capture = self.capture_sync_range(target_lsn)?;
        self.flush_captured_file(&capture, &mut timings)?;
        let receipt = self.record_durable_range(previously_committed, &capture, &mut timings);
        Ok((Some(receipt), timings))
    }

    /// Whether an earlier commit already made `target_lsn` durable. A `0`
    /// target is always covered. Records the coalescing in `timings` and the
    /// coordinator's counter when it is.
    fn covered_by_prior_commit(&self, target_lsn: u64, timings: &mut JournalSyncTimings) -> bool {
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            timings.journal_coalesced = true;
            self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Fail if a failed commit or append poisoned the journal.
    fn reject_if_poisoned(&self) -> io::Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        Ok(())
    }

    /// Take [`Self::sync_lock`], counting this caller as a waiter when another
    /// fsync is in flight, and record how long the wait took.
    fn acquire_sync_lock(&self, timings: &mut JournalSyncTimings) -> MutexGuard<'_, ()> {
        let lock_started = std::time::Instant::now();
        let guard = if let Some(guard) = self.sync_lock.try_lock() {
            guard
        } else {
            timings.journal_waiter = true;
            self.journal_waiters.fetch_add(1, Ordering::Relaxed);
            self.sync_lock.lock()
        };
        timings.journal_lock_wait = lock_started.elapsed();
        guard
    }

    /// Capture what a sync needs, under the journal lock.
    ///
    /// Every published group is already complete in the segment, so the
    /// sync only needs to make its bytes durable. Under the journal lock,
    /// capture the newest visible LSN (everything at or below it has been
    /// fully written), the group sequence, and a handle to the current
    /// file; then release the lock so publishers run during the fsync.
    /// Syncing through the newest visible LSN, not just `target_lsn`, lets
    /// one fsync cover every caller that queued behind the previous one.
    fn capture_sync_range(&self, target_lsn: u64) -> io::Result<SyncCapture> {
        let mut journal = self.journal.lock();
        if journal.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        let through_lsn = self.visible_lsn.load(Ordering::Acquire);
        if target_lsn > through_lsn {
            return Err(io::Error::other(
                "published sync target is not yet visible in the journal",
            ));
        }
        let file = match journal.file.try_clone() {
            Ok(file) => file,
            Err(error) => {
                self.poisoned.store(true, Ordering::Release);
                journal.poisoned = true;
                return Err(error);
            }
        };
        Ok(SyncCapture {
            file,
            through_lsn,
            sequence: journal.next_sequence.saturating_sub(1),
            path: journal.path.clone(),
            file_len: journal.file_len,
        })
    }

    /// Fsync the captured handle with no journal lock held, recording the
    /// fsync time in `timings`.
    fn flush_captured_file(
        &self,
        capture: &SyncCapture,
        timings: &mut JournalSyncTimings,
    ) -> io::Result<()> {
        let fsync_started = std::time::Instant::now();
        #[cfg(test)]
        let hooked = self.run_fsync_hook();
        #[cfg(not(test))]
        let hooked: io::Result<()> = Ok(());
        if let Err(error) = hooked.and_then(|()| capture.file.sync_all()) {
            // Never retry a failed fsync and report success: the kernel may
            // have dropped the dirty pages. Poison until reopen and rescan.
            self.poisoned.store(true, Ordering::Release);
            self.journal.lock().poisoned = true;
            return Err(error);
        }
        timings.journal_fsync = fsync_started.elapsed();
        Ok(())
    }

    /// Run the test-only fsync hook, if one is set, so a test can park the
    /// sync before its fsync or make it fail.
    #[cfg(test)]
    fn run_fsync_hook(&self) -> io::Result<()> {
        let hook = self.fsync_hook.lock().clone();
        match hook {
            Some(hook) => hook(),
            None => Ok(()),
        }
    }

    /// Publish the durable boundary a successful fsync established and build
    /// the receipt for the range this call advanced.
    fn record_durable_range(
        &self,
        previously_committed: u64,
        capture: &SyncCapture,
        timings: &mut JournalSyncTimings,
    ) -> CommitReceipt {
        timings.journal_records = capture.through_lsn.saturating_sub(previously_committed);
        self.warn_if_slow_fsync(capture.path.as_path(), capture.through_lsn, timings);
        self.note_commit(capture.through_lsn);
        self.promote_pool_committed(capture.through_lsn);
        // The fsync has returned, so every byte up to the captured length is
        // durable. Record that, after the fact and without an fsync of its own,
        // so recovery can tell a crash tail above it from corruption below it.
        self.journal.lock().write_mark(capture.file_len);
        // The groups were appended at publication, so this fsync wrote no
        // bytes: the receipt reports the durable range this call advanced.
        CommitReceipt {
            sequence: capture.sequence,
            first_lsn: previously_committed.saturating_add(1),
            last_lsn: capture.through_lsn,
            bytes_written: 0,
        }
    }

    /// Report an fsync slower than [`SLOW_FSYNC_WARN`] as it happens, with the
    /// lock wait and waiter count needed to tell contention from disk latency.
    fn warn_if_slow_fsync(&self, path: &Path, target_lsn: u64, timings: &JournalSyncTimings) {
        if timings.journal_fsync >= SLOW_FSYNC_WARN {
            let _ = writeln!(
                io::stderr().lock(),
                "mtxdb: slow WAL fsync {}ms (through lsn {target_lsn}, lock wait {}ms, {} records, in-flight {}, waiters {}, {})",
                timings.journal_fsync.as_millis(),
                timings.journal_lock_wait.as_millis(),
                timings.journal_records,
                timings.journal_in_flight,
                self.journal_waiters.load(Ordering::Relaxed),
                path.display(),
            );
        }
    }

    /// Publish one autocommit mutation as its own complete group, without
    /// fsyncing it.
    ///
    /// The LSN is assigned under the publication lock and the group is
    /// visible to read-only overlays when this returns. Durability is
    /// separate: a later [`Self::sync_through`] or the background committer
    /// fsyncs every visible group at once, so many small groups still share
    /// one fsync.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned, its LSN space is
    /// exhausted, or the append fails. A partial append poisons the
    /// underlying journal; publication is rejected until reopen/recovery.
    pub fn publish_group(&self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        self.publish_groups(&[(None, mutations)], false)?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cannot publish an empty group")
            })
    }

    /// Like [`Self::publish_group`], tagging every mutation with `pool`.
    /// Required when the coordinator's segment is pool-tagged; the pool-less
    /// form would be rejected there.
    ///
    /// # Errors
    /// Same as [`Self::publish_group`].
    #[cfg(feature = "multi-reader")]
    pub fn publish_group_tagged(
        &self,
        pool: ShardType,
        mutations: &[Mutation],
    ) -> io::Result<CommitReceipt> {
        self.publish_groups(&[(Some(pool), mutations)], false)?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cannot publish an empty group")
            })
    }

    /// Publish several pool-tagged transaction batches as one visible journal
    /// group.
    ///
    /// This is the atomic publication boundary for a transaction that writes
    /// multiple pools through a shared coordinator. The caller must provide
    /// each pool at most once; empty batches are ignored.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned or the group cannot be
    /// appended.
    #[cfg(feature = "multi-reader")]
    pub fn publish_tagged_groups(
        &self,
        batches: &[(ShardType, &[Mutation])],
    ) -> io::Result<CommitReceipt> {
        let staged = batches
            .iter()
            .map(|(pool, mutations)| (Some(*pool), *mutations))
            .collect::<Vec<_>>();
        self.publish_groups(&staged, true)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot publish an empty transaction group",
            )
        })
    }

    /// Single publication path. Holds `publication` for the whole append so
    /// LSN assignment, the append and the visibility advance are one step.
    ///
    /// A transaction group (`transaction` set) is published before its writes
    /// are materialized into the packs. Until [`Self::transaction_materialized`]
    /// says otherwise, no pool may claim coverage at or past its first frame:
    /// the packs do not hold those frames yet, so a claim (and the reclaim it
    /// would license) could lose an acknowledged commit to a crash. It is
    /// registered here, under the journal lock, so no sync can promote a
    /// watermark past the group before the group is on the floor list.
    fn publish_groups(
        &self,
        staged: &[(Option<ShardType>, &[Mutation])],
        transaction: bool,
    ) -> io::Result<Option<CommitReceipt>> {
        let _publication = self.publication.lock();
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
        let group_mutations: Vec<(Option<ShardType>, Mutation)> = staged
            .iter()
            .flat_map(|(pool, mutations)| mutations.iter().cloned().map(|m| (*pool, m)))
            .collect();
        if group_mutations.is_empty() {
            return Ok(None);
        }
        let expected_first_lsn = journal.next_lsn;
        let expected_count = u64::try_from(group_mutations.len()).unwrap_or(u64::MAX);
        let sequence = self
            .sequence
            .as_ref()
            .map(|counter| counter.fetch_add(1, Ordering::Relaxed));
        let receipt = match journal.append_group_for_current_mode(&group_mutations, sequence) {
            Ok(receipt) => receipt,
            Err(error) => {
                if journal.poisoned {
                    self.poisoned.store(true, Ordering::Release);
                }
                return Err(self.explain_full_segment(error, &journal));
            }
        };
        if transaction {
            self.unmaterialized.lock().insert(receipt.first_lsn);
        }
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
            "appended group must cover every mutation"
        );
        // Queue this group's per-pool extents before it becomes visible, still
        // under the journal lock. A sync that captures the new visible LSN
        // then always finds the extents and promotes each pool's watermark;
        // queuing after the visibility advance would let a sync race past them.
        let extents = pool_extents(
            receipt.last_lsn,
            group_mutations.iter().map(|(pool, _)| *pool),
        );
        if !extents.is_empty() {
            self.pending_promotions
                .lock()
                .push((receipt.last_lsn, extents));
        }
        self.published_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        self.visible_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        // Publish the read-committed change boundary for cross-process workers
        // at the exact point the group becomes visible, before any fsync: a
        // reader may observe a published-but-not-yet-durable group.
        self.bump_publish_signal();
        drop(journal);
        // Wake the committer only when a burst is at or above its early-flush
        // bound; this is the hot path, so ordinary publishes must not pay a
        // futex wake. `committer_wake_threshold` is zero when no committer is
        // running. Only the first publish at or above the bound wakes in a
        // commit epoch; any commit advances the epoch and re-arms the next
        // crossing.
        let threshold = self.committer_wake_threshold.load(Ordering::Relaxed);
        if threshold > 0 && self.pending_count() >= threshold && self.claim_threshold_wake() {
            #[cfg(test)]
            self.threshold_wakes.fetch_add(1, Ordering::Relaxed);
            self.wake_durable_waiters();
        }
        Ok(Some(receipt))
    }

    /// Capture the current boundary and wait for a durable group covering it.
    ///
    /// # Errors
    /// Returns an error if committing the captured boundary fails.
    pub fn sync(&self) -> io::Result<Option<CommitReceipt>> {
        let target = self.capture_sync_target();
        self.sync_through(target)
    }

    /// Register `target_lsn` as a durability request without blocking.
    ///
    /// Returns a [`DurabilityToken`] to pass to [`Self::wait_durable`]. The
    /// request is additive: it never weakens [`Self::sync_through`] /
    /// [`Self::sync`], which remain immediate barriers. With a background
    /// committer running, `wait_durable` blocks until the committer's group
    /// covers the token; without one it performs the blocking commit itself.
    #[must_use]
    pub fn request_durable(&self, target_lsn: u64) -> DurabilityToken {
        self.durable_requests.fetch_add(1, Ordering::Relaxed);
        self.durable_requested
            .fetch_max(target_lsn, Ordering::AcqRel);
        // Wake the committer (for a threshold flush) and any waiters parked on
        // the old boundary.
        self.wake_durable_waiters();
        DurabilityToken { lsn: target_lsn }
    }

    /// Current length of the journal segment file, including its header.
    #[must_use]
    pub fn segment_len(&self) -> u64 {
        self.journal.lock().file_len
    }

    /// Largest the segment may grow before commits are refused.
    #[must_use]
    pub fn segment_cap(&self) -> u64 {
        self.journal.lock().segment_cap
    }

    /// The segment length past which a stalled reclaim is treated as an
    /// emergency: three quarters of the cap.
    fn emergency_len(&self) -> u64 {
        let cap = self.segment_cap();
        cap.saturating_sub(cap / 4)
    }

    /// Whether the segment has reached the emergency zone, where a checkpoint
    /// tail still in flight must be waited for rather than let run on.
    #[must_use]
    pub fn in_emergency_zone(&self) -> bool {
        self.segment_len() >= self.emergency_len()
    }

    /// How much a stalled segment must grow before the next forced checkpoint
    /// is worth trying: an eighth of the trigger.
    #[cfg(feature = "multi-reader")]
    fn reclaim_retry_growth(&self) -> u64 {
        self.reclaim_trigger_len() / 8
    }

    /// Whether a sync of `pool` should force a reclaiming checkpoint now.
    ///
    /// True once the segment is over the trigger, except while a reclaim is
    /// known to be stalled. Then a pool that is not itself holding the reclaim
    /// back would only repeat a checkpoint that cannot help, so it waits until
    /// the segment has grown by an eighth of the trigger or some pool's
    /// coverage advanced. A pool that is holding it back always checkpoints:
    /// that is what un-stalls it. The emergency zone does not lift the wait:
    /// measured, it only repeated checkpoints that reclaimed nothing.
    #[must_use]
    pub fn should_force_reclaim_checkpoint(&self, pool: Option<ShardType>) -> bool {
        let len = self.segment_len();
        if len <= self.reclaim_trigger_len() {
            return false;
        }
        self.stall_permits_forcing(len, pool)
    }

    /// Whether a stalled reclaim (if there is one) still lets `pool`'s sync
    /// force a checkpoint at segment length `len`.
    #[cfg(feature = "multi-reader")]
    fn stall_permits_forcing(&self, len: u64, pool: Option<ShardType>) -> bool {
        let Some(stall) = *self.reclaim_stall.lock() else {
            return true;
        };
        self.coverage_epoch.load(Ordering::Acquire) != stall.coverage_epoch
            || len >= stall.at_len.saturating_add(self.reclaim_retry_growth())
            || pool.is_some_and(|pool| self.reclaim_blockers().contains(&pool))
    }

    /// Per-pool journals have no other pool to wait on, so nothing stalls.
    #[cfg(not(feature = "multi-reader"))]
    fn stall_permits_forcing(&self, _len: u64, _pool: Option<ShardType>) -> bool {
        let _ = self;
        true
    }

    /// What the journal's in-memory group directory holds: its entry count and
    /// allocated bytes.
    #[must_use]
    pub fn group_directory_stats(&self) -> GroupDirectoryStats {
        self.journal.lock().directory_stats()
    }

    /// Segment length above which a sync forces a reclaiming checkpoint.
    #[must_use]
    pub fn reclaim_trigger_len(&self) -> u64 {
        self.reclaim_trigger_len.load(Ordering::Relaxed)
    }

    /// Set the segment length above which a sync forces a reclaiming
    /// checkpoint. Tests use a small value to cross it without writing tens of
    /// MiB. It is not available outside tests: a value of zero would force a
    /// checkpoint on every sync and one above the segment cap would switch the
    /// forced reclaim off.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn set_reclaim_trigger_len(&self, len: u64) {
        self.reclaim_trigger_len.store(len, Ordering::Relaxed);
    }

    /// Shrink the segment cap, and the trigger with it (a quarter of the cap),
    /// so a test can reach the hard limit with a few hundred KiB.
    #[cfg(all(test, feature = "multi-reader"))]
    pub(crate) fn set_segment_cap(&self, cap: u64) {
        self.journal.lock().segment_cap = cap;
        self.set_reclaim_trigger_len(cap / 4);
    }

    /// Highest LSN any caller has requested through [`Self::request_durable`].
    #[must_use]
    pub fn durable_requested(&self) -> u64 {
        self.durable_requested.load(Ordering::Acquire)
    }

    /// Lifetime durability accounting: requests versus real fsyncs, records
    /// per fsync, and blocked-wait latency.
    #[must_use]
    pub fn durability_stats(&self) -> DurabilityStats {
        DurabilityStats {
            durable_requests: self.durable_requests.load(Ordering::Relaxed),
            durable_waits_already_durable: self
                .durable_waits_already_durable
                .load(Ordering::Relaxed),
            durable_wait: self.durable_wait.snapshot(),
            sync_requests: self.sync_calls.load(Ordering::Relaxed),
            sync_waiters: self.journal_waiters.load(Ordering::Relaxed),
            sync_coalesced: self.coalesced_syncs.load(Ordering::Relaxed),
            commits: self.commits.load(Ordering::Relaxed),
            commit_records: self.commit_records.load(Ordering::Relaxed),
            max_commit_records: self.max_commit_records.load(Ordering::Relaxed),
        }
    }

    /// Number of published mutations not yet durably committed. Used as the
    /// background committer's early-flush bound.
    #[must_use]
    pub fn pending_count(&self) -> u64 {
        self.published_lsn
            .load(Ordering::Acquire)
            .saturating_sub(self.committed_lsn.load(Ordering::Acquire))
    }

    /// Whether a background committer is currently running.
    #[must_use]
    pub fn has_background_committer(&self) -> bool {
        matches!(*self.background.lock(), BackgroundState::Running(_))
    }

    /// Number of background group commits that appended and fsynced a group.
    #[must_use]
    pub fn background_commits(&self) -> u64 {
        self.background_commits.load(Ordering::Relaxed)
    }

    /// Number of background commit attempts already covered by a concurrent
    /// durable group. Idle timer ticks (nothing pending) are not counted.
    #[must_use]
    pub fn background_coalesced(&self) -> u64 {
        self.background_coalesced.load(Ordering::Relaxed)
    }

    fn background_failure_detail(&self) -> Option<BackgroundFailure> {
        self.background_failure.lock().clone()
    }

    /// Return the terminal background-committer failure, if any, without
    /// consuming it. Callers use this to retain dirty work and retry through
    /// their own scheduling layer; `wait_durable` remains the blocking API.
    #[must_use]
    pub fn background_failure_message(&self) -> Option<String> {
        self.background_failure
            .lock()
            .as_ref()
            .map(|failure| failure.message.clone())
    }

    /// Wake the committer and any `wait_durable` callers.
    ///
    /// Takes [`Self::durable_lock`] so the wake cannot be lost to the race
    /// where a waiter checks its predicate and then parks after the notifier
    /// has already notified: the notifier either holds the lock while the
    /// waiter is parked (waking it) or blocks until the waiter releases it.
    fn wake_durable_waiters(&self) {
        let _guard = self.durable_lock.lock();
        self.durable_cv.notify_all();
    }

    /// Claim the wake for the current commit epoch.
    ///
    /// Returns `true` only for the publish that wakes the committer for this
    /// epoch, so a burst costs one wake. The claim is retried because a commit
    /// can advance the epoch between the load and the CAS; re-reading the epoch
    /// after a successful claim ensures that newer commit also gets a wake
    /// instead of being suppressed by a claim that was valid a moment earlier.
    fn claim_threshold_wake(&self) -> bool {
        self.claim_threshold_wake_with(|| {})
    }

    /// [`Self::claim_threshold_wake`] with a hook run between a successful claim
    /// and the epoch recheck, so tests can force a commit into that window.
    fn claim_threshold_wake_with(&self, mut between_claim_and_recheck: impl FnMut()) -> bool {
        loop {
            let notified = self.threshold_notified_epoch.load(Ordering::Acquire);
            let epoch = self.commit_epoch.load(Ordering::Acquire);
            if notified == epoch {
                return false;
            }
            if self
                .threshold_notified_epoch
                .compare_exchange(notified, epoch, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                between_claim_and_recheck();
                if self.commit_epoch.load(Ordering::Acquire) == epoch {
                    return true;
                }
            }
        }
    }

    /// Wake the committer and durability waiters when a durable commit left the
    /// pending queue at or above the early-flush bound.
    ///
    /// Publishers that arrived during an fsync read the pre-commit epoch and
    /// suppressed their wake, so nothing would re-arm the parked committer
    /// until the next publish or timer tick. This closes that window from the
    /// commit side, after the epoch has advanced. It is a no-op when no
    /// committer is running (`committer_wake_threshold` is zero).
    fn wake_committer_if_backlogged(&self) {
        let threshold = self.committer_wake_threshold.load(Ordering::Relaxed);
        if threshold > 0 && self.pending_count() >= threshold {
            self.wake_durable_waiters();
        }
    }

    /// Block until the group covering `token` is durable.
    ///
    /// With a background committer running this waits for it to flush; the
    /// bounded poll interval lets the waiter notice a committer that stopped or
    /// a poisoned journal. A committer that failed terminally is reported
    /// rather than waited on. Without a committer, this is exactly
    /// [`Self::sync_through`] on the token's LSN — the historical blocking
    /// behavior, so a caller that never opts into background commits is
    /// unaffected.
    ///
    /// # Errors
    /// Returns an error if the target was never published, the journal is
    /// poisoned, the committer failed, or the committing sync fails.
    pub fn wait_durable(&self, token: DurabilityToken) -> io::Result<Option<CommitReceipt>> {
        let target = token.lsn;
        if target == 0 || target <= self.committed_lsn.load(Ordering::Acquire) {
            self.durable_waits_already_durable
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let result = self.wait_durable_blocking(target);
        self.durable_wait.observe(started.elapsed());
        result
    }

    fn wait_durable_blocking(&self, target: u64) -> io::Result<Option<CommitReceipt>> {
        loop {
            if target <= self.committed_lsn.load(Ordering::Acquire) {
                return Ok(None);
            }
            // Recheck every predicate under `durable_lock`, the same lock the
            // notifiers take, so a state change cannot slip between check and
            // park. The poll interval remains a backstop, not a requirement.
            let mut guard = self.durable_lock.lock();
            if target <= self.committed_lsn.load(Ordering::Acquire) {
                return Ok(None);
            }
            if self.poisoned.load(Ordering::Acquire) {
                return Err(io::Error::other(
                    "journal is poisoned after a failed commit",
                ));
            }
            // A stopped committer falls through to a direct commit, preserving
            // the historical no-committer behavior and error text, but a
            // committer that failed must not be silently papered over.
            if let Some(failure) = self.background_failure_detail() {
                return Err(failure.into_io());
            }
            if !self.has_background_committer() {
                drop(guard);
                return self.sync_through(target);
            }
            // Only with a live committer is an unpublished target validated
            // here; the no-committer path above reaches `sync_through` first.
            if target > self.published_lsn.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "sync target has not been published",
                ));
            }
            self.durable_cv.wait_for(&mut guard, DURABILITY_WAIT_POLL);
        }
    }

    /// Perform one bounded group commit over everything published so far.
    ///
    /// Used by the background committer's timer and by
    /// [`Self::stop_background_committer`]. A no-op returning `None` when
    /// nothing is pending (so an idle tick neither fsyncs nor inflates the
    /// coalescing counter). This is the journal-level durability boundary; a
    /// caller that also needs its packfile checkpoint rewritten should use the
    /// storage-level sync path instead.
    ///
    /// # Errors
    /// Propagates a durable-commit failure from [`Self::sync_through`].
    pub fn flush_durable(&self) -> io::Result<Option<CommitReceipt>> {
        let target = self.capture_sync_target();
        if target == 0 || target <= self.committed_lsn.load(Ordering::Acquire) {
            return Ok(None);
        }
        let receipt = self.sync_through(target)?;
        if receipt.is_some() {
            self.background_commits.fetch_add(1, Ordering::Relaxed);
        } else {
            // Another caller's group covered this target first.
            self.background_coalesced.fetch_add(1, Ordering::Relaxed);
        }
        // A group commit may cover waiters parked on a boundary below `target`.
        self.wake_durable_waiters();
        Ok(receipt)
    }

    /// Start a background thread that fsyncs the pending WAL group on a bounded
    /// interval, coalescing every mutation published in the window into one
    /// group. Flushes early once `max_pending` records are outstanding.
    /// Idempotent: a call while one is running (or stopping) is a no-op.
    ///
    /// The thread holds only a [`Weak`] reference between iterations, so
    /// dropping the last `Arc<JournalCoordinator>` lets it exit within one
    /// interval without an `Arc` cycle. That is a bounded delay, not immediate
    /// teardown — call [`Self::stop_background_committer`] for deterministic
    /// cleanup and a final flush.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] if `interval` is zero or
    /// `max_pending` is zero (both would busy-loop or flush continuously), or
    /// another error if the committer thread cannot be spawned.
    pub fn start_background_committer(
        self: &Arc<Self>,
        config: GroupCommitConfig,
    ) -> io::Result<()> {
        if config.interval.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "background committer interval must be greater than zero",
            ));
        }
        if config.max_pending == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "background committer max_pending must be greater than zero",
            ));
        }
        let mut slot = self.background.lock();
        // Running or Stopping: never spawn a second committer.
        if !matches!(*slot, BackgroundState::Stopped) {
            return Ok(());
        }
        *self.background_failure.lock() = None;
        // A failed run can leave the notification epoch equal to the current
        // commit epoch, which would suppress the first threshold wake after a
        // restart. Reset it so the first crossing always wakes.
        self.threshold_notified_epoch
            .store(u64::MAX, Ordering::Release);
        let stop = Arc::new(AtomicBool::new(false));
        let weak = Arc::downgrade(self);
        let stop_clone = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("mtxdb-journal-committer".to_owned())
            .spawn(move || {
                // Catch a panic so the terminal failure is recorded and waiters
                // are released instead of polling forever on a dead worker.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Self::background_committer_loop(&weak, config, &stop_clone)
                }));
                if let Some(coordinator) = weak.upgrade() {
                    let failure = match outcome {
                        Ok(Ok(())) => None,
                        Ok(Err(failure)) => Some(failure),
                        Err(_) => Some(BackgroundFailure {
                            kind: io::ErrorKind::Other,
                            message: "background committer panicked".to_owned(),
                        }),
                    };
                    coordinator.finish_background(failure);
                }
            })
            .map_err(|error| {
                io::Error::other(format!("failed to spawn journal committer: {error}"))
            })?;
        self.committer_wake_threshold
            .store(config.max_pending, Ordering::Release);
        *slot = BackgroundState::Running(BackgroundCommitter {
            stop,
            handle: Some(handle),
        });
        Ok(())
    }

    /// Signal the background committer to stop, join it, then flush any final
    /// pending group so a quiet stream's last write is durable before this
    /// returns. Safe to call when no committer is running (it still flushes).
    ///
    /// The committer slot is held in a `Stopping` state across the join, so a
    /// concurrent [`Self::start_background_committer`] cannot spawn a second
    /// worker mid-teardown.
    ///
    /// # Errors
    /// Propagates a failure from the final [`Self::flush_durable`], and surfaces
    /// a terminal committer failure recorded by the worker.
    pub fn stop_background_committer(&self) -> io::Result<()> {
        let handle = {
            let mut slot = self.background.lock();
            match std::mem::replace(&mut *slot, BackgroundState::Stopping) {
                BackgroundState::Running(mut committer) => {
                    committer.stop.store(true, Ordering::Release);
                    committer.handle.take()
                }
                other => {
                    *slot = other;
                    None
                }
            }
        };
        self.committer_wake_threshold.store(0, Ordering::Release);
        self.wake_durable_waiters();
        if let Some(handle) = handle {
            // A panic in the committer is caught and recorded by the worker's
            // wrapper; joining still succeeds and the failure is surfaced below.
            let _ = handle.join();
        }
        // Transition `Stopping -> Stopped` and take the recorded failure under
        // one `background` lock. Otherwise a `start_background_committer` that
        // sees `Stopped` can clear `background_failure` in the window between
        // the transition and the take, silently losing the worker's failure.
        let failure = {
            let mut slot = self.background.lock();
            if matches!(*slot, BackgroundState::Stopping) {
                *slot = BackgroundState::Stopped;
            }
            self.background_failure.lock().take()
        };
        self.flush_durable()?;
        if let Some(failure) = failure {
            return Err(failure.into_io());
        }
        Ok(())
    }

    /// Mark the committer slot stopped, publish any terminal failure, and wake
    /// waiters. Called by the worker itself as it exits.
    fn finish_background(&self, failure: Option<BackgroundFailure>) {
        // Transition, publish the failure, and clear the wake threshold under a
        // single `background` lock. A concurrent `start_background_committer`
        // (which holds that lock to clear `background_failure` and set its own
        // threshold) therefore either sees this worker still `Running` and does
        // nothing, or runs after the failure is already recorded; it cannot
        // erase the failure or have its threshold zeroed by this exiting worker.
        {
            let mut slot = self.background.lock();
            // Leave `Stopping` alone: `stop_background_committer` owns that
            // transition and takes the failure after joining this thread.
            if matches!(*slot, BackgroundState::Running(_)) {
                *slot = BackgroundState::Stopped;
                self.committer_wake_threshold.store(0, Ordering::Release);
            }
            if let Some(failure) = failure {
                *self.background_failure.lock() = Some(failure);
            }
        }
        self.wake_durable_waiters();
    }

    fn background_committer_loop(
        weak: &Weak<Self>,
        config: GroupCommitConfig,
        stop: &Arc<AtomicBool>,
    ) -> Result<(), BackgroundFailure> {
        let interval = config.interval;
        loop {
            let Some(coordinator) = weak.upgrade() else {
                return Ok(());
            };
            let now = std::time::Instant::now();
            let deadline = now.checked_add(interval).unwrap_or(now);
            {
                let mut guard = coordinator.durable_lock.lock();
                loop {
                    if stop.load(Ordering::Acquire) || coordinator.poisoned.load(Ordering::Acquire)
                    {
                        break;
                    }
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    if coordinator.pending_count() >= config.max_pending {
                        break;
                    }
                    let timeout = deadline.saturating_duration_since(now);
                    #[cfg(test)]
                    coordinator.committer_parks.fetch_add(1, Ordering::Release);
                    coordinator.durable_cv.wait_for(&mut guard, timeout);
                }
            }
            if stop.load(Ordering::Acquire) || coordinator.poisoned.load(Ordering::Acquire) {
                return Ok(());
            }
            // The final flush on stop is done by `stop_background_committer`
            // after join, so a stop that lands here does not double-fsync.
            if let Err(error) = coordinator.flush_durable() {
                if coordinator.poisoned.load(Ordering::Acquire) {
                    // Poison is already reported to waiters via the poison bit.
                    return Ok(());
                }
                return Err(BackgroundFailure::from_io(&error));
            }
        }
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
        // Wait out any in-flight fsync: the rewrite replaces the segment file,
        // and a sync must not flush a handle to the replaced inode.
        let _sync = self.sync_lock.lock();
        let mut journal = self.journal.lock();
        let replay_limit = self.replay_reclaim_limit(covered_lsn);
        let reclaim = journal.reclaim_through(replay_limit)?;
        // The rewrite replaced the segment inode and moved its base LSN without
        // necessarily publishing a group. Advance the signal so a gated reader
        // rescans and detects the base jump instead of serving an index older
        // than the reclaimed prefix.
        self.bump_publish_signal();
        Ok(reclaim)
    }

    /// [`Self::reclaim_through`] recording the pool whose missing coverage
    /// stopped the cut on the returned [`Reclaim`]. Diagnostics only.
    #[cfg(feature = "multi-reader")]
    fn reclaim_through_with_blocker(
        &self,
        covered_lsn: u64,
        blocked_by: Option<ShardType>,
    ) -> io::Result<Reclaim> {
        let _sync = self.sync_lock.lock();
        let mut journal = self.journal.lock();
        let replay_limit = self.replay_reclaim_limit(covered_lsn);
        let blocked_by = (replay_limit == covered_lsn)
            .then_some(blocked_by)
            .flatten();
        let reclaim = journal.reclaim_through_with_blocker(replay_limit, blocked_by)?;
        // See [`Self::reclaim_through`]: a reader gated on the signal must be
        // forced to rescan after the segment is rewritten in place.
        self.bump_publish_signal();
        Ok(reclaim)
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
        let path = path.as_ref();
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
        let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        scan_bytes(&bytes, base_sequence, base_lsn, version, || {
            read_durable_len_from_header(&bytes, version, file_len)
        })
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
        Self::scan_read_only_range(path, start, u64::MAX, expected_lsn)
    }

    /// Scan complete groups only through `end`, an absolute group boundary.
    /// This keeps paged replay from reading and decoding later groups that do
    /// not belong to the requested page.
    #[allow(clippy::verbose_file_reads)]
    fn scan_read_only_range(
        path: impl AsRef<Path>,
        start: u64,
        end: u64,
        expected_lsn: u64,
    ) -> io::Result<Scan> {
        let path = path.as_ref();
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
        let (version, _base_sequence, base_lsn) = validate_file_header(&header)?;
        let scan_end = end.min(len);
        if start >= scan_end {
            return Ok(Scan {
                groups: Vec::new(),
                valid_len: start.min(scan_end),
                truncated_tail: false,
                base_lsn,
                consumed_tail: Vec::new(),
                group_ends: Vec::new(),
            });
        }
        file.seek(SeekFrom::Start(start))?;
        let mut tail = Vec::new();
        file.take(scan_end.saturating_sub(start))
            .read_to_end(&mut tail)?;
        scan_groups_from(&tail, start, 0, expected_lsn, base_lsn, version, || {
            read_durable_len_from_header(&header, version, len)
        })
    }

    /// Read up to `max_len` bytes of the segment ending at `end_offset` from an
    /// already-open `file` into `window`, never reaching back into the file
    /// header. Returns `false` when the file is shorter than `end_offset` or
    /// has no group bytes before it, leaving `window` unspecified.
    ///
    /// A read-only overlay keeps this window from the end of the last group it
    /// consumed and compares it on later refreshes: file length alone cannot
    /// show that the bytes already consumed are still the same ones. The
    /// window is compared byte for byte instead of relying on the group
    /// trailer, because the trailer is not a content fingerprint: the group
    /// checksum covers `header || records` and every record already ends with
    /// its own CRC32, so for groups of the same shape it comes out identical
    /// whatever the payload. The window includes the last record's own CRC,
    /// which does depend on the record's full content.
    ///
    /// Reads through the caller's descriptor with a positioned read, so the
    /// refresh path pays one syscall and reuses `window`'s allocation instead
    /// of opening the file each time. The caller must notice a replaced file
    /// itself (its descriptor keeps the old inode). Never repairs or creates
    /// the file.
    ///
    /// # Errors
    /// Returns `io::Error` if the read fails.
    pub(crate) fn read_tail_ending_at(
        file: &File,
        end_offset: u64,
        max_len: u64,
        window: &mut Vec<u8>,
    ) -> io::Result<bool> {
        let header_len = u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX);
        if end_offset <= header_len {
            return Ok(false);
        }
        let start = end_offset.saturating_sub(max_len).max(header_len);
        let window_len = usize::try_from(end_offset.saturating_sub(start))
            .map_err(|_| invalid_data("tail window exceeds the address space"))?;
        window.clear();
        window.resize(window_len, 0);
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_exact_at(file, window, start);
        #[cfg(not(unix))]
        let read = {
            let mut handle = file;
            handle
                .seek(SeekFrom::Start(start))
                .and_then(|_| handle.read_exact(window))
        };
        match read {
            Ok(()) => Ok(true),
            // The file is shorter than the window's end: the consumed prefix is gone.
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
            Err(error) => Err(error),
        }
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
        Self::open_versioned(path, JournalVersion::per_pool(), 1)
    }

    /// Open a shared (multi-pool) journal segment, or create one if the file is
    /// absent. The segment is created as `JournalVersion::V5PoolTagged`, so
    /// every mutation frame appended through it must carry a pool tag.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidData` if an existing segment is not
    /// the pool-tagged version.
    #[cfg(feature = "multi-reader")]
    pub fn open_shared(path: impl AsRef<Path>) -> io::Result<(Self, Scan)> {
        Self::open_versioned(path, JournalVersion::shared(), 1)
    }

    /// Like [`Self::open_shared`], but when the segment is first created its
    /// LSN space begins at `base_lsn` rather than 1. An existing segment
    /// ignores `base_lsn` and keeps its own.
    ///
    /// Internal to the root-level open path: an arbitrary `base_lsn` would let
    /// a caller create a segment whose numbering does not line up with the
    /// pools' recorded coverage, so this is not public. Callers that open a
    /// database root should go through
    /// [`crate::database::SharedDatabase::open`], which computes the seed from
    /// the pools' `journal.lsn` and passes it here.
    ///
    /// # Errors
    /// Same as [`Self::open_shared`].
    #[cfg(feature = "multi-reader")]
    pub(crate) fn open_shared_with_base(
        path: impl AsRef<Path>,
        base_lsn: u64,
    ) -> io::Result<(Self, Scan)> {
        Self::open_versioned(path, JournalVersion::shared(), base_lsn.max(1))
    }

    /// Open a journal whose on-disk version must be `version`, creating it with
    /// that version and base LSN `base_lsn` when the file is absent or shorter
    /// than the header.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidData` if an existing complete
    /// segment was written by a different version.
    fn open_versioned(
        path: impl AsRef<Path>,
        version: JournalVersion,
        base_lsn: u64,
    ) -> io::Result<(Self, Scan)> {
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
            write_file_header(&mut file, version, 1, base_lsn)?;
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
        let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, version, || {
            read_durable_len_from_header(&bytes, version, file_len)
        })?;
        if scan.truncated_tail {
            file.set_len(scan.valid_len)?;
        }
        file.seek(SeekFrom::Start(scan.valid_len))?;
        if scan.truncated_tail || !scan.groups.is_empty() {
            // What recovery keeps may exist only in the page cache (a process
            // crash leaves it there), yet the coordinator will report it
            // committed. Make it durable now, and record that in the hint, so a
            // later sync never claims bytes that were not fsynced. The same
            // fsync makes a truncation durable, including one that dropped
            // every group, so a dropped tail cannot reappear.
            file.sync_all()?;
        }
        let next_sequence = scan
            .groups
            .last()
            .map_or(base_sequence, |group| group.sequence.saturating_add(1));
        let next_lsn = scan
            .groups
            .last()
            .map_or(base_lsn, |group| group.last_lsn.saturating_add(1));

        let mark_generation = if version.is_pool_tagged() {
            newest_mark(&bytes).map_or(0, |slot| slot.generation)
        } else {
            0
        };
        let mut journal = Self {
            path,
            file,
            file_len: scan.valid_len,
            base_sequence,
            base_lsn,
            mark_generation,
            version,
            next_sequence,
            next_lsn,
            poisoned: false,
            segment_cap: MAX_SEGMENT_LEN,
            groups: marks_from_scan(&scan),
        };
        if !scan.groups.is_empty() {
            // Recovery fsynced what it kept above, so record that in the mark.
            journal.write_mark(scan.valid_len);
        }
        Ok((journal, scan))
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
    /// (which requires the multi-reader-only
    /// `append_group_tagged_with_sequence` method).
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
    #[cfg(feature = "multi-reader")]
    pub fn append_group_tagged_with_sequence(
        &mut self,
        mutations: &[(Option<ShardType>, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        self.append_group_inner(mutations, sequence)
    }

    fn append_group_for_current_mode(
        &mut self,
        mutations: &[(Option<ShardType>, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        #[cfg(feature = "multi-reader")]
        {
            self.append_group_tagged_with_sequence(mutations, sequence)
        }
        #[cfg(not(feature = "multi-reader"))]
        {
            let untagged = mutations
                .iter()
                .cloned()
                .map(|(_, mutation)| mutation)
                .collect::<Vec<_>>();
            self.append_group_with_sequence(&untagged, sequence)
        }
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
        if self.file_len.saturating_add(group_size) > self.segment_cap {
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

        // One write per group: every autocommit publish lands here, so the
        // group goes out as a single contiguous buffer instead of three
        // syscalls. A torn write still leaves an incomplete tail that recovery
        // truncates, exactly as before.
        let mut frame = Vec::with_capacity(
            header
                .len()
                .saturating_add(payload.len())
                .saturating_add(trailer.len()),
        );
        frame.extend_from_slice(&header);
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(&trailer);
        let write_result = self.file.write_all(&frame);
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(error);
        }
        self.file_len = self.file_len.saturating_add(group_size);
        self.note_appended_group(sequence, first_lsn, last_lsn, mutations);

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
        // The fsync returned, so everything appended so far is durable: record
        // that in the durability mark, after the fact.
        self.write_mark(self.file_len);
        Ok(())
    }

    /// Record, in the slot not holding the newest mark, that `durable_len`
    /// bytes are durable. Call it only after an fsync
    /// that covered them, so the mark can only under-claim.
    ///
    /// Only pool-tagged segments keep a mark. The slot is written with a
    /// positioned write, so it never moves the append cursor. A failed write
    /// is ignored: the previous mark stays in the other slot, or recovery sees
    /// no mark and truncates conservatively.
    fn write_mark(&mut self, durable_len: u64) {
        if !self.version.is_pool_tagged() {
            return;
        }
        let generation = self.mark_generation.saturating_add(1);
        let bytes = encode_mark_slot(MarkSlot {
            generation,
            durable_len,
        });
        let slot = usize::try_from(generation % 2).unwrap_or(0);
        let offset = u64::try_from(MARK_SLOT_OFFSETS[slot]).unwrap_or(u64::MAX);
        #[cfg(unix)]
        let written = std::os::unix::fs::FileExt::write_all_at(&self.file, &bytes, offset);
        // Elsewhere a positioned write moves the shared cursor, so restore it
        // to the end of the segment, where appends continue. Appends take the
        // journal lock too, so nothing writes in between.
        #[cfg(not(unix))]
        let written = self
            .file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| self.file.write_all(&bytes))
            .and_then(|()| self.file.seek(SeekFrom::Start(self.file_len)).map(drop));
        if written.is_ok() {
            self.mark_generation = generation;
        }
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
    /// `JournalCoordinator::with_shared_sequence`.
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
        self.reclaim_through_with_blocker(covered_lsn, None)
    }

    /// [`Self::reclaim_through`] with the pool whose missing coverage stopped
    /// the cut recorded on the returned [`Reclaim`], for diagnostics only.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::reclaim_through`].
    pub fn reclaim_through_with_blocker(
        &mut self,
        covered_lsn: u64,
        blocked_by: Option<ShardType>,
    ) -> io::Result<Reclaim> {
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
        let mut reclaimed = if self.directory_matches_file() {
            self.reclaim_through_directory(covered_lsn)?
        } else {
            self.reclaim_through_scan(covered_lsn)?
        };
        reclaimed.blocked_by = blocked_by;
        Ok(reclaimed)
    }

    /// Record a group just appended, ending at `file_len`, in the directory.
    fn note_appended_group(
        &mut self,
        sequence: u64,
        first_lsn: u64,
        last_lsn: u64,
        mutations: &[(Option<ShardType>, Mutation)],
    ) {
        self.groups.push(GroupMark {
            sequence,
            first_lsn,
            last_lsn,
            end_offset: self.file_len,
            pools: mutations
                .iter()
                .fold(0, |mask, (pool, _)| mask | pool_bit(*pool)),
        });
    }

    /// The pools that hold reclaim back: those with a frame in the oldest group
    /// still needed that have not reported coverage through it. Empty when the
    /// directory cannot be trusted, or the oldest needed group holds a frame no
    /// pool can be asked to cover (an untagged one).
    #[cfg(feature = "multi-reader")]
    fn blocking_pools(&self, covered: &HashMap<ShardType, u64>) -> Vec<ShardType> {
        if !self.directory_matches_file() {
            return Vec::new();
        }
        for group in &self.groups {
            let blockers: Vec<ShardType> = ShardType::ALL
                .into_iter()
                .enumerate()
                .filter(|(index, pool)| {
                    group.pools & (1_u8 << index) != 0
                        && !covered.get(pool).is_some_and(|lsn| *lsn >= group.last_lsn)
                })
                .map(|(_, pool)| pool)
                .collect();
            if !blockers.is_empty() {
                return blockers;
            }
            if group.pools & UNATTRIBUTED_POOL_BIT != 0 {
                return Vec::new();
            }
        }
        Vec::new()
    }

    /// What the group directory currently holds.
    fn directory_stats(&self) -> GroupDirectoryStats {
        GroupDirectoryStats {
            groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
            allocated_bytes: u64::try_from(
                self.groups
                    .capacity()
                    .saturating_mul(std::mem::size_of::<GroupMark>()),
            )
            .unwrap_or(u64::MAX),
        }
    }

    /// Whether the in-memory group directory describes the file exactly: its
    /// last group ends where the file ends.
    fn directory_matches_file(&self) -> bool {
        let directory_end = self
            .groups
            .last()
            .map_or(FILE_HEADER_LEN as u64, |group| group.end_offset);
        directory_end == self.file_len
    }

    /// The newest LSN through which a shared segment may be reclaimed, given
    /// each pool's reported coverage: the last group of the longest prefix in
    /// which every pool with a frame has covered the group. The directory being
    /// stale is reported apart from nothing being covered, so the caller can
    /// scan the file in that case.
    #[cfg(feature = "multi-reader")]
    fn shared_reclaim_boundary(&self, covered: &HashMap<ShardType, u64>) -> SharedBoundary {
        if !self.directory_matches_file() {
            return SharedBoundary::Untrusted;
        }
        boundary_through(&self.groups, covered)
            .map_or(SharedBoundary::NothingCovered, |(lsn, blocked_by)| {
                SharedBoundary::Through(lsn, blocked_by)
            })
    }

    /// Reclaim using the group directory: pick the cut from it and read only
    /// the retained suffix from disk. The dropped prefix is never read.
    fn reclaim_through_directory(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        let cut = self
            .groups
            .partition_point(|group| group.last_lsn <= covered_lsn);
        let Some(last_dropped) = cut.checked_sub(1).and_then(|index| self.groups.get(index)) else {
            return Ok(Reclaim {
                retained_groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
                ..Default::default()
            });
        };
        let copy_started = std::time::Instant::now();
        let header_len = FILE_HEADER_LEN as u64;
        let cut_offset = last_dropped.end_offset;
        let retained = self.groups.get(cut..).unwrap_or_default();
        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let mut rebuilt = file_header_bytes(self.version, new_base_sequence, new_base_lsn);
        if let Some(first) = retained.first() {
            // Read the suffix straight into the buffer that becomes the new
            // segment, after its header, so it is allocated once.
            let header_len = rebuilt.len();
            read_range_into(
                &self.path,
                cut_offset,
                self.file_len.saturating_sub(cut_offset),
                &mut rebuilt,
            )?;
            // The suffix is copied byte for byte, so check it the way the full
            // scan would have: it must hold exactly the groups the directory
            // says, each valid. Anything else means the directory is stale, and
            // the scan path decides instead.
            let Ok(checked) = scan_groups_from(
                rebuilt.get(header_len..).unwrap_or_default(),
                cut_offset,
                first.sequence,
                first.first_lsn,
                new_base_lsn,
                self.version,
                || u64::MAX,
            ) else {
                return self.reclaim_through_scan(covered_lsn);
            };
            if checked.groups.len() != retained.len()
                || checked.valid_len != self.file_len
                || checked.truncated_tail
            {
                return self.reclaim_through_scan(covered_lsn);
            }
        }
        let reclaimed_bytes = self
            .file_len
            .saturating_sub(u64::try_from(rebuilt.len()).unwrap_or(u64::MAX));
        let retained_groups = u64::try_from(retained.len()).unwrap_or(u64::MAX);
        let shift = cut_offset.saturating_sub(header_len);
        let survivors = retained
            .iter()
            .map(|group| GroupMark {
                end_offset: group.end_offset.saturating_sub(shift),
                ..*group
            })
            .collect::<Vec<_>>();
        let copy = copy_started.elapsed();
        let rebuilt_len = rebuilt.len();
        let fsync_started = std::time::Instant::now();
        self.install_rebuilt(rebuilt, new_base_sequence, new_base_lsn)?;
        let fsync = fsync_started.elapsed();
        self.groups = survivors;
        Ok(Reclaim {
            retained_groups,
            reclaimed_bytes,
            retained_bytes: u64::try_from(rebuilt_len.saturating_sub(FILE_HEADER_LEN))
                .unwrap_or(u64::MAX),
            copy,
            fsync,
            ..Default::default()
        })
    }

    /// Reclaim by reading and decoding the whole segment. Used when the group
    /// directory cannot be trusted, and it rebuilds the directory from what it
    /// finds.
    fn reclaim_through_scan(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        let scan_started = std::time::Instant::now();
        let bytes = fs::read(&self.path)?;
        let (version, base_sequence, base_lsn) = validate_file_header(&bytes)?;
        // Reclaim rewrites the segment from its own live file, so any invalid
        // group there is corruption, not a crash tail: treat every byte as
        // durable and let it fail.
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, version, || u64::MAX)?;
        let boundary = scan_started.elapsed();
        let retained = scan
            .groups
            .iter()
            .filter(|group| group.last_lsn > covered_lsn)
            .collect::<Vec<_>>();
        if retained.len() == scan.groups.len() {
            self.groups = marks_from_scan(&scan);
            return Ok(Reclaim {
                retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
                boundary,
                ..Default::default()
            });
        }

        let copy_started = std::time::Instant::now();
        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let mut rebuilt = file_header_bytes(self.version, new_base_sequence, new_base_lsn);
        let mut survivors = Vec::with_capacity(retained.len());
        for group in &retained {
            encode_group(group, self.version, &mut rebuilt)?;
            survivors.push(GroupMark {
                sequence: group.sequence,
                first_lsn: group.first_lsn,
                last_lsn: group.last_lsn,
                end_offset: u64::try_from(rebuilt.len()).unwrap_or(u64::MAX),
                pools: group
                    .entries
                    .iter()
                    .fold(0, |mask, entry| mask | pool_bit(entry.pool)),
            });
        }
        let copy = copy_started.elapsed();
        let retained_groups = u64::try_from(retained.len()).unwrap_or(u64::MAX);
        let rebuilt_len = rebuilt.len();
        let reclaimed_bytes =
            u64::try_from(bytes.len().saturating_sub(rebuilt.len())).unwrap_or(u64::MAX);
        let fsync_started = std::time::Instant::now();
        self.install_rebuilt(rebuilt, new_base_sequence, new_base_lsn)?;
        let fsync = fsync_started.elapsed();
        self.groups = survivors;
        Ok(Reclaim {
            retained_groups,
            reclaimed_bytes,
            retained_bytes: u64::try_from(rebuilt_len.saturating_sub(FILE_HEADER_LEN))
                .unwrap_or(u64::MAX),
            boundary,
            copy,
            fsync,
            ..Default::default()
        })
    }

    /// Durably replace the segment with `rebuilt` (a new header plus the
    /// retained groups) and point this handle at it.
    fn install_rebuilt(
        &mut self,
        mut rebuilt: Vec<u8>,
        new_base_sequence: u64,
        new_base_lsn: u64,
    ) -> io::Result<()> {
        // The rebuilt file is fsynced before it replaces the segment, so all of
        // it is durable. Put its mark inside it, first generation, so the mark
        // is durable with the data and the old segment's marks, which described
        // offsets that no longer exist, are simply gone.
        let rebuilt_mark = self.version.is_pool_tagged().then(|| MarkSlot {
            generation: 1,
            durable_len: u64::try_from(rebuilt.len()).unwrap_or(u64::MAX),
        });
        if let Some(mark) = rebuilt_mark {
            let offset = MARK_SLOT_OFFSETS[1];
            let end = offset.saturating_add(MARK_SLOT_LEN);
            rebuilt[offset..end].copy_from_slice(&encode_mark_slot(mark));
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
        self.file_len = u64::try_from(rebuilt.len()).unwrap_or(u64::MAX);
        self.base_sequence = new_base_sequence;
        self.base_lsn = new_base_lsn;
        self.mark_generation = u64::from(rebuilt_mark.is_some());
        if let Err(error) = sync_parent_dir(&self.path) {
            self.poisoned = true;
            return Err(error);
        }
        if let Err(error) = self.file.seek(SeekFrom::End(0)) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }
}

/// Append `len` bytes of the file at `path`, starting at `start`, to `out`.
fn read_range_into(path: &Path, start: u64, len: u64, out: &mut Vec<u8>) -> io::Result<()> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let len = usize::try_from(len).map_err(|_| invalid_data("range too large"))?;
    let filled = out.len();
    out.resize(filled.saturating_add(len), 0);
    if let Err(error) = file.read_exact(out.get_mut(filled..).unwrap_or_default()) {
        out.truncate(filled);
        return Err(error);
    }
    Ok(())
}

fn file_header_bytes(version: JournalVersion, base_sequence: u64, base_lsn: u64) -> Vec<u8> {
    let mut header = vec![0_u8; FILE_HEADER_LEN];
    header[FH_MAGIC].copy_from_slice(FILE_MAGIC);
    header[FH_VERSION].copy_from_slice(&version.as_u32().to_le_bytes());
    header[FH_BASE_SEQUENCE].copy_from_slice(&base_sequence.to_le_bytes());
    header[FH_BASE_LSN].copy_from_slice(&base_lsn.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header[FH_BASE_CRC_COVERED]);
    header[FH_BASE_CRC].copy_from_slice(&crc.finalize().to_le_bytes());
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
    crc.update(&header[FH_BASE_CRC_COVERED]);
    if u32::from_le_bytes(header[FH_BASE_CRC].try_into().expect("fixed slice")) != crc.finalize() {
        return Err(invalid_data("journal header checksum mismatch"));
    }
    Ok((
        version,
        u64::from_le_bytes(header[FH_BASE_SEQUENCE].try_into().expect("fixed slice")),
        u64::from_le_bytes(header[FH_BASE_LSN].try_into().expect("fixed slice")),
    ))
}

/// One durability-mark slot: which generation it is, and what it claims.
#[derive(Clone, Copy)]
struct MarkSlot {
    generation: u64,
    durable_len: u64,
}

fn encode_mark_slot(slot: MarkSlot) -> [u8; MARK_SLOT_LEN] {
    let mut bytes = [0_u8; MARK_SLOT_LEN];
    bytes[..8].copy_from_slice(&slot.generation.to_le_bytes());
    bytes[8..16].copy_from_slice(&slot.durable_len.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&bytes[..16]);
    bytes[16..20].copy_from_slice(&crc.finalize().to_le_bytes());
    bytes
}

/// Decode one slot: `None` if it is unwritten (all zero) or fails its checksum.
fn decode_mark_slot(bytes: &[u8]) -> Option<MarkSlot> {
    let bytes = bytes.get(..MARK_SLOT_LEN)?;
    let mut crc = Hasher::new();
    crc.update(&bytes[..16]);
    if u32::from_le_bytes(bytes[16..20].try_into().ok()?) != crc.finalize() {
        return None;
    }
    let slot = MarkSlot {
        generation: u64::from_le_bytes(bytes[..8].try_into().ok()?),
        durable_len: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
    };
    // Generation zero is never written, so a checksum-valid slot claiming it
    // is not a mark.
    (slot.generation != 0).then_some(slot)
}

/// The newest valid mark in a segment's header region, or `None` when neither
/// slot holds one: unwritten, torn, or corrupt. `None` means "durability
/// unknown", which only ever widens what recovery is willing to truncate.
fn newest_mark(bytes: &[u8]) -> Option<MarkSlot> {
    MARK_SLOT_OFFSETS
        .iter()
        .filter_map(|&offset| bytes.get(offset..).and_then(decode_mark_slot))
        .max_by_key(|slot| slot.generation)
}

/// How many leading bytes of the segment are known durable. Per-pool segments
/// keep no mark and count every byte durable, so an invalid group fails closed.
/// A missing or unreadable mark, or one claiming more than the file holds,
/// yields `0`.
fn read_durable_len_from_header(bytes: &[u8], version: JournalVersion, file_len: u64) -> u64 {
    if !version.is_pool_tagged() {
        return u64::MAX;
    }
    newest_mark(bytes)
        .map(|slot| slot.durable_len)
        .filter(|&durable_len| durable_len <= file_len)
        .unwrap_or(0)
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
    durable_len: impl Fn() -> u64,
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
        durable_len,
    )
}

/// What validating one group at a cursor found.
enum GroupStep {
    /// A complete, valid group that occupies `total_len` bytes.
    Group {
        group: GroupHeader,
        entries: Vec<JournalEntry>,
        total_len: usize,
    },
    /// Not enough bytes remain for a complete group.
    Incomplete,
}

/// Why a group failed validation, split by whether a crash can explain it.
enum GroupFault {
    /// Damage that a torn or missing page can produce: a bad group header or a
    /// bad commit trailer or checksum. Above the durable mark this is an
    /// unacknowledged crash tail; at or below it, it is corruption.
    Crash(io::Error),
    /// A group whose header checksum verified but whose contents are wrong
    /// (sequence or LSN regression, impossible bounds, undecodable records).
    /// A crash cannot do that to an append-only file, so this is a bug or real
    /// corruption wherever it appears and is never truncated.
    Fatal(io::Error),
}

/// Validate the group starting at `cursor` in `bytes`.
fn scan_one_group(
    bytes: &[u8],
    cursor: usize,
    base_offset: u64,
    expected_sequence: u64,
    expected_lsn: u64,
    version: JournalVersion,
) -> Result<GroupStep, GroupFault> {
    let remaining = bytes.len().saturating_sub(cursor);
    if remaining < GROUP_HEADER_LEN {
        return Ok(GroupStep::Incomplete);
    }
    let header = bytes
        .get(cursor..cursor.saturating_add(GROUP_HEADER_LEN))
        .ok_or_else(|| GroupFault::Crash(invalid_data("truncated journal group header")))?;
    let group = parse_group_header(header).map_err(GroupFault::Crash)?;
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
        || u64::from(group.record_count).saturating_mul(MIN_FRAME_LEN as u64) > group.payload_len
    {
        return Err(GroupFault::Fatal(invalid_data(
            "invalid journal sequence or group bounds",
        )));
    }
    let payload_len = usize::try_from(group.payload_len).map_err(|_| {
        GroupFault::Fatal(invalid_data("journal group length exceeds address space"))
    })?;
    let total_len = GROUP_HEADER_LEN
        .checked_add(payload_len)
        .and_then(|len| len.checked_add(GROUP_TRAILER_LEN))
        .ok_or_else(|| GroupFault::Fatal(invalid_data("journal group length overflow")))?;
    if remaining < total_len {
        return Ok(GroupStep::Incomplete);
    }
    let payload_index = cursor.saturating_add(GROUP_HEADER_LEN);
    let payload = verify_group_payload(bytes, header, &group, payload_index, payload_len)
        .map_err(GroupFault::Crash)?;
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
    )
    .map_err(GroupFault::Fatal)?;
    Ok(GroupStep::Group {
        group,
        entries,
        total_len,
    })
}

/// Scan complete groups from `bytes`, which begins at absolute file offset
/// `base_offset`.
///
/// `expected_sequence` is a floor, not an exact match: a segment sharing a
/// global sequence with other pools may skip values. `expected_lsn` must match
/// the first group's `first_lsn` exactly, since LSNs stay contiguous within a
/// segment. `Scan::valid_len` is absolute in the file, not relative to `bytes`.
///
/// `durable_len` yields the embedded durability mark: how many
/// leading bytes of the file a completed fsync covered. It is called at most
/// once, and only when a group fails validation in a way a crash can explain. A group that fails validation
/// in a way a crash can explain ([`GroupFault::Crash`]) at or above it was never
/// acknowledged, so the scan stops there and reports a truncated tail, dropping
/// everything after it, even later groups that happen to be intact. The same
/// failure below it is corruption and is returned as an error. Return `u64::MAX`
/// to treat every byte as durable, so any invalid group fails; return `0` when
/// nothing is known, so the first invalid group ends the scan.
fn scan_groups_from(
    bytes: &[u8],
    base_offset: u64,
    mut expected_sequence: u64,
    mut expected_lsn: u64,
    segment_base_lsn: u64,
    version: JournalVersion,
    durable_len: impl Fn() -> u64,
) -> io::Result<Scan> {
    let mut cursor = 0usize;
    let mut valid_len = base_offset;
    let mut groups = Vec::new();
    let mut group_ends = Vec::new();
    let mut truncated_tail = false;

    while cursor < bytes.len() {
        let (group, entries, total_len) = match scan_one_group(
            bytes,
            cursor,
            base_offset,
            expected_sequence,
            expected_lsn,
            version,
        ) {
            Ok(GroupStep::Group {
                group,
                entries,
                total_len,
            }) => (group, entries, total_len),
            Ok(GroupStep::Incomplete) => {
                truncated_tail = true;
                break;
            }
            Err(GroupFault::Fatal(error)) => return Err(error),
            Err(GroupFault::Crash(error)) => {
                let at = base_offset.saturating_add(u64::try_from(cursor).unwrap_or(u64::MAX));
                // The durability mark matters only here, on a failure a crash
                // can explain, so it is read only now. Reading it on every scan
                // would cost a file open and read on the hot incremental path
                // for something a healthy segment never consults.
                if at >= durable_len() {
                    truncated_tail = true;
                    break;
                }
                return Err(error);
            }
        };
        groups.push(CommittedGroup {
            sequence: group.sequence,
            first_lsn: group.first_lsn,
            last_lsn: group.last_lsn,
            entries,
        });
        cursor = cursor.saturating_add(total_len);
        valid_len = base_offset.saturating_add(u64::try_from(cursor).unwrap_or(u64::MAX));
        group_ends.push(valid_len);
        expected_sequence = group
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal group sequence overflow"))?;
        expected_lsn = group
            .last_lsn
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal LSN overflow"))?;
    }

    let consumed_tail = bytes[cursor.saturating_sub(CONSUMED_TAIL_LEN)..cursor].to_vec();
    Ok(Scan {
        groups,
        valid_len,
        truncated_tail,
        base_lsn: segment_base_lsn,
        consumed_tail,
        group_ends,
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
            if !matches!(tag, 1..=4) {
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
    crate::shard::sync_directory(parent)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
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
#[cfg(feature = "multi-reader")]
pub struct SharedWalLock {
    _lock: crate::shard::WriterLock,
}

#[cfg(feature = "multi-reader")]
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
        let path = db_root.join(".mtxdb.wal.lock");
        Ok(Self {
            _lock: crate::shard::ShardPool::acquire_lock_path(&path)?,
        })
    }
}

#[cfg(test)]
#[path = "test_journal.rs"]
mod tests;
