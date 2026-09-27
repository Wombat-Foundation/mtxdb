//! Root-level handle for a shared-WAL database.
//!
//! [`SharedDatabase`] is the safe end-to-end entry point for the shared
//! durability fence: it opens a database root, acquires the single root writer
//! lock, opens the root's pool-tagged `wal.bin` as one
//! [`crate::journal::JournalCoordinator`], opens every named pool, attaches each to that one
//! coordinator, and replays recovered groups. It holds the root lock for its
//! whole lifetime, so exactly one writer process owns the database.
//!
//! If the root's WAL is missing, the fresh segment's LSN space is seeded above
//! every pool's `journal.lsn` so a pool's recorded coverage stays meaningful
//! and numbering can never restart beneath it; see `shared_wal_seed_lsn`.
//! Callers that need to drive a single pool directly can still use
//! [`crate::journal::SharedWalLock`] with
//! [`PackfileStorage::enable_shared_journal`](crate::PackfileStorage::enable_shared_journal).

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::journal::{
    CommitReceipt, Journal, JournalCoordinator, SharedWalLock, StagedLookup, TxnStage,
    TxnStageState,
};
use crate::layout::{DatabaseLayout, ShardType};
use crate::packfile::storage::PackfileStorage;
use crate::storage::{NodeId, StorageEngine, StorageError};

/// Write and verification policy for a single storage pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolPolicy {
    /// Whether new records written through this pool attempt zstd compression.
    pub compress: bool,
    /// How much CRC32 checksum verification this pool applies.
    pub checksum_policy: crate::packfile::ChecksumPolicy,
}

impl Default for PoolPolicy {
    fn default() -> Self {
        Self {
            compress: true,
            checksum_policy: crate::packfile::ChecksumPolicy::Full,
        }
    }
}

/// Policies for every named pool in a shared-WAL database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolPolicies {
    /// Policy for the state pool (`ShardType::State`).
    pub state: PoolPolicy,
    /// Policy for the event DAG pool (`ShardType::EventDag`).
    pub event_dag: PoolPolicy,
    /// Policy for the edges pool (`ShardType::Edges`).
    pub edges: PoolPolicy,
}

impl PoolPolicies {
    /// Return the policy associated with `shard`.
    #[must_use]
    pub fn for_shard(&self, shard: ShardType) -> &PoolPolicy {
        match shard {
            ShardType::State => &self.state,
            ShardType::EventDag => &self.event_dag,
            ShardType::Edges => &self.edges,
        }
    }
}

/// A database root open for writing through one shared durability fence.
///
/// Dropping it releases the root writer lock and the pools it opened.
pub struct SharedDatabase {
    layout: DatabaseLayout,
    coordinator: Arc<JournalCoordinator>,
    pools: [Arc<PackfileStorage>; 3],
    /// Published transactions whose materialization still needs to be
    /// completed. The queue owns the transaction stage, so dropping a caller's
    /// handle cannot orphan the visibility overlay.
    recovery_queue: parking_lot::Mutex<Vec<RecoveryItem>>,
    /// Serializes materialization across transaction handles and recovery
    /// workers. The stage's applied-bit check and mutation application must be
    /// one critical section.
    recovery_lifecycle: parking_lot::Mutex<()>,
    commit_phases: CommitPhases,
    /// Held for the lifetime of the handle: one writer per database root.
    _lock: SharedWalLock,
}

/// Total time and call count for one commit phase.
#[derive(Default)]
struct PhaseTimer {
    nanos: AtomicU64,
    calls: AtomicU64,
}

impl PhaseTimer {
    fn add(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> PhaseTiming {
        PhaseTiming {
            total: Duration::from_nanos(self.nanos.load(Ordering::Relaxed)),
            calls: self.calls.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
struct CommitPhases {
    overlay: PhaseTimer,
    publish: PhaseTimer,
    register_wait: PhaseTimer,
    materialize_wait: PhaseTimer,
    materialize: PhaseTimer,
}

/// Accumulated time and call count of one commit phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseTiming {
    /// Summed wall time across all calls.
    pub total: Duration,
    /// Number of times the phase ran.
    pub calls: u64,
}

/// Where transaction commits have spent their time since the database opened.
///
/// `*_wait` phases are time blocked acquiring the recovery lifecycle lock, so
/// a large value there is contention between commits, not work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitPhaseStats {
    /// Activating the read overlay on every pool.
    pub overlay: PhaseTiming,
    /// Appending the journal group (includes waiting for the journal lock).
    pub publish: PhaseTiming,
    /// Waiting for the recovery lifecycle lock before registering the stage.
    pub register_wait: PhaseTiming,
    /// Waiting for the recovery lifecycle lock before materializing.
    pub materialize_wait: PhaseTiming,
    /// Applying the staged writes to the pools.
    pub materialize: PhaseTiming,
}

struct RecoveryItem {
    receipt: CommitReceipt,
    stage: Arc<TxnStage>,
    overlay: TransactionOverlayGuard,
}

struct TransactionOverlayGuard {
    /// The pools whose overlay this guard holds: only those the transaction
    /// stages writes for.
    pools: Vec<Arc<PackfileStorage>>,
}

impl Drop for TransactionOverlayGuard {
    fn drop(&mut self) {
        for pool in &self.pools {
            pool.deactivate_transaction_overlay();
        }
    }
}

/// How a transaction reaches its database: borrowed for the common in-process
/// case, or owned so the transaction can outlive the caller's borrow (held
/// across an FFI boundary or moved between threads).
enum DatabaseRef<'a> {
    Borrowed(&'a SharedDatabase),
    Owned(Arc<SharedDatabase>),
}

impl std::ops::Deref for DatabaseRef<'_> {
    type Target = SharedDatabase;

    fn deref(&self) -> &SharedDatabase {
        match self {
            Self::Borrowed(database) => database,
            Self::Owned(database) => database,
        }
    }
}

/// Storage transaction whose mutations remain invisible to other readers until
/// commit. The transaction itself reads its own writes through [`Self::get`].
pub struct DatabaseTransaction<'a> {
    database: DatabaseRef<'a>,
    stage: Arc<TxnStage>,
    lifecycle: parking_lot::Mutex<()>,
}

impl DatabaseTransaction<'_> {
    /// Stage a record for `pool` without changing its live pack or index.
    ///
    /// # Errors
    /// Returns an error if the transaction's bounded staging budget is
    /// exhausted or it has already been committed/aborted.
    pub fn put(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        data: &crate::storage::NodeData,
    ) -> io::Result<()> {
        let _lifecycle = self.lifecycle.lock();
        self.stage
            .stage_put(pool, collection_id, node_id, data.bytes.to_vec())
    }

    /// Stage removal of a collection.
    ///
    /// # Errors
    /// Returns an error if the transaction's staging budget is exhausted or
    /// it has already been committed/aborted.
    pub fn delete_collection(&self, pool: ShardType, collection_id: [u8; 16]) -> io::Result<()> {
        let _lifecycle = self.lifecycle.lock();
        self.stage.stage_delete_collection(pool, collection_id)
    }

    /// Read records through this transaction: its own staged writes first,
    /// then the live pool. Results are in the order of `node_ids`.
    ///
    /// The newest staged mutation for a record wins. A staged collection delete
    /// hides that collection's records, both staged and live, until a later
    /// staged put writes one back. Nothing here is visible to other readers.
    ///
    /// # Errors
    /// Returns an error if the live pool cannot be read.
    pub fn get(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[NodeId],
    ) -> Result<Vec<Option<crate::storage::NodeData>>, StorageError> {
        // Resolve against the stage under the lifecycle lock, which is what
        // keeps that answer consistent with a concurrent commit or abort. The
        // live read below is only for records the stage says nothing about, so
        // it needs no lock and must not hold up a commit behind a large read.
        let staged = {
            let _lifecycle = self.lifecycle.lock();
            self.stage.lookup_many(pool, collection_id, node_ids)
        };
        let mut results = Vec::with_capacity(node_ids.len());
        let mut live_slots = Vec::new();
        let mut live_ids = Vec::new();
        for (slot, (node_id, lookup)) in node_ids.iter().zip(staged).enumerate() {
            match lookup {
                StagedLookup::Put(payload) => results.push(Some(crate::storage::NodeData::new(
                    bytes::Bytes::from(payload),
                ))),
                StagedLookup::Deleted => results.push(None),
                StagedLookup::Absent => {
                    results.push(None);
                    live_slots.push(slot);
                    live_ids.push(*node_id);
                }
            }
        }
        if !live_ids.is_empty() {
            let live = self
                .database
                .pool(pool)
                .get_many(collection_id, &live_ids)?;
            for (slot, value) in live_slots.into_iter().zip(live) {
                results[slot] = value;
            }
        }
        Ok(results)
    }

    /// Commit the staged mutations. Pack/index application is performed once;
    /// journal publication can be retried if the post-commit callback fails.
    ///
    /// # Errors
    /// Returns an error if storage application or shared-WAL publication
    /// fails. After storage application succeeds, retrying this method is safe.
    pub fn commit(&self) -> Result<(), StorageError> {
        let _lifecycle = self.lifecycle.lock();
        if self.stage.state() == TxnStageState::Active {
            if self.stage.is_empty() {
                self.stage
                    .mark_empty_published()
                    .map_err(StorageError::Io)?;
                return Ok(());
            }
            let started = Instant::now();
            let mut overlay = Some(
                self.database
                    .activate_transaction_overlay(self.stage.touched_pools())?,
            );
            self.database.commit_phases.overlay.add(started.elapsed());
            let started = Instant::now();
            let published = self.database.publish_transaction(&self.stage);
            self.database.commit_phases.publish.add(started.elapsed());
            if let Err(error) = published {
                drop(overlay.take());
                return Err(StorageError::Io(error));
            }
            let Some(receipt) = self.stage.published_receipt() else {
                // An empty transaction has no journal group to recover. The
                // stage is completed by publish() itself, so its overlay can
                // be released without entering the recovery queue.
                if self.stage.state() == TxnStageState::Published {
                    drop(overlay.take());
                    return Ok(());
                }
                drop(overlay.take());
                return Err(StorageError::Internal(
                    "published transaction has no journal receipt".to_owned(),
                ));
            };
            // Keep recovery from removing a duplicate queue entry between
            // registration and transferring ownership of this overlay.
            let started = Instant::now();
            let _recovery_lifecycle = self.database.recovery_lifecycle.lock();
            self.database
                .commit_phases
                .register_wait
                .add(started.elapsed());
            self.database.register_new_recovery_stage(
                Arc::clone(&self.stage),
                receipt,
                &mut overlay,
            )?;
        }
        if matches!(self.stage.state(), TxnStageState::JournalPublished) {
            self.stage
                .begin_materialization()
                .map_err(StorageError::Io)?;
        }
        if matches!(
            self.stage.state(),
            TxnStageState::JournalPublished | TxnStageState::Materializing
        ) {
            // Take this lock before checking the recovery queue. A recovery
            // worker may otherwise finish and remove this stage between the
            // membership check and lock acquisition.
            let started = Instant::now();
            let _recovery_lifecycle = self.database.recovery_lifecycle.lock();
            self.database
                .commit_phases
                .materialize_wait
                .add(started.elapsed());
            match self.stage.state() {
                TxnStageState::JournalPublished => {
                    self.stage
                        .begin_materialization()
                        .map_err(StorageError::Io)?;
                }
                TxnStageState::Materializing => {
                    self.database
                        .ensure_recovery_stage_registered(&self.stage)?;
                }
                _ => return Ok(()),
            }
            let started = Instant::now();
            let materialized = self.database.materialize_transaction(&self.stage);
            self.database
                .commit_phases
                .materialize
                .add(started.elapsed());
            materialized?;
            self.stage.mark_published().map_err(StorageError::Io)?;
            self.database.finish_recovery_stage(&self.stage)?;
        }
        Ok(())
    }

    /// Abort the transaction. Since staged mutations have not touched storage,
    /// abort is O(1) and leaves no revocation records behind.
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] if journal publication or
    /// materialization has already started. A published transaction cannot be
    /// rolled back through this API.
    pub fn abort(&self) -> Result<(), StorageError> {
        let _lifecycle = self.lifecycle.lock();
        if self.stage.state() != TxnStageState::Active {
            return Err(StorageError::Internal(
                "a published or materializing transaction cannot be aborted".to_owned(),
            ));
        }
        self.stage.discard();
        Ok(())
    }

    /// Current transaction lifecycle state.
    #[must_use]
    pub fn state(&self) -> TxnStageState {
        self.stage.state()
    }
}

impl SharedDatabase {
    /// Open `root` for writing through one shared WAL using default pool policies.
    ///
    /// The root is initialized with the shared layout if it does not yet
    /// exist. This blocks with `WouldBlock` if another live process holds the
    /// root writer lock.
    ///
    /// # Errors
    /// Returns an error if the root lock is held, the shared segment cannot be
    /// opened, or any pool cannot be opened or attached.
    pub fn open(root: PathBuf) -> Result<Self, StorageError> {
        Self::open_with_policies(root, PoolPolicies::default())
    }

    /// Open `root` for writing through one shared WAL with explicit per-pool policies.
    ///
    /// The root is initialized with the shared layout if it does not yet
    /// exist. This blocks with `WouldBlock` if another live process holds the
    /// root writer lock.
    ///
    /// # Errors
    /// Returns an error if the root lock is held, the shared segment cannot be
    /// opened, or any pool cannot be opened or attached.
    pub fn open_with_policies(root: PathBuf, policies: PoolPolicies) -> Result<Self, StorageError> {
        let layout = DatabaseLayout::open(root)?;
        let wal_path = layout.shared_wal_path();
        let lock = SharedWalLock::acquire(layout.root())?;
        let seed_lsn = shared_wal_seed_lsn(&layout)?;
        let (journal, scan) = Journal::open_shared_with_base(&wal_path, seed_lsn)?;
        let coordinator = Arc::new(JournalCoordinator::new(journal, &scan));

        let mut pools = Vec::with_capacity(ShardType::ALL.len());
        for shard in ShardType::ALL {
            let dir = layout.pool_dir(shard)?;
            let policy = policies.for_shard(shard);
            let store =
                PackfileStorage::open_with_policies(dir, policy.compress, policy.checksum_policy)?;
            store.enable_shared_journal(Arc::clone(&coordinator), shard)?;
            store.replay_journal()?;
            pools.push(Arc::new(store));
        }
        let pools: [Arc<PackfileStorage>; 3] = pools.try_into().map_err(|_| {
            StorageError::Internal("a shared database must open exactly three pools".into())
        })?;

        // This process owns every pool, so it can make a lagging one checkpoint
        // when its silence has stalled WAL reclaim into the emergency zone. The
        // coordinator holds only weak references, so it does not keep the pools
        // alive.
        let weak_pools: Vec<std::sync::Weak<PackfileStorage>> =
            pools.iter().map(Arc::downgrade).collect();
        coordinator.set_blocker_remediation(move |pool| {
            let storage = weak_pools
                .get(shard_index(pool))
                .and_then(std::sync::Weak::upgrade);
            if let Some(Err(error)) = storage.map(|storage| storage.force_index_checkpoint()) {
                eprintln!("warning: checkpointing {pool:?} to unblock WAL reclaim failed: {error}");
            }
        });

        Ok(Self {
            layout,
            coordinator,
            pools,
            recovery_queue: parking_lot::Mutex::new(Vec::new()),
            recovery_lifecycle: parking_lot::Mutex::new(()),
            commit_phases: CommitPhases::default(),
            _lock: lock,
        })
    }

    /// The validated database layout.
    #[must_use]
    pub fn layout(&self) -> &DatabaseLayout {
        &self.layout
    }

    /// The one coordinator every pool publishes through.
    #[must_use]
    pub fn coordinator(&self) -> &Arc<JournalCoordinator> {
        &self.coordinator
    }

    /// The open store for `shard`.
    #[must_use]
    pub fn pool(&self, shard: ShardType) -> &Arc<PackfileStorage> {
        &self.pools[shard_index(shard)]
    }

    /// The State store (`ShardType::State`).
    #[must_use]
    pub fn state(&self) -> &Arc<PackfileStorage> {
        self.pool(ShardType::State)
    }

    /// The Event DAG store (`ShardType::EventDag`).
    #[must_use]
    pub fn event_dag(&self) -> &Arc<PackfileStorage> {
        self.pool(ShardType::EventDag)
    }

    /// The Edges store (`ShardType::Edges`), hosting PREV and AUTH edge collections.
    #[must_use]
    pub fn edges(&self) -> &Arc<PackfileStorage> {
        self.pool(ShardType::Edges)
    }

    /// Begin a storage transaction whose writes are invisible until commit.
    #[must_use]
    pub fn begin_transaction(&self) -> DatabaseTransaction<'_> {
        DatabaseTransaction {
            database: DatabaseRef::Borrowed(self),
            stage: Arc::new(TxnStage::new()),
            lifecycle: parking_lot::Mutex::new(()),
        }
    }

    /// Begin a transaction that owns a handle to this database, so it is not
    /// tied to a borrow and can be held across an FFI boundary or moved to
    /// another thread. It runs the same commit sequence as
    /// [`Self::begin_transaction`]: overlay activation, journal publication,
    /// recovery registration and materialization.
    #[must_use]
    pub fn begin_owned_transaction(self: &Arc<Self>) -> DatabaseTransaction<'static> {
        DatabaseTransaction {
            database: DatabaseRef::Owned(Arc::clone(self)),
            stage: Arc::new(TxnStage::new()),
            lifecycle: parking_lot::Mutex::new(()),
        }
    }

    /// Where transaction commits have spent their time so far.
    #[must_use]
    pub fn commit_phase_stats(&self) -> CommitPhaseStats {
        let phases = &self.commit_phases;
        CommitPhaseStats {
            overlay: phases.overlay.snapshot(),
            publish: phases.publish.snapshot(),
            register_wait: phases.register_wait.snapshot(),
            materialize_wait: phases.materialize_wait.snapshot(),
            materialize: phases.materialize.snapshot(),
        }
    }

    /// Activate the read overlay on each pool in `touched` (indexed like
    /// [`ShardType::ALL`]). A pool with no staged writes has nothing the
    /// overlay would need to show, so activating it would only rescan the
    /// journal for no reader.
    fn activate_transaction_overlay(
        &self,
        touched: [bool; ShardType::ALL.len()],
    ) -> Result<TransactionOverlayGuard, StorageError> {
        let wal_path = self.layout.shared_wal_path();
        let mut held: Vec<Arc<PackfileStorage>> = Vec::with_capacity(touched.len());
        for (index, pool) in ShardType::ALL.into_iter().enumerate() {
            if !touched.get(index).copied().unwrap_or(false) {
                continue;
            }
            let storage = Arc::clone(&self.pools[index]);
            if let Err(error) = storage.activate_transaction_overlay(&wal_path, pool) {
                for previous in &held {
                    previous.deactivate_transaction_overlay();
                }
                return Err(error);
            }
            held.push(storage);
        }
        Ok(TransactionOverlayGuard { pools: held })
    }

    fn register_new_recovery_stage(
        &self,
        stage: Arc<TxnStage>,
        receipt: CommitReceipt,
        overlay: &mut Option<TransactionOverlayGuard>,
    ) -> Result<(), StorageError> {
        if stage.published_receipt() != Some(receipt) {
            return Err(StorageError::Internal(
                "transaction recovery receipt does not match its stage".to_owned(),
            ));
        }
        let mut queue = self.recovery_queue.lock();
        if queue.iter().any(|candidate| {
            candidate.receipt.first_lsn == receipt.first_lsn
                && candidate.receipt.last_lsn == receipt.last_lsn
        }) {
            return Err(StorageError::Internal(
                "transaction recovery stage is already registered".to_owned(),
            ));
        }
        let overlay = overlay.take().ok_or_else(|| {
            StorageError::Internal("transaction overlay has already been transferred".to_owned())
        })?;
        queue.push(RecoveryItem {
            receipt,
            stage,
            overlay,
        });
        Ok(())
    }

    fn ensure_recovery_stage_registered(&self, stage: &TxnStage) -> Result<(), StorageError> {
        let Some(receipt) = stage.published_receipt() else {
            return Err(StorageError::Internal(
                "materializing transaction has no journal receipt".to_owned(),
            ));
        };
        if self.recovery_queue.lock().iter().any(|candidate| {
            candidate.receipt.first_lsn == receipt.first_lsn
                && candidate.receipt.last_lsn == receipt.last_lsn
        }) {
            Ok(())
        } else {
            Err(StorageError::Internal(
                "materializing transaction is not registered for recovery".to_owned(),
            ))
        }
    }

    fn finish_recovery_stage(&self, stage: &TxnStage) -> Result<(), StorageError> {
        let mut queue = self.recovery_queue.lock();
        if let Some(index) = queue
            .iter()
            .position(|candidate| std::ptr::eq(candidate.stage.as_ref(), stage))
        {
            let item = queue.swap_remove(index);
            // Its writes are all in the packs now, so pool coverage may pass it.
            self.coordinator
                .transaction_materialized(item.receipt.first_lsn);
            drop(item.overlay);
            Ok(())
        } else {
            Err(StorageError::Internal(
                "published transaction is missing its recovery stage".to_owned(),
            ))
        }
    }

    fn materialize_transaction(&self, stage: &TxnStage) -> Result<(), StorageError> {
        let batches = stage.snapshot_mutations();
        for pool in ShardType::ALL {
            for (index, mutation) in batches[shard_index(pool)].iter().enumerate() {
                if stage.mutation_applied(pool, index) {
                    continue;
                }
                self.pool(pool).apply_transaction_mutation(mutation)?;
                stage
                    .mark_mutation_applied(pool, index)
                    .map_err(StorageError::Io)?;
            }
        }
        Ok(())
    }

    /// Materialize published transactions retained by the database recovery
    /// queue. This is the explicit recovery/worker boundary; transaction
    /// destructors never perform storage I/O.
    ///
    /// # Errors
    /// Returns the first materialization error. The corresponding stage stays
    /// queued and its overlay remains active for a later retry.
    pub fn recover_pending_transactions(&self) -> Result<(), StorageError> {
        let _recovery_lifecycle = self.recovery_lifecycle.lock();
        let stages = self
            .recovery_queue
            .lock()
            .iter()
            .map(|item| (item.receipt, Arc::clone(&item.stage)))
            .collect::<Vec<_>>();
        for (receipt, stage) in stages {
            debug_assert_eq!(
                stage.published_receipt(),
                Some(receipt),
                "recovery queue receipt must identify its stage"
            );
            if stage.state() == TxnStageState::JournalPublished {
                stage.begin_materialization().map_err(StorageError::Io)?;
            }
            if stage.state() != TxnStageState::Materializing {
                continue;
            }
            self.materialize_transaction(&stage)?;
            stage.mark_published().map_err(StorageError::Io)?;
            self.finish_recovery_stage(&stage)?;
        }
        Ok(())
    }

    /// Publish one transaction-owned mutation group through the shared WAL.
    ///
    /// All mutations staged for the three pools are appended as one tagged
    /// journal group. A later durability operation may fsync that group, but
    /// readers see either the complete group or none of it.
    ///
    /// # Errors
    /// Returns an error if staging is inactive, the coordinator is poisoned,
    /// or the journal group cannot be appended.
    pub fn publish_transaction(&self, stage: &TxnStage) -> io::Result<()> {
        stage.publish(
            Some(&self.coordinator),
            Some(&self.coordinator),
            Some(&self.coordinator),
        )
    }
}

/// The base LSN a fresh shared WAL for `layout` must start above: one past the
/// highest per-pool `journal.lsn` recorded beside a pool's last checkpoint.
///
/// If the root's WAL is missing while a pool has recorded checkpoint coverage,
/// the segment that replaces it must begin above every one of those
/// watermarks, so each pool's coverage stays meaningful in the single shared
/// LSN space: a restarted writer can neither replay past a pool's coverage nor
/// reclaim beneath a fresh frame that only looks covered because numbering
/// restarted. Returns `1` for a root with no recorded coverage (a brand-new
/// database).
///
/// This is the seed `Journal::open_shared_with_base` consumes; it is kept
/// internal so a caller cannot create a shared segment with an arbitrary base
/// that disagrees with the pools' recorded coverage. Open a root through
/// [`SharedDatabase::open`] instead.
///
/// # Errors
/// Returns an error if a pool directory cannot be created or read.
pub(crate) fn shared_wal_seed_lsn(layout: &DatabaseLayout) -> Result<u64, StorageError> {
    let mut watermark = 0_u64;
    for shard in ShardType::ALL {
        let dir = layout.pool_dir(shard)?;
        watermark = watermark.max(PackfileStorage::read_journal_lsn(&dir));
    }
    Ok(watermark.saturating_add(1))
}

/// Index of `shard` in the fixed `[State, EventDag, Edges]` pool array.
const fn shard_index(shard: ShardType) -> usize {
    match shard {
        ShardType::State => 0,
        ShardType::EventDag => 1,
        ShardType::Edges => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::{shard_index, DatabaseTransaction, SharedDatabase, TransactionOverlayGuard};
    use crate::journal::Journal;
    use crate::layout::ShardType;
    use crate::packfile::storage::PackfileStorage;
    use crate::storage::{NodeData, NodeId, StorageEngine};
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};

    fn test_root(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("mtxdb-shared-db-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn prepare_partial_materialization(
        database: &SharedDatabase,
        transaction: &DatabaseTransaction<'_>,
        overlay: &mut Option<TransactionOverlayGuard>,
        state_collection: [u8; 16],
        event_collection: [u8; 16],
    ) {
        transaction
            .put(
                ShardType::State,
                state_collection,
                node(1),
                &NodeData::new(bytes::Bytes::from_static(b"state")),
            )
            .unwrap();
        transaction
            .put(
                ShardType::EventDag,
                event_collection,
                node(2),
                &NodeData::new(bytes::Bytes::from_static(b"event")),
            )
            .unwrap();
        *overlay = Some(activate_for(database, &transaction.stage));
        database.publish_transaction(&transaction.stage).unwrap();
        transaction.stage.begin_materialization().unwrap();
        let receipt = transaction.stage.published_receipt().unwrap();
        database
            .register_new_recovery_stage(Arc::clone(&transaction.stage), receipt, overlay)
            .unwrap();
        let mutation =
            transaction.stage.snapshot_mutations()[shard_index(ShardType::State)][0].clone();
        database
            .pool(ShardType::State)
            .apply_transaction_mutation(&mutation)
            .unwrap();
        transaction
            .stage
            .mark_mutation_applied(ShardType::State, 0)
            .unwrap();
    }

    fn payload_of(value: Option<&NodeData>) -> Option<Vec<u8>> {
        value.map(|data| data.bytes.to_vec())
    }

    fn data(bytes: &'static [u8]) -> NodeData {
        NodeData::new(bytes::Bytes::from_static(bytes))
    }

    fn live_get(
        database: &SharedDatabase,
        pool: ShardType,
        collection: [u8; 16],
        id: NodeId,
    ) -> Option<Vec<u8>> {
        payload_of(database.pool(pool).get(&collection, &id).unwrap().as_ref())
    }

    /// Activate the overlay for exactly the pools `stage` has writes for, as
    /// commit does.
    fn activate_for(
        database: &SharedDatabase,
        stage: &crate::journal::TxnStage,
    ) -> TransactionOverlayGuard {
        database
            .activate_transaction_overlay(stage.touched_pools())
            .unwrap()
    }

    fn own_get(
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        collection: [u8; 16],
        id: NodeId,
    ) -> Option<Vec<u8>> {
        let values = transaction.get(pool, &collection, &[id]).unwrap();
        payload_of(values[0].as_ref())
    }

    /// A transaction reads its own staged writes, while the live pool stays
    /// untouched until commit. After commit the live pool sees the write.
    #[test]
    fn a_transaction_reads_its_own_writes_and_others_do_not() {
        let root = test_root("transaction_reads_own_writes");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        let transaction = database.begin_transaction();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"staged"))
            .unwrap();

        assert_eq!(
            own_get(&transaction, ShardType::State, collection, node(1)),
            Some(b"staged".to_vec())
        );
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            None,
            "no other reader may see an uncommitted write"
        );
        transaction.commit().unwrap();
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            Some(b"staged".to_vec()),
            "commit makes the write visible to everyone"
        );
        assert_eq!(
            own_get(&transaction, ShardType::State, collection, node(1)),
            Some(b"staged".to_vec()),
            "the committed transaction still reads it, now from the live pool"
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The newest staged write wins, reads fall back to live data for records
    /// the transaction has not touched, and results keep the request order.
    #[test]
    fn transaction_reads_prefer_the_newest_write_then_the_live_pool() {
        let root = test_root("transaction_reads_layering");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        database
            .pool(ShardType::State)
            .put(&collection, &node(1), &data(b"live-1"))
            .unwrap();
        database
            .pool(ShardType::State)
            .put(&collection, &node(2), &data(b"live-2"))
            .unwrap();

        let transaction = database.begin_transaction();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"first"))
            .unwrap();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"second"))
            .unwrap();
        let got = transaction
            .get(ShardType::State, &collection, &[node(2), node(1), node(9)])
            .unwrap();
        assert_eq!(payload_of(got[0].as_ref()), Some(b"live-2".to_vec()));
        assert_eq!(payload_of(got[1].as_ref()), Some(b"second".to_vec()));
        assert_eq!(payload_of(got[2].as_ref()), None);
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            Some(b"live-1".to_vec()),
            "live data is untouched until commit"
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A staged collection delete hides the collection's live records and the
    /// transaction's own earlier puts; a put staged after it is newer and is
    /// seen. Other collections are unaffected, and the live pool keeps
    /// everything until commit.
    #[test]
    fn a_staged_collection_delete_hides_older_data_but_not_newer_puts() {
        let root = test_root("transaction_reads_delete");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let doomed = [7u8; 16];
        let other = [8u8; 16];
        for collection in [doomed, other] {
            database
                .pool(ShardType::State)
                .put(&collection, &node(1), &data(b"live"))
                .unwrap();
        }

        let transaction = database.begin_transaction();
        transaction
            .put(ShardType::State, doomed, node(2), &data(b"before-delete"))
            .unwrap();
        transaction
            .delete_collection(ShardType::State, doomed)
            .unwrap();
        transaction
            .put(ShardType::State, doomed, node(3), &data(b"after-delete"))
            .unwrap();

        let got = transaction
            .get(ShardType::State, &doomed, &[node(1), node(2), node(3)])
            .unwrap();
        assert_eq!(payload_of(got[0].as_ref()), None, "live record hidden");
        assert_eq!(payload_of(got[1].as_ref()), None, "earlier put hidden");
        assert_eq!(
            payload_of(got[2].as_ref()),
            Some(b"after-delete".to_vec()),
            "a later put is visible"
        );
        assert_eq!(
            own_get(&transaction, ShardType::State, other, node(1)),
            Some(b"live".to_vec()),
            "other collections are unaffected"
        );
        assert_eq!(
            live_get(&database, ShardType::State, doomed, node(1)),
            Some(b"live".to_vec()),
            "the delete is not applied to the live pool before commit"
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// An aborted transaction leaves nothing behind: its staged writes are gone
    /// for the transaction and never reached the live pool.
    #[test]
    fn an_aborted_transaction_leaves_no_trace() {
        let root = test_root("transaction_reads_abort");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        let transaction = database.begin_transaction();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"rolled-back"))
            .unwrap();
        assert_eq!(
            own_get(&transaction, ShardType::State, collection, node(1)),
            Some(b"rolled-back".to_vec())
        );
        transaction.abort().unwrap();
        assert_eq!(
            own_get(&transaction, ShardType::State, collection, node(1)),
            None,
            "an aborted transaction no longer serves its staged writes"
        );
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            None
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Two open transactions do not see each other's staged writes.
    #[test]
    fn open_transactions_are_isolated_from_each_other() {
        let root = test_root("transaction_isolation");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        let first = database.begin_transaction();
        let second = database.begin_transaction();
        first
            .put(ShardType::State, collection, node(1), &data(b"first"))
            .unwrap();
        second
            .put(ShardType::State, collection, node(2), &data(b"second"))
            .unwrap();

        assert_eq!(
            own_get(&first, ShardType::State, collection, node(1)),
            Some(b"first".to_vec())
        );
        assert_eq!(own_get(&first, ShardType::State, collection, node(2)), None);
        assert_eq!(
            own_get(&second, ShardType::State, collection, node(2)),
            Some(b"second".to_vec())
        );
        assert_eq!(
            own_get(&second, ShardType::State, collection, node(1)),
            None
        );
        drop(first);
        drop(second);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// While a commit is part-way (journal published, only some mutations
    /// applied to the pools, as after a failed post-commit callback that will be
    /// retried) the stage no longer answers, but the transaction must still read
    /// its own writes. The live store serves the whole group through the
    /// transaction overlay, including the mutation not yet applied.
    #[test]
    fn a_transaction_still_reads_its_writes_while_its_commit_is_part_way() {
        let root = test_root("transaction_reads_mid_commit");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let state_collection = [0x61; 16];
        let event_collection = [0x62; 16];
        let transaction = database.begin_transaction();
        let mut overlay = None;
        prepare_partial_materialization(
            &database,
            &transaction,
            &mut overlay,
            state_collection,
            event_collection,
        );

        assert_eq!(
            own_get(&transaction, ShardType::State, state_collection, node(1)),
            Some(b"state".to_vec()),
            "an applied mutation is read from the pool"
        );
        assert_eq!(
            own_get(&transaction, ShardType::EventDag, event_collection, node(2)),
            Some(b"event".to_vec()),
            "a published but not yet applied mutation is read through the overlay"
        );
        drop(overlay);
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A large batch of reads against a large stage preserves request order
    /// and resolves every id. This is a correctness test for the batch
    /// layering; the single-pass complexity is implemented by `lookup_many`.
    #[test]
    fn a_large_batch_read_preserves_order_and_values() {
        const RECORDS: u16 = 4000;
        let root = test_root("transaction_reads_large_batch");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        let id_of = |index: u16| -> NodeId {
            let mut id = [0u8; 16];
            id[..2].copy_from_slice(&index.to_le_bytes());
            id
        };
        let transaction = database.begin_transaction();
        for index in 0..RECORDS {
            transaction
                .put(
                    ShardType::State,
                    collection,
                    id_of(index),
                    &NodeData::new(bytes::Bytes::from(index.to_le_bytes().to_vec())),
                )
                .unwrap();
        }
        let ids: Vec<NodeId> = (0..RECORDS).map(id_of).collect();
        let values = transaction
            .get(ShardType::State, &collection, &ids)
            .unwrap();
        assert_eq!(
            transaction.stage.lookup_many_calls(),
            1,
            "the batch is resolved by one staged scan"
        );
        assert_eq!(values.len(), usize::from(RECORDS));
        for (index, value) in values.iter().enumerate() {
            let expected = u16::try_from(index).unwrap().to_le_bytes().to_vec();
            assert_eq!(payload_of(value.as_ref()), Some(expected));
        }
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// An owned transaction is not tied to a borrow: it can move to another
    /// thread and still runs the whole commit sequence, so the data reaches the
    /// live pool.
    #[test]
    fn an_owned_transaction_moves_across_threads_and_commits() {
        let root = test_root("owned_transaction");
        let database = Arc::new(SharedDatabase::open(root.clone()).unwrap());
        let collection = [7u8; 16];
        let transaction = database.begin_owned_transaction();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"owned"))
            .unwrap();

        let handle = std::thread::spawn(move || {
            assert_eq!(
                own_get(&transaction, ShardType::State, collection, node(1)),
                Some(b"owned".to_vec())
            );
            transaction.commit().unwrap();
        });
        handle.join().unwrap();
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            Some(b"owned".to_vec())
        );
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Staged records are scoped to their pool: the same collection and id in
    /// another pool is a different record.
    #[test]
    fn staged_reads_are_scoped_to_their_pool() {
        let root = test_root("transaction_reads_pool_scope");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        let transaction = database.begin_transaction();
        transaction
            .put(ShardType::State, collection, node(1), &data(b"state"))
            .unwrap();
        assert_eq!(
            own_get(&transaction, ShardType::EventDag, collection, node(1)),
            None
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Regression: once a checkpoint had reclaimed the shared WAL, every
    /// transaction commit failed with a `WouldBlock` that no retry cleared and
    /// the staged data was never applied. The overlay setup ran a reader's gap
    /// check against a writer that never sets its own read coverage. A plain
    /// pool sync is enough to reclaim, so this is the ordinary case.
    #[test]
    fn a_transaction_commits_after_a_checkpoint_has_reclaimed_the_wal() {
        let root = test_root("commit_after_checkpoint");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [7u8; 16];
        database
            .pool(ShardType::State)
            .put(&[0x99; 16], &node(1), &data(b"seed"))
            .unwrap();
        database.pool(ShardType::State).sync().unwrap();
        let wal =
            crate::journal::Journal::scan_read_only(database.layout().shared_wal_path()).unwrap();
        assert!(
            wal.base_lsn > 1,
            "the sync must have reclaimed the WAL past LSN 1 for this to test anything"
        );

        let transaction = database.begin_transaction();
        transaction
            .put(
                ShardType::State,
                collection,
                node(1),
                &data(b"after-checkpoint"),
            )
            .unwrap();
        transaction.commit().unwrap();
        assert_eq!(
            live_get(&database, ShardType::State, collection, node(1)),
            Some(b"after-checkpoint".to_vec())
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The owned transaction is what crosses an FFI boundary, which needs it to
    /// be both `Send` and `Sync`. This fails to compile if that stops holding.
    #[test]
    fn an_owned_transaction_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DatabaseTransaction<'static>>();
    }

    fn node(id: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = id;
        bytes
    }

    #[test]
    fn opens_a_root_with_all_pools_behind_one_coordinator() {
        let root = test_root("one_coordinator");
        let db = SharedDatabase::open(root.clone()).unwrap();
        for shard in ShardType::ALL {
            let pool = db.pool(shard);
            assert!(
                Arc::ptr_eq(pool.journal().as_ref().unwrap(), db.coordinator()),
                "every pool must share the one coordinator"
            );
        }
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn transaction_abort_keeps_staged_mutations_invisible() {
        let root = test_root("transaction_abort");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x51; 16];
        let node_id = node(1);
        let transaction = db.begin_transaction();
        transaction
            .put(
                ShardType::State,
                collection,
                node_id,
                &NodeData::new(bytes::Bytes::from_static(b"not committed")),
            )
            .unwrap();
        assert!(db
            .pool(ShardType::State)
            .get(&collection, &node_id)
            .unwrap()
            .is_none());
        transaction.abort().unwrap();
        drop(transaction);
        assert!(db
            .pool(ShardType::State)
            .get(&collection, &node_id)
            .unwrap()
            .is_none());
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn transaction_rejects_mutations_after_completion() {
        let root = test_root("transaction_terminal_mutations");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x5A; 16];
        let node_id = node(1);
        let data = NodeData::new(bytes::Bytes::from_static(b"late"));

        let published = database.begin_transaction();
        published.commit().unwrap();
        assert!(published
            .put(ShardType::State, collection, node_id, &data)
            .is_err());
        assert!(published
            .delete_collection(ShardType::State, collection)
            .is_err());

        let discarded = database.begin_transaction();
        discarded.abort().unwrap();
        assert!(discarded
            .put(ShardType::State, collection, node_id, &data)
            .is_err());
        assert!(discarded
            .delete_collection(ShardType::State, collection)
            .is_err());

        drop(database);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The overlay is reused across lone transactions to avoid rescanning the
    /// WAL. A stale entry from an earlier transaction must never shadow a value
    /// written directly to the pool afterwards.
    #[test]
    fn reused_overlay_does_not_shadow_a_later_direct_write() {
        let root = test_root("overlay_reuse_shadow");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x71; 16];
        let first = db.begin_transaction();
        first
            .put(ShardType::EventDag, collection, node(1), &data(b"old"))
            .unwrap();
        first.commit().unwrap();
        db.pool(ShardType::EventDag)
            .put(&collection, &node(1), &data(b"new"))
            .unwrap();
        let second = db.begin_transaction();
        second
            .put(ShardType::EventDag, collection, node(2), &data(b"other"))
            .unwrap();
        second.commit().unwrap();
        assert_eq!(
            live_get(&db, ShardType::EventDag, collection, node(1)),
            Some(b"new".to_vec())
        );
        assert_eq!(
            live_get(&db, ShardType::EventDag, collection, node(2)),
            Some(b"other".to_vec())
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A retained overlay is cleared with its delete boundary, so an earlier
    /// transaction's collection delete must neither resurrect old records nor
    /// hide records written after it.
    #[test]
    fn reused_overlay_keeps_a_collection_delete_from_resurrecting_records() {
        let root = test_root("overlay_reuse_delete");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x72; 16];
        let put = |id: u8, value: &'static [u8]| {
            let txn = db.begin_transaction();
            txn.put(ShardType::EventDag, collection, node(id), &data(value))
                .unwrap();
            txn.commit().unwrap();
        };
        put(1, b"before");
        let delete = db.begin_transaction();
        delete
            .delete_collection(ShardType::EventDag, collection)
            .unwrap();
        delete.commit().unwrap();
        put(2, b"after");
        assert_eq!(
            live_get(&db, ShardType::EventDag, collection, node(1)),
            None
        );
        assert_eq!(
            live_get(&db, ShardType::EventDag, collection, node(2)),
            Some(b"after".to_vec())
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Lone transactions alternate between two pools while a checkpoint
    /// reclaims the WAL in between. The retained overlays must rebuild past the
    /// replaced segment and keep applying only their own pool's frames.
    #[test]
    fn reused_overlay_survives_reclaim_with_interleaved_pools() {
        let root = test_root("overlay_reuse_reclaim");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let state = [0x73; 16];
        let events = [0x74; 16];
        let commit_pair = |round: u8| {
            for (pool, collection) in [(ShardType::State, state), (ShardType::EventDag, events)] {
                let txn = db.begin_transaction();
                txn.put(pool, collection, node(round), &data(b"value"))
                    .unwrap();
                txn.commit().unwrap();
            }
        };
        for round in 1..=3 {
            commit_pair(round);
        }
        db.pool(ShardType::State).sync_all().unwrap();
        db.pool(ShardType::EventDag).sync_all().unwrap();
        for round in 4..=6 {
            commit_pair(round);
        }
        for round in 1..=6 {
            assert_eq!(
                live_get(&db, ShardType::State, state, node(round)),
                Some(b"value".to_vec())
            );
            assert_eq!(
                live_get(&db, ShardType::EventDag, events, node(round)),
                Some(b"value".to_vec())
            );
            // Each pool holds only its own collection.
            assert_eq!(live_get(&db, ShardType::State, events, node(round)), None);
            assert_eq!(live_get(&db, ShardType::EventDag, state, node(round)), None);
        }
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Commit cost must not grow with the unreclaimed WAL: a lone commit
    /// scans only the group it appended, however long the segment already is.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn lone_commits_scan_only_the_appended_suffix() {
        let root = test_root("overlay_scan_bounded");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x75; 16];
        let pool = db.pool(ShardType::EventDag);
        let commit = |seq: u16| {
            let before = pool.transaction_overlay_scanned_bytes();
            let txn = db.begin_transaction();
            for record in 0..4u8 {
                let mut id = [0u8; 16];
                id[..2].copy_from_slice(&seq.to_le_bytes());
                id[15] = record;
                txn.put(ShardType::EventDag, collection, id, &data(b"payload"))
                    .unwrap();
            }
            txn.commit().unwrap();
            pool.transaction_overlay_scanned_bytes() - before
        };
        for seq in 0..50 {
            commit(seq);
        }
        let early = commit(50);
        for seq in 51..500 {
            commit(seq);
        }
        let late = commit(500);
        assert!(early > 0, "a commit must scan its own group");
        assert!(
            late <= early.saturating_mul(2),
            "commit scanned {late} bytes after 500 commits but {early} after 50: \
             the overlay is rescanning the WAL"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// While a transaction holds the overlay it is on the read path, so its
    /// staged group is visible before materialization; once the last user
    /// releases it, the overlay leaves the read path, and writer reads stop
    /// refreshing a journal that only repeats the index.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn transaction_overlay_is_on_the_read_path_only_while_a_transaction_uses_it() {
        let root = test_root("overlay_read_path");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x76; 16];
        let pool = db.pool(ShardType::EventDag);
        assert!(!pool.read_journal_installed());
        let txn = db.begin_transaction();
        txn.put(ShardType::EventDag, collection, node(1), &data(b"staged"))
            .unwrap();
        let overlay = activate_for(&db, &txn.stage);
        db.publish_transaction(&txn.stage).unwrap();
        assert!(pool.read_journal_installed());
        let visible = pool.get_read_committed(&collection, &[node(1)]).unwrap();
        assert_eq!(payload_of(visible[0].as_ref()), Some(b"staged".to_vec()));
        drop(overlay);
        assert!(!pool.read_journal_installed());
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// After many lone commits and a checkpoint, a writer's read-committed reads
    /// must not have a journal overlay installed, and must still return every
    /// committed record.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn writer_reads_leave_the_overlay_off_after_commits_and_sync() {
        let root = test_root("overlay_writer_reads");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x77; 16];
        let pool = db.pool(ShardType::EventDag);
        for seq in 0..200u8 {
            let txn = db.begin_transaction();
            txn.put(ShardType::EventDag, collection, node(seq), &data(b"value"))
                .unwrap();
            txn.commit().unwrap();
            assert!(!pool.read_journal_installed());
        }
        pool.sync_all().unwrap();
        for seq in 0..200u8 {
            let read = pool.get_read_committed(&collection, &[node(seq)]).unwrap();
            assert!(read[0].is_some(), "record {seq} must stay readable");
        }
        assert!(!pool.read_journal_installed());
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// One thread commits while another reads: a reader must see every commit
    /// the writer has finished, whether or not the overlay is installed at that
    /// moment.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn concurrent_reads_see_every_finished_commit() {
        use std::sync::atomic::{AtomicU8, Ordering};
        let root = test_root("overlay_concurrent");
        let db = Arc::new(SharedDatabase::open(root.clone()).unwrap());
        let collection = [0x78; 16];
        let finished = Arc::new(AtomicU8::new(0));
        let writer = {
            let db = Arc::clone(&db);
            let finished = Arc::clone(&finished);
            std::thread::Builder::new()
                .stack_size(16 << 20)
                .spawn(move || {
                    for seq in 1..=200u8 {
                        let txn = db.begin_transaction();
                        txn.put(ShardType::EventDag, collection, node(seq), &data(b"value"))
                            .unwrap();
                        txn.commit().unwrap();
                        finished.store(seq, Ordering::Release);
                    }
                })
                .unwrap()
        };
        let pool = Arc::clone(db.pool(ShardType::EventDag));
        while finished.load(Ordering::Acquire) < 200 {
            let done = finished.load(Ordering::Acquire);
            if done > 0 {
                let read = pool.get_read_committed(&collection, &[node(done)]).unwrap();
                assert!(
                    read[0].is_some(),
                    "commit {done} finished but is unreadable"
                );
            }
        }
        writer.join().unwrap();
        drop(pool);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// While a transaction overlay is active, a read of a key the overlay does
    /// not hold must fall through to the live index once, not bounce between
    /// the overlay path and the durable path until the stack overflows.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn reading_an_unstaged_key_during_an_active_overlay_reaches_the_index() {
        let root = test_root("overlay_unstaged_read");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x79; 16];
        let pool = db.pool(ShardType::EventDag);
        pool.put(&collection, &node(1), &data(b"durable")).unwrap();
        let txn = db.begin_transaction();
        txn.put(ShardType::EventDag, collection, node(2), &data(b"staged"))
            .unwrap();
        let overlay = activate_for(&db, &txn.stage);
        db.publish_transaction(&txn.stage).unwrap();
        let by_get = pool.get(&collection, &node(1)).unwrap();
        assert_eq!(payload_of(by_get.as_ref()), Some(b"durable".to_vec()));
        let by_many = pool.get_many(&collection, &[node(1), node(3)]).unwrap();
        assert_eq!(payload_of(by_many[0].as_ref()), Some(b"durable".to_vec()));
        assert!(by_many[1].is_none());
        drop(overlay);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Small enough that a few hundred KiB of commits cross it.
    #[cfg(feature = "multi-reader")]
    const TEST_RECLAIM_TRIGGER_LEN: u64 = 256 << 10;

    /// Write committed transactions that stage into each of the `active` pools
    /// and sync every round, idle pools first while the segment is still
    /// large. Returns the largest WAL seen after a round's syncs (past round
    /// 0) and asserts that no idle pool ever rewrites its checkpoint or moves its
    /// `journal.lsn` after its first checkpoint, while every active pool does
    /// checkpoint once the segment is large. The segment can only be reclaimed
    /// once all the active pools have reported coverage.
    #[cfg(feature = "multi-reader")]
    fn run_bounded_wal_rounds(name: &str, active: &[ShardType], defer_rewrites: bool) -> u64 {
        use crate::PackfileStorage;
        assert!(!active.is_empty(), "at least one pool must be active");
        for (index, pool) in active.iter().enumerate() {
            assert!(
                !active.iter().take(index).any(|earlier| earlier == pool),
                "duplicate active pool: {pool:?}"
            );
        }
        let root = test_root(name);
        let db = SharedDatabase::open(root.clone()).unwrap();
        db.coordinator()
            .set_reclaim_trigger_len(TEST_RECLAIM_TRIGGER_LEN);
        if defer_rewrites {
            // A budget that would defer every structural rewrite for an hour must
            // not hold back the checkpoint the size trigger forces.
            for pool in ShardType::ALL {
                db.pool(pool)
                    .set_checkpoint_rewrite_budget(std::time::Duration::from_secs(3600), 0);
            }
        }
        let collection = [0x7a; 16];
        let wal = db.layout().shared_wal_path();
        let mut order: Vec<ShardType> = ShardType::ALL
            .into_iter()
            .filter(|pool| !active.contains(pool))
            .collect();
        order.extend_from_slice(active);
        let lsn_of = |pool: ShardType| {
            PackfileStorage::read_journal_lsn(&db.layout().pool_dir(pool).unwrap())
        };
        let mut idle_lsn: Vec<Option<u64>> = vec![None; ShardType::ALL.len()];
        let mut active_advanced = vec![false; ShardType::ALL.len()];
        let mut next = 0u64;
        let mut largest_after_sync = 0u64;
        for round in 0..8 {
            for _ in 0..60 {
                let txn = db.begin_transaction();
                for _ in 0..20 {
                    let mut id = [0u8; 16];
                    id[..8].copy_from_slice(&next.to_le_bytes());
                    next = next.saturating_add(1);
                    for pool in active {
                        txn.put(*pool, collection, id, &NodeData::from_slice(&[0x5a; 256]))
                            .unwrap();
                    }
                }
                txn.commit().unwrap();
            }
            for pool in &order {
                let before = db.pool(*pool).durable_coverage();
                db.pool(*pool).sync_all().unwrap();
                let advanced = db.pool(*pool).durable_coverage() > before;
                let index = ShardType::ALL
                    .iter()
                    .position(|known| known == pool)
                    .unwrap();
                if active.contains(pool) {
                    // Whether it took a full checkpoint or a delta coverage batch,
                    // the size trigger must make it advance its coverage.
                    active_advanced[index] |= round > 0 && advanced;
                } else if round == 0 {
                    idle_lsn[index] = Some(lsn_of(*pool));
                } else {
                    let timings = db.pool(*pool).sync_timings().unwrap();
                    assert!(
                        !advanced && timings.checkpoint.is_zero() && timings.delta_log.is_zero(),
                        "round {round}: idle {pool:?} did index or coverage work"
                    );
                    assert_eq!(
                        Some(lsn_of(*pool)),
                        idle_lsn[index],
                        "round {round}: idle {pool:?} moved its coverage"
                    );
                }
            }
            if round > 0 {
                largest_after_sync = largest_after_sync.max(std::fs::metadata(&wal).unwrap().len());
            }
        }
        for pool in active {
            let index = ShardType::ALL
                .iter()
                .position(|known| known == pool)
                .unwrap();
            assert!(
                active_advanced[index],
                "{pool:?} never advanced its coverage under the size trigger"
            );
        }
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        largest_after_sync
    }

    /// The WAL must stay below the reclaim trigger after each round's syncs.
    #[cfg(feature = "multi-reader")]
    fn assert_wal_bounded(name: &str, active: &[ShardType]) {
        assert_wal_bounded_with(name, active, false);
    }

    #[cfg(feature = "multi-reader")]
    fn assert_wal_bounded_with(name: &str, active: &[ShardType], defer_rewrites: bool) {
        let largest = run_bounded_wal_rounds(name, active, defer_rewrites);
        assert!(
            largest < TEST_RECLAIM_TRIGGER_LEN,
            "{active:?}: the WAL stayed at {largest} bytes after a round's syncs; it must be \
             reclaimed once past {TEST_RECLAIM_TRIGGER_LEN} bytes"
        );
    }

    /// `sync_all` after the first checkpoint only appends index deltas, which do
    /// not record journal coverage, so the shared WAL was never reclaimed again
    /// and grew until commits failed with "journal segment is full". A large
    /// segment must force a full checkpoint so the WAL stays bounded, whichever
    /// pool carries the frames: each pool must be compared with its own
    /// committed LSN and its own `journal.lsn`.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn repeated_syncs_keep_the_shared_wal_bounded_with_only_state_active() {
        assert_wal_bounded("wal_bounded_state", &[ShardType::State]);
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn repeated_syncs_keep_the_shared_wal_bounded_with_only_event_dag_active() {
        assert_wal_bounded("wal_bounded_event", &[ShardType::EventDag]);
    }

    #[cfg(feature = "multi-reader")]
    #[test]
    fn repeated_syncs_keep_the_shared_wal_bounded_with_only_edges_active() {
        assert_wal_bounded("wal_bounded_edges", &[ShardType::Edges]);
    }

    /// With two contributing pools the segment is only reclaimable once both
    /// have reported coverage, and the third stays idle throughout.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn repeated_syncs_keep_the_shared_wal_bounded_with_two_pools_active() {
        assert_wal_bounded("wal_bounded_two", &[ShardType::State, ShardType::EventDag]);
    }

    /// A configured checkpoint-rewrite deferral budget must not postpone the
    /// checkpoint the size trigger forces: the segment would fill and commits
    /// would fail.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_deferral_budget_does_not_hold_back_the_forced_reclaim() {
        assert_wal_bounded_with("wal_bounded_deferred", &[ShardType::EventDag], true);
    }

    /// A transaction that stages writes for one pool must not activate (and so
    /// rescan the journal for) the overlay on the others.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn overlay_is_activated_only_on_the_pools_a_transaction_stages() {
        let root = test_root("overlay_touched_pools");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let collection = [0x7b; 16];
        let txn = db.begin_transaction();
        txn.put(ShardType::EventDag, collection, node(1), &data(b"staged"))
            .unwrap();
        assert_eq!(txn.stage.touched_pools(), [false, true, false]);
        let overlay = activate_for(&db, &txn.stage);
        assert!(db.pool(ShardType::EventDag).read_journal_installed());
        assert!(!db.pool(ShardType::State).read_journal_installed());
        assert!(!db.pool(ShardType::Edges).read_journal_installed());
        drop(overlay);
        assert!(!db.pool(ShardType::EventDag).read_journal_installed());
        // A transaction that stages for two pools touches exactly those two.
        let mixed = db.begin_transaction();
        mixed
            .put(ShardType::State, collection, node(2), &data(b"state"))
            .unwrap();
        mixed
            .put(ShardType::Edges, collection, node(3), &data(b"edge"))
            .unwrap();
        assert_eq!(mixed.stage.touched_pools(), [true, false, true]);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Commit `count` transactions, each staging 20 records into every pool in
    /// `pools`. Stops at the first error.
    #[cfg(feature = "multi-reader")]
    fn commit_batch(
        db: &SharedDatabase,
        pools: &[ShardType],
        next: &mut u64,
        count: usize,
    ) -> Result<(), crate::storage::StorageError> {
        use crate::storage::StorageError;
        let collection = [0x7c; 16];
        for _ in 0..count {
            let txn = db.begin_transaction();
            for _ in 0..20 {
                let mut id = [0u8; 16];
                id[..8].copy_from_slice(&next.to_le_bytes());
                *next = next.saturating_add(1);
                for pool in pools {
                    txn.put(*pool, collection, id, &NodeData::from_slice(&[0x5a; 256]))
                        .map_err(StorageError::Io)?;
                }
            }
            txn.commit()?;
        }
        Ok(())
    }

    /// A small database whose segment cap is 1 MiB, so its trigger is 256 KiB
    /// and its emergency line 768 KiB.
    #[cfg(feature = "multi-reader")]
    fn small_segment_database(name: &str) -> (SharedDatabase, PathBuf) {
        let root = test_root(name);
        let db = SharedDatabase::open(root.clone()).unwrap();
        db.coordinator().set_segment_cap(1 << 20);
        (db, root)
    }

    /// `State` and `EventDag` both have committed frames, but only `EventDag` is ever
    /// synced. With nothing to remediate the lag, reclaim stalls: it must be
    /// reported with the pool named, `EventDag` must stop repeating a checkpoint
    /// that cannot help, and a commit that finally hits the hard limit must say
    /// which pool it is waiting on instead of failing silently.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_pool_that_never_reports_stalls_reclaim_visibly_and_without_thrashing() {
        let (db, root) = small_segment_database("lagging_no_remediation");
        db.coordinator().set_blocker_remediation(|_| {});
        let pools = [ShardType::State, ShardType::EventDag];
        let coordinator = db.coordinator();
        let emergency = coordinator.segment_cap() - coordinator.segment_cap() / 4;
        let trigger = coordinator.reclaim_trigger_len();
        let mut next = 0u64;
        // Syncs and checkpoints taken while the segment is stalled but short of
        // the emergency line, where the back-off applies.
        let mut backed_off_syncs = 0usize;
        let mut backed_off_checkpoints = 0usize;
        let mut rounds = 0usize;
        let failure = loop {
            // About 10 KiB per round: well under the retry growth (32 KiB).
            if let Err(error) = commit_batch(&db, &pools, &mut next, 1) {
                break error;
            }
            let in_back_off_zone =
                coordinator.is_reclaim_stalled() && coordinator.segment_len() < emergency;
            db.pool(ShardType::EventDag).sync_all().unwrap();
            if in_back_off_zone {
                backed_off_syncs += 1;
                if !db
                    .pool(ShardType::EventDag)
                    .sync_timings()
                    .unwrap()
                    .checkpoint
                    .is_zero()
                {
                    backed_off_checkpoints += 1;
                }
            }
            rounds += 1;
            assert!(rounds < 400, "the segment never filled");
        };
        let message = failure.to_string();
        assert!(message.contains("segment is full"), "{message}");
        assert!(
            message.contains("waiting on pools") && message.contains("State"),
            "the error must name the lagging pool: {message}"
        );
        assert!(coordinator.is_reclaim_stalled());
        assert_eq!(coordinator.reclaim_stalls(), 1, "one stall, reported once");
        assert_eq!(coordinator.reclaim_blockers(), vec![ShardType::State]);
        // The segment may only retry after growing by an eighth of the trigger,
        // so the zone allows at most (emergency - trigger) / (trigger / 8)
        // retries however many syncs happen in it.
        let bound = usize::try_from((emergency - trigger) / (trigger / 8)).unwrap() + 2;
        assert!(
            backed_off_checkpoints <= bound,
            "{backed_off_checkpoints} forced checkpoints in the back-off zone; at most {bound}"
        );
        assert!(
            backed_off_checkpoints * 2 < backed_off_syncs,
            "EventDag checkpointed {backed_off_checkpoints} times in {backed_off_syncs} syncs"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Pools sync one after another, so the first to cross the trigger cannot
    /// yet reclaim what the others have not reported. That is normal: it must
    /// neither be counted nor logged as a stall, and the later pool's coverage
    /// clears the pending state.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_healthy_sequential_sync_is_not_reported_as_a_stall() {
        let (db, root) = small_segment_database("healthy_sequential");
        let pools = [ShardType::State, ShardType::EventDag];
        let coordinator = db.coordinator();
        let mut next = 0u64;
        let mut saw_pending = false;
        for round in 0..150 {
            commit_batch(&db, &pools, &mut next, 1)
                .unwrap_or_else(|error| panic!("round {round}: {error}"));
            db.pool(ShardType::EventDag).sync_all().unwrap();
            saw_pending |= coordinator.is_reclaim_stalled();
            db.pool(ShardType::State).sync_all().unwrap();
            assert_eq!(
                coordinator.reclaim_stalls(),
                0,
                "round {round}: a healthy crossing was reported as a stall"
            );
        }
        assert!(
            saw_pending,
            "the first pool to cross must have seen a pending stall, or this proves nothing"
        );
        assert!(
            !coordinator.is_reclaim_stalled(),
            "the second pool's coverage must clear it"
        );
        assert!(coordinator.reclaim_trigger_len() > 0);
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A pool that stays silent is reported, but only after the segment has
    /// grown by an eighth of the trigger past the first failed reclaim.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_silent_contributor_is_reported_after_the_grace_period() {
        let (db, root) = small_segment_database("stall_grace");
        db.coordinator().set_blocker_remediation(|_| {});
        let pools = [ShardType::State, ShardType::EventDag];
        let coordinator = db.coordinator();
        let grace = coordinator.reclaim_trigger_len() / 8;
        let mut next = 0u64;
        let mut first_len = None;
        let mut reported_at = None;
        for _ in 0..200 {
            if commit_batch(&db, &pools, &mut next, 1).is_err() {
                break;
            }
            db.pool(ShardType::EventDag).sync_all().unwrap();
            if first_len.is_none() && coordinator.is_reclaim_stalled() {
                first_len = Some(coordinator.segment_len());
                assert_eq!(coordinator.reclaim_stalls(), 0, "reported with no grace");
            }
            if reported_at.is_none() && coordinator.reclaim_stalls() > 0 {
                reported_at = Some(coordinator.segment_len());
            }
        }
        let first_len = first_len.expect("the pending stall was never seen");
        let reported_at = reported_at.expect("a silent pool was never reported");
        assert!(
            reported_at >= first_len + grace,
            "reported at {reported_at}, only {} past the first failure",
            reported_at - first_len
        );
        assert_eq!(coordinator.reclaim_stalls(), 1, "one stall, reported once");
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Measurement, not policy: with a pool that never reports, count what the
    /// emergency zone costs. Every sync there may force a checkpoint; this
    /// records how many did, whether any reclaimed anything, and how much
    /// headroom the commits had left.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn emergency_zone_cost_with_a_pool_that_never_reports() {
        let (db, root) = small_segment_database("emergency_cost");
        db.coordinator().set_blocker_remediation(|_| {});
        let pools = [ShardType::State, ShardType::EventDag];
        let coordinator = db.coordinator();
        let cap = coordinator.segment_cap();
        let emergency = cap - cap / 4;
        let mut next = 0u64;
        let mut syncs = 0usize;
        let mut checkpoints = 0usize;
        let mut no_progress = 0usize;
        let mut slowest = std::time::Duration::ZERO;
        let mut min_headroom = u64::MAX;
        let mut rounds = 0usize;
        loop {
            if commit_batch(&db, &pools, &mut next, 1).is_err() {
                break;
            }
            let before = coordinator.segment_len();
            if before >= emergency {
                let cover_before = db.pool(ShardType::EventDag).durable_coverage();
                let started = std::time::Instant::now();
                db.pool(ShardType::EventDag).sync_all().unwrap();
                slowest = slowest.max(started.elapsed());
                syncs += 1;
                let timings = db.pool(ShardType::EventDag).sync_timings().unwrap();
                let forced = !timings.checkpoint.is_zero()
                    || db.pool(ShardType::EventDag).durable_coverage() > cover_before;
                if forced {
                    checkpoints += 1;
                    if coordinator.segment_len() >= before {
                        no_progress += 1;
                    }
                }
                min_headroom = min_headroom.min(cap.saturating_sub(coordinator.segment_len()));
            } else {
                db.pool(ShardType::EventDag).sync_all().unwrap();
            }
            rounds += 1;
            assert!(rounds < 400, "the segment never filled");
        }
        eprintln!(
            "emergency zone: cap={cap} emergency={emergency} syncs={syncs} \
             checkpoints={checkpoints} no_progress={no_progress} slowest={slowest:?} \
             min_headroom={min_headroom}"
        );
        // Nothing can reclaim while `State` is silent, so a forced step may
        // only repeat after the segment has grown by an eighth of the trigger.
        let growth = coordinator.reclaim_trigger_len() / 8;
        let bound = usize::try_from((cap - emergency) / growth).unwrap() + 2;
        assert!(
            syncs > bound,
            "the zone must be crossed by more syncs than the bound"
        );
        assert!(
            checkpoints <= bound,
            "{checkpoints} forced steps in {syncs} syncs; at most {bound}"
        );
        assert!(min_headroom > 0, "a commit had no room left");
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Copy `source` to `dest`, then cut every pool's packs in the copy back to
    /// what an fsync had covered when the copy was taken.
    #[cfg(feature = "multi-reader")]
    fn crash_image(db: &SharedDatabase, source: &std::path::Path, dest: &std::path::Path) {
        fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                let target = to.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_tree(&entry.path(), &target);
                } else {
                    std::fs::copy(entry.path(), &target).unwrap();
                }
            }
        }
        let _ = std::fs::remove_dir_all(dest);
        copy_tree(source, dest);
        for pool in ShardType::ALL {
            let live = db.layout().pool_dir(pool).unwrap();
            let relative = live.strip_prefix(source).unwrap();
            db.pool(pool).test_cut_packs_to_synced(&dest.join(relative));
        }
    }

    /// Discriminating check for `checkpoint_covered_lsn`: after any mix of
    /// writes and syncs, the disk a power cut would leave (packs cut to their
    /// fsynced length, WAL as it is) must still hold every record that was
    /// acknowledged and made durable by a full sync of its pool.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_power_cut_image_never_loses_a_record_a_claim_covered() {
        let root = test_root("cut_image_source");
        let image = test_root("cut_image_copy");
        let collection = [0x6e; 16];
        let pools = [ShardType::State, ShardType::EventDag];
        let db = SharedDatabase::open(root.clone()).unwrap();
        db.coordinator().set_reclaim_trigger_len(1);
        let mut durable: Vec<(ShardType, [u8; 16])> = Vec::new();
        let mut pending: Vec<(ShardType, [u8; 16])> = Vec::new();
        let mut next = 0u64;
        for round in 0..40u32 {
            let mut fresh = Vec::new();
            for pool in pools {
                let mut node = [0u8; 16];
                node[..8].copy_from_slice(&next.to_le_bytes());
                next += 1;
                db.pool(pool)
                    .put(&collection, &node, &NodeData::from_slice(&[0x11; 64]))
                    .unwrap();
                fresh.push((pool, node));
            }
            let txn = db.begin_transaction();
            for pool in pools {
                let mut node = [0u8; 16];
                node[..8].copy_from_slice(&next.to_le_bytes());
                next += 1;
                txn.put(pool, collection, node, &NodeData::from_slice(&[0x22; 64]))
                    .unwrap();
                fresh.push((pool, node));
            }
            txn.commit().unwrap();
            pending.extend(fresh);
            match round % 4 {
                0 => db.pool(ShardType::EventDag).sync().unwrap(),
                1 => db.pool(ShardType::State).sync().unwrap(),
                2 => {
                    db.pool(ShardType::State).sync_all().unwrap();
                    db.pool(ShardType::EventDag).sync_all().unwrap();
                    durable.append(&mut pending);
                }
                _ => {}
            }
            crash_image(&db, &root, &image);
            let recovered = SharedDatabase::open(image.clone()).unwrap();
            for (pool, node) in &durable {
                assert!(
                    recovered
                        .pool(*pool)
                        .get(&collection, node)
                        .unwrap()
                        .is_some(),
                    "round {round}: a durably synced record in {pool:?} is gone from the power-cut image"
                );
            }
            drop(recovered);
        }
        drop(db);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(image);
    }

    /// The same lag, but the database can make the silent pool checkpoint. Once
    /// the segment reaches the emergency zone it does, reclaim succeeds, and no
    /// commit ever fails.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_lagging_pool_is_checkpointed_for_the_caller_before_the_segment_fills() {
        use crate::PackfileStorage;
        let (db, root) = small_segment_database("lagging_remediated");
        let pools = [ShardType::State, ShardType::EventDag];
        let state_lsn =
            || PackfileStorage::read_journal_lsn(&db.layout().pool_dir(ShardType::State).unwrap());
        let mut next = 0u64;
        for round in 0..120 {
            commit_batch(&db, &pools, &mut next, 2)
                .unwrap_or_else(|error| panic!("round {round}: {error}"));
            db.pool(ShardType::EventDag).sync_all().unwrap();
            assert!(
                db.coordinator().segment_len() < db.coordinator().segment_cap(),
                "round {round}: the segment reached its cap"
            );
        }
        assert!(
            db.coordinator().reclaim_stalls() >= 1,
            "a stall must have been seen"
        );
        assert!(state_lsn() > 0, "State was never made to checkpoint");
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A burst of commits between syncs can outrun the headroom the trigger
    /// leaves: the segment fills, commits are refused with a clear error, and
    /// one sync reclaims enough that commits work again with nothing lost.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_commit_burst_that_fills_the_segment_is_refused_then_recovers_after_a_sync() {
        let (db, root) = small_segment_database("burst_fills_segment");
        let pools = [ShardType::EventDag];
        let mut next = 0u64;
        let mut committed = 0u64;
        let error = loop {
            match commit_batch(&db, &pools, &mut next, 1) {
                Ok(()) => committed = committed.saturating_add(1),
                Err(error) => break error,
            }
            assert!(committed < 1000, "the segment never filled");
        };
        let message = error.to_string();
        assert!(message.contains("segment is full"), "{message}");
        // Contract, taken at the refusal (a later reclaim would clear a stall
        // anyway): the only contributing pool is the one that synced, so nothing
        // was waiting on another pool. The refusal is the whole failure path;
        // there is no throttle, and no stall was recorded.
        assert!(!db.coordinator().is_reclaim_stalled());
        assert_eq!(db.coordinator().reclaim_stalls(), 0);
        assert!(db.coordinator().segment_len() > db.coordinator().reclaim_trigger_len());
        // No sync ran during the burst, so it went straight past the trigger.
        db.pool(ShardType::EventDag).sync_all().unwrap();
        assert!(db.coordinator().segment_len() < db.coordinator().reclaim_trigger_len());
        commit_batch(&db, &pools, &mut next, 1).unwrap();
        // Every commit that succeeded before the refusal is still readable.
        let collection = [0x7c; 16];
        let mut first = [0u8; 16];
        first[..8].copy_from_slice(&0u64.to_le_bytes());
        assert!(db
            .pool(ShardType::EventDag)
            .get_read_committed(&collection, &[first])
            .unwrap()[0]
            .is_some());
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A transaction group is published to the WAL before its writes are
    /// materialized into the packs. A checkpoint in that window must not record
    /// coverage of frames the packs do not hold yet, or the WAL could be
    /// reclaimed past them and a crash would lose an acknowledged commit.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn a_checkpoint_does_not_claim_an_unmaterialized_transaction() {
        use crate::PackfileStorage;
        let root = test_root("checkpoint_unmaterialized");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let transaction = db.begin_transaction();
        transaction
            .put(ShardType::State, [0x91; 16], node(1), &data(b"state"))
            .unwrap();
        let mut overlay = Some(activate_for(&db, &transaction.stage));
        db.publish_transaction(&transaction.stage).unwrap();
        transaction.stage.begin_materialization().unwrap();
        let receipt = transaction.stage.published_receipt().unwrap();
        db.register_new_recovery_stage(Arc::clone(&transaction.stage), receipt, &mut overlay)
            .unwrap();
        // The group is published but nothing has been written to the State pack.
        for pool in ShardType::ALL {
            db.pool(pool).sync_all().unwrap();
        }
        let covered =
            PackfileStorage::read_journal_lsn(&db.layout().pool_dir(ShardType::State).unwrap());
        assert!(
            covered < receipt.first_lsn,
            "State claims coverage through {covered} but its frame at LSN {} is not in any pack",
            receipt.first_lsn
        );
        // The consequence a claim would have: the WAL group, the only copy of an
        // acknowledged commit, being reclaimed.
        let kept = crate::journal::Journal::scan_read_only(db.layout().shared_wal_path())
            .unwrap()
            .groups
            .len();
        assert_eq!(
            kept, 1,
            "the WAL must keep the group until its writes are in the packs"
        );
        drop((transaction, overlay));
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Two pools carry frames and each advances its coverage by a delta batch,
    /// never a full checkpoint. The shared segment is reclaimed only once both
    /// have reported: after the first pool's step the second still holds it,
    /// and after the second's the reclaimed segment holds no covered group.
    #[cfg(feature = "multi-reader")]
    #[test]
    fn shared_reclaim_waits_for_every_pools_delta_coverage_batch() {
        let (db, root) = small_segment_database("shared_delta_coverage");
        let pools = [ShardType::State, ShardType::EventDag];
        let coordinator = db.coordinator();
        let mut next = 0u64;
        // A first sync of each pool takes its initial full checkpoint.
        commit_batch(&db, &pools, &mut next, 1).unwrap();
        for pool in pools {
            db.pool(pool).sync_all().unwrap();
        }
        // Grow the segment past the trigger with both pools' frames.
        while coordinator.segment_len() <= coordinator.reclaim_trigger_len() {
            commit_batch(&db, &pools, &mut next, 1).unwrap();
        }
        let before = coordinator.segment_len();
        db.pool(ShardType::EventDag).sync_all().unwrap();
        let timings = db.pool(ShardType::EventDag).sync_timings().unwrap();
        assert!(
            timings.checkpoint.is_zero(),
            "EventDag must advance by a delta batch"
        );
        assert!(!timings.delta_log.is_zero());
        assert_eq!(
            coordinator.segment_len(),
            before,
            "State has not reported, so nothing may be reclaimed"
        );
        assert_eq!(coordinator.reclaim_blockers(), vec![ShardType::State]);

        db.pool(ShardType::State).sync_all().unwrap();
        let timings = db.pool(ShardType::State).sync_timings().unwrap();
        assert!(
            timings.checkpoint.is_zero(),
            "State must advance by a delta batch"
        );
        assert!(
            coordinator.segment_len() < before,
            "both pools have reported, so the segment shrinks"
        );
        assert!(coordinator.reclaim_blockers().is_empty());
        drop(db);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn transaction_commit_applies_and_publishes_once() {
        let root = test_root("transaction_commit");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let state_collection = [0x61; 16];
        let event_collection = [0x62; 16];
        let transaction = db.begin_transaction();
        transaction
            .put(
                ShardType::State,
                state_collection,
                node(1),
                &NodeData::new(bytes::Bytes::from_static(b"state")),
            )
            .unwrap();
        transaction
            .put(
                ShardType::EventDag,
                event_collection,
                node(2),
                &NodeData::new(bytes::Bytes::from_static(b"event")),
            )
            .unwrap();
        assert!(db
            .pool(ShardType::State)
            .get(&state_collection, &node(1))
            .unwrap()
            .is_none());
        transaction.commit().unwrap();
        transaction.commit().unwrap();
        assert!(db
            .pool(ShardType::State)
            .get(&state_collection, &node(1))
            .unwrap()
            .is_some());
        assert!(db
            .pool(ShardType::EventDag)
            .get(&event_collection, &node(2))
            .unwrap()
            .is_some());
        assert_eq!(
            transaction.state(),
            crate::journal::TxnStageState::Published
        );
        let scan = crate::journal::Journal::scan_read_only(root.join("wal.bin")).unwrap();
        assert_eq!(
            scan.groups.len(),
            1,
            "one transaction must emit one journal group"
        );
        assert_eq!(scan.groups[0].entries.len(), 2);
        assert_eq!(
            scan.groups[0]
                .entries
                .iter()
                .map(|entry| entry.pool)
                .collect::<Vec<_>>(),
            vec![Some(ShardType::EventDag), Some(ShardType::State)]
        );
        drop(transaction);
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn retrying_commit_and_recovery_can_run_concurrently() {
        let root = test_root("transaction_concurrent_recovery");
        let database = Arc::new(SharedDatabase::open(root.clone()).unwrap());
        let transaction = Arc::new(database.begin_transaction());
        let collection = [0x66; 16];
        let node_id = node(1);
        transaction
            .put(
                ShardType::State,
                collection,
                node_id,
                &NodeData::new(bytes::Bytes::from_static(b"concurrent")),
            )
            .unwrap();

        let mut overlay = Some(activate_for(&database, &transaction.stage));
        database.publish_transaction(&transaction.stage).unwrap();
        transaction.stage.begin_materialization().unwrap();
        let receipt = transaction.stage.published_receipt().unwrap();
        database
            .register_new_recovery_stage(Arc::clone(&transaction.stage), receipt, &mut overlay)
            .unwrap();

        let recovery_start = Arc::new(Barrier::new(2));
        let recovery_guard = database.recovery_lifecycle.lock();
        std::thread::scope(|scope| {
            let recovery_database = Arc::clone(&database);
            let recovery_start_thread = Arc::clone(&recovery_start);
            let recovery = scope.spawn(move || {
                recovery_start_thread.wait();
                recovery_database.recover_pending_transactions()
            });

            recovery_start.wait();
            let commit_transaction = Arc::clone(&transaction);
            let commit_started = Arc::new(Barrier::new(2));
            let commit_started_thread = Arc::clone(&commit_started);
            let commit = scope.spawn(move || {
                commit_started_thread.wait();
                commit_transaction.commit()
            });
            commit_started.wait();

            // Both calls have started while the recovery lifecycle is held;
            // releasing it makes them contend on the same published stage.
            drop(recovery_guard);
            recovery.join().unwrap().unwrap();
            commit.join().unwrap().unwrap();
        });

        assert_eq!(
            transaction.state(),
            crate::journal::TxnStageState::Published
        );
        assert!(database
            .pool(ShardType::State)
            .get(&collection, &node_id)
            .unwrap()
            .is_some());
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn published_transaction_is_visible_before_materialization() {
        let root = test_root("transaction_overlay");
        let state_collection = [0x61; 16];
        let event_collection = [0x62; 16];
        let state_node = node(1);
        let event_node = node(2);
        let database = SharedDatabase::open(root.clone()).unwrap();
        let old_node = node(9);
        database
            .pool(ShardType::State)
            .put(
                &state_collection,
                &old_node,
                &NodeData::new(bytes::Bytes::from_static(b"old")),
            )
            .unwrap();
        let transaction = database.begin_transaction();
        transaction
            .put(
                ShardType::State,
                state_collection,
                state_node,
                &NodeData::new(bytes::Bytes::from_static(b"state")),
            )
            .unwrap();
        transaction
            .put(
                ShardType::EventDag,
                event_collection,
                event_node,
                &NodeData::new(bytes::Bytes::from_static(b"event")),
            )
            .unwrap();

        // Keep the database alive separately: the transaction borrows it and
        // its overlay must remain active while the published group is not yet
        // materialized into either live index.
        let mut overlay = Some(activate_for(&database, &transaction.stage));
        database.publish_transaction(&transaction.stage).unwrap();
        let receipt = transaction.stage.published_receipt().unwrap();

        assert_eq!(
            database
                .pool(ShardType::State)
                .get(&state_collection, &state_node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"state"
        );
        assert_eq!(
            database
                .pool(ShardType::State)
                .get(&state_collection, &old_node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"old",
            "overlay reads must fall back to pre-existing live records"
        );
        assert_eq!(
            database
                .pool(ShardType::EventDag)
                .get(&event_collection, &event_node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"event"
        );
        let during_materialization = transaction
            .get(ShardType::EventDag, &event_collection, &[event_node])
            .unwrap();
        assert_eq!(
            during_materialization[0]
                .as_ref()
                .map(|data| data.bytes.as_ref()),
            Some(b"event".as_ref()),
            "a retrying transaction must keep read-your-writes through its overlay"
        );

        assert!(transaction.abort().is_err());
        transaction.stage.begin_materialization().unwrap();
        database
            .register_new_recovery_stage(Arc::clone(&transaction.stage), receipt, &mut overlay)
            .unwrap();
        drop(transaction);
        database.recover_pending_transactions().unwrap();
        drop(database);
        let reopened = SharedDatabase::open(root.clone()).unwrap();
        assert!(reopened
            .pool(ShardType::State)
            .get(&state_collection, &state_node)
            .unwrap()
            .is_some());
        assert!(reopened
            .pool(ShardType::EventDag)
            .get(&event_collection, &event_node)
            .unwrap()
            .is_some());
        drop(reopened);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn partial_materialization_retries_through_overlay_and_drains_it() {
        let root = test_root("transaction_materialization_retry");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let state_collection = [0x71; 16];
        let event_collection = [0x72; 16];
        let state_node = node(1);
        let event_node = node(2);
        let transaction = database.begin_transaction();
        let mut overlay = None;
        prepare_partial_materialization(
            &database,
            &transaction,
            &mut overlay,
            state_collection,
            event_collection,
        );
        let state_mutation =
            transaction.stage.snapshot_mutations()[shard_index(ShardType::State)][0].clone();
        database
            .pool(ShardType::State)
            .apply_transaction_mutation(&state_mutation)
            .unwrap();
        transaction
            .stage
            .mark_mutation_applied(ShardType::State, 0)
            .unwrap();
        assert_eq!(
            transaction.state(),
            crate::journal::TxnStageState::Materializing
        );
        assert!(database
            .pool(ShardType::State)
            .get(&state_collection, &state_node)
            .unwrap()
            .is_some());
        assert!(database
            .pool(ShardType::EventDag)
            .get(&event_collection, &event_node)
            .unwrap()
            .is_some());

        transaction.commit().unwrap();
        assert_eq!(
            transaction.state(),
            crate::journal::TxnStageState::Published
        );
        assert_eq!(
            database
                .pool(ShardType::EventDag)
                .get(&event_collection, &event_node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"event",
            "ordinary reads must work after the overlay is released"
        );
        assert_eq!(
            database
                .pool(ShardType::EventDag)
                .collection_len(&event_collection)
                .unwrap(),
            Some(1),
            "collection length must remain correct after the overlay is released"
        );
        drop(transaction);
        drop(database);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dropped_partial_transaction_recovers_from_wal_and_drains_overlay() {
        let root = test_root("transaction_orphan_recovery");
        let database = SharedDatabase::open(root.clone()).unwrap();
        let state_collection = [0x81; 16];
        let event_collection = [0x82; 16];
        let state_node = node(1);
        let event_node = node(2);
        let transaction = database.begin_transaction();
        let mut overlay = None;
        prepare_partial_materialization(
            &database,
            &transaction,
            &mut overlay,
            state_collection,
            event_collection,
        );
        let state_mutation =
            transaction.stage.snapshot_mutations()[shard_index(ShardType::State)][0].clone();
        database
            .pool(ShardType::State)
            .apply_transaction_mutation(&state_mutation)
            .unwrap();
        transaction
            .stage
            .mark_mutation_applied(ShardType::State, 0)
            .unwrap();
        drop(transaction);

        // Drop only leaves the published stage in the database-owned queue;
        // the explicit recovery boundary performs the materialization.
        database.recover_pending_transactions().unwrap();
        assert!(database
            .pool(ShardType::State)
            .get(&state_collection, &state_node)
            .unwrap()
            .is_some());
        assert!(database
            .pool(ShardType::EventDag)
            .get(&event_collection, &event_node)
            .unwrap()
            .is_some());
        assert_eq!(
            database
                .pool(ShardType::EventDag)
                .collection_len(&event_collection)
                .unwrap(),
            Some(1)
        );
        drop(database);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_written_pool_replays_on_reopen() {
        let root = test_root("replay");
        let collection = [0x11u8; 16];
        {
            let db = SharedDatabase::open(root.clone()).unwrap();
            db.pool(ShardType::State)
                .put(
                    &collection,
                    &node(1),
                    &NodeData::new(bytes::Bytes::from_static(b"x")),
                )
                .unwrap();
            db.pool(ShardType::State).sync_all().unwrap();
        }
        let db = SharedDatabase::open(root.clone()).unwrap();
        let got = db
            .pool(ShardType::State)
            .get(&collection, &node(1))
            .unwrap()
            .expect("replayed node must be readable after reopen");
        assert_eq!(got.bytes.as_ref(), b"x");
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn checkpoint_coverage_does_not_regress_to_zero_after_reclaim() {
        let root = test_root("coverage_no_regress");
        let state_collection = [0x33u8; 16];
        let event_collection = [0x44u8; 16];
        {
            let db = SharedDatabase::open(root.clone()).unwrap();
            db.pool(ShardType::State)
                .put(
                    &state_collection,
                    &node(1),
                    &NodeData::from_slice(b"state-1"),
                )
                .unwrap();
            db.pool(ShardType::State).sync_all().unwrap();
            db.pool(ShardType::State).force_index_checkpoint().unwrap();
            assert!(db.coordinator().committed_lsn_for_pool(ShardType::State) > 0);

            // Event-DAG checkpoints a later group; prefix reclaim drops every
            // retained group, including state's, leaving state with no frames.
            db.pool(ShardType::EventDag)
                .put(
                    &event_collection,
                    &node(2),
                    &NodeData::from_slice(b"event-1"),
                )
                .unwrap();
            db.pool(ShardType::EventDag).sync_all().unwrap();
            db.pool(ShardType::EventDag)
                .force_index_checkpoint()
                .unwrap();
            assert_eq!(
                Journal::scan_read_only(root.join("wal.bin"))
                    .unwrap()
                    .groups
                    .len(),
                0,
                "every retained group must be reclaimed before the restart"
            );
        }

        let before = PackfileStorage::read_journal_lsn(&root.join("pools/mtpl-state"));
        assert!(
            before > 0,
            "state's durable checkpoint coverage must survive the reclaim"
        );

        // Reopen: the fresh coordinator has no state frame to derive a
        // watermark from. A checkpoint must preserve the recorded coverage
        // rather than erase it, or the next replay would start from zero.
        let db = SharedDatabase::open(root.clone()).unwrap();
        db.pool(ShardType::State).force_index_checkpoint().unwrap();
        assert_eq!(
            PackfileStorage::read_journal_lsn(&root.join("pools/mtpl-state")),
            before,
            "a checkpoint must never regress its covered LSN to zero"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// If the root's WAL is missing while a pool has recorded checkpoint
    /// coverage, the fresh segment must start above that coverage, so a pool's
    /// recorded LSN stays meaningful in the one shared LSN space and numbering
    /// can never restart beneath it.
    #[test]
    fn a_missing_wal_is_seeded_above_the_pools_recorded_coverage() {
        let root = test_root("seed_missing_wal");
        drop(crate::layout::DatabaseLayout::open(root.clone()).unwrap());
        std::fs::write(
            root.join("pools/mtpl-state/journal.lsn"),
            7u64.to_le_bytes(),
        )
        .unwrap();
        assert!(!root.join("wal.bin").exists());

        let db = SharedDatabase::open(root.clone()).unwrap();
        let scan = Journal::scan_read_only(root.join("wal.bin")).unwrap();
        assert!(
            scan.base_lsn > 7,
            "the fresh shared WAL must start above the recorded coverage"
        );
        db.pool(ShardType::State)
            .put(&[0x55u8; 16], &node(3), &NodeData::from_slice(b"live"))
            .unwrap();
        db.pool(ShardType::State).sync_all().unwrap();
        assert!(
            db.coordinator().committed_lsn_for_pool(ShardType::State) > 7,
            "a new shared frame must be numbered above the watermark"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_second_writer_on_one_root_is_refused() {
        let root = test_root("one_writer");
        let first = SharedDatabase::open(root.clone()).unwrap();
        let second = SharedDatabase::open(root.clone());
        assert!(
            second.is_err(),
            "a second writer must not open the same root"
        );
        drop(first);
        // Releasing the first handle frees the root for a new writer.
        SharedDatabase::open(root.clone()).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_per_pool_journal_is_refused_inside_a_shared_root() {
        let root = test_root("legacy_gate");
        let db = SharedDatabase::open(root.clone()).unwrap();
        let err = db
            .pool(ShardType::State)
            .enable_journal(root.join("pools/mtpl-state/wal.bin"))
            .unwrap_err();
        assert!(
            err.to_string().contains("inside database root"),
            "unexpected error: {err}"
        );
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_uses_default_pool_policies_enabling_compression() {
        let root = test_root("default_policies");
        let db = SharedDatabase::open(root.clone()).unwrap();
        for shard in ShardType::ALL {
            assert!(
                db.pool(shard).is_compression_enabled(),
                "default policy must enable compression for {shard:?}"
            );
            assert_eq!(
                db.pool(shard).checksum_policy(),
                crate::packfile::ChecksumPolicy::Full
            );
        }
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_with_policies_applies_per_pool_settings() {
        let root = test_root("explicit_policies");
        let mut policies = super::PoolPolicies::default();
        policies.state.compress = false;
        policies.edges.checksum_policy = crate::packfile::ChecksumPolicy::WriteOnly;

        let db = SharedDatabase::open_with_policies(root.clone(), policies).unwrap();
        assert!(!db.state().is_compression_enabled());
        assert!(db.event_dag().is_compression_enabled());
        assert!(db.edges().is_compression_enabled());
        assert_eq!(
            db.edges().checksum_policy(),
            crate::packfile::ChecksumPolicy::WriteOnly
        );
        drop(db);
        let _ = std::fs::remove_dir_all(&root);
    }
}
