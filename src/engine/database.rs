//! Root-level handle for a shared-WAL database.
//!
//! [`Database`] is the safe end-to-end entry point for the shared
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

const POOL_COUNT: usize = ShardType::ALL.len();

/// Policies for every named pool in a database.
///
/// One slot per [`ShardType`], indexed by [`ShardType::index`], so adding a pool
/// adds a slot here without any other change. Start from the defaults and set
/// the pools that differ with [`Self::with`] or [`Self::for_shard_mut`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolPolicies {
    policies: [PoolPolicy; POOL_COUNT],
}

impl PoolPolicies {
    /// Every pool set to `policy`.
    #[must_use]
    pub const fn uniform(policy: PoolPolicy) -> Self {
        Self {
            policies: [policy; POOL_COUNT],
        }
    }

    /// Return the policy associated with `shard`.
    #[must_use]
    pub fn for_shard(&self, shard: ShardType) -> &PoolPolicy {
        &self.policies[shard.index()]
    }

    /// Mutable access to the policy associated with `shard`.
    pub fn for_shard_mut(&mut self, shard: ShardType) -> &mut PoolPolicy {
        &mut self.policies[shard.index()]
    }

    /// These policies with `shard` set to `policy`.
    #[must_use]
    pub fn with(mut self, shard: ShardType, policy: PoolPolicy) -> Self {
        *self.for_shard_mut(shard) = policy;
        self
    }
}

/// A database root open for writing: every pool behind one write-ahead log,
/// with atomic cross-pool transactions and record-version compare-and-set.
///
/// Available in every build; no feature flag is needed to open or write a
/// database. Attaching *another process* to a live database as a read-committed
/// reader (`PackfileStorage::open_read_committed_shared`) needs the
/// `multi-reader` feature. Dropping the handle releases the root writer lock and
/// the pools it opened.
pub struct Database {
    layout: DatabaseLayout,
    coordinator: Arc<JournalCoordinator>,
    pools: [Arc<PackfileStorage>; POOL_COUNT],
    txn_stage_limit_bytes: usize,
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

/// The pre-`Database` name, kept for existing callers. The shared WAL it was
/// named for is now simply how every `Database` works.
#[deprecated(note = "use Database instead")]
pub type SharedDatabase = Database;

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
    Borrowed(&'a Database),
    Owned(Arc<Database>),
}

impl std::ops::Deref for DatabaseRef<'_> {
    type Target = Database;

    fn deref(&self) -> &Database {
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

    /// Stage a collection-version precondition. [`Self::commit`] rejects the
    /// transaction with [`StorageError::StaleRead`] if the collection's logical
    /// version at publication differs from `expected`.
    ///
    /// The token is only a valid compare-and-swap safeguard when it is read
    /// together with the data it guards: a version read after the data can be
    /// newer than the data and let a stale update pass.
    ///
    /// # Errors
    /// Returns an error if the transaction has already been committed or
    /// aborted.
    pub fn expect_collection_version(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        expected: u64,
    ) -> io::Result<()> {
        let _lifecycle = self.lifecycle.lock();
        self.stage.stage_expectation(pool, collection_id, expected)
    }

    /// Stage a per-record version precondition. [`Self::commit`] rejects the
    /// transaction with [`StorageError::StaleRead`] if the record's write LSN
    /// at publication differs from `expected`. `expected` is the token returned
    /// by [`Self::get_with_record_versions`] for the same record (the
    /// collection's delete LSN when the record was absent, or `0` for a legacy
    /// frame).
    ///
    /// # Errors
    /// Returns an error if the transaction has already been committed or
    /// aborted.
    pub fn expect_record_version(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        expected: u64,
    ) -> io::Result<()> {
        let _lifecycle = self.lifecycle.lock();
        self.stage
            .stage_record_expectation(pool, collection_id, node_id, expected)
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
        // Keep this transaction's stage fixed until the live records and
        // version token have been sampled; otherwise a concurrent commit on
        // this handle could make staged values pair with a post-commit token.
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

    /// Read records through the transaction's own staged writes first, then the
    /// live store, returning the collection's logical version alongside them.
    ///
    /// The version is the live collection clock; this transaction's staged
    /// writes are uncommitted and do not advance it. Pass it to
    /// [`Self::expect_collection_version`] to make the commit conditional on
    /// the collection not having changed since this read.
    ///
    /// # Errors
    /// Returns an error if the live pool cannot be read.
    pub fn get_with_collection_version(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[NodeId],
    ) -> Result<(Vec<Option<crate::storage::NodeData>>, u64), StorageError> {
        // Keep this handle's staged view fixed until the live versioned read
        // completes; otherwise this transaction could commit between stage
        // lookup and the live token sample.
        let _lifecycle = self.lifecycle.lock();
        let staged = self.stage.lookup_many(pool, collection_id, node_ids);
        let mut results = Vec::with_capacity(node_ids.len());
        let mut live_slots = Vec::new();
        let mut live_ids = Vec::new();
        for (slot, (node_id, lookup)) in node_ids.iter().zip(staged).enumerate() {
            match lookup {
                StagedLookup::Put(payload) => {
                    results.push(Some(crate::storage::NodeData::new(bytes::Bytes::from(
                        payload,
                    ))));
                }
                StagedLookup::Deleted => results.push(None),
                StagedLookup::Absent => {
                    results.push(None);
                    live_slots.push(slot);
                    live_ids.push(*node_id);
                }
            }
        }
        let (live, version) = self
            .database
            .pool(pool)
            .get_with_collection_version(collection_id, &live_ids)?;
        for (slot, value) in live_slots.into_iter().zip(live) {
            results[slot] = value;
        }
        Ok((results, version))
    }

    /// Read records through the transaction's own staged writes first, then
    /// the live store, returning each record's write LSN alongside them.
    ///
    /// Pass a token to [`Self::expect_record_version`] to make the commit
    /// conditional on that exact record not having changed since this read.
    /// This transaction's own staged writes are uncommitted and have no
    /// published version, so their token is `0`; read before staging writes
    /// when using the tokens as preconditions.
    ///
    /// # Errors
    /// Returns an error if the live pool cannot be read.
    pub fn get_with_record_versions(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[NodeId],
    ) -> Result<(Vec<Option<crate::storage::NodeData>>, Vec<u64>), StorageError> {
        let _lifecycle = self.lifecycle.lock();
        let staged = self.stage.lookup_many(pool, collection_id, node_ids);
        let mut results = Vec::with_capacity(node_ids.len());
        let mut versions = vec![0u64; node_ids.len()];
        let mut live_slots = Vec::new();
        let mut live_ids = Vec::new();
        for (slot, (node_id, lookup)) in node_ids.iter().zip(staged).enumerate() {
            match lookup {
                StagedLookup::Put(payload) => {
                    results.push(Some(crate::storage::NodeData::new(bytes::Bytes::from(
                        payload,
                    ))));
                }
                StagedLookup::Deleted => results.push(None),
                StagedLookup::Absent => {
                    results.push(None);
                    live_slots.push(slot);
                    live_ids.push(*node_id);
                }
            }
        }
        if !live_ids.is_empty() {
            let (live, live_versions) = self
                .database
                .pool(pool)
                .get_with_record_versions(collection_id, &live_ids)?;
            for ((slot, value), version) in live_slots.into_iter().zip(live).zip(live_versions) {
                results[slot] = value;
                versions[slot] = version;
            }
        }
        Ok((results, versions))
    }

    /// Read node payloads and the collection version without copying the
    /// payload bytes. The returned [`bytes::Bytes`] handles share the storage
    /// buffers owned by the underlying read; only the small result containers
    /// are newly allocated.
    ///
    /// # Errors
    ///
    /// Propagates storage errors from the versioned read.
    pub fn get_with_collection_version_bytes(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[NodeId],
    ) -> Result<(Vec<Option<bytes::Bytes>>, u64), StorageError> {
        self.get_with_collection_version(pool, collection_id, node_ids)
            .map(|(records, version)| {
                (
                    records
                        .into_iter()
                        .map(|record| record.map(|data| data.bytes))
                        .collect(),
                    version,
                )
            })
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
                let expectations = self.stage.expectations();
                let record_expectations = self.stage.record_expectations();
                if !expectations.is_empty() || !record_expectations.is_empty() {
                    self.database
                        .coordinator
                        .check_expectations(&expectations, &record_expectations)
                        .map_err(stale_read_from_publish)?;
                }
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
                return Err(stale_read_from_publish(error));
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

impl Database {
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
        Self::open_with_policies_and_stage_limit(
            root,
            policies,
            crate::journal::MAX_TXN_STAGE_BYTES,
        )
    }

    #[cfg(all(test, feature = "bitmaps"))]
    pub(crate) fn open_with_txn_stage_limit(
        root: PathBuf,
        limit_bytes: usize,
    ) -> Result<Self, StorageError> {
        Self::open_with_policies_and_stage_limit(root, PoolPolicies::default(), limit_bytes)
    }

    fn open_with_policies_and_stage_limit(
        root: PathBuf,
        policies: PoolPolicies,
        txn_stage_limit_bytes: usize,
    ) -> Result<Self, StorageError> {
        let layout = DatabaseLayout::open(root)?;
        let wal_path = layout.shared_wal_path();
        let lock = SharedWalLock::acquire(layout.root())?;
        let seed_lsn = shared_wal_seed_lsn(&layout);
        let (journal, scan) = Journal::open_shared_with_base(&wal_path, seed_lsn)?;
        let coordinator = Arc::new(JournalCoordinator::new(journal, &scan));

        let mut pools = Vec::with_capacity(ShardType::ALL.len());
        for shard in ShardType::ALL {
            let dir = layout.pool_path(shard);
            let policy = policies.for_shard(shard);
            let store =
                PackfileStorage::open_shared_member(dir, policy.compress, policy.checksum_policy)?;
            // This process holds the exclusive WAL lock, so its in-memory
            // index is authoritative and a negative lookup is a true miss.
            // Left on, every miss after a sync (which moves the durable
            // fingerprint) rescans the whole collection and bumps its
            // generation, which in turn makes the next sync write a
            // whole-index snapshot: per-sync cost proportional to the store.
            store.set_refresh_on_miss(false);
            store.enable_shared_journal(Arc::clone(&coordinator), shard)?;
            store.replay_journal()?;
            pools.push(Arc::new(store));
        }
        let pools: [Arc<PackfileStorage>; POOL_COUNT] = pools.try_into().map_err(|_| {
            StorageError::Internal(format!(
                "a shared database must open exactly {POOL_COUNT} pools"
            ))
        })?;

        // This process owns every pool, so it can make a lagging one checkpoint
        // when its silence has stalled WAL reclaim into the emergency zone. The
        // coordinator holds only weak references, so it does not keep the pools
        // alive.
        let weak_pools: Vec<std::sync::Weak<PackfileStorage>> =
            pools.iter().map(Arc::downgrade).collect();
        coordinator.set_blocker_remediation(move |pool| {
            let storage = weak_pools
                .get(pool.index())
                .and_then(std::sync::Weak::upgrade);
            if let Some(Err(error)) =
                storage.map(|storage| storage.force_index_checkpoint_detached())
            {
                eprintln!("warning: checkpointing {pool:?} to unblock WAL reclaim failed: {error}");
            }
        });

        Ok(Self {
            layout,
            coordinator,
            pools,
            txn_stage_limit_bytes,
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
        &self.pools[shard.index()]
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

    /// The server-info store (`ShardType::ServerInfo`).
    #[must_use]
    pub fn server_info(&self) -> &Arc<PackfileStorage> {
        self.pool(ShardType::ServerInfo)
    }

    /// Begin a storage transaction whose writes are invisible until commit.
    #[must_use]
    pub fn begin_transaction(&self) -> DatabaseTransaction<'_> {
        DatabaseTransaction {
            database: DatabaseRef::Borrowed(self),
            stage: Arc::new(TxnStage::with_limit(self.txn_stage_limit_bytes)),
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
            stage: Arc::new(TxnStage::with_limit(self.txn_stage_limit_bytes)),
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
            let touched = item.stage.touched_pools();
            for (index, pool) in ShardType::ALL.into_iter().enumerate() {
                if touched[index] {
                    self.pool(pool).note_materialized_lsn(item.receipt.last_lsn);
                }
            }
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
        // The published group's last LSN is stamped into each record it writes,
        // so a record's optimistic token survives reclaim of the group's WAL.
        let write_lsn = stage
            .published_receipt()
            .map_or(0, |receipt| receipt.last_lsn);
        let batches = stage.snapshot_mutations();
        for pool in ShardType::ALL {
            for (index, mutation) in batches[pool.index()].iter().enumerate() {
                if stage.mutation_applied(pool, index) {
                    continue;
                }
                self.pool(pool)
                    .apply_transaction_mutation(mutation, write_lsn)?;
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
    /// All mutations staged for the four pools are appended as one tagged
    /// journal group. A later durability operation may fsync that group, but
    /// readers see either the complete group or none of it.
    ///
    /// # Errors
    /// Returns an error if staging is inactive, the coordinator is poisoned,
    /// or the journal group cannot be appended.
    pub fn publish_transaction(&self, stage: &TxnStage) -> io::Result<()> {
        stage.publish([Some(self.coordinator.as_ref()); ShardType::ALL.len()])
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
/// [`Database::open`] instead.
///
pub(crate) fn shared_wal_seed_lsn(layout: &DatabaseLayout) -> u64 {
    let mut watermark = 0_u64;
    for shard in ShardType::ALL {
        let dir = layout.pool_path(shard);
        watermark = watermark.max(PackfileStorage::read_journal_lsn(&dir));
    }
    watermark.saturating_add(1)
}

/// Convert a publish-path error into a typed storage error, surfacing a
/// collection-version conflict as [`StorageError::StaleRead`].
fn stale_read_from_publish(error: std::io::Error) -> StorageError {
    if let Some(stale) = error
        .get_ref()
        .and_then(|source| source.downcast_ref::<crate::journal::StaleVersion>())
    {
        return StorageError::StaleRead {
            pool: stale.pool,
            collection_id: stale.collection_id,
            expected: stale.expected,
            actual: stale.actual,
        };
    }
    StorageError::Io(error)
}

#[cfg(test)]
#[path = "test_database.rs"]
pub(crate) mod tests;
