//! Root-level handle for a shared-WAL database.
//!
//! [`SharedDatabase`] is the safe end-to-end entry point for the shared
//! durability fence: it opens a database root, acquires the single root writer
//! lock, opens the root's pool-tagged `wal.bin` as one
//! [`crate::journal::JournalCoordinator`], opens every named pool, attaches each to that one
//! coordinator, and replays recovered groups. It holds the root lock for its
//! whole lifetime, so exactly one writer process owns the database.
//!
//! A legacy per-pool root (one whose `db.meta` predates the WAL-layout field,
//! or was written as [`WalLayout::PerPool`]) is *not* rejected: it is opened
//! onto a fresh root-level shared segment, and its old `pools/*/wal.bin`
//! files are assumed already checkpointed and are ignored. The fresh segment's
//! LSN space is seeded above every per-pool `journal.lsn` so the legacy
//! coverage values stay meaningful; see `shared_wal_seed_lsn`. Callers that
//! need to drive a single pool directly can still use
//! [`crate::journal::SharedWalLock`] with
//! [`PackfileStorage::enable_shared_journal`](crate::PackfileStorage::enable_shared_journal).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

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
    /// Held for the lifetime of the handle: one writer per database root.
    _lock: SharedWalLock,
}

struct RecoveryItem {
    receipt: CommitReceipt,
    stage: Arc<TxnStage>,
    overlay: TransactionOverlayGuard,
}

struct TransactionOverlayGuard {
    pools: [Arc<PackfileStorage>; 3],
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
        let _lifecycle = self.lifecycle.lock();
        let staged = self.stage.lookup_many(pool, collection_id, node_ids);
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
            let mut overlay = Some(self.database.activate_transaction_overlay()?);
            if let Err(error) = self.database.publish_transaction(&self.stage) {
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
            let _recovery_lifecycle = self.database.recovery_lifecycle.lock();
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
            let _recovery_lifecycle = self.database.recovery_lifecycle.lock();
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
            self.database.materialize_transaction(&self.stage)?;
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

        Ok(Self {
            layout,
            coordinator,
            pools,
            recovery_queue: parking_lot::Mutex::new(Vec::new()),
            recovery_lifecycle: parking_lot::Mutex::new(()),
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

    fn activate_transaction_overlay(&self) -> Result<TransactionOverlayGuard, StorageError> {
        let wal_path = self.layout.shared_wal_path();
        let mut activated = 0usize;
        for pool in ShardType::ALL {
            if let Err(error) = self
                .pool(pool)
                .activate_transaction_overlay(&wal_path, pool)
            {
                for previous in ShardType::ALL.into_iter().take(activated) {
                    self.pool(previous).deactivate_transaction_overlay();
                }
                return Err(error);
            }
            activated = activated.saturating_add(1);
        }
        Ok(TransactionOverlayGuard {
            pools: std::array::from_fn(|index| Arc::clone(&self.pools[index])),
        })
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
/// A root driven through per-pool journals stores each pool's durable coverage
/// in that pool's own (now-retired) LSN space. The shared segment that replaces
/// them must begin above every one of those watermarks, so each legacy coverage
/// value stays meaningful in the single shared space: a restarted shared writer
/// can neither replay past a pool's legacy coverage nor reclaim beneath a fresh
/// frame that only looks covered because numbering restarted. Returns `1` for a
/// root with no recorded coverage (a brand-new database).
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
        *overlay = Some(database.activate_transaction_overlay().unwrap());
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
        assert_eq!(db.layout().wal_layout(), crate::layout::WalLayout::Shared);
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

        let mut overlay = Some(database.activate_transaction_overlay().unwrap());
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
        let mut overlay = Some(database.activate_transaction_overlay().unwrap());
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

    #[test]
    fn a_legacy_pool_coverage_seeds_the_shared_wal_above_it() {
        // The layout marker is informational (see `DatabaseLayout::open`): both
        // a PerPool-marked legacy root and a Shared-marked root whose sidecars
        // predate the shared segment must seed the fresh WAL above the highest
        // per-pool watermark, so legacy coverage values stay meaningful in the
        // one shared LSN space.
        for (name, layout_code) in [("legacy_seed_perpool", 0u8), ("legacy_seed_shared", 1u8)] {
            let root = test_root(name);
            std::fs::create_dir_all(root.join("pools/mtpl-state")).unwrap();
            let mut meta = Vec::from(b"MTXD".as_slice());
            meta.push(1);
            meta.extend_from_slice(&[0u8; 8]);
            meta[4 + 1] = layout_code; // reserved[0] is the WAL-layout byte
            meta.extend_from_slice(b"state\nevent\nedges\n");
            std::fs::write(root.join("db.meta"), meta).unwrap();
            // A checkpoint watermark recorded before the shared segment existed.
            std::fs::write(
                root.join("pools/mtpl-state/journal.lsn"),
                7u64.to_le_bytes(),
            )
            .unwrap();

            let db = SharedDatabase::open(root.clone()).unwrap();
            let scan = Journal::scan_read_only(root.join("wal.bin")).unwrap();
            assert!(
                scan.base_lsn > 7,
                "{name}: the fresh shared WAL must start above the recorded coverage"
            );
            db.pool(ShardType::State)
                .put(&[0x55u8; 16], &node(3), &NodeData::from_slice(b"live"))
                .unwrap();
            db.pool(ShardType::State).sync_all().unwrap();
            assert!(
                db.coordinator().committed_lsn_for_pool(ShardType::State) > 7,
                "{name}: a new shared frame must be numbered above the watermark"
            );
            drop(db);
            let _ = std::fs::remove_dir_all(&root);
        }
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
    fn a_legacy_root_opens_onto_a_fresh_root_wal() {
        let root = test_root("legacy_open");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("db.meta"),
            // `WalLayout::PerPool` marker: reserved WAL-layout byte 0.
            {
                let mut bytes = Vec::from(b"MTXD".as_slice());
                bytes.push(1);
                bytes.extend_from_slice(&[0u8; 8]);
                bytes.extend_from_slice(b"state\nevent\nedges\n");
                bytes
            },
        )
        .unwrap();
        // A stale per-pool WAL that must be ignored, not replayed.
        let stale_pool = root.join("pools/mtpl-state");
        std::fs::create_dir_all(&stale_pool).unwrap();
        std::fs::write(stale_pool.join("wal.bin"), b"not a real segment").unwrap();

        let db = SharedDatabase::open(root.clone()).unwrap();
        assert_eq!(db.layout().wal_layout(), crate::layout::WalLayout::PerPool);
        assert!(root.join("wal.bin").is_file(), "a root WAL is created");
        db.pool(ShardType::State)
            .put(&[0x22u8; 16], &node(2), &NodeData::from_slice(b"live"))
            .unwrap();
        db.pool(ShardType::State).sync_all().unwrap();
        let got = db
            .pool(ShardType::State)
            .get(&[0x22u8; 16], &node(2))
            .unwrap()
            .expect("a write on the fresh shared WAL must be readable");
        assert_eq!(&got.bytes[..], b"live");
        drop(db);
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
