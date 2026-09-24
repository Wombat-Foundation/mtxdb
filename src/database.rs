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
#![cfg(not(target_arch = "wasm32"))]

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::journal::{
    CommitReceipt, Journal, JournalCoordinator, SharedWalLock, TxnStage, TxnStageState,
};
use crate::layout::{DatabaseLayout, ShardType};
use crate::packfile::storage::PackfileStorage;
use crate::storage::StorageError;

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
    /// Held for the lifetime of the handle: one writer per database root.
    _lock: SharedWalLock,
}

/// Storage transaction whose mutations remain invisible until commit.
pub struct DatabaseTransaction<'a> {
    database: &'a SharedDatabase,
    stage: TxnStage,
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
        self.stage
            .stage_put(pool, collection_id, node_id, data.bytes.to_vec())
    }

    /// Stage removal of a collection.
    ///
    /// # Errors
    /// Returns an error if the transaction's staging budget is exhausted or
    /// it has already been committed/aborted.
    pub fn delete_collection(&self, pool: ShardType, collection_id: [u8; 16]) -> io::Result<()> {
        self.stage.stage_delete_collection(pool, collection_id)
    }

    /// Commit the staged mutations. Pack/index application is performed once;
    /// journal publication can be retried if the post-commit callback fails.
    ///
    /// # Errors
    /// Returns an error if storage application or shared-WAL publication
    /// fails. After storage application succeeds, retrying this method is safe.
    pub fn commit(&self) -> Result<(), StorageError> {
        if self.stage.state() == TxnStageState::Active {
            self.database
                .publish_transaction(&self.stage)
                .map_err(StorageError::Io)?;
        }
        if self.stage.state() == TxnStageState::JournalPublished {
            let batches = self.stage.snapshot_mutations();
            for pool in ShardType::ALL {
                for (index, mutation) in batches[shard_index(pool)].iter().enumerate() {
                    if self.stage.mutation_applied(pool, index) {
                        continue;
                    }
                    self.database
                        .pool(pool)
                        .apply_transaction_mutation(mutation)?;
                    self.stage
                        .mark_mutation_applied(pool, index)
                        .map_err(StorageError::Io)?;
                }
            }
            self.stage.mark_applied().map_err(StorageError::Io)?;
            self.stage.mark_published().map_err(StorageError::Io)?;
        }
        Ok(())
    }

    /// Abort the transaction. Since staged mutations have not touched storage,
    /// abort is O(1) and leaves no revocation records behind.
    pub fn abort(&self) {
        self.stage.discard();
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
            database: self,
            stage: TxnStage::new(),
        }
    }

    /// Publish all legacy pending mutations through the database coordinator.
    ///
    /// A shared database has one coordinator for all pools, so this is one
    /// cross-pool visibility operation. It does not fsync; call `sync_all` or
    /// the normal durability path separately.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned or the pending queue cannot
    /// be appended as one complete group.
    pub fn publish_pending(&self) -> io::Result<Option<CommitReceipt>> {
        self.coordinator.publish_pending()
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
    use super::SharedDatabase;
    use crate::journal::Journal;
    use crate::layout::ShardType;
    use crate::packfile::storage::PackfileStorage;
    use crate::storage::{NodeData, NodeId, StorageEngine};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn test_root(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("mtxdb-shared-db-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
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
        transaction.abort();
        assert!(db
            .pool(ShardType::State)
            .get(&collection, &node_id)
            .unwrap()
            .is_none());
        drop(db);
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
        drop(db);
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

        let before = PackfileStorage::read_journal_lsn(&root.join("pools/state"));
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
            PackfileStorage::read_journal_lsn(&root.join("pools/state")),
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
            std::fs::create_dir_all(root.join("pools/state")).unwrap();
            let mut meta = Vec::from(b"MTXD".as_slice());
            meta.push(1);
            meta.extend_from_slice(&[0u8; 8]);
            meta[4 + 1] = layout_code; // reserved[0] is the WAL-layout byte
            meta.extend_from_slice(b"state\nevent\nedges\n");
            std::fs::write(root.join("db.meta"), meta).unwrap();
            // A checkpoint watermark recorded before the shared segment existed.
            std::fs::write(root.join("pools/state/journal.lsn"), 7u64.to_le_bytes()).unwrap();

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
        let stale_pool = root.join("pools/state");
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
            .enable_journal(root.join("pools/state/wal.bin"))
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
