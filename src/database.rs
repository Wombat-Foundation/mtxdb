//! Root-level handle for a shared-WAL database.
//!
//! [`SharedDatabase`] is the safe end-to-end entry point for the shared
//! durability fence: it opens a database root, acquires the single root writer
//! lock, opens the root's pool-tagged `wal.bin` as one
//! [`crate::journal::JournalCoordinator`], opens every named pool, attaches each to that one
//! coordinator, and replays recovered groups. It holds the root lock for its
//! whole lifetime, so exactly one writer process owns the database.
//!
//! Legacy per-pool roots are never opened; [`DatabaseLayout`] fails closed on
//! them. Callers that need to drive a single pool directly can still use
//! [`crate::journal::SharedWalLock`] with
//! [`PackfileStorage::enable_shared_journal`](crate::PackfileStorage::enable_shared_journal).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use crate::journal::{Journal, JournalCoordinator, SharedWalLock};
use crate::layout::{DatabaseLayout, ShardType};
use crate::packfile::storage::PackfileStorage;
use crate::storage::StorageError;

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

impl SharedDatabase {
    /// Open `root` for writing through one shared WAL.
    ///
    /// The root is initialized with the shared layout if it does not yet
    /// exist. This blocks with `WouldBlock` if another live process holds the
    /// root writer lock.
    ///
    /// # Errors
    /// Returns an error if the root is a legacy per-pool root, the root lock is
    /// held, the shared segment cannot be opened, or any pool cannot be opened
    /// or attached.
    pub fn open(root: PathBuf) -> Result<Self, StorageError> {
        let layout = DatabaseLayout::open(root)?;
        let wal_path = layout.shared_wal_path();
        let lock = SharedWalLock::acquire(layout.root())?;
        // A root that was driven through per-pool journals carries each pool's
        // durable coverage as a `journal.lsn` below its pool directory. Start
        // the fresh shared segment above all of them, so every legacy coverage
        // value stays meaningful in the one shared LSN space: a restarted
        // shared writer can never replay past (or reclaim beneath) coverage
        // that belonged to a different numbering.
        let seed_lsn = {
            let mut watermark = 0_u64;
            for shard in ShardType::ALL {
                let dir = layout.pool_dir(shard)?;
                watermark = watermark.max(PackfileStorage::read_journal_lsn(&dir));
            }
            watermark.saturating_add(1)
        };
        let (journal, scan) = Journal::open_shared_seeded(&wal_path, seed_lsn)?;
        let coordinator = Arc::new(JournalCoordinator::new(journal, &scan));

        let mut pools = Vec::with_capacity(ShardType::ALL.len());
        for shard in ShardType::ALL {
            let dir = layout.pool_dir(shard)?;
            let store = PackfileStorage::open(dir)?;
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
}

/// Index of `shard` in the fixed `[State, EventDag, AuthChain]` pool array.
const fn shard_index(shard: ShardType) -> usize {
    match shard {
        ShardType::State => 0,
        ShardType::EventDag => 1,
        ShardType::AuthChain => 2,
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
        let root = test_root("legacy_seed");
        std::fs::create_dir_all(root.join("pools/state")).unwrap();
        std::fs::write(root.join("db.meta"), {
            let mut bytes = Vec::from(b"MDBD".as_slice());
            bytes.push(1);
            bytes.extend_from_slice(&[0u8; 8]);
            bytes.extend_from_slice(b"state\nevent-dag\nauth-chain\n");
            bytes
        })
        .unwrap();
        // A per-pool checkpoint watermark from the legacy layout.
        std::fs::write(root.join("pools/state/journal.lsn"), 7u64.to_le_bytes()).unwrap();

        let db = SharedDatabase::open(root.clone()).unwrap();
        let scan = Journal::scan_read_only(root.join("wal.bin")).unwrap();
        assert!(
            scan.base_lsn > 7,
            "the fresh shared WAL must start above the legacy per-pool coverage"
        );
        db.pool(ShardType::State)
            .put(&[0x55u8; 16], &node(3), &NodeData::from_slice(b"live"))
            .unwrap();
        db.pool(ShardType::State).sync_all().unwrap();
        assert!(
            db.coordinator().committed_lsn_for_pool(ShardType::State) > 7,
            "a new shared frame must be numbered above the legacy watermark"
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
    fn a_legacy_root_opens_onto_a_fresh_root_wal() {
        let root = test_root("legacy_open");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("db.meta"),
            // `WalLayout::PerPool` marker: reserved WAL-layout byte 0.
            {
                let mut bytes = Vec::from(b"MDBD".as_slice());
                bytes.push(1);
                bytes.extend_from_slice(&[0u8; 8]);
                bytes.extend_from_slice(b"state\nevent-dag\nauth-chain\n");
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
}
