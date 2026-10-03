use super::{Database, DatabaseTransaction, TransactionOverlayGuard};
use crate::journal::Journal;
use crate::layout::ShardType;
use crate::packfile::storage::PackfileStorage;
use crate::storage::{NodeData, NodeId, StorageEngine};
use std::path::PathBuf;
use std::sync::{Arc, Barrier};

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-shared-db-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn prepare_partial_materialization(
    database: &Database,
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
    let mutation = transaction.stage.snapshot_mutations()[ShardType::State.index()][0].clone();
    database
        .pool(ShardType::State)
        .apply_transaction_mutation(&mutation, 0)
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
    database: &Database,
    pool: ShardType,
    collection: [u8; 16],
    id: NodeId,
) -> Option<Vec<u8>> {
    payload_of(database.pool(pool).get(&collection, &id).unwrap().as_ref())
}

/// Activate the overlay for exactly the pools `stage` has writes for, as
/// commit does.
fn activate_for(database: &Database, stage: &crate::journal::TxnStage) -> TransactionOverlayGuard {
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Arc::new(Database::open(root.clone()).unwrap());
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
    let database = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
    let collection = [7u8; 16];
    database
        .pool(ShardType::State)
        .put(&[0x99; 16], &node(1), &data(b"seed"))
        .unwrap();
    database.pool(ShardType::State).sync().unwrap();
    let wal = crate::journal::Journal::scan_read_only(database.layout().shared_wal_path()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
fn server_info_is_a_shared_wal_transaction_pool() {
    let root = test_root("server_info_shared_wal");
    let collection = [0x64; 16];
    let db = Database::open(root.clone()).unwrap();
    let transaction = db.begin_transaction();
    transaction
        .put(
            ShardType::ServerInfo,
            collection,
            node(4),
            &data(b"server-info"),
        )
        .unwrap();
    transaction
        .put(ShardType::State, collection, node(5), &data(b"state"))
        .unwrap();
    assert_eq!(
        transaction.stage.touched_pools(),
        [true, false, false, true]
    );

    transaction.commit().unwrap();
    let scan = Journal::scan_read_only(root.join("wal.bin")).unwrap();
    assert_eq!(scan.groups.len(), 1);
    let pools = scan.groups[0]
        .entries
        .iter()
        .map(|entry| entry.pool)
        .collect::<Vec<_>>();
    assert!(pools.contains(&ShardType::State));
    assert!(pools.contains(&ShardType::ServerInfo));

    drop(transaction);
    drop(db);
    let reopened = Database::open(root.clone()).unwrap();
    assert_eq!(
        reopened
            .pool(ShardType::ServerInfo)
            .get(&collection, &node(4))
            .unwrap()
            .unwrap()
            .bytes
            .as_ref(),
        b"server-info"
    );
    drop(reopened);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn transaction_abort_keeps_staged_mutations_invisible() {
    let root = test_root("transaction_abort");
    let db = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
#[test]
fn lone_commits_scan_only_the_appended_suffix() {
    let root = test_root("overlay_scan_bounded");
    let db = Database::open(root.clone()).unwrap();
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

/// On a shared WAL, one pool's checkpoint tail in flight holds only that
/// pool's coverage: the other pool checkpoints and reports as usual, the
/// shared segment keeps what the held pool has not covered (and names it as
/// the blocker), a crash at that instant loses nothing, and once the tail
/// installs its image the segment is reclaimed.
#[test]
fn a_pool_in_a_background_checkpoint_holds_only_its_own_coverage() {
    let root = test_root("shared_wal_background_checkpoint");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x5B; 16];
    let state = db.pool(ShardType::State);
    let events = db.pool(ShardType::EventDag);
    state.set_background_checkpoint(true);
    let commit = |seq: u8| {
        let txn = db.begin_transaction();
        for pool in [ShardType::State, ShardType::EventDag] {
            txn.put(pool, collection, node(seq), &data(b"payload"))
                .unwrap();
        }
        txn.commit().unwrap();
    };
    commit(1);
    let release = state.hold_next_checkpoint_tail(false);
    state.sync().unwrap();
    assert!(state.checkpoint_in_flight());
    let state_before = state.durable_coverage();

    commit(2);
    events.sync_all().unwrap();
    assert!(
        events.durable_coverage() > 0,
        "the other pool checkpoints and reports as usual"
    );
    assert_eq!(
        state.durable_coverage(),
        state_before,
        "the held pool claims nothing until its image is durable"
    );
    assert!(
        db.coordinator()
            .reclaim_blockers()
            .contains(&ShardType::State),
        "the held pool is what holds the shared segment back"
    );
    // Another sync of the held pool appends only; it starts no second tail.
    state.sync().unwrap();
    assert!(state.checkpoint_in_flight());

    // A crash now (packs cut to what an fsync covered): the copy reopens
    // with every committed record.
    let crashed = test_root("shared_wal_background_checkpoint_crash");
    crash_image(&db, &root, &crashed);
    let after_crash = Database::open(crashed.clone()).unwrap();
    for pool in [ShardType::State, ShardType::EventDag] {
        for seq in [1, 2] {
            assert!(
                live_get(&after_crash, pool, collection, node(seq)).is_some(),
                "{pool:?} record {seq} must survive a crash mid-tail"
            );
        }
    }
    drop(after_crash);
    let _ = std::fs::remove_dir_all(crashed);

    drop(release);
    state.wait_for_checkpoint();
    let covered_by_tail = state.durable_coverage();
    assert!(covered_by_tail > state_before);
    // The tail covered what was committed at its handoff. Commit 2 came
    // later, so State still holds the segment until a coverage step covers
    // it: force one by making the segment over its trigger.
    assert!(
        db.coordinator()
            .reclaim_blockers()
            .contains(&ShardType::State),
        "commit 2 is not covered by the tail's image"
    );
    db.coordinator().set_reclaim_trigger_len(1);
    state.sync_all().unwrap();
    state.wait_for_checkpoint();
    events.sync_all().unwrap();
    assert!(state.durable_coverage() > covered_by_tail);
    assert!(
        !db.coordinator()
            .reclaim_blockers()
            .contains(&ShardType::State),
        "once State covers commit 2 it no longer holds the segment"
    );
    drop(db);
    let reopened = Database::open(root.clone()).unwrap();
    for pool in [ShardType::State, ShardType::EventDag] {
        for seq in [1, 2] {
            assert!(live_get(&reopened, pool, collection, node(seq)).is_some());
        }
    }
    drop(reopened);
    let _ = std::fs::remove_dir_all(root);
}

/// Remediation of a pool whose checkpoint tail is already running starts no
/// second tail (it would rotate the epoch again and race the first tail on
/// the same checkpoint file); remediation of the other pool starts its own
/// and leaves the held pool's coverage alone. Every record survives.
#[test]
fn remediating_a_pool_with_a_tail_running_starts_no_second_tail() {
    let root = test_root("shared_wal_remediation_with_tail");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x5C; 16];
    let state = db.pool(ShardType::State);
    let events = db.pool(ShardType::EventDag);
    state.set_background_checkpoint(true);
    events.set_background_checkpoint(true);
    let commit = |seq: u8| {
        let txn = db.begin_transaction();
        for pool in [ShardType::State, ShardType::EventDag] {
            txn.put(pool, collection, node(seq), &data(b"payload"))
                .unwrap();
        }
        txn.commit().unwrap();
    };
    commit(1);
    let release = state.hold_next_checkpoint_tail(false);
    state.sync().unwrap();
    assert!(state.checkpoint_in_flight());
    assert_eq!(state.stats().checkpoint_tails_started, 1);
    let state_before = state.durable_coverage();

    commit(2);
    state.force_index_checkpoint_detached().unwrap();
    assert_eq!(
        state.stats().checkpoint_tails_started,
        1,
        "remediating a pool whose tail is running must not start another"
    );
    events.force_index_checkpoint_detached().unwrap();
    assert_eq!(events.stats().checkpoint_tails_started, 1);
    events.wait_for_checkpoint();
    assert!(events.durable_coverage() > 0);
    assert_eq!(
        state.durable_coverage(),
        state_before,
        "the other pool's remediation leaves the held pool's coverage alone"
    );

    // A crash at this instant — State's tail still held, EventDag's already
    // installed — must lose nothing: the copy reopens with every committed
    // record of both pools.
    let crashed = test_root("shared_wal_remediation_with_tail_crash");
    crash_image(&db, &root, &crashed);
    let after_crash = Database::open(crashed.clone()).unwrap();
    for pool in [ShardType::State, ShardType::EventDag] {
        for seq in [1, 2] {
            assert!(
                live_get(&after_crash, pool, collection, node(seq)).is_some(),
                "{pool:?} record {seq} must survive a crash with a tail held"
            );
        }
    }
    drop(after_crash);
    let _ = std::fs::remove_dir_all(crashed);

    drop(release);
    state.wait_for_checkpoint();
    assert!(state.durable_coverage() > state_before);
    drop(db);
    let reopened = Database::open(root.clone()).unwrap();
    for pool in [ShardType::State, ShardType::EventDag] {
        for seq in [1, 2] {
            assert!(
                live_get(&reopened, pool, collection, node(seq)).is_some(),
                "{pool:?} record {seq} must survive"
            );
        }
    }
    drop(reopened);
    let _ = std::fs::remove_dir_all(root);
}

/// While a transaction holds the overlay it is on the read path, so its
/// staged group is visible before materialization; once the last user
/// releases it, the overlay leaves the read path, and writer reads stop
/// refreshing a journal that only repeats the index.
#[test]
fn transaction_overlay_is_on_the_read_path_only_while_a_transaction_uses_it() {
    let root = test_root("overlay_read_path");
    let db = Database::open(root.clone()).unwrap();
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
#[test]
fn writer_reads_leave_the_overlay_off_after_commits_and_sync() {
    let root = test_root("overlay_writer_reads");
    let db = Database::open(root.clone()).unwrap();
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
#[test]
fn concurrent_reads_see_every_finished_commit() {
    use std::sync::atomic::{AtomicU8, Ordering};
    let root = test_root("overlay_concurrent");
    let db = Arc::new(Database::open(root.clone()).unwrap());
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
#[test]
fn reading_an_unstaged_key_during_an_active_overlay_reaches_the_index() {
    let root = test_root("overlay_unstaged_read");
    let db = Database::open(root.clone()).unwrap();
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
const TEST_RECLAIM_TRIGGER_LEN: u64 = 256 << 10;

/// Write committed transactions that stage into each of the `active` pools
/// and sync every round, idle pools first while the segment is still
/// large. Returns the largest WAL seen after a round's syncs (past round
/// 0) and asserts that no idle pool ever rewrites its checkpoint or moves its
/// `journal.lsn` after its first checkpoint, while every active pool does
/// checkpoint once the segment is large. The segment can only be reclaimed
/// once all the active pools have reported coverage.
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
    let db = Database::open(root.clone()).unwrap();
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
    let lsn_of = |pool: ShardType| PackfileStorage::read_journal_lsn(&db.layout().pool_path(pool));
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
fn assert_wal_bounded(name: &str, active: &[ShardType]) {
    assert_wal_bounded_with(name, active, false);
}

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
#[test]
fn repeated_syncs_keep_the_shared_wal_bounded_with_only_state_active() {
    assert_wal_bounded("wal_bounded_state", &[ShardType::State]);
}

#[test]
fn repeated_syncs_keep_the_shared_wal_bounded_with_only_event_dag_active() {
    assert_wal_bounded("wal_bounded_event", &[ShardType::EventDag]);
}

#[test]
fn repeated_syncs_keep_the_shared_wal_bounded_with_only_edges_active() {
    assert_wal_bounded("wal_bounded_edges", &[ShardType::Edges]);
}

/// With two contributing pools the segment is only reclaimable once both
/// have reported coverage, and the third stays idle throughout.
#[test]
fn repeated_syncs_keep_the_shared_wal_bounded_with_two_pools_active() {
    assert_wal_bounded("wal_bounded_two", &[ShardType::State, ShardType::EventDag]);
}

/// A configured checkpoint-rewrite deferral budget must not postpone the
/// checkpoint the size trigger forces: the segment would fill and commits
/// would fail.
#[test]
fn a_deferral_budget_does_not_hold_back_the_forced_reclaim() {
    assert_wal_bounded_with("wal_bounded_deferred", &[ShardType::EventDag], true);
}

/// A transaction that stages writes for one pool must not activate (and so
/// rescan the journal for) the overlay on the others.
#[test]
fn overlay_is_activated_only_on_the_pools_a_transaction_stages() {
    let root = test_root("overlay_touched_pools");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x7b; 16];
    let txn = db.begin_transaction();
    txn.put(ShardType::EventDag, collection, node(1), &data(b"staged"))
        .unwrap();
    assert_eq!(txn.stage.touched_pools(), [false, true, false, false]);
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
    assert_eq!(mixed.stage.touched_pools(), [true, false, true, false]);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Commit `count` transactions, each staging 20 records into every pool in
/// `pools`. Stops at the first error.
fn commit_batch(
    db: &Database,
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
fn small_segment_database(name: &str) -> (Database, PathBuf) {
    let root = test_root(name);
    let db = Database::open(root.clone()).unwrap();
    db.coordinator().set_segment_cap(1 << 20);
    (db, root)
}

/// `State` and `EventDag` both have committed frames, but only `EventDag` is ever
/// synced. With nothing to remediate the lag, reclaim stalls: it must be
/// reported with the pool named, `EventDag` must stop repeating a checkpoint
/// that cannot help, and a commit that finally hits the hard limit must say
/// which pool it is waiting on instead of failing silently.
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
pub(crate) fn crash_image(db: &Database, source: &std::path::Path, dest: &std::path::Path) {
    fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                // Writer lock files are live byte-range locked handles on
                // Windows and are not part of a crash image.
                if matches!(
                    entry.file_name().to_str(),
                    Some(".mtxdb.lock" | ".mtxdb.wal.lock")
                ) {
                    continue;
                }
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
/// writes and syncs, the disk image a power loss would leave (packs cut to
/// their fsynced length, WAL as it is) must still hold every record that was
/// acknowledged and made durable by a full sync of its pool.
//
// This is deliberately a crash-image test, not a claim about a physical
// device losing power. It models loss of dirty page-cache bytes by truncating
// packfiles to their last known synced length; filesystem/controller write
// reordering and lying hardware caches are outside what a portable Rust test
// can establish.
#[test]
fn a_power_cut_image_never_loses_a_record_a_claim_covered() {
    let root = test_root("cut_image_source");
    let image = test_root("cut_image_copy");
    let collection = [0x6e; 16];
    let pools = [ShardType::State, ShardType::EventDag];
    let db = Database::open(root.clone()).unwrap();
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
        let recovered = Database::open(image.clone()).unwrap();
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
#[test]
fn a_checkpoint_does_not_claim_an_unmaterialized_transaction() {
    use crate::PackfileStorage;
    let root = test_root("checkpoint_unmaterialized");
    let db = Database::open(root.clone()).unwrap();
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

/// Shared-WAL coverage advances only after each pool writes the checkpoint
/// version table. The shared segment is reclaimed only once both have
/// reported: after the first pool's step the second still holds it, and after
/// the second's the reclaimed segment holds no covered group.
#[test]
fn shared_reclaim_waits_for_every_pools_version_checkpoint() {
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
        !timings.checkpoint.is_zero(),
        "EventDag coverage must anchor its logical-version table"
    );
    assert_eq!(
        coordinator.segment_len(),
        before,
        "State has not reported, so nothing may be reclaimed"
    );
    assert_eq!(coordinator.reclaim_blockers(), vec![ShardType::State]);

    db.pool(ShardType::State).sync_all().unwrap();
    let timings = db.pool(ShardType::State).sync_timings().unwrap();
    assert!(
        !timings.checkpoint.is_zero(),
        "State coverage must anchor its logical-version table"
    );
    assert!(
        coordinator.segment_len() < before,
        "both pools have reported, so the segment shrinks"
    );
    assert_eq!(coordinator.reclaim_blockers(), Vec::<ShardType>::new());
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn transaction_commit_applies_and_publishes_once() {
    let root = test_root("transaction_commit");
    let db = Database::open(root.clone()).unwrap();
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
        vec![ShardType::EventDag, ShardType::State]
    );
    drop(transaction);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn versioned_transaction_read_rejects_stale_commit_and_retries() {
    let root = test_root("versioned_transaction_read");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::Edges;
    let collection = [0x73; 16];
    let key = node(1);
    let other_key = node(2);

    let seed = db.begin_transaction();
    seed.put(pool, collection, key, &data(b"seed")).unwrap();
    seed.commit().unwrap();

    let stale = db.begin_transaction();
    let (records, expected) = stale
        .get_with_collection_version(pool, &collection, &[key])
        .unwrap();
    assert_eq!(payload_of(records[0].as_ref()), Some(b"seed".to_vec()));
    stale
        .put(pool, collection, key, &data(b"stale-write"))
        .unwrap();

    let concurrent = db.begin_transaction();
    concurrent
        .put(pool, collection, other_key, &data(b"concurrent"))
        .unwrap();
    concurrent.commit().unwrap();

    stale
        .expect_collection_version(pool, collection, expected)
        .unwrap();
    match stale.commit() {
        Err(crate::storage::StorageError::StaleRead {
            pool: found_pool,
            collection_id,
            expected: found_expected,
            actual,
        }) => {
            assert_eq!(found_pool, pool);
            assert_eq!(collection_id, collection);
            assert_eq!(found_expected, expected);
            assert!(actual > expected);
        }
        other => panic!("expected stale-read rejection, got {other:?}"),
    }

    let retry = db.begin_transaction();
    let (records, version) = retry
        .get_with_collection_version(pool, &collection, &[key, other_key])
        .unwrap();
    assert_eq!(payload_of(records[0].as_ref()), Some(b"seed".to_vec()));
    assert_eq!(
        payload_of(records[1].as_ref()),
        Some(b"concurrent".to_vec())
    );
    retry
        .put(pool, collection, key, &data(b"retried-write"))
        .unwrap();
    retry
        .expect_collection_version(pool, collection, version)
        .unwrap();
    retry.commit().unwrap();
    assert_eq!(
        live_get(&db, pool, collection, key),
        Some(b"retried-write".to_vec())
    );

    let (_, scan_version) = db
        .pool(pool)
        .get_with_collection_version(&collection, &[key])
        .unwrap();
    let (scan, _cursor, lease) = db
        .pool(pool)
        .scan_collection_at_snapshot(&collection)
        .unwrap();
    let scanned = scan.collect::<Result<Vec<_>, _>>().unwrap();
    assert!(!scanned.is_empty());
    assert!(db
        .pool(pool)
        .recheck_collection_version(&collection, scan_version)
        .unwrap());
    drop(lease);

    let later = db.begin_transaction();
    later
        .put(pool, collection, other_key, &data(b"after-scan"))
        .unwrap();
    later.commit().unwrap();
    assert!(!db
        .pool(pool)
        .recheck_collection_version(&collection, scan_version)
        .unwrap());

    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn versioned_transaction_byte_read_preserves_payload_without_reencoding() {
    let root = test_root("versioned_transaction_bytes");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::Edges;
    let collection = [0x74; 16];
    let key = node(1);

    let writer = db.begin_transaction();
    writer
        .put(pool, collection, key, &data(b"zero-copy-payload"))
        .unwrap();
    writer.commit().unwrap();

    let reader = db.begin_transaction();
    let (records, _version) = reader
        .get_with_collection_version_bytes(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_deref(), Some(&b"zero-copy-payload"[..]));

    drop(reader);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn versioned_read_retries_when_publication_is_not_materialized() {
    let root = test_root("versioned_read_unmaterialized");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::Edges;
    let collection = [0x75; 16];
    let key = node(1);

    let transaction = db.begin_transaction();
    transaction
        .put(pool, collection, key, &data(b"published-not-materialized"))
        .unwrap();
    // Deliberately publish without installing an overlay or materializing the
    // transaction. A no-overlay read must not claim MAX coverage here.
    db.publish_transaction(&transaction.stage).unwrap();

    let error = db
        .pool(pool)
        .get_with_collection_version(&collection, &[key])
        .unwrap_err();
    assert!(
        matches!(error, crate::storage::StorageError::WouldBlock(_)),
        "expected a retryable read while materialization is pending, got {error:?}"
    );

    drop(transaction);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn versioned_point_reads_match_their_token_during_concurrent_publication() {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    let root = test_root("versioned_point_read_concurrent");
    let db = Arc::new(Database::open(root.clone()).unwrap());
    let pool = ShardType::Edges;
    let collection = [0x74; 16];
    let key = node(3);

    let seed = db.begin_transaction();
    seed.put(
        pool,
        collection,
        key,
        &NodeData::new(0u64.to_le_bytes().to_vec().into()),
    )
    .unwrap();
    seed.commit().unwrap();
    let coordinator = db.coordinator();
    let seed_version = coordinator.collection_version(pool, &collection);
    let values = Arc::new(Mutex::new(HashMap::from([(seed_version, 0u64)])));
    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(2));

    let samples = std::thread::scope(|scope| {
        let writer_db = Arc::clone(&db);
        let writer_values = Arc::clone(&values);
        let writer_done = Arc::clone(&done);
        let writer_start = Arc::clone(&start);
        let writer = scope.spawn(move || {
            writer_start.wait();
            for value in 1u64..=96 {
                let transaction = writer_db.begin_transaction();
                transaction
                    .put(
                        pool,
                        collection,
                        key,
                        &NodeData::new(value.to_le_bytes().to_vec().into()),
                    )
                    .unwrap();
                transaction.commit().unwrap();
                let version = writer_db
                    .coordinator()
                    .collection_version(pool, &collection);
                writer_values.lock().unwrap().insert(version, value);
                std::thread::yield_now();
            }
            writer_done.store(true, Ordering::Release);
        });

        let reader_db = Arc::clone(&db);
        let reader_done = Arc::clone(&done);
        let reader_start = Arc::clone(&start);
        let reader = scope.spawn(move || {
            reader_start.wait();
            let mut samples = Vec::new();
            while !reader_done.load(Ordering::Acquire) {
                let (records, version) = match reader_db
                    .pool(pool)
                    .get_with_collection_version(&collection, &[key])
                {
                    Ok(records) => records,
                    Err(error) if error.is_would_block() => {
                        std::thread::yield_now();
                        continue;
                    }
                    Err(error) => panic!("versioned read failed: {error}"),
                };
                let value = records[0]
                    .as_ref()
                    .map(|record| u64::from_le_bytes(record.bytes[..].try_into().unwrap()));
                samples.push((version, value));
            }
            samples
        });

        writer.join().unwrap();
        reader.join().unwrap()
    });

    assert!(!samples.is_empty(), "reader must overlap writer activity");
    let values = values.lock().unwrap();
    let (mut last_version, mut last_value) = (0u64, 0u64);
    for (version, value) in samples {
        // A published group can become readable to the reader before the
        // writer's own clock sample observes the bump, so a token may lag the
        // newest write it covers. The guarantee is one-directional: the data
        // must reflect every write up to the token, never fewer. A token names
        // an already-recorded publish, and the value read must be at least that
        // publish's value.
        let recorded = values
            .get(&version)
            .copied()
            .expect("a returned token always corresponds to a recorded publish");
        let value = value.expect("the seeded key must always read back");
        assert!(
            value >= recorded,
            "value {value} read for token {version} predates its publish {recorded}"
        );
        assert!(
            version >= last_version && value >= last_value,
            "version and value must not go backwards"
        );
        last_version = version;
        last_value = value;
    }
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn retrying_commit_and_recovery_can_run_concurrently() {
    let root = test_root("transaction_concurrent_recovery");
    let database = Arc::new(Database::open(root.clone()).unwrap());
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

/// The unmaterialized floor is the only thing that keeps a versioned read from
/// certifying a published-but-unmaterialized group. It must be pinned while
/// such a group exists and released once the group reaches the packs, or every
/// versioned read wedges behind a phantom publication.
#[test]
fn unmaterialized_floor_pins_until_group_is_materialized() {
    use std::sync::Arc;

    let root = test_root("unmaterialized_floor");
    let database = Database::open(root.clone()).unwrap();
    let collection = [0x5au8; 16];
    let pool = ShardType::State;

    let settled = database.begin_transaction();
    settled
        .put(
            pool,
            collection,
            node(1),
            &NodeData::new(bytes::Bytes::from_static(b"settled")),
        )
        .unwrap();
    settled.commit().unwrap();
    assert_eq!(
        database.coordinator().unmaterialized_floor(),
        u64::MAX,
        "a fully materialized group must not hold the floor down"
    );

    let pending = database.begin_transaction();
    pending
        .put(
            pool,
            collection,
            node(2),
            &NodeData::new(bytes::Bytes::from_static(b"pending")),
        )
        .unwrap();
    let mut overlay = Some(activate_for(&database, &pending.stage));
    database.publish_transaction(&pending.stage).unwrap();
    let receipt = pending.stage.published_receipt().unwrap();
    assert!(
        database.coordinator().unmaterialized_floor() < receipt.first_lsn,
        "a published-but-unmaterialized group must pin the floor below its first frame"
    );

    pending.stage.begin_materialization().unwrap();
    database
        .register_new_recovery_stage(Arc::clone(&pending.stage), receipt, &mut overlay)
        .unwrap();
    drop(pending);
    database.recover_pending_transactions().unwrap();
    assert_eq!(
        database.coordinator().unmaterialized_floor(),
        u64::MAX,
        "materialization must release the floor"
    );

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
    let database = Database::open(root.clone()).unwrap();
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
    let reopened = Database::open(root.clone()).unwrap();
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
    let database = Database::open(root.clone()).unwrap();
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
        transaction.stage.snapshot_mutations()[ShardType::State.index()][0].clone();
    database
        .pool(ShardType::State)
        .apply_transaction_mutation(&state_mutation, 0)
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
    let database = Database::open(root.clone()).unwrap();
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
        transaction.stage.snapshot_mutations()[ShardType::State.index()][0].clone();
    database
        .pool(ShardType::State)
        .apply_transaction_mutation(&state_mutation, 0)
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
        let db = Database::open(root.clone()).unwrap();
        db.pool(ShardType::State)
            .put(
                &collection,
                &node(1),
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        db.pool(ShardType::State).sync_all().unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
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
        let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    std::fs::create_dir_all(root.join("pools/mtpl-state")).unwrap();
    std::fs::write(
        root.join("pools/mtpl-state/journal.lsn"),
        7u64.to_le_bytes(),
    )
    .unwrap();
    assert!(!root.join("wal.bin").exists());

    let db = Database::open(root.clone()).unwrap();
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
    let first = Database::open(root.clone()).unwrap();
    let second = Database::open(root.clone());
    assert!(
        second.is_err(),
        "a second writer must not open the same root"
    );
    drop(first);
    // Releasing the first handle frees the root for a new writer.
    Database::open(root.clone()).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_per_pool_journal_is_refused_inside_a_shared_root() {
    let root = test_root("legacy_gate");
    let db = Database::open(root.clone()).unwrap();
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
    let db = Database::open(root.clone()).unwrap();
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
    policies.for_shard_mut(ShardType::State).compress = false;
    policies.for_shard_mut(ShardType::Edges).checksum_policy =
        crate::packfile::ChecksumPolicy::WriteOnly;

    let db = Database::open_with_policies(root.clone(), policies).unwrap();
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

/// Every concurrent versioned read-modify-write must survive. The version
/// token has to cover the data it is paired with: if a reader can observe a
/// just-published version alongside pre-publication data, its conditional
/// commit validates against that version and overwrites another writer's
/// update. A lost token means exactly that hole.
#[test]
fn concurrent_versioned_rmw_keeps_every_update() {
    use std::sync::Arc;

    const WORKERS: u8 = 16;

    let root = test_root("concurrent_versioned_rmw");
    let db = Arc::new(Database::open(root.clone()).unwrap());
    let pool = ShardType::Edges;
    let collection = [0x7au8; 16];
    let key = node(1);

    let seed = db.begin_transaction();
    seed.put(pool, collection, key, &NodeData::new(bytes::Bytes::new()))
        .unwrap();
    seed.commit().unwrap();

    let barrier = Arc::new(Barrier::new(usize::from(WORKERS)));
    let handles: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let token = worker + 1;
                loop {
                    let transaction = db.begin_transaction();
                    let (records, version) =
                        match transaction.get_with_collection_version(pool, &collection, &[key]) {
                            Ok(records) => records,
                            Err(error) if error.is_would_block() => continue,
                            Err(error) => panic!("versioned read failed: {error}"),
                        };
                    let mut list = records[0]
                        .as_ref()
                        .map_or_else(Vec::new, |record| record.bytes.to_vec());
                    if !list.contains(&token) {
                        list.push(token);
                    }
                    transaction
                        .expect_collection_version(pool, collection, version)
                        .unwrap();
                    transaction
                        .put(
                            pool,
                            collection,
                            key,
                            &NodeData::new(bytes::Bytes::from(list)),
                        )
                        .unwrap();
                    if let Err(error) = transaction.commit() {
                        if error.is_stale_read() {
                            continue;
                        }
                        panic!("commit failed: {error}");
                    }
                    break;
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    db.pool(pool).sync_all().unwrap();
    let mut final_list = db.pool(pool).get_many(&collection, &[key]).unwrap()[0]
        .as_ref()
        .map_or_else(Vec::new, |record| record.bytes.to_vec());
    final_list.sort_unstable();
    assert_eq!(final_list, (1..=WORKERS).collect::<Vec<u8>>());

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A record's write token is stable across re-reads and across a reopen, where
/// it is seeded from the retained WAL plus, for reclaimed groups, the frame's
/// `last_write_lsn`.
#[test]
fn record_versions_are_stable_and_survive_reopen() {
    let root = test_root("record_version_reopen");
    let pool = ShardType::State;
    let collection = [0x74u8; 16];
    let key = node(1);
    let write_version;
    {
        let db = Database::open(root.clone()).unwrap();
        let txn = db.begin_transaction();
        txn.put(
            pool,
            collection,
            key,
            &NodeData::new(bytes::Bytes::from_static(b"one")),
        )
        .unwrap();
        txn.commit().unwrap();
        let reader = db.begin_transaction();
        let (records, versions) = reader
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"one");
        write_version = versions[0];
        assert!(write_version > 0, "a written record has a non-zero token");
        let again = db.begin_transaction();
        let (_, versions2) = again
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        assert_eq!(versions2[0], write_version, "warm token is stable");
        db.pool(pool).sync_all().unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
    let reader = db.begin_transaction();
    let (records, versions) = reader
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"one");
    assert_eq!(
        versions[0], write_version,
        "reopened token must match the original write"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A record the live map does not know (its WAL group was reclaimed) is seeded
/// from its frame's own `last_write_lsn`.
#[test]
fn cold_record_token_seeds_from_frame_metadata() {
    let root = test_root("record_frame_seed");
    let pool = ShardType::State;
    let collection = [0x75u8; 16];
    let key = node(1);
    let write_version;
    {
        let db = Database::open(root.clone()).unwrap();
        let txn = db.begin_transaction();
        txn.put(
            pool,
            collection,
            key,
            &NodeData::new(bytes::Bytes::from_static(b"one")),
        )
        .unwrap();
        txn.commit().unwrap();
        let reader = db.begin_transaction();
        let (_, versions) = reader
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        write_version = versions[0];
        db.pool(pool).sync_all().unwrap();
    }
    // Drop the retained WAL so the reopen cannot seed the token from it; the
    // only remaining source is the record's frame metadata.
    let _ = std::fs::remove_file(root.join("wal.bin"));
    let db = Database::open(root.clone()).unwrap();
    let reader = db.begin_transaction();
    let (records, versions) = reader
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"one");
    assert_eq!(
        versions[0], write_version,
        "cold token must seed from the frame's last_write_lsn"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A frame whose record token is `0` — a legacy v4 frame with no
/// `last_write_lsn` tag, or a write made without a journal — reads as version
/// `0`, not as a never-written sentinel.
#[test]
fn zero_token_frame_reads_version_zero() {
    let root = test_root("record_legacy_zero");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::State;
    let collection = [0x76u8; 16];
    let key = node(1);
    // Apply directly with a zero token and no publication, modelling a legacy
    // frame or a write made without a journal.
    let mutation = crate::journal::Mutation::Put {
        collection_id: collection,
        node_id: key,
        payload: b"legacy".to_vec(),
    };
    db.pool(pool)
        .apply_transaction_mutation(&mutation, 0)
        .unwrap();

    let reader = db.begin_transaction();
    let (records, versions) = reader
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"legacy");
    assert_eq!(versions[0], 0, "a zero-token frame reads as version 0");
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A direct (autocommit) put stamps a non-zero token into its frame, so the
/// token survives reclamation of the WAL group and a reopen.
#[test]
fn direct_put_token_survives_reclaim_and_reopen() {
    let root = test_root("record_direct_reclaim");
    let pool = ShardType::State;
    let collection = [0x77u8; 16];
    let key = node(1);
    let write_version;
    {
        let db = Database::open(root.clone()).unwrap();
        db.pool(pool)
            .put(
                &collection,
                &key,
                &NodeData::new(bytes::Bytes::from_static(b"direct")),
            )
            .unwrap();
        let reader = db.begin_transaction();
        let (records, versions) = reader
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"direct");
        write_version = versions[0];
        assert!(
            write_version > 0,
            "a direct put now stamps a non-zero token"
        );
        db.pool(pool).sync_all().unwrap();
    }
    // Reclaim the WAL group; the frame's own stamp must still name the token.
    let _ = std::fs::remove_file(root.join("wal.bin"));
    let db = Database::open(root.clone()).unwrap();
    let reader = db.begin_transaction();
    let (records, versions) = reader
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"direct");
    assert_eq!(
        versions[0], write_version,
        "a direct-put token must survive reclaim and reopen"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A direct put that was published but not checkpointed is replayed on reopen,
/// and the replayed frame carries the group's token rather than a zero one.
#[test]
fn unsynced_direct_put_replays_with_its_token() {
    let root = test_root("record_direct_replay");
    let pool = ShardType::State;
    let collection = [0x78u8; 16];
    let key = node(1);
    let write_version;
    {
        let db = Database::open(root.clone()).unwrap();
        db.pool(pool)
            .put(
                &collection,
                &key,
                &NodeData::new(bytes::Bytes::from_static(b"direct")),
            )
            .unwrap();
        let reader = db.begin_transaction();
        let (_, versions) = reader
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        write_version = versions[0];
        assert!(write_version > 0);
        // Deliberately no sync: the group is durable in the WAL only and the
        // next open must replay it with the group's token.
    }
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(
        db.pool(pool)
            .record_last_write_lsn(&collection, &key)
            .unwrap(),
        Some(write_version),
        "replay must stamp the recovered frame with the group token"
    );
    let reader = db.begin_transaction();
    let (records, versions) = reader
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(records[0].as_ref().unwrap().bytes.as_ref(), b"direct");
    assert_eq!(versions[0], write_version);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Removing a collection leaves its records absent, but their token resolves to
/// the delete's LSN so a pre-delete token is stale and a delete token passes.
#[test]
fn absent_record_resolves_to_collection_delete_lsn() {
    let root = test_root("record_delete_token");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::State;
    let collection = [0x73u8; 16];
    let key = node(1);
    let seed = db.begin_transaction();
    seed.put(
        pool,
        collection,
        key,
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.commit().unwrap();
    let write_version = {
        let reader = db.begin_transaction();
        let (records, versions) = reader
            .get_with_record_versions(pool, &collection, &[key])
            .unwrap();
        assert!(records[0].is_some());
        assert!(versions[0] > 0);
        versions[0]
    };

    let deleter = db.begin_transaction();
    deleter.delete_collection(pool, collection).unwrap();
    deleter.commit().unwrap();

    let after = db.begin_transaction();
    let (records, versions) = after
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert!(records[0].is_none(), "the record is gone after the delete");
    let delete_version = versions[0];
    assert!(
        delete_version > write_version,
        "the absent token must advance to the delete LSN: {delete_version} vs {write_version}"
    );

    let stale = db.begin_transaction();
    stale
        .expect_record_version(pool, collection, key, write_version)
        .unwrap();
    let error = stale.commit().unwrap_err();
    assert!(
        error.is_stale_read(),
        "a pre-delete token must be rejected: {error}"
    );

    let fresh = db.begin_transaction();
    fresh
        .expect_record_version(pool, collection, key, delete_version)
        .unwrap();
    fresh.commit().unwrap();

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Per-record tokens only conflict for the same record: two writers touching
/// different records of one collection both commit.
#[test]
fn disjoint_record_writes_do_not_conflict() {
    use std::sync::{Arc, Barrier};

    let root = test_root("record_disjoint");
    let db = Arc::new(Database::open(root.clone()).unwrap());
    let pool = ShardType::State;
    let collection = [0x72u8; 16];
    let keys = [node(1), node(2)];
    let barrier = Arc::new(Barrier::new(keys.len()));
    let handles: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let key = *key;
            std::thread::spawn(move || {
                barrier.wait();
                let txn = db.begin_transaction();
                let (records, versions) = txn
                    .get_with_record_versions(pool, &collection, &[key])
                    .unwrap();
                assert!(records[0].is_none());
                txn.expect_record_version(pool, collection, key, versions[0])
                    .unwrap();
                txn.put(
                    pool,
                    collection,
                    key,
                    &NodeData::new(bytes::Bytes::from(vec![u8::try_from(index).unwrap() + 1])),
                )
                .unwrap();
                txn.commit()
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }
    for key in keys {
        assert!(db.pool(pool).get(&collection, &key).unwrap().is_some());
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Per-record tokens conflict for the same record: the second writer holding a
/// stale token is rejected.
#[test]
fn same_record_write_conflicts() {
    let root = test_root("record_same_key_conflict");
    let db = Database::open(root.clone()).unwrap();
    let pool = ShardType::State;
    let collection = [0x71u8; 16];
    let key = node(1);
    let seed = db.begin_transaction();
    seed.put(
        pool,
        collection,
        key,
        &NodeData::new(bytes::Bytes::from_static(b"seed")),
    )
    .unwrap();
    seed.commit().unwrap();

    let first = db.begin_transaction();
    let second = db.begin_transaction();
    let (_, first_versions) = first
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    let (_, second_versions) = second
        .get_with_record_versions(pool, &collection, &[key])
        .unwrap();
    assert_eq!(first_versions, second_versions);
    let version = first_versions[0];
    assert!(version > 0);

    first
        .expect_record_version(pool, collection, key, version)
        .unwrap();
    first
        .put(
            pool,
            collection,
            key,
            &NodeData::new(bytes::Bytes::from_static(b"first")),
        )
        .unwrap();
    first.commit().unwrap();

    second
        .expect_record_version(pool, collection, key, version)
        .unwrap();
    second
        .put(
            pool,
            collection,
            key,
            &NodeData::new(bytes::Bytes::from_static(b"second")),
        )
        .unwrap();
    let error = second.commit().unwrap_err();
    assert!(
        error.is_stale_read(),
        "a stale same-record token must be rejected: {error}"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

fn files_under(dir: &std::path::Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            files_under(&entry.path(), out);
        } else {
            out.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
}

/// A fresh shared database holds one root lock and descriptor: no per-pool
/// `.mtxdb.lock`, `pool.meta` or `store.meta`, and no pack until a pool is
/// written.
#[test]
fn a_fresh_shared_database_creates_only_the_files_it_needs() {
    let root = test_root("lean-layout");
    let database = Database::open(root.clone()).unwrap();
    let mut names = Vec::new();
    files_under(&root, &mut names);
    assert!(!names.iter().any(|n| n == ".mtxdb.lock"), "{names:?}");
    assert!(!names.iter().any(|n| n == "store.meta"), "{names:?}");
    assert!(
        !names.iter().any(|n| std::path::Path::new(n)
            .extension()
            .is_some_and(|e| e == "pack")),
        "{names:?}"
    );
    assert!(!names.iter().any(|n| n == "pool.meta"), "{names:?}");

    database
        .pool(ShardType::State)
        .put(&[1; 16], &node(1), &data(b"x"))
        .unwrap();
    let mut after = Vec::new();
    files_under(&root, &mut after);
    assert_eq!(
        after
            .iter()
            .filter(|n| std::path::Path::new(n)
                .extension()
                .is_some_and(|e| e == "pack"))
            .count(),
        1,
        "only the written pool gets a pack: {after:?}"
    );
}

/// Every pool's data reads back through the read-only path workers use, with
/// the seed derived from `db.meta` and no `pool.meta` on disk.
#[test]
fn pools_written_through_a_shared_database_read_back_read_only() {
    let root = test_root("seed-read-back");
    let collection = [3u8; 16];
    {
        let database = Database::open(root.clone()).unwrap();
        for (index, shard) in ShardType::ALL.into_iter().enumerate() {
            let id = u8::try_from(index).unwrap().saturating_add(1);
            let payload = [id; 8];
            database
                .pool(shard)
                .put(
                    &collection,
                    &node(id),
                    &NodeData::new(bytes::Bytes::from(payload.to_vec())),
                )
                .unwrap();
        }
        database.coordinator().sync().unwrap();
    }
    let mut names = Vec::new();
    files_under(&root, &mut names);
    assert!(!names.iter().any(|n| n == "pool.meta"), "{names:?}");

    let layout = crate::layout::DatabaseLayout::open_read_only(root.clone()).unwrap();
    for (index, shard) in ShardType::ALL.into_iter().enumerate() {
        let id = u8::try_from(index).unwrap().saturating_add(1);
        let reader =
            PackfileStorage::open_read_only(layout.pool_dir_read_only(shard).unwrap()).unwrap();
        assert_eq!(
            reader
                .get(&collection, &node(id))
                .unwrap()
                .map(|d| d.bytes.to_vec()),
            Some(vec![id; 8]),
            "{shard:?}"
        );
    }
}

fn pool_dirs(root: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.join("pools"))
        .map(|entries| {
            entries
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// `init` creates only the root. A pool's directory and first pack appear with
/// its first write, and only for the pool that was written.
#[test]
fn pools_are_created_by_their_first_write() {
    let root = test_root("lazy-pools");
    let database = Database::open(root.clone()).unwrap();
    assert!(pool_dirs(&root).is_empty(), "{:?}", pool_dirs(&root));
    for shard in ShardType::ALL {
        assert!(database
            .pool(shard)
            .get(&[9; 16], &node(1))
            .unwrap()
            .is_none());
    }
    assert!(
        pool_dirs(&root).is_empty(),
        "reads must not create a pool: {:?}",
        pool_dirs(&root)
    );

    database
        .pool(ShardType::Edges)
        .put(&[9; 16], &node(1), &data(b"edge"))
        .unwrap();
    assert_eq!(pool_dirs(&root), vec!["mtpl-edges".to_owned()]);
    database.coordinator().sync().unwrap();
    drop(database);

    let reopened = Database::open(root.clone()).unwrap();
    assert_eq!(pool_dirs(&root), vec!["mtpl-edges".to_owned()]);
    assert_eq!(
        reopened
            .pool(ShardType::Edges)
            .get(&[9; 16], &node(1))
            .unwrap()
            .map(|d| d.bytes.to_vec()),
        Some(b"edge".to_vec())
    );
}

/// A read-only open of a pool that does not exist yet is an empty pool and
/// creates nothing.
#[test]
fn a_read_only_open_of_an_absent_pool_is_empty_and_creates_nothing() {
    let root = test_root("absent-pool-read-only");
    drop(Database::open(root.clone()).unwrap());
    let layout = crate::layout::DatabaseLayout::open_read_only(root.clone()).unwrap();
    for shard in ShardType::ALL {
        let reader = PackfileStorage::open_read_only(layout.pool_path(shard)).unwrap();
        assert!(reader.get(&[9; 16], &node(1)).unwrap().is_none());
    }
    assert!(pool_dirs(&root).is_empty(), "{:?}", pool_dirs(&root));
}

/// A read-committed reader opened while a pool does not exist yet must see the
/// pool's data once the writer creates it: first through the shared WAL, then
/// after the writer's sync has made packs and a checkpoint.
#[test]
#[cfg(feature = "multi-reader")]
fn a_reader_opened_before_a_pool_exists_sees_its_first_writes() {
    let root = test_root("reader-before-pool");
    let database = Database::open(root.clone()).unwrap();
    let pool_dir = database.layout().pool_path(ShardType::State);
    assert!(!pool_dir.exists());
    let reader = PackfileStorage::open_read_committed_shared(
        pool_dir.clone(),
        database.layout().shared_wal_path(),
        ShardType::State,
    )
    .unwrap();
    assert!(!pool_dir.exists(), "a reader must not create the pool");
    let collection = [4u8; 16];
    assert!(reader.get_read_committed(&collection, &[node(1)]).unwrap()[0].is_none());

    database
        .pool(ShardType::State)
        .put(&collection, &node(1), &data(b"first"))
        .unwrap();
    let seen = |label: &str| {
        let got = reader.get_read_committed(&collection, &[node(1)]).unwrap();
        assert_eq!(
            got[0].as_ref().map(|d| d.bytes.to_vec()),
            Some(b"first".to_vec()),
            "{label}"
        );
    };
    seen("after publish, before sync");
    database.coordinator().sync().unwrap();
    database.pool(ShardType::State).sync_all().unwrap();
    seen("after the writer synced and created the pool");
    database
        .pool(ShardType::State)
        .force_index_checkpoint()
        .unwrap();
    seen("after a full checkpoint");
    assert_eq!(
        reader
            .get(&collection, &node(1))
            .unwrap()
            .map(|d| d.bytes.to_vec()),
        Some(b"first".to_vec()),
        "the durable read path must also see the new pool"
    );
}

/// Setting one pool's policy changes that pool and no other, for every pool in
/// `ShardType::ALL`, so a new pool cannot silently share another's slot.
#[test]
fn every_pool_has_its_own_policy_slot() {
    use super::{PoolPolicies, PoolPolicy};
    let baseline = PoolPolicy::default();
    let changed = PoolPolicy {
        compress: !baseline.compress,
        ..baseline
    };
    for shard in ShardType::ALL {
        let policies = PoolPolicies::default().with(shard, changed);
        for other in ShardType::ALL {
            let expected = if other == shard { changed } else { baseline };
            assert_eq!(
                *policies.for_shard(other),
                expected,
                "{shard:?} -> {other:?}"
            );
        }
    }
    assert_eq!(
        *PoolPolicies::uniform(changed).for_shard(ShardType::ALL[0]),
        changed
    );
}

/// The worst case of a power loss: packs cut to their fsynced length and the
/// WAL cut to its durable mark, so every frame written since the last fsync is
/// lost. Whatever a full sync made durable must survive, and nothing
/// half-applied may appear: a transaction is whole or absent.
//
// Like the test above, this validates the database's recovery contract against
// a portable crash image. It does not simulate device-level cache reordering.
#[test]
fn a_worst_case_power_cut_keeps_what_was_synced_and_never_half_applies() {
    let root = test_root("worst_cut_source");
    let image = test_root("worst_cut_image");
    let collection = [0x6f; 16];
    let db = Database::open(root.clone()).unwrap();
    let pools = [ShardType::State, ShardType::EventDag];
    let mut durable: Vec<(ShardType, [u8; 16])> = Vec::new();
    let mut unsynced: Vec<Vec<(ShardType, [u8; 16])>> = Vec::new();
    let mut next = 0u64;
    for round in 0..30u32 {
        // One transaction touching two pools: both writes or neither.
        let txn = db.begin_transaction();
        let mut written = Vec::new();
        for pool in pools {
            let mut node = [0u8; 16];
            node[..8].copy_from_slice(&next.to_le_bytes());
            next += 1;
            txn.put(pool, collection, node, &NodeData::from_slice(&[0x33; 64]))
                .unwrap();
            written.push((pool, node));
        }
        txn.commit().unwrap();
        unsynced.push(written);
        if round % 4 == 0 {
            for pool in pools {
                db.pool(pool).sync_all().unwrap();
            }
            for batch in unsynced.drain(..) {
                durable.extend(batch);
            }
        }

        crash_image(&db, &root, &image);
        crate::journal::cut_segment_image_to_durable_mark(&image.join("wal.bin")).unwrap();
        let recovered = Database::open(image.clone()).unwrap();
        for (pool, node) in &durable {
            assert!(
                recovered
                    .pool(*pool)
                    .get(&collection, node)
                    .unwrap()
                    .is_some(),
                "round {round}: a durably synced record in {pool:?} is gone after a worst-case cut"
            );
        }
        for batch in &unsynced {
            let present: Vec<bool> = batch
                .iter()
                .map(|(pool, node)| {
                    recovered
                        .pool(*pool)
                        .get(&collection, node)
                        .unwrap()
                        .is_some()
                })
                .collect();
            assert!(
                present.iter().all(|p| *p) || present.iter().all(|p| !*p),
                "round {round}: a transaction was half applied: {present:?}"
            );
        }
        drop(recovered);
    }
    drop(db);
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(image);
}

/// A `Database` transaction's `delete_collection` must hide the collection's
/// records from reads in the same process, also when they were already synced
/// to the packs. Today they stay readable until the store is reopened.
#[test]
#[ignore = "engine bug: a committed delete_collection is not visible to live reads once the records were synced; takes effect after reopen"]
fn a_committed_delete_collection_hides_synced_records() {
    let root = test_root("delete_visible_after_sync");
    let db = Database::open(root.clone()).unwrap();
    let doomed = [0x71_u8; 16];
    let put = db.begin_transaction();
    put.put(ShardType::Edges, doomed, node(1), &data(b"doomed"))
        .unwrap();
    put.commit().unwrap();
    db.pool(ShardType::Edges).sync_all().unwrap();

    let delete = db.begin_transaction();
    delete.delete_collection(ShardType::Edges, doomed).unwrap();
    delete.commit().unwrap();

    let read = db.begin_transaction();
    let (records, _) = read
        .get_with_record_versions(ShardType::Edges, &doomed, &[node(1)])
        .unwrap();
    assert!(records[0].is_none(), "the deleted record is still readable");
    drop(read);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// Repacking a collection that holds no records must not bring back the
/// records of a collection deleted before it. Today they reappear.
#[test]
#[ignore = "engine bug: repack_collection_reachable on an empty collection resurrects a deleted collection's records"]
fn repacking_an_empty_collection_resurrects_a_deleted_one() {
    let root = test_root("repack_empty_resurrects");
    let db = Database::open(root.clone()).unwrap();
    let (doomed, empty) = ([0x71_u8; 16], [0x72_u8; 16]);
    let put = db.begin_transaction();
    put.put(ShardType::Edges, doomed, node(1), &data(b"doomed"))
        .unwrap();
    put.commit().unwrap();
    let delete = db.begin_transaction();
    delete.delete_collection(ShardType::Edges, doomed).unwrap();
    delete.commit().unwrap();
    let read = |db: &Database| {
        db.begin_transaction()
            .get_with_record_versions(ShardType::Edges, &doomed, &[node(1)])
            .unwrap()
            .0[0]
            .is_some()
    };
    assert!(!read(&db), "the delete itself must take effect");
    db.pool(ShardType::Edges)
        .repack_collection_reachable(&empty, |_, _| Vec::new())
        .unwrap();
    assert!(!read(&db), "the repack brought the deleted record back");
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
