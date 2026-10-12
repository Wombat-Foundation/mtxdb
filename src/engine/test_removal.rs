//! Versioned record removal: the frame flag, transaction staging, the
//! read-committed overlay, durable reads and scans, WAL replay, re-adding a
//! removed key, and repack with an older shard retained.

use super::{Database, DatabaseTransaction, TransactionOverlayGuard};
use crate::journal::{Mutation, TxnStageState};
use crate::layout::ShardType;
use crate::packfile::storage::PackfileStorage;
use crate::packfile::{self, FrameMetadata, Record, FLAG_METADATA, FLAG_REMOVED};
use crate::storage::{NodeData, NodeId, StorageEngine, StorageError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-removal-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn node(id: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = id;
    bytes
}

fn data(bytes: &'static [u8]) -> NodeData {
    NodeData::new(bytes::Bytes::from_static(bytes))
}

fn payload(value: Option<NodeData>) -> Option<Vec<u8>> {
    value.map(|data| data.bytes.to_vec())
}

fn live(db: &Database, pool: ShardType, collection: [u8; 16], id: NodeId) -> Option<Vec<u8>> {
    payload(db.pool(pool).get(&collection, &id).unwrap())
}

fn committed(db: &Database, pool: ShardType, collection: [u8; 16], id: NodeId) -> Option<Vec<u8>> {
    let read = db
        .pool(pool)
        .get_read_committed(&collection, &[id])
        .unwrap();
    payload(read.into_iter().next().unwrap())
}

fn scanned(db: &Database, pool: ShardType, collection: [u8; 16]) -> HashMap<NodeId, Vec<u8>> {
    db.pool(pool)
        .scan_collection(&collection)
        .unwrap()
        .map(|entry| {
            let (id, data) = entry.unwrap();
            (id, data.bytes.to_vec())
        })
        .collect()
}

fn put_committed(
    db: &Database,
    pool: ShardType,
    collection: [u8; 16],
    id: NodeId,
    v: &'static [u8],
) {
    let txn = db.begin_transaction();
    txn.put(pool, collection, id, &data(v)).unwrap();
    txn.commit().unwrap();
}

fn remove_committed(db: &Database, pool: ShardType, collection: [u8; 16], id: NodeId) {
    let txn = db.begin_transaction();
    txn.remove(pool, collection, id).unwrap();
    txn.commit().unwrap();
}

fn activate_for(db: &Database, txn: &DatabaseTransaction<'_>) -> TransactionOverlayGuard {
    db.activate_transaction_overlay(txn.stage.touched_pools())
        .unwrap()
}

// ---------------------------------------------------------------- codec ---

fn removed_record(metadata: FrameMetadata) -> Record {
    Record {
        collection_id: [0x31; 16],
        hash: node(9),
        data: bytes::Bytes::new(),
        metadata: Some(metadata),
    }
}

/// A removal with no other metadata has no metadata block: the flag alone
/// carries it, and it survives encode and decode.
#[test]
fn a_metadata_less_removed_frame_round_trips_on_the_flag_alone() {
    let record = removed_record(FrameMetadata {
        removed: true,
        ..FrameMetadata::default()
    });
    let frame = packfile::encode_record_with_options(&record, false, true).unwrap();
    assert_ne!(frame[4] & FLAG_REMOVED, 0, "FLAG_REMOVED is written");
    assert_eq!(frame[4] & FLAG_METADATA, 0, "no metadata block is needed");

    let decoded = packfile::read_record(&mut frame.as_slice())
        .unwrap()
        .unwrap();
    assert!(decoded.metadata.is_some_and(|metadata| metadata.removed));
    assert!(decoded.data.is_empty());
}

/// With a metadata block present the flag and the block both survive.
#[test]
fn a_removed_frame_with_an_lsn_keeps_both_the_flag_and_the_metadata() {
    let record = removed_record(FrameMetadata {
        removed: true,
        last_write_lsn: Some(77),
        ..FrameMetadata::default()
    });
    let frame = packfile::encode_record_with_options(&record, false, true).unwrap();
    assert_ne!(frame[4] & FLAG_REMOVED, 0);
    assert_ne!(frame[4] & FLAG_METADATA, 0);
    let metadata = packfile::read_record(&mut frame.as_slice())
        .unwrap()
        .unwrap()
        .metadata
        .unwrap();
    assert!(metadata.removed);
    assert_eq!(metadata.last_write_lsn, Some(77));
}

/// An ordinary frame, including one with an empty payload, is not removed.
#[test]
fn an_empty_payload_alone_is_not_a_removal() {
    let record = Record {
        collection_id: [0x31; 16],
        hash: node(9),
        data: bytes::Bytes::new(),
        metadata: None,
    };
    let frame = packfile::encode_record_with_options(&record, false, true).unwrap();
    assert_eq!(frame[4] & FLAG_REMOVED, 0);
    let decoded = packfile::read_record(&mut frame.as_slice())
        .unwrap()
        .unwrap();
    assert!(decoded.metadata.is_none_or(|metadata| !metadata.removed));
}

/// A reader that does not know the removed bit rejects the frame. The flag
/// byte is CRC-covered, so clearing the bit from the allowed set is modeled
/// by setting a bit no build knows and checking the same rejection path.
#[test]
fn an_unknown_flag_bit_is_rejected_not_misread_as_a_live_value() {
    let record = removed_record(FrameMetadata {
        removed: true,
        ..FrameMetadata::default()
    });
    let mut frame = packfile::encode_record_with_options(&record, false, false).unwrap();
    frame[4] |= 0x80;
    assert!(packfile::read_record(&mut frame.as_slice()).is_err());
}

// ------------------------------------------------------ staging, overlay ---

/// A staged removal hides the live value from its own transaction only; other
/// readers keep the value until commit, then see it gone.
#[test]
fn a_staged_removal_is_private_until_commit() {
    let root = test_root("staged");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x41; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v");
    put_committed(&db, ShardType::State, collection, node(2), b"keep");

    let txn = db.begin_transaction();
    txn.remove(ShardType::State, collection, node(1)).unwrap();
    let own = txn
        .get(ShardType::State, &collection, &[node(1), node(2)])
        .unwrap();
    assert_eq!(
        payload(own[0].clone()),
        None,
        "the transaction sees its removal"
    );
    assert_eq!(payload(own[1].clone()), Some(b"keep".to_vec()));
    assert_eq!(
        live(&db, ShardType::State, collection, node(1)),
        Some(b"v".to_vec()),
        "other readers keep the value until commit"
    );
    txn.commit().unwrap();
    assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
    assert_eq!(committed(&db, ShardType::State, collection, node(1)), None);
    assert_eq!(
        live(&db, ShardType::State, collection, node(2)),
        Some(b"keep".to_vec()),
        "a neighbouring key is untouched"
    );
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Staging order decides: put then remove removes, remove then put keeps.
#[test]
fn the_last_staged_write_to_a_key_wins() {
    let root = test_root("order");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x42; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"old");
    put_committed(&db, ShardType::State, collection, node(2), b"old");

    let txn = db.begin_transaction();
    txn.put(ShardType::State, collection, node(1), &data(b"new"))
        .unwrap();
    txn.remove(ShardType::State, collection, node(1)).unwrap();
    txn.remove(ShardType::State, collection, node(2)).unwrap();
    txn.put(ShardType::State, collection, node(2), &data(b"new"))
        .unwrap();
    txn.commit().unwrap();

    assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
    assert_eq!(
        live(&db, ShardType::State, collection, node(2)),
        Some(b"new".to_vec())
    );
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Removing a key that was never live emits nothing and leaves the collection
/// version alone, so a repeated sweep does not invalidate concurrent readers.
#[test]
fn removing_an_absent_key_does_not_advance_the_collection_version() {
    let root = test_root("absent");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x43; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v");
    let pool = db.pool(ShardType::State);
    let (_, before) = pool
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();

    remove_committed(&db, ShardType::State, collection, node(9));
    // put-then-remove of a key that was never live is also a no-op.
    let txn = db.begin_transaction();
    txn.put(ShardType::State, collection, node(8), &data(b"x"))
        .unwrap();
    txn.remove(ShardType::State, collection, node(8)).unwrap();
    txn.commit().unwrap();

    let (_, after) = pool
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();
    assert_eq!(before, after, "a no-op removal must not bump the version");
    assert_eq!(live(&db, ShardType::State, collection, node(8)), None);
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Removing a live key does bump the version.
#[test]
fn removing_a_live_key_advances_the_collection_version() {
    let root = test_root("live_version");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x44; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v");
    let pool = db.pool(ShardType::State);
    let (_, before) = pool
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();
    remove_committed(&db, ShardType::State, collection, node(1));
    let (_, after) = pool
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();
    assert!(after > before);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A removal staged against a value another writer then replaces is stale:
/// the commit is rejected instead of removing the newer value.
#[test]
fn a_removal_loses_to_a_concurrent_write_of_the_same_key() {
    let root = test_root("stale");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x45; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v1");

    let remover = db.begin_transaction();
    remover
        .remove(ShardType::State, collection, node(1))
        .unwrap();
    put_committed(&db, ShardType::State, collection, node(1), b"v2");

    assert!(matches!(
        remover.commit(),
        Err(StorageError::StaleRead { .. })
    ));
    assert_eq!(
        live(&db, ShardType::State, collection, node(1)),
        Some(b"v2".to_vec()),
        "the newer value must survive"
    );
    drop(remover);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// The overlay: a published-but-unapplied removal already hides the key from
/// read-committed readers, and a later put through the same overlay shows it
/// again.
#[test]
fn the_overlay_hides_a_published_removal_and_shows_a_later_put() {
    let root = test_root("overlay");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x46; 16];
    let other = [0x47; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v");

    let txn = db.begin_transaction();
    txn.remove(ShardType::State, collection, node(1)).unwrap();
    txn.put(ShardType::EventDag, other, node(2), &data(b"root"))
        .unwrap();
    let overlay = activate_for(&db, &txn);
    db.publish_transaction(&txn.stage).unwrap();

    assert_eq!(
        committed(&db, ShardType::State, collection, node(1)),
        None,
        "published removal hides the durable value"
    );
    assert_eq!(
        committed(&db, ShardType::EventDag, other, node(2)),
        Some(b"root".to_vec())
    );
    let counted = db
        .pool(ShardType::State)
        .scan_collection(&collection)
        .unwrap()
        .count();
    assert_eq!(counted, 0, "scans honor the overlay removal too");
    drop(overlay);
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// -------------------------------------------------------- durable reads ---

/// After sync and reopen the removed key is absent from `get` and from scans,
/// and its neighbours remain.
#[test]
fn a_removed_key_stays_absent_across_reopen() {
    let root = test_root("reopen");
    let collection = [0x51; 16];
    {
        let db = Database::open(root.clone()).unwrap();
        put_committed(&db, ShardType::State, collection, node(1), b"gone");
        put_committed(&db, ShardType::State, collection, node(2), b"stay");
        remove_committed(&db, ShardType::State, collection, node(1));
        db.pool(ShardType::State).sync_all().unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
    assert_eq!(
        live(&db, ShardType::State, collection, node(2)),
        Some(b"stay".to_vec())
    );
    let scan = scanned(&db, ShardType::State, collection);
    assert!(!scan.contains_key(&node(1)));
    assert_eq!(scan.get(&node(2)), Some(&b"stay".to_vec()));
    let many = db
        .pool(ShardType::State)
        .get_many(&collection, &[node(1), node(2)])
        .unwrap();
    assert_eq!(payload(many[0].clone()), None);
    assert_eq!(payload(many[1].clone()), Some(b"stay".to_vec()));
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A removed frame with no payload is distinct from a live empty value.
#[test]
fn a_live_empty_value_is_not_a_removal() {
    let root = test_root("empty_live");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x52; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"");
    assert_eq!(
        live(&db, ShardType::State, collection, node(1)),
        Some(Vec::new()),
        "an empty payload is an ordinary live value"
    );
    remove_committed(&db, ShardType::State, collection, node(1));
    assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// ------------------------------------------------------------- re-adding ---

/// The HAMT re-emits unchanged nodes: a put of the same key and content after
/// a removal must succeed and read back, before and after a reopen.
#[test]
fn a_removed_key_can_be_re_added_with_the_same_content() {
    let root = test_root("readd");
    let collection = [0x61; 16];
    {
        let db = Database::open(root.clone()).unwrap();
        put_committed(&db, ShardType::State, collection, node(1), b"same");
        remove_committed(&db, ShardType::State, collection, node(1));
        assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
        put_committed(&db, ShardType::State, collection, node(1), b"same");
        assert_eq!(
            live(&db, ShardType::State, collection, node(1)),
            Some(b"same".to_vec())
        );
        db.pool(ShardType::State).sync_all().unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(
        live(&db, ShardType::State, collection, node(1)),
        Some(b"same".to_vec()),
        "the newer put supersedes the marker after a rebuild"
    );
    assert_eq!(
        scanned(&db, ShardType::State, collection).get(&node(1)),
        Some(&b"same".to_vec())
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A removal does not outlive its collection: after the collection is
/// deleted and rewritten, only the new writes exist.
#[test]
fn a_collection_delete_then_new_puts_does_not_resurrect_a_removed_key() {
    let root = test_root("coll_delete");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x62; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"old");
    put_committed(&db, ShardType::State, collection, node(2), b"old");
    remove_committed(&db, ShardType::State, collection, node(1));

    let txn = db.begin_transaction();
    txn.delete_collection(ShardType::State, collection).unwrap();
    txn.commit().unwrap();
    put_committed(&db, ShardType::State, collection, node(3), b"new");

    assert_eq!(live(&db, ShardType::State, collection, node(1)), None);
    assert_eq!(live(&db, ShardType::State, collection, node(2)), None);
    assert_eq!(
        live(&db, ShardType::State, collection, node(3)),
        Some(b"new".to_vec())
    );
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// ---------------------------------------------------------------- replay ---

/// A transaction that removed a key and was published but only partly applied
/// completes through recovery, and the removal then survives a reopen.
#[test]
fn a_published_removal_is_completed_by_recovery_and_survives_reopen() {
    let root = test_root("replay");
    let state = [0x71; 16];
    let events = [0x72; 16];
    {
        let db = Database::open(root.clone()).unwrap();
        put_committed(&db, ShardType::State, state, node(1), b"v");

        let txn = db.begin_transaction();
        txn.remove(ShardType::State, state, node(1)).unwrap();
        txn.put(ShardType::EventDag, events, node(2), &data(b"root"))
            .unwrap();
        let mut overlay = Some(activate_for(&db, &txn));
        db.publish_transaction(&txn.stage).unwrap();
        txn.stage.begin_materialization().unwrap();
        let receipt = txn.stage.published_receipt().unwrap();
        db.register_new_recovery_stage(Arc::clone(&txn.stage), receipt, &mut overlay)
            .unwrap();
        assert_eq!(txn.state(), TxnStageState::Materializing);
        // Nothing has been applied to the pools: only the overlay knows.
        assert_eq!(committed(&db, ShardType::State, state, node(1)), None);
        assert_eq!(
            committed(&db, ShardType::EventDag, events, node(2)),
            Some(b"root".to_vec())
        );
        drop(txn);
        db.recover_pending_transactions().unwrap();
        assert_eq!(live(&db, ShardType::State, state, node(1)), None);
        assert_eq!(
            live(&db, ShardType::EventDag, events, node(2)),
            Some(b"root".to_vec())
        );
        db.pool(ShardType::State).sync_all().unwrap();
        db.pool(ShardType::EventDag).sync_all().unwrap();
    }
    let db = Database::open(root.clone()).unwrap();
    assert_eq!(live(&db, ShardType::State, state, node(1)), None);
    assert!(!scanned(&db, ShardType::State, state).contains_key(&node(1)));
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// The WAL mutation for a removal round-trips through the journal encoding.
#[test]
fn a_removal_mutation_round_trips_through_the_journal() {
    let root = test_root("wal_codec");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x73; 16];
    put_committed(&db, ShardType::State, collection, node(1), b"v");
    let txn = db.begin_transaction();
    txn.remove(ShardType::State, collection, node(1)).unwrap();
    let staged = txn.stage.snapshot_mutations();
    assert_eq!(
        staged[ShardType::State.index()],
        vec![Mutation::RemoveRecord {
            collection_id: collection,
            node_id: node(1),
        }]
    );
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// ---------------------------------------------------------------- repack ---

fn open_small(dir: &Path) -> PackfileStorage {
    PackfileStorage::open_with_max_shard_bytes(dir.to_path_buf(), 48 * 1024).unwrap()
}

fn filler_id(index: u32) -> NodeId {
    let mut id = [0u8; 16];
    id[..4].copy_from_slice(&index.to_le_bytes());
    id[8..12].copy_from_slice(&index.wrapping_mul(2_654_435_761).to_le_bytes());
    id
}

/// 1 KiB that zstd cannot shrink, so shard growth tracks the record count.
fn incompressible(seed: u32) -> Vec<u8> {
    let mut state = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

fn fill(store: &PackfileStorage, collection: &[u8; 16], start: u32, count: u32) {
    let end = start.checked_add(count).expect("test range fits in u32");
    let entries: Vec<(NodeId, NodeData)> = (start..end)
        .map(|index| {
            (
                filler_id(index),
                NodeData::from_slice(&incompressible(index)),
            )
        })
        .collect();
    store.put_many(collection, &entries).unwrap();
}

fn remove_directly(store: &PackfileStorage, collection: [u8; 16], id: NodeId, lsn: u64) {
    store
        .apply_transaction_mutation(
            &Mutation::RemoveRecord {
                collection_id: collection,
                node_id: id,
            },
            lsn,
        )
        .unwrap();
}

/// Delete the persisted index (checkpoint and delta epochs) so the next open
/// rebuilds from the shards alone.
fn drop_index_sidecars(dir: &Path) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("index.") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn pack_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "pack"))
        .count()
}

fn visible(store: &PackfileStorage, collection: &[u8; 16], id: &NodeId) -> bool {
    store.get(collection, id).unwrap().is_some()
}

fn scan_has(store: &PackfileStorage, collection: &[u8; 16], id: &NodeId) -> bool {
    store
        .scan_collection(collection)
        .unwrap()
        .any(|entry| entry.unwrap().0 == *id)
}

/// Repack over a removal: the key stays absent after the repack, after a
/// reopen on the checkpoint, and after a rebuild with the checkpoint deleted.
#[test]
fn repack_keeps_a_removed_key_absent_across_reopen_and_rebuild() {
    let dir = test_root("repack_basic");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x81; 16];
    let victim = node(200);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 20);
        store.sync_all().unwrap();
        assert!(pack_count(&dir) > 1, "the workload must span several packs");
        assert!(!visible(&store, &collection, &victim));

        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        assert!(
            !visible(&store, &collection, &victim),
            "absent after repack"
        );
        assert!(!scan_has(&store, &collection, &victim));
        store.sync_all().unwrap();
    }
    {
        let store = open_small(&dir);
        assert!(
            !visible(&store, &collection, &victim),
            "absent after reopen"
        );
        assert!(!scan_has(&store, &collection, &victim));
    }
    // Force a rebuild from the shards themselves.
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert!(
        !visible(&store, &collection, &victim),
        "absent after a rebuild from shards"
    );
    assert!(!scan_has(&store, &collection, &victim));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// The resurrection case. Another collection keeps the oldest shard alive, so
/// repack of the first collection retires the shard holding the marker but
/// leaves the shard holding the older value. If repack drops the marker
/// without removing the older bytes from the live set, a rebuild from shards
/// finds only the older value and the key comes back. This must stay absent.
#[test]
fn repack_with_a_retained_older_shard_does_not_resurrect_a_removed_key() {
    let dir = test_root("repack_retained");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x82; 16];
    let bystander = [0x83; 16];
    let victim = node(201);
    {
        let store = open_small(&dir);
        // The oldest shard holds the victim's value and a bystander record
        // that keeps that shard referenced after the repack.
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        store
            .put(&bystander, &node(1), &NodeData::from_slice(b"keep"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 120);
        store.sync_all().unwrap();
        assert!(
            pack_count(&dir) > 2,
            "need the old value and marker in different packs"
        );

        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        assert!(
            !visible(&store, &collection, &victim),
            "absent after repack"
        );
        assert!(
            visible(&store, &bystander, &node(1)),
            "the bystander keeps the old shard alive"
        );
        store.sync_all().unwrap();
    }
    // Rebuild from the shards, as after losing the checkpoint.
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert!(
        !visible(&store, &collection, &victim),
        "a retained older shard must not bring the removed key back"
    );
    assert!(!scan_has(&store, &collection, &victim));
    assert!(visible(&store, &bystander, &node(1)));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// Control for the resurrection test: the same layout with no repack. A rebuild
/// from the shards sees the marker after the older value, so the key stays
/// absent. If this fails the problem is the rebuild, not the repack.
#[test]
fn a_rebuild_without_repack_keeps_a_removed_key_absent_with_an_older_shard() {
    let dir = test_root("rebuild_control");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x85; 16];
    let victim = node(203);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 120);
        store.sync_all().unwrap();
        assert!(pack_count(&dir) > 2);
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert!(!visible(&store, &collection, &victim));
    assert!(!scan_has(&store, &collection, &victim));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// A key removed and then re-added before repack keeps its newer value
/// through repack and a rebuild.
#[test]
fn repack_keeps_a_value_that_was_re_added_after_removal() {
    let dir = test_root("repack_readd");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x84; 16];
    let victim = node(202);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        fill(&store, &collection, 0, 60);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 60);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"new"))
            .unwrap();
        store.sync_all().unwrap();
        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        store.sync_all().unwrap();
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert_eq!(
        payload(store.get(&collection, &victim).unwrap()),
        Some(b"new".to_vec())
    );
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// A carried marker must keep suppressing the older value across further
/// repacks, not just the first one, while the older shard is still retained.
#[test]
fn a_carried_marker_survives_a_second_repack_and_a_rebuild() {
    let dir = test_root("repack_twice");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x86; 16];
    let bystander = [0x87; 16];
    let victim = node(204);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        store
            .put(&bystander, &node(1), &NodeData::from_slice(b"keep"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 120);
        store.sync_all().unwrap();
        for round in 0..3u32 {
            store
                .repack_collection_reachable(&collection, |_, _| Vec::new())
                .unwrap();
            assert!(
                !visible(&store, &collection, &victim),
                "absent after repack {round}"
            );
            // New writes between repacks move shard layout around.
            fill(&store, &collection, 5000 + round * 10, 10);
            store.sync_all().unwrap();
        }
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert!(!visible(&store, &collection, &victim));
    assert!(!scan_has(&store, &collection, &victim));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// Re-adding a removed key after its marker was carried must win over the
/// carried marker, including after a rebuild from shards.
#[test]
fn a_key_re_added_after_a_carried_marker_stays_live() {
    let dir = test_root("repack_readd_carried");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x88; 16];
    let bystander = [0x89; 16];
    let victim = node(205);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        store
            .put(&bystander, &node(1), &NodeData::from_slice(b"keep"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 120);
        store.sync_all().unwrap();
        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        store
            .put(&collection, &victim, &NodeData::from_slice(b"new"))
            .unwrap();
        store.sync_all().unwrap();
        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        store.sync_all().unwrap();
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert_eq!(
        payload(store.get(&collection, &victim).unwrap()),
        Some(b"new".to_vec())
    );
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// Repack must be invisible to version tokens: a record version read before
/// the repack still matches after it, for a live key and a removed one.
#[test]
fn repack_does_not_change_record_version_tokens() {
    let root = test_root("repack_tokens");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x8a; 16];
    let bystander = [0x8b; 16];
    put_committed(&db, ShardType::State, bystander, node(1), b"keep");
    put_committed(&db, ShardType::State, collection, node(1), b"live");
    put_committed(&db, ShardType::State, collection, node(2), b"gone");
    remove_committed(&db, ShardType::State, collection, node(2));
    let pool = db.pool(ShardType::State);
    let tokens = |pool: &PackfileStorage| {
        pool.get_with_record_versions(&collection, &[node(1), node(2)])
            .unwrap()
            .1
    };
    let before = tokens(pool);
    assert!(before.iter().all(|version| *version > 0));
    pool.sync_all().unwrap();
    pool.repack_collection_reachable(&collection, |_, _| Vec::new())
        .unwrap();
    pool.sync_all().unwrap();
    assert_eq!(before, tokens(pool), "repack must not move version tokens");
    assert_eq!(live(&db, ShardType::State, collection, node(2)), None);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// The batch repack path must carry markers like the single-collection one:
/// a collection repacked alone while a bystander keeps the older shard alive.
#[test]
fn batch_repack_with_a_retained_older_shard_does_not_resurrect_a_removed_key() {
    let dir = test_root("batch_retained");
    std::fs::create_dir_all(&dir).unwrap();
    let collection = [0x8c; 16];
    let bystander = [0x8d; 16];
    let victim = node(206);
    {
        let store = open_small(&dir);
        store
            .put(&collection, &victim, &NodeData::from_slice(b"old"))
            .unwrap();
        store
            .put(&bystander, &node(1), &NodeData::from_slice(b"keep"))
            .unwrap();
        fill(&store, &collection, 0, 120);
        remove_directly(&store, collection, victim, 1);
        fill(&store, &collection, 1000, 120);
        store.sync_all().unwrap();
        for _ in 0..2 {
            store
                .repack_collections_reachable(&[collection], |_, _| Vec::new())
                .unwrap();
            assert!(!visible(&store, &collection, &victim));
            assert!(visible(&store, &bystander, &node(1)));
            store.sync_all().unwrap();
        }
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    assert!(
        !visible(&store, &collection, &victim),
        "batch repack must not let a retained older shard bring the key back"
    );
    assert!(!scan_has(&store, &collection, &victim));
    assert!(visible(&store, &bystander, &node(1)));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// A batch that repacks several collections at once keeps each one's removal.
#[test]
fn batch_repack_of_several_collections_keeps_each_removal() {
    let dir = test_root("batch_several");
    std::fs::create_dir_all(&dir).unwrap();
    let first = [0x8e; 16];
    let second = [0x8f; 16];
    let bystander = [0x90; 16];
    let victim = node(207);
    {
        let store = open_small(&dir);
        for collection in [first, second] {
            store
                .put(&collection, &victim, &NodeData::from_slice(b"old"))
                .unwrap();
        }
        store
            .put(&bystander, &node(1), &NodeData::from_slice(b"keep"))
            .unwrap();
        fill(&store, &first, 0, 60);
        fill(&store, &second, 0, 60);
        remove_directly(&store, first, victim, 1);
        remove_directly(&store, second, victim, 1);
        fill(&store, &first, 1000, 60);
        fill(&store, &second, 1000, 60);
        store.sync_all().unwrap();
        store
            .repack_collections_reachable(&[first, second], |_, _| Vec::new())
            .unwrap();
        store.sync_all().unwrap();
    }
    drop_index_sidecars(&dir);
    let store = open_small(&dir);
    for collection in [first, second] {
        assert!(!visible(&store, &collection, &victim));
        assert!(!scan_has(&store, &collection, &victim));
    }
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

// ------------------------------------------------------------ change feed ---

/// A committed removal reaches replay consumers as its own `RemoveRecord`
/// mutation, so a replica does not mistake it for a missing record. A removal
/// of an absent key emits nothing, and a snapshot scan taken before the
/// removal still holds the key: snapshot plus feed reproduce the live state.
#[test]
fn the_change_feed_carries_a_removal_as_its_own_mutation() {
    use crate::journal::Mutation as JournalMutation;
    let root = test_root("change_feed");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x91; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"v");
    put_committed(&db, pool, collection, node(2), b"keep");

    let (scan, cursor, lease) = db
        .pool(pool)
        .scan_collection_at_snapshot(&collection)
        .unwrap();

    let txn = db.begin_transaction();
    txn.remove(pool, collection, node(1)).unwrap();
    txn.remove(pool, collection, node(77)).unwrap(); // never existed: no mutation
    txn.commit().unwrap();

    let snapshot: Vec<NodeId> = scan.map(|entry| entry.unwrap().0).collect();
    assert!(
        snapshot.contains(&node(1)),
        "the snapshot taken before the removal still holds the key"
    );

    db.coordinator().sync().unwrap();
    let page = db.coordinator().changes_since(&cursor, 16).unwrap();
    let mutations: Vec<&JournalMutation> = page
        .groups
        .iter()
        .flat_map(|group| group.entries.iter())
        .map(|entry| &entry.mutation)
        .collect();
    assert_eq!(
        mutations,
        vec![&JournalMutation::RemoveRecord {
            collection_id: collection,
            node_id: node(1),
        }],
        "exactly one RemoveRecord, none for the absent key"
    );
    assert!(!page.has_more);
    drop(lease);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A snapshot scan taken before a removal keeps yielding the key's older
/// value even after repack retires the shards and drops the marker: the scan
/// pins its shards, so the repack cannot pull the bytes out from under it.
#[test]
fn a_snapshot_scan_survives_a_repack_that_drops_the_marker() {
    let root = test_root("snapshot_repack");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x92; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"v");
    put_committed(&db, pool, collection, node(2), b"keep");
    db.pool(pool).sync_all().unwrap();

    let (scan, _cursor, lease) = db
        .pool(pool)
        .scan_collection_at_snapshot(&collection)
        .unwrap();

    remove_committed(&db, pool, collection, node(1));
    db.pool(pool).sync_all().unwrap();
    db.pool(pool)
        .repack_collection_reachable(&collection, |_, _| Vec::new())
        .unwrap();
    db.pool(pool).sync_all().unwrap();

    let during: HashMap<NodeId, Vec<u8>> = scan
        .map(|entry| {
            let (id, data) = entry.expect("a pinned snapshot must not lose its bytes");
            (id, data.bytes.to_vec())
        })
        .collect();
    assert_eq!(during.get(&node(1)), Some(&b"v".to_vec()));
    assert_eq!(during.get(&node(2)), Some(&b"keep".to_vec()));
    drop(lease);
    assert_eq!(
        live(&db, pool, collection, node(1)),
        None,
        "the live view stays removed"
    );
    assert_eq!(
        scanned(&db, pool, collection).get(&node(1)),
        None,
        "a scan taken after the repack does not see the key"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Removals page like any other mutation: a page size of one splits two
/// removal groups across pages, `has_more` is true until the last, and the
/// next cursor resumes with the second removal.
#[test]
fn removals_page_across_a_page_boundary_in_order() {
    use crate::journal::Mutation as JournalMutation;
    let root = test_root("change_feed_paging");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0x93; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"a");
    put_committed(&db, pool, collection, node(2), b"b");

    let (scan, cursor, lease) = db
        .pool(pool)
        .scan_collection_at_snapshot(&collection)
        .unwrap();
    drop(scan);
    remove_committed(&db, pool, collection, node(1));
    remove_committed(&db, pool, collection, node(2));
    db.coordinator().sync().unwrap();

    let removed_ids = |page: &crate::journal::JournalChangesPage| -> Vec<NodeId> {
        page.groups
            .iter()
            .flat_map(|group| group.entries.iter())
            .filter_map(|entry| match &entry.mutation {
                JournalMutation::RemoveRecord { node_id, .. } => Some(*node_id),
                _ => None,
            })
            .collect()
    };
    let first = db.coordinator().changes_since(&cursor, 1).unwrap();
    assert_eq!(removed_ids(&first), vec![node(1)]);
    assert!(first.has_more, "a second removal group follows");

    let second = db
        .coordinator()
        .changes_since(&first.next_cursor, 1)
        .unwrap();
    assert_eq!(removed_ids(&second), vec![node(2)]);
    assert!(!second.has_more);
    drop(lease);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// ----------------------------------------------------------- batch removal ---

fn limited_txn(db: &Database, limit_bytes: usize) -> DatabaseTransaction<'_> {
    DatabaseTransaction {
        database: super::DatabaseRef::Borrowed(db),
        stage: Arc::new(crate::journal::TxnStage::with_limit(limit_bytes)),
        lifecycle: parking_lot::Mutex::new(()),
    }
}

/// A batch removes the live keys, ignores absent and duplicate ones, and
/// leaves every other key and the version of untouched collections alone.
#[test]
fn remove_many_removes_live_keys_and_ignores_absent_and_duplicate_ones() {
    let root = test_root("remove_many_basic");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa1; 16];
    let pool = ShardType::State;
    for id in 1..=5u8 {
        put_committed(&db, pool, collection, node(id), b"v");
    }

    let txn = db.begin_transaction();
    txn.remove_many(
        pool,
        collection,
        &[node(1), node(2), node(2), node(50), node(4)],
    )
    .unwrap();
    txn.commit().unwrap();

    for gone in [1u8, 2, 4] {
        assert_eq!(live(&db, pool, collection, node(gone)), None);
    }
    for kept in [3u8, 5] {
        assert_eq!(live(&db, pool, collection, node(kept)), Some(b"v".to_vec()));
    }
    assert_eq!(live(&db, pool, collection, node(50)), None);
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A batch of only absent keys writes nothing and keeps the version, exactly
/// like repeated single removes.
#[test]
fn remove_many_of_only_absent_keys_does_not_advance_the_version() {
    let root = test_root("remove_many_absent");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa2; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"v");
    let (_, before) = db
        .pool(pool)
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();

    let txn = db.begin_transaction();
    txn.remove_many(pool, collection, &[node(8), node(9)])
        .unwrap();
    txn.commit().unwrap();

    let (_, after) = db
        .pool(pool)
        .get_with_collection_version(&collection, &[node(1)])
        .unwrap();
    assert_eq!(before, after);
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Staged puts are normalized per key, as with single removes: put-then-remove
/// of a never-live key is a no-op, of a live key a removal.
#[test]
fn remove_many_normalizes_earlier_staged_puts() {
    let root = test_root("remove_many_staged");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa3; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"old");

    let txn = db.begin_transaction();
    txn.put(pool, collection, node(1), &data(b"new")).unwrap(); // live key
    txn.put(pool, collection, node(2), &data(b"new")).unwrap(); // never live
    txn.remove_many(pool, collection, &[node(1), node(2)])
        .unwrap();
    txn.commit().unwrap();

    assert_eq!(live(&db, pool, collection, node(1)), None);
    assert_eq!(live(&db, pool, collection, node(2)), None);
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// The whole batch is resolved with one staged-mutation scan, not one per key.
#[test]
fn remove_many_scans_the_stage_once_for_the_whole_batch() {
    let root = test_root("remove_many_scan_once");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa4; 16];
    let pool = ShardType::State;
    let ids: Vec<NodeId> = (0..=255u8).map(node).collect();
    for id in &ids {
        put_committed(&db, pool, collection, *id, b"v");
    }
    let txn = db.begin_transaction();
    let before = txn.stage.lookup_many_calls();
    txn.remove_many(pool, collection, &ids).unwrap();
    assert_eq!(txn.stage.lookup_many_calls() - before, 1);
    txn.commit().unwrap();
    for id in &ids {
        assert_eq!(live(&db, pool, collection, *id), None);
    }
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Normalizing a staged put releases exactly the budget it held, so repeating
/// put-then-remove does not exhaust the stage.
#[test]
fn remove_many_releases_the_budget_of_the_puts_it_forgets() {
    let root = test_root("remove_many_budget");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa5; 16];
    let pool = ShardType::State;
    let txn = limited_txn(&db, 1000);
    for _ in 0..200 {
        txn.put(pool, collection, node(1), &data(b"x")).unwrap();
        txn.remove_many(pool, collection, &[node(1)]).unwrap();
    }
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// A batch that does not fit the budget stages nothing, and a smaller batch
/// still fits afterwards.
#[test]
fn an_oversized_remove_many_stages_nothing() {
    let root = test_root("remove_many_oversized");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa6; 16];
    let pool = ShardType::State;
    let ids: Vec<NodeId> = (1..=5u8).map(node).collect();
    for id in &ids {
        put_committed(&db, pool, collection, *id, b"v");
    }
    // Room for three removals (64 bytes each), not five.
    let txn = limited_txn(&db, 64 * 3);
    assert!(txn.remove_many(pool, collection, &ids).is_err());
    let seen = txn.get(pool, &collection, &ids).unwrap();
    assert!(
        seen.iter().all(Option::is_some),
        "a rejected batch must not hide any key from its own transaction"
    );
    txn.remove_many(pool, collection, &ids[..3]).unwrap();
    txn.commit().unwrap();
    for id in &ids[..3] {
        assert_eq!(live(&db, pool, collection, *id), None);
    }
    for id in &ids[3..] {
        assert_eq!(live(&db, pool, collection, *id), Some(b"v".to_vec()));
    }
    drop(txn);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Removing many keys loses to a concurrent write of any one of them.
#[test]
fn remove_many_is_stale_if_any_key_changed_concurrently() {
    let root = test_root("remove_many_stale");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa7; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"v1");
    put_committed(&db, pool, collection, node(2), b"v1");

    let remover = db.begin_transaction();
    remover
        .remove_many(pool, collection, &[node(1), node(2)])
        .unwrap();
    put_committed(&db, pool, collection, node(2), b"v2");

    assert!(matches!(
        remover.commit(),
        Err(StorageError::StaleRead { .. })
    ));
    assert_eq!(live(&db, pool, collection, node(1)), Some(b"v1".to_vec()));
    assert_eq!(live(&db, pool, collection, node(2)), Some(b"v2".to_vec()));
    drop(remover);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

// ------------------------------------------------------ removal statistics ---

fn fresh_store(name: &str) -> (PathBuf, PackfileStorage) {
    let dir = test_root(name);
    std::fs::create_dir_all(&dir).unwrap();
    let store = open_small(&dir);
    (dir, store)
}

#[test]
fn removal_stats_count_live_and_removed_keys_and_reclaimable_bytes() {
    let (dir, store) = fresh_store("stats_basic");
    let collection = [0xb1; 16];
    for id in 1..=3u8 {
        store
            .put(&collection, &node(id), &NodeData::from_slice(b"value"))
            .unwrap();
    }
    let before = store.removal_stats(&collection).unwrap();
    assert_eq!(before.live_keys, 3);
    assert_eq!(before.removed_keys, 0);
    assert_eq!(before.superseded_frames, 0);
    assert_eq!(before.reclaimable_bytes, 0);
    assert_eq!(before.frames, 3);

    assert!(store.remove_record(&collection, &node(2)).unwrap());
    let after = store.removal_stats(&collection).unwrap();
    assert_eq!(after.live_keys, 2);
    assert_eq!(after.removed_keys, 1);
    assert_eq!(after.marker_frames, 1);
    assert_eq!(
        after.superseded_frames, 1,
        "the removed value is superseded"
    );
    assert_eq!(after.frames, 4);
    assert!(after.reclaimable_bytes > 0);
    assert!(after.reclaimable_bytes < after.total_bytes);
    assert_eq!(store.removed_keys(&collection).unwrap(), vec![node(2)]);

    // Removing a key that is no longer live writes nothing.
    assert!(!store.remove_record(&collection, &node(2)).unwrap());
    assert!(!store.remove_record(&collection, &node(99)).unwrap());
    assert_eq!(store.removal_stats(&collection).unwrap(), after);
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_re_added_key_is_live_again_and_its_old_versions_are_superseded() {
    let (dir, store) = fresh_store("stats_readd");
    let collection = [0xb2; 16];
    store
        .put(&collection, &node(1), &NodeData::from_slice(b"one"))
        .unwrap();
    assert!(store.remove_record(&collection, &node(1)).unwrap());
    store
        .put(&collection, &node(1), &NodeData::from_slice(b"two"))
        .unwrap();
    let stats = store.removal_stats(&collection).unwrap();
    assert_eq!(stats.live_keys, 1);
    assert_eq!(stats.removed_keys, 0);
    assert_eq!(stats.superseded_frames, 2, "the first value and the marker");
    assert!(store.removed_keys(&collection).unwrap().is_empty());
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// Repack gives the space back: the frame count and the total fall, and a
/// further repack drops the marker too once no older copy remains.
#[test]
fn repack_reduces_what_removal_stats_report() {
    let (dir, store) = fresh_store("stats_repack");
    let collection = [0xb3; 16];
    fill(&store, &collection, 0, 60);
    let victim = node(210);
    store
        .put(&collection, &victim, &NodeData::from_slice(b"old"))
        .unwrap();
    assert!(store.remove_record(&collection, &victim).unwrap());
    store.sync_all().unwrap();
    let before = store.removal_stats(&collection).unwrap();
    assert_eq!(before.removed_keys, 1);

    for _ in 0..2 {
        store
            .repack_collection_reachable(&collection, |_, _| Vec::new())
            .unwrap();
        store.sync_all().unwrap();
    }
    let after = store.removal_stats(&collection).unwrap();
    assert_eq!(after.live_keys, before.live_keys);
    assert_eq!(
        after.removed_keys, 0,
        "the marker is gone once nothing older remains"
    );
    assert_eq!(after.superseded_frames, 0);
    assert!(after.total_bytes < before.total_bytes);
    assert!(!visible(&store, &collection, &victim));
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// With a journal enabled the standalone removal is published, so the change
/// feed carries it like a transactional one.
#[test]
fn remove_record_publishes_to_an_enabled_journal() {
    use crate::journal::Mutation as JournalMutation;
    let dir = test_root("remove_record_journal");
    std::fs::create_dir_all(&dir).unwrap();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.enable_journal(dir.join("wal.bin")).unwrap();
    let collection = [0xb4; 16];
    store
        .put(&collection, &node(1), &NodeData::from_slice(b"v"))
        .unwrap();
    let (scan, cursor, lease) = store.scan_collection_at_snapshot(&collection).unwrap();
    drop(scan);

    assert!(store.remove_record(&collection, &node(1)).unwrap());
    let journal = store.journal().unwrap();
    journal.sync().unwrap();
    let page = journal.changes_since(&cursor, 8).unwrap();
    let mutations: Vec<&JournalMutation> = page
        .groups
        .iter()
        .flat_map(|group| group.entries.iter())
        .map(|entry| &entry.mutation)
        .collect();
    assert_eq!(
        mutations,
        vec![&JournalMutation::RemoveRecord {
            collection_id: collection,
            node_id: node(1),
        }]
    );
    assert!(!visible(&store, &collection, &node(1)));
    drop(lease);
    drop(store);
    let _ = std::fs::remove_dir_all(dir);
}

/// The HAMT re-emits an unchanged node under a new root: a put of the same key
/// with identical bytes. A sweep that staged the node's removal before that
/// put must go stale rather than delete a node a new root now references.
#[test]
fn a_removal_loses_to_a_concurrent_re_put_of_identical_content() {
    let root = test_root("stale_identical");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa8; 16];
    let pool = ShardType::State;
    put_committed(&db, pool, collection, node(1), b"same");

    let sweep = db.begin_transaction();
    sweep.remove(pool, collection, node(1)).unwrap();
    put_committed(&db, pool, collection, node(1), b"same");

    assert!(matches!(
        sweep.commit(),
        Err(StorageError::StaleRead { .. })
    ));
    assert_eq!(
        live(&db, pool, collection, node(1)),
        Some(b"same".to_vec()),
        "the republished node must survive the stale sweep"
    );
    drop(sweep);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

/// Same, through the batch path a sweep actually uses.
#[test]
fn a_batch_removal_loses_to_a_concurrent_re_put_of_identical_content() {
    let root = test_root("stale_identical_batch");
    let db = Database::open(root.clone()).unwrap();
    let collection = [0xa9; 16];
    let pool = ShardType::State;
    for id in 1..=3u8 {
        put_committed(&db, pool, collection, node(id), b"same");
    }

    let sweep = db.begin_transaction();
    sweep
        .remove_many(pool, collection, &[node(1), node(2), node(3)])
        .unwrap();
    put_committed(&db, pool, collection, node(2), b"same");

    assert!(matches!(
        sweep.commit(),
        Err(StorageError::StaleRead { .. })
    ));
    for id in 1..=3u8 {
        assert_eq!(
            live(&db, pool, collection, node(id)),
            Some(b"same".to_vec())
        );
    }
    drop(sweep);
    drop(db);
    let _ = std::fs::remove_dir_all(root);
}
