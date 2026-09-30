use super::logical_head::*;
use crate::database::SharedDatabase;
use crate::layout::ShardType;
use crate::storage::{NodeData, StorageError};
use std::path::PathBuf;

const POOL: ShardType = ShardType::State;
const HEADS: [u8; 16] = [0x48; 16];

fn test_root(name: &str) -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("mtxdb-logical-head-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn set(db: &SharedDatabase, id: &[u8; 16], target: u8, meta: &'static [u8]) {
    let heads = LogicalHead::new(POOL, HEADS);
    let txn = db.begin_transaction();
    let read = heads.read(&txn, id).unwrap();
    heads
        .stage_replace(
            &txn,
            id,
            read.token,
            &LogicalHeadValue::new([target; 16], meta),
        )
        .unwrap();
    txn.commit().unwrap();
}

fn current(db: &SharedDatabase, id: &[u8; 16]) -> Option<LogicalHeadValue> {
    let txn = db.begin_transaction();
    LogicalHead::new(POOL, HEADS).read(&txn, id).unwrap().value
}

#[test]
fn encoding_roundtrips_and_rejects_malformed() {
    let value = LogicalHeadValue::new([2u8; 16], b"meta".as_slice());
    let bytes = encode_logical_head(&value).unwrap();
    assert_eq!(decode_logical_head(&bytes).unwrap(), value);
    assert!(decode_logical_head(b"LHP1").is_err());
    let mut truncated = bytes.to_vec();
    truncated.pop();
    assert!(decode_logical_head(&truncated).is_err());
}

#[test]
fn create_then_update_replaces_head() {
    let root = test_root("update");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let id = [1u8; 16];
    assert_eq!(current(&db, &id), None);
    set(&db, &id, 2, b"first");
    assert_eq!(
        current(&db, &id),
        Some(LogicalHeadValue::new([2u8; 16], b"first".as_slice()))
    );
    set(&db, &id, 3, b"second");
    assert_eq!(
        current(&db, &id),
        Some(LogicalHeadValue::new([3u8; 16], b"second".as_slice()))
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(feature = "multi-reader")]
#[test]
fn stale_writer_is_rejected_by_engine_cas() {
    let root = test_root("stale");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let heads = LogicalHead::new(POOL, HEADS);
    let id = [4u8; 16];
    set(&db, &id, 5, b"a");

    let slow = db.begin_transaction();
    let stale = heads.read(&slow, &id).unwrap();
    set(&db, &id, 6, b"b");

    heads
        .stage_replace(
            &slow,
            &id,
            stale.token,
            &LogicalHeadValue::new([7u8; 16], b"c".as_slice()),
        )
        .unwrap();
    let error = slow.commit().unwrap_err();
    assert!(matches!(error, StorageError::StaleRead { .. }), "{error}");
    assert_eq!(
        current(&db, &id),
        Some(LogicalHeadValue::new([6u8; 16], b"b".as_slice()))
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// A head swap commits atomically with other records in the same
/// transaction, and a stale head rolls the companion write back too.
#[cfg(feature = "multi-reader")]
#[test]
fn head_swap_is_atomic_with_companion_write() {
    let root = test_root("atomic");
    let db = SharedDatabase::open(root.clone()).unwrap();
    let heads = LogicalHead::new(POOL, HEADS);
    let id = [8u8; 16];
    let other = [0x49u8; 16];
    let companion_id = [9u8; 16];
    set(&db, &id, 1, b"v1");

    let slow = db.begin_transaction();
    let read = heads.read(&slow, &id).unwrap();
    set(&db, &id, 2, b"v2");
    slow.put(
        POOL,
        other,
        companion_id,
        &NodeData::new(bytes::Bytes::from_static(b"companion")),
    )
    .unwrap();
    heads
        .stage_replace(
            &slow,
            &id,
            read.token,
            &LogicalHeadValue::new([3u8; 16], b"v3".as_slice()),
        )
        .unwrap();
    assert!(slow.commit().unwrap_err().is_stale_read());

    let check = db.begin_transaction();
    let (records, _) = check
        .get_with_record_versions(POOL, &other, &[companion_id])
        .unwrap();
    assert!(
        records[0].is_none(),
        "companion write must not be published"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reopen_reconstructs_head() {
    let root = test_root("reopen");
    let id = [0xA1u8; 16];
    {
        let db = SharedDatabase::open(root.clone()).unwrap();
        set(&db, &id, 1, b"one");
        set(&db, &id, 2, b"two");
    }
    let db = SharedDatabase::open(root.clone()).unwrap();
    assert_eq!(
        current(&db, &id),
        Some(LogicalHeadValue::new([2u8; 16], b"two".as_slice()))
    );
    // A token read after reopen is seeded from the frame and still guards a
    // replace.
    set(&db, &id, 3, b"three");
    assert_eq!(
        current(&db, &id),
        Some(LogicalHeadValue::new([3u8; 16], b"three".as_slice()))
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
