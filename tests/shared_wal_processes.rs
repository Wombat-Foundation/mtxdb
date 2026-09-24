//! Two real OS processes, one shared WAL.
//!
//! The parent test is the writer: it opens a [`SharedDatabase`] (which holds
//! the root writer lock for the process's lifetime) and commits a record to
//! the shared segment. It then re-executes this test binary as a *separate*
//! process whose only job is to open the same root read-only and read the
//! committed record through the shared read-committed overlay.
//!
//! This is the cross-process half of the shared durability fence: the reader
//! takes no writer lock, never touches the writer's handle, and must observe
//! a committed record the writer produced in another process.

#![cfg(feature = "multi-reader")]
// Integration tests are test code by construction; this file has no non-test
// items to wrap in a `#[cfg(test)]` module.
#![allow(clippy::tests_outside_test_module)]

use std::path::PathBuf;

use mtxdb::{NodeData, PackfileStorage, ShardType, SharedDatabase, StorageEngine};

const COLLECTION: [u8; 16] = [0x5a; 16];
const NODE: [u8; 16] = [0x33; 16];
const PAYLOAD: &[u8] = b"committed by the writer process";
const READER_ENV: &str = "MTXDB_SHARED_READER_ROOT";

/// The reader-process entry point. Runs only when re-executed with
/// [`READER_ENV`] set; a normal test run (no env) returns immediately.
#[test]
fn reader_process_enters_and_reads() {
    let Ok(root) = std::env::var(READER_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    let pool_dir = root.join("pools").join(ShardType::State.as_str());
    let wal_path = root.join("wal.bin");

    let reader = PackfileStorage::open_read_committed_shared(pool_dir, wal_path, ShardType::State)
        .expect("reader process must open the shared WAL read-only");
    let got = reader
        .get_read_committed(&COLLECTION, &[NODE])
        .expect("read-committed lookup must succeed");
    let got = got[0]
        .as_ref()
        .expect("the writer's committed record must be visible to another process");
    assert_eq!(&got.bytes[..], PAYLOAD);
}

#[test]
fn a_reader_process_sees_a_live_writers_committed_record() {
    let root = std::env::temp_dir().join(format!("mtxdb-shared-wal-procs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let db = SharedDatabase::open(root.clone()).expect("writer opens the shared root");
    let state = db.pool(ShardType::State);
    state
        .put(
            &COLLECTION,
            &NODE,
            &NodeData::new(bytes::Bytes::from_static(PAYLOAD)),
        )
        .expect("writer commits a record");
    state.sync_all().expect("writer makes it durable");

    let exe = std::env::current_exe().expect("current test executable");
    let status = std::process::Command::new(exe)
        .args([
            "--exact",
            "reader_process_enters_and_reads",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(READER_ENV, &root)
        .status()
        .expect("spawn the reader process");
    assert!(status.success(), "the reader process must succeed");

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
