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

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::{NodeData, PackfileStorage, ShardType, SharedDatabase, StorageEngine};

const COLLECTION: [u8; 16] = [0x5a; 16];
const NODE: [u8; 16] = [0x33; 16];
const PAYLOAD: &[u8] = b"committed by the writer process";
const READER_ENV: &str = "MTXDB_SHARED_READER_ROOT";
const STALE_READER_ENV: &str = "MTXDB_SHARED_STALE_READER_ROOT";

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

/// A reader that must already be open, with a stale index, before the writer
/// publishes. It establishes its durable baseline with a miss, signals the
/// writer, then re-reads the same key after the writer says it has published.
///
/// This models a long-lived read-only worker: a freshly opened reader can
/// discover an eagerly written frame during its open-time rescan, so only a
/// reader whose index predates the write exercises the read-committed overlay.
#[test]
fn stale_reader_process_baselines_then_rereads_after_publish() {
    let Ok(root) = std::env::var(STALE_READER_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    let pool_dir = root.join("pools").join(ShardType::State.as_str());
    let wal_path = root.join("wal.bin");
    let baseline_done = root.join("baseline.done");
    let published = root.join("published.done");
    let result = root.join("reader.result");

    let reader = PackfileStorage::open_read_committed_shared(pool_dir, wal_path, ShardType::State)
        .expect("reader process must open the shared WAL read-only");
    // Establish the durable baseline before the writer writes. The key is
    // absent, and the reader keeps this index snapshot (and its last-seen
    // durable fingerprint) for the rest of the test.
    let before = reader
        .get_read_committed(&COLLECTION, &[NODE])
        .expect("baseline lookup must succeed");
    assert!(
        before[0].is_none(),
        "baseline read must miss before the writer writes"
    );
    std::fs::write(&baseline_done, b"1").expect("signal baseline ready");

    wait_for(&published, Duration::from_secs(30));
    let after = reader
        .get_read_committed(&COLLECTION, &[NODE])
        .expect("re-read must succeed");
    let saw_it = after[0]
        .as_ref()
        .is_some_and(|data| data.bytes.as_ref() == PAYLOAD);
    let outcome: &[u8] = if saw_it { b"ok" } else { b"miss" };
    std::fs::write(&result, outcome).expect("write reader result");
}

/// Re-execute this test binary to run only [`reader_process_enters_and_reads`]
/// against `root`, returning whether that process succeeded.
fn spawn_reader(root: &Path) -> bool {
    spawn_reader_entry("reader_process_enters_and_reads", READER_ENV, root)
        .wait()
        .expect("wait for the reader process")
        .success()
}

/// Spawn `entry` as a separate process with its root-directory env var set.
fn spawn_reader_entry(entry: &str, env_var: &str, root: &Path) -> std::process::Child {
    let exe = std::env::current_exe().expect("current test executable");
    std::process::Command::new(exe)
        .args(["--exact", entry, "--nocapture", "--test-threads=1"])
        .env(env_var, root)
        .spawn()
        .expect("spawn the reader process")
}

/// Block until `path` exists, panicking on timeout.
fn wait_for(path: &Path, timeout: Duration) {
    let started = Instant::now();
    while !path.exists() {
        assert!(
            started.elapsed() < timeout,
            "timed out waiting for {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
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

    assert!(
        spawn_reader(&root),
        "the reader process must succeed after a durable write"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}

/// The same cross-process read, but the writer never makes the record durable:
/// it appends the queued group only at the read-committed boundary
/// (`publish_pending`, no `sync_all`). The worker's index predates the write,
/// so the *only* way it can observe the record is the shared read-committed
/// overlay. This is the guarantee the worker-mode publish path depends on:
/// visibility is not gated on durability, and a stale worker's next read sees
/// a committed-but-unfsynced write as soon as it is published.
#[test]
fn a_stale_worker_sees_a_published_but_not_yet_durable_record() {
    let root =
        std::env::temp_dir().join(format!("mtxdb-shared-wal-published-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let db = SharedDatabase::open(root.clone()).expect("writer opens the shared root");
    let state = db.pool(ShardType::State);

    // Start the worker while the store is still empty, so its index snapshot
    // is stale by the time the writer publishes.
    let mut reader = spawn_reader_entry(
        "stale_reader_process_baselines_then_rereads_after_publish",
        STALE_READER_ENV,
        &root,
    );
    let baseline_done = root.join("baseline.done");
    wait_for(&baseline_done, Duration::from_secs(30));

    state
        .put(
            &COLLECTION,
            &NODE,
            &NodeData::new(bytes::Bytes::from_static(PAYLOAD)),
        )
        .expect("writer commits a record");
    db.publish_pending()
        .expect("writer publishes the pending group")
        .expect("a pending group exists to publish");
    std::fs::write(root.join("published.done"), b"1").expect("signal publish");

    let status = reader.wait().expect("wait for the reader process");
    assert!(status.success(), "the reader process must succeed");
    assert_eq!(
        std::fs::read(root.join("reader.result")).expect("reader result"),
        b"ok",
        "the stale worker must see the published-but-unsynced record"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
