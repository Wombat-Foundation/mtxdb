//! What a reopen costs as a function of the index size and the delta-log size,
//! to choose a delta-log cap that keeps reopen inside a time budget.
//!
//! Measurement only: no production policy changes. For each index size it
//! builds a store (N distinct records, then one full checkpoint), then grows the
//! delta log by overwriting existing records (so the index does not grow and no
//! structural snapshot is written) up to each requested log size, and at every
//! point reopens the store and prints where the open time went:
//! checkpoint decode, fingerprint validation, index materialization, delta
//! replay (and its operation count), and whether the first read is usable.
//!
//! Each point is measured twice: in this process (warm page cache, warm
//! allocator) and in a fresh child process (cold allocator; the page cache is
//! still warm unless you drop it yourself between the build and the child, see
//! `MTXDB_RO_KEEP`). The child opens the store, reads one record, prints its
//! timings and exits without dropping it.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench reopen_cost
//! ```
//!
//! Env knobs: `MTXDB_RO_RECORDS` (comma list of index sizes in records, default
//! `400000,1600000`), `MTXDB_RO_LOG_PERCENT` (comma list of delta-log size as a
//! percent of the checkpoint file, default `25,50,100,150`; a point whose log
//! would reach 180 MiB is skipped, since the real policy rewrites the checkpoint
//! at 192 MiB), `MTXDB_RO_KEEP=1` (keep the directories, and print each one so
//! you can drop the page cache and rerun a child by hand with
//! `MTXDB_RO_REOPEN_DIR=<dir>`), `MTXDB_BENCH_ROOT`.
//!
//! The result to read is: for index size N and log size C, is `total` under
//! your budget, and is the time in `checkpoint decode` (load) or `delta replay`
//! (log)? `delta replay ops / ms` is the replay throughput the cap must respect.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::packfile::storage::{OpenPath, OpenTimings};
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xC3; 16];
/// Delta-log length past which the real policy rewrites the checkpoint.
const LOG_LIMIT_BYTES: u64 = 180 << 20;
const OVERWRITE_BATCH: u64 = 50_000;

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn node_id(seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&seq.to_le_bytes());
    id
}

fn list(key: &str, default: &str) -> Vec<u64> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |meta| meta.len())
}

/// The sizes of the checkpoint and the current delta log in `dir`.
fn on_disk_sizes(dir: &Path) -> (u64, u64) {
    let mut checkpoint = 0;
    let mut log = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let len = file_len(&entry.path());
            if name == "index.checkpoint" {
                checkpoint = len;
            } else if name.starts_with("index.delta.") {
                log = log.max(len);
            }
        }
    }
    (checkpoint, log)
}

fn print_open(label: &str, timings: &OpenTimings, first_read: Duration, dir: &Path) {
    let (checkpoint, log) = on_disk_sizes(dir);
    let replay_ms = millis(timings.delta_replay);
    let rate = if replay_ms > 0.0 {
        format!(
            "{:.0} ops/ms",
            f64::from(u32::try_from(timings.delta_replay_operations).unwrap_or(u32::MAX))
                / replay_ms
        )
    } else {
        "n/a".to_owned()
    };
    println!(
        "  {label:<8} total {:>7.0} ms | path {:?} | checkpoint {} MiB, log {} MiB | \
         checkpoint decode {:.0}, fingerprint {:.0}, index materialization {:.0}, \
         delta replay {replay_ms:.0} ({} ops, {rate}), shard open {:.0} | first read {:.2} ms",
        millis(timings.total),
        timings.path,
        checkpoint >> 20,
        log >> 20,
        millis(timings.checkpoint_decode),
        millis(timings.fingerprint),
        millis(timings.index_materialization),
        timings.delta_replay_operations,
        millis(timings.shard_open),
        millis(first_read),
    );
}

/// Open the store in `dir`, read one record, and report.
fn open_and_report(label: &str, dir: &Path) -> PackfileStorage {
    let store = PackfileStorage::open(dir.to_path_buf()).expect("open store");
    let started = Instant::now();
    let found = store
        .get(&COLLECTION, &node_id(0))
        .expect("read after open")
        .is_some();
    let first_read = started.elapsed();
    assert!(found, "record 0 must be readable right after the open");
    let timings = store.open_timings().expect("open timings");
    print_open(label, &timings, first_read, dir);
    assert_eq!(timings.path, OpenPath::Checkpoint, "not a checkpoint open");
    store
}

fn build_base(dir: &Path, records: u64) -> PackfileStorage {
    let _ = std::fs::remove_dir_all(dir);
    let store = PackfileStorage::open(dir.to_path_buf()).expect("open new store");
    let data = NodeData::from_slice(&[0x5a; 16]);
    let started = Instant::now();
    for seq in 0..records {
        store
            .put(&COLLECTION, &node_id(seq), &data)
            .expect("put record");
    }
    store.sync_all().expect("sync_all");
    let (checkpoint, log) = on_disk_sizes(dir);
    println!(
        "index {records} records: built in {:.1} s, checkpoint {} MiB, log {} MiB",
        started.elapsed().as_secs_f64(),
        checkpoint >> 20,
        log >> 20
    );
    store
}

/// Overwrite existing records (index size unchanged) until the delta log is at
/// least `target` bytes. Returns false if the log stopped growing.
fn grow_log_to(store: &PackfileStorage, dir: &Path, records: u64, target: u64) -> bool {
    let data = NodeData::from_slice(&[0xa5; 16]);
    let mut next = 0u64;
    loop {
        let (_, log) = on_disk_sizes(dir);
        if log >= target {
            return true;
        }
        for _ in 0..OVERWRITE_BATCH {
            store
                .put(
                    &COLLECTION,
                    &node_id(next.checked_rem(records).unwrap_or(0)),
                    &data,
                )
                .expect("overwrite");
            next = next.saturating_add(1);
        }
        store.sync_all().expect("sync_all");
        if on_disk_sizes(dir).1 == log {
            return false;
        }
    }
}

fn reopen_in_child(dir: &Path) {
    let status = std::process::Command::new(std::env::current_exe().expect("current exe"))
        .env("MTXDB_RO_REOPEN_DIR", dir)
        .status()
        .expect("spawn child");
    assert!(status.success(), "child reopen failed");
}

fn main() {
    if let Some(dir) = std::env::var_os("MTXDB_RO_REOPEN_DIR") {
        // Fresh-process mode: open, report, and exit without dropping the store.
        let store = open_and_report("fresh", &PathBuf::from(dir));
        std::mem::forget(store);
        std::process::exit(0);
    }
    let sizes = list("MTXDB_RO_RECORDS", "400000,1600000");
    let percents = list("MTXDB_RO_LOG_PERCENT", "25,50,100,150");
    let keep = std::env::var("MTXDB_RO_KEEP").is_ok_and(|value| value == "1");
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");

    for records in sizes {
        let dir = base.join(format!(
            "mtxdb_bench_reopen_{}_{records}",
            std::process::id()
        ));
        let mut store = build_base(&dir, records);
        let (checkpoint, _) = on_disk_sizes(&dir);
        for percent in &percents {
            let target = checkpoint.saturating_mul(*percent) / 100;
            if target >= LOG_LIMIT_BYTES {
                println!(
                    "  log {percent}% of checkpoint = {} MiB: skipped (at the rewrite limit)",
                    target >> 20
                );
                continue;
            }
            if !grow_log_to(&store, &dir, records, target) {
                println!("  log {percent}%: the log stopped growing (a checkpoint rewrite?); stopping this size");
                break;
            }
            drop(store);
            println!("  log target {percent}% of checkpoint");
            // Only one process may hold the writer lock: report here, release
            // it, let the child open and report, then take it back to continue.
            drop(open_and_report("warm", &dir));
            reopen_in_child(&dir);
            store = PackfileStorage::open(dir.clone()).expect("reopen to continue");
            if keep {
                println!("  kept: {}", dir.display());
            }
        }
        drop(store);
        if !keep {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
