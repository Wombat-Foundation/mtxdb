//! Sweep of the WAL reclaim trigger against one shared WAL, with and without a
//! held replay lease, on synthetic data only.
//!
//! Per setting it reports writer throughput, `WouldBlock` refusals, reader
//! latency (p50/p99) and reader `WouldBlock` rate, reclaim count (a sync that
//! shrank the WAL), reclaim sync time, and peak segment fill. The trigger is
//! validated against the emergency boundary (three quarters of the cap), so an
//! unlimited trigger is not expressible: the largest accepted trigger is the
//! closest control. Results characterize this host, not other hardware.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench reclaim_trigger_sweep
//! ```
//!
//! Knobs: `MTXDB_RTS_SECS` (per run, default 6), `MTXDB_RTS_RUNS` (default 3),
//! `MTXDB_RTS_COMMITS` (commits per sync, default 100), `MTXDB_RTS_BATCH`
//! (default 20), `MTXDB_RTS_PAYLOAD` (default 2048), `MTXDB_BENCH_ROOT`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use mtxdb::config::RuntimeConfig;
use mtxdb::layout::ShardType;
use mtxdb::storage::{NodeData, NodeId, StorageError};
use mtxdb::Database;

const COLLECTION: [u8; 16] = [7; 16];

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn node_id(seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&seq.to_le_bytes());
    id
}

fn is_would_block(error: &StorageError) -> bool {
    matches!(error, StorageError::Io(e) if e.kind() == std::io::ErrorKind::WouldBlock)
}

fn pct(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

#[derive(Default)]
struct Run {
    commits: u64,
    refused_commits: u64,
    reads: u64,
    refused_reads: u64,
    read_us: Vec<u128>,
    reclaims: u64,
    reclaim_ms: Vec<f64>,
    peak_wal: u64,
    secs: f64,
}

fn run_once(root: &std::path::Path, trigger: u64, lease: bool, secs: u64) -> (Run, u64) {
    let _ = std::fs::remove_dir_all(root);
    let config = RuntimeConfig {
        reclaim_trigger_bytes: Some(trigger),
        ..RuntimeConfig::default()
    };
    let db = Database::open_with_config(root.to_path_buf(), config).expect("open");
    let cap = db.coordinator().segment_cap();
    let commits_per_sync = env_u64("MTXDB_RTS_COMMITS", 100);
    let batch = env_u64("MTXDB_RTS_BATCH", 20);
    let payload = usize::try_from(env_u64("MTXDB_RTS_PAYLOAD", 2048)).unwrap();
    let data = NodeData::from_slice(&vec![0x5a; payload]);
    let wal = db.layout().shared_wal_path();
    let wal_len = || std::fs::metadata(&wal).map_or(0, |m| m.len());

    // Seed so the reader has something to find, then optionally hold a lease
    // from a snapshot taken before the measured writes.
    let txn = db.begin_transaction();
    for i in 0..64 {
        txn.put(ShardType::EventDag, COLLECTION, node_id(i), &data)
            .expect("seed");
    }
    txn.commit().expect("seed commit");
    db.pool(ShardType::EventDag).sync_all().expect("seed sync");
    let _lease = lease.then(|| {
        db.pool(ShardType::EventDag)
            .scan_collections_at_snapshot(&[COLLECTION])
            .expect("snapshot")
            .2
    });

    let stop = AtomicBool::new(false);
    let reader_ids: Vec<NodeId> = (0..16).map(node_id).collect();
    let mut run = Run::default();
    let started = Instant::now();
    let deadline = Duration::from_secs(secs);
    let reader_store = db.pool(ShardType::EventDag);
    let (reads, refused_reads, read_us) = std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let (mut reads, mut refused, mut lat) = (0u64, 0u64, Vec::new());
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                match reader_store.get_read_committed(&COLLECTION, &reader_ids) {
                    Ok(_) => {}
                    Err(e) if is_would_block(&e) => refused += 1,
                    Err(e) => panic!("read: {e}"),
                }
                lat.push(t.elapsed().as_micros());
                reads += 1;
            }
            (reads, refused, lat)
        });
        let mut next = 1000u64;
        while started.elapsed() < deadline {
            for _ in 0..commits_per_sync {
                let txn = db.begin_transaction();
                let mut staged = true;
                for _ in 0..batch {
                    if txn
                        .put(ShardType::EventDag, COLLECTION, node_id(next), &data)
                        .is_err()
                    {
                        staged = false;
                        break;
                    }
                    next += 1;
                }
                match staged.then(|| txn.commit()) {
                    Some(Ok(())) => run.commits += 1,
                    Some(Err(e)) if is_would_block(&e) => run.refused_commits += 1,
                    Some(Err(e)) => panic!("commit: {e}"),
                    None => run.refused_commits += 1,
                }
                run.peak_wal = run.peak_wal.max(wal_len());
            }
            let before = wal_len();
            run.peak_wal = run.peak_wal.max(before);
            let t = Instant::now();
            for pool in ShardType::ALL {
                db.pool(pool).sync_all().expect("sync_all");
            }
            if wal_len() < before {
                run.reclaims += 1;
                run.reclaim_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        stop.store(true, Ordering::Relaxed);
        reader.join().expect("reader")
    });
    run.secs = started.elapsed().as_secs_f64();
    run.reads = reads;
    run.refused_reads = refused_reads;
    run.read_us = read_us;
    drop(db);
    (run, cap)
}

fn main() {
    let secs = env_u64("MTXDB_RTS_SECS", 6);
    let runs = env_u64("MTXDB_RTS_RUNS", 3);
    let base = std::env::var_os("MTXDB_BENCH_ROOT")
        .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    std::fs::create_dir_all(&base).expect("bench root");
    let root = base.join(format!("mtxdb_bench_rts_{}", std::process::id()));

    // Probe the live cap, then derive the settings from it.
    let (_, cap) = run_once(&root, 1 << 20, false, 0);
    let max_trigger = cap - cap / 4 - 1;
    let settings = [
        ("quarter", cap / 4),
        ("half", cap / 2),
        ("max(~3/4)", max_trigger),
    ];
    println!(
        "segment cap {} MiB; {secs}s x {runs} runs per row",
        cap >> 20
    );
    println!(
        "{:<10} {:<5} {:>9} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9}",
        "trigger",
        "lease",
        "commit/s",
        "wb-commit",
        "reclaims",
        "reclaim ms",
        "peak MiB",
        "reads/s",
        "wb-read",
        "p50 us",
        "p99 us"
    );
    for (name, trigger) in settings {
        for lease in [false, true] {
            for n in 0..runs {
                let (mut r, _) = run_once(&root, trigger, lease, secs);
                r.read_us.sort_unstable();
                r.reclaim_ms.sort_by(f64::total_cmp);
                let med = r
                    .reclaim_ms
                    .get(r.reclaim_ms.len() / 2)
                    .copied()
                    .unwrap_or(0.0);
                println!(
                    "{:<10} {:<5} {:>9.0} {:>8} {:>8} {:>9.0} {:>9} {:>8.0} {:>8} {:>9} {:>9}  run {n}",
                    name,
                    if lease { "held" } else { "none" },
                    r.commits as f64 / r.secs,
                    r.refused_commits,
                    r.reclaims,
                    med,
                    r.peak_wal >> 20,
                    r.reads as f64 / r.secs,
                    r.refused_reads,
                    pct(&r.read_us, 0.5),
                    pct(&r.read_us, 0.99),
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
