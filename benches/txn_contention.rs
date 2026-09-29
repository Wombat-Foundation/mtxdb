//! Concurrent transaction commits against one shared WAL with the background
//! group committer running, the shape Synapse workers produce.
//!
//! `txn_commit` measures one writer. This measures what happens to commit
//! latency when `MTXDB_CT_WRITERS` threads each stage `MTXDB_CT_BATCH` records
//! and commit, `MTXDB_CT_COMMITS` times, while the committer fsyncs every
//! `MTXDB_CT_INTERVAL_MS`. It reports per-commit latency percentiles and where
//! the commits spent their time (`SharedDatabase::commit_phase_stats`), so a
//! slow commit can be attributed to publish, lock waits or materialization.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench txn_contention
//! ```
//!
//! Env knobs: `MTXDB_CT_WRITERS` (default 8), `MTXDB_CT_COMMITS` (per writer,
//! default 200), `MTXDB_CT_BATCH` (records per commit, default 20),
//! `MTXDB_CT_PAYLOAD` (default 256), `MTXDB_CT_INTERVAL_MS` (default 100),
//! `MTXDB_BENCH_ROOT`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::journal::JournalCoordinator;
use mtxdb::layout::ShardType;
use mtxdb::storage::{NodeData, NodeId};
use mtxdb::{GroupCommitConfig, PhaseTiming, SharedDatabase};

const COLLECTION: [u8; 16] = [0xA5; 16];

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn node_id(writer: u64, seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&writer.to_le_bytes());
    id[8..].copy_from_slice(&seq.to_le_bytes());
    id
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn phase_line(name: &str, phase: PhaseTiming) -> String {
    let avg = if phase.calls == 0 {
        0.0
    } else {
        phase.total.as_secs_f64() * 1e3 / phase.calls as f64
    };
    format!(
        "  {name:<18} {:>10.1} ms total {:>7} calls {:>9.3} ms avg",
        phase.total.as_secs_f64() * 1e3,
        phase.calls,
        avg
    )
}

fn main() {
    let writers = env_u64("MTXDB_CT_WRITERS", 8);
    let commits = env_u64("MTXDB_CT_COMMITS", 200);
    let batch = env_u64("MTXDB_CT_BATCH", 20);
    let payload = env_u64("MTXDB_CT_PAYLOAD", 256) as usize;
    let interval = Duration::from_millis(env_u64("MTXDB_CT_INTERVAL_MS", 100));

    let base = std::env::var_os("MTXDB_BENCH_ROOT")
        .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    std::fs::create_dir_all(&base).unwrap();
    let root = base.join(format!("mtxdb_bench_txn_contention_{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    let db = Arc::new(SharedDatabase::open(root.clone()).expect("open shared database"));
    let coordinator: Arc<JournalCoordinator> = Arc::clone(db.coordinator());
    coordinator
        .start_background_committer(GroupCommitConfig::with_interval(interval))
        .expect("start committer");

    let started = Instant::now();
    let handles: Vec<_> = (0..writers)
        .map(|writer| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let data = NodeData::new(Bytes::from(vec![0x5Au8; payload]));
                let mut latencies = Vec::with_capacity(commits as usize);
                for commit in 0..commits {
                    let txn = db.begin_transaction();
                    for record in 0..batch {
                        let id = node_id(writer, commit * batch + record);
                        txn.put(ShardType::EventDag, COLLECTION, id, &data)
                            .expect("stage");
                    }
                    let t = Instant::now();
                    txn.commit().expect("commit");
                    latencies.push(t.elapsed());
                }
                latencies
            })
        })
        .collect();
    let mut all: Vec<Duration> = handles
        .into_iter()
        .flat_map(|h| h.join().expect("writer thread"))
        .collect();
    let wall = started.elapsed();
    coordinator
        .stop_background_committer()
        .expect("stop committer");

    all.sort();
    let stats = db.commit_phase_stats();
    println!("writers={writers} commits/writer={commits} batch={batch} payload={payload} interval={interval:?}");
    println!(
        "wall {:.1} ms, {} commits, {:.0} commits/s",
        wall.as_secs_f64() * 1e3,
        all.len(),
        all.len() as f64 / wall.as_secs_f64()
    );
    println!(
        "commit latency ms: p50 {:.3} p95 {:.3} p99 {:.3} max {:.3}",
        percentile(&all, 0.50).as_secs_f64() * 1e3,
        percentile(&all, 0.95).as_secs_f64() * 1e3,
        percentile(&all, 0.99).as_secs_f64() * 1e3,
        all.last().unwrap().as_secs_f64() * 1e3
    );
    println!("{}", phase_line("overlay", stats.overlay));
    println!("{}", phase_line("publish", stats.publish));
    println!("{}", phase_line("register_wait", stats.register_wait));
    println!("{}", phase_line("materialize_wait", stats.materialize_wait));
    println!("{}", phase_line("materialize", stats.materialize));
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
