//! Long-running committed writes against one shared WAL, syncing every round,
//! to show the segment stays bounded and what the reclaim sync costs.
//!
//! A sync after the first checkpoint only appends an index delta, which records
//! no journal coverage, so before the size trigger existed the WAL was never
//! reclaimed again and commits failed at the 256 MiB segment limit. This drives
//! enough writes to cross the trigger several times and fails if the WAL ever
//! grows past twice the trigger.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench wal_steady_state
//! ```
//!
//! Env knobs: `MTXDB_WS_ROUNDS` (default 120), `MTXDB_WS_COMMITS` (per round,
//! default 200), `MTXDB_WS_BATCH` (records per commit, default 20),
//! `MTXDB_WS_PAYLOAD` (default 256), `MTXDB_BENCH_ROOT`.

use std::time::{Duration, Instant};

use mtxdb::layout::ShardType;
use mtxdb::storage::{NodeData, NodeId};
use mtxdb::SharedDatabase;

const COLLECTION: [u8; 16] = [7; 16];
/// Matches the non-test `RECLAIM_TRIGGER_LEN` in `journal.rs`.
const TRIGGER_BYTES: u64 = 64 << 20;

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

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn main() {
    let rounds = env_u64("MTXDB_WS_ROUNDS", 120);
    let commits = env_u64("MTXDB_WS_COMMITS", 200);
    let batch = env_u64("MTXDB_WS_BATCH", 20);
    let payload = usize::try_from(env_u64("MTXDB_WS_PAYLOAD", 256)).expect("payload fits usize");

    let base = std::env::var_os("MTXDB_BENCH_ROOT")
        .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");
    let root = base.join(format!("mtxdb_bench_wal_steady_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let db = SharedDatabase::open(root.clone()).expect("open shared database");
    let wal = db.layout().shared_wal_path();
    let wal_len = || std::fs::metadata(&wal).expect("stat wal").len();
    let data = NodeData::from_slice(&vec![0x5a; payload]);

    let mut next = 0u64;
    let mut delta_syncs = Vec::new();
    let mut peak = 0u64;
    println!("rounds={rounds} commits/round={commits} batch={batch} payload={payload}");
    for round in 0..rounds {
        for _ in 0..commits {
            let txn = db.begin_transaction();
            for _ in 0..batch {
                txn.put(ShardType::EventDag, COLLECTION, node_id(next), &data)
                    .expect("stage");
                next = next.saturating_add(1);
            }
            txn.commit().expect("commit");
        }
        let before = wal_len();
        peak = peak.max(before);
        let started = Instant::now();
        let mut per_pool = Vec::new();
        for pool in ShardType::ALL {
            let pool_started = Instant::now();
            db.pool(pool).sync_all().expect("sync_all");
            per_pool.push(millis(pool_started.elapsed()));
        }
        let total = millis(started.elapsed());
        let after = wal_len();
        if after < before {
            println!(
                "round {round}: reclaim {} KiB -> {} KiB, sync {total:.0} ms, per pool {:?}",
                before >> 10,
                after >> 10,
                per_pool.iter().map(|ms| ms.round()).collect::<Vec<_>>()
            );
        } else {
            delta_syncs.push(total);
        }
    }

    delta_syncs.sort_by(f64::total_cmp);
    if let (Some(median), Some(max)) = (delta_syncs.get(delta_syncs.len() / 2), delta_syncs.last()) {
        println!(
            "delta syncs: {} samples, median {median:.0} ms, max {max:.0} ms",
            delta_syncs.len()
        );
    }
    println!("peak WAL before a sync: {} KiB", peak >> 10);
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        peak < TRIGGER_BYTES * 2,
        "the WAL reached {peak} bytes: reclaim is not keeping it bounded"
    );
}
