//! What a pool that never reports coverage costs at the real segment cap.
//!
//! Two pools take every transaction; only `EventDag` is ever synced, so `State`
//! never reports and reclaim cannot advance until something makes it. The run
//! goes on until commits are refused at the segment cap (or `MTXDB_WL_ROUNDS`
//! is reached) and prints one line per round from the emergency line (three
//! quarters of the cap) on: WAL length, sync latency, whether the sync moved
//! `EventDag`'s coverage, and the stall count. It measures cost only; the
//! bound on forced steps is asserted by a unit test at a 1 MiB cap.
//!
//! By default the database remediates the lagging pool (checkpoints `State`
//! from the emergency zone). `MTXDB_WL_REMEDIATE=0` turns that off, which shows
//! the cost of a stall nothing can clear and ends in the hard-cap refusal.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench wal_lagging_pool
//! MTXDB_WL_REMEDIATE=0 MTXDB_BENCH_ROOT=... cargo bench ... --bench wal_lagging_pool
//! ```
//!
//! Reaching 192 MiB of WAL at the defaults takes a few thousand commits; the
//! disk needs about 2 GiB free. Env knobs: `MTXDB_WL_COMMITS` (per round,
//! default 100), `MTXDB_WL_BATCH` (records per commit, default 20),
//! `MTXDB_WL_PAYLOAD` (default 256), `MTXDB_WL_ROUNDS` (default 400; a run that
//! is never refused writes about 190 MiB of packs per 163 rounds),
//! `MTXDB_WL_REMEDIATE`, `MTXDB_WL_SYNC_STATE=1` (also sync `State`, so nothing
//! lags: a control run), `MTXDB_WL_SLOW_MS` (also print any sync slower than
//! this, default 400), `MTXDB_BENCH_ROOT`.

use std::time::{Duration, Instant};

use mtxdb::layout::ShardType;
use mtxdb::storage::{NodeData, NodeId};
use mtxdb::SharedDatabase;

const COLLECTION: [u8; 16] = [9; 16];

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

/// Full checkpoints taken so far, per pool, in `ShardType::ALL` order.
fn checkpoint_counts(db: &SharedDatabase) -> Vec<u64> {
    ShardType::ALL
        .iter()
        .map(|pool| db.pool(*pool).stats().checkpoint_writes)
        .collect()
}

/// The phases of the active pool's sync.
fn print_phases(timings: &mtxdb::packfile::storage::SyncTimings) {
    println!(
        "  phases: pack_flush {:.0} ms, pack_fsync {:.0} ms, delta_log {:.0} ms, \
         checkpoint {:.0} ms, wal {:.0} ms, journal_fsync {:.0} ms",
        millis(timings.pack_flush),
        millis(timings.pack_fsync),
        millis(timings.delta_log),
        millis(timings.checkpoint),
        millis(timings.wal),
        millis(timings.journal_fsync),
    );
}

/// Where the time of every full checkpoint taken since `before` went. A
/// remediation checkpoint runs inside the sync, so `State` can appear too.
fn print_checkpoint_breakdowns(db: &SharedDatabase, before: &[u64]) {
    for (pool, was) in ShardType::ALL.iter().zip(before) {
        if db.pool(*pool).stats().checkpoint_writes <= *was {
            continue;
        }
        let Some(b) = db.pool(*pool).checkpoint_breakdown() else {
            continue;
        };
        println!(
            "  {pool:?} checkpoint {:.0} ms: pre-sync {:.0}, lock wait {:.0}, locked sync {:.0}, \
             snapshot {:.0}, serialize {:.0}, write {:.0}, dir sync {:.0}, journal.lsn {:.0}, \
             WAL reclaim {:.0}, retire {:.0}",
            millis(b.total),
            millis(b.pre_sync),
            millis(b.lock_wait),
            millis(b.locked_sync),
            millis(b.snapshot),
            millis(b.serialize),
            millis(b.write),
            millis(b.directory_sync),
            millis(b.journal_lsn),
            millis(b.reclaim),
            millis(b.retire),
        );
    }
}

fn main() {
    let commits = env_u64("MTXDB_WL_COMMITS", 100);
    let batch = env_u64("MTXDB_WL_BATCH", 20);
    let payload = usize::try_from(env_u64("MTXDB_WL_PAYLOAD", 256)).expect("payload fits usize");
    let max_rounds = env_u64("MTXDB_WL_ROUNDS", 400);
    let sync_state = env_u64("MTXDB_WL_SYNC_STATE", 0) != 0;
    let slow_ms = f64::from(u32::try_from(env_u64("MTXDB_WL_SLOW_MS", 400)).unwrap_or(u32::MAX));
    let remediate = env_u64("MTXDB_WL_REMEDIATE", 1) != 0;

    let base = std::env::var_os("MTXDB_BENCH_ROOT")
        .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");
    let root = base.join(format!("mtxdb_bench_wal_lagging_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let db = SharedDatabase::open(root.clone()).expect("open shared database");
    let coordinator = db.coordinator();
    if !remediate {
        coordinator.set_blocker_remediation(|_| {});
    }
    let cap = coordinator.segment_cap();
    let emergency = cap.saturating_sub(cap / 4);
    let data = NodeData::from_slice(&vec![0x5a; payload]);
    println!(
        "cap={} MiB trigger={} MiB emergency={} MiB remediation={remediate} \
         commits/round={commits} batch={batch} payload={payload}",
        cap >> 20,
        coordinator.reclaim_trigger_len() >> 20,
        emergency >> 20
    );

    let mut next = 0u64;
    let mut zone_syncs = 0u64;
    let mut zone_forced = 0u64;
    let mut zone_ms = 0.0f64;
    let mut slowest = 0.0f64;
    let mut smallest_headroom = u64::MAX;
    let mut refusal = None;
    'rounds: for round in 0..max_rounds {
        for _ in 0..commits {
            let txn = db.begin_transaction();
            for _ in 0..batch {
                for pool in [ShardType::State, ShardType::EventDag] {
                    txn.put(pool, COLLECTION, node_id(next), &data)
                        .expect("stage");
                }
                next = next.saturating_add(1);
            }
            if let Err(error) = txn.commit() {
                refusal = Some((round, error.to_string()));
                break 'rounds;
            }
        }
        let len = coordinator.segment_len();
        let coverage_before = db.pool(ShardType::EventDag).durable_coverage();
        let checkpoints_before = checkpoint_counts(&db);
        let started = Instant::now();
        db.pool(ShardType::EventDag).sync_all().expect("sync_all");
        let sync_ms = millis(started.elapsed());
        let event_timings = db
            .pool(ShardType::EventDag)
            .sync_timings()
            .expect("sync timings");
        if sync_state {
            db.pool(ShardType::State).sync_all().expect("sync_all");
        }
        if len < emergency && sync_ms < slow_ms {
            continue;
        }
        let moved = db.pool(ShardType::EventDag).durable_coverage() > coverage_before;
        zone_syncs = zone_syncs.saturating_add(1);
        zone_forced = zone_forced.saturating_add(u64::from(moved));
        zone_ms += sync_ms;
        slowest = slowest.max(sync_ms);
        smallest_headroom = smallest_headroom.min(cap.saturating_sub(coordinator.segment_len()));
        println!(
            "round {round}: wal {} KiB -> {} KiB, sync {sync_ms:.0} ms, coverage moved {moved}, \
             stalled {}, stalls {}, blockers {:?}",
            len >> 10,
            coordinator.segment_len() >> 10,
            coordinator.is_reclaim_stalled(),
            coordinator.reclaim_stalls(),
            coordinator.reclaim_blockers()
        );
        print_phases(&event_timings);
        print_checkpoint_breakdowns(&db, &checkpoints_before);
    }

    println!(
        "emergency zone: {zone_syncs} syncs, {zone_forced} moved coverage, \
         {:.0} ms total, slowest {slowest:.0} ms, smallest headroom {} KiB",
        zone_ms,
        if smallest_headroom == u64::MAX {
            0
        } else {
            smallest_headroom >> 10
        }
    );
    match refusal {
        Some((round, message)) => println!("commits refused in round {round}: {message}"),
        None => println!("no commit was refused in {max_rounds} rounds"),
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&root);
}
