//! Sweep of the WAL reclaim trigger against one shared WAL, with and without a
//! held replay lease, on synthetic data only.
//!
//! Per setting it reports writer throughput, `WouldBlock` refusals, reader
//! latency in nanoseconds (p50/p99/max, split by whether a sync was in flight
//! when the read started), reader `WouldBlock` count, reads that missed the
//! newest published id ("stale"), syncs where a reclaim was due (WAL above the
//! trigger) versus those that shrank the WAL, replay leases revoked by the
//! cap, elapsed time of due syncs that shrank the WAL (all-pool `sync_all`, so
//! checkpoint and sync work, not reclaim alone), and peak segment fill. `N` readers chase the newest committed ids. A due sync that
//! did not shrink the WAL is a blocked reclaim, e.g. by a held lease. The
//! trigger is validated against the emergency boundary (three quarters of the
//! cap), so an unlimited trigger is not expressible: the largest accepted
//! trigger is the closest control. Results characterize this host only.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench reclaim_trigger_sweep
//! ```
//!
//! Knobs: `MTXDB_RTS_SECS` (wall-clock ceiling per run, default 60),
//! `MTXDB_RTS_WRITE_MIB` (payload target per run, default 384; a run also
//! stops after 1000 refused commits), `MTXDB_RTS_RUNS` (default 3),
//! `MTXDB_RTS_READERS` (default 2), `MTXDB_RTS_COMMITS` (commits per sync,
//! default 100), `MTXDB_RTS_BATCH` (default 20), `MTXDB_RTS_PAYLOAD`
//! (default 2048), `MTXDB_BENCH_ROOT`.
//!
//! Lease caps: `MTXDB_RTS_LEASE_CAP_MIB` sets one fixed per-lease cap (unset or
//! 0 is unlimited) and is reported per row; the unlimited held-lease row is
//! kept as the control. `MTXDB_RTS_LEASE_CAP_SWEEP=1` instead runs unlimited,
//! trigger + 32 MiB, and trigger / 2 for every trigger. A cap at or below the
//! trigger can be exceeded before the first due reclaim, because the cap is
//! enforced only inside a reclaim. `wb-cmt` counts commits refused with
//! `WouldBlock`; blocked reclaims are `due - recl`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

fn pct(sorted: &[u64], p: f64) -> u64 {
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
    missing_new: u64,
    /// Read latencies in nanoseconds, split by whether a sync (and so a
    /// possible reclaim) was in flight when the read started.
    calm_ns: Vec<u64>,
    sync_ns: Vec<u64>,
    /// Syncs that began with the WAL above the trigger (a reclaim was due).
    due: u64,
    /// Due syncs after which the WAL had shrunk.
    reclaims: u64,
    revoked: u64,
    reclaim_ms: Vec<f64>,
    peak_wal: u64,
    secs: f64,
    /// Nominal payload committed, in bytes, and why the run ended.
    written: u64,
    stop: &'static str,
}

fn run_once(
    root: &std::path::Path,
    trigger: u64,
    lease: bool,
    max_secs: u64,
    write_target: u64,
    readers: usize,
    lease_cap: Option<u64>,
) -> (Run, u64) {
    let _ = std::fs::remove_dir_all(root);
    let config = RuntimeConfig {
        reclaim_trigger_bytes: Some(trigger),
        replay_lease_max_retained_bytes: lease_cap,
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
    let syncing = AtomicBool::new(false);
    // Highest id a commit has published. Readers chase it, so they read newly
    // committed records, and the newest id must be visible once published.
    let latest = AtomicU64::new(1000);
    let mut run = Run::default();
    let started = Instant::now();
    let deadline = Duration::from_secs(max_secs);
    // Nominal payload bytes committed; every row does the same amount of work
    // regardless of host speed, unless the segment fills first.
    let mut written = 0u64;
    let store = db.pool(ShardType::EventDag);
    let per_reader = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..readers)
            .map(|_| {
                scope.spawn(|| {
                    let mut r = Run::default();
                    while !stop.load(Ordering::Relaxed) {
                        let top = latest.load(Ordering::Acquire);
                        let ids: Vec<NodeId> =
                            (top.saturating_sub(15)..=top).map(node_id).collect();
                        let in_sync = syncing.load(Ordering::Acquire);
                        let t = Instant::now();
                        let got = store.get_read_committed(&COLLECTION, &ids);
                        let ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
                        match got {
                            Ok(rows) => {
                                if top > 1000 && rows.last().is_some_and(Option::is_none) {
                                    r.missing_new += 1;
                                }
                            }
                            Err(e) if is_would_block(&e) => r.refused_reads += 1,
                            Err(e) => panic!("read: {e}"),
                        }
                        // A sync that ended mid-read still counts as calm; the
                        // split is by the state at read start.
                        if in_sync || syncing.load(Ordering::Acquire) {
                            r.sync_ns.push(ns);
                        } else {
                            r.calm_ns.push(ns);
                        }
                        r.reads += 1;
                    }
                    r
                })
            })
            .collect();
        let mut next = 1000u64;
        let keep_going = |written: u64, refused: u64| {
            started.elapsed() < deadline && written < write_target && refused < 1000
        };
        while keep_going(written, run.refused_commits) {
            for _ in 0..commits_per_sync {
                // Stop mid-batch too, so a row never overshoots its target.
                if !keep_going(written, run.refused_commits) {
                    break;
                }
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
                    Some(Ok(())) => {
                        run.commits += 1;
                        written += batch * payload as u64;
                        latest.store(next - 1, Ordering::Release);
                    }
                    Some(Err(e)) if is_would_block(&e) => run.refused_commits += 1,
                    Some(Err(e)) => panic!("commit: {e}"),
                    None => run.refused_commits += 1,
                }
                run.peak_wal = run.peak_wal.max(wal_len());
            }
            let before = wal_len();
            run.peak_wal = run.peak_wal.max(before);
            let due = before > trigger;
            let t = Instant::now();
            syncing.store(true, Ordering::Release);
            for pool in ShardType::ALL {
                db.pool(pool).sync_all().expect("sync_all");
            }
            syncing.store(false, Ordering::Release);
            if due {
                run.due += 1;
                if wal_len() < before {
                    run.reclaims += 1;
                    run.reclaim_ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        handles
            .into_iter()
            .map(|h| h.join().expect("reader"))
            .collect::<Vec<_>>()
    });
    run.secs = started.elapsed().as_secs_f64();
    run.written = written;
    run.stop = if written >= write_target {
        "target"
    } else if run.refused_commits >= 1000 {
        "refusals"
    } else {
        "time"
    };
    for r in per_reader {
        run.reads += r.reads;
        run.refused_reads += r.refused_reads;
        run.missing_new += r.missing_new;
        run.calm_ns.extend(r.calm_ns);
        run.sync_ns.extend(r.sync_ns);
    }
    run.revoked = db.coordinator().replay_leases_revoked();
    drop(db);
    (run, cap)
}

fn main() {
    // Upper bound per run; each run stops earlier once it has committed
    // `MTXDB_RTS_WRITE_MIB` of payload or been refused 1000 commits.
    let secs = env_u64("MTXDB_RTS_SECS", 60);
    let write_target = env_u64("MTXDB_RTS_WRITE_MIB", 384) << 20;
    let runs = env_u64("MTXDB_RTS_RUNS", 3);
    // Optional fixed per-lease cap in MiB; unset or zero means unlimited.
    let fixed_lease_cap = std::env::var("MTXDB_RTS_LEASE_CAP_MIB")
        .ok()
        .map(|value| {
            let mib = value
                .parse::<u64>()
                .expect("MTXDB_RTS_LEASE_CAP_MIB must be a non-negative integer");
            mib.checked_mul(1 << 20)
                .expect("MTXDB_RTS_LEASE_CAP_MIB is too large")
        })
        .filter(|bytes| *bytes > 0);
    let relative_cap_sweep = std::env::var("MTXDB_RTS_LEASE_CAP_SWEEP")
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" => true,
            "0" | "false" => false,
            _ => panic!("MTXDB_RTS_LEASE_CAP_SWEEP must be 0, 1, true, or false"),
        })
        .unwrap_or(false);
    assert!(
        !(relative_cap_sweep && std::env::var_os("MTXDB_RTS_LEASE_CAP_MIB").is_some()),
        "use either MTXDB_RTS_LEASE_CAP_SWEEP or MTXDB_RTS_LEASE_CAP_MIB, not both"
    );
    let readers = usize::try_from(env_u64("MTXDB_RTS_READERS", 2))
        .unwrap()
        .max(1);
    let base = std::env::var_os("MTXDB_BENCH_ROOT")
        .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    std::fs::create_dir_all(&base).expect("bench root");
    let root = base.join(format!("mtxdb_bench_rts_{}", std::process::id()));

    let (_, cap) = run_once(&root, 1 << 20, false, 0, 0, 1, None);
    let max_trigger = cap - cap / 4 - 1;
    let settings = [
        ("quarter", cap / 4),
        ("half", cap / 2),
        ("max(~3/4)", max_trigger),
    ];
    let wt = write_target >> 20;
    println!(
        "segment cap {} MiB; lease cap {}; <= {secs}s / {wt} MiB payload x {runs} runs per row; {readers} reader(s); latencies in ns per 16-key read batch (ids built before the timer)",
        cap >> 20,
        if relative_cap_sweep {
            "relative sweep".to_owned()
        } else {
            fixed_lease_cap.map_or_else(|| "off".to_owned(), |b| format!("{} MiB", b >> 20))
        }
    );
    println!(
        "{:<10} {:<5} {:<12} {:>8} {:>7} {:>4} {:>4} {:>5} {:>7} {:>7} {:>8} {:>8} {:>7} {:>9} {:>9} {:>9} {:>10} {:>10} {:>9} {:>7} {:>8} {:>8}",
        "trigger", "lease", "lease cap", "commit/s", "wb-cmt", "due", "recl", "revkd", "dueshrk ms", "peakMiB", "reads/s",
        "wb-read", "stale", "calm p50", "calm p99", "sync p99", "sync max", "calm max", "calm n", "sync n", "doneMiB", "stop"
    );
    for (name, trigger) in settings {
        let mut held_caps = if relative_cap_sweep {
            vec![
                ("unlimited", None),
                ("trigger+32", Some(trigger + (32 << 20))),
                ("trigger/2", Some(trigger / 2)),
            ]
        } else {
            vec![("unlimited", None)]
        };
        if let Some(fixed_cap) = fixed_lease_cap {
            held_caps.push(("fixed", Some(fixed_cap)));
        }
        let cases = std::iter::once((false, "off", None))
            .chain(held_caps.into_iter().map(|(label, cap)| (true, label, cap)));
        for (lease, cap_label, lease_cap) in cases {
            for n in 0..runs {
                let (mut r, _) = run_once(
                    &root,
                    trigger,
                    lease,
                    secs,
                    write_target,
                    readers,
                    lease_cap,
                );
                r.calm_ns.sort_unstable();
                r.sync_ns.sort_unstable();
                r.reclaim_ms.sort_by(f64::total_cmp);
                let med = r
                    .reclaim_ms
                    .get(r.reclaim_ms.len() / 2)
                    .copied()
                    .unwrap_or(0.0);
                println!(
                    "{:<10} {:<5} {:<12} {:>8.0} {:>7} {:>4} {:>4} {:>5} {:>7.0} {:>7} {:>8.0} {:>8} {:>7} {:>9} {:>9} {:>9} {:>10} {:>10} {:>9} {:>7} {:>8} {:>8}  run {n}",
                    name,
                    if lease { "held" } else { "none" },
                    cap_label,
                    r.commits as f64 / r.secs,
                    r.refused_commits,
                    r.due,
                    r.reclaims,
                    r.revoked,
                    med,
                    r.peak_wal >> 20,
                    r.reads as f64 / r.secs,
                    r.refused_reads,
                    r.missing_new,
                    pct(&r.calm_ns, 0.5),
                    pct(&r.calm_ns, 0.99),
                    pct(&r.sync_ns, 0.99),
                    r.sync_ns.last().copied().unwrap_or(0),
                    r.calm_ns.last().copied().unwrap_or(0),
                    r.calm_ns.len(),
                    r.sync_ns.len(),
                    r.written >> 20,
                    r.stop,
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
