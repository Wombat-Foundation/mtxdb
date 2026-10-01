//! Plain-autocommit vs staged-transaction writes through the shared WAL.
//!
//! The question this answers: does staging a persist batch in one
//! [`DatabaseTransaction`](mtxdb::DatabaseTransaction) actually beat writing
//! each record straight to the pool, and at what batch size?
//!
//! Every run writes the same `records` × `payload` bytes into one pool, and
//! differs only in how the writes are grouped and when durability is asked
//! for:
//!
//! - **plain-single** — one `put` per record, each publishing its own journal
//!   group (autocommit). No fsync until the end.
//! - **plain-single-durable** — the same, with a `sync()` after every record.
//! - **plain-many(B)** — one `put_many` of `B` records. Note this still emits
//!   one journal group per **record**, not per call: `put_many` publishes
//!   through the same per-entry path as `put`. The group count is therefore
//!   identical to `plain-single`, which is the point — without a transaction
//!   there is no way to coalesce WAL groups.
//! - **plain-many-durable(B)** — the same, with a `sync()` after every call.
//! - **txn-whole** — one transaction staging every record, one commit, so one
//!   journal group for the lot.
//! - **txn-whole-durable** — the same, with the single durability sync.
//! - **txn-batched(B)** — one transaction per `B` records, one group per
//!   commit. The default run sweeps several `B` so the practical batch size is
//!   visible from one invocation (see `MTXDB_TXN_BATCHES`).
//! - **txn-batched-durable(B)** — the same, with a `sync()` after each commit.
//! - **txn-ryw(B)** — like `txn-batched`, but reads each group back through
//!   `txn.get` before committing, so the read-your-writes cost is measured.
//!
//! Reported per run: wall time (split into staging, commit, and sync), journal
//! groups and WAL bytes (skipped where a per-boundary sync reclaimed the
//! segment), real fsyncs, records made durable per fsync, peak staged bytes
//! against the 64 MiB stage budget, and the read-your-writes cost.
//!
//! Run it on both an HDD and a tmpfs/SSD; the verdict is device-dependent:
//!
//! ```text
//! # tmpfs (default scratch root)
//! cargo bench --manifest-path benches/Cargo.toml --bench txn_commit
//!
//! # a real disk
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench txn_commit
//! ```
//!
//! Env knobs (all optional):
//! - `MTXDB_TXN_RECORDS` (default 100000)
//! - `MTXDB_TXN_PAYLOAD` (default 256)
//! - `MTXDB_TXN_BATCH` (default 1000) — records per journal group/transaction
//! - `MTXDB_TXN_BATCHES` — comma-separated transaction sizes to sweep
//!   (default: `B/4, B, 4B` clamped to the dataset)
//! - `MTXDB_TXN_PASSES` (default 3) — median of the reported times
//! - `MTXDB_TXN_RUNS` — substring filter; only matching run names execute
//! - `MTXDB_TXN_COMPRESS=0` disables per-record zstd (default on)
//! - `MTXDB_TXN_STAGE_PROBE=1` stages past the 64 MiB budget to show the
//!   failure is a clean error, not a crash or a silent drop
//! - `MTXDB_BENCH_ROOT` redirects scratch data off tmpfs
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::journal::{Journal, MAX_TXN_STAGE_BYTES};
use mtxdb::layout::ShardType;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::{PoolPolicies, PoolPolicy, Database};

/// One fixed collection, so every run fights the same index/shard layout.
const COLLECTION: [u8; 16] = [0xA5; 16];
/// Which pool to write; EventDag matches the event_json persist path.
const SHARD: ShardType = ShardType::EventDag;

// ── Deterministic ids and payloads ──────────────────────────────────

/// Well-mixed 64-bit permutation (splitmix64), so synthetic node ids spread
/// across index buckets the way real content-address hashes do.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn node_id(idx: usize) -> NodeId {
    let a = splitmix64(idx as u64);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

/// Incompressible-enough payload stream, so a compression-on run measures
/// zstd-on-noise rather than an artificially tiny frame.
struct PayloadGen {
    state: u64,
}

impl PayloadGen {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(len);
        while buf.len() < len {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            buf.extend_from_slice(&self.state.to_le_bytes());
        }
        buf.truncate(len);
        buf
    }
}

// ── Config / scratch root / formatting helpers ──────────────────────

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

/// PID-suffixed scratch root under `MTXDB_BENCH_ROOT` (or the temp dir), so
/// overlapping runs don't clobber each other and a real disk can be targeted.
fn bench_root() -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    static RUN: AtomicU64 = AtomicU64::new(0);
    ROOT.get_or_init(|| {
        let base =
            std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
        fs::create_dir_all(&base).unwrap();
        loop {
            let token = RUN.fetch_add(1, Ordering::Relaxed);
            let candidate = base.join(format!(
                "mtxdb_bench_txn_commit_{}_{token}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    panic!("create benchmark root {}: {error}", candidate.display());
                }
            }
        }
    })
    .clone()
}

/// Filesystem type of the mount holding `path`, from `/proc/mounts`. Used only
/// to label the report, so a run against tmpfs is not mistaken for a cold one.
fn mount_fstype(path: &Path) -> Option<String> {
    let canonical = path.canonicalize().ok()?;
    let mounts = fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split(' ');
        let _dev = fields.next()?;
        let mount_point = fields.next()?;
        let fstype = fields.next()?;
        let mount_path = Path::new(mount_point);
        if canonical.starts_with(mount_path) {
            let depth = mount_path.components().count();
            if best.as_ref().map_or(true, |(d, _)| depth >= *d) {
                best = Some((depth, fstype.to_owned()));
            }
        }
    }
    best.map(|(_, fstype)| fstype)
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Open one fresh shared-WAL database root, with per-pool compression chosen
/// by `MTXDB_TXN_COMPRESS`.
fn open_db(dir: PathBuf, compress: bool) -> Database {
    let policy = PoolPolicy {
        compress,
        checksum_policy: ChecksumPolicy::Full,
    };
    let policies = PoolPolicies {
        state: policy,
        event_dag: policy,
        edges: policy,
        server_info: policy,
    };
    Database::open_with_policies(dir, policies).expect("open shared database")
}

// ── Run definitions ─────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// One `put` per record, autocommit.
    PlainSingle,
    /// One `put_many` of `batch` records per group, autocommit.
    PlainMany,
    /// One transaction staging every record.
    TxnWhole,
    /// One transaction per `batch` records.
    TxnBatched,
    /// Like `TxnBatched`, but reads each group back before committing.
    TxnRyw,
}

struct RunConfig {
    name: String,
    kind: Kind,
    batch: usize,
    /// `sync()` after every group instead of once at the end.
    durable: bool,
}

/// The run set. `sweep` gives several transaction sizes for `txn-batched`, so
/// the practical batch size is visible from one run; `batch` is the base size
/// used by the plain-many, durable, and read-your-writes runs.
fn configs(records: usize, batch: usize, sweep: &[usize]) -> Vec<RunConfig> {
    let batch = batch.max(1);
    let mut configs = vec![
        RunConfig {
            name: "plain-single".to_owned(),
            kind: Kind::PlainSingle,
            batch: 1,
            durable: false,
        },
        RunConfig {
            name: "plain-single-durable".to_owned(),
            kind: Kind::PlainSingle,
            batch: 1,
            durable: true,
        },
        RunConfig {
            name: "plain-many".to_owned(),
            kind: Kind::PlainMany,
            batch,
            durable: false,
        },
        RunConfig {
            name: "plain-many-durable".to_owned(),
            kind: Kind::PlainMany,
            batch,
            durable: true,
        },
        // `txn-whole` is listed unconditionally; `main` drops it when the whole
        // dataset would exceed the stage budget, which depends on the payload.
        RunConfig {
            name: "txn-whole".to_owned(),
            kind: Kind::TxnWhole,
            batch: records,
            durable: false,
        },
        RunConfig {
            name: "txn-whole-durable".to_owned(),
            kind: Kind::TxnWhole,
            batch: records,
            durable: true,
        },
    ];
    for &size in sweep {
        let size = size.clamp(1, records.max(1));
        configs.push(RunConfig {
            name: format!("txn-batched-B{size}"),
            kind: Kind::TxnBatched,
            batch: size,
            durable: false,
        });
    }
    configs.push(RunConfig {
        name: format!("txn-batched-durable-B{batch}"),
        kind: Kind::TxnBatched,
        batch,
        durable: true,
    });
    configs.push(RunConfig {
        name: format!("txn-ryw-B{batch}"),
        kind: Kind::TxnRyw,
        batch,
        durable: false,
    });
    configs
}

// ── One run ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct Sample {
    stage: Duration,
    commit: Duration,
    sync: Duration,
    ryw: Duration,
    total: Duration,
    /// Journal groups and WAL bytes at the final scan. `None` when a
    /// per-boundary sync reclaimed the segment, so the count is meaningless.
    wal: Option<(u64, u64)>,
    fsyncs: u64,
    durable_records: u64,
    ryw_reads: u64,
    staged_peak: u64,
}

fn sync_pool(db: &Database) -> Duration {
    let started = Instant::now();
    db.pool(SHARD).sync().expect("pool sync");
    started.elapsed()
}

fn run_once(
    cfg: &RunConfig,
    entries: &[(NodeId, Bytes)],
    payload: usize,
    compress: bool,
) -> Sample {
    let dir = bench_root().join(cfg.name.as_str());
    let _ = fs::remove_dir_all(&dir);
    let db = open_db(dir.clone(), compress);
    let collection = COLLECTION;

    let batch = cfg.batch.max(1);
    let mut stage = Duration::ZERO;
    let mut commit = Duration::ZERO;
    let mut sync = Duration::ZERO;
    let mut ryw = Duration::ZERO;
    let mut ryw_reads = 0_u64;
    let mut staged_peak = 0_usize;

    let stats_before = db.coordinator().durability_stats();
    let started = Instant::now();

    match cfg.kind {
        Kind::PlainSingle => {
            for (id, bytes) in entries {
                let data = NodeData::new(bytes.clone());
                let timer = Instant::now();
                db.pool(SHARD).put(&collection, id, &data).expect("put");
                stage += timer.elapsed();
                if cfg.durable {
                    sync += sync_pool(&db);
                }
            }
        }
        Kind::PlainMany => {
            for chunk in entries.chunks(batch) {
                let batch_entries: Vec<(NodeId, NodeData)> = chunk
                    .iter()
                    .map(|(id, bytes)| (*id, NodeData::new(bytes.clone())))
                    .collect();
                let timer = Instant::now();
                db.pool(SHARD)
                    .put_many(&collection, &batch_entries)
                    .expect("put_many");
                stage += timer.elapsed();
                if cfg.durable {
                    sync += sync_pool(&db);
                }
            }
        }
        Kind::TxnWhole => {
            let txn = db.begin_transaction();
            for (id, bytes) in entries {
                let data = NodeData::new(bytes.clone());
                let timer = Instant::now();
                txn.put(SHARD, collection, *id, &data).expect("stage put");
                stage += timer.elapsed();
            }
            staged_peak = entries.len().saturating_mul(payload);
            let timer = Instant::now();
            txn.commit().expect("commit");
            commit += timer.elapsed();
            if cfg.durable {
                sync += sync_pool(&db);
            }
        }
        Kind::TxnBatched | Kind::TxnRyw => {
            let read_your_writes = cfg.kind == Kind::TxnRyw;
            for chunk in entries.chunks(batch) {
                let txn = db.begin_transaction();
                for (id, bytes) in chunk {
                    let data = NodeData::new(bytes.clone());
                    let timer = Instant::now();
                    txn.put(SHARD, collection, *id, &data).expect("stage put");
                    stage += timer.elapsed();
                }
                staged_peak = staged_peak.max(chunk.len().saturating_mul(payload));
                if read_your_writes {
                    let ids: Vec<NodeId> = chunk.iter().map(|(id, _)| *id).collect();
                    let timer = Instant::now();
                    let got = txn.get(SHARD, &collection, &ids).expect("read-your-writes");
                    ryw += timer.elapsed();
                    ryw_reads += ids.len() as u64;
                    for (slot, (_, bytes)) in chunk.iter().enumerate() {
                        let value = got[slot].as_ref().expect("staged read must resolve");
                        assert_eq!(
                            value.bytes.as_ref(),
                            bytes.as_ref(),
                            "staged read returned the wrong payload"
                        );
                    }
                }
                let timer = Instant::now();
                txn.commit().expect("commit");
                commit += timer.elapsed();
                if cfg.durable {
                    sync += sync_pool(&db);
                }
            }
        }
    }

    // Count published groups and WAL bytes before the closing sync, which may
    // reclaim the segment. With per-boundary durability the segment has
    // already been reclaimed, so the count is not meaningful.
    let wal = if cfg.durable {
        None
    } else {
        let scan = Journal::scan_read_only(db.layout().shared_wal_path()).expect("scan wal");
        let bytes: u64 = scan
            .groups
            .iter()
            .flat_map(|group| group.entries.iter())
            .map(|entry| entry.frame_len)
            .sum();
        Some((scan.groups.len() as u64, bytes))
    };

    // One closing sync makes everything durable and pays the fsync the
    // non-durable runs were deferring.
    sync += sync_pool(&db);
    let total = started.elapsed();
    let stats_after = db.coordinator().durability_stats();

    // Spot-check that the live pool serves the writes.
    for index in [0, entries.len() / 2, entries.len().saturating_sub(1)] {
        let (id, bytes) = &entries[index];
        let value = db
            .pool(SHARD)
            .get(&collection, id)
            .expect("live get")
            .expect("live record must exist");
        assert_eq!(
            value.bytes.as_ref(),
            bytes.as_ref(),
            "committed value must match the bytes staged for this record"
        );
    }

    drop(db);
    let _ = fs::remove_dir_all(&dir);

    Sample {
        stage,
        commit,
        sync,
        ryw,
        total,
        wal,
        fsyncs: stats_after.commits.saturating_sub(stats_before.commits),
        durable_records: stats_after
            .commit_records
            .saturating_sub(stats_before.commit_records),
        ryw_reads,
        staged_peak: staged_peak as u64,
    }
}

/// Median of a sample's duration fields, with counters taken from the last
/// (deterministic) sample.
fn reduce(samples: &[Sample]) -> Sample {
    let median = |pick: fn(&Sample) -> Duration| -> Duration {
        let mut all: Vec<Duration> = samples.iter().map(pick).collect();
        all.sort_unstable();
        all.get(all.len() / 2).copied().unwrap_or(Duration::ZERO)
    };
    let mut reduced = samples.last().expect("at least one pass").clone();
    reduced.stage = median(|s| s.stage);
    reduced.commit = median(|s| s.commit);
    reduced.sync = median(|s| s.sync);
    reduced.ryw = median(|s| s.ryw);
    reduced.total = median(|s| s.total);
    reduced
}

// ── Stage-budget probe ──────────────────────────────────────────────

/// Stage records one at a time past `MAX_TXN_STAGE_BYTES` and report where
/// `put` refuses. Proves the budget fails as a clean error and that a
/// partially-staged transaction can still be aborted.
fn probe_stage_budget(payload: usize) {
    let dir = bench_root().join("stage-probe");
    let _ = fs::remove_dir_all(&dir);
    let db = open_db(dir.clone(), true);
    let txn = db.begin_transaction();
    let mut payload_gen = PayloadGen::new(0xB0_0B);
    let mut staged = 0_usize;
    let mut staged_bytes = 0_usize;
    let failure = loop {
        let bytes = Bytes::from(payload_gen.bytes(payload));
        let id = node_id(staged);
        let data = NodeData::new(bytes);
        if let Err(error) = txn.put(SHARD, COLLECTION, id, &data) {
            break Some((staged, staged_bytes, error.to_string()));
        }
        staged += 1;
        staged_bytes += payload;
        if staged_bytes > MAX_TXN_STAGE_BYTES.saturating_mul(2) {
            break None;
        }
    };
    println!();
    println!(
        "── stage-budget probe (payload {}) ──",
        fmt_bytes(payload as u64)
    );
    println!(
        "  budget:            {}",
        fmt_bytes(MAX_TXN_STAGE_BYTES as u64)
    );
    println!("  accounting:        payload + per-mutation framing (~64 B each),");
    println!("                     so refusal lands below the budget in payload bytes");
    match failure {
        Some((count, bytes, message)) => {
            println!(
                "  refused at:        {count} records ({} payload)",
                fmt_bytes(bytes as u64)
            );
            println!("  error:             {message}");
            println!("  abort after:       ok (a partially-staged transaction still aborts)");
        }
        None => println!(
            "  note: never refused below {} — budget not exercised",
            fmt_bytes(MAX_TXN_STAGE_BYTES.saturating_mul(2) as u64)
        ),
    }
    txn.abort().expect("abort a partially-staged transaction");
    drop(db);
    let _ = fs::remove_dir_all(&dir);
}

// ── Report ──────────────────────────────────────────────────────────

fn print_row(label: &str, records: usize, sample: &Sample) {
    let wal_groups = sample
        .wal
        .map_or_else(|| "-".to_owned(), |(groups, _)| groups.to_string());
    let wal_bytes = sample
        .wal
        .map_or_else(|| "-".to_owned(), |(_, bytes)| fmt_bytes(bytes));
    let ryw = if sample.ryw_reads == 0 {
        "-".to_owned()
    } else {
        format!(
            "{:.2} us/read",
            sample.ryw.as_secs_f64() * 1e6 / sample.ryw_reads as f64
        )
    };
    println!(
        "  {label:<22} {records:>8} {groups:>8} {wal:>10} {fsyncs:>7} {per:>10} \
         {stage:>9.1} {commit:>9.1} {sync:>9.1} {total:>9.1} {ryw:>12} {staged:>10}",
        groups = wal_groups,
        wal = wal_bytes,
        fsyncs = sample.fsyncs,
        per = format!("{:.0}", sample.records_per_fsync(records)),
        stage = sample.stage.as_secs_f64() * 1e3,
        commit = sample.commit.as_secs_f64() * 1e3,
        sync = sample.sync.as_secs_f64() * 1e3,
        total = sample.total.as_secs_f64() * 1e3,
        ryw = ryw,
        staged = fmt_bytes(sample.staged_peak),
    );
}

impl Sample {
    fn records_per_fsync(&self, records: usize) -> f64 {
        if self.fsyncs == 0 {
            0.0
        } else {
            records as f64 / self.fsyncs as f64
        }
    }
}

fn main() {
    let records = env_usize("MTXDB_TXN_RECORDS", 100_000).max(1);
    let payload = env_usize("MTXDB_TXN_PAYLOAD", 256).max(1);
    let batch = env_usize("MTXDB_TXN_BATCH", 1_000).max(1);
    let passes = env_usize("MTXDB_TXN_PASSES", 3).max(1);
    let compress = std::env::var("MTXDB_BENCH_COMPRESS").as_deref() != Ok("0");
    let filter = std::env::var("MTXDB_TXN_RUNS").ok();
    let whole_fits = (records as u128 * payload as u128) <= MAX_TXN_STAGE_BYTES as u128;
    // Transaction sizes to sweep. Default: a quarter, the base, and four times
    // the base, clamped to the dataset.
    let sweep: Vec<usize> = std::env::var("MTXDB_TXN_BATCHES").map_or_else(
        |_| {
            let mut sizes = vec![batch / 4, batch, batch.saturating_mul(4)];
            sizes
                .iter_mut()
                .for_each(|size| *size = (*size).clamp(1, records));
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        |raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|size| *size > 0)
                .collect()
        },
    );

    let root = bench_root();
    let fstype = mount_fstype(&root).unwrap_or_else(|| "unknown".to_owned());

    println!("═══════════════════════════════════════════════════════════════");
    println!("  TXN vs PLAIN WRITE BENCHMARK (shared WAL)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  records:      {records}");
    println!(
        "  payload:      {} ({})",
        payload,
        fmt_bytes((records * payload) as u64)
    );
    println!(
        "  batch (B):    {batch}  -> {} groups/boundaries",
        records.div_ceil(batch)
    );
    println!("  txn sweep:    {sweep:?}");
    println!("  passes:       {passes} (median of stage/commit/sync/ryw/total)");
    println!("  compression:  {}", if compress { "on" } else { "off" });
    println!("  filesystem:   {fstype}");
    println!("  scratch dir:  {}", root.display());
    if !whole_fits {
        println!(
            "  note:         dataset {} exceeds the 64 MiB stage budget; txn-whole skipped",
            fmt_bytes((records * payload) as u64)
        );
    }

    let selected: Vec<RunConfig> = configs(records, batch, &sweep)
        .into_iter()
        .filter(|cfg| whole_fits || cfg.kind != Kind::TxnWhole)
        .filter(|cfg| match &filter {
            Some(needle) => cfg.name.contains(needle.as_str()),
            None => true,
        })
        .collect();
    if selected.is_empty() {
        eprintln!("no run matches MTXDB_TXN_RUNS={filter:?}");
        std::process::exit(2);
    }

    // Build the record set once; only the ids and payload bytes are reused,
    // and clones are cheap `Bytes` refcount bumps.
    let mut payload_gen = PayloadGen::new(0xDEAD_BEEF);
    let entries: Vec<(NodeId, Bytes)> = (0..records)
        .map(|index| {
            let bytes = Bytes::from(payload_gen.bytes(payload));
            (node_id(index), bytes)
        })
        .collect();

    println!();
    println!(
        "  {:<22} {:>8} {:>8} {:>10} {:>7} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12} {:>10}",
        "run",
        "records",
        "groups",
        "wal bytes",
        "fsyncs",
        "rec/fsync",
        "stage ms",
        "commit ms",
        "sync ms",
        "total ms",
        "ryw",
        "staged",
    );

    // Rotate the run order each pass so no run always follows the same one.
    // Iterate an index order rather than reversing `selected`, so `samples`
    // stays keyed to each config across every pass.
    let mut samples: Vec<Vec<Sample>> = vec![Vec::new(); selected.len()];
    for pass in 0..passes {
        let order: Vec<usize> = if pass % 2 == 0 {
            (0..selected.len()).collect()
        } else {
            (0..selected.len()).rev().collect()
        };
        for index in order {
            let cfg = &selected[index];
            let sample = run_once(cfg, &entries, payload, compress);
            print_row(&cfg.name, records, &sample);
            samples[index].push(sample);
        }
    }

    println!();
    println!("  ── medians ──");
    for (cfg, runs) in selected.iter().zip(samples.iter()) {
        let sample = reduce(runs);
        print_row(&cfg.name, records, &sample);
    }
    println!("═══════════════════════════════════════════════════════════════");

    for (cfg, runs) in selected.iter().zip(samples.iter()) {
        let sample = reduce(runs);
        println!(
            "bench: txn_commit RUN={} RECORDS={} PAYLOAD={} BATCH={} PASSES={} \
             STAGE_MS={:.3} COMMIT_MS={:.3} SYNC_MS={:.3} TOTAL_MS={:.3} FSYNCS={} \
             RECORDS_PER_FSYNC={:.1} DURABLE_RECORDS={} GROUPS={} WAL_BYTES={} \
             RYW_READS={} RYW_US={:.3} STAGED_BYTES={}",
            cfg.name,
            records,
            payload,
            cfg.batch,
            passes,
            sample.stage.as_secs_f64() * 1e3,
            sample.commit.as_secs_f64() * 1e3,
            sample.sync.as_secs_f64() * 1e3,
            sample.total.as_secs_f64() * 1e3,
            sample.fsyncs,
            sample.records_per_fsync(records),
            sample.durable_records,
            sample.wal.map_or(0, |(groups, _)| groups),
            sample.wal.map_or(0, |(_, bytes)| bytes),
            sample.ryw_reads,
            sample.ryw.as_secs_f64() * 1e6,
            sample.staged_peak,
        );
    }

    if env_flag("MTXDB_TXN_STAGE_PROBE") {
        probe_stage_budget(payload);
    }

    let _ = fs::remove_dir_all(&root);
}
