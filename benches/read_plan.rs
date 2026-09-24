//! Cold-cache benchmark for `get_many`'s opt-in merged read plan.
//!
//! Builds ONE collection of `MTXDB_READ_PLAN_RECORDS` sequential appends,
//! then fetches two target sets of `MTXDB_READ_PLAN_TARGETS` ids each:
//!
//! - **dense**: a contiguous run of insertion order, so the target frames
//!   are physically adjacent;
//! - **random**: uniformly drawn insertion order, so the target frames are
//!   physically dispersed across the collection.
//!
//! Each target set is read twice — once with `ReadPlanPolicy::disabled()`
//! (`off`) and once with a merged plan (`hdd`) — evicting the page cache
//! before every read. The point is to see whether melding nearby candidates
//! into sequential `madvise(MADV_WILLNEED)` extents cuts cold-read time,
//! disk bytes, or major faults, and to show the planned-extent counters
//! actually firing.
//!
//! Run with `cargo bench --manifest-path benches/Cargo.toml --bench read_plan`.
//!
//! For a trustworthy cold cache, `vmtouch -e` alone only drops *clean*
//! pages. The strongest signal comes from dropping the whole page cache
//! between reads as root:
//!
//! ```text
//! sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
//! ```
//!
//! Set `MTXDB_READ_PLAN_MANUAL_DROP=1` to have the bench pause between
//! every read so that drop can be run by hand; it then prints the exact
//! command to paste.
//!
//! Env knobs (all optional):
//! - `MTXDB_READ_PLAN_RECORDS` (default 200000)
//! - `MTXDB_READ_PLAN_TARGETS` (default 100000)
//! - `MTXDB_READ_PLAN_PAYLOAD` (default 1024)
//! - `MTXDB_READ_PLAN_GAP` / `_EXTENT` / `_MIN` override the `hdd` preset.
//! - `MTXDB_BENCH_ROOT` redirects scratch data off tmpfs (see `bench_root`).
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::{PackfileStorage, ReadPlanPolicy};

const ROOM: [u8; 16] = [0xAB; 16];

// ── Deterministic id generation ─────────────────────────────────────

/// Well-mixed 64-bit permutation (splitmix64). Real content-address hashes
/// are uniform, so the synthetic node ids must be too, or the lossy index's
/// bucket selection collapses into one probe cluster.
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

/// Minimal xorshift64 PRNG for target sampling (ids themselves use
/// `splitmix64`, which is a permutation, not a stream).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

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

// ── Kernel I/O + fault counters ─────────────────────────────────────

/// Point-in-time snapshot of this process's page-fault and I/O counters.
///
/// Reads go through `mmap`, so a major fault (or, more reliably,
/// `read_bytes`) is this engine's analog of a physical read; `syscr` is
/// reported to show it stays flat, proving the mmap path does the work.
#[derive(Debug, Clone, Copy, Default)]
struct Snapshot {
    major_faults: u64,
    minor_faults: u64,
    read_syscalls: u64,
    disk_read_bytes: u64,
}

impl Snapshot {
    fn capture() -> Option<Self> {
        fn read_proc(path: &str) -> Option<String> {
            fs::read_to_string(path).ok()
        }

        let stat = read_proc("/proc/self/stat")?;
        // `comm` (2nd field) can contain spaces/parens; split after the
        // last ')' so minflt/majflt land at fixed offsets.
        let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        let minor_faults = fields.get(7)?.parse().ok()?;
        let major_faults = fields.get(9)?.parse().ok()?;

        let io = read_proc("/proc/self/io")?;
        let field = |prefix: &str| -> u64 {
            io.lines()
                .find_map(|line| line.strip_prefix(prefix))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
        };

        Some(Self {
            major_faults,
            minor_faults,
            read_syscalls: field("syscr:"),
            disk_read_bytes: field("read_bytes:"),
        })
    }

    fn delta(self, earlier: Self) -> Self {
        Self {
            major_faults: self.major_faults.saturating_sub(earlier.major_faults),
            minor_faults: self.minor_faults.saturating_sub(earlier.minor_faults),
            read_syscalls: self.read_syscalls.saturating_sub(earlier.read_syscalls),
            disk_read_bytes: self.disk_read_bytes.saturating_sub(earlier.disk_read_bytes),
        }
    }
}

// ── Cache eviction / scratch root ───────────────────────────────────

/// Best-effort page-cache eviction via `vmtouch -e`. Only drops *clean*
/// pages, so callers must have synced first (the bench does). No root
/// needed; returns `false` if vmtouch is missing or fails.
fn evict_dir(dir: &Path) -> bool {
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn vmtouch_on_path() -> bool {
    std::process::Command::new("vmtouch")
        .arg("-h")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// PID-suffixed scratch root, so overlapping bench runs don't clobber each
/// other. `MTXDB_BENCH_ROOT` is honored so GB-scale runs land on a real
/// disk rather than a RAM-backed tmpfs.
fn bench_root() -> std::path::PathBuf {
    static ROOT: OnceLock<std::path::PathBuf> = OnceLock::new();
    static RUN: AtomicU64 = AtomicU64::new(0);
    ROOT.get_or_init(|| {
        let base = std::env::var_os("MTXDB_BENCH_ROOT")
            .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
        fs::create_dir_all(&base).unwrap();
        loop {
            let token = RUN.fetch_add(1, Ordering::Relaxed);
            let candidate = base.join(format!(
                "mtxdb_bench_read_plan_{}_{token}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create benchmark root {}: {error}", candidate.display()),
            }
        }
    })
    .clone()
}

// ── Config ──────────────────────────────────────────────────────────

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The `hdd` preset, with optional per-field env overrides so a run can
/// sweep the merge gap without touching code.
fn hdd_policy() -> ReadPlanPolicy {
    let mut policy = ReadPlanPolicy::hdd();
    if let Ok(v) = std::env::var("MTXDB_READ_PLAN_GAP") {
        if let Ok(n) = v.parse() {
            policy.merge_gap_bytes = n;
        }
    }
    if let Ok(v) = std::env::var("MTXDB_READ_PLAN_EXTENT") {
        if let Ok(n) = v.parse() {
            policy.max_extent_bytes = n;
        }
    }
    if let Ok(v) = std::env::var("MTXDB_READ_PLAN_MIN") {
        if let Ok(n) = v.parse() {
            policy.min_batch_candidates = n;
        }
    }
    policy
}

// ── Measurement ─────────────────────────────────────────────────────

struct Row {
    policy: &'static str,
    target: &'static str,
    found: usize,
    total: usize,
    elapsed: Duration,
    evicted: bool,
    major_faults: Option<u64>,
    minor_faults: Option<u64>,
    disk_read_bytes: Option<u64>,
    read_syscalls: Option<u64>,
    extents: u64,
    prefetch_bytes: u64,
    skipped: u64,
}

fn fmt_opt(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |v| v.to_string())
}

/// Wait for the operator to drop the page cache, when manual mode is on.
/// Prints the exact command so it can be pasted verbatim.
fn wait_for_manual_drop(label: &str) {
    println!();
    println!("  ▶ drop the page cache now, then press Enter to run `{label}`:");
    println!("      sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'");
    print!("  waiting… ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
}

fn measure(
    store: &PackfileStorage,
    dir: &Path,
    policy_name: &'static str,
    target_name: &'static str,
    targets: &[NodeId],
    policy: ReadPlanPolicy,
    manual_drop: bool,
) -> Row {
    if manual_drop {
        wait_for_manual_drop(&format!("{policy_name} / {target_name}"));
    }

    store.set_read_plan_policy(policy);
    // Force decode from mmap, not the in-process node cache, so both
    // policies hit the same path.
    store.collection_cache(&ROOM).clear();
    let evicted = if manual_drop { true } else { evict_dir(dir) };

    let stats_before = store.stats();
    let io_before = Snapshot::capture();
    let started = Instant::now();
    let found = store
        .get_many(&ROOM, targets)
        .unwrap()
        .iter()
        .filter(|value| value.is_some())
        .count();
    let elapsed = started.elapsed();
    let io_after = Snapshot::capture();
    let stats_after = store.stats();

    Row {
        policy: policy_name,
        target: target_name,
        found,
        total: targets.len(),
        elapsed,
        evicted,
        major_faults: io_before
            .zip(io_after)
            .map(|(before, after)| after.delta(before).major_faults),
        minor_faults: io_before
            .zip(io_after)
            .map(|(before, after)| after.delta(before).minor_faults),
        disk_read_bytes: io_before
            .zip(io_after)
            .map(|(before, after)| after.delta(before).disk_read_bytes),
        read_syscalls: io_before
            .zip(io_after)
            .map(|(before, after)| after.delta(before).read_syscalls),
        extents: stats_after
            .read_plan_extents
            .saturating_sub(stats_before.read_plan_extents),
        prefetch_bytes: stats_after
            .read_plan_prefetch_bytes
            .saturating_sub(stats_before.read_plan_prefetch_bytes),
        skipped: stats_after
            .read_plan_skipped_extents
            .saturating_sub(stats_before.read_plan_skipped_extents),
    }
}

// ── Scenario ────────────────────────────────────────────────────────

fn run(records: usize, target_count: usize, payload_len: usize, manual_drop: bool) {
    let target_count = target_count.min(records);
    let dir = bench_root();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    // The read-plan counters are only incremented while stats tracking is
    // on; the prefetch itself is not gated on it.
    store.set_stats_enabled(true);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  READ-PLAN BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  records:      {records}");
    println!("  targets/set:  {target_count}");
    println!("  payload:      {payload_len} bytes");
    println!("  scratch dir:  {}", dir.display());
    if manual_drop {
        println!("  cache:        MANUAL drop between every read");
    } else if vmtouch_on_path() {
        println!("  cache:        vmtouch -e between reads (clean pages only)");
    } else {
        println!("  cache:        WARM — vmtouch not found; numbers are not cold");
    }

    // ── Ingest ──
    println!();
    println!("  [1/3] ingesting…");
    let mut payload_gen = PayloadGen::new(0xDEAD_BEEF);
    let ingest_start = Instant::now();
    const CHUNK: usize = 20_000;
    let mut written = 0usize;
    while written < records {
        let end = (written + CHUNK).min(records);
        let batch: Vec<(NodeId, NodeData)> = (written..end)
            .map(|i| {
                (
                    node_id(i),
                    NodeData::new(Bytes::from(payload_gen.bytes(payload_len))),
                )
            })
            .collect();
        store.put_many(&ROOM, &batch).unwrap();
        written = end;
    }
    store.sync_all().unwrap();
    println!(
        "        {records} records in {:.2?}",
        ingest_start.elapsed()
    );

    // ── Target sets ──
    let dense: Vec<NodeId> = (0..target_count).map(node_id).collect();

    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    let mut seen: HashSet<usize> = HashSet::with_capacity(target_count);
    let mut random_idx: Vec<usize> = Vec::with_capacity(target_count);
    while random_idx.len() < target_count {
        let candidate = (rng.next_u64() % records as u64) as usize;
        if seen.insert(candidate) {
            random_idx.push(candidate);
        }
    }
    let random: Vec<NodeId> = random_idx.into_iter().map(node_id).collect();

    // ── Measure ──
    println!();
    println!("  [2/3] measuring (off vs hdd, dense vs random)…");
    let policy_hdd = hdd_policy();
    println!(
        "        hdd: gap={} B extent={} B min_batch={}",
        policy_hdd.merge_gap_bytes, policy_hdd.max_extent_bytes, policy_hdd.min_batch_candidates
    );

    let combos: [(&'static str, ReadPlanPolicy, &'static str, &Vec<NodeId>); 4] = [
        ("off", ReadPlanPolicy::disabled(), "dense", &dense),
        ("hdd", policy_hdd, "dense", &dense),
        ("off", ReadPlanPolicy::disabled(), "random", &random),
        ("hdd", policy_hdd, "random", &random),
    ];

    let rows: Vec<Row> = combos
        .into_iter()
        .map(|(name, policy, target_name, targets)| {
            measure(
                &store,
                &dir,
                name,
                target_name,
                targets,
                policy,
                manual_drop,
            )
        })
        .collect();

    // ── Report ──
    println!();
    println!("  [3/3] results");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:<6} {:<7} {:>9} {:>10} {:>9} {:>11} {:>9} {:>9}",
        "policy", "target", "found", "elapsed", "majflt", "disk read", "extents", "prefetch"
    );
    for r in &rows {
        println!(
            "  {:<6} {:<7} {:>9} {:>10.2?} {:>9} {:>11} {:>9} {:>9}",
            r.policy,
            r.target,
            format!("{}/{}", r.found, r.total),
            r.elapsed,
            fmt_opt(r.major_faults),
            fmt_opt(r.disk_read_bytes),
            r.extents,
            r.prefetch_bytes,
        );
    }
    println!("═══════════════════════════════════════════════════════════════");

    for r in &rows {
        println!(
            "bench: read_plan POLICY={} TARGET={} FOUND={}/{} RECORDS={} TARGETS={} \
             PAYLOAD={} ELAPSED_US={:.1} MAJFLT={} MINFLT={} SYSCALLS={} DISK_READ_BYTES={} \
             EXTENTS={} PREFETCH_BYTES={} SKIPPED={} EVICTED={}",
            r.policy,
            r.target,
            r.found,
            r.total,
            records,
            target_count,
            payload_len,
            r.elapsed.as_secs_f64() * 1e6,
            fmt_opt(r.major_faults),
            fmt_opt(r.minor_faults),
            fmt_opt(r.read_syscalls),
            fmt_opt(r.disk_read_bytes),
            r.extents,
            r.prefetch_bytes,
            r.skipped,
            r.evicted,
        );
    }

    if let Some(off) = rows
        .iter()
        .find(|r| r.policy == "off" && r.target == "dense")
    {
        if let Some(hdd) = rows
            .iter()
            .find(|r| r.policy == "hdd" && r.target == "dense")
        {
            let speedup = off.elapsed.as_secs_f64() / hdd.elapsed.as_secs_f64();
            println!("bench: read_plan_ratio TARGET=dense HDD_VS_OFF={speedup:.3}");
        }
    }
    if let Some(off) = rows
        .iter()
        .find(|r| r.policy == "off" && r.target == "random")
    {
        if let Some(hdd) = rows
            .iter()
            .find(|r| r.policy == "hdd" && r.target == "random")
        {
            let speedup = off.elapsed.as_secs_f64() / hdd.elapsed.as_secs_f64();
            println!("bench: read_plan_ratio TARGET=random HDD_VS_OFF={speedup:.3}");
        }
    }

    if rows.iter().any(|r| !r.evicted) && !manual_drop {
        println!();
        println!("  NOTE: some reads ran against a warm cache. For cold numbers,");
        println!("        install vmtouch or rerun with MTXDB_READ_PLAN_MANUAL_DROP=1.");
    }

    drop(store);
    let _ = fs::remove_dir_all(&dir);
}

fn main() {
    let records = env_usize("MTXDB_READ_PLAN_RECORDS", 200_000);
    let target_count = env_usize("MTXDB_READ_PLAN_TARGETS", 100_000);
    let payload_len = env_usize("MTXDB_READ_PLAN_PAYLOAD", 1024);
    let manual_drop = std::env::var_os("MTXDB_READ_PLAN_MANUAL_DROP").is_some();

    if !manual_drop && !vmtouch_on_path() {
        eprintln!("⚠ WARNING: `vmtouch` not found on PATH — reads will be WARM.");
        eprintln!("  For a cold cache, rerun with MTXDB_READ_PLAN_MANUAL_DROP=1");
        eprintln!("  and drop the page cache between reads.");
    }

    run(records, target_count, payload_len, manual_drop);
}
