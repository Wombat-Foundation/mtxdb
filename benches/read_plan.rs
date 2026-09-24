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
/// pages of the named files, and only while no process has them mapped —
/// callers must have dropped the store first. No root needed; returns
/// `false` if vmtouch is missing or fails.
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

/// Filesystem type of the mount holding `path`, from `/proc/mounts` (the
/// longest mount point that prefixes `path`). `None` if `/proc/mounts`
/// is unreadable or no entry matches (non-Linux).
fn mount_fstype(path: &Path) -> Option<String> {
    let canonical = path.canonicalize().ok()?;
    let mounts = fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        // Fields are space-separated; the mount point (field 2) escapes
        // spaces as `\040`, which canonical paths won't contain here.
        let mut fields = line.split(' ');
        let _dev = fields.next()?;
        let mount_point = fields.next()?;
        let fstype = fields.next()?;
        let mount_path = Path::new(mount_point);
        if canonical.starts_with(mount_path) {
            let depth = mount_path.components().count();
            let deeper_or_equal = best.as_ref().map_or(true, |(d, _)| depth >= *d);
            if deeper_or_equal {
                best = Some((depth, fstype.to_owned()));
            }
        }
    }
    best.map(|(_, fstype)| fstype)
}

/// True when `path` lives on an in-memory filesystem (`tmpfs`, `ramfs`,
/// `devtmpfs`), where "cold reads" are a contradiction: the data never
/// leaves RAM and neither `drop_caches` nor `vmtouch` can evict it.
fn is_ram_backed(path: &Path) -> bool {
    matches!(
        mount_fstype(path).as_deref(),
        Some("tmpfs" | "ramfs" | "devtmpfs")
    )
}

/// Run the root page-cache drop directly, rather than asking the operator
/// to paste it. Returns `Ok(())` only if the write succeeded (i.e. the
/// bench is running as root); a permission error comes back as `Err` so the
/// caller can tell "ran and failed" from "not attempted".
fn drop_page_caches() -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open("/proc/sys/vm/drop_caches")?;
    file.write_all(b"3")?;
    file.flush()
}

/// Whether this process can drop the page cache itself (i.e. is root and
/// `/proc/sys/vm/drop_caches` is writable).
fn root_drop_available() -> bool {
    fs::OpenOptions::new()
        .write(true)
        .open("/proc/sys/vm/drop_caches")
        .is_ok()
}

/// How each pass evicts the page cache before its read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eviction {
    /// `drop_page_caches()` in-process (bench is root).
    RootDrop,
    /// `vmtouch -e <dir>`; only clean, unmapped pages.
    Vmtouch,
    /// Pause and let the operator run `drop_caches` by hand.
    Manual,
}

/// What the eviction step did for one pass, reported verbatim in the
/// `EVICTED=` field. A single `true`/`false` flattens three distinct states:
/// a command that ran and succeeded, a human drop we cannot observe, and no
/// eviction at all. Keeping them separate is what makes `EVICTED=requested`
/// readable as "we asked, but nothing confirms the pages left."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictionStatus {
    /// The eviction command ran and reported success.
    Ran,
    /// The operator was prompted to drop by hand; the result is unobservable.
    Requested,
    /// No eviction was available (missing tool / would have failed).
    Unavailable,
}

impl EvictionStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ran => "true",
            Self::Requested => "requested",
            Self::Unavailable => "false",
        }
    }
}

/// Pick the strongest available eviction strategy, in order of reliability.
fn select_eviction(manual_drop: bool) -> Eviction {
    if manual_drop {
        Eviction::Manual
    } else if root_drop_available() {
        Eviction::RootDrop
    } else {
        Eviction::Vmtouch
    }
}

/// Wait for the operator to press Enter, after printing the exact command
/// so it can be pasted verbatim.
fn wait_for_manual_drop(label: &str) {
    println!();
    println!("  ▶ drop the page cache now, then press Enter to run `{label}`:");
    println!("      sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'");
    print!("  waiting… ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
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

/// Median of the gaps between consecutive sorted record indices, in bytes.
///
/// A target set is only informative about the merge gap if it lands in the
/// regime the preset was reasoned for: one wanted record every ~0.5-2 MB.
/// Densities far above that (most records wanted) make both policies read
/// the whole file; far below, every read is an isolated seek and no melding
/// can happen. This is the number that says which regime the run hit, from
/// record indices and the payload stride.
fn median_target_gap_bytes(sorted_indices: &[usize], stride_bytes: u64) -> u64 {
    if sorted_indices.len() < 2 {
        return 0;
    }
    let mut gaps: Vec<u64> = sorted_indices
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as u64 * stride_bytes)
        .collect();
    gaps.sort_unstable();
    gaps[gaps.len() / 2]
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

/// Total size in bytes of every regular file under `dir`, recursively. Used
/// to express a read's device bytes as a fraction of the store, so a full
/// scan is visible as ~1.0 rather than having to be inferred.
fn store_size_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            total += store_size_bytes(&entry.path());
        } else if kind.is_file() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

// ── Measurement ─────────────────────────────────────────────────────

struct Row {
    policy: &'static str,
    target: &'static str,
    found: usize,
    total: usize,
    elapsed: Duration,
    evicted: EvictionStatus,
    major_faults: Option<u64>,
    minor_faults: Option<u64>,
    disk_read_bytes: Option<u64>,
    read_syscalls: Option<u64>,
    extents: u64,
    prefetch_bytes: u64,
    skipped: u64,
}

/// Median of a set of samples, rounded down. Empty input yields
/// `Duration::ZERO`; callers only reach here with a non-empty pass count.
fn median_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn fmt_opt(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |v| v.to_string())
}

/// One timed read, with the store reopened cold for each pass.
///
/// The store must be **dropped** before the cache is dropped and reopened
/// after: while it is alive its shards are mmapped, and both
/// `drop_caches` and `vmtouch -e` skip pages a live process has mapped, so
/// an in-place eviction leaves them resident and silently measures a warm
/// cache. A fresh `open_read_only` also has no in-process node cache, so
/// every id goes through index lookup and mmap decode.
fn measure_once(
    dir: &Path,
    policy_name: &'static str,
    target_name: &'static str,
    targets: &[NodeId],
    policy: ReadPlanPolicy,
    eviction: Eviction,
) -> Row {
    if eviction == Eviction::Manual {
        wait_for_manual_drop(&format!("{policy_name} / {target_name}"));
    }

    // Step 1: evict while nothing has the shards mapped. `EVICTED` records
    // whether an eviction command actually ran and reported success — not
    // whether the pages left, which neither syscall tells us.
    let evicted = match eviction {
        Eviction::RootDrop => {
            if drop_page_caches().is_ok() {
                EvictionStatus::Ran
            } else {
                EvictionStatus::Unavailable
            }
        }
        Eviction::Vmtouch => {
            if evict_dir(dir) {
                EvictionStatus::Ran
            } else {
                EvictionStatus::Unavailable
            }
        }
        // A human ran `drop_caches`; we cannot observe the result.
        Eviction::Manual => EvictionStatus::Requested,
    };

    // Step 2: reopen fresh, apply the policy, and time a mmap-backed read.
    let store = PackfileStorage::open_read_only(dir.to_path_buf()).unwrap();
    store.set_stats_enabled(true);
    store.set_read_plan_policy(policy);

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

/// Reduce a combo's per-pass rows to one: median elapsed, counters from the
/// last pass (they are per-read totals, not accumulated).
fn reduce_passes(mut rows: Vec<Row>) -> Row {
    let elapsed_samples: Vec<Duration> = rows.iter().map(|r| r.elapsed).collect();
    let mut row = rows.pop().expect("at least one pass");
    row.elapsed = median_duration(&elapsed_samples);
    row
}

/// Measure every `(policy, target)` combo, interleaving the passes: one read
/// of each combo per pass, then the next pass. Running a combo's passes back
/// to back would let the policy measured first warm the drive's cache and
/// bias the one measured second; interleaving spreads that across both.
fn measure(
    dir: &Path,
    combos: &[(&'static str, ReadPlanPolicy, &'static str, &Vec<NodeId>)],
    eviction: Eviction,
    passes: usize,
) -> Vec<Row> {
    let mut samples: Vec<Vec<Row>> = (0..combos.len()).map(|_| Vec::new()).collect();
    for _ in 0..passes {
        for (index, (policy_name, policy, target_name, targets)) in combos.iter().enumerate() {
            samples[index].push(measure_once(
                dir,
                policy_name,
                target_name,
                targets,
                *policy,
                eviction,
            ));
        }
    }
    samples.into_iter().map(reduce_passes).collect()
}

// ── Scenario ────────────────────────────────────────────────────────

fn run(records: usize, target_count: usize, payload_len: usize, manual_drop: bool, passes: usize) {
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
    println!("  passes:       {passes} (median reported)");
    println!("  scratch dir:  {}", dir.display());
    let fstype = mount_fstype(&dir).unwrap_or_else(|| "unknown".to_owned());
    println!("  filesystem:   {fstype}");
    if manual_drop {
        println!("  cache:        MANUAL drop between every read");
    } else if root_drop_available() {
        println!("  cache:        in-process drop_page_caches() between reads");
    } else if vmtouch_on_path() {
        println!("  cache:        vmtouch -e between reads (store closed first)");
    } else {
        println!("  cache:        WARM — no eviction available; numbers are not cold");
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
    //
    // Dense: a contiguous prefix, the "read most of the file" case.
    // Sparse: uniformly sampled across the *whole* dataset (not clustered),
    // the regime the merge gap is supposed to matter in. Its achieved
    // density is reported below, because a run that lands nowhere near the
    // 0.5-2 MB gap regime cannot say anything about the threshold.
    let dense: Vec<NodeId> = (0..target_count).map(node_id).collect();

    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    let mut seen: HashSet<usize> = HashSet::with_capacity(target_count);
    let mut sparse_idx: Vec<usize> = Vec::with_capacity(target_count);
    while sparse_idx.len() < target_count {
        let candidate = (rng.next_u64() % records as u64) as usize;
        if seen.insert(candidate) {
            sparse_idx.push(candidate);
        }
    }
    let sparse: Vec<NodeId> = sparse_idx.iter().copied().map(node_id).collect();

    // The payload is ~payload_len bytes plus the frame's fixed overhead and
    // CRC, so record stride is a little larger; using payload_len is close
    // enough for a regime hint.
    let mut sparse_sorted = sparse_idx.clone();
    sparse_sorted.sort_unstable();
    let stride = payload_len as u64;
    let median_gap = median_target_gap_bytes(&sparse_sorted, stride);
    let span_bytes = records as u64 * stride;
    println!();
    println!("  target density (sparse):");
    println!(
        "        {} targets / {} (~1 per {})",
        target_count,
        fmt_bytes(span_bytes),
        fmt_bytes(median_gap),
    );

    // From here on the measured path needs a store it can close and
    // reopen, so release the ingest writer first.
    drop(store);

    // Actual on-disk store size, to express a read as a fraction of the
    // store and catch full scans / re-reads the byte floor alone cannot.
    let store_bytes = store_size_bytes(&dir);
    println!("  store size:   {}", fmt_bytes(store_bytes));

    // ── Measure ──
    //
    // Dense is a small contiguous prefix that always reads nearly the whole
    // small store and says nothing about the merge threshold. It is on by
    // default for the ratio comparison; skip it with MTXDB_READ_PLAN_DENSE=0
    // to halve the measured rows when only the sparse verdict matters.
    let include_dense = std::env::var("MTXDB_READ_PLAN_DENSE")
        .map(|v| v != "0")
        .unwrap_or(true);
    println!();
    println!(
        "  [2/3] measuring (off vs hdd, {}, {passes} interleaved passes, median)…",
        if include_dense {
            "dense + sparse"
        } else {
            "sparse only"
        }
    );
    let policy_hdd = hdd_policy();
    println!(
        "        hdd: gap={} B extent={} B min_batch={}",
        policy_hdd.merge_gap_bytes, policy_hdd.max_extent_bytes, policy_hdd.min_batch_candidates
    );

    let mut combos: Vec<(&'static str, ReadPlanPolicy, &'static str, &Vec<NodeId>)> = Vec::new();
    if include_dense {
        combos.push(("off", ReadPlanPolicy::disabled(), "dense", &dense));
        combos.push(("hdd", policy_hdd, "dense", &dense));
    }
    combos.push(("off", ReadPlanPolicy::disabled(), "sparse", &sparse));
    combos.push(("hdd", policy_hdd, "sparse", &sparse));

    let eviction = select_eviction(manual_drop);
    let rows: Vec<Row> = measure(&dir, &combos, eviction, passes);

    // ── Report ──
    //
    // Every column is right-aligned to an explicit width so the header and
    // the rows line up, including the `{}/{}` "found" column whose width is
    // chosen to fit the longest `found/total` pair.
    const POLICY_W: usize = 6;
    const TARGET_W: usize = 7;
    const FOUND_W: usize = 17;
    const ELAPSED_W: usize = 10;
    const NUM_W: usize = 11;
    println!();
    println!("  [3/3] results");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:<POLICY_W$} {:<TARGET_W$} {:>FOUND_W$} {:>ELAPSED_W$} {:>NUM_W$} {:>NUM_W$} {:>NUM_W$} {:>NUM_W$}",
        "policy", "target", "found", "elapsed", "majflt", "disk read", "extents", "prefetch"
    );
    for r in &rows {
        println!(
            "  {:<POLICY_W$} {:<TARGET_W$} {:>FOUND_W$} {:>ELAPSED_W$} {:>NUM_W$} {:>NUM_W$} {:>NUM_W$} {:>NUM_W$}",
            r.policy,
            r.target,
            format!("{}/{}", r.found, r.total),
            format!("{:.2?}", r.elapsed),
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
             PAYLOAD={} PASSES={} ELAPSED_US={:.1} MAJFLT={} MINFLT={} SYSCALLS={} \
             DISK_READ_BYTES={} EXTENTS={} PREFETCH_BYTES={} SKIPPED={} EVICTED={}",
            r.policy,
            r.target,
            r.found,
            r.total,
            records,
            target_count,
            payload_len,
            passes,
            r.elapsed.as_secs_f64() * 1e6,
            fmt_opt(r.major_faults),
            fmt_opt(r.minor_faults),
            fmt_opt(r.read_syscalls),
            fmt_opt(r.disk_read_bytes),
            r.extents,
            r.prefetch_bytes,
            r.skipped,
            r.evicted.as_str(),
        );
    }

    // A ratio is only evidence if the baseline read actually left RAM.
    // `read_bytes` (block layer) is the trustworthy signal, not major
    // faults: once a sequential pattern triggers readahead, later pages
    // arrive with no demand fault at all, so `majflt` stays near zero while
    // real device I/O happens.
    //
    // The gate is the sparse `off` row only. The dense set is a small
    // contiguous prefix, so its device bytes are legitimately a tiny
    // fraction of the store and can sit below the floor even on a fully cold
    // read -- checking it would suppress a good run. Dense is reported for
    // completeness, not used as a gate.
    //
    // Two checks, because the byte floor alone is not enough:
    //   * too few bytes -> eviction under-delivered and part of the read was
    //     served from cache, so the timing is meaningless. Require at least
    //     half of one 4 KiB page per target (a deliberately loose lower
    //     bound on the true page count).
    //   * too many bytes -> the read moved far more than the whole store,
    //     which is re-reads/scan amplification, not a clean cold read. Warn
    //     rather than pass silently.
    // The bytes-vs-store fraction is printed either way; it is the number
    // that shows at a glance whether a run was a full scan.
    const PAGE_SIZE: u64 = 4096;
    let off_sparse = rows
        .iter()
        .find(|r| r.policy == "off" && r.target == "sparse");
    let off_bytes = off_sparse.and_then(|r| r.disk_read_bytes).unwrap_or(0);
    let floor = off_sparse.map_or(0, |r| r.total as u64 * PAGE_SIZE / 2);
    let fraction = if store_bytes == 0 {
        None
    } else {
        Some(off_bytes as f64 / store_bytes as f64)
    };
    let off_cold = off_sparse.is_some() && off_bytes >= floor;
    if off_cold {
        if let Some(fraction) = fraction {
            println!();
            println!(
                "  sparse `off` read {} off the device = {fraction:.2}× the {} store",
                fmt_bytes(off_bytes),
                fmt_bytes(store_bytes),
            );
        }
        if store_bytes > 0 && off_bytes > store_bytes.saturating_mul(2) {
            println!();
            println!("  ⚠ sparse `off` read more than twice the store — likely re-reads or");
            println!("    scan amplification rather than a clean single scan; treat the");
            println!("    ratio with caution.");
        }
        for target in ["dense", "sparse"] {
            let off = rows
                .iter()
                .find(|r| r.policy == "off" && r.target == target);
            let hdd = rows
                .iter()
                .find(|r| r.policy == "hdd" && r.target == target);
            if let (Some(off), Some(hdd)) = (off, hdd) {
                let speedup = off.elapsed.as_secs_f64() / hdd.elapsed.as_secs_f64();
                println!("bench: read_plan_ratio TARGET={target} HDD_VS_OFF={speedup:.3}");
            }
        }
        if off_sparse.is_some_and(|r| r.major_faults.unwrap_or(0) == 0) {
            println!();
            println!("  note: major faults are 0 even though device bytes are non-zero —");
            println!("        readahead is serving the pages, which is expected here.");
        }
    } else {
        println!();
        println!("  ✗ COLD READ NOT ACHIEVED — ratios suppressed.");
        println!("    The sparse `off` row read less than half of one page per target");
        println!("    off the device, so at least part of the read was served from");
        println!("    cache and any difference is noise.");
        println!("    Point MTXDB_BENCH_ROOT at a real (non-tmpfs) disk and run as");
        println!("    root so drop_page_caches() can evict, or use vmtouch with the");
        println!("    store closed (already done here) and no live mappings.");
    }

    let _ = fs::remove_dir_all(&dir);
}

fn main() {
    let records = env_usize("MTXDB_READ_PLAN_RECORDS", 200_000);
    let target_count = env_usize("MTXDB_READ_PLAN_TARGETS", 100_000);
    let payload_len = env_usize("MTXDB_READ_PLAN_PAYLOAD", 1024);
    let passes = env_usize("MTXDB_READ_PLAN_PASSES", 3).max(1);
    let manual_drop = std::env::var_os("MTXDB_READ_PLAN_MANUAL_DROP").is_some();

    // Refuse a RAM-backed scratch dir: on tmpfs the data never leaves RAM,
    // and neither drop_caches nor vmtouch can evict it — every number would
    // be warm no matter how the rest is wired.
    let root = bench_root();
    if is_ram_backed(&root) {
        eprintln!("✗ REFUSING TO RUN: scratch dir is on an in-memory filesystem.");
        eprintln!("    dir:   {}", root.display());
        eprintln!(
            "    fstype: {}",
            mount_fstype(&root).unwrap_or_else(|| "unknown".to_owned())
        );
        eprintln!("  A cold read is impossible there. Point MTXDB_BENCH_ROOT at a");
        eprintln!("  real disk, e.g.:");
        eprintln!("    MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \\");
        eprintln!("      cargo bench --bench read_plan");
        let _ = fs::remove_dir_all(&root);
        std::process::exit(1);
    }

    if !manual_drop && !root_drop_available() && !vmtouch_on_path() {
        eprintln!("⚠ WARNING: no eviction available — reads will be WARM.");
        eprintln!("  Run as root (for drop_page_caches), install vmtouch, or set");
        eprintln!("  MTXDB_READ_PLAN_MANUAL_DROP=1.");
    }

    run(records, target_count, payload_len, manual_drop, passes);
}
