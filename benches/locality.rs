//! Stage 1 locality benchmark: measures the I/O cost of data dispersion.
//!
//! Ingests a Zipfian-distributed workload across many collections, then
//! benchmarks a single elephant collection's point lookups before and
//! after a batch repack — isolating whether physically clustering a
//! hot collection's records into fewer packfiles actually reduces I/O.
//!
//! Run with `cargo bench --bench locality`. Requires `vmtouch` on `$PATH`
//! for cold-cache eviction; gracefully degrades if absent.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::fs;
use std::path::Path;
use std::time::Instant;

use bytes::Bytes;
use mtxdb_core::packfile::{self, storage::PackfileStorage};
use mtxdb_core::storage::{NodeData, NodeId, StorageEngine};

const MAX_DATA_LEN: usize = 65535 - 100;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn node_id(idx: usize) -> NodeId {
    let mut id = [0u8; 16];
    let a = splitmix64(idx as u64);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

// ── Opaque payload generator ───────────────────────────────────────

struct PseudoRandomPayload {
    state: u64,
}

impl PseudoRandomPayload {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn generate_bytes(&mut self, len: usize) -> Vec<u8> {
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

// ── Zipfian collection sampler ─────────────────────────────────────

struct ZipfianCollectionSampler {
    collections: Vec<[u8; 16]>,
    elephant_idx: usize,
    weights: Vec<f64>,
}

impl ZipfianCollectionSampler {
    fn new(count: usize) -> Self {
        let collections: Vec<[u8; 16]> = (0..count as u64)
            .map(|i| {
                let mut id = [0u8; 16];
                id[..8].copy_from_slice(&i.to_le_bytes());
                id
            })
            .collect();

        // Zipfian weights: collection 0 is the "elephant" with weight ~1.0,
        // each subsequent collection gets weight 1/k^s (s=1.0 for mild skew).
        let weights: Vec<f64> = (0..count as u64)
            .map(|k| if k == 0 { 1.0 } else { 1.0 / (k as f64) })
            .collect();
        let total: f64 = weights.iter().sum();

        Self {
            collections,
            elephant_idx: 0,
            weights: weights.iter().map(|w| w / total).collect(),
        }
    }

    fn sample(&self, idx: usize) -> [u8; 16] {
        // A single xorshift step from a seed that only differs in its low
        // bits barely perturbs the high bits `f64()` extracts (>>11), so a
        // plain `Rng::new(idx ^ const).f64()` here returned ~0.1986 for
        // almost every idx — the elephant collection's ~12.8% share of the
        // Zipfian mass never won a draw, so the whole locality signal this
        // benchmark exists to measure never fired. `splitmix64` mixes bits
        // properly in one pass, so use it directly instead.
        let r = (splitmix64(idx as u64 ^ 0xCAFE_BABE) >> 11) as f64 / (1u64 << 53) as f64;
        let mut cumulative = 0.0;
        for (i, w) in self.weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                return self.collections[i];
            }
        }
        self.collections[self.elephant_idx]
    }

    fn elephant_id(&self) -> [u8; 16] {
        self.collections[self.elephant_idx]
    }
}

// ── OS-level I/O counters ───────────────────────────────────────────

/// Point-in-time snapshot of this process's page-fault, read-syscall, and
/// block-layer I/O counters, used to approximate "random seeks" against
/// the mmap-backed packfiles. Reads here go through `mmap`, not
/// `pread`/`read`, so a major page fault — the kernel pulling a page in
/// from the backing file because it wasn't already resident — is one
/// analog this engine has to a physical random seek; `read_syscalls` is
/// reported alongside it mostly to show it stays flat (proving the mmap
/// path really is the one doing the work).
///
/// `major_faults` alone is noisier than it looks: once a sequential
/// access pattern triggers one major fault, the kernel's readahead often
/// prefetches the next several pages in the background, so *those*
/// pages get classified as minor faults on touch even though the bytes
/// still came off disk within the same access — readahead reclassifies
/// real disk I/O from major to minor, it doesn't eliminate it. That
/// makes `major_faults` swing between runs depending on exact readahead
/// timing. `disk_read_bytes` (`/proc/self/io`'s `read_bytes`) doesn't
/// have that ambiguity: the block layer only increments it when bytes
/// are actually fetched from the storage device, regardless of which
/// fault (or none, for a readahead-only page) pulled them in — so it's
/// the more trustworthy "did we really hit disk" number of the two.
///
/// Linux-only (`/proc/self/stat`, `/proc/self/io`); `capture` returns
/// `None` everywhere else.
#[derive(Debug, Clone, Copy, Default)]
struct IoSnapshot {
    major_faults: u64,
    minor_faults: u64,
    read_syscalls: u64,
    disk_read_bytes: u64,
}

impl IoSnapshot {
    #[cfg(target_os = "linux")]
    fn capture() -> Option<Self> {
        // `fs::read_to_string` grows its buffer in doubling steps (32,
        // 32, 64, 128, 256...) plus a final zero-byte EOF read — that's
        // ~6 `read()` syscalls to slurp a ~200-byte procfs file, which
        // swamps the very syscall count this function exists to measure
        // (confirmed with `strace`: a single `capture()` call was
        // responsible for essentially the entire "10-11 read syscalls"
        // reported around a query loop that itself makes none — all
        // record reads go through `mmap`, not `read`/`pread`). A single
        // fixed 1 KiB buffer comfortably fits both files' content in one
        // `read()` call each, so `capture()`'s own footprint no longer
        // dominates whatever it's trying to observe.
        fn read_proc_file(path: &str) -> Option<String> {
            use std::io::Read as _;
            let mut file = fs::File::open(path).ok()?;
            let mut buf = [0u8; 1024];
            let n = file.read(&mut buf).ok()?;
            Some(String::from_utf8_lossy(&buf[..n]).into_owned())
        }

        // /proc/self/stat's `comm` field (2nd, parenthesized) can itself
        // contain spaces or parens, so locate fields by splitting after
        // the *last* ')' rather than by raw whitespace index.
        let stat = read_proc_file("/proc/self/stat")?;
        let after_comm = stat.rsplit_once(')')?.1;
        let fields: Vec<&str> = after_comm.split_whitespace().collect();
        // After the comm field, `state` is field 3 (index 0), so
        // minflt (field 10) is index 7 and majflt (field 12) is index 9.
        let minor_faults = fields.get(7)?.parse().ok()?;
        let major_faults = fields.get(9)?.parse().ok()?;

        // Best-effort: some kernels/containers lack task I/O accounting
        // (`/proc/self/io` absent or fields missing), in which case just
        // report 0 for both rather than losing the fault counts above too.
        let io_text = read_proc_file("/proc/self/io");
        let find_field = |prefix: &str| -> u64 {
            io_text
                .as_deref()
                .and_then(|io| {
                    io.lines()
                        .find_map(|line| line.strip_prefix(prefix))
                        .and_then(|v| v.trim().parse().ok())
                })
                .unwrap_or(0)
        };
        let read_syscalls = find_field("syscr:");
        let disk_read_bytes = find_field("read_bytes:");

        Some(Self {
            major_faults,
            minor_faults,
            read_syscalls,
            disk_read_bytes,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn capture() -> Option<Self> {
        None
    }

    /// Counters accumulated between an earlier snapshot and `self`.
    fn delta(self, earlier: Self) -> Self {
        Self {
            major_faults: self.major_faults.saturating_sub(earlier.major_faults),
            minor_faults: self.minor_faults.saturating_sub(earlier.minor_faults),
            read_syscalls: self.read_syscalls.saturating_sub(earlier.read_syscalls),
            disk_read_bytes: self.disk_read_bytes.saturating_sub(earlier.disk_read_bytes),
        }
    }
}

/// Column width for `row` labels. Chosen so a `row`'s value column
/// lines up with a `subrow`'s: 2-space indent + `ROW_WIDTH` equals
/// 4-space indent + `SUBROW_WIDTH`. Still comfortably fits the longest
/// row label in use, "Post-repack packs:" at 18 chars.
const ROW_WIDTH: usize = 19;
/// Column width for `subrow` labels, sized to the longest one in use
/// ("packs referenced:", 17 chars).
const SUBROW_WIDTH: usize = 17;

/// Prints a top-level `label: value` row, label left-padded to a fixed
/// column so every row's value lines up regardless of label length.
fn row(label: &str, value: impl std::fmt::Display) {
    println!("  {label:<ROW_WIDTH$} {value}");
}

/// Prints an indented `label: value` row nested under the most recent
/// `row(...)` — used for the handful of sub-measurements (disk bytes,
/// faults, syscalls) that belong to one timed window (Open or Query)
/// without repeating that window's name on every line.
fn subrow(label: &str, value: impl std::fmt::Display) {
    println!("    {label:<SUBROW_WIDTH$} {value}");
}

/// Prints one I/O snapshot delta as indented sub-rows under whichever
/// timed window (Open or Query) the caller just printed a `row` for.
fn print_io_delta(delta: Option<IoSnapshot>) {
    match delta {
        Some(d) => {
            subrow("disk read:", format_bytes(d.disk_read_bytes));
            subrow("major faults:", d.major_faults);
            subrow("minor faults:", d.minor_faults);
            subrow("read syscalls:", d.read_syscalls);
        }
        None => subrow("I/O counters:", "unavailable (Linux only)"),
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

// ── Fragmentation ───────────────────────────────────────────────────

/// Counts the number of maximal contiguous runs of `collection_id`'s own
/// records across the true physical on-disk layout of every shard in
/// `base_dir` — a sharper fragmentation signal than `Pack IDs Referenced`
/// (distinct files touched), since a collection's records can be
/// non-contiguous *within* a single shard file too: multiple collections
/// sharing one "home" shard interleave their writes at the byte level,
/// so one collection's data can show up as many small scattered runs
/// even inside a single file. Repacking should collapse many such runs
/// (e.g. hundreds, spread across many files) into one contiguous run in
/// one file.
///
/// Delegates to `mtxdb_core::packfile::layout::physical_layout` — the
/// same scan `mtxdb collections --layout`'s `runs` column uses — rather
/// than reimplementing the scan here, so both stay backed by one
/// canonical implementation instead of two that could silently drift.
fn count_segments(base_dir: &Path, collection_id: &[u8; 16]) -> u64 {
    packfile::layout::physical_layout(base_dir)
        .ok()
        .and_then(|layout| layout.collections.get(collection_id).map(|c| c.segments))
        .unwrap_or(0)
}

// ── Cache eviction ─────────────────────────────────────────────────

/// Outcome of attempting to evict `dir`'s page-cache contents via
/// `vmtouch -e`, distinguished so a caller can tell "vmtouch isn't
/// installed" apart from "vmtouch ran but failed" — both leave a "cold"
/// phase silently measuring a warm cache instead, which is worth a loud
/// warning rather than a quiet fallback.
enum Eviction {
    Evicted,
    NotFound,
    Failed(String),
}

impl Eviction {
    fn cache_state_label(&self) -> String {
        match self {
            Self::Evicted => "Cold (Evicted)".to_string(),
            Self::NotFound => "Warm (vmtouch NOT FOUND on PATH)".to_string(),
            Self::Failed(msg) => format!("Warm (vmtouch eviction FAILED: {msg})"),
        }
    }

    fn warn_if_not_evicted(&self) {
        match self {
            Self::Evicted => {}
            Self::NotFound => eprintln!(
                "  WARNING: `vmtouch` not found on PATH — this phase is measuring a WARM \
                 cache, not a cold one. Install vmtouch (e.g. `sudo apt-get install vmtouch`) \
                 for real cold-cache numbers."
            ),
            Self::Failed(msg) => eprintln!(
                "  WARNING: `vmtouch -e` failed ({msg}) — this phase is measuring a WARM \
                 cache, not a cold one."
            ),
        }
    }
}

fn drop_caches_for_dir(dir: &Path) -> Eviction {
    match std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .output()
    {
        Ok(output) if output.status.success() => Eviction::Evicted,
        Ok(output) => Eviction::Failed(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Eviction::NotFound,
        Err(e) => Eviction::Failed(e.to_string()),
    }
}

/// Whether `vmtouch` is callable on `$PATH` at all. Checked once up front
/// so a missing install is a loud banner at startup, not something a
/// reader has to notice buried in a "Cache state: Warm" line four phases in.
fn vmtouch_on_path() -> bool {
    std::process::Command::new("vmtouch")
        .arg("-h")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

// ── Benchmark ──────────────────────────────────────────────────────

fn shard_bytes_label(n: Option<u64>) -> String {
    n.map_or_else(|| "default (~256 MB)".to_string(), |n| n.to_string())
}

/// Runs the ingest → pre-repack query → repack → post-repack query cycle.
///
/// `repack_max_shard_bytes` is the shard cap the *repack* step reopens the
/// store with — normally the same as `max_shard_bytes` (the cap data was
/// ingested under), but a caller can pass a larger cap (e.g. `None` for
/// the default ~256 MB) to make the repack actually consolidate a pile of
/// tiny shards into a handful of big ones, instead of just rewriting each
/// collection into equally-tiny replacement shards.
pub fn run_stage1_locality_benchmark(
    total_records: usize,
    collection_count: usize,
    max_shard_bytes: Option<u64>,
    repack_max_shard_bytes: Option<u64>,
) {
    let temp_dir =
        std::env::temp_dir().join(format!("mtxdb_stage1_locality_{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sampler = ZipfianCollectionSampler::new(collection_count);
    let mut payload_gen = PseudoRandomPayload::new(0xDEAD_BEEF);
    let mut elephant_nodes = Vec::new();

    println!(
        "\n[1/4] Ingesting opaque Zipfian workload (max_shard_bytes={})...",
        shard_bytes_label(max_shard_bytes)
    );
    let store = match max_shard_bytes {
        Some(max_shard_bytes) => {
            PackfileStorage::open_with_max_shard_bytes(temp_dir.clone(), max_shard_bytes).unwrap()
        }
        None => PackfileStorage::open(temp_dir.clone()).unwrap(),
    };

    let all_collections: Vec<[u8; 16]> = sampler.collections.clone();

    for i in 0..total_records {
        let collection = sampler.sample(i);
        let nid = node_id(i);

        // Scale payload size to the shard cap itself when one is given, so a
        // tiny `max_shard_bytes` (e.g. a 10 KB test run) still fits several
        // records per shard instead of overflowing MAX_SHARDS on oversized
        // single-record payloads.
        let payload_ceiling = max_shard_bytes.map_or(MAX_DATA_LEN, |cap| {
            usize::try_from(cap / 8).unwrap_or(MAX_DATA_LEN).max(64)
        });
        let payload_size = (1024 + (i % 30) * 1024)
            .min(MAX_DATA_LEN)
            .min(payload_ceiling);
        let payload = payload_gen.generate_bytes(payload_size);

        if collection == sampler.elephant_id() {
            elephant_nodes.push(nid);
        }

        store
            .put(&collection, &nid, &NodeData::new(Bytes::from(payload)))
            .unwrap();
    }

    store.sync_all().unwrap();

    let initial_pack_count = store.shard_summaries().len();
    row("Rotated packs:", initial_pack_count);

    // --- Phase B: Pre-Repack Query Benchmark (Unclustered / Cold) ---
    println!("\n[2/4] Measuring PRE-REPACK query latency...");
    let sample_keys: Vec<_> = elephant_nodes
        .iter()
        .step_by((elephant_nodes.len() / 500).max(1))
        .cloned()
        .collect();

    drop(store);
    let evicted_pre = drop_caches_for_dir(&temp_dir);

    // Two separate windows, not one combined one: `open_read_only` does a
    // full, eager scan of *every* pack file to rebuild *every*
    // collection's index (see `PackfileStorage::open_with_options`) — a
    // whole-store cost that scales with total store size, not with the
    // elephant collection's own layout, so repacking one collection
    // should never be expected to move it much. Bundling that cost
    // together with the elephant-only point-lookup cost below would
    // dilute the actual locality signal this benchmark exists to show
    // inside a number dominated by something repack doesn't touch.
    let t0_io = IoSnapshot::capture();
    let t0 = Instant::now();

    let store_pre = PackfileStorage::open_read_only(temp_dir.clone()).unwrap();

    let t1_io = IoSnapshot::capture();
    let t1 = Instant::now();
    let pre_open_time = t1.duration_since(t0);
    let pre_open_io_delta = t0_io.zip(t1_io).map(|(b, a)| a.delta(b));

    let mut pre_found = 0;
    for nid in &sample_keys {
        if store_pre
            .get(&sampler.elephant_id(), nid)
            .unwrap()
            .is_some()
        {
            pre_found += 1;
        }
    }

    let t2_io = IoSnapshot::capture();
    let t2 = Instant::now();
    let pre_latency = t2.duration_since(t1);
    let pre_io_delta = t1_io.zip(t2_io).map(|(b, a)| a.delta(b));

    let pre_packs_touched = store_pre
        .collection_referenced_pack_ids(&sampler.elephant_id())
        .len();
    let pre_segments = count_segments(&temp_dir, &sampler.elephant_id());

    row("Cache state:", evicted_pre.cache_state_label());
    evicted_pre.warn_if_not_evicted();
    row(
        "Open time:",
        format!("{pre_open_time:.2?} (whole store, expect ~flat)"),
    );
    print_io_delta(pre_open_io_delta);
    row(
        "Query latency:",
        format!("{pre_latency:.2?} (elephant only)"),
    );
    subrow("packs referenced:", pre_packs_touched);
    subrow("segments:", pre_segments);
    print_io_delta(pre_io_delta);

    // --- Phase C: Batch Repack ---
    if repack_max_shard_bytes == max_shard_bytes {
        println!("\n[3/4] Executing Batch Repack...");
    } else {
        println!(
            "\n[3/4] Executing Batch Repack (compacting to max_shard_bytes={})...",
            shard_bytes_label(repack_max_shard_bytes)
        );
    }
    drop(store_pre);
    let store_repack = match repack_max_shard_bytes {
        Some(repack_max_shard_bytes) => {
            PackfileStorage::open_with_max_shard_bytes(temp_dir.clone(), repack_max_shard_bytes)
                .unwrap()
        }
        None => PackfileStorage::open(temp_dir.clone()).unwrap(),
    };
    let repack_start = Instant::now();

    store_repack
        .repack_collections_reachable(&all_collections, |_hash, _data| Vec::new())
        .unwrap();
    store_repack.sync_all().unwrap();
    let repack_time = repack_start.elapsed();

    let post_pack_count = store_repack.shard_summaries().len();
    row("Repack duration:", format!("{repack_time:.2?}"));
    row(
        "Post-repack packs:",
        format!("{post_pack_count} (was {initial_pack_count})"),
    );

    // --- Phase D: Post-Repack Query Benchmark (Clustered / Cold) ---
    println!("\n[4/4] Measuring POST-REPACK query latency...");
    drop(store_repack);
    let evicted_post = drop_caches_for_dir(&temp_dir);

    // Same open/query split and reasoning as the pre-repack window above.
    let t0_io = IoSnapshot::capture();
    let t0 = Instant::now();

    let store_post = PackfileStorage::open_read_only(temp_dir.clone()).unwrap();

    let t1_io = IoSnapshot::capture();
    let t1 = Instant::now();
    let post_open_time = t1.duration_since(t0);
    let post_open_io_delta = t0_io.zip(t1_io).map(|(b, a)| a.delta(b));

    let mut post_found = 0;
    for nid in &sample_keys {
        if store_post
            .get(&sampler.elephant_id(), nid)
            .unwrap()
            .is_some()
        {
            post_found += 1;
        }
    }

    let t2_io = IoSnapshot::capture();
    let t2 = Instant::now();
    let post_latency = t2.duration_since(t1);
    let post_io_delta = t1_io.zip(t2_io).map(|(b, a)| a.delta(b));
    let post_packs_touched = store_post
        .collection_referenced_pack_ids(&sampler.elephant_id())
        .len();
    let post_segments = count_segments(&temp_dir, &sampler.elephant_id());

    row("Cache state:", evicted_post.cache_state_label());
    evicted_post.warn_if_not_evicted();
    row(
        "Open time:",
        format!("{post_open_time:.2?} (whole store, expect ~flat)"),
    );
    print_io_delta(post_open_io_delta);
    row(
        "Query latency:",
        format!("{post_latency:.2?} (elephant only)"),
    );
    subrow("packs referenced:", post_packs_touched);
    subrow("segments:", post_segments);
    print_io_delta(post_io_delta);

    // --- Summary ---
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  STAGE 1 LOCALITY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    if repack_max_shard_bytes == max_shard_bytes {
        row("Max shard bytes:", shard_bytes_label(max_shard_bytes));
    } else {
        row(
            "Max shard bytes:",
            format!(
                "{} -> {} (compacted)",
                shard_bytes_label(max_shard_bytes),
                shard_bytes_label(repack_max_shard_bytes)
            ),
        );
    }
    row("Total records:", total_records);
    row("Collections:", collection_count);
    row("Elephant nodes:", elephant_nodes.len());
    row("Sample size:", sample_keys.len());
    println!("  ───────────────────────────────────────────────────────────");
    row(
        "Pack IDs:",
        format!("{initial_pack_count} -> {post_pack_count}"),
    );
    row(
        "Open time:",
        format!("{pre_open_time:.2?} -> {post_open_time:.2?} (whole store)"),
    );
    if let (Some(pre_open_io), Some(post_open_io)) = (pre_open_io_delta, post_open_io_delta) {
        subrow(
            "disk read:",
            format!(
                "{} -> {}",
                format_bytes(pre_open_io.disk_read_bytes),
                format_bytes(post_open_io.disk_read_bytes)
            ),
        );
        subrow(
            "minor faults:",
            format!(
                "{} -> {}",
                pre_open_io.minor_faults, post_open_io.minor_faults
            ),
        );
        subrow(
            "major faults:",
            format!(
                "{} -> {}",
                pre_open_io.major_faults, post_open_io.major_faults
            ),
        );
        subrow(
            "read syscalls:",
            format!(
                "{} -> {}",
                pre_open_io.read_syscalls, post_open_io.read_syscalls
            ),
        );
    }
    row(
        "Query latency:",
        format!("{pre_latency:.2?} -> {post_latency:.2?} (elephant only)"),
    );
    subrow(
        "packs referenced:",
        format!("{pre_packs_touched} -> {post_packs_touched}"),
    );
    subrow("segments:", format!("{pre_segments} -> {post_segments}"));
    if let (Some(pre_io), Some(post_io)) = (pre_io_delta, post_io_delta) {
        subrow(
            "disk read:",
            format!(
                "{} -> {}",
                format_bytes(pre_io.disk_read_bytes),
                format_bytes(post_io.disk_read_bytes)
            ),
        );
        subrow(
            "minor faults:",
            format!("{} -> {}", pre_io.minor_faults, post_io.minor_faults),
        );
        subrow(
            "major faults:",
            format!("{} -> {}", pre_io.major_faults, post_io.major_faults),
        );
        subrow(
            "read syscalls:",
            format!("{} -> {}", pre_io.read_syscalls, post_io.read_syscalls),
        );
    }
    row(
        "Speedup:",
        format!(
            "{:.2}x",
            pre_latency.as_secs_f64() / post_latency.as_secs_f64()
        ),
    );
    row(
        "Found:",
        format!(
            "pre={pre_found}/{} post={post_found}/{}",
            sample_keys.len(),
            sample_keys.len()
        ),
    );
    println!("═══════════════════════════════════════════════════════════════");

    let _ = fs::remove_dir_all(&temp_dir);
}

fn main() {
    println!("mtxdb stage 1 locality benchmark");
    if vmtouch_on_path() {
        println!("`vmtouch` found on PATH — cold-cache eviction enabled.");
    } else {
        eprintln!("⚠ WARNING: `vmtouch` not found on PATH.");
        eprintln!(
            "  Every \"cold\" phase below will silently run against a WARM cache instead \
             — install it (e.g. `sudo apt-get install vmtouch`) for real cold-cache numbers."
        );
    }

    // A writable ShardPool keeps every discovered pack file open for its
    // whole lifetime (see `ShardPool::open_internal`), so shard count here
    // is bounded by the process's fd ulimit, not just MAX_SHARDS (4096) —
    // a too-small max_shard_bytes with too many records blew past a
    // default 1024-fd ulimit and failed with "Too many open files" before
    // this was tuned down. 32 KB / 800 records keeps it around ~60 packs.
    println!("\n\n### Phase A: tiny shards (max_shard_bytes = 32 KB) ###");
    run_stage1_locality_benchmark(800, 20, Some(32 * 1024), Some(32 * 1024));

    println!("\n\n### Phase B: default shards (max_shard_bytes = ~256 MB) ###");
    run_stage1_locality_benchmark(50_000, 500, None, None);

    // Phase A's repack reopened the store at the *same* 32 KB cap it
    // ingested under, so it just rewrote each collection into equally
    // tiny replacement shards (122 -> 122) — it never actually
    // consolidated anything. This phase repeats Phase A's exact ingest,
    // but repacks into the default ~256 MB cap instead, so the same 122
    // tiny packs should collapse into a handful of large ones — and
    // latency/page-faults/syscalls should drop accordingly.
    println!("\n\n### Phase C: compact tiny packs into ~256 MB packs ###");
    run_stage1_locality_benchmark(800, 20, Some(32 * 1024), None);
}
