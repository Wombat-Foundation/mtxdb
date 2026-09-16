//! End-to-end scan benchmark: write → sync → bounded vs exhaustive scan.
//!
//! Measures the real cost difference between:
//! - streaming a full shard scan (old behavior)
//! - bounded scan that stops after N matches (new behavior)
//! - collection-filtered scan that skips unrelated shards
//!
//! Run via `cargo bench --bench scan` from the `benches/` directory. The
//! default run is intentionally limited to the small and medium datasets;
//! set `MTXDB_BENCH_SCAN_FULL=1` to include the large and multi-shard cases.
//!
//! On a machine where the repository is on a removable or encrypted mount,
//! isolate build and benchmark I/O with, for example:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/mtxdb-cargo-target \
//! MTXDB_BENCH_ROOT=/tmp/mtxdb-scan-data \
//! timeout 90s cargo bench --manifest-path benches/Cargo.toml --bench scan
//! ```
//!
//! The dataset sizes can be overridden with `MTXDB_BENCH_SCAN_COLLECTIONS`,
//! `MTXDB_BENCH_SCAN_RECORDS_PER_COLLECTION`, and
//! `MTXDB_BENCH_SCAN_BOUNDED_LIMIT`. The multi-shard threshold can be
//! overridden with `MTXDB_BENCH_SCAN_MAX_SHARD_BYTES`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::collections::HashSet;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use mtxdb_core::packfile::scan_packfile_iter;
use mtxdb_core::storage::{NodeData, StorageEngine};
use mtxdb_core::{DatabaseLayout, PackfileStorage, ShardPool, ShardType};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn node_id(collection_idx: usize, record_idx: usize) -> [u8; 16] {
    let mut id = [0u8; 16];
    let a = splitmix64((collection_idx as u64) << 32 | record_idx as u64);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

fn collection_id(idx: usize) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&(idx as u64).to_le_bytes());
    id
}

// ── Temp directory ─────────────────────────────────────────────────

static BENCH_ROOT: OnceLock<std::path::PathBuf> = OnceLock::new();
static RUN: AtomicU64 = AtomicU64::new(0);

struct ScratchDir {
    path: std::path::PathBuf,
}

impl ScratchDir {
    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn bench_root() -> std::path::PathBuf {
    BENCH_ROOT
        .get_or_init(|| {
            let base = std::env::var_os("MTXDB_BENCH_ROOT")
                .map_or_else(std::env::temp_dir, std::path::PathBuf::from);
            fs::create_dir_all(&base).unwrap();
            base
        })
        .clone()
}

fn fresh_dir(label: &str) -> ScratchDir {
    let base = bench_root();
    let token = RUN.fetch_add(1, Ordering::Relaxed);
    let dir = base.join(format!("mtxdb_scan_bench_{label}_{token}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    ScratchDir { path: dir }
}

// ── Benchmark harness ──────────────────────────────────────────────

struct ScanBenchConfig {
    collections: usize,
    records_per_collection: usize,
    bounded_limit: usize,
}

fn run_scan_bench(config: &ScanBenchConfig) {
    run_scan_bench_inner(config, None);
}

fn run_scan_bench_sharded(config: &ScanBenchConfig, max_shard_bytes: u64) {
    run_scan_bench_inner(config, Some(max_shard_bytes));
}

fn run_scan_bench_inner(config: &ScanBenchConfig, max_shard_bytes: Option<u64>) {
    let root = fresh_dir("scan");
    let ScanBenchConfig {
        collections,
        records_per_collection,
        bounded_limit,
    } = *config;

    let total_records = collections * records_per_collection;

    eprintln!(
        "  setup: {collections} collections × {records_per_collection} records = {total_records} total"
    );

    // ── Initialize database layout ──────────────────────────────────
    let layout = DatabaseLayout::open(root.path().to_owned()).unwrap();
    let pool_dir = layout.pool_dir(ShardType::EventDag).unwrap();

    // ── Write phase ─────────────────────────────────────────────────
    let store = if let Some(max_bytes) = max_shard_bytes {
        PackfileStorage::open_with_max_shard_bytes(pool_dir.clone(), max_bytes).unwrap()
    } else {
        PackfileStorage::open_with_cache(pool_dir.clone(), 2_000).unwrap()
    };

    let t_write = Instant::now();
    for col in 0..collections {
        let cid = collection_id(col);
        let entries: Vec<([u8; 16], NodeData)> = (0..records_per_collection)
            .map(|r| {
                let id = node_id(col, r);
                let mut bytes = Vec::with_capacity(32);
                bytes.extend_from_slice(b"scnb");
                bytes.extend_from_slice(&(r as u64).to_le_bytes());
                bytes.extend_from_slice(&format!("record-{col}-{r}").into_bytes());
                let mut data = NodeData::new(bytes::Bytes::from(bytes));
                data.children = vec![];
                (id, data)
            })
            .collect();
        store.put_many(&cid, &entries).unwrap();
    }
    store.sync_all().unwrap();
    drop(store);
    let write_elapsed = t_write.elapsed();
    let pack_size: u64 = fs::read_dir(&pool_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "pack"))
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum();
    eprintln!(
        "  write: {write_elapsed:.2?}  disk: {:.2} MiB",
        pack_size as f64 / 1048576.0
    );

    // ── Enumerate shards ────────────────────────────────────────────
    let pool = ShardPool::open_read_only(pool_dir.clone()).unwrap();
    let mut shards = pool.all_shards();
    shards.sort_unstable_by_key(|(_, shard)| shard.pack_id);
    let shard_count = shards.len();
    eprintln!("  shards: {shard_count}");

    // ── Collection-filtered shard set ───────────────────────────────
    let target_collection = collection_id(0);
    let collection_packs: Option<HashSet<u64>> =
        PackfileStorage::collection_shards_from_disk(&pool_dir)
            .and_then(|mut cols| cols.remove(&target_collection))
            .map(|packs| packs.into_iter().collect());
    let filtered_shard_count = collection_packs
        .as_ref()
        .map(|cp| {
            shards
                .iter()
                .filter(|(_, s)| cp.contains(&s.pack_id))
                .count()
        })
        .unwrap_or(shard_count);
    eprintln!("  target collection shards: {filtered_shard_count}/{shard_count}");

    // ── Benchmark 1: Full scan (all shards, all records) ────────────
    let mut full_scan_count = 0usize;
    let t_full = Instant::now();
    for (_, shard) in &shards {
        let scanner = scan_packfile_iter(&shard.path, false).unwrap();
        for record in scanner {
            let _ = record.unwrap();
            full_scan_count += 1;
        }
    }
    let full_elapsed = t_full.elapsed();
    eprintln!("  full scan:     {full_elapsed:.2?}  ({full_scan_count} records)");

    // ── Benchmark 2: Bounded scan (all shards, stop at N) ───────────
    let mut bounded_count = 0usize;
    let t_bounded = Instant::now();
    'outer: for (_, shard) in &shards {
        let scanner = scan_packfile_iter(&shard.path, false).unwrap();
        for record in scanner {
            let (cid, _, _) = record.unwrap();
            if cid == target_collection {
                bounded_count += 1;
                if bounded_count >= bounded_limit {
                    break 'outer;
                }
            }
        }
    }
    let bounded_elapsed = t_bounded.elapsed();
    eprintln!(
        "  bounded scan:  {bounded_elapsed:.2?}  ({bounded_count} matches, limit {bounded_limit})"
    );

    // ── Benchmark 3: Collection-filtered + bounded ──────────────────
    let mut filtered_count = 0usize;
    let t_filtered = Instant::now();
    'filtered: for (_, shard) in &shards {
        if let Some(ref cp) = collection_packs {
            if !cp.contains(&shard.pack_id) {
                continue;
            }
        }
        let scanner = scan_packfile_iter(&shard.path, false).unwrap();
        for record in scanner {
            let (cid, _, _) = record.unwrap();
            if cid == target_collection {
                filtered_count += 1;
                if filtered_count >= bounded_limit {
                    break 'filtered;
                }
            }
        }
    }
    let filtered_elapsed = t_filtered.elapsed();
    eprintln!("  filtered scan: {filtered_elapsed:.2?}  ({filtered_count} matches)");

    // ── Benchmark 4: Full scan with CRC verification ────────────────
    let mut verified_count = 0usize;
    let t_verified = Instant::now();
    for (_, shard) in &shards {
        let scanner = scan_packfile_iter(&shard.path, true).unwrap();
        for record in scanner {
            let _ = record.unwrap();
            verified_count += 1;
        }
    }
    let verified_elapsed = t_verified.elapsed();
    eprintln!("  verified scan: {verified_elapsed:.2?}  ({verified_count} records, CRC checked)");

    // ── Summary ─────────────────────────────────────────────────────
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  SCAN BENCHMARK RESULTS");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  collections:          {collections}");
    eprintln!("  records/collection:   {records_per_collection}");
    eprintln!("  total records:        {total_records}");
    eprintln!("  shards:               {shard_count}");
    eprintln!("  collection shards:    {filtered_shard_count}");
    eprintln!("  bounded limit:        {bounded_limit}");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  full scan:            {full_elapsed:.2?}");
    eprintln!(
        "  bounded scan:         {bounded_elapsed:.2?}  ({})",
        speedup(full_elapsed, bounded_elapsed)
    );
    eprintln!(
        "  filtered scan:        {filtered_elapsed:.2?}  ({})",
        speedup(full_elapsed, filtered_elapsed)
    );
    eprintln!(
        "  verified scan:        {verified_elapsed:.2?}  ({})",
        speedup(full_elapsed, verified_elapsed)
    );
    eprintln!("  ───────────────────────────────────────────────────────────");

    let ratio = bounded_elapsed.as_secs_f64() / full_elapsed.as_secs_f64();
    if ratio < 0.5 {
        eprintln!("  ✓ bounded scan is significantly faster than full scan");
    } else if ratio < 0.8 {
        eprintln!("  ~ bounded scan is moderately faster than full scan");
    } else {
        eprintln!("  ✗ bounded scan is not significantly faster (small dataset?)");
    }
    eprintln!("═══════════════════════════════════════════════════════════════");

    drop(shards);
    drop(pool);
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {value:?}"))
        })
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {value:?}"))
        })
        .unwrap_or(default)
}

fn configured(config: ScanBenchConfig) -> ScanBenchConfig {
    ScanBenchConfig {
        collections: env_usize("MTXDB_BENCH_SCAN_COLLECTIONS", config.collections),
        records_per_collection: env_usize(
            "MTXDB_BENCH_SCAN_RECORDS_PER_COLLECTION",
            config.records_per_collection,
        ),
        bounded_limit: env_usize("MTXDB_BENCH_SCAN_BOUNDED_LIMIT", config.bounded_limit),
    }
}

fn speedup(full: Duration, partial: Duration) -> String {
    if partial.as_nanos() == 0 {
        return "∞x".to_owned();
    }
    let ratio = full.as_secs_f64() / partial.as_secs_f64();
    format!("{ratio:.1}x")
}

fn main() {
    eprintln!("scan benchmark — bounded vs exhaustive packfile scanning");
    eprintln!();

    // Small dataset: demonstrates the fixed overhead dominates
    eprintln!("── small dataset ──");
    run_scan_bench(&configured(ScanBenchConfig {
        collections: 5,
        records_per_collection: 1_000,
        bounded_limit: 10,
    }));
    eprintln!();

    // Medium dataset: realistic shard layout with interleaved collections
    eprintln!("── medium dataset ──");
    run_scan_bench(&configured(ScanBenchConfig {
        collections: 20,
        records_per_collection: 5_000,
        bounded_limit: 50,
    }));

    if std::env::var("MTXDB_BENCH_SCAN_FULL").as_deref() != Ok("1") {
        return;
    }
    eprintln!();

    // Large dataset: stresses the streaming path
    eprintln!("── large dataset ──");
    run_scan_bench(&configured(ScanBenchConfig {
        collections: 50,
        records_per_collection: 10_000,
        bounded_limit: 100,
    }));
    eprintln!();

    // Multi-shard: forces shard rotation at 1 MiB to show shard-skipping benefit
    eprintln!("── multi-shard (1 MiB shards) ──");
    run_scan_bench_sharded(
        &configured(ScanBenchConfig {
            collections: 20,
            records_per_collection: 5_000,
            bounded_limit: 50,
        }),
        env_u64("MTXDB_BENCH_SCAN_MAX_SHARD_BYTES", 1024 * 1024),
    );
}
