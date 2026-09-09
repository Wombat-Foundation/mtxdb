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
use mtxdb_core::packfile::storage::PackfileStorage;
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

// ── Cache eviction ─────────────────────────────────────────────────

fn drop_caches_for_dir(dir: &Path) -> bool {
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

// ── Benchmark ──────────────────────────────────────────────────────

pub fn run_stage1_locality_benchmark(
    total_records: usize,
    collection_count: usize,
    max_shard_bytes: Option<u64>,
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
        max_shard_bytes.map_or_else(|| "default (~256 MB)".to_string(), |n| n.to_string())
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
    println!("  Rotated Packs:        {initial_pack_count}");

    // --- Phase B: Pre-Repack Query Benchmark (Unclustered / Cold) ---
    println!("\n[2/4] Measuring PRE-REPACK query latency...");
    let sample_keys: Vec<_> = elephant_nodes
        .iter()
        .step_by((elephant_nodes.len() / 500).max(1))
        .cloned()
        .collect();

    drop(store);
    let evicted_pre = drop_caches_for_dir(&temp_dir);

    let store_pre = PackfileStorage::open_read_only(temp_dir.clone()).unwrap();
    let pre_start = Instant::now();
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
    let pre_latency = pre_start.elapsed();

    let pre_packs_touched = store_pre
        .collection_referenced_pack_ids(&sampler.elephant_id())
        .len();

    println!(
        "  Cache state:          {}",
        if evicted_pre {
            "Cold (Evicted)"
        } else {
            "Warm (No vmtouch)"
        }
    );
    println!("  Pre-Repack Latency:   {pre_latency:.2?}");
    println!("  Pack IDs Referenced:  {pre_packs_touched}");

    // --- Phase C: Batch Repack ---
    println!("\n[3/4] Executing Batch Repack...");
    drop(store_pre);
    let store_repack = match max_shard_bytes {
        Some(max_shard_bytes) => {
            PackfileStorage::open_with_max_shard_bytes(temp_dir.clone(), max_shard_bytes).unwrap()
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
    println!("  Repack Duration:      {repack_time:.2?}");
    println!("  Post-Repack Packs:    {post_pack_count} (was {initial_pack_count})");

    // --- Phase D: Post-Repack Query Benchmark (Clustered / Cold) ---
    println!("\n[4/4] Measuring POST-REPACK query latency...");
    drop(store_repack);
    let evicted_post = drop_caches_for_dir(&temp_dir);

    let store_post = PackfileStorage::open_read_only(temp_dir.clone()).unwrap();
    let post_start = Instant::now();
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
    let post_latency = post_start.elapsed();
    let post_packs_touched = store_post
        .collection_referenced_pack_ids(&sampler.elephant_id())
        .len();

    println!(
        "  Cache state:          {}",
        if evicted_post {
            "Cold (Evicted)"
        } else {
            "Warm (No vmtouch)"
        }
    );
    println!("  Post-Repack Latency:  {post_latency:.2?}");
    println!("  Pack IDs Referenced:  {post_packs_touched}");

    // --- Summary ---
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  STAGE 1 LOCALITY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Max shard bytes:      {}",
        max_shard_bytes.map_or_else(|| "default (~256 MB)".to_string(), |n| n.to_string())
    );
    println!("  Total records:        {total_records}");
    println!("  Collections:          {collection_count}");
    println!("  Elephant nodes:       {}", elephant_nodes.len());
    println!("  Sample size:          {}", sample_keys.len());
    println!("  ───────────────────────────────────────────────────────────");
    println!("  Pack IDs:             {initial_pack_count} -> {post_pack_count}");
    println!("  Latency:              {pre_latency:.2?} -> {post_latency:.2?}");
    println!(
        "  Speedup:              {:.2}x",
        pre_latency.as_secs_f64() / post_latency.as_secs_f64()
    );
    println!(
        "  Found:                pre={pre_found}/{} post={post_found}/{}",
        sample_keys.len(),
        sample_keys.len()
    );
    println!("═══════════════════════════════════════════════════════════════");

    let _ = fs::remove_dir_all(&temp_dir);
}

fn main() {
    println!("mtxdb stage 1 locality benchmark");
    println!("Requires `vmtouch` on PATH for cold-cache eviction.");

    // A writable ShardPool keeps every discovered pack file open for its
    // whole lifetime (see `ShardPool::open_internal`), so shard count here
    // is bounded by the process's fd ulimit, not just MAX_SHARDS (4096) —
    // a too-small max_shard_bytes with too many records blew past a
    // default 1024-fd ulimit and failed with "Too many open files" before
    // this was tuned down. 32 KB / 800 records keeps it around ~60 packs.
    println!("\n\n### Phase A: tiny shards (max_shard_bytes = 32 KB) ###");
    run_stage1_locality_benchmark(800, 20, Some(32 * 1024));

    println!("\n\n### Phase B: default shards (max_shard_bytes = ~256 MB) ###");
    run_stage1_locality_benchmark(50_000, 500, None);
}
