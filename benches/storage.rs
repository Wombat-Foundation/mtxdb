#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use mtxdb::cache::NodeCache;
use mtxdb::packfile_storage::PackfileStorage;
use mtxdb::storage::{NodeData, NodeId, NodeRef, StorageEngine};

const ROOM: [u8; 16] = [0xAB; 16];

// ── Minimal xorshift64 PRNG ─────────────────────────────────────────

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

    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Well-mixed 64-bit permutation (splitmix64). Real content-address hashes
/// (BLAKE2b/SHA-256) are uniformly distributed, so the synthetic node IDs
/// must be too — otherwise the lossy index's `hash[..8]` bucket selection
/// collapses every entry into one linear-probe cluster.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ── Synthetic Matrix room DAG generator ─────────────────────────────

struct DagGenerator {
    prev_events: Vec<Vec<usize>>,
    auth_events: Vec<Vec<usize>>,
    tips: Vec<usize>,
}

impl DagGenerator {
    fn generate(total_events: usize, fork_prob: f64, join_depth: usize) -> Self {
        let mut prev_events = Vec::with_capacity(total_events);
        let mut auth_events = Vec::with_capacity(total_events);
        let mut tips: Vec<usize> = vec![0];
        let mut rng = Rng::new(0xDEAD_BEEF);
        let mut pending_joins: Vec<Vec<usize>> = Vec::new();

        prev_events.push(vec![]);
        auth_events.push(vec![]);

        for i in 1..total_events {
            let mut extra_prevs = Vec::new();
            if !pending_joins.is_empty() && (i % join_depth == 0 || tips.len() >= 8) {
                extra_prevs = pending_joins.remove(0);
            }

            let tip_idx = if tips.len() == 1 { 0 } else { tips.len() - 1 };
            let parent = tips[tip_idx];

            let mut prev = vec![parent];
            prev.extend(extra_prevs);
            prev_events.push(prev);
            auth_events.push(vec![0]);
            tips[tip_idx] = i;

            if rng.f64() < fork_prob && tips.len() < 8 {
                tips.push(i);
            }

            if tips.len() > 4 {
                let orphan = tips.remove(0);
                pending_joins.push(vec![orphan]);
            }
        }

        for orphan_chain in pending_joins {
            if let Some(last) = prev_events.last_mut() {
                last.extend(orphan_chain);
            }
        }

        Self {
            prev_events,
            auth_events,
            tips,
        }
    }

    fn len(&self) -> usize {
        self.prev_events.len()
    }

    fn node_id(idx: usize) -> NodeId {
        let mut id = [0u8; 16];
        let a = splitmix64(idx as u64);
        let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
        id[..8].copy_from_slice(&a.to_le_bytes());
        id[8..].copy_from_slice(&b.to_le_bytes());
        id
    }

    fn node_data(&self, idx: usize) -> (NodeId, NodeData) {
        let id = Self::node_id(idx);
        let prev: Vec<NodeId> = self.prev_events[idx]
            .iter()
            .map(|&i| Self::node_id(i))
            .collect();
        let auth: Vec<NodeId> = self.auth_events[idx]
            .iter()
            .map(|&i| Self::node_id(i))
            .collect();

        let prev_len = u16::try_from(prev.len()).unwrap_or(u16::MAX);
        let auth_len = u16::try_from(auth.len()).unwrap_or(u16::MAX);
        let cap = 14 + prev.len() * 16 + auth.len() * 16;
        let mut bytes = Vec::with_capacity(cap);
        bytes.extend_from_slice(b"mxdu");
        bytes.extend_from_slice(&(idx as u64).to_le_bytes());
        bytes.extend_from_slice(&prev_len.to_le_bytes());
        for id in &prev {
            bytes.extend_from_slice(id);
        }
        bytes.extend_from_slice(&auth_len.to_le_bytes());
        for id in &auth {
            bytes.extend_from_slice(id);
        }

        let mut data = NodeData::new(bytes::Bytes::from(bytes));
        data.children = prev.into_iter().map(NodeRef::Lazy).collect();
        (id, data)
    }

    fn parse_children(bytes: &[u8]) -> Vec<NodeId> {
        if bytes.len() < 14 {
            return vec![];
        }
        let prev_count = u16::from_le_bytes([bytes[12], bytes[13]]) as usize;
        let mut children = Vec::with_capacity(prev_count);
        let mut offset = 14;
        for _ in 0..prev_count {
            if offset + 16 > bytes.len() {
                break;
            }
            let mut id = [0u8; 16];
            id.copy_from_slice(&bytes[offset..offset + 16]);
            children.push(id);
            offset += 16;
        }
        children
    }

    fn traversal_order(&self) -> Vec<usize> {
        let mut visited = vec![false; self.prev_events.len()];
        let mut order = Vec::with_capacity(self.prev_events.len());
        let mut queue: Vec<usize> = self.tips.clone();

        while let Some(idx) = queue.pop() {
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            order.push(idx);
            for &prev in &self.prev_events[idx] {
                if !visited[prev] {
                    queue.push(prev);
                }
            }
        }
        order
    }

    fn total_edge_refs(&self) -> usize {
        self.prev_events.iter().map(Vec::len).sum()
    }
}

// ── I/O measurement via /proc/self/io ───────────────────────────────

#[derive(Default, Clone, Copy)]
struct IoStats {
    rchar: u64,
    read_bytes: u64,
    #[allow(dead_code)]
    write_bytes: u64,
    syscr: u64,
}

impl IoStats {
    fn read_now() -> Self {
        let content = fs::read_to_string("/proc/self/io").unwrap_or_default();
        let mut stats = Self::default();
        for line in content.lines() {
            if let Some(v) = line.strip_prefix("rchar: ") {
                stats.rchar = v.trim().parse().unwrap_or(0);
            }
            if let Some(v) = line.strip_prefix("read_bytes: ") {
                stats.read_bytes = v.trim().parse().unwrap_or(0);
            }
            if let Some(v) = line.strip_prefix("write_bytes: ") {
                stats.write_bytes = v.trim().parse().unwrap_or(0);
            }
            if let Some(v) = line.strip_prefix("syscr: ") {
                stats.syscr = v.trim().parse().unwrap_or(0);
            }
        }
        stats
    }
}

fn pack_dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "pack") {
                total += entry.metadata().map_or(0, |m| m.len());
            }
        }
    }
    total
}

// ── Benchmark harness ───────────────────────────────────────────────

fn run_benchmark(label: &str, total_events: usize, cache_entries: usize) {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = PathBuf::from(home).join(format!("bmdb_bench_{label}_{total_events}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let dag = DagGenerator::generate(total_events, 0.15, 10);
    let traversal = dag.traversal_order();
    let node_ids: Vec<NodeId> = traversal
        .iter()
        .map(|&i| DagGenerator::node_id(i))
        .collect();

    // ── Write phase ──
    let store = PackfileStorage::open_with_cache(
        dir.clone(),
        NodeCache::new(cache_entries),
        Some(DagGenerator::parse_children),
    )
    .unwrap();

    let t_write = Instant::now();
    for i in 0..dag.len() {
        let (id, data) = dag.node_data(i);
        store.put(&ROOM, &id, &data).unwrap();
    }
    let write_elapsed = t_write.elapsed();
    let pack_size = pack_dir_size(&dir);

    // ── Clear cache to force packfile re-reads ──
    store.cache().clear();

    // ── Read phase: backward traversal simulating /sync ──
    let io_before = IoStats::read_now();
    let t_read = Instant::now();

    let mut get_calls = 0u64;
    let mut get_found = 0u64;
    let mut get_not_found = 0u64;

    for id in node_ids.iter() {
        get_calls += 1;
        match store.get(&ROOM, id) {
            Ok(Some(_)) => get_found += 1,
            Ok(None) => get_not_found += 1,
            Err(_) => get_not_found += 1,
        }
    }

    let cache_hits = store.cache().hits();
    let cache_misses = store.cache().misses();

    let read_elapsed = t_read.elapsed();
    let io_after = IoStats::read_now();
    let logical_reads = io_after.rchar.saturating_sub(io_before.rchar);
    let disk_reads = io_after.read_bytes.saturating_sub(io_before.read_bytes);
    let read_syscalls = io_after.syscr.saturating_sub(io_before.syscr);

    let cache_total = cache_hits + cache_misses;
    let hit_rate = if cache_total > 0 {
        (cache_hits as f64 / cache_total as f64) * 100.0
    } else {
        0.0
    };
    let index_loss_rate = if get_calls > 0 {
        (get_not_found as f64 / get_calls as f64) * 100.0
    } else {
        0.0
    };
    let avg_edges = dag.total_edge_refs() as f64 / total_events as f64;

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  {label}: {total_events} events, cache={cache_entries} entries");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Pack size:           {:.2} MB", pack_size as f64 / 1e6);
    eprintln!(
        "  Write throughput:    {:.0} events/sec",
        total_events as f64 / write_elapsed.as_secs_f64()
    );
    eprintln!("  Avg edges/event:    {avg_edges:.2}");
    eprintln!("  Total edge refs:    {}", dag.total_edge_refs());
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Metric A: I/O (cache cleared, packfile re-reads from disk)");
    eprintln!("    rchar (logical):   {logical_reads} bytes");
    eprintln!("    read_bytes (disk): {disk_reads} bytes");
    eprintln!("    read syscalls:     {read_syscalls}");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Metric B: Cache efficiency (of nodes that reach cache)");
    eprintln!("    Cache lookups:     {cache_total}");
    eprintln!("    Cache hits:        {cache_hits} ({hit_rate:.1}%)");
    eprintln!("    Cache misses:      {cache_misses}");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Index accuracy (lossy fanout)");
    eprintln!("    Total get() calls: {get_calls}");
    eprintln!("    Found by index:    {get_found}");
    eprintln!("    Lost to collision: {get_not_found} ({index_loss_rate:.1}%)");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Wall-clock (read):  {read_elapsed:.2?}");
    eprintln!(
        "  Throughput:         {:.0} gets/sec",
        get_calls as f64 / read_elapsed.as_secs_f64()
    );
    eprintln!();

    let _ = fs::remove_dir_all(&dir);
}

fn main() {
    eprintln!("mdb benchmark harness — cold-read measurement");
    eprintln!("Note: PackGeneration::Drop deletes packfiles on drop,");
    eprintln!("so we clear the cache and re-read from open packfiles.");
    eprintln!();

    run_benchmark("small", 1_000, 2_000);
    run_benchmark("medium", 10_000, 500);
    run_benchmark("large", 100_000, 2_000);
    run_benchmark("pressure", 100_000, 100);

    // ── Connectivity check ──
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  GRAPH CONNECTIVITY DIAGNOSTIC");
    eprintln!("═══════════════════════════════════════════════════════════════");
    let dag_check = DagGenerator::generate(100_000, 0.15, 10);
    let traversal_check = dag_check.traversal_order();
    let reachable = traversal_check.len();
    let total = dag_check.len();
    let pct = reachable as f64 / total as f64 * 100.0;
    eprintln!("  Total events:      {total}");
    eprintln!("  Reachable from tips: {reachable} ({pct:.1}%)");
    eprintln!("  Tips:              {}", dag_check.tips.len());
    eprintln!("═══════════════════════════════════════════════════════════════");

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  DECISION MATRIX");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  If physical seeks > 6 on 'large':  → Concurrent Frontier I/O");
    eprintln!("  If cache miss ratio > 80%:          → DAG Sidecar (edge packing)");
    eprintln!("  If seeks < 4 AND hit rate > 90%:   → Ready for integration");
    eprintln!("═══════════════════════════════════════════════════════════════");
}
