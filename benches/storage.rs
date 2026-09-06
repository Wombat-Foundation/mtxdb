#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::collections::{HashSet, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use mtxdb::cache::NodeCache;
use mtxdb::storage::{NodeData, NodeId, NodeRef, StorageEngine};
use mtxdb::PackfileStorage;

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

        // A small, slowly-rotating pool of "current create/power-levels/
        // join-rules" events, standing in for the handful of auth events
        // real Matrix events reference. This is deliberately shallow and
        // heavily shared — distinct in shape from prev_events, which grows
        // with every event.
        let mut auth_pool: Vec<usize> = vec![0];

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

            // Reference the room's create event plus one other, more
            // recently-rotated pool member (e.g. current power levels).
            let recent = auth_pool[rng.next_u64() as usize % auth_pool.len()];
            auth_events.push(vec![auth_pool[0], recent]);
            tips[tip_idx] = i;

            if rng.f64() < fork_prob && tips.len() < 8 {
                tips.push(i);
            }

            if tips.len() > 4 {
                let orphan = tips.remove(0);
                pending_joins.push(vec![orphan]);
            }

            // Power-levels/join-rules changes are rare; rotate the pool
            // slowly rather than on every event.
            if i % 1000 == 0 {
                auth_pool.push(i);
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

    // jscpd:ignore-start
    // False-positive match against src/repack.rs's
    // `impl std::error::Error for RepackError` — token-shape coincidence,
    // not related logic.
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
    // jscpd:ignore-end

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

// jscpd:ignore-start
// pack_dir_size and drop_caches_for_dir below false-positive-match against
// src/packfile_storage.rs's ten_record_fixture/test_batch_put_get —
// token-shape coincidence (both iterate/map over a short range), not
// related logic.
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

/// Evict every `.pack` file under `dir` from the page cache, so a
/// subsequent read is a genuine cold read rather than served from
/// already-resident pages. No root required (unlike
/// `/proc/sys/vm/drop_caches`, which also isn't scoped to one directory).
///
/// Uses `posix_fadvise(POSIX_FADV_DONTNEED)`, which only evicts *clean*
/// pages — anything not yet fsynced won't be dropped, so callers should
/// only rely on this after a write phase that has synced (or, as here,
/// after reopening the store fresh so nothing is dirty in this process).
fn drop_caches_for_dir(dir: &std::path::Path) {
    // Shells out to `vmtouch -e`, which wraps the same posix_fadvise(2)
    // eviction this needs, rather than adding a dependency (or an unsafe
    // FFI call of our own) just for one syscall. Silently does nothing if
    // vmtouch isn't installed — callers should treat this as best-effort.
    let _ = std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}
// jscpd:ignore-end

// ── Benchmark harness ───────────────────────────────────────────────

/// Measured results from one scenario run, used to drive the decision
/// matrix on real signals instead of structurally-fixed ones.
struct BenchResult {
    read_syscalls: u64,
    index_loss_rate: f64,
    warm_hit_rate: f64,
    cold_gets_per_sec: f64,
    warm_gets_per_sec: f64,
}

fn run_benchmark(label: &str, total_events: usize, cache_entries: usize) -> BenchResult {
    let dir = std::env::temp_dir().join(format!("mtxdb_bench_{label}_{total_events}"));
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

    // ── Cold read phase: backward traversal simulating /sync, cache empty ──
    let io_before = IoStats::read_now();
    let t_read = Instant::now();

    let mut get_calls = 0u64;
    let mut get_found = 0u64;
    let mut get_not_found = 0u64;
    let mut get_errors = 0u64;

    for id in node_ids.iter() {
        get_calls += 1;
        match store.get(&ROOM, id) {
            Ok(Some(_)) => get_found += 1,
            Ok(None) => get_not_found += 1,
            Err(_) => get_errors += 1,
        }
    }

    let cold_hits = store.cache().hits();
    let cold_misses = store.cache().misses();

    let read_elapsed = t_read.elapsed();
    let io_after = IoStats::read_now();
    let logical_reads = io_after.rchar.saturating_sub(io_before.rchar);
    let disk_reads = io_after.read_bytes.saturating_sub(io_before.read_bytes);
    let read_syscalls = io_after.syscr.saturating_sub(io_before.syscr);

    let cold_total = cold_hits + cold_misses;
    let cold_hit_rate = if cold_total > 0 {
        (cold_hits as f64 / cold_total as f64) * 100.0
    } else {
        0.0
    };
    let index_loss_rate = if get_calls > 0 {
        (get_not_found as f64 / get_calls as f64) * 100.0
    } else {
        0.0
    };
    let cold_gets_per_sec = get_calls as f64 / read_elapsed.as_secs_f64();

    // ── Warm read phase: same traversal, cache left populated from cold pass.
    // This is what makes cache effectiveness an actually-measured quantity:
    // hit rate here reflects genuine reuse under cache_entries capacity,
    // not a value fixed by construction (cold pass always reads a cleared
    // cache, so its hit rate is 0/N by definition and proves nothing about
    // the cache itself).
    //
    // Note node_ids has no repeats (each id is visited exactly once by the
    // traversal), so a hit here only happens if an item is still resident
    // from the cold pass when the warm pass reaches it again — which, for a
    // pure linear once-through scan, only occurs once cache_entries covers
    // the whole working set. Expect ~100% when cache_entries >= total_events
    // and ~0% otherwise: that binary split is the real finding (a cache
    // this size buys nothing on a single full scan; only cross-request
    // temporal locality — e.g. repeated /sync of the same recent range —
    // would benefit, which this harness does not model). ──
    let hits_before_warm = store.cache().hits();
    let misses_before_warm = store.cache().misses();
    let t_warm = Instant::now();

    let mut warm_found = 0u64;
    for id in node_ids.iter() {
        if let Ok(Some(_)) = store.get(&ROOM, id) {
            warm_found += 1
        }
    }

    let warm_elapsed = t_warm.elapsed();
    let warm_hits = store.cache().hits() - hits_before_warm;
    let warm_misses = store.cache().misses() - misses_before_warm;
    let warm_total = warm_hits + warm_misses;
    let warm_hit_rate = if warm_total > 0 {
        (warm_hits as f64 / warm_total as f64) * 100.0
    } else {
        0.0
    };
    let warm_gets_per_sec = warm_found as f64 / warm_elapsed.as_secs_f64();

    let avg_edges = dag.total_edge_refs() as f64 / total_events as f64;

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  {label}: {total_events} events, cache={cache_entries} entries");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  Pack size:               {:.2} MB",
        pack_size as f64 / 1e6
    );
    eprintln!(
        "  Write throughput:        {:.0} events/sec",
        total_events as f64 / write_elapsed.as_secs_f64()
    );
    eprintln!("  Avg edges/event:         {avg_edges:.2}");
    eprintln!("  Total edge refs:         {}", dag.total_edge_refs());
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Metric A: I/O (cold: cache cleared, packfile re-reads from disk)");
    eprintln!("    rchar (logical):       {logical_reads} bytes");
    eprintln!("    read_bytes (disk):     {disk_reads} bytes");
    eprintln!("    read syscalls:         {read_syscalls}");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Metric B: Cache efficiency");
    eprintln!("    Cold hit rate:         {cold_hit_rate:.1}% (0% on purpose/cache clear)");
    eprintln!("    Warm hit rate:         {warm_hit_rate:.1}% ({warm_hits}/{warm_total})");
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Index accuracy (lossy fanout)");
    eprintln!("    Total get() calls:     {get_calls}");
    eprintln!("    Found by index:        {get_found}");
    eprintln!("    Lost to collision:     {get_not_found} ({index_loss_rate:.1}%)");
    if get_errors > 0 {
        eprintln!("    Read errors:           {get_errors}");
    }
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Wall-clock (cold read):  {read_elapsed:.2?} ({cold_gets_per_sec:.0} gets/sec)");
    eprintln!("  Wall-clock (warm read):  {warm_elapsed:.2?} ({warm_gets_per_sec:.0} gets/sec)");
    eprintln!();

    let _ = fs::remove_dir_all(&dir);

    BenchResult {
        read_syscalls,
        index_loss_rate,
        warm_hit_rate,
        cold_gets_per_sec,
        warm_gets_per_sec,
    }
}

// ── Synthetic HAMT state trie (root + L1, structural sharing) ───────
//
// Fixed depth of 2 (root, then one of 32 L1 buckets) rather than a full
// log32(N)-deep trie: this matches the "spine" this session already
// identified as the universally-hot, small part of a real HAMT (root
// rewritten every write, L1 rewritten roughly every 32nd write), and it's
// enough to give a state lookup a real depth multiplier (2 reads, not 1)
// with genuine cross-event sharing on untouched buckets — which is the
// property that actually matters for these percentages, not modeling
// arbitrary depth for its own sake.
const HAMT_BUCKETS: u64 = 32;
const HAMT_GENESIS: u64 = u64::MAX;

fn hamt_node_id(kind: u8, idx: u64) -> NodeId {
    let seed = idx ^ (u64::from(kind) << 60) ^ 0xC0FF_EE00_C0FF_EE00;
    let a = splitmix64(seed);
    let b = splitmix64(a ^ 0x1357_9BDF_2468_ACE0);
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

/// Which event last touched `bucket`, as of event `as_of` (inclusive).
/// Every event i touches bucket `i % HAMT_BUCKETS`, so the owner is the
/// largest such i <= as_of — a closed form, no history table needed.
fn l1_owner(as_of: u64, bucket: u64) -> u64 {
    if as_of < bucket {
        HAMT_GENESIS
    } else {
        as_of - ((as_of - bucket) % HAMT_BUCKETS)
    }
}

// ── Read-intent instrumentation ──────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Intent {
    /// Fetching an event purely to extract its prev/auth routing.
    GraphWalk,
    /// Fetching a HAMT node to navigate the room state trie.
    StateTrie,
    /// Fetching an event because its JSON body is actually needed.
    Timeline,
}

#[derive(Default)]
struct IntentStats {
    calls: AtomicU64,
    bytes: AtomicU64,
}

impl IntentStats {
    fn record(&self, len: usize) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(len as u64, Ordering::Relaxed);
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

/// Wraps a `PackfileStorage`, tagging every `get` with why it was fetched
/// so the read budget can be split by intent instead of treated as one
/// undifferentiated stream of `get()` calls.
struct InstrumentedStorage {
    inner: PackfileStorage,
    graph_walk: IntentStats,
    state_trie: IntentStats,
    timeline: IntentStats,
}

impl InstrumentedStorage {
    fn new(inner: PackfileStorage) -> Self {
        Self {
            inner,
            graph_walk: IntentStats::default(),
            state_trie: IntentStats::default(),
            timeline: IntentStats::default(),
        }
    }

    fn get_intent(&self, id: &NodeId, intent: Intent) -> bool {
        match self.inner.get(&ROOM, id) {
            Ok(Some(data)) => {
                let stats = match intent {
                    Intent::GraphWalk => &self.graph_walk,
                    Intent::StateTrie => &self.state_trie,
                    Intent::Timeline => &self.timeline,
                };
                stats.record(data.bytes.len());
                true
            }
            _ => false,
        }
    }
}

/// Backward auth-chain walk from `start`, bounded by `max_depth`. Real auth
/// chains are shallow (bounded by how many times power-levels/join-rules
/// actually changed), so this terminates naturally on the rotating pool
/// long before `max_depth` matters in practice.
fn auth_chain(dag: &DagGenerator, start: usize, max_depth: usize) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0usize));
    visited.insert(start);
    while let Some((idx, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for &a in &dag.auth_events[idx] {
            if visited.insert(a) {
                queue.push_back((a, depth + 1));
            }
        }
    }
    visited
}

/// Simulates a state-resolution-v2-shaped read workload — auth-chain
/// fork/join edge-chasing plus HAMT descent — instead of a naive linear
/// scan, and reports what fraction of reads are spent on graph edges vs.
/// state-trie nodes vs. event bodies. This is the number the DAG-sidecar
/// vs. materialized-state vs. integration-as-is decision should be made
/// on, not a guess.
fn run_intent_benchmark(total_events: usize) {
    let dir = std::env::temp_dir().join(format!("mtxdb_bench_intent_{total_events}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // fork_prob high enough, join_depth short enough, to guarantee several
    // divergent tips to reconcile — that's the workload this scenario
    // exists to exercise.
    let dag = DagGenerator::generate(total_events, 0.3, 5);

    let store = PackfileStorage::open_with_cache(
        dir.clone(),
        NodeCache::new(2000),
        Some(DagGenerator::parse_children),
    )
    .unwrap();

    // Write phase: each event's own node, plus its as-of-that-event HAMT
    // root and the L1 bucket it touches. See l1_owner: every event i
    // touches bucket i % HAMT_BUCKETS, so this also writes exactly the
    // nodes l1_owner will resolve to later.
    for i in 0..dag.len() {
        let (id, data) = dag.node_data(i);
        store.put(&ROOM, &id, &data).unwrap();

        let root_id = hamt_node_id(0, i as u64);
        store
            .put(
                &ROOM,
                &root_id,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();
        let l1_id = hamt_node_id(1, i as u64);
        store
            .put(
                &ROOM,
                &l1_id,
                &NodeData::new(bytes::Bytes::from_static(b"l1")),
            )
            .unwrap();
    }
    // Genesis L1 node for buckets no event has touched yet.
    store
        .put(
            &ROOM,
            &hamt_node_id(1, HAMT_GENESIS),
            &NodeData::new(bytes::Bytes::from_static(b"l1-genesis")),
        )
        .unwrap();

    store.cache().clear();
    let inst = InstrumentedStorage::new(store);

    // Reconcile 2+ divergent tips, the way state-res v2 actually shapes
    // the work: walk each fork's auth chain, compute the auth-difference
    // (union minus what's common to every fork), then auth-check each
    // event in that difference against the fork's current state.
    let tips: Vec<usize> = dag.tips.clone();
    assert!(tips.len() >= 2, "scenario requires multiple divergent tips");

    let chains: Vec<HashSet<usize>> = tips.iter().map(|&t| auth_chain(&dag, t, 64)).collect();

    for chain in &chains {
        for &idx in chain {
            inst.get_intent(&DagGenerator::node_id(idx), Intent::GraphWalk);
        }
    }

    let common: HashSet<usize> = chains.iter().skip(1).fold(chains[0].clone(), |acc, c| {
        acc.intersection(c).copied().collect()
    });
    let union: HashSet<usize> = chains.iter().flatten().copied().collect();
    let auth_difference: Vec<usize> = union.difference(&common).copied().collect();

    for (tip, &tip_idx) in tips.iter().enumerate() {
        for &idx in &auth_difference {
            if !chains[tip].contains(&idx) {
                continue;
            }
            inst.get_intent(&DagGenerator::node_id(idx), Intent::Timeline);

            // Auth-check: resolve 2 representative state keys against this
            // fork's current root — root + L1 bucket, real depth-2 descent.
            let root_id = hamt_node_id(0, tip_idx as u64);
            inst.get_intent(&root_id, Intent::StateTrie);
            for bucket in [0u64, 7u64] {
                let owner = l1_owner(tip_idx as u64, bucket);
                inst.get_intent(&hamt_node_id(1, owner), Intent::StateTrie);
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);

    let graph_calls = inst.graph_walk.calls();
    let state_calls = inst.state_trie.calls();
    let timeline_calls = inst.timeline.calls();
    let total_calls = graph_calls + state_calls + timeline_calls;

    let graph_bytes = inst.graph_walk.bytes();
    let state_bytes = inst.state_trie.bytes();
    let timeline_bytes = inst.timeline.bytes();
    let total_bytes = graph_bytes + state_bytes + timeline_bytes;

    let pct = |part: u64, total: u64| {
        if total > 0 {
            part as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    };

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  READ-INTENT BREAKDOWN ({total_events} events, {} tips)",
        tips.len()
    );
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  GraphWalk:  {graph_calls:>6} calls ({:5.1}%)   {graph_bytes:>8} bytes ({:5.1}%)",
        pct(graph_calls, total_calls),
        pct(graph_bytes, total_bytes)
    );
    eprintln!(
        "  StateTrie:  {state_calls:>6} calls ({:5.1}%)   {state_bytes:>8} bytes ({:5.1}%)",
        pct(state_calls, total_calls),
        pct(state_bytes, total_bytes)
    );
    eprintln!(
        "  Timeline:   {timeline_calls:>6} calls ({:5.1}%)   {timeline_bytes:>8} bytes ({:5.1}%)",
        pct(timeline_calls, total_calls),
        pct(timeline_bytes, total_bytes)
    );
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Decide on CALL SHARE, not byte share: on HDD-class media a");
    eprintln!("  seek costs orders of magnitude more than the bytes it returns");
    eprintln!("  (see the sequential-vs-random table in the blog post), so a");
    eprintln!("  fetch of a few tiny HAMT-node bytes costs the same seek as a");
    eprintln!("  full event body. Byte share is shown for context only — it");
    eprintln!("  will systematically understate small-payload categories like");
    eprintln!("  StateTrie relative to their real I/O cost.");
    eprintln!("  If GraphWalk dominates call share -> Edge packing");
    eprintln!("  If StateTrie dominates call share -> invest in packfile/index");
    eprintln!("  If Timeline dominates call share  -> neither helps here");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
}

// ── Reaction-swarm adversarial scenario ──────────────────────────────
//
// PackfileStorage::put() may invoke maybe_repack(), which can call
// repack_room(). However, this benchmark never calls set_live_roots(),
// so maybe_repack() returns early without repacking — the room's
// packfile grows monotonically in arrival order. That makes the real
// threat model here "random-offset reads into a large, cold, ever-growing
// file", not "did this survive a repack". This scenario measures whether
// an attacker choosing reaction targets from the OLDEST part of a room's
// history (the furthest possible physical offset from the write head)
// costs more than organic reactions to RECENT messages, and whether the
// sort-then-read fix in get_many (Part 3) narrows that gap.
fn reaction_id(salt: u64, idx: u64) -> NodeId {
    let seed = idx ^ salt ^ 0xDEAD_BEEF_1234_5678u64.rotate_left(3);
    let a = splitmix64(seed);
    let b = splitmix64(a ^ 0x0BAD_F00D_0BAD_F00D);
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

fn run_reaction_swarm_benchmark(history_len: usize, swarm_size: usize) {
    let dir = std::env::temp_dir().join(format!("mtxdb_bench_swarm_{history_len}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // A long, mostly-linear room history — the base messages that will be
    // reacted to. Low fork probability: this is about history depth, not
    // fork/join shape (that's what run_intent_benchmark covers).
    let dag = DagGenerator::generate(history_len, 0.02, 50);
    let store = PackfileStorage::open_with_cache(dir.clone(), NodeCache::new(2000), None).unwrap();

    for i in 0..dag.len() {
        let (id, data) = dag.node_data(i);
        store.put(&ROOM, &id, &data).unwrap();
    }

    // The swarm itself: swarm_size small reaction stub events. Their own
    // storage cost is trivial and, being appended together, physically
    // contiguous for free — not what this measures.
    let swarm_ids: Vec<NodeId> = (0..swarm_size as u64)
        .map(|i| reaction_id(0xAAAA, i))
        .collect();
    for id in &swarm_ids {
        store
            .put(
                &ROOM,
                id,
                &NodeData::new(bytes::Bytes::from_static(b"reaction")),
            )
            .unwrap();
    }

    // Adversarial targets: the OLDEST swarm_size events — maximally distant
    // from the write head. Organic targets: the MOST RECENT swarm_size
    // events prior to the swarm — where real reaction behavior clusters.
    let adversarial_targets: Vec<NodeId> = (0..swarm_size.min(history_len))
        .map(DagGenerator::node_id)
        .collect();
    let organic_targets: Vec<NodeId> = (history_len.saturating_sub(swarm_size)..history_len)
        .map(DagGenerator::node_id)
        .collect();

    struct Row {
        mode: &'static str,
        target: &'static str,
        found: usize,
        total: usize,
        elapsed: std::time::Duration,
        syscalls: u64,
        disk_read_bytes: u64,
    }

    // Shared measurement scaffolding: clear cache, evict page cache, time
    // an arbitrary read strategy `f` (which returns how many targets it
    // found), and package the result. `measure`/`measure_naive` below
    // differ only in `f` — get_many's sort-then-read vs. a naive per-id
    // loop — not in setup or bookkeeping.
    let timed_read = |mode: &'static str,
                      target: &'static str,
                      targets: &[NodeId],
                      f: &dyn Fn(&[NodeId]) -> usize|
     -> Row {
        store.cache().clear();
        drop_caches_for_dir(&dir);
        let io_before = IoStats::read_now();
        let t = Instant::now();
        let found = f(targets);
        let elapsed = t.elapsed();
        let io_after = IoStats::read_now();
        Row {
            mode,
            target,
            found,
            total: targets.len(),
            elapsed,
            syscalls: io_after.syscr.saturating_sub(io_before.syscr),
            disk_read_bytes: io_after.read_bytes.saturating_sub(io_before.read_bytes),
        }
    };

    let measure = |mode: &'static str, target: &'static str, targets: &[NodeId]| -> Row {
        timed_read(mode, target, targets, &|targets| {
            store
                .get_many(&ROOM, targets)
                .unwrap()
                .iter()
                .filter(|r| r.is_some())
                .count()
        })
    };

    // Naive (unsorted, per-id) fetch for comparison — quantifies what
    // get_many's sort-then-read actually buys on this exact pattern.
    let measure_naive = |mode: &'static str, target: &'static str, targets: &[NodeId]| -> Row {
        timed_read(mode, target, targets, &|targets| {
            targets
                .iter()
                .filter(|id| store.get(&ROOM, id).unwrap().is_some())
                .count()
        })
    };

    let rows = [
        measure("get_many", "adversarial", &adversarial_targets),
        measure("get_many", "organic", &organic_targets),
        measure_naive("naive", "adversarial", &adversarial_targets),
        measure_naive("naive", "organic", &organic_targets),
    ];

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  REACTION SWARM ({history_len} history events, {swarm_size} reactions)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  {:<10} {:<12} {:>10} {:>12} {:>10} {:>14}",
        "mode", "target", "found", "elapsed", "syscalls", "disk read"
    );
    for r in &rows {
        eprintln!(
            "  {:<10} {:<12} {:>10} {:>12.2?} {:>10} {:>11} B",
            r.mode,
            r.target,
            format!("{}/{}", r.found, r.total),
            r.elapsed,
            r.syscalls,
            r.disk_read_bytes
        );
    }
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  If adversarial >> organic under get_many: sort-then-read");
    eprintln!("  doesn't fully mitigate worst-case target selection — the");
    eprintln!("  attacker still forces genuinely scattered physical reads,");
    eprintln!("  just visited in a sane order. Compare against the naive");
    eprintln!("  rows to see how much of the gap sorting actually closes.");
    eprintln!("═══════════════════════════════════════════════════════════════");
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
    let large = run_benchmark("large", 100_000, 2_000);
    let pressure = run_benchmark("pressure", 100_000, 100);

    run_intent_benchmark(20_000);

    run_reaction_swarm_benchmark(50_000, 500);

    // ── Connectivity check ──
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  GRAPH CONNECTIVITY");
    eprintln!("═══════════════════════════════════════════════════════════════");
    let dag_check = DagGenerator::generate(100_000, 0.15, 10);
    let traversal_check = dag_check.traversal_order();
    let reachable = traversal_check.len();
    let total = dag_check.len();
    let pct = reachable as f64 / total as f64 * 100.0;
    eprintln!("  Total events:         {total}");
    eprintln!("  Reachable from tips:  {reachable} ({pct:.1}%)");
    eprintln!("  Tips:                 {}", dag_check.tips.len());
    eprintln!("═══════════════════════════════════════════════════════════════");

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  DECISION MATRIX ('large' scenario, measured signals only)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  read syscalls (cold):  {}", large.read_syscalls);
    eprintln!("  index loss rate:       {:.2}%", large.index_loss_rate);
    eprintln!("  warm cache hit rate:   {:.1}%", large.warm_hit_rate);
    eprintln!(
        "  cold throughput:       {:.0} gets/sec",
        large.cold_gets_per_sec
    );
    eprintln!(
        "  warm throughput:       {:.0} gets/sec ({:.1}x cold)",
        large.warm_gets_per_sec,
        large.warm_gets_per_sec / large.cold_gets_per_sec
    );
    eprintln!("  small cache warm hit:  {:.1}%", pressure.warm_hit_rate);
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  If read syscalls (cold) > 6:     -> Concurrent Frontier I/O");
    eprintln!("    (mmap collapses per-record reads to O(1) syscalls; a rise");
    eprintln!("    here means the mmap path regressed or was bypassed.)");
    eprintln!();
    eprintln!("  If warm hit rate < 50% at pressure cache size:");
    eprintln!("                                 -> DAG Sidecar (edge packing)");
    eprintln!("    (cache_entries is smaller than the working set, so even a");
    eprintln!("    same-scan re-read gets no reuse; packing edges alongside");
    eprintln!("    nodes would cut re-fetches instead of relying on the LRU.)");
    eprintln!();
    eprintln!("  If index loss rate > 0.1%:        -> widen index capacity/tag");
    eprintln!("═══════════════════════════════════════════════════════════════");
}
