//! Throughput/latency microbenchmarks for `PackfileStorage`.
//!
//! Not part of the public crate API — run via `cargo bench`. Lints are
//! relaxed here since bench code favors straightforward arithmetic and
//! formatting over the pedantic style enforced on the library itself.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::collections::{HashSet, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use mtxdb_core::storage::{NodeData, NodeId, NodeRef, StorageEngine};
use mtxdb_core::PackfileStorage;

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

// ── Synthetic Matrix room DAG generator ───────────────────────────────────

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
            if !pending_joins.is_empty() && i % join_depth == 0 {
                // Merge every outstanding orphan at the periodic boundary, not
                // just one. `tips` is capped below the old `tips.len() >= 8`
                // pressure valve (see the `tips.len() > 4` cull below), so that
                // valve never fired and joins could not keep pace with forks --
                // stranding a third or more of the DAG's joins for the terminal
                // bulk-absorb pass and leaving the workload's merge structure
                // unrepresentative.
                extra_prevs = std::mem::take(&mut pending_joins)
                    .into_iter()
                    .flatten()
                    .collect();
            }

            let tip_idx = if tips.len() == 1 { 0 } else { tips.len() - 1 };
            let parent = tips[tip_idx];

            let mut prev = vec![parent];
            prev.extend(extra_prevs);
            prev_events.push(prev);

            // Reference the collection's create event plus one other, more
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

        // Absorb remaining orphan chains into bounded synthetic events rather
        // than dropping them when the final event reaches the record-size
        // limit.  Every generated node must remain reachable from a tip: the
        // benchmark's connectivity check is also its guard against silently
        // measuring only a subset of the data it wrote.
        let max_prevs = 4000;
        let remaining: Vec<usize> = pending_joins.into_iter().flatten().collect();
        for chunk in remaining.chunks(max_prevs) {
            let absorb_id = prev_events.len();
            prev_events.push(chunk.to_vec());
            auth_events.push(vec![auth_pool[0]]);
            tips.push(absorb_id);
        }

        // The final synthetic nodes are themselves tips, so the original
        // orphan chains are reachable without putting an oversized prev list
        // on the last ordinary event.
        if remaining.is_empty() {
            debug_assert_eq!(prev_events.len(), total_events);
        } else {
            debug_assert!(prev_events.len() > total_events);
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
    /// Capture the kernel counters when they are available.
    ///
    /// Returning `None` is materially different from reporting zero: a zero
    /// is a valid measurement, while `/proc` may be absent outside Linux (or
    /// in a restricted container).  The machine-readable benchmark rows keep
    /// that distinction as `n/a` so CSV history never records an unavailable
    /// counter as a real zero-I/O result.
    fn read_now() -> Option<Self> {
        let content = fs::read_to_string("/proc/self/io").ok()?;
        let mut rchar = None;
        let mut read_bytes = None;
        let mut syscr = None;
        let mut write_bytes = 0;
        for line in content.lines() {
            if let Some(v) = line.strip_prefix("rchar: ") {
                rchar = v.trim().parse().ok();
            }
            if let Some(v) = line.strip_prefix("read_bytes: ") {
                read_bytes = v.trim().parse().ok();
            }
            if let Some(v) = line.strip_prefix("write_bytes: ") {
                write_bytes = v.trim().parse().unwrap_or(0);
            }
            if let Some(v) = line.strip_prefix("syscr: ") {
                syscr = v.trim().parse().ok();
            }
        }
        Some(Self {
            rchar: rchar?,
            read_bytes: read_bytes?,
            write_bytes,
            syscr: syscr?,
        })
    }
}

fn format_io_metric(value: Option<u64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
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
fn drop_caches_for_dir(dir: &std::path::Path) -> bool {
    // Shells out to `vmtouch -e`, which wraps the same posix_fadvise(2)
    // eviction this needs, rather than adding a dependency (or an unsafe
    // FFI call of our own) just for one syscall. Silently does nothing if
    // vmtouch isn't installed — callers should treat this as best-effort.
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
// jscpd:ignore-end

// ── Benchmark harness ───────────────────────────────────────────────

/// Root for benchmark scratch data: `MTXDB_BENCH_ROOT` env override, else
/// the session temp dir. GB-scale sweeps (1–1000 GB) must not run on a
/// RAM-backed tmpfs; point this at a real disk with headroom.
///
/// Suffixed with this process's PID: every scratch path built under this
/// root is otherwise a deterministic function of the scenario label (see
/// the `mtxdb_bench_*` join sites below), so two benchmark processes
/// running concurrently against the same root — e.g. overlapping `cargo
/// bench` invocations, or CI jobs sharing `MTXDB_BENCH_ROOT` — would each
/// `remove_dir_all` and recreate the other's active scratch directory,
/// corrupting or crashing both runs.
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
                "mtxdb_bench_run_{}_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                token
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

/// Measured results from one scenario run, used to drive the decision
/// matrix on real signals instead of structurally-fixed ones.
struct BenchResult {
    read_syscalls: Option<u64>,
    index_loss_rate: f64,
    warm_hit_rate: f64,
    cold_gets_per_sec: f64,
    warm_gets_per_sec: f64,
}

fn run_benchmark(label: &str, total_events: usize, cache_entries: usize) -> BenchResult {
    let dir = bench_root().join(format!("mtxdb_bench_{label}_{total_events}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let dag = DagGenerator::generate(total_events, 0.15, 10);
    let generated_events = dag.len();
    let traversal = dag.traversal_order();
    let node_ids: Vec<NodeId> = traversal
        .iter()
        .map(|&i| DagGenerator::node_id(i))
        .collect();

    // ── Write phase ──
    let store = PackfileStorage::open_with_cache(dir.clone(), cache_entries).unwrap();

    let t_write = Instant::now();
    for i in 0..dag.len() {
        let (id, data) = dag.node_data(i);
        store.put(&ROOM, &id, &data).unwrap();
    }
    let write_elapsed = t_write.elapsed();
    let pack_size = pack_dir_size(&dir);

    // ── Clear cache to force packfile re-reads ──
    store.collection_cache(&ROOM).clear();

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

    let cold_hits = store.collection_cache(&ROOM).hits();
    let cold_misses = store.collection_cache(&ROOM).misses();

    let read_elapsed = t_read.elapsed();
    let io_after = IoStats::read_now();
    let io_delta = io_before.zip(io_after);
    let logical_reads = io_delta.map(|(before, after)| after.rchar.saturating_sub(before.rchar));
    let disk_reads =
        io_delta.map(|(before, after)| after.read_bytes.saturating_sub(before.read_bytes));
    let read_syscalls = io_delta.map(|(before, after)| after.syscr.saturating_sub(before.syscr));

    let cold_total = cold_hits + cold_misses;
    let cold_hit_rate = if cold_total > 0 {
        (cold_hits as f64 / cold_total as f64) * 100.0
    } else {
        0.0
    };
    let completed_lookups = get_found + get_not_found;
    let index_loss_rate = if completed_lookups > 0 {
        (get_not_found as f64 / completed_lookups as f64) * 100.0
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
    let hits_before_warm = store.collection_cache(&ROOM).hits();
    let misses_before_warm = store.collection_cache(&ROOM).misses();
    let t_warm = Instant::now();

    let mut warm_found = 0u64;
    for id in node_ids.iter() {
        if let Ok(Some(_)) = store.get(&ROOM, id) {
            warm_found += 1
        }
    }

    let warm_elapsed = t_warm.elapsed();
    let warm_hits = store.collection_cache(&ROOM).hits() - hits_before_warm;
    let warm_misses = store.collection_cache(&ROOM).misses() - misses_before_warm;
    let warm_total = warm_hits + warm_misses;
    let warm_hit_rate = if warm_total > 0 {
        (warm_hits as f64 / warm_total as f64) * 100.0
    } else {
        0.0
    };
    let warm_gets_per_sec = warm_found as f64 / warm_elapsed.as_secs_f64();

    let avg_edges = dag.total_edge_refs() as f64 / generated_events as f64;

    println!(
        "bench: locality L={label} N={generated_events} CACHE={cache_entries} \
         PACK_BYTES={pack_size} WRITE_EVENTS_PER_SEC={:.0} READ_SYSCALLS={} \
         DISK_READ_BYTES={} INDEX_LOSS_PCT={index_loss_rate:.4} \
         COLD_GETS_PER_SEC={cold_gets_per_sec:.0} WARM_HIT_PCT={warm_hit_rate:.4} \
         WARM_GETS_PER_SEC={warm_gets_per_sec:.0}",
        generated_events as f64 / write_elapsed.as_secs_f64(),
        format_io_metric(read_syscalls),
        format_io_metric(disk_reads),
    );

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  {label}: {generated_events} generated events ({total_events} requested), cache={cache_entries} entries");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  Pack size:               {:.2} MB",
        pack_size as f64 / 1e6
    );
    eprintln!(
        "  Write throughput:        {:.0} events/sec",
        generated_events as f64 / write_elapsed.as_secs_f64()
    );
    eprintln!("  Avg edges/event:         {avg_edges:.2}");
    eprintln!("  Total edge refs:         {}", dag.total_edge_refs());
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  Metric A: I/O (cold: cache cleared, packfile re-reads from disk)");
    eprintln!(
        "    rchar (logical):       {} bytes",
        format_io_metric(logical_reads)
    );
    eprintln!(
        "    read_bytes (disk):     {} bytes",
        format_io_metric(disk_reads)
    );
    eprintln!(
        "    read syscalls:         {}",
        format_io_metric(read_syscalls)
    );
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

    drop(store);
    let _ = fs::remove_dir_all(&dir);

    BenchResult {
        read_syscalls,
        index_loss_rate,
        warm_hit_rate,
        cold_gets_per_sec,
        warm_gets_per_sec,
    }
}

/// Deterministic, incompressible payload seeded per node. A shared constant
/// payload would zstd-compress to near-zero on disk and make a "1 GB open"
/// measure a tiny scan; per-node splitmix keeps the on-disk size honest.
fn incompressible_payload(seed: u64, bytes_len: usize) -> bytes::Bytes {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(bytes_len);
    while out.len() < bytes_len {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    out.truncate(bytes_len);
    bytes::Bytes::from(out)
}

/// Measure the cost hidden by a short-lived `mtxdb get` invocation.
///
/// The normal read benchmarks above deliberately keep one store open, which is
/// the right model for Synapse but not for the CLI. A CLI process must rebuild
/// every collection index before its first point lookup because those indexes
/// are currently in-memory only (see `DESIGN-open-and-index-persistence.md`).
/// Keep this separate from the point-read benchmark so a future persisted
/// lookup-index can make the open time fall without obscuring the cost of the
/// final `get` itself.
///
/// `target_gb` sets the nominal dataset size: the node count is derived from it
/// and `payload_bytes`, and the on-disk pack size tracks it because payloads
/// are incompressible. Returns the ``bench:``-line label and, when the page
/// cache was actually dropped, the evicted open time; `None` means eviction was
/// unavailable or failed, so the caller must not treat the sample as cold.
#[allow(clippy::uninlined_format_args)]
fn run_oneshot_open_benchmark(
    target_gb: f64,
    collection_count: usize,
    payload_bytes: usize,
) -> (String, Option<f64>) {
    assert!(
        collection_count > 0,
        "benchmark needs at least one collection"
    );
    assert!(payload_bytes > 0, "benchmark needs a nonzero payload size");

    let dir = bench_root().join(format!("mtxdb_bench_oneshot_open_gb_{target_gb}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let total_nodes = (target_gb * 1e9) as usize / payload_bytes;
    assert!(total_nodes > 0, "target_gb is too small for payload_bytes");
    let target_node_index = total_nodes / 2;
    let target_id = DagGenerator::node_id(target_node_index);
    let mut target_collection = [0u8; 16];
    // The target node lives in the same round-robin collection the write loop
    // used; querying a hardcoded collection 0 would miss whenever
    // `target_node_index % collection_count != 0` (e.g. 1 GB / 39 cols → 1).
    target_collection[..8].copy_from_slice(
        &u64::try_from(target_node_index % collection_count)
            .expect("usize always fits in u64")
            .to_le_bytes(),
    );

    let store = PackfileStorage::open(dir.clone()).unwrap();
    let write_started = Instant::now();
    for node in 0..total_nodes {
        let mut collection = [0u8; 16];
        collection[..8].copy_from_slice(
            &u64::try_from(node % collection_count)
                .expect("usize always fits in u64")
                .to_le_bytes(),
        );
        let payload = incompressible_payload(node as u64, payload_bytes);
        store
            .put(
                &collection,
                &DagGenerator::node_id(node),
                &NodeData::new(payload),
            )
            .unwrap();
    }
    let write_elapsed = write_started.elapsed();
    store.sync_all().unwrap();
    drop(store);

    let pack_bytes = pack_dir_size(&dir);
    let measure = || {
        let started = Instant::now();
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        let open_elapsed = started.elapsed();

        // LossyIndex resident bytes (sum over all per-collection indices). This
        // is the number that becomes the checkpoint-file size in the persisted
        // index design; it is isolated from the mmap'd packfile page cache.
        let index_bytes: u64 = store
            .collection_summaries()
            .iter()
            .map(|(_, _, bytes, _)| *bytes as u64)
            .sum();

        let lookup_started = Instant::now();
        let found = store.get(&target_collection, &target_id).unwrap().is_some();
        let lookup_elapsed = lookup_started.elapsed();
        assert!(found, "target must survive reopening");
        (open_elapsed, lookup_elapsed, index_bytes)
    };

    // The write phase leaves pages resident, providing the honest warm
    // baseline. Eviction is explicitly reported below rather than calling the
    // following measurement "cold" when vmtouch is unavailable.
    let (warm_open, warm_lookup, _) = measure();
    let evicted = drop_caches_for_dir(&dir);
    let (after_evict_open, after_evict_lookup, index_bytes) = measure();

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  ONE-SHOT CLI OPEN + GET ({target_gb} GB nominal)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Nodes / collections:    {total_nodes} / {collection_count}");
    eprintln!(
        "  Pack bytes:              {:.2} MB",
        pack_bytes as f64 / 1e6
    );
    eprintln!(
        "  Index resident (slots):  {:.2} MB",
        index_bytes as f64 / 1e6
    );
    eprintln!(
        "  Write phase:             {:.2} s",
        write_elapsed.as_secs_f64()
    );
    eprintln!("  Warm open/index rebuild: {warm_open:.2?}");
    eprintln!("  Warm point lookup:       {warm_lookup:.2?}");
    eprintln!(
        "  {} open/index rebuild: {after_evict_open:.2?}",
        if evicted {
            "Evicted-page"
        } else {
            "No-eviction"
        }
    );
    eprintln!(
        "  {} point lookup:       {after_evict_lookup:.2?}",
        if evicted {
            "Evicted-page"
        } else {
            "No-eviction"
        }
    );
    if !evicted {
        eprintln!("  Note: vmtouch unavailable or failed; no disk-cold claim is made.");
    }
    eprintln!();

    let label = format!("{target_gb:.3}");
    let label = label.trim_end_matches('0').trim_end_matches('.');
    let (evicted_open_us, evicted_lookup_us) = if evicted {
        (
            format!("{:.1}", after_evict_open.as_secs_f64() * 1e6),
            format!("{:.1}", after_evict_lookup.as_secs_f64() * 1e6),
        )
    } else {
        ("n/a".to_owned(), "n/a".to_owned())
    };
    println!(
        "bench: open L={label}gb N={total_nodes} COLS={collection_count} WRITE_MS={:.1} \
         PACK={pack_bytes} INDEX={index_bytes} WARM_OPEN_US={warm_open_us:.1} \
         WARM_LOOKUP_US={warm_lookup_us:.1} EVICTED_OPEN_US={evicted_open_us} \
         EVICTED_LOOKUP_US={evicted_lookup_us} EVICTED={evicted}",
        write_elapsed.as_secs_f64() * 1e3,
        warm_open_us = warm_open.as_secs_f64() * 1e6,
        warm_lookup_us = warm_lookup.as_secs_f64() * 1e6,
    );

    let lb = label.to_owned();
    let evicted_open_secs = evicted.then_some(after_evict_open.as_secs_f64());
    let _ = fs::remove_dir_all(&dir);
    (lb, evicted_open_secs)
}

/// Sweep the one-shot open metric over a bracketed set of dataset sizes, plus
/// a per-GB scaling check when at least two points were measured.
///
/// Defaults to `0.1` GB so `cargo bench` / CI stays cheap; set
/// `MTXDB_BENCH_OPEN_GB=1,100,1000` for the real curve (see
/// `DESIGN-open-and-index-persistence.md`).
fn run_open_size_sweep() {
    let gbs: Vec<f64> = match std::env::var("MTXDB_BENCH_OPEN_GB") {
        Ok(raw) => raw
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<f64>()
                    .expect("invalid MTXDB_BENCH_OPEN_GB")
            })
            .filter(|g| *g > 0.0)
            .collect(),
        Err(std::env::VarError::NotPresent) => vec![0.1],
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MTXDB_BENCH_OPEN_GB must be valid UTF-8")
        }
    };
    assert!(
        !gbs.is_empty(),
        "MTXDB_BENCH_OPEN_GB must list at least one size"
    );

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(gbs.len());
    for gb in &gbs {
        let (_, evicted_open_secs) = run_oneshot_open_benchmark(*gb, 39, 1024);
        if let Some(evicted_open_secs) = evicted_open_secs {
            points.push((*gb, evicted_open_secs));
        }
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));

    if points.len() >= 2 {
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("  OPEN SCALING SHAPE (evicted open time vs dataset GB)");
        eprintln!("═══════════════════════════════════════════════════════════════");
        for (idx, (gb, secs)) in points.iter().enumerate().skip(1) {
            let (prev_gb, prev_secs) = points[idx - 1];
            let open_ratio = secs / prev_secs;
            let gb_ratio = gb / prev_gb;
            let per_gb = open_ratio / gb_ratio;
            let shape = if per_gb > 1.1 {
                "superlinear"
            } else if per_gb < 0.9 {
                "sublinear"
            } else {
                "linear"
            };
            eprintln!(
                "  {prev_gb} GB -> {gb} GB: open x{open_ratio:.2} vs data x{gb_ratio:.2} \
                 ({per_gb:.2}x per GB, {shape})"
            );
        }
        eprintln!("  Linear (=> index rebuild is a straight scan); superlinear suggests");
        eprintln!("  page-cache pressure or rehash amplification driving the bigger sizes.");
        eprintln!();
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
    /// Fetching a HAMT node to navigate the collection state trie.
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
    let dir = bench_root().join(format!("mtxdb_bench_intent_{total_events}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // fork_prob high enough, join_depth short enough, to guarantee several
    // divergent tips to reconcile — that's the workload this scenario
    // exists to exercise.
    let dag = DagGenerator::generate(total_events, 0.3, 5);
    let generated_events = dag.len();

    let store = PackfileStorage::open_with_cache(dir.clone(), 2000).unwrap();

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

    store.collection_cache(&ROOM).clear();
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

    let graph_calls = inst.graph_walk.calls();
    let state_calls = inst.state_trie.calls();
    let timeline_calls = inst.timeline.calls();
    let total_calls = graph_calls + state_calls + timeline_calls;

    let graph_bytes = inst.graph_walk.bytes();
    let state_bytes = inst.state_trie.bytes();
    let timeline_bytes = inst.timeline.bytes();
    let total_bytes = graph_bytes + state_bytes + timeline_bytes;

    drop(inst);
    let _ = fs::remove_dir_all(&dir);

    let pct = |part: u64, total: u64| {
        if total > 0 {
            part as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    };

    println!(
        "bench: intent EVENTS={generated_events} GRAPH_CALLS={graph_calls} \
         STATE_CALLS={state_calls} TIMELINE_CALLS={timeline_calls} \
         GRAPH_BYTES={graph_bytes} STATE_BYTES={state_bytes} \
         TIMELINE_BYTES={timeline_bytes}",
    );

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  READ-INTENT BREAKDOWN ({generated_events} events ({total_events} requested), {} tips)",
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
// PackfileStorage::put() never triggers automatic repacks — repack is
// purely caller-invoked (repack_collection_reachable). This benchmark never
// calls it, so the collection's packfile grows monotonically in arrival order.
// That makes the real threat model here
// "random-offset reads into a large, cold, ever-growing file", not "did
// this survive a repack". This scenario measures whether an attacker
// choosing reaction targets from the OLDEST part of a collection's history
// (the furthest possible physical offset from the write head) costs more
// than organic reactions to RECENT messages, and whether the sort-then-read
// fix in get_many (Part 3) narrows that gap.
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
    let dir = bench_root().join(format!("mtxdb_bench_swarm_{history_len}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // A long, mostly-linear collection history — the base messages that will be
    // reacted to. Low fork probability: this is about history depth, not
    // fork/join shape (that's what run_intent_benchmark covers).
    let dag = DagGenerator::generate(history_len, 0.02, 50);
    let store = PackfileStorage::open_with_cache(dir.clone(), 2000).unwrap();

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
        syscalls: Option<u64>,
        disk_read_bytes: Option<u64>,
        evicted: bool,
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
        store.collection_cache(&ROOM).clear();
        let evicted = drop_caches_for_dir(&dir);
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
            syscalls: io_before
                .zip(io_after)
                .map(|(before, after)| after.syscr.saturating_sub(before.syscr)),
            disk_read_bytes: io_before
                .zip(io_after)
                .map(|(before, after)| after.read_bytes.saturating_sub(before.read_bytes)),
            evicted,
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

    for r in &rows {
        println!(
            "bench: swarm HISTORY={history_len} SWARM={swarm_size} MODE={} TARGET={} \
             FOUND={}/{} ELAPSED_US={:.1} SYSCALLS={} DISK_READ_BYTES={} EVICTED={}",
            r.mode,
            r.target,
            r.found,
            r.total,
            r.elapsed.as_secs_f64() * 1e6,
            format_io_metric(r.syscalls),
            format_io_metric(r.disk_read_bytes),
            r.evicted,
        );
    }

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  REACTION SWARM ({history_len} history events, {swarm_size} reactions)");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  {:<10} {:<12} {:>10} {:>12} {:>10} {:>14}",
        "mode", "target", "found", "elapsed", "syscalls", "disk read"
    );
    for r in &rows {
        eprintln!(
            "  {:<10} {:<12} {:>10} {:>12.2?} {:>10} {:>12}",
            r.mode,
            r.target,
            format!("{}/{}", r.found, r.total),
            r.elapsed,
            format_io_metric(r.syscalls),
            format_io_metric(r.disk_read_bytes),
        );
    }
    eprintln!("  ───────────────────────────────────────────────────────────");
    eprintln!("  If adversarial >> organic under get_many: sort-then-read");
    eprintln!("  doesn't fully mitigate worst-case target selection — the");
    eprintln!("  attacker still forces genuinely scattered physical reads,");
    eprintln!("  just visited in a sane order. Compare against the naive");
    eprintln!("  rows to see how much of the gap sorting actually closes.");
    eprintln!("═══════════════════════════════════════════════════════════════");
    if rows.iter().any(|r| !r.evicted) {
        eprintln!("  Note: vmtouch unavailable or failed; the above timings are warm reads.");
    }
    eprintln!();

    drop(store);
    let _ = fs::remove_dir_all(&dir);
}

// ── Repack amplification scenario ────────────────────────────────────
//
// This benchmark previously called the now-retired repack_collection_rewrite,
// which had a real bug worth recording: its scan step never deduplicated
// physical record occurrences, so every repack re-copied every prior
// repack's redundant copies on top of the current live set. That's
// exponential in cycle count (T_k = 2*T_{k-1} + I for I inserts per
// cycle), not the O(n^2) originally assumed here — confirmed empirically
// when a 20,000-event run failed to complete (unbounded memory growth,
// then a Corrupt read once shard rotation collided with the same call's
// stale offsets — see repack_collection_reachable's docs on pin_shards).
//
// repack_collection_reachable's live-set/adjacency construction is inherently
// deduplicated (derived from a hash-keyed map, not a raw per-occurrence
// Vec), so this exponential blowup does not apply to it. Calling it here
// with no live roots configured still repacks the *entire* collection on every
// cycle (nothing is known to be garbage), which is the legitimate O(n^2)
// case the original comment intended:
//
//   sum_{k=1}^{N/interval} (k * interval) ≈ N^2 / (2 * interval)
//
// i.e. quadratic in N for a fixed interval. Doubling total_events at the
// same interval should show roughly 4x repack time, not 2x, if this
// bound holds. Demonstrating the bounded O(n) case requires configuring
// live_roots to a fixed-size set instead — not done here; see this
// session's plan notes on the flat-vs-topo/GC-bound benchmark.
fn run_repack_benchmark(total_events: usize, repack_interval: usize) {
    let dir = bench_root().join(format!("mtxdb_bench_repack_{total_events}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let store = PackfileStorage::open_with_cache(dir.clone(), 2000).unwrap();
    let collection_id = [0x99; 16];

    let mut total_repack_time = std::time::Duration::ZERO;
    let mut repack_count = 0;

    let t_start = Instant::now();
    for i in 0..total_events {
        let mut id = [0u8; 16];
        // splitmix64 on i to spread sequential counters across many buckets
        let x = splitmix64(i as u64);
        id[..8].copy_from_slice(&x.to_le_bytes());
        let data = NodeData::new(bytes::Bytes::from(format!("repack payload {i}")));
        store.put(&collection_id, &id, &data).unwrap();

        // Simulate an external GC worker polling and triggering repack.
        if (i + 1) % repack_interval == 0 {
            let t_repack = Instant::now();
            store
                .repack_collection_reachable(&collection_id, |_hash, _data| Vec::new())
                .unwrap();
            total_repack_time += t_repack.elapsed();
            repack_count += 1;
        }
    }
    let total_elapsed = t_start.elapsed();
    let write_only_time = total_elapsed.saturating_sub(total_repack_time);

    eprintln!(
        "bench: repack amplification ({total_events} events, repack every {repack_interval})"
    );
    println!(
        "bench: repack EVENTS={total_events} INTERVAL={repack_interval} RUNS={repack_count} \
         TOTAL_MS={:.1} WRITE_MS={:.1} REPACK_MS={:.1}",
        total_elapsed.as_secs_f64() * 1e3,
        write_only_time.as_secs_f64() * 1e3,
        total_repack_time.as_secs_f64() * 1e3,
    );
    eprintln!("  total time:   {total_elapsed:.2?}");
    eprintln!("  write time:   {write_only_time:.2?}");
    eprintln!("  repack time:  {total_repack_time:.2?} (across {repack_count} runs)");
    eprintln!("  If total_events doubles and repack time roughly quadruples (not doubles),");
    eprintln!("  that's the O(n^2) full-rewrite cost, empirically, not just argued.");
    eprintln!();

    store.delete_collection(&collection_id).unwrap();
    drop(store);
    let _ = fs::remove_dir_all(&dir);
}

/// Tracks real end-to-end latency around the epoch-handoff protocol (see
/// `PackfileStorage::persist_index_checkpoint`): a full checkpoint rewrite's
/// serialize + fsync + rename now runs unlocked, so a concurrent `put()` no
/// longer waits on the rewrite's lock. This bench measures the *observed*
/// latency at real disk speed, which also includes genuine device-level
/// contention (`put()`'s own buffered write queuing behind the rewrite's on the
/// same disk) that the lock-scope fix neither causes nor removes — so
/// `PUT_APPEND_MAX_MS` approaching or exceeding `REWRITE_MS` here is expected
/// at times and not on its own evidence of a regression. The actual lock-scope
/// claim (put doesn't block on the *lock*) is proven deterministically,
/// independent of disk speed, by the `test_put_does_not_block_on_slow_checkpoint_rewrite`
/// unit test (uses an injected delay instead of real I/O, runs in
/// milliseconds on every `cargo test`). This bench exists to track the real
/// append-latency number as a trend, not as a pass/fail gate.
fn run_checkpoint_rewrite_latency_benchmark(collections: usize, records_per_collection: usize) {
    let dir = bench_root().join(format!(
        "mtxdb_bench_checkpoint_latency_{collections}x{records_per_collection}"
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let store = PackfileStorage::open(dir.clone()).unwrap();

    for c in 0..collections {
        let mut collection = [0u8; 16];
        collection[0] = 0xCC;
        collection[1] = c as u8;
        for i in 0..records_per_collection {
            let mut id = [0u8; 16];
            id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
            store
                .put(
                    &collection,
                    &id,
                    &NodeData::new(bytes::Bytes::from(format!("seed {c} {i}"))),
                )
                .unwrap();
        }
    }
    store.sync_all().unwrap();

    // Baseline: one full rewrite in isolation (forced via a structural
    // invalidation), no concurrent puts.
    let throwaway = [0xDDu8; 16];
    store
        .put(
            &throwaway,
            &[1u8; 16],
            &NodeData::new(bytes::Bytes::from("x")),
        )
        .unwrap();
    store.delete_collection(&throwaway).unwrap();
    let rewrite_started = Instant::now();
    store.sync_all().unwrap();
    let baseline_rewrite = rewrite_started.elapsed();

    // Now measure put() latency while full rewrites run continuously in the
    // background — each iteration forces a fresh invalidation so the
    // background thread keeps taking the full-rewrite path throughout.
    let done = std::sync::atomic::AtomicBool::new(false);
    let rewrites_done = AtomicU64::new(0);
    let max_put_latency_micros = AtomicU64::new(0);
    // Set by the rewrite thread after its very first sync_all, so the
    // sampler cannot finish all 500 puts before any rewrite has begun —
    // without this, a fast/lucky runner yields a serial (non-concurrent)
    // sample with REWRITES=0.
    let first_rewrite_done = std::sync::atomic::AtomicBool::new(false);

    std::thread::scope(|scope| {
        {
            let store = &store;
            let done = &done;
            let rewrites_done = &rewrites_done;
            let first_rewrite_done = &first_rewrite_done;
            scope.spawn(move || {
                let mut r = 0u64;
                while !done.load(Ordering::Relaxed) {
                    let mut throwaway = [0u8; 16];
                    throwaway[0] = 0xDD;
                    throwaway[1..9].copy_from_slice(&r.to_le_bytes());
                    store
                        .put(
                            &throwaway,
                            &[1u8; 16],
                            &NodeData::new(bytes::Bytes::from("x")),
                        )
                        .unwrap();
                    store.delete_collection(&throwaway).unwrap();
                    store.sync_all().unwrap();
                    r += 1;
                    if r == 1 {
                        first_rewrite_done.store(true, Ordering::Relaxed);
                    }
                }
                rewrites_done.store(r, Ordering::Relaxed);
            });
        }
        {
            let store = &store;
            let done = &done;
            let max_put_latency_micros = &max_put_latency_micros;
            let first_rewrite_done = &first_rewrite_done;
            scope.spawn(move || {
                // Start barrier: don't begin sampling until the background
                // rewrite thread has completed its first sync_all, so the
                // sampled put()s actually run concurrently with rewrites.
                // Bounded, not an unconditional spin: if the rewrite thread
                // panics (e.g. an `.unwrap()` on setup/IO) before setting
                // the flag, an unconditional wait here would spin forever,
                // since `thread::scope` only surfaces that panic once every
                // spawned thread returns. Time out instead, so this thread
                // itself unwinds and the scope can join and propagate the
                // real failure.
                let barrier_deadline = Instant::now() + std::time::Duration::from_secs(30);
                while !first_rewrite_done.load(Ordering::Relaxed) {
                    if Instant::now() >= barrier_deadline {
                        // Signal the rewrite thread's `while !done` loop
                        // before panicking, whether it's dead (a panic) or
                        // just slow (a genuinely large first full rewrite on
                        // a mechanical drive, which this project targets, can
                        // plausibly exceed 30s) — either way, panicking here
                        // without this would leave that loop spinning
                        // forever with nothing left to ever set `done`,
                        // hanging the process instead of surfacing a
                        // failure.
                        done.store(true, Ordering::Relaxed);
                        panic!(
                            "background rewrite thread never completed its first sync_all \
                             within 30s (it may have panicked before reaching that point, \
                             or simply be slower than this timeout on this disk)"
                        );
                    }
                    std::thread::yield_now();
                }
                let mut collection = [0u8; 16];
                collection[0] = 0xEE;
                for i in 0..500u64 {
                    let mut id = [0u8; 16];
                    id[1..9].copy_from_slice(&i.to_le_bytes());
                    let started = Instant::now();
                    store
                        .put(&collection, &id, &NodeData::new(bytes::Bytes::from("y")))
                        .unwrap();
                    let elapsed_micros = started.elapsed().as_micros() as u64;
                    max_put_latency_micros.fetch_max(elapsed_micros, Ordering::Relaxed);
                }
                done.store(true, Ordering::Relaxed);
            });
        }
    });

    let max_put_latency =
        std::time::Duration::from_micros(max_put_latency_micros.load(Ordering::Relaxed));

    eprintln!(
        "bench: checkpoint rewrite vs. concurrent put latency ({collections} collections x \
         {records_per_collection} records)"
    );
    println!(
        "bench: checkpoint_latency COLLECTIONS={collections} RECORDS_PER_COLLECTION={records_per_collection} \
         REWRITE_MS={:.3} PUT_APPEND_MAX_MS={:.3} REWRITES={}",
        baseline_rewrite.as_secs_f64() * 1e3,
        max_put_latency.as_secs_f64() * 1e3,
        rewrites_done.load(Ordering::Relaxed),
    );
    eprintln!("  baseline full rewrite:      {baseline_rewrite:.2?}");
    eprintln!("  max concurrent append:      {max_put_latency:.2?}");
    eprintln!(
        "  concurrent rewrites during put sampling: {}",
        rewrites_done.load(Ordering::Relaxed)
    );
    eprintln!("  Note: PUT_APPEND_MAX_MS is append/page-cache latency only — the timed put()");
    eprintln!("  does NOT fsync; durability is committed by the background thread's separate");
    eprintln!("  sync_all(). It approaching or exceeding REWRITE_MS here does NOT by itself mean");
    eprintln!(
        "  the lock scope regressed — put() still does its own buffered write, which can queue"
    );
    eprintln!(
        "  behind the rewrite's write+fsync at the OS/disk level even though no lock is held."
    );
    eprintln!(
        "  That's real, expected device contention, not a correctness bug. The lock-scope claim"
    );
    eprintln!(
        "  itself (put does not wait ON THE LOCK for a slow rewrite) is proven deterministically,"
    );
    eprintln!(
        "  independent of disk speed, by test_put_does_not_block_on_slow_checkpoint_rewrite."
    );
    eprintln!("  Use this bench to track real append-latency trends, not as a pass/fail signal.");
    eprintln!();

    drop(store);
    let _ = fs::remove_dir_all(&dir);
}

/// Measure the cost of unknown-key lookups separately from successful reads.
/// Plain `get` should be an index probe; `get_many_with_refresh` additionally
/// shows the cost and frequency of the read-miss refresh policy.
fn run_unknown_key_benchmark() {
    let dir = bench_root().join("mtxdb_bench_unknown_keys");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    const RECORDS: usize = 10_000;
    const MISSES: usize = 1_000;
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let records: Vec<_> = (0..RECORDS)
        .map(|index| {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&(index as u64).to_be_bytes());
            (id, NodeData::new(bytes::Bytes::from_static(b"benchmark")))
        })
        .collect();
    store.put_many(&ROOM, &records).unwrap();
    store.sync().unwrap();

    let missing: Vec<NodeId> = (RECORDS..RECORDS + MISSES)
        .map(|index| {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&(index as u64).to_be_bytes());
            id
        })
        .collect();

    store.set_stats_enabled(true);
    let direct_start = Instant::now();
    let direct_found = missing
        .iter()
        .filter(|id| std::hint::black_box(store.get(&ROOM, id).unwrap()).is_some())
        .count();
    let direct_elapsed = direct_start.elapsed();
    let direct_stats = store.stats();

    store.reset_stats();
    let refresh_start = Instant::now();
    let refresh_found = missing
        .iter()
        .filter(|id| {
            std::hint::black_box(
                store
                    .get_many_with_refresh(&ROOM, std::slice::from_ref(*id))
                    .unwrap()[0]
                    .is_some(),
            )
        })
        .count();
    let refresh_elapsed = refresh_start.elapsed();
    let refresh_stats = store.stats();

    println!(
        "bench: unknown_keys RECORDS={RECORDS} MISSES={MISSES} \
         DIRECT_FOUND={direct_found} DIRECT_NS_PER_LOOKUP={} \
         REFRESH_FOUND={refresh_found} REFRESH_NS_PER_LOOKUP={} \
         REFRESHES={} REFRESH_SKIPS={} REBUILDS={}",
        direct_elapsed.as_nanos() / MISSES as u128,
        refresh_elapsed.as_nanos() / MISSES as u128,
        refresh_stats.miss_refreshes,
        refresh_stats.miss_refresh_skips,
        refresh_stats.index_rebuild_count.saturating_sub(direct_stats.index_rebuild_count),
    );
    eprintln!(
        "unknown-key lookup cost: direct={direct_elapsed:?}, refresh-aware={refresh_elapsed:?}; \
         refreshes={}, skips={}, rebuilds={}",
        refresh_stats.miss_refreshes,
        refresh_stats.miss_refresh_skips,
        refresh_stats
            .index_rebuild_count
            .saturating_sub(direct_stats.index_rebuild_count),
    );

    let _ = fs::remove_dir_all(&dir);
}

/// Reader-side cost of the read-committed overlay when a writer keeps
/// checkpointing and reclaiming the journal underneath it.
///
/// A reclaim that outruns the reader's loaded coverage forces
/// `reload_index_from_checkpoint()` — a full checkpoint re-scan — on the read
/// path, retried up to 8× internally (≤191 ms of backoff) before the read
/// fails. This is the cost the delete/purge paths pay: `censor_events` and
/// `purge_events` read event JSON through `get_read_committed`, and a
/// concurrent writer sync+reclaim can invalidate the overlay mid-read. Reports
/// per-read latency and how many reads exhausted the retry budget
/// (`WouldBlock`) or hit a genuine gap (`Corrupt`) — the 500 path.
fn run_read_committed_reload_benchmark(
    seed_collections: usize,
    seed_records: usize,
    read_ops: usize,
) {
    use mtxdb_core::journal::Journal;
    use mtxdb_core::storage::StorageError;

    let dir = bench_root().join(format!(
        "mtxdb_bench_read_committed_reload_{seed_collections}x{seed_records}"
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.bin");

    let collection = [0xC1u8; 16];
    let records: Vec<(NodeId, NodeData)> = (0..seed_records)
        .map(|index| {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&(index as u64).to_be_bytes());
            (id, NodeData::new(bytes::Bytes::from_static(b"seed")))
        })
        .collect();

    // Durable seed: a non-trivial index for the reader to reload.
    let seed = PackfileStorage::open(dir.clone()).unwrap();
    for c in 0..seed_collections {
        let mut coll = [0xC0u8; 16];
        coll[1..9].copy_from_slice(&(c as u64).to_be_bytes());
        seed.put_many(&coll, &records).unwrap();
    }
    seed.put_many(&collection, &records).unwrap();
    seed.sync_all().unwrap();
    drop(seed);

    // Empty journal segment the reader attaches to.
    let (journal, _) = Journal::open(&wal).unwrap();
    drop(journal);

    let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
    reader.enable_read_journal(&wal).unwrap();

    let writer = PackfileStorage::open(dir.clone()).unwrap();
    writer.enable_journal(&wal).unwrap();

    let read_ids: Vec<NodeId> = records.iter().map(|(id, _)| *id).collect();
    let done = AtomicBool::new(false);
    let writer_rounds = AtomicU64::new(0);
    let mut latencies_us: Vec<u128> = Vec::with_capacity(read_ops);
    let mut retryable = 0usize;
    let mut corrupt = 0usize;

    std::thread::scope(|scope| {
        {
            let writer = &writer;
            let done = &done;
            let writer_rounds = &writer_rounds;
            scope.spawn(move || {
                let mut round = 0u64;
                while !done.load(Ordering::Relaxed) {
                    // Create then delete a throwaway collection so each sync
                    // takes the full-checkpoint path. A delta append would not
                    // advance coverage or reclaim, so the reader would never
                    // need to reload and the bench would measure nothing.
                    let mut throwaway = [0xDDu8; 16];
                    throwaway[1..9].copy_from_slice(&round.to_le_bytes());
                    writer
                        .put(
                            &throwaway,
                            &[1u8; 16],
                            &NodeData::new(bytes::Bytes::from_static(b"x")),
                        )
                        .unwrap();
                    writer.delete_collection(&throwaway).unwrap();
                    writer.sync_all().unwrap();
                    round += 1;
                    writer_rounds.store(round, Ordering::Relaxed);
                }
            });
        }

        for _ in 0..read_ops {
            let start = Instant::now();
            match std::hint::black_box(reader.get_read_committed(&collection, &read_ids)) {
                Ok(_) => {}
                Err(StorageError::Io(error))
                    if error.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    retryable += 1;
                }
                Err(_) => corrupt += 1,
            }
            latencies_us.push(start.elapsed().as_micros());
        }
        done.store(true, Ordering::Relaxed);
    });

    latencies_us.sort_unstable();
    let percentile = |p: usize| -> u128 {
        let index = latencies_us.len().saturating_sub(1) * p / 100;
        latencies_us.get(index).copied().unwrap_or(0)
    };
    let (p50, p95, p99) = (percentile(50), percentile(95), percentile(99));
    let max = latencies_us.last().copied().unwrap_or(0);
    let rounds = writer_rounds.load(Ordering::Relaxed);

    println!(
        "bench: read_committed_reload SEED={seed_collections}x{seed_records} READS={read_ops} \
         WRITER_ROUNDS={rounds} WOULD_BLOCK={retryable} CORRUPT={corrupt} \
         P50_US={p50} P95_US={p95} P99_US={p99} MAX_US={max}"
    );
    eprintln!(
        "read-committed reload: {read_ops} reads against {rounds} writer checkpoint+reclaim \
         rounds; p50={p50}us p95={p95}us p99={p99}us max={max}us; \
         exhausted-retry-budget (WouldBlock)={retryable}, corrupt={corrupt}"
    );

    let _ = fs::remove_dir_all(&dir);
}

fn main() {
    eprintln!("mdb benchmark harness — cold-read measurement");
    eprintln!("Note: shard Drop deletes superseded shard files on drop,");
    eprintln!("so we clear the cache and re-read from open packfiles.");
    eprintln!();

    run_benchmark("small", 1_000, 2_000);
    run_benchmark("medium", 10_000, 500);
    let large = run_benchmark("large", 100_000, 2_000);
    let pressure = run_benchmark("pressure", 100_000, 100);

    run_open_size_sweep();

    run_intent_benchmark(20_000);

    run_reaction_swarm_benchmark(50_000, 500);

    run_repack_benchmark(10_000, 1_000);
    run_repack_benchmark(20_000, 1_000);

    run_checkpoint_rewrite_latency_benchmark(60, 5_000);

    run_unknown_key_benchmark();

    run_read_committed_reload_benchmark(32, 2_000, 300);

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
    eprintln!(
        "  read syscalls (cold):  {}",
        format_io_metric(large.read_syscalls)
    );
    eprintln!("  index loss rate:       {:.4}%", large.index_loss_rate);
    eprintln!("  warm cache hit rate:   {:.4}%", large.warm_hit_rate);
    eprintln!(
        "  cold throughput:       {:.0} gets/sec",
        large.cold_gets_per_sec
    );
    eprintln!(
        "  warm throughput:       {:.0} gets/sec ({:.4}x cold)",
        large.warm_gets_per_sec,
        large.warm_gets_per_sec / large.cold_gets_per_sec
    );
    eprintln!("  small cache warm hit:  {:.4}%", pressure.warm_hit_rate);
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
