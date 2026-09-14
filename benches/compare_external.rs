//! Three-way synthetic comparison — mtxdb (`PackfileStorage`) vs MDBX vs
//! SQLite — so the "is mtxdb actually faster at moderate inputs?" claim from
//! `DESIGN-open-and-index-persistence.md` is measured against the real
//! baselines, not argued.
//!
//! Opt-in, never built or shipped by default:
//!
//! ```text
//! cargo bench --features compare-external --bench compare_external
//! MTXDB_BENCH_EXT_GB=1,10 cargo bench --features compare-external --bench compare_external
//! ```
//!
//! Same synthetic dataset on every engine (16-byte keys, 1 KB incompressible
//! values, deterministic, same RNG), so the numbers are directly comparable:
//! batch write, warm/cold open, sampled point lookups, small batch append,
//! plus resident-index memory for mtxdb and on-disk file bytes for MDBX/SQLite
//! (reported under distinct labels — the two are *not* the same quantity).
//!
//! Not part of the public crate API. Lints are relaxed here for the same
//! reason as `benches/storage.rs`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::borrow::Cow;
use std::fs;
use std::time::Instant;

use mtxdb_core::storage::{NodeData, NodeId, StorageEngine};
use mtxdb_core::PackfileStorage;

// ── Shared deterministic RNG + dataset builders ──────────────────────

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

/// Well-mixed 64-bit permutation (splitmix64); matches `benches/storage.rs`
/// so record IDs mean the same thing across both harnesses.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The 16-byte content-addressable node ID for synthetic record `idx`.
fn node_id(idx: usize) -> NodeId {
    let mut id = [0u8; 16];
    let a = splitmix64(idx as u64);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

/// Deterministic, incompressible 1 KB payload seeded per node. A shared
/// constant payload would zstd-compress to near-zero where an engine stores
/// pre-compressed records (mtxdb) and understate the honest dataset size.
fn payload(seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(1024);
    while out.len() < 1024 {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    out
}

/// Total bytes of regular files under `dir` (dataset + aux files).
fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            total += entry.metadata().map_or(0, |m| m.len());
        }
    }
    total
}

/// Process-wide (RSS, PSS) in bytes from `/proc/self/smaps_rollup`, best
/// effort. PSS apportions shared file-backed pages (mmap'd packfiles, the
/// mmap-backed MDBX/SQLite files) across mappings, so it is the honest
/// cross-engine number; RSS overcounts shared pages if this process ever
/// shares them with another. Returns `(0, 0)` on non-Linux or if the file
/// is unreadable, and callers must not treat that as a real zero.
fn smaps_rollup() -> (u64, u64) {
    let Ok(contents) = fs::read_to_string("/proc/self/smaps_rollup") else {
        return (0, 0);
    };
    let parse_kb = |line: &str, prefix: &str| -> Option<u64> {
        line.strip_prefix(prefix).map(|rest| {
            rest.split_whitespace()
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
                * 1024
        })
    };
    let mut rss = 0u64;
    let mut pss = 0u64;
    for line in contents.lines() {
        if let Some(v) = parse_kb(line, "Rss:") {
            rss = v;
        } else if let Some(v) = parse_kb(line, "Pss:") {
            pss = v;
        }
    }
    (rss, pss)
}

/// Evict every file under `dir` from the page cache (best-effort, vmtouch).
/// Same rationale as `benches/storage.rs::drop_caches_for_dir`.
fn drop_caches_for_dir(dir: &std::path::Path) -> bool {
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

// ── Per-engine runs ─────────────────────────────────────────────────

const PAYLOAD_BYTES: usize = 1024;
const COLLECTIONS: usize = 32;
const LOOKUP_SAMPLES: usize = 100_000;
const APPEND_RECORDS: usize = 1_000;
/// Small post-growth transactions used to measure the steady-state delta-log
/// path separately from the first append, which may legitimately resize an
/// index and therefore require a full checkpoint rewrite.
const STEADY_APPEND_RECORDS: usize = 256;
const STEADY_APPEND_BATCHES: usize = 3;
/// Structural grows can land inside the sampling window at any dataset size.
/// Keep sampling until this many delta-path batches are collected, bounded so
/// a broken delta path fails clearly instead of emitting a NaN average.
const MAX_STEADY_APPEND_ATTEMPTS: usize = STEADY_APPEND_BATCHES * 8;
/// The point-read sweep uses unique IDs, so it measures cache misses. Keep
/// the comparison cache-free by default; opt into the production 100k-entry
/// per-collection cache with `MTXDB_BENCH_CACHE_CAPACITY=100000` when
/// measuring a reuse/cache-hit workload.
const DEFAULT_BENCH_CACHE_CAPACITY: usize = 0;
const BUILD_BATCH_RECORDS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Mtxdb,
    Mdbx,
    Sqlite,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Mtxdb => "mtxdb",
            Self::Mdbx => "mdbx",
            Self::Sqlite => "sqlite",
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Run {
    write_ms: f64,
    warm_open_ms: f64,
    cold_open_ms: f64,
    lookup_us: f64,
    append_ms: f64,
    append_puts_ms: f64,
    append_sync_ms: f64,
    append_loop_ms: Option<f64>,
    append_sync_all_ms: Option<f64>,
    steady_append_ms: f64,
    steady_append_puts_ms: f64,
    steady_append_sync_ms: f64,
    files: u64,
    mem: u64,
    mem_label: &'static str,
    /// (RSS, PSS) bytes from `/proc/self/smaps_rollup`, sampled right after
    /// the warm open (before any lookups touch pages) and again after the
    /// sampled point-lookup pass. This isolates open-time residency from
    /// pages the lookup pass pulls in, and is the one apples-to-apples
    /// memory number across engines — `mem`/`mem_label` above are not.
    rss_pss_open: (u64, u64),
    rss_pss_warm: (u64, u64),
}

fn collection_for(node: usize) -> [u8; 16] {
    let mut collection = [0u8; 16];
    collection[..8].copy_from_slice(&(node % COLLECTIONS).to_le_bytes());
    collection
}

fn checksum_policy_from_env() -> mtxdb_core::packfile::ChecksumPolicy {
    // MTXDB_BENCH_CHECKSUM=writeonly|disabled lowers the per-frame checksum
    // policy from the default Full (see ChecksumPolicy): writeonly still
    // writes real CRCs but skips the hashing pass on point-lookup reads,
    // disabled also stops computing them on writes.
    match std::env::var("MTXDB_BENCH_CHECKSUM").as_deref() {
        Ok("writeonly") => mtxdb_core::packfile::ChecksumPolicy::WriteOnly,
        Ok("disabled") => mtxdb_core::packfile::ChecksumPolicy::Disabled,
        _ => mtxdb_core::packfile::ChecksumPolicy::Full,
    }
}

fn cache_capacity_from_env() -> usize {
    match std::env::var("MTXDB_BENCH_CACHE_CAPACITY") {
        Ok(raw) => raw
            .parse()
            .expect("MTXDB_BENCH_CACHE_CAPACITY must be a non-negative integer"),
        Err(std::env::VarError::NotPresent) => DEFAULT_BENCH_CACHE_CAPACITY,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MTXDB_BENCH_CACHE_CAPACITY must be valid UTF-8")
        }
    }
}

fn mtxdb_open(dir: &std::path::Path) -> PackfileStorage {
    // MTXDB_BENCH_COMPRESS=0 opens the store with per-record zstd disabled, to
    // measure the raw append path against mdbx/sqlite. The bench payload is
    // seeded-incompressible anyway, so compression only pure overhead here.
    let compress_off = std::env::var("MTXDB_BENCH_COMPRESS").as_deref() == Ok("0");
    let checksum = checksum_policy_from_env();
    let cache_capacity = cache_capacity_from_env();
    // This bench syncs explicitly (sync_all after build, sync after append),
    // so it opts into the buffered append policy: frames accumulate for one
    // positioned write per ~1 MiB instead of one per record.
    let policy = mtxdb_core::shard::AppendPolicy::buffered();
    PackfileStorage::open_with_cache_and_policies(
        dir.to_path_buf(),
        cache_capacity,
        !compress_off,
        checksum,
    )
    .unwrap()
    .with_append_policy(policy)
}

fn mtxdb_open_read_only(dir: &std::path::Path) -> PackfileStorage {
    PackfileStorage::open_read_only_with_cache_and_policies(
        dir.to_path_buf(),
        cache_capacity_from_env(),
        checksum_policy_from_env(),
    )
    .unwrap()
}

fn run_mtxdb(dir: &std::path::Path, nodes: usize) -> Run {
    // ── Build ──
    let started = Instant::now();
    let store_rw = mtxdb_open(dir);
    for bucket in 0..COLLECTIONS {
        for first in (bucket..nodes).step_by(COLLECTIONS * BUILD_BATCH_RECORDS) {
            let entries = (first..nodes.min(first + COLLECTIONS * BUILD_BATCH_RECORDS))
                .step_by(COLLECTIONS)
                .map(|node| {
                    (
                        node_id(node),
                        NodeData::new(bytes::Bytes::from(payload(node as u64))),
                    )
                })
                .collect::<Vec<_>>();
            store_rw
                .put_many(&collection_for(bucket), &entries)
                .unwrap();
        }
    }
    store_rw.sync_all().unwrap();
    let write_ms = started.elapsed().as_secs_f64() * 1e3;
    if let Some(sync) = store_rw.sync_timings() {
        eprintln!(
            "    mtxdb sync_all: flush {:.2}ms + fsync {:.2}ms + sidecar {:.2}ms + delta {:.2}ms + checkpoint {:.2}ms = {:.2}ms",
            sync.pack_flush.as_secs_f64() * 1e3,
            sync.pack_fsync.as_secs_f64() * 1e3,
            sync.sidecar.as_secs_f64() * 1e3,
            sync.delta_log.as_secs_f64() * 1e3,
            sync.checkpoint.as_secs_f64() * 1e3,
            sync.total.as_secs_f64() * 1e3,
        );
    }
    drop(store_rw);

    // Resident index bytes, isolated from the mmap'd packfiles: sum of the
    // per-collection LossyIndex slot arrays (the checkpoint-size input).
    let (mem, mem_label) = {
        let store = mtxdb_open_read_only(dir);
        let bytes: u64 = store
            .collection_summaries()
            .iter()
            .map(|(_, _, b, _)| *b as u64)
            .sum();
        drop(store);
        (bytes, "index_bytes")
    };
    let files = dir_bytes(dir);

    // ── Warm open + sampled point lookups ──
    let started = Instant::now();
    let store = mtxdb_open_read_only(dir);
    let warm_open_ms = started.elapsed().as_secs_f64() * 1e3;
    if let Some(open) = store.open_timings() {
        eprintln!(
            "    mtxdb open path={:?}: shard {:.2}ms + metadata {:.2}ms + checkpoint {:.2}ms + fingerprint {:.2}ms + replay {:.2}ms + materialize {:.2}ms + full_scan {:.2}ms = {:.2}ms",
            open.path,
            open.shard_open.as_secs_f64() * 1e3,
            open.metadata_load.as_secs_f64() * 1e3,
            open.checkpoint_decode.as_secs_f64() * 1e3,
            open.fingerprint.as_secs_f64() * 1e3,
            open.delta_replay.as_secs_f64() * 1e3,
            open.index_materialization.as_secs_f64() * 1e3,
            open.full_scan.as_secs_f64() * 1e3,
            open.total.as_secs_f64() * 1e3,
        );
    }
    let rss_pss_open = smaps_rollup();
    let lookup_started = Instant::now();
    for node in 0..LOOKUP_SAMPLES.min(nodes) {
        assert!(
            store
                .get(&collection_for(node), &node_id(node))
                .unwrap()
                .is_some(),
            "every written node must be readable back"
        );
    }
    let lookup_us = lookup_started.elapsed().as_secs_f64() * 1e6 / LOOKUP_SAMPLES.min(nodes) as f64;
    let rss_pss_warm = smaps_rollup();
    drop(store);

    // ── Cold checkpoint open + append ──
    // This deliberately measures reopening the persisted index after page
    // eviction, which is comparable to MDBX/SQLite reopening their persisted
    // B-trees. A checkpoint-free mtxdb index rebuild is a distinct diagnostic,
    // not an apples-to-apples external-engine metric.
    let evicted = drop_caches_for_dir(dir);
    let started = Instant::now();
    let store = mtxdb_open_read_only(dir);
    let cold_open_ms = started.elapsed().as_secs_f64() * 1e3;
    drop(store);

    // ── Append ──
    let store_rw = mtxdb_open(dir);
    // (a) apples-to-apples "one batch, one sync": 1k nodes across the 32
    //     collections, one generation per collection via put_many, then a
    //     single dirty-scoped sync() — the primitive Synapse's 1 s timer
    //     calls, NOT sync_all(). That pairs with MDBX's one txn/commit and
    //     SQLite's one tx/commit below.
    let append_puts_started = Instant::now();
    {
        for bucket in 0..COLLECTIONS {
            let mut entries = Vec::with_capacity(APPEND_RECORDS / COLLECTIONS + 1);
            for i in 0..APPEND_RECORDS {
                let node = nodes + i;
                if node % COLLECTIONS == bucket {
                    entries.push((
                        node_id(node),
                        NodeData::new(bytes::Bytes::from(payload(node as u64))),
                    ));
                }
            }
            store_rw
                .put_many(&collection_for(bucket), &entries)
                .unwrap();
        }
    }
    let append_puts_ms = append_puts_started.elapsed().as_secs_f64() * 1e3;

    let append_sync_started = Instant::now();
    store_rw.sync().unwrap();
    let append_sync_ms = append_sync_started.elapsed().as_secs_f64() * 1e3;
    let append_ms = append_puts_ms + append_sync_ms;
    if let Some(sync) = store_rw.sync_timings() {
        eprintln!(
            "    mtxdb sync: flush {:.2}ms + fsync {:.2}ms + delta {:.2}ms + checkpoint {:.2}ms = {:.2}ms",
            sync.pack_flush.as_secs_f64() * 1e3,
            sync.pack_fsync.as_secs_f64() * 1e3,
            sync.delta_log.as_secs_f64() * 1e3,
            sync.checkpoint.as_secs_f64() * 1e3,
            sync.total.as_secs_f64() * 1e3,
        );
    }

    // One sync_all() on the same store: isolates the full-fsync + whole-sidecar
    // rewrite that sync()/sync_dirty skips — the doc footnote, not the number.
    let append_sync_all_started = Instant::now();
    store_rw.sync_all().unwrap();
    let append_sync_all_ms = append_sync_all_started.elapsed().as_secs_f64() * 1e3;

    // (b) The first append above can be the capacity-boundary transaction:
    // report the ordinary append-only case separately, after that one-time
    // grow has been checkpointed. Each small batch stays well below the new
    // table's next 75%-load boundary and should take the delta path — but
    // whether a *later* boundary lands inside this fixed window depends on
    // the dataset size (a grow invalidates the delta log, so that batch's
    // sync falls back to a full checkpoint rewrite). A contaminated batch is
    // excluded from the steady averages. Collect the requested number of
    // delta-path samples even if a boundary lands in this window.
    let mut steady_puts_ms = 0.0;
    let mut steady_sync_ms = 0.0;
    let mut steady_delta_batches = 0;
    let mut steady_attempts: usize = 0;
    for batch in 0..MAX_STEADY_APPEND_ATTEMPTS {
        steady_attempts = steady_attempts.saturating_add(1);
        let puts_started = Instant::now();
        for bucket in 0..COLLECTIONS {
            let mut entries = Vec::with_capacity(STEADY_APPEND_RECORDS / COLLECTIONS + 1);
            for i in 0..STEADY_APPEND_RECORDS {
                let node = nodes + APPEND_RECORDS + batch * STEADY_APPEND_RECORDS + i;
                if node % COLLECTIONS == bucket {
                    entries.push((
                        node_id(node),
                        NodeData::new(bytes::Bytes::from(payload(node as u64))),
                    ));
                }
            }
            store_rw
                .put_many(&collection_for(bucket), &entries)
                .unwrap();
        }
        let sync_started = Instant::now();
        store_rw.sync().unwrap();
        let sync = store_rw.sync_timings().expect("sync must be timed");
        if sync.delta_log > std::time::Duration::ZERO
            && sync.checkpoint == std::time::Duration::ZERO
        {
            steady_delta_batches += 1;
            steady_puts_ms += puts_started.elapsed().as_secs_f64() * 1e3;
            steady_sync_ms += sync_started.elapsed().as_secs_f64() * 1e3;
            if steady_delta_batches == STEADY_APPEND_BATCHES {
                break;
            }
        } else {
            eprintln!(
                "    note: steady batch {batch} grew a collection index, so its sync \
                 took the full-checkpoint path; excluded from the steady average"
            );
        }
    }
    assert!(
        steady_delta_batches == STEADY_APPEND_BATCHES,
        "collected {steady_delta_batches}/{STEADY_APPEND_BATCHES} delta-path steady batches \
         in {MAX_STEADY_APPEND_ATTEMPTS} attempts; cannot report a representative steady append"
    );
    let steady_append_puts_ms = steady_puts_ms / steady_delta_batches as f64;
    let steady_append_sync_ms = steady_sync_ms / steady_delta_batches as f64;
    let steady_append_ms = steady_append_puts_ms + steady_append_sync_ms;

    // (c) periodic durability on the real per-event path: 1k individual puts
    //     with no sync in the window. The 1 s timer absorbs the fsync cost
    //     elsewhere, so this is the latency an append adds in production.
    let append_loop_started = Instant::now();
    for i in 0..APPEND_RECORDS {
        let node = nodes + APPEND_RECORDS + steady_attempts * STEADY_APPEND_RECORDS + i;
        store_rw
            .put(
                &collection_for(node),
                &node_id(node),
                &NodeData::new(bytes::Bytes::from(payload(node as u64))),
            )
            .unwrap();
    }
    let append_loop_ms = append_loop_started.elapsed().as_secs_f64() * 1e3;
    drop(store_rw);

    if !evicted {
        eprintln!("  Note: vmtouch unavailable or failed; no disk-cold claim is made (mtxdb).");
    }

    Run {
        write_ms,
        warm_open_ms,
        cold_open_ms,
        lookup_us,
        append_ms,
        append_puts_ms,
        append_sync_ms,
        append_loop_ms: Some(append_loop_ms),
        append_sync_all_ms: Some(append_sync_all_ms),
        steady_append_ms,
        steady_append_puts_ms,
        steady_append_sync_ms,
        files,
        mem,
        mem_label,
        rss_pss_open,
        rss_pss_warm,
    }
}

#[allow(clippy::used_underscore_binding)]
fn run_mdbx(dir: &std::path::Path, nodes: usize) -> Run {
    use libmdbx::{Database, NoWriteMap, TableFlags, WriteFlags};

    // ── Build ──
    let started = Instant::now();
    {
        let db: Database<NoWriteMap> = Database::open(dir).unwrap();
        let txn = db.begin_rw_txn().unwrap();
        let table = txn.create_table(None, TableFlags::empty()).unwrap();
        for node in 0..nodes {
            let p = payload(node as u64);
            txn.put(&table, node_id(node), &p, WriteFlags::empty())
                .unwrap();
        }
        txn.commit().unwrap();
        drop(db);
    }
    let write_ms = started.elapsed().as_secs_f64() * 1e3;

    let files = dir_bytes(dir);
    // MDBX keeps its entire index+data in the mmap'd database file; report
    // that footprint rather than a process-RSS guess.
    let mem = files;
    let mem_label = "file_bytes";

    // ── Warm open + sampled point lookups ──
    let started = Instant::now();
    let db: Database<NoWriteMap> = Database::open(dir).unwrap();
    let txn = db.begin_ro_txn().unwrap();
    let table = txn.open_table(None).unwrap();
    let warm_open_ms = started.elapsed().as_secs_f64() * 1e3;
    let rss_pss_open = smaps_rollup();
    let lookup_started = Instant::now();
    for node in 0..LOOKUP_SAMPLES.min(nodes) {
        let key = node_id(node);
        assert!(
            txn.get::<Cow<'_, [u8]>>(&table, &key).unwrap().is_some(),
            "every written key must be readable back"
        );
    }
    let lookup_us = lookup_started.elapsed().as_secs_f64() * 1e6 / LOOKUP_SAMPLES.min(nodes) as f64;
    let rss_pss_warm = smaps_rollup();
    drop(txn);
    drop(db);

    // ── Cold open + append ──
    let evicted = drop_caches_for_dir(dir);
    let started = Instant::now();
    let db: Database<NoWriteMap> = Database::open(dir).unwrap();
    {
        let txn = db.begin_ro_txn().unwrap();
        let _ = txn.open_table(None).unwrap();
    }
    let cold_open_ms = started.elapsed().as_secs_f64() * 1e3;

    let append_puts_started = Instant::now();
    let txn = db.begin_rw_txn().unwrap();
    let table = txn.open_table(None).unwrap();
    for i in 0..APPEND_RECORDS {
        let node = nodes + i;
        let p = payload(node as u64);
        txn.put(&table, node_id(node), &p, WriteFlags::empty())
            .unwrap();
    }
    let append_puts_ms = append_puts_started.elapsed().as_secs_f64() * 1e3;
    let append_sync_started = Instant::now();
    txn.commit().unwrap();
    let append_sync_ms = append_sync_started.elapsed().as_secs_f64() * 1e3;
    let append_ms = append_puts_ms + append_sync_ms;

    let mut steady_puts_ms = 0.0;
    let mut steady_sync_ms = 0.0;
    for batch in 0..STEADY_APPEND_BATCHES {
        let puts_started = Instant::now();
        let txn = db.begin_rw_txn().unwrap();
        let table = txn.open_table(None).unwrap();
        for i in 0..STEADY_APPEND_RECORDS {
            let node = nodes + APPEND_RECORDS + batch * STEADY_APPEND_RECORDS + i;
            let p = payload(node as u64);
            txn.put(&table, node_id(node), &p, WriteFlags::empty())
                .unwrap();
        }
        steady_puts_ms += puts_started.elapsed().as_secs_f64() * 1e3;
        let sync_started = Instant::now();
        txn.commit().unwrap();
        steady_sync_ms += sync_started.elapsed().as_secs_f64() * 1e3;
    }
    drop(db);

    if !evicted {
        eprintln!("  Note: vmtouch unavailable or failed; no disk-cold claim is made (mdbx).");
    }

    Run {
        write_ms,
        warm_open_ms,
        cold_open_ms,
        lookup_us,
        append_ms,
        append_puts_ms,
        append_sync_ms,
        append_loop_ms: None,
        append_sync_all_ms: None,
        steady_append_ms: (steady_puts_ms + steady_sync_ms) / STEADY_APPEND_BATCHES as f64,
        steady_append_puts_ms: steady_puts_ms / STEADY_APPEND_BATCHES as f64,
        steady_append_sync_ms: steady_sync_ms / STEADY_APPEND_BATCHES as f64,
        files,
        mem,
        mem_label,
        rss_pss_open,
        rss_pss_warm,
    }
}

#[allow(clippy::used_underscore_binding)]
fn run_sqlite(dir: &std::path::Path, nodes: usize) -> Run {
    use rusqlite::{params, Connection};

    let db_path = dir.join("nodes.sqlite");
    fn open(path: &std::path::Path) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL;")
            .unwrap();
        conn
    }

    // ── Build ──
    let started = Instant::now();
    {
        let conn = open(&db_path);
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nodes(pk BLOB PRIMARY KEY, val BLOB) WITHOUT ROWID;",
        )
        .unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        {
            let mut stmt = tx
                .prepare("INSERT OR IGNORE INTO nodes(pk, val) VALUES (?1, ?2)")
                .unwrap();
            for node in 0..nodes {
                let p = payload(node as u64);
                stmt.execute(params![node_id(node), p]).unwrap();
            }
        }
        tx.commit().unwrap();
        drop(conn);
    }
    let write_ms = started.elapsed().as_secs_f64() * 1e3;

    let files = dir_bytes(dir);
    let mem = files;
    let mem_label = "file_bytes";

    // ── Warm open + sampled point lookups ──
    let started = Instant::now();
    let conn = open(&db_path);
    let mut stmt = conn.prepare("SELECT val FROM nodes WHERE pk=?1").unwrap();
    let warm_open_ms = started.elapsed().as_secs_f64() * 1e3;
    let rss_pss_open = smaps_rollup();
    let lookup_started = Instant::now();
    for node in 0..LOOKUP_SAMPLES.min(nodes) {
        let row: Vec<u8> = stmt
            .query_row(params![node_id(node)], |row| row.get(0))
            .unwrap();
        assert!(!row.is_empty(), "every written value must be readable back");
    }
    let lookup_us = lookup_started.elapsed().as_secs_f64() * 1e6 / LOOKUP_SAMPLES.min(nodes) as f64;
    let rss_pss_warm = smaps_rollup();
    drop(stmt);
    drop(conn);

    // ── Cold open + append ──
    let evicted = drop_caches_for_dir(dir);
    let started = Instant::now();
    let conn = open(&db_path);
    drop(conn.prepare("SELECT val FROM nodes WHERE pk=?1").unwrap());
    let cold_open_ms = started.elapsed().as_secs_f64() * 1e3;

    let append_puts_started = Instant::now();
    let tx = conn.unchecked_transaction().unwrap();
    {
        let mut stmt = tx
            .prepare("INSERT OR IGNORE INTO nodes(pk, val) VALUES (?1, ?2)")
            .unwrap();
        for i in 0..APPEND_RECORDS {
            let node = nodes + i;
            let p = payload(node as u64);
            stmt.execute(params![node_id(node), p]).unwrap();
        }
    }
    let append_puts_ms = append_puts_started.elapsed().as_secs_f64() * 1e3;
    let append_sync_started = Instant::now();
    tx.commit().unwrap();
    let append_sync_ms = append_sync_started.elapsed().as_secs_f64() * 1e3;
    let append_ms = append_puts_ms + append_sync_ms;

    let mut steady_puts_ms = 0.0;
    let mut steady_sync_ms = 0.0;
    for batch in 0..STEADY_APPEND_BATCHES {
        let puts_started = Instant::now();
        let tx = conn.unchecked_transaction().unwrap();
        {
            let mut stmt = tx
                .prepare("INSERT OR IGNORE INTO nodes(pk, val) VALUES (?1, ?2)")
                .unwrap();
            for i in 0..STEADY_APPEND_RECORDS {
                let node = nodes + APPEND_RECORDS + batch * STEADY_APPEND_RECORDS + i;
                let p = payload(node as u64);
                stmt.execute(params![node_id(node), p]).unwrap();
            }
        }
        steady_puts_ms += puts_started.elapsed().as_secs_f64() * 1e3;
        let sync_started = Instant::now();
        tx.commit().unwrap();
        steady_sync_ms += sync_started.elapsed().as_secs_f64() * 1e3;
    }
    drop(conn);

    if !evicted {
        eprintln!("  Note: vmtouch unavailable or failed; no disk-cold claim is made (sqlite).");
    }

    Run {
        write_ms,
        warm_open_ms,
        cold_open_ms,
        lookup_us,
        append_ms,
        append_puts_ms,
        append_sync_ms,
        append_loop_ms: None,
        append_sync_all_ms: None,
        steady_append_ms: (steady_puts_ms + steady_sync_ms) / STEADY_APPEND_BATCHES as f64,
        steady_append_puts_ms: steady_puts_ms / STEADY_APPEND_BATCHES as f64,
        steady_append_sync_ms: steady_sync_ms / STEADY_APPEND_BATCHES as f64,
        files,
        mem,
        mem_label,
        rss_pss_open,
        rss_pss_warm,
    }
}

// ── Driver ──────────────────────────────────────────────────────────

/// Root for benchmark scratch data: `MTXDB_BENCH_ROOT` env override, else
/// the session temp dir. Sizeable sweeps (100 GB–1 TB) must not run on a
/// RAM-backed tmpfs; point this at a real disk with headroom.
fn bench_root() -> std::path::PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, std::path::PathBuf::from)
}

fn run_backend(backend: Backend, target_gb: f64) {
    let dir = bench_root().join(format!(
        "mtxdb_bench_compare_{}_{target_gb}",
        backend.name()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let nodes = (target_gb * 1e9) as usize / PAYLOAD_BYTES;
    assert!(nodes > 0, "target_gb too small for the 1 KB payload");

    let run = match backend {
        Backend::Mtxdb => run_mtxdb(&dir, nodes),
        Backend::Mdbx => run_mdbx(&dir, nodes),
        Backend::Sqlite => run_sqlite(&dir, nodes),
    };

    let label = format!("{target_gb:.3}");
    let label = label.trim_end_matches('0').trim_end_matches('.');
    let loop_part = run
        .append_loop_ms
        .map_or(String::new(), |v| format!(" APPEND_LOOP_MS={v:.2}"));
    let sync_all_part = run
        .append_sync_all_ms
        .map_or(String::new(), |v| format!(" APPEND_SYNC_ALL_MS={v:.2}"));
    println!(
        "bench: external ENG={} L={label}gb N={nodes} WRITE_MS={:.1} WARM_OPEN_MS={:.3} \
         COLD_OPEN_MS={:.3} LOOKUP_US={:.2} APPEND={APPEND_RECORDS} APPEND_MS={:.2} \
         APPEND_PUTS_MS={:.2} APPEND_SYNC_MS={:.2}{loop_part}{sync_all_part} \
         STEADY_APPEND={STEADY_APPEND_RECORDS} STEADY_APPEND_MS={:.2} \
         STEADY_APPEND_PUTS_MS={:.2} STEADY_APPEND_SYNC_MS={:.2} \
         FILES={} MEM={} MEM_LABEL={} RSS_OPEN={} PSS_OPEN={} RSS_WARM={} PSS_WARM={} CACHE_CAPACITY={}",
        backend.name(),
        run.write_ms,
        run.warm_open_ms,
        run.cold_open_ms,
        run.lookup_us,
        run.append_ms,
        run.append_puts_ms,
        run.append_sync_ms,
        run.steady_append_ms,
        run.steady_append_puts_ms,
        run.steady_append_sync_ms,
        run.files,
        run.mem,
        run.mem_label,
        run.rss_pss_open.0,
        run.rss_pss_open.1,
        run.rss_pss_warm.0,
        run.rss_pss_warm.1,
        cache_capacity_from_env(),
    );

    let loop_note = run.append_loop_ms.map_or(String::new(), |v| {
        format!(" (per-put loop {v:.2}ms, no sync in window)")
    });
    eprintln!("  [{:>6}] {backend:?} @ {label} GB: write {:.2}s, warm open {:.3}ms, cold open {:.3}ms, lookup {:.2}us, append {:.2}ms = puts {:.2}ms + sync {:.2}ms{loop_note}, files {:.1}MB, mem {:.1}MB ({})",
        backend.name(), run.write_ms / 1e3, run.warm_open_ms, run.cold_open_ms,
        run.lookup_us, run.append_ms, run.append_puts_ms, run.append_sync_ms,
        run.files as f64 / 1e6, run.mem as f64 / 1e6, run.mem_label);

    // mtxdb's mem_label is index_bytes — the checkpoint-size input. Report
    // the fork clearly when the backend's working-set label differs.
    if run.mem_label != "index_bytes" {
        eprintln!(
            "    ({} reports its on-disk file footprint, not an in-RAM index.)",
            backend.name()
        );
    }
    eprintln!(
        "    steady append ({STEADY_APPEND_BATCHES} × {STEADY_APPEND_RECORDS} records): puts {:.2}ms + sync {:.2}ms = {:.2}ms/batch",
        run.steady_append_puts_ms, run.steady_append_sync_ms, run.steady_append_ms
    );
    if run.rss_pss_open == (0, 0) && run.rss_pss_warm == (0, 0) {
        eprintln!("    Note: /proc/self/smaps_rollup unavailable; no RSS/PSS claim is made.");
    } else {
        eprintln!(
            "    memory (PSS, cross-engine comparable): open {:.1}MB -> after lookups {:.1}MB  (RSS: {:.1}MB -> {:.1}MB)",
            run.rss_pss_open.1 as f64 / 1e6,
            run.rss_pss_warm.1 as f64 / 1e6,
            run.rss_pss_open.0 as f64 / 1e6,
            run.rss_pss_warm.0 as f64 / 1e6,
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

/// Parse `MTXDB_BENCH_EXT_ENGINE` (unset -> run all three in one process,
/// the historical default; `mtxdb`/`mdbx`/`sqlite` -> run only that one).
///
/// Single-engine mode exists for `scripts/external_bench.py`, which invokes
/// this binary three times, once per engine, so each backend gets its own
/// fresh process for the RSS/PSS sampling in `run_backend`. Sequential
/// in-process runs share one address space: allocator retention and a prior
/// engine's still-resident pages would bias later engines' PSS/RSS, since
/// `/proc/self/smaps_rollup` reports the whole process, not per-backend.
fn engines_from_env() -> Vec<Backend> {
    match std::env::var("MTXDB_BENCH_EXT_ENGINE") {
        Ok(raw) => match raw.trim() {
            "mtxdb" => vec![Backend::Mtxdb],
            "mdbx" => vec![Backend::Mdbx],
            "sqlite" => vec![Backend::Sqlite],
            other => panic!("invalid MTXDB_BENCH_EXT_ENGINE: {other:?}"),
        },
        Err(std::env::VarError::NotPresent) => {
            vec![Backend::Mtxdb, Backend::Mdbx, Backend::Sqlite]
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MTXDB_BENCH_EXT_ENGINE must be valid UTF-8")
        }
    }
}

fn main() {
    let engines = engines_from_env();
    eprintln!("external comparison bench — mtxdb vs MDBX vs SQLite (default 0.1 GB per engine)");
    eprintln!("set MTXDB_BENCH_EXT_GB=1,100,1000 (comma-separated GB targets) for the real curve");
    if engines.len() == 1 {
        eprintln!(
            "MTXDB_BENCH_EXT_ENGINE={} set; running only that backend in this process",
            engines[0].name()
        );
    }
    eprintln!();

    let gbs: Vec<f64> = match std::env::var("MTXDB_BENCH_EXT_GB") {
        Ok(raw) => raw
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<f64>()
                    .expect("invalid MTXDB_BENCH_EXT_GB")
            })
            .filter(|g| *g > 0.0)
            .collect(),
        Err(std::env::VarError::NotPresent) => vec![0.1],
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("MTXDB_BENCH_EXT_GB must be valid UTF-8")
        }
    };
    assert!(
        !gbs.is_empty(),
        "MTXDB_BENCH_EXT_GB must list at least one size"
    );

    for gb in &gbs {
        for backend in &engines {
            run_backend(*backend, *gb);
        }
    }

    eprintln!();
    eprintln!("Interpretation (see DESIGN-open-and-index-persistence.md §1.2):");
    eprintln!("  cold open is where mtxdb's full-scan index rebuild shows up; lookups");
    eprintln!("  are where the resident LossyIndex should beat the B+tree walkers.");
}
