//! Sidecar-build backlog vs writer latency: does a rotation wait for a pack
//! index build?
//!
//! A pack that fills is sealed and its `.idx` sidecar is built in the
//! background (fsync + checksum scan of the whole pack). Before the worker
//! pool, a rotation that found every builder busy joined the oldest one, so
//! writer latency scaled with sidecar-build time. This bench makes builders
//! fall behind writers on purpose and records what the writer sees.
//!
//! Scenarios (`MTXDB_SEAL_SCENARIOS`, comma list, default all):
//!
//! - `keepup`    builders keep up (no delay): normal-path overhead.
//! - `busy`      every worker busy, then more rotations.
//! - `sustained` writes outpace builders for many packs; backlog and drain.
//! - `multi`     `MTXDB_SEAL_POOLS` pools write concurrently (threads,
//!   memory, disk contention).
//! - `reopen`    cost of an unindexed pack: reopen with sidecars versus
//!   after deleting them (frame-scanner fallback). `ShardPool::read_at` is
//!   offset-addressed and never touches the sidecar, so a pending pack costs
//!   a point read nothing here; the sidecar is paid at open/recovery, and by
//!   a paged index, which this bench does not drive.
//!
//! Two modes: `MTXDB_SEAL_DELAY_MS` adds a fixed delay per build (reproducible
//! saturation; needs the `test-util` feature, which this bench enables), and
//! `0` runs real builds against `MTXDB_BENCH_ROOT` (point it at an HDD or SSD
//! directory). Delay applies to `busy`, `sustained` and `multi`.
//!
//! ```text
//! cargo bench --manifest-path benches/Cargo.toml --bench seal_rotation
//! MTXDB_SEAL_DELAY_MS=0 MTXDB_SEAL_PACK_MIB=256 MTXDB_BENCH_ROOT=/mnt/hdd/x \
//!   cargo bench --manifest-path benches/Cargo.toml --bench seal_rotation
//! ```
//!
//! # Env
//!
//! - `MTXDB_SEAL_PACK_MIB` (default 8): rotation threshold per pack. Small
//!   packs compare quickly; production-size packs expose real scan cost.
//! - `MTXDB_SEAL_PACKS` (default 24): packs written per pool.
//! - `MTXDB_SEAL_PAYLOAD` (default 4096): bytes per record.
//! - `MTXDB_SEAL_DELAY_MS` (default 400): per-build delay, 0 = real builds.
//! - `MTXDB_SEAL_POOLS` (default 4): pools in `multi`.
//!
//! The worker count is `MAX_SEAL_WORKERS` in `src/engine/shard.rs` (4); it is
//! reported, not configurable.
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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::packfile::Record;
use mtxdb::shard::ShardPool;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy)]
struct Cfg {
    pack_bytes: u64,
    packs: u64,
    payload: usize,
    delay: Duration,
    pools: usize,
}

fn cfg() -> Cfg {
    Cfg {
        pack_bytes: env_u64("MTXDB_SEAL_PACK_MIB", 8) << 20,
        packs: env_u64("MTXDB_SEAL_PACKS", 24),
        payload: env_u64("MTXDB_SEAL_PAYLOAD", 4096) as usize,
        delay: Duration::from_millis(env_u64("MTXDB_SEAL_DELAY_MS", 400)),
        pools: env_u64("MTXDB_SEAL_POOLS", 4) as usize,
    }
}

fn fresh_dir(label: &str) -> PathBuf {
    static RUN: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(
        || std::env::temp_dir().join("mtxdb-bench-seal-rotation"),
        PathBuf::from,
    );
    let dir = root.join(format!(
        "{label}-{}-{}",
        std::process::id(),
        RUN.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create bench dir");
    dir
}

fn record(seq: u64, payload: usize) -> Record {
    let mut hash = [0u8; 16];
    hash[..8].copy_from_slice(&seq.to_le_bytes());
    // Incompressible-ish so pack size tracks the payload.
    let mut data = vec![0u8; payload];
    let mut x = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for chunk in data.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let b = x.to_le_bytes();
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
    Record {
        collection_id: [1; 16],
        hash,
        data: Bytes::from(data),
        metadata: None,
    }
}

fn rss_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn ms(d: Duration) -> String {
    format!("{:.2}", d.as_secs_f64() * 1e3)
}

fn line(label: &str, v: &mut [Duration]) {
    v.sort_unstable();
    println!(
        "  {label:<10} n={:<7} p50={:>9} p95={:>9} p99={:>9} max={:>9} ms",
        v.len(),
        ms(pct(v, 0.50)),
        ms(pct(v, 0.95)),
        ms(pct(v, 0.99)),
        ms(v.last().copied().unwrap_or_default()),
    );
}

#[derive(Default)]
struct Peaks {
    queued: usize,
    running: usize,
    workers: usize,
    rss_kib: u64,
}

struct Writer {
    ordinary: Vec<Duration>,
    rotation: Vec<Duration>,
    bytes: u64,
    elapsed: Duration,
}

fn write_packs(pool: &ShardPool, c: &Cfg, seed: u64) -> Writer {
    let per_pack = c.pack_bytes / (c.payload as u64 + 64);
    let total = per_pack * c.packs;
    let mut w = Writer {
        ordinary: Vec::with_capacity(total as usize),
        rotation: Vec::new(),
        bytes: 0,
        elapsed: Duration::ZERO,
    };
    let mut prev_slot = None;
    let start = Instant::now();
    for i in 0..total {
        let rec = record(seed.wrapping_mul(1 << 32).wrapping_add(i), c.payload);
        let t = Instant::now();
        let (slot, _) = pool.put_record(&rec).expect("put");
        let took = t.elapsed();
        // A put that lands in a different pack than the last one rotated.
        if prev_slot.is_some_and(|p| p != slot) {
            w.rotation.push(took);
        } else {
            w.ordinary.push(took);
        }
        prev_slot = Some(slot);
        w.bytes += rec.data.len() as u64;
    }
    w.elapsed = start.elapsed();
    w
}

/// Run `pools` writers at once with a backlog/RSS sampler; print the report.
fn run(name: &str, c: &Cfg, pools: usize, delay: Duration) {
    println!(
        "\n== {name}: pools={pools} pack={} MiB packs/pool={} delay={} ms",
        c.pack_bytes >> 20,
        c.packs,
        delay.as_millis()
    );
    let opened: Vec<Arc<ShardPool>> = (0..pools)
        .map(|i| {
            let dir = fresh_dir(&format!("{name}-p{i}"));
            let p = ShardPool::open_with_max_shard_bytes(dir, c.pack_bytes).expect("open");
            p.set_seal_build_delay(delay);
            Arc::new(p)
        })
        .collect();

    let peaks = Arc::new(Mutex::new(Peaks::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (pools, peaks, stop) = (opened.clone(), peaks.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let (mut q, mut r, mut wk) = (0, 0, 0);
                for p in &pools {
                    let b = p.seal_backlog();
                    q += b.queued;
                    r += b.running;
                    wk += b.workers;
                }
                let mut pk = peaks.lock().unwrap();
                pk.queued = pk.queued.max(q);
                pk.running = pk.running.max(r);
                pk.workers = pk.workers.max(wk);
                pk.rss_kib = pk.rss_kib.max(rss_kib());
                drop(pk);
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };

    let cfg_ref = Arc::new(*c);
    let handles: Vec<_> = opened
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (p, c) = (p.clone(), cfg_ref.clone());
            std::thread::spawn(move || write_packs(&p, &c, i as u64 + 1))
        })
        .collect();
    let mut ordinary = Vec::new();
    let mut rotation = Vec::new();
    let (mut bytes, mut write_secs) = (0u64, 0f64);
    for h in handles {
        let w = h.join().unwrap();
        ordinary.extend(w.ordinary);
        rotation.extend(w.rotation);
        bytes += w.bytes;
        write_secs = write_secs.max(w.elapsed.as_secs_f64());
    }
    let backlog_at_end: usize = opened
        .iter()
        .map(|p| {
            let b = p.seal_backlog();
            b.queued + b.running
        })
        .sum();

    let drain = Instant::now();
    for p in &opened {
        p.wait_for_seals();
    }
    let drain = drain.elapsed();
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();

    let mut sidecars = 0usize;
    let mut packs = 0usize;
    for p in &opened {
        for s in 0..p.shard_count() as u16 {
            if let Some(sh) = p.get_shard(s) {
                packs += 1;
                sidecars += usize::from(mtxdb::index::packidx::pack_index_path(&sh.path).exists());
            }
        }
    }

    line("put", &mut ordinary);
    line("rotation", &mut rotation);
    let pk = peaks.lock().unwrap();
    println!(
        "  write: {:.1} MiB/s during writes ({:.2}s); drain after last put {:.2}s",
        bytes as f64 / (1 << 20) as f64 / write_secs,
        write_secs,
        drain.as_secs_f64()
    );
    println!(
        "  backlog: peak queued={} running={} workers={}; pending at end of writes={}",
        pk.queued, pk.running, pk.workers, backlog_at_end
    );
    println!(
        "  resources: peak RSS={:.0} MiB; packs={} sidecars={} (sealed packs are all indexed once drained; the active pack never is)",
        pk.rss_kib as f64 / 1024.0,
        packs,
        sidecars
    );
}

/// Unindexed-pack cost: reopen with sidecars, then without.
fn reopen(c: &Cfg) {
    println!(
        "\n== reopen: pack={} MiB packs={} (real builds)",
        c.pack_bytes >> 20,
        c.packs
    );
    let dir = fresh_dir("reopen");
    {
        let p = ShardPool::open_with_max_shard_bytes(dir.clone(), c.pack_bytes).unwrap();
        write_packs(&p, c, 1);
        p.wait_for_seals();
        p.sync_all().unwrap();
    }
    let time_open = |label: &str| {
        let t = Instant::now();
        let p = ShardPool::open_with_max_shard_bytes(dir.clone(), c.pack_bytes).unwrap();
        let open = t.elapsed();
        let scans = p.open_timings().map_or(0, |t| t.packfile_recovery_calls);
        println!(
            "  {label:<22} open={:>9} ms scanned_packs={scans}",
            ms(open)
        );
        p.wait_for_seals();
    };
    time_open("with sidecars");
    let mut removed = 0;
    for e in fs::read_dir(&dir).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "idx") {
            fs::remove_file(e.path()).unwrap();
            removed += 1;
        }
    }
    println!("  removed {removed} sidecars");
    time_open("sidecars missing");
    time_open("rebuilt (again)");
}

fn main() {
    let c = cfg();
    let want = std::env::var("MTXDB_SEAL_SCENARIOS")
        .unwrap_or_else(|_| "keepup,busy,sustained,multi,reopen".to_owned());
    let on = |s: &str| want.split(',').any(|w| w.trim() == s);
    println!(
        "seal_rotation: workers<=4 payload={}B build delay={} ms",
        c.payload,
        c.delay.as_millis()
    );
    if on("keepup") {
        run("keepup", &c, 1, Duration::ZERO);
    }
    if on("busy") {
        // Few packs: just past the worker count, so the next rotation meets
        // every worker busy.
        let busy = Cfg { packs: 8, ..c };
        run("busy", &busy, 1, c.delay);
    }
    if on("sustained") {
        run("sustained", &c, 1, c.delay);
    }
    if on("multi") {
        run("multi", &c, c.pools, c.delay);
    }
    if on("reopen") {
        reopen(&c);
    }
}
