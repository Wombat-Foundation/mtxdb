//! What it costs to build a room's reconciliation population from the store,
//! to decide whether persisted runs are worth building.
//!
//! For each room size the parent builds a store of N owners (distinct event
//! ids, each logged with its 24-byte payload), then runs two builds in a fresh
//! child process each, so peak RSS is not polluted by the other:
//!
//! - `rebuild`: resolve every short id to its key, hash the event id, sort;
//! - `snapshot`: `population_snapshot`, which reads the owner log.
//!
//! Each child prints cold build time (page cache stays warm unless you drop
//! it) and peak resident growth over its pre-build baseline. A last table
//! computes, from the layout alone, the bytes a fenced run would read for one
//! node at several depths against the resident population.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench population_build
//! ```
//!
//! Env knobs: `MTXDB_PB_OWNERS` (comma list, default `10000,100000,1000000`;
//! add `10000000` for the large case, about 8 GB of packs), `MTXDB_BENCH_ROOT`,
//! `MTXDB_PB_KEEP=1`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use mtxdb::database::Database;
use mtxdb::layout::ShardType;
use mtxdb::population::encode_owner_payload;
use mtxdb::short_id::{BatchEvent, ShortIdIndex};
use rezzy_recon::{ElementHash, EventIdFormat, SortedPopulation};

const SCOPE: [u8; 16] = [0x62; 16];
const BATCH: u32 = 1000;
const RESOLVE_CHUNK: u32 = 4096;

/// `bytes` in MiB (saturating far above any size measured here).
fn mib(bytes: u64) -> f64 {
    f64::from(u32::try_from(bytes / 1024).unwrap_or(u32::MAX)) / 1024.0
}

fn index() -> ShortIdIndex {
    ShortIdIndex::new(ShardType::Edges, SCOPE)
}

/// A spread-out 32-byte digest for event `i`.
fn digest(i: u32) -> [u8; 32] {
    let mut out = [0_u8; 32];
    let mut x = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for chunk in out.chunks_exact_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_be_bytes());
    }
    out
}

fn event_id(i: u32) -> Vec<u8> {
    format!("${}", URL_SAFE_NO_PAD.encode(digest(i))).into_bytes()
}

fn root_dir() -> PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from)
}

fn status_kib(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix(field)?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(0)
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .flatten()
            .filter_map(|entry| entry.metadata().ok())
            .map(|meta| meta.len())
            .fold(0, u64::saturating_add)
    })
}

fn build_store(dir: &Path, owners: u32) {
    let db = Database::open(dir.to_path_buf()).expect("open");
    let started = Instant::now();
    let mut next = 0;
    while next < owners {
        let end = next.saturating_add(BATCH).min(owners);
        let keys: Vec<Vec<u8>> = (next..end).map(event_id).collect();
        let payloads: Vec<[u8; 24]> = (next..end)
            .map(|i| encode_owner_payload(ElementHash::from_digest32(digest(i))))
            .collect();
        let events: Vec<BatchEvent<'_>> = keys
            .iter()
            .zip(&payloads)
            .map(|(key, payload)| BatchEvent {
                owner: key,
                families: &[],
                owner_payload: Some(payload),
            })
            .collect();
        index().record_events(&db, &events).expect("record");
        next = end;
        // Syncing lets the journal reclaim; without it a large build fills
        // the segment and commits fail.
        if (next / BATCH) % 20 == 0 {
            db.edges().sync_all().expect("sync");
        }
    }
    db.edges().sync_all().expect("sync");
    println!(
        "  build store: {:.1}s, {:.1} MiB on disk",
        started.elapsed().as_secs_f64(),
        mib(dir_bytes(dir))
    );
}

/// Peak anonymous memory (`RssAnon`) while `work` runs, sampled every
/// millisecond. `VmHWM` also counts file-backed pages of the mapped packs, so
/// it overstates what the build allocates.
fn peak_anon_kib<T>(work: impl FnOnce() -> T) -> (T, u64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let stop = AtomicBool::new(false);
    let peak = AtomicU64::new(status_kib("RssAnon:"));
    let baseline = peak.load(Ordering::Relaxed);
    let value = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(status_kib("RssAnon:"), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
        let value = work();
        peak.fetch_max(status_kib("RssAnon:"), Ordering::Relaxed);
        stop.store(true, Ordering::Relaxed);
        value
    });
    (value, peak.load(Ordering::Relaxed).saturating_sub(baseline))
}

fn child(mode: &str, dir: &Path) {
    let db = Database::open(dir.to_path_buf()).expect("open");
    let started = Instant::now();
    let (len, anon_kib) = peak_anon_kib(|| build(mode, &db));
    println!(
        "CHILD {mode} len={len} ms={:.0} peak_anon_growth_mib={:.1} vmhwm_mib={:.0}",
        started.elapsed().as_secs_f64() * 1e3,
        mib(anon_kib.saturating_mul(1024)),
        mib(status_kib("VmHWM:").saturating_mul(1024))
    );
}

fn build(mode: &str, db: &Database) -> usize {
    match mode {
        "snapshot" => index().population_snapshot(db).expect("snapshot").len(),
        "rebuild" => {
            let next = index().counter(db).expect("counter");
            let mut hashes = Vec::new();
            let mut id = 1;
            while id < next {
                let end = id.saturating_add(RESOLVE_CHUNK).min(next);
                let ids: Vec<u32> = (id..end).collect();
                for key in index()
                    .resolve(db, &ids)
                    .expect("resolve")
                    .into_iter()
                    .flatten()
                {
                    let text = std::str::from_utf8(&key).expect("utf8 key");
                    hashes.push(
                        ElementHash::from_matrix_event_id(text, EventIdFormat::V4Plus)
                            .expect("hash"),
                    );
                }
                id = end;
            }
            SortedPopulation::new(hashes).len()
        }
        other => panic!("unknown mode {other}"),
    }
}

fn run_child(mode: &str, dir: &Path) {
    let output = Command::new(std::env::current_exe().expect("exe"))
        .env("MTXDB_PB_CHILD", mode)
        .env("MTXDB_PB_DIR", dir)
        .output()
        .expect("spawn child");
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(rest) = line.strip_prefix("CHILD ") {
            println!("  {rest}");
        }
    }
    if !output.status.success() {
        println!(
            "  {mode} FAILED: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn fence_table(owners: u64) {
    println!(
        "  resident SortedPopulation: {:.1} MiB; fenced node read (h64 column slice):",
        mib(owners.saturating_mul(24))
    );
    for depth in [0_u32, 4, 8, 12, 16] {
        let bytes = owners.saturating_mul(8) >> depth;
        println!("    depth {depth:>2}: {:.3} MiB", mib(bytes));
    }
}

fn main() {
    if let Ok(mode) = std::env::var("MTXDB_PB_CHILD") {
        let dir = PathBuf::from(std::env::var_os("MTXDB_PB_DIR").expect("dir"));
        child(&mode, &dir);
        return;
    }
    let sizes: Vec<u32> = std::env::var("MTXDB_PB_OWNERS")
        .unwrap_or_else(|_| "10000,100000,1000000".to_owned())
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    for owners in sizes {
        println!("owners = {owners}");
        let dir = root_dir().join(format!(
            "mtxdb-population-build-{owners}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        build_store(&dir, owners);
        run_child("rebuild", &dir);
        run_child("snapshot", &dir);
        fence_table(u64::from(owners));
        if std::env::var_os("MTXDB_PB_KEEP").is_some() {
            println!("  kept {}", dir.display());
        } else {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
