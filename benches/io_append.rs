//! Append-path A/B: eager vs buffered pack writes, and per-record vs batched
//! WAL publication.
//!
//! The engine has two orthogonal write-path choices that this bench measures
//! separately:
//!
//! - **pack append policy** — `eager` writes one frame per put, `buffered`
//!   accumulates a shard's frames and writes them with one positioned write
//!   per ~1 MiB flush.
//! - **publication grouping** — a `put` loop publishes one journal group per
//!   record, while one `put_many` call publishes the whole batch as a single
//!   group (one `writev` frame on Unix, so the payload is not copied into a
//!   second contiguous buffer).
//!
//! Every run writes the same records/payload into one collection, ends in one
//! `sync_all`, and reports the ingest time, the bytes, the journal group count
//! and WAL bytes observed before the closing sync, and the sync-phase split.
//!
//! # Run
//!
//! ```text
//! cargo bench --manifest-path benches/Cargo.toml --bench io_append
//!
//! # real disk instead of the default scratch root
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench io_append
//! ```
//!
//! # Env
//!
//! - `MTXDB_IO_RECORDS` (default 100000)
//! - `MTXDB_IO_PAYLOAD` (default 256) — payload bytes per record
//! - `MTXDB_IO_BATCH` (default 1000) — records per `put_many` call / WAL group
//! - `MTXDB_IO_JOURNAL=0` — omit the shared-WAL path entirely, isolating pack
//!   append cost
//! - `MTXDB_BENCH_COMPRESS=0` — disable per-record zstd (default on)
//! - `MTXDB_PREALLOC=1` — enable Linux `fallocate` block reservation (off by
//!   default), so a run can be repeated with and without it
//! - `MTXDB_BENCH_ROOT` — scratch root (defaults to the system temp dir)
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::journal::Journal;
use mtxdb::packfile::storage::SyncTimings;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::shard::AppendPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xA7; 16];

/// Well-mixed 64-bit permutation, so synthetic IDs spread across index buckets
/// like real content-address hashes.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn node_id(index: usize) -> NodeId {
    let a = splitmix64(index as u64);
    let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&a.to_le_bytes());
    id[8..].copy_from_slice(&b.to_le_bytes());
    id
}

/// Deterministic, poorly-compressible payload stream.
fn payload(seed: u64, len: usize) -> Bytes {
    let mut state = seed | 1;
    let mut bytes = vec![0u8; len];
    for chunk in bytes.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let word = state.to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    Bytes::from(bytes)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn scratch_root() -> PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(
        || std::env::temp_dir().join("mtxdb-bench-io-append"),
        PathBuf::from,
    )
}

fn fresh_dir() -> PathBuf {
    static RUN: AtomicU64 = AtomicU64::new(0);
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    let root = ROOT.get_or_init(|| {
        let root = scratch_root();
        fs::create_dir_all(&root).expect("create benchmark root");
        root
    });
    loop {
        let token = RUN.fetch_add(1, Ordering::Relaxed);
        let candidate = root.join(format!("mtxdb_io_append_{}_{token}", std::process::id()));
        match fs::create_dir(&candidate) {
            Ok(()) => return candidate,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("create benchmark dir {}: {error}", candidate.display()),
        }
    }
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

fn fmt_duration(duration: Duration) -> String {
    format!("{:.2}", duration.as_secs_f64() * 1e3)
}

#[derive(Clone, Copy)]
struct Config {
    records: usize,
    payload_len: usize,
    batch: usize,
    journal: bool,
    compress: bool,
    buffered: bool,
    put_many: bool,
}

struct Row {
    name: &'static str,
    ingest: Duration,
    sync: SyncTimings,
    bytes: u64,
    journal_groups: Option<u64>,
    journal_bytes: Option<u64>,
}

fn run_once(config: Config, name: &'static str) -> Row {
    let dir = fresh_dir();
    let wal = dir.join("wal.bin");
    let store = PackfileStorage::open_with_cache_and_policies(
        dir.clone(),
        10_000,
        config.compress,
        ChecksumPolicy::Full,
    )
    .expect("open store");
    let store = if config.buffered {
        store.with_append_policy(AppendPolicy::buffered())
    } else {
        store
    };
    if config.journal {
        store.enable_journal(&wal).expect("enable journal");
    }

    let entries: Vec<(NodeId, NodeData)> = (0..config.records)
        .map(|index| {
            (
                node_id(index),
                NodeData::new(payload(0xDEAD_BEEF ^ index as u64, config.payload_len)),
            )
        })
        .collect();
    let bytes = (config.records as u64).saturating_mul(config.payload_len as u64);

    let started = Instant::now();
    if config.put_many {
        for chunk in entries.chunks(config.batch.max(1)) {
            store.put_many(&COLLECTION, chunk).expect("put_many");
        }
    } else {
        for (id, data) in &entries {
            store.put(&COLLECTION, id, data).expect("put");
        }
    }
    let ingest = started.elapsed();

    // Count groups before the closing sync may reclaim the segment.
    let (journal_groups, journal_bytes) = if config.journal {
        let scan = Journal::scan_read_only(&wal).expect("scan wal");
        let bytes = scan
            .groups
            .iter()
            .flat_map(|group| group.entries.iter())
            .map(|entry| entry.frame_len)
            .sum();
        (Some(scan.groups.len() as u64), Some(bytes))
    } else {
        (None, None)
    };

    store.sync_all().expect("sync_all");
    let sync = store.sync_timings().expect("sync timings");

    drop(store);
    fs::remove_dir_all(&dir).ok();

    Row {
        name,
        ingest,
        sync,
        bytes,
        journal_groups,
        journal_bytes,
    }
}

fn print_header(config: Config) {
    println!("mtxdb append-path A/B");
    println!(
        "  records:   {}  payload: {}  batch: {}",
        config.records, config.payload_len, config.batch
    );
    println!(
        "  journal:   {}   compression: {}",
        if config.journal { "on" } else { "off" },
        if config.compress { "on" } else { "off" }
    );
    println!(
        "  prealloc:  {}",
        match std::env::var("MTXDB_PREALLOC").as_deref() {
            Ok("1") => "on (MTXDB_PREALLOC=1; Linux fallocate)",
            _ => "off (default)",
        }
    );
    println!();
    println!(
        "{:<26} {:>10} {:>10} {:>9} {:>12} {:>9} {:>11} {:>9}",
        "scenario",
        "ingest ms",
        "MiB/s",
        "sync ms",
        "wal groups",
        "wal size",
        "pack flush",
        "wal fsync"
    );
}

fn print_row(row: &Row) {
    let seconds = row.ingest.as_secs_f64().max(f64::MIN_POSITIVE);
    let mib_s = (row.bytes as f64 / (1024.0 * 1024.0)) / seconds;
    let groups = row
        .journal_groups
        .map_or_else(|| "-".to_owned(), |groups| groups.to_string());
    let wal_size = row.journal_bytes.map_or_else(|| "-".to_owned(), fmt_bytes);
    println!(
        "{:<26} {:>10} {:>10.1} {:>9} {:>12} {:>9} {:>11} {:>9}",
        row.name,
        fmt_duration(row.ingest),
        mib_s,
        fmt_duration(row.sync.total),
        groups,
        wal_size,
        fmt_duration(row.sync.pack_flush),
        fmt_duration(row.sync.wal),
    );
}

fn main() {
    let config = Config {
        records: env_usize("MTXDB_IO_RECORDS", 100_000),
        payload_len: env_usize("MTXDB_IO_PAYLOAD", 256),
        batch: env_usize("MTXDB_IO_BATCH", 1_000),
        journal: std::env::var("MTXDB_IO_JOURNAL").as_deref() != Ok("0"),
        compress: std::env::var("MTXDB_BENCH_COMPRESS").as_deref() != Ok("0"),
        buffered: false,
        put_many: false,
    };

    print_header(config);
    let mut rows = Vec::new();
    for (name, buffered, put_many) in [
        ("eager / put", false, false),
        ("buffered / put", true, false),
        ("eager / put_many", false, true),
        ("buffered / put_many", true, true),
    ] {
        let config = Config {
            buffered,
            put_many,
            ..config
        };
        let row = run_once(config, name);
        print_row(&row);
        rows.push(row);
    }

    println!();
    let eager_put = &rows[0];
    let buffered_put = &rows[1];
    let eager_many = &rows[2];
    let buffered_many = &rows[3];
    println!(
        "buffered vs eager (per-record put): {:.2}x ingest speedup",
        eager_put.ingest.as_secs_f64() / buffered_put.ingest.as_secs_f64().max(f64::MIN_POSITIVE)
    );
    println!(
        "put_many vs put (eager):           {:.2}x ingest speedup, {} vs {} WAL groups",
        eager_put.ingest.as_secs_f64() / eager_many.ingest.as_secs_f64().max(f64::MIN_POSITIVE),
        eager_many
            .journal_groups
            .map_or_else(|| "-".to_owned(), |value| value.to_string()),
        eager_put
            .journal_groups
            .map_or_else(|| "-".to_owned(), |value| value.to_string()),
    );
    println!(
        "put_many vs put (buffered):        {:.2}x ingest speedup",
        buffered_put.ingest.as_secs_f64()
            / buffered_many.ingest.as_secs_f64().max(f64::MIN_POSITIVE)
    );
}
