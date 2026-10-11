//! Where the prototype's id -> key read time goes.
//!
//! The chunk-size sweep showed id -> key latency growing with the encoded
//! chunk size. This splits one read into layers, under each combination of the
//! pool's two per-byte policies (zstd compression on write, CRC32 verification
//! on read):
//!
//! - decode only: `key_at` on bytes already in memory (no storage);
//! - fetch only: `get` of the key-chunk record, nothing decoded;
//! - fetch + decode: the id -> key read the lookup bench times;
//! - floor: the baseline store's `resolve` of one id, same policy.
//!
//! Warm: single calls over a seeded sample, repeated `MTXDB_EL_REPS` times
//! (default 3) so run-to-run spread shows. Cold: per repetition the store's files
//! the database is reopened, its files evicted (`sync` + `vmtouch -e`; reopen
//! itself reads the store into the page cache), and the first
//! `MTXDB_EL_COLD` (default 500) calls timed. Per-call timing adds clock-read
//! overhead of a few tens of nanoseconds. Chunk size comes from
//! `MTXDB_EL_CHUNK`; each policy writes its own store, so sizes are reported
//! too. Layer costs are measured independently, not subtracted.
//!
//! ```text
//! MTXDB_EL_CHUNK=32 MTXDB_EL_TSV=room.tsv \
//!   MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_read_split
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use mtxdb::database::{Database, PoolPolicies, PoolPolicy};
use mtxdb::layout::ShardType;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::storage::StorageEngine;

#[path = "edge_layout_common.rs"]
mod common;

use common::{
    baseline_scope, chunk_size, format, len_f64, load, locate, mib, num, record_rows, sizes,
    spread, write_chunked, Row, Variant, CHUNK_COLLECTION, KIND_KEY_CHUNK,
};

const MICROS: f64 = 1_000_000.0;

/// One combination of the pool's per-byte policies.
struct Policy {
    label: &'static str,
    compress: bool,
    checksum: ChecksumPolicy,
}

const POLICIES: [Policy; 4] = [
    Policy {
        label: "no compression, no CRC",
        compress: false,
        checksum: ChecksumPolicy::Disabled,
    },
    Policy {
        label: "no compression, CRC on read",
        compress: false,
        checksum: ChecksumPolicy::Full,
    },
    Policy {
        label: "zstd, no CRC",
        compress: true,
        checksum: ChecksumPolicy::Disabled,
    },
    Policy {
        label: "zstd, CRC on read (default)",
        compress: true,
        checksum: ChecksumPolicy::Full,
    },
];

fn open(dir: &Path, policy: &Policy) -> Database {
    let pool = PoolPolicy {
        compress: policy.compress,
        checksum_policy: policy.checksum,
    };
    Database::open_with_policies(
        dir.to_path_buf(),
        PoolPolicies::default().with(ShardType::Edges, pool),
    )
    .expect("open")
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic sample of `n` row indexes.
fn sample(rows: usize, n: usize) -> Vec<usize> {
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let modulus = u64::try_from(rows).unwrap_or(1).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            usize::try_from(x.checked_rem(modulus).unwrap_or(0)).unwrap_or(0)
        })
        .collect()
}

/// Mean microseconds per call over `n` calls of `f`, and how many calls it
/// reported as wrong.
fn time_calls(n: usize, mut f: impl FnMut(usize) -> bool) -> (f64, usize) {
    let mut total = Duration::ZERO;
    let mut wrong = 0_usize;
    for i in 0..n {
        let started = Instant::now();
        let ok = f(i);
        total = total.saturating_add(started.elapsed());
        wrong = wrong.saturating_add(usize::from(!ok));
    }
    (total.as_secs_f64() * MICROS / len_f64(n.max(1)), wrong)
}

/// Repeat a layer `reps` times and print each repetition's mean.
fn layer(label: &str, reps: usize, n: usize, mut f: impl FnMut(usize) -> bool) {
    let mut means = Vec::new();
    let mut wrong = 0_usize;
    for _ in 0..reps {
        let (mean, bad) = time_calls(n, &mut f);
        means.push(format!("{mean:.2}"));
        wrong = wrong.saturating_add(bad);
    }
    println!(
        "    {label:<24} {} us/call | mismatches {wrong}",
        means.join(" ")
    );
}

fn evict(dir: &Path) {
    let _ = Command::new("sync").status();
    let _ = Command::new("vmtouch").arg("-e").arg(dir).output();
}

/// `vmtouch`'s resident-page line for `dir`, to show what is in the page cache.
fn residency(dir: &Path) -> String {
    Command::new("vmtouch")
        .arg(dir)
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .find(|line| line.contains("Resident Pages"))
                .map(|line| line.trim().to_owned())
        })
        .unwrap_or_else(|| "residency unknown".to_owned())
}

/// Cold layer: per repetition, reopen, then evict the store's files, and time
/// the first `n` calls of `f` against that handle. Reopen reads the whole store
/// back into the page cache, so eviction must come after it; the in-memory
/// index survives, so the calls measure pack reads. The residency line printed
/// after eviction shows what the page cache held.
fn cold_layer<T>(
    label: &str,
    (dir, policy): (&Path, &Policy),
    (reps, n): (usize, usize),
    f: impl Fn(&Database, usize) -> T,
    ok: impl Fn(usize, T) -> bool,
) {
    let mut means = Vec::new();
    let mut wrong = 0_usize;
    let mut resident = Vec::new();
    for _ in 0..reps {
        let db = open(dir, policy);
        evict(dir);
        resident.push(residency(dir));
        let mut total = Duration::ZERO;
        for i in 0..n {
            let started = Instant::now();
            let out = f(&db, i);
            total = total.saturating_add(started.elapsed());
            wrong = wrong.saturating_add(usize::from(!ok(i, out)));
        }
        means.push(format!(
            "{:.2}",
            total.as_secs_f64() * MICROS / len_f64(n.max(1))
        ));
    }
    println!(
        "    cold {label:<19} {} us/call | mismatches {wrong} | after reopen: {}",
        means.join(" "),
        resident.join(" ; ")
    );
}

fn fetch(db: &Database, chunk: usize) -> Option<bytes::Bytes> {
    let id = spread(KIND_KEY_CHUNK, u32::try_from(chunk).expect("chunk"));
    db.edges()
        .get(&CHUNK_COLLECTION, &id)
        .expect("get")
        .map(|data| data.bytes)
}

/// Prototype layers for one policy.
fn proto_layers(
    paths: (&Path, &Policy),
    rows: &[Row],
    picks: &[usize],
    (reps, cold_n): (usize, usize),
) {
    let (dir, policy) = paths;
    let db = open(dir, policy);
    let (ids, stats) = write_chunked(&db, rows);
    db.edges().force_index_checkpoint().expect("checkpoint");
    drop(db);
    let on_disk = sizes(dir);
    println!(
        "  prototype: {:.2} MiB on disk for {:.2} MiB of payload ({:.2}x)",
        mib(on_disk.total()),
        mib(stats.payload_bytes()),
        num(on_disk.total()) / num(stats.payload_bytes().max(1)),
    );
    let db = open(dir, policy);
    let n = picks.len();
    let located: Vec<(usize, usize)> = picks
        .iter()
        .map(|i| locate(ids.by_key[rows[*i].id.as_str()] as usize))
        .collect();
    let raw = fetch(&db, located[0].0).expect("chunk");
    println!(
        "    record read per lookup: {} bytes (decoded payload)",
        raw.len()
    );
    // Positions cycle through the chunk's first eight entries; the offset
    // table makes the position irrelevant to decode cost.
    layer("decode only", reps, n, |i| {
        format::key_at(&raw, i & 7).is_ok()
    });
    layer("fetch only", reps, n, |i| {
        fetch(&db, located[i].0).is_some()
    });
    layer("fetch + decode", reps, n, |i| {
        let (chunk, pos) = located[i];
        fetch(&db, chunk)
            .and_then(|record| format::key_at(&record, pos).ok().map(<[u8]>::to_vec))
            .is_some_and(|key| key == rows[picks[i]].id.as_bytes())
    });
    drop(db);
    proto_cold(paths, rows, picks, &located, (reps, cold_n.min(n)));
}

/// Cold prototype layers: fetch only, then fetch + decode.
fn proto_cold(
    (dir, policy): (&Path, &Policy),
    rows: &[Row],
    picks: &[usize],
    located: &[(usize, usize)],
    (reps, n): (usize, usize),
) {
    let paths = (dir, policy);
    cold_layer(
        "fetch only",
        paths,
        (reps, n),
        |db, i| fetch(db, located[i].0).is_some(),
        |_, found| found,
    );
    cold_layer(
        "fetch + decode",
        paths,
        (reps, n),
        |db, i| {
            let (chunk, pos) = located[i];
            fetch(db, chunk).and_then(|r| format::key_at(&r, pos).ok().map(<[u8]>::to_vec))
        },
        |i, key| key.is_some_and(|k| k == rows[picks[i]].id.as_bytes()),
    );
}

/// The baseline store's one-id `resolve`, as the floor for this policy.
fn baseline_floor(
    paths: (&Path, &Policy),
    rows: &[Row],
    picks: &[usize],
    (reps, cold_n): (usize, usize),
) {
    let (dir, policy) = paths;
    let all = Variant {
        label: "all",
        prev: true,
        auth: true,
        rel: true,
    };
    let db = open(dir, policy);
    record_rows(&db, rows, all);
    db.edges().force_index_checkpoint().expect("checkpoint");
    drop(db);
    let db = open(dir, policy);
    let scope = baseline_scope();
    let ids: Vec<u32> = picks
        .iter()
        .map(|i| {
            scope
                .lookup(&db, rows[*i].id.as_bytes())
                .expect("lookup")
                .expect("id")
        })
        .collect();
    layer("floor: baseline resolve", reps, picks.len(), |i| {
        scope
            .resolve(&db, &ids[i..=i])
            .expect("resolve")
            .into_iter()
            .next()
            .flatten()
            .is_some_and(|key| key == rows[picks[i]].id.as_bytes())
    });
    drop(db);
    cold_layer(
        "floor: resolve",
        paths,
        (reps, cold_n.min(picks.len())),
        |db, i| scope.resolve(db, &ids[i..=i]).expect("resolve"),
        |i, keys| {
            keys.into_iter()
                .next()
                .flatten()
                .is_some_and(|key| key == rows[picks[i]].id.as_bytes())
        },
    );
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let n = env_usize("MTXDB_EL_OPS", 20_000);
    let reps = env_usize("MTXDB_EL_REPS", 3);
    let cold_n = env_usize("MTXDB_EL_COLD", 500);
    let rows = load(&tsv, env_usize("MTXDB_EL_EVENTS", usize::MAX));
    let picks = sample(rows.len(), n);
    println!(
        "{} events, {n} sampled ops x {reps} reps, {} ids per chunk",
        rows.len(),
        chunk_size()
    );
    let pid = std::process::id();
    for (k, policy) in POLICIES.iter().enumerate() {
        println!("[{}]", policy.label);
        let proto = base.join(format!("mtxdb_bench_edgesplit_{pid}_{k}_proto"));
        let floor = base.join(format!("mtxdb_bench_edgesplit_{pid}_{k}_base"));
        for dir in [&proto, &floor] {
            let _ = std::fs::remove_dir_all(dir);
        }
        proto_layers((&proto, policy), &rows, &picks, (reps, cold_n));
        baseline_floor((&floor, policy), &rows, &picks, (reps, cold_n));
        for dir in [&proto, &floor] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
