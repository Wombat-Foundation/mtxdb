//! Where the time of a pack recovery scan goes, on one pack, layer by layer.
//!
//! A writable open validates every frame of every pack (`recover_packfile`), and
//! that was about 80% of a reopen at 1.2 us per frame (about 49 MiB/s of cached
//! data). This times the same pack under each layer, so the cost can be assigned
//! before anything is changed:
//!
//! - `read only`: read the whole file into memory (the I/O floor).
//! - `crc only`: one CRC32 over those bytes (the most the checksum can cost).
//! - `header walk`: `scan_packfile_skip_payload`, frame headers only, no payload
//!   read and no CRC (parse cost plus its seeks).
//! - `old scan`: the scan as it was: an 8 KiB `BufReader`, two `stream_position`
//!   calls per frame, every record collected.
//! - `scan (collect)`: `scan_and_recover_packfile` now: a 1 MiB buffer and a
//!   byte counter instead of `stream_position`, every record collected.
//! - `recover (count)`: `recover_packfile`: the same validation and truncation,
//!   no per-record result.
//!
//! Every variant except the first three validates every frame's CRC. Warm cache
//! only (drop the page cache yourself for a cold run; needs root).
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench pack_scan_profile
//! ```
//!
//! Env knobs: `MTXDB_PS_RECORDS` (records to write, default 1,000,000),
//! `MTXDB_PS_PAYLOAD` (bytes, default 16), `MTXDB_PS_REPS` (default 5),
//! `MTXDB_BENCH_ROOT`.

use std::hint::black_box;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::packfile::{
    read_header, read_record_metadata, recover_packfile, scan_and_recover_packfile,
    scan_packfile_skip_payload,
};
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xE1; 16];

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn node_id(seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&seq.to_le_bytes());
    id
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// The scan as it was before the change: default 8 KiB buffer, the position
/// asked of the file twice per frame, every record collected.
fn old_scan(path: &Path) -> usize {
    use std::io::Seek;
    let mut reader = BufReader::new(std::fs::File::open(path).expect("open pack"));
    let mut entries = Vec::new();
    if read_header(&mut reader).expect("header").is_none() {
        return 0;
    }
    let mut last_valid = reader.stream_position().expect("position");
    loop {
        let offset = reader.stream_position().expect("position");
        match read_record_metadata(&mut reader) {
            Ok(Some(meta)) => {
                entries.push((meta.collection_id, meta.hash, offset));
                last_valid = reader.stream_position().expect("position");
            }
            _ => break,
        }
    }
    black_box(last_valid);
    entries.len()
}

/// Run `work` `reps` times; returns the fastest and the median.
fn time(reps: u64, mut work: impl FnMut()) -> (Duration, Duration) {
    let mut samples: Vec<Duration> = (0..reps)
        .map(|_| {
            let started = Instant::now();
            work();
            started.elapsed()
        })
        .collect();
    samples.sort();
    (
        samples.first().copied().unwrap_or_default(),
        samples.get(samples.len() / 2).copied().unwrap_or_default(),
    )
}

fn report(label: &str, timing: (Duration, Duration), frames: u64, pack_bytes: u64) {
    let (best, median) = timing;
    let mib = f64::from(u32::try_from(pack_bytes >> 20).unwrap_or(u32::MAX));
    let per_frame = best.as_secs_f64() * 1e6 / f64::from(u32::try_from(frames).unwrap_or(u32::MAX));
    println!(
        "  {label:<18} best {:>7.1} ms  median {:>7.1} ms  {:>7.0} MiB/s  {per_frame:.3} us/frame",
        millis(best),
        millis(median),
        mib / best.as_secs_f64().max(1e-9),
    );
}

fn pack_path(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("read store dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "pack"))
        .expect("the store has a pack")
}

fn main() {
    let records = env_u64("MTXDB_PS_RECORDS", 1_000_000);
    let payload = usize::try_from(env_u64("MTXDB_PS_PAYLOAD", 16)).expect("payload fits usize");
    let reps = env_u64("MTXDB_PS_REPS", 5).max(1);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");
    let dir = base.join(format!("mtxdb_bench_pack_scan_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let store = PackfileStorage::open(dir.clone()).expect("open store");
    let data = NodeData::from_slice(&vec![0x5a; payload]);
    for seq in 0..records {
        store
            .put(&COLLECTION, &node_id(seq), &data)
            .expect("put record");
    }
    store.sync_all().expect("sync_all");
    drop(store);

    let path = pack_path(&dir);
    let pack_bytes = std::fs::metadata(&path).expect("stat pack").len();
    println!(
        "pack: {records} records, {} MiB ({:.1} bytes/frame), {reps} reps, warm cache",
        pack_bytes >> 20,
        f64::from(u32::try_from(pack_bytes).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(records).unwrap_or(u32::MAX)),
    );

    report(
        "read only",
        time(reps, || {
            let buffer = std::fs::read(&path).expect("read pack");
            assert_eq!(
                u64::try_from(buffer.len()).unwrap_or(0),
                pack_bytes,
                "the whole pack was read"
            );
        }),
        records,
        pack_bytes,
    );
    let bytes = std::fs::read(&path).expect("read pack");
    report(
        "crc only",
        time(reps, || {
            black_box(crc32fast::hash(&bytes));
        }),
        records,
        pack_bytes,
    );
    report(
        "header walk",
        time(reps, || {
            assert_eq!(
                u64::try_from(scan_packfile_skip_payload(&path).expect("scan").len()).unwrap_or(0),
                records,
                "the header walk saw every record"
            );
        }),
        records,
        pack_bytes,
    );
    report(
        "old scan",
        time(reps, || {
            assert_eq!(
                u64::try_from(old_scan(&path)).unwrap_or(0),
                records,
                "the old scan saw every record"
            );
        }),
        records,
        pack_bytes,
    );
    report(
        "scan (collect)",
        time(reps, || {
            assert_eq!(
                u64::try_from(scan_and_recover_packfile(&path).expect("scan").len()).unwrap_or(0),
                records,
                "the collecting scan saw every record"
            );
        }),
        records,
        pack_bytes,
    );
    report(
        "recover (count)",
        time(reps, || {
            assert_eq!(
                recover_packfile(&path).expect("recover").records,
                records,
                "the counting recovery saw every record"
            );
        }),
        records,
        pack_bytes,
    );
    let _ = std::fs::remove_dir_all(&dir);
}
