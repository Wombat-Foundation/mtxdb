//! Measure the slots-only checkpoint prototype.
//!
//! Reports: checkpoint bytes (slots-only v1), delta-epoch bytes, reopened
//! index heap / mapped slot bytes / process RSS, put-path cost and
//! identity-read counts, and the peak RSS of a checkpoint-less full-scan
//! open. Set `MTXDB_MEASURE_EVICT=1` to evict the page cache before the cold
//! rewrite. The cold rewrite uses a deterministic spread sample; set
//! `MTXDB_MEASURE_SAMPLE=first` or `stride` to compare access patterns.
//!
//! ```text
//! MTXDB_MEASURE_RECORDS=4000000 cargo run --release --example checkpoint_memory
//! ```

// A measurement harness: byte counts and ratios are printed, never used for
// control flow, and the phases are intentionally linear.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE;
use mtxdb::index::delta::INDEX_DELTA_FILE;
use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::PackfileStorage;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const COLLECTION: [u8; 16] = [0xC1; 16];
const BATCH: u64 = 8192;

fn rss_bytes() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    pages.saturating_mul(4096)
}

fn id_for(seq: u64) -> [u8; 16] {
    let mut state = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&next().to_le_bytes());
    id[8..].copy_from_slice(&next().to_le_bytes());
    id
}

/// A 16-byte payload, incompressible enough that shard size tracks records.
fn payload(seq: u64) -> [u8; 16] {
    let mut state = seq.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    out
}

/// Sum the three index footprints over every collection.
///
/// `logical` is `memory_usage` (the operator-facing capacity estimate);
/// `heap` and `mapped` come from `collection_index_residency`, which separates
/// private heap from the checkpoint-mapped slot array.
fn index_memory(store: &PackfileStorage) -> (usize, usize, usize, usize) {
    let mut heap = 0usize;
    let mut mapped = 0usize;
    let mut logical = 0usize;
    let mut slots = 0usize;
    for (collection_id, _entries, memory, _capacity) in store.collection_summaries() {
        logical = logical.saturating_add(memory);
        if let Some((_entries, resident, mapped_bytes, capacity)) =
            store.collection_index_residency(&collection_id)
        {
            heap = heap.saturating_add(resident);
            mapped = mapped.saturating_add(mapped_bytes);
            slots = slots.saturating_add(capacity as usize);
        }
    }
    (heap, mapped, logical, slots)
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

fn delta_bytes(dir: &Path) -> u64 {
    let prefix = format!("{INDEX_DELTA_FILE}.");
    std::fs::read_dir(dir)
        .map_or(0, |entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
                .map(|entry| entry.metadata().map_or(0, |m| m.len()))
                .sum()
        })
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn put_range(store: &PackfileStorage, start: u64, count: u64) -> Duration {
    let started = Instant::now();
    let end = start.saturating_add(count);
    let mut seq = start;
    while seq < end {
        let n = (end - seq).min(BATCH);
        let entries: Vec<([u8; 16], NodeData)> = (seq..seq + n)
            .map(|i| (id_for(i), NodeData::from_slice(&payload(i))))
            .collect();
        store.put_many(&COLLECTION, &entries).unwrap();
        seq += n;
    }
    started.elapsed()
}

/// Pick a stride coprime to the record count so samples cover the full range.
fn spread_stride(total_records: u64) -> u64 {
    if total_records <= 1 {
        return 1;
    }
    let mut stride = 199_999 % total_records;
    if stride == 0 {
        stride = 1;
    }
    while gcd(stride, total_records) != 1 {
        stride = (stride + 1) % total_records;
        if stride == 0 {
            stride = 1;
        }
    }
    stride
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

/// Deterministic, non-contiguous sample of `count` distinct keys below `records`.
///
/// Scattering the rewritten keys is the point: the first `count` ids sit in
/// adjacent frames, so a page-cache drop only faults in a handful of pack
/// pages and readahead hides the rest. A fixed stride is cheaper, but can
/// alias shard and page boundaries; this sample avoids that.
fn spread_sample(records: u64, count: u64) -> Vec<u64> {
    assert!(records > 0, "spread sample needs a non-empty key range");
    assert!(
        count <= records,
        "spread sample count {count} exceeds record count {records}"
    );
    let mut seen = std::collections::HashSet::new();
    let count = usize::try_from(count).expect("sample count exceeds address space");
    let mut out = Vec::with_capacity(count);
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    while out.len() < count {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let index = z % records;
        if seen.insert(index) {
            out.push(index);
        }
    }
    out
}

/// Overwrite an explicit set of existing keys.
fn put_indices(store: &PackfileStorage, indices: &[u64]) -> Duration {
    let started = Instant::now();
    let batch = usize::try_from(BATCH).expect("batch size exceeds address space");
    for chunk in indices.chunks(batch) {
        let entries: Vec<([u8; 16], NodeData)> = chunk
            .iter()
            .map(|index| {
                (
                    id_for(*index),
                    NodeData::from_slice(&payload(index.wrapping_add(1))),
                )
            })
            .collect();
        store.put_many(&COLLECTION, &entries).unwrap();
    }
    started.elapsed()
}

fn peak_rss_during<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let running = Arc::new(AtomicBool::new(true));
    let peak = Arc::new(AtomicU64::new(rss_bytes()));
    let flag = Arc::clone(&running);
    let seen = Arc::clone(&peak);
    let sampler = std::thread::spawn(move || {
        while flag.load(Ordering::Relaxed) {
            seen.fetch_max(rss_bytes(), Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(2));
        }
    });
    let out = f();
    running.store(false, Ordering::Relaxed);
    let _ = sampler.join();
    (out, peak.load(Ordering::Relaxed))
}

fn main() {
    let records: u64 = std::env::var("MTXDB_MEASURE_RECORDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000_000);
    let dir: PathBuf = match std::env::var("MTXDB_MEASURE_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(_) => std::env::temp_dir().join(format!("mtxdb_measure_{}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let checkpoint = dir.join(INDEX_CHECKPOINT_FILE);

    if let Ok(mode) = std::env::var("MTXDB_MEASURE_MODE") {
        assert_eq!(mode, "slots", "MTXDB_MEASURE_MODE must be slots");
        run_cold_rewrite(&dir, records);
        return;
    }

    println!("records={records} dir={}", dir.display());
    println!("rss at start: {:.1} MiB", mib(rss_bytes()));

    let store = PackfileStorage::open(dir.clone()).unwrap();
    let build = put_range(&store, 0, records);
    store.sync_all().unwrap();
    let mut live = records;
    let built = records as f64 / build.as_secs_f64();
    println!(
        "build: {:.2}s ({:.0} records/s), delta epoch bytes: {:.1} MiB",
        build.as_secs_f64(),
        built,
        mib(delta_bytes(&dir))
    );

    // Checkpointing always writes only the packed locator slots.
    let started = Instant::now();
    store.force_index_checkpoint().unwrap();
    let checkpoint_ms = started.elapsed().as_secs_f64() * 1e3;
    let checkpoint_bytes = file_len(&checkpoint);
    let loaded = mtxdb::index::checkpoint::read_checkpoint(&checkpoint).expect("checkpoint");
    drop(loaded);
    println!(
        "slots-only checkpoint: {:.1} MiB ({:.2} B/rec), write {:.0} ms",
        mib(checkpoint_bytes),
        checkpoint_bytes as f64 / records as f64,
        checkpoint_ms
    );

    // Reopen the slots-only checkpoint and measure write costs.
    drop(store);
    let rss_before = rss_bytes();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let (heap, mapped, logical, slots) = index_memory(&store);
    println!(
        "reopen (slots-only): heap {:.1} MiB ({:.2} B/rec), mapped {:.1} MiB ({:.2} B/slot), logical {:.2} B/rec, rss +{:.1} MiB (before {:.1})",
        mib(heap as u64),
        heap as f64 / records as f64,
        mib(mapped as u64),
        mapped as f64 / slots.max(1) as f64,
        logical as f64 / records as f64,
        mib(rss_bytes().saturating_sub(rss_before)),
        mib(rss_before)
    );
    let before = store.stats().index_identity_reads;
    let append_ids = put_range(&store, live, 50_000);
    live += 50_000;
    let append_identity = store.stats().index_identity_reads - before;
    let before = store.stats().index_identity_reads;
    let rewrite_ids = put_range(&store, 0, 20_000);
    let rewrite_identity = store.stats().index_identity_reads - before;
    println!(
        "put cost: append 50k {:.0} ms ({:.2} us/rec), rewrite 20k {:.0} ms ({:.2} us/rec), identity reads append/rewrite {append_identity}/{rewrite_identity}",
        append_ids.as_secs_f64() * 1e3,
        append_ids.as_secs_f64() * 1e6 / 50_000.0,
        rewrite_ids.as_secs_f64() * 1e3,
        rewrite_ids.as_secs_f64() * 1e6 / 20_000.0,
    );
    store.sync_all().unwrap();
    println!(
        "delta after 70k post-reopen writes: {:.1} MiB",
        mib(delta_bytes(&dir))
    );
    drop(store);

    // Full-scan open peak, after losing the persisted index.
    for entry in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(INDEX_CHECKPOINT_FILE) || name.starts_with(INDEX_DELTA_FILE) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    let baseline = rss_bytes();
    let (store, peak) = peak_rss_during(|| PackfileStorage::open(dir.clone()).unwrap());
    let (heap, _mapped, _logical, _slots) = index_memory(&store);
    println!(
        "full-scan open: peak rss {:.1} MiB (from {:.1} MiB, +{:.1}), final heap {:.1} MiB ({:.2} B/rec)",
        mib(peak),
        mib(baseline),
        mib(peak.saturating_sub(baseline)),
        mib(heap as u64),
        heap as f64 / live as f64
    );
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Best-effort page-cache eviction via `vmtouch -e`, reporting what it did.
fn evict_dir(dir: &Path) -> String {
    match std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .output()
    {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            text.lines()
                .rev()
                .find(|line| line.contains("Evicted"))
                .unwrap_or("evicted")
                .trim()
                .to_owned()
        }
        Ok(out) => format!(
            "vmtouch failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        Err(error) => format!("vmtouch unavailable: {error}"),
    }
}

/// One configuration per process: build, write that mode's checkpoint,
/// reopen, optionally evict the page cache, then rewrite existing keys twice.
fn run_cold_rewrite(dir: &Path, records: u64) {
    let label = "slots-only";
    let store = PackfileStorage::open(dir.to_path_buf()).unwrap();
    let build = put_range(&store, 0, records);
    store.sync_all().unwrap();
    store.force_index_checkpoint().unwrap();
    let checkpoint = file_len(&dir.join(INDEX_CHECKPOINT_FILE));
    let loaded = mtxdb::index::checkpoint::read_checkpoint(&dir.join(INDEX_CHECKPOINT_FILE))
        .expect("checkpoint");
    drop(loaded);
    println!(
        "[{label}] build {:.1}s, checkpoint {:.1} MiB ({:.2} B/rec)",
        build.as_secs_f64(),
        mib(checkpoint),
        checkpoint as f64 / records as f64,
    );
    drop(store);

    let reopened = PackfileStorage::open(dir.to_path_buf()).unwrap();
    let (heap, mapped, _logical, slots) = index_memory(&reopened);
    let evicted = if std::env::var("MTXDB_MEASURE_EVICT").as_deref() == Ok("1") {
        evict_dir(dir)
    } else {
        "not requested".to_owned()
    };
    println!(
        "[{label}] reopen heap {heap} B ({:.6} B/rec), mapped {:.1} MiB ({:.2} B/slot); eviction: {evicted}",
        heap as f64 / records as f64,
        mib(mapped as u64),
        mapped as f64 / slots.max(1) as f64,
    );

    let sample_size = 20_000u64.min(records);
    let sample_mode = std::env::var("MTXDB_MEASURE_SAMPLE").unwrap_or_default();
    let (first_keys, sample_label) = match sample_mode.as_str() {
        "first" => (
            (0..sample_size).map(|i| i % records).collect::<Vec<u64>>(),
            "first-20k",
        ),
        "stride" => (
            (0..sample_size)
                .map(|i| i.wrapping_mul(spread_stride(records)) % records)
                .collect::<Vec<u64>>(),
            "stride",
        ),
        _ => (spread_sample(records, sample_size), "spread"),
    };
    let before = reopened.stats().index_identity_reads;
    let first = put_indices(&reopened, &first_keys);
    let first_reads = reopened.stats().index_identity_reads - before;
    let before = reopened.stats().index_identity_reads;
    let second = put_indices(&reopened, &first_keys);
    let second_reads = reopened.stats().index_identity_reads - before;
    println!(
        "[{label}] rewrite 20k ({sample_label} sample) across {records} keys: first {:.1} ms ({:.3} us/rec) identity_reads={first_reads}; second {:.1} ms ({:.3} us/rec) identity_reads={second_reads}",
        first.as_secs_f64() * 1e3,
        first.as_secs_f64() * 1e6 / sample_size as f64,
        second.as_secs_f64() * 1e3,
        second.as_secs_f64() * 1e6 / sample_size as f64,
    );

    drop(reopened);
    let _ = std::fs::remove_dir_all(dir);
}
