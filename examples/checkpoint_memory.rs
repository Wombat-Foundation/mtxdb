//! Measure the slots-only checkpoint prototype.
//!
//! Reports, separately: full-checkpoint bytes with and without identity
//! tables, delta-epoch bytes, reopened index heap / mapped slot bytes /
//! process RSS, put-path cost and identity-read counts, and the peak RSS of a
//! checkpoint-less full-scan open.
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
fn index_memory(store: &PackfileStorage) -> (usize, usize, usize) {
    let mut heap = 0usize;
    let mut mapped = 0usize;
    let mut logical = 0usize;
    for (collection_id, _entries, memory, _capacity) in store.collection_summaries() {
        logical = logical.saturating_add(memory);
        if let Some((_entries, resident, mapped_bytes, _capacity)) =
            store.collection_index_residency(&collection_id)
        {
            heap = heap.saturating_add(resident);
            mapped = mapped.saturating_add(mapped_bytes);
        }
    }
    (heap, mapped, logical)
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn delta_bytes(dir: &Path) -> u64 {
    let prefix = format!("{INDEX_DELTA_FILE}.");
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
                .map(|entry| entry.metadata().map(|m| m.len()).unwrap_or(0))
                .sum()
        })
        .unwrap_or(0)
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

fn put_strided(store: &PackfileStorage, total_records: u64, count: u64) -> Duration {
    assert!(
        total_records > 0,
        "strided writes need a non-empty key range"
    );
    let started = Instant::now();
    let stride = spread_stride(total_records);
    let mut done = 0u64;
    while done < count {
        let n = (count - done).min(BATCH);
        let entries: Vec<([u8; 16], NodeData)> = (done..done + n)
            .map(|i| {
                let seq = i.wrapping_mul(stride) % total_records;
                (
                    id_for(seq),
                    NodeData::from_slice(&payload(seq.wrapping_add(1))),
                )
            })
            .collect();
        store.put_many(&COLLECTION, &entries).unwrap();
        done += n;
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
        let slots_only = match mode.as_str() {
            "identities" => false,
            "slots" => true,
            other => panic!("MTXDB_MEASURE_MODE must be identities|slots, got {other}"),
        };
        run_cold_rewrite(&dir, records, slots_only);
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

    // 1. Full checkpoint with identity tables.
    let started = Instant::now();
    store.force_index_checkpoint().unwrap();
    let identity_ms = started.elapsed().as_secs_f64() * 1e3;
    let identity_bytes = file_len(&checkpoint);
    let loaded = mtxdb::index::checkpoint::read_checkpoint(&checkpoint).expect("checkpoint");
    let identity_has_tables = loaded.collections.iter().any(|c| c.has_homes_tails);
    drop(loaded);
    println!(
        "checkpoint with identities: {:.1} MiB ({:.2} B/rec), write {:.0} ms, has_homes_tails={identity_has_tables}",
        mib(identity_bytes),
        identity_bytes as f64 / records as f64,
        identity_ms
    );

    // 2. Reopen from the identity checkpoint and measure put cost there.
    drop(store);
    let rss_before = rss_bytes();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let (heap, mapped, logical) = index_memory(&store);
    println!(
        "reopen (identities): heap {:.1} MiB ({:.2} B/rec), mapped {:.1} MiB, logical {:.2} B/rec, rss +{:.1} MiB (before {:.1})",
        mib(heap as u64),
        heap as f64 / records as f64,
        mib(mapped as u64),
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
        "put cost (identities): append 50k {:.0} ms ({:.2} us/rec), rewrite 20k {:.0} ms ({:.2} us/rec), identity reads append/rewrite {append_identity}/{rewrite_identity}",
        append_ids.as_secs_f64() * 1e3,
        append_ids.as_secs_f64() * 1e6 / 50_000.0,
        rewrite_ids.as_secs_f64() * 1e3,
        rewrite_ids.as_secs_f64() * 1e6 / 20_000.0,
    );
    store.sync_all().unwrap();
    println!(
        "delta after 70k post-reopen writes (identities): {:.1} MiB",
        mib(delta_bytes(&dir))
    );
    drop(store);

    // 3. Full checkpoint without identity tables, from the same live state.
    let store = PackfileStorage::open(dir.clone()).unwrap();
    store.set_checkpoint_slots_only(true);
    let started = Instant::now();
    store.force_index_checkpoint().unwrap();
    let slots_ms = started.elapsed().as_secs_f64() * 1e3;
    let slots_bytes = file_len(&checkpoint);
    let loaded = mtxdb::index::checkpoint::read_checkpoint(&checkpoint).expect("checkpoint");
    let slots_has_tables = loaded.collections.iter().any(|c| c.has_homes_tails);
    drop(loaded);
    println!(
        "checkpoint slots-only:      {:.1} MiB ({:.2} B/rec), write {:.0} ms, has_homes_tails={slots_has_tables}",
        mib(slots_bytes),
        slots_bytes as f64 / live as f64,
        slots_ms
    );
    let (heap, _mapped, logical) = index_memory(&store);
    println!(
        "live writer index (unchanged by the toggle): heap {:.1} MiB ({:.2} B/rec), logical {:.2} B/rec",
        mib(heap as u64),
        heap as f64 / live as f64,
        logical as f64 / live as f64
    );
    drop(store);

    // 4. Reopen from the slots-only checkpoint.
    let rss_before = rss_bytes();
    let started = Instant::now();
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let open_ms = started.elapsed().as_secs_f64() * 1e3;
    let (heap, mapped, logical) = index_memory(&store);
    println!(
        "reopen (slots-only): {:.0} ms, heap {:.2} MiB ({:.2} B/rec), mapped {:.1} MiB, logical {:.2} B/rec, rss +{:.1} MiB (before {:.1})",
        open_ms,
        mib(heap as u64),
        heap as f64 / live as f64,
        mib(mapped as u64),
        logical as f64 / live as f64,
        mib(rss_bytes().saturating_sub(rss_before)),
        mib(rss_before)
    );
    let before = store.stats().index_identity_reads;
    let append_slots = put_range(&store, live, 50_000);
    live += 50_000;
    let append_identity = store.stats().index_identity_reads - before;
    let before = store.stats().index_identity_reads;
    let rewrite_slots = put_range(&store, 0, 20_000);
    let rewrite_identity = store.stats().index_identity_reads - before;
    println!(
        "put cost (slots-only): append 50k {:.0} ms ({:.2} us/rec), rewrite 20k {:.0} ms ({:.2} us/rec), identity reads append/rewrite {append_identity}/{rewrite_identity}",
        append_slots.as_secs_f64() * 1e3,
        append_slots.as_secs_f64() * 1e6 / 50_000.0,
        rewrite_slots.as_secs_f64() * 1e3,
        rewrite_slots.as_secs_f64() * 1e6 / 20_000.0,
    );
    store.sync_all().unwrap();
    println!(
        "delta after 70k post-reopen writes (slots-only): {:.1} MiB",
        mib(delta_bytes(&dir))
    );
    drop(store);

    // 5. Full-scan open peak, after losing the persisted index.
    for entry in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(INDEX_CHECKPOINT_FILE) || name.starts_with(INDEX_DELTA_FILE) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    let baseline = rss_bytes();
    let (store, peak) = peak_rss_during(|| PackfileStorage::open(dir.clone()).unwrap());
    let (heap, _mapped, _logical) = index_memory(&store);
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
fn run_cold_rewrite(dir: &Path, records: u64, slots_only: bool) {
    let label = if slots_only {
        "slots-only"
    } else {
        "identities"
    };
    let store = PackfileStorage::open(dir.to_path_buf()).unwrap();
    let build = put_range(&store, 0, records);
    store.sync_all().unwrap();
    store.set_checkpoint_slots_only(slots_only);
    store.force_index_checkpoint().unwrap();
    let checkpoint = file_len(&dir.join(INDEX_CHECKPOINT_FILE));
    let loaded = mtxdb::index::checkpoint::read_checkpoint(&dir.join(INDEX_CHECKPOINT_FILE))
        .expect("checkpoint");
    let has_tables = loaded.collections.iter().any(|c| c.has_homes_tails);
    drop(loaded);
    println!(
        "[{label}] build {:.1}s, checkpoint {:.1} MiB ({:.2} B/rec), has_homes_tails={has_tables}",
        build.as_secs_f64(),
        mib(checkpoint),
        checkpoint as f64 / records as f64,
    );
    drop(store);

    let reopened = PackfileStorage::open(dir.to_path_buf()).unwrap();
    let (heap, mapped, _logical) = index_memory(&reopened);
    let evicted = if std::env::var("MTXDB_MEASURE_EVICT").as_deref() == Ok("1") {
        evict_dir(dir)
    } else {
        "not requested".to_owned()
    };
    println!(
        "[{label}] reopen heap {:.2} MiB, mapped {:.1} MiB; eviction: {evicted}",
        mib(heap as u64),
        mib(mapped as u64),
    );

    let before = reopened.stats().index_identity_reads;
    let first = put_strided(&reopened, records, 20_000);
    let first_reads = reopened.stats().index_identity_reads - before;
    let before = reopened.stats().index_identity_reads;
    let second = put_strided(&reopened, records, 20_000);
    let second_reads = reopened.stats().index_identity_reads - before;
    println!(
        "[{label}] strided rewrite 20k across {records} keys: first {:.1} ms ({:.3} us/rec) identity_reads={first_reads}; second {:.1} ms ({:.3} us/rec) identity_reads={second_reads}",
        first.as_secs_f64() * 1e3,
        first.as_secs_f64() * 1e6 / 20_000.0,
        second.as_secs_f64() * 1e3,
        second.as_secs_f64() * 1e6 / 20_000.0,
    );

    drop(reopened);
    let _ = std::fs::remove_dir_all(dir);
}
