//! Prototype bench: a segmented point index vs the resident `LossyIndex` image.
//!
//! Compares two ways to make `(logical_id -> locator)` durable for a single
//! collection:
//!
//! 1. **current**: the resident open-addressed `LossyIndex` plus its slots-only
//!    serialized image (what a per-collection checkpoint carries).
//! 2. **segmented**: a bounded mutable tail plus immutable sorted segments and
//!    an atomically published manifest (see `support/segmented_index.rs`).
//!
//! The workload is deliberately the simplest one the primary index actually
//! serves: 1M uniformly distributed 16-byte ids mapped to `u64` locators, with
//! no removals or overwrites. That is the *best* case for both designs. Warm
//! page cache, release build, one process.
//!
//! ```text
//! cargo bench --manifest-path benches/Cargo.toml --bench index_segments
//! ```
//!
//! Env knobs:
//! - `MTXDB_IS_ENTRIES` (default 1,000,000)
//! - `MTXDB_IS_LOOKUPS` (default 1,000,000)
//! - `MTXDB_IS_APPEND` (default 65,536)
//! - `MTXDB_IS_FSYNC` (default 1)
//! - `MTXDB_IS_KEEP` (default 0; keep the probe directories)
//! - `MTXDB_BENCH_ROOT` (default `/tmp/opencode/mtxdb-index-segments`)
//!
//! A flush sweep runs at 64K / 256K / 1M tail entries. The append comparison
//! runs at 64K.

#[path = "support/segmented_index.rs"]
mod segmented_index;

use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mtxdb::index::{IndexConfig, LossyIndex};

use segmented_index::{SegmentedIndex, SegmentedOptions};

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

fn key(index: u64) -> [u8; 16] {
    let mut x = index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    let mut out = [0_u8; 16];
    for chunk in out.chunks_mut(8) {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        chunk.copy_from_slice(&x.to_le_bytes());
    }
    out
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn us_per(duration: Duration, count: u64) -> f64 {
    duration.as_secs_f64() * 1e6 / count.max(1) as f64
}

fn ns_per(duration: Duration, count: u64) -> f64 {
    duration.as_secs_f64() * 1e9 / count.max(1) as f64
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn unique_dir(root: &Path, label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    root.join(format!("{label}-{}-{stamp}", std::process::id()))
}

fn write_all_sync(path: &Path, bytes: &[u8], fsync: bool) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    if fsync {
        file.sync_all()?;
    }
    Ok(())
}

fn lookup_workload(
    count: u64,
    hits: bool,
    lookup: impl Fn(&[u8; 16]) -> Option<u64>,
) -> (Duration, u64) {
    // Warm the index and the mapping before timing.
    for i in 0..(count / 16).max(1024) {
        let id = if hits {
            key(i % count.max(1))
        } else {
            key(count.saturating_add(i))
        };
        black_box(lookup(&id));
    }
    let mut found = 0_u64;
    let started = Instant::now();
    for i in 0..count {
        let id = if hits {
            key((i.wrapping_mul(2_654_435_761)) % count.max(1))
        } else {
            key(count.saturating_add(i))
        };
        if black_box(lookup(&id)).is_some() {
            found = found.saturating_add(1);
        }
    }
    (started.elapsed(), found)
}

struct CurrentResult {
    append_insert: Duration,
    append_image: Duration,
    append_bytes: u64,
}

fn bench_current(
    root: &Path,
    entries: u64,
    lookups: u64,
    append: u64,
    fsync: bool,
) -> CurrentResult {
    println!("\n=== current: resident LossyIndex + slots image ===");
    let capacity = usize::try_from(entries.saturating_mul(2)).unwrap_or(usize::MAX);

    let started = Instant::now();
    let index = LossyIndex::with_config(capacity, IndexConfig::default());
    for i in 0..entries {
        index
            .insert(&key(i), (i % 4096) as u16, i)
            .expect("insert into pre-sized index");
    }
    let insert = started.elapsed();
    println!(
        "insert            {entries:>10}  {:>9.1} ms  {:>7.1} ns/op",
        ms(insert),
        ns_per(insert, entries)
    );
    println!(
        "resident          {:>10.1} MiB  {:>7.1} B/record  (capacity {} slots)",
        mib(index.resident_bytes() as u64),
        index.resident_bytes() as f64 / entries as f64,
        index.capacity()
    );

    let started = Instant::now();
    let image = index.serialize();
    let serialize = started.elapsed();
    println!(
        "serialize         {:>10.1} MiB  {:>9.1} ms  {:>7.1} ns/record",
        mib(image.len() as u64),
        ms(serialize),
        ns_per(serialize, entries)
    );

    let image_path = root.join("current-image.bin");
    let started = Instant::now();
    write_all_sync(&image_path, &image, fsync).expect("write current image");
    let write = started.elapsed();
    println!(
        "write+fsync       {:>10.1} MiB  {:>9.1} ms  ({})",
        mib(image.len() as u64),
        ms(write),
        if fsync { "fsync" } else { "no fsync" }
    );
    let started = Instant::now();
    let reopened =
        LossyIndex::deserialize_with_config(&image, IndexConfig::default()).expect("deserialize");
    let reopen = started.elapsed();
    println!(
        "reopen            {:>10.1} MiB  {:>9.1} ms  -> {:>6.1} MiB heap",
        mib(image.len() as u64),
        ms(reopen),
        mib(reopened.resident_bytes() as u64)
    );

    let (hit_time, hits) = lookup_workload(lookups, true, |id| {
        reopened.lookup(id).map(|(_, offset)| offset)
    });
    let (miss_time, misses) = lookup_workload(lookups, false, |id| {
        reopened.lookup(id).map(|(_, offset)| offset)
    });
    println!(
        "lookup hit        {hits:>10}  {:>9.1} ns/op",
        ns_per(hit_time, lookups)
    );
    println!(
        "lookup miss       {misses:>10}  {:>9.1} ns/op",
        ns_per(miss_time, lookups)
    );
    assert_eq!(hits, lookups, "current index must find every hit");
    // The slots-only image has no `homes`/`tails`, so a rare tag collision can
    // surface as a candidate until the open hydrates identity from the packs.
    // Production pays a frame read + verify for each; report the count rather
    // than pretending the raw image is exact.
    println!(
        "miss candidates   {misses:>10}  ({:.1} ppm; slots-only image has no identities)",
        misses as f64 * 1e6 / lookups.max(1) as f64
    );
    drop(reopened);

    // Incremental append on the live, identity-hydrated table, then a fresh
    // whole image. Slots-only serialization cannot accept writes until its
    // identities are hydrated from the pack files.
    let started = Instant::now();
    for j in 0..append {
        index
            .insert(
                &key(entries.saturating_add(j)),
                ((entries + j) % 4096) as u16,
                entries + j,
            )
            .expect("append insert");
    }
    let append_insert = started.elapsed();
    let started = Instant::now();
    let image2 = index.serialize();
    let serialized = started.elapsed();
    let image2_path = root.join("current-image-append.bin");
    let started = Instant::now();
    write_all_sync(&image2_path, &image2, fsync).expect("write appended image");
    let written = started.elapsed();
    println!(
        "append {append:>7} + image   insert {:>7.1} ms, serialize {:>7.1} ms, write {:>7.1} ms, {:>7.1} MiB rewritten",
        ms(append_insert),
        ms(serialized),
        ms(written),
        mib(image2.len() as u64)
    );
    CurrentResult {
        append_insert,
        append_image: serialized + written,
        append_bytes: image2.len() as u64,
    }
}

fn bench_segmented(
    root: &Path,
    entries: u64,
    lookups: u64,
    flush_entries: usize,
    fsync: bool,
) -> SegmentedOptions {
    println!("\n=== segmented: flush={flush_entries} bloom=10 bits/key fsync={fsync} ===");
    let dir = unique_dir(root, &format!("seg-{flush_entries}"));
    let options = SegmentedOptions {
        flush_entries,
        bloom_bits_per_key: 10,
        compact_threshold: usize::MAX,
        fsync,
    };

    let started = Instant::now();
    let mut index = SegmentedIndex::create(dir.clone(), options).expect("create segmented");
    for i in 0..entries {
        index
            .insert(key(i), i)
            .expect("insert into segmented index");
    }
    index.sync().expect("sync segmented index");
    let build = started.elapsed();
    let stats = index.stats();
    println!(
        "insert+flush      {entries:>10}  {:>9.1} ms  {:>7.1} ns/op",
        ms(build),
        ns_per(build, entries)
    );
    println!(
        "segments          {:>10}        {:>7} tail entries",
        index.segment_count(),
        index.tail_len()
    );
    println!(
        "mapped            {:>10.1} MiB  {:>7.2} B/record",
        mib(index.mapped_bytes()),
        index.mapped_bytes() as f64 / entries as f64
    );
    println!(
        "durable           {:>10.1} MiB  (segment writes {} / {:.1} MiB, manifests {} / {} B, tail log {:.1} MiB)",
        mib(index.disk_bytes()),
        stats.segment_writes,
        mib(stats.segment_bytes),
        stats.manifest_writes,
        stats.manifest_bytes,
        mib(stats.tail_log_bytes)
    );
    println!(
        "syncs             {:>10}        {:>7.1} ms total build",
        stats.flushes,
        ms(build)
    );

    let (hit_time, hits) = lookup_workload(lookups, true, |id| index.lookup(id));
    let (miss_time, misses) = lookup_workload(lookups, false, |id| index.lookup(id));
    println!(
        "lookup hit        {hits:>10}  {:>9.1} ns/op",
        ns_per(hit_time, lookups)
    );
    println!(
        "lookup miss       {misses:>10}  {:>9.1} ns/op",
        ns_per(miss_time, lookups)
    );
    assert_eq!(hits, lookups, "segmented index must find every hit");
    assert_eq!(misses, 0, "segmented index must find no absent key");

    drop(index);
    let started = Instant::now();
    let mut reopened = SegmentedIndex::open(dir.clone(), options).expect("open segmented");
    let reopen = started.elapsed();
    let started = Instant::now();
    reopened.verify_all().expect("verify segmented");
    let verify = started.elapsed();
    println!(
        "reopen trusted    {:>10.1} MiB  {:>9.3} ms  ({} segments, {} tail)",
        mib(reopened.mapped_bytes()),
        ms(reopen),
        reopened.segment_count(),
        reopened.tail_len()
    );
    println!("reopen + verify   {:>10}        {:>9.1} ms", "", ms(verify));

    let (hit_time, hits) = lookup_workload(lookups, true, |id| reopened.lookup(id));
    let (miss_time, misses) = lookup_workload(lookups, false, |id| reopened.lookup(id));
    println!(
        "reopen hit        {hits:>10}  {:>9.1} ns/op",
        ns_per(hit_time, lookups)
    );
    println!(
        "reopen miss       {misses:>10}  {:>9.1} ns/op",
        ns_per(miss_time, lookups)
    );
    assert_eq!(
        hits, lookups,
        "reopened segmented index must find every hit"
    );
    assert_eq!(
        misses, 0,
        "reopened segmented index must find no absent key"
    );

    // Full-merge compaction, measured explicitly.
    if reopened.segment_count() > 1 {
        let before_segments = reopened.segment_count();
        let started = Instant::now();
        let bytes = reopened.compact().expect("compact");
        let compaction = started.elapsed();
        println!(
            "compact {before_segments}->1    {:>10.1} MiB written  {:>9.1} ms",
            mib(bytes),
            ms(compaction)
        );
        let (hit_time, hits) = lookup_workload(lookups, true, |id| reopened.lookup(id));
        let (miss_time, misses) = lookup_workload(lookups, false, |id| reopened.lookup(id));
        println!(
            "compact hit       {hits:>10}  {:>9.1} ns/op",
            ns_per(hit_time, lookups)
        );
        println!(
            "compact miss      {misses:>10}  {:>9.1} ns/op",
            ns_per(miss_time, lookups)
        );
        assert_eq!(hits, lookups);
        assert_eq!(misses, 0);
    }

    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&dir);
    }
    options
}

fn bench_segmented_append(root: &Path, entries: u64, append: u64, fsync: bool) -> (Duration, u64) {
    println!("\n=== segmented incremental append ({append} keys) ===");
    let dir = unique_dir(root, "seg-append");
    let options = SegmentedOptions {
        flush_entries: 65_536,
        bloom_bits_per_key: 10,
        compact_threshold: usize::MAX,
        fsync,
    };
    let mut index = SegmentedIndex::create(dir.clone(), options).expect("create segmented");
    for i in 0..entries {
        index.insert(key(i), i).expect("build insert");
    }
    index.sync().expect("build sync");
    let before = index.stats();

    let started = Instant::now();
    for j in 0..append {
        index
            .insert(key(entries.saturating_add(j)), entries + j)
            .expect("append insert");
    }
    index.sync().expect("append sync");
    let append_time = started.elapsed();
    let after = index.stats();
    let written = after
        .segment_bytes
        .saturating_sub(before.segment_bytes)
        .saturating_add(after.manifest_bytes.saturating_sub(before.manifest_bytes));
    println!(
        "append+sync       {:>10}  {:>9.1} ms  {:>7.1} us/op  {:>7.2} MiB written  ({} -> {} segments)",
        append,
        ms(append_time),
        us_per(append_time, append),
        mib(written),
        before.segment_writes,
        after.segment_writes
    );
    assert_eq!(index.lookup(&key(entries)), Some(entries));
    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&dir);
    }
    (append_time, written)
}

fn main() {
    let entries = env_u64("MTXDB_IS_ENTRIES", 1_000_000);
    let lookups = env_u64("MTXDB_IS_LOOKUPS", 1_000_000);
    let append = env_u64("MTXDB_IS_APPEND", 65_536);
    let fsync = env_bool("MTXDB_IS_FSYNC", true);
    let bench_root = std::env::var("MTXDB_BENCH_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/opencode/mtxdb-index-segments"));
    fs::create_dir_all(&bench_root).expect("bench root");
    let root = unique_dir(&bench_root, "index-segments-run");
    fs::create_dir_all(&root).expect("run root");

    println!(
        "workload: {entries} keys (16B id -> u64 locator), {lookups} lookups, {} MiB index target",
        mib(entries.saturating_mul(24))
    );
    println!("root: {}", root.display());

    let current = bench_current(&root, entries, lookups, append, fsync);
    println!(
        "\ncurrent append summary: insert {:>7.1} ms, image {:>7.1} ms, {:>7.1} MiB rewritten",
        ms(current.append_insert),
        ms(current.append_image),
        mib(current.append_bytes)
    );

    for flush in [65_536_usize, 262_144, 1_048_576] {
        bench_segmented(&root, entries, lookups, flush, fsync);
    }

    let (seg_append, seg_bytes) = bench_segmented_append(&root, entries, append, fsync);
    println!(
        "\nappend comparison: current rewrites {:.1} MiB in {:.1} ms; segmented writes {:.2} MiB in {:.1} ms (insert+sync, {:.1} us/op)",
        mib(current.append_bytes),
        ms(current.append_image),
        mib(seg_bytes),
        ms(seg_append),
        us_per(seg_append, append)
    );

    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&root);
    }
}
