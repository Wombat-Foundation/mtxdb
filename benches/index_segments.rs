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
    std::env::var(key).ok().map_or(default, |value| {
        matches!(value.as_str(), "1" | "true" | "yes" | "on")
    })
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

/// Bench counts and byte totals stay below 4 GiB / 4G entries. Clamping keeps
/// the reporting conversion lossless for the actual workload and avoids a
/// `u64 as f64` cast.
fn f64_of_u64(value: u64) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn f64_of_usize(value: usize) -> f64 {
    f64_of_u64(u64::try_from(value).unwrap_or(u64::MAX))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn us_per(duration: Duration, count: u64) -> f64 {
    duration.as_secs_f64() * 1e6 / f64_of_u64(count.max(1))
}

fn ns_per(duration: Duration, count: u64) -> f64 {
    duration.as_secs_f64() * 1e9 / f64_of_u64(count.max(1))
}

fn mib(bytes: u64) -> f64 {
    f64_of_u64(bytes) / (1024.0 * 1024.0)
}

fn unique_dir(root: &Path, label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
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

/// Run a fixed lookup workload and return (elapsed, candidate count).
fn lookup_workload(
    entries: u64,
    count: u64,
    hits: bool,
    lookup: impl Fn(&[u8; 16]) -> Option<u64>,
) -> (Duration, u64) {
    let present = entries.max(1);
    let warm = (count >> 4).max(1024);
    for i in 0..warm {
        let id = if hits {
            key(i.checked_rem(present).unwrap_or(0))
        } else {
            key(present.saturating_add(i))
        };
        black_box(lookup(&id));
    }
    let mut found = 0_u64;
    let started = Instant::now();
    for i in 0..count {
        let id = if hits {
            key(i
                .wrapping_mul(2_654_435_761)
                .checked_rem(present)
                .unwrap_or(0))
        } else {
            key(present.saturating_add(i))
        };
        if black_box(lookup(&id)).is_some() {
            found = found.saturating_add(1);
        }
    }
    (started.elapsed(), found)
}

struct CurrentResult {
    insert: Duration,
    image: Duration,
    bytes: u64,
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
            .insert(&key(i), u16::try_from(i & 4095).expect("slot fits u16"), i)
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
        mib(u64::try_from(index.resident_bytes()).unwrap_or(u64::MAX)),
        f64_of_usize(index.resident_bytes()) / f64_of_u64(entries),
        index.capacity()
    );

    let started = Instant::now();
    let image = index.serialize();
    let serialize = started.elapsed();
    println!(
        "serialize         {:>10.1} MiB  {:>9.1} ms  {:>7.1} ns/record",
        mib(u64::try_from(image.len()).unwrap_or(u64::MAX)),
        ms(serialize),
        ns_per(serialize, entries)
    );

    let image_path = root.join("current-image.bin");
    let started = Instant::now();
    write_all_sync(&image_path, &image, fsync).expect("write current image");
    let write = started.elapsed();
    println!(
        "write+fsync       {:>10.1} MiB  {:>9.1} ms  ({})",
        mib(u64::try_from(image.len()).unwrap_or(u64::MAX)),
        ms(write),
        if fsync { "fsync" } else { "no fsync" }
    );
    let started = Instant::now();
    let reopened =
        LossyIndex::deserialize_with_config(&image, IndexConfig::default()).expect("deserialize");
    let reopen = started.elapsed();
    println!(
        "reopen            {:>10.1} MiB  {:>9.1} ms  -> {:>6.1} MiB heap",
        mib(u64::try_from(image.len()).unwrap_or(u64::MAX)),
        ms(reopen),
        mib(u64::try_from(reopened.resident_bytes()).unwrap_or(u64::MAX))
    );

    let (hit_time, hits) = lookup_workload(entries, lookups, true, |id| {
        reopened.lookup(id).map(|(_, offset)| offset)
    });
    let (miss_time, misses) = lookup_workload(entries, lookups, false, |id| {
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
        f64_of_u64(misses) * 1e6 / f64_of_u64(lookups.max(1))
    );

    drop(reopened);
    bench_current_append(root, &index, entries, append, fsync)
}

fn bench_current_append(
    root: &Path,
    index: &LossyIndex,
    entries: u64,
    append: u64,
    fsync: bool,
) -> CurrentResult {
    // Incremental append: more inserts, then a fresh whole image.
    let started = Instant::now();
    for j in 0..append {
        let next = entries.saturating_add(j);
        index
            .insert(
                &key(next),
                u16::try_from(next & 4095).expect("slot fits u16"),
                next,
            )
            .expect("append insert");
    }
    let append_insert = started.elapsed();
    let started = Instant::now();
    let image = index.serialize();
    let serialized = started.elapsed();
    let image_path = root.join("current-image-append.bin");
    let started = Instant::now();
    write_all_sync(&image_path, &image, fsync).expect("write appended image");
    let written = started.elapsed();
    println!(
        "append {append:>7} + image   insert {:>7.1} ms, serialize {:>7.1} ms, write {:>7.1} ms, {:>7.1} MiB rewritten",
        ms(append_insert),
        ms(serialized),
        ms(written),
        mib(u64::try_from(image.len()).unwrap_or(u64::MAX))
    );
    CurrentResult {
        insert: append_insert,
        image: serialized.saturating_add(written),
        bytes: u64::try_from(image.len()).unwrap_or(u64::MAX),
    }
}

fn print_segmented_build(index: &SegmentedIndex, entries: u64, build: Duration, fsync: bool) {
    let stats = index.stats();
    println!(
        "insert+flush      {entries:>10}  {:>9.1} ms  {:>7.1} ns/op  (fsync={fsync})",
        ms(build),
        ns_per(build, entries)
    );
    println!(
        "segments          {:>10}        {:>7} tail entries",
        index.segment_count(),
        index.tail_len()
    );
    println!(
        "loaded            {:>10.1} MiB  {:>7.2} B/record",
        mib(index.resident_bytes()),
        f64_of_u64(index.resident_bytes()) / f64_of_u64(entries)
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
        "flushes           {:>10}        {:>7.1} ms total build",
        stats.flushes,
        ms(build)
    );
}

fn print_lookup_pair(label: &str, index: &SegmentedIndex, entries: u64, lookups: u64) {
    let (hit_time, hits) = lookup_workload(entries, lookups, true, |id| index.lookup(id));
    let (miss_time, misses) = lookup_workload(entries, lookups, false, |id| index.lookup(id));
    println!(
        "{label} hit        {hits:>10}  {:>9.1} ns/op",
        ns_per(hit_time, lookups)
    );
    println!(
        "{label} miss       {misses:>10}  {:>9.1} ns/op",
        ns_per(miss_time, lookups)
    );
    assert_eq!(
        hits, lookups,
        "{label}: segmented index must find every hit"
    );
    assert_eq!(
        misses, 0,
        "{label}: segmented index must find no absent key"
    );
}

fn bench_segmented(root: &Path, entries: u64, lookups: u64, flush_entries: usize, fsync: bool) {
    println!("\n=== segmented: flush={flush_entries} bloom=10 bits/key fsync={fsync} ===");
    let dir = unique_dir(root, &format!("seg-{flush_entries}"));
    let options = SegmentedOptions {
        flush_entries,
        bloom_bits_per_key: 10,
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
    print_segmented_build(&index, entries, build, fsync);
    print_lookup_pair("lookup", &index, entries, lookups);

    drop(index);
    let started = Instant::now();
    let mut reopened = SegmentedIndex::open(dir.clone(), options).expect("open segmented");
    let reopen = started.elapsed();
    let started = Instant::now();
    reopened.verify_all().expect("verify segmented");
    let verify = started.elapsed();
    println!(
        "reopen loaded     {:>10.1} MiB  {:>9.3} ms  ({} segments, {} tail)",
        mib(reopened.resident_bytes()),
        ms(reopen),
        reopened.segment_count(),
        reopened.tail_len()
    );
    println!("reopen + verify   {:>10}        {:>9.1} ms", "", ms(verify));
    print_lookup_pair("reopen", &reopened, entries, lookups);

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
        print_lookup_pair("compact", &reopened, entries, lookups);
    }

    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&dir);
    }
}

fn bench_segmented_append(
    root: &Path,
    entries: u64,
    append: u64,
    fsync: bool,
    flush_entries: usize,
) -> (Duration, u64) {
    println!("\n=== segmented incremental append ({append} keys, flush={flush_entries}) ===");
    let dir = unique_dir(root, "seg-append");
    let options = SegmentedOptions {
        flush_entries,
        bloom_bits_per_key: 10,
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
        let next = entries.saturating_add(j);
        index.insert(key(next), next).expect("append insert");
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
    assert_eq!(
        index.lookup(&key(entries)),
        Some(entries),
        "appended key must be findable"
    );
    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&dir);
    }
    (append_time, written)
}

fn prototype_self_check(root: &Path, fsync: bool) {
    let dir = unique_dir(root, "self-check");
    let options = SegmentedOptions {
        flush_entries: 4,
        bloom_bits_per_key: 10,
        fsync,
    };
    let mut index = SegmentedIndex::create(dir.clone(), options).expect("create self-check");
    index.insert(key(0), 10).expect("insert k0");
    index.insert(key(1), 11).expect("insert k1");
    index.insert(key(0), 12).expect("overwrite k0 in tail");
    index.sync().expect("first segment");
    assert_eq!(
        index.lookup(&key(0)),
        Some(12),
        "newest tail value must win"
    );
    assert_eq!(
        index.lookup(&key(1)),
        Some(11),
        "unrelated key must survive"
    );
    assert_eq!(index.lookup(&key(2)), None, "absent key must miss");

    // Leave a tail record un-synced, then reopen: the tail log must replay it.
    index.insert(key(2), 13).expect("insert k2");
    drop(index);
    let mut index = SegmentedIndex::open(dir.clone(), options).expect("reopen self-check");
    assert_eq!(
        index.lookup(&key(0)),
        Some(12),
        "segment value across reopen"
    );
    assert_eq!(index.lookup(&key(2)), Some(13), "tail replay across reopen");

    // A second segment and a full merge must keep the newest version.
    index.insert(key(0), 14).expect("overwrite k0 in new tail");
    index.sync().expect("second segment");
    assert_eq!(index.lookup(&key(0)), Some(14), "newer segment must win");
    index.compact().expect("self-check compact");
    assert_eq!(
        index.lookup(&key(0)),
        Some(14),
        "compaction must keep newest"
    );
    assert_eq!(
        index.lookup(&key(1)),
        Some(11),
        "compaction must keep all keys"
    );
    assert_eq!(
        index.lookup(&key(2)),
        Some(13),
        "compaction must keep tail keys"
    );
    drop(index);
    let index = SegmentedIndex::open(dir.clone(), options).expect("reopen after compact");
    assert_eq!(
        index.lookup(&key(0)),
        Some(14),
        "compacted value across reopen"
    );
    assert_eq!(
        index.lookup(&key(1)),
        Some(11),
        "compacted key across reopen"
    );
    assert_eq!(
        index.lookup(&key(2)),
        Some(13),
        "compacted tail key across reopen"
    );
    drop(index);
    let _ = fs::remove_dir_all(&dir);
    println!("self-check: overwrite, tail replay, segment ordering and compaction OK");
}

fn main() {
    let entries = env_u64("MTXDB_IS_ENTRIES", 1_000_000);
    let lookups = env_u64("MTXDB_IS_LOOKUPS", 1_000_000);
    let append = env_u64("MTXDB_IS_APPEND", 65_536);
    let fsync = env_bool("MTXDB_IS_FSYNC", true);
    let bench_root = std::env::var("MTXDB_BENCH_ROOT").map_or_else(
        |_| PathBuf::from("/tmp/opencode/mtxdb-index-segments"),
        PathBuf::from,
    );
    fs::create_dir_all(&bench_root).expect("bench root");
    let root = unique_dir(&bench_root, "index-segments-run");
    fs::create_dir_all(&root).expect("run root");

    println!(
        "workload: {entries} keys (16B id -> u64 locator), {lookups} lookups, {} MiB segment target",
        mib(entries.saturating_mul(24))
    );
    println!("root: {}", root.display());
    prototype_self_check(&root, fsync);

    let current = bench_current(&root, entries, lookups, append, fsync);
    println!(
        "\ncurrent append summary: insert {:>7.1} ms, image {:>7.1} ms, {:>7.1} MiB rewritten",
        ms(current.insert),
        ms(current.image),
        mib(current.bytes)
    );

    for flush in [65_536_usize, 262_144, 1_048_576] {
        bench_segmented(&root, entries, lookups, flush, fsync);
    }

    for flush in [65_536_usize, 262_144, 1_048_576] {
        let (seg_append, seg_bytes) = bench_segmented_append(&root, entries, append, fsync, flush);
        println!(
            "append comparison flush={flush}: current rewrites {:.1} MiB in {:.1} ms; segmented writes {:.2} MiB in {:.1} ms (insert+sync, {:.1} us/op)",
            mib(current.bytes),
            ms(current.image),
            mib(seg_bytes),
            ms(seg_append),
            us_per(seg_append, append)
        );
    }

    if !env_bool("MTXDB_IS_KEEP", false) {
        let _ = fs::remove_dir_all(&root);
    }
}
