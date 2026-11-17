//! Compare warm point-lookup latency with resident locators and sealed-pack
//! sidecars at increasing pack counts.
//!
//! ```text
//! cargo run --release --example paged_lookup_latency
//! ```
//!
//! `MTXDB_LOOKUP_ROUNDS` controls repetitions (default 5). The benchmark
//! measures `get_location`, which performs the production locator search and
//! verifies the winning frame but avoids payload decode/cache effects. Hit
//! keys are spread over the older 90% of the collection; misses are unique.
//! `MTXDB_LOOKUP_SHARD_BYTES` changes the seal size, and
//! `MTXDB_LOOKUP_PACK_COUNTS=1000` runs only a selected pack-count fixture.
//! `MTXDB_SIDECAR_STALL_TIMEOUT_SECS` sets the fixture's no-progress timeout
//! for sidecar builds (default 5).
//! With `MTXDB_LOOKUP_ROOT` set, fixtures are kept there and a complete one
//! (marked by its `fixture-records` file) is reused by later runs with the same
//! pack count, seal size and payload size.

#![allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]

use mtxdb::index::IndexResidency;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::shard::AppendPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const COLLECTION: [u8; 16] = [0x4D; 16];
const DEFAULT_SHARD_BYTES: u64 = 8192;
const PAYLOAD_BYTES: usize = 120;
const PROBES: usize = 500;
/// Default no-progress timeout for asynchronous sidecar builds. Override with
/// `MTXDB_SIDECAR_STALL_TIMEOUT_SECS` for large or slow fixtures.
const DEFAULT_SIDECAR_STALL_TIMEOUT_SECS: u64 = 5;
const SIDECAR_POLL_INTERVAL: Duration = Duration::from_millis(50);

fn sidecar_stall_timeout() -> Duration {
    let seconds = std::env::var("MTXDB_SIDECAR_STALL_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SIDECAR_STALL_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn shard_bytes() -> u64 {
    env_u64("MTXDB_LOOKUP_SHARD_BYTES", DEFAULT_SHARD_BYTES)
}

fn pack_counts() -> Vec<usize> {
    std::env::var("MTXDB_LOOKUP_PACK_COUNTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .filter(|count| *count > 0)
                .collect()
        })
        .filter(|counts: &Vec<usize>| !counts.is_empty())
        .unwrap_or_else(|| vec![10, 100, 1000])
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn node_id(sequence: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&mix(sequence).to_le_bytes());
    id[8..].copy_from_slice(&mix(sequence ^ 0xA5A5_A5A5_5A5A_5A5A).to_le_bytes());
    id
}

fn payload_len() -> usize {
    let len = env_usize("MTXDB_LOOKUP_PAYLOAD_BYTES", PAYLOAD_BYTES);
    assert!(len > 0, "MTXDB_LOOKUP_PAYLOAD_BYTES must be positive");
    len
}

/// `len` incompressible bytes, including a partial final word.
fn payload(sequence: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    for (chunk, word) in bytes.chunks_mut(8).zip(0u64..) {
        let value = mix(sequence.wrapping_add(word)).to_le_bytes();
        chunk.copy_from_slice(&value[..chunk.len()]);
    }
    bytes
}

fn sidecar_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .expect("read fixture directory")
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "idx"))
        .count()
}

/// File in a fixture directory recording how many records it holds; present
/// only once the fixture is complete.
const FIXTURE_RECORDS: &str = "fixture-records";

/// Record count of a complete fixture with at least `wanted` sealed packs.
fn fixture_records(dir: &Path, wanted: usize) -> Option<u64> {
    let records = std::fs::read_to_string(dir.join(FIXTURE_RECORDS))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (sidecar_count(dir) >= wanted).then_some(records)
}

fn append_batch(store: &PackfileStorage, sequence: &mut u64, count: u64, payload_len: usize) {
    let capacity = usize::try_from(count).unwrap_or(0);
    let mut batch = Vec::with_capacity(capacity);
    for _ in 0..count {
        let id = node_id(*sequence);
        let data = payload(*sequence, payload_len);
        batch.push((id, NodeData::from_slice(&data)));
        *sequence += 1;
    }
    store
        .put_many(&COLLECTION, &batch)
        .expect("write fixture batch");
}

fn build_fixture(dir: &Path, wanted_packs: usize, shard_bytes: u64, payload_len: usize) -> u64 {
    // Buffer frames in memory between syncs: the build ends in `sync_all`,
    // so per-frame positioned writes would only slow it down.
    let store = PackfileStorage::open_with_max_shard_bytes(dir.to_path_buf(), shard_bytes)
        .expect("open fixture store")
        .with_append_policy(AppendPolicy::buffered());
    let mut sequence = 0u64;
    // A rough first guess at records per pack (frame overhead is about 48
    // bytes). The top-up below corrects it from the measured count.
    let frame_bytes = u64::try_from(payload_len)
        .unwrap_or(u64::MAX)
        .saturating_add(48);
    let records_per_shard = (shard_bytes / frame_bytes).max(1);
    let initial_records = u64::try_from(wanted_packs + 1)
        .expect("pack count fits in u64")
        .saturating_mul(records_per_shard);
    let mut remaining = initial_records;
    while remaining > 0 {
        let count = records_per_shard.min(remaining);
        append_batch(&store, &mut sequence, count, payload_len);
        remaining -= count;
    }
    store.sync_all().expect("seal fixture packs");
    // The estimate above can be far off (frames are about 180 bytes, not
    // 341), and every sync rewrites the whole index checkpoint, so top up
    // from the measured records per pack instead of syncing once per batch.
    let wanted = u64::try_from(wanted_packs).expect("pack count fits in u64");
    let stall_timeout = sidecar_stall_timeout();
    let mut last_sealed = 0u64;
    let mut stalled_since = None;
    loop {
        let sealed = u64::try_from(sidecar_count(dir)).expect("pack count fits in u64");
        if sealed >= wanted {
            break;
        }
        if sealed > last_sealed {
            last_sealed = sealed;
            stalled_since = None;
        } else {
            let since = *stalled_since.get_or_insert_with(Instant::now);
            assert!(
                since.elapsed() < stall_timeout,
                "fixture top-up stalled at {sealed} of {wanted} sidecars; \
                 mtxdb-pack-index may have failed (check its log)"
            );
            std::thread::sleep(SIDECAR_POLL_INTERVAL);
            continue;
        }
        let per_pack = (sequence / sealed.max(1)).max(1);
        let mut remaining = (wanted - sealed + 1).saturating_mul(per_pack);
        while remaining > 0 {
            let count = records_per_shard.min(remaining);
            append_batch(&store, &mut sequence, count, payload_len);
            remaining -= count;
        }
        store.sync_all().expect("seal fixture packs");
    }
    store
        .force_index_checkpoint()
        .expect("write resident checkpoint");
    drop(store);
    sequence
}

fn open(dir: &Path, residency: IndexResidency, shard_bytes: u64) -> PackfileStorage {
    PackfileStorage::open_with_max_shard_bytes_and_residency(
        dir.to_path_buf(),
        shard_bytes,
        ChecksumPolicy::Full,
        residency,
    )
    .expect("open lookup fixture")
}

fn average_lookup(store: &PackfileStorage, ids: &[NodeId], expected_present: bool) -> Duration {
    let started = Instant::now();
    let mut mismatches = 0usize;
    for id in ids {
        let found = black_box(store.get_location(&COLLECTION, id).expect("point lookup"));
        if found.is_some() != expected_present {
            mismatches = mismatches.saturating_add(1);
        }
    }
    let elapsed = started.elapsed();
    assert_eq!(
        mismatches,
        0,
        "{} of {} probes returned the wrong presence (expected {expected_present})",
        mismatches,
        ids.len()
    );
    let probe_count = u32::try_from(ids.len()).expect("probe count fits in u32");
    elapsed.div_f64(f64::from(probe_count))
}

fn measure(
    dir: &Path,
    residency: IndexResidency,
    probes: &[NodeId],
    expected_present: bool,
    rounds: usize,
    shard_bytes: u64,
) -> (Duration, Duration, usize, usize) {
    let mut best_open = Duration::MAX;
    let mut best_lookup = Duration::MAX;
    let mut heap_bytes = 0;
    let mut mapped_bytes = 0;
    for _ in 0..rounds {
        let open_started = Instant::now();
        let store = open(dir, residency, shard_bytes);
        best_open = best_open.min(open_started.elapsed());
        if let Some((_entries, heap, mapped, _capacity)) =
            store.collection_index_residency(&COLLECTION)
        {
            heap_bytes = heap;
            mapped_bytes = mapped;
        }
        best_lookup = best_lookup.min(average_lookup(&store, probes, expected_present));
        drop(store);
    }
    (best_open, best_lookup, heap_bytes, mapped_bytes)
}

fn main() {
    let rounds = env_usize("MTXDB_LOOKUP_ROUNDS", 5);
    assert!(rounds > 0, "MTXDB_LOOKUP_ROUNDS must be positive");
    let shard_bytes = shard_bytes();
    let payload_len = payload_len();
    let (root, owned_root) = std::env::var_os("MTXDB_LOOKUP_ROOT").map_or_else(
        || {
            (
                std::env::temp_dir().join(format!(
                    "mtxdb-paged-lookup-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                )),
                true,
            )
        },
        |path| (PathBuf::from(path), false),
    );
    std::fs::create_dir_all(&root).expect("create benchmark root");

    println!(
        "warm open and get_location latency, best of {rounds}; {PROBES} hits and misses; {shard_bytes} byte seal; {payload_len} byte payload"
    );
    println!("packs\tmode\topen_ms\thit_us\tmiss_us\theap_mib\tmapped_mib\tsealed_packs\trecords");
    for wanted in pack_counts() {
        // Key the fixture by everything that shapes its packs, so a reused
        // fixture always matches the requested seal and payload size.
        let base = root.join(format!("base-{wanted}-s{shard_bytes}-p{payload_len}"));
        let total_records = if let Some(records) = fixture_records(&base, wanted) {
            // A Paged open deletes the resident checkpoint, so restore it
            // before the Resident measurement, as a fresh build would.
            PackfileStorage::open_with_max_shard_bytes(base.clone(), shard_bytes)
                .expect("open reused fixture")
                .force_index_checkpoint()
                .expect("rewrite resident checkpoint");
            records
        } else {
            // Build under a per-run directory and publish it with an atomic
            // rename. A concurrent run with the same parameters therefore
            // never sees (or deletes) this directory while it is incomplete.
            let building = root.join(format!(
                ".build-{wanted}-s{shard_bytes}-p{payload_len}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&building).expect("create fixture directory");
            let records = build_fixture(&building, wanted, shard_bytes, payload_len);
            std::fs::write(building.join(FIXTURE_RECORDS), records.to_string())
                .expect("record fixture size");
            if fixture_records(&base, wanted).is_some() {
                let _ = std::fs::remove_dir_all(&building);
                fixture_records(&base, wanted).expect("completed fixture")
            } else {
                match std::fs::rename(&building, &base) {
                    Ok(()) => records,
                    Err(error) => {
                        // Another run may have published the completed fixture
                        // while this one was building.
                        if let Some(records) = fixture_records(&base, wanted) {
                            let _ = std::fs::remove_dir_all(&building);
                            records
                        } else {
                            panic!(
                                "cannot install fixture at {}: {error}; \
                                 remove an incomplete fixture directory and retry",
                                base.display()
                            );
                        }
                    }
                }
            }
        };

        let hit_range = total_records.saturating_mul(9) / 10;
        let hits: Vec<NodeId> = (0..PROBES)
            .map(|index| node_id((index as u64).saturating_mul(hit_range) / (PROBES as u64)))
            .collect();
        // Fixture ids count up from zero, so ids counted down from the top of
        // the sequence space are never written.
        let misses: Vec<NodeId> = (0..PROBES)
            .map(|index| node_id(u64::MAX - index as u64))
            .collect();
        // Measure Resident first: it consumes the checkpoint the fixture
        // wrote. The first Paged open then deletes that checkpoint, so later
        // Paged rounds exercise the real full-scan open.
        for (label, residency) in [
            ("resident", IndexResidency::Resident),
            ("paged", IndexResidency::Paged),
        ] {
            let (open, hit, heap, mapped) =
                measure(&base, residency, &hits, true, rounds, shard_bytes);
            let (_, miss, _, _) = measure(&base, residency, &misses, false, rounds, shard_bytes);
            println!(
                "{wanted}\t{label}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
                open.as_secs_f64() * 1e3,
                hit.as_secs_f64() * 1e6,
                miss.as_secs_f64() * 1e6,
                heap as f64 / (1024.0 * 1024.0),
                mapped as f64 / (1024.0 * 1024.0),
                sidecar_count(&base),
                total_records
            );
        }
    }
    if owned_root {
        let _ = std::fs::remove_dir_all(&root);
    }
}
