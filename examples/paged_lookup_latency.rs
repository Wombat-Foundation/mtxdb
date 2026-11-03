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

#![allow(clippy::arithmetic_side_effects, clippy::cast_precision_loss)]

use mtxdb::index::IndexResidency;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const COLLECTION: [u8; 16] = [0x4D; 16];
const DEFAULT_SHARD_BYTES: u64 = 8192;
const PAYLOAD_BYTES: usize = 120;
const PROBES: usize = 500;

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

fn payload(sequence: u64) -> [u8; PAYLOAD_BYTES] {
    let mut bytes = [0u8; PAYLOAD_BYTES];
    for (chunk, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(0u64..) {
        chunk.copy_from_slice(&mix(sequence.wrapping_add(word)).to_le_bytes());
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

fn build_fixture(dir: &Path, wanted_packs: usize, shard_bytes: u64) -> Vec<NodeId> {
    let store = PackfileStorage::open_with_max_shard_bytes(dir.to_path_buf(), shard_bytes)
        .expect("open fixture store");
    let mut ids = Vec::new();
    let mut sequence = 0u64;
    // This incompressible record shape uses roughly 341 bytes per frame, so
    // scale the batch with the seal size and add more until the target pack
    // count is met if the estimate is low.
    let records_per_shard = (shard_bytes / 341).max(1);
    let initial_records = u64::try_from(wanted_packs + 1)
        .expect("pack count fits in u64")
        .saturating_mul(records_per_shard);
    for _ in 0..initial_records {
        let id = node_id(sequence);
        let data = payload(sequence);
        store
            .put(&COLLECTION, &id, &NodeData::from_slice(&data))
            .expect("write fixture record");
        ids.push(id);
        sequence += 1;
    }
    store.sync_all().expect("seal fixture packs");
    while sidecar_count(dir) < wanted_packs {
        for _ in 0..records_per_shard {
            let id = node_id(sequence);
            let data = payload(sequence);
            store
                .put(&COLLECTION, &id, &NodeData::from_slice(&data))
                .expect("write fixture record");
            ids.push(id);
            sequence += 1;
        }
        store.sync_all().expect("seal fixture packs");
    }
    store
        .force_index_checkpoint()
        .expect("write resident checkpoint");
    drop(store);
    ids
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

fn average_lookup(store: &PackfileStorage, ids: &[NodeId]) -> Duration {
    let started = Instant::now();
    for id in ids {
        black_box(store.get_location(&COLLECTION, id).expect("point lookup"));
    }
    let probe_count = u32::try_from(ids.len()).expect("probe count fits in u32");
    started.elapsed().div_f64(f64::from(probe_count))
}

fn measure(
    dir: &Path,
    residency: IndexResidency,
    probes: &[NodeId],
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
        best_lookup = best_lookup.min(average_lookup(&store, probes));
        drop(store);
    }
    (best_open, best_lookup, heap_bytes, mapped_bytes)
}

fn main() {
    let rounds = env_usize("MTXDB_LOOKUP_ROUNDS", 5);
    assert!(rounds > 0, "MTXDB_LOOKUP_ROUNDS must be positive");
    let shard_bytes = shard_bytes();
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
        "warm open and get_location latency, best of {rounds}; {PROBES} hits and misses; {shard_bytes} byte seal"
    );
    println!("packs\tmode\topen_ms\thit_us\tmiss_us\theap_mib\tmapped_mib\tsealed_packs\trecords");
    for wanted in pack_counts() {
        let base = root.join(format!("base-{wanted}"));
        std::fs::create_dir_all(&base).expect("create fixture directory");
        let ids = build_fixture(&base, wanted, shard_bytes);

        let hit_range = ids.len().saturating_mul(9) / 10;
        let hits: Vec<NodeId> = (0..PROBES)
            .map(|index| ids[index * hit_range / PROBES])
            .collect();
        let misses: Vec<NodeId> = (0..PROBES)
            .map(|index| node_id(u64::try_from(ids.len()).unwrap() + index as u64 + 10_000_000))
            .collect();
        // Measure Resident first: it consumes the checkpoint the fixture
        // wrote. The first Paged open then deletes that checkpoint, so later
        // Paged rounds exercise the real full-scan open.
        for (label, residency) in [
            ("resident", IndexResidency::Resident),
            ("paged", IndexResidency::Paged),
        ] {
            let (open, hit, heap, mapped) = measure(&base, residency, &hits, rounds, shard_bytes);
            let (_, miss, _, _) = measure(&base, residency, &misses, rounds, shard_bytes);
            println!(
                "{wanted}\t{label}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
                open.as_secs_f64() * 1e3,
                hit.as_secs_f64() * 1e6,
                miss.as_secs_f64() * 1e6,
                heap as f64 / (1024.0 * 1024.0),
                mapped as f64 / (1024.0 * 1024.0),
                sidecar_count(&base),
                ids.len()
            );
        }
        std::fs::remove_dir_all(&base).ok();
    }
    if owned_root {
        let _ = std::fs::remove_dir_all(&root);
    }
}
