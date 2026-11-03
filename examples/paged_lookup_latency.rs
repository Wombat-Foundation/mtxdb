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

#![allow(clippy::arithmetic_side_effects)]

use mtxdb::index::IndexResidency;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const COLLECTION: [u8; 16] = [0x4D; 16];
const SHARD_BYTES: u64 = 8192;
const PAYLOAD_BYTES: usize = 120;
const PROBES: usize = 500;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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
    for (chunk, word) in bytes.chunks_exact_mut(8).zip(0u64..) {
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

fn build_fixture(dir: &Path, wanted_packs: usize) -> Vec<NodeId> {
    let store = PackfileStorage::open_with_max_shard_bytes(dir.to_path_buf(), SHARD_BYTES)
        .expect("open fixture store");
    let mut ids = Vec::new();
    let mut sequence = 0u64;
    // This incompressible record shape fills about 24 frames per 8 KiB shard.
    // Leave one shard active; every earlier shard gets a sidecar once the
    // collection moves to the next home. If the measured frame density differs
    // on a platform, add one shard of records and sync until the target is met.
    let records_per_shard = 24u64;
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

fn copy_fixture(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create lookup-mode directory");
    for entry in std::fs::read_dir(from).expect("read fixture").flatten() {
        let path = entry.path();
        if path.is_file() {
            std::fs::copy(&path, to.join(entry.file_name())).expect("copy fixture file");
        }
    }
}

fn open(dir: &Path, residency: IndexResidency) -> PackfileStorage {
    PackfileStorage::open_with_max_shard_bytes_and_residency(
        dir.to_path_buf(),
        SHARD_BYTES,
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

fn measure(dir: &Path, residency: IndexResidency, probes: &[NodeId], rounds: usize) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let store = open(dir, residency);
        best = best.min(average_lookup(&store, probes));
        drop(store);
    }
    best
}

fn main() {
    let rounds = env_usize("MTXDB_LOOKUP_ROUNDS", 5);
    let root: PathBuf =
        std::env::temp_dir().join(format!("mtxdb-paged-lookup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create benchmark root");

    println!("warm get_location latency, best of {rounds}; {PROBES} hits and misses");
    println!("packs\tmode\thit_us\tmiss_us\tsealed_packs\trecords");
    for wanted in [10usize, 100, 1000] {
        let base = root.join(format!("base-{wanted}"));
        std::fs::create_dir_all(&base).expect("create fixture directory");
        let ids = build_fixture(&base, wanted);
        let resident_dir = root.join(format!("resident-{wanted}"));
        let paged_dir = root.join(format!("paged-{wanted}"));
        copy_fixture(&base, &resident_dir);
        copy_fixture(&base, &paged_dir);

        let hit_range = ids.len().saturating_mul(9) / 10;
        let hits: Vec<NodeId> = (0..PROBES)
            .map(|index| ids[index * hit_range / PROBES])
            .collect();
        let misses: Vec<NodeId> = (0..PROBES)
            .map(|index| node_id(u64::try_from(ids.len()).unwrap() + index as u64 + 10_000_000))
            .collect();
        for (label, dir, residency) in [
            ("resident", &resident_dir, IndexResidency::Resident),
            ("paged", &paged_dir, IndexResidency::Paged),
        ] {
            let hit = measure(dir, residency, &hits, rounds);
            let miss = measure(dir, residency, &misses, rounds);
            println!(
                "{wanted}\t{label}\t{:.3}\t{:.3}\t{}\t{}",
                hit.as_secs_f64() * 1e6,
                miss.as_secs_f64() * 1e6,
                sidecar_count(&base),
                ids.len()
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}
