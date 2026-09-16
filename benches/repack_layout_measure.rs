//! Characterize what drives interleaved fragmentation
//! (`mtxdb shards --layout`'s runs / excess-runs metric) and verify that a
//! batch repack consolidates each collection into a single contiguous run.
//!
//! Run with `cargo bench --manifest-path benches/Cargo.toml --bench repack_layout_measure`
//! (or `cd benches && cargo bench --bench repack_layout_measure`).
#![allow(
    clippy::arithmetic_side_effects,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::path::Path;

use mtxdb_core::packfile::layout::physical_layout;
use mtxdb_core::storage::{NodeData, NodeId, StorageEngine};
use mtxdb_core::PackfileStorage;

const COLLECTIONS: usize = 250;
const ROUNDS: usize = 40;
const PER_ROUND: usize = 3;

fn collection_id(seed: u16) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[..2].copy_from_slice(&seed.to_le_bytes());
    let mut previous = id[0];
    for byte in id.iter_mut().skip(2) {
        previous = previous
            .wrapping_mul(31)
            .wrapping_add(seed.rotate_left(3).to_le_bytes()[0]);
        *byte = previous;
    }
    id
}

fn node_id(seed: u64, index: u64) -> NodeId {
    let mut id = [0u8; 16];
    let mut x = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    id[..8].copy_from_slice(&x.to_le_bytes());
    let mut y = x.wrapping_add(0x7372_9A1E_4288_1F7D);
    y = (y ^ (y >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    y = (y ^ (y >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    id[8..].copy_from_slice(&(y ^ (y >> 31)).to_le_bytes());
    id
}

fn measure(dir: &Path) -> (u64, u64, u64) {
    let layout = physical_layout(dir).unwrap();
    let collections: usize = layout.packs.values().map(|p| p.collections.len()).sum();
    let runs: u64 = layout.packs.values().map(|p| p.segments).sum();
    let largest = layout
        .packs
        .values()
        .map(|p| p.largest_segment_bytes)
        .max()
        .unwrap_or(0);
    let excess = runs.saturating_sub(collections as u64);
    (runs, excess, largest)
}

fn report_interleave_vs_repack() {
    let dir = std::env::temp_dir().join(format!("mtxdb_interleave_repack_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let store = PackfileStorage::open(dir.clone()).unwrap();

    // Simulate the workload's write pattern: many collections sharing one
    // active pack, each appending in small interleaved batches.
    for round in 0..ROUNDS {
        for c in 0..COLLECTIONS {
            let id = collection_id(u16::try_from(c).unwrap());
            let mut entries = Vec::with_capacity(PER_ROUND);
            for k in 0..PER_ROUND {
                let nid = node_id(
                    u64::try_from(c).unwrap(),
                    u64::try_from(round * PER_ROUND + k).unwrap(),
                );
                entries.push((nid, NodeData::new(bytes::Bytes::from(vec![0xAB; 128]))));
            }
            store.put_many(&id, &entries).unwrap();
        }
    }
    store.sync_all().unwrap();

    let total_nodes = COLLECTIONS * ROUNDS * PER_ROUND;
    let (runs_pre, excess_pre, _) = measure(&dir);
    println!(
        "\nPRE-REPACK:  runs={runs_pre}  excess={excess_pre}  collections={COLLECTIONS}  nodes={total_nodes}"
    );

    let all: Vec<[u8; 16]> = (0..COLLECTIONS)
        .map(|c| collection_id(u16::try_from(c).unwrap()))
        .collect();
    store
        .repack_collections_reachable(&all, |_hash, _data| Vec::new())
        .unwrap();
    store.sync_all().unwrap();

    let (runs_post, excess_post, largest_post) = measure(&dir);
    println!(
        "POST-REPACK: runs={runs_post}  excess={excess_post}  largest_run={largest_post} bytes"
    );

    drop(store);
    std::fs::remove_dir_all(&dir).unwrap();
}

fn main() {
    report_interleave_vs_repack();
}
