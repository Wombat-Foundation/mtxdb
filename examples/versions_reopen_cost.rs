//! Compare the reopen cost of the two homes for collection versions:
//!
//! * A: one `index.versions` file (`24 B × collections` plus framing), read in
//!   one sequential pass.
//! * B: a version in each sealed pack's sidecar header, which requires one
//!   header read per pack.
//!
//! The harness builds the requested number of sealed packs, then times the
//! sidecar-open work only. B is measured as `open_for_pack` over every sidecar,
//! summing each index's `entry_count()`. It reads no collection version and
//! performs no max-fold, so B is a lower bound on the version-recovery cost.
//!
//! ```text
//! MTXDB_VERSIONS_PACKS=1000 MTXDB_VERSIONS_COLLECTIONS=64 \
//!   cargo run --release --example versions_reopen_cost
//! ```
//!
//! Numbers are warm-cache. A cold B is worse: every sidecar is a separate
//! random read, while A stays one file.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use mtxdb::index::{packidx, versions, IndexResidency};
use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::PackfileStorage;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const COLLECTION_SHARD_BYTES: u64 = 8 * 1024;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn collection_ids(count: usize) -> Vec<[u8; 16]> {
    (0..count)
        .map(|index| {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&(index as u64).to_le_bytes());
            id
        })
        .collect()
}

fn sidecars(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "idx"))
        .collect()
}

fn record_id(sequence: usize) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&(sequence as u64).to_le_bytes());
    id[8..].copy_from_slice(&(sequence as u64 ^ 0xA5A5_A5A5_A5A5_A5A5).to_le_bytes());
    id
}

/// Write round-robin across the collections until the store has sealed at
/// least `wanted` packs.
fn build_packs(dir: &Path, collections: &[[u8; 16]], wanted: usize) {
    let store =
        PackfileStorage::open_with_max_shard_bytes(dir.to_path_buf(), COLLECTION_SHARD_BYTES)
            .expect("open writer");
    let payload = [0x5Au8; 120];
    let mut sequence = 0usize;
    loop {
        for collection in collections {
            store
                .put(
                    collection,
                    &record_id(sequence),
                    &NodeData::from_slice(&payload),
                )
                .expect("put");
            sequence += 1;
        }
        if sequence.is_multiple_of(collections.len() * 16) && sidecars(dir).len() >= wanted {
            break;
        }
        assert!(
            sequence < 4_000_000,
            "wrote too many records without sealing"
        );
    }
    store.sync_all().expect("sync");
    drop(store);
}

fn sample<F: FnMut() -> usize>(rounds: usize, mut run: F) -> (Duration, usize) {
    let mut best = Duration::MAX;
    let mut last = 0;
    for _ in 0..rounds {
        let started = Instant::now();
        last = run();
        best = best.min(started.elapsed());
    }
    (best, last)
}

fn main() {
    let wanted = env_usize("MTXDB_VERSIONS_PACKS", 1_000);
    let collection_count = env_usize("MTXDB_VERSIONS_COLLECTIONS", 64);
    let rounds = env_usize("MTXDB_VERSIONS_ROUNDS", 25);
    assert!(
        collection_count > 0,
        "MTXDB_VERSIONS_COLLECTIONS must be positive"
    );
    assert!(rounds > 0, "MTXDB_VERSIONS_ROUNDS must be positive");
    let collections = collection_ids(collection_count);
    let dir = std::env::temp_dir().join(format!(
        "mtxdb-versions-cost-{}-{}",
        std::process::id(),
        wanted
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    println!(
        "building {wanted} sealed packs across {collection_count} collections in {}",
        dir.display()
    );
    let build_started = Instant::now();
    build_packs(&dir, &collections, wanted);
    let packs = sidecars(&dir);
    let pack_count = packs.len();
    let build_elapsed = build_started.elapsed();
    println!("built {pack_count} packs in {build_elapsed:?}");
    assert!(
        packs.len() >= wanted,
        "only {} of {wanted} packs sealed",
        packs.len()
    );

    let versions_path = dir.join(versions::INDEX_VERSIONS_FILE);
    let version_entries: Vec<([u8; 16], u64)> = collections
        .iter()
        .enumerate()
        .map(|(index, collection)| (*collection, 1_000 + index as u64))
        .collect();
    versions::write_versions(&versions_path, &version_entries, 0).expect("write versions");
    let bytes = std::fs::metadata(&versions_path).map_or(0, |meta| meta.len());
    let version_count = version_entries.len();
    let bytes_per_collection = usize::try_from(bytes).unwrap_or(0) / version_entries.len();
    println!("A: {version_count} entries, {bytes} bytes ({bytes_per_collection} B/collection)");

    let (a_duration, a_entries) = sample(rounds, || {
        versions::read_versions(&versions_path)
            .expect("read versions")
            .versions
            .len()
    });

    let mut b_misses = 0usize;
    let (b_duration, b_entries) = sample(rounds, || {
        let mut seen = 0usize;
        let mut missed = 0usize;
        for sidecar in &packs {
            let pack = sidecar.with_extension("pack");
            if let Some(index) = packidx::open_for_pack(&pack) {
                seen =
                    seen.saturating_add(usize::try_from(index.entry_count()).unwrap_or(usize::MAX));
            } else {
                missed += 1;
            }
        }
        assert_eq!(
            missed, 0,
            "failed to open {missed} of {pack_count} sidecars"
        );
        b_misses = missed;
        seen
    });
    assert_eq!(
        b_misses, 0,
        "every pack must have a readable sidecar for the B measurement"
    );

    // The cheapest form of B: one open and one fixed-size header read per
    // sidecar, no full-body verification. This is the floor if B rides along
    // with a paged open that already maps every sidecar (where its marginal
    // cost is only the version field itself, i.e. near zero I/O).
    let (b_header_duration, b_header_bytes) = sample(rounds, || {
        use std::io::Read;
        let mut header = [0u8; 64];
        let mut total = 0usize;
        for sidecar in &packs {
            if let Ok(mut file) = std::fs::File::open(sidecar) {
                if file.read_exact(&mut header).is_ok() {
                    total = total.saturating_add(header.len());
                }
            }
        }
        total
    });

    println!("A read_versions: {a_duration:?} best of {rounds} ({a_entries} entries)");
    println!(
        "B open_for_pack x{pack_count}: {b_duration:?} best of {rounds} \
         ({b_entries} sidecar entries touched)"
    );
    println!(
        "B header-only x{pack_count}: {b_header_duration:?} best of {rounds} \
         ({b_header_bytes} header bytes folded)"
    );
    if a_duration.as_nanos() > 0 {
        let ratio = b_duration.as_secs_f64() / a_duration.as_secs_f64();
        println!("B/A ratio: {ratio:.1}x");
    }
    let per_pack = b_duration
        .checked_div(u32::try_from(pack_count).unwrap_or(u32::MAX))
        .unwrap_or_default();
    println!("per-pack B cost: {per_pack:?}");

    match PackfileStorage::open_with_max_shard_bytes_and_residency(
        dir.clone(),
        COLLECTION_SHARD_BYTES,
        mtxdb::packfile::ChecksumPolicy::Full,
        IndexResidency::Paged,
    ) {
        Ok(open) => {
            let timings = open.open_timings();
            println!("paged full open timings: {timings:#?}");
            drop(open);
        }
        Err(error) => println!("paged full open: failed: {error}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}
