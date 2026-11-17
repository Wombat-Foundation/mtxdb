//! Measure Paged open and lookup latency with and without a merged run.
//!
//! ```text
//! cargo run --release --example paged_run_latency -- <fixture-dir> [unmerged-packs]
//! ```
//!
//! `<fixture-dir>` is a complete `paged_lookup_latency` fixture (it has a
//! `fixture-records` file). The example measures Paged open, hits and misses
//! with sidecars only, merges every sealed pack except the newest
//! `unmerged-packs` (default 16, capped to leave at least one pack to merge)
//! into one run, measures again, and removes the run afterward. Paged opens
//! update index metadata in the fixture (including removing `index.checkpoint`
//! and writing `index.versions`). `MTXDB_LOOKUP_ROUNDS` sets the rounds
//! (default 3); the best round is reported.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::index::packidx::{open_for_pack, PackIndexIdentity};
use mtxdb::index::run::{merge_to_path, run_file_name, MergeOptions, MergeSource};
use mtxdb::index::IndexResidency;
use mtxdb::packfile::{read_header, ChecksumPolicy};
use mtxdb::storage::NodeId;
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0x4D; 16];
const PROBES: u64 = 500;
const SHARD_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_UNMERGED_PACKS: usize = 16;

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The fixture's record ids, as `paged_lookup_latency` writes them.
fn node_id(sequence: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&mix(sequence).to_le_bytes());
    id[8..].copy_from_slice(&mix(sequence ^ 0xA5A5_A5A5_5A5A_5A5A).to_le_bytes());
    id
}

fn open(dir: &Path) -> PackfileStorage {
    PackfileStorage::open_with_max_shard_bytes_and_residency(
        dir.to_path_buf(),
        SHARD_BYTES,
        ChecksumPolicy::Full,
        IndexResidency::Paged,
    )
    .expect("open fixture")
}

fn average_lookup(store: &PackfileStorage, ids: &[NodeId]) -> Duration {
    let started = Instant::now();
    for id in ids {
        black_box(store.get_location(&COLLECTION, id).expect("point lookup"));
    }
    started.elapsed().div_f64(f64::from(
        u32::try_from(ids.len()).expect("probe count fits u32"),
    ))
}

/// Verify probe presence outside the lookup timings.
fn check_presence(store: &PackfileStorage, ids: &[NodeId], expect_present: bool) {
    for id in ids {
        let found = store.get_location(&COLLECTION, id).expect("point lookup");
        assert_eq!(
            found.is_some(),
            expect_present,
            "unexpected lookup presence for probe {id:02x?}"
        );
    }
}

/// Best `(open, hit, miss)` over `rounds` fresh opens.
fn measure(
    dir: &Path,
    hits: &[NodeId],
    misses: &[NodeId],
    rounds: usize,
) -> (Duration, Duration, Duration) {
    let mut best = (Duration::MAX, Duration::MAX, Duration::MAX);
    for _ in 0..rounds {
        let started = Instant::now();
        let store = open(dir);
        best.0 = best.0.min(started.elapsed());
        check_presence(&store, hits, true);
        check_presence(&store, misses, false);
        best.1 = best.1.min(average_lookup(&store, hits));
        best.2 = best.2.min(average_lookup(&store, misses));
    }
    best
}

/// Every pack in `dir`, oldest first, with its identity.
fn packs(dir: &Path) -> Vec<(PackIndexIdentity, PathBuf)> {
    let mut packs: Vec<_> = std::fs::read_dir(dir)
        .expect("read fixture")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "pack"))
        .map(|path| {
            let mut file = std::fs::File::open(&path).expect("open pack");
            let header = read_header(&mut file)
                .expect("read pack header")
                .expect("pack header");
            let identity = PackIndexIdentity {
                pack_id: header.pack_id,
                creation_seq: u64::from(header.creation_seq),
                pack_file_len: file.metadata().expect("pack metadata").len(),
            };
            (identity, path)
        })
        .collect();
    packs.sort_by_key(|(identity, _)| (identity.creation_seq, identity.pack_id.0));
    packs
}

fn print_row(label: &str, (open, hit, miss): (Duration, Duration, Duration)) {
    println!(
        "{label}\t{:.3}\t{:.3}\t{:.3}",
        open.as_secs_f64() * 1e3,
        hit.as_secs_f64() * 1e6,
        miss.as_secs_f64() * 1e6,
    );
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut args = std::env::args_os().skip(1);
    let dir = PathBuf::from(
        args.next()
            .expect("usage: paged_run_latency <fixture-dir> [unmerged-packs]"),
    );
    let requested_unmerged = args.next().map(|value| {
        value
            .to_string_lossy()
            .parse::<usize>()
            .expect("unmerged-packs must be a non-negative integer")
    });
    let rounds: usize = std::env::var("MTXDB_LOOKUP_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    assert!(rounds > 0, "MTXDB_LOOKUP_ROUNDS must be at least 1");
    let records: u64 = std::fs::read_to_string(dir.join("fixture-records"))
        .expect("fixture-records")
        .trim()
        .parse()
        .expect("record count");
    let existing_runs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read fixture")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "run"))
        .collect();
    assert!(
        existing_runs.is_empty(),
        "fixture already contains run files that Paged open would load: {existing_runs:?}"
    );

    let hit_range = records.saturating_mul(9) / 10;
    let hits: Vec<NodeId> = (0..PROBES)
        .map(|index| node_id(index.saturating_mul(hit_range) / PROBES))
        .collect();
    let misses: Vec<NodeId> = (0..PROBES).map(|index| node_id(u64::MAX ^ index)).collect();

    let all = packs(&dir);
    let mut sealed = Vec::new();
    let mut missing: Vec<PathBuf> = Vec::new();
    for (identity, path) in &all {
        match open_for_pack(path) {
            Some(sidecar) => {
                if let Some(unsealed) = missing.first() {
                    panic!(
                        "sidecar gap: {} has no openable sidecar, but later pack {} does",
                        unsealed.display(),
                        path.display()
                    );
                }
                sealed.push((*identity, sidecar));
            }
            None => missing.push(path.clone()),
        }
    }
    assert!(
        !sealed.is_empty(),
        "fixture has no openable sealed-pack sidecars"
    );
    assert!(
        missing.len() <= 1,
        "only the active pack may lack a sidecar, but {} trailing packs do: {missing:?}",
        missing.len()
    );
    let default_unmerged = DEFAULT_UNMERGED_PACKS.min(sealed.len().saturating_sub(1));
    let unmerged = requested_unmerged.unwrap_or(default_unmerged);
    assert!(
        unmerged < sealed.len(),
        "unmerged-packs ({unmerged}) must be less than the {} openable sealed packs",
        sealed.len()
    );
    let covered = sealed
        .len()
        .checked_sub(unmerged)
        .expect("unmerged pack count was validated");
    let last_covered = covered
        .checked_sub(1)
        .expect("at least one pack is covered");
    // Packs left outside the run, including the active pack without a
    // sidecar, so the reported label matches the on-disk pack set.
    let unmerged_total = all
        .len()
        .checked_sub(covered)
        .expect("covered packs come from the fixture");

    println!("mode\topen_ms\thit_us\tmiss_us");
    print_row("sidecars", measure(&dir, &hits, &misses, rounds));

    let sources: Vec<MergeSource<'_>> = sealed[..covered]
        .iter()
        .map(|(_, sidecar)| MergeSource::Sidecar(sidecar))
        .collect();
    let run_path = dir.join(run_file_name(&sealed[0].0, &sealed[last_covered].0));
    assert!(
        !run_path.exists(),
        "refusing to overwrite existing run file {}",
        run_path.display()
    );
    let _cleanup = RemoveRunOnDrop(run_path.clone());
    let started = Instant::now();
    let summary = merge_to_path(
        &sources,
        &MergeOptions {
            drop_removals: true,
            ..MergeOptions::default()
        },
        &run_path,
    )
    .expect("merge run")
    .expect("non-empty run");
    println!(
        "merged {covered} of {} packs ({} with sidecars) in {:.1} ms: {} entries, {} blocks, {:.1} MiB",
        all.len(),
        sealed.len(),
        started.elapsed().as_secs_f64() * 1e3,
        summary.entry_count,
        summary.block_count,
        f64::from(u32::try_from(summary.file_len >> 10).unwrap_or(u32::MAX)) / 1024.0,
    );
    drop(sources);
    drop(sealed);

    let with_run = measure(&dir, &hits, &misses, rounds);
    print_row(&format!("run+{unmerged_total}"), with_run);
}

struct RemoveRunOnDrop(PathBuf);

impl Drop for RemoveRunOnDrop {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.0) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "failed to remove temporary run {}: {error}",
                    self.0.display()
                );
            }
        }
    }
}
