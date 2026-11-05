//! Measure `put` latency while a Paged store builds and installs its first
//! merged run, against a control pass with no merge in flight.
//!
//! ```text
//! cargo run --release --example paged_merge_put_latency -- <fixture-dir> <scratch-dir>
//! ```
//!
//! `<fixture-dir>` is a complete `paged_lookup_latency` fixture with no run
//! yet. If `fixture-records` is present, new ids continue after it; otherwise
//! the example uses a high, non-overlapping id range. The source is never
//! written: the example
//! copies it to `<scratch-dir>`, which must not exist, and deletes the copy at
//! the end unless `MTXDB_KEEP_SCRATCH` is set.
//!
//! The copy is opened writable in Paged mode, so the first `put` drains the
//! sealed packs and starts a merge over every unmerged sealed pack. Puts of
//! fresh 120-byte records then run at `MTXDB_PUT_RATE` per second (default
//! 5,000) until the run file is published, and for `MTXDB_PUT_TAIL_MS` more
//! (default 1,000) so the `put` that installs the run is sampled. The control
//! pass repeats the same number of puts at the same rate once the merge has
//! landed. No `sync` runs inside either sample. Each sample is a single
//! `put` call's wall time.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::index::run::RunIndex;
use mtxdb::index::IndexResidency;
use mtxdb::packfile::ChecksumPolicy;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0x4D; 16];
const PAYLOAD_BYTES: usize = 120;
const SHARD_BYTES: u64 = 4 * 1024 * 1024;
/// Upper bound on the merge pass, in case no run is ever published.
const MERGE_PASS_LIMIT: Duration = Duration::from_secs(300);

fn env_u64(name: &str, default: u64) -> u64 {
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

/// The fixture's record ids, as `paged_lookup_latency` writes them.
fn node_id(sequence: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&mix(sequence).to_le_bytes());
    id[8..].copy_from_slice(&mix(sequence ^ 0xA5A5_A5A5_5A5A_5A5A).to_le_bytes());
    id
}

/// Incompressible bytes, as `paged_lookup_latency` writes them.
fn payload(sequence: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; PAYLOAD_BYTES];
    for (chunk, word) in bytes.chunks_mut(8).zip(0u64..) {
        let value = mix(sequence.wrapping_add(word)).to_le_bytes();
        chunk.copy_from_slice(&value[..chunk.len()]);
    }
    bytes
}

fn copy_fixture(from: &Path, to: &Path) {
    assert!(!to.exists(), "{} already exists", to.display());
    std::fs::create_dir_all(to).expect("create scratch directory");
    for entry in std::fs::read_dir(from).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        let path = entry.path();
        if path.is_file() {
            std::fs::copy(&path, to.join(entry.file_name())).expect("copy fixture file");
        }
    }
}

fn run_files(dir: &Path) -> Vec<PathBuf> {
    let mut runs: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read scratch directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "run"))
        .collect();
    runs.sort();
    runs
}

fn count_ext(dir: &Path, ext: &str) -> usize {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|e| e == ext))
            .count()
    })
}

/// Paced puts: one every `interval`, each sample is that call's wall time.
struct Pacer {
    started: Instant,
    interval: Duration,
    issued: u32,
}

impl Pacer {
    fn new(rate: u64) -> Self {
        Self {
            started: Instant::now(),
            interval: Duration::from_secs(1)
                .checked_div(u32::try_from(rate.max(1)).unwrap_or(u32::MAX))
                .unwrap_or_default(),
            issued: 0,
        }
    }

    fn wait(&mut self) {
        let due = self
            .interval
            .checked_mul(self.issued)
            .and_then(|offset| self.started.checked_add(offset));
        if let Some(ahead) = due.and_then(|due| due.checked_duration_since(Instant::now())) {
            std::thread::sleep(ahead);
        }
        self.issued = self.issued.saturating_add(1);
    }
}

fn put_one(store: &PackfileStorage, sequence: u64) -> Duration {
    let id = node_id(sequence);
    let data = NodeData::from_slice(&payload(sequence));
    let started = Instant::now();
    store.put(&COLLECTION, &id, &data).expect("put");
    started.elapsed()
}

fn percentile(sorted: &[Duration], per_mille: usize) -> Duration {
    let rank = sorted
        .len()
        .saturating_mul(per_mille)
        .div_ceil(1000)
        .saturating_sub(1);
    sorted.get(rank).copied().unwrap_or_default()
}

fn report(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let us = |value: Duration| value.as_secs_f64() * 1e6;
    println!(
        "{label}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
        samples.len(),
        us(percentile(&samples, 500)),
        us(percentile(&samples, 990)),
        us(percentile(&samples, 999)),
        us(samples.last().copied().unwrap_or_default()),
    );
}

/// Note a put slower than 2 ms with the pack and sidecar counts, so a stall
/// can be matched to a pack rotation and its eviction.
fn note_slow(label: &str, index: usize, at: Duration, sample: Duration, dir: &Path) {
    if sample > Duration::from_millis(2) {
        eprintln!(
            "slow {label} put #{index} at {:.3} s: {:.1} ms ({} packs, {} sidecars)",
            at.as_secs_f64(),
            sample.as_secs_f64() * 1e3,
            count_ext(dir, "pack"),
            count_ext(dir, "idx")
        );
    }
}

/// Puts until the run is published plus `tail`. The first put starts the
/// merge. Returns the samples and when the run file appeared.
fn merge_pass(
    store: &PackfileStorage,
    dir: &Path,
    sequence: &mut u64,
    rate: u64,
    tail: Duration,
) -> (Vec<Duration>, Option<Duration>) {
    let mut samples = Vec::new();
    let mut pacer = Pacer::new(rate);
    let started = Instant::now();
    let mut published: Option<Duration> = None;
    loop {
        pacer.wait();
        let sample = put_one(store, *sequence);
        note_slow("merge", samples.len(), started.elapsed(), sample, dir);
        samples.push(sample);
        *sequence = sequence.saturating_add(1);
        if published.is_none() && samples.len() % 64 == 0 && !run_files(dir).is_empty() {
            published = Some(started.elapsed());
        }
        let done =
            published.is_some_and(|at| started.elapsed() >= at.checked_add(tail).unwrap_or(at));
        if done || started.elapsed() >= MERGE_PASS_LIMIT {
            return (samples, published);
        }
    }
}

/// `count` puts at `rate` with no merge in flight.
fn control_pass(
    store: &PackfileStorage,
    dir: &Path,
    sequence: &mut u64,
    count: usize,
    rate: u64,
) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(count);
    let mut pacer = Pacer::new(rate);
    let started = Instant::now();
    for _ in 0..count {
        pacer.wait();
        let sample = put_one(store, *sequence);
        note_slow("control", samples.len(), started.elapsed(), sample, dir);
        samples.push(sample);
        *sequence = sequence.saturating_add(1);
    }
    samples
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: paged_merge_put_latency <fixture-dir> <scratch-dir>";
    let fixture = PathBuf::from(args.next().expect(usage));
    let scratch = PathBuf::from(args.next().expect(usage));
    let rate = env_u64("MTXDB_PUT_RATE", 5_000);
    let tail = Duration::from_millis(env_u64("MTXDB_PUT_TAIL_MS", 1_000));
    assert!(
        tail > Duration::ZERO,
        "MTXDB_PUT_TAIL_MS must be positive so the put that installs the run is sampled"
    );
    let records = std::fs::read_to_string(fixture.join("fixture-records"))
        .ok()
        .and_then(|contents| contents.trim().parse().ok())
        .unwrap_or(1_u64 << 63);
    assert!(
        run_files(&fixture).is_empty(),
        "the fixture already has a run; this measures the first merge"
    );

    let copy_started = Instant::now();
    copy_fixture(&fixture, &scratch);
    eprintln!(
        "copied fixture in {:.1} s",
        copy_started.elapsed().as_secs_f64()
    );

    let open_started = Instant::now();
    let store = PackfileStorage::open_with_max_shard_bytes_and_residency(
        scratch.clone(),
        SHARD_BYTES,
        ChecksumPolicy::Full,
        IndexResidency::Paged,
    )
    .expect("open scratch copy");
    eprintln!(
        "opened Paged writable in {:.1} ms",
        open_started.elapsed().as_secs_f64() * 1e3
    );

    let mut sequence = records;
    let (merge, published) = merge_pass(&store, &scratch, &mut sequence, rate, tail);
    // A run file can appear before its worker reports completion and the
    // storage generation installs it. Settle all resulting merges before the
    // control pass so the two samples do not overlap installation.
    store.settle_run_merges();
    let runs = run_files(&scratch);
    // The control measures puts with no merge in flight, so it is only valid
    // once a merge has actually published a run.
    let (control, control_runs) = if published.is_some() {
        let control = control_pass(&store, &scratch, &mut sequence, merge.len(), rate);
        (control, run_files(&scratch))
    } else {
        (Vec::new(), Vec::new())
    };

    match published {
        Some(at) => println!(
            "run published {:.2} s after the first put (rate {rate}/s)",
            at.as_secs_f64()
        ),
        None => println!("no run published within {MERGE_PASS_LIMIT:?}"),
    }
    for path in &runs {
        let covered = match RunIndex::open(path) {
            Ok(run) => run.packs().len(),
            Err(error) => {
                eprintln!("could not open published run {}: {error}", path.display());
                0
            }
        };
        println!(
            "run {} covers {covered} packs",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }
    if published.is_some() && control_runs != runs {
        println!("warning: the run set changed during the control pass: {control_runs:?}");
    }
    println!("pass\tputs\tp50_us\tp99_us\tp999_us\tmax_us");
    report("merge", merge);
    if published.is_some() {
        report("control", control);
    }

    drop(store);
    if std::env::var_os("MTXDB_KEEP_SCRATCH").is_none() {
        std::fs::remove_dir_all(&scratch).expect("remove scratch copy");
    }
}
