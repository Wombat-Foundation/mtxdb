//! Shared-WAL replay paging and multi-collection snapshots under load.
//!
//! The first run measures incremental resume pages while four pool writers
//! publish and parallel readers catch up. The second repeats with shared WAL
//! reclaim enabled, exercising lease advancement and stale-offset fallback.
//! During each run the main thread times 1/2/4/8-collection snapshots against
//! the EventDag pool while its writer is active.
//!
//! Run with `cargo bench --manifest-path benches/Cargo.toml --bench replay_paging`.
//! Set `MTXDB_BENCH_ROOT` to place scratch files on a chosen device.
//!
//! Knobs: `MTXDB_RP_WRITERS_PER_POOL` (default 1), `MTXDB_RP_READERS` (2),
//! `MTXDB_RP_WRITES` per pool (5000), `MTXDB_RP_BATCH` (32), `MTXDB_RP_PAGE`
//! (64), `MTXDB_RP_SEED_GROUPS_PER_POOL` (1000), and `MTXDB_RP_PAYLOAD` (128).
#![allow(
    clippy::arithmetic_side_effects,
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::journal::{Journal, JournalCoordinator, JournalReplayLease};
use mtxdb::layout::ShardType;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const POOLS: [ShardType; 4] = [
    ShardType::Edges,
    ShardType::EventDag,
    ShardType::State,
    ShardType::ServerInfo,
];
const SNAPSHOT_COLLECTIONS: usize = 8;

struct ScratchDir(PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct BenchDb {
    stores: Vec<Arc<PackfileStorage>>,
    journal: Arc<JournalCoordinator>,
    collections: Vec<[u8; 16]>,
    _scratch: ScratchDir,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn node_id(writer: usize, sequence: usize) -> NodeId {
    let mut id = [0; 16];
    id[..8].copy_from_slice(&u64::try_from(writer).unwrap_or(u64::MAX).to_le_bytes());
    id[8..].copy_from_slice(&u64::try_from(sequence).unwrap_or(u64::MAX).to_le_bytes());
    id
}

fn collection_id(index: usize) -> [u8; 16] {
    let mut id = [0; 16];
    id[..8].copy_from_slice(&(index as u64).to_le_bytes());
    id
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let index = (samples.len() - 1).saturating_mul(percentile) / 100;
    samples[index]
}

fn per_second(count: u128, duration: Duration) -> u128 {
    count.saturating_mul(1_000_000_000) / duration.as_nanos().max(1)
}

fn setup(label: &str, seed_groups: usize, payload_size: usize, checkpoint_seed: bool) -> BenchDb {
    let parent =
        std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    fs::create_dir_all(&parent).expect("create benchmark root");
    let root = parent.join(format!(
        "mtxdb_replay_paging_{}_{}",
        std::process::id(),
        label
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("remove stale benchmark directory");
    }
    fs::create_dir_all(&root).expect("create benchmark directory");
    let scratch = ScratchDir(root.clone());

    let wal_path = root.join("journal.wal");
    let (journal, scan) = Journal::open_shared(&wal_path).expect("open shared journal");
    let journal = Arc::new(JournalCoordinator::new(journal, &scan));
    let stores: Vec<_> = POOLS
        .iter()
        .enumerate()
        .map(|(index, pool)| {
            let store = Arc::new(
                PackfileStorage::open(root.join(format!("pool-{index}")))
                    .expect("open packfile pool"),
            );
            store
                .enable_shared_journal(Arc::clone(&journal), *pool)
                .expect("attach shared journal");
            store
        })
        .collect();
    let collections: Vec<_> = (0..SNAPSHOT_COLLECTIONS).map(collection_id).collect();
    let payload = NodeData::new(Bytes::from(vec![0xA7; payload_size]));
    for (pool_index, store) in stores.iter().enumerate() {
        for sequence in 0..seed_groups {
            let collection = if pool_index == 1 {
                collections[sequence % collections.len()]
            } else {
                collection_id(100 + pool_index)
            };
            store
                .put(&collection, &node_id(pool_index, sequence), &payload)
                .expect("seed pack and journal");
        }
    }
    if checkpoint_seed {
        for store in &stores {
            store.sync_all().expect("sync seeded pool");
        }
    } else {
        // Flush pack bytes and make the WAL durable without writing checkpoint
        // coverage. The fallback arm takes a cursor next, then checkpoints
        // under its lease so reclaim can advance exactly through that cursor.
        for store in &stores {
            store.flush_all().expect("flush seeded pack pool");
        }
        journal.sync().expect("sync seeded journal");
    }
    BenchDb {
        stores,
        journal,
        collections,
        _scratch: scratch,
    }
}

fn run_case(label: &str, with_reclaim: bool) -> Result<(), Box<dyn Error>> {
    let writers_per_pool = env_usize("MTXDB_RP_WRITERS_PER_POOL", 1);
    let readers = env_usize("MTXDB_RP_READERS", 2);
    let writes = env_usize("MTXDB_RP_WRITES", 5_000);
    let batch = env_usize("MTXDB_RP_BATCH", 32);
    let page_size = env_usize("MTXDB_RP_PAGE", 64);
    let seed_groups = env_usize("MTXDB_RP_SEED_GROUPS_PER_POOL", 1_000);
    let payload_size = env_usize("MTXDB_RP_PAYLOAD", 128);
    // Keep the seeded WAL retained until the initial replay lease is installed.
    // Checkpointing every pool here can reclaim the shared prefix past the
    // EventDag pool's older per-pool coverage watermark.
    let db = setup(label, seed_groups, payload_size, false);

    // Capture all eight collections together; the resulting cursor is the
    // common replay start for every reader thread.
    let (scans, cursor, initial_lease) = db.stores[1]
        .scan_collections_at_snapshot(&db.collections)
        .expect("take initial multi-collection snapshot");
    let initial_records: usize = scans
        .into_iter()
        .map(|(_, scan)| scan.map(Result::unwrap).count())
        .sum();
    let stats_before = db.journal.replay_page_stats();

    let worker_count = POOLS.len() * writers_per_pool;
    let reclaimer_count = usize::from(with_reclaim);
    let barrier = Arc::new(Barrier::new(worker_count + readers + reclaimer_count + 1));
    let writers_done = Arc::new(AtomicUsize::new(0));
    let page_latencies = Arc::new(Mutex::new(Vec::<Duration>::new()));
    let replayed_groups = Arc::new(AtomicUsize::new(0));
    let replayed_entries = Arc::new(AtomicUsize::new(0));
    let replayed_bytes = Arc::new(AtomicU64::new(0));

    let mut writer_handles = Vec::with_capacity(worker_count);
    let mut reader_handles = Vec::with_capacity(readers);
    for pool_index in 0..POOLS.len() {
        for writer_index in 0..writers_per_pool {
            let store = Arc::clone(&db.stores[pool_index]);
            let barrier = Arc::clone(&barrier);
            let done = Arc::clone(&writers_done);
            let collections = db.collections.clone();
            writer_handles.push(std::thread::spawn(move || {
                barrier.wait();
                let pool_tag = u8::try_from(pool_index).unwrap_or(u8::MAX);
                let payload = NodeData::new(Bytes::from(vec![pool_tag; payload_size]));
                let started = Instant::now();
                for sequence in 0..writes {
                    let collection = if pool_index == 1 {
                        collections[sequence % collections.len()]
                    } else {
                        collection_id(100 + pool_index)
                    };
                    store
                        .put(
                            &collection,
                            &node_id(pool_index * writers_per_pool + writer_index + 10, sequence),
                            &payload,
                        )
                        .expect("publish concurrent write");
                    if (sequence + 1) % batch == 0 {
                        if with_reclaim {
                            store.sync_all().expect("checkpoint writer batch");
                        } else {
                            store
                                .journal()
                                .expect("journal")
                                .sync()
                                .expect("sync writer batch");
                        }
                    }
                }
                if with_reclaim {
                    store.sync_all().expect("checkpoint final writer batch");
                } else {
                    store
                        .journal()
                        .expect("journal")
                        .sync()
                        .expect("sync final writer batch");
                }
                done.fetch_add(1, Ordering::Release);
                started.elapsed()
            }));
        }
    }

    // Each reader independently pins and drains the same snapshot, modelling
    // concurrent rebuild consumers and their independent retention floors.
    let mut initial_lease = Some(initial_lease);
    for reader_index in 0..readers {
        let journal = Arc::clone(&db.journal);
        let (reader_cursor, lease): (_, JournalReplayLease) = if reader_index == 0 {
            (
                cursor.clone(),
                initial_lease.take().expect("initial snapshot lease"),
            )
        } else {
            let (scans, cursor, lease) = db.stores[1]
                .scan_collections_at_snapshot(&db.collections)
                .expect("take reader snapshot");
            drop(scans);
            (cursor, lease)
        };
        let barrier = Arc::clone(&barrier);
        let done = Arc::clone(&writers_done);
        let latencies = Arc::clone(&page_latencies);
        let groups = Arc::clone(&replayed_groups);
        let entries = Arc::clone(&replayed_entries);
        let bytes = Arc::clone(&replayed_bytes);
        reader_handles.push(std::thread::spawn(move || {
            let _lease = lease;
            let mut cursor = reader_cursor;
            barrier.wait();
            let replay_started = Instant::now();
            let mut local_pages = Vec::new();
            loop {
                let done_before_read = done.load(Ordering::Acquire);
                let started = Instant::now();
                let page = journal
                    .changes_since(&cursor, page_size)
                    .expect("read replay page");
                if !page.groups.is_empty() {
                    local_pages.push(started.elapsed());
                    groups.fetch_add(page.groups.len(), Ordering::Relaxed);
                    entries.fetch_add(
                        page.groups
                            .iter()
                            .map(|group| group.entries.len())
                            .sum::<usize>(),
                        Ordering::Relaxed,
                    );
                    bytes.fetch_add(
                        page.groups
                            .iter()
                            .flat_map(|group| group.entries.iter())
                            .map(|entry| entry.frame_len)
                            .sum::<u64>(),
                        Ordering::Relaxed,
                    );
                    _lease.advance(page.through_lsn);
                    cursor = page.next_cursor;
                } else if done_before_read == worker_count {
                    break;
                } else {
                    std::thread::yield_now();
                }
            }
            latencies
                .lock()
                .expect("latency samples")
                .extend(local_pages);
            replay_started.elapsed()
        }));
    }

    let reclaimer = if with_reclaim {
        let journal = Arc::clone(&db.journal);
        let barrier = Arc::clone(&barrier);
        let done = Arc::clone(&writers_done);
        Some(std::thread::spawn(move || {
            barrier.wait();
            while done.load(Ordering::Acquire) != worker_count {
                journal.reclaim_shared().expect("shared WAL reclaim");
                std::thread::sleep(Duration::from_millis(2));
            }
            journal.reclaim_shared().expect("final shared WAL reclaim");
        }))
    } else {
        None
    };

    barrier.wait();

    // Vary the number of locked collections while the pool's writers are live.
    let mut snapshot_samples = Vec::new();
    for width in [1, 2, 4, 8] {
        let started = Instant::now();
        let (scans, _, lease) = db.stores[1]
            .scan_collections_at_snapshot(&db.collections[..width])
            .expect("take concurrent multi-collection snapshot");
        let count: usize = scans
            .into_iter()
            .map(|(_, scan)| scan.map(Result::unwrap).count())
            .sum();
        snapshot_samples.push((width, started.elapsed(), count));
        drop(lease);
    }

    let mut writer_max = Duration::ZERO;
    for handle in writer_handles {
        writer_max = writer_max.max(handle.join().expect("writer worker"));
    }
    let mut replay_elapsed = Duration::ZERO;
    for handle in reader_handles {
        replay_elapsed = replay_elapsed.max(handle.join().expect("reader worker"));
    }
    if let Some(handle) = reclaimer {
        handle.join().expect("reclaimer worker");
    }

    let mut page_latencies = page_latencies.lock().expect("latency samples").clone();
    page_latencies.sort_unstable();
    let stats_after = db.journal.replay_page_stats();
    let resumed = stats_after
        .resumed_pages
        .saturating_sub(stats_before.resumed_pages);
    let fallbacks = stats_after
        .full_scan_fallbacks
        .saturating_sub(stats_before.full_scan_fallbacks);
    let written = worker_count * writes;

    println!(
        "case={label} reclaim={with_reclaim} pools={} writers={writers_per_pool}/pool readers={readers} writes/pool={writes} batch={batch} page={page_size} seed/pool={seed_groups} payload={payload_size}",
        POOLS.len()
    );
    println!("initial_snapshot_records={initial_records}");
    for (width, elapsed, count) in snapshot_samples {
        println!(
            "snapshot_collections={width} records={count} latency_ms={:.3}",
            elapsed.as_secs_f64() * 1e3
        );
    }
    println!(
        "writer_max_ms={:.3} writes/s={} replayed_groups={} replayed_entries={} replayed_frame_bytes={}",
        writer_max.as_secs_f64() * 1e3,
        per_second(u128::try_from(written).unwrap_or(u128::MAX), writer_max),
        replayed_groups.load(Ordering::Relaxed),
        replayed_entries.load(Ordering::Relaxed),
        replayed_bytes.load(Ordering::Relaxed)
    );
    let replay_groups =
        u128::try_from(replayed_groups.load(Ordering::Relaxed)).unwrap_or(u128::MAX);
    let replay_bytes = u128::from(replayed_bytes.load(Ordering::Relaxed));
    println!(
        "replay_throughput groups/s={} frame_bytes/s={}",
        per_second(replay_groups, replay_elapsed),
        per_second(replay_bytes, replay_elapsed)
    );
    println!("replay_path resumed_pages={resumed} full_scan_fallbacks={fallbacks}");
    if !page_latencies.is_empty() {
        println!(
            "page_latency_ms p50={:.3} p95={:.3} p99={:.3} samples={}",
            percentile(&page_latencies, 50).as_secs_f64() * 1e3,
            percentile(&page_latencies, 95).as_secs_f64() * 1e3,
            percentile(&page_latencies, 99).as_secs_f64() * 1e3,
            page_latencies.len()
        );
    }
    Ok(())
}

/// Force a cursor-offset fallback in an isolated database: report coverage
/// through the group before the cursor, reclaim that prefix, append one durable
/// group, then page from the cursor's stale offset at a still-valid LSN.
fn run_reclaim_fallback_arm() -> Result<(), Box<dyn Error>> {
    let payload_size = env_usize("MTXDB_RP_PAYLOAD", 128);
    let seed_groups = env_usize("MTXDB_RP_FALLBACK_SEED_GROUPS", 4_096);
    let db = setup("forced_fallback", seed_groups, payload_size, false);
    // ServerInfo is seeded last, so its pool watermark reaches the shared WAL
    // tail. Using EventDag here would leave later pools' seed groups after the
    // cursor and make the fallback page contain more than the test group.
    let (scans, cursor, lease) = db.stores[3]
        .scan_collections_at_snapshot(&db.collections)
        .expect("take fallback-arm snapshot");
    drop(scans);

    let covered_lsn = cursor
        .lsn()
        .checked_sub(1)
        .ok_or("fallback benchmark needs a nonzero cursor LSN")?;
    for pool in POOLS {
        db.journal.report_pool_coverage(pool, covered_lsn);
    }
    let reclaimed = db
        .journal
        .reclaim_shared()?
        .ok_or("fallback benchmark failed to reclaim the covered seed prefix")?;
    if reclaimed.reclaimed_bytes == 0 {
        return Err("fallback benchmark reclaim removed no WAL bytes".into());
    }
    let retained = Journal::scan_read_only(db.journal.path())?;
    if retained.base_lsn != cursor.lsn() {
        return Err(format!(
            "expected reclaim base {} to preserve cursor {}, got {}",
            cursor.lsn(),
            cursor.lsn(),
            retained.base_lsn
        )
        .into());
    }
    drop(lease);

    db.stores[1].put(
        &db.collections[0],
        &node_id(999, 0),
        &NodeData::new(Bytes::from(vec![0xD3; payload_size])),
    )?;
    db.journal.sync()?;

    let before = db.journal.replay_page_stats();
    let started = Instant::now();
    let page = db.journal.changes_since(&cursor, 16)?;
    let elapsed = started.elapsed();
    let after = db.journal.replay_page_stats();
    let full_scan_fallbacks = after
        .full_scan_fallbacks
        .saturating_sub(before.full_scan_fallbacks);
    if full_scan_fallbacks != 1 || page.groups.len() != 1 {
        return Err(format!(
            "expected one reclaimed-offset full-scan fallback returning one group; got fallbacks={full_scan_fallbacks}, groups={}",
            page.groups.len()
        )
        .into());
    }
    println!(
        "case=forced_fallback groups={} latency_ms={:.3} full_scan_fallbacks={} resumed_pages={}",
        page.groups.len(),
        elapsed.as_secs_f64() * 1e3,
        full_scan_fallbacks,
        after.resumed_pages.saturating_sub(before.resumed_pages)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_case("steady", false)?;
    run_case("reclaim", true)?;
    run_reclaim_fallback_arm()?;
    Ok(())
}
