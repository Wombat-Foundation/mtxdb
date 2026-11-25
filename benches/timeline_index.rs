//! Scaling harness for the prototype ordered timeline sidecar.
//!
//! Run with `cargo bench --manifest-path benches/Cargo.toml --bench timeline_index`.
//! `MTXDB_TIMELINE_SIZES` defaults to `10000,100000,1000000` and
//! `MTXDB_TIMELINE_BATCH` defaults to 4096. Set `MTXDB_BENCH_ROOT` to place
//! scratch data on the device under evaluation. Index appends use the
//! single-writer, untransactional API, so transaction/journal overhead is not
//! included. Set `MTXDB_TIMELINE_SCAN_BASELINE=1` to time five scan-plus-room-filter
//! passes per dataset; the sorted materialization used as the equality oracle is
//! not part of that number. Small tail appends after the bulk load expose the
//! marginal root-manifest publish cost that large setup batches hide.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::pedantic
)]

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use bytes::Bytes;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::timeline::{TimelineEntry, TimelineIndex, TIMELINE_LEAF_CAP};
use mtxdb::PackfileStorage;

const EVENT_COLLECTION: [u8; 16] = *b"timeline-events!";
const ROOM: [u8; 16] = *b"timeline-room-01";
const PAYLOAD_BYTES: usize = 512;

fn node_id(n: u64) -> NodeId {
    let mut id = [0; 16];
    id[..8].copy_from_slice(&n.to_be_bytes());
    id[8..].copy_from_slice(&n.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_be_bytes());
    id
}

fn entry(n: u64) -> TimelineEntry {
    let id = node_id(n);
    let mut reference = [0; 32];
    reference[..16].copy_from_slice(&id);
    reference[16..].copy_from_slice(&n.wrapping_mul(17).to_be_bytes().repeat(2));
    TimelineEntry::new(ROOM, n as i64 - 500_000, n as i64 - 500_000, id, reference)
}

fn payload(n: u64) -> Bytes {
    let mut bytes = vec![(n % 251) as u8; PAYLOAD_BYTES];
    bytes[..16].copy_from_slice(&ROOM);
    let ordering = (n as i64 - 500_000).to_be_bytes();
    bytes[16..24].copy_from_slice(&ordering);
    bytes[24..32].copy_from_slice(&ordering);
    Bytes::from(bytes)
}

fn front_entry(offset: u64) -> TimelineEntry {
    let offset_i64 = i64::try_from(offset).expect("front offset fits i64");
    let position = -500_001_i64
        .checked_sub(offset_i64)
        .expect("front position fits i64");
    let sequence = u64::MAX
        .checked_sub(offset)
        .expect("front node id sequence fits u64");
    let id = node_id(sequence);
    let mut reference = [0; 32];
    reference[..16].copy_from_slice(&id);
    reference[16..24].copy_from_slice(&offset.to_be_bytes());
    TimelineEntry::new(ROOM, position, position, id, reference)
}

fn front_payload(offset: u64) -> Bytes {
    let fill = u8::try_from(offset.checked_rem(251).expect("nonzero payload modulus"))
        .expect("payload fill fits u8");
    let mut bytes = vec![fill; PAYLOAD_BYTES];
    bytes[..16].copy_from_slice(&ROOM);
    let offset_i64 = i64::try_from(offset).expect("front offset fits i64");
    let position = -500_001_i64
        .checked_sub(offset_i64)
        .expect("front position fits i64");
    let ordering = position.to_be_bytes();
    bytes[16..24].copy_from_slice(&ordering);
    bytes[24..32].copy_from_slice(&ordering);
    Bytes::from(bytes)
}

fn sample_ns(iterations: usize, mut operation: impl FnMut()) -> (u128, u128) {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    (samples[0], samples[samples.len() / 2])
}

fn scan_collection_entries(store: &PackfileStorage) -> Vec<TimelineEntry> {
    let scan = store
        .scan_collection(&EVENT_COLLECTION)
        .expect("start event collection scan");
    let mut entries = Vec::new();
    for record in scan {
        let (id, data) = record.expect("read scanned event");
        if data.bytes.len() < 32 || data.bytes[..16] != ROOM {
            continue;
        }
        let mut sequence = [0; 8];
        sequence.copy_from_slice(&id[..8]);
        entries.push(entry(u64::from_be_bytes(sequence)));
    }
    entries
}

fn scan_collection_room_filter_count(store: &PackfileStorage) -> usize {
    store
        .scan_collection(&EVENT_COLLECTION)
        .expect("start event collection scan")
        .filter_map(|record| {
            let (_, data) = record.expect("read scanned event");
            (data.bytes.len() >= 32 && data.bytes[..16] == ROOM).then_some(())
        })
        .count()
}

/// Full materialization plus sort, used only as a correctness oracle for the
/// midpoint page. It is deliberately not part of the timed scan baseline: the
/// sort is O(N log N) work that the scan path does not necessarily perform.
fn scan_collection_page(
    store: &PackfileStorage,
    cursor: TimelineEntry,
    limit: usize,
) -> Vec<TimelineEntry> {
    let mut entries = scan_collection_entries(store);
    entries.sort_unstable();
    let start = entries.partition_point(|candidate| *candidate <= cursor);
    entries
        .get(start..start.saturating_add(limit).min(entries.len()))
        .unwrap_or_default()
        .to_vec()
}

fn root() -> PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

fn sizes() -> Vec<usize> {
    std::env::var("MTXDB_TIMELINE_SIZES")
        .unwrap_or_else(|_| "10000,100000,1000000".to_owned())
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .filter(|n| *n > 0)
        .collect()
}

fn run(n: usize, batch: usize) {
    let dir = root().join(format!("mtxdb_timeline_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    let store = PackfileStorage::open(dir.clone()).expect("open store");
    let index = TimelineIndex::open(&store, "bench-room-timeline");
    let mut append_us = 0u128;
    let mut event_write_us = 0u128;

    for start in (0..n).step_by(batch) {
        let end = (start + batch).min(n);
        let entries: Vec<_> = (start..end).map(|i| entry(i as u64)).collect();
        let records: Vec<_> = (start..end)
            .map(|i| (node_id(i as u64), NodeData::new(payload(i as u64))))
            .collect();

        let t = Instant::now();
        store
            .put_many(&EVENT_COLLECTION, &records)
            .expect("write event payloads");
        event_write_us += t.elapsed().as_micros();

        let t = Instant::now();
        index
            .append_untransactional(&entries)
            .expect("append timeline entries");
        append_us += t.elapsed().as_micros();
    }
    store.sync_all().expect("sync store");

    let root_id = index
        .head()
        .expect("read index head")
        .expect("index root")
        .0;
    let middle_seed = index
        .cursor(root_id, entry((n / 2).saturating_sub(1) as u64))
        .expect("valid timeline cursor geometry");
    let middle_cursor = index
        .page(&ROOM, Some(&middle_seed), true, 1)
        .expect("get midpoint cursor from a preceding page")
        .next_cursor
        .unwrap_or(middle_seed);
    let mut fetch_count = 0;
    for page_size in [50, 128, 512] {
        for (start_name, start_cursor) in [("first", None), ("middle", Some(middle_cursor))] {
            for forward in [true, false] {
                let label = if forward { "forward" } else { "reverse" };
                let (min_ns, median_ns) = sample_ns(20, || {
                    black_box(
                        index
                            .page(&ROOM, start_cursor.as_ref(), forward, page_size)
                            .expect("timeline page"),
                    );
                });
                let first = index
                    .page(&ROOM, start_cursor.as_ref(), forward, page_size)
                    .expect("representative timeline page");
                let (resume_min_ns, resume_median_ns) = if let Some(cursor) = first.next_cursor {
                    sample_ns(20, || {
                        black_box(
                            index
                                .page(&ROOM, Some(&cursor), forward, page_size)
                                .expect("resume timeline page"),
                        );
                    })
                } else {
                    (0, 0)
                };
                println!(
                    "timeline_page n={n} start={start_name} direction={label} limit={page_size} min_ns={min_ns} median_ns={median_ns} resume_min_ns={resume_min_ns} resume_median_ns={resume_median_ns}"
                );
                if page_size == 128 && start_name == "first" && forward {
                    let ids: Vec<_> = first.entries.iter().map(|e| e.id).collect();
                    let (fetch_min_ns, fetch_median_ns) = sample_ns(20, || {
                        black_box(
                            store
                                .get_many(&EVENT_COLLECTION, &ids)
                                .expect("fetch selected event payloads"),
                        );
                    });
                    fetch_count = first.entries.len();
                    println!(
                        "timeline_fetch n={n} count={fetch_count} min_ns={fetch_min_ns} median_ns={fetch_median_ns}"
                    );
                }
            }
        }
    }

    // A key below every stored entry makes reverse_start return None only after
    // the root manifest has been fetched and decoded, so this prices the O(L)
    // manifest step without reading any leaf.
    let below = TimelineEntry::new(ROOM, i64::MIN, i64::MIN, [0u8; 16], [0u8; 32]);
    let below_cursor = index
        .cursor(root_id, below)
        .expect("valid timeline cursor geometry");
    let (root_min_ns, root_median_ns) = sample_ns(20, || {
        black_box(
            index
                .page(&ROOM, Some(&below_cursor), false, 1)
                .expect("root-only timeline page"),
        );
    });
    println!(
        "timeline_root_only n={n} min_ns={root_min_ns} median_ns={root_median_ns} note=manifest_fetch_and_decode_no_leaf_reads"
    );

    if std::env::var("MTXDB_TIMELINE_SCAN_BASELINE").is_ok_and(|value| value == "1") {
        if n <= 100_000 {
            let index_page = index
                .page(&ROOM, Some(&middle_cursor), true, 128)
                .expect("midpoint index page");
            let scan_page = scan_collection_page(&store, middle_cursor.entry(), 128);
            assert_eq!(
                scan_page, index_page.entries,
                "scan correctness oracle and timeline index should return the same midpoint page"
            );
        } else {
            println!("timeline_scan_correctness_oracle n={n} skipped=over_100000");
        }
        let (scan_min_ns, scan_median_ns) = sample_ns(5, || {
            black_box(scan_collection_room_filter_count(&store));
        });
        println!(
            "timeline_scan_filter_baseline n={n} samples=5 min_ns={scan_min_ns} median_ns={scan_median_ns} matched_room_entries={n} note=full_collection_scan_plus_room_filter_no_order_or_materialization"
        );
    }

    // Reopen the storage handle and repeat the first-page read. This measures
    // a fresh engine handle; the OS page cache is uncontrolled and may be warm.
    drop(index);
    drop(store);
    let reopened = PackfileStorage::open(dir.clone()).expect("reopen store");
    let reopened_index = TimelineIndex::open(&reopened, "bench-room-timeline");
    let t = Instant::now();
    black_box(
        reopened_index
            .page(&ROOM, None, true, 128)
            .expect("first page after reopen"),
    );
    let reopen_first_ns = t.elapsed().as_nanos();
    let (reopen_min_ns, reopen_median_ns) = sample_ns(20, || {
        black_box(
            reopened_index
                .page(&ROOM, None, true, 128)
                .expect("page after reopen"),
        );
    });
    println!(
        "timeline_summary n={n} batch={batch} event_write_plus_primary_index_us={event_write_us} index_append_untransactional_us={append_us} reopened_handle_first_page_ns={reopen_first_ns} reopened_handle_warm_page_min_ns={reopen_min_ns} reopened_handle_warm_page_median_ns={reopen_median_ns}"
    );
    assert_eq!(
        fetch_count,
        128.min(n),
        "forward page should contain all records selected for fetch"
    );

    // Measure marginal publish cost at the populated tree size, after all
    // read metrics above have run against exactly n entries. Payload writes
    // happen first and outside the timer; each timed call publishes only the
    // corresponding timeline entry, including its leaf/root/head writes.
    const SINGLE_TAIL_WRITES: usize = 20;
    const BATCHED_TAIL_WRITES: usize = 20;
    const TAIL_BATCH_SIZE: usize = 16;
    let tail_total = SINGLE_TAIL_WRITES + BATCHED_TAIL_WRITES * TAIL_BATCH_SIZE;
    let tail_records: Vec<_> = (0..tail_total)
        .map(|offset| {
            let sequence = n as u64 + offset as u64;
            (node_id(sequence), NodeData::new(payload(sequence)))
        })
        .collect();
    let tail_seed_start = Instant::now();
    reopened
        .put_many(&EVENT_COLLECTION, &tail_records)
        .expect("seed tail-append payloads");
    let tail_payload_seed_us = tail_seed_start.elapsed().as_micros();

    let mut single_samples = Vec::with_capacity(SINGLE_TAIL_WRITES);
    let mut single_first_ns = 0_u128;
    for offset in 0..SINGLE_TAIL_WRITES {
        let entry = entry(n as u64 + offset as u64);
        let start = Instant::now();
        reopened_index
            .append_untransactional(std::slice::from_ref(&entry))
            .expect("single-entry tail append");
        let elapsed = start.elapsed().as_nanos();
        if offset == 0 {
            single_first_ns = elapsed;
        }
        single_samples.push(elapsed);
    }
    single_samples.sort_unstable();
    let single_max_ns = *single_samples.last().expect("tail single samples");

    let mut batch_samples = Vec::with_capacity(BATCHED_TAIL_WRITES);
    let mut batch_first_ns = 0_u128;
    for batch_index in 0..BATCHED_TAIL_WRITES {
        let start_sequence =
            n as u64 + SINGLE_TAIL_WRITES as u64 + (batch_index * TAIL_BATCH_SIZE) as u64;
        let batch_entries: Vec<_> = (0..TAIL_BATCH_SIZE)
            .map(|offset| entry(start_sequence + offset as u64))
            .collect();
        let start = Instant::now();
        reopened_index
            .append_untransactional(&batch_entries)
            .expect("16-entry tail append");
        let elapsed = start.elapsed().as_nanos();
        if batch_index == 0 {
            batch_first_ns = elapsed;
        }
        batch_samples.push(elapsed);
    }
    batch_samples.sort_unstable();
    let batch_max_ns = *batch_samples.last().expect("tail batch samples");
    println!(
        "timeline_tail_publish n={n} leaves_before~{} payload_seed_records={tail_total} payload_seed_us={tail_payload_seed_us} single_entries={} single_first_ns={single_first_ns} single_min_ns={} single_median_ns={} single_max_ns={single_max_ns} batch_entries={} batch_count={} batch_first_ns={batch_first_ns} batch_min_ns={} batch_median_ns={} batch_max_ns={batch_max_ns}",
        n.div_ceil(TIMELINE_LEAF_CAP),
        SINGLE_TAIL_WRITES,
        single_samples[0],
        single_samples[single_samples.len() / 2],
        TAIL_BATCH_SIZE,
        BATCHED_TAIL_WRITES,
        batch_samples[0],
        batch_samples[batch_samples.len() / 2],
    );

    let front_records: Vec<_> = (0..tail_total)
        .map(|offset| {
            let offset = u64::try_from(offset).expect("front offset fits u64");
            let sequence = u64::MAX
                .checked_sub(offset)
                .expect("front node id sequence fits u64");
            (node_id(sequence), NodeData::new(front_payload(offset)))
        })
        .collect();
    let front_seed_start = Instant::now();
    reopened
        .put_many(&EVENT_COLLECTION, &front_records)
        .expect("seed front-backfill payloads");
    let front_payload_seed_us = front_seed_start.elapsed().as_micros();

    let mut front_single_samples = Vec::with_capacity(SINGLE_TAIL_WRITES);
    let mut front_single_first_ns = 0_u128;
    for offset in 0..SINGLE_TAIL_WRITES {
        let offset = u64::try_from(offset).expect("front offset fits u64");
        let entry = front_entry(offset);
        let start = Instant::now();
        reopened_index
            .append_untransactional(std::slice::from_ref(&entry))
            .expect("single-entry front append");
        let elapsed = start.elapsed().as_nanos();
        if offset == 0 {
            front_single_first_ns = elapsed;
        }
        front_single_samples.push(elapsed);
    }
    front_single_samples.sort_unstable();
    let front_single_max_ns = *front_single_samples.last().expect("front single samples");

    let mut front_batch_samples = Vec::with_capacity(BATCHED_TAIL_WRITES);
    let mut front_batch_first_ns = 0_u128;
    for batch_index in 0..BATCHED_TAIL_WRITES {
        let start_offset = SINGLE_TAIL_WRITES
            .checked_add(
                batch_index
                    .checked_mul(TAIL_BATCH_SIZE)
                    .expect("front batch offset fits usize"),
            )
            .expect("front batch offset fits usize");
        let batch_entries: Vec<_> = (0..TAIL_BATCH_SIZE)
            .map(|offset| {
                let offset = start_offset
                    .checked_add(offset)
                    .and_then(|value| u64::try_from(value).ok())
                    .expect("front batch entry offset fits u64");
                front_entry(offset)
            })
            .collect();
        let start = Instant::now();
        reopened_index
            .append_untransactional(&batch_entries)
            .expect("16-entry front append");
        let elapsed = start.elapsed().as_nanos();
        if batch_index == 0 {
            front_batch_first_ns = elapsed;
        }
        front_batch_samples.push(elapsed);
    }
    front_batch_samples.sort_unstable();
    let front_batch_max_ns = *front_batch_samples.last().expect("front batch samples");
    println!(
        "timeline_front_publish n={n} leaves_before~{} payload_seed_records={tail_total} payload_seed_us={front_payload_seed_us} single_entries={} single_first_ns={front_single_first_ns} single_min_ns={} single_median_ns={} single_max_ns={front_single_max_ns} batch_entries={} batch_count={} batch_first_ns={front_batch_first_ns} batch_min_ns={} batch_median_ns={} batch_max_ns={front_batch_max_ns}",
        n.div_ceil(TIMELINE_LEAF_CAP),
        SINGLE_TAIL_WRITES,
        front_single_samples[0],
        front_single_samples[front_single_samples.len() / 2],
        TAIL_BATCH_SIZE,
        BATCHED_TAIL_WRITES,
        front_batch_samples[0],
        front_batch_samples[front_batch_samples.len() / 2],
    );
    reopened.sync_all().expect("sync timeline publishes");
    drop(reopened_index);
    drop(reopened);
    let _ = std::fs::remove_dir_all(dir);
}

fn main() {
    let batch = std::env::var("MTXDB_TIMELINE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(4096);
    for n in sizes() {
        run(n, batch);
    }
}
