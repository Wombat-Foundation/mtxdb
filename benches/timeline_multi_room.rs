//! Multi-room cold-cache harness for the shared Matrix event-order indexes.
//!
//! Run with:
//! `cargo bench --manifest-path benches/Cargo.toml --bench timeline_multi_room`
//!
//! Defaults to a deterministic Zipfian synthetic corpus
//! (`MTXDB_MULTI_EVENTS=100000`, `MTXDB_MULTI_ROOMS=100`). Set
//! `MTXDB_TIMELINE_EXPORT` to a CSV length manifest with
//! `room,topo,stream,payload_bytes` rows to drive the same reads with real
//! event lengths. Every cold sample closes the handle, syncs, evicts the
//! selected files through `vmtouch -e`, and reopens before timing.
//! `MTXDB_TIMELINE_SAMPLES` controls cold repetitions.
//!
//! Output rows separate index traversal (`multi_room_page`), locator plus body
//! fetches (`multi_room_fetch`), and mutations (`multi_room_mutate`). I/O
//! counters come from `/proc/self/io`, `/proc/self/stat`, and cgroup v2
//! `memory.stat`; they are process-wide deltas around each measurement.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::pedantic
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::ordered_timeline::{
    open_event_timeline_indexes, page_stream_up_to, stage_event_timeline_batch,
    stage_purge_room_timeline, stage_remove_event_timeline_batch, EventTimelineEntry,
};
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::{interop, Database, ShardType};

const PAYLOAD_MIN: usize = 256;
const PAYLOAD_STEP: usize = 240;
const PAYLOAD_MAX: usize = 4096;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn bench_root() -> PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mtxdb-timeline-multi-room"))
}

fn node_id(sequence: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&sequence.to_be_bytes());
    id[8..].copy_from_slice(&sequence.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_be_bytes());
    id
}

fn event_ref(node: NodeId) -> [u8; 32] {
    let mut reference = [0u8; 32];
    reference[..16].copy_from_slice(&node);
    reference[16..].copy_from_slice(&node);
    reference
}

fn payload(sequence: u64, len: usize) -> Bytes {
    let fill = u8::try_from(sequence % 251).unwrap_or(0);
    let mut bytes = vec![fill; len];
    let header = node_id(sequence);
    let copy = header.len().min(bytes.len());
    bytes[..copy].copy_from_slice(&header[..copy]);
    Bytes::from(bytes)
}

struct Room {
    id: Vec<u8>,
    scope: [u8; 16],
    collection: [u8; 16],
    next_stream: i64,
    max_stream: i64,
}

impl Room {
    fn new(id: Vec<u8>) -> Self {
        Self {
            scope: interop::room_scope_id(&id),
            collection: interop::room_event_collection_id(&id),
            id,
            next_stream: 1,
            max_stream: 0,
        }
    }
}

struct SpecEvent {
    room: usize,
    topo: i64,
    stream: i64,
    bytes: usize,
}

fn synthetic_rooms(rooms: usize, events: usize) -> (Vec<Room>, Vec<usize>) {
    let weights: Vec<f64> = (0..rooms)
        .map(|index| 1.0 / ((index.saturating_add(1)) as f64).powf(1.1))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut counts: Vec<usize> = weights
        .iter()
        .map(|weight| ((events as f64 * weight / total).round() as usize).max(1))
        .collect();
    let assigned: usize = counts.iter().sum();
    if assigned != events {
        let largest = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .map(|(index, _)| index)
            .unwrap_or(0);
        if assigned < events {
            counts[largest] = counts[largest].saturating_add(events - assigned);
        } else {
            counts[largest] = counts[largest].saturating_sub(assigned - events);
        }
    }
    let rooms = (0..rooms)
        .map(|index| Room::new(format!("room-{index}").into_bytes()))
        .collect();
    (rooms, counts)
}

fn manifest_rooms(path: &Path) -> Result<(Vec<Room>, Vec<SpecEvent>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut rooms: Vec<Room> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut events = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 4 {
            return Err(format!(
                "{}:{}: expected room,topo,stream,payload_bytes",
                path.display(),
                line_number.saturating_add(1)
            ));
        }
        let topo = fields[1]
            .parse::<i64>()
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1))?;
        let stream = fields[2]
            .parse::<i64>()
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1))?;
        let bytes = fields[3]
            .parse::<usize>()
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_number + 1))?;
        if bytes == 0 {
            return Err(format!(
                "{}:{}: payload_bytes must be non-zero",
                path.display(),
                line_number.saturating_add(1)
            ));
        }
        let room_index = *index.entry(fields[0].to_owned()).or_insert_with(|| {
            rooms.push(Room::new(fields[0].as_bytes().to_vec()));
            rooms.len().saturating_sub(1)
        });
        if let Some(room) = rooms.get_mut(room_index) {
            room.max_stream = room.max_stream.max(stream);
        }
        events.push(SpecEvent {
            room: room_index,
            topo,
            stream,
            bytes,
        });
    }
    if events.is_empty() {
        return Err(format!("{}: no events", path.display()));
    }
    Ok((rooms, events))
}

#[derive(Default, Clone, Copy)]
struct IoSnapshot {
    syscr: u64,
    rchar: u64,
    read_bytes: u64,
    majflt: u64,
    pgmajfault: Option<u64>,
}

impl IoSnapshot {
    fn read() -> Self {
        let mut snapshot = Self::default();
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            for line in io.lines() {
                let mut parts = line.split_whitespace();
                let key = parts.next().unwrap_or_default();
                let value = parts.next().and_then(|value| value.parse().ok());
                match key {
                    "syscr:" => snapshot.syscr = value.unwrap_or(0),
                    "rchar:" => snapshot.rchar = value.unwrap_or(0),
                    "read_bytes:" => snapshot.read_bytes = value.unwrap_or(0),
                    _ => {}
                }
            }
        }
        if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
            if let Some((_, rest)) = stat.rsplit_once(')') {
                let fields: Vec<&str> = rest.split_whitespace().collect();
                snapshot.majflt = fields
                    .get(9)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
            }
        }
        snapshot.pgmajfault = cgroup_pgmajfault();
        snapshot
    }

    fn delta(self, before: Self) -> Self {
        Self {
            syscr: self.syscr.saturating_sub(before.syscr),
            rchar: self.rchar.saturating_sub(before.rchar),
            read_bytes: self.read_bytes.saturating_sub(before.read_bytes),
            majflt: self.majflt.saturating_sub(before.majflt),
            pgmajfault: match (self.pgmajfault, before.pgmajfault) {
                (Some(after), Some(before)) => Some(after.saturating_sub(before)),
                (after, _) => after,
            },
        }
    }
}

fn cgroup_pgmajfault() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim();
    let path = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    let stat = std::fs::read_to_string(path.join("memory.stat"))
        .or_else(|_| std::fs::read_to_string("/sys/fs/cgroup/memory.stat"))
        .ok()?;
    stat.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? == "pgmajfault" {
            parts.next()?.parse().ok()
        } else {
            None
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EvictScope {
    All,
    PacksOnly,
    IndexOnly,
}

impl EvictScope {
    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::PacksOnly => "packs_only",
            Self::IndexOnly => "index_only",
        }
    }
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn evict(dir: &Path, scope: EvictScope) -> bool {
    let mut paths = Vec::new();
    match scope {
        EvictScope::All => paths.push(dir.to_path_buf()),
        EvictScope::PacksOnly | EvictScope::IndexOnly => {
            let mut files = Vec::new();
            walk_files(dir, &mut files);
            paths.extend(files.into_iter().filter(|path| {
                let extension = path.extension().and_then(|value| value.to_str());
                match scope {
                    EvictScope::PacksOnly => extension == Some("pack"),
                    EvictScope::IndexOnly => matches!(extension, Some("run" | "idx")),
                    EvictScope::All => true,
                }
            }));
        }
    }
    evict_paths(&paths)
}

fn evict_paths(paths: &[PathBuf]) -> bool {
    if paths.is_empty() {
        return true;
    }
    let mut ok = true;
    for chunk in paths.chunks(256) {
        ok &= std::process::Command::new("vmtouch")
            .arg("-e")
            .args(chunk)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
    }
    ok
}

fn median(values: &mut [Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn min(values: &[Duration]) -> Duration {
    values.iter().copied().min().unwrap_or(Duration::ZERO)
}

fn write_synthetic(db: &Database, rooms: &mut [Room], counts: &[usize], batch: usize) {
    let mut sequence = 0u64;
    let mut completed = vec![0usize; rooms.len()];
    loop {
        let mut progressed = false;
        for (room_index, room) in rooms.iter_mut().enumerate() {
            let target = counts.get(room_index).copied().unwrap_or(0);
            let done = completed.get(room_index).copied().unwrap_or(0);
            if done >= target {
                continue;
            }
            progressed = true;
            let take = target.saturating_sub(done).min(batch);
            let mut entries = Vec::with_capacity(take);
            let mut payloads = Vec::with_capacity(take);
            for _ in 0..take {
                sequence = sequence.saturating_add(1);
                let len = PAYLOAD_MIN
                    .saturating_add(
                        usize::try_from(sequence % 16)
                            .unwrap_or(0)
                            .saturating_mul(PAYLOAD_STEP),
                    )
                    .min(PAYLOAD_MAX);
                let node = node_id(sequence);
                let stream = if sequence.is_multiple_of(50) {
                    -i64::try_from(sequence / 50).unwrap_or(i64::MAX)
                } else {
                    let stream = room.next_stream;
                    room.next_stream = room.next_stream.saturating_add(1);
                    stream
                };
                room.max_stream = room.max_stream.max(stream);
                let topo = stream
                    .saturating_add(i64::from(room.scope[0]) * 1_000_000)
                    .saturating_sub(if sequence.is_multiple_of(20) { 7 } else { 0 });
                entries.push(EventTimelineEntry::new(
                    room.scope,
                    topo,
                    stream,
                    node,
                    event_ref(node),
                ));
                payloads.push((node, NodeData::new(payload(sequence, len))));
            }
            completed[room_index] = done.saturating_add(take);
            let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
            let transaction = db.begin_transaction();
            stage_event_timeline_batch(
                &transaction,
                ShardType::EventDag,
                room.collection,
                &payloads,
                &topo_index,
                &stream_index,
                &entries,
            )
            .expect("stage multi-room event batch");
            transaction.commit().expect("commit multi-room event batch");
        }
        if !progressed {
            break;
        }
    }
}

fn write_manifest(db: &Database, rooms: &[Room], events: &[SpecEvent], batch: usize) {
    let mut by_room: Vec<Vec<&SpecEvent>> = vec![Vec::new(); rooms.len()];
    for event in events {
        if let Some(slot) = by_room.get_mut(event.room) {
            slot.push(event);
        }
    }
    let mut sequence = 0u64;
    let rounds = by_room
        .iter()
        .map(|room| room.len().div_ceil(batch))
        .max()
        .unwrap_or(0);
    let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
    for round in 0..rounds {
        for (room_index, room_events) in by_room.iter().enumerate() {
            let start = round.saturating_mul(batch);
            let Some(slice) = room_events.get(start..) else {
                continue;
            };
            let take = slice.len().min(batch);
            if take == 0 {
                continue;
            }
            let segment = &slice[..take];
            let Some(room) = rooms.get(room_index) else {
                continue;
            };
            let mut entries = Vec::with_capacity(take);
            let mut payloads = Vec::with_capacity(take);
            for event in segment {
                sequence = sequence.saturating_add(1);
                let node = node_id(sequence);
                entries.push(EventTimelineEntry::new(
                    room.scope,
                    event.topo,
                    event.stream,
                    node,
                    event_ref(node),
                ));
                payloads.push((node, NodeData::new(payload(sequence, event.bytes))));
            }
            let transaction = db.begin_transaction();
            stage_event_timeline_batch(
                &transaction,
                ShardType::EventDag,
                room.collection,
                &payloads,
                &topo_index,
                &stream_index,
                &entries,
            )
            .expect("stage manifest event batch");
            transaction.commit().expect("commit manifest event batch");
        }
    }
}

fn main() {
    let root = bench_root();
    let _ = std::fs::remove_dir_all(&root);
    let events = env_usize("MTXDB_MULTI_EVENTS", 100_000);
    let room_count = env_usize("MTXDB_MULTI_ROOMS", 100);
    let batch = env_usize("MTXDB_MULTI_BATCH", 64).max(1);
    let samples = env_usize("MTXDB_TIMELINE_SAMPLES", 3).max(1);

    let manifest = std::env::var_os("MTXDB_TIMELINE_EXPORT").map(PathBuf::from);
    let (mut rooms, counts, manifest_events) = match manifest.as_deref() {
        Some(path) => match manifest_rooms(path) {
            Ok((rooms, manifest_events)) => {
                let counts =
                    manifest_events
                        .iter()
                        .fold(vec![0usize; rooms.len()], |mut acc, event| {
                            if let Some(count) = acc.get_mut(event.room) {
                                *count = count.saturating_add(1);
                            }
                            acc
                        });
                (rooms, counts, Some(manifest_events))
            }
            Err(error) => {
                eprintln!("timeline_multi_room: {error}");
                std::process::exit(2);
            }
        },
        None => {
            let (rooms, counts) = synthetic_rooms(room_count, events);
            (rooms, counts, None)
        }
    };
    let total_events: usize = counts.iter().sum();
    let mut db = Database::open(root.clone()).expect("open benchmark database");

    let write_started = Instant::now();
    if let Some(manifest_events) = manifest_events.as_deref() {
        write_manifest(&db, &rooms, manifest_events, batch);
    } else {
        write_synthetic(&db, &mut rooms, &counts, batch);
    }
    db.event_dag().sync_all().expect("sync benchmark database");
    let write_ms = write_started.elapsed().as_secs_f64() * 1e3;
    println!(
        "multi_room_load events={total_events} rooms={} batch={batch} write_ms={write_ms:.1}",
        rooms.len()
    );

    let mut ranked: Vec<usize> = rooms
        .iter()
        .enumerate()
        .filter(|(_, room)| room.max_stream > 0)
        .map(|(index, _)| index)
        .collect();
    ranked.sort_unstable_by_key(|index| rooms.get(*index).map_or(0, |room| room.max_stream));
    let mut tiers: Vec<usize> = Vec::new();
    if let Some(index) = ranked.last() {
        tiers.push(*index);
    }
    if ranked.len() > 2 {
        if let Some(index) = ranked.get(ranked.len() / 2) {
            tiers.push(*index);
        }
    }
    if let Some(index) = ranked.first() {
        tiers.push(*index);
    }
    tiers.dedup();

    for (tier_rank, tier) in tiers.clone().into_iter().enumerate() {
        let Some(room) = rooms.get(tier) else {
            continue;
        };
        let room_id = String::from_utf8_lossy(&room.id).into_owned();
        for (order, limit) in [
            ("topo", 50usize),
            ("topo", 128),
            ("stream_up_to", 50),
            ("stream_up_to", 128),
        ] {
            let mut pages: Vec<Duration> = Vec::new();
            let mut last_io = IoSnapshot::default();
            for _ in 0..samples {
                db.event_dag().sync_all().expect("sync before eviction");
                drop(db);
                db = Database::open(root.clone()).expect("reopen benchmark database");
                let evicted = evict(&root, EvictScope::All);
                if !evicted {
                    eprintln!("timeline_multi_room: vmtouch -e unavailable; cold rows are warm");
                }
                let (topo_index, stream_index) =
                    open_event_timeline_indexes(db.event_dag().as_ref());
                let before = IoSnapshot::read();
                let (elapsed, returned) = match order {
                    "topo" => {
                        let started = Instant::now();
                        let page = topo_index
                            .page(&room.scope, None, true, limit)
                            .expect("cold topological page");
                        (started.elapsed(), page.entries.len())
                    }
                    _ => {
                        let started = Instant::now();
                        let page = page_stream_up_to(
                            &stream_index,
                            &room.scope,
                            None,
                            true,
                            limit,
                            room.max_stream,
                        )
                        .expect("cold stream page");
                        (started.elapsed(), page.entries.len())
                    }
                };
                last_io = IoSnapshot::read().delta(before);
                std::hint::black_box(returned);
                pages.push(elapsed);
            }
            let min_us = min(&pages).as_secs_f64() * 1e6;
            let median_us = median(&mut pages).as_secs_f64() * 1e6;
            println!(
                "multi_room_page room={room_id} tier={} order={order} limit={limit} min_us={min_us:.1} median_us={median_us:.1} syscr={} disk_read_bytes={} majflt={} cgroup_pgmajfault={}",
                tier_label(tier_rank),
                last_io.syscr,
                last_io.read_bytes,
                last_io.majflt,
                last_io.pgmajfault.map_or(u64::MAX, |value| value),
            );
        }
    }

    for scope in [
        EvictScope::All,
        EvictScope::PacksOnly,
        EvictScope::IndexOnly,
    ] {
        let Some(room_index) = tiers.first().copied() else {
            continue;
        };
        let Some(room) = rooms.get(room_index) else {
            continue;
        };
        db.event_dag().sync_all().expect("sync before fetch");
        drop(db);
        db = Database::open(root.clone()).expect("reopen benchmark database");
        let (topo_index, _stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
        let page = topo_index
            .page(&room.scope, None, true, 128)
            .expect("warm index page");
        let ids: Vec<NodeId> = page
            .entries
            .iter()
            .map(|entry| entry.event_node_id)
            .collect();
        let _ = evict(&root, scope);
        let before = IoSnapshot::read();
        let started = Instant::now();
        let payloads = db
            .pool(ShardType::EventDag)
            .get_many(&room.collection, &ids)
            .expect("cold payload fetch");
        let elapsed = started.elapsed();
        let io = IoSnapshot::read().delta(before);
        let returned: usize = payloads
            .iter()
            .flatten()
            .map(|record| record.bytes.len())
            .sum();
        let amplification = if returned == 0 {
            0.0
        } else {
            io.read_bytes as f64 / returned as f64
        };
        println!(
            "multi_room_fetch evict={} count={} median_us={:.1} syscr={} rchar={} disk_read_bytes={} payload_bytes_returned={returned} read_amp_ratio={amplification:.3} majflt={} cgroup_pgmajfault={}",
            scope.label(),
            ids.len(),
            elapsed.as_secs_f64() * 1e6,
            io.syscr,
            io.rchar,
            io.read_bytes,
            io.majflt,
            io.pgmajfault.map_or(u64::MAX, |value| value),
        );
    }

    let Some(largest) = tiers.first().copied() else {
        return;
    };
    let Some(room) = rooms.get(largest) else {
        return;
    };
    let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
    let mut promotion_times = Vec::new();
    let mut sequence = u64::from(u32::MAX);
    for _ in 0..20 {
        sequence = sequence.saturating_add(1);
        let node = node_id(sequence);
        let entry = EventTimelineEntry::new(
            room.scope,
            i64::try_from(sequence).unwrap_or(i64::MAX),
            i64::try_from(sequence).unwrap_or(i64::MAX),
            node,
            event_ref(node),
        );
        let transaction = db.begin_transaction();
        stage_event_timeline_batch(
            &transaction,
            ShardType::EventDag,
            room.collection,
            &[(node, NodeData::new(payload(sequence, PAYLOAD_MIN)))],
            &topo_index,
            &stream_index,
            &[entry],
        )
        .expect("stage promotion");
        let started = Instant::now();
        transaction.commit().expect("commit promotion");
        promotion_times.push(started.elapsed());
    }
    let page = topo_index
        .page(&room.scope, None, true, 128)
        .expect("mutation source page");
    let remove_entries: Vec<EventTimelineEntry> = page
        .entries
        .iter()
        .map(|key| {
            EventTimelineEntry::new(
                key.room_scope,
                key.topological_ordering,
                key.stream_ordering,
                key.event_node_id,
                key.event_ref,
            )
        })
        .collect();
    let transaction = db.begin_transaction();
    let started = Instant::now();
    let removed = stage_remove_event_timeline_batch(
        &transaction,
        ShardType::EventDag,
        &topo_index,
        &stream_index,
        &remove_entries,
    )
    .expect("stage batch removal");
    transaction.commit().expect("commit batch removal");
    let remove_us = started.elapsed().as_secs_f64() * 1e6;
    let removed_count = removed.map_or(0, |roots| roots.0 .1);

    let transaction = db.begin_transaction();
    let started = Instant::now();
    let purged = stage_purge_room_timeline(
        &transaction,
        ShardType::EventDag,
        &topo_index,
        &stream_index,
        &room.scope,
    )
    .expect("stage room purge");
    transaction.commit().expect("commit room purge");
    let purge_us = started.elapsed().as_secs_f64() * 1e6;
    let single_median_us = median(&mut promotion_times).as_secs_f64() * 1e6;
    println!(
        "multi_room_mutate single_median_us={single_median_us:.1} batch_remove_count={removed_count} batch_remove_us={remove_us:.1} purge_removed={} purge_us={purge_us:.1}",
        purged.0 .1,
    );
}

fn tier_label(rank: usize) -> &'static str {
    match rank {
        0 => "large",
        1 => "medium",
        _ => "small",
    }
}
