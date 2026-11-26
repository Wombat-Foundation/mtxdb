//! Multi-room cold-cache harness for the shared Matrix event-order indexes.
//!
//! Run with:
//! `cargo bench --manifest-path benches/Cargo.toml --bench timeline_multi_room`
//!
//! Defaults to a deterministic Zipfian synthetic corpus
//! (`MTXDB_MULTI_EVENTS=1000000`, `MTXDB_MULTI_ROOMS=1000`). Set
//! `MTXDB_TIMELINE_EXPORT` to a CSV length manifest with
//! `room,topo,stream,payload_bytes` rows to drive size-shaped synthetic reads.
//! `MTXDB_PAYLOAD_MIN` and `MTXDB_PAYLOAD_MAX` override synthetic payload bounds.
//! `MTXDB_MAX_SHARD_BYTES` rotates the event pool's active pack so sealed sidecars exist.
//! The load is synced and reclaimed once before read phases. Each resident
//! cold sample closes the database, evicts selected files, then reopens before
//! timing; the manifest mode does not read payloads from an existing database.
//! `MTXDB_TIMELINE_SAMPLES` controls cold repetitions; warm pages use 20.
//!
//! Output rows separate index traversal (`multi_room_page`), locator plus body
//! fetches (`multi_room_fetch`), and mutations (`multi_room_mutate`). I/O
//! counters come from `/proc/self/io`, `/proc/self/stat`, and cgroup v2
//! `memory.stat`; they are process-wide deltas around each measurement.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bytes::Bytes;
use mtxdb::index::IndexResidency;
use mtxdb::ordered_timeline::{
    open_event_timeline_indexes, page_stream_up_to, stage_event_timeline_batch,
    stage_promote_outlier_to_timeline, stage_purge_room_timeline,
    stage_remove_event_timeline_batch, EventTimelineEntry,
};
use mtxdb::packfile::storage::PackfileStorage;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::{interop, Database, DatabaseLayout, PoolPolicies, RuntimeConfig, ShardType};
use rustix::fs::{fadvise, Advice};

const PAYLOAD_MIN: usize = 256;
const PAYLOAD_STEP: usize = 240;
const PAYLOAD_MAX: usize = 4096;
const EMPTY_SHARED_WAL_LEN: u64 = 12_288;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn bench_root() -> PathBuf {
    std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/bench-scratch/timeline-multi-room")
        },
        PathBuf::from,
    )
}

fn warn_if_tmpfs(root: &Path) {
    let Ok(fs) = rustix::fs::statfs(root) else {
        return;
    };
    if u64::try_from(fs.f_type).unwrap_or(u64::MAX) == 0x0102_1994 {
        eprintln!(
            "timeline_multi_room: {} is on tmpfs; fadvise and read_bytes cannot observe cold I/O",
            root.display()
        );
    }
}

fn node_id(sequence: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&sequence.to_be_bytes());
    id[8..].copy_from_slice(&sequence.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_be_bytes());
    id
}

fn payload(sequence: u64, len: usize) -> Bytes {
    const TEXT_SPAN: &[u8] = b"\"type\":\"m.room.message\",\"content\":{\"body\":\"hello world\"},";
    let mut bytes = vec![0u8; len];
    for (chunk_index, chunk) in bytes.chunks_mut(64).enumerate() {
        let random_len = chunk.len().min(16);
        let mut state = sequence ^ u64::try_from(chunk_index).unwrap_or(u64::MAX);
        for (offset, byte) in chunk.iter_mut().take(random_len).enumerate() {
            if offset.is_multiple_of(8) {
                state = splitmix64(state);
            }
            let shift = u32::try_from(offset % 8).unwrap_or(0).saturating_mul(8);
            *byte = u8::try_from((state >> shift) & u64::from(u8::MAX)).unwrap_or(0);
        }
        for (offset, byte) in chunk.iter_mut().enumerate().skip(random_len) {
            let text_offset = offset
                .saturating_sub(random_len)
                .checked_rem(TEXT_SPAN.len())
                .unwrap_or(0);
            *byte = TEXT_SPAN.get(text_offset).copied().unwrap_or_default();
        }
    }
    let header = node_id(sequence);
    let copy = header.len().min(bytes.len());
    bytes[..copy].copy_from_slice(&header[..copy]);
    Bytes::from(bytes)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
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
    let weights: Vec<u64> = (0..rooms)
        .map(|index| {
            1_000_000_u64
                .checked_div(
                    u64::try_from(index.saturating_add(1))
                        .unwrap_or(u64::MAX)
                        .max(1),
                )
                .unwrap_or(0)
        })
        .collect();
    let total = weights.iter().copied().fold(0u64, u64::saturating_add);
    let events = u64::try_from(events).unwrap_or(u64::MAX);
    let mut counts: Vec<usize> = weights
        .iter()
        .map(|weight| {
            let scaled = events
                .saturating_mul(*weight)
                .checked_div(total.max(1))
                .unwrap_or(0);
            usize::try_from(scaled).unwrap_or(usize::MAX).max(1)
        })
        .collect();
    let events = usize::try_from(events).unwrap_or(usize::MAX);
    let assigned = counts.iter().copied().fold(0usize, usize::saturating_add);
    if assigned != events {
        let largest = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .map_or(0, |(index, _)| index);
        if assigned < events {
            counts[largest] = counts[largest].saturating_add(events.abs_diff(assigned));
        } else {
            counts[largest] = counts[largest].saturating_sub(assigned.abs_diff(events));
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
        let topo = fields[1].parse::<i64>().map_err(|error| {
            format!(
                "{}:{}: {error}",
                path.display(),
                line_number.saturating_add(1)
            )
        })?;
        let stream = fields[2].parse::<i64>().map_err(|error| {
            format!(
                "{}:{}: {error}",
                path.display(),
                line_number.saturating_add(1)
            )
        })?;
        let bytes = fields[3].parse::<usize>().map_err(|error| {
            format!(
                "{}:{}: {error}",
                path.display(),
                line_number.saturating_add(1)
            )
        })?;
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

fn proc_status_kib(field: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? == field {
            parts.next()?.parse().ok()
        } else {
            None
        }
    })
}

fn proc_smaps_kib(field: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    text.lines().find_map(|line| {
        let (name, value) = line.rsplit_once(':')?;
        if name.trim() != field {
            return None;
        }
        value.split_whitespace().next()?.parse().ok()
    })
}

fn file_kind(path: &Path) -> &'static str {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let extension = path.extension().and_then(|value| value.to_str());
    if extension == Some("pack") {
        "pack"
    } else if name == "wal.bin" {
        "wal"
    } else if name.starts_with("index.")
        || name.starts_with("shard_")
        || matches!(extension, Some("idx" | "run"))
    {
        "index"
    } else {
        "other"
    }
}

fn directory_bytes(root: &Path) -> (u64, u64, u64, u64, u64) {
    let mut files = Vec::new();
    walk_files(root, &mut files);
    let mut pack = 0u64;
    let mut index = 0u64;
    let mut run = 0u64;
    let mut wal = 0u64;
    let mut other = 0u64;
    for path in files {
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        let len = metadata.len();
        match file_kind(&path) {
            "pack" => pack = pack.saturating_add(len),
            "index" => {
                if path.extension().and_then(|value| value.to_str()) == Some("run") {
                    run = run.saturating_add(len);
                } else {
                    index = index.saturating_add(len);
                }
            }
            "wal" => wal = wal.saturating_add(len),
            _ => other = other.saturating_add(len),
        }
    }
    (pack, index, run, wal, other)
}

fn raw_pack_frame_count(root: &Path) -> std::io::Result<u64> {
    let mut files = Vec::new();
    walk_files(root, &mut files);
    files
        .into_iter()
        .filter(|path| file_kind(path) == "pack")
        .try_fold(0u64, |total, path| {
            let count = mtxdb::packfile::scan_packfile_iter(&path, false)?
                .try_fold(0u64, |count, frame| frame.map(|_| count.saturating_add(1)))?;
            Ok(total.saturating_add(count))
        })
}

fn collection_pack_bytes(root: &Path, collection_id: &[u8; 16]) -> std::io::Result<u64> {
    let mut files = Vec::new();
    walk_files(root, &mut files);
    files
        .into_iter()
        .filter(|path| file_kind(path) == "pack")
        .try_fold(0_u64, |total, path| {
            let file_len = std::fs::metadata(&path)?.len();
            let mut previous = None;
            let mut collection_bytes = 0_u64;
            for entry in mtxdb::packfile::scan_packfile_iter(&path, false)? {
                let (entry_collection, _, offset) = entry?;
                if let Some((previous_collection, previous_offset)) = previous {
                    let frame_len = offset.saturating_sub(previous_offset);
                    if previous_collection == *collection_id {
                        collection_bytes = collection_bytes.saturating_add(frame_len);
                    }
                }
                previous = Some((entry_collection, offset));
            }
            if let Some((previous_collection, previous_offset)) = previous {
                if previous_collection == *collection_id {
                    collection_bytes =
                        collection_bytes.saturating_add(file_len.saturating_sub(previous_offset));
                }
            }
            Ok(total.saturating_add(collection_bytes))
        })
}

fn collection_payload_bytes(
    store: &dyn StorageEngine,
    collection_id: &[u8; 16],
) -> Result<(usize, u64), Box<dyn std::error::Error>> {
    let ids = store.collection_record_ids(collection_id)?;
    let records = store.get_many_bytes(collection_id, &ids)?;
    let bytes = records.into_iter().flatten().fold(0_u64, |total, payload| {
        total.saturating_add(u64::try_from(payload.len()).unwrap_or(u64::MAX))
    });
    Ok((ids.len(), bytes))
}

fn print_ordered_write_metrics(db: &Database, root: &Path, events: usize, batch: usize) {
    let (topo, stream) = open_event_timeline_indexes(db.event_dag().as_ref());
    let topo_raw = collection_payload_bytes(db.event_dag().as_ref(), &topo.collection_id())
        .expect("read raw topological index bytes");
    let stream_raw = collection_payload_bytes(db.event_dag().as_ref(), &stream.collection_id())
        .expect("read raw stream index bytes");
    let topo_pack = collection_pack_bytes(root, &topo.collection_id())
        .expect("scan topological index frame bytes");
    let stream_pack = collection_pack_bytes(root, &stream.collection_id())
        .expect("scan stream index frame bytes");
    let raw_total = topo_raw.1.saturating_add(stream_raw.1);
    let pack_total = topo_pack.saturating_add(stream_pack);
    let divisor = u64::try_from(events.max(1)).unwrap_or(u64::MAX);
    let raw_per_event = raw_total.checked_div(divisor).unwrap_or_default();
    let packed_per_event = pack_total.checked_div(divisor).unwrap_or_default();
    println!(
        "multi_room_index_write batch={batch} events={events} topo_records={} topo_raw_bytes={} topo_pack_bytes={topo_pack} stream_records={} stream_raw_bytes={} stream_pack_bytes={stream_pack} combined_raw_bytes={raw_total} combined_pack_bytes={pack_total} raw_bytes_per_event={} packed_bytes_per_event={}",
        topo_raw.0,
        topo_raw.1,
        stream_raw.0,
        stream_raw.1,
        raw_per_event,
        packed_per_event,
    );
}

fn extension_bytes(root: &Path) -> Vec<(String, u64)> {
    let mut files = Vec::new();
    walk_files(root, &mut files);
    let mut totals = BTreeMap::new();
    for path in files {
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let extension = path.extension().and_then(|value| value.to_str());
        let kind = if extension == Some("idx") {
            "idx".to_owned()
        } else if extension == Some("run") {
            "run".to_owned()
        } else if extension == Some("pack") {
            "pack".to_owned()
        } else if name == "wal.bin" {
            "wal".to_owned()
        } else if name.starts_with("index.delta.") {
            "index.delta".to_owned()
        } else if name.starts_with("index.checkpoint") {
            "index.checkpoint".to_owned()
        } else if name.starts_with("index.versions") {
            "index.versions".to_owned()
        } else if name.starts_with("shard_") {
            "shard_metadata".to_owned()
        } else {
            path.extension()
                .and_then(|value| value.to_str())
                .map_or_else(|| "(no extension)".to_owned(), str::to_owned)
        };
        let total = totals.entry(kind).or_insert(0u64);
        *total = total.saturating_add(metadata.len());
    }
    totals.into_iter().collect()
}

fn evict(dir: &Path, scope: EvictScope) -> std::io::Result<()> {
    let mut files = Vec::new();
    walk_files(dir, &mut files);
    let paths: Vec<PathBuf> = files
        .into_iter()
        .filter(|path| match scope {
            EvictScope::All => true,
            EvictScope::PacksOnly => file_kind(path) == "pack",
            EvictScope::IndexOnly => file_kind(path) == "index",
        })
        .collect();
    evict_paths(&paths)
}

fn evict_paths(paths: &[PathBuf]) -> std::io::Result<()> {
    for path in paths {
        if !path.is_file() {
            continue;
        }
        let file = std::fs::File::open(path)?;
        fadvise(&file, 0, None, Advice::DontNeed).map_err(std::io::Error::from)?;
    }
    Ok(())
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

fn open_benchmark_database(root: PathBuf) -> Database {
    let mut config = if std::env::var_os("MTXDB_CACHE_CAPACITY").is_some() {
        RuntimeConfig::from_env().expect("benchmark runtime config")
    } else {
        RuntimeConfig::default()
    };
    // Each advice mode selects the point mapping itself; an ambient setting
    // must not change the "default" rows.
    config.point_random_mmap = false;
    let mut policies = PoolPolicies::default();
    if let Some(bytes) = std::env::var("MTXDB_MAX_SHARD_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    {
        policies.for_shard_mut(ShardType::EventDag).max_shard_bytes = Some(bytes);
    }
    Database::open_with_policies_and_config(root, policies, config)
        .expect("open benchmark database")
}

fn maintain_journal(db: &Database) {
    let coordinator = db.coordinator();
    if coordinator.segment_len() > coordinator.reclaim_trigger_len() {
        let _ = db.event_dag().sync_all();
        let _ = coordinator.reclaim_shared();
    }
}

fn print_extension_breakdown(root: &Path, phase: &str) {
    let rendered: Vec<String> = extension_bytes(root)
        .into_iter()
        .map(|(kind, bytes)| format!("{kind}={bytes}"))
        .collect();
    println!("multi_room_files phase={phase} {}", rendered.join(" "));
}

fn print_memory_diagnostics(db: &Database, root: &Path, rooms: &[Room]) {
    let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
    let count_records = |collection_id: &[u8; 16]| {
        db.event_dag()
            .collection_len(collection_id)
            .ok()
            .flatten()
            .unwrap_or(0)
    };
    let payload_records = rooms.iter().fold(0u64, |total, room| {
        let count = u64::try_from(count_records(&room.collection)).unwrap_or(u64::MAX);
        total.saturating_add(count)
    });
    let topo_records = count_records(&topo_index.collection_id());
    let stream_records = count_records(&stream_index.collection_id());
    let (pack_bytes, idx_bytes, run_bytes, wal_bytes, other_bytes) = directory_bytes(root);
    let mut index_bytes = 0_u64;
    let mut cache_bytes = 0_usize;
    for shard in ShardType::ALL {
        let pool = db.pool(shard);
        index_bytes = index_bytes.saturating_add(pool.stats().index_bytes);
        cache_bytes = cache_bytes.saturating_add(pool.resident_cache_stats().bytes);
    }
    println!(
        "multi_room_memory rss_kib={} anon_kib={} file_kib={} shmem_kib={} hwm_kib={} pss_anon_kib={} pss_file_kib={} index_bytes={index_bytes} cache_bytes={cache_bytes} pack_bytes={pack_bytes} idx_bytes={idx_bytes} run_bytes={run_bytes} wal_bytes={wal_bytes} other_bytes={other_bytes} payload_records={payload_records} topo_records={topo_records} stream_records={stream_records}",
        proc_status_kib("VmRSS:").unwrap_or(0),
        proc_status_kib("RssAnon:").unwrap_or(0),
        proc_status_kib("RssFile:").unwrap_or(0),
        proc_status_kib("RssShmem:").unwrap_or(0),
        proc_status_kib("VmHWM:").unwrap_or(0),
        proc_smaps_kib("Pss_Anon").unwrap_or(0),
        proc_smaps_kib("Pss_File").unwrap_or(0),
    );
}

fn reopen_benchmark_database(root: &Path, db: Database) -> Database {
    let before_files = directory_bytes(root);
    let before_frames = raw_pack_frame_count(root).expect("count raw pack frames before reopen");
    let open_before = IoSnapshot::read();
    drop(db);
    let db = open_benchmark_database(root.to_path_buf());
    let open_io = IoSnapshot::read().delta(open_before);
    let after_files = directory_bytes(root);
    let after_frames = raw_pack_frame_count(root).expect("count raw pack frames after reopen");
    println!(
        "multi_room_reopen read_bytes={} majflt={} cgroup_pgmajfault={} pack_before={} pack_after={} frames_before={} frames_after={} wal_before={} wal_after={}",
        open_io.read_bytes,
        open_io.majflt,
        open_io.pgmajfault.unwrap_or(u64::MAX),
        before_files.0,
        after_files.0,
        before_frames,
        after_frames,
        before_files.3,
        after_files.3,
    );
    print_extension_breakdown(root, "after_reopen");
    db
}

fn checkpoint_all_pools(db: &Database, root: &Path) -> (u64, u64) {
    let before = raw_pack_frame_count(root).expect("count raw pack frames before checkpoint");
    for pool in ShardType::ALL {
        db.pool(pool)
            .sync_all()
            .expect("sync benchmark pool before checkpoint");
    }
    let coverage_before = ShardType::ALL
        .into_iter()
        .map(|pool| {
            format!(
                "{}:{}/{}",
                pool.as_str(),
                db.pool(pool).durable_coverage(),
                db.coordinator().committed_lsn_for_pool(pool),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    for pool in ShardType::ALL {
        db.pool(pool)
            .force_index_checkpoint()
            .expect("checkpoint benchmark pool before measurements");
    }
    let reclaim = db
        .coordinator()
        .reclaim_shared()
        .expect("reclaim shared WAL after full checkpoint");
    let after = raw_pack_frame_count(root).expect("count raw pack frames after checkpoint");
    let coverage_after = ShardType::ALL
        .into_iter()
        .map(|pool| {
            format!(
                "{}:{}/{}",
                pool.as_str(),
                db.pool(pool).durable_coverage(),
                db.coordinator().committed_lsn_for_pool(pool),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "multi_room_checkpoint coverage_before={coverage_before} coverage_after={coverage_after} blockers={:?} reclaimed={}",
        db.coordinator().reclaim_blockers(),
        reclaim.is_some(),
    );
    (before, after)
}

fn write_synthetic(
    db: &Database,
    rooms: &mut [Room],
    counts: &[usize],
    batch: usize,
    payload_min: usize,
    payload_max: usize,
) {
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
                let len = payload_min
                    .saturating_add(
                        usize::try_from(sequence % 16)
                            .unwrap_or(0)
                            .saturating_mul(PAYLOAD_STEP),
                    )
                    .min(payload_max);
                let node = node_id(sequence);
                let stream = if sequence.is_multiple_of(50) {
                    i64::try_from(sequence / 50)
                        .unwrap_or(i64::MAX)
                        .saturating_neg()
                } else {
                    let stream = room.next_stream;
                    room.next_stream = room.next_stream.saturating_add(1);
                    stream
                };
                room.max_stream = room.max_stream.max(stream);
                let topo = stream
                    .saturating_add(i64::from(room.scope[0]).saturating_mul(1_000_000))
                    .saturating_sub(if sequence.is_multiple_of(20) { 7 } else { 0 });
                entries.push(EventTimelineEntry::new(room.scope, topo, 0, stream, node));
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
            maintain_journal(db);
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
            let Some(selected_room) = rooms.get(room_index) else {
                continue;
            };
            let mut entries = Vec::with_capacity(take);
            let mut payloads = Vec::with_capacity(take);
            for event in segment {
                sequence = sequence.saturating_add(1);
                let node = node_id(sequence);
                entries.push(EventTimelineEntry::new(
                    selected_room.scope,
                    event.topo,
                    0,
                    event.stream,
                    node,
                ));
                payloads.push((node, NodeData::new(payload(sequence, event.bytes))));
            }
            let transaction = db.begin_transaction();
            stage_event_timeline_batch(
                &transaction,
                ShardType::EventDag,
                selected_room.collection,
                &payloads,
                &topo_index,
                &stream_index,
                &entries,
            )
            .expect("stage manifest event batch");
            transaction.commit().expect("commit manifest event batch");
            maintain_journal(db);
        }
    }
}

/// Mapping-advice variants for the cold-page readahead A/B.
///
/// `MTXDB_PAGE_ADVICE_AB=1` runs both; otherwise `MTXDB_PAGE_RANDOM_ADVICE=1`
/// selects the advised variant alone (default is the unadvised baseline).
fn advice_modes() -> Vec<&'static str> {
    if std::env::var_os("MTXDB_PAGE_ADVICE_AB").is_some() {
        let mut modes = vec!["default", "random", "point", "index"];
        if std::env::var_os("MTXDB_PAGE_AB_REVERSE").is_some() {
            modes.reverse();
        }
        modes
    } else if std::env::var_os("MTXDB_PAGE_POINT_MAPPING").is_some() {
        vec!["point"]
    } else if std::env::var("MTXDB_PAGE_RANDOM_ADVICE").is_ok_and(|value| value != "0") {
        vec!["random"]
    } else {
        vec!["default"]
    }
}

fn point_get_modes() -> Vec<&'static str> {
    let mut modes = vec!["default", "point", "index"];
    if std::env::var_os("MTXDB_PAGE_AB_REVERSE").is_some() {
        modes.reverse();
    }
    modes
}

fn mapped_pack_vmas() -> usize {
    std::fs::read_to_string("/proc/self/maps").map_or(0, |maps| {
        maps.lines()
            .filter(|line| {
                let mut fields = line.split_whitespace();
                let path = fields.nth(5);
                path.is_some_and(|path| {
                    Path::new(path).extension().is_some_and(|ext| ext == "pack")
                }) && fields.next().is_none()
            })
            .count()
    })
}

fn run_page_rows(
    mut db: Database,
    root: &Path,
    rooms: &[Room],
    tiers: &[usize],
    samples: usize,
) -> Database {
    for mode in advice_modes() {
        db = run_page_rows_for_mode(db, root, rooms, tiers, samples, mode);
    }
    db
}

#[allow(clippy::too_many_lines)]
fn run_page_rows_for_mode(
    mut db: Database,
    root: &Path,
    rooms: &[Room],
    tiers: &[usize],
    samples: usize,
    mode: &str,
) -> Database {
    for (tier_rank, tier) in tiers.iter().copied().enumerate() {
        let Some(selected) = rooms.get(tier) else {
            continue;
        };
        let room_id = String::from_utf8_lossy(&selected.id).into_owned();
        for (order, limit) in [
            ("topo", 50usize),
            ("topo", 128),
            ("stream_up_to", 50),
            ("stream_up_to", 128),
        ] {
            let mut pages: Vec<Duration> = Vec::new();
            let mut last_io = IoSnapshot::default();
            let mut last_open_io = IoSnapshot::default();
            let mut last_pack_vmas = 0usize;
            for _ in 0..samples {
                drop(db);
                evict(root, EvictScope::All).expect("evict files before cold read");
                let open_before = IoSnapshot::read();
                db = open_benchmark_database(root.to_path_buf());
                last_open_io = IoSnapshot::read().delta(open_before);
                let _ = db.event_dag().unmap_shards();
                if mode == "point" {
                    db.event_dag().set_point_random_mmap(true);
                }
                if mode == "index" {
                    db.event_dag().set_point_random_mmap_index(true);
                }
                if mode == "random" {
                    let advised = db.event_dag().advise_random_shards();
                    std::hint::black_box(advised);
                }
                evict(root, EvictScope::All).expect("evict files after open");
                let (topo_index, stream_index) =
                    open_event_timeline_indexes(db.event_dag().as_ref());
                let before = IoSnapshot::read();
                let (elapsed, returned) = if order == "topo" {
                    let started = Instant::now();
                    let page = topo_index
                        .page(&selected.scope, None, true, limit)
                        .expect("cold topological page");
                    (started.elapsed(), page.entries.len())
                } else {
                    let started = Instant::now();
                    let page = page_stream_up_to(
                        &stream_index,
                        &selected.scope,
                        None,
                        true,
                        limit,
                        selected.max_stream,
                    )
                    .expect("cold stream page");
                    (started.elapsed(), page.entries.len())
                };
                last_io = IoSnapshot::read().delta(before);
                last_pack_vmas = mapped_pack_vmas();
                std::hint::black_box(returned);
                pages.push(elapsed);
            }
            let min_us = min(&pages).as_secs_f64() * 1e6;
            let median_us = median(&mut pages).as_secs_f64() * 1e6;
            println!(
                "multi_room_page room={room_id} tier={} order={order} limit={limit} advice={mode} min_us={min_us:.1} median_us={median_us:.1} pack_vmas={last_pack_vmas} syscr={} disk_read_bytes={} open_read_bytes={} open_majflt={} majflt={} cgroup_pgmajfault={} open_cgroup_pgmajfault={}",
                tier_label(tier_rank),
                last_io.syscr,
                last_io.read_bytes,
                last_open_io.read_bytes,
                last_open_io.majflt,
                last_io.majflt,
                last_io.pgmajfault.unwrap_or(u64::MAX),
                last_open_io.pgmajfault.unwrap_or(u64::MAX),
            );
            let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
            let mut warm_pages = Vec::with_capacity(20);
            for _ in 0..20 {
                let started = Instant::now();
                let returned = if order == "topo" {
                    topo_index
                        .page(&selected.scope, None, true, limit)
                        .expect("warm topological page")
                        .entries
                        .len()
                } else {
                    page_stream_up_to(
                        &stream_index,
                        &selected.scope,
                        None,
                        true,
                        limit,
                        selected.max_stream,
                    )
                    .expect("warm stream page")
                    .entries
                    .len()
                };
                std::hint::black_box(returned);
                warm_pages.push(started.elapsed());
            }
            println!(
                "multi_room_page_warm room={room_id} tier={} order={order} limit={limit} advice={mode} samples={} min_us={:.1} median_us={:.1}",
                tier_label(tier_rank),
                warm_pages.len(),
                min(&warm_pages).as_secs_f64() * 1e6,
                median(&mut warm_pages).as_secs_f64() * 1e6,
            );
        }
    }

    run_mapping_control(&db, root, mode);
    drop(db);
    open_benchmark_database(root.to_path_buf())
}

/// Walk the normal mapping on the same handle that just ran the page rows.
/// This keeps the point mapping live while measuring sequential access.
fn run_mapping_control(db: &Database, root: &Path, mode: &str) {
    // Keep this to one walk: advice is shared by the mapping, while repeated
    // walks can become cache-resident even after fadvise(DONTNEED).
    evict(root, EvictScope::All).expect("evict before mapping control");
    let before = IoSnapshot::read();
    let started = Instant::now();
    let pages = db.event_dag().touch_mapped_pages();
    let elapsed = started.elapsed();
    let io = IoSnapshot::read().delta(before);
    println!(
        "multi_room_seq_mapping advice={mode} pages={pages} walk=first elapsed_us={:.1} majflt={} disk_read_bytes={}",
        elapsed.as_secs_f64() * 1e6,
        io.majflt,
        io.read_bytes,
    );
}

/// Sequential single-record reads through `get`: one room's event bodies in
/// topological (append) order, cold, with the point mapping off and on.
///
/// `run_mapping_control` walks the shared mapping directly, so it cannot see
/// what these reads cost: with the point mapping enabled every `get` faults
/// through the random-advised view and loses fault-around.
fn run_sequential_get_rows(root: &Path, rooms: &[Room], samples: usize) {
    const RECORDS: usize = 256;
    let Some(selected) = rooms.first() else {
        return;
    };
    let ids: Vec<NodeId> = {
        let db = open_benchmark_database(root.to_path_buf());
        let (topo_index, _) = open_event_timeline_indexes(db.event_dag().as_ref());
        topo_index
            .page(&selected.scope, None, true, RECORDS)
            .expect("select sequential get ids")
            .entries
            .iter()
            .map(|entry| entry.event_node_id)
            .collect()
    };
    if ids.is_empty() {
        return;
    }
    let room_id = String::from_utf8_lossy(&selected.id).into_owned();
    for mode in point_get_modes() {
        let mut times = Vec::with_capacity(samples);
        for sample in 0..samples {
            evict(root, EvictScope::All).expect("evict before sequential get");
            let open_before = IoSnapshot::read();
            let db = open_benchmark_database(root.to_path_buf());
            let open_io = IoSnapshot::read().delta(open_before);
            db.event_dag().set_point_random_mmap(mode == "point");
            db.event_dag().set_point_random_mmap_index(mode == "index");
            evict(root, EvictScope::All).expect("evict after sequential get open");
            let before = IoSnapshot::read();
            let started = Instant::now();
            for id in &ids {
                std::hint::black_box(
                    db.event_dag()
                        .get(&selected.collection, id)
                        .expect("sequential get"),
                );
            }
            let elapsed = started.elapsed();
            let io = IoSnapshot::read().delta(before);
            times.push(elapsed);
            println!(
                "multi_room_seq_get room={room_id} mode={mode} sample={} records={} us={:.1} disk_read_bytes={} open_read_bytes={} majflt={} pack_vmas={}",
                sample.saturating_add(1),
                ids.len(),
                elapsed.as_secs_f64() * 1e6,
                io.read_bytes,
                open_io.read_bytes,
                io.majflt,
                mapped_pack_vmas(),
            );
        }
        println!(
            "multi_room_seq_get_summary room={room_id} mode={mode} records={} samples={} min_us={:.1} median_us={:.1}",
            ids.len(),
            times.len(),
            min(&times).as_secs_f64() * 1e6,
            median(&mut times).as_secs_f64() * 1e6,
        );
    }
}

/// Disk-order variant of [`run_sequential_get_rows`].
///
/// Same records, same modes, but ordered by physical `(shard slot, offset)`
/// instead of topological order. Sorting by location does not make records
/// adjacent, so the layout line reports how many distinct slots were touched
/// and how large the gaps between consecutive frames are.
fn run_disk_order_get_rows(root: &Path, rooms: &[Room], samples: usize) {
    const RECORDS: usize = 256;
    let Some(selected) = rooms.first() else {
        return;
    };
    let ids: Vec<NodeId> = {
        let db = open_benchmark_database(root.to_path_buf());
        let (topo_index, _) = open_event_timeline_indexes(db.event_dag().as_ref());
        topo_index
            .page(&selected.scope, None, true, RECORDS)
            .expect("select disk-order get ids")
            .entries
            .iter()
            .map(|entry| entry.event_node_id)
            .collect()
    };
    if ids.is_empty() {
        return;
    }
    let mut located: Vec<(u16, u64, NodeId)> = {
        let db = open_benchmark_database(root.to_path_buf());
        ids.iter()
            .filter_map(|id| {
                db.event_dag()
                    .get_slot_location(&selected.collection, id)
                    .expect("locate disk-order record")
                    .map(|(slot, offset)| (slot, offset, *id))
            })
            .collect()
    };
    located.sort_unstable_by_key(|&(slot, offset, _)| (slot, offset));

    let slots: std::collections::BTreeSet<u16> = located.iter().map(|(slot, ..)| *slot).collect();
    let mut gaps = Vec::with_capacity(located.len().saturating_sub(1));
    for pair in located.windows(2) {
        let [(slot_a, offset_a, _), (slot_b, offset_b, _)] = pair else {
            continue;
        };
        if slot_a == slot_b && offset_b > offset_a {
            gaps.push(offset_b.saturating_sub(*offset_a));
        }
    }
    let max_gap = gaps.iter().copied().max().unwrap_or(0);
    let over_page = gaps.iter().filter(|&&gap| gap > 4096).count();
    let span = located
        .first()
        .zip(located.last())
        .map_or(0, |((_, start, _), (_, end, _))| end.saturating_sub(*start));
    let room_id = String::from_utf8_lossy(&selected.id).into_owned();
    println!(
        "multi_room_seq_get_layout room={room_id} records={} located={} slots={} span_bytes={span} max_gap_bytes={max_gap} gaps_over_page={over_page} first={:?} last={:?}",
        ids.len(),
        located.len(),
        slots.len(),
        located.first().map(|(s, o, _)| (*s, *o)),
        located.last().map(|(s, o, _)| (*s, *o)),
    );
    if located.is_empty() {
        return;
    }
    let ordered: Vec<NodeId> = located.iter().map(|&(_, _, id)| id).collect();
    for mode in point_get_modes() {
        let mut times = Vec::with_capacity(samples);
        for sample in 0..samples {
            evict(root, EvictScope::All).expect("evict before disk-order get");
            let open_before = IoSnapshot::read();
            let db = open_benchmark_database(root.to_path_buf());
            let open_io = IoSnapshot::read().delta(open_before);
            db.event_dag().set_point_random_mmap(mode == "point");
            db.event_dag().set_point_random_mmap_index(mode == "index");
            evict(root, EvictScope::All).expect("evict after disk-order get open");
            let before = IoSnapshot::read();
            let started = Instant::now();
            for id in &ordered {
                std::hint::black_box(
                    db.event_dag()
                        .get(&selected.collection, id)
                        .expect("disk-order get"),
                );
            }
            let elapsed = started.elapsed();
            let io = IoSnapshot::read().delta(before);
            times.push(elapsed);
            println!(
                "multi_room_seq_get_disk room={room_id} mode={mode} sample={} records={} us={:.1} disk_read_bytes={} open_read_bytes={} majflt={} pack_vmas={}",
                sample.saturating_add(1),
                ordered.len(),
                elapsed.as_secs_f64() * 1e6,
                io.read_bytes,
                open_io.read_bytes,
                io.majflt,
                mapped_pack_vmas(),
            );
        }
        println!(
            "multi_room_seq_get_disk_summary room={room_id} mode={mode} records={} samples={} min_us={:.1} median_us={:.1}",
            ordered.len(),
            times.len(),
            min(&times).as_secs_f64() * 1e6,
            median(&mut times).as_secs_f64() * 1e6,
        );
    }
}

/// Measure single-record reads of a physically adjacent run of event frames.
/// The selected run may span rooms because the event collection is shared.
fn run_adjacent_get_rows(root: &Path, rooms: &[Room], samples: usize) {
    const RECORDS: usize = 256;
    let Some(selected) = rooms.first() else {
        return;
    };
    let mut located = {
        let db = open_benchmark_database(root.to_path_buf());
        db.event_dag()
            .collection_live_record_locations(&selected.collection)
            .expect("scan event record locations")
            .unwrap_or_default()
    };
    located.sort_unstable_by_key(|&(_, slot, offset, _)| (slot, offset));

    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    for index in 1..=located.len() {
        // `index` starts at 1, so this never saturates.
        let previous = index.saturating_sub(1);
        let contiguous = index < located.len()
            && located[previous].1 == located[index].1
            && located[previous].2.saturating_add(located[previous].3) == located[index].2;
        if !contiguous {
            let run_len = index.saturating_sub(run_start);
            if run_len > best_len {
                best_start = run_start;
                best_len = run_len;
            }
            run_start = index;
        }
    }
    let Some(run) = located.get(best_start..best_start.saturating_add(best_len.min(RECORDS)))
    else {
        return;
    };
    if run.len() < 2 {
        println!("multi_room_seq_get_adjacent_skipped reason=no_adjacent_event_frames");
        return;
    }
    let ids: Vec<NodeId> = run.iter().map(|&(id, ..)| id).collect();
    let slot = run[0].1;
    let first_offset = run[0].2;
    let last_end = run
        .last()
        .map_or(first_offset, |record| record.2.saturating_add(record.3));
    println!(
        "multi_room_seq_get_adjacent_layout records={} shard_slot={slot} first_offset={first_offset} span_bytes={} frame_bytes={}",
        ids.len(),
        last_end.saturating_sub(first_offset),
        run.iter().map(|record| record.3).sum::<u64>(),
    );

    let modes = point_get_modes();
    let room_id = String::from_utf8_lossy(&selected.id).into_owned();
    for mode in modes {
        let mut times = Vec::with_capacity(samples);
        for sample in 0..samples {
            evict(root, EvictScope::All).expect("evict before adjacent get");
            let db = open_benchmark_database(root.to_path_buf());
            db.event_dag().set_point_random_mmap(mode == "point");
            db.event_dag().set_point_random_mmap_index(mode == "index");
            evict(root, EvictScope::All).expect("evict after adjacent get open");
            let before = IoSnapshot::read();
            let started = Instant::now();
            for id in &ids {
                std::hint::black_box(
                    db.event_dag()
                        .get(&selected.collection, id)
                        .expect("adjacent get"),
                );
            }
            let elapsed = started.elapsed();
            let io = IoSnapshot::read().delta(before);
            times.push(elapsed);
            println!(
                "multi_room_seq_get_adjacent room={room_id} mode={mode} sample={} records={} us={:.1} disk_read_bytes={} majflt={} pack_vmas={}",
                sample.saturating_add(1),
                ids.len(),
                elapsed.as_secs_f64() * 1e6,
                io.read_bytes,
                io.majflt,
                mapped_pack_vmas(),
            );
        }
        println!(
            "multi_room_seq_get_adjacent_summary room={room_id} mode={mode} records={} samples={} min_us={:.1} median_us={:.1}",
            ids.len(),
            times.len(),
            min(&times).as_secs_f64() * 1e6,
            median(&mut times).as_secs_f64() * 1e6,
        );
    }
}

/// Measure cold reads through the standalone paged-index reader. Unlike the
/// shared-WAL `Database`, this read-only pool can keep sealed-pack locators
/// paged, so locator faults happen in the measured read rather than at open.
fn run_paged_page_rows(root: &Path, rooms: &[Room], tiers: &[usize], samples: usize) {
    for mode in advice_modes() {
        run_paged_page_rows_for_mode(root, rooms, tiers, samples, mode);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "cold samples and reporting share the same paged reader"
)]
fn run_paged_page_rows_for_mode(
    root: &Path,
    rooms: &[Room],
    tiers: &[usize],
    samples: usize,
    mode: &str,
) {
    let extension_sizes = extension_bytes(root);
    let idx_bytes = extension_sizes
        .iter()
        .find_map(|(extension, bytes)| (extension == "idx").then_some(*bytes))
        .unwrap_or(0);
    let run_bytes = extension_sizes
        .iter()
        .find_map(|(extension, bytes)| (extension == "run").then_some(*bytes))
        .unwrap_or(0);
    let wal_bytes = directory_bytes(root).3;
    if wal_bytes > EMPTY_SHARED_WAL_LEN {
        println!("multi_room_page_paged_skipped reason=shared_wal_pending wal_bytes={wal_bytes}");
        return;
    }
    if idx_bytes.saturating_add(run_bytes) == 0 {
        println!(
            "multi_room_page_paged_skipped reason=no_sealed_index_sidecars idx_bytes={idx_bytes} run_bytes={run_bytes}"
        );
        return;
    }
    let layout = DatabaseLayout::open_read_only(root.to_path_buf())
        .expect("open benchmark layout read-only");
    let pool_path = layout
        .pool_dir_read_only(ShardType::EventDag)
        .expect("find event DAG pool");
    let config = RuntimeConfig {
        index_residency: Some(IndexResidency::Paged),
        ..RuntimeConfig::default()
    };

    for (tier_rank, tier) in tiers.iter().copied().enumerate() {
        let Some(selected) = rooms.get(tier) else {
            continue;
        };
        let room_id = String::from_utf8_lossy(&selected.id).into_owned();
        for (order, limit) in [
            ("topo", 50usize),
            ("topo", 128),
            ("stream_up_to", 50),
            ("stream_up_to", 128),
        ] {
            let mut elapsed_samples = Vec::with_capacity(samples);
            let mut last_io = IoSnapshot::default();
            let mut last_open_io = IoSnapshot::default();
            let mut last_pack_vmas = 0usize;
            for _ in 0..samples {
                evict(root, EvictScope::All).expect("evict before paged open");
                let open_before = IoSnapshot::read();
                let store = PackfileStorage::open_read_only_with_config(pool_path.clone(), &config)
                    .expect("open paged event DAG pool");
                last_open_io = IoSnapshot::read().delta(open_before);
                let _ = store.unmap_shards();
                if mode == "point" {
                    store.set_point_random_mmap(true);
                } else if mode == "index" {
                    store.set_point_random_mmap_index(true);
                } else if mode == "random" {
                    std::hint::black_box(store.advise_random_shards());
                }
                evict(root, EvictScope::All).expect("evict files after paged open");
                {
                    let (topo_index, stream_index) = open_event_timeline_indexes(&store);
                    let before = IoSnapshot::read();
                    let started = Instant::now();
                    let returned = if order == "topo" {
                        topo_index
                            .page(&selected.scope, None, true, limit)
                            .expect("paged topological page")
                            .entries
                            .len()
                    } else {
                        page_stream_up_to(
                            &stream_index,
                            &selected.scope,
                            None,
                            true,
                            limit,
                            selected.max_stream,
                        )
                        .expect("paged stream page")
                        .entries
                        .len()
                    };
                    elapsed_samples.push(started.elapsed());
                    last_io = IoSnapshot::read().delta(before);
                    last_pack_vmas = mapped_pack_vmas();
                    std::hint::black_box(returned);
                }
            }
            let min_us = min(&elapsed_samples).as_secs_f64() * 1e6;
            let median_us = median(&mut elapsed_samples).as_secs_f64() * 1e6;
            println!(
                "multi_room_page_paged room={room_id} tier={} order={order} limit={limit} advice={mode} samples={} min_us={min_us:.1} median_us={median_us:.1} pack_vmas={last_pack_vmas} open_read_bytes={} disk_read_bytes={} open_majflt={} majflt={} open_cgroup_pgmajfault={} cgroup_pgmajfault={}",
                tier_label(tier_rank),
                elapsed_samples.len(),
                last_open_io.read_bytes,
                last_io.read_bytes,
                last_open_io.majflt,
                last_io.majflt,
                last_open_io.pgmajfault.unwrap_or(u64::MAX),
                last_io.pgmajfault.unwrap_or(u64::MAX),
            );
        }
    }
}

fn run_paged_fetch_rows(root: &Path, rooms: &[Room], tier: Option<usize>, samples: usize) {
    let Some(selected) = tier.and_then(|index| rooms.get(index)) else {
        return;
    };
    let wal_bytes = directory_bytes(root).3;
    let extension_sizes = extension_bytes(root);
    let has_sidecars = extension_sizes
        .iter()
        .any(|(extension, bytes)| matches!(extension.as_str(), "idx" | "run") && *bytes > 0);
    if wal_bytes > EMPTY_SHARED_WAL_LEN || !has_sidecars {
        println!(
            "multi_room_fetch_paged_skipped reason={} wal_bytes={wal_bytes} has_sidecars={has_sidecars}",
            if wal_bytes > EMPTY_SHARED_WAL_LEN {
                "shared_wal_pending"
            } else {
                "no_sealed_index_sidecars"
            }
        );
        return;
    }
    let db = open_benchmark_database(root.to_path_buf());
    let (topo_index, _) = open_event_timeline_indexes(db.event_dag().as_ref());
    let id_page = topo_index
        .page(&selected.scope, None, true, 128)
        .expect("select paged fetch ids");
    let ids = id_page
        .entries
        .iter()
        .map(|entry| entry.event_node_id)
        .collect::<Vec<_>>();
    drop(db);

    let layout = DatabaseLayout::open_read_only(root.to_path_buf())
        .expect("open benchmark layout read-only");
    let pool_path = layout
        .pool_dir_read_only(ShardType::EventDag)
        .expect("find event DAG pool");
    let config = RuntimeConfig {
        index_residency: Some(IndexResidency::Paged),
        ..RuntimeConfig::default()
    };
    for scope in [
        EvictScope::All,
        EvictScope::PacksOnly,
        EvictScope::IndexOnly,
    ] {
        let mut timings = Vec::with_capacity(samples);
        let mut last_io = IoSnapshot::default();
        let mut last_open_io = IoSnapshot::default();
        let mut returned = 0usize;
        let mut unmapped_bytes = 0u64;
        for _ in 0..samples {
            evict(root, scope).expect("evict before paged fetch open");
            let open_before = IoSnapshot::read();
            let store = PackfileStorage::open_read_only_with_config(pool_path.clone(), &config)
                .expect("open paged event DAG pool for fetch");
            last_open_io = IoSnapshot::read().delta(open_before);
            unmapped_bytes = store.unmap_shards();
            evict(root, scope).expect("evict files after paged fetch open");
            let before = IoSnapshot::read();
            let started = Instant::now();
            let payloads = store
                .get_many(&selected.collection, &ids)
                .expect("paged cold payload fetch");
            timings.push(started.elapsed());
            last_io = IoSnapshot::read().delta(before);
            returned = payloads
                .iter()
                .flatten()
                .map(|record| record.bytes.len())
                .fold(0usize, usize::saturating_add);
        }
        let min_us = min(&timings).as_secs_f64() * 1e6;
        let median_us = median(&mut timings).as_secs_f64() * 1e6;
        let amplification_bp = if returned == 0 {
            0
        } else {
            last_io
                .read_bytes
                .saturating_mul(10_000)
                .checked_div(u64::try_from(returned).unwrap_or(u64::MAX).max(1))
                .unwrap_or(0)
        };
        println!(
            "multi_room_fetch_paged evict={} count={} samples={} min_us={min_us:.1} median_us={median_us:.1} unmapped_bytes={unmapped_bytes} syscr={} rchar={} disk_read_bytes={} open_read_bytes={} open_majflt={} payload_bytes_returned={returned} read_amp_bp={amplification_bp} majflt={} cgroup_pgmajfault={} open_cgroup_pgmajfault={}",
            scope.label(),
            ids.len(),
            samples,
            last_io.syscr,
            last_io.rchar,
            last_io.read_bytes,
            last_open_io.read_bytes,
            last_open_io.majflt,
            last_io.majflt,
            last_io.pgmajfault.unwrap_or(u64::MAX),
            last_open_io.pgmajfault.unwrap_or(u64::MAX),
        );
    }
}

fn run_fetch_rows(
    mut db: Database,
    root: &Path,
    rooms: &[Room],
    tier: Option<usize>,
    samples: usize,
) -> Database {
    for scope in [
        EvictScope::All,
        EvictScope::PacksOnly,
        EvictScope::IndexOnly,
    ] {
        let Some(room_index) = tier else {
            return db;
        };
        let Some(selected) = rooms.get(room_index) else {
            continue;
        };
        let (topo_index, _stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
        let page = topo_index
            .page(&selected.scope, None, true, 128)
            .expect("warm index page");
        let ids: Vec<NodeId> = page
            .entries
            .iter()
            .map(|entry| entry.event_node_id)
            .collect();
        let mut elapsed_samples = Vec::with_capacity(samples);
        let mut last_io = IoSnapshot::default();
        let mut last_open_io = IoSnapshot::default();
        let mut returned = 0usize;
        for _ in 0..samples {
            drop(db);
            evict(root, scope).expect("evict selected files before fetch");
            let open_before = IoSnapshot::read();
            db = open_benchmark_database(root.to_path_buf());
            let open_io = IoSnapshot::read().delta(open_before);
            last_open_io = open_io;
            evict(root, scope).expect("evict selected files after open");
            let before = IoSnapshot::read();
            let started = Instant::now();
            let payloads = db
                .pool(ShardType::EventDag)
                .get_many(&selected.collection, &ids)
                .expect("cold payload fetch");
            elapsed_samples.push(started.elapsed());
            last_io = IoSnapshot::read().delta(before);
            returned = payloads
                .iter()
                .flatten()
                .map(|record| record.bytes.len())
                .fold(0usize, usize::saturating_add);
        }
        let min_us = min(&elapsed_samples).as_secs_f64() * 1e6;
        let median_us = median(&mut elapsed_samples).as_secs_f64() * 1e6;
        let amplification_bp = if returned == 0 {
            0
        } else {
            last_io
                .read_bytes
                .saturating_mul(10_000)
                .checked_div(u64::try_from(returned).unwrap_or(u64::MAX).max(1))
                .unwrap_or(0)
        };
        println!(
            "multi_room_fetch evict={} count={} samples={} min_us={min_us:.1} median_us={median_us:.1} syscr={} rchar={} disk_read_bytes={} open_read_bytes={} open_majflt={} payload_bytes_returned={returned} read_amp_bp={amplification_bp} majflt={} cgroup_pgmajfault={} open_cgroup_pgmajfault={}",
            scope.label(),
            ids.len(),
            samples,
            last_io.syscr,
            last_io.rchar,
            last_io.read_bytes,
            last_open_io.read_bytes,
            last_open_io.majflt,
            last_io.majflt,
            last_io.pgmajfault.unwrap_or(u64::MAX),
            last_open_io.pgmajfault.unwrap_or(u64::MAX),
        );
    }

    db
}

fn run_mutations(db: &Database, rooms: &[Room], tier: Option<usize>, payload_min: usize) {
    let Some(largest) = tier else {
        return;
    };
    let Some(largest_room) = rooms.get(largest) else {
        return;
    };
    let (topo_index, stream_index) = open_event_timeline_indexes(db.event_dag().as_ref());
    let mut promotion_times = Vec::new();
    let mut sequence = u64::from(u32::MAX);
    let mut promotions = Vec::new();
    for _ in 0..20 {
        sequence = sequence.saturating_add(1);
        let node = node_id(sequence);
        let entry = EventTimelineEntry::new(
            largest_room.scope,
            i64::try_from(sequence).unwrap_or(i64::MAX),
            0,
            i64::try_from(sequence).unwrap_or(i64::MAX),
            node,
        );
        promotions.push((entry, node));
    }
    let seed = db.begin_transaction();
    for (_, node) in &promotions {
        seed.put(
            ShardType::EventDag,
            largest_room.collection,
            *node,
            &NodeData::new(payload(sequence, payload_min)),
        )
        .expect("seed outlier payload");
    }
    seed.commit().expect("commit outlier payloads");
    for (entry, _) in promotions {
        let started = Instant::now();
        let transaction = db.begin_transaction();
        stage_promote_outlier_to_timeline(
            &transaction,
            ShardType::EventDag,
            largest_room.collection,
            &topo_index,
            &stream_index,
            &[entry],
        )
        .expect("stage promotion");
        transaction.commit().expect("commit promotion");
        promotion_times.push(started.elapsed());
    }
    let page = topo_index
        .page(&largest_room.scope, None, true, 128)
        .expect("mutation source page");
    let remove_entries: Vec<EventTimelineEntry> =
        page.entries.iter().map(|key| key.entry()).collect();
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
        &largest_room.scope,
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

fn select_tiers(rooms: &[Room]) -> Vec<usize> {
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
            if !tiers.contains(index) {
                tiers.push(*index);
            }
        }
    }
    if let Some(index) = ranked.first() {
        if !tiers.contains(index) {
            tiers.push(*index);
        }
    }
    tiers
}

fn main() {
    let root = bench_root();
    let open_only = std::env::var_os("MTXDB_MULTI_OPEN_ONLY").is_some();
    let page_only = std::env::var_os("MTXDB_MULTI_PAGE_ONLY").is_some();
    if !open_only {
        let _ = std::fs::remove_dir_all(&root);
    }
    let _ = std::fs::create_dir_all(&root);
    warn_if_tmpfs(&root);
    let events = env_usize("MTXDB_MULTI_EVENTS", 1_000_000);
    let room_count = env_usize("MTXDB_MULTI_ROOMS", 1_000);
    let batch = env_usize("MTXDB_MULTI_BATCH", 64).max(1);
    let samples = env_usize("MTXDB_TIMELINE_SAMPLES", 10).max(1);
    let payload_min = env_usize("MTXDB_PAYLOAD_MIN", PAYLOAD_MIN);
    let payload_max = env_usize("MTXDB_PAYLOAD_MAX", PAYLOAD_MAX);
    if payload_min > payload_max {
        eprintln!("timeline_multi_room: MTXDB_PAYLOAD_MIN exceeds MTXDB_PAYLOAD_MAX");
        std::process::exit(2);
    }

    let manifest = std::env::var_os("MTXDB_TIMELINE_EXPORT").map(PathBuf::from);
    let (mut rooms, counts, manifest_events) = if let Some(path) = manifest.as_deref() {
        match manifest_rooms(path) {
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
        }
    } else {
        let (rooms, counts) = synthetic_rooms(room_count, events);
        (rooms, counts, None)
    };
    let total_events: usize = counts.iter().sum();
    if open_only {
        let db = open_benchmark_database(root.clone());
        println!(
            "multi_room_open_only events={total_events} rooms={}",
            rooms.len()
        );
        print_memory_diagnostics(&db, &root, &rooms);
        print_extension_breakdown(&root, "open_only");
        return;
    }
    let db = open_benchmark_database(root.clone());

    let write_started = Instant::now();
    if let Some(manifest_events) = manifest_events.as_deref() {
        write_manifest(&db, &rooms, manifest_events, batch);
    } else {
        write_synthetic(&db, &mut rooms, &counts, batch, payload_min, payload_max);
    }
    let (frames_before_checkpoint, frames_after_checkpoint) = checkpoint_all_pools(&db, &root);
    let write_ms = write_started.elapsed().as_secs_f64() * 1e3;
    println!(
        "multi_room_load events={total_events} rooms={} batch={batch} write_ms={write_ms:.1} payload_min={payload_min} payload_max={payload_max} raw_frames_before_checkpoint={frames_before_checkpoint} raw_frames_after_checkpoint={frames_after_checkpoint} wal_bytes={}",
        rooms.len(),
        directory_bytes(&root).3,
    );
    print_memory_diagnostics(&db, &root, &rooms);
    print_extension_breakdown(&root, "load");
    print_ordered_write_metrics(&db, &root, total_events, batch);

    let tiers = select_tiers(&rooms);

    let db = reopen_benchmark_database(&root, db);
    print_memory_diagnostics(&db, &root, &rooms);
    drop(db);
    run_paged_page_rows(&root, &rooms, &tiers, samples);
    if !page_only {
        run_paged_fetch_rows(&root, &rooms, tiers.first().copied(), samples);
    }
    if std::env::var_os("MTXDB_PAGE_ADVICE_AB").is_some() {
        run_sequential_get_rows(&root, &rooms, samples);
        run_disk_order_get_rows(&root, &rooms, samples);
        run_adjacent_get_rows(&root, &rooms, samples);
    }
    let mut db = open_benchmark_database(root.clone());
    db = run_page_rows(db, &root, &rooms, &tiers, samples);
    if page_only {
        return;
    }
    db = run_fetch_rows(db, &root, &rooms, tiers.first().copied(), samples);
    print_memory_diagnostics(&db, &root, &rooms);
    print_extension_breakdown(&root, "after_reads");
    run_mutations(&db, &rooms, tiers.first().copied(), payload_min);
    print_extension_breakdown(&root, "after_mutations");
}

fn tier_label(rank: usize) -> &'static str {
    match rank {
        0 => "large",
        1 => "medium",
        _ => "small",
    }
}
