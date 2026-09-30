//! Matched Matrix-shaped workload for native mtxdb versus SQLite.
//!
//! Two scenarios, each measured only on backends that can do them honestly:
//!
//! - **point lookup**: event by ID, and batch of event IDs, on mtxdb and
//!   SQLite;
//! - **timeline pagination**: one room's events ordered by depth, paged
//!   forward and backward with a limit, on SQLite only.  mtxdb has no native
//!   room/order range API yet, so it does not implement the timeline trait and
//!   no native row is emitted — the SQLite row is the baseline that native
//!   range reads must later compete against, indexed on `(room_id, depth)`.
//!
//! Each backend is measured three ways so cold and warm costs stay separate:
//!
//! - **warm, already-open**: reads right after load, while the handle and the
//!   page cache are both hot;
//! - **cold open**: the handle is dropped, the backend's own files are
//!   evicted with `vmtouch -e`, and the reopen plus the first reads are timed;
//! - **warm open**: the handle is reopened again with no eviction.
//!
//! `vmtouch` only drops *clean, unmapped* pages, so the backend handle must be
//! dropped first. When `vmtouch` is absent, or the scratch root is RAM-backed,
//! the "cold" fields are actually warm and `EVICTED` says so.
//!
//! Env knobs (all optional):
//! - `MTXDB_BENCH_ROOT` redirects scratch data off a RAM-backed tmpfs.
//! - `MTXDB_BENCH_TIMELINE_DIAGNOSTICS=1` additionally prints the
//!   `EXPLAIN QUERY PLAN` for both directions and sweeps page sizes, emitted
//!   as `diag:` lines that the regression parser deliberately ignores.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::pedantic,
    clippy::uninlined_format_args
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bytes::Bytes;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;
use rusqlite::{params, params_from_iter, Connection};

const ROOM_COUNT: usize = 8;
const EVENTS_PER_ROOM: usize = 2_000;
const BATCH_SIZE: usize = 32;
const LOOKUPS: usize = 10_000;
const TIMELINE_PAGE: usize = 100;
/// Page sizes swept only under `MTXDB_BENCH_TIMELINE_DIAGNOSTICS`.
const TIMELINE_DIAG_PAGE_SIZES: [usize; 6] = [25, 50, 100, 250, 500, 1_000];
const SEED: u64 = 0x4d_54_58_44_42_2d_4d_58;

#[derive(Clone)]
struct Event {
    id: NodeId,
    room: [u8; 16],
    depth: u64,
    payload: Vec<u8>,
}

struct Dataset {
    events: Vec<Event>,
}

impl Dataset {
    fn build() -> Self {
        let mut events = Vec::with_capacity(ROOM_COUNT * EVENTS_PER_ROOM);
        // Depth-major so rooms are interleaved: consecutive insertions belong
        // to different rooms, which is what makes a room/depth index matter.
        // `event_index` stays unique, so event ids are too.
        for depth in 0..EVENTS_PER_ROOM {
            for room_index in 0..ROOM_COUNT {
                let event_index = depth * ROOM_COUNT + room_index;
                let event_id = id(SEED ^ event_index as u64, 0x6576656e74);
                let mut payload = format!(
                    r#"{{"event_id":"{event_index:032x}","room_id":"{room_index:04x}","type":"m.room.message","depth":{depth}}}"#
                )
                .into_bytes();
                payload.extend_from_slice(&event_id);
                events.push(Event {
                    id: event_id,
                    room: room_id(room_index),
                    depth: depth as u64,
                    payload,
                });
            }
        }
        Self { events }
    }

    /// Every room id, in room-index order.
    fn rooms(&self) -> Vec<[u8; 16]> {
        (0..ROOM_COUNT).map(room_id).collect()
    }

    fn sample_ids(&self) -> Vec<NodeId> {
        (0..LOOKUPS)
            // A coprime stride gives a mixed, repeat-free sample.  Keeping
            // IDs unique makes point and SQL-IN batch hit counts comparable.
            .map(|i| self.events[i * 7_919 % self.events.len()].id)
            .collect()
    }
}

trait Backend {
    fn name(&self) -> &'static str;
    fn load(&mut self, dataset: &Dataset);
    /// Drop the handle so its files are unmapped and `vmtouch` can evict them.
    fn close(&mut self);
    /// Reopen the backend; timed for the cold/warm open phases.
    fn open(&mut self);
    /// The backend-private files measured for footprint and eviction.
    fn disk_path(&self) -> PathBuf;
    fn lookup(&self, id: NodeId) -> Option<Vec<u8>>;
    fn lookup_many(&self, ids: &[NodeId]) -> usize;
}

struct MtxdbBackend {
    root: PathBuf,
    store: Option<PackfileStorage>,
}

impl MtxdbBackend {
    fn new(root: PathBuf) -> Self {
        Self { root, store: None }
    }

    fn pool(&self) -> PathBuf {
        self.root.join("pool")
    }
}

impl Backend for MtxdbBackend {
    fn name(&self) -> &'static str {
        "mtxdb"
    }

    fn load(&mut self, dataset: &Dataset) {
        let store = PackfileStorage::open_with_cache(self.pool(), 0).unwrap();
        let entries = dataset
            .events
            .iter()
            .map(|event| {
                (
                    event.id,
                    NodeData::new(Bytes::copy_from_slice(&event.payload)),
                )
            })
            .collect::<Vec<_>>();
        store
            .put_many(&[0u8; 16], &entries)
            .expect("write Matrix event fixture");
        store.sync_all().unwrap();
        self.store = Some(store);
    }

    fn close(&mut self) {
        self.store = None;
    }

    fn open(&mut self) {
        self.store = Some(PackfileStorage::open_with_cache(self.pool(), 0).unwrap());
    }

    fn disk_path(&self) -> PathBuf {
        self.root.clone()
    }

    fn lookup(&self, id: NodeId) -> Option<Vec<u8>> {
        self.store
            .as_ref()
            .unwrap()
            .get(&[0u8; 16], &id)
            .unwrap()
            .map(|node| node.bytes.to_vec())
    }

    fn lookup_many(&self, ids: &[NodeId]) -> usize {
        self.store
            .as_ref()
            .unwrap()
            .get_many(&[0u8; 16], ids)
            .unwrap()
            .into_iter()
            .filter(Option::is_some)
            .count()
    }
}

struct SqliteBackend {
    path: PathBuf,
    conn: Option<Connection>,
    /// Create the `(room_id, depth)` index. Only the timeline scenario wants
    /// it; the point scenario leaves it off so neither side pays for an index
    /// its workload never uses.
    with_index: bool,
}

impl SqliteBackend {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            conn: None,
            with_index: false,
        }
    }

    fn indexed(mut self) -> Self {
        self.with_index = true;
        self
    }
}

impl Backend for SqliteBackend {
    fn name(&self) -> &'static str {
        "sqlite"
    }

    fn load(&mut self, dataset: &Dataset) {
        let conn = Connection::open(&self.path).unwrap();
        let mut schema = String::from(
            "PRAGMA journal_mode=DELETE;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE events(
                 event_id BLOB PRIMARY KEY,
                 room_id BLOB NOT NULL,
                 depth INTEGER NOT NULL,
                 payload BLOB NOT NULL
             ) WITHOUT ROWID;",
        );
        if self.with_index {
            schema.push_str("CREATE INDEX events_room_depth ON events(room_id, depth);");
        }
        conn.execute_batch(&schema).unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        {
            let mut statement = tx
                .prepare(
                    "INSERT INTO events(event_id, room_id, depth, payload) VALUES (?1, ?2, ?3, ?4)",
                )
                .unwrap();
            for event in &dataset.events {
                statement
                    .execute(params![event.id, event.room, event.depth, event.payload])
                    .unwrap();
            }
        }
        tx.commit().unwrap();
        self.conn = Some(conn);
    }

    fn close(&mut self) {
        self.conn = None;
    }

    fn open(&mut self) {
        self.conn = Some(Connection::open(&self.path).unwrap());
    }

    fn disk_path(&self) -> PathBuf {
        self.path
            .parent()
            .map_or_else(|| self.path.clone(), Path::to_path_buf)
    }

    fn lookup(&self, id: NodeId) -> Option<Vec<u8>> {
        self.conn
            .as_ref()
            .unwrap()
            .query_row(
                "SELECT payload FROM events WHERE event_id = ?1",
                params![id],
                |row| row.get(0),
            )
            .ok()
    }

    fn lookup_many(&self, ids: &[NodeId]) -> usize {
        let placeholders = vec!["?"; ids.len()].join(",");
        let sql = format!("SELECT event_id FROM events WHERE event_id IN ({placeholders})");
        self.conn
            .as_ref()
            .unwrap()
            .prepare(&sql)
            .unwrap()
            .query_map(params_from_iter(ids.iter().map(|id| id.to_vec())), |_| {
                Ok(())
            })
            .unwrap()
            .count()
    }
}

/// A backend with a room timeline: events in one room ordered by depth, paged
/// by an inclusive depth cursor. Deliberately implemented only by engines that
/// have a native range index; mtxdb's does not exist yet, so it does not
/// implement this trait and no native timeline row is emitted.
trait TimelineBackend: Backend {
    /// Up to `limit` depths in `room`, inclusive of `cursor`, in `forward`
    /// (ascending) or backward (descending) order.
    fn page(&self, room: [u8; 16], cursor: u64, forward: bool, limit: usize) -> Vec<u64>;

    /// Human-readable `EXPLAIN QUERY PLAN` lines for one direction, if the
    /// engine exposes them. Defaults to none for engines that don't.
    fn explain(&self, _room: [u8; 16], _forward: bool) -> Vec<String> {
        Vec::new()
    }
}

impl TimelineBackend for SqliteBackend {
    fn page(&self, room: [u8; 16], cursor: u64, forward: bool, limit: usize) -> Vec<u64> {
        let sql = if forward {
            "SELECT depth FROM events WHERE room_id = ?1 AND depth >= ?2 \
             ORDER BY depth ASC LIMIT ?3"
        } else {
            "SELECT depth FROM events WHERE room_id = ?1 AND depth <= ?2 \
             ORDER BY depth DESC LIMIT ?3"
        };
        let conn = self.conn.as_ref().unwrap();
        let mut statement = conn.prepare_cached(sql).unwrap();
        let rows = statement
            .query_map(
                params![room, to_sql_int(cursor), to_sql_int(limit as u64)],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        rows.map(|row| u64::try_from(row.unwrap()).unwrap())
            .collect()
    }

    fn explain(&self, room: [u8; 16], forward: bool) -> Vec<String> {
        let sql = if forward {
            "EXPLAIN QUERY PLAN SELECT depth FROM events \
             WHERE room_id = ?1 AND depth >= ?2 ORDER BY depth ASC LIMIT ?3"
        } else {
            "EXPLAIN QUERY PLAN SELECT depth FROM events \
             WHERE room_id = ?1 AND depth <= ?2 ORDER BY depth DESC LIMIT ?3"
        };
        let conn = self.conn.as_ref().unwrap();
        let mut statement = conn.prepare(sql).unwrap();
        let rows = statement
            .query_map(params![room, 0i64, 100i64], |row| row.get::<_, String>(3))
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
    }
}

fn to_sql_int(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn id(seed: u64, domain: u64) -> NodeId {
    let a = splitmix64(seed ^ domain);
    let b = splitmix64(a ^ 0x7372_9a1e_4288_1f7d);
    let mut result = [0u8; 16];
    result[..8].copy_from_slice(&a.to_le_bytes());
    result[8..].copy_from_slice(&b.to_le_bytes());
    result
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn room_id(room_index: usize) -> [u8; 16] {
    id(SEED ^ room_index as u64, 0x726f6f6d)
}

/// Time point lookups, returning `(hits, microseconds per lookup)`.
fn time_point<B: Backend>(backend: &B, ids: &[NodeId]) -> (usize, f64) {
    let started = Instant::now();
    let found = ids
        .iter()
        .filter(|id| backend.lookup(**id).is_some())
        .count();
    let per_lookup = started.elapsed().as_secs_f64() * 1e6 / ids.len() as f64;
    (found, per_lookup)
}

/// Time batched lookups, returning `(hits, microseconds per lookup)`.
fn time_batch<B: Backend>(backend: &B, ids: &[NodeId]) -> (usize, f64) {
    let started = Instant::now();
    let mut found = 0;
    for chunk in ids.chunks(BATCH_SIZE) {
        found += backend.lookup_many(chunk);
    }
    let per_lookup = started.elapsed().as_secs_f64() * 1e6 / ids.len() as f64;
    (found, per_lookup)
}

fn run<B: Backend>(mut backend: B, dataset: &Dataset, ids: &[NodeId]) {
    let started = Instant::now();
    backend.load(dataset);
    let write_ms = started.elapsed().as_secs_f64() * 1e3;
    let path = backend.disk_path();
    let bytes = directory_bytes(&path);

    // Warm read, handle and page cache both hot right after load.
    let (warm_found, warm_point_us) = time_point(&backend, ids);
    let (warm_batch_found, warm_batch_us) = time_batch(&backend, ids);

    // Cold open: drop the mapping, evict only this backend's files, reopen.
    backend.close();
    let evicted = evict_dir(&path);
    let started = Instant::now();
    backend.open();
    let cold_open_ms = started.elapsed().as_secs_f64() * 1e3;
    let (cold_found, cold_point_us) = time_point(&backend, ids);
    let (cold_batch_found, cold_batch_us) = time_batch(&backend, ids);

    // Warm open with the same (now hot) cache, no eviction.
    backend.close();
    let started = Instant::now();
    backend.open();
    let warm_open_ms = started.elapsed().as_secs_f64() * 1e3;

    let total = ids.len();
    assert_eq!(
        warm_found, total,
        "warm point lookups must find every event"
    );
    assert_eq!(
        warm_batch_found, total,
        "warm batch lookups must find every event"
    );
    assert_eq!(
        cold_found, total,
        "cold point lookups must find every event"
    );
    assert_eq!(
        cold_batch_found, total,
        "cold batch lookups must find every event"
    );

    println!(
        "bench: matrix BACKEND={} ROOMS={} EVENTS={} LOOKUPS={} BATCH={} \
         WRITE_MS={:.3} BYTES={} EVICTED={} \
         WARM_OPEN_MS={:.3} COLD_OPEN_MS={:.3} \
         WARM_POINT_US={:.3} WARM_BATCH_US={:.3} \
         COLD_POINT_US={:.3} COLD_BATCH_US={:.3}",
        backend.name(),
        ROOM_COUNT,
        dataset.events.len(),
        total,
        BATCH_SIZE,
        write_ms,
        bytes,
        evicted,
        warm_open_ms,
        cold_open_ms,
        warm_point_us,
        warm_batch_us,
        cold_point_us,
        cold_batch_us,
    );
}

fn elapsed_us(started: Instant, divisor: usize) -> f64 {
    if divisor == 0 {
        0.0
    } else {
        started.elapsed().as_secs_f64() * 1e6 / divisor as f64
    }
}

/// Walk one direction across every room's timeline, one page at a time.
/// Returns `(pages, events, microseconds per page)`.
fn time_direction<B: TimelineBackend>(
    backend: &B,
    rooms: &[[u8; 16]],
    forward: bool,
    page_size: usize,
) -> (usize, usize, f64) {
    let mut pages = 0usize;
    let mut events = 0usize;
    let started = Instant::now();
    for &room in rooms {
        let mut cursor = if forward { 0 } else { u64::MAX };
        loop {
            let page = backend.page(room, cursor, forward, page_size);
            if page.is_empty() {
                break;
            }
            pages += 1;
            events += page.len();
            // The last element is the far edge of the page: the largest depth
            // when descending, the smallest when ascending.
            let edge = page[page.len() - 1];
            if page.len() < page_size {
                break;
            }
            if forward {
                cursor = edge.saturating_add(1);
            } else if edge == 0 {
                break;
            } else {
                cursor = edge - 1;
            }
        }
    }
    (pages, events, elapsed_us(started, pages))
}

/// Opt-in, `diag:`-prefixed diagnostics: the query plans plus a page-size
/// sweep, so B-tree direction cost can be told apart from page-size and
/// first-touch effects. `bench:` regression metrics are untouched.
fn run_timeline_diagnostics<B: TimelineBackend>(backend: &mut B, rooms: &[[u8; 16]], path: &Path) {
    if let Some(&room) = rooms.first() {
        for (label, forward) in [("fwd", true), ("rev", false)] {
            for line in backend.explain(room, forward) {
                println!(
                    "diag: matrix_timeline_plan BACKEND={} DIR={} PLAN={}",
                    backend.name(),
                    label,
                    line
                );
            }
        }
    }

    for &page_size in &TIMELINE_DIAG_PAGE_SIZES {
        let (warm_fwd_pages, _, warm_fwd_us) = time_direction(backend, rooms, true, page_size);
        let (warm_bwd_pages, _, warm_bwd_us) = time_direction(backend, rooms, false, page_size);
        for (label, pages, page_us) in [
            ("fwd", warm_fwd_pages, warm_fwd_us),
            ("rev", warm_bwd_pages, warm_bwd_us),
        ] {
            println!(
                "diag: matrix_timeline_sweep BACKEND={} CACHE=warm PAGE={} DIR={} PAGES={} PAGE_US={:.3}",
                backend.name(),
                page_size,
                label,
                pages,
                page_us
            );
        }

        // Each cold direction gets its own eviction, as in the default row.
        backend.close();
        let fwd_evicted = evict_dir(path);
        backend.open();
        let (cold_fwd_pages, _, cold_fwd_us) = time_direction(backend, rooms, true, page_size);

        backend.close();
        let rev_evicted = evict_dir(path);
        backend.open();
        let (cold_bwd_pages, _, cold_bwd_us) = time_direction(backend, rooms, false, page_size);
        for (label, pages, page_us, evicted) in [
            ("fwd", cold_fwd_pages, cold_fwd_us, fwd_evicted),
            ("rev", cold_bwd_pages, cold_bwd_us, rev_evicted),
        ] {
            println!(
                "diag: matrix_timeline_sweep BACKEND={} CACHE=cold PAGE={} DIR={} PAGES={} PAGE_US={:.3} EVICTED={}",
                backend.name(),
                page_size,
                label,
                pages,
                page_us,
                evicted
            );
        }
    }
}

fn timeline_diagnostics_enabled() -> bool {
    std::env::var_os("MTXDB_BENCH_TIMELINE_DIAGNOSTICS").is_some_and(|value| value != "0")
}

fn run_timeline<B: TimelineBackend>(mut backend: B, dataset: &Dataset) {
    let started = Instant::now();
    backend.load(dataset);
    let write_ms = started.elapsed().as_secs_f64() * 1e3;
    let path = backend.disk_path();
    let bytes = directory_bytes(&path);
    let rooms = dataset.rooms();

    // Warm each direction with no eviction.
    let (warm_fwd_pages, warm_fwd_events, warm_fwd_us) =
        time_direction(&backend, &rooms, true, TIMELINE_PAGE);
    let (warm_bwd_pages, warm_bwd_events, warm_bwd_us) =
        time_direction(&backend, &rooms, false, TIMELINE_PAGE);

    // Cold each direction under its own eviction, so neither inherits the
    // other's first-touch cost. This is what keeps the direction comparison
    // honest; measuring them back-to-back on one cache does not.
    backend.close();
    let mut evicted = evict_dir(&path);
    backend.open();
    let (cold_fwd_pages, cold_fwd_events, cold_fwd_us) =
        time_direction(&backend, &rooms, true, TIMELINE_PAGE);

    backend.close();
    evicted &= evict_dir(&path);
    backend.open();
    let (cold_bwd_pages, cold_bwd_events, cold_bwd_us) =
        time_direction(&backend, &rooms, false, TIMELINE_PAGE);

    let expected = ROOM_COUNT * EVENTS_PER_ROOM;
    assert_eq!(
        warm_fwd_events, expected,
        "warm forward pages must cover every event"
    );
    assert_eq!(
        warm_bwd_events, expected,
        "warm backward pages must cover every event"
    );
    assert_eq!(
        cold_fwd_events, expected,
        "cold forward pages must cover every event"
    );
    assert_eq!(
        cold_bwd_events, expected,
        "cold backward pages must cover every event"
    );
    // Same page count each way is the observable form of "identical event
    // sets in reverse order".
    assert_eq!(
        warm_fwd_pages, warm_bwd_pages,
        "timeline must page identically forward and backward"
    );
    assert_eq!(
        cold_fwd_pages, cold_bwd_pages,
        "cold timeline must page identically forward and backward"
    );

    println!(
        "bench: matrix_timeline BACKEND={} ROOMS={} EVENTS={} PAGE={} \
         WRITE_MS={:.3} BYTES={} EVICTED={} \
         WARM_FWD_PAGES={} WARM_BWD_PAGES={} \
         WARM_FWD_PAGE_US={:.3} WARM_BWD_PAGE_US={:.3} \
         COLD_FWD_PAGE_US={:.3} COLD_BWD_PAGE_US={:.3}",
        backend.name(),
        ROOM_COUNT,
        dataset.events.len(),
        TIMELINE_PAGE,
        write_ms,
        bytes,
        evicted,
        warm_fwd_pages,
        warm_bwd_pages,
        warm_fwd_us,
        warm_bwd_us,
        cold_fwd_us,
        cold_bwd_us,
    );

    if timeline_diagnostics_enabled() {
        run_timeline_diagnostics(&mut backend, &rooms, &path);
    }
}

/// Best-effort page-cache eviction via `vmtouch -e`, which only drops *clean,
/// unmapped* pages of the named path. No root needed; `false` if vmtouch is
/// missing or fails (in which case the "cold" numbers are warm).
fn evict_dir(dir: &Path) -> bool {
    std::process::Command::new("vmtouch")
        .arg("-e")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn vmtouch_on_path() -> bool {
    std::process::Command::new("vmtouch")
        .arg("-h")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Filesystem type of the mount holding `path`, from `/proc/mounts` (the
/// longest mount point that prefixes `path`). `None` if `/proc/mounts` is
/// unreadable or no entry matches (non-Linux).
fn mount_fstype(path: &Path) -> Option<String> {
    let canonical = path.canonicalize().ok()?;
    let mounts = fs::read_to_string("/proc/mounts").ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let mut fields = line.split(' ');
        let _dev = fields.next()?;
        let mount_point = fields.next()?;
        let fstype = fields.next()?;
        let mount_path = Path::new(mount_point);
        if canonical.starts_with(mount_path) {
            let depth = mount_path.components().count();
            let deeper_or_equal = best.as_ref().map_or(true, |(d, _)| depth >= *d);
            if deeper_or_equal {
                best = Some((depth, fstype.to_owned()));
            }
        }
    }
    best.map(|(_, fstype)| fstype)
}

/// True when `path` lives on an in-memory filesystem, where "cold reads" are a
/// contradiction: the data never leaves RAM and neither `drop_caches` nor
/// `vmtouch` can evict it.
fn is_ram_backed(path: &Path) -> bool {
    matches!(
        mount_fstype(path).as_deref(),
        Some("tmpfs" | "ramfs" | "devtmpfs")
    )
}

/// PID-suffixed scratch root; `MTXDB_BENCH_ROOT` redirects it off a RAM-backed
/// tmpfs so the cold phases mean something.
fn bench_root() -> PathBuf {
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    base.join(format!("mtxdb_matrix_workload_{}", std::process::id()))
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                directory_bytes(&path)
            } else {
                entry.metadata().map_or(0, |m| m.len())
            }
        })
        .sum()
}

fn main() {
    let root = bench_root();
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    if !vmtouch_on_path() {
        eprintln!(
            "⚠ WARNING: `vmtouch` not found on PATH — COLD_* fields measure a WARM cache. \
             Install it (e.g. `apt install vmtouch`) for real cold-cache numbers."
        );
    } else if is_ram_backed(&root) {
        eprintln!(
            "⚠ WARNING: scratch root {} is RAM-backed — cold reads are impossible there. \
             Set MTXDB_BENCH_ROOT to a disk-backed path.",
            root.display()
        );
    }

    let dataset = Dataset::build();
    let ids = dataset.sample_ids();

    run(MtxdbBackend::new(root.join("mtxdb")), &dataset, &ids);
    // Keep SQLite in its own directory so its footprint is measured without
    // the mtxdb pool's bytes.
    let sqlite_dir = root.join("sqlite");
    fs::create_dir_all(&sqlite_dir).unwrap();
    run(
        SqliteBackend::new(sqlite_dir.join("events.sqlite")),
        &dataset,
        &ids,
    );

    // Timeline pagination: SQLite baseline, indexed on (room_id, depth). mtxdb
    // is intentionally absent until a native room/order range API exists.
    let timeline_dir = root.join("sqlite_timeline");
    fs::create_dir_all(&timeline_dir).unwrap();
    run_timeline(
        SqliteBackend::new(timeline_dir.join("events.sqlite")).indexed(),
        &dataset,
    );
    eprintln!(
        "note: mtxdb timeline not emitted — no native room/order range API yet; \
         the matrix_timeline row is the SQLite baseline it must compete against."
    );

    let _ = fs::remove_dir_all(&root);
}
