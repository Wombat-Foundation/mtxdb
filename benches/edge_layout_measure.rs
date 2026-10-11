//! What the short-id edge store costs on disk for a real room.
//!
//! Reads a tab-separated room (`event_id, prev ids, auth ids, relation target`,
//! comma-separated lists; see `MTXDB_EL_TSV`), records it through
//! `ShortIdIndex::record_events` in batches of 1000, forces a full index
//! checkpoint, and prints the bytes the store occupies. Variants isolate the
//! cost: owners only (no edges), prev only, prev + auth, and all three
//! families. The difference between a variant and the owners-only store is what
//! the edge lists (and the target ids they allocate) add.
//!
//! A last section writes the same room in a prototype chunked layout through the
//! same `Database`, so pack framing and index costs are measured the same way.
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_measure
//! ```

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Path, PathBuf};

use mtxdb::database::Database;
use mtxdb::layout::ShardType;
use mtxdb::short_id::{BatchEvent, EdgeFamily, EdgeKey, FamilyEdges, ShortIdIndex};
use mtxdb::storage::{NodeData, StorageEngine};

const SCOPE: [u8; 16] = [0x71; 16];
const CHUNK_COLLECTION: [u8; 16] = [0x72; 16];
const BATCH: usize = 1000;
const SYNC_EVERY: usize = 20;
const CHUNK: usize = 256;
const PREV: EdgeFamily = EdgeFamily::plain(1);
const AUTH: EdgeFamily = EdgeFamily::plain(2);
const REL: EdgeFamily = EdgeFamily::typed(3);
const MIB: f64 = 1_048_576.0;
const TWO_POW_32: f64 = 4_294_967_296.0;
const TWO_POW_32_INT: u64 = 4_294_967_296;

struct Row {
    id: String,
    prev: Vec<String>,
    auth: Vec<String>,
    rel: Option<String>,
}

/// Which edge families a variant records.
#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    prev: bool,
    auth: bool,
    rel: bool,
}

/// Bytes of the store's pack, checkpoint and delta files.
#[derive(Clone, Copy)]
struct Sizes {
    packs: u64,
    checkpoint: u64,
    delta: u64,
}

impl Sizes {
    fn total(self) -> u64 {
        self.packs
            .saturating_add(self.checkpoint)
            .saturating_add(self.delta)
    }
}

/// A `u64` as `f64`, exact below 2^53 (far above any size measured here).
fn num(n: u64) -> f64 {
    let high = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(n & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * TWO_POW_32 + f64::from(low)
}

fn len_f64(n: usize) -> f64 {
    num(u64::try_from(n).unwrap_or(u64::MAX))
}

fn mib(n: u64) -> f64 {
    num(n) / MIB
}

/// `n` bytes per `events` events.
fn per(n: u64, events: usize) -> f64 {
    num(n) / len_f64(events.max(1))
}

fn split(field: &str) -> Vec<String> {
    field
        .split(',')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn load(path: &Path, limit: usize) -> Vec<Row> {
    std::fs::read_to_string(path)
        .expect("read tsv")
        .lines()
        .take(limit)
        .filter_map(|line| {
            let mut cols = line.split('\t');
            let id = cols.next()?.to_owned();
            let prev = split(cols.next()?);
            let auth = split(cols.next()?);
            let rel = cols.next().filter(|r| !r.is_empty()).map(str::to_owned);
            Some(Row {
                id,
                prev,
                auth,
                rel,
            })
        })
        .collect()
}

/// Bytes of every file under `dir` whose name contains `needle`.
fn tree_bytes(dir: &Path, needle: &str) -> io::Result<u64> {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            total = total.saturating_add(tree_bytes(&path, needle)?);
        } else if entry.file_name().to_string_lossy().contains(needle) {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn sizes(dir: &Path) -> Sizes {
    let measure = |needle: &str| tree_bytes(dir, needle).expect("measure directory size");
    Sizes {
        packs: measure(".pack"),
        checkpoint: measure("checkpoint"),
        delta: measure("delta"),
    }
}

fn pack_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            pack_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "pack") {
            out.push(path);
        }
    }
    Ok(())
}

/// What a frame's payload is, by its leading magic bytes.
fn kind_of(data: &[u8]) -> &'static str {
    match data.get(..4) {
        Some(b"SIDF") => "forward (key -> id)",
        Some(b"SIDR") => "reverse (id -> key)",
        Some(b"SIDE") => "edge list",
        Some(b"OWNL") => "owner log",
        Some(b"SIDC") => "counter",
        Some(b"PEDG") => "chunk: edge lists",
        Some(b"PREV") => "chunk: id -> key",
        Some(b"PFWD") => "run: key hash -> id",
        _ => "other",
    }
}

#[derive(Default)]
struct KindTotals {
    records: u64,
    payload: u64,
    frames: u64,
}

/// Classify every frame in the store's packs and report record counts, payload
/// bytes and on-disk frame bytes per kind.
fn breakdown(dir: &Path, events: usize) {
    let mut packs = Vec::new();
    pack_files(dir, &mut packs).expect("list packs");
    let mut kinds: BTreeMap<&'static str, KindTotals> = BTreeMap::new();
    for pack in &packs {
        let len = std::fs::metadata(pack).expect("stat pack").len();
        let records: Vec<(u64, mtxdb::packfile::Record)> = mtxdb::packfile::scan_records_iter(pack)
            .expect("scan")
            .map(|item| item.expect("record"))
            .collect();
        for (i, (offset, record)) in records.iter().enumerate() {
            let end = records.get(i.saturating_add(1)).map_or(len, |next| next.0);
            let totals = kinds.entry(kind_of(&record.data)).or_default();
            totals.records = totals.records.saturating_add(1);
            totals.payload = totals
                .payload
                .saturating_add(u64::try_from(record.data.len()).unwrap_or(u64::MAX));
            totals.frames = totals.frames.saturating_add(end.saturating_sub(*offset));
        }
    }
    println!("  pack frames by kind ({events} events):");
    let mut all_records = 0_u64;
    for (kind, totals) in &kinds {
        all_records = all_records.saturating_add(totals.records);
        println!(
            "    {kind:<22} {:>8} records  payload {:>6.1} B/event ({:>5.1} B/rec)  frames {:>6.1} B/event ({:>5.1} B/rec)",
            totals.records,
            per(totals.payload, events),
            num(totals.payload) / num(totals.records.max(1)),
            per(totals.frames, events),
            num(totals.frames) / num(totals.records.max(1)),
        );
    }
    let on_disk = sizes(dir);
    println!(
        "    {all_records} records; checkpoint {:.1} B/record, delta log {:.1} B/record",
        num(on_disk.checkpoint) / num(all_records.max(1)),
        num(on_disk.delta) / num(all_records.max(1)),
    );
}

/// One owner's edge keys per family.
struct Lists<'a> {
    prev: Vec<EdgeKey<'a>>,
    auth: Vec<EdgeKey<'a>>,
    rel: Vec<EdgeKey<'a>>,
}

fn lists_for<'a>(row: &'a Row, variant: Variant) -> Lists<'a> {
    let plain = |ids: &'a [String]| -> Vec<EdgeKey<'a>> {
        ids.iter().map(|t| EdgeKey::plain(t.as_bytes())).collect()
    };
    Lists {
        prev: if variant.prev {
            plain(&row.prev)
        } else {
            Vec::new()
        },
        auth: if variant.auth {
            plain(&row.auth)
        } else {
            Vec::new()
        },
        rel: if variant.rel {
            row.rel
                .iter()
                .map(|t| EdgeKey {
                    target: t.as_bytes(),
                    kind: 1,
                })
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn families<'a>(lists: &'a Lists<'a>) -> Vec<FamilyEdges<'a>> {
    let mut out = Vec::new();
    for (family, edges) in [(PREV, &lists.prev), (AUTH, &lists.auth), (REL, &lists.rel)] {
        if !edges.is_empty() {
            out.push(FamilyEdges { family, edges });
        }
    }
    out
}

/// Record every row through `ShortIdIndex`; returns the number of edges written.
fn record_rows(db: &Database, rows: &[Row], variant: Variant) -> usize {
    let scope = ShortIdIndex::new(ShardType::Edges, SCOPE);
    let mut edge_count = 0_usize;
    for (n, chunk) in rows.chunks(BATCH).enumerate() {
        let lists: Vec<Lists<'_>> = chunk.iter().map(|row| lists_for(row, variant)).collect();
        let fams: Vec<Vec<FamilyEdges<'_>>> = lists.iter().map(families).collect();
        for l in &lists {
            edge_count = edge_count
                .saturating_add(l.prev.len())
                .saturating_add(l.auth.len())
                .saturating_add(l.rel.len());
        }
        let events: Vec<BatchEvent<'_>> = chunk
            .iter()
            .zip(&fams)
            .map(|(row, f)| BatchEvent {
                owner: row.id.as_bytes(),
                families: f,
                owner_payload: None,
            })
            .collect();
        scope.record_events(db, &events).expect("record");
        if n.saturating_add(1) % SYNC_EVERY == 0 {
            db.edges().sync_all().expect("sync");
        }
    }
    db.edges().sync_all().expect("sync");
    edge_count
}

fn run(base: &Path, rows: &[Row], variant: Variant) -> u64 {
    let label = variant.label;
    let dir = base.join(format!(
        "mtxdb_bench_edgelayout_{}_{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let edge_count = record_rows(&db, rows, variant);
    let before = sizes(&dir);
    db.edges()
        .force_index_checkpoint()
        .expect("force checkpoint");
    let after = sizes(&dir);
    drop(db);
    println!(
        "[{label}] before forced checkpoint: packs {:.2} MiB, checkpoint {:.2}, delta {:.2} | after: packs {:.2}, checkpoint {:.2}, delta {:.2}",
        mib(before.packs),
        mib(before.checkpoint),
        mib(before.delta),
        mib(after.packs),
        mib(after.checkpoint),
        mib(after.delta),
    );
    if variant.rel {
        breakdown(&dir, rows.len());
    }
    let total = after.total();
    println!(
        "  total {:.2} MiB = {:.1} B/event, {:.1} B/edge ({edge_count} edges)",
        mib(total),
        per(total, rows.len()),
        num(total) / len_f64(edge_count.max(1)),
    );
    let _ = std::fs::remove_dir_all(&dir);
    total
}

/// A spread 16-byte record id for a prototype record, so ids do not share a prefix.
fn spread(kind: u8, n: u32) -> [u8; 16] {
    let mut x = u64::from(kind)
        .saturating_mul(TWO_POW_32_INT)
        .saturating_add(u64::from(n))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut out = [0_u8; 16];
    for half in out.chunks_exact_mut(8) {
        x ^= x >> 31;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        half.copy_from_slice(&x.to_le_bytes());
    }
    out
}

/// A 16-byte hash of an event id (prototype key for the forward runs).
fn key_hash(key: &str) -> [u8; 16] {
    let mut a = 0xcbf2_9ce4_8422_2325_u64;
    let mut b = 0x8422_2325_cbf2_9ce4_u64;
    for byte in key.bytes() {
        a = (a ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        b = (b.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    let mut h = [0_u8; 16];
    h[..8].copy_from_slice(&a.to_be_bytes());
    h[8..].copy_from_slice(&b.to_be_bytes());
    h
}

/// Per-id prototype lists: prev ids, auth ids, relation ids.
type IdLists = (Vec<u32>, Vec<u32>, Vec<u32>);

/// Dense short ids in first-seen order, with each id's edge lists.
#[derive(Default)]
struct Ids<'a> {
    by_key: HashMap<&'a str, u32>,
    keys: Vec<&'a str>,
    lists: Vec<IdLists>,
}

impl<'a> Ids<'a> {
    fn id(&mut self, key: &'a str) -> u32 {
        if let Some(id) = self.by_key.get(key) {
            return *id;
        }
        let id = u32::try_from(self.keys.len()).expect("id fits u32");
        self.by_key.insert(key, id);
        self.keys.push(key);
        self.lists.push(IdLists::default());
        id
    }
}

fn chunk_bounds(chunk: usize, len: usize) -> (usize, usize) {
    let start = chunk.saturating_mul(CHUNK);
    (start, start.saturating_add(CHUNK).min(len))
}

fn encode_edge_chunk(ids: &Ids<'_>, chunk: usize) -> Vec<u8> {
    let (start, end) = chunk_bounds(chunk, ids.lists.len());
    let mut body = Vec::new();
    let mut offsets = Vec::with_capacity(CHUNK);
    for (prev, auth, rel) in &ids.lists[start..end] {
        offsets.push(u32::try_from(body.len()).expect("chunk fits u32"));
        for list in [prev, auth, rel] {
            body.push(u8::try_from(list.len()).unwrap_or(u8::MAX));
        }
        for id in prev.iter().chain(auth).chain(rel) {
            body.extend_from_slice(&id.to_be_bytes());
        }
    }
    let mut out = b"PEDG".to_vec();
    out.extend_from_slice(&u32::try_from(chunk).expect("chunk").to_be_bytes());
    out.extend_from_slice(
        &u16::try_from(end.saturating_sub(start))
            .expect("count")
            .to_be_bytes(),
    );
    for offset in offsets {
        out.extend_from_slice(&offset.to_be_bytes());
    }
    out.extend_from_slice(&body);
    out
}

fn encode_key_chunk(ids: &Ids<'_>, chunk: usize) -> Vec<u8> {
    let (start, end) = chunk_bounds(chunk, ids.keys.len());
    let mut out = b"PREV".to_vec();
    out.extend_from_slice(&u32::try_from(chunk).expect("chunk").to_be_bytes());
    for key in &ids.keys[start..end] {
        out.push(u8::try_from(key.len()).expect("key length"));
        out.extend_from_slice(key.as_bytes());
    }
    out
}

fn encode_run(entries: &mut [([u8; 16], u32)]) -> Vec<u8> {
    entries.sort_unstable();
    let mut out = b"PFWD".to_vec();
    out.extend_from_slice(&u32::try_from(entries.len()).expect("run").to_be_bytes());
    for (hash, id) in entries.iter() {
        out.extend_from_slice(hash);
        out.extend_from_slice(&id.to_be_bytes());
    }
    out
}

/// Allocate ids for one batch of rows and fill in their edge lists.
fn assign_batch<'a>(ids: &mut Ids<'a>, batch: &'a [Row]) {
    for row in batch {
        let owner = ids.id(&row.id);
        let mut prev: Vec<u32> = row.prev.iter().map(|t| ids.id(t)).collect();
        let mut auth: Vec<u32> = row.auth.iter().map(|t| ids.id(t)).collect();
        let rel: Vec<u32> = row.rel.iter().map(|t| ids.id(t)).collect();
        prev.sort_unstable();
        prev.dedup();
        auth.sort_unstable();
        auth.dedup();
        ids.lists[owner as usize] = (prev, auth, rel);
    }
}

/// Prototype layout, written through the same `Database` as the baseline: dense
/// short ids in first-seen order; edge lists and the id -> key map in chunks of
/// 256 ids (the tail chunk rewritten each batch); the key -> id map as one
/// sorted run of (16-byte key hash, id) per batch.
fn chunked(base: &Path, rows: &[Row]) {
    let dir = base.join(format!(
        "mtxdb_bench_edgelayout_{}_chunked",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let store = db.edges();
    let mut ids = Ids::default();
    let mut payload_bytes = 0_u64;
    let mut puts = 0_u64;
    for (n, batch) in rows.chunks(BATCH).enumerate() {
        let first_new = ids.keys.len();
        assign_batch(&mut ids, batch);
        let mut put = |id: [u8; 16], bytes: &[u8]| {
            payload_bytes = payload_bytes.saturating_add(u64::try_from(bytes.len()).unwrap_or(0));
            puts = puts.saturating_add(1);
            store
                .put(&CHUNK_COLLECTION, &id, &NodeData::from_slice(bytes))
                .expect("put");
        };
        let mut run: Vec<([u8; 16], u32)> = (first_new..ids.keys.len())
            .map(|i| (key_hash(ids.keys[i]), u32::try_from(i).expect("id")))
            .collect();
        put(
            spread(3, u32::try_from(n).expect("batch")),
            &encode_run(&mut run),
        );
        for chunk in (first_new / CHUNK)..ids.keys.len().div_ceil(CHUNK) {
            let number = u32::try_from(chunk).expect("chunk");
            put(spread(1, number), &encode_edge_chunk(&ids, chunk));
            put(spread(2, number), &encode_key_chunk(&ids, chunk));
        }
        if n.saturating_add(1) % SYNC_EVERY == 0 {
            store.sync_all().expect("sync");
        }
    }
    store.sync_all().expect("sync");
    let before = sizes(&dir);
    store.force_index_checkpoint().expect("checkpoint");
    let after = sizes(&dir);
    drop(db);
    println!(
        "[chunked prototype] {} ids, {puts} puts, {:.2} MiB of payload written",
        ids.keys.len(),
        mib(payload_bytes),
    );
    println!(
        "  before forced checkpoint: packs {:.2} MiB, checkpoint {:.2}, delta {:.2} | after: packs {:.2}, checkpoint {:.2}, delta {:.2}",
        mib(before.packs),
        mib(before.checkpoint),
        mib(before.delta),
        mib(after.packs),
        mib(after.checkpoint),
        mib(after.delta),
    );
    breakdown(&dir, rows.len());
    println!(
        "  total {:.2} MiB = {:.1} B/event",
        mib(after.total()),
        per(after.total(), rows.len())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let limit = std::env::var("MTXDB_EL_EVENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, limit);
    println!("{} events from {}", rows.len(), tsv.display());
    let variant = |label, prev, auth, rel| Variant {
        label,
        prev,
        auth,
        rel,
    };
    let owners = run(&base, &rows, variant("owners only", false, false, false));
    let prev = run(&base, &rows, variant("prev", true, false, false));
    let with_auth = run(&base, &rows, variant("prev+auth", true, true, false));
    let all = run(
        &base,
        &rows,
        variant("prev+auth+relations", true, true, true),
    );
    println!(
        "marginal over owners-only, B/event: prev {:+.1}, auth {:+.1}, relations {:+.1}",
        (num(prev) - num(owners)) / len_f64(rows.len()),
        (num(with_auth) - num(prev)) / len_f64(rows.len()),
        (num(all) - num(with_auth)) / len_f64(rows.len()),
    );
    chunked(&base, &rows);
}
