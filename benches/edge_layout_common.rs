//! Shared by the `edge_layout_*` benches: the room loader, the baseline
//! `ShortIdIndex` writer, the chunked prototype's layout and writer, and size
//! helpers.
//!
//! The prototype is a benchmark-only layout; nothing here is a production format.
//! Each bench target uses a subset of these items.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

#[path = "edge_layout_format.rs"]
pub(crate) mod format;

use mtxdb::database::Database;
use mtxdb::layout::ShardType;
use mtxdb::short_id::{BatchEvent, EdgeFamily, EdgeKey, FamilyEdges, ShortIdIndex};
use mtxdb::storage::{NodeData, StorageEngine};

pub(crate) const SCOPE: [u8; 16] = [0x71; 16];
pub(crate) const CHUNK_COLLECTION: [u8; 16] = [0x72; 16];
pub(crate) const BATCH: usize = 1000;
pub(crate) const SYNC_EVERY: usize = 20;
const DEFAULT_CHUNK: usize = 256;
pub(crate) const PREV: EdgeFamily = EdgeFamily::plain(1);
pub(crate) const AUTH: EdgeFamily = EdgeFamily::plain(2);
pub(crate) const REL: EdgeFamily = EdgeFamily::typed(3);
/// Record-id kinds for [`spread`].
pub(crate) const KIND_EDGE_CHUNK: u8 = 1;
pub(crate) const KIND_KEY_CHUNK: u8 = 2;
pub(crate) const KIND_RUN: u8 = 3;
const MIB: f64 = 1_048_576.0;
const TWO_POW_32: f64 = 4_294_967_296.0;
const TWO_POW_32_INT: u64 = 4_294_967_296;

/// Ids per prototype chunk: `MTXDB_EL_CHUNK`, default 256. At most 256, so
/// the `u16` offsets and counts in key chunks cannot overflow even with
/// maximum-length keys.
///
/// # Panics
/// If `MTXDB_EL_CHUNK` is not an integer in `1..=256`.
pub(crate) fn chunk_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("MTXDB_EL_CHUNK").map_or(DEFAULT_CHUNK, |v| {
            v.parse()
                .ok()
                .filter(|n| (1..=DEFAULT_CHUNK).contains(n))
                .expect("MTXDB_EL_CHUNK must be an integer in 1..=256")
        })
    })
}

/// The chunk that holds `id`, and its position inside the chunk.
pub(crate) fn locate(id: usize) -> (usize, usize) {
    format::locate(id, chunk_size())
}

/// One event of the room.
pub(crate) struct Row {
    pub(crate) id: String,
    pub(crate) prev: Vec<String>,
    pub(crate) auth: Vec<String>,
    pub(crate) rel: Option<String>,
}

/// Which edge families a baseline variant records.
#[derive(Clone, Copy)]
pub(crate) struct Variant {
    pub(crate) label: &'static str,
    pub(crate) prev: bool,
    pub(crate) auth: bool,
    pub(crate) rel: bool,
}

/// Bytes of the store's pack, checkpoint and delta files.
#[derive(Clone, Copy)]
pub(crate) struct Sizes {
    pub(crate) packs: u64,
    pub(crate) checkpoint: u64,
    pub(crate) delta: u64,
}

impl Sizes {
    pub(crate) fn total(self) -> u64 {
        self.packs
            .saturating_add(self.checkpoint)
            .saturating_add(self.delta)
    }
}

/// A `u64` as `f64`, exact below 2^53 (far above any size measured here).
pub(crate) fn num(n: u64) -> f64 {
    let high = u32::try_from(n >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(n & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * TWO_POW_32 + f64::from(low)
}

pub(crate) fn len_f64(n: usize) -> f64 {
    num(u64::try_from(n).unwrap_or(u64::MAX))
}

pub(crate) fn mib(n: u64) -> f64 {
    num(n) / MIB
}

/// `n` bytes per `events` events.
pub(crate) fn per(n: u64, events: usize) -> f64 {
    num(n) / len_f64(events.max(1))
}

fn split(field: &str) -> Vec<String> {
    field
        .split(',')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Read a tab-separated room: `event_id, prev ids, auth ids, relation target`.
pub(crate) fn load(path: &Path, limit: usize) -> Vec<Row> {
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
pub(crate) fn tree_bytes(dir: &Path, needle: &str) -> io::Result<u64> {
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

pub(crate) fn sizes(dir: &Path) -> Sizes {
    let measure = |needle: &str| tree_bytes(dir, needle).expect("measure directory size");
    Sizes {
        packs: measure(".pack"),
        checkpoint: measure("checkpoint"),
        delta: measure("delta"),
    }
}

/// One owner's edge keys per family.
struct Lists<'a> {
    prev: Vec<EdgeKey<'a>>,
    auth: Vec<EdgeKey<'a>>,
    rel: Vec<EdgeKey<'a>>,
}

fn plain_keys(ids: &[String]) -> Vec<EdgeKey<'_>> {
    ids.iter().map(|t| EdgeKey::plain(t.as_bytes())).collect()
}

fn lists_for(row: &Row, variant: Variant) -> Lists<'_> {
    Lists {
        prev: if variant.prev {
            plain_keys(&row.prev)
        } else {
            Vec::new()
        },
        auth: if variant.auth {
            plain_keys(&row.auth)
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

/// The baseline scope every bench reads and writes through.
pub(crate) fn baseline_scope() -> ShortIdIndex {
    ShortIdIndex::new(ShardType::Edges, SCOPE)
}

/// Record every row through `ShortIdIndex`; returns the number of edges written.
pub(crate) fn record_rows(db: &Database, rows: &[Row], variant: Variant) -> usize {
    let scope = baseline_scope();
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

/// A spread 16-byte record id for a prototype record, so ids do not share a prefix.
pub(crate) fn spread(kind: u8, n: u32) -> [u8; 16] {
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
pub(crate) fn key_hash(key: &str) -> [u8; 16] {
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
pub(crate) type IdLists = (Vec<u32>, Vec<u32>, Vec<u32>);

/// Dense short ids in first-seen order, with each id's edge lists.
#[derive(Default)]
pub(crate) struct Ids<'a> {
    pub(crate) by_key: HashMap<&'a str, u32>,
    pub(crate) keys: Vec<&'a str>,
    pub(crate) lists: Vec<IdLists>,
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
    format::chunk_bounds(chunk, chunk_size(), len)
}

fn encode_edge_chunk(ids: &Ids<'_>, chunk: usize) -> Vec<u8> {
    let (start, end) = chunk_bounds(chunk, ids.lists.len());
    let mut body = Vec::new();
    let mut offsets = Vec::with_capacity(chunk_size());
    for (prev, auth, rel) in &ids.lists[start..end] {
        offsets.push(u32::try_from(body.len()).expect("chunk fits u32"));
        for list in [prev, auth, rel] {
            body.push(u8::try_from(list.len()).expect("list fits u8"));
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
    format::encode_key_chunk(&ids.keys[start..end], u32::try_from(chunk).expect("chunk"))
        .expect("key chunk encodes")
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

/// Allocate ids for one batch of rows and fill in their edge lists; returns the
/// chunks whose edge lists changed (including chunks of earlier ids whose list
/// is only filled in now, because the id was first seen as another event's edge).
fn assign_batch<'a>(ids: &mut Ids<'a>, batch: &'a [Row]) -> BTreeSet<usize> {
    let mut dirty = BTreeSet::new();
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
        dirty.insert(locate(owner as usize).0);
    }
    dirty
}

/// Puts, bytes and time spent in `put` for one class of prototype record.
#[derive(Clone, Copy, Default)]
pub(crate) struct PutStats {
    pub(crate) puts: u64,
    pub(crate) bytes: u64,
    pub(crate) time: Duration,
}

/// What writing the prototype cost, split by record class and by whether the
/// record was new or a rewrite of an earlier version.
#[derive(Default)]
pub(crate) struct WriteStats {
    pub(crate) runs: PutStats,
    pub(crate) edge_new: PutStats,
    pub(crate) edge_rewrite: PutStats,
    pub(crate) key_new: PutStats,
    pub(crate) key_rewrite: PutStats,
    pub(crate) sync_time: Duration,
}

impl WriteStats {
    pub(crate) fn payload_bytes(&self) -> u64 {
        [
            self.runs,
            self.edge_new,
            self.edge_rewrite,
            self.key_new,
            self.key_rewrite,
        ]
        .iter()
        .fold(0_u64, |acc, s| acc.saturating_add(s.bytes))
    }

    pub(crate) fn total_puts(&self) -> u64 {
        [
            self.runs,
            self.edge_new,
            self.edge_rewrite,
            self.key_new,
            self.key_rewrite,
        ]
        .iter()
        .fold(0_u64, |acc, s| acc.saturating_add(s.puts))
    }
}

fn timed_put(store: &dyn StorageEngine, stats: &mut PutStats, id: [u8; 16], bytes: &[u8]) {
    let started = Instant::now();
    store
        .put(&CHUNK_COLLECTION, &id, &NodeData::from_slice(bytes))
        .expect("put");
    stats.time = stats.time.saturating_add(started.elapsed());
    stats.puts = stats.puts.saturating_add(1);
    stats.bytes = stats
        .bytes
        .saturating_add(u64::try_from(bytes.len()).unwrap_or(0));
}

/// Write the prototype layout through `db`: dense short ids in first-seen order;
/// edge lists and the id -> key map in chunks of [`chunk_size`] ids, every chunk that
/// changed in a batch rewritten whole; the key -> id map as one sorted run of
/// (16-byte key hash, id) per batch. Syncs every [`SYNC_EVERY`] batches.
pub(crate) fn write_chunked<'a>(db: &Database, rows: &'a [Row]) -> (Ids<'a>, WriteStats) {
    let store = db.edges();
    let mut ids = Ids::default();
    let mut stats = WriteStats::default();
    let mut seen_edge: HashSet<usize> = HashSet::new();
    let mut seen_key: HashSet<usize> = HashSet::new();
    for (n, batch) in rows.chunks(BATCH).enumerate() {
        let first_new = ids.keys.len();
        let mut dirty = assign_batch(&mut ids, batch);
        let tail = ids.keys.len().div_ceil(chunk_size());
        dirty.extend(locate(first_new).0..tail);
        let mut run: Vec<([u8; 16], u32)> = (first_new..ids.keys.len())
            .map(|i| (key_hash(ids.keys[i]), u32::try_from(i).expect("id")))
            .collect();
        let batch_no = u32::try_from(n).expect("batch");
        timed_put(
            store.as_ref(),
            &mut stats.runs,
            spread(KIND_RUN, batch_no),
            &encode_run(&mut run),
        );
        for chunk in dirty {
            let number = u32::try_from(chunk).expect("chunk");
            let bucket = if seen_edge.insert(chunk) {
                &mut stats.edge_new
            } else {
                &mut stats.edge_rewrite
            };
            timed_put(
                store.as_ref(),
                bucket,
                spread(KIND_EDGE_CHUNK, number),
                &encode_edge_chunk(&ids, chunk),
            );
        }
        for chunk in locate(first_new).0..tail {
            let number = u32::try_from(chunk).expect("chunk");
            let bucket = if seen_key.insert(chunk) {
                &mut stats.key_new
            } else {
                &mut stats.key_rewrite
            };
            timed_put(
                store.as_ref(),
                bucket,
                spread(KIND_KEY_CHUNK, number),
                &encode_key_chunk(&ids, chunk),
            );
        }
        if n.saturating_add(1) % SYNC_EVERY == 0 {
            let started = Instant::now();
            store.sync_all().expect("sync");
            stats.sync_time = stats.sync_time.saturating_add(started.elapsed());
        }
    }
    let started = Instant::now();
    store.sync_all().expect("sync");
    stats.sync_time = stats.sync_time.saturating_add(started.elapsed());
    (ids, stats)
}
