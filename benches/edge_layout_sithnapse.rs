//! Write cost of sithnapse's auth-edge layout, replayed on a real room.
//!
//! Sithnapse (`mtxdb_syn.rs`) keeps, per event short id, one record holding the
//! outbound auth list and, per auth event, one record holding its child list.
//! Each event writes its own outbound record, then read-modify-writes the child
//! record of every auth event it names (aggregated per parent within a batch).
//! A hot parent (the create or power-levels event) therefore has a record that
//! grows with the room and is rewritten whole on every append.
//!
//! This is a benchmark-only replay of that write shape through the same
//! `Database` and policies as the other `edge_layout_*` benches. It includes
//! forward/reverse id records and the allocation counter, but is not
//! sithnapse's exact node-id derivation, lookup path, SQL fetches or closure cache. It
//! preloads `MTXDB_EL_PRELOAD` events in batches of 1,000, then appends
//! `MTXDB_EL_APPEND` in batches of 1, 2, 10 and 100 (`MTXDB_EL_BATCHES`), syncing
//! every `MTXDB_EL_SYNC_EVERY` batches (default 20), and reports wall time,
//! on-disk growth, payload put, the largest record put, and the dead share
//! against a simulated compaction.
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_sithnapse
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mtxdb::database::Database;
use mtxdb::storage::{NodeData, StorageEngine};

#[path = "edge_layout_common.rs"]
mod common;

use common::{
    assign_batch, compact_latest, key_hash, len_f64, load, mib, num, sizes, spread, Ids, Latest,
    Row, Writer, CHUNK_COLLECTION,
};

const KIND_COUNTER: u8 = 3;
const KIND_OUT: u8 = 4;
const KIND_CHILD: u8 = 5;
const KIND_REVERSE: u8 = 6;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Sithnapse's encoding: version byte, big-endian count, big-endian ids.
fn encode(list: &[u32]) -> Vec<u8> {
    let mut buf = vec![1_u8];
    buf.extend_from_slice(&u32::try_from(list.len()).unwrap_or(u32::MAX).to_be_bytes());
    for id in list {
        buf.extend_from_slice(&id.to_be_bytes());
    }
    buf
}

fn decode(bytes: &[u8]) -> Vec<u32> {
    bytes
        .get(5..)
        .unwrap_or_default()
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Everything the replay carries across batches and phases.
#[derive(Default)]
struct State<'a> {
    ids: Ids<'a>,
    latest: Latest,
    put_bytes: u64,
    puts: u64,
    largest: usize,
    sync: Duration,
    /// `Some(span)`: children are stored in one record per `(parent, child id
    /// range of span)`, so an append rewrites only the newest range. `None`:
    /// sithnapse's one whole list per parent.
    span: Option<u32>,
}

impl State<'_> {
    fn put(&mut self, store: &dyn StorageEngine, id: [u8; 16], bytes: Vec<u8>) {
        store
            .put(&CHUNK_COLLECTION, &id, &NodeData::from_slice(&bytes))
            .expect("put");
        self.put_bytes = self
            .put_bytes
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        self.puts = self.puts.saturating_add(1);
        self.largest = self.largest.max(bytes.len());
        self.latest.insert(id, bytes);
    }

    fn sync(&mut self, db: &Database) {
        let started = Instant::now();
        db.edges().sync_all().expect("sync");
        self.sync = self.sync.saturating_add(started.elapsed());
    }
}

impl State<'_> {
    /// Add `new` children to `parent`'s stored list(s), deduplicating.
    fn append_children(&mut self, store: &dyn StorageEngine, parent: u32, new: Vec<u32>) {
        let mut by_record: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for child in new {
            let range = self
                .span
                .map_or(0, |span| child.checked_div(span).unwrap_or(0));
            by_record.entry(range).or_default().push(child);
        }
        for (range, kids) in by_record {
            let id = child_record(parent, range, self.span);
            let mut children = store
                .get(&CHUNK_COLLECTION, &id)
                .expect("get")
                .map(|d| decode(&d.bytes))
                .unwrap_or_default();
            let mut seen: HashSet<u32> = children.iter().copied().collect();
            let before = children.len();
            children.extend(kids.into_iter().filter(|c| seen.insert(*c)));
            if children.len() != before {
                self.put(store, id, encode(&children));
            }
        }
    }
}

/// The record holding `parent`'s children in id `range`.
fn child_record(parent: u32, range: u32, span: Option<u32>) -> [u8; 16] {
    match span {
        None => spread(KIND_CHILD, parent),
        Some(_) => key_hash(&format!("child:{parent}:{range}")),
    }
}

/// Read all of `parent`'s children; returns them and the number of gets.
fn children_of(
    store: &dyn StorageEngine,
    parent: u32,
    (max_id, span): (u32, Option<u32>),
) -> (Vec<u32>, usize) {
    let ranges = span.map_or(0..=0, |span| {
        parent.checked_div(span).unwrap_or(0)..=max_id.checked_div(span).unwrap_or(0)
    });
    let mut out = Vec::new();
    let mut gets = 0_usize;
    for range in ranges {
        gets = gets.saturating_add(1);
        if let Some(data) = store
            .get(&CHUNK_COLLECTION, &child_record(parent, range, span))
            .expect("get")
        {
            out.extend(decode(&data.bytes));
        }
    }
    (out, gets)
}

impl<'a> State<'a> {
    /// Replay `rows` in batches: outbound record per event, then one
    /// read-modify-write per distinct parent in the batch.
    fn append(&mut self, db: &Database, rows: &'a [Row], (batch, sync_every): (usize, usize)) {
        let store = db.edges();
        for (n, chunk) in rows.chunks(batch).enumerate() {
            let first_new = self.ids.keys.len();
            let _ = assign_batch(&mut self.ids, chunk, false);
            let new_keys: Vec<(u32, String)> = self.ids.keys[first_new..]
                .iter()
                .enumerate()
                .map(|(offset, key)| {
                    (
                        u32::try_from(first_new.saturating_add(offset)).expect("short id"),
                        (*key).to_owned(),
                    )
                })
                .collect();
            for (short_id, key) in new_keys {
                self.put(
                    store.as_ref(),
                    key_hash(&key),
                    short_id.to_be_bytes().to_vec(),
                );
                self.put(
                    store.as_ref(),
                    spread(KIND_REVERSE, short_id),
                    key.into_bytes(),
                );
            }
            if self.ids.keys.len() > first_new {
                self.put(
                    store.as_ref(),
                    spread(KIND_COUNTER, 0),
                    u32::try_from(self.ids.keys.len())
                        .expect("counter")
                        .to_be_bytes()
                        .to_vec(),
                );
            }
            let mut by_parent: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
            for row in chunk {
                let owner = self.ids.by_key[row.id.as_str()];
                let auth = self.ids.lists[owner as usize].1.clone();
                self.put(store.as_ref(), spread(KIND_OUT, owner), encode(&auth));
                for parent in auth {
                    by_parent.entry(parent).or_default().push(owner);
                }
            }
            for (parent, new) in by_parent {
                self.append_children(store.as_ref(), parent, new);
            }
            if n.saturating_add(1).checked_rem(sync_every) == Some(0) {
                self.sync(db);
            }
        }
    }
}

fn case(
    base: &std::path::Path,
    rows: &[Row],
    preload: usize,
    (batch, sync_every, span): (usize, usize, Option<u32>),
) {
    let dir = base.join(format!("mtxdb_bench_sithnapse_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let (head, tail) = rows.split_at(preload);
    let mut state = State {
        span,
        ..State::default()
    };
    state.append(&db, head, (1_000, 20));
    state.sync(&db);
    let before = sizes(&dir).total();
    let (bytes0, count0, sync0) = (state.put_bytes, state.puts, state.sync);
    state.largest = 0;
    let started = Instant::now();
    state.append(&db, tail, (batch, sync_every));
    let wall = started.elapsed();
    state.sync(&db);
    let after = sizes(&dir);
    let label = span.map_or_else(|| "whole-list".to_owned(), |s| format!("range-{s}"));
    let events = len_f64(tail.len().max(1));
    let (took, settled) = compact_latest(&state.latest, &dir.with_extension("compact"));
    println!(
        "  {label} batch {batch:>3}: {:>9.1} us/event | growth {:>8.1} B/event | put {:>8.1} B/event, {} puts, largest {:.1} KiB | sync {:.0} ms",
        wall.as_secs_f64() * 1e6 / events,
        num(after.total().saturating_sub(before)) / events,
        num(state.put_bytes.saturating_sub(bytes0)) / events,
        state.puts.saturating_sub(count0),
        num(u64::try_from(state.largest).unwrap_or(0)) / 1024.0,
        state.sync.saturating_sub(sync0).as_secs_f64() * 1e3,
    );
    println!(
        "      packs {:.2} MiB, {:.2} after simulated compaction ({:.0}% dead), copy {:.0} ms",
        mib(after.packs),
        mib(settled.packs),
        100.0 * (1.0 - mib(settled.packs) / mib(after.packs).max(f64::MIN_POSITIVE)),
        took.as_secs_f64() * 1e3,
    );
    child_reads(&db, rows, &state.ids, span);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bytes put so far that replaced an earlier version of a chunk.
fn rewrite_bytes(writer: &Writer<'_>) -> u64 {
    writer
        .stats
        .edge_rewrite
        .bytes
        .saturating_add(writer.stats.key_rewrite.bytes)
        .saturating_add(writer.stats.child_rewrite.bytes)
}

fn chunked_case(
    base: &std::path::Path,
    rows: &[Row],
    preload: usize,
    (batch, sync_every): (usize, usize),
) {
    let dir = base.join(format!(
        "mtxdb_bench_sithnapse_chunked_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let (head, tail) = rows.split_at(preload);
    let mut writer = Writer::default();
    writer.append_bidirectional(&db, head, (1_000, 20));
    writer.sync(&db);
    let before = sizes(&dir).total();
    let (bytes0, puts0, sync0) = (
        writer.stats.payload_bytes(),
        writer.stats.total_puts(),
        writer.stats.sync_time,
    );
    let rewritten0 = rewrite_bytes(&writer);
    let started = Instant::now();
    writer.append_bidirectional(&db, tail, (batch, sync_every));
    let wall = started.elapsed();
    writer.sync(&db);
    let after = sizes(&dir);
    let events = len_f64(tail.len().max(1));
    let new_payload = writer.stats.payload_bytes().saturating_sub(bytes0);
    let replaced = rewrite_bytes(&writer).saturating_sub(rewritten0);
    let sync = writer.stats.sync_time.saturating_sub(sync0);
    let (took, settled) = writer.simulate_compaction(&dir.with_extension("compact"));
    println!(
        "  chunked batch {batch:>3}: {:>9.1} us/event | growth {:>8.1} B/event | payload {:>8.1} B/event, rewrites {:>5.1}%, {} puts | sync {:.0} ms",
        wall.as_secs_f64() * 1e6 / events,
        num(after.total().saturating_sub(before)) / events,
        num(new_payload) / events,
        100.0 * num(replaced) / num(new_payload.max(1)),
        writer.stats.total_puts().saturating_sub(puts0),
        sync.as_secs_f64() * 1e3,
    );
    println!(
        "      packs {:.2} MiB, {:.2} after simulated compaction ({:.0}% dead), copy {:.0} ms",
        mib(after.packs),
        mib(settled.packs),
        100.0 * (1.0 - mib(settled.packs) / mib(after.packs).max(f64::MIN_POSITIVE)),
        took.as_secs_f64() * 1e3,
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Read cost of a child list after the appends: the busiest parent, and a sample
/// of parents drawn from the events' auth lists (so busy parents weigh in as
/// often as they are named).
fn child_reads(db: &Database, rows: &[Row], ids: &Ids<'_>, span: Option<u32>) {
    let store = db.edges();
    let max_id = u32::try_from(ids.keys.len().saturating_sub(1)).unwrap_or(u32::MAX);
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for parent in rows.iter().flat_map(|row| &row.auth) {
        let seen = counts.entry(parent.as_str()).or_default();
        *seen = seen.saturating_add(1);
    }
    let Some((hot, _)) = counts.iter().max_by_key(|(key, n)| (**n, **key)) else {
        return;
    };
    let timed = |parent: u32, reps: usize| {
        let started = Instant::now();
        let mut shape = (0, 0);
        for _ in 0..reps {
            let (children, gets) = children_of(store.as_ref(), parent, (max_id, span));
            shape = (children.len(), gets);
        }
        (started.elapsed().as_secs_f64() * 1e6 / len_f64(reps), shape)
    };
    let (hot_us, (hot_children, hot_gets)) = timed(ids.by_key[*hot], 20);
    let sample: Vec<u32> = rows
        .iter()
        .flat_map(|row| &row.auth)
        .step_by(37)
        .filter_map(|key| ids.by_key.get(key.as_str()).copied())
        .take(2_000)
        .collect();
    let started = Instant::now();
    let gets = sample.iter().fold(0_usize, |acc, parent| {
        acc.saturating_add(children_of(store.as_ref(), *parent, (max_id, span)).1)
    });
    let typical = started.elapsed().as_secs_f64() * 1e6 / len_f64(sample.len().max(1));
    println!(
        "      child reads: busiest parent {hot_us:.0} us ({hot_children} children, {hot_gets} gets); sampled parents {typical:.1} us ({:.1} gets)",
        num(u64::try_from(gets).unwrap_or(0)) / len_f64(sample.len().max(1)),
    );
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let loaded = load(&tsv, usize::MAX);
    // Sithnapse's auth-chain store is specifically auth adjacency in both
    // directions; omit prev and relation payloads from this comparison.
    let rows: Vec<Row> = loaded
        .into_iter()
        .map(|row| Row {
            id: row.id,
            prev: Vec::new(),
            auth: row.auth,
            rel: None,
        })
        .collect();
    let preload = env_usize("MTXDB_EL_PRELOAD", 60_000).min(rows.len());
    let append = env_usize("MTXDB_EL_APPEND", 2_000).min(rows.len().saturating_sub(preload));
    let sync_every = env_usize("MTXDB_EL_SYNC_EVERY", 20);
    let batches: Vec<usize> = std::env::var("MTXDB_EL_BATCHES").map_or_else(
        |_| vec![1, 2, 10, 100],
        |v| v.split(',').filter_map(|s| s.parse().ok()).collect(),
    );
    let rows = &rows[..preload.saturating_add(append)];
    println!(
        "sithnapse-shaped writes: {preload} events preloaded (batches of 1000), {append} appended, sync every {sync_every} batches"
    );
    let span = env_usize("MTXDB_EL_SPAN", 2_048);
    let span = u32::try_from(span).ok().filter(|s| *s > 0);
    for batch in batches {
        case(&base, rows, preload, (batch, sync_every, None));
        case(&base, rows, preload, (batch, sync_every, span));
        chunked_case(&base, rows, preload, (batch, sync_every));
    }
}
