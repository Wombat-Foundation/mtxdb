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
            let created = children.is_empty();
            children.extend(kids.into_iter().filter(|c| seen.insert(*c)));
            if children.len() != before {
                self.put(store, id, encode(&children));
            }
            if created && self.span.is_some() {
                self.note_range(store, parent, range);
            }
        }
    }

    /// Record that `parent` has children in `range`, so reads probe only the
    /// ranges that exist. Written once per (parent, range), when it first appears.
    fn note_range(&mut self, store: &dyn StorageEngine, parent: u32, range: u32) {
        let id = meta_record(parent);
        let mut ranges = store
            .get(&CHUNK_COLLECTION, &id)
            .expect("get")
            .map(|d| decode(&d.bytes))
            .unwrap_or_default();
        if let Err(at) = ranges.binary_search(&range) {
            ranges.insert(at, range);
            self.put(store, id, encode(&ranges));
        }
    }
}

/// The record listing the id ranges in which `parent` has children.
fn meta_record(parent: u32) -> [u8; 16] {
    key_hash(&format!("child-ranges:{parent}"))
}

/// The record holding `parent`'s children in id `range`.
fn child_record(parent: u32, range: u32, span: Option<u32>) -> [u8; 16] {
    match span {
        None => spread(KIND_CHILD, parent),
        Some(_) => key_hash(&format!("child:{parent}:{range}")),
    }
}

/// Read all of `parent`'s children; returns them and the number of gets. A
/// segmented layout reads the parent's range list, then only those ranges.
fn children_of(store: &dyn StorageEngine, parent: u32, span: Option<u32>) -> (Vec<u32>, usize) {
    let get = |id: [u8; 16]| {
        store
            .get(&CHUNK_COLLECTION, &id)
            .expect("get")
            .map(|d| decode(&d.bytes))
    };
    let Some(span) = span else {
        return (get(child_record(parent, 0, None)).unwrap_or_default(), 1);
    };
    let ranges = get(meta_record(parent)).unwrap_or_default();
    let mut gets = 1_usize;
    let mut out = Vec::new();
    for range in ranges {
        gets = gets.saturating_add(1);
        out.extend(get(child_record(parent, range, Some(span))).unwrap_or_default());
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

/// xorshift64, seeded, for a reproducible trace.
fn next_rand(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn pick(state: &mut u64, len: usize) -> usize {
    usize::try_from(next_rand(state))
        .unwrap_or(0)
        .checked_rem(len)
        .unwrap_or(0)
}

/// Every id that `starts` can reach through auth edges, starts included.
fn ancestors(auth_of: &HashMap<u32, Vec<u32>>, starts: &[u32]) -> HashSet<u32> {
    let mut seen: HashSet<u32> = starts.iter().copied().collect();
    let mut frontier = starts.to_vec();
    while let Some(id) = frontier.pop() {
        for parent in auth_of.get(&id).into_iter().flatten() {
            if seen.insert(*parent) {
                frontier.push(*parent);
            }
        }
    }
    seen
}

/// The parents whose child lists sithnapse's v2.1 forward walk
/// (`get_forward_reachable_short_ids`) would read, built from the source rows.
/// Each simulated conflict has a root, plus two descendants found by a random
/// forward walk; the walk reads the child list of every node it expands, and only
/// expands children that are ancestors of a conflicted event. `hot_percent` of
/// the conflicts start at one of the busiest parents. Both choices are
/// assumptions, since no real access trace exists; the trace is seeded.
fn build_trace(
    rows: &[Row],
    ids: &Ids<'_>,
    expected: &HashMap<u32, Vec<u32>>,
    (limit, hot_percent): (usize, usize),
) -> Vec<u32> {
    let auth_of: HashMap<u32, Vec<u32>> = rows
        .iter()
        .map(|row| {
            (
                ids.by_key[row.id.as_str()],
                row.auth.iter().map(|a| ids.by_key[a.as_str()]).collect(),
            )
        })
        .collect();
    let mut busiest: Vec<u32> = expected.keys().copied().collect();
    busiest.sort_unstable_by_key(|id| std::cmp::Reverse(expected[id].len()));
    busiest.truncate(5);
    let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
    let mut trace = Vec::new();
    while trace.len() < limit {
        let root = if pick(&mut rng, 100) < hot_percent {
            busiest[pick(&mut rng, busiest.len())]
        } else {
            ids.by_key[rows[pick(&mut rng, rows.len())].id.as_str()]
        };
        let mut starts = vec![root];
        let mut at = root;
        for _ in 0..40 {
            let Some(kids) = expected.get(&at).filter(|k| !k.is_empty()) else {
                break;
            };
            at = kids[pick(&mut rng, kids.len())];
            if starts.len() < 3 && pick(&mut rng, 8) == 0 {
                starts.push(at);
            }
        }
        starts.push(at);
        let candidates = ancestors(&auth_of, &starts);
        let mut seen: HashSet<u32> = HashSet::new();
        let mut frontier = vec![root];
        while let Some(current) = frontier.pop() {
            if !seen.insert(current) {
                continue;
            }
            trace.push(current);
            for child in expected.get(&current).into_iter().flatten() {
                if candidates.contains(child) && !seen.contains(child) {
                    frontier.push(*child);
                }
            }
        }
    }
    trace.truncate(limit);
    trace
}

/// Replay `trace` against the store, one child-list read per entry, checking each
/// list against the source. Reports latency and gets per read, split between the
/// busiest parents ("hot": 1,000 or more children) and the rest.
fn replay_trace(
    store: &dyn StorageEngine,
    trace: &[u32],
    expected: &HashMap<u32, Vec<u32>>,
    span: Option<u32>,
) {
    let mut hot: Vec<(f64, usize)> = Vec::new();
    let mut typical: Vec<(f64, usize)> = Vec::new();
    let mut bad = 0_usize;
    for parent in trace {
        let started = Instant::now();
        let (mut got, gets) = children_of(store, *parent, span);
        let micros = started.elapsed().as_secs_f64() * 1e6;
        got.sort_unstable();
        let mut want = expected.get(parent).cloned().unwrap_or_default();
        want.sort_unstable();
        want.dedup();
        bad = bad.saturating_add(usize::from(got != want));
        if want.len() >= 1_000 {
            hot.push((micros, gets));
        } else {
            typical.push((micros, gets));
        }
    }
    let all: Vec<(f64, usize)> = hot.iter().chain(&typical).copied().collect();
    for (label, reads) in [("all", &all), ("hot", &hot), ("typical", &typical)] {
        let mut sorted: Vec<f64> = reads.iter().map(|r| r.0).collect();
        sorted.sort_by(f64::total_cmp);
        let at = |pct: usize| {
            sorted
                .get(sorted.len().saturating_sub(1).saturating_mul(pct) / 100)
                .copied()
                .unwrap_or(0.0)
        };
        let count = len_f64(reads.len().max(1));
        let mean: f64 = sorted.iter().sum::<f64>() / count;
        let gets: f64 = reads.iter().map(|r| len_f64(r.1)).sum::<f64>() / count;
        println!(
            "      trace {label:<7} {:>6} reads: mean {mean:>8.1} us, p50 {:>8.1} p95 {:>8.1} p99 {:>8.1} | {gets:.1} gets/read",
            reads.len(),
            at(50),
            at(95),
            at(99),
        );
    }
    println!("      trace mismatches {bad}");
}

/// Read cost of a child list after the appends: the busiest parent, and a sample
/// of parents drawn from the events' auth lists (so busy parents weigh in as
/// often as they are named). Every list read is checked against the source rows.
fn child_reads(db: &Database, rows: &[Row], ids: &Ids<'_>, span: Option<u32>) {
    let store = db.edges();
    let mut expected: HashMap<u32, Vec<u32>> = HashMap::new();
    for row in rows {
        let child = ids.by_key[row.id.as_str()];
        for parent in &row.auth {
            expected
                .entry(ids.by_key[parent.as_str()])
                .or_default()
                .push(child);
        }
    }
    let Some((hot, _)) = expected.iter().max_by_key(|(id, kids)| (kids.len(), **id)) else {
        return;
    };
    let mut bad = 0_usize;
    let mut check = |parent: u32, got: &mut Vec<u32>| {
        got.sort_unstable();
        let mut want = expected.get(&parent).cloned().unwrap_or_default();
        want.sort_unstable();
        want.dedup();
        bad = bad.saturating_add(usize::from(*got != want));
    };
    let started = Instant::now();
    let reps = 20_usize;
    let mut shape = (Vec::new(), 0);
    for _ in 0..reps {
        shape = children_of(store.as_ref(), *hot, span);
    }
    let hot_us = started.elapsed().as_secs_f64() * 1e6 / len_f64(reps);
    let (mut hot_children, hot_gets) = shape;
    let hot_len = hot_children.len();
    check(*hot, &mut hot_children);
    let sample: Vec<u32> = rows
        .iter()
        .flat_map(|row| &row.auth)
        .step_by(37)
        .filter_map(|key| ids.by_key.get(key.as_str()).copied())
        .take(2_000)
        .collect();
    let mut gets = 0_usize;
    let mut elapsed = Duration::ZERO;
    for parent in &sample {
        let started = Instant::now();
        let (mut got, n) = children_of(store.as_ref(), *parent, span);
        elapsed = elapsed.saturating_add(started.elapsed());
        gets = gets.saturating_add(n);
        check(*parent, &mut got);
    }
    let count = len_f64(sample.len().max(1));
    println!(
        "      child reads: busiest parent {hot_us:.0} us ({hot_len} children, {hot_gets} gets); sampled parents {:.1} us ({:.1} gets); mismatches {bad}",
        elapsed.as_secs_f64() * 1e6 / count,
        num(u64::try_from(gets).unwrap_or(0)) / count,
    );
    let trace = build_trace(
        rows,
        ids,
        &expected,
        (
            env_usize("MTXDB_EL_TRACE_READS", 20_000),
            env_usize("MTXDB_EL_HOT_STARTS", 10),
        ),
    );
    if let Some(path) = std::env::var_os("MTXDB_EL_TRACE").filter(|_| span.is_none()) {
        let lines: Vec<String> = trace.iter().map(u32::to_string).collect();
        std::fs::write(path, lines.join("\n")).expect("write trace");
    }
    replay_trace(store.as_ref(), &trace, &expected, span);
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
    let spans: Vec<u32> = std::env::var("MTXDB_EL_SPANS").map_or_else(
        |_| vec![1_024, 2_048, 8_192],
        |v| {
            v.split(',')
                .filter_map(|s| s.parse().ok())
                .filter(|s| *s > 0)
                .collect()
        },
    );
    let chunked = env_usize("MTXDB_EL_CHUNKED", 1) != 0;
    for batch in batches {
        case(&base, rows, preload, (batch, sync_every, None));
        for span in &spans {
            case(&base, rows, preload, (batch, sync_every, Some(*span)));
        }
        if chunked {
            chunked_case(&base, rows, preload, (batch, sync_every));
        }
    }
}
