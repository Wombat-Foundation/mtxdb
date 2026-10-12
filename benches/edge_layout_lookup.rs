//! Read and write cost of the chunked edge-layout prototype against the
//! current `ShortIdIndex` store, on the same room.
//!
//! Benchmark-only: the prototype is not a production format, and its on-disk
//! shape is the one `edge_layout_common.rs` writes. Both layouts live in the
//! same kind of `Database` pool, so storage reads cost the same.
//!
//! Workload assumptions, stated once:
//! - One room (`MTXDB_EL_TSV`), events recorded in file order in batches of
//!   1000; ids are dense and first-seen for the prototype.
//! - Lookups are a fixed seeded sample of `MTXDB_EL_OPS` (default 20000) events;
//!   hits ask for those events, misses ask for the same ids with `~` appended.
//! - The three paths are key -> id, id -> key and "all three edge lists of an
//!   owner". The baseline reads three records for the last; the prototype reads
//!   one chunk. The baseline has no batch API for key -> id or edges, so its
//!   "batch" loops over single reads; only `resolve` is a native batch.
//! - Prototype key -> id returns a hit only after the id -> key chunk confirms
//!   the key, so a hash collision cannot pass as a hit. `naive` reads every run
//!   record on each call; `merged` holds all runs in memory as one sorted
//!   vector, loaded once (its load time is printed).
//! - Every result is checked against the source ids (outside the timed region),
//!   and the whole room is round-tripped once per layout before timing.
//! - Cold = reopen the database, then `sync` and evict the store's files with
//!   `vmtouch -e` (reopen itself reads the store into the page cache), then time
//!   the first `MTXDB_EL_COLD` (default 1000) calls. The in-memory index
//!   survives, so cold measures pack reads.
//! - The confirmed variants are not collision-safe either: each run stores one
//!   id per 128-bit hash and lookup yields one candidate, so if two distinct
//!   keys collided, the check would reject the other key instead of finding its
//!   id. A real design needs collision buckets or another way to disambiguate.
//! - `proto-hash-only` is the merged variant without the key check: a hit is
//!   whatever the hash matched. It is NOT assumed safe; a query whose hash
//!   collides with a stored key would return that key's id, and 0 mismatches on
//!   this sample does not show otherwise. Reported only to price the check.
//! - Id -> key chunks carry a `u16` offset table, so a position is read without
//!   walking the chunk.
//! - Writes are timed separately: the prototype rewrites every chunk that
//!   changed in a batch, and bytes written are reported apart from latency.
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_lookup
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::database::Database;
use mtxdb::short_id::{Edge, EdgeFamily, ShortIdIndex};
use mtxdb::storage::StorageEngine;

#[path = "edge_layout_common.rs"]
mod common;

use common::format;
use common::{
    baseline_scope, evict_cold, key_hash, len_f64, load, locate, mib, num, per, record_rows, sizes,
    spread, write_chunked, PutStats, Row, Variant, WriteStats, AUTH, CHUNK_COLLECTION,
    KIND_EDGE_CHUNK, KIND_KEY_CHUNK, KIND_RUN, PREV, REL,
};

/// A key's three edge lists as target ids: prev, auth, relations.
type Edges = [Vec<u32>; 3];
/// Bytes of one prototype run entry: 16-byte key hash and a big-endian id.
const MICROS: f64 = 1_000_000.0;

/// What a layout can do. A single read passes a slice of one.
trait Layout {
    fn key_to_ids(&self, keys: &[&str]) -> Vec<Option<u32>>;
    fn id_to_keys(&self, ids: &[u32]) -> Vec<Option<String>>;
    fn edges(&self, ids: &[u32]) -> Vec<Edges>;
    /// One edge list per id: `which` 0 = prev, 1 = auth, 2 = relations. The
    /// default reads all three; the baseline reads only the family asked for,
    /// as `MatrixAdjacency` does.
    fn list(&self, ids: &[u32], which: usize) -> Vec<Vec<u32>> {
        self.edges(ids)
            .into_iter()
            .map(|mut lists| std::mem::take(&mut lists[which]))
            .collect()
    }
    /// Hash matches rejected because the stored key differed; `None` when the
    /// layout has no such check.
    fn rejected(&self) -> Option<u64> {
        None
    }
}

/// Opens a layout fresh, with no in-memory state from earlier calls.
type Open<'a> = Box<dyn Fn() -> Box<dyn Layout> + 'a>;

/// The current `ShortIdIndex` store.
struct Baseline {
    db: Database,
    scope: ShortIdIndex,
}

impl Baseline {
    fn open(dir: &Path) -> Self {
        Self {
            db: Database::open(dir.to_path_buf()).expect("open baseline"),
            scope: baseline_scope(),
        }
    }

    fn family(&self, id: u32, family: EdgeFamily) -> Vec<u32> {
        self.scope
            .edges(&self.db, id, family)
            .expect("edges")
            .map(|edges| edges.iter().map(|Edge { target, .. }| *target).collect())
            .unwrap_or_default()
    }
}

const FAMILIES: [EdgeFamily; 3] = [PREV, AUTH, REL];

impl Layout for Baseline {
    fn list(&self, ids: &[u32], which: usize) -> Vec<Vec<u32>> {
        ids.iter()
            .map(|id| self.family(*id, FAMILIES[which]))
            .collect()
    }

    fn key_to_ids(&self, keys: &[&str]) -> Vec<Option<u32>> {
        keys.iter()
            .map(|key| self.scope.lookup(&self.db, key.as_bytes()).expect("lookup"))
            .collect()
    }

    fn id_to_keys(&self, ids: &[u32]) -> Vec<Option<String>> {
        self.scope
            .resolve(&self.db, ids)
            .expect("resolve")
            .into_iter()
            .map(|key| key.map(|bytes| String::from_utf8(bytes).expect("utf8 key")))
            .collect()
    }

    fn edges(&self, ids: &[u32]) -> Vec<Edges> {
        ids.iter()
            .map(|id| {
                [
                    self.family(*id, PREV),
                    self.family(*id, AUTH),
                    self.family(*id, REL),
                ]
            })
            .collect()
    }
}

/// The chunked prototype read through the same `Database`.
struct Proto {
    db: Database,
    runs: usize,
    /// All runs as one sorted vector, when loaded (`merged` variant).
    merged: Option<Vec<([u8; 16], u32)>>,
    /// Hash matches whose key did not match.
    rejected: std::cell::Cell<u64>,
    /// Confirm a hash match against the stored key. Off only in the explicitly
    /// labeled hash-only variant, which trusts 128-bit hash identity.
    confirm: bool,
}

fn be32(bytes: &[u8], at: usize) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(at..at.saturating_add(4))?.try_into().ok()?;
    Some(u32::from_be_bytes(raw))
}

/// The three lists at `pos` in an edge-list chunk record.
fn edges_at(chunk: &[u8], pos: usize) -> Option<Edges> {
    let count = usize::from(u16::from_be_bytes(chunk.get(8..10)?.try_into().ok()?));
    let offsets = 10_usize;
    let body = offsets.saturating_add(count.saturating_mul(4));
    let off = usize::try_from(be32(chunk, offsets.saturating_add(pos.saturating_mul(4)))?).ok()?;
    let head = body.saturating_add(off);
    let lens = chunk.get(head..head.saturating_add(3))?;
    let mut at = head.saturating_add(3);
    let mut out: Edges = [Vec::new(), Vec::new(), Vec::new()];
    for (list, len) in out.iter_mut().zip(lens) {
        for _ in 0..*len {
            list.push(be32(chunk, at)?);
            at = at.saturating_add(4);
        }
    }
    Some(out)
}

impl Proto {
    fn open(dir: &Path, merge: bool, confirm: bool, runs: usize) -> Self {
        let mut proto = Self {
            db: Database::open(dir.to_path_buf()).expect("open prototype"),
            runs,
            merged: None,
            rejected: std::cell::Cell::new(0),
            confirm,
        };
        if merge {
            proto.merged = Some(proto.load_runs());
        }
        proto
    }

    fn fetch(&self, kind: u8, n: usize) -> Option<bytes::Bytes> {
        let id = spread(kind, u32::try_from(n).expect("record number"));
        self.db
            .edges()
            .get(&CHUNK_COLLECTION, &id)
            .expect("get")
            .map(|data| data.bytes)
    }

    fn load_runs(&self) -> Vec<([u8; 16], u32)> {
        let mut all = Vec::new();
        for run in 0..self.runs {
            let record = self.fetch(KIND_RUN, run).expect("run record");
            for entry in record
                .get(8..)
                .unwrap_or_default()
                .chunks_exact(format::RUN_ENTRY)
            {
                let hash: [u8; 16] = entry[..16].try_into().expect("hash");
                all.push((hash, be32(entry, 16).expect("id")));
            }
        }
        all.sort_unstable();
        all
    }

    /// Every candidate id per key hash, before the key check. A later run may
    /// hold a different key with the same hash, so every run is read.
    fn candidates(&self, hashes: &[[u8; 16]]) -> Vec<Vec<u32>> {
        if let Some(merged) = &self.merged {
            return hashes
                .iter()
                .map(|hash| format::merged_candidates(merged, hash))
                .collect();
        }
        let mut out = vec![Vec::new(); hashes.len()];
        for run in 0..self.runs {
            let record = self.fetch(KIND_RUN, run).expect("run record");
            let entries = record.get(8..).unwrap_or_default();
            for (found, hash) in out.iter_mut().zip(hashes) {
                found.extend(format::run_candidates(entries, hash));
            }
        }
        out
    }

    /// `ids` grouped by the chunk that holds them: chunk -> (slot, position).
    fn by_chunk(ids: &[Option<u32>]) -> BTreeMap<usize, Vec<(usize, usize)>> {
        let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (slot, id) in ids.iter().enumerate() {
            if let Some(id) = id {
                let (chunk, pos) = locate(*id as usize);
                groups.entry(chunk).or_default().push((slot, pos));
            }
        }
        groups
    }

    fn keys_of(&self, ids: &[Option<u32>]) -> Vec<Option<String>> {
        let mut out = vec![None; ids.len()];
        for (chunk, members) in Self::by_chunk(ids) {
            let Some(record) = self.fetch(KIND_KEY_CHUNK, chunk) else {
                continue;
            };
            for (slot, pos) in members {
                out[slot] = format::key_at(&record, pos)
                    .ok()
                    .map(|k| String::from_utf8_lossy(k).into_owned());
            }
        }
        out
    }
}

impl Layout for Proto {
    fn key_to_ids(&self, keys: &[&str]) -> Vec<Option<u32>> {
        let hashes: Vec<[u8; 16]> = keys.iter().map(|k| key_hash(k)).collect();
        let found = self.candidates(&hashes);
        if !self.confirm {
            return found.iter().map(|ids| ids.first().copied()).collect();
        }
        // One batched read of every candidate's key chunk, then confirm each key
        // against all of its candidates. An unreadable candidate is an error.
        let flat: Vec<Option<u32>> = found.iter().flatten().copied().map(Some).collect();
        let mut stored = self.keys_of(&flat).into_iter();
        found
            .iter()
            .zip(keys)
            .map(|(ids, key)| {
                let mut taken: Vec<Option<String>> = stored.by_ref().take(ids.len()).collect();
                let mut next = ids.iter().zip(taken.iter_mut());
                let hit = format::confirm(ids, key, &mut |id| {
                    next.find(|(candidate, _)| **candidate == id)
                        .and_then(|(_, key)| key.take())
                })
                .unwrap_or_else(|why| panic!("lookup failed closed: {why:?}"));
                self.rejected.set(
                    self.rejected.get().saturating_add(
                        u64::try_from(ids.len().saturating_sub(usize::from(hit.is_some())))
                            .unwrap_or(u64::MAX),
                    ),
                );
                hit
            })
            .collect()
    }

    fn id_to_keys(&self, ids: &[u32]) -> Vec<Option<String>> {
        let ids: Vec<Option<u32>> = ids.iter().copied().map(Some).collect();
        self.keys_of(&ids)
    }

    fn edges(&self, ids: &[u32]) -> Vec<Edges> {
        let ids: Vec<Option<u32>> = ids.iter().copied().map(Some).collect();
        let mut out: Vec<Edges> = vec![Edges::default(); ids.len()];
        for (chunk, members) in Self::by_chunk(&ids) {
            let record = self.fetch(KIND_EDGE_CHUNK, chunk).expect("edge chunk");
            for (slot, pos) in members {
                out[slot] = edges_at(&record, pos).expect("edge entry");
            }
        }
        out
    }

    fn rejected(&self) -> Option<u64> {
        self.confirm.then(|| self.rejected.get())
    }
}

/// The sampled workload and the ids the layout under test assigned.
struct Ctx<'a> {
    rows: &'a [Row],
    /// `rows[i]`'s id in the layout under test.
    row_ids: Vec<u32>,
    /// Row indexes of the sampled events.
    sample: &'a [usize],
    /// Absent keys, one per sampled event.
    misses: &'a [String],
}

#[derive(Clone, Copy)]
enum Op {
    KeyHit,
    KeyMiss,
    IdToKey,
    Edges,
    /// `targets_of` for prev: key -> id, edge list, resolve every target.
    PrevOf,
    /// `targets_of` for auth.
    AuthOf,
}

impl Op {
    const ALL: [Self; 6] = [
        Self::KeyHit,
        Self::KeyMiss,
        Self::IdToKey,
        Self::Edges,
        Self::PrevOf,
        Self::AuthOf,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::KeyHit => "key -> id (hit)",
            Self::KeyMiss => "key -> id (miss)",
            Self::IdToKey => "id -> key",
            Self::Edges => "edge lists (all 3)",
            Self::PrevOf => "prev_of (replay)",
            Self::AuthOf => "auth_of (replay)",
        }
    }
}

/// Source edge keys of `row`, per list: sorted and de-duplicated.
fn expected_edges(row: &Row) -> [Vec<String>; 3] {
    let tidy = |ids: &[String]| {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    [
        tidy(&row.prev),
        tidy(&row.auth),
        row.rel.clone().into_iter().collect(),
    ]
}

/// Run one call of `op` over sample positions `range`; returns mismatches
/// against the source rows. Verification is not part of the timed region.
fn check_edges(layout: &dyn Layout, ctx: &Ctx<'_>, range: &Range<usize>, got: &[Edges]) -> usize {
    let mut bad = 0_usize;
    for (pos, lists) in range.clone().zip(got) {
        let want = expected_edges(&ctx.rows[ctx.sample[pos]]);
        for (ids, want) in lists.iter().zip(&want) {
            let keys = layout.id_to_keys(ids);
            let mut have: Vec<String> = keys.into_iter().flatten().collect();
            have.sort_unstable();
            if have.len() != ids.len() || have != *want {
                bad = bad.saturating_add(1);
            }
        }
    }
    bad
}

/// One timed call of `op` plus its verification; returns the call's time and
/// the number of results that disagreed with the source.
fn call(layout: &dyn Layout, ctx: &Ctx<'_>, op: Op, range: &Range<usize>) -> (Duration, usize) {
    let rows = |i: usize| &ctx.rows[ctx.sample[i]];
    let ids = || -> Vec<u32> { range.clone().map(|i| ctx.row_ids[ctx.sample[i]]).collect() };
    match op {
        Op::KeyHit | Op::KeyMiss => {
            let hit = matches!(op, Op::KeyHit);
            let keys: Vec<&str> = range
                .clone()
                .map(|i| {
                    if hit {
                        rows(i).id.as_str()
                    } else {
                        ctx.misses[i].as_str()
                    }
                })
                .collect();
            let started = Instant::now();
            let got = layout.key_to_ids(&keys);
            let took = started.elapsed();
            let bad = range
                .clone()
                .zip(&got)
                .filter(|(i, id)| {
                    let want = hit.then(|| ctx.row_ids[ctx.sample[*i]]);
                    **id != want
                })
                .count();
            (took, bad)
        }
        Op::IdToKey => {
            let ids = ids();
            let started = Instant::now();
            let got = layout.id_to_keys(&ids);
            let took = started.elapsed();
            let bad = range
                .clone()
                .zip(&got)
                .filter(|(i, key)| key.as_deref() != Some(rows(*i).id.as_str()))
                .count();
            (took, bad)
        }
        Op::PrevOf | Op::AuthOf => {
            let which = usize::from(matches!(op, Op::AuthOf));
            let keys: Vec<&str> = range.clone().map(|i| rows(i).id.as_str()).collect();
            let started = Instant::now();
            let targets = targets_of(layout, &keys, which);
            let took = started.elapsed();
            let bad = range
                .clone()
                .zip(&targets)
                .filter(|(i, got)| {
                    let mut have = (*got).clone();
                    have.sort_unstable();
                    have != expected_edges(rows(*i))[which]
                })
                .count();
            (took, bad)
        }
        Op::Edges => {
            let ids = ids();
            let started = Instant::now();
            let got = layout.edges(&ids);
            let took = started.elapsed();
            (took, check_edges(layout, ctx, range, &got))
        }
    }
}

/// Benchmark-only replay of `MatrixAdjacency::targets_of`'s read pattern, one
/// call for the given keys: key -> id, the family's edge list, then one batched
/// id -> key resolve over every target. Returns each key's target keys.
fn targets_of(layout: &dyn Layout, keys: &[&str], which: usize) -> Vec<Vec<String>> {
    let ids = layout.key_to_ids(keys);
    let found: Vec<u32> = ids.iter().flatten().copied().collect();
    let lists = layout.list(&found, which);
    let all: Vec<u32> = lists.iter().flatten().copied().collect();
    let mut resolved = layout.id_to_keys(&all).into_iter();
    lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|_| resolved.next().flatten().unwrap_or_default())
                .collect()
        })
        .collect()
}

/// Benchmark-only replay of `RoomAuth::auth_chain_ids`' read pattern without a
/// persisted closure: key -> id, then a breadth-first walk reading the auth list
/// of each frontier in one batched call, then one id -> key resolve over every
/// member. Returns the member keys and the number of levels walked.
fn auth_chain(layout: &dyn Layout, key: &str) -> Option<(Vec<String>, usize)> {
    let root = layout.key_to_ids(&[key])[0]?;
    let mut seen: HashSet<u32> = HashSet::from([root]);
    let mut frontier = vec![root];
    let mut levels = 0_usize;
    while !frontier.is_empty() {
        let next: Vec<u32> = layout
            .list(&frontier, 1)
            .into_iter()
            .flatten()
            .filter(|id| seen.insert(*id))
            .collect();
        levels = levels.saturating_add(usize::from(!next.is_empty()));
        frontier = next;
    }
    seen.remove(&root);
    let members: Vec<u32> = seen.into_iter().collect();
    let keys = layout.id_to_keys(&members).into_iter().flatten().collect();
    Some((keys, levels))
}

/// The auth chain of `key` computed from the source rows alone.
fn reference_chain(rows: &[Row], index: &HashMap<&str, usize>, key: &str) -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut frontier = vec![key];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for k in frontier {
            if let Some(i) = index.get(k) {
                for auth in &rows[*i].auth {
                    if seen.insert(auth.as_str()) {
                        next.push(auth.as_str());
                    }
                }
            }
        }
        frontier = next;
    }
    let mut out: Vec<String> = seen.into_iter().map(str::to_owned).collect();
    out.sort_unstable();
    out
}

/// One timed auth-chain walk.
struct Walk {
    age: usize,
    members: usize,
    levels: usize,
    micros: f64,
}

/// Auth-chain walks over `count` sampled events, reported by event age (file
/// order terciles). Correctness is checked against the source outside the timed
/// region. Warm.
fn walks(name: &str, layout: &dyn Layout, rows: &[Row], count: usize) {
    let index: HashMap<&str, usize> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), i))
        .collect();
    let mut done = Vec::new();
    let mut bad = 0_usize;
    for pick in sample(rows.len(), count) {
        let key = rows[pick].id.as_str();
        let started = Instant::now();
        let got = auth_chain(layout, key);
        let took = started.elapsed();
        let Some((mut members, levels)) = got else {
            bad = bad.saturating_add(1);
            continue;
        };
        members.sort_unstable();
        if members != reference_chain(rows, &index, key) {
            bad = bad.saturating_add(1);
        }
        done.push(Walk {
            age: pick.saturating_mul(3).checked_div(rows.len()).unwrap_or(0),
            members: members.len(),
            levels,
            micros: micros(took),
        });
    }
    for (age, label) in ["oldest third", "middle third", "newest third"]
        .iter()
        .enumerate()
    {
        let group: Vec<&Walk> = done.iter().filter(|w| w.age == age).collect();
        let n = len_f64(group.len().max(1));
        let total: f64 = group.iter().map(|w| w.micros).sum();
        let members: f64 = group.iter().map(|w| len_f64(w.members)).sum();
        let levels: f64 = group.iter().map(|w| len_f64(w.levels)).sum();
        let mut sorted: Vec<f64> = group.iter().map(|w| w.micros).collect();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted
            .get(sorted.len().saturating_sub(1).saturating_mul(99) / 100)
            .copied()
            .unwrap_or(0.0);
        println!(
            "  {name:<15} auth chain (replay) {label}: {} walks, mean {:.1} members, {:.1} levels, {:.1} us/walk ({:.2} us/member), p99 {p99:.0} us",
            group.len(),
            members / n,
            levels / n,
            total / n,
            total / members.max(1.0),
        );
    }
    println!("  {name:<15} auth chain mismatches {bad}");
}

/// Latency of `n` calls of `size` operations each, and mismatches found.
struct Run {
    calls: Vec<Duration>,
    size: usize,
    bad: usize,
}

fn run(layout: &dyn Layout, ctx: &Ctx<'_>, op: Op, size: usize, n: usize) -> Run {
    let mut calls = Vec::new();
    let mut bad = 0_usize;
    let mut start = 0_usize;
    while start < n {
        let end = start.saturating_add(size).min(n);
        let (took, wrong) = call(layout, ctx, op, &(start..end));
        calls.push(took);
        bad = bad.saturating_add(wrong);
        start = end;
    }
    Run { calls, size, bad }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * MICROS
}

fn report(layout: &str, op: Op, mode: &str, run: &Run) {
    let mut sorted = run.calls.clone();
    sorted.sort_unstable();
    let total = sorted
        .iter()
        .fold(Duration::ZERO, |acc, d| acc.saturating_add(*d));
    let per_call = micros(total) / len_f64(sorted.len().max(1));
    let at = |pct: usize| {
        let i = sorted.len().saturating_sub(1).saturating_mul(pct) / 100;
        sorted.get(i).copied().map_or(0.0, micros)
    };
    let max = sorted.last().copied().map_or(0.0, micros);
    println!(
        "  {layout:<15} {:<19} {mode:<10} {:>9.2} us/op | per call: mean {per_call:>9.2} p50 {:>9.2} p99 {:>9.2} max {max:>10.2} us | mismatches {}",
        op.label(),
        per_call / len_f64(run.size),
        at(50),
        at(99),
        run.bad,
    );
}

/// Warm table: an untimed pass, then single calls and batches of `batch`.
fn warm(name: &str, layout: &dyn Layout, ctx: &Ctx<'_>, n: usize, batch: usize) {
    for op in Op::ALL {
        let _ = run(layout, ctx, op, batch, n);
        report(name, op, "single", &run(layout, ctx, op, 1, n));
        report(
            name,
            op,
            &format!("batch {batch}"),
            &run(layout, ctx, op, batch, n),
        );
    }
}

/// Cold table: per op, reopen, then evict the store's files, and time the first
/// `cold` single calls. Reopen reads the store back into the page cache, so
/// eviction must follow it; the in-memory index survives, so the calls measure
/// pack reads.
fn cold(name: &str, dir: &Path, open: &dyn Fn() -> Box<dyn Layout>, ctx: &Ctx<'_>, cold_n: usize) {
    for op in Op::ALL {
        let layout = open();
        match evict_cold(dir) {
            Ok(_) => report(name, op, "cold", &run(layout.as_ref(), ctx, op, 1, cold_n)),
            Err(why) => println!("  {name} {} cold SKIPPED: {why}", op.label()),
        }
    }
}

/// Look every key of the room up and read every id back; returns each row's id.
fn round_trip(layout: &dyn Layout, rows: &[Row]) -> Vec<u32> {
    let mut row_ids = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(1000) {
        let keys: Vec<&str> = chunk.iter().map(|r| r.id.as_str()).collect();
        let ids: Vec<u32> = layout
            .key_to_ids(&keys)
            .into_iter()
            .map(|id| id.expect("every key has an id"))
            .collect();
        let back = layout.id_to_keys(&ids);
        for (row, key) in chunk.iter().zip(back) {
            assert_eq!(key.as_deref(), Some(row.id.as_str()), "id round trip");
        }
        row_ids.extend(ids);
    }
    row_ids
}

/// Deterministic sample of `n` row indexes.
fn sample(rows: usize, n: usize) -> Vec<usize> {
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let modulus = u64::try_from(rows).unwrap_or(1).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            usize::try_from(x.checked_rem(modulus).unwrap_or(0)).unwrap_or(0)
        })
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn put_line(label: &str, stats: PutStats) {
    println!(
        "    {label:<22} {:>6} puts {:>7.0} B/put {:>9.2} MiB payload {:>9.1} ms in put ({:>8.1} us/put)",
        stats.puts,
        num(stats.bytes) / num(stats.puts.max(1)),
        mib(stats.bytes),
        stats.time.as_secs_f64() * 1000.0,
        micros(stats.time) / num(stats.puts.max(1)),
    );
}

fn report_writes(stats: &WriteStats, baseline: Duration, prototype: Duration, events: usize) {
    println!(
        "write cost (same {events} events, synced every 20 batches; prototype {} puts):",
        stats.total_puts()
    );
    println!(
        "  baseline record_rows wall {:.2} s | prototype write_chunked wall {:.2} s (sync {:.2} s)",
        baseline.as_secs_f64(),
        prototype.as_secs_f64(),
        stats.sync_time.as_secs_f64(),
    );
    put_line("run (new, per batch)", stats.runs);
    put_line("edge chunk (first write)", stats.edge_new);
    put_line("edge chunk (rewrite)", stats.edge_rewrite);
    put_line("key chunk (first write)", stats.key_new);
    put_line("key chunk (rewrite)", stats.key_rewrite);
    let rewritten = stats
        .edge_rewrite
        .bytes
        .saturating_add(stats.key_rewrite.bytes);
    println!(
        "  payload written {:.2} MiB, of which rewrites {:.2} MiB ({:.0}%)",
        mib(stats.payload_bytes()),
        mib(rewritten),
        100.0 * num(rewritten) / num(stats.payload_bytes().max(1)),
    );
}

/// Round-trip the room, then run the warm and cold tables for one layout.
fn bench_layout(
    name: &str,
    dir: &Path,
    open: &dyn Fn() -> Box<dyn Layout>,
    rows: &[Row],
    (picks, misses): (&[usize], &[String]),
    (n, batch, cold_n): (usize, usize, usize),
) {
    let started = Instant::now();
    let layout = open();
    println!(
        "[{name}] open {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let started = Instant::now();
    let row_ids = round_trip(layout.as_ref(), rows);
    println!(
        "  full-room round trip ok ({} keys, {:.2} s)",
        rows.len(),
        started.elapsed().as_secs_f64()
    );
    let ctx = Ctx {
        rows,
        row_ids,
        sample: picks,
        misses,
    };
    warm(name, layout.as_ref(), &ctx, n, batch);
    walks(
        name,
        layout.as_ref(),
        rows,
        env_usize("MTXDB_EL_WALKS", 600),
    );
    match layout.rejected() {
        Some(count) => println!("  hash matches rejected by the key check: {count}"),
        None => println!("  key check: not applicable"),
    }
    drop(layout);
    cold(name, dir, open, &ctx, cold_n);
}

fn main() {
    let Some(tsv) = std::env::var_os("MTXDB_EL_TSV").map(PathBuf::from) else {
        eprintln!("edge_layout_lookup: MTXDB_EL_TSV not set; skipping");
        return;
    };
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let n = env_usize("MTXDB_EL_OPS", 20_000);
    // Cold calls index into the same sample as the warm ones.
    let cold_n = env_usize("MTXDB_EL_COLD", 1_000).min(n);
    let rows = load(&tsv, env_usize("MTXDB_EL_EVENTS", usize::MAX));
    println!(
        "{} events, {n} sampled ops, {cold_n} cold ops, {} ids per chunk",
        rows.len(),
        common::chunk_size()
    );
    let pid = std::process::id();
    let dir_a = base.join(format!("mtxdb_bench_edgelookup_{pid}_baseline"));
    let dir_b = base.join(format!("mtxdb_bench_edgelookup_{pid}_chunked"));
    for dir in [&dir_a, &dir_b] {
        let _ = std::fs::remove_dir_all(dir);
    }

    let all = Variant {
        label: "all",
        prev: true,
        auth: true,
        rel: true,
    };
    let db = Database::open(dir_a.clone()).expect("open");
    let started = Instant::now();
    record_rows(&db, &rows, all);
    let baseline_wall = started.elapsed();
    db.edges().force_index_checkpoint().expect("checkpoint");
    drop(db);

    let db = Database::open(dir_b.clone()).expect("open");
    let started = Instant::now();
    let (ids, stats) = write_chunked(&db, &rows);
    let prototype_wall = started.elapsed();
    db.edges().force_index_checkpoint().expect("checkpoint");
    drop(db);
    let runs = rows.len().div_ceil(common::BATCH);
    let hashes: HashSet<[u8; 16]> = ids.keys.iter().map(|k| key_hash(k)).collect();
    println!(
        "on disk ({} families): baseline {:.2} MiB = {:.1} B/event, prototype {:.2} MiB = {:.1} B/event; {} ids, {} distinct 128-bit key hashes",
        all.label,
        mib(sizes(&dir_a).total()),
        per(sizes(&dir_a).total(), rows.len()),
        mib(sizes(&dir_b).total()),
        per(sizes(&dir_b).total(), rows.len()),
        ids.keys.len(),
        hashes.len(),
    );
    report_writes(&stats, baseline_wall, prototype_wall, rows.len());

    let picks = sample(rows.len(), n);
    let misses: Vec<String> = picks.iter().map(|i| format!("{}~", rows[*i].id)).collect();
    // A zero batch would never advance.
    let batch = env_usize("MTXDB_EL_BATCH", 100).max(1);
    let layouts: [(&str, &Path, Open<'_>); 4] = [
        (
            "baseline",
            &dir_a,
            Box::new(|| Box::new(Baseline::open(&dir_a))),
        ),
        (
            "proto-naive",
            &dir_b,
            Box::new(|| Box::new(Proto::open(&dir_b, false, true, runs))),
        ),
        (
            "proto-merged",
            &dir_b,
            Box::new(|| Box::new(Proto::open(&dir_b, true, true, runs))),
        ),
        (
            "proto-hash-only",
            &dir_b,
            Box::new(|| Box::new(Proto::open(&dir_b, true, false, runs))),
        ),
    ];
    for (name, dir, open) in &layouts {
        bench_layout(
            name,
            dir,
            open.as_ref(),
            &rows,
            (&picks, &misses),
            (n, batch, cold_n),
        );
    }
    for dir in [&dir_a, &dir_b] {
        let _ = std::fs::remove_dir_all(dir);
    }
}
