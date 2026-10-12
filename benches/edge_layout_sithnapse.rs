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
//! `Database` and policies as the other `edge_layout_*` benches. It is not
//! sithnapse's engine, its short-id lookups, SQL fetches or closure cache. It
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

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mtxdb::database::Database;
use mtxdb::storage::{NodeData, StorageEngine};

#[path = "edge_layout_common.rs"]
mod common;

use common::{
    assign_batch, compact_latest, len_f64, load, mib, num, sizes, spread, Ids, Latest, Row,
    CHUNK_COLLECTION,
};

const KIND_OUT: u8 = 4;
const KIND_CHILD: u8 = 5;

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

impl<'a> State<'a> {
    /// Replay `rows` in batches: outbound record per event, then one
    /// read-modify-write per distinct parent in the batch.
    fn append(&mut self, db: &Database, rows: &'a [Row], (batch, sync_every): (usize, usize)) {
        let store = db.edges();
        for (n, chunk) in rows.chunks(batch).enumerate() {
            assign_batch(&mut self.ids, chunk);
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
                let id = spread(KIND_CHILD, parent);
                let mut children = store
                    .get(&CHUNK_COLLECTION, &id)
                    .expect("get")
                    .map(|d| decode(&d.bytes))
                    .unwrap_or_default();
                let mut seen: HashSet<u32> = children.iter().copied().collect();
                let before = children.len();
                children.extend(new.into_iter().filter(|c| seen.insert(*c)));
                if children.len() != before {
                    self.put(store.as_ref(), id, encode(&children));
                }
            }
            if n.saturating_add(1).checked_rem(sync_every) == Some(0) {
                self.sync(db);
            }
        }
    }
}

fn case(base: &std::path::Path, rows: &[Row], preload: usize, (batch, sync_every): (usize, usize)) {
    let dir = base.join(format!("mtxdb_bench_sithnapse_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let (head, tail) = rows.split_at(preload);
    let mut state = State::default();
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
    let events = len_f64(tail.len().max(1));
    let (took, settled) = compact_latest(&state.latest, &dir.with_extension("compact"));
    println!(
        "  batch {batch:>3}: {:>9.1} us/event | growth {:>8.1} B/event | put {:>8.1} B/event, {} puts, largest {:.1} KiB | sync {:.0} ms",
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
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, usize::MAX);
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
    for batch in batches {
        case(&base, rows, preload, (batch, sync_every));
    }
}
