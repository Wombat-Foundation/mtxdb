//! End-to-end search-pack cost on real room data: index, then query.
//!
//! The corpus is a TSV (`event_id room_id sender type ts body`, `jq @tsv`
//! escaping) extracted from merged room DAGs:
//!
//! ```text
//! cat merged-*.jsonl | jq -r 'select((.content.body|type)=="string") |
//!   [.event_id,.room_id,.sender,.type,.origin_server_ts,.content.body] | @tsv' \
//!   | awk -F'\t' '!seen[$1]++' > corpus.tsv
//! MTXDB_SEARCH_CORPUS=corpus.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench search_scan
//! ```
//!
//! Times `index_many` once, then each query shape `MTXDB_SEARCH_REPS` times
//! (default 7), warm cache. Prints best/median, hits and ns per record.
//! `MTXDB_SEARCH_TERMS` overrides the common term (default `the`) and
//! `MTXDB_SEARCH_RARE` the rare one (default `wombat`).

use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mtxdb::{PackfileStorage, SearchDocument, SearchIndexes, SearchQuery};

fn unescape(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') | None => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    out
}

/// Parse the corpus; the second value is how many lines were unusable.
fn load(path: &str) -> (Vec<SearchDocument>, usize) {
    let text = std::fs::read_to_string(path).expect("read corpus");
    let total = text.lines().count();
    let docs: Vec<SearchDocument> = text
        .lines()
        .filter_map(|line| {
            let mut f = line.splitn(6, '\t');
            Some(SearchDocument {
                event_id: f.next()?.to_owned(),
                room_id: f.next()?.to_owned(),
                sender: f.next()?.to_owned(),
                event_type: f.next()?.to_owned(),
                timestamp: f.next()?.parse().ok()?,
                body: Some(unescape(f.next()?)),
            })
        })
        .collect();
    let skipped = total - docs.len();
    (docs, skipped)
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Nanoseconds per record. The count goes through `u32` (exact in `f64`); a
/// corpus past `u32::MAX` records saturates and would under-report, so it
/// fails loudly instead.
fn ns_per_record(elapsed: Duration, records: usize) -> f64 {
    let records = u32::try_from(records).expect("corpus fits u32 records");
    elapsed.as_secs_f64() * 1e9 / f64::from(records.max(1))
}

fn run(index: &SearchIndexes<'_>, label: &str, query: &SearchQuery, reps: usize, records: usize) {
    let mut hits = 0;
    let mut samples: Vec<Duration> = (0..reps)
        .map(|_| {
            let started = Instant::now();
            hits = black_box(index.search(query).expect("search")).len();
            started.elapsed()
        })
        .collect();
    samples.sort();
    let best = samples.first().copied().unwrap_or_default();
    let median = samples.get(samples.len() / 2).copied().unwrap_or_default();
    println!(
        "  {label:<28} best {:>7.1} ms  median {:>7.1} ms  {hits:>7} hits  {:>6.0} ns/record",
        millis(best),
        millis(median),
        ns_per_record(best, records),
    );
}

/// Open a fresh store under the bench root and index the whole corpus.
fn build(docs: &[SearchDocument]) -> (PathBuf, PackfileStorage, Duration) {
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");
    let dir = base.join(format!("mtxdb_bench_search_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = PackfileStorage::open(dir.clone()).expect("open store");
    let started = Instant::now();
    SearchIndexes::open(&store, "bench")
        .index_many(docs)
        .expect("index");
    let indexed = started.elapsed();
    store.sync_all().expect("sync");
    (dir, store, indexed)
}

/// Midpoint of the corpus timestamp range, without overflow.
fn mid_timestamp(lo: u64, hi: u64) -> u64 {
    lo.saturating_add(hi.abs_diff(lo) / 2)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn main() {
    let Ok(corpus) = std::env::var("MTXDB_SEARCH_CORPUS") else {
        eprintln!("search_scan: MTXDB_SEARCH_CORPUS not set; skipping");
        return;
    };
    let reps = env_or("MTXDB_SEARCH_REPS", "7")
        .parse()
        .unwrap_or(7usize)
        .max(1);
    let common = env_or("MTXDB_SEARCH_TERMS", "the");
    let rare = env_or("MTXDB_SEARCH_RARE", "wombat");
    let (docs, skipped) = load(&corpus);
    if skipped > 0 {
        eprintln!("search_scan: skipped {skipped} malformed corpus lines");
    }
    let first = docs.first().expect("corpus is not empty");
    let bytes: usize = docs
        .iter()
        .filter_map(|d| d.body.as_ref())
        .map(String::len)
        .sum();
    let (dir, store, indexed) = build(&docs);
    println!(
        "corpus: {} docs, {} MiB body text, {reps} reps, warm cache; index_many {:.0} ms",
        docs.len(),
        bytes >> 20,
        millis(indexed)
    );

    let index = SearchIndexes::open(&store, "bench");
    let n = docs.len();
    let (lo, hi) = docs.iter().fold((u64::MAX, 0), |(l, h), d| {
        (l.min(d.timestamp), h.max(d.timestamp))
    });
    let terms = |t: &[&str]| t.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let cases = [
        ("no terms (match all)", SearchQuery::default()),
        (
            "common term",
            SearchQuery {
                terms: terms(&[&common]),
                ..SearchQuery::default()
            },
        ),
        (
            "rare term",
            SearchQuery {
                terms: terms(&[&rare]),
                ..SearchQuery::default()
            },
        ),
        (
            "no match",
            SearchQuery {
                terms: terms(&["zzqxjv"]),
                ..SearchQuery::default()
            },
        ),
        (
            "two terms (AND)",
            SearchQuery {
                terms: terms(&[&common, "and"]),
                ..SearchQuery::default()
            },
        ),
        (
            "common term, limit 50",
            SearchQuery {
                terms: terms(&[&common]),
                limit: 50,
                ..SearchQuery::default()
            },
        ),
        (
            "room filter",
            SearchQuery {
                room_id: Some(first.room_id.clone()),
                ..SearchQuery::default()
            },
        ),
        (
            "sender filter + term",
            SearchQuery {
                sender: Some(first.sender.clone()),
                terms: terms(&[&common]),
                ..SearchQuery::default()
            },
        ),
        (
            "time range (newer half)",
            SearchQuery {
                time_range: Some((mid_timestamp(lo, hi), hi)),
                ..SearchQuery::default()
            },
        ),
    ];
    for (label, query) in &cases {
        run(&index, label, query, reps, n);
    }
    drop(index);
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}
