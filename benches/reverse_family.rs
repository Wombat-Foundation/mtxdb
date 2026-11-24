//! Measure the cost of deriving a parent -> children view from the forward
//! prev rows: `CsrStore::read_family` (what `reverse_family` reads), the
//! `ReverseCsr` build, and lookups on the built view.
//!
//! This is the request-scoped alternative to a persisted reverse index: build
//! once, then answer K lookups. Compare `build` / K plus `lookup` against the
//! per-lookup cost of the persisted index.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/tmp/mtxdb-reverse \
//!   cargo bench --manifest-path benches/Cargo.toml --bench reverse_family
//! ```
//!
//! Knobs: `MTXDB_REV_SIZES` (comma list of event counts, default
//! `10000,100000,1000000`), `MTXDB_REV_SEGMENTS` (sealed segments before the
//! live tail, default 4), `MTXDB_REV_LOOKUPS` (default 1000), `MTXDB_BENCH_ROOT`.
//! Each event gets a prev edge to its predecessor and, for every fourth event,
//! a second edge to an earlier event, so some parents have several children.
//! A second family has an empty row for every owner, to measure per-row work
//! without edge payloads. A typed sparse family models relation rows among a
//! much denser family.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mtxdb::csr::store::{CsrStore, RowRecord, ScopeId};
use mtxdb::csr::{Edge, EdgeFamily, ReverseCsr};
use mtxdb::{Database, ShardType};

const BATCH: usize = 4096;
const SPARSE_STRIDE: usize = 1000;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn sizes() -> Vec<usize> {
    std::env::var("MTXDB_REV_SIZES")
        .unwrap_or_else(|_| "10000,100000,1000000".to_owned())
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

fn u(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn edges_of(owner: usize) -> Vec<Edge> {
    let mut edges = vec![Edge {
        target: u(owner.saturating_sub(1)),
        kind: 0,
    }];
    if owner.is_multiple_of(4) && owner > 8 {
        edges.push(Edge {
            // rotate_left gives deterministic spread without an overflowing
            // multiplication or architecture-dependent truncation.
            target: u(owner
                .rotate_left(13)
                .checked_rem(owner)
                .expect("owner is nonzero in this branch")),
            kind: 0,
        });
    }
    edges
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let value = f();
    (value, started.elapsed())
}

#[allow(clippy::too_many_lines)]
fn run(base: &std::path::Path, events: usize, segments: usize, lookups: usize) {
    let root = base.join(format!("mtxdb-reverse-{}-{events}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let db = Database::open(root.clone()).expect("open database");
    let store = CsrStore::open(db.layout(), ShardType::State);
    let family = EdgeFamily::plain(0);
    let scope = ScopeId([7; 16]);
    let writer = store.writer(&db).expect("writer");
    assert!(u32::try_from(events).is_ok(), "event ids must fit u32");
    let empty_family = EdgeFamily::plain(1);
    let sparse_family = EdgeFamily::typed(2);
    let segment_count = segments.checked_add(1).expect("segment count overflow");
    let per_segment = events
        .checked_div(segment_count.max(1))
        .expect("segment divisor is nonzero");
    let mut next = 0usize;
    let mut sealed = 0usize;
    while next < events {
        let end = next.saturating_add(BATCH).min(events);
        let mut batch = Vec::with_capacity(end.saturating_sub(next).saturating_mul(3));
        for owner in next..end {
            batch.push(RowRecord {
                family,
                owner: u(owner),
                edges: edges_of(owner),
            });
            batch.push(RowRecord {
                family: empty_family,
                owner: u(owner),
                edges: Vec::new(),
            });
            if owner > 0 && owner.is_multiple_of(SPARSE_STRIDE) {
                batch.push(RowRecord {
                    family: sparse_family,
                    owner: u(owner),
                    edges: vec![Edge {
                        target: u(owner.saturating_sub(1)),
                        kind: 1,
                    }],
                });
            }
        }
        writer.append_rows(scope, &batch).expect("append");
        next = end;
        let next_boundary = sealed
            .checked_add(1)
            .and_then(|count| per_segment.checked_mul(count));
        if sealed < segments && next_boundary.is_some_and(|boundary| next >= boundary) {
            writer.seal().expect("seal");
            sealed = sealed.checked_add(1).expect("sealed count overflow");
        }
    }

    // Every owner has a row in this family, but the rows contain no edges.
    let (empty_rows, empty_first) = timed(|| {
        store
            .read_family(scope, empty_family)
            .expect("empty read_family")
    });
    assert_eq!(empty_rows.len(), events, "empty family row count");
    assert!(
        empty_rows.iter().all(|(_, edges)| edges.is_empty()),
        "empty family should contain no edge payloads"
    );
    let (_, empty_repeat) = timed(|| {
        store
            .read_family(scope, empty_family)
            .expect("empty read_family")
    });

    // A few typed rows amid the dense PREV and empty-edge families exercise
    // the family's directory-range lookup and bounded payload read.
    let (sparse_rows, sparse_first) = timed(|| {
        store
            .read_family(scope, sparse_family)
            .expect("sparse read_family")
    });
    let expected_sparse_rows = events.saturating_sub(1) / SPARSE_STRIDE;
    assert_eq!(
        sparse_rows.len(),
        expected_sparse_rows,
        "sparse family row count"
    );
    assert!(
        sparse_rows
            .iter()
            .all(|(_, edges)| edges.len() == 1 && edges[0].kind == 1),
        "each sparse family row should contain one typed edge"
    );
    let (_, sparse_repeat) = timed(|| {
        store
            .read_family(scope, sparse_family)
            .expect("sparse read_family")
    });

    // First populated-family read, then a repeat.
    let (rows, first_read) = timed(|| store.read_family(scope, family).expect("read_family"));
    let (_, second_read) = timed(|| store.read_family(scope, family).expect("read_family"));
    let edge_count: usize = rows.iter().map(|(_, e)| e.len()).sum();
    let (rev, build) =
        timed(|| ReverseCsr::from_forward_rows(family, &rows).expect("reverse build"));

    let parents = rev.parents();
    let (hits, lookup_total) = timed(|| {
        let mut children = 0usize;
        for i in 0..lookups {
            if !parents.is_empty() {
                let slot = i
                    .checked_mul(7919)
                    .expect("lookup index multiplication overflow")
                    .checked_rem(parents.len())
                    .expect("parent list is nonempty");
                let parent = parents[slot];
                children = children
                    .checked_add(rev.children_of(parent).map_or(0, <[Edge]>::len))
                    .expect("child count overflow");
            }
        }
        children
    });
    let per_lookup = lookup_total
        .checked_div(u32::try_from(lookups.max(1)).unwrap_or(u32::MAX))
        .unwrap_or(Duration::ZERO);
    let full = first_read.saturating_add(build);
    println!(
        "events {events:>8} edges {edge_count:>8} sealed {sealed}+tail | empty-edge {empty_first:>10.2?}/{empty_repeat:>10.2?} | sparse {}/{sparse_first:>10.2?}/{sparse_repeat:>10.2?} | populated {first_read:>10.2?}/{second_read:>10.2?} | build {build:>10.2?} | full derive {full:>10.2?}",
        sparse_rows.len()
    );
    println!(
        "{:>60} {lookups} lookups: {per_lookup:?}/lookup ({hits} children); derive amortised over K={lookups}: {:?}/lookup; over K=10: {:?}/lookup",
        "",
        full.checked_div(u32::try_from(lookups.max(1)).unwrap_or(u32::MAX)).unwrap_or(Duration::ZERO),
        full.checked_div(10).unwrap_or(Duration::ZERO),
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn main() {
    let segments = env_usize("MTXDB_REV_SEGMENTS", 4);
    let lookups = env_usize("MTXDB_REV_LOOKUPS", 1000);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("bench root");
    for events in sizes() {
        run(&base, events, segments, lookups);
    }
}
