//! Measure the CSR store read paths: on-disk point lookups, the first
//! whole-scope read, the total for one pass over all scopes, and the first
//! whole-scope read after a new seal.
//!
//! ```text
//! MTXDB_BENCH_ROOT=/tmp/mtxdb-csr-read \
//!   cargo bench --manifest-path benches/Cargo.toml --bench csr_read_cache
//! ```
//!
//! Knobs: `MTXDB_CSR_SCOPES` (default 1024), `MTXDB_CSR_ROWS` (rows per scope,
//! default 128), `MTXDB_CSR_ROUNDS` (point-read rounds over 64 scopes, default
//! 5), `MTXDB_BENCH_ROOT`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mtxdb::csr::store::{CsrStore, RowRecord, ScopeId};
use mtxdb::csr::{Edge, EdgeFamily};
use mtxdb::{Database, ShardType};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn scope_id(n: usize) -> ScopeId {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&u64::try_from(n).unwrap_or(u64::MAX).to_le_bytes());
    ScopeId(bytes)
}

fn build(
    store: &CsrStore,
    db: &Database,
    scopes: usize,
    rows: usize,
    family: EdgeFamily,
) -> Duration {
    let writer = store.writer(db).expect("writer");
    let started = Instant::now();
    for scope in 0..scopes {
        let batch: Vec<RowRecord> = (0..rows)
            .map(|owner| {
                let owner = u32::try_from(owner).unwrap_or(u32::MAX);
                RowRecord {
                    family,
                    owner,
                    edges: vec![
                        Edge {
                            target: owner,
                            kind: 0,
                        },
                        Edge {
                            target: owner.saturating_add(1),
                            kind: 1,
                        },
                    ],
                }
            })
            .collect();
        writer.append_rows(scope_id(scope), &batch).expect("append");
    }
    writer.seal().expect("seal");
    started.elapsed()
}

fn point_reads(
    store: &CsrStore,
    keys: &[(EdgeFamily, u32)],
    scopes: usize,
    rounds: usize,
) -> (Duration, u64) {
    let point_scopes = scopes.min(64);
    let mut total = Duration::ZERO;
    let mut ops = 0u64;
    for _round in 0..rounds {
        for scope in 0..point_scopes {
            let started = Instant::now();
            let found = store.read_rows(scope_id(scope), keys).expect("read_rows");
            total = total.saturating_add(started.elapsed());
            ops = ops.saturating_add(1);
            assert_eq!(
                found.len(),
                keys.len(),
                "read_rows must return one entry per key"
            );
            assert!(found.iter().all(Option::is_some), "point reads must hit");
        }
    }
    (
        total
            .checked_div(u32::try_from(ops.max(1)).unwrap_or(u32::MAX))
            .unwrap_or(Duration::ZERO),
        ops,
    )
}

fn whole_scope_reads(store: &CsrStore, scopes: usize, rows: usize) -> (Duration, Duration) {
    let started = Instant::now();
    let mut first = Duration::ZERO;
    for scope in 0..scopes {
        let read = Instant::now();
        let data = store.read_scope(scope_id(scope)).expect("read_scope");
        if scope == 0 {
            first = read.elapsed();
        }
        assert_eq!(data.rows.len(), rows, "read_scope must return every row");
    }
    (first, started.elapsed())
}

fn read_after_seal(store: &CsrStore, db: &Database, family: EdgeFamily, rows: usize) -> Duration {
    let writer = store.writer(db).expect("writer");
    writer
        .append_rows(
            scope_id(0),
            &[RowRecord {
                family,
                owner: 0,
                edges: vec![Edge { target: 7, kind: 2 }],
            }],
        )
        .expect("append after seal");
    writer.seal().expect("second seal");
    let started = Instant::now();
    let data = store
        .read_scope(scope_id(0))
        .expect("read after second seal");
    let elapsed = started.elapsed();
    assert_eq!(
        data.rows.len(),
        rows,
        "read_scope after a seal must return every row"
    );
    elapsed
}

fn main() {
    let scopes = env_usize("MTXDB_CSR_SCOPES", 1024);
    let rows = env_usize("MTXDB_CSR_ROWS", 128);
    assert!(
        scopes > 0 && rows > 0,
        "scope and row counts must be nonzero"
    );
    let rounds = env_usize("MTXDB_CSR_ROUNDS", 5).max(1);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("bench root");
    let root = base.join(format!("mtxdb-csr-read-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let db = Database::open(root.clone()).expect("open database");
    let store = CsrStore::open(db.layout(), ShardType::State);
    let family = EdgeFamily::typed(1);

    let build = build(&store, &db, scopes, rows, family);
    let keys: Vec<(EdgeFamily, u32)> = (0..rows.min(16))
        .map(|owner| (family, u32::try_from(owner).unwrap_or(u32::MAX)))
        .collect();
    let (point_per_op, point_ops) = point_reads(&store, &keys, scopes, rounds);
    let (first_scope, all_scopes) = whole_scope_reads(&store, scopes, rows);
    let per_scope = all_scopes
        .checked_div(u32::try_from(scopes.max(1)).unwrap_or(u32::MAX))
        .unwrap_or(Duration::ZERO);
    let after_seal = read_after_seal(&store, &db, family, rows);

    println!(
        "scopes {scopes} x {rows} rows; build+seal {build:?}; {} MiB-ish",
        scopes.saturating_mul(rows).saturating_mul(48) >> 20
    );
    println!(
        "read_rows({} keys): {point_per_op:?}/op over {point_ops} ops ({} scopes x {rounds} rounds)",
        keys.len(),
        scopes.min(64)
    );
    println!(
        "read_scope: first {first_scope:?}, all {scopes} scopes {all_scopes:?} ({per_scope:?}/scope)"
    );
    println!("read_scope after a new seal: {after_seal:?}");

    let _ = std::fs::remove_dir_all(&root);
}
