//! What the short-id edge store costs on disk for a real room.
//!
//! Reads a tab-separated room (`event_id, prev ids, auth ids, relation target`,
//! comma-separated lists; see `MTXDB_EL_TSV`), records it through
//! `ShortIdIndex::record_events` in batches of 1000, and prints the bytes the
//! store occupies. Several variants isolate the cost: owners only (no edges),
//! prev only, prev + auth, and all three families. The difference between a variant
//! and the owners-only store is what the edge lists (and the target ids they
//! allocate) add.
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_measure
//! ```

use std::path::{Path, PathBuf};

use mtxdb::database::Database;
use mtxdb::layout::ShardType;
use mtxdb::short_id::{BatchEvent, EdgeFamily, EdgeKey, FamilyEdges, ShortIdIndex};

const SCOPE: [u8; 16] = [0x71; 16];
const BATCH: usize = 1000;
const PREV: EdgeFamily = EdgeFamily::plain(1);
const AUTH: EdgeFamily = EdgeFamily::plain(2);
const REL: EdgeFamily = EdgeFamily::typed(3);

struct Row {
    id: String,
    prev: Vec<String>,
    auth: Vec<String>,
    rel: Option<String>,
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
            Some(Row { id, prev, auth, rel })
        })
        .collect()
}

fn tree_bytes(dir: &Path, needle: &str) -> u64 {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += tree_bytes(&path, needle);
        } else if entry.file_name().to_string_lossy().contains(needle) {
            total += entry.metadata().map_or(0, |m| m.len());
        }
    }
    total
}

fn run(base: &Path, rows: &[Row], label: &str, prev: bool, auth: bool, rel: bool) -> u64 {
    let dir = base.join(format!("mtxdb_bench_edgelayout_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let scope = ShortIdIndex::new(ShardType::Edges, SCOPE);
    let mut edge_count = 0usize;
    for (n, chunk) in rows.chunks(BATCH).enumerate() {
        let key_lists: Vec<(Vec<EdgeKey<'_>>, Vec<EdgeKey<'_>>, Vec<EdgeKey<'_>>)> = chunk
            .iter()
            .map(|row| {
                let p: Vec<_> = if prev { row.prev.iter().map(|t| EdgeKey::plain(t.as_bytes())).collect() } else { vec![] };
                let a: Vec<_> = if auth { row.auth.iter().map(|t| EdgeKey::plain(t.as_bytes())).collect() } else { vec![] };
                let r: Vec<_> = if rel { row.rel.iter().map(|t| EdgeKey { target: t.as_bytes(), kind: 1 }).collect() } else { vec![] };
                (p, a, r)
            })
            .collect();
        let fam_lists: Vec<Vec<FamilyEdges<'_>>> = key_lists
            .iter()
            .map(|(p, a, r)| {
                let mut f = Vec::new();
                if !p.is_empty() { f.push(FamilyEdges { family: PREV, edges: p }); }
                if !a.is_empty() { f.push(FamilyEdges { family: AUTH, edges: a }); }
                if !r.is_empty() { f.push(FamilyEdges { family: REL, edges: r }); }
                f
            })
            .collect();
        edge_count += key_lists.iter().map(|(p, a, r)| p.len() + a.len() + r.len()).sum::<usize>();
        let events: Vec<BatchEvent<'_>> = chunk
            .iter()
            .zip(&fam_lists)
            .map(|(row, fams)| BatchEvent { owner: row.id.as_bytes(), families: fams, owner_payload: None })
            .collect();
        scope.record_events(&db, &events).expect("record");
        if (n + 1) % 20 == 0 {
            db.edges().sync_all().expect("sync");
        }
    }
    db.edges().sync_all().expect("sync");
    drop(db);
    let packs = tree_bytes(&dir, ".pack");
    let checkpoint = tree_bytes(&dir, "checkpoint");
    let delta = tree_bytes(&dir, "delta");
    let total = packs + checkpoint + delta;
    println!(
        "{label:<22} packs {:>7.2} MiB  checkpoint {:>6.2}  delta {:>6.2}  = {:>7.2} MiB | {:>6.1} B/event, {:>5.1} B/edge ({edge_count} edges)",
        packs as f64 / 1048576.0,
        checkpoint as f64 / 1048576.0,
        delta as f64 / 1048576.0,
        total as f64 / 1048576.0,
        total as f64 / rows.len() as f64,
        total as f64 / edge_count.max(1) as f64,
    );
    let _ = std::fs::remove_dir_all(&dir);
    total
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let limit = std::env::var("MTXDB_EL_EVENTS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, limit);
    println!("{} events from {}", rows.len(), tsv.display());
    let owners = run(&base, &rows, "owners only", false, false, false);
    let p = run(&base, &rows, "prev", true, false, false);
    let pa = run(&base, &rows, "prev+auth", true, true, false);
    let all = run(&base, &rows, "prev+auth+relations", true, true, true);
    println!(
        "marginal over owners-only: prev {:+.1} B/event, auth {:+.1}, relations {:+.1}",
        (p as f64 - owners as f64) / rows.len() as f64,
        (pa as f64 - p as f64) / rows.len() as f64,
        (all as f64 - pa as f64) / rows.len() as f64,
    );
}
