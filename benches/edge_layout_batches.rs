//! Write cost when events arrive in small batches.
//!
//! The other `edge_layout_*` benches write in batches of 1,000. Events arrive
//! one or two at a time, and the prototype rewrites every chunk a batch touches,
//! so small batches may amplify. This preloads the first `MTXDB_EL_PRELOAD`
//! (default 60,000) events in batches of 1,000 into the baseline store and into
//! the prototype, then appends the next `MTXDB_EL_APPEND` (default 2,000) in
//! batches of 1, 2, 10 and 100 with a sync after every batch, so each arrival is
//! durable. Per case (`MTXDB_EL_REPS`, default 3, fresh stores each) it prints wall time, on-disk growth (packs +
//! checkpoint + delta, no forced checkpoint) and, for the prototype, payload
//! bytes put, the share that rewrites an earlier chunk, puts and sync time, then
//! a CSV (also written to `MTXDB_EL_CSV` when set).
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_batches
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::database::Database;

#[path = "edge_layout_common.rs"]
mod common;

use common::{len_f64, load, num, record_rows_in, sizes, Row, Variant, Writer, BATCH, SYNC_EVERY};

const BATCH_SIZES: [usize; 4] = [1, 2, 10, 100];
const ALL: Variant = Variant {
    label: "all",
    prev: true,
    auth: true,
    rel: true,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn fresh(base: &Path, label: &str) -> (PathBuf, Database) {
    let dir = base.join(format!(
        "mtxdb_bench_edgebatches_{}_{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    (dir, db)
}

fn per_event_us(wall: Duration, events: usize) -> f64 {
    wall.as_secs_f64() * 1e6 / len_f64(events.max(1))
}

/// One measured case.
struct Case {
    layout: &'static str,
    batch: usize,
    rep: usize,
    wall_us: f64,
    growth_per_event: f64,
    /// Prototype only: payload bytes put per event, rewrite share (%), puts, sync ms.
    proto: Option<(f64, f64, u64, f64)>,
}

fn baseline(base: &Path, rows: &[Row], preload: usize, (batch, rep): (usize, usize)) -> Case {
    let (dir, db) = fresh(base, "baseline");
    let (head, tail) = rows.split_at(preload);
    record_rows_in(&db, head, ALL, (BATCH, SYNC_EVERY));
    let before = sizes(&dir).total();
    let started = Instant::now();
    record_rows_in(&db, tail, ALL, (batch, 1));
    let wall = started.elapsed();
    let growth = sizes(&dir).total().saturating_sub(before);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
    Case {
        layout: "baseline",
        batch,
        rep,
        wall_us: per_event_us(wall, tail.len()),
        growth_per_event: num(growth) / len_f64(tail.len().max(1)),
        proto: None,
    }
}

fn rewrite_bytes(stats: &common::WriteStats) -> u64 {
    stats
        .edge_rewrite
        .bytes
        .saturating_add(stats.key_rewrite.bytes)
}

fn prototype(base: &Path, rows: &[Row], preload: usize, (batch, rep): (usize, usize)) -> Case {
    let (dir, db) = fresh(base, "prototype");
    let (head, tail) = rows.split_at(preload);
    let mut writer = Writer::default();
    writer.append(&db, head, (BATCH, SYNC_EVERY));
    writer.sync(&db);
    let before = sizes(&dir).total();
    let (put0, rewrite0) = (writer.stats.payload_bytes(), rewrite_bytes(&writer.stats));
    let (start_puts, start_sync) = (writer.stats.total_puts(), writer.stats.sync_time);
    let started = Instant::now();
    writer.append(&db, tail, (batch, 1));
    let wall = started.elapsed();
    let growth = sizes(&dir).total().saturating_sub(before);
    let put = writer.stats.payload_bytes().saturating_sub(put0);
    let rewrite = rewrite_bytes(&writer.stats).saturating_sub(rewrite0);
    let put_count = writer.stats.total_puts().saturating_sub(start_puts);
    let sync = writer.stats.sync_time.saturating_sub(start_sync);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
    Case {
        layout: "prototype",
        batch,
        rep,
        wall_us: per_event_us(wall, tail.len()),
        growth_per_event: num(growth) / len_f64(tail.len().max(1)),
        proto: Some((
            num(put) / len_f64(tail.len().max(1)),
            100.0 * num(rewrite) / num(put.max(1)),
            put_count,
            sync.as_secs_f64() * 1e3,
        )),
    }
}

fn csv_line(case: &Case) -> String {
    let (put, share, puts, sync) = case.proto.unwrap_or((0.0, 0.0, 0, 0.0));
    format!(
        "{},{},{},{:.1},{:.1},{put:.1},{share:.1},{puts},{sync:.1}",
        case.layout, case.batch, case.rep, case.wall_us, case.growth_per_event
    )
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, usize::MAX);
    let preload = env_usize("MTXDB_EL_PRELOAD", 60_000).min(rows.len());
    let append = env_usize("MTXDB_EL_APPEND", 2_000).min(rows.len().saturating_sub(preload));
    let reps = env_usize("MTXDB_EL_REPS", 3);
    let rows = &rows[..preload.saturating_add(append)];
    println!(
        "{preload} events preloaded in batches of {BATCH}, then {append} appended, sync after every batch; chunk {} ids, {reps} reps",
        common::chunk_size()
    );
    let mut lines = vec![
        "layout,batch,rep,wall_us_per_event,growth_b_per_event,payload_b_per_event,rewrite_pct,puts,sync_ms"
            .to_owned(),
    ];
    for batch in BATCH_SIZES {
        for rep in 0..reps {
            for case in [
                baseline(&base, rows, preload, (batch, rep)),
                prototype(&base, rows, preload, (batch, rep)),
            ] {
                println!(
                    "  {:<9} batch {:>4} rep {}: {:>9.1} us/event | growth {:>7.1} B/event{}",
                    case.layout,
                    case.batch,
                    case.rep,
                    case.wall_us,
                    case.growth_per_event,
                    case.proto.map_or_else(String::new, |(put, share, puts, sync)| format!(
                        " | put {put:.1} B/event, rewrites {share:.0}%, {puts} puts, sync {sync:.0} ms"
                    )),
                );
                lines.push(csv_line(&case));
            }
        }
    }
    let csv = lines.join("\n");
    println!("{csv}");
    if let Some(path) = std::env::var_os("MTXDB_EL_CSV") {
        std::fs::write(path, format!("{csv}\n")).expect("write csv");
    }
}
