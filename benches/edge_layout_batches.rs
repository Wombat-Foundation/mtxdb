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
//! the garbage left by rewritten chunks (live = newest payload per record id,
//! against pack bytes; compaction is simulated by copying the live records into
//! a fresh store, not the engine's path), then a CSV (also written to `MTXDB_EL_CSV` when set).
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

use common::{
    len_f64, load, mib, num, record_rows_in, record_rows_in_measured, sizes, Row, Variant, Writer,
    BATCH, SYNC_EVERY,
};

const BATCH_SIZES: [usize; 4] = [1, 2, 10, 100];

/// Batch sizes to run, from `MTXDB_EL_BATCHES` (comma-separated), or all.
fn batch_sizes() -> Vec<usize> {
    std::env::var("MTXDB_EL_BATCHES").map_or_else(
        |_| BATCH_SIZES.to_vec(),
        |v| v.split(',').filter_map(|s| s.parse().ok()).collect(),
    )
}

/// Sync-every-N-batches during the append phase, from `MTXDB_EL_SYNC_EVERY`.
/// Default 1 (sync after every batch). Larger values model a caller that
/// commits without fsyncing each time.
fn append_sync_every() -> usize {
    env_usize("MTXDB_EL_SYNC_EVERY", 1)
}
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
    sync_ms: f64,
    /// Prototype only: payload bytes put per event, rewrite share (%), puts.
    proto: Option<(f64, f64, u64)>,
    /// Prototype only: garbage after the append phase.
    garbage: Option<Garbage>,
}

/// Dead chunk versions after the append phase, measured over the whole store
/// (preload included). "Live" is the newest payload per record id.
struct Garbage {
    put_mib: f64,
    live_mib: f64,
    packs_mib: f64,
    compact_ms: f64,
    compacted_packs_mib: f64,
}

fn baseline(
    base: &Path,
    rows: &[Row],
    preload: usize,
    (batch, rep, sync_every): (usize, usize, usize),
) -> Case {
    let (dir, db) = fresh(base, "baseline");
    let (head, tail) = rows.split_at(preload);
    record_rows_in(&db, head, ALL, (BATCH, SYNC_EVERY));
    let before = sizes(&dir).total();
    let started = Instant::now();
    let (_, periodic_sync) = record_rows_in_measured(&db, tail, ALL, (batch, sync_every));
    let wall = started.elapsed();
    // Separate the final durability barrier from the append timing, matching
    // the prototype path below.
    let started = Instant::now();
    db.edges().sync_all().expect("final sync");
    let sync = periodic_sync.saturating_add(started.elapsed());
    let growth = sizes(&dir).total().saturating_sub(before);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
    Case {
        layout: "baseline",
        batch,
        rep,
        wall_us: per_event_us(wall, tail.len()),
        growth_per_event: num(growth) / len_f64(tail.len().max(1)),
        sync_ms: sync.as_secs_f64() * 1e3,
        proto: None,
        garbage: None,
    }
}

fn rewrite_bytes(stats: &common::WriteStats) -> u64 {
    stats
        .edge_rewrite
        .bytes
        .saturating_add(stats.key_rewrite.bytes)
}

fn prototype(
    base: &Path,
    rows: &[Row],
    preload: usize,
    (batch, rep, sync_every): (usize, usize, usize),
) -> Case {
    let (dir, db) = fresh(base, "prototype");
    let (head, tail) = rows.split_at(preload);
    let mut writer = Writer::default();
    writer.append(&db, head, (BATCH, SYNC_EVERY));
    writer.sync(&db);
    let before = sizes(&dir).total();
    let (put0, rewrite0) = (writer.stats.payload_bytes(), rewrite_bytes(&writer.stats));
    let (start_puts, start_sync) = (writer.stats.total_puts(), writer.stats.sync_time);
    let started = Instant::now();
    writer.append(&db, tail, (batch, sync_every));
    let wall = started.elapsed();
    let started = Instant::now();
    writer.sync(&db);
    let final_sync = started.elapsed();
    let after = sizes(&dir);
    let growth = after.total().saturating_sub(before);
    let (compact, compacted) = writer.simulate_compaction(&dir.with_extension("compact"));
    let garbage = Garbage {
        put_mib: mib(writer.stats.payload_bytes()),
        live_mib: mib(writer.live_bytes()),
        packs_mib: mib(after.packs),
        compact_ms: compact.as_secs_f64() * 1e3,
        compacted_packs_mib: mib(compacted.packs),
    };
    let put = writer.stats.payload_bytes().saturating_sub(put0);
    let rewrite = rewrite_bytes(&writer.stats).saturating_sub(rewrite0);
    let put_count = writer.stats.total_puts().saturating_sub(start_puts);
    let sync = writer
        .stats
        .sync_time
        .saturating_sub(start_sync)
        .saturating_add(final_sync);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
    Case {
        layout: "prototype",
        batch,
        rep,
        wall_us: per_event_us(wall, tail.len()),
        growth_per_event: num(growth) / len_f64(tail.len().max(1)),
        sync_ms: sync.as_secs_f64() * 1e3,
        proto: Some((
            num(put) / len_f64(tail.len().max(1)),
            100.0 * num(rewrite) / num(put.max(1)),
            put_count,
        )),
        garbage: Some(garbage),
    }
}

fn csv_line(case: &Case) -> String {
    let (put, share, puts) = case.proto.unwrap_or((0.0, 0.0, 0));
    format!(
        "{},{},{},{:.1},{:.1},{put:.1},{share:.1},{puts},{:.1}",
        case.layout, case.batch, case.rep, case.wall_us, case.growth_per_event, case.sync_ms
    )
}

fn main() {
    let tsv = PathBuf::from(std::env::var_os("MTXDB_EL_TSV").expect("set MTXDB_EL_TSV"));
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, usize::MAX);
    let preload = env_usize("MTXDB_EL_PRELOAD", 60_000).min(rows.len());
    let append = env_usize("MTXDB_EL_APPEND", 2_000).min(rows.len().saturating_sub(preload));
    let reps = env_usize("MTXDB_EL_REPS", 3);
    let sync_every = append_sync_every();
    let batches = batch_sizes();
    let rows = &rows[..preload.saturating_add(append)];
    println!(
        "{preload} events preloaded in batches of {BATCH}, then {append} appended, sync every {sync_every} batches; chunk {} ids, {reps} reps, batches {batches:?}",
        common::chunk_size()
    );
    let mut lines = vec![
        "layout,batch,rep,wall_us_per_event,growth_b_per_event,payload_b_per_event,rewrite_pct,puts,sync_ms"
            .to_owned(),
    ];
    for batch in batches {
        for rep in 0..reps {
            for case in [
                baseline(&base, rows, preload, (batch, rep, sync_every)),
                prototype(&base, rows, preload, (batch, rep, sync_every)),
            ] {
                println!(
                    "  {:<9} batch {:>4} rep {}: {:>9.1} us/event | growth {:>7.1} B/event{}",
                    case.layout,
                    case.batch,
                    case.rep,
                    case.wall_us,
                    case.growth_per_event,
                    case.proto.map_or_else(
                        || format!(" | sync {:.1} ms", case.sync_ms),
                        |(put, share, puts)| format!(
                        " | put {put:.1} B/event, rewrites {share:.0}%, {puts} puts, sync {:.1} ms",
                        case.sync_ms
                    )
                    ),
                );
                if let Some(g) = &case.garbage {
                    println!(
                        "      payload: {:.2} MiB put, {:.2} live | packs: {:.2} MiB, {:.2} after simulated compaction ({:.0}% dead), copy {:.0} ms",
                        g.put_mib,
                        g.live_mib,
                        g.packs_mib,
                        g.compacted_packs_mib,
                        100.0 * (1.0 - g.compacted_packs_mib / g.packs_mib.max(f64::MIN_POSITIVE)),
                        g.compact_ms,
                    );
                }
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
