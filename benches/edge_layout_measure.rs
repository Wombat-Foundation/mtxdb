//! What the short-id edge store costs on disk for a real room.
//!
//! Reads a tab-separated room (`event_id, prev ids, auth ids, relation target`,
//! comma-separated lists; see `MTXDB_EL_TSV`), records it through
//! `ShortIdIndex::record_events` in batches of 1000, forces a full index
//! checkpoint, and prints the bytes the store occupies. Variants isolate the
//! cost: owners only (no edges), prev only, prev + auth, and all three
//! families. The difference between a variant and the owners-only store is what
//! the edge lists (and the target ids they allocate) add.
//!
//! A last section writes the same room in a prototype chunked layout through the
//! same `Database`, so pack framing and index costs are measured the same way.
//!
//! ```text
//! MTXDB_EL_TSV=room.tsv MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo +1.90.0 bench --manifest-path benches/Cargo.toml --bench edge_layout_measure
//! ```

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use mtxdb::database::Database;

#[path = "edge_layout_common.rs"]
mod common;

use common::{len_f64, load, mib, num, per, record_rows, sizes, write_chunked, Row, Variant};

fn pack_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            pack_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "pack") {
            out.push(path);
        }
    }
    Ok(())
}

/// What a frame's payload is, by its leading magic bytes.
fn kind_of(data: &[u8]) -> &'static str {
    match data.get(..4) {
        Some(b"SIDF") => "forward (key -> id)",
        Some(b"SIDR") => "reverse (id -> key)",
        Some(b"SIDE") => "edge list",
        Some(b"OWNL") => "owner log",
        Some(b"SIDC") => "counter",
        Some(b"PEDG") => "chunk: edge lists",
        Some(b"PREV") => "chunk: id -> key",
        Some(b"PFWD") => "run: key hash -> id",
        _ => "other",
    }
}

#[derive(Default)]
struct KindTotals {
    records: u64,
    payload: u64,
    frames: u64,
}

/// Classify every frame in the store's packs and report record counts, payload
/// bytes and on-disk frame bytes per kind.
fn breakdown(dir: &Path, events: usize) {
    let mut packs = Vec::new();
    pack_files(dir, &mut packs).expect("list packs");
    let mut kinds: BTreeMap<&'static str, KindTotals> = BTreeMap::new();
    for pack in &packs {
        let len = std::fs::metadata(pack).expect("stat pack").len();
        // One record of lookahead gives each frame its end offset without
        // holding the pack's payloads in memory.
        let mut records = mtxdb::packfile::scan_records_iter(pack)
            .expect("scan")
            .map(|item| item.expect("record"))
            .peekable();
        while let Some((offset, record)) = records.next() {
            let end = records.peek().map_or(len, |next| next.0);
            let totals = kinds.entry(kind_of(&record.data)).or_default();
            totals.records = totals.records.saturating_add(1);
            totals.payload = totals
                .payload
                .saturating_add(u64::try_from(record.data.len()).unwrap_or(u64::MAX));
            totals.frames = totals.frames.saturating_add(end.saturating_sub(offset));
        }
    }
    println!("  pack frames by kind ({events} events):");
    let mut all_records = 0_u64;
    for (kind, totals) in &kinds {
        all_records = all_records.saturating_add(totals.records);
        println!(
            "    {kind:<22} {:>8} records  payload {:>6.1} B/event ({:>5.1} B/rec)  frames {:>6.1} B/event ({:>5.1} B/rec)",
            totals.records,
            per(totals.payload, events),
            num(totals.payload) / num(totals.records.max(1)),
            per(totals.frames, events),
            num(totals.frames) / num(totals.records.max(1)),
        );
    }
    let on_disk = sizes(dir);
    println!(
        "    {all_records} records; checkpoint {:.1} B/record, delta log {:.1} B/record",
        num(on_disk.checkpoint) / num(all_records.max(1)),
        num(on_disk.delta) / num(all_records.max(1)),
    );
}

fn run(base: &Path, rows: &[Row], variant: Variant) -> u64 {
    let label = variant.label;
    let dir = base.join(format!(
        "mtxdb_bench_edgelayout_{}_{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let edge_count = record_rows(&db, rows, variant);
    let before = sizes(&dir);
    db.edges()
        .force_index_checkpoint()
        .expect("force checkpoint");
    let after = sizes(&dir);
    drop(db);
    println!(
        "[{label}] before forced checkpoint: packs {:.2} MiB, checkpoint {:.2}, delta {:.2} | after: packs {:.2}, checkpoint {:.2}, delta {:.2}",
        mib(before.packs),
        mib(before.checkpoint),
        mib(before.delta),
        mib(after.packs),
        mib(after.checkpoint),
        mib(after.delta),
    );
    if variant.rel {
        breakdown(&dir, rows.len());
    }
    let total = after.total();
    println!(
        "  total {:.2} MiB = {:.1} B/event, {:.1} B/edge ({edge_count} edges)",
        mib(total),
        per(total, rows.len()),
        num(total) / len_f64(edge_count.max(1)),
    );
    let _ = std::fs::remove_dir_all(&dir);
    total
}
/// Prototype layout, written through the same `Database` as the baseline.
fn chunked(base: &Path, rows: &[Row]) {
    let dir = base.join(format!(
        "mtxdb_bench_edgelayout_{}_chunked",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(dir.clone()).expect("open");
    let (ids, stats) = write_chunked(&db, rows);
    let before = sizes(&dir);
    db.edges().force_index_checkpoint().expect("checkpoint");
    let after = sizes(&dir);
    drop(db);
    println!(
        "[chunked prototype] {} ids, {} puts, {:.2} MiB of payload written",
        ids.keys.len(),
        stats.total_puts(),
        mib(stats.payload_bytes()),
    );
    println!(
        "  before forced checkpoint: packs {:.2} MiB, checkpoint {:.2}, delta {:.2} | after: packs {:.2}, checkpoint {:.2}, delta {:.2}",
        mib(before.packs),
        mib(before.checkpoint),
        mib(before.delta),
        mib(after.packs),
        mib(after.checkpoint),
        mib(after.delta),
    );
    breakdown(&dir, rows.len());
    println!(
        "  total {:.2} MiB = {:.1} B/event",
        mib(after.total()),
        per(after.total(), rows.len())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let Some(tsv) = std::env::var_os("MTXDB_EL_TSV").map(PathBuf::from) else {
        eprintln!("edge_layout_measure: MTXDB_EL_TSV not set; skipping");
        return;
    };
    let limit = std::env::var("MTXDB_EL_EVENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    let rows = load(&tsv, limit);
    println!("{} events from {}", rows.len(), tsv.display());
    let variant = |label, prev, auth, rel| Variant {
        label,
        prev,
        auth,
        rel,
    };
    let owners = run(&base, &rows, variant("owners only", false, false, false));
    let prev = run(&base, &rows, variant("prev", true, false, false));
    let with_auth = run(&base, &rows, variant("prev+auth", true, true, false));
    let all = run(
        &base,
        &rows,
        variant("prev+auth+relations", true, true, true),
    );
    println!(
        "marginal over owners-only, B/event: prev {:+.1}, auth {:+.1}, relations {:+.1}",
        (num(prev) - num(owners)) / len_f64(rows.len()),
        (num(with_auth) - num(prev)) / len_f64(rows.len()),
        (num(all) - num(with_auth)) / len_f64(rows.len()),
    );
    chunked(&base, &rows);
}
