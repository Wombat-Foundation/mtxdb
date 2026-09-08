use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use base64::Engine as _;
use mtxdb_core::storage::{NodeData, StorageEngine};
use mtxdb_core::PackfileStorage;
use simd_json::prelude::*;
use simd_json::OwnedValue;

use crate::{Cli, Commands};

/// Human-readable byte count (`512 B`, `4.3 KB`, `1.2 MB`, `2.1 GB`) —
/// raw byte counts in a shard listing are unreadable past a few digits.
#[allow(
    clippy::cast_precision_loss,
    reason = "display-only rounding to 1 decimal place; losing bits below f64's 52-bit mantissa at exabyte scale is invisible at that precision"
)]
fn fmt_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let max_unit = UNITS.len().saturating_sub(1);
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < max_unit {
        value /= 1024.0;
        unit = unit.saturating_add(1);
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len().saturating_mul(2)),
        |mut s, b| {
            let _ = write!(s, "{b:02X}");
            s
        },
    )
}

pub(crate) fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Put { room, id, data } => cmd_put(cli, room, id, data),
        Commands::Get { room, id } => cmd_get(cli, room, id),
        Commands::Rooms => cmd_rooms(cli),
        Commands::Shards => cmd_shards(cli),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { path } => cmd_scan(path),
        Commands::Import { path, room } => cmd_import(cli, path, room.as_deref()),
        Commands::Repack {
            room,
            shard,
            root,
            topo,
        } => cmd_repack(cli, room.as_deref(), *shard, root, *topo),
        Commands::Delete { room, yes } => cmd_delete(cli, room, *yes),
        Commands::Sync => cmd_sync(cli),
    }
}

fn parse_room_id(hex: &str) -> anyhow::Result<[u8; 16]> {
    if hex.len() != 32 {
        bail!("room ID must be 32 hex characters, got {}", hex.len());
    }
    let bytes = hex::decode(hex).context("invalid hex in room ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

fn parse_node_id(hex: &str) -> anyhow::Result<[u8; 16]> {
    if hex.len() != 32 {
        bail!("node ID must be 32 hex characters, got {}", hex.len());
    }
    let bytes = hex::decode(hex).context("invalid hex in node ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

/// Open the store as its exclusive writer. Fails fast if another process
/// (e.g. a live embedder) already holds the writer lock — required for
/// any command that mutates data.
fn open_store(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    PackfileStorage::open(dir.into()).context("failed to open store")
}

/// Open the store read-only — coexists with a live writer process rather
/// than contending with it. For commands that only ever read room data.
fn open_store_read_only(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    PackfileStorage::open_read_only(dir.into()).context("failed to open store")
}

fn cmd_put(cli: &Cli, room: &str, id: &str, data: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    let node_data = NodeData::new(bytes::Bytes::from(data.as_bytes().to_vec()));
    store.put(&room_id, &node_id, &node_data)?;
    let room_hex = hex_encode(&room_id);
    let id_hex = hex_encode(&node_id);
    eprintln!("put {id_hex} into room {room_hex} ({} bytes)", data.len());
    Ok(())
}

fn cmd_get(cli: &Cli, room: &str, id: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store_read_only(cli)?;
    match store.get(&room_id, &node_id)? {
        Some(data) => {
            io::stdout().write_all(&data.bytes)?;
            io::stdout().write_all(b"\n")?;
        }
        None => {
            bail!("not found");
        }
    }
    Ok(())
}

/// Scan every shard's packfile to count records per room — the I/O
/// (reading record headers) is unavoidable since room ownership only
/// exists inside the packfile, but this skips building a `LossyIndex`
/// and `NodeCache` per room, which `open_store_read_only` would do
/// purely to hand back counts and then throw everything away.
fn cmd_rooms(cli: &Cli) -> anyhow::Result<()> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let mut counts: std::collections::HashMap<[u8; 16], usize> = std::collections::HashMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().is_some_and(|e| e == "pack") {
            continue;
        }
        let records = mtxdb_core::packfile::scan_packfile(&path)?;
        for (room_id, _hash, _offset) in &records {
            #[allow(clippy::arithmetic_side_effects)]
            {
                counts.entry(*room_id).and_modify(|c| *c += 1).or_insert(1);
            }
        }
    }

    if counts.is_empty() {
        eprintln!("no rooms found");
        return Ok(());
    }
    let mut rooms: Vec<([u8; 16], usize)> = counts.into_iter().collect();
    rooms.sort_unstable_by_key(|&(id, _)| id);
    for (i, (room_id, count)) in rooms.iter().enumerate() {
        let hex = hex_encode(room_id);
        println!("  {i}: 0x{hex} ({count} records)");
    }
    Ok(())
}

/// Deliberately bypasses both `PackfileStorage` and `ShardPool` — it
/// needs no room data and no open file handles. Instead it globs shard
/// filenames for size/epoch, then decodes `shard_stats.bin` directly
/// for the IO/sync counters. Zero `File::open` calls; the only I/O is
/// `stat()` per shard file and one read of the small stats snapshot.
/// Safe to run against a directory a live writer process owns.
fn cmd_shards(cli: &Cli) -> anyhow::Result<()> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let mut shard_entries = glob_shard_files(dir)?;
    shard_entries.sort_unstable_by_key(|&(id, _, _)| id);

    if shard_entries.is_empty() {
        eprintln!("no shards found");
        return Ok(());
    }

    let (stats_map, persisted_at) = decode_stats_snapshot(dir);
    print_shard_table(&shard_entries, &stats_map);
    print_stats_age(persisted_at);
    Ok(())
}

/// Glob `shard_*.pack` files in `dir`, parsing `slot_id`, epoch,
/// and file size from each filename and its metadata.
fn glob_shard_files(dir: &Path) -> anyhow::Result<Vec<(u16, u64, u64)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "pack") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some(id_hex) = stem.strip_prefix("shard_") {
                    let (slot_hex, epoch) = match id_hex.split_once('_') {
                        Some((slot, gen_hex)) => {
                            let gen = u64::from_str_radix(gen_hex, 16).unwrap_or(0);
                            (slot, gen)
                        }
                        None => (id_hex, 0),
                    };
                    if let Ok(slot_id) = u16::from_str_radix(slot_hex, 16) {
                        let file_bytes = entry.metadata()?.len();
                        entries.push((slot_id, epoch, file_bytes));
                    }
                }
            }
        }
    }
    Ok(entries)
}

/// Composite `(slot_id, epoch)` key to `(write_count,
/// bytes_written, sync_count)`, as decoded from `shard_stats.bin`.
type ShardStatsMap = std::collections::HashMap<u64, (u64, u64, u64)>;

/// Decode `shard_stats.bin` — same binary format as
/// `ShardPool::restore_persisted_stats`, but standalone. Returns a map
/// from a composite `(slot_id, epoch)` key to `(write_count,
/// bytes_written, sync_count)` and the snapshot's persisted-at timestamp.
fn decode_stats_snapshot(dir: &Path) -> (ShardStatsMap, Option<u64>) {
    const STATS_MAGIC: &[u8; 4] = b"MSTA";
    const STATS_HEADER_LEN: usize = 4 + 1 + 8; // magic + version + persisted_at
    const STATS_RECORD_LEN: usize = 2 + 8 + 8 * 3;
    const EPOCH_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    let mut stats_map = std::collections::HashMap::new();
    let mut persisted_at = None;

    let stats_path = dir.join("shard_stats.bin");
    let Ok(buf) = fs::read(stats_path) else {
        return (stats_map, persisted_at);
    };
    if buf.len() < STATS_HEADER_LEN || &buf[0..4] != STATS_MAGIC || buf[4] != 3 {
        return (stats_map, persisted_at);
    }

    persisted_at = Some(u64::from_le_bytes(buf[5..13].try_into().unwrap_or([0; 8])));
    let body = &buf[STATS_HEADER_LEN..];
    for chunk in body.chunks(STATS_RECORD_LEN) {
        if let Ok(rec) = <&[u8; STATS_RECORD_LEN]>::try_from(chunk) {
            let shard_id = u16::from_le_bytes(rec[0..2].try_into().unwrap());
            let epoch = u64::from_le_bytes(rec[2..10].try_into().unwrap());
            let write_count = u64::from_le_bytes(rec[10..18].try_into().unwrap());
            let bytes_written = u64::from_le_bytes(rec[18..26].try_into().unwrap());
            let sync_count = u64::from_le_bytes(rec[26..34].try_into().unwrap());
            let key = u64::from(shard_id) << 48 | (epoch & EPOCH_MASK);
            stats_map.insert(key, (write_count, bytes_written, sync_count));
        }
    }
    (stats_map, persisted_at)
}

/// Print the shard table header and rows.
fn print_shard_table(shard_entries: &[(u16, u64, u64)], stats_map: &ShardStatsMap) {
    const EPOCH_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    println!(
        "{:>6}  {:>18}  {:>10}  {:>8}  {:>10}  {:>6}",
        "slot", "epoch", "bytes", "writes", "written", "syncs"
    );
    for &(slot_id, epoch, file_bytes) in shard_entries {
        let key = u64::from(slot_id) << 48 | (epoch & EPOCH_MASK);
        let (wc, bw, sc) = stats_map.get(&key).copied().unwrap_or_default();
        println!(
            "{:>6}  {:>18}  {:>10}  {:>8}  {:>10}  {:>6}",
            slot_id,
            format!("{epoch:#018x}"),
            fmt_bytes(file_bytes),
            wc,
            fmt_bytes(bw),
            sc,
        );
    }
    println!("{} shards", shard_entries.len());
}

/// Print how old the stats snapshot is (or that none exists).
fn print_stats_age(persisted_at: Option<u64>) {
    match persisted_at {
        Some(ts) => {
            let age_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |now| now.as_secs().saturating_sub(ts));
            eprintln!(
                "stats snapshot: {} old (counters above may lag a live writer between its flushes)",
                fmt_duration(age_secs)
            );
        }
        None => eprintln!(
            "stats snapshot: none persisted yet — counters above are all zero by default, not necessarily real"
        ),
    }
}

/// Formats a duration in seconds as a short human-readable age string.
fn fmt_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

fn cmd_info(cli: &Cli, room: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store_read_only(cli)?;
    let hex = hex_encode(&room_id);
    match store.room_index_info(&room_id) {
        Some((len, mem)) => {
            println!("room {hex}: {len} records, {mem} bytes index memory");
        }
        None => {
            eprintln!("room {hex}: not found");
        }
    }
    Ok(())
}

fn cmd_scan(path: &PathBuf) -> anyhow::Result<()> {
    let records = mtxdb_core::packfile::scan_packfile(path)?;
    println!(
        "shard: {} bytes, {} records",
        std::fs::metadata(path)?.len(),
        records.len()
    );
    for (room_id, node_id, offset) in &records {
        let room_hex = hex_encode(room_id);
        let id_hex = hex_encode(node_id);
        println!("  room={room_hex} id={id_hex} @ {offset}");
    }
    Ok(())
}

fn cmd_import(cli: &Cli, path: &Path, room_override: Option<&str>) -> anyhow::Result<()> {
    let mut content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let val: OwnedValue = simd_json::to_owned_value(&mut content).context("invalid JSON")?;

    let mut event_count = 0u64;
    let mut skipped = 0u64;

    let store = open_store(cli)?;

    let room_id = if let Some(r) = room_override {
        parse_room_id(r)?
    } else {
        let rid = match &val {
            OwnedValue::Object(obj) => match obj.get("room_id") {
                Some(OwnedValue::String(s)) => Some(s.as_str()),
                _ => None,
            },
            _ => None,
        }
        .context("could not detect room_id")?;
        let hash = blake3::hash(rid.as_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        id
    };

    let room_hex = hex_encode(&room_id);

    let arr_pdus = val["pdus"].as_array();
    let arr_auth = val["auth_chain"].as_array();
    for arr in [arr_pdus, arr_auth].into_iter().flatten() {
        for ev in arr {
            let sha = match ev {
                OwnedValue::Object(obj) => match obj.get("hashes") {
                    Some(OwnedValue::Object(h)) => match h.get("sha256") {
                        Some(OwnedValue::String(s)) => Some(s.as_str()),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            };
            let Some(sha) = sha else {
                skipped = skipped.saturating_add(1);
                continue;
            };

            let id_bytes = match base64::engine::general_purpose::STANDARD_NO_PAD.decode(sha) {
                Ok(b) if b.len() >= 16 => {
                    let mut id = [0u8; 16];
                    id.copy_from_slice(&b[..16]);
                    id
                }
                _ => {
                    skipped = skipped.saturating_add(1);
                    continue;
                }
            };

            let event_bytes = ev.to_string().into_bytes();
            let data = NodeData::new(bytes::Bytes::from(event_bytes));
            store.put(&room_id, &id_bytes, &data)?;
            event_count = event_count.saturating_add(1);
        }
    }

    eprintln!("imported {event_count} events to room {room_hex}");
    if skipped > 0 {
        eprintln!("skipped {skipped} events (missing sha256 hash)");
    }

    Ok(())
}

fn cmd_repack(
    cli: &Cli,
    room: Option<&str>,
    shard: Option<u16>,
    roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    match (room, shard) {
        (Some(room), None) => cmd_repack_room(cli, room, roots, topo),
        (None, Some(shard_id)) => cmd_repack_shard(cli, shard_id, roots, topo),
        // clap's ArgGroup(required, conflicting) already rules both of
        // these out before we get here; kept as a hard error rather than
        // silently picking one, since reaching it means that guarantee
        // broke.
        _ => bail!("exactly one of --room or --shard is required"),
    }
}

fn cmd_repack_room(cli: &Cli, room: &str, roots: &[String], topo: bool) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;

    if !roots.is_empty() {
        if !topo {
            bail!("--root requires --topo; without --topo there are no edges so only the specified roots would be kept");
        }
        // Imported Matrix `prev_events` entries are event IDs, whereas this
        // store is indexed by the truncated content hashes in `hashes.sha256`.
        // We do not persist an event-ID → NodeId map, so treating those IDs as
        // hashes would make a rooted repack retain only the supplied roots and
        // silently collect their ancestors. Refuse the destructive operation
        // until that mapping is available.
        bail!(
            "--root cannot be used with Matrix topology yet: imported event IDs cannot be resolved to stored node IDs; refusing a rooted repack that could discard ancestors"
        );
    }
    if topo {
        eprintln!("warning: --topo without --root means no GC; all records preserved");
    }

    // repack_room_reachable is the engine's one repack entry point (see
    // mtxdb-core). --topo controls whether real prev_events-derived edges
    // are used for ordering; without it, every record is treated as its
    // own root (dedup only, no dependency ordering).
    let hex = hex_encode(&room_id);
    let (kept, dropped) = if topo {
        eprintln!("warning: edge extraction is approximate; prev_events event IDs are not resolved to stored node hashes");
        store.repack_room_reachable(&room_id, extract_matrix_edges)?
    } else {
        store.repack_room_reachable(&room_id, |_hash, _data| Vec::new())?
    };
    eprintln!("repacked {hex}: {kept} kept, {dropped} dropped");

    Ok(())
}

/// Compacts every shard transitively touched by rooms referencing
/// `shard_id` — not just `shard_id` itself. A room's live data can span
/// more than one shard (`PackfileStorage::repack_closure` finds the full
/// closure), and repacking a room always rewrites its *entire* live set
/// regardless of which shard triggered the repack, so a shard-scoped
/// compaction has to account for everything that repack will actually
/// touch, not just the one shard named on the command line.
///
/// Runs a non-mutating preflight first (`PackfileStorage::plan_room_repack`
/// per room in the closure): prints how many rooms and shards are
/// involved and the expected shard count/slack after compaction, then
/// prompts for confirmation before performing any real repack. `--root`
/// doesn't apply here since live roots are inherently per-room — the
/// same `--topo`/no-`--topo` edge-extraction choice applies uniformly
/// across every room the closure touches.
///
/// The preflight runs against a **read-only** open, not the exclusive
/// writer — an interactive confirmation pause can take arbitrarily long,
/// and holding the writer lock for that whole window would lock out a
/// real writer process (e.g. Synapse) for however long the person takes
/// to respond. The writer lock is only acquired after confirmation, and
/// the closure is recomputed fresh under it before any repack runs — if
/// it's grown since the read-only preview (a write landed in the gap),
/// that's reported rather than silently repacking a larger scope than
/// what was shown.
fn cmd_repack_shard(
    cli: &Cli,
    shard_id: u16,
    live_roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    if !live_roots.is_empty() {
        bail!("--root requires --room — live roots are per-room, not meaningful for --shard");
    }
    if topo {
        eprintln!("warning: edge extraction is approximate; prev_events event IDs are not resolved to stored node hashes");
    } else {
        eprintln!("warning: --topo without --root means no GC; all records preserved");
    }

    let (preview_rooms, preview_shards) = {
        let preview_store = open_store_read_only(cli)?;
        let (rooms, shards) = preview_store.repack_closure(shard_id)?;
        if rooms.is_empty() {
            eprintln!("no rooms reference shard {shard_id}");
            return Ok(());
        }

        let mut total_kept_bytes: u64 = 0;
        let mut total_kept = 0usize;
        let mut total_dropped = 0usize;
        for room_id in &rooms {
            let plan = if topo {
                preview_store.plan_room_repack(room_id, extract_matrix_edges)?
            } else {
                preview_store.plan_room_repack(room_id, |_hash, _data| Vec::new())?
            };
            total_kept_bytes = total_kept_bytes.saturating_add(plan.kept_bytes);
            total_kept = total_kept.saturating_add(plan.kept);
            total_dropped = total_dropped.saturating_add(plan.dropped);
        }

        let max_shard_bytes = mtxdb_core::shard::MAX_SHARD_BYTES;
        let expected_shards = total_kept_bytes
            .checked_add(max_shard_bytes.saturating_sub(1))
            .map_or(1, |rounded| rounded / max_shard_bytes)
            .max(1);
        let slop = expected_shards
            .saturating_mul(max_shard_bytes)
            .saturating_sub(total_kept_bytes);

        eprintln!(
            "this will repack {} room{} across {} shard{} ({shards:?})",
            rooms.len(),
            if rooms.len() == 1 { "" } else { "s" },
            shards.len(),
            if shards.len() == 1 { "" } else { "s" },
        );
        eprintln!(
            "expected result: ~{expected_shards} shard{} ({} kept, {} dropped, ~{} slack in the last shard)",
            if expected_shards == 1 { "" } else { "s" },
            total_kept,
            total_dropped,
            fmt_bytes(slop),
        );
        (rooms, shards)
        // preview_store (and its non-exclusive read-only handle) drops
        // here, before the confirmation prompt — nothing about a
        // read-only open blocks a real writer anyway, but there's no
        // reason to keep it open through an indefinite human pause.
    };

    eprint!("press Enter to continue, Ctrl+C to abort: ");
    io::stderr().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    // Only now do we take the exclusive writer lock — and the very first
    // thing done under it is recomputing the closure fresh, since the
    // read-only preview above is, by construction, a snapshot that could
    // be arbitrarily stale by the time a human finishes reading it.
    let store = open_store(cli)?;
    let (rooms, touched_shards) = store.repack_closure(shard_id)?;
    if rooms.is_empty() {
        eprintln!("shard {shard_id} is no longer referenced by any room — nothing to do");
        return Ok(());
    }
    let grew = rooms.iter().any(|r| !preview_rooms.contains(r))
        || touched_shards.iter().any(|s| !preview_shards.contains(s));
    if grew {
        eprintln!(
            "note: the closure grew since the preview (now {} rooms / {} shards) — repacking the current, authoritative closure",
            rooms.len(),
            touched_shards.len()
        );
    }

    let mut results = Vec::with_capacity(rooms.len());
    for room_id in &rooms {
        let (kept, dropped) = if topo {
            store.repack_room_reachable(room_id, extract_matrix_edges)?
        } else {
            store.repack_room_reachable(room_id, |_hash, _data| Vec::new())?
        };
        eprintln!(
            "repacked {}: {kept} kept, {dropped} dropped",
            hex_encode(room_id)
        );
        results.push((kept, dropped));
    }
    let (final_kept, final_dropped) = results
        .iter()
        .fold((0usize, 0usize), |(k, d), &(kept, dropped)| {
            (k.saturating_add(kept), d.saturating_add(dropped))
        });
    eprintln!(
        "done: {} rooms repacked, {final_kept} kept, {final_dropped} dropped",
        results.len()
    );
    Ok(())
}

fn extract_matrix_edges(_hash: &[u8; 16], data: &[u8]) -> Vec<mtxdb_core::NodeId> {
    let mut input = data.to_vec();
    let val: simd_json::OwnedValue = match simd_json::to_owned_value(&mut input) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut edges = Vec::new();
    if let Some(prev) = val["prev_events"].as_array() {
        for ev in prev {
            if let Some(s) = ev.as_str() {
                if let Ok(bytes) = base64::engine::general_purpose::STANDARD_NO_PAD.decode(s) {
                    if bytes.len() >= 16 {
                        let mut id = [0u8; 16];
                        id.copy_from_slice(&bytes[..16]);
                        edges.push(id);
                    }
                }
            }
        }
    }
    edges
}

fn cmd_delete(cli: &Cli, room: &str, yes: bool) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let hex = hex_encode(&room_id);

    if !yes {
        eprintln!("This will permanently delete all data for room {hex}.");
        eprint!("Are you sure? [y/N] ");
        io::stderr().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    let count = store.room_index_info(&room_id).map_or(0, |(len, _)| len);
    store.delete_room(&room_id)?;
    eprintln!("deleted {count} records for room {hex}");
    Ok(())
}

/// Opens the store as writer (which always does a full scan and builds
/// the shard→room directory and shard stats in memory regardless of
/// whether either has ever been persisted), then persists both —
/// bootstrapping `shard_stats.bin`/`shard_rooms.bin` for a store whose
/// writer process has never called `sync_all`, or just refreshing them
/// on demand.
fn cmd_sync(cli: &Cli) -> anyhow::Result<()> {
    let store = open_store(cli)?;
    store.sync_all()?;
    eprintln!("synced: persisted shard IO stats and shard\u{2192}room directory");
    Ok(())
}
