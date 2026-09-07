use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use base64::Engine as _;
use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::{PackfileStorage, ShardPool};
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
            let _ = write!(s, "{b:02x}");
            s
        },
    )
}

pub fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Put { room, id, data } => cmd_put(cli, room, id, data),
        Commands::Get { room, id } => cmd_get(cli, room, id),
        Commands::Rooms => cmd_rooms(cli),
        Commands::Shards => cmd_shards(cli),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { path } => cmd_scan(path),
        Commands::Import { path, room } => cmd_import(cli, path, room.as_deref()),
        Commands::Repack { room, root, topo } => cmd_repack(cli, room, root, *topo),
        Commands::Delete { room, yes } => cmd_delete(cli, room, *yes),
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

fn cmd_rooms(cli: &Cli) -> anyhow::Result<()> {
    let store = open_store_read_only(cli)?;
    let summaries = store.room_summaries();
    if summaries.is_empty() {
        eprintln!("no rooms found");
    } else {
        for (i, (room_id, count, _mem)) in summaries.iter().enumerate() {
            let hex = hex_encode(room_id);
            eprintln!("  {i}: {hex} ({count} records)");
        }
    }
    Ok(())
}

/// Deliberately bypasses `PackfileStorage` entirely — it needs no room
/// data at all, so going through `open_store_read_only` would pay for
/// (and pointlessly limit itself to) rebuilding every room's index. This
/// is the one command genuinely safe to run against a directory a live
/// writer process owns: `ShardPool::open_read_only` takes no lock and
/// never touches record content, only shard file metadata.
fn cmd_shards(cli: &Cli) -> anyhow::Result<()> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let pool = ShardPool::open_read_only(dir.into()).context("failed to open store")?;
    let mut shards = pool.summaries();
    if shards.is_empty() {
        eprintln!("no shards found");
        return Ok(());
    }
    shards.sort_unstable_by_key(|s| s.shard_id);
    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>8}  {:>10}  {:>6}",
        "shard", "generation", "bytes", "writes", "written", "syncs"
    );
    for s in &shards {
        eprintln!(
            "{:>6}  {:>10}  {:>10}  {:>8}  {:>10}  {:>6}",
            s.shard_id,
            s.generation,
            fmt_bytes(s.file_bytes),
            s.stats.write_count,
            fmt_bytes(s.stats.bytes_written),
            s.stats.sync_count,
        );
    }
    eprintln!(
        "{} shards, {} retired over lifetime",
        shards.len(),
        pool.retired_count()
    );
    Ok(())
}

fn cmd_info(cli: &Cli, room: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store_read_only(cli)?;
    let hex = hex_encode(&room_id);
    match store.room_index_info(&room_id) {
        Some((len, mem)) => {
            eprintln!("room {hex}: {len} records, {mem} bytes index memory");
        }
        None => {
            eprintln!("room {hex}: not found");
        }
    }
    Ok(())
}

fn cmd_scan(path: &PathBuf) -> anyhow::Result<()> {
    let records = mtxdb::packfile::scan_packfile(path)?;
    eprintln!(
        "shard: {} bytes, {} records",
        std::fs::metadata(path)?.len(),
        records.len()
    );
    for (room_id, node_id, offset) in &records {
        let room_hex = hex_encode(room_id);
        let id_hex = hex_encode(node_id);
        eprintln!("  room={room_hex} id={id_hex} @ {offset}");
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

fn cmd_repack(cli: &Cli, room: &str, roots: &[String], topo: bool) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;

    if !roots.is_empty() {
        if !topo {
            bail!("--root requires --topo; without --topo there are no edges so only the specified roots would be kept");
        }
        let root_ids: Vec<mtxdb::NodeId> = roots
            .iter()
            .map(|r| parse_node_id(r))
            .collect::<anyhow::Result<_>>()?;
        store.set_live_roots(&room_id, root_ids);
    } else if topo {
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

fn extract_matrix_edges(_hash: &[u8; 16], data: &[u8]) -> Vec<mtxdb::NodeId> {
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
        io::stdout().flush()?;
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
