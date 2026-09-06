use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use base64::Engine as _;
use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::PackfileStorage;
use simd_json::prelude::*;
use simd_json::OwnedValue;

use crate::{Cli, Commands};

pub fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Put { room, id, data } => cmd_put(cli, room, id, data),
        Commands::Get { room, id } => cmd_get(cli, room, id),
        Commands::Rooms => cmd_rooms(cli),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { path } => cmd_scan(path),
        Commands::Import { path, room } => cmd_import(cli, path, room.as_deref()),
        Commands::Repack { room, root } => cmd_repack(cli, room, root),
        Commands::Delete { room, yes } => cmd_delete(cli, room, *yes),
        Commands::Bench { count } => cmd_bench(cli, *count),
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

fn open_store(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    PackfileStorage::open(dir.into()).context("failed to open store")
}

fn cmd_put(cli: &Cli, room: &str, id: &str, data: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    let node_data = NodeData::new(bytes::Bytes::from(data.as_bytes().to_vec()));
    store.put(&room_id, &node_id, &node_data)?;
    let room_hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
    let id_hex: String = node_id.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("put {id_hex} into room {room_hex} ({} bytes)", data.len());
    Ok(())
}

fn cmd_get(cli: &Cli, room: &str, id: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
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
    let store = open_store(cli)?;
    let indexes = store.indexes().read();
    if indexes.is_empty() {
        eprintln!("no rooms found");
    } else {
        for (i, (room_id, idx)) in indexes.iter().enumerate() {
            let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
            eprintln!("  {i}: {hex} ({} records)", idx.len());
        }
    }
    Ok(())
}

fn cmd_info(cli: &Cli, room: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;
    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
    let indexes = store.indexes().read();
    match indexes.get(&room_id) {
        Some(idx) => {
            eprintln!(
                "room {hex}: {} records, {} bytes index memory",
                idx.len(),
                idx.memory_usage()
            );
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
        let room_hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
        let id_hex: String = node_id.iter().map(|b| format!("{b:02x}")).collect();
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
        let mut id = [0u8; 16];
        let bytes = rid.as_bytes();
        let len = bytes.len().min(16);
        id[..len].copy_from_slice(&bytes[..len]);
        id
    };

    let room_hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();

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
            let sha = match sha {
                Some(s) => s,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            let id_bytes = match base64::engine::general_purpose::STANDARD_NO_PAD.decode(sha) {
                Ok(b) if b.len() >= 16 => {
                    let mut id = [0u8; 16];
                    id.copy_from_slice(&b[..16]);
                    id
                }
                _ => {
                    skipped += 1;
                    continue;
                }
            };

            let event_bytes = ev.to_string().into_bytes();
            let data = NodeData::new(bytes::Bytes::from(event_bytes));
            store.put(&room_id, &id_bytes, &data)?;
            event_count += 1;
        }
    }

    eprintln!("imported {event_count} events to room {room_hex}");
    if skipped > 0 {
        eprintln!("skipped {skipped} events (missing sha256 hash)");
    }

    Ok(())
}

fn cmd_repack(cli: &Cli, room: &str, roots: &[String]) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;

    let root_ids: Vec<mtxdb::NodeId> = roots
        .iter()
        .map(|r| parse_node_id(r))
        .collect::<anyhow::Result<_>>()?;

    store.set_live_roots(&room_id, root_ids.clone());

    // Trigger a repack by setting a very low threshold and writing a dummy record
    store.set_repack_threshold_bytes(0);
    let dummy = mtxdb::NodeData::new(bytes::Bytes::from_static(b""));
    store.put(&room_id, &[0u8; 16], &dummy)?;

    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("repacked {hex}");
    Ok(())
}

fn cmd_delete(cli: &Cli, room: &str, yes: bool) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();

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
    let indexes = store.indexes().read();
    let count = indexes.get(&room_id).map_or(0, |idx| idx.len());
    drop(indexes);
    store.delete_room(&room_id)?;
    eprintln!("deleted {count} records for room {hex}");
    Ok(())
}

fn cmd_bench(cli: &Cli, count: usize) -> anyhow::Result<()> {
    let store = open_store(cli)?;
    let room_id = [0u8; 16];

    let payload = vec![0xABu8; 256];
    let mut ids: Vec<[u8; 16]> = Vec::with_capacity(count);

    let write_start = std::time::Instant::now();
    for i in 0..count {
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&(i as u64).to_le_bytes());
        let data = NodeData::new(bytes::Bytes::from(payload.clone()));
        store.put(&room_id, &id, &data)?;
        ids.push(id);
    }
    let write_elapsed = write_start.elapsed();

    let read_start = std::time::Instant::now();
    for id in &ids {
        let _ = store.get(&room_id, id)?;
    }
    let read_elapsed = read_start.elapsed();

    let write_ops = count as f64 / write_elapsed.as_secs_f64();
    let read_ops = count as f64 / read_elapsed.as_secs_f64();
    let mb = (count * 256) as f64 / 1_000_000.0;

    eprintln!("bench: {count} records, 256 bytes payload");
    eprintln!(
        "  write: {write_elapsed:?} ({write_ops:.0} ops/sec, {:.1} MB/s)",
        mb / write_elapsed.as_secs_f64()
    );
    eprintln!(
        "  read:  {read_elapsed:?} ({read_ops:.0} ops/sec, {:.1} MB/s)",
        mb / read_elapsed.as_secs_f64()
    );

    Ok(())
}
