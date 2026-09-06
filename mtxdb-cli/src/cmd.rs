use std::fs;
use std::io::{self, Write};
use std::path::Path;

use anyhow::{bail, Context};
use mtxdb::packfile;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

use crate::{Cli, Commands};

pub fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Put { room, id, data } => cmd_put(cli, room, id, data),
        Commands::Get { room, id } => cmd_get(cli, room, id),
        Commands::Rooms => cmd_rooms(cli),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { path } => cmd_scan(path),
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
    PackfileStorage::open(cli.dir.clone())
        .with_context(|| format!("failed to open store at {}", cli.dir.display()))
}

fn cmd_put(cli: &Cli, room: &str, id: &str, data: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    let node_data = NodeData::new(bytes::Bytes::from(data.as_bytes().to_vec()));
    store.put(&room_id, &node_id, &node_data)?;
    eprintln!("ok");
    Ok(())
}

fn cmd_get(cli: &Cli, room: &str, id: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    match store.get(&room_id, &node_id)? {
        Some(data) => {
            io::stdout().write_all(&data.bytes)?;
            writeln!(io::stdout())?;
        }
        None => {
            eprintln!("not found");
        }
    }
    Ok(())
}

fn cmd_rooms(cli: &Cli) -> anyhow::Result<()> {
    let store = open_store(cli)?;
    let packs = store.repack_manager().packs().read();
    if packs.is_empty() {
        eprintln!("no rooms");
        return Ok(());
    }
    let mut room_ids: Vec<[u8; 16]> = packs.keys().copied().collect();
    room_ids.sort();
    for rid in &room_ids {
        let hex: String = rid.iter().map(|b| format!("{b:02x}")).collect();
        let gen = &packs[rid];
        let size = gen.file.metadata().map_or(0, |m| m.len());
        eprintln!("{hex}  pack={:02x}  size={size}", gen.pack_id);
    }
    Ok(())
}

fn cmd_info(cli: &Cli, room: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;

    let gen = store
        .repack_manager()
        .get_pack(&room_id)
        .context("room not found")?;

    let size = gen.file.metadata().map_or(0, |m| m.len());
    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();

    let index_len = store.indexes().read().get(&room_id).map_or(0, |i| i.len());

    eprintln!("room:    {hex}");
    eprintln!("pack_id: {:02x}", gen.pack_id);
    eprintln!("file:    {}", gen.path.display());
    eprintln!("size:    {size} bytes");
    eprintln!("index:   {index_len} entries");
    eprintln!("cache:   {} entries", store.cache().len());

    Ok(())
}

fn cmd_scan(path: &Path) -> anyhow::Result<()> {
    let entries = packfile::scan_packfile(path)?;
    eprintln!("{} records", entries.len());
    for (hash, offset) in &entries {
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        eprintln!("  {hex}  offset={offset}");
    }
    Ok(())
}

fn cmd_repack(cli: &Cli, room: &str, roots: &[String]) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let store = open_store(cli)?;

    let root_ids: Vec<NodeId> = roots
        .iter()
        .map(|r| parse_node_id(r))
        .collect::<anyhow::Result<_>>()?;

    store.set_live_roots(&room_id, root_ids.clone());

    let resolver = |id: &NodeId| -> Option<(NodeData, Vec<NodeId>)> {
        let data = store.get(&room_id, id).ok().flatten()?;
        Some((data, vec![]))
    };

    let gen = store
        .repack_manager()
        .repack_room(room_id, root_ids, resolver)?;

    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("repacked {hex} -> pack {:02x}", gen.pack_id);
    Ok(())
}

fn cmd_delete(cli: &Cli, room: &str, yes: bool) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let hex: String = room_id.iter().map(|b| format!("{b:02x}")).collect();

    if !yes {
        eprintln!("This will permanently delete all data for room {hex}.");
        eprintln!("Type 'yes' to confirm:");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim() != "yes" {
            eprintln!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    store.delete_room(&room_id)?;
    eprintln!("deleted {hex}");
    Ok(())
}

fn cmd_bench(cli: &Cli, count: usize) -> anyhow::Result<()> {
    let dir = cli.dir.join("__bench__");
    fs::create_dir_all(&dir)?;

    let store = PackfileStorage::open(dir.clone())?;
    let room_id = [0xAA; 16];

    // Write phase
    let t_write = std::time::Instant::now();
    for i in 0..count {
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&i.to_le_bytes());
        id[8..12].copy_from_slice(&(i as u32 + 1).to_le_bytes());
        let data = NodeData::new(bytes::Bytes::from(format!("record {i}")));
        store.put(&room_id, &id, &data)?;
    }
    let write_elapsed = t_write.elapsed();
    let writes_per_sec = count as f64 / write_elapsed.as_secs_f64();

    // Read phase (cache hit)
    store.cache().clear();
    let t_read = std::time::Instant::now();
    let mut found = 0u64;
    for i in 0..count {
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&i.to_le_bytes());
        id[8..12].copy_from_slice(&(i as u32 + 1).to_le_bytes());
        if store.get(&room_id, &id)?.is_some() {
            found += 1;
        }
    }
    let read_elapsed = t_read.elapsed();
    let reads_per_sec = count as f64 / read_elapsed.as_secs_f64();

    eprintln!("bench: {count} records");
    eprintln!("  write: {write_elapsed:.2?} ({writes_per_sec:.0}/sec)");
    eprintln!("  read:  {read_elapsed:.2?} ({reads_per_sec:.0}/sec)");
    eprintln!("  found: {found}/{count}");

    // Cleanup
    drop(store);
    let _ = fs::remove_dir_all(&dir);

    Ok(())
}
