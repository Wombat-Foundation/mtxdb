use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use mtxdb_core::shard::ShardPool;
use mtxdb_core::storage::{NodeData, StorageEngine};
use mtxdb_core::{DatabaseLayout, PackfileStorage, ShardType};
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

/// Decimal megabytes, rounded exactly to five fractional digits, for index
/// memory shown in human-facing room output.
fn fmt_megabytes(bytes: usize) -> String {
    const BYTES_PER_MB: u64 = 1_000_000;
    const FRACTION_SCALE: u64 = 100_000;

    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    let mut whole = bytes / BYTES_PER_MB;
    let fractional_bytes = bytes % BYTES_PER_MB;
    let mut fraction = fractional_bytes
        .saturating_mul(FRACTION_SCALE)
        .saturating_add(BYTES_PER_MB / 2)
        / BYTES_PER_MB;
    if fraction == FRACTION_SCALE {
        whole = whole.saturating_add(1);
        fraction = 0;
    }
    format!("{whole}.{fraction:05} MB")
}

/// Physical disk use, where millibyte precision is enough to distinguish
/// small records without making large-room listings visually noisy.
fn fmt_disk_megabytes(bytes: u64) -> String {
    const BYTES_PER_MB: u64 = 1_000_000;
    const FRACTION_SCALE: u64 = 1_000;

    let mut whole = bytes / BYTES_PER_MB;
    let mut fraction = (bytes % BYTES_PER_MB)
        .saturating_mul(FRACTION_SCALE)
        .saturating_add(BYTES_PER_MB / 2)
        / BYTES_PER_MB;
    if fraction == FRACTION_SCALE {
        whole = whole.saturating_add(1);
        fraction = 0;
    }
    format!("{whole}.{fraction:03} MB")
}

/// Sum the physical frames belonging to each room. This is deliberately a
/// disk-footprint figure: superseded frames remain charged to the room until
/// a repack reclaims them, while the shard header is shared and unallocated.
/// Reads only each frame's length and room ID, then discards its hash,
/// payload, and CRC through one sequential buffered scan; a room listing
/// never allocates payloads or checks CRCs.
fn room_disk_bytes(dir: &Path) -> anyhow::Result<HashMap<[u8; 16], u64>> {
    // 1-byte flags + 4-byte uncompressed_len + 16-byte room_id + 16-byte
    // hash — the fixed part of a v3 frame's payload, preceding the
    // (possibly zstd-compressed) node bytes.
    const FRAME_FIXED_LEN: u64 = 1 + 4 + 16 + 16;

    let mut by_room = HashMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .extension()
            .is_some_and(|extension| extension == "pack")
        {
            continue;
        }
        let file = fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        if mtxdb_core::packfile::read_header(&mut reader)?.is_none() {
            continue;
        }
        let mut discard = [0_u8; 8192];
        loop {
            let mut len = [0_u8; 4];
            match reader.read_exact(&mut len) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
            let frame_len = u64::from(u32::from_le_bytes(len));
            if !(FRAME_FIXED_LEN..=u64::from(mtxdb_core::packfile::MAX_RECORD_LEN))
                .contains(&frame_len)
            {
                bail!("invalid record length {frame_len} while measuring room disk usage");
            }
            // Skip the flags byte and uncompressed_len field to reach room_id
            // — this scan only needs the room ID, not whether/how the node
            // bytes that follow are compressed.
            let mut flags_and_uncompressed_len = [0_u8; 5];
            match reader.read_exact(&mut flags_and_uncompressed_len) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
            let mut room_id = [0_u8; 16];
            match reader.read_exact(&mut room_id) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
            let mut remaining = frame_len.saturating_sub(21).saturating_add(4);
            let mut complete = true;
            while remaining != 0 {
                let chunk_len = usize::try_from(remaining)
                    .unwrap_or(usize::MAX)
                    .min(discard.len());
                match reader.read_exact(&mut discard[..chunk_len]) {
                    Ok(()) => remaining = remaining.saturating_sub(chunk_len as u64),
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                        complete = false;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            // A concurrent append may expose a torn tail. If the final read
            // was short, leave it for the next listing once it is complete.
            if !complete {
                break;
            }
            let total = by_room.entry(room_id).or_insert(0_u64);
            *total = total.saturating_add(frame_len.saturating_add(8));
        }
    }
    Ok(by_room)
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

/// Ask the user to explicitly approve a destructive operation.
///
/// Returns `false` for every response except `y`/`Y`, so an empty line or
/// EOF is always safe by default.
fn confirm(prompt: &str) -> anyhow::Result<bool> {
    print!("{prompt} [y/N] ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().eq_ignore_ascii_case("y"))
}

pub(crate) fn run(cli: &Cli) -> anyhow::Result<()> {
    match &cli.command {
        Commands::Put { room, id, data } => cmd_put(cli, room, id, data),
        Commands::Get { room, id } => cmd_get(cli, room.as_deref(), id),
        Commands::Namespaces { all } => cmd_namespaces(cli, *all),
        Commands::Shards { all } => cmd_shards(cli, *all),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { shard } => cmd_scan(cli, shard),
        Commands::Import { paths, room } => cmd_import(cli, paths, room.as_deref()),
        Commands::Export { room } => cmd_export(cli, room),
        Commands::Repack {
            room,
            shards,
            all,
            root,
            topo,
        } => cmd_repack(cli, room.as_deref(), shards, *all, root, *topo),
        Commands::Delete { rooms, yes } => cmd_delete(cli, rooms, *yes),
        Commands::Completions { .. } => unreachable!("main emits completion scripts directly"),
        Commands::Sync => cmd_sync(cli),
    }
}

fn parse_room_id(hex: &str) -> anyhow::Result<[u8; 16]> {
    let hex = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    if hex.len() != 32 {
        bail!("room ID must be 32 hex characters, got {}", hex.len());
    }
    let bytes = hex::decode(hex).context("invalid hex in room ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

fn parse_node_id(hex: &str) -> anyhow::Result<[u8; 16]> {
    let hex = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    if hex.len() != 32 {
        bail!("node ID must be 32 hex characters, got {}", hex.len());
    }
    let bytes = hex::decode(hex).context("invalid hex in node ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

/// Resolve either the fixed-width storage key or a Matrix event ID. Imported
/// Matrix events use the first 128 bits of `BLAKE3(event_id)` as their key.
fn parse_get_id(id: &str) -> anyhow::Result<[u8; 16]> {
    if id.starts_with('$') {
        let hash = blake3::hash(id.as_bytes());
        let mut node_id = [0_u8; 16];
        node_id.copy_from_slice(&hash.as_bytes()[..16]);
        Ok(node_id)
    } else {
        parse_node_id(id)
    }
}

/// Open the store as its exclusive writer. Fails fast if another process
/// (e.g. a live embedder) already holds the writer lock — required for
/// any command that mutates data.
fn open_store(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    PackfileStorage::open(selected_pool_dir(cli)?).context("failed to open store")
}

/// Open the store read-only — coexists with a live writer process rather
/// than contending with it. For commands that only ever read room data.
fn open_store_read_only(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    PackfileStorage::open_read_only(selected_pool_dir(cli)?).context("failed to open store")
}

/// Resolve the selected independent shard pool below the database root.
///
/// The CLI deliberately never opens a root directory as a raw packfile pool:
/// doing so would recreate the flat layout and mix unrelated lifecycles.
fn open_layout(cli: &Cli) -> anyhow::Result<DatabaseLayout> {
    let root = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    DatabaseLayout::open(root.into())
        .with_context(|| format!("failed to open mtxdb database root `{}`", root.display()))
}

fn pool_dir(layout: &DatabaseLayout, shard_type: ShardType) -> anyhow::Result<PathBuf> {
    layout
        .pool_dir(shard_type)
        .with_context(|| format!("failed to open {} shard pool", shard_type.as_str()))
}

fn selected_pool_dir(cli: &Cli) -> anyhow::Result<PathBuf> {
    pool_dir(&open_layout(cli)?, cli.shard_type)
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

fn cmd_get(cli: &Cli, room: Option<&str>, id: &str) -> anyhow::Result<()> {
    let node_id = parse_get_id(id)?;
    let store = open_store_read_only(cli)?;
    let matches: Vec<([u8; 16], NodeData)> = match room {
        Some(room) => {
            let room_id = parse_room_id(room)?;
            store
                .get(&room_id, &node_id)?
                .map(|data| vec![(room_id, data)])
                .unwrap_or_default()
        }
        None => room_ids(cli)?
            .into_iter()
            .filter_map(|room_id| match store.get(&room_id, &node_id) {
                Ok(Some(data)) => Some(Ok((room_id, data))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<_, _>>()?,
    };
    match matches.as_slice() {
        [] => bail!("not found"),
        [(_, data)] => {
            io::stdout().write_all(&data.bytes)?;
            io::stdout().write_all(b"\n")?;
        }
        _ => bail!(
            "node ID {id} is present in multiple rooms ({}); specify --room",
            matches
                .iter()
                .map(|(room_id, _)| hex_encode(room_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
    Ok(())
}

/// Enumerate live rooms from the persisted directory. Stores created before
/// that sidecar existed fall back to a one-time index rebuild; `mtxdb sync`
/// makes future calls fast.
fn room_ids(cli: &Cli) -> anyhow::Result<Vec<[u8; 16]>> {
    let dir = selected_pool_dir(cli)?;
    if PackfileStorage::room_directory_persisted_at(&dir).is_some() {
        return Ok(PackfileStorage::room_directory_from_disk(&dir)
            .into_iter()
            .map(|(room_id, _)| room_id)
            .collect());
    }
    Ok(open_store_read_only(cli)?
        .room_summaries()
        .into_iter()
        .map(|(room_id, _, _)| room_id)
        .collect())
}

fn cmd_namespaces(cli: &Cli, all: bool) -> anyhow::Result<()> {
    if all {
        let layout = open_layout(cli)?;
        for (index, shard_type) in ShardType::ALL.into_iter().enumerate() {
            if index != 0 {
                println!();
            }
            println!("{}:", shard_type.as_str());
            cmd_namespaces_in_dir(&pool_dir(&layout, shard_type)?)?;
        }
        return Ok(());
    }
    cmd_namespaces_in_dir(&selected_pool_dir(cli)?)
}

/// List logical namespaces from one pool. Cross-pool aggregation is deliberately
/// avoided: each pool owns an independent 16-byte namespace and lifecycle.
fn cmd_namespaces_in_dir(dir: &Path) -> anyhow::Result<()> {
    // The persisted room directory is a fast listing snapshot, not proof
    // that the shard files are readable by this binary. Validate the small
    // immutable header of every shard before trusting it, so a pre-cutover
    // store does not misleadingly appear empty just because its sidecar is
    // empty or stale.
    validate_packfile_headers(dir)?;
    let rooms = match PackfileStorage::room_summaries_from_disk(dir) {
        Some(rooms) => rooms,
        // A named pool is created with the database layout, before it has
        // necessarily received a first write. `open_read_only` quite
        // properly rejects a directory with no shard files, but for a
        // listing that simply means there are no namespaces to show.
        None if glob_shard_files(dir)?.is_empty() => Vec::new(),
        // Old stores have no sidecar yet. Keep the complete, slower fallback
        // so `namespaces` remains useful until `mtxdb sync` writes one.
        None => PackfileStorage::open_read_only(dir.into())
            .context("failed to open store")?
            .room_summaries(),
    };
    let room_shards = PackfileStorage::room_shards_from_disk(dir);
    let disk_bytes = room_disk_bytes(dir)?;

    if rooms.is_empty() {
        println!("no namespaces found");
        return Ok(());
    }

    println!(
        "  {:>4}  {:<34}  {:>7}  {:>6}  {:>12}  {:>13}",
        "slot", "namespace", "nodes", "shards", "index RAM", "disk (MB)"
    );
    let mut total_nodes = 0_usize;
    let mut total_memory = 0_usize;
    let mut total_disk_bytes = 0_u64;
    for (i, (room_id, nodes, memory)) in rooms.iter().enumerate() {
        let hex = hex_encode(room_id);
        let shards = room_shards
            .as_ref()
            .and_then(|by_room| by_room.get(room_id))
            .map_or_else(
                || "?".to_owned(),
                |shards| {
                    if shards.len() == 1 {
                        String::new()
                    } else {
                        shards.len().to_string()
                    }
                },
            );
        total_nodes = total_nodes
            .checked_add(*nodes)
            .context("total room node count overflow")?;
        total_memory = total_memory
            .checked_add(*memory)
            .context("total room index memory overflow")?;
        let disk = disk_bytes.get(room_id).copied().unwrap_or(0);
        total_disk_bytes = total_disk_bytes.saturating_add(disk);
        println!(
            "  {i:>4}  0x{hex}  {nodes:>7}  {shards:>6}  {:>12}  {:>13}",
            fmt_megabytes(*memory),
            fmt_disk_megabytes(disk),
        );
    }
    println!();
    println!(
        "  {:>4}  {:<34}  {total_nodes:>7}  {:>6}  {:>12}  {:>13}",
        "",
        "total",
        "",
        fmt_megabytes(total_memory),
        fmt_disk_megabytes(total_disk_bytes),
    );
    Ok(())
}

/// Confirm that every packfile in `dir` uses the format this CLI can read.
/// This reads only the fixed 4 KiB shard descriptors; it never scans frames.
fn validate_packfile_headers(dir: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .extension()
            .is_some_and(|extension| extension == "pack")
        {
            continue;
        }
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open shard `{}`", path.display()))?;
        mtxdb_core::packfile::read_header(&mut BufReader::new(file))
            .with_context(|| format!("unsupported or corrupt shard `{}`", path.display()))?;
    }
    Ok(())
}

/// Deliberately bypasses both `PackfileStorage` and `ShardPool` — it
/// needs no room data and no packfile reads. Instead it globs shard
/// filenames for size/rotation, then decodes the small `shard_stats.bin` and
/// `shard_rooms.bin` sidecars for counters and live-node counts.
/// Safe to run against a directory a live writer process owns.
fn cmd_shards(cli: &Cli, all: bool) -> anyhow::Result<()> {
    if all {
        let layout = open_layout(cli)?;
        for (index, shard_type) in ShardType::ALL.into_iter().enumerate() {
            if index != 0 {
                println!();
            }
            println!("{}:", shard_type.as_str());
            cmd_shards_in_dir(&pool_dir(&layout, shard_type)?)?;
        }
        return Ok(());
    }
    cmd_shards_in_dir(&selected_pool_dir(cli)?)
}

/// List shard metadata from one pool only. This reads directory entries and
/// sidecars; it never scans packfile contents or opens room indexes.
fn cmd_shards_in_dir(dir: &Path) -> anyhow::Result<()> {
    let mut shard_entries = glob_shard_files(dir)?;
    shard_entries.sort_unstable_by_key(|&(id, _, _)| id);

    if shard_entries.is_empty() {
        println!("no shards found in `{}` (store is empty)", dir.display());
        return Ok(());
    }

    let (stats_map, persisted_at) = decode_stats_snapshot(dir);
    let node_counts = PackfileStorage::shard_node_counts_from_disk(dir);
    let room_counts = PackfileStorage::shard_room_counts_from_disk(dir);
    let total_rooms = room_counts
        .as_ref()
        .map(|_| PackfileStorage::room_directory_from_disk(dir).len());
    print_shard_table(
        &shard_entries,
        &stats_map,
        node_counts.as_ref(),
        room_counts.as_ref(),
        total_rooms,
    );
    println!("* active shard");
    println!(
        "{} shard(s), {}",
        shard_entries.len(),
        stats_snapshot_summary(persisted_at)
    );
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
fn print_shard_table(
    shard_entries: &[(u16, u64, u64)],
    stats_map: &ShardStatsMap,
    node_counts: Option<&std::collections::HashMap<u16, u64>>,
    room_counts: Option<&std::collections::HashMap<u16, u64>>,
    total_rooms: Option<usize>,
) {
    const EPOCH_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    // `ShardPool::open_internal` restores the highest occupied slot as its
    // append destination. Mirror that recovery rule here without opening a
    // writer just to render an inspection table.
    let active_slot = shard_entries.iter().map(|(slot, _, _)| *slot).max();

    println!(
        "{:>6}  {:>18}  {:>10}  {:>8}  {:>6}  {:>6}",
        "slot", "rotation", "bytes", "nodes", "rooms", "syncs",
    );
    let mut total_bytes = 0u64;
    let mut total_nodes = node_counts.map(|_| 0u64);
    let mut total_syncs = 0u64;
    for &(slot_id, epoch, file_bytes) in shard_entries {
        let key = u64::from(slot_id) << 48 | (epoch & EPOCH_MASK);
        let (_, _, sc) = stats_map.get(&key).copied().unwrap_or_default();
        let nodes = node_counts
            .and_then(|counts| counts.get(&slot_id))
            .map_or_else(|| "?".to_owned(), u64::to_string);
        let rooms = room_counts
            .and_then(|counts| counts.get(&slot_id))
            .map_or_else(|| "?".to_owned(), u64::to_string);
        println!(
            "{:>6}  {:>18}  {:>10}  {:>8}  {:>6}  {:>6}",
            format!(
                "{slot_id}{}",
                if active_slot == Some(slot_id) {
                    "*"
                } else {
                    " "
                }
            ),
            format!("{epoch:#018x}"),
            fmt_bytes(file_bytes),
            nodes,
            rooms,
            sc,
        );
        total_bytes = total_bytes.saturating_add(file_bytes);
        total_syncs = total_syncs.saturating_add(sc);
        if let (Some(total), Some(counts)) = (&mut total_nodes, node_counts) {
            *total = total.saturating_add(counts.get(&slot_id).copied().unwrap_or(0));
        }
    }
    println!();
    println!(
        "{:>6}  {:>18}  {:>10}  {:>8}  {:>6}  {:>6}",
        "",
        "total",
        fmt_bytes(total_bytes),
        total_nodes.map_or_else(|| "?".to_owned(), |count| count.to_string()),
        total_rooms.map_or_else(|| "?".to_owned(), |count| count.to_string()),
        total_syncs,
    );
}

/// Short summary of the persisted shard-statistics snapshot age.
fn stats_snapshot_summary(persisted_at: Option<u64>) -> String {
    match persisted_at {
        Some(ts) => {
            let age_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |now| now.as_secs().saturating_sub(ts));
            format!("snapshot: {} old", fmt_duration(age_secs))
        }
        None => "snapshot: none persisted".to_owned(),
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
    let room_id = match room.parse::<usize>() {
        Ok(slot) => {
            let rooms = room_ids(cli)?;
            rooms
                .get(slot)
                .copied()
                .with_context(|| format!("room slot {slot} not found"))?
        }
        Err(_) => parse_room_id(room)?,
    };
    let hex = hex_encode(&room_id);
    let dir = selected_pool_dir(cli)?;

    // The persisted directory is enough for the numerical summary and the
    // room's physical placement. Do not rebuild every room index merely to
    // answer `info` for one room.
    if let (Some(summaries), Some(room_shards)) = (
        PackfileStorage::room_summaries_from_disk(&dir),
        PackfileStorage::room_shards_from_disk(&dir),
    ) {
        if let Some((_, len, mem)) = summaries.into_iter().find(|(id, _, _)| *id == room_id) {
            println!("room {hex}: {len} nodes, {} index RAM", fmt_megabytes(mem));
            let shards = room_shards.get(&room_id).cloned().unwrap_or_default();
            print_room_shards(&shards);
            if let Some(details) = matrix_room_details_from_cache(&dir, &room_id) {
                print_matrix_room_details(details, None);
            } else {
                println!(
                    "  {:<12} scanning {} shard{}...",
                    "metadata:",
                    shards.len(),
                    if shards.len() == 1 { "" } else { "s" }
                );
                let details = matrix_room_details_from_shards(&dir, &room_id, &shards)
                    .unwrap_or_else(|error| {
                        eprintln!("warning: unable to inspect Matrix room metadata: {error}");
                        MatrixRoomDetails::default()
                    });
                if let Err(error) = persist_matrix_room_details(&dir, &room_id, &details) {
                    eprintln!("warning: unable to cache Matrix room metadata: {error}");
                }
                print_matrix_room_details(details, Some(shards.len()));
            }
            return Ok(());
        }
        eprintln!("room {hex}: not found");
        return Ok(());
    }

    // A store predating the inspection sidecar has no cheap authoritative
    // summary. Preserve the old full-scan fallback until `mtxdb sync` can
    // create the sidecar.
    let store = open_store_read_only(cli)?;
    match store.room_index_info(&room_id) {
        Some((len, mem)) => {
            println!("room {hex}: {len} nodes, {} index RAM", fmt_megabytes(mem));
            let shards = store.room_referenced_shards(&room_id);
            print_room_shards(&shards);
            print_matrix_room_details(matrix_room_details(&store, &dir, &room_id)?, None);
        }
        None => eprintln!("room {hex}: not found"),
    }
    Ok(())
}

#[derive(Clone, Default)]
struct MatrixRoomDetails {
    room_id: Option<String>,
    create: Option<String>,
}

const MATRIX_ROOM_DETAILS_MAGIC: &[u8; 4] = b"MMRM";
const MATRIX_ROOM_DETAILS_VERSION: u8 = 1;
const MATRIX_ROOM_DETAILS_HEADER_LEN: usize = 4 + 1;
const MATRIX_ROOM_DETAILS_RECORD_HEADER_LEN: usize = 16 + 2 + 2;

fn matrix_room_details_path(dir: &Path) -> PathBuf {
    dir.join("matrix_room_details.bin")
}

/// Load one cached Matrix description. This cache is only presentation
/// metadata; packfiles and the shard→room directory remain authoritative.
fn matrix_room_details_from_cache(dir: &Path, room_id: &[u8; 16]) -> Option<MatrixRoomDetails> {
    let buf = fs::read(matrix_room_details_path(dir)).ok()?;
    if buf.len() < MATRIX_ROOM_DETAILS_HEADER_LEN
        || &buf[0..4] != MATRIX_ROOM_DETAILS_MAGIC
        || buf[4] != MATRIX_ROOM_DETAILS_VERSION
    {
        return None;
    }
    let mut offset = MATRIX_ROOM_DETAILS_HEADER_LEN;
    while offset < buf.len() {
        let header_end = offset.checked_add(MATRIX_ROOM_DETAILS_RECORD_HEADER_LEN)?;
        let header = buf.get(offset..header_end)?;
        let matrix_len = usize::from(u16::from_le_bytes(header[16..18].try_into().ok()?));
        let create_len = usize::from(u16::from_le_bytes(header[18..20].try_into().ok()?));
        let data_len = matrix_len.checked_add(create_len)?;
        let data_end = header_end.checked_add(data_len)?;
        let data = buf.get(header_end..data_end)?;
        if &header[0..16] == room_id {
            let matrix_room_id = std::str::from_utf8(&data[..matrix_len])
                .ok()
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            let create = std::str::from_utf8(&data[matrix_len..])
                .ok()
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            return Some(MatrixRoomDetails {
                room_id: matrix_room_id,
                create,
            });
        }
        offset = data_end;
    }
    None
}

/// Atomically update the optional CLI metadata cache. It saves subsequent
/// `info` calls from decoding arbitrary event payloads in large shards.
fn persist_matrix_room_details(
    dir: &Path,
    room_id: &[u8; 16],
    details: &MatrixRoomDetails,
) -> io::Result<()> {
    if details.room_id.is_none() && details.create.is_none() {
        return Ok(());
    }
    let mut entries = read_all_matrix_room_details(dir);
    entries.insert(*room_id, details.clone());

    let mut buf = Vec::new();
    buf.extend_from_slice(MATRIX_ROOM_DETAILS_MAGIC);
    buf.push(MATRIX_ROOM_DETAILS_VERSION);
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_unstable_by_key(|(room_id, _)| *room_id);
    for (room_id, details) in entries {
        let matrix_room_id = details.room_id.unwrap_or_default();
        let create = details.create.unwrap_or_default();
        let matrix_len = u16::try_from(matrix_room_id.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Matrix room ID exceeds cache limit",
            )
        })?;
        let create_len = u16::try_from(create.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "create metadata exceeds cache limit",
            )
        })?;
        buf.extend_from_slice(&room_id);
        buf.extend_from_slice(&matrix_len.to_le_bytes());
        buf.extend_from_slice(&create_len.to_le_bytes());
        buf.extend_from_slice(matrix_room_id.as_bytes());
        buf.extend_from_slice(create.as_bytes());
    }
    let path = matrix_room_details_path(dir);
    let tmp_path = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    fs::write(&tmp_path, buf)?;
    fs::rename(tmp_path, path)
}

fn read_all_matrix_room_details(
    dir: &Path,
) -> std::collections::HashMap<[u8; 16], MatrixRoomDetails> {
    let mut entries = std::collections::HashMap::new();
    let Ok(buf) = fs::read(matrix_room_details_path(dir)) else {
        return entries;
    };
    if buf.len() < MATRIX_ROOM_DETAILS_HEADER_LEN
        || &buf[0..4] != MATRIX_ROOM_DETAILS_MAGIC
        || buf[4] != MATRIX_ROOM_DETAILS_VERSION
    {
        return entries;
    }
    let mut offset = MATRIX_ROOM_DETAILS_HEADER_LEN;
    while let Some(header_end) = offset.checked_add(MATRIX_ROOM_DETAILS_RECORD_HEADER_LEN) {
        let Some(header) = buf.get(offset..header_end) else {
            break;
        };
        let Ok(matrix_len) = <[u8; 2]>::try_from(&header[16..18]) else {
            break;
        };
        let Ok(create_len) = <[u8; 2]>::try_from(&header[18..20]) else {
            break;
        };
        let data_len = usize::from(u16::from_le_bytes(matrix_len))
            .checked_add(usize::from(u16::from_le_bytes(create_len)));
        let Some(data_end) = data_len.and_then(|length| header_end.checked_add(length)) else {
            break;
        };
        let Some(data) = buf.get(header_end..data_end) else {
            break;
        };
        let Ok(matrix_room_id) =
            std::str::from_utf8(&data[..usize::from(u16::from_le_bytes(matrix_len))])
        else {
            break;
        };
        let Ok(create) = std::str::from_utf8(&data[usize::from(u16::from_le_bytes(matrix_len))..])
        else {
            break;
        };
        let mut room_id = [0u8; 16];
        room_id.copy_from_slice(&header[0..16]);
        entries.insert(
            room_id,
            MatrixRoomDetails {
                room_id: (!matrix_room_id.is_empty()).then(|| matrix_room_id.to_owned()),
                create: (!create.is_empty()).then(|| create.to_owned()),
            },
        );
        offset = data_end;
    }
    entries
}

/// Find the human-facing Matrix metadata carried by live JSON events. Pack
/// scans can contain superseded frames, so every candidate is resolved through
/// the room's live index before being inspected.
fn matrix_room_details(
    store: &PackfileStorage,
    dir: &Path,
    room_id: &[u8; 16],
) -> anyhow::Result<MatrixRoomDetails> {
    let mut details = MatrixRoomDetails::default();
    let mut seen = std::collections::HashSet::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .extension()
            .is_some_and(|extension| extension == "pack")
        {
            continue;
        }
        for (record_room_id, node_id, _) in mtxdb_core::packfile::scan_packfile(&path)? {
            if &record_room_id != room_id || !seen.insert(node_id) {
                continue;
            }
            let Some(data) = store.get(room_id, &node_id)? else {
                continue;
            };
            let mut bytes = data.bytes.to_vec();
            let Ok(event) = simd_json::to_owned_value(&mut bytes) else {
                continue;
            };
            if details.room_id.is_none() {
                details.room_id = event_room_id(&event).map(str::to_owned);
            }
            if details.create.is_none() {
                details.create = matrix_create_details(&event);
            }
            if details.room_id.is_some() && details.create.is_some() {
                return Ok(details);
            }
        }
    }
    Ok(details)
}

/// Read Matrix metadata from only the shards containing `room_id`. This is
/// deliberately streaming: common Matrix room IDs and create events occur
/// near the beginning of topology-ordered data, so `info` can stop as soon
/// as both fields are found rather than scanning unrelated shards or loading
/// a full store index.
fn matrix_room_details_from_shards(
    dir: &Path,
    room_id: &[u8; 16],
    shard_ids: &[u16],
) -> anyhow::Result<MatrixRoomDetails> {
    let pool = ShardPool::open_read_only(dir.into()).context("failed to open shard store")?;
    let mut details = MatrixRoomDetails::default();
    for shard_id in shard_ids {
        let Some(shard) = pool.get_shard(*shard_id) else {
            continue;
        };
        let file = fs::File::open(&shard.path)?;
        let mut reader = BufReader::new(file);
        if mtxdb_core::packfile::read_header(&mut reader)?.is_none() {
            continue;
        }
        loop {
            let record = match mtxdb_core::packfile::read_record(&mut reader) {
                Ok(Some(record)) => record,
                Ok(None) => break,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            };
            if &record.room_id != room_id {
                continue;
            }
            update_matrix_room_details(&mut details, &record.data);
            if details.room_id.is_some() && details.create.is_some() {
                return Ok(details);
            }
        }
    }
    Ok(details)
}

fn print_matrix_room_details(details: MatrixRoomDetails, scanned_shards: Option<usize>) {
    if let Some(matrix_room_id) = details.room_id {
        println!("  {:<12} {matrix_room_id}", "Matrix room:");
    }
    if let Some(create) = details.create {
        println!("  {:<12} {create}", "create:");
    } else if let Some(shard_count) = scanned_shards {
        println!(
            "  {:<12} not found (scanned all {shard_count} shard{})",
            "create:",
            if shard_count == 1 { "" } else { "s" }
        );
    } else {
        println!("  {:<12} not found", "create:");
    }
}

fn print_room_shards(shards: &[u16]) {
    match shards {
        [shard] => println!("  {:<12} {shard}", "shard:"),
        [] => {}
        _ => println!(
            "  {:<12} {}",
            "shards:",
            shards
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn update_matrix_room_details(details: &mut MatrixRoomDetails, data: &[u8]) {
    let mut bytes = data.to_vec();
    let Ok(event) = simd_json::to_owned_value(&mut bytes) else {
        return;
    };
    if details.room_id.is_none() {
        details.room_id = event_room_id(&event).map(str::to_owned);
    }
    if details.create.is_none() {
        details.create = matrix_create_details(&event);
    }
}

fn cmd_scan(cli: &Cli, selector: &str) -> anyhow::Result<()> {
    let pool_dir = selected_pool_dir(cli)?;
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let shard = if let Some(rotation) = selector
        .strip_prefix("0x")
        .or_else(|| selector.strip_prefix("0X"))
    {
        let rotation =
            u64::from_str_radix(rotation, 16).context("invalid hexadecimal shard rotation")?;
        pool.all_shards()
            .into_iter()
            .find_map(|(_, shard)| (shard.epoch == rotation).then_some(shard))
            .with_context(|| format!("shard rotation {selector} not found"))?
    } else {
        let slot = selector
            .parse::<u16>()
            .context("shard slot must be a decimal u16 or a 0x-prefixed rotation")?;
        pool.get_shard(slot)
            .with_context(|| format!("shard slot {slot} not found"))?
    };
    let path = &shard.path;
    let records = mtxdb_core::packfile::scan_packfile(path)?;
    println!(
        "shard: {} bytes, {} writes",
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

fn cmd_import(
    cli: &Cli,
    paths: &[std::path::PathBuf],
    room_override: Option<&str>,
) -> anyhow::Result<()> {
    // One writer owns the entire batch: it prevents another writer from
    // interleaving halfway through a shell glob, and lets us publish one
    // complete shard/room snapshot once the final input has been handled.
    let store = open_store(cli)?;
    let mut failures = 0_usize;
    for (index, path) in paths.iter().enumerate() {
        if index != 0 {
            eprintln!();
        }
        if let Err(error) = cmd_import_file(&store, path, room_override) {
            eprintln!("{}: {error:#}", path.display());
            failures = failures.saturating_add(1);
        }
    }
    store
        .sync_all()
        .context("persisting import shard summaries")?;
    if failures != 0 {
        bail!("import completed with {failures} failed input file(s)");
    }
    Ok(())
}

/// Export every live record in one room as a JSONL stream. A read-only frame
/// scan avoids rebuilding every room's in-memory index just to enumerate one
/// room. Scanning shards in rotation order lets a later physical copy replace
/// an older one with the same node ID.
fn cmd_export(cli: &Cli, room: &str) -> anyhow::Result<()> {
    let room_id = parse_room_id(room)?;
    let pool_dir = selected_pool_dir(cli)?;
    let Some(summaries) = PackfileStorage::room_summaries_from_disk(&pool_dir) else {
        bail!("room directory is unavailable; run `mtxdb sync` before exporting");
    };
    if !summaries.iter().any(|(id, _, _)| *id == room_id) {
        bail!("room {room} not found");
    }
    let room_shards = PackfileStorage::room_shards_from_disk(&pool_dir)
        .and_then(|by_room| by_room.get(&room_id).cloned())
        .context("room shard directory is unavailable; run `mtxdb sync` before exporting")?;
    let room_shards: HashSet<u16> = room_shards.into_iter().collect();

    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let mut shards: Vec<_> = pool
        .all_shards()
        .into_iter()
        .filter(|(shard_id, _)| room_shards.contains(shard_id))
        .collect();
    shards.sort_unstable_by_key(|(_, shard)| shard.epoch);

    let mut locations = HashMap::new();
    let mut ordered_ids = Vec::new();
    for (shard_index, (_, shard)) in shards.iter().enumerate() {
        for (candidate_room, node_id, offset) in mtxdb_core::packfile::scan_packfile(&shard.path)? {
            if candidate_room == room_id {
                if !locations.contains_key(&node_id) {
                    ordered_ids.push(node_id);
                }
                locations.insert(node_id, (shard_index, offset));
            }
        }
    }

    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let mut exported = 0_usize;
    for node_id in ordered_ids {
        let (shard_index, offset) = locations
            .get(&node_id)
            .copied()
            .context("export record location disappeared during scan")?;
        let (_, shard) = &shards[shard_index];
        let record = ShardPool::read_at(shard, offset)?;
        output.write_all(&record.data)?;
        output.write_all(b"\n")?;
        exported = exported.saturating_add(1);
    }
    output.flush()?;
    eprintln!(
        "exported {exported} records from room {}",
        hex_encode(&room_id)
    );
    Ok(())
}

fn cmd_import_file(
    store: &PackfileStorage,
    path: &Path,
    room_override: Option<&str>,
) -> anyhow::Result<()> {
    let content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let is_jsonl = path
        .extension()
        .is_some_and(|extension| extension == "jsonl");
    let (events, detected_room) = if is_jsonl {
        match parse_jsonl_events(&content) {
            Ok(events) => events,
            Err(jsonl_error) => match parse_federation_events(&content) {
                // Some existing DAG exports carry a `.jsonl` suffix despite
                // being a pretty-printed federation JSON document.
                Ok(events) => events,
                Err(_) => return Err(jsonl_error),
            },
        }
    } else {
        parse_federation_events(&content)?
    };

    if events.is_empty() {
        bail!("no events found in {}", path.display());
    }
    let first_event_ids: Vec<&str> = events.iter().filter_map(event_id).take(3).collect();

    let mut event_count = 0u64;
    let mut skipped = 0u64;
    let mut already_present = 0u64;

    let room_id = if let Some(r) = room_override {
        parse_room_id(r)?
    } else {
        let rid = detected_room.with_context(|| {
            let first_event = events
                .first()
                .and_then(event_id)
                .unwrap_or("<missing event_id>");
            format!(
                "could not detect room_id (first event: {first_event}); pass --room for this input"
            )
        })?;
        let hash = blake3::hash(rid.as_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        id
    };

    let room_hex = hex_encode(&room_id);

    for ev in &events {
        let Some(incoming_event_id) = event_id(ev) else {
            skipped = skipped.saturating_add(1);
            continue;
        };
        let event_hash = blake3::hash(incoming_event_id.as_bytes());
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&event_hash.as_bytes()[..16]);

        let event_bytes = ev.encode().into_bytes();
        if let Some(existing) = store.get(&room_id, &id_bytes)? {
            let mut existing_bytes = existing.bytes.to_vec();
            let existing_event_id = simd_json::to_owned_value(&mut existing_bytes)
                .ok()
                .and_then(|event| event_id(&event).map(str::to_owned));
            if existing_event_id.as_deref() != Some(incoming_event_id) {
                bail!(
                    "node ID {} collides with a different event_id; refusing to overwrite it",
                    hex_encode(&id_bytes)
                );
            }
            already_present = already_present.saturating_add(1);
            continue;
        }
        let data = NodeData::new(bytes::Bytes::from(event_bytes));
        store.put(&room_id, &id_bytes, &data)?;
        event_count = event_count.saturating_add(1);
    }

    eprintln!("imported {event_count} events to room {room_hex}");
    if !first_event_ids.is_empty() {
        eprintln!(
            "first {} event{}: {}",
            first_event_ids.len(),
            if first_event_ids.len() == 1 { "" } else { "s" },
            first_event_ids.join(", ")
        );
    }
    if skipped > 0 {
        eprintln!("skipped {skipped} events (missing event_id)");
    }
    if already_present > 0 {
        eprintln!("{already_present} events already present");
    }

    Ok(())
}

fn parse_jsonl_events(content: &[u8]) -> anyhow::Result<(Vec<OwnedValue>, Option<String>)> {
    let text = std::str::from_utf8(content).context("input is not valid UTF-8 JSONL")?;
    let mut events = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line_number = line_number
            .checked_add(1)
            .context("JSONL line number overflow")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut bytes = line.as_bytes().to_vec();
        let event = simd_json::to_owned_value(&mut bytes)
            .with_context(|| format!("invalid JSONL event on line {line_number}"))?;
        events.push(event);
    }
    let detected_room = events.iter().find_map(event_room_id).map(str::to_owned);
    Ok((events, detected_room))
}

fn parse_federation_events(content: &[u8]) -> anyhow::Result<(Vec<OwnedValue>, Option<String>)> {
    let mut bytes = content.to_vec();
    let val: OwnedValue = simd_json::to_owned_value(&mut bytes).context("invalid JSON")?;
    let detected_room = event_room_id(&val).map(str::to_owned);
    let pdus = val["pdus"].as_array();
    let auth_chain = val["auth_chain"].as_array();
    if pdus.is_none() && auth_chain.is_none() {
        bail!("expected Matrix federation JSON with a `pdus` or `auth_chain` array, or a .jsonl file containing one event per line");
    }
    let events = [pdus, auth_chain]
        .into_iter()
        .flatten()
        .flat_map(|events| events.iter().cloned())
        .collect();
    Ok((events, detected_room))
}

fn event_room_id(value: &OwnedValue) -> Option<&str> {
    match value {
        OwnedValue::Object(object) => match object.get("room_id") {
            Some(OwnedValue::String(room_id)) => Some(room_id.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn event_id(value: &OwnedValue) -> Option<&str> {
    match value {
        OwnedValue::Object(object) => match object.get("event_id") {
            Some(OwnedValue::String(event_id)) => Some(event_id.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn event_string_field<'a>(value: &'a OwnedValue, field: &str) -> Option<&'a str> {
    match value {
        OwnedValue::Object(object) => match object.get(field) {
            Some(OwnedValue::String(value)) => Some(value.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn nested_event_string_field<'a>(
    value: &'a OwnedValue,
    object: &str,
    field: &str,
) -> Option<&'a str> {
    match value {
        OwnedValue::Object(fields) => match fields.get(object) {
            Some(OwnedValue::Object(nested)) => match nested.get(field) {
                Some(OwnedValue::String(value)) => Some(value.as_str()),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn matrix_create_details(event: &OwnedValue) -> Option<String> {
    if event_string_field(event, "type") != Some("m.room.create") {
        return None;
    }
    let event_id = event_id(event).unwrap_or("<missing event_id>");
    let sender = event_string_field(event, "sender").unwrap_or("<missing sender>");
    let creator = nested_event_string_field(event, "content", "creator");
    let version = nested_event_string_field(event, "content", "room_version");
    let mut create = format!("{event_id} by {sender}");
    if let Some(creator) = creator {
        let _ = write!(create, "; creator {creator}");
    }
    if let Some(version) = version {
        let _ = write!(create, "; room version {version}");
    }
    Some(create)
}

fn cmd_repack(
    cli: &Cli,
    room: Option<&str>,
    shards: &[String],
    all: bool,
    roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    let target = match (room, shards.is_empty(), all) {
        (Some(room), true, false) => RepackTarget::Room(parse_room_id(room)?),
        (None, false, false) => RepackTarget::Shards(parse_shard_selectors(shards)?),
        (None, true, true) => RepackTarget::All,
        // Clap rejects the both-targets case through `conflicts_with`; this
        // branch gives the missing-target case a readable diagnostic.
        _ => {
            return Err(anyhow!(
                "exactly one of --room <room> | --shard <shard> | --all is required"
            ))
        }
    };

    if !roots.is_empty() {
        match &target {
            RepackTarget::Shards(_) | RepackTarget::All => {
                bail!(
                    "--root requires --room — live roots are per-room, not meaningful for --shard"
                );
            }
            RepackTarget::Room(_) if !topo => {
                bail!("--root requires --topo; without --topo there are no edges so only the specified roots would be kept");
            }
            RepackTarget::Room(_) => {
                bail!(
                    "--root cannot be used with Matrix topology yet: imported event IDs cannot be resolved to stored node IDs; refusing a rooted repack that could discard ancestors"
                );
            }
        }
    }
    if topo {
        println!("warning: edge extraction is approximate; prev_events event IDs are not resolved to stored node hashes");
    } else {
        println!("warning: --topo without --root means no GC; all records preserved");
    }
    cmd_repack_target(cli, &target, topo)
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
struct RepackPreview {
    rooms: Vec<[u8; 16]>,
    shards: Vec<u16>,
}

enum RepackTarget {
    Room([u8; 16]),
    Shards(Vec<u16>),
    All,
}

/// Parse decimal shard slots and inclusive `START-END` ranges, returning an
/// ordered deduplicated set so overlapping selections are repacked once.
fn parse_shard_selectors(selectors: &[String]) -> anyhow::Result<Vec<u16>> {
    let mut slots = std::collections::BTreeSet::new();
    for selector in selectors {
        let (start, end) = if let Some((start, end)) = selector.split_once('-') {
            if start.is_empty() || end.is_empty() || end.contains('-') {
                bail!("invalid shard range `{selector}`; use decimal START-END");
            }
            let start = start
                .parse::<u16>()
                .with_context(|| format!("invalid shard range start `{start}`"))?;
            let end = end
                .parse::<u16>()
                .with_context(|| format!("invalid shard range end `{end}`"))?;
            if start > end {
                bail!("invalid shard range `{selector}`; start exceeds end");
            }
            (start, end)
        } else {
            let slot = selector.parse::<u16>().with_context(|| {
                format!("invalid shard slot `{selector}`; use decimal slots or START-END ranges")
            })?;
            (slot, slot)
        };
        slots.extend(start..=end);
    }
    Ok(slots.into_iter().collect())
}

fn resolve_repack_target(
    store: &PackfileStorage,
    target: &RepackTarget,
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    match target {
        RepackTarget::Room(room_id) => {
            if store.room_index_info(room_id).is_none() {
                bail!("room {} not found", hex_encode(room_id));
            }
            Ok((vec![*room_id], store.room_referenced_shards(room_id)))
        }
        RepackTarget::Shards(shard_ids) => resolve_repack_shards(store, shard_ids),
        RepackTarget::All => {
            let shard_ids = store
                .shard_summaries()
                .into_iter()
                .map(|summary| summary.shard_id)
                .collect::<Vec<_>>();
            resolve_repack_shards(store, &shard_ids)
        }
    }
}

fn resolve_repack_shards(
    store: &PackfileStorage,
    shard_ids: &[u16],
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    let mut rooms = std::collections::BTreeSet::new();
    let mut shards = std::collections::BTreeSet::new();
    for shard_id in shard_ids {
        let (closure_rooms, closure_shards) = store.repack_closure(*shard_id)?;
        rooms.extend(closure_rooms);
        shards.extend(closure_shards);
    }
    Ok((rooms.into_iter().collect(), shards.into_iter().collect()))
}

fn repack_preview(
    cli: &Cli,
    target: &RepackTarget,
    topo: bool,
) -> anyhow::Result<Option<RepackPreview>> {
    println!("preflight: scanning shards and rebuilding live indexes...");
    let preview_store = open_store_read_only(cli)?;
    println!("preflight: resolving shard closure...");
    let (rooms, shards) = resolve_repack_target(&preview_store, target)?;
    if rooms.is_empty() {
        match target {
            RepackTarget::Shards(shard_ids) => {
                let slots = shard_ids
                    .iter()
                    .map(u16::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("no rooms reference selected shards: {slots}");
            }
            RepackTarget::All => println!("no rooms found in active shards"),
            RepackTarget::Room(_) => {}
        }
        return Ok(None);
    }

    let mut total_kept = 0usize;
    let mut total_dropped = 0usize;
    let mut total_dropped_bytes = 0u64;
    println!(
        "preflight: scanning {} shard{} for {} room{}...",
        shards.len(),
        if shards.len() == 1 { "" } else { "s" },
        rooms.len(),
        if rooms.len() == 1 { "" } else { "s" },
    );
    let plans = if topo {
        preview_store.plan_rooms_repack(&rooms, extract_matrix_edges)?
    } else {
        preview_store.plan_rooms_repack(&rooms, |_hash, _data| Vec::new())?
    };
    for plan in plans {
        total_kept = total_kept.saturating_add(plan.kept);
        total_dropped = total_dropped.saturating_add(plan.dropped);
        total_dropped_bytes = total_dropped_bytes.saturating_add(plan.dropped_bytes);
    }

    let shard_slots = shards
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let shard_sizes: std::collections::HashMap<u16, u64> = preview_store
        .shard_summaries()
        .into_iter()
        .map(|summary| (summary.shard_id, summary.file_bytes))
        .collect();
    let total_input_bytes = shards.iter().fold(0u64, |total, shard_id| {
        total.saturating_add(shard_sizes.get(shard_id).copied().unwrap_or(0))
    });

    println!(
        "this will repack {} room{} across {} shard{} (slots: {shard_slots})",
        rooms.len(),
        if rooms.len() == 1 { "" } else { "s" },
        shards.len(),
        if shards.len() == 1 { "" } else { "s" },
    );
    let widest_slot = shards
        .iter()
        .map(u16::to_string)
        .map(|slot| slot.len())
        .max()
        .unwrap_or(1);
    let label_width = "slot ".len().saturating_add(widest_slot);
    for shard_id in &shards {
        let bytes = shard_sizes.get(shard_id).copied().unwrap_or(0);
        println!("  slot {shard_id:>widest_slot$}: {:>9}", fmt_bytes(bytes));
    }
    println!(
        "  {:>label_width$}: {:>9}",
        "total",
        fmt_bytes(total_input_bytes)
    );
    println!(
        "expected result: {total_kept} nodes across {} room{} rewritten",
        rooms.len(),
        if rooms.len() == 1 { "" } else { "s" }
    );
    println!(
        "                 {total_dropped} nodes / {} pruned",
        fmt_bytes(total_dropped_bytes),
    );
    // preview_store (and its non-exclusive read-only handle) drops
    // here, before the confirmation prompt — nothing about a
    // read-only open blocks a real writer anyway, but there's no
    // reason to keep it open through an indefinite human pause.
    Ok(Some(RepackPreview { rooms, shards }))
}

fn repack_rooms(
    store: &PackfileStorage,
    rooms: &[[u8; 16]],
    topo: bool,
) -> anyhow::Result<(usize, usize)> {
    let results = if topo {
        store.repack_rooms_reachable_with_progress(
            rooms,
            extract_matrix_edges,
            |room_id, from, to, nodes, complete| {
                if complete {
                    println!("  copy complete: {nodes} nodes; output slot {to} active");
                } else {
                    let room_id = room_id.expect("rotation progress has a room");
                    println!(
                        "  copy progress: {nodes} nodes; output rotated {from} → {to} while copying room {}",
                        hex_encode(&room_id),
                    );
                }
            },
        )?
    } else {
        store.repack_rooms_reachable_with_progress(
            rooms,
            |_hash, _data| Vec::new(),
            |room_id, from, to, nodes, complete| {
                if complete {
                    println!("  copy complete: {nodes} nodes; output slot {to} active");
                } else {
                    let room_id = room_id.expect("rotation progress has a room");
                    println!(
                        "  copy progress: {nodes} nodes; output rotated {from} → {to} while copying room {}",
                        hex_encode(&room_id),
                    );
                }
            },
        )?
    };
    let (final_kept, final_dropped) = results
        .iter()
        .fold((0usize, 0usize), |(k, d), &(_, kept, dropped)| {
            (k.saturating_add(kept), d.saturating_add(dropped))
        });
    println!(
        "done: {} rooms repacked in one shard batch, {final_kept} kept, {final_dropped} dropped",
        results.len()
    );
    Ok((final_kept, final_dropped))
}

fn cmd_repack_target(cli: &Cli, target: &RepackTarget, topo: bool) -> anyhow::Result<()> {
    let Some(preview) = repack_preview(cli, target, topo)? else {
        return Ok(());
    };

    if !confirm("Apply this repack?")? {
        println!("aborted");
        return Ok(());
    }

    // Only now do we take the exclusive writer lock — and the very first
    // thing done under it is recomputing the closure fresh, since the
    // read-only preview above is, by construction, a snapshot that could
    // be arbitrarily stale by the time a human finishes reading it.
    let store = open_store(cli)?;
    let (rooms, touched_shards) = resolve_repack_target(&store, target)?;
    if rooms.is_empty() {
        println!("repack target is no longer referenced by any room — nothing to do");
        return Ok(());
    }
    let grew = rooms.iter().any(|r| !preview.rooms.contains(r))
        || touched_shards.iter().any(|s| !preview.shards.contains(s));
    if grew {
        println!(
            "note: the closure grew since the preview (now {} rooms / {} shards) — repacking the current, authoritative closure",
            rooms.len(),
            touched_shards.len()
        );
    }

    repack_rooms(&store, &rooms, topo)?;
    // `cmd_shards` deliberately reads the persisted shard→room directory
    // instead of reopening and scanning every packfile. Refresh it before
    // the immediate post-repack table so freshly-created rotations have
    // real node counts rather than `?`.
    store.persist_shard_rooms()?;
    drop(store);
    println!("post-repack shard state:");
    cmd_shards(cli, false)?;
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

fn cmd_delete(cli: &Cli, rooms: &[String], yes: bool) -> anyhow::Result<()> {
    let mut room_ids: Vec<[u8; 16]> = rooms
        .iter()
        .map(|room| parse_room_id(room))
        .collect::<anyhow::Result<_>>()?;
    room_ids.sort_unstable();
    room_ids.dedup();

    if !yes {
        println!(
            "This will permanently delete all data for {} room(s):",
            room_ids.len()
        );
        for room_id in &room_ids {
            println!("  {}", hex_encode(room_id));
        }
        if !confirm("Delete these rooms?")? {
            println!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    for room_id in room_ids {
        let count = store.room_index_info(&room_id).map_or(0, |(len, _)| len);
        store.delete_room(&room_id)?;
        println!("deleted {count} nodes for room {}", hex_encode(&room_id));
    }
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

#[cfg(test)]
mod tests {
    use super::{fmt_disk_megabytes, fmt_megabytes};

    #[test]
    fn index_memory_megabytes_uses_fixed_point_rounding() {
        assert_eq!(fmt_megabytes(524_328), "0.52433 MB");
        assert_eq!(fmt_megabytes(999_999), "1.00000 MB");
    }

    #[test]
    fn disk_megabytes_uses_three_fractional_digits() {
        assert_eq!(fmt_disk_megabytes(42_280), "0.042 MB");
        assert_eq!(fmt_disk_megabytes(999_999), "1.000 MB");
    }
}
