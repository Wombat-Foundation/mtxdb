use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use mtxdb_core::shard::ShardPool;
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

/// Decimal megabytes, rounded exactly to four fractional digits, for the
/// human-facing `info` output.
fn fmt_megabytes(bytes: usize) -> String {
    const BYTES_PER_MB: u64 = 1_000_000;
    const FRACTION_SCALE: u64 = 10_000;

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
    format!("{whole}.{fraction:04} MB")
}

/// Percentage of `part` over `total`, rounded exactly to two decimals.
fn fmt_percent(part: u64, total: u64) -> String {
    if total == 0 {
        return "0.00%".to_owned();
    }
    let hundredths = part
        .saturating_mul(10_000)
        .saturating_add(total / 2)
        .checked_div(total)
        .unwrap_or(u64::MAX);
    format!("{}.{:02}%", hundredths / 100, hundredths % 100)
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
        Commands::Get { room, id } => cmd_get(cli, room, id),
        Commands::Rooms => cmd_rooms(cli),
        Commands::Shards => cmd_shards(cli),
        Commands::Info { room } => cmd_info(cli, room),
        Commands::Scan { shard } => cmd_scan(cli, shard),
        Commands::Import { paths, room } => cmd_import(cli, paths, room.as_deref()),
        Commands::Repack {
            room,
            shard,
            root,
            topo,
        } => cmd_repack(cli, room.as_deref(), *shard, root, *topo),
        Commands::Delete { room, yes } => cmd_delete(cli, room, *yes),
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

/// Open the store as its exclusive writer. Fails fast if another process
/// (e.g. a live embedder) already holds the writer lock — required for
/// any command that mutates data.
fn open_store(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    require_store_dir(dir)?;
    PackfileStorage::open(dir.into()).context("failed to open store")
}

/// Open the store read-only — coexists with a live writer process rather
/// than contending with it. For commands that only ever read room data.
fn open_store_read_only(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    require_store_dir(dir)?;
    PackfileStorage::open_read_only(dir.into()).context("failed to open store")
}

/// Ensure the selected store location exists before issuing an operation.
/// This turns the otherwise opaque `read_dir`/`open` ENOENT into an action
/// the person running the CLI can take.
fn require_store_dir(dir: &Path) -> anyhow::Result<()> {
    match fs::metadata(dir) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => bail!("storage path `{}` is not a directory", dir.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => bail!(
            "storage directory `{}` does not exist; create it with `mkdir -p {}` or choose one with --dir DIR",
            dir.display(),
            dir.display(),
        ),
        Err(error) => Err(error).with_context(|| {
            format!("cannot access storage directory `{}`", dir.display())
        }),
    }
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

/// Scan every shard's packfile to count append entries per room — the I/O
/// (reading record headers) is unavoidable since room ownership only
/// exists inside the packfile, but this skips building a `LossyIndex`
/// and `NodeCache` per room, which `open_store_read_only` would do
/// purely to hand back counts and then throw everything away.
fn room_frame_counts(dir: &Path) -> anyhow::Result<Vec<([u8; 16], usize)>> {
    require_store_dir(dir)?;
    let mut counts: std::collections::HashMap<[u8; 16], usize> = std::collections::HashMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().is_some_and(|e| e == "pack") {
            continue;
        }
        let records = mtxdb_core::packfile::scan_packfile(&path)?;
        for (room_id, _hash, _offset) in &records {
            counts
                .entry(*room_id)
                .and_modify(|count| *count = count.saturating_add(1))
                .or_insert(1);
        }
    }

    let mut rooms: Vec<([u8; 16], usize)> = counts.into_iter().collect();
    rooms.sort_unstable_by_key(|&(id, _)| id);
    Ok(rooms)
}

fn cmd_rooms(cli: &Cli) -> anyhow::Result<()> {
    let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let rooms = room_frame_counts(dir)?;

    if rooms.is_empty() {
        eprintln!("no rooms found");
        return Ok(());
    }

    let store = open_store_read_only(cli)?;
    println!(
        "  {:>4}  {:<34}  {:>7}  {:>10}  {:>7}",
        "slot", "room", "nodes", "index RAM", "writes"
    );
    for (i, (room_id, count)) in rooms.iter().enumerate() {
        let hex = hex_encode(room_id);
        let (nodes, memory) = store.room_index_info(room_id).unwrap_or((0, 0));
        println!(
            "  {i:>4}  0x{hex}  {nodes:>7}  {:>10}  {count:>7}",
            fmt_megabytes(memory),
        );
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
        println!("no shards found in `{}` (store is empty)", dir.display());
        return Ok(());
    }

    let (stats_map, persisted_at) = decode_stats_snapshot(dir);
    let node_counts = open_store_read_only(cli)?.shard_node_counts();
    print_shard_table(&shard_entries, &stats_map, &node_counts);
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
    require_store_dir(dir)?;
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
    node_counts: &std::collections::HashMap<u16, u64>,
) {
    const EPOCH_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    println!(
        "{:>6}  {:>18}  {:>10}  {:>8}  {:>8}  {:>10}  {:>6}",
        "slot", "epoch", "bytes", "nodes", "writes", "written", "syncs"
    );
    for &(slot_id, epoch, file_bytes) in shard_entries {
        let key = u64::from(slot_id) << 48 | (epoch & EPOCH_MASK);
        let (wc, bw, sc) = stats_map.get(&key).copied().unwrap_or_default();
        println!(
            "{:>6}  {:>18}  {:>10}  {:>8}  {:>8}  {:>10}  {:>6}",
            slot_id,
            format!("{epoch:#018x}"),
            fmt_bytes(file_bytes),
            node_counts.get(&slot_id).copied().unwrap_or(0),
            wc,
            fmt_bytes(bw),
            sc,
        );
    }
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
            let dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
            let rooms = room_frame_counts(dir)?;
            rooms
                .get(slot)
                .map(|(room_id, _)| *room_id)
                .with_context(|| format!("room slot {slot} not found"))?
        }
        Err(_) => parse_room_id(room)?,
    };
    let store = open_store_read_only(cli)?;
    let hex = hex_encode(&room_id);
    match store.room_index_info(&room_id) {
        Some((len, mem)) => {
            println!("room {hex}: {len} nodes, {} index RAM", fmt_megabytes(mem));
        }
        None => {
            eprintln!("room {hex}: not found");
        }
    }
    Ok(())
}

fn cmd_scan(cli: &Cli, selector: &str) -> anyhow::Result<()> {
    let base_dir = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let pool = ShardPool::open_read_only(base_dir.into()).context("failed to open shard store")?;
    let shard = if let Some(epoch) = selector
        .strip_prefix("0x")
        .or_else(|| selector.strip_prefix("0X"))
    {
        let epoch = u64::from_str_radix(epoch, 16).context("invalid hexadecimal shard epoch")?;
        pool.all_shards()
            .into_iter()
            .find_map(|(_, shard)| (shard.epoch == epoch).then_some(shard))
            .with_context(|| format!("shard epoch {selector} not found"))?
    } else {
        let slot = selector
            .parse::<u16>()
            .context("shard slot must be a decimal u16 or a 0x-prefixed epoch")?;
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
    for path in paths {
        cmd_import_file(cli, path, room_override)?;
    }
    Ok(())
}

fn cmd_import_file(cli: &Cli, path: &Path, room_override: Option<&str>) -> anyhow::Result<()> {
    let content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let is_jsonl = path
        .extension()
        .is_some_and(|extension| extension == "jsonl");
    let (events, detected_room) = if is_jsonl {
        let text = std::str::from_utf8(&content)
            .with_context(|| format!("{} is not valid UTF-8 JSONL", path.display()))?;
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
        (events, detected_room)
    } else {
        let mut content = content;
        let val: OwnedValue = simd_json::to_owned_value(&mut content).context("invalid JSON")?;
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
        (events, detected_room)
    };

    if events.is_empty() {
        bail!("no events found in {}", path.display());
    }

    let mut event_count = 0u64;
    let mut skipped = 0u64;
    let mut already_present = 0u64;

    let store = open_store(cli)?;

    let room_id = if let Some(r) = room_override {
        parse_room_id(r)?
    } else {
        let rid = detected_room.context("could not detect room_id; pass --room for this input")?;
        let hash = blake3::hash(rid.as_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        id
    };

    let room_hex = hex_encode(&room_id);

    for ev in &events {
        let Some(id_bytes) = event_node_id(ev) else {
            skipped = skipped.saturating_add(1);
            continue;
        };

        let event_bytes = ev.to_string().into_bytes();
        if let Some(existing) = store.get(&room_id, &id_bytes)? {
            if existing.bytes.as_ref() != event_bytes.as_slice() {
                bail!(
                    "node ID {} already exists with different payload; refusing to overwrite it",
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
    if skipped > 0 {
        eprintln!("skipped {skipped} events (missing sha256 hash)");
    }
    if already_present > 0 {
        eprintln!("{already_present} events already present");
    }

    Ok(())
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

fn event_node_id(value: &OwnedValue) -> Option<[u8; 16]> {
    let OwnedValue::Object(event) = value else {
        return None;
    };
    let Some(OwnedValue::Object(hashes)) = event.get("hashes") else {
        return None;
    };
    let Some(OwnedValue::String(sha)) = hashes.get("sha256") else {
        return None;
    };
    let decoded = base64::engine::general_purpose::STANDARD_NO_PAD
        .decode(sha)
        .ok()?;
    let bytes: [u8; 16] = decoded.get(..16)?.try_into().ok()?;
    Some(bytes)
}

fn cmd_repack(
    cli: &Cli,
    room: Option<&str>,
    shard: Option<u16>,
    roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    let target = match (room, shard) {
        (Some(room), None) => RepackTarget::Room(parse_room_id(room)?),
        (None, Some(shard_id)) => RepackTarget::Shard(shard_id),
        // Clap rejects the both-targets case through `conflicts_with`; this
        // branch gives the missing-target case a readable diagnostic.
        _ => {
            return Err(anyhow!(
                "exactly one of --room <room> | --shard <shard> is required"
            ))
        }
    };

    if !roots.is_empty() {
        match target {
            RepackTarget::Shard(_) => {
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
    cmd_repack_target(cli, target, topo)
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

#[derive(Clone, Copy)]
enum RepackTarget {
    Room([u8; 16]),
    Shard(u16),
}

fn resolve_repack_target(
    store: &PackfileStorage,
    target: RepackTarget,
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    match target {
        RepackTarget::Room(room_id) => {
            if store.room_index_info(&room_id).is_none() {
                bail!("room {} not found", hex_encode(&room_id));
            }
            Ok((vec![room_id], store.room_referenced_shards(&room_id)))
        }
        RepackTarget::Shard(shard_id) => Ok(store.repack_closure(shard_id)?),
    }
}

fn repack_preview(
    cli: &Cli,
    target: RepackTarget,
    topo: bool,
) -> anyhow::Result<Option<RepackPreview>> {
    let preview_store = open_store_read_only(cli)?;
    let (rooms, shards) = resolve_repack_target(&preview_store, target)?;
    if rooms.is_empty() {
        if let RepackTarget::Shard(shard_id) = target {
            println!("no rooms reference shard {shard_id}");
        }
        return Ok(None);
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
    for shard_id in &shards {
        let bytes = shard_sizes.get(shard_id).copied().unwrap_or(0);
        println!("  slot {shard_id}: {:>9}", fmt_bytes(bytes));
    }
    println!("  total:    {:>9}", fmt_bytes(total_input_bytes));
    println!(
        "expected result: ~{expected_shards} shard{} ({total_kept} nodes across {} room{} rewritten)",
        if expected_shards == 1 { "" } else { "s" },
        rooms.len(),
        if rooms.len() == 1 { "" } else { "s" },
    );
    println!(
        "                 ~{} / {} = {}; {total_dropped} nodes pruned; ~{} free shard",
        fmt_bytes(total_kept_bytes),
        fmt_bytes(total_input_bytes),
        fmt_percent(total_kept_bytes, total_input_bytes),
        fmt_bytes(slop),
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
    let mut results = Vec::with_capacity(rooms.len());
    for room_id in rooms {
        let (kept, dropped) = if topo {
            store.repack_room_reachable(room_id, extract_matrix_edges)?
        } else {
            store.repack_room_reachable(room_id, |_hash, _data| Vec::new())?
        };
        println!(
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
    println!(
        "done: {} rooms repacked, {final_kept} kept, {final_dropped} dropped",
        results.len()
    );
    Ok((final_kept, final_dropped))
}

fn cmd_repack_target(cli: &Cli, target: RepackTarget, topo: bool) -> anyhow::Result<()> {
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
    drop(store);
    println!("post-repack shard state:");
    cmd_shards(cli)?;
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
        if !confirm("Delete this room?")? {
            eprintln!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    let count = store.room_index_info(&room_id).map_or(0, |(len, _)| len);
    store.delete_room(&room_id)?;
    eprintln!("deleted {count} nodes for room {hex}");
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
    use super::fmt_megabytes;

    #[test]
    fn index_memory_megabytes_uses_fixed_point_rounding() {
        assert_eq!(fmt_megabytes(524_328), "0.5243 MB");
        assert_eq!(fmt_megabytes(999_999), "1.0000 MB");
    }
}
