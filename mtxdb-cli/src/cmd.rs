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
/// memory shown in human-facing collection output.
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
/// small records without making large-collection listings visually noisy.
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

/// Raw, on-disk layout. It intentionally includes superseded frames: this is
/// the physical interleaving a sequential scan sees, not a claim about a
/// collection's live index footprint.
#[derive(Default)]
struct PhysicalLayout {
    collections: HashMap<[u8; 16], CollectionPhysicalLayout>,
    packs: HashMap<u64, PackPhysicalLayout>,
}

#[derive(Default)]
struct CollectionPhysicalLayout {
    disk_bytes: u64,
    pack_bytes: HashMap<u64, u64>,
    segments: u64,
    largest_segment_bytes: u64,
}

#[derive(Default)]
struct PackPhysicalLayout {
    collections: HashSet<[u8; 16]>,
    segments: u64,
    largest_segment_bytes: u64,
}

/// Scan physical frame placement without allocating payloads. A torn active
/// tail is ignored, just as the ordinary disk accounting scan does.
fn physical_layout(dir: &Path) -> anyhow::Result<PhysicalLayout> {
    // 1-byte flags + 4-byte uncompressed_len + 16-byte collection_id + 16-byte
    // hash — the fixed part of a v3 frame's payload, preceding the
    // (possibly zstd-compressed) node bytes.
    const FRAME_FIXED_LEN: u64 = 1 + 4 + 16 + 16;

    let mut layout = PhysicalLayout::default();
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
        let Some(header) = mtxdb_core::packfile::read_header(&mut reader)? else {
            continue;
        };
        let pack_id = header.pack_id;
        let mut current_run: Option<([u8; 16], u64)> = None;
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
                bail!("invalid record length {frame_len} while measuring collection disk usage");
            }
            // Skip the flags byte and uncompressed_len field to reach collection_id
            // — this scan only needs the collection ID, not whether/how the node
            // bytes that follow are compressed.
            let mut flags_and_uncompressed_len = [0_u8; 5];
            match reader.read_exact(&mut flags_and_uncompressed_len) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
            let mut collection_id = [0_u8; 16];
            match reader.read_exact(&mut collection_id) {
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
            let record_bytes = frame_len.saturating_add(8);
            let collection = layout.collections.entry(collection_id).or_default();
            collection.disk_bytes = collection.disk_bytes.saturating_add(record_bytes);
            let pack_bytes = collection.pack_bytes.entry(pack_id).or_default();
            *pack_bytes = pack_bytes.saturating_add(record_bytes);
            layout
                .packs
                .entry(pack_id)
                .or_default()
                .collections
                .insert(collection_id);

            match &mut current_run {
                Some((run_collection, run_bytes)) if *run_collection == collection_id => {
                    *run_bytes = run_bytes.saturating_add(record_bytes);
                }
                Some(_) => {
                    finish_physical_run(&mut layout, pack_id, current_run.take());
                    current_run = Some((collection_id, record_bytes));
                }
                None => current_run = Some((collection_id, record_bytes)),
            }
        }
        finish_physical_run(&mut layout, pack_id, current_run);
    }
    Ok(layout)
}

fn finish_physical_run(layout: &mut PhysicalLayout, pack_id: u64, run: Option<([u8; 16], u64)>) {
    let Some((collection_id, bytes)) = run else {
        return;
    };
    let collection = layout.collections.entry(collection_id).or_default();
    collection.segments = collection.segments.saturating_add(1);
    collection.largest_segment_bytes = collection.largest_segment_bytes.max(bytes);
    let pack = layout.packs.entry(pack_id).or_default();
    pack.segments = pack.segments.saturating_add(1);
    pack.largest_segment_bytes = pack.largest_segment_bytes.max(bytes);
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
        Commands::Put {
            collection,
            id,
            data,
        } => cmd_put(cli, collection, id, data),
        Commands::Get { collection, id } => cmd_get(cli, collection.as_deref(), id),
        Commands::Collections { all, layout, sort } => {
            cmd_collections(cli, *all, *layout, sort.as_deref())
        }
        Commands::Shards { all, layout, sort } => cmd_shards(cli, *all, *layout, sort.as_deref()),
        Commands::Info { collection } => cmd_info(cli, collection),
        Commands::Scan { shard } => cmd_scan(cli, shard),
        Commands::Import { paths, collection } => cmd_import(cli, paths, collection.as_deref()),
        Commands::Export { collection } => cmd_export(cli, collection),
        Commands::Repack {
            collection,
            shards,
            all,
            root,
            topo,
        } => cmd_repack(cli, collection.as_deref(), shards, *all, root, *topo),
        Commands::Delete { collections, yes } => cmd_delete(cli, collections, *yes),
        Commands::Completions { .. } => unreachable!("main emits completion scripts directly"),
        Commands::Sync => cmd_sync(cli),
    }
}

fn parse_collection_id(hex: &str) -> anyhow::Result<[u8; 16]> {
    let hex = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    if hex.len() != 32 {
        bail!("collection ID must be 32 hex characters, got {}", hex.len());
    }
    let bytes = hex::decode(hex).context("invalid hex in collection ID")?;
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
/// than contending with it. For commands that only ever read collection data.
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

fn cmd_put(cli: &Cli, collection: &str, id: &str, data: &str) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    let node_data = NodeData::new(bytes::Bytes::from(data.as_bytes().to_vec()));
    store.put(&collection_id, &node_id, &node_data)?;
    let collection_hex = hex_encode(&collection_id);
    let id_hex = hex_encode(&node_id);
    eprintln!(
        "put {id_hex} into collection {collection_hex} ({} bytes)",
        data.len()
    );
    Ok(())
}

fn cmd_get(cli: &Cli, collection: Option<&str>, id: &str) -> anyhow::Result<()> {
    let node_id = parse_get_id(id)?;
    let store = open_store_read_only(cli)?;
    let matches: Vec<([u8; 16], NodeData)> = match collection {
        Some(collection) => {
            let collection_id = parse_collection_id(collection)?;
            store
                .get(&collection_id, &node_id)?
                .map(|data| vec![(collection_id, data)])
                .unwrap_or_default()
        }
        None => collection_ids(cli)?
            .into_iter()
            .filter_map(|collection_id| match store.get(&collection_id, &node_id) {
                Ok(Some(data)) => Some(Ok((collection_id, data))),
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
            "node ID {id} is present in multiple collections ({}); specify --collection",
            matches
                .iter()
                .map(|(collection_id, _)| hex_encode(collection_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
    Ok(())
}

/// Enumerate live collections from the persisted directory. Stores created before
/// that sidecar existed fall back to a one-time index rebuild; `mtxdb sync`
/// makes future calls fast.
fn collection_ids(cli: &Cli) -> anyhow::Result<Vec<[u8; 16]>> {
    let dir = selected_pool_dir(cli)?;
    if PackfileStorage::collection_directory_persisted_at(&dir).is_some() {
        return Ok(PackfileStorage::collection_directory_from_disk(&dir)
            .into_iter()
            .map(|(collection_id, _)| collection_id)
            .collect());
    }
    Ok(open_store_read_only(cli)?
        .collection_summaries()
        .into_iter()
        .map(|(collection_id, _, _)| collection_id)
        .collect())
}

fn cmd_collections(cli: &Cli, all: bool, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    if all {
        let db_layout = open_layout(cli)?;
        for (index, shard_type) in ShardType::ALL.into_iter().enumerate() {
            if index != 0 {
                println!();
            }
            println!("{}:", shard_type.as_str());
            cmd_collections_in_dir(&pool_dir(&db_layout, shard_type)?, layout, sort)?;
        }
        return Ok(());
    }
    cmd_collections_in_dir(&selected_pool_dir(cli)?, layout, sort)
}

/// Bytes that could be moved to maximize one collection's largest pack extent.
fn avoidable_spread_bytes(stats: Option<&CollectionPhysicalLayout>) -> u64 {
    let Some(stats) = stats else {
        return 0;
    };
    let largest_pack = stats.pack_bytes.values().copied().max().unwrap_or(0);
    stats
        .disk_bytes
        .min(mtxdb_core::shard::MAX_SHARD_BYTES)
        .saturating_sub(largest_pack)
}

/// List logical collections from one pool. Cross-pool aggregation is deliberately
/// avoided: each pool owns an independent 16-byte namespace and lifecycle.
#[allow(
    clippy::too_many_lines,
    reason = "the command intentionally keeps its table construction and summary together"
)]
fn cmd_collections_in_dir(dir: &Path, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    // The persisted collection directory is a fast listing snapshot, not proof
    // that the shard files are readable by this binary. Validate the small
    // immutable header of every shard before trusting it, so a pre-cutover
    // store does not misleadingly appear empty just because its sidecar is
    // empty or stale.
    validate_packfile_headers(dir)?;
    let collections = match PackfileStorage::collection_summaries_from_disk(dir) {
        Some(collections) => collections,
        // A named pool is created with the database layout, before it has
        // necessarily received a first write. `open_read_only` quite
        // properly rejects a directory with no shard files, but for a
        // listing that simply means there are no collections to show.
        None if glob_pack_files(dir)?.is_empty() => Vec::new(),
        // Old stores have no sidecar yet. Keep the complete, slower fallback
        // so `collections` remains useful until `mtxdb sync` writes one.
        None => PackfileStorage::open_read_only(dir.into())
            .context("failed to open store")?
            .collection_summaries(),
    };
    let collection_shards = PackfileStorage::collection_shards_from_disk(dir);
    let physical = physical_layout(dir)?;
    let disk_bytes: HashMap<_, _> = physical
        .collections
        .iter()
        .map(|(id, stats)| (*id, stats.disk_bytes))
        .collect();

    if collections.is_empty() {
        println!("no collections found");
        return Ok(());
    }

    if let Some(column) = sort {
        if !matches!(
            column,
            "slot"
                | "collection"
                | "nodes"
                | "shards"
                | "index-ram"
                | "disk"
                | "packs"
                | "avoidable"
                | "segments"
                | "fragmentation"
        ) {
            bail!("unknown collections sort column `{column}`");
        }
    }
    let mut ordered: Vec<_> = collections.iter().enumerate().collect();
    ordered.sort_by(|(left_slot, left), (right_slot, right)| {
        let left_layout = physical.collections.get(&left.0);
        let right_layout = physical.collections.get(&right.0);
        let score = |stats: Option<&CollectionPhysicalLayout>| {
            stats.map_or(0, |s| s.segments.saturating_sub(s.pack_bytes.len() as u64))
        };
        let ordering = match sort.unwrap_or("slot") {
            "collection" => left.0.cmp(&right.0),
            "nodes" => right.1.cmp(&left.1),
            "shards" | "packs" => right_layout
                .map_or(0, |s| s.pack_bytes.len())
                .cmp(&left_layout.map_or(0, |s| s.pack_bytes.len())),
            "index-ram" => right.2.cmp(&left.2),
            "disk" => right_layout
                .map_or(0, |s| s.disk_bytes)
                .cmp(&left_layout.map_or(0, |s| s.disk_bytes)),
            "avoidable" => {
                avoidable_spread_bytes(right_layout).cmp(&avoidable_spread_bytes(left_layout))
            }
            "segments" | "fragmentation" => score(right_layout).cmp(&score(left_layout)),
            _ => left_slot.cmp(right_slot),
        };
        ordering.then_with(|| left.0.cmp(&right.0))
    });
    if layout {
        println!(
            "  {:>4}  {:<34}  {:>7}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
            "slot", "collection", "nodes", "packs", "disk", "runs", "largest", "avoidable"
        );
    } else {
        println!(
            "  {:>4}  {:<34}  {:>7}  {:>6}  {:>12}  {:>13}",
            "slot", "collection", "nodes", "shards", "index RAM", "disk"
        );
    }
    let mut total_nodes = 0_usize;
    let mut total_memory = 0_usize;
    let mut total_disk_bytes = 0_u64;
    for (i, (collection_id, nodes, memory)) in ordered {
        let hex = hex_encode(collection_id);
        let shards = collection_shards
            .as_ref()
            .and_then(|by_collection| by_collection.get(collection_id))
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
            .context("total collection node count overflow")?;
        total_memory = total_memory
            .checked_add(*memory)
            .context("total collection index memory overflow")?;
        let disk = disk_bytes.get(collection_id).copied().unwrap_or(0);
        total_disk_bytes = total_disk_bytes.saturating_add(disk);
        if layout {
            let stats = physical.collections.get(collection_id);
            let packs = stats.map_or(0, |s| s.pack_bytes.len());
            let runs = stats.map_or(0, |s| s.segments);
            let largest = stats.map_or(0, |s| s.largest_segment_bytes);
            let avoidable = avoidable_spread_bytes(stats);
            println!(
                "  {i:>4}  0x{hex}  {nodes:>7}  {packs:>6}  {:>13}  {runs:>5}  {:>10}  {:>13}",
                fmt_disk_megabytes(disk),
                fmt_bytes(largest),
                fmt_bytes(avoidable)
            );
        } else {
            println!(
                "  {i:>4}  0x{hex}  {nodes:>7}  {shards:>6}  {:>12}  {:>13}",
                fmt_megabytes(*memory),
                fmt_disk_megabytes(disk),
            );
        }
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
    if layout {
        let total_segments: u64 = physical.collections.values().map(|s| s.segments).sum();
        let spread = physical
            .collections
            .values()
            .filter(|s| s.pack_bytes.len() > 1)
            .count();
        println!("physical layout: {spread} collection(s) span multiple packs; {total_segments} contiguous runs (includes superseded frames)");
    }
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
/// needs no collection data and no frame reads. Instead it validates pack
/// headers, then decodes the small `shard_stats.bin` and
/// `shard_collections.bin` sidecars for counters and live-node counts.
/// Safe to run against a directory a live writer process owns.
fn cmd_shards(cli: &Cli, all: bool, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    if all {
        let db_layout = open_layout(cli)?;
        for (index, shard_type) in ShardType::ALL.into_iter().enumerate() {
            if index != 0 {
                println!();
            }
            println!("{}:", shard_type.as_str());
            cmd_shards_in_dir(&pool_dir(&db_layout, shard_type)?, layout, sort)?;
        }
        return Ok(());
    }
    cmd_shards_in_dir(&selected_pool_dir(cli)?, layout, sort)
}

/// List shard metadata from one pool only. This reads directory entries and
/// sidecars; it never scans packfile contents or opens collection indexes.
fn cmd_shards_in_dir(dir: &Path, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    let mut shard_entries = glob_pack_files(dir)?;
    shard_entries.sort_unstable_by_key(|&(id, _, _)| id);

    if shard_entries.is_empty() {
        println!("no shards found in `{}` (store is empty)", dir.display());
        return Ok(());
    }

    let (stats_map, persisted_at) = decode_stats_snapshot(dir);
    let node_counts = PackfileStorage::shard_node_counts_from_disk(dir);
    let collection_counts = PackfileStorage::shard_collection_counts_from_disk(dir);
    let needs_layout = layout || matches!(sort, Some("segments" | "interleaving"));
    let physical = needs_layout.then(|| physical_layout(dir)).transpose()?;
    if let Some(column) = sort {
        if !matches!(
            column,
            "pack" | "bytes" | "nodes" | "collections" | "syncs" | "segments" | "interleaving"
        ) {
            bail!("unknown shards sort column `{column}`");
        }
        shard_entries.sort_by(|left, right| {
            let value = |entry: &(u64, u64, u8)| match column {
                "bytes" => entry.1,
                "nodes" => node_counts
                    .as_ref()
                    .and_then(|counts| counts.get(&entry.0))
                    .copied()
                    .unwrap_or(0),
                "collections" => collection_counts
                    .as_ref()
                    .and_then(|counts| counts.get(&entry.0))
                    .copied()
                    .unwrap_or(0),
                "syncs" => stats_map.get(&entry.0).copied().unwrap_or_default().2,
                "segments" => physical
                    .as_ref()
                    .and_then(|stats| stats.packs.get(&entry.0))
                    .map_or(0, |stats| stats.segments),
                "interleaving" => physical
                    .as_ref()
                    .and_then(|stats| stats.packs.get(&entry.0))
                    .map_or(0, |stats| {
                        stats
                            .segments
                            .saturating_sub(stats.collections.len() as u64)
                    }),
                _ => 0,
            };
            if column == "pack" {
                left.0.cmp(&right.0)
            } else {
                value(right)
                    .cmp(&value(left))
                    .then_with(|| left.0.cmp(&right.0))
            }
        });
    }
    let total_collections = collection_counts
        .as_ref()
        .map(|_| PackfileStorage::collection_directory_from_disk(dir).len());
    print_shard_table(
        &shard_entries,
        &stats_map,
        node_counts.as_ref(),
        collection_counts.as_ref(),
        total_collections,
    );
    if let Some(physical) = physical {
        let runs: u64 = physical.packs.values().map(|stats| stats.segments).sum();
        let interleaved: u64 = physical
            .packs
            .values()
            .map(|stats| {
                stats
                    .segments
                    .saturating_sub(stats.collections.len() as u64)
            })
            .sum();
        println!("physical layout: {runs} contiguous runs; {interleaved} excess runs from interleaving (includes superseded frames)");
        println!(
            "{:>19}  {:>11}  {:>6}  {:>10}  {:>12}",
            "pack id", "collections", "runs", "excess", "largest run"
        );
        for (pack_id, _, _) in &shard_entries {
            let stats = physical.packs.get(pack_id);
            let collections = stats.map_or(0, |stats| stats.collections.len());
            let runs = stats.map_or(0, |stats| stats.segments);
            let excess = runs.saturating_sub(collections as u64);
            let largest = stats.map_or(0, |stats| stats.largest_segment_bytes);
            println!(
                "0x{pack_id:016x}  {collections:>11}  {runs:>6}  {excess:>10}  {:>12}",
                fmt_bytes(largest)
            );
        }
    }
    println!("* active pack");
    println!(
        "{} pack(s), {}",
        shard_entries.len(),
        stats_snapshot_summary(persisted_at)
    );
    Ok(())
}

/// Discover only canonical v4 `pack_{pack_id:016x}.pack` files.
///
fn glob_pack_files(dir: &Path) -> anyhow::Result<Vec<(u64, u64, u8)>> {
    let mut packs = Vec::new();
    let mut seen = HashSet::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.extension().is_some_and(|e| e == "pack") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let id_hex = match stem.strip_prefix("pack_") {
            Some(id_hex) => id_hex,
            None if stem.starts_with("shard_") => {
                bail!(
                    "found pre-v4 shard file {}; reset it rather than opening it as v4",
                    path.display()
                );
            }
            None => continue,
        };
        if id_hex.len() != 16
            || !id_hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            bail!(
                "invalid v4 pack filename {}; expected pack_{{16 lowercase hex digits}}.pack",
                path.display()
            );
        }
        let pack_id = u64::from_str_radix(id_hex, 16)?;
        if !seen.insert(pack_id) {
            bail!("duplicate pack_id {pack_id:#x} in {}", dir.display());
        }
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open pack `{}`", path.display()))?;
        let header = mtxdb_core::packfile::read_header(&mut BufReader::new(file))
            .with_context(|| format!("unsupported or corrupt pack `{}`", path.display()))?
            .with_context(|| format!("invalid pack header `{}`", path.display()))?;
        if header.pack_id != pack_id {
            bail!(
                "pack {} identifies itself as {:#x}",
                path.display(),
                header.pack_id
            );
        }
        packs.push((
            pack_id,
            entry.metadata()?.len(),
            mtxdb_core::packfile::VERSION,
        ));
    }
    packs.sort_unstable_by_key(|(pack_id, _, _)| *pack_id);
    Ok(packs)
}

/// Pack ID to `(write_count, bytes_written, sync_count)`, as decoded from
/// `shard_stats.bin`.
type ShardStatsMap = std::collections::HashMap<u64, (u64, u64, u64)>;

/// Decode `shard_stats.bin` — same binary format as
/// `ShardPool::restore_persisted_stats`, but standalone. Returns a map
/// from a `pack_id` key to `(write_count,
/// bytes_written, sync_count)` and the snapshot's persisted-at timestamp.
fn decode_stats_snapshot(dir: &Path) -> (ShardStatsMap, Option<u64>) {
    const STATS_MAGIC: &[u8; 4] = b"MSTA";
    const STATS_VERSION: u8 = 4;
    const STATS_HEADER_LEN: usize = 4 + 1 + 8;
    const STATS_RECORD_LEN: usize = 8 + 8 * 3;

    let mut stats_map = std::collections::HashMap::new();
    let mut persisted_at = None;

    let stats_path = dir.join("shard_stats.bin");
    let Ok(buf) = fs::read(stats_path) else {
        return (stats_map, persisted_at);
    };
    if buf.len() < STATS_HEADER_LEN || &buf[0..4] != STATS_MAGIC || buf[4] != STATS_VERSION {
        return (stats_map, persisted_at);
    }

    persisted_at = Some(u64::from_le_bytes(buf[5..13].try_into().unwrap_or([0; 8])));
    let body = &buf[STATS_HEADER_LEN..];
    for chunk in body.chunks(STATS_RECORD_LEN) {
        if let Ok(rec) = <&[u8; STATS_RECORD_LEN]>::try_from(chunk) {
            let pack_id = u64::from_le_bytes(rec[0..8].try_into().unwrap());
            let write_count = u64::from_le_bytes(rec[8..16].try_into().unwrap());
            let bytes_written = u64::from_le_bytes(rec[16..24].try_into().unwrap());
            let sync_count = u64::from_le_bytes(rec[24..32].try_into().unwrap());
            stats_map.insert(pack_id, (write_count, bytes_written, sync_count));
        }
    }
    (stats_map, persisted_at)
}

/// Print the shard table header and rows.
fn print_shard_table(
    shard_entries: &[(u64, u64, u8)],
    stats_map: &ShardStatsMap,
    node_counts: Option<&std::collections::HashMap<u64, u64>>,
    collection_counts: Option<&std::collections::HashMap<u64, u64>>,
    total_collections: Option<usize>,
) {
    // `ShardPool::open_internal` restores the newest pack as its append
    // destination. Mirror that recovery rule here without opening a writer.
    let active_pack_id = shard_entries.iter().map(|(pack_id, _, _)| *pack_id).max();

    println!(
        "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>6}",
        "pack id", "ver", "bytes", "nodes", "collections", "syncs",
    );
    let mut total_bytes = 0u64;
    let mut total_nodes = node_counts.map(|_| 0u64);
    let mut total_syncs = 0u64;
    for &(pack_id, file_bytes, version) in shard_entries {
        let (_, _, sc) = stats_map.get(&pack_id).copied().unwrap_or_default();
        let nodes = node_counts
            .and_then(|counts| counts.get(&pack_id))
            .map_or_else(|| "?".to_owned(), u64::to_string);
        let collections = collection_counts
            .and_then(|counts| counts.get(&pack_id))
            .map_or_else(|| "?".to_owned(), u64::to_string);
        println!(
            "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>6}",
            format!(
                "0x{pack_id:016x}{}",
                if active_pack_id == Some(pack_id) {
                    "*"
                } else {
                    " "
                }
            ),
            version,
            fmt_bytes(file_bytes),
            nodes,
            collections,
            sc,
        );
        total_bytes = total_bytes.saturating_add(file_bytes);
        total_syncs = total_syncs.saturating_add(sc);
        if let (Some(total), Some(counts)) = (&mut total_nodes, node_counts) {
            *total = total.saturating_add(counts.get(&pack_id).copied().unwrap_or(0));
        }
    }
    println!();
    println!(
        "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>6}",
        "total",
        "",
        fmt_bytes(total_bytes),
        total_nodes.map_or_else(|| "?".to_owned(), |count| count.to_string()),
        total_collections.map_or_else(|| "?".to_owned(), |count| count.to_string()),
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

fn cmd_info(cli: &Cli, collection: &str) -> anyhow::Result<()> {
    let collection_id = match collection.parse::<usize>() {
        Ok(slot) => {
            let collections = collection_ids(cli)?;
            collections
                .get(slot)
                .copied()
                .with_context(|| format!("collection slot {slot} not found"))?
        }
        Err(_) => parse_collection_id(collection)?,
    };
    let hex = hex_encode(&collection_id);
    let dir = selected_pool_dir(cli)?;

    // The persisted directory is enough for the numerical summary and the
    // collection's physical placement. Do not rebuild every collection index merely to
    // answer `info` for one collection.
    if let (Some(summaries), Some(collection_shards)) = (
        PackfileStorage::collection_summaries_from_disk(&dir),
        PackfileStorage::collection_shards_from_disk(&dir),
    ) {
        if let Some((_, len, mem)) = summaries
            .into_iter()
            .find(|(id, _, _)| *id == collection_id)
        {
            println!(
                "collection {hex}: {len} nodes, {} index RAM",
                fmt_megabytes(mem)
            );
            let shards = collection_shards
                .get(&collection_id)
                .cloned()
                .unwrap_or_default();
            print_collection_shards(&shards);
            if let Some(details) = matrix_room_details_from_cache(&dir, &collection_id) {
                print_matrix_room_details(details, None);
            } else {
                println!(
                    "  {:<12} scanning {} shard{}...",
                    "metadata:",
                    shards.len(),
                    if shards.len() == 1 { "" } else { "s" }
                );
                let details = matrix_room_details_from_shards(&dir, &collection_id, &shards)
                    .unwrap_or_else(|error| {
                        eprintln!("warning: unable to inspect Matrix room metadata: {error}");
                        MatrixRoomDetails::default()
                    });
                if let Err(error) = persist_matrix_room_details(&dir, &collection_id, &details) {
                    eprintln!("warning: unable to cache Matrix room metadata: {error}");
                }
                print_matrix_room_details(details, Some(shards.len()));
            }
            return Ok(());
        }
        eprintln!("collection {hex}: not found");
        return Ok(());
    }

    // A store predating the inspection sidecar has no cheap authoritative
    // summary. Preserve the old full-scan fallback until `mtxdb sync` can
    // create the sidecar.
    let store = open_store_read_only(cli)?;
    match store.collection_index_info(&collection_id) {
        Some((len, mem)) => {
            println!(
                "collection {hex}: {len} nodes, {} index RAM",
                fmt_megabytes(mem)
            );
            let shards = store.collection_referenced_pack_ids(&collection_id);
            print_collection_shards(&shards);
            print_matrix_room_details(matrix_room_details(&store, &dir, &collection_id)?, None);
        }
        None => eprintln!("collection {hex}: not found"),
    }
    Ok(())
}

#[derive(Clone, Default)]
struct MatrixRoomDetails {
    matrix_room_id: Option<String>,
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
/// metadata; packfiles and the shard→collection directory remain authoritative.
fn matrix_room_details_from_cache(
    dir: &Path,
    collection_id: &[u8; 16],
) -> Option<MatrixRoomDetails> {
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
        if &header[0..16] == collection_id {
            let matrix_room_id = std::str::from_utf8(&data[..matrix_len])
                .ok()
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            let create = std::str::from_utf8(&data[matrix_len..])
                .ok()
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            return Some(MatrixRoomDetails {
                matrix_room_id,
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
    collection_id: &[u8; 16],
    details: &MatrixRoomDetails,
) -> io::Result<()> {
    if details.matrix_room_id.is_none() && details.create.is_none() {
        return Ok(());
    }
    let mut entries = read_all_matrix_room_details(dir);
    entries.insert(*collection_id, details.clone());

    let mut buf = Vec::new();
    buf.extend_from_slice(MATRIX_ROOM_DETAILS_MAGIC);
    buf.push(MATRIX_ROOM_DETAILS_VERSION);
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_unstable_by_key(|(collection_id, _)| *collection_id);
    for (collection_id, details) in entries {
        let matrix_room_id = details.matrix_room_id.clone().unwrap_or_default();
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
        buf.extend_from_slice(&collection_id);
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
        let mut collection_id = [0u8; 16];
        collection_id.copy_from_slice(&header[0..16]);
        entries.insert(
            collection_id,
            MatrixRoomDetails {
                matrix_room_id: (!matrix_room_id.is_empty()).then(|| matrix_room_id.to_owned()),
                create: (!create.is_empty()).then(|| create.to_owned()),
            },
        );
        offset = data_end;
    }
    entries
}

/// Find the human-facing Matrix metadata carried by live JSON events. Pack
/// scans can contain superseded frames, so every candidate is resolved through
/// the collection's live index before being inspected.
fn matrix_room_details(
    store: &PackfileStorage,
    dir: &Path,
    collection_id: &[u8; 16],
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
        for (record_collection_id, node_id, _) in mtxdb_core::packfile::scan_packfile(&path)? {
            if &record_collection_id != collection_id || !seen.insert(node_id) {
                continue;
            }
            let Some(data) = store.get(collection_id, &node_id)? else {
                continue;
            };
            let mut bytes = data.bytes.to_vec();
            let Ok(event) = simd_json::to_owned_value(&mut bytes) else {
                continue;
            };
            if details.matrix_room_id.is_none() {
                details.matrix_room_id = event_room_id(&event).map(str::to_owned);
            }
            if details.create.is_none() {
                details.create = matrix_create_details(&event);
            }
            if details.matrix_room_id.is_some() && details.create.is_some() {
                return Ok(details);
            }
        }
    }
    Ok(details)
}

/// Read Matrix metadata from only the shards containing `collection_id`. This is
/// deliberately streaming: common Matrix room IDs and create events occur
/// near the beginning of topology-ordered data, so `info` can stop as soon
/// as both fields are found rather than scanning unrelated shards or loading
/// a full store index.
fn matrix_room_details_from_shards(
    dir: &Path,
    collection_id: &[u8; 16],
    pack_ids: &[u64],
) -> anyhow::Result<MatrixRoomDetails> {
    let pool = ShardPool::open_read_only(dir.into()).context("failed to open shard store")?;
    let mut details = MatrixRoomDetails::default();
    for &pack_id in pack_ids {
        let Some(shard) = pool
            .all_shards()
            .into_iter()
            .find(|(_, s)| s.pack_id == pack_id)
            .map(|(_, s)| s)
        else {
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
            if &record.collection_id != collection_id {
                continue;
            }
            update_matrix_room_details(&mut details, &record.data);
            if details.matrix_room_id.is_some() && details.create.is_some() {
                return Ok(details);
            }
        }
    }
    Ok(details)
}

fn print_matrix_room_details(details: MatrixRoomDetails, scanned_shards: Option<usize>) {
    if let Some(matrix_room_id) = details.matrix_room_id {
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

fn print_collection_shards(shards: &[u64]) {
    match shards {
        [shard] => println!("  {:<12} 0x{shard:016x}", "pack:"),
        [] => {}
        _ => println!(
            "  {:<12} {}",
            "packs:",
            shards
                .iter()
                .map(|id| format!("0x{id:016x}"))
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
    if details.matrix_room_id.is_none() {
        details.matrix_room_id = event_room_id(&event).map(str::to_owned);
    }
    if details.create.is_none() {
        details.create = matrix_create_details(&event);
    }
}

/// Parse an operator-facing pack identifier. Slots are deliberately not
/// accepted here: they are recycled implementation details, while a pack ID
/// is the permanent identity printed by `mtxdb shards`.
fn parse_pack_id_selector(selector: &str) -> anyhow::Result<u64> {
    let Some(hex) = selector.strip_prefix("0x") else {
        bail!("invalid pack ID `{selector}`; use the 0x-prefixed ID shown by `mtxdb shards`");
    };
    if hex.is_empty() || hex.len() > 16 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid pack ID `{selector}`; expected 1–16 hexadecimal digits after 0x");
    }
    u64::from_str_radix(hex, 16).with_context(|| format!("invalid pack ID `{selector}`"))
}

fn cmd_scan(cli: &Cli, selector: &str) -> anyhow::Result<()> {
    let pack_id = parse_pack_id_selector(selector)?;
    let pool_dir = selected_pool_dir(cli)?;
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let shard = pool
        .all_shards()
        .into_iter()
        .find_map(|(_, shard)| (shard.pack_id == pack_id).then_some(shard))
        .with_context(|| format!("pack ID 0x{pack_id:016x} not found"))?;
    let path = &shard.path;
    let records = mtxdb_core::packfile::scan_packfile(path)?;
    println!(
        "pack 0x{pack_id:016x}: {} bytes, {} records",
        std::fs::metadata(path)?.len(),
        records.len()
    );
    for (collection_id, node_id, offset) in &records {
        let collection_hex = hex_encode(collection_id);
        let id_hex = hex_encode(node_id);
        println!("  collection={collection_hex} id={id_hex} @ {offset}");
    }
    Ok(())
}

fn cmd_import(
    cli: &Cli,
    paths: &[std::path::PathBuf],
    collection_override: Option<&str>,
) -> anyhow::Result<()> {
    // One writer owns the entire batch: it prevents another writer from
    // interleaving halfway through a shell glob, and lets us publish one
    // complete shard/collection snapshot once the final input has been handled.
    let store = open_store(cli)?;
    let mut failures = 0_usize;
    for (index, path) in paths.iter().enumerate() {
        if index != 0 {
            eprintln!();
        }
        if let Err(error) = cmd_import_file(&store, path, collection_override) {
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

/// Export every live JSON record in one collection as a JSONL stream. A read-only
/// frame scan avoids stale persisted collection summaries and rebuilding every
/// collection's in-memory index just to enumerate one collection. Scanning packs
/// in pack-ID order lets a later physical copy replace an older one with the same
/// node ID.
fn cmd_export(cli: &Cli, collection: &str) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let pool_dir = selected_pool_dir(cli)?;
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let mut shards = pool.all_shards();
    shards.sort_unstable_by_key(|(_, shard)| shard.pack_id);

    let mut locations = HashMap::new();
    let mut ordered_ids = Vec::new();
    for (shard_index, (_, shard)) in shards.iter().enumerate() {
        for (candidate_collection, node_id, offset) in
            mtxdb_core::packfile::scan_packfile(&shard.path)?
        {
            if candidate_collection == collection_id {
                if !locations.contains_key(&node_id) {
                    ordered_ids.push(node_id);
                }
                locations.insert(node_id, (shard_index, offset));
            }
        }
    }
    if locations.is_empty() {
        bail!("collection {collection} not found");
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
        let mut bytes = record.data.to_vec();
        let value: OwnedValue = simd_json::to_owned_value(&mut bytes)
            .with_context(|| format!("record {} is not valid JSON", hex_encode(&node_id)))?;
        let json = value.to_string();
        output.write_all(json.as_bytes())?;
        output.write_all(b"\n")?;
        exported = exported.saturating_add(1);
    }
    output.flush()?;
    eprintln!(
        "exported {exported} records from collection {}",
        hex_encode(&collection_id)
    );
    Ok(())
}

fn cmd_import_file(
    store: &PackfileStorage,
    path: &Path,
    collection_override: Option<&str>,
) -> anyhow::Result<()> {
    let content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let is_jsonl = path
        .extension()
        .is_some_and(|extension| extension == "jsonl");
    let (events, detected_collection) = if is_jsonl {
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
    let mut event_count = 0u64;
    let mut skipped = 0u64;
    let mut already_present = 0u64;
    // Successful imports are the nominal case and need no event-ID noise.
    // Keep a small sample only for the duplicate case, where it helps an
    // operator identify which input/store overlap caused the no-op.
    let mut already_present_ids = Vec::with_capacity(3);

    let collection_id = if let Some(r) = collection_override {
        parse_collection_id(r)?
    } else {
        let rid = detected_collection.with_context(|| {
            let first_event = events
                .first()
                .and_then(event_id)
                .unwrap_or("<missing event_id>");
            format!(
                "could not detect collection_id (first event: {first_event}); pass --collection for this input"
            )
        })?;
        let hash = blake3::hash(rid.as_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        id
    };

    let collection_hex = hex_encode(&collection_id);

    for ev in &events {
        let Some(incoming_event_id) = event_id(ev) else {
            skipped = skipped.saturating_add(1);
            continue;
        };
        let event_hash = blake3::hash(incoming_event_id.as_bytes());
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&event_hash.as_bytes()[..16]);

        let event_bytes = ev.encode().into_bytes();
        if let Some(existing) = store.get(&collection_id, &id_bytes)? {
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
            if already_present_ids.len() < 3 {
                already_present_ids.push(incoming_event_id);
            }
            continue;
        }
        let data = NodeData::new(bytes::Bytes::from(event_bytes));
        store.put(&collection_id, &id_bytes, &data)?;
        event_count = event_count.saturating_add(1);
    }

    eprintln!("imported {event_count} events to collection {collection_hex}");
    if skipped > 0 {
        eprintln!("skipped {skipped} events (missing event_id)");
    }
    if already_present > 0 {
        let suffix = if already_present > already_present_ids.len() as u64 {
            ", ..."
        } else {
            ""
        };
        eprintln!(
            "{already_present} events already present [{}{suffix}]",
            already_present_ids.join(", "),
        );
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
    let detected_collection = events.iter().find_map(event_room_id).map(str::to_owned);
    Ok((events, detected_collection))
}

fn parse_federation_events(content: &[u8]) -> anyhow::Result<(Vec<OwnedValue>, Option<String>)> {
    let mut bytes = content.to_vec();
    let val: OwnedValue = simd_json::to_owned_value(&mut bytes).context("invalid JSON")?;
    let detected_collection = event_room_id(&val).map(str::to_owned);
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
    Ok((events, detected_collection))
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
    collection: Option<&str>,
    shards: &[String],
    all: bool,
    roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    let target = match (collection, shards.is_empty(), all) {
        (Some(collection), true, false) => {
            RepackTarget::Collection(parse_collection_id(collection)?)
        }
        (None, false, false) => RepackTarget::Packs(parse_pack_selectors(shards)?),
        (None, true, true) => RepackTarget::All,
        // Clap rejects the both-targets case through `conflicts_with`; this
        // branch gives the missing-target case a readable diagnostic.
        _ => {
            return Err(anyhow!(
                "exactly one of --collection <collection> | --shard <shard> | --all is required"
            ))
        }
    };

    if !roots.is_empty() {
        match &target {
            RepackTarget::Packs(_) | RepackTarget::All => {
                bail!(
                    "--root requires --collection — live roots are per-collection, not meaningful for --shard"
                );
            }
            RepackTarget::Collection(_) if !topo => {
                bail!("--root requires --topo; without --topo there are no edges so only the specified roots would be kept");
            }
            RepackTarget::Collection(_) => {
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

/// Compacts every pack transitively touched by collections referencing a
/// selected pack — not just that pack itself. A collection's live data can span
/// more than one pack (`PackfileStorage::repack_closure` finds the full
/// closure), and repacking a collection always rewrites its *entire* live set
/// regardless of which pack triggered the repack, so a pack-scoped
/// compaction has to account for everything that repack will actually
/// touch, not just the one shard named on the command line.
///
/// Runs a non-mutating preflight first (`PackfileStorage::plan_collection_repack`
/// per collection in the closure): prints how many collections and packs are
/// involved and the expected pack count/slack after compaction, then
/// prompts for confirmation before performing any real repack. `--root`
/// doesn't apply here since live roots are inherently per-collection — the
/// same `--topo`/no-`--topo` edge-extraction choice applies uniformly
/// across every collection the closure touches.
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
    collections: Vec<[u8; 16]>,
    pack_ids: Vec<u64>,
}

enum RepackTarget {
    Collection([u8; 16]),
    Packs(Vec<u64>),
    All,
}

/// Parse one or more permanent pack IDs. Numeric slots and ranges are not
/// accepted because slots are recycled implementation details.
fn parse_pack_selectors(selectors: &[String]) -> anyhow::Result<Vec<u64>> {
    let mut pack_ids = std::collections::BTreeSet::new();
    for selector in selectors {
        pack_ids.insert(parse_pack_id_selector(selector)?);
    }
    Ok(pack_ids.into_iter().collect())
}

fn resolve_repack_target(
    store: &PackfileStorage,
    target: &RepackTarget,
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    match target {
        RepackTarget::Collection(collection_id) => {
            if store.collection_index_info(collection_id).is_none() {
                bail!("collection {} not found", hex_encode(collection_id));
            }
            Ok((
                vec![*collection_id],
                store.collection_referenced_shards(collection_id),
            ))
        }
        RepackTarget::Packs(pack_ids) => resolve_repack_packs(store, pack_ids),
        RepackTarget::All => {
            let slots = store
                .shard_summaries()
                .into_iter()
                .map(|summary| summary.slot)
                .collect::<Vec<_>>();
            resolve_repack_slots(store, &slots)
        }
    }
}

fn resolve_repack_packs(
    store: &PackfileStorage,
    pack_ids: &[u64],
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    let slots = pack_ids
        .iter()
        .map(|&pack_id| {
            store
                .shard_summaries()
                .into_iter()
                .find(|summary| summary.pack_id == pack_id)
                .map(|summary| summary.slot)
                .with_context(|| format!("pack ID 0x{pack_id:016x} not found"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    resolve_repack_slots(store, &slots)
}

fn resolve_repack_slots(
    store: &PackfileStorage,
    slots: &[u16],
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    let mut collections = std::collections::BTreeSet::new();
    let mut shards = std::collections::BTreeSet::new();
    for slot in slots {
        let (closure_collections, closure_shards) = store.repack_closure(*slot)?;
        collections.extend(closure_collections);
        shards.extend(closure_shards);
    }
    Ok((
        collections.into_iter().collect(),
        shards.into_iter().collect(),
    ))
}

fn repack_preview(
    cli: &Cli,
    target: &RepackTarget,
    topo: bool,
) -> anyhow::Result<Option<RepackPreview>> {
    println!("preflight: scanning packs and rebuilding live indexes...");
    let preview_store = open_store_read_only(cli)?;
    println!("preflight: resolving pack closure...");
    let (collections, shards) = resolve_repack_target(&preview_store, target)?;
    if collections.is_empty() {
        match target {
            RepackTarget::Packs(pack_ids) => {
                let pack_ids = pack_ids
                    .iter()
                    .map(|id| format!("0x{id:016x}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("no collections reference selected packs: {pack_ids}");
            }
            RepackTarget::All => println!("no collections found in active packs"),
            RepackTarget::Collection(_) => {}
        }
        return Ok(None);
    }

    let mut total_kept = 0usize;
    let mut total_dropped = 0usize;
    let mut total_dropped_bytes = 0u64;
    println!(
        "preflight: scanning {} pack{} for {} collection{}...",
        shards.len(),
        if shards.len() == 1 { "" } else { "s" },
        collections.len(),
        if collections.len() == 1 { "" } else { "s" },
    );
    let plans = if topo {
        preview_store.plan_collections_repack(&collections, extract_matrix_edges)?
    } else {
        preview_store.plan_collections_repack(&collections, |_hash, _data| Vec::new())?
    };
    for plan in plans {
        total_kept = total_kept.saturating_add(plan.kept);
        total_dropped = total_dropped.saturating_add(plan.dropped);
        total_dropped_bytes = total_dropped_bytes.saturating_add(plan.dropped_bytes);
    }

    let pack_summaries: std::collections::HashMap<u16, (u64, u64)> = preview_store
        .shard_summaries()
        .into_iter()
        .map(|summary| (summary.slot, (summary.pack_id, summary.file_bytes)))
        .collect();
    let pack_ids = shards
        .iter()
        .filter_map(|slot| pack_summaries.get(slot).map(|(pack_id, _)| *pack_id))
        .collect::<Vec<_>>();
    let total_input_bytes = shards.iter().fold(0u64, |total, slot| {
        total.saturating_add(pack_summaries.get(slot).map_or(0, |(_, bytes)| *bytes))
    });
    let pack_labels = pack_ids
        .iter()
        .map(|id| format!("0x{id:016x}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "this will repack {} collection{} across {} pack{} ({pack_labels})",
        collections.len(),
        if collections.len() == 1 { "" } else { "s" },
        shards.len(),
        if shards.len() == 1 { "" } else { "s" },
    );
    let label_width = "0x0000000000000000".len();
    for slot in &shards {
        let (pack_id, bytes) = pack_summaries.get(slot).copied().unwrap_or((0, 0));
        println!("  0x{pack_id:016x}: {:>9}", fmt_bytes(bytes));
    }
    println!(
        "  {:>label_width$}: {:>9}",
        "total",
        fmt_bytes(total_input_bytes)
    );
    println!(
        "expected result: {total_kept} nodes across {} collection{} rewritten",
        collections.len(),
        if collections.len() == 1 { "" } else { "s" }
    );
    println!(
        "                 {total_dropped} nodes / {} pruned",
        fmt_bytes(total_dropped_bytes),
    );
    // preview_store (and its non-exclusive read-only handle) drops
    // here, before the confirmation prompt — nothing about a
    // read-only open blocks a real writer anyway, but there's no
    // reason to keep it open through an indefinite human pause.
    Ok(Some(RepackPreview {
        collections,
        pack_ids,
    }))
}

fn repack_collections(
    store: &PackfileStorage,
    collections: &[[u8; 16]],
    topo: bool,
) -> anyhow::Result<(usize, usize)> {
    let pack_label = |slot| {
        store
            .shard_summaries()
            .into_iter()
            .find(|summary| summary.slot == slot)
            .map_or_else(
                || "retired pack".to_owned(),
                |summary| format!("0x{:016x}", summary.pack_id),
            )
    };
    let results = if topo {
        store.repack_collections_reachable_with_progress(
            collections,
            extract_matrix_edges,
            |collection_id, from, to, nodes, complete| {
                if complete {
                    println!("  copy complete: {nodes} nodes; output {} active", pack_label(to));
                } else {
                    let collection_id = collection_id.expect("pack rotation progress has a collection");
                    println!(
                        "  copy progress: {nodes} nodes; output rotated {} → {} while copying collection {}",
                        pack_label(from),
                        pack_label(to),
                        hex_encode(&collection_id),
                    );
                }
            },
        )?
    } else {
        store.repack_collections_reachable_with_progress(
            collections,
            |_hash, _data| Vec::new(),
            |collection_id, from, to, nodes, complete| {
                if complete {
                    println!("  copy complete: {nodes} nodes; output {} active", pack_label(to));
                } else {
                    let collection_id = collection_id.expect("pack rotation progress has a collection");
                    println!(
                        "  copy progress: {nodes} nodes; output rotated {} → {} while copying collection {}",
                        pack_label(from),
                        pack_label(to),
                        hex_encode(&collection_id),
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
        "done: {} collections repacked in one shard batch, {final_kept} kept, {final_dropped} dropped",
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
    let (collections, touched_shards) = resolve_repack_target(&store, target)?;
    if collections.is_empty() {
        println!("repack target is no longer referenced by any collection — nothing to do");
        return Ok(());
    }
    let grew = collections.iter().any(|r| !preview.collections.contains(r))
        || touched_shards.iter().any(|slot| {
            store
                .shard_summaries()
                .into_iter()
                .find(|summary| summary.slot == *slot)
                .map_or(true, |summary| !preview.pack_ids.contains(&summary.pack_id))
        });
    if grew {
        println!(
            "note: the closure grew since the preview (now {} collections / {} packs) — repacking the current, authoritative closure",
            collections.len(),
            touched_shards.len()
        );
    }

    repack_collections(&store, &collections, topo)?;
    // `cmd_shards` deliberately reads the persisted shard→collection directory
    // instead of reopening and scanning every packfile. Refresh it before
    // the immediate post-repack table so freshly-created packs have
    // real node counts rather than `?`.
    store.persist_shard_collections()?;
    drop(store);
    println!("post-repack shard state:");
    cmd_shards(cli, false, false, None)?;
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

fn cmd_delete(cli: &Cli, collections: &[String], yes: bool) -> anyhow::Result<()> {
    let mut collection_ids: Vec<[u8; 16]> = collections
        .iter()
        .map(|collection| parse_collection_id(collection))
        .collect::<anyhow::Result<_>>()?;
    collection_ids.sort_unstable();
    collection_ids.dedup();

    if !yes {
        println!(
            "This will permanently delete all data for {} collection(s):",
            collection_ids.len()
        );
        for collection_id in &collection_ids {
            println!("  {}", hex_encode(collection_id));
        }
        if !confirm("Delete these collections?")? {
            println!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    for collection_id in collection_ids {
        let count = store
            .collection_index_info(&collection_id)
            .map_or(0, |(len, _)| len);
        store.delete_collection(&collection_id)?;
        println!(
            "deleted {count} nodes for collection {}",
            hex_encode(&collection_id)
        );
    }
    Ok(())
}

/// Opens the store as writer (which always does a full scan and builds
/// the shard→collection directory and shard stats in memory regardless of
/// whether either has ever been persisted), then persists both —
/// bootstrapping `shard_stats.bin`/`shard_collections.bin` for a store whose
/// writer process has never called `sync_all`, or just refreshing them
/// on demand.
fn cmd_sync(cli: &Cli) -> anyhow::Result<()> {
    let store = open_store(cli)?;
    store.sync_all()?;
    eprintln!("synced: persisted shard IO stats and shard\u{2192}collection directory");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        avoidable_spread_bytes, event_room_id, fmt_disk_megabytes, fmt_megabytes,
        matrix_create_details, parse_pack_id_selector, parse_pack_selectors, physical_layout,
        CollectionPhysicalLayout,
    };
    use bytes::Bytes;
    use mtxdb_core::packfile::{write_header, write_record, Record};
    use simd_json::OwnedValue;
    use std::fs::File;

    fn owned_value(json: &str) -> OwnedValue {
        let mut bytes = json.as_bytes().to_vec();
        simd_json::to_owned_value(&mut bytes).expect("valid JSON fixture")
    }

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

    #[test]
    fn pack_selectors_accept_only_pack_ids_and_deduplicate() {
        assert_eq!(parse_pack_id_selector("0x3").unwrap(), 3);
        assert!(
            parse_pack_id_selector("3").is_err(),
            "decimal slots are not pack IDs"
        );
        assert!(
            parse_pack_id_selector("0x3-5").is_err(),
            "ranges are not pack IDs"
        );
        assert_eq!(
            parse_pack_selectors(&["0x3".to_owned(), "0x0002".to_owned(), "0x3".to_owned()])
                .unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn event_room_id_reads_the_real_matrix_wire_key() {
        let event = owned_value(r#"{"room_id": "!abc:example.org", "type": "m.room.message"}"#);
        assert_eq!(event_room_id(&event), Some("!abc:example.org"));
    }

    #[test]
    fn event_room_id_ignores_collection_id_and_missing_field() {
        // `collection_id` is mtxdb's own storage key, not a Matrix wire field —
        // event_room_id must not be fooled by it (regression for the room->collection rename).
        let event = owned_value(r#"{"collection_id": "deadbeef", "type": "m.room.message"}"#);
        assert_eq!(event_room_id(&event), None);

        assert_eq!(event_room_id(&owned_value("{}")), None);
        assert_eq!(event_room_id(&owned_value("42")), None);
        assert_eq!(
            event_room_id(&owned_value(r#"{"room_id": 42}"#)),
            None,
            "non-string room_id must not be coerced"
        );
    }

    #[test]
    fn matrix_create_details_matches_the_real_event_type() {
        let event = owned_value(
            r#"{
                "type": "m.room.create",
                "event_id": "$create:example.org",
                "sender": "@alice:example.org",
                "content": {"creator": "@alice:example.org", "room_version": "10"}
            }"#,
        );
        assert_eq!(
            matrix_create_details(&event).as_deref(),
            Some(
                "$create:example.org by @alice:example.org; creator @alice:example.org; room version 10"
            )
        );
    }

    #[test]
    fn matrix_create_details_ignores_non_create_events() {
        // A stray "m.collection.create" (a leftover from a bad room->collection rename)
        // must never match — only the real Matrix wire event type does.
        let wrong_type = owned_value(r#"{"type": "m.collection.create"}"#);
        assert_eq!(matrix_create_details(&wrong_type), None);

        let message = owned_value(r#"{"type": "m.room.message"}"#);
        assert_eq!(matrix_create_details(&message), None);
    }

    #[test]
    fn matrix_create_details_tolerates_missing_optional_fields() {
        let event = owned_value(r#"{"type": "m.room.create"}"#);
        assert_eq!(
            matrix_create_details(&event).as_deref(),
            Some("<missing event_id> by <missing sender>")
        );
    }

    #[test]
    fn physical_layout_counts_cross_pack_spread_and_interleaved_runs() {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_cli_layout_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let collection_a = [0xA1; 16];
        let collection_b = [0xB2; 16];
        for (pack_id, records) in [
            (0_u64, vec![collection_a, collection_b, collection_a]),
            (1_u64, vec![collection_a]),
        ] {
            let path = dir.join(format!("pack_{pack_id:016x}.pack"));
            let mut file = File::create(path).unwrap();
            write_header(&mut file, pack_id).unwrap();
            for (index, collection_id) in records.into_iter().enumerate() {
                write_record(
                    &mut file,
                    &Record {
                        collection_id,
                        hash: [u8::try_from(index).expect("fixture index fits in u8"); 16],
                        data: Bytes::from_static(b"payload"),
                    },
                )
                .unwrap();
            }
        }

        let layout = physical_layout(&dir).unwrap();
        let a = &layout.collections[&collection_a];
        assert_eq!(a.pack_bytes.len(), 2, "A spans two packs");
        assert_eq!(a.segments, 3, "A-B-A in pack 0 plus A in pack 1");
        assert_eq!(layout.packs[&0].segments, 3);
        assert_eq!(layout.packs[&1].segments, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn avoidable_spread_excludes_a_collections_required_spill() {
        let capacity = mtxdb_core::shard::MAX_SHARD_BYTES;
        let mut ideal = CollectionPhysicalLayout {
            disk_bytes: capacity.saturating_add(100),
            ..CollectionPhysicalLayout::default()
        };
        ideal.pack_bytes.insert(0, capacity);
        ideal.pack_bytes.insert(1, 100);
        assert_eq!(avoidable_spread_bytes(Some(&ideal)), 0);

        let mut fragmented = ideal;
        fragmented.pack_bytes.clear();
        fragmented.pack_bytes.insert(0, capacity / 2);
        fragmented.pack_bytes.insert(1, capacity / 2 + 100);
        assert_eq!(
            avoidable_spread_bytes(Some(&fragmented)),
            capacity.saturating_sub(capacity / 2 + 100)
        );
    }
}
