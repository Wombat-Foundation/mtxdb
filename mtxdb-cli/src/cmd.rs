use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use sha2::{Digest, Sha256};

use mtxdb::packfile::layout::{avoidable_spread_bytes, physical_layout, CollectionPhysicalLayout};
use mtxdb::packfile::storage::{OpenPath, RuntimeStats};
use mtxdb::shard::ShardPool;
use mtxdb::storage::{NodeData, StorageEngine};
use mtxdb::{
    derive_collection_id, frame_digest, CollectionKeyRule, CollectionTemplate, DatabaseLayout,
    DigestAlgorithm, FrameIdInput, FrameIdPolicy, PackfileStorage, PayloadPolicy,
    RecordIdentityRule, ShardType, COLLECTION_TYPE_PROTOCOL_BASE,
};
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

/// A collection index's occupancy as a percentage of its slot-table
/// capacity — how full the hash table is, not how full the collection's
/// storage is. The table rejects new inserts (forcing a grow or rebuild)
/// once this crosses 75%, so a figure already near or at that mark
/// explains why a collection's index just grew or is about to.
fn fmt_load_factor(len: usize, capacity: u32) -> String {
    if capacity == 0 {
        return "load factor: n/a".to_owned();
    }
    format!(
        "load factor: {:.1}% ({len}/{capacity})",
        load_factor_percent(len, capacity)
    )
}

/// Numeric index occupancy percentage, for sorting `collections` by load.
/// Returns `0.0` for a zero/unknown capacity so a sort never panics.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn load_factor_percent(len: usize, capacity: u32) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        (len as f64 / f64::from(capacity)) * 100.0
    }
}

/// Column-friendly index occupancy ("24.2%"), for `collections` tables where
/// `fmt_load_factor`'s long form would not fit a fixed-width column.
fn fmt_load_percent(len: usize, capacity: u32) -> String {
    if capacity == 0 {
        return "n/a".to_owned();
    }
    format!("{:.1}%", load_factor_percent(len, capacity))
}

/// Decimal kilobytes for per-collection index allocations, where MB would
/// obscure the useful differences between small power-of-two tables.
fn fmt_index_kilobytes(bytes: usize) -> String {
    const BYTES_PER_KB: u64 = 1_000;
    const FRACTION_SCALE: u64 = 100;

    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    let mut whole = bytes / BYTES_PER_KB;
    let mut fraction = (bytes % BYTES_PER_KB)
        .saturating_mul(FRACTION_SCALE)
        .saturating_add(BYTES_PER_KB / 2)
        / BYTES_PER_KB;
    if fraction == FRACTION_SCALE {
        whole = whole.saturating_add(1);
        fraction = 0;
    }
    format!("{whole}.{fraction:02} KB")
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
        Commands::Get {
            collection,
            id,
            raw,
        } => cmd_get(cli, collection.as_deref(), id, *raw),
        Commands::Collections {
            all,
            layout,
            sort,
            limit,
        } => cmd_collections(cli, *all, *layout, sort.as_deref(), *limit),
        Commands::Shards { all, layout, sort } => cmd_shards(cli, *all, *layout, sort.as_deref()),
        Commands::Stats { json } => cmd_stats(cli, *json),
        Commands::Info { collection } => cmd_info(cli, collection),
        Commands::Scan {
            selector,
            verbose,
            limit,
            id,
            collection,
            raw,
            sort,
            reverse,
        } => cmd_scan(
            cli,
            selector,
            *verbose,
            *limit,
            id.as_deref(),
            collection.as_deref(),
            *raw,
            sort.as_deref(),
            *reverse,
        ),
        Commands::Import {
            paths,
            collection,
            template,
        } => cmd_import(cli, paths, collection.as_deref(), template.as_deref()),
        Commands::Export { collection } => cmd_export(cli, collection),
        Commands::Repack {
            collection,
            packs,
            all,
            root,
            topo,
        } => cmd_repack(cli, collection.as_deref(), packs, *all, root, *topo),
        Commands::Delete { collections, yes } => cmd_delete(cli, collections, *yes),
        Commands::Completions { .. } => unreachable!("main emits completion scripts directly"),
        Commands::Sync { all } => cmd_sync(cli, *all),
        Commands::Init => cmd_init(cli),
        Commands::SubprocessWriter { path } | Commands::SubprocessWriterAppend { path } => {
            cmd_subprocess_writer(path, true, false)
        }
        Commands::SubprocessWriterUnsynced { path } => cmd_subprocess_writer(path, false, false),
        Commands::SubprocessWriterSeed { path } => cmd_subprocess_writer(path, true, true),
        Commands::SubprocessReader { path } => cmd_subprocess_reader(path),
    }
}

const SUBPROCESS_COLLECTION: [u8; 16] = [0x01; 16];
const SUBPROCESS_RECORD: [u8; 16] = [0xD0; 16];
const SUBPROCESS_SEED: [u8; 16] = [0xC0; 16];

fn cmd_subprocess_writer(path: &str, sync: bool, seed: bool) -> anyhow::Result<()> {
    let store = PackfileStorage::open(PathBuf::from(path)).context("failed to open test store")?;
    let id = if seed {
        SUBPROCESS_SEED
    } else {
        SUBPROCESS_RECORD
    };
    let payload: &[u8] = if seed { b"seed" } else { b"synced" };
    let data = NodeData::new(bytes::Bytes::from_static(payload));
    store
        .put(&SUBPROCESS_COLLECTION, &id, &data)
        .context("failed to write test record")?;
    if sync {
        store.sync().context("failed to sync test record")?;
    }
    Ok(())
}

fn cmd_subprocess_reader(path: &str) -> anyhow::Result<()> {
    let store = PackfileStorage::open_read_only(PathBuf::from(path))
        .context("failed to open test store read-only")?;
    store.reset_stats();
    let result = store
        .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
        .context("failed to read test record")?;
    let stats = store.stats();
    if result.first().and_then(Option::as_ref).is_some() {
        println!("RECOVERED");
    } else {
        println!("NOT_FOUND");
    }
    println!("miss_refreshes={}", stats.miss_refreshes);
    Ok(())
}

/// Create a new mtxdb database root: `db.meta` plus an empty directory for
/// every independent shard pool. The only command allowed to bring a store
/// into existence -- every other command resolves the root read-only (see
/// `open_layout`) and errors instead of creating one on the fly.
fn cmd_init(cli: &Cli) -> anyhow::Result<()> {
    let root = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    let already_initialized = root.join("db.meta").is_file();
    DatabaseLayout::open(root.into()).with_context(|| {
        format!(
            "failed to initialize mtxdb database root `{}`",
            root.display()
        )
    })?;
    if already_initialized {
        println!("mtxdb database already initialized at `{}`", root.display());
    } else {
        println!("initialized mtxdb database at `{}`", root.display());
        for shard_type in ShardType::ALL {
            println!("  pools/{}", shard_type.as_str());
        }
    }
    Ok(())
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

/// Resolve either the fixed-width storage key or a Matrix event ID. `$id`
/// is hashed exactly the way Synapse's embedded mirror derives it (see
/// `synapse_event_node_id`), since the point of this sigil is finding data
/// Synapse itself wrote — not CLI-imported data, which uses the import
/// template's own (SHA-256) identity rule instead (`matrix_event_node_id`).
fn parse_get_id(id: &str, namespace: Option<&str>) -> anyhow::Result<[u8; 16]> {
    if let Some(event_id) = id.strip_prefix('$') {
        let namespace = namespace.context(
            "resolving `$event_id` requires --namespace (or MTXDB_NAMESPACE) set to the same \
             namespace Synapse's embedded mirror was configured with",
        )?;
        Ok(synapse_event_node_id(namespace, event_id))
    } else {
        parse_node_id(id)
    }
}

/// Collection type discriminator for Matrix room collections, mixed into
/// [`derive_collection_id`]. Matrix is a protocol extension, so it draws from
/// the protocol-owned range rather than a core-internal value.
const MATRIX_ROOM_COLLECTION_TYPE: u16 = COLLECTION_TYPE_PROTOCOL_BASE;

/// The Matrix import template's accepted identity algorithm: SHA-256 truncated
/// to the 128-bit node ID used by the packfile index. Repack edge extraction,
/// `--root`, and CLI-imported collections use this same derivation — distinct
/// from `synapse_event_node_id`, which matches Synapse's own live mirror.
fn matrix_event_node_id(event_id: &str) -> anyhow::Result<[u8; 16]> {
    derive_template_key("sha2-256", event_id)
}

/// Matches `event_node_id` in Synapse's `rust/src/database/mtxdb.rs` byte
/// for byte: the first 16 bytes of
/// `SHA-256("event_json:event:" + namespace + "\0" + event_id)`. `$id`
/// lookups only find anything if this derivation is identical to Synapse's.
fn synapse_event_node_id(namespace: &str, event_id: &str) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"event_json:event:");
    hasher.update(namespace.as_bytes());
    hasher.update(b"\0");
    hasher.update(event_id.as_bytes());
    let hash = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&hash[..16]);
    id
}

/// Matches `event_dag_room_id` in Synapse's `rust/src/database/mtxdb.rs`:
/// the room-scoped `EventDag` collection ID that `!room_id` resolves to. The
/// first 16 bytes of `SHA-256("event_json:dag:" + namespace + "\0" + room_id)`.
fn matrix_room_collection_id(namespace: &str, room_id: &str) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"event_json:dag:");
    hasher.update(namespace.as_bytes());
    hasher.update(b"\0");
    hasher.update(room_id.as_bytes());
    let hash = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&hash[..16]);
    id
}

/// Resolve either the fixed-width collection ID or a `!room_id`, the latter
/// hashed the same way Synapse's embedded mirror derives a room's `EventDag`
/// collection (see `matrix_room_collection_id`).
fn parse_collection_selector(selector: &str, namespace: Option<&str>) -> anyhow::Result<[u8; 16]> {
    if let Some(room_id) = selector.strip_prefix('!') {
        let namespace = namespace.context(
            "resolving `!room_id` requires --namespace (or MTXDB_NAMESPACE) set to the same \
             namespace Synapse's embedded mirror was configured with",
        )?;
        Ok(matrix_room_collection_id(namespace, room_id))
    } else {
        parse_collection_id(selector)
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
///
/// Every command except `init` resolves the database root read-only: this
/// never creates `db.meta` or any pool directory as a side effect of simply
/// pointing the CLI at a path (see `cmd_init`). A missing root is a clear
/// error pointing at `mtxdb init`, not silent on-disk state.
fn open_layout(cli: &Cli) -> anyhow::Result<DatabaseLayout> {
    let root = cli.dir.as_deref().unwrap_or_else(|| Path::new("."));
    DatabaseLayout::open_read_only(root.into()).with_context(|| {
        format!(
            "no mtxdb database at `{}` -- run `mtxdb init` first",
            root.display()
        )
    })
}

fn pool_dir(layout: &DatabaseLayout, shard_type: ShardType) -> anyhow::Result<PathBuf> {
    layout
        .pool_dir_read_only(shard_type)
        .with_context(|| format!("failed to open {} shard pool", shard_type.as_str()))
}

fn selected_pool_dir(cli: &Cli) -> anyhow::Result<PathBuf> {
    pool_dir(&open_layout(cli)?, cli.require_shard_type()?)
}

/// A collection ID not found under `cli.shard_type` is often just in one of
/// the *other* independent shard-type pools (`-t`/`--shard-type` defaults
/// to `event-dag`, so a `state`-only collection ID "not found" there is the
/// single most common `scan`/`info` support question). Cheaply checks each
/// other pool's persisted sidecar (no full-scan fallback, no index rebuild)
/// and, if any has it, returns a one-line hint naming them; `String::new()`
/// otherwise, so callers can append it to a "not found" message unconditionally.
fn other_shard_type_hint(cli: &Cli, collection_id: &[u8; 16]) -> String {
    let Ok(layout) = open_layout(cli) else {
        return String::new();
    };
    let Some(current) = cli.shard_type else {
        return String::new();
    };
    let found: Vec<&str> = ShardType::ALL
        .into_iter()
        .filter(|&shard_type| shard_type != current)
        .filter_map(|shard_type| {
            pool_dir(&layout, shard_type)
                .ok()
                .map(|dir| (shard_type, dir))
        })
        .filter(|(_, dir)| {
            PackfileStorage::collection_shards_from_disk(dir)
                .is_some_and(|shards| shards.contains_key(collection_id))
        })
        .map(|(shard_type, _)| shard_type.as_str())
        .collect();
    if found.is_empty() {
        String::new()
    } else {
        format!(
            " (not in -t {}; found in -t {})",
            current.as_str(),
            found.join(", -t ")
        )
    }
}

/// Best-effort hint for a node ID found in another independent shard pool.
/// Unlike collection lookup, this must probe the selected pools' indexes
/// because `get` has no collection ID from which to use the sidecar.
fn other_shard_type_node_hint(cli: &Cli, node_id: &[u8; 16]) -> String {
    let Ok(layout) = open_layout(cli) else {
        return String::new();
    };
    let Some(current) = cli.shard_type else {
        return String::new();
    };
    let found: Vec<&str> = ShardType::ALL
        .into_iter()
        .filter(|&shard_type| shard_type != current)
        .filter(|&shard_type| {
            let Ok(dir) = pool_dir(&layout, shard_type) else {
                return false;
            };
            let Ok(store) = PackfileStorage::open_read_only(dir) else {
                return false;
            };
            store
                .collection_summaries()
                .into_iter()
                .any(|summary| matches!(store.get(&summary.0, node_id), Ok(Some(_))))
        })
        .map(ShardType::as_str)
        .collect();
    if found.is_empty() {
        String::new()
    } else {
        format!(
            " (not in -t {}; found in -t {})",
            current.as_str(),
            found.join(", -t ")
        )
    }
}

fn cmd_put(cli: &Cli, collection: &str, id: &str, data: &str) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let node_id = parse_node_id(id)?;
    let store = open_store(cli)?;
    let node_data = NodeData::new(bytes::Bytes::from(data.as_bytes().to_vec()));
    store.put(&collection_id, &node_id, &node_data)?;
    store.sync()?;
    let collection_hex = hex_encode(&collection_id);
    let id_hex = hex_encode(&node_id);
    eprintln!(
        "put {id_hex} into collection {collection_hex} ({} bytes)",
        data.len()
    );
    Ok(())
}

fn emit_get_data(data: &NodeData, raw: bool) -> anyhow::Result<()> {
    let emitted = if raw {
        data.bytes.to_vec()
    } else if let Some(rendered) = pretty_print_payload(&data.bytes) {
        rendered
    } else {
        hex_bytes(&data.bytes).into_bytes()
    };
    io::stdout().write_all(&emitted)?;
    // Raw output stays byte-exact. All textual output is
    // newline-terminated, including encoded binary fallbacks.
    if !raw && !emitted.ends_with(b"\n") {
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

fn get_matches_in_store(
    store: &PackfileStorage,
    collection: Option<&str>,
    node_id: &[u8; 16],
    namespace: Option<&str>,
) -> anyhow::Result<Vec<([u8; 16], NodeData)>> {
    match collection {
        Some(collection) => {
            let collection_id = parse_collection_selector(collection, namespace)?;
            Ok(store
                .get(&collection_id, node_id)?
                .map(|data| vec![(collection_id, data)])
                .unwrap_or_default())
        }
        None => store
            .collection_summaries()
            .into_iter()
            .map(|s| s.0)
            .filter_map(|collection_id| match store.get(&collection_id, node_id) {
                Ok(Some(data)) => Some(Ok((collection_id, data))),
                Ok(None) => None,
                Err(error) => Some(Err(anyhow::Error::from(error))),
            })
            .collect(),
    }
}

fn cmd_get(cli: &Cli, collection: Option<&str>, id: &str, raw: bool) -> anyhow::Result<()> {
    let node_id = parse_get_id(id, cli.namespace.as_deref())?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matches: Vec<(ShardType, [u8; 16], NodeData)> = Vec::new();
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let Ok(store) = PackfileStorage::open_read_only(dir) else {
                continue;
            };
            for (col_id, data) in
                get_matches_in_store(&store, collection, &node_id, cli.namespace.as_deref())?
            {
                matches.push((shard_type, col_id, data));
            }
        }
        match matches.as_slice() {
            [] => bail!("not found"),
            [(_, _, data)] => emit_get_data(data, raw),
            _ => {
                let locations = matches
                    .iter()
                    .map(|(shard_type, col_id, _)| {
                        format!(
                            "-t {} collection {}",
                            shard_type.as_str(),
                            hex_encode(col_id)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "node ID {id} is present in multiple pools/collections ({locations}); specify -t and/or --collection"
                );
            }
        }
    } else {
        let store = open_store_read_only(cli)?;
        let matches = get_matches_in_store(&store, collection, &node_id, cli.namespace.as_deref())?;
        match matches.as_slice() {
            [] => bail!("not found{}", other_shard_type_node_hint(cli, &node_id)),
            [(_, data)] => emit_get_data(data, raw),
            _ => bail!(
                "node ID {id} is present in multiple collections ({}); specify --collection",
                matches
                    .iter()
                    .map(|(collection_id, _)| hex_encode(collection_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// Render arbitrary payload bytes safely for terminal output. Raw bytes stay
/// available through `get --raw`; the default uses a single hexadecimal token.
fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(2usize.saturating_add(bytes.len().saturating_mul(2)));
    output.push_str("0x");
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

/// Pretty-print a stream of JSON objects or arrays, returning `None` for a
/// non-JSON payload.
fn pretty_json_stream(bytes: &[u8]) -> Option<Vec<u8>> {
    let values = split_json_stream(bytes)?;
    let mut output = Vec::new();
    for value in values {
        let mut value = value.to_vec();
        let json = simd_json::to_owned_value(&mut value).ok()?;
        let formatted = json.encode_pp();
        output.extend_from_slice(formatted.as_bytes());
        output.push(b'\n');
    }
    Some(output)
}

/// Splits adjacent top-level JSON objects/arrays while respecting strings and
/// escapes. Matrix payloads are objects, but arrays are equally valid JSON
/// documents and cheap to support here.
fn split_json_stream(bytes: &[u8]) -> Option<Vec<&[u8]>> {
    let mut values = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start = start.saturating_add(1);
        }
        if start == bytes.len() {
            break;
        }
        let opener = *bytes.get(start)?;
        if !matches!(opener, b'{' | b'[') {
            return None;
        }
        let mut depth = 0_u32;
        let mut quoted = false;
        let mut escaped = false;
        let mut end = start;
        for (offset, byte) in bytes[start..].iter().copied().enumerate() {
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
                continue;
            }
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => depth = depth.checked_add(1)?,
                b'}' | b']' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        end = start.checked_add(offset)?.checked_add(1)?;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end == start || quoted {
            return None;
        }
        values.push(&bytes[start..end]);
        start = end;
    }
    (!values.is_empty()).then_some(values)
}

/// Decode Synapse's `event_json`-mirror record layout: a fixed 8-byte binary
/// header (big-endian `i32` `format_version`, big-endian `u32` length of
/// `internal_metadata`) followed by the `internal_metadata` JSON string and
/// then the PDU JSON string, with no delimiter between the two. mtxdb
/// stores this as an opaque blob — it has no concept of the layout — so a
/// plain JSON parse of the whole value fails; this recognizes that specific
/// shape instead of leaving it as a "(non-JSON)" dead end.
///
/// Returns `None` for anything that doesn't fit the shape (including
/// ordinary single-JSON-document payloads, which callers should try first),
/// so this never mis-decodes unrelated record types.
fn decode_event_json_record(data: &[u8]) -> Option<(i32, Vec<u8>, Vec<u8>)> {
    const HEADER_LEN: usize = 8;
    if data.len() < HEADER_LEN {
        return None;
    }
    let format_version = i32::from_be_bytes(data[0..4].try_into().ok()?);
    let metadata_len = usize::try_from(u32::from_be_bytes(data[4..8].try_into().ok()?)).ok()?;
    let metadata_end = HEADER_LEN.checked_add(metadata_len)?;
    let metadata_bytes = data.get(HEADER_LEN..metadata_end)?;
    let json_bytes = data.get(metadata_end..)?;
    // Both fields must themselves be complete, single JSON documents with
    // nothing left over — anything else is not this record shape.
    let metadata_pretty = single_json_document(metadata_bytes)?;
    let json_pretty = single_json_document(json_bytes)?;
    Some((format_version, metadata_pretty, json_pretty))
}

/// Like [`pretty_json_stream`], but requires the bytes to be exactly one
/// JSON document with nothing trailing — used to validate a slice carved out
/// of a larger binary record, where leftover bytes mean the slice boundary
/// was wrong, not that formatting should stop early.
fn single_json_document(bytes: &[u8]) -> Option<Vec<u8>> {
    let values = split_json_stream(bytes)?;
    let [value] = values.as_slice() else {
        return None;
    };
    let mut value = value.to_vec();
    let json = simd_json::to_owned_value(&mut value).ok()?;
    Some(json.encode_pp().into_bytes())
}

/// Decode a Synapse state-group HAMT root and pretty-print its structure.
///
/// Wire format (`MTHR`, version 1): a big-endian `u16` room-prefix length,
/// the room-prefix bytes, a big-endian `u16` room-ID length, the UTF-8 room
/// ID, a 32-byte root hash, and a 2048-byte lattice of 1024 big-endian lanes.
/// Returns `None` when the bytes don't match this layout.
fn decode_hamt_root(bytes: &[u8]) -> Option<Vec<u8>> {
    const MAGIC: &[u8; 4] = b"MTHR";
    const ROOT_HASH_LEN: usize = 32;
    const LATTICE_LEN: usize = 2048;

    if bytes.get(..MAGIC.len())? != MAGIC.as_slice() || bytes.len() < 7 {
        return None;
    }
    if bytes.get(MAGIC.len()).copied()? != 0x01 {
        return None;
    }
    let prefix_len = u16::from_be_bytes(bytes[5..7].try_into().ok()?) as usize;
    let prefix_start = 7usize;
    let room_id_len_offset = prefix_start.checked_add(prefix_len)?;
    let room_id_len_end = room_id_len_offset.checked_add(2)?;
    let room_id_len = u16::from_be_bytes(
        bytes
            .get(room_id_len_offset..room_id_len_end)?
            .try_into()
            .ok()?,
    ) as usize;
    let room_id_start = room_id_len_end;
    let root_hash_start = room_id_start.checked_add(room_id_len)?;
    let lattice_start = root_hash_start.checked_add(ROOT_HASH_LEN)?;
    let end = lattice_start.checked_add(LATTICE_LEN)?;
    if bytes.len() != end {
        return None;
    }
    let room_id = core::str::from_utf8(bytes.get(room_id_start..root_hash_start)?).ok()?;
    let room_prefix = bytes.get(prefix_start..room_id_len_offset)?;
    let root_hash = bytes.get(root_hash_start..lattice_start)?;
    let mut lattice_hasher = Sha256::new();
    lattice_hasher.update(bytes.get(lattice_start..end)?);
    let lattice_digest = lattice_hasher.finalize();

    let mut out = Vec::new();
    writeln!(out, "// HAMT state-group root (Synapse wire v1)").unwrap();
    writeln!(out, "// room prefix: 0x{}", hex::encode(room_prefix)).unwrap();
    writeln!(out, "// room ID: {room_id:?}").unwrap();
    writeln!(out, "// root hash: {}...", hex::encode(&root_hash[..8])).unwrap();
    writeln!(out, "// lattice: {LATTICE_LEN} bytes (1024 u16 lanes)").unwrap();
    writeln!(
        out,
        "// lattice digest (SHA-256): {}",
        hex::encode(lattice_digest)
    )
    .unwrap();
    writeln!(out, "// total payload: {} bytes", bytes.len()).unwrap();
    Some(out)
}

/// Decode a rezzy-format CHAMP HAMT node and pretty-print its structure.
///
/// Wire format (32-byte structural hashes):
/// ```text
/// [0..4]    magic (`MTHN`)
/// [4]       wire version (0x01)
/// [5..9]    datamap (u32 LE) -- bitmap of leaf slots
/// [9..13]   nodemap (u32 LE) -- bitmap of child slots
/// [13..17]  leaf_count (u32 LE)
/// [17..21]  child_count (u32 LE)
/// [21..]    inline leaves (K,V pairs in datamap bit order)
///           Each leaf: two length-prefixed UTF-8 strings
///           K = serde_json::to_string(&(EventType, StateKey))
///               (a JSON 2-element array, e.g. `["m.room.member","@a:b"]`)
///           V = EventId, as a plain string
/// [...]     child hashes (child_count x 32 bytes in nodemap bit order)
/// ```
///
/// Returns `None` when the bytes don't match this layout.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_lines)]
fn decode_hamt_node(bytes: &[u8]) -> Option<Vec<u8>> {
    const MAGIC: &[u8; 4] = b"MTHN";
    const WIRE_V1: u8 = 0x01;
    const HEADER_LEN: usize = 21;
    const HASH_LEN: usize = 32;

    // Synapse currently persists rezzy's wire-v1 layout. Its structural
    // hashes are 32 bytes; the record kind and codec version are separate.
    if bytes.get(..MAGIC.len())? != MAGIC.as_slice() || bytes.get(4).copied()? != WIRE_V1 {
        return None;
    }
    if bytes.len() < HEADER_LEN {
        return None;
    }

    let datamap = u32::from_le_bytes(bytes[5..9].try_into().ok()?);
    let nodemap = u32::from_le_bytes(bytes[9..13].try_into().ok()?);
    if (datamap & nodemap) != 0 {
        return None;
    }
    let leaf_count = u32::from_le_bytes(bytes[13..17].try_into().ok()?) as usize;
    let child_count = u32::from_le_bytes(bytes[17..21].try_into().ok()?) as usize;
    let expected_leaves = datamap.count_ones() as usize;
    let expected_children = nodemap.count_ones() as usize;
    if leaf_count != expected_leaves || child_count != expected_children {
        return None;
    }

    // Walk leaves to find where the child hash region starts.
    let child_region_len = child_count * HASH_LEN;
    let mut cursor = HEADER_LEN;
    for _ in 0..leaf_count {
        if !skip_hamt_leaf_value(bytes, &mut cursor) {
            return None;
        }
    }
    let leaf_end = cursor;
    if bytes.len() != leaf_end + child_region_len {
        return None;
    }

    // Validation passed.  Now decode for display -- if any leaf fails to
    // decode as the expected two-string pair, reject the entire node
    // rather than returning a partial rendering.
    let mut leaves = Vec::with_capacity(leaf_count);
    let mut leaf_cursor = HEADER_LEN;
    for slot in 0..32u32 {
        if datamap & (1 << slot) == 0 {
            continue;
        }
        let pair = decode_hamt_leaf_pair(bytes, &mut leaf_cursor)?;
        leaves.push((slot, pair));
    }

    // All leaves decoded.  Build output.
    let mut out = Vec::new();
    writeln!(out, "// HAMT CHAMP node (rezzy wire v1)").unwrap();
    writeln!(
        out,
        "// datamap: {datamap:#010x} ({} leaves), nodemap: {nodemap:#010x} ({} children)",
        datamap.count_ones(),
        nodemap.count_ones(),
    )
    .unwrap();
    writeln!(out, "// total payload: {} bytes", bytes.len()).unwrap();

    for (leaf_index, (slot, (event_type, state_key, event_id))) in leaves.iter().enumerate() {
        writeln!(
            out,
            "  leaf[{leaf_index}] slot={slot}: ({event_type:?}, {state_key:?}) = {event_id:?}"
        )
        .unwrap();
    }

    let mut child_index = 0;
    for slot in 0..32u32 {
        if nodemap & (1 << slot) == 0 {
            continue;
        }
        let start = leaf_end + child_index * HASH_LEN;
        let hash = &bytes[start..start + HASH_LEN];
        writeln!(
            out,
            "  child[{child_index}] slot={slot}: {}...",
            hex::encode(&hash[..8])
        )
        .unwrap();
        child_index += 1;
    }

    Some(out)
}

/// Try to skip a HAMT leaf encoded as two length-prefixed strings: K (a
/// JSON-encoded `(EventType, StateKey)` array) and V (`EventId`). Advances
/// `cursor` past the leaf on success. This matches
/// `synapse/rust/src/state_hamt.rs`'s `HamtNode<String, String>`, where K is
/// `serde_json::to_string(&(event_type, state_key))`, not a raw tuple
/// codec -- there are exactly two length-prefixed strings per leaf on disk.
#[allow(clippy::arithmetic_side_effects)]
fn skip_hamt_leaf_value(bytes: &[u8], cursor: &mut usize) -> bool {
    let start = *cursor;
    // K: JSON-encoded (EventType, StateKey), as one length-prefixed string.
    if !skip_length_prefixed_string(bytes, cursor) {
        return false;
    }
    // V = EventId: length-prefixed UTF-8.
    if !skip_length_prefixed_string(bytes, cursor) {
        *cursor = start;
        return false;
    }
    true
}

/// Try to skip a `u32 LE length` + `length bytes` string.
#[allow(clippy::arithmetic_side_effects)]
fn skip_length_prefixed_string(bytes: &[u8], cursor: &mut usize) -> bool {
    let start = *cursor;
    if bytes.len() < start + 4 {
        return false;
    }
    let len = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
    let end = start + 4 + len;
    if end > bytes.len() {
        return false;
    }
    *cursor = end;
    true
}

/// Decode a HAMT leaf's `(EventType, StateKey, EventId)` from its on-disk
/// two-string encoding: K is a length-prefixed JSON 2-element array
/// (`["event.type","state key"]`), V is a length-prefixed plain string
/// (`EventId`). Restores the cursor if decoding fails partway through.
fn decode_hamt_leaf_pair(bytes: &[u8], cursor: &mut usize) -> Option<(String, String, String)> {
    let start = *cursor;
    let key_json = decode_length_prefixed_string(bytes, cursor)?;
    let Some(event_id) = decode_length_prefixed_string(bytes, cursor) else {
        *cursor = start;
        return None;
    };
    let mut key_bytes = key_json.into_bytes();
    let Ok(value) = simd_json::to_owned_value(&mut key_bytes) else {
        *cursor = start;
        return None;
    };
    let simd_json::OwnedValue::Array(elements) = value else {
        *cursor = start;
        return None;
    };
    let [event_type, state_key] = elements.as_slice() else {
        *cursor = start;
        return None;
    };
    let (Some(event_type), Some(state_key)) = (event_type.as_str(), state_key.as_str()) else {
        *cursor = start;
        return None;
    };
    Some((event_type.to_owned(), state_key.to_owned(), event_id))
}

/// Decode a `u32 LE length` + `length bytes` UTF-8 string.
#[allow(clippy::arithmetic_side_effects)]
fn decode_length_prefixed_string(bytes: &[u8], cursor: &mut usize) -> Option<String> {
    let start = *cursor;
    if bytes.len() < start + 4 {
        return None;
    }
    let len = u32::from_le_bytes(bytes[start..start + 4].try_into().ok()?) as usize;
    let end = start + 4 + len;
    if end > bytes.len() {
        return None;
    }
    let s = core::str::from_utf8(&bytes[start + 4..end]).ok()?;
    *cursor = end;
    Some(s.to_owned())
}

/// Pretty-print a record payload for display, recognizing known mtxdb/Synapse
/// record shapes beyond plain JSON. Tries a plain JSON stream first (the
/// common case), then the `event_json` mirror layout, then the rezzy CHAMP
/// HAMT node wire format. Returns `None` when nothing recognized the bytes,
/// so callers can fall back to a raw/binary display.
fn pretty_print_payload(bytes: &[u8]) -> Option<Vec<u8>> {
    if let Some(plain) = pretty_json_stream(bytes) {
        return Some(plain);
    }
    if let Some(root) = decode_hamt_root(bytes) {
        return Some(root);
    }
    if let Some(hamt) = decode_hamt_node(bytes) {
        return Some(hamt);
    }
    let (format_version, metadata, json) = decode_event_json_record(bytes)?;
    let mut output = Vec::new();
    output.extend_from_slice(
        format!("// event_json record (format_version={format_version})\n// internal_metadata:\n")
            .as_bytes(),
    );
    output.extend_from_slice(&metadata);
    output.extend_from_slice(b"\n// json:\n");
    output.extend_from_slice(&json);
    output.push(b'\n');
    Some(output)
}

/// Enumerate live collections from the persisted directory. Stores created before
/// that sidecar existed fall back to a one-time index rebuild; `mtxdb sync`
/// makes future calls fast.
fn collection_ids_in_dir(dir: &Path) -> anyhow::Result<Vec<[u8; 16]>> {
    if PackfileStorage::collection_directory_persisted_at(dir).is_some() {
        return Ok(PackfileStorage::collection_directory_from_disk(dir)
            .into_iter()
            .map(|(collection_id, _)| collection_id)
            .collect());
    }
    let store =
        PackfileStorage::open_read_only(dir.to_path_buf()).context("failed to open store")?;
    Ok(store
        .collection_summaries()
        .into_iter()
        .map(|(collection_id, _, _, _)| collection_id)
        .collect())
}

fn collection_ids(cli: &Cli) -> anyhow::Result<Vec<[u8; 16]>> {
    collection_ids_in_dir(&selected_pool_dir(cli)?)
}

fn cmd_collections(
    cli: &Cli,
    all: bool,
    layout: bool,
    sort: Option<&str>,
    limit: i64,
) -> anyhow::Result<()> {
    // `--all` explicitly asks for every pool; `-t all` (no specific shard
    // type selected) means the same thing — without this, `-t all` fell
    // through to `selected_pool_dir`'s `require_shard_type`, which always
    // rejects an unselected type, even though `-t all` completes and parses
    // as a legitimate value.
    if all || cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let types: Vec<ShardType> = cli.shard_types().collect();
        for (index, shard_type) in types.into_iter().enumerate() {
            if index != 0 {
                println!();
                println!();
            }
            print_section_header(shard_type);
            cmd_collections_in_dir(&pool_dir(&db_layout, shard_type)?, layout, sort, limit)?;
        }
        return Ok(());
    }
    cmd_collections_in_dir(&selected_pool_dir(cli)?, layout, sort, limit)
}

/// List logical collections from one pool. Cross-pool aggregation is deliberately
/// avoided: each pool owns an independent 16-byte namespace and lifecycle.
#[allow(
    clippy::too_many_lines,
    reason = "the command intentionally keeps its table construction and summary together"
)]
fn cmd_collections_in_dir(
    dir: &Path,
    layout: bool,
    sort: Option<&str>,
    limit: i64,
) -> anyhow::Result<()> {
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
                | "index"
                | "load"
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
            "index" => right.2.cmp(&left.2),
            "load" => load_factor_percent(right.1, right.3)
                .total_cmp(&load_factor_percent(left.1, left.3)),
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
            "  {:>5}  {:<34}  {:>7}  {:>6}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
            "slot", "collection", "nodes", "load", "packs", "disk", "runs", "largest", "avoidable"
        );
    } else {
        println!(
            "  {:>5}  {:<34}  {:>7}  {:>6}  {:>6}  {:>12}  {:>13}",
            "slot", "collection", "nodes", "load", "shards", "index", "disk"
        );
    }
    let mut total_nodes = 0_usize;
    let mut total_memory = 0_usize;
    let mut total_disk_bytes = 0_u64;
    let total_rows = ordered.len();
    for (_, (collection_id, nodes, memory, _)) in &ordered {
        total_nodes = total_nodes
            .checked_add(*nodes)
            .context("total collection node count overflow")?;
        total_memory = total_memory
            .checked_add(*memory)
            .context("total collection index memory overflow")?;
        let disk = disk_bytes.get(collection_id).copied().unwrap_or(0);
        total_disk_bytes = total_disk_bytes.saturating_add(disk);
    }
    let max_rows = if limit <= 0 {
        usize::MAX
    } else {
        usize::try_from(limit).unwrap_or(usize::MAX)
    };
    for (i, (collection_id, nodes, memory, capacity)) in ordered.into_iter().take(max_rows) {
        let hex = hex_encode(collection_id);
        let load = fmt_load_percent(*nodes, *capacity);
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
        let disk = disk_bytes.get(collection_id).copied().unwrap_or(0);
        if layout {
            let stats = physical.collections.get(collection_id);
            let packs = stats.map_or(0, |s| s.pack_bytes.len());
            let runs = stats.map_or(0, |s| s.segments);
            let largest = stats.map_or(0, |s| s.largest_segment_bytes);
            let avoidable = avoidable_spread_bytes(stats);
            println!(
                "  {i:>5}  0x{hex}  {nodes:>7}  {load:>6}  {packs:>6}  {:>13}  {runs:>5}  {:>10}  {:>13}",
                fmt_disk_megabytes(disk),
                fmt_bytes(largest),
                fmt_bytes(avoidable)
            );
        } else {
            println!(
                "  {i:>5}  0x{hex}  {nodes:>7}  {load:>6}  {shards:>6}  {:>12}  {:>13}",
                fmt_index_kilobytes(*memory),
                fmt_disk_megabytes(disk),
            );
        }
    }
    println!();
    if layout {
        println!(
            "  {:>5}  {:<34}  {:>7}  {:>6}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
            "",
            "total",
            total_nodes,
            "",
            "",
            fmt_disk_megabytes(total_disk_bytes),
            "",
            "",
            ""
        );
    } else {
        println!(
            "  {:>5}  {:<34}  {total_nodes:>7}  {:>6}  {:>6}  {:>12}  {:>13}",
            "",
            "total",
            "",
            "",
            fmt_index_kilobytes(total_memory),
            fmt_disk_megabytes(total_disk_bytes),
        );
    }
    if layout {
        let total_segments: u64 = physical.collections.values().map(|s| s.segments).sum();
        let spread = physical
            .collections
            .values()
            .filter(|s| s.pack_bytes.len() > 1)
            .count();
        println!("physical layout: {spread} collection(s) span multiple packs; {total_segments} contiguous runs (includes superseded frames)");
    }
    if max_rows < total_rows {
        println!("note: only showing top {max_rows}; use `-l 0` to show all");
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
        mtxdb::packfile::read_header(&mut BufReader::new(file))
            .with_context(|| format!("unsupported or corrupt shard `{}`", path.display()))?;
    }
    Ok(())
}

/// Deliberately bypasses both `PackfileStorage` and `ShardPool` — it
/// needs no collection data and no frame reads. Instead it validates pack
/// headers, then decodes the small `shard_stats.bin` and
/// `shard_collections.bin` sidecars for counters and live-node counts.
/// Safe to run against a directory a live writer process owns.
/// Rule printed above and below each shard-type's block when a command
/// lists every independent pool at once (`--all`) — the section header and
/// table alone read as one undifferentiated wall of numbers otherwise.
const SECTION_RULE: &str =
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";

/// Prints one `--all` section's fenced header: a rule, an uppercased
/// `-- NAME --` banner (hyphens in the pool's directory name become spaces,
/// e.g. `event-dag` -> `EVENT DAG`), another rule, then a blank line.
fn print_section_header(shard_type: ShardType) {
    let banner = shard_type.as_str().replace('-', " ").to_uppercase();
    println!("{SECTION_RULE}");
    println!("-- {banner} --");
    println!("{SECTION_RULE}");
    println!();
}

/// Whether `--layout`'s interleaving count is worth an actionable note:
/// at least one extra run per collection on average (excess runs ≥ number
/// of collections sharing packs). Below that — a few collections each split
/// a couple of times by a shared active pack — it's expected write
/// interleaving, not a fragmentation problem worth advice about.
#[must_use]
fn interleaving_worth_noting(collections: u64, excess_runs: u64) -> bool {
    excess_runs >= collections.max(1)
}

/// Physical-layout block of `mtxdb shards --layout`: per-pack contiguous
/// runs, interleaving excess, and — when fragmentation is worth acting on —
/// the note that repack is never automatic.
fn print_pack_physical_layout(
    shard_entries: &[(u64, u64, u8)],
    physical: &mtxdb::packfile::layout::PhysicalLayout,
) {
    // Aggregate only over the packs listed in `shard_entries`: the caller
    // passes every pack for `mtxdb shards --layout` (pool-wide totals) but a
    // single pack for `mtxdb info 0x…`, where reporting pool-wide runs while
    // displaying only the selected pack would mislead.
    let listed_packs = shard_entries
        .iter()
        .filter_map(|(pack_id, _, _)| physical.packs.get(pack_id));
    let runs: u64 = listed_packs.clone().map(|stats| stats.segments).sum();
    let interleaved: u64 = listed_packs
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
    for (pack_id, _, _) in shard_entries {
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
    let collections_total: u64 = shard_entries
        .iter()
        .filter_map(|(pack_id, _, _)| physical.packs.get(pack_id))
        .map(|stats| stats.collections.len() as u64)
        .sum();
    if interleaving_worth_noting(collections_total, interleaved) {
        println!(
            "note: {interleaved} excess runs from interleaving across collections sharing packs"
        );
        println!("      repack is never automatic — run `mtxdb repack --all` to fold each");
        println!("      collection into a single contiguous run");
    }
}

fn cmd_shards(cli: &Cli, all: bool, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    // See the matching comment in `cmd_collections`: `-t all` must behave
    // like `--all`, not fall through to `require_shard_type`'s rejection.
    if all || cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let types: Vec<ShardType> = cli.shard_types().collect();
        for (index, shard_type) in types.into_iter().enumerate() {
            if index != 0 {
                println!();
                println!();
            }
            print_section_header(shard_type);
            cmd_shards_in_dir(&pool_dir(&db_layout, shard_type)?, layout, sort)?;
        }
        return Ok(());
    }
    cmd_shards_in_dir(&selected_pool_dir(cli)?, layout, sort)
}

/// Print runtime, open, and persisted pool statistics.
///
/// A read-only open has no live read/write history of its own, so the runtime
/// counters shown are those of *this* open (a fresh process): the meaningful
/// numbers are the open-phase breakdown (a real cold-open measurement against
/// the persisted checkpoint) and the persisted per-shard IO/sync counters
/// restored from `shard_stats.bin`. In-process embedders get the full live
/// picture from `PackfileStorage::stats()` instead — this command is the
/// on-disk + cold-open view.
fn read_stats_for_dir(
    dir: &Path,
) -> anyhow::Result<(RuntimeStats, Vec<mtxdb::shard::ShardSummary>)> {
    if glob_pack_files(dir)?.is_empty() {
        Ok((RuntimeStats::default(), Vec::new()))
    } else {
        let store =
            PackfileStorage::open_read_only(dir.to_path_buf()).context("failed to open store")?;
        let stats = store.stats();
        let summaries = store.shard_summaries();
        Ok((stats, summaries))
    }
}

fn cmd_stats_in_dir(dir: &Path, json: bool) -> anyhow::Result<()> {
    let (stats, summaries) = read_stats_for_dir(dir)?;
    if json {
        print_stats_json(dir, &stats, &summaries);
    } else {
        print_stats_table(dir, &stats, &summaries);
    }
    Ok(())
}

fn cmd_stats(cli: &Cli, json: bool) -> anyhow::Result<()> {
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let types: Vec<ShardType> = cli.shard_types().collect();
        if json {
            let mut fields = Vec::new();
            for shard_type in &types {
                let dir = pool_dir(&db_layout, *shard_type)?;
                let (stats, summaries) = read_stats_for_dir(&dir)?;
                fields.push((
                    shard_type.as_str(),
                    stats_json_object(&dir, &stats, &summaries),
                ));
            }
            println!("{}", json_object(&fields, ""));
        } else {
            for (index, shard_type) in types.into_iter().enumerate() {
                if index != 0 {
                    println!();
                    println!();
                }
                print_section_header(shard_type);
                cmd_stats_in_dir(&pool_dir(&db_layout, shard_type)?, false)?;
            }
        }
        return Ok(());
    }
    cmd_stats_in_dir(&selected_pool_dir(cli)?, json)
}

/// Milliseconds with two decimals (`2.12 ms`).
#[allow(
    clippy::cast_precision_loss,
    reason = "display-only rounding to 2 decimals; at typical durations this is exact to ~10µs"
)]
fn fmt_ms(duration: std::time::Duration) -> String {
    format!("{:.2} ms", duration.as_secs_f64() * 1000.0)
}

fn open_path_label(path: OpenPath) -> &'static str {
    match path {
        OpenPath::Checkpoint => "checkpoint",
        OpenPath::FullScan => "full-scan (no usable checkpoint; cold rebuild)",
    }
}

#[allow(clippy::too_many_lines)]
fn print_stats_table(dir: &Path, stats: &RuntimeStats, summaries: &[mtxdb::shard::ShardSummary]) {
    println!("mtxdb stats: {}", dir.display());
    println!(
        "  created by     mtxdb {}",
        mtxdb::shard::store_created_by_version(dir)
            .as_deref()
            .unwrap_or("unknown (created before version tracking, or marker unreadable)")
    );
    println!();
    println!("  open");
    match &stats.last_open_timings {
        Some(timings) => {
            println!("    path             {}", open_path_label(timings.path));
            println!(
                "    shard open       {}   metadata {}",
                fmt_ms(timings.shard_open),
                fmt_ms(timings.metadata_load)
            );
            println!(
                "    checkpoint       {}   materialize {}   delta replay {}",
                fmt_ms(timings.checkpoint_decode),
                fmt_ms(timings.index_materialization),
                fmt_ms(timings.delta_replay)
            );
            println!("    total            {}", fmt_ms(timings.total));
        }
        None => println!("    (none recorded)"),
    }
    println!(
        "  collections        {}    index bytes {}",
        stats.collection_count,
        fmt_bytes(stats.index_bytes)
    );
    println!();
    println!("  runtime (this process)");
    println!(
        "    put (single)     {} calls    {}",
        stats.put_calls,
        fmt_bytes(stats.put_bytes)
    );
    println!(
        "    put (batch)      {} batches / {} records / {}",
        stats.put_many_calls,
        stats.put_many_records,
        fmt_bytes(stats.put_many_bytes)
    );
    println!(
        "    put fast path    {} calls   materialized   {} calls",
        stats.put_many_fast_path_calls, stats.put_many_clone_path_calls
    );
    println!(
        "    index clone      {}   grows {}   rebuilds {}   max probe len {}",
        fmt_ms(stats.index_clone_time),
        stats.index_grow_count,
        stats.index_rebuild_count,
        stats.max_index_probe_len
    );
    println!(
        "    delta            {} invalidations   {} checkpoint writes   {} delta appends",
        stats.delta_invalidations, stats.checkpoint_writes, stats.delta_appends
    );
    println!(
        "    sync             {} calls   dirty-lock wait {}",
        stats.sync_calls,
        fmt_ms(stats.dirty_lock_wait)
    );
    println!(
        "    get (read ctrs)  {} calls / {} misses   get_many {} batches / {} records / {} misses",
        stats.get_calls,
        stats.get_misses,
        stats.get_many_calls,
        stats.get_many_records,
        stats.get_many_misses
    );
    println!(
        "    read amplification {} index candidates / {} candidate reads / {} hash mismatches / {} get_many shard touches",
        stats.index_candidates,
        stats.candidate_reads,
        stats.candidate_hash_mismatches,
        stats.get_many_shards_touched
    );
    println!(
        "    read scatter     {} est. runs / {} span / {} frame bytes",
        stats.read_many_runs,
        fmt_bytes(stats.read_many_span_bytes),
        fmt_bytes(stats.candidate_frame_bytes)
    );
    println!(
        "    repack           {} reps / {} kept / {} dropped",
        stats.repack.repack_count, stats.repack.kept_total, stats.repack.dropped_total
    );
    println!(
        "    cache            {} hits / {} misses ({:.1}%)",
        stats.cache.hits,
        stats.cache.misses,
        stats.cache.hit_rate * 100.0
    );
    println!();
    println!("  per-shard (persisted through `shard_stats.bin`):");
    println!("    pack    bytes      writes   syncs");
    for summary in summaries {
        println!(
            "    {:>4}   {:>10}   {:>7}   {:>5}",
            summary.pack_id,
            fmt_bytes(summary.stats.bytes_written),
            summary.stats.write_count,
            summary.stats.sync_count
        );
    }
}

/// JSON variant of `mtxdb stats`: open breakdown, runtime counters, and
/// per-shard persisted counters as nested objects. Built by hand (no serde
/// dependency) — every value is a plain number or a path string.
#[allow(clippy::too_many_lines)]
fn stats_json_object(
    dir: &Path,
    stats: &RuntimeStats,
    summaries: &[mtxdb::shard::ShardSummary],
) -> String {
    let open_path = stats
        .last_open_timings
        .as_ref()
        .map_or("null".to_owned(), |t| match t.path {
            OpenPath::Checkpoint => json_string_raw("checkpoint"),
            OpenPath::FullScan => json_string_raw("full-scan"),
        });
    let open_timings = stats.last_open_timings.as_ref().map_or_else(
        || "null".to_owned(),
        |t| {
            json_object(
                &[
                    ("shard_open", ms_json(t.shard_open)),
                    ("metadata_load", ms_json(t.metadata_load)),
                    ("checkpoint_decode", ms_json(t.checkpoint_decode)),
                    ("fingerprint", ms_json(t.fingerprint)),
                    ("index_materialization", ms_json(t.index_materialization)),
                    ("delta_replay", ms_json(t.delta_replay)),
                    ("full_scan", ms_json(t.full_scan)),
                    ("total", ms_json(t.total)),
                ],
                "  ",
            )
        },
    );
    let runtime = json_object(
        &[
            ("put_calls", stats.put_calls.to_string()),
            ("put_bytes", stats.put_bytes.to_string()),
            ("put_many_calls", stats.put_many_calls.to_string()),
            ("put_many_records", stats.put_many_records.to_string()),
            ("put_many_bytes", stats.put_many_bytes.to_string()),
            (
                "put_many_fast_path_calls",
                stats.put_many_fast_path_calls.to_string(),
            ),
            (
                "put_many_clone_path_calls",
                stats.put_many_clone_path_calls.to_string(),
            ),
            (
                "index_clone_time_ns",
                stats.index_clone_time.as_nanos().to_string(),
            ),
            ("index_grow_count", stats.index_grow_count.to_string()),
            ("index_rebuild_count", stats.index_rebuild_count.to_string()),
            ("delta_invalidations", stats.delta_invalidations.to_string()),
            ("checkpoint_writes", stats.checkpoint_writes.to_string()),
            ("delta_appends", stats.delta_appends.to_string()),
            ("sync_calls", stats.sync_calls.to_string()),
            ("get_calls", stats.get_calls.to_string()),
            ("get_misses", stats.get_misses.to_string()),
            ("get_many_calls", stats.get_many_calls.to_string()),
            ("get_many_records", stats.get_many_records.to_string()),
            ("get_many_misses", stats.get_many_misses.to_string()),
            ("index_candidates", stats.index_candidates.to_string()),
            ("candidate_reads", stats.candidate_reads.to_string()),
            (
                "candidate_hash_mismatches",
                stats.candidate_hash_mismatches.to_string(),
            ),
            (
                "get_many_shards_touched",
                stats.get_many_shards_touched.to_string(),
            ),
            (
                "candidate_frame_bytes",
                stats.candidate_frame_bytes.to_string(),
            ),
            ("read_many_runs", stats.read_many_runs.to_string()),
            (
                "read_many_span_bytes",
                stats.read_many_span_bytes.to_string(),
            ),
            ("repack_count", stats.repack.repack_count.to_string()),
            ("repack_kept_total", stats.repack.kept_total.to_string()),
            (
                "repack_dropped_total",
                stats.repack.dropped_total.to_string(),
            ),
            ("cache_hits", stats.cache.hits.to_string()),
            ("cache_misses", stats.cache.misses.to_string()),
            ("cache_hit_rate", stats.cache.hit_rate.to_string()),
            ("max_index_probe_len", stats.max_index_probe_len.to_string()),
            (
                "dirty_lock_wait_ns",
                stats.dirty_lock_wait.as_nanos().to_string(),
            ),
        ],
        "  ",
    );

    let mut shards = String::from("[\n");
    for (index, summary) in summaries.iter().enumerate() {
        let comma = if index == summaries.len().saturating_sub(1) {
            ""
        } else {
            ","
        };
        let _ = writeln!(
            shards,
            "    {{\"pack_id\":{},\"bytes\":{},\"writes\":{},\"syncs\":{}}}{comma}",
            summary.pack_id,
            summary.stats.bytes_written,
            summary.stats.write_count,
            summary.stats.sync_count
        );
    }
    shards.push_str("  ]");

    let created_by =
        mtxdb::shard::store_created_by_version(dir).map_or("null".to_owned(), |v| json_string(&v));
    let top = vec![
        ("dir", json_string(&dir.to_string_lossy())),
        ("created_by_mtxdb_version", created_by),
        ("open_path", open_path),
        ("open_count", stats.open_count.to_string()),
        ("collections", stats.collection_count.to_string()),
        ("index_bytes", stats.index_bytes.to_string()),
        ("open_timings_ms", open_timings),
        ("runtime", runtime),
        ("shards", shards),
    ];
    json_object(&top, "")
}

fn print_stats_json(dir: &Path, stats: &RuntimeStats, summaries: &[mtxdb::shard::ShardSummary]) {
    println!("{}", stats_json_object(dir, stats, summaries));
}

/// A complete `{ ... }` JSON object from pre-rendered fields, with
/// comma/indent bookkeeping so no trailing commas ever appear.
fn json_object(fields: &[(&str, String)], indent: &str) -> String {
    let mut out = String::from("{\n");
    for (index, (key, value)) in fields.iter().enumerate() {
        let comma = if index == fields.len().saturating_sub(1) {
            ""
        } else {
            ","
        };
        let _ = writeln!(out, "{indent}    \"{key}\": {value}{comma}");
    }
    let _ = writeln!(out, "{indent}}}");
    out
}

/// Milliseconds as a JSON number with 3 decimals (`2.120`).
#[allow(
    clippy::cast_precision_loss,
    reason = "display-only; sub-ns precision is meaningless in a JSON stats dump"
)]
fn ms_json(duration: std::time::Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1000.0)
}

/// A JSON string literal (no quotes added — callers pass raw text).
fn json_string_raw(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len().saturating_add(2));
    escaped.push('"');
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character <= '\u{1f}' => {
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

/// A JSON string value including surrounding quotes.
fn json_string(text: &str) -> String {
    json_string_raw(text)
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
    let (index_requirement, index_requirements_by_shard) = index_requirements_from_disk(dir)
        .map_or((None, None), |(total, by_shard)| {
            (Some(total), Some(by_shard))
        });
    print_shard_table(
        &shard_entries,
        &stats_map,
        node_counts.as_ref(),
        collection_counts.as_ref(),
        total_collections,
        index_requirement,
        index_requirements_by_shard.as_ref(),
    );
    if let Some(physical) = physical {
        print_pack_physical_layout(&shard_entries, &physical);
    }
    println!("* active pack");
    println!(
        "{} pack(s), {}",
        shard_entries.len(),
        stats_snapshot_summary(persisted_at)
    );
    Ok(())
}

/// Calculate each pack's associated collection-index allocation and the
/// de-duplicated allocation for the whole pool from the persisted directory.
fn index_requirements_from_disk(dir: &Path) -> Option<(usize, HashMap<u64, usize>)> {
    let summaries = PackfileStorage::collection_summaries_from_disk(dir)?;
    let collection_shards = PackfileStorage::collection_shards_from_disk(dir)?;
    let mut total = 0_usize;
    let mut by_shard = HashMap::new();
    for (collection_id, _, memory, _capacity) in summaries {
        total = total.checked_add(memory)?;
        for pack_id in collection_shards.get(&collection_id)? {
            let entry = by_shard.entry(*pack_id).or_insert(0_usize);
            *entry = entry.checked_add(memory)?;
        }
    }
    Some((total, by_shard))
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
        let header = mtxdb::packfile::read_header(&mut BufReader::new(file))
            .with_context(|| format!("unsupported or corrupt pack `{}`", path.display()))?
            .with_context(|| format!("invalid pack header `{}`", path.display()))?;
        if header.pack_id != pack_id {
            bail!(
                "pack {} identifies itself as {:#x}",
                path.display(),
                header.pack_id
            );
        }
        packs.push((pack_id, entry.metadata()?.len(), mtxdb::packfile::VERSION));
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
    index_requirement: Option<usize>,
    index_requirements_by_shard: Option<&HashMap<u64, usize>>,
) {
    // `ShardPool::open_internal` restores the newest pack as its append
    // destination. Mirror that recovery rule here without opening a writer.
    let active_pack_id = shard_entries.iter().map(|(pack_id, _, _)| *pack_id).max();

    println!(
        "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
        "pack id", "ver", "bytes", "nodes", "collections", "index", "syncs",
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
        let index_requirement = index_requirements_by_shard
            .and_then(|requirements| requirements.get(&pack_id))
            .map_or_else(|| "?".to_owned(), |bytes| fmt_megabytes(*bytes));
        println!(
            "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
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
            index_requirement,
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
        "{:>19}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
        "total",
        "",
        fmt_bytes(total_bytes),
        total_nodes.map_or_else(|| "?".to_owned(), |count| count.to_string()),
        total_collections.map_or_else(|| "?".to_owned(), |count| count.to_string()),
        index_requirement.map_or_else(|| "?".to_owned(), fmt_megabytes),
        total_syncs,
    );
    if let (Some(total), Some(by_shard)) = (index_requirement, index_requirements_by_shard) {
        let per_shard_total = by_shard.values().copied().sum::<usize>();
        if per_shard_total > total {
            println!("note: per-pack index requirements overlap for collections spanning packs; total is de-duplicated");
        }
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

/// `info` accepts a collection selector (slot index or 32-hex collection ID)
/// or a pack selector (`0x`-prefixed, 1–16 hex digits, as printed by
/// `mtxdb shards`). Dispatch on the hex length the same way `scan` does.
fn cmd_info(cli: &Cli, selector: &str) -> anyhow::Result<()> {
    if selector
        .strip_prefix("0x")
        .or_else(|| selector.strip_prefix("0X"))
        .is_some_and(|hex| hex.len() != 32)
    {
        return cmd_info_pack(cli, selector);
    }
    cmd_info_collection(cli, selector)
}

/// Print a pack's age (header `created_at`) and how long it stayed the
/// active append target (span to its file's last-modified time), from the
/// same header `mtxdb shards` already validates when discovering packs.
fn print_pack_lifetime(path: &Path) {
    let created_at = fs::File::open(path).ok().and_then(|file| {
        mtxdb::packfile::read_header(&mut BufReader::new(file))
            .ok()
            .flatten()
            .map(|header| header.created_at)
    });
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let modified = fs::metadata(path).ok().and_then(|metadata| {
        metadata.modified().ok().and_then(|modified| {
            modified
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_secs())
        })
    });
    match (created_at, modified) {
        (Some(created_at), Some(modified)) => {
            println!(
                "created: {} ago, last write: {} ago (active {} span)",
                fmt_duration(now.saturating_sub(created_at)),
                fmt_duration(now.saturating_sub(modified)),
                fmt_duration(modified.saturating_sub(created_at))
            );
        }
        (Some(created_at), None) => {
            println!(
                "created: {} ago",
                fmt_duration(now.saturating_sub(created_at))
            );
        }
        _ => {}
    }
}

/// Print the same per-pack summary row `mtxdb shards` would, for one pack ID.
#[allow(
    clippy::too_many_lines,
    reason = "prints one pack's stats, physical layout, and per-collection breakdown in sequence"
)]
fn print_pack_info(
    dir: &Path,
    pack_id: u64,
    shard_entries: &[(u64, u64, u8)],
    shard_type: ShardType,
) {
    let (stats_map, _) = decode_stats_snapshot(dir);
    let node_counts = PackfileStorage::shard_node_counts_from_disk(dir);
    let collection_counts = PackfileStorage::shard_collection_counts_from_disk(dir);
    // Scoped to the selected pack, not the pool: `shard_entries` (and thus
    // the table's per-row bytes/nodes/syncs) already contain only this one
    // pack, so the "total" row below must match rather than mixing in
    // pool-wide collection/index figures from every other pack.
    let total_collections = collection_counts
        .as_ref()
        .and_then(|counts| counts.get(&pack_id))
        .map(|&count| usize::try_from(count).unwrap_or(usize::MAX));
    let index_requirements_by_shard =
        index_requirements_from_disk(dir).map(|(_, by_shard)| by_shard);
    let index_requirement = index_requirements_by_shard
        .as_ref()
        .and_then(|by_shard| by_shard.get(&pack_id))
        .copied();
    print_shard_table(
        shard_entries,
        &stats_map,
        node_counts.as_ref(),
        collection_counts.as_ref(),
        total_collections,
        index_requirement,
        index_requirements_by_shard.as_ref(),
    );

    println!();
    println!("type: {}", shard_type.as_str());
    println!("generation: {pack_id} (0x{pack_id:016x})");
    let path = dir.join(format!("pack_{pack_id:016x}.pack"));
    println!("path: {}", path.display());

    print_pack_lifetime(&path);

    let (write_count, bytes_written, sync_count) =
        stats_map.get(&pack_id).copied().unwrap_or_default();
    println!(
        "writes: {write_count} ({} written), {sync_count} sync{}",
        fmt_bytes(bytes_written),
        if sync_count == 1 { "" } else { "s" }
    );
    println!(
        "collections: {}",
        collection_counts
            .as_ref()
            .and_then(|counts| counts.get(&pack_id))
            .copied()
            .unwrap_or(0)
    );

    let collection_shards = PackfileStorage::collection_shards_from_disk(dir);
    match physical_layout(dir) {
        Ok(physical) => {
            print_pack_physical_layout(shard_entries, &physical);
            if let Some(pack_layout) = physical.packs.get(&pack_id) {
                let mut collections: Vec<[u8; 16]> =
                    pack_layout.collections.iter().copied().collect();
                collections.sort_unstable();
                println!();
                println!("{:>34}  {:>10}", "collection", "bytes");
                for collection_id in collections {
                    let collection_layout = physical.collections.get(&collection_id);
                    let bytes = collection_layout
                        .and_then(|layout| layout.pack_bytes.get(&pack_id))
                        .copied()
                        .unwrap_or(0);
                    // A selected pack that holds only superseded frames for this
                    // collection is absent from `collection_shards` (which lists
                    // only live/reachable packs), so `pack_id` itself may not be
                    // among `shards` here -- subtracting one unconditionally
                    // would then undercount the collection's other live packs.
                    let shards = collection_shards
                        .as_ref()
                        .and_then(|shards| shards.get(&collection_id));
                    let other_packs = shards.map_or(0, |shards| {
                        if shards.contains(&pack_id) {
                            shards.len().saturating_sub(1)
                        } else {
                            shards.len()
                        }
                    });
                    let note = if other_packs > 0 {
                        let spread = collection_layout
                            .map_or(0, CollectionPhysicalLayout::avoidable_spread_bytes);
                        format!(
                            "(also in {other_packs} other pack{}, {} avoidably spread)",
                            if other_packs == 1 { "" } else { "s" },
                            fmt_bytes(spread)
                        )
                    } else {
                        String::new()
                    };
                    println!(
                        "{:>34}  {:>10}  {}",
                        hex_encode(&collection_id),
                        fmt_bytes(bytes),
                        note
                    );
                }
            }
        }
        Err(error) => {
            eprintln!(
                "warning: unable to compute physical layout for `{}`: {error}",
                dir.display()
            );
        }
    }
}

fn cmd_info_pack(cli: &Cli, selector: &str) -> anyhow::Result<()> {
    let pack_id = parse_pack_id_selector(selector)?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let shard_entries: Vec<(u64, u64, u8)> = glob_pack_files(&dir)?
                .into_iter()
                .filter(|&(id, _, _)| id == pack_id)
                .collect();
            if !shard_entries.is_empty() {
                if matched_any {
                    println!();
                    println!();
                }
                print_section_header(shard_type);
                print_pack_info(&dir, pack_id, &shard_entries, shard_type);
                matched_any = true;
            }
        }
        if !matched_any {
            eprintln!("pack 0x{pack_id:016x}: not found");
        }
        return Ok(());
    }
    let dir = selected_pool_dir(cli)?;
    let shard_entries: Vec<(u64, u64, u8)> = glob_pack_files(&dir)?
        .into_iter()
        .filter(|&(id, _, _)| id == pack_id)
        .collect();
    if shard_entries.is_empty() {
        eprintln!("pack 0x{pack_id:016x}: not found");
        return Ok(());
    }
    print_pack_info(&dir, pack_id, &shard_entries, cli.require_shard_type()?);
    Ok(())
}

fn print_collection_info(
    dir: &Path,
    collection_id: &[u8; 16],
    len: usize,
    mem: usize,
    capacity: u32,
    shards: &[u64],
) {
    let hex = hex_encode(collection_id);
    println!(
        "collection {hex}: {len} nodes, {} index ({})",
        fmt_megabytes(mem),
        fmt_load_factor(len, capacity)
    );
    print_collection_shards(shards);
    if let Some(details) = matrix_room_details_from_cache(dir, collection_id) {
        print_matrix_room_details(details, None);
    } else {
        println!(
            "  {:<12} scanning {} shard{}...",
            "metadata:",
            shards.len(),
            if shards.len() == 1 { "" } else { "s" }
        );
        let details =
            matrix_room_details_from_shards(dir, collection_id, shards).unwrap_or_else(|error| {
                eprintln!("warning: unable to inspect Matrix room metadata: {error}");
                MatrixRoomDetails::default()
            });
        if let Err(error) = persist_matrix_room_details(dir, collection_id, &details) {
            eprintln!("warning: unable to cache Matrix room metadata: {error}");
        }
        print_matrix_room_details(details, Some(shards.len()));
    }
}

fn inspect_collection_in_dir(dir: &Path, collection_id: &[u8; 16]) -> anyhow::Result<bool> {
    if let (Some(summaries), Some(collection_shards)) = (
        PackfileStorage::collection_summaries_from_disk(dir),
        PackfileStorage::collection_shards_from_disk(dir),
    ) {
        if let Some((_, len, mem, capacity)) = summaries
            .into_iter()
            .find(|(id, _, _, _)| id == collection_id)
        {
            let shards = collection_shards
                .get(collection_id)
                .cloned()
                .unwrap_or_default();
            print_collection_info(dir, collection_id, len, mem, capacity, &shards);
            return Ok(true);
        }
        return Ok(false);
    }

    let Ok(store) = PackfileStorage::open_read_only(dir.to_path_buf()) else {
        return Ok(false);
    };
    if let Some((len, mem, capacity)) = store.collection_index_info(collection_id) {
        let hex = hex_encode(collection_id);
        println!(
            "collection {hex}: {len} nodes, {} index ({})",
            fmt_megabytes(mem),
            fmt_load_factor(len, capacity)
        );
        let shards = store.collection_referenced_pack_ids(collection_id);
        print_collection_shards(&shards);
        print_matrix_room_details(matrix_room_details(&store, dir, collection_id)?, None);
        return Ok(true);
    }
    Ok(false)
}

fn cmd_info_collection(cli: &Cli, collection: &str) -> anyhow::Result<()> {
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let collection_id = match collection.parse::<usize>() {
                Ok(slot) => {
                    let Ok(collections) = collection_ids_in_dir(&dir) else {
                        continue;
                    };
                    match collections.get(slot) {
                        Some(&id) => id,
                        None => continue,
                    }
                }
                Err(_) => parse_collection_selector(collection, cli.namespace.as_deref())?,
            };
            let has_collection =
                if let Some(summaries) = PackfileStorage::collection_summaries_from_disk(&dir) {
                    summaries.iter().any(|(id, _, _, _)| id == &collection_id)
                } else if let Ok(store) = PackfileStorage::open_read_only(dir.clone()) {
                    store.collection_index_info(&collection_id).is_some()
                } else {
                    false
                };
            if has_collection {
                if matched_any {
                    println!();
                    println!();
                }
                print_section_header(shard_type);
                inspect_collection_in_dir(&dir, &collection_id)?;
                matched_any = true;
            }
        }
        if !matched_any {
            let display = match parse_collection_selector(collection, cli.namespace.as_deref()) {
                Ok(id) => hex_encode(&id),
                Err(_) => collection.to_owned(),
            };
            eprintln!("collection {display}: not found");
        }
        return Ok(());
    }

    let collection_id = match collection.parse::<usize>() {
        Ok(slot) => {
            let collections = collection_ids(cli)?;
            collections
                .get(slot)
                .copied()
                .with_context(|| format!("collection slot {slot} not found"))?
        }
        Err(_) => parse_collection_selector(collection, cli.namespace.as_deref())?,
    };
    let hex = hex_encode(&collection_id);
    let dir = selected_pool_dir(cli)?;

    if inspect_collection_in_dir(&dir, &collection_id)? {
        return Ok(());
    }

    eprintln!(
        "collection {hex}: not found{}",
        other_shard_type_hint(cli, &collection_id)
    );
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
        for (record_collection_id, node_id, _) in mtxdb::packfile::scan_packfile(&path)? {
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
        if mtxdb::packfile::read_header(&mut reader)?.is_none() {
            continue;
        }
        loop {
            let record = match mtxdb::packfile::read_record(&mut reader) {
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
        if hex.len() == 32 {
            bail!(
                "invalid pack ID `{selector}`; that's 32 hex digits, which looks like a \
                 collection ID, not a pack ID — pack IDs are 1–16 hex digits, shown by \
                 `mtxdb shards`"
            );
        }
        bail!("invalid pack ID `{selector}`; expected 1–16 hexadecimal digits after 0x");
    }
    u64::from_str_radix(hex, 16).with_context(|| format!("invalid pack ID `{selector}`"))
}

/// Common options shared by `scan_pack` and `cmd_scan_collection`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "CLI scan flags map 1:1 to bools"
)]
struct ScanOptions {
    verbose: bool,
    limit: i64,
    node_id: Option<[u8; 16]>,
    raw: bool,
    sort: Option<SortColumn>,
    reverse: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortColumn {
    Payload,
    Offset,
}

impl SortColumn {
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "payload" => Ok(Self::Payload),
            "offset" => Ok(Self::Offset),
            other => bail!("unknown scan sort column `{other}`"),
        }
    }
}

impl ScanOptions {
    fn max_rows(&self) -> usize {
        scan_limit(self.limit)
    }

    fn is_sorting(&self) -> bool {
        self.sort.is_some()
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "maps 1:1 to CLI args before building ScanOptions"
)]
fn cmd_scan(
    cli: &Cli,
    selector: &str,
    verbose: bool,
    limit: i64,
    id: Option<&str>,
    collection: Option<&str>,
    raw: bool,
    sort: Option<&str>,
    reverse: bool,
) -> anyhow::Result<()> {
    let sort_column = sort.map(SortColumn::from_str).transpose()?;
    if sort_column == Some(SortColumn::Payload) && (!verbose || raw) {
        bail!("scan sorting requires --verbose and cannot be combined with --raw");
    }
    let node_id = id.map(parse_node_id).transpose()?;
    let collection_filter = collection
        .map(|value| parse_collection_selector(value, cli.namespace.as_deref()))
        .transpose()?;
    let opts = ScanOptions {
        verbose,
        limit,
        node_id,
        raw,
        sort: sort_column,
        reverse,
    };
    // Mirror `cmd_info`'s routing exactly: a selector is a pack ID only when
    // it's `0x`-prefixed with something other than 32 hex digits after it —
    // everything else (bare 32 hex digits, or `0x` + 32 hex digits) is a
    // collection ID. `info` and `scan` used to disagree here (`scan`
    // required the `0x` prefix on a collection ID; `info` didn't), which
    // made the same selector work for one command and not the other.
    let looks_like_pack_id = selector
        .strip_prefix("0x")
        .or_else(|| selector.strip_prefix("0X"))
        .is_some_and(|hex| hex.len() != 32);
    if !looks_like_pack_id {
        if collection_filter.is_some() {
            bail!("--collection is only valid when scanning a pack ID; the selector already identifies the collection");
        }
        return cmd_scan_collection(cli, selector, &opts);
    }
    let pack_id = parse_pack_id_selector(selector)?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let pool_dir = pool_dir(&db_layout, shard_type)?;
            let Ok(pool) = ShardPool::open_read_only(pool_dir) else {
                continue;
            };
            let shard = pool
                .all_shards()
                .into_iter()
                .find_map(|(_, shard)| (shard.pack_id == pack_id).then_some(shard));
            if let Some(shard) = shard {
                if matched_any {
                    println!();
                    println!();
                }
                print_section_header(shard_type);
                scan_pack(cli, &shard, pack_id, collection_filter, &opts, shard_type)?;
                matched_any = true;
            }
        }
        if !matched_any {
            eprintln!("pack 0x{pack_id:016x}: not found");
        }
        return Ok(());
    }
    let shard_type = cli.require_shard_type()?;
    let pool_dir = selected_pool_dir(cli)?;
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let shard = pool
        .all_shards()
        .into_iter()
        .find_map(|(_, shard)| (shard.pack_id == pack_id).then_some(shard))
        .with_context(|| format!("pack ID 0x{pack_id:016x} not found"))?;
    scan_pack(cli, &shard, pack_id, collection_filter, &opts, shard_type)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the pack scan helper receives the explicit scan filters and output options"
)]
fn scan_pack(
    _cli: &Cli,
    shard: &std::sync::Arc<mtxdb::shard::Shard>,
    pack_id: u64,
    collection_filter: Option<[u8; 16]>,
    opts: &ScanOptions,
    shard_type: ShardType,
) -> anyhow::Result<()> {
    let path = &shard.path;
    let max_rows = opts.max_rows();
    let mut records = Vec::new();
    let mut matched_records = 0usize;
    let mut truncated = false;
    let verify_payload = opts.verbose || opts.raw;
    for record in mtxdb::packfile::scan_packfile_iter(path, verify_payload)? {
        let (record_collection, record_id, offset) = record?;
        // TODO: tied to MSRV 1.81.0 — replace with .is_none_or() once the
        // minimum is bumped to 1.82+.
        if !collection_filter.map_or(true, |wanted| record_collection == wanted)
            || !opts.node_id.map_or(true, |wanted| record_id == wanted)
        {
            continue;
        }
        matched_records = matched_records.saturating_add(1);
        if opts.is_sorting() || records.len() < max_rows {
            records.push((record_collection, record_id, offset));
        } else {
            truncated = true;
            break;
        }
    }
    if opts.is_sorting() {
        match opts.sort {
            Some(SortColumn::Payload) => {
                records.sort_by_cached_key(|(_, _, offset)| {
                    ShardPool::read_at_committed(shard, *offset, true)
                        .map(|record| record.data.to_vec())
                        .unwrap_or_default()
                });
            }
            Some(SortColumn::Offset) => {
                records.sort_by_key(|(_, _, offset)| *offset);
            }
            None => unreachable!(),
        }
        if opts.reverse {
            records.reverse();
        }
        truncated = records.len() > max_rows;
    }
    if opts.raw {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        for (_, _, offset) in records.iter().take(max_rows) {
            let data = ShardPool::read_at_committed(shard, *offset, true)?;
            out.write_all(&data.data)?;
            if opts.verbose {
                eprintln!(
                    "pack 0x{pack_id:016x}: raw frame @ {offset} ({} bytes, checksum verified)",
                    data.data.len()
                );
            }
        }
        out.flush()?;
        if truncated {
            eprintln!("note: only showing top {max_rows}; use `-l 0` to show all");
        }
        return Ok(());
    }
    if truncated {
        println!(
            "pack 0x{pack_id:016x}: {} bytes, showing first {max_rows} matching records (at least {})",
            std::fs::metadata(path)?.len(),
            matched_records,
        );
    } else {
        println!(
            "pack 0x{pack_id:016x}: {} bytes, {} records",
            std::fs::metadata(path)?.len(),
            matched_records,
        );
    }
    print_scan_table_header("COLLECTION", scan_payload_label(shard_type));
    for (collection_id, node_id, offset) in records.iter().take(max_rows) {
        let collection_hex = hex_encode(collection_id);
        let id_hex = hex_encode(node_id);
        let data = opts
            .verbose
            .then(|| ShardPool::read_at_committed(shard, *offset, true))
            .transpose()?;
        let payload = data
            .as_ref()
            .map(|data| scan_payload_cell(&data.data, shard_type));
        println!(
            "{}",
            scan_table_row(&collection_hex, &id_hex, *offset, payload.as_deref())
        );
        if let Some(data) =
            data.filter(|data| scan_payload_suffix(&data.data, shard_type).is_none())
        {
            print_scan_payload(&data.data);
        }
    }
    if truncated {
        println!("note: only showing top {max_rows}; use `-l 0` to show all");
    }
    Ok(())
}

/// Header for the scan table's aligned columns. `location_label` names the
/// first column ("COLLECTION" for a pack scan, "PACK" for a collection
/// scan) — the other column is whichever of the two identifies each row.
fn print_scan_table_header(location_label: &str, payload_label: &str) {
    println!(
        "  {:<34} {:<34} {:>10}  {payload_label}",
        location_label, "ID", "OFFSET"
    );
}

fn scan_payload_label(shard_type: ShardType) -> &'static str {
    if shard_type == ShardType::State {
        "PAYLOAD / PTR"
    } else {
        "PAYLOAD"
    }
}

/// One aligned row: `location` is a collection or pack ID (hex), `id` the
/// node ID (hex). `payload` is `None` for a non-`--verbose` scan (no frame
/// data was read).
fn scan_table_row(location: &str, id: &str, offset: u64, payload: Option<&str>) -> String {
    format!(
        "  {location:<34} {id:<34} {offset:>10}  {}",
        payload.unwrap_or("-")
    )
}

/// The PAYLOAD cell for one row: a decodable payload prints below the row
/// instead (its formatted form spans lines), so the cell just says so.
fn scan_payload_cell(data: &[u8], shard_type: ShardType) -> String {
    scan_payload_suffix(data, shard_type).unwrap_or_else(|| "(decoded below)".to_owned())
}

/// Print every physical frame for a collection across all packs. This is a
/// diagnostic scan, so superseded copies are deliberately retained in the
/// output; use `export` to enumerate only the collection's live records.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "pool directory, options, and section-header formatting flags passed down"
)]
fn cmd_scan_collection_in_pool(
    pool_dir: &Path,
    collection_id: [u8; 16],
    opts: &ScanOptions,
    shard_type: ShardType,
    show_section_header: bool,
    needs_section_spacing: bool,
) -> anyhow::Result<usize> {
    let pool = match ShardPool::open_read_only(pool_dir.to_path_buf()) {
        Ok(pool) => pool,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error).context("failed to open shard store"),
    };
    let mut shards = pool.all_shards();
    shards.sort_unstable_by_key(|(_, shard)| shard.pack_id);
    let collection_packs = PackfileStorage::collection_shards_from_disk(pool_dir)
        .and_then(|mut collections| collections.remove(&collection_id))
        .map(|packs| packs.into_iter().collect::<HashSet<_>>());

    let mut packs = 0_usize;
    let max_rows = opts.max_rows();
    let bounded = max_rows != usize::MAX;
    let mut context = CollectionScanContext {
        collection_id,
        node_id: opts.node_id,
        mode: CollectionScanMode::new(opts.verbose, opts.raw, opts.sort),
        max_rows,
        frames: 0,
        raw_matches: Vec::new(),
        header_printed: false,
        shard_type,
        sorted_records: Vec::new(),
        reverse: opts.reverse,
        show_section_header,
        needs_section_spacing,
    };
    for (_, shard) in shards {
        if let Some(collection_packs) = &collection_packs {
            if !collection_packs.contains(&shard.pack_id) {
                continue;
            }
        }
        let matched_pack = scan_collection_shard(&shard, &mut context)?;
        packs = packs.saturating_add(usize::from(matched_pack));
        if bounded && !opts.is_sorting() && context.frames >= max_rows {
            break;
        }
    }

    let frames = context.frames;
    if frames == 0 {
        return Ok(0);
    }
    sort_collection_records(&mut context);
    if context.mode.is_raw() {
        if context.show_section_header && !context.raw_matches.is_empty() {
            if context.needs_section_spacing {
                println!();
                println!();
            }
            print_section_header(context.shard_type);
        }
        let stdout = io::stdout();
        let mut out = stdout.lock();
        for (shard, _, offset) in context.raw_matches.iter().take(max_rows) {
            let data = ShardPool::read_at_committed(shard, *offset, true)?;
            out.write_all(&data.data)?;
            if opts.verbose {
                eprintln!(
                    "collection {}: raw frame @ {offset} ({} bytes, checksum verified)",
                    hex_encode(&collection_id),
                    data.data.len()
                );
            }
        }
        out.flush()?;
        if bounded && frames >= max_rows {
            eprintln!("note: showing first {max_rows} matching records; use -l 0 to scan all");
        } else {
            eprint_scan_limit_note(frames, max_rows);
        }
        return Ok(frames);
    }
    if context.mode.is_sorting() {
        if context.show_section_header && !context.header_printed {
            if context.needs_section_spacing {
                println!();
                println!();
            }
            print_section_header(context.shard_type);
            context.header_printed = true;
        }
        if bounded && frames >= max_rows {
            println!(
                "collection {}: showing first {frames} physical records (at least {packs} pack{})",
                hex_encode(&collection_id),
                if packs == 1 { "" } else { "s" },
            );
        } else {
            println!(
                "collection {}: {frames} physical record{} across {packs} pack{}",
                hex_encode(&collection_id),
                if frames == 1 { "" } else { "s" },
                if packs == 1 { "" } else { "s" },
            );
            print_scan_limit_note(frames, max_rows);
        }
        print_scan_table_header("PACK", scan_payload_label(context.shard_type));
        let sorted_records = context.sorted_records.clone();
        for (shard, record_id, offset) in sorted_records.iter().take(max_rows) {
            print_collection_record(shard, *record_id, *offset, &mut context)?;
        }
        return Ok(frames);
    }
    if bounded && frames >= max_rows {
        println!(
            "collection {}: showing first {frames} physical records (at least {packs} pack{})",
            hex_encode(&collection_id),
            if packs == 1 { "" } else { "s" },
        );
    } else {
        println!(
            "collection {}: {frames} physical record{} across {packs} pack{}",
            hex_encode(&collection_id),
            if frames == 1 { "" } else { "s" },
            if packs == 1 { "" } else { "s" },
        );
        print_scan_limit_note(frames, max_rows);
    }
    Ok(frames)
}

/// Print every physical frame for a collection across all packs. This is a
/// diagnostic scan, so superseded copies are deliberately retained in the
/// output; use `export` to enumerate only the collection's live records.
fn cmd_scan_collection(cli: &Cli, selector: &str, opts: &ScanOptions) -> anyhow::Result<()> {
    let collection_id = parse_collection_selector(selector, cli.namespace.as_deref())?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let pool_dir = pool_dir(&db_layout, shard_type)?;
            let frames = cmd_scan_collection_in_pool(
                &pool_dir,
                collection_id,
                opts,
                shard_type,
                true,
                matched_any,
            )?;
            if frames > 0 {
                matched_any = true;
            }
        }
        if !matched_any {
            bail!("no matching physical record found in collection {selector}");
        }
        return Ok(());
    }
    let shard_type = cli.require_shard_type()?;
    let pool_dir = selected_pool_dir(cli)?;
    let frames =
        cmd_scan_collection_in_pool(&pool_dir, collection_id, opts, shard_type, false, false)?;
    if frames == 0 {
        bail!(
            "no matching physical record found in collection {selector}{}",
            other_shard_type_hint(cli, &collection_id)
        );
    }
    Ok(())
}

fn sort_collection_records(context: &mut CollectionScanContext) {
    if context.mode.is_sorting() {
        match context.mode.sort_column() {
            Some(SortColumn::Payload) => {
                context
                    .sorted_records
                    .sort_by_cached_key(|(shard, _, offset)| {
                        ShardPool::read_at_committed(shard, *offset, true)
                            .map(|record| record.data.to_vec())
                            .unwrap_or_default()
                    });
            }
            Some(SortColumn::Offset) => {
                context.sorted_records.sort_by_key(|(_, _, offset)| *offset);
            }
            None => unreachable!(),
        }
        if context.reverse {
            context.sorted_records.reverse();
        }
    }
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "diagnostic scan state tracks output mode, row limits, and table header formatting"
)]
struct CollectionScanContext {
    collection_id: [u8; 16],
    node_id: Option<[u8; 16]>,
    mode: CollectionScanMode,
    max_rows: usize,
    frames: usize,
    raw_matches: Vec<(std::sync::Arc<mtxdb::shard::Shard>, [u8; 16], u64)>,
    header_printed: bool,
    shard_type: ShardType,
    sorted_records: Vec<(std::sync::Arc<mtxdb::shard::Shard>, [u8; 16], u64)>,
    reverse: bool,
    show_section_header: bool,
    needs_section_spacing: bool,
}

#[derive(Clone, Copy)]
enum CollectionScanMode {
    Plain,
    Verbose { sort: Option<SortColumn> },
    Raw { verbose: bool },
}

impl CollectionScanMode {
    fn new(verbose: bool, raw: bool, sort: Option<SortColumn>) -> Self {
        if raw {
            Self::Raw { verbose }
        } else if verbose {
            Self::Verbose { sort }
        } else {
            Self::Plain
        }
    }

    fn reads_payload(self) -> bool {
        !matches!(self, Self::Plain)
    }

    fn is_raw(self) -> bool {
        matches!(self, Self::Raw { .. })
    }

    fn is_sorting(self) -> bool {
        matches!(self, Self::Verbose { sort: Some(_) })
    }

    fn sort_column(self) -> Option<SortColumn> {
        match self {
            Self::Verbose { sort } => sort,
            _ => None,
        }
    }

    fn verbose(self) -> bool {
        matches!(self, Self::Verbose { .. } | Self::Raw { verbose: true })
    }
}

fn scan_collection_shard(
    shard: &std::sync::Arc<mtxdb::shard::Shard>,
    context: &mut CollectionScanContext,
) -> anyhow::Result<bool> {
    let records = mtxdb::packfile::scan_packfile_iter(&shard.path, context.mode.reads_payload())?;
    let mut matched_pack = false;
    for record in records {
        let (record_collection_id, record_id, offset) = record?;
        let id_matches = context.node_id.map_or(true, |wanted| record_id == wanted);
        if record_collection_id != context.collection_id || !id_matches {
            continue;
        }
        matched_pack = true;
        context.frames = context.frames.saturating_add(1);
        if context.mode.is_raw() {
            context.raw_matches.push((shard.clone(), record_id, offset));
        } else if context.mode.is_sorting() {
            context
                .sorted_records
                .push((shard.clone(), record_id, offset));
        } else if context.frames <= context.max_rows {
            print_collection_record(shard, record_id, offset, context)?;
        }
        if context.max_rows != usize::MAX
            && !context.mode.is_sorting()
            && context.frames >= context.max_rows
        {
            break;
        }
    }
    Ok(matched_pack)
}

fn print_collection_record(
    shard: &std::sync::Arc<mtxdb::shard::Shard>,
    record_id: [u8; 16],
    offset: u64,
    context: &mut CollectionScanContext,
) -> anyhow::Result<()> {
    if !context.header_printed {
        if context.show_section_header {
            if context.needs_section_spacing {
                println!();
                println!();
            }
            print_section_header(context.shard_type);
        }
        print_scan_table_header("PACK", scan_payload_label(context.shard_type));
        context.header_printed = true;
    }
    let data = context
        .mode
        .verbose()
        .then(|| ShardPool::read_at_committed(shard, offset, true))
        .transpose()?;
    let payload = data
        .as_ref()
        .map(|data| scan_payload_cell(&data.data, context.shard_type));
    println!(
        "{}",
        scan_table_row(
            &format!("0x{:016x}", shard.pack_id),
            &hex_encode(&record_id),
            offset,
            payload.as_deref(),
        )
    );
    if let Some(data) =
        data.filter(|data| scan_payload_suffix(&data.data, context.shard_type).is_none())
    {
        print_scan_payload(&data.data);
    }
    Ok(())
}

fn scan_limit(limit: i64) -> usize {
    if limit <= 0 {
        usize::MAX
    } else {
        usize::try_from(limit).unwrap_or(usize::MAX)
    }
}

fn print_scan_limit_note(total: usize, limit: usize) {
    if limit < total {
        println!("note: only showing top {limit}; use `-l 0` to show all");
    }
}

/// Same as [`print_scan_limit_note`] but to stderr -- for use after `--raw`
/// output, where stdout must stay a clean byte stream (see `--raw`'s help).
fn eprint_scan_limit_note(total: usize, limit: usize) {
    if limit < total {
        eprintln!("note: only showing top {limit}; use `-l 0` to show all");
    }
}

/// Print a readable payload for `scan --verbose` without ever treating an
/// arbitrary binary record as terminal text.
fn print_scan_payload(data: &[u8]) {
    let pretty = pretty_print_payload(data).expect("undecodable payloads use an inline suffix");
    for line in String::from_utf8_lossy(&pretty).lines() {
        println!("    {line}");
    }
}

/// Return an inline summary for a payload `pretty_print_payload` can't
/// decode; a decodable payload (plain JSON, or a recognized record shape
/// like `event_json`) is printed below its record instead because its
/// formatted representation spans lines.
fn scan_payload_suffix(data: &[u8], shard_type: ShardType) -> Option<String> {
    if shard_type == ShardType::State && data.len() == 8 {
        let state_group = u64::from_be_bytes(data.try_into().ok()?);
        return Some(format!("PTR: 0x{state_group:016x}"));
    }
    if pretty_print_payload(data).is_some() {
        return None;
    }
    Some(match data.len() {
        0 => "0 bytes [TOMBSTONE - GC'd ON REPACK]".to_owned(),
        _ => match data.get(0..4) {
            Some(magic) => format!(
                "{} bytes (undecodable, magic=0x{:08x})",
                data.len(),
                u32::from_be_bytes(magic.try_into().unwrap())
            ),
            None => format!("{} bytes (undecodable, too short)", data.len()),
        },
    })
}

fn cmd_import(
    cli: &Cli,
    paths: &[std::path::PathBuf],
    collection_override: Option<&str>,
    template: Option<&Path>,
) -> anyhow::Result<()> {
    let import_template = compile_import_template(template)?;
    // One writer owns the entire batch: it prevents another writer from
    // interleaving halfway through a shell glob, and lets us publish one
    // complete shard/collection snapshot once the final input has been handled.
    let pool_dir = selected_pool_dir(cli)?;
    // Admission is based on the actual pack contents rather than a sidecar:
    // a stale summary must not make an unestablished room look established.
    let mut established_collections = matrix_create_collections_on_disk(&pool_dir)?;
    // Import buffers appends (one positioned write per ~1 MiB of frames
    // instead of one per record) and ends with a single `sync_all`, which
    // flushes and fsyncs everything below — the explicit durability schedule
    // the buffered append policy is meant for. A `put` command that syncs
    // per record stays on the default eager path.
    let store = open_store(cli)?.with_append_policy(mtxdb::shard::AppendPolicy::buffered());
    let mut failures = 0_usize;
    for (index, path) in paths.iter().enumerate() {
        if index != 0 {
            eprintln!();
        }
        if let Err(error) = cmd_import_file(
            &store,
            &pool_dir,
            path,
            collection_override,
            &import_template,
            &mut established_collections,
        ) {
            eprintln!("{}: {error:#}", path.display());
            failures = failures.saturating_add(1);
        }
    }
    store
        .sync_all()
        .context("persisting import shard and collection summaries")?;
    if failures != 0 {
        bail!("import completed with {failures} failed input file(s)");
    }
    Ok(())
}

/// The built-in Matrix archive profile, used whenever `--template` is not
/// given. It is kept identical to `matrix-event-v1.json`'s identity,
/// membership, and payload rules, so a caller who never passes `--template`
/// still runs through the same generic extraction path as one who does.
fn default_matrix_import_template() -> CollectionTemplate {
    CollectionTemplate {
        name: "matrix-event-v1".into(),
        collection_kind: "room".into(),
        record_identity: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: "/room_id".into(),
            collection_type: MATRIX_ROOM_COLLECTION_TYPE,
            display_id_pointer: "/room_id".into(),
        },
    }
}

/// Compile the selected template into the executable profile the importer
/// runs identity and collection extraction through. With no `--template`,
/// this is [`default_matrix_import_template`]; the importer has one real
/// implementation, so a file-based template must describe that same profile
/// rather than silently taking effect as a different one.
fn validate_required_keys<'a>(
    template: &'a OwnedValue,
    path: &Path,
) -> anyhow::Result<(&'a str, &'a str)> {
    let required = [
        (["format"].as_slice(), "mtxdb.collection-template/v1"),
        (["name"].as_slice(), "matrix-event-v1"),
        (
            ["record", "identity", "extract", "kind"].as_slice(),
            "json-pointer-rfc-6901",
        ),
        (
            ["record", "identity", "extract", "path"].as_slice(),
            "/event_id",
        ),
        (
            ["collection", "membership", "extract", "kind"].as_slice(),
            "json-pointer-rfc-6901",
        ),
        (
            ["collection", "membership", "extract", "path"].as_slice(),
            "/room_id",
        ),
    ];
    let mut identity_pointer = None;
    let mut membership_pointer = None;
    for (keys, expected) in required {
        let actual = template_string_at(template, keys);
        if actual != Some(expected) {
            bail!(
                "template {} must set {} to {:?}, got {:?}",
                path.display(),
                keys.join("."),
                expected,
                actual
            );
        }
        if keys == ["record", "identity", "extract", "path"].as_slice() {
            identity_pointer = actual;
        } else if keys == ["collection", "membership", "extract", "path"].as_slice() {
            membership_pointer = actual;
        }
    }
    let identity_pointer = identity_pointer.context("record identity pointer missing")?;
    let membership_pointer = membership_pointer.context("collection membership pointer missing")?;
    Ok((identity_pointer, membership_pointer))
}

fn extract_display_id_pointer<'a>(
    template: &'a OwnedValue,
    membership_pointer: &'a str,
    path: &Path,
) -> anyhow::Result<&'a str> {
    let display_id_pointer = membership_pointer;
    if let Some(labels) = template.get("collection").and_then(|c| c.get("labels")) {
        let OwnedValue::Array(labels) = labels else {
            bail!(
                "template {} collection.labels must be an array of label objects; got {labels:?}",
                path.display()
            );
        };
        for label in labels.iter() {
            let OwnedValue::Object(obj) = label else {
                bail!(
                    "template {} declares a collection label that is not an object",
                    path.display()
                );
            };
            let Some(OwnedValue::String(name)) = obj.get("name") else {
                bail!(
                    "template {} declares a collection label without a name",
                    path.display()
                );
            };
            if name != "display_id" {
                continue;
            }
            let Some(OwnedValue::String(val)) = obj.get("value") else {
                bail!(
                    "template {} requests a display label with a non-string value {:?}, which \
                     the importer cannot persist; only the membership value is supported as a \
                     display identifier",
                    path.display(),
                    obj.get("value")
                );
            };
            if val != "membership-value" {
                bail!(
                    "template {} requests display label {val:?}, which the importer cannot \
                     persist; only the membership value is supported as a display identifier",
                    path.display()
                );
            }
        }
    }
    Ok(display_id_pointer)
}

fn compile_import_template(path: Option<&Path>) -> anyhow::Result<CollectionTemplate> {
    let Some(path) = path else {
        return Ok(default_matrix_import_template());
    };
    let mut bytes =
        fs::read(path).with_context(|| format!("reading template {}", path.display()))?;
    let template: OwnedValue = simd_json::to_owned_value(&mut bytes)
        .with_context(|| format!("template {} is not valid JSON", path.display()))?;
    let (identity_pointer, membership_pointer) = validate_required_keys(&template, path)?;
    if template_bool_at(&template, &["establishment", "required"]) != Some(true) {
        bail!(
            "template {} must require an establishment record",
            path.display()
        );
    }
    let node_id_algorithm = template_string_at(
        &template,
        ["record", "identity", "internal_key", "algorithm"].as_slice(),
    )
    .unwrap_or("sha2-256");
    let record_digest_algorithm = digest_algorithm_for(node_id_algorithm)
        .with_context(|| format!("template {} record identity internal_key", path.display()))?;
    // The collection-id derivation is now a fixed, core-owned function of a
    // compact type discriminator rather than a template-selected digest
    // algorithm. The Matrix import profile is the room type.
    let collection_type = MATRIX_ROOM_COLLECTION_TYPE;
    if let Some(policy) = template
        .get("record")
        .and_then(|r| r.get("payload"))
        .and_then(|p| p.get("policy"))
    {
        match policy {
            OwnedValue::String(s) if s == "retain-source" => {}
            OwnedValue::String(s) => {
                bail!(
                    "template {} payload policy {s:?} is not supported: only `retain-source` is \
                     implemented, so a `projection` policy must not silently compile to \
                     full-source retention",
                    path.display()
                );
            }
            other => {
                bail!(
                    "template {} payload policy must be a string naming a supported policy; got \
                     {other:?}",
                    path.display()
                );
            }
        }
    }
    // The importer persists no collection-label metadata, so a template that
    // requests a distinct display identifier cannot honor that request. Reject
    // it rather than silently dropping the configured label.
    let display_id_pointer = extract_display_id_pointer(&template, membership_pointer, path)?;

    Ok(CollectionTemplate {
        name: "matrix-event-v1".into(),
        collection_kind: "room".into(),
        record_identity: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: identity_pointer.to_owned(),
            },
            digest_algorithm: record_digest_algorithm,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: membership_pointer.to_owned(),
            collection_type,
            display_id_pointer: display_id_pointer.to_owned(),
        },
    })
}

/// Resolve a template's digest-algorithm name to the engine's [`DigestAlgorithm`].
fn digest_algorithm_for(algorithm: &str) -> anyhow::Result<DigestAlgorithm> {
    match algorithm {
        "sha2-256" => Ok(DigestAlgorithm::Sha256),
        other => bail!("unsupported internal-key algorithm `{other}`"),
    }
}

/// Digest algorithms `derive_template_key` knows how to compute. Checked at
/// template-compile time so an unsupported algorithm is rejected up front,
/// rather than surfacing mid-import on the first record that needs it.
fn validate_digest_algorithm(algorithm: &str) -> anyhow::Result<()> {
    digest_algorithm_for(algorithm).map(|_| ())
}

/// Resolve an RFC 6901 JSON pointer against a parsed event, returning the
/// string it names. Supports object-member and array-index segments; the
/// empty pointer denotes the whole document per the RFC. A non-empty
/// pointer must start with `/`.
///
/// Segments are validated per the RFC: `~` may only be followed by `0` or
/// `1` (invalid escapes like `~2` are rejected), and array indices must
/// not carry leading zeroes (except the single digit `0`).
fn extract_pointer_string<'a>(value: &'a OwnedValue, pointer: &str) -> Option<&'a str> {
    let mut current = value;
    if !pointer.is_empty() {
        let rest = pointer.strip_prefix('/')?;
        for raw_segment in rest.split('/') {
            // RFC 6901 §4: only ~0 and ~1 are valid escape sequences.
            // Reject any ~ not followed by 0 or 1 before unescaping.
            let mut chars = raw_segment.chars();
            while let Some(c) = chars.next() {
                if c == '~' {
                    match chars.next() {
                        Some('0' | '1') => {}
                        _ => return None,
                    }
                }
            }
            let segment = raw_segment.replace("~1", "/").replace("~0", "~");
            current = match current {
                OwnedValue::Object(object) => object.get(segment.as_str())?,
                OwnedValue::Array(array) => {
                    // RFC 6901: array indices are non-negative integers
                    // without leading zeroes; "0" is the sole zero.
                    let index = parse_array_index(&segment)?;
                    array.get(index)?
                }
                _ => return None,
            };
        }
    }
    match current {
        OwnedValue::String(value) => Some(value.as_str()),
        _ => None,
    }
}

/// Parse an RFC 6901 array index segment. Rejects leading zeroes (except
/// for the single digit `"0"`), non-digit characters, and signs. RFC 6901
/// permits only `0` or `[1-9][0-9]*`.
fn parse_array_index(segment: &str) -> Option<usize> {
    let bytes = segment.as_bytes();
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if bytes.len() > 1 && bytes[0] == b'0' {
        return None; // leading zero
    }
    segment.parse::<usize>().ok()
}

/// Derive mtxdb's internal 128-bit lookup key for one template-extracted value
/// using the named algorithm. The algorithm is assumed already validated by
/// [`validate_digest_algorithm`] (either at template-compile time or in
/// [`default_matrix_import_template`]).
///
/// # Key width
///
/// For `"sha2-256"` the full 32-byte SHA-256 digest is computed but only the
/// first 16 bytes are returned as the [`crate::storage::NodeId`]. The current
/// `NodeId` cannot provide a 256-bit payload digest; a full payload digest
/// belongs in pack-record metadata, computed by the storage layer on write.
fn derive_template_key(algorithm: &str, extracted: &str) -> anyhow::Result<[u8; 16]> {
    validate_digest_algorithm(algorithm)?;
    let hash = Sha256::digest(extracted.as_bytes());
    let mut id = [0u8; 16];
    id.copy_from_slice(&hash[..16]);
    Ok(id)
}

/// Run a template's record-identity rule against one event, producing the
/// node ID that storage keys it by.
fn template_node_id(
    template: &CollectionTemplate,
    event: &OwnedValue,
) -> anyhow::Result<Option<[u8; 16]>> {
    // The importer only derives identities from a source pointer today. Any
    // other policy must fail loudly rather than hash empty input and mint a
    // meaningless key.
    if !matches!(
        &template.record_identity.policy,
        FrameIdPolicy::Pointer { .. }
    ) {
        bail!(
            "record identity policy {:?} is not supported by the importer",
            template.record_identity.policy
        );
    }
    let resolve =
        |pointer: &str| extract_pointer_string(event, pointer).map(|s| s.as_bytes().to_vec());
    let input = FrameIdInput {
        payload: &[],
        descriptor: &[],
        canonical: None,
        resolve: &resolve,
    };
    let Some(digest) = frame_digest(
        &template.record_identity.policy,
        template.record_identity.digest_algorithm,
        &input,
    ) else {
        return Ok(None);
    };
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    Ok(Some(id))
}

/// Run a template's collection-key rule against an already-extracted
/// membership value (e.g. a Matrix `room_id`), producing the collection ID.
fn template_collection_id(template: &CollectionTemplate, membership_value: &str) -> [u8; 16] {
    derive_collection_id(
        template.collection_key.collection_type,
        membership_value.as_bytes(),
    )
}

fn template_string_at<'a>(value: &'a OwnedValue, keys: &[&str]) -> Option<&'a str> {
    let mut value = value;
    for key in keys {
        let OwnedValue::Object(object) = value else {
            return None;
        };
        value = object.get(*key)?;
    }
    match value {
        OwnedValue::String(value) => Some(value),
        _ => None,
    }
}

fn template_bool_at(value: &OwnedValue, keys: &[&str]) -> Option<bool> {
    let mut value = value;
    for key in keys {
        let OwnedValue::Object(object) = value else {
            return None;
        };
        value = object.get(*key)?;
    }
    match value {
        OwnedValue::Static(simd_json::StaticNode::Bool(value)) => Some(*value),
        _ => None,
    }
}

/// Export every live JSON record in one collection as a JSONL stream. A read-only
/// frame scan avoids stale persisted collection summaries and rebuilding every
/// collection's in-memory index just to enumerate one collection. Scanning packs
/// in pack-ID order lets a later physical copy replace an older one with the same
/// node ID.
fn cmd_export(cli: &Cli, collection: &str) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let pool_dir = selected_pool_dir(cli)?;
    let store = open_store_read_only(cli)?;
    if store.collection_index_info(&collection_id).is_none() {
        bail!("collection {collection} not found");
    }
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let mut shards = pool.all_shards();
    shards.sort_unstable_by_key(|(_, shard)| shard.pack_id);

    let mut seen = HashSet::new();
    let mut ordered_ids = Vec::new();
    for (_, shard) in &shards {
        for (candidate_collection, node_id, _) in mtxdb::packfile::scan_packfile(&shard.path)? {
            if candidate_collection == collection_id
                && seen.insert(node_id)
                && store.get(&collection_id, &node_id)?.is_some()
            {
                ordered_ids.push(node_id);
            }
        }
    }

    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let mut exported = 0_usize;
    for node_id in ordered_ids {
        if let Some(data) = store.get(&collection_id, &node_id)? {
            output.write_all(&data.bytes)?;
            output.write_all(b"\n")?;
            exported = exported.saturating_add(1);
        }
    }
    output.flush()?;
    eprintln!(
        "exported {exported} records from collection {}",
        hex_encode(&collection_id)
    );
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "the file-level import orchestration intentionally handles several import phases"
)]
fn cmd_import_file(
    store: &PackfileStorage,
    dir: &Path,
    path: &Path,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
) -> anyhow::Result<()> {
    let content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let is_jsonl = path
        .extension()
        .is_some_and(|extension| extension == "jsonl");

    if is_jsonl {
        let (events, detected_collection) =
            parse_jsonl_events(&content).or_else(|jsonl_error| {
                // Some existing DAG exports carry a `.jsonl` suffix despite
                // being a pretty-printed federation JSON document.
                parse_federation_events(&content).map_err(|federation_error| {
                    anyhow!(
                        "{jsonl_error}; also not a valid Matrix federation document: {federation_error}"
                    )
                })
            })?;
        if events.is_empty() {
            bail!("no events found in {}", path.display());
        }
        import_pdu_events(
            store,
            dir,
            path,
            &events,
            &[],
            detected_collection.as_deref(),
            collection_override,
            template,
            established_collections,
        )
    } else {
        let federation = parse_federation_input(&content)?;
        let detected_collection = federation
            .pdus
            .iter()
            .chain(federation.auth_chain.iter())
            .find_map(event_room_id)
            .map(str::to_owned);
        // `parse_federation_input` validates the document shape; this check
        // separately rejects a well-shaped but event-empty response.
        if federation.pdus.is_empty() && federation.auth_chain.is_empty() {
            bail!("no events found in {}", path.display());
        }

        // Verify auth chain edges: every auth_events reference must point
        // to a known event_id in either pdus or auth_chain.
        let dangling = verify_auth_chain_edges(&federation.pdus, &federation.auth_chain);
        if !dangling.is_empty() {
            for (source, target) in &dangling {
                eprintln!("warning: auth_events dangling reference: {source} -> {target}");
            }
        }

        // Import PDUs through the normal event-DAG path.
        let pdu_count = if federation.pdus.is_empty() {
            0
        } else {
            import_pdu_events(
                store,
                dir,
                path,
                &federation.pdus,
                &federation.auth_chain,
                detected_collection.as_deref(),
                collection_override,
                template,
                established_collections,
            )?;
            // Re-count: import_pdu_events already printed its summary.
            // We just need the count for the auth chain summary below.
            federation
                .pdus
                .iter()
                .filter(|ev| event_id(ev).is_some())
                .count() as u64
        };

        // Import auth chain events into the auth-chain shard pool.
        if !federation.auth_chain.is_empty() {
            // Derive the auth-chain pool dir from the event-dag pool dir.
            // pool_dir is {root}/pools/event-dag; auth-chain is {root}/pools/auth-chain.
            let auth_chain_dir = dir
                .parent()
                .map(|p| p.join("auth-chain"))
                .context("deriving auth-chain pool path")?;
            fs::create_dir_all(&auth_chain_dir)?;
            let auth_store =
                PackfileStorage::open(auth_chain_dir).context("opening auth-chain store")?;
            let mut auth_count = 0u64;
            let mut auth_skipped = 0u64;
            // Auth-chain events may span multiple rooms (collections). Group
            // them, then batch-probe each group with `get_many` and insert
            // only the missing records via `put_many` — one read plan and one
            // index/pack generation per collection instead of N individual
            // `put` round trips.
            let mut by_collection: HashMap<[u8; 16], Vec<ImportBatchEntry>> = HashMap::new();
            for ev in &federation.auth_chain {
                let Some(incoming_event_id) = event_id(ev) else {
                    auth_skipped = auth_skipped.saturating_add(1);
                    continue;
                };
                let Some(id_bytes) = template_node_id(template, ev)? else {
                    auth_skipped = auth_skipped.saturating_add(1);
                    continue;
                };
                let collection_id = event_room_id(ev).map_or([0u8; 16], |room_id| {
                    template_collection_id(template, room_id)
                });
                let event_bytes = ev.encode().into_bytes();
                by_collection.entry(collection_id).or_default().push((
                    id_bytes,
                    incoming_event_id.to_owned(),
                    NodeData::new(bytes::Bytes::from(event_bytes)),
                ));
            }
            for (collection_id, entries) in by_collection {
                // Same dedup rule as the PDU path: the 128-bit node ID is a
                // truncated hash, so two distinct event IDs mapping to one key
                // is a collision to reject, not collapse; a repeat with the
                // same event_id is the same logical event.
                let mut first: Vec<ImportBatchEntry> = Vec::with_capacity(entries.len());
                let mut seen_ids: HashMap<[u8; 16], String> = HashMap::with_capacity(entries.len());
                for (id_bytes, incoming_event_id, data) in entries {
                    if let Some(first_event_id) = seen_ids.get(&id_bytes) {
                        if first_event_id != &incoming_event_id {
                            bail!(
                                "node ID {} maps to multiple event IDs in the input; refusing to merge them",
                                hex_encode(&id_bytes)
                            );
                        }
                        continue;
                    }
                    seen_ids.insert(id_bytes, incoming_event_id.clone());
                    first.push((id_bytes, incoming_event_id, data));
                }
                let probe_ids: Vec<[u8; 16]> = first.iter().map(|(id, _, _)| *id).collect();
                let existing = auth_store.get_many(&collection_id, &probe_ids)?;
                let mut to_write: Vec<([u8; 16], NodeData)> = Vec::new();
                for ((id_bytes, incoming_event_id, data), existing_opt) in
                    first.into_iter().zip(existing)
                {
                    if let Some(existing_record) = existing_opt {
                        reject_cross_record_collision(
                            &id_bytes,
                            &incoming_event_id,
                            &existing_record,
                        )?;
                    } else {
                        to_write.push((id_bytes, data));
                    }
                }
                if !to_write.is_empty() {
                    auth_store.put_many(&collection_id, &to_write)?;
                    // Deliberate semantic change from the pre-batching loop,
                    // which counted every accepted input event including ones
                    // already on disk: `auth_count` now reports only genuinely
                    // newly-written auth-chain events, so "imported N" means
                    // records actually appended, and a reimport reports 0.
                    auth_count = auth_count.saturating_add(to_write.len() as u64);
                }
            }
            auth_store.sync_all()?;
            eprintln!(
                "imported {auth_count} auth-chain events ({} dangling references)",
                dangling.len()
            );
            if auth_skipped > 0 {
                eprintln!("skipped {auth_skipped} auth-chain events (missing event_id)");
            }
        }

        let _ = pdu_count;
        Ok(())
    }
}

/// A decoded import entry keyed by node ID, retaining the source event ID for
/// collision validation before the record is written.
type ImportBatchEntry = ([u8; 16], String, NodeData);

/// An already-present record under the same node ID must carry the same
/// `event_id` as the incoming event, or the two genuinely differ and
/// overwriting would lose data. Decode the stored record and refuse on a
/// mismatch. The 128-bit node ID is a truncated hash, so such a cross-record
/// collision is part of the format's threat model.
fn reject_cross_record_collision(
    id_bytes: &[u8; 16],
    incoming_event_id: &str,
    existing_record: &NodeData,
) -> anyhow::Result<()> {
    let mut existing_bytes = existing_record.bytes.to_vec();
    let existing_event_id = simd_json::to_owned_value(&mut existing_bytes)
        .ok()
        .and_then(|event| event_id(&event).map(str::to_owned));
    if existing_event_id.as_deref() != Some(incoming_event_id) {
        bail!(
            "node ID {} collides with a different event_id; refusing to overwrite it",
            hex_encode(id_bytes)
        );
    }
    Ok(())
}

/// Import PDU events through the normal event-DAG path.
#[allow(
    clippy::too_many_arguments,
    reason = "all parameters are necessary for the import flow"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the PDU import phase intentionally owns validation, batching, and state-group work"
)]
fn import_pdu_events(
    store: &PackfileStorage,
    dir: &Path,
    path: &Path,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
) -> anyhow::Result<()> {
    if events.is_empty() {
        bail!("no events found in {}", path.display());
    }
    let mut event_count = 0u64;
    let mut skipped = 0u64;
    let mut already_present = 0u64;
    let mut already_present_ids = Vec::with_capacity(3);

    let (collection_id, batch_has_create) = resolve_import_collection(
        events,
        detected_collection,
        collection_override,
        template,
        established_collections,
    )?;

    let collection_hex = hex_encode(&collection_id);

    // Decode and dedup in one pass, retaining the first occurrence of each
    // node ID directly — no intermediate full-size buffer, so a large
    // federation response's bodies are held once, not twice. The 128-bit
    // node ID is a truncated hash, so two distinct event IDs can in
    // principle truncate to the same key; that is a collision and must be
    // rejected rather than silently collapsed. A same-ID repeat that carries
    // the same event_id is the same logical event and counts as already
    // present — the old per-event loop would have found the record its first
    // occurrence just stored.
    let first: Vec<([u8; 16], String, NodeData)> = {
        let mut first = Vec::with_capacity(events.len());
        let mut seen_ids: HashMap<[u8; 16], String> = HashMap::with_capacity(events.len());
        for ev in events {
            let Some(incoming_event_id) = event_id(ev) else {
                skipped = skipped.saturating_add(1);
                continue;
            };
            let Some(id_bytes) = template_node_id(template, ev)? else {
                skipped = skipped.saturating_add(1);
                continue;
            };

            if let Some(first_event_id) = seen_ids.get(&id_bytes) {
                if first_event_id != incoming_event_id {
                    bail!(
                        "node ID {} maps to multiple event IDs in the input; refusing to merge them",
                        hex_encode(&id_bytes)
                    );
                }
                already_present = already_present.saturating_add(1);
                if already_present_ids.len() < 3 {
                    already_present_ids.push(incoming_event_id.to_owned());
                }
                continue;
            }
            seen_ids.insert(id_bytes, incoming_event_id.to_owned());
            let event_bytes = ev.encode().into_bytes();
            first.push((
                id_bytes,
                incoming_event_id.to_owned(),
                NodeData::new(bytes::Bytes::from(event_bytes)),
            ));
        }
        first
    };

    // One batched index probe for dedup/collision checks, then one `put_many`
    // for only the genuinely-new records — instead of N get+put round trips,
    // one index/pack generation, and a locality-ordered rather than
    // offset-coalesced set of candidate reads (get_many sorts candidates but
    // does not merge ranges).
    let probe_ids: Vec<[u8; 16]> = first.iter().map(|(id, _, _)| *id).collect();
    let existing = store.get_many(&collection_id, &probe_ids)?;
    let mut to_write: Vec<([u8; 16], NodeData)> = Vec::new();
    for ((id_bytes, incoming_event_id, data), existing_opt) in first.into_iter().zip(existing) {
        if let Some(existing_record) = existing_opt {
            reject_cross_record_collision(&id_bytes, &incoming_event_id, &existing_record)?;
            already_present = already_present.saturating_add(1);
            if already_present_ids.len() < 3 {
                already_present_ids.push(incoming_event_id);
            }
        } else {
            to_write.push((id_bytes, data));
        }
    }
    if !to_write.is_empty() {
        store.put_many(&collection_id, &to_write)?;
        event_count = event_count.saturating_add(to_write.len() as u64);
    }

    if batch_has_create {
        established_collections.insert(collection_id);
        let details = matrix_room_details_from_events(events);
        if let Err(error) = persist_matrix_room_details(dir, &collection_id, &details) {
            eprintln!("warning: unable to cache Matrix room metadata: {error}");
        }
    }

    // Compute state groups from the event DAG and persist the mappings.
    let state_groups = match compute_state_groups(events, auth_chain) {
        Ok(groups) => groups,
        Err(cycle_events) => {
            eprintln!(
                "warning: skipping state-group computation: DAG has missing parents or cycles involving {} event(s)",
                cycle_events.len()
            );
            HashMap::new()
        }
    };
    if !state_groups.is_empty() {
        use mtxdb::auxiliary::AuxiliaryIndex;
        let aux = AuxiliaryIndex::open(store, "matrix-state-groups");
        let mut state_count = 0u64;
        for (event_id, state_group_id) in &state_groups {
            // Key: event_id, Value: state_group_id (base64url).
            if let Err(error) = aux.put(event_id.as_bytes(), state_group_id.as_bytes()) {
                eprintln!("warning: unable to persist state group for {event_id}: {error}");
            } else {
                state_count = state_count.saturating_add(1);
            }
        }
        if state_count > 0 {
            eprintln!("computed {state_count} state groups for collection {collection_hex}");
        }
    }

    if let Some(room_id) = detected_collection {
        eprintln!("imported {event_count} events to collection {collection_hex} (room {room_id})");
    } else {
        eprintln!("imported {event_count} events to collection {collection_hex}");
    }
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

/// Resolve a Matrix input's room collection and prove that it is established
/// before any record from the input is written.
fn resolve_import_collection(
    events: &[OwnedValue],
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &HashSet<[u8; 16]>,
) -> anyhow::Result<([u8; 16], bool)> {
    let room_ids: HashSet<_> = events.iter().filter_map(event_room_id).collect();
    if room_ids.len() > 1 {
        bail!("input contains events for multiple room IDs; split it into one room per input");
    }
    let room_id = room_ids.into_iter().next();
    let collection_id = if let Some(value) = collection_override {
        let collection_id = parse_collection_id(value)?;
        if let Some(room_id) = room_id {
            let expected = template_collection_id(template, room_id);
            if collection_id != expected {
                bail!(
                    "--collection {value} does not match Matrix room_id {room_id}; refusing to mix room data into another collection"
                );
            }
        }
        collection_id
    } else {
        let room_id = room_id.or(detected_collection).with_context(|| {
            let first_event = events
                .first()
                .and_then(event_id)
                .unwrap_or("<missing event_id>");
            format!(
                "could not detect collection_id (first event: {first_event}); pass --collection for this input"
            )
        })?;
        template_collection_id(template, room_id)
    };
    let batch_has_create = room_id.is_some_and(|room_id| matrix_batch_has_create(events, room_id));
    if !batch_has_create && !established_collections.contains(&collection_id) {
        let room = room_id.unwrap_or("the selected collection");
        bail!(
            "refusing to import events for {room}: no valid m.room.create event is in this input or already on disk"
        );
    }
    Ok((collection_id, batch_has_create))
}

/// Return every collection that already contains a valid Matrix establishing
/// record. This deliberately decodes pack payloads: the collection directory
/// says nothing about whether a room has a create event.
fn matrix_create_collections_on_disk(dir: &Path) -> anyhow::Result<HashSet<[u8; 16]>> {
    let mut established = HashSet::new();
    if !dir.exists() || glob_pack_files(dir)?.is_empty() {
        return Ok(established);
    }
    let deleted_path = dir.join("deleted.collections");
    let deleted_bytes = fs::read(&deleted_path).unwrap_or_default();
    let deleted: HashSet<[u8; 16]> = deleted_bytes
        .chunks_exact(16)
        .map(|chunk| {
            let mut id = [0u8; 16];
            id.copy_from_slice(chunk);
            id
        })
        .collect();
    let pool = ShardPool::open_read_only(dir.into())
        .context("opening shard store for Matrix create validation")?;
    for (_, shard) in pool.all_shards() {
        let file = fs::File::open(&shard.path)?;
        let mut reader = BufReader::new(file);
        if mtxdb::packfile::read_header(&mut reader)?.is_none() {
            continue;
        }
        while let Some(record) = mtxdb::packfile::read_record(&mut reader)? {
            if deleted.contains(&record.collection_id) {
                continue;
            }
            let mut bytes = record.data.to_vec();
            let Ok(event) = simd_json::to_owned_value(&mut bytes) else {
                continue;
            };
            if matrix_create_event_id(&event).is_some() {
                established.insert(record.collection_id);
            }
        }
    }
    Ok(established)
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
    let federation = parse_federation_input(content)?;
    let events: Vec<OwnedValue> = federation
        .pdus
        .into_iter()
        .chain(federation.auth_chain)
        .collect();
    let detected_collection = events.iter().find_map(event_room_id).map(str::to_owned);
    Ok((events, detected_collection))
}

/// Parsed Matrix federation response with PDUs and auth chain kept separate.
struct FederationInput {
    pdus: Vec<OwnedValue>,
    auth_chain: Vec<OwnedValue>,
}

const FEDERATION_JSON_HINT: &str =
    "expected Matrix federation JSON with a `pdus` array, an `auth_chain` array, or both; JSONL files must contain one event per line";

fn parse_federation_input(content: &[u8]) -> anyhow::Result<FederationInput> {
    let mut bytes = content.to_vec();
    let val: OwnedValue = simd_json::to_owned_value(&mut bytes).context("invalid JSON")?;
    let has_pdu_field = val.get("pdus").is_some();
    let has_auth_chain_field = val.get("auth_chain").is_some();
    if !has_pdu_field && !has_auth_chain_field {
        bail!("{FEDERATION_JSON_HINT}");
    }
    let pdus = match val.get("pdus") {
        Some(value) => value
            .as_array()
            .cloned()
            .ok_or_else(|| anyhow!("{FEDERATION_JSON_HINT} (`pdus` must be an array)"))?,
        None => Vec::new(),
    };
    let auth_chain = match val.get("auth_chain") {
        Some(value) => value
            .as_array()
            .cloned()
            .ok_or_else(|| anyhow!("{FEDERATION_JSON_HINT} (`auth_chain` must be an array)"))?,
        None => Vec::new(),
    };
    Ok(FederationInput { pdus, auth_chain })
}

/// Verify that every `auth_events` reference in a batch points to a known
/// `event_id` (present in either pdus or `auth_chain`). Returns the list of
/// dangling references for the caller to decide how to handle.
fn verify_auth_chain_edges(
    pdus: &[OwnedValue],
    auth_chain: &[OwnedValue],
) -> Vec<(String, String)> {
    let mut known = HashSet::new();
    for ev in pdus.iter().chain(auth_chain.iter()) {
        if let Some(eid) = event_id(ev) {
            known.insert(eid.to_owned());
        }
    }
    let mut dangling = Vec::new();
    for ev in pdus.iter().chain(auth_chain.iter()) {
        let Some(eid) = event_id(ev) else {
            continue;
        };
        let OwnedValue::Object(fields) = ev else {
            continue;
        };
        let Some(OwnedValue::Array(auth_events)) = fields.get("auth_events") else {
            continue;
        };
        for reference in auth_events.iter() {
            let target = match reference {
                OwnedValue::String(s) => Some(s.as_str()),
                OwnedValue::Array(parts) => parts.first().and_then(|v| v.as_str()),
                _ => None,
            };
            if let Some(target) = target {
                if !known.contains(target) {
                    dangling.push((eid.to_owned(), target.to_owned()));
                }
            }
        }
    }
    dangling
}

/// Build an in-memory event DAG from a set of events.
///
/// Derive the in-memory DAG `short_id` for an event: the first 8 bytes of
/// SHA-256(`event_id`) read as a little-endian `u64`.
///
/// This is a purely local, dense-ish key for the in-memory frontier. It is
/// independent of the on-disk [`mtxdb::NodeId`] (which is the first 16 bytes
/// of SHA-256 of the extracted identity) and need not be stable across
/// algorithm changes.
fn event_short_id(event_id: &str) -> u64 {
    let hash = Sha256::digest(event_id.as_bytes());
    let mut short_id_bytes = [0u8; 8];
    short_id_bytes.copy_from_slice(&hash[..8]);
    u64::from_le_bytes(short_id_bytes)
}

/// Build an in-memory event DAG from a set of events.
///
/// Each event is hashed to a `u64` `short_id` via [`event_short_id`], then
/// inserted into an [`ActiveRoomFrontier`] with its `prev_events` and
/// `auth_events` edges resolved to the same `short_id` space.
///
/// Returns the frontier and the bidirectional id maps.
fn build_event_dag(
    events: &[OwnedValue],
) -> (
    mtxdb::dag::ActiveRoomFrontier,
    HashMap<String, u64>,
    HashMap<u64, String>,
) {
    use mtxdb::dag::ActiveRoomFrontier;

    let mut frontier = ActiveRoomFrontier::new();
    let mut id_map: HashMap<String, u64> = HashMap::new();
    for ev in events {
        if let Some(eid) = event_id(ev) {
            let short_id = event_short_id(eid);
            id_map.insert(eid.to_owned(), short_id);
        }
    }
    for ev in events {
        let Some(eid) = event_id(ev) else {
            continue;
        };
        let Some(&short_id) = id_map.get(eid) else {
            continue;
        };
        let raw_prevs = extract_event_edge_ids(ev, "prev_events", &id_map);
        let raw_auths = extract_event_edge_ids(ev, "auth_events", &id_map);
        frontier.insert_event(short_id, &raw_prevs, &raw_auths);
    }
    let reverse_map: HashMap<u64, String> = id_map
        .iter()
        .map(|(eid, &sid)| (sid, eid.clone()))
        .collect();
    (frontier, id_map, reverse_map)
}

/// Extract the short IDs for an edge array (`prev_events` or `auth_events`).
fn extract_event_edge_ids(
    event: &OwnedValue,
    field: &str,
    id_map: &HashMap<String, u64>,
) -> Vec<u64> {
    let OwnedValue::Object(fields) = event else {
        return Vec::new();
    };
    let Some(OwnedValue::Array(arr)) = fields.get(field) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            let eid = match item {
                OwnedValue::String(s) => Some(s.as_str()),
                OwnedValue::Array(parts) => parts.first().and_then(|v| v.as_str()),
                _ => None,
            };
            eid.and_then(|eid| id_map.get(eid).copied())
        })
        .collect()
}

/// Topologically sort the DAG nodes in `frontier` using Kahn's algorithm.
///
/// Returns events in topological order (parents before children), or
/// `Err` with the list of events involved in a cycle or whose parents
/// are missing from the DAG.
fn topo_sort_dag(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    reverse_map: &HashMap<u64, String>,
) -> Result<Vec<usize>, Vec<String>> {
    let n = frontier.nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency: parent_idx -> list of child local_ids.
    let mut in_degree = vec![0usize; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (idx, _node) in frontier.nodes.iter().enumerate() {
        for edge in frontier.prev_edges(idx) {
            if edge.is_resident() {
                let parent_idx = edge.arena_index();
                children[parent_idx].push(idx);
                in_degree[idx] = in_degree[idx]
                    .checked_add(1)
                    .expect("DAG in-degree overflow");
            } else {
                // Disk-resident edge: parent not in this batch.
                in_degree[idx] = in_degree[idx]
                    .checked_add(1)
                    .expect("DAG in-degree overflow");
            }
        }
    }

    // Seed the queue with nodes that have zero in-degree.
    let mut queue: Vec<usize> = Vec::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push(idx);
        }
    }

    let mut sorted = Vec::with_capacity(n);
    while let Some(idx) = queue.pop() {
        sorted.push(idx);
        for &child in &children[idx] {
            in_degree[child] = in_degree[child]
                .checked_sub(1)
                .expect("DAG in-degree underflow");
            if in_degree[child] == 0 {
                queue.push(child);
            }
        }
    }

    if sorted.len() == n {
        Ok(sorted)
    } else {
        // Collect events involved in cycles or with missing parents.
        let mut problematic: Vec<String> = Vec::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg > 0 {
                let eid = reverse_map
                    .get(&frontier.nodes[idx].short_id)
                    .cloned()
                    .unwrap_or_else(|| format!("short:{}", frontier.nodes[idx].short_id));
                problematic.push(eid);
            }
        }
        problematic.sort();
        Err(problematic)
    }
}

/// Compute state groups for a set of events by walking the event DAG in
/// topological order.
///
/// `events` are the PDUs; `auth_chain` events (if any) are included in
/// the DAG so that `auth_edges` resolve correctly, but only PDU state
/// events contribute to the state set.
///
/// Returns a map from `event_id` -> `state_group_id` (base64url-encoded
/// SHA-256 digest of the state set). Each event inherits the state from
/// its `prev_events` and applies its own state change (if it is a state
/// event with `state_key`).
///
/// # Errors
/// Returns `Err` with involved event IDs if the DAG contains a cycle or
/// references parents absent from the combined event set.
fn compute_state_groups(
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
) -> Result<HashMap<String, String>, Vec<String>> {
    let all_events: Vec<&OwnedValue> = events.iter().chain(auth_chain.iter()).collect();
    let all_owned: Vec<OwnedValue> = all_events.into_iter().cloned().collect();
    let (frontier, id_map, reverse_map) = build_event_dag(&all_owned);
    if frontier.is_empty() {
        return Ok(HashMap::new());
    }

    let sorted = topo_sort_dag(&frontier, &reverse_map)?;

    let mut events_by_sid: HashMap<u64, &OwnedValue> = HashMap::new();
    for ev in &all_owned {
        if let Some(eid) = event_id(ev) {
            if let Some(&sid) = id_map.get(eid) {
                events_by_sid.insert(sid, ev);
            }
        }
    }

    let mut state_at: HashMap<u64, StateSet> = HashMap::new();
    let mut result: HashMap<String, String> = HashMap::new();

    for &idx in &sorted {
        let node = &frontier.nodes[idx];
        let short_id = node.short_id;

        // Merge state from all prev_events.
        let mut merged = StateSet::new();
        for edge in frontier.prev_edges(idx) {
            let parent_id = if edge.is_resident() {
                frontier.nodes[edge.arena_index()].short_id
            } else {
                continue;
            };
            if let Some(parent_state) = state_at.get(&parent_id) {
                merged.merge(parent_state);
            }
        }

        // If this is a state event, apply the state change.
        if let Some(ev) = events_by_sid.get(&short_id) {
            if is_state_event(ev) {
                let key = state_key(ev);
                let state_type = event_string_field(ev, "type").unwrap_or("");
                let eid = event_id(ev).unwrap_or("").to_owned();
                merged.set(state_type, &key, eid);
            }
        }

        let state_group_id = merged.digest_base64url();
        if let Some(eid) = reverse_map.get(&short_id) {
            result.insert(eid.clone(), state_group_id);
        }
        state_at.insert(short_id, merged);
    }

    Ok(result)
}

/// A set of state events keyed by (type, `state_key`).
#[derive(Debug, Clone, Default)]
struct StateSet {
    entries: HashMap<(String, String), String>,
}

impl StateSet {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn set(&mut self, event_type: &str, state_key: &str, event_id: String) {
        self.entries
            .insert((event_type.to_owned(), state_key.to_owned()), event_id);
    }

    /// Merge another state set into this one. When both sets contain the
    /// same (type, `state_key`) with different event IDs the conflict is
    /// logged; the existing entry is kept (first-wins) which matches the
    /// event ordering enforced by the topological sort caller.
    fn merge(&mut self, other: &StateSet) {
        for (key, eid) in &other.entries {
            self.entries
                .entry(key.clone())
                .or_insert_with(|| eid.clone());
        }
    }

    /// Deterministic hash of the state set for use as a state-group ID.
    ///
    /// The digest is SHA-256 over the sorted `(type, state_key,
    /// event_id)` entries. This is **not** an `LtHash`; it is a standard
    /// collision-resistant hash suitable for identifying state sets.
    fn digest_base64url(&self) -> String {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort();

        let mut hasher_input = Vec::new();
        for ((event_type, state_key), event_id) in &entries {
            hasher_input.extend_from_slice(event_type.as_bytes());
            hasher_input.push(0);
            hasher_input.extend_from_slice(state_key.as_bytes());
            hasher_input.push(0);
            hasher_input.extend_from_slice(event_id.as_bytes());
            hasher_input.push(0);
        }
        let hash = Sha256::digest(&hasher_input);
        URL_SAFE_NO_PAD.encode(&hash[..])
    }
}

/// Check if an event is a state event (has a `state_key` field).
fn is_state_event(event: &OwnedValue) -> bool {
    let OwnedValue::Object(fields) = event else {
        return false;
    };
    matches!(fields.get("state_key"), Some(OwnedValue::String(_)))
}

/// Extract the `state_key` from a state event.
fn state_key(event: &OwnedValue) -> String {
    let OwnedValue::Object(fields) = event else {
        return String::new();
    };
    match fields.get("state_key") {
        Some(OwnedValue::String(s)) => s.clone(),
        _ => String::new(),
    }
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

/// A create event establishes a Matrix room only when it is a state event
/// with the empty state key and a stable event ID.
fn matrix_create_event_id(event: &OwnedValue) -> Option<&str> {
    (event_string_field(event, "type") == Some("m.room.create")
        && event_string_field(event, "state_key") == Some(""))
    .then(|| event_id(event))
    .flatten()
}

/// Matrix v1/v2 commonly encode auth events as `[event_id, hashes]`; later
/// versions encode them as event-ID strings. Accept both wire forms.
fn event_auth_references(event: &OwnedValue, target: &str) -> bool {
    let OwnedValue::Object(fields) = event else {
        return false;
    };
    let Some(OwnedValue::Array(auth_events)) = fields.get("auth_events") else {
        return false;
    };
    auth_events.iter().any(|reference| match reference {
        OwnedValue::String(event_id) => event_id == target,
        OwnedValue::Array(parts) => {
            matches!(parts.first(), Some(OwnedValue::String(event_id)) if event_id == target)
        }
        _ => false,
    })
}

/// A v1–v11 create event normally carries `room_id`; v12 permits the create
/// event to omit it, so associate that create through an auth edge from a room
/// event in the same batch.
fn matrix_batch_has_create(events: &[OwnedValue], room_id: &str) -> bool {
    events.iter().any(|event| {
        let Some(create_id) = matrix_create_event_id(event) else {
            return false;
        };
        event_room_id(event) == Some(room_id)
            || events.iter().any(|candidate| {
                event_room_id(candidate) == Some(room_id)
                    && event_auth_references(candidate, create_id)
            })
    })
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

/// Build `MatrixRoomDetails` directly from a parsed import batch, with no
/// disk access: the caller already knows the batch establishes the room
/// (`matrix_batch_has_create`), so both fields can be read out of the events
/// already sitting in memory instead of round-tripping through a shard scan
/// the way `matrix_room_details_from_shards` has to for pre-existing data.
fn matrix_room_details_from_events(events: &[OwnedValue]) -> MatrixRoomDetails {
    let mut details = MatrixRoomDetails::default();
    for event in events {
        if details.matrix_room_id.is_none() {
            details.matrix_room_id = event_room_id(event).map(str::to_owned);
        }
        if details.create.is_none() {
            details.create = matrix_create_details(event);
        }
        if details.matrix_room_id.is_some() && details.create.is_some() {
            break;
        }
    }
    details
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
    packs: &[String],
    all: bool,
    roots: &[String],
    topo: bool,
) -> anyhow::Result<()> {
    let target = match (collection, packs.is_empty(), all) {
        (Some(collection), true, false) => {
            RepackTarget::Collection(parse_collection_id(collection)?)
        }
        (None, false, false) => RepackTarget::Packs(parse_pack_selectors(packs)?),
        (None, true, true) => RepackTarget::All,
        // Clap rejects the both-targets case through `conflicts_with`; this
        // branch gives the missing-target case a readable diagnostic.
        _ => {
            return Err(anyhow!(
                "exactly one of --collection <collection> | --pack <pack> | --all is required"
            ))
        }
    };

    if !roots.is_empty() {
        match &target {
            RepackTarget::Packs(_) | RepackTarget::All => {
                bail!(
                    "--root requires --collection — live roots are per-collection, not meaningful for --pack"
                );
            }
            RepackTarget::Collection(_) if !topo => {
                bail!("--root requires --topo; without --topo there are no edges so only the specified roots would be kept");
            }
            RepackTarget::Collection(_) => {
                // Allowed
            }
        }
    }
    if topo {
        println!("warning: edge extraction is approximate; prev_events event IDs are not resolved to stored node hashes");
    } else {
        println!("warning: --topo without --root means no GC; all records preserved");
    }
    cmd_repack_target(cli, &target, topo, roots)
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
    roots: &[String],
) -> anyhow::Result<Option<RepackPreview>> {
    println!("preflight: scanning packs and rebuilding live indexes...");
    let preview_store = open_store_read_only(cli)?;
    if let RepackTarget::Collection(collection_id) = target {
        if !roots.is_empty() {
            let mut root_ids = Vec::new();
            for root in roots {
                root_ids.push(matrix_event_node_id(root)?);
            }
            preview_store.set_live_roots(collection_id, root_ids);
        }
    }
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

fn cmd_repack_target(
    cli: &Cli,
    target: &RepackTarget,
    topo: bool,
    roots: &[String],
) -> anyhow::Result<()> {
    let Some(preview) = repack_preview(cli, target, topo, roots)? else {
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
    if let RepackTarget::Collection(collection_id) = target {
        if !roots.is_empty() {
            let mut root_ids = Vec::new();
            for root in roots {
                root_ids.push(matrix_event_node_id(root)?);
            }
            store.set_live_roots(collection_id, root_ids);
        }
    }
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

fn extract_matrix_edges(_hash: &[u8; 16], data: &[u8]) -> Vec<mtxdb::NodeId> {
    let mut input = data.to_vec();
    let val: simd_json::OwnedValue = match simd_json::to_owned_value(&mut input) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut edges = Vec::new();
    if let Some(prev) = val.get("prev_events").and_then(|v| v.as_array()) {
        for ev in prev {
            let event_id = if let Some(s) = ev.as_str() {
                Some(s)
            } else if let Some(arr) = ev.as_array() {
                arr.first().and_then(|v| v.as_str())
            } else {
                None
            };
            if let Some(s) = event_id {
                // Matrix template compilation currently permits sha2-256,
                // so edge extraction uses the same derivation as import. The
                // storage callback cannot return an error; if that supported
                // identity configuration ever changes, omit this edge rather
                // than panicking in the CLI.
                if let Ok(id) = matrix_event_node_id(s) {
                    edges.push(id);
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
            .map_or(0, |(len, _, _)| len);
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
fn cmd_sync(cli: &Cli, all: bool) -> anyhow::Result<()> {
    if all {
        let db_layout = open_layout(cli)?;
        for shard_type in ShardType::ALL {
            let dir = pool_dir(&db_layout, shard_type)?;
            // The database layout creates every named pool eagerly, but a
            // writer open creates its first empty pack. `sync --all` is
            // metadata maintenance, not an instruction to materialize every
            // possible pool, so leave unused pools untouched.
            if glob_pack_files(&dir)?.is_empty() {
                eprintln!("{}: skipped (no packfiles)", shard_type.as_str());
                continue;
            }
            let store = PackfileStorage::open(dir).context("failed to open store")?;
            store.sync_all()?;
            eprintln!(
                "{}: synced: persisted shard IO stats and shard\u{2192}collection directory",
                shard_type.as_str()
            );
        }
        return Ok(());
    }
    let store = open_store(cli)?;
    store.sync_all()?;
    eprintln!("synced: persisted shard IO stats and shard\u{2192}collection directory");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_event_dag, cmd_collections, cmd_get, cmd_import_file, cmd_info, cmd_scan, cmd_shards,
        cmd_stats, cmd_sync, compile_import_template, compute_state_groups,
        decode_event_json_record, decode_hamt_node, decode_hamt_root,
        default_matrix_import_template, derive_template_key, event_id, event_room_id,
        event_short_id, extract_pointer_string, fmt_disk_megabytes, fmt_megabytes, glob_pack_files,
        import_pdu_events, interleaving_worth_noting, matrix_batch_has_create,
        matrix_create_details, matrix_room_collection_id, parse_federation_input,
        parse_pack_id_selector, parse_pack_selectors, pretty_print_payload,
        resolve_import_collection, scan_payload_suffix, synapse_event_node_id,
        template_collection_id, template_node_id, verify_auth_chain_edges, CollectionTemplate,
        StateSet, MATRIX_ROOM_COLLECTION_TYPE,
    };
    use crate::{Cli, Commands};
    use bytes::Bytes;
    use mtxdb::packfile::storage::PackfileStorage;
    use mtxdb::storage::{NodeData, StorageEngine};
    use mtxdb::template::{CollectionKeyRule, FrameIdPolicy, PayloadPolicy, RecordIdentityRule};
    use mtxdb::{DatabaseLayout, DigestAlgorithm, ShardType};
    use sha2::{Digest, Sha256};
    use simd_json::prelude::Writable;
    use simd_json::OwnedValue;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use base64::Engine;

    fn owned_value(json: &str) -> OwnedValue {
        let mut bytes = json.as_bytes().to_vec();
        simd_json::to_owned_value(&mut bytes).expect("valid JSON fixture")
    }

    #[test]
    fn state_scan_decodes_big_endian_state_group_payload() {
        assert_eq!(
            scan_payload_suffix(&2u64.to_be_bytes(), ShardType::State),
            Some("PTR: 0x0000000000000002".to_owned())
        );
        assert_eq!(
            scan_payload_suffix(&2u64.to_be_bytes(), ShardType::EventDag),
            Some("8 bytes (undecodable, magic=0x00000000)".to_owned())
        );
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mtxdb-cli-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn compile_reject(template: &[u8]) -> anyhow::Result<CollectionTemplate> {
        let dir = unique_temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("template.json");
        std::fs::write(&path, template).unwrap();
        let result = compile_import_template(Some(&path));
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn interleaving_note_fires_only_on_average_fragmentation() {
        // The user's state-pool case: ~10 extra runs per collection.
        assert!(interleaving_worth_noting(10319, 104_933));
        // One extra run per collection on average is the threshold.
        assert!(interleaving_worth_noting(250, 250));
        // A few collections each split a couple of times is noise.
        assert!(!interleaving_worth_noting(250, 200));
        // No interleaving at all, or tiny stores, stays quiet.
        assert!(!interleaving_worth_noting(250, 0));
        assert!(!interleaving_worth_noting(1, 0));
        // Degenerate guards (unit-test the clamp, not realistic data).
        assert!(interleaving_worth_noting(0, 1));
        assert!(!interleaving_worth_noting(0, 0));
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
    fn sync_all_does_not_materialize_empty_pools() {
        let dir = unique_temp_dir();
        let layout = DatabaseLayout::open(dir.clone()).unwrap();
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: Some(ShardType::State),
            namespace: None,
            command: Commands::Sync { all: true },
        };

        cmd_sync(&cli, true).unwrap();

        for shard_type in ShardType::ALL {
            let pool = layout.pool_dir_read_only(shard_type).unwrap();
            assert!(
                glob_pack_files(&pool).unwrap().is_empty(),
                "sync --all must not create a pack in the empty {} pool",
                shard_type.as_str()
            );
        }
        drop(layout);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shard_types_yields_all_three_when_no_type_is_selected() {
        // `-t all` parses to `shard_type: None` (main.rs's `"all" => None`
        // match arm) -- this is the exact case that used to make
        // `require_shard_type` reject `collections`/`shards` even though
        // `-t all` completes and parses as a legitimate value.
        let cli = Cli {
            dir: None,
            shard_type: None,
            namespace: None,
            command: Commands::Collections {
                all: false,
                layout: false,
                sort: None,
                limit: -1,
            },
        };
        assert_eq!(
            cli.shard_types().collect::<Vec<_>>(),
            ShardType::ALL.to_vec()
        );
    }

    #[test]
    fn shard_types_yields_just_the_selected_type() {
        let cli = Cli {
            dir: None,
            shard_type: Some(ShardType::State),
            namespace: None,
            command: Commands::Collections {
                all: false,
                layout: false,
                sort: None,
                limit: -1,
            },
        };
        assert_eq!(
            cli.shard_types().collect::<Vec<_>>(),
            vec![ShardType::State]
        );
    }

    #[test]
    fn collections_dash_t_all_iterates_every_pool_without_the_all_flag() {
        // Regression test for the `-t all` vs `--all` inconsistency: before
        // the fix, `shard_type: None` with `all: false` (i.e. `-t all` typed
        // without also passing `--all`) errored with "this command requires
        // a specific shard type" instead of behaving like `--all`.
        let dir = unique_temp_dir();
        DatabaseLayout::open(dir.clone()).unwrap();
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Collections {
                all: false,
                layout: false,
                sort: None,
                limit: -1,
            },
        };

        cmd_collections(&cli, false, false, None, -1).unwrap();

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shards_dash_t_all_iterates_every_pool_without_the_all_flag() {
        let dir = unique_temp_dir();
        DatabaseLayout::open(dir.clone()).unwrap();
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Shards {
                all: false,
                layout: false,
                sort: None,
            },
        };

        cmd_shards(&cli, false, false, None).unwrap();

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn collections_with_a_specific_type_still_targets_only_that_pool() {
        // Unaffected-path guard: a specific `-t` selection (the default,
        // and the common case) must still behave exactly as before --
        // single-pool, no iteration, no `open_layout` overhead.
        let dir = unique_temp_dir();
        DatabaseLayout::open(dir.clone()).unwrap();
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: Some(ShardType::State),
            namespace: None,
            command: Commands::Collections {
                all: false,
                layout: false,
                sort: None,
                limit: -1,
            },
        };

        cmd_collections(&cli, false, false, None, -1).unwrap();

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stats_dash_t_all_iterates_every_pool() {
        let dir = unique_temp_dir();
        DatabaseLayout::open(dir.clone()).unwrap();
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Stats { json: false },
        };
        cmd_stats(&cli, false).unwrap();
        cmd_stats(&cli, true).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn get_dash_t_all_finds_record_across_pools() {
        let dir = unique_temp_dir();
        let layout = DatabaseLayout::open(dir.clone()).unwrap();
        let state_dir = layout.pool_dir_read_only(ShardType::State).unwrap();
        let store = PackfileStorage::open(state_dir).unwrap();
        let col_id = [0x01; 16];
        let node_id = [0x02; 16];
        let data = NodeData::new(Bytes::from_static(b"{\"hello\":\"world\"}\n"));
        store.put(&col_id, &node_id, &data).unwrap();
        store.sync().unwrap();

        let col_hex = hex::encode(col_id);
        let node_hex = hex::encode(node_id);

        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Get {
                collection: None,
                id: node_hex.clone(),
                raw: false,
            },
        };

        cmd_get(&cli, None, &node_hex, false).unwrap();
        cmd_get(&cli, Some(&col_hex), &node_hex, true).unwrap();

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn info_dash_t_all_finds_collection_across_pools() {
        let dir = unique_temp_dir();
        let layout = DatabaseLayout::open(dir.clone()).unwrap();
        let state_dir = layout.pool_dir_read_only(ShardType::State).unwrap();
        let store = PackfileStorage::open(state_dir).unwrap();
        let col_id = [0x01; 16];
        let node_id = [0x02; 16];
        let data = NodeData::new(Bytes::from_static(b"hello"));
        store.put(&col_id, &node_id, &data).unwrap();
        store.sync().unwrap();

        let col_hex = hex::encode(col_id);
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Info {
                collection: col_hex.clone(),
            },
        };

        cmd_info(&cli, &col_hex).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn scan_dash_t_all_finds_collection_across_pools() {
        let dir = unique_temp_dir();
        let layout = DatabaseLayout::open(dir.clone()).unwrap();
        let state_dir = layout.pool_dir_read_only(ShardType::State).unwrap();
        let store = PackfileStorage::open(state_dir).unwrap();
        let col_id = [0x01; 16];
        let node_id = [0x02; 16];
        let data = NodeData::new(Bytes::from_static(b"hello"));
        store.put(&col_id, &node_id, &data).unwrap();
        store.sync().unwrap();

        let col_hex = hex::encode(col_id);
        let cli = Cli {
            dir: Some(dir.clone()),
            shard_type: None,
            namespace: None,
            command: Commands::Scan {
                selector: col_hex.clone(),
                verbose: false,
                limit: -1,
                id: None,
                collection: None,
                raw: false,
                sort: None,
                reverse: false,
            },
        };

        cmd_scan(&cli, &col_hex, false, -1, None, None, false, None, false).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Template whose record identity derives from `sender` rather than the
    /// `event_id`, so two events with distinct `event_ids` can deliberately map to
    /// the same storage node ID — modelling the truncated-hash collision the
    /// import dedup must reject rather than collapse.
    fn sender_identity_template() -> CollectionTemplate {
        CollectionTemplate {
            name: "collisions".into(),
            collection_kind: "federation".into(),
            record_identity: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/sender".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            collection_key: CollectionKeyRule {
                pointer: "/room_id".into(),
                collection_type: MATRIX_ROOM_COLLECTION_TYPE,
                display_id_pointer: "/room_id".into(),
            },
        }
    }

    #[allow(clippy::type_complexity)]
    fn import_fixture(
        name: &str,
    ) -> (
        PackfileStorage,
        PathBuf,
        PathBuf,
        CollectionTemplate,
        [u8; 16],
    ) {
        let dir = unique_temp_dir().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let template = sender_identity_template();
        let collection_id = template_collection_id(&template, "!room");
        let path = dir.join("fixture.json");
        (store, dir, path, template, collection_id)
    }

    fn import_event(event_id_: &str, sender: &str, room: &str) -> OwnedValue {
        owned_value(&format!(
            r#"{{"event_id":"{event_id_}","sender":"{sender}","room_id":"{room}","type":"m.room.message","content":{{}}}}"#
        ))
    }

    #[test]
    fn import_dedups_repeated_event_id_within_one_input() {
        let (store, dir, path, template, collection_id) = import_fixture("import_dedup");
        let mut established = HashSet::new();
        established.insert(collection_id);
        let events = vec![
            import_event("$a", "@alice", "!room"),
            import_event("$a", "@alice", "!room"),
        ];
        import_pdu_events(
            &store,
            &dir,
            &path,
            &events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
        let stats = store.stats();
        assert_eq!(
            stats.put_many_calls, 1,
            "a same-event repeat must collapse into one batched write"
        );
        assert_eq!(
            stats.put_many_records, 1,
            "the repeated event_id must not be written twice"
        );
        let node_id = template_node_id(&template, &events[0]).unwrap().unwrap();
        assert!(
            store.get(&collection_id, &node_id).unwrap().is_some(),
            "the single stored record must be readable under its node ID"
        );
    }

    #[test]
    fn import_rejects_two_event_ids_mapping_to_one_node_id() {
        let (store, dir, path, template, collection_id) = import_fixture("import_input_collision");
        let mut established = HashSet::new();
        established.insert(collection_id);
        let events = vec![
            import_event("$a", "@alice", "!room"),
            import_event("$b", "@alice", "!room"),
        ];
        let error = import_pdu_events(
            &store,
            &dir,
            &path,
            &events,
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .expect_err("distinct event_ids truncating to one node ID must be rejected");
        assert!(
            error.to_string().contains("maps to multiple event IDs"),
            "saw: {error}"
        );
        assert_eq!(
            store.stats().put_many_calls,
            0,
            "no records may be written when the input itself collides"
        );
    }

    #[test]
    fn import_rejects_cross_record_event_id_collision() {
        let (store, dir, path, template, collection_id) =
            import_fixture("import_cross_record_collision");
        let existing = import_event("$c", "@alice", "!room");
        let node_id = template_node_id(&template, &existing).unwrap().unwrap();
        store
            .put(
                &collection_id,
                &node_id,
                &NodeData::new(Bytes::from(existing.encode().into_bytes())),
            )
            .unwrap();

        let incoming = import_event("$a", "@alice", "!room");
        let mut established = HashSet::new();
        established.insert(collection_id);
        let error = import_pdu_events(
            &store,
            &dir,
            &path,
            &[incoming],
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .expect_err("an on-disk record with a differing event_id must refuse to be overwritten");
        assert!(
            error
                .to_string()
                .contains("collides with a different event_id"),
            "saw: {error}"
        );
        assert_eq!(
            store.stats().put_many_calls,
            0,
            "a collision must not trigger a write"
        );
    }

    #[test]
    fn import_keeps_existing_record_when_event_id_matches() {
        let (store, dir, path, template, collection_id) = import_fixture("import_keep_same_event");
        let event = import_event("$a", "@alice", "!room");
        let node_id = template_node_id(&template, &event).unwrap().unwrap();
        store
            .put(
                &collection_id,
                &node_id,
                &NodeData::new(Bytes::from(event.clone().encode().into_bytes())),
            )
            .unwrap();

        let mut established = HashSet::new();
        established.insert(collection_id);
        import_pdu_events(
            &store,
            &dir,
            &path,
            &[event],
            &[],
            None,
            None,
            &template,
            &mut established,
        )
        .unwrap();
        assert_eq!(
            store.stats().put_many_calls,
            0,
            "a matching existing record must be classified present, not rewritten"
        );
    }

    /// Open every pack in a pool and total the records it physically holds.
    fn count_pack_records(pool_dir: &Path) -> u64 {
        let mut total = 0;
        if !pool_dir.exists() {
            return total;
        }
        for (pack_id, _, _) in glob_pack_files(pool_dir).unwrap() {
            let pack = pool_dir.join(format!("pack_{pack_id:016x}.pack"));
            let file = std::fs::File::open(pack).unwrap();
            let mut reader = std::io::BufReader::new(file);
            if mtxdb::packfile::read_header(&mut reader).unwrap().is_none() {
                continue;
            }
            while let Some(record) = mtxdb::packfile::read_record(&mut reader).unwrap() {
                let _ = record;
                total = total.saturating_add(1);
            }
        }
        total
    }

    #[test]
    fn auth_chain_reimport_writes_nothing_new() {
        let root = unique_temp_dir();
        let pool_dir = root.join("pools").join("event-dag");
        std::fs::create_dir_all(&pool_dir).unwrap();
        let store = PackfileStorage::open(pool_dir.clone()).unwrap();
        let input = root.join("export.json");
        std::fs::write(
            &input,
            r#"{
                "pdus": [
                    {"event_id":"$c","room_id":"!room","sender":"@server",
                     "type":"m.room.create","state_key":"","content":{"creator":"@server"},
                     "auth_events":[]}
                ],
                "auth_chain": [
                    {"event_id":"$a1","room_id":"!room","sender":"@server",
                     "type":"m.room.member","state_key":"@server",
                     "content":{"membership":"join"},"auth_events":[]}
                ]
            }"#,
        )
        .unwrap();
        let template = default_matrix_import_template();
        let mut established = HashSet::new();

        cmd_import_file(&store, &pool_dir, &input, None, &template, &mut established).unwrap();
        // Declares the create so batch_has_create resolves for any follow-up.
        cmd_import_file(&store, &pool_dir, &input, None, &template, &mut established).unwrap();

        let auth_dir = pool_dir.parent().unwrap().join("auth-chain");
        assert_eq!(
            count_pack_records(&auth_dir),
            1,
            "the auth-chain pool must hold exactly the one imported auth event, unchanged by the reimport"
        );
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

    /// Build a Synapse `event_json` mirror record: big-endian `i32`
    /// `format_version`, big-endian `u32` `internal_metadata` length, then
    /// the two JSON documents back to back with no delimiter.
    fn encode_event_json_record(
        format_version: i32,
        internal_metadata: &str,
        json: &str,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&format_version.to_be_bytes());
        buf.extend_from_slice(
            &u32::try_from(internal_metadata.len())
                .unwrap()
                .to_be_bytes(),
        );
        buf.extend_from_slice(internal_metadata.as_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf
    }

    #[test]
    fn decode_event_json_record_splits_metadata_from_pdu() {
        // The exact shape reported against a real event_json mirror record:
        // internal_metadata carrying a device_id, followed directly by the
        // PDU JSON with no delimiter between the two documents.
        let metadata = r#"{"device_id":"KIVSBMDOUC"}"#;
        let pdu = r#"{"signatures":{"test":{"ed25519:a_lPym":"sig"}},"unsigned":{"age_ts":4},"room_id":"!DZJveaTFXeqOdyrdeh:test","auth_events":[],"prev_events":[],"content":{"room_version":"11"},"depth":1,"hashes":{"sha256":"3sk9WqGW2B6W7tRNY5xwZ6qpHMiZX/hJWHBYx42uVyY"},"origin_server_ts":4,"sender":"@u1:test","state_key":"","type":"m.room.create"}"#;
        let record = encode_event_json_record(3, metadata, pdu);

        let (format_version, decoded_metadata, decoded_json) =
            decode_event_json_record(&record).expect("recognized event_json record shape");
        assert_eq!(format_version, 3);
        let decoded_metadata = String::from_utf8(decoded_metadata).unwrap();
        let decoded_json = String::from_utf8(decoded_json).unwrap();
        assert!(decoded_metadata.contains("KIVSBMDOUC"));
        assert!(decoded_json.contains("m.room.create"));

        // pretty_print_payload must reach the same decode, not the plain
        // single-JSON path (the whole record is not valid JSON on its own).
        let pretty = pretty_print_payload(&record).expect("event_json record is decodable");
        let pretty = String::from_utf8(pretty).unwrap();
        assert!(pretty.contains("format_version=3"));
        assert!(pretty.contains("KIVSBMDOUC"));
        assert!(pretty.contains("m.room.create"));
    }

    #[test]
    fn decode_event_json_record_rejects_plain_json_and_garbage() {
        // A plain single JSON document is not this record shape — callers
        // must try `pretty_json_stream` first, not route ordinary payloads
        // through this decoder.
        assert!(decode_event_json_record(br#"{"type":"m.room.message"}"#).is_none());
        // Arbitrary short binary data, and a length field pointing past the
        // end of the buffer, must not panic and must not be mistaken for a
        // valid record.
        assert!(decode_event_json_record(b"\x00\x01").is_none());
        assert!(decode_event_json_record(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, b'x']).is_none());
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
    fn matrix_batch_requires_a_valid_create_for_its_room() {
        let direct_create = owned_value(
            r#"{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org"}"#,
        );
        let missing_state_key = owned_value(
            r#"{"type":"m.room.create","event_id":"$not-a-state-event","room_id":"!room:example.org"}"#,
        );
        let message = owned_value(
            r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org"}"#,
        );

        assert!(matrix_batch_has_create(
            &[direct_create, message.clone()],
            "!room:example.org"
        ));
        assert!(!matrix_batch_has_create(
            &[missing_state_key, message],
            "!room:example.org"
        ));
    }

    #[test]
    fn matrix_batch_associates_a_room_id_less_create_through_auth() {
        let create = owned_value(r#"{"type":"m.room.create","state_key":"","event_id":"$create"}"#);
        let modern_auth = owned_value(
            r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org","auth_events":["$create"]}"#,
        );
        let legacy_auth = owned_value(
            r#"{"type":"m.room.message","event_id":"$legacy","room_id":"!room:example.org","auth_events":[["$create",{}]]}"#,
        );

        assert!(matrix_batch_has_create(
            &[create.clone(), modern_auth],
            "!room:example.org"
        ));
        assert!(matrix_batch_has_create(
            &[create, legacy_auth],
            "!room:example.org"
        ));
    }

    #[test]
    fn import_admission_rejects_an_unestablished_room_before_writing() {
        let message = owned_value(
            r#"{"type":"m.room.message","event_id":"$message","room_id":"!room:example.org"}"#,
        );
        let template = default_matrix_import_template();
        let error = resolve_import_collection(&[message], None, None, &template, &HashSet::new())
            .expect_err("a room without a batch or persisted create must be rejected");
        assert!(error
            .to_string()
            .contains("no valid m.room.create event is in this input or already on disk"));
    }

    #[test]
    fn import_admission_accepts_a_batch_that_establishes_its_room() {
        let create = owned_value(
            r#"{"type":"m.room.create","state_key":"","event_id":"$create","room_id":"!room:example.org"}"#,
        );
        let template = default_matrix_import_template();
        let (collection_id, batch_has_create) =
            resolve_import_collection(&[create], None, None, &template, &HashSet::new()).unwrap();
        assert_eq!(
            collection_id,
            template_collection_id(&template, "!room:example.org")
        );
        assert!(batch_has_create);
    }

    #[test]
    fn checked_in_matrix_template_matches_the_executable_importer() {
        let template_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../templates/matrix-event-v1.json");
        let compiled = compile_import_template(Some(&template_path))
            .expect("checked-in Matrix template must be executable");
        assert_eq!(compiled, default_matrix_import_template());
    }

    #[test]
    fn compile_import_template_rejects_an_unsupported_digest_algorithm() {
        let template = br#"{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {
                "identity": {
                    "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                    "internal_key": {"algorithm": "sha256-truncated"}
                }
            },
            "collection": {
                "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
            },
            "establishment": {"required": true}
        }"#;
        let error = compile_reject(template).expect_err(
            "an unsupported digest algorithm must be rejected at compile time, not mid-import",
        );
        assert!(
            format!("{error:#}").contains("unsupported internal-key algorithm"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn compile_import_template_rejects_a_projection_payload_policy() {
        let template = br#"{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {
                "identity": {
                    "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                    "internal_key": {"algorithm": "sha2-256"}
                },
                "payload": {"policy": "projection", "include": ["/type", "/sender"]}
            },
            "collection": {
                "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
            },
            "establishment": {"required": true}
        }"#;
        let error = compile_reject(template).expect_err(
            "a projection payload policy must be rejected instead of degrading to full-source retention",
        );
        assert!(
            format!("{error:#}").contains(r#"payload policy "projection" is not supported"#),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn compile_import_template_rejects_a_non_string_payload_policy() {
        let template = br#"{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {
                "identity": {
                    "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                    "internal_key": {"algorithm": "sha2-256"}
                },
                "payload": {"policy": 42}
            },
            "collection": {
                "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}}
            },
            "establishment": {"required": true}
        }"#;
        let error = compile_reject(template).expect_err(
            "a present non-string payload policy must be rejected instead of silently compiling to source retention",
        );
        assert!(
            format!("{error:#}").contains("payload policy must be a string"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn compile_import_template_rejects_a_distinct_display_label() {
        let template = br#"{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {
                "identity": {
                    "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                    "internal_key": {"algorithm": "sha2-256"}
                },
                "payload": {"policy": "retain-source"}
            },
            "collection": {
                "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}},
                "labels": [{"name": "display_id", "value": "/canonical_alias"}]
            },
            "establishment": {"required": true}
        }"#;
        let error = compile_reject(template).expect_err(
            "a distinct display label must be rejected since the importer cannot persist it",
        );
        assert!(
            format!("{error:#}").contains("display label"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn compile_import_template_rejects_non_array_labels() {
        let template = br#"{
            "format": "mtxdb.collection-template/v1",
            "name": "matrix-event-v1",
            "record": {
                "identity": {
                    "extract": {"kind": "json-pointer-rfc-6901", "path": "/event_id"},
                    "internal_key": {"algorithm": "sha2-256"}
                },
                "payload": {"policy": "retain-source"}
            },
            "collection": {
                "membership": {"extract": {"kind": "json-pointer-rfc-6901", "path": "/room_id"}},
                "labels": {"name": "display_id", "value": "membership-value"}
            },
            "establishment": {"required": true}
        }"#;
        let error = compile_reject(template).expect_err(
            "a present non-array collection.labels must not be silently treated as absent",
        );
        assert!(
            format!("{error:#}").contains("collection.labels must be an array"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn compile_import_template_rejects_a_malformed_display_label() {
        for label_value in [
            r#"{"kind":"json-pointer-rfc-6901"}"#,
            r#"{"path": 42}"#,
            r"42",
            r"null",
            r#"["/canonical_alias"]"#,
            r"true",
        ] {
            let template = format!(
                r#"{{
                "format": "mtxdb.collection-template/v1",
                "name": "matrix-event-v1",
                "record": {{
                    "identity": {{
                        "extract": {{"kind": "json-pointer-rfc-6901", "path": "/event_id"}},
                        "internal_key": {{"algorithm": "sha2-256"}}
                    }},
                    "payload": {{"policy": "retain-source"}}
                }},
                "collection": {{
                    "membership": {{"extract": {{"kind": "json-pointer-rfc-6901", "path": "/room_id"}}}},
                    "labels": [{{"name": "display_id", "value": {label_value}}}]
                }},
                "establishment": {{"required": true}}
            }}"#
            );
            let error = compile_reject(template.as_bytes()).expect_err(
                "a malformed display label value must not silently fall back to the membership pointer",
            );
            assert!(
                format!("{error:#}").contains("display label"),
                "unexpected error for label value {label_value}: {error:#}"
            );
        }
    }

    #[test]
    fn pointer_extraction_supports_array_indices_and_rejects_bad_pointers() {
        let event =
            owned_value(r#"{"auth_events": ["$a", "$b"], "event_id": "$c", "a~1b": "escaped"}"#);
        // Valid array index.
        assert_eq!(extract_pointer_string(&event, "/auth_events/1"), Some("$b"));
        // Valid tilde escaping: ~0 → ~, ~1 → /.
        assert_eq!(extract_pointer_string(&event, "/a~01b"), Some("escaped"));
        // A non-empty pointer must start with `/`; a bare field name is not
        // a valid RFC 6901 pointer and must not silently resolve anything.
        assert_eq!(extract_pointer_string(&event, "event_id"), None);
        // Out-of-range and non-numeric array segments must not panic or
        // fall through to some other value.
        assert_eq!(extract_pointer_string(&event, "/auth_events/9"), None);
        assert_eq!(extract_pointer_string(&event, "/auth_events/x"), None);
        // Leading zeroes in array indices are forbidden by RFC 6901.
        assert_eq!(extract_pointer_string(&event, "/auth_events/01"), None);
        assert_eq!(extract_pointer_string(&event, "/auth_events/00"), None);
        // Signs and non-digit characters are forbidden by RFC 6901.
        assert_eq!(extract_pointer_string(&event, "/auth_events/+1"), None);
        assert_eq!(extract_pointer_string(&event, "/auth_events/-1"), None);
        // Invalid tilde escapes (~2, ~a, trailing ~) are forbidden.
        assert_eq!(extract_pointer_string(&event, "/a~2b"), None);
        assert_eq!(extract_pointer_string(&event, "/a~ab"), None);
        assert_eq!(extract_pointer_string(&event, "/a~"), None);
    }

    // `physical_layout`/`avoidable_spread_bytes` coverage now lives with
    // their implementation in `mtxdb::packfile::layout::tests` —
    // this crate just imports and displays them, nothing left to test here.

    // ---- HAMT decoder tests ----

    /// Encode a length-prefixed UTF-8 string (u32 LE length + bytes).
    fn encode_lps(s: &str) -> Vec<u8> {
        let len = u32::try_from(s.len()).unwrap();
        let cap = usize::try_from(len).unwrap().checked_add(s.len()).unwrap();
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        out
    }

    /// Matches `synapse/rust/src/state_hamt.rs`'s
    /// `serde_json::to_string(&(event_type, state_key))` -- the real K
    /// encoding is a single JSON 2-element array string, not a raw tuple
    /// codec.
    fn hamt_leaf_key_json(event_type: &str, state_key: &str) -> String {
        use simd_json::prelude::Writable;
        simd_json::OwnedValue::Array(Box::new(vec![
            simd_json::OwnedValue::from(event_type),
            simd_json::OwnedValue::from(state_key),
        ]))
        .encode()
    }

    /// Build a rezzy wire-v1 HAMT node from leaf triples and child hashes.
    ///
    /// Each leaf is encoded as two length-prefixed strings matching the real
    /// Synapse format: K = `serde_json::to_string((event_type, state_key))`,
    /// V = `event_id`.
    fn build_hamt_node(leaves: &[(&str, &str, &str)], child_hashes: &[[u8; 32]]) -> Vec<u8> {
        let mut datamap: u32 = 0;
        let mut nodemap: u32 = 0;
        for i in 0..leaves.len() {
            datamap |= 1u32.checked_shl(u32::try_from(i).unwrap()).unwrap_or(0);
        }
        let leaf_count = u32::try_from(leaves.len()).unwrap();
        let child_count = u32::try_from(child_hashes.len()).unwrap();
        for i in 0..child_hashes.len() {
            let slot = leaf_count.checked_add(u32::try_from(i).unwrap()).unwrap();
            nodemap |= 1u32.checked_shl(slot).unwrap_or(0);
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(b"MTHN");
        buf.push(0x01); // codec version
        buf.extend_from_slice(&datamap.to_le_bytes());
        buf.extend_from_slice(&nodemap.to_le_bytes());
        buf.extend_from_slice(&leaf_count.to_le_bytes());
        buf.extend_from_slice(&child_count.to_le_bytes());
        for (et, sk, eid) in leaves {
            let key_json = hamt_leaf_key_json(et, sk);
            buf.extend_from_slice(&encode_lps(&key_json));
            buf.extend_from_slice(&encode_lps(eid));
        }
        for hash in child_hashes {
            buf.extend_from_slice(hash);
        }
        buf
    }

    #[test]
    fn hamt_single_leaf_no_children() {
        let node = build_hamt_node(&[("m.room.name", "", "$abc:server")], &[]);
        let out = decode_hamt_node(&node).expect("should decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("m.room.name"));
        assert!(text.contains("$abc:server"));
        assert!(text.contains("1 leaves"));
        assert!(text.contains("0 children"));
    }

    #[test]
    fn hamt_multiple_leaves_with_children() {
        let h1 = [0x42u8; 32];
        let h2 = [0x99u8; 32];
        let node = build_hamt_node(
            &[
                ("m.room.member", "@a:b", "$ev1:server"),
                ("m.room.create", "", "$ev2:server"),
            ],
            &[h1, h2],
        );
        let out = decode_hamt_node(&node).expect("should decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("@a:b"));
        assert!(text.contains("$ev1:server"));
        assert!(text.contains("$ev2:server"));
        assert!(text.contains(&hex::encode(&h1[..8])));
        assert!(text.contains(&hex::encode(&h2[..8])));
    }

    #[test]
    fn hamt_empty_strings() {
        let node = build_hamt_node(&[("", "", "")], &[]);
        let out = decode_hamt_node(&node).expect("should decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("leaf[0]"));
    }

    #[test]
    fn hamt_empty_node() {
        let node = build_hamt_node(&[], &[]);
        let out = decode_hamt_node(&node).expect("should decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("0 leaves"));
        assert!(text.contains("0 children"));
    }

    #[test]
    fn hamt_state_group_root() {
        let room_prefix = [0xB2, 0xA2, 0xFD, 0xEA, 0xB7, 0x14, 0xDE, 0x6D];
        let room_id = "owusNddwskNpuHuitQ:test";
        let root_hash = [0xAB; 32];
        let lattice = [0xCD; 2048];
        let mut encoded = b"MTHR\x01".to_vec();
        encoded.extend_from_slice(&u16::try_from(room_prefix.len()).unwrap().to_be_bytes());
        encoded.extend_from_slice(&room_prefix);
        encoded.extend_from_slice(&u16::try_from(room_id.len()).unwrap().to_be_bytes());
        encoded.extend_from_slice(room_id.as_bytes());
        encoded.extend_from_slice(&root_hash);
        encoded.extend_from_slice(&lattice);
        assert_eq!(encoded.len(), 2120);

        let out = decode_hamt_root(&encoded).expect("should decode root");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("b2a2fdeab714de6d"));
        assert!(text.contains(room_id));
        assert!(text.contains("abababababababab"));
        assert!(text.contains("2048 bytes"));
        assert!(text.contains("lattice digest (SHA-256): "));
    }

    #[test]
    fn hamt_invalid_utf8_rejected() {
        let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
        // Corrupt the event type string: write an invalid UTF-8 sequence.
        // The length prefix says 10 bytes, but we fill with 0xFF.
        let invalid = b"\x0a\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff";
        // Find where the first leaf's key starts (after the 21-byte header).
        node[21..21 + invalid.len()].copy_from_slice(invalid);
        assert!(decode_hamt_node(&node).is_none());
    }

    #[test]
    fn hamt_wrong_version_rejected() {
        let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
        node[4] = 0x00;
        assert!(decode_hamt_node(&node).is_none());
    }

    #[test]
    fn hamt_random_binary_not_hamt() {
        // Arbitrary payload starting with 0x01 should not decode merely from
        // the version byte; the complete header and body must also validate.
        let data = vec![0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff];
        assert!(decode_hamt_node(&data).is_none());
    }

    #[test]
    fn hamt_truncated_payload_rejected() {
        let full = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
        // Truncate after header -- leaf data missing.
        let trunc: Vec<u8> = full[..20].to_vec();
        assert!(decode_hamt_node(&trunc).is_none());
    }

    #[test]
    fn hamt_overlong_payload_rejected() {
        let mut node = build_hamt_node(&[("m.room.name", "", "$ok")], &[]);
        node.extend_from_slice(&[0u8; 16]);
        assert!(decode_hamt_node(&node).is_none());
    }

    #[test]
    fn hamt_pretty_print_returns_hamt() {
        let node = build_hamt_node(&[("m.room.topic", "room", "$t:server")], &[]);
        let out = pretty_print_payload(&node).expect("should recognize HAMT");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("HAMT CHAMP"));
    }

    #[test]
    fn hamt_datamap_nodemap_overlap_rejected() {
        // Both bitmaps claim slot 0.
        let mut buf = b"MTHN\x01".to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes()); // datamap: slot 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // nodemap: slot 0 (overlap)
        buf.extend_from_slice(&1u32.to_le_bytes()); // leaf_count
        buf.extend_from_slice(&1u32.to_le_bytes()); // child_count
        assert!(decode_hamt_node(&buf).is_none());
    }

    #[test]
    fn hamt_leaf_count_mismatch_rejected() {
        // datamap says 1 leaf, but leaf_count says 0.
        let mut buf = b"MTHN\x01".to_vec();
        buf.extend_from_slice(&1u32.to_le_bytes()); // datamap
        buf.extend_from_slice(&0u32.to_le_bytes()); // nodemap
        buf.extend_from_slice(&0u32.to_le_bytes()); // leaf_count (wrong)
        buf.extend_from_slice(&0u32.to_le_bytes()); // child_count
        assert!(decode_hamt_node(&buf).is_none());
    }

    #[test]
    fn hamt_rezzy_encoded_fixture() {
        use rezzy::hamt::PersistedInternalNode;

        // Build a node using rezzy's own encoder -- this is the ground
        // truth. K = String (JSON key), V = String (event_id), matching
        // the actual `HamtNode<String, String>` instantiation Synapse uses.
        let leaves = vec![
            (
                hamt_leaf_key_json("m.room.member", "@alice:example.org"),
                "$ev1:example.org".to_owned(),
            ),
            (
                hamt_leaf_key_json("m.room.name", ""),
                "$ev2:example.org".to_owned(),
            ),
        ];
        let child_hash = [0xABu8; 32];
        let node = PersistedInternalNode {
            datamap: 0b0011, // slots 0 and 1
            nodemap: 0b0100, // slot 2
            leaves,
            child_hashes: vec![child_hash],
        };
        let encoded = node.encode_v1();

        // Our decoder must accept these exact bytes.
        let out = decode_hamt_node(&encoded).expect("rezzy-encoded node must decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("@alice:example.org"));
        assert!(text.contains("$ev1:example.org"));
        assert!(text.contains("$ev2:example.org"));
        assert!(text.contains("m.room.member"));
        assert!(text.contains("m.room.name"));
        assert!(text.contains(&hex::encode(&child_hash[..8])));
    }

    #[test]
    fn hamt_rezzy_roundtrip_single_leaf() {
        use rezzy::hamt::PersistedInternalNode;

        let node = PersistedInternalNode {
            datamap: 0x01,
            nodemap: 0x00,
            leaves: vec![(
                hamt_leaf_key_json("m.room.create", ""),
                "$create:server".to_owned(),
            )],
            child_hashes: vec![],
        };
        let encoded = node.encode_v1();
        let out = decode_hamt_node(&encoded).expect("must decode");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("m.room.create"));
        assert!(text.contains("$create:server"));
        assert!(text.contains("1 leaves"));
        assert!(text.contains("0 children"));
    }

    // ── Import → event DAG → auth chain → state group integration tests ──

    #[test]
    fn build_event_dag_resolves_prev_and_auth_edges() {
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{"creator":"@a:x"}}"#,
        );
        let member = owned_value(
            r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
        );
        let msg = owned_value(
            r#"{"event_id":"$msg","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$join"],"auth_events":["$join","$create"],"content":{"body":"hi"}}"#,
        );
        let (frontier, _id_map, _rev) = build_event_dag(&[create, member, msg]);
        assert_eq!(frontier.nodes.len(), 3);
        // $msg should have 2 prev edges and 2 auth edges
        let msg_node = &frontier.nodes[2];
        assert_eq!(msg_node.prev.1, 1); // 1 prev_events
        assert_eq!(msg_node.auth.1, 2); // 2 auth_events
    }

    #[test]
    fn compute_state_groups_assigns_groups_per_event() {
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{"creator":"@a:x"}}"#,
        );
        let member = owned_value(
            r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
        );
        let msg = owned_value(
            r#"{"event_id":"$msg","room_id":"!r:x","type":"m.room.message","sender":"@a:x","prev_events":["$join"],"auth_events":["$join"],"content":{"body":"hi"}}"#,
        );
        let groups = compute_state_groups(&[create, member, msg], &[]).unwrap();
        // All three events should have a state group.
        assert!(groups.contains_key("$create"));
        assert!(groups.contains_key("$join"));
        assert!(groups.contains_key("$msg"));
        // State group IDs are valid unpadded base64url.
        for sg in groups.values() {
            assert_ne!(sg, "");
            assert!(sg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        }
    }

    #[test]
    fn compute_state_groups_empty_input() {
        let groups = compute_state_groups(&[], &[]).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn compute_state_groups_deterministic() {
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","state_key":"","sender":"@a:x","content":{}}"#,
        );
        let member = owned_value(
            r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","state_key":"@a:x","sender":"@a:x","prev_events":["$create"],"auth_events":["$create"],"content":{}}"#,
        );
        let g1 = compute_state_groups(&[create.clone(), member.clone()], &[]).unwrap();
        let g2 = compute_state_groups(&[create, member], &[]).unwrap();
        assert_eq!(g1, g2);
    }

    #[test]
    fn verify_auth_chain_edges_detects_dangling() {
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","sender":"@a:x","content":{}}"#,
        );
        let member = owned_value(
            r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","sender":"@a:x","auth_events":["$create"],"content":{}}"#,
        );
        // All edges resolve.
        let dangling = verify_auth_chain_edges(&[create.clone(), member.clone()], &[]);
        assert_eq!(dangling, Vec::<(String, String)>::new());

        // $missing is not in either set.
        let bad = owned_value(
            r#"{"event_id":"$bad","room_id":"!r:x","type":"m.room.message","sender":"@a:x","auth_events":["$missing"],"content":{}}"#,
        );
        let dangling = verify_auth_chain_edges(&[create, member, bad], &[]);
        assert_eq!(dangling.len(), 1);
        assert_eq!(dangling[0].0, "$bad");
        assert_eq!(dangling[0].1, "$missing");
    }

    #[test]
    fn verify_auth_chain_edges_auth_chain_as_known_set() {
        // An auth_events reference to an event only in the auth_chain
        // (not in pdus) should still resolve.
        let create = owned_value(
            r#"{"event_id":"$create","room_id":"!r:x","type":"m.room.create","sender":"@a:x","content":{}}"#,
        );
        let member = owned_value(
            r#"{"event_id":"$join","room_id":"!r:x","type":"m.room.member","sender":"@a:x","auth_events":["$create"],"content":{}}"#,
        );
        let dangling = verify_auth_chain_edges(&[member], &[create]);
        assert_eq!(dangling, Vec::<(String, String)>::new());
    }

    #[test]
    fn parse_federation_input_separates_pdus_and_auth_chain() {
        let json = r#"{
            "pdus": [{"event_id": "$p1", "room_id": "!r:x", "type": "m.room.message"}],
            "auth_chain": [{"event_id": "$a1", "room_id": "!r:x", "type": "m.room.create"}]
        }"#;
        let input = parse_federation_input(json.as_bytes()).unwrap();
        assert_eq!(input.pdus.len(), 1);
        assert_eq!(input.auth_chain.len(), 1);
        assert_eq!(event_id(&input.pdus[0]), Some("$p1"));
        assert_eq!(event_id(&input.auth_chain[0]), Some("$a1"));
    }

    #[test]
    fn parse_federation_input_empty_rejected() {
        let json = r#"{"unrelated": true}"#;
        assert!(parse_federation_input(json.as_bytes()).is_err());
    }

    #[test]
    fn parse_federation_input_requires_an_array_and_rejects_wrong_types() {
        for json in [
            r#"{"unrelated": true}"#,
            r#"{"pdus": null, "auth_chain": []}"#,
            r#"{"pdus": [], "auth_chain": null}"#,
            r#"{"pdus": 3, "auth_chain": []}"#,
            r#"{"pdus": {}, "auth_chain": []}"#,
        ] {
            assert!(
                parse_federation_input(json.as_bytes()).is_err(),
                "malformed federation document was accepted: {json}"
            );
        }
    }

    #[test]
    fn parse_federation_input_accepts_either_array_or_both() {
        for json in [
            r#"{"pdus": []}"#,
            r#"{"auth_chain": []}"#,
            r#"{"pdus": [], "auth_chain": []}"#,
        ] {
            let input = parse_federation_input(json.as_bytes()).unwrap();
            assert_eq!(input.pdus, Vec::<OwnedValue>::new());
            assert_eq!(input.auth_chain, Vec::<OwnedValue>::new());
        }
    }

    #[test]
    fn state_set_digest_is_deterministic() {
        let mut s1 = StateSet::new();
        s1.set("m.room.create", "", "$create".into());
        s1.set("m.room.member", "@a:x", "$join".into());
        let mut s2 = StateSet::new();
        s2.set("m.room.create", "", "$create".into());
        s2.set("m.room.member", "@a:x", "$join".into());
        assert_eq!(s1.digest_base64url(), s2.digest_base64url());
    }

    #[test]
    fn state_set_merge_takes_first_wins() {
        let mut base = StateSet::new();
        base.set("m.room.name", "", "$old".into());
        let mut override_ = StateSet::new();
        override_.set("m.room.name", "", "$new".into());
        override_.set("m.room.topic", "", "$topic".into());
        base.merge(&override_);
        // First-wins: $old should survive because base already had the key.
        assert_eq!(
            &base.entries[&("m.room.name".into(), String::new())],
            "$old"
        );
        // New key should be added.
        assert_eq!(
            &base.entries[&("m.room.topic".into(), String::new())],
            "$topic"
        );
    }

    #[test]
    fn state_set_empty_digest_is_empty_base64url() {
        let s = StateSet::new();
        let digest = s.digest_base64url();
        let expected = Sha256::digest([]);
        assert_eq!(
            digest,
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&expected[..])
        );
    }

    /// Pins the SHA-256 state-set digest so switching algorithm (or changing
    /// the `(type, state_key, event_id)` framing) is a visible, deliberate
    /// format break rather than a silent re-identification of every state
    /// group.
    #[test]
    fn state_set_digest_golden_vector() {
        let mut s = StateSet::new();
        s.set("m.room.name", "", "$old".into());
        assert_eq!(
            s.digest_base64url(),
            "S8Afb--pNuREj_ps-f7RgFQbx0EWjWQQrS2pWiPKk28"
        );
    }

    /// Pins `derive_template_key("sha2-256", ..)`: the first 16 bytes of
    /// SHA-256 of the extracted value, with no domain separation. This is the
    /// identity rule behind `matrix_event_node_id`, so the expected value is
    /// hard-coded rather than recomputed with the same `sha2` calls.
    #[test]
    fn derive_template_key_golden_vector() {
        let id = derive_template_key("sha2-256", "$abc123:example.org").unwrap();
        assert_eq!(hex::encode(id), "8abcf87e5c91a02a1875014f69bb0747");
        // Only the truncation width is under test here; a different input must
        // produce a different id.
        assert_ne!(
            id,
            derive_template_key("sha2-256", "$other:example.org").unwrap()
        );
    }

    /// Pins the in-memory DAG `short_id` derivation: the first 8 bytes of
    /// SHA-256(`event_id`) as a little-endian `u64`.
    #[test]
    fn event_short_id_golden_vector() {
        assert_eq!(
            event_short_id("$abc123:example.org"),
            3_071_614_772_319_927_434
        );
    }

    /// Reimplements `event_node_id`/`event_dag_room_id` from Synapse's
    /// `rust/src/database/mtxdb.rs` independently of `synapse_event_node_id`/
    /// `matrix_room_collection_id`, so a drift between the two derivations
    /// (e.g. someone editing the domain-separation prefix in only one place)
    /// fails this test instead of silently breaking `$event_id`/`!room_id`
    /// lookups against a live Synapse-written database.
    fn reference_node_id(prefix: &[u8], namespace: &str, value: &str) -> [u8; 16] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(prefix);
        hasher.update(namespace.as_bytes());
        hasher.update(b"\0");
        hasher.update(value.as_bytes());
        let hash = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash[..16]);
        id
    }

    #[test]
    fn synapse_event_node_id_matches_reference_derivation() {
        let namespace = "example.org";
        let event_id = "$abc123:example.org";
        assert_eq!(
            synapse_event_node_id(namespace, event_id),
            reference_node_id(b"event_json:event:", namespace, event_id)
        );
        // Different namespaces must not collide.
        assert_ne!(
            synapse_event_node_id(namespace, event_id),
            synapse_event_node_id("other.org", event_id)
        );
    }

    #[test]
    fn matrix_room_collection_id_matches_reference_derivation() {
        let namespace = "example.org";
        let room_id = "!roomid:example.org";
        assert_eq!(
            matrix_room_collection_id(namespace, room_id),
            reference_node_id(b"event_json:dag:", namespace, room_id)
        );
        assert_ne!(
            matrix_room_collection_id(namespace, room_id),
            matrix_room_collection_id("other.org", room_id)
        );
    }
}
