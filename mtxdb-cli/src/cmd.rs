use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context};

use mtxdb::auxiliary::AuxiliaryIndex;
use mtxdb::csr_adjacency::{CsrAdjacency, EventRecord, AUTH, PREV};
use mtxdb::export_schema::{RejectionCode, SoftFailCode};
use mtxdb::packfile::layout::{avoidable_spread_bytes, physical_layout, CollectionPhysicalLayout};
use mtxdb::packfile::storage::{CollectionSummary, OpenPath, RuntimeStats};
use mtxdb::packfile::PackId;
use mtxdb::room_auth::{RoomAuth, MAX_BATCH_EVENTS};
use mtxdb::room_flag::{Flag, FlagChange, FlagQuery, RoomFlagIndex};
use mtxdb::server_keys::KeyTier;
use mtxdb::shard::ShardPool;
use mtxdb::state_group::{StateGroupDerivation, StateGroupTrust};
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::{
    decode_verdict, derive_collection_id, encode_verdict, event_verdict_collection_id,
    frame_digest, record_logical_id, state_group_instance_id, CollectionKeyRule,
    CollectionMetadata, CollectionTemplate, Database, DatabaseLayout, DigestAlgorithm,
    EstablishmentRule, FrameIdInput, FrameIdPolicy, MatrixRoomVersion, PackfileStorage,
    PayloadPolicy, RecordIdentityRule, ShardType, Verdict,
};
use rezzy::auth::auth_types_for_event_ref;
use rezzy::json::{self, Object, Value as OwnedValue};
use rezzy::{redaction_preserved_keys, LeanEvent, RedactionRule, StateResVersion};

use crate::export_import;
use crate::interchange::{self, KeyRecord, RoomEvent, RoomExport};
use crate::state_policy::{resolution_policy, ResolutionPolicy, StateView};
use crate::state_resolve::{resolve_parents, LeanEventContext, StateEntries};
use crate::{Cli, Commands, PacksAction};

const MAX_DEBUG_UNRESOLVED: usize = 20;
/// Byte width of an STGP state-group instance id (128-bit).
const STATE_GROUP_ID_LENGTH: usize = 16;

trait JsonValueExt {
    fn encode(&self) -> String;
    fn encode_pp(&self) -> String;
}

impl JsonValueExt for OwnedValue {
    fn encode(&self) -> String {
        json::write_string_value(self).expect("JSON serialization cannot fail")
    }

    fn encode_pp(&self) -> String {
        json::write_string_pretty(self).expect("JSON serialization cannot fail")
    }
}

/// Auxiliary namespace for the derived `event_id -> state-group instance` cache.
///
/// The namespace is versioned. These values are derived data, so any change to
/// the derivation (hashing, resolution rules, ordering) must bump the suffix.
/// A new namespace starts empty and is repopulated by recomputation on the next
/// import; this is the only safe invalidation, since a well-formed cached value
/// that is semantically stale cannot be detected without recomputing it. For
/// the same reason a corrupted-but-well-formed entry is not self-correcting;
/// recovering from that requires purging the namespace.
///
/// The value is a raw 128-bit `STGP` state-group *instance* id, derived from
/// the sorted parent instance ids and the state `LtHash` (the unkeyed digest
/// used by Synapse/sithnapse and materialized into each `MTHR` root). The
/// `STGP` record holds the parent links and the shared `STAT` root id, so equal
/// state content reached through different parents stays distinct.
const STATE_GROUP_NAMESPACE: &str = "sys:matrix-state-groups:v1";

/// Event -> best-effort state-group instance id for events whose base is missing
/// history (an ancestor body absent from the source). The state is derived
/// from the events that are present, so it can lack changes made by the absent
/// ancestors. Nothing seeds a walk from this namespace and replay does not
/// treat it as a cached complete result; it exists so readers get a best-effort
/// state instead of none. Keep the persisted namespace stable for existing
/// databases; its `partial` label predates this clearer code-level name.
const STATE_GROUP_BEST_EFFORT_NAMESPACE: &str = "sys:matrix-state-groups-best-effort:v1";

/// Whether the importer should print per-phase timings. Off by default so a
/// normal import log stays stable for the tests and for readers who only want
/// counts; set `MTXDB_IMPORT_TIMINGS=1` to see where the seconds go.
pub(crate) fn import_timings_enabled() -> bool {
    std::env::var_os("MTXDB_IMPORT_TIMINGS").is_some()
}

/// Print one completed import phase when timings are enabled.
pub(crate) fn report_import_phase(phase: &str, elapsed: std::time::Duration) {
    if import_timings_enabled() {
        eprintln!("  timing: {phase} {elapsed:?}");
    }
}

/// Whether `value` is a well-formed 128-bit STGP state-group instance id.
fn valid_state_group_id(value: &[u8]) -> bool {
    value.len() == STATE_GROUP_ID_LENGTH
}

enum StateGroupLoad {
    Complete(HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>),
    Missing,
    Invalid {
        event_id: String,
        reason: &'static str,
    },
}

/// Map a batch of cache values (in `event_ids` order) to a [`StateGroupLoad`].
///
/// Split out from [`load_state_groups`] so the cardinality guard is testable
/// without a storage backend that deliberately violates `StorageEngine::get_many`.
fn state_groups_from_values(event_ids: &[String], values: Vec<Option<Vec<u8>>>) -> StateGroupLoad {
    if values.len() != event_ids.len() {
        // StorageEngine::get_many promises request cardinality. A mismatch is
        // a broken-backend condition, not an ordinary cache miss; recompute
        // safely instead.
        return StateGroupLoad::Invalid {
            event_id: "<batch>".to_owned(),
            reason: "cache returned the wrong number of values",
        };
    }
    let mut groups = HashMap::with_capacity(event_ids.len());
    for (event_id, value) in event_ids.iter().zip(values) {
        let Some(value) = value else {
            return StateGroupLoad::Missing;
        };
        if !valid_state_group_id(&value) {
            return StateGroupLoad::Invalid {
                event_id: event_id.clone(),
                reason: "invalid state-group instance ID width",
            };
        }
        let mut id = [0u8; STATE_GROUP_ID_LENGTH];
        id.copy_from_slice(&value);
        groups.insert(event_id.clone(), id);
    }
    StateGroupLoad::Complete(groups)
}

fn load_state_groups(
    aux: &AuxiliaryIndex<'_, PackfileStorage>,
    event_ids: &[String],
) -> Result<StateGroupLoad, mtxdb::storage::StorageError> {
    if event_ids.is_empty() {
        return Ok(StateGroupLoad::Complete(HashMap::new()));
    }
    let keys = event_ids.iter().map(String::as_bytes).collect::<Vec<_>>();
    let values = aux.get_many(&keys)?;
    Ok(state_groups_from_values(event_ids, values))
}

/// Load whichever mappings are already cached, without requiring every id to
/// be present.
///
/// Repair recomputes the whole batch when any mapping is missing, but the
/// mappings it already has are immutable and need not be written again. This
/// returns the present subset so publication can touch only new or changed
/// entries instead of rewriting one record per stored event on every file.
fn load_cached_state_group_subset(
    aux: &AuxiliaryIndex<'_, PackfileStorage>,
    event_ids: &[String],
) -> Result<HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>, mtxdb::storage::StorageError> {
    if event_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let keys = event_ids.iter().map(String::as_bytes).collect::<Vec<_>>();
    let values = aux.get_many(&keys)?;
    if values.len() != event_ids.len() {
        return Ok(HashMap::new());
    }
    let mut groups = HashMap::with_capacity(event_ids.len());
    for (event_id, value) in event_ids.iter().zip(values) {
        let Some(value) = value else {
            continue;
        };
        if !valid_state_group_id(&value) {
            continue;
        }
        let mut id = [0u8; STATE_GROUP_ID_LENGTH];
        id.copy_from_slice(&value);
        groups.insert(event_id.clone(), id);
    }
    Ok(groups)
}

/// Human-readable byte count (`512 B`, `4.3 KB`, `1.2 MB`, `2.1 GB`) —
/// raw byte counts in a shard listing are unreadable past a few digits.
fn fmt_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let max_unit = UNITS.len().saturating_sub(1);
    let mut unit = 0;
    let mut divisor = 1_u64;
    while n.checked_div(divisor).expect("byte divisor is nonzero") >= 1024 && unit < max_unit {
        divisor = divisor.saturating_mul(1024);
        unit = unit.saturating_add(1);
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        let formatted = format_scaled(n, divisor, 1, UNITS[unit]);
        if formatted.starts_with("1024.0 ") && unit < max_unit {
            format_scaled(
                n,
                divisor.saturating_mul(1024),
                1,
                UNITS[unit.saturating_add(1)],
            )
        } else {
            formatted
        }
    }
}

/// Integer with thousands separators (`77468` -> `77,468`).
fn fmt_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len().saturating_add(digits.len() / 3));
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && digits.len().saturating_sub(i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Decimal size scaled to the largest fitting unit with three significant
/// digits (`9026000` -> `9.03 MB`), so index memory and disk read alike.
fn fmt_size_decimal(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut unit = 0;
    let max_unit = UNITS.len().saturating_sub(1);
    let mut divisor = 1_u64;
    while bytes.checked_div(divisor).expect("byte divisor is nonzero") >= 1000 && unit < max_unit {
        divisor = divisor.saturating_mul(1000);
        unit = unit.saturating_add(1);
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        let whole = bytes.checked_div(divisor).expect("byte divisor is nonzero");
        let decimals = if whole >= 100 {
            0
        } else if whole >= 10 {
            1
        } else {
            2
        };
        let formatted = format_scaled(bytes, divisor, decimals, UNITS[unit]);
        if formatted.starts_with("1000 ") && unit < max_unit {
            format_scaled(
                bytes,
                divisor.saturating_mul(1000),
                2,
                UNITS[unit.saturating_add(1)],
            )
        } else {
            formatted
        }
    }
}

fn format_scaled(bytes: u64, divisor: u64, decimals: u32, unit: &str) -> String {
    let whole = bytes.checked_div(divisor).expect("byte divisor is nonzero");
    let scale = 10_u128.pow(decimals);
    let remainder = bytes.checked_rem(divisor).expect("byte divisor is nonzero");
    let fractional = u128::from(remainder)
        .checked_mul(scale)
        .expect("scaled byte remainder fits u128")
        .checked_add(u128::from(divisor / 2))
        .expect("rounded byte remainder fits u128")
        .checked_div(u128::from(divisor))
        .expect("byte divisor is nonzero");
    let rounded = if fractional == scale {
        (u128::from(whole.saturating_add(1)), 0_u128)
    } else {
        (u128::from(whole), fractional)
    };
    if decimals == 0 {
        format!("{} {unit}", rounded.0)
    } else {
        format!(
            "{}.{:0width$} {unit}",
            rounded.0,
            rounded.1,
            width = decimals as usize
        )
    }
}

/// One totals row per database, then the sum, for `collections --per-db`.
fn print_per_db_totals(rows: &[(PathBuf, usize, usize, usize, u64, usize)]) {
    let names: Vec<String> = rows.iter().map(|r| r.0.display().to_string()).collect();
    let width = names.iter().map(String::len).max().unwrap_or(8).max(8);
    println!(
        "per database\n  {:<width$}  {:>11}  {:>10}  {:>12}  {:>10}",
        "database", "collections", "nodes", "index memory", "disk"
    );
    for (name, row) in names.iter().zip(rows) {
        println!(
            "  {name:<width$}  {:>11}  {:>10}  {:>12}  {:>10}",
            fmt_thousands(row.1 as u64),
            fmt_thousands(row.2 as u64),
            fmt_size_decimal(row.3 as u64),
            fmt_size_decimal(row.4)
        );
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

/// Render a logical id (collection, node, or record) in the CLI's canonical
/// form: `0x` followed by uppercase hex. Every id the CLI prints uses this form,
/// and every id selector it accepts must be written in this form — there is no
/// bare-hex shorthand.
fn format_id(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2usize.saturating_add(bytes.len().saturating_mul(2)));
    s.push_str("0x");
    for b in bytes {
        let _ = write!(s, "{b:02X}");
    }
    s
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

fn command_name(cmd: &Commands) -> &'static str {
    match cmd {
        Commands::Init => "init",
        Commands::Put { .. } => "put",
        Commands::Get { .. } => "get",
        Commands::Scan { .. } => "scan",
        Commands::Search { .. } => "search",
        Commands::Delete { .. } => "delete",
        Commands::Tombstones { .. } => "tombstones",
        Commands::Remove { .. } => "remove",
        Commands::Shards { .. } => "shards",
        Commands::Collections { .. } => "collections",
        Commands::Info { .. } => "info",
        Commands::Sync { .. } => "sync",
        Commands::ReclaimWal => "reclaim-wal",
        Commands::Stats { .. } => "stats",
        Commands::Memory { .. } => "memory",
        Commands::Timeline { .. } => "timeline",
        Commands::Flags { .. } => "flags",
        Commands::Meta { .. } => "meta",
        Commands::Import { .. } => "import",
        Commands::Export { .. } => "export",
        Commands::Packs { .. } => "packs",
        Commands::Repack { .. } => "repack",
        Commands::Completions { .. } => "completions",
        _ => "internal",
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run(cli: &Cli) -> anyhow::Result<()> {
    if cli.coalesce {
        match &cli.command {
            Commands::Shards { .. }
            | Commands::Collections { .. }
            | Commands::Stats { .. }
            | Commands::Get { .. }
            | Commands::Sync { .. }
            | Commands::Scan { .. }
            | Commands::Info { .. }
            | Commands::Packs { .. }
            | Commands::Repack { out: Some(_), .. } => {}
            Commands::Repack { out: None, .. } => {
                bail!("--coalesce is not supported for in-place repack; coalescing repack requires --out <DIR>");
            }
            _ => {
                let name = command_name(&cli.command);
                bail!("--coalesce is not supported by `mtxdb {name}`");
            }
        }
        if cli.dirs_or_default().len() == 1
            && !matches!(&cli.command, Commands::Repack { out: Some(_), .. })
        {
            eprintln!("warning: --coalesce has no effect with only one database directory");
        }
    }

    match &cli.command {
        Commands::Put {
            collection,
            id,
            data,
        } => {
            cli.require_single_dir("put")?;
            cmd_put(cli, collection, id, data)
        }
        Commands::Get {
            collection,
            id,
            raw,
            verbose,
            header,
            decode,
        } => cmd_get(
            cli,
            collection.as_deref(),
            id,
            *raw,
            *verbose,
            *header,
            decode.as_deref(),
        ),
        Commands::Collections {
            all,
            layout,
            canonical,
            per_db,
            sort,
            limit,
        } => cmd_collections(
            cli,
            &CollectionOptions {
                flags: CollectionFlags::from_bools([*all, *layout, *canonical, *per_db]),
                sort: sort.as_deref(),
                limit: *limit,
            },
        ),
        Commands::Shards { all, layout, sort } => cmd_shards(cli, *all, *layout, sort.as_deref()),
        Commands::Stats { json } => cmd_stats(cli, *json),
        Commands::Memory { evict, evict_pages } => cmd_memory(cli, *evict, *evict_pages),
        Commands::Timeline {
            room,
            limit,
            reverse,
        } => cmd_timeline(cli, room, *limit, *reverse),
        Commands::Flags {
            room,
            flag,
            not,
            count,
            limit,
            rebuild,
        } => cmd_flags(cli, room, flag.as_deref(), *not, *count, *limit, *rebuild),
        Commands::Meta {
            target,
            json,
            limit,
            offset,
            decode,
        } => cmd_meta(cli, target, *json, *limit, *offset, decode.as_deref()),
        Commands::Info {
            selector,
            pack,
            collection,
            stats,
        } => {
            if let Some(pack) = pack {
                cmd_info(cli, pack, Some(InfoTarget::Pack), *stats)
            } else if let Some(collection) = collection {
                cmd_info(cli, collection, Some(InfoTarget::Collection), *stats)
            } else if let Some(selector) = selector {
                cmd_info(cli, selector, None, *stats)
            } else {
                cmd_info_default(cli, *stats)
            }
        }
        Commands::Scan {
            selector,
            verbose,
            header,
            decode,
            limit,
            id,
            collection,
            raw,
            sort,
            reverse,
            live,
        } => cmd_scan(
            cli,
            selector,
            *verbose,
            *header,
            decode.as_deref(),
            *limit,
            id.as_deref(),
            collection.as_deref(),
            *raw,
            sort.as_deref(),
            *reverse,
            *live,
        ),
        Commands::Search {
            collection,
            template,
            term,
            format,
            limit,
        } => cmd_search(cli, collection, template.as_deref(), term, format, *limit),
        Commands::Import {
            paths,
            collection,
            template,
            no_state_hamts,
        } => {
            cli.require_single_dir("import")?;
            cmd_import(
                cli,
                paths,
                collection.as_deref(),
                template.as_deref(),
                *no_state_hamts,
            )
        }
        Commands::Export {
            collection,
            format,
            metadata,
        } => {
            cli.require_single_dir("export")?;
            cmd_export(cli, collection, format, *metadata)
        }
        Commands::Packs { action } => cmd_packs(cli, action),
        Commands::Repack {
            collection,
            packs,
            all,
            out,
            yes,
        } => {
            let topo = repack_uses_dependency_layout(cli)?;
            if let Some(out_dir) = out {
                cmd_repack_coalesced(cli, out_dir, collection.as_deref(), packs, *all, topo, *yes)
            } else {
                cli.require_single_dir("repack")?;
                cmd_repack(cli, collection.as_deref(), packs, *all, topo, *yes)
            }
        }
        Commands::Delete { collections, yes } => {
            cli.require_single_dir("delete")?;
            cmd_delete(cli, collections, *yes)
        }
        Commands::Tombstones {
            collection,
            keys,
            detail,
        } => {
            cli.require_single_dir("tombstones")?;
            cmd_tombstones(cli, collection, *keys, *detail)
        }
        Commands::Remove {
            collection,
            ids,
            yes,
        } => {
            cli.require_single_dir("remove")?;
            cmd_remove(cli, collection, ids, *yes)
        }
        Commands::Completions { .. } => unreachable!("main emits completion scripts directly"),
        // `-t all` means every pool, same as `sync --all`.
        Commands::Sync { all } => cmd_sync(cli, *all || cli.shard_type.is_none()),
        Commands::ReclaimWal => {
            cli.require_single_dir("reclaim-wal")?;
            cmd_reclaim_wal(cli)
        }
        Commands::Init => {
            cli.require_single_dir("init")?;
            cmd_init(cli)
        }
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
    let store = open_pool(PathBuf::from(path)).context("failed to open test store")?;
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
    let store =
        open_pool_read_only(PathBuf::from(path)).context("failed to open test store read-only")?;
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
    let root = cli.single_dir();
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
        // Only the root descriptor exists yet: each pool is created by its first
        // write, so list the pools it will hold rather than directories.
        let pools: Vec<&str> = ShardType::ALL.iter().map(|shard| shard.as_str()).collect();
        println!("  pools (created on first write): {}", pools.join(", "));
    }
    Ok(())
}

/// Parse a collection's 128-bit logical ID from 32 hex digits prefixed by
/// the canonical lowercase `0x`. Reject other prefixes, wrong lengths, and
/// invalid hex.
fn parse_collection_id(value: &str) -> anyhow::Result<[u8; 16]> {
    let hex = value.strip_prefix("0x").with_context(|| {
        format!("collection ID `{value}` must be 0x-prefixed (32 hex digits, e.g. `0x0123…`)")
    })?;
    if hex.len() != 32 {
        bail!(
            "collection ID must be 32 hex characters after `0x`, got {}",
            hex.len()
        );
    }
    let bytes = hex::decode(hex).context("invalid hex in collection ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

/// Parse a node's 128-bit logical ID from 32 hex digits prefixed by the
/// canonical lowercase `0x`. Reject other prefixes, wrong lengths, and
/// invalid hex.
fn parse_node_id(value: &str) -> anyhow::Result<[u8; 16]> {
    let hex = value.strip_prefix("0x").with_context(|| {
        format!("node ID `{value}` must be 0x-prefixed (32 hex digits, e.g. `0x0123…`)")
    })?;
    if hex.len() != 32 {
        bail!(
            "node ID must be 32 hex characters after `0x`, got {}",
            hex.len()
        );
    }
    let bytes = hex::decode(hex).context("invalid hex in node ID")?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(id)
}

/// Resolve a `0x`-prefixed logical ID, or derive the node ID import stores a
/// `$`-prefixed Matrix event under. The `$` is part of the hashed event ID, so
/// the selector is passed to [`matrix_event_node_id`] unchanged. A malformed
/// logical ID returns the error from [`parse_node_id`].
fn parse_get_id(id: &str) -> anyhow::Result<[u8; 16]> {
    if id.starts_with('$') {
        matrix_event_node_id(id)
    } else {
        parse_node_id(id)
    }
}

/// Member namespace discriminator for Matrix room collections, mixed into
/// [`derive_collection_id`]. Matrix room events use the `EVNT` namespace.
const MATRIX_ROOM_MEMBER_NAMESPACE: Option<[u8; 4]> = Some(mtxdb::MEMBER_NAMESPACE_EVNT);
#[deprecated(note = "use MATRIX_ROOM_MEMBER_NAMESPACE")]
#[allow(dead_code)]
const MATRIX_ROOM_POOL_DST: Option<[u8; 4]> = MATRIX_ROOM_MEMBER_NAMESPACE;

fn blake3_digest(data: &[u8]) -> [u8; 32] {
    mtxdb::content_digest(DigestAlgorithm::Blake3, data)
}

/// Derive a 128-bit lookup ID from the BLAKE3 digest of the complete event ID
/// supplied by the caller, including any `$` sigil. Repack edge extraction
/// and Matrix records imported with the bundled template use this same
/// derivation.
fn matrix_event_node_id(event_id: &str) -> anyhow::Result<[u8; 16]> {
    derive_template_key("blake3-128", event_id)
}

/// Derive the BLAKE3 collection ID for a canonical Matrix room ID.
fn matrix_room_collection_id(room_id: &str) -> [u8; 16] {
    derive_collection_id(MATRIX_ROOM_MEMBER_NAMESPACE, room_id.as_bytes())
}

/// Resolve a collection selector: a `0x`-prefixed logical ID (the engine's
/// template-independent form), or a canonical ID carrying the Matrix profile's
/// room sigil `!` — a *template* property, not an engine constant — which is
/// routed with the BLAKE3 pool-scoped collection identity rule. Bare hex is
/// rejected.
fn parse_collection_selector(selector: &str) -> anyhow::Result<[u8; 16]> {
    if selector.starts_with('!') {
        Ok(matrix_room_collection_id(selector))
    } else {
        parse_collection_id(selector)
    }
}

/// Parse the `MTXDB_COMPRESS` override for writable opens.
///
/// `Ok(None)` means the variable is unset and the pool policies decide.
fn compression_override() -> anyhow::Result<Option<bool>> {
    let Ok(value) = std::env::var("MTXDB_COMPRESS") else {
        return Ok(None);
    };
    let compress = match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => anyhow::bail!(
            "MTXDB_COMPRESS must be a boolean (0/1, false/true, no/yes, off/on), got {value:?}"
        ),
    };
    Ok(Some(compress))
}

/// Open a writable pool with the CLI's resolved runtime config.
fn open_pool(dir: PathBuf) -> io::Result<PackfileStorage> {
    if let Some(compress) = compression_override()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?
    {
        PackfileStorage::open_with_cache_and_policies(
            dir,
            crate::runtime_config().cache_capacity,
            compress,
            mtxdb::packfile::ChecksumPolicy::Full,
        )
    } else {
        PackfileStorage::open_with_config(dir, crate::runtime_config())
    }
}

/// Open a read-only pool with the CLI's resolved runtime config.
fn open_pool_read_only(dir: PathBuf) -> io::Result<PackfileStorage> {
    PackfileStorage::open_read_only_with_config(dir, crate::runtime_config())
}

/// Open a shared-WAL database with the Matrix pool policies and CLI runtime
/// config. `MTXDB_COMPRESS` overrides every pool's compression choice for
/// controlled format/performance comparisons; unset keeps the Matrix defaults.
fn open_database(dir: PathBuf) -> anyhow::Result<Database> {
    let mut policies = mtxdb::matrix_pool_policies();
    if let Some(compress) = compression_override()? {
        for shard in ShardType::ALL {
            policies.for_shard_mut(shard).compress = compress;
        }
    }
    Database::open_with_policies_and_config(dir, policies, *crate::runtime_config())
        .map_err(Into::into)
}

/// Open the store as its exclusive writer. Fails fast if another process
/// (e.g. a live embedder) already holds the writer lock — required for
/// any command that mutates data.
fn open_store(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let pool = selected_pool_dir(cli)?;
    let store = open_pool(pool.clone()).with_context(|| {
        let lock_path = pool.join(".mtxdb.lock");
        if lock_path.exists() {
            format!(
                "failed to open store for writing at `{}`: a lock file exists and could not be replaced (check directory and lock file permissions, e.g. `sudo chown -R $USER <DIR>`)",
                pool.display()
            )
        } else {
            format!("failed to open store for writing at `{}`", pool.display())
        }
    })?;
    store.set_read_plan_policy(cli.read_plan);
    Ok(store)
}

/// Open the store read-only — coexists with a live writer process rather
/// than contending with it. For commands that only ever read collection data.
fn open_store_read_only(cli: &Cli) -> anyhow::Result<PackfileStorage> {
    let store = open_pool_read_only(selected_pool_dir(cli)?).context("failed to open store")?;
    store.set_read_plan_policy(cli.read_plan);
    Ok(store)
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
    let root = cli.single_dir();
    DatabaseLayout::open_read_only(root.into()).with_context(|| {
        format!(
            "no mtxdb database at `{}` -- run `mtxdb init` first",
            root.display()
        )
    })
}

/// Collect all valid database directories among the CLI's targeted paths,
/// skipping non-directories and invalid databases with warnings.
pub(crate) fn valid_database_dirs(cli: &Cli) -> anyhow::Result<Vec<PathBuf>> {
    let dirs = cli.dirs_or_default();
    let mut valid = Vec::new();
    for dir in &dirs {
        if !dir.is_dir() {
            eprintln!("skipping `{}`: not a directory", dir.display());
            continue;
        }
        match DatabaseLayout::open_read_only(dir.clone()) {
            Ok(_) => valid.push(dir.clone()),
            Err(e) => {
                eprintln!(
                    "skipping `{}`: not a valid mtxdb database ({e})",
                    dir.display()
                );
            }
        }
    }
    if valid.is_empty() {
        bail!(
            "none of the {} specified targets are mtxdb databases",
            dirs.len()
        );
    }
    Ok(valid)
}

fn database_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| path.display().to_string())
}

fn database_labels(paths: &[PathBuf]) -> HashMap<PathBuf, String> {
    let mut map = HashMap::new();
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for path in paths {
        let name = database_label(path);
        let count = name_counts.entry(name).or_insert(0);
        *count = count.saturating_add(1);
    }
    for path in paths {
        let name = database_label(path);
        if name_counts.get(&name).copied().unwrap_or(0) > 1 {
            map.insert(path.clone(), path.display().to_string());
        } else {
            map.insert(path.clone(), name);
        }
    }
    map
}

/// Run a command over all targeted database roots.
///
/// When a single directory (or default `.`) is targeted, runs `f` directly.
/// When multiple directories are targeted (e.g. `mtxdb shards -d *`), iterates over each,
/// printing a section header for each database root and skipping non-databases with a note.
fn run_multi_dir<F>(cli: &Cli, mut f: F) -> anyhow::Result<()>
where
    F: FnMut(&Cli) -> anyhow::Result<()>,
{
    let dirs = cli.dirs_or_default();
    if dirs.len() <= 1 {
        return f(cli);
    }

    let mut executed: usize = 0;
    let mut failed: usize = 0;
    for dir in &dirs {
        if !dir.is_dir() {
            eprintln!("skipping `{}`: not a directory", dir.display());
            continue;
        }
        match DatabaseLayout::open_read_only(dir.clone()) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "skipping `{}`: not a valid mtxdb database ({e})",
                    dir.display()
                );
                continue;
            }
        }

        if executed > 0 {
            println!();
            println!();
        }
        println!("=== Database: {} ===", dir.display());
        executed = executed.saturating_add(1);

        let sub_cli = cli.with_dir(dir.clone());
        if let Err(error) = f(&sub_cli) {
            eprintln!("Error: {error:#}");
            failed = failed.saturating_add(1);
        }
    }

    if executed == 0 {
        bail!(
            "none of the {} specified targets are mtxdb databases",
            dirs.len()
        );
    }
    if failed == executed {
        bail!("all {executed} database targets failed");
    }
    if failed > 0 {
        eprintln!("warning: {failed} of {executed} database targets failed");
    }
    Ok(())
}

/// A pool's directory. A pool is created by its first write, so a pool that was
/// never written has no directory yet and is simply empty: the path is returned
/// either way, and opening it read-only yields an empty store.
#[allow(
    clippy::unnecessary_wraps,
    reason = "about thirty call sites are written against the fallible form, \
              several as `let Ok(..) = .. else { continue }`"
)]
fn pool_dir(layout: &DatabaseLayout, shard_type: ShardType) -> anyhow::Result<PathBuf> {
    Ok(layout.pool_path(shard_type))
}

/// Return a pool that exists on disk, or `None` for one never written. `-t all`
/// skips those, and
/// read-only inspection must not create them.
#[allow(
    clippy::unnecessary_wraps,
    reason = "kept fallible to match `pool_dir` at its call sites"
)]
fn existing_pool_dir(
    layout: &DatabaseLayout,
    shard_type: ShardType,
) -> anyhow::Result<Option<PathBuf>> {
    let path = layout.pool_path(shard_type);
    Ok(path.is_dir().then_some(path))
}

fn selected_pool_dir(cli: &Cli) -> anyhow::Result<PathBuf> {
    pool_dir(&open_layout(cli)?, cli.require_shard_type()?)
}

/// A collection ID not found under `cli.shard_type` is often just in one of
/// the *other* independent shard-type pools (`-t`/`--shard-type` defaults
/// to `event`, so a `state`-only collection ID "not found" there is the
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
            let Ok(store) = open_pool_read_only(dir) else {
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
    let collection_hex = format_id(&collection_id);
    let id_hex = format_id(&node_id);
    eprintln!(
        "put {id_hex} into collection {collection_hex} ({} bytes)",
        data.len()
    );
    Ok(())
}

fn emit_get_data(data: &NodeData, raw: bool, decode: Option<&str>) -> anyhow::Result<()> {
    let emitted = if raw || decode == Some("raw") {
        data.bytes.to_vec()
    } else if let Some(mode) = decode {
        match mode {
            "json" => pretty_json_stream(&data.bytes)
                .ok_or_else(|| anyhow::anyhow!("payload is not valid JSON"))?,
            "hamt" => decode_hamt_root(&data.bytes)
                .or_else(|| decode_hamt_node(&data.bytes))
                .ok_or_else(|| anyhow::anyhow!("payload is not a valid HAMT root or node"))?,
            "state" => {
                if let Some(hamt) =
                    decode_hamt_root(&data.bytes).or_else(|| decode_hamt_node(&data.bytes))
                {
                    hamt
                } else {
                    bail!("payload is not a HAMT root or node");
                }
            }
            "auto" => {
                if let Some(rendered) = pretty_print_payload(&data.bytes) {
                    rendered
                } else {
                    hex_bytes(&data.bytes).into_bytes()
                }
            }
            other => {
                bail!("unsupported decode format `{other}`; choose json, hamt, state, raw, or auto")
            }
        }
    } else if let Some(rendered) = pretty_print_payload(&data.bytes) {
        rendered
    } else {
        hex_bytes(&data.bytes).into_bytes()
    };
    io::stdout().write_all(&emitted)?;
    // Raw output stays byte-exact. All textual output is
    // newline-terminated, including encoded binary fallbacks.
    if !raw && decode != Some("raw") && !emitted.ends_with(b"\n") {
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

fn print_get_header(
    node_id: &[u8; 16],
    collection_id: &[u8; 16],
    shard_type: ShardType,
    data: &NodeData,
    store: Option<&PackfileStorage>,
) {
    eprintln!("header:");
    eprintln!("  collection:  {}", format_id(collection_id));
    eprintln!("  node:        {}", format_id(node_id));
    eprintln!("  pool:        {}", shard_type.as_str());
    eprintln!("  payload:     {} bytes", data.bytes.len());
    if *node_id == mtxdb::COLLECTION_METADATA_RECORD_ID {
        if let Some(meta) = CollectionMetadata::decode(&data.bytes) {
            let pool = meta.member_namespace.as_ref().map_or_else(
                || "none".to_owned(),
                |d| String::from_utf8_lossy(d).into_owned(),
            );
            let role = meta.role.as_ref().map_or_else(
                || "unspecified".to_owned(),
                std::string::ToString::to_string,
            );
            let schema = meta.schema.as_deref().unwrap_or("none");
            let verified = meta.verify_collection_id(collection_id);
            let verify_str = if verified {
                "valid"
            } else {
                "MISMATCH / CORRUPT"
            };
            eprintln!("  genesis metadata:");
            eprintln!("    pool:      {pool}");
            eprintln!("    role:      {role}");
            eprintln!("    schema:    {schema}");
            eprintln!(
                "    canonical: {}",
                String::from_utf8_lossy(&meta.collection_canonical_id)
            );
            eprintln!("    identity:  {verify_str}");
        }
    } else if let Some(store) = store {
        if let Ok(Some(meta)) = store.get_collection_metadata(collection_id) {
            let role = meta.role.as_ref().map_or_else(
                || "unspecified".to_owned(),
                std::string::ToString::to_string,
            );
            let schema = meta.schema.as_deref().unwrap_or("none");
            let verified = meta.verify_collection_id(collection_id);
            let verify_str = if verified {
                "valid"
            } else {
                "MISMATCH / CORRUPT"
            };
            eprintln!("  collection metadata:");
            eprintln!("    role:      {role}");
            eprintln!("    schema:    {schema}");
            eprintln!(
                "    canonical: {}",
                String::from_utf8_lossy(&meta.collection_canonical_id)
            );
            eprintln!("    identity:  {verify_str}");
        }
    }
}

fn print_get_verbose(
    requested_id: &str,
    node_id: &[u8; 16],
    shard_type: ShardType,
    collection_id: &[u8; 16],
    data: &NodeData,
) {
    eprintln!("record:");
    eprintln!("  requested:  {requested_id}");
    eprintln!("  node:       {}", format_id(node_id));
    eprintln!("  collection: {}", format_id(collection_id));
    eprintln!("  pool:       {}", shard_type.as_str());
    eprintln!("  payload:    {} bytes", data.bytes.len());

    let bytes = data.bytes.to_vec();
    let Ok(event) = OwnedValue::parse_bytes(&bytes) else {
        eprintln!("  format:     binary/non-JSON");
        return;
    };
    let Some(kind) = event_string_field(&event, "type") else {
        eprintln!("  format:     JSON");
        return;
    };
    eprintln!("  type:       {kind}");
    for (label, field) in [
        ("event", "event_id"),
        ("room", "room_id"),
        ("sender", "sender"),
        ("state key", "state_key"),
    ] {
        if let Some(value) = event_string_field(&event, field) {
            eprintln!("  {label:<11}{value}");
        }
    }
    if let OwnedValue::Object(fields) = &event {
        if let Some(depth) = fields.get("depth").and_then(OwnedValue::as_i64) {
            eprintln!("  depth:      {depth}");
        }
        if let Some(timestamp) = fields.get("origin_server_ts").and_then(OwnedValue::as_i64) {
            eprintln!("  origin ts:  {timestamp}");
        }
    }
}

fn get_matches_in_store(
    store: &PackfileStorage,
    collection: Option<&str>,
    node_id: &[u8; 16],
) -> anyhow::Result<Vec<([u8; 16], NodeData)>> {
    match collection {
        Some(collection) => {
            let collection_id = parse_collection_selector(collection)?;
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

fn extract_origin_server_ts(bytes: &[u8]) -> Option<u64> {
    let copy = bytes.to_vec();
    let event = OwnedValue::parse_bytes(&copy).ok()?;
    if let OwnedValue::Object(fields) = &event {
        fields
            .get("origin_server_ts")
            .and_then(OwnedValue::as_u64)
            .or_else(|| {
                fields
                    .get("origin_server_ts")
                    .and_then(OwnedValue::as_i64)
                    .and_then(|ts| u64::try_from(ts).ok())
            })
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_get_single(
    cli: &Cli,
    collection: Option<&str>,
    id: &str,
    raw: bool,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
) -> anyhow::Result<()> {
    let node_id = parse_get_id(id)?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matches: Vec<(ShardType, [u8; 16], NodeData)> = Vec::new();
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let Ok(store) = open_pool_read_only(dir) else {
                continue;
            };
            for (col_id, data) in get_matches_in_store(&store, collection, &node_id)? {
                matches.push((shard_type, col_id, data));
            }
        }
        match matches.as_slice() {
            [] => bail!("not found"),
            [(shard_type, collection_id, data)] => {
                if header {
                    let store = pool_dir(&db_layout, *shard_type)
                        .ok()
                        .and_then(|dir| open_pool_read_only(dir).ok());
                    print_get_header(&node_id, collection_id, *shard_type, data, store.as_ref());
                }
                if verbose {
                    print_get_verbose(id, &node_id, *shard_type, collection_id, data);
                }
                emit_get_data(data, raw, decode)
            }
            _ => {
                let locations = matches
                    .iter()
                    .map(|(shard_type, col_id, _)| {
                        format!(
                            "-t {} collection {}",
                            shard_type.as_str(),
                            format_id(col_id)
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
        let matches = get_matches_in_store(&store, collection, &node_id)?;
        match matches.as_slice() {
            [] => bail!("not found{}", other_shard_type_node_hint(cli, &node_id)),
            [(collection_id, data)] => {
                if header {
                    print_get_header(
                        &node_id,
                        collection_id,
                        cli.require_shard_type()?,
                        data,
                        Some(&store),
                    );
                }
                if verbose {
                    print_get_verbose(id, &node_id, cli.require_shard_type()?, collection_id, data);
                }
                emit_get_data(data, raw, decode)
            }
            _ => bail!(
                "node ID {id} is present in multiple collections ({}); specify --collection",
                matches
                    .iter()
                    .map(|(collection_id, _)| format_id(collection_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_get_coalesced(
    cli: &Cli,
    collection: Option<&str>,
    id: &str,
    raw: bool,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
) -> anyhow::Result<()> {
    struct Candidate {
        db_dir: PathBuf,
        shard_type: ShardType,
        collection_id: [u8; 16],
        data: NodeData,
        origin_server_ts: Option<u64>,
    }

    let valid_dirs = valid_database_dirs(cli)?;
    let node_id = parse_get_id(id)?;

    let mut candidates = Vec::new();
    let shard_types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };

    for db_dir in &valid_dirs {
        let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
            continue;
        };
        for &shard_type in &shard_types {
            let Ok(dir) = pool_dir(&layout, shard_type) else {
                continue;
            };
            let Ok(store) = open_pool_read_only(dir) else {
                continue;
            };
            for (col_id, data) in get_matches_in_store(&store, collection, &node_id)? {
                let origin_server_ts = extract_origin_server_ts(&data.bytes);
                candidates.push(Candidate {
                    db_dir: db_dir.clone(),
                    shard_type,
                    collection_id: col_id,
                    data,
                    origin_server_ts,
                });
            }
        }
    }

    if candidates.is_empty() {
        bail!("not found in any of the {} databases", valid_dirs.len());
    }

    let first_payload = &candidates[0].data.bytes;
    let all_identical = candidates.iter().all(|c| c.data.bytes == *first_payload);

    let winner = if all_identical {
        &candidates[0]
    } else {
        let mut sorted_candidates: Vec<&Candidate> = candidates.iter().collect();
        sorted_candidates.sort_by(|a, b| {
            b.origin_server_ts
                .cmp(&a.origin_server_ts)
                .then_with(|| a.shard_type.as_str().cmp(b.shard_type.as_str()))
                .then_with(|| a.collection_id.cmp(&b.collection_id))
                .then_with(|| a.db_dir.to_string_lossy().cmp(&b.db_dir.to_string_lossy()))
                .then_with(|| a.data.bytes.cmp(&b.data.bytes))
        });

        let selected = sorted_candidates[0];
        let distinct_payloads = {
            let mut set = HashSet::new();
            for c in &candidates {
                set.insert(&c.data.bytes);
            }
            set.len()
        };

        let strategy_msg = match selected.origin_server_ts {
            Some(ts) => format!("newest by origin_server_ts: {ts}"),
            None => "deterministic lexical fallback".to_owned(),
        };

        eprintln!(
            "warning: conflicting records found for ID {id} ({distinct_payloads} distinct payloads across {} databases); selecting {strategy_msg} from `{}`",
            candidates.len(),
            selected.db_dir.display()
        );

        selected
    };

    if header {
        let store = DatabaseLayout::open_read_only(winner.db_dir.clone())
            .ok()
            .and_then(|layout| pool_dir(&layout, winner.shard_type).ok())
            .and_then(|dir| open_pool_read_only(dir).ok());
        print_get_header(
            &node_id,
            &winner.collection_id,
            winner.shard_type,
            &winner.data,
            store.as_ref(),
        );
    }

    if verbose {
        print_get_verbose(
            id,
            &node_id,
            winner.shard_type,
            &winner.collection_id,
            &winner.data,
        );
        eprintln!("  database:   {}", winner.db_dir.display());
        eprintln!("  candidates: {} database match(es)", candidates.len());
    }

    emit_get_data(&winner.data, raw, decode)
}

#[allow(clippy::too_many_arguments)]
fn cmd_get(
    cli: &Cli,
    collection: Option<&str>,
    id: &str,
    raw: bool,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
) -> anyhow::Result<()> {
    let dirs = cli.dirs_or_default();
    if dirs.len() > 1 {
        if cli.coalesce {
            return cmd_get_coalesced(cli, collection, id, raw, verbose, header, decode);
        }
        return run_multi_dir(cli, |sub_cli| {
            cmd_get_single(sub_cli, collection, id, raw, verbose, header, decode)
        });
    }
    cmd_get_single(cli, collection, id, raw, verbose, header, decode)
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
        let value = value.to_vec();
        let json = OwnedValue::parse_bytes(&value).ok()?;
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
    let value = value.to_vec();
    let json = OwnedValue::parse_bytes(&value).ok()?;
    Some(json.encode_pp().into_bytes())
}

/// Decode a Synapse state-group HAMT root and pretty-print its structure.
///
/// Wire format (`MTHR`, version 1): a big-endian `u16` room-prefix length,
/// the room-prefix bytes, a big-endian `u16` room-ID length, the UTF-8 room
/// ID, a 32-byte root hash, and a 2048-byte lattice of 1024 little-endian lanes.
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
    let lattice_bytes = bytes.get(lattice_start..end)?;
    let lattice_digest = blake3_digest(lattice_bytes);

    // The lattice is 1024 little-endian u16 lanes; its unkeyed LtHash digest is
    // the state-group id (BLAKE2b over the lanes), the same identity the
    // event->state-group aux index stores.
    let mut lanes = [0u16; 1024];
    for (lane, [lo, hi]) in lanes.iter_mut().zip(lattice_bytes.as_chunks::<2>().0) {
        *lane = u16::from_le_bytes([*lo, *hi]);
    }
    let lattice = rezzy::incremental::LtLattice::<1024>::from(lanes);
    let state_group_id = rezzy::hamt::state_group_id_from_lthash(&lattice);

    let mut out = Vec::new();
    writeln!(out, "// HAMT state-group root (Synapse wire v1)").unwrap();
    writeln!(out, "// room prefix: 0x{}", hex::encode(room_prefix)).unwrap();
    writeln!(out, "// room ID: {room_id}").unwrap();
    writeln!(
        out,
        "// root MTHN node id: 0x{}",
        hex::encode(&root_hash[..16])
    )
    .unwrap();
    writeln!(out, "// root node id: 0x{}", hex::encode(&root_hash[..16])).unwrap();
    writeln!(out, "// lattice: {LATTICE_LEN} bytes (1024 u16 lanes)").unwrap();
    writeln!(
        out,
        "// state-group id (LtHash): 0x{}",
        hex::encode(state_group_id)
    )
    .unwrap();
    writeln!(
        out,
        "// logical digest (BLAKE3): 0x{}",
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
///           K = JSON array string of (EventType, StateKey), the same bytes as
///               Synapse's `serde_json::to_string`, written by `rezzy::json`
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
            "  child[{child_index}] slot={slot}: node_id: 0x{}",
            hex::encode(&hash[..16])
        )
        .unwrap();
        child_index += 1;
    }

    Some(out)
}

/// Decode an `STGP\x01` state-group instance record into a readable form: its
/// parent instance ids, state `LtHash` digest, shared `STAT` root id, and the
/// resolver/trust derivation that produced it.
/// Returns `None` for any other payload.
fn decode_state_group(bytes: &[u8]) -> Option<Vec<u8>> {
    let instance = mtxdb::decode_state_group_record(bytes).ok()?;
    let mut out = Vec::new();
    writeln!(out, "// STGP state-group instance (v1)").unwrap();
    writeln!(
        out,
        "// {} parent(s), {} bytes",
        instance.parents.len(),
        bytes.len()
    )
    .unwrap();
    if instance.parents.is_empty() {
        writeln!(out, "  parents: (none)").unwrap();
    } else {
        writeln!(out, "  parents:").unwrap();
        for parent in &instance.parents {
            writeln!(out, "    0x{}", hex::encode(parent)).unwrap();
        }
    }
    writeln!(
        out,
        "  lthash (state identity): 0x{}",
        hex::encode(instance.lthash)
    )
    .unwrap();
    writeln!(out, "  STAT root id: 0x{}", hex::encode(instance.root_id)).unwrap();
    writeln!(out, "  derivation: {}", instance.derivation.as_str()).unwrap();
    Some(out)
}

/// Try to skip a HAMT leaf encoded as two length-prefixed strings: K (a
/// JSON-encoded `(EventType, StateKey)` array) and V (`EventId`). Advances
/// `cursor` past the leaf on success. This matches
/// `synapse/rust/src/state_hamt.rs`'s `HamtNode<String, String>`, where K is
/// Synapse's `serde_json::to_string(&(event_type, state_key))` (written here by
/// `rezzy::json`), not a raw tuple
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
    let key_bytes = key_json.into_bytes();
    let Ok(value) = OwnedValue::parse_bytes(&key_bytes) else {
        *cursor = start;
        return None;
    };
    let OwnedValue::Array(elements) = value else {
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
    if let Some(state_group) = decode_state_group(bytes) {
        return Some(state_group);
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

/// Resolve the pool set for commands that can list every independent pool.
/// An explicit `--all` must override the CLI's default `event` selection;
/// otherwise `collections --all`/`shards --all` silently list only one pool.
fn listing_shard_types(cli: &Cli, all: bool) -> Vec<ShardType> {
    if all {
        ShardType::ALL.to_vec()
    } else {
        cli.shard_types().collect()
    }
}

#[derive(Clone, Copy)]
struct CollectionFlags(u8);

impl CollectionFlags {
    const ALL: u8 = 1;
    const LAYOUT: u8 = 2;
    const CANONICAL: u8 = 4;
    const PER_DB: u8 = 8;

    fn from_bools(values: [bool; 4]) -> Self {
        let mut bits = 0;
        if values[0] {
            bits |= Self::ALL;
        }
        if values[1] {
            bits |= Self::LAYOUT;
        }
        if values[2] {
            bits |= Self::CANONICAL;
        }
        if values[3] {
            bits |= Self::PER_DB;
        }
        Self(bits)
    }

    fn all(self) -> bool {
        self.0 & Self::ALL != 0
    }

    fn layout(self) -> bool {
        self.0 & Self::LAYOUT != 0
    }

    fn canonical(self) -> bool {
        self.0 & Self::CANONICAL != 0
    }

    fn per_db(self) -> bool {
        self.0 & Self::PER_DB != 0
    }
}

struct CollectionOptions<'a> {
    flags: CollectionFlags,
    sort: Option<&'a str>,
    limit: i64,
}

fn cmd_collections(cli: &Cli, options: &CollectionOptions<'_>) -> anyhow::Result<()> {
    if cli.coalesce {
        let valid_dirs = valid_database_dirs(cli)?;
        if valid_dirs.len() > 1 {
            return cmd_collections_coalesced(cli, &valid_dirs, options);
        }
    }
    run_multi_dir(cli, |sub_cli| {
        cmd_collections_single(
            sub_cli,
            options.flags.all(),
            options.flags.layout(),
            options.flags.canonical(),
            options.sort,
            options.limit,
        )
    })
}

/// Print the collections table header row for the given view. Shared by the
/// coalesced and single-directory report paths so their column layouts cannot
/// drift apart.
fn print_collection_table_header(
    layout: bool,
    canonical: bool,
    canonical_width: usize,
    role_width: usize,
) {
    let canonical_field_width = canonical_width.saturating_add(2).saturating_add(role_width);
    if layout {
        if canonical {
            println!("  {:<34}  {:<canonical_field_width$}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}", "collection", "canonical", "nodes", "idx load", "packs", "disk", "runs", "largest", "avoidable");
        } else {
            println!(
                "  {:<34}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
                "collection", "nodes", "idx load", "packs", "disk", "runs", "largest", "avoidable"
            );
        }
    } else if canonical {
        println!(
            "  {:<34}  {:<canonical_field_width$}  {:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
            "collection", "canonical", "nodes", "idx load", "shards", "index", "disk"
        );
    } else {
        println!(
            "  {:<34}  {:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
            "collection", "nodes", "idx load", "shards", "index", "disk"
        );
    }
}

fn display_collection_role(role: &str) -> &str {
    match role {
        // Keep old stores readable after the role was shortened in the store.
        "event_dag" => "event",
        role => role,
    }
}

/// Return the canonical identity and optional role suffix separately. Metadata
/// currently renders roles as `canonical (role)`; keeping the suffix out of the
/// identity width makes the role column line up across rows.
fn split_canonical_display(display: &str) -> (&str, Option<&str>) {
    let Some((identity, role)) = display.rsplit_once(" (") else {
        return (display, None);
    };
    if role.ends_with(')') || role.ends_with(")*") {
        (identity, Some(role))
    } else {
        (display, None)
    }
}

fn canonical_column_width<'a>(displays: impl IntoIterator<Item = &'a String>) -> usize {
    displays
        .into_iter()
        .map(|display| split_canonical_display(display).0.len())
        .max()
        .unwrap_or(0)
        .max("[unregistered]".len())
        .max("canonical".len())
}

fn canonical_role_width<'a>(displays: impl IntoIterator<Item = &'a String>) -> usize {
    displays
        .into_iter()
        .filter_map(|display| split_canonical_display(display).1)
        .map(|role| role.len().saturating_add(1))
        .max()
        .unwrap_or(0)
}

fn format_canonical_display(display: &str, width: usize, role_width: usize) -> String {
    let (identity, role) = split_canonical_display(display);
    let mut formatted = format!("{identity:<width$}");
    if let Some(role) = role {
        formatted.push_str("  (");
        formatted.push_str(role);
    }
    let used_role_width = role.map_or(0, |role| role.len().saturating_add(1));
    formatted.push_str(&" ".repeat(role_width.saturating_sub(used_role_width)));
    formatted
}

#[allow(
    clippy::too_many_lines,
    reason = "coalesced table construction and formatting is kept together"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the collections flags; a one-off options struct would only rename them"
)]
fn cmd_collections_coalesced(
    cli: &Cli,
    valid_dirs: &[PathBuf],
    options: &CollectionOptions<'_>,
) -> anyhow::Result<()> {
    struct CoalescedCol {
        id: [u8; 16],
        canonical: Option<String>,
        canonical_conflict: bool,
        nodes: usize,
        memory: usize,
        capacity: u32,
        disk_bytes: u64,
        shards_count: usize,
        shards_known: bool,
        runs: u64,
        largest_segment: u64,
        avoidable: u64,
    }

    let all = options.flags.all();
    let layout = options.flags.layout();
    let canonical = options.flags.canonical();
    let per_db = options.flags.per_db();
    let sort = options.sort;
    let limit = options.limit;

    let types = listing_shard_types(cli, all);
    let physical_needed = layout
        || sort.is_some_and(|column| {
            matches!(
                column,
                "disk" | "packs" | "avoidable" | "segments" | "fragmentation"
            )
        });

    for (type_index, shard_type) in types.into_iter().enumerate() {
        if type_index != 0 {
            println!();
            println!();
        }
        print_section_header(shard_type);

        let mut map: HashMap<[u8; 16], CoalescedCol> = HashMap::new();
        let mut total_dbs_with_collections = HashSet::new();
        // (database, collections, nodes, index bytes, disk bytes)
        let mut db_rows: Vec<(PathBuf, usize, usize, usize, u64, usize)> = Vec::new();

        for db_dir in valid_dirs {
            let Ok(layout_db) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                continue;
            };
            let Ok(pool_dir) = pool_dir(&layout_db, shard_type) else {
                continue;
            };
            if glob_pack_files(&pool_dir).map_or(true, |p| p.is_empty()) {
                continue;
            }
            let summaries = match PackfileStorage::collection_summaries_from_disk(&pool_dir) {
                Some(s) => s,
                None => {
                    if let Ok(store) = open_pool_read_only(pool_dir.clone()) {
                        store.collection_summaries()
                    } else {
                        Vec::new()
                    }
                }
            };
            if summaries.is_empty() {
                continue;
            }
            total_dbs_with_collections.insert(db_dir.clone());
            let collection_shards = PackfileStorage::collection_shards_from_disk(&pool_dir);
            let sidecar_disk = PackfileStorage::collection_disk_bytes_from_disk(&pool_dir);
            // A pool with no persisted sidecars (a live writer that has not
            // checkpointed) has no disk or shard figures to read, so scan its
            // packs instead of reporting zero.
            let physical =
                if physical_needed || sidecar_disk.is_none() || collection_shards.is_none() {
                    physical_layout(&pool_dir).ok()
                } else {
                    None
                };
            let canonical_map: HashMap<[u8; 16], String> = if canonical {
                open_pool_read_only(pool_dir.clone())
                    .ok()
                    .map_or_else(HashMap::new, |store| {
                        canonical_collection_ids(&store, &summaries)
                    })
            } else {
                HashMap::new()
            };

            let mut db_row = (db_dir.clone(), 0_usize, 0_usize, 0_usize, 0_u64, 0_usize);
            for (col_id, nodes, memory, capacity) in summaries {
                let disk = physical
                    .as_ref()
                    .and_then(|p| p.collections.get(&col_id))
                    .map(|s| s.disk_bytes)
                    .or_else(|| {
                        sidecar_disk
                            .as_ref()
                            .and_then(|sd| sd.get(&col_id).copied())
                    })
                    .unwrap_or(0);
                let stats = physical.as_ref().and_then(|p| p.collections.get(&col_id));
                let shard_count_in_this_db = collection_shards
                    .as_ref()
                    .and_then(|cs| cs.get(&col_id))
                    .map(|shards| shards.len().max(1));
                let runs = stats.map_or(0, |s| s.segments);
                let largest = stats.map_or(0, |s| s.largest_segment_bytes);
                let avoidable = avoidable_spread_bytes(stats);

                let entry = map.entry(col_id).or_insert_with(|| CoalescedCol {
                    id: col_id,
                    canonical: None,
                    canonical_conflict: false,
                    nodes: 0,
                    memory: 0,
                    capacity: 0,
                    disk_bytes: 0,
                    shards_count: 0,
                    shards_known: true,
                    runs: 0,
                    largest_segment: 0,
                    avoidable: 0,
                });

                db_row.1 = db_row.1.saturating_add(1);
                db_row.2 = db_row.2.saturating_add(nodes);
                db_row.3 = db_row.3.saturating_add(memory);
                db_row.4 = db_row.4.saturating_add(disk);
                db_row.5 = if db_row.5 == 0 {
                    memory
                } else {
                    db_row.5.min(memory)
                };
                entry.nodes = entry.nodes.saturating_add(nodes);
                entry.memory = entry.memory.saturating_add(memory);
                entry.capacity = entry.capacity.saturating_add(capacity);
                entry.disk_bytes = entry.disk_bytes.saturating_add(disk);
                if let Some(shard_count) = shard_count_in_this_db {
                    entry.shards_count = entry.shards_count.saturating_add(shard_count);
                } else {
                    entry.shards_known = false;
                }
                entry.runs = entry.runs.saturating_add(runs);
                entry.largest_segment = entry.largest_segment.max(largest);
                entry.avoidable = entry.avoidable.saturating_add(avoidable);
                if let Some(c) = canonical_map.get(&col_id) {
                    if let Some(existing) = &entry.canonical {
                        if existing == "[unregistered]" && c != "[unregistered]" {
                            entry.canonical = Some(c.clone());
                        } else if existing != "[unregistered]"
                            && c != "[unregistered]"
                            && existing != c
                        {
                            entry.canonical_conflict = true;
                            eprintln!(
                                "warning: conflicting canonical IDs for collection {}: `{existing}` vs `{c}`",
                                format_id(&col_id)
                            );
                        }
                    } else {
                        entry.canonical = Some(c.clone());
                    }
                }
            }
            db_rows.push(db_row);
        }

        if map.is_empty() {
            println!(
                "no collections found across {} database(s)",
                valid_dirs.len()
            );
            continue;
        }

        let mut ordered: Vec<CoalescedCol> = map.into_values().collect();
        if let Some(column) = sort {
            if !matches!(
                column,
                "collection"
                    | "nodes"
                    | "shards"
                    | "index"
                    | "idx-load"
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
        ordered.sort_by(|left, right| {
            let ordering = match sort.unwrap_or("") {
                "nodes" => right.nodes.cmp(&left.nodes),
                "shards" | "packs" => right.shards_count.cmp(&left.shards_count),
                "index" => right.memory.cmp(&left.memory),
                "idx-load" | "load" => load_factor_percent(right.nodes, right.capacity)
                    .total_cmp(&load_factor_percent(left.nodes, left.capacity)),
                "disk" => right.disk_bytes.cmp(&left.disk_bytes),
                "avoidable" => right.avoidable.cmp(&left.avoidable),
                "segments" | "fragmentation" => right.runs.cmp(&left.runs),
                _ => left.id.cmp(&right.id),
            };
            ordering.then_with(|| left.id.cmp(&right.id))
        });

        let canonical_width =
            canonical_column_width(ordered.iter().filter_map(|c| c.canonical.as_ref()));
        let role_width = canonical_role_width(ordered.iter().filter_map(|c| c.canonical.as_ref()));

        print_collection_table_header(layout, canonical, canonical_width, role_width);

        let mut total_nodes = 0_usize;
        let mut total_memory = 0_usize;
        let mut total_disk_bytes = 0_u64;
        let mut total_capacity = 0_u64;
        let total_rows = ordered.len();
        for item in &ordered {
            total_capacity = total_capacity.saturating_add(u64::from(item.capacity));
            total_nodes = total_nodes.saturating_add(item.nodes);
            total_memory = total_memory.saturating_add(item.memory);
            total_disk_bytes = total_disk_bytes.saturating_add(item.disk_bytes);
        }

        let max_rows = if limit <= 0 {
            usize::MAX
        } else {
            usize::try_from(limit).unwrap_or(usize::MAX)
        };

        let has_canonical_conflict = ordered.iter().any(|c| c.canonical_conflict);

        for item in ordered.into_iter().take(max_rows) {
            let hex = format_id(&item.id);
            let canonical_id = if item.canonical_conflict {
                format!("{}*", item.canonical.as_deref().unwrap_or("[unregistered]"))
            } else {
                item.canonical
                    .as_deref()
                    .unwrap_or("[unregistered]")
                    .to_owned()
            };
            let canonical_display =
                format_canonical_display(&canonical_id, canonical_width, role_width);
            let load = fmt_load_percent(item.nodes, item.capacity);
            let shards = if !item.shards_known {
                "?".to_owned()
            } else if item.shards_count == 1 {
                String::new()
            } else {
                item.shards_count.to_string()
            };
            let disk_display = fmt_disk_megabytes(item.disk_bytes);
            if layout {
                let avoidable_str = if item.avoidable > 0 {
                    fmt_disk_megabytes(item.avoidable)
                } else {
                    "-".to_owned()
                };
                if canonical {
                    println!(
                        "  {hex:<34}  {canonical_display}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
                        item.nodes, load, shards, disk_display, item.runs, fmt_bytes(item.largest_segment), avoidable_str
                    );
                } else {
                    println!(
                        "  {hex:<34}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
                        item.nodes,
                        load,
                        shards,
                        disk_display,
                        item.runs,
                        fmt_bytes(item.largest_segment),
                        avoidable_str
                    );
                }
            } else if canonical {
                println!(
                    "  {hex:<34}  {canonical_display}  {:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
                    item.nodes,
                    load,
                    shards,
                    fmt_megabytes(item.memory),
                    disk_display
                );
            } else {
                println!(
                    "  {hex:<34}  {:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
                    item.nodes,
                    load,
                    shards,
                    fmt_megabytes(item.memory),
                    disk_display
                );
            }
        }
        println!();
        if per_db {
            print_per_db_totals(&db_rows);
            println!();
        }
        let db_count = total_dbs_with_collections.len();
        let per_node = |bytes: u64| {
            if total_nodes == 0 {
                "-".to_owned()
            } else {
                format!(
                    "{} B/node",
                    bytes
                        .checked_div(u64::try_from(total_nodes).expect("node count fits u64"))
                        .expect("total nodes is nonzero")
                )
            }
        };
        let load = if total_capacity == 0 {
            "-".to_owned()
        } else {
            let tenths = u128::try_from(total_nodes)
                .expect("node count fits u128")
                .checked_mul(1000)
                .expect("scaled node count fits u128")
                .checked_div(u128::from(total_capacity))
                .expect("total capacity is nonzero");
            format!("{}.{:01}% load", tenths / 10, tenths % 10)
        };
        println!("summary");
        println!(
            "  collections   {} distinct across {db_count} database(s)",
            fmt_thousands(total_rows as u64)
        );
        println!(
            "  nodes         {} (summed over databases; replicas counted separately)",
            fmt_thousands(total_nodes as u64)
        );
        println!(
            "  index memory  {} (hash-table capacity; {}, {load})",
            fmt_size_decimal(total_memory as u64),
            per_node(total_memory as u64)
        );
        println!(
            "  disk          {} (physical, incl. superseded frames; {})",
            fmt_size_decimal(total_disk_bytes),
            per_node(total_disk_bytes)
        );
        let index_count: u64 = db_rows.iter().map(|r| r.1 as u64).sum();
        let smallest_index = db_rows
            .iter()
            .filter_map(|row| (row.5 > 0).then_some(row.5))
            .min()
            .unwrap_or(0);
        if total_memory as u64 > total_disk_bytes && smallest_index > 0 {
            println!(
                "  note          index exceeds disk: each per-database collection index has a minimum size ({} x {} = {})",
                fmt_thousands(index_count),
                fmt_size_decimal(smallest_index as u64),
                fmt_size_decimal((smallest_index as u64).saturating_mul(index_count))
            );
        }
        if has_canonical_conflict {
            println!("* conflicting canonical collection IDs detected across databases");
        }
    }
    Ok(())
}

fn cmd_collections_single(
    cli: &Cli,
    all: bool,
    layout: bool,
    canonical: bool,
    sort: Option<&str>,
    limit: i64,
) -> anyhow::Result<()> {
    // `--all` explicitly asks for every pool; `-t all` (no specific shard
    // type selected) means the same thing.
    if all || cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let types = listing_shard_types(cli, all);
        for (index, shard_type) in types.into_iter().enumerate() {
            if index != 0 {
                println!();
                println!();
            }
            print_section_header(shard_type);
            cmd_collections_in_dir(
                &pool_dir(&db_layout, shard_type)?,
                layout,
                canonical,
                sort,
                limit,
            )?;
        }
        return Ok(());
    }
    cmd_collections_in_dir(&selected_pool_dir(cli)?, layout, canonical, sort, limit)
}

fn canonical_collection_ids(
    store: &PackfileStorage,
    collections: &[CollectionSummary],
) -> HashMap<[u8; 16], String> {
    collections
        .iter()
        .map(|(id, _, _, _)| {
            let display = store
                .get_collection_metadata(id)
                .ok()
                .flatten()
                .map_or_else(
                    || "[unregistered]".to_owned(),
                    |metadata| {
                        let raw = String::from_utf8_lossy(&metadata.collection_canonical_id);
                        if metadata.verify_collection_id(id) {
                            metadata.role.as_ref().map_or_else(
                                || raw.clone().into_owned(),
                                |role| format!("{raw} ({})", display_collection_role(role)),
                            )
                        } else {
                            eprintln!(
                                "warning: collection {} canonical ID `{raw}` fails derivation verification (CORRUPT)",
                                format_id(id)
                            );
                            format!("[mismatch: {raw}]")
                        }
                    },
                );
            (*id, display)
        })
        .collect()
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
    canonical: bool,
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
        None => open_pool_read_only(dir.into())
            .context("failed to open store")?
            .collection_summaries(),
    };
    let collection_shards = PackfileStorage::collection_shards_from_disk(dir);
    let physical_needed = layout
        || sort.is_some_and(|column| {
            matches!(
                column,
                "disk" | "packs" | "avoidable" | "segments" | "fragmentation"
            )
        });
    let sidecar_disk = PackfileStorage::collection_disk_bytes_from_disk(dir);
    // Without persisted sidecars (a live writer that has not checkpointed)
    // there is nothing to read disk or shard figures from; scan the packs
    // rather than print `?`, as the coalesced report does.
    let physical = if physical_needed || sidecar_disk.is_none() || collection_shards.is_none() {
        physical_layout(dir).ok()
    } else {
        None
    };
    if physical_needed && physical.is_none() {
        bail!(
            "failed to scan the physical pack layout of `{}`",
            dir.display()
        );
    }
    let disk_bytes: HashMap<_, _> = physical
        .as_ref()
        .map(|layout| {
            layout
                .collections
                .iter()
                .map(|(id, stats)| (*id, stats.disk_bytes))
                .collect()
        })
        .or(sidecar_disk.clone())
        .unwrap_or_default();
    let disk_known = physical.is_some() || sidecar_disk.is_some();

    if collections.is_empty() {
        println!("no collections found");
        return Ok(());
    }

    if let Some(column) = sort {
        if !matches!(
            column,
            "collection"
                | "nodes"
                | "shards"
                | "index"
                | "idx-load"
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
    ordered.sort_by(|(left_order, left), (right_order, right)| {
        let left_layout = physical.as_ref().and_then(|p| p.collections.get(&left.0));
        let right_layout = physical.as_ref().and_then(|p| p.collections.get(&right.0));
        let score = |stats: Option<&CollectionPhysicalLayout>| {
            stats.map_or(0, |s| s.segments.saturating_sub(s.pack_bytes.len() as u64))
        };
        let ordering = match sort.unwrap_or("") {
            "collection" => left.0.cmp(&right.0),
            "nodes" => right.1.cmp(&left.1),
            "shards" | "packs" => right_layout
                .map_or(0, |s| s.pack_bytes.len())
                .cmp(&left_layout.map_or(0, |s| s.pack_bytes.len())),
            "index" => right.2.cmp(&left.2),
            "idx-load" | "load" => load_factor_percent(right.1, right.3)
                .total_cmp(&load_factor_percent(left.1, left.3)),
            "disk" => right_layout
                .map_or(0, |s| s.disk_bytes)
                .cmp(&left_layout.map_or(0, |s| s.disk_bytes)),
            "avoidable" => {
                avoidable_spread_bytes(right_layout).cmp(&avoidable_spread_bytes(left_layout))
            }
            "segments" | "fragmentation" => score(right_layout).cmp(&score(left_layout)),
            _ => left_order.cmp(right_order),
        };
        ordering.then_with(|| left.0.cmp(&right.0))
    });
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
    let displayed: Vec<_> = ordered.into_iter().take(max_rows).collect();
    let canonical_ids: HashMap<[u8; 16], String> = if canonical {
        let displayed_summaries: Vec<_> = displayed.iter().map(|(_, s)| **s).collect();
        open_pool_read_only(dir.to_path_buf())
            .ok()
            .map_or_else(HashMap::new, |store| {
                canonical_collection_ids(&store, &displayed_summaries)
            })
    } else {
        HashMap::new()
    };
    let canonical_width = canonical_column_width(canonical_ids.values());
    let role_width = canonical_role_width(canonical_ids.values());
    let canonical_field_width = canonical_width.saturating_add(2).saturating_add(role_width);
    print_collection_table_header(layout, canonical, canonical_width, role_width);
    for (_, (collection_id, nodes, memory, capacity)) in displayed {
        let hex = format_id(collection_id);
        let canonical_id = canonical_ids
            .get(collection_id)
            .map_or("[unregistered]", String::as_str);
        let canonical_display = format_canonical_display(canonical_id, canonical_width, role_width);
        let load = fmt_load_percent(*nodes, *capacity);
        let shard_count = collection_shards
            .as_ref()
            .and_then(|by_collection| by_collection.get(collection_id))
            .map(Vec::len);
        let shards = shard_count.map_or_else(
            || "?".to_owned(),
            |count| {
                if count == 1 {
                    String::new()
                } else {
                    count.to_string()
                }
            },
        );
        let disk = disk_bytes.get(collection_id).copied().unwrap_or(0);
        let disk_display = if disk_known {
            fmt_disk_megabytes(disk)
        } else {
            "?".to_owned()
        };
        if layout {
            let stats = physical
                .as_ref()
                .and_then(|p| p.collections.get(collection_id));
            let packs = stats.map_or(0, |s| s.pack_bytes.len());
            let runs = stats.map_or(0, |s| s.segments);
            let largest = stats.map_or(0, |s| s.largest_segment_bytes);
            let avoidable = avoidable_spread_bytes(stats);
            if canonical {
                println!("  {hex:<34}  {canonical_display}  {nodes:>7}  {load:>8}  {packs:>6}  {:>13}  {runs:>5}  {:>10}  {:>13}", disk_display, fmt_bytes(largest), fmt_bytes(avoidable));
            } else {
                println!("  {hex:<34}  {nodes:>7}  {load:>8}  {packs:>6}  {:>13}  {runs:>5}  {:>10}  {:>13}", disk_display, fmt_bytes(largest), fmt_bytes(avoidable));
            }
        } else if canonical {
            println!("  {hex:<34}  {canonical_display}  {nodes:>7}  {load:>8}  {shards:>6}  {:>12}  {:>13}", fmt_index_kilobytes(*memory), disk_display);
        } else {
            println!(
                "  {hex}  {nodes:>7}  {load:>8}  {shards:>6}  {:>12}  {:>13}",
                fmt_index_kilobytes(*memory),
                disk_display
            );
        }
    }
    println!();
    if layout {
        if canonical {
            println!(
                "  {:<34}  {:<canonical_field_width$}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
                "total",
                "",
                total_nodes,
                "",
                "",
                if disk_known {
                    fmt_disk_megabytes(total_disk_bytes)
                } else {
                    "?".to_owned()
                },
                "",
                "",
                ""
            );
        } else {
            println!(
                "  {:<34}  {:>7}  {:>8}  {:>6}  {:>13}  {:>5}  {:>10}  {:>13}",
                "total",
                total_nodes,
                "",
                "",
                if disk_known {
                    fmt_disk_megabytes(total_disk_bytes)
                } else {
                    "?".to_owned()
                },
                "",
                "",
                ""
            );
        }
    } else if canonical {
        println!(
            "  {:<34}  {:<canonical_field_width$}  {total_nodes:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
            "total",
            "",
            "",
            "",
            fmt_index_kilobytes(total_memory),
            if disk_known {
                fmt_disk_megabytes(total_disk_bytes)
            } else {
                "?".to_owned()
            },
        );
    } else {
        println!(
            "  {:<34}  {total_nodes:>7}  {:>8}  {:>6}  {:>12}  {:>13}",
            "total",
            "",
            "",
            fmt_index_kilobytes(total_memory),
            if disk_known {
                fmt_disk_megabytes(total_disk_bytes)
            } else {
                "?".to_owned()
            },
        );
    }
    if layout {
        let physical = physical.as_ref().expect("layout requires physical scan");
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
    // A pool never written has no directory and no shard headers to validate.
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_none_or(|extension| extension != "pack") {
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
/// e.g. `event` -> `EVENT`), another rule, then a blank line.
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
    shard_entries: &[(PackId, u64, u8)],
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
        "{:>21}  {:>11}  {:>6}  {:>10}  {:>12}",
        "pack", "collections", "runs", "excess", "largest run"
    );
    for (pack_id, _, _) in shard_entries {
        let stats = physical.packs.get(pack_id);
        let collections = stats.map_or(0, |stats| stats.collections.len());
        let runs = stats.map_or(0, |stats| stats.segments);
        let excess = runs.saturating_sub(collections as u64);
        let largest = stats.map_or(0, |stats| stats.largest_segment_bytes);
        println!(
            "0x{}  {collections:>11}  {runs:>6}  {excess:>10}  {:>12}",
            &pack_id.as_hex()[..16],
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
    if cli.coalesce {
        let valid_dirs = valid_database_dirs(cli)?;
        if valid_dirs.len() > 1 {
            return cmd_shards_coalesced(cli, &valid_dirs, all, layout, sort);
        }
    }
    run_multi_dir(cli, |sub_cli| cmd_shards_single(sub_cli, all, layout, sort))
}

#[allow(
    clippy::too_many_lines,
    reason = "coalesced shard table construction and formatting is kept together"
)]
fn cmd_shards_coalesced(
    cli: &Cli,
    valid_dirs: &[PathBuf],
    all: bool,
    layout: bool,
    sort: Option<&str>,
) -> anyhow::Result<()> {
    struct ShardRow {
        db_label: String,
        pack_id: PackId,
        file_bytes: u64,
        version: u8,
        is_active: bool,
        nodes: Option<u64>,
        collections: Option<u64>,
        index_bytes: Option<usize>,
        syncs: u64,
        segments: u64,
        interleaving: u64,
    }

    let types = listing_shard_types(cli, all);
    let labels = database_labels(valid_dirs);
    let needs_layout = layout || matches!(sort, Some("segments" | "interleaving"));

    for (type_index, shard_type) in types.into_iter().enumerate() {
        if type_index != 0 {
            println!();
            println!();
        }
        print_section_header(shard_type);

        let mut rows = Vec::new();
        let mut unique_collections = HashSet::new();
        let mut total_bytes: u64 = 0;
        let mut total_syncs: u64 = 0;
        let mut total_nodes: u64 = 0;
        let mut has_nodes = false;
        let mut total_index_bytes: usize = 0;
        let mut dbs_with_packs = HashSet::new();

        for db_dir in valid_dirs {
            let Ok(layout_db) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                continue;
            };
            let Ok(pool_dir) = pool_dir(&layout_db, shard_type) else {
                continue;
            };
            let Ok(mut shard_entries) = glob_pack_files(&pool_dir) else {
                continue;
            };
            if shard_entries.is_empty() {
                continue;
            }
            shard_entries.sort_unstable_by_key(|&(id, _, _)| id);
            let active_pack_id = shard_entries.last().map(|(id, _, _)| *id);
            let (stats_map, _) = decode_stats_snapshot(&pool_dir);
            let node_counts = PackfileStorage::shard_node_counts_from_disk(&pool_dir);
            let collection_counts = PackfileStorage::shard_collection_counts_from_disk(&pool_dir);
            let physical = if needs_layout {
                physical_layout(&pool_dir).ok()
            } else {
                None
            };
            let (pool_total_index, index_by_shard) = index_requirements_from_disk(&pool_dir)
                .map_or((0, None), |(total, by_shard)| (total, Some(by_shard)));
            total_index_bytes = total_index_bytes.saturating_add(pool_total_index);

            for (col_id, _) in PackfileStorage::collection_directory_from_disk(&pool_dir) {
                unique_collections.insert(col_id);
            }

            dbs_with_packs.insert(db_dir.clone());
            let db_label = labels
                .get(db_dir)
                .cloned()
                .unwrap_or_else(|| db_dir.display().to_string());

            for &(pack_id, file_bytes, version) in &shard_entries {
                let (_, _, sc) = stats_map.get(&pack_id).copied().unwrap_or_default();
                let nodes = node_counts.as_ref().and_then(|c| c.get(&pack_id)).copied();
                if let Some(n) = nodes {
                    total_nodes = total_nodes.saturating_add(n);
                    has_nodes = true;
                }
                let collections = collection_counts
                    .as_ref()
                    .and_then(|c| c.get(&pack_id))
                    .copied();
                let index_bytes = index_by_shard
                    .as_ref()
                    .and_then(|r| r.get(&pack_id))
                    .copied();
                let segments = physical
                    .as_ref()
                    .and_then(|p| p.packs.get(&pack_id))
                    .map_or(0, |s| s.segments);
                let interleaving = physical
                    .as_ref()
                    .and_then(|p| p.packs.get(&pack_id))
                    .map_or(0, |s| s.segments.saturating_sub(s.collections.len() as u64));

                total_bytes = total_bytes.saturating_add(file_bytes);
                total_syncs = total_syncs.saturating_add(sc);

                rows.push(ShardRow {
                    db_label: db_label.clone(),
                    pack_id,
                    file_bytes,
                    version,
                    is_active: active_pack_id == Some(pack_id),
                    nodes,
                    collections,
                    index_bytes,
                    syncs: sc,
                    segments,
                    interleaving,
                });
            }
        }

        if rows.is_empty() {
            println!(
                "no shards found across {} database(s) (stores are empty)",
                valid_dirs.len()
            );
            continue;
        }

        if let Some(column) = sort {
            if !matches!(
                column,
                "database"
                    | "pack"
                    | "bytes"
                    | "nodes"
                    | "collections"
                    | "syncs"
                    | "segments"
                    | "interleaving"
            ) {
                bail!("unknown shards sort column `{column}`");
            }
            rows.sort_by(|left, right| {
                let ordering = match column {
                    "database" => left.db_label.cmp(&right.db_label),
                    "pack" => left.pack_id.cmp(&right.pack_id),
                    "bytes" => right.file_bytes.cmp(&left.file_bytes),
                    "nodes" => right.nodes.unwrap_or(0).cmp(&left.nodes.unwrap_or(0)),
                    "collections" => right
                        .collections
                        .unwrap_or(0)
                        .cmp(&left.collections.unwrap_or(0)),
                    "syncs" => right.syncs.cmp(&left.syncs),
                    "segments" => right.segments.cmp(&left.segments),
                    "interleaving" => right.interleaving.cmp(&left.interleaving),
                    _ => std::cmp::Ordering::Equal,
                };
                ordering
                    .then_with(|| left.db_label.cmp(&right.db_label))
                    .then_with(|| left.pack_id.cmp(&right.pack_id))
            });
        } else {
            rows.sort_by(|left, right| {
                left.db_label
                    .cmp(&right.db_label)
                    .then_with(|| left.pack_id.cmp(&right.pack_id))
            });
        }

        let db_width = rows
            .iter()
            .map(|r| r.db_label.len())
            .max()
            .unwrap_or(8)
            .max(8);

        println!(
            "{:<db_width$}  {:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
            "database", "pack", "ver", "bytes", "nodes", "collections", "index", "syncs",
        );
        for row in &rows {
            let nodes_str = row.nodes.map_or_else(|| "?".to_owned(), |n| n.to_string());
            let cols_str = row
                .collections
                .map_or_else(|| "?".to_owned(), |c| c.to_string());
            let index_str = row
                .index_bytes
                .map_or_else(|| "?".to_owned(), fmt_megabytes);
            println!(
                "{:<db_width$}  {:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
                row.db_label,
                format!(
                    "0x{}{}",
                    &row.pack_id.as_hex()[..16],
                    if row.is_active { " *" } else { "  " }
                ),
                row.version,
                fmt_bytes(row.file_bytes),
                nodes_str,
                cols_str,
                index_str,
                row.syncs,
            );
        }
        println!();
        println!(
            "{:<db_width$}  {:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
            "total",
            "",
            "",
            fmt_bytes(total_bytes),
            if has_nodes {
                total_nodes.to_string()
            } else {
                "?".to_owned()
            },
            unique_collections.len().to_string(),
            fmt_megabytes(total_index_bytes),
            total_syncs,
        );
        println!("* active pack in database");
        println!(
            "{} pack(s) across {} database(s)",
            rows.len(),
            dbs_with_packs.len()
        );
        println!(
            "note: unique collections deduplicated, pack/index/node counts summed across databases"
        );
    }
    Ok(())
}

fn cmd_shards_single(cli: &Cli, all: bool, layout: bool, sort: Option<&str>) -> anyhow::Result<()> {
    // See the matching comment in `cmd_collections`: `-t all` must behave
    // like `--all`, not list only the default event pool.
    if all || cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let types = listing_shard_types(cli, all);
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
        let store = open_pool_read_only(dir.to_path_buf()).context("failed to open store")?;
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
    if cli.coalesce {
        let valid_dirs = valid_database_dirs(cli)?;
        if valid_dirs.len() > 1 {
            return cmd_stats_coalesced(cli, &valid_dirs, json);
        }
    }
    run_multi_dir(cli, |sub_cli| cmd_stats_single(sub_cli, json))
}

fn cmd_memory(cli: &Cli, evict: bool, evict_pages: bool) -> anyhow::Result<()> {
    run_multi_dir(cli, |sub_cli| {
        cmd_memory_single(sub_cli, evict, evict_pages)
    })
}

fn cmd_memory_single(cli: &Cli, evict: bool, evict_pages: bool) -> anyhow::Result<()> {
    let layout = open_layout(cli)?;
    if evict_pages {
        let (files, bytes) = evict_database_pages(layout.root())?;
        println!(
            "{} — requested page-cache eviction for {files} files ({} total file bytes)",
            layout.root().display(),
            fmt_bytes(bytes),
        );
        println!("note: eviction is advisory; dirty or actively mapped pages may remain");
        return Ok(());
    }

    let selected = cli.shard_type;
    let mut reported = false;

    for shard_type in cli.shard_types() {
        let pool = if selected.is_some() {
            Some(pool_dir(&layout, shard_type)?)
        } else {
            existing_pool_dir(&layout, shard_type)?
        };
        let Some(pool) = pool else {
            continue;
        };
        let store = open_pool_read_only(pool.clone()).context("failed to open store")?;
        let before = store.resident_cache_stats();
        let evicted = if evict {
            store.evict_resident_caches()
        } else {
            0
        };
        let after = store.resident_cache_stats();
        if evict {
            println!(
                "{} ({}) — evicted {} decoded nodes; {} remain across {} collection caches (hits={}, misses={})",
                pool.display(),
                shard_type.as_str(),
                evicted,
                after.entries,
                after.collections,
                after.hits,
                after.misses,
            );
        } else {
            println!(
                "{} ({}) — {} decoded nodes resident across {} collection caches (hits={}, misses={})",
                pool.display(),
                shard_type.as_str(),
                before.entries,
                before.collections,
                before.hits,
                before.misses,
            );
        }
        reported = true;
    }

    if !reported && selected.is_none() {
        bail!("no shard pools found")
    }
    Ok(())
}

/// Ask Linux to discard cached pages for regular files under a database root.
/// This walks paths directly instead of opening `PackfileStorage`, which would
/// rebuild/load its index and warm the pages this command is trying to evict.
fn evict_database_pages(root: &Path) -> anyhow::Result<(usize, u64)> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = root;
        bail!("`memory --evict-pages` is currently supported only on Linux");
    }

    #[cfg(target_os = "linux")]
    {
        let mut pending = vec![root.to_path_buf()];
        let mut files = 0usize;
        let mut bytes = 0u64;
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir)
                .with_context(|| format!("reading database directory {}", dir.display()))?
            {
                let entry = entry?;
                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                let path = entry.path();
                let metadata = entry.metadata()?;
                let file = fs::File::open(&path).with_context(|| {
                    format!("opening {} for page-cache eviction", path.display())
                })?;
                // The kernel skips dirty pages, so flush first; a failed
                // flush only leaves some pages resident.
                let _ = file.sync_data();
                rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::DontNeed).with_context(
                    || format!("requesting page-cache eviction for {}", path.display()),
                )?;
                files = files.saturating_add(1);
                bytes = bytes.saturating_add(metadata.len());
            }
        }
        Ok((files, bytes))
    }
}

fn cmd_timeline(cli: &Cli, room: &str, limit: i64, reverse: bool) -> anyhow::Result<()> {
    run_multi_dir(cli, |sub_cli| {
        cmd_timeline_single(sub_cli, room, limit, reverse)
    })
}

fn cmd_timeline_single(cli: &Cli, room: &str, limit: i64, reverse: bool) -> anyhow::Result<()> {
    let layout = open_layout(cli)?;
    let pool = pool_dir(&layout, ShardType::EventDag)?;
    let store = open_pool_read_only(pool).context("failed to open EventDag store")?;
    let timeline_index = mtxdb::timeline::TimelineIndex::open(&store, room);
    let room_scope = mtxdb::timeline::room_scope_id(room.as_bytes());
    let limit_usize = usize::try_from(limit.max(0)).unwrap_or(usize::MAX);
    let page = timeline_index.page(&room_scope, None, !reverse, limit_usize)?;
    if page.entries.is_empty() {
        println!("No timeline entries found for room {room}");
        return Ok(());
    }
    println!(
        "{:<6} {:<12} {:<12} {:<34} event_id",
        "idx", "depth", "stream", "node_id"
    );
    println!("{}", "-".repeat(90));
    for (idx, entry) in page.entries.iter().enumerate() {
        let node_hex = hex::encode(entry.node_id);
        let locator_col = mtxdb::interop::event_locator_collection_id(&entry.node_id);
        let event_id_str = store
            .get(&locator_col, &entry.node_id)?
            .and_then(|data| {
                let (eid, _) = mtxdb::interop::decode_event_locator(&data.bytes).ok()?;
                std::str::from_utf8(eid).ok().map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "-".to_owned());
        println!(
            "{:<6} {:<12} {:<12} {:<34} {}",
            idx.saturating_add(1),
            entry.topological_ordering,
            entry.stream_ordering,
            node_hex,
            event_id_str
        );
    }
    if page.next_cursor.is_some() {
        println!("\n(more events available)");
    }
    Ok(())
}

/// Query or rebuild a room's flag index.
///
/// The flag-index API reads through `Database`, so even `--count` and a named
/// query open the database and take the exclusive root writer lock. They fail
/// while another live process holds it; there is no read-only flag path yet.
fn cmd_flags(
    cli: &Cli,
    room: &str,
    flag_name: Option<&str>,
    not: bool,
    count_only: bool,
    limit: usize,
    rebuild: bool,
) -> anyhow::Result<()> {
    run_multi_dir(cli, |sub_cli| {
        cmd_flags_single(sub_cli, room, flag_name, not, count_only, limit, rebuild)
    })
}

fn parse_flag(name: &str) -> anyhow::Result<mtxdb::room_flag::Flag> {
    let normalized = name.to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "present" => Ok(mtxdb::room_flag::Flag::Present),
        "timeline" => Ok(mtxdb::room_flag::Flag::Timeline),
        "outlier" => Ok(mtxdb::room_flag::Flag::Outlier),
        "rejected" => Ok(mtxdb::room_flag::Flag::Rejected),
        "soft-failed" | "soft-fail" | "softfail" => Ok(mtxdb::room_flag::Flag::SoftFailed),
        "redacted" => Ok(mtxdb::room_flag::Flag::Redacted),
        other if other.starts_with("custom:") || other.starts_with("custom-") => {
            let num_str = other.trim_start_matches("custom:").trim_start_matches("custom-");
            let idx = num_str.parse::<u8>().map_err(|_| anyhow::anyhow!("invalid custom flag index '{num_str}'"))?;
            mtxdb::room_flag::Flag::custom(idx).ok_or_else(|| anyhow::anyhow!("custom flag index {idx} exceeds maximum {}", mtxdb::room_flag::Flag::MAX_CUSTOM))
        }
        other => anyhow::bail!(
            "unknown flag '{other}'; valid flags: present, timeline, outlier, rejected, soft-failed, redacted, custom:<0..=191>"
        ),
    }
}

fn cmd_flags_single(
    cli: &Cli,
    room: &str,
    flag_name: Option<&str>,
    not: bool,
    count_only: bool,
    limit: usize,
    rebuild: bool,
) -> anyhow::Result<()> {
    let layout = open_layout(cli)?;
    let db = open_database(layout.root().to_path_buf())
        .with_context(|| format!("opening database at {}", layout.root().display()))?;
    let index = mtxdb::room_flag::RoomFlagIndex::new(room.as_bytes());

    if rebuild {
        println!("Rebuilding flag index for room {room}...");
        let start = std::time::Instant::now();
        let count = index.rebuild(&db)?;
        let elapsed = start.elapsed();
        println!("Rebuilt flag index for room {room}: {count} events indexed in {elapsed:?}");
        return Ok(());
    }

    if let Some(name) = flag_name {
        let flag = parse_flag(name)?;
        let query = if not {
            mtxdb::room_flag::FlagQuery::not(flag)
        } else {
            mtxdb::room_flag::FlagQuery::has(flag)
        };

        let label = if not {
            format!("not({})", flag.as_str())
        } else {
            flag.as_str().to_owned()
        };

        if count_only {
            let start = std::time::Instant::now();
            let count = index.count(&db, &query)?;
            let elapsed = start.elapsed();
            println!("{room} [{label}]: {count} events ({elapsed:?})");
        } else {
            let start = std::time::Instant::now();
            let events = index.select_bounded(&db, &query, limit)?;
            let elapsed = start.elapsed();
            println!(
                "{room} [{label}]: found {} events (limited to {limit}) in {elapsed:?}:",
                events.len()
            );
            for (idx, eid) in events.iter().enumerate() {
                let id_str = String::from_utf8_lossy(eid);
                println!("  {:>4}. {id_str}", idx.saturating_add(1));
            }
        }
        return Ok(());
    }

    // Default: Summary of all standard flags
    println!("Flag summary for room {room}:");
    println!("{:<14} {:>10} {:>14}", "Flag", "Count", "Query Time");
    println!("{}", "-".repeat(42));

    let flags = [
        mtxdb::room_flag::Flag::Present,
        mtxdb::room_flag::Flag::Timeline,
        mtxdb::room_flag::Flag::Outlier,
        mtxdb::room_flag::Flag::SoftFailed,
        mtxdb::room_flag::Flag::Rejected,
        mtxdb::room_flag::Flag::Redacted,
    ];

    for flag in flags {
        let query = mtxdb::room_flag::FlagQuery::has(flag);
        let start = std::time::Instant::now();
        let count = index.count(&db, &query)?;
        let elapsed = start.elapsed();
        println!("{:<14} {:>10} {:>14?}", flag.as_str(), count, elapsed);
    }

    Ok(())
}

#[allow(
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    reason = "coalesced stats generation keeps table and json output together"
)]
fn cmd_stats_coalesced(cli: &Cli, valid_dirs: &[PathBuf], json: bool) -> anyhow::Result<()> {
    struct CoalescedStats {
        shard_type: ShardType,
        dbs_count: usize,
        unique_collections: usize,
        total_index_bytes: usize,
        total_packs: usize,
        total_disk_bytes: u64,
        total_syncs: u64,
        total_writes: u64,
        total_bytes_written: u64,
    }

    let types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };

    let mut all_stats = Vec::new();

    for shard_type in types {
        let mut dbs_with_data = 0_usize;
        let mut collections_set = HashSet::new();
        let mut total_index_bytes = 0_usize;
        let mut total_packs = 0_usize;
        let mut total_disk_bytes = 0_u64;
        let mut total_syncs = 0_u64;
        let mut total_writes = 0_u64;
        let mut total_bytes_written = 0_u64;

        for db_dir in valid_dirs {
            let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                continue;
            };
            let Ok(pool_dir) = pool_dir(&layout, shard_type) else {
                continue;
            };
            let Ok(packs) = glob_pack_files(&pool_dir) else {
                continue;
            };
            if packs.is_empty() {
                continue;
            }
            dbs_with_data = dbs_with_data.saturating_add(1);
            total_packs = total_packs.saturating_add(packs.len());
            for (_, bytes, _) in &packs {
                total_disk_bytes = total_disk_bytes.saturating_add(*bytes);
            }
            let (stats_map, _) = decode_stats_snapshot(&pool_dir);
            for (wc, bw, sc) in stats_map.values() {
                total_writes = total_writes.saturating_add(*wc);
                total_bytes_written = total_bytes_written.saturating_add(*bw);
                total_syncs = total_syncs.saturating_add(*sc);
            }
            for (col_id, _) in PackfileStorage::collection_directory_from_disk(&pool_dir) {
                collections_set.insert(col_id);
            }
            if let Some((idx_total, _)) = index_requirements_from_disk(&pool_dir) {
                total_index_bytes = total_index_bytes.saturating_add(idx_total);
            }
        }

        all_stats.push(CoalescedStats {
            shard_type,
            dbs_count: dbs_with_data,
            unique_collections: collections_set.len(),
            total_index_bytes,
            total_packs,
            total_disk_bytes,
            total_syncs,
            total_writes,
            total_bytes_written,
        });
    }

    if json {
        let mut fields = Vec::new();
        for s in &all_stats {
            let sub_fields = vec![
                ("databases", s.dbs_count.to_string()),
                ("collections", s.unique_collections.to_string()),
                ("index_bytes", s.total_index_bytes.to_string()),
                ("packs", s.total_packs.to_string()),
                ("disk_bytes", s.total_disk_bytes.to_string()),
                ("sync_count", s.total_syncs.to_string()),
                ("write_count", s.total_writes.to_string()),
                ("bytes_written", s.total_bytes_written.to_string()),
            ];
            fields.push((s.shard_type.as_str(), json_object(&sub_fields, "  ")));
        }
        println!("{}", json_object(&fields, ""));
    } else {
        for (index, s) in all_stats.iter().enumerate() {
            if index != 0 {
                println!();
                println!();
            }
            print_section_header(s.shard_type);
            println!("mtxdb stats: coalesced across {} database(s)", s.dbs_count);
            println!(
                "  collections        {} (unique)   index bytes {}",
                s.unique_collections,
                fmt_bytes(u64::try_from(s.total_index_bytes).unwrap_or(u64::MAX))
            );
            println!(
                "  packs              {}            disk bytes  {}",
                s.total_packs,
                fmt_bytes(s.total_disk_bytes)
            );
            println!("  persisted (shard_stats.bin)");
            println!(
                "    writes           {} calls      {}",
                s.total_writes,
                fmt_bytes(s.total_bytes_written)
            );
            println!("    syncs            {}", s.total_syncs);
            println!("  (note: collection IDs deduplicated, pack/index/disk metrics summed across databases)");
        }
    }
    Ok(())
}

fn cmd_stats_single(cli: &Cli, json: bool) -> anyhow::Result<()> {
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
        "    journal          {} calls / {} records   waiters {}   coalesced {}",
        stats.sync_totals.journal_sync_calls,
        stats.sync_totals.journal_records,
        stats.sync_totals.journal_waiters,
        stats.sync_totals.journal_coalesced
    );
    let journal_groups = stats
        .sync_totals
        .journal_sync_calls
        .saturating_sub(stats.sync_totals.journal_coalesced);
    println!(
        "      batching       {} records / {} groups   coalesced {}/{}",
        stats.sync_totals.journal_records,
        journal_groups,
        stats.sync_totals.journal_coalesced,
        stats.sync_totals.journal_sync_calls
    );
    println!(
        "      wal            total {}   lock wait {}   fsync {}",
        fmt_ms(stats.sync_totals.wal),
        fmt_ms(stats.sync_totals.journal_lock_wait),
        fmt_ms(stats.sync_totals.journal_fsync)
    );
    println!(
        "      wal max        lock wait {}   fsync {}",
        fmt_ms(stats.sync_totals.max_journal_lock_wait),
        fmt_ms(stats.sync_totals.max_journal_fsync)
    );
    println!(
        "      pending age    total {}   dirty-lock wait {}",
        fmt_ms(stats.sync_totals.pending_publish_age),
        fmt_ms(stats.sync_totals.dirty_lock_wait)
    );
    println!(
        "      latency        fsync {:?}   lock-wait {:?}   peak in-flight {}",
        stats.sync_diagnostics.fsync_latency.buckets,
        stats.sync_diagnostics.lock_wait_latency.buckets,
        stats.sync_diagnostics.peak_journal_in_flight
    );
    println!(
        "    publish          {} calls   {}",
        stats.publish_calls,
        fmt_ms(stats.publish_time)
    );
    if let Some(timings) = stats.last_sync_timings {
        println!(
            "    last journal     {}   wal {}   lock wait {}   fsync {}   records {}",
            if timings.failed { "FAILED" } else { "ok" },
            fmt_ms(timings.wal),
            fmt_ms(timings.journal_lock_wait),
            fmt_ms(timings.journal_fsync),
            timings.journal_records
        );
    }
    for (index, sample) in stats.sync_diagnostics.worst_syncs.iter().enumerate() {
        println!(
            "    worst[{index:>2}]       {} pid {} total {} wal {} fsync {} lock {} sidecar {} delta {} checkpoint {} in-flight {} records {} path {}",
            if sample.failed { "FAILED" } else { "ok" },
            sample.process_id,
            fmt_ms(sample.total),
            fmt_ms(sample.wal),
            fmt_ms(sample.journal_fsync),
            fmt_ms(sample.journal_lock_wait),
            fmt_ms(sample.sidecar),
            fmt_ms(sample.delta_log),
            fmt_ms(sample.checkpoint),
            sample.journal_in_flight,
            sample.journal_records,
            sample.journal_path.as_deref().unwrap_or("-")
        );
    }
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
        "    read plan        {} prefetched extents / {} bytes / {} skipped",
        stats.read_plan_extents,
        fmt_bytes(stats.read_plan_prefetch_bytes),
        stats.read_plan_skipped_extents
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
    let pack_id_width = summaries
        .iter()
        .map(|summary| summary.pack_id.to_string().len())
        .max()
        .unwrap_or(4)
        .max("pack".len());
    println!(
        "    {:<pack_id_width$}  {:>10}  {:>7}  {:>5}",
        "pack", "bytes", "writes", "syncs"
    );
    for summary in summaries {
        println!(
            "    {:<pack_id_width$}  {:>10}  {:>7}  {:>5}",
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
            ("read_plan_extents", stats.read_plan_extents.to_string()),
            (
                "read_plan_prefetch_bytes",
                stats.read_plan_prefetch_bytes.to_string(),
            ),
            (
                "read_plan_skipped_extents",
                stats.read_plan_skipped_extents.to_string(),
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
    let journal_groups = stats
        .sync_totals
        .journal_sync_calls
        .saturating_sub(stats.sync_totals.journal_coalesced);
    let sync_totals = json_object(
        &[
            ("calls", stats.sync_totals.calls.to_string()),
            ("total_ns", stats.sync_totals.total.as_nanos().to_string()),
            ("wal_ns", stats.sync_totals.wal.as_nanos().to_string()),
            (
                "journal_lock_wait_ns",
                stats.sync_totals.journal_lock_wait.as_nanos().to_string(),
            ),
            (
                "journal_fsync_ns",
                stats.sync_totals.journal_fsync.as_nanos().to_string(),
            ),
            (
                "journal_sync_calls",
                stats.sync_totals.journal_sync_calls.to_string(),
            ),
            (
                "journal_records",
                stats.sync_totals.journal_records.to_string(),
            ),
            (
                "journal_waiters",
                stats.sync_totals.journal_waiters.to_string(),
            ),
            (
                "journal_coalesced",
                stats.sync_totals.journal_coalesced.to_string(),
            ),
            ("journal_groups", journal_groups.to_string()),
            (
                "max_journal_lock_wait_ns",
                stats
                    .sync_totals
                    .max_journal_lock_wait
                    .as_nanos()
                    .to_string(),
            ),
            (
                "max_journal_fsync_ns",
                stats.sync_totals.max_journal_fsync.as_nanos().to_string(),
            ),
            (
                "dirty_lock_wait_ns",
                stats.sync_totals.dirty_lock_wait.as_nanos().to_string(),
            ),
            (
                "pending_publish_age_ns",
                stats.sync_totals.pending_publish_age.as_nanos().to_string(),
            ),
        ],
        "  ",
    );
    let last_sync = stats.last_sync_timings.map_or_else(
        || "null".to_owned(),
        |timings| {
            json_object(
                &[
                    ("total_ns", timings.total.as_nanos().to_string()),
                    ("failed", timings.failed.to_string()),
                    ("wal_ns", timings.wal.as_nanos().to_string()),
                    (
                        "journal_lock_wait_ns",
                        timings.journal_lock_wait.as_nanos().to_string(),
                    ),
                    (
                        "journal_fsync_ns",
                        timings.journal_fsync.as_nanos().to_string(),
                    ),
                    ("journal_sync_calls", timings.journal_sync_calls.to_string()),
                    ("journal_records", timings.journal_records.to_string()),
                    ("journal_in_flight", timings.journal_in_flight.to_string()),
                    ("journal_waiters", timings.journal_waiters.to_string()),
                    ("journal_coalesced", timings.journal_coalesced.to_string()),
                ],
                "  ",
            )
        },
    );
    let histogram_json = |buckets: [u64; 5]| {
        format!(
            "[{}, {}, {}, {}, {}]",
            buckets[0], buckets[1], buckets[2], buckets[3], buckets[4]
        )
    };
    let worst_syncs = format!(
        "[{}]",
        stats
            .sync_diagnostics
            .worst_syncs
            .iter()
            .map(|sample| {
                json_object(
                    &[
                        ("timestamp_ms", sample.timestamp_ms.to_string()),
                        ("failed", sample.failed.to_string()),
                        ("process_id", sample.process_id.to_string()),
                        (
                            "journal_path",
                            sample
                                .journal_path
                                .as_ref()
                                .map_or_else(|| "null".to_owned(), |path| json_string(path)),
                        ),
                        ("total_ns", sample.total.as_nanos().to_string()),
                        ("pack_flush_ns", sample.pack_flush.as_nanos().to_string()),
                        ("pack_fsync_ns", sample.pack_fsync.as_nanos().to_string()),
                        ("sidecar_ns", sample.sidecar.as_nanos().to_string()),
                        ("delta_log_ns", sample.delta_log.as_nanos().to_string()),
                        ("checkpoint_ns", sample.checkpoint.as_nanos().to_string()),
                        (
                            "dirty_lock_wait_ns",
                            sample.dirty_lock_wait.as_nanos().to_string(),
                        ),
                        (
                            "pending_publish_age_ns",
                            sample.pending_publish_age.as_nanos().to_string(),
                        ),
                        ("wal_ns", sample.wal.as_nanos().to_string()),
                        (
                            "journal_lock_wait_ns",
                            sample.journal_lock_wait.as_nanos().to_string(),
                        ),
                        (
                            "journal_fsync_ns",
                            sample.journal_fsync.as_nanos().to_string(),
                        ),
                        ("journal_records", sample.journal_records.to_string()),
                        ("journal_in_flight", sample.journal_in_flight.to_string()),
                        ("journal_waiters", sample.journal_waiters.to_string()),
                        ("journal_coalesced", sample.journal_coalesced.to_string()),
                    ],
                    "    ",
                )
            })
            .collect::<Vec<_>>()
            .join(","),
    );
    let sync_diagnostics = json_object(
        &[
            (
                "fsync_latency_buckets",
                histogram_json(stats.sync_diagnostics.fsync_latency.buckets),
            ),
            (
                "lock_wait_latency_buckets",
                histogram_json(stats.sync_diagnostics.lock_wait_latency.buckets),
            ),
            (
                "peak_journal_in_flight",
                stats.sync_diagnostics.peak_journal_in_flight.to_string(),
            ),
            ("worst_syncs", worst_syncs),
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
            json_string(&summary.pack_id.to_string()),
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
        ("last_sync", last_sync),
        ("sync_totals", sync_totals),
        ("sync_diagnostics", sync_diagnostics),
        ("publish_calls", stats.publish_calls.to_string()),
        ("publish_time_ns", stats.publish_time.as_nanos().to_string()),
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

/// A quoted, escaped JSON string literal for `text` (callers pass unescaped text).
fn json_string_raw(text: &str) -> String {
    // Reuse rezzy-json's string encoder rather than maintaining a second one.
    // (`write_json_string` replaces the allocation here once the pinned rezzy
    // revision is bumped; `write_string_value` exists on the current rev.)
    rezzy::json::write_string_value(&rezzy::json::Value::String(text.to_owned()))
        .expect("writing a JSON string to a String cannot fail")
}

/// A JSON string value including surrounding quotes.
pub(crate) fn json_string(text: &str) -> String {
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
            let value = |entry: &(PackId, u64, u8)| match column {
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
fn index_requirements_from_disk(dir: &Path) -> Option<(usize, HashMap<PackId, usize>)> {
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

/// Discover canonical `pack_<hex>.pack` files.
///
/// The filename carries only a truncated (optionally full) address prefix; the
/// authoritative 128-bit address is read from the header and checked against
/// the filename prefix. Returns `(address, bytes, format version)` ordered by
/// address.
fn glob_pack_files(dir: &Path) -> anyhow::Result<Vec<(PackId, u64, u8)>> {
    let mut packs = Vec::new();
    let mut seen = HashSet::new();
    // A pool is created by its first write, so a pool directory that does not
    // exist holds no packs.
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(packs),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "pack") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.starts_with("shard_") {
            bail!(
                "found unsupported shard_ pack file {}; reset it rather than opening it",
                path.display()
            );
        }
        let Some((prefix_bytes, prefix_digits)) = PackId::parse_filename_prefix(stem) else {
            if stem.starts_with("pack_") {
                bail!(
                    "invalid pack filename {}; expected pack_<lowercase hex digits>.pack",
                    path.display()
                );
            }
            // Another application's `.pack` file: ignore it.
            continue;
        };
        let name_hex = mtxdb::packfile::hex_lower(&prefix_bytes[..prefix_digits.div_ceil(2)]);
        let file = fs::File::open(&path)
            .with_context(|| format!("failed to open pack `{}`", path.display()))?;
        let header = mtxdb::packfile::read_header(&mut BufReader::new(file))
            .with_context(|| format!("unsupported or corrupt pack `{}`", path.display()))?
            .with_context(|| format!("invalid pack header `{}`", path.display()))?;
        if !header.pack_id.as_hex().starts_with(&name_hex) {
            bail!(
                "pack {} identifies itself as {} but its filename prefix disagrees",
                path.display(),
                header.pack_id
            );
        }
        if !seen.insert(header.pack_id) {
            bail!(
                "duplicate pack address {} in {}",
                header.pack_id,
                dir.display()
            );
        }
        packs.push((
            header.pack_id,
            entry.metadata()?.len(),
            mtxdb::packfile::VERSION,
        ));
    }
    packs.sort_unstable_by_key(|(pack_id, _, _)| *pack_id);
    Ok(packs)
}

/// Pack address to `(write_count, bytes_written, sync_count)`, as decoded from
/// `shard_stats.bin`.
type ShardStatsMap = std::collections::HashMap<PackId, (u64, u64, u64)>;

/// Decode `shard_stats.bin` — same binary format as
/// `ShardPool::restore_persisted_stats`, but standalone. Returns a map
/// from a pack address to `(write_count, bytes_written, sync_count)` and the
/// snapshot's persisted-at timestamp.
fn decode_stats_snapshot(dir: &Path) -> (ShardStatsMap, Option<u64>) {
    const STATS_MAGIC: &[u8; 4] = b"MSTA";
    const STATS_VERSION: u8 = 1;
    const STATS_HEADER_LEN: usize = 4 + 1 + 8;
    const STATS_RECORD_LEN: usize = mtxdb::packfile::PACK_ID_LEN + 8 * 3;

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
            let pack_id = PackId(rec[0..mtxdb::packfile::PACK_ID_LEN].try_into().unwrap());
            let cursor = mtxdb::packfile::PACK_ID_LEN;
            let write_count = u64::from_le_bytes(rec[cursor..cursor + 8].try_into().unwrap());
            let bytes_written =
                u64::from_le_bytes(rec[cursor + 8..cursor + 16].try_into().unwrap());
            let sync_count = u64::from_le_bytes(rec[cursor + 16..cursor + 24].try_into().unwrap());
            stats_map.insert(pack_id, (write_count, bytes_written, sync_count));
        }
    }
    (stats_map, persisted_at)
}

/// Print the shard table header and rows.
fn print_shard_table(
    shard_entries: &[(PackId, u64, u8)],
    stats_map: &ShardStatsMap,
    node_counts: Option<&std::collections::HashMap<PackId, u64>>,
    collection_counts: Option<&std::collections::HashMap<PackId, u64>>,
    total_collections: Option<usize>,
    index_requirement: Option<usize>,
    index_requirements_by_shard: Option<&HashMap<PackId, usize>>,
) {
    // `ShardPool::open_internal` restores the last-slotted pack as its append
    // destination. Mirror that recovery rule here without opening a writer.
    // Shards are discovered in address order, so the last entry is the newest
    // slot (addresses are random, so this is discovery order, not age).
    let active_pack_id = shard_entries.last().map(|(pack_id, _, _)| *pack_id);

    println!(
        "{:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
        "pack", "ver", "bytes", "nodes", "collections", "index", "syncs",
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
            "{:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
            format!(
                "0x{}{}",
                &pack_id.as_hex()[..16],
                if active_pack_id == Some(pack_id) {
                    " *"
                } else {
                    "  "
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
        "{:>21}  {:>3}  {:>10}  {:>8}  {:>11}  {:>12}  {:>6}",
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

/// `info` accepts a collection selector (a canonical sigil, a bare `0x`-less
/// sigil, or a `0x`-prefixed 32-hex logical ID) or a pack selector
/// (`0x`-prefixed, 1–16 hex digits, as printed by `mtxdb shards`).
///
/// A `0x`-prefixed 32-hex selector is genuinely ambiguous now that a pack's
/// full address is 32 hex digits: it can name either a collection or a pack,
/// and the two are indistinguishable by length. `classify_info_selector`
/// therefore resolves that length against the selected store — a live pack
/// with that exact address, and/or a live collection with that id — and
/// errors when both match so the caller must disambiguate explicitly
/// (`--pack` or `--collection`). Bare hex is not accepted.
/// What an `info` selector names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InfoTarget {
    Pack,
    Collection,
}

/// One live pack discovered in a selected pool, with enough provenance to name
/// it in an ambiguity diagnostic.
struct PackLocation {
    database: PathBuf,
    shard_type: ShardType,
    pack_id: PackId,
    file_bytes: u64,
    version: u8,
}

impl std::fmt::Display for PackLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}]",
            self.pack_id,
            pool_label(&self.database, self.shard_type)
        )
    }
}

/// A short human label for a pool: the database dir name plus shard type.
fn pool_label(database: &Path, shard_type: ShardType) -> String {
    let db = database
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| database.to_str().unwrap_or("?"));
    format!("{db}:{shard_type:?}")
}

/// Every live pack across the selected pools, with its database and shard type.
///
/// Cost: opens every selected database and globs every selected pool's pack
/// directory. This is fine once per `info` invocation, but must not be called
/// per-selector inside a multi-selector command — hoist the pool walk and
/// reuse it instead. Discovery failures are skipped; an invalid database
/// selection yields an empty inventory.
fn pack_locations_in_selected_pools(cli: &Cli) -> Vec<PackLocation> {
    let Ok(dirs) = valid_database_dirs(cli) else {
        return Vec::new();
    };
    let shard_types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };
    let mut locations = Vec::new();
    for db_dir in dirs {
        let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
            continue;
        };
        for &shard_type in &shard_types {
            let Ok(dir) = pool_dir(&layout, shard_type) else {
                continue;
            };
            if let Ok(files) = glob_pack_files(&dir) {
                locations.extend(files.into_iter().map(|(pack_id, file_bytes, version)| {
                    PackLocation {
                        database: db_dir.clone(),
                        shard_type,
                        pack_id,
                        file_bytes,
                        version,
                    }
                }));
            }
        }
    }
    locations
}

/// Whether a live collection with this id exists in any selected pool.
///
/// Same cost caveat as [`pack_locations_in_selected_pools`]: a single `info`
/// disambiguation is fine, a per-selector loop in a multi-selector command is
/// not. Unreadable databases and pools are skipped; returns `false` if no
/// readable pool reports a match.
fn collection_id_exists(cli: &Cli, target_id: &[u8; 16]) -> bool {
    let Ok(dirs) = valid_database_dirs(cli) else {
        return false;
    };
    let shard_types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };
    for db_dir in dirs {
        let Ok(layout) = DatabaseLayout::open_read_only(db_dir) else {
            continue;
        };
        for &shard_type in &shard_types {
            let Ok(dir) = pool_dir(&layout, shard_type) else {
                continue;
            };
            let has = if let Some(summaries) = PackfileStorage::collection_summaries_from_disk(&dir)
            {
                summaries.iter().any(|(id, _, _, _)| id == target_id)
            } else if let Ok(store) = open_pool_read_only(dir) {
                store.collection_index_info(target_id).is_some()
            } else {
                false
            };
            if has {
                return true;
            }
        }
    }
    false
}

/// Classify an `info`/`scan` selector by inference. See
/// [`classify_info_selector_explicit`]; this is the no-override form.
fn classify_info_selector(cli: &Cli, selector: &str) -> anyhow::Result<InfoTarget> {
    classify_info_selector_explicit(cli, selector, None)
}

/// Classify a selector, optionally under an explicit namespace.
///
/// With `explicit` set (from `info --pack` / `info --collection`), the requested
/// type is used directly — inference is skipped — and the selector is validated
/// against that type: `--pack` accepts the full 0x-prefixed 32-hex address or a
/// 1-16 hex filename prefix that resolves to exactly one pack (ambiguity errors
/// and asks for the full address); `--collection` accepts an exact 32-hex id or
/// a canonical `!room` sigil. Either errors if nothing of that type matches.
/// `--pack` also rejects a single address present in multiple selected pools.
///
/// Without it, classification is inferred by the digits after the canonical
/// lowercase `0x` prefix: 1-16 name a pack prefix, exactly 32 can name either a
/// full pack address or a collection, and any other count is an error. A
/// selector without a `0x` prefix is classified as a collection without
/// validation; an uppercase `0X` prefix is rejected. Short pack prefixes are
/// also classified without validation; callers must parse them before use.
/// For an ambiguous 32-hex selector the database is checked:
/// - If only an existing pack matches, it resolves as [`InfoTarget::Pack`].
/// - If only a collection exists (or neither exists yet), it resolves as [`InfoTarget::Collection`].
/// - If both a live pack and a collection match the same ID, an error requires
///   an explicit `--pack` or `--collection` (with the exact 32-hex id).
///
/// Returns the target plus the reusable pack inventory when pack
/// classification required a discovery walk.
fn classify_info_selector_with_inventory(
    cli: &Cli,
    selector: &str,
    explicit: Option<InfoTarget>,
) -> anyhow::Result<(InfoTarget, Option<Vec<PackLocation>>)> {
    match explicit {
        Some(InfoTarget::Pack) => {
            // A full 32-hex address, or a 1-16 hex filename prefix (64 bits is
            // the ergonomic default). A prefix must resolve to exactly one
            // pack in exactly one pool; anything else is reported with the
            // specific conflict. Never a collection.
            let pack_selector = parse_pack_id_selector(selector)?;
            let locations = pack_locations_in_selected_pools(cli);
            let matched: Vec<&PackLocation> = locations
                .iter()
                .filter(|location| pack_selector.matches(&location.pack_id))
                .collect();
            let distinct: std::collections::BTreeSet<PackId> =
                matched.iter().map(|location| location.pack_id).collect();
            match (distinct.len(), matched.len()) {
                (0, _) => bail!("pack {selector}: not found"),
                // Hand the walk to the caller so the display path need not
                // repeat it.
                (1, 1) => return Ok((InfoTarget::Pack, Some(locations))),
                (1, _) => {
                    // One pack, present in more than one selected pool.
                    let where_ = matched
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    bail!(
                        "pack {selector} exists in {} selected pools ({where_}); narrow with \
                         --dir or -t to a single pool",
                        matched.len()
                    )
                }
                (n, m) => {
                    // More than one distinct pack matches the prefix.
                    let ids = distinct
                        .iter()
                        .map(PackId::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    let pool_hint = if m > n {
                        " (some also appear in more than one pool, so narrow with --dir or -t)"
                    } else {
                        ""
                    };
                    bail!(
                        "pack prefix `{selector}` matches {n} different packs ({ids}); use the \
                         full 0x-prefixed {PACK_ID_HEX_LEN}-hex id{pool_hint}"
                    )
                }
            }
        }
        Some(InfoTarget::Collection) => {
            let id = parse_collection_selector(selector)?;
            if !collection_id_exists(cli, &id) {
                bail!("collection {selector}: not found");
            }
            return Ok((InfoTarget::Collection, None));
        }
        None => {}
    }
    let Some(hex) = selector.strip_prefix("0x") else {
        if selector.starts_with("0X") {
            bail!("invalid ID `{selector}`: the prefix must be lowercase `0x`");
        }
        return Ok((InfoTarget::Collection, None));
    };
    match hex.len() {
        // A bare prefix does not need a walk to classify; the display path
        // resolves it against the pool it was narrowed to.
        1..=16 => Ok((InfoTarget::Pack, None)),
        32 => {
            let pack_id = parse_pack_id_selector(selector)
                .ok()
                .and_then(|sel| match sel {
                    PackSelector::Exact(id) => Some(id),
                    PackSelector::Prefix(_) => None,
                });
            let collection_id = parse_collection_id(selector).ok();

            let locations = pack_locations_in_selected_pools(cli);
            let pack_matches =
                pack_id.is_some_and(|id| locations.iter().any(|location| location.pack_id == id));
            let collection_matches = collection_id.is_some_and(|id| collection_id_exists(cli, &id));

            if pack_matches && collection_matches {
                bail!(
                    "selector `{selector}` matches both a pack and a collection; specify one:\n  \
                     mtxdb info --pack {selector}\n  \
                     mtxdb info --collection {selector}"
                );
            }
            if pack_matches {
                Ok((InfoTarget::Pack, Some(locations)))
            } else {
                Ok((InfoTarget::Collection, None))
            }
        }
        found => {
            bail!(
                "invalid ID `{selector}`: after `0x` expected 32 hex digits (a collection or full pack address) \
                 or 1-16 (a pack filename prefix), found {found} characters{}",
                if hex.starts_with("0x") {
                    " (the `0x` prefix is doubled)"
                } else {
                    ""
                }
            )
        }
    }
}

/// Classify a selector, returning only the target. Thin wrapper over
/// [`classify_info_selector_with_inventory`] for callers that do not reuse the
/// pack inventory the classification may have walked.
fn classify_info_selector_explicit(
    cli: &Cli,
    selector: &str,
    explicit: Option<InfoTarget>,
) -> anyhow::Result<InfoTarget> {
    Ok(classify_info_selector_with_inventory(cli, selector, explicit)?.0)
}

#[allow(clippy::too_many_lines)]
fn cmd_info_coalesced(
    cli: &Cli,
    selector: &str,
    deep: bool,
    explicit: Option<InfoTarget>,
) -> anyhow::Result<()> {
    let (target, inventory) = classify_info_selector_with_inventory(cli, selector, explicit)?;
    let valid_dirs = valid_database_dirs(cli)?;
    let shard_types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };

    match target {
        InfoTarget::Collection => {
            let collection_id = parse_collection_selector(selector)?;
            let mut matches = Vec::new();
            for db_dir in &valid_dirs {
                let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                    continue;
                };
                for &shard_type in &shard_types {
                    let Ok(dir) = pool_dir(&layout, shard_type) else {
                        continue;
                    };
                    let has_collection = if let Some(summaries) =
                        PackfileStorage::collection_summaries_from_disk(&dir)
                    {
                        summaries.iter().any(|(id, _, _, _)| id == &collection_id)
                    } else if let Ok(store) = open_pool_read_only(dir.clone()) {
                        store.collection_index_info(&collection_id).is_some()
                    } else {
                        false
                    };
                    if has_collection {
                        matches.push((db_dir.clone(), shard_type, dir));
                    }
                }
            }

            if matches.is_empty() {
                eprintln!(
                    "collection {}: not found across {} database(s)",
                    format_id(&collection_id),
                    valid_dirs.len()
                );
                return Ok(());
            }

            if matches.len() == 1 {
                let (db_dir, shard_type, dir) = &matches[0];
                println!("database: {}", db_dir.display());
                println!("pool:     {}", shard_type.as_str());
                inspect_collection_in_dir(dir, &collection_id, deep);
            } else {
                for (i, (db_dir, shard_type, dir)) in matches.iter().enumerate() {
                    if i > 0 {
                        println!();
                    }
                    println!(
                        "=== Database: {} (pool: {}) ===",
                        db_dir.display(),
                        shard_type.as_str()
                    );
                    inspect_collection_in_dir(dir, &collection_id, deep);
                }
                println!();
                println!(
                    "note: collection found in {} of {} database(s)",
                    matches.len(),
                    valid_dirs.len()
                );
            }
            Ok(())
        }
        InfoTarget::Pack => {
            let selector = parse_pack_id_selector(selector)?;
            let mut matches = Vec::new();
            for db_dir in &valid_dirs {
                let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                    continue;
                };
                for &shard_type in &shard_types {
                    let Ok(dir) = pool_dir(&layout, shard_type) else {
                        continue;
                    };
                    let files = if let Some(locations) = inventory.as_deref() {
                        locations
                            .iter()
                            .filter(|location| {
                                location.database == *db_dir && location.shard_type == shard_type
                            })
                            .map(|location| {
                                (location.pack_id, location.file_bytes, location.version)
                            })
                            .collect()
                    } else {
                        let Ok(files) = glob_pack_files(&dir) else {
                            continue;
                        };
                        files
                    };
                    let ids: Vec<PackId> = files.iter().map(|&(id, _, _)| id).collect();
                    if !ids.iter().any(|id| selector.matches(id)) {
                        continue;
                    }
                    let pack_id = selector.resolve(ids.iter())?;
                    let shard_entries: Vec<(PackId, u64, u8)> = files
                        .into_iter()
                        .filter(|&(id, _, _)| id == pack_id)
                        .collect();
                    matches.push((db_dir.clone(), shard_type, dir, pack_id, shard_entries));
                }
            }

            if matches.is_empty() {
                eprintln!(
                    "pack {selector}: not found across {} database(s)",
                    valid_dirs.len()
                );
                return Ok(());
            }

            for (i, (db_dir, shard_type, dir, pack_id, shard_entries)) in matches.iter().enumerate()
            {
                if i > 0 {
                    println!();
                }
                println!(
                    "=== Database: {} (pool: {}) ===",
                    db_dir.display(),
                    shard_type.as_str()
                );
                let path = find_pack_path(dir, *pack_id)?;
                print_pack_info(dir, &path, *pack_id, shard_entries, *shard_type);
            }
            if matches.len() > 1 {
                println!();
                println!(
                    "note: pack found in {} of {} database(s)",
                    matches.len(),
                    valid_dirs.len()
                );
            }
            Ok(())
        }
    }
}

/// Show the small, human-facing database and pool metadata summary used when
/// `info` has no selector. This deliberately reads headers and sidecar sizes
/// without opening a writer or scanning pack payloads.
fn cmd_info_default(cli: &Cli, stats: bool) -> anyhow::Result<()> {
    if cli.coalesce {
        let valid_dirs = valid_database_dirs(cli)?;
        if valid_dirs.len() > 1 {
            for (index, dir) in valid_dirs.iter().enumerate() {
                if index != 0 {
                    println!();
                    println!();
                }
                println!("=== Database: {} ===", dir.display());
                cmd_info_default(&cli.with_dir(dir.clone()), stats)?;
            }
            return Ok(());
        }
    }
    run_multi_dir(cli, |sub_cli| cmd_info_default_single(sub_cli, stats))
}

fn metadata_file_summary(path: &Path) -> String {
    let Ok(bytes) = fs::read(path) else {
        return "missing".to_owned();
    };
    let magic = bytes.get(..4).map_or_else(
        || "<short>".to_owned(),
        |raw| String::from_utf8_lossy(raw).into_owned(),
    );
    let version = bytes
        .get(4)
        .map_or_else(|| "?".to_owned(), std::string::ToString::to_string);
    format!(
        "valid-looking magic={magic:?} version={version} ({} bytes)",
        bytes.len()
    )
}

/// Inspect on-disk artifacts without opening a writer or acquiring any of the
/// database locks. This intentionally works when the normal database opener
/// cannot validate one of the files: `meta` is a recovery aid, not another
/// database-open path.
fn cmd_meta(
    cli: &Cli,
    target: &str,
    json: bool,
    limit: i64,
    offset: i64,
    decode: Option<&str>,
) -> anyhow::Result<()> {
    if cli.dirs.len() > 1 {
        bail!("`mtxdb meta` accepts only one --dir target");
    }
    let root = cli.single_dir();
    let limit = if limit <= 0 {
        usize::MAX
    } else {
        usize::try_from(limit).unwrap_or(usize::MAX)
    };
    let offset = usize::try_from(offset.max(0)).unwrap_or(usize::MAX);
    let mut report = MetaReport::default();

    match target {
        "overview" => {
            meta_db(root, &mut report);
            meta_sidecars(root, &mut report);
            meta_locks(root, &mut report);
            meta_pools(root, &mut report, limit, offset, false);
            meta_wal(root, &mut report, limit, offset, decode);
            meta_checkpoints(root, &mut report);
            meta_deltas(root, &mut report, limit, offset);
            meta_health(root, &mut report);
        }
        "db" => meta_db(root, &mut report),
        "wal" => meta_wal(root, &mut report, limit, offset, decode),
        "checkpoint" => meta_checkpoints(root, &mut report),
        "delta" => meta_deltas(root, &mut report, limit, offset),
        "sidecars" => meta_sidecars(root, &mut report),
        "packs" => meta_pools(root, &mut report, limit, offset, true),
        "locks" => meta_locks(root, &mut report),
        "raw" => meta_raw(root, &mut report, limit, offset),
        _ => unreachable!("clap validates meta target"),
    }

    report.print(json);
    Ok(())
}

#[derive(Default)]
struct MetaReport {
    lines: Vec<String>,
    note_count: usize,
    warn_count: usize,
    error_count: usize,
    unknown_count: usize,
    json_records: Vec<OwnedValue>,
}

impl MetaReport {
    fn line(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.lines.push(line.clone());
        let mut fields = Object::new();
        fields.insert("kind".to_owned(), OwnedValue::from("text"));
        fields.insert("message".to_owned(), OwnedValue::from(line));
        self.json_records.push(OwnedValue::Object(fields));
    }

    fn finding(&mut self, level: &str, path: &Path, message: impl std::fmt::Display) {
        match level {
            "NOTE" => self.note_count = self.note_count.saturating_add(1),
            "WARN" => self.warn_count = self.warn_count.saturating_add(1),
            "ERROR" => self.error_count = self.error_count.saturating_add(1),
            _ => self.unknown_count = self.unknown_count.saturating_add(1),
        }
        let severity = if matches!(level, "NOTE" | "WARN" | "ERROR") {
            level
        } else {
            "UNKNOWN"
        };
        let message = message.to_string();
        let path = path.display().to_string();
        self.lines.push(format!("[{severity}] {path}: {message}"));
        let mut fields = Object::new();
        fields.insert("kind".to_owned(), OwnedValue::from("diagnostic"));
        fields.insert("severity".to_owned(), OwnedValue::from(severity));
        fields.insert("path".to_owned(), OwnedValue::from(path));
        fields.insert("message".to_owned(), OwnedValue::from(message));
        self.json_records.push(OwnedValue::Object(fields));
    }

    fn print(self, json: bool) {
        if json {
            let value = self.json_value();
            println!("{}", value.encode());
        } else {
            for line in self.lines {
                println!("{line}");
            }
        }
    }

    fn json_value(self) -> OwnedValue {
        let mut records = self.json_records;
        let mut summary = Object::new();
        summary.insert("kind".to_owned(), OwnedValue::from("summary"));
        summary.insert(
            "note".to_owned(),
            OwnedValue::from(u64::try_from(self.note_count).unwrap_or(u64::MAX)),
        );
        summary.insert(
            "warn".to_owned(),
            OwnedValue::from(u64::try_from(self.warn_count).unwrap_or(u64::MAX)),
        );
        summary.insert(
            "error".to_owned(),
            OwnedValue::from(u64::try_from(self.error_count).unwrap_or(u64::MAX)),
        );
        summary.insert(
            "unknown".to_owned(),
            OwnedValue::from(u64::try_from(self.unknown_count).unwrap_or(u64::MAX)),
        );
        records.insert(0, OwnedValue::Object(summary));
        OwnedValue::Array(records)
    }
}

fn meta_pool_dirs(root: &Path) -> impl Iterator<Item = (ShardType, PathBuf)> + '_ {
    ShardType::ALL
        .into_iter()
        .map(|pool| (pool, root.join(pool.as_str())))
}

fn meta_sidecars(root: &Path, report: &mut MetaReport) {
    report.line(format!("database: {}", root.display()));
    meta_file(root, "db.meta", report);
    for (pool, dir) in meta_pool_dirs(root) {
        report.line(format!("pool: {} ({})", pool.as_str(), dir.display()));
        for name in [
            "pool.meta",
            "store.meta",
            "journal.lsn",
            "shard_stats.bin",
            "shard_collections.bin",
        ] {
            meta_file(&dir, name, report);
        }
    }
}

fn meta_db(root: &Path, report: &mut MetaReport) {
    let path = root.join(mtxdb::layout::DB_META_FILENAME);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.finding("WARN", &path, "missing database descriptor");
            return;
        }
        Err(error) => {
            report.finding("ERROR", &path, error);
            return;
        }
    };
    let pools = bytes.get(13..).map_or_else(
        || "<truncated>".to_owned(),
        |bytes| String::from_utf8_lossy(bytes).replace('\n', ", "),
    );
    report.line(format!(
        "db.meta: {} bytes ({}, pools={pools})",
        bytes.len(),
        metadata_magic(&bytes),
    ));
}

fn meta_file(dir: &Path, name: &str, report: &mut MetaReport) {
    let path = dir.join(name);
    match fs::metadata(&path) {
        Ok(metadata) => {
            if name == "journal.lsn" {
                match fs::read(&path) {
                    Ok(bytes) if bytes.len() >= 8 => report.line(format!(
                        "  {name}: {} (covered_lsn={})",
                        metadata.len(),
                        u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]))
                    )),
                    Ok(_) => report.finding("WARN", &path, "truncated journal.lsn"),
                    Err(error) => report.finding("ERROR", &path, error),
                }
            } else if name == "shard_collections.bin" {
                match mtxdb::packfile::storage::read_persisted_shard_collections(dir) {
                    Some(directory) => report.line(format!(
                        "  {name}: {} bytes (fingerprint={:#x}, records={}, persisted_at={})",
                        metadata.len(),
                        directory.fingerprint,
                        directory.records.len(),
                        directory.persisted_at
                    )),
                    None => report.finding("WARN", &path, "invalid or unsupported sidecar"),
                }
            } else {
                let mut header = [0_u8; 5];
                match fs::File::open(&path).and_then(|mut file| {
                    use std::io::Read as _;
                    file.read_exact(&mut header)
                }) {
                    Ok(()) => report.line(format!(
                        "  {name}: {} bytes ({})",
                        metadata.len(),
                        metadata_magic(&header)
                    )),
                    Err(error) => report.finding("WARN", &path, error),
                }
                if name == "store.meta" {
                    if let Some(version) = mtxdb::shard::store_created_by_version(dir) {
                        report.line(format!("    created_by: {version}"));
                    }
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.line(format!("  {name}: missing"));
        }
        Err(error) => report.finding("ERROR", &path, error),
    }
}

fn metadata_magic(bytes: &[u8]) -> String {
    let magic = bytes.get(..4).map_or_else(
        || "<short>".to_owned(),
        |raw| String::from_utf8_lossy(raw).into_owned(),
    );
    let version = bytes
        .get(4)
        .map_or_else(|| "?".to_owned(), std::string::ToString::to_string);
    format!("magic={magic:?} version={version}")
}

fn meta_locks(root: &Path, report: &mut MetaReport) {
    let wal_lock = root.join(mtxdb::layout::WAL_LOCK_FILENAME);
    if wal_lock.exists() {
        meta_lock_line(&wal_lock, report);
    }
    for (_, dir) in meta_pool_dirs(root) {
        let path = dir.join(".mtxdb.lock");
        if path.exists() {
            meta_lock_line(&path, report);
        }
    }
}

fn meta_lock_line(path: &Path, report: &mut MetaReport) {
    match mtxdb::ShardPool::lock_holder_info(path) {
        Some(info) => {
            let holder = if info.has_starttime {
                if info.running {
                    "running"
                } else {
                    "not-running"
                }
            } else if info.running {
                "exists"
            } else {
                "not-found"
            };
            report.line(format!(
                "lock {}: pid={} {} confidence={}",
                path.display(),
                info.pid,
                holder,
                if info.has_starttime {
                    "full"
                } else {
                    "limited"
                }
            ));
        }
        None => report.finding("WARN", path, "present but PID is unreadable"),
    }
}

fn meta_pools(root: &Path, report: &mut MetaReport, limit: usize, offset: usize, enumerate: bool) {
    for (pool, dir) in meta_pool_dirs(root) {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                report.finding("ERROR", &dir, error);
                continue;
            }
        };
        let mut packs = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "pack"))
            .collect::<Vec<_>>();
        packs.sort();
        report.line(format!(
            "pool {}: {} pack file(s)",
            pool.as_str(),
            packs.len()
        ));
        if !enumerate {
            continue;
        }
        for path in packs.into_iter().skip(offset).take(limit) {
            match pack_identity(&path) {
                PackIdentity::Valid { pack_id, length } => report.line(format!(
                    "  {}: {} bytes, pack_id={pack_id}",
                    path.display(),
                    length
                )),
                PackIdentity::NonCanonical(message) => {
                    report.finding(PackIssueLevel::Note.as_str(), &path, message);
                }
                PackIdentity::Invalid(error) => {
                    report.finding(PackIssueLevel::Warn.as_str(), &path, error);
                }
            }
        }
    }
}

enum PackIdentity {
    Valid { pack_id: PackId, length: u64 },
    NonCanonical(String),
    Invalid(anyhow::Error),
}

#[derive(Clone, Copy)]
enum PackIssueLevel {
    Note,
    Warn,
}

impl PackIssueLevel {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Note => "NOTE",
            Self::Warn => "WARN",
        }
    }
}

/// Read a canonical filename's full pack ID and file length, checking prefix
/// agreement with the header. Noncanonical names are reported separately;
/// I/O and header-validation failures become `PackIdentity::Invalid`.
fn pack_identity(path: &Path) -> PackIdentity {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(str::to_owned);
    let Some(stem) = stem else {
        return PackIdentity::Invalid(anyhow!("pack filename is not valid UTF-8"));
    };
    let Some((prefix_bytes, prefix_digits)) = PackId::parse_filename_prefix(&stem) else {
        return PackIdentity::NonCanonical(format!(
            "pack filename `{stem}` is non-canonical; expected pack_<lowercase hex digits>.pack"
        ));
    };
    let name_hex = mtxdb::packfile::hex_lower(&prefix_bytes[..prefix_digits.div_ceil(2)]);
    let result = (|| -> anyhow::Result<(PackId, u64)> {
        let length = fs::metadata(path)?.len();
        let mut file = BufReader::new(fs::File::open(path)?);
        let header =
            mtxdb::packfile::read_header(&mut file)?.ok_or_else(|| anyhow!("not a packfile"))?;
        if !header.pack_id.as_hex().starts_with(&name_hex) {
            bail!(
                "pack filename prefix {name_hex} disagrees with header id {}",
                header.pack_id
            );
        }
        Ok((header.pack_id, length))
    })();
    match result {
        Ok((pack_id, length)) => PackIdentity::Valid { pack_id, length },
        Err(error) => PackIdentity::Invalid(error),
    }
}

fn meta_wal(
    root: &Path,
    report: &mut MetaReport,
    limit: usize,
    offset: usize,
    decode: Option<&str>,
) {
    let mut paths = vec![root.join("wal.bin")];
    paths.extend(meta_pool_dirs(root).map(|(_, dir)| dir.join("wal.bin")));
    paths.sort();
    paths.dedup();
    if root.join("wal.bin").is_file() {
        for (_, dir) in meta_pool_dirs(root) {
            let pool_wal = dir.join("wal.bin");
            if pool_wal.is_file() {
                report.finding(
                    "WARN",
                    &pool_wal,
                    "per-pool WAL is present but ignored because root wal.bin selects shared WAL",
                );
            }
        }
    }
    for path in paths.into_iter().filter(|path| path.is_file()) {
        match mtxdb::journal::Journal::scan_read_only(&path) {
            Ok(scan) => {
                report.line(format!(
                    "wal {}: {} group(s), valid_len={}, base_lsn={}, truncated_tail={}",
                    path.display(),
                    scan.groups.len(),
                    scan.valid_len,
                    scan.base_lsn,
                    scan.truncated_tail
                ));
                if scan.truncated_tail {
                    report.finding(
                        "WARN",
                        &path,
                        "truncated tail; complete groups were retained",
                    );
                }
                for group in scan
                    .groups
                    .iter()
                    .flat_map(|group| group.entries.iter())
                    .skip(offset)
                    .take(limit)
                {
                    report.line(format!(
                        "  lsn={} offset={} frame_len={} pool={:?} {}",
                        group.lsn,
                        group.offset,
                        group.frame_len,
                        group.pool,
                        format_mutation(&group.mutation, decode)
                    ));
                }
            }
            Err(error) => report.finding("WARN", &path, error),
        }
    }
}

fn format_mutation(mutation: &mtxdb::journal::Mutation, decode: Option<&str>) -> String {
    match mutation {
        mtxdb::journal::Mutation::Put {
            collection_id,
            node_id,
            payload,
        } => {
            let mut line = format!(
                "kind=put collection=0x{} node=0x{} payload_len={}",
                hex::encode(collection_id),
                hex::encode(node_id),
                payload.len()
            );
            if let Some(format) = decode {
                match format {
                    "raw" => {
                        let _ = write!(line, " payload_hex=0x{}", hex::encode(payload));
                    }
                    "json" | "auto" => {
                        let bytes = payload.clone();
                        if std::str::from_utf8(&bytes).is_ok() {
                            if let Ok(value) = OwnedValue::parse_bytes(&bytes) {
                                let _ = write!(line, " payload={value}");
                            }
                        }
                    }
                    _ => {}
                }
            }
            line
        }
        mtxdb::journal::Mutation::RemoveRecord {
            collection_id,
            node_id,
        } => format!(
            "kind=remove_record collection=0x{} node=0x{}",
            hex::encode(collection_id),
            hex::encode(node_id)
        ),
        mtxdb::journal::Mutation::DeleteCollection { collection_id } => format!(
            "kind=delete_collection collection=0x{}",
            hex::encode(collection_id)
        ),
    }
}

fn meta_checkpoints(root: &Path, report: &mut MetaReport) {
    for (_, dir) in meta_pool_dirs(root) {
        let path = dir.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE);
        if !path.is_file() {
            continue;
        }
        match mtxdb::index::checkpoint::read_checkpoint(&path) {
            Some(checkpoint) => {
                report.line(format!(
                    "checkpoint {}: covered_lsn={} fingerprint={:#x} collections={} packs={}",
                    path.display(),
                    checkpoint.covered_lsn,
                    checkpoint.fingerprint,
                    checkpoint.collections.len(),
                    checkpoint.pack_table.len()
                ));
                match meta_pack_fingerprint(&dir) {
                    Ok((fingerprint, issues)) => {
                        let issue_count = issues.len();
                        let invalid_count = issues
                            .iter()
                            .filter(|(level, _)| matches!(level, PackIssueLevel::Warn))
                            .count();
                        for (level, issue) in issues {
                            report.finding(level.as_str(), &dir, issue);
                        }
                        if invalid_count != 0 {
                            report.finding(
                                "NOTE",
                                &dir,
                                format!(
                                    "fingerprint comparison skipped: {invalid_count} invalid pack issue(s); {issue_count} total pack issue(s)"
                                ),
                            );
                        } else if fingerprint != checkpoint.fingerprint {
                            report.finding(
                                "WARN",
                                &path,
                                format!(
                                    "fingerprint mismatch: checkpoint={:#x} packs={:#x}",
                                    checkpoint.fingerprint, fingerprint
                                ),
                            );
                        }
                    }
                    Err(error) => report.finding("WARN", &dir, error),
                }
            }
            None => report.finding("WARN", &path, "invalid, corrupt, or unsupported checkpoint"),
        }
    }
}

fn meta_pack_fingerprint(dir: &Path) -> anyhow::Result<(u64, Vec<(PackIssueLevel, String)>)> {
    let mut packs = Vec::new();
    let mut issues = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "pack") {
            continue;
        }
        match pack_identity(&path) {
            PackIdentity::Valid { pack_id, length } => packs.push((pack_id, length)),
            PackIdentity::NonCanonical(message) => issues.push((
                PackIssueLevel::Note,
                format!("{}: {message}", path.display()),
            )),
            PackIdentity::Invalid(error) => {
                issues.push((PackIssueLevel::Warn, format!("{}: {error}", path.display())));
            }
        }
    }
    Ok((mtxdb::index::checkpoint::pack_fingerprint(&packs), issues))
}

fn meta_health(root: &Path, report: &mut MetaReport) {
    let shared_wal = root.join("wal.bin");
    for (pool, dir) in meta_pool_dirs(root) {
        let checkpoint_path = dir.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE);
        let checkpoint = mtxdb::index::checkpoint::read_checkpoint(&checkpoint_path);
        let journal_lsn = PackfileStorage::read_journal_lsn(&dir);
        let wal_path = if shared_wal.is_file() {
            shared_wal.clone()
        } else {
            dir.join("wal.bin")
        };
        let Ok(scan) = mtxdb::journal::Journal::scan_read_only(&wal_path) else {
            continue;
        };
        let wal_lsn = scan.groups.last().map(|group| group.last_lsn);
        if checkpoint.is_some() || journal_lsn != 0 || wal_lsn.is_some() {
            report.line(format!(
                "health {}: journal.lsn={} checkpoint.covered_lsn={} wal.last_lsn={}",
                pool.as_str(),
                journal_lsn,
                checkpoint.as_ref().map_or(0, |value| value.covered_lsn),
                wal_lsn.map_or_else(|| "header-only/empty".to_owned(), |lsn| lsn.to_string())
            ));
        }
        if journal_lsn < checkpoint.as_ref().map_or(0, |value| value.covered_lsn) {
            report.finding("NOTE", &dir, "journal.lsn is behind checkpoint coverage (may be transient during checkpoint persistence)");
        }
        if wal_lsn.is_some_and(|lsn| lsn < journal_lsn) {
            report.finding("WARN", &wal_path, "WAL last LSN is behind journal.lsn");
        }
    }
}

fn meta_deltas(root: &Path, report: &mut MetaReport, limit: usize, offset: usize) {
    for (_, dir) in meta_pool_dirs(root) {
        let epochs = match mtxdb::index::delta::list_epochs(&dir) {
            Ok(epochs) => epochs,
            Err(error) => {
                report.finding("ERROR", &dir, error);
                continue;
            }
        };
        let checkpoint_fingerprint = mtxdb::index::checkpoint::read_checkpoint(
            &dir.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE),
        )
        .map(|checkpoint| checkpoint.fingerprint);
        for (fingerprint, path) in epochs {
            if checkpoint_fingerprint != Some(fingerprint) {
                report.finding(
                    "NOTE",
                    &path,
                    format!("epoch is not selected by current checkpoint fingerprint ({checkpoint_fingerprint:?}); likely orphaned or stale"),
                );
            }
            if let Some(log) = mtxdb::index::delta::read_delta_log(&path) {
                report.line(format!(
                "delta {} epoch={fingerprint:#x}: v1 operations={} base={:#x} tail={:#x} committed_len={} torn_tail={} coverage={}",
                path.display(),
                log.operations.len(),
                log.base_fingerprint,
                log.tail_fingerprint,
                log.file_len,
                log.torn_tail,
                log.coverage.map_or_else(|| "none".to_owned(), |lsn| lsn.to_string())
            ));
                if log.torn_tail {
                    report.finding(
                        "WARN",
                        &path,
                        "truncated or invalid tail; committed prefix decoded",
                    );
                }
                for operation in log.operations.iter().skip(offset).take(limit) {
                    report.line(format!("  {}", format_delta_operation(operation)));
                }
            } else {
                report.finding("WARN", &path, "invalid, corrupt, or unsupported delta log");
            }
        }
    }
}

fn format_delta_frame(frame: &mtxdb::index::format::DeltaFrame) -> String {
    format!(
        "kind=incremental collection=0x{} bucket={} generation={} slot={:#x}",
        hex::encode(frame.collection_id),
        frame.bucket,
        frame.generation,
        frame.slot
    )
}

fn format_delta_operation(operation: &mtxdb::index::delta::DeltaOperation) -> String {
    use mtxdb::index::delta::DeltaOperation;
    match operation {
        DeltaOperation::Incremental(frame) => format_delta_frame(frame),
        DeltaOperation::CollectionSnapshot {
            collection_id,
            generation,
            order_key,
            index_blob,
        } => format!(
            "kind=collection_snapshot collection=0x{} generation={} order_key={} index_blob_len={}",
            hex::encode(collection_id),
            generation,
            order_key,
            index_blob.len()
        ),
        DeltaOperation::CollectionTombstone {
            collection_id,
            generation,
        } => format!(
            "kind=collection_tombstone collection=0x{} generation={}",
            hex::encode(collection_id),
            generation
        ),
        DeltaOperation::Coverage { covered_lsn } => {
            format!("kind=coverage covered_lsn={covered_lsn}")
        }
        DeltaOperation::Versions {
            covered_lsn,
            entries,
        } => format!(
            "kind=versions covered_lsn={covered_lsn} collections={}",
            entries.len()
        ),
        DeltaOperation::Redo(record) => {
            use mtxdb::index::redo::RedoOp;
            let detail = match record.op {
                RedoOp::Set {
                    full_hash,
                    pack_id,
                    offset,
                    record_len,
                    removed,
                } => format!(
                    "op=set hash=0x{} pack_id={pack_id} offset={offset} record_len={record_len} removed={removed}",
                    hex::encode(full_hash)
                ),
                RedoOp::CollectionTombstone => "op=collection_tombstone".to_owned(),
            };
            format!(
                "kind=redo collection=0x{} delta_seq={} base_generation={} {detail}",
                hex::encode(record.collection_id),
                record.delta_seq,
                record.base_generation
            )
        }
    }
}

fn meta_raw(root: &Path, report: &mut MetaReport, limit: usize, offset: usize) {
    let mut files = vec![
        root.join(mtxdb::layout::DB_META_FILENAME),
        root.join("wal.bin"),
    ];
    files.extend(meta_pool_dirs(root).flat_map(|(_, dir)| {
        [
            "pool.meta",
            "store.meta",
            "journal.lsn",
            "index.checkpoint",
            "shard_stats.bin",
            "shard_collections.bin",
        ]
        .into_iter()
        .map(move |name| dir.join(name))
    }));
    for (_, dir) in meta_pool_dirs(root) {
        if let Ok(epochs) = mtxdb::index::delta::list_epochs(&dir) {
            files.extend(epochs.into_iter().map(|(_, path)| path));
        }
    }
    let mut remaining_offset = offset;
    let mut remaining_limit = limit;
    for path in files.into_iter().filter(|path| path.is_file()) {
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                report.finding("ERROR", &path, error);
                continue;
            }
        };
        report.line(format!("raw {} ({} bytes)", path.display(), metadata.len()));
        if remaining_limit == 0 {
            break;
        }
        let total_chunks =
            usize::try_from(metadata.len().saturating_add(15) / 16).unwrap_or(usize::MAX);
        let skip = remaining_offset.min(total_chunks);
        remaining_offset = remaining_offset.saturating_sub(skip);
        let available = total_chunks.saturating_sub(skip);
        let take = available.min(remaining_limit);
        if take == 0 {
            continue;
        }
        let mut file = match fs::File::open(&path) {
            Ok(file) => file,
            Err(error) => {
                report.finding("ERROR", &path, error);
                continue;
            }
        };
        if let Err(error) = file.seek(SeekFrom::Start(
            u64::try_from(skip).unwrap_or(u64::MAX).saturating_mul(16),
        )) {
            report.finding("ERROR", &path, error);
            continue;
        }
        let byte_count = take.saturating_mul(16);
        let mut bytes = vec![0_u8; byte_count];
        let read = match file.read(&mut bytes) {
            Ok(read) => read,
            Err(error) => {
                report.finding("ERROR", &path, error);
                continue;
            }
        };
        bytes.truncate(read);
        for (index, chunk) in bytes.chunks(16).enumerate() {
            let hex = chunk
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            let ascii = chunk
                .iter()
                .map(|byte| {
                    if byte.is_ascii_graphic() {
                        *byte as char
                    } else {
                        '.'
                    }
                })
                .collect::<String>();
            report.line(format!(
                "  {:08x}  {hex:<47}  |{ascii}|",
                skip.saturating_add(index).saturating_mul(16)
            ));
        }
        remaining_limit = remaining_limit.saturating_sub(bytes.chunks(16).count());
    }
}

fn cmd_info_default_single(cli: &Cli, stats: bool) -> anyhow::Result<()> {
    let layout = open_layout(cli)?;
    let root = cli.single_dir();
    let db_meta = root.join(mtxdb::layout::DB_META_FILENAME);
    let wal = "shared";

    println!("database: {}", root.display());
    println!("  db.meta:      {}", metadata_file_summary(&db_meta));
    println!("  wal layout:   {wal}");
    println!();

    for (index, shard_type) in cli.shard_types().enumerate() {
        if index != 0 {
            println!();
        }
        let dir = pool_dir(&layout, shard_type)?;
        let packs = glob_pack_files(&dir)?;
        let pack_bytes = packs.iter().map(|(_, bytes, _)| *bytes).sum::<u64>();
        let collection_count =
            PackfileStorage::collection_summaries_from_disk(&dir).map(|summaries| summaries.len());
        let checkpoint = dir.join(mtxdb::index::checkpoint::INDEX_CHECKPOINT_FILE);
        let delta_epochs = mtxdb::index::delta::list_epochs(&dir).unwrap_or_default();

        println!("pool: {} ({})", shard_type.as_str(), dir.display());
        println!(
            "  pool.meta:    {}",
            metadata_file_summary(&dir.join("pool.meta"))
        );
        println!(
            "  store.meta:   {}",
            metadata_file_summary(&dir.join("store.meta"))
        );
        println!(
            "  packs:        {} ({})",
            packs.len(),
            fmt_bytes(pack_bytes)
        );
        println!(
            "  collections:  {}",
            collection_count.map_or_else(|| "unknown".to_owned(), |n| n.to_string())
        );
        println!(
            "  index:        checkpoint={} delta={}",
            if checkpoint.is_file() {
                fmt_bytes(fs::metadata(&checkpoint)?.len())
            } else {
                "missing".to_owned()
            },
            if delta_epochs.is_empty() {
                "missing".to_owned()
            } else {
                let bytes = delta_epochs
                    .iter()
                    .filter_map(|(_, path)| fs::metadata(path).ok().map(|meta| meta.len()))
                    .sum::<u64>();
                format!("{} epoch(s), {}", delta_epochs.len(), fmt_bytes(bytes))
            }
        );
        println!(
            "  shard sidecars: stats={} collections={}",
            if dir.join("shard_stats.bin").is_file() {
                "present"
            } else {
                "missing"
            },
            if dir.join("shard_collections.bin").is_file() {
                "present"
            } else {
                "missing"
            }
        );

        if stats {
            let (runtime, summaries) = read_stats_for_dir(&dir)?;
            println!("  stats:        {} pack summaries", summaries.len());
            if let Some(timing) = runtime.last_open_timings {
                println!("  open path:    {}", open_path_label(timing.path));
            }
        }
    }
    Ok(())
}

fn cmd_info(
    cli: &Cli,
    selector: &str,
    explicit: Option<InfoTarget>,
    deep: bool,
) -> anyhow::Result<()> {
    if cli.coalesce {
        return cmd_info_coalesced(cli, selector, deep, explicit);
    }
    run_multi_dir(cli, |sub_cli| {
        cmd_info_single(sub_cli, selector, deep, explicit)
    })
}

fn cmd_info_single(
    cli: &Cli,
    selector: &str,
    deep: bool,
    explicit: Option<InfoTarget>,
) -> anyhow::Result<()> {
    let (target, inventory) = classify_info_selector_with_inventory(cli, selector, explicit)?;
    match target {
        InfoTarget::Pack => cmd_info_pack(cli, selector, inventory.as_deref()),
        InfoTarget::Collection => cmd_info_collection(cli, selector, deep),
    }
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
    path: &Path,
    pack_id: PackId,
    shard_entries: &[(PackId, u64, u8)],
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
    println!("pack: {pack_id}");
    println!("path: {}", path.display());

    print_pack_lifetime(path);

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
                        format_id(&collection_id),
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

/// Display matching packs in the selected pools. An optional inventory must
/// belong to this database and avoids repeating discovery. Ambiguous prefixes
/// and discovery errors propagate; a missing pack is an error only when a
/// specific pool is selected.
fn cmd_info_pack(
    cli: &Cli,
    selector: &str,
    inventory: Option<&[PackLocation]>,
) -> anyhow::Result<()> {
    let selector = parse_pack_id_selector(selector)?;
    // Reuse the pack walk classification already did for this database when it
    // is available; only a bare prefix (classified without a walk) falls back
    // to globbing the pool.
    let pool_packs = |shard_type: ShardType| -> Option<Vec<(PackId, u64, u8)>> {
        inventory.map(|locations| {
            locations
                .iter()
                .filter(|location| location.shard_type == shard_type)
                .map(|location| (location.pack_id, location.file_bytes, location.version))
                .collect()
        })
    };
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let all = match pool_packs(shard_type) {
                Some(all) => all,
                None => glob_pack_files(&dir)?,
            };
            let ids: Vec<PackId> = all.iter().map(|&(id, _, _)| id).collect();
            if !ids.iter().any(|id| selector.matches(id)) {
                continue;
            }
            let pack_id = selector.resolve(ids.iter())?;
            let shard_entries: Vec<(PackId, u64, u8)> = all
                .into_iter()
                .filter(|&(id, _, _)| id == pack_id)
                .collect();
            if matched_any {
                println!();
                println!();
            }
            print_section_header(shard_type);
            let path = find_pack_path(&dir, pack_id)?;
            print_pack_info(&dir, &path, pack_id, &shard_entries, shard_type);
            matched_any = true;
        }
        if !matched_any {
            eprintln!("pack {selector}: not found");
        }
        return Ok(());
    }
    let dir = selected_pool_dir(cli)?;
    let shard_type = cli.require_shard_type()?;
    let all = match pool_packs(shard_type) {
        Some(all) => all,
        None => glob_pack_files(&dir)?,
    };
    let ids: Vec<PackId> = all.iter().map(|&(id, _, _)| id).collect();
    let pack_id = selector.resolve(ids.iter())?;
    let shard_entries: Vec<(PackId, u64, u8)> = all
        .into_iter()
        .filter(|&(id, _, _)| id == pack_id)
        .collect();
    if shard_entries.is_empty() {
        eprintln!("pack {selector}: not found");
        return Ok(());
    }
    let path = find_pack_path(&dir, pack_id)?;
    print_pack_info(&dir, &path, pack_id, &shard_entries, shard_type);
    Ok(())
}

fn print_collection_info(
    dir: &Path,
    collection_id: &[u8; 16],
    len: usize,
    mem: usize,
    capacity: u32,
    shards: &[PackId],
    deep: bool,
) {
    let hex = format_id(collection_id);
    println!(
        "collection {hex}: {len} nodes, {} index ({})",
        fmt_megabytes(mem),
        fmt_load_factor(len, capacity)
    );
    print_collection_shards(shards);
    let room_extension = open_pool_read_only(dir.to_path_buf())
        .ok()
        .and_then(|store| matrix_room_extension_from_store(&store, collection_id));
    if deep && room_extension.is_some() {
        match physical_layout(dir) {
            Ok(physical) => {
                if let Some(layout) = physical.collections.get(collection_id) {
                    println!("  physical (includes superseded frames):");
                    println!("    disk:             {}", fmt_bytes(layout.disk_bytes));
                    println!("    packs:            {}", layout.pack_bytes.len());
                    println!("    segments:         {}", layout.segments);
                    println!(
                        "    largest run:      {}",
                        fmt_bytes(layout.largest_segment_bytes)
                    );
                    println!(
                        "    avoidable spread: {}",
                        fmt_bytes(layout.avoidable_spread_bytes())
                    );
                    if layout.pack_bytes.len() > 1 {
                        let mut pack_bytes: Vec<_> = layout.pack_bytes.iter().collect();
                        pack_bytes.sort_unstable_by_key(|(pack_id, _)| **pack_id);
                        println!("    per pack:");
                        for (pack_id, bytes) in pack_bytes {
                            println!("      {}: {}", pack_id.filename_stem(), fmt_bytes(*bytes));
                        }
                    }
                }
            }
            Err(error) => eprintln!("  physical: unavailable ({error})"),
        }
    }
    match room_extension {
        Some(extension) => print_matrix_room_extension(&extension),
        None => println!("  {:<12} not found", "create:"),
    }
    print_collection_details(dir, collection_id, deep);
}

fn inspect_collection_in_dir(dir: &Path, collection_id: &[u8; 16], deep: bool) -> bool {
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
            print_collection_info(dir, collection_id, len, mem, capacity, &shards, deep);
            return true;
        }
        return false;
    }

    let Ok(store) = open_pool_read_only(dir.to_path_buf()) else {
        return false;
    };
    if let Some((len, mem, capacity)) = store.collection_index_info(collection_id) {
        let hex = format_id(collection_id);
        println!(
            "collection {hex}: {len} nodes, {} index ({})",
            fmt_megabytes(mem),
            fmt_load_factor(len, capacity)
        );
        let shards = store.collection_referenced_pack_ids(collection_id);
        print_collection_shards(&shards);
        match matrix_room_extension_from_store(&store, collection_id) {
            Some(extension) => print_matrix_room_extension(&extension),
            None => println!("  {:<12} not found", "create:"),
        }
        print_collection_details(dir, collection_id, deep);
        return true;
    }
    false
}

fn cmd_info_collection(cli: &Cli, collection: &str, deep: bool) -> anyhow::Result<()> {
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let dir = pool_dir(&db_layout, shard_type)?;
            let collection_id = parse_collection_selector(collection)?;
            let has_collection =
                if let Some(summaries) = PackfileStorage::collection_summaries_from_disk(&dir) {
                    summaries.iter().any(|(id, _, _, _)| id == &collection_id)
                } else if let Ok(store) = open_pool_read_only(dir.clone()) {
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
                inspect_collection_in_dir(&dir, &collection_id, deep);
                matched_any = true;
            }
        }
        if !matched_any {
            eprintln!(
                "collection {}: not found",
                format_id(&parse_collection_selector(collection)?)
            );
        }
        return Ok(());
    }

    let collection_id = parse_collection_selector(collection)?;
    let hex = format_id(&collection_id);
    let dir = selected_pool_dir(cli)?;

    if inspect_collection_in_dir(&dir, &collection_id, deep) {
        return Ok(());
    }

    eprintln!(
        "collection {hex}: not found{}",
        other_shard_type_hint(cli, &collection_id)
    );
    Ok(())
}

/// Structured Matrix room configuration carried in a collection's
/// [`CollectionMetadata::extension`] blob.
///
/// The blob is self-describing so a reader can learn the room version (which
/// selects the redaction and reference-hash rules) from the header alone,
/// without seeking to and parsing the establishment record. Core never
/// interprets it; it is handed back to the CLI on open.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MatrixRoomExtension {
    room_id: Option<String>,
    create_event_id: Option<String>,
    room_version: Option<String>,
    creator: Option<String>,
    creation_ts: Option<i64>,
}

const MATRIX_ROOM_EXT: &str = "matrix.room";
const MATRIX_ROOM_EXT_FMT: u32 = 1;

impl MatrixRoomExtension {
    /// Read the extension out of an import batch. The caller already knows the
    /// batch establishes the room, so no disk access is needed.
    fn from_events(events: &[OwnedValue]) -> Self {
        let mut extension = Self::default();
        for event in events {
            if extension.room_id.is_none() {
                extension.room_id = event_room_id(event).map(str::to_owned);
            }
            if extension.create_event_id.is_none() {
                if let Some(create_event_id) = matrix_create_event_id(event) {
                    extension.create_event_id = Some(create_event_id.to_owned());
                    extension.creator =
                        nested_event_string_field(event, "content", "creator").map(str::to_owned);
                    extension.room_version =
                        nested_event_string_field(event, "content", "room_version")
                            .map(str::to_owned);
                    extension.creation_ts =
                        event.get("origin_server_ts").and_then(OwnedValue::as_i64);
                }
            }
            if extension.room_id.is_some() && extension.create_event_id.is_some() {
                break;
            }
        }
        extension
    }

    /// Serialize the self-describing JSON blob stored in
    /// [`CollectionMetadata::extension`].
    fn encode_blob(&self) -> Vec<u8> {
        let mut fields = vec![
            format!("\"ext\":{}", OwnedValue::from(MATRIX_ROOM_EXT).encode()),
            format!("\"fmt\":{MATRIX_ROOM_EXT_FMT}"),
        ];
        for (key, value) in [
            ("room_id", &self.room_id),
            ("create_event_id", &self.create_event_id),
            ("room_version", &self.room_version),
            ("creator", &self.creator),
        ] {
            if let Some(value) = value {
                fields.push(format!(
                    "\"{key}\":{}",
                    OwnedValue::from(value.as_str()).encode()
                ));
            }
        }
        if let Some(timestamp) = self.creation_ts {
            fields.push(format!("\"creation_ts\":{timestamp}"));
        }
        format!("{{{}}}", fields.join(",")).into_bytes()
    }

    fn decode_blob(blob: &[u8]) -> Option<Self> {
        let bytes = blob.to_vec();
        let value = OwnedValue::parse_bytes(&bytes).ok()?;
        if event_string_field(&value, "ext") != Some(MATRIX_ROOM_EXT) {
            return None;
        }
        Some(Self {
            room_id: event_string_field(&value, "room_id").map(str::to_owned),
            create_event_id: event_string_field(&value, "create_event_id").map(str::to_owned),
            room_version: event_string_field(&value, "room_version").map(str::to_owned),
            creator: event_string_field(&value, "creator").map(str::to_owned),
            creation_ts: value.get("creation_ts").and_then(OwnedValue::as_i64),
        })
    }
}

/// The Matrix room extension recorded on disk, if any.
fn matrix_room_extension_from_store(
    store: &PackfileStorage,
    collection_id: &[u8; 16],
) -> Option<MatrixRoomExtension> {
    let metadata = store
        .get_collection_metadata(collection_id)
        .ok()
        .flatten()?;
    MatrixRoomExtension::decode_blob(metadata.extension.as_deref()?)
}

/// Build the genesis metadata record for a just-established Matrix room.
///
/// `collection_canonical_id` is the resolved external identity — the room id,
/// or the create event's event id for room version 12. It is always present:
/// callers only reach this after [`resolve_import_collection`] accepted the
/// batch as an establishment, which requires a resolved canonical identity.
/// The v12 create-event-only case is exactly where `extension.room_id` is
/// `None` yet a canonical id (the create event's id) still exists.
fn collection_metadata_for(
    template: &CollectionTemplate,
    extension: &MatrixRoomExtension,
    collection_canonical_id: &str,
) -> CollectionMetadata {
    CollectionMetadata {
        member_namespace: template.collection_key.member_namespace,
        collection_canonical_id: collection_canonical_id.as_bytes().to_vec(),
        record_id_rule: template.record_id_rule.clone(),
        payload: template.payload.clone(),
        extension: Some(extension.encode_blob()),
        role: Some("event".to_owned()),
        schema: Some("matrix.event.v1".to_owned()),
    }
}

/// Build and persist the state HAMTs and `STGP` derivation records for a room.
///
/// `STAT` content is materialized once per distinct state `LtHash`; every
/// derivation, including merges that reach the same state through different
/// parents, gets its own `STGP` record whose id is derived from the sorted
/// parent instance ids and the `LtHash`. Returns the number of records written.
/// Node ids are the first 16 bytes of each node's structural hash; roots use a
/// domain-separated id derived from the room and `LtHash`, so roots and
/// structural-hash nodes cannot collide. A record already present with
/// identical bytes is skipped; a different value under the same id is a hard
/// error.
#[allow(clippy::too_many_lines)]
fn materialize_state_hamts(
    state_store: &PackfileStorage,
    room_id: &str,
    instances: &HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance>,
) -> anyhow::Result<usize> {
    let Some(stat_collection) = crate::state_hamt::state_hamt_collection_id(room_id) else {
        return Ok(0);
    };
    let Some(stgp_collection) = mtxdb::state_group_collection_id(room_id) else {
        return Ok(0);
    };
    let room_prefix = crate::state_hamt::room_hamt_prefix(room_id);

    let mut seen_stat: HashMap<[u8; 16], bytes::Bytes> = HashMap::new();
    let mut stat_records: Vec<(NodeId, NodeData)> = Vec::new();
    let mut seen_stgp: HashMap<[u8; 16], bytes::Bytes> = HashMap::new();
    let mut stgp_records: Vec<(NodeId, NodeData)> = Vec::new();
    let mut content_done: HashSet<[u8; 32]> = HashSet::new();

    // A group's content is immutable and addressed by its lattice digest, so a
    // root that already exists means every node under it was already written.
    // Rebuilding the tree to discover that is the dominant per-file cost when
    // the same room arrives in many chunks: check the roots in one batch first
    // and skip the build for content this store already has.
    let mut existing_roots: HashSet<NodeId> = HashSet::new();
    if !instances.is_empty() && state_store.collection_exists(&stat_collection) {
        let mut root_ids = Vec::with_capacity(instances.len());
        let mut seen_content = HashSet::with_capacity(instances.len());
        for instance in instances.values() {
            if instance.set.entries.is_empty() || !seen_content.insert(instance.lthash) {
                continue;
            }
            root_ids.push(crate::state_hamt::state_hamt_root_node_id(
                room_id,
                &instance.lthash,
            ));
        }
        for (node, record) in root_ids
            .iter()
            .zip(state_store.get_many(&stat_collection, &root_ids)?)
        {
            if record.is_some() {
                existing_roots.insert(*node);
            }
        }
    }

    for (instance_id, instance) in instances {
        if instance.set.entries.is_empty() {
            continue;
        }
        let content_root = crate::state_hamt::state_hamt_root_node_id(room_id, &instance.lthash);
        if content_done.insert(instance.lthash) && !existing_roots.contains(&content_root) {
            let mut entries: Vec<(String, String, String)> = instance
                .set
                .entries
                .iter()
                .map(|((event_type, state_key), event_id)| {
                    (event_type.clone(), state_key.clone(), event_id.clone())
                })
                .collect();
            entries.sort();
            let built = crate::state_hamt::build_state_hamt(room_id, &room_prefix, &entries)
                .map_err(|error| anyhow!(error))?;
            if built.state_group_id != instance.lthash {
                bail!("state HAMT digest mismatch for room {room_id}");
            }
            for (hash, node_bytes) in built.nodes {
                let mut id = [0u8; 16];
                id.copy_from_slice(&hash[..16]);
                push_hamt_record(
                    &mut seen_stat,
                    &mut stat_records,
                    id,
                    bytes::Bytes::from(node_bytes),
                )?;
            }
            let root_node =
                crate::state_hamt::state_hamt_root_node_id(room_id, &built.state_group_id);
            push_hamt_record(
                &mut seen_stat,
                &mut stat_records,
                root_node,
                bytes::Bytes::from(built.root_record),
            )?;
        }

        let record = mtxdb::StateGroupInstance {
            parents: instance.parents.clone(),
            lthash: instance.lthash,
            root_id: crate::state_hamt::state_hamt_root_node_id(room_id, &instance.lthash),
            derivation: instance.derivation,
        };
        push_hamt_record(
            &mut seen_stgp,
            &mut stgp_records,
            *instance_id,
            bytes::Bytes::from(mtxdb::encode_state_group_record(&record)),
        )?;
    }

    let stat_records = if stat_records.is_empty() {
        Vec::new()
    } else {
        prepare_state_collection(state_store, &stat_collection, &state_hamt_metadata(room_id))?;
        let records = filter_new_records(state_store, &stat_collection, stat_records)?;
        if !records.is_empty() {
            state_store.put_many(&stat_collection, &records)?;
        }
        records
    };
    let stgp_records = if stgp_records.is_empty() {
        Vec::new()
    } else {
        prepare_state_collection(
            state_store,
            &stgp_collection,
            &state_group_metadata(room_id),
        )?;
        let records = filter_new_records(state_store, &stgp_collection, stgp_records)?;
        if !records.is_empty() {
            state_store.put_many(&stgp_collection, &records)?;
        }
        records
    };
    Ok(stat_records.len().saturating_add(stgp_records.len()))
}

/// Establish a state collection, resetting one that already has records but no
/// genesis metadata.
///
/// Older server migrations wrote HAMT roots directly with `put_many` and no
/// collection metadata. Those collections cannot be read or appended to. The
/// import is rebuilding the room's state, so discard the unusable records and
/// establish the collection before writing the new ones.
fn prepare_state_collection(
    state_store: &PackfileStorage,
    collection_id: &[u8; 16],
    metadata: &CollectionMetadata,
) -> anyhow::Result<()> {
    if state_store
        .get_collection_metadata(collection_id)?
        .is_none()
        && state_store.collection_exists(collection_id)
    {
        anyhow::ensure!(
            state_store.collection_len(collection_id)?.unwrap_or(0) == 0,
            "state collection {} has records but no genesis metadata; refusing to reset records that may be referenced by imported state roots",
            format_id(collection_id)
        );
    }
    state_store
        .ensure_collection_metadata(collection_id, metadata)
        .context("writing state collection metadata")?;
    Ok(())
}

/// Drop records already present in `collection` and reject an id already bound
/// to different bytes.
///
/// `STAT` content and `STGP` instance records are immutable, so a write must
/// never overwrite an existing payload: a differing payload under the same id
/// is a hard collision, not a last-writer-wins update.
fn filter_new_records<S: StorageEngine + ?Sized>(
    state_store: &S,
    collection_id: &[u8; 16],
    records: Vec<(NodeId, NodeData)>,
) -> anyhow::Result<Vec<(NodeId, NodeData)>> {
    if records.is_empty() {
        return Ok(records);
    }
    let ids: Vec<NodeId> = records.iter().map(|(id, _)| *id).collect();
    let existing = state_store.get_many_bytes(collection_id, &ids)?;
    let mut fresh = Vec::new();
    for ((id, data), existing) in records.into_iter().zip(existing) {
        match existing {
            None => fresh.push((id, data)),
            Some(bytes) if bytes == data.bytes => {}
            Some(_) => bail!(
                "immutable record {} in collection {} already exists with different bytes",
                format_id(&id),
                format_id(collection_id)
            ),
        }
    }
    Ok(fresh)
}

/// Insert a HAMT record unless an identical one was already queued.
fn push_hamt_record(
    seen: &mut HashMap<[u8; 16], bytes::Bytes>,
    records: &mut Vec<(NodeId, NodeData)>,
    id: [u8; 16],
    value: bytes::Bytes,
) -> anyhow::Result<()> {
    match seen.entry(id) {
        std::collections::hash_map::Entry::Occupied(existing) => {
            if existing.get() != &value {
                bail!(
                    "HAMT record id {} collides with a different value",
                    format_id(&id)
                );
            }
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(value.clone());
            records.push((id, NodeData::new(value)));
        }
    }
    Ok(())
}

/// Genesis metadata for a room's `STAT` HAMT collection, matching the layout
/// `sithnapse` writes.
fn state_hamt_metadata(room_id: &str) -> CollectionMetadata {
    CollectionMetadata {
        member_namespace: Some(*b"STAT"),
        collection_canonical_id: room_id.as_bytes().to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("state_hamt".to_owned()),
        schema: Some("sithnapse.state-hamt.v1".to_owned()),
    }
}

/// Genesis metadata for a room's `STGP` state-group instance collection.
fn state_group_metadata(room_id: &str) -> CollectionMetadata {
    CollectionMetadata {
        member_namespace: Some(*b"STGP"),
        collection_canonical_id: room_id.as_bytes().to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("state_group".to_owned()),
        schema: Some("sithnapse.state-group.v1".to_owned()),
    }
}

fn print_matrix_room_extension(extension: &MatrixRoomExtension) {
    if let Some(room_id) = &extension.room_id {
        println!("  {:<12} {room_id}", "Matrix room:");
    }
    let Some(create_event_id) = &extension.create_event_id else {
        println!("  {:<12} not found", "create:");
        return;
    };
    let mut create = create_event_id.clone();
    if let Some(creator) = &extension.creator {
        let _ = write!(create, "; creator {creator}");
    }
    if let Some(room_version) = &extension.room_version {
        let _ = write!(create, "; room version {room_version}");
    }
    if let Some(timestamp) = extension.creation_ts {
        let _ = write!(create, "; {}", format_matrix_timestamp(timestamp));
    }
    println!("  {:<12} {create}", "create:");
}

/// Live-record statistics for a collection whose records are Matrix events,
/// gathered by one pass over the packs that hold it.
#[derive(Default)]
struct RoomEventStats {
    /// Physical frames for this collection, including superseded rewrites.
    frames: usize,
    /// Distinct live records excluding the genesis metadata record.
    records: usize,
    /// Records whose payload is empty (tombstones).
    tombstones: usize,
    /// Records whose payload parsed as a JSON object.
    json_events: usize,
    /// Live records by recognized payload kind.
    kinds: HashMap<String, usize>,
    min_size: Option<usize>,
    max_size: usize,
    state_events: usize,
    payload_bytes: u64,
    types: HashMap<String, usize>,
    senders: HashMap<String, usize>,
    servers: HashMap<String, usize>,
    months: HashMap<String, usize>,
    min_ts: Option<i64>,
    max_ts: Option<i64>,
    min_depth: Option<i64>,
    max_depth: Option<i64>,
    event_ids: HashSet<String>,
    prev_refs: HashSet<String>,
    auth_refs: HashSet<String>,
    /// Latest (by timestamp) empty-state-key event content per room-config type.
    room_state: HashMap<String, (i64, OwnedValue)>,
    /// Latest membership per user: (timestamp, membership).
    members: HashMap<String, (i64, String)>,
    /// Largest payloads: (bytes, event id, type).
    largest: Vec<(usize, String, String)>,
}

/// Room-config state types worth summarizing in `info --stats`.
const ROOM_CONFIG_TYPES: [&str; 6] = [
    "m.room.name",
    "m.room.topic",
    "m.room.canonical_alias",
    "m.room.join_rules",
    "m.room.history_visibility",
    "m.room.encryption",
];

fn string_array(fields: &Object, key: &str) -> Vec<String> {
    match fields.get(key) {
        Some(OwnedValue::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                OwnedValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one pass gathers every statistic; splitting would only thread the accumulator"
)]
fn scan_room_event_stats(dir: &Path, collection_id: &[u8; 16]) -> anyhow::Result<RoomEventStats> {
    let pool =
        ShardPool::open_read_only(dir.to_path_buf()).context("failed to open shard store")?;
    let mut shards = pool.all_shards();
    shards.sort_unstable_by_key(|(_, shard)| shard.pack_id);
    let collection_packs = PackfileStorage::collection_shards_from_disk(dir)
        .and_then(|mut collections| collections.remove(collection_id))
        .map(|packs| packs.into_iter().collect::<HashSet<_>>());

    let mut stats = RoomEventStats::default();
    // Later frames supersede earlier ones for the same record id.
    let mut latest: HashMap<[u8; 16], (Arc<mtxdb::shard::Shard>, u64)> = HashMap::new();
    for (_, shard) in shards {
        if let Some(packs) = &collection_packs {
            if !packs.contains(&shard.pack_id) {
                continue;
            }
        }
        for record in mtxdb::packfile::scan_packfile_iter(&shard.path, false)? {
            let (record_collection, record_id, offset) = record?;
            if &record_collection != collection_id {
                continue;
            }
            stats.frames = stats.frames.saturating_add(1);
            latest.insert(record_id, (Arc::clone(&shard), offset));
        }
    }
    for (record_id, (shard, offset)) in latest {
        if record_id == mtxdb::COLLECTION_METADATA_RECORD_ID {
            continue;
        }
        let record = ShardPool::read_at_committed(&shard, offset, false)?;
        stats.records = stats.records.saturating_add(1);
        if record.data.is_empty() {
            stats.tombstones = stats.tombstones.saturating_add(1);
            continue;
        }
        let size = record.data.len();
        stats.payload_bytes = stats
            .payload_bytes
            .saturating_add(u64::try_from(size).unwrap_or(u64::MAX));
        stats.min_size = Some(stats.min_size.map_or(size, |m| m.min(size)));
        stats.max_size = stats.max_size.max(size);
        let bytes = record.data.to_vec();
        let parsed = OwnedValue::parse_bytes(&bytes).ok();
        let Some(event @ OwnedValue::Object(_)) = parsed else {
            let kind = classify_record_payload(&record.data);
            let count = stats.kinds.entry(kind.to_owned()).or_default();
            *count = count.saturating_add(1);
            continue;
        };
        let OwnedValue::Object(fields) = &event else {
            continue;
        };
        stats.json_events = stats.json_events.saturating_add(1);
        let count = stats.kinds.entry("JSON event".to_owned()).or_default();
        *count = count.saturating_add(1);
        let kind = event_string_field(&event, "type").unwrap_or("");
        let ts = fields.get("origin_server_ts").and_then(OwnedValue::as_i64);
        let event_id = event_id(&event).unwrap_or("").to_owned();
        if is_state_event(&event) {
            stats.state_events = stats.state_events.saturating_add(1);
            let state_key = event_string_field(&event, "state_key").unwrap_or("");
            let when = ts.unwrap_or(0);
            if kind == "m.room.member" {
                let membership =
                    nested_event_string_field(&event, "content", "membership").unwrap_or("?");
                let newer = stats
                    .members
                    .get(state_key)
                    .is_none_or(|(seen, _)| when >= *seen);
                if newer {
                    stats
                        .members
                        .insert(state_key.to_owned(), (when, membership.to_owned()));
                }
            } else if state_key.is_empty() && ROOM_CONFIG_TYPES.contains(&kind) {
                let newer = stats
                    .room_state
                    .get(kind)
                    .is_none_or(|(seen, _)| when >= *seen);
                if newer {
                    if let Some(content) = fields.get("content") {
                        stats
                            .room_state
                            .insert(kind.to_owned(), (when, content.clone()));
                    }
                }
            }
        }
        let bump = |map: &mut HashMap<String, usize>, key: &str| {
            let count = map.entry(key.to_owned()).or_default();
            *count = count.saturating_add(1);
        };
        bump(&mut stats.types, kind);
        if let Some(sender) = event_string_field(&event, "sender") {
            bump(&mut stats.senders, sender);
            if let Some((_, server)) = sender.split_once(':') {
                bump(&mut stats.servers, server);
            }
        }
        if let Some(ts) = ts {
            stats.min_ts = Some(stats.min_ts.map_or(ts, |m| m.min(ts)));
            stats.max_ts = Some(stats.max_ts.map_or(ts, |m| m.max(ts)));
            bump(&mut stats.months, &format_utc_ms(ts)[..7]);
        }
        if let Some(depth) = fields.get("depth").and_then(OwnedValue::as_i64) {
            stats.min_depth = Some(stats.min_depth.map_or(depth, |m| m.min(depth)));
            stats.max_depth = Some(stats.max_depth.map_or(depth, |m| m.max(depth)));
        }
        stats.prev_refs.extend(string_array(fields, "prev_events"));
        stats.auth_refs.extend(string_array(fields, "auth_events"));
        stats
            .largest
            .push((size, event_id.clone(), kind.to_owned()));
        stats
            .largest
            .sort_by_key(|entry| std::cmp::Reverse(entry.0));
        stats.largest.truncate(3);
        stats.event_ids.insert(event_id);
    }
    Ok(stats)
}

/// Format a millisecond Unix timestamp as `YYYY-MM-DD HH:MM` UTC.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "calendar arithmetic on one i64 timestamp; every intermediate is bounded by it"
)]
fn format_utc_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let minutes_of_day = secs.rem_euclid(86_400) / 60;
    // Civil-from-days (proleptic Gregorian), after Howard Hinnant.
    let z = days.saturating_add(719_468);
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        minutes_of_day / 60,
        minutes_of_day % 60
    )
}

/// Format a Matrix creation timestamp as the compact human-facing UTC form
/// used by room summaries, e.g. `5:30 PM 30 Sep 2109`.
fn format_matrix_timestamp(ms: i64) -> String {
    let formatted = format_utc_ms(ms);
    let Some((date, time)) = formatted.split_once(' ') else {
        return formatted;
    };
    let mut date_parts = date.split('-');
    let (Some(year), Some(month), Some(day)) =
        (date_parts.next(), date_parts.next(), date_parts.next())
    else {
        return formatted;
    };
    let mut time_parts = time.split(':');
    let (Some(hour), Some(minute)) = (time_parts.next(), time_parts.next()) else {
        return formatted;
    };
    let Ok(hour24) = hour.parse::<u32>() else {
        return formatted;
    };
    let hour12 = match hour24 % 12 {
        0 => 12,
        hour => hour,
    };
    let meridiem = if hour24 < 12 { "AM" } else { "PM" };
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let Ok(month_number) = month.parse::<usize>() else {
        return formatted;
    };
    let Some(month_name) = month_number
        .checked_sub(1)
        .and_then(|index| months.get(index))
    else {
        return formatted;
    };
    format!("{hour12}:{minute} {meridiem} {day} {month_name} {year}")
}

fn top_counts(counts: &HashMap<String, usize>, limit: usize) -> String {
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    entries
        .into_iter()
        .take(limit)
        .map(|(name, count)| format!("{name} ({count})"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Name a non-JSON-object payload by its recognizable wire format.
fn classify_record_payload(data: &[u8]) -> &'static str {
    if data.starts_with(b"MTHN") {
        "HAMT node"
    } else if data.starts_with(b"MTHR") {
        "HAMT state-group root"
    } else if data.starts_with(b"STGP") {
        "state-group instance"
    } else if data.starts_with(b"AUX1") {
        "auxiliary index value"
    } else if decode_event_json_record(data).is_some() {
        "Synapse event_json mirror"
    } else if data
        .iter()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| matches!(byte, b'{' | b'['))
    {
        "malformed JSON"
    } else {
        "opaque binary"
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "a flat list of independent report lines"
)]
fn print_room_event_stats(stats: &RoomEventStats) {
    let live = stats.records.saturating_sub(stats.tombstones);
    println!(
        "  {:<12} {live} live, {} tombstoned, {} superseded frames",
        "records:",
        stats.tombstones,
        stats.frames.saturating_sub(stats.records.saturating_add(1)),
    );
    if !stats.kinds.is_empty() {
        println!("  {:<12} {}", "kinds:", top_counts(&stats.kinds, 6));
    }
    if let Some(min) = stats.min_size {
        println!(
            "  {:<12} min {}; avg {}; max {}",
            "payload:",
            fmt_bytes(u64::try_from(min).unwrap_or(u64::MAX)),
            fmt_bytes(
                stats
                    .payload_bytes
                    .checked_div(u64::try_from(live).unwrap_or(0))
                    .unwrap_or(0)
            ),
            fmt_bytes(u64::try_from(stats.max_size).unwrap_or(u64::MAX))
        );
    }
    if stats.json_events == 0 {
        return;
    }
    println!(
        "  {:<12} {} JSON, {} state",
        "events:", stats.json_events, stats.state_events
    );
    let text = |kind: &str, field: &str| {
        stats
            .room_state
            .get(kind)
            .and_then(|(_, content)| match content {
                OwnedValue::Object(fields) => match fields.get(field) {
                    Some(OwnedValue::String(value)) if !value.is_empty() => Some(value.clone()),
                    _ => None,
                },
                _ => None,
            })
    };
    let mut config = Vec::new();
    if let Some(name) = text("m.room.name", "name") {
        config.push(format!("name {name:?}"));
    }
    if let Some(topic) = text("m.room.topic", "topic") {
        let topic: String = topic.chars().take(80).collect();
        config.push(format!("topic {topic:?}"));
    }
    if let Some(alias) = text("m.room.canonical_alias", "alias") {
        config.push(format!("alias {alias}"));
    }
    if let Some(rule) = text("m.room.join_rules", "join_rule") {
        config.push(format!("join rule {rule}"));
    }
    if let Some(visibility) = text("m.room.history_visibility", "history_visibility") {
        config.push(format!("history {visibility}"));
    }
    if let Some(algorithm) = text("m.room.encryption", "algorithm") {
        config.push(format!("encrypted ({algorithm})"));
    }
    if !config.is_empty() {
        println!(
            "  {:<12} {} (latest by timestamp)",
            "room state:",
            config.join("; ")
        );
    }
    if !stats.members.is_empty() {
        let mut by_membership: HashMap<String, usize> = HashMap::new();
        for (_, membership) in stats.members.values() {
            let count = by_membership.entry(membership.clone()).or_default();
            *count = count.saturating_add(1);
        }
        println!(
            "  {:<12} {} users: {}",
            "members:",
            stats.members.len(),
            top_counts(&by_membership, 6)
        );
    }
    if let (Some(min), Some(max)) = (stats.min_ts, stats.max_ts) {
        println!(
            "  {:<12} {} \u{2192} {} UTC ({} active months; busiest {})",
            "time span:",
            format_utc_ms(min),
            format_utc_ms(max),
            stats.months.len(),
            top_counts(&stats.months, 3)
        );
    }
    if let (Some(min), Some(max)) = (stats.min_depth, stats.max_depth) {
        println!("  {:<12} {min} \u{2192} {max}", "depth:");
    }
    let missing_prev = stats.prev_refs.difference(&stats.event_ids).count();
    let missing_auth = stats.auth_refs.difference(&stats.event_ids).count();
    let extremities = stats.event_ids.difference(&stats.prev_refs).count();
    println!(
        "  {:<12} {extremities} forward extremities; {missing_prev} missing prev_events; {missing_auth} missing auth_events",
        "dag:"
    );
    if !stats.types.is_empty() {
        println!(
            "  {:<12} {} distinct; {}",
            "types:",
            stats.types.len(),
            top_counts(&stats.types, 8)
        );
    }
    if !stats.senders.is_empty() {
        println!(
            "  {:<12} {} distinct; {}",
            "senders:",
            stats.senders.len(),
            top_counts(&stats.senders, 5)
        );
    }
    if !stats.servers.is_empty() {
        println!(
            "  {:<12} {} distinct; {}",
            "servers:",
            stats.servers.len(),
            top_counts(&stats.servers, 5)
        );
    }
    if !stats.largest.is_empty() {
        let largest = stats
            .largest
            .iter()
            .map(|(bytes, id, kind)| {
                format!(
                    "{} {kind} {id}",
                    fmt_bytes(u64::try_from(*bytes).unwrap_or(u64::MAX))
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        println!("  {:<12} {largest}", "largest:");
    }
}

/// Print a collection's identity rules from its genesis record (cheap), and,
/// with `deep`, event statistics from a full pass over its records.
fn print_collection_details(dir: &Path, collection_id: &[u8; 16], deep: bool) {
    let mut matrix_room = false;
    if let Ok(store) = open_pool_read_only(dir.to_path_buf()) {
        if let Ok(Some(metadata)) = store.get_collection_metadata(collection_id) {
            matrix_room = metadata
                .extension
                .as_deref()
                .and_then(MatrixRoomExtension::decode_blob)
                .is_some();
            let pool = metadata.member_namespace.map_or_else(
                || "none".to_owned(),
                |dst| String::from_utf8_lossy(&dst).into_owned(),
            );
            let role_str = metadata.role.as_ref().map_or_else(
                || "unspecified".to_owned(),
                std::string::ToString::to_string,
            );
            let schema_str = metadata.schema.as_deref().unwrap_or("none");
            let verified = metadata.verify_collection_id(collection_id);
            let verify_str = if verified {
                "valid"
            } else {
                "MISMATCH / CORRUPT"
            };
            println!(
                "  {:<12} pool {pool}; role {role_str}; schema {schema_str}; canonical id {} (verification: {verify_str})",
                "identity:",
                String::from_utf8_lossy(&metadata.collection_canonical_id)
            );
            let policy = match &metadata.record_id_rule.policy {
                FrameIdPolicy::Pointer { pointer } => format!("{pointer} value"),
                other => format!("{other:?}"),
            };
            println!(
                "  {:<12} {:?} of {policy}; payload {:?}",
                "record id:", metadata.record_id_rule.digest_algorithm, metadata.payload
            );
        }
    }
    if !deep {
        println!("  (run with --stats for event statistics; this scans every record)");
        return;
    }
    if !matrix_room {
        println!(
            "  {:<12} skipped (collection has no Matrix room metadata)",
            "stats:"
        );
        return;
    }
    let started = std::time::Instant::now();
    match scan_room_event_stats(dir, collection_id) {
        Ok(stats) => {
            print_room_event_stats(&stats);
            println!(
                "  {:<12} {:.1}s",
                "scan time:",
                started.elapsed().as_secs_f64()
            );
        }
        Err(error) => println!("  {:<12} failed: {error:#}", "stats:"),
    }
}

fn print_collection_shards(shards: &[PackId]) {
    match shards {
        [shard] => println!("  {:<12} {shard}", "pack:"),
        [] => {}
        _ => println!(
            "  {:<12} {}",
            "packs:",
            shards
                .iter()
                .map(PackId::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// An operator-facing pack identifier. Slots are deliberately not accepted
/// here: they are recycled implementation details, while a pack address is the
/// permanent identity printed by `mtxdb shards`. A selector is either the full
/// 32-hex address or a 1–16 hex filename prefix; the prefix is resolved against
/// a pool's discovered packs, where it must be unique.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PackSelector {
    /// A full 128-bit address, matched exactly.
    Exact(PackId),
    /// A lowercase hex prefix of an address (typically the 16-hex filename stem).
    Prefix(String),
}

impl PackSelector {
    /// Parse a lowercase `0x`-prefixed full address or 1–16 hex digit prefix.
    ///
    /// Rejects malformed selectors and the full all-zero address. A zero prefix
    /// is allowed; existence and uniqueness are checked separately by `resolve`.
    fn parse(selector: &str) -> anyhow::Result<Self> {
        let Some(hex) = selector.strip_prefix("0x") else {
            bail!(
                "invalid pack ID `{selector}`; use the lowercase 0x-prefixed ID shown by \
                 `mtxdb shards`"
            );
        };
        if hex.is_empty()
            || !hex
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            bail!("invalid pack ID `{selector}`; expected lowercase hexadecimal digits after 0x");
        }
        if hex.len() == PACK_ID_HEX_LEN {
            let id = PackId::from_hex(hex)
                .ok_or_else(|| anyhow!("invalid pack ID `{selector}`; not a 128-bit address"))?;
            return Ok(Self::Exact(id));
        }
        if hex.len() > PACK_FILENAME_PREFIX_HEX {
            bail!(
                "invalid pack ID `{selector}`; expected 1–{PACK_FILENAME_PREFIX_HEX} hex digits \
                 (a filename prefix) or {PACK_ID_HEX_LEN} (the full address)"
            );
        }
        Ok(Self::Prefix(hex.to_owned()))
    }

    /// Whether `id` matches this selector.
    fn matches(&self, id: &PackId) -> bool {
        match self {
            Self::Exact(exact) => id == exact,
            Self::Prefix(prefix) => id.has_hex_prefix(prefix),
        }
    }

    /// Resolve to exactly one address among `candidates`, erroring on zero or
    /// multiple matches (a short prefix must be unambiguous).
    fn resolve<'a>(
        &self,
        candidates: impl IntoIterator<Item = &'a PackId>,
    ) -> anyhow::Result<PackId> {
        let matches: Vec<PackId> = candidates
            .into_iter()
            .filter(|id| self.matches(id))
            .copied()
            .collect();
        match matches.as_slice() {
            [id] => Ok(*id),
            [] => bail!("pack {self}: not found"),
            _ => bail!("pack {self}: matches multiple packs; use a longer selector"),
        }
    }
}

impl std::fmt::Display for PackSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact(id) => write!(f, "{id}"),
            Self::Prefix(prefix) => write!(f, "0x{prefix}"),
        }
    }
}

/// Hex digits in a full [`PackId`] address.
const PACK_ID_HEX_LEN: usize = mtxdb::packfile::PACK_ID_LEN * 2;
/// Hex digits in the truncated filename prefix of a pack address.
const PACK_FILENAME_PREFIX_HEX: usize = mtxdb::packfile::PACK_FILENAME_PREFIX_HEX;

/// Parse an operator-facing pack identifier into a selector.
fn parse_pack_id_selector(selector: &str) -> anyhow::Result<PackSelector> {
    PackSelector::parse(selector)
}

/// Common options shared by `scan_pack` and `cmd_scan_collection`.
#[allow(
    clippy::struct_excessive_bools,
    reason = "CLI scan flags map 1:1 to bools"
)]
struct ScanOptions {
    verbose: bool,
    header: bool,
    decode: Option<String>,
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
    clippy::too_many_lines,
    clippy::fn_params_excessive_bools,
    reason = "maps 1:1 to CLI args before building ScanOptions"
)]
fn cmd_scan_coalesced(
    cli: &Cli,
    selector: &str,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
    limit: i64,
    id: Option<&str>,
    collection: Option<&str>,
    raw: bool,
    sort: Option<&str>,
    reverse: bool,
    live: bool,
) -> anyhow::Result<()> {
    if live {
        bail!("--live is not supported with --coalesce");
    }
    let target = classify_info_selector(cli, selector)?;
    let valid_dirs = valid_database_dirs(cli)?;
    let shard_types: Vec<ShardType> = if let Some(st) = cli.shard_type {
        vec![st]
    } else {
        cli.shard_types().collect()
    };

    let mut matching_dirs = Vec::new();
    match target {
        InfoTarget::Collection => {
            let collection_id = parse_collection_selector(selector)?;
            for db_dir in &valid_dirs {
                let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                    continue;
                };
                let mut found_in_db = false;
                for &shard_type in &shard_types {
                    let Ok(dir) = pool_dir(&layout, shard_type) else {
                        continue;
                    };
                    let has = if let Some(summaries) =
                        PackfileStorage::collection_summaries_from_disk(&dir)
                    {
                        summaries.iter().any(|(id, _, _, _)| id == &collection_id)
                    } else if let Ok(store) = open_pool_read_only(dir) {
                        store.collection_index_info(&collection_id).is_some()
                    } else {
                        false
                    };
                    if has {
                        found_in_db = true;
                        break;
                    }
                }
                if found_in_db {
                    matching_dirs.push(db_dir.clone());
                }
            }
        }
        InfoTarget::Pack => {
            let selector = parse_pack_id_selector(selector)?;
            for db_dir in &valid_dirs {
                let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
                    continue;
                };
                let mut found_in_db = false;
                for &shard_type in &shard_types {
                    let Ok(dir) = pool_dir(&layout, shard_type) else {
                        continue;
                    };
                    if let Ok(files) = glob_pack_files(&dir) {
                        if files.iter().any(|&(id, _, _)| selector.matches(&id)) {
                            found_in_db = true;
                            break;
                        }
                    }
                }
                if found_in_db {
                    matching_dirs.push(db_dir.clone());
                }
            }
        }
    }

    if matching_dirs.is_empty() {
        bail!(
            "no matching {} `{selector}` found across {} database(s)",
            match target {
                InfoTarget::Collection => "collection",
                InfoTarget::Pack => "pack",
            },
            valid_dirs.len()
        );
    }

    if matching_dirs.len() == 1 {
        let sub_cli = cli.with_dir(matching_dirs[0].clone());
        println!("database: {}", matching_dirs[0].display());
        cmd_scan_single(
            &sub_cli, selector, verbose, header, decode, limit, id, collection, raw, sort, reverse,
            live,
        )?;
    } else {
        for (i, db_dir) in matching_dirs.iter().enumerate() {
            if i > 0 {
                println!();
            }
            println!("=== Database: {} ===", db_dir.display());
            let sub_cli = cli.with_dir(db_dir.clone());
            if let Err(e) = cmd_scan_single(
                &sub_cli, selector, verbose, header, decode, limit, id, collection, raw, sort,
                reverse, live,
            ) {
                eprintln!("error in `{}`: {e:#}", db_dir.display());
            }
        }
        println!();
        println!(
            "note: matched in {} of {} database(s)",
            matching_dirs.len(),
            valid_dirs.len()
        );
    }
    Ok(())
}

struct SearchTemplate {
    fields: Vec<String>,
    identity: Option<String>,
}

fn search_template(path: Option<&Path>) -> anyhow::Result<SearchTemplate> {
    let Some(path) = path else {
        return Ok(SearchTemplate {
            fields: vec![
                "/type".to_owned(),
                "/sender".to_owned(),
                "/content/body".to_owned(),
                "/content/name".to_owned(),
                "/content/topic".to_owned(),
                "/content/displayname".to_owned(),
            ],
            identity: Some("/event_id".to_owned()),
        });
    };
    let bytes = fs::read(path).with_context(|| format!("reading template {}", path.display()))?;
    let value = OwnedValue::parse_bytes(&bytes)
        .with_context(|| format!("template {} is not valid JSON", path.display()))?;
    let fields = match value.get("search").and_then(|search| search.get("fields")) {
        Some(OwnedValue::Array(fields)) => fields
            .iter()
            .map(|field| match field {
                OwnedValue::String(pointer) if valid_json_pointer(pointer) => Ok(pointer.clone()),
                other => bail!(
                    "template {} search.fields entries must be valid JSON pointers; got {other:?}",
                    path.display()
                ),
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        _ => bail!(
            "template {} must declare search.fields as an array of JSON pointers",
            path.display()
        ),
    };
    if fields.is_empty() {
        bail!(
            "template {} search.fields must not be empty",
            path.display()
        );
    }
    let identity =
        template_string_at(&value, &["record", "identity", "extract", "path"]).map(str::to_owned);
    Ok(SearchTemplate { fields, identity })
}

fn valid_json_pointer(pointer: &str) -> bool {
    if pointer.is_empty() {
        return true;
    }
    let Some(rest) = pointer.strip_prefix('/') else {
        return false;
    };
    for segment in rest.split('/') {
        let mut chars = segment.chars();
        while let Some(character) = chars.next() {
            if character == '~' && !matches!(chars.next(), Some('0' | '1')) {
                return false;
            }
        }
    }
    true
}

fn search_record_matches(record: &OwnedValue, fields: &[String], term: &str) -> bool {
    let needle = term.to_lowercase();
    fields.iter().any(|pointer| {
        extract_pointer_string(record, pointer)
            .is_some_and(|value| value.to_lowercase().contains(&needle))
    })
}

fn cmd_search(
    cli: &Cli,
    collection: &str,
    template_path: Option<&Path>,
    term: &str,
    format: &str,
    limit: i64,
) -> anyhow::Result<()> {
    run_multi_dir(cli, |sub_cli| {
        cmd_search_single(sub_cli, collection, template_path, term, format, limit)
    })
}

fn cmd_search_single(
    cli: &Cli,
    collection: &str,
    template_path: Option<&Path>,
    term: &str,
    format: &str,
    limit: i64,
) -> anyhow::Result<()> {
    if term.is_empty() {
        bail!("search term must not be empty");
    }
    let spec = search_template(template_path)?;
    let collection_id = parse_collection_selector(collection)?;
    let layout = open_layout(cli)?;
    let store = open_pool_read_only(pool_dir(&layout, ShardType::EventDag)?)
        .context("failed to open EventDag store")?;
    let max_results = scan_limit(limit);
    let mut results = 0usize;
    for entry in store.scan_collection(&collection_id)? {
        let (node_id, data) = entry?;
        if node_id == mtxdb::COLLECTION_METADATA_RECORD_ID {
            continue;
        }
        let record = OwnedValue::parse_bytes(&data.bytes)
            .with_context(|| format!("record {} is not valid JSON", hex::encode(node_id)))?;
        if !search_record_matches(&record, &spec.fields, term) {
            continue;
        }
        // Records carry no `event_id` field, so the Matrix event ID comes from the
        // event locator, as `timeline` does; the node ID is the last resort.
        let id = spec
            .identity
            .as_deref()
            .and_then(|pointer| extract_pointer_string(&record, pointer))
            .map(str::to_owned)
            .or_else(|| {
                let locator_col = mtxdb::interop::event_locator_collection_id(&node_id);
                let located = store.get(&locator_col, &node_id).ok().flatten()?;
                let (eid, _) = mtxdb::interop::decode_event_locator(&located.bytes).ok()?;
                std::str::from_utf8(eid).ok().map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| format_id(&node_id));
        if format == "text" && results != 0 {
            println!();
        }
        match format {
            "json" => {
                let mut view = record.clone();
                if let Some(object) = view.as_object_mut() {
                    object
                        .entry("event_id".to_owned())
                        .or_insert_with(|| OwnedValue::String(id.clone()));
                }
                println!("{view}");
            }
            "text" => {
                let sender = extract_pointer_string(&record, "/sender").unwrap_or("?");
                let kind = extract_pointer_string(&record, "/type").unwrap_or("?");
                let when = record
                    .get("origin_server_ts")
                    .and_then(OwnedValue::as_i64)
                    .map_or_else(|| "-".to_owned(), format_matrix_timestamp);
                println!("{when} {sender} {kind} {id}");
                if let Some(body) = ["/content/body", "/content/name", "/content/topic"]
                    .iter()
                    .find_map(|pointer| extract_pointer_string(&record, pointer))
                {
                    println!("  {body}");
                }
            }
            _ => println!("{id}"),
        }
        results = results.saturating_add(1);
        if max_results != 0 && results >= max_results {
            break;
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    reason = "maps 1:1 to CLI args before building ScanOptions"
)]
fn cmd_scan(
    cli: &Cli,
    selector: &str,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
    limit: i64,
    id: Option<&str>,
    collection: Option<&str>,
    raw: bool,
    sort: Option<&str>,
    reverse: bool,
    live: bool,
) -> anyhow::Result<()> {
    if cli.coalesce {
        return cmd_scan_coalesced(
            cli, selector, verbose, header, decode, limit, id, collection, raw, sort, reverse, live,
        );
    }
    run_multi_dir(cli, |sub_cli| {
        cmd_scan_single(
            sub_cli, selector, verbose, header, decode, limit, id, collection, raw, sort, reverse,
            live,
        )
    })
}

#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    reason = "maps 1:1 to CLI args before building ScanOptions"
)]
fn cmd_scan_single(
    cli: &Cli,
    selector: &str,
    verbose: bool,
    header: bool,
    decode: Option<&str>,
    limit: i64,
    id: Option<&str>,
    collection: Option<&str>,
    raw: bool,
    sort: Option<&str>,
    reverse: bool,
    live: bool,
) -> anyhow::Result<()> {
    let sort_column = sort.map(SortColumn::from_str).transpose()?;
    if sort_column == Some(SortColumn::Payload) && (!verbose && decode.is_none() || raw) {
        bail!("scan sorting requires --verbose or --decode and cannot be combined with --raw");
    }
    let node_id = id.map(parse_node_id).transpose()?;
    let collection_filter = collection.map(parse_collection_selector).transpose()?;
    let opts = ScanOptions {
        verbose,
        header,
        decode: decode.map(str::to_owned),
        limit,
        node_id,
        raw,
        sort: sort_column,
        reverse,
    };
    // `info` and `scan` share `classify_info_selector`, so the same selector
    // works for both and a malformed one gets the same explanation.
    let looks_like_pack_id = classify_info_selector(cli, selector)? == InfoTarget::Pack;
    if !looks_like_pack_id {
        if collection_filter.is_some() {
            bail!("--collection is only valid when scanning a pack ID; the selector already identifies the collection");
        }
        if live {
            return cmd_scan_collection_live(cli, selector, &opts);
        }
        return cmd_scan_collection(cli, selector, &opts);
    }
    if live {
        bail!("--live requires a collection selector, not a pack ID");
    }
    let selector = parse_pack_id_selector(selector)?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let Some(pool_dir) = existing_pool_dir(&db_layout, shard_type)? else {
                continue;
            };
            let Ok(pool) = ShardPool::open_read_only(pool_dir) else {
                continue;
            };
            let shards = pool.all_shards();
            let ids: Vec<PackId> = shards.iter().map(|(_, shard)| shard.pack_id).collect();
            if !ids.iter().any(|id| selector.matches(id)) {
                continue;
            }
            let pack_id = selector.resolve(ids.iter())?;
            if let Some((_, shard)) = shards
                .into_iter()
                .find(|(_, shard)| shard.pack_id == pack_id)
            {
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
            eprintln!("pack {selector}: not found");
        }
        return Ok(());
    }
    let shard_type = cli.require_shard_type()?;
    let pool_dir = selected_pool_dir(cli)?;
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let shards = pool.all_shards();
    let ids: Vec<PackId> = shards.iter().map(|(_, shard)| shard.pack_id).collect();
    let pack_id = selector.resolve(ids.iter())?;
    let shard = shards
        .into_iter()
        .find(|(_, shard)| shard.pack_id == pack_id)
        .map(|(_, shard)| shard)
        .with_context(|| format!("pack {selector} not found"))?;
    scan_pack(cli, &shard, pack_id, collection_filter, &opts, shard_type)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the pack scan helper receives the explicit scan filters and output options"
)]
fn scan_pack(
    _cli: &Cli,
    shard: &std::sync::Arc<mtxdb::shard::Shard>,
    pack_id: PackId,
    collection_filter: Option<[u8; 16]>,
    opts: &ScanOptions,
    shard_type: ShardType,
) -> anyhow::Result<()> {
    let path = &shard.path;
    let max_rows = opts.max_rows();
    let mut records = Vec::new();
    let mut matched_records = 0usize;
    let mut truncated = false;
    let verify_payload = opts.verbose || opts.header || opts.decode.is_some() || opts.raw;
    for record in mtxdb::packfile::scan_packfile_iter(path, verify_payload)? {
        let (record_collection, record_id, offset) = record?;
        // TODO: tied to MSRV 1.81.0 — replace with .is_none_or() once the
        // minimum is bumped to 1.82+.
        if collection_filter.is_some_and(|wanted| record_collection != wanted)
            || opts.node_id.is_some_and(|wanted| record_id != wanted)
        {
            continue;
        }
        matched_records = matched_records.saturating_add(1);
        if opts.is_sorting() || opts.reverse || records.len() < max_rows {
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
        truncated = records.len() > max_rows;
    }
    if opts.reverse {
        records.reverse();
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
                    "pack {pack_id}: raw frame @ {offset} ({} bytes, checksum verified)",
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
            "pack {pack_id}: {} bytes, showing first {max_rows} matching records (at least {})",
            std::fs::metadata(path)?.len(),
            matched_records,
        );
    } else {
        println!(
            "pack {pack_id}: {} bytes, {} records",
            std::fs::metadata(path)?.len(),
            matched_records,
        );
    }
    print_scan_table_header("COLLECTION", scan_payload_label(shard_type));
    for (collection_id, node_id, offset) in records.iter().take(max_rows) {
        let collection_hex = format_id(collection_id);
        let id_hex = format_id(node_id);
        let should_decode = opts.decode.is_some() || (opts.verbose && !opts.header);
        // State scans render the event -> state-group mapping inline, so they
        // read frames even without `--verbose`/`--decode`; other shards keep
        // the read-free default.
        let need_record = should_decode || opts.header || shard_type == ShardType::State;
        let data = need_record
            .then(|| ShardPool::read_at_committed(shard, *offset, true))
            .transpose()?;
        let payload = data
            .as_ref()
            .map(|data| scan_payload_cell(&data.data, should_decode));
        println!(
            "{}",
            scan_table_row(&collection_hex, &id_hex, *offset, payload.as_deref())
        );
        if opts.header {
            if let Some(record) = &data {
                print_scan_record_header(record, collection_id);
            }
        }
        if should_decode {
            if let Some(data) = data.filter(|data| scan_payload_suffix(&data.data).is_none()) {
                print_scan_payload(&data.data, opts.decode.as_deref());
            }
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
/// node ID (hex). `payload` is `None` when the scan mode does not request
/// payload data, so no frame payload was read.
fn scan_table_row(location: &str, id: &str, offset: u64, payload: Option<&str>) -> String {
    format!(
        "  {location:<34} {id:<34} {offset:>10}  {}",
        payload.unwrap_or("-")
    )
}

/// The PAYLOAD cell for one row: a decodable payload prints below the row
/// instead (its formatted form spans lines), so the cell just says so.
fn scan_payload_cell(data: &[u8], decodes_below: bool) -> String {
    scan_payload_suffix(data).unwrap_or_else(|| {
        if decodes_below {
            format!("{} bytes (decoded below)", data.len())
        } else {
            format!("{} bytes", data.len())
        }
    })
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
        mode: CollectionScanMode::new(
            opts.verbose,
            opts.raw,
            opts.sort,
            opts.header,
            opts.decode.is_some(),
        ),
        max_rows,
        frames: 0,
        raw_matches: Vec::new(),
        header_printed: false,
        shard_type,
        sorted_records: Vec::new(),
        reverse: opts.reverse,
        show_section_header,
        needs_section_spacing,
        header: opts.header,
        decode: opts.decode.clone(),
    };
    for (_, shard) in shards {
        if let Some(collection_packs) = &collection_packs {
            if !collection_packs.contains(&shard.pack_id) {
                continue;
            }
        }
        let matched_pack = scan_collection_shard(&shard, &mut context)?;
        packs = packs.saturating_add(usize::from(matched_pack));
        if bounded && !opts.is_sorting() && !opts.reverse && context.frames >= max_rows {
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
                    format_id(&collection_id),
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
    if context.mode.is_sorting() || context.reverse {
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
                format_id(&collection_id),
                if packs == 1 { "" } else { "s" },
            );
        } else {
            println!(
                "collection {}: {frames} physical record{} across {packs} pack{}",
                format_id(&collection_id),
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
            format_id(&collection_id),
            if packs == 1 { "" } else { "s" },
        );
    } else {
        println!(
            "collection {}: {frames} physical record{} across {packs} pack{}",
            format_id(&collection_id),
            if frames == 1 { "" } else { "s" },
            if packs == 1 { "" } else { "s" },
        );
        print_scan_limit_note(frames, max_rows);
    }
    Ok(frames)
}

/// Print a collection's live records from its index.
///
/// Unlike [`cmd_scan_collection`], this never walks frames that belong to
/// other collections and never sees superseded versions or tombstones. It is
/// the fast view for "what is in this collection" on a large packfile.
#[allow(
    clippy::too_many_lines,
    reason = "the live scan keeps its index resolution, ordering, and output modes in one place"
)]
/// One live frame: its shard, offset, node id, and the full record when the
/// output mode needs the payload (`None` for a header-only read).
type LiveScanRow = (
    Arc<mtxdb::shard::Shard>,
    u64,
    [u8; 16],
    Option<mtxdb::packfile::Record>,
);

#[allow(
    clippy::too_many_lines,
    reason = "the live scan resolves locations, reads rows, and renders each output mode in one pass"
)]
fn cmd_scan_collection_live(cli: &Cli, selector: &str, opts: &ScanOptions) -> anyhow::Result<()> {
    let (collection_id, rows, total_live) = cmd_scan_collection_live_rows(cli, selector, opts)?;
    render_live_scan_rows(collection_id, &rows, total_live, cli, opts)
}

fn cmd_scan_collection_live_rows(
    cli: &Cli,
    selector: &str,
    opts: &ScanOptions,
) -> anyhow::Result<([u8; 16], Vec<LiveScanRow>, usize)> {
    anyhow::ensure!(
        cli.shard_type.is_some(),
        "`scan --live` requires a specific shard type (-t state, -t events, or -t edges); \
         `-t all` cannot select a single live collection index"
    );
    let collection_id = parse_collection_selector(selector)?;
    let store = open_store_read_only(cli)?;
    let Some(locations) = store.collection_live_locations(&collection_id)? else {
        bail!(
            "collection {} not found{}",
            format_id(&collection_id),
            other_shard_type_hint(cli, &collection_id)
        );
    };
    let pool =
        ShardPool::open_read_only(selected_pool_dir(cli)?).context("failed to open shard store")?;
    let shards = pool.all_shards();
    let mut by_slot: HashMap<u16, Arc<mtxdb::shard::Shard>> = HashMap::new();
    for (slot, shard) in &shards {
        by_slot.insert(*slot, Arc::clone(shard));
    }

    // A node-id filter is a point lookup: ask the index for the winning frame
    // instead of enumerating the collection.
    let locations = if let Some(node_id) = opts.node_id {
        let Some((pack_id, offset)) = store.get_location(&collection_id, &node_id)? else {
            bail!(
                "node {} not found in collection {}",
                format_id(&node_id),
                format_id(&collection_id)
            );
        };
        let Some((slot, _)) = shards.iter().find(|(_, shard)| shard.pack_id == pack_id) else {
            bail!(
                "pack {} for node {} is not open",
                pack_id.filename_stem(),
                format_id(&node_id)
            );
        };
        vec![(*slot, offset)]
    } else {
        locations
    };

    let mut records: Vec<(Arc<mtxdb::shard::Shard>, u64)> = locations
        .into_iter()
        .filter_map(|(slot, offset)| by_slot.get(&slot).map(|shard| (Arc::clone(shard), offset)))
        .collect();
    records.sort_unstable_by_key(|(shard, offset)| (shard.pack_id, *offset));
    if opts.reverse {
        records.reverse();
    }
    let total_live = records.len();
    let max_rows = opts.max_rows();
    // The live index stores only `(slot, offset)`. A plain scan needs the
    // fixed frame identity to print each node ID, but it need not touch the
    // payload or checksum. Read whole records only for output modes that use
    // payload bytes or metadata.
    let read_payload = opts.raw
        || opts.verbose
        || opts.header
        || opts.decode.is_some()
        || opts.sort == Some(SortColumn::Payload);
    let mut rows: Vec<LiveScanRow> = Vec::with_capacity(if opts.is_sorting() {
        records.len()
    } else {
        records.len().min(max_rows)
    });
    for (shard, offset) in records {
        if !opts.is_sorting() && rows.len() >= max_rows {
            break;
        }
        if read_payload {
            let record = ShardPool::read_at_committed(&shard, offset, true)?;
            let removed = record
                .metadata
                .as_ref()
                .is_some_and(|metadata| metadata.removed);
            if record.collection_id != collection_id || removed {
                continue;
            }
            rows.push((shard, offset, record.hash, Some(record)));
        } else {
            let (record_collection, record_id, removed) =
                pool.record_identity_at(&shard, offset)?;
            if record_collection != collection_id || removed {
                continue;
            }
            rows.push((shard, offset, record_id, None));
        }
    }
    if opts.is_sorting() {
        if opts.sort == Some(SortColumn::Payload) {
            rows.sort_by(|a, b| {
                a.3.as_ref()
                    .expect("payload sort reads each record")
                    .data
                    .cmp(&b.3.as_ref().expect("payload sort reads each record").data)
            });
        } else {
            rows.sort_by_key(|(_, offset, _, _)| *offset);
        }
        if opts.reverse {
            rows.reverse();
        }
    }

    Ok((collection_id, rows, total_live))
}

fn render_live_scan_rows(
    collection_id: [u8; 16],
    rows: &[LiveScanRow],
    total_live: usize,
    cli: &Cli,
    opts: &ScanOptions,
) -> anyhow::Result<()> {
    let shard_type = cli.require_shard_type()?;
    let max_rows = opts.max_rows();
    let bounded = max_rows != usize::MAX;
    if opts.raw {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        for (_, offset, _, record) in rows.iter().take(max_rows) {
            out.write_all(&record.as_ref().expect("raw scan reads each record").data)?;
            if opts.verbose {
                eprintln!(
                    "collection {}: live raw frame @ {offset} ({} bytes, checksum verified)",
                    format_id(&collection_id),
                    record
                        .as_ref()
                        .expect("raw scan reads each record")
                        .data
                        .len()
                );
            }
        }
        out.flush()?;
        eprintln!(
            "collection {}: {total_live} live record{} (indexed view; superseded frames omitted)",
            format_id(&collection_id),
            if total_live == 1 { "" } else { "s" },
        );
        if bounded && total_live > max_rows {
            eprintln!("note: showing first {max_rows}; use -l 0 to show all");
        }
        return Ok(());
    }

    let mut context = CollectionScanContext {
        collection_id,
        node_id: opts.node_id,
        mode: CollectionScanMode::new(
            opts.verbose,
            opts.raw,
            opts.sort,
            opts.header,
            opts.decode.is_some(),
        ),
        max_rows,
        frames: 0,
        raw_matches: Vec::new(),
        header_printed: false,
        shard_type,
        sorted_records: Vec::new(),
        reverse: opts.reverse,
        show_section_header: false,
        needs_section_spacing: false,
        header: opts.header,
        decode: opts.decode.clone(),
    };
    for (shard, offset, record_id, record) in rows.iter().take(max_rows) {
        context.frames = context.frames.saturating_add(1);
        print_collection_record_with_data(
            shard,
            *record_id,
            *offset,
            record.as_ref(),
            &mut context,
        );
    }
    let frames = context.frames;
    if bounded && total_live > max_rows {
        println!(
            "collection {}: showing first {frames} live records (indexed view; superseded frames omitted)",
            format_id(&collection_id),
        );
    } else {
        println!(
            "collection {}: {total_live} live record{} (indexed view; superseded frames omitted)",
            format_id(&collection_id),
            if total_live == 1 { "" } else { "s" },
        );
        print_scan_limit_note(total_live, max_rows);
    }
    Ok(())
}

/// Print every physical frame for a collection across all packs. This is a
/// diagnostic scan, so superseded copies are deliberately retained in the
/// output; use `export` to enumerate only the collection's live records.
fn cmd_scan_collection(cli: &Cli, selector: &str, opts: &ScanOptions) -> anyhow::Result<()> {
    let collection_id = parse_collection_selector(selector)?;
    if cli.shard_type.is_none() {
        let db_layout = open_layout(cli)?;
        let mut matched_any = false;
        for shard_type in cli.shard_types() {
            let Some(pool_dir) = existing_pool_dir(&db_layout, shard_type)? else {
                continue;
            };
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
    }
    if context.reverse {
        if context.mode.is_raw() {
            context.raw_matches.reverse();
        } else {
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
    header: bool,
    decode: Option<String>,
}

#[derive(Clone, Copy)]
enum CollectionScanMode {
    Plain,
    Verbose { sort: Option<SortColumn> },
    Raw { verbose: bool },
}

impl CollectionScanMode {
    #[allow(clippy::fn_params_excessive_bools)]
    fn new(verbose: bool, raw: bool, sort: Option<SortColumn>, header: bool, decode: bool) -> Self {
        if raw {
            Self::Raw { verbose }
        } else if verbose || header || decode {
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
        let id_matches = context.node_id.is_none_or(|wanted| record_id == wanted);
        if record_collection_id != context.collection_id || !id_matches {
            continue;
        }
        matched_pack = true;
        context.frames = context.frames.saturating_add(1);
        if context.mode.is_raw() {
            context.raw_matches.push((shard.clone(), record_id, offset));
        } else if context.mode.is_sorting() || context.reverse {
            context
                .sorted_records
                .push((shard.clone(), record_id, offset));
        } else if context.frames <= context.max_rows {
            print_collection_record(shard, record_id, offset, context)?;
        }
        if context.max_rows != usize::MAX
            && !context.mode.is_sorting()
            && !context.reverse
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
    let should_decode = context.decode.is_some() || (context.mode.verbose() && !context.header);
    // State scans show the event -> state-group pointer without an explicit
    // verbose/decode request, matching the pack scan.
    let need_record = should_decode || context.header || context.shard_type == ShardType::State;
    let data = need_record
        .then(|| ShardPool::read_at_committed(shard, offset, true))
        .transpose()?;
    print_collection_record_with_data(shard, record_id, offset, data.as_ref(), context);
    Ok(())
}

fn print_collection_record_with_data(
    shard: &std::sync::Arc<mtxdb::shard::Shard>,
    record_id: [u8; 16],
    offset: u64,
    data: Option<&mtxdb::packfile::Record>,
    context: &mut CollectionScanContext,
) {
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
    let should_decode = context.decode.is_some() || (context.mode.verbose() && !context.header);
    let payload = data.map(|data| {
        if record_id == mtxdb::COLLECTION_METADATA_RECORD_ID {
            "collection metadata".to_owned()
        } else {
            scan_payload_cell(&data.data, should_decode)
        }
    });
    println!(
        "{}",
        scan_table_row(
            &shard.pack_id.filename_stem(),
            &format_id(&record_id),
            offset,
            payload.as_deref(),
        )
    );
    if context.header {
        if let Some(record) = data {
            print_scan_record_header(record, &context.collection_id);
        }
    }
    if should_decode {
        if let Some(data) = data.filter(|data| scan_payload_suffix(&data.data).is_none()) {
            print_scan_payload(&data.data, context.decode.as_deref());
        }
    }
}

fn print_scan_record_header(record: &mtxdb::packfile::Record, collection_id: &[u8; 16]) {
    if *collection_id == mtxdb::COLLECTION_METADATA_RECORD_ID {
        if let Some(meta) = CollectionMetadata::decode(&record.data) {
            let pool = meta.member_namespace.as_ref().map_or_else(
                || "none".to_owned(),
                |d| String::from_utf8_lossy(d).into_owned(),
            );
            let role = meta.role.as_ref().map_or_else(
                || "unspecified".to_owned(),
                std::string::ToString::to_string,
            );
            let schema = meta.schema.as_deref().unwrap_or("none");
            let verified = meta.verify_collection_id(collection_id);
            let verify_str = if verified {
                "valid"
            } else {
                "MISMATCH / CORRUPT"
            };
            println!(
                "    [header] genesis metadata: pool {pool}, role {role}, schema {schema}, canonical id {} ({verify_str})",
                String::from_utf8_lossy(&meta.collection_canonical_id)
            );
        } else {
            println!("    [header] genesis metadata (corrupt TLV block)");
        }
    } else if let Some(meta) = &record.metadata {
        let mut details = Vec::new();
        if let Some(logical_id) = &meta.logical_id {
            details.push(format!("logical_id: 0x{}", hex::encode(logical_id)));
        }
        if let Some(content_digest) = &meta.content_digest {
            details.push(format!("digest: 0x{}", hex::encode(content_digest)));
        }
        if let Some(role) = &meta.role {
            details.push(format!("role: {}", String::from_utf8_lossy(role)));
        }
        if details.is_empty() {
            println!("    [header] frame metadata block present (no recognized fields)");
        } else {
            println!("    [header] frame metadata: {}", details.join(", "));
        }
    } else {
        println!("    [header] standard frame (no metadata block)");
    }
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
fn print_scan_payload(data: &[u8], decode: Option<&str>) {
    let rendered = if let Some(mode) = decode {
        match mode {
            "json" => pretty_json_stream(data),
            "hamt" => decode_hamt_root(data)
                .or_else(|| decode_hamt_node(data))
                .or_else(|| decode_state_group(data)),
            "state" => decode_hamt_root(data)
                .or_else(|| decode_hamt_node(data))
                .or_else(|| decode_state_group(data)),
            "raw" => Some(hex_bytes(data).into_bytes()),
            _ => pretty_print_payload(data),
        }
    } else {
        pretty_print_payload(data)
    };
    if let Some(pretty) = rendered {
        for line in String::from_utf8_lossy(&pretty).lines() {
            println!("    {line}");
        }
    } else {
        println!("    {}", hex_bytes(data));
    }
}

/// Return an inline summary for a payload `pretty_print_payload` can't
/// decode; a decodable payload (plain JSON, or a recognized record shape
/// like `event_json`) is printed below its record instead because its
/// formatted representation spans lines.
fn scan_payload_suffix(data: &[u8]) -> Option<String> {
    if data.starts_with(b"AUX1") {
        // Auxiliary values retain their four-byte magic and full 32-byte
        // key digest on disk.  The scanner does not have the logical key
        // needed to verify that digest, but it can still render the value
        // portion instead of incorrectly calling the whole envelope opaque.
        const AUXILIARY_VALUE_OFFSET: usize = 4 + 32;
        if data.len() < AUXILIARY_VALUE_OFFSET {
            return Some(format!("{} bytes (malformed AUX1 envelope)", data.len()));
        }
        let value = &data[AUXILIARY_VALUE_OFFSET..];
        // A 16-byte auxiliary value is an `event -> STGP instance`
        // pointer; render it in the `PTR:` form.
        if value.len() == STATE_GROUP_ID_LENGTH {
            return Some(format!("PTR: {}", format_id(value)));
        }
        let rendered = std::str::from_utf8(value)
            .ok()
            .filter(|text| text.bytes().all(|byte| !byte.is_ascii_control()))
            .map_or_else(|| hex_bytes(value), ToOwned::to_owned);
        return Some(format!("AUX1 value: {rendered}"));
    }
    if pretty_print_payload(data).is_some() {
        return None;
    }
    Some(match data.len() {
        0 => "0 bytes [empty payload: a removal marker or an empty value; see `mtxdb tombstones`]"
            .to_owned(),
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

#[allow(
    clippy::too_many_lines,
    reason = "the import command owns setup, the data pass, and the audit sidecars"
)]
fn cmd_import(
    cli: &Cli,
    paths: &[std::path::PathBuf],
    collection_override: Option<&str>,
    template: Option<&Path>,
    no_state_hamts: bool,
) -> anyhow::Result<()> {
    let import_started = Instant::now();
    let setup_started = Instant::now();
    let import_template = compile_import_template(template)?;
    // One writer owns the entire batch: it prevents another writer from
    // interleaving halfway through a shell glob, and lets us publish one
    // complete shard/collection snapshot once the final input has been handled.
    let pool_dir = selected_pool_dir(cli)?;
    // Admission is based on the actual pack contents rather than a sidecar:
    // a stale summary must not make an unestablished room look established.
    let mut established_collections = matrix_create_collections_on_disk(&pool_dir)?;
    // One `Database` owns every pool for the whole import, so the event, state and
    // edges pools share one write-ahead log and the root's single-writer lock: no
    // other writer can interleave halfway through a shell glob, and the per-room
    // auth adjacency is recorded through the same database as the records it
    // describes. Every pool the import touches is synced once at the end.
    let layout = open_layout(cli)?;
    let shard = cli.require_shard_type()?;
    let db = open_database(layout.root().to_path_buf()).with_context(|| {
        format!(
            "failed to open the database at `{}` for writing",
            layout.root().display()
        )
    })?;
    let store: &PackfileStorage = db.pool(shard);
    store.set_read_plan_policy(cli.read_plan);
    // State-group mappings belong in the State pool, not the event pool.
    let state_store: &PackfileStorage = db.pool(ShardType::State);
    state_store.set_read_plan_policy(cli.read_plan);
    if no_state_hamts {
        eprintln!("state HAMTs: skipped (--no-state-hamts)");
    }
    report_import_phase("setup", setup_started.elapsed());
    let data_started = Instant::now();
    let mut run = import_data(
        &db,
        store,
        state_store,
        paths,
        collection_override,
        &import_template,
        &mut established_collections,
        !no_state_hamts,
    );
    report_import_phase(
        "data pass (inputs and key reconciliation)",
        data_started.elapsed(),
    );
    let audit_started = Instant::now();
    let hash_audit_path = layout.root().join("content-hash-mismatches.jsonl");
    if let Err(error) = write_hash_audit(&hash_audit_path, &run.hash_audit) {
        eprintln!("writing {}: {error:#}", hash_audit_path.display());
        run.failures = run.failures.saturating_add(1);
        run.all_exports = false;
    } else {
        let differs = run
            .hash_audit
            .iter()
            .filter(|row| !row.content_matches_redaction)
            .count();
        eprintln!(
            "content-hash audit: {differs} content-differing events; details for {} total mismatches in {}",
            run.hash_audit.len(),
            hash_audit_path.display()
        );
    }
    let signature_audit_path = layout.root().join("signature-findings.jsonl");
    if let Err(error) = write_signature_audit(&signature_audit_path, &run.signature_audit) {
        eprintln!("writing {}: {error:#}", signature_audit_path.display());
        run.failures = run.failures.saturating_add(1);
        run.all_exports = false;
    } else {
        eprintln!(
            "signature audit: {} events did not verify; details in {}",
            run.signature_audit.len(),
            signature_audit_path.display()
        );
    }
    eprintln!(
        "signature late: {} events verified after their key's cached valid_until_ts",
        run.signature_verified_late
    );
    let rejected_total = run
        .rejection_counts
        .values()
        .copied()
        .fold(0_usize, usize::saturating_add);
    eprintln!("derived rejections: {rejected_total} events");
    for (code, count) in run.rejection_counts.iter().filter(|(_, count)| **count > 0) {
        eprintln!("  {count:>8}  {}", RejectionCode::from_u8(*code).tag());
    }
    report_import_phase("audit sidecars and summary", audit_started.elapsed());
    let (failures, server_keys_imported) = (run.failures, run.server_keys);
    let sync_started = Instant::now();
    store
        .sync_all()
        .context("persisting import shard and collection summaries")?;
    report_import_phase("primary pool sync", sync_started.elapsed());
    if shard != ShardType::State {
        let sync_started = Instant::now();
        state_store
            .sync_all()
            .context("persisting state-group summaries")?;
        report_import_phase("state pool sync", sync_started.elapsed());
    }
    if server_keys_imported {
        let sync_started = Instant::now();
        db.pool(ShardType::ServerInfo)
            .sync_all()
            .context("persisting imported signing keys")?;
        report_import_phase("server-info pool sync", sync_started.elapsed());
    }
    let finish_started = Instant::now();
    finish_import(&db, &run)?;
    report_import_phase("finish marker", finish_started.elapsed());
    if import_timings_enabled() {
        eprintln!("import: wall time {:?}", import_started.elapsed());
    }
    if failures != 0 {
        bail!("import completed with {failures} failed input file(s)");
    }
    Ok(())
}

/// What the inputs of one `mtxdb import` run amounted to.
#[derive(Debug)]
struct ImportRun {
    /// Inputs that failed.
    failures: usize,
    /// A keys export wrote into the server-info pool.
    server_keys: bool,
    /// Event records written through mtxdb exports.
    exported_events: u64,
    /// Auth closures written through mtxdb exports.
    exported_closures: u64,
    /// Every input was an mtxdb export that imported cleanly. The marker
    /// vouches only for such a run: anything else leaves records the server
    /// cannot read, or data that did not all arrive.
    all_exports: bool,
    /// Whether usable signing-key bytes were present after key inputs were
    /// processed, including keys already stored by an earlier import.
    signature_keys_available: bool,
    /// Keys imported from this run's keys export, when present.
    signature_catalog: Option<export_import::SignatureKeyCatalog>,
    /// Content-hash mismatches captured with the same redaction used to store
    /// each event and written as a JSONL sidecar after the input pass.
    hash_audit: Vec<export_import::HashAudit>,
    /// Events whose signature did not verify, written as a JSONL sidecar.
    signature_audit: Vec<export_import::SignatureAudit>,
    /// Newly imported bindings whose candidate timestamps must be reconciled.
    imported_key_bindings: HashSet<export_import::KeyBinding>,
    /// Verified time ranges observed across this run's room files.
    verified_keys: HashMap<export_import::KeyBinding, (u64, u64)>,
    /// Derived rejections by stored rejection code, across the run.
    rejection_counts: std::collections::BTreeMap<u8, usize>,
    /// Events with no usable in-window key for their signer.
    signature_no_key: usize,
    /// Events without the material needed to attempt signature verification.
    signature_no_material: usize,
    /// Events rejected because a usable key had a bad signature.
    signature_invalid: usize,
    /// Events that verified after their key's cached `valid_until_ts`.
    signature_verified_late: usize,
}

fn absorb_outcome(run: &mut ImportRun, outcome: ImportOutcome) {
    run.server_keys = run.server_keys || outcome.server_keys;
    if let Some(catalog) = outcome.signature_catalog {
        if let Some(existing) = &mut run.signature_catalog {
            existing.extend(catalog);
        } else {
            run.signature_catalog = Some(catalog);
        }
    }
    run.hash_audit.extend(outcome.hash_audit);
    run.signature_audit.extend(outcome.signature_audit);
    for (code, count) in outcome.rejection_counts {
        let total = run.rejection_counts.entry(code).or_insert(0);
        *total = total.saturating_add(count);
    }
    run.signature_no_key = run
        .signature_no_key
        .saturating_add(outcome.signature_no_key);
    run.signature_no_material = run
        .signature_no_material
        .saturating_add(outcome.signature_no_material);
    run.signature_invalid = run
        .signature_invalid
        .saturating_add(outcome.signature_invalid);
    run.signature_verified_late = run
        .signature_verified_late
        .saturating_add(outcome.signature_verified_late);
    run.imported_key_bindings.extend(outcome.key_bindings);
    for observation in outcome.verified_keys {
        run.verified_keys
            .entry(observation.binding)
            .and_modify(|(first, last)| {
                *first = (*first).min(observation.first_seen_ms);
                *last = (*last).max(observation.last_active_ms);
            })
            .or_insert((observation.first_seen_ms, observation.last_active_ms));
    }
    match outcome.exported {
        Some(counts) => {
            run.exported_events = run.exported_events.saturating_add(counts.events);
            run.exported_closures = run.exported_closures.saturating_add(counts.closures);
        }
        None => run.all_exports = false,
    }
}

/// Replace the run's content-hash audit sidecar, including an empty file when
/// this run had no computable mismatches so stale rows cannot survive.
fn write_hash_audit(path: &Path, rows: &[export_import::HashAudit]) -> anyhow::Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        writeln!(writer, "{}", row.json_line())
            .with_context(|| format!("writing {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", path.display()))
}

/// Replace the run's signature sidecar, including an empty file when every
/// event verified so stale rows cannot survive.
fn write_signature_audit(
    path: &Path,
    rows: &[export_import::SignatureAudit],
) -> anyhow::Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        writeln!(writer, "{}", row.json_line())
            .with_context(|| format!("writing {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", path.display()))
}

/// `paths` with every server-keys export first, the rest after them, each
/// group in argument order.
fn keys_first(paths: &[PathBuf]) -> Vec<&PathBuf> {
    let (keys, rest): (Vec<_>, Vec<_>) = paths.iter().partition(|path| is_server_keys_export(path));
    keys.into_iter().chain(rest).collect()
}

/// The tagged format declared by an input's first nonblank line. Read only
/// that line so large room exports need not be loaded during input ordering.
/// A read or parse failure is left to the normal import pass, which reports it
/// with the input path.
fn input_export_format(path: &Path) -> Option<interchange::InputFormat> {
    const PROBE_BYTES: u64 = 64 * 1024;
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader
            .by_ref()
            .take(PROBE_BYTES)
            .read_until(b'\n', &mut line)
        {
            Ok(0) | Err(_) => return None,
            Ok(_) if line.iter().all(u8::is_ascii_whitespace) => {}
            Ok(_) => {
                // A line longer than the probe is only partially read. Do not
                // classify from a truncated JSON object: leave the format
                // undetermined so a long keys export is not mistaken for a
                // room/non-export input.
                if line.len() as u64 >= PROBE_BYTES && !line.ends_with(b"\n") {
                    return None;
                }
                return interchange::detect(&line);
            }
        }
    }
}

fn is_server_keys_export(path: &Path) -> bool {
    input_export_format(path) == Some(interchange::InputFormat::Keys)
}

/// Whether the server-info pool contains at least one signing key with public
/// key bytes. Notary payloads and evicted digest-only bindings do not enable
/// signature verification.
fn has_signing_key_material(store: &PackfileStorage) -> anyhow::Result<bool> {
    for (collection, entries, _, _) in store.collection_summaries() {
        if entries == 0 {
            continue;
        }
        for item in store.scan_collection(&collection)? {
            let (_, data) = item?;
            let Ok((origin, key)) = mtxdb::server_keys::decode_server_key_record(&data.bytes)
            else {
                continue;
            };
            if mtxdb::server_keys::origin_keys_collection_id(&origin) == collection
                && !key.key.is_empty()
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The data phase of a run: import every input. Server keys are imported
/// before room exports regardless of argument order, so later room verdict
/// derivation can consult the imported keys. Commits no marker, so a run that
/// stops after this leaves the database visibly incomplete.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the data pass shares every open pool and batch option with its file reader, and its loop carries the per-file timing probe"
)]
fn import_data(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    paths: &[std::path::PathBuf],
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
    write_state_hamts: bool,
) -> ImportRun {
    // Clear the marker before reading any input: an input that fails to parse
    // never reaches the importers that clear it, and must not leave the marker
    // of an earlier run vouching for a store this run may have changed.
    let marker_cleared = export_import::clear_marker(db);
    if let Err(error) = &marker_cleared {
        eprintln!("clearing the import marker: {error:#}");
    }
    let mut run = ImportRun {
        failures: usize::from(marker_cleared.is_err()),
        server_keys: false,
        exported_events: 0,
        exported_closures: 0,
        all_exports: !paths.is_empty() && marker_cleared.is_ok(),
        signature_keys_available: false,
        signature_catalog: None,
        hash_audit: Vec::new(),
        signature_audit: Vec::new(),
        imported_key_bindings: HashSet::new(),
        verified_keys: HashMap::new(),
        signature_no_key: 0,
        signature_no_material: 0,
        signature_invalid: 0,
        signature_verified_late: 0,
        rejection_counts: std::collections::BTreeMap::new(),
    };
    let mut key_state_checked = false;
    let mut key_import_failed = false;
    let mut signature_checks_enabled = false;
    let mut room_cache = ImportRunCache::default();
    for (index, path) in keys_first(paths).into_iter().enumerate() {
        if index != 0 {
            eprintln!();
        }
        if !key_state_checked && input_export_format(path) == Some(interchange::InputFormat::Room) {
            key_state_checked = true;
            let mut key_pool_ready = true;
            if run.server_keys {
                if let Err(error) = db.pool(ShardType::ServerInfo).sync_all() {
                    eprintln!(
                        "signature checks off: unable to persist imported signing keys: {error:#}"
                    );
                    run.failures = run.failures.saturating_add(1);
                    run.all_exports = false;
                    key_pool_ready = false;
                }
            }
            if key_pool_ready {
                if let Some(catalog) = &run.signature_catalog {
                    run.signature_keys_available = catalog.has_key_material();
                } else {
                    match has_signing_key_material(db.server_info()) {
                        Ok(available) => run.signature_keys_available = available,
                        Err(error) => {
                            eprintln!(
                                "signature checks off: unable to inspect the signing-key store: {error:#}"
                            );
                            run.failures = run.failures.saturating_add(1);
                            run.all_exports = false;
                        }
                    }
                }
            }
            if key_import_failed {
                eprintln!("signature checks off: signing keys failed to import");
            } else if !run.signature_keys_available {
                eprintln!("signature checks off: no signing keys loaded");
            }
            signature_checks_enabled =
                key_pool_ready && run.signature_keys_available && !key_import_failed;
        }
        let is_keys_input = input_export_format(path) == Some(interchange::InputFormat::Keys);
        let file_started = Instant::now();
        match cmd_import_file(
            db,
            store,
            state_store,
            path,
            collection_override,
            template,
            established_collections,
            signature_checks_enabled,
            run.signature_catalog.as_ref(),
            write_state_hamts,
            Some(&mut room_cache),
        ) {
            Ok(outcome) => absorb_outcome(&mut run, outcome),
            Err(error) => {
                eprintln!("{}: {error:#}", path.display());
                run.failures = run.failures.saturating_add(1);
                run.all_exports = false;
                if is_keys_input {
                    key_import_failed = true;
                }
            }
        }
        if std::env::var_os("MTXDB_DIRECT_DEBUG").is_some() || import_timings_enabled() {
            eprintln!(
                "import: {} took {:?}",
                path.display(),
                file_started.elapsed()
            );
        }
    }
    let reconciliation_started = Instant::now();
    reconcile_verified_key_observations(db, &mut run);
    report_import_phase(
        "verified-key reconciliation",
        reconciliation_started.elapsed(),
    );
    run
}

fn reconcile_verified_key_observations(db: &Database, run: &mut ImportRun) {
    let bindings: Vec<_> = run.imported_key_bindings.iter().collect();
    let updates: Vec<_> = bindings
        .iter()
        .map(|binding| {
            (
                binding.origin.as_slice(),
                binding.key_id.as_slice(),
                run.verified_keys.get(binding).copied(),
            )
        })
        .collect();
    match mtxdb::server_keys::set_verified_observations(db, &updates) {
        Ok(missing) => {
            for index in missing {
                let binding = bindings[index];
                eprintln!(
                    "updating verified signing-key metadata: {}/{} was not found",
                    String::from_utf8_lossy(&binding.origin),
                    String::from_utf8_lossy(&binding.key_id)
                );
                run.failures = run.failures.saturating_add(1);
                run.all_exports = false;
            }
        }
        Err(error) => {
            eprintln!("updating verified signing-key metadata: {error}");
            run.failures = run.failures.saturating_add(1);
            run.all_exports = false;
        }
    }
}

/// The final phase of a run: make the data durable, then commit the marker in
/// a transaction of its own. Does nothing unless the whole run was clean
/// mtxdb exports.
fn finish_import(db: &Database, run: &ImportRun) -> anyhow::Result<()> {
    if !run.all_exports {
        return Ok(());
    }
    export_import::sync_data(db)?;
    export_import::write_marker(
        db,
        export_import::ImportCounts {
            events: run.exported_events,
            closures: run.exported_closures,
        },
    )?;
    db.pool(ShardType::ServerInfo)
        .sync_all()
        .context("persisting the import marker")?;
    eprintln!(
        "import complete: {} events, {} auth closures",
        run.exported_events, run.exported_closures
    );
    let corroborated = run
        .verified_keys
        .keys()
        .filter(|binding| run.imported_key_bindings.contains(*binding))
        .count();
    eprintln!(
        "signature checks: {} events with no usable key; {} without signature material; {} invalid signatures; {} verified late (past their key's cached valid_until_ts); {} of {} keys corroborated (at least one in-window verifying signature)",
        run.signature_no_key,
        run.signature_no_material,
        run.signature_invalid,
        run.signature_verified_late,
        corroborated,
        run.imported_key_bindings.len()
    );
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
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: "/room_id".into(),
            member_namespace: MATRIX_ROOM_MEMBER_NAMESPACE,
            display_id_pointer: "/room_id".into(),
        },
        establishment: Some(EstablishmentRule {
            selector: "type == m.room.create && state_key == ''".into(),
        }),
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
        for label in labels {
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
    let bytes = fs::read(path).with_context(|| format!("reading template {}", path.display()))?;
    let template: OwnedValue = OwnedValue::parse_bytes(&bytes)
        .with_context(|| format!("template {} is not valid JSON", path.display()))?;
    let (identity_pointer, membership_pointer) = validate_required_keys(&template, path)?;
    // Every collection has exactly one establishment record; there is no
    // optional or multi-record cardinality. A template that cannot name one is
    // rejected rather than compiling to a collection with no genesis.
    let selector = template_string_at(&template, &["establishment", "selector"])
        .filter(|selector| !selector.is_empty())
        .with_context(|| {
            format!(
                "template {} must define a non-empty establishment.selector",
                path.display()
            )
        })?;
    let establishment = EstablishmentRule {
        selector: selector.to_owned(),
    };
    let node_id_algorithm = template_string_at(
        &template,
        ["record", "identity", "internal_key", "algorithm"].as_slice(),
    )
    .unwrap_or("blake3-128");
    let record_digest_algorithm = digest_algorithm_for(node_id_algorithm)
        .with_context(|| format!("template {} record identity internal_key", path.display()))?;
    // The collection-id derivation is a fixed, core-owned function of a pool
    // namespace discriminator rather than a template-selected digest algorithm.
    // The Matrix import profile targets the EventDag pool.
    let member_namespace = MATRIX_ROOM_MEMBER_NAMESPACE;
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
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: identity_pointer.to_owned(),
            },
            digest_algorithm: record_digest_algorithm,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: membership_pointer.to_owned(),
            member_namespace,
            display_id_pointer: display_id_pointer.to_owned(),
        },
        establishment: Some(establishment),
    })
}

/// Resolve a template's digest-algorithm name to the engine's [`DigestAlgorithm`].
fn digest_algorithm_for(algorithm: &str) -> anyhow::Result<DigestAlgorithm> {
    match algorithm {
        "sha2-256" => Ok(DigestAlgorithm::Sha256),
        "blake3-128" => Ok(DigestAlgorithm::Blake3),
        other => bail!("unsupported internal-key algorithm `{other}`"),
    }
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
/// using the named algorithm. Template compilation validates the algorithm
/// before records reach this path.
///
/// # Key width
///
/// The selected algorithm computes a full 32-byte digest, but only the first
/// 16 bytes are returned as the routing [`NodeId`].
fn derive_template_key(algorithm: &str, extracted: &str) -> anyhow::Result<[u8; 16]> {
    let digest_algorithm = digest_algorithm_for(algorithm)?;
    Ok(record_logical_id(&mtxdb::content_digest(
        digest_algorithm,
        extracted.as_bytes(),
    )))
}

/// Derive a node ID from one already-extracted record-identity value.
fn template_pointer_node_id(
    template: &CollectionTemplate,
    pointer: &str,
    value: &str,
) -> anyhow::Result<[u8; 16]> {
    let resolve = |req: &str| (req == pointer).then(|| value.as_bytes().to_vec());
    let input = FrameIdInput {
        payload: &[],
        descriptor: &[],
        canonical: None,
        resolve: &resolve,
    };
    let Some(digest) = frame_digest(
        &template.record_id_rule.policy,
        template.record_id_rule.digest_algorithm,
        &input,
    ) else {
        bail!("failed to compute frame digest for record identity value '{value}'");
    };
    Ok(record_logical_id(&digest))
}

/// Produce the node ID for an event ID string under the template's record-identity rule.
fn template_event_id_node_id(
    template: &CollectionTemplate,
    event_id: &str,
) -> anyhow::Result<[u8; 16]> {
    let FrameIdPolicy::Pointer { pointer } = &template.record_id_rule.policy else {
        bail!(
            "record identity policy {:?} is not supported by the raw importer",
            template.record_id_rule.policy
        );
    };
    if pointer != "/event_id" {
        bail!("raw Matrix import requires record identity pointer '/event_id', got '{pointer}'");
    }
    template_pointer_node_id(template, pointer, event_id)
}

/// Run a template's record-identity rule against one event, producing the
/// node ID that storage keys it by.
fn template_node_id(
    template: &CollectionTemplate,
    event: &OwnedValue,
) -> anyhow::Result<Option<[u8; 16]>> {
    let FrameIdPolicy::Pointer { pointer } = &template.record_id_rule.policy else {
        bail!(
            "record identity policy {:?} is not supported by the importer",
            template.record_id_rule.policy
        );
    };
    let Some(val) = extract_pointer_string(event, pointer) else {
        return Ok(None);
    };
    template_pointer_node_id(template, pointer, val).map(Some)
}

/// Run a template's collection-key rule against an already-extracted
/// membership value (e.g. a Matrix `room_id`), producing the collection ID.
fn template_collection_id(template: &CollectionTemplate, membership_value: &str) -> [u8; 16] {
    derive_collection_id(
        template.collection_key.member_namespace,
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

/// Output layout for `mtxdb export`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportFormat {
    /// One raw stored payload per line, re-importable by `mtxdb import`.
    Payload,
    /// One schema-versioned envelope per line, carrying frame metadata.
    Envelope,
}

/// Schema identifier stamped on every envelope line.
const EXPORT_ENVELOPE_SCHEMA: &str = "mtxdb.export/v1";

/// Export every live record in one collection as a JSONL stream. A read-only
/// frame scan avoids stale persisted collection summaries and rebuilding every
/// collection's in-memory index just to enumerate one collection. Scanning packs
/// in pack-ID order lets a later physical copy replace an older one with the same
/// node ID.
///
/// `--format jsonl` (default) is the re-importable interchange form.
/// `--format envelope` is an **inspection-only** representation: it carries
/// the exact stored bytes (`payload_base64`), a decoded JSON convenience
/// value, and frame metadata from the same physical frame. It enumerates only
/// pack-resident frames (like jsonl's ordering scan) and `mtxdb import` does
/// not accept it.
fn cmd_export(cli: &Cli, collection: &str, format: &str, metadata: bool) -> anyhow::Result<()> {
    let format = match (format, metadata) {
        ("jsonl", false) => ExportFormat::Payload,
        ("jsonl", true) | ("envelope", _) => ExportFormat::Envelope,
        (other, _) => bail!("unsupported export format `{other}`; choose jsonl or envelope"),
    };
    let collection_id = parse_collection_selector(collection)?;
    let pool_dir = selected_pool_dir(cli)?;
    let store = open_store_read_only(cli)?;
    if store.collection_index_info(&collection_id).is_none() {
        bail!("collection {collection} not found");
    }
    let pool = ShardPool::open_read_only(pool_dir).context("failed to open shard store")?;
    let lines = export_lines(&store, &pool, &collection_id, format)?;

    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    for line in &lines {
        output.write_all(line)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    eprintln!(
        "exported {} records from collection {}",
        lines.len(),
        format_id(&collection_id)
    );
    Ok(())
}

/// Enumerate one collection's live records and render each as a line (without
/// its trailing newline). Payload format returns the raw stored bytes; envelope
/// format reads the winning frame so payload and metadata come from one frame.
fn export_lines(
    store: &PackfileStorage,
    pool: &ShardPool,
    collection_id: &[u8; 16],
    format: ExportFormat,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let shards = pool.all_shards();

    // Enumerate the collection's node ids in physical scan order; `seen` fixes
    // the output order at first sight. The winning frame for each id is not
    // chosen here: pack addresses are random, so ordering by address would not
    // reproduce the live index. The envelope path asks the index for the live
    // location instead (see `get_location`), so it always exports the same
    // frame that `get` would return.
    let mut seen = HashSet::new();
    let mut ordered_ids = Vec::new();
    let mut pack_paths: HashMap<PackId, PathBuf> = HashMap::new();
    for (_, shard) in &shards {
        pack_paths.insert(shard.pack_id, shard.path.clone());
        for_each_frame(&shard.path, |_offset, frame| {
            if frame.collection_id != *collection_id {
                return Ok(());
            }
            if seen.insert(frame.hash) {
                ordered_ids.push(frame.hash);
            }
            Ok(())
        })?;
    }

    // Keep one open handle per pack for the whole pass: envelope export reads
    // the winning frame at its offset, and reopening each pack per record would
    // turn a large export into an open/close storm.
    let mut readers: HashMap<PackId, BufReader<fs::File>> = HashMap::new();

    let mut lines = Vec::with_capacity(ordered_ids.len());
    for node_id in ordered_ids {
        match format {
            ExportFormat::Payload => {
                if let Some(data) = store.get(collection_id, &node_id)? {
                    lines.push(data.bytes.to_vec());
                }
            }
            ExportFormat::Envelope => {
                // Read the winning frame directly so the payload and the frame
                // metadata in one envelope always describe the same frame,
                // rather than mixing an index-resolved payload with a
                // separately scanned metadata copy. The winning location comes
                // from the live index, matching what `get` returns.
                let Some((pack_id, offset)) = store.get_location(collection_id, &node_id)? else {
                    continue;
                };
                let Some(path) = pack_paths.get(&pack_id) else {
                    continue;
                };
                let record = read_frame_at(&mut readers, pack_id, path, offset)?;
                lines.push(
                    export_envelope_line(
                        collection_id,
                        &node_id,
                        &record.data,
                        record.metadata.as_ref(),
                    )
                    .into_bytes(),
                );
            }
        }
    }
    Ok(lines)
}

/// Read one frame at a known committed byte offset from a pack, reusing a
/// cached handle (opened on first use) so a per-record read is a seek, not an
/// open/close pair.
fn read_frame_at(
    readers: &mut HashMap<PackId, BufReader<fs::File>>,
    pack_id: PackId,
    path: &Path,
    offset: u64,
) -> anyhow::Result<mtxdb::packfile::Record> {
    let reader = match readers.entry(pack_id) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let file =
                fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            entry.insert(BufReader::new(file))
        }
    };
    reader
        .seek(SeekFrom::Start(offset))
        .with_context(|| format!("seeking to offset {offset} in {}", path.display()))?;
    mtxdb::packfile::read_record(reader)?
        .with_context(|| format!("frame at offset {offset} in {} is missing", path.display()))
}

/// Call `visit` for every frame in a pack file with its byte offset and parsed
/// identity/metadata. Payloads are streamed for CRC verification but never
/// materialized or decompressed. A torn active tail stops the scan cleanly.
fn for_each_frame(
    path: &Path,
    mut visit: impl FnMut(u64, mtxdb::packfile::RecordMetadata) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    // A missing magic header is a broken packfile, not an empty one; without
    // this check the record loop would start at offset 4 and mis-frame.
    mtxdb::packfile::read_header(&mut reader)?.with_context(|| {
        format!(
            "not an mtxdb packfile (missing magic header): {}",
            path.display()
        )
    })?;
    loop {
        let offset = reader.stream_position()?;
        match mtxdb::packfile::read_record_metadata(&mut reader) {
            Ok(Some(frame)) => visit(offset, frame)?,
            Ok(None) => break,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

/// Render one `mtxdb.export/v1` envelope as a single JSON line.
fn export_envelope_line(
    collection_id: &[u8; 16],
    node_id: &[u8; 16],
    payload: &[u8],
    metadata: Option<&mtxdb::packfile::FrameMetadata>,
) -> String {
    let mut fields: Vec<(&'static str, String)> = vec![
        ("schema", json_string(EXPORT_ENVELOPE_SCHEMA)),
        (
            "collection_id",
            json_string(&format!("0x{}", hex::encode(collection_id))),
        ),
        (
            "node_id",
            json_string(&format!("0x{}", hex::encode(node_id))),
        ),
    ];
    fields.extend(payload_fields(payload));
    fields.push((
        "metadata",
        metadata.map_or_else(|| "null".to_owned(), render_frame_metadata),
    ));
    json_compact_object(&fields)
}

/// Compact, single-line JSON object from pre-rendered fields. Keys are static
/// ASCII literals, so only the values need escaping (done by the callers).
pub(crate) fn json_compact_object(fields: &[(&str, String)]) -> String {
    let mut out = String::from("{");
    for (index, (key, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(key);
        out.push_str("\":");
        out.push_str(value);
    }
    out.push('}');
    out
}

/// Payload fields shared by the export envelope and the pack dump:
/// `payload_base64` is always the exact stored bytes (so the representation is
/// lossless); `payload` is a decoded convenience value present only when the
/// bytes are a single JSON document. The decoded value may differ
/// byte-for-byte from the stored bytes (whitespace, key order, numeric
/// spelling), which is why `payload_base64` is authoritative.
fn payload_fields(payload: &[u8]) -> Vec<(&'static str, String)> {
    let mut fields = vec![("payload_base64", json_string(&base64_encode(payload)))];
    match compact_json_value(payload) {
        Some(value) => {
            fields.push(("payload_encoding", json_string("json")));
            fields.push(("payload", value));
        }
        None => fields.push(("payload_encoding", json_string("base64"))),
    }
    fields
}

/// Compact single-line JSON encoding of one stored value, or `None` if the
/// payload is not a single JSON document (e.g. a binary HAMT node).
fn compact_json_value(bytes: &[u8]) -> Option<String> {
    let copy = bytes.to_vec();
    let value = OwnedValue::parse_bytes(&copy).ok()?;
    Some(value.encode())
}

/// Render a frame metadata block as a compact JSON object.
fn render_frame_metadata(metadata: &mtxdb::packfile::FrameMetadata) -> String {
    let mut fields: Vec<(&str, String)> = Vec::new();
    if let Some(logical_id) = &metadata.logical_id {
        fields.push((
            "logical_id",
            json_string(&format!("0x{}", hex::encode(logical_id))),
        ));
    }
    if let Some(content_digest) = &metadata.content_digest {
        fields.push((
            "content_digest",
            json_string(&format!("0x{}", hex::encode(content_digest))),
        ));
        fields.push((
            "digest_algorithm",
            json_string(digest_algorithm_name(metadata.digest_algorithm)),
        ));
        fields.push((
            "digest_algorithm_id",
            json_string(&metadata.digest_algorithm.id().to_string()),
        ));
    }
    if let Some(role) = &metadata.role {
        let (encoding, value) = match std::str::from_utf8(role) {
            Ok(text) => ("utf8", text.to_owned()),
            Err(_) => ("base64", base64_encode(role)),
        };
        fields.push((
            "role",
            json_compact_object(&[
                ("encoding", json_string(encoding)),
                ("value", json_string(&value)),
            ]),
        ));
    }
    if !metadata.unknown.is_empty() {
        let encoded: Vec<String> = metadata
            .unknown
            .iter()
            .map(|(tag, value)| {
                json_compact_object(&[
                    ("tag", tag.to_string()),
                    ("value_base64", json_string(&base64_encode(value))),
                ])
            })
            .collect();
        fields.push(("unknown", format!("[{}]", encoded.join(","))));
    }
    json_compact_object(&fields)
}

/// Human-readable name for a content-digest algorithm.
fn digest_algorithm_name(algorithm: DigestAlgorithm) -> &'static str {
    match algorithm {
        DigestAlgorithm::Sha256 => "sha256",
        DigestAlgorithm::Blake3 => "blake3",
        DigestAlgorithm::Unknown(_) => "unknown",
    }
}

/// Standard (padded) base64, used for non-JSON payloads and metadata bytes.
fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Schema identifier stamped on every `packs dump` line.
const PACK_DUMP_SCHEMA: &str = "mtxdb.pack.dump/v1";

/// Read-only `packs` group: list packs or dump one pack's frames as JSONL.
fn cmd_packs(cli: &Cli, action: &PacksAction) -> anyhow::Result<()> {
    match action {
        PacksAction::List { all } => cmd_shards(cli, *all, false, None),
        PacksAction::Inspect { pack } => cmd_info_pack(cli, pack, None),
        PacksAction::Dump {
            pack,
            collection,
            out,
        } => cmd_packs_dump(cli, pack, collection.as_deref(), out.as_deref()),
        PacksAction::Extract {
            pack,
            collection,
            out,
        } => cmd_packs_extract(cli, pack, collection, out),
    }
}

/// Find the actual directory entry for a pack address.
fn find_pack_path(dir: &Path, pack_id: PackId) -> anyhow::Result<PathBuf> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "pack") {
            continue;
        }
        let Ok(file) = fs::File::open(&path) else {
            continue;
        };
        let Ok(header) = mtxdb::packfile::read_header(&mut BufReader::new(file)) else {
            continue;
        };
        if header.is_some_and(|header| header.pack_id == pack_id) {
            return Ok(path);
        }
    }
    bail!("pack {pack_id} not found in {}", dir.display())
}

/// Resolve a pack selector to its shard pool type, physical file path, and
/// address. Refuses to resolve if the selector is ambiguous across multiple
/// pools unless `-t` narrows the search.
fn resolve_pack_file(
    cli: &Cli,
    selector: &PackSelector,
) -> anyhow::Result<(ShardType, PathBuf, PackId)> {
    let layout = open_layout(cli)?;
    let mut matches: Vec<(ShardType, PathBuf, PackId)> = Vec::new();
    for shard_type in cli.shard_types() {
        let dir = pool_dir(&layout, shard_type)?;
        if !dir.is_dir() {
            continue;
        }
        let Ok(packs) = glob_pack_files(&dir) else {
            continue;
        };
        for (pack_id, _, _) in packs {
            if selector.matches(&pack_id) {
                matches.push((shard_type, find_pack_path(&dir, pack_id)?, pack_id));
            }
        }
    }
    match matches.as_slice() {
        [(shard_type, path, pack_id)] => Ok((*shard_type, path.clone(), *pack_id)),
        [] => bail!("pack {selector}: not found"),
        _ => {
            let pools: Vec<&str> = matches.iter().map(|(kind, _, _)| kind.as_str()).collect();
            bail!(
                "pack {selector} is ambiguous (matches in {}); pass -t <type> or a longer selector",
                pools.join(", ")
            );
        }
    }
}

/// Dump every frame in one pack, in physical order and without deduplication,
/// as one JSON object per line. This is a complete view of a single pack's
/// contents (all collections, superseded frames included) rather than a
/// collection's live set. `payload_base64` carries the exact stored bytes;
/// `payload` is a decoded convenience value for JSON payloads.
fn cmd_packs_dump(
    cli: &Cli,
    pack_selector: &str,
    collection_selector: Option<&str>,
    out: Option<&Path>,
) -> anyhow::Result<()> {
    cli.require_single_dir("packs dump")?;
    let selector = parse_pack_id_selector(pack_selector)?;
    let collection_filter = collection_selector
        .map(parse_collection_selector)
        .transpose()?;

    // Refuse to merge two different packs (e.g. a short prefix matching in more
    // than one pool) into one output stream.
    let (shard_type, path, pack_id) = resolve_pack_file(cli, &selector)?;

    let stdout = io::stdout();
    let mut writer: Box<dyn Write> = match out {
        Some(path) => Box::new(BufWriter::new(
            fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(stdout.lock())),
    };

    let mut written = 0_usize;
    let mut scanner = mtxdb::packfile::scan_records_iter(&path)?;
    for frame in scanner.by_ref() {
        let (offset, record) = frame?;
        if collection_filter.is_some_and(|filter| record.collection_id != filter) {
            continue;
        }
        writer
            .write_all(pack_dump_line(shard_type.as_str(), &pack_id, offset, &record).as_bytes())?;
        writer.write_all(b"\n")?;
        written = written.saturating_add(1);
    }
    writer.flush()?;
    if scanner.torn_tail() {
        eprintln!(
            "warning: pack {pack_id} ends in a torn frame; its tail was not dumped \
             (a concurrent writer can cause this)"
        );
    }
    eprintln!(
        "dumped {written} frames from pack {pack_id} ({} pool)",
        shard_type.as_str()
    );
    Ok(())
}

/// Extract frames belonging to a collection from one pack into a new valid
/// packfile with a fresh pack header, copying frame bytes verbatim.
fn cmd_packs_extract(
    cli: &Cli,
    pack_selector: &str,
    collection_selector: &str,
    out: &Path,
) -> anyhow::Result<()> {
    cli.require_single_dir("packs extract")?;
    let selector = parse_pack_id_selector(pack_selector)?;
    let collection_id = parse_collection_selector(collection_selector)?;
    let (shard_type, source_path, pack_id) = resolve_pack_file(cli, &selector)?;

    // Guard against accidental source file truncation when --out points to the same file.
    if let (Ok(src_canon), Ok(dst_canon)) = (source_path.canonicalize(), out.canonicalize()) {
        if src_canon == dst_canon {
            bail!(
                "--out cannot be the same file as the source pack `{}`",
                source_path.display()
            );
        }
    } else if source_path == out {
        bail!(
            "--out cannot be the same file as the source pack `{}`",
            source_path.display()
        );
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating directory `{}`", parent.display()))?;
        }
    }

    // The extracted pack is a brand-new globally unique pack, so it gets a
    // fresh random address rather than inheriting the source's (or one derived
    // from the output filename).
    let dest_pack_id = PackId::random();

    let stats = mtxdb::packfile::extract_packfile_collection(
        &source_path,
        out,
        &collection_id,
        &dest_pack_id,
    )
    .with_context(|| {
        format!(
            "extracting collection {} from `{}` into `{}`",
            format_id(&collection_id),
            source_path.display(),
            out.display(),
        )
    })?;

    if stats.torn_tail {
        eprintln!(
            "warning: pack {pack_id} ends in a torn frame; its tail was not extracted \
             (a concurrent writer can cause this)"
        );
    }
    eprintln!(
        "extracted {} frames ({} bytes) from pack {pack_id} ({} pool) into {} as new pack {}",
        stats.frames_extracted,
        stats.frame_bytes_written,
        shard_type.as_str(),
        out.display(),
        dest_pack_id,
    );
    Ok(())
}

/// Render one `mtxdb.pack.dump/v1` line for a pack frame.
fn pack_dump_line(
    pool: &str,
    pack_id: &PackId,
    offset: u64,
    record: &mtxdb::packfile::Record,
) -> String {
    let mut fields: Vec<(&'static str, String)> = vec![
        ("schema", json_string(PACK_DUMP_SCHEMA)),
        ("pool", json_string(pool)),
        ("pack_id", json_string(&pack_id.to_string())),
        ("offset", offset.to_string()),
        (
            "collection_id",
            json_string(&format!("0x{}", hex::encode(record.collection_id))),
        ),
        (
            "node_id",
            json_string(&format!("0x{}", hex::encode(record.hash))),
        ),
    ];
    fields.extend(payload_fields(&record.data));
    fields.push((
        "metadata",
        record
            .metadata
            .as_ref()
            .map_or_else(|| "null".to_owned(), render_frame_metadata),
    ));
    json_compact_object(&fields)
}

/// What one input file contributed, so the batch can finish only the pools it
/// actually wrote to.
#[derive(Debug, Default)]
struct ImportOutcome {
    /// A `server_keys` file wrote records into the server-info pool, which the
    /// batch must sync once: that pool is otherwise outside `cmd_import`'s
    /// default sync set.
    server_keys: bool,
    /// Parsed key records, retained so each room can verify without scanning
    /// the live `ServerInfo` pool during journal reclamation.
    signature_catalog: Option<export_import::SignatureKeyCatalog>,
    /// The input was an `mtxdb-export` file, written in the server's layout;
    /// holds the event records it wrote (zero for a keys export). Only a run
    /// whose every input did this can be vouched for by the marker.
    exported: Option<export_import::ImportCounts>,
    /// Content-hash mismatches from the room export, if this was one.
    hash_audit: Vec<export_import::HashAudit>,
    /// Events whose signature did not verify, from the room export.
    signature_audit: Vec<export_import::SignatureAudit>,
    /// Key bindings written by this input, whose candidate observation times
    /// must be replaced by verified observations after all rooms are read.
    key_bindings: Vec<export_import::KeyBinding>,
    /// Verified key observations found in this input's room export.
    verified_keys: Vec<export_import::VerifiedKeyObservation>,
    /// Events that verified after their key's cached deadline.
    /// Derived rejections by stored rejection code, from this input.
    rejection_counts: std::collections::BTreeMap<u8, usize>,
    /// Events with no usable in-window key for their signer.
    signature_no_key: usize,
    /// Events without the material needed to attempt signature verification.
    signature_no_material: usize,
    /// Events rejected because a usable key had a bad signature.
    signature_invalid: usize,
    /// Events that verified after their key's cached `valid_until_ts`.
    signature_verified_late: usize,
}

/// Import one input file through the state the batch shares with its readers.
#[allow(clippy::too_many_arguments)]
/// Matrix events already loaded for a room during this `import` run.
///
/// A run that imports many chunks otherwise rescans and re-parses every
/// earlier chunk from disk for each new file. The cache holds the authoritative
/// stored timeline events (redacted, with auth projections overlaid), so later
/// files start from memory. It lives only for one command invocation.
///
/// The same cache owns the run-scoped **waiting set**: ids the room references
/// but no store holds yet, split by the edge family that cites them. Building
/// it is one O(N) pass over the room's CSR rows; after that, each file asks a
/// file-sized question ("does this file supply a waiting id?") instead of
/// rescanning every stored event's references.
///
/// It also carries the direct-path switch and a count of files the direct path
/// handled. The switch defaults to `MTXDB_DIRECT_IMPORT` but a caller (a test,
/// or a future caller that wants the fast path unconditionally) can set it
/// explicitly, so the gate is not a process-global side effect.
struct ImportRunCache {
    rooms: HashMap<[u8; 16], Vec<OwnedValue>>,
    waiting: HashMap<[u8; 16], RoomWaiting>,
    reverse: HashMap<[u8; 16], RoomReverse>,
    direct: bool,
    direct_files: usize,
    /// Stored signing keys by server for this run. Each lookup scans the whole
    /// server-info pool, so rooms share one read per server. Cleared when a
    /// keys file is imported.
    server_keys: HashMap<String, Vec<mtxdb::server_keys::ServerKey>>,
}

impl Default for ImportRunCache {
    fn default() -> Self {
        Self {
            rooms: HashMap::new(),
            waiting: HashMap::new(),
            reverse: HashMap::new(),
            direct: direct_import_enabled(),
            direct_files: 0,
            server_keys: HashMap::new(),
        }
    }
}

/// Ids a room references that no store holds yet, split by edge family.
///
/// `prev` and `auth` entries have different dirty closures. Satisfying a
/// `prev` id can change the state groups of the citing event's descendants;
/// satisfying an `auth` id changes only the verdicts of the events that cite
/// it, never their state. Keeping the families apart is what lets a file
/// decide, in O(|F|), whether it can touch earlier events at all.
///
/// A stored timeline body satisfies both families. An auth-chain body
/// satisfies `auth` — it is the same event, merely stored as a provider — but
/// not `prev`, because it has no state group and cannot be a parent state
/// base. A projection is an overlay on a body, not a body, so it satisfies
/// neither family.
#[derive(Debug, Default, Clone)]
struct RoomWaiting {
    room_id: String,
    prev: HashSet<String>,
    auth: HashSet<String>,
    /// Timeline bodies the rebuild saw referenced. A cheap lower bound on the
    /// room's stored events, used only to bound the direct path's dirty set.
    stored_events: usize,
}

/// Two waiting sets are equal when they wait for the same ids. The
/// `stored_events` hint describes the room at the time it was built, so an
/// incrementally updated set and a rebuild need not agree on it.
impl PartialEq for RoomWaiting {
    fn eq(&self, other: &Self) -> bool {
        self.room_id == other.room_id && self.prev == other.prev && self.auth == other.auth
    }
}

impl Eq for RoomWaiting {}

/// The waiting ids a file supplies, by family.
#[derive(Debug, Default, Clone)]
struct WaitingHits {
    prev: HashSet<String>,
    auth: HashSet<String>,
}

impl RoomWaiting {
    /// Build the set from references and the bodies that exist right now.
    fn from_scratch(
        room_id: &str,
        prev_refs: impl IntoIterator<Item = String>,
        auth_refs: impl IntoIterator<Item = String>,
        timeline_present: &HashSet<String>,
        anywhere_present: &HashSet<String>,
    ) -> Self {
        Self {
            room_id: room_id.to_owned(),
            prev: prev_refs
                .into_iter()
                .filter(|id| !timeline_present.contains(id))
                .collect(),
            auth: auth_refs
                .into_iter()
                .filter(|id| !anywhere_present.contains(id))
                .collect(),
            stored_events: timeline_present.len(),
        }
    }

    /// Answer "does this file supply an id an earlier event cited without
    /// having the body?" without rescanning the room.
    fn hits<'a>(&self, supplied: impl IntoIterator<Item = &'a String>) -> WaitingHits {
        let mut hits = WaitingHits::default();
        for id in supplied {
            if self.prev.contains(id) {
                hits.prev.insert(id.clone());
            }
            if self.auth.contains(id) {
                hits.auth.insert(id.clone());
            }
        }
        hits
    }

    /// Apply one successfully written file to the set.
    ///
    /// `timeline_arrived` are ids whose timeline body is now stored;
    /// `auth_arrived` are ids whose body is now in the auth-chain collection.
    /// Only a timeline arrival clears `prev`; either arrival clears `auth`. A
    /// reference enters a family when that family's body store has no record.
    fn observe_written(
        &mut self,
        timeline_arrived: &HashSet<String>,
        auth_arrived: &HashSet<String>,
        references: impl IntoIterator<Item = (String, bool)>,
        timeline_present: &HashSet<String>,
        anywhere_present: &HashSet<String>,
    ) {
        // A timeline body is also an auth body, so it clears both families.
        for id in timeline_arrived {
            self.prev.remove(id);
            self.auth.remove(id);
        }
        // An auth-chain body can satisfy check 4, but it has no state group and
        // must not clear a `prev` reference.
        for id in auth_arrived {
            self.auth.remove(id);
        }
        for (id, is_prev) in references {
            if timeline_arrived.contains(&id) {
                // A timeline body satisfies both families; it also cannot be a
                // gap for this file's own references.
                continue;
            }
            if is_prev {
                // An auth-chain body is still not a timeline body: it cannot
                // seed a descendant's state group, so the prev gap remains.
                if !timeline_present.contains(&id) {
                    self.prev.insert(id);
                }
            } else if !anywhere_present.contains(&id) {
                self.auth.insert(id);
            }
        }
    }
}

/// Reverse edges for one room in the direct import's own id space.
///
/// Built from the same CSR rows the waiting set reads, then updated with each
/// file's timeline edges. The direct path needs the children of a late arrival
/// to decide which stored events to re-judge; the CSR store keeps only forward
/// rows, so this is the transpose held for the lifetime of one `import` run.
/// It is keyed by event id (not by either short-id space) because that is what
/// verdicts, projections and state-group mappings use.
#[derive(Debug, Default, Clone)]
struct RoomReverse {
    room_id: String,
    prev: HashMap<String, Vec<String>>,
    auth: HashMap<String, Vec<String>>,
}

/// Build one family's parent-event-id -> child-event-id map from the CSR
/// snapshot.
fn reverse_edges_by_event_id(
    db: &Database,
    adjacency: &CsrAdjacency,
    family: mtxdb::csr::EdgeFamily,
) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let reverse = adjacency.reverse_family(db, family)?;
    let parents = reverse.parents().to_vec();
    let mut short_ids = parents.clone();
    for parent in &parents {
        if let Some(children) = reverse.children_of(*parent) {
            short_ids.extend(children.iter().map(|edge| edge.target));
        }
    }
    short_ids.sort_unstable();
    short_ids.dedup();
    let resolved = adjacency.resolve_event_ids(db, &short_ids)?;
    let names: HashMap<u32, String> = short_ids
        .iter()
        .copied()
        .zip(resolved)
        .filter_map(|(short, name)| name.map(|name| (short, name)))
        .collect();
    let mut edges = HashMap::with_capacity(parents.len());
    for parent in parents {
        let Some(parent_name) = names.get(&parent) else {
            continue;
        };
        let mut children: Vec<String> = reverse
            .children_of(parent)
            .unwrap_or(&[])
            .iter()
            .filter_map(|edge| names.get(&edge.target).cloned())
            .collect();
        children.sort();
        children.dedup();
        edges.insert(parent_name.clone(), children);
    }
    Ok(edges)
}

/// Ids among `ids` whose body is stored in the timeline collection, and ids
/// whose body is stored in either body store a raw import can read: the
/// timeline collection or the `AUTH`-namespace auth-chain collection.
///
/// A projection is an overlay on a body, not a body: it never makes an event
/// readable on its own, so it is deliberately not counted here.
fn stored_body_ids(
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    template: &CollectionTemplate,
    collection_id: &[u8; 16],
    room_id: &str,
    ids: &HashSet<String>,
) -> anyhow::Result<(HashSet<String>, HashSet<String>)> {
    if ids.is_empty() {
        return Ok((HashSet::new(), HashSet::new()));
    }
    let mut ordered: Vec<(String, NodeId)> = Vec::with_capacity(ids.len());
    for id in ids {
        ordered.push((id.clone(), template_event_id_node_id(template, id)?));
    }
    let nodes: Vec<NodeId> = ordered.iter().map(|(_, node)| *node).collect();
    let timeline = store.get_many(collection_id, &nodes)?;
    let auth_collection =
        derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), room_id.as_bytes());
    let auth = state_store.get_many(&auth_collection, &nodes)?;
    let mut timeline_present = HashSet::with_capacity(ordered.len());
    let mut body_present = HashSet::with_capacity(ordered.len());
    for (((id, _), timeline), auth) in ordered.iter().zip(timeline).zip(auth) {
        if timeline.is_some() {
            timeline_present.insert(id.clone());
        }
        if timeline.is_some() || auth.is_some() {
            body_present.insert(id.clone());
        }
    }
    Ok((timeline_present, body_present))
}

/// Ids among `ids` whose body is stored in the room's auth-chain collection.
///
/// This is the prior-file presence probe for `auth_events` references. A
/// timeline body is not included because an id already in the stored timeline
/// is excluded from the candidate set before this is called.
fn auth_pool_ids(
    state_store: &PackfileStorage,
    template: &CollectionTemplate,
    room_id: &str,
    ids: &HashSet<String>,
) -> anyhow::Result<HashSet<String>> {
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    let mut ordered: Vec<(String, NodeId)> = Vec::with_capacity(ids.len());
    for id in ids {
        ordered.push((id.clone(), template_event_id_node_id(template, id)?));
    }
    let nodes: Vec<NodeId> = ordered.iter().map(|(_, node)| *node).collect();
    let auth_collection =
        derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), room_id.as_bytes());
    let records = state_store.get_many(&auth_collection, &nodes)?;
    let mut present = HashSet::with_capacity(ordered.len());
    for ((id, _), record) in ordered.iter().zip(records) {
        if record.is_some() {
            present.insert(id.clone());
        }
    }
    Ok(present)
}

impl ImportRunCache {
    /// Rebuild the waiting set for a collection from the room's CSR rows and
    /// the three stores that can hold an event body.
    fn rebuild_waiting(
        &mut self,
        db: &Database,
        store: &PackfileStorage,
        state_store: &PackfileStorage,
        template: &CollectionTemplate,
        collection_id: &[u8; 16],
        room_id: &str,
    ) -> anyhow::Result<&RoomWaiting> {
        let (prev_refs, auth_refs) = RoomAuth::new(room_id).referenced_event_ids_by_family(db)?;
        let mut all: HashSet<String> = prev_refs.iter().cloned().collect();
        all.extend(auth_refs.iter().cloned());
        let (timeline_present, anywhere_present) =
            stored_body_ids(store, state_store, template, collection_id, room_id, &all)?;
        let entry = RoomWaiting::from_scratch(
            room_id,
            prev_refs,
            auth_refs,
            &timeline_present,
            &anywhere_present,
        );
        self.waiting.insert(*collection_id, entry);
        Ok(self
            .waiting
            .get(collection_id)
            .expect("entry was just inserted"))
    }

    /// The collection's waiting set, rebuilt once per run if it is not there.
    ///
    /// The only staleness check is the room id: an import run never deletes,
    /// removes or repacks a room between files, so CSR rows only grow. A caller
    /// that adds such an operation between files must rebuild instead.
    fn ensure_waiting(
        &mut self,
        db: &Database,
        store: &PackfileStorage,
        state_store: &PackfileStorage,
        template: &CollectionTemplate,
        collection_id: &[u8; 16],
        room_id: &str,
    ) -> anyhow::Result<&RoomWaiting> {
        let stale = self
            .waiting
            .get(collection_id)
            .is_none_or(|entry| entry.room_id != room_id);
        if stale {
            return self.rebuild_waiting(db, store, state_store, template, collection_id, room_id);
        }
        Ok(self
            .waiting
            .get(collection_id)
            .expect("waiting set was just checked"))
    }

    /// The collection's reverse edges, transposed from the CSR store once per
    /// run.
    fn ensure_reverse(
        &mut self,
        db: &Database,
        collection_id: &[u8; 16],
        room_id: &str,
    ) -> anyhow::Result<&RoomReverse> {
        let stale = self
            .reverse
            .get(collection_id)
            .is_none_or(|entry| entry.room_id != room_id);
        if stale {
            let adjacency = CsrAdjacency::new(room_id);
            let prev = reverse_edges_by_event_id(db, &adjacency, PREV)?;
            let auth = reverse_edges_by_event_id(db, &adjacency, AUTH)?;
            self.reverse.insert(
                *collection_id,
                RoomReverse {
                    room_id: room_id.to_owned(),
                    prev,
                    auth,
                },
            );
        }
        Ok(self
            .reverse
            .get(collection_id)
            .expect("reverse edges were just checked"))
    }

    /// The stored events a file's late arrivals can change: the direct citers
    /// of every supplied waiting id, plus all of their `prev` descendants.
    ///
    /// An `auth` arrival changes the citer's verdict, which may change the
    /// state it contributes; a `prev` arrival changes the citer's base state.
    /// Both effects propagate to descendants through `prev_events`, never
    /// through `auth_events`, because a verdict is not an auth input.
    fn dirty_closure(
        &mut self,
        db: &Database,
        collection_id: &[u8; 16],
        room_id: &str,
        hits: &WaitingHits,
    ) -> anyhow::Result<HashSet<String>> {
        if hits.prev.is_empty() && hits.auth.is_empty() {
            return Ok(HashSet::new());
        }
        let reverse = self.ensure_reverse(db, collection_id, room_id)?;
        let mut dirty: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        for hit in hits.prev.iter().chain(hits.auth.iter()) {
            for edges in [
                hits.prev
                    .contains(hit)
                    .then(|| reverse.prev.get(hit))
                    .flatten(),
                hits.auth
                    .contains(hit)
                    .then(|| reverse.auth.get(hit))
                    .flatten(),
            ]
            .into_iter()
            .flatten()
            {
                for child in edges {
                    if dirty.insert(child.clone()) {
                        queue.push_back(child.clone());
                    }
                }
            }
        }
        // The citers are the roots. Two relations propagate the change and
        // must both be followed to a fixpoint:
        // - `prev`: a changed verdict or a newly complete base changes the
        //   state this event contributes to its descendants.
        // - `auth`: a changed verdict can flip the `DependsOnRejectedAuthEvent`
        //   cascade for every event that cites it.
        while let Some(parent) = queue.pop_front() {
            for child in reverse
                .prev
                .get(&parent)
                .into_iter()
                .chain(reverse.auth.get(&parent))
                .flatten()
            {
                if dirty.insert(child.clone()) {
                    queue.push_back(child.clone());
                }
            }
        }
        Ok(dirty)
    }

    /// Add one written file's timeline edges to the cached reverse maps.
    ///
    /// A no-op when no late arrival has forced the transpose yet: the next
    /// caller builds from the CSR store, which already includes these rows.
    fn observe_adjacency<'a>(
        &mut self,
        collection_id: &[u8; 16],
        room_id: &str,
        written_events: impl IntoIterator<Item = &'a OwnedValue>,
    ) {
        let Some(reverse) = self.reverse.get_mut(collection_id) else {
            return;
        };
        if reverse.room_id != room_id {
            return;
        }
        for event in written_events {
            let Some(child) = event_id(event) else {
                continue;
            };
            for (field, map) in [
                ("prev_events", &mut reverse.prev),
                ("auth_events", &mut reverse.auth),
            ] {
                let Some(list) = event.get(field).and_then(OwnedValue::as_array) else {
                    continue;
                };
                for reference in list {
                    let id = match reference {
                        OwnedValue::String(id) => Some(id.as_str()),
                        OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                        _ => None,
                    };
                    let Some(id) = id else { continue };
                    // Append without a duplicate scan: each event is normally
                    // written once per run, and a duplicate child only costs a
                    // repeated visit that the closure's `HashSet` collapses.
                    // Scanning the whole child list per edge made this O(n²)
                    // across a run and dominated the gapped import.
                    map.entry(id.to_owned()).or_default().push(child.to_owned());
                }
            }
        }
    }

    /// Fold one written file into the waiting set. Rebuilds first if this run
    /// has not touched the collection yet, so the first file's own references
    /// are accounted for.
    #[allow(clippy::too_many_arguments)]
    fn observe_written<'a>(
        &mut self,
        db: &Database,
        store: &PackfileStorage,
        state_store: &PackfileStorage,
        template: &CollectionTemplate,
        collection_id: &[u8; 16],
        room_id: &str,
        timeline_arrived: &HashSet<String>,
        auth_arrived: &HashSet<String>,
        written_events: impl IntoIterator<Item = &'a OwnedValue>,
    ) -> anyhow::Result<()> {
        self.ensure_waiting(db, store, state_store, template, collection_id, room_id)?;
        let mut references: Vec<(String, bool)> = Vec::new();
        let mut candidates: HashSet<String> = HashSet::new();
        for event in written_events {
            // The adjacency recorder skips events without an event id; the
            // waiting set must describe exactly the rows it wrote.
            if event_id(event).is_none() {
                continue;
            }
            for field in ["prev_events", "auth_events"] {
                let Some(list) = event.get(field).and_then(OwnedValue::as_array) else {
                    continue;
                };
                for reference in list {
                    let id = match reference {
                        OwnedValue::String(id) => Some(id.as_str()),
                        OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                        _ => None,
                    };
                    let Some(id) = id else { continue };
                    let is_prev = field == "prev_events";
                    // A timeline body is present for both families. An
                    // auth-chain body is present only for `auth`; a `prev`
                    // reference to it stays a gap and must still be recorded.
                    if timeline_arrived.contains(id) || (!is_prev && auth_arrived.contains(id)) {
                        continue;
                    }
                    references.push((id.to_owned(), is_prev));
                    candidates.insert(id.to_owned());
                }
            }
        }
        let (timeline_present, anywhere_present) = stored_body_ids(
            store,
            state_store,
            template,
            collection_id,
            room_id,
            &candidates,
        )?;
        let entry = self
            .waiting
            .get_mut(collection_id)
            .expect("waiting set was ensured above");
        entry.observe_written(
            timeline_arrived,
            auth_arrived,
            references,
            &timeline_present,
            &anywhere_present,
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_import_file(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    path: &Path,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
    signature_checks_enabled: bool,
    signature_catalog: Option<&export_import::SignatureKeyCatalog>,
    write_state_hamts: bool,
    room_cache: Option<&mut ImportRunCache>,
) -> anyhow::Result<ImportOutcome> {
    let mut context = ImportContext {
        db,
        store,
        state_store,
        template,
        established_collections,
        signature_checks_enabled,
        signature_catalog,
        write_state_hamts,
        room_cache,
    };
    context.import(path, collection_override)
}

/// The state every reader of one input file works against.
///
/// Splitting it out is what lets each input format be a small reader of its
/// own instead of one function that handles every format in sequence.
/// `established_collections` is the part that carries across files: a create
/// event seen in an earlier file leaves its collection established for the
/// rest of the batch.
struct ImportContext<'a> {
    db: &'a Database,
    store: &'a PackfileStorage,
    state_store: &'a PackfileStorage,
    template: &'a CollectionTemplate,
    established_collections: &'a mut HashSet<[u8; 16]>,
    signature_checks_enabled: bool,
    signature_catalog: Option<&'a export_import::SignatureKeyCatalog>,
    write_state_hamts: bool,
    room_cache: Option<&'a mut ImportRunCache>,
}

impl ImportContext<'_> {
    /// Import one file, choosing a reader by what its first line says.
    fn import(
        &mut self,
        path: &Path,
        collection_override: Option<&str>,
    ) -> anyhow::Result<ImportOutcome> {
        let content = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        // An export carries its own framing, and a room export is a `.jsonl`
        // file whose lines are records rather than events, so it has to be
        // recognized before any reader of raw Matrix events sees it.
        if let Some(input_format) = interchange::detect(&content) {
            return self.import_export(path, input_format, &content, collection_override);
        }
        if path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            return self.import_jsonl(path, &content, collection_override);
        }
        self.import_federation(path, &content, collection_override)
    }

    /// An `mtxdb-export/v1` file: a room export, or a `server_keys` export.
    fn import_export(
        &mut self,
        path: &Path,
        input_format: interchange::InputFormat,
        content: &[u8],
        collection_override: Option<&str>,
    ) -> anyhow::Result<ImportOutcome> {
        match input_format {
            interchange::InputFormat::Room => {
                let export = interchange::parse_room(content)
                    .with_context(|| format!("reading {}", path.display()))?;
                if !self.signature_checks_enabled {
                    self.import_room_export(&export, collection_override, None)
                        .with_context(|| format!("importing {}", path.display()))
                } else if let Some(catalog) = self.signature_catalog {
                    // The run's keys file does not replace keys already
                    // stored by an earlier import: verification must see both.
                    let mut merged = self.read_stored_keys_for_room(&export)?;
                    merged.extend_from(catalog);
                    self.import_room_export(&export, collection_override, Some(&merged))
                        .with_context(|| format!("importing {}", path.display()))
                } else {
                    let catalog = self.read_stored_keys_for_room(&export)?;
                    self.import_room_export(&export, collection_override, Some(&catalog))
                        .with_context(|| format!("importing {}", path.display()))
                }
            }
            interchange::InputFormat::Keys => {
                let records = interchange::parse_keys(content)
                    .with_context(|| format!("reading {}", path.display()))?;
                let signature_catalog = export_import::SignatureKeyCatalog::from_records(&records);
                let key_bindings = self.import_keys_export(path, &records)?;
                // The stored keys just changed; anything remembered is stale.
                if let Some(cache) = self.room_cache.as_deref_mut() {
                    cache.server_keys.clear();
                }
                Ok(ImportOutcome {
                    server_keys: true,
                    signature_catalog: Some(signature_catalog),
                    exported: Some(export_import::ImportCounts::default()),
                    hash_audit: Vec::new(),
                    key_bindings,
                    ..ImportOutcome::default()
                })
            }
        }
    }

    /// Signing keys already stored for the servers `export` names, reusing
    /// what earlier rooms of this run read.
    fn read_stored_keys_for_room(
        &mut self,
        export: &interchange::RoomExport,
    ) -> anyhow::Result<export_import::SignatureKeyCatalog> {
        let mut unshared = HashMap::new();
        let memo = match self.room_cache.as_deref_mut() {
            Some(cache) => &mut cache.server_keys,
            None => &mut unshared,
        };
        export_import::SignatureKeyCatalog::read_for_room(self.db.server_info(), export, memo)
            .with_context(|| format!("reading signing keys needed by {}", export.room_id))
    }

    /// A `.jsonl` file: one event per line, or a federation document for the
    /// DAG exports that carry a `.jsonl` name despite being one document.
    fn import_jsonl(
        &mut self,
        path: &Path,
        content: &[u8],
        collection_override: Option<&str>,
    ) -> anyhow::Result<ImportOutcome> {
        let (events, detected_collection, raw_adjacency) =
            parse_jsonl_events(content).or_else(|jsonl_error| {
                parse_federation_events(content)
                    .map(|(events, collection)| (events, collection, Vec::new()))
                    .map_err(|federation_error| {
                        anyhow!(
                            "{jsonl_error}; also not a valid Matrix federation document: \
                             {federation_error}"
                        )
                    })
            })?;
        if events.is_empty() {
            bail!("no events found in {}", path.display());
        }
        import_pdu_events_with_raw_state_hamts(
            self.db,
            self.store,
            self.state_store,
            path,
            &events,
            &[],
            Some(&raw_adjacency),
            detected_collection.as_deref(),
            collection_override,
            self.template,
            self.established_collections,
            self.write_state_hamts,
            self.room_cache.as_deref_mut(),
        )?;
        Ok(ImportOutcome::default())
    }

    /// A federation response document: its PDUs through the event DAG, its
    /// auth-chain bodies in the State pool.
    fn import_federation(
        &mut self,
        path: &Path,
        content: &[u8],
        collection_override: Option<&str>,
    ) -> anyhow::Result<ImportOutcome> {
        let federation = parse_federation_input(content)?;
        // `parse_federation_input` validates the document shape; this check
        // separately rejects a well-shaped but event-empty response.
        if federation.pdus.is_empty() && federation.auth_chain.is_empty() {
            bail!("no events found in {}", path.display());
        }
        let detected_collection = federation
            .pdus
            .iter()
            .chain(federation.auth_chain.iter())
            .find_map(event_room_id)
            .map(str::to_owned);
        // Every `auth_events` reference must point at an event_id in either
        // `pdus` or `auth_chain`; dangling ones are imported but reported.
        let dangling = verify_auth_chain_edges(&federation.pdus, &federation.auth_chain);
        for (source, target) in &dangling {
            eprintln!("warning: auth_events dangling reference: {source} -> {target}");
        }
        if !federation.pdus.is_empty() {
            self.import_pdus(
                path,
                &federation.pdus,
                &federation.auth_chain,
                detected_collection.as_deref(),
                collection_override,
            )?;
        }
        if !federation.auth_chain.is_empty() {
            self.import_auth_chain(
                &federation.auth_chain,
                &dangling,
                detected_collection.as_deref(),
            )?;
        }
        Ok(ImportOutcome::default())
    }

    /// A federation document's PDUs through the ordinary event-DAG path.
    fn import_pdus(
        &mut self,
        path: &Path,
        pdus: &[OwnedValue],
        auth_chain: &[OwnedValue],
        detected_collection: Option<&str>,
        collection_override: Option<&str>,
    ) -> anyhow::Result<()> {
        import_pdu_events_with_raw_state_hamts(
            self.db,
            self.store,
            self.state_store,
            path,
            pdus,
            auth_chain,
            None,
            detected_collection,
            collection_override,
            self.template,
            self.established_collections,
            self.write_state_hamts,
            self.room_cache.as_deref_mut(),
        )
    }

    /// Write a federation document's auth-chain bodies into the State pool.
    ///
    /// Auth-chain events may span several rooms, so they are grouped first
    /// and then probed and written one batch per room rather than one round
    /// trip per event.
    ///
    /// A document with auth-chain bodies and no PDUs still carries information
    /// the waiting set needs: those bodies satisfy `auth` references and their
    /// own references can create new gaps. Fold them in after the writes, one
    /// room's worth at a time, so references from another room are not
    /// attributed to this room's timeline collection.
    fn import_auth_chain(
        &mut self,
        auth_chain: &[OwnedValue],
        dangling: &[(String, String)],
        detected_collection: Option<&str>,
    ) -> anyhow::Result<()> {
        // The auth chain's event bodies go to the State pool, in their own
        // `AUTH`-namespace collections.
        let auth_store: &PackfileStorage = self.db.state();
        // Carry the caller's `--read-plan` choice over from the event store.
        auth_store.set_read_plan_policy(self.store.read_plan_policy());
        // Record the derived adjacency before writing any auth events, so a
        // structural conflict rejects the import before the edges pool is
        // mutated.
        record_csr_adjacency(self.db, self.template, auth_chain)?;
        let (by_collection, skipped) =
            group_auth_chain(auth_chain, self.template, detected_collection)?;
        let direct = self
            .room_cache
            .as_deref()
            .map_or_else(direct_import_enabled, |cache| cache.direct);
        let direct_identity = matches!(
            &self.template.record_id_rule.policy,
            FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
        );
        let mut events_by_room: HashMap<String, Vec<&OwnedValue>> = HashMap::new();
        for event in auth_chain {
            if let Some(room_id) = auth_chain_event_room(event, detected_collection) {
                events_by_room.entry(room_id).or_default().push(event);
            }
        }
        let mut written = 0_u64;
        for ((collection_id, room_id), entries) in by_collection {
            let auth_arrived: HashSet<String> =
                entries.iter().map(|(_, id, _)| id.clone()).collect();
            written = written.saturating_add(write_auth_collection(
                auth_store,
                self.template,
                collection_id,
                room_id.clone(),
                entries,
            )?);
            if direct && direct_identity {
                if let Some(cache) = self.room_cache.as_deref_mut() {
                    let timeline_collection = template_collection_id(self.template, &room_id);
                    cache.observe_written(
                        self.db,
                        self.store,
                        self.state_store,
                        self.template,
                        &timeline_collection,
                        &room_id,
                        &HashSet::new(),
                        &auth_arrived,
                        events_by_room.get(&room_id).into_iter().flatten().copied(),
                    )?;
                }
            }
        }
        auth_store.sync_all()?;
        eprintln!(
            "imported {written} auth-chain events ({} dangling references)",
            dangling.len()
        );
        if skipped > 0 {
            eprintln!("skipped {skipped} edges events (missing event_id)");
        }
        Ok(())
    }

    /// Import one `mtxdb-export/v1` room file.
    ///
    /// The untouched PDU bytes are stored as exported, and the edges, locators
    /// and auth closures are rebuilt from them. Rejections are derived by
    /// running the auth rules over the events; the export's own rejection
    /// flags are not trusted. State groups and their HAMTs are reconstructed
    /// from the imported room events before this import can complete.
    #[allow(
        clippy::too_many_lines,
        reason = "the export import keeps its validation, state reconstruction, and verdict stages together"
    )]
    fn import_room_export(
        &mut self,
        export: &RoomExport,
        collection_override: Option<&str>,
        signature_keys: Option<&export_import::SignatureKeyCatalog>,
    ) -> anyhow::Result<ImportOutcome> {
        if collection_override.is_some() {
            bail!("--collection does not apply to an mtxdb export: the server's layout derives the room's collections itself");
        }
        let phase_start = Instant::now();
        let mut lap = phase_start;
        let mut phases: Vec<(&str, std::time::Duration)> = Vec::new();
        let events: Vec<OwnedValue> = export
            .events
            .iter()
            .map(|event| event.unflagged(&export.room_id))
            .collect();
        phases.push(("unflag", lap.elapsed()));
        lap = Instant::now();
        if !events
            .iter()
            .any(|event| matrix_create_event_id(event).is_some())
        {
            bail!(
                "refusing to import events for {}: no valid m.room.create event is in this export",
                export.room_id
            );
        }
        validate_establishment_room_version(&events)?;
        let room_version = events
            .iter()
            .find(|event| matrix_create_event_id(event).is_some())
            .and_then(|event| nested_event_string_field(event, "content", "room_version"))
            .unwrap_or_default()
            .to_owned();
        phases.push(("validate", lap.elapsed()));
        lap = Instant::now();
        let mut findings = export_import::derived_auth_findings_with_keys(export, signature_keys)?;
        let mut rejected = export_import::rejected_event_ids(export, &findings);
        phases.push(("findings(hash+sig+judge)", lap.elapsed()));
        lap = Instant::now();
        let mut walk_rejections = HashMap::new();
        let (computation, unchecked) = compute_import_auth_before(
            &events,
            &[],
            &export.events,
            &export.room_id,
            &room_version,
            StateResVersion::from_room_version(&room_version),
            &mut rejected,
            &mut walk_rejections,
            &HashSet::new(),
            None,
            None,
            &HashMap::new(),
        );
        phases.push(("state walk+judge", lap.elapsed()));
        lap = Instant::now();
        export_import::apply_walk_rejections(export, &mut findings, &walk_rejections);
        if unchecked > 0 {
            eprintln!(
                "  warning: {unchecked} event(s) could not be checked against the state before them"
            );
        }
        export_import::clear_marker(self.db)?;
        let written = export_import::write_room_with_resolved_findings(
            self.db,
            export,
            export_import::BATCH_BYTES,
            &findings,
        )?;
        phases.push(("write records", lap.elapsed()));
        lap = Instant::now();
        if self.write_state_hamts {
            repair_import_state_groups(
                self.state_store,
                &events,
                &[],
                &export.room_id,
                &room_version,
                Some(computation),
            )?;
        }
        phases.push(("state HAMT+map", lap.elapsed()));
        if crate::debug_enabled() {
            let parts: Vec<String> = phases
                .iter()
                .map(|(name, took)| format!("{name} {}ms", took.as_millis()))
                .collect();
            eprintln!(
                "  timing: {} events in {}ms: {}",
                export.events.len(),
                phase_start.elapsed().as_millis(),
                parts.join(", ")
            );
        }
        eprintln!(
            "  {}",
            export_import::room_collections_line(&export.room_id)
        );
        eprintln!(
            "  imported {} events, {} auth closures",
            written.events, written.closures
        );
        for warning in findings.warnings() {
            eprintln!("  warning: {warning}");
        }
        eprintln!("  exported by {}", export.server);
        Ok(ImportOutcome {
            server_keys: false,
            exported: Some(written),
            hash_audit: findings.hash_audit,
            signature_audit: findings.signature_audit,
            verified_keys: findings.verified_keys,
            signature_no_key: findings.signature_no_key,
            signature_no_material: findings.signature_no_material,
            signature_invalid: findings.signature_failures.len(),
            signature_verified_late: findings.signature_verified_late,
            rejection_counts: findings.rejection_counts,
            ..ImportOutcome::default()
        })
    }

    /// Import one `server_keys` export into the server-info pool, in the
    /// layout the server reads: one record per key in its origin's collection,
    /// and each notary payload stored as the export decoded it.
    fn import_keys_export(
        &self,
        path: &Path,
        records: &[KeyRecord],
    ) -> anyhow::Result<Vec<export_import::KeyBinding>> {
        export_import::clear_marker(self.db)?;
        export_import::write_keys(self.db, records)
            .with_context(|| format!("importing signing keys from {}", path.display()))?;
        let (mut provisional, mut permanent, mut notary) = (0_usize, 0_usize, 0_usize);
        for record in records {
            match record {
                KeyRecord::Key { key, .. } if key.tier == KeyTier::Provisional => {
                    provisional = provisional.saturating_add(1);
                }
                KeyRecord::Key { .. } => permanent = permanent.saturating_add(1),
                KeyRecord::Notary { .. } => notary = notary.saturating_add(1),
            }
        }
        eprintln!(
            "imported {} signing-key records ({provisional} provisional, \
             {permanent} permanent, {notary} notary)",
            records.len()
        );
        Ok(records
            .iter()
            .filter_map(|record| match record {
                KeyRecord::Key { origin, key } => Some(export_import::KeyBinding {
                    origin: origin.as_bytes().to_vec(),
                    key_id: key.key_id.clone(),
                }),
                KeyRecord::Notary { .. } => None,
            })
            .collect())
    }
}

/// Walk the room's state groups once before verdicts are written. Each
/// event is judged in topological order: events the findings pass already
/// rejected are excluded, and every other event gets check 4 followed by
/// check 5 against the state before it. A rejection is fed back into the
/// walk, so a rejected state event contributes no state to later events.
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the walk and its per-event auth context are kept together"
)]
fn compute_import_auth_before(
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    room_events: &[RoomEvent],
    room_id: &str,
    room_version: &str,
    state_version: Option<StateResVersion>,
    rejected: &mut HashSet<String>,
    walk_rejections: &mut HashMap<String, (u8, String)>,
    already_redacted: &HashSet<String>,
    judge_only: Option<&HashSet<String>>,
    lean_input: Option<&[Option<LeanEvent<String>>]>,
    seeds: &HashMap<String, Arc<SharedState>>,
) -> (PartialStateComputation, usize) {
    let policy = resolution_policy(room_version, StateGroupTrust::TrustedImport);
    let Some(state_version) = state_version else {
        return (
            compute_state_groups_partial_with(
                events,
                auth_chain,
                StateView::Federated,
                room_id,
                policy,
                Some(rejected),
                seeds,
                |_, _, _, _, _| None,
            ),
            0,
        );
    };

    let index: HashMap<&[u8], usize> = room_events
        .iter()
        .enumerate()
        .map(|(position, event)| (event.event_id.as_bytes(), position))
        .collect();
    // Judge hash-mismatched events in the same canonical redacted form the
    // findings pass uses, so check 5 cannot reject an event whose redaction
    // the auth rules accept. The incremental import computes this once and
    // passes it in; otherwise build it here.
    let redacted_buf;
    let lean_buf;
    let lean: &[Option<LeanEvent<String>>] = if let Some(lean) = lean_input {
        lean
    } else {
        redacted_buf =
            export_import::redactions_for_export(room_events, room_version, already_redacted);
        lean_buf = export_import::lean_redacted_events(room_events, &redacted_buf, room_id);
        &lean_buf
    };
    let by_id: HashMap<&str, &RoomEvent> = room_events
        .iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect();
    let create = room_events
        .iter()
        .position(|event| {
            event.pdu.get("type").and_then(OwnedValue::as_str) == Some("m.room.create")
                && event.pdu.get("state_key").and_then(OwnedValue::as_str) == Some("")
        })
        .and_then(|position| lean[position].as_ref());

    let mut unchecked = 0_usize;
    let computation = compute_state_groups_partial_with(
        events,
        auth_chain,
        StateView::Federated,
        room_id,
        policy,
        Some(rejected),
        seeds,
        |event_id, event, base, base_complete, current| {
            // Only events whose inputs can have changed need judging; the
            // walk still derives state for every event so later events get
            // the right base.
            if judge_only.is_some_and(|only| !only.contains(event_id)) {
                return None;
            }
            let Some(room_event) = by_id.get(event_id) else {
                unchecked = unchecked.saturating_add(1);
                return None;
            };
            let Some(position) = index.get(event_id.as_bytes()).copied() else {
                unchecked = unchecked.saturating_add(1);
                return None;
            };
            if let Some(current) = current {
                let cites_rejected = room_event.auth_events.as_array().is_some_and(|ids| {
                    ids.iter()
                        .any(|id| id.as_str().is_some_and(|id| current.contains(id)))
                });
                if cites_rejected {
                    let code = RejectionCode::DependsOnRejectedAuthEvent.to_u8();
                    let reason = "auth_events cites a rejected event".to_owned();
                    walk_rejections.insert(event_id.to_owned(), (code, reason.clone()));
                    return Some((code, reason));
                }
            }
            if !base_complete {
                // A missing parent, a merge that fell back to the
                // first-parent rule, or an ancestor with either gap means
                // this base state is incomplete. Leave the event unchecked
                // rather than inventing a rejection.
                unchecked = unchecked.saturating_add(1);
                return None;
            }
            let is_create = event.get("type").and_then(OwnedValue::as_str) == Some("m.room.create")
                && event.get("state_key").and_then(OwnedValue::as_str) == Some("");
            if base.entries.is_empty() && !is_create {
                unchecked = unchecked.saturating_add(1);
                return None;
            }
            let entries = auth_entries_from_base(event, base, state_version, room_version);
            match export_import::judge_event_with_state(
                room_event,
                lean[position].as_ref(),
                &index,
                lean,
                create,
                state_version,
                &entries,
            ) {
                Ok(export_import::EventAuthOutcome::Refused { code, reason }) => {
                    walk_rejections.insert(event_id.to_owned(), (code, reason.clone()));
                    Some((code, reason))
                }
                Ok(export_import::EventAuthOutcome::Passed) => None,
                Ok(export_import::EventAuthOutcome::Unchecked) => {
                    unchecked = unchecked.saturating_add(1);
                    None
                }
                Err(err) => {
                    eprintln!("  warning: error judging event {event_id}: {err}");
                    unchecked = unchecked.saturating_add(1);
                    None
                }
            }
        },
    );
    (computation, unchecked)
}

/// Group an auth chain's events by the collection each one belongs to.
///
/// Returns the entries to write per collection, and how many events had no
/// usable identity to group them by.
/// The room an auth-chain event belongs to, matching the grouping rule.
fn auth_chain_event_room(event: &OwnedValue, detected_collection: Option<&str>) -> Option<String> {
    event_room_id(event)
        .map(str::to_owned)
        .or_else(|| {
            if matrix_create_event_id(event).is_some() {
                establishment_identity(event)
            } else {
                None
            }
        })
        .or_else(|| detected_collection.map(str::to_owned))
}

fn group_auth_chain(
    auth_chain: &[OwnedValue],
    template: &CollectionTemplate,
    detected_collection: Option<&str>,
) -> anyhow::Result<(AuthChainCollections, u64)> {
    let mut by_collection: AuthChainCollections = HashMap::new();
    let mut skipped = 0_u64;
    for event in auth_chain {
        let Some(incoming_event_id) = event_id(event) else {
            skipped = skipped.saturating_add(1);
            continue;
        };
        let Some(id_bytes) = template_node_id(template, event)? else {
            skipped = skipped.saturating_add(1);
            continue;
        };
        let Some(room_id) = auth_chain_event_room(event, detected_collection) else {
            skipped = skipped.saturating_add(1);
            continue;
        };
        let collection_id =
            derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), room_id.as_bytes());
        let event_bytes = event.encode().into_bytes();
        by_collection
            .entry((collection_id, room_id))
            .or_default()
            .push((
                id_bytes,
                incoming_event_id.to_owned(),
                NodeData::new(bytes::Bytes::from(event_bytes)),
            ));
    }
    Ok((by_collection, skipped))
}

/// Write one collection's worth of auth-chain events with a single probe and
/// write, returning how many records were appended.
///
/// Same dedup rule as the PDU path: the 128-bit node ID is a truncated hash,
/// so two distinct event IDs mapping to one key is a collision to reject, not
/// collapse; a repeat with the same `event_id` is the same logical event.
fn write_auth_collection(
    auth_store: &PackfileStorage,
    template: &CollectionTemplate,
    collection_id: [u8; 16],
    room_id: String,
    entries: Vec<ImportBatchEntry>,
) -> anyhow::Result<u64> {
    let mut first: Vec<ImportBatchEntry> = Vec::with_capacity(entries.len());
    let mut seen_ids: HashMap<[u8; 16], String> = HashMap::with_capacity(entries.len());
    for (id_bytes, incoming_event_id, data) in entries {
        if let Some(first_event_id) = seen_ids.get(&id_bytes) {
            if first_event_id != &incoming_event_id {
                bail!(
                    "node ID {} maps to multiple event IDs in the input; refusing to merge them",
                    format_id(&id_bytes)
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
    for ((id_bytes, incoming_event_id, data), existing_opt) in first.into_iter().zip(existing) {
        if let Some(existing_record) = existing_opt {
            reject_cross_record_collision(&id_bytes, &incoming_event_id, &existing_record)?;
        } else {
            to_write.push((id_bytes, data));
        }
    }
    if to_write.is_empty() {
        return Ok(0);
    }
    let auth_metadata = CollectionMetadata {
        member_namespace: Some(mtxdb::MEMBER_NAMESPACE_AUTH),
        collection_canonical_id: room_id.into_bytes(),
        record_id_rule: template.record_id_rule.clone(),
        payload: template.payload.clone(),
        extension: None,
        role: Some("auth_chain".to_owned()),
        schema: Some("matrix.event.v1".to_owned()),
    };
    auth_store.create_or_put_established(&collection_id, &auth_metadata, &to_write)?;
    // Deliberate semantic change from the pre-batching loop, which counted
    // every accepted input event including ones already on disk: this counts
    // only genuinely newly-written edges records, so "imported N" means
    // records actually appended, and a reimport reports 0.
    Ok(to_write.len() as u64)
}

/// A decoded import entry keyed by node ID, retaining the source event ID for
/// collision validation before the record is written.
type ImportBatchEntry = ([u8; 16], String, NodeData);

/// Auth-chain entries grouped by the collection each event belongs to.
type AuthChainCollections = HashMap<([u8; 16], String), Vec<ImportBatchEntry>>;

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
    let existing_bytes = existing_record.bytes.to_vec();
    let existing_event_id = OwnedValue::parse_bytes(&existing_bytes)
        .ok()
        .and_then(|event| event_id(&event).map(str::to_owned));
    if existing_event_id.as_deref() != Some(incoming_event_id) {
        bail!(
            "node ID {} collides with a different event_id; refusing to overwrite it",
            format_id(id_bytes)
        );
    }
    Ok(())
}

/// Apply rezzy's room-versioned Matrix redaction before persisting an event.
/// The importer still keeps the original value in memory for collection and
/// DAG/state analysis; only the stored payload is redacted.
fn redacted_event_bytes(event: &OwnedValue, room_version: &str) -> anyhow::Result<bytes::Bytes> {
    // Rezzy applies the room-version redaction rules and number policy:
    // v1-v5 retain historical non-canonical numbers; v6+ enforce canonical JSON.
    let encoded = rezzy::try_canonical_redacted_json(event, room_version).map_err(|error| {
        anyhow::anyhow!("failed to redact event for room version {room_version}: {error}")
    })?;
    Ok(bytes::Bytes::from(encoded.into_bytes()))
}
fn import_room_version(
    store: &PackfileStorage,
    collection_id: &[u8; 16],
    events: &[OwnedValue],
) -> anyhow::Result<String> {
    let version = if let Some(version) = events
        .iter()
        .find_map(|event| nested_event_string_field(event, "content", "room_version"))
    {
        version.to_owned()
    } else {
        store
            .get_collection_metadata(collection_id)
            .ok()
            .flatten()
            .and_then(|metadata| MatrixRoomExtension::decode_blob(metadata.extension.as_deref()?))
            .and_then(|extension| extension.room_version)
            .with_context(|| {
                format!(
                    "cannot redact imported events for collection {}: room version is unavailable",
                    format_id(collection_id)
                )
            })?
    };
    // Redaction, auth and state resolution all need a version rezzy can
    // classify. Failing here prevents storing events with stripped content,
    // wrong event ids, or unchecked auth.
    if !importer_room_version(&version).is_some_and(MatrixRoomVersion::is_supported) {
        bail!(
            "room version {version:?} is not supported; mtxdb requires v4 or later and recognises MSC3389"
        );
    }
    Ok(version)
}

/// Whether the direct per-file import is enabled. It stays opt-in until the
/// chunk-permutation differential suite proves byte-identical output against
/// the whole-room path; otherwise every import keeps the current behaviour.
fn direct_import_enabled() -> bool {
    std::env::var_os("MTXDB_DIRECT_IMPORT").is_some()
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
#[cfg(test)]
fn import_pdu_events(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    path: &Path,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
) -> anyhow::Result<()> {
    import_pdu_events_with_state_hamts(
        db,
        store,
        state_store,
        path,
        events,
        auth_chain,
        detected_collection,
        collection_override,
        template,
        established_collections,
        true,
    )
}

/// [`import_pdu_events`] with the state-HAMT write setting threaded through.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn import_pdu_events_with_state_hamts(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    path: &Path,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
    write_state_hamts: bool,
) -> anyhow::Result<()> {
    import_pdu_events_with_raw_state_hamts(
        db,
        store,
        state_store,
        path,
        events,
        auth_chain,
        None,
        detected_collection,
        collection_override,
        template,
        established_collections,
        write_state_hamts,
        None,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the PDU import phase owns validation, batching, and state-group work"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the PDU import phase is one sequential pipeline; splitting it would pass state by hand"
)]
#[cfg(test)]
fn import_pdu_events_with_raw(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    path: &Path,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    raw_adjacency: Option<&[rezzy::OwnedMatrixEvent]>,
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
) -> anyhow::Result<()> {
    import_pdu_events_with_raw_state_hamts(
        db,
        store,
        state_store,
        path,
        events,
        auth_chain,
        raw_adjacency,
        detected_collection,
        collection_override,
        template,
        established_collections,
        true,
        None,
    )
}

/// Like `import_pdu_events`, with the state-HAMT write setting threaded
/// through, so `--no-state-hamts` also covers raw JSONL and federation inputs.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn import_pdu_events_with_raw_state_hamts(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    path: &Path,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    raw_adjacency: Option<&[rezzy::OwnedMatrixEvent]>,
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &mut HashSet<[u8; 16]>,
    write_state_hamts: bool,
    mut room_cache: Option<&mut ImportRunCache>,
) -> anyhow::Result<()> {
    if events.is_empty() {
        bail!("no events found in {}", path.display());
    }
    let mut event_count = 0u64;
    let mut skipped = 0u64;
    let mut already_present = 0u64;
    let mut already_present_ids = Vec::with_capacity(3);
    // Input event ids that already have a stored record. New records have no
    // verdict or flags yet, so the writer can skip a point lookup for them.
    let mut existed_before: HashSet<String> = HashSet::new();

    let resolved = resolve_import_collection(
        events,
        detected_collection,
        collection_override,
        template,
        established_collections,
    )?;
    let collection_id = resolved.collection_id;
    // `RoomEvent::from_pdu` fills a missing `room_id` from the batch, but an
    // event that carries a different room must not be analyzed (or stored)
    // under this batch's identity. `resolve_import_collection` already
    // rejects mixed rooms in `events`; auth-chain PDUs can arrive from a
    // federation response too, so check both lists here.
    for event in events.iter().chain(auth_chain) {
        if let Some(room) = event_room_id(event) {
            if room != resolved.canonical_id.as_str() {
                bail!(
                    "event {} belongs to room {room}, not the imported room {}",
                    event_id(event).unwrap_or("<missing event_id>"),
                    resolved.canonical_id
                );
            }
        }
    }
    let room_version = import_room_version(store, &collection_id, events)?;

    let collection_hex = format_id(&collection_id);

    // Ids supplied by this file. Computed before the room is reloaded so the
    // reload can exclude them: the input copies are authoritative for the
    // current batch, and replacing them with their redacted stored form would
    // drop the unredacted auth fields the checks read.
    let current_ids: HashSet<String> = events
        .iter()
        .filter_map(event_id)
        .map(str::to_owned)
        .collect();

    let first: Vec<([u8; 16], String, usize)> = {
        let mut first = Vec::with_capacity(events.len());
        let mut seen_ids: HashMap<[u8; 16], String> = HashMap::with_capacity(events.len());
        for (index, ev) in events.iter().enumerate() {
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
                        format_id(&id_bytes)
                    );
                }
                already_present = already_present.saturating_add(1);
                if already_present_ids.len() < 3 {
                    already_present_ids.push(incoming_event_id.to_owned());
                }
                continue;
            }
            seen_ids.insert(id_bytes, incoming_event_id.to_owned());
            first.push((id_bytes, incoming_event_id.to_owned(), index));
        }
        first
    };

    let probe_ids: Vec<[u8; 16]> = first.iter().map(|(id, _, _)| *id).collect();
    // Ids whose timeline body this file supplies (newly written or already
    // stored). Only these can satisfy a `prev` reference.
    let body_ids: HashSet<String> = first
        .iter()
        .map(|(_, event_id, _)| event_id.clone())
        .collect();
    let existing = store.get_many(&collection_id, &probe_ids)?;
    let mut to_write: Vec<([u8; 16], NodeData)> = Vec::new();
    for ((id_bytes, incoming_event_id, index), existing_opt) in first.into_iter().zip(existing) {
        if let Some(existing_record) = existing_opt {
            reject_cross_record_collision(&id_bytes, &incoming_event_id, &existing_record)?;
            existed_before.insert(incoming_event_id.clone());
            already_present = already_present.saturating_add(1);
            if already_present_ids.len() < 3 {
                already_present_ids.push(incoming_event_id);
            }
        } else {
            // Redaction is only needed for records this import will write. A
            // reimport or a duplicate-heavy batch must not pay to re-redact
            // every event before discovering it is already stored.
            let event_bytes = redacted_event_bytes(&events[index], &room_version)?;
            to_write.push((id_bytes, NodeData::new(event_bytes)));
        }
    }
    // A file that supplies an id an earlier file referenced but did not have
    // can change that earlier event. The run-scoped waiting set answers that
    // in O(file) instead of rescanning every stored event's references. Probe
    // before this file's own edges are recorded, so an id cited only by
    // another event in this file is not a late arrival.
    let direct_identity = matches!(
        &template.record_id_rule.policy,
        FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
    );
    // The run cache carries the direct-path switch so a test can exercise the
    // fast path without mutating a process-global environment variable.
    let direct = room_cache
        .as_deref()
        .map_or_else(direct_import_enabled, |cache| cache.direct);
    let mut local_cache = ImportRunCache::default();
    let (waiting_hits, stored_events_hint) = if direct && direct_identity && !to_write.is_empty() {
        let cache = room_cache.as_deref_mut().unwrap_or(&mut local_cache);
        let waiting = cache.ensure_waiting(
            db,
            store,
            state_store,
            template,
            &collection_id,
            &resolved.canonical_id,
        )?;
        (waiting.hits(current_ids.iter()), waiting.stored_events)
    } else {
        (WaitingHits::default(), 0)
    };
    // The reverse-edge closure of the waiting hits is the set of stored events
    // this file can change. Computing it before the writes keeps the closure
    // to the edges that existed before this file arrived.
    let mut dirty_closure_bailed = false;
    let closure_started = Instant::now();
    let dirty_ids = if direct
        && direct_identity
        && !to_write.is_empty()
        && (!waiting_hits.prev.is_empty() || !waiting_hits.auth.is_empty())
    {
        let cache = room_cache.as_deref_mut().unwrap_or(&mut local_cache);
        match cache.dirty_closure(db, &collection_id, &resolved.canonical_id, &waiting_hits) {
            Ok(ids)
                if stored_events_hint > 0 && ids.len().saturating_mul(2) > stored_events_hint =>
            {
                // Re-judging most of the room costs more than the whole-room
                // path it would replace. Bail before loading bodies or
                // walking; the transpose has already been paid once.
                if std::env::var_os("MTXDB_DIRECT_DEBUG").is_some() {
                    eprintln!(
                        "direct: fallback, |D|={} exceeds half of {stored_events_hint} stored event(s)",
                        ids.len()
                    );
                }
                dirty_closure_bailed = true;
                HashSet::new()
            }
            Ok(ids) => ids,
            Err(error) => {
                eprintln!(
                    "  warning: raw import reverse-edge lookup failed; using the whole-room path: {error:#}"
                );
                dirty_closure_bailed = true;
                HashSet::new()
            }
        }
    } else {
        HashSet::new()
    };
    if std::env::var_os("MTXDB_DIRECT_DEBUG").is_some() {
        eprintln!(
            "direct: closure took {:?} for {} id(s)",
            closure_started.elapsed(),
            dirty_ids.len()
        );
    }
    record_csr_adjacency_with_raw(db, template, events, raw_adjacency)?;

    // Preserve causal order even when input arrives newest-first.
    sort_events_topologically(&mut to_write, events, template)?;

    if resolved.batch_has_create {
        established_collections.insert(collection_id);
        let extension = MatrixRoomExtension::from_events(events);
        let metadata = collection_metadata_for(template, &extension, &resolved.canonical_id);
        store.create_or_put_established(&collection_id, &metadata, &to_write)?;
        event_count = event_count.saturating_add(to_write.len() as u64);
    } else if !to_write.is_empty() {
        store.put_many(&collection_id, &to_write)?;
        event_count = event_count.saturating_add(to_write.len() as u64);
    }
    let new_projections = write_raw_auth_projections(
        store,
        &resolved.canonical_id,
        template,
        &room_version,
        events,
        auth_chain,
    )?;
    // A projection for an event this file also supplies is part of that
    // event's own raw input, not a repair. Only a projection written for an
    // id that was already stored can change an earlier verdict, so only that
    // case disables the direct path.
    let repaired_projections = new_projections
        .iter()
        .filter(|id| existed_before.contains(*id))
        .count();
    // Fold the successfully written bodies into the run-scoped waiting set
    // after the writes, never before: a record that was skipped or rejected by
    // the collision check must stay waiting for a later file.
    if direct && direct_identity {
        if let Some(cache) = room_cache.as_mut() {
            let auth_arrived: HashSet<String> = auth_chain
                .iter()
                .filter(|event| template_node_id(template, event).ok().flatten().is_some())
                .filter_map(event_id)
                .map(str::to_owned)
                .collect();
            let observe_started = Instant::now();
            cache.observe_written(
                db,
                store,
                state_store,
                template,
                &collection_id,
                &resolved.canonical_id,
                &body_ids,
                &auth_arrived,
                events.iter().chain(auth_chain.iter()),
            )?;
            let written_at = Instant::now();
            cache.observe_adjacency(&collection_id, &resolved.canonical_id, events.iter());
            if std::env::var_os("MTXDB_DIRECT_DEBUG").is_some() {
                eprintln!(
                    "direct: observe waiting {:?}, adjacency {:?}",
                    written_at.duration_since(observe_started),
                    written_at.elapsed()
                );
            }
        }
    }
    // A file whose events are all stored already, whose auth chain is empty,
    // and whose projections are all present carries no new information. Every
    // prior verdict, flag and state group is still valid, so skip reloading,
    // re-judging and repairing the whole room. Reimports stay O(file), not
    // O(room).
    //
    // Verify the verdicts actually exist first: a crash between the event
    // write and the verdict write would otherwise leave this file permanently
    // unjudged.
    let skip_judging = to_write.is_empty()
        && auth_chain.is_empty()
        && new_projections.is_empty()
        && all_verdicts_present(db, template, &current_ids)?;
    // A clean append (nothing this file supplies was missing before) only
    // changes this file's own verdicts. Judge it directly against the stored
    // prev state groups and fall back to the whole-room walk on any condition
    // the direct path cannot prove.
    let direct_handled = direct
        && !dirty_closure_bailed
        && !skip_judging
        && write_state_hamts
        && auth_chain.is_empty()
        && !resolved.batch_has_create
        && try_direct_raw_import(
            db,
            store,
            state_store,
            template,
            &collection_id,
            &resolved.canonical_id,
            &room_version,
            events,
            &existed_before,
            &waiting_hits,
            &dirty_ids,
            repaired_projections,
        )?;
    if direct_handled {
        if let Some(cache) = room_cache.as_mut() {
            cache.rooms.remove(&collection_id);
            cache.direct_files = cache.direct_files.saturating_add(1);
        }
        eprintln!("  raw import: direct incremental path");
    }
    if !skip_judging && !direct_handled {
        let phase_loaded = Instant::now();
        // Existing records are part of the state the new batch inherits. Reading
        // them back is the simplest way to let check 5 run across separate files:
        // the walk sees the whole room, not just this file. Events written by
        // this file are excluded from the reloaded set: their input copies are
        // authoritative and still carry the unredacted auth fields.
        let stored_events = match room_cache
            .as_mut()
            .and_then(|cache| cache.rooms.remove(&collection_id))
        {
            Some(cached) => cached,
            None => load_collection_events(store, &collection_id, &resolved.canonical_id)?,
        };
        if import_timings_enabled() {
            eprintln!("  timing: load room {:?}", phase_loaded.elapsed());
        }
        let prior: Vec<OwnedValue> = stored_events
            .into_iter()
            .filter(|event| event_id(event).is_none_or(|id| !current_ids.contains(id)))
            .collect();
        let mut stored_ids: HashSet<String> = prior
            .iter()
            .filter_map(event_id)
            .map(str::to_owned)
            .collect();
        stored_ids.extend(current_ids.iter().cloned());
        // These events are already stored in redacted form. Their content hash
        // intentionally no longer matches, so re-redacting them would strip the
        // auth projection that load_collection_events just laid over them.
        let already_redacted: HashSet<String> = prior
            .iter()
            .filter_map(event_id)
            .map(str::to_owned)
            .collect();
        let mut all_events: Vec<OwnedValue> = prior;
        let mut seen_event_ids: HashSet<String> = all_events
            .iter()
            .filter_map(event_id)
            .map(str::to_owned)
            .collect();
        for event in events.iter().chain(auth_chain) {
            let Some(id) = event_id(event) else {
                continue;
            };
            if seen_event_ids.insert(id.to_owned()) {
                all_events.push(event.clone());
            }
        }

        // An earlier event can only change if this file supplies an id that was
        // missing from the room before it arrived. Appending to a complete DAG
        // supplies none, so only this file's events need judging, verdict
        // writes and flag writes. A late parent still falls back to the full
        // set.
        let supplied: HashSet<&str> = current_ids
            .iter()
            .map(String::as_str)
            .chain(auth_chain.iter().filter_map(event_id))
            .collect();
        // The two reference families use the same presence rule as the
        // waiting set. A `prev` citation needs a timeline body: that is the
        // state base. An `auth` citation is satisfied by a body in either the
        // timeline or the auth-chain store: it is the same event, only the
        // reference kind differs. Projections are overlays, not bodies.
        let mut missing_prev: HashSet<&str> = HashSet::new();
        let mut missing_auth: HashSet<&str> = HashSet::new();
        for event in &all_events {
            let Some(id) = event_id(event) else {
                continue;
            };
            if !already_redacted.contains(id) {
                continue;
            }
            for (field, target) in [
                ("prev_events", &mut missing_prev),
                ("auth_events", &mut missing_auth),
            ] {
                let Some(references) = event.get(field).and_then(OwnedValue::as_array) else {
                    continue;
                };
                for reference in references {
                    if let Some(referenced) = reference.as_str() {
                        if !already_redacted.contains(referenced) {
                            target.insert(referenced);
                        }
                    }
                }
            }
        }
        // A body that already existed before this file is not new information
        // when the same event id is supplied again. Probe only the auth-chain
        // store here: a candidate is by construction absent from the prior
        // timeline, and the current file's timeline writes must remain "new"
        // even though they are already durable when this runs.
        if direct_identity {
            let auth_candidates: HashSet<String> =
                missing_auth.iter().map(|id| (*id).to_owned()).collect();
            let prior_auth = auth_pool_ids(
                state_store,
                template,
                &resolved.canonical_id,
                &auth_candidates,
            )?;
            missing_auth.retain(|id| !prior_auth.contains(*id));
        }
        let clean_append = !supplied
            .iter()
            .any(|id| missing_prev.contains(id) || missing_auth.contains(id));

        // Bodies cited by this room that live only in an earlier federation
        // import's auth-chain collection are part of the local judging
        // universe, exactly as the direct path includes them. They are not
        // timeline rows and not state-DAG nodes: only the findings/check-4
        // index sees them.
        let known: HashSet<String> = all_events
            .iter()
            .filter_map(event_id)
            .map(str::to_owned)
            .collect();
        let mut cited_external: HashSet<String> = HashSet::new();
        for event in &all_events {
            for reference in event_reference_ids(event) {
                if !known.contains(&reference) {
                    cited_external.insert(reference);
                }
            }
        }
        let external_auth: Vec<OwnedValue> = if !direct_identity || cited_external.is_empty() {
            Vec::new()
        } else {
            load_event_bodies(
                store,
                state_store,
                template,
                &resolved.canonical_id,
                &collection_id,
                &cited_external,
            )?
            .0
            .into_values()
            .collect()
        };
        let rewrite_ids: &HashSet<String> = if clean_append {
            &current_ids
        } else {
            &stored_ids
        };
        let judge_only = clean_append.then_some(&current_ids);
        // Prior rejections are still cascade inputs for this file's events even
        // when prior events are not re-judged. Read them from the room flag
        // index in one query rather than re-deriving every stored verdict.
        let prior_rejected: HashSet<String> = if clean_append {
            RoomFlagIndex::new(resolved.canonical_id.as_bytes())
                .select(db, &FlagQuery::has(Flag::Rejected))
                .unwrap_or_default()
                .into_iter()
                .filter_map(|id| String::from_utf8(id).ok())
                .collect()
        } else {
            HashSet::new()
        };

        let room_events: Vec<RoomEvent> = all_events
            .iter()
            .chain(external_auth.iter())
            .filter_map(|event| RoomEvent::from_pdu(event, &resolved.canonical_id))
            .collect();
        let synthetic = RoomExport {
            server: "raw-import".to_owned(),
            room_id: resolved.canonical_id.clone(),
            events: room_events,
            timeline: false,
        };
        let phase_redact = Instant::now();
        let redacted = export_import::redactions_for_export(
            &synthetic.events,
            &room_version,
            &already_redacted,
        );
        // The findings pass and the state walk both read the same lean events.
        // On a clean append only this file's events and their transitive auth
        // closure are judged, so do not rebuild a lean event for the whole room.
        let needed_lean: Option<HashSet<String>> = judge_only.map(|seeds| {
            let by_id: HashMap<&str, &OwnedValue> = all_events
                .iter()
                .filter_map(|event| event_id(event).map(|id| (id, event)))
                .collect();
            let mut needed: HashSet<String> = seeds.iter().cloned().collect();
            let mut stack: Vec<String> = seeds.iter().cloned().collect();
            // Check 5 looks up every entry of the state before the event, and
            // those entries are state events. Include all of them so an entry
            // does not resolve to an empty lean slot.
            for event in &all_events {
                if event.get("state_key").is_some() {
                    if let Some(id) = event_id(event) {
                        needed.insert(id.to_owned());
                    }
                }
            }
            if let Some(create) = all_events.iter().find(|event| {
                event.get("type").and_then(OwnedValue::as_str) == Some("m.room.create")
                    && event.get("state_key").and_then(OwnedValue::as_str) == Some("")
            }) {
                if let Some(id) = event_id(create) {
                    if needed.insert(id.to_owned()) {
                        stack.push(id.to_owned());
                    }
                }
            }
            while let Some(id) = stack.pop() {
                let Some(event) = by_id.get(id.as_str()) else {
                    continue;
                };
                let Some(auth_events) = event.get("auth_events").and_then(OwnedValue::as_array)
                else {
                    continue;
                };
                for auth in auth_events {
                    if let Some(auth_id) = auth.as_str() {
                        if needed.insert(auth_id.to_owned()) {
                            stack.push(auth_id.to_owned());
                        }
                    }
                }
            }
            needed
        });
        let lean = if let Some(needed) = &needed_lean {
            export_import::lean_redacted_events_for(
                &synthetic.events,
                &redacted,
                &synthetic.room_id,
                |id| needed.contains(id),
            )
        } else {
            export_import::lean_redacted_events(&synthetic.events, &redacted, &synthetic.room_id)
        };
        report_import_phase("redaction/lean", phase_redact.elapsed());
        let phase_findings = Instant::now();
        let mut findings = export_import::derived_auth_findings_for_raw_with_inputs(
            &synthetic,
            &room_version,
            &already_redacted,
            judge_only,
            &prior_rejected,
            &redacted,
            &lean,
        )?;
        let mut rejected = export_import::rejected_event_ids(&synthetic, &findings);
        rejected.extend(prior_rejected.iter().cloned());
        report_import_phase("auth findings", phase_findings.elapsed());
        let phase_walk = Instant::now();
        let mut walk_rejections = HashMap::new();
        let (computation, unchecked) = compute_import_auth_before(
            &all_events,
            &[],
            &synthetic.events,
            &synthetic.room_id,
            &room_version,
            StateResVersion::from_room_version(&room_version),
            &mut rejected,
            &mut walk_rejections,
            &already_redacted,
            judge_only,
            Some(&lean),
            &HashMap::new(),
        );
        export_import::apply_walk_rejections(&synthetic, &mut findings, &walk_rejections);
        report_import_phase("state/auth walk", phase_walk.elapsed());
        if unchecked > 0 {
            eprintln!(
                "  warning: {unchecked} event(s) could not be checked against the state before them"
            );
        }
        for warning in findings.current_warnings(&current_ids) {
            eprintln!("  warning: {warning}");
        }
        let rejected_total = synthetic
            .events
            .iter()
            .zip(&findings.forced)
            .filter(|(event, forced)| forced.is_some() && current_ids.contains(&event.event_id))
            .count();
        if rejected_total > 0 {
            eprintln!("  derived rejections: {rejected_total} events");
        }

        let existing_ids: HashSet<&str> = already_redacted
            .iter()
            .map(String::as_str)
            .chain(existed_before.iter().map(String::as_str))
            .collect();
        let phase_verdicts = Instant::now();
        let (changed_earlier, changed_earlier_rejected) = write_raw_verdicts_and_flags(
            db,
            &synthetic.events,
            &findings.forced,
            &stored_ids,
            &current_ids,
            rewrite_ids,
            &existing_ids,
            template,
            &resolved.canonical_id,
        )?;
        if changed_earlier > 0 {
            eprintln!(
                "  updated earlier verdicts: {changed_earlier} events ({changed_earlier_rejected} rejected)"
            );
        }
        report_import_phase("verdicts/flags", phase_verdicts.elapsed());

        if write_state_hamts {
            let phase_publish = Instant::now();
            repair_import_state_groups(
                state_store,
                &all_events,
                &[],
                &resolved.canonical_id,
                &room_version,
                Some(computation),
            )?;
            report_import_phase("state publish (total)", phase_publish.elapsed());
        }

        // Keep the room in memory for the next file in this run. Only stored
        // timeline events are cached: auth-chain-only events must not become
        // flags or verdict rows on a later file.
        if let Some(cache) = room_cache {
            let auth_only: HashSet<&str> = auth_chain
                .iter()
                .filter_map(event_id)
                .filter(|id| !current_ids.contains(*id))
                .collect();
            if auth_only.is_empty() {
                cache.rooms.insert(collection_id, all_events);
            } else {
                let mut entry = all_events;
                entry.retain(|event| event_id(event).is_none_or(|id| !auth_only.contains(id)));
                cache.rooms.insert(collection_id, entry);
            }
        }
    }

    if let Some(room_id) = detected_collection {
        eprintln!("{room_id} ({collection_hex})");
    } else {
        eprintln!("collection {collection_hex}");
    }
    eprintln!("  imported: {event_count} events");
    if skipped > 0 {
        eprintln!("  skipped: {skipped} events (missing event_id)");
    }
    if already_present > 0 {
        let suffix = if already_present > already_present_ids.len() as u64 {
            ", ..."
        } else {
            ""
        };
        eprintln!(
            "  already present: {already_present} [{}{suffix}]",
            already_present_ids.join(", "),
        );
    }

    Ok(())
}

/// The sidecar collection holding unredacted auth-relevant fields, keyed by
/// event node id.
fn raw_auth_projection_collection_id(room_id: &str) -> [u8; 16] {
    derive_collection_id(
        Some(mtxdb::MEMBER_NAMESPACE_INTL),
        format!("raw-auth-projection:{room_id}").as_bytes(),
    )
}

/// Whether redaction preserves a `content` path for this event type and room
/// version. Unknown versions fail closed in rezzy, so every auth field is
/// projected rather than silently skipped.
fn redaction_preserves(event_type: &str, room_version: &str, path: &str) -> bool {
    match redaction_preserved_keys(event_type, room_version) {
        RedactionRule::All => true,
        RedactionRule::Keys(keys) => keys.contains(&path),
        RedactionRule::None => false,
    }
}

/// Fields redaction drops but the auth rules still read.
///
/// Version classification comes from rezzy's redaction table rather than a
/// numeric parse here, so unstable versions (`MSC3389`, `MSC4242`, `NutraCdo`) get
/// the same treatment as their numeric rule sets.
fn raw_auth_projection(event: &OwnedValue, room_version: &str) -> Option<OwnedValue> {
    // Unknown versions are rejected before import. Do not invent an
    // everything-is-unpreserved rule if this helper is called directly.
    if !importer_room_version(room_version).is_some_and(MatrixRoomVersion::is_supported) {
        return None;
    }
    event.get("state_key")?;
    let kind = event.get("type")?.as_str()?;
    let OwnedValue::Object(content) = event.get("content")? else {
        return None;
    };
    let mut projected = Object::new();
    match kind {
        "m.room.member" => {
            // The auth rule reads the signed token, not display_name or the
            // surrounding signatures.
            if !redaction_preserves(kind, room_version, "third_party_invite.signed") {
                if let Some(OwnedValue::Object(third_party)) = content.get("third_party_invite") {
                    if let Some(signed) = third_party.get("signed") {
                        let mut token = Object::new();
                        token.insert("signed".to_owned(), signed.clone());
                        projected
                            .insert("third_party_invite".to_owned(), OwnedValue::Object(token));
                    }
                }
            }
            if !redaction_preserves(kind, room_version, "join_authorised_via_users_server") {
                if let Some(value) = content.get("join_authorised_via_users_server") {
                    projected.insert("join_authorised_via_users_server".to_owned(), value.clone());
                }
            }
        }
        "m.room.create" => {
            if !redaction_preserves(kind, room_version, "m.federate") {
                if let Some(value) = content.get("m.federate") {
                    projected.insert("m.federate".to_owned(), value.clone());
                }
            }
        }
        "m.room.power_levels" => {
            if !redaction_preserves(kind, room_version, "invite") {
                if let Some(value) = content.get("invite") {
                    projected.insert("invite".to_owned(), value.clone());
                }
            }
        }
        "m.room.third_party_invite" => {
            // v11 redaction still strips this event's entire content, so its
            // public keys need a projection in every room version.
            for key in ["public_key", "public_keys"] {
                if !redaction_preserves(kind, room_version, key) {
                    if let Some(value) = content.get(key) {
                        projected.insert(key.to_owned(), value.clone());
                    }
                }
            }
        }
        _ => return None,
    }
    (!projected.is_empty()).then_some(OwnedValue::Object(projected))
}

/// Whether the event's declared content hash matches its current content.
///
/// Events without `hashes.sha256` cannot be verified here, so they get no auth
/// projection. The event itself is still stored and judged in its redacted
/// form, which is the spec's fallback.
fn content_hash_verifies(event: &OwnedValue, room_version: &str) -> bool {
    let Some(expected) = event
        .get("hashes")
        .and_then(|hashes| hashes.get("sha256"))
        .and_then(OwnedValue::as_str)
    else {
        return false;
    };
    rezzy::compute_content_hash(event, room_version).is_ok_and(|computed| computed == expected)
}

fn write_raw_auth_projections(
    store: &PackfileStorage,
    room_id: &str,
    template: &CollectionTemplate,
    room_version: &str,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
) -> anyhow::Result<Vec<String>> {
    // Never guess at an unknown version. The establishment validator rejects
    // these before an import, but the projection writer is also called
    // directly by tests and future callers.
    if !importer_room_version(room_version).is_some_and(MatrixRoomVersion::is_supported) {
        anyhow::bail!(
            "room version {room_version:?} is not supported for auth projection; mtxdb requires v4 or later"
        );
    }
    let mut seen = HashSet::new();
    let mut records = Vec::new();
    for event in events.iter().chain(auth_chain) {
        // Cheap field check first: only a handful of event types carry auth
        // fields redaction can drop, and content hashing every stored event on
        // every file is pure waste when most events can never project.
        let Some(projection) = raw_auth_projection(event, room_version) else {
            continue;
        };
        // A projection is auth input. It may only be taken from content whose
        // hash verifies; otherwise a forged event could seed the fields that
        // later files overlay onto their redacted copy.
        if !content_hash_verifies(event, room_version) {
            continue;
        }
        // auth_chain-only events are projected too: step 3 will use them as
        // edge and state providers without storing them as timeline events.
        let Some(node) = template_node_id(template, event)? else {
            continue;
        };
        if !seen.insert(node) {
            continue;
        }
        let Some(id) = event_id(event) else {
            continue;
        };
        records.push((
            node,
            id.to_owned(),
            NodeData::new(projection.encode().into_bytes().into()),
        ));
    }
    if records.is_empty() {
        return Ok(Vec::new());
    }
    let collection = raw_auth_projection_collection_id(room_id);
    let ids: Vec<[u8; 16]> = records.iter().map(|(id, _, _)| *id).collect();
    let existing = store.get_many(&collection, &ids)?;
    let mut to_write = Vec::with_capacity(records.len());
    let mut written_ids = Vec::with_capacity(records.len());
    for ((node, id, data), existing) in records.into_iter().zip(existing) {
        if let Some(existing) = existing {
            if existing.bytes != data.bytes {
                anyhow::bail!(
                    "auth projection for event {id} conflicts with the stored projection; event ids are content hashes from v4 on, so this is corrupted data"
                );
            }
            continue;
        }
        to_write.push((node, data));
        written_ids.push(id);
    }
    if !to_write.is_empty() {
        let metadata = CollectionMetadata {
            member_namespace: Some(mtxdb::MEMBER_NAMESPACE_INTL),
            collection_canonical_id: format!("raw-auth-projection:{room_id}").into_bytes(),
            record_id_rule: template.record_id_rule.clone(),
            payload: template.payload.clone(),
            extension: None,
            role: Some("raw-auth-projection".to_owned()),
            schema: Some("matrix.event.auth-projection.v1".to_owned()),
        };
        store
            .create_or_put_established(&collection, &metadata, &to_write)
            .context("storing raw auth projections")?;
    }
    Ok(written_ids)
}

fn overlay_auth_projection(event: &mut OwnedValue, projection: OwnedValue) {
    let OwnedValue::Object(fields) = event else {
        return;
    };
    let OwnedValue::Object(projected) = projection else {
        return;
    };
    if !matches!(fields.get("content"), Some(OwnedValue::Object(_))) {
        fields.insert("content".to_owned(), OwnedValue::Object(Object::new()));
    }
    let Some(OwnedValue::Object(content)) = fields.get_mut("content") else {
        return;
    };
    for (key, value) in projected {
        content.insert(key, value);
    }
}

/// The event ids an event cites in `prev_events` and `auth_events`.
fn event_reference_ids(event: &OwnedValue) -> Vec<String> {
    let mut ids = Vec::new();
    for field in ["prev_events", "auth_events"] {
        let Some(references) = event.get(field).and_then(OwnedValue::as_array) else {
            continue;
        };
        for reference in references {
            let id = match reference {
                OwnedValue::String(id) => Some(id.as_str()),
                OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                _ => None,
            };
            if let Some(id) = id {
                ids.push(id.to_owned());
            }
        }
    }
    ids
}

/// Load event bodies by event id from the room's timeline collection or its
/// auth-chain collection, overlaying the auth projection on either one.
///
/// Returns the bodies and, separately, the ids whose body came from the
/// timeline collection. Only that second set is a stored *redacted* timeline
/// row; an auth-chain body is stored raw and must still be redacted when its
/// content hash fails. A projection is an overlay, never a body source.
fn load_event_bodies(
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    template: &CollectionTemplate,
    room_id: &str,
    collection_id: &[u8; 16],
    ids: &HashSet<String>,
) -> anyhow::Result<(HashMap<String, OwnedValue>, HashSet<String>)> {
    let mut out = HashMap::with_capacity(ids.len());
    let mut timeline_ids = HashSet::with_capacity(ids.len());
    if ids.is_empty() {
        return Ok((out, timeline_ids));
    }
    let mut ordered: Vec<(String, [u8; 16])> = Vec::with_capacity(ids.len());
    for id in ids {
        ordered.push((id.clone(), template_event_id_node_id(template, id)?));
    }
    let nodes: Vec<[u8; 16]> = ordered.iter().map(|(_, node)| *node).collect();
    let timeline = store.get_many(collection_id, &nodes)?;
    let auth_collection =
        derive_collection_id(Some(mtxdb::MEMBER_NAMESPACE_AUTH), room_id.as_bytes());
    let auth = state_store.get_many(&auth_collection, &nodes)?;
    let projections = store.get_many(&raw_auth_projection_collection_id(room_id), &nodes)?;
    for (((id, _node), timeline), (auth, projection)) in ordered
        .into_iter()
        .zip(timeline)
        .zip(auth.into_iter().zip(projections))
    {
        let from_timeline = timeline.is_some();
        let Some(record) = timeline.or(auth) else {
            continue;
        };
        let Ok(mut value) = OwnedValue::parse_bytes(&record.bytes) else {
            continue;
        };
        if let Some(projection) = projection {
            if let Ok(projection) = OwnedValue::parse_bytes(&projection.bytes) {
                overlay_auth_projection(&mut value, projection);
            }
        }
        if from_timeline {
            timeline_ids.insert(id.clone());
        }
        out.insert(id, value);
    }
    Ok((out, timeline_ids))
}

fn state_set_from_entries(entries: Vec<(String, String, String)>) -> StateSet {
    let mut set = StateSet::new();
    for (event_type, state_key, event_id) in entries {
        set.set(&event_type, &state_key, event_id);
    }
    set
}

/// Try to import a clean append without reloading or re-walking the room.
///
/// Returns `true` when the file was fully handled. `false` means some
/// precondition the direct path cannot prove holds (a late parent, a merge
/// that needs the room's auth closure, a missing body, an unchecked event), so
/// the caller must use the whole-room path. The direct path never writes a
/// partial result: all judging happens before any verdict or flag is staged.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the direct path owns seeding, local judging, and delta writes"
)]
/// Whether every id already has a stored verdict record.
///
/// The reimport fast path skips judging when a file has no new bytes. That is
/// only safe if the previous import actually finished writing verdicts; a
/// crash between the event write and the verdict write must not make the file
/// permanently unjudged.
fn all_verdicts_present(
    db: &Database,
    template: &CollectionTemplate,
    ids: &HashSet<String>,
) -> anyhow::Result<bool> {
    if ids.is_empty() {
        return Ok(true);
    }
    if !matches!(
        &template.record_id_rule.policy,
        FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
    ) {
        return Ok(false);
    }
    let mut by_collection: HashMap<[u8; 16], Vec<NodeId>> = HashMap::new();
    for id in ids {
        let node = template_event_id_node_id(template, id)?;
        by_collection
            .entry(event_verdict_collection_id(&node))
            .or_default()
            .push(node);
    }
    let store = db.pool(ShardType::EventDag);
    for (collection, nodes) in by_collection {
        if store
            .get_many(&collection, &nodes)?
            .iter()
            .any(Option::is_none)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the direct path owns seeding, local judging, and delta writes"
)]
fn try_direct_raw_import(
    db: &Database,
    store: &PackfileStorage,
    state_store: &PackfileStorage,
    template: &CollectionTemplate,
    collection_id: &[u8; 16],
    room_id: &str,
    room_version: &str,
    events: &[OwnedValue],
    existed_before: &HashSet<String>,
    waiting_hits: &WaitingHits,
    dirty_ids: &HashSet<String>,
    repaired_projections: usize,
) -> anyhow::Result<bool> {
    if !matches!(
        &template.record_id_rule.policy,
        FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
    ) {
        return Ok(false);
    }
    let Some(state_version) = StateResVersion::from_room_version(room_version) else {
        return Ok(false);
    };
    let direct_debug = std::env::var_os("MTXDB_DIRECT_DEBUG").is_some();
    let direct_started = Instant::now();
    // A file that repairs an existing event's auth projection can change that
    // event's verdict, so the whole-room path (which re-judges every id in the
    // file) must handle it even when new events are also present. A projection
    // for a new event is part of that event's own input and does not count.
    if repaired_projections > 0 {
        if direct_debug {
            eprintln!("direct: fallback, {repaired_projections} repaired auth projection(s)");
        }
        return Ok(false);
    }
    let mut new_ids: HashSet<String> = HashSet::new();
    let mut new_events: Vec<OwnedValue> = Vec::new();
    for event in events {
        let Some(id) = event_id(event) else {
            return Ok(false);
        };
        if existed_before.contains(id) || !new_ids.insert(id.to_owned()) {
            continue;
        }
        new_events.push(event.clone());
    }
    if new_events.is_empty() {
        return Ok(false);
    }
    // D is the reverse-edge closure the caller computed from the waiting hits:
    // the direct citers of a late body and their `prev` descendants. Every id
    // in D was already stored, so its timeline body (with any auth projection)
    // is the authoritative input, exactly as the whole-room path reloads it.
    let dirty_ids: HashSet<String> = dirty_ids
        .iter()
        .filter(|id| !new_ids.contains(*id))
        .cloned()
        .collect();
    let mut dirty_events: Vec<OwnedValue> = Vec::new();
    let mut dirty_timeline: HashSet<String> = HashSet::new();
    if !dirty_ids.is_empty() {
        let (loaded, loaded_timeline) = load_event_bodies(
            store,
            state_store,
            template,
            room_id,
            collection_id,
            &dirty_ids,
        )?;
        // A reverse row can name an auth-chain-only body, which has no verdict
        // and cannot seed a state group. The whole-room path never re-judges
        // those either, so fall back rather than invent a timeline event.
        if loaded_timeline.len() != dirty_ids.len() {
            if direct_debug {
                eprintln!(
                    "direct: fallback, {} of {} dirty ids have no timeline body",
                    dirty_ids.len().saturating_sub(loaded_timeline.len()),
                    dirty_ids.len()
                );
            }
            return Ok(false);
        }
        let mut ordered: Vec<&String> = dirty_ids.iter().collect();
        ordered.sort();
        for id in ordered {
            if let Some(event) = loaded.get(id) {
                dirty_events.push(event.clone());
            }
        }
        dirty_timeline = loaded_timeline;
    }
    if direct_debug {
        eprintln!(
            "direct: loaded {} dirty body(ies) in {:?}",
            dirty_events.len(),
            direct_started.elapsed()
        );
    }
    let mut rejudge_ids = new_ids.clone();
    rejudge_ids.extend(dirty_ids.iter().cloned());
    // The state walk needs both the new file and the stored events it can
    // change. The findings pass sees the same union through the export below.
    let mut walked_events = new_events.clone();
    walked_events.extend(dirty_events);

    let aux = AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    // Every citation must already be present, or be in this file.
    let mut external_refs: HashSet<String> = HashSet::new();
    for event in &walked_events {
        for reference in event_reference_ids(event) {
            if !rejudge_ids.contains(&reference) {
                external_refs.insert(reference);
            }
        }
    }
    // Cited events that no store holds stay absent from the local universe.
    // A missing `prev` leaves its child unseeded (the topo sort blocks it); a
    // missing `auth` makes the findings pass report `MissingAuthEvent`. Both
    // match what the whole-room path derives from the same room. A body held
    // only by a previous federation import's auth-chain collection is a valid
    // auth provider, not a gap.
    let (external, external_timeline) = load_event_bodies(
        store,
        state_store,
        template,
        room_id,
        collection_id,
        &external_refs,
    )?;

    // Load the after-state of every parent outside this file. A parent with no
    // stored group is left unseeded: the topological sort then blocks it and
    // its descendants, so they get no group, exactly as the whole-room path
    // leaves a gapped ancestry unmapped.
    let mut seed_ids: HashSet<String> = HashSet::new();
    for event in &walked_events {
        let Some(prevs) = event.get("prev_events").and_then(OwnedValue::as_array) else {
            continue;
        };
        for reference in prevs {
            let id = match reference {
                OwnedValue::String(id) => Some(id.as_str()),
                OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                _ => None,
            };
            let Some(id) = id else { continue };
            if !rejudge_ids.contains(id) {
                seed_ids.insert(id.to_owned());
            }
        }
    }
    let seed_ids: Vec<String> = seed_ids.into_iter().collect();
    let keys: Vec<&[u8]> = seed_ids.iter().map(String::as_bytes).collect();
    let loaded = aux.get_many(&keys)?;
    let mut seeds: HashMap<String, Arc<SharedState>> = HashMap::with_capacity(seed_ids.len());
    let mut unseeded = 0_usize;
    for (id, value) in seed_ids.iter().zip(loaded) {
        let Some(value) = value else {
            // Referenced but never resolved; no state group to seed.
            unseeded = unseeded.saturating_add(1);
            continue;
        };
        if !valid_state_group_id(&value) {
            return Ok(false);
        }
        let mut group = [0u8; STATE_GROUP_ID_LENGTH];
        group.copy_from_slice(&value);
        let Some(entries) =
            crate::state_hamt::load_state_group_entries(state_store, room_id, &group)?
        else {
            return Ok(false);
        };
        let set = state_set_from_entries(entries);
        // A stored group is only published by the whole-room path when its
        // computation was complete (the writer suppresses publication when
        // `incomplete` is non-empty), so a stored group is a sound seed.
        seeds.insert(id.clone(), SharedState::new(set, group, true));
    }
    if direct_debug && unseeded > 0 {
        eprintln!("direct: {unseeded} prev id(s) have no stored state group to seed");
    }

    // Auth-relevant state keys read by check 5 for this file's events. Only
    // these state bodies are needed from the seed; everything else in the
    // state set is carried through the walk but never materialized as an
    // event body in this file.
    let mut needed_keys: HashSet<(String, String)> = HashSet::new();
    needed_keys.insert(("m.room.create".to_owned(), String::new()));
    let empty_content = OwnedValue::Object(Object::default());
    for event in &walked_events {
        let event_type = event.get("type").and_then(OwnedValue::as_str).unwrap_or("");
        let sender = event
            .get("sender")
            .and_then(OwnedValue::as_str)
            .unwrap_or("");
        let state_key = event.get("state_key").and_then(OwnedValue::as_str);
        let content = event.get("content").unwrap_or(&empty_content);
        needed_keys.extend(
            auth_types_for_event_ref(
                event_type,
                sender,
                state_key,
                content,
                state_version,
                room_version,
            )
            .into_iter()
            .map(|(event_type, state_key)| (event_type.to_owned(), state_key.to_owned())),
        );
    }
    // Seed state bodies are different: a published state group must name
    // events that exist somewhere. If one cannot be loaded the store is
    // inconsistent, so fall back rather than judge against a partial state.
    let mut seed_refs: HashSet<String> = HashSet::new();
    for seed in seeds.values() {
        for (key, event_id) in &seed.set.entries {
            if needed_keys.contains(key) && !rejudge_ids.contains(event_id) {
                seed_refs.insert(event_id.clone());
            }
        }
    }

    // Build the local event universe: this file, its cited events, and the
    // seed state entries the auth rules read.
    let mut local: HashMap<String, OwnedValue> =
        HashMap::with_capacity(walked_events.len().saturating_add(external_refs.len()));
    for event in &walked_events {
        if let Some(id) = event_id(event) {
            local.insert(id.to_owned(), event.clone());
        }
    }
    for (id, event) in external {
        local.entry(id).or_insert(event);
    }
    let missing: HashSet<String> = seed_refs
        .iter()
        .filter(|id| !local.contains_key(*id))
        .cloned()
        .collect();
    let mut seed_timeline: HashSet<String> = HashSet::new();
    if !missing.is_empty() {
        let (loaded, loaded_timeline) = load_event_bodies(
            store,
            state_store,
            template,
            room_id,
            collection_id,
            &missing,
        )?;
        if loaded.len() != missing.len() {
            return Ok(false);
        }
        for (id, event) in loaded {
            local.insert(id, event);
        }
        seed_timeline = loaded_timeline;
    }

    // Only a timeline body is stored in the redacted timeline form. An
    // auth-chain body is raw and must still be redacted when its hash fails.
    let mut already_redacted: HashSet<String> = external_timeline;
    already_redacted.extend(dirty_timeline);
    already_redacted.extend(seed_timeline);
    already_redacted.retain(|id| !new_ids.contains(id));
    let mut local_events: Vec<OwnedValue> = Vec::with_capacity(local.len());
    // Sort by event id: `local` is a hash map, and findings warnings are
    // emitted in encounter order. A stable universe keeps diagnostics stable
    // across runs and across gate-on/gate-off comparisons.
    let mut local_ids: Vec<&String> = local.keys().collect();
    local_ids.sort();
    for id in local_ids {
        local_events.push(local[id].clone());
    }
    let local_rooms: Vec<RoomEvent> = local_events
        .iter()
        .filter_map(|event| RoomEvent::from_pdu(event, room_id))
        .collect();
    if local_rooms.len() != local_events.len() {
        return Ok(false);
    }
    let redacted =
        export_import::redactions_for_export(&local_rooms, room_version, &already_redacted);
    let lean = export_import::lean_redacted_events(&local_rooms, &redacted, room_id);
    let local_export = RoomExport {
        server: "raw-import".to_owned(),
        room_id: room_id.to_owned(),
        events: local_rooms,
        timeline: false,
    };

    // Prior rejections are cascade inputs. The direct path depends on them, so
    // an unreadable flag index must fall back rather than look unrejected.
    let mut prior_rejected: HashSet<String> =
        match RoomFlagIndex::new(room_id.as_bytes()).select(db, &FlagQuery::has(Flag::Rejected)) {
            Ok(ids) => ids
                .into_iter()
                .filter_map(|id| String::from_utf8(id).ok())
                .collect(),
            Err(error) => {
                if direct_debug {
                    eprintln!("direct: fallback, rejection flags unreadable: {error}");
                }
                return Ok(false);
            }
        };
    // A re-judged event must start from a clean slate; its old rejection is
    // not a cascade input. Rejections outside the dirty set stay seeds.
    prior_rejected.retain(|id| !rejudge_ids.contains(id));

    let judged_at = Instant::now();
    if direct_debug {
        eprintln!(
            "direct: built a {}-event universe in {:?}",
            local_events.len(),
            judged_at.duration_since(direct_started)
        );
    }
    // Check 4 (state-before-event) and the hash/signature audits run over this
    // file against the local universe. The state walk contributes check 5
    // only, through the same helper the whole-room path uses.
    let mut findings = export_import::derived_auth_findings_for_raw_with_inputs(
        &local_export,
        room_version,
        &already_redacted,
        Some(&rejudge_ids),
        &prior_rejected,
        &redacted,
        &lean,
    )?;
    let mut rejected = export_import::rejected_event_ids(&local_export, &findings);
    rejected.extend(prior_rejected.iter().cloned());
    let walk_started = Instant::now();
    if direct_debug {
        eprintln!(
            "direct: findings took {:?}",
            walk_started.duration_since(judged_at)
        );
    }
    let mut walk_rejections: HashMap<String, (u8, String)> = HashMap::new();
    let (computation, unchecked) = compute_import_auth_before(
        &walked_events,
        &[],
        &local_export.events,
        room_id,
        room_version,
        Some(state_version),
        &mut rejected,
        &mut walk_rejections,
        &already_redacted,
        Some(&rejudge_ids),
        Some(&lean),
        &seeds,
    );
    if direct_debug {
        eprintln!(
            "direct: walk took {:?} for {} judged id(s)",
            walk_started.elapsed(),
            rejudge_ids.len()
        );
    }
    export_import::apply_walk_rejections(&local_export, &mut findings, &walk_rejections);
    for warning in findings.current_warnings(&new_ids) {
        eprintln!("  warning: {warning}");
    }
    if unchecked > 0 {
        eprintln!(
            "  warning: {unchecked} event(s) could not be checked against the state before them"
        );
    }
    // A merge that fell back because the walk was missing part of the auth
    // closure is not provable with this file's event pool; the whole-room path
    // has the full set.
    if !computation.incomplete.is_empty() {
        if direct_debug {
            eprintln!(
                "direct: fallback, incomplete merges: {:?}",
                computation.incomplete
            );
        }
        return Ok(false);
    }
    if direct_debug {
        eprintln!(
            "direct: |F|={}, |D|={}, waited prev={} auth={}, |mapped|={}, seeds={}, unchecked={unchecked}, gapped={}, unmapped={}",
            new_events.len(),
            dirty_ids.len(),
            waiting_hits.prev.len(),
            waiting_hits.auth.len(),
            computation.groups.len(),
            seeds.len(),
            computation.gapped.len(),
            rejudge_ids.len().saturating_sub(computation.groups.len()),
        );
    }

    // Publish content-addressed nodes and records before any mapping points at
    // them, then publish the event -> group mappings.
    let publish_started = Instant::now();
    if !computation.instances.is_empty() {
        materialize_state_hamts(state_store, room_id, &computation.instances)?;
    }
    let mappings: Vec<(String, [u8; STATE_GROUP_ID_LENGTH])> = computation
        .groups
        .iter()
        .map(|(id, group)| (id.clone(), *group))
        .collect();
    let entries: Vec<(&[u8], &[u8])> = mappings
        .iter()
        .map(|(id, group)| (id.as_bytes(), group.as_slice()))
        .collect();
    if !entries.is_empty() {
        aux.put_many(&entries)
            .context("persisting direct-import state-group mappings")?;
    }
    if direct_debug {
        eprintln!("direct: publish took {:?}", publish_started.elapsed());
    }

    // `existing_ids` is D only: those records already have a verdict whose
    // soft-fail claim must survive this re-judge. The new file's events were
    // just written and have nothing to read back.
    let existing_ids: HashSet<&str> = dirty_ids.iter().map(String::as_str).collect();
    let write_started = Instant::now();
    write_raw_verdicts_and_flags(
        db,
        &local_export.events,
        &findings.forced,
        &rejudge_ids,
        &new_ids,
        &rejudge_ids,
        &existing_ids,
        template,
        room_id,
    )?;
    if direct_debug {
        eprintln!(
            "direct: write took {:?}, total {:?}",
            write_started.elapsed(),
            direct_started.elapsed()
        );
    }
    Ok(true)
}

fn load_collection_events(
    store: &PackfileStorage,
    collection_id: &[u8; 16],
    room_id: &str,
) -> anyhow::Result<Vec<OwnedValue>> {
    let projections = raw_auth_projection_collection_id(room_id);
    // One sequential scan beats one random lookup per stored event. Only
    // events with a non-empty projection have a record, so this map is small.
    let mut projected = HashMap::new();
    for record in store.scan_collection(&projections)? {
        let (node, data) = record?;
        if let Ok(projection) = OwnedValue::parse_bytes(&data.bytes) {
            projected.insert(node, projection);
        }
    }
    let mut events = Vec::new();
    for record in store.scan_collection(collection_id)? {
        let (node, data) = record?;
        let Ok(mut value) = OwnedValue::parse_bytes(&data.bytes) else {
            continue;
        };
        if event_id(&value).is_none() {
            continue;
        }
        if let Some(projection) = projected.get(&node).cloned() {
            overlay_auth_projection(&mut value, projection);
        }
        events.push(value);
    }
    Ok(events)
}

/// One raw event's identity and derived verdict, grouped for flag writing.
type RawFlagRow = (String, [u8; 16], Verdict, bool);

/// Store the raw import's derived verdicts and mirror them into the room's
/// flag bitmap, so rejected events are countable and selectable.
///
/// Every event already stored in the collection is rewritten, not just the
/// current file's events. A later file can complete a base that an earlier
/// file lacked, which changes an earlier event's derived verdict; leaving the
/// old verdict and flags behind would describe state the state groups no
/// longer use.
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "the writer needs the stored events, their derived verdicts, and the room identity"
)]
fn write_raw_verdicts_and_flags(
    db: &Database,
    all_events: &[RoomEvent],
    forced: &[Option<u8>],
    stored_ids: &HashSet<String>,
    current_ids: &HashSet<String>,
    rewrite_ids: &HashSet<String>,
    existing_ids: &HashSet<&str>,
    template: &CollectionTemplate,
    canonical_id: &str,
) -> anyhow::Result<(usize, usize)> {
    if !matches!(
        &template.record_id_rule.policy,
        FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
    ) {
        return Ok((0, 0));
    }
    let forced_by_id: HashMap<&str, Option<u8>> = all_events
        .iter()
        .zip(forced)
        .map(|(event, code)| (event.event_id.as_str(), *code))
        .collect();

    let mut changed_earlier = 0_usize;
    let mut changed_earlier_rejected = 0_usize;
    let mut by_room: BTreeMap<String, Vec<RawFlagRow>> = BTreeMap::new();
    for event in all_events {
        if !stored_ids.contains(&event.event_id) || !rewrite_ids.contains(&event.event_id) {
            continue;
        }
        let Some(node) = template_node_id(template, &event.pdu)? else {
            continue;
        };
        let room = event
            .pdu
            .get("room_id")
            .and_then(OwnedValue::as_str)
            .unwrap_or(canonical_id)
            .to_owned();
        let forced = forced_by_id.get(event.event_id.as_str()).copied().flatten();
        // A record written by this file has no verdict or flags yet, so there
        // is nothing to read back and compare against.
        let existing = if existing_ids.contains(event.event_id.as_str()) {
            db.pool(ShardType::EventDag)
                .get(&event_verdict_collection_id(&node), &node)?
                .map(|data| decode_verdict(&data.bytes).map(|(_, verdict)| verdict))
                .transpose()?
        } else {
            None
        };
        let mut verdict = Verdict {
            rejection: forced,
            outlier: forced.is_some(),
            ..Verdict::default()
        };
        if forced.is_none() {
            // The stored event is redacted, so its source soft-fail claim is
            // gone by the next file. Keep the verdict written by the file that
            // supplied the event.
            verdict.soft_fail = if current_ids.contains(&event.event_id) {
                (event.metadata.soft_failed == Some(true)).then(|| SoftFailCode::Imported.to_u8())
            } else {
                existing.as_ref().and_then(|verdict| verdict.soft_fail)
            };
        }
        let write_verdict = existing.as_ref() != Some(&verdict);
        by_room.entry(room).or_default().push((
            event.event_id.clone(),
            node,
            verdict,
            write_verdict,
        ));
    }

    for (room, rows) in by_room {
        let flag_index = RoomFlagIndex::new(room.as_bytes());
        let auth = RoomAuth::new(&room);
        let ids: Vec<&str> = rows.iter().map(|row| row.0.as_str()).collect();
        let allocated = auth.short_ids(db, &ids)?;
        let mut changes = Vec::with_capacity(rows.len().saturating_mul(4));
        let tx = db.begin_transaction();
        for ((event_id, node, verdict, write_verdict), short) in rows.into_iter().zip(allocated) {
            // Only overwrite when the derived verdict changed; empty records
            // for unchanged events would grow the store on every reimport.
            if write_verdict {
                if !current_ids.contains(&event_id) {
                    changed_earlier = changed_earlier.saturating_add(1);
                    if verdict.rejection.is_some() {
                        changed_earlier_rejected = changed_earlier_rejected.saturating_add(1);
                    }
                }
                tx.put(
                    ShardType::EventDag,
                    event_verdict_collection_id(&node),
                    node,
                    &NodeData::new(encode_verdict(event_id.as_bytes(), &verdict)?.into()),
                )?;
            }

            for (flag, wanted) in [
                (Flag::Present, true),
                (Flag::Rejected, verdict.rejection.is_some()),
                (Flag::Outlier, verdict.outlier),
                (Flag::SoftFailed, verdict.soft_fail.is_some()),
            ] {
                changes.push(if wanted {
                    FlagChange::set(short, flag)
                } else {
                    FlagChange::clear(short, flag)
                });
            }
        }
        // `stage` compares each touched chunk before writing; a reimport thus
        // stages no chunk or metadata writes even though it builds this list.
        flag_index.stage(&tx, &changes)?;
        tx.commit()?;
    }
    Ok((changed_earlier, changed_earlier_rejected))
}

fn sort_events_topologically(
    to_write: &mut [([u8; 16], NodeData)],
    events: &[OwnedValue],
    template: &CollectionTemplate,
) -> anyhow::Result<()> {
    // A single record has no order to preserve; an empty batch has nothing to
    // sort. Avoid building a full-room DAG for the common incremental append.
    if to_write.len() <= 1 {
        return Ok(());
    }
    let Some(order) = topological_event_order_with_template(events, template)? else {
        return Ok(());
    };
    let mut order_by_node = HashMap::with_capacity(order.len());
    for event in events {
        if let (Some(event_id), Some(node_id)) =
            (event_id(event), template_node_id(template, event)?)
        {
            if let Some(position) = order.get(event_id) {
                order_by_node.insert(node_id, *position);
            }
        }
    }
    to_write.sort_by_key(|(node_id, _)| order_by_node.get(node_id).copied().unwrap_or(usize::MAX));
    Ok(())
}

/// Read the batch's cached `event_id` -> state-group mappings.
///
/// Returns `None` when the cache is absent, incomplete, invalid, or unreadable
/// — every case that forces the batch to recompute — and reports why.
fn load_cached_state_groups(
    aux: &AuxiliaryIndex<'_, PackfileStorage>,
    graph_event_ids: &[String],
) -> Option<HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>> {
    if graph_event_ids.is_empty() {
        return None;
    }
    match load_state_groups(aux, graph_event_ids) {
        Ok(StateGroupLoad::Complete(groups)) => Some(groups),
        Ok(StateGroupLoad::Missing) => None,
        Ok(StateGroupLoad::Invalid { event_id, reason }) => {
            eprintln!(
                "warning: invalid cached state-group mapping for {event_id} ({reason}); recomputing state groups"
            );
            None
        }
        Err(error) => {
            eprintln!(
                "warning: unable to read state-group mappings; recomputing state groups: {error}"
            );
            None
        }
    }
}

/// Report the merges that fell back to first-parent because this batch is
/// missing part of the auth closure state resolution reads.
fn warn_incomplete_merges(missing: &[String]) {
    if missing.is_empty() {
        return;
    }
    eprintln!(
        "warning: {} merge(s) fell back to first-parent state; this batch lacks these event(s) from the auth closure resolution reads",
        missing.len()
    );
    for event_id in missing.iter().take(MAX_DEBUG_UNRESOLVED) {
        eprintln!("  missing {event_id}");
    }
    if missing.len() > MAX_DEBUG_UNRESOLVED {
        eprintln!(
            "  ... {} more",
            missing.len().saturating_sub(MAX_DEBUG_UNRESOLVED)
        );
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the repair phase owns cache loading, judging, and state publication together"
)]
fn repair_import_state_groups(
    state_store: &PackfileStorage,
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    canonical_id: &str,
    room_version: &str,
    precomputed: Option<PartialStateComputation>,
) -> anyhow::Result<()> {
    // Replayed batches still need state-group repair when the derived index is
    // incomplete, but a complete replay should not rebuild the whole DAG.
    // Read the existing mappings in one backend batch and only walk the DAG
    // when at least one resolved mapping is absent. Resolved mappings are
    // trusted as immutable federated-import results; unresolved events and
    // events whose base is missing history are deliberately left absent so a
    // later import can repair them. Batches with such events therefore pay the
    // DAG-walk cost again on replay; skipping that safely would require
    // persisting the missing-parent set.
    let mut seen_state_event_ids = HashSet::new();
    let graph_event_ids = events
        .iter()
        .chain(auth_chain.iter())
        .filter_map(event_id)
        .map(str::to_owned)
        .filter(|event_id| seen_state_event_ids.insert(event_id.clone()))
        .collect::<Vec<_>>();
    let phase_cache = Instant::now();
    let aux = AuxiliaryIndex::open(state_store, STATE_GROUP_NAMESPACE);
    let cached = load_cached_state_groups(&aux, &graph_event_ids);
    let cached_subset = load_cached_state_group_subset(&aux, &graph_event_ids).unwrap_or_default();
    let loaded = cached.is_some();
    report_import_phase("state cache load", phase_cache.elapsed());
    let mut state_groups = cached.unwrap_or_default();
    let mut computed_instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance> =
        HashMap::new();
    // Whether the computed mappings may be cached. A merge that fell back
    // because this batch was missing part of the auth closure produces a
    // first-parent group under a resolver-tagged id; caching it would hide the
    // fact that a fuller import can compute the real resolution instead.
    let mut publishable = true;
    let mut partial_groups: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]> = HashMap::new();
    let phase_walk = Instant::now();
    let unresolved_count = if loaded {
        0
    } else {
        let policy = resolution_policy(room_version, StateGroupTrust::TrustedImport);
        if !policy.resolvable() {
            eprintln!(
                "warning: room version {room_version} has no state-resolution rules mtxdb recognizes; state merges use the first-parent rule"
            );
        }
        let state_computation = if let Some(computation) = precomputed {
            computation
        } else {
            let unflagged_events: Vec<OwnedValue> = events
                .iter()
                .map(|ev| interchange::unflag_pdu(ev, canonical_id))
                .collect();
            let unflagged_auth: Vec<OwnedValue> = auth_chain
                .iter()
                .map(|ev| interchange::unflag_pdu(ev, canonical_id))
                .collect();
            let mut room_events: Vec<RoomEvent> =
                Vec::with_capacity(unflagged_events.len().saturating_add(unflagged_auth.len()));
            let mut seen_ids = HashSet::new();
            for ev in unflagged_events.iter().chain(unflagged_auth.iter()) {
                if let Some(eid) = event_id(ev) {
                    if seen_ids.insert(eid.to_owned()) {
                        if let Some(room_ev) = RoomEvent::from_pdu(ev, canonical_id) {
                            room_events.push(room_ev);
                        }
                    }
                }
            }
            let mut rejected = HashSet::new();
            let mut walk_rejections = HashMap::new();
            let (comp, unchecked) = compute_import_auth_before(
                &unflagged_events,
                &unflagged_auth,
                &room_events,
                canonical_id,
                room_version,
                StateResVersion::from_room_version(room_version),
                &mut rejected,
                &mut walk_rejections,
                &HashSet::new(),
                None,
                None,
                &HashMap::new(),
            );
            if unchecked > 0 {
                eprintln!(
                        "  warning: {unchecked} event(s) could not be checked against the state before them"
                    );
            }
            comp
        };
        let unresolved = state_computation.unresolved;
        publishable = state_computation.incomplete.is_empty();
        warn_incomplete_merges(&state_computation.incomplete);
        state_groups = state_computation.groups;
        computed_instances = state_computation.instances;
        computed_instances.extend(state_computation.partial_instances);
        partial_groups = state_computation.partial;
        if !unresolved.is_empty() {
            eprintln!(
                "warning: {} event(s) are in parent cycles and have no derived state",
                unresolved.len()
            );
            if crate::debug_enabled() {
                let shown = unresolved.iter().take(MAX_DEBUG_UNRESOLVED);
                eprintln!("debug: unresolved state events:");
                for event_id in shown {
                    eprintln!("  {event_id}");
                }
                if unresolved.len() > MAX_DEBUG_UNRESOLVED {
                    eprintln!(
                        "  ... {} more",
                        unresolved.len().saturating_sub(MAX_DEBUG_UNRESOLVED)
                    );
                }
            }
        }
        unresolved.len()
    };
    let graph_event_count = graph_event_ids.len();
    report_import_phase("state walk/repair", phase_walk.elapsed());
    let mapped_event_count = state_groups.len();
    let unique_state_group_count = state_groups.values().collect::<HashSet<_>>().len();
    let partial_event_count = partial_groups.len();
    let stateless_event_count = graph_event_count
        .saturating_sub(mapped_event_count)
        .saturating_sub(partial_event_count);
    let source = if loaded { "loaded" } else { "computed" };
    eprintln!(
        "  state groups: {source} {unique_state_group_count} state groups; state: {} of {graph_event_count} events have a stored state group ({mapped_event_count} complete, {partial_event_count} partial forward extremities; an ancestor body is absent), {stateless_event_count} without ({unresolved_count} in cycles)",
        mapped_event_count.saturating_add(partial_event_count)
    );
    let phase_hamt = Instant::now();
    let mut materialized = computed_instances.is_empty();
    if !computed_instances.is_empty() {
        match materialize_state_hamts(state_store, canonical_id, &computed_instances) {
            Ok(written) => {
                materialized = true;
                eprintln!(
                    "  state HAMT: {written} records materialized for {} state groups",
                    computed_instances.len()
                );
            }
            Err(error) => return Err(error).context("persisting imported state HAMTs"),
        }
    }
    report_import_phase("state HAMT materialize", phase_hamt.elapsed());
    let phase_mappings = Instant::now();
    if materialized && !loaded && !state_groups.is_empty() && publishable {
        // Publish mappings only after their target HAMTs are durable. A later
        // replay must not mistake a partially materialized repair for a
        // complete cached result.
        // Publish mappings only after their target HAMTs are durable. A later
        // replay must not mistake a partially materialized repair for a
        // complete cached result. Only new or changed mappings are written;
        // rewriting one record per stored event on every file is pure write
        // amplification.
        let entries: Vec<(&[u8], &[u8])> = state_groups
            .iter()
            .filter(|(event_id, instance_id)| cached_subset.get(*event_id) != Some(*instance_id))
            .map(|(event_id, instance_id)| (event_id.as_bytes(), instance_id.as_slice()))
            .collect();
        if !entries.is_empty() {
            aux.put_many(&entries)
                .context("persisting imported state-group mappings")?;
        }
    }
    if materialized && !loaded && !partial_groups.is_empty() {
        let sg_best_effort_aux_idx =
            AuxiliaryIndex::open(state_store, STATE_GROUP_BEST_EFFORT_NAMESPACE);
        let cached_partial =
            load_cached_state_group_subset(&sg_best_effort_aux_idx, &graph_event_ids)
                .unwrap_or_default();
        let entries: Vec<(&[u8], &[u8])> = partial_groups
            .iter()
            .filter(|(event_id, instance_id)| cached_partial.get(*event_id) != Some(*instance_id))
            .map(|(event_id, instance_id)| (event_id.as_bytes(), instance_id.as_slice()))
            .collect();
        if !entries.is_empty() {
            sg_best_effort_aux_idx
                .put_many(&entries)
                .context("persisting partial state-group mappings")?;
        }
    }
    report_import_phase("state mappings publish", phase_mappings.elapsed());
    Ok(())
}

/// Resolve a Matrix input's room collection and prove that it is established
/// before any record from the input is written.
/// A resolved target collection for an import batch.
#[derive(Debug)]
struct ResolvedCollection {
    /// 128-bit routing key the batch's records are written under.
    collection_id: [u8; 16],
    /// Canonical external identity the routing key was derived from.
    canonical_id: String,
    /// Whether this batch carries the collection's establishment record.
    batch_has_create: bool,
}

/// Map a declared room version onto the numeric policy that governs it.
///
/// `MatrixRoomVersion` covers stock versions only. MSC3389 is v10 with
/// relation-preserving redaction, so its event-id and collection rules are
/// v10's. Rezzy still owns the actual redaction and auth decisions.
fn importer_room_version(value: &str) -> Option<MatrixRoomVersion> {
    match value {
        "org.matrix.msc3389.10" => Some(MatrixRoomVersion::V10),
        _ => MatrixRoomVersion::parse(value),
    }
}

/// Parse the room version declared by an establishment record's
/// `content.room_version`.
fn event_room_version(event: &OwnedValue) -> Option<MatrixRoomVersion> {
    importer_room_version(nested_event_string_field(event, "content", "room_version")?)
}

/// The collection identity an establishment record assigns, normalized to the
/// form later batches reference.
///
/// [`MatrixRoomVersion::collection_key_pointer`] selects the source field
/// (`/event_id` for v12, `/room_id` otherwise), and
/// [`MatrixRoomVersion::normalize_collection_identity`] maps v12's create
/// event id (`$<hash>`) onto the `!<hash>` room id ordinary events carry, so
/// an establishment batch and a later batch of ordinary events derive the same
/// collection.
fn establishment_identity(create: &OwnedValue) -> Option<String> {
    let version = event_room_version(create);
    let pointer = version.map_or("/room_id", MatrixRoomVersion::collection_key_pointer);
    let value = extract_pointer_string(create, pointer)?;
    Some(match version {
        Some(version) => version.normalize_collection_identity(value),
        None => value.to_owned(),
    })
}

/// The canonical external identity of the collection an import batch belongs
/// to, under the template's membership pointer and the Matrix room-version
/// policy.
///
/// The template's `collection.membership.extract` pointer names the default
/// membership field (the Matrix template uses `/room_id`). A version can move
/// that field onto the establishment record: room version 12 derives the room
/// identity from the accepted create event, so
/// [`MatrixRoomVersion::collection_key_pointer`] selects `/event_id` there and
/// [`MatrixRoomVersion::normalize_collection_identity`] converts it to the
/// `!<hash>` room id ordinary events carry. A batch carrying more than one
/// membership value is rejected rather than silently coalesced. Returns `None`
/// when no event carries a usable value.
fn collection_canonical_id(
    events: &[OwnedValue],
    template: &CollectionTemplate,
) -> anyhow::Result<Option<String>> {
    let membership: HashSet<&str> = events
        .iter()
        .filter_map(|event| extract_pointer_string(event, &template.collection_key.pointer))
        .collect();
    if membership.len() > 1 {
        bail!(
            "input contains events for multiple {} values; split it into one collection per input",
            template.collection_key.pointer
        );
    }
    if let Some(create) = events
        .iter()
        .find(|event| matrix_create_event_id(event).is_some())
    {
        if let Some(identity) = establishment_identity(create) {
            return Ok(Some(identity));
        }
    }
    Ok(membership.into_iter().next().map(str::to_owned))
}

fn resolve_import_collection(
    events: &[OwnedValue],
    detected_collection: Option<&str>,
    collection_override: Option<&str>,
    template: &CollectionTemplate,
    established_collections: &HashSet<[u8; 16]>,
) -> anyhow::Result<ResolvedCollection> {
    let canonical_id = collection_canonical_id(events, template)?
        .or_else(|| detected_collection.map(str::to_owned));
    let collection_id = if let Some(value) = collection_override {
        let collection_id = parse_collection_id(value)?;
        if let Some(canonical_id) = &canonical_id {
            let expected = template_collection_id(template, canonical_id);
            if collection_id != expected {
                bail!(
                    "--collection {value} does not match collection identity {canonical_id}; refusing to mix room data into another collection"
                );
            }
        }
        collection_id
    } else {
        let canonical_id = canonical_id.clone().with_context(|| {
            let first_event = events
                .first()
                .and_then(event_id)
                .unwrap_or("<missing event_id>");
            format!(
                "could not detect collection_id (first event: {first_event}); pass --collection for this input"
            )
        })?;
        template_collection_id(template, &canonical_id)
    };
    let batch_has_create = canonical_id
        .as_deref()
        .is_some_and(|canonical| matrix_batch_has_create(events, canonical));
    if batch_has_create {
        validate_establishment_room_version(events)?;
    }
    if !batch_has_create && !established_collections.contains(&collection_id) {
        let room = canonical_id.as_deref().unwrap_or("the selected collection");
        bail!(
            "refusing to import events for {room}: no valid m.room.create event is in this input or already on disk"
        );
    }
    Ok(ResolvedCollection {
        collection_id,
        canonical_id: canonical_id.unwrap_or_default(),
        batch_has_create,
    })
}

/// Enforce the room-version floor on the batch's establishment record.
///
/// mtxdb requires v4+: v1/v2 event IDs are server-assigned (not
/// content-addressed, and not guaranteed unique), and v3 encodes reference
/// hashes with non-URL-safe base64. A create event that omits
/// `content.room_version` is v1 by the spec and is rejected rather than
/// silently defaulted.
fn validate_establishment_room_version(events: &[OwnedValue]) -> anyhow::Result<()> {
    let Some(create) = events
        .iter()
        .find(|event| matrix_create_event_id(event).is_some())
    else {
        return Ok(());
    };
    let declared = nested_event_string_field(create, "content", "room_version");
    let version = declared.and_then(importer_room_version).with_context(|| {
        format!(
            "m.room.create declares room version {:?}; mtxdb requires v4 or later",
            declared.unwrap_or("<absent>")
        )
    })?;
    if !version.is_supported() {
        bail!("room version {version:?} is not supported; mtxdb requires v4 or later");
    }
    Ok(())
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
        .as_chunks::<16>()
        .0
        .iter()
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
            let bytes = record.data.to_vec();
            let Ok(event) = OwnedValue::parse_bytes(&bytes) else {
                continue;
            };
            if matrix_create_event_id(&event).is_some() {
                established.insert(record.collection_id);
            }
        }
    }
    Ok(established)
}

fn parse_jsonl_events(
    content: &[u8],
) -> anyhow::Result<(
    Vec<OwnedValue>,
    Option<String>,
    Vec<rezzy::OwnedMatrixEvent>,
)> {
    let text = std::str::from_utf8(content).context("input is not valid UTF-8 JSONL")?;
    let mut events = Vec::new();
    let mut raw_lines = Vec::new();
    let mut raw_detected_collection = None;
    // Reusable scratch for zero-alloc room_id extraction from raw bytes.
    // Allocated once here, grows to fit the largest event, then zero-alloc
    // on every subsequent call (steady-state: 0 heap allocations per event).
    let mut scratch = rezzy::MatrixEventScratch::with_capacity(16, 16, 64);
    for (line_number, line) in text.lines().enumerate() {
        let line_number = line_number
            .checked_add(1)
            .context("JSONL line number overflow")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if raw_detected_collection.is_none() {
            if let Ok(view) = rezzy::extract_matrix_event_into(line.as_bytes(), &mut scratch) {
                raw_detected_collection = view.room_id.map(str::to_owned);
            }
        }
        let bytes = line.as_bytes().to_vec();
        let event = OwnedValue::parse_bytes(&bytes)
            .with_context(|| format!("invalid JSONL event on line {line_number}"))?;
        events.push(event);
        raw_lines.push(line.as_bytes());
    }
    let detected_collection = raw_detected_collection
        .or_else(|| events.iter().find_map(event_room_id).map(str::to_owned));
    let raw_adjacency = rezzy::OwnedMatrixEvent::extract_batch(&raw_lines)
        .map_err(|error| anyhow!("extracting raw Matrix adjacency metadata: {error:?}"))?;
    Ok((events, detected_collection, raw_adjacency))
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
    let bytes = content.to_vec();
    let val: OwnedValue = OwnedValue::parse_bytes(&bytes).context("invalid JSON")?;
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
        for reference in auth_events {
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
/// BLAKE3(`event_id`) read as a little-endian `u64`.
///
/// This is a purely local, dense-ish key for the in-memory frontier. It is
/// independent of the on-disk [`mtxdb::NodeId`] (which is the first 16 bytes
/// of BLAKE3 of the extracted identity) and need not be stable across
/// algorithm changes.
fn event_short_id(event_id: &str) -> u64 {
    let hash = blake3_digest(event_id.as_bytes());
    let mut short_id_bytes = [0u8; 8];
    short_id_bytes.copy_from_slice(&hash[..8]);
    u64::from_le_bytes(short_id_bytes)
}

/// Build an in-memory event DAG from a set of events.
///
/// Each event is hashed to a `u64` `short_id` via [`event_short_id`], then
/// inserted into an [`mtxdb::dag::ActiveRoomFrontier`] with its `prev_events` and
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
    build_event_dag_with_missing_edges(events, false)
}

/// Build a DAG with optional synthetic IDs for edges to events outside the
/// input batch. State-group computation enables these IDs so absent parents
/// remain non-resident and block descendants; import ordering leaves them out
/// so older parents do not disable the existing in-batch topological order.
fn build_event_dag_with_missing_edges(
    events: &[OwnedValue],
    include_missing_edges: bool,
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
        let raw_prevs = extract_event_edge_ids(ev, "prev_events", &id_map, include_missing_edges);
        let raw_auths = extract_event_edge_ids(ev, "auth_events", &id_map, include_missing_edges);
        frontier.insert_event(short_id, &raw_prevs, &raw_auths);
    }
    frontier.rebind_resident_edges();
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
    include_missing_edges: bool,
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
            eid.and_then(|eid| match id_map.get(eid).copied() {
                Some(short_id) => Some(short_id),
                None if include_missing_edges => Some(event_short_id(eid)),
                None => None,
            })
        })
        .collect()
}

/// Topologically sort the DAG nodes in `frontier` using Kahn's algorithm.
///
/// Returns events in topological order (parents before children), or
/// `Err` with the list of events involved in a cycle or whose parents
/// are missing from the DAG.
#[cfg(test)]
fn topo_sort_dag(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    reverse_map: &HashMap<u64, String>,
) -> Result<Vec<usize>, Vec<String>> {
    topo_sort_dag_with_tie_break(frontier, reverse_map, None)
}

fn topo_sort_dag_with_tie_break(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    reverse_map: &HashMap<u64, String>,
    node_ids: Option<&HashMap<u64, [u8; 16]>>,
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
    let mut queue: BinaryHeap<Reverse<(Vec<u8>, usize)>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            let node_id = reverse_map
                .get(&frontier.nodes[idx].short_id)
                .cloned()
                .unwrap_or_else(|| format!("short:{:016x}", frontier.nodes[idx].short_id));
            let tie = node_ids
                .and_then(|ids| ids.get(&frontier.nodes[idx].short_id))
                .map_or_else(|| node_id.into_bytes(), |id| id.to_vec());
            queue.push(Reverse((tie, idx)));
        }
    }

    let mut sorted = Vec::with_capacity(n);
    while let Some(Reverse((_tie, idx))) = queue.pop() {
        sorted.push(idx);
        for &child in &children[idx] {
            in_degree[child] = in_degree[child]
                .checked_sub(1)
                .expect("DAG in-degree underflow");
            if in_degree[child] == 0 {
                let node_id = reverse_map
                    .get(&frontier.nodes[child].short_id)
                    .cloned()
                    .unwrap_or_else(|| format!("short:{:016x}", frontier.nodes[child].short_id));
                let tie = node_ids
                    .and_then(|ids| ids.get(&frontier.nodes[child].short_id))
                    .map_or_else(|| node_id.into_bytes(), |id| id.to_vec());
                queue.push(Reverse((tie, child)));
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

#[cfg(test)]
fn topological_event_order(events: &[OwnedValue]) -> Option<HashMap<String, usize>> {
    let (frontier, _id_map, reverse_map) = build_event_dag(events);
    let sorted = topo_sort_dag(&frontier, &reverse_map).ok()?;
    Some(event_order_map(&frontier, &reverse_map, sorted))
}

fn topological_event_order_with_template(
    events: &[OwnedValue],
    template: &CollectionTemplate,
) -> anyhow::Result<Option<HashMap<String, usize>>> {
    let (frontier, _id_map, reverse_map) = build_event_dag(events);
    let mut node_ids = HashMap::new();
    for event in events {
        if let (Some(event_id), Some(node_id)) =
            (event_id(event), template_node_id(template, event)?)
        {
            node_ids.insert(event_short_id(event_id), node_id);
        }
    }
    let sorted = topo_sort_dag_with_tie_break(&frontier, &reverse_map, Some(&node_ids)).ok();
    Ok(sorted.map(|sorted| event_order_map(&frontier, &reverse_map, sorted)))
}

fn event_order_map(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    reverse_map: &HashMap<u64, String>,
    sorted: Vec<usize>,
) -> HashMap<String, usize> {
    sorted
        .into_iter()
        .enumerate()
        .filter_map(|(order, index)| {
            reverse_map
                .get(&frontier.nodes[index].short_id)
                .cloned()
                .map(|event_id| (event_id, order))
        })
        .collect()
}

/// Results from a partial state-group walk.
///
/// `groups` contains the `event_id` -> `state_group_id` mappings that could
/// be resolved. `unresolved` contains events blocked by missing parents or
/// cycles.
struct PartialStateComputation {
    groups: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>,
    unresolved: Vec<String>,
    /// One record per derived state-group instance, retained so an import can
    /// materialize its `STGP` record and (deduplicated) `STAT` content.
    instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance>,
    /// Events whose base state was incomplete because a `prev_events` entry was
    /// missing or itself gapped. Their derived groups are in `partial`, not
    /// `groups`, so a later import with the ancestor present can publish the
    /// complete group without colliding with a cached partial one.
    gapped: Vec<String>,
    /// Event ids at least one merge could not resolve because this batch does
    /// not carry their auth closure. Non-empty means those merges fell back to
    /// the first-parent rule, so `groups` must not be cached as though
    /// the real resolver had run — a later, fuller import recomputes them.
    incomplete: Vec<String>,
    /// Divergent parent merges the walk processed, including union fallbacks.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "asserted by the gap and fork walk tests")
    )]
    merges: usize,
    /// Groups derived on a base missing history (see [`StateGroupWalk`]).
    partial: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>,
    partial_instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance>,
}

/// What one DAG walk produced: the per-event group map, the derived instances
/// worth materializing, events whose base was gapped, and any auth-closure gap
/// that forced a fallback.
struct StateGroupWalk {
    groups: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>,
    instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance>,
    /// Events whose base was gapped; their groups are in `partial`.
    gapped: Vec<String>,
    incomplete: Vec<String>,
    merges: usize,
    /// Groups derived on a base missing history, keyed by event. Never read as
    /// a seed: they are coverage evidence, not a complete state.
    partial: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]>,
    /// Instances behind `partial`, for HAMT materialization.
    partial_instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance>,
}

/// One resolved state-group derivation: the parent instance ids it descends
/// from, the resulting state `LtHash`, the state set to materialize into
/// the shared `STAT` content, and the derivation that produced it.
struct StateInstance {
    parents: Vec<[u8; STATE_GROUP_ID_LENGTH]>,
    lthash: [u8; 32],
    set: StateSet,
    derivation: StateGroupDerivation,
}

/// Compute state groups for a set of events by walking the event DAG in
/// topological order.
///
/// `events` are the PDUs; `auth_chain` events (if any) are included in
/// the DAG so that `auth_edges` resolve correctly, but only PDU state
/// events contribute to the state set.
///
/// Returns a map from `event_id` -> 128-bit `STGP` state-group instance id for
/// resolvable events. Each event inherits the state from its `prev_events` and
/// applies its own state change (if it is a state event with `state_key`). A
/// state-changing event, or a merge whose parents do not already share a single
/// instance, derives a new instance keyed by the parent instance ids and the
/// state `LtHash`; an ordinary event that changes nothing reuses its parent's
/// instance. Events blocked by missing parents or cycles are reported
/// separately in `unresolved`.
/// The state-group walk a caller without an import context uses: the federated
/// view, and whatever room version the batch's own `m.room.create` declares.
/// A batch with no establishment record resolves under the retired merge
/// rather than guessing a room version it was never told.
#[cfg(test)]
fn compute_state_groups_partial(
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    room_id: &str,
) -> PartialStateComputation {
    compute_state_groups_partial_in_view(events, auth_chain, StateView::Federated, room_id)
}

#[cfg(test)]
fn compute_state_groups_partial_in_view(
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    view: StateView,
    room_id: &str,
) -> PartialStateComputation {
    compute_state_groups_partial_with(
        events,
        auth_chain,
        view,
        room_id,
        batch_resolution_policy(events),
        None,
        &HashMap::new(),
        |_, _, _, _, _| None,
    )
}

/// The resolver and trust claim a batch gets when no caller states one.
#[cfg(test)]
fn batch_resolution_policy(events: &[OwnedValue]) -> ResolutionPolicy {
    let room_version = events
        .iter()
        .find(|event| matrix_create_event_id(event).is_some())
        .and_then(|event| nested_event_string_field(event, "content", "room_version"))
        .unwrap_or_default();
    resolution_policy(room_version, StateGroupTrust::TrustedImport)
}

/// Select the auth events this event needs from its base state.
fn auth_entries_from_base(
    event: &OwnedValue,
    base: &StateSet,
    version: StateResVersion,
    room_version: &str,
) -> export_import::AuthEntries {
    let event_type = event
        .get("type")
        .and_then(OwnedValue::as_str)
        .unwrap_or_default();
    let sender = event
        .get("sender")
        .and_then(OwnedValue::as_str)
        .unwrap_or_default();
    let state_key = event.get("state_key").and_then(OwnedValue::as_str);
    let empty_content = OwnedValue::Object(Object::default());
    let content = event.get("content").unwrap_or(&empty_content);
    auth_types_for_event_ref(
        event_type,
        sender,
        state_key,
        content,
        version,
        room_version,
    )
    .into_iter()
    .filter_map(|(event_type, state_key)| {
        let key = (event_type.to_owned(), state_key.to_owned());
        base.entries
            .get(&key)
            .map(|event_id| (key.0, key.1, event_id.clone()))
    })
    .collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "the walk needs the DAG, the view, the seed states, and its base-state observer"
)]
fn compute_state_groups_partial_with(
    events: &[OwnedValue],
    auth_chain: &[OwnedValue],
    view: StateView,
    room_id: &str,
    policy: ResolutionPolicy,
    rejected: Option<&mut HashSet<String>>,
    seed_states: &HashMap<String, Arc<SharedState>>,
    judge: impl FnMut(
        &str,
        &OwnedValue,
        &StateSet,
        bool,
        Option<&HashSet<String>>,
    ) -> Option<(u8, String)>,
) -> PartialStateComputation {
    // Borrow the input when there is no auth chain to merge in; cloning every
    // parsed event just to concatenate two slices doubles peak memory on large
    // rooms.
    let combined: Vec<OwnedValue>;
    let all_owned: &[OwnedValue] = if auth_chain.is_empty() {
        events
    } else {
        combined = events.iter().chain(auth_chain.iter()).cloned().collect();
        &combined
    };
    if crate::debug_enabled() {
        let direct_missing = missing_prev_event_ids(all_owned);
        if !direct_missing.is_empty() {
            eprintln!(
                "debug: {} event(s) directly reference missing prev_events",
                direct_missing.len()
            );
        }
    }
    let (frontier, id_map, reverse_map) = build_event_dag_with_missing_edges(all_owned, true);
    if frontier.is_empty() {
        return PartialStateComputation {
            groups: HashMap::new(),
            unresolved: Vec::new(),
            instances: HashMap::new(),
            gapped: Vec::new(),
            incomplete: Vec::new(),
            merges: 0,
            partial: HashMap::new(),
            partial_instances: HashMap::new(),
        };
    }

    let (sorted, unresolved) = partial_topo_sort_dag(&frontier, &reverse_map);

    let mut events_by_sid: HashMap<u64, &OwnedValue> = HashMap::new();
    for ev in all_owned {
        if let Some(eid) = event_id(ev) {
            if let Some(&sid) = id_map.get(eid) {
                events_by_sid.insert(sid, ev);
            }
        }
    }

    let result = walk_state_groups(
        room_id,
        &frontier,
        &sorted,
        &id_map,
        &reverse_map,
        &events_by_sid,
        view,
        policy,
        rejected,
        seed_states,
        judge,
    );

    PartialStateComputation {
        groups: result.groups,
        unresolved,
        instances: result.instances,
        gapped: result.gapped,
        incomplete: result.incomplete,
        merges: result.merges,
        partial: result.partial,
        partial_instances: result.partial_instances,
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the walk needs the DAG, the state view, and its base-state observer"
)]
fn walk_state_groups(
    room_id: &str,
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    sorted: &[usize],
    id_map: &HashMap<String, u64>,
    reverse_map: &HashMap<u64, String>,
    events_by_sid: &HashMap<u64, &OwnedValue>,
    view: StateView,
    policy: ResolutionPolicy,
    mut rejected: Option<&mut HashSet<String>>,
    seed_states: &HashMap<String, Arc<SharedState>>,
    mut judge: impl FnMut(
        &str,
        &OwnedValue,
        &StateSet,
        bool,
        Option<&HashSet<String>>,
    ) -> Option<(u8, String)>,
) -> StateGroupWalk {
    let mut remaining_children = count_resident_children(frontier, sorted);

    // Built once for the whole walk and shared by every merge: rezzy events
    // are materialized lazily and each id is walked at most once.
    let mut context = LeanEventContext::new(events_by_sid.values().copied());
    let mut incomplete: HashSet<String> = HashSet::new();
    let mut gapped: HashSet<String> = HashSet::new();
    let mut partial: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]> = HashMap::new();
    let mut partial_instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance> = HashMap::new();

    let empty_set = StateSet::new();
    let empty = SharedState::new(
        empty_set,
        state_group_instance_id(room_id, &[], &StateSet::new().lthash(), policy.derivation),
        true,
    );
    let mut state_at: HashMap<u64, Arc<SharedState>> = HashMap::new();
    let mut result: HashMap<String, [u8; STATE_GROUP_ID_LENGTH]> = HashMap::new();
    let mut instances: HashMap<[u8; STATE_GROUP_ID_LENGTH], StateInstance> = HashMap::new();
    let seed_instances: HashSet<[u8; STATE_GROUP_ID_LENGTH]> =
        seed_states.values().map(|seed| seed.instance).collect();
    let mut merge_time = std::time::Duration::ZERO;
    let mut merge_count = 0usize;
    let mut merge_phases = crate::state_resolve::MergePhaseTimes::default();

    for &idx in sorted {
        let short_id = frontier.nodes[idx].short_id;
        let event_value = events_by_sid.get(&short_id).copied();
        let is_create = event_value.is_some_and(|ev| {
            ev.get("type").and_then(OwnedValue::as_str) == Some("m.room.create")
                && ev.get("state_key").and_then(OwnedValue::as_str) == Some("")
        });
        let claimed_prev_ids: Vec<&str> = event_value
            .and_then(|ev| ev.get("prev_events"))
            .and_then(OwnedValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|entry| match entry {
                        OwnedValue::String(id) => Some(id.as_str()),
                        OwnedValue::Array(fields) => fields.first().and_then(OwnedValue::as_str),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut parent_ids: Vec<u64> = Vec::new();
        let mut parents: Vec<Arc<SharedState>> = Vec::new();
        let mut parent_instances: Vec<[u8; STATE_GROUP_ID_LENGTH]> = Vec::new();
        let mut base_incomplete = false;
        let mut seen_parents: HashSet<u64> = HashSet::new();
        for prev_id in &claimed_prev_ids {
            if let Some(&parent_id) = id_map.get(*prev_id) {
                if !seen_parents.insert(parent_id) {
                    continue;
                }
                parent_ids.push(parent_id);
                if let Some(parent_state) = state_at.get(&parent_id) {
                    parent_instances.push(parent_state.instance);
                    parents.push(Arc::clone(parent_state));
                } else {
                    base_incomplete = true;
                }
            } else if let Some(seed) = seed_states.get(*prev_id) {
                let parent_id = event_short_id(prev_id);
                if !seen_parents.insert(parent_id) {
                    continue;
                }
                parent_instances.push(seed.instance);
                parents.push(Arc::clone(seed));
            } else {
                base_incomplete = true;
            }
        }
        if !is_create && parents.is_empty() {
            base_incomplete = true;
        }
        if seen_parents.len() != claimed_prev_ids.len() {
            base_incomplete = true;
        }
        // Incompleteness is inherited: a child of an event with a gapped base
        // has a gapped base too, even when all of its own parents are present.
        base_incomplete = base_is_incomplete(base_incomplete, &parents);
        // Keep the gap before a resolver fallback can add to it: a gapped
        // merge uses the union estimate below, but only a missing auth closure
        // on a complete base makes the whole batch's other groups unsafe.
        let base_gapped = base_incomplete;

        // Parents that already agree share their state, so a plain message
        // event costs no allocation. A divergent complete base goes through
        // state resolution. A gapped base takes the existing union fallback:
        // its result is partial and never published as complete.
        let base: Arc<SharedState> = match parents.as_slice() {
            [] => Arc::clone(&empty),
            [only] => Arc::clone(only),
            [first, rest @ ..]
                if rest
                    .iter()
                    .all(|p| Arc::ptr_eq(p, first) || p.lthash == first.lthash) =>
            {
                Arc::clone(first)
            }
            [..] => {
                let merge_started = Instant::now();
                let (merged, fallback) = merge_parent_states(
                    &mut context,
                    &parents,
                    &parent_instances,
                    room_id,
                    policy,
                    base_gapped,
                    &mut incomplete,
                    &mut merge_phases,
                );
                merge_time = merge_time.saturating_add(merge_started.elapsed());
                merge_count = merge_count.saturating_add(1);
                base_incomplete |= fallback;
                merged
            }
        };

        // Test-only experiment: fill-only auth-event seeding at missing-prev gap
        // roots. Compiled out of every non-test build and inert unless a test
        // arms it.
        #[cfg(test)]
        let base = {
            let missing_prev_root = !claimed_prev_ids.is_empty()
                && claimed_prev_ids
                    .iter()
                    .any(|prev| !id_map.contains_key(*prev) && !seed_states.contains_key(*prev));
            if missing_prev_root {
                seed_experiment::seed_gap_root(
                    &base,
                    event_value,
                    id_map,
                    events_by_sid,
                    &parent_instances,
                    room_id,
                    policy,
                )
                .unwrap_or(base)
            } else {
                base
            }
        };

        let event_id = reverse_map.get(&short_id);
        if let (Some(event_id), Some(event_value)) = (event_id, event_value) {
            let already_rejected = rejected
                .as_deref()
                .is_some_and(|set| set.contains(event_id));
            if !already_rejected {
                let current = rejected.as_deref();
                if judge(event_id, event_value, &base.set, !base_incomplete, current).is_some() {
                    if let Some(set) = rejected.as_deref_mut() {
                        set.insert(event_id.clone());
                    }
                }
            }
        }

        let state = derive_state(
            event_value,
            event_id.map(String::as_str),
            &base,
            view,
            &parent_instances,
            room_id,
            policy,
            rejected.as_deref(),
            !base_incomplete,
        );

        // A partial state is kept only where nothing descends from the event
        // (a forward extremity): that is the room's best-effort current state.
        // Keeping one for every event copies the whole state map per state
        // event, which is quadratic in memory on rooms with many members.
        let keep_partial =
            base_gapped && remaining_children.get(&short_id).copied().unwrap_or(0) == 0;
        if keep_partial {
            partial_instances
                .entry(state.instance)
                .or_insert_with(|| StateInstance {
                    parents: canonical_parents(&state.parents),
                    lthash: state.lthash,
                    set: state.set.clone(),
                    derivation: policy.derivation,
                });
        }
        if !base_gapped && !seed_instances.contains(&state.instance) {
            instances
                .entry(state.instance)
                .or_insert_with(|| StateInstance {
                    parents: canonical_parents(&state.parents),
                    lthash: state.lthash,
                    set: state.set.clone(),
                    derivation: policy.derivation,
                });
        }

        if let Some(eid) = reverse_map.get(&short_id) {
            if base_gapped {
                // No state group is published for an event whose base is
                // missing history: a later direct import would seed it as
                // complete. The whole-room path must leave the same set
                // unmapped so the two paths agree.
                gapped.insert(eid.clone());
                if keep_partial {
                    partial.insert(eid.clone(), state.instance);
                }
            } else {
                result.insert(eid.clone(), state.instance);
            }
        }
        for parent_id in parent_ids {
            if let Some(count) = remaining_children.get_mut(&parent_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    state_at.remove(&parent_id);
                }
            }
        }
        if remaining_children.get(&short_id).copied().unwrap_or(0) > 0 {
            state_at.insert(short_id, state);
        }
    }
    let mut incomplete: Vec<String> = incomplete.into_iter().collect();
    incomplete.sort();
    let mut gapped: Vec<String> = gapped.into_iter().collect();
    gapped.sort();
    if merge_count > 0 {
        report_import_phase(&format!("state merges ({merge_count})"), merge_time);
        report_import_phase("state merges convert", merge_phases.convert);
        report_import_phase("state merges ensure", merge_phases.ensure);
        report_import_phase("state merges resolve", merge_phases.resolve);
        report_import_phase("state merges back", merge_phases.back);
    }
    StateGroupWalk {
        groups: result,
        instances,
        gapped,
        incomplete,
        merges: merge_count,
        partial,
        partial_instances,
    }
}

/// Whether a base is incomplete, including gaps inherited from any parent.
fn base_is_incomplete(own_gap: bool, parents: &[Arc<SharedState>]) -> bool {
    own_gap || parents.iter().any(|parent| !parent.complete)
}

/// How many resident children still need each event's state. Once the last
/// child is processed the state is dropped, so memory follows the DAG's
/// frontier width instead of holding one state per event for the whole run.
fn count_resident_children(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    sorted: &[usize],
) -> HashMap<u64, usize> {
    let mut remaining_children: HashMap<u64, usize> = HashMap::new();
    for &idx in sorted {
        for edge in frontier.prev_edges(idx) {
            if edge.is_resident() {
                let count = remaining_children
                    .entry(frontier.nodes[edge.arena_index()].short_id)
                    .or_default();
                *count = count.saturating_add(1);
            }
        }
    }
    remaining_children
}

/// The state an event descends to.
///
/// Only a state event the view keeps produces a new state set. A node whose
/// parents do not already share one instance derives a new instance holding
/// that state, so the merge topology survives even when the state itself did
/// not change. Anything else shares its parent's state and costs no copy.
#[allow(
    clippy::too_many_arguments,
    reason = "the derivation needs the event, the inherited state, and the identity inputs together"
)]
fn derive_state(
    event: Option<&OwnedValue>,
    event_id_ref: Option<&str>,
    base: &Arc<SharedState>,
    view: StateView,
    parent_instances: &[[u8; STATE_GROUP_ID_LENGTH]],
    room_id: &str,
    policy: ResolutionPolicy,
    rejected: Option<&HashSet<String>>,
    base_complete: bool,
) -> Arc<SharedState> {
    let excluded = event.is_some_and(|ev| match rejected {
        Some(set) => view.excludes_import(ev, event_id_ref.unwrap_or(""), set),
        None => view.excludes(ev),
    });
    match event {
        Some(ev) if is_state_event(ev) && !excluded => {
            let mut set = base.set.clone();
            set.set(
                event_string_field(ev, "type").unwrap_or(""),
                &state_key(ev),
                event_id(ev).unwrap_or("").to_owned(),
            );
            let instance = state_group_instance_id(
                room_id,
                parent_instances,
                &set.lthash(),
                policy.derivation,
            );
            SharedState::with_parents(set, instance, base_complete, parent_instances.to_vec())
        }
        _ if distinct_instances(parent_instances) > 1 => {
            let instance =
                state_group_instance_id(room_id, parent_instances, &base.lthash, policy.derivation);
            SharedState::with_parents(
                base.set.clone(),
                instance,
                base_complete,
                parent_instances.to_vec(),
            )
        }
        _ if base_complete != base.complete => SharedState::with_parents(
            base.set.clone(),
            base.instance,
            base_complete,
            base.parents.clone(),
        ),
        _ => Arc::clone(base),
    }
}

/// Process one divergent merge: complete bases use state resolution, while a
/// gapped base takes the union fallback because its result stays partial.
///
/// Parents that already agree reach `walk_state_groups`'s earlier arms and
/// never get here, so this is only ever called with at least two distinct
/// states. For complete bases, when rezzy cannot run — no recognized room
/// version or missing auth closure — the union fallback records the missing
/// auth references so the batch cannot cache the result as a real resolution.
/// For gapped bases the same union operation is an intentional best-effort
/// estimate, and the gap itself prevents publication as complete.
#[allow(
    clippy::too_many_arguments,
    reason = "the merge needs the context, the parents, and the identity inputs together"
)]
fn merge_parent_states(
    context: &mut LeanEventContext<'_>,
    parents: &[Arc<SharedState>],
    parent_instances: &[[u8; STATE_GROUP_ID_LENGTH]],
    room_id: &str,
    policy: ResolutionPolicy,
    force_union: bool,
    incomplete: &mut HashSet<String>,
    phases: &mut crate::state_resolve::MergePhaseTimes,
) -> (Arc<SharedState>, bool) {
    let parent_entries: Vec<&StateEntries> =
        parents.iter().map(|parent| &parent.set.entries).collect();
    let mut fallback = false;
    let union = |parents: &[Arc<SharedState>]| {
        // Start from the first parent's persistent set (an O(1) clone) and add
        // only what the others contribute: the same first-parent-wins result as
        // inserting every parent's entries into an empty set.
        let Some((first, rest)) = parents.split_first() else {
            return StateSet::new();
        };
        let mut merged = first.set.clone();
        for parent in rest {
            merged.merge(&parent.set);
        }
        merged
    };
    let merged = if force_union {
        // A merge whose base is gapped is never published as complete and
        // judges leniently, so it takes the union instead of paying for a
        // state resolution whose result is only a best-effort estimate.
        fallback = true;
        union(parents)
    } else {
        match resolve_parents(context, &parent_entries, policy, phases) {
            Ok(entries) => StateSet::from_entries(entries),
            Err(gap) => {
                fallback = true;
                incomplete.extend(gap.missing);
                union(parents)
            }
        }
    };
    let instance = state_group_instance_id(
        room_id,
        parent_instances,
        &merged.lthash(),
        policy.derivation,
    );
    let complete = !fallback && parents.iter().all(|parent| parent.complete);
    (
        SharedState::with_parents(merged, instance, complete, parent_instances.to_vec()),
        fallback,
    )
}

/// Number of distinct instance ids in `instances`.
fn distinct_instances(instances: &[[u8; STATE_GROUP_ID_LENGTH]]) -> usize {
    let mut ids = instances.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

/// Sort and deduplicate parent instance ids for a canonical record.
fn canonical_parents(parents: &[[u8; STATE_GROUP_ID_LENGTH]]) -> Vec<[u8; STATE_GROUP_ID_LENGTH]> {
    let mut canonical = parents.to_vec();
    canonical.sort_unstable();
    canonical.dedup();
    canonical
}

/// Return the events whose `prev_events` reference a parent absent from this
/// import batch. This remains useful for diagnostics even though the partial
/// state walker no longer rejects the entire batch up front.
fn missing_prev_event_ids(events: &[OwnedValue]) -> Vec<String> {
    let known: HashSet<&str> = events.iter().filter_map(event_id).collect();
    let mut missing = Vec::new();
    for event in events {
        let Some(event_id) = event_id(event) else {
            continue;
        };
        let OwnedValue::Object(fields) = event else {
            continue;
        };
        let Some(OwnedValue::Array(prev_events)) = fields.get("prev_events") else {
            continue;
        };
        if prev_events.iter().any(|reference| {
            let parent = match reference {
                OwnedValue::String(id) => Some(id.as_str()),
                OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                _ => None,
            };
            parent.is_some_and(|parent| !known.contains(parent))
        }) {
            missing.push(event_id.to_owned());
        }
    }
    missing.sort();
    missing
}

/// Sort the DAG's resident events. A parent absent from the batch contributes
/// no ordering edge: the state walk marks the event's base as missing history
/// (a partial state) instead of dropping it and all of its descendants. Only a
/// cycle and its descendants stay unresolved. Independent components continue
/// to be processed normally.
fn partial_topo_sort_dag(
    frontier: &mtxdb::dag::ActiveRoomFrontier,
    reverse_map: &HashMap<u64, String>,
) -> (Vec<usize>, Vec<String>) {
    let n = frontier.nodes.len();
    let mut in_degree = vec![0usize; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let blocked = vec![false; n];

    for (idx, _node) in frontier.nodes.iter().enumerate() {
        for edge in frontier.prev_edges(idx) {
            if edge.is_resident() {
                let parent_idx = edge.arena_index();
                children[parent_idx].push(idx);
                in_degree[idx] = in_degree[idx].saturating_add(1);
            }
        }
    }

    let mut queue: BinaryHeap<Reverse<(String, usize)>> = BinaryHeap::new();
    for (idx, &degree) in in_degree.iter().enumerate() {
        if !blocked[idx] && degree == 0 {
            let event_id = reverse_map
                .get(&frontier.nodes[idx].short_id)
                .cloned()
                .unwrap_or_else(|| format!("short:{:016x}", frontier.nodes[idx].short_id));
            queue.push(Reverse((event_id, idx)));
        }
    }

    let mut sorted = Vec::with_capacity(n.saturating_sub(blocked.iter().filter(|&&v| v).count()));
    let mut processed = vec![false; n];
    while let Some(Reverse((_, idx))) = queue.pop() {
        if processed[idx] {
            continue;
        }
        processed[idx] = true;
        sorted.push(idx);
        for &child in &children[idx] {
            in_degree[child] = in_degree[child].saturating_sub(1);
            if !blocked[child] && in_degree[child] == 0 {
                let event_id = reverse_map
                    .get(&frontier.nodes[child].short_id)
                    .cloned()
                    .unwrap_or_else(|| format!("short:{:016x}", frontier.nodes[child].short_id));
                queue.push(Reverse((event_id, child)));
            }
        }
    }

    let mut unresolved = processed
        .iter()
        .enumerate()
        .filter(|&(_idx, done)| !done)
        .map(|(idx, _done)| {
            reverse_map
                .get(&frontier.nodes[idx].short_id)
                .cloned()
                .unwrap_or_else(|| format!("short:{:016x}", frontier.nodes[idx].short_id))
        })
        .collect::<Vec<_>>();
    unresolved.sort();
    (sorted, unresolved)
}

/// A state set with its `LtHash` and state-group instance id computed once, so
/// events that share a state also share them instead of re-sorting and
/// re-hashing.
struct SharedState {
    set: StateSet,
    lthash: [u8; 32],
    instance: [u8; STATE_GROUP_ID_LENGTH],
    /// Whether every event this state descends from had a complete base. A
    /// missing parent or a resolution fallback marks the whole subtree, so
    /// descendants are not judged against state built on a gap.
    complete: bool,
    /// The parent instances this state's own instance record names. A state
    /// shared by descendants (a message event reuses its parent's state) keeps
    /// the parents of the event that created it, so any holder can publish the
    /// correct record rather than one that points the instance at itself.
    parents: Vec<[u8; STATE_GROUP_ID_LENGTH]>,
}

impl SharedState {
    fn new(set: StateSet, instance: [u8; STATE_GROUP_ID_LENGTH], complete: bool) -> Arc<Self> {
        Self::with_parents(set, instance, complete, Vec::new())
    }

    fn with_parents(
        set: StateSet,
        instance: [u8; STATE_GROUP_ID_LENGTH],
        complete: bool,
        parents: Vec<[u8; STATE_GROUP_ID_LENGTH]>,
    ) -> Arc<Self> {
        let lthash = set.lthash();
        Arc::new(Self {
            set,
            lthash,
            instance,
            complete,
            parents,
        })
    }
}

/// A set of state events keyed by (type, `state_key`).
///
/// The entries are a persistent map and the `LtHash` is maintained alongside
/// them, so cloning a set is O(1) and a state event updates the digest with one
/// remove and one insert instead of rehashing every entry.
#[derive(Debug, Clone, Default)]
struct StateSet {
    entries: StateEntries,
    lattice: rezzy::incremental::LtHash,
}

impl StateSet {
    fn new() -> Self {
        Self::default()
    }

    /// Wrap resolved entries, hashing them once.
    fn from_entries(entries: StateEntries) -> Self {
        let mut lattice = rezzy::incremental::LtHash::default();
        for ((event_type, state_key), event_id) in &entries {
            lattice.insert(event_type, state_key, event_id);
        }
        Self { entries, lattice }
    }

    fn set(&mut self, event_type: &str, state_key: &str, event_id: String) {
        self.lattice.insert(event_type, state_key, &event_id);
        if let Some(old) = self
            .entries
            .insert((event_type.to_owned(), state_key.to_owned()), event_id)
        {
            self.lattice.remove(event_type, state_key, &old);
        }
    }
    /// Union another state set into this one, keeping the existing entry when
    /// both sets name different event ids for the same `(type, state_key)`.
    ///
    /// This is the first-parent-wins fallback, and it only runs where
    /// state resolution cannot: no recognized room version, or a batch missing
    /// part of the auth closure the resolver reads. Every other merge goes
    /// through [`merge_parent_states`] instead.
    fn merge(&mut self, other: &StateSet) {
        for (key, eid) in &other.entries {
            if !self.entries.contains_key(key) {
                self.set(&key.0, &key.1, eid.clone());
            }
        }
    }

    /// Unkeyed `LtHash` digest over the logical `(type, state_key, event_id)`
    /// entries: the state identity shared by every derivation that reaches it.
    ///
    /// `LtHash` addition is commutative, so entry order does not matter. This
    /// matches `sithnapse`'s state identity and the lattice embedded in every
    /// materialized `MTHR` root.
    fn lthash(&self) -> [u8; 32] {
        self.lattice.digest()
    }

    /// The `LtHash` digest as unpadded base64url, for golden tests.
    #[cfg(test)]
    fn digest_base64url(&self) -> String {
        crate::state_hamt::encode_state_group_id(&self.lthash())
    }
}

/// Check if an event is a state event (has a `state_key` field).
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

pub(crate) fn event_id(value: &OwnedValue) -> Option<&str> {
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
fn matrix_batch_has_create(events: &[OwnedValue], collection_canonical_id: &str) -> bool {
    events.iter().any(|event| {
        let Some(create_id) = matrix_create_event_id(event) else {
            return false;
        };
        // Room version 12 carries the collection identity in the create
        // event's own `event_id` (normalized to `!<hash>`); earlier versions
        // store it as `room_id`.
        establishment_identity(event).as_deref() == Some(collection_canonical_id)
            || event_room_id(event) == Some(collection_canonical_id)
            || events.iter().any(|candidate| {
                event_room_id(candidate) == Some(collection_canonical_id)
                    && event_auth_references(candidate, create_id)
            })
    })
}

pub(crate) fn event_string_field<'a>(value: &'a OwnedValue, field: &str) -> Option<&'a str> {
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

struct CoalescedSourceDb {
    path: PathBuf,
    store: PackfileStorage,
    pool_dir: PathBuf,
}

const COALESCE_REPACK_CHUNK_SIZE: usize = 1000;

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "orchestrates discovery, candidate scanning, conflict resolution, DAG ordering, and canonical pack generation"
)]
fn cmd_repack_coalesced(
    cli: &Cli,
    out_dir: &Path,
    collection: Option<&str>,
    packs: &[String],
    all: bool,
    topo: bool,
    yes: bool,
) -> anyhow::Result<()> {
    if cli.dirs.len() > 1 && !cli.coalesce {
        bail!("repacking across multiple database roots requires --coalesce (-c)");
    }
    if !packs.is_empty() {
        bail!("--pack is not supported with coalescing repack; use --all or --collection <collection>");
    }
    if collection.is_none() && !all {
        bail!("coalescing repack requires --all or --collection <collection>");
    }
    if topo {
        println!("note: all records preserved, laid out dependencies first");
    } else {
        println!(
            "note: this pool's ordering policy is not topological; records written in id order"
        );
    }

    let valid_dirs = valid_database_dirs(cli)?;
    for dir in &valid_dirs {
        if let (Ok(can_in), Ok(can_out)) = (dir.canonicalize(), out_dir.canonicalize()) {
            if can_in == can_out {
                bail!(
                    "output directory `{}` cannot be one of the source databases",
                    out_dir.display()
                );
            }
        }
    }
    if out_dir.exists() {
        if !out_dir.is_dir() {
            bail!("output path `{}` is not a directory", out_dir.display());
        }
        let mut entries = fs::read_dir(out_dir)
            .with_context(|| format!("failed to read output directory `{}`", out_dir.display()))?;
        if entries.next().is_some() {
            bail!(
                "output directory `{}` already exists and is not empty",
                out_dir.display()
            );
        }
    }

    let shard_type = cli.require_shard_type()?;
    let mut sources = Vec::new();
    for db_dir in &valid_dirs {
        let Ok(layout) = DatabaseLayout::open_read_only(db_dir.clone()) else {
            continue;
        };
        let Ok(pool_dir) = layout.pool_dir_read_only(shard_type) else {
            continue;
        };
        if glob_pack_files(&pool_dir).map_or(true, |p| p.is_empty()) {
            continue;
        }
        let store = open_pool_read_only(pool_dir.clone())?;
        sources.push(CoalescedSourceDb {
            path: db_dir.clone(),
            store,
            pool_dir,
        });
    }
    if sources.is_empty() {
        bail!(
            "no source databases contain active data for {} pool",
            shard_type.as_str()
        );
    }

    let target_collections: HashSet<[u8; 16]> = if let Some(col_str) = collection {
        let cid = parse_collection_selector(col_str)?;
        let mut set = HashSet::new();
        set.insert(cid);
        set
    } else {
        let mut set = HashSet::new();
        for src in &sources {
            for cid in src.store.collection_ids() {
                set.insert(cid);
            }
        }
        set
    };

    let mut candidates_by_col: HashMap<
        [u8; 16],
        BTreeMap<mtxdb::NodeId, Vec<(PathBuf, NodeData)>>,
    > = HashMap::new();

    for src in &sources {
        let mut pack_paths = Vec::new();
        if let Ok(entries) = fs::read_dir(&src.pool_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "pack") {
                    pack_paths.push(p);
                }
            }
        }
        pack_paths.sort();

        let mut seen_in_src: HashMap<[u8; 16], HashSet<mtxdb::NodeId>> = HashMap::new();
        for pack_path in &pack_paths {
            let entries = match mtxdb::packfile::scan_packfile(pack_path) {
                Ok(e) => e,
                Err(err) => {
                    eprintln!(
                        "warning: failed to scan packfile `{}`: {err}",
                        pack_path.display()
                    );
                    continue;
                }
            };
            for (col_id, node_id, _offset) in entries {
                if target_collections.contains(&col_id) {
                    seen_in_src.entry(col_id).or_default().insert(node_id);
                }
            }
        }

        for &col_id in &target_collections {
            if src.store.collection_index_info(&col_id).is_some() {
                seen_in_src
                    .entry(col_id)
                    .or_default()
                    .insert(mtxdb::COLLECTION_METADATA_RECORD_ID);
            }
        }

        for (col_id, node_ids) in seen_in_src {
            let col_map = candidates_by_col.entry(col_id).or_default();
            for node_id in node_ids {
                match src.store.get(&col_id, &node_id) {
                    Ok(Some(data)) => {
                        col_map
                            .entry(node_id)
                            .or_default()
                            .push((src.path.clone(), data));
                    }
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!(
                            "warning: failed to get node {} in collection {} from `{}`: {err}",
                            hex::encode(node_id),
                            format_id(&col_id),
                            src.path.display(),
                        );
                    }
                }
            }
        }
    }

    let mut total_collections = 0usize;
    let mut total_nodes = 0usize;
    for nodes in candidates_by_col.values() {
        if !nodes.is_empty() {
            total_collections = total_collections.saturating_add(1);
            total_nodes = total_nodes.saturating_add(nodes.len());
        }
    }
    if total_collections == 0 {
        println!("no collections found to repack");
        return Ok(());
    }
    println!(
        "coalescing repack preflight: {} collection{}, {} unique nodes across {} database{} -> {}",
        total_collections,
        if total_collections == 1 { "" } else { "s" },
        total_nodes,
        sources.len(),
        if sources.len() == 1 { "" } else { "s" },
        out_dir.display(),
    );
    if !yes && !confirm("Proceed with coalescing repack?")? {
        println!("aborted");
        return Ok(());
    }

    let target_layout = DatabaseLayout::open(out_dir.to_path_buf())?;
    let target_pool_dir = target_layout.pool_dir(shard_type)?;
    let target_store = open_pool(target_pool_dir)?;

    let mut sorted_collection_ids: Vec<[u8; 16]> = candidates_by_col.keys().copied().collect();
    sorted_collection_ids.sort_unstable();

    let mut written_collections = 0usize;
    let mut written_nodes = 0usize;

    for collection_id in sorted_collection_ids {
        let nodes_map = candidates_by_col.remove(&collection_id).unwrap_or_default();
        if nodes_map.is_empty() {
            continue;
        }

        let mut resolved_nodes: HashMap<mtxdb::NodeId, NodeData> =
            HashMap::with_capacity(nodes_map.len());
        for (node_id, mut candidates) in nodes_map {
            if candidates.is_empty() {
                continue;
            }
            let data = if candidates.len() == 1 {
                candidates.remove(0).1
            } else {
                let first_bytes = candidates[0].1.bytes.clone();
                if candidates.iter().all(|(_, d)| d.bytes == first_bytes) {
                    candidates.remove(0).1
                } else {
                    let mut ranked: Vec<(PathBuf, NodeData, Option<u64>)> = candidates
                        .into_iter()
                        .map(|(db, d)| {
                            let ts = extract_origin_server_ts(&d.bytes);
                            (db, d, ts)
                        })
                        .collect();
                    ranked.sort_by(|a, b| {
                        b.2.cmp(&a.2)
                            .then_with(|| a.0.to_string_lossy().cmp(&b.0.to_string_lossy()))
                    });
                    let distinct_payloads = {
                        let mut set = HashSet::new();
                        for (_, d, _) in &ranked {
                            set.insert(&d.bytes);
                        }
                        set.len()
                    };
                    let winner = ranked.remove(0);
                    let ts_msg = match winner.2 {
                        Some(ts) => format!("origin_server_ts: {ts}"),
                        None => "lexicographical tie-break".to_owned(),
                    };
                    eprintln!(
                        "warning: conflicting records found for collection {} node {} ({} distinct payloads); selecting newest ({ts_msg}) from `{}`",
                        format_id(&collection_id),
                        hex::encode(node_id),
                        distinct_payloads,
                        winner.0.display()
                    );
                    winner.1
                }
            };
            resolved_nodes.insert(node_id, data);
        }

        let ordered_hashes: Vec<[u8; 16]> = if topo {
            let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
            for (node_id, data) in &resolved_nodes {
                let edges = extract_matrix_edges(node_id, &data.bytes);
                adjacency.insert(*node_id, edges);
            }
            let mut live_hashes: Vec<[u8; 16]> = resolved_nodes.keys().copied().collect();
            live_hashes.sort_unstable();
            let csr = mtxdb::csr::Csr::build_from_edges(&live_hashes, &adjacency);
            if let Some(layered) = csr.layered_order() {
                layered
                    .into_iter()
                    .map(|idx| *csr.hash_of(idx).expect("valid local id"))
                    .collect()
            } else {
                eprintln!(
                    "warning: cyclic graph detected in collection {}; falling back to chronological order",
                    format_id(&collection_id)
                );
                let mut sorted = live_hashes;
                sorted.sort_by(|a, b| {
                    let ts_a = resolved_nodes
                        .get(a)
                        .and_then(|d| extract_origin_server_ts(&d.bytes));
                    let ts_b = resolved_nodes
                        .get(b)
                        .and_then(|d| extract_origin_server_ts(&d.bytes));
                    ts_a.cmp(&ts_b).then_with(|| a.cmp(b))
                });
                sorted
            }
        } else {
            let mut sorted: Vec<[u8; 16]> = resolved_nodes.keys().copied().collect();
            sorted.sort_by(|a, b| {
                let ts_a = resolved_nodes
                    .get(a)
                    .and_then(|d| extract_origin_server_ts(&d.bytes));
                let ts_b = resolved_nodes
                    .get(b)
                    .and_then(|d| extract_origin_server_ts(&d.bytes));
                ts_a.cmp(&ts_b).then_with(|| a.cmp(b))
            });
            sorted
        };

        let mut entries: Vec<(mtxdb::NodeId, NodeData)> = Vec::with_capacity(ordered_hashes.len());
        for hash in ordered_hashes {
            if let Some(data) = resolved_nodes.remove(&hash) {
                entries.push((hash, data));
            }
        }

        let meta_pos = entries
            .iter()
            .position(|(id, _)| *id == mtxdb::COLLECTION_METADATA_RECORD_ID);
        if let Some(pos) = meta_pos {
            if pos != 0 {
                let meta_entry = entries.remove(pos);
                entries.insert(0, meta_entry);
            }
        }

        let node_count = entries.len();
        for chunk in entries.chunks(COALESCE_REPACK_CHUNK_SIZE) {
            target_store.put_many(&collection_id, chunk)?;
        }

        written_collections = written_collections.saturating_add(1);
        written_nodes = written_nodes.saturating_add(node_count);
    }

    target_store.persist_shard_collections()?;
    target_store.sync()?;
    drop(target_store);

    println!(
        "coalescing repack complete: {} collection{}, {} nodes written to {}",
        written_collections,
        if written_collections == 1 { "" } else { "s" },
        written_nodes,
        out_dir.display()
    );
    println!("post-repack shard state:");
    let target_cli = cli.with_dir(out_dir.to_path_buf());
    cmd_shards_single(&target_cli, false, false, None)?;
    Ok(())
}

fn cmd_repack(
    cli: &Cli,
    collection: Option<&str>,
    packs: &[String],
    all: bool,
    topo: bool,
    yes: bool,
) -> anyhow::Result<()> {
    let target = match (collection, packs.is_empty(), all) {
        (Some(collection), true, false) => {
            RepackTarget::Collection(parse_collection_selector(collection)?)
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

    if topo {
        println!("note: all records preserved, laid out dependencies first");
    } else {
        println!(
            "note: this pool's ordering policy is not topological; records written in id order"
        );
    }
    cmd_repack_target(cli, &target, topo, yes)
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
/// prompts for confirmation before performing any real repack. Repack keeps
/// every record: the same edge extraction lays out every collection the closure
/// touches, dependencies first, and nothing is dropped as unreachable.
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
    pack_ids: Vec<PackId>,
}

enum RepackTarget {
    Collection([u8; 16]),
    Packs(Vec<PackSelector>),
    All,
}

/// Parse one or more permanent pack selectors. Numeric slots and ranges are not
/// accepted because slots are recycled implementation details.
fn parse_pack_selectors(selectors: &[String]) -> anyhow::Result<Vec<PackSelector>> {
    let mut selectors_out = std::collections::BTreeSet::new();
    for selector in selectors {
        selectors_out.insert(parse_pack_id_selector(selector)?);
    }
    Ok(selectors_out.into_iter().collect())
}

fn resolve_repack_target(
    store: &PackfileStorage,
    target: &RepackTarget,
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    match target {
        RepackTarget::Collection(collection_id) => {
            if store.collection_index_info(collection_id).is_none() {
                bail!("collection {} not found", format_id(collection_id));
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
    selectors: &[PackSelector],
) -> anyhow::Result<(Vec<[u8; 16]>, Vec<u16>)> {
    let summaries = store.shard_summaries();
    let ids: Vec<PackId> = summaries.iter().map(|summary| summary.pack_id).collect();
    let slots = selectors
        .iter()
        .map(|selector| {
            let pack_id = selector.resolve(ids.iter())?;
            summaries
                .iter()
                .find(|summary| summary.pack_id == pack_id)
                .map(|summary| summary.slot)
                .with_context(|| format!("pack {selector} not found"))
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
            RepackTarget::Packs(selectors) => {
                let selectors = selectors
                    .iter()
                    .map(PackSelector::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("no collections reference selected packs: {selectors}");
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

    let pack_summaries: std::collections::HashMap<u16, (PackId, u64)> = preview_store
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
        .map(PackId::filename_stem)
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "this will repack {} collection{} across {} pack{} ({pack_labels})",
        collections.len(),
        if collections.len() == 1 { "" } else { "s" },
        shards.len(),
        if shards.len() == 1 { "" } else { "s" },
    );
    let label_width = "pack_0000000000000000".len();
    for slot in &shards {
        let (pack_id, bytes) = pack_summaries
            .get(slot)
            .copied()
            .unwrap_or((PackId([0; mtxdb::packfile::PACK_ID_LEN]), 0));
        println!("  {}: {:>9}", pack_id.filename_stem(), fmt_bytes(bytes));
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

/// Whether repack lays out the selected pool's records by dependency.
///
/// The pool's [`mtxdb::PoolPolicy::ordering`] decides: a topological policy
/// (`State`, `EventDag`) gets edge extraction and a roots-first layout; others
/// (`ServerInfo`) keep id order.
fn repack_uses_dependency_layout(cli: &Cli) -> anyhow::Result<bool> {
    let shard = cli.require_shard_type()?;
    Ok(mtxdb::matrix_pool_policies()
        .for_shard(shard)
        .ordering
        .is_topological())
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
                |summary| summary.pack_id.filename_stem(),
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
                        format_id(&collection_id),
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
                        format_id(&collection_id),
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
    yes: bool,
) -> anyhow::Result<()> {
    let Some(preview) = repack_preview(cli, target, topo)? else {
        return Ok(());
    };

    if !yes && !confirm("Apply this repack?")? {
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
                .is_none_or(|summary| !preview.pack_ids.contains(&summary.pack_id))
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
    if let Some(edges) = crate::state_hamt::extract_state_edges(data) {
        return edges;
    }
    let input = data.to_vec();
    let val: OwnedValue = match OwnedValue::parse_bytes(&input) {
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
                // Edge extraction uses the same node-id derivation as import
                // (`matrix_event_node_id`, BLAKE3-128). The storage callback
                // cannot return an error, so if that derivation ever fails,
                // omit this edge rather than panicking in the CLI.
                if let Ok(id) = matrix_event_node_id(s) {
                    edges.push(id);
                }
            }
        }
    }
    edges
}

/// The longest `rel_type` the importer records. Longer values are not real
/// relation types and are dropped, not errors.
const MAX_RELATION_TYPE_BYTES: usize = 255;

/// One Matrix event's auth-index fields, owned for safe batch retention.
///
/// The batch must outlive the reusable parsing scratch used by raw extraction;
/// therefore these fields deliberately do not borrow from a `MatrixEventView`
/// or from the source `OwnedValue`.
struct MatrixEventFields {
    event_id: String,
    prev: Vec<String>,
    auth: Vec<String>,
    /// `(target event id, relation type)` from `content.m.relates_to`.
    relation: Option<(String, String)>,
}

impl MatrixEventFields {
    fn from_owned(event: &rezzy::OwnedMatrixEvent) -> Self {
        Self {
            event_id: event.event_id.clone().unwrap_or_default(),
            prev: event.prev_events.clone(),
            auth: event.auth_events.clone(),
            relation: event.relates_to.clone(),
        }
    }

    fn from_event(event_id: &str, event: &OwnedValue) -> Self {
        let mut fields = Self {
            event_id: event_id.to_owned(),
            prev: Vec::new(),
            auth: Vec::new(),
            relation: None,
        };
        let OwnedValue::Object(object) = event else {
            return fields;
        };
        for (field, targets) in [
            ("prev_events", &mut fields.prev),
            ("auth_events", &mut fields.auth),
        ] {
            if let Some(OwnedValue::Array(values)) = object.get(field) {
                for value in values {
                    // Room versions 1 and 2 reference `[event_id, hashes]`.
                    let target = match value {
                        OwnedValue::String(id) => Some(id.as_str()),
                        OwnedValue::Array(parts) => parts.first().and_then(OwnedValue::as_str),
                        _ => None,
                    };
                    if let Some(target) = target {
                        targets.push(target.to_owned());
                    }
                }
            }
        }
        // One relation per event, from `rel_type` and `event_id`, as Synapse's
        // `event_relations` stores it. A bare `m.in_reply_to` reply fallback has no
        // `rel_type` and is not a relation there either.
        if let Some(OwnedValue::Object(content)) = object.get("content") {
            if let Some(OwnedValue::Object(relates_to)) = content.get("m.relates_to") {
                if let (Some(OwnedValue::String(rel_type)), Some(OwnedValue::String(target))) =
                    (relates_to.get("rel_type"), relates_to.get("event_id"))
                {
                    // `rel_type` is user-supplied and becomes a dictionary entry, so
                    // one too long to be a real identifier is dropped rather than
                    // failing the whole import. Relations are best-effort here.
                    if rel_type.len() <= MAX_RELATION_TYPE_BYTES {
                        fields.relation = Some((target.to_owned(), rel_type.to_owned()));
                    }
                }
            }
        }
        fields
    }

    fn as_record<'a>(&'a self, prev: &'a [&'a str], auth: &'a [&'a str]) -> EventRecord<'a> {
        EventRecord {
            event_id: self.event_id.as_str(),
            prev,
            auth,
            relation: self.relation.as_ref().map(|(target, rel_type)| {
                mtxdb::csr_adjacency::RelationRef {
                    target: target.as_str(),
                    rel_type: rel_type.as_str(),
                }
            }),
        }
    }
}

/// Record each Matrix event's `prev_events`, `auth_events` and relation in its
/// room's auth index, one atomic batch per room.
///
/// Parent references are Matrix event IDs, so a record identity other than
/// `/event_id` cannot name a referenced event; those templates record nothing.
///
/// Call this before writing the primary event records: a structural conflict (the
/// same event with different `prev` or `auth`) is rejected as a unit before any
/// event is written, and the batch is atomic, so a failure leaves no partial
/// adjacency. Recording is idempotent, so a later re-import repairs adjacency for
/// events that were written but not recorded.
fn record_csr_adjacency(
    db: &Database,
    template: &CollectionTemplate,
    events: &[OwnedValue],
) -> anyhow::Result<()> {
    record_csr_adjacency_with_raw(db, template, events, None).map(|_| ())
}

fn record_csr_adjacency_with_raw(
    db: &Database,
    template: &CollectionTemplate,
    events: &[OwnedValue],
    raw_adjacency: Option<&[rezzy::OwnedMatrixEvent]>,
) -> anyhow::Result<HashMap<String, u32>> {
    if !matches!(
        &template.record_id_rule.policy,
        FrameIdPolicy::Pointer { pointer } if pointer == "/event_id"
    ) {
        return Ok(HashMap::new());
    }
    let mut shorts = HashMap::with_capacity(events.len());
    // Ordered, so the order adjacency is recorded in is the same on every run.
    let mut rooms: BTreeMap<&str, Vec<MatrixEventFields>> = BTreeMap::new();
    for (index, event) in events.iter().enumerate() {
        let (Some(room_id), Some(event_id)) = (event_room_id(event), event_id(event)) else {
            continue;
        };
        let fields = raw_adjacency.and_then(|raw| raw.get(index)).map_or_else(
            || MatrixEventFields::from_event(event_id, event),
            MatrixEventFields::from_owned,
        );
        rooms.entry(room_id).or_default().push(fields);
    }
    for (room_id, fields) in rooms {
        let room = RoomAuth::new(room_id);
        for chunk in fields.chunks(MAX_BATCH_EVENTS) {
            // One contiguous reference buffer per edge kind avoids a nested
            // Vec allocation for every event while preserving EventRecord's
            // borrowed-slice API.
            let prev_capacity = chunk.iter().map(|field| field.prev.len()).sum();
            let auth_capacity = chunk.iter().map(|field| field.auth.len()).sum();
            let mut prev = Vec::with_capacity(prev_capacity);
            let mut auth = Vec::with_capacity(auth_capacity);
            let mut ranges = Vec::with_capacity(chunk.len());
            for field in chunk {
                let prev_start = prev.len();
                prev.extend(field.prev.iter().map(String::as_str));
                let auth_start = auth.len();
                auth.extend(field.auth.iter().map(String::as_str));
                ranges.push((prev_start, prev.len(), auth_start, auth.len()));
            }
            let records: Vec<EventRecord<'_>> = chunk
                .iter()
                .zip(ranges.iter())
                .map(|(field, &(prev_start, prev_end, auth_start, auth_end))| {
                    field.as_record(&prev[prev_start..prev_end], &auth[auth_start..auth_end])
                })
                .collect();
            let recorded = room
                .record_events(db, &records)
                .with_context(|| format!("recording auth adjacency for room {room_id}"))?;
            for (field, recorded) in chunk.iter().zip(recorded) {
                shorts.insert(field.event_id.clone(), recorded.short_id);
            }
        }
    }
    Ok(shorts)
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
            println!("  {}", format_id(collection_id));
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
            format_id(&collection_id)
        );
    }
    Ok(())
}

fn cmd_tombstones(cli: &Cli, collection: &str, keys: bool, detail: bool) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let store = open_store_read_only(cli)?;
    let counts = store
        .removal_counts(&collection_id)
        .ok_or_else(|| anyhow::anyhow!("no such collection: {}", format_id(&collection_id)))?;
    println!("collection:        {}", format_id(&collection_id));
    println!("live keys:         {}", counts.live_keys);
    println!("removed keys:      {}", counts.removed_keys);
    if detail {
        let stats = store.removal_stats(&collection_id)?;
        println!("frames:            {}", stats.frames);
        println!("  removal markers: {}", stats.marker_frames);
        println!("  superseded:      {}", stats.superseded_frames);
        println!("total bytes:       {}", stats.total_bytes);
        println!("reclaimable bytes: {}", stats.reclaimable_bytes);
    }
    if keys {
        for key in store.removed_keys(&collection_id)? {
            println!("removed {}", format_id(&key));
        }
    }
    Ok(())
}

fn cmd_remove(cli: &Cli, collection: &str, ids: &[String], yes: bool) -> anyhow::Result<()> {
    let collection_id = parse_collection_id(collection)?;
    let mut node_ids: Vec<[u8; 16]> = ids
        .iter()
        .map(|id| parse_node_id(id))
        .collect::<anyhow::Result<_>>()?;
    node_ids.sort_unstable();
    node_ids.dedup();

    if !yes {
        println!(
            "This will remove {} record(s) from collection {}:",
            node_ids.len(),
            format_id(&collection_id)
        );
        for node_id in &node_ids {
            println!("  {}", format_id(node_id));
        }
        if !confirm("Remove these records?")? {
            println!("aborted");
            return Ok(());
        }
    }

    let store = open_store(cli)?;
    let mut removed = 0usize;
    for node_id in &node_ids {
        if store.remove_record(&collection_id, node_id)? {
            removed = removed.saturating_add(1);
        } else {
            eprintln!("skipped {}: not a live record", format_id(node_id));
        }
    }
    store.sync()?;
    println!(
        "removed {removed} of {} record(s) from collection {}",
        node_ids.len(),
        format_id(&collection_id)
    );
    Ok(())
}

/// Opens the store as writer (which always does a full scan and builds
/// the shard→collection directory and shard stats in memory regardless of
/// whether either has ever been persisted), then persists both —
/// bootstrapping `shard_stats.bin`/`shard_collections.bin` for a store whose
/// writer process has never called `sync_all`, or just refreshing them
/// on demand.
fn cmd_sync(cli: &Cli, all: bool) -> anyhow::Result<()> {
    run_multi_dir(cli, |sub_cli| cmd_sync_single(sub_cli, all))
}

/// Force a full checkpoint in every pool through the shared `Database`, so
/// each pool reports coverage to the same journal coordinator and the WAL can
/// reclaim through the oldest covered group.
fn cmd_reclaim_wal(cli: &Cli) -> anyhow::Result<()> {
    let layout = open_layout(cli)?;
    let db = open_database(layout.root().to_path_buf())
        .context("opening the shared database to reclaim its WAL")?;
    let before = db.coordinator().segment_len();

    for shard in ShardType::ALL {
        db.pool(shard)
            .force_index_checkpoint()
            .with_context(|| format!("checkpointing the {shard:?} pool to reclaim the WAL"))?;
    }
    for shard in ShardType::ALL {
        db.pool(shard).wait_for_checkpoint();
    }

    let coordinator = db.coordinator();
    let after = coordinator.segment_len();
    println!("shared WAL: {before} -> {after} bytes");
    let blockers = coordinator.reclaim_blockers();
    if !blockers.is_empty() {
        println!("reclaim still blocked by pools: {blockers:?}");
    }
    Ok(())
}

fn cmd_sync_single(cli: &Cli, all: bool) -> anyhow::Result<()> {
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
            let store = open_pool(dir.clone()).with_context(|| {
                let lock_path = dir.join(".mtxdb.lock");
                if lock_path.exists() {
                    format!(
                        "failed to open store for writing at `{}`: a lock file exists and could not be replaced (check directory and lock file permissions, e.g. `sudo chown -R $USER <DIR>`)",
                        dir.display()
                    )
                } else {
                    format!("failed to open store for writing at `{}`", dir.display())
                }
            })?;
            store.sync()?;
            eprintln!(
                "{}: synced: persisted shard IO stats and shard\u{2192}collection directory",
                shard_type.as_str()
            );
        }
        return Ok(());
    }
    let store = open_store(cli)?;
    store.sync()?;
    eprintln!(
        "synced: persisted shard IO stats, shard\u{2192}collection directory, and disk sizes"
    );
    Ok(())
}

#[cfg(test)]
#[path = "test_cmd.rs"]
mod tests;

/// Test-only experiment: fill-only seeding of a missing-prev gap root's base
/// state from the `(type, state_key)` keys its cited `auth_events` name. It
/// fills only keys the inherited base lacks, skips conflicting, unresolvable,
/// non-state and rejected citations (counting each), and builds the seeded base
/// incomplete with a fresh instance id from its own content and the real prev
/// parents. It never exists in a non-test build.
#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "test-only experiment counters"
)]
pub(super) mod seed_experiment {
    use super::*;
    use std::cell::RefCell;

    /// Counters for one armed run.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub(crate) struct SeedStats {
        pub(crate) gap_roots: u64,
        pub(crate) cited: u64,
        pub(crate) filled: u64,
        pub(crate) agree_with_inherited: u64,
        pub(crate) differs_from_inherited: u64,
        pub(crate) missing_body: u64,
        pub(crate) non_state: u64,
        pub(crate) rejected: u64,
        pub(crate) conflicts: u64,
        /// Event ids whose own citations supplied a seed.
        pub(crate) suppliers: HashSet<String>,
    }

    struct Armed {
        rejected: HashSet<String>,
        stats: SeedStats,
    }

    thread_local! {
        static ARMED: RefCell<Option<Armed>> = const { RefCell::new(None) };
    }

    /// Arm seeding for this thread with the baseline walk's final rejections.
    pub(crate) fn arm(rejected: HashSet<String>) {
        ARMED.with(|cell| {
            *cell.borrow_mut() = Some(Armed {
                rejected,
                stats: SeedStats::default(),
            });
        });
    }

    /// Disarm and return what the armed run counted.
    pub(crate) fn disarm() -> SeedStats {
        ARMED.with(|cell| {
            cell.borrow_mut()
                .take()
                .map(|a| a.stats)
                .unwrap_or_default()
        })
    }

    #[allow(clippy::too_many_arguments, reason = "mirrors the walk's own inputs")]
    pub(super) fn seed_gap_root(
        base: &Arc<SharedState>,
        event: Option<&OwnedValue>,
        id_map: &HashMap<String, u64>,
        events_by_sid: &HashMap<u64, &OwnedValue>,
        parent_instances: &[[u8; STATE_GROUP_ID_LENGTH]],
        room_id: &str,
        policy: ResolutionPolicy,
    ) -> Option<Arc<SharedState>> {
        ARMED.with(|cell| {
            let mut guard = cell.borrow_mut();
            let armed = guard.as_mut()?;
            let event = event?;
            armed.stats.gap_roots += 1;
            let supplier = event_id(event).unwrap_or("").to_owned();
            let cited_ids: Vec<&str> = event
                .get("auth_events")
                .and_then(OwnedValue::as_array)
                .map(|a| a.iter().filter_map(OwnedValue::as_str).collect())
                .unwrap_or_default();
            let mut keyed: std::collections::BTreeMap<(String, String), String> =
                std::collections::BTreeMap::new();
            let mut conflicted: HashSet<(String, String)> = HashSet::new();
            for id in cited_ids {
                armed.stats.cited += 1;
                if armed.rejected.contains(id) {
                    armed.stats.rejected += 1;
                    continue;
                }
                let Some(body) = id_map.get(id).and_then(|sid| events_by_sid.get(sid)) else {
                    armed.stats.missing_body += 1;
                    continue;
                };
                if !is_state_event(body) {
                    armed.stats.non_state += 1;
                    continue;
                }
                let key = (
                    event_string_field(body, "type").unwrap_or("").to_owned(),
                    state_key(body),
                );
                match keyed.get(&key) {
                    Some(existing) if existing != id => {
                        if conflicted.insert(key) {
                            armed.stats.conflicts += 1;
                        }
                    }
                    _ => {
                        keyed.insert(key, id.to_owned());
                    }
                }
            }
            let mut seeded = base.set.clone();
            let mut changed = false;
            for (key, id) in keyed {
                if conflicted.contains(&key) {
                    continue;
                }
                match seeded.entries.get(&key) {
                    None => {
                        seeded.set(&key.0, &key.1, id);
                        armed.stats.filled += 1;
                        changed = true;
                    }
                    Some(existing) if *existing == id => armed.stats.agree_with_inherited += 1,
                    Some(_) => armed.stats.differs_from_inherited += 1,
                }
            }
            if !changed {
                return None;
            }
            armed.stats.suppliers.insert(supplier);
            let instance = state_group_instance_id(
                room_id,
                parent_instances,
                &seeded.lthash(),
                policy.derivation,
            );
            Some(SharedState::with_parents(
                seeded,
                instance,
                false,
                parent_instances.to_vec(),
            ))
        })
    }
}
