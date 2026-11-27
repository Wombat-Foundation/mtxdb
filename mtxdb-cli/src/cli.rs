//! CLI for the mtxdb content-addressed storage engine.

mod auth_check;
mod cmd;
mod export_import;
mod gap_manifest;
mod interchange;
mod state_hamt;
mod state_policy;
mod state_resolve;

#[cfg(test)]
#[path = "../../src/test_scratch.rs"]
mod test_scratch;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Context as _;
use clap::{Arg, ArgAction, Command};
use mtxdb::{ReadPlanPolicy, ShardType};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static DEBUG_ENABLED: OnceLock<bool> = OnceLock::new();
/// Parsed once from `MTXDB_*` in `main`. Tests that call `cmd::run` directly
/// fall back to the library defaults instead of reading the ambient
/// environment.
static RUNTIME_CONFIG: OnceLock<mtxdb::RuntimeConfig> = OnceLock::new();

pub(crate) fn runtime_config() -> &'static mtxdb::RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(mtxdb::RuntimeConfig::default)
}

fn shard_type_from_arg(value: &str) -> Option<ShardType> {
    match value {
        "state" => Some(ShardType::State),
        "event" | "events" | "event-dag" => Some(ShardType::EventDag),
        "server-info" | "server_info" => Some(ShardType::ServerInfo),
        "all" => None,
        _ => unreachable!("clap validates shard type"),
    }
}

pub(crate) fn debug_enabled() -> bool {
    DEBUG_ENABLED.get().copied().unwrap_or(false)
}

/// Remove leftover test databases from runs whose process is gone.
///
/// The test helpers only clear their own path before creating it, so a crashed
/// or interrupted `cargo test` leaves a database behind and back-to-back runs
/// eventually fill the temp filesystem. Sweep once per test process and drop
/// directories whose owning PID is dead. The current PID is skipped, which
/// keeps concurrently running test binaries safe.
#[cfg(test)]
pub(crate) fn sweep_stale_test_dirs() {
    use std::sync::Once;
    static SWEEP: Once = Once::new();
    SWEEP.call_once(|| {
        const PREFIXES: [&str; 6] = [
            "mtxdb-cli-test-",
            "mtxdb-closure-test-",
            "mtxdb-state-keyframes-",
            "mtxdb-run-test-",
            "mtxdb-packidx-test-",
            "mtxdb-test-room-flag-",
        ];
        let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
            return;
        };
        let current = std::process::id();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(rest) = PREFIXES.iter().find_map(|prefix| name.strip_prefix(prefix)) else {
                continue;
            };
            let Some(pid) = rest
                .split('-')
                .next()
                .and_then(|pid| pid.parse::<u32>().ok())
            else {
                continue;
            };
            if pid == current || test_process_is_alive(pid) {
                continue;
            }
            let _ = std::fs::remove_dir_all(entry.path());
        }
    });
}

#[cfg(all(test, target_os = "linux"))]
fn test_process_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(all(test, not(target_os = "linux")))]
fn test_process_is_alive(_pid: u32) -> bool {
    // There is no cheap portable liveness probe, so leave other processes'
    // directories alone rather than risk deleting a live test database.
    true
}

#[derive(Clone)]
pub(crate) struct Cli {
    pub(crate) dirs: Vec<PathBuf>,
    pub(crate) shard_type: Option<ShardType>,
    pub(crate) coalesce: bool,
    pub(crate) read_plan: ReadPlanPolicy,
    pub(crate) command: Commands,
}

impl Cli {
    /// Return the single selected shard type, or bail if `-t all` was used.
    pub(crate) fn require_shard_type(&self) -> anyhow::Result<ShardType> {
        self.shard_type
            .context("this command requires a specific shard type (-t state or -t events)")
    }

    /// Iterate over the selected shard types (one if specific, all three if `-t all`).
    pub(crate) fn shard_types(&self) -> impl Iterator<Item = ShardType> + '_ {
        self.shard_type.into_iter().chain(
            self.shard_type
                .is_none()
                .then_some(ShardType::ALL)
                .into_iter()
                .flatten(),
        )
    }

    /// Return the single directory target or the default ".", bailing if multiple were specified.
    pub(crate) fn require_single_dir(&self, cmd: &str) -> anyhow::Result<&std::path::Path> {
        if self.dirs.len() > 1 {
            anyhow::bail!(
                "`mtxdb {cmd}` accepts only a single --dir target, but {} were provided",
                self.dirs.len()
            );
        }
        Ok(self.single_dir())
    }

    /// Return the first directory target or the default ".".
    pub(crate) fn single_dir(&self) -> &std::path::Path {
        self.dirs
            .first()
            .map_or_else(|| std::path::Path::new("."), PathBuf::as_path)
    }

    /// Return all target directories, defaulting to `["."]` if none were specified.
    pub(crate) fn dirs_or_default(&self) -> Vec<PathBuf> {
        if self.dirs.is_empty() {
            vec![PathBuf::from(".")]
        } else {
            self.dirs.clone()
        }
    }

    /// Create a clone targeting a single specific directory.
    pub(crate) fn with_dir(&self, dir: PathBuf) -> Self {
        Self {
            dirs: vec![dir],
            shard_type: self.shard_type,
            coalesce: self.coalesce,
            read_plan: self.read_plan,
            command: self.command.clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) enum Commands {
    Put {
        collection: String,
        id: String,
        data: String,
    },
    Get {
        collection: Option<String>,
        id: String,
        raw: bool,
        verbose: bool,
        header: bool,
        decode: Option<String>,
    },
    Collections {
        all: bool,
        layout: bool,
        canonical: bool,
        per_db: bool,
        sort: Option<String>,
        limit: i64,
    },
    Stats {
        json: bool,
    },
    Memory {
        evict_pages: bool,
    },
    Timeline {
        room: String,
        limit: i64,
        reverse: bool,
    },
    Flags {
        room: String,
        flag: Option<String>,
        not: bool,
        count: bool,
        limit: usize,
        rebuild: bool,
    },
    Meta {
        target: String,
        json: bool,
        limit: i64,
        offset: i64,
        decode: Option<String>,
    },
    Info {
        /// Positional collection, room, or pack selector.
        selector: Option<String>,
        stats: bool,
        tombstones: bool,
        keys: bool,
        detail: bool,
        json: bool,
    },
    Scan {
        selector: String,
        verbose: bool,
        header: bool,
        decode: Option<String>,
        limit: i64,
        id: Option<String>,
        collection: Option<String>,
        raw: bool,
        sort: Option<String>,
        reverse: bool,
        live: bool,
    },
    Search {
        collection: String,
        template: Option<PathBuf>,
        term: String,
        format: String,
        limit: i64,
    },
    Import {
        paths: Vec<PathBuf>,
        collection: Option<String>,
        template: Option<PathBuf>,
        no_state_hamts: bool,
        require_complete_prevs: bool,
        allow_waived_gaps: bool,
    },
    Gaps {
        action: GapAction,
    },
    Export {
        collection: String,
        format: String,
        metadata: bool,
    },
    Packs {
        action: PacksAction,
    },
    Repack {
        collection: Option<String>,
        packs: Vec<String>,
        all: bool,
        out: Option<PathBuf>,
        yes: bool,
    },
    Drop {
        collections: Vec<String>,
        yes: bool,
    },
    RoomCheck {
        room: String,
        auth: bool,
        prevs: bool,
        state_members: bool,
        status: Option<String>,
        history: bool,
        limit: usize,
    },
    Members {
        room: String,
        json: bool,
    },
    Remove {
        collection: String,
        ids: Vec<String>,
        yes: bool,
    },
    Completions {
        shell: String,
    },
    Sync {
        all: bool,
    },
    ReclaimWal,
    Init,
}

/// Sub-operations for the read-only `packs` inspection group.
#[derive(Clone)]
pub(crate) enum PacksAction {
    /// List open packs with size and IO/sync stats.
    List {
        all: bool,
        layout: bool,
        sort: Option<String>,
    },
    /// Scan a pack's physical frames.
    Scan {
        selector: String,
        verbose: bool,
        header: bool,
        decode: Option<String>,
        limit: i64,
        id: Option<String>,
        collection: Option<String>,
        raw: bool,
        sort: Option<String>,
        reverse: bool,
    },
    /// Dump every frame in one pack as decoded JSONL.
    Dump {
        pack: String,
        collection: Option<String>,
        out: Option<PathBuf>,
    },
    /// Extract frames for one collection into a new standalone pack.
    Extract {
        pack: String,
        collection: String,
        out: PathBuf,
    },
}

/// Explicit actions over the per-room missing-prev manifest.
#[derive(Clone)]
pub(crate) enum GapAction {
    Ingest {
        room: String,
        run: String,
        report: PathBuf,
    },
    Waive {
        room: String,
        gap: String,
        by: String,
        reason: String,
    },
    Anchor {
        room: String,
        gap: String,
        event: String,
        after: bool,
        source: String,
    },
}

fn build_cli() -> Command {
    let cmd = global_args(Command::new("mtxdb").subcommand_precedence_over_arg(true))
        .version(concat!(
            env!("CARGO_PKG_VERSION"),
            " (",
            env!("GIT_DESCRIBE"),
            ")"
        ))
        .disable_version_flag(true)
        .arg(
            Arg::new("version")
                .short('v')
                .long("version")
                .action(ArgAction::SetTrue)
                .help("Print version"),
        )
        .about("CLI for the mtxdb content-addressed storage engine")
        .subcommand(sub_collections())
        .subcommand(sub_stats())
        .subcommand(sub_memory())
        .subcommand(sub_meta())
        .subcommand(sub_sync())
        .subcommand(sub_reclaim_wal())
        .subcommand(sub_completions())
        .subcommand(sub_import())
        .subcommand(sub_export())
        .subcommand(sub_packs())
        .subcommand(sub_repack())
        .subcommand(sub_search())
        .subcommand(sub_scan())
        .subcommand(sub_info())
        .subcommand(sub_drop())
        .subcommand(sub_remove())
        .subcommand(sub_put())
        .subcommand(sub_get())
        .subcommand(sub_init())
        .subcommand(sub_room());
    scope_flags(cmd)
}

/// Commands that aggregate across several database roots (`--coalesce`).
const COALESCE_COMMANDS: &[&[&str]] = &[
    &["collections"],
    &["stats"],
    &["get"],
    &["scan"],
    &["info"],
    &["packs", "list"],
    &["packs", "scan"],
    &["repack"],
];

/// Commands that run batch reads, where `--read-plan` changes how they are issued.
const READ_PLAN_COMMANDS: &[&[&str]] = &[
    &["get"],
    &["scan"],
    &["export"],
    &["import"],
    &["info"],
    &["room", "check"],
    &["repack"],
];

/// Commands that choose an independent pool with `-t`. The rest (room
/// commands, `search`, `meta`, `reclaim-wal`, `init`) work on fixed pools or
/// on the whole database, so the flag would be ignored there.
const SHARD_TYPE_COMMANDS: &[&[&str]] = &[
    &["collections"],
    &["stats"],
    &["memory"],
    &["get"],
    &["put"],
    &["remove"],
    &["drop"],
    &["scan"],
    &["info"],
    &["import"],
    &["sync"],
    &["repack"],
    &["export"],
    &["packs"],
];

/// Attach `--coalesce`, `--read-plan` and `--shard-type` only to the commands
/// that honor them, so every other command's help stays free of them. A path
/// names a nested subcommand, e.g. `["room", "auth-check"]`.
fn scope_flags(mut cmd: Command) -> Command {
    type Scope = (&'static [&'static [&'static str]], fn() -> Arg);
    let scopes: [Scope; 3] = [
        (COALESCE_COMMANDS, coalesce_arg),
        (READ_PLAN_COMMANDS, read_plan_arg),
        (SHARD_TYPE_COMMANDS, shard_type_arg),
    ];
    for (paths, arg) in scopes {
        for path in paths {
            cmd = attach_to_path(cmd, path, arg());
        }
    }
    cmd
}

fn attach_to_path(cmd: Command, path: &[&str], arg: Arg) -> Command {
    let Some((name, rest)) = path.split_first() else {
        return cmd.arg(arg);
    };
    cmd.mut_subcommand(*name, |sub| attach_to_path(sub, rest, arg))
}

/// The flag shared by every subcommand (`--dir`), plus `--debug`.
fn global_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("dir")
            .short('d')
            .long("dir")
            .env("MTXDB_DIR")
            .value_name("DIR")
            .action(ArgAction::Append)
            .num_args(1)
            .global(true)
            .help("Database root directory (or multiple directories)"),
    )
    .arg(
        Arg::new("debug")
            .long("debug")
            .action(ArgAction::SetTrue)
            .global(true)
            .help("Enable additional diagnostic output"),
    )
}

fn coalesce_arg() -> Arg {
    Arg::new("coalesce")
        .short('c')
        .long("coalesce")
        .action(ArgAction::SetTrue)
        .global(true)
        .help("Aggregate results across multiple database roots (repack requires --out)")
}

fn shard_type_arg() -> Arg {
    Arg::new("shard_type")
        .short('t')
        .long("shard-type")
        .env("MTXDB_SHARD_TYPE")
        .value_name("TYPE")
        .default_value("events")
        .value_parser([
            "state",
            "event",
            "events",
            "event-dag",
            "server-info",
            "server_info",
            "all",
        ])
        .hide_possible_values(true)
        .global(true)
        .help("Independent shard pool to operate on (use 'all' to target every pool)")
}

fn read_plan_arg() -> Arg {
    Arg::new("read_plan")
        .long("read-plan")
        .value_name("MODE")
        .default_value("plain")
        .value_parser(["plain", "prefetch"])
        .global(true)
        .help("Merged read prefetch for batch reads: 'plain' (default, independent reads) or 'prefetch' to meld nearby candidates into sequential extents")
}

fn sub_stats() -> Command {
    Command::new("stats")
        .about("Runtime, open, and persisted pool statistics")
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .help("Emit machine-readable JSON instead of a table"),
        )
}

fn sub_memory() -> Command {
    Command::new("memory")
        .about("Report in-process cache state or request filesystem page eviction")
        .long_about(
            "Report heap objects decoded by the current mtxdb process, or request eviction of the selected database's filesystem pages.\n\n".to_owned()
                + "The report does not include durable records, indexes, memory-mapped files, or "
                + "the operating system's filesystem page cache. A one-shot CLI process "
                + "starts with an empty decoded-node cache, so this report is primarily useful "
                + "with --evict-pages. That flag is a Linux-only, best-effort request "
                + "to discard clean page-cache pages for files under the database directory; "
                + "the kernel may retain dirty or actively mapped pages.",
        )
        .arg(
            Arg::new("evict-pages")
                .long("evict-pages")
                .action(ArgAction::SetTrue)
                .help("Request eviction of clean filesystem pages for database files (Linux; best effort)"),
        )
}

fn sub_timeline() -> Command {
    Command::new("timeline")
        .about("Page events in a room's timeline in topological and stream order")
        .arg(
            Arg::new("room")
                .help("The room id (e.g. !room:example.org)")
                .required(true)
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .short('l')
                .help("Maximum number of events to show")
                .value_name("N")
                .default_value("50")
                .value_parser(clap::value_parser!(i64)),
        )
        .arg(
            Arg::new("reverse")
                .long("reverse")
                .short('r')
                .action(ArgAction::SetTrue)
                .help("Page timeline backwards from newest to oldest"),
        )
}

fn sub_flags() -> Command {
    Command::new("flags")
        .about("Query or count room events by disposition or property flags")
        .arg(
            Arg::new("room")
                .help("The room id (e.g. !room:example.org)")
                .required(true)
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .arg(
            Arg::new("flag")
                .long("flag")
                .short('f')
                .help("Flag name to filter by (present, timeline, outlier, rejected, soft-failed, redacted)")
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .arg(
            Arg::new("not")
                .long("not")
                .action(ArgAction::SetTrue)
                .requires("flag")
                .help("Invert the flag query (events without this flag)"),
        )
        .arg(
            Arg::new("count")
                .long("count")
                .action(ArgAction::SetTrue)
                .help("Print only the count of matching events"),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .short('l')
                .value_name("N")
                .help("Maximum number of events to select")
                .default_value("50")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("rebuild")
                .long("rebuild")
                .action(ArgAction::SetTrue)
                .help("Rebuild the room flag index from authoritative records"),
        )
}

fn sub_meta() -> Command {
    Command::new("meta")
        .about("Inspect on-disk metadata and durability artifacts (read-only)")
        .arg(
            Arg::new("target")
                .value_name("TARGET")
                .default_value("overview")
                .value_parser([
                    "overview",
                    "db",
                    "wal",
                    "checkpoint",
                    "delta",
                    "sidecars",
                    "packs",
                    "locks",
                    "raw",
                ])
                .help("Artifact group to inspect (default: overview)"),
        )
        .arg(limit_arg())
        .arg(
            Arg::new("offset")
                .long("offset")
                .default_value("0")
                .value_parser(clap::value_parser!(i64))
                .help("Number of decoded records to skip"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .help("Emit diagnostic records as JSON"),
        )
        .arg(
            Arg::new("decode")
                .long("decode")
                .value_name("FORMAT")
                .num_args(0..=1)
                .default_missing_value("auto")
                .require_equals(true)
                .value_parser(["auto", "json", "raw"])
                .help("Decode WAL payloads when possible (json or raw)"),
        )
}

fn sub_collections() -> Command {
    Command::new("collections")
        .about("List logical collections in the selected shard pool")
        .arg(all_arg("List collections in every independent pool"))
        .arg(layout_arg())
        .arg(
            Arg::new("canonical")
                .long("canonical")
                .action(ArgAction::SetTrue)
                .help("Show each collection's canonical ID"),
        )
        .arg(
            Arg::new("per_db")
                .long("per-db")
                .action(ArgAction::SetTrue)
                .help("With --coalesce, also print one totals row per database"),
        )
        .arg(limit_arg())
        .arg(sort_arg("collection, nodes, idx-load (alias: load), shards, index, disk, packs, avoidable, segments, fragmentation"))
}

fn all_arg(help: &'static str) -> Arg {
    Arg::new("all")
        .short('a')
        .long("all")
        .action(ArgAction::SetTrue)
        .help(help)
}

fn yes_arg(help: &'static str) -> Arg {
    Arg::new("yes")
        .short('y')
        .long("yes")
        .action(ArgAction::SetTrue)
        .help(help)
}

fn layout_arg() -> Arg {
    Arg::new("layout")
        .long("layout")
        .action(ArgAction::SetTrue)
        .help("Scan physical pack layout and show fragmentation columns and summary")
}

fn limit_arg() -> Arg {
    Arg::new("limit")
        .short('l')
        .long("limit")
        .value_name("N")
        .default_value("50")
        .allow_hyphen_values(true)
        .value_parser(clap::value_parser!(i64))
        .help("Maximum rows to show (default: 50; 0 or negative means unlimited)")
}

fn sort_arg(help: &'static str) -> Arg {
    Arg::new("sort")
        .short('s')
        .long("sort")
        .value_name("COLUMN")
        .help(help)
}

fn sub_init() -> Command {
    Command::new("init")
        .about("Create a new mtxdb database root (db.meta); pools are created on first write.")
}

fn sub_sync() -> Command {
    Command::new("sync")
        .about("Bootstrap or refresh persisted shard stats")
        .arg(all_arg(
            "Sync every independent pool, not just the selected `-t` one",
        ))
}

fn sub_reclaim_wal() -> Command {
    Command::new("reclaim-wal")
        .about("Checkpoint every pool and reclaim the shared write-ahead log")
        .long_about(
            "Checkpoint every pool and reclaim the shared write-ahead log.\n\n".to_owned()
                + "`-t` is ignored: a group in the shared segment can only be dropped once \
                   every pool that contributed a frame to it has reported coverage, so all \
                   pools are checkpointed. The report covers bytes, retained groups, the LSN \
                   and sequence windows, each pool's coverage, and checkpoint timings. It is \
                   printed to stdout/stderr only; mtxdb writes no log file for it.",
        )
}

fn sub_completions() -> Command {
    Command::new("completions")
        .about("Print shell completion script")
        .hide(true)
        .display_order(usize::MAX)
        .arg(Arg::new("shell").required(true).value_parser([
            "bash",
            "elvish",
            "fish",
            "powershell",
            "zsh",
        ]))
}

fn sub_scan() -> Command {
    scan_command(false)
}

fn sub_pack_scan() -> Command {
    scan_command(true)
}

fn scan_command(pack_only: bool) -> Command {
    let selector = Arg::new("selector")
        .required(true)
        .value_name(if pack_only { "PACK_ID" } else { "COLLECTION" })
        .help(if pack_only {
            "Pack identity (full 0x-prefixed id or unique 1-16 hex filename prefix)"
        } else {
            "Collection ID (0x-prefixed 32 hex digits) or canonical room ID"
        });
    Command::new("scan")
        .about(if pack_only {
            "Scan a pack's physical records"
        } else {
            "Scan a collection's physical records"
        })
        .arg(selector)
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .action(ArgAction::SetTrue)
                .help("Print each frame's payload when available"),
        )
        .arg(
            Arg::new("header")
                .long("header")
                .action(ArgAction::SetTrue)
                .help("Print frame header details (offsets, sizes, flags, and metadata TLVs)"),
        )
        .arg(
            Arg::new("decode")
                .long("decode")
                .value_name("FORMAT")
                .num_args(0..=1)
                .default_missing_value("auto")
                .require_equals(true)
                .value_parser(["auto", "json", "hamt", "state", "raw"])
                .help("Decode and display payload format (e.g. json, hamt, state, raw, or auto)"),
        )
        .arg(
            Arg::new("id")
                .long("id")
                .value_name("NODE_ID")
                .help("Restrict the physical scan to one node ID"),
        )
        .arg(Arg::new("raw").long("raw").action(ArgAction::SetTrue).help(
            "Write matching frames' payload bytes verbatim, concatenated, to stdout (all \
                     matches, or up to --limit; use -l 0 for no cap). With --verbose, a per-frame \
                     context line goes to stderr so stdout stays a clean byte stream",
        ))
        .arg(sort_arg("payload, offset"))
        .arg(
            Arg::new("reverse")
                .short('R')
                .long("reverse")
                .action(ArgAction::SetTrue)
                .help("Reverse the sort order (descending) when used with -s"),
        )
        .arg(limit_arg())
        .args(if pack_only {
            vec![Arg::new("collection")
                .long("collection")
                .value_name("COLLECTION")
                .help("Restrict the scan to this collection")]
        } else {
            Vec::new()
        })
        .args(if pack_only {
            Vec::new()
        } else {
            vec![Arg::new("live")
                .long("live")
                .alias("indexed")
                .action(ArgAction::SetTrue)
                .help(
                    "Read the live collection index instead of scanning physical frames; \
                         superseded frames and tombstones are omitted",
                )]
        })
}

fn sub_search() -> Command {
    Command::new("search")
        .about("Search template-declared string fields in a collection")
        .long_about(
            "Stream the collection's JSON event records and match a case-insensitive substring \
             against the RFC 6901 pointers in the template's `search.fields` array. This reads \
             the primary collection directly; query time grows with collection size.",
        )
        .arg(
            Arg::new("collection")
                .required(true)
                .value_name("COLLECTION")
                .help("Collection ID (0x-prefixed) or canonical Matrix room ID"),
        )
        .arg(
            Arg::new("term")
                .long("term")
                .required(true)
                .value_name("TEXT")
                .help("Case-insensitive substring to find in searchable fields"),
        )
        .arg(
            Arg::new("template")
                .long("template")
                .value_name("FILE")
                .help("Collection template declaring searchable JSON pointers"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .default_value("id")
                .value_parser(["id", "json", "text"])
                .help(
                    "Output per match: `id` prints the Matrix event ID (falling back to the \
                     node ID), `json` prints the full event, `text` prints \
                     `<timestamp> <sender> <type>` followed by the message body",
                ),
        )
        .arg(limit_arg())
}

fn sub_info() -> Command {
    Command::new("info")
        .about("Show database, collection, or pack storage info")
        .arg(
            Arg::new("selector")
                .required(false)
                .value_name("SELECTOR")
                .help(
                "Collection ID, canonical room ID, or pack ID; omit for database and pool metadata",
            ),
        )
        .arg(
            Arg::new("stats")
                .long("stats")
                .action(ArgAction::SetTrue)
                .help(
                    "For a collection, also scan every record for event statistics (room \
                     state, members, DAG health, activity, senders). Slower on large rooms.",
                ),
        )
        .arg(
            Arg::new("tombstones")
                .long("tombstones")
                .action(ArgAction::SetTrue)
                .requires("selector")
                .help("Also report live and removed key counts"),
        )
        .arg(
            Arg::new("detail")
                .long("detail")
                .action(ArgAction::SetTrue)
                .requires("tombstones")
                .help("With --tombstones, include frame counts and byte totals"),
        )
        .arg(
            Arg::new("keys")
                .long("keys")
                .action(ArgAction::SetTrue)
                .requires("tombstones")
                .help("With --tombstones, list removed keys"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .requires("tombstones")
                .help("With --tombstones, emit JSON instead of text"),
        )
}

fn sub_import() -> Command {
    Command::new("import")
        .about("Import Matrix events, federation documents, or mtxdb exports")
        .long_about(
            "Import Matrix federation events from a JSON document or JSONL event stream, or an \
             `mtxdb-export/v1` JSONL export, which this command detects from the first line. An \
             export is either a room export (`header`, `event`*, `end` records) or a `server_keys` \
             export (one `server_keys` record per line); an export is written in the \
             layout a server reads (events, locators, prev/auth edges, verdicts, signing keys), \
             and a run of exports only ends by committing the import marker; without it the \
             database is incomplete. A JSON document must contain a `pdus` array, an `auth_chain` array, \
             or both; a `.jsonl` file contains one event per line. Each imported event needs an \
             `event_id`. The collection identity comes from `collection_id` unless --collection \
             is supplied.",
        )
        .arg(
            Arg::new("path")
                .required(true)
                .num_args(1..)
                .value_name("FILE")
                .help(
                    "Matrix federation JSON documents, JSONL event streams, or mtxdb-export/v1 \
                     files",
                ),
        )
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .help("Collection ID (0x-prefixed, 32 hex digits). Auto-detected if omitted"),
        )
        .arg(
            Arg::new("template")
                .long("template")
                .value_name("FILE")
                .help("JSON collection template; currently supports matrix-event-v1"),
        )
        .arg(
            Arg::new("no-state-hamts")
                .long("no-state-hamts")
                .action(ArgAction::SetTrue)
                .help(
                    "Do not reconstruct state groups or Synapse-format state HAMTs during a room \
                     import. Use this when the target server builds its own state, for example \
                     congruent's v20 migration.",
                ),
        )
        .arg(
            Arg::new("require-complete-prevs")
                .long("require-complete-prevs")
                .action(ArgAction::SetTrue)
                .help(
                    "Fail the import if any room it touches still has an unresolved prev that \
                     is not anchored. A gap known to be absent on the queried servers still \
                     fails: absence is evidence, not completion. The failure leaves the \
                     import marker unwritten.",
                ),
        )
        .arg(
            Arg::new("allow-waived-gaps")
                .long("allow-waived-gaps")
                .action(ArgAction::SetTrue)
                .requires("require-complete-prevs")
                .help("With --require-complete-prevs, accept gaps an operator waived"),
        )
}

fn sub_gaps() -> Command {
    Command::new("gaps")
        .about("Inspect and record unresolved Matrix prev_events gaps for a room")
        .arg(room_arg())
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("ingest")
                .about("Record per-server outcomes from a rezzy federation gap-fill report")
                .arg(
                    Arg::new("run")
                        .long("run")
                        .required(true)
                        .value_name("ASSESSMENT_ID")
                        .help("Stable label for this gap-fill assessment; reuse it to re-ingest the same report"),
                )
                .arg(
                    Arg::new("report")
                        .required(true)
                        .value_name("REPORT.json")
                        .help("JSON report emitted by `rezzy federation gap-fill`"),
                ),
        )
        .subcommand(
            Command::new("waive")
                .about("Record a decision not to chase a gap (policy only; the gap stays a gap)")
                .arg(gap_id_arg())
                .arg(
                    Arg::new("by")
                        .long("by")
                        .required(true)
                        .value_name("WHO")
                        .help("Who made the decision"),
                )
                .arg(
                    Arg::new("reason")
                        .long("reason")
                        .required(true)
                        .value_name("TEXT")
                        .help("Why the gap is not being chased"),
                ),
        )
        .subcommand(
            Command::new("anchor")
                .about("Record a trusted state reference for a gap")
                .arg(gap_id_arg())
                .arg(
                    Arg::new("event")
                        .long("event")
                        .required(true)
                        .value_name("EVENT_ID")
                        .help("The event the reference describes"),
                )
                .arg(
                    Arg::new("boundary")
                        .long("boundary")
                        .required(true)
                        .value_name("before|after")
                        .value_parser(["before", "after"])
                        .help("Whether the reference is the state before or after --event (`/state_ids` is before)"),
                )
                .arg(
                    Arg::new("source")
                        .long("source")
                        .required(true)
                        .value_name("TEXT")
                        .help("Where the state came from, e.g. a server and request"),
                ),
        )
}

/// One room selector, shared by every room-scoped command.
fn room_arg() -> Arg {
    Arg::new("room")
        .required(true)
        .value_name("ROOM_ID")
        .value_parser(clap::builder::NonEmptyStringValueParser::new())
        .help("Matrix room ID, e.g. !room:server")
}

fn gap_id_arg() -> Arg {
    Arg::new("gap")
        .long("gap")
        .required(true)
        .value_name("MISSING_EVENT_ID")
        .help("The tracked gap (the missing prev event id)")
}

fn sub_export() -> Command {
    Command::new("export")
        .about("Export a collection's records as JSONL to stdout")
        .long_about(
            "Export a collection's stored records as one JSON value per line on stdout. The default \
             `jsonl` output is accepted by `mtxdb import`. A `!room:server` selector or a \
             `0x`-prefixed collection ID are both accepted. `--format envelope` (or `--metadata`) \
             emits a schema-versioned `mtxdb.export/v1` envelope per record with the exact stored \
             bytes, frame metadata (logical id, content digest, role), and a decoded JSON payload; \
             envelopes are inspection-only and are NOT accepted by `mtxdb import`.",
        )
        .arg(
            Arg::new("collection")
                .required(true)
                .value_name("COLLECTION")
                .help("Collection ID (0x-prefixed) or Matrix room ID (!room:server)"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .value_parser(["jsonl", "envelope"])
                .help("Output format: raw payload lines (jsonl, default) or schema-versioned envelopes (envelope)"),
        )
        .arg(
            Arg::new("metadata")
                .long("metadata")
                .action(ArgAction::SetTrue)
                .help("Include frame metadata (implies --format envelope)"),
        )
}

fn sub_packs() -> Command {
    Command::new("packs")
        .about("List, scan, dump, or extract pack data")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("list")
                .about("List open packs with size and IO/sync stats")
                .arg(all_arg("List packs in every independent pool"))
                .arg(layout_arg())
                .arg(sort_arg(
                    "pack, bytes, nodes, collections, syncs, segments, interleaving",
                )),
        )
        .subcommand(sub_pack_scan())
        .subcommand(sub_packs_dump())
        .subcommand(sub_packs_extract())
}

fn sub_packs_dump() -> Command {
    Command::new("dump")
        .about("Dump every frame in one pack as decoded JSONL")
        .long_about(
            "Dump every frame in a pack, in physical order and without deduplication, as one \
             JSON object per line: pool, pack identity, frame offset, collection id, node \
             id, payload, and frame metadata. `payload_base64` is the exact stored bytes; \
             `payload` is a decoded JSON convenience value (which may differ in whitespace, \
             key order, or numeric spelling) and is absent for binary payloads. Unlike \
             `export`, this is a complete view of one pack (all collections, superseded \
             frames included), not a collection's live set. Pack identities are globally \
             unique addresses, so if the same id is present in more than one selected pool \
             it is still rejected as ambiguous unless -t selects one.",
        )
        .arg(
            Arg::new("pack")
                .short('p')
                .long("pack")
                .required(true)
                .value_name("PACK_ID")
                .help("Pack identity (0x-prefixed full 32-hex id, or a unique 1-16 hex filename prefix)"),
        )
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .value_name("COLLECTION")
                .help("Only dump frames for this collection (0x-prefixed id or !room:server)"),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .value_name("FILE")
                .help("Write JSONL to FILE instead of stdout"),
        )
}

fn sub_packs_extract() -> Command {
    Command::new("extract")
        .about("Extract frames for one collection into a new standalone pack")
        .long_about(
            "Extract frames belonging to a collection from one pack into a new valid packfile, \
             copying frame bytes verbatim (preserving compression, metadata, and CRCs) with a \
             fresh pack header. The collection can be specified as a `0x`-prefixed 32-hex-digit ID \
             or a Matrix room ID (`!room:server`).",
        )
        .arg(
            Arg::new("pack")
                .short('p')
                .long("pack")
                .required(true)
                .value_name("PACK_ID")
                .help("Source pack identity (full 32-hex 0x-prefixed id, or a unique filename prefix)"),
        )
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .required(true)
                .value_name("COLLECTION")
                .help("Collection to extract (0x-prefixed id or !room:server)"),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .required(true)
                .value_name("FILE")
                .help("Path for the extracted packfile; it is stamped with a fresh random pack identity"),
        )
}

fn sub_repack() -> Command {
    Command::new("repack")
        .about("Trigger manual repack over a collection or pack closure")
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .conflicts_with("pack"),
        )
        .arg(
            Arg::new("pack")
                .short('p')
                .long("pack")
                .conflicts_with("collection")
                .num_args(1..)
                .value_name("PACK_ID")
                .help("Repack collections referencing packs (repeat -p for multiple pack IDs)"),
        )
        .arg(
            all_arg("Repack every collection in every active pack")
                .conflicts_with_all(["collection", "pack"]),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .value_name("DIR")
                .help("Destination directory for coalescing repack into a new canonical database"),
        )
        .arg(yes_arg("Skip interactive confirmation"))
}

fn sub_drop() -> Command {
    Command::new("drop")
        .about("Drop one or more collections and all their data")
        .arg(
            Arg::new("collection")
                .required(true)
                .num_args(1..)
                .value_name("COLLECTION")
                .help("Collection IDs to drop"),
        )
        .arg(yes_arg("Skip confirmation prompt"))
}

fn sub_auth_check() -> Command {
    Command::new("check")
        .about("Check a room's prev_events gaps and auth_events graph")
        .long_about(
            "Check room completeness across the `prev_events` gap manifest and the \
             embedded `auth_events` graph. By default both checks run; pass --prevs or \
             --auth to run only one. The auth graph is re-derived from event records and \
             compared with the embedded edge table. Reports: \
             short ids with no outbound edge record (a closure walk reports an incomplete \
             graph), events whose embedded edges differ from their `auth_events` (a \
             truncated or extra edge list), edge records with no event record, and \
             persisted state events that have no short id at all. With --state-members, \
             also compare state-HAMT members against auth short ids (O(groups × entries)). \
             The record layouts are \
             the homeserver's (sithnapse), not this crate's; an unknown record version is \
             reported, not guessed at. Reads the state and event pools. Read-only; \
             coexists with a running writer. The auth graph reader attaches to the root's \
             shared WAL, so `--auth` requires a database with a usable shared WAL.",
        )
        .arg(
            Arg::new("room")
                .required(true)
                .value_name("ROOM_ID")
                .help("Matrix room ID, e.g. !room:server"),
        )
        .arg(Arg::new("auth").long("auth").action(ArgAction::SetTrue).help("Check auth_events graph integrity (default when neither --auth nor --prevs is given)"))
        .arg(Arg::new("prevs").long("prevs").action(ArgAction::SetTrue).help("List unresolved prev_events gaps (default when neither --auth nor --prevs is given)"))
        .arg(
            Arg::new("status")
                .long("status")
                .value_name("STATUS")
                .value_parser(["open", "absent", "waived", "anchored"])
                .help("With prevs, only show gaps with this derived status"),
        )
        .arg(Arg::new("history").long("history").action(ArgAction::SetTrue).help("With prevs, print recorded server answers"))
        .arg(
            Arg::new("state-members")
                .long("state-members")
                .action(ArgAction::SetTrue)
                .help("Implies --auth; compare state HAMT members with auth short ids (O(groups × entries))"),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .short('l')
                .value_name("N")
                .value_parser(clap::value_parser!(usize))
                .default_value("50")
                .help("Maximum rows to print per category (0 for no cap)"),
        )
}

fn sub_members() -> Command {
    Command::new("members")
        .about("Count a room's latest membership per user, read from its event records")
        .long_about(
            "Scan the room's EventDag collection once and report each user's latest \
             membership by origin_server_ts, grouped by membership value. This counts \
             distinct state keys, not events: a user who joined and later left counts \
             once, as `leave`. Read-only; coexists with a running writer. It does not \
             read the current state HAMT, so an unapplied or unverifiable membership \
             event still contributes its latest value.",
        )
        .arg(
            Arg::new("room")
                .required(true)
                .value_name("ROOM")
                .help("Matrix room ID (starting with `!`)"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .action(ArgAction::SetTrue)
                .help("Print a single JSON object instead of text"),
        )
}

/// Room-scoped inspection and gap management. Every child selects its own
/// room, and none of them takes a shard pool or coalescing flag.
fn sub_room() -> Command {
    Command::new("room")
        .about("Inspect or manage one Matrix room")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(sub_timeline())
        .subcommand(sub_flags())
        .subcommand(sub_members())
        .subcommand(sub_auth_check())
        .subcommand(sub_gaps())
}

fn sub_remove() -> Command {
    Command::new("remove")
        .about("Remove one or more records from a collection")
        .long_about(
            "Append a removal marker for each key. Reads, scans and counts then treat the \
             key as absent; the space comes back at the next repack. A key that is not \
             live is skipped and writes nothing.",
        )
        .arg(
            Arg::new("collection")
                .required(true)
                .value_name("COLLECTION")
                .help("Namespace ID holding the records"),
        )
        .arg(
            Arg::new("id")
                .required(true)
                .num_args(1..)
                .value_name("ID")
                .help("Record IDs to remove"),
        )
        .arg(yes_arg("Skip confirmation prompt"))
}

fn sub_put() -> Command {
    Command::new("put")
        .about("Insert a record")
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .required(true),
        )
        .arg(Arg::new("id").short('i').long("id").required(true))
        .arg(Arg::new("data").short('a').long("data").required(true))
}

fn sub_get() -> Command {
    Command::new("get")
        .about("Retrieve a record")
        .arg(Arg::new("collection").short('r').long("collection"))
        .arg(
            Arg::new("id")
                .index(1)
                .required(true)
                .help("Node ID (0x-prefixed, 32 hex digits) or Matrix event ID"),
        )
        .arg(
            Arg::new("raw")
                .long("raw")
                .action(ArgAction::SetTrue)
                .help("Emit payload bytes verbatim instead of pretty-printing JSON"),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .action(ArgAction::SetTrue)
                .help("Print record and Matrix event metadata to stderr"),
        )
        .arg(
            Arg::new("header")
                .long("header")
                .action(ArgAction::SetTrue)
                .help("Print frame header and collection metadata details to stderr"),
        )
        .arg(
            Arg::new("decode")
                .long("decode")
                .value_name("FORMAT")
                .num_args(0..=1)
                .default_missing_value("auto")
                .require_equals(true)
                .help("Decode and display payload format (e.g. json, hamt, state, raw, or auto)"),
        )
}

/// Carry runtime read settings into the `--read-plan` policy. Commands apply
/// that policy after opening a store, which would otherwise replace what the
/// runtime config set at open.
fn apply_runtime_read_settings(plan: &mut ReadPlanPolicy, config: &mtxdb::RuntimeConfig) {
    plan.point_random_mmap = config.point_random_mmap;
    plan.point_random_mmap_index = config.ordered_index_point_random_mmap;
}

/// Map the `--read-plan` CLI mode to a store policy.
///
/// `clap`'s value parser restricts the argument to these modes, so the
/// fallthrough is unreachable in practice.
fn read_plan_from_mode(mode: &str) -> ReadPlanPolicy {
    match mode {
        "plain" => ReadPlanPolicy::disabled(),
        "prefetch" => ReadPlanPolicy::prefetch(),
        _ => unreachable!("clap validates read-plan mode"),
    }
}

/// With `--coalesce`, also take the databases in the current directory: the
/// directory itself if it is a database root, else each immediate child that
/// is one. Already-listed directories are not repeated.
fn add_cwd_databases(dirs: &mut Vec<PathBuf>) {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let mut found = Vec::new();
    if mtxdb::is_database_root(&cwd).unwrap_or(false) {
        found.push(cwd);
    } else if let Ok(entries) = std::fs::read_dir(&cwd) {
        for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if path.is_dir() && mtxdb::is_database_root(&path).unwrap_or(false) {
                found.push(path);
            }
        }
        found.sort();
    }
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let mut seen: Vec<PathBuf> = dirs.iter().map(|d| canon(d)).collect();
    for path in found {
        let c = canon(&path);
        if !seen.contains(&c) {
            seen.push(c);
            dirs.push(path);
        }
    }
}

/// Let one `-d`/`--dir` take several directories, so a shell glob works
/// (`mtxdb -d pid-* collections`). Clap gives `-d` exactly one value, so each
/// bare token after it that `is_dir` is rewritten to its own `-d`, and one
/// that `is_file` is dropped (so `-d *` skips stray files). The first other
/// token (a subcommand, a key, a flag) ends the run. A token naming a
/// subcommand is never rewritten, and once one is seen files are kept, since
/// they are that subcommand's operands (`mtxdb import -d db events.json`).
fn expand_dir_args(
    args: impl IntoIterator<Item = std::ffi::OsString>,
    is_dir: impl Fn(&Path) -> bool,
    is_file: impl Fn(&Path) -> bool,
    is_subcommand: impl Fn(&str) -> bool,
) -> Vec<std::ffi::OsString> {
    // `Value` is the token right after `-d` (clap's own value); `Run` is any
    // further bare tokens, which are the extra directories.
    #[derive(PartialEq)]
    enum State {
        Idle,
        Value,
        Run,
    }

    let mut out = Vec::new();
    let mut state = State::Idle;
    let mut seen_subcommand = false;
    for arg in args {
        let text = arg.to_string_lossy();
        let bare_run = state == State::Run && !text.starts_with('-');
        if text == "-d" || text == "--dir" {
            state = State::Value;
            out.push(arg);
        } else if state == State::Value {
            state = State::Run;
            out.push(arg);
        } else if state != State::Value && !text.starts_with('-') && is_subcommand(&text) {
            seen_subcommand = true;
            state = State::Idle;
            out.push(arg);
        } else if bare_run && is_dir(Path::new(&arg)) {
            out.push("-d".into());
            out.push(arg);
        } else if bare_run && !seen_subcommand && is_file(Path::new(&arg)) {
            // A stray file in a glob; not a database, not a subcommand.
        } else {
            state = State::Idle;
            out.push(arg);
        }
    }
    out
}

/// Map `mtxdb room <subcommand>` to its flat `Commands` value.
fn parse_room_command(m: &clap::ArgMatches) -> Commands {
    let Some((name, sub)) = m.subcommand() else {
        unreachable!("clap requires a room subcommand");
    };
    match name {
        "timeline" => Commands::Timeline {
            room: sub.get_one::<String>("room").unwrap().clone(),
            limit: *sub
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
            reverse: sub.get_flag("reverse"),
        },
        "flags" => Commands::Flags {
            room: sub.get_one::<String>("room").unwrap().clone(),
            flag: sub.get_one::<String>("flag").cloned(),
            not: sub.get_flag("not"),
            count: sub.get_flag("count"),
            limit: *sub
                .get_one::<usize>("limit")
                .expect("clap supplies a default limit"),
            rebuild: sub.get_flag("rebuild"),
        },
        "members" => Commands::Members {
            room: sub.get_one::<String>("room").unwrap().clone(),
            json: sub.get_flag("json"),
        },
        "check" => Commands::RoomCheck {
            room: sub.get_one::<String>("room").unwrap().clone(),
            auth: sub.get_flag("auth"),
            prevs: sub.get_flag("prevs"),
            state_members: sub.get_flag("state-members"),
            status: sub.get_one::<String>("status").cloned(),
            history: sub.get_flag("history"),
            limit: *sub.get_one::<usize>("limit").unwrap(),
        },
        "gaps" => {
            let room = sub.get_one::<String>("room").unwrap().clone();
            let action = match sub.subcommand() {
                Some(("ingest", action)) => GapAction::Ingest {
                    room,
                    run: action.get_one::<String>("run").unwrap().clone(),
                    report: PathBuf::from(action.get_one::<String>("report").unwrap()),
                },
                Some(("waive", action)) => GapAction::Waive {
                    room,
                    gap: action.get_one::<String>("gap").unwrap().clone(),
                    by: action.get_one::<String>("by").unwrap().clone(),
                    reason: action.get_one::<String>("reason").unwrap().clone(),
                },
                Some(("anchor", action)) => GapAction::Anchor {
                    room,
                    gap: action.get_one::<String>("gap").unwrap().clone(),
                    event: action.get_one::<String>("event").unwrap().clone(),
                    after: action.get_one::<String>("boundary").unwrap() == "after",
                    source: action.get_one::<String>("source").unwrap().clone(),
                },
                _ => unreachable!("clap requires a gaps action"),
            };
            Commands::Gaps { action }
        }
        _ => unreachable!("clap validates the room subcommand"),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the exhaustive clap-to-command mapping is clearest in one match"
)]
fn parse_cli() -> Cli {
    let command = build_cli();
    let matches = command.clone().get_matches_from(expand_dir_args(
        std::env::args_os(),
        Path::is_dir,
        Path::is_file,
        |name| command.find_subcommand(name).is_some(),
    ));

    if matches.get_flag("version") {
        println!("mtxdb {}", build_cli().get_version().unwrap());
        std::process::exit(0);
    }

    let mut dirs: Vec<PathBuf> = matches
        .get_many::<String>("dir")
        .map(|vals| vals.map(PathBuf::from).collect())
        .unwrap_or_default();
    // `--coalesce`, `--read-plan` and `--shard-type` are scoped to the commands
    // that honor them, so read them from the selected (innermost) subcommand
    // and fall back to the defaults where a command does not define them.
    let mut leaf = &matches;
    while let Some((_, sub)) = leaf.subcommand() {
        leaf = sub;
    }
    let coalesce = leaf
        .try_get_one::<bool>("coalesce")
        .ok()
        .flatten()
        .copied()
        .unwrap_or(false);
    // Precedence: explicit `-d`, then a database root in the current
    // directory, then `MTXDB_DIR`. The env var is only a fallback default.
    if matches.value_source("dir") == Some(clap::parser::ValueSource::EnvVariable) {
        if let Ok(cwd) = std::env::current_dir() {
            if mtxdb::is_database_root(&cwd).unwrap_or(false) {
                dirs = vec![cwd];
            }
        }
    }
    if coalesce {
        add_cwd_databases(&mut dirs);
    }
    let _ = DEBUG_ENABLED.set(matches.get_flag("debug"));
    let read_plan = read_plan_from_mode(
        leaf.try_get_one::<String>("read_plan")
            .ok()
            .flatten()
            .map_or("plain", String::as_str),
    );
    let shard_type = shard_type_from_arg(
        leaf.try_get_one::<String>("shard_type")
            .ok()
            .flatten()
            .map_or("events", String::as_str),
    );

    let command = match matches.subcommand() {
        Some(("put", m)) => Commands::Put {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            id: m.get_one::<String>("id").unwrap().clone(),
            data: m.get_one::<String>("data").unwrap().clone(),
        },
        Some(("get", m)) => Commands::Get {
            collection: m.get_one::<String>("collection").cloned(),
            id: m.get_one::<String>("id").unwrap().clone(),
            raw: m.get_flag("raw"),
            verbose: m.get_flag("verbose"),
            header: m.get_flag("header"),
            decode: m.get_one::<String>("decode").cloned(),
        },
        Some(("collections", m)) => Commands::Collections {
            all: m.get_flag("all"),
            layout: m.get_flag("layout"),
            canonical: m.get_flag("canonical"),
            per_db: m.get_flag("per_db"),
            sort: m.get_one::<String>("sort").cloned(),
            limit: *m
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
        },
        Some(("stats", m)) => Commands::Stats {
            json: m.get_flag("json"),
        },
        Some(("memory", m)) => Commands::Memory {
            evict_pages: m.get_flag("evict-pages"),
        },
        Some(("meta", m)) => Commands::Meta {
            target: m.get_one::<String>("target").unwrap().clone(),
            json: m.get_flag("json"),
            limit: *m
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
            offset: *m
                .get_one::<i64>("offset")
                .expect("clap supplies a default offset"),
            decode: m.get_one::<String>("decode").cloned(),
        },
        Some(("info", m)) => Commands::Info {
            selector: m.get_one::<String>("selector").cloned(),
            stats: m.get_flag("stats"),
            tombstones: m.get_flag("tombstones"),
            keys: m.get_flag("keys"),
            detail: m.get_flag("detail"),
            json: m.get_flag("json"),
        },
        Some(("scan", m)) => Commands::Scan {
            selector: m.get_one::<String>("selector").unwrap().clone(),
            verbose: m.get_flag("verbose"),
            header: m.get_flag("header"),
            decode: m.get_one::<String>("decode").cloned(),
            id: m.get_one::<String>("id").cloned(),
            collection: None,
            raw: m.get_flag("raw"),
            sort: m.get_one::<String>("sort").cloned(),
            reverse: m.get_flag("reverse"),
            live: m.get_flag("live"),
            limit: *m
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
        },
        Some(("search", m)) => Commands::Search {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            template: m.get_one::<String>("template").map(PathBuf::from),
            term: m.get_one::<String>("term").unwrap().clone(),
            format: m
                .get_one::<String>("format")
                .expect("clap supplies a default format")
                .clone(),
            limit: *m
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
        },
        Some(("import", m)) => Commands::Import {
            paths: m
                .get_many::<String>("path")
                .unwrap()
                .map(PathBuf::from)
                .collect(),
            collection: m.get_one::<String>("collection").cloned(),
            template: m.get_one::<String>("template").map(PathBuf::from),
            no_state_hamts: m.get_flag("no-state-hamts"),
            require_complete_prevs: m.get_flag("require-complete-prevs"),
            allow_waived_gaps: m.get_flag("allow-waived-gaps"),
        },
        Some(("export", m)) => Commands::Export {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            format: m
                .get_one::<String>("format")
                .cloned()
                .unwrap_or_else(|| "jsonl".to_owned()),
            metadata: m.get_flag("metadata"),
        },
        Some(("packs", m)) => Commands::Packs {
            action: match m.subcommand() {
                Some(("list", sub)) => PacksAction::List {
                    all: sub.get_flag("all"),
                    layout: sub.get_flag("layout"),
                    sort: sub.get_one::<String>("sort").cloned(),
                },
                Some(("scan", sub)) => PacksAction::Scan {
                    selector: sub.get_one::<String>("selector").unwrap().clone(),
                    verbose: sub.get_flag("verbose"),
                    header: sub.get_flag("header"),
                    decode: sub.get_one::<String>("decode").cloned(),
                    limit: *sub.get_one::<i64>("limit").unwrap(),
                    id: sub.get_one::<String>("id").cloned(),
                    collection: sub.get_one::<String>("collection").cloned(),
                    raw: sub.get_flag("raw"),
                    sort: sub.get_one::<String>("sort").cloned(),
                    reverse: sub.get_flag("reverse"),
                },
                Some(("dump", sub)) => PacksAction::Dump {
                    pack: sub.get_one::<String>("pack").unwrap().clone(),
                    collection: sub.get_one::<String>("collection").cloned(),
                    out: sub.get_one::<String>("out").map(PathBuf::from),
                },
                Some(("extract", sub)) => PacksAction::Extract {
                    pack: sub.get_one::<String>("pack").unwrap().clone(),
                    collection: sub.get_one::<String>("collection").unwrap().clone(),
                    out: PathBuf::from(sub.get_one::<String>("out").unwrap()),
                },
                _ => unreachable!("subcommand_required enforces a packs action"),
            },
        },
        Some(("repack", m)) => Commands::Repack {
            collection: m.get_one::<String>("collection").cloned(),
            packs: m
                .get_many::<String>("pack")
                .into_iter()
                .flatten()
                .cloned()
                .collect(),
            all: m.get_flag("all"),
            out: m.get_one::<String>("out").map(PathBuf::from),
            yes: m.get_flag("yes"),
        },
        Some(("drop", m)) => Commands::Drop {
            collections: m
                .get_many::<String>("collection")
                .unwrap()
                .cloned()
                .collect(),
            yes: m.get_flag("yes"),
        },
        Some(("room", m)) => parse_room_command(m),
        Some(("remove", m)) => Commands::Remove {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            ids: m.get_many::<String>("id").unwrap().cloned().collect(),
            yes: m.get_flag("yes"),
        },
        Some(("completions", m)) => Commands::Completions {
            shell: m.get_one::<String>("shell").unwrap().clone(),
        },
        Some(("sync", m)) => Commands::Sync {
            all: m.get_flag("all"),
        },
        Some(("reclaim-wal", _)) => Commands::ReclaimWal,
        Some(("init", _)) => Commands::Init,
        _ => {
            build_cli().print_help().unwrap();
            std::process::exit(0);
        }
    };

    Cli {
        dirs,
        shard_type,
        coalesce,
        read_plan,
        command,
    }
}

// `main` deliberately does not return `Result`: std's `Termination` prints the
// error via `Debug`, and anyhow's `Debug` includes a backtrace whenever
// `RUST_BACKTRACE` is set in the environment — for release builds too. A
// missing database is an expected user error, so we render the chain with
// `Display` instead and let the profile/env decide nothing.
/// Parse command-line arguments and execute the selected mtxdb command.
pub fn run() -> std::process::ExitCode {
    #[cfg(feature = "logging")]
    let _ = mtxdb::logging::init();

    let mut cli = parse_cli();
    if let Commands::Completions { shell } = &cli.command {
        #[cfg(feature = "completions")]
        {
            let shell = match shell.as_str() {
                "bash" => clap_complete::Shell::Bash,
                "elvish" => clap_complete::Shell::Elvish,
                "fish" => clap_complete::Shell::Fish,
                "powershell" => clap_complete::Shell::PowerShell,
                "zsh" => clap_complete::Shell::Zsh,
                _ => unreachable!("Clap validates the shell name"),
            };
            // Rebuild without the hidden `completions` subcommand —
            // `.hide(true)` only suppresses --help, not shell completions.
            let base = build_cli();
            let mut command = clap::Command::new("mtxdb");
            for arg in base.get_arguments() {
                command = command.arg(arg.clone());
            }
            for sub in base.get_subcommands() {
                if sub.get_name() != "completions" {
                    command = command.subcommand(sub.clone());
                }
            }
            clap_complete::generate(shell, &mut command, "mtxdb", &mut std::io::stdout());
            return std::process::ExitCode::SUCCESS;
        }
        #[cfg(not(feature = "completions"))]
        {
            let _ = shell;
            eprintln!(
                "Error: this build was compiled without shell-completion support \
                 (enable the `completions` feature)"
            );
            return std::process::ExitCode::FAILURE;
        }
    }
    let runtime_config = match mtxdb::RuntimeConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("Error: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    apply_runtime_read_settings(&mut cli.read_plan, &runtime_config);
    let _ = RUNTIME_CONFIG.set(runtime_config);

    match cmd::run(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[path = "test_parse.rs"]
mod parse_tests;
