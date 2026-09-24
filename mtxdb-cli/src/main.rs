//! CLI for the mtxdb content-addressed storage engine.

mod cmd;

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Arg, ArgAction, Command};
use mtxdb::ShardType;

#[derive(Clone)]
pub(crate) struct Cli {
    pub(crate) dirs: Vec<PathBuf>,
    pub(crate) shard_type: Option<ShardType>,
    pub(crate) coalesce: bool,
    pub(crate) command: Commands,
}

impl Cli {
    /// Return the single selected shard type, or bail if `-t all` was used.
    pub(crate) fn require_shard_type(&self) -> anyhow::Result<ShardType> {
        self.shard_type.context(
            "this command requires a specific shard type (-t state, -t events, or -t edges)",
        )
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
        sort: Option<String>,
        limit: i64,
    },
    Shards {
        all: bool,
        layout: bool,
        sort: Option<String>,
    },
    Stats {
        json: bool,
    },
    Info {
        collection: String,
        stats: bool,
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
    },
    Import {
        paths: Vec<PathBuf>,
        collection: Option<String>,
        template: Option<PathBuf>,
    },
    Export {
        collection: String,
    },
    Repack {
        collection: Option<String>,
        packs: Vec<String>,
        all: bool,
        root: Vec<String>,
        topo: bool,
        out: Option<PathBuf>,
        yes: bool,
    },
    Delete {
        collections: Vec<String>,
        yes: bool,
    },
    Completions {
        shell: String,
    },
    Sync {
        all: bool,
    },
    Init,
    /// Internal: write a test record and sync (for subprocess tests).
    SubprocessWriter {
        path: String,
    },
    /// Internal: write a test record without sync (for subprocess tests).
    SubprocessWriterUnsynced {
        path: String,
    },
    /// Internal: seed test data and sync (for subprocess tests).
    SubprocessWriterSeed {
        path: String,
    },
    /// Internal: append and sync test data (for subprocess tests).
    SubprocessWriterAppend {
        path: String,
    },
    /// Internal: read with refresh (for subprocess tests).
    SubprocessReader {
        path: String,
    },
}

fn build_cli() -> Command {
    global_args(Command::new("mtxdb").subcommand_precedence_over_arg(true))
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
        .arg(
            Arg::new("version_upper")
                .short('V')
                .action(ArgAction::SetTrue)
                .hide(true),
        )
        .about("CLI for the mtxdb content-addressed storage engine")
        .subcommand(sub_shards())
        .subcommand(sub_collections())
        .subcommand(sub_stats())
        .subcommand(sub_sync())
        .subcommand(sub_completions())
        .subcommand(sub_import())
        .subcommand(sub_export())
        .subcommand(sub_repack())
        .subcommand(sub_scan())
        .subcommand(sub_info())
        .subcommand(sub_delete())
        .subcommand(sub_put())
        .subcommand(sub_get())
        .subcommand(sub_init())
        .subcommand(sub_subprocess_writer())
        .subcommand(sub_subprocess_writer_unsynced())
        .subcommand(sub_subprocess_writer_seed())
        .subcommand(sub_subprocess_writer_append())
        .subcommand(sub_subprocess_reader())
}

/// The two flags shared by every subcommand (`--dir`, `--shard-type`).
fn global_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("dir")
            .short('d')
            .long("dir")
            .env("MTXDB_DIR")
            .value_name("DIR")
            .action(ArgAction::Append)
            .num_args(1..)
            .global(true)
            .help("Database root directory (or multiple directories)"),
    )
    .arg(
        Arg::new("coalesce")
            .short('c')
            .long("coalesce")
            .action(ArgAction::SetTrue)
            .global(true)
            .help("Aggregate supported read-only reports across multiple database roots (shards, collections, stats, get) or coalescing repack (--out)"),
    )
    .arg(
        Arg::new("shard_type")
            .short('t')
            .long("shard-type")
            .env("MTXDB_SHARD_TYPE")
            .value_name("TYPE")
            .default_value("events")
            .value_parser(["state", "events", "event-dag", "edges", "all"])
            .hide_possible_values(true)
            .global(true)
            .help("Independent shard pool to operate on (use 'all' to target every pool)"),
    )
}

fn sub_shards() -> Command {
    Command::new("shards")
        .about("List open packs with size and IO/sync stats")
        .arg(
            Arg::new("all")
                .short('a')
                .long("all")
                .action(ArgAction::SetTrue)
                .help("List packs in every independent pool"),
        )
        .arg(layout_arg())
        .arg(sort_arg(
            "pack, bytes, nodes, collections, syncs, segments, interleaving",
        ))
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

fn sub_collections() -> Command {
    Command::new("collections")
        .about("List logical collections in the selected shard pool")
        .arg(
            Arg::new("all")
                .short('a')
                .long("all")
                .action(ArgAction::SetTrue)
                .help("List collections in every independent pool"),
        )
        .arg(layout_arg())
        .arg(
            Arg::new("canonical")
                .long("canonical")
                .action(ArgAction::SetTrue)
                .help("Show each collection's canonical ID"),
        )
        .arg(limit_arg())
        .arg(sort_arg("collection, nodes, load, shards, index, disk, packs, avoidable, segments, fragmentation"))
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
    Command::new("init").about(
        "Create a new mtxdb database root (db.meta + a directory per shard pool). \
         The only command that creates a store -- every other command errors \
         if it doesn't already exist.",
    )
}

fn sub_subprocess_writer() -> Command {
    Command::new("subprocess-writer")
        .about("Internal: write a test record and sync")
        .hide(true)
        .arg(Arg::new("path").required(true).index(1))
}

fn sub_subprocess_writer_unsynced() -> Command {
    Command::new("subprocess-writer-unsynced")
        .about("Internal: write a test record without sync")
        .hide(true)
        .arg(Arg::new("path").required(true).index(1))
}

fn sub_subprocess_writer_seed() -> Command {
    Command::new("subprocess-writer-seed")
        .about("Internal: seed test data and sync")
        .hide(true)
        .arg(Arg::new("path").required(true).index(1))
}

fn sub_subprocess_writer_append() -> Command {
    Command::new("subprocess-writer-append")
        .about("Internal: append and sync test data")
        .hide(true)
        .arg(Arg::new("path").required(true).index(1))
}

fn sub_subprocess_reader() -> Command {
    Command::new("subprocess-reader")
        .about("Internal: read with refresh")
        .hide(true)
        .arg(Arg::new("path").required(true).index(1))
}

fn sub_sync() -> Command {
    Command::new("sync")
        .about("Bootstrap or refresh persisted shard stats")
        .arg(
            Arg::new("all")
                .short('a')
                .long("all")
                .action(ArgAction::SetTrue)
                .help("Sync every independent pool, not just the selected `-t` one"),
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
    Command::new("scan")
        .about("Scan a packfile or collection and print physical records")
        .arg(
            Arg::new("selector")
                .required(true)
                .value_name("PACK_ID|COLLECTION")
                .help(
                    "A `0x`-prefixed collection ID (32 hex digits after `0x`), or a pack ID \
                     from `mtxdb shards` (also `0x`-prefixed, 1-16 hex digits) — e.g. `mtxdb \
                     scan 0x0102030405060708090a0b0c0d0e0f10` or `mtxdb scan 0x1`",
                ),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
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
                .help("Decode and display payload format (e.g. json, hamt, state, raw, or auto)"),
        )
        .arg(
            Arg::new("id")
                .long("id")
                .value_name("NODE_ID")
                .help("Restrict the physical scan to one node ID"),
        )
        .arg(
            Arg::new("collection")
                .long("collection")
                .value_name("COLLECTION")
                .help("With a pack selector, restrict the scan to this collection"),
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
}

fn sub_info() -> Command {
    Command::new("info")
        .about("Show storage info for a collection or a pack")
        .arg(
            Arg::new("collection")
                .required(true)
                .value_name("PACK_ID|COLLECTION")
                .help(
                    "A `0x`-prefixed collection ID (32 hex digits after `0x`) or a pack ID \
                     from `mtxdb shards` (also `0x`-prefixed, 1-16 hex digits) — e.g. `mtxdb info \
                     0x0102030405060708090a0b0c0d0e0f10` or `mtxdb info 0x1`",
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
}

fn sub_import() -> Command {
    Command::new("import")
        .about("Import Matrix federation events from JSON or JSONL")
        .long_about(
            "Import Matrix federation events from a JSON document or JSONL event stream. A JSON \
             document must contain a `pdus` array, an `auth_chain` array, or both; a `.jsonl` \
             file contains one event per line. Each imported event needs an `event_id`. The \
             collection identity comes from `collection_id` unless --collection is supplied.",
        )
        .arg(
            Arg::new("path")
                .required(true)
                .num_args(1..)
                .value_name("FILE")
                .help("Matrix federation JSON documents or JSONL event streams"),
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
}

fn sub_export() -> Command {
    Command::new("export")
        .about("Export a collection's records as JSONL to stdout")
        .long_about(
            "Export a collection's stored records as one JSON value per line on stdout. Redirect the \
             output to make an input accepted by `mtxdb import`.",
        )
        .arg(
            Arg::new("collection")
                .required(true)
                .value_name("COLLECTION")
                .help("Collection ID (0x-prefixed, 32 hex digits)"),
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
            Arg::new("all")
                .long("all")
                .conflicts_with_all(["collection", "pack"])
                .action(ArgAction::SetTrue)
                .help("Repack every collection in every active pack"),
        )
        .arg(Arg::new("root").short('o').long("root").num_args(1..))
        .arg(
            Arg::new("topo")
                .long("topo")
                .action(ArgAction::SetTrue)
                .help("Repack in topological order (requires edge-capable data format)"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .value_name("DIR")
                .help("Destination directory for coalescing repack into a new canonical database"),
        )
        .arg(
            Arg::new("yes")
                .short('y')
                .long("yes")
                .action(ArgAction::SetTrue)
                .help("Skip interactive confirmation"),
        )
}

fn sub_delete() -> Command {
    Command::new("delete")
        .about("Delete all data for one or more collections")
        .arg(
            Arg::new("collection")
                .required(true)
                .num_args(1..)
                .value_name("COLLECTION")
                .help("Namespace IDs to delete"),
        )
        .arg(
            Arg::new("yes")
                .long("yes")
                .action(ArgAction::SetTrue)
                .help("Skip confirmation prompt"),
        )
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
                .required_unless_present("id_option")
                .help("Node ID (0x-prefixed, 32 hex digits) or Matrix event ID"),
        )
        .arg(
            Arg::new("id_option")
                .short('i')
                .long("id")
                .conflicts_with("id")
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
                .help("Decode and display payload format (e.g. json, hamt, state, raw, or auto)"),
        )
}

#[allow(
    clippy::too_many_lines,
    reason = "the exhaustive clap-to-command mapping is clearest in one match"
)]
fn parse_cli() -> Cli {
    let matches = build_cli().get_matches();

    if matches.get_flag("version") || matches.get_flag("version_upper") {
        println!("mtxdb {}", build_cli().get_version().unwrap());
        std::process::exit(0);
    }

    let dirs: Vec<PathBuf> = matches
        .get_many::<String>("dir")
        .map(|vals| vals.map(PathBuf::from).collect())
        .unwrap_or_default();
    let coalesce = matches.get_flag("coalesce");
    let shard_type = match matches
        .get_one::<String>("shard_type")
        .map(String::as_str)
        .expect("clap supplies the default shard type")
    {
        "state" => Some(ShardType::State),
        "events" | "event-dag" => Some(ShardType::EventDag),
        "edges" => Some(ShardType::Edges),
        "all" => None,
        _ => unreachable!("clap validates shard type"),
    };

    let command = match matches.subcommand() {
        Some(("put", m)) => Commands::Put {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            id: m.get_one::<String>("id").unwrap().clone(),
            data: m.get_one::<String>("data").unwrap().clone(),
        },
        Some(("get", m)) => Commands::Get {
            collection: m.get_one::<String>("collection").cloned(),
            id: m
                .get_one::<String>("id")
                .or_else(|| m.get_one::<String>("id_option"))
                .expect("clap requires either positional ID or --id")
                .clone(),
            raw: m.get_flag("raw"),
            verbose: m.get_flag("verbose"),
            header: m.get_flag("header"),
            decode: m.get_one::<String>("decode").cloned(),
        },
        Some(("collections", m)) => Commands::Collections {
            all: m.get_flag("all"),
            layout: m.get_flag("layout"),
            canonical: m.get_flag("canonical"),
            sort: m.get_one::<String>("sort").cloned(),
            limit: *m
                .get_one::<i64>("limit")
                .expect("clap supplies a default limit"),
        },
        Some(("shards", m)) => Commands::Shards {
            all: m.get_flag("all"),
            layout: m.get_flag("layout"),
            sort: m.get_one::<String>("sort").cloned(),
        },
        Some(("stats", m)) => Commands::Stats {
            json: m.get_flag("json"),
        },
        Some(("info", m)) => Commands::Info {
            collection: m.get_one::<String>("collection").unwrap().clone(),
            stats: m.get_flag("stats"),
        },
        Some(("scan", m)) => Commands::Scan {
            selector: m.get_one::<String>("selector").unwrap().clone(),
            verbose: m.get_flag("verbose"),
            header: m.get_flag("header"),
            decode: m.get_one::<String>("decode").cloned(),
            id: m.get_one::<String>("id").cloned(),
            collection: m.get_one::<String>("collection").cloned(),
            raw: m.get_flag("raw"),
            sort: m.get_one::<String>("sort").cloned(),
            reverse: m.get_flag("reverse"),
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
        },
        Some(("export", m)) => Commands::Export {
            collection: m.get_one::<String>("collection").unwrap().clone(),
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
            root: m
                .get_many::<String>("root")
                .into_iter()
                .flatten()
                .cloned()
                .collect(),
            topo: m.get_flag("topo"),
            out: m.get_one::<String>("out").map(PathBuf::from),
            yes: m.get_flag("yes"),
        },
        Some(("delete", m)) => Commands::Delete {
            collections: m
                .get_many::<String>("collection")
                .unwrap()
                .cloned()
                .collect(),
            yes: m.get_flag("yes"),
        },
        Some(("completions", m)) => Commands::Completions {
            shell: m.get_one::<String>("shell").unwrap().clone(),
        },
        Some(("sync", m)) => Commands::Sync {
            all: m.get_flag("all"),
        },
        Some(("init", _)) => Commands::Init,
        Some(("subprocess-writer", m)) => Commands::SubprocessWriter {
            path: m.get_one::<String>("path").unwrap().clone(),
        },
        Some(("subprocess-writer-unsynced", m)) => Commands::SubprocessWriterUnsynced {
            path: m.get_one::<String>("path").unwrap().clone(),
        },
        Some(("subprocess-writer-seed", m)) => Commands::SubprocessWriterSeed {
            path: m.get_one::<String>("path").unwrap().clone(),
        },
        Some(("subprocess-writer-append", m)) => Commands::SubprocessWriterAppend {
            path: m.get_one::<String>("path").unwrap().clone(),
        },
        Some(("subprocess-reader", m)) => Commands::SubprocessReader {
            path: m.get_one::<String>("path").unwrap().clone(),
        },
        _ => {
            build_cli().print_help().unwrap();
            std::process::exit(0);
        }
    };

    Cli {
        dirs,
        shard_type,
        coalesce,
        command,
    }
}

// `main` deliberately does not return `Result`: std's `Termination` prints the
// error via `Debug`, and anyhow's `Debug` includes a backtrace whenever
// `RUST_BACKTRACE` is set in the environment — for release builds too. A
// missing database is an expected user error, so we render the chain with
// `Display` instead and let the profile/env decide nothing.
fn main() -> std::process::ExitCode {
    let cli = parse_cli();
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
    match cmd::run(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn test_dir_parsing() {
        let m = build_cli()
            .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "dir2"])
            .unwrap();
        let dirs: Vec<_> = m
            .get_many::<String>("dir")
            .unwrap()
            .map(String::as_str)
            .collect();
        assert_eq!(dirs, vec!["dir1", "dir2"]);

        let m = build_cli()
            .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "-d", "dir2"])
            .unwrap();
        let dirs: Vec<_> = m
            .get_many::<String>("dir")
            .unwrap()
            .map(String::as_str)
            .collect();
        assert_eq!(dirs, vec!["dir1", "dir2"]);

        let m = build_cli()
            .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "dir2", "-a"])
            .unwrap();
        let dirs: Vec<_> = m
            .get_many::<String>("dir")
            .unwrap()
            .map(String::as_str)
            .collect();
        assert_eq!(dirs, vec!["dir1", "dir2"]);

        let m = build_cli()
            .try_get_matches_from([
                "mtxdb",
                "scan",
                "0x0102030405060708090a0b0c0d0e0f10",
                "-d",
                "dir1",
                "dir2",
            ])
            .unwrap();
        let dirs: Vec<_> = m
            .get_many::<String>("dir")
            .unwrap()
            .map(String::as_str)
            .collect();
        assert_eq!(dirs, vec!["dir1", "dir2"]);
        assert_eq!(m.subcommand_name(), Some("scan"));

        let m = build_cli()
            .try_get_matches_from(["mtxdb", "shards", "-d", "dir1", "dir2", "-c"])
            .unwrap();
        assert!(m.get_flag("coalesce"));

        let m = build_cli()
            .try_get_matches_from(["mtxdb", "--coalesce", "shards", "-d", "dir1", "dir2"])
            .unwrap();
        assert!(m.get_flag("coalesce"));

        let m = build_cli()
            .try_get_matches_from([
                "mtxdb", "repack", "-d", "dir1", "dir2", "-c", "--out", "target", "-y",
            ])
            .unwrap();
        assert!(m.get_flag("coalesce"));
        let sub = m.subcommand_matches("repack").unwrap();
        assert_eq!(
            sub.get_one::<String>("out").map(String::as_str),
            Some("target")
        );
        assert!(sub.get_flag("yes"));
    }
}
