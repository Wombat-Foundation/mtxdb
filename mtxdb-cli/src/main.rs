//! CLI for the mtxdb content-addressed storage engine.

mod cmd;

use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};
use mtxdb_core::ShardType;

pub(crate) struct Cli {
    pub(crate) dir: Option<PathBuf>,
    pub(crate) shard_type: ShardType,
    pub(crate) command: Commands,
}

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
    },
    Collections {
        all: bool,
        layout: bool,
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
    },
    Scan {
        selector: String,
        verbose: bool,
        limit: i64,
        id: Option<String>,
        collection: Option<String>,
        raw: bool,
        sort: Option<String>,
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
        shards: Vec<String>,
        all: bool,
        root: Vec<String>,
        topo: bool,
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
}

fn build_cli() -> Command {
    global_args(Command::new("mtxdb"))
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
        .subcommand(sub_stats())
        .subcommand(sub_collections())
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
}

/// The two flags shared by every subcommand (`--dir`, `--shard-type`).
fn global_args(cmd: Command) -> Command {
    cmd.arg(
        Arg::new("dir")
            .short('d')
            .long("dir")
            .env("MTXDB_DIR")
            .value_name("DIR")
            .global(true)
            .help("Database root directory"),
    )
    .arg(
        Arg::new("shard_type")
            .short('t')
            .long("shard-type")
            .env("MTXDB_SHARD_TYPE")
            .value_name("TYPE")
            .default_value("event-dag")
            .value_parser(["state", "event-dag", "auth-chain"])
            .hide_possible_values(true)
            .global(true)
            .help("Independent shard pool to operate on"),
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
        .arg(limit_arg())
        .arg(sort_arg("slot, collection, nodes, shards, index, disk, packs, avoidable, segments, fragmentation"))
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
                    "A 32-hex-digit collection ID (`0x`-prefix optional), or a pack ID from \
                     `mtxdb shards` (1-16 hex digits, `0x`-prefix required) — e.g. `mtxdb scan \
                     0102030405060708090a0b0c0d0e0f10` or `mtxdb scan 0x1`",
                ),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(ArgAction::SetTrue)
                .help("Print each frame's JSON payload when available"),
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
        .arg(sort_arg("payload"))
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
                    "A 32-hex-digit collection ID (`0x`-prefix optional), a collection's \
                     slot index from `mtxdb collections`, or a pack ID from `mtxdb shards` \
                     (1-16 hex digits, `0x`-prefix required) — e.g. `mtxdb info \
                     0x0102030405060708090a0b0c0d0e0f10` or `mtxdb info 0x1`",
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
             namespace comes from `collection_id` unless --collection is supplied.",
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
                .help("Collection ID (hex, 32 chars). Auto-detected if omitted"),
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
                .help("Collection ID (hex, 32 chars)"),
        )
}

fn sub_repack() -> Command {
    Command::new("repack")
        .about("Trigger manual repack over a collection or pack closure")
        .arg(
            Arg::new("collection")
                .short('r')
                .long("collection")
                .conflicts_with("shard"),
        )
        .arg(
            Arg::new("shard")
                .short('s')
                .long("shard")
                .conflicts_with("collection")
                .num_args(1..)
                .value_name("PACK_ID")
                .help("Repack collections referencing packs (repeat -s for multiple pack IDs)"),
        )
        .arg(
            Arg::new("all")
                .long("all")
                .conflicts_with_all(["collection", "shard"])
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
                .help("Node ID (32 hex characters) or Matrix event ID"),
        )
        .arg(
            Arg::new("id_option")
                .short('i')
                .long("id")
                .conflicts_with("id")
                .help("Node ID (32 hex characters) or Matrix event ID"),
        )
        .arg(
            Arg::new("raw")
                .long("raw")
                .action(ArgAction::SetTrue)
                .conflicts_with("text")
                .help("Emit payload bytes verbatim instead of pretty-printing JSON"),
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

    let dir = matches.get_one::<String>("dir").map(PathBuf::from);
    let shard_type = match matches
        .get_one::<String>("shard_type")
        .map(String::as_str)
        .expect("clap supplies the default shard type")
    {
        "state" => ShardType::State,
        "event-dag" => ShardType::EventDag,
        "auth-chain" => ShardType::AuthChain,
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
        },
        Some(("collections", m)) => Commands::Collections {
            all: m.get_flag("all"),
            layout: m.get_flag("layout"),
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
        },
        Some(("scan", m)) => Commands::Scan {
            selector: m.get_one::<String>("selector").unwrap().clone(),
            verbose: m.get_flag("verbose"),
            id: m.get_one::<String>("id").cloned(),
            collection: m.get_one::<String>("collection").cloned(),
            raw: m.get_flag("raw"),
            sort: m.get_one::<String>("sort").cloned(),
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
            shards: m
                .get_many::<String>("shard")
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
        _ => {
            build_cli().print_help().unwrap();
            std::process::exit(0);
        }
    };

    Cli {
        dir,
        shard_type,
        command,
    }
}

fn main() -> anyhow::Result<()> {
    let cli = parse_cli();
    if let Commands::Completions { shell } = &cli.command {
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
        return Ok(());
    }
    cmd::run(&cli)
}
