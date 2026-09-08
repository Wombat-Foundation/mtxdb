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
    },
    Collections {
        all: bool,
    },
    Shards {
        all: bool,
    },
    Info {
        collection: String,
    },
    Scan {
        shard: String,
    },
    Import {
        paths: Vec<PathBuf>,
        collection: Option<String>,
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
    Sync,
}

fn build_cli() -> Command {
    Command::new("mtxdb")
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
        .arg(
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
        .subcommand(
            Command::new("shards")
                .about("List open shard slots with size, rotation, and IO/sync stats")
                .arg(
                    Arg::new("all")
                        .short('a')
                        .long("all")
                        .action(ArgAction::SetTrue)
                        .help("List shards in every independent pool"),
                ),
        )
        .subcommand(
            Command::new("collections")
                .about("List logical collections in the selected shard pool")
                .arg(
                    Arg::new("all")
                        .short('a')
                        .long("all")
                        .action(ArgAction::SetTrue)
                        .help("List collections in every independent pool"),
                ),
        )
        .subcommand(Command::new("sync").about("Bootstrap or refresh persisted shard stats"))
        .subcommand(
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
                ])),
        )
        .subcommand(sub_import())
        .subcommand(sub_export())
        .subcommand(sub_repack())
        .subcommand(
            Command::new("scan")
                .about("Scan a packfile and print records")
                .arg(
                    Arg::new("shard")
                        .required(true)
                        .value_name("SLOT | EPOCH")
                        .help("Decimal shard slot or 0x-prefixed shard rotation from `shards`"),
                ),
        )
        .subcommand(
            Command::new("info")
                .about("Show storage info for a collection")
                .arg(
                    Arg::new("collection")
                        .required(true)
                        .value_name("COLLECTION"),
                ),
        )
        .subcommand(sub_delete())
        .subcommand(sub_put())
        .subcommand(sub_get())
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
        .about("Trigger manual repack over closure of collection or shard closure")
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
                .value_name("SLOT | START-END")
                .help(
                    "Repack collections referencing shard slots (for example: -s 0 1 2 or -s 0-3)",
                ),
        )
        .arg(
            Arg::new("all")
                .long("all")
                .conflicts_with_all(["collection", "shard"])
                .action(ArgAction::SetTrue)
                .help("Repack every collection in every active shard"),
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
                .short('i')
                .long("id")
                .required(true)
                .help("Node ID (32 hex characters) or Matrix event ID"),
        )
}

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
            id: m.get_one::<String>("id").unwrap().clone(),
        },
        Some(("collections", m)) => Commands::Collections {
            all: m.get_flag("all"),
        },
        Some(("shards", m)) => Commands::Shards {
            all: m.get_flag("all"),
        },
        Some(("info", m)) => Commands::Info {
            collection: m.get_one::<String>("collection").unwrap().clone(),
        },
        Some(("scan", m)) => Commands::Scan {
            shard: m.get_one::<String>("shard").unwrap().clone(),
        },
        Some(("import", m)) => Commands::Import {
            paths: m
                .get_many::<String>("path")
                .unwrap()
                .map(PathBuf::from)
                .collect(),
            collection: m.get_one::<String>("collection").cloned(),
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
        Some(("sync", _)) => Commands::Sync,
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
