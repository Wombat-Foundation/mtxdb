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
        room: String,
        id: String,
        data: String,
    },
    Get {
        room: Option<String>,
        id: String,
    },
    Rooms,
    Shards {
        all: bool,
    },
    Info {
        room: String,
    },
    Scan {
        shard: String,
    },
    Import {
        paths: Vec<PathBuf>,
        room: Option<String>,
    },
    Export {
        room: String,
    },
    Repack {
        room: Option<String>,
        shards: Vec<String>,
        all: bool,
        root: Vec<String>,
        topo: bool,
    },
    Delete {
        rooms: Vec<String>,
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
        .subcommand(Command::new("rooms").about("List rooms in the store"))
        .subcommand(Command::new("sync").about("Bootstrap or refresh persisted shard/room stats"))
        .subcommand(
            Command::new("completions")
                .about("Print shell completion script")
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
                .about("Show storage info for a room")
                .arg(Arg::new("room").required(true).value_name("ROOM")),
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
             file contains one event per line. Each imported event needs an `event_id`. The room \
             comes from `room_id` unless --room is supplied.",
        )
        .arg(
            Arg::new("path")
                .required(true)
                .num_args(1..)
                .value_name("FILE")
                .help("Matrix federation JSON documents or JSONL event streams"),
        )
        .arg(
            Arg::new("room")
                .short('r')
                .long("room")
                .help("Room ID (hex, 32 chars). Auto-detected if omitted"),
        )
}

fn sub_export() -> Command {
    Command::new("export")
        .about("Export a room's records as JSONL to stdout")
        .long_about(
            "Export a room's stored records as one JSON value per line on stdout. Redirect the \
             output to make an input accepted by `mtxdb import`.",
        )
        .arg(
            Arg::new("room")
                .required(true)
                .value_name("ROOM")
                .help("Room ID (hex, 32 chars)"),
        )
}

fn sub_repack() -> Command {
    Command::new("repack")
        .about("Trigger a manual repack for a room, or every room referencing a shard")
        .arg(
            Arg::new("room")
                .short('r')
                .long("room")
                .conflicts_with("shard"),
        )
        .arg(
            Arg::new("shard")
                .short('s')
                .long("shard")
                .conflicts_with("room")
                .num_args(1..)
                .value_name("SLOT | START-END")
                .help("Repack rooms referencing shard slots (for example: -s 0 1 2 or -s 0-3)"),
        )
        .arg(
            Arg::new("all")
                .long("all")
                .conflicts_with_all(["room", "shard"])
                .action(ArgAction::SetTrue)
                .help("Repack every room in every active shard"),
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
        .about("Delete all data for one or more rooms")
        .arg(
            Arg::new("room")
                .required(true)
                .num_args(1..)
                .value_name("ROOM")
                .help("Room IDs to delete"),
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
        .arg(Arg::new("room").short('r').long("room").required(true))
        .arg(Arg::new("id").short('i').long("id").required(true))
        .arg(Arg::new("data").short('a').long("data").required(true))
}

fn sub_get() -> Command {
    Command::new("get")
        .about("Retrieve a record")
        .arg(Arg::new("room").short('r').long("room"))
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
            room: m.get_one::<String>("room").unwrap().clone(),
            id: m.get_one::<String>("id").unwrap().clone(),
            data: m.get_one::<String>("data").unwrap().clone(),
        },
        Some(("get", m)) => Commands::Get {
            room: m.get_one::<String>("room").cloned(),
            id: m.get_one::<String>("id").unwrap().clone(),
        },
        Some(("rooms", _)) => Commands::Rooms,
        Some(("shards", m)) => Commands::Shards {
            all: m.get_flag("all"),
        },
        Some(("info", m)) => Commands::Info {
            room: m.get_one::<String>("room").unwrap().clone(),
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
            room: m.get_one::<String>("room").cloned(),
        },
        Some(("export", m)) => Commands::Export {
            room: m.get_one::<String>("room").unwrap().clone(),
        },
        Some(("repack", m)) => Commands::Repack {
            room: m.get_one::<String>("room").cloned(),
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
            rooms: m.get_many::<String>("room").unwrap().cloned().collect(),
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
        let mut command = build_cli();
        clap_complete::generate(shell, &mut command, "mtxdb", &mut std::io::stdout());
        return Ok(());
    }
    cmd::run(&cli)
}
