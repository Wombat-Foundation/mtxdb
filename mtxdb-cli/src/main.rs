//! CLI for the mtxdb content-addressed storage engine.

mod cmd;

use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};

pub(crate) struct Cli {
    pub(crate) dir: Option<PathBuf>,
    pub(crate) command: Commands,
}

pub(crate) enum Commands {
    Put {
        room: String,
        id: String,
        data: String,
    },
    Get {
        room: String,
        id: String,
    },
    Rooms,
    Shards,
    Info {
        room: String,
    },
    Scan {
        path: PathBuf,
    },
    Import {
        path: PathBuf,
        room: Option<String>,
    },
    Repack {
        room: Option<String>,
        shard: Option<u16>,
        root: Vec<String>,
        topo: bool,
    },
    Delete {
        room: String,
        yes: bool,
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
                .help("Base directory for packfiles"),
        )
        .subcommand(
            Command::new("shards")
                .about("List open shard slots with size, epoch, and IO/sync stats"),
        )
        .subcommand(Command::new("rooms").about("List rooms in the store"))
        .subcommand(Command::new("sync").about("Bootstrap or refresh persisted shard/room stats"))
        .subcommand(sub_import())
        .subcommand(sub_repack())
        .subcommand(
            Command::new("scan")
                .about("Scan a packfile and print records")
                .arg(Arg::new("path").required(true)),
        )
        .subcommand(
            Command::new("info")
                .about("Show storage info for a room")
                .arg(Arg::new("room").short('r').long("room").required(true)),
        )
        .subcommand(sub_delete())
        .subcommand(sub_put())
        .subcommand(sub_get())
}

fn sub_import() -> Command {
    Command::new("import")
        .about("Import a JSON DAG file (rezzy-compatible format)")
        .arg(Arg::new("path").required(true))
        .arg(
            Arg::new("room")
                .short('r')
                .long("room")
                .help("Room ID (hex, 32 chars). Auto-detected if omitted"),
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
                .value_parser(clap::value_parser!(u16))
                .help("Repack every room still referencing this shard id"),
        )
        .group(
            clap::ArgGroup::new("repack_target")
                .args(["room", "shard"])
                .required(true),
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
        .about("Delete all data for a room")
        .arg(Arg::new("room").short('r').long("room").required(true))
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
        .arg(Arg::new("room").short('r').long("room").required(true))
        .arg(Arg::new("id").short('i').long("id").required(true))
}

fn parse_cli() -> Cli {
    let matches = build_cli().get_matches();

    if matches.get_flag("version") || matches.get_flag("version_upper") {
        println!("mtxdb {}", build_cli().get_version().unwrap());
        std::process::exit(0);
    }

    let dir = matches.get_one::<String>("dir").map(PathBuf::from);

    let command = match matches.subcommand() {
        Some(("put", m)) => Commands::Put {
            room: m.get_one::<String>("room").unwrap().clone(),
            id: m.get_one::<String>("id").unwrap().clone(),
            data: m.get_one::<String>("data").unwrap().clone(),
        },
        Some(("get", m)) => Commands::Get {
            room: m.get_one::<String>("room").unwrap().clone(),
            id: m.get_one::<String>("id").unwrap().clone(),
        },
        Some(("rooms", _)) => Commands::Rooms,
        Some(("shards", _)) => Commands::Shards,
        Some(("info", m)) => Commands::Info {
            room: m.get_one::<String>("room").unwrap().clone(),
        },
        Some(("scan", m)) => Commands::Scan {
            path: PathBuf::from(m.get_one::<String>("path").unwrap()),
        },
        Some(("import", m)) => Commands::Import {
            path: PathBuf::from(m.get_one::<String>("path").unwrap()),
            room: m.get_one::<String>("room").cloned(),
        },
        Some(("repack", m)) => Commands::Repack {
            room: m.get_one::<String>("room").cloned(),
            shard: m.get_one::<u16>("shard").copied(),
            root: m
                .get_many::<String>("root")
                .into_iter()
                .flatten()
                .cloned()
                .collect(),
            topo: m.get_flag("topo"),
        },
        Some(("delete", m)) => Commands::Delete {
            room: m.get_one::<String>("room").unwrap().clone(),
            yes: m.get_flag("yes"),
        },
        Some(("sync", _)) => Commands::Sync,
        _ => {
            build_cli().print_help().unwrap();
            std::process::exit(0);
        }
    };

    Cli { dir, command }
}

fn main() -> anyhow::Result<()> {
    let cli = parse_cli();
    cmd::run(&cli)
}
