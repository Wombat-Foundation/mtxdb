#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects,
    clippy::pedantic
)]

mod cmd;

use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};

pub struct Cli {
    pub dir: Option<PathBuf>,
    pub command: Commands,
}

pub enum Commands {
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
        room: String,
        root: Vec<String>,
    },
    Delete {
        room: String,
        yes: bool,
    },
    Bench {
        count: usize,
    },
}

fn build_cli() -> Command {
    Command::new("mtxdb")
        .about("CLI for the mtxdb content-addressed storage engine")
        .arg(
            Arg::new("dir")
                .short('d')
                .long("dir")
                .env("MTXDB_DIR")
                .value_name("DIR")
                .help("Base directory for packfiles"),
        )
        .subcommand(
            Command::new("put")
                .about("Insert a record")
                .arg(Arg::new("room").short('r').long("room").required(true))
                .arg(Arg::new("id").short('i').long("id").required(true))
                .arg(Arg::new("data").short('a').long("data").required(true)),
        )
        .subcommand(
            Command::new("get")
                .about("Retrieve a record")
                .arg(Arg::new("room").short('r').long("room").required(true))
                .arg(Arg::new("id").short('i').long("id").required(true)),
        )
        .subcommand(Command::new("rooms").about("List rooms in the store"))
        .subcommand(
            Command::new("info")
                .about("Show storage info for a room")
                .arg(Arg::new("room").short('r').long("room").required(true)),
        )
        .subcommand(
            Command::new("scan")
                .about("Scan a packfile and print records")
                .arg(Arg::new("path").required(true)),
        )
        .subcommand(
            Command::new("import")
                .about("Import a JSON DAG file (rezzy-compatible format)")
                .arg(Arg::new("path").required(true))
                .arg(
                    Arg::new("room")
                        .short('r')
                        .long("room")
                        .help("Room ID (hex, 32 chars). Auto-detected if omitted"),
                ),
        )
        .subcommand(
            Command::new("repack")
                .about("Trigger a manual repack for a room")
                .arg(Arg::new("room").short('r').long("room").required(true))
                .arg(
                    Arg::new("root")
                        .short('o')
                        .long("root")
                        .num_args(1..)
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("delete")
                .about("Delete all data for a room")
                .arg(Arg::new("room").short('r').long("room").required(true))
                .arg(
                    Arg::new("yes")
                        .long("yes")
                        .action(ArgAction::SetTrue)
                        .help("Skip confirmation prompt"),
                ),
        )
        .subcommand(
            Command::new("bench")
                .about("Run a quick write/read benchmark")
                .arg(
                    Arg::new("count")
                        .short('c')
                        .long("count")
                        .default_value("10000"),
                ),
        )
}

fn parse_cli() -> Cli {
    let matches = build_cli().get_matches();

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
            room: m.get_one::<String>("room").unwrap().clone(),
            root: m.get_many::<String>("root").unwrap().cloned().collect(),
        },
        Some(("delete", m)) => Commands::Delete {
            room: m.get_one::<String>("room").unwrap().clone(),
            yes: m.get_flag("yes"),
        },
        Some(("bench", m)) => Commands::Bench {
            count: m
                .get_one::<String>("count")
                .unwrap()
                .parse()
                .expect("invalid count"),
        },
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
