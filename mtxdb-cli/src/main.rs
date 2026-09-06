#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::arithmetic_side_effects,
    clippy::pedantic
)]

mod cmd;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "mtxdb",
    about = "CLI for the mtxdb content-addressed storage engine",
    version,
    author
)]
pub struct Cli {
    /// Base directory for packfiles (default: $HOME/.mtxdb)
    #[arg(short, long, env = "MTXDB_DIR")]
    pub dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Insert a record
    Put {
        /// Room ID (hex, 32 chars)
        #[arg(short, long)]
        room: String,

        /// Node ID (hex, 32 chars)
        #[arg(short, long)]
        id: String,

        /// Data (UTF-8 string)
        #[arg(short, long)]
        data: String,
    },

    /// Retrieve a record
    Get {
        /// Room ID (hex, 32 chars)
        #[arg(short, long)]
        room: String,

        /// Node ID (hex, 32 chars)
        #[arg(short, long)]
        id: String,
    },

    /// List rooms in the store
    Rooms,

    /// Show storage info for a room
    Info {
        /// Room ID (hex, 32 chars)
        #[arg(short, long)]
        room: String,
    },

    /// Scan a packfile and print records
    Scan {
        /// Packfile path
        path: PathBuf,
    },

    /// Trigger a manual repack for a room
    Repack {
        /// Room ID (hex, 32 chars)
        #[arg(short, long)]
        room: String,

        /// Root node IDs to preserve (hex, repeatable)
        #[arg(short, long, num_args(1..))]
        root: Vec<String>,
    },

    /// Delete all data for a room
    Delete {
        /// Room ID (hex, 32 chars)
        #[arg(short, long)]
        room: String,

        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Run a quick write/read benchmark
    Bench {
        /// Number of records
        #[arg(short, long, default_value = "10000")]
        count: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    cmd::run(&cli)
}
