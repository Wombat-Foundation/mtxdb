//! Split sealed-sidecar open cost into mapping and full verification.
//!
//! Paged open calls `open_for_pack`, which maps each sidecar and then runs a
//! full `verify()` (CRC, sorted ids, fanout, frame offsets). This example times
//! the two halves separately over the oldest packs in a shard directory.
//!
//! ```text
//! cargo run --release --example sidecar_verify_split -- <shard-dir> [max-packs]
//! ```
//!
//! The first round may include page-cache misses; later rounds are warm.

use std::fs::File;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::index::packidx::{pack_index_path, PackIndex, PackIndexIdentity};
use mtxdb::packfile::{read_header, PackId};

const ROUNDS: u32 = 3;
const DEFAULT_MAX_PACKS: usize = 1_001;

fn identity(pack_path: &Path) -> std::io::Result<PackIndexIdentity> {
    let mut file = File::open(pack_path)?;
    let header = read_header(&mut file)?.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing pack header")
    })?;
    Ok(PackIndexIdentity {
        pack_id: PackId(header.pack_id.0),
        creation_seq: u64::from(header.creation_seq),
        pack_file_len: file.metadata()?.len(),
    })
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: sidecar_verify_split <shard-dir> [max-packs]")?,
    );
    let max_packs = match args.next() {
        Some(value) => value
            .to_str()
            .ok_or("max-packs must be UTF-8")?
            .parse::<usize>()?,
        None => DEFAULT_MAX_PACKS,
    };

    let mut packs = std::fs::read_dir(&dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "pack")
        })
        .filter(|path| pack_index_path(path).exists())
        .map(|path| identity(&path).map(|identity| (identity.creation_seq, path, identity)))
        .collect::<Result<Vec<_>, _>>()?;
    packs.sort_by_key(|(creation_seq, _, _)| *creation_seq);
    packs.truncate(max_packs);
    let paths = packs
        .iter()
        .map(|(_, path, identity)| (pack_index_path(path), *identity))
        .collect::<Vec<_>>();

    println!("packs={} rounds={ROUNDS}", paths.len());
    for round in 1..=ROUNDS {
        let mut map_open = Duration::ZERO;
        let mut verify = Duration::ZERO;
        for (index_path, expected) in &paths {
            let started = Instant::now();
            let index = PackIndex::open_mmap(index_path, *expected)?;
            map_open = map_open.saturating_add(started.elapsed());

            let started = Instant::now();
            index.verify()?;
            verify = verify.saturating_add(started.elapsed());
            black_box(index);
        }
        println!(
            "round={round} open_mmap_total_ms={:.3} verify_total_ms={:.3} combined_ms={:.3}",
            millis(map_open),
            millis(verify),
            millis(map_open.saturating_add(verify)),
        );
    }
    Ok(())
}
