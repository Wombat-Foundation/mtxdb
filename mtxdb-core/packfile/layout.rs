//! Physical, on-disk layout scanning: how collections' records are
//! physically interleaved across shard files, independent of any
//! collection's live index. This intentionally includes superseded
//! frames — it reflects the physical interleaving a sequential scan
//! sees, not a claim about any collection's live index footprint.
//!
//! Shared by `mtxdb-cli`'s `collections --layout`/`shards --layout`
//! inspection commands and locality benchmarks, so fragmentation
//! numbers (segments, avoidable spread bytes) come from one canonical
//! scan instead of two implementations that could silently drift apart.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::Path;

use super::{read_header, FRAME_FIXED_LEN, MAX_RECORD_LEN};

/// Raw, on-disk physical layout of every pack file in a directory.
#[derive(Debug, Default)]
pub struct PhysicalLayout {
    /// Per-collection physical footprint, keyed by collection ID.
    pub collections: HashMap<[u8; 16], CollectionPhysicalLayout>,
    /// Per-pack physical composition, keyed by `pack_id`.
    pub packs: HashMap<u64, PackPhysicalLayout>,
}

/// One collection's physical footprint across every pack file scanned.
#[derive(Debug, Default, Clone)]
pub struct CollectionPhysicalLayout {
    /// Total on-disk bytes (frame + CRC) across every pack, including
    /// superseded frames a live index would no longer reference.
    pub disk_bytes: u64,
    /// On-disk bytes contributed to each pack this collection appears in.
    pub pack_bytes: HashMap<u64, u64>,
    /// Number of maximal contiguous runs of this collection's own
    /// records, across every shard file — a fragmentation signal
    /// sharper than "distinct packs touched": one collection's records
    /// can be non-contiguous *within* a single shard file too, when
    /// interleaved with other collections sharing that shard.
    pub segments: u64,
    /// Size in bytes of this collection's single largest contiguous run.
    pub largest_segment_bytes: u64,
}

/// One pack file's physical composition.
#[derive(Debug, Default, Clone)]
pub struct PackPhysicalLayout {
    /// Every collection with at least one record in this pack.
    pub collections: HashSet<[u8; 16]>,
    /// Number of maximal contiguous single-collection runs in this pack.
    pub segments: u64,
    /// Size in bytes of this pack's single largest contiguous run.
    pub largest_segment_bytes: u64,
}

impl CollectionPhysicalLayout {
    /// Bytes that could be moved to maximize this collection's largest
    /// pack extent — i.e. spread beyond what a single `MAX_SHARD_BYTES`
    /// pack could hold regardless of layout, which is required spill,
    /// not avoidable fragmentation.
    #[must_use]
    pub fn avoidable_spread_bytes(&self) -> u64 {
        let largest_pack = self.pack_bytes.values().copied().max().unwrap_or(0);
        self.disk_bytes
            .min(crate::shard::MAX_SHARD_BYTES)
            .saturating_sub(largest_pack)
    }
}

/// Bytes that could be moved to maximize one collection's largest pack
/// extent. `None` (collection has no physical footprint at all) is 0.
#[must_use]
pub fn avoidable_spread_bytes(stats: Option<&CollectionPhysicalLayout>) -> u64 {
    stats.map_or(0, CollectionPhysicalLayout::avoidable_spread_bytes)
}

/// Scan physical frame placement across every `.pack` file in `dir`,
/// without allocating payloads. A torn active tail (from a concurrent
/// writer) is treated as end-of-file, same as the ordinary disk-usage
/// scan.
///
/// # Errors
/// Returns `io::Error` on directory read failure, file I/O failure, or
/// an invalid/corrupt record length.
pub fn physical_layout(dir: &Path) -> io::Result<PhysicalLayout> {
    let mut layout = PhysicalLayout::default();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .extension()
            .is_some_and(|extension| extension == "pack")
        {
            continue;
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let Some(header) = read_header(&mut reader)? else {
            continue;
        };
        let pack_id = header.pack_id;
        let mut current_run: Option<([u8; 16], u64)> = None;
        let mut discard = [0_u8; 8192];
        loop {
            let mut len = [0_u8; 4];
            match reader.read_exact(&mut len) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            let frame_len = u64::from(u32::from_le_bytes(len));
            if !(u64::from(FRAME_FIXED_LEN)..=u64::from(MAX_RECORD_LEN)).contains(&frame_len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid record length {frame_len} while measuring collection disk usage"
                    ),
                ));
            }
            // Skip the flags byte and uncompressed_len field to reach
            // collection_id — this scan only needs the collection ID,
            // not whether/how the node bytes that follow are compressed.
            let mut flags_and_uncompressed_len = [0_u8; 5];
            match reader.read_exact(&mut flags_and_uncompressed_len) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            let mut collection_id = [0_u8; 16];
            match reader.read_exact(&mut collection_id) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            let mut remaining = frame_len.saturating_sub(21).saturating_add(4);
            let mut complete = true;
            while remaining != 0 {
                let chunk_len = usize::try_from(remaining)
                    .unwrap_or(usize::MAX)
                    .min(discard.len());
                match reader.read_exact(&mut discard[..chunk_len]) {
                    Ok(()) => remaining = remaining.saturating_sub(chunk_len as u64),
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                        complete = false;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            // A concurrent append may expose a torn tail. If the final
            // read was short, leave it for the next scan once complete.
            if !complete {
                break;
            }
            let record_bytes = frame_len.saturating_add(8);
            let collection = layout.collections.entry(collection_id).or_default();
            collection.disk_bytes = collection.disk_bytes.saturating_add(record_bytes);
            let pack_bytes = collection.pack_bytes.entry(pack_id).or_default();
            *pack_bytes = pack_bytes.saturating_add(record_bytes);
            layout
                .packs
                .entry(pack_id)
                .or_default()
                .collections
                .insert(collection_id);

            match &mut current_run {
                Some((run_collection, run_bytes)) if *run_collection == collection_id => {
                    *run_bytes = run_bytes.saturating_add(record_bytes);
                }
                Some(_) => {
                    finish_physical_run(&mut layout, pack_id, current_run.take());
                    current_run = Some((collection_id, record_bytes));
                }
                None => current_run = Some((collection_id, record_bytes)),
            }
        }
        finish_physical_run(&mut layout, pack_id, current_run);
    }
    Ok(layout)
}

fn finish_physical_run(layout: &mut PhysicalLayout, pack_id: u64, run: Option<([u8; 16], u64)>) {
    let Some((collection_id, bytes)) = run else {
        return;
    };
    let collection = layout.collections.entry(collection_id).or_default();
    collection.segments = collection.segments.saturating_add(1);
    collection.largest_segment_bytes = collection.largest_segment_bytes.max(bytes);
    let pack = layout.packs.entry(pack_id).or_default();
    pack.segments = pack.segments.saturating_add(1);
    pack.largest_segment_bytes = pack.largest_segment_bytes.max(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packfile::{write_header, write_record, Record};
    use bytes::Bytes;

    #[test]
    fn physical_layout_counts_cross_pack_spread_and_interleaved_runs() {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_core_layout_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let collection_a = [0xA1; 16];
        let collection_b = [0xB2; 16];
        for (pack_id, records) in [
            (0_u64, vec![collection_a, collection_b, collection_a]),
            (1_u64, vec![collection_a]),
        ] {
            let path = dir.join(format!("pack_{pack_id:016x}.pack"));
            let mut file = File::create(path).unwrap();
            write_header(&mut file, pack_id).unwrap();
            for (index, collection_id) in records.into_iter().enumerate() {
                write_record(
                    &mut file,
                    &Record {
                        collection_id,
                        hash: [u8::try_from(index).expect("fixture index fits in u8"); 16],
                        data: Bytes::from_static(b"payload"),
                    },
                )
                .unwrap();
            }
        }

        let layout = physical_layout(&dir).unwrap();
        let a = &layout.collections[&collection_a];
        assert_eq!(a.pack_bytes.len(), 2, "A spans two packs");
        assert_eq!(a.segments, 3, "A-B-A in pack 0 plus A in pack 1");
        assert_eq!(layout.packs[&0].segments, 3);
        assert_eq!(layout.packs[&1].segments, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn avoidable_spread_excludes_a_collections_required_spill() {
        let capacity = crate::shard::MAX_SHARD_BYTES;
        let mut ideal = CollectionPhysicalLayout {
            disk_bytes: capacity.saturating_add(100),
            ..CollectionPhysicalLayout::default()
        };
        ideal.pack_bytes.insert(0, capacity);
        ideal.pack_bytes.insert(1, 100);
        assert_eq!(avoidable_spread_bytes(Some(&ideal)), 0);

        let mut fragmented = ideal;
        fragmented.pack_bytes.clear();
        fragmented.pack_bytes.insert(0, capacity / 2);
        fragmented.pack_bytes.insert(1, capacity / 2 + 100);
        assert_eq!(
            avoidable_spread_bytes(Some(&fragmented)),
            capacity.saturating_sub(capacity / 2 + 100)
        );
    }
}
