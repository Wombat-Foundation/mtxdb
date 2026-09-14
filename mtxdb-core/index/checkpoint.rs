//! On-disk serialization of the per-collection indexes (`index.checkpoint`).
//!
//! Packfiles stay authoritative: this file is a rebuildable acceleration
//! structure. A full `rewrite` (see `PackfileStorage::persist_index_checkpoint`)
//! is done atomically (temp file + rename) whenever the delta log can't
//! continue — a structural change, a broken continuation, or the log's size
//! cap. Between rewrites, incremental index mutations are persisted as
//! batches in the `index.delta` log ([`crate::index::delta`]); the checkpoint
//! pins the exact pack set those frames continue, and a crash at any point
//! leaves a fingerprint mismatch the next open resolves with a rescan — never
//! a wrong replay.
//!
//! Layout (all little-endian, fixed width — see [`crate::index::format`]):
//!
//! ```text
//!   [CheckpointHeader 64B]
//!   [CollectionDirEntry * count, 40B each]
//!   [collection 0 raw slots: capacity * 8B]
//!   [collection 1 raw slots: capacity * 8B]
//!   ...
//! ```
//!
//! The header's `pack_fingerprint` is a deterministic hash of the
//! `(pack_id, length)` set the checkpoint describes. An opener only trusts
//! the file when that fingerprint matches the packs currently on disk (optionally
//! with a gated delta replay continuing it); any append, rotation, or repack
//! changes it and forces a rescan.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::Mmap;

use super::format::{
    CheckpointHeader, CollectionDirEntry, CHECKPOINT_HEADER_LEN, COLLECTION_DIR_ENTRY_LEN,
};

/// Magic identifying the persisted-index checkpoint format.
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"MTXIDX01";
/// Current wire version (see [`CheckpointHeader::version`]).
pub const CHECKPOINT_VERSION: u32 = 1;
/// File name of the persisted index checkpoint inside a store's base dir.
pub const INDEX_CHECKPOINT_FILE: &str = "index.checkpoint";

/// Disambiguates concurrent checkpoint tmp filenames within this process,
/// paired with the process id for uniqueness across processes — same pattern
/// as `SHARD_ROOMS_TMP_COUNTER` / `shard::STATS_TMP_COUNTER`.
static CHECKPOINT_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A checkpoint read off disk, validated enough to trust for index rebuild.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// The pack-fingerprint the checkpoint was written against.
    pub fingerprint: u64,
    /// One entry per collection, in the checkpoint's directory order.
    pub collections: Vec<LoadedCollection>,
    /// Keeps the raw slot arrays alive for mmap-backed indexes built from
    /// this checkpoint.
    pub mmap: Arc<Mmap>,
}

/// One collection's raw slots in the checkpoint mapping.
#[derive(Debug)]
pub struct LoadedCollection {
    /// The collection whose slots these are.
    pub collection_id: [u8; 16],
    /// Checkpoint generation for this collection (reserved for a future
    /// delta-log; always 0 today).
    pub generation: u64,
    /// Absolute byte offset of the raw `capacity * 8` slot array.
    pub slots_offset: usize,
    /// Allocated index capacity.
    pub capacity: u32,
    /// Number of non-empty slots, recorded at checkpoint write time.
    pub slot_count: u32,
}

/// Deterministic FNV-1a hash of the `(pack_id, file_len)` set a checkpoint
/// (or an open store's current packs) describes. The set is sorted before
/// hashing so the result is order-independent.
#[must_use]
pub fn pack_fingerprint(packs: &[(u64, u64)]) -> u64 {
    let mut sorted: Vec<(u64, u64)> = packs.to_vec();
    sorted.sort_unstable();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (pack_id, file_len) in sorted {
        for byte in pack_id.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in file_len.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Decode the 8-byte capacity prefix of a `LossyIndex::serialize` blob.
fn blob_capacity(blob: &[u8]) -> Option<u64> {
    let head: [u8; 8] = blob.get(..8)?.try_into().ok()?;
    Some(u64::from_le_bytes(head))
}

/// Number of occupied (non-`0u64`) slots in a `LossyIndex::serialize` blob's
/// slot section.
fn blob_slot_count(blob: &[u8]) -> u32 {
    u32::try_from(
        blob[8..]
            .chunks_exact(8)
            .filter(|slot| *slot != [0u8; 8])
            .count(),
    )
    .unwrap_or(u32::MAX)
}

/// Write a complete checkpoint atomically: temp file, fsync, rename over the
/// final path. `collections` holds each collection's `serialize` blob plus the
/// index generation it was written at (recorded in its directory entry so a
/// delta log's frames can be gated against the same generation on replay),
/// the first collection listed becoming the first in the directory order.
///
/// # Errors
/// Returns `io::Error` on any failure; the previous checkpoint (if any) is
/// left intact in that case.
pub fn write_checkpoint(
    path: &Path,
    fingerprint: u64,
    collections: &[([u8; 16], u64, &[u8])],
) -> std::io::Result<()> {
    let count = u32::try_from(collections.len())
        .map_err(|_| std::io::Error::other("too many collections for checkpoint u32"))?;
    let directory_bytes = u64::try_from(collections.len())
        .ok()
        .and_then(|n| n.checked_mul(COLLECTION_DIR_ENTRY_LEN as u64))
        .ok_or_else(|| std::io::Error::other("checkpoint directory size overflow"))?;
    let mut slots_bytes: u64 = 0;
    for (_, _, blob) in collections {
        let blob_len = u64::try_from(blob.len().saturating_sub(8))
            .map_err(|_| std::io::Error::other("collection slot array too large"))?;
        slots_bytes = slots_bytes
            .checked_add(blob_len)
            .ok_or_else(|| std::io::Error::other("checkpoint slots size overflow"))?;
    }

    let header = CheckpointHeader {
        magic: CHECKPOINT_MAGIC,
        version: CHECKPOINT_VERSION,
        collection_count: count,
        directory_bytes,
        slots_bytes,
        pack_fingerprint: fingerprint,
    };
    let total_len = CHECKPOINT_HEADER_LEN
        .saturating_add(usize::try_from(directory_bytes).unwrap_or(usize::MAX))
        .saturating_add(usize::try_from(slots_bytes).unwrap_or(usize::MAX));
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&header.encode());

    let mut slots_offset: u64 = 0;
    let mut dir_entries = Vec::with_capacity(collections.len());
    for (collection_id, generation, blob) in collections {
        let capacity = blob_capacity(blob)
            .ok_or_else(|| std::io::Error::other("collection slot blob missing capacity"))?;
        let capacity32 = u32::try_from(capacity)
            .map_err(|_| std::io::Error::other("index capacity exceeds u32"))?;
        let slot_array_len: u64 = blob[8..]
            .len()
            .try_into()
            .map_err(|_| std::io::Error::other("collection slot array too large"))?;
        dir_entries.push(CollectionDirEntry {
            collection_id: *collection_id,
            generation: *generation,
            slots_offset,
            capacity: capacity32,
            slot_count: blob_slot_count(blob),
        });
        slots_offset = slots_offset
            .checked_add(slot_array_len)
            .ok_or_else(|| std::io::Error::other("checkpoint slots offset overflow"))?;
    }
    for entry in &dir_entries {
        buf.extend_from_slice(&entry.encode());
    }
    for (_, _, blob) in collections {
        buf.extend_from_slice(&blob[8..]);
    }

    let unique = CHECKPOINT_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = PathBuf::from(format!(
        "{}.tmp.{}.{unique}",
        path.display(),
        std::process::id()
    ));
    let write_result = (|| -> std::io::Result<()> {
        let mut tmp = fs::File::create(&tmp_path)?;
        tmp.write_all(&buf)?;
        tmp.sync_all()
    })();
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    fs::rename(&tmp_path, path)
}

/// Read and validate a checkpoint. Returns `None` for any problem — a
/// missing, truncated, or malformed file is treated identically to "no
/// checkpoint", and the caller falls back to rebuilding from the packs.
#[must_use]
pub fn read_checkpoint(path: &Path) -> Option<LoadedCheckpoint> {
    let file = fs::File::open(path).ok()?;
    let mmap = Arc::new(crate::packfile::map_pack(&file).ok()?);
    let buf: &[u8] = &mmap;
    let header_bytes: [u8; CHECKPOINT_HEADER_LEN] = buf[..CHECKPOINT_HEADER_LEN].try_into().ok()?;
    let header = CheckpointHeader::decode(&header_bytes)?;
    if header.magic != CHECKPOINT_MAGIC || header.version != CHECKPOINT_VERSION {
        return None;
    }
    let count = header.collection_count as usize;
    if header.directory_bytes
        != u64::try_from(count)
            .ok()?
            .checked_mul(COLLECTION_DIR_ENTRY_LEN as u64)?
    {
        return None;
    }
    let slot_base =
        CHECKPOINT_HEADER_LEN.checked_add(usize::try_from(header.directory_bytes).ok()?)?;
    if buf.len() != slot_base.checked_add(usize::try_from(header.slots_bytes).ok()?)? {
        return None;
    }

    let mut collections = Vec::with_capacity(count);
    for i in 0..count {
        let entry_offset =
            CHECKPOINT_HEADER_LEN.checked_add(i.checked_mul(COLLECTION_DIR_ENTRY_LEN)?)?;
        let entry_bytes: [u8; COLLECTION_DIR_ENTRY_LEN] = buf
            .get(entry_offset..entry_offset.saturating_add(COLLECTION_DIR_ENTRY_LEN))?
            .try_into()
            .ok()?;
        let entry = CollectionDirEntry::decode(&entry_bytes)?;
        if entry.capacity < 16 || !entry.capacity.is_power_of_two() {
            return None;
        }
        let slots_len = (entry.capacity as usize).checked_mul(8)?;
        let region = slot_base.checked_add(usize::try_from(entry.slots_offset).ok()?)?;
        buf.get(region..region.checked_add(slots_len)?)?;
        collections.push(LoadedCollection {
            collection_id: entry.collection_id,
            generation: entry.generation,
            slots_offset: region,
            capacity: entry.capacity,
            slot_count: entry.slot_count,
        });
    }

    Some(LoadedCheckpoint {
        fingerprint: header.pack_fingerprint,
        collections,
        mmap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::LossyIndex;

    fn hash_for(seed: u16, i: usize) -> [u8; 16] {
        let mut hash = [0u8; 16];
        hash[..2].copy_from_slice(&seed.wrapping_add(u16::try_from(i).unwrap()).to_le_bytes());
        hash
    }

    fn index_with_entries(seed: u16, count: usize) -> LossyIndex {
        let index = LossyIndex::new(count.max(16).saturating_mul(2));
        for i in 0..count {
            let shard = u16::try_from(i % 64).unwrap();
            let _ = index.insert(&hash_for(seed, i), shard, (i as u64).wrapping_mul(128));
        }
        index
    }

    #[test]
    fn fingerprint_is_deterministic_but_sensitive() {
        let packs = [(3, 100), (1, 50), (2, 75)];
        assert_eq!(pack_fingerprint(&packs), pack_fingerprint(&packs));
        assert_eq!(
            pack_fingerprint(&packs),
            pack_fingerprint(&[(1, 50), (2, 75), (3, 100)])
        );
        // A length change (append/rotation) must change the fingerprint.
        assert_ne!(
            pack_fingerprint(&packs),
            pack_fingerprint(&[(1, 51), (2, 75), (3, 100)])
        );
        // A pack-id change (repack/retire) must change it too.
        assert_ne!(
            pack_fingerprint(&packs),
            pack_fingerprint(&[(1, 50), (2, 75), (4, 100)])
        );
    }

    #[test]
    fn checkpoint_round_trips_index_slots() {
        let cases: Vec<([u8; 16], u16, usize)> =
            vec![([1u8; 16], 1, 40), ([2u8; 16], 2, 200), ([3u8; 16], 3, 5)];
        let indexes: Vec<LossyIndex> = cases
            .iter()
            .map(|(_, seed, count)| index_with_entries(*seed, *count))
            .collect();
        let blobs: Vec<([u8; 16], Vec<u8>)> = cases
            .iter()
            .zip(&indexes)
            .map(|((id, _, _), index)| (*id, index.serialize()))
            .collect();
        let dir = std::env::temp_dir().join(format!("mtxdb_checkpoint_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_CHECKPOINT_FILE);

        let fingerprint = pack_fingerprint(&[(7, 12345)]);
        write_checkpoint(
            &path,
            fingerprint,
            &blobs
                .iter()
                .map(|(id, blob)| (*id, 0, blob.as_slice()))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let loaded = read_checkpoint(&path).unwrap_or_else(|| {
            panic!(
                "valid checkpoint should read; file has {} bytes",
                std::fs::read(&path).map_or(0, |b| b.len())
            )
        });
        assert_eq!(loaded.fingerprint, fingerprint);
        assert_eq!(
            loaded.collections.len(),
            3,
            "checkpoint must preserve directory order and count"
        );
        let mmap = Arc::clone(&loaded.mmap);
        for ((case, loaded), (_blob_id, expected_blob)) in
            cases.iter().zip(&loaded.collections).zip(&blobs)
        {
            let (expected_id, seed, count) = case;
            assert_eq!(loaded.collection_id, *expected_id);
            let slots_len = (loaded.capacity as usize).saturating_mul(8);
            assert_eq!(
                &mmap[loaded.slots_offset..loaded.slots_offset.saturating_add(slots_len)],
                &expected_blob[8..],
                "slots must round-trip verbatim"
            );

            // The mmap-backed index must find every hash the live index had.
            let loaded_index = LossyIndex::from_mmap_slots(
                Arc::clone(&mmap),
                loaded.slots_offset,
                loaded.capacity,
                loaded.slot_count,
            );
            let original_index = LossyIndex::deserialize(expected_blob).unwrap();
            for i in 0..*count {
                let hash = hash_for(*seed, i);
                assert_eq!(
                    loaded_index.lookup(&hash),
                    original_index.lookup(&hash),
                    "loaded index must agree with the source index on spot lookups"
                );
                assert!(
                    loaded_index.lookup(&hash).is_some(),
                    "round-tripped index lost a record"
                );
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_checkpoint_round_trips() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_checkpoint_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_CHECKPOINT_FILE);
        write_checkpoint(&path, pack_fingerprint(&[]), &[]).unwrap();
        let loaded = read_checkpoint(&path).expect("empty checkpoint is still a valid file");
        assert_eq!(loaded.fingerprint, pack_fingerprint(&[]));
        assert!(loaded.collections.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupt_checkpoints_read_as_none() {
        let dir = std::env::temp_dir().join(format!("mtxdb_checkpoint_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_CHECKPOINT_FILE);

        // Missing file.
        assert!(read_checkpoint(&path).is_none());

        // Truncated (valid header, cut slot section).
        let blobs = [([1u8; 16], index_with_entries(1, 40).serialize())];
        write_checkpoint(
            &path,
            0,
            &blobs
                .iter()
                .map(|(id, b)| (*id, 0, b.as_slice()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len().saturating_sub(10)]).unwrap();
        assert!(read_checkpoint(&path).is_none());

        // Wrong magic.
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0xEE; 64]).unwrap();
        assert!(read_checkpoint(&path).is_none());

        // Wrong version byte in an otherwise valid file.
        write_checkpoint(
            &path,
            0,
            &blobs
                .iter()
                .map(|(id, b)| (*id, 0, b.as_slice()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, {
            let mut v = full.clone();
            v[8..12].copy_from_slice(&99u32.to_le_bytes());
            v
        })
        .unwrap();
        assert!(read_checkpoint(&path).is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
