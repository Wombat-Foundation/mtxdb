//! Immutable, per-pack locator index format.
//!
//! The pack remains authoritative. This file stores only
//! `(collection_id, logical_id) -> byte_offset` locators and can be rebuilt by
//! scanning the pack. The module is not wired into pack sealing or open yet.
//!
//! All integers are little-endian. Version 1 is laid out as:
//!
//! ```text
//! [64-byte header]
//! [collection directory: 24 bytes per collection]
//! [collection sections, ordered by collection_id]
//! [the CRC32 field in the header covers the whole file, zeroed while hashing]
//! ```
//!
//! A directory entry contains a 16-byte collection id, a `u32` entry count,
//! and a `u32` offset relative to the start of the section area. A section is
//! `fanout[256]` (cumulative `u32` counts), sorted 16-byte logical ids, then
//! parallel `u32` pack offsets. Duplicate `(collection_id, logical_id)` input
//! records are collapsed to the greatest offset, which is the newest frame in
//! that pack.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::packfile::PackId;

/// Magic identifying a v1 per-pack sorted locator index.
pub const PACK_INDEX_MAGIC: [u8; 8] = *b"MTXIDX01";
/// Current on-disk format version.
pub const PACK_INDEX_VERSION: u32 = 1;
/// Fixed header length in bytes.
pub const PACK_INDEX_HEADER_LEN: usize = 64;
/// One collection directory entry in bytes.
pub const PACK_INDEX_COLLECTION_ENTRY_LEN: usize = 24;
/// Cumulative first-byte fanout table length in bytes.
pub const PACK_INDEX_FANOUT_LEN: usize = 256 * 4;
const CRC_RANGE: std::ops::Range<usize> = 60..64;
const MAX_FILE_LEN: u64 = 0xFFFF_FFFF;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One locator supplied to [`PackIndex::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackIndexInput {
    /// The collection containing the logical record.
    pub collection_id: [u8; 16],
    /// The record's full logical id.
    pub logical_id: [u8; 16],
    /// Byte offset of the frame in the pack.
    pub offset: u64,
}

/// A validated in-memory view of a per-pack sorted index.
#[derive(Debug, Clone)]
pub struct PackIndex {
    bytes: Vec<u8>,
    collections: Vec<CollectionView>,
    pack_id: PackId,
    creation_seq: u64,
    pack_file_len: u64,
    entry_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct CollectionView {
    id: [u8; 16],
    count: u32,
    section_at: usize,
}

/// Errors found while constructing or validating a pack index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackIndexError {
    /// The encoding is truncated, malformed, unsorted, or internally inconsistent.
    Invalid,
    /// A pack offset or encoded file section exceeds the v1 `u32` limit.
    TooLarge,
    /// An input offset does not point inside the declared pack extent.
    OffsetOutsidePack {
        /// The input frame offset.
        offset: u64,
        /// The pack extent used during construction.
        pack_len: u64,
    },
}

impl std::fmt::Display for PackIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid => f.write_str("invalid pack index"),
            Self::TooLarge => f.write_str("pack index v1 exceeds its u32 size limit"),
            Self::OffsetOutsidePack { offset, pack_len } => {
                write!(f, "pack offset {offset} is outside pack length {pack_len}")
            }
        }
    }
}

impl std::error::Error for PackIndexError {}

impl PackIndex {
    /// Build a canonical v1 index, collapsing duplicate ids to their greatest offset.
    ///
    /// # Errors
    /// Returns an error if an input offset is outside `pack_file_len`, or if
    /// v1's 32-bit offsets/file-relative section positions cannot represent
    /// the result.
    pub fn build(
        pack_id: PackId,
        creation_seq: u64,
        pack_file_len: u64,
        inputs: &[PackIndexInput],
    ) -> Result<Self, PackIndexError> {
        if pack_id.is_zero() {
            return Err(PackIndexError::Invalid);
        }
        let mut by_collection: BTreeMap<[u8; 16], BTreeMap<[u8; 16], u32>> = BTreeMap::new();
        for input in inputs {
            if input.offset >= pack_file_len {
                return Err(PackIndexError::OffsetOutsidePack {
                    offset: input.offset,
                    pack_len: pack_file_len,
                });
            }
            let offset = u32::try_from(input.offset).map_err(|_| PackIndexError::TooLarge)?;
            by_collection
                .entry(input.collection_id)
                .or_default()
                .entry(input.logical_id)
                .and_modify(|old| *old = (*old).max(offset))
                .or_insert(offset);
        }
        Self::from_sorted_maps(pack_id, creation_seq, pack_file_len, by_collection)
    }

    fn from_sorted_maps(
        pack_id: PackId,
        creation_seq: u64,
        pack_file_len: u64,
        by_collection: BTreeMap<[u8; 16], BTreeMap<[u8; 16], u32>>,
    ) -> Result<Self, PackIndexError> {
        let collection_count =
            u32::try_from(by_collection.len()).map_err(|_| PackIndexError::TooLarge)?;
        let entry_count = by_collection
            .values()
            .try_fold(0_u64, |sum, entries| {
                sum.checked_add(u64::try_from(entries.len()).ok()?)
            })
            .ok_or(PackIndexError::TooLarge)?;
        let directory_bytes = by_collection
            .len()
            .checked_mul(PACK_INDEX_COLLECTION_ENTRY_LEN)
            .ok_or(PackIndexError::TooLarge)?;
        let directory_start = PACK_INDEX_HEADER_LEN;
        let sections_start = directory_start
            .checked_add(directory_bytes)
            .ok_or(PackIndexError::TooLarge)?;
        let mut sections = Vec::new();
        let mut directory = Vec::with_capacity(directory_bytes);
        let mut views = Vec::with_capacity(by_collection.len());
        for (collection_id, entries) in by_collection {
            let count = u32::try_from(entries.len()).map_err(|_| PackIndexError::TooLarge)?;
            if count == 0 {
                return Err(PackIndexError::Invalid);
            }
            let relative_offset =
                u32::try_from(sections.len()).map_err(|_| PackIndexError::TooLarge)?;
            let section_at = sections_start
                .checked_add(sections.len())
                .ok_or(PackIndexError::TooLarge)?;
            directory.extend_from_slice(&collection_id);
            directory.extend_from_slice(&count.to_le_bytes());
            directory.extend_from_slice(&relative_offset.to_le_bytes());

            let mut fanout = [0_u32; 256];
            for logical_id in entries.keys() {
                let first = usize::from(logical_id[0]);
                fanout[first] = fanout[first]
                    .checked_add(1)
                    .ok_or(PackIndexError::TooLarge)?;
            }
            let mut cumulative = 0_u32;
            for frequency in &mut fanout {
                cumulative = cumulative
                    .checked_add(*frequency)
                    .ok_or(PackIndexError::TooLarge)?;
                *frequency = cumulative;
                sections.extend_from_slice(&cumulative.to_le_bytes());
            }
            for logical_id in entries.keys() {
                sections.extend_from_slice(logical_id);
            }
            for offset in entries.values() {
                sections.extend_from_slice(&offset.to_le_bytes());
            }
            views.push(CollectionView {
                id: collection_id,
                count,
                section_at,
            });
        }

        let file_len = sections_start
            .checked_add(sections.len())
            .ok_or(PackIndexError::TooLarge)?;
        if u64::try_from(file_len).map_err(|_| PackIndexError::TooLarge)? > MAX_FILE_LEN {
            return Err(PackIndexError::TooLarge);
        }
        let mut bytes = Vec::with_capacity(file_len);
        bytes.extend_from_slice(&PACK_INDEX_MAGIC);
        bytes.extend_from_slice(pack_id.as_bytes());
        bytes.extend_from_slice(&creation_seq.to_le_bytes());
        bytes.extend_from_slice(&entry_count.to_le_bytes());
        bytes.extend_from_slice(&collection_count.to_le_bytes());
        bytes.extend_from_slice(&pack_file_len.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(file_len)
                .map_err(|_| PackIndexError::TooLarge)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        debug_assert_eq!(
            bytes.len(),
            PACK_INDEX_HEADER_LEN,
            "encoded header fields must exactly fill the fixed header"
        );
        bytes.extend_from_slice(&directory);
        bytes.extend_from_slice(&sections);
        let crc = crc_with_zeroed_field(&bytes).ok_or(PackIndexError::Invalid)?;
        bytes[CRC_RANGE].copy_from_slice(&crc.to_le_bytes());
        Ok(Self {
            bytes,
            collections: views,
            pack_id,
            creation_seq,
            pack_file_len,
            entry_count,
        })
    }

    /// Parse and fully validate a v1 pack index.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PACK_INDEX_HEADER_LEN || bytes.get(..8)? != PACK_INDEX_MAGIC {
            return None;
        }
        let mut pack_bytes = [0_u8; 16];
        pack_bytes.copy_from_slice(bytes.get(8..24)?);
        let pack_id = PackId(pack_bytes);
        if pack_id.is_zero() {
            return None;
        }
        let creation_seq = read_u64(bytes, 24)?;
        let entry_count = read_u64(bytes, 32)?;
        let collection_count = usize::try_from(read_u32(bytes, 40)?).ok()?;
        let pack_file_len = read_u64(bytes, 44)?;
        let declared_len = usize::try_from(read_u64(bytes, 52)?).ok()?;
        if declared_len != bytes.len()
            || bytes.len() > usize::try_from(MAX_FILE_LEN).ok()?
            || read_u32(bytes, 60)? != crc_with_zeroed_field(bytes)?
        {
            return None;
        }
        let directory_len = collection_count.checked_mul(PACK_INDEX_COLLECTION_ENTRY_LEN)?;
        let sections_start = PACK_INDEX_HEADER_LEN.checked_add(directory_len)?;
        if sections_start > bytes.len() {
            return None;
        }
        let mut collections = Vec::with_capacity(collection_count);
        let mut previous_id: Option<[u8; 16]> = None;
        let mut total_entries = 0_u64;
        let mut expected_section_offset = 0_usize;
        let mut previous_end = sections_start;
        for i in 0..collection_count {
            let at = PACK_INDEX_HEADER_LEN.checked_add(i.checked_mul(24)?)?;
            let id: [u8; 16] = bytes.get(at..at.checked_add(16)?)?.try_into().ok()?;
            if previous_id.is_some_and(|previous| previous >= id) {
                return None;
            }
            previous_id = Some(id);
            let count = read_u32(bytes, at.checked_add(16)?)?;
            if count == 0 {
                return None;
            }
            let relative_offset = usize::try_from(read_u32(bytes, at.checked_add(20)?)?).ok()?;
            if relative_offset != expected_section_offset {
                return None;
            }
            let section_at = sections_start.checked_add(relative_offset)?;
            let count_usize = usize::try_from(count).ok()?;
            let section_len = PACK_INDEX_FANOUT_LEN.checked_add(count_usize.checked_mul(20)?)?;
            let end = section_at.checked_add(section_len)?;
            if section_at != previous_end || end > bytes.len() {
                return None;
            }
            validate_section(bytes, section_at, count)?;
            total_entries = total_entries.checked_add(u64::from(count))?;
            expected_section_offset = expected_section_offset.checked_add(section_len)?;
            previous_end = end;
            collections.push(CollectionView {
                id,
                count,
                section_at,
            });
        }
        if previous_end != bytes.len() || total_entries != entry_count {
            return None;
        }
        Some(Self {
            bytes: bytes.to_vec(),
            collections,
            pack_id,
            creation_seq,
            pack_file_len,
            entry_count,
        })
    }

    /// The pack identity recorded in the header.
    #[must_use]
    pub const fn pack_id(&self) -> PackId {
        self.pack_id
    }

    /// The pack's creation sequence recorded in the header.
    #[must_use]
    pub const fn creation_seq(&self) -> u64 {
        self.creation_seq
    }

    /// The exact pack length this index was built for.
    #[must_use]
    pub const fn pack_file_len(&self) -> u64 {
        self.pack_file_len
    }

    /// Number of unique `(collection_id, logical_id)` locators.
    #[must_use]
    pub const fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Encoded bytes, including the header and CRC-protected body.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether the index belongs to the exact pack identity and current length.
    #[must_use]
    pub fn matches_pack(&self, pack_id: PackId, creation_seq: u64, pack_file_len: u64) -> bool {
        self.pack_id == pack_id
            && self.creation_seq == creation_seq
            && self.pack_file_len == pack_file_len
    }

    /// Resolve a key to its frame offset in this pack.
    #[must_use]
    pub fn get(&self, collection_id: &[u8; 16], logical_id: &[u8; 16]) -> Option<u32> {
        let collection = self
            .collections
            .binary_search_by_key(collection_id, |view| view.id)
            .ok()
            .and_then(|index| self.collections.get(index))?;
        let count = usize::try_from(collection.count).ok()?;
        let fan_at = collection.section_at;
        let first = usize::from(logical_id[0]);
        let lo = if first == 0 {
            0
        } else {
            let previous_first = first.checked_sub(1)?;
            usize::try_from(read_u32(
                &self.bytes,
                fan_at.checked_add(previous_first.checked_mul(4)?)?,
            )?)
            .ok()?
        };
        let hi = usize::try_from(read_u32(
            &self.bytes,
            fan_at.checked_add(first.checked_mul(4)?)?,
        )?)
        .ok()?;
        if lo > hi || hi > count {
            return None;
        }
        let ids_at = fan_at.checked_add(PACK_INDEX_FANOUT_LEN)?;
        let mut left = lo;
        let mut right = hi;
        while left < right {
            let middle = left.checked_add(right.checked_sub(left)? / 2)?;
            let at = ids_at.checked_add(middle.checked_mul(16)?)?;
            let end = at.checked_add(16)?;
            match self.bytes.get(at..end)?.cmp(logical_id) {
                std::cmp::Ordering::Less => left = middle.checked_add(1)?,
                std::cmp::Ordering::Greater => right = middle,
                std::cmp::Ordering::Equal => {
                    let offset_table_at = ids_at.checked_add(count.checked_mul(16)?)?;
                    let offset_entry_at = offset_table_at.checked_add(middle.checked_mul(4)?)?;
                    return read_u32(&self.bytes, offset_entry_at);
                }
            }
        }
        None
    }

    /// Atomically replace `path` with this index, syncing the file and parent directory.
    ///
    /// # Errors
    /// Returns an I/O error if writing, syncing, renaming, or syncing the
    /// parent directory fails. If the directory sync fails after rename, the
    /// new file may already be visible; a temp file may remain if cleanup
    /// itself fails.
    pub fn write_atomic(&self, path: &Path) -> io::Result<()> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut temp_path = PathBuf::from(parent);
        temp_path.push(format!(".{name}.{}.{}.tmp", std::process::id(), counter));
        let mut created_temp = false;
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            created_temp = true;
            file.write_all(&self.bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp_path, path)?;
            File::open(parent)?.sync_all()
        })();
        if result.is_err() && created_temp {
            let _ = fs::remove_file(&temp_path);
        }
        result
    }
}

fn validate_section(bytes: &[u8], section_at: usize, count: u32) -> Option<()> {
    let count = usize::try_from(count).ok()?;
    let mut previous_fan = 0_u32;
    for byte in 0..256_usize {
        let value = read_u32(bytes, section_at.checked_add(byte.checked_mul(4)?)?)?;
        if value < previous_fan || usize::try_from(value).ok()? > count {
            return None;
        }
        previous_fan = value;
    }
    if usize::try_from(previous_fan).ok()? != count {
        return None;
    }
    let ids_at = section_at.checked_add(PACK_INDEX_FANOUT_LEN)?;
    let ids_len = count.checked_mul(16)?;
    let ids_end = ids_at.checked_add(ids_len)?;
    let ids = bytes.get(ids_at..ids_end)?;
    let mut previous: Option<[u8; 16]> = None;
    let mut observed_fan = [0_u32; 256];
    for id_bytes in ids.chunks_exact(16) {
        let id: [u8; 16] = id_bytes.try_into().ok()?;
        if previous.is_some_and(|prev| prev >= id) {
            return None;
        }
        previous = Some(id);
        let frequency = &mut observed_fan[usize::from(id[0])];
        *frequency = frequency.checked_add(1)?;
    }
    let mut cumulative = 0_u32;
    for (byte, frequency) in observed_fan.into_iter().enumerate() {
        cumulative = cumulative.checked_add(frequency)?;
        if read_u32(bytes, section_at.checked_add(byte.checked_mul(4)?)?)? != cumulative {
            return None;
        }
    }
    let offsets_at = ids_end;
    for i in 0..count {
        let offset = read_u32(bytes, offsets_at.checked_add(i.checked_mul(4)?)?)?;
        if u64::from(offset) >= read_u64(bytes, 44)? {
            return None;
        }
    }
    Some(())
}

fn crc_with_zeroed_field(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < CRC_RANGE.end {
        return None;
    }
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes.get(..CRC_RANGE.start)?);
    hasher.update(&[0; 4]);
    hasher.update(bytes.get(CRC_RANGE.end..)?);
    Some(hasher.finalize())
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    Some(u32::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    Some(u64::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_packidx.rs"]
mod tests;
