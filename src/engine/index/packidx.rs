//! Immutable, per-pack sorted locator-index format.
//!
//! Packs remain authoritative. The sidecar stores
//! `(collection_id, logical_id) -> byte_offset` and can be rebuilt by scanning
//! its pack. This module defines the format and I/O primitives; pack sealing
//! and store-open integration are separate work.
//!
//! The `index_artifacts` benchmark writes frames in arrival order; it does not
//! measure repacked packs. Repacking rewrites a collection's live frames in
//! topological order ([`crate::packfile::storage::PackfileStorage::repack_collection_reachable`]
//! copies in `topo_order()`). Neither order sorts by logical id, so sparse
//! sampling cannot scan between sampled ids in these layouts. Supporting that
//! would require an id-ordered write path and would trade away the locality
//! repack creates. Truncated ids (drop the trailing bytes, then verify the full
//! id from the frame header and probe adjacent equal prefixes) project to about
//! 40% less sidecar data, before accounting for any additional lookup metadata;
//! this format has no production mmap loader yet. Today only tests and the
//! `index_artifacts` bench call [`PackIndex::open_mmap`].
//!
//! Version 1 is little-endian:
//!
//! ```text
//! [64-byte header]
//! [collection directory: 20 bytes per collection]
//! [collection sections in collection-id order]
//! ```
//!
//! The `MTXIDX01` magic identifies the version. The header CRC field is zeroed
//! while calculating CRC32 over the complete file. Each directory entry is a
//! 16-byte collection id followed by a `u32` entry count. Its section starts
//! immediately after the previous section and contains cumulative
//! `fanout[256]`, sorted 16-byte logical ids, parallel `u32` offsets, then one
//! flags byte per entry. Bit 0 marks a removal version; the offset still points
//! at the removal frame, so append order decides which version wins. Duplicate
//! `(collection_id, logical_id)` inputs keep the greatest offset.
//!
//! [`PackIndex::build`] sorts one flat input vector and is intended for
//! in-memory construction. It does not implement external sorting; a bounded
//! memory builder must merge sorted spill runs before encoding.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;

use crate::packfile::{PackId, HEADER_LEN};

/// Magic identifying the v1 sorted locator-index format.
pub const PACK_INDEX_MAGIC: [u8; 8] = *b"MTXIDX01";
/// Fixed header length in bytes.
pub const PACK_INDEX_HEADER_LEN: usize = 64;
/// One collection directory entry in bytes.
pub const PACK_INDEX_COLLECTION_ENTRY_LEN: usize = 20;
/// Cumulative first-byte fanout table length in bytes.
pub const PACK_INDEX_FANOUT_LEN: usize = 256 * 4;
/// Bytes per locator in a collection section: a 16-byte id and a `u32` offset.
pub const PACK_INDEX_SECTION_ENTRY_LEN: usize = 20;
/// Bytes of per-entry flags in a collection section.
pub const PACK_INDEX_FLAG_ENTRY_LEN: usize = 1;
/// Flag bit marking a removal version.
pub const PACK_INDEX_FLAG_REMOVED: u8 = 1;
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
    /// Whether the frame at `offset` is a removal version.
    pub removed: bool,
}

/// What a locator names: a live frame or a removal version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackIndexLocator {
    /// A live frame at this byte offset.
    Frame(u32),
    /// A removal version at this byte offset. The frame is present, but the key
    /// is absent from the collection.
    Removed(u32),
}

/// Expected identity and extent when opening an index for a pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackIndexIdentity {
    /// Stable pack identity.
    pub pack_id: PackId,
    /// Pack creation sequence from its header.
    pub creation_seq: u64,
    /// Current pack file length.
    pub pack_file_len: u64,
}

/// An owned, fully verified pack-index image.
#[derive(Debug, Clone)]
pub struct PackIndex {
    bytes: Vec<u8>,
    collections: Vec<CollectionView>,
    identity: PackIndexIdentity,
    entry_count: u64,
}

/// A memory-mapped index opened without reading or checksumming its whole body.
///
/// Fast open validates header metadata, file extent, and collection-directory
/// bounds. It trusts the immutable sidecar's body; call [`Self::verify`] when a
/// complete CRC, ordering, fanout, and offset validation is needed. The caller
/// decides when this trust is warranted, for example after validating a durable
/// pack-seal manifest or by periodically calling `verify`.
///
/// The mapped file must never be modified or truncated in place while this
/// value exists. Publish replacements with [`PackIndex::write_atomic`], which
/// renames a completed temporary file over the path; do not reopen the mapped
/// path for writing.
#[derive(Debug)]
pub struct MappedPackIndex {
    map: Mmap,
    collections: Vec<CollectionView>,
    identity: PackIndexIdentity,
    entry_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct CollectionView {
    id: [u8; 16],
    count: u32,
    fan_at: usize,
    ids_at: usize,
    offsets_at: usize,
    flags_at: usize,
}

#[derive(Clone, Copy)]
struct CollectionRange {
    id: [u8; 16],
    start: usize,
    end: usize,
}

/// Errors found while constructing or fully validating a pack index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackIndexError {
    /// The encoding is truncated, malformed, unsorted, or internally inconsistent.
    Invalid,
    /// A pack offset or encoded file section exceeds the v1 `u32` limit.
    TooLarge,
    /// An offset points into the reserved pack header instead of frame data.
    OffsetBeforePackData {
        /// The invalid frame offset.
        offset: u64,
    },
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
            Self::OffsetBeforePackData { offset } => {
                write!(f, "pack offset {offset} is before frame data")
            }
            Self::OffsetOutsidePack { offset, pack_len } => {
                write!(f, "pack offset {offset} is outside pack length {pack_len}")
            }
        }
    }
}

impl std::error::Error for PackIndexError {}

/// Fast-open failures distinguish invalid/stale indexes from operating-system errors.
#[derive(Debug)]
pub enum PackIndexOpenError {
    /// File open, metadata, or mapping failed.
    Io(io::Error),
    /// The sidecar is malformed or does not match the requested pack.
    InvalidOrStale,
}

impl std::fmt::Display for PackIndexOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::InvalidOrStale => f.write_str("pack index is invalid or stale"),
        }
    }
}

impl std::error::Error for PackIndexOpenError {}

impl From<io::Error> for PackIndexOpenError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl PackIndex {
    /// Build a canonical v1 index, collapsing duplicate ids to their greatest offset.
    ///
    /// `inputs` is consumed so sorting and duplicate removal reuse its
    /// allocation. For inputs larger than RAM, externally merge sorted runs
    /// before calling the eventual streaming encoder.
    ///
    /// # Errors
    /// Returns an error if a pack id is zero, an offset is outside frame data
    /// or the pack extent, or v1 cannot represent the output.
    pub fn build(
        identity: PackIndexIdentity,
        mut inputs: Vec<PackIndexInput>,
    ) -> Result<Self, PackIndexError> {
        if identity.pack_id.is_zero() {
            return Err(PackIndexError::Invalid);
        }
        let minimum_offset = u64::try_from(HEADER_LEN).map_err(|_| PackIndexError::TooLarge)?;
        for input in &inputs {
            if input.offset < minimum_offset {
                return Err(PackIndexError::OffsetBeforePackData {
                    offset: input.offset,
                });
            }
            if input.offset >= identity.pack_file_len {
                return Err(PackIndexError::OffsetOutsidePack {
                    offset: input.offset,
                    pack_len: identity.pack_file_len,
                });
            }
            u32::try_from(input.offset).map_err(|_| PackIndexError::TooLarge)?;
        }
        inputs.sort_unstable_by(|left, right| {
            left.collection_id
                .cmp(&right.collection_id)
                .then_with(|| left.logical_id.cmp(&right.logical_id))
                .then_with(|| right.offset.cmp(&left.offset))
        });
        inputs.dedup_by(|later, earlier| {
            later.collection_id == earlier.collection_id && later.logical_id == earlier.logical_id
        });
        Self::encode_sorted(identity, &inputs)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_sorted(
        identity: PackIndexIdentity,
        inputs: &[PackIndexInput],
    ) -> Result<Self, PackIndexError> {
        let groups = group_collections(inputs)?;
        let collection_count = u32::try_from(groups.len()).map_err(|_| PackIndexError::TooLarge)?;
        let entry_count = u64::try_from(inputs.len()).map_err(|_| PackIndexError::TooLarge)?;
        let file_len = encoded_file_len(&groups)?;

        let mut bytes = Vec::with_capacity(file_len);
        encode_header(
            &mut bytes,
            identity,
            entry_count,
            collection_count,
            file_len,
        )?;
        encode_directory(&mut bytes, &groups)?;

        let mut collections = Vec::with_capacity(groups.len());
        for group in groups {
            collections.push(encode_collection(&mut bytes, inputs, group)?);
        }
        let crc = crc_with_zeroed_field(&bytes).ok_or(PackIndexError::Invalid)?;
        bytes[CRC_RANGE].copy_from_slice(&crc.to_le_bytes());
        Ok(Self {
            bytes,
            collections,
            identity,
            entry_count,
        })
    }

    /// Parse and fully validate a v1 pack index, including its CRC and entries.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let parsed = parse_layout(bytes, true)?;
        Some(Self {
            bytes: bytes.to_vec(),
            collections: parsed.collections,
            identity: parsed.identity,
            entry_count: parsed.entry_count,
        })
    }

    /// Map an index after checking its header and directory against the expected pack.
    ///
    /// This does not read the full file or verify its CRC. Call [`MappedPackIndex::verify`]
    /// when complete body validation is required. A failed or stale sidecar is
    /// disposable and callers should rebuild it from the pack.
    ///
    /// # Errors
    /// Returns [`PackIndexOpenError::InvalidOrStale`] when the file's metadata
    /// does not match `expected` or its header/directory cannot describe the
    /// mapped extent; operating-system errors are returned as `Io`.
    pub fn open_mmap(
        path: &Path,
        expected: PackIndexIdentity,
    ) -> Result<MappedPackIndex, PackIndexOpenError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let actual_len = metadata.len();
        if !(64..=MAX_FILE_LEN).contains(&actual_len) || usize::try_from(actual_len).is_err() {
            return Err(PackIndexOpenError::InvalidOrStale);
        }
        // SAFETY: pack indexes are immutable after atomic publication. The
        // caller must not truncate or mutate this file while the mapping lives.
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&file)? };
        if u64::try_from(map.len()).ok() != Some(actual_len) {
            return Err(PackIndexOpenError::InvalidOrStale);
        }
        let parsed = parse_layout(&map, false).ok_or(PackIndexOpenError::InvalidOrStale)?;
        if parsed.identity != expected {
            return Err(PackIndexOpenError::InvalidOrStale);
        }
        Ok(MappedPackIndex {
            map,
            collections: parsed.collections,
            identity: parsed.identity,
            entry_count: parsed.entry_count,
        })
    }

    /// The pack identity and extent recorded in the header.
    #[must_use]
    pub const fn identity(&self) -> PackIndexIdentity {
        self.identity
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

    /// Resolve a key to a live frame offset in this pack.
    ///
    /// Returns `None` for a missing key and for a key whose newest version is a
    /// removal. Use [`Self::get_locator`] when the difference matters.
    #[must_use]
    pub fn get(&self, collection_id: &[u8; 16], logical_id: &[u8; 16]) -> Option<u32> {
        match lookup(&self.bytes, &self.collections, collection_id, logical_id)? {
            PackIndexLocator::Frame(offset) => Some(offset),
            PackIndexLocator::Removed(_) => None,
        }
    }

    /// Resolve a key to the newest version recorded in this pack, including a
    /// removal version.
    #[must_use]
    pub fn get_locator(
        &self,
        collection_id: &[u8; 16],
        logical_id: &[u8; 16],
    ) -> Option<PackIndexLocator> {
        lookup(&self.bytes, &self.collections, collection_id, logical_id)
    }

    /// Atomically publish this index, syncing the file and parent directory where supported.
    ///
    /// Treat any error as an absent/disposable index: rename may have completed
    /// before a parent-directory sync failed, but the installed file is always
    /// the complete, checksummed encoding. A caller may ignore it and rebuild.
    ///
    /// # Errors
    /// Returns an I/O error if creating/writing/syncing the temporary file or
    /// renaming it fails. On Unix, a parent-directory sync error may be
    /// returned after the new file is visible.
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
            sync_parent(parent)
        })();
        if result.is_err() && created_temp {
            let _ = fs::remove_file(&temp_path);
        }
        result
    }
}

fn group_collections(inputs: &[PackIndexInput]) -> Result<Vec<CollectionRange>, PackIndexError> {
    let mut groups = Vec::new();
    let mut cursor = 0_usize;
    while cursor < inputs.len() {
        let id = inputs[cursor].collection_id;
        let start = cursor;
        while cursor < inputs.len() && inputs[cursor].collection_id == id {
            cursor = cursor.checked_add(1).ok_or(PackIndexError::TooLarge)?;
        }
        groups.push(CollectionRange {
            id,
            start,
            end: cursor,
        });
    }
    Ok(groups)
}

fn encoded_file_len(groups: &[CollectionRange]) -> Result<usize, PackIndexError> {
    let directory_len = groups
        .len()
        .checked_mul(PACK_INDEX_COLLECTION_ENTRY_LEN)
        .ok_or(PackIndexError::TooLarge)?;
    let sections_len = groups
        .iter()
        .try_fold(0_usize, |total, group| {
            let count = group.end.checked_sub(group.start)?;
            let section_len = PACK_INDEX_FANOUT_LEN
                .checked_add(count.checked_mul(PACK_INDEX_SECTION_ENTRY_LEN)?)?
                .checked_add(count.checked_mul(PACK_INDEX_FLAG_ENTRY_LEN)?)?;
            total.checked_add(section_len)
        })
        .ok_or(PackIndexError::TooLarge)?;
    let file_len = PACK_INDEX_HEADER_LEN
        .checked_add(directory_len)
        .and_then(|len| len.checked_add(sections_len))
        .ok_or(PackIndexError::TooLarge)?;
    if u64::try_from(file_len).map_err(|_| PackIndexError::TooLarge)? > MAX_FILE_LEN {
        return Err(PackIndexError::TooLarge);
    }
    Ok(file_len)
}

fn encode_header(
    bytes: &mut Vec<u8>,
    identity: PackIndexIdentity,
    entry_count: u64,
    collection_count: u32,
    file_len: usize,
) -> Result<(), PackIndexError> {
    bytes.extend_from_slice(&PACK_INDEX_MAGIC);
    bytes.extend_from_slice(identity.pack_id.as_bytes());
    bytes.extend_from_slice(&identity.creation_seq.to_le_bytes());
    bytes.extend_from_slice(&entry_count.to_le_bytes());
    bytes.extend_from_slice(&collection_count.to_le_bytes());
    bytes.extend_from_slice(&identity.pack_file_len.to_le_bytes());
    bytes.extend_from_slice(
        &u64::try_from(file_len)
            .map_err(|_| PackIndexError::TooLarge)?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    debug_assert_eq!(
        bytes.len(),
        PACK_INDEX_HEADER_LEN,
        "header fields must exactly fill the fixed header"
    );
    Ok(())
}

fn encode_directory(bytes: &mut Vec<u8>, groups: &[CollectionRange]) -> Result<(), PackIndexError> {
    for group in groups {
        let count = group
            .end
            .checked_sub(group.start)
            .ok_or(PackIndexError::Invalid)?;
        bytes.extend_from_slice(&group.id);
        bytes.extend_from_slice(
            &u32::try_from(count)
                .map_err(|_| PackIndexError::TooLarge)?
                .to_le_bytes(),
        );
    }
    Ok(())
}

fn encode_collection(
    bytes: &mut Vec<u8>,
    inputs: &[PackIndexInput],
    group: CollectionRange,
) -> Result<CollectionView, PackIndexError> {
    let count = group
        .end
        .checked_sub(group.start)
        .ok_or(PackIndexError::Invalid)?;
    let count_u32 = u32::try_from(count).map_err(|_| PackIndexError::TooLarge)?;
    let fan_at = bytes.len();
    let mut frequencies = [0_u32; 256];
    for input in &inputs[group.start..group.end] {
        let first_byte = usize::from(input.logical_id[0]);
        frequencies[first_byte] = frequencies[first_byte]
            .checked_add(1)
            .ok_or(PackIndexError::TooLarge)?;
    }
    let mut cumulative = 0_u32;
    for frequency in frequencies {
        cumulative = cumulative
            .checked_add(frequency)
            .ok_or(PackIndexError::TooLarge)?;
        bytes.extend_from_slice(&cumulative.to_le_bytes());
    }
    let ids_at = bytes.len();
    let expected_ids_at = fan_at
        .checked_add(PACK_INDEX_FANOUT_LEN)
        .ok_or(PackIndexError::TooLarge)?;
    for input in &inputs[group.start..group.end] {
        bytes.extend_from_slice(&input.logical_id);
    }
    let offsets_at = bytes.len();
    for input in &inputs[group.start..group.end] {
        bytes.extend_from_slice(
            &u32::try_from(input.offset)
                .map_err(|_| PackIndexError::TooLarge)?
                .to_le_bytes(),
        );
    }
    let flags_at = bytes.len();
    for input in &inputs[group.start..group.end] {
        bytes.push(if input.removed {
            PACK_INDEX_FLAG_REMOVED
        } else {
            0
        });
    }
    debug_assert_eq!(
        ids_at, expected_ids_at,
        "logical-id section must immediately follow the fanout table"
    );
    Ok(CollectionView {
        id: group.id,
        count: count_u32,
        fan_at,
        ids_at,
        offsets_at,
        flags_at,
    })
}

impl MappedPackIndex {
    /// Pack identity and extent recorded in the index header.
    #[must_use]
    pub const fn identity(&self) -> PackIndexIdentity {
        self.identity
    }

    /// Number of unique locators in the index.
    #[must_use]
    pub const fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Resolve a key to a live frame offset in this pack.
    ///
    /// Returns `None` for a missing key and for a key whose newest version is a
    /// removal. Use [`Self::get_locator`] when the difference matters.
    #[must_use]
    pub fn get(&self, collection_id: &[u8; 16], logical_id: &[u8; 16]) -> Option<u32> {
        match lookup(&self.map, &self.collections, collection_id, logical_id)? {
            PackIndexLocator::Frame(offset) => Some(offset),
            PackIndexLocator::Removed(_) => None,
        }
    }

    /// Resolve a key to the newest version recorded in this pack, including a
    /// removal version.
    #[must_use]
    pub fn get_locator(
        &self,
        collection_id: &[u8; 16],
        logical_id: &[u8; 16],
    ) -> Option<PackIndexLocator> {
        lookup(&self.map, &self.collections, collection_id, logical_id)
    }

    /// Tell the operating system that lookups will access the mapping randomly.
    ///
    /// This can reduce sequential readahead for scattered lookups. It is only
    /// a hint; the mapping remains correct if the operating system ignores it.
    ///
    /// # Errors
    /// Returns an I/O error if the operating system cannot apply the advice.
    #[cfg(unix)]
    pub fn advise_random(&self) -> io::Result<()> {
        self.map.advise(memmap2::Advice::Random)
    }

    /// Verify CRC, sorted ids, exact fanout values, and every frame offset.
    ///
    /// # Errors
    /// Returns [`PackIndexError::Invalid`] when any encoded byte or entry
    /// violates the v1 format.
    pub fn verify(&self) -> Result<(), PackIndexError> {
        parse_layout(&self.map, true)
            .filter(|parsed| parsed.identity == self.identity)
            .map(|_| ())
            .ok_or(PackIndexError::Invalid)
    }
}

fn lookup(
    bytes: &[u8],
    collections: &[CollectionView],
    collection_id: &[u8; 16],
    logical_id: &[u8; 16],
) -> Option<PackIndexLocator> {
    let collection = collections
        .binary_search_by_key(collection_id, |view| view.id)
        .ok()
        .and_then(|index| collections.get(index))?;
    let first = usize::from(logical_id[0]);
    let lo = if first == 0 {
        0
    } else {
        usize::try_from(read_fan(bytes, collection.fan_at, first.checked_sub(1)?)?).ok()?
    };
    let hi = usize::try_from(read_fan(bytes, collection.fan_at, first)?).ok()?;
    let count = usize::try_from(collection.count).ok()?;
    if lo > hi || hi > count {
        return None;
    }
    let mut left = lo;
    let mut right = hi;
    while left < right {
        let middle = left.checked_add(right.checked_sub(left)? / 2)?;
        let id_at = collection.ids_at.checked_add(middle.checked_mul(16)?)?;
        let id_end = id_at.checked_add(16)?;
        match bytes.get(id_at..id_end)?.cmp(logical_id) {
            std::cmp::Ordering::Less => left = middle.checked_add(1)?,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => {
                let offset_at = collection.offsets_at.checked_add(middle.checked_mul(4)?)?;
                let offset = read_u32(bytes, offset_at)?;
                let flags_at = collection.flags_at.checked_add(middle)?;
                let flags = *bytes.get(flags_at)?;
                return Some(if flags & PACK_INDEX_FLAG_REMOVED != 0 {
                    PackIndexLocator::Removed(offset)
                } else {
                    PackIndexLocator::Frame(offset)
                });
            }
        }
    }
    None
}

fn parse_layout(bytes: &[u8], verify_body: bool) -> Option<ParsedLayout> {
    if bytes.len() < PACK_INDEX_HEADER_LEN || bytes.get(..8)? != PACK_INDEX_MAGIC {
        return None;
    }
    let mut pack_bytes = [0_u8; 16];
    pack_bytes.copy_from_slice(bytes.get(8..24)?);
    let pack_id = PackId(pack_bytes);
    if pack_id.is_zero() {
        return None;
    }
    let identity = PackIndexIdentity {
        pack_id,
        creation_seq: read_u64(bytes, 24)?,
        pack_file_len: read_u64(bytes, 44)?,
    };
    let entry_count = read_u64(bytes, 32)?;
    let collection_count = usize::try_from(read_u32(bytes, 40)?).ok()?;
    let declared_len = usize::try_from(read_u64(bytes, 52)?).ok()?;
    if declared_len != bytes.len() || u64::try_from(bytes.len()).ok()? > MAX_FILE_LEN {
        return None;
    }
    if verify_body && read_u32(bytes, 60)? != crc_with_zeroed_field(bytes)? {
        return None;
    }
    let directory_len = collection_count.checked_mul(PACK_INDEX_COLLECTION_ENTRY_LEN)?;
    let mut section_at = PACK_INDEX_HEADER_LEN.checked_add(directory_len)?;
    if section_at > bytes.len() {
        return None;
    }
    let mut collections = Vec::new();
    collections.try_reserve(collection_count).ok()?;
    let mut previous_id: Option<[u8; 16]> = None;
    let mut total_entries = 0_u64;
    for i in 0..collection_count {
        let dir_at =
            PACK_INDEX_HEADER_LEN.checked_add(i.checked_mul(PACK_INDEX_COLLECTION_ENTRY_LEN)?)?;
        let id: [u8; 16] = bytes
            .get(dir_at..dir_at.checked_add(16)?)?
            .try_into()
            .ok()?;
        if previous_id.is_some_and(|previous| previous >= id) {
            return None;
        }
        previous_id = Some(id);
        let count = read_u32(bytes, dir_at.checked_add(16)?)?;
        if count == 0 {
            return None;
        }
        let count_usize = usize::try_from(count).ok()?;
        let fan_at = section_at;
        let ids_at = fan_at.checked_add(PACK_INDEX_FANOUT_LEN)?;
        let offsets_at = ids_at.checked_add(count_usize.checked_mul(16)?)?;
        let flags_at = offsets_at.checked_add(count_usize.checked_mul(4)?)?;
        let section_end =
            flags_at.checked_add(count_usize.checked_mul(PACK_INDEX_FLAG_ENTRY_LEN)?)?;
        if section_end > bytes.len() {
            return None;
        }
        total_entries = total_entries.checked_add(u64::from(count))?;
        collections.push(CollectionView {
            id,
            count,
            fan_at,
            ids_at,
            offsets_at,
            flags_at,
        });
        section_at = section_end;
    }
    if section_at != bytes.len() || total_entries != entry_count {
        return None;
    }
    let parsed = ParsedLayout {
        identity,
        entry_count,
        collections,
    };
    if verify_body {
        validate_body(bytes, &parsed)?;
    }
    Some(parsed)
}

struct ParsedLayout {
    identity: PackIndexIdentity,
    entry_count: u64,
    collections: Vec<CollectionView>,
}

fn validate_body(bytes: &[u8], parsed: &ParsedLayout) -> Option<()> {
    let min_offset = u64::try_from(HEADER_LEN).ok()?;
    for collection in &parsed.collections {
        let count = usize::try_from(collection.count).ok()?;
        let mut previous_fan = 0_u32;
        let mut observed_fan = [0_u32; 256];
        let mut previous_id: Option<[u8; 16]> = None;
        for first_byte in 0..256_usize {
            let value = read_fan(bytes, collection.fan_at, first_byte)?;
            if value < previous_fan || usize::try_from(value).ok()? > count {
                return None;
            }
            previous_fan = value;
        }
        if usize::try_from(previous_fan).ok()? != count {
            return None;
        }
        for i in 0..count {
            let id_at = collection.ids_at.checked_add(i.checked_mul(16)?)?;
            let id: [u8; 16] = bytes.get(id_at..id_at.checked_add(16)?)?.try_into().ok()?;
            if previous_id.is_some_and(|previous| previous >= id) {
                return None;
            }
            previous_id = Some(id);
            let frequency = &mut observed_fan[usize::from(id[0])];
            *frequency = frequency.checked_add(1)?;
            let offset_at = collection.offsets_at.checked_add(i.checked_mul(4)?)?;
            let offset = u64::from(read_u32(bytes, offset_at)?);
            let flags = *bytes.get(collection.flags_at.checked_add(i)?)?;
            if flags & !PACK_INDEX_FLAG_REMOVED != 0 {
                return None;
            }
            if offset < min_offset || offset >= parsed.identity.pack_file_len {
                return None;
            }
        }
        let mut cumulative = 0_u32;
        for (byte, frequency) in observed_fan.into_iter().enumerate() {
            cumulative = cumulative.checked_add(frequency)?;
            if read_fan(bytes, collection.fan_at, byte)? != cumulative {
                return None;
            }
        }
    }
    Some(())
}

fn read_fan(bytes: &[u8], fan_at: usize, byte: usize) -> Option<u32> {
    read_u32(bytes, fan_at.checked_add(byte.checked_mul(4)?)?)
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

#[cfg(unix)]
fn sync_parent(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg_attr(not(unix), allow(clippy::unnecessary_wraps))]
#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_packidx.rs"]
mod tests;
