//! Merged pack-range runs: a derived Paged lookup cache over sealed sidecars.
//!
//! A run holds the newest locator of every `(collection_id, logical_id)` found
//! in a contiguous range of a pool's sealed packs, so a Paged lookup can probe
//! one run instead of one sidecar per pack. Runs are disposable: sidecars stay
//! authoritative, and a missing, stale or corrupt run costs only lookup speed.
//! See `docs/2026-11-03-paged-merged-runs.md`.
//!
//! Version 1 is little-endian:
//!
//! ```text
//! [64-byte header]
//! [entry blocks, back to back]
//! [metadata: pack table, collection directory, block table]
//! ```
//!
//! The header holds the `MTXRUN01` magic, the version, the block size `B`,
//! the entry, pack, collection and block counts, the metadata offset and
//! length, and `meta_crc32` over the header (with that field zeroed) and the
//! metadata. The metadata sits after the blocks because a streaming merge
//! learns the collection and block counts only as it writes.
//!
//! - **Pack table:** `pack_count` entries of `(pack_id, creation_seq,
//!   pack_file_len)`, strictly increasing in `(creation_seq, pack_id)`.
//! - **Collection directory:** `(collection_id, entry_count, first_block,
//!   block_count)`, strictly increasing by id. A collection is never empty.
//! - **Block table:** `(fence, data_offset, entries, crc32)`, where `fence` is
//!   the block's first logical id. Blocks never span collections.
//! - **Entry block:** up to `B` sorted 16-byte logical ids, then one 7-byte
//!   locator per id: `u16` pack ordinal, `u32` frame offset, `u8` flags.
//!
//! A lookup binary-searches the collection directory, then that collection's
//! fences, then one block. There is no interpolation search: record ids are
//! attacker-influenced. Opening a run checks only the header, metadata CRC and
//! layout, which is O(packs + collections + blocks). A block's CRC and
//! contents are checked on its first read and remembered in a bitmap; a
//! failure marks the whole run invalid. Lookups report that as
//! [`RunCorrupt`], never as a miss, so the caller falls back to sidecars.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use memmap2::Mmap;

use crate::packfile::{PackId, HEADER_LEN};

use super::packidx::{MappedPackIndex, PackIndexIdentity, PackIndexLocator};

/// Magic identifying the v1 run format.
pub const RUN_MAGIC: [u8; 8] = *b"MTXRUN01";
/// Format version stored after the magic.
pub const RUN_VERSION: u32 = 1;
/// Fixed header length in bytes.
pub const RUN_HEADER_LEN: usize = 64;
/// Default entries per block.
pub const RUN_DEFAULT_BLOCK_ENTRIES: u32 = 256;
/// Most packs one run can cover: pack ordinals are `u16`.
pub const RUN_MAX_PACKS: usize = 1 << 16;
/// Flag bit marking a removal version, as in sidecars.
pub const RUN_FLAG_REMOVED: u8 = 1;

const ID_LEN: usize = 16;
const LOCATOR_LEN: usize = 7;
const ENTRY_LEN: usize = ID_LEN + LOCATOR_LEN;
const PACK_ENTRY_LEN: usize = 32;
const COLLECTION_ENTRY_LEN: usize = 32;
const BLOCK_ENTRY_LEN: usize = 32;
const CRC_AT: usize = 56;

/// A run failed a check after open. The run must not answer lookups; the
/// caller probes the covered packs' sidecars instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunCorrupt;

impl std::fmt::Display for RunCorrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("run block failed verification")
    }
}

impl std::error::Error for RunCorrupt {}

/// Errors from writing, opening or merging runs.
#[derive(Debug)]
pub enum RunError {
    /// An operating-system error.
    Io(io::Error),
    /// The file is not a valid v1 run: bad magic, version, CRC or layout.
    Invalid,
    /// An input run failed block verification during a merge.
    Corrupt(RunCorrupt),
    /// Entries were not supplied in strictly increasing `(collection, id)`
    /// order.
    OutOfOrder,
    /// A locator names a pack ordinal outside the pack table, or an offset
    /// outside that pack.
    BadLocator,
    /// The pack table is empty, too large, unsorted or has duplicates.
    BadPackTable,
    /// Merge inputs name the same pack twice or interleave in pack order.
    OverlappingInputs,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "run I/O error: {error}"),
            Self::Invalid => f.write_str("invalid run file"),
            Self::Corrupt(error) => write!(f, "corrupt merge input: {error}"),
            Self::OutOfOrder => f.write_str("run entries out of order"),
            Self::BadLocator => f.write_str("run locator outside its pack"),
            Self::BadPackTable => f.write_str("invalid run pack table"),
            Self::OverlappingInputs => f.write_str("run merge inputs overlap"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<io::Error> for RunError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<RunCorrupt> for RunError {
    fn from(error: RunCorrupt) -> Self {
        Self::Corrupt(error)
    }
}

/// A run lookup result: the pack that holds the newest version, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunHit {
    /// Index into the run's pack table.
    pub ordinal: u16,
    /// Identity of that pack.
    pub pack: PackIndexIdentity,
    /// The newest version's frame, live or removal.
    pub locator: PackIndexLocator,
}

/// One merged entry, in the order a run stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunEntry {
    /// Collection of the record.
    pub collection_id: [u8; 16],
    /// The record's logical id.
    pub logical_id: [u8; 16],
    /// Index into the run's pack table.
    pub ordinal: u16,
    /// The frame, live or removal.
    pub locator: PackIndexLocator,
}

/// Shape of a written run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunSummary {
    /// Unique `(collection, id)` entries.
    pub entry_count: u64,
    /// Collections with at least one entry.
    pub collection_count: u32,
    /// Entry blocks.
    pub block_count: u32,
    /// Bytes of the run file.
    pub file_len: u64,
}

fn pack_order(pack: &PackIndexIdentity) -> (u64, [u8; 16]) {
    (pack.creation_seq, pack.pack_id.0)
}

fn locator_parts(locator: PackIndexLocator) -> (u32, u8) {
    match locator {
        PackIndexLocator::Frame(offset) => (offset, 0),
        PackIndexLocator::Removed(offset) => (offset, RUN_FLAG_REMOVED),
    }
}

fn is_removed(locator: PackIndexLocator) -> bool {
    matches!(locator, PackIndexLocator::Removed(_))
}

/// Whether `offset` can name a frame inside `pack`.
fn offset_in_pack(offset: u32, pack: &PackIndexIdentity) -> bool {
    let offset = u64::from(offset);
    u64::try_from(HEADER_LEN).is_ok_and(|min| offset >= min) && offset < pack.pack_file_len
}

fn validate_pack_table(packs: &[PackIndexIdentity]) -> Result<(), RunError> {
    if packs.is_empty() || packs.len() > RUN_MAX_PACKS {
        return Err(RunError::BadPackTable);
    }
    if packs.iter().any(|pack| pack.pack_id.is_zero()) {
        return Err(RunError::BadPackTable);
    }
    if packs
        .windows(2)
        .any(|pair| pack_order(&pair[0]) >= pack_order(&pair[1]))
    {
        return Err(RunError::BadPackTable);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct CollectionMeta {
    id: [u8; 16],
    entry_count: u64,
    first_block: u32,
    block_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    fence: [u8; 16],
    data_offset: u64,
    entries: u32,
    crc: u32,
}

/// Streams sorted entries into a run file.
///
/// Memory is one block of entries plus the metadata: 32 bytes per block and
/// per collection, so O(entries / B) rather than O(entries).
#[derive(Debug)]
pub struct RunWriter {
    file: BufWriter<File>,
    path: PathBuf,
    packs: Vec<PackIndexIdentity>,
    block_entries: u32,
    collections: Vec<CollectionMeta>,
    blocks: Vec<BlockMeta>,
    pending_ids: Vec<u8>,
    pending_locators: Vec<u8>,
    pending_count: u32,
    pending_fence: [u8; 16],
    data_end: u64,
    last: Option<([u8; 16], [u8; 16])>,
    entry_count: u64,
}

impl RunWriter {
    /// Create `path` (truncating it) for a run over `packs`, which must be
    /// strictly increasing in `(creation_seq, pack_id)`.
    ///
    /// # Errors
    /// Returns [`RunError::BadPackTable`] for an invalid pack table, an error
    /// for a zero `block_entries`, or an I/O error.
    pub fn create(
        path: &Path,
        packs: Vec<PackIndexIdentity>,
        block_entries: u32,
    ) -> Result<Self, RunError> {
        validate_pack_table(&packs)?;
        if block_entries == 0 {
            return Err(RunError::Invalid);
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut file = BufWriter::new(file);
        file.write_all(&[0; RUN_HEADER_LEN])?;
        let capacity = usize::try_from(block_entries).unwrap_or(0);
        Ok(Self {
            file,
            path: path.to_path_buf(),
            packs,
            block_entries,
            collections: Vec::new(),
            blocks: Vec::new(),
            pending_ids: Vec::with_capacity(capacity.saturating_mul(ID_LEN)),
            pending_locators: Vec::with_capacity(capacity.saturating_mul(LOCATOR_LEN)),
            pending_count: 0,
            pending_fence: [0; 16],
            data_end: u64::try_from(RUN_HEADER_LEN).unwrap_or(u64::MAX),
            last: None,
            entry_count: 0,
        })
    }

    /// Append one entry. Entries must arrive in strictly increasing
    /// `(collection_id, logical_id)` order.
    ///
    /// # Errors
    /// Returns [`RunError::OutOfOrder`] or [`RunError::BadLocator`] for input
    /// the reader would reject, or an I/O error.
    pub fn push(&mut self, entry: RunEntry) -> Result<(), RunError> {
        let key = (entry.collection_id, entry.logical_id);
        if self.last.is_some_and(|last| last >= key) {
            return Err(RunError::OutOfOrder);
        }
        let pack = self
            .packs
            .get(usize::from(entry.ordinal))
            .ok_or(RunError::BadLocator)?;
        let (offset, flags) = locator_parts(entry.locator);
        if !offset_in_pack(offset, pack) {
            return Err(RunError::BadLocator);
        }
        let new_collection = self
            .last
            .is_none_or(|(collection, _)| collection != entry.collection_id);
        if new_collection {
            self.flush_block()?;
            let first_block = u32::try_from(self.blocks.len()).map_err(|_| RunError::Invalid)?;
            self.collections.push(CollectionMeta {
                id: entry.collection_id,
                entry_count: 0,
                first_block,
                block_count: 0,
            });
        }
        if self.pending_count == 0 {
            self.pending_fence = entry.logical_id;
        }
        self.pending_ids.extend_from_slice(&entry.logical_id);
        self.pending_locators
            .extend_from_slice(&entry.ordinal.to_le_bytes());
        self.pending_locators
            .extend_from_slice(&offset.to_le_bytes());
        self.pending_locators.push(flags);
        self.pending_count = self.pending_count.checked_add(1).ok_or(RunError::Invalid)?;
        self.last = Some(key);
        if self.pending_count >= self.block_entries {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> Result<(), RunError> {
        if self.pending_count == 0 {
            return Ok(());
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.pending_ids);
        hasher.update(&self.pending_locators);
        self.file.write_all(&self.pending_ids)?;
        self.file.write_all(&self.pending_locators)?;
        self.blocks.push(BlockMeta {
            fence: self.pending_fence,
            data_offset: self.data_end,
            entries: self.pending_count,
            crc: hasher.finalize(),
        });
        let written = u64::from(self.pending_count)
            .checked_mul(u64::try_from(ENTRY_LEN).map_err(|_| RunError::Invalid)?)
            .ok_or(RunError::Invalid)?;
        self.data_end = self
            .data_end
            .checked_add(written)
            .ok_or(RunError::Invalid)?;
        self.entry_count = self
            .entry_count
            .checked_add(u64::from(self.pending_count))
            .ok_or(RunError::Invalid)?;
        let collection = self.collections.last_mut().ok_or(RunError::Invalid)?;
        collection.entry_count = collection
            .entry_count
            .checked_add(u64::from(self.pending_count))
            .ok_or(RunError::Invalid)?;
        collection.block_count = collection
            .block_count
            .checked_add(1)
            .ok_or(RunError::Invalid)?;
        self.pending_ids.clear();
        self.pending_locators.clear();
        self.pending_count = 0;
        Ok(())
    }

    /// Write the metadata and header and fsync the file. A run with no
    /// entries is not written: the file is removed and `None` returned.
    ///
    /// # Errors
    /// Returns an I/O error, or [`RunError::Invalid`] if a count overflows the
    /// format.
    pub fn finish(mut self) -> Result<Option<RunSummary>, RunError> {
        self.flush_block()?;
        if self.entry_count == 0 {
            drop(self.file);
            fs::remove_file(&self.path)?;
            return Ok(None);
        }
        let pack_count = u32::try_from(self.packs.len()).map_err(|_| RunError::Invalid)?;
        let collection_count =
            u32::try_from(self.collections.len()).map_err(|_| RunError::Invalid)?;
        let block_count = u32::try_from(self.blocks.len()).map_err(|_| RunError::Invalid)?;
        let mut meta = Vec::new();
        for pack in &self.packs {
            meta.extend_from_slice(&pack.pack_id.0);
            meta.extend_from_slice(&pack.creation_seq.to_le_bytes());
            meta.extend_from_slice(&pack.pack_file_len.to_le_bytes());
        }
        for collection in &self.collections {
            meta.extend_from_slice(&collection.id);
            meta.extend_from_slice(&collection.entry_count.to_le_bytes());
            meta.extend_from_slice(&collection.first_block.to_le_bytes());
            meta.extend_from_slice(&collection.block_count.to_le_bytes());
        }
        for block in &self.blocks {
            meta.extend_from_slice(&block.fence);
            meta.extend_from_slice(&block.data_offset.to_le_bytes());
            meta.extend_from_slice(&block.entries.to_le_bytes());
            meta.extend_from_slice(&block.crc.to_le_bytes());
        }
        let meta_len = u64::try_from(meta.len()).map_err(|_| RunError::Invalid)?;
        let mut header = [0_u8; RUN_HEADER_LEN];
        header[0..8].copy_from_slice(&RUN_MAGIC);
        header[8..12].copy_from_slice(&RUN_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&self.block_entries.to_le_bytes());
        header[16..24].copy_from_slice(&self.entry_count.to_le_bytes());
        header[24..28].copy_from_slice(&pack_count.to_le_bytes());
        header[28..32].copy_from_slice(&collection_count.to_le_bytes());
        header[32..36].copy_from_slice(&block_count.to_le_bytes());
        header[40..48].copy_from_slice(&self.data_end.to_le_bytes());
        header[48..56].copy_from_slice(&meta_len.to_le_bytes());
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        hasher.update(&meta);
        header[CRC_AT..CRC_AT + 4].copy_from_slice(&hasher.finalize().to_le_bytes());
        self.file.write_all(&meta)?;
        self.file.flush()?;
        let mut file = self
            .file
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.sync_all()?;
        Ok(Some(RunSummary {
            entry_count: self.entry_count,
            collection_count,
            block_count,
            file_len: self
                .data_end
                .checked_add(meta_len)
                .ok_or(RunError::Invalid)?,
        }))
    }
}

/// A memory-mapped run, opened after checking only its header, metadata CRC
/// and layout.
///
/// The mapped file must never be modified in place while this value exists;
/// runs are published by renaming a completed temporary file.
#[derive(Debug)]
pub struct RunIndex {
    map: Mmap,
    packs: Vec<PackIndexIdentity>,
    collections: Vec<CollectionMeta>,
    block_entries: u32,
    block_table_at: usize,
    entry_count: u64,
    verified: Vec<AtomicU64>,
    invalid: AtomicBool,
}

impl RunIndex {
    /// Open and check a run. The caller still has to confirm that every pack
    /// in [`Self::packs`] exists with that identity before trusting it.
    ///
    /// # Errors
    /// Returns [`RunError::Invalid`] if the header, metadata CRC or layout is
    /// wrong, or an I/O error.
    pub fn open(path: &Path) -> Result<Self, RunError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len < u64::try_from(RUN_HEADER_LEN).map_err(|_| RunError::Invalid)? {
            return Err(RunError::Invalid);
        }
        // SAFETY: runs are immutable after atomic publication by rename; this
        // module never opens a published run for writing, and the caller must
        // not truncate or mutate the file while the mapping lives.
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&file)? };
        if u64::try_from(map.len()).ok() != Some(len) {
            return Err(RunError::Invalid);
        }
        Self::from_map(map).ok_or(RunError::Invalid)
    }

    fn from_map(map: Mmap) -> Option<Self> {
        let header = RunHeader::parse(&map)?;
        let packs = header.parse_packs(&map)?;
        let collections = header.parse_collections(&map)?;
        let words = usize::try_from(header.block_count).ok()?.div_ceil(64);
        let verified = (0..words).map(|_| AtomicU64::new(0)).collect();
        Some(Self {
            map,
            packs,
            collections,
            block_entries: header.block_entries,
            block_table_at: header.block_table_at()?,
            entry_count: header.entry_count,
            verified,
            invalid: AtomicBool::new(false),
        })
    }

    /// The covered packs, oldest first. Entries name packs by index here.
    #[must_use]
    pub fn packs(&self) -> &[PackIndexIdentity] {
        &self.packs
    }

    /// Unique `(collection, id)` entries in the run.
    #[must_use]
    pub const fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Entries per block, as recorded in the header.
    #[must_use]
    pub const fn block_entries(&self) -> u32 {
        self.block_entries
    }

    /// Bytes of the mapped run file.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        u64::try_from(self.map.len()).unwrap_or(u64::MAX)
    }

    /// Whether a block check has failed. An invalid run answers every lookup
    /// with [`RunCorrupt`].
    #[must_use]
    pub fn is_invalid(&self) -> bool {
        self.invalid.load(Ordering::Acquire)
    }

    /// Entries recorded for one collection, from the verified directory.
    #[must_use]
    pub fn collection_entry_count(&self, collection_id: &[u8; 16]) -> Option<u64> {
        self.collection(collection_id).map(|meta| meta.entry_count)
    }

    fn collection(&self, collection_id: &[u8; 16]) -> Option<&CollectionMeta> {
        self.collections
            .binary_search_by_key(collection_id, |meta| meta.id)
            .ok()
            .and_then(|index| self.collections.get(index))
    }

    /// The newest version of `logical_id` among the covered packs.
    ///
    /// `Ok(None)` means none of the covered packs holds the key.
    ///
    /// # Errors
    /// Returns [`RunCorrupt`] if the run is invalid or the block that would
    /// hold the key fails its check. That is never a miss.
    pub fn lookup(
        &self,
        collection_id: &[u8; 16],
        logical_id: &[u8; 16],
    ) -> Result<Option<RunHit>, RunCorrupt> {
        if self.is_invalid() {
            return Err(RunCorrupt);
        }
        let Some(collection) = self.collection(collection_id) else {
            return Ok(None);
        };
        // The last block whose fence is at or below the id.
        let mut low = 0_u32;
        let mut high = collection.block_count;
        while low < high {
            let middle = low
                .checked_add(high.checked_sub(low).ok_or(RunCorrupt)? / 2)
                .ok_or(RunCorrupt)?;
            let block = collection
                .first_block
                .checked_add(middle)
                .ok_or(RunCorrupt)?;
            let fence = self.block_meta(block)?.fence;
            if fence <= *logical_id {
                low = middle.checked_add(1).ok_or(RunCorrupt)?;
            } else {
                high = middle;
            }
        }
        let Some(position) = low.checked_sub(1) else {
            return Ok(None);
        };
        let block_index = collection
            .first_block
            .checked_add(position)
            .ok_or(RunCorrupt)?;
        let block = self.block(collection, block_index)?;
        let Some(index) = block.find(logical_id) else {
            return Ok(None);
        };
        let (ordinal, locator) = block.locator(index).ok_or(RunCorrupt)?;
        let pack = *self.packs.get(usize::from(ordinal)).ok_or(RunCorrupt)?;
        Ok(Some(RunHit {
            ordinal,
            pack,
            locator,
        }))
    }

    /// Visit one collection's entries in logical-id order.
    ///
    /// # Errors
    /// Returns [`RunCorrupt`] if any of the collection's blocks fails its
    /// check; entries visited before the failure must be discarded.
    pub fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        mut visit: impl FnMut(RunHit, [u8; 16]),
    ) -> Result<(), RunCorrupt> {
        if self.is_invalid() {
            return Err(RunCorrupt);
        }
        let Some(collection) = self.collection(collection_id) else {
            return Ok(());
        };
        let end = collection
            .first_block
            .checked_add(collection.block_count)
            .ok_or(RunCorrupt)?;
        for block_index in collection.first_block..end {
            let block = self.block(collection, block_index)?;
            for index in 0..block.count {
                let id = block.id(index).ok_or(RunCorrupt)?;
                let (ordinal, locator) = block.locator(index).ok_or(RunCorrupt)?;
                let pack = *self.packs.get(usize::from(ordinal)).ok_or(RunCorrupt)?;
                visit(
                    RunHit {
                        ordinal,
                        pack,
                        locator,
                    },
                    id,
                );
            }
        }
        Ok(())
    }

    /// Every entry in `(collection, id)` order, verifying blocks as it goes.
    pub fn entries(&self) -> impl Iterator<Item = Result<RunEntry, RunCorrupt>> + '_ {
        self.collections.iter().flat_map(move |collection| {
            let end = collection
                .first_block
                .saturating_add(collection.block_count);
            (collection.first_block..end).flat_map(move |block_index| {
                let block = self.block(collection, block_index);
                let count = block.as_ref().map_or(1, |block| block.count);
                (0..count).map(move |index| {
                    let block = block.as_ref().map_err(|error| *error)?;
                    Ok(RunEntry {
                        collection_id: collection.id,
                        logical_id: block.id(index).ok_or(RunCorrupt)?,
                        ordinal: block.locator(index).ok_or(RunCorrupt)?.0,
                        locator: block.locator(index).ok_or(RunCorrupt)?.1,
                    })
                })
            })
        })
    }

    /// Check every block, as a scrub does.
    ///
    /// # Errors
    /// Returns [`RunCorrupt`] at the first failing block.
    pub fn verify_all(&self) -> Result<(), RunCorrupt> {
        for collection in &self.collections {
            let end = collection
                .first_block
                .checked_add(collection.block_count)
                .ok_or(RunCorrupt)?;
            for block_index in collection.first_block..end {
                self.block(collection, block_index)?;
            }
        }
        Ok(())
    }

    fn block_meta(&self, block: u32) -> Result<BlockMeta, RunCorrupt> {
        block_meta_at(&self.map, self.block_table_at, block).ok_or(RunCorrupt)
    }

    /// A block's bytes, checked on first access.
    fn block(&self, collection: &CollectionMeta, block: u32) -> Result<BlockView<'_>, RunCorrupt> {
        if self.is_invalid() {
            return Err(RunCorrupt);
        }
        let view = self.block_view(collection, block);
        if view.is_err() {
            self.invalid.store(true, Ordering::Release);
        }
        view
    }

    fn block_view(
        &self,
        collection: &CollectionMeta,
        block: u32,
    ) -> Result<BlockView<'_>, RunCorrupt> {
        let meta = self.block_meta(block)?;
        let count = usize::try_from(meta.entries).map_err(|_| RunCorrupt)?;
        let start = usize::try_from(meta.data_offset).map_err(|_| RunCorrupt)?;
        let end = start
            .checked_add(count.checked_mul(ENTRY_LEN).ok_or(RunCorrupt)?)
            .ok_or(RunCorrupt)?;
        let view = BlockView {
            bytes: self.map.get(start..end).ok_or(RunCorrupt)?,
            count,
        };
        let word_index = usize::try_from(block / 64).map_err(|_| RunCorrupt)?;
        let bit = 1_u64 << (block % 64);
        let word = self.verified.get(word_index).ok_or(RunCorrupt)?;
        if word.load(Ordering::Acquire) & bit != 0 {
            return Ok(view);
        }
        let last_block = collection
            .first_block
            .checked_add(collection.block_count)
            .and_then(|end| end.checked_sub(1))
            .ok_or(RunCorrupt)?;
        let next_fence = if block < last_block {
            Some(
                self.block_meta(block.checked_add(1).ok_or(RunCorrupt)?)?
                    .fence,
            )
        } else {
            None
        };
        if !view.is_valid(meta, next_fence, &self.packs) {
            return Err(RunCorrupt);
        }
        word.fetch_or(bit, Ordering::AcqRel);
        Ok(view)
    }
}

/// Header fields, checked against the file length and metadata CRC.
struct RunHeader {
    block_entries: u32,
    entry_count: u64,
    pack_count: usize,
    collection_count: usize,
    block_count: u32,
    meta_at: usize,
}

impl RunHeader {
    fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.get(0..8)? != RUN_MAGIC || read_u32(bytes, 8)? != RUN_VERSION {
            return None;
        }
        let header = Self {
            block_entries: read_u32(bytes, 12)?,
            entry_count: read_u64(bytes, 16)?,
            pack_count: usize::try_from(read_u32(bytes, 24)?).ok()?,
            collection_count: usize::try_from(read_u32(bytes, 28)?).ok()?,
            block_count: read_u32(bytes, 32)?,
            meta_at: usize::try_from(read_u64(bytes, 40)?).ok()?,
        };
        let meta_len = usize::try_from(read_u64(bytes, 48)?).ok()?;
        if header.block_entries == 0
            || header.entry_count == 0
            || read_u32(bytes, 36)? != 0
            || read_u32(bytes, 60)? != 0
        {
            return None;
        }
        let expected_meta = header
            .pack_count
            .checked_mul(PACK_ENTRY_LEN)?
            .checked_add(header.collection_count.checked_mul(COLLECTION_ENTRY_LEN)?)?
            .checked_add(
                usize::try_from(header.block_count)
                    .ok()?
                    .checked_mul(BLOCK_ENTRY_LEN)?,
            )?;
        if header.meta_at < RUN_HEADER_LEN
            || meta_len != expected_meta
            || header.meta_at.checked_add(meta_len)? != bytes.len()
        {
            return None;
        }
        let mut fixed = [0_u8; RUN_HEADER_LEN];
        fixed.copy_from_slice(bytes.get(..RUN_HEADER_LEN)?);
        let stored_crc = read_u32(&fixed, CRC_AT)?;
        fixed.get_mut(CRC_AT..CRC_AT.checked_add(4)?)?.fill(0);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&fixed);
        hasher.update(bytes.get(header.meta_at..)?);
        (hasher.finalize() == stored_crc).then_some(header)
    }

    fn collections_at(&self) -> Option<usize> {
        self.meta_at
            .checked_add(self.pack_count.checked_mul(PACK_ENTRY_LEN)?)
    }

    fn block_table_at(&self) -> Option<usize> {
        self.collections_at()?
            .checked_add(self.collection_count.checked_mul(COLLECTION_ENTRY_LEN)?)
    }

    fn parse_packs(&self, bytes: &[u8]) -> Option<Vec<PackIndexIdentity>> {
        let mut packs = Vec::new();
        packs.try_reserve(self.pack_count).ok()?;
        for index in 0..self.pack_count {
            let at = self
                .meta_at
                .checked_add(index.checked_mul(PACK_ENTRY_LEN)?)?;
            packs.push(PackIndexIdentity {
                pack_id: PackId(read_id(bytes, at)?),
                creation_seq: read_u64(bytes, at.checked_add(16)?)?,
                pack_file_len: read_u64(bytes, at.checked_add(24)?)?,
            });
        }
        validate_pack_table(&packs).ok()?;
        Some(packs)
    }

    /// The collection directory, after checking that collections are sorted
    /// and non-empty, blocks are contiguous and tile the data region exactly,
    /// and fences increase within each collection.
    fn parse_collections(&self, bytes: &[u8]) -> Option<Vec<CollectionMeta>> {
        let collections_at = self.collections_at()?;
        let block_table_at = self.block_table_at()?;
        let mut collections: Vec<CollectionMeta> = Vec::new();
        collections.try_reserve(self.collection_count).ok()?;
        let mut next_block = 0_u32;
        let mut next_data = u64::try_from(RUN_HEADER_LEN).ok()?;
        let mut total_entries = 0_u64;
        let entry_len = u64::try_from(ENTRY_LEN).ok()?;
        for index in 0..self.collection_count {
            let at = collections_at.checked_add(index.checked_mul(COLLECTION_ENTRY_LEN)?)?;
            let collection = CollectionMeta {
                id: read_id(bytes, at)?,
                entry_count: read_u64(bytes, at.checked_add(16)?)?,
                first_block: read_u32(bytes, at.checked_add(24)?)?,
                block_count: read_u32(bytes, at.checked_add(28)?)?,
            };
            let end = collection.first_block.checked_add(collection.block_count)?;
            if collections
                .last()
                .is_some_and(|previous| previous.id >= collection.id)
                || collection.entry_count == 0
                || collection.block_count == 0
                || collection.first_block != next_block
                || end > self.block_count
            {
                return None;
            }
            let mut collection_entries = 0_u64;
            let mut previous_fence: Option<[u8; 16]> = None;
            for block in collection.first_block..end {
                let meta = block_meta_at(bytes, block_table_at, block)?;
                if meta.data_offset != next_data
                    || meta.entries == 0
                    || meta.entries > self.block_entries
                    || previous_fence.is_some_and(|fence| fence >= meta.fence)
                {
                    return None;
                }
                previous_fence = Some(meta.fence);
                next_data =
                    next_data.checked_add(u64::from(meta.entries).checked_mul(entry_len)?)?;
                collection_entries = collection_entries.checked_add(u64::from(meta.entries))?;
            }
            if collection_entries != collection.entry_count {
                return None;
            }
            total_entries = total_entries.checked_add(collection_entries)?;
            next_block = end;
            collections.push(collection);
        }
        let complete = next_block == self.block_count
            && total_entries == self.entry_count
            && next_data == u64::try_from(self.meta_at).ok()?;
        complete.then_some(collections)
    }
}

fn block_meta_at(bytes: &[u8], table_at: usize, block: u32) -> Option<BlockMeta> {
    let at = table_at.checked_add(usize::try_from(block).ok()?.checked_mul(BLOCK_ENTRY_LEN)?)?;
    Some(BlockMeta {
        fence: read_id(bytes, at)?,
        data_offset: read_u64(bytes, at.checked_add(16)?)?,
        entries: read_u32(bytes, at.checked_add(24)?)?,
        crc: read_u32(bytes, at.checked_add(28)?)?,
    })
}

struct BlockView<'a> {
    bytes: &'a [u8],
    count: usize,
}

impl BlockView<'_> {
    fn id(&self, index: usize) -> Option<[u8; 16]> {
        read_id(self.bytes, index.checked_mul(ID_LEN)?)
    }

    fn locator(&self, index: usize) -> Option<(u16, PackIndexLocator)> {
        let at = self
            .count
            .checked_mul(ID_LEN)?
            .checked_add(index.checked_mul(LOCATOR_LEN)?)?;
        let ordinal = u16::from_le_bytes(self.bytes.get(at..at.checked_add(2)?)?.try_into().ok()?);
        let offset = read_u32(self.bytes, at.checked_add(2)?)?;
        let flags = *self.bytes.get(at.checked_add(6)?)?;
        let locator = if flags & RUN_FLAG_REMOVED != 0 {
            PackIndexLocator::Removed(offset)
        } else {
            PackIndexLocator::Frame(offset)
        };
        Some((ordinal, locator))
    }

    fn find(&self, logical_id: &[u8; 16]) -> Option<usize> {
        let mut low = 0_usize;
        let mut high = self.count;
        while low < high {
            let middle = low.checked_add(high.checked_sub(low)? / 2)?;
            match self.id(middle)?.cmp(logical_id) {
                std::cmp::Ordering::Less => low = middle.checked_add(1)?,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Some(middle),
            }
        }
        None
    }

    /// CRC, strictly increasing ids starting at the fence and below the next
    /// block's fence, known flags, and every locator inside its pack.
    fn is_valid(
        &self,
        meta: BlockMeta,
        next_fence: Option<[u8; 16]>,
        packs: &[PackIndexIdentity],
    ) -> bool {
        if crc32fast::hash(self.bytes) != meta.crc || self.id(0) != Some(meta.fence) {
            return false;
        }
        let mut previous: Option<[u8; 16]> = None;
        for index in 0..self.count {
            let Some(id) = self.id(index) else {
                return false;
            };
            if previous.is_some_and(|previous| previous >= id) {
                return false;
            }
            previous = Some(id);
            let Some((ordinal, locator)) = self.locator(index) else {
                return false;
            };
            let Some(flags_at) = self
                .count
                .checked_mul(ID_LEN)
                .and_then(|ids| ids.checked_add(index.checked_mul(LOCATOR_LEN)?))
                .and_then(|at| at.checked_add(6))
            else {
                return false;
            };
            if self
                .bytes
                .get(flags_at)
                .is_none_or(|flags| flags & !RUN_FLAG_REMOVED != 0)
            {
                return false;
            }
            let (offset, _) = locator_parts(locator);
            if packs
                .get(usize::from(ordinal))
                .is_none_or(|pack| !offset_in_pack(offset, pack))
            {
                return false;
            }
        }
        match (previous, next_fence) {
            (Some(last), Some(next)) => last < next,
            (Some(_), None) => true,
            (None, _) => false,
        }
    }
}

/// One input to [`merge_to_path`].
#[derive(Debug, Clone, Copy)]
pub enum MergeSource<'a> {
    /// One sealed pack's sidecar.
    Sidecar(&'a MappedPackIndex),
    /// An existing run.
    Run(&'a RunIndex),
}

impl MergeSource<'_> {
    fn packs(&self) -> Vec<PackIndexIdentity> {
        match self {
            Self::Sidecar(index) => vec![index.identity()],
            Self::Run(run) => run.packs().to_vec(),
        }
    }
}

/// How a merge treats its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeOptions {
    /// Drop removal versions. Set this only when the output covers the pool's
    /// oldest sealed pack, so no older run or sidecar can hold the key.
    pub drop_removals: bool,
    /// Entries per output block.
    pub block_entries: u32,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            drop_removals: false,
            block_entries: RUN_DEFAULT_BLOCK_ENTRIES,
        }
    }
}

type Cursor<'a> = Box<dyn Iterator<Item = Result<RunEntry, RunCorrupt>> + 'a>;

/// Merge heap key: `(collection, id)` ascending, then the newest pack first,
/// then the input index.
type HeapKey = Reverse<([u8; 16], [u8; 16], Reverse<u16>, usize)>;

/// Merge sidecars and runs covering a contiguous pack range into one run at
/// `path`, keeping the newest version of each key.
///
/// The output is written to `<path>.tmp`, fsynced, renamed to `path`, and the
/// directory fsynced. An output with no entries (every key was a dropped
/// removal) is not written and returns `None`. Memory is one entry per input
/// plus the writer's per-block metadata.
///
/// # Errors
/// Returns [`RunError::OverlappingInputs`] if two inputs share a pack or
/// interleave in pack order, [`RunError::Corrupt`] if an input run fails a
/// block check, and I/O errors. On error no file is left at `path`.
pub fn merge_to_path(
    sources: &[MergeSource<'_>],
    options: MergeOptions,
    path: &Path,
) -> Result<Option<RunSummary>, RunError> {
    let mut packs: Vec<PackIndexIdentity> = sources.iter().flat_map(MergeSource::packs).collect();
    packs.sort_by_key(pack_order);
    if packs
        .windows(2)
        .any(|pair| pack_order(&pair[0]) == pack_order(&pair[1]))
    {
        return Err(RunError::OverlappingInputs);
    }
    validate_pack_table(&packs)?;
    let ordinal_of = |pack: &PackIndexIdentity| -> Result<u16, RunError> {
        let index = packs
            .binary_search_by_key(&pack_order(pack), pack_order)
            .map_err(|_| RunError::OverlappingInputs)?;
        u16::try_from(index).map_err(|_| RunError::BadPackTable)
    };

    let mut cursors: Vec<Cursor<'_>> = Vec::with_capacity(sources.len());
    for source in sources {
        let source_packs = source.packs();
        let remap = source_packs
            .iter()
            .map(&ordinal_of)
            .collect::<Result<Vec<u16>, RunError>>()?;
        // A source's packs must be consecutive in the output's pack order, so
        // inputs cannot interleave.
        if remap
            .windows(2)
            .any(|pair| pair[0].checked_add(1) != Some(pair[1]))
        {
            return Err(RunError::OverlappingInputs);
        }
        match *source {
            MergeSource::Sidecar(index) => {
                let ordinal = *remap.first().ok_or(RunError::BadPackTable)?;
                cursors.push(Box::new(index.entries().map(
                    move |(collection_id, logical_id, locator)| {
                        Ok(RunEntry {
                            collection_id,
                            logical_id,
                            ordinal,
                            locator,
                        })
                    },
                )));
            }
            MergeSource::Run(run) => {
                cursors.push(Box::new(run.entries().map(move |entry| {
                    let mut entry = entry?;
                    entry.ordinal = *remap.get(usize::from(entry.ordinal)).ok_or(RunCorrupt)?;
                    Ok(entry)
                })));
            }
        }
    }

    let mut temp: OsString = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    let result = write_merged(&mut cursors, packs, options, &temp);
    let summary = match result {
        Ok(summary) => summary,
        Err(error) => {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
    };
    if summary.is_some() {
        if let Err(error) = fs::rename(&temp, path) {
            let _ = fs::remove_file(&temp);
            return Err(error.into());
        }
        if let Some(parent) = path.parent() {
            sync_dir(parent)?;
        }
    }
    Ok(summary)
}

fn write_merged(
    cursors: &mut [Cursor<'_>],
    packs: Vec<PackIndexIdentity>,
    options: MergeOptions,
    temp: &Path,
) -> Result<Option<RunSummary>, RunError> {
    let mut writer = RunWriter::create(temp, packs, options.block_entries)?;
    // Min-heap on (collection, id), newest pack first among equal keys.
    let mut heap: BinaryHeap<HeapKey> = BinaryHeap::with_capacity(cursors.len());
    let mut heads: Vec<Option<RunEntry>> = vec![None; cursors.len()];
    for (source, cursor) in cursors.iter_mut().enumerate() {
        advance(cursor, source, &mut heads, &mut heap)?;
    }
    while let Some(Reverse((collection_id, logical_id, _, source))) = heap.pop() {
        // Read the head without clearing it: `advance` compares the next
        // entry against it to catch an input that steps backwards.
        let winner = heads
            .get(source)
            .copied()
            .flatten()
            .ok_or(RunError::Invalid)?;
        advance(
            cursors.get_mut(source).ok_or(RunError::Invalid)?,
            source,
            &mut heads,
            &mut heap,
        )?;
        // Skip every older version of the same key.
        while let Some(Reverse((next_collection, next_id, _, next_source))) = heap.peek().copied() {
            if (next_collection, next_id) != (collection_id, logical_id) {
                break;
            }
            heap.pop();
            advance(
                cursors.get_mut(next_source).ok_or(RunError::Invalid)?,
                next_source,
                &mut heads,
                &mut heap,
            )?;
        }
        if options.drop_removals && is_removed(winner.locator) {
            continue;
        }
        writer.push(winner)?;
    }
    writer.finish()
}

fn advance(
    cursor: &mut Cursor<'_>,
    source: usize,
    heads: &mut [Option<RunEntry>],
    heap: &mut BinaryHeap<HeapKey>,
) -> Result<(), RunError> {
    let Some(entry) = cursor.next().transpose()? else {
        return Ok(());
    };
    // Each input is sorted; a step backwards means a corrupt input, and the
    // writer would reject the merged order anyway.
    if heads
        .get(source)
        .ok_or(RunError::Invalid)?
        .is_some_and(|previous| {
            (previous.collection_id, previous.logical_id) >= (entry.collection_id, entry.logical_id)
        })
    {
        return Err(RunError::OutOfOrder);
    }
    heap.push(Reverse((
        entry.collection_id,
        entry.logical_id,
        Reverse(entry.ordinal),
        source,
    )));
    *heads.get_mut(source).ok_or(RunError::Invalid)? = Some(entry);
    Ok(())
}

fn read_id(bytes: &[u8], at: usize) -> Option<[u8; 16]> {
    bytes.get(at..at.checked_add(16)?)?.try_into().ok()
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
#[path = "test_run.rs"]
mod tests;
