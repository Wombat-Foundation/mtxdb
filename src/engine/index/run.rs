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
//! [metadata: pack table, collection directory, block table, pack stats]
//! ```
//!
//! The header holds the `MTXRUN01` magic, the version, the block size `B`,
//! the entry, pack, collection, block and stats counts, the metadata offset and
//! length, and `meta_crc32` over the header (with that field zeroed) and the
//! metadata. The metadata sits after the blocks because a streaming merge
//! learns the collection and block counts only as it writes.
//!
//! - **Pack table:** `pack_count` entries of `(pack_id, creation_seq,
//!   pack_file_len)`, strictly increasing in `(creation_seq, pack_id)`.
//! - **Collection directory:** `(collection_id, entry_count, first_block,
//!   block_count)`, strictly increasing by id. A collection is never empty.
//! - **Block table:** `(fence, last, data_offset, entries, crc32)`, where
//!   `fence` and `last` are the block's first and last logical ids. Blocks
//!   never span collections, and within a collection each block's `fence` is
//!   above the previous block's `last`, so block selection cannot be misled by
//!   damaged data.
//! - **Pack stats:** one `(pack ordinal, collection_id, entry_count,
//!   disk_bytes)` row per collection present in a covered pack, sorted by
//!   collection then ordinal. These are each pack's own sidecar directory
//!   values, carried through merges unchanged, so open can rebuild per-pack
//!   accounting without opening the covered sidecars.
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
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

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
const BLOCK_ENTRY_LEN: usize = 48;
const STATS_ENTRY_LEN: usize = 34;
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
    /// A merge was cancelled before it completed.
    Cancelled,
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
            Self::Cancelled => f.write_str("run merge cancelled"),
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

/// One covered pack's accounting for one collection, as its sidecar directory
/// recorded it: every locator and frame byte in that pack, including versions
/// a newer pack supersedes. Merges carry these rows forward rather than
/// recounting merged entries, which hold only the newest version of each key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunPackStats {
    /// Index into the run's pack table.
    pub ordinal: u16,
    /// The collection.
    pub collection_id: [u8; 16],
    /// Unique locators the pack's sidecar records for the collection.
    pub entry_count: u64,
    /// Bytes the collection's frames occupy in the pack.
    pub disk_bytes: u64,
}

fn stats_key(row: &RunPackStats) -> ([u8; 16], u16) {
    (row.collection_id, row.ordinal)
}

/// Rows strictly increasing by `(collection, ordinal)`, naming known packs,
/// each with at least one entry.
fn stats_are_valid(rows: &[RunPackStats], pack_count: usize) -> bool {
    rows.iter()
        .all(|row| usize::from(row.ordinal) < pack_count && row.entry_count > 0)
        && rows
            .windows(2)
            .all(|pair| stats_key(&pair[0]) < stats_key(&pair[1]))
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
    // Pack ids name packs everywhere else, so one id twice (with different
    // sequences) is as invalid as an exact duplicate.
    let mut ids: Vec<[u8; 16]> = packs.iter().map(|pack| pack.pack_id.0).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
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
    last: [u8; 16],
    data_offset: u64,
    entries: u32,
    crc: u32,
}

/// Streams sorted entries into a run file.
///
/// Memory is one block of entries plus metadata: 48 bytes per block, 32 per
/// collection, and one counter per `(collection, pack)` pair in `seen_stats`.
/// `finish` uses those counters to reject explicit pack stats that omit a pair
/// the run holds, and to derive stats when the caller supplied none. For `N`
/// entries, block size `B`, `C` collections, `P` collection-pack pairs, and
/// `K` packs, memory is O(B + N/B + C + P + K); `P` can be O(N).
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
    pending_last: [u8; 16],
    data_end: u64,
    last: Option<([u8; 16], [u8; 16])>,
    entry_count: u64,
    stats: Vec<RunPackStats>,
    seen_stats: BTreeMap<([u8; 16], u16), u64>,
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
            pending_last: [0; 16],
            data_end: u64::try_from(RUN_HEADER_LEN).unwrap_or(u64::MAX),
            last: None,
            entry_count: 0,
            stats: Vec::new(),
            seen_stats: BTreeMap::new(),
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
        // Track every pushed pair even when explicit stats were supplied
        // before the first entry: `finish` uses this to reject stats that omit
        // a pair the run actually contains.
        let count = self
            .seen_stats
            .entry((entry.collection_id, entry.ordinal))
            .or_insert(0);
        *count = count.checked_add(1).ok_or(RunError::Invalid)?;
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
        self.pending_last = entry.logical_id;
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

    /// Record the covered packs' per-collection accounting, in any order.
    ///
    /// # Errors
    /// Returns [`RunError::BadLocator`] if a row names an unknown pack, has no
    /// entries, or repeats a `(collection, pack)` pair.
    pub fn set_pack_stats(&mut self, mut stats: Vec<RunPackStats>) -> Result<(), RunError> {
        stats.sort_unstable_by_key(stats_key);
        if !stats_are_valid(&stats, self.packs.len()) {
            return Err(RunError::BadLocator);
        }
        self.stats = stats;
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
            last: self.pending_last,
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
        if self.stats.is_empty() {
            self.stats = self
                .seen_stats
                .iter()
                .map(|(&(collection_id, ordinal), &entry_count)| RunPackStats {
                    ordinal,
                    collection_id,
                    entry_count,
                    disk_bytes: 0,
                })
                .collect();
            self.stats.sort_unstable_by_key(stats_key);
        }
        // Every pushed (collection, pack) pair needs a stats row. Without one,
        // storage omits the pair from collection membership and disk-byte
        // accounting even though the run holds its entries.
        let mut stats_iter = self.stats.iter().peekable();
        for key in self.seen_stats.keys() {
            while stats_iter.peek().is_some_and(|row| stats_key(row) < *key) {
                stats_iter.next();
            }
            if stats_iter.peek().map(|row| stats_key(row)) != Some(*key) {
                return Err(RunError::BadLocator);
            }
        }
        let pack_count = u32::try_from(self.packs.len()).map_err(|_| RunError::Invalid)?;
        let collection_count =
            u32::try_from(self.collections.len()).map_err(|_| RunError::Invalid)?;
        let block_count = u32::try_from(self.blocks.len()).map_err(|_| RunError::Invalid)?;
        let stats_count = u32::try_from(self.stats.len()).map_err(|_| RunError::Invalid)?;
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
            meta.extend_from_slice(&block.last);
            meta.extend_from_slice(&block.data_offset.to_le_bytes());
            meta.extend_from_slice(&block.entries.to_le_bytes());
            meta.extend_from_slice(&block.crc.to_le_bytes());
        }
        for row in &self.stats {
            meta.extend_from_slice(&row.ordinal.to_le_bytes());
            meta.extend_from_slice(&row.collection_id);
            meta.extend_from_slice(&row.entry_count.to_le_bytes());
            meta.extend_from_slice(&row.disk_bytes.to_le_bytes());
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
        header[36..40].copy_from_slice(&stats_count.to_le_bytes());
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
    path: PathBuf,
    packs: Vec<PackIndexIdentity>,
    collections: Vec<CollectionMeta>,
    block_entries: u32,
    block_table_at: usize,
    stats_at: usize,
    stats_count: usize,
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
        Self::from_map(map, path.to_path_buf()).ok_or(RunError::Invalid)
    }

    fn from_map(map: Mmap, path: PathBuf) -> Option<Self> {
        let header = RunHeader::parse(&map)?;
        let packs = header.parse_packs(&map)?;
        let collections = header.parse_collections(&map)?;
        header.check_stats(&map)?;
        let words = usize::try_from(header.block_count).ok()?.div_ceil(64);
        let verified = (0..words).map(|_| AtomicU64::new(0)).collect();
        Some(Self {
            map,
            path,
            packs,
            collections,
            block_entries: header.block_entries,
            block_table_at: header.block_table_at()?,
            stats_at: header.stats_at()?,
            stats_count: header.stats_count,
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

    /// Filesystem path from which this immutable run mapping was opened.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Each covered pack's per-collection accounting, sorted by collection then
    /// pack ordinal. Checked at open.
    pub fn pack_stats(&self) -> impl Iterator<Item = RunPackStats> + '_ {
        (0..self.stats_count).filter_map(move |index| stats_row_at(&self.map, self.stats_at, index))
    }

    /// The [`Self::pack_stats`] rows for one collection, oldest pack first,
    /// found by binary search over the sorted stats table.
    pub fn pack_stats_in_collection<'a>(
        &'a self,
        collection_id: &'a [u8; 16],
    ) -> impl Iterator<Item = RunPackStats> + 'a {
        let row = move |index| stats_row_at(&self.map, self.stats_at, index);
        let mut low = 0_usize;
        let mut high = self.stats_count;
        while low < high {
            let middle = low.midpoint(high);
            match row(middle) {
                Some(stats) if stats.collection_id < *collection_id => {
                    low = middle.saturating_add(1);
                }
                _ => high = middle,
            }
        }
        (low..self.stats_count).map_while(move |index| {
            row(index).filter(|stats| stats.collection_id == *collection_id)
        })
    }

    /// Unique `(collection, id)` entries in the run.
    #[must_use]
    pub const fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Number of pack stats rows recorded in this run.
    #[must_use]
    pub const fn stats_count(&self) -> usize {
        self.stats_count
    }

    /// Whether this run records per-collection pack accounting rows.
    #[must_use]
    pub const fn has_pack_stats(&self) -> bool {
        self.stats_count > 0
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
        let block = self.block(block_index)?;
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
            let block = self.block(block_index)?;
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
                let block = self.block(block_index);
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

    /// Check every block of one collection, so a scan can then stream its
    /// entries without buffering them against a later block failure.
    ///
    /// # Errors
    /// Returns [`RunCorrupt`] if the run is invalid or one of the
    /// collection's blocks fails its check.
    pub fn verify_collection(&self, collection_id: &[u8; 16]) -> Result<(), RunCorrupt> {
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
            self.block(block_index)?;
        }
        Ok(())
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
                self.block(block_index)?;
            }
        }
        Ok(())
    }

    fn block_meta(&self, block: u32) -> Result<BlockMeta, RunCorrupt> {
        block_meta_at(&self.map, self.block_table_at, block).ok_or(RunCorrupt)
    }

    /// A block's bytes, checked on first access.
    fn block(&self, block: u32) -> Result<BlockView<'_>, RunCorrupt> {
        if self.is_invalid() {
            return Err(RunCorrupt);
        }
        let view = self.block_view(block);
        if view.is_err() {
            self.invalid.store(true, Ordering::Release);
        }
        view
    }

    fn block_view(&self, block: u32) -> Result<BlockView<'_>, RunCorrupt> {
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
        if !view.is_valid(meta, &self.packs) {
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
    stats_count: usize,
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
            stats_count: usize::try_from(read_u32(bytes, 36)?).ok()?,
            meta_at: usize::try_from(read_u64(bytes, 40)?).ok()?,
        };
        let meta_len = usize::try_from(read_u64(bytes, 48)?).ok()?;
        if header.block_entries == 0 || header.entry_count == 0 || read_u32(bytes, 60)? != 0 {
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
            )?
            .checked_add(header.stats_count.checked_mul(STATS_ENTRY_LEN)?)?;
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

    fn stats_at(&self) -> Option<usize> {
        self.block_table_at()?.checked_add(
            usize::try_from(self.block_count)
                .ok()?
                .checked_mul(BLOCK_ENTRY_LEN)?,
        )
    }

    /// Check the stats rows without keeping them: O(rows) at open.
    fn check_stats(&self, bytes: &[u8]) -> Option<()> {
        let at = self.stats_at()?;
        let mut previous: Option<RunPackStats> = None;
        for index in 0..self.stats_count {
            let row = stats_row_at(bytes, at, index)?;
            if usize::from(row.ordinal) >= self.pack_count
                || row.entry_count == 0
                || previous.is_some_and(|previous| stats_key(&previous) >= stats_key(&row))
            {
                return None;
            }
            previous = Some(row);
        }
        Some(())
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
    /// and block id ranges are disjoint and increasing within each collection.
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
            let mut previous_last: Option<[u8; 16]> = None;
            for block in collection.first_block..end {
                let meta = block_meta_at(bytes, block_table_at, block)?;
                if meta.data_offset != next_data
                    || meta.entries == 0
                    || meta.entries > self.block_entries
                    || meta.fence > meta.last
                    || (meta.entries == 1) != (meta.fence == meta.last)
                    || previous_last.is_some_and(|last| last >= meta.fence)
                {
                    return None;
                }
                previous_last = Some(meta.last);
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

fn stats_row_at(bytes: &[u8], table_at: usize, index: usize) -> Option<RunPackStats> {
    let at = table_at.checked_add(index.checked_mul(STATS_ENTRY_LEN)?)?;
    Some(RunPackStats {
        ordinal: u16::from_le_bytes(bytes.get(at..at.checked_add(2)?)?.try_into().ok()?),
        collection_id: read_id(bytes, at.checked_add(2)?)?,
        entry_count: read_u64(bytes, at.checked_add(18)?)?,
        disk_bytes: read_u64(bytes, at.checked_add(26)?)?,
    })
}

fn block_meta_at(bytes: &[u8], table_at: usize, block: u32) -> Option<BlockMeta> {
    let at = table_at.checked_add(usize::try_from(block).ok()?.checked_mul(BLOCK_ENTRY_LEN)?)?;
    Some(BlockMeta {
        fence: read_id(bytes, at)?,
        last: read_id(bytes, at.checked_add(16)?)?,
        data_offset: read_u64(bytes, at.checked_add(32)?)?,
        entries: read_u32(bytes, at.checked_add(40)?)?,
        crc: read_u32(bytes, at.checked_add(44)?)?,
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

    /// CRC, strictly increasing ids from the block table's fence to its last
    /// id, known flags, and every locator inside its pack. Open has already
    /// checked that block id ranges are disjoint and increasing, so a block
    /// that passes cannot hide a key another block was chosen for.
    fn is_valid(&self, meta: BlockMeta, packs: &[PackIndexIdentity]) -> bool {
        let last_index = self.count.checked_sub(1);
        if crc32fast::hash(self.bytes) != meta.crc
            || self.id(0) != Some(meta.fence)
            || last_index.and_then(|index| self.id(index)) != Some(meta.last)
        {
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
        previous.is_some()
    }
}

/// One input to [`merge_to_path`].
#[derive(Debug, Clone, Copy)]
pub enum MergeSource<'a> {
    /// One sealed pack's sidecar. It must have passed
    /// [`MappedPackIndex::verify`], as [`super::packidx::open_for_pack`]
    /// sidecars have: the merge reads entries without re-checking them, so an
    /// unverified damaged sidecar could publish a run that is missing keys.
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

    /// This source's accounting rows with ordinals mapped through `remap`.
    fn stats(&self, remap: &[u16]) -> Result<Vec<RunPackStats>, RunError> {
        match self {
            Self::Sidecar(index) => {
                let ordinal = *remap.first().ok_or(RunError::BadPackTable)?;
                let mut rows = Vec::new();
                let mut missing = false;
                index.for_each_collection(|collection_id, disk_bytes| {
                    match index.collection_entry_count(&collection_id) {
                        Some(entry_count) => rows.push(RunPackStats {
                            ordinal,
                            collection_id,
                            entry_count,
                            disk_bytes,
                        }),
                        None => missing = true,
                    }
                });
                if missing {
                    return Err(RunError::Invalid);
                }
                Ok(rows)
            }
            Self::Run(run) => run
                .pack_stats()
                .map(|mut row| {
                    row.ordinal = *remap
                        .get(usize::from(row.ordinal))
                        .ok_or(RunError::BadPackTable)?;
                    Ok(row)
                })
                .collect(),
        }
    }
}

/// How a merge treats its inputs.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Drop removal versions. Set this only when the output covers the pool's
    /// oldest sealed pack, so no older run or sidecar can hold the key.
    pub drop_removals: bool,
    /// Entries per output block.
    pub block_entries: u32,
    /// Optional cancellation signal. When set, an in-progress merge aborts early.
    pub stop: Option<Arc<AtomicBool>>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            drop_removals: false,
            block_entries: RUN_DEFAULT_BLOCK_ENTRIES,
            stop: None,
        }
    }
}

type Cursor<'a> = Box<dyn Iterator<Item = Result<RunEntry, RunCorrupt>> + 'a>;

/// Merge heap key: `(collection, id)` ascending, then the newest pack first,
/// then the input index.
type HeapKey = Reverse<([u8; 16], [u8; 16], Reverse<u16>, usize)>;

/// One lock per output path. `merge_to_path` writes every call through the
/// same `<path>.tmp`, so concurrent calls for one destination must not
/// overlap; different destinations still proceed independently.
fn merge_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(lock) = locks.get(path).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(path.to_path_buf(), Arc::downgrade(&lock));
    lock
}

/// Merge sidecars and runs covering a contiguous pack range into one run at
/// `path`, keeping the newest version of each key.
///
/// The output is written to `<path>.tmp`, fsynced, renamed to `path`, and the
/// directory fsynced. An output with no entries (every key was a dropped
/// removal) is not written, any earlier run at `path` is removed, and `None`
/// is returned. Each input's per-pack accounting is copied into the output's
/// pack stats with remapped ordinals. The merge checks that inputs do not overlap or interleave; that
/// they cover every pack of the pool's range is the caller's check. Memory is one entry per input
/// plus the writer's per-block metadata.
///
/// # Errors
/// Returns [`RunError::OverlappingInputs`] if two inputs share a pack or
/// interleave in pack order, [`RunError::Corrupt`] if an input run fails a
/// block check, [`RunError::Cancelled`] if the stop signal is set, and I/O
/// errors. On error no file is left at `path`.
///
/// Concurrent calls for the same `path` are serialized: every call stages
/// through the same `<path>.tmp` name before its atomic rename.
pub fn merge_to_path(
    sources: &[MergeSource<'_>],
    options: &MergeOptions,
    path: &Path,
) -> Result<Option<RunSummary>, RunError> {
    let merge_lock = merge_lock(path);
    let _merge_guard = merge_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let mut stats = Vec::new();
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
        stats.extend(source.stats(&remap)?);
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
    let result = write_merged(&mut cursors, packs, stats, options, &temp);
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
    } else {
        // Nothing survived the merge. A run left at `path` from an earlier
        // merge of this range must not outlive it.
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        }
    }
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    crate::shard::sync_directory(parent)?;
    Ok(summary)
}

fn write_merged(
    cursors: &mut [Cursor<'_>],
    packs: Vec<PackIndexIdentity>,
    stats: Vec<RunPackStats>,
    options: &MergeOptions,
    temp: &Path,
) -> Result<Option<RunSummary>, RunError> {
    let mut writer = RunWriter::create(temp, packs, options.block_entries)?;
    writer.set_pack_stats(stats)?;
    // Min-heap on (collection, id), newest pack first among equal keys.
    let mut heap: BinaryHeap<HeapKey> = BinaryHeap::with_capacity(cursors.len());
    let mut heads: Vec<Option<RunEntry>> = vec![None; cursors.len()];
    for (source, cursor) in cursors.iter_mut().enumerate() {
        advance(cursor, source, &mut heads, &mut heap)?;
    }
    while let Some(Reverse((collection_id, logical_id, _, source))) = heap.pop() {
        if options
            .stop
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Relaxed))
        {
            return Err(RunError::Cancelled);
        }
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

/// Extension of a published run file.
pub const RUN_EXTENSION: &str = "run";

/// File name for a run covering `first..=last` by creation sequence.
#[must_use]
pub fn run_file_name(first: &PackIndexIdentity, last: &PackIndexIdentity) -> String {
    format!(
        "{:020}-{:020}.{RUN_EXTENSION}",
        first.creation_seq, last.creation_seq
    )
}

/// A run accepted by [`load_pool_runs`].
#[derive(Debug)]
pub struct PoolRun {
    /// The published run file.
    pub path: PathBuf,
    /// The opened run.
    pub run: RunIndex,
}

/// Open every run in `dir` that can serve the pool's current packs.
///
/// `pool` is every pack in the pool, sorted by `(creation_seq, pack_id)`. A
/// run is kept only if it opens, every covered pack is in `pool` with the same
/// identity, and its packs are consecutive in `pool` (so no uncovered pack
/// falls inside its range). When a newly merged run contains older run ranges,
/// the containing run wins and the superseded files are rejected. Partial
/// overlaps are rejected together. Kept runs are returned oldest first.
///
/// When `writable`, rejected runs and leftover `.run.tmp` files are deleted:
/// runs are derived data and the covered packs fall back to their sidecars. A
/// read-only open leaves them in place.
#[must_use]
pub fn load_pool_runs(dir: &Path, pool: &[PackIndexIdentity], writable: bool) -> Vec<PoolRun> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut rejected = Vec::new();
    let mut candidates: Vec<(usize, usize, PoolRun)> = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.ends_with(".run.tmp") {
            rejected.push(path);
            continue;
        }
        if path
            .extension()
            .is_none_or(|extension| extension != RUN_EXTENSION)
        {
            continue;
        }
        match RunIndex::open(&path)
            .ok()
            .and_then(|run| pool_span(&run, pool).map(|span| (span, run)))
        {
            Some(((start, end), run)) => {
                if run.entry_count() > 0 && !run.has_pack_stats() {
                    rejected.push(path);
                } else {
                    candidates.push((start, end, PoolRun { path, run }));
                }
            }
            None => rejected.push(path),
        }
    }
    // Prefer strict supersets: a crash after publishing a merged run but
    // before swapping generations leaves the old, valid input runs beside the
    // new file. The superset contains their complete coverage and is itself
    // safe to use because every pack identity was checked above. Incomparable
    // partial overlaps remain ambiguous and are all rejected.
    candidates.sort_by_key(|(start, end, _)| {
        (std::cmp::Reverse(end.saturating_sub(*start)), *start, *end)
    });
    let mut selected: Vec<(usize, usize, PoolRun)> = Vec::new();
    let mut ambiguous = Vec::new();
    for candidate in candidates {
        let (start, end, _) = &candidate;
        let mut partial_overlap = false;
        for (selected_start, selected_end, _) in &selected {
            if start > selected_end || end < selected_start {
                continue;
            }
            if (*start <= *selected_start && *end >= *selected_end)
                || (*selected_start <= *start && *selected_end >= *end)
            {
                // The already-selected range is at least as large, or this
                // candidate contains it. Sorting by descending size means
                // the latter is only possible for equal-size duplicates.
                partial_overlap = false;
                break;
            }
            partial_overlap = true;
            break;
        }
        if partial_overlap {
            ambiguous.push((candidate.0, candidate.1));
            rejected.push(candidate.2.path);
        } else if selected.iter().any(|(selected_start, selected_end, _)| {
            *selected_start <= *start && *selected_end >= *end
        }) {
            rejected.push(candidate.2.path);
        } else {
            selected.push(candidate);
        }
    }
    // If an incomparable overlap was found, reject every selected run it
    // intersects as well. This deliberately falls back to authoritative
    // sidecars rather than guessing which coverage is newer.
    selected.retain(|(start, end, pool_run)| {
        let conflicts = ambiguous
            .iter()
            .any(|(other_start, other_end)| *start <= *other_end && *end >= *other_start);
        if conflicts {
            rejected.push(pool_run.path.clone());
        }
        !conflicts
    });
    selected.sort_by_key(|(start, end, _)| (*start, *end));
    let kept = selected.into_iter().map(|(_, _, run)| run).collect();
    if writable {
        for path in rejected {
            let _ = fs::remove_file(path);
        }
    }
    kept
}

/// The inclusive index range of `run`'s packs within `pool`, or `None` if any
/// pack is missing, has a different identity, or the packs are not
/// consecutive in `pool`.
fn pool_span(run: &RunIndex, pool: &[PackIndexIdentity]) -> Option<(usize, usize)> {
    let first = run.packs().first()?;
    let start = pool
        .binary_search_by_key(&pack_order(first), pack_order)
        .ok()?;
    let end = start.checked_add(run.packs().len().checked_sub(1)?)?;
    let covered = pool.get(start..=end)?;
    (covered == run.packs()).then_some((start, end))
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

#[cfg(test)]
#[path = "test_run.rs"]
mod tests;
