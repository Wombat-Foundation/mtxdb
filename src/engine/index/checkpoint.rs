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
//!   [CheckpointHeader 104B]
//!   [CollectionDirEntry * count, 56B each]
//!   [collection 0 slots: capacity * 8B]
//!   [collection 1 slots: capacity * 8B]
//!   [PackTableEntry * pack_table_count, 20B each]
//!   [CollectionVersionEntry * logical_version_count, 24B each]
//!   ...
//! ```
//!
//! The header's `pack_fingerprint` is a deterministic hash of the
//! `(pack_id, length)` set the checkpoint describes. An opener only trusts
//! the file when that fingerprint matches the packs currently on disk (optionally
//! with a gated delta replay continuing it); any append, rotation, or repack
//! changes it and forces a rescan.

use std::collections::HashSet;
use std::fs;
use std::io::{self, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use memmap2::Mmap;

use super::bucket::RawBucketTable;
use super::format::{
    CheckpointHeader, CollectionDirEntry, CollectionVersionEntry, PackTableEntry,
    CHECKPOINT_HEADER_LEN, COLLECTION_DIR_ENTRY_LEN, COLLECTION_VERSION_ENTRY_LEN,
    PACK_TABLE_ENTRY_LEN,
};
use crate::packfile::PackId;

/// Magic identifying the persisted-index checkpoint format.
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"MTXI0002";
/// Current v2 wire version (see [`CheckpointHeader::version`]).
///
/// This is a hard-cutover format; a mismatched version causes the checkpoint
/// to be rebuilt rather than decoded through a different layout.
pub const CHECKPOINT_VERSION: u32 = 2;
/// File name of the persisted index checkpoint inside a store's base dir.
pub const INDEX_CHECKPOINT_FILE: &str = "index.checkpoint";

fn supported_checkpoint_header(header: &CheckpointHeader) -> bool {
    header.magic == CHECKPOINT_MAGIC && header.version == CHECKPOINT_VERSION
}

/// Whether an opener verifies the checkpoint body's CRC32.
///
/// This policy is intentionally independent from [`crate::packfile::ChecksumPolicy`]:
/// frame checksums, delta-log checksums, and checkpoint integrity protect
/// different on-disk structures and must not be disabled together by accident.
/// `WriteOnly` retains the CRC when writing a checkpoint for offline/recovery
/// tooling, but trusts it during normal opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointChecksumPolicy {
    /// Verify the persisted checkpoint CRC32 at every open (the default).
    Full,
    /// Retain the CRC32 on write but skip its read-time verification.
    WriteOnly,
}

impl CheckpointChecksumPolicy {
    /// Select the process-wide checkpoint policy from `MTXDB_CHECKPOINT_CHECKSUM`.
    /// Unknown values intentionally retain the safe default.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("MTXDB_CHECKPOINT_CHECKSUM").as_deref() {
            Ok("writeonly") => Self::WriteOnly,
            Ok("full" | _) | Err(_) => Self::Full,
        }
    }

    #[must_use]
    fn verifies_reads(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Disambiguates concurrent checkpoint tmp filenames within this process,
/// paired with the process id for uniqueness across processes — same pattern
/// as `SHARD_ROOMS_TMP_COUNTER` / `shard::STATS_TMP_COUNTER`.
static CHECKPOINT_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A checkpoint read off disk, validated enough to trust for index rebuild.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Wire-format version; only version 2 bucket snapshots are accepted.
    pub version: u32,
    /// The pack-fingerprint the checkpoint was written against.
    pub fingerprint: u64,
    /// Journal LSN this checkpoint's index snapshot covers, captured atomically
    /// with the snapshot. Zero when the checkpoint was written without a
    /// journal.
    pub covered_lsn: u64,
    /// The highest redo `delta_seq` this checkpoint incorporates.
    pub base_delta_seq: u64,
    /// One entry per collection, in the checkpoint's directory order.
    pub collections: Vec<LoadedCollection>,
    /// Keeps the raw snapshot bytes alive while collections are decoded.
    pub mmap: Arc<Mmap>,
    /// `(writer's local shard slot, pack_id)` for every pack live at
    /// checkpoint-write time. Every `slot` embedded in this checkpoint's
    /// raw index slots (and in any delta frame that continues it) refers to
    /// one of these slots — translate through this table to a reader's own
    /// local slot rather than trusting the slot number directly. See the
    /// `CHECKPOINT_VERSION` doc comment for why.
    pub pack_table: Vec<(u16, PackId)>,
    /// Per-pool logical versions captured through `covered_lsn`, including
    /// IDs of collections deleted after their last live record was written.
    pub logical_versions: Vec<([u8; 16], u64)>,
}

/// One collection's raw slots in the checkpoint mapping.
#[derive(Debug)]
pub struct LoadedCollection {
    /// The collection whose slots these are.
    pub collection_id: [u8; 16],
    /// Checkpoint generation for this collection (reserved for a future
    /// delta-log; always 0 today).
    pub generation: u64,
    /// Absolute byte offset of the encoded bucket snapshot.
    pub slots_offset: usize,
    /// Allocated index capacity.
    pub capacity: u32,
    /// Number of non-empty slots, recorded at checkpoint write time.
    pub slot_count: u32,
    /// Byte length of the encoded bucket snapshot.
    pub snapshot_len: usize,
}

/// Deterministic FNV-1a hash of the `(pack_id, file_len)` set a checkpoint
/// (or an open store's current packs) describes. The set is sorted before
/// hashing so the result is order-independent.
#[must_use]
pub fn pack_fingerprint(packs: &[(PackId, u64)]) -> u64 {
    let mut sorted: Vec<(PackId, u64)> = packs.to_vec();
    sorted.sort_unstable();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (pack_id, file_len) in sorted {
        for byte in pack_id.as_bytes() {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in file_len.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

struct CheckpointLayout {
    collection_count: u32,
    directory_bytes: u64,
    slots_bytes: u64,
    pack_table_count: u32,
    pack_table_bytes: u64,
    logical_version_count: u32,
    logical_version_bytes: u64,
    snapshot_counts: Vec<usize>,
}

fn checkpoint_layout(
    covered_lsn: u64,
    collections: &[([u8; 16], u64, &[u8])],
    pack_table: &[(u16, PackId)],
    logical_versions: &[([u8; 16], u64)],
) -> io::Result<CheckpointLayout> {
    if logical_versions
        .windows(2)
        .any(|pair| pair[0].0 >= pair[1].0)
        || logical_versions
            .iter()
            .any(|(_, version)| *version > covered_lsn)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint logical versions must be sorted, unique, and covered by its LSN",
        ));
    }
    let collection_count = u32::try_from(collections.len())
        .map_err(|_| io::Error::other("too many collections for checkpoint u32"))?;
    let directory_bytes = u64::try_from(collections.len())
        .ok()
        .and_then(|count| count.checked_mul(COLLECTION_DIR_ENTRY_LEN as u64))
        .ok_or_else(|| io::Error::other("checkpoint directory size overflow"))?;
    let mut slots_bytes = 0_u64;
    let mut snapshot_counts = Vec::with_capacity(collections.len());
    for (_, _, blob) in collections {
        let (_, entry_count) = RawBucketTable::validate_snapshot(blob)
            .map_err(|error| io::Error::other(format!("invalid bucket snapshot: {error}")))?;
        let blob_len = u64::try_from(blob.len())
            .map_err(|_| io::Error::other("bucket snapshot length exceeds u64"))?;
        slots_bytes = slots_bytes
            .checked_add(blob_len)
            .ok_or_else(|| io::Error::other("checkpoint snapshot bytes overflow"))?;
        if u32::try_from(blob.len()).is_err() {
            return Err(io::Error::other(
                "bucket snapshot exceeds directory length field",
            ));
        }
        snapshot_counts.push(entry_count);
    }
    let pack_table_count = u32::try_from(pack_table.len())
        .map_err(|_| io::Error::other("too many packs for checkpoint pack table u32"))?;
    let pack_table_bytes = u64::try_from(pack_table.len())
        .ok()
        .and_then(|count| count.checked_mul(PACK_TABLE_ENTRY_LEN as u64))
        .ok_or_else(|| io::Error::other("checkpoint pack table size overflow"))?;
    let logical_version_count = u32::try_from(logical_versions.len())
        .map_err(|_| io::Error::other("too many logical versions for checkpoint u32"))?;
    let logical_version_bytes = u64::try_from(logical_versions.len())
        .ok()
        .and_then(|count| count.checked_mul(COLLECTION_VERSION_ENTRY_LEN as u64))
        .ok_or_else(|| io::Error::other("checkpoint logical-version size overflow"))?;
    Ok(CheckpointLayout {
        collection_count,
        directory_bytes,
        slots_bytes,
        pack_table_count,
        pack_table_bytes,
        logical_version_count,
        logical_version_bytes,
        snapshot_counts,
    })
}

fn encode_checkpoint_body(
    collections: &[([u8; 16], u64, &[u8])],
    pack_table: &[(u16, PackId)],
    logical_versions: &[([u8; 16], u64)],
    layout: &CheckpointLayout,
) -> io::Result<Vec<u8>> {
    let directory_len = usize::try_from(layout.directory_bytes)
        .map_err(|_| io::Error::other("checkpoint directory exceeds address space"))?;
    let slots_len = usize::try_from(layout.slots_bytes)
        .map_err(|_| io::Error::other("checkpoint snapshots exceed address space"))?;
    let pack_len = usize::try_from(layout.pack_table_bytes)
        .map_err(|_| io::Error::other("checkpoint pack table exceeds address space"))?;
    let versions_len = usize::try_from(layout.logical_version_bytes)
        .map_err(|_| io::Error::other("checkpoint logical versions exceed address space"))?;
    let total_len = CHECKPOINT_HEADER_LEN
        .checked_add(directory_len)
        .and_then(|len| len.checked_add(slots_len))
        .and_then(|len| len.checked_add(pack_len))
        .and_then(|len| len.checked_add(versions_len))
        .ok_or_else(|| io::Error::other("checkpoint length exceeds address space"))?;
    let mut body = Vec::with_capacity(total_len);
    body.resize(CHECKPOINT_HEADER_LEN, 0);
    let mut body_offset = 0_u64;
    for ((collection_id, generation, blob), entry_count) in
        collections.iter().zip(&layout.snapshot_counts)
    {
        let entry = CollectionDirEntry {
            collection_id: *collection_id,
            generation: *generation,
            slots_offset: body_offset,
            homes_offset: 0,
            tails_offset: 0,
            capacity: u32::try_from(blob.len())
                .map_err(|_| io::Error::other("bucket snapshot exceeds u32"))?,
            slot_count: u32::try_from(*entry_count)
                .map_err(|_| io::Error::other("bucket snapshot entry count exceeds u32"))?,
        };
        body.extend_from_slice(&entry.encode());
        body_offset = body_offset
            .checked_add(
                u64::try_from(blob.len())
                    .map_err(|_| io::Error::other("bucket snapshot length exceeds u64"))?,
            )
            .ok_or_else(|| io::Error::other("checkpoint body offset overflow"))?;
    }
    for (_, _, blob) in collections {
        body.extend_from_slice(blob);
    }
    for &(slot, pack_id) in pack_table {
        body.extend_from_slice(&PackTableEntry { slot, pack_id }.encode());
    }
    for &(collection_id, last_write_lsn) in logical_versions {
        body.extend_from_slice(
            &CollectionVersionEntry {
                collection_id,
                last_write_lsn,
            }
            .encode(),
        );
    }
    Ok(body)
}

/// Write an MTXI0002 checkpoint atomically from validated bucket snapshots.
///
/// The snapshots and their pack fingerprint must describe the same captured
/// storage state. Each collection's generation is stored in the directory for
/// delta replay. The temporary file is synced before it replaces `path`.
/// The caller must capture the fingerprint and snapshots while holding the
/// collection writer locks so they describe the same storage state.
///
/// # Errors
/// Returns an I/O error if metadata is invalid, a bucket snapshot is malformed,
/// or writing, syncing, or renaming the checkpoint fails.
pub fn write_checkpoint(
    path: &Path,
    fingerprint: u64,
    covered_lsn: u64,
    base_delta_seq: u64,
    collections: &[([u8; 16], u64, &[u8])],
    pack_table: &[(u16, PackId)],
    logical_versions: &[([u8; 16], u64)],
) -> std::io::Result<()> {
    let layout = checkpoint_layout(covered_lsn, collections, pack_table, logical_versions)?;
    let mut buf = encode_checkpoint_body(collections, pack_table, logical_versions, &layout)?;
    let content_crc32 = crc32fast::hash(&buf[CHECKPOINT_HEADER_LEN..]);

    let header = CheckpointHeader {
        magic: CHECKPOINT_MAGIC,
        version: CHECKPOINT_VERSION,
        collection_count: layout.collection_count,
        directory_bytes: layout.directory_bytes,
        slots_bytes: layout.slots_bytes,
        pack_fingerprint: fingerprint,
        content_crc32,
        homes_bytes: 0,
        tails_bytes: 0,
        covered_lsn,
        pack_table_count: layout.pack_table_count,
        pack_table_bytes: layout.pack_table_bytes,
        base_delta_seq,
        logical_version_count: layout.logical_version_count,
        logical_version_bytes: layout.logical_version_bytes,
    };
    buf[..CHECKPOINT_HEADER_LEN].copy_from_slice(&header.encode());

    let unique = CHECKPOINT_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = PathBuf::from(format!(
        "{}.tmp.{}.{unique}",
        path.display(),
        std::process::id()
    ));
    let write_result = (|| -> std::io::Result<()> {
        let mut tmp = fs::File::create(&tmp_path)?;
        tmp.write_all(&buf)?;
        // `sync_data` (fdatasync), not `sync_all` (fsync): per POSIX,
        // fdatasync durably flushes the file's data plus whatever metadata
        // is needed to retrieve it (the extended length included) — that's
        // the whole content/length guarantee this write needs. It skips
        // metadata that's *not* needed to read the data back (mtime/ctime).
        // That's a real distinction, not a claim that all inode metadata is
        // irrelevant here generally (permissions/ownership can still matter
        // operationally) — it just happens that none of the metadata this
        // skips is read back through this temp name: it's renamed over
        // `path` immediately below and never opened by this name again. The
        // caller (`persist_index_checkpoint`) separately fsyncs the base
        // directory after the rename, which is what makes the rename itself
        // durable; if that directory fsync fails, reopen falls back to
        // whichever checkpoint (old or new) is actually intact rather than
        // trusting the rename — this change doesn't alter that fallback.
        tmp.sync_data()
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
    read_checkpoint_with_policy(path, CheckpointChecksumPolicy::from_env())
}

struct CheckpointRegions {
    collection_count: usize,
    slot_base: usize,
    slots_bytes: usize,
    pack_table_base: usize,
    logical_versions_base: usize,
}

fn checkpoint_regions(buf: &[u8], header: &CheckpointHeader) -> Option<CheckpointRegions> {
    let collection_count = usize::try_from(header.collection_count).ok()?;
    if header.directory_bytes
        != u64::try_from(collection_count)
            .ok()?
            .checked_mul(COLLECTION_DIR_ENTRY_LEN as u64)?
        || header.homes_bytes != 0
        || header.tails_bytes != 0
        || header.pack_table_count > u32::from(u16::MAX).saturating_add(1)
        || header.pack_table_bytes
            != u64::from(header.pack_table_count).checked_mul(PACK_TABLE_ENTRY_LEN as u64)?
        || header.logical_version_bytes
            != u64::from(header.logical_version_count)
                .checked_mul(COLLECTION_VERSION_ENTRY_LEN as u64)?
    {
        return None;
    }
    let slot_base =
        CHECKPOINT_HEADER_LEN.checked_add(usize::try_from(header.directory_bytes).ok()?)?;
    let slots_bytes = usize::try_from(header.slots_bytes).ok()?;
    let pack_table_base = slot_base.checked_add(slots_bytes)?;
    let logical_versions_base =
        pack_table_base.checked_add(usize::try_from(header.pack_table_bytes).ok()?)?;
    let file_len =
        logical_versions_base.checked_add(usize::try_from(header.logical_version_bytes).ok()?)?;
    (buf.len() == file_len).then_some(CheckpointRegions {
        collection_count,
        slot_base,
        slots_bytes,
        pack_table_base,
        logical_versions_base,
    })
}

fn read_checkpoint_collections(
    buf: &[u8],
    count: usize,
    slot_base: usize,
    slots_bytes: usize,
    pack_table_base: usize,
) -> Option<Vec<LoadedCollection>> {
    let mut collections = Vec::with_capacity(count);
    let mut body_offset = 0_usize;
    for index in 0..count {
        let entry_offset =
            CHECKPOINT_HEADER_LEN.checked_add(index.checked_mul(COLLECTION_DIR_ENTRY_LEN)?)?;
        let entry_bytes: [u8; COLLECTION_DIR_ENTRY_LEN] = buf
            .get(entry_offset..entry_offset.checked_add(COLLECTION_DIR_ENTRY_LEN)?)?
            .try_into()
            .ok()?;
        let entry = CollectionDirEntry::decode(&entry_bytes)?;
        if entry.homes_offset != 0
            || entry.tails_offset != 0
            || usize::try_from(entry.slots_offset).ok()? != body_offset
        {
            return None;
        }
        let region = slot_base.checked_add(body_offset)?;
        let snapshot_len = usize::try_from(entry.capacity).ok()?;
        let end = region.checked_add(snapshot_len)?;
        // `read_checkpoint_with_policy` has already checked the enclosing
        // body CRC when policy is `Full`. Under `WriteOnly`, both CRCs are
        // intentionally trusted, while snapshot structure is still checked.
        let (bucket_count, entry_count) =
            RawBucketTable::validate_snapshot_with_checksum(buf.get(region..end)?, false).ok()?;
        if u32::try_from(entry_count).ok()? != entry.slot_count || end > pack_table_base {
            return None;
        }
        let capacity = u32::try_from(bucket_count.checked_mul(4)?).ok()?;
        body_offset = body_offset.checked_add(snapshot_len)?;
        collections.push(LoadedCollection {
            collection_id: entry.collection_id,
            generation: entry.generation,
            slots_offset: region,
            capacity,
            slot_count: entry.slot_count,
            snapshot_len,
        });
    }
    (body_offset == slots_bytes).then_some(collections)
}

fn read_checkpoint_pack_table(buf: &[u8], count: usize, base: usize) -> Option<Vec<(u16, PackId)>> {
    let mut pack_table = Vec::with_capacity(count);
    let mut seen_slots = HashSet::with_capacity(count);
    let mut seen_pack_ids = HashSet::with_capacity(count);
    for index in 0..count {
        let start = base.checked_add(index.checked_mul(PACK_TABLE_ENTRY_LEN)?)?;
        let bytes: [u8; PACK_TABLE_ENTRY_LEN] = buf
            .get(start..start.checked_add(PACK_TABLE_ENTRY_LEN)?)?
            .try_into()
            .ok()?;
        let entry = PackTableEntry::decode(&bytes)?;
        if !seen_slots.insert(entry.slot) || !seen_pack_ids.insert(entry.pack_id) {
            return None;
        }
        pack_table.push((entry.slot, entry.pack_id));
    }
    Some(pack_table)
}

fn read_checkpoint_logical_versions(
    buf: &[u8],
    count: usize,
    base: usize,
    covered_lsn: u64,
) -> Option<Vec<([u8; 16], u64)>> {
    let mut versions = Vec::with_capacity(count);
    let mut previous_id = None;
    for index in 0..count {
        let start = base.checked_add(index.checked_mul(COLLECTION_VERSION_ENTRY_LEN)?)?;
        let bytes: [u8; COLLECTION_VERSION_ENTRY_LEN] = buf
            .get(start..start.checked_add(COLLECTION_VERSION_ENTRY_LEN)?)?
            .try_into()
            .ok()?;
        let entry = CollectionVersionEntry::decode(&bytes)?;
        if previous_id.is_some_and(|previous| previous >= entry.collection_id)
            || entry.last_write_lsn > covered_lsn
        {
            return None;
        }
        previous_id = Some(entry.collection_id);
        versions.push((entry.collection_id, entry.last_write_lsn));
    }
    Some(versions)
}

/// As [`read_checkpoint`], with an explicit integrity policy for callers that
/// need deterministic configuration rather than the process environment.
#[must_use]
pub fn read_checkpoint_with_policy(
    path: &Path,
    checksum_policy: CheckpointChecksumPolicy,
) -> Option<LoadedCheckpoint> {
    let file = fs::File::open(path).ok()?;
    let mmap = Arc::new(crate::packfile::map_pack(&file).ok()?);
    let buf: &[u8] = &mmap;
    let header_bytes: [u8; CHECKPOINT_HEADER_LEN] =
        buf.get(..CHECKPOINT_HEADER_LEN)?.try_into().ok()?;
    let header = CheckpointHeader::decode(&header_bytes)?;
    if !supported_checkpoint_header(&header) {
        return None;
    }
    let regions = checkpoint_regions(buf, &header)?;

    // Verify the whole directory+slots body against the header's CRC in one
    // pass, in place of re-deriving each collection's occupancy by walking
    // its slots. This is strictly stronger than the old per-collection walk
    // (it also covers the directory entries — `capacity`/`slot_count`
    // themselves — which the walk never checked), and costs the same O(bytes)
    // as a single SIMD-accelerated hash instead of a manual scan-and-branch
    // loop over every slot of every collection.
    if checksum_policy.verifies_reads() {
        let content_crc32 = crc32fast::hash(buf.get(CHECKPOINT_HEADER_LEN..)?);
        if content_crc32 != header.content_crc32 {
            return None;
        }
    }

    let collections = read_checkpoint_collections(
        buf,
        regions.collection_count,
        regions.slot_base,
        regions.slots_bytes,
        regions.pack_table_base,
    )?;
    let pack_table = read_checkpoint_pack_table(
        buf,
        usize::try_from(header.pack_table_count).ok()?,
        regions.pack_table_base,
    )?;
    let logical_versions = read_checkpoint_logical_versions(
        buf,
        usize::try_from(header.logical_version_count).ok()?,
        regions.logical_versions_base,
        header.covered_lsn,
    )?;

    Some(LoadedCheckpoint {
        version: header.version,
        fingerprint: header.pack_fingerprint,
        covered_lsn: header.covered_lsn,
        base_delta_seq: header.base_delta_seq,
        collections,
        mmap,
        pack_table,
        logical_versions,
    })
}

/// Lightweight read of the persisted checkpoint's `pack_fingerprint`.
///
/// Returns `Ok(Some(fp))` for a valid checkpoint, `Ok(None)` if the file
/// is missing (fresh store), and `Err` if the file exists but is truncated,
/// has invalid magic/version, or an I/O error occurs.
///
/// # Errors
///
/// Returns a [`crate::index::delta::DeltaFingerprintError`] when an existing
/// checkpoint cannot be read or validated.
pub fn read_pack_fingerprint(
    path: &Path,
) -> Result<Option<u64>, crate::index::delta::DeltaFingerprintError> {
    Ok(read_checkpoint_summary(path)?.map(|summary| summary.fingerprint))
}

/// What a checkpoint's header says, without reading its body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointSummary {
    /// The pack fingerprint the checkpoint was written against.
    pub fingerprint: u64,
    /// The journal LSN its index covers, or zero when written without a journal.
    pub covered_lsn: u64,
    /// The highest redo `delta_seq` it incorporates.
    pub base_delta_seq: u64,
}

/// Lightweight read of a checkpoint's header: its `pack_fingerprint` and the
/// journal `covered_lsn` it records.
///
/// Same contract as [`read_pack_fingerprint`]: `Ok(None)` for a missing file,
/// `Err` for a truncated one or one with an invalid magic or version.
///
/// # Errors
///
/// Returns a [`crate::index::delta::DeltaFingerprintError`] when an existing
/// checkpoint cannot be read or validated.
pub fn read_checkpoint_summary(
    path: &Path,
) -> Result<Option<CheckpointSummary>, crate::index::delta::DeltaFingerprintError> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(crate::index::delta::io_error(path, "open checkpoint", e)),
    };
    let mut buf = [0u8; CHECKPOINT_HEADER_LEN];
    let mut handle = file.take(CHECKPOINT_HEADER_LEN as u64);
    handle
        .read_exact(&mut buf)
        .map_err(|e| crate::index::delta::io_error(path, "read checkpoint header", e))?;
    let header = CheckpointHeader::decode(&buf)
        .ok_or_else(|| crate::index::delta::invalid(path, "checkpoint header decode failed"))?;
    if !supported_checkpoint_header(&header) {
        return Err(crate::index::delta::invalid(
            path,
            "invalid checkpoint magic/version",
        ));
    }
    Ok(Some(CheckpointSummary {
        fingerprint: header.pack_fingerprint,
        covered_lsn: header.covered_lsn,
        base_delta_seq: header.base_delta_seq,
    }))
}

/// Result of reading the durable fingerprint.
#[derive(Debug, Clone, Copy)]
pub struct DurableFingerprint {
    /// The durable pack fingerprint (checkpoint fp advanced by delta log tail).
    pub fingerprint: u64,
    /// Whether the delta log ended with a torn tail (expected if writer crashed mid-append).
    /// If true, a refresh may still be needed to pick up the incomplete batch.
    pub torn_tail: bool,
}

/// Read the durable pack fingerprint for a store directory.
///
/// The durable fingerprint is the checkpoint's `pack_fingerprint`, advanced
/// by the validated delta log's `tail_fingerprint` when one exists and its
/// `base_fingerprint` matches the checkpoint.  This detects synced appends
/// that only wrote to the delta log without rewriting the checkpoint.
///
/// Returns `Ok(Some(fp))` for a validated durable fingerprint, `Ok(None)`
/// when there is no checkpoint (fresh store), and `Err` when the checkpoint
/// is corrupt or the delta log cannot be reliably read. On `Err` the caller
/// must treat the state as unknown and force a refresh.
///
/// # Errors
///
/// Returns a [`crate::index::delta::DeltaFingerprintError`] when the delta
/// log exists but cannot be validated or read reliably.
pub fn read_durable_fingerprint(
    base_dir: &Path,
) -> Result<Option<DurableFingerprint>, crate::index::delta::DeltaFingerprintError> {
    let ckpt_path = base_dir.join(INDEX_CHECKPOINT_FILE);
    let Some(ckpt_fp) = read_pack_fingerprint(&ckpt_path)? else {
        return Ok(None);
    };
    let delta_path = base_dir.join(format!(
        "{}.{:016x}",
        crate::index::delta::INDEX_DELTA_FILE,
        ckpt_fp,
    ));
    // Use the lightweight forward parser to avoid decoding every DeltaFrame.
    match crate::index::delta::read_delta_tail_fingerprint(&delta_path) {
        Ok(Some(tail_fp)) if tail_fp.base_fingerprint == ckpt_fp => Ok(Some(DurableFingerprint {
            fingerprint: tail_fp.tail_fingerprint,
            torn_tail: tail_fp.torn_tail,
        })),
        Ok(Some(_) | None) => Ok(Some(DurableFingerprint {
            fingerprint: ckpt_fp,
            torn_tail: false,
        })),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{BucketIndex, IndexConfig};

    fn temp_checkpoint(label: &str) -> PathBuf {
        crate::test_scratch::root().join(format!(
            "mtxdb-checkpoint-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn rewrite_crc(bytes: &mut [u8]) {
        let mut header_bytes: [u8; CHECKPOINT_HEADER_LEN] =
            bytes[..CHECKPOINT_HEADER_LEN].try_into().unwrap();
        let mut header = CheckpointHeader::decode(&header_bytes).unwrap();
        header.content_crc32 = crc32fast::hash(&bytes[CHECKPOINT_HEADER_LEN..]);
        header_bytes.copy_from_slice(&header.encode());
        bytes[..CHECKPOINT_HEADER_LEN].copy_from_slice(&header_bytes);
    }

    fn write_bucket_snapshot_checkpoint(path: &Path, pack_table: &[(u16, PackId)]) {
        let index = BucketIndex::with_config(16, IndexConfig::default());
        let blob = index.serialize();
        write_checkpoint(
            path,
            0x1234,
            10,
            0,
            &[([0xA1; 16], 0, &blob)],
            pack_table,
            &[([0xA1; 16], 7)],
        )
        .unwrap();
    }

    fn write_legacy_v1_empty_checkpoint(path: &Path) {
        let mut bytes = vec![0_u8; CHECKPOINT_HEADER_LEN];
        bytes.extend_from_slice(
            &CollectionDirEntry {
                collection_id: [0xA1; 16],
                generation: 3,
                slots_offset: 0,
                homes_offset: 0,
                tails_offset: 0,
                capacity: 16,
                slot_count: 0,
            }
            .encode(),
        );
        bytes.resize(CHECKPOINT_HEADER_LEN + COLLECTION_DIR_ENTRY_LEN + 16 * 8, 0);
        let header = CheckpointHeader {
            magic: *b"MTXI0001",
            version: 1,
            collection_count: 1,
            directory_bytes: COLLECTION_DIR_ENTRY_LEN as u64,
            slots_bytes: 16 * 8,
            pack_fingerprint: 0x1234,
            content_crc32: crc32fast::hash(&bytes[CHECKPOINT_HEADER_LEN..]),
            homes_bytes: 0,
            tails_bytes: 0,
            covered_lsn: 0,
            pack_table_count: 0,
            pack_table_bytes: 0,
            base_delta_seq: 0,
            logical_version_count: 0,
            logical_version_bytes: 0,
        };
        bytes[..CHECKPOINT_HEADER_LEN].copy_from_slice(&header.encode());
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn checkpoint_slot_range_cannot_extend_into_pack_table() {
        let path = temp_checkpoint("slot-bounds");
        write_bucket_snapshot_checkpoint(&path, &[(0, PackId([0xB2; 16]))]);
        let mut bytes = fs::read(&path).unwrap();
        let header =
            CheckpointHeader::decode(bytes[..CHECKPOINT_HEADER_LEN].try_into().unwrap()).unwrap();
        let directory_at = CHECKPOINT_HEADER_LEN;
        let directory_end = directory_at + COLLECTION_DIR_ENTRY_LEN;
        let mut directory =
            CollectionDirEntry::decode(&bytes[directory_at..directory_end]).unwrap();
        directory.slots_offset = header.slots_bytes;
        bytes[directory_at..directory_end].copy_from_slice(&directory.encode());
        rewrite_crc(&mut bytes);
        fs::write(&path, bytes).unwrap();

        assert!(read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_none());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_only_rejects_a_slot_count_that_understates_the_slots() {
        let path = temp_checkpoint("understated-slot-count");
        let index = BucketIndex::with_config(16, IndexConfig::default());
        index
            .insert_with_identity(&[0xA5; 16], 0, 128, false, |_| Ok::<_, ()>([0xA5; 16]))
            .unwrap()
            .unwrap();
        let blob = index.serialize();
        write_checkpoint(&path, 0x1234, 10, 0, &[([0xA1; 16], 0, &blob)], &[], &[]).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        let directory_at = CHECKPOINT_HEADER_LEN;
        let directory_end = directory_at + COLLECTION_DIR_ENTRY_LEN;
        let mut directory =
            CollectionDirEntry::decode(&bytes[directory_at..directory_end]).unwrap();
        assert_eq!(directory.slot_count, 1);
        directory.slot_count = 0;
        bytes[directory_at..directory_end].copy_from_slice(&directory.encode());
        // Leave the body CRC stale: WriteOnly deliberately skips it, which is
        // exactly the policy this understated count used to slip through.
        fs::write(&path, bytes).unwrap();

        assert!(
            read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::WriteOnly).is_none(),
            "an understated slot_count must not become the loaded index length"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_rejects_duplicate_pack_slots_and_addresses() {
        let path = temp_checkpoint("duplicate-pack-table");
        for packs in [
            [(1, PackId([0xB1; 16])), (1, PackId([0xB2; 16]))],
            [(1, PackId([0xB1; 16])), (2, PackId([0xB1; 16]))],
        ] {
            write_bucket_snapshot_checkpoint(&path, &packs);
            assert!(
                read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_none(),
                "duplicate pack identity must be rejected even with a valid checksum: {packs:?}"
            );
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_rejects_unordered_or_uncovered_collection_versions() {
        let path = temp_checkpoint("invalid-version-table");
        let index = BucketIndex::with_config(16, IndexConfig::default());
        let blob = index.serialize();
        for versions in [
            [([0xA1; 16], 7), ([0xA1; 16], 8)],
            [([0xA2; 16], 7), ([0xA1; 16], 8)],
            [([0xA1; 16], 7), ([0xA2; 16], 11)],
        ] {
            let error = write_checkpoint(
                &path,
                0x1234,
                10,
                0,
                &[([0xA1; 16], 0, &blob)],
                &[],
                &versions,
            )
            .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            // Start with a readable, checksummed table, then model malformed
            // metadata independently of checksum corruption.
            write_checkpoint(
                &path,
                0x1234,
                10,
                0,
                &[([0xA1; 16], 0, &blob)],
                &[],
                &[([0xA1; 16], 7), ([0xA2; 16], 8)],
            )
            .unwrap();
            assert!(read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_some());
            let mut bytes = fs::read(&path).unwrap();
            let table_start = bytes.len() - 2 * COLLECTION_VERSION_ENTRY_LEN;
            for (i, (collection_id, last_write_lsn)) in versions.into_iter().enumerate() {
                let offset = table_start + i * COLLECTION_VERSION_ENTRY_LEN;
                bytes[offset..offset + COLLECTION_VERSION_ENTRY_LEN].copy_from_slice(
                    &CollectionVersionEntry {
                        collection_id,
                        last_write_lsn,
                    }
                    .encode(),
                );
            }
            rewrite_crc(&mut bytes);
            fs::write(&path, bytes).unwrap();
            assert!(read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_none());
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn bucket_checkpoint_restores_a_valid_owned_index() {
        let path = temp_checkpoint("bucket-snapshot");
        write_bucket_snapshot_checkpoint(&path, &[]);
        let loaded = read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).unwrap();
        let collection = &loaded.collections[0];
        assert_eq!(loaded.version, CHECKPOINT_VERSION);
        let snapshot_end = collection.slots_offset + collection.snapshot_len;
        let snapshot = loaded
            .mmap
            .get(collection.slots_offset..snapshot_end)
            .unwrap();
        let index = BucketIndex::deserialize_with_config(snapshot, IndexConfig::default()).unwrap();
        assert!(index.is_empty());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn v1_checkpoint_is_rejected_for_sequential_scan_fallback() {
        let path = temp_checkpoint("legacy-v1");
        write_legacy_v1_empty_checkpoint(&path);
        assert!(read_checkpoint_with_policy(&path, CheckpointChecksumPolicy::Full).is_none());
        let _ = fs::remove_file(path);
    }
}
