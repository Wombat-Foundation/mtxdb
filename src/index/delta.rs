//! The incremental index delta log (`index.delta`).
//!
//! Packfiles stay authoritative and the checkpoint stays a rebuildable
//! acceleration structure; this log is a further acceleration on top of the
//! checkpoint. Between full checkpoint rewrites, a session's index changes are
//! recorded as fixed-width [`DeltaFrame`] records in v2 logs, or as
//! collection-level operations in v3 logs, so `sync()` can append a small
//! batch instead of rewriting the whole checkpoint. A crash at
//! any point degrades to the fingerprint gates below and an ordinary rescan.
//!
//! # Trusted gates (all must pass, else the log is ignored and the opener
//! falls back to a full rescan / the next sync rewrites the checkpoint):
//!
//! - The log's `base_fingerprint` must equal the checkpoint it continues.
//! - The log's `tail_fingerprint` must equal the fingerprint of the packs
//!   currently on disk (the state the frames were written against).
//! - V2 frame generations must match the checkpoint's recorded generation.
//!   V3 collection snapshots rebase one collection at a new generation, after
//!   which incremental frames must match that generation in file order.
//!
//! # Layout (all little-endian):
//!
//! ```text
//!   [DeltaLogHeader 16B]       magic "MDLG" | version | reserved | base_fingerprint
//!   [DeltaBatchHeader 8B]      magic "MDLB" | frame_count
//!   [v2: DeltaFrame * frame_count] fixed 36B records
//!   [v3: framed collection operations] variable-width records
//!   [DeltaLogTrailer 16B]      magic "DLTR" | reserved | tail_fingerprint
//!   [DeltaBatchHeader ..]*     -- further batches, each count-framed and trailer-terminated
//! ```
//!
//! Framing is count-driven, not magic-sniffed: a reader consumes exactly the
//! number of records named by each batch header, then expects the trailer. A
//! collection ID that happens to spell a magic value can never misdirect the
//! parser. A torn suffix is ignored and only complete batches are trusted.

use std::fs;
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};
use std::{fmt, io};

use super::format::{DeltaFrame, DELTA_FRAME_LEN};

/// File name of the incremental index delta log inside a store's base dir.
pub const INDEX_DELTA_FILE: &str = "index.delta";
/// Bytes in one log header: magic(4) + version(1) + reserved(3) + base fingerprint(8).
pub const DELTA_LOG_HEADER_LEN: usize = 16;
/// Bytes in one batch header: magic(4) + `frame_count`(4). Determines the
/// number of following 36-byte frames arithmetically.
pub const DELTA_BATCH_HEADER_LEN: usize = 8;
/// Bytes in one batch trailer: magic(4) + reserved(4) + tail fingerprint(8).
pub const DELTA_LOG_TRAILER_LEN: usize = 16;

/// Hard ceiling on the delta log file size accepted by the readers. The
/// writer's cap is the same size, large enough for a whole-collection v3
/// snapshot while still bounding replay memory; it exists so a corrupted or maliciously oversized
/// `index.delta` (e.g. a truncated-looking length field, or the file
/// replaced wholesale) can't make an opener allocate an unbounded buffer
/// before any structural validation runs. Generous relative to the writer's
/// own cap so it never rejects a real log.
const MAX_DELTA_LOG_FILE_BYTES: u64 = 256 * 1024 * 1024;

const DELTA_LOG_MAGIC: &[u8; 4] = b"MDLG";
/// Current wire version (see the header's version byte).
///
/// v2 repurposes the trailer's 4 reserved bytes as a CRC32 covering the
/// batch's frame bytes and its tail fingerprint, so a batch corrupted after
/// being written — whether in the frames or just the fingerprint — but which
/// still happens to pass the fingerprint/generation gates in `storage.rs` is
/// caught and dropped at read time instead of being replayed as valid index
/// state. A v1 log fails the version check below and falls back to a full
/// rescan, same as any other structurally rejected log.
const DELTA_LOG_VERSION: u8 = 2;
/// Current v3 delta-log version, with variable-length collection operations.
const DELTA_LOG_VERSION_V3: u8 = 3;
const DELTA_BATCH_MAGIC: &[u8; 4] = b"MDLB";
const DELTA_LOG_TRAILER_MAGIC: &[u8; 4] = b"DLTR";

const BASE_FINGERPRINT_OFFSET: usize = 8;
const TRAILER_FINGERPRINT_OFFSET: usize = 8;
/// Offset of the batch's frame-bytes CRC32 within the trailer's reserved
/// field (bytes 4..8, between the magic and the tail fingerprint).
const TRAILER_CRC_OFFSET: usize = 4;

/// V3 frame kinds. The frame envelope is `kind:u8 | payload_len:u32 | payload | crc32:u32`.
const V3_INCREMENTAL: u8 = 0x01;
const V3_COLLECTION_SNAPSHOT: u8 = 0x02;
const V3_COLLECTION_TOMBSTONE: u8 = 0x03;
const V3_FRAME_HEADER_LEN: usize = 1 + 4;
const V3_FRAME_TRAILER_LEN: usize = 4;
const V3_SNAPSHOT_FIXED_LEN: usize = 16 + 8 + 8 + 4;
const V3_TOMBSTONE_LEN: usize = 16 + 8;

/// One v3 operation in a count-framed delta batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaOperation {
    /// Apply one fixed-shape insertion to a reconstructed collection index.
    Incremental(DeltaFrame),
    /// Create or replace a collection from a complete serialized index.
    /// `order_key` preserves the store's durable collection order even when
    /// the inspection sidecar is unavailable or stale.
    CollectionSnapshot {
        /// Collection being replaced or created.
        collection_id: [u8; 16],
        /// Structural generation represented by `index_blob`.
        generation: u64,
        /// Stable position in `collection_order`.
        order_key: u64,
        /// `LossyIndex::serialize()` bytes; semantic validation occurs during
        /// replay, where the configured index policy is available.
        index_blob: Vec<u8>,
    },
    /// Remove a collection from the replayed checkpoint state.
    CollectionTombstone {
        /// Collection being deleted.
        collection_id: [u8; 16],
        /// Generation observed by the deletion operation.
        generation: u64,
    },
}

/// Encode one self-describing v3 frame. The CRC covers the kind, payload
/// length, and payload, so corruption in either framing or contents is caught.
///
/// # Errors
/// Returns `InvalidInput` if the encoded payload length does not fit `u32`.
pub fn encode_v3_frame(operation: &DeltaOperation) -> io::Result<Vec<u8>> {
    let (kind, payload) = match operation {
        DeltaOperation::Incremental(frame) => (V3_INCREMENTAL, frame.encode().to_vec()),
        DeltaOperation::CollectionSnapshot {
            collection_id,
            generation,
            order_key,
            index_blob,
        } => {
            let blob_len = u32::try_from(index_blob.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "snapshot blob exceeds u32")
            })?;
            let mut payload =
                Vec::with_capacity(V3_SNAPSHOT_FIXED_LEN.saturating_add(index_blob.len()));
            payload.extend_from_slice(collection_id);
            payload.extend_from_slice(&generation.to_le_bytes());
            payload.extend_from_slice(&order_key.to_le_bytes());
            payload.extend_from_slice(&blob_len.to_le_bytes());
            payload.extend_from_slice(index_blob);
            (V3_COLLECTION_SNAPSHOT, payload)
        }
        DeltaOperation::CollectionTombstone {
            collection_id,
            generation,
        } => {
            let mut payload = Vec::with_capacity(V3_TOMBSTONE_LEN);
            payload.extend_from_slice(collection_id);
            payload.extend_from_slice(&generation.to_le_bytes());
            (V3_COLLECTION_TOMBSTONE, payload)
        }
    };
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "delta frame exceeds u32"))?;
    let frame_len = V3_FRAME_HEADER_LEN
        .checked_add(payload.len())
        .and_then(|length| length.checked_add(V3_FRAME_TRAILER_LEN))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "delta frame length overflow")
        })?;
    let mut encoded = Vec::with_capacity(frame_len);
    encoded.push(kind);
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&payload);
    encoded.extend_from_slice(&crc32fast::hash(&encoded).to_le_bytes());
    Ok(encoded)
}

/// Decode one v3 frame from the front of `bytes`, returning its operation and
/// consumed length. Extra bytes belong to subsequent frames in the same batch.
/// Unknown kinds, invalid lengths, or checksum failures return `None`.
#[must_use]
pub fn decode_v3_frame(bytes: &[u8]) -> Option<(DeltaOperation, usize)> {
    let kind = *bytes.first()?;
    let payload_len_bytes: [u8; 4] = bytes.get(1..5)?.try_into().ok()?;
    let payload_len = usize::try_from(u32::from_le_bytes(payload_len_bytes)).ok()?;
    let payload_start = V3_FRAME_HEADER_LEN;
    let payload_end = payload_start.checked_add(payload_len)?;
    let frame_end = payload_end.checked_add(V3_FRAME_TRAILER_LEN)?;
    let payload = bytes.get(payload_start..payload_end)?;
    let stored_crc = u32::from_le_bytes(bytes.get(payload_end..frame_end)?.try_into().ok()?);
    if crc32fast::hash(bytes.get(..payload_end)?) != stored_crc {
        return None;
    }

    let operation = match kind {
        V3_INCREMENTAL => DeltaFrame::decode(payload).map(DeltaOperation::Incremental)?,
        V3_COLLECTION_SNAPSHOT => {
            if payload.len() < V3_SNAPSHOT_FIXED_LEN {
                return None;
            }
            let collection_id = payload.get(..16)?.try_into().ok()?;
            let generation = u64::from_le_bytes(payload.get(16..24)?.try_into().ok()?);
            let order_key = u64::from_le_bytes(payload.get(24..32)?.try_into().ok()?);
            let blob_len =
                usize::try_from(u32::from_le_bytes(payload.get(32..36)?.try_into().ok()?)).ok()?;
            let blob_end = V3_SNAPSHOT_FIXED_LEN.checked_add(blob_len)?;
            if blob_end != payload.len() {
                return None;
            }
            DeltaOperation::CollectionSnapshot {
                collection_id,
                generation,
                order_key,
                index_blob: payload.get(V3_SNAPSHOT_FIXED_LEN..blob_end)?.to_vec(),
            }
        }
        V3_COLLECTION_TOMBSTONE => {
            if payload.len() != V3_TOMBSTONE_LEN {
                return None;
            }
            DeltaOperation::CollectionTombstone {
                collection_id: payload.get(..16)?.try_into().ok()?,
                generation: u64::from_le_bytes(payload.get(16..24)?.try_into().ok()?),
            }
        }
        _ => return None,
    };
    Some((operation, frame_end))
}

/// A decoded delta log that passed its structural gates.
#[derive(Debug)]
pub struct DeltaLog {
    /// The checkpoint fingerprint this log continues from.
    pub base_fingerprint: u64,
    /// Every committed frame in file order (torn trailing batches are absent).
    pub frames: Vec<DeltaFrame>,
    /// The pack fingerprint the final committed batch was written against.
    pub tail_fingerprint: u64,
    /// Byte length of the committed region in the log file (through the last
    /// committed trailer). A session continuing this log appends at this
    /// boundary — it is the answer to "has a base header already been
    /// written", independent of any fresh-session default.
    pub file_len: u64,
    /// Whether a torn tail was encountered after the last committed batch.
    pub torn_tail: bool,
}

/// Encode the fixed-width log header for `base_fingerprint`.
#[must_use]
pub fn encode_header(base_fingerprint: u64) -> [u8; DELTA_LOG_HEADER_LEN] {
    let mut bytes = [0u8; DELTA_LOG_HEADER_LEN];
    bytes[..4].copy_from_slice(DELTA_LOG_MAGIC);
    bytes[4] = DELTA_LOG_VERSION;
    bytes[BASE_FINGERPRINT_OFFSET..16].copy_from_slice(&base_fingerprint.to_le_bytes());
    bytes
}

/// Encode the fixed-width batch header for `frame_count`.
#[must_use]
pub fn encode_batch_header(frame_count: u32) -> [u8; DELTA_BATCH_HEADER_LEN] {
    let mut bytes = [0u8; DELTA_BATCH_HEADER_LEN];
    bytes[..4].copy_from_slice(DELTA_BATCH_MAGIC);
    bytes[4..8].copy_from_slice(&frame_count.to_le_bytes());
    bytes
}

/// Encode the fixed-width batch trailer for `tail_fingerprint`, with `crc`
/// the CRC32 of the batch's frame bytes and `tail_fingerprint` together
/// (stored at byte offset 4 of the trailer's reserved field).
#[must_use]
pub fn encode_trailer(tail_fingerprint: u64, crc: u32) -> [u8; DELTA_LOG_TRAILER_LEN] {
    let mut bytes = [0u8; DELTA_LOG_TRAILER_LEN];
    bytes[..4].copy_from_slice(DELTA_LOG_TRAILER_MAGIC);
    bytes[TRAILER_CRC_OFFSET..8].copy_from_slice(&crc.to_le_bytes());
    bytes[TRAILER_FINGERPRINT_OFFSET..16].copy_from_slice(&tail_fingerprint.to_le_bytes());
    bytes
}

/// CRC32 covering one batch's frame bytes *and* its `tail_fingerprint`, as
/// stored in the trailer's reserved field. Folding the fingerprint into the
/// CRC means fingerprint-only corruption (the frame bytes untouched) is also
/// caught here, rather than relying solely on the fingerprint gate in
/// `storage.rs` matching it against the wrong thing.
#[must_use]
fn batch_crc(frames_bytes: &[u8], tail_fingerprint: u64) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(frames_bytes);
    hasher.update(&tail_fingerprint.to_le_bytes());
    hasher.finalize()
}

/// Validated result of a lightweight delta-log scan.
#[derive(Debug, Clone, Copy)]
pub struct DeltaTailFingerprint {
    /// The `base_fingerprint` from the log header (matches the checkpoint
    /// that this delta continues).
    pub base_fingerprint: u64,
    /// The last committed `tail_fingerprint` from the validated batches.
    pub tail_fingerprint: u64,
    /// Whether a torn tail was encountered after the last committed batch.
    /// `true` means the file ended mid-batch (expected if writer crashed);
    /// `false` means we read to EOF cleanly after a complete batch.
    pub torn_tail: bool,
}

/// Failure while inspecting a delta log's durable fingerprint.
#[derive(Debug)]
pub enum DeltaFingerprintError {
    /// The delta log could not be accessed or read.
    Io {
        /// Delta-log path involved in the operation.
        path: PathBuf,
        /// Operation being performed when the I/O error occurred.
        operation: &'static str,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// The delta log has an invalid or unsupported on-disk structure.
    Invalid {
        /// Delta-log path involved in the validation failure.
        path: PathBuf,
        /// Human-readable validation failure.
        reason: String,
    },
}

impl fmt::Display for DeltaFingerprintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                path,
                operation,
                source,
            } => write!(formatter, "{} {}: {source}", operation, path.display()),
            Self::Invalid { path, reason } => {
                write!(
                    formatter,
                    "invalid delta fingerprint {}: {reason}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for DeltaFingerprintError {}

pub(crate) fn invalid(path: &Path, reason: impl Into<String>) -> DeltaFingerprintError {
    DeltaFingerprintError::Invalid {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn io_error(
    path: &Path,
    operation: &'static str,
    source: io::Error,
) -> DeltaFingerprintError {
    DeltaFingerprintError::Io {
        path: path.to_owned(),
        operation,
        source,
    }
}

fn read_v3_tail_fingerprint(
    file: &mut fs::File,
    path: &Path,
    len: u64,
    header: &[u8; DELTA_LOG_HEADER_LEN],
) -> Result<DeltaTailFingerprint, DeltaFingerprintError> {
    let base_fingerprint = u64::from_le_bytes(
        header[BASE_FINGERPRINT_OFFSET..DELTA_LOG_HEADER_LEN]
            .try_into()
            .map_err(|_| invalid(path, "invalid base fingerprint"))?,
    );
    let mut pos = u64::try_from(DELTA_LOG_HEADER_LEN)
        .map_err(|_| invalid(path, "log header length overflows u64"))?;
    let mut last_tail = None;

    while pos < len {
        let batch_header_end = pos
            .checked_add(u64::try_from(DELTA_BATCH_HEADER_LEN).unwrap_or(u64::MAX))
            .ok_or_else(|| invalid(path, "batch header offset overflow"))?;
        if batch_header_end > len {
            break;
        }
        file.seek(std::io::SeekFrom::Start(pos))
            .map_err(|error| io_error(path, "seek v3 batch header", error))?;
        let mut batch_header = [0u8; DELTA_BATCH_HEADER_LEN];
        if let Err(error) = file.read_exact(&mut batch_header) {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(io_error(path, "read v3 batch header", error));
        }
        if &batch_header[..4] != DELTA_BATCH_MAGIC {
            break;
        }
        let operation_count = u32::from_le_bytes(
            batch_header[4..8]
                .try_into()
                .map_err(|_| invalid(path, "invalid v3 operation count"))?,
        );
        let Some((cursor, mut batch_crc)) =
            scan_v3_frames(file, path, len, batch_header_end, operation_count)?
        else {
            break;
        };
        let trailer_end = cursor
            .checked_add(u64::try_from(DELTA_LOG_TRAILER_LEN).unwrap_or(u64::MAX))
            .ok_or_else(|| invalid(path, "v3 batch trailer boundary overflow"))?;
        if trailer_end > len {
            break;
        }
        file.seek(std::io::SeekFrom::Start(cursor))
            .map_err(|error| io_error(path, "seek v3 batch trailer", error))?;
        let mut trailer = [0u8; DELTA_LOG_TRAILER_LEN];
        file.read_exact(&mut trailer)
            .map_err(|error| io_error(path, "read v3 batch trailer", error))?;
        if &trailer[..4] != DELTA_LOG_TRAILER_MAGIC {
            break;
        }
        let tail_fingerprint = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
        batch_crc.update(&trailer[8..16]);
        let stored_crc = u32::from_le_bytes(trailer[4..8].try_into().unwrap());
        if batch_crc.finalize() != stored_crc {
            break;
        }
        last_tail = Some(tail_fingerprint);
        pos = trailer_end;
    }

    let Some(tail_fingerprint) = last_tail else {
        return Err(invalid(path, "no committed v3 delta batch"));
    };
    Ok(DeltaTailFingerprint {
        base_fingerprint,
        tail_fingerprint,
        torn_tail: pos < len,
    })
}

fn scan_v3_frames(
    file: &mut fs::File,
    path: &Path,
    file_len: u64,
    mut cursor: u64,
    operation_count: u32,
) -> Result<Option<(u64, crc32fast::Hasher)>, DeltaFingerprintError> {
    let mut batch_crc = crc32fast::Hasher::new();
    for _ in 0..operation_count {
        let Some(next) = scan_v3_frame(file, path, file_len, cursor, &mut batch_crc)? else {
            return Ok(None);
        };
        cursor = next;
    }
    Ok(Some((cursor, batch_crc)))
}

fn scan_v3_frame(
    file: &mut fs::File,
    path: &Path,
    file_len: u64,
    cursor: u64,
    batch_crc: &mut crc32fast::Hasher,
) -> Result<Option<u64>, DeltaFingerprintError> {
    let header_len = u64::try_from(V3_FRAME_HEADER_LEN)
        .map_err(|_| invalid(path, "v3 frame header length overflows u64"))?;
    let Some(header_end) = cursor
        .checked_add(header_len)
        .filter(|end| *end <= file_len)
    else {
        return Ok(None);
    };
    file.seek(std::io::SeekFrom::Start(cursor))
        .map_err(|error| io_error(path, "seek v3 frame header", error))?;
    let mut header = [0u8; V3_FRAME_HEADER_LEN];
    file.read_exact(&mut header)
        .map_err(|error| io_error(path, "read v3 frame header", error))?;
    let payload_len = u64::from(u32::from_le_bytes(
        header[1..5]
            .try_into()
            .map_err(|_| invalid(path, "invalid v3 frame length"))?,
    ));
    let payload_end = header_end
        .checked_add(payload_len)
        .ok_or_else(|| invalid(path, "v3 frame payload boundary overflow"))?;
    let checksum_end = payload_end
        .checked_add(u64::try_from(V3_FRAME_TRAILER_LEN).unwrap_or(u64::MAX))
        .ok_or_else(|| invalid(path, "v3 frame length overflow"))?;
    if checksum_end > file_len {
        return Ok(None);
    }
    let fixed_len = match header[0] {
        V3_INCREMENTAL if payload_len == u64::try_from(DELTA_FRAME_LEN).unwrap_or(u64::MAX) => 0,
        V3_COLLECTION_SNAPSHOT
            if payload_len >= u64::try_from(V3_SNAPSHOT_FIXED_LEN).unwrap_or(u64::MAX) =>
        {
            V3_SNAPSHOT_FIXED_LEN
        }
        V3_COLLECTION_TOMBSTONE
            if payload_len == u64::try_from(V3_TOMBSTONE_LEN).unwrap_or(u64::MAX) =>
        {
            0
        }
        _ => return Ok(None),
    };
    let mut frame_crc = crc32fast::Hasher::new();
    frame_crc.update(&header);
    batch_crc.update(&header);
    let mut remaining = payload_len;
    if fixed_len > 0 {
        let mut fixed = [0u8; V3_SNAPSHOT_FIXED_LEN];
        file.read_exact(&mut fixed)
            .map_err(|error| io_error(path, "read v3 snapshot header", error))?;
        let blob_len = u64::from(u32::from_le_bytes(
            fixed[32..36]
                .try_into()
                .map_err(|_| invalid(path, "invalid v3 snapshot length"))?,
        ));
        if u64::try_from(fixed_len)
            .unwrap_or(u64::MAX)
            .checked_add(blob_len)
            != Some(payload_len)
        {
            return Ok(None);
        }
        frame_crc.update(&fixed);
        batch_crc.update(&fixed);
        remaining = remaining.saturating_sub(u64::try_from(fixed_len).unwrap_or(u64::MAX));
    }
    let mut chunk = [0u8; 16 * 1024];
    while remaining > 0 {
        let chunk_size = usize::try_from(remaining.min(chunk.len() as u64)).unwrap_or(chunk.len());
        file.read_exact(&mut chunk[..chunk_size])
            .map_err(|error| io_error(path, "read v3 frame payload", error))?;
        frame_crc.update(&chunk[..chunk_size]);
        batch_crc.update(&chunk[..chunk_size]);
        remaining = remaining.saturating_sub(u64::try_from(chunk_size).unwrap_or(u64::MAX));
    }
    let mut stored_crc = [0u8; V3_FRAME_TRAILER_LEN];
    file.read_exact(&mut stored_crc)
        .map_err(|error| io_error(path, "read v3 frame checksum", error))?;
    if u32::from_le_bytes(stored_crc) != frame_crc.finalize() {
        return Ok(None);
    }
    batch_crc.update(&stored_crc);
    Ok(Some(checksum_end))
}

/// Lightweight forward scan of the delta log's last committed
/// `tail_fingerprint`.
///
/// Walks each batch from the log header forward — reading batch headers,
/// computing frame/trailer boundaries with checked arithmetic, and
/// verifying CRC — retaining the last valid `tail_fingerprint`.  Stops at
/// the first torn or corrupt suffix.  This avoids decoding individual
/// [`DeltaFrame`] records while still validating every batch boundary.
///
/// When a torn trailing batch is encountered the function returns the
/// last _valid_ `tail_fingerprint` (from a prior committed batch), which
/// is the correct durable generation for the committed prefix.  Returns
/// `None` only when the file is missing, too short, has an unrecognized
/// magic/version, or has no committed batches at all. A missing file is
/// returned as `Ok(None)` so callers can fall back to the checkpoint.
///
/// # Errors
///
/// Returns [`DeltaFingerprintError`] when the file cannot be read or its
/// committed structure is invalid.
#[allow(
    clippy::too_many_lines,
    reason = "parses the fixed-width log format in one pass"
)]
pub fn read_delta_tail_fingerprint(
    path: &Path,
) -> Result<Option<DeltaTailFingerprint>, DeltaFingerprintError> {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error(path, "open delta log", error)),
    };
    let len = file
        .metadata()
        .map_err(|error| io_error(path, "stat delta log", error))?
        .len();
    let log_header_len = u64::try_from(DELTA_LOG_HEADER_LEN)
        .map_err(|_| invalid(path, "log header length overflows u64"))?;
    let batch_header_len = u64::try_from(DELTA_BATCH_HEADER_LEN)
        .map_err(|_| invalid(path, "batch header length overflows u64"))?;
    let trailer_len = u64::try_from(DELTA_LOG_TRAILER_LEN)
        .map_err(|_| invalid(path, "trailer length overflows u64"))?;
    let delta_frame_len =
        u64::try_from(DELTA_FRAME_LEN).map_err(|_| invalid(path, "frame length overflows u64"))?;

    // Reject files that exceed the writer's own cap.
    if len > MAX_DELTA_LOG_FILE_BYTES {
        return Err(invalid(path, "file exceeds maximum delta-log size"));
    }

    // --- log header ---
    if len < log_header_len {
        return Err(invalid(path, "truncated log header"));
    }
    let mut log_hdr = [0u8; DELTA_LOG_HEADER_LEN];
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|error| io_error(path, "seek delta log header", error))?;
    file.read_exact(&mut log_hdr)
        .map_err(|error| io_error(path, "read delta log header", error))?;
    if &log_hdr[..4] != DELTA_LOG_MAGIC {
        return Err(invalid(path, "unrecognized magic"));
    }
    if log_hdr[4] == DELTA_LOG_VERSION_V3 {
        return read_v3_tail_fingerprint(&mut file, path, len, &log_hdr).map(Some);
    }
    if log_hdr[4] != DELTA_LOG_VERSION {
        return Err(invalid(path, "unrecognized version"));
    }
    let base_fp = u64::from_le_bytes(
        log_hdr[8..16]
            .try_into()
            .map_err(|_| invalid(path, "invalid base fingerprint"))?,
    );

    let mut pos = log_header_len;
    let mut tail_fingerprint: Option<u64> = None;

    loop {
        // Enough room for a batch header + trailer?
        let batch_header_end = pos
            .checked_add(batch_header_len)
            .ok_or_else(|| invalid(path, "batch header offset overflow"))?;
        if batch_header_end
            .checked_add(trailer_len)
            .ok_or_else(|| invalid(path, "batch trailer boundary overflow"))?
            > len
        {
            break;
        }
        // --- batch header ---
        let mut batch_hdr = [0u8; DELTA_BATCH_HEADER_LEN];
        file.seek(std::io::SeekFrom::Start(pos))
            .map_err(|error| io_error(path, "seek batch header", error))?;
        file.read_exact(&mut batch_hdr)
            .map_err(|error| io_error(path, "read batch header", error))?;
        if &batch_hdr[..4] != DELTA_BATCH_MAGIC {
            return match tail_fingerprint {
                Some(tail_fingerprint) => Ok(Some(DeltaTailFingerprint {
                    base_fingerprint: base_fp,
                    tail_fingerprint,
                    torn_tail: pos < len,
                })),
                None => Err(invalid(
                    path,
                    format!("invalid batch header at offset {pos}"),
                )),
            };
        }
        let frame_count = u64::from(u32::from_le_bytes(
            batch_hdr[4..8]
                .try_into()
                .map_err(|_| invalid(path, "invalid frame count"))?,
        ));
        let frame_bytes_len = frame_count
            .checked_mul(delta_frame_len)
            .ok_or_else(|| invalid(path, "frame byte length overflow"))?;
        let trailer_start = batch_header_end
            .checked_add(frame_bytes_len)
            .ok_or_else(|| invalid(path, "trailer offset overflow"))?;
        if trailer_start
            .checked_add(trailer_len)
            .ok_or_else(|| invalid(path, "trailer boundary overflow"))?
            > len
        {
            // Torn tail: not enough bytes for complete batch + trailer
            break;
        }
        // --- trailer ---
        let mut trailer = [0u8; DELTA_LOG_TRAILER_LEN];
        file.seek(std::io::SeekFrom::Start(trailer_start))
            .map_err(|error| io_error(path, "seek delta trailer", error))?;
        file.read_exact(&mut trailer)
            .map_err(|error| io_error(path, "read delta trailer", error))?;
        if &trailer[..4] != DELTA_LOG_TRAILER_MAGIC {
            break;
        }
        let tfp = u64::from_le_bytes(
            trailer[TRAILER_FINGERPRINT_OFFSET..16]
                .try_into()
                .map_err(|_| invalid(path, "invalid trailer fingerprint"))?,
        );
        let stored_crc = u32::from_le_bytes(
            trailer[TRAILER_CRC_OFFSET..8]
                .try_into()
                .map_err(|_| invalid(path, "invalid trailer CRC"))?,
        );
        // --- frame bytes for CRC ---
        let frames_start = batch_header_end;
        let frames_len_usize = usize::try_from(frame_bytes_len)
            .map_err(|_| invalid(path, "frame bytes do not fit in memory"))?;
        let mut frames = vec![0u8; frames_len_usize];
        file.seek(std::io::SeekFrom::Start(frames_start))
            .map_err(|error| io_error(path, "seek delta frames", error))?;
        file.read_exact(&mut frames)
            .map_err(|error| io_error(path, "read delta frames", error))?;
        if batch_crc(&frames, tfp) != stored_crc {
            if tail_fingerprint.is_some() {
                break;
            }
            return Err(invalid(path, format!("invalid batch CRC at offset {pos}")));
        }
        tail_fingerprint = Some(tfp);
        pos = trailer_start
            .checked_add(trailer_len)
            .ok_or_else(|| invalid(path, "next batch offset overflow"))?;
    }
    // Determine if we stopped due to a torn tail (insufficient bytes for next batch)
    // or because we reached EOF cleanly after a complete batch.
    let torn_tail = pos < len;
    match tail_fingerprint {
        Some(tail_fingerprint) => Ok(Some(DeltaTailFingerprint {
            base_fingerprint: base_fp,
            tail_fingerprint,
            torn_tail,
        })),
        None => Err(invalid(path, "no committed delta batch")),
    }
}

/// Length in bytes of one fully framed batch carrying `frame_count` frames.
#[must_use]
pub fn batch_len(frame_count: usize) -> Option<usize> {
    DELTA_BATCH_HEADER_LEN
        .checked_add(frame_count.checked_mul(DELTA_FRAME_LEN)?)?
        .checked_add(DELTA_LOG_TRAILER_LEN)
}

/// Length in bytes of one framed v3 batch carrying `operations`, or `None` if a
/// payload does not fit the frame's `u32` length or the total overflows `usize`.
#[must_use]
pub fn v3_batch_len(operations: &[DeltaOperation]) -> Option<usize> {
    let mut total = DELTA_BATCH_HEADER_LEN.checked_add(DELTA_LOG_TRAILER_LEN)?;
    for operation in operations {
        let payload_len = match operation {
            DeltaOperation::Incremental(_) => DELTA_FRAME_LEN,
            DeltaOperation::CollectionSnapshot { index_blob, .. } => {
                V3_SNAPSHOT_FIXED_LEN.checked_add(index_blob.len())?
            }
            DeltaOperation::CollectionTombstone { .. } => V3_TOMBSTONE_LEN,
        };
        u32::try_from(payload_len).ok()?;
        let frame_len = V3_FRAME_HEADER_LEN
            .checked_add(payload_len)?
            .checked_add(V3_FRAME_TRAILER_LEN)?;
        total = total.checked_add(frame_len)?;
    }
    Some(total)
}

/// Read and structurally validate a v2 delta log. Returns `None` for a missing
/// file, a bad header, a v3 log (use [`read_delta_log_v3`] for those), a log
/// with no complete committed batch (entirely torn), or any other structural
/// inconsistency — the caller treats that as "no delta".
///
/// Batch boundaries come from each batch header's `frame_count`, never from
/// scanning for magic, so no frame payload can redirect the parse. A torn
/// trailing batch (fewer bytes than `frame_count` frames + a trailer) is
/// dropped; everything up to the last complete batch is returned.
#[must_use]
pub fn read_delta_log(path: &Path) -> Option<DeltaLog> {
    // Check the size before allocating and then read from the *same* handle:
    // `metadata` + a separate `fs::read` would size the buffer off the
    // metadata call, letting the file be swapped or grown between the two so
    // the allocation exceeds the limit. Opening once pins the inode and make
    // the measured length and the bytes read describe the same file; a
    // bounded read from that handle then cannot allocate past
    // `MAX_DELTA_LOG_FILE_BYTES`.
    use std::io::Read;
    let mut file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len > MAX_DELTA_LOG_FILE_BYTES {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    file.read_exact(&mut buf).ok()?;
    if buf.len() < DELTA_LOG_HEADER_LEN {
        return None;
    }
    if &buf[..4] != DELTA_LOG_MAGIC {
        return None;
    }
    if buf[4] == DELTA_LOG_VERSION_V3 {
        // This API returns the v2 fixed-frame representation. V3 callers use
        // `read_delta_log_v3` instead.
        return None;
    }
    if buf[4] != DELTA_LOG_VERSION {
        return None;
    }
    let base_fingerprint = u64::from_le_bytes(buf[BASE_FINGERPRINT_OFFSET..16].try_into().ok()?);

    let mut frames: Vec<DeltaFrame> = Vec::new();
    let mut tail_fingerprint: Option<u64> = None;
    let mut offset = DELTA_LOG_HEADER_LEN;
    while offset < buf.len() {
        if buf.len().saturating_sub(offset) < DELTA_BATCH_HEADER_LEN {
            break;
        }
        if &buf[offset..offset.saturating_add(4)] != DELTA_BATCH_MAGIC {
            break;
        }
        let Some(header_count) = buf.get(offset.saturating_add(4)..offset.saturating_add(8)) else {
            break;
        };
        let frame_count = u32::from_le_bytes(header_count.try_into().ok()?) as usize;
        let Some(batch_len) = batch_len(frame_count) else {
            break;
        };
        let Some(batch_end) = offset.checked_add(batch_len) else {
            break;
        };
        if batch_end > buf.len() {
            // Torn final batch: fewer bytes than frame_count frames + trailer.
            break;
        }
        let frames_start = offset.saturating_add(DELTA_BATCH_HEADER_LEN);
        let trailer_start = batch_end.saturating_sub(DELTA_LOG_TRAILER_LEN);
        if &buf[trailer_start..trailer_start.saturating_add(4)] != DELTA_LOG_TRAILER_MAGIC {
            break;
        }
        let frames_bytes = &buf[frames_start..trailer_start];
        let stored_crc = u32::from_le_bytes(
            buf[trailer_start.saturating_add(TRAILER_CRC_OFFSET)..trailer_start.saturating_add(8)]
                .try_into()
                .ok()?,
        );
        let batch_tail_fingerprint = u64::from_le_bytes(
            buf[trailer_start.saturating_add(TRAILER_FINGERPRINT_OFFSET)
                ..trailer_start.saturating_add(16)]
                .try_into()
                .ok()?,
        );
        if batch_crc(frames_bytes, batch_tail_fingerprint) != stored_crc {
            // The frame bytes or the tail fingerprint were corrupted after
            // being written (bit rot, torn/partial write not otherwise
            // caught by the length framing, etc). Trust nothing from this
            // batch onward — same as a torn trailer, everything already
            // accumulated from earlier committed batches is kept, but this
            // and any later batch are dropped rather than replayed as valid
            // index state.
            break;
        }
        for frame_bytes in frames_bytes.chunks_exact(DELTA_FRAME_LEN) {
            frames.push(DeltaFrame::decode(frame_bytes)?);
        }
        tail_fingerprint = Some(batch_tail_fingerprint);
        offset = batch_end;
    }
    let torn_tail = offset < buf.len();
    Some(DeltaLog {
        base_fingerprint,
        frames,
        tail_fingerprint: tail_fingerprint?,
        // `offset` is the frontier of the last committed batch: the loop only
        // breaks past a committed trailer or before a torn/unparseable tail.
        file_len: u64::try_from(offset).unwrap_or(u64::MAX),
        torn_tail,
    })
}

/// A decoded v3 delta log that passed its structural gates.
#[derive(Debug)]
pub struct DeltaLogV3 {
    /// The checkpoint fingerprint this log continues from.
    pub base_fingerprint: u64,
    /// The pack fingerprint the final committed batch was written against.
    pub tail_fingerprint: u64,
    /// Byte length of the committed region through the last committed trailer.
    pub file_len: u64,
    /// Whether a torn or unparsable tail was encountered after the last batch.
    pub torn_tail: bool,
    /// Every committed operation in file order (torn trailing batches absent).
    pub operations: Vec<DeltaOperation>,
}

/// Read and structurally validate a v3 delta log.
///
/// Unlike v2 the frame region is variable-length, so after the fixed batch
/// header the reader decodes exactly `frame_count` self-describing frames and
/// expects the trailer immediately after the last one — it never scans for
/// magic inside a payload. A frame that fails to decode, a short frame region,
/// or a bad batch CRC after at least one committed batch is treated as the end
/// of the trustworthy prefix; the fingerprint gates in storage reject a log
/// that has lost a later batch. Returns `None` for a missing file, a bad
/// header, the wrong version, or a log with no complete committed batch.
#[must_use]
pub fn read_delta_log_v3(path: &Path) -> Option<DeltaLogV3> {
    use std::io::Read;
    let mut file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len > MAX_DELTA_LOG_FILE_BYTES {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    file.read_exact(&mut buf).ok()?;
    if buf.len() < DELTA_LOG_HEADER_LEN {
        return None;
    }
    if &buf[..4] != DELTA_LOG_MAGIC || buf[4] != DELTA_LOG_VERSION_V3 {
        return None;
    }
    let base_fingerprint = u64::from_le_bytes(buf[BASE_FINGERPRINT_OFFSET..16].try_into().ok()?);

    let mut operations: Vec<DeltaOperation> = Vec::new();
    let mut tail_fingerprint: Option<u64> = None;
    let mut offset = DELTA_LOG_HEADER_LEN;
    while offset < buf.len() {
        let Some(header_end) = offset.checked_add(DELTA_BATCH_HEADER_LEN) else {
            break;
        };
        let Some(header) = buf.get(offset..header_end) else {
            break;
        };
        if &header[..4] != DELTA_BATCH_MAGIC {
            break;
        }
        let frame_count = u32::from_le_bytes(header[4..8].try_into().ok()?) as usize;

        let mut cursor = header_end;
        let mut batch_operations: Vec<DeltaOperation> = Vec::new();
        let mut decoded_all = true;
        for _ in 0..frame_count {
            let Some(remaining) = buf.get(cursor..) else {
                decoded_all = false;
                break;
            };
            let Some((operation, consumed)) = decode_v3_frame(remaining) else {
                decoded_all = false;
                break;
            };
            if consumed == 0 {
                decoded_all = false;
                break;
            }
            batch_operations.push(operation);
            let Some(next) = cursor.checked_add(consumed) else {
                decoded_all = false;
                break;
            };
            cursor = next;
        }
        if !decoded_all {
            break;
        }

        let Some(trailer_end) = cursor.checked_add(DELTA_LOG_TRAILER_LEN) else {
            break;
        };
        let Some(trailer) = buf.get(cursor..trailer_end) else {
            break;
        };
        if &trailer[..4] != DELTA_LOG_TRAILER_MAGIC {
            break;
        }
        let Some(frames) = buf.get(header_end..cursor) else {
            break;
        };
        let stored_crc = u32::from_le_bytes(trailer[TRAILER_CRC_OFFSET..8].try_into().ok()?);
        let batch_tail_fingerprint =
            u64::from_le_bytes(trailer[TRAILER_FINGERPRINT_OFFSET..16].try_into().ok()?);
        if batch_crc(frames, batch_tail_fingerprint) != stored_crc {
            break;
        }
        operations.append(&mut batch_operations);
        tail_fingerprint = Some(batch_tail_fingerprint);
        offset = trailer_end;
    }
    let torn_tail = offset < buf.len();
    Some(DeltaLogV3 {
        base_fingerprint,
        tail_fingerprint: tail_fingerprint?,
        file_len: u64::try_from(offset).unwrap_or(u64::MAX),
        torn_tail,
        operations,
    })
}

/// Append one batch (optional header, then a count-framed batch) to the delta
/// log. Returns the number of bytes appended. `write_header` must be `true`
/// when writing to a freshly created (or truncated) file that doesn't yet
/// carry the base header.
///
/// Deliberately does not fsync. The log is a rebuildable acceleration
/// structure: a crash may lose or tear its tail, which the framing + batch
/// CRC below detect, and the next open then rescans the packfiles. The pack
/// data the frames describe is synced *before* this append (`sync()` writes
/// shards first), so skipping this fsync cannot lose acknowledged data.
///
/// # Errors
/// Returns `io::Error` on any write failure.
pub fn append_batch(
    path: &Path,
    write_header: bool,
    base_fingerprint: u64,
    frames: &[DeltaFrame],
    tail_fingerprint: u64,
) -> std::io::Result<usize> {
    let frame_count = u32::try_from(frames.len())
        .map_err(|_| std::io::Error::other("delta batch exceeds u32 frame count"))?;
    let mut appended = 0usize;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if write_header {
        file.write_all(&encode_header(base_fingerprint))?;
        appended = appended.saturating_add(DELTA_LOG_HEADER_LEN);
    }
    file.write_all(&encode_batch_header(frame_count))?;
    appended = appended.saturating_add(DELTA_BATCH_HEADER_LEN);
    let mut frames_bytes = Vec::with_capacity(frames.len().saturating_mul(DELTA_FRAME_LEN));
    for frame in frames {
        frames_bytes.extend_from_slice(&frame.encode());
    }
    file.write_all(&frames_bytes)?;
    appended = appended.saturating_add(frames_bytes.len());
    file.write_all(&encode_trailer(
        tail_fingerprint,
        batch_crc(&frames_bytes, tail_fingerprint),
    ))?;
    appended = appended.saturating_add(DELTA_LOG_TRAILER_LEN);
    Ok(appended)
}

/// Append one v3 batch of length-prefixed operations (optional header, then a
/// count-framed batch) to the delta log. Returns the number of bytes appended.
///
/// Mirrors [`append_batch`], but the frame region holds self-describing
/// [`DeltaOperation`] frames whose widths vary per operation, so the batch
/// header's operation count (not a fixed frame width) bounds the region. When
/// `write_header` is set, the file gets a v3 log header, starting a fresh v3
/// epoch — only legitimate once a full checkpoint established the new base.
/// Deliberately does not fsync: same rebuildable-acceleration contract as
/// [`append_batch`], so a crash may tear the tail the framing drops.
///
/// # Errors
/// Returns an error if the operation count exceeds `u32` or a frame fails to
/// encode (a payload length beyond `u32`).
pub fn append_v3_batch(
    path: &Path,
    write_header: bool,
    base_fingerprint: u64,
    operations: &[DeltaOperation],
    tail_fingerprint: u64,
) -> std::io::Result<usize> {
    let operation_count = u32::try_from(operations.len())
        .map_err(|_| std::io::Error::other("delta batch exceeds u32 operation count"))?;
    let mut appended = 0usize;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if write_header {
        let mut header = encode_header(base_fingerprint);
        header[4] = DELTA_LOG_VERSION_V3;
        file.write_all(&header)?;
        appended = appended.saturating_add(DELTA_LOG_HEADER_LEN);
    }
    file.write_all(&encode_batch_header(operation_count))?;
    appended = appended.saturating_add(DELTA_BATCH_HEADER_LEN);
    let mut frames_bytes = Vec::new();
    for operation in operations {
        frames_bytes.extend_from_slice(&encode_v3_frame(operation)?);
    }
    file.write_all(&frames_bytes)?;
    appended = appended.saturating_add(frames_bytes.len());
    file.write_all(&encode_trailer(
        tail_fingerprint,
        batch_crc(&frames_bytes, tail_fingerprint),
    ))?;
    appended = appended.saturating_add(DELTA_LOG_TRAILER_LEN);
    Ok(appended)
}

/// Errors that can arise while applying a replayed delta log to a
/// checkpoint-backed index. Any of these means the log is structurally
/// inconsistent with the checkpoint it claims to continue, and the whole log
/// must be rejected (opener falls back to a rescan).
#[derive(Debug, PartialEq, Eq)]
pub enum DeltaReplayError {
    /// A frame targets a bucket outside the checkpoint index's capacity.
    FrameOutOfBounds {
        /// The frame's target bucket.
        bucket: u32,
        /// The checkpoint index's capacity.
        capacity: u32,
    },
    /// Replay requires an owned (materialized) slot array.
    RequiresOwnedIndex,
    /// A frame carried the empty-slot sentinel (`0`). A legitimate insert
    /// never produces one — `record_delta` only ever logs a freshly built,
    /// non-empty `IndexEntry` — so this can only be log corruption or a
    /// structurally invalid frame. Storing it as-is would silently erase
    /// whatever live entry currently occupies that bucket and truncate the
    /// probe chain past it, stranding any entries beyond it.
    EmptySlot {
        /// The frame's target bucket.
        bucket: u32,
    },
    /// A frame's packed entry named a slot at or above
    /// [`crate::shard::MAX_SHARDS`], which no live shard can occupy. A
    /// legitimate writer never produces one (`IndexEntry::new` rejects it), so
    /// this can only be log corruption or a structurally invalid frame.
    /// Accepting it would install an entry the shard scan cannot resolve.
    SlotOutOfRange {
        /// The frame's target bucket.
        bucket: u32,
        /// The decoded slot, at or above `MAX_SHARDS`.
        slot: u16,
    },
}

impl std::fmt::Display for DeltaReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameOutOfBounds { bucket, capacity } => {
                write!(
                    f,
                    "delta frame bucket {bucket} exceeds index capacity {capacity}"
                )
            }
            Self::RequiresOwnedIndex => write!(f, "delta replay requires an owned index"),
            Self::EmptySlot { bucket } => {
                write!(
                    f,
                    "delta frame at bucket {bucket} carries the empty-slot sentinel"
                )
            }
            Self::SlotOutOfRange { bucket, slot } => {
                write!(
                    f,
                    "delta frame at bucket {bucket} names shard slot {slot}, \
                     which is not below MAX_SHARDS"
                )
            }
        }
    }
}

impl std::error::Error for DeltaReplayError {}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn test_frame(bucket: u32, generation: u64) -> DeltaFrame {
        DeltaFrame {
            collection_id: [0x44, 0x4C, 0x54, 0x52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            bucket,
            generation,
            slot: u64::from(bucket).wrapping_mul(7).wrapping_add(1),
        }
    }

    #[test]
    fn header_and_batch_header_and_trailer_round_trip() {
        let header = encode_header(0xDEAD_BEEF);
        assert_eq!(&header[..4], DELTA_LOG_MAGIC);
        assert_eq!(header[4], DELTA_LOG_VERSION);
        assert_eq!(
            u64::from_le_bytes(header[BASE_FINGERPRINT_OFFSET..16].try_into().unwrap()),
            0xDEAD_BEEF
        );

        let batch = encode_batch_header(12);
        assert_eq!(&batch[..4], DELTA_BATCH_MAGIC);
        assert_eq!(u32::from_le_bytes(batch[4..8].try_into().unwrap()), 12);

        let trailer = encode_trailer(0xCAFE_F00D, 0x1234_5678);
        assert_eq!(&trailer[..4], DELTA_LOG_TRAILER_MAGIC);
        assert_eq!(
            u32::from_le_bytes(trailer[TRAILER_CRC_OFFSET..8].try_into().unwrap()),
            0x1234_5678
        );
        assert_eq!(
            u64::from_le_bytes(trailer[TRAILER_FINGERPRINT_OFFSET..16].try_into().unwrap()),
            0xCAFE_F00D
        );
    }

    #[test]
    fn single_batch_round_trips_as_committed() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_single_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        let frames = vec![test_frame(0, 3), test_frame(1, 3), test_frame(2, 3)];
        append_batch(&path, true, 7, &frames, 99).unwrap();

        let log = read_delta_log(&path).expect("valid log reads");
        assert_eq!(log.base_fingerprint, 7);
        assert_eq!(log.tail_fingerprint, 99);
        assert_eq!(log.frames, frames);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn multiple_batches_keep_every_committed_frame() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_multi_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        append_batch(&path, true, 7, &[test_frame(0, 3), test_frame(1, 3)], 11).unwrap();
        append_batch(
            &path,
            false,
            7,
            &[test_frame(2, 3), test_frame(3, 3), test_frame(4, 3)],
            22,
        )
        .unwrap();

        let log = read_delta_log(&path).expect("multi-batch log reads");
        assert_eq!(log.tail_fingerprint, 22, "the final trailer wins");
        assert_eq!(log.frames.len(), 5);
        assert_eq!(log.frames[4], test_frame(4, 3));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn torn_final_batch_is_dropped() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_torn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        // Two committed batches, then a third batch with a torn trailer.
        append_batch(&path, true, 7, &[test_frame(0, 3), test_frame(1, 3)], 11).unwrap();
        append_batch(&path, false, 7, &[test_frame(2, 3), test_frame(3, 3)], 22).unwrap();
        append_batch(&path, false, 7, &[test_frame(4, 3), test_frame(5, 3)], 33).unwrap();
        let full = std::fs::read(&path).unwrap();
        // Cut 4 bytes off the shortest possible third batch (frames fully
        // written, trailer partial).
        let torn_len = full
            .len()
            .saturating_sub(DELTA_LOG_TRAILER_LEN.saturating_sub(4));
        std::fs::write(&path, &full[..torn_len]).unwrap();

        let log = read_delta_log(&path).expect("torn tail must not fail the committed prefix");
        assert_eq!(log.tail_fingerprint, 22);
        assert_eq!(
            log.frames.len(),
            4,
            "frames of the unfinalized batch are dropped"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unfinalized_frames_are_never_stranded() {
        // A complete batch followed by a batch whose frames are fully written
        // but whose trailer is missing entirely must also drop that batch.
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_no_trailer_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();

        // Append a second batch's header + frames but no trailer.
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&encode_batch_header(2)).unwrap();
        file.write_all(&test_frame(1, 3).encode()).unwrap();
        file.write_all(&test_frame(2, 3).encode()).unwrap();
        let _ = file.sync_all();

        let log = read_delta_log(&path).expect("frame-without-trailer batch is dropped, not fatal");
        assert_eq!(log.tail_fingerprint, 11);
        assert_eq!(log.frames.len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupted_frame_bytes_are_rejected_by_crc() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_crc_corrupt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        // A committed batch, then a second batch whose frame bytes are
        // corrupted after being fully written — the length framing alone
        // can't catch this; only the CRC can.
        append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();
        append_batch(&path, false, 7, &[test_frame(1, 3), test_frame(2, 3)], 22).unwrap();
        let mut full = std::fs::read(&path).unwrap();
        // Flip a byte inside the second batch's frame region (well past the
        // first batch's header+frame+trailer).
        let corrupt_at = full.len() - DELTA_LOG_TRAILER_LEN - 1;
        full[corrupt_at] ^= 0xFF;
        std::fs::write(&path, &full).unwrap();

        let log = read_delta_log(&path).expect("first committed batch must still be trusted");
        assert_eq!(log.tail_fingerprint, 11, "corrupted batch must be dropped");
        assert_eq!(log.frames.len(), 1);
        assert_eq!(log.frames[0], test_frame(0, 3));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupted_tail_fingerprint_is_rejected_by_crc() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_fp_corrupt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);
        // A committed batch, then a second batch whose frame bytes are
        // untouched but whose trailer fingerprint is corrupted after being
        // written. The CRC covers the fingerprint too, so this must be
        // caught even though every frame decodes fine.
        append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();
        append_batch(&path, false, 7, &[test_frame(1, 3), test_frame(2, 3)], 22).unwrap();
        let mut full = std::fs::read(&path).unwrap();
        // Flip the last byte of the file: the trailer's tail_fingerprint
        // field, not any frame byte.
        let last = full.len() - 1;
        full[last] ^= 0xFF;
        std::fs::write(&path, &full).unwrap();

        let log = read_delta_log(&path).expect("first committed batch must still be trusted");
        assert_eq!(
            log.tail_fingerprint, 11,
            "batch with corrupted fingerprint must be dropped"
        );
        assert_eq!(log.frames.len(), 1);
        assert_eq!(log.frames[0], test_frame(0, 3));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn entirely_torn_or_missing_log_is_none() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_torn_all_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        // Missing file.
        assert!(read_delta_log(&path).is_none());

        // Header but no complete committed batch (header + frames, no trailer).
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&encode_header(7)).unwrap();
        file.write_all(&encode_batch_header(1)).unwrap();
        file.write_all(&test_frame(0, 3).encode()).unwrap();
        let _ = file.sync_all();
        assert!(read_delta_log(&path).is_none());

        // Bad magic.
        std::fs::write(&path, vec![0xAB; 64]).unwrap();
        assert!(read_delta_log(&path).is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn v3_batch_bytes(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
        let mut frames = Vec::new();
        for operation in operations {
            frames.extend_from_slice(&encode_v3_frame(operation).unwrap());
        }
        let crc = batch_crc(&frames, tail_fingerprint);
        let mut batch = Vec::from(encode_batch_header(
            u32::try_from(operations.len()).unwrap(),
        ));
        batch.extend_from_slice(&frames);
        batch.extend_from_slice(&encode_trailer(tail_fingerprint, crc));
        batch
    }

    fn sample_v3_operations() -> Vec<DeltaOperation> {
        vec![
            DeltaOperation::Incremental(DeltaFrame {
                collection_id: [7; 16],
                bucket: 1,
                generation: 4,
                slot: 99,
            }),
            DeltaOperation::CollectionSnapshot {
                collection_id: [8; 16],
                generation: 5,
                order_key: 1,
                index_blob: vec![1, 2, 3, 4],
            },
            DeltaOperation::CollectionTombstone {
                collection_id: [9; 16],
                generation: 6,
            },
        ]
    }

    fn v3_log(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
        let mut log = Vec::from(encode_header(0xABC));
        log[4] = DELTA_LOG_VERSION_V3;
        log.extend_from_slice(&v3_batch_bytes(operations, tail_fingerprint));
        log
    }

    #[test]
    fn v3_log_round_trips_operations() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_round_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        let operations = sample_v3_operations();
        let mut log = v3_log(&operations[..2], 0x111);
        log.extend_from_slice(&v3_batch_bytes(&operations[2..], 0x222));
        std::fs::write(&path, &log).unwrap();

        let decoded = read_delta_log_v3(&path).expect("valid v3 log reads");
        assert_eq!(decoded.base_fingerprint, 0xABC);
        assert_eq!(decoded.tail_fingerprint, 0x222);
        assert_eq!(decoded.operations, operations);
        assert!(!decoded.torn_tail);
        assert_eq!(decoded.file_len, u64::try_from(log.len()).unwrap());
        let tail = read_delta_tail_fingerprint(&path)
            .unwrap()
            .expect("v3 tail scan succeeds");
        assert_eq!(tail.base_fingerprint, decoded.base_fingerprint);
        assert_eq!(tail.tail_fingerprint, decoded.tail_fingerprint);
        assert_eq!(tail.torn_tail, decoded.torn_tail);
        // The v2 reader must not misread a v3 epoch.
        assert!(read_delta_log(&path).is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn v3_torn_tail_keeps_the_committed_prefix() {
        let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_torn_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        let operations = sample_v3_operations();
        let mut log = v3_log(&operations[..1], 0x111);
        let committed_len = u64::try_from(log.len()).unwrap();
        // A second batch with a header and one frame but no trailer.
        log.extend_from_slice(&encode_batch_header(2));
        log.extend_from_slice(&encode_v3_frame(&operations[1]).unwrap());
        std::fs::write(&path, &log).unwrap();

        let decoded = read_delta_log_v3(&path).expect("committed prefix is trusted");
        assert_eq!(decoded.operations, vec![operations[0].clone()]);
        assert_eq!(decoded.tail_fingerprint, 0x111);
        assert_eq!(decoded.file_len, committed_len);
        assert!(decoded.torn_tail);
        let tail = read_delta_tail_fingerprint(&path)
            .unwrap()
            .expect("tail scan keeps the committed prefix");
        assert_eq!(tail.tail_fingerprint, decoded.tail_fingerprint);
        assert!(tail.torn_tail);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn v3_corrupt_batch_drops_that_batch_and_later() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_v3_corrupt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        let operations = sample_v3_operations();
        let mut log = v3_log(&operations[..1], 0x111);
        let first_batch_end = log.len();
        log.extend_from_slice(&v3_batch_bytes(&operations[1..], 0x222));
        // Corrupt one byte inside the second batch's frame region.
        let corrupt_at = first_batch_end
            .saturating_add(DELTA_BATCH_HEADER_LEN)
            .saturating_add(4);
        log[corrupt_at] ^= 0x20;
        std::fs::write(&path, &log).unwrap();

        let decoded = read_delta_log_v3(&path).expect("first batch is trusted");
        assert_eq!(decoded.operations, vec![operations[0].clone()]);
        assert_eq!(decoded.tail_fingerprint, 0x111);
        assert!(decoded.torn_tail);
        assert_eq!(decoded.file_len, u64::try_from(first_batch_end).unwrap());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn v3_reader_rejects_v2_and_empty_logs() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_v3_reject_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        // A v2 header (the default `encode_header` version) is not a v3 log.
        std::fs::write(&path, encode_header(1)).unwrap();
        assert!(read_delta_log_v3(&path).is_none());

        // A v3 header with no committed batch is not replayable.
        let mut header_only = Vec::from(encode_header(1));
        header_only[4] = DELTA_LOG_VERSION_V3;
        std::fs::write(&path, header_only).unwrap();
        assert!(read_delta_log_v3(&path).is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn v3_frames_round_trip_every_variant() {
        // Incremental, CollectionSnapshot, and CollectionTombstone: each must
        // survive encode -> decode byte-for-byte.
        for operation in sample_v3_operations() {
            let frame = encode_v3_frame(&operation).expect("frame encodes");
            let (decoded, consumed) = decode_v3_frame(&frame).expect("frame decodes");
            assert_eq!(decoded, operation);
            assert_eq!(consumed, frame.len());
        }
    }

    #[test]
    fn v3_frame_decode_consumes_exactly_one_frame() {
        let operations = sample_v3_operations();
        let first = encode_v3_frame(&operations[0]).unwrap();
        let second = encode_v3_frame(&operations[1]).unwrap();
        let mut concatenated = first.clone();
        concatenated.extend_from_slice(&second);

        let (decoded, consumed) = decode_v3_frame(&concatenated).expect("first frame decodes");
        assert_eq!(decoded, operations[0]);
        assert_eq!(consumed, first.len());

        let (decoded_second, second_consumed) =
            decode_v3_frame(&concatenated[consumed..]).expect("second frame decodes");
        assert_eq!(decoded_second, operations[1]);
        assert_eq!(second_consumed, second.len());
    }

    #[test]
    fn v3_frame_decode_rejects_corruption_and_truncation() {
        let operation = DeltaOperation::CollectionSnapshot {
            collection_id: [5; 16],
            generation: 2,
            order_key: 1,
            index_blob: vec![9, 8, 7, 6],
        };
        let frame = encode_v3_frame(&operation).unwrap();

        // Every truncation short of the complete frame must fail, not panic.
        for shorter in [0, 1, V3_FRAME_HEADER_LEN, frame.len().saturating_sub(1)] {
            assert!(decode_v3_frame(&frame[..shorter]).is_none());
        }

        // A flipped payload byte must be caught by the frame CRC.
        let mut corrupt_payload = frame.clone();
        corrupt_payload[V3_FRAME_HEADER_LEN] ^= 0x40;
        assert!(decode_v3_frame(&corrupt_payload).is_none());

        // ...as must a flipped CRC byte.
        let mut corrupt_crc = frame.clone();
        let crc_byte = corrupt_crc.len().saturating_sub(1);
        corrupt_crc[crc_byte] ^= 0x01;
        assert!(decode_v3_frame(&corrupt_crc).is_none());

        // A length field reaching past the buffer must fail, not allocate.
        let mut bad_len = frame.clone();
        bad_len[1..5].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_v3_frame(&bad_len).is_none());

        // A plausible, shorter length has enough bytes for framing, so this
        // specifically proves the CRC covers the length field. The decoder
        // must reject it at checksum validation rather than only at bounds
        // checking.
        let mut bad_len = frame.clone();
        let payload_len = u32::from_le_bytes(frame[1..5].try_into().unwrap());
        bad_len[1..5].copy_from_slice(&payload_len.checked_sub(1).unwrap().to_le_bytes());
        assert!(decode_v3_frame(&bad_len).is_none());

        // An unknown operation kind with a recomputed valid CRC is rejected.
        let mut unknown = frame.clone();
        unknown[0] = 0x7F;
        let crc_len = unknown.len().saturating_sub(V3_FRAME_TRAILER_LEN);
        let crc = crc32fast::hash(&unknown[..crc_len]).to_le_bytes();
        unknown[crc_len..].copy_from_slice(&crc);
        assert!(decode_v3_frame(&unknown).is_none());
    }

    #[test]
    fn v3_frame_encoded_sizes_are_exact() {
        let incremental = DeltaOperation::Incremental(test_frame(3, 5));
        assert_eq!(
            encode_v3_frame(&incremental).unwrap().len(),
            V3_FRAME_HEADER_LEN + 36 + V3_FRAME_TRAILER_LEN,
            "incremental framing is header + fixed payload + crc"
        );

        let snapshot = DeltaOperation::CollectionSnapshot {
            collection_id: [0x11; 16],
            generation: 7,
            order_key: 42,
            index_blob: vec![0xAB; 128],
        };
        assert_eq!(
            encode_v3_frame(&snapshot).unwrap().len(),
            V3_FRAME_HEADER_LEN + V3_SNAPSHOT_FIXED_LEN + 128 + V3_FRAME_TRAILER_LEN,
            "snapshot framing is header + fixed fields + blob + crc"
        );

        let tombstone = DeltaOperation::CollectionTombstone {
            collection_id: [0x22; 16],
            generation: 9,
        };
        assert_eq!(
            encode_v3_frame(&tombstone).unwrap().len(),
            V3_FRAME_HEADER_LEN + V3_TOMBSTONE_LEN + V3_FRAME_TRAILER_LEN,
            "tombstone framing is header + fixed fields + crc"
        );
    }

    #[test]
    fn v3_snapshot_rejects_blob_length_mismatch() {
        // Valid framing and CRC whose advertised index_blob length runs past
        // the payload. The existing corruption test covers a bad *framing*
        // length; this covers the separate blob length field inside a snapshot.
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x11; 16]);
        payload.extend_from_slice(&7_u64.to_le_bytes());
        payload.extend_from_slice(&42_u64.to_le_bytes());
        payload.extend_from_slice(&128_u32.to_le_bytes());
        payload.extend_from_slice(&[0u8; 8]);
        let mut frame = Vec::new();
        frame.push(V3_COLLECTION_SNAPSHOT);
        frame.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
        assert!(
            decode_v3_frame(&frame).is_none(),
            "declared blob length must match the payload exactly"
        );
    }

    #[test]
    fn v3_append_batch_round_trips_through_the_reader() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_delta_v3_append_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(INDEX_DELTA_FILE);

        let first = sample_v3_operations();
        let mut expected_len = DELTA_LOG_HEADER_LEN
            .saturating_add(DELTA_BATCH_HEADER_LEN)
            .saturating_add(DELTA_LOG_TRAILER_LEN);
        for operation in &first {
            expected_len = expected_len.saturating_add(encode_v3_frame(operation).unwrap().len());
        }
        let appended = append_v3_batch(&path, true, 0xABC, &first, 0x111).expect("append v3 batch");
        assert_eq!(appended, expected_len, "append reports the bytes it wrote");

        let second = [DeltaOperation::CollectionTombstone {
            collection_id: [3; 16],
            generation: 4,
        }];
        append_v3_batch(&path, false, 0xABC, &second, 0x222).expect("continue the v3 log");

        let decoded = read_delta_log_v3(&path).expect("appended v3 log reads");
        assert_eq!(decoded.base_fingerprint, 0xABC);
        assert_eq!(decoded.tail_fingerprint, 0x222);
        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(decoded.operations, expected);
        assert!(!decoded.torn_tail);
        assert_eq!(decoded.file_len, fs::metadata(&path).unwrap().len());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
