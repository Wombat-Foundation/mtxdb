//! The incremental index delta log (`index.delta`).
//!
//! Packfiles stay authoritative and the checkpoint stays a rebuildable
//! acceleration structure; this log is a further acceleration on top of the
//! checkpoint. Between full checkpoint rewrites, a session's index changes are
//! recorded as fixed-width [`DeltaFrame`] records so that `sync()` can append
//! a few kilobyte batch instead of rewriting the whole checkpoint. A crash at
//! any point degrades to the fingerprint gates below and an ordinary rescan.
//!
//! # Trusted gates (all must pass, else the log is ignored and the opener
//! falls back to a full rescan / the next sync rewrites the checkpoint):
//!
//! - The log's `base_fingerprint` must equal the checkpoint it continues.
//! - The log's `tail_fingerprint` must equal the fingerprint of the packs
//!   currently on disk (the state the frames were written against).
//! - Every frame's `generation` must match the checkpoint's recorded
//!   generation for that collection (guards against a delta for a pre-resize
//!   table being applied to a resized/repacked one).
//!
//! # Layout (all little-endian, fixed width — see [`crate::index::format`]):
//!
//! ```text
//!   [DeltaLogHeader 16B]       magic "MDLG" | version | reserved | base_fingerprint
//!   [DeltaBatchHeader 8B]      magic "MDLB" | frame_count
//!   [DeltaFrame * frame_count] fixed 36B records
//!   [DeltaLogTrailer 16B]      magic "DLTR" | reserved | tail_fingerprint
//!   [DeltaBatchHeader ..]*     -- further batches, each count-framed and trailer-terminated
//! ```
//!
//! Framing is arithmetic, not magic-sniffed: a reader advances by exactly
//! `frame_count * DELTA_FRAME_LEN` bytes from each batch header, and only then
//! expects the trailer. A batch header's `frame_count` (not any record's
//! payload bytes) determines where the next record begins — a collection ID
//! that happens to spell a magic value can never misdirect the parser. A torn
//! suffix is exactly "not enough bytes remain for `frame_count` frames plus a
//! trailer", and only complete batches are trusted.

use std::fs;
use std::io::Write as _;
use std::path::Path;

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

/// Hard ceiling on the delta log file size this reader will allocate for.
/// The writer (`storage.rs`) forces a full checkpoint rewrite well before
/// this — its own cap is 8 MiB — so a well-behaved log never approaches this
/// value; it exists purely so a corrupted or maliciously oversized
/// `index.delta` (e.g. a truncated-looking length field, or the file
/// replaced wholesale) can't make an opener allocate an unbounded buffer
/// before any structural validation runs. Generous relative to the writer's
/// own cap so it never rejects a real log.
const MAX_DELTA_LOG_FILE_BYTES: u64 = 64 * 1024 * 1024;

const DELTA_LOG_MAGIC: &[u8; 4] = b"MDLG";
/// Current wire version (see the header's version byte).
///
/// v2 repurposes the trailer's 4 reserved bytes as a CRC32 of the batch's
/// frame bytes, so a batch whose frames were
/// corrupted after being written — but which still happens to pass the
/// fingerprint/generation gates in `storage.rs` — is caught and dropped at
/// read time instead of being replayed as valid index state. A v1 log fails
/// the version check below and falls back to a full rescan, same as any
/// other structurally rejected log.
const DELTA_LOG_VERSION: u8 = 2;
const DELTA_BATCH_MAGIC: &[u8; 4] = b"MDLB";
const DELTA_LOG_TRAILER_MAGIC: &[u8; 4] = b"DLTR";

const BASE_FINGERPRINT_OFFSET: usize = 8;
const TRAILER_FINGERPRINT_OFFSET: usize = 8;
/// Offset of the batch's frame-bytes CRC32 within the trailer's reserved
/// field (bytes 4..8, between the magic and the tail fingerprint).
const TRAILER_CRC_OFFSET: usize = 4;

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
/// the CRC32 of the batch's frame bytes (stored at byte offset 4 of the
/// trailer's reserved field).
#[must_use]
pub fn encode_trailer(tail_fingerprint: u64, crc: u32) -> [u8; DELTA_LOG_TRAILER_LEN] {
    let mut bytes = [0u8; DELTA_LOG_TRAILER_LEN];
    bytes[..4].copy_from_slice(DELTA_LOG_TRAILER_MAGIC);
    bytes[TRAILER_CRC_OFFSET..8].copy_from_slice(&crc.to_le_bytes());
    bytes[TRAILER_FINGERPRINT_OFFSET..16].copy_from_slice(&tail_fingerprint.to_le_bytes());
    bytes
}

/// CRC32 of one batch's frame bytes (the payload between the batch header
/// and trailer), as stored in the trailer's reserved field.
#[must_use]
fn frames_crc(frames_bytes: &[u8]) -> u32 {
    crc32fast::hash(frames_bytes)
}

/// Length in bytes of one fully framed batch carrying `frame_count` frames.
#[must_use]
pub fn batch_len(frame_count: usize) -> Option<usize> {
    DELTA_BATCH_HEADER_LEN
        .checked_add(frame_count.checked_mul(DELTA_FRAME_LEN)?)?
        .checked_add(DELTA_LOG_TRAILER_LEN)
}

/// Read and structurally validate the delta log. Returns `None` for a missing
/// file, a bad header, a log with no complete committed batch (entirely torn),
/// or any other structural inconsistency — the caller treats that as "no
/// delta".
///
/// Batch boundaries come from each batch header's `frame_count`, never from
/// scanning for magic, so no frame payload can redirect the parse. A torn
/// trailing batch (fewer bytes than `frame_count` frames + a trailer) is
/// dropped; everything up to the last complete batch is returned.
#[must_use]
pub fn read_delta_log(path: &Path) -> Option<DeltaLog> {
    // Check the size before allocating: `fs::read` would otherwise size its
    // buffer off an unvalidated on-disk length, letting a corrupted or
    // oversized file exhaust memory before any structural check runs.
    if fs::metadata(path).ok()?.len() > MAX_DELTA_LOG_FILE_BYTES {
        return None;
    }
    let buf = fs::read(path).ok()?;
    if buf.len() < DELTA_LOG_HEADER_LEN {
        return None;
    }
    if &buf[..4] != DELTA_LOG_MAGIC || buf[4] != DELTA_LOG_VERSION {
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
        if frames_crc(frames_bytes) != stored_crc {
            // The frame bytes were corrupted after being written (bit rot,
            // torn/partial write not otherwise caught by the length framing,
            // etc). Trust nothing from this batch onward — same as a torn
            // trailer, everything already accumulated from earlier committed
            // batches is kept, but this and any later batch are dropped
            // rather than replayed as valid index state.
            break;
        }
        for frame_bytes in frames_bytes.chunks_exact(DELTA_FRAME_LEN) {
            frames.push(DeltaFrame::decode(frame_bytes)?);
        }
        tail_fingerprint = Some(u64::from_le_bytes(
            buf[trailer_start.saturating_add(TRAILER_FINGERPRINT_OFFSET)
                ..trailer_start.saturating_add(16)]
                .try_into()
                .ok()?,
        ));
        offset = batch_end;
    }
    Some(DeltaLog {
        base_fingerprint,
        frames,
        tail_fingerprint: tail_fingerprint?,
        // `offset` is the frontier of the last committed batch: the loop only
        // breaks past a committed trailer or before a torn/unparseable tail.
        file_len: u64::try_from(offset).unwrap_or(u64::MAX),
    })
}

/// Append one batch (optional header, then a count-framed batch) to the delta
/// log and fsync it. Returns the number of bytes appended. `write_header` must
/// be `true` when writing to a freshly created (or truncated) file that
/// doesn't yet carry the base header.
///
/// # Errors
/// Returns `io::Error` on any write or fsync failure.
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
    file.write_all(&encode_trailer(tail_fingerprint, frames_crc(&frames_bytes)))?;
    appended = appended.saturating_add(DELTA_LOG_TRAILER_LEN);
    file.sync_all()?;
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
}
