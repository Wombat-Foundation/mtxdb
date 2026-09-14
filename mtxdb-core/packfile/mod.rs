/// Physical, on-disk layout scanning — see [`layout::physical_layout`].
pub mod layout;
/// The [`PackfileStorage`](storage::PackfileStorage) engine and its supporting types.
pub mod storage;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, Write};
use std::path::Path;

use bytes::Bytes;

/// Magic bytes identifying an mtxdb packfile: "MTDB"
pub const MAGIC: [u8; 4] = *b"MTDB";

/// Packfile format version byte following `MAGIC` in the header (see
/// [`write_header`]/[`read_header`]).
///
/// Version 4: replaces the v3 `shard_id`/`epoch` header fields with a
/// single `pack_id: u64` — the pool-local monotonic identity for the
/// shard. Adds `features: u32`. Filename changes from
/// `shard_{slot:04x}_{epoch:016x}.pack` to `shard_{pack_id:016x}.pack`.
/// Hard cutover: v3 files are rejected at open.
pub const VERSION: u8 = 0x04;

/// Total reserved header size in bytes: every shard file's first record
/// starts at exactly this offset. One 4KiB page — ample collection for the
/// descriptor fields plus future growth, with no benefit to a larger
/// reservation (see [`write_header`] for the field layout).
pub const HEADER_LEN: usize = 4096;

/// Byte layout within the reserved header, up to where the CRC starts.
/// Everything from `CRC_COVERED_LEN` to `HEADER_LEN` is the CRC itself
/// (4 bytes) followed by zero padding.
const CRC_COVERED_LEN: usize = 4 // magic
    + 1 // version
    + 4 // header_len (u32)
    + 8 // pack_id (u64)
    + 8 // created_at (u64, unix seconds)
    + 4; // feature_flags (u32, reserved)

/// A shard's immutable descriptor, parsed from its reserved header.
///
/// Recording `pack_id` in the file itself (not just its filename) lets a
/// reader detect a shard file that's been copied or renamed
/// inconsistently — the two should always agree, and a mismatch means
/// something outside mtxdb moved this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardHeader {
    /// Pool-local monotonic pack ID. The shard's permanent external identity.
    pub pack_id: u64,
    /// Unix-seconds creation timestamp.
    pub created_at: u64,
}

/// Maximum record size (64KB). Bounds the *uncompressed* on-disk frame
/// length (`FRAME_FIXED_LEN + data.len()`, i.e. what the frame would
/// take up were it stored raw) — checked at write time against the
/// caller's plaintext `data`, and at read time against the actual on-disk
/// frame length, which never exceeds the uncompressed bound (compression
/// is only ever used when it shrinks a frame; see [`write_record`]).
/// Reject anything larger during recovery scan.
pub const MAX_RECORD_LEN: u32 = 64 * 1024;

/// Per-frame flag: `data` is zstd-compressed on disk; decompress to
/// `uncompressed_len` bytes before returning it to the caller.
///
/// Public because inspection/recovery tooling needs to interpret the flag
/// bits of arbitrary frames and `shard.rs`'s `mmap`-based inline frame parser
/// (which duplicates this module's frame layout for zero-copy reads) shares
/// the same flag bit instead of hardcoding it.
pub const FLAG_COMPRESSED: u8 = 0x01;

/// Per-frame flag: the 4-byte checksum field holds zeros, not a real CRC32,
/// and must not be verified by any reader. Written when the store was
/// configured with [`ChecksumPolicy::Disabled`] — the flag, not the checksum
/// value, is what signals "don't verify", because a real CRC32 can also
/// legitimately be zero and a Full-mode reader must never treat a
/// CRC-disabled frame as corrupt.
pub const FLAG_CRC_DISABLED: u8 = 0x02;

/// How much of the per-frame CRC32 this store writes and verifies.
///
/// Frames are always *formatted* with a 4-byte checksum field (the framing
/// is fixed); the policy controls whether that field is a real checksum and
/// whether readers check it:
///
/// * `Full` (default) — every written record gets a CRC32, every read
///   verifies it. Catches torn writes mid-file and silent bit-rot per record.
/// * `WriteOnly` — records keep their CRC32 (so scanning/recovery tooling
///   can still verify them), but point-lookup reads skip the hashing pass. A
///   corrupt-but-plausibly-framed record may be returned as data.
/// * `Disabled` — records are written with a zero checksum and the
///   [`FLAG_CRC_DISABLED`] flag; nothing is computed or verified anywhere.
///   Every such frame is permanently unverifiable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChecksumPolicy {
    /// Every written record gets a CRC32 and every read verifies it (default).
    Full,
    /// Records keep their CRC32, but point-lookup reads skip hashing them.
    WriteOnly,
    /// Records are written with a zero checksum and [`FLAG_CRC_DISABLED`];
    /// nothing is computed or verified anywhere.
    Disabled,
}

impl ChecksumPolicy {
    /// Whether frames written under this policy carry a real CRC32.
    #[must_use]
    pub fn computes_checksum(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Whether point-lookup reads under this policy hash and compare the CRC.
    #[must_use]
    pub fn verifies_reads(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Byte length of the fixed part of a v3 frame's payload, i.e. everything
/// between the length prefix and the node bytes: 1-byte flags + 4-byte
/// `uncompressed_len` + 16-byte `collection_id` + 16-byte `hash`.
pub(crate) const FRAME_FIXED_LEN: u32 = 1 + 4 + 16 + 16;

/// Maximum plaintext node payload in one frame. This is lower than
/// [`MAX_RECORD_LEN`] because the fixed v3 frame fields consume space too.
pub const MAX_DATA_LEN: u32 = MAX_RECORD_LEN - FRAME_FIXED_LEN;

/// A scanned `(collection_id, hash, file_offset)` entry from a packfile.
pub type ScanEntry = ([u8; 16], [u8; 16], u64);

/// A single record in the packfile.
///
/// Frame layout on disk (v3):
/// ```text
/// [u32 len]              — byte length of everything below down to (not including) crc32, little-endian
/// [u8 flags]             — bit 0: node bytes are zstd-compressed
/// [u32 uncompressed_len] — plaintext length of `data`; equals the node bytes' own length when flags == 0
/// [16-byte collection_id]      — collection this record belongs to (for shard scan recovery)
/// [16-byte hash]         — structural hash (index-rebuild metadata only, NOT for verification)
/// [node bytes]           — opaque node payload, zstd-compressed iff flags bit 0 is set
/// [u32 crc32]            — CRC32 covering len + flags + uncompressed_len + collection_id + hash + node_bytes
/// ```
///
/// A frame is only ever written compressed when doing so makes it
/// smaller — otherwise `data` is stored raw with `flags == 0` — so the
/// on-disk frame length never exceeds the plaintext one.
///
/// **Security invariant:** The framed hash is index-rebuild metadata only.
/// Verification always compares against the *caller-requested* hash, never
/// against the hash stored in the frame. The CRC covers the bytes as
/// written (compressed, when compressed) — it's verified *before*
/// decompression is attempted, so a corrupt frame is never fed to the
/// decompressor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The collection this record belongs to.
    pub collection_id: [u8; 16],
    /// The structural hash framed alongside the record (index-rebuild metadata only).
    pub hash: [u8; 16],
    /// The opaque node payload.
    pub data: Bytes,
}

/// Read a record length prefix, distinguishing clean EOF from a torn prefix.
///
/// A zero-byte read is the normal end of an append-only pack. Once even one
/// byte of the four-byte prefix exists, however, the frame is incomplete and
/// must surface as `UnexpectedEof` so recovery can truncate it before a later
/// append would land after corrupt bytes.
fn read_frame_len_prefix(reader: &mut impl Read) -> io::Result<Option<[u8; 4]>> {
    let mut len_buf = [0u8; 4];
    let mut bytes_read = 0;
    while bytes_read < 4 {
        let n = reader.read(&mut len_buf[bytes_read..])?;
        if n == 0 {
            if bytes_read == 0 {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "torn length prefix",
            ));
        }
        bytes_read = bytes_read.saturating_add(n);
    }
    Ok(Some(len_buf))
}

impl Record {
    /// Upper bound on this record's on-disk frame size (length prefix,
    /// fixed fields, plaintext node bytes, and CRC) — i.e. the size were
    /// it written uncompressed. The actual on-disk size after
    /// [`write_record`] may be smaller when the payload compresses;
    /// callers that need the exact size use `write_record`'s return
    /// value instead. Useful as a safe capacity estimate before writing.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        4_usize
            .wrapping_add(FRAME_FIXED_LEN as usize)
            .wrapping_add(self.data.len())
            .wrapping_add(4)
    }
}

/// Scaffold `Mmap::map`, isolating the single unsafe call in this crate.
///
/// # Safety
/// This is the only `unsafe` block in the codebase. The mapping it creates
/// is sound for the following reasons:
///
/// 1. **Append-only file growth.** A shard file only ever grows by appends
///    that serialize through `put()`. `read_at` remaps when a read lands
///    beyond the current mapping and the file has grown, so the mapping is
///    never stale relative to the data being read.
/// 2. **No truncation or size change.** A mapped `File` is never
///    `set_len`-shrunk or extended after the mapping is created.
/// 3. **Stable file descriptor.** The `Shard` struct owns the only `File`
///    handle and is kept alive by `Arc`s, so the descriptor remains valid
///    for the mapping's lifetime.
///
/// # Errors
/// Returns `io::Error` if the file cannot be memory-mapped.
#[allow(unsafe_code)]
pub fn map_pack(file: &File) -> io::Result<memmap2::Mmap> {
    // SAFETY: See the `map_pack` doc comment — append-only file, remapped on
    // growth, never truncated or shrunk, and a stable fd for the mapping's
    // lifetime.
    unsafe { memmap2::Mmap::map(file) }
}

/// zstd compression level used for record payloads. A low level: packfile
/// frames are small (<= [`MAX_RECORD_LEN`]) and written on the hot append
/// path, so this favors write throughput over squeezing out the last few
/// percent of ratio.
const ZSTD_LEVEL: i32 = 3;

/// Write a single record into the packfile, compressing its payload with
/// zstd when doing so makes the frame smaller (falls back to storing it
/// raw otherwise — e.g. already-compressed or very small payloads often
/// don't shrink).
///
/// # Errors
/// Returns `io::Error` on write failure, or if `record.data` would make
/// even an uncompressed frame exceed [`MAX_RECORD_LEN`] (checked against
/// the plaintext on-disk frame length — `FRAME_FIXED_LEN` plus
/// `record.data.len()` — before compression, so this bound is never
/// looser than what [`read_record`] will actually accept).
///
/// # Panics
/// Panics if the record's fixed fields plus plaintext data exceed `u32::MAX`.
pub fn write_record(writer: &mut impl Write, record: &Record) -> io::Result<u64> {
    write_record_with_options(writer, record, true)
}

/// Like [`write_record`], but lets the caller skip the zstd attempt
/// entirely via `compress: false`.
///
/// Some pools (e.g. HAMT nodes/roots, whose bytes are dense structural
/// hashes rather than text) never shrink under zstd — the compressor
/// still has to run its full match-finding pass before falling back to
/// raw storage, so a pool that never benefits pays that cost on every
/// single write for nothing. `compress: false` skips the attempt
/// altogether and always stores the payload raw. This never changes the
/// on-disk frame format: a `false` caller just always takes the "doesn't
/// shrink" branch that `compress: true` already falls back to at read
/// time, so raw and compressed frames continue to coexist exactly as
/// documented on [`Record`].
///
/// # Errors
/// Same as [`write_record`].
///
/// # Panics
/// Same as [`write_record`].
pub fn write_record_with_options(
    writer: &mut impl Write,
    record: &Record,
    compress: bool,
) -> io::Result<u64> {
    let buf = encode_record_with_options(record, compress, true)?;
    writer.write_all(&buf)?;
    Ok(buf.len() as u64)
}

/// Encode one record into its complete on-disk frame.
///
/// Kept separate from [`write_record_with_options`] so appenders which use
/// positioned writes can retain their file cursor and avoid an `lseek` per
/// record. The result includes the length prefix and checksum. When
/// `write_checksum` is false the frame is emitted with a zero checksum and the
/// [`FLAG_CRC_DISABLED`] flag, per [`ChecksumPolicy::Disabled`].
///
/// # Errors
/// Returns `io::Error` if the payload is too large to frame.
pub(crate) fn encode_record_with_options(
    record: &Record,
    compress: bool,
    write_checksum: bool,
) -> io::Result<Vec<u8>> {
    let uncompressed_len =
        u32::try_from(record.data.len()).expect("record payload exceeds u32::MAX");
    let plaintext_frame_len = FRAME_FIXED_LEN
        .checked_add(uncompressed_len)
        .expect("record frame length exceeds u32::MAX");
    if plaintext_frame_len > MAX_RECORD_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("record payload too large: {plaintext_frame_len} > {MAX_RECORD_LEN}"),
        ));
    }

    let compressed = compress
        .then(|| zstd::bulk::compress(&record.data, ZSTD_LEVEL).ok())
        .flatten();
    let (base_flags, node_bytes): (u8, &[u8]) = match &compressed {
        Some(c) if c.len() < record.data.len() => (FLAG_COMPRESSED, c.as_slice()),
        _ => (0, &record.data),
    };
    let flags = if write_checksum {
        base_flags
    } else {
        base_flags | FLAG_CRC_DISABLED
    };

    let frame_len = FRAME_FIXED_LEN
        .checked_add(u32::try_from(node_bytes.len()).expect("bounded by MAX_RECORD_LEN"))
        .expect("bounded by MAX_RECORD_LEN");
    let total_len = 4_u64.wrapping_add(u64::from(frame_len)).wrapping_add(4);

    // Assemble the whole frame in one buffer and issue a single
    // `write_all`, rather than one syscall per field. `writer` is
    // usually a raw `File` clone (shard files are written directly,
    // not through a `BufWriter`), so 7 separate small writes meant 7
    // separate `write()` syscalls per record — real, avoidable
    // overhead on the hottest path in the engine, unrelated to actual
    // disk speed.
    let mut buf = Vec::with_capacity(usize::try_from(total_len).unwrap_or(0));
    buf.extend_from_slice(&frame_len.to_le_bytes());
    buf.push(flags);
    buf.extend_from_slice(&uncompressed_len.to_le_bytes());
    buf.extend_from_slice(&record.collection_id);
    buf.extend_from_slice(&record.hash);
    buf.extend_from_slice(node_bytes);

    // CRC covers len + flags + uncompressed_len + collection_id + hash + node_bytes,
    // i.e. the bytes as written to disk (compressed, when compressed) —
    // exactly `buf`'s contents so far, hashed in one pass since CRC32
    // over one contiguous buffer is identical to the same bytes hashed
    // via several `update` calls. Under a period of `write_checksum ==
    // false` the field is written as zeros (with FLAG_CRC_DISABLED set
    // above) and no hashing pass happens at all.
    let checksum = if write_checksum {
        let mut crc = crc32fast::Hasher::new();
        crc.update(&buf);
        crc.finalize()
    } else {
        0
    };
    buf.extend_from_slice(&checksum.to_le_bytes());

    debug_assert_eq!(
        buf.len() as u64,
        total_len,
        "encoded frame length must match its length prefix"
    );
    Ok(buf)
}

/// Read a single record from the packfile. Returns `None` on EOF.
///
/// # Errors
/// Returns `io::Error` on read failure or `io::ErrorKind::InvalidData`
/// if the record length is invalid, an unrecognized flag bit is set, the
/// CRC check fails, or (for a compressed frame) decompression fails or
/// yields a length other than the framed `uncompressed_len`.
///
/// # Panics
/// Panics if the payload length does not fit in `usize` (always true on
/// 64-bit targets where `usize >= 32 bits`).
pub fn read_record(reader: &mut impl Read) -> io::Result<Option<Record>> {
    let Some(len_buf) = read_frame_len_prefix(reader)? else {
        return Ok(None);
    };

    let frame_len = u32::from_le_bytes(len_buf);
    if !(FRAME_FIXED_LEN..=MAX_RECORD_LEN).contains(&frame_len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid record length: {frame_len}"),
        ));
    }

    let mut payload = vec![0u8; usize::try_from(frame_len).expect("u32 always fits in usize")];
    reader.read_exact(&mut payload)?;

    let mut crc_buf = [0u8; 4];
    reader.read_exact(&mut crc_buf)?;

    let flags = payload[0];
    if flags & !(FLAG_COMPRESSED | FLAG_CRC_DISABLED) != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported record flags: {flags:#04x}"),
        ));
    }

    // Verify CRC over the bytes as written (compressed, if compressed) —
    // before any decompression is attempted, so a corrupt frame is never
    // fed to the decompressor. Frames carrying FLAG_CRC_DISABLED hold a
    // zero checksum and are skipped, not compared.
    if flags & FLAG_CRC_DISABLED == 0 {
        let mut crc = crc32fast::Hasher::new();
        crc.update(&len_buf);
        crc.update(&payload);
        let expected = crc.finalize();
        let actual = u32::from_le_bytes(crc_buf);
        if expected != actual {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CRC mismatch: expected {expected:08x}, got {actual:08x}"),
            ));
        }
    }

    let uncompressed_len = u32::from_le_bytes(payload[1..5].try_into().unwrap());
    let mut collection_id = [0u8; 16];
    collection_id.copy_from_slice(&payload[5..21]);
    let mut hash = [0u8; 16];
    hash.copy_from_slice(&payload[21..37]);
    let node_bytes = &payload[37..];

    let data = if flags & FLAG_COMPRESSED != 0 {
        if uncompressed_len > MAX_DATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("framed uncompressed_len too large: {uncompressed_len} > {MAX_DATA_LEN}"),
            ));
        }
        let decompressed = zstd::bulk::decompress(
            node_bytes,
            usize::try_from(uncompressed_len).expect("checked above"),
        )
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("zstd decompress failed: {e}"),
            )
        })?;
        if decompressed.len() != usize::try_from(uncompressed_len).expect("checked above") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "decompressed length {} != framed uncompressed_len {uncompressed_len}",
                    decompressed.len()
                ),
            ));
        }
        Bytes::from(decompressed)
    } else {
        if usize::try_from(uncompressed_len).expect("u32 always fits in usize") != node_bytes.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw node length {} != framed uncompressed_len {uncompressed_len}",
                    node_bytes.len()
                ),
            ));
        }
        Bytes::copy_from_slice(node_bytes)
    };

    Ok(Some(Record {
        collection_id,
        hash,
        data,
    }))
}

/// A frame's `collection_id`/`hash` metadata, without its node payload — what
/// [`scan_packfile`]/[`scan_packfile_from`]/[`scan_and_recover_packfile`]
/// actually need. See [`read_record_metadata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordMetadata {
    /// The collection this record belongs to.
    pub collection_id: [u8; 16],
    /// The structural hash framed alongside the record.
    pub hash: [u8; 16],
}

/// Buffer size for streaming node bytes through [`read_record_metadata`]
/// without allocating a buffer proportional to the frame's payload size.
/// Large enough to keep syscall/CRC-update overhead low, small enough to
/// stay a stack buffer.
const SCAN_DISCARD_BUF_LEN: usize = 8192;

/// Read one frame's metadata (`collection_id`, `hash`) without allocating,
/// decompressing, or otherwise materializing its node payload — the scan
/// path's counterpart to [`read_record`], which fully decodes a frame for
/// callers that actually need its data.
///
/// Node bytes are streamed through a small fixed buffer and fed into the
/// running CRC as they're read, then discarded — this still detects
/// corruption in the payload region (the CRC covers the same bytes
/// [`read_record`] verifies), it just never buffers or decompresses them.
/// Deliberately a streaming *read*, not a `Seek` past the payload: seeking
/// would skip CRC verification of the region entirely (silently defeating
/// [`scan_and_recover_packfile`]'s whole purpose) and, on non-SSD media,
/// replace one sequential scan with many small seeks.
///
/// # Errors
/// Same conditions as [`read_record`] (invalid length, unsupported flags,
/// CRC mismatch, invalid `uncompressed_len`).
///
/// # Panics
/// Never in practice: the only internal `checked_sub`/`expect` pair
/// subtracts `FRAME_FIXED_LEN` from `frame_len`, which the preceding
/// range check already guarantees is `>= FRAME_FIXED_LEN`.
pub fn read_record_metadata(reader: &mut impl Read) -> io::Result<Option<RecordMetadata>> {
    let Some(len_buf) = read_frame_len_prefix(reader)? else {
        return Ok(None);
    };

    let frame_len = u32::from_le_bytes(len_buf);
    if !(FRAME_FIXED_LEN..=MAX_RECORD_LEN).contains(&frame_len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid record length: {frame_len}"),
        ));
    }

    let mut fixed = [0u8; FRAME_FIXED_LEN as usize];
    reader.read_exact(&mut fixed)?;

    let flags = fixed[0];
    if flags & !(FLAG_COMPRESSED | FLAG_CRC_DISABLED) != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported record flags: {flags:#04x}"),
        ));
    }
    let uncompressed_len = u32::from_le_bytes(fixed[1..5].try_into().unwrap());
    if flags & FLAG_COMPRESSED != 0 {
        if uncompressed_len > MAX_DATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("framed uncompressed_len too large: {uncompressed_len} > {MAX_DATA_LEN}"),
            ));
        }
    } else {
        let node_bytes_len = frame_len
            .checked_sub(FRAME_FIXED_LEN)
            .expect("frame_len >= FRAME_FIXED_LEN, checked above");
        if uncompressed_len != node_bytes_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "raw node length {node_bytes_len} != framed uncompressed_len {uncompressed_len}"
                ),
            ));
        }
    }
    let mut collection_id = [0u8; 16];
    collection_id.copy_from_slice(&fixed[5..21]);
    let mut hash = [0u8; 16];
    hash.copy_from_slice(&fixed[21..37]);

    let mut crc = if flags & FLAG_CRC_DISABLED == 0 {
        let mut crc = crc32fast::Hasher::new();
        crc.update(&len_buf);
        crc.update(&fixed);
        Some(crc)
    } else {
        None
    };

    // frame_len >= FRAME_FIXED_LEN is guaranteed by the range check above.
    let mut remaining: usize = frame_len
        .checked_sub(FRAME_FIXED_LEN)
        .expect("frame_len >= FRAME_FIXED_LEN, checked above")
        as usize;

    // For a CRC-disabled frame the checksum field is zero and unused; the
    // payload still must be consumed to advance the stream, just without the
    // hashing pass.
    let mut discard = [0u8; SCAN_DISCARD_BUF_LEN];
    while remaining > 0 {
        let chunk_len = remaining.min(discard.len());
        reader.read_exact(&mut discard[..chunk_len])?;
        if let Some(crc) = &mut crc {
            crc.update(&discard[..chunk_len]);
        }
        remaining = remaining
            .checked_sub(chunk_len)
            .expect("chunk_len <= remaining by construction (min above)");
    }

    let mut crc_buf = [0u8; 4];
    reader.read_exact(&mut crc_buf)?;
    if let Some(crc) = crc {
        let expected = crc.finalize();
        let actual = u32::from_le_bytes(crc_buf);
        if expected != actual {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CRC mismatch: expected {expected:08x}, got {actual:08x}"),
            ));
        }
    }

    Ok(Some(RecordMetadata {
        collection_id,
        hash,
    }))
}

/// Write a new pack file's [`HEADER_LEN`]-byte reserved header: magic,
/// version, header length, `pack_id`, creation time, and a CRC over all
/// of the above — zero-padded to fill `HEADER_LEN`. Written once, at
/// creation, and never mutated again (see [`VERSION`]'s doc for why).
///
/// # Errors
/// Returns `io::Error` on write failure, or if the system clock is
/// before the Unix epoch (treated as a hard error rather than silently
/// recording a wrong creation time).
///
/// # Panics
/// Never in practice: the only internal conversion (`HEADER_LEN` as
/// `u32`) is a compile-time constant well within range.
pub fn write_header(writer: &mut impl Write, pack_id: u64) -> io::Result<()> {
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs();

    let mut buf = [0u8; HEADER_LEN];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    let header_len = u32::try_from(HEADER_LEN).expect("HEADER_LEN fits in u32");
    buf[5..9].copy_from_slice(&header_len.to_le_bytes());
    buf[9..17].copy_from_slice(&pack_id.to_le_bytes());
    buf[17..25].copy_from_slice(&created_at.to_le_bytes());
    // buf[25..29] (feature_flags) stays zero — reserved for future use.

    let crc = crc32fast::hash(&buf[..CRC_COVERED_LEN]);
    buf[CRC_COVERED_LEN..CRC_COVERED_LEN.wrapping_add(4)].copy_from_slice(&crc.to_le_bytes());
    // The rest of buf is already zero-initialized padding out to HEADER_LEN.

    writer.write_all(&buf)
}

/// Read just the version byte from a packfile header (bytes 4..5, right
/// after the 4-byte [`MAGIC`]). Returns `Ok(None)` if the file is too
/// short or doesn't start with [`MAGIC`].
///
/// # Errors
///
/// Returns `io::Error` for a non-EOF read failure.
pub fn read_version(reader: &mut impl Read) -> io::Result<Option<u8>> {
    let mut magic = [0u8; 4];
    match reader.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    if magic != MAGIC {
        return Ok(None);
    }
    let mut version = [0u8; 1];
    match reader.read_exact(&mut version) {
        Ok(()) => {}
        // `read_version` is deliberately only a lightweight probe.  A file
        // ending immediately after MAGIC is still too short to identify a
        // version, just like one ending part-way through MAGIC.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    Ok(Some(version[0]))
}

/// Read and validate a shard file's reserved header.
///
/// Returns `Ok(None)` only for something that genuinely isn't an mdb
/// packfile at all — an empty file, or one whose first bytes don't match
/// [`MAGIC`]. Everything else that's wrong is a real `Err`, not a quiet
/// `None`, specifically so a caller (or one of the `scan_*` functions,
/// which all propagate this via `?`) can't mistake "this store is a
/// format I don't understand" for "this store has no data":
///
/// - Right magic, wrong version (most notably a pre-v2 store): a
///   distinct, identifiable error naming the version found, not folded
///   into "not a packfile".
/// - Right magic and version, but truncated before a full `HEADER_LEN`
///   header, or a CRC mismatch: corruption, reported as such.
///
/// # Errors
/// Returns `io::Error` (`Unsupported`) if the version doesn't match
/// [`VERSION`], or (`InvalidData`) if the header is truncated or its CRC
/// doesn't match, or on I/O failure.
///
/// # Panics
/// Never in practice: every internal `try_into`/`from_le_bytes` slices a
/// fixed, in-range region of the just-read header buffer.
pub fn read_header(reader: &mut impl Read) -> io::Result<Option<ShardHeader>> {
    // Phase 1: just enough to identify the format (magic + version),
    // before committing to reading a full HEADER_LEN buffer — a small
    // pre-cutover v1 file (bare 5-byte header) can easily be shorter
    // than HEADER_LEN in total, and version mismatch has to be detected
    // regardless of how much data follows it.
    let mut magic = [0u8; 4];
    match reader.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    if magic != MAGIC {
        return Ok(None);
    }
    let mut version = [0u8; 1];
    match reader.read_exact(&mut version) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing version byte",
            ));
        }
        Err(e) => return Err(e),
    }
    let version = version[0];
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "packfile format version {version:#04x} is not supported by this build \
                 (only {VERSION:#04x}) — this looks like a pre-cutover store; \
                 reset or migrate it rather than opening it with this version"
            ),
        ));
    }

    // Phase 2: magic and version confirmed — read the rest of the
    // reserved header. Running out of bytes here means a genuinely
    // truncated v2 header, which is corruption, not "not a packfile".
    let mut buf = [0u8; HEADER_LEN];
    buf[..4].copy_from_slice(&magic);
    buf[4] = version;
    reader.read_exact(&mut buf[5..]).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("truncated v2 shard header: shorter than {HEADER_LEN} bytes"),
            )
        } else {
            e
        }
    })?;

    let header_len = u32::from_le_bytes(buf[5..9].try_into().unwrap());
    if header_len as usize != HEADER_LEN {
        // A version byte we recognize but a header length we don't is
        // corruption too, for the same reason as above — VERSION and
        // HEADER_LEN have only ever shipped together.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("shard header declares length {header_len}, expected {HEADER_LEN}"),
        ));
    }

    let expected_crc = u32::from_le_bytes(
        buf[CRC_COVERED_LEN..CRC_COVERED_LEN.wrapping_add(4)]
            .try_into()
            .unwrap(),
    );
    let actual_crc = crc32fast::hash(&buf[..CRC_COVERED_LEN]);
    if actual_crc != expected_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("shard header CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"),
        ));
    }

    let pack_id = u64::from_le_bytes(buf[9..17].try_into().unwrap());
    let created_at = u64::from_le_bytes(buf[17..25].try_into().unwrap());

    Ok(Some(ShardHeader {
        pack_id,
        created_at,
    }))
}

/// Open or create a packfile, writing the header if it's new.
///
/// `pack_id` is the caller's expectation for this file — derived from
/// its filename. On creation it's written into the new header; on
/// opening an existing file it's cross-checked against what the header
/// actually says, so a shard file that's been copied or renamed
/// inconsistently with its own recorded identity is caught here rather
/// than silently trusted.
///
/// # Errors
/// Returns `io::Error` on open/write failure, `io::ErrorKind::InvalidData`
/// if the existing header is invalid or its CRC fails, or
/// `io::ErrorKind::InvalidData` if the header's recorded `pack_id`
/// doesn't match what the filename says it should be.
pub fn open_packfile(path: &Path, create: bool, pack_id: u64) -> io::Result<File> {
    if create {
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;

        if file.metadata()?.len() == 0 {
            write_header(&mut file, pack_id)?;
            file.sync_all()?;
        } else {
            let mut reader = BufReader::new(&file);
            let Some(header) = read_header(&mut reader)? else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid packfile header",
                ));
            };
            if header.pack_id != pack_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard file {} identifies itself as pack_id {:#018x} in its header, \
                         but its filename says pack_id {pack_id:#018x} — \
                         copied or renamed inconsistently with its own history",
                        path.display(),
                        header.pack_id,
                    ),
                ));
            }
        }
        Ok(file)
    } else {
        // Read-only path: open with read-only permissions, validate
        // the header, never write or create.
        let file = File::open(path)?;
        let mut reader = BufReader::new(&file);
        let Some(header) = read_header(&mut reader)? else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid packfile header",
            ));
        };
        if header.pack_id != pack_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shard file {} identifies itself as pack_id {:#018x} in its header, \
                     but its filename says pack_id {pack_id:#018x} — \
                     copied or renamed inconsistently with its own history",
                    path.display(),
                    header.pack_id,
                ),
            ));
        }
        Ok(file)
    }
}

/// Scan an existing packfile and return `(collection_id, hash, offset)` entries.
/// Used during startup to rebuild per-collection indexes from shard files.
///
/// Does **not** truncate the file — safe to call on an active shard while
/// concurrent appends are landing. A torn tail (a crashed write that leaves
/// an incomplete final frame) stops the scan gracefully and returns the
/// entries found before it. Anything else `read_record` rejects — an
/// invalid length, a CRC mismatch — is mid-file corruption, not a torn
/// tail, and is propagated as an error rather than silently discarded.
///
/// # Errors
/// Returns `io::Error` on I/O failure during open or header read, or on
/// any read-loop error other than `UnexpectedEof`.
pub fn scan_packfile(path: &Path) -> io::Result<Vec<ScanEntry>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    if read_header(&mut reader)?.is_none() {
        return Ok(entries);
    }

    loop {
        let offset = reader.stream_position()?;
        match read_record_metadata(&mut reader) {
            Ok(Some(meta)) => {
                entries.push((meta.collection_id, meta.hash, offset));
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                // Propagate real corruption (CRC mismatch, invalid length)
                // rather than silently discarding it — the caller decides
                // whether to recover or fail.
                return Err(e);
            }
        }
    }

    Ok(entries)
}

/// Scan a packfile starting from `start_offset`, returning only records
/// whose byte position is >= `start_offset`. Used for incremental repack:
/// the caller tracks how far through each shard file it has already scanned
/// and only needs the newly-appended bytes.
///
/// The offset must point to a valid record boundary (the start of a `[u32
/// len]` frame) — typically the byte position immediately after the last
/// record returned by a previous scan. Passing the file's end position
/// returns an empty vec without error.
///
/// Unlike [`scan_and_recover_packfile`], this never truncates — a
/// concurrent writer may be actively appending, so mutating the file is
/// unsafe. Torn tails are treated as EOF (same as [`scan_packfile`]).
///
/// # Errors
/// Returns `io::Error` on I/O failure other than `UnexpectedEof`.
pub fn scan_packfile_from(path: &Path, start_offset: u64) -> io::Result<Vec<ScanEntry>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Seek past the reserved header (HEADER_LEN bytes) to the first
    // record. If start_offset is already past the header, seek directly
    // there.
    if start_offset == 0 {
        if read_header(&mut reader)?.is_none() {
            return Ok(Vec::new());
        }
    } else {
        use std::io::Seek;
        let file_len = reader.get_ref().metadata()?.len();
        if start_offset > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("start_offset {start_offset} exceeds file length {file_len} (likely a stale cursor from a previous epoch)"),
            ));
        }
        reader.seek(io::SeekFrom::Start(start_offset))?;
    }

    let mut entries = Vec::new();
    loop {
        let offset = reader.stream_position()?;
        match read_record_metadata(&mut reader) {
            Ok(Some(meta)) => {
                entries.push((meta.collection_id, meta.hash, offset));
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }

    Ok(entries)
}

/// Scan a packfile and truncate any torn tail at the last valid record
/// boundary. Used during explicit recovery to repair a packfile before
/// reopening for append.
///
/// Only truncates on `UnexpectedEof` (a torn tail from a crashed write).
/// Other errors (CRC mismatch, invalid length) are propagated without
/// truncating — mid-file corruption should not discard valid records
/// that follow the corrupt frame.
///
/// # Errors
/// Returns `io::Error` on read or truncate failure.
pub fn scan_and_recover_packfile(path: &Path) -> io::Result<Vec<ScanEntry>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    if read_header(&mut reader)?.is_none() {
        return Ok(entries);
    }

    let mut last_valid_offset = reader.stream_position()?;
    let mut truncated = false;
    loop {
        let offset = reader.stream_position()?;
        match read_record_metadata(&mut reader) {
            Ok(Some(meta)) => {
                entries.push((meta.collection_id, meta.hash, offset));
                last_valid_offset = reader.stream_position()?;
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Torn tail: a write crashed mid-frame. Truncate to the
                // last valid record boundary so future appends don't land
                // after a corrupt frame.
                truncated = true;
                break;
            }
            Err(e) => {
                // Real corruption (CRC mismatch, invalid length) — propagate
                // without truncating. Later valid records may exist past the
                // corrupt frame; truncating would lose them.
                return Err(e);
            }
        }
    }

    // Only truncate if we actually hit a torn tail — not on clean EOF
    // or mid-file corruption (which was propagated as an error above).
    if truncated {
        drop(reader);
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(last_valid_offset)?;
        file.sync_all()?;
    }

    Ok(entries)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn test_record(collection_id: [u8; 16], hash: [u8; 16], data: &[u8]) -> Record {
        Record {
            collection_id,
            hash,
            data: Bytes::copy_from_slice(data),
        }
    }

    fn test_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_pf_{name}_{id}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_record_raw(hash: [u8; 16], data: &[u8]) -> Record {
        Record {
            collection_id: [0xAA; 16],
            hash,
            data: Bytes::copy_from_slice(data),
        }
    }

    #[test]
    fn test_write_read_roundtrip() {
        let record = test_record_raw([0xaa; 16], b"hello world");
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(record, read);
    }

    /// Highly-compressible, well-above-threshold data should be stored
    /// with the compressed flag set and round-trip exactly.
    #[test]
    fn test_write_read_roundtrip_compressed() {
        let data = vec![0x42u8; 8192];
        let record = test_record_raw([0xaa; 16], &data);
        let mut buf = Vec::new();
        let written = write_record(&mut buf, &record).unwrap();
        assert_eq!(written, buf.len() as u64);
        assert!(
            (buf.len() as u64) < record.serialized_len() as u64,
            "highly compressible data should shrink on disk: {} vs uncompressed upper bound {}",
            buf.len(),
            record.serialized_len()
        );

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(record, read);
    }

    /// `read_record_metadata` must agree with `read_record` on `collection_id`/`hash`
    /// for both a compressed and a raw-fallback frame, without touching
    /// (or needing to decompress) the payload.
    #[test]
    fn test_read_record_metadata_matches_read_record() {
        for data in [vec![0x42u8; 8192], b"hi".to_vec()] {
            let record = test_record_raw([0x77; 16], &data);
            let mut buf = Vec::new();
            write_record(&mut buf, &record).unwrap();

            let mut cursor = Cursor::new(&buf);
            let full = read_record(&mut cursor).unwrap().unwrap();

            let mut cursor = Cursor::new(&buf);
            let meta = read_record_metadata(&mut cursor).unwrap().unwrap();

            assert_eq!(meta.collection_id, full.collection_id);
            assert_eq!(meta.hash, full.hash);
        }
    }

    /// A frame spanning multiple `SCAN_DISCARD_BUF_LEN`-sized chunks must
    /// still stream correctly (payload larger than one discard buffer).
    #[test]
    fn test_read_record_metadata_spans_multiple_discard_chunks() {
        // Incompressible so it stays large on disk and forces several
        // discard-buffer iterations through the streaming loop.
        let data: Vec<u8> = (0..(SCAN_DISCARD_BUF_LEN * 3 + 500))
            .map(|i| (i as u64).wrapping_mul(2_654_435_761).to_le_bytes()[0])
            .collect();
        let record = test_record_raw([0x99; 16], &data);
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        let mut cursor = Cursor::new(&buf);
        let meta = read_record_metadata(&mut cursor).unwrap().unwrap();
        assert_eq!(meta.collection_id, record.collection_id);
        assert_eq!(meta.hash, record.hash);
    }

    /// Corruption inside the node-payload region must still be caught by
    /// `read_record_metadata` — it streams payload bytes through the CRC
    /// as it discards them rather than skipping them outright, so this
    /// must fail exactly like `read_record` does on the same corrupted
    /// buffer, not silently succeed because the payload was never
    /// "read" for its own sake.
    #[test]
    fn test_read_record_metadata_detects_payload_corruption() {
        let record = test_record_raw([0xbb; 16], b"payload region corruption test");
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        // Flip a byte inside the node-bytes region (right after the fixed
        // 37-byte header: len(4) + flags(1) + uncompressed_len(4) +
        // collection_id(16) + hash(16)).
        let corrupt_at = 4 + FRAME_FIXED_LEN as usize + 3;
        buf[corrupt_at] ^= 0xff;

        let mut cursor = Cursor::new(&buf);
        let err = read_record_metadata(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("CRC mismatch"));
    }

    /// Data that doesn't shrink under zstd (tiny/incompressible) must fall
    /// back to being stored raw, not expanded on disk.
    #[test]
    fn test_write_record_falls_back_to_raw_when_incompressible() {
        let record = test_record_raw([0xaa; 16], b"hi");
        let mut buf = Vec::new();
        let written = write_record(&mut buf, &record).unwrap();
        assert_eq!(written, buf.len() as u64);
        assert_eq!(buf.len(), record.serialized_len());

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(record, read);
    }

    #[test]
    fn test_multiple_records() {
        let records = vec![
            test_record_raw([0x01; 16], b"first"),
            test_record_raw([0x02; 16], b"second record"),
            test_record_raw([0x03; 16], &vec![0xff; 1024]),
        ];

        let mut buf = Vec::new();
        for r in &records {
            write_record(&mut buf, r).unwrap();
        }

        let mut cursor = Cursor::new(&buf);
        for expected in &records {
            let read = read_record(&mut cursor).unwrap().unwrap();
            assert_eq!(*expected, read);
        }
        assert!(read_record(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn test_crc_corruption_detected() {
        let record = test_record_raw([0xaa; 16], b"test data");
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        // Flip a byte in the node payload (after the fixed frame header:
        // len(4) + flags(1) + uncompressed_len(4) + collection_id(16) + hash(16))
        buf[41] ^= 0xff;

        let mut cursor = Cursor::new(&buf);
        let result = read_record(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_without_checksum_sets_flag_and_zeroes_checksum() {
        let record = test_record_raw([0xbb; 16], b"unverified payload");
        let buf = encode_record_with_options(&record, false, false).unwrap();

        // flags byte sits right after the u32 length prefix.
        let flags = buf[4];
        assert_eq!(flags & FLAG_CRC_DISABLED, FLAG_CRC_DISABLED);
        assert_eq!(flags & FLAG_COMPRESSED, 0);

        // The 4 trailing bytes are the checksum field — must be zero.
        let checksum = u32::from_le_bytes(buf[buf.len() - 4..].try_into().unwrap());
        assert_eq!(checksum, 0);
    }

    #[test]
    fn test_read_skips_crc_for_crc_disabled_frames() {
        let record = test_record_raw([0xcc; 16], b"unverified payload");
        let mut buf = encode_record_with_options(&record, false, false).unwrap();

        // Corrupt the payload region of a CRC-disabled frame: the reader must
        // not compare anything (the checksum field is zero), so the record
        // still decodes — with the tampered bytes surfacing as data.
        let node_bytes_start = 4 + FRAME_FIXED_LEN as usize;
        let corrupt_at = node_bytes_start + 2;
        let mut expected = b"unverified payload".to_vec();
        expected[2] ^= 0xff;
        buf[corrupt_at] ^= 0xff;

        let mut cursor = Cursor::new(&buf);
        let got = read_record(&mut cursor).unwrap().expect("must decode");
        assert_eq!(got.data.as_ref(), expected.as_slice());
    }

    #[test]
    fn test_read_still_verifies_when_only_the_checksum_flag_is_set_on_a_valid_frame() {
        // A frame written WITHOUT the disabled flag must keep failing reads
        // when its payload is still tampered with — verification must not be
        // accidentally skipped just because `FLAG_CRC_DISABLED` exists.
        let record = test_record_raw([0xdd; 16], b"verified payload");
        let mut buf = encode_record_with_options(&record, false, true).unwrap();
        buf[41] ^= 0xff;

        let mut cursor = Cursor::new(&buf);
        let result = read_record(&mut cursor);
        assert!(result.is_err(), "valid frames must still be verified");
    }

    #[test]
    fn test_write_record_with_options_keeps_full_checksums() {
        // The public write path stays on Full policy: frames carry the flag
        // clear and a real CRC, which `read_record` accepts.
        let record = test_record_raw([0xee; 16], b"still verified");
        let mut buf = Vec::new();
        write_record_with_options(&mut buf, &record, false).unwrap();

        let mut cursor = Cursor::new(&buf);
        let got = read_record(&mut cursor).unwrap().expect("must decode");
        assert_eq!(got.data.as_ref(), b"still verified");
        assert_eq!(buf[4] & FLAG_CRC_DISABLED, 0);
    }

    #[test]
    fn test_zero_length_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);

        let mut cursor = Cursor::new(&buf);
        let result = read_record(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_roundtrip() {
        let mut buf = Vec::new();
        write_header(&mut buf, 42).unwrap();
        assert_eq!(buf.len(), HEADER_LEN);

        let mut cursor = Cursor::new(&buf);
        let header = read_header(&mut cursor).unwrap().expect("valid header");
        assert_eq!(header.pack_id, 42);
        assert!(header.created_at > 0);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut buf = vec![0u8; HEADER_LEN];
        buf[0..4].copy_from_slice(b"BADC");

        let mut cursor = Cursor::new(&buf);
        assert!(read_header(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn test_header_empty_returns_none() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        assert!(read_header(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn test_header_crc_mismatch_is_an_error_not_none() {
        let mut buf = Vec::new();
        write_header(&mut buf, 1).unwrap();
        // Corrupt a byte inside the CRC-covered region (the pack_id
        // field) without touching magic/version/header_len — this must
        // surface as corruption, not as "not a packfile".
        buf[12] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        let err = read_header(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_record_serialized_len() {
        let r = Record {
            collection_id: [0u8; 16],
            hash: [0u8; 16],
            data: Bytes::from_static(b"hello"),
        };
        assert_eq!(r.serialized_len(), 4 + FRAME_FIXED_LEN as usize + 5 + 4);
    }

    #[test]
    fn test_write_record_payload_too_large() {
        let r = Record {
            collection_id: [0u8; 16],
            hash: [0u8; 16],
            data: Bytes::from(vec![0u8; MAX_RECORD_LEN as usize + 1]),
        };
        let mut buf = Vec::new();
        let err = write_record(&mut buf, &r).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// The largest *incompressible* payload that still fits at the write
    /// boundary — `FRAME_FIXED_LEN + data.len() == MAX_RECORD_LEN` — must
    /// round-trip. Regression test: an earlier version bounded the write
    /// check by `32 + data.len()` (the pre-compression frame's fixed
    /// size) while the read bound was `FRAME_FIXED_LEN + data.len()`, a
    /// 5-byte mismatch that let a write succeed at exactly this size and
    /// then fail on read.
    #[test]
    fn test_write_read_roundtrip_at_max_record_len_boundary() {
        let data_len = MAX_RECORD_LEN as usize - FRAME_FIXED_LEN as usize;
        // Pseudorandom, not zeros/repeats, so zstd can't shrink it below
        // the raw size — this must take the uncompressed-fallback path.
        let data: Vec<u8> = (0..data_len)
            .map(|i| (i as u64).wrapping_mul(2_654_435_761).to_le_bytes()[0])
            .collect();
        let record = test_record_raw([0xaa; 16], &data);
        let mut buf = Vec::new();
        let written = write_record(&mut buf, &record).unwrap();
        assert_eq!(written, buf.len() as u64);

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(record, read);
    }

    #[test]
    fn test_read_record_non_eof_io_error() {
        struct FailRead;
        impl Read for FailRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke"))
            }
        }
        let result = read_record(&mut FailRead);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn test_read_header_non_eof_io_error() {
        struct FailRead;
        impl Read for FailRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "nope"))
            }
        }
        let result = read_header(&mut FailRead);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn test_open_packfile_invalid_header() {
        let dir = test_dir("packfile_invalid_header");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        std::fs::write(&path, b"BADC\x02extra").unwrap();
        let result = open_packfile(&path, false, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    /// Direct test of the identity cross-check: a file whose *header* is
    /// perfectly valid (right magic, version, CRC) but whose embedded
    /// `pack_id` doesn't match what the caller expects (i.e. what the
    /// filename says) must be rejected — this is the actual detection
    /// claim, independent of any test fixture happening to already agree
    /// with its own filename.
    #[test]
    fn test_open_packfile_rejects_identity_mismatch() {
        let dir = test_dir("packfile_identity_mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pack_0000000000000002.pack");

        // Header genuinely says pack_id 0 — a valid v4 header on its
        // own terms, just not what this filename claims.
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let err = open_packfile(&path, false, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("0x0000000000000000") && msg.contains("0x0000000000000002"),
            "error must name both the header's actual pack_id and the filename's expected one, got: {msg}"
        );

        // The identical header opened under a matching expectation must
        // succeed — the check is about the mismatch, not the file itself.
        let ok_path = dir.join("pack_0000000000000000.pack");
        std::fs::write(&ok_path, &buf).unwrap();
        open_packfile(&ok_path, false, 0).unwrap();
    }

    /// A recognizable pre-cutover v1 file (right magic, version 0x01,
    /// otherwise well-formed) must be refused with a specific,
    /// identifiable error — not silently treated as absent/empty data.
    #[test]
    fn test_open_packfile_refuses_v1_with_specific_error() {
        let dir = test_dir("packfile_v1_refused");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");

        // A valid v1 file: 5-byte header (magic + version 0x01) followed
        // by a real, well-formed record — this is what an actual
        // pre-cutover store's shard file looks like, not a corrupt one.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(0x01);
        write_record(&mut buf, &test_record_raw([0x11; 16], b"old data")).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let err = open_packfile(&path, false, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        let msg = err.to_string();
        assert!(
            msg.contains("0x01") && msg.to_lowercase().contains("reset or migrate"),
            "error must identify the offending version and advise resetting/migrating, got: {msg}"
        );

        // The same file must not be silently readable as "0 entries" via
        // the standalone scan helpers either — this is the actual "too
        // quiet" failure mode: a v1 store must error, not look empty.
        assert_eq!(
            scan_packfile(&path).unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
        assert_eq!(
            scan_and_recover_packfile(&path).unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
        assert_eq!(
            scan_packfile_from(&path, 0).unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
    }

    /// The same v1-rejection must also hold for a v1 file too short to
    /// fill a v2 `HEADER_LEN` buffer — version mismatch must be caught by
    /// the 5-byte prefix check before ever attempting to read a full
    /// `HEADER_LEN`, not collapsed into "empty" by an early EOF.
    #[test]
    fn test_read_header_refuses_short_v1_file_not_silently_empty() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(0x01); // v1, and nothing else — far shorter than HEADER_LEN
        let mut cursor = Cursor::new(&buf);
        let err = read_header(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn test_read_version_magic_without_version_is_not_an_invalid_header() {
        let mut cursor = Cursor::new(MAGIC);
        assert_eq!(read_version(&mut cursor).unwrap(), None);
    }

    #[test]
    fn test_scan_packfile_empty_file() {
        let dir = test_dir("scan_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        std::fs::write(&path, b"").unwrap();
        let entries = scan_packfile(&path).unwrap();
        assert_eq!(entries, vec![]);
    }

    #[test]
    fn test_scan_packfile_torn_tail() {
        let dir = test_dir("scan_torn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        // Write header + one valid record + a torn tail: fewer than 4
        // bytes, so read_record can't even complete reading the length
        // prefix and hits UnexpectedEof.
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(&mut buf, &test_record_raw([0xaa; 16], b"data")).unwrap();
        buf.extend_from_slice(&[0xff; 3]); // torn trailing bytes
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_packfile(&path).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_scan_and_recover_truncates_torn_tail() {
        let dir = test_dir("recover_torn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(&mut buf, &test_record_raw([0xaa; 16], b"good")).unwrap();
        let valid_len = buf.len();
        // Simulate a realistic torn tail: valid length prefix declaring a
        // full frame, but fewer payload bytes actually present (missing
        // tail of the node bytes and the CRC). This mimics a crash mid-write.
        let declared_frame_len = FRAME_FIXED_LEN + 4; // fixed header + 4 bytes of data
        buf.extend_from_slice(&declared_frame_len.to_le_bytes());
        buf.push(0); // flags: uncompressed
        buf.extend_from_slice(&4u32.to_le_bytes()); // uncompressed_len
        buf.extend_from_slice(&[0xdd; 16]); // collection_id
        buf.extend_from_slice(&[0xbb; 16]); // hash
        buf.extend_from_slice(&[0xcc; 2]); // partial data (short of declared len; CRC never written)
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len as u64);
    }

    /// A partial length prefix is a torn frame, not clean EOF. Recovery must
    /// remove it so subsequent appends start at a valid record boundary.
    #[test]
    fn test_scan_and_recover_truncates_partial_length_prefix() {
        let dir = test_dir("recover_partial_prefix");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pack_0000000000000000.pack");
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(&mut buf, &test_record_raw([0xaa; 16], b"good")).unwrap();
        let valid_len = buf.len();
        buf.extend_from_slice(&[0x12, 0x34, 0x56]);
        std::fs::write(&path, &buf).unwrap();

        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len as u64);
    }

    #[test]
    fn test_scan_and_recover_clean_file() {
        let dir = test_dir("recover_clean");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(&mut buf, &test_record_raw([0xbb; 16], b"ok")).unwrap();
        write_record(&mut buf, &test_record_raw([0xcc; 16], b"ok2")).unwrap();
        let expected_len = buf.len();
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), expected_len as u64);
    }

    #[test]
    fn test_scan_and_recover_empty_header() {
        let dir = test_dir("recover_noheader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        std::fs::write(&path, b"").unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries, vec![]);
    }

    #[test]
    fn test_scan_packfile_returns_collection_id() {
        let dir = test_dir("scan_collection_id");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        let collection1 = [0x01; 16];
        let collection2 = [0x02; 16];
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(
            &mut buf,
            &test_record(collection1, [0xAA; 16], b"collection1 msg"),
        )
        .unwrap();
        write_record(
            &mut buf,
            &test_record(collection2, [0xBB; 16], b"collection2 msg"),
        )
        .unwrap();
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_packfile(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, collection1);
        assert_eq!(entries[0].1, [0xAA; 16]);
        assert_eq!(entries[1].0, collection2);
        assert_eq!(entries[1].1, [0xBB; 16]);
    }
}
