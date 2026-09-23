/// Physical, on-disk layout scanning — see [`layout::physical_layout`].
pub mod layout;
/// The [`PackfileStorage`](storage::PackfileStorage) engine and its supporting types.
pub mod storage;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, Write};
use std::path::Path;

use bytes::Bytes;

use crate::storage::DigestAlgorithm;

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

/// Per-frame flag: a versioned, length-delimited metadata area follows the
/// fixed header and precedes the node bytes (see [`FrameMetadata`]).
///
/// Metadata is stored **outside** the (possibly compressed) node bytes so a
/// reader can inspect identity/role without decompressing, and so generic
/// records with no metadata are byte-for-byte unchanged. A reader that does
/// not understand this flag MUST reject the frame rather than misparse it;
/// the byte layout differs from the flag's absence.
pub const FLAG_METADATA: u8 = 0x04;

/// Any flag bit a v4 reader understands. A frame carrying a bit outside this
/// mask is rejected rather than misparsed.
const KNOWN_FLAGS: u8 = FLAG_COMPRESSED | FLAG_CRC_DISABLED | FLAG_METADATA;

/// Version byte for the optional per-frame metadata area.
pub const METADATA_VERSION: u8 = 0x01;

/// TLV type: the full 256-bit logical identity digest. The value is exactly
/// 32 bytes.
pub const META_TAG_LOGICAL_ID: u8 = 0x01;

/// TLV type: the optional 256-bit content digest of the stored bytes. The
/// value is exactly 32 bytes. Distinct from [`META_TAG_LOGICAL_ID`]: a
/// content digest is an attribute of a stored version, not an identity.
pub const META_TAG_CONTENT_DIGEST: u8 = 0x02;

/// TLV type: a role/schema identifier (UTF-8 bytes, length-delimited).
pub const META_TAG_ROLE: u8 = 0x03;

/// TLV type: the algorithm that produced [`META_TAG_CONTENT_DIGEST`]. The
/// value is exactly 1 byte, matching [`DigestAlgorithm::id`]. Absent means the
/// reader should assume the default algorithm (SHA-256) for older writers.
pub const META_TAG_DIGEST_ALGORITHM: u8 = 0x04;

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

/// Streaming packfile metadata scanner.
pub struct PackfileScanner {
    reader: BufReader<File>,
    file_end: u64,
    verify_payload: bool,
    done: bool,
}

/// Open a streaming scanner for a packfile.
///
/// When `verify_payload` is true, payloads and CRCs are consumed and
/// validated. When false, only frame metadata and the declared frame bounds
/// are checked, which is appropriate for diagnostic listing.
///
/// # Errors
/// Returns an I/O error if the packfile cannot be opened or its header cannot
/// be read.
pub fn scan_packfile_iter(path: &Path, verify_payload: bool) -> io::Result<PackfileScanner> {
    let file = File::open(path)?;
    let file_end = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let done = read_header(&mut reader)?.is_none();
    Ok(PackfileScanner {
        reader,
        file_end,
        verify_payload,
        done,
    })
}

impl Iterator for PackfileScanner {
    type Item = io::Result<ScanEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let offset = match self.reader.stream_position() {
            Ok(offset) => offset,
            Err(error) => {
                self.done = true;
                return Some(Err(error));
            }
        };
        let result = if self.verify_payload {
            read_record_metadata(&mut self.reader)
        } else {
            read_record_metadata_skip_payload_with_end(&mut self.reader, self.file_end)
        };
        match result {
            Ok(Some(metadata)) => Some(Ok((metadata.collection_id, metadata.hash, offset))),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}

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
    /// Optional versioned per-record metadata (see [`FrameMetadata`]). `None`
    /// for a generic frame, which is byte-for-byte identical to a v4 record
    /// with no metadata area.
    pub metadata: Option<FrameMetadata>,
}

/// Optional, versioned metadata carried in a frame when [`FLAG_METADATA`] is
/// set. Stored between the fixed header and the node bytes, outside any
/// compression, so it can be inspected without decompressing.
///
/// The layout on disk is:
///
/// ```text
/// [u8 metadata_version]  — must equal METADATA_VERSION
/// [u32 metadata_len]     — byte length of the TLV block below (little-endian)
/// [TLV block]            — metadata_len bytes, each entry:
///     [u8 tag][u32 value_len][value_len bytes]
/// ```
///
/// The metadata bytes and the `metadata_len` field are inside the frame's
/// CRC-covered region. Unknown tags are preserved by the codec so a reader
/// built against a newer writer can round-trip fields it does not interpret.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameMetadata {
    /// Full 256-bit logical identity digest, if the template supplies one.
    pub logical_id: Option<[u8; 32]>,
    /// 256-bit digest of the stored bytes for this version, if supplied.
    pub content_digest: Option<[u8; 32]>,
    /// The hash function that produced [`Self::content_digest`]. Defaults to
    /// SHA-256; recorded per record so a template or configuration can choose
    /// another 256-bit algorithm without a format break.
    pub digest_algorithm: DigestAlgorithm,
    /// Role/schema identifier, if supplied.
    pub role: Option<Vec<u8>>,
    /// Tags this build does not interpret, preserved verbatim so a
    /// read/rewrite cycle does not discard a newer writer's fields.
    pub unknown: Vec<(u8, Vec<u8>)>,
}

impl FrameMetadata {
    /// Whether this metadata carries no fields at all. A frame with empty
    /// metadata is encoded without [`FLAG_METADATA`], so it stays
    /// byte-identical to a generic record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logical_id.is_none()
            && self.content_digest.is_none()
            && self.role.is_none()
            && self.unknown.is_empty()
    }

    /// Encode the `[metadata_version][metadata_len][TLV...]` block.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the TLV block exceeds `u32::MAX` bytes.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut tlv = Vec::new();
        if let Some(id) = &self.logical_id {
            push_tlv(&mut tlv, META_TAG_LOGICAL_ID, id)?;
        }
        if let Some(digest) = &self.content_digest {
            push_tlv(&mut tlv, META_TAG_CONTENT_DIGEST, digest)?;
            // The algorithm is only meaningful alongside a digest. Emit it
            // even for the default so the on-disk record is self-describing
            // rather than relying on a reader-side default.
            push_tlv(
                &mut tlv,
                META_TAG_DIGEST_ALGORITHM,
                &[self.digest_algorithm.id()],
            )?;
        }
        if let Some(role) = &self.role {
            push_tlv(&mut tlv, META_TAG_ROLE, role)?;
        }
        for (tag, value) in &self.unknown {
            push_tlv(&mut tlv, *tag, value)?;
        }
        let len = u32::try_from(tlv.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "frame metadata too large"))?;
        let mut out = Vec::with_capacity(1usize.saturating_add(4).saturating_add(tlv.len()));
        out.push(METADATA_VERSION);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&tlv);
        Ok(out)
    }

    /// Decode a metadata block from `bytes`, returning the metadata and the
    /// number of bytes consumed.
    ///
    /// # Errors
    /// Returns `InvalidData` on an unsupported version, a truncated block, a
    /// mismatched declared length, or a duplicate/known-tag length violation.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let version = *bytes
            .first()
            .ok_or_else(|| invalid_data("frame metadata: missing version byte"))?;
        if version != METADATA_VERSION {
            return Err(invalid_data(&format!(
                "frame metadata: unsupported version {version}"
            )));
        }
        let len_bytes: [u8; 4] = bytes
            .get(1..5)
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| invalid_data("frame metadata: truncated length"))?;
        let tlv_len = u32::from_le_bytes(len_bytes) as usize;
        let tlv_start = 5usize;
        let tlv_end = tlv_start
            .checked_add(tlv_len)
            .ok_or_else(|| invalid_data("frame metadata: length overflow"))?;
        let tlv = bytes
            .get(tlv_start..tlv_end)
            .ok_or_else(|| invalid_data("frame metadata: truncated TLV block"))?;

        let mut metadata = Self::default();
        let mut cursor = 0usize;
        while cursor < tlv.len() {
            let tag = *tlv
                .get(cursor)
                .ok_or_else(|| invalid_data("frame metadata: truncated tag"))?;
            let value_len_start = cursor
                .checked_add(1)
                .ok_or_else(|| invalid_data("frame metadata: cursor overflow"))?;
            let value_len_end = cursor
                .checked_add(5)
                .ok_or_else(|| invalid_data("frame metadata: cursor overflow"))?;
            let value_len_bytes: [u8; 4] = tlv
                .get(value_len_start..value_len_end)
                .and_then(|slice| slice.try_into().ok())
                .ok_or_else(|| invalid_data("frame metadata: truncated value length"))?;
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;
            let value_start = value_len_end;
            let value_end = value_start
                .checked_add(value_len)
                .ok_or_else(|| invalid_data("frame metadata: value length overflow"))?;
            let value = tlv
                .get(value_start..value_end)
                .ok_or_else(|| invalid_data("frame metadata: truncated value"))?;
            match tag {
                META_TAG_LOGICAL_ID => {
                    metadata.logical_id = Some(expect_32(value, "logical_id")?);
                }
                META_TAG_CONTENT_DIGEST => {
                    metadata.content_digest = Some(expect_32(value, "content_digest")?);
                }
                META_TAG_DIGEST_ALGORITHM => {
                    let id = *value
                        .first()
                        .ok_or_else(|| invalid_data("frame metadata: empty digest_algorithm"))?;
                    if value.len() != 1 {
                        return Err(invalid_data(
                            "frame metadata: digest_algorithm must be 1 byte",
                        ));
                    }
                    metadata.digest_algorithm = DigestAlgorithm::from_id(id);
                }
                META_TAG_ROLE => metadata.role = Some(value.to_vec()),
                _ => metadata.unknown.push((tag, value.to_vec())),
            }
            cursor = value_end;
        }

        Ok((metadata, tlv_end))
    }
}

fn push_tlv(out: &mut Vec<u8>, tag: u8, value: &[u8]) -> io::Result<()> {
    let len = u32::try_from(value.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame metadata value too large",
        )
    })?;
    out.push(tag);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn expect_32(value: &[u8], field: &str) -> io::Result<[u8; 32]> {
    <[u8; 32]>::try_from(value)
        .map_err(|_| invalid_data(&format!("frame metadata: {field} must be 32 bytes")))
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
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
#[cfg(feature = "zstd")]
const ZSTD_LEVEL: i32 = 3;

/// Compress `data` at [`ZSTD_LEVEL`], or `None` if the compressor errors (the
/// caller then stores the frame raw). Only built with the `zstd` feature.
#[cfg(feature = "zstd")]
fn zstd_maybe_compress(data: &[u8]) -> Option<Vec<u8>> {
    zstd::bulk::compress(data, ZSTD_LEVEL).ok()
}

/// Decompress a frame's node bytes to exactly `expected_len` (the framed
/// `uncompressed_len`, already range-checked by the caller). Shared by the
/// buffered [`read_record`] path and the shard zero-copy decode path.
#[cfg(feature = "zstd")]
pub(crate) fn zstd_decompress(node_bytes: &[u8], expected_len: usize) -> io::Result<Vec<u8>> {
    let decompressed = zstd::bulk::decompress(node_bytes, expected_len).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("zstd decompress failed: {e}"),
        )
    })?;
    if decompressed.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "decompressed length {} != framed uncompressed_len {expected_len}",
                decompressed.len()
            ),
        ));
    }
    Ok(decompressed)
}

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
    let metadata_block = match &record.metadata {
        Some(metadata) if !metadata.is_empty() => Some(metadata.encode()?),
        _ => None,
    };
    let metadata_len = metadata_block.as_ref().map_or(0, |block| {
        u32::try_from(block.len()).expect("bounded below")
    });
    let plaintext_frame_len = FRAME_FIXED_LEN
        .checked_add(metadata_len)
        .expect("record frame length exceeds u32::MAX")
        .checked_add(uncompressed_len)
        .expect("record frame length exceeds u32::MAX");
    if plaintext_frame_len > MAX_RECORD_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("record payload too large: {plaintext_frame_len} > {MAX_RECORD_LEN}"),
        ));
    }

    #[cfg(feature = "zstd")]
    let compressed = compress
        .then(|| zstd_maybe_compress(&record.data))
        .flatten();
    #[cfg(not(feature = "zstd"))]
    let compressed: Option<Vec<u8>> = {
        let _ = compress;
        None
    };
    let (base_flags, node_bytes): (u8, &[u8]) = match &compressed {
        Some(c) if c.len() < record.data.len() => (FLAG_COMPRESSED, c.as_slice()),
        _ => (0, &record.data),
    };
    let mut flags = if write_checksum {
        base_flags
    } else {
        base_flags | FLAG_CRC_DISABLED
    };
    if metadata_block.is_some() {
        flags |= FLAG_METADATA;
    }

    let frame_len = FRAME_FIXED_LEN
        .checked_add(metadata_len)
        .expect("bounded by MAX_RECORD_LEN")
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
    if let Some(block) = &metadata_block {
        buf.extend_from_slice(block);
    }
    buf.extend_from_slice(node_bytes);

    // CRC covers len + flags + uncompressed_len + collection_id + hash +
    // metadata + node_bytes, i.e. the bytes as written to disk (metadata
    // included, node bytes compressed when compressed) — exactly `buf`'s
    // contents so far, hashed in one pass since CRC32 over one contiguous
    // buffer is identical to the same bytes hashed via several `update`
    // calls. Under a period of `write_checksum == false` the field is
    // written as zeros (with FLAG_CRC_DISABLED set above) and no hashing
    // pass happens at all.
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
    if flags & !KNOWN_FLAGS != 0 {
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

    // Optional metadata sits between the fixed header and the node bytes.
    let rest = &payload[37..];
    let (metadata, node_bytes) = if flags & FLAG_METADATA != 0 {
        let (metadata, consumed) = FrameMetadata::decode(rest)?;
        let node_bytes = rest
            .get(consumed..)
            .ok_or_else(|| invalid_data("frame metadata consumes past payload"))?;
        (Some(metadata), node_bytes)
    } else {
        (None, rest)
    };

    let data = if flags & FLAG_COMPRESSED != 0 {
        if uncompressed_len > MAX_DATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("framed uncompressed_len too large: {uncompressed_len} > {MAX_DATA_LEN}"),
            ));
        }
        #[cfg(feature = "zstd")]
        {
            Bytes::from(zstd_decompress(
                node_bytes,
                usize::try_from(uncompressed_len).expect("checked above"),
            )?)
        }
        #[cfg(not(feature = "zstd"))]
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "frame is zstd-compressed but this build was compiled without the `zstd` feature",
            ));
        }
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
        metadata,
    }))
}

/// A frame's `collection_id`/`hash` metadata, without its node payload — what
/// [`scan_packfile`]/[`scan_packfile_from`]/[`scan_and_recover_packfile`]
/// actually need. See [`read_record_metadata`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordMetadata {
    /// The collection this record belongs to.
    pub collection_id: [u8; 16],
    /// The structural hash framed alongside the record.
    pub hash: [u8; 16],
    /// Optional per-record metadata, if the frame carried [`FLAG_METADATA`].
    pub metadata: Option<FrameMetadata>,
}

/// Buffer size for streaming node bytes through [`read_record_metadata`]
/// without allocating a buffer proportional to the frame's payload size.
/// Large enough to keep syscall/CRC-update overhead low, small enough to
/// stay a stack buffer.
const SCAN_DISCARD_BUF_LEN: usize = 8192;

/// Shared header fields parsed from the fixed portion of a frame.
///
/// Extracted once by [`read_frame_header`] and consumed by either
/// [`read_record_metadata`] (CRC-verified) or
/// [`read_record_metadata_skip_payload`] (seek-past).
struct FrameHeader {
    len_buf: [u8; 4],
    frame_len: u32,
    flags: u8,
    fixed: [u8; FRAME_FIXED_LEN as usize],
    collection_id: [u8; 16],
    hash: [u8; 16],
    /// Optional metadata parsed from the frame, if [`FLAG_METADATA`] was set.
    metadata: Option<FrameMetadata>,
    /// The raw on-disk metadata block (`[version][len][TLV...]`), empty when
    /// absent. Retained so callers can feed it into the frame CRC (the block
    /// is covered by the checksum) and subtract its length from `frame_len`
    /// to find the node-bytes region.
    metadata_block: Vec<u8>,
}

/// Resolve the on-disk metadata block length (`5 + tlv_len`) and bound it
/// against the frame that must contain it, *before* the caller allocates.
///
/// `tlv_len` comes straight from disk. The metadata block (its 5-byte prefix
/// plus the TLV bytes) shares the frame's post-header body with the node bytes,
/// so it must satisfy `block_len <= frame_len - FRAME_FIXED_LEN`. Because
/// `frame_len` is itself capped at [`MAX_RECORD_LEN`], this both rejects a
/// frame whose metadata overruns it and caps the metadata allocation at the
/// record-size limit instead of the `u32` field's ~4 GiB range.
fn checked_metadata_block_len(tlv_len: u32, frame_len: u32) -> io::Result<u32> {
    let block_len = 5u32
        .checked_add(tlv_len)
        .ok_or_else(|| invalid_data("frame metadata: length overflow"))?;
    let remaining_frame_body = frame_len
        .checked_sub(FRAME_FIXED_LEN)
        .ok_or_else(|| invalid_data("frame metadata overruns the frame"))?;
    if block_len > remaining_frame_body {
        return Err(invalid_data("frame metadata exceeds frame length"));
    }
    Ok(block_len)
}

/// Read and validate the frame length prefix and fixed header, returning the
/// parsed fields. The caller is responsible for consuming the payload and CRC
/// (or seeking past them).
fn read_frame_header(reader: &mut impl Read) -> io::Result<Option<FrameHeader>> {
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
    if flags & !KNOWN_FLAGS != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported record flags: {flags:#04x}"),
        ));
    }

    // Read the optional metadata block (version byte + u32 length + TLV).
    let (metadata, metadata_block) = if flags & FLAG_METADATA != 0 {
        let mut prefix = [0u8; 5];
        reader.read_exact(&mut prefix)?;
        let version = prefix[0];
        if version != METADATA_VERSION {
            return Err(invalid_data(&format!(
                "frame metadata: unsupported version {version}"
            )));
        }
        let tlv_len = u32::from_le_bytes(prefix[1..5].try_into().expect("fixed slice"));
        // Resolve and bound the metadata block *before* allocating: `tlv_len`
        // is disk-controlled, so the allocation below must never be sized by it
        // unchecked.
        let block_len = checked_metadata_block_len(tlv_len, frame_len)?;
        let mut block = vec![0u8; usize::try_from(block_len).expect("u32 fits in usize")];
        block[..5].copy_from_slice(&prefix);
        reader.read_exact(&mut block[5..])?;
        let (metadata, consumed) = FrameMetadata::decode(&block)?;
        if consumed != block.len() {
            return Err(invalid_data(
                "frame metadata: trailing bytes after TLV block",
            ));
        }
        (Some(metadata), block)
    } else {
        (None, Vec::new())
    };
    let metadata_len = u32::try_from(metadata_block.len()).expect("bounded by frame length");

    let node_region_len = frame_len
        .checked_sub(FRAME_FIXED_LEN)
        .and_then(|len| len.checked_sub(metadata_len))
        .ok_or_else(|| invalid_data("frame metadata overruns the frame"))?;
    let uncompressed_len = u32::from_le_bytes(fixed[1..5].try_into().unwrap());
    if flags & FLAG_COMPRESSED != 0 {
        if uncompressed_len > MAX_DATA_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("framed uncompressed_len too large: {uncompressed_len} > {MAX_DATA_LEN}"),
            ));
        }
    } else if uncompressed_len != node_region_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "raw node length {node_region_len} != framed uncompressed_len {uncompressed_len}"
            ),
        ));
    }

    let mut collection_id = [0u8; 16];
    collection_id.copy_from_slice(&fixed[5..21]);
    let mut hash = [0u8; 16];
    hash.copy_from_slice(&fixed[21..37]);

    Ok(Some(FrameHeader {
        len_buf,
        frame_len,
        flags,
        fixed,
        collection_id,
        hash,
        metadata,
        metadata_block,
    }))
}

/// Read one frame's metadata (`collection_id`, `hash`) without allocating,
/// decompressing, or otherwise materializing its node payload — the scan
/// path's counterpart to [`read_record`], which fully decodes a frame for
/// callers that actually need its data.
///
/// Node bytes are streamed through a small fixed buffer and fed into the
/// running CRC as they're read, then discarded — this still detects
/// corruption in the payload region (the CRC covers the same bytes
/// [`read_record`] verifies), but never buffers or decompresses them.
/// Deliberately a streaming read, rather than seeking past the payload,
/// because seeking would skip CRC verification of the region.
///
/// Node bytes are streamed through a small fixed buffer and fed into the
/// running CRC as they're read, then discarded.
///
/// # Errors
/// Returns an I/O error for invalid length, unsupported flags, truncated
/// payload/CRC data, or a CRC mismatch.
///
/// # Panics
/// Never in practice: `read_frame_header` validates that `frame_len` is at
/// least `FRAME_FIXED_LEN` before the checked subtraction below.
pub fn read_record_metadata(reader: &mut impl Read) -> io::Result<Option<RecordMetadata>> {
    let Some(header) = read_frame_header(reader)? else {
        return Ok(None);
    };

    let mut crc = if header.flags & FLAG_CRC_DISABLED == 0 {
        let mut crc = crc32fast::Hasher::new();
        crc.update(&header.len_buf);
        crc.update(&header.fixed);
        crc.update(&header.metadata_block);
        Some(crc)
    } else {
        None
    };

    // frame_len >= FRAME_FIXED_LEN is guaranteed by the range check above;
    // subtract the metadata block already consumed by read_frame_header.
    let mut remaining: usize = header
        .frame_len
        .checked_sub(FRAME_FIXED_LEN)
        .and_then(|len| len.checked_sub(u32::try_from(header.metadata_block.len()).ok()?))
        .expect("frame_len >= FRAME_FIXED_LEN + metadata_len, checked in read_frame_header")
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
        collection_id: header.collection_id,
        hash: header.hash,
        metadata: header.metadata,
    }))
}

/// Read one frame's metadata and seek over its payload and CRC without
/// reading them. This is intended for diagnostic scans where payload
/// integrity is not being checked. The frame length, flags, and raw payload
/// length invariants are still validated.
///
/// # Errors
/// Returns an I/O error if the frame prefix or metadata cannot be read, if
/// the frame metadata is invalid, or if seeking past the payload fails.
///
/// # Panics
/// Never in practice: the fixed-size metadata buffer makes all internal
/// conversions and the frame-length subtraction provably valid after the
/// preceding bounds checks.
pub fn read_record_metadata_skip_payload(
    reader: &mut (impl Read + Seek),
) -> io::Result<Option<RecordMetadata>> {
    let payload_start = reader.stream_position()?;
    let file_end = reader.seek(io::SeekFrom::End(0))?;
    reader.seek(io::SeekFrom::Start(payload_start))?;
    read_record_metadata_skip_payload_with_end(reader, file_end)
}

fn read_record_metadata_skip_payload_with_end(
    reader: &mut (impl Read + Seek),
    file_end: u64,
) -> io::Result<Option<RecordMetadata>> {
    let Some(header) = read_frame_header(reader)? else {
        return Ok(None);
    };

    // The frame length excludes the four-byte CRC, while the fixed portion
    // includes the metadata and flags but not the node bytes or the
    // already-consumed metadata block.
    let skip = u64::from(
        header
            .frame_len
            .checked_sub(FRAME_FIXED_LEN)
            .and_then(|len| len.checked_sub(u32::try_from(header.metadata_block.len()).ok()?))
            .expect("frame_len >= FRAME_FIXED_LEN + metadata_len, checked in read_frame_header"),
    )
    .checked_add(4)
    .expect("frame payload length fits u64");
    let payload_start = reader.stream_position()?;
    let frame_end = payload_start
        .checked_add(skip)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "record end overflows u64"))?;
    if frame_end > file_end {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "record payload or CRC is truncated",
        ));
    }
    reader.seek(io::SeekFrom::Start(frame_end))?;
    Ok(Some(RecordMetadata {
        collection_id: header.collection_id,
        hash: header.hash,
        metadata: header.metadata,
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

/// Scan a packfile's metadata without reading frame payloads or CRCs.
/// Intended for diagnostic listing, not recovery or integrity verification.
///
/// # Errors
/// Returns an I/O error on failure to open, read, seek, or validate the
/// packfile.
pub fn scan_packfile_skip_payload(path: &Path) -> io::Result<Vec<ScanEntry>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    if read_header(&mut reader)?.is_none() {
        return Ok(entries);
    }
    let file_end = reader.get_ref().metadata()?.len();
    loop {
        let offset = reader.stream_position()?;
        match read_record_metadata_skip_payload_with_end(&mut reader, file_end) {
            Ok(Some(meta)) => entries.push((meta.collection_id, meta.hash, offset)),
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
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
            metadata: None,
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
            metadata: None,
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
    #[cfg(feature = "zstd")]
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

    /// A metadata-bearing frame must round-trip its metadata through both the
    /// full `read_record` path and the metadata-only scan path, including
    /// zero-copy-preserving structural fields.
    #[test]
    fn test_metadata_roundtrip() {
        let metadata = FrameMetadata {
            logical_id: Some([0x11; 32]),
            content_digest: Some([0x22; 32]),
            digest_algorithm: DigestAlgorithm::Sha256,
            role: Some(b"canonical_event".to_vec()),
            unknown: vec![(0x7f, vec![1, 2, 3])],
        };
        let record = Record {
            collection_id: [0xAA; 16],
            hash: [0xBB; 16],
            data: Bytes::from_static(b"hello metadata"),
            metadata: Some(metadata.clone()),
        };
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(read.data, record.data);
        assert_eq!(read.metadata.as_ref(), Some(&metadata));

        let mut cursor = Cursor::new(&buf);
        let meta = read_record_metadata(&mut cursor).unwrap().unwrap();
        assert_eq!(meta.metadata.as_ref(), Some(&metadata));

        let mut cursor = Cursor::new(&buf);
        let skipped = read_record_metadata_skip_payload(&mut cursor)
            .unwrap()
            .unwrap();
        assert_eq!(skipped.metadata.as_ref(), Some(&metadata));
    }

    /// The metadata flag must be set only when metadata is present; a record
    /// with `None` (or empty) metadata stays byte-identical to a v4 frame with
    /// no metadata area, so generic records pay nothing.
    #[test]
    fn test_no_metadata_frame_is_unchanged() {
        let plain = test_record_raw([0xCC; 16], b"payload");
        let mut plain_buf = Vec::new();
        write_record(&mut plain_buf, &plain).unwrap();
        assert_eq!(plain_buf[4] & FLAG_METADATA, 0);

        let empty = Record {
            metadata: Some(FrameMetadata::default()),
            ..plain.clone()
        };
        let mut empty_buf = Vec::new();
        write_record(&mut empty_buf, &empty).unwrap();
        assert_eq!(empty_buf, plain_buf);
    }

    /// A metadata-bearing frame with a CRC-disabled policy must still write,
    /// parse, and skip correctly (the metadata block is inside the frame's
    /// CRC region, but CRC-disabled frames simply carry no checksum).
    #[test]
    fn test_metadata_with_crc_disabled() {
        let record = Record {
            collection_id: [0x01; 16],
            hash: [0x02; 16],
            data: Bytes::from_static(b"body"),
            metadata: Some(FrameMetadata {
                logical_id: Some([0x09; 32]),
                ..FrameMetadata::default()
            }),
        };
        let mut buf = Vec::new();
        encode_record_with_options(&record, true, false)
            .map(|bytes| buf.extend_from_slice(&bytes))
            .unwrap();

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(read.metadata, record.metadata);
        assert_eq!(read.data, record.data);
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

    /// A frame whose metadata block declares more bytes than the frame can
    /// hold must be rejected *before* the reader allocates a buffer sized by
    /// that disk-supplied length, rather than requesting an allocation far
    /// larger than the frame (and the `MAX_RECORD_LEN` cap) permits.
    #[test]
    fn test_metadata_length_cannot_exceed_frame() {
        let frame_len = FRAME_FIXED_LEN + 5;
        let mut buf = Vec::new();
        buf.extend_from_slice(&frame_len.to_le_bytes());
        let mut fixed = [0u8; FRAME_FIXED_LEN as usize];
        fixed[0] = FLAG_METADATA;
        buf.extend_from_slice(&fixed);
        // Metadata prefix: a valid version, then a `tlv_len` claiming the
        // largest possible metadata block (a ~4 GiB allocation without the
        // bound), far beyond the five bytes of metadata space the frame has.
        buf.push(METADATA_VERSION);
        buf.extend_from_slice(&(u32::MAX - 5).to_le_bytes());

        let mut cursor = Cursor::new(&buf);
        let Err(err) = read_frame_header(&mut cursor) else {
            panic!("oversized metadata must be rejected");
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("metadata"),
            "unexpected error: {err}"
        );
    }

    /// `checked_metadata_block_len` is inclusive at the frame body and rejects
    /// one byte past it, so the guard is verified independently of allocation.
    #[test]
    fn test_checked_metadata_block_len_boundary() {
        let frame_len = FRAME_FIXED_LEN + 20;
        // block_len = 5 + tlv_len; body = 20, so tlv_len = 15 is exactly full.
        assert_eq!(checked_metadata_block_len(15, frame_len).unwrap(), 20);
        // One byte past the body is rejected.
        assert_eq!(
            checked_metadata_block_len(16, frame_len)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        // A `tlv_len` that overflows the 5-byte prefix is rejected too.
        assert_eq!(
            checked_metadata_block_len(u32::MAX, frame_len)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        // A frame shorter than the fixed header cannot hold any metadata.
        // Production callers reject it earlier via the `FRAME_FIXED_LEN
        // ..= MAX_RECORD_LEN` range check; this pins the helper's own behavior.
        assert_eq!(
            checked_metadata_block_len(0, FRAME_FIXED_LEN - 1)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    /// The bound is inclusive end to end: a metadata block that exactly fills
    /// the frame's post-header body (zero node bytes) is a valid maximum-size
    /// frame and must still decode, so the fix rejects only overrun.
    #[test]
    fn test_metadata_at_frame_body_boundary_decodes() {
        // One unknown TLV entry sized so the block exactly fills the frame
        // body: block = 5 (prefix) + 5 (tag + value_len) + value.
        let value_len = 64usize;
        let tlv_len = u32::try_from(5 + value_len).expect("test TLV length fits u32");
        let block_len = 5 + tlv_len;
        let frame_len = FRAME_FIXED_LEN + block_len;

        let mut buf = Vec::new();
        buf.extend_from_slice(&frame_len.to_le_bytes());
        let mut fixed = [0u8; FRAME_FIXED_LEN as usize];
        // `uncompressed_len` stays 0, matching the zero-length node region.
        fixed[0] = FLAG_METADATA | FLAG_CRC_DISABLED;
        buf.extend_from_slice(&fixed);
        buf.push(METADATA_VERSION);
        buf.extend_from_slice(&tlv_len.to_le_bytes());
        buf.push(0x7f); // unknown tag, preserved verbatim
        buf.extend_from_slice(
            &u32::try_from(value_len)
                .expect("test value length fits u32")
                .to_le_bytes(),
        );
        buf.extend_from_slice(&vec![0xAB; value_len]);
        // Zero node bytes, then the (unused) 4-byte checksum field.
        buf.extend_from_slice(&[0u8; 4]);

        let mut cursor = Cursor::new(&buf);
        let meta = read_record_metadata(&mut cursor)
            .expect("boundary-size frame must not error")
            .expect("boundary-size frame must decode");
        assert_eq!(
            meta.metadata.expect("metadata present").unknown,
            vec![(0x7f, vec![0xAB; value_len])]
        );
    }

    /// Without the `zstd` feature, a frame flagged compressed must be rejected
    /// with a clear error rather than silently misread as raw.
    #[cfg(not(feature = "zstd"))]
    #[test]
    fn test_compressed_frame_rejected_without_zstd_feature() {
        let record = test_record_raw([0xaa; 16], b"payload");
        // Raw, checksum-less frame; flip the compressed flag on (the flags
        // byte sits right after the u32 frame length). Checksum off so the
        // flag flip doesn't trip the CRC check first.
        let mut buf = encode_record_with_options(&record, false, false).unwrap();
        buf[4] |= FLAG_COMPRESSED;

        let mut cursor = Cursor::new(&buf);
        let err = read_record(&mut cursor).unwrap_err();
        assert!(
            err.to_string().contains("without the `zstd` feature"),
            "unexpected error: {err}"
        );
    }

    /// Data that doesn't shrink under zstd (tiny/incompressible) must fall
    /// back to being stored raw, not expanded on disk.
    #[cfg(feature = "zstd")]
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
            metadata: None,
        };
        assert_eq!(r.serialized_len(), 4 + FRAME_FIXED_LEN as usize + 5 + 4);
    }

    #[test]
    fn test_write_record_payload_too_large() {
        let r = Record {
            collection_id: [0u8; 16],
            hash: [0u8; 16],
            data: Bytes::from(vec![0u8; MAX_RECORD_LEN as usize + 1]),
            metadata: None,
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

    /// `scan_packfile_skip_payload` must agree with `scan_packfile` on every
    /// `(collection_id, hash, offset)` triple, for both a compressed and a
    /// raw-fallback frame, without reading the payload it seeks over.
    #[test]
    fn test_scan_packfile_skip_payload_matches_scan_packfile() {
        let dir = test_dir("scan_skip_payload");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        let collection1 = [0x01; 16];
        let collection2 = [0x02; 16];
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(
            &mut buf,
            &test_record(collection1, [0xAA; 16], b"short raw payload"),
        )
        .unwrap();
        write_record(
            &mut buf,
            &test_record(collection2, [0xBB; 16], &vec![0x42u8; 8192]),
        )
        .unwrap();
        std::fs::write(&path, &buf).unwrap();

        let full = scan_packfile(&path).unwrap();
        let skipped = scan_packfile_skip_payload(&path).unwrap();
        assert_eq!(full, skipped);
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn test_scan_packfile_skip_payload_empty_file() {
        let dir = test_dir("scan_skip_payload_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        std::fs::write(&path, b"").unwrap();
        let entries = scan_packfile_skip_payload(&path).unwrap();
        assert_eq!(entries, vec![]);
    }

    #[test]
    fn test_scan_packfile_skip_payload_stops_at_truncated_crc() {
        let dir = test_dir("scan_skip_payload_truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_00.pack");
        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        write_record(&mut buf, &test_record([0x01; 16], [0xAA; 16], b"complete")).unwrap();
        let complete_len = buf.len();
        write_record(&mut buf, &test_record([0x02; 16], [0xBB; 16], b"torn")).unwrap();
        buf.truncate(buf.len() - 2);
        std::fs::write(&path, &buf).unwrap();

        let entries = scan_packfile_skip_payload(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].2, HEADER_LEN as u64);
        assert!(buf.len() > complete_len);
    }

    /// The metadata bound must also hold on the on-disk scan paths: a packfile
    /// whose frame declares a ~4 GiB metadata block is reported as corrupt,
    /// not used to size an allocation, by both the full and skip-payload scans.
    #[test]
    fn test_scan_rejects_oversized_frame_metadata() {
        let dir = test_dir("scan_oversized_metadata");
        let path = dir.join("shard_00.pack");

        let record = Record {
            collection_id: [0x01; 16],
            hash: [0xAA; 16],
            data: Bytes::from_static(b"payload"),
            metadata: Some(FrameMetadata {
                logical_id: Some([0x09; 32]),
                ..FrameMetadata::default()
            }),
        };
        let mut frame = encode_record_with_options(&record, false, false).unwrap();
        // The metadata block starts right after the fixed header; its
        // `tlv_len` field is the u32 following the version byte.
        let tlv_len_at = 4 + FRAME_FIXED_LEN as usize + 1;
        frame[tlv_len_at..tlv_len_at + 4].copy_from_slice(&(u32::MAX - 5).to_le_bytes());

        let mut buf = Vec::new();
        write_header(&mut buf, 0).unwrap();
        buf.extend_from_slice(&frame);
        std::fs::write(&path, &buf).unwrap();

        for err in [
            scan_packfile(&path).unwrap_err(),
            scan_packfile_skip_payload(&path).unwrap_err(),
        ] {
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
            assert!(
                err.to_string().contains("metadata"),
                "unexpected error: {err}"
            );
        }
    }
}
