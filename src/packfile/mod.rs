mod entropy;

/// Physical, on-disk layout scanning — see [`layout::physical_layout`].
pub mod layout;
pub(crate) mod publish_signal;
/// The [`PackfileStorage`](storage::PackfileStorage) engine and its supporting types.
pub mod storage;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use bytes::Bytes;

use crate::storage::DigestAlgorithm;

/// Magic bytes identifying an mtxdb packfile: "MTDB"
pub const MAGIC: [u8; 4] = *b"MTDB";

/// Packfile format version byte following `MAGIC` in the header (see
/// [`write_header`]/[`read_header`]).
///
/// Version 5: replaces the pool-local monotonic `pack_id: u64` identity with a
/// globally unique, cryptographically random 128-bit [`PackId`]. The
/// address is unique across pools, databases, and hosts, so an extracted or
/// imported pack can be adopted verbatim (no rename, no header rewrite) and
/// referenced globally. Filename changes from `pack_{pack_id:016x}.pack` to
/// `pack_{short_address}.pack`. Hard cutover: earlier versions are rejected.
///
/// There is no persisted numeric id, runtime slot, or creation sequence in the
/// header: identity is the address, ordering is the caller's concern (by
/// timestamp/offset or an explicit conflict policy), and the runtime
/// slot/index is entirely internal to the storage implementation.
pub const VERSION: u8 = 0x05;

/// Total reserved header size in bytes: every shard file's first record
/// starts at exactly this offset. One 4KiB page — ample collection for the
/// descriptor fields plus future growth, with no benefit to a larger
/// reservation (see [`write_header`] for the field layout).
pub const HEADER_LEN: usize = 4096;

/// Byte length of a [`PackId`].
pub const PACK_ID_LEN: usize = 16;

/// Filenames carry a shortened address prefix by default (64 bits); a colliding
/// prefix is disambiguated with the full 128-bit address. The full address in
/// the pack header remains authoritative. The parser also accepts legacy
/// underscore-joined 16-hex groups, up to the full address width.
pub const PACK_FILENAME_PREFIX_HEX: usize = 16;

/// A pack's globally unique, immutable identity: 16 cryptographically random
/// bytes assigned at creation.
///
/// Unlike the old pool-local monotonic counter, this address is unique across
/// pools, databases, and hosts, so an extracted or imported pack keeps the same
/// identity wherever it lands. It is written once at pack creation and never
/// changes while the pack grows.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PackId(pub [u8; PACK_ID_LEN]);

impl PackId {
    /// Generate a fresh nonzero random address from the operating system's CSPRNG.
    ///
    /// Fills the full [`PACK_ID_LEN`] bytes from the operating system's CSPRNG
    /// (`/dev/urandom` on unix, `BCryptGenRandom` on Windows). The identity is
    /// externally visible and portable across pools and hosts, so
    /// it must carry real entropy rather than a hash of a weaker seed: this
    /// yields the full 128-bit collision bound the identity is documented to
    /// have (birthday bound ≈ 2^64 packs).
    ///
    /// # Panics
    /// Panics if the OS entropy source is unavailable. This cannot be handled
    /// meaningfully at the call site: every caller (pack creation, extract,
    /// import) needs a globally unique identity, and silently falling back to
    /// a weaker source would reintroduce the collision risk this method exists
    /// to avoid. OS entropy is available on every supported platform.
    #[must_use]
    pub fn random() -> Self {
        // The all-zero address is the reserved empty sentinel, so a draw that
        // lands on it (probability 2^-128) is discarded rather than returned:
        // `random` guarantees a nonzero identity by contract.
        loop {
            let mut bytes = [0u8; PACK_ID_LEN];
            entropy::fill(&mut bytes).expect("OS entropy source is available");
            let id = Self(bytes);
            if !id.is_zero() {
                return id;
            }
        }
    }

    /// The raw 16 address bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; PACK_ID_LEN] {
        &self.0
    }

    /// Whether this is the all-zero address, which is reserved as the
    /// "no pack"/empty sentinel and is never a valid pack identity.
    ///
    /// [`PackId::random`] never produces it; the zero value only appears as a
    /// placeholder in tombstone records and empty table slots. Trust
    /// boundaries (header reads, hex parsing, pack creation/import) reject it
    /// so a stray zero can never be adopted as a real pack's identity.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&byte| byte == 0)
    }

    /// The full 32 lowercase hex digits of the address.
    #[must_use]
    pub fn as_hex(&self) -> String {
        hex_lower(&self.0)
    }

    /// The canonical filename stem for an un-disambiguated pack:
    /// `pack_<first 16 hex digits>`. See [`PackId::filename_for`] for the
    /// full-width form used when a truncated prefix would collide.
    #[must_use]
    pub fn filename_stem(&self) -> String {
        format!("pack_{}", &hex_lower(&self.0)[..PACK_FILENAME_PREFIX_HEX])
    }

    /// The canonical filename (`stem` + `.pack`) for an un-disambiguated pack.
    #[must_use]
    pub fn filename(&self) -> String {
        format!("{}.pack", self.filename_stem())
    }

    /// The filename to use when creating this pack among `siblings`.
    ///
    /// Normally the compact `pack_<16 hex digits>.pack`. If that 16-hex prefix
    /// collides with an existing sibling, fall back to the full 32-hex name so
    /// the choice is a single deterministic widening at creation time. This is
    /// only consulted when *creating* a pack: an existing file's name is fixed
    /// on disk and never recomputed, so adding a later sibling can never rename
    /// an already-written pack.
    ///
    /// A pack placed in the pool out of band (a manual file copy) bypasses this
    /// check, so a hand-copied pack should use its full 32-hex name: a 16-hex
    /// name cannot be widened later, and a collision then surfaces as a
    /// filename/header disagreement at discovery.
    #[must_use]
    pub fn filename_for(&self, siblings: &[PackId]) -> String {
        let hex = self.as_hex();
        let prefix = &hex[..PACK_FILENAME_PREFIX_HEX];
        let collides = siblings
            .iter()
            .any(|other| other != self && other.as_hex()[..PACK_FILENAME_PREFIX_HEX] == *prefix);
        if collides {
            format!("pack_{hex}.pack")
        } else {
            self.filename()
        }
    }

    /// Parse an address from a filename stem (the part before `.pack`). Accepts
    /// the compact `pack_<16hex>` form, legacy underscore-joined groups of
    /// 16-hex chunks up to 32 digits total, and the full `pack_<32hex>` form
    /// emitted by [`PackId::filename_for`] on collision.
    ///
    /// Returns the parsed prefix bytes and how many hex digits were present, so
    /// callers can match against a full address's prefix.
    ///
    /// # Errors
    /// Returns `None` if the stem is not one of those forms, if the combined
    /// digits exceed a full [`PACK_ID_LEN`] address, or if it is the full
    /// all-zero address (reserved, never a valid identity). A *truncated*
    /// all-zero prefix is still accepted: it can legitimately belong to a
    /// nonzero 128-bit address whose leading bits happen to be zero.
    #[must_use]
    pub fn parse_filename_prefix(stem: &str) -> Option<(Vec<u8>, usize)> {
        let rest = stem.strip_prefix("pack_")?;
        let mut digits = String::new();
        for (index, group) in rest.split('_').enumerate() {
            // The first group is normally the 16-hex truncated prefix, but the
            // collision fallback emitted by [`PackId::filename_for`] is a single
            // full 32-hex address. Later groups are always 16-hex continuation
            // chunks of an underscore-joined prefix.
            let valid_len = group.len() == PACK_FILENAME_PREFIX_HEX
                || (index == 0 && group.len() == PACK_ID_LEN * 2);
            if !valid_len {
                return None;
            }
            if !group
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return None;
            }
            digits.push_str(group);
        }
        if digits.is_empty() {
            return None;
        }
        // Cap at a full address. `filename_for` emits only a 16-hex or 32-hex
        // name; without this a longer underscore-joined stem would parse here
        // and then fail the prefix-agreement check downstream.
        if digits.len() > PACK_ID_LEN * 2 {
            return None;
        }
        let byte_len = digits.len() / 2;
        let mut bytes = vec![0u8; byte_len];
        for (index, chunk) in digits.as_bytes().chunks_exact(2).enumerate() {
            let hi = hex_value(chunk[0])?;
            let lo = hex_value(chunk[1])?;
            bytes[index] = (hi << 4) | lo;
        }
        if bytes.len() == PACK_ID_LEN && bytes.iter().all(|&byte| byte == 0) {
            return None;
        }
        Some((bytes, digits.len()))
    }

    /// Parse a full address from exactly 32 lowercase hex digits, without a
    /// `0x` prefix.
    ///
    /// # Errors
    /// Returns `None` for the wrong length, any non-lowercase-hex character,
    /// or the all-zero address (reserved, never a valid identity).
    #[must_use]
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != PACK_ID_LEN * 2 {
            return None;
        }
        let mut bytes = [0u8; PACK_ID_LEN];
        for (index, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
            let hi = hex_value(chunk[0])?;
            let lo = hex_value(chunk[1])?;
            bytes[index] = (hi << 4) | lo;
        }
        let id = Self(bytes);
        if id.is_zero() {
            return None;
        }
        Some(id)
    }

    /// Whether this address's lowercase hex begins with the given lowercase
    /// hex prefix.
    #[must_use]
    pub fn has_hex_prefix(&self, prefix: &str) -> bool {
        hex_lower(&self.0).starts_with(prefix)
    }
}

impl std::fmt::Debug for PackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PackId(0x{})", hex_lower(&self.0))
    }
}

impl std::fmt::Display for PackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", hex_lower(&self.0))
    }
}

/// Lowercase hex encoding without pulling in a formatter dependency.
///
/// Public so sibling modules (e.g. `shard`) can render address prefixes when
/// validating filename agreement.
#[must_use]
pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

/// Decode one lowercase-hex nibble, rejecting uppercase and non-hex.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0' => Some(0),
        b'1' => Some(1),
        b'2' => Some(2),
        b'3' => Some(3),
        b'4' => Some(4),
        b'5' => Some(5),
        b'6' => Some(6),
        b'7' => Some(7),
        b'8' => Some(8),
        b'9' => Some(9),
        b'a' => Some(10),
        b'b' => Some(11),
        b'c' => Some(12),
        b'd' => Some(13),
        b'e' => Some(14),
        b'f' => Some(15),
        _ => None,
    }
}

/// Byte layout within the reserved header, up to where the CRC starts.
/// Everything from `CRC_COVERED_LEN` to `HEADER_LEN` is the CRC itself
/// (4 bytes) followed by zero padding.
const CRC_COVERED_LEN: usize = 4 // magic
    + 1 // version
    + 4 // header_len (u32)
    + PACK_ID_LEN // pack_id
    + 8 // created_at (u64, unix seconds)
    + 4; // feature_flags (u32, reserved)

/// A shard's immutable descriptor, parsed from its reserved header.
///
/// Recording the [`PackId`] in the file itself (not just its filename)
/// lets a reader detect a shard file that's been copied or renamed
/// inconsistently — the two should always agree (by prefix), and a mismatch
/// means something outside mtxdb moved this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardHeader {
    /// The pack's globally unique, immutable identity.
    pub pack_id: PackId,
    /// Unix-seconds creation timestamp (telemetry only; never identity).
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
/// be read or validated, or its length cannot be obtained. A file that
/// [`read_header`] identifies as a non-packfile yields an empty scanner.
pub fn scan_packfile_iter(path: &Path, verify_payload: bool) -> io::Result<PackfileScanner> {
    let file = File::open(path)?;
    scan_packfile_iter_from_file(file, verify_payload)
}

/// Build a scanner from an already-open shard handle. Collection snapshots
/// open each shard while holding the collection lock, then perform the
/// potentially long scan after releasing that lock.
///
/// The handle must be positioned at the start of the header. Captures its
/// current length and uses the same `verify_payload` behavior as
/// [`scan_packfile_iter`]. A non-packfile yields an empty scanner; metadata,
/// header-read, and header-validation errors propagate.
pub(crate) fn scan_packfile_iter_from_file(
    file: File,
    verify_payload: bool,
) -> io::Result<PackfileScanner> {
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
    read_record_metadata_scratch(reader, &mut [0u8; SCAN_DISCARD_BUF_LEN])
}

/// [`read_record_metadata`] with the payload-discard buffer supplied by the
/// caller. A scan over many frames passes one buffer for all of them: zeroing a
/// fresh 8 KiB for every frame costs more than parsing a small one.
fn read_record_metadata_scratch(
    reader: &mut impl Read,
    discard: &mut [u8; SCAN_DISCARD_BUF_LEN],
) -> io::Result<Option<RecordMetadata>> {
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
/// Returns `io::Error` on write failure, if the system clock is before the
/// Unix epoch (treated as a hard error rather than silently recording a wrong
/// creation time), or if `pack_id` is the reserved all-zero address — refused
/// so a creation or import path can never persist it as a real identity.
///
/// # Panics
/// Never in practice: the only internal conversion (`HEADER_LEN` as
/// `u32`) is a compile-time constant well within range.
pub fn write_header(writer: &mut impl Write, pack_id: &PackId) -> io::Result<()> {
    if pack_id.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to write the reserved all-zero pack id to a header",
        ));
    }
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs();

    let mut buf = [0u8; HEADER_LEN];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    let header_len = u32::try_from(HEADER_LEN).expect("HEADER_LEN fits in u32");
    buf[5..9].copy_from_slice(&header_len.to_le_bytes());
    let mut cursor: usize = 9;
    buf[cursor..cursor.saturating_add(PACK_ID_LEN)].copy_from_slice(pack_id.as_bytes());
    cursor = cursor.saturating_add(PACK_ID_LEN);
    buf[cursor..cursor.saturating_add(8)].copy_from_slice(&created_at.to_le_bytes());
    cursor = cursor.saturating_add(8);
    // feature_flags (u32) follows; stays zero — reserved for future use.
    let _ = cursor;

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
/// Returns `Ok(None)` for a file shorter than the four-byte magic or whose
/// first four bytes don't match [`MAGIC`]. Everything else that's wrong is a
/// real `Err`, not a quiet `None`, specifically so a caller (or one of the `scan_*` functions,
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
/// [`VERSION`], or (`InvalidData`) if the header is truncated after the magic,
/// declares the wrong length, has a CRC mismatch, or contains an all-zero pack
/// ID. Other I/O errors propagate.
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

    let mut cursor: usize = 9;
    let mut pack_id = [0u8; PACK_ID_LEN];
    pack_id.copy_from_slice(&buf[cursor..cursor.saturating_add(PACK_ID_LEN)]);
    let pack_id = PackId(pack_id);
    // The all-zero address is the empty sentinel and never a real identity.
    // Checked after the CRC so the rejection is based on authenticated bytes.
    if pack_id.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard header carries the reserved all-zero pack id",
        ));
    }
    cursor = cursor.saturating_add(PACK_ID_LEN);
    let created_at = u64::from_le_bytes(buf[cursor..cursor.saturating_add(8)].try_into().unwrap());

    Ok(Some(ShardHeader {
        pack_id,
        created_at,
    }))
}

/// Open or create a packfile, writing the header if it's new.
///
/// With `create` set, opens for writing and initializes and syncs the header
/// of an empty file. Otherwise opens an existing file read-only.
///
/// `pack_id` is the caller's expected full identity, written on initialization
/// and checked against an existing header. This function does not validate the
/// filename, which may contain only a prefix of the identity.
///
/// # Errors
/// Propagates open, metadata, header-write, and sync errors. Returns
/// `InvalidInput` when initializing with an all-zero ID, `Unsupported` for an
/// unsupported header version, or `InvalidData` for an invalid header or an
/// identity mismatch.
pub fn open_packfile(path: &Path, create: bool, pack_id: &PackId) -> io::Result<File> {
    if create {
        let mut options = OpenOptions::new();
        options.read(true).create(true);
        // Unix flushes use positioned writes through the append-capable
        // handle. Windows' append access masks FILE_WRITE_DATA, which is
        // required by set_len during torn-tail rollback; its write path
        // therefore opens an ordinary writable handle instead.
        #[cfg(unix)]
        options.append(true);
        #[cfg(not(unix))]
        options.write(true);
        let mut file = options.open(path)?;

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
            if header.pack_id != *pack_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard file {} identifies itself as {} in its header, \
                         but its filename says {} — \
                         copied or renamed inconsistently with its own history",
                        path.display(),
                        header.pack_id,
                        pack_id,
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
        if header.pack_id != *pack_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shard file {} identifies itself as {} in its header, \
                     but its filename says {} — \
                     copied or renamed inconsistently with its own history",
                    path.display(),
                    header.pack_id,
                    pack_id,
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

/// Streaming reader yielding each frame's full [`Record`] — payload, identity,
/// and optional metadata — in file order, paired with the frame's byte offset.
///
/// This is the materializing counterpart to [`scan_packfile`] (which reads only
/// identity metadata). A torn tail (a crashed append) is treated as end-of-file,
/// matching the other scanners; mid-file corruption is propagated as an error.
#[derive(Debug)]
pub struct RecordScanner {
    reader: BufReader<File>,
    done: bool,
    torn_tail: bool,
}

impl RecordScanner {
    /// Whether the scan stopped at a torn (truncated) trailing frame rather
    /// than at a clean frame boundary. A concurrent writer can legitimately
    /// leave an in-flight torn tail, so this is a diagnostic signal, not
    /// necessarily corruption.
    #[must_use]
    pub fn torn_tail(&self) -> bool {
        self.torn_tail
    }
}

/// Open a streaming scanner yielding full [`Record`]s from a packfile.
///
/// # Errors
/// Returns an I/O error if the packfile cannot be opened or its header cannot
/// be read, and [`io::ErrorKind::InvalidData`] if the file does not start with
/// an mtxdb packfile header (rather than silently yielding an empty stream).
pub fn scan_records_iter(path: &Path) -> io::Result<RecordScanner> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    match read_header(&mut reader)? {
        Some(_) => Ok(RecordScanner {
            reader,
            done: false,
            torn_tail: false,
        }),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "not an mtxdb packfile (missing magic header): {}",
                path.display()
            ),
        )),
    }
}

impl Iterator for RecordScanner {
    type Item = io::Result<(u64, Record)>;

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
        match read_record(&mut self.reader) {
            Ok(Some(record)) => Some(Ok((offset, record))),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                self.done = true;
                self.torn_tail = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}

impl std::iter::FusedIterator for RecordScanner {}

/// What [`extract_packfile_collection`] extracted from a packfile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackExtractStats {
    /// Number of frames extracted for the target collection.
    pub frames_extracted: u64,
    /// Total bytes of extracted frames written (excluding the header).
    pub frame_bytes_written: u64,
    /// Total frames scanned in the source packfile.
    pub frames_scanned: u64,
    /// Whether the scan stopped at a torn trailing frame.
    pub torn_tail: bool,
}

/// Extract all frames belonging to `target_collection` from `source_path` and write
/// them verbatim into a new valid packfile at `dest_path` with header `dest_pack_id`.
///
/// Every frame's framing and CRC is verified. Frame bytes (including compression
/// and optional frame metadata) are copied without decompressing or re-serializing,
/// ensuring byte-level fidelity. If the source pack ends in a torn tail, extraction
/// stops cleanly at the last valid frame and records `torn_tail: true` in the returned
/// stats.
fn validate_extracted_frame(frame_buf: &[u8], frame_len: u32) -> io::Result<()> {
    let total_frame_len = frame_buf.len();
    let flags = *frame_buf
        .get(4)
        .ok_or_else(|| invalid_data("truncated frame"))?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported record flags: {flags:#04x}"),
        ));
    }

    if flags & FLAG_METADATA != 0 {
        let meta_offset = (4_usize).saturating_add(FRAME_FIXED_LEN as usize);
        let header_end = meta_offset.saturating_add(5).saturating_add(4);
        if total_frame_len < header_end {
            return Err(invalid_data("frame too short for metadata header"));
        }
        let version = *frame_buf
            .get(meta_offset)
            .ok_or_else(|| invalid_data("missing metadata version"))?;
        if version != METADATA_VERSION {
            return Err(invalid_data(&format!(
                "frame metadata: unsupported version {version}"
            )));
        }
        let tlv_bytes: [u8; 4] = frame_buf
            .get(meta_offset.saturating_add(1)..meta_offset.saturating_add(5))
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| invalid_data("truncated metadata length prefix"))?;
        let tlv_len = u32::from_le_bytes(tlv_bytes);
        let _ = checked_metadata_block_len(tlv_len, frame_len)?;
    }

    if flags & FLAG_CRC_DISABLED == 0 {
        let crc_start = total_frame_len.saturating_sub(4);
        let expected_bytes: [u8; 4] = frame_buf
            .get(crc_start..total_frame_len)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| invalid_data("truncated CRC"))?;
        let expected_crc = u32::from_le_bytes(expected_bytes);
        let actual_crc = crc32fast::hash(&frame_buf[..crc_start]);
        if actual_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "frame CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                ),
            ));
        }
    }

    Ok(())
}

/// Extract all frames belonging to `target_collection` from `source_path` and write
/// them verbatim into a new valid packfile at `dest_path` with header `dest_pack_id`.
///
/// Every frame's framing and CRC is verified. Frame bytes (including compression
/// and optional frame metadata) are copied without decompressing or re-serializing,
/// ensuring byte-level fidelity. If the source pack ends in a torn tail, extraction
/// stops cleanly at the last valid frame and records `torn_tail: true` in the returned
/// stats. The destination is created or truncated, then flushed and synced on
/// success; an error may leave partial output. The caller must ensure it is a
/// different file from the source.
///
/// # Errors
/// Propagates file I/O, header-validation, header-write, and sync errors,
/// including `InvalidInput` for an all-zero `dest_pack_id`. Invalid frame
/// lengths, framing, or CRCs return `InvalidData`.
pub fn extract_packfile_collection(
    source_path: &Path,
    dest_path: &Path,
    target_collection: &[u8; 16],
    dest_pack_id: &PackId,
) -> io::Result<PackExtractStats> {
    let src_file = File::open(source_path)?;
    let mut reader = BufReader::with_capacity(RECOVERY_BUFFER_BYTES, src_file);

    let Some(_header) = read_header(&mut reader)? else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "not an mtxdb packfile (missing magic header): {}",
                source_path.display()
            ),
        ));
    };

    let dst_file = File::create(dest_path)?;
    let mut writer = BufWriter::new(dst_file);
    write_header(&mut writer, dest_pack_id)?;

    let mut stats = PackExtractStats {
        frames_extracted: 0,
        frame_bytes_written: 0,
        frames_scanned: 0,
        torn_tail: false,
    };

    let mut frame_buf = Vec::new();

    loop {
        let len_buf = match read_frame_len_prefix(&mut reader) {
            Ok(Some(buf)) => buf,
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                stats.torn_tail = true;
                break;
            }
            Err(e) => return Err(e),
        };

        let frame_len = u32::from_le_bytes(len_buf);
        if !(FRAME_FIXED_LEN..=MAX_RECORD_LEN).contains(&frame_len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid record length: {frame_len}"),
            ));
        }

        let total_frame_len = (frame_len as usize).saturating_add(8);
        frame_buf.resize(total_frame_len, 0);
        frame_buf[0..4].copy_from_slice(&len_buf);

        match reader.read_exact(&mut frame_buf[4..total_frame_len]) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                stats.torn_tail = true;
                break;
            }
            Err(e) => return Err(e),
        }

        validate_extracted_frame(&frame_buf, frame_len)?;

        stats.frames_scanned = stats.frames_scanned.saturating_add(1);

        let mut frame_collection = [0u8; 16];
        frame_collection.copy_from_slice(&frame_buf[9..25]);

        if frame_collection == *target_collection {
            writer.write_all(&frame_buf)?;
            stats.frames_extracted = stats.frames_extracted.saturating_add(1);
            stats.frame_bytes_written = stats
                .frame_bytes_written
                .saturating_add(u64::try_from(total_frame_len).unwrap_or(u64::MAX));
        }
    }

    writer.flush()?;
    writer.get_ref().sync_all()?;

    Ok(stats)
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
    let mut entries = Vec::new();
    recover_packfile_with(path, |entry| entries.push(entry))?;
    Ok(entries)
}

/// What [`recover_packfile`] found in one pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackRecovery {
    /// Complete, CRC-valid records in the pack.
    pub records: u64,
    /// Offset just past the last valid record (the header's end for an empty pack).
    pub valid_len: u64,
    /// Whether a torn tail was cut off at `valid_len`.
    pub truncated: bool,
}

/// [`scan_and_recover_packfile`] for a caller that only needs the pack made
/// safe to append to: the same validation of every frame and CRC, and the same
/// truncation of a torn tail, but no per-record result is built.
///
/// # Errors
/// As [`scan_and_recover_packfile`].
pub fn recover_packfile(path: &Path) -> io::Result<PackRecovery> {
    recover_packfile_with(path, |_| {})
}

/// Buffer for a recovery scan: a few large sequential reads, not one per 8 KiB.
const RECOVERY_BUFFER_BYTES: usize = 1 << 20;

/// A reader that counts the bytes it hands out, so the scan knows each frame's
/// offset without asking the file for its position (an `lseek` system call per
/// frame on a buffered reader).
struct CountingReader<R> {
    inner: R,
    position: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.position = self
            .position
            .saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        Ok(read)
    }
}

/// The shared recovery walk: validates every frame, calls `on_record` for each
/// valid one, and truncates a torn tail.
fn recover_packfile_with(
    path: &Path,
    mut on_record: impl FnMut(ScanEntry),
) -> io::Result<PackRecovery> {
    let file = File::open(path)?;
    let mut reader = CountingReader {
        inner: BufReader::with_capacity(RECOVERY_BUFFER_BYTES, file),
        position: 0,
    };
    let mut recovery = PackRecovery {
        records: 0,
        valid_len: 0,
        truncated: false,
    };

    if read_header(&mut reader)?.is_none() {
        return Ok(recovery);
    }

    recovery.valid_len = reader.position;
    let mut scratch = [0u8; SCAN_DISCARD_BUF_LEN];
    loop {
        let offset = reader.position;
        match read_record_metadata_scratch(&mut reader, &mut scratch) {
            Ok(Some(meta)) => {
                on_record((meta.collection_id, meta.hash, offset));
                recovery.records = recovery.records.saturating_add(1);
                recovery.valid_len = reader.position;
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Torn tail: a write crashed mid-frame. Truncate to the
                // last valid record boundary so future appends don't land
                // after a corrupt frame.
                recovery.truncated = true;
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
    if recovery.truncated {
        drop(reader);
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(recovery.valid_len)?;
        file.sync_all()?;
    }

    Ok(recovery)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_packfile.rs"]
mod tests;
