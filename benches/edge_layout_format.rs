//! The prototype's id -> key chunk format as pure functions, shared by the
//! `edge_layout_*` benches and `tests/edge_layout_format.rs`.
//!
//! Chunk size is an explicit parameter everywhere, so tests need no process-wide
//! state. Each bench target and the test target use a subset of these items.
#![allow(dead_code)]

/// Most ids a chunk may hold: with 255-byte keys the last body offset is 65,280,
/// which still fits the `u16` offset table.
pub(crate) const MAX_CHUNK: usize = 256;
/// Longest key a chunk entry can hold (one length byte).
pub(crate) const MAX_KEY: usize = 255;
/// Magic, chunk number and entry count, before the offset table.
pub(crate) const KEY_HEADER: usize = 10;
const KEY_MAGIC: &[u8; 4] = b"PREV";

/// Why a chunk could not be encoded or read.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FormatError {
    /// More keys than [`MAX_CHUNK`].
    TooManyKeys,
    /// A key longer than [`MAX_KEY`].
    KeyTooLong,
    /// A body offset that does not fit `u16`.
    OffsetOverflow,
    /// The record ends before something it promises.
    Truncated,
    /// The record does not start with the key-chunk magic.
    BadMagic,
    /// The entry count exceeds [`MAX_CHUNK`].
    BadCount,
    /// The requested position is not below the entry count.
    PositionOutOfRange,
}

/// The chunk that holds `id` and its position inside it.
pub(crate) fn locate(id: usize, size: usize) -> (usize, usize) {
    (
        id.checked_div(size).unwrap_or(0),
        id.checked_rem(size).unwrap_or(0),
    )
}

/// First id of `chunk` and one past its last, for `len` ids in total.
pub(crate) fn chunk_bounds(chunk: usize, size: usize, len: usize) -> (usize, usize) {
    let start = chunk.saturating_mul(size);
    (start, start.saturating_add(size).min(len))
}

/// Encode one id -> key chunk: magic, chunk number, entry count, a `u16` body
/// offset per entry, then the entries (length byte + key).
pub(crate) fn encode_key_chunk(keys: &[&str], chunk: u32) -> Result<Vec<u8>, FormatError> {
    if keys.len() > MAX_CHUNK {
        return Err(FormatError::TooManyKeys);
    }
    let mut body = Vec::new();
    let mut offsets = Vec::with_capacity(keys.len());
    for key in keys {
        offsets.push(u16::try_from(body.len()).map_err(|_| FormatError::OffsetOverflow)?);
        body.push(u8::try_from(key.len()).map_err(|_| FormatError::KeyTooLong)?);
        body.extend_from_slice(key.as_bytes());
    }
    let count = u16::try_from(offsets.len()).map_err(|_| FormatError::TooManyKeys)?;
    let mut out = KEY_MAGIC.to_vec();
    out.extend_from_slice(&chunk.to_be_bytes());
    out.extend_from_slice(&count.to_be_bytes());
    for offset in offsets {
        out.extend_from_slice(&offset.to_be_bytes());
    }
    out.extend_from_slice(&body);
    Ok(out)
}

/// The key at `pos` in an encoded key chunk. Never panics on malformed input.
pub(crate) fn key_at(chunk: &[u8], pos: usize) -> Result<&[u8], FormatError> {
    if chunk.get(..4) != Some(&KEY_MAGIC[..]) {
        return Err(if chunk.len() < 4 {
            FormatError::Truncated
        } else {
            FormatError::BadMagic
        });
    }
    let count = usize::from(u16::from_be_bytes(
        chunk
            .get(8..KEY_HEADER)
            .and_then(|b| b.try_into().ok())
            .ok_or(FormatError::Truncated)?,
    ));
    if count > MAX_CHUNK {
        return Err(FormatError::BadCount);
    }
    if pos >= count {
        return Err(FormatError::PositionOutOfRange);
    }
    let slot = KEY_HEADER.saturating_add(pos.saturating_mul(2));
    let off = usize::from(u16::from_be_bytes(
        chunk
            .get(slot..slot.saturating_add(2))
            .and_then(|b| b.try_into().ok())
            .ok_or(FormatError::Truncated)?,
    ));
    let head = KEY_HEADER
        .saturating_add(count.saturating_mul(2))
        .saturating_add(off);
    let len = usize::from(*chunk.get(head).ok_or(FormatError::Truncated)?);
    let start = head.saturating_add(1);
    chunk
        .get(start..start.saturating_add(len))
        .ok_or(FormatError::Truncated)
}

/// Bytes of one forward-run entry: 16-byte key hash and a big-endian id.
pub(crate) const RUN_ENTRY: usize = 20;

/// A 128-bit key hash. It selects candidates; it never establishes identity.
pub(crate) type KeyHash = [u8; 16];

/// A forward run record: every `(hash, id)` pair, sorted by hash then id. Equal
/// hashes are kept, not deduplicated; they form a contiguous collision bucket.
pub(crate) fn encode_run(entries: &mut [(KeyHash, u32)]) -> Vec<u8> {
    entries.sort_unstable();
    let mut out = b"PFWD".to_vec();
    out.extend_from_slice(&u32::try_from(entries.len()).expect("run").to_be_bytes());
    for (hash, id) in entries.iter() {
        out.extend_from_slice(hash);
        out.extend_from_slice(&id.to_be_bytes());
    }
    out
}

/// Every id stored under `hash` in a run record's entries (the bytes after its
/// 8-byte header), in id order.
pub(crate) fn run_candidates(entries: &[u8], hash: &KeyHash) -> Vec<u32> {
    let count = entries.len() / RUN_ENTRY;
    let hash_at = |i: usize| entries.get(i.saturating_mul(RUN_ENTRY)..).map(|e| &e[..16]);
    let (mut lo, mut hi) = (0_usize, count);
    while lo < hi {
        let mid = lo.saturating_add(hi.saturating_sub(lo) / 2);
        if hash_at(mid).is_some_and(|h| h < &hash[..]) {
            lo = mid.saturating_add(1);
        } else {
            hi = mid;
        }
    }
    let mut out = Vec::new();
    for i in lo..count {
        let at = i.saturating_mul(RUN_ENTRY);
        let Some(entry) = entries.get(at..at.saturating_add(RUN_ENTRY)) else {
            break;
        };
        if entry[..16] != hash[..] {
            break;
        }
        out.push(u32::from_be_bytes([
            entry[16], entry[17], entry[18], entry[19],
        ]));
    }
    out
}

/// Every id stored under `hash` in all runs merged into one sorted vector.
pub(crate) fn merged_candidates(sorted: &[(KeyHash, u32)], hash: &KeyHash) -> Vec<u32> {
    let from = sorted.partition_point(|entry| entry.0 < *hash);
    sorted[from..]
        .iter()
        .take_while(|entry| entry.0 == *hash)
        .map(|entry| entry.1)
        .collect()
}

/// Why a lookup could not give an answer. Both fail closed: neither is a miss.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LookupError {
    /// A candidate's key could not be read, so it cannot be ruled out.
    KeyUnreadable(u32),
    /// Two candidates carry the same full key.
    DuplicateKey,
}

/// Confirm `candidates` against `key`: exactly one stored key must match. A miss
/// is returned only after every candidate was read successfully.
pub(crate) fn confirm(
    candidates: &[u32],
    key: &str,
    key_of: &mut dyn FnMut(u32) -> Option<String>,
) -> Result<Option<u32>, LookupError> {
    let mut found = None;
    for id in candidates {
        let stored = key_of(*id).ok_or(LookupError::KeyUnreadable(*id))?;
        if stored == key && found.replace(*id).is_some() {
            return Err(LookupError::DuplicateKey);
        }
    }
    Ok(found)
}
