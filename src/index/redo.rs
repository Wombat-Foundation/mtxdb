//! Logical redo records: the proposed replacement for the physical
//! [`DeltaFrame`](super::format::DeltaFrame) (design: `docs/2026-09-27-index-checkpoint-design.md`,
//! "Proposed logical-redo contract"). **Scaffold only**: nothing writes or reads
//! these records in the delta log yet.
//!
//! A record names a change by content identity, not by table position, so it
//! replays into a table of any capacity and a resize cannot invalidate it. It is
//! a fixed 80-byte little-endian payload; the outer delta operation supplies the
//! frame type, length and CRC.
//!
//! ```text
//!   0..16   collection_id
//!  16       op                1 = set, 2 = collection tombstone
//!  17       flags             must be zero
//!  18..20   reserved          must be zero
//!  20..36   full_hash         zero for a tombstone
//!  36..44   pack_id           zero for a tombstone
//!  44..48   reserved          must be zero (was the process-local shard slot)
//!  48..56   offset            zero for a tombstone; a set above MAX_OFFSET is rejected
//!  56..60   record_len        zero for a tombstone; nonzero for a set
//!  60..68   delta_seq         per-pool ordering key, not a WAL LSN
//!  68..76   base_generation   collection incarnation being continued
//!  76..80   reserved          must be zero
//! ```

use super::IndexEntry;

/// Encoded length of one record.
pub const REDO_RECORD_LEN: usize = 80;

const OP_SET: u8 = 1;
const OP_COLLECTION_TOMBSTONE: u8 = 2;

/// What a record does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedoOp {
    /// Point `full_hash` at the record at `(pack_id, offset)`. The locator is a
    /// hint: replay must resolve it and confirm the hash before installing it.
    Set {
        /// The authoritative content identity.
        full_hash: [u8; 16],
        /// The pack holding the record (stable across processes, unlike a slot).
        pack_id: u64,
        /// Byte offset of the frame within the pack.
        offset: u64,
        /// Expected on-disk length of the frame.
        record_len: u32,
    },
    /// Drop the whole collection. The native index has no per-key delete.
    CollectionTombstone,
}

/// One logical redo record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RedoRecord {
    /// The collection the record applies to.
    pub collection_id: [u8; 16],
    /// The change.
    pub op: RedoOp,
    /// Per-pool ordering key, assigned when the mutation is recorded.
    pub delta_seq: u64,
    /// The collection incarnation this record continues.
    pub base_generation: u64,
}

/// Why a record or a sequence of records was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedoError {
    /// The payload is not exactly [`REDO_RECORD_LEN`] bytes.
    BadLength(usize),
    /// The operation byte is neither a set nor a collection tombstone.
    UnknownOp(u8),
    /// A byte that must be zero is not.
    NonzeroReserved,
    /// A set's offset does not fit an index entry.
    OffsetTooLarge(u64),
    /// A set carries a zero record length.
    ZeroRecordLen,
    /// A tombstone carries an identity or a locator.
    TombstoneCarriesLocator,
    /// `delta_seq` did not strictly increase (or did not start above the base).
    NonMonotonic {
        /// The sequence number that had to be exceeded.
        after: u64,
        /// The offending sequence number.
        got: u64,
    },
}

impl std::fmt::Display for RedoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadLength(len) => write!(f, "redo record is {len} bytes, not {REDO_RECORD_LEN}"),
            Self::UnknownOp(op) => write!(f, "unknown redo operation {op}"),
            Self::NonzeroReserved => write!(f, "a reserved redo byte is not zero"),
            Self::OffsetTooLarge(offset) => {
                write!(f, "redo offset {offset} exceeds the index's maximum")
            }
            Self::ZeroRecordLen => write!(f, "a redo set has a zero record length"),
            Self::TombstoneCarriesLocator => {
                write!(f, "a redo tombstone carries an identity or locator")
            }
            Self::NonMonotonic { after, got } => {
                write!(f, "delta_seq {got} does not exceed {after}")
            }
        }
    }
}

impl std::error::Error for RedoError {}

fn put(buf: &mut [u8; REDO_RECORD_LEN], at: usize, bytes: &[u8]) {
    if let Some(dest) = buf.get_mut(at..at.saturating_add(bytes.len())) {
        dest.copy_from_slice(bytes);
    }
}

fn take<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    let mut out = [0u8; N];
    if let Some(src) = bytes.get(at..at.saturating_add(N)) {
        out.copy_from_slice(src);
    }
    out
}

impl RedoRecord {
    /// Encode the record.
    ///
    /// # Errors
    /// A set whose offset exceeds [`IndexEntry::MAX_OFFSET`] or whose record
    /// length is zero cannot be represented.
    pub fn encode(&self) -> Result<[u8; REDO_RECORD_LEN], RedoError> {
        let mut buf = [0u8; REDO_RECORD_LEN];
        put(&mut buf, 0, &self.collection_id);
        match self.op {
            RedoOp::Set {
                full_hash,
                pack_id,
                offset,
                record_len,
            } => {
                if offset > IndexEntry::MAX_OFFSET {
                    return Err(RedoError::OffsetTooLarge(offset));
                }
                if record_len == 0 {
                    return Err(RedoError::ZeroRecordLen);
                }
                put(&mut buf, 16, &[OP_SET]);
                put(&mut buf, 20, &full_hash);
                put(&mut buf, 36, &pack_id.to_le_bytes());
                put(&mut buf, 48, &offset.to_le_bytes());
                put(&mut buf, 56, &record_len.to_le_bytes());
            }
            RedoOp::CollectionTombstone => put(&mut buf, 16, &[OP_COLLECTION_TOMBSTONE]),
        }
        put(&mut buf, 60, &self.delta_seq.to_le_bytes());
        put(&mut buf, 68, &self.base_generation.to_le_bytes());
        Ok(buf)
    }

    /// Decode exactly one record.
    ///
    /// # Errors
    /// Any deviation from the layout: wrong length, unknown operation, a nonzero
    /// reserved byte, an offset above the index's maximum, a zero record length
    /// on a set, or an identity or locator on a tombstone.
    pub fn decode(bytes: &[u8]) -> Result<Self, RedoError> {
        if bytes.len() != REDO_RECORD_LEN {
            return Err(RedoError::BadLength(bytes.len()));
        }
        let reserved_clear = [17..20, 44..48, 76..80].into_iter().all(|range| {
            bytes
                .get(range)
                .is_some_and(|part| part.iter().all(|b| *b == 0))
        });
        if !reserved_clear {
            return Err(RedoError::NonzeroReserved);
        }
        let op_byte = bytes.get(16).copied().unwrap_or(0);
        let full_hash = take::<16>(bytes, 20);
        let pack_id = u64::from_le_bytes(take(bytes, 36));
        let offset = u64::from_le_bytes(take(bytes, 48));
        let record_len = u32::from_le_bytes(take(bytes, 56));
        let op = match op_byte {
            OP_SET => {
                if offset > IndexEntry::MAX_OFFSET {
                    return Err(RedoError::OffsetTooLarge(offset));
                }
                if record_len == 0 {
                    return Err(RedoError::ZeroRecordLen);
                }
                RedoOp::Set {
                    full_hash,
                    pack_id,
                    offset,
                    record_len,
                }
            }
            OP_COLLECTION_TOMBSTONE => {
                if full_hash != [0; 16] || pack_id != 0 || offset != 0 || record_len != 0 {
                    return Err(RedoError::TombstoneCarriesLocator);
                }
                RedoOp::CollectionTombstone
            }
            other => return Err(RedoError::UnknownOp(other)),
        };
        Ok(Self {
            collection_id: take(bytes, 0),
            op,
            delta_seq: u64::from_le_bytes(take(bytes, 60)),
            base_generation: u64::from_le_bytes(take(bytes, 68)),
        })
    }
}

/// Check that a log's records are in strictly increasing `delta_seq` order and
/// that the first is above `base_sequence` (the last sequence the checkpoint
/// incorporates). Replay rejects the log otherwise.
///
/// # Errors
/// [`RedoError::NonMonotonic`] at the first record that does not exceed its
/// predecessor.
pub fn validate_sequence(records: &[RedoRecord], base_sequence: u64) -> Result<(), RedoError> {
    let mut after = base_sequence;
    for record in records {
        if record.delta_seq <= after {
            return Err(RedoError::NonMonotonic {
                after,
                got: record.delta_seq,
            });
        }
        after = record.delta_seq;
    }
    Ok(())
}

#[cfg(test)]
#[path = "test_redo.rs"]
mod tests;
