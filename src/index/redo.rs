//! Logical redo records: the proposed replacement for the physical
//! [`DeltaFrame`](super::format::DeltaFrame) (design: `docs/2026-09-27-index-checkpoint-design.md`,
//! "Proposed logical-redo contract"). **Scaffold only**: nothing writes or reads
//! these records in the delta log yet.
//!
//! A record names a change by content identity, not by table position, so it
//! replays into a table of any capacity and a resize cannot invalidate it. It is
//! a fixed 100-byte little-endian payload; the outer delta operation supplies the
//! frame type, length and CRC.
//!
//! ```text
//!   0..16    collection_id
//!  16        op                1 = set, 2 = collection tombstone
//!  17        flags             must be zero
//!  18..20    reserved          must be zero
//!  20..36    full_hash         zero for a tombstone
//!  36..68    pack_id           32-byte global pack identity, zero for a tombstone
//!  68..76    offset            zero for a tombstone; a set above MAX_OFFSET is rejected
//!  76..80    record_len        zero for a tombstone; nonzero for a set
//!  80..88    delta_seq         per-pool ordering key, not a WAL LSN
//!  88..96    base_generation   collection incarnation being continued
//!  96..100   reserved          must be zero
//! ```

use crate::packfile::PackId;

use super::IndexEntry;

/// Encoded length of one record.
pub const REDO_RECORD_LEN: usize = 100;

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
        /// The pack holding the record (globally unique and immutable).
        pack_id: PackId,
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
                put(&mut buf, 36, pack_id.as_bytes());
                put(&mut buf, 68, &offset.to_le_bytes());
                put(&mut buf, 76, &record_len.to_le_bytes());
            }
            RedoOp::CollectionTombstone => put(&mut buf, 16, &[OP_COLLECTION_TOMBSTONE]),
        }
        put(&mut buf, 80, &self.delta_seq.to_le_bytes());
        put(&mut buf, 88, &self.base_generation.to_le_bytes());
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
        let reserved_clear = [17..20, 96..100].into_iter().all(|range| {
            bytes
                .get(range)
                .is_some_and(|part| part.iter().all(|b| *b == 0))
        });
        if !reserved_clear {
            return Err(RedoError::NonzeroReserved);
        }
        let op_byte = bytes.get(16).copied().unwrap_or(0);
        let full_hash = take::<16>(bytes, 20);
        let pack_id = PackId(take::<32>(bytes, 36));
        let offset = u64::from_le_bytes(take(bytes, 68));
        let record_len = u32::from_le_bytes(take(bytes, 76));
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
                if full_hash != [0; 16]
                    || pack_id != PackId([0; 32])
                    || offset != 0
                    || record_len != 0
                {
                    return Err(RedoError::TombstoneCarriesLocator);
                }
                RedoOp::CollectionTombstone
            }
            other => return Err(RedoError::UnknownOp(other)),
        };
        Ok(Self {
            collection_id: take(bytes, 0),
            op,
            delta_seq: u64::from_le_bytes(take(bytes, 80)),
            base_generation: u64::from_le_bytes(take(bytes, 88)),
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
mod tests {
    use super::*;

    fn set(seq: u64) -> RedoRecord {
        RedoRecord {
            collection_id: [0xC1; 16],
            op: RedoOp::Set {
                full_hash: [0xA5; 16],
                pack_id: PackId([7; 32]),
                offset: 4096,
                record_len: 61,
            },
            delta_seq: seq,
            base_generation: 3,
        }
    }

    fn tombstone(seq: u64) -> RedoRecord {
        RedoRecord {
            collection_id: [0xC2; 16],
            op: RedoOp::CollectionTombstone,
            delta_seq: seq,
            base_generation: 4,
        }
    }

    #[test]
    fn a_record_round_trips() {
        for record in [set(9), tombstone(10)] {
            let bytes = record.encode().unwrap();
            assert_eq!(bytes.len(), REDO_RECORD_LEN);
            assert_eq!(RedoRecord::decode(&bytes), Ok(record));
        }
    }

    #[test]
    fn every_wrong_length_is_rejected() {
        let bytes = set(1).encode().unwrap();
        for len in 0..REDO_RECORD_LEN {
            assert_eq!(
                RedoRecord::decode(&bytes[..len]),
                Err(RedoError::BadLength(len)),
                "a {len}-byte prefix"
            );
        }
        let mut long = bytes.to_vec();
        long.push(0);
        assert_eq!(
            RedoRecord::decode(&long),
            Err(RedoError::BadLength(REDO_RECORD_LEN + 1))
        );
    }

    #[test]
    fn unknown_operations_and_nonzero_reserved_bytes_are_rejected() {
        let good = set(1).encode().unwrap();
        for op in [0u8, 3, 255] {
            let mut bytes = good;
            bytes[16] = op;
            assert_eq!(RedoRecord::decode(&bytes), Err(RedoError::UnknownOp(op)));
        }
        for at in [17usize, 18, 19, 96, 97, 98, 99] {
            let mut bytes = good;
            bytes[at] = 1;
            assert_eq!(
                RedoRecord::decode(&bytes),
                Err(RedoError::NonzeroReserved),
                "byte {at}"
            );
        }
    }

    #[test]
    fn an_offset_above_the_index_maximum_is_rejected_both_ways() {
        let too_far = IndexEntry::MAX_OFFSET + 1;
        let mut record = set(1);
        record.op = RedoOp::Set {
            full_hash: [1; 16],
            pack_id: PackId([1; 32]),
            offset: too_far,
            record_len: 61,
        };
        assert_eq!(record.encode(), Err(RedoError::OffsetTooLarge(too_far)));
        let mut bytes = set(1).encode().unwrap();
        bytes[68..76].copy_from_slice(&too_far.to_le_bytes());
        assert_eq!(
            RedoRecord::decode(&bytes),
            Err(RedoError::OffsetTooLarge(too_far))
        );
        // The largest representable offset is accepted.
        let mut ok = set(1);
        ok.op = RedoOp::Set {
            full_hash: [1; 16],
            pack_id: PackId([1; 32]),
            offset: IndexEntry::MAX_OFFSET,
            record_len: 61,
        };
        assert_eq!(RedoRecord::decode(&ok.encode().unwrap()), Ok(ok));
    }

    #[test]
    fn a_set_needs_a_length_and_a_tombstone_carries_no_locator() {
        let mut bytes = set(1).encode().unwrap();
        bytes[76..80].fill(0);
        assert_eq!(RedoRecord::decode(&bytes), Err(RedoError::ZeroRecordLen));
        for at in [20usize, 36, 68, 76] {
            let mut bytes = tombstone(1).encode().unwrap();
            bytes[at] = 1;
            assert_eq!(
                RedoRecord::decode(&bytes),
                Err(RedoError::TombstoneCarriesLocator),
                "byte {at}"
            );
        }
    }

    /// Flipping any single bit either fails to decode or decodes to a different
    /// record: no corruption of the payload reads back as the original.
    #[test]
    fn no_single_bit_flip_reads_back_as_the_original() {
        for record in [set(9), tombstone(10)] {
            let good = record.encode().unwrap();
            for bit in 0..REDO_RECORD_LEN * 8 {
                let mut bytes = good;
                bytes[bit / 8] ^= 1 << (bit % 8);
                assert_ne!(
                    RedoRecord::decode(&bytes),
                    Ok(record),
                    "bit {bit} flipped without changing the record"
                );
            }
        }
    }

    #[test]
    fn delta_seq_must_strictly_increase_from_above_the_base() {
        assert_eq!(validate_sequence(&[], 5), Ok(()));
        assert_eq!(
            validate_sequence(&[set(6), tombstone(7), set(9)], 5),
            Ok(())
        );
        assert_eq!(
            validate_sequence(&[set(5)], 5),
            Err(RedoError::NonMonotonic { after: 5, got: 5 }),
            "the first record must exceed the checkpoint's base sequence"
        );
        assert_eq!(
            validate_sequence(&[set(6), set(6)], 5),
            Err(RedoError::NonMonotonic { after: 6, got: 6 }),
            "a repeat is rejected"
        );
        assert_eq!(
            validate_sequence(&[set(6), set(8), tombstone(7)], 5),
            Err(RedoError::NonMonotonic { after: 8, got: 7 }),
            "a regression is rejected, tombstones included"
        );
    }
}
