#![allow(clippy::tests_outside_test_module)]

use mtxdb::index::redo::*;
use mtxdb::index::IndexEntry;

fn set(seq: u64) -> RedoRecord {
    RedoRecord {
        collection_id: [0xC1; 16],
        op: RedoOp::Set {
            full_hash: [0xA5; 16],
            pack_id: 7,
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
    for at in [17usize, 18, 19, 44, 45, 46, 47, 76, 77, 78, 79] {
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
        pack_id: 1,
        offset: too_far,
        record_len: 61,
    };
    assert_eq!(record.encode(), Err(RedoError::OffsetTooLarge(too_far)));
    let mut bytes = set(1).encode().unwrap();
    bytes[48..56].copy_from_slice(&too_far.to_le_bytes());
    assert_eq!(
        RedoRecord::decode(&bytes),
        Err(RedoError::OffsetTooLarge(too_far))
    );
    // The largest representable offset is accepted.
    let mut ok = set(1);
    ok.op = RedoOp::Set {
        full_hash: [1; 16],
        pack_id: 1,
        offset: IndexEntry::MAX_OFFSET,
        record_len: 61,
    };
    assert_eq!(RedoRecord::decode(&ok.encode().unwrap()), Ok(ok));
}

#[test]
fn a_set_needs_a_length_and_a_tombstone_carries_no_locator() {
    let mut bytes = set(1).encode().unwrap();
    bytes[56..60].fill(0);
    assert_eq!(RedoRecord::decode(&bytes), Err(RedoError::ZeroRecordLen));
    for at in [20usize, 36, 48, 56] {
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
