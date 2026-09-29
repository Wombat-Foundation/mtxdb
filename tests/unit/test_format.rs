#![allow(clippy::tests_outside_test_module)]

use mtxdb::index::format::*;
use mtxdb::packfile::{PackId, PACK_ID_LEN};

#[test]
fn fixed_width_records_round_trip() {
    let delta = DeltaFrame {
        collection_id: [1; 16],
        bucket: 2,
        generation: 3,
        slot: 4,
    };
    assert_eq!(DeltaFrame::decode(&delta.encode()), Some(delta));

    let header = CheckpointHeader {
        magic: *b"MTXI0001",
        version: 4,
        collection_count: 2,
        directory_bytes: 112,
        slots_bytes: 128,
        pack_fingerprint: 9,
        content_crc32: 0xDEAD_BEEF,
        homes_bytes: 0,
        tails_bytes: 0,
        covered_lsn: 11,
        pack_table_count: 2,
        pack_table_bytes: 24,
        base_delta_seq: 0x0123_4567_89AB,
    };
    assert_eq!(CheckpointHeader::decode(&header.encode()), Some(header));

    let entry = CollectionDirEntry {
        collection_id: [5; 16],
        generation: 6,
        slots_offset: 7,
        homes_offset: 100,
        tails_offset: 200,
        capacity: 8,
        slot_count: 9,
    };
    assert_eq!(CollectionDirEntry::decode(&entry.encode()), Some(entry));

    let pack_entry = PackTableEntry {
        slot: 3,
        pack_id: PackId([0xAB; PACK_ID_LEN]),
    };
    assert_eq!(
        PackTableEntry::decode(&pack_entry.encode()),
        Some(pack_entry)
    );
}
