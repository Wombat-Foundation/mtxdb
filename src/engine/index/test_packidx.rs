use super::*;

fn sample_inputs() -> Vec<PackIndexInput> {
    vec![
        PackIndexInput {
            collection_id: [2; 16],
            logical_id: [0x80; 16],
            offset: 40,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x10; 16],
            offset: 20,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x10; 16],
            offset: 80,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x20; 16],
            offset: 60,
        },
    ]
}

fn sample_index() -> PackIndex {
    PackIndex::build(PackId([0x42; 16]), 19, 128, &sample_inputs()).unwrap()
}

#[test]
fn sorted_pack_index_roundtrips_and_keeps_newest_duplicate() {
    let built = sample_index();
    let decoded = PackIndex::decode(built.as_bytes()).unwrap();

    assert_eq!(decoded.pack_id(), PackId([0x42; 16]));
    assert_eq!(decoded.creation_seq(), 19);
    assert_eq!(decoded.pack_file_len(), 128);
    assert_eq!(decoded.entry_count(), 3);
    assert_eq!(decoded.get(&[1; 16], &[0x10; 16]), Some(80));
    assert_eq!(decoded.get(&[1; 16], &[0x20; 16]), Some(60));
    assert_eq!(decoded.get(&[2; 16], &[0x80; 16]), Some(40));
    assert_eq!(decoded.get(&[3; 16], &[0x10; 16]), None);
    assert!(decoded.matches_pack(PackId([0x42; 16]), 19, 128));
    assert!(!decoded.matches_pack(PackId([0x42; 16]), 18, 128));
    assert!(!decoded.matches_pack(PackId([0x42; 16]), 19, 127));
}

#[test]
fn empty_pack_index_is_valid() {
    let built = PackIndex::build(PackId([0x24; 16]), 0, 0, &[]).unwrap();
    let decoded = PackIndex::decode(built.as_bytes()).unwrap();
    assert_eq!(decoded.entry_count(), 0);
    assert_eq!(decoded.get(&[1; 16], &[2; 16]), None);
}

#[test]
fn pack_index_rejects_bad_crc_truncation_and_trailing_bytes() {
    let built = sample_index();
    let bytes = built.as_bytes();

    let mut corrupt = bytes.to_vec();
    let body_byte = PACK_INDEX_HEADER_LEN + PACK_INDEX_COLLECTION_ENTRY_LEN;
    corrupt[body_byte] ^= 1;
    assert!(PackIndex::decode(&corrupt).is_none());
    assert!(PackIndex::decode(&bytes[..bytes.len() - 1]).is_none());

    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(PackIndex::decode(&trailing).is_none());
}

#[test]
fn pack_index_rejects_noncanonical_fanout_even_with_valid_crc() {
    let built = sample_index();
    let mut bytes = built.as_bytes().to_vec();
    let first_section = PACK_INDEX_HEADER_LEN + PACK_INDEX_COLLECTION_ENTRY_LEN * 2;
    bytes[first_section..first_section + 4].copy_from_slice(&3_u32.to_le_bytes());
    let crc = crc_with_zeroed_field(&bytes).unwrap();
    bytes[CRC_RANGE].copy_from_slice(&crc.to_le_bytes());
    assert!(PackIndex::decode(&bytes).is_none());
}

#[test]
fn pack_index_rejects_offsets_outside_pack_extent() {
    let input = [PackIndexInput {
        collection_id: [1; 16],
        logical_id: [2; 16],
        offset: 12,
    }];
    assert_eq!(
        PackIndex::build(PackId([0x42; 16]), 0, 12, &input).unwrap_err(),
        PackIndexError::OffsetOutsidePack {
            offset: 12,
            pack_len: 12,
        }
    );
}

#[test]
fn pack_index_writer_publishes_a_decodable_file() {
    let path = std::env::temp_dir().join(format!(
        "mtxdb-packidx-test-{}-{}.idx",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let index = sample_index();
    index.write_atomic(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let decoded = PackIndex::decode(&bytes).unwrap();
    assert_eq!(decoded.entry_count(), 3);
    fs::remove_file(path).unwrap();
}
