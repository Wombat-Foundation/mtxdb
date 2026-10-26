use super::*;

fn frame_offset(delta: u64) -> u64 {
    u64::try_from(HEADER_LEN)
        .unwrap()
        .checked_add(delta)
        .unwrap()
}

fn frame_offset_u32(delta: u64) -> u32 {
    u32::try_from(frame_offset(delta)).unwrap()
}

fn sample_identity() -> PackIndexIdentity {
    PackIndexIdentity {
        pack_id: PackId([0x42; 16]),
        creation_seq: 19,
        pack_file_len: frame_offset(128),
    }
}

fn sample_inputs() -> Vec<PackIndexInput> {
    vec![
        PackIndexInput {
            collection_id: [2; 16],
            logical_id: [0x80; 16],
            offset: frame_offset(40),
            removed: false,
            disk_bytes: 0,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x10; 16],
            offset: frame_offset(20),
            removed: false,
            disk_bytes: 0,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x10; 16],
            offset: frame_offset(80),
            removed: false,
            disk_bytes: 0,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x20; 16],
            offset: frame_offset(60),
            removed: false,
            disk_bytes: 0,
        },
    ]
}

fn sample_index() -> PackIndex {
    PackIndex::build(sample_identity(), sample_inputs()).unwrap()
}

#[test]
fn tombstone_wins_only_when_it_is_the_newest_version() {
    let collection = [9; 16];
    let key = [7; 16];
    let build = |versions: &[(u64, bool)]| {
        PackIndex::build(
            sample_identity(),
            versions
                .iter()
                .map(|(offset, removed)| PackIndexInput {
                    collection_id: collection,
                    logical_id: key,
                    offset: frame_offset(*offset),
                    removed: *removed,
                    disk_bytes: 0,
                })
                .collect(),
        )
        .unwrap()
    };

    let removed_last = build(&[(20, false), (40, true)]);
    assert_eq!(
        removed_last.get_locator(&collection, &key),
        Some(PackIndexLocator::Removed(
            u32::try_from(frame_offset(40)).unwrap()
        ))
    );
    assert_eq!(removed_last.get(&collection, &key), None);

    let re_added_last = build(&[(20, false), (40, true), (60, false)]);
    assert_eq!(
        re_added_last.get_locator(&collection, &key),
        Some(PackIndexLocator::Frame(
            u32::try_from(frame_offset(60)).unwrap()
        ))
    );
    assert!(PackIndex::decode(re_added_last.as_bytes()).is_some());
}

#[test]
fn sorted_pack_index_roundtrips_and_keeps_newest_duplicate() {
    let built = sample_index();
    let decoded = PackIndex::decode(built.as_bytes()).unwrap();

    assert_eq!(decoded.identity().pack_id, PackId([0x42; 16]));
    assert_eq!(decoded.identity().creation_seq, 19);
    assert_eq!(decoded.identity().pack_file_len, frame_offset(128));
    assert_eq!(decoded.entry_count(), 3);
    assert_eq!(
        decoded.get(&[1; 16], &[0x10; 16]),
        Some(frame_offset_u32(80))
    );
    assert_eq!(
        decoded.get(&[1; 16], &[0x20; 16]),
        Some(frame_offset_u32(60))
    );
    assert_eq!(
        decoded.get(&[2; 16], &[0x80; 16]),
        Some(frame_offset_u32(40))
    );
    assert_eq!(decoded.get(&[3; 16], &[0x10; 16]), None);
    assert_eq!(decoded.identity(), built.identity());
}

#[test]
fn empty_pack_index_is_valid() {
    let built = PackIndex::build(
        PackIndexIdentity {
            pack_id: PackId([0x24; 16]),
            creation_seq: 0,
            pack_file_len: frame_offset(0),
        },
        vec![],
    )
    .unwrap();
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
fn pack_index_rejects_offsets_before_frame_data() {
    let input = [PackIndexInput {
        collection_id: [1; 16],
        logical_id: [2; 16],
        offset: frame_offset(0).checked_sub(1).unwrap(),
        removed: false,
        disk_bytes: 0,
    }];
    assert_eq!(
        PackIndex::build(
            PackIndexIdentity {
                pack_id: PackId([0x42; 16]),
                creation_seq: 0,
                pack_file_len: frame_offset(1),
            },
            input.to_vec(),
        )
        .unwrap_err(),
        PackIndexError::OffsetBeforePackData {
            offset: frame_offset(0).checked_sub(1).unwrap(),
        }
    );
}

#[test]
fn pack_index_rejects_offsets_at_pack_end_during_build_and_decode() {
    let end = frame_offset(128);
    let input = PackIndexInput {
        collection_id: [1; 16],
        logical_id: [2; 16],
        offset: end,
        removed: false,
        disk_bytes: 0,
    };
    assert_eq!(
        PackIndex::build(
            PackIndexIdentity {
                pack_id: PackId([0x42; 16]),
                creation_seq: 0,
                pack_file_len: end,
            },
            vec![input],
        )
        .unwrap_err(),
        PackIndexError::OffsetOutsidePack {
            offset: end,
            pack_len: end,
        }
    );

    let mut bytes = sample_index().as_bytes().to_vec();
    let section_at = PACK_INDEX_HEADER_LEN
        .checked_add(PACK_INDEX_COLLECTION_ENTRY_LEN.checked_mul(2).unwrap())
        .unwrap();
    let count = 2_usize;
    let first_offset_at = section_at
        .checked_add(PACK_INDEX_FANOUT_LEN)
        .unwrap()
        .checked_add(count.checked_mul(16).unwrap())
        .unwrap();
    bytes[first_offset_at..first_offset_at + 4]
        .copy_from_slice(&u32::try_from(end).unwrap().to_le_bytes());
    let crc = crc_with_zeroed_field(&bytes).unwrap();
    bytes[CRC_RANGE].copy_from_slice(&crc.to_le_bytes());
    assert!(PackIndex::decode(&bytes).is_none());
}

#[test]
fn mapped_open_checks_identity_and_verify_detects_body_corruption() {
    let path = test_path();
    let index = sample_index();
    index.write_atomic(&path).unwrap();

    let mapped = PackIndex::open_mmap(&path, sample_identity()).unwrap();
    assert_eq!(
        mapped.get(&[1; 16], &[0x10; 16]),
        Some(frame_offset_u32(80))
    );
    assert_eq!(mapped.verify(), Ok(()));
    drop(mapped);

    for expected in [
        PackIndexIdentity {
            pack_id: PackId([0x43; 16]),
            ..sample_identity()
        },
        PackIndexIdentity {
            creation_seq: 20,
            ..sample_identity()
        },
        PackIndexIdentity {
            pack_file_len: frame_offset(129),
            ..sample_identity()
        },
    ] {
        assert!(matches!(
            PackIndex::open_mmap(&path, expected),
            Err(PackIndexOpenError::InvalidOrStale)
        ));
    }

    let mut corrupt = index.as_bytes().to_vec();
    let first_section = PACK_INDEX_HEADER_LEN
        .checked_add(PACK_INDEX_COLLECTION_ENTRY_LEN.checked_mul(2).unwrap())
        .unwrap();
    let ids_at = first_section.checked_add(PACK_INDEX_FANOUT_LEN).unwrap();
    corrupt[ids_at] ^= 1;
    fs::write(&path, corrupt).unwrap();
    let mapped = PackIndex::open_mmap(&path, sample_identity()).unwrap();
    assert_eq!(mapped.verify(), Err(PackIndexError::Invalid));
    drop(mapped);

    let truncated_path = test_path();
    let truncated = &index.as_bytes()[..index.as_bytes().len().checked_sub(1).unwrap()];
    fs::write(&truncated_path, truncated).unwrap();
    assert!(matches!(
        PackIndex::open_mmap(&truncated_path, sample_identity()),
        Err(PackIndexOpenError::InvalidOrStale)
    ));
    fs::remove_file(path).unwrap();
    fs::remove_file(truncated_path).unwrap();
}

#[test]
fn fanout_edges_and_equal_ids_are_isolated_by_collection() {
    let inputs = vec![
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x00; 16],
            offset: frame_offset(1),
            removed: false,
            disk_bytes: 0,
        },
        PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0xff; 16],
            offset: frame_offset(2),
            removed: false,
            disk_bytes: 0,
        },
        PackIndexInput {
            collection_id: [2; 16],
            logical_id: [0x00; 16],
            offset: frame_offset(3),
            removed: false,
            disk_bytes: 0,
        },
    ];
    let index = PackIndex::build(
        PackIndexIdentity {
            pack_file_len: frame_offset(128),
            ..sample_identity()
        },
        inputs,
    )
    .unwrap();
    assert_eq!(index.get(&[1; 16], &[0x00; 16]), Some(frame_offset_u32(1)));
    assert_eq!(index.get(&[1; 16], &[0xff; 16]), Some(frame_offset_u32(2)));
    assert_eq!(index.get(&[2; 16], &[0x00; 16]), Some(frame_offset_u32(3)));
    assert_eq!(index.get(&[2; 16], &[0xff; 16]), None);
}

#[test]
fn write_atomic_replaces_an_existing_complete_index() {
    let path = test_path();
    let first = sample_index();
    first.write_atomic(&path).unwrap();
    let replacement = PackIndex::build(
        sample_identity(),
        vec![PackIndexInput {
            collection_id: [1; 16],
            logical_id: [0x10; 16],
            offset: frame_offset(100),
            removed: false,
            disk_bytes: 0,
        }],
    )
    .unwrap();
    replacement.write_atomic(&path).unwrap();
    let decoded = PackIndex::decode(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(decoded.entry_count(), 1);
    assert_eq!(
        decoded.get(&[1; 16], &[0x10; 16]),
        Some(frame_offset_u32(100))
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn pack_index_writer_publishes_a_decodable_file() {
    let path = test_path();
    let index = sample_index();
    index.write_atomic(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let decoded = PackIndex::decode(&bytes).unwrap();
    assert_eq!(decoded.entry_count(), 3);
    fs::remove_file(path).unwrap();
}

#[test]
fn build_for_pack_round_trips_live_and_removed_records() {
    use crate::storage::{NodeData, StorageEngine as _};
    use crate::PackfileStorage;

    let dir = std::env::temp_dir().join(format!("mtxdb-packidx-pack-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let store = PackfileStorage::open(dir.clone()).unwrap();
    let collection = [0xAA; 16];
    let live = [1; 16];
    let removed = [2; 16];
    store
        .put(&collection, &live, &NodeData::from_slice(b"live"))
        .unwrap();
    store
        .put(&collection, &removed, &NodeData::from_slice(b"gone"))
        .unwrap();
    store.remove(&collection, &removed).unwrap();
    store.sync_all().unwrap();
    drop(store);

    let pack = fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "pack"))
        .expect("the store wrote a pack");
    build_and_write(&pack, true).unwrap();
    let index = open_for_pack(&pack).expect("the sealed sidecar opens");
    assert!(!index.identity().pack_id.is_zero());
    let expected_disk_bytes: u64 = crate::packfile::scan_packfile_from_with_removal(&pack, 0, true)
        .unwrap()
        .iter()
        .filter(|entry| entry.collection_id == collection)
        .map(|entry| entry.disk_bytes)
        .sum();
    assert_eq!(
        index.collection_disk_bytes(&collection),
        Some(expected_disk_bytes),
        "the sidecar preserves the pack's physical byte total"
    );
    let mut live_seen = false;
    let mut removed_seen = false;
    index.for_each_entry(|collection_id, logical_id, locator| {
        assert_eq!(collection_id, collection);
        if logical_id == live {
            live_seen = matches!(locator, PackIndexLocator::Frame(_));
        }
        if logical_id == removed {
            removed_seen = matches!(locator, PackIndexLocator::Removed(_));
        }
    });
    assert!(live_seen && removed_seen);
    let _ = fs::remove_dir_all(&dir);
}

fn test_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "mtxdb-packidx-test-{}-{}.idx",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}
