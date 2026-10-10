#![allow(clippy::tests_outside_test_module)]

use bytes::Bytes;
use mtxdb::packfile::layout::*;
use mtxdb::packfile::{write_header_with_creation_seq, write_record, PackId, Record, PACK_ID_LEN};
use std::fs::File;
use std::io;

fn pack_id_for(n: u64) -> PackId {
    let mut bytes = [0u8; PACK_ID_LEN];
    bytes[..8].copy_from_slice(&n.to_be_bytes());
    // Explicit nonzero marker in the trailing byte so `n == 0` is still a
    // valid identity (the all-zero address is reserved); the filename
    // prefix is unchanged.
    bytes[PACK_ID_LEN - 1] = 0xA5;
    PackId(bytes)
}

#[test]
fn physical_layout_counts_cross_pack_spread_and_interleaved_runs() {
    let dir = std::env::temp_dir().join(format!(
        "mtxdb_layout_{}_{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("test")
            .replace(':', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let collection_a = [0xA1; 16];
    let collection_b = [0xB2; 16];
    for (pack_id, records) in [
        (
            pack_id_for(0),
            vec![collection_a, collection_b, collection_a],
        ),
        (pack_id_for(1), vec![collection_a]),
    ] {
        let path = dir.join(pack_id.filename());
        let mut file = File::create(path).unwrap();
        write_header_with_creation_seq(&mut file, &pack_id, 1).unwrap();
        for (index, collection_id) in records.into_iter().enumerate() {
            write_record(
                &mut file,
                &Record {
                    collection_id,
                    hash: [u8::try_from(index).expect("fixture index fits in u8"); 16],
                    data: Bytes::from_static(b"payload"),
                    metadata: None,
                },
            )
            .unwrap();
        }
    }

    let layout = physical_layout(&dir).unwrap();
    let a = &layout.collections[&collection_a];
    assert_eq!(a.pack_bytes.len(), 2, "A spans two packs");
    assert_eq!(a.segments, 3, "A-B-A in pack 0 plus A in pack 1");
    assert_eq!(layout.packs[&pack_id_for(0)].segments, 3);
    assert_eq!(layout.packs[&pack_id_for(1)].segments, 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn avoidable_spread_excludes_a_collections_required_spill() {
    let capacity = mtxdb::shard::MAX_SHARD_BYTES;
    let mut ideal = CollectionPhysicalLayout {
        disk_bytes: capacity.saturating_add(100),
        ..CollectionPhysicalLayout::default()
    };
    ideal.pack_bytes.insert(pack_id_for(0), capacity);
    ideal.pack_bytes.insert(pack_id_for(1), 100);
    assert_eq!(avoidable_spread_bytes(Some(&ideal)), 0);

    let mut fragmented = ideal;
    fragmented.pack_bytes.clear();
    fragmented.pack_bytes.insert(pack_id_for(0), capacity / 2);
    fragmented
        .pack_bytes
        .insert(pack_id_for(1), capacity / 2 + 100);
    assert_eq!(
        avoidable_spread_bytes(Some(&fragmented)),
        capacity.saturating_sub(capacity / 2 + 100)
    );
}

#[test]
fn physical_layout_ignores_unrelated_pack_files_but_rejects_bad_pool_packs() {
    let dir = std::env::temp_dir().join(format!(
        "mtxdb_layout_foreign_{}_{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("test")
            .replace(':', "_")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("other-app.pack"), b"not an mdb pack").unwrap();
    assert!(physical_layout(&dir).unwrap().packs.is_empty());

    std::fs::write(dir.join("pack_0000000000000000.pack"), b"not an mdb pack").unwrap();
    assert_eq!(
        physical_layout(&dir).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    std::fs::remove_dir_all(dir).unwrap();
}
