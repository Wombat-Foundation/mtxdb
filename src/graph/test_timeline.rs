use super::*;
use crate::database::Database;
use crate::layout::ShardType;
use crate::storage::{InMemoryStorage, StorageEngine};
use std::path::PathBuf;

fn test_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-timeline-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn entry(room_id: [u8; 16], order: i64, seed: u8) -> TimelineEntry {
    let mut node_id = [0u8; 16];
    node_id[0] = seed;
    node_id[15] = 0x11;
    let mut event_ref = [0u8; 32];
    event_ref[0] = seed;
    TimelineEntry::new(room_id, order, i64::from(seed), node_id, event_ref)
}

fn span(
    min_entry: TimelineEntry,
    max_entry: TimelineEntry,
    entry_count: u64,
) -> SpanDescriptor<TimelineEntry> {
    SpanDescriptor {
        min_entry,
        max_entry,
        entry_count,
        child_id: [0; 16],
    }
}

fn sorted_reference(entries: &[TimelineEntry], room_id: &[u8; 16]) -> Vec<TimelineEntry> {
    let mut reference: Vec<TimelineEntry> = entries
        .iter()
        .filter(|entry| entry.scope == *room_id)
        .copied()
        .collect();
    reference.sort_unstable();
    reference.dedup();
    reference
}

fn collect<S: StorageEngine + ?Sized>(
    index: &TimelineIndex<'_, S>,
    room_id: &[u8; 16],
    forward: bool,
    limit: usize,
) -> Vec<TimelineEntry> {
    let mut all = Vec::new();
    let mut cursor: Option<TimelineCursor> = None;
    loop {
        let page = index
            .page(room_id, cursor.as_ref(), forward, limit)
            .unwrap();
        if page.entries.is_empty() {
            break;
        }
        all.extend(page.entries.iter().copied());
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    all
}

#[test]
fn cursor_encoding_roundtrips() {
    let entry = entry([1u8; 16], 7, 3);
    let cursor = TimelineCursor::new([0xAB; 16], entry, DomainTag::new([0xCD; 16]))
        .expect("valid timeline cursor geometry");
    let bytes = cursor.encode();
    assert_eq!(bytes.len(), CURSOR_LEN);
    assert_eq!(TimelineCursor::decode(&bytes), Some(cursor));
    assert_eq!(TimelineCursor::decode(&bytes[..bytes.len() - 1]), None);
    let mut corrupt = bytes;
    corrupt[0] ^= 0xff;
    assert_eq!(TimelineCursor::decode(&corrupt), None);
}

#[test]
fn ordered_index_writes_and_pages_through_height_four() {
    let root = test_root("ordered_height_four_writes");
    let db = Database::open(root.clone()).unwrap();
    let engine = db.event_dag();
    let index = OrderedIndex::<_, TinyKey>::open(engine.as_ref(), "tiny_height_four");

    let empty_tx = db.begin_transaction();
    index
        .stage_append(&empty_tx, ShardType::EventDag, &[])
        .unwrap();
    empty_tx.commit().unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 0);

    index.build(&[TinyKey(1_000)]).unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 1);

    let stage_entries: Vec<_> = (1_001..1_017).map(TinyKey).collect();
    let stage_tx = db.begin_transaction();
    index
        .stage_append(&stage_tx, ShardType::EventDag, &stage_entries)
        .unwrap();
    stage_tx.commit().unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 2);

    let level_three_entries: Vec<_> = (1_017..1_065).map(TinyKey).collect();
    index.append_untransactional(&level_three_entries).unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 3);

    let level_four_entries: Vec<_> = (1_065..1_257).map(TinyKey).collect();
    index.append_untransactional(&level_four_entries).unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 4);

    index.append_untransactional(&[TinyKey(1_257)]).unwrap();
    let front_tx = db.begin_transaction();
    index
        .stage_append(&front_tx, ShardType::EventDag, &[TinyKey(999)])
        .unwrap();
    front_tx.commit().unwrap();
    assert_eq!(ordered_height(&index, engine.as_ref()), 4);

    let page = index.page(&(), None, true, 300).unwrap();
    assert_eq!(page.entries.len(), 259);
    assert_eq!(page.entries.first().copied(), Some(TinyKey(999)));
    assert_eq!(page.entries.last().copied(), Some(TinyKey(1_257)));

    drop(index);
    drop(db);
    std::fs::remove_dir_all(root).unwrap();
}

fn ordered_height<S: StorageEngine + ?Sized, K: OrderedKey>(
    index: &OrderedIndex<'_, S, K>,
    engine: &S,
) -> u8 {
    let (root_id, _) = index.head().unwrap().unwrap();
    let root_record = engine
        .get(&index.collection_id(), &root_id)
        .unwrap()
        .unwrap();
    decode_root::<K>(&root_record.bytes, index.domain())
        .unwrap()
        .height
}

#[test]
fn ordered_cursor_rejects_another_key_domain() {
    let engine = InMemoryStorage::new();
    let first = OrderedIndex::<_, TinyKey>::open(&engine, "tiny_cursor_first");
    let second = OrderedIndex::<_, TinyKey>::open(&engine, "tiny_cursor_second");
    let root_id = first.build(&[TinyKey(4)]).unwrap();
    let cursor = first.cursor(root_id, TinyKey(4)).unwrap();
    let error = second.page(&(), Some(&cursor), true, 1).unwrap_err();
    assert!(matches!(error, StorageError::Corrupt(_)));
}

#[test]
fn with_collection_id_uses_the_supplied_id() {
    let engine = InMemoryStorage::new();
    let name = "test";
    let supplied = [0xAB; 16];
    assert_ne!(supplied, timeline_collection_id(name));
    let index = TimelineIndex::with_collection_id(&engine, name, supplied);
    assert_eq!(index.collection_id(), supplied);
}

#[test]
fn forward_pages_cover_a_room_in_order() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, String::from("test"));
    let mut entries = Vec::new();
    for seed in 0..40u8 {
        entries.push(entry([1u8; 16], i64::from(seed / 2), seed));
        entries.push(entry([2u8; 16], i64::from(seed), seed + 100));
    }
    index.build(&entries).unwrap();

    let forward = collect(&index, &[1u8; 16], true, 3);
    assert_eq!(forward, sorted_reference(&entries, &[1u8; 16]));
    assert_eq!(forward.len(), 40);
}

#[test]
fn reverse_pages_are_forward_reversed() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
    let mut entries = Vec::new();
    for seed in 0..37u8 {
        entries.push(entry([1u8; 16], i64::from(seed), seed));
    }
    index.build(&entries).unwrap();

    let forward = collect(&index, &[1u8; 16], true, 4);
    let mut reverse = collect(&index, &[1u8; 16], false, 4);
    reverse.reverse();
    assert_eq!(forward, reverse);
    assert_eq!(reverse, sorted_reference(&entries, &[1u8; 16]));
}

#[test]
fn cursor_pins_its_snapshot_across_a_rebuild() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
    let first: Vec<TimelineEntry> = (0..10u8)
        .map(|seed| entry([1u8; 16], i64::from(seed), seed))
        .collect();
    index.build(&first).unwrap();

    let page = index.page(&[1u8; 16], None, true, 2).unwrap();
    assert_eq!(page.entries.len(), 2);
    let pinned = page.next_cursor.expect("cursor pins the first snapshot");

    // Rebuild with more entries; the head now names a different root.
    let mut later = first.clone();
    later.extend((10..20u8).map(|seed| entry([1u8; 16], i64::from(seed), seed)));
    index.build(&later).unwrap();

    // Continuing the old cursor still reads the original snapshot only.
    let mut rest = Vec::new();
    let mut cursor = Some(pinned);
    while let Some(current) = cursor {
        let page = index.page(&[1u8; 16], Some(&current), true, 3).unwrap();
        rest.extend(page.entries.iter().copied());
        cursor = page.next_cursor;
    }
    assert_eq!(rest.len(), 8);
    assert!(
        rest.iter().all(|entry| entry.primary < 10),
        "old snapshot only"
    );
}

#[test]
fn cursor_rejects_a_different_room() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
    let entries = vec![entry([1u8; 16], 0, 1), entry([2u8; 16], 0, 2)];
    index.build(&entries).unwrap();

    let other = index.page(&[2u8; 16], None, true, 1).unwrap();
    let cursor = other.next_cursor.expect("one more entry remains in room 2");
    let error = index.page(&[1u8; 16], Some(&cursor), true, 1).unwrap_err();
    assert!(matches!(error, StorageError::Corrupt(_)));
}

#[test]
fn empty_index_pages_nothing() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
    index.build(&[]).unwrap();
    let page = index.page(&[1u8; 16], None, true, 5).unwrap();
    assert!(
        page.entries.is_empty(),
        "an empty index must page no entries"
    );
    assert!(page.next_cursor.is_none());
    assert!(index.head().unwrap().is_some());
}

#[test]
fn append_inserts_into_existing_ranges_and_ignores_duplicates() {
    let room = [1u8; 16];
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "append_ranges");
    let original: Vec<_> = (0..300u16)
        .map(|n| full(room, i64::from(n * 2), i64::from(n), (n % 251) as u8))
        .collect();
    index.build(&original).unwrap();

    let before = index.page(&room, None, true, 3).unwrap();
    let old_cursor = before.next_cursor.unwrap();
    let inserted = [
        full(room, -1, -1, 240),
        full(room, 199, 1000, 241),
        full(room, 401, 1001, 242),
        original[100], // already present
    ];
    index.append_untransactional(&inserted).unwrap();

    let mut expected = original;
    expected.extend(inserted);
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(collect(&index, &room, true, 17), expected);

    // The cursor continues to read the root that was current before append.
    let old_page = index.page(&room, Some(&old_cursor), true, 300).unwrap();
    assert!(old_page
        .entries
        .iter()
        .all(|entry| !inserted[..3].contains(entry)));
}

#[test]
fn append_to_empty_index_and_empty_append_are_well_defined() {
    let room = [3u8; 16];
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "append_empty");
    let entries = [full(room, 2, 4, 1), full(room, 1, 3, 2)];

    index.append_untransactional(&entries).unwrap();
    let mut expected = entries.to_vec();
    expected.sort_unstable();
    assert_eq!(collect(&index, &room, true, 10), expected);
    let root_id = index.head().unwrap().unwrap().0;
    assert_eq!(index.append_untransactional(&[]).unwrap(), root_id);
}

#[test]
fn append_routes_gap_entries_to_the_room_or_nonfull_leaf() {
    let room_a = [1; 16];
    let room_b = [2; 16];
    let room_a_max = full(room_a, i64::MAX, i64::MAX, 1);
    let room_b_min = full(room_b, 0, 0, 2);
    let room_b_backfill = full(room_b, -1, -1, 3);
    let cross_room = [
        span(full(room_a, i64::MIN, i64::MIN, 4), room_a_max, 128),
        span(room_b_min, full(room_b, 10, 10, 5), 4),
    ];
    assert_eq!(target_span_index(&cross_room, 1, &room_b_backfill), 1);

    let room = [3; 16];
    let left_max = full(room, 20, 20, 6);
    let right_min = full(room, 40, 40, 7);
    let same_room_gap = [
        span(full(room, 0, 0, 8), left_max, TIMELINE_LEAF_CAP as u64),
        span(right_min, full(room, 60, 60, 9), 1),
    ];
    let gap_entry = full(room, 30, 30, 10);
    assert_eq!(target_span_index(&same_room_gap, 0, &gap_entry), 1);
}

#[test]
fn backfill_into_next_room_routes_across_a_directory_gap() {
    let room_a = [10; 16];
    let room_b = [11; 16];
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "append_room_gap");
    let event_count = TIMELINE_LEAF_CAP
        .checked_mul(TIMELINE_DIRECTORY_FANOUT)
        .expect("test event count fits usize");
    let mut entries: Vec<_> = (0..event_count)
        .map(|position| {
            let position = i64::try_from(position).expect("test position fits i64");
            full(room_a, position, position, 1)
        })
        .collect();
    let room_b_entry = full(room_b, 0, 0, 1);
    entries.push(room_b_entry);
    index.build(&entries).unwrap();

    let backfills = [
        full(room_b, -1, -1, 2),
        full(room_b, -2, -2, 3),
        full(room_b, -3, -3, 4),
    ];
    for backfill in backfills {
        index.append_untransactional(&[backfill]).unwrap();
    }

    let mut expected = vec![room_b_entry];
    expected.extend(backfills);
    expected.sort_unstable();
    assert_eq!(collect(&index, &room_b, true, 10), expected);
    assert_eq!(
        index.page(&room_a, None, true, 100).unwrap().entries,
        entries[..100]
    );
}

#[test]
fn empty_stage_append_on_a_new_index_publishes_an_empty_root() {
    let db_path = test_root("stage_empty_new_index");
    let db = Database::open(db_path.clone()).unwrap();
    let index = TimelineIndex::open(db.event_dag().as_ref(), "stage_empty_new_index");
    let transaction = db.begin_transaction();
    let root_id = index
        .stage_append(&transaction, ShardType::EventDag, &[])
        .unwrap();
    transaction.commit().unwrap();

    assert_eq!(index.head().unwrap().unwrap().0, root_id);
    assert_eq!(
        index.page(&[1; 16], None, true, 10).unwrap().entries.len(),
        0
    );

    drop(db);
    let _ = std::fs::remove_dir_all(db_path);
}

#[test]
fn height_three_root_pages_through_two_directory_levels() {
    let room = [12; 16];
    let entry = full(room, 1, 1, 1);
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "height_three_page");
    index.ensure_metadata().unwrap();

    let leaf_payload = encode_leaf(&[entry]).unwrap();
    let leaf_id = content_id(&leaf_payload);
    index.write_record(&leaf_id, leaf_payload).unwrap();
    let leaf_span = SpanDescriptor {
        child_id: leaf_id,
        ..span(entry, entry, 1)
    };
    let level_one_payload = encode_directory(&[leaf_span], 1).unwrap();
    let level_one_id = content_id(&level_one_payload);
    index
        .write_record(&level_one_id, level_one_payload)
        .unwrap();
    let level_one_span = SpanDescriptor {
        child_id: level_one_id,
        ..leaf_span
    };
    let level_two_payload = encode_directory(&[level_one_span], 2).unwrap();
    let level_two_id = content_id(&level_two_payload);
    index
        .write_record(&level_two_id, level_two_payload)
        .unwrap();
    let root_span = SpanDescriptor {
        child_id: level_two_id,
        ..leaf_span
    };
    let root_payload = encode_root(&[root_span], 1, 3, index.domain()).unwrap();
    let root_id = content_id(&root_payload);
    index.write_record(&root_id, root_payload).unwrap();
    index
        .write_record(
            &TIMELINE_HEAD_ID,
            encode_head::<ScopedSeqKey>(&root_id, 1, index.domain()).unwrap(),
        )
        .unwrap();

    assert_eq!(index.page(&room, None, true, 10).unwrap().entries, [entry]);
    assert_eq!(index.page(&room, None, false, 10).unwrap().entries, [entry]);
}

#[test]
fn transactional_append_uses_head_cas_and_can_be_retried() {
    let db_path = test_root("append_cas");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [4u8; 16];
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_cas");
    index.build(&[full(room_id, 1, 1, 1)]).unwrap();

    let first_entry = full(room_id, 2, 2, 2);
    let second_entry = full(room_id, 3, 3, 3);
    let first = db.begin_transaction();
    let second = db.begin_transaction();
    index
        .stage_append(&first, ShardType::EventDag, &[first_entry])
        .unwrap();
    index
        .stage_append(&second, ShardType::EventDag, &[second_entry])
        .unwrap();

    first.commit().unwrap();
    assert!(matches!(
        second.commit(),
        Err(crate::storage::StorageError::StaleRead { .. })
    ));
    assert_eq!(
        collect(&index, &room_id, true, 10),
        vec![full(room_id, 1, 1, 1), first_entry]
    );

    let retry = db.begin_transaction();
    index
        .stage_append(&retry, ShardType::EventDag, &[second_entry])
        .unwrap();
    retry.commit().unwrap();
    assert_eq!(
        collect(&index, &room_id, true, 10),
        vec![full(room_id, 1, 1, 1), first_entry, second_entry]
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);
}

#[test]
fn all_duplicates_append_does_not_bump_head_version() {
    let db_path = test_root("append_dupes");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [7u8; 16];
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_dupes");
    let entry = full(room_id, 1, 1, 1);
    index.build(&[entry]).unwrap();

    let (_, before) = db
        .event_dag()
        .get_with_record_versions(&index.collection_id(), &[TIMELINE_HEAD_ID])
        .unwrap();

    // Stage an append whose entries are all already present.
    let txn = db.begin_transaction();
    index
        .stage_append(&txn, ShardType::EventDag, &[entry, entry])
        .unwrap();
    txn.commit().unwrap();

    let (_, after) = db
        .event_dag()
        .get_with_record_versions(&index.collection_id(), &[TIMELINE_HEAD_ID])
        .unwrap();
    assert_eq!(
        before, after,
        "an all-duplicates append must not bump the head version"
    );
    assert_eq!(collect(&index, &room_id, true, 10), vec![entry]);
    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);
}

#[test]
fn concurrent_first_appends_use_absent_head_cas() {
    let db_path = test_root("append_first_cas");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [5u8; 16];
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_first_cas");
    let first_entry = full(room_id, 1, 1, 1);
    let second_entry = full(room_id, 2, 2, 2);
    let first = db.begin_transaction();
    let second = db.begin_transaction();

    index
        .stage_append(&first, ShardType::EventDag, &[first_entry])
        .unwrap();
    index
        .stage_append(&second, ShardType::EventDag, &[second_entry])
        .unwrap();
    first.commit().unwrap();
    assert!(matches!(
        second.commit(),
        Err(crate::storage::StorageError::StaleRead { .. })
    ));
    assert_eq!(collect(&index, &room_id, true, 10), vec![first_entry]);

    let retry = db.begin_transaction();
    index
        .stage_append(&retry, ShardType::EventDag, &[second_entry])
        .unwrap();
    retry.commit().unwrap();
    assert_eq!(
        collect(&index, &room_id, true, 10),
        vec![first_entry, second_entry]
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);
}

#[test]
fn empty_transactional_append_does_not_register_a_head_expectation() {
    let db_path = test_root("append_empty_transaction");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [6u8; 16];
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_empty_transaction");
    index.build(&[full(room_id, 1, 1, 1)]).unwrap();

    let empty = db.begin_transaction();
    let writer = db.begin_transaction();
    assert_eq!(
        index
            .stage_append(&empty, ShardType::EventDag, &[])
            .unwrap(),
        index.head().unwrap().unwrap().0
    );
    index
        .stage_append(&writer, ShardType::EventDag, &[full(room_id, 2, 2, 2)])
        .unwrap();
    writer.commit().unwrap();
    empty.commit().unwrap();

    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);
}

#[test]
fn transactional_append_rejects_a_different_database_pool() {
    let first_path = test_root("append_wrong_pool_index");
    let second_path = test_root("append_wrong_pool_transaction");
    let first_db = Database::open(first_path.clone()).unwrap();
    let second_db = Database::open(second_path.clone()).unwrap();
    let index = TimelineIndex::open(first_db.event_dag().as_ref(), "append_wrong_pool");
    let transaction = second_db.begin_transaction();

    assert!(matches!(
        index.stage_append(&transaction, ShardType::EventDag, &[]),
        Err(crate::storage::StorageError::Internal(message))
            if message.contains("different storage pools")
    ));

    drop(transaction);
    drop(second_db);
    drop(first_db);
    let _ = std::fs::remove_dir_all(first_path);
    let _ = std::fs::remove_dir_all(second_path);
}

#[test]
fn transactional_append_of_duplicates_does_not_rewrite_the_head() {
    let db_path = test_root("append_duplicate_transaction");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [7u8; 16];
    let existing = full(room_id, 1, 1, 1);
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_duplicate_transaction");
    let old_root = index.build(&[existing]).unwrap();

    let observer = db.begin_transaction();
    let (_, versions) = observer
        .get_with_record_versions(
            ShardType::EventDag,
            &index.collection_id(),
            &[TIMELINE_HEAD_ID],
        )
        .unwrap();
    observer
        .expect_record_version(
            ShardType::EventDag,
            index.collection_id(),
            TIMELINE_HEAD_ID,
            versions[0],
        )
        .unwrap();

    let duplicate = db.begin_transaction();
    assert_eq!(
        index
            .stage_append(&duplicate, ShardType::EventDag, &[existing])
            .unwrap(),
        old_root
    );
    duplicate.commit().unwrap();
    observer.commit().unwrap();
    assert_eq!(index.head().unwrap().unwrap().0, old_root);

    drop(db);
    let _ = std::fs::remove_dir_all(db_path);
}

#[test]
fn duplicate_only_append_survives_a_head_moved_before_its_commit() {
    let db_path = test_root("append_duplicate_then_head_moves");
    let db = Database::open(db_path.clone()).unwrap();
    let room_id = [8u8; 16];
    let existing = full(room_id, 1, 1, 1);
    let moved = full(room_id, 2, 2, 2);
    let index = TimelineIndex::open(db.event_dag().as_ref(), "append_duplicate_then_head_moves");
    index.build(&[existing]).unwrap();

    let duplicate = db.begin_transaction();
    index
        .stage_append(&duplicate, ShardType::EventDag, &[existing])
        .unwrap();

    let writer = db.begin_transaction();
    index
        .stage_append(&writer, ShardType::EventDag, &[moved])
        .unwrap();
    writer.commit().unwrap();

    duplicate.commit().unwrap();
    assert_eq!(collect(&index, &room_id, true, 10), vec![existing, moved]);

    drop(db);
    let _ = std::fs::remove_dir_all(&db_path);
}

fn full(room: [u8; 16], topological: i64, stream: i64, seed: u8) -> TimelineEntry {
    let mut node_id = [0u8; 16];
    node_id[0] = seed;
    TimelineEntry::new(room, topological, stream, node_id, [seed; 32])
}

#[test]
fn order_is_topological_then_stream_then_node() {
    let room = [1u8; 16];
    // Shuffled on purpose; includes backfilled (negative) stream positions and
    // equal positions that only `node_id` can separate.
    let entries = vec![
        full(room, 5, 10, 1),
        full(room, 2, 99, 2),
        full(room, 5, -4, 3),
        full(room, 5, 10, 0),
        full(room, -1, 7, 4),
        full(room, 2, i64::MIN, 5),
    ];
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
    index.build(&entries).unwrap();

    let keys: Vec<(i64, i64, u8)> = collect(&index, &room, true, 2)
        .iter()
        .map(|e| (e.primary, e.secondary, e.id[0]))
        .collect();
    assert_eq!(
        keys,
        vec![
            (-1, 7, 4),
            (2, i64::MIN, 5),
            (2, 99, 2),
            (5, -4, 3),
            (5, 10, 0),
            (5, 10, 1),
        ]
    );
    let mut backward = collect(&index, &room, false, 2);
    backward.reverse();
    assert_eq!(backward, collect(&index, &room, true, 5));
}

#[test]
fn wire_encoding_is_byte_comparable_and_roundtrips() {
    let room = [1u8; 16];
    let values = [i64::MIN, -5, -1, 0, 1, 7, i64::MAX];
    for pair in values.windows(2) {
        assert!(encode_position(pair[0]) < encode_position(pair[1]));
    }
    for value in values {
        assert_eq!(decode_position(encode_position(value)), value);
    }
    let cursor = TimelineCursor::new(
        [9u8; 16],
        full(room, -3, i64::MIN, 6),
        DomainTag::new([0xD0; 16]),
    )
    .unwrap();
    assert_eq!(TimelineCursor::decode(&cursor.encode()), Some(cursor));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TinyKey(u64);

impl OrderedKey for TinyKey {
    type Prefix = ();

    const WIRE_LEN: usize = 8;
    const SCHEMA_TAG: [u8; 4] = *b"TINY";
    const LEAF_CAP: usize = 4;
    const DIR_FANOUT: usize = 4;

    fn prefix(&self) {}

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_be_bytes());
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let value = u64::from_be_bytes(
            bytes
                .try_into()
                .map_err(|_| StorageError::Corrupt("tiny key length".to_owned()))?,
        );
        Ok(Self(value))
    }
}

#[test]
fn tiny_geometry_grows_and_pages_through_multiple_heights() {
    let db_path = test_root("tiny_geometry");
    let db = Database::open(db_path.clone()).unwrap();
    let index = OrderedIndex::<_, TinyKey>::open(db.event_dag().as_ref(), "tiny_geometry");
    let entries: Vec<TinyKey> = (1_000..2_100).map(TinyKey).collect();
    index.build(&entries).unwrap();

    let root_id = index.head().unwrap().unwrap().0;
    let root_bytes = db
        .event_dag()
        .get(&index.collection_id(), &root_id)
        .unwrap()
        .unwrap();
    let root = decode_root::<TinyKey>(&root_bytes.bytes, index.domain()).unwrap();
    assert!(
        root.height >= 4,
        "tiny geometry should reach height 4+, got {}",
        root.height
    );

    index
        .append(&db, ShardType::EventDag, &[TinyKey(999)])
        .unwrap();
    let transaction = db.begin_transaction();
    index
        .stage_append(
            &transaction,
            ShardType::EventDag,
            &[TinyKey(998), TinyKey(997)],
        )
        .unwrap();
    transaction.commit().unwrap();

    let first = index.page(&(), None, true, 3).unwrap();
    assert_eq!(first.entries, [TinyKey(997), TinyKey(998), TinyKey(999)]);
    let last = index.page(&(), None, false, 2).unwrap();
    assert_eq!(last.entries, [TinyKey(2_099), TinyKey(2_098)]);

    let wrong_domain =
        OrderedCursor::new(root_id, TinyKey(1_000), DomainTag::new([0xAB; 16])).unwrap();
    assert!(
        index.page(&(), Some(&wrong_domain), true, 1).is_err(),
        "a cursor from another domain must be rejected"
    );

    drop(db);
    let _ = std::fs::remove_dir_all(db_path);
}

#[test]
fn tiny_tree_removal_demotes_and_prefix_purge_keeps_old_cursors_valid() {
    let db_path = test_root("tiny_remove_demote");
    let db = Database::open(db_path.clone()).unwrap();
    let index = OrderedIndex::<_, TinyKey>::open(db.event_dag().as_ref(), "tiny_remove_demote");
    let entries: Vec<_> = (1_000..2_100).map(TinyKey).collect();
    index.build(&entries).unwrap();
    let old_page = index.page(&(), None, true, 1).unwrap();
    let old_cursor = old_page.next_cursor.unwrap();
    assert!(ordered_height(&index, db.event_dag().as_ref()) >= 4);

    let removed_keys: Vec<_> = (1_000..2_098).map(TinyKey).collect();
    let transaction = db.begin_transaction();
    let (_root_after_remove, removed) = index
        .stage_remove(&transaction, ShardType::EventDag, &removed_keys)
        .unwrap();
    assert_eq!(removed, 1_098);
    transaction.commit().unwrap();

    let remaining = index.page(&(), None, true, 10).unwrap();
    assert_eq!(remaining.entries, [TinyKey(2_098), TinyKey(2_099)]);
    let old_snapshot = index.page(&(), Some(&old_cursor), true, 2).unwrap();
    assert_eq!(old_snapshot.entries, [TinyKey(1_001), TinyKey(1_002)]);

    let transaction = db.begin_transaction();
    let (_empty_root, removed) = index
        .stage_purge_prefix(&transaction, ShardType::EventDag, ())
        .unwrap();
    assert_eq!(removed, 2);
    transaction.commit().unwrap();
    let count = index.head().unwrap().unwrap().1;
    assert_eq!(count, 0);
    assert_eq!(ordered_height(&index, db.event_dag().as_ref()), 0);
    assert_eq!(index.page(&(), None, true, 10).unwrap().entries.len(), 0);

    drop(db);
    let _ = std::fs::remove_dir_all(db_path);
}

#[test]
fn removal_collapses_height_and_keeps_a_canonical_empty_root() {
    let engine = InMemoryStorage::new();
    let index = OrderedIndex::<_, TinyKey>::open(&engine, "tiny_remove_height");
    let entries: Vec<TinyKey> = (1..=300).map(TinyKey).collect();
    index.build(&entries).unwrap();
    assert_eq!(ordered_height(&index, &engine), 4);

    let transaction = index.remove_untransactional(&(5..=300).map(TinyKey).collect::<Vec<_>>());
    assert!(transaction.is_ok());
    assert_eq!(ordered_height(&index, &engine), 1);
    assert_eq!(index.head().unwrap().unwrap().1, 4);
    assert_eq!(
        index.page(&(), None, true, 10).unwrap().entries,
        [TinyKey(1), TinyKey(2), TinyKey(3), TinyKey(4)]
    );

    index
        .remove_untransactional(&[TinyKey(1), TinyKey(2), TinyKey(3), TinyKey(4)])
        .unwrap();
    assert_eq!(ordered_height(&index, &engine), 0);
    assert_eq!(index.head().unwrap().unwrap().1, 0);
    assert_eq!(index.page(&(), None, true, 10).unwrap().entries.len(), 0);
}

#[test]
fn removing_absent_keys_leaves_the_root_unchanged() {
    let engine = InMemoryStorage::new();
    let index = OrderedIndex::<_, TinyKey>::open(&engine, "tiny_remove_gaps");
    let root = index
        .build(&[TinyKey(10), TinyKey(20), TinyKey(30)])
        .unwrap();
    let page = index.page(&(), None, true, 10).unwrap();

    let unchanged = index
        .remove_untransactional(&[TinyKey(15), TinyKey(25)])
        .unwrap();
    assert_eq!(unchanged.0, root);
    assert_eq!(
        index.page(&(), None, true, 10).unwrap().entries,
        page.entries
    );
}

#[test]
fn a_cursor_pinned_before_removal_keeps_reading_its_snapshot() {
    let engine = InMemoryStorage::new();
    let index = OrderedIndex::<_, TinyKey>::open(&engine, "tiny_remove_cursor");
    let entries: Vec<TinyKey> = (1..=100).map(TinyKey).collect();
    index.build(&entries).unwrap();

    let first = index.page(&(), None, true, 5).unwrap();
    let pinned = first.next_cursor.expect("full page has a cursor");
    index
        .remove_untransactional(&(1..=50).map(TinyKey).collect::<Vec<_>>())
        .unwrap();

    let old = index.page(&(), Some(&pinned), true, 5).unwrap();
    assert_eq!(
        old.entries,
        [TinyKey(6), TinyKey(7), TinyKey(8), TinyKey(9), TinyKey(10)]
    );
    let current = index.page(&(), None, true, 5).unwrap();
    assert_eq!(
        current.entries,
        [
            TinyKey(51),
            TinyKey(52),
            TinyKey(53),
            TinyKey(54),
            TinyKey(55)
        ]
    );
}

#[test]
fn a_transactional_remove_and_mutate_publish_one_root() {
    let root = test_root("transactional_remove_mutate");
    let db = Database::open(root.clone()).unwrap();
    let index = OrderedIndex::<_, TinyKey>::open(db.event_dag().as_ref(), "tiny_mutate");

    let entries: Vec<TinyKey> = (1..=300).map(TinyKey).collect();
    index.build(&entries).unwrap();

    let remove_tx = db.begin_transaction();
    let (removed_root, _removed) = index
        .stage_remove(
            &remove_tx,
            ShardType::EventDag,
            &(2..=300).map(TinyKey).collect::<Vec<_>>(),
        )
        .unwrap();
    remove_tx.commit().unwrap();
    assert_eq!(ordered_height(&index, db.event_dag().as_ref()), 1);
    assert_eq!(index.head().unwrap().unwrap().0, removed_root);
    assert_eq!(
        index.page(&(), None, true, 10).unwrap().entries,
        [TinyKey(1)]
    );

    let mutate_tx = db.begin_transaction();
    index
        .stage_mutate(
            &mutate_tx,
            ShardType::EventDag,
            &[TinyKey(1)],
            &[TinyKey(7), TinyKey(9)],
        )
        .unwrap();
    mutate_tx.commit().unwrap();
    assert_eq!(
        index.page(&(), None, true, 10).unwrap().entries,
        [TinyKey(7), TinyKey(9)]
    );
    assert_eq!(index.head().unwrap().unwrap().1, 2);

    drop(db);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn prefix_purge_drops_one_room_without_reading_its_subtrees() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "purge_rooms");
    let room_a = [0xAA; 16];
    let room_b = [0xBB; 16];
    let mut entries = Vec::new();
    for order in 1..=200 {
        entries.push(entry(room_a, order, u8::try_from(order % 251).unwrap_or(0)));
    }
    for order in 1..=100 {
        entries.push(entry(room_b, order, u8::try_from(order % 251).unwrap_or(1)));
    }
    index.build(&entries).unwrap();

    index.purge_prefix_untransactional(room_a).unwrap();
    assert_eq!(collect(&index, &room_a, true, 17).len(), 0);
    assert_eq!(collect(&index, &room_b, true, 17).len(), 100);
    assert_eq!(index.head().unwrap().unwrap().1, 100);
    assert_eq!(ordered_height(&index, &engine), 1);

    index.purge_prefix_untransactional(room_b).unwrap();
    assert_eq!(collect(&index, &room_b, true, 17).len(), 0);
    assert_eq!(index.head().unwrap().unwrap().1, 0);
    assert_eq!(ordered_height(&index, &engine), 0);
}
