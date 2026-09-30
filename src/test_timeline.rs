use super::*;
use crate::storage::InMemoryStorage;

fn entry(room_id: [u8; 16], order: i64, seed: u8) -> TimelineEntry {
    let mut node_id = [0u8; 16];
    node_id[0] = seed;
    node_id[15] = 0x11;
    let mut event_ref = [0u8; 32];
    event_ref[0] = seed;
    TimelineEntry::new(room_id, order, i64::from(seed), node_id, event_ref)
}

fn sorted_reference(entries: &[TimelineEntry], room_id: &[u8; 16]) -> Vec<TimelineEntry> {
    let mut reference: Vec<TimelineEntry> = entries
        .iter()
        .filter(|entry| entry.room_id == *room_id)
        .copied()
        .collect();
    reference.sort_unstable();
    reference.dedup();
    reference
}

fn collect(
    index: &TimelineIndex<'_, InMemoryStorage>,
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
    let cursor = TimelineCursor::new([0xAB; 16], entry);
    let bytes = cursor.encode();
    assert_eq!(bytes.len(), CURSOR_LEN);
    assert_eq!(TimelineCursor::decode(&bytes), Some(cursor));
    assert_eq!(TimelineCursor::decode(&bytes[..bytes.len() - 1]), None);
    let mut corrupt = bytes;
    corrupt[0] ^= 0xff;
    assert_eq!(TimelineCursor::decode(&corrupt), None);
}

#[test]
fn forward_pages_cover_a_room_in_order() {
    let engine = InMemoryStorage::new();
    let index = TimelineIndex::open(&engine, "test");
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
        rest.iter().all(|entry| entry.topological_ordering < 10),
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
        .map(|e| (e.topological_ordering, e.stream_ordering, e.node_id[0]))
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
    let cursor = TimelineCursor::new([9u8; 16], full(room, -3, i64::MIN, 6));
    assert_eq!(TimelineCursor::decode(&cursor.encode()), Some(cursor));
}
