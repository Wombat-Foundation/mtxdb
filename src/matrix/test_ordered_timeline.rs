use super::*;
use crate::database::Database;
use crate::storage::StorageEngine;
use bytes::Bytes;
use std::path::PathBuf;

fn test_dir(name: &str) -> PathBuf {
    let path = crate::test_scratch::root().join(format!(
        "mtxdb-ordered-event-timeline-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn event(room: [u8; 16], depth: i64, stream: i64, id: u8) -> EventTimelineEntry {
    let mut node_id = [0; 16];
    node_id[15] = id;
    EventTimelineEntry::new(room, depth, stream, node_id)
}

#[test]
fn event_key_orders_match_their_memcomparable_encodings() {
    assert_eq!(EventTopoKey::WIRE_LEN, 48);
    assert_eq!(EventStreamKey::WIRE_LEN, 48);
    let entries = [
        event([1; 16], -4, 11, 1),
        event([1; 16], -4, -2, 2),
        event([1; 16], 0, i64::MIN, 3),
        event([2; 16], i64::MAX, 1, 4),
    ];
    for pair in entries.windows(2) {
        let topo_left = pair[0].topo_key();
        let topo_right = pair[1].topo_key();
        let mut topo_left_wire = Vec::new();
        let mut topo_right_wire = Vec::new();
        topo_left.encode_into(&mut topo_left_wire);
        topo_right.encode_into(&mut topo_right_wire);
        assert_eq!(EventTopoKey::decode(&topo_left_wire).unwrap(), topo_left);
        assert_eq!(
            topo_left.cmp(&topo_right),
            topo_left_wire.cmp(&topo_right_wire)
        );

        let stream_left = pair[0].stream_key();
        let stream_right = pair[1].stream_key();
        let mut stream_left_wire = Vec::new();
        let mut stream_right_wire = Vec::new();
        stream_left.encode_into(&mut stream_left_wire);
        stream_right.encode_into(&mut stream_right_wire);
        assert_eq!(
            EventStreamKey::decode(&stream_left_wire).unwrap(),
            stream_left
        );
        assert_eq!(
            stream_left.cmp(&stream_right),
            stream_left_wire.cmp(&stream_right_wire)
        );
    }
}

#[test]
fn dual_index_batch_is_atomic_and_watermark_pages_resume_without_skips() {
    let test_path = test_dir("dual_atomic_resume");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x31; 16];
    let entries = [
        event(room, 10, 2, 1),
        event(room, 11, 3, 2),
        event(room, 12, 4, 3),
    ];
    let payload_collection = [0xA5; 16];
    let payloads: Vec<_> = entries
        .iter()
        .map(|entry| {
            (
                entry.event_node_id,
                NodeData::new(Bytes::from(vec![entry.event_node_id[15]])),
            )
        })
        .collect();

    let transaction = db.begin_transaction();
    stage_event_timeline_batch(
        &transaction,
        ShardType::EventDag,
        payload_collection,
        &payloads,
        &topo_index,
        &stream_index,
        &entries,
    )
    .unwrap();
    transaction.commit().unwrap();

    let topo_page = topo_index.page(&room, None, true, 10).unwrap();
    assert_eq!(topo_page.entries.len(), entries.len());
    assert_eq!(topo_page.entries[0].event_node_id, entries[0].event_node_id);
    let visible = page_stream_up_to(&stream_index, &room, None, true, 10, 3).unwrap();
    assert_eq!(
        visible
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    let resume = visible
        .next_cursor
        .expect("truncated page preserves visible cursor");
    assert_eq!(resume.entry().stream_ordering, 3);
    let advanced = page_stream_up_to(&stream_index, &room, Some(&resume), true, 3, 4).unwrap();
    assert_eq!(advanced.entries.len(), 1);
    assert_eq!(advanced.entries[0].stream_ordering, 4);

    let reverse = page_stream_up_to(&stream_index, &room, None, false, 10, 3).unwrap();
    assert_eq!(
        reverse
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [3, 2]
    );

    for payload in &payloads {
        assert!(pool.get(&payload_collection, &payload.0).unwrap().is_some());
    }

    let (root_id, _) = stream_index.head().unwrap().unwrap();
    let cross_room_cursor = stream_index
        .cursor(root_id, entries[2].stream_key())
        .unwrap();
    let other_room = [0x32; 16];
    assert!(page_stream_up_to(
        &stream_index,
        &other_room,
        Some(&cross_room_cursor),
        false,
        10,
        3,
    )
    .is_err());

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn empty_timeline_batch_does_not_stage_index_heads() {
    let test_path = test_dir("empty_batch");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let transaction = db.begin_transaction();
    let roots = stage_event_timeline_batch(
        &transaction,
        ShardType::EventDag,
        [0xAC; 16],
        &[],
        &topo_index,
        &stream_index,
        &[],
    )
    .unwrap();
    assert!(roots.is_none());
    transaction.commit().unwrap();
    assert!(topo_index.head().unwrap().is_none());
    assert!(stream_index.head().unwrap().is_none());
    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn a_rising_watermark_requires_rebinding_a_pinned_stream_cursor() {
    let test_path = test_dir("stream_watermark_rebind");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x51; 16];
    let payload_collection = [0xC7; 16];

    // R1 publishes streams 2 and 4 while stream 3 is still in flight, so only
    // stream 2 is contiguous.
    let first_tx = db.begin_transaction();
    stage_event_timeline_batch(
        &first_tx,
        ShardType::EventDag,
        payload_collection,
        &[],
        &topo_index,
        &stream_index,
        &[event(room, 2, 2, 1), event(room, 4, 4, 2)],
    )
    .unwrap();
    first_tx.commit().unwrap();

    let first = page_stream_up_to(&stream_index, &room, None, true, 1, 2).unwrap();
    assert_eq!(
        first
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [2]
    );
    let pinned = first.next_cursor.expect("truncated page has a cursor");

    // R2 fills stream 3, so the watermark advances to 4.
    let second_tx = db.begin_transaction();
    stage_event_timeline_batch(
        &second_tx,
        ShardType::EventDag,
        payload_collection,
        &[],
        &topo_index,
        &stream_index,
        &[event(room, 3, 3, 3)],
    )
    .unwrap();
    second_tx.commit().unwrap();

    // A cursor pinned to R1 cannot see stream 3 even though the watermark now
    // covers it; this is the hazard that rebinding removes.
    let stale = page_stream_up_to(&stream_index, &room, Some(&pinned), true, 10, 4).unwrap();
    assert_eq!(
        stale
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [4]
    );

    let rebased = rebind_stream_cursor_to_head(&stream_index, &room, &pinned)
        .unwrap()
        .expect("the index has a head");
    let advanced = page_stream_up_to(&stream_index, &room, Some(&rebased), true, 10, 4).unwrap();
    assert_eq!(
        advanced
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [3, 4]
    );

    assert!(
        rebind_stream_cursor_to_head(&stream_index, &[0x52; 16], &pinned).is_err(),
        "rebinding must reject a cursor from another room"
    );

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn a_stale_dual_index_commit_does_not_publish_its_payload() {
    let test_path = test_dir("dual_stale_commit");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x42; 16];
    let payload_collection = [0xB6; 16];
    let stale_entry = event(room, 1, 1, 1);
    let live_entry = event(room, 2, 2, 2);
    let stale_payload = [(
        stale_entry.event_node_id,
        NodeData::new(Bytes::from_static(b"stale")),
    )];
    let live_payload = [(
        live_entry.event_node_id,
        NodeData::new(Bytes::from_static(b"live")),
    )];

    let stale_tx = db.begin_transaction();
    stage_event_timeline_batch(
        &stale_tx,
        ShardType::EventDag,
        payload_collection,
        &stale_payload,
        &topo_index,
        &stream_index,
        &[stale_entry],
    )
    .unwrap();

    let live_tx = db.begin_transaction();
    stage_event_timeline_batch(
        &live_tx,
        ShardType::EventDag,
        payload_collection,
        &live_payload,
        &topo_index,
        &stream_index,
        &[live_entry],
    )
    .unwrap();
    live_tx.commit().unwrap();
    assert!(stale_tx.commit().is_err());

    assert!(pool
        .get(&payload_collection, &stale_entry.event_node_id)
        .unwrap()
        .is_none());
    let topo_page = topo_index.page(&room, None, true, 10).unwrap();
    let stream_page = stream_index.page(&room, None, true, 10).unwrap();
    assert_eq!(topo_page.entries.len(), 1);
    assert_eq!(stream_page.entries.len(), 1);
    assert_eq!(topo_page.entries[0].event_node_id, live_entry.event_node_id);
    assert_eq!(
        stream_page.entries[0].event_node_id,
        live_entry.event_node_id
    );

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn remove_and_purge_keep_both_orders_consistent() {
    let test_path = test_dir("remove_purge");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x61; 16];
    let other = [0x62; 16];
    let payload_collection = [0xD8; 16];
    let entries = [
        event(room, 1, 1, 1),
        event(room, 2, 2, 2),
        event(room, 3, 3, 3),
        event(other, 1, 1, 4),
        event(other, 2, 2, 5),
    ];

    let tx = db.begin_transaction();
    stage_event_timeline_batch(
        &tx,
        ShardType::EventDag,
        payload_collection,
        &[],
        &topo_index,
        &stream_index,
        &entries,
    )
    .unwrap();
    tx.commit().unwrap();

    let tx = db.begin_transaction();
    let removed = stage_remove_event_timeline_batch(
        &tx,
        ShardType::EventDag,
        &topo_index,
        &stream_index,
        &[entries[1]],
    )
    .unwrap()
    .expect("a non-empty removal stages both indexes");
    tx.commit().unwrap();
    assert_eq!(removed, ((removed.0 .0, 1), (removed.1 .0, 1)));
    let topo_page = topo_index.page(&room, None, true, 10).unwrap();
    assert_eq!(
        topo_page
            .entries
            .iter()
            .map(|entry| entry.stream_ordering)
            .collect::<Vec<_>>(),
        [1, 3]
    );

    let tx = db.begin_transaction();
    let (topo_removed, stream_removed) =
        stage_purge_room_timeline(&tx, ShardType::EventDag, &topo_index, &stream_index, &other)
            .unwrap();
    tx.commit().unwrap();
    assert_eq!(topo_removed.1, 2);
    assert_eq!(stream_removed.1, 2);
    assert_eq!(
        topo_index
            .page(&other, None, true, 10)
            .unwrap()
            .entries
            .len(),
        0
    );
    assert_eq!(
        stream_index
            .page(&other, None, true, 10)
            .unwrap()
            .entries
            .len(),
        0
    );
    assert_eq!(
        topo_index
            .page(&room, None, true, 10)
            .unwrap()
            .entries
            .len(),
        2
    );

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn promoting_an_outlier_stages_both_orders_without_payloads() {
    let test_path = test_dir("promote_outlier");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x71; 16];
    let entry = event(room, 4, 4, 1);

    let tx = db.begin_transaction();
    let roots = stage_promote_outlier_to_timeline(
        &tx,
        ShardType::EventDag,
        [0xE9; 16],
        &topo_index,
        &stream_index,
        &[entry],
    )
    .unwrap();
    tx.commit().unwrap();
    assert!(roots.is_some());
    assert_eq!(
        topo_index.page(&room, None, true, 10).unwrap().entries,
        [entry.topo_key()]
    );
    assert_eq!(
        stream_index.page(&room, None, true, 10).unwrap().entries,
        [entry.stream_key()]
    );

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}

#[test]
fn an_expired_cursor_falls_back_to_keyset_pagination() {
    let test_path = test_dir("expired_cursor_fallback");
    let db = Database::open(test_path.clone()).unwrap();
    let pool = db.pool(ShardType::EventDag);
    let (topo_index, stream_index) = open_event_timeline_indexes(pool.as_ref());
    let room = [0x81; 16];
    let entries = [
        event(room, 1, 1, 1),
        event(room, 2, 2, 2),
        event(room, 3, 3, 3),
    ];

    let tx = db.begin_transaction();
    stage_event_timeline_batch(
        &tx,
        ShardType::EventDag,
        [0xFA; 16],
        &[],
        &topo_index,
        &stream_index,
        &entries,
    )
    .unwrap();
    tx.commit().unwrap();

    let first = topo_index.page(&room, None, true, 1).unwrap();
    let cursor = first.next_cursor.expect("full page has a cursor");

    // Move the head to a newer snapshot before the old root is reclaimed.
    let next_tx = db.begin_transaction();
    stage_event_timeline_batch(
        &next_tx,
        ShardType::EventDag,
        [0xFA; 16],
        &[],
        &topo_index,
        &stream_index,
        &[event(room, 4, 4, 4)],
    )
    .unwrap();
    next_tx.commit().unwrap();

    let evict = db.begin_transaction();
    evict
        .remove(
            ShardType::EventDag,
            topo_index.collection_id(),
            cursor.root_id(),
        )
        .unwrap();
    evict.commit().unwrap();

    let page = page_with_rebind_fallback(&topo_index, &room, Some(&cursor), true, 10).unwrap();
    assert_eq!(
        page.entries
            .iter()
            .map(|entry| entry.event_node_id)
            .collect::<Vec<_>>(),
        [
            entries[1].event_node_id,
            entries[2].event_node_id,
            event(room, 4, 4, 4).event_node_id,
        ]
    );

    drop(db);
    std::fs::remove_dir_all(test_path).unwrap();
}
