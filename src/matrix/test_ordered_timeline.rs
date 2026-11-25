use super::*;
use crate::database::Database;
use crate::storage::StorageEngine;
use bytes::Bytes;
use std::path::PathBuf;

fn test_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "mtxdb-ordered-event-timeline-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn event(room: [u8; 16], depth: i64, stream: i64, id: u8) -> EventTimelineEntry {
    let mut node_id = [0; 16];
    node_id[15] = id;
    let mut event_ref = [0; 32];
    event_ref[31] = id;
    EventTimelineEntry::new(room, depth, stream, node_id, event_ref)
}

#[test]
fn event_key_orders_match_their_memcomparable_encodings() {
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
                NodeData::new(Bytes::from(vec![entry.event_ref[31]])),
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
