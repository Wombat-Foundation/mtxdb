#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::database::Database;
    use crate::interop::{
        auth_counter_record_id, auth_reverse_record_id, decode_verdict, encode_auth_counter,
        encode_verdict, event_record_id, event_verdict_collection_id, room_auth_collection_id,
        room_event_collection_id, Verdict,
    };
    use crate::layout::ShardType;
    use crate::room_flag::{Flag, FlagChange, FlagQuery, RoomFlagIndex, CHUNK_CAPACITY};
    use crate::storage::{NodeData, StorageEngine};

    fn test_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mtxdb-test-room-flag-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn empty_room_flag_index_reports_zero_and_empty() {
        let db = Database::open(test_root("empty")).unwrap();
        let index = RoomFlagIndex::new(b"!room:example.com");

        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 0);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
            0
        );
        assert_eq!(
            index.count(&db, &FlagQuery::not(Flag::Rejected)).unwrap(),
            0
        );
        assert!(index
            .select_shorts(&db, &FlagQuery::has(Flag::Present))
            .unwrap()
            .is_empty());
        assert!(index
            .select(&db, &FlagQuery::has(Flag::Present))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn single_chunk_counts_and_queries() {
        let db = Database::open(test_root("single_chunk")).unwrap();
        let room_id = b"!single_chunk:example.com";
        let index = RoomFlagIndex::new(room_id);

        let tx = db.begin_transaction();
        // 5 events: 1, 2, 3, 4, 5
        let mut changes = Vec::new();
        for id in 1..=5 {
            changes.push(FlagChange::set(id, Flag::Present));
        }
        // event 2 is soft-failed
        changes.push(FlagChange::set(2, Flag::SoftFailed));
        // event 3 is rejected
        changes.push(FlagChange::set(3, Flag::Rejected));
        // event 4 is both soft-failed and rejected
        changes.push(FlagChange::set(4, Flag::SoftFailed));
        changes.push(FlagChange::set(4, Flag::Rejected));
        // event 5 is an outlier
        changes.push(FlagChange::set(5, Flag::Outlier));

        index.stage(&tx, &changes).unwrap();
        tx.commit().unwrap();

        // Counts
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 5);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
            2
        );
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
            2
        );
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Outlier)).unwrap(), 1);

        // Not(Rejected) = Present \ Rejected = {1, 2, 5}
        assert_eq!(
            index.count(&db, &FlagQuery::not(Flag::Rejected)).unwrap(),
            3
        );
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::not(Flag::Rejected))
                .unwrap(),
            vec![1, 2, 5]
        );

        // Soft-failed AND NOT rejected = {2, 4} \ {3, 4} = {2}
        let query_soft_not_rejected = FlagQuery::and([
            FlagQuery::has(Flag::SoftFailed),
            FlagQuery::not(Flag::Rejected),
        ]);
        assert_eq!(index.count(&db, &query_soft_not_rejected).unwrap(), 1);
        assert_eq!(
            index.select_shorts(&db, &query_soft_not_rejected).unwrap(),
            vec![2]
        );

        // Soft-failed OR Rejected = {2, 3, 4}
        let query_problematic = FlagQuery::or([
            FlagQuery::has(Flag::SoftFailed),
            FlagQuery::has(Flag::Rejected),
        ]);
        assert_eq!(index.count(&db, &query_problematic).unwrap(), 3);
        assert_eq!(
            index.select_shorts(&db, &query_problematic).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn multi_chunk_boundary_and_clearing() {
        let db = Database::open(test_root("multi_chunk")).unwrap();
        let room_id = b"!multi_chunk:example.com";
        let index = RoomFlagIndex::new(room_id);

        // Spanning chunk 0, 1, 2
        let id_chunk0 = 10;
        let id_chunk1 = CHUNK_CAPACITY + 15;
        let id_chunk2 = CHUNK_CAPACITY * 2 + 50;

        let tx = db.begin_transaction();
        index
            .stage(
                &tx,
                &[
                    FlagChange::set(id_chunk0, Flag::Present),
                    FlagChange::set(id_chunk1, Flag::Present),
                    FlagChange::set(id_chunk2, Flag::Present),
                    FlagChange::set(id_chunk0, Flag::Timeline),
                    FlagChange::set(id_chunk1, Flag::Timeline),
                    FlagChange::set(id_chunk2, Flag::SoftFailed),
                ],
            )
            .unwrap();
        tx.commit().unwrap();

        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 3);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
            2
        );
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
            1
        );
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Timeline))
                .unwrap(),
            vec![id_chunk0, id_chunk1]
        );

        // Clear Timeline on id_chunk1 and set Outlier
        let tx2 = db.begin_transaction();
        index
            .stage(
                &tx2,
                &[
                    FlagChange::clear(id_chunk1, Flag::Timeline),
                    FlagChange::set(id_chunk1, Flag::Outlier),
                ],
            )
            .unwrap();
        tx2.commit().unwrap();

        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
            1
        );
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Timeline))
                .unwrap(),
            vec![id_chunk0]
        );
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Outlier)).unwrap(), 1);
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Outlier))
                .unwrap(),
            vec![id_chunk1]
        );
    }

    #[test]
    fn select_resolves_reverse_event_ids() {
        let db = Database::open(test_root("select_rev")).unwrap();
        let room_id = b"!reverse:example.com";
        let index = RoomFlagIndex::new(room_id);
        let auth_col = room_auth_collection_id(room_id);

        // Populate auth reverse records
        let tx = db.begin_transaction();
        let ev1 = b"$event_1:example.com";
        let ev2 = b"$event_2:example.com";
        tx.put(
            ShardType::Edges,
            auth_col,
            auth_reverse_record_id(1),
            &NodeData::new(bytes::Bytes::from_static(ev1)),
        )
        .unwrap();
        tx.put(
            ShardType::Edges,
            auth_col,
            auth_reverse_record_id(2),
            &NodeData::new(bytes::Bytes::from_static(ev2)),
        )
        .unwrap();

        index
            .stage(
                &tx,
                &[
                    FlagChange::set(1, Flag::Present),
                    FlagChange::set(1, Flag::SoftFailed),
                    FlagChange::set(2, Flag::Present),
                ],
            )
            .unwrap();
        tx.commit().unwrap();

        let soft = index
            .select(&db, &FlagQuery::has(Flag::SoftFailed))
            .unwrap();
        assert_eq!(soft, vec![ev1.to_vec()]);

        let all_present = index.select(&db, &FlagQuery::has(Flag::Present)).unwrap();
        assert_eq!(all_present, vec![ev1.to_vec(), ev2.to_vec()]);

        // Bounded select
        let limited = index
            .select_bounded(&db, &FlagQuery::has(Flag::Present), 1)
            .unwrap();
        assert_eq!(limited, vec![ev1.to_vec()]);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn rebuild_populates_flag_index_from_events_and_verdicts() {
        let db = Database::open(test_root("rebuild")).unwrap();
        let room_id = b"!rebuild:example.com";
        let index = RoomFlagIndex::new(room_id);
        let auth_col = room_auth_collection_id(room_id);
        let event_col = room_event_collection_id(room_id);

        let ev1 = b"$rebuild_ev1:example.com";
        let ev2 = b"$rebuild_ev2:example.com";
        let node1 = event_record_id(ev1);
        let node2 = event_record_id(ev2);

        let tx = db.begin_transaction();
        // 1. Counter = 2
        tx.put(
            ShardType::Edges,
            auth_col,
            auth_counter_record_id(),
            &NodeData::new(bytes::Bytes::from(encode_auth_counter(2).to_vec())),
        )
        .unwrap();
        // 2. Reverse records
        tx.put(
            ShardType::Edges,
            auth_col,
            auth_reverse_record_id(1),
            &NodeData::new(bytes::Bytes::from_static(ev1)),
        )
        .unwrap();
        tx.put(
            ShardType::Edges,
            auth_col,
            auth_reverse_record_id(2),
            &NodeData::new(bytes::Bytes::from_static(ev2)),
        )
        .unwrap();
        // 3. Event payloads in room_event_collection
        tx.put(
            ShardType::EventDag,
            event_col,
            node1,
            &NodeData::new(bytes::Bytes::from_static(b"body1")),
        )
        .unwrap();
        tx.put(
            ShardType::EventDag,
            event_col,
            node2,
            &NodeData::new(bytes::Bytes::from_static(b"body2")),
        )
        .unwrap();
        // 4. Verdict for ev1: soft_fail
        let v1 = Verdict {
            soft_fail: Some(1),
            ..Verdict::default()
        };
        tx.put(
            ShardType::EventDag,
            event_verdict_collection_id(&node1),
            node1,
            &NodeData::new(bytes::Bytes::from(encode_verdict(ev1, &v1).unwrap())),
        )
        .unwrap();
        // 5. Verdict for ev2: outlier + redacted
        let v2 = Verdict {
            outlier: true,
            redacted_by: Some(b"$redactor:example.com".to_vec()),
            ..Verdict::default()
        };
        tx.put(
            ShardType::EventDag,
            event_verdict_collection_id(&node2),
            node2,
            &NodeData::new(bytes::Bytes::from(encode_verdict(ev2, &v2).unwrap())),
        )
        .unwrap();
        tx.commit().unwrap();

        // 6. Append ev1 to TimelineIndex
        let timeline_index = crate::timeline::TimelineIndex::open(
            db.pool(ShardType::EventDag).as_ref(),
            std::str::from_utf8(room_id).unwrap(),
        );
        let room_scope = crate::timeline::room_scope_id(room_id);
        timeline_index
            .append(
                &db,
                ShardType::EventDag,
                &[crate::timeline::TimelineEntry::new(
                    room_scope,
                    1,
                    1,
                    node1,
                    crate::timeline::event_id_ref(ev1),
                )],
            )
            .unwrap();

        // Rebuild index
        let count = index.rebuild(&db).unwrap();
        assert_eq!(count, 2);

        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 2);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
            1
        );
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap(),
            1
        );
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Outlier)).unwrap(), 1);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Redacted)).unwrap(),
            1
        );
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
            0
        );

        let soft = index
            .select(&db, &FlagQuery::has(Flag::SoftFailed))
            .unwrap();
        assert_eq!(soft, vec![ev1.to_vec()]);

        let timeline = index.select(&db, &FlagQuery::has(Flag::Timeline)).unwrap();
        assert_eq!(timeline, vec![ev1.to_vec()]);

        let redacted = index.select(&db, &FlagQuery::has(Flag::Redacted)).unwrap();
        assert_eq!(redacted, vec![ev2.to_vec()]);

        // Verify rebuild clears stale bits: manually set Rejected on short 1, then rebuild
        let tx_stale = db.begin_transaction();
        index
            .stage(&tx_stale, &[FlagChange::set(1, Flag::Rejected)])
            .unwrap();
        tx_stale.commit().unwrap();
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
            1
        );

        index.rebuild(&db).unwrap();
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
            0
        );
    }

    #[test]
    fn benchmark_flag_query_vs_raw_scan() {
        const TOTAL_EVENTS: u32 = 5000;
        const SOFT_FAILED_COUNT: u32 = 25;

        let db = Database::open(test_root("bench")).unwrap();
        let room_id = b"!bench:example.com";
        let index = RoomFlagIndex::new(room_id);
        let auth_col = room_auth_collection_id(room_id);
        let event_col = room_event_collection_id(room_id);

        let mut flag_changes = Vec::new();
        let mut all_event_ids = Vec::with_capacity(TOTAL_EVENTS as usize);

        let tx = db.begin_transaction();
        for i in 1..=TOTAL_EVENTS {
            let eid = format!("$event_{i}:example.com");
            let eid_bytes = eid.into_bytes();
            let node_id = event_record_id(&eid_bytes);
            all_event_ids.push((eid_bytes.clone(), node_id));

            tx.put(
                ShardType::Edges,
                auth_col,
                auth_reverse_record_id(i),
                &NodeData::new(bytes::Bytes::from(eid_bytes.clone())),
            )
            .unwrap();
            tx.put(
                ShardType::EventDag,
                event_col,
                node_id,
                &NodeData::new(bytes::Bytes::from_static(b"{\"dummy\":true}")),
            )
            .unwrap();

            flag_changes.push(FlagChange::set(i, Flag::Present));
            if i % (TOTAL_EVENTS / SOFT_FAILED_COUNT) == 0 {
                flag_changes.push(FlagChange::set(i, Flag::SoftFailed));
                let v = Verdict {
                    soft_fail: Some(1),
                    ..Verdict::default()
                };
                tx.put(
                    ShardType::EventDag,
                    event_verdict_collection_id(&node_id),
                    node_id,
                    &NodeData::new(bytes::Bytes::from(encode_verdict(&eid_bytes, &v).unwrap())),
                )
                .unwrap();
            }
        }
        index.stage(&tx, &flag_changes).unwrap();
        tx.commit().unwrap();

        // 1. Measure RoomFlagIndex::count
        let start_count = Instant::now();
        let count = index.count(&db, &FlagQuery::has(Flag::SoftFailed)).unwrap();
        let dur_count = start_count.elapsed();
        assert_eq!(count, u64::from(SOFT_FAILED_COUNT));

        // 2. Measure RoomFlagIndex::select
        let start_select = Instant::now();
        let selected = index
            .select(&db, &FlagQuery::has(Flag::SoftFailed))
            .unwrap();
        let dur_select = start_select.elapsed();
        assert_eq!(selected.len(), SOFT_FAILED_COUNT as usize);

        // 3. Measure Raw Scan: checking verdict for each event
        let start_scan = Instant::now();
        let mut scan_count = 0_u64;
        for (_eid, node_id) in &all_event_ids {
            let verdict_col = event_verdict_collection_id(node_id);
            if let Some(v_data) = db
                .pool(ShardType::EventDag)
                .get(&verdict_col, node_id)
                .unwrap()
            {
                if let Ok((_, verdict)) = decode_verdict(&v_data.bytes) {
                    if verdict.soft_fail.is_some() {
                        scan_count += 1;
                    }
                }
            }
        }
        let dur_scan = start_scan.elapsed();
        assert_eq!(scan_count, u64::from(SOFT_FAILED_COUNT));

        eprintln!("\n=== Benchmark ({TOTAL_EVENTS} events, {SOFT_FAILED_COUNT} soft-failed) ===");
        eprintln!("RoomFlagIndex::count  : {dur_count:?}");
        eprintln!("RoomFlagIndex::select : {dur_select:?}");
        eprintln!("Raw Event/Verdict Scan: {dur_scan:?}");
        if dur_count.as_secs_f64() > 0.0 {
            eprintln!(
                "Count speedup: {:.1}x faster",
                dur_scan.as_secs_f64() / dur_count.as_secs_f64()
            );
        }
        if dur_select.as_secs_f64() > 0.0 {
            eprintln!(
                "Select speedup: {:.1}x faster",
                dur_scan.as_secs_f64() / dur_select.as_secs_f64()
            );
        }
    }
}
