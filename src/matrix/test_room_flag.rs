#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use crate::csr_adjacency::CsrAdjacency;
    use crate::database::{Database, DatabaseTransaction};
    use crate::interop::{
        decode_verdict, encode_verdict, event_record_id, event_verdict_collection_id,
        room_event_collection_id, room_flag_collection_id, Verdict,
    };
    use crate::layout::ShardType;
    use crate::room_flag::{
        flag_chunk_node_id, Flag, FlagChange, FlagQuery, RebuildInputs, RoomFlagIndex,
        CHUNK_CAPACITY, FLAG_META_ID,
    };
    use crate::storage::{NodeData, StorageEngine};
    use bytes::Bytes;

    struct TestRoot(PathBuf);

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    impl TestRoot {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    fn test_root(name: &str) -> TestRoot {
        let path = std::env::temp_dir().join(format!(
            "mtxdb-test-room-flag-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        TestRoot(path)
    }

    #[test]
    fn empty_room_flag_index_reports_zero_and_empty() {
        let root = test_root("empty");
        let db = Database::open(root.path().to_path_buf()).unwrap();
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
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Present))
                .unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(
            index.select(&db, &FlagQuery::has(Flag::Present)).unwrap(),
            Vec::<Vec<u8>>::new()
        );
    }

    #[test]
    fn no_op_stage_writes_no_records() {
        let root = test_root("no_op_stage");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!no_op_stage:example.com";
        let index = RoomFlagIndex::new(room_id);
        let change = [
            FlagChange::set(1, Flag::Present),
            FlagChange::set(1, Flag::Rejected),
        ];
        let tx = db.begin_transaction();
        index.stage(&tx, &change).unwrap();
        tx.commit().unwrap();

        let meta = index.meta(&db).unwrap();
        let flag_col = room_flag_collection_id(room_id);
        let chunk_col = index.chunk_collection(meta.generation);
        let read = db.begin_transaction();
        let (_, meta_before) = read
            .get_with_record_versions(ShardType::EventDag, &flag_col, &[FLAG_META_ID])
            .unwrap();
        let (_, chunks_before) = read
            .get_with_record_versions(
                ShardType::EventDag,
                &chunk_col,
                &[
                    flag_chunk_node_id(Flag::Present, 0),
                    flag_chunk_node_id(Flag::Rejected, 0),
                ],
            )
            .unwrap();
        drop(read);

        let tx = db.begin_transaction();
        index.stage(&tx, &change).unwrap();
        tx.commit().unwrap();
        let tx = db.begin_transaction();
        index
            .stage(&tx, &[FlagChange::clear(999, Flag::Rejected)])
            .unwrap();
        tx.commit().unwrap();

        let read = db.begin_transaction();
        let (_, meta_after) = read
            .get_with_record_versions(ShardType::EventDag, &flag_col, &[FLAG_META_ID])
            .unwrap();
        let (_, chunks_after) = read
            .get_with_record_versions(
                ShardType::EventDag,
                &chunk_col,
                &[
                    flag_chunk_node_id(Flag::Present, 0),
                    flag_chunk_node_id(Flag::Rejected, 0),
                ],
            )
            .unwrap();
        drop(read);
        assert_eq!(meta_after, meta_before, "no-op stage rewrote flag metadata");
        assert_eq!(
            chunks_after, chunks_before,
            "no-op stage rewrote flag chunks"
        );
    }

    #[test]
    fn clears_do_not_raise_the_indexed_short_id_bound() {
        let root = test_root("clear_bound");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let index = RoomFlagIndex::new(b"!clear_bound:example.com");

        let tx = db.begin_transaction();
        index
            .stage(
                &tx,
                &[
                    FlagChange::set(1, Flag::Present),
                    // A clear cannot create a bit, so this high short id must
                    // not become the metadata's scan bound.
                    FlagChange::clear(100_000, Flag::Rejected),
                ],
            )
            .unwrap();
        tx.commit().unwrap();

        assert_eq!(index.meta(&db).unwrap().max_short_id, 1);
    }

    #[test]
    fn single_chunk_counts_and_queries() {
        let root = test_root("single_chunk");
        let db = Database::open(root.path().to_path_buf()).unwrap();
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
        let root = test_root("multi_chunk");
        let db = Database::open(root.path().to_path_buf()).unwrap();
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
        let root = test_root("select_rev");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!reverse:example.com";
        let index = RoomFlagIndex::new(room_id);

        let ev1 = b"$event_1:example.com";
        let ev2 = b"$event_2:example.com";
        seed_shorts(&db, room_id, &[ev1, ev2]);
        let tx = db.begin_transaction();

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
        let root = test_root("rebuild");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!rebuild:example.com";
        let index = RoomFlagIndex::new(room_id);
        let event_col = room_event_collection_id(room_id);

        let ev1 = b"$rebuild_ev1:example.com";
        let ev2 = b"$rebuild_ev2:example.com";
        let node1 = event_record_id(ev1);
        let node2 = event_record_id(ev2);

        seed_shorts(&db, room_id, &[ev1, ev2]);
        let tx = db.begin_transaction();
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

        // A caller-defined flag has no event source to re-derive it from, so a
        // rebuild must carry it into the new generation.
        let tx_custom = db.begin_transaction();
        index
            .stage(
                &tx_custom,
                &[FlagChange::set(2, Flag::custom(7).expect("in range"))],
            )
            .unwrap();
        tx_custom.commit().unwrap();
        assert_eq!(
            index
                .count(&db, &FlagQuery::has(Flag::custom(7).expect("in range")))
                .unwrap(),
            1
        );

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
        assert_eq!(
            index
                .count(&db, &FlagQuery::has(Flag::custom(7).expect("in range")))
                .unwrap(),
            1,
            "a rebuild must retain caller-defined custom flags"
        );
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_flag_query_vs_raw_scan() {
        const TOTAL_EVENTS: u32 = 5000;
        const SOFT_FAILED_COUNT: u32 = 25;

        let root = test_root("bench");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!bench:example.com";
        let index = RoomFlagIndex::new(room_id);
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

    #[test]
    fn repeated_stage_in_single_transaction_succeeds_commit() {
        let root = test_root("repeated_stage");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!repeated:example.com";
        let index = RoomFlagIndex::new(room_id);

        // Commit initial record so chunk and meta have published versions > 0
        let tx1 = db.begin_transaction();
        index
            .stage(&tx1, &[FlagChange::set(1, Flag::Present)])
            .unwrap();
        tx1.commit().unwrap();
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 1);

        // Now in a single transaction, stage multiple batches touching the same chunk and meta
        let tx2 = db.begin_transaction();
        index
            .stage(&tx2, &[FlagChange::set(2, Flag::Present)])
            .unwrap();
        // A no-op stage after the earlier stage already wrote this
        // transaction's chunk and metadata must still commit.
        index
            .stage(&tx2, &[FlagChange::set(2, Flag::Present)])
            .unwrap();
        index
            .stage(&tx2, &[FlagChange::set(3, Flag::Present)])
            .unwrap();
        index
            .stage(
                &tx2,
                &[
                    FlagChange::set(2, Flag::Timeline),
                    FlagChange::set(3, Flag::Timeline),
                ],
            )
            .unwrap();

        // This commit must succeed and not abort with StaleRead { expected: 0, actual: ... }
        tx2.commit()
            .expect("repeated stage in one transaction should not fail commit with StaleRead");

        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 3);
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Timeline)).unwrap(),
            2
        );
    }

    #[test]
    fn stage_guards_live_journal_less_chunk_versions() {
        let root = test_root("journal_less_chunk_version");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let index = RoomFlagIndex::new(b"!journal-less:example.com");
        let chunk_col = index.chunk_collection(1);
        let chunk_id = flag_chunk_node_id(Flag::Present, 0);

        // Seed a pack record outside the transaction journal. Its live record
        // version is zero, but it still needs a CAS expectation when staged.
        let mut initial = crate::bitmap_set::BitmapSet::new(index.domain());
        initial.insert(1);
        db.pool(ShardType::EventDag)
            .put(
                &chunk_col,
                &chunk_id,
                &NodeData::new(Bytes::from(initial.encode().unwrap())),
            )
            .unwrap();
        db.pool(ShardType::EventDag).sync().unwrap();

        let staged = db.begin_transaction();
        index
            .stage(&staged, &[FlagChange::set(2, Flag::Present)])
            .unwrap();

        let concurrent = db.begin_transaction();
        let mut replacement = crate::bitmap_set::BitmapSet::new(index.domain());
        replacement.insert(3);
        concurrent
            .put(
                ShardType::EventDag,
                chunk_col,
                chunk_id,
                &NodeData::new(Bytes::from(replacement.encode().unwrap())),
            )
            .unwrap();
        concurrent.commit().unwrap();

        assert!(matches!(
            staged.commit(),
            Err(crate::storage::StorageError::StaleRead { .. })
        ));
    }

    #[test]
    fn stage_guards_live_journal_less_meta_versions() {
        let root = test_root("journal_less_meta_version");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let index = RoomFlagIndex::new(b"!journal-less-meta:example.com");
        let meta = crate::room_flag::encode_meta(crate::room_flag::FlagMeta {
            generation: 1,
            max_short_id: 5,
        });

        // Seed metadata outside the transaction journal. Its live record
        // version is zero, but it still needs a CAS expectation when staged.
        db.pool(ShardType::EventDag)
            .put(
                index.collection(),
                &FLAG_META_ID,
                &NodeData::new(Bytes::from(meta.to_vec())),
            )
            .unwrap();
        db.pool(ShardType::EventDag).sync().unwrap();

        let staged = db.begin_transaction();
        index
            .stage(&staged, &[FlagChange::set(6, Flag::Present)])
            .unwrap();

        let concurrent = db.begin_transaction();
        concurrent
            .put(
                ShardType::EventDag,
                *index.collection(),
                FLAG_META_ID,
                &NodeData::new(Bytes::from(meta.to_vec())),
            )
            .unwrap();
        concurrent.commit().unwrap();

        assert!(matches!(
            staged.commit(),
            Err(crate::storage::StorageError::StaleRead { .. })
        ));
    }

    #[test]
    fn custom_flag_bounds_and_byte_mapping() {
        assert_eq!(Flag::MAX_CUSTOM, 191);

        // Checked constructor bounds
        for i in 0..=191 {
            let flag = Flag::custom(i).expect("index within bounds");
            assert_eq!(flag.to_byte(), 64_u8 + i);
            assert_eq!(Flag::from_byte(64_u8 + i), Some(flag));
        }
        assert_eq!(Flag::custom(192), None);
        assert_eq!(Flag::custom(255), None);

        // Standard flags round-trip
        let standard = [
            (Flag::Present, 0),
            (Flag::Timeline, 1),
            (Flag::Outlier, 2),
            (Flag::Rejected, 3),
            (Flag::SoftFailed, 4),
            (Flag::Redacted, 5),
        ];
        for (flag, byte) in standard {
            assert_eq!(flag.to_byte(), byte);
            assert_eq!(Flag::from_byte(byte), Some(flag));
        }

        // Reserved bytes 6..=63 return None
        for b in 6..=63 {
            assert_eq!(
                Flag::from_byte(b),
                None,
                "byte {b} is reserved and must return None"
            );
        }

        // Custom flags byte mapping (64..=255)
        for i in 0..=191 {
            let custom_flag = Flag::custom(i).unwrap();
            let byte = 64_u8.saturating_add(i);
            assert_eq!(custom_flag.to_byte(), byte);
            assert_eq!(Flag::from_byte(byte), Some(custom_flag));
        }
    }

    #[test]
    fn rebuild_generation_swap_and_metadata_evolution() {
        let root = test_root("rebuild_gen");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!gen_swap:example.com";
        let index = RoomFlagIndex::new(room_id);
        seed_shorts(&db, room_id, &[b"$ev1:example.com"]);
        let tx = db.begin_transaction();
        seed_single_event_room(&tx, room_id);
        tx.commit().unwrap();

        // Initial unindexed state
        let initial_meta = index.meta(&db).unwrap();
        assert_eq!(initial_meta.generation, 1);

        let count = index.rebuild(&db).unwrap();
        assert_eq!(count, 1);
        let meta1 = index.meta(&db).unwrap();
        assert_eq!(meta1.generation, 2);
        assert_eq!(meta1.max_short_id, 1);
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 1);

        // Second rebuild should increment generation to 3
        let count2 = index.rebuild(&db).unwrap();
        assert_eq!(count2, 1);
        let meta2 = index.meta(&db).unwrap();
        assert_eq!(meta2.generation, 3);
        assert_eq!(meta2.max_short_id, 1);
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 1);
    }

    /// Allocate room short ids 1..=n for `ids`, in order, through the room's
    /// adjacency event scope.
    fn seed_shorts(db: &Database, room_id: &[u8], ids: &[&[u8]]) {
        let adjacency = CsrAdjacency::new(std::str::from_utf8(room_id).unwrap());
        for id in ids {
            adjacency
                .record_event(db, std::str::from_utf8(id).unwrap(), &[], &[], None)
                .unwrap();
        }
    }

    /// Seeds a room with one event (`$ev1:example.com`, auth counter 1) so the
    /// rebuild tests start from the same minimal authorised-event state.
    fn seed_single_event_room(tx: &DatabaseTransaction<'_>, room_id: &[u8]) {
        let event_col = room_event_collection_id(room_id);
        let ev1 = b"$ev1:example.com";
        let node1 = event_record_id(ev1);
        tx.put(
            ShardType::EventDag,
            event_col,
            node1,
            &NodeData::new(bytes::Bytes::from_static(b"ev1 body")),
        )
        .unwrap();
    }

    fn assert_noop_stage_is_fenced_by_meta(db: &Database, index: &RoomFlagIndex) {
        let before_noop = index.meta(db).unwrap();
        let noop_tx = db.begin_transaction();
        index
            .stage(&noop_tx, &[FlagChange::set(1, Flag::Present)])
            .unwrap();
        let publish_tx = db.begin_transaction();
        let (_, publish_versions) = publish_tx
            .get_with_record_versions(
                index.pool(),
                index.collection(),
                &[crate::room_flag::FLAG_META_ID],
            )
            .unwrap();
        let publish_token = publish_versions.first().copied().unwrap_or(0);
        publish_tx
            .put(
                index.pool(),
                *index.collection(),
                crate::room_flag::FLAG_META_ID,
                &NodeData::new(bytes::Bytes::from(
                    crate::room_flag::encode_meta(crate::room_flag::FlagMeta {
                        generation: before_noop.generation.saturating_add(1),
                        max_short_id: before_noop.max_short_id,
                    })
                    .to_vec(),
                )),
            )
            .unwrap();
        publish_tx
            .expect_record_version(
                index.pool(),
                *index.collection(),
                crate::room_flag::FLAG_META_ID,
                publish_token,
            )
            .unwrap();
        publish_tx.commit().unwrap();
        let commit_res = noop_tx.commit();
        assert!(commit_res.is_err());
        assert!(commit_res.unwrap_err().is_stale_read());
    }

    #[test]
    fn stale_meta_token_after_rebuild_publish_fails_commit() {
        let root = test_root("concurrent_rebuild");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!concurrent:example.com";
        let index = RoomFlagIndex::new(room_id);
        seed_shorts(&db, room_id, &[b"$ev1:example.com"]);
        let tx = db.begin_transaction();
        seed_single_event_room(&tx, room_id);
        index
            .stage(&tx, &[FlagChange::set(1, Flag::Present)])
            .unwrap();
        tx.commit().unwrap();

        // 1. Snapshot the current metadata and its record version token
        let (_meta_nodes, meta_versions) = db
            .begin_transaction()
            .get_with_record_versions(
                index.pool(),
                index.collection(),
                &[crate::room_flag::FLAG_META_ID],
            )
            .unwrap();
        let meta_token = meta_versions.first().copied().unwrap_or(0);

        // 2. A writer updates an existing short ID (1). Even though
        // max_short_id does not grow, the metadata CAS record must change so a
        // rebuild cannot publish a generation that misses this flag update.
        let tx_writer = db.begin_transaction();
        index
            .stage(&tx_writer, &[FlagChange::set(1, Flag::Timeline)])
            .unwrap();
        tx_writer.commit().unwrap();

        // Verify FLAG_META_ID record version changed.
        let (_meta_nodes_after, meta_versions_after) = db
            .begin_transaction()
            .get_with_record_versions(
                index.pool(),
                index.collection(),
                &[crate::room_flag::FLAG_META_ID],
            )
            .unwrap();
        assert_ne!(
            meta_versions_after[0], meta_token,
            "FLAG_META_ID must guard updates even when max_short_id stays fixed"
        );

        // 3. A generation swap (rebuild publish) updates FLAG_META_ID
        let (current_meta, new_gen, publish_meta_token) =
            index.reserve_rebuild_generation(&db).unwrap();
        let publish_tx = db.begin_transaction();
        let new_meta = crate::room_flag::FlagMeta {
            generation: new_gen,
            max_short_id: current_meta.max_short_id,
        };
        publish_tx
            .put(
                index.pool(),
                *index.collection(),
                crate::room_flag::FLAG_META_ID,
                &NodeData::new(bytes::Bytes::from(
                    crate::room_flag::encode_meta(new_meta).to_vec(),
                )),
            )
            .unwrap();
        publish_tx
            .expect_record_version(
                index.pool(),
                *index.collection(),
                crate::room_flag::FLAG_META_ID,
                publish_meta_token,
            )
            .unwrap();
        publish_tx.commit().unwrap();

        // 4. An in-flight writer that prepared with the old meta_token now attempts to commit;
        // because stage kept expect_record_version on FLAG_META_ID, it must fail with StaleRead
        let inflight_tx = db.begin_transaction();
        inflight_tx
            .expect_record_version(
                index.pool(),
                *index.collection(),
                crate::room_flag::FLAG_META_ID,
                meta_token,
            )
            .unwrap();
        let commit_res = inflight_tx.commit();
        assert!(commit_res.is_err());
        assert!(commit_res.unwrap_err().is_stale_read());

        assert_noop_stage_is_fenced_by_meta(&db, &index);
    }

    #[test]
    fn rebuild_fails_if_flag_committed_after_reservation() {
        let root = test_root("rebuild_flag_race");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!rebuild_race:example.com";
        let index = RoomFlagIndex::new(room_id);
        seed_shorts(&db, room_id, &[b"$ev1:example.com"]);
        let tx = db.begin_transaction();
        seed_single_event_room(&tx, room_id);
        index
            .stage(&tx, &[FlagChange::set(1, Flag::Present)])
            .unwrap();
        tx.commit().unwrap();

        // 1. Reserve rebuild generation and capture snapshot meta_token
        let (current_meta, new_gen, meta_token) = index.reserve_rebuild_generation(&db).unwrap();
        let target_col = index.chunk_collection(new_gen);

        // 2. Commit a flag change after reservation but before rebuild publish
        let tx_writer = db.begin_transaction();
        index
            .stage(&tx_writer, &[FlagChange::set(1, Flag::Timeline)])
            .unwrap();
        tx_writer.commit().unwrap();

        // 3. Attempt to publish via the library rebuild path
        let res = index.rebuild_generation(
            &db,
            RebuildInputs {
                counter: 1,
                current_meta,
                new_gen,
                target_col,
                meta_token,
            },
        );
        assert!(res.is_err());
        assert!(
            res.unwrap_err().is_stale_read(),
            "rebuild publish must abort when a flag change commits after reservation"
        );

        // 4. Verify that the committed flag change remains intact and generation is not corrupted
        let active_meta = index.meta(&db).unwrap();
        assert_eq!(active_meta.generation, current_meta.generation);
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Present))
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            index
                .select_shorts(&db, &FlagQuery::has(Flag::Timeline))
                .unwrap(),
            vec![1]
        );
    }

    #[test]
    fn rebuild_cleans_leftover_chunks_from_crashed_rebuild() {
        let root = test_root("crash_rebuild");
        let db = Database::open(root.path().to_path_buf()).unwrap();
        let room_id = b"!crash:example.com";
        let index = RoomFlagIndex::new(room_id);
        seed_shorts(&db, room_id, &[b"$ev1:example.com"]);
        let tx = db.begin_transaction();
        seed_single_event_room(&tx, room_id);
        tx.commit().unwrap();

        // 1. Target generation for a rebuild starting from default generation 1 is generation 2.
        // Simulate a crashed rebuild that staged leftover/poisoned data into generation 2's chunk collection:
        let target_gen = 2_u64;
        let abandoned_col = index.chunk_collection(target_gen);
        let crash_tx = db.begin_transaction();
        index
            .stage_into_collection(
                &crash_tx,
                abandoned_col,
                &[FlagChange::set(1, Flag::Rejected)],
            )
            .unwrap();
        crash_tx.commit().unwrap();

        // 2. Next rebuild runs to completion, pre-emptively clearing generation 2 before writing
        let count = index.rebuild(&db).unwrap();
        assert_eq!(count, 1);

        let published_meta = index.meta(&db).unwrap();
        assert_eq!(published_meta.generation, target_gen);

        // Stale leftover bits from crashed generation 2 must NOT be present!
        assert_eq!(
            index.count(&db, &FlagQuery::has(Flag::Rejected)).unwrap(),
            0
        );
        assert_eq!(index.count(&db, &FlagQuery::has(Flag::Present)).unwrap(), 1);
    }
}
