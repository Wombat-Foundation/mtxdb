//! End-to-end coverage of the persisted-index checkpoint (`index.checkpoint`):
//! the fast open path must serve exactly the data a full rescan would, the
//! slow path must still work when the checkpoint is stale/corrupt/missing,
//! and the two must agree on every spot lookup.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use bytes::Bytes;

    use mtxdb_core::storage::{NodeData, NodeId, StorageEngine};
    use mtxdb_core::PackfileStorage;

    fn collection_id(seed: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[15] = seed;
        id[0] = seed.wrapping_mul(7);
        id
    }

    fn node_id(seed: u64, index: u64) -> NodeId {
        let mut id = [0u8; 16];
        let mut x = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        id[..8].copy_from_slice(&x.to_le_bytes());
        let mut y = x.wrapping_add(0x7372_9A1E_4288_1F7D);
        y = (y ^ (y >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        y = (y ^ (y >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        id[8..].copy_from_slice(&(y ^ (y >> 31)).to_le_bytes());
        id
    }

    fn make_store(dir: &Path) -> PackfileStorage {
        let store = PackfileStorage::open(dir.to_path_buf()).unwrap();
        for c in 0..3u8 {
            let mut entries = Vec::new();
            for i in 0..150u64 {
                let nid = node_id(u64::from(c), i);
                let mut payload = vec![c; 64];
                payload[..8].copy_from_slice(&i.to_le_bytes());
                entries.push((nid, NodeData::new(Bytes::from(payload))));
            }
            store.put_many(&collection_id(c), &entries).unwrap();
        }
        store
    }

    fn payload_for(c: u8, i: u64) -> Vec<u8> {
        let mut payload = vec![c; 64];
        payload[..8].copy_from_slice(&i.to_le_bytes());
        payload
    }

    fn assert_all_records(store: &PackfileStorage) {
        for c in 0..3u8 {
            let cid = collection_id(c);
            for i in 0..150u64 {
                let nid = node_id(u64::from(c), i);
                let got = store
                    .get(&cid, &nid)
                    .expect("record must survive a reopen")
                    .expect("record must survive a reopen");
                assert_eq!(got.bytes.as_ref(), payload_for(c, i).as_slice());
            }
        }
        assert_eq!(
            store.collection_ids().len(),
            3,
            "all three collections survive"
        );
    }

    fn checkpoint_path(dir: &Path) -> PathBuf {
        dir.join(mtxdb_core::index::checkpoint::INDEX_CHECKPOINT_FILE)
    }

    #[test]
    fn reopen_from_checkpoint_serves_the_same_data() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_index_checkpoint_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        assert!(
            checkpoint_path(&dir).exists(),
            "sync_all must leave a persisted index checkpoint"
        );
        drop(store);

        // Writable reopen: fast path should load from the checkpoint.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);
        drop(reopened);

        // Read-only reopen (the lookup-heavy inspection path) likewise.
        let read_only = PackfileStorage::open_read_only(dir.clone()).unwrap();
        assert_all_records(&read_only);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_and_sync_timings_are_recorded() {
        use mtxdb_core::packfile::storage::OpenPath;

        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_open_timings_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();

        let sync = store
            .sync_timings()
            .expect("sync_all must record its phase timings");
        assert!(
            sync.total >= sync.pack_flush + sync.pack_fsync + sync.sidecar + sync.checkpoint,
            "the phases must all fit inside the measured total ({} vs {})",
            sync.total.as_nanos(),
            (sync.pack_flush + sync.pack_fsync + sync.sidecar + sync.checkpoint).as_nanos()
        );
        drop(store);

        // A synced store reopens on the checkpoint fast path, and the open's
        // breakdown must say so (this is the deserialize-vs-IO question the
        // instrumentation exists to answer — pin which path it reports).
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let open = reopened
            .open_timings()
            .expect("open must record its phase timings");
        assert_eq!(
            open.path,
            OpenPath::Checkpoint,
            "a store committed by sync_all must reopen from its checkpoint"
        );
        assert!(
            open.total >= open.index_materialization,
            "materialization must fit inside the open total"
        );
        assert!(
            open.full_scan == std::time::Duration::ZERO,
            "checkpoint path must not pay the fallback scan"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupt_or_missing_checkpoint_falls_back_to_rescan() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_index_checkpoint_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        drop(store);

        // Corrupt: clobber the file and reopen — must fall back and still be right.
        std::fs::write(checkpoint_path(&dir), vec![0xAB; 128]).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);
        drop(reopened);

        // Missing: remove it entirely — same fallback.
        std::fs::remove_file(checkpoint_path(&dir)).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unsynced_append_invalidates_checkpoint_then_recovers() {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_sync_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();

        // Append WITHOUT syncing: the pack on disk changes but the checkpoint
        // does not, so the next open must notice the fingerprint mismatch and
        // rescan rather than trusting the stale index.
        let cid = collection_id(1);
        let extra: Vec<(NodeId, u64)> = (150..200u64).map(|i| (node_id(1, i), i)).collect();
        store
            .put_many(
                &cid,
                &extra
                    .iter()
                    .map(|(nid, i)| (*nid, NodeData::new(Bytes::from(payload_for(9, *i)))))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        // Commit the appended frames to the page cache WITHOUT syncing: the
        // packs on disk change but the checkpoint does not (and buffered
        // bytes would be invisible to a fresh process entirely), so the next
        // open must notice the fingerprint mismatch and rescan rather than
        // trusting the stale index. The appended records survive because
        // they were flushed, not because they were synced.
        store.flush_all().unwrap();
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);
        for (nid, i) in &extra {
            let got = reopened
                .get(&cid, nid)
                .expect("unsynced record must be readable after reopen")
                .expect("unsynced record must be readable after reopen");
            assert_eq!(
                got.bytes.as_ref(),
                payload_for(9, *i).as_slice(),
                "unsynced record payload round-trips"
            );
        }
        drop(reopened);

        // After a sync the checkpoint is fresh again and the fast path works.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&store);
        store.sync_all().unwrap();
        drop(store);
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Accounting regression for the buffered append policy: a process crash
    /// with bytes still in the RAM buffer must leave neither the pack files
    /// nor the persisted fingerprint claiming those bytes. If either ran
    /// ahead of real pack bytes, a later open could *falsely* pass the
    /// fingerprint gate and serve an index describing data that was never
    /// durably written — a worse failure than the rescan fallback. Assert
    /// both halves independently at the last flushed boundary.
    #[test]
    fn buffered_crash_before_flush_leaves_pack_length_and_fingerprint_at_last_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_buffered_crash_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();

        // Snapshot the committed boundary BEFORE the unsynced buffered put:
        // the pack file's on-disk length and the checkpoint bytes holding
        // the pack_fingerprint (computed from exactly those lengths).
        let pack_path = dir.join("pack_0000000000000000.pack");
        let pack_len_before = std::fs::metadata(&pack_path).unwrap().len();
        let checkpoint_before = std::fs::read(checkpoint_path(&dir)).unwrap();
        assert!(
            pack_len_before > 0,
            "the synced store must have committed pack bytes to snapshot"
        );

        // Buffer one more record and never flush it; dropping the store is
        // the crash-equivalent (no flush, no sync, no checkpoint rewrite).
        // NOTE: this couples ShardPool::drop to "best-effort stats persist
        // only, no pending-buffer flush". If drop ever gains a flush-on-drop
        // convenience path, this test would start validating drop's flush
        // instead of crash semantics -- it should fail loudly, not silently
        // pass, so keep the drop path here honest.
        let store = store.with_append_policy(mtxdb_core::shard::AppendPolicy::buffered());
        let cid = collection_id(1);
        let ghost = node_id(1, 200);
        store
            .put(
                &cid,
                &ghost,
                &NodeData::new(Bytes::from(payload_for(9, 200))),
            )
            .unwrap();
        drop(store);

        // (1) The buffered bytes never reached the pack file: its length is
        // exactly the last flushed boundary, no tail.
        assert_eq!(
            std::fs::metadata(&pack_path).unwrap().len(),
            pack_len_before,
            "crash with an unflushed buffered record must not grow the pack file"
        );

        // (2) The persisted fingerprint did not incorporate the buffered
        // bytes either: the checkpoint is byte-identical to the pre-crash
        // snapshot, so it can never falsely validate a longer pack.
        assert_eq!(
            std::fs::read(checkpoint_path(&dir)).unwrap(),
            checkpoint_before,
            "crash with an unflushed buffered record must not rewrite the checkpoint"
        );

        // The fingerprint gate on the reopened store therefore still sits on
        // exactly the committed reality: clean open, synced records present,
        // ghost absent — no mismatch, no false pass, nothing resurrected.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_all_records(&reopened);
        assert!(
            reopened.get(&cid, &ghost).unwrap().is_none(),
            "a never-flushed buffered record must not exist after a fresh open"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn deleted_collections_are_not_resurrected_by_stale_checkpoint() {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_index_checkpoint_del_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        let doomed = collection_id(2);
        store.delete_collection(&doomed).unwrap();
        // Deliberately NOT synced again: the on-disk checkpoint still lists the
        // deleted collection. The logical-delete set must win over the stale
        // checkpoint on reopen.
        drop(store);

        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let ids = reopened.collection_ids();
        assert_eq!(ids.len(), 2, "deleted collection must not come back");
        assert!(
            !ids.contains(&doomed),
            "deleted collection must not come back"
        );
        assert!(
            reopened
                .get(&doomed, &node_id(2, 0))
                .expect("lookup of a deleted collection is an error-free miss")
                .is_none(),
            "deleted collection's records must not be served"
        );
        // The surviving collections are fully intact.
        for c in 0..2u8 {
            assert!(reopened
                .get(&collection_id(c), &node_id(u64::from(c), 0))
                .expect("surviving collection lookup succeeds")
                .is_some());
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn clean_sync_skips_checkpoint_rewrite_but_a_write_rewrites() {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_dirty_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        let first_mtime = std::fs::metadata(checkpoint_path(&dir))
            .unwrap()
            .modified()
            .unwrap();

        // A second sync with no writes between must not rewrite the checkpoint.
        store.sync_all().unwrap();
        let second_mtime = std::fs::metadata(checkpoint_path(&dir))
            .unwrap()
            .modified()
            .unwrap();
        let first_key = duration_as_nanos(first_mtime, second_mtime);
        assert_eq!(first_key, 0, "a clean sync must not rewrite the checkpoint");
        drop(store);

        // A write followed by sync must rewrite it.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let cid = collection_id(1);
        let nid = node_id(1, 777);
        store
            .put(&cid, &nid, &NodeData::new(Bytes::from(payload_for(1, 777))))
            .unwrap();
        store.sync_all().unwrap();
        let third_mtime = std::fs::metadata(checkpoint_path(&dir))
            .unwrap()
            .modified()
            .unwrap();
        assert_ne!(
            duration_as_nanos(second_mtime, third_mtime),
            0,
            "a write followed by sync must rewrite the checkpoint"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn duration_as_nanos(from: std::time::SystemTime, to: std::time::SystemTime) -> u128 {
        to.duration_since(from)
            .unwrap_or_else(|e| e.duration())
            .as_nanos()
    }
}
