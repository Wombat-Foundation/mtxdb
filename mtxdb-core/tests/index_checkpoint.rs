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

    /// Locate the current delta-log epoch file, if any. The on-disk name is
    /// now `index.delta.<hex fingerprint>` (see `PackfileStorage::delta_path`)
    /// rather than a fixed name, so tests that need to see the file directly
    /// have to search for it rather than joining a constant.
    fn current_delta_path(dir: &Path) -> Option<PathBuf> {
        use mtxdb_core::index::delta::INDEX_DELTA_FILE;
        let prefix = format!("{INDEX_DELTA_FILE}.");
        std::fs::read_dir(dir).ok()?.find_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.starts_with(&prefix).then_some(path)
        })
    }

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
        drop(read_only);

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

        // The writable store holds the directory's writer lock (a second
        // writer on the same dir is refused), so release it before reopening.
        drop(reopened);

        // Stale fingerprint: a checkpoint that decodes cleanly but no longer
        // matches the packs must fall back to the full scan — and record the
        // fruitless fingerprint attempt it made (per the struct doc).
        std::fs::write(checkpoint_path(&dir), {
            let mut bytes = std::fs::read(checkpoint_path(&dir)).unwrap();
            bytes[32] ^= 0xFF;
            bytes
        })
        .unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let open = reopened
            .open_timings()
            .expect("open must record its phase timings");
        assert_eq!(
            open.path,
            OpenPath::FullScan,
            "stale fingerprint must rescan"
        );
        assert!(
            open.full_scan > std::time::Duration::ZERO,
            "the fallback must actually scan"
        );
        assert!(
            open.fingerprint > std::time::Duration::ZERO,
            "the stale-fingerprint attempt must be attributed, not dropped"
        );
        assert!(
            open.fingerprint <= open.total,
            "fingerprint must fit inside the open total"
        );
        assert_all_records(&reopened);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shard_directory_bookkeeping_gate_and_fallback() {
        use mtxdb_core::packfile::storage::{BookkeepingSource, OpenPath};

        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_sidecar_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        drop(store);

        let sidecar = dir.join("shard_collections.bin");
        let header_fingerprint_offset = 5usize;
        let last_assert = |reopened: &PackfileStorage, source: BookkeepingSource| {
            let open = reopened
                .open_timings()
                .expect("open must record its phase timings");
            assert_eq!(
                open.path,
                OpenPath::Checkpoint,
                "every case here keeps a valid checkpoint; only bookkeeping source changes"
            );
            assert_eq!(
                open.bookkeeping_source, source,
                "bookkeeping must come from the expected source"
            );
            assert_all_records(reopened);
        };

        // Happy path: a valid fingerprint-gated directory serves the open's
        // per-shard bookkeeping without walking any slots.
        assert!(
            sidecar.exists(),
            "sync_all must persist the shard→collection directory"
        );
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        last_assert(&reopened, BookkeepingSource::Sidecar);
        drop(reopened);

        // Corrupt: garbage bytes → the slot-walk fallback, still correct.
        std::fs::write(&sidecar, vec![0xAB; 256]).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        last_assert(&reopened, BookkeepingSource::SlotScan);
        drop(reopened);

        // Stale fingerprint: regenerate a valid directory (write + sync), then
        // flip one byte inside its fingerprint header so it no longer matches
        // the checkpoint's → fall back, still correct.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let cid = collection_id(1);
        let extra = node_id(1, 555);
        store
            .put(
                &cid,
                &extra,
                &NodeData::new(Bytes::from(payload_for(1, 555))),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);
        std::fs::write(&sidecar, {
            let mut bytes = std::fs::read(&sidecar).unwrap();
            bytes[header_fingerprint_offset] ^= 0xFF;
            bytes
        })
        .unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        last_assert(&reopened, BookkeepingSource::SlotScan);
        drop(reopened);

        // Missing: removed entirely → same fallback, still correct.
        std::fs::remove_file(&sidecar).unwrap();
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        last_assert(&reopened, BookkeepingSource::SlotScan);
        drop(reopened);

        // The gate also recovers: one more write + sync regenerates a valid
        // directory and the fast path returns.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let extra = node_id(2, 555);
        store
            .put(
                &collection_id(2),
                &extra,
                &NodeData::new(Bytes::from(payload_for(2, 555))),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        last_assert(&reopened, BookkeepingSource::Sidecar);

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
    #[allow(clippy::too_many_lines)]
    fn clean_sync_writes_nothing_but_a_write_appends_delta_and_structural_change_rewrites() {
        use mtxdb_core::packfile::storage::OpenPath;

        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_dirty_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        let first_checkpoint = std::fs::read(checkpoint_path(&dir)).unwrap();
        assert!(
            current_delta_path(&dir).is_none(),
            "the initial sync rewrites and re-bases, leaving no delta log"
        );

        // A second sync with no writes between writes nothing at all: neither
        // the checkpoint nor a delta log.
        store.sync_all().unwrap();
        assert_eq!(
            std::fs::read(checkpoint_path(&dir)).unwrap(),
            first_checkpoint,
            "a clean sync must not rewrite the checkpoint"
        );
        assert!(
            current_delta_path(&dir).is_none(),
            "a clean sync must not create a delta log"
        );
        drop(store);

        // A plain write's sync appends one delta batch and must NOT rewrite the
        // checkpoint — that is the entire point of the delta writer.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let cid = collection_id(1);
        let nid = node_id(1, 777);
        store
            .put(&cid, &nid, &NodeData::new(Bytes::from(payload_for(1, 777))))
            .unwrap();
        store.sync_all().unwrap();
        let sync = store
            .sync_timings()
            .expect("sync must record its phase timings");
        assert!(
            sync.delta_log > std::time::Duration::ZERO,
            "a plain write's sync must persist via a delta append"
        );
        assert_eq!(
            sync.checkpoint,
            std::time::Duration::ZERO,
            "a plain write's sync must not rewrite the checkpoint"
        );
        assert_eq!(
            std::fs::read(checkpoint_path(&dir)).unwrap(),
            first_checkpoint,
            "a delta append leaves the checkpoint byte-identical"
        );
        assert!(
            current_delta_path(&dir).is_some(),
            "a plain write's sync must leave a delta log"
        );
        drop(store);

        // The append replays on the fast reopen path and serves the record.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let open = reopened
            .open_timings()
            .expect("open must record its phase timings");
        assert_eq!(
            open.path,
            OpenPath::Checkpoint,
            "a delta log must replay onto its checkpoint, not rescan"
        );
        assert!(
            open.delta_replay > std::time::Duration::ZERO,
            "reopening a delta-extended store must pay the replay"
        );
        let got = reopened
            .get(&cid, &nid)
            .expect("lookup succeeds")
            .expect("appended record must be served");
        assert_eq!(got.bytes.as_ref(), payload_for(1, 777).as_slice());
        drop(reopened);

        // A structural change (a delete invalidates the log mid-session) must
        // force the next dirty sync into a full checkpoint rewrite, which
        // re-bases and truncates the delta log.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &cid,
                &node_id(1, 778),
                &NodeData::new(Bytes::from(payload_for(1, 778))),
            )
            .unwrap();
        store.delete_collection(&collection_id(2)).unwrap();
        store.sync_all().unwrap();
        let sync = store
            .sync_timings()
            .expect("sync must record its phase timings");
        assert!(
            sync.checkpoint > std::time::Duration::ZERO,
            "a structural change must eventually rewrite the checkpoint"
        );
        assert_eq!(
            sync.delta_log,
            std::time::Duration::ZERO,
            "a structural change must not try to append"
        );
        assert!(
            current_delta_path(&dir).is_none(),
            "the rewrite re-bases the log and truncates its file"
        );
        assert_ne!(
            std::fs::read(checkpoint_path(&dir)).unwrap(),
            first_checkpoint,
            "the structural rewrite must change the checkpoint"
        );
        drop(store);

        // The rewritten checkpoint serves both writes and keeps the deletion.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        assert_eq!(
            reopened.collection_ids().len(),
            2,
            "the deleted collection must not come back"
        );
        for (nid, i) in [(node_id(1, 777), 777), (node_id(1, 778), 778)] {
            let got = reopened
                .get(&cid, &nid)
                .expect("lookup succeeds")
                .expect("record survives the structural rewrite");
            assert_eq!(got.bytes.as_ref(), payload_for(1, i).as_slice());
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression for the inherited-log `write_header` bug: a session that opens
    /// onto an existing committed delta log must CONTINUE it (one batch per
    /// sync, no second base header). The old logic derived "fresh file?" from
    /// a session-default `log_bytes == 0`, so a reopened session's first append
    /// re-wrote the 16-byte log header MID-FILE; the reader then stopped at the
    /// first batch magic and stranded the first batch's tail fingerprint,
    /// rejecting the whole log on the next open.
    #[test]
    fn reopened_session_continues_an_inherited_delta_log() {
        use mtxdb_core::index::delta;
        use mtxdb_core::packfile::storage::OpenPath;

        let dir = std::env::temp_dir().join(format!(
            "mtxdb_index_checkpoint_delta_continue_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let store = make_store(&dir);
        store.sync_all().unwrap();
        drop(store);

        // Session 1: one write, one append batch.
        let cid = collection_id(1);
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &cid,
                &node_id(1, 900),
                &NodeData::new(Bytes::from(payload_for(1, 900))),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);
        let delta_path =
            current_delta_path(&dir).expect("session 1's append must leave a delta log epoch");
        assert_eq!(
            delta::read_delta_log(&delta_path)
                .expect("session 1 must leave a decodable log")
                .frames
                .len(),
            1,
            "session 1 appends exactly one frame"
        );

        // Session 2 inherits the committed log; its append must continue the
        // file, not re-write the base header.
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &cid,
                &node_id(1, 901),
                &NodeData::new(Bytes::from(payload_for(1, 901))),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        let log = delta::read_delta_log(&delta_path)
            .expect("both appends must decode as one continued log");
        assert_eq!(
            log.frames.len(),
            2,
            "the reopened session's append must decode as a second batch"
        );

        // The tail is whatever the packs actually are now — the gate check
        // below proves it (Checkpoint path requires tail == local fingerprint),
        // and both records must be served.
        let reopened = PackfileStorage::open(dir.clone()).unwrap();
        let open = reopened
            .open_timings()
            .expect("open must record its phase timings");
        assert_eq!(
            open.path,
            OpenPath::Checkpoint,
            "a two-batch continued log must replay cleanly (tail == local fingerprint)"
        );
        assert!(open.delta_replay > std::time::Duration::ZERO);
        for (nid, i) in [(node_id(1, 900), 900), (node_id(1, 901), 901)] {
            let got = reopened
                .get(&cid, &nid)
                .expect("lookup succeeds")
                .expect("record survives the continued log");
            assert_eq!(got.bytes.as_ref(), payload_for(1, i).as_slice());
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Subprocess reader/writer tests for multi-process durability.
#[cfg(test)]
mod subprocess_tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use mtxdb_core::storage::{NodeData, NodeId, StorageEngine};
    use mtxdb_core::PackfileStorage;

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Initialize a test database with proper structure (db.meta + pools/).
    /// Returns (`database_root`, `pool_dir`).
    fn setup_test_db(name: &str) -> (PathBuf, PathBuf) {
        let db_root = std::env::temp_dir().join(format!(
            "mtxdb_subprocess_{}_{}_{}",
            name,
            std::process::id(),
            TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&db_root);
        std::fs::create_dir_all(&db_root).unwrap();

        // Run mtxdb init to create db.meta and pools/ directories
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("init")
            .output()
            .expect("mtxdb init must run");
        assert!(
            output.status.success(),
            "mtxdb init failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Pool directory for PackfileStorage operations
        let pool_dir = db_root.join("pools").join("event-dag");
        (db_root, pool_dir)
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb_subprocess_{}_{}_{}",
            name,
            std::process::id(),
            TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn mtxdb_cli() -> PathBuf {
        // Build the CLI binary via cargo if needed, then use it directly
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or(".".to_owned());
        // The mtxdb-cli binary is built in the workspace target dir, not the package target dir
        // CARGO_MANIFEST_DIR is mtxdb-core, so workspace root is parent dir
        let workspace_root = std::path::Path::new(&manifest_dir).parent().unwrap();
        let target_dir = std::env::var("CARGO_TARGET_DIR")
            .unwrap_or_else(|_| format!("{}/target", workspace_root.display()));
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_owned());

        let path = std::path::Path::new(&target_dir)
            .join(&profile)
            .join("mtxdb");
        eprintln!("DEBUG: mtxdb_cli: manifest_dir={}, workspace_root={}, target_dir={}, profile={}, path={:?}", manifest_dir, workspace_root.display(), target_dir, profile, path);
        if path.exists() {
            return path;
        }

        // Fallback: build with cargo and use the built binary
        let output = std::process::Command::new("cargo")
            .arg("build")
            .arg("--package")
            .arg("mtxdb-cli")
            .current_dir(workspace_root)
            .output()
            .expect("cargo build mtxdb-cli must succeed");
        assert!(
            output.status.success(),
            "cargo build mtxdb-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        std::path::Path::new(&target_dir)
            .join(&profile)
            .join("mtxdb")
    }

    const SUBPROCESS_COLLECTION: NodeId = [0x01; 16];
    const SUBPROCESS_RECORD: NodeId = [0xD0; 16];
    const SUBPROCESS_SEED: NodeId = [0xC0; 16];

    #[test]
    fn unsynced_append_is_visible_but_not_durable() {
        let dir = test_dir("unsynced_visible");

        // Seed data
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &SUBPROCESS_COLLECTION,
                &SUBPROCESS_SEED,
                &NodeData::new(Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: append WITHOUT sync (using default Eager append policy)
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&dir)
            .arg("subprocess-writer-unsynced")
            .arg(dir.to_str().unwrap())
            .output()
            .expect("subprocess writer must run");
        assert!(
            output.status.success(),
            "writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // With Eager append policy (default), unsynced writes are immediately
        // written to pack files and visible to other processes, but not durable
        // (would be lost on crash before sync).
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(
            result[0].is_some(),
            "unsynced record should be visible with Eager append policy"
        );
        assert_eq!(result[0].as_ref().unwrap().bytes.as_ref(), b"synced");

        // Original seed still visible
        let result = store.get(&SUBPROCESS_COLLECTION, &SUBPROCESS_SEED).unwrap();
        assert!(result.is_some(), "seed must survive");

        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn synced_append_is_recovered() {
        let dir = test_dir("synced_recovered");

        // Seed data
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &SUBPROCESS_COLLECTION,
                &SUBPROCESS_SEED,
                &NodeData::new(Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: append WITH sync
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&dir)
            .arg("subprocess-writer")
            .arg(dir.to_str().unwrap())
            .output()
            .expect("subprocess writer must run");
        assert!(
            output.status.success(),
            "writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see the synced record via refresh
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(
            result[0].is_some(),
            "synced record must be recovered after refresh"
        );
        assert_eq!(result[0].as_ref().unwrap().bytes.as_ref(), b"synced");

        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn concurrent_misses_perform_one_refresh() {
        let (db_root, pool_dir) = setup_test_db("concurrent_misses");

        // Seed data
        let store = PackfileStorage::open(pool_dir.clone()).unwrap();
        store
            .put(
                &SUBPROCESS_COLLECTION,
                &SUBPROCESS_SEED,
                &NodeData::new(Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: append and sync
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("subprocess-writer")
            .arg(pool_dir.to_str().unwrap())
            .output()
            .expect("subprocess writer must run");
        assert!(
            output.status.success(),
            "writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Multiple reader processes: each should trigger at most one refresh
        for _ in 0..3 {
            let output = Command::new(&cli)
                .arg("--dir")
                .arg(&db_root)
                .arg("subprocess-reader")
                .arg(pool_dir.to_str().unwrap())
                .output()
                .expect("subprocess reader must run");
            assert!(
                output.status.success(),
                "reader failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains("RECOVERED"),
                "reader must recover: {stdout}"
            );
            // miss_refreshes should be exactly 1 (one refresh per process)
            assert!(
                stdout.contains("miss_refreshes=1"),
                "exactly one refresh: {stdout}"
            );
        }

        std::fs::remove_dir_all(&db_root).unwrap();
    }

    #[test]
    fn replacement_mutation() {
        let (db_root, pool_dir) = setup_test_db("replacement");

        // Seed data
        let store = PackfileStorage::open(pool_dir.clone()).unwrap();
        store
            .put(
                &SUBPROCESS_COLLECTION,
                &SUBPROCESS_SEED,
                &NodeData::new(Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: replace the record (same key, new value) and sync
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("put")
            .arg("-r")
            .arg(hex::encode(SUBPROCESS_COLLECTION))
            .arg("-i")
            .arg(hex::encode(SUBPROCESS_RECORD))
            .arg("-a")
            .arg("replaced")
            .output()
            .expect("subprocess writer must run");
        assert!(
            output.status.success(),
            "writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see replaced value
        let store = PackfileStorage::open_read_only(pool_dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(result[0].is_some(), "replaced record must be visible");
        assert_eq!(result[0].as_ref().unwrap().bytes.as_ref(), b"replaced");

        drop(store);
        std::fs::remove_dir_all(&db_root).unwrap();
    }

    #[test]
    fn truncation_mutation() {
        let (db_root, pool_dir) = setup_test_db("truncation");

        // Seed data with 10 records
        let store = PackfileStorage::open(pool_dir.clone()).unwrap();
        let mut entries = Vec::new();
        for i in 0..10u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            let data = NodeData::new(Bytes::from(format!("value-{i}")));
            entries.push((id, data));
        }
        store.put_many(&SUBPROCESS_COLLECTION, &entries).unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: delete half the records (simulating truncation)
        let cli = mtxdb_cli();
        for i in 5..10u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            let output = Command::new(&cli)
                .arg("--dir")
                .arg(&db_root)
                .arg("put")
                .arg("-r")
                .arg(hex::encode(SUBPROCESS_COLLECTION))
                .arg("-i")
                .arg(hex::encode(id))
                .arg("-a")
                .arg("")
                .output()
                .expect("delete must run");
            assert!(
                output.status.success(),
                "delete failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see remaining records, not deleted ones
        let store = PackfileStorage::open_read_only(pool_dir.clone()).unwrap();
        let mut keys = Vec::new();
        for i in 0..10u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            keys.push(id);
        }
        let results = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &keys)
            .unwrap();

        for i in 0..5 {
            assert!(results[i].is_some(), "record {i} must survive truncation");
            assert_eq!(
                results[i].as_ref().unwrap().bytes.as_ref(),
                format!("value-{i}").as_bytes()
            );
        }
        for i in 5..10 {
            assert!(
                results[i].is_none(),
                "record {i} must be deleted after truncation"
            );
        }

        drop(store);
        std::fs::remove_dir_all(&db_root).unwrap();
    }

    #[test]
    fn checkpoint_recovery() {
        let dir = test_dir("checkpoint_recovery");

        // Seed data
        let store = PackfileStorage::open(dir.clone()).unwrap();
        store
            .put(
                &SUBPROCESS_COLLECTION,
                &SUBPROCESS_SEED,
                &NodeData::new(Bytes::from_static(b"seed")),
            )
            .unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: append and sync (creates checkpoint + delta)
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&dir)
            .arg("subprocess-writer")
            .arg(dir.to_str().unwrap())
            .output()
            .expect("subprocess writer must run");
        assert!(
            output.status.success(),
            "writer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // New writer process: must recover from checkpoint
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(result[0].is_some(), "checkpoint recovery must work");
        assert_eq!(result[0].as_ref().unwrap().bytes.as_ref(), b"synced");

        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shard_mutation_recovery() {
        let (db_root, pool_dir) = setup_test_db("shard_mutation");

        // Seed data with enough records to force pack rotation
        let store =
            PackfileStorage::open_with_max_shard_bytes(pool_dir.clone(), 1024 * 1024).unwrap(); // 1MB

        let mut entries = Vec::new();
        for i in 0..500u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            let data = NodeData::new(Bytes::from(format!("value-{i}")));
            entries.push((id, data));
        }
        store.put_many(&SUBPROCESS_COLLECTION, &entries).unwrap();
        store.sync_all().unwrap();
        drop(store);

        // Writer: append more records (may cause rotation) and sync
        let cli = mtxdb_cli();
        let mut entries = Vec::new();
        for i in 500..1000u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            let data = NodeData::new(Bytes::from(format!("value-{i}")));
            entries.push((id, data));
        }

        // Use the CLI to add records
        for (id, data) in &entries {
            let output = Command::new(&cli)
                .arg("--dir")
                .arg(&db_root)
                .arg("put")
                .arg("-r")
                .arg(hex::encode(SUBPROCESS_COLLECTION))
                .arg("-i")
                .arg(hex::encode(id))
                .arg("-a")
                .arg(String::from_utf8_lossy(data.bytes.as_ref()).to_string())
                .output()
                .expect("put must run");
            assert!(
                output.status.success(),
                "put failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&db_root)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see all records after refresh
        let store = PackfileStorage::open_read_only(pool_dir.clone()).unwrap();
        let mut keys = Vec::new();
        for i in 0..1000u64 {
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&i.to_le_bytes());
            keys.push(id);
        }
        let results = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &keys)
            .unwrap();

        for i in 0..1000 {
            assert!(
                results[i].is_some(),
                "record {i} must survive shard rotation"
            );
            assert_eq!(
                results[i].as_ref().unwrap().bytes.as_ref(),
                format!("value-{i}").as_bytes()
            );
        }

        drop(store);
        std::fs::remove_dir_all(&db_root).unwrap();
    }
}
