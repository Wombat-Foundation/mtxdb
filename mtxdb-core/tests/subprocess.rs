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
        let mut path = std::env::current_exe().unwrap();
        path.pop(); // deps/
        path.pop(); // debug/ or release/
        path.push("mtxdb");
        if path.exists() {
            return path;
        }
        // Fallback: build with cargo and use the built binary
        let output = std::process::Command::new("cargo")
            .arg("build")
            .arg("--package")
            .arg("mtxdb-cli")
            .current_dir(std::env::var("CARGO_MANIFEST_DIR").unwrap_or(".".to_owned()))
            .output()
            .expect("cargo build mtxdb-cli must succeed");
        assert!(
            output.status.success(),
            "cargo build mtxdb-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        path.pop(); // remove "mtxdb"
        path.push("mtxdb");
        path
    }

    const SUBPROCESS_COLLECTION: NodeId = [0x01; 16];
    const SUBPROCESS_RECORD: NodeId = [0xD0; 16];
    const SUBPROCESS_SEED: NodeId = [0xC0; 16];

    #[test]
    fn unsynced_append_stays_invisible() {
        let dir = test_dir("unsynced_invisible");

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

        // Writer: append WITHOUT sync
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

        // Reader: must NOT see the unsynced record
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(
            result[0].is_none(),
            "unsynced record must be invisible to read-only opener"
        );

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
        let dir = test_dir("concurrent_misses");

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

        // Writer: append and sync
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

        // Multiple reader processes: each should trigger at most one refresh
        for _ in 0..3 {
            let output = Command::new(&cli)
                .arg("--dir")
                .arg(&dir)
                .arg("subprocess-reader")
                .arg(dir.to_str().unwrap())
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

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn replacement_mutation() {
        let dir = test_dir("replacement");

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

        // Writer: replace the record (same key, new value) and sync
        let cli = mtxdb_cli();
        let output = Command::new(&cli)
            .arg("--dir")
            .arg(&dir)
            .arg("put")
            .arg("-r")
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
            .arg(&dir)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see replaced value
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
        let result = store
            .get_many_with_refresh(&SUBPROCESS_COLLECTION, &[SUBPROCESS_RECORD])
            .unwrap();
        assert!(result[0].is_some(), "replaced record must be visible");
        assert_eq!(result[0].as_ref().unwrap().bytes.as_ref(), b"replaced");

        drop(store);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn truncation_mutation() {
        let dir = test_dir("truncation");

        // Seed data with 10 records
        let store = PackfileStorage::open(dir.clone()).unwrap();
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
                .arg(&dir)
                .arg("put")
                .arg("-r")
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
            .arg(&dir)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see remaining records, not deleted ones
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
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
        std::fs::remove_dir_all(&dir).unwrap();
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
        let dir = test_dir("shard_mutation");

        // Seed data with enough records to force pack rotation
        let store = PackfileStorage::open_with_max_shard_bytes(dir.clone(), 1024 * 1024).unwrap(); // 1MB

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
                .arg(&dir)
                .arg("put")
                .arg("-r")
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
            .arg(&dir)
            .arg("sync")
            .output()
            .expect("sync must run");
        assert!(
            output.status.success(),
            "sync failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        // Reader: must see all records after refresh
        let store = PackfileStorage::open_read_only(dir.clone()).unwrap();
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
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
