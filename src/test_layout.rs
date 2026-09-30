use super::{DatabaseLayout, ShardType, DB_META_FILENAME};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static CLEANUP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct CleanupDir(PathBuf);

impl CleanupDir {
    fn new(name: &str) -> Self {
        let id = CLEANUP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("mtxdb-layout-{name}-{}-{id}", std::process::id())))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn test_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("mtxdb-layout-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    path
}

/// A root written before the pool directories were renamed lists the old
/// names. It must be rejected loudly, not opened with its data orphaned in
/// `pools/state` etc. beside freshly created, empty `pools/mtpl-*`.
#[test]
fn a_descriptor_listing_the_old_pool_names_is_rejected() {
    let root = test_dir("old_pool_names");
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    drop(layout);
    let meta_path = root.join(DB_META_FILENAME);
    let mut contents = fs::read(&meta_path).unwrap();
    let header_len = contents.len() - super::db_meta_pool_list().len();
    contents.truncate(header_len);
    contents.extend_from_slice(b"state\nevent\nedges\n");
    fs::write(&meta_path, contents).unwrap();

    assert_eq!(
        super::is_database_root(&root).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    assert!(DatabaseLayout::open(root.clone()).is_err());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn pool_seeds_are_nonzero_distinct_and_stable_across_reopens() {
    let root = test_dir("pool_seeds");
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    let seeds: Vec<u64> = ShardType::ALL
        .iter()
        .map(|t| layout.pool_seed(*t))
        .collect();
    assert!(seeds.iter().all(|seed| *seed != 0));
    let mut unique = seeds.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), seeds.len(), "each pool gets its own seed");

    let reopened = DatabaseLayout::open(root.clone()).unwrap();
    let read_only = DatabaseLayout::open_read_only(root.clone()).unwrap();
    for shard_type in ShardType::ALL {
        assert_eq!(layout.pool_seed(shard_type), reopened.pool_seed(shard_type));
        assert_eq!(
            layout.pool_seed(shard_type),
            read_only.pool_seed(shard_type)
        );
        let dir = layout.pool_dir(shard_type).unwrap();
        assert_eq!(
            super::enclosing_pool_seed(&dir).unwrap(),
            Some(layout.pool_seed(shard_type))
        );
    }

    let other = test_dir("pool_seeds_other");
    let other_layout = DatabaseLayout::open(other.clone()).unwrap();
    assert_ne!(
        layout.pool_seed(ShardType::State),
        other_layout.pool_seed(ShardType::State),
        "two roots must not share a seed"
    );
    assert_eq!(
        super::enclosing_pool_seed(&root.join("not-a-pool")).unwrap(),
        None
    );
    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(other);
}

#[test]
fn a_zero_seed_or_old_version_descriptor_is_rejected() {
    let root = test_dir("bad_descriptor");
    drop(DatabaseLayout::open(root.clone()).unwrap());
    let meta_path = root.join(DB_META_FILENAME);
    let good = fs::read(&meta_path).unwrap();

    let mut zero = good.clone();
    zero[5..13].fill(0);
    fs::write(&meta_path, zero).unwrap();
    assert!(DatabaseLayout::open(root.clone()).is_err());

    let mut old = good;
    old[4] = 2;
    fs::write(&meta_path, old).unwrap();
    assert!(DatabaseLayout::open_read_only(root.clone()).is_err());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn initializes_named_pool_layout() {
    let root = test_dir("initialize");
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    assert!(root.join(DB_META_FILENAME).is_file());
    for shard_type in ShardType::ALL {
        assert!(
            !root.join("pools").join(shard_type.as_str()).exists(),
            "init creates only the root; pools appear on first write"
        );
        assert_eq!(
            layout.pool_path(shard_type),
            root.join("pools").join(shard_type.as_str())
        );
    }
    assert_eq!(
        layout.pool_dir(ShardType::State).unwrap(),
        root.join("pools/mtpl-state")
    );
    assert_eq!(
        layout.pool_dir(ShardType::EventDag).unwrap(),
        root.join("pools/mtpl-event")
    );
    assert_eq!(
        layout.pool_dir(ShardType::Edges).unwrap(),
        root.join("pools/mtpl-edges")
    );
}

#[test]
fn concurrent_first_open_installs_one_complete_descriptor() {
    let root = CleanupDir::new("concurrent_first_open");
    let root_path = root.path().to_path_buf();
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let root = root_path.clone();
            std::thread::spawn(move || DatabaseLayout::open(root).unwrap())
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }

    assert!(super::parse_db_meta(&fs::read(root.path().join(DB_META_FILENAME)).unwrap()).is_some());
    assert!(fs::read_dir(root.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".db.meta.create.")
    }));
}

#[test]
fn descriptor_install_falls_back_to_rename_when_hard_links_are_unavailable() {
    let root = CleanupDir::new("hard_link_fallback");
    fs::create_dir_all(root.path()).unwrap();
    let temporary = root.path().join(".db.meta.create.test");
    let descriptor = root.path().join(DB_META_FILENAME);
    fs::write(&temporary, super::db_meta_bytes(0x1234_5678_9abc_def1)).unwrap();

    super::DatabaseLayout::install_descriptor_temp(&temporary, &descriptor, |_, _| {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "hard links unavailable in test",
        ))
    })
    .unwrap();

    assert!(super::parse_db_meta(&fs::read(&descriptor).unwrap()).is_some());
    assert!(!temporary.exists());
}

#[test]
fn descriptor_install_accepts_a_temp_swept_by_a_concurrent_opener() {
    let root = CleanupDir::new("swept_temp_race");
    fs::create_dir_all(root.path()).unwrap();
    let temporary = root.path().join(".db.meta.create.racing.1");
    let descriptor = root.path().join(DB_META_FILENAME);
    fs::write(&temporary, super::db_meta_bytes(0x1234_5678_9abc_def1)).unwrap();

    super::DatabaseLayout::install_descriptor_temp(&temporary, &descriptor, |temp, target| {
        fs::remove_file(temp)?;
        fs::write(target, super::db_meta_bytes(0x1234_5678_9abc_def1))?;
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "concurrent opener swept the temp after install",
        ))
    })
    .unwrap();

    assert!(super::parse_db_meta(&fs::read(&descriptor).unwrap()).is_some());
}

#[test]
fn first_open_reaps_orphaned_descriptor_temps() {
    let root = CleanupDir::new("orphaned_descriptor_temps");
    fs::create_dir_all(root.path()).unwrap();
    let orphan = root.path().join(".db.meta.create.dead.1");
    fs::write(&orphan, b"interrupted descriptor").unwrap();

    DatabaseLayout::open(root.path().to_path_buf()).unwrap();

    assert!(root.path().join(DB_META_FILENAME).is_file());
    assert!(!orphan.exists());
}

#[test]
fn refuses_legacy_flat_packfiles() {
    let root = test_dir("legacy");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("shard_0000_0000000000000000.pack"), b"legacy").unwrap();
    let err = DatabaseLayout::open(root).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("legacy flat"));
}

#[test]
fn read_only_open_never_initializes_a_root() {
    let root = test_dir("read_only_missing");
    let err = DatabaseLayout::open_read_only(root.clone()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(!root.exists());
}

#[test]
fn rejects_a_descriptor_with_an_unknown_version() {
    let root = test_dir("bad_version");
    fs::create_dir_all(&root).unwrap();
    let mut bytes = super::db_meta_bytes(0x1234_5678_9abc_def1);
    bytes[4] = super::DB_META_VERSION.wrapping_add(1);
    fs::write(root.join(DB_META_FILENAME), &bytes).unwrap();
    let err = DatabaseLayout::open(root).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("unrecognized"));
}

#[test]
fn rejects_a_truncated_descriptor() {
    let root = test_dir("truncated_descriptor");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join(DB_META_FILENAME), b"MTXD").unwrap();
    let err = DatabaseLayout::open(root).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn reopening_an_existing_root_round_trips_the_descriptor() {
    let root = test_dir("reopen_descriptor");
    DatabaseLayout::open(root.clone()).unwrap();
    // A second open of the same root must accept the descriptor it just
    // wrote — this is the ordinary "reopen an existing database" path.
    DatabaseLayout::open(root.clone()).unwrap();
    DatabaseLayout::open_read_only(root).unwrap();
}

#[test]
fn read_only_open_rejects_a_missing_descriptor_without_creating_one() {
    let root = test_dir("read_only_no_descriptor");
    fs::create_dir_all(&root).unwrap();
    let err = DatabaseLayout::open_read_only(root.clone()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(err.to_string().contains("missing database descriptor"));
    // A failed read-only open must not leave a descriptor behind — this is
    // the `mtxdb collections` path and it must never mutate the store.
    assert!(!root.join(DB_META_FILENAME).exists());
}

#[test]
fn read_only_open_rejects_a_corrupt_descriptor() {
    let root = test_dir("read_only_corrupt_descriptor");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join(DB_META_FILENAME), b"garbage, not a descriptor").unwrap();
    let err = DatabaseLayout::open_read_only(root).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("unrecognized"));
}

#[test]
fn read_only_open_rejects_a_legacy_flat_store() {
    let root = test_dir("read_only_legacy");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join(DB_META_FILENAME),
        super::db_meta_bytes(0x1234_5678_9abc_def1),
    )
    .unwrap();
    fs::write(root.join("shard_0000_0000000000000000.pack"), b"legacy").unwrap();
    let err = DatabaseLayout::open_read_only(root).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("legacy flat"));
}

#[test]
fn read_only_open_surfaces_a_read_error_without_panicking() {
    let root = test_dir("read_only_descriptor_is_dir");
    // A `db.meta` that is a directory makes `fs::read` fail; the error must
    // propagate as `Err`, never panic or abort.
    fs::create_dir_all(root.join(DB_META_FILENAME)).unwrap();
    assert!(DatabaseLayout::open_read_only(root).is_err());
}

#[test]
fn a_root_keeps_its_wal_at_the_root() {
    let root = test_dir("wal_at_root");
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    assert_eq!(
        layout.shared_wal_path(),
        root.join("wal.bin"),
        "every root's WAL lives at the root, never in a pool directory"
    );
}

/// A version 1 descriptor carried a WAL-layout byte that no longer exists.
/// It is rejected outright, so a root written that way is never opened and
/// reinterpreted.
#[test]
fn a_version_one_descriptor_is_rejected() {
    let root = test_dir("version_one");
    fs::create_dir_all(&root).unwrap();
    let mut bytes = super::db_meta_bytes(0x1234_5678_9abc_def1);
    bytes[4] = 1;
    bytes[5] = 1; // the old WAL-layout byte
    fs::write(root.join(DB_META_FILENAME), &bytes).unwrap();
    assert_eq!(
        DatabaseLayout::open(root.clone()).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    assert_eq!(
        super::is_database_root(&root).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[test]
fn enclosing_root_finds_a_shared_root_from_a_pool_directory() {
    let root = test_dir("enclosing_root");
    let layout = DatabaseLayout::open(root.clone()).unwrap();
    let pool = layout.pool_dir(ShardType::State).unwrap();
    let found = super::enclosing_root(&pool).unwrap().unwrap();
    assert_eq!(found, root);

    let standalone = test_dir("enclosing_root_none");
    fs::create_dir_all(&standalone).unwrap();
    assert!(super::enclosing_root(&standalone).unwrap().is_none());
}
