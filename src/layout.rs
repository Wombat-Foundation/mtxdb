//! Database-root layout and named packfile pools.
//!
//! A [`crate::packfile::storage::PackfileStorage`] owns exactly one pool;
//! it must be opened on one of this module's pool directories, never on the
//! database root. Keeping that boundary explicit gives state, event-DAG, and
//! edges data independent shard, GC, durability, and writer-lock
//! lifecycles.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Immutable descriptor at a database root.
///
/// Layout: `magic(4) | version(1) | reserved(8) | pool list`. The reserved
/// bytes are zero-filled and unused today; they exist so a future format
/// revision can add a field without another exact-byte-equality break, the
/// same way `pool.meta` carries a version byte. Not deployed anywhere yet,
/// so the descriptor is parsed (magic + version + pool list), not compared
/// byte-for-byte — a mismatched version is rejected with a specific error
/// rather than silently misparsed. Every root has one shared WAL at its root,
/// so the descriptor records no WAL layout; version 1 descriptors, which did,
/// are rejected. The shipped CLI, C ABI, and WASI entry
/// points validate this root and open their selected named pool through this
/// type; [`PackfileStorage`](crate::PackfileStorage) remains available as a
/// lower-level single-pool API.
const DB_META_MAGIC: &[u8; 4] = b"MTXD";
const DB_META_VERSION: u8 = 2;
const DB_META_RESERVED_LEN: usize = 8;
/// The pool list a descriptor must carry, built from the pool directory names
/// so it can never drift from them. A root whose descriptor lists other names
/// (an older layout) is rejected by [`validate_db_meta`] instead of being
/// opened with its data orphaned in directories nothing reads.
fn db_meta_pool_list() -> Vec<u8> {
    let mut list = Vec::new();
    for shard_type in ShardType::ALL {
        list.extend_from_slice(shard_type.as_str().as_bytes());
        list.push(b'\n');
    }
    list
}
/// File name of the database-root descriptor.
pub const DB_META_FILENAME: &str = "db.meta";

/// `magic + version` header length shared by the writer and the validator.
const DB_META_HEADER_LEN: usize = 4 + 1 + DB_META_RESERVED_LEN;
/// Build the on-disk descriptor bytes for a database root.
fn db_meta_bytes() -> Vec<u8> {
    let pool_list = db_meta_pool_list();
    let mut buf = Vec::with_capacity(DB_META_HEADER_LEN.saturating_add(pool_list.len()));
    buf.extend_from_slice(DB_META_MAGIC);
    buf.push(DB_META_VERSION);
    buf.extend_from_slice(&[0u8; DB_META_RESERVED_LEN]);
    buf.extend_from_slice(&pool_list);
    buf
}

/// Validate a descriptor read from disk: correct magic, a version this build
/// understands, and the expected pool list.
fn validate_db_meta(contents: &[u8]) -> bool {
    if contents.len() < DB_META_HEADER_LEN {
        return false;
    }
    if &contents[..4] != DB_META_MAGIC {
        return false;
    }
    if contents[4] != DB_META_VERSION {
        return false;
    }
    contents[DB_META_HEADER_LEN..] == db_meta_pool_list()[..]
}

/// Whether `root` is a database root: it has a valid `db.meta`. `Ok(false)`
/// when there is no descriptor at all.
///
/// # Errors
/// Returns `InvalidData` if a descriptor exists but is unrecognized, so a
/// caller can never mistake a corrupt or old-format root for a missing one.
pub fn is_database_root(root: &Path) -> io::Result<bool> {
    let meta_path = root.join(DB_META_FILENAME);
    let contents = match fs::read(&meta_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if !validate_db_meta(&contents) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unrecognized mtxdb database descriptor: {}",
                meta_path.display()
            ),
        ));
    }
    Ok(true)
}

/// Nearest enclosing database root for `dir`, if any.
///
/// Walks `dir` and its ancestors looking for a `db.meta` descriptor. This is
/// how a store opened on `root/pools/<pool>` discovers that it lives inside a
/// database root (see the per-pool journal gate in
/// [`PackfileStorage::enable_journal`](crate::PackfileStorage::enable_journal)).
///
/// # Errors
/// Returns `InvalidData` if a descriptor is found but unrecognized, so a
/// caller can never mistake a corrupt root for a standalone store.
pub fn enclosing_root(dir: &Path) -> io::Result<Option<PathBuf>> {
    for ancestor in dir.ancestors() {
        let meta_path = ancestor.join(DB_META_FILENAME);
        if !meta_path.is_file() {
            continue;
        }
        let contents = fs::read(&meta_path)?;
        if !validate_db_meta(&contents) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unrecognized mtxdb database descriptor: {}",
                    meta_path.display()
                ),
            ));
        }
        return Ok(Some(ancestor.to_path_buf()));
    }
    Ok(None)
}

/// A named independent packfile pool in an mtxdb database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardType {
    /// HAMT nodes, roots, and state-group sidecars.
    State,
    /// Event JSON plus collection-DAG-oriented event data.
    EventDag,
    /// Edges pool: houses previous-event edges (`PREV`) and auth-chain edges (`AUTH`).
    Edges,
}

impl ShardType {
    /// Every shard type defined by the current database layout.
    pub const ALL: [Self; 3] = [Self::State, Self::EventDag, Self::Edges];

    /// Stable on-disk directory name for this pool.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::State => "mtpl-state",
            Self::EventDag => "mtpl-event",
            Self::Edges => "mtpl-edges",
        }
    }

    /// Stable 4-byte physical pool tag used in physical layout and diagnostic labeling.
    ///
    /// # Note
    /// This is a physical storage pool tag (e.g. `EDGE`), **not** a logical
    /// member namespace (`PREV` / `AUTH`). Collection derivation MUST use logical
    /// member namespaces, never this physical pool tag.
    #[must_use]
    pub const fn physical_pool_tag(self) -> [u8; 4] {
        match self {
            Self::State => *b"STAT",
            Self::EventDag => *b"EVNT",
            Self::Edges => *b"EDGE",
        }
    }
}

/// Validated database root from which named pool paths can be derived.
#[derive(Debug, Clone)]
pub struct DatabaseLayout {
    root: PathBuf,
}

impl DatabaseLayout {
    /// Open or initialize a database root.
    ///
    /// # Errors
    /// Returns an error if the root cannot be created/read or its descriptor
    /// is unknown.
    pub fn open(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        Self::reject_legacy_flat_store(&root)?;
        let meta_path = root.join(DB_META_FILENAME);
        if meta_path.exists() {
            let contents = fs::read(&meta_path)?;
            if !validate_db_meta(&contents) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unrecognized mtxdb database descriptor: {}",
                        meta_path.display()
                    ),
                ));
            }
        } else {
            Self::write_descriptor(&meta_path)?;
        }
        for shard_type in ShardType::ALL {
            fs::create_dir_all(root.join("pools").join(shard_type.as_str()))?;
        }
        Ok(Self { root })
    }

    /// Return a named pool's directory, creating its parent directory.
    ///
    /// The pool itself is initialized by `PackfileStorage::open`.
    ///
    /// # Errors
    /// Returns an error if the pool parent cannot be created.
    pub fn pool_dir(&self, shard_type: ShardType) -> io::Result<PathBuf> {
        let path = self.root.join("pools").join(shard_type.as_str());
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Return a named pool's directory without creating it or any parent.
    ///
    /// # Errors
    /// Returns an error if the pool directory does not exist.
    pub fn pool_dir_read_only(&self, shard_type: ShardType) -> io::Result<PathBuf> {
        let path = self.root.join("pools").join(shard_type.as_str());
        if !path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("pool directory does not exist: {}", path.display()),
            ));
        }
        Ok(path)
    }

    /// Open a database root in read-only mode.
    ///
    /// Unlike [`Self::open`], this never creates directories or writes
    /// `db.meta`. It validates the root already exists and contains a valid
    /// descriptor.
    ///
    /// # Errors
    /// Returns an error if the root does not exist, has no descriptor, or
    /// contains an unrecognized one.
    pub fn open_read_only(root: PathBuf) -> io::Result<Self> {
        if !root.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("database root does not exist: {}", root.display()),
            ));
        }
        let meta_path = root.join(DB_META_FILENAME);
        if !meta_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing database descriptor: {}", meta_path.display()),
            ));
        }
        let contents = fs::read(&meta_path)?;
        if !validate_db_meta(&contents) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unrecognized mtxdb database descriptor: {}",
                    meta_path.display()
                ),
            ));
        }
        Self::reject_legacy_flat_store(&root)?;
        Ok(Self { root })
    }

    /// The database root this layout was opened from.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of the root-level shared WAL, `<root>/wal.bin`.
    ///
    /// Every root is driven through this one root-level segment, and no pool
    /// directory holds a WAL of its own.
    #[must_use]
    pub fn shared_wal_path(&self) -> PathBuf {
        self.root.join("wal.bin")
    }

    fn reject_legacy_flat_store(root: &Path) -> io::Result<()> {
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "pack")
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "legacy flat mtxdb store at {}; migrate it or use a fresh database root before enabling named pools",
                        root.display()
                    ),
                ));
            }
        }
        Ok(())
    }

    fn write_descriptor(path: &Path) -> io::Result<()> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                file.write_all(&db_meta_bytes())?;
                file.sync_all()
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let contents = fs::read(path)?;
                if validate_db_meta(&contents) {
                    Ok(())
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unrecognized mtxdb database descriptor: {}", path.display()),
                    ))
                }
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DatabaseLayout, ShardType, DB_META_FILENAME};
    use std::fs;

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
    fn initializes_named_pool_layout() {
        let root = test_dir("initialize");
        let layout = DatabaseLayout::open(root.clone()).unwrap();
        assert!(root.join(DB_META_FILENAME).is_file());
        for shard_type in ShardType::ALL {
            assert!(root.join("pools").join(shard_type.as_str()).is_dir());
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
        let mut bytes = super::db_meta_bytes();
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
        fs::write(root.join(DB_META_FILENAME), super::db_meta_bytes()).unwrap();
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
        let mut bytes = super::db_meta_bytes();
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
}
