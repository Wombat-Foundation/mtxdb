//! Database-root layout and named packfile pools.
//!
//! A [`crate::packfile::storage::PackfileStorage`] owns exactly one pool;
//! it must be opened on one of this module's pool directories, never on the
//! database root. Keeping that boundary explicit gives state, event-DAG, and
//! auth-chain data independent shard, GC, durability, and writer-lock
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
/// rather than silently misparsed. The shipped CLI, C ABI, and WASI entry
/// points validate this root and open their selected named pool through this
/// type; [`PackfileStorage`](crate::PackfileStorage) remains available as a
/// lower-level single-pool API.
const DB_META_MAGIC: &[u8; 4] = b"MDBD";
const DB_META_VERSION: u8 = 1;
const DB_META_RESERVED_LEN: usize = 8;
const DB_META_POOL_LIST: &[u8] = b"state\nevent-dag\nauth-chain\n";
const DB_META_FILENAME: &str = "db.meta";

/// `magic + version` header length shared by the writer and the validator.
const DB_META_HEADER_LEN: usize = 4 + 1 + DB_META_RESERVED_LEN;

/// Build the on-disk descriptor bytes for a fresh database root.
fn db_meta_bytes() -> Vec<u8> {
    let mut buf = Vec::with_capacity(DB_META_HEADER_LEN.saturating_add(DB_META_POOL_LIST.len()));
    buf.extend_from_slice(DB_META_MAGIC);
    buf.push(DB_META_VERSION);
    buf.extend_from_slice(&[0u8; DB_META_RESERVED_LEN]);
    buf.extend_from_slice(DB_META_POOL_LIST);
    buf
}

/// Validate a descriptor read from disk: correct magic, a version this
/// build understands, and the expected pool list. The reserved bytes are
/// not validated — a future version may give them meaning, but this build
/// only ever writes them zero-filled.
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
    contents[DB_META_HEADER_LEN..] == *DB_META_POOL_LIST
}

/// A named independent packfile pool in an mtxdb database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardType {
    /// HAMT nodes, roots, and state-group sidecars.
    State,
    /// Event JSON plus collection-DAG-oriented event data.
    EventDag,
    /// Auth-chain manifests and their closure traversal data.
    AuthChain,
}

impl ShardType {
    /// Every shard type defined by the current database layout.
    pub const ALL: [Self; 3] = [Self::State, Self::EventDag, Self::AuthChain];

    /// Stable on-disk directory name for this pool.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::EventDag => "event-dag",
            Self::AuthChain => "auth-chain",
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
    /// Refuses to overlay the pool layout onto a legacy flat store. Such a
    /// store has packfiles directly below `root`; accepting it and then
    /// creating `root/pools/` would make existing data silently disappear
    /// from the new caller's view.
    ///
    /// # Errors
    /// Returns an error if the root cannot be created/read, its descriptor is
    /// unknown, or it contains a legacy flat packfile layout.
    pub fn open(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        // Always check for root-level legacy .pack files, not just on
        // first open. A stray .pack at root after db.meta exists means
        // data was silently dropped into the wrong location.
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
    /// `db.meta`. It validates the root already exists, contains a valid
    /// descriptor, and is not a legacy flat store.
    ///
    /// # Errors
    /// Returns an error if the root does not exist, contains an
    /// unrecognized descriptor, or has a legacy flat layout.
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

    fn reject_legacy_flat_store(root: &Path) -> io::Result<()> {
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
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
            root.join("pools/state")
        );
        assert_eq!(
            layout.pool_dir(ShardType::EventDag).unwrap(),
            root.join("pools/event-dag")
        );
        assert_eq!(
            layout.pool_dir(ShardType::AuthChain).unwrap(),
            root.join("pools/auth-chain")
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
        fs::write(root.join(DB_META_FILENAME), b"MDBD").unwrap();
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
}
