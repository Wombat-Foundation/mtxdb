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
const DB_META: &[u8] = b"MDBD\x01\nstate\nevent-dag\nauth-chain\n";
const DB_META_FILENAME: &str = "db.meta";

/// A named independent packfile pool in an mtxdb database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardType {
    /// HAMT nodes, roots, and state-group sidecars.
    State,
    /// Event JSON plus room-DAG-oriented event data.
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
        let meta_path = root.join(DB_META_FILENAME);
        if meta_path.exists() {
            let contents = fs::read(&meta_path)?;
            if contents != DB_META {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unrecognized mtxdb database descriptor: {}",
                        meta_path.display()
                    ),
                ));
            }
        } else {
            Self::reject_legacy_flat_store(&root)?;
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
                file.write_all(DB_META)?;
                file.sync_all()
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let contents = fs::read(path)?;
                if contents == DB_META {
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
}
