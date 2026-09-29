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
use std::sync::atomic::{AtomicU64, Ordering};

static DESCRIPTOR_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFileGuard {
    path: Option<PathBuf>,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

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
        let (temporary, mut file) = loop {
            let suffix = DESCRIPTOR_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!(
                ".{}.create.{}.{}",
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(DB_META_FILENAME),
                std::process::id(),
                suffix,
            );
            let temporary = path.with_file_name(name);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => break (temporary, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        };
        let _temporary_guard = TempFileGuard::new(temporary.clone());
        file.write_all(&db_meta_bytes())?;
        file.sync_all()?;
        drop(file);

        // hard_link installs the fully-written file atomically without
        // replacing a descriptor another opener may have installed first.
        match fs::hard_link(&temporary, path) {
            Ok(()) => {
                // The descriptor contents are synced before installation.
                // Directory sync is best-effort: some supported filesystems
                // reject it, and a lost first-install directory entry can be
                // safely recreated on the next open. `Ok` therefore means
                // installed, not guaranteed durable across sudden power loss.
                let parent = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                let _ = crate::shard::sync_directory(parent);
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                Self::validate_existing_descriptor(path)
            }
            Err(error) => Err(error),
        }
    }

    fn validate_existing_descriptor(path: &Path) -> io::Result<()> {
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
}

#[cfg(test)]
#[path = "test_layout.rs"]
mod tests;
