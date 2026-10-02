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

struct TempFileGuard(PathBuf);

impl TempFileGuard {
    /// Remove `path` on drop, ignoring cleanup errors.
    fn new(path: PathBuf) -> Self {
        Self(path)
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
/// Immutable descriptor at a database root.
///
/// Layout: `magic(4) | version(1) | seed(8) | version_len(1) | created_by |
/// pool list`. The seed is random and nonzero, written once when the root is
/// created; every pool's index seed derives from it (see
/// [`DatabaseLayout::pool_seed`]), so a reader knows a pool's seed before the
/// pool exists on disk. `created_by` is the `mtxdb` version that created the
/// root (diagnostic only). The descriptor is parsed, not compared
/// byte-for-byte: a mismatched version is rejected rather than misparsed.
/// Every root has one shared WAL at its root, so the descriptor records no WAL
/// layout. The shipped CLI, C ABI, and WASI entry points validate this root and
/// open their selected named pool through this type;
/// [`PackfileStorage`](crate::PackfileStorage) remains available as a
/// lower-level single-pool API.
const DB_META_MAGIC: &[u8; 4] = b"MTXD";
const DB_META_VERSION: u8 = 3;
const DB_META_SEED_LEN: usize = 8;
/// The pool list a descriptor must carry, built from the pool directory names
/// so it can never drift from them. A root whose descriptor lists other names
/// is rejected by [`parse_db_meta`] instead of being opened with its data
/// orphaned in directories nothing reads.
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

/// `magic + version + seed + version_len` header length.
const DB_META_FIXED_LEN: usize = 4 + 1 + DB_META_SEED_LEN + 1;

/// A parsed, valid database descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DbMeta {
    seed: u64,
    created_by: String,
}

/// A fresh nonzero random seed.
fn random_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    loop {
        let seed = RandomState::new().build_hasher().finish();
        if seed != 0 {
            return seed;
        }
    }
}

/// Build the on-disk descriptor bytes for a database root.
fn db_meta_bytes(seed: u64) -> Vec<u8> {
    let pool_list = db_meta_pool_list();
    let created_by = env!("CARGO_PKG_VERSION").as_bytes();
    let created_by = &created_by[..created_by.len().min(usize::from(u8::MAX))];
    let mut buf = Vec::with_capacity(
        DB_META_FIXED_LEN
            .saturating_add(created_by.len())
            .saturating_add(pool_list.len()),
    );
    buf.extend_from_slice(DB_META_MAGIC);
    buf.push(DB_META_VERSION);
    buf.extend_from_slice(&seed.to_le_bytes());
    buf.push(u8::try_from(created_by.len()).unwrap_or(u8::MAX));
    buf.extend_from_slice(created_by);
    buf.extend_from_slice(&pool_list);
    buf
}

/// Parse a descriptor read from disk: correct magic, a version this build
/// understands, a nonzero seed, and the expected pool list.
fn parse_db_meta(contents: &[u8]) -> Option<DbMeta> {
    if contents.len() < DB_META_FIXED_LEN
        || &contents[..4] != DB_META_MAGIC
        || contents[4] != DB_META_VERSION
    {
        return None;
    }
    let seed = u64::from_le_bytes(contents[5..5 + DB_META_SEED_LEN].try_into().ok()?);
    if seed == 0 {
        return None;
    }
    let version_len = usize::from(contents[DB_META_FIXED_LEN - 1]);
    let created_end = DB_META_FIXED_LEN.checked_add(version_len)?;
    let created_by =
        String::from_utf8(contents.get(DB_META_FIXED_LEN..created_end)?.to_vec()).ok()?;
    (contents[created_end..] == db_meta_pool_list()[..]).then_some(DbMeta { seed, created_by })
}

fn read_db_meta(path: &Path) -> io::Result<DbMeta> {
    parse_db_meta(&fs::read(path)?).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unrecognized mtxdb database descriptor: {}", path.display()),
        )
    })
}

/// The index seed of the pool at `pool_dir` when it lives at
/// `<root>/pools/<pool>` inside a database root; `None` for a standalone pool
/// directory, which keeps its own `pool.meta`.
///
/// # Errors
/// Returns `InvalidData` if the enclosing `db.meta` is unrecognized.
pub fn enclosing_pool_seed(pool_dir: &Path) -> io::Result<Option<u64>> {
    let Some(name) = pool_dir.file_name().and_then(|name| name.to_str()) else {
        return Ok(None);
    };
    let Some(shard_type) = ShardType::ALL.into_iter().find(|t| t.as_str() == name) else {
        return Ok(None);
    };
    let Some(pools) = pool_dir
        .parent()
        .filter(|p| p.file_name().is_some_and(|n| n == "pools"))
    else {
        return Ok(None);
    };
    let Some(root) = pools.parent() else {
        return Ok(None);
    };
    let meta_path = root.join(DB_META_FILENAME);
    if !meta_path.is_file() {
        return Ok(None);
    }
    Ok(Some(pool_seed(read_db_meta(&meta_path)?.seed, shard_type)))
}

/// The `mtxdb` version that created the database root enclosing `pool_dir`.
#[must_use]
pub fn enclosing_created_by(pool_dir: &Path) -> Option<String> {
    let root = pool_dir.parent()?.parent()?;
    read_db_meta(&root.join(DB_META_FILENAME))
        .ok()
        .map(|meta| meta.created_by)
}

/// Derive a pool's index seed from the root seed and the pool's tag. Never zero.
fn pool_seed(root_seed: u64, shard_type: ShardType) -> u64 {
    let mut label = Vec::with_capacity(12);
    label.extend_from_slice(&root_seed.to_le_bytes());
    label.extend_from_slice(&shard_type.physical_pool_tag());
    let digest = crate::storage::DigestAlgorithm::Blake3.digest(&label);
    let seed = u64::from_le_bytes(digest[..8].try_into().unwrap_or([1; 8]));
    if seed == 0 {
        1
    } else {
        seed
    }
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
    if parse_db_meta(&contents).is_none() {
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
        if parse_db_meta(&contents).is_none() {
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
    /// Server metadata, including federation signing keys and raw key responses.
    ServerInfo,
}

impl ShardType {
    /// Every shard type defined by the current database layout.
    pub const ALL: [Self; 4] = [Self::State, Self::EventDag, Self::Edges, Self::ServerInfo];

    /// This pool's position in [`Self::ALL`], the canonical pool order. Use it to
    /// index per-pool arrays so they cannot drift from `ALL` when a pool is added.
    #[must_use]
    pub const fn index(self) -> usize {
        let mut position = 0;
        while position < Self::ALL.len() {
            if Self::ALL[position] as usize == self as usize {
                return position;
            }
            position = position.saturating_add(1);
        }
        // Unreachable: every variant is in `ALL`, and a test checks it.
        Self::ALL.len()
    }

    /// Stable on-disk directory name for this pool.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::State => "mtpl-state",
            Self::EventDag => "mtpl-event",
            Self::Edges => "mtpl-edges",
            Self::ServerInfo => "mtpl-server-info",
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
            Self::ServerInfo => *b"SINF",
        }
    }
}

/// Validated database root from which named pool paths can be derived.
#[derive(Debug, Clone)]
pub struct DatabaseLayout {
    root: PathBuf,
    seed: u64,
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
        if !meta_path.exists() {
            Self::write_descriptor(&meta_path)?;
        }
        let seed = read_db_meta(&meta_path)?.seed;
        Ok(Self { root, seed })
    }

    /// The index seed of `shard_type`'s pool, derived from the root seed so it
    /// is known before the pool exists on disk.
    #[must_use]
    pub fn pool_seed(&self, shard_type: ShardType) -> u64 {
        pool_seed(self.seed, shard_type)
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

    /// A named pool's directory path, without creating or checking it. An
    /// absent directory is an empty pool: the pool creates it with its first
    /// pack.
    #[must_use]
    pub fn pool_path(&self, shard_type: ShardType) -> PathBuf {
        self.root.join("pools").join(shard_type.as_str())
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
        let seed = read_db_meta(&meta_path)?.seed;
        Self::reject_legacy_flat_store(&root)?;
        Ok(Self { root, seed })
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

    /// Install a complete database descriptor from a synced temporary file.
    ///
    /// Accepts a valid descriptor installed by a concurrent opener. File creation,
    /// write, sync, installation, and validation errors propagate; temporary-file
    /// cleanup and parent-directory sync are best-effort.
    fn write_descriptor(path: &Path) -> io::Result<()> {
        let (temporary, file) = loop {
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
        // Rebind so the handle is declared after the guard: on the `?` error
        // paths `file` then closes first, and on Windows the guard's
        // `remove_file` is not blocked by the still-open handle.
        let mut file = file;
        file.write_all(&db_meta_bytes(random_seed()))?;
        file.sync_all()?;
        drop(file);

        // hard_link installs the fully-written file atomically without
        // replacing a descriptor another opener may have installed first.
        Self::install_descriptor_temp(&temporary, path, |source, destination| {
            fs::hard_link(source, destination)
        })
    }

    /// Install the synced first-create descriptor at `temporary` into `path`.
    ///
    /// Uses `hard_link`, falling back to rename on `Unsupported` or
    /// `PermissionDenied`. An existing destination or a concurrently removed
    /// temporary file is accepted only if the destination validates. Other
    /// installation and validation errors propagate. Parent sync and cleanup
    /// are best-effort.
    fn install_descriptor_temp(
        temporary: &Path,
        path: &Path,
        hard_link: impl FnOnce(&Path, &Path) -> io::Result<()>,
    ) -> io::Result<()> {
        match hard_link(temporary, path) {
            Ok(()) => {
                Self::sync_descriptor_parent(path)?;
                Self::sweep_descriptor_temps(path, "create");
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                Self::validate_existing_descriptor(path)?;
                Self::sync_descriptor_parent(path)?;
                Self::sweep_descriptor_temps(path, "create");
                Ok(())
            }
            // A concurrent opener can install the canonical file and sweep
            // this temp after we created it. In that case the canonical file
            // is the result to validate, not a reason to fail initialization.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Self::validate_existing_descriptor(path)
            }
            Err(error)
                if error.kind() == io::ErrorKind::Unsupported
                    || error.kind() == io::ErrorKind::PermissionDenied =>
            {
                // Some filesystems do not support hard links (and some report
                // that as `PermissionDenied`). The source is a fully synced
                // sibling temp and first-create bytes are deterministic, so
                // same-directory rename is an atomic install fallback. The seed
                // is random, so never replace a descriptor another opener has
                // already installed (checked above). Any other error is a real
                // fault and is propagated below.
                if path.exists() {
                    return Self::validate_existing_descriptor(path);
                }
                match fs::rename(temporary, path) {
                    Ok(()) => {
                        Self::sync_descriptor_parent(path)?;
                        Self::sweep_descriptor_temps(path, "create");
                        Ok(())
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {
                        Self::validate_existing_descriptor(path)
                    }
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        Self::validate_existing_descriptor(path)
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Sync the descriptor's parent directory so the installed entry survives a
    /// power loss. A rooted pool has no `pool.meta` fallback for the seed, so a
    /// lost `db.meta` would be recreated with a different seed; failures
    /// therefore propagate. Only filesystems that report directory sync as
    /// unsupported are tolerated.
    fn sync_descriptor_parent(path: &Path) -> io::Result<()> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        match crate::shard::sync_directory(parent) {
            Err(error) if error.kind() == io::ErrorKind::Unsupported => Ok(()),
            other => other,
        }
    }

    /// Remove regular sibling files named `.<descriptor>.<kind>.*`.
    ///
    /// Directory-read, metadata, and removal failures are ignored.
    fn sweep_descriptor_temps(path: &Path, kind: &str) {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let prefix = format!(
            ".{}.{}.",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(DB_META_FILENAME),
            kind,
        );
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(&prefix))
                    && entry.file_type().is_ok_and(|kind| kind.is_file())
                {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Validate the current descriptor format and pool list at `path`.
    ///
    /// Propagates read errors and returns `InvalidData` for an unrecognized
    /// descriptor.
    fn validate_existing_descriptor(path: &Path) -> io::Result<()> {
        read_db_meta(path).map(|_| ())
    }
}

#[cfg(test)]
#[path = "test_layout.rs"]
mod tests;
