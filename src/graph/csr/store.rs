//! Pool-scoped CSR graph store outside the packs.
//!
//! There is one store per source pool at `<db_root>/graph/<pool>/`, shared by
//! every collection in that pool. A row is addressed by
//! `(scope, family, owner)`, where `scope` is the engine's generic 16-byte
//! collection id, `family` is an [`EdgeFamily`], and `owner` is a dense local
//! id. Nothing here knows about rooms.
//!
//! Storage is an append-only tail log plus immutable sealed segments:
//!
//! - [`CsrWriter::append_rows`] appends one framed record per changed row and
//!   syncs the tail. Rewriting one row never rewrites the others.
//! - [`CsrWriter::seal`] folds the tail into a new immutable `seg-N.csr` file
//!   (temp file, fsync, rename, directory fsync), then truncates the tail. A
//!   crash between the rename and the truncate replays the same rows again,
//!   which is idempotent.
//! - [`CsrStore::read_scope`] replays segments oldest to newest, then the tail.
//!   Rows are immutable per `(scope, family, owner)`, so the newest record
//!   wins. A torn final tail frame is dropped.
//!
//! A purge is a tombstone record: it clears the scope during replay, so rows
//! already sealed into shared segments are hidden without rewriting them.
//!
//! # Concurrency
//!
//! One writer owns a store directory at a time: the database root's writer
//! lock (`SharedWalLock`, held for the life of the `Database`) keeps a second
//! writer process out, and CSR writes go through the `Database`. The decoded
//! tail cache relies on that. It records how much of `tail.log` it has
//! verified, so a second process appending or truncating the same directory
//! could leave that offset stale and a later append could cut the other
//! writer's frames. Readers in other processes only read files and are safe.
//!
//! Writes enforce that: the write methods are crate-private, and the public
//! [`CsrStore::writer`] needs the owning `Database` (so the lock is held) and
//! rejects one whose root is not the store's. Threads sharing one `Database`
//! are serialized by a per-directory write lock that every mutation holds.
//! Reads and writes share the per-directory lock so a reader cannot observe a
//! seal between segment enumeration and tail-cache replay.
//!
//! The tail stays under [`SEAL_THRESHOLD`] because both tail writers,
//! `append_rows` and `purge_scope`, seal after
//! writing. A new writer path must do the same, or the tail and its cached
//! rows grow without bound.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock, Weak};

use super::{Edge, EdgeFamily};
use crate::database::Database;
use crate::layout::{DatabaseLayout, ShardType};

const SEGMENT_MAGIC: [u8; 4] = *b"MCSS";
const SEGMENT_VERSION: u8 = 1;
const SEGMENT_PREFIX: &str = "seg-";
const SEGMENT_SUFFIX: &str = ".csr";
const TAIL_FILE: &str = "tail.log";

/// Seal once the tail reaches this many bytes.
pub const SEAL_THRESHOLD: u64 = 4 * 1024 * 1024;

const RECORD_HEADER: usize = 16 + 2 + 1 + 1 + 4 + 4;
const EDGE_SIZE: usize = 4 + 2;
/// Segment header: magic, version, scope count, record count, payload length.
const SEGMENT_HEADER: usize = 4 + 1 + 4 + 4 + 4;
/// One scope index entry: scope, flags, block offset, block length, directory
/// length. The directory is `(family, owner) -> row`.
const SCOPE_ENTRY: usize = 16 + 4 + 4 + 4 + 4;
/// One directory entry: family, owner, row offset in the data region, row
/// length, row checksum.
const DIR_ENTRY: usize = 2 + 4 + 4 + 4 + 4;
const SCOPE_FLAG_PURGE: u32 = 1;
const BLOCK_COUNT: usize = 4;

/// A generic collection scope: the engine's 16-byte collection id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub [u8; 16]);

/// One row to append: a family, an owner local id and its edges.
#[derive(Debug, Clone)]
pub struct RowRecord {
    /// The edge family.
    pub family: EdgeFamily,
    /// The owner local id.
    pub owner: u32,
    /// The owner's targets in this family.
    pub edges: Vec<Edge>,
}

/// A row read back from the store.
#[derive(Debug, Clone)]
pub struct StoredRow {
    /// The edge family.
    pub family: EdgeFamily,
    /// The owner local id.
    pub owner: u32,
    /// The owner's targets in this family.
    pub edges: Vec<Edge>,
}

/// Every row currently visible for one scope.
#[derive(Debug, Clone, Default)]
pub struct ScopeData {
    /// The scope's rows, in no defined order.
    pub rows: Vec<StoredRow>,
}

/// One pool's shared CSR store.
#[derive(Debug, Clone)]
pub struct CsrStore {
    dir: PathBuf,
}

/// Write access to one pool's store, tied to the database that owns it.
///
/// The decoded-tail cache assumes a single writer per store directory, and the
/// database root's writer lock provides it for the life of the [`Database`].
/// Borrowing the `Database` here makes a write without that lock impossible,
/// and [`CsrStore::writer`] refuses a database whose root is not the store's.
/// Serialization between threads is the store's per-directory write lock, so
/// the raw crate-private methods take it too. This handle only adds the
/// root-lock proof; it does not take the lock itself.
#[derive(Clone, Copy)]
pub struct CsrWriter<'a> {
    store: &'a CsrStore,
    _database: &'a Database,
}

impl CsrWriter<'_> {
    /// Append one framed record per row to the tail; see the module docs.
    ///
    /// # Errors
    /// I/O errors, or a row with too many edges.
    pub fn append_rows(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        self.store.append_rows(scope, rows)
    }

    /// Append while the caller holds [`CsrStore::with_write_lock`].
    pub(crate) fn append_rows_locked(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        self.store.append_rows_locked(scope, rows)
    }

    /// Hide every row of `scope` with a tombstone record.
    ///
    /// # Errors
    /// I/O errors.
    pub fn purge_scope(&self, scope: ScopeId) -> io::Result<()> {
        self.store.purge_scope(scope)
    }

    /// Fold the tail into a new immutable segment.
    ///
    /// # Errors
    /// I/O errors.
    pub fn seal(&self) -> io::Result<()> {
        self.store.seal()
    }
}

type RowKey = ([u8; 16], u16, u32);

impl CsrStore {
    /// Address the store for `pool` under `layout`.
    #[must_use]
    pub fn open(layout: &DatabaseLayout, pool: ShardType) -> Self {
        Self {
            dir: physical_path(&layout.graph_path(pool)),
        }
    }

    fn store_lock(&self) -> Arc<RwLock<()>> {
        let mut locks = WRITE_LOCKS.lock().unwrap_or_else(PoisonError::into_inner);
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(&self.dir).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(RwLock::new(()));
        locks.insert(self.dir.clone(), Arc::downgrade(&lock));
        lock
    }

    /// Run a mutation while excluding reads and other mutations on this store.
    /// Compound adjacency writes use this to keep their read/merge/append
    /// sequence atomic with respect to other writers.
    pub(crate) fn with_write_lock<R>(&self, operation: impl FnOnce() -> R) -> R {
        let lock = self.store_lock();
        let _guard = lock.write().unwrap_or_else(PoisonError::into_inner);
        operation()
    }

    /// Read while no mutation (seal, append, purge) can change the files.
    /// Readers may overlap each other.
    fn with_read_lock<R>(&self, operation: impl FnOnce() -> R) -> R {
        let lock = self.store_lock();
        let _guard = lock.read().unwrap_or_else(PoisonError::into_inner);
        operation()
    }

    /// Write access to this store through the database that owns its root.
    ///
    /// # Errors
    /// `InvalidInput` when this is not one of `database`'s graph stores, since
    /// that database's writer lock would not cover it. A path-prefix check is
    /// not enough: a database rooted above this one does not own it.
    pub fn writer<'a>(&'a self, database: &'a Database) -> io::Result<CsrWriter<'a>> {
        let layout = database.layout();
        let owned = ShardType::ALL
            .into_iter()
            .any(|pool| physical_path(&layout.graph_path(pool)) == self.dir);
        if !owned {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "csr store does not belong to this database",
            ));
        }
        Ok(CsrWriter {
            store: self,
            _database: database,
        })
    }

    /// The store directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn tail_path(&self) -> PathBuf {
        self.dir.join(TAIL_FILE)
    }

    /// Append changed rows and sync the tail.
    ///
    /// Empty input does nothing and does not create the store directory.
    ///
    /// # Errors
    /// I/O errors, or a row with more than `u32::MAX` edges.
    pub(crate) fn append_rows(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        let lock = self.store_lock();
        let _guard = lock.write().unwrap_or_else(PoisonError::into_inner);
        self.append_rows_locked(scope, rows)
    }

    /// Append while the caller holds this directory's write lock.
    fn append_rows_locked(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)?;
        let valid = self.with_tail(|tail| tail.valid)?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.tail_path())?;
        let len = file.metadata()?.len();
        if valid < len {
            // Drop a torn final frame so new records follow a valid boundary.
            file.set_len(valid)?;
        }
        file.seek(SeekFrom::End(0))?;
        let mut buffer = Vec::new();
        for row in rows {
            let payload = encode_row(scope, row.family, row.owner, &row.edges, false)?;
            write_frame(&mut buffer, &payload);
        }
        file.write_all(&buffer)?;
        file.sync_data()?;
        self.maybe_seal()?;
        Ok(())
    }

    /// Hide every row of `scope` with a tombstone and sync the tail.
    ///
    /// # Errors
    /// I/O errors.
    pub(crate) fn purge_scope(&self, scope: ScopeId) -> io::Result<()> {
        let lock = self.store_lock();
        let _guard = lock.write().unwrap_or_else(PoisonError::into_inner);
        self.purge_scope_locked(scope)
    }

    /// Append a purge while the caller holds this directory's write lock.
    fn purge_scope_locked(&self, scope: ScopeId) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.tail_path())?;
        let valid = self.with_tail(|tail| tail.valid)?;
        let len = file.metadata()?.len();
        if valid < len {
            file.set_len(valid)?;
        }
        file.seek(SeekFrom::End(0))?;
        let payload = encode_row(scope, EdgeFamily::plain(0), 0, &[], true)?;
        let mut buffer = Vec::new();
        write_frame(&mut buffer, &payload);
        file.write_all(&buffer)?;
        file.sync_data()?;
        // Tombstones grow the tail like rows do, so a purge-only workload must
        // seal at the same threshold or the tail and its cache grow unbounded.
        self.maybe_seal()?;
        Ok(())
    }

    /// Apply the tail as a new immutable segment, then truncate the tail.
    ///
    /// # Errors
    /// I/O errors or a corrupt frame that is not merely torn at the end.
    pub(crate) fn seal(&self) -> io::Result<()> {
        let lock = self.store_lock();
        let _guard = lock.write().unwrap_or_else(PoisonError::into_inner);
        self.seal_locked()
    }

    /// Seal while the caller holds this directory's write lock.
    fn seal_locked(&self) -> io::Result<()> {
        if !self.dir.is_dir() {
            return Ok(());
        }
        let (records, valid) = self.read_tail()?;
        let tail = self.tail_path();
        if records.is_empty() {
            if tail.exists() {
                let len = fs::metadata(&tail)?.len();
                if (valid as u64) < len {
                    OpenOptions::new()
                        .write(true)
                        .open(&tail)?
                        .set_len(valid as u64)?;
                }
            }
            return Ok(());
        }
        let mut groups = group_tail(&records)?;
        let file_bytes = encode_segment(&mut groups, records.len())?;

        let seq = self.next_segment_number();
        let name = format!("{SEGMENT_PREFIX}{seq:012}{SEGMENT_SUFFIX}");
        let target = self.dir.join(&name);
        let temp = self.dir.join(format!(".{name}.tmp"));
        {
            let mut file = File::create(&temp)?;
            file.write_all(&file_bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &target)?;
        sync_dir(&self.dir)?;
        // The segment is durable; only now is it safe to drop the tail.
        OpenOptions::new().write(true).open(&tail)?.set_len(0)?;
        Ok(())
    }

    /// Seal if the tail is large enough.
    fn maybe_seal(&self) -> io::Result<()> {
        if fs::metadata(self.tail_path()).map(|meta| meta.len())? >= SEAL_THRESHOLD {
            self.seal_locked()?;
        }
        Ok(())
    }

    /// Read every visible row of `scope`.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_scope(&self, scope: ScopeId) -> io::Result<ScopeData> {
        self.with_read_lock(|| self.read_scope_locked(scope))
    }

    fn read_scope_locked(&self, scope: ScopeId) -> io::Result<ScopeData> {
        let mut rows: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
        for (_, path) in self.segments()? {
            apply_segment(&path, scope, None, None, &mut rows)?;
        }
        self.with_tail(|tail| {
            // A tombstone in the tail is newer than every sealed segment.
            if tail.purged.contains(&scope.0) {
                rows.clear();
            }
            for (key, value) in &tail.rows {
                if key.0 == scope.0 {
                    rows.insert(*key, value.clone());
                }
            }
        })?;
        let mut out = Vec::new();
        for ((_, family, owner), (typed, edges)) in rows {
            out.push(StoredRow {
                family: EdgeFamily { id: family, typed },
                owner,
                edges,
            });
        }
        out.sort_by_key(|row| (row.family.id, row.owner));
        Ok(ScopeData { rows: out })
    }

    /// Read specific `(family, owner)` rows, in the order requested.
    ///
    /// This is the hot path for recording and point reads: sealed segments are
    /// sought by their `(family, owner)` directory, so a lookup does not decode
    /// the rest of the collection. `None` means the row is not recorded.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_rows(
        &self,
        scope: ScopeId,
        keys: &[(EdgeFamily, u32)],
    ) -> io::Result<Vec<Option<Vec<Edge>>>> {
        self.with_read_lock(|| self.read_rows_locked(scope, keys))
    }

    pub(crate) fn read_rows_locked(
        &self,
        scope: ScopeId,
        keys: &[(EdgeFamily, u32)],
    ) -> io::Result<Vec<Option<Vec<Edge>>>> {
        let mut wanted: HashSet<(u16, u32)> = HashSet::with_capacity(keys.len());
        for (family, owner) in keys {
            wanted.insert((family.id, *owner));
        }
        let mut found: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
        for (_, path) in self.segments()? {
            apply_segment(&path, scope, Some(&wanted), None, &mut found)?;
        }
        self.with_tail(|tail| {
            if tail.purged.contains(&scope.0) {
                found.clear();
            }
            for &(family, owner) in &wanted {
                let key = (scope.0, family, owner);
                if let Some(value) = tail.rows.get(&key) {
                    found.insert(key, value.clone());
                }
            }
        })?;
        Ok(keys
            .iter()
            .map(|(family, owner)| {
                found
                    .get(&(scope.0, family.id, *owner))
                    .map(|(_, edges)| edges.clone())
            })
            .collect())
    }

    /// Read all rows for one family without decoding the other families.
    ///
    /// Sealed scope directories are ordered by `(family, owner)`, so this reads
    /// only the selected family's contiguous directory range and row payloads.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_family(
        &self,
        scope: ScopeId,
        family: EdgeFamily,
    ) -> io::Result<Vec<(u32, Vec<Edge>)>> {
        self.with_read_lock(|| self.read_family_locked(scope, family))
    }

    fn read_family_locked(
        &self,
        scope: ScopeId,
        family: EdgeFamily,
    ) -> io::Result<Vec<(u32, Vec<Edge>)>> {
        let mut rows: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
        for (_, path) in self.segments()? {
            apply_segment(&path, scope, None, Some(family), &mut rows)?;
        }
        self.with_tail(|tail| {
            if tail.purged.contains(&scope.0) {
                rows.clear();
            }
            for (&(row_scope, row_family, owner), value) in &tail.rows {
                if row_scope == scope.0 && row_family == family.id {
                    rows.insert((row_scope, row_family, owner), value.clone());
                }
            }
        })?;
        let mut selected = Vec::with_capacity(rows.len());
        for ((_, _, owner), (typed, edges)) in rows {
            if typed != family.typed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "csr family typing does not match stored rows",
                ));
            }
            selected.push((owner, edges));
        }
        selected.sort_by_key(|(owner, _)| *owner);
        Ok(selected)
    }

    fn next_segment_number(&self) -> u64 {
        self.segments().map_or(0, |segments| {
            segments.last().map_or(0, |(seq, _)| seq.saturating_add(1))
        })
    }

    fn segments(&self) -> io::Result<Vec<(u64, PathBuf)>> {
        if !self.dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut segments = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(seq) = name
                .strip_prefix(SEGMENT_PREFIX)
                .and_then(|name| name.strip_suffix(SEGMENT_SUFFIX))
                .and_then(|digits| digits.parse::<u64>().ok())
            else {
                continue;
            };
            segments.push((seq, path));
        }
        segments.sort_by_key(|(seq, _)| *seq);
        Ok(segments)
    }

    /// Run `read` over the decoded tail, first decoding only the bytes appended
    /// since the last call. A file shorter than what was decoded (a seal)
    /// discards the cache.
    fn with_tail<R>(&self, read: impl FnOnce(&TailState) -> R) -> io::Result<R> {
        let len = match fs::metadata(self.tail_path()) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error),
        };
        // Take the entry out so no lock is held across disk reads.
        let cached = TAIL_CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.dir);
        let mut state = match cached {
            Some(state) if state.valid <= len => state,
            _ => TailState::default(),
        };
        if state.valid < len {
            let (frames, valid) = self.read_tail_from(state.valid)?;
            for frame in &frames {
                state.apply(&frame.payload)?;
            }
            state.valid = state.valid.saturating_add(valid as u64);
        }
        let result = read(&state);
        TAIL_CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(self.dir.clone(), state);
        Ok(result)
    }

    /// The tail's valid frames, read fresh from disk, and the byte length they
    /// cover. `seal` needs them in order and uncached.
    fn read_tail(&self) -> io::Result<(Vec<Frame>, usize)> {
        self.read_tail_from(0)
    }

    /// Decode the tail frames that start at byte `offset`, returning them and
    /// how many bytes from `offset` they cover.
    fn read_tail_from(&self, offset: u64) -> io::Result<(Vec<Frame>, usize)> {
        let mut file = match File::open(self.tail_path()) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
            Err(error) => return Err(error),
        };
        file.seek(SeekFrom::Start(offset))?;
        let mut data = Vec::new();
        #[allow(
            clippy::verbose_file_reads,
            reason = "fs::read cannot start at an offset; only the new tail bytes are wanted"
        )]
        file.read_to_end(&mut data)?;
        Ok(decode_frames(&data))
    }
}

/// Group the tail by scope, then by `(family, owner)` inside each scope, so a
/// point or batch read can seek straight to the rows it asked for.
fn group_tail(records: &[Frame]) -> io::Result<Vec<([u8; 16], ScopeGroup)>> {
    let mut groups: Vec<([u8; 16], ScopeGroup)> = Vec::new();
    let mut group_index: HashMap<[u8; 16], usize> = HashMap::new();
    for record in records {
        if record.payload.len() < RECORD_HEADER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "csr row has no scope",
            ));
        }
        let (scope, family, owner, purge, _) = decode_row(&record.payload)?;
        let slot = *group_index.entry(scope.0).or_insert_with(|| {
            groups.push((scope.0, ScopeGroup::default()));
            groups.len().saturating_sub(1)
        });
        let group = &mut groups[slot].1;
        if purge {
            group.rows.clear();
            group.index.clear();
            group.purged = true;
            continue;
        }
        let key = (family.id, owner);
        if let Some(&at) = group.index.get(&key) {
            group.rows[at].1.clone_from(&record.payload);
        } else {
            group.index.insert(key, group.rows.len());
            group.rows.push((key, record.payload.clone()));
        }
    }
    Ok(groups)
}

fn too_large(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// One scope's block: its `(family, owner)` directory followed by the row data.
fn encode_scope_block(group: &mut ScopeGroup) -> io::Result<(Vec<u8>, Vec<u8>)> {
    group.rows.sort_by_key(|(key, _)| *key);
    let mut directory =
        Vec::with_capacity(BLOCK_COUNT.saturating_add(group.rows.len().saturating_mul(DIR_ENTRY)));
    directory.extend_from_slice(
        &u32::try_from(group.rows.len())
            .map_err(|_| too_large("too many csr rows"))?
            .to_le_bytes(),
    );
    let mut data = Vec::new();
    for ((family, owner), row) in &group.rows {
        let offset =
            u32::try_from(data.len()).map_err(|_| too_large("csr scope block too large"))?;
        let len = u32::try_from(row.len()).map_err(|_| too_large("csr row too large"))?;
        directory.extend_from_slice(&family.to_le_bytes());
        directory.extend_from_slice(&owner.to_le_bytes());
        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(&len.to_le_bytes());
        directory.extend_from_slice(&crc32fast::hash(row).to_le_bytes());
        data.extend_from_slice(row);
    }
    Ok((directory, data))
}

/// Lay the grouped tail out as one segment file: header, scope index, payload
/// and a checksum over the payload.
fn encode_segment(groups: &mut [([u8; 16], ScopeGroup)], records: usize) -> io::Result<Vec<u8>> {
    let mut payload = Vec::new();
    let mut index = Vec::with_capacity(groups.len().saturating_mul(SCOPE_ENTRY));
    for (scope, group) in groups.iter_mut() {
        let purged = group.purged;
        let (directory, data) = encode_scope_block(group)?;
        let block_len = directory.len().saturating_add(data.len());
        let block_off =
            u32::try_from(payload.len()).map_err(|_| too_large("csr segment too large"))?;
        index.extend_from_slice(scope);
        index.extend_from_slice(&if purged { SCOPE_FLAG_PURGE } else { 0 }.to_le_bytes());
        index.extend_from_slice(&block_off.to_le_bytes());
        index.extend_from_slice(
            &u32::try_from(block_len)
                .map_err(|_| too_large("csr scope block too large"))?
                .to_le_bytes(),
        );
        index.extend_from_slice(
            &u32::try_from(directory.len())
                .map_err(|_| too_large("csr directory too large"))?
                .to_le_bytes(),
        );
        payload.extend_from_slice(&directory);
        payload.extend_from_slice(&data);
    }
    let mut file_bytes = Vec::with_capacity(
        SEGMENT_HEADER
            .saturating_add(index.len())
            .saturating_add(payload.len())
            .saturating_add(4),
    );
    file_bytes.extend_from_slice(&SEGMENT_MAGIC);
    file_bytes.push(SEGMENT_VERSION);
    file_bytes.extend_from_slice(
        &u32::try_from(groups.len())
            .map_err(|_| too_large("too many csr scopes"))?
            .to_le_bytes(),
    );
    file_bytes.extend_from_slice(
        &u32::try_from(records)
            .map_err(|_| too_large("too many csr rows"))?
            .to_le_bytes(),
    );
    file_bytes.extend_from_slice(
        &u32::try_from(payload.len())
            .map_err(|_| too_large("csr segment too large"))?
            .to_le_bytes(),
    );
    file_bytes.extend_from_slice(&index);
    file_bytes.extend_from_slice(&payload);
    file_bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    Ok(file_bytes)
}

/// The decoded tail of one store directory: the byte length it covers, its rows
/// by key, and the scopes a tombstone in the tail has purged.
///
/// The tail seals at [`SEAL_THRESHOLD`], so this holds at most one tail's worth
/// of rows. Sealed segments are not cached; they are read by directory.
#[derive(Default)]
struct TailState {
    valid: u64,
    rows: HashMap<RowKey, (bool, Vec<Edge>)>,
    purged: HashSet<[u8; 16]>,
}

impl TailState {
    fn apply(&mut self, payload: &[u8]) -> io::Result<()> {
        let (scope, family, owner, purge, edges) = decode_row(payload)?;
        if purge {
            self.rows
                .retain(|(row_scope, _, _), _| *row_scope != scope.0);
            self.purged.insert(scope.0);
        } else {
            self.rows
                .insert((scope.0, family.id, owner), (family.typed, edges));
        }
        Ok(())
    }
}

static TAIL_CACHE: LazyLock<Mutex<HashMap<PathBuf, TailState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// One read/write lock per store directory, shared by every handle in the
/// process.
///
/// The root lock keeps other processes out, but threads can write to one store
/// through a [`CsrWriter`] or, inside the crate, the store's own methods.
/// Appends, purges and seals each open the tail, compare it with the verified
/// length and may truncate or replace it, so two of them at once would cut
/// each other's frames. Every mutation therefore holds the write side from the
/// tail check through its seal, and every read holds the read side so it cannot
/// see a seal between enumerating segments and replaying the tail.
static WRITE_LOCKS: LazyLock<Mutex<HashMap<PathBuf, Weak<RwLock<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolve existing symlinks in a store path while preserving nonexistent
/// suffixes (stores are created lazily with their first write).
fn physical_path(path: &Path) -> PathBuf {
    let mut suffix = Vec::new();
    let mut ancestor = path;
    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            return path.to_path_buf();
        };
        suffix.push(name.to_os_string());
        let Some(parent) = ancestor.parent() else {
            return path.to_path_buf();
        };
        ancestor = parent;
    }
    let mut resolved = fs::canonicalize(ancestor).unwrap_or_else(|_| ancestor.to_path_buf());
    for component in suffix.into_iter().rev() {
        resolved.push(component);
    }
    resolved
}

struct Frame {
    payload: Vec<u8>,
}

#[derive(Default)]
struct ScopeGroup {
    rows: Vec<((u16, u32), Vec<u8>)>,
    index: HashMap<(u16, u32), usize>,
    purged: bool,
}

fn write_frame(out: &mut Vec<u8>, payload: &[u8]) {
    out.extend_from_slice(
        &u32::try_from(payload.len())
            .expect("csr payload fits u32")
            .to_le_bytes(),
    );
    out.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    out.extend_from_slice(payload);
}

fn le_u32(data: &[u8], at: usize) -> Option<u32> {
    let bytes: [u8; 4] = data.get(at..at.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn le_u16(data: &[u8], at: usize) -> Option<u16> {
    let bytes: [u8; 2] = data.get(at..at.checked_add(2)?)?.try_into().ok()?;
    Some(u16::from_le_bytes(bytes))
}

fn decode_frames(data: &[u8]) -> (Vec<Frame>, usize) {
    let mut frames = Vec::new();
    let mut at = 0usize;
    while at.saturating_add(8) <= data.len() {
        let (Some(len), Some(crc)) = (le_u32(data, at), le_u32(data, at.saturating_add(4))) else {
            break;
        };
        let len = len as usize;
        let start = at.saturating_add(8);
        let Some(end) = start.checked_add(len) else {
            break;
        };
        if end > data.len() {
            break;
        }
        let payload = &data[start..end];
        if crc32fast::hash(payload) != crc {
            break;
        }
        frames.push(Frame {
            payload: payload.to_vec(),
        });
        at = end;
    }
    (frames, at)
}

fn encode_row(
    scope: ScopeId,
    family: EdgeFamily,
    owner: u32,
    edges: &[Edge],
    purge: bool,
) -> io::Result<Vec<u8>> {
    let count = u32::try_from(edges.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "csr row has too many edges"))?;
    let mut out =
        Vec::with_capacity(RECORD_HEADER.saturating_add(edges.len().saturating_mul(EDGE_SIZE)));
    out.extend_from_slice(&scope.0);
    out.extend_from_slice(&family.id.to_le_bytes());
    out.push(u8::from(family.typed));
    out.push(u8::from(purge));
    out.extend_from_slice(&owner.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    for edge in edges {
        out.extend_from_slice(&edge.target.to_le_bytes());
        out.extend_from_slice(&edge.kind.to_le_bytes());
    }
    Ok(out)
}

fn decode_row(payload: &[u8]) -> io::Result<(ScopeId, EdgeFamily, u32, bool, Vec<Edge>)> {
    if payload.len() < RECORD_HEADER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "csr row is shorter than its header",
        ));
    }
    let mut scope = [0u8; 16];
    scope.copy_from_slice(&payload[..16]);
    let family = u16::from_le_bytes([payload[16], payload[17]]);
    let typed = payload[18] != 0;
    let purge = payload[19] != 0;
    let owner = u32::from_le_bytes([payload[20], payload[21], payload[22], payload[23]]);
    let count = u32::from_le_bytes([payload[24], payload[25], payload[26], payload[27]]) as usize;
    let expected = RECORD_HEADER
        .checked_add(count.saturating_mul(EDGE_SIZE))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "csr row length overflows"))?;
    if payload.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "csr row length does not match its edge count",
        ));
    }
    // The length check above makes the body exactly `count` whole edges.
    let edges: Vec<Edge> = payload[RECORD_HEADER..]
        .as_chunks::<EDGE_SIZE>()
        .0
        .iter()
        .map(|edge| Edge {
            target: u32::from_le_bytes([edge[0], edge[1], edge[2], edge[3]]),
            kind: u16::from_le_bytes([edge[4], edge[5]]),
        })
        .collect();
    Ok((
        ScopeId(scope),
        EdgeFamily { id: family, typed },
        owner,
        purge,
        edges,
    ))
}

fn apply_payload(payload: &[u8], rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>) -> io::Result<()> {
    let (scope, family, owner, purge, edges) = decode_row(payload)?;
    if purge {
        rows.retain(|(row_scope, _, _), _| *row_scope != scope.0);
    } else {
        rows.insert((scope.0, family.id, owner), (family.typed, edges));
    }
    Ok(())
}

/// Apply only `scope`'s block from one sealed segment.
///
/// The segment carries a small scope index, so a read seeks straight to its
/// own rows and never decodes the other scopes sharing the file.
fn apply_segment(
    path: &Path,
    scope: ScopeId,
    wanted: Option<&HashSet<(u16, u32)>>,
    family: Option<EdgeFamily>,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    let (mut file, index_bytes, payload_start, payload_len) = open_segment(path)?;
    let corrupt = |what: &str| segment_error(path, what);
    for entry in index_bytes.as_chunks::<SCOPE_ENTRY>().0 {
        if entry[..16] != scope.0 {
            continue;
        }
        let flags = u32::from_le_bytes([entry[16], entry[17], entry[18], entry[19]]);
        let block_off = u32::from_le_bytes([entry[20], entry[21], entry[22], entry[23]]) as usize;
        let block_len = u32::from_le_bytes([entry[24], entry[25], entry[26], entry[27]]) as usize;
        let directory_len =
            u32::from_le_bytes([entry[28], entry[29], entry[30], entry[31]]) as usize;
        if block_off
            .checked_add(block_len)
            .is_none_or(|end| end > payload_len)
        {
            return Err(corrupt("scope block out of range"));
        }
        if directory_len < BLOCK_COUNT || directory_len > block_len {
            return Err(corrupt("scope directory out of range"));
        }
        if flags & SCOPE_FLAG_PURGE != 0 {
            rows.retain(|(row_scope, _, _), _| *row_scope != scope.0);
        }
        let data_len = block_len.saturating_sub(directory_len);
        let directory_start = payload_start.saturating_add(block_off);
        file.seek(SeekFrom::Start(directory_start as u64))?;
        let mut count_bytes = [0u8; BLOCK_COUNT];
        file.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;
        if BLOCK_COUNT.saturating_add(count.saturating_mul(DIR_ENTRY)) != directory_len {
            return Err(corrupt("scope directory length does not match its count"));
        }
        let entries_start = directory_start.saturating_add(BLOCK_COUNT);
        let data_start = directory_start.saturating_add(directory_len);
        if let Some(wanted) = wanted {
            // Sorted directory: probe it on disk instead of reading it.
            for &(family, owner) in wanted {
                if let Some((offset, len, crc)) =
                    find_directory_entry(&mut file, entries_start, count, family, owner)?
                {
                    read_row_at(&mut file, data_start, data_len, offset, len, crc, rows)?;
                }
            }
        } else if let Some(family) = family {
            read_family_directory(
                &mut file,
                entries_start,
                count,
                data_start,
                data_len,
                family.id,
                rows,
            )?;
        } else {
            read_whole_directory(&mut file, entries_start, count, data_start, data_len, rows)
                .map_err(|e| match e.kind() {
                    io::ErrorKind::UnexpectedEof => corrupt("scope directory entry is short"),
                    _ => e,
                })?;
        }
        return Ok(());
    }
    Ok(())
}

fn segment_error(path: &Path, what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("csr segment {} ({what})", path.display()),
    )
}

/// Open a sealed segment and validate its header and length: the file, the
/// raw scope index, where the payload starts and how long it is.
fn open_segment(path: &Path) -> io::Result<(File, Vec<u8>, usize, usize)> {
    let corrupt = |what: &str| segment_error(path, what);
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut header = [0u8; SEGMENT_HEADER];
    file.read_exact(&mut header)?;
    if header[..4] != SEGMENT_MAGIC {
        return Err(corrupt("bad magic"));
    }
    if header[4] != SEGMENT_VERSION {
        return Err(corrupt("unsupported version"));
    }
    let scope_count = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;
    let index_len = scope_count
        .checked_mul(SCOPE_ENTRY)
        .ok_or_else(|| corrupt("index overflows"))?;
    let mut index_bytes = vec![0u8; index_len];
    file.read_exact(&mut index_bytes)?;
    let payload_len = u32::from_le_bytes([header[13], header[14], header[15], header[16]]) as usize;
    let payload_start = SEGMENT_HEADER
        .checked_add(index_len)
        .ok_or_else(|| corrupt("index overflows"))?;
    if (payload_start as u64)
        .checked_add(payload_len as u64)
        .and_then(|end| end.checked_add(4))
        != Some(file_len)
    {
        return Err(corrupt("length does not match file size"));
    }
    Ok((file, index_bytes, payload_start, payload_len))
}

/// Read every row of one scope directory.
fn read_whole_directory(
    file: &mut File,
    entries_start: usize,
    count: usize,
    data_start: usize,
    data_len: usize,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    file.seek(SeekFrom::Start(entries_start as u64))?;
    let mut directory = vec![0u8; count.saturating_mul(DIR_ENTRY)];
    file.read_exact(&mut directory)?;
    for item in directory.as_chunks::<DIR_ENTRY>().0 {
        let (Some(offset), Some(len), Some(crc)) =
            (le_u32(item, 6), le_u32(item, 10), le_u32(item, 14))
        else {
            return Err(io::ErrorKind::UnexpectedEof.into());
        };
        read_row_at(
            file,
            data_start,
            data_len,
            offset as usize,
            len as usize,
            crc,
            rows,
        )?;
    }
    Ok(())
}

/// Read only the contiguous directory range for `family_id`.
fn read_family_directory(
    file: &mut File,
    entries_start: usize,
    count: usize,
    data_start: usize,
    data_len: usize,
    family_id: u16,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    let lower = family_bound(file, entries_start, count, family_id, false)?;
    let upper = family_bound(file, entries_start, count, family_id, true)?;
    if lower == upper {
        return Ok(());
    }
    let byte_len = upper.saturating_sub(lower).saturating_mul(DIR_ENTRY);
    let mut directory = vec![0u8; byte_len];
    file.seek(SeekFrom::Start(
        entries_start.saturating_add(lower.saturating_mul(DIR_ENTRY)) as u64,
    ))?;
    file.read_exact(&mut directory)?;
    for item in directory.as_chunks::<DIR_ENTRY>().0 {
        let (Some(offset), Some(len), Some(crc)) =
            (le_u32(item, 6), le_u32(item, 10), le_u32(item, 14))
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "csr family directory entry is truncated",
            ));
        };
        read_row_at(
            file,
            data_start,
            data_len,
            offset as usize,
            len as usize,
            crc,
            rows,
        )?;
    }
    Ok(())
}

/// Lower or upper bound of `family_id` in the sorted scope directory.
fn family_bound(
    file: &mut File,
    entries_start: usize,
    count: usize,
    family_id: u16,
    upper: bool,
) -> io::Result<usize> {
    let mut low = 0usize;
    let mut high = count;
    let mut entry = [0u8; DIR_ENTRY];
    while low < high {
        let mid = low.saturating_add(high.saturating_sub(low) / 2);
        file.seek(SeekFrom::Start(
            entries_start.saturating_add(mid.saturating_mul(DIR_ENTRY)) as u64,
        ))?;
        file.read_exact(&mut entry)?;
        let current = le_u16(&entry, 0).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "csr directory entry is truncated",
            )
        })?;
        if current < family_id || (upper && current == family_id) {
            low = mid.saturating_add(1);
        } else {
            high = mid;
        }
    }
    Ok(low)
}

/// Binary-search a sorted directory for one `(family, owner)` entry.
fn find_directory_entry(
    file: &mut File,
    entries_start: usize,
    count: usize,
    family: u16,
    owner: u32,
) -> io::Result<Option<(usize, usize, u32)>> {
    use std::cmp::Ordering;
    let mut low = 0usize;
    let mut high = count;
    let mut entry = [0u8; DIR_ENTRY];
    while low < high {
        let mid = low.saturating_add(high.saturating_sub(low) / 2);
        file.seek(SeekFrom::Start(
            entries_start.saturating_add(mid.saturating_mul(DIR_ENTRY)) as u64,
        ))?;
        file.read_exact(&mut entry)?;
        let found = (
            u16::from_le_bytes([entry[0], entry[1]]),
            u32::from_le_bytes([entry[2], entry[3], entry[4], entry[5]]),
        );
        match found.cmp(&(family, owner)) {
            Ordering::Less => low = mid.saturating_add(1),
            Ordering::Greater => high = mid,
            Ordering::Equal => {
                return Ok(Some((
                    u32::from_le_bytes([entry[6], entry[7], entry[8], entry[9]]) as usize,
                    u32::from_le_bytes([entry[10], entry[11], entry[12], entry[13]]) as usize,
                    u32::from_le_bytes([entry[14], entry[15], entry[16], entry[17]]),
                )));
            }
        }
    }
    Ok(None)
}

fn read_row_at(
    file: &mut File,
    data_start: usize,
    data_len: usize,
    offset: usize,
    len: usize,
    crc: u32,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    let corrupt = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_owned());
    if offset.checked_add(len).is_none_or(|end| end > data_len) {
        return Err(corrupt("csr row out of range"));
    }
    file.seek(SeekFrom::Start(data_start.saturating_add(offset) as u64))?;
    let mut row = vec![0u8; len];
    file.read_exact(&mut row)?;
    if crc32fast::hash(&row) != crc {
        return Err(corrupt("csr row checksum mismatch"));
    }
    apply_payload(&row, rows).map_err(|_| corrupt("malformed csr row"))
}

/// Flush the directory's entries. The engine's helper opens the handle in the
/// way each platform allows; a plain `File::open` on a directory fails with
/// "Access is denied" on Windows.
fn sync_dir(dir: &Path) -> io::Result<()> {
    crate::shard::sync_directory(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_root(name: &str) -> PathBuf {
        // The counter keeps two tests that pick the same name from deleting
        // each other's directory when they run in parallel.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "mtxdb-csr-store-{name}-{}-{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn layout(name: &str) -> DatabaseLayout {
        DatabaseLayout::open(unique_root(name)).unwrap()
    }

    fn row(target: u32, kind: u16) -> RowRecord {
        RowRecord {
            family: EdgeFamily::typed(3),
            owner: 1,
            edges: vec![Edge { target, kind }],
        }
    }

    #[test]
    fn writer_is_tied_to_the_store_owner() {
        let root = unique_root("writer");
        let db = crate::database::Database::open(root.clone()).expect("open database");
        let store = CsrStore::open(db.layout(), ShardType::State);
        assert!(store.writer(&db).is_ok(), "the owning database may write");

        // A database rooted inside this store's root keeps its own lock, which
        // does not cover the parent. The old path-prefix check accepted this.
        let foreign = DatabaseLayout::open(root.join("nested")).expect("nested layout");
        let foreign = CsrStore::open(&foreign, ShardType::State);
        assert!(
            foreign.writer(&db).is_err(),
            "a database rooted elsewhere must not authorize this store"
        );

        drop(db);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_layouts_share_the_physical_store_lock() {
        let root = unique_root("symlink-owner");
        let alias = root.with_extension("alias");
        let layout = DatabaseLayout::open(root.clone()).unwrap();
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        let alias_layout = DatabaseLayout::open(alias.clone()).unwrap();
        let real = CsrStore::open(&layout, ShardType::State);
        let linked = CsrStore::open(&alias_layout, ShardType::State);
        assert_eq!(real.dir(), linked.dir());
        assert!(Arc::ptr_eq(&real.store_lock(), &linked.store_lock()));
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_file(alias);
    }

    #[test]
    fn appends_and_reads_one_scope() {
        let layout = layout("roundtrip");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([7; 16]);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        let data = store.read_scope(scope).unwrap();
        assert_eq!(data.rows.len(), 1);
        assert_eq!(data.rows[0].owner, 1);
        assert_eq!(data.rows[0].edges[0].target, 2);
        assert_eq!(data.rows[0].family.id, 3);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn scopes_are_isolated_in_one_store() {
        let layout = layout("scopes");
        let store = CsrStore::open(&layout, ShardType::State);
        let a = ScopeId([1; 16]);
        let b = ScopeId([2; 16]);
        store.append_rows(a, &[row(2, 1)]).unwrap();
        store.append_rows(b, &[row(3, 4)]).unwrap();
        assert_eq!(store.read_scope(a).unwrap().rows.len(), 1);
        assert_eq!(store.read_scope(b).unwrap().rows.len(), 1);
        assert_eq!(store.read_scope(a).unwrap().rows[0].edges[0].target, 2);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn seal_folds_the_tail_and_survives_reopen() {
        let layout = layout("seal");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([9; 16]);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        store.seal().unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 1);
        // Reopen the store: only the segment holds the row.
        let reopened = CsrStore::open(&layout, ShardType::State);
        assert_eq!(reopened.read_scope(scope).unwrap().rows.len(), 1);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn a_sealed_segment_serves_each_scope_independently() {
        let layout = layout("sealed-scopes");
        let store = CsrStore::open(&layout, ShardType::State);
        let a = ScopeId([1; 16]);
        let b = ScopeId([2; 16]);
        store.append_rows(a, &[row(2, 1)]).unwrap();
        store.append_rows(b, &[row(3, 4)]).unwrap();
        store.seal().unwrap();
        assert_eq!(store.read_scope(a).unwrap().rows[0].edges[0].target, 2);
        assert_eq!(store.read_scope(b).unwrap().rows[0].edges[0].target, 3);
        assert!(store.read_scope(ScopeId([3; 16])).unwrap().rows.is_empty());
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn point_reads_use_the_sealed_directory() {
        let layout = layout("point");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([1; 16]);
        store
            .append_rows(
                scope,
                &[
                    RowRecord {
                        family: EdgeFamily::typed(3),
                        owner: 1,
                        edges: vec![Edge { target: 2, kind: 1 }],
                    },
                    RowRecord {
                        family: EdgeFamily::plain(1),
                        owner: 2,
                        edges: Vec::new(),
                    },
                ],
            )
            .unwrap();
        store.seal().unwrap();
        let rows = store
            .read_rows(
                scope,
                &[
                    (EdgeFamily::typed(3), 1),
                    (EdgeFamily::plain(1), 2),
                    (EdgeFamily::plain(1), 9),
                ],
            )
            .unwrap();
        assert_eq!(rows[0].as_ref().unwrap()[0].target, 2);
        assert!(rows[1].as_ref().unwrap().is_empty(), "recorded empty row");
        assert!(rows[2].is_none(), "unrecorded row");
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn family_reads_select_only_that_family_from_segments_and_tail() {
        let layout = layout("family-read");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([8; 16]);
        store
            .append_rows(
                scope,
                &[
                    RowRecord {
                        family: EdgeFamily::typed(3),
                        owner: 1,
                        edges: vec![Edge { target: 2, kind: 1 }],
                    },
                    RowRecord {
                        family: EdgeFamily::plain(1),
                        owner: 2,
                        edges: vec![Edge { target: 4, kind: 0 }],
                    },
                ],
            )
            .unwrap();
        store.seal().unwrap();
        store
            .append_rows(
                scope,
                &[RowRecord {
                    family: EdgeFamily::typed(3),
                    owner: 5,
                    edges: vec![Edge { target: 6, kind: 2 }],
                }],
            )
            .unwrap();

        let rows = store.read_family(scope, EdgeFamily::typed(3)).unwrap();
        assert_eq!(
            rows,
            vec![
                (1, vec![Edge { target: 2, kind: 1 }]),
                (5, vec![Edge { target: 6, kind: 2 }])
            ]
        );
        let plain = store.read_family(scope, EdgeFamily::plain(1)).unwrap();
        assert_eq!(plain, vec![(2, vec![Edge { target: 4, kind: 0 }])]);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn tail_reads_follow_appends_a_seal_and_a_purge() {
        let layout = layout("tail-cache");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([5; 16]);
        let other = ScopeId([6; 16]);
        let family = EdgeFamily::typed(3);
        let point = |owner| {
            store
                .read_rows(scope, &[(family, owner)])
                .unwrap()
                .remove(0)
                .map(|edges| edges[0].target)
        };

        assert_eq!(point(1), None);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        assert_eq!(point(1), Some(2));
        // Another scope's append must not leak into this scope's reads.
        store.append_rows(other, &[row(3, 1)]).unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 1);
        // A second append is picked up without re-decoding the first.
        let second = RowRecord {
            owner: 2,
            ..row(4, 0)
        };
        store.append_rows(scope, &[second]).unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 2);
        // Sealing empties the tail; the rows must survive in the segment.
        store.seal().unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 2);
        assert_eq!(point(2), Some(4));
        // A tombstone in the new tail hides the sealed rows, only for its scope.
        store.purge_scope(scope).unwrap();
        assert_eq!(point(1), None);
        assert!(store.read_scope(scope).unwrap().rows.is_empty());
        assert_eq!(store.read_scope(other).unwrap().rows.len(), 1);
        // Rows recorded after the purge are visible again.
        store.append_rows(scope, &[row(7, 1)]).unwrap();
        assert_eq!(point(1), Some(7));
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn purge_hides_sealed_rows() {
        let layout = layout("purge");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([4; 16]);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        store.seal().unwrap();
        store.purge_scope(scope).unwrap();
        assert!(store.read_scope(scope).unwrap().rows.is_empty());
        store.seal().unwrap();
        assert!(store.read_scope(scope).unwrap().rows.is_empty());
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn a_purge_that_crosses_the_threshold_seals_the_tail() {
        let layout = layout("purge_seals");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([8; 16]);
        let frame = |owner, edges: &[Edge], purge| {
            let payload = encode_row(scope, EdgeFamily::typed(3), owner, edges, purge).unwrap();
            u64::try_from(payload.len()).unwrap() + 8
        };
        let tombstone = frame(0, &[], true);
        let empty_row = frame(1, &[], false);
        // One row sized so the tail sits just under the threshold, and one
        // tombstone then carries it over.
        let edges_needed =
            (SEAL_THRESHOLD - tombstone - empty_row).div_ceil(u64::try_from(EDGE_SIZE).unwrap());
        let edges = vec![Edge { target: 0, kind: 0 }; usize::try_from(edges_needed).unwrap()];
        store
            .append_rows(
                scope,
                &[RowRecord {
                    family: EdgeFamily::typed(3),
                    owner: 1,
                    edges,
                }],
            )
            .unwrap();
        let tail_len = || fs::metadata(store.tail_path()).map_or(0, |meta| meta.len());
        assert!(
            tail_len() > 0 && tail_len() < SEAL_THRESHOLD,
            "setup is under the threshold"
        );
        assert_eq!(
            store.segments().unwrap().len(),
            0,
            "the setup must not seal a segment"
        );

        store.purge_scope(scope).unwrap();
        assert_eq!(tail_len(), 0, "the purge seals the tail");
        assert_eq!(store.segments().unwrap().len(), 1);
        assert!(store.read_scope(scope).unwrap().rows.is_empty());
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn a_writer_needs_the_database_that_owns_the_store() {
        let owner_root = layout("writer_owner").root().to_path_buf();
        let other_root = layout("writer_other").root().to_path_buf();
        {
            let owner = Database::open(owner_root.clone()).unwrap();
            let other = Database::open(other_root.clone()).unwrap();
            let store = CsrStore::open(owner.layout(), ShardType::State);
            let scope = ScopeId([9; 16]);

            let refused = store
                .writer(&other)
                .err()
                .expect("a foreign database is refused");
            assert_eq!(refused.kind(), io::ErrorKind::InvalidInput);

            let writer = store.writer(&owner).unwrap();
            writer.append_rows(scope, &[row(2, 1)]).unwrap();
            assert_eq!(store.read_scope(scope).unwrap().rows.len(), 1);
            writer.purge_scope(scope).unwrap();
            assert!(store.read_scope(scope).unwrap().rows.is_empty());
        }
        let _ = fs::remove_dir_all(owner_root);
        let _ = fs::remove_dir_all(other_root);
    }

    #[test]
    fn concurrent_writers_on_one_database_lose_no_rows() {
        const THREADS: u32 = 4;
        const PER_THREAD: u32 = 150;
        let root = layout("writer_threads").root().to_path_buf();
        {
            let db = Database::open(root.clone()).unwrap();
            let store = CsrStore::open(db.layout(), ShardType::State);
            std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (store, db) = (&store, &db);
                    scope.spawn(move || {
                        let writer = store.writer(db).unwrap();
                        let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
                        for owner in 0..PER_THREAD {
                            let row = RowRecord {
                                family: EdgeFamily::typed(3),
                                owner,
                                edges: vec![
                                    Edge {
                                        target: owner,
                                        kind: 1
                                    };
                                    64
                                ],
                            };
                            writer.append_rows(scope_id, &[row]).unwrap();
                        }
                    });
                }
            });
            for thread in 0..THREADS {
                let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
                let rows = store.read_scope(scope_id).unwrap().rows;
                assert_eq!(
                    rows.len(),
                    usize::try_from(PER_THREAD).unwrap(),
                    "scope {thread} lost rows to a concurrent writer"
                );
            }
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_direct_mutations_serialize_on_the_store() {
        const THREADS: u32 = 4;
        const PER_THREAD: u32 = 150;
        let layout = layout("direct_writer_threads");
        let store = CsrStore::open(&layout, ShardType::State);
        for thread in 0..THREADS {
            let scope_id = ScopeId([u8::try_from(thread + 10).unwrap(); 16]);
            store.append_rows(scope_id, &[row(1, 1)]).unwrap();
        }
        let purge_barrier = std::sync::Barrier::new(usize::try_from(THREADS).unwrap());
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let (store, purge_barrier) = (&store, &purge_barrier);
                scope.spawn(move || {
                    let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
                    // A failure here must still reach the barrier, or the
                    // other workers would block forever instead of reporting.
                    let appended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        for owner in 0..PER_THREAD {
                            let row = RowRecord {
                                family: EdgeFamily::typed(3),
                                owner,
                                edges: vec![
                                    Edge {
                                        target: owner,
                                        kind: 1
                                    };
                                    64
                                ],
                            };
                            store.append_rows(scope_id, &[row]).unwrap();
                            // Seal while the other threads are still appending.
                            if owner % 50 == 49 {
                                store.seal().unwrap();
                            }
                        }
                    }));
                    purge_barrier.wait();
                    if let Err(panic) = appended {
                        std::panic::resume_unwind(panic);
                    }
                    let purge_scope = ScopeId([u8::try_from(thread + 10).unwrap(); 16]);
                    store.purge_scope(purge_scope).unwrap();
                });
            }
        });
        for thread in 0..THREADS {
            let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
            assert_eq!(
                store.read_scope(scope_id).unwrap().rows.len(),
                usize::try_from(PER_THREAD).unwrap(),
                "scope {thread} lost rows to a direct concurrent writer"
            );
        }
        for thread in 0..THREADS {
            let purge_scope = ScopeId([u8::try_from(thread + 10).unwrap(); 16]);
            assert!(store.read_scope(purge_scope).unwrap().rows.is_empty());
        }
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn a_torn_tail_frame_is_dropped() {
        let layout = layout("torn");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([5; 16]);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        let mut bytes = fs::read(store.tail_path()).unwrap();
        bytes.extend_from_slice(&[0xff; 3]);
        fs::write(store.tail_path(), &bytes).unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 1);
        // A later append repairs the boundary instead of writing past garbage.
        store.append_rows(scope, &[row(6, 1)]).unwrap();
        let data = store.read_scope(scope).unwrap();
        assert_eq!(data.rows.len(), 1);
        assert_eq!(data.rows[0].edges[0].target, 6);
        let _ = fs::remove_dir_all(layout.root());
    }
}
