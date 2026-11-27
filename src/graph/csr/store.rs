//! Pool-scoped CSR graph store outside the packs.
//!
//! There is one store per source pool at `<db_root>/mtgr-<pool>/`, shared by
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock, Weak};
use std::time::{Instant, SystemTime};

use super::{Edge, EdgeFamily};
use crate::database::Database;
use crate::layout::{DatabaseLayout, ShardType};

const SEGMENT_MAGIC: [u8; 4] = *b"MCSS";
const SEGMENT_VERSION: u8 = 1;
/// Read retries allowed while another process seals segments under a reader.
///
/// Writers in this process are excluded by the store lock, so only a foreign
/// writer can keep moving the segment generation. Bounded so a reader reports
/// contention instead of spinning forever.
const MAX_READ_ATTEMPTS: usize = 16;
const SEGMENT_PREFIX: &str = "seg-";
const SEGMENT_SUFFIX: &str = ".csr";
/// Upper bound on one buffered read of a family's contiguous row span. A row
/// larger than this is still read whole, so the buffer is at most one row over.
#[cfg(not(test))]
const FAMILY_READ_CHUNK: usize = 4 * 1024 * 1024;
/// Tests use a tiny cap so a handful of rows exercises several chunks.
#[cfg(test)]
const FAMILY_READ_CHUNK: usize = 512;
const TAIL_FILE: &str = "tail.log";

/// Seal once the tail reaches this many bytes.
pub const SEAL_THRESHOLD: u64 = 4 * 1024 * 1024;

const RECORD_HEADER: usize = 16 + 2 + 1 + 1 + 4 + 4;
const EDGE_SIZE: usize = 4 + 2;
/// Segment header: magic, version, scope count, record count, payload length.
const SEGMENT_HEADER: usize = 4 + 1 + 4 + 4 + 4;
/// One scope index entry: scope, flags, block offset, block length, directory
/// length, directory checksum. The directory is `(family, owner) -> row`.
///
/// The directory checksum lets a point read verify the entries it probed
/// without reading the scope's row data, which each row checksums itself.
const SCOPE_ENTRY: usize = 16 + 4 + 4 + 4 + 4 + 4;
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

    /// Append and sync while the caller holds [`CsrStore::with_write_lock`].
    /// Only tests need the sync-inside-the-lock form; production appends use
    /// [`Self::append_rows_deferred_locked`] and sync after releasing the lock.
    #[cfg(test)]
    pub(crate) fn append_rows_locked(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        self.store.append_rows_locked(scope, rows, true)
    }

    /// Append without syncing the tail, while the caller holds
    /// [`CsrStore::with_write_lock`]. The rows are readable at once but are
    /// not durable until [`CsrStore::sync_tail`] returns; the caller must not
    /// publish any claim that covers them before then.
    pub(crate) fn append_rows_deferred_locked(
        &self,
        scope: ScopeId,
        rows: &[RowRecord],
    ) -> io::Result<()> {
        self.store.append_rows_locked(scope, rows, false)
    }

    /// Hide every row of `scope` with a tombstone record.
    ///
    /// # Errors
    /// I/O errors.
    pub fn purge_scope(&self, scope: ScopeId) -> io::Result<()> {
        self.store.purge_scope(scope)
    }

    /// Append a purge while the caller already holds the store's write lock.
    pub(crate) fn purge_scope_locked(&self, scope: ScopeId) -> io::Result<()> {
        self.store.purge_scope_locked(scope)
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

#[cfg(test)]
thread_local! {
    /// Tail syncs issued by this thread; tests assert on the count.
    static TAIL_SYNCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Whether this thread's last tail sync ran while the store's write lock
    /// was held (by any thread). Tests assert a durable append releases the
    /// lock before it syncs, so overlapping readers never wait on the flush.
    static LAST_SYNC_HELD_LOCK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn count_tail_sync() {
    TAIL_SYNCS.with(|count| count.set(count.get().saturating_add(1)));
}

/// Tail syncs issued by the current thread (test builds only).
#[cfg(test)]
pub(crate) fn test_tail_syncs() -> usize {
    TAIL_SYNCS.with(std::cell::Cell::get)
}

/// Whether the current thread's most recent tail sync ran under the store's
/// write lock (test builds only).
#[cfg(test)]
pub(crate) fn test_last_sync_held_lock() -> bool {
    LAST_SYNC_HELD_LOCK.with(std::cell::Cell::get)
}

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
        let waited = Instant::now();
        let _guard = lock.read().unwrap_or_else(PoisonError::into_inner);
        READ_LOCK_WAITS.fetch_add(1, Ordering::Relaxed);
        add_nanos(&READ_LOCK_WAIT_NANOS, waited);
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
    /// The rows are written under the store's write lock and the tail is
    /// synced after releasing it: holding the lock across `fsync` would stall
    /// every overlapping reader for the length of the flush. This returns only
    /// after the rows are durable.
    ///
    /// Empty input does nothing and does not create the store directory.
    ///
    /// # Errors
    /// I/O errors, or a row with more than `u32::MAX` edges.
    pub(crate) fn append_rows(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let lock = self.store_lock();
        {
            let _guard = lock.write().unwrap_or_else(PoisonError::into_inner);
            self.append_rows_locked(scope, rows, false)?;
        }
        self.sync_tail()
    }

    /// Make the tail's contents durable. Its directory entry was made durable
    /// when the tail was created. A store that has no tail yet has nothing to
    /// sync. Safe to call when nothing is pending.
    ///
    /// # Errors
    /// I/O errors.
    pub(crate) fn sync_tail(&self) -> io::Result<()> {
        // FlushFileBuffers on Windows requires a handle opened for writing;
        // a read-only File::open handle makes sync_data fail with
        // ERROR_ACCESS_DENIED even though the tail itself is writable.
        match OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.tail_path())
        {
            Ok(file) => {
                #[cfg(test)]
                let _trace = crate::test_trace::span("csr.tail_sync", &self.dir);
                #[cfg(test)]
                LAST_SYNC_HELD_LOCK.with(|held| held.set(self.store_lock().try_write().is_err()));
                file.sync_data()?;
                #[cfg(test)]
                count_tail_sync();
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        }
        Ok(())
    }

    /// Append while the caller holds this directory's write lock. `sync`
    /// makes the tail durable before returning; sealing stays durable either
    /// way, because it syncs the segment before it truncates the tail.
    fn append_rows_locked(&self, scope: ScopeId, rows: &[RowRecord], sync: bool) -> io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let created_dir = !self.dir.is_dir();
        fs::create_dir_all(&self.dir)?;
        let created_tail = !self.tail_path().exists();
        let valid = self.tail_valid()?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.tail_path())?;
        // A synced file is not reachable after a crash until its directory
        // entry is durable. Entries change only when something is created, so
        // sync the directory then, not on every append.
        if created_tail {
            sync_dir(&self.dir)?;
        }
        if created_dir {
            if let Some(parent) = self.dir.parent() {
                sync_dir(parent)?;
            }
        }
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
        if sync {
            #[cfg(test)]
            let _trace = crate::test_trace::span("csr.tail_sync", &self.dir);
            file.sync_data()?;
            #[cfg(test)]
            count_tail_sync();
        }
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
    pub(crate) fn purge_scope_locked(&self, scope: ScopeId) -> io::Result<()> {
        self.purge_scopes_locked(&[scope])
    }

    /// Append tombstones for several scopes and sync the tail once, while the
    /// caller holds this directory's write lock.
    pub(crate) fn purge_scopes_locked(&self, scopes: &[ScopeId]) -> io::Result<()> {
        if scopes.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.tail_path())?;
        let valid = self.tail_valid()?;
        let len = file.metadata()?.len();
        if valid < len {
            file.set_len(valid)?;
        }
        file.seek(SeekFrom::End(0))?;
        let mut buffer = Vec::new();
        for scope in scopes {
            let payload = encode_row(*scope, EdgeFamily::plain(0), 0, &[], true)?;
            write_frame(&mut buffer, &payload);
        }
        file.write_all(&buffer)?;
        file.sync_data()?;
        // Tombstones grow the tail like rows do, so a purge-only workload must
        // seal at the same threshold or the tail and its cache grow unbounded.
        self.maybe_seal()?;
        Ok(())
    }

    /// Every scope with rows in sealed segments or the verified tail.
    pub(crate) fn all_scopes(&self) -> io::Result<std::collections::BTreeSet<[u8; 16]>> {
        self.with_read_lock(|| {
            let mut scopes = std::collections::BTreeSet::new();
            for (_, path) in self.segments()? {
                let (_, index_bytes, _, _) = open_segment(&path)?;
                for entry in index_bytes.as_chunks::<SCOPE_ENTRY>().0 {
                    let mut scope = [0_u8; 16];
                    scope.copy_from_slice(&entry[..16]);
                    scopes.insert(scope);
                }
            }
            let (frames, _) = self.read_tail()?;
            for frame in &frames {
                let (scope, _, _, _, _) = decode_row(&frame.payload)?;
                scopes.insert(scope.0);
            }
            Ok(scopes)
        })
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
        self.replace_scopes_locked(&[])
    }

    /// Seal, replacing each listed scope's rows, while the caller holds this
    /// directory's write lock. All listed scopes reach the directory together:
    /// the new segment is published by one rename, so a crash before it leaves
    /// every listed scope as it was.
    ///
    /// That is not the whole guarantee. The tail truncate below is unsynced,
    /// and the process can stop between the rename and the truncate. Either
    /// way the old tail survives, and it is applied after sealed segments, so
    /// its rows override the new segment: rows the replacement deleted can
    /// reappear, while rows that only lived in older sealed segments stay
    /// replaced. The state is mixed, not old-or-new.
    ///
    /// Recovery is therefore mandatory before the store serves reads or writes.
    /// Writes matter as much as reads: a reused short id could inherit a
    /// resurrected orphan row. Recovery re-filters the existing rows against
    /// the recovered id limits rather than rebuilding them from the log, so
    /// running it again converges. It is not automatic: a caller must invoke it
    /// after opening a writable store.
    ///
    /// The tail truncate is not synced here: a caller replacing rows must sync
    /// the tail after.
    pub(crate) fn replace_scopes_locked(
        &self,
        replacements: &[(ScopeId, Vec<RowRecord>)],
    ) -> io::Result<()> {
        if !self.dir.is_dir() {
            return Ok(());
        }
        let (mut records, valid) = self.read_tail()?;
        // A tombstone then the survivors, grouped like tail records but never
        // written to the tail, where a crash could keep only some of them.
        for (scope, rows) in replacements {
            let payload = encode_row(*scope, EdgeFamily::plain(0), 0, &[], true)?;
            records.push(Frame { payload });
            for row in rows {
                let payload = encode_row(*scope, row.family, row.owner, &row.edges, false)?;
                records.push(Frame { payload });
            }
        }
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
        if tail.exists() {
            OpenOptions::new().write(true).open(&tail)?.set_len(0)?;
        }
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
        let mut attempts = 0_usize;
        let rows = loop {
            attempts = attempts.saturating_add(1);
            if attempts > MAX_READ_ATTEMPTS {
                return Err(read_contention(&self.dir));
            }
            let segments = self.segments()?;
            let generation = generation_of(&segments);
            let mut rows: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
            for (_, path) in &segments {
                apply_segment(path, scope, None, None, &mut rows)?;
            }
            let Some(()) = self.with_tail(generation, |tail| {
                // A tombstone in the tail is newer than every sealed segment.
                if tail.purged.contains(&scope.0) {
                    rows.clear();
                }
                for (key, value) in &tail.rows {
                    if key.0 == scope.0 {
                        rows.insert(*key, value.clone());
                    }
                }
            })?
            else {
                continue;
            };
            break rows;
        };
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
        #[cfg(test)]
        let _trace = crate::test_trace::span("csr.read_rows_with_lock", self.dir());
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
        let mut attempts = 0_usize;
        let found = loop {
            attempts = attempts.saturating_add(1);
            if attempts > MAX_READ_ATTEMPTS {
                return Err(read_contention(&self.dir));
            }
            let segments = self.segments()?;
            let generation = generation_of(&segments);
            let mut found: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
            for (_, path) in &segments {
                apply_segment(path, scope, Some(&wanted), None, &mut found)?;
            }
            let Some(()) = self.with_tail(generation, |tail| {
                if tail.purged.contains(&scope.0) {
                    found.clear();
                }
                for &(family, owner) in &wanted {
                    let key = (scope.0, family, owner);
                    if let Some(value) = tail.rows.get(&key) {
                        found.insert(key, value.clone());
                    }
                }
            })?
            else {
                continue;
            };
            break found;
        };
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

    pub(crate) fn read_family_locked(
        &self,
        scope: ScopeId,
        family: EdgeFamily,
    ) -> io::Result<Vec<(u32, Vec<Edge>)>> {
        let mut attempts = 0_usize;
        let rows = loop {
            attempts = attempts.saturating_add(1);
            if attempts > MAX_READ_ATTEMPTS {
                return Err(read_contention(&self.dir));
            }
            let segments = self.segments()?;
            let generation = generation_of(&segments);
            let mut rows: HashMap<RowKey, (bool, Vec<Edge>)> = HashMap::new();
            for (_, path) in &segments {
                apply_segment(path, scope, None, Some(family), &mut rows)?;
            }
            let Some(()) = self.with_tail(generation, |tail| {
                if tail.purged.contains(&scope.0) {
                    rows.clear();
                }
                for (&(row_scope, row_family, owner), value) in &tail.rows {
                    if row_scope == scope.0 && row_family == family.id {
                        rows.insert((row_scope, row_family, owner), value.clone());
                    }
                }
            })?
            else {
                continue;
            };
            break rows;
        };
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

    /// The sealed segments in order, listing the directory only when it may
    /// have changed.
    ///
    /// A listing is reused while the directory's identity and modification
    /// time match the cached ones, and only if the directory was already
    /// quiet for [`SEGMENT_CACHE_QUIET`] when it was listed. The quiet margin
    /// covers coarse timestamps: a seal landing in the same timestamp tick as
    /// the listing would otherwise leave a stale list that still matches.
    fn segments(&self) -> io::Result<Vec<(u64, PathBuf)>> {
        let stamp = dir_stamp(&self.dir);
        if let Some(stamp) = &stamp {
            let cache = SEGMENT_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(entry) = cache.get(&self.dir) {
                if entry.stamp == *stamp && entry.quiet {
                    return Ok(entry.segments.clone());
                }
            }
        }
        // Judged before listing: the quiet margin must already have held when
        // the listing began, not become true while it ran.
        let quiet = stamp.as_ref().is_some_and(|stamp| {
            SystemTime::now()
                .duration_since(stamp.modified)
                .is_ok_and(|age| age >= SEGMENT_CACHE_QUIET)
        });
        let started = Instant::now();
        let listed = self.list_segments();
        SEGMENT_LISTINGS.fetch_add(1, Ordering::Relaxed);
        add_nanos(&SEGMENT_LISTING_NANOS, started);
        #[cfg(test)]
        {
            let mut counts = SEGMENT_LISTINGS_BY_DIR
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let count = counts.entry(self.dir.clone()).or_default();
            *count = count.saturating_add(1);
        }
        let segments = listed?;
        // Stamp taken before the listing: a change during the listing makes
        // the stamp differ on the next read, never the other way round.
        if let Some(stamp) = stamp {
            SEGMENT_CACHE
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(
                    self.dir.clone(),
                    SegmentCacheEntry {
                        stamp,
                        quiet,
                        segments: segments.clone(),
                    },
                );
        }
        Ok(segments)
    }

    fn list_segments(&self) -> io::Result<Vec<(u64, PathBuf)>> {
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
    /// discards the cache. `generation` is the newest sealed segment the caller
    /// observed; `None` means a seal intervened, so the caller must reread the
    /// segment list and try again.
    fn with_tail<R>(
        &self,
        generation: Option<u64>,
        read: impl FnOnce(&TailState) -> R,
    ) -> io::Result<Option<R>> {
        // Reject a seal that landed after the caller sampled the segments.
        if self.segment_generation()? != generation {
            return Ok(None);
        }
        let len = match fs::metadata(self.tail_path()) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error),
        };
        // One shared entry per directory, locked for the read. A reader that
        // arrives while another holds it waits for a short extension instead of
        // finding no entry and re-decoding the whole tail from byte 0.
        let entry = Arc::clone(
            TAIL_CACHE
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(self.dir.clone())
                .or_default(),
        );
        let mut state = entry.lock().unwrap_or_else(PoisonError::into_inner);
        if state.valid > len || state.generation != generation {
            *state = TailState::default();
        }
        state.generation = generation;
        TAIL_READS.fetch_add(1, Ordering::Relaxed);
        if state.valid < len {
            let decoding = Instant::now();
            let from_start = state.valid == 0;
            if from_start {
                TAIL_FULL_DECODE_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(test)]
            if state.valid == 0 {
                let mut counts = TAIL_FULL_DECODES
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let count = counts.entry(self.dir.clone()).or_default();
                *count = count.saturating_add(1);
            }
            let (frames, valid) = self.read_tail_from(state.valid)?;
            for frame in &frames {
                state.apply(&frame.payload)?;
            }
            state.valid = state.valid.saturating_add(valid as u64);
            TAIL_DECODED_BYTES.fetch_add(valid as u64, Ordering::Relaxed);
            add_nanos(&TAIL_DECODE_NANOS, decoding);
        }
        // A seal publishes a segment before truncating the tail. If its
        // sequence changed while we loaded the tail, this view is stale.
        if self.segment_generation()? != generation {
            return Ok(None);
        }
        Ok(Some(read(&state)))
    }

    fn segment_generation(&self) -> io::Result<Option<u64>> {
        Ok(generation_of(&self.segments()?))
    }

    /// The tail prefix verified under the current segment generation.
    fn tail_valid(&self) -> io::Result<u64> {
        let mut attempts = 0_usize;
        loop {
            attempts = attempts.saturating_add(1);
            if attempts > MAX_READ_ATTEMPTS {
                return Err(read_contention(&self.dir));
            }
            let generation = self.segment_generation()?;
            if let Some(valid) = self.with_tail(generation, |tail| tail.valid)? {
                return Ok(valid);
            }
        }
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

/// The newest sealed segment sequence, used to tell one tail incarnation from
/// the next. A seal always publishes its segment before it truncates the tail.
fn generation_of(segments: &[(u64, PathBuf)]) -> Option<u64> {
    segments.last().map(|(sequence, _)| *sequence)
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
/// and a checksum over the index.
///
/// Rows carry their own checksums in the scope directory, so a point read
/// trusts the bytes it read without touching the rest of the payload.
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
        index.extend_from_slice(&crc32fast::hash(&directory).to_le_bytes());
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
    file_bytes.extend_from_slice(&crc32fast::hash(&index).to_le_bytes());
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
    /// Highest sealed segment observed when this prefix was decoded.
    generation: Option<u64>,
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

/// Process-wide read-path counters for the CSR stores, so a slow forward read
/// can be attributed to lock wait, tail decoding or directory listings.
static READ_LOCK_WAITS: AtomicU64 = AtomicU64::new(0);
static READ_LOCK_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static TAIL_READS: AtomicU64 = AtomicU64::new(0);
static TAIL_FULL_DECODE_COUNT: AtomicU64 = AtomicU64::new(0);
static TAIL_DECODED_BYTES: AtomicU64 = AtomicU64::new(0);
static TAIL_DECODE_NANOS: AtomicU64 = AtomicU64::new(0);
static SEGMENT_LISTINGS: AtomicU64 = AtomicU64::new(0);
static SEGMENT_LISTING_NANOS: AtomicU64 = AtomicU64::new(0);

fn add_nanos(counter: &AtomicU64, started: Instant) {
    counter.fetch_add(
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// A snapshot of the CSR read-path counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadStats {
    /// Store read-lock acquisitions.
    pub read_lock_waits: u64,
    /// Total nanoseconds spent acquiring the store read lock.
    pub read_lock_wait_nanos: u64,
    /// Tail reads (`with_tail` calls that reached the cache).
    pub tail_reads: u64,
    /// Tail reads that had to decode from byte 0.
    pub tail_full_decodes: u64,
    /// Bytes decoded from the tail, extensions included.
    pub tail_decoded_bytes: u64,
    /// Total nanoseconds spent decoding the tail.
    pub tail_decode_nanos: u64,
    /// `read_dir` listings of a store directory.
    pub segment_listings: u64,
    /// Total nanoseconds spent listing store directories.
    pub segment_listing_nanos: u64,
}

/// Snapshot the CSR read-path counters (cumulative since process start).
#[must_use]
pub fn read_stats() -> ReadStats {
    ReadStats {
        read_lock_waits: READ_LOCK_WAITS.load(Ordering::Relaxed),
        read_lock_wait_nanos: READ_LOCK_WAIT_NANOS.load(Ordering::Relaxed),
        tail_reads: TAIL_READS.load(Ordering::Relaxed),
        tail_full_decodes: TAIL_FULL_DECODE_COUNT.load(Ordering::Relaxed),
        tail_decoded_bytes: TAIL_DECODED_BYTES.load(Ordering::Relaxed),
        tail_decode_nanos: TAIL_DECODE_NANOS.load(Ordering::Relaxed),
        segment_listings: SEGMENT_LISTINGS.load(Ordering::Relaxed),
        segment_listing_nanos: SEGMENT_LISTING_NANOS.load(Ordering::Relaxed),
    }
}

/// How long a store directory must have been unchanged before a listing of it
/// is trusted without listing again.
const SEGMENT_CACHE_QUIET: std::time::Duration = std::time::Duration::from_secs(2);

/// Identity and modification time of a store directory. Creating, renaming or
/// removing an entry in it changes the modification time.
#[derive(Clone, PartialEq, Eq)]
struct DirStamp {
    modified: SystemTime,
    #[cfg(unix)]
    inode: u64,
}

struct SegmentCacheEntry {
    stamp: DirStamp,
    /// The directory had been unchanged for [`SEGMENT_CACHE_QUIET`] when it
    /// was listed, so the stamp is a safe validator.
    quiet: bool,
    segments: Vec<(u64, PathBuf)>,
}

fn dir_stamp(dir: &Path) -> Option<DirStamp> {
    let meta = fs::metadata(dir).ok()?;
    if !meta.is_dir() {
        return None;
    }
    Some(DirStamp {
        modified: meta.modified().ok()?,
        #[cfg(unix)]
        inode: std::os::unix::fs::MetadataExt::ino(&meta),
    })
}

static SEGMENT_CACHE: LazyLock<Mutex<HashMap<PathBuf, SegmentCacheEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Tests count directory listings per store.
#[cfg(test)]
static SEGMENT_LISTINGS_BY_DIR: LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Tests count reads that decoded a tail from byte 0.
#[cfg(test)]
static TAIL_FULL_DECODES: LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static TAIL_CACHE: LazyLock<Mutex<HashMap<PathBuf, Arc<Mutex<TailState>>>>> =
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

fn decode_frames(data: &[u8]) -> (Vec<Frame>, usize) {
    let mut frames = Vec::new();
    let mut at = 0usize;
    while at.saturating_add(8) <= data.len() {
        let (Some(len), Some(crc)) = (le_u32(data, at), le_u32(data, at.saturating_add(4))) else {
            break;
        };
        let len = len as usize;
        // Every real payload carries a row header, so a zero-length frame is
        // an unwritten hole, not a record. The CRC-32 of nothing is 0, so a
        // zero-filled gap would otherwise pass the checksum as a run of empty
        // frames and poison every later read.
        if len == 0 {
            break;
        }
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

/// One scope's verified directory inside a sealed segment.
///
/// Sealed segments are immutable, so this is the unit the optional directory
/// cache keeps: reading it means reading and checksumming the whole directory.
struct ScopeDir {
    purge: bool,
    /// The directory entries, checksum-verified, without the count prefix.
    entries: Vec<u8>,
    data_start: usize,
    data_len: usize,
}

/// Locate `scope` in a segment's index and read and verify its directory.
/// `None` when the segment holds no block for the scope.
fn read_scope_dir(
    path: &Path,
    file: &mut File,
    index_bytes: &[u8],
    payload_start: usize,
    payload_len: usize,
    scope: ScopeId,
) -> io::Result<Option<ScopeDir>> {
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
        let directory_crc = u32::from_le_bytes([entry[32], entry[33], entry[34], entry[35]]);
        if block_off
            .checked_add(block_len)
            .is_none_or(|end| end > payload_len)
        {
            return Err(corrupt("scope block out of range"));
        }
        if directory_len < BLOCK_COUNT || directory_len > block_len {
            return Err(corrupt("scope directory out of range"));
        }
        let data_len = block_len.saturating_sub(directory_len);
        let directory_start = payload_start.saturating_add(block_off);
        file.seek(SeekFrom::Start(directory_start as u64))?;
        let mut directory = vec![0u8; directory_len];
        file.read_exact(&mut directory)?;
        if crc32fast::hash(&directory) != directory_crc {
            return Err(corrupt("scope directory checksum mismatch"));
        }
        let count = u32::from_le_bytes(
            directory
                .get(..BLOCK_COUNT)
                .ok_or_else(|| corrupt("scope directory is short"))?
                .try_into()
                .map_err(|_| corrupt("scope directory is short"))?,
        ) as usize;
        if BLOCK_COUNT.saturating_add(count.saturating_mul(DIR_ENTRY)) != directory_len {
            return Err(corrupt("scope directory length does not match its count"));
        }
        directory.drain(..BLOCK_COUNT);
        return Ok(Some(ScopeDir {
            purge: flags & SCOPE_FLAG_PURGE != 0,
            entries: directory,
            data_start: directory_start.saturating_add(directory_len),
            data_len,
        }));
    }
    Ok(None)
}

/// Apply one scope's directory: honor its purge flag, then read the wanted
/// rows, the family's rows or every row.
fn apply_scope_dir(
    file: &mut File,
    dir: &ScopeDir,
    scope: ScopeId,
    wanted: Option<&HashSet<(u16, u32)>>,
    family: Option<EdgeFamily>,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    if dir.purge {
        rows.retain(|(row_scope, _, _), _| *row_scope != scope.0);
    }
    let entries = &dir.entries;
    if let Some(wanted) = wanted {
        // The directory is sorted, so probe it in memory.
        for &(family, owner) in wanted {
            if let Some((offset, len, crc)) = find_directory_entry(entries, family, owner) {
                read_row_at(file, dir.data_start, dir.data_len, offset, len, crc, rows)?;
            }
        }
    } else if let Some(family) = family {
        read_family_directory(file, entries, dir.data_start, dir.data_len, family.id, rows)?;
    } else {
        read_whole_directory(file, entries, dir.data_start, dir.data_len, rows)?;
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
    if wanted.is_some() {
        return dir_cache::apply_point(path, scope, wanted, rows);
    }
    if let Some(family) = family {
        return dir_cache::apply_family(path, scope, family, rows);
    }
    let (mut file, index_bytes, payload_start, payload_len) = open_segment(path)?;
    if let Some(dir) = read_scope_dir(
        path,
        &mut file,
        &index_bytes,
        payload_start,
        payload_len,
        scope,
    )? {
        apply_scope_dir(&mut file, &dir, scope, wanted, family, rows)?;
    }
    Ok(())
}

/// Cache of verified scope directories for point reads.
///
/// A point read otherwise reads and checksums each segment's whole scope
/// directory on every lookup, so its cost grows with the total rows stored.
/// Sealed segments never change, so the directory is read once and reused. An
/// entry is trusted only while the segment file still has the identity (size,
/// mtime and inode) it had when cached, which also covers a store directory
/// deleted and recreated under the same path. The cache is bounded: past
/// [`CAP_BYTES`] it is cleared and refilled by later reads.
mod dir_cache {
    use super::{
        apply_scope_dir, open_segment, read_scope_dir, Edge, File, HashMap, HashSet, LazyLock,
        Mutex, Path, PathBuf, PoisonError, RowKey, ScopeDir, ScopeId,
    };
    use std::io;
    use std::sync::Arc;
    use std::time::SystemTime;

    /// Clear the cache once cached directories exceed this many bytes.
    const CAP_BYTES: usize = 256 * 1024 * 1024;

    #[derive(PartialEq, Eq, Clone, Copy)]
    struct Stamp {
        len: u64,
        modified: Option<SystemTime>,
        inode: u64,
    }

    type CacheKey = (PathBuf, [u8; 16]);
    type CacheValue = (Stamp, Option<Arc<ScopeDir>>);

    fn stamp_of(file: &File) -> io::Result<Stamp> {
        let meta = file.metadata()?;
        #[cfg(unix)]
        let inode = std::os::unix::fs::MetadataExt::ino(&meta);
        #[cfg(not(unix))]
        let inode = 0;
        Ok(Stamp {
            len: meta.len(),
            modified: meta.modified().ok(),
            inode,
        })
    }

    #[derive(Default)]
    struct Cache {
        map: HashMap<CacheKey, CacheValue>,
        bytes: usize,
    }

    static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(Cache::default()));

    /// What one cache entry costs. A present directory keeps its entry count;
    /// an absent one still holds its key and an `Option`, so it is never free
    /// (otherwise a scan over many absent scopes would grow the map unbounded).
    fn dir_charge(key: &CacheKey, dir: Option<&Arc<ScopeDir>>) -> usize {
        dir.map_or_else(
            || key.0.as_os_str().len().saturating_add(64),
            |dir| dir.entries.len().max(1),
        )
    }

    /// Open `path` and return its cached verified directory, if the segment
    /// still has the identity the cached entry was built from.
    fn cached_dir(path: &Path, scope: ScopeId) -> io::Result<(File, Option<Arc<ScopeDir>>)> {
        let file = File::open(path)?;
        let stamp = stamp_of(&file)?;
        let key = (path.to_path_buf(), scope.0);
        let hit = CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .get(&key)
            .filter(|(cached, _)| *cached == stamp)
            .map(|(_, dir)| dir.clone());
        let dir = if let Some(dir) = hit {
            dir
        } else {
            let (mut opened, index_bytes, payload_start, payload_len) = open_segment(path)?;
            let dir = read_scope_dir(
                path,
                &mut opened,
                &index_bytes,
                payload_start,
                payload_len,
                scope,
            )?
            .map(Arc::new);
            let size = dir_charge(&key, dir.as_ref());
            let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
            if cache.bytes.saturating_add(size) > CAP_BYTES {
                cache.map.clear();
                cache.bytes = 0;
            }
            if let Some((_, old)) = cache.map.insert(key.clone(), (stamp, dir.clone())) {
                cache.bytes = cache.bytes.saturating_sub(dir_charge(&key, old.as_ref()));
            }
            cache.bytes = cache.bytes.saturating_add(size);
            dir
        };
        Ok((file, dir))
    }

    pub(super) fn apply_point(
        path: &Path,
        scope: ScopeId,
        wanted: Option<&HashSet<(u16, u32)>>,
        rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
    ) -> io::Result<()> {
        let (mut file, dir) = cached_dir(path, scope)?;
        if let Some(dir) = dir {
            apply_scope_dir(&mut file, &dir, scope, wanted, None, rows)?;
        }
        Ok(())
    }

    /// Family reads reuse the verified immutable scope directory. Without
    /// this cache every sparse family read rereads and checksums entries for
    /// all the other families in the scope.
    pub(super) fn apply_family(
        path: &Path,
        scope: ScopeId,
        family: super::EdgeFamily,
        rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
    ) -> io::Result<()> {
        let (mut file, dir) = cached_dir(path, scope)?;
        if let Some(dir) = dir {
            apply_scope_dir(&mut file, &dir, scope, None, Some(family), rows)?;
        }
        Ok(())
    }
}

fn segment_error(path: &Path, what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("csr segment {} ({what})", path.display()),
    )
}

/// The reader lost `MAX_READ_ATTEMPTS` races against a writer that keeps
/// sealing segments. Reported as `WouldBlock` so a caller may retry.
fn read_contention(dir: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::WouldBlock,
        format!(
            "csr store {} is sealing continuously; retry the read",
            dir.display()
        ),
    )
}

/// Open a sealed segment and validate its header, index checksum and length.
///
/// The payload is not read here. Each row is checksummed in its directory
/// entry, so a point read verifies the row it reads instead of the segment.
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
    let checksum_at = (payload_start as u64)
        .checked_add(payload_len as u64)
        .ok_or_else(|| corrupt("payload overflows"))?;
    file.seek(SeekFrom::Start(checksum_at))?;
    let mut stored_crc = [0u8; 4];
    file.read_exact(&mut stored_crc)?;
    if crc32fast::hash(&index_bytes) != u32::from_le_bytes(stored_crc) {
        return Err(corrupt("index checksum mismatch"));
    }
    file.seek(SeekFrom::Start(payload_start as u64))?;
    Ok((file, index_bytes, payload_start, payload_len))
}

/// Read every row of one scope directory.
fn read_whole_directory(
    file: &mut File,
    entries: &[u8],
    data_start: usize,
    data_len: usize,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    for item in entries.as_chunks::<DIR_ENTRY>().0 {
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

/// Read every row whose directory entry belongs to `family_id`.
fn read_family_directory(
    file: &mut File,
    entries: &[u8],
    data_start: usize,
    data_len: usize,
    family_id: u16,
    rows: &mut HashMap<RowKey, (bool, Vec<Edge>)>,
) -> io::Result<()> {
    let directory = entries.as_chunks::<DIR_ENTRY>().0;
    // Entries are encoded in `(family, owner)` order. Find this family's
    // contiguous range so an empty family doesn't scan unrelated rows.
    let start =
        directory.partition_point(|item| u16::from_le_bytes([item[0], item[1]]) < family_id);
    let end = directory.partition_point(|item| u16::from_le_bytes([item[0], item[1]]) <= family_id);
    let selected = &directory[start..end];
    let Some(first) = selected.first() else {
        return Ok(());
    };

    let fields = |item: &[u8; DIR_ENTRY]| -> io::Result<(usize, usize, u32)> {
        let (Some(offset), Some(len), Some(crc)) =
            (le_u32(item, 6), le_u32(item, 10), le_u32(item, 14))
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "csr family directory entry is truncated",
            ));
        };
        Ok((offset as usize, len as usize, crc))
    };

    let (first_offset, first_len, _) = fields(first)?;
    let mut expected_offset = first_offset
        .checked_add(first_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "csr row out of range"))?;
    let mut contiguous = true;
    for item in selected.iter().skip(1) {
        let (offset, len, _) = fields(item)?;
        if offset != expected_offset {
            contiguous = false;
        }
        expected_offset = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "csr row out of range"))?;
    }

    if !contiguous {
        // Preserve the general reader's behavior for any valid future layout
        // whose selected rows are not stored contiguously.
        for item in selected {
            let (offset, len, crc) = fields(item)?;
            read_row_at(file, data_start, data_len, offset, len, crc, rows)?;
        }
        return Ok(());
    }

    if expected_offset > data_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "csr row out of range",
        ));
    }
    // The rows are contiguous, so read them in bounded chunks on row
    // boundaries (a row larger than the chunk is read alone) rather than one
    // allocation for the whole family. Every row is still checksummed.
    let out_of_range = || io::Error::new(io::ErrorKind::InvalidData, "csr row out of range");
    let mut next = 0;
    while next < selected.len() {
        let (chunk_start, first_len, _) = fields(&selected[next])?;
        let mut chunk_end = next.saturating_add(1);
        let mut span = first_len;
        while chunk_end < selected.len() {
            let (_, len, _) = fields(&selected[chunk_end])?;
            let prospective = span.checked_add(len).ok_or_else(out_of_range)?;
            if prospective > FAMILY_READ_CHUNK {
                break;
            }
            span = prospective;
            chunk_end = chunk_end.saturating_add(1);
        }
        let mut batch = vec![0_u8; span];
        let absolute_start = data_start
            .checked_add(chunk_start)
            .ok_or_else(out_of_range)?;
        file.seek(SeekFrom::Start(absolute_start as u64))?;
        file.read_exact(&mut batch)?;
        for item in &selected[next..chunk_end] {
            let (offset, len, crc) = fields(item)?;
            let relative_start = offset.checked_sub(chunk_start).ok_or_else(out_of_range)?;
            let relative_end = relative_start.checked_add(len).ok_or_else(out_of_range)?;
            let row = batch
                .get(relative_start..relative_end)
                .ok_or_else(out_of_range)?;
            if crc32fast::hash(row) != crc {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "csr row checksum mismatch",
                ));
            }
            apply_payload(row, rows)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "malformed csr row"))?;
        }
        next = chunk_end;
    }
    Ok(())
}

/// Binary-search a sorted in-memory directory for one `(family, owner)` entry.
fn find_directory_entry(entries: &[u8], family: u16, owner: u32) -> Option<(usize, usize, u32)> {
    use std::cmp::Ordering;
    let mut low = 0usize;
    let mut high = entries.len() / DIR_ENTRY;
    while low < high {
        let mid = low.saturating_add(high.saturating_sub(low) / 2);
        let item = &entries[mid.saturating_mul(DIR_ENTRY)..][..DIR_ENTRY];
        let found = (
            u16::from_le_bytes([item[0], item[1]]),
            u32::from_le_bytes([item[2], item[3], item[4], item[5]]),
        );
        match found.cmp(&(family, owner)) {
            Ordering::Less => low = mid.saturating_add(1),
            Ordering::Greater => high = mid,
            Ordering::Equal => {
                return Some((
                    u32::from_le_bytes([item[6], item[7], item[8], item[9]]) as usize,
                    u32::from_le_bytes([item[10], item[11], item[12], item[13]]) as usize,
                    u32::from_le_bytes([item[14], item[15], item[16], item[17]]),
                ));
            }
        }
    }
    None
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
        let root = crate::test_scratch::root().join(format!(
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

    fn tail_syncs() -> usize {
        TAIL_SYNCS.with(std::cell::Cell::get)
    }

    fn row_for(owner: u32, target: u32) -> RowRecord {
        RowRecord {
            family: EdgeFamily::typed(3),
            owner,
            edges: vec![Edge { target, kind: 1 }],
        }
    }

    #[test]
    fn durable_appends_sync_once_per_call() {
        let layout = layout("durable-count");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([3; 16]);
        let before = tail_syncs();
        for owner in 1..=3 {
            store.append_rows(scope, &[row_for(owner, 9)]).unwrap();
        }
        assert_eq!(tail_syncs() - before, 3);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn durable_append_syncs_after_releasing_the_write_lock() {
        let layout = layout("sync-outside-lock");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([5; 16]);
        store.append_rows(scope, &[row_for(1, 9)]).unwrap();
        assert!(
            !test_last_sync_held_lock(),
            "a durable append must release the write lock before it fsyncs"
        );
        // The probe really does see the lock: a sync issued under it reports so.
        store.with_write_lock(|| store.sync_tail()).unwrap();
        assert!(test_last_sync_held_lock());
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn concurrent_readers_share_one_decoded_tail() {
        let layout = layout("shared-tail");
        let store = Arc::new(CsrStore::open(&layout, ShardType::State));
        let scope = ScopeId([7; 16]);
        let batch: Vec<_> = (2..=400).map(|owner| row_for(owner, 9)).collect();
        store
            .with_write_lock(|| store.append_rows_locked(scope, &batch, false))
            .unwrap();
        let family = row_for(1, 9).family;
        // Warm the cache, then count decodes from byte 0 across racing readers.
        store.read_rows(scope, &[(family, 2)]).unwrap();
        let decodes = |dir: &Path| {
            TAIL_FULL_DECODES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(dir)
                .copied()
                .unwrap_or(0)
        };
        let before = decodes(store.dir());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    for owner in 2..300 {
                        let rows = store.read_rows(scope, &[(family, owner)]).unwrap();
                        assert!(rows[0].is_some());
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(
            decodes(store.dir()),
            before,
            "readers must extend one shared decoded tail, not re-decode it"
        );
        let _ = fs::remove_dir_all(layout.root());
    }

    fn listings(dir: &Path) -> usize {
        SEGMENT_LISTINGS_BY_DIR
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(dir)
            .copied()
            .unwrap_or(0)
    }

    #[test]
    fn a_quiet_store_directory_is_listed_once_and_a_seal_is_still_seen() {
        let layout = layout("segment-cache");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([8; 16]);
        store
            .with_write_lock(|| store.append_rows_locked(scope, &[row_for(1, 9)], false))
            .unwrap();
        let family = row_for(1, 9).family;
        // The first read lists and caches; the rest must not list again.
        store.read_rows(scope, &[(family, 1)]).unwrap();
        // Mark the just-listed entry as quiet to exercise cache reuse without
        // sleeping for the production quiet interval.
        SEGMENT_CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_mut(store.dir())
            .unwrap()
            .quiet = true;
        let before = listings(store.dir());
        for _ in 0..50 {
            assert!(store.read_rows(scope, &[(family, 1)]).unwrap()[0].is_some());
        }
        assert_eq!(listings(store.dir()), before, "a quiet directory is cached");
        // A seal changes the directory, so the cache must notice at once and the
        // sealed rows must still be read.
        store.seal().unwrap();
        assert!(store.read_rows(scope, &[(family, 1)]).unwrap()[0].is_some());
        assert!(
            listings(store.dir()) > before,
            "a seal invalidates the cache"
        );
        assert_eq!(store.segments().unwrap().len(), 1);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn a_recently_changed_directory_is_never_trusted_from_the_cache() {
        let layout = layout("segment-cache-racy");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([9; 16]);
        store
            .with_write_lock(|| store.append_rows_locked(scope, &[row_for(1, 9)], false))
            .unwrap();
        store.seal().unwrap();
        // Just changed, so every read lists again instead of risking a stale
        // list inside one timestamp tick.
        let family = row_for(1, 9).family;
        store.read_rows(scope, &[(family, 1)]).unwrap();
        let before = listings(store.dir());
        store.read_rows(scope, &[(family, 1)]).unwrap();
        assert!(listings(store.dir()) > before);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn deferred_appends_sync_only_at_the_barrier() {
        let layout = layout("deferred-count");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([4; 16]);
        let before = tail_syncs();
        for owner in 1..=3 {
            store
                .with_write_lock(|| store.append_rows_locked(scope, &[row_for(owner, 9)], false))
                .unwrap();
        }
        assert_eq!(tail_syncs() - before, 0, "deferred appends must not sync");
        // Readable at once, before any sync.
        assert_eq!(store.read_scope(scope).unwrap().rows.len(), 3);
        store.sync_tail().unwrap();
        assert_eq!(tail_syncs() - before, 1, "the barrier syncs exactly once");
        // A store with no tail has nothing to sync.
        let empty_layout = self::layout("deferred-empty");
        CsrStore::open(&empty_layout, ShardType::State)
            .sync_tail()
            .unwrap();
        let reopened = CsrStore::open(&layout, ShardType::State);
        assert_eq!(reopened.read_scope(scope).unwrap().rows.len(), 3);
        let _ = fs::remove_dir_all(layout.root());
        let _ = fs::remove_dir_all(empty_layout.root());
    }

    #[test]
    fn a_damaged_deferred_tail_replays_only_its_valid_prefix() {
        // A crash can persist later pages while losing earlier ones, and can
        // cut a frame short. Each case gets its own store because the tail
        // cache assumes nothing else edits the file.
        for case in 0..3 {
            let layout = layout(&format!("deferred-torn-{case}"));
            let store = CsrStore::open(&layout, ShardType::State);
            let scope = ScopeId([5; 16]);
            for owner in 1..=3 {
                store
                    .with_write_lock(|| {
                        store.append_rows_locked(scope, &[row_for(owner, 9)], false)
                    })
                    .unwrap();
            }
            store.sync_tail().unwrap();
            drop(store);
            let tail = CsrStore::open(&layout, ShardType::State).tail_path();
            let mut bytes = fs::read(&tail).unwrap();
            let frame = bytes.len() / 3;
            match case {
                0 => bytes.truncate(bytes.len() - 3),
                1 => bytes[frame..2 * frame].fill(0),
                _ => bytes[frame + frame / 2] ^= 0xFF,
            }
            fs::write(&tail, &bytes).unwrap();
            // A restart starts with no decoded tail cache.
            TAIL_CACHE
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&tail.parent().unwrap().to_path_buf());
            let reopened = CsrStore::open(&layout, ShardType::State);
            let owners: Vec<u32> = reopened
                .read_scope(scope)
                .unwrap()
                .rows
                .iter()
                .map(|r| r.owner)
                .collect();
            assert!(owners.len() < 3, "case {case}: damage replayed: {owners:?}");
            assert!(owners.contains(&1), "case {case}: valid prefix lost");
            // Appending after the damage recovers and keeps the prefix.
            reopened.append_rows(scope, &[row_for(7, 9)]).unwrap();
            let after = reopened.read_scope(scope).unwrap().rows;
            assert!(after.iter().any(|r| r.owner == 7));
            assert!(after.iter().any(|r| r.owner == 1));
            let _ = fs::remove_dir_all(layout.root());
        }
    }

    #[test]
    fn a_crash_between_the_seal_rename_and_the_tail_truncate_replays_to_the_same_rows() {
        let layout = layout("seal-window");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([6; 16]);
        for owner in 1..=3 {
            store
                .with_write_lock(|| {
                    store.append_rows_locked(scope, &[row_for(owner, 10 + owner)], false)
                })
                .unwrap();
        }
        store.sync_tail().unwrap();
        let tail = store.tail_path();
        let full_tail = fs::read(&tail).unwrap();
        let resolved = |store: &CsrStore| {
            let mut rows: Vec<(u32, Vec<(u32, u16)>)> = store
                .read_scope(scope)
                .unwrap()
                .rows
                .iter()
                .map(|r| {
                    (
                        r.owner,
                        r.edges.iter().map(|e| (e.target, e.kind)).collect(),
                    )
                })
                .collect();
            rows.sort();
            rows
        };
        let before = resolved(&store);
        assert_eq!(before.len(), 3);
        store.seal().unwrap();
        // The segment is durable but the truncate was lost: the old frames are
        // back in the tail next to the sealed segment holding the same rows.
        fs::write(&tail, &full_tail).unwrap();
        TAIL_CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&store.dir);
        let reopened = CsrStore::open(&layout, ShardType::State);
        assert_eq!(
            resolved(&reopened),
            before,
            "replay must resolve to the same rows"
        );
        // Re-importing the same rows changes nothing, and a later seal keeps it so.
        for owner in 1..=3 {
            reopened
                .append_rows(scope, &[row_for(owner, 10 + owner)])
                .unwrap();
        }
        reopened.seal().unwrap();
        assert_eq!(resolved(&reopened), before);
        let _ = fs::remove_dir_all(layout.root());
    }

    /// Rows for three families in one scope: a typed family with edges, a plain
    /// family with edges for some owners, and a plain family of empty rows.
    fn mixed_family_rows() -> Vec<RowRecord> {
        let mut rows = Vec::new();
        for owner in 1..=200_u32 {
            rows.push(RowRecord {
                family: EdgeFamily::typed(3),
                owner,
                edges: vec![
                    Edge {
                        target: owner.saturating_add(1),
                        kind: 1,
                    },
                    Edge {
                        target: owner.saturating_add(2),
                        kind: 2,
                    },
                ],
            });
            if owner % 2 == 0 {
                rows.push(RowRecord {
                    family: EdgeFamily::plain(1),
                    owner,
                    edges: vec![Edge {
                        target: owner / 2,
                        kind: 0,
                    }],
                });
            }
            rows.push(RowRecord {
                family: EdgeFamily::plain(2),
                owner,
                edges: Vec::new(),
            });
        }
        rows
    }

    fn sorted_family(
        store: &CsrStore,
        scope: ScopeId,
        family: EdgeFamily,
    ) -> Vec<(u32, Vec<(u32, u16)>)> {
        let mut rows: Vec<(u32, Vec<(u32, u16)>)> = store
            .read_family(scope, family)
            .unwrap()
            .into_iter()
            .map(|(owner, edges)| {
                (
                    owner,
                    edges.into_iter().map(|e| (e.target, e.kind)).collect(),
                )
            })
            .collect();
        rows.sort();
        rows
    }

    #[test]
    fn family_read_matches_point_reads_across_families() {
        let layout = layout("family-batched");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([9; 16]);
        let rows = mixed_family_rows();
        for chunk in rows.chunks(37) {
            store.append_rows(scope, chunk).unwrap();
        }
        // Once from the tail, then again from a sealed segment.
        for sealed in [false, true] {
            if sealed {
                store.seal().unwrap();
            }
            for family in [
                EdgeFamily::typed(3),
                EdgeFamily::plain(1),
                EdgeFamily::plain(2),
            ] {
                let expected: Vec<(u32, Vec<(u32, u16)>)> = {
                    let mut e: Vec<_> = rows
                        .iter()
                        .filter(|r| r.family.id == family.id)
                        .map(|r| {
                            (
                                r.owner,
                                r.edges.iter().map(|e| (e.target, e.kind)).collect(),
                            )
                        })
                        .collect();
                    e.sort();
                    e
                };
                let got = sorted_family(&store, scope, family);
                assert_eq!(got, expected, "family {} sealed={sealed}", family.id);
                // A point read of every owner agrees with the family read.
                let keys: Vec<(EdgeFamily, u32)> =
                    expected.iter().map(|(owner, _)| (family, *owner)).collect();
                let points = store.read_rows(scope, &keys).unwrap();
                for ((owner, edges), point) in expected.iter().zip(points) {
                    let point: Vec<(u32, u16)> = point
                        .unwrap_or_default()
                        .into_iter()
                        .map(|e| (e.target, e.kind))
                        .collect();
                    assert_eq!(&point, edges, "owner {owner} family {}", family.id);
                }
            }
            // A family with no rows reads as empty, not as an error.
            assert_eq!(
                store.read_family(scope, EdgeFamily::plain(9)).unwrap(),
                Vec::new()
            );
        }
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn family_read_rejects_a_corrupted_sealed_row() {
        let layout = layout("family-batched-corrupt");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([10; 16]);
        let rows: Vec<RowRecord> = mixed_family_rows()
            .into_iter()
            .filter(|r| r.family.id == EdgeFamily::typed(3).id)
            .collect();
        store.append_rows(scope, &rows).unwrap();
        store.seal().unwrap();
        assert_eq!(
            sorted_family(&store, scope, EdgeFamily::typed(3)).len(),
            200
        );
        let segment = store.segments().unwrap().pop().unwrap().1;
        let mut bytes = fs::read(&segment).unwrap();
        let at = bytes.len() * 2 / 3;
        bytes[at] ^= 0xFF;
        fs::write(&segment, &bytes).unwrap();
        let error = store
            .read_family(scope, EdgeFamily::typed(3))
            .expect_err("a flipped byte in a sealed row must not read back");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let _ = fs::remove_dir_all(layout.root());
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
        assert!(
            store.read_scope(ScopeId([3; 16])).unwrap().rows.is_empty(),
            "value should be empty"
        );
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
        assert!(
            store.read_scope(scope).unwrap().rows.is_empty(),
            "value should be empty"
        );
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
        assert!(
            store.read_scope(scope).unwrap().rows.is_empty(),
            "value should be empty"
        );
        store.seal().unwrap();
        assert!(
            store.read_scope(scope).unwrap().rows.is_empty(),
            "value should be empty"
        );
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
        assert!(
            store.read_scope(scope).unwrap().rows.is_empty(),
            "value should be empty"
        );
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
            assert!(
                store.read_scope(scope).unwrap().rows.is_empty(),
                "value should be empty"
            );
        }
        let _ = fs::remove_dir_all(owner_root);
        let _ = fs::remove_dir_all(other_root);
    }

    #[test]
    fn concurrent_writers_on_one_database_lose_no_rows() {
        const THREADS: u32 = 4;
        const PER_THREAD: u32 = 150;
        // Per-call durable appends cost an fsync each, so the bulk goes through
        // the deferred path (the same lock and tail, no per-append sync) and a
        // few durable appends per thread keep that path concurrent too.
        const DURABLE: u32 = 4;
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
                        let make_row = |owner: u32| RowRecord {
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
                        for owner in 0..PER_THREAD {
                            store
                                .with_write_lock(|| {
                                    writer.append_rows_deferred_locked(scope_id, &[make_row(owner)])
                                })
                                .unwrap();
                        }
                        let before = tail_syncs();
                        for owner in PER_THREAD..PER_THREAD.saturating_add(DURABLE) {
                            writer.append_rows(scope_id, &[make_row(owner)]).unwrap();
                        }
                        assert_eq!(
                            tail_syncs().saturating_sub(before),
                            usize::try_from(DURABLE).unwrap(),
                            "each durable append syncs the tail once"
                        );
                    });
                }
            });
            // The deferred rows are readable already; make them durable too.
            store.sync_tail().unwrap();
            for thread in 0..THREADS {
                let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
                let rows = store.read_scope(scope_id).unwrap().rows;
                assert_eq!(
                    rows.len(),
                    usize::try_from(PER_THREAD.saturating_add(DURABLE)).unwrap(),
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
        // The bulk goes through the deferred append path: the same per-store
        // write lock and tail as a durable append, without one fsync per call.
        // `concurrent_durable_appends_sync_their_own_tail` keeps the durable
        // path concurrent.
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
                            store
                                .with_write_lock(|| {
                                    store.append_rows_locked(scope_id, &[row], false)
                                })
                                .unwrap();
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
        // Deferred rows are readable already; make them durable too.
        store.sync_tail().unwrap();
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
            assert!(
                store.read_scope(purge_scope).unwrap().rows.is_empty(),
                "value should be empty"
            );
        }
        let _ = fs::remove_dir_all(layout.root());
    }

    /// The per-call durable path under concurrency: every thread's append syncs
    /// the tail once, and no row is lost. Small on purpose: each append is an
    /// fsync.
    #[test]
    fn concurrent_durable_appends_sync_their_own_tail() {
        const THREADS: u32 = 4;
        const PER_THREAD: u32 = 6;
        let layout = layout("durable_writer_threads");
        let store = CsrStore::open(&layout, ShardType::State);
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let store = &store;
                scope.spawn(move || {
                    let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
                    let before = tail_syncs();
                    for owner in 0..PER_THREAD {
                        store.append_rows(scope_id, &[row_for(owner, 9)]).unwrap();
                    }
                    assert_eq!(
                        tail_syncs().saturating_sub(before),
                        usize::try_from(PER_THREAD).unwrap(),
                        "each durable append syncs the tail once"
                    );
                });
            }
        });
        for thread in 0..THREADS {
            let scope_id = ScopeId([u8::try_from(thread).unwrap(); 16]);
            assert_eq!(
                store.read_scope(scope_id).unwrap().rows.len(),
                usize::try_from(PER_THREAD).unwrap(),
                "scope {thread} lost rows to a durable concurrent writer"
            );
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

    #[test]
    fn a_damaged_segment_index_or_directory_is_rejected() {
        for damaged in ["index", "directory"] {
            let layout = layout(&format!("segment-damage-{damaged}"));
            let store = CsrStore::open(&layout, ShardType::State);
            let scope = ScopeId([11; 16]);
            store.append_rows(scope, &[row(2, 1)]).unwrap();
            store.seal().unwrap();
            let (_, path) = store.segments().unwrap().pop().unwrap();
            let mut bytes = fs::read(&path).unwrap();
            let scope_count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
            let payload_start =
                SEGMENT_HEADER.saturating_add(scope_count.saturating_mul(SCOPE_ENTRY));
            let block_off = u32::from_le_bytes([
                bytes[SEGMENT_HEADER + 20],
                bytes[SEGMENT_HEADER + 21],
                bytes[SEGMENT_HEADER + 22],
                bytes[SEGMENT_HEADER + 23],
            ]) as usize;
            let at = if damaged == "index" {
                SEGMENT_HEADER
            } else {
                payload_start
                    .saturating_add(block_off)
                    .saturating_add(BLOCK_COUNT)
            };
            bytes[at] ^= 0x40;
            fs::write(&path, &bytes).unwrap();
            let error = store.read_scope(scope).unwrap_err();
            assert_eq!(
                error.kind(),
                io::ErrorKind::InvalidData,
                "{damaged} damage must not become a successful miss"
            );
            let _ = fs::remove_dir_all(layout.root());
        }
    }

    #[test]
    fn a_seal_invalidates_a_cached_tail_that_grows_back() {
        let layout = layout("tail-generation");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([12; 16]);
        let family = EdgeFamily::typed(3);
        store.append_rows(scope, &[row_for(1, 10)]).unwrap();
        store.seal().unwrap();
        store.purge_scope(scope).unwrap();
        // Prime the cache with the purge, then seal it into a new segment.
        assert!(
            store.read_scope(scope).unwrap().rows.is_empty(),
            "value should be empty"
        );
        let old_len = fs::metadata(store.tail_path()).unwrap().len();
        store.seal().unwrap();
        // Simulate another process refilling the tail until it is at least as
        // long as the cached prefix. The old purge must not hide this row.
        let edges = vec![Edge { target: 3, kind: 1 }; usize::try_from(old_len).unwrap()];
        let payload = encode_row(scope, family, 2, &edges, false).unwrap();
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &payload);
        assert!(u64::try_from(bytes.len()).unwrap() >= old_len);
        fs::write(store.tail_path(), &bytes).unwrap();
        let rows = store.read_scope(scope).unwrap().rows;
        assert_eq!(rows.len(), 1, "the stale purge hid the refilled tail");
        assert_eq!(rows[0].owner, 2);
        let _ = fs::remove_dir_all(layout.root());
    }
}
