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
//! - [`CsrStore::append_rows`] appends one framed record per changed row and
//!   syncs the tail. Rewriting one row never rewrites the others.
//! - [`CsrStore::seal`] folds the tail into a new immutable `seg-N.csr` file
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
//! Decoded rows are cached in process, one cache per store directory. A read
//! decodes only tail bytes appended since the previous read; each sealed
//! segment is decoded once when it appears, for every scope at once. The cache
//! is in-process state, not a durability boundary. Correctness assumes one
//! writer per directory: the database root lock enforces one writer process,
//! and this cache's lock serializes appends, purges and seals inside it.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::layout::{DatabaseLayout, ShardType};
use crate::short_id::{Edge, EdgeFamily};

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

type RowKey = (u16, u32);
type ScopeRows = HashMap<RowKey, (bool, Vec<Edge>)>;
type CacheRows = HashMap<[u8; 16], ScopeRows>;

/// One store directory's decoded state.
///
/// `signature` identifies the sealed segments; `tail_applied` is the length of
/// the valid `tail.log` prefix already applied. They are tracked separately so
/// a growing tail does not force a segment reload, and a new segment does not
/// force re-reading the tail.
#[derive(Debug, Default)]
struct StoreCache {
    signature: Vec<(u64, u64)>,
    rows: CacheRows,
    tail_applied: u64,
    /// Whether the segment directory has been listed at least once.
    loaded: bool,
    /// Test-only: sealed segments decoded into this cache.
    #[cfg(test)]
    segment_reads: u64,
}

static CACHES: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<StoreCache>>>>> = OnceLock::new();

fn cache_for(dir: &Path) -> Arc<Mutex<StoreCache>> {
    let caches = CACHES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut caches = lock(caches);
    Arc::clone(caches.entry(dir.to_path_buf()).or_default())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl CsrStore {
    /// Address the store for `pool` under `layout`.
    #[must_use]
    pub fn open(layout: &DatabaseLayout, pool: ShardType) -> Self {
        Self {
            dir: layout.graph_path(pool),
        }
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
    pub fn append_rows(&self, scope: ScopeId, rows: &[RowRecord]) -> io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.dir)?;
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache, false)?;
        let valid = cache.tail_applied;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
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
        let sealed = self.maybe_seal()?;
        self.refresh_cache(&mut cache, sealed)?;
        Ok(())
    }

    /// Hide every row of `scope` with a tombstone and sync the tail.
    ///
    /// # Errors
    /// I/O errors.
    pub fn purge_scope(&self, scope: ScopeId) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache, false)?;
        let valid = cache.tail_applied;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.tail_path())?;
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
        self.refresh_cache(&mut cache, false)?;
        Ok(())
    }

    /// Apply the tail as a new immutable segment, then truncate the tail.
    ///
    /// # Errors
    /// I/O errors or a corrupt frame that is not merely torn at the end.
    pub fn seal(&self) -> io::Result<()> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.seal_locked()?;
        self.refresh_cache(&mut cache, true)
    }

    /// Seal without taking the directory cache lock.
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

    /// Seal if the tail is large enough, reporting whether it did.
    fn maybe_seal(&self) -> io::Result<bool> {
        if fs::metadata(self.tail_path()).map(|meta| meta.len())? >= SEAL_THRESHOLD {
            self.seal_locked()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Read every visible row of `scope`.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_scope(&self, scope: ScopeId) -> io::Result<ScopeData> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache, false)?;
        let mut out = Vec::new();
        if let Some(rows) = cache.rows.get(&scope.0) {
            for (&(family, owner), (typed, edges)) in rows {
                out.push(StoredRow {
                    family: EdgeFamily {
                        id: family,
                        typed: *typed,
                    },
                    owner,
                    edges: edges.clone(),
                });
            }
        }
        out.sort_by_key(|row| (row.family.id, row.owner));
        Ok(ScopeData { rows: out })
    }

    /// Read specific `(family, owner)` rows, in the order requested.
    ///
    /// This is the hot path for recording and point reads. Rows come from the
    /// directory cache; `None` means the row is not recorded.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_rows(
        &self,
        scope: ScopeId,
        keys: &[(EdgeFamily, u32)],
    ) -> io::Result<Vec<Option<Vec<Edge>>>> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache, false)?;
        let rows = cache.rows.get(&scope.0);
        Ok(keys
            .iter()
            .map(|(family, owner)| {
                rows.and_then(|rows| rows.get(&(family.id, *owner)))
                    .map(|(_, edges)| edges.clone())
            })
            .collect())
    }

    /// Fold new segments and new tail bytes into the cache.
    fn refresh_cache(&self, cache: &mut StoreCache, force_scan: bool) -> io::Result<()> {
        let tail_len = self.tail_file_len()?;
        // A seal creates a segment and truncates the tail. Writers take this
        // cache's lock, so without a forced scan a changed tail length is the
        // only in-process evidence of a seal; a warm unchanged read skips the
        // directory entirely.
        let rescan = force_scan || !cache.loaded || tail_len < cache.tail_applied;
        if rescan {
            let segments = self.segments()?;
            let signature = segment_signature(&segments)?;
            if signature != cache.signature {
                if signature.len() > cache.signature.len()
                    && signature.starts_with(&cache.signature)
                {
                    for (_, path) in &segments[cache.signature.len()..] {
                        Self::decode_segment(path, cache)?;
                    }
                } else {
                    cache.rows.clear();
                    for (_, path) in &segments {
                        Self::decode_segment(path, cache)?;
                    }
                }
                cache.signature = signature;
                // A seal replaced the tail; replaying it is idempotent if a
                // crash left the old tail in place.
                cache.tail_applied = 0;
            } else if tail_len < cache.tail_applied {
                // The tail was truncated without a new segment: rebuild.
                cache.rows.clear();
                for (_, path) in &segments {
                    Self::decode_segment(path, cache)?;
                }
                cache.tail_applied = 0;
            }
            cache.loaded = true;
        }
        let tail_len = self.tail_file_len()?;
        if tail_len > cache.tail_applied {
            let (frames, valid) = self.read_tail_from(cache.tail_applied)?;
            for frame in &frames {
                apply_payload(&frame.payload, &mut cache.rows)?;
            }
            cache.tail_applied = cache
                .tail_applied
                .saturating_add(u64::try_from(valid).unwrap_or(u64::MAX));
        }
        Ok(())
    }

    fn decode_segment(path: &Path, cache: &mut StoreCache) -> io::Result<()> {
        apply_segment(path, &mut cache.rows)?;
        #[cfg(test)]
        {
            cache.segment_reads = cache.segment_reads.saturating_add(1);
        }
        Ok(())
    }

    fn tail_file_len(&self) -> io::Result<u64> {
        match fs::metadata(self.tail_path()) {
            Ok(meta) => Ok(meta.len()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(error),
        }
    }

    fn next_segment_number(&self) -> u64 {
        self.segments()
            .map(|segments| segments.last().map_or(0, |(seq, _)| seq.saturating_add(1)))
            .unwrap_or(0)
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

    fn read_tail(&self) -> io::Result<(Vec<Frame>, usize)> {
        self.read_tail_from(0)
    }

    fn read_tail_from(&self, offset: u64) -> io::Result<(Vec<Frame>, usize)> {
        let data = match fs::read(self.tail_path()) {
            Ok(data) => data,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok((Vec::new(), 0));
            }
            Err(error) => return Err(error),
        };
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(data.len());
        Ok(decode_frames(&data[start..]))
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
    let mut edges = Vec::with_capacity(count);
    let mut at = RECORD_HEADER;
    for _ in 0..count {
        let (Some(target), Some(kind)) =
            (le_u32(payload, at), le_u16(payload, at.saturating_add(4)))
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "csr row is shorter than its edge count",
            ));
        };
        edges.push(Edge { target, kind });
        at = at.saturating_add(EDGE_SIZE);
    }
    Ok((
        ScopeId(scope),
        EdgeFamily { id: family, typed },
        owner,
        purge,
        edges,
    ))
}

fn apply_payload(payload: &[u8], rows: &mut CacheRows) -> io::Result<()> {
    let (scope, family, owner, purge, edges) = decode_row(payload)?;
    if purge {
        rows.remove(&scope.0);
    } else {
        rows.entry(scope.0)
            .or_default()
            .insert((family.id, owner), (family.typed, edges));
    }
    Ok(())
}

/// Apply every scope block in one sealed segment.
fn apply_segment(path: &Path, rows: &mut CacheRows) -> io::Result<()> {
    let corrupt = |what: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("csr segment {} ({what})", path.display()),
        )
    };
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
    for entry in index_bytes.chunks_exact(SCOPE_ENTRY) {
        apply_scope_block(&mut file, entry, payload_start, payload_len, rows)
            .map_err(|_| corrupt("malformed scope block"))?;
    }
    Ok(())
}

/// Apply one scope's block from a segment already positioned at its header.
fn apply_scope_block(
    file: &mut File,
    entry: &[u8],
    payload_start: usize,
    payload_len: usize,
    rows: &mut CacheRows,
) -> io::Result<()> {
    let corrupt = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_owned());
    let mut scope = [0u8; 16];
    scope.copy_from_slice(&entry[..16]);
    let flags = le_u32(entry, 16).ok_or_else(|| corrupt("scope entry is short"))?;
    let block_off =
        usize::try_from(le_u32(entry, 20).ok_or_else(|| corrupt("scope entry is short"))?)
            .map_err(|_| corrupt("scope block offset is too large"))?;
    let block_len =
        usize::try_from(le_u32(entry, 24).ok_or_else(|| corrupt("scope entry is short"))?)
            .map_err(|_| corrupt("scope block length is too large"))?;
    let directory_len =
        usize::try_from(le_u32(entry, 28).ok_or_else(|| corrupt("scope entry is short"))?)
            .map_err(|_| corrupt("scope directory length is too large"))?;
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
        rows.remove(&scope);
    }
    let data_len = block_len.saturating_sub(directory_len);
    let directory_start = payload_start
        .checked_add(block_off)
        .ok_or_else(|| corrupt("scope block offset overflows"))?;
    file.seek(SeekFrom::Start(directory_start as u64))?;
    let mut count_bytes = [0u8; BLOCK_COUNT];
    file.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;
    let directory_bytes = count
        .checked_mul(DIR_ENTRY)
        .ok_or_else(|| corrupt("scope directory overflows"))?;
    if BLOCK_COUNT.checked_add(directory_bytes) != Some(directory_len) {
        return Err(corrupt("scope directory length does not match its count"));
    }
    let entries_start = directory_start
        .checked_add(BLOCK_COUNT)
        .ok_or_else(|| corrupt("scope directory offset overflows"))?;
    let data_start = directory_start
        .checked_add(directory_len)
        .ok_or_else(|| corrupt("scope data offset overflows"))?;
    file.seek(SeekFrom::Start(entries_start as u64))?;
    let mut directory = vec![0u8; directory_bytes];
    file.read_exact(&mut directory)?;
    for item in directory.chunks_exact(DIR_ENTRY) {
        let (Some(offset), Some(len), Some(crc)) =
            (le_u32(item, 6), le_u32(item, 10), le_u32(item, 14))
        else {
            return Err(corrupt("scope directory entry is short"));
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

fn read_row_at(
    file: &mut File,
    data_start: usize,
    data_len: usize,
    offset: usize,
    len: usize,
    crc: u32,
    rows: &mut CacheRows,
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

fn segment_signature(segments: &[(u64, PathBuf)]) -> io::Result<Vec<(u64, u64)>> {
    let mut signature = Vec::with_capacity(segments.len());
    for (seq, path) in segments {
        signature.push((*seq, fs::metadata(path)?.len()));
    }
    Ok(signature)
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(name: &str) -> DatabaseLayout {
        let root =
            std::env::temp_dir().join(format!("mtxdb-csr-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        DatabaseLayout::open(root).unwrap()
    }

    fn row(target: u32, kind: u16) -> RowRecord {
        RowRecord {
            family: EdgeFamily::typed(3),
            owner: 1,
            edges: vec![Edge { target, kind }],
        }
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
    fn directory_cache_decodes_each_segment_once_across_scopes() {
        let layout = layout("cache-scopes");
        let store = CsrStore::open(&layout, ShardType::State);
        let a = ScopeId([6; 16]);
        let b = ScopeId([7; 16]);
        let cache = cache_for(&store.dir);
        let segment_reads = || {
            let cache = lock(&cache);
            cache.segment_reads
        };
        store.append_rows(a, &[row(2, 1)]).unwrap();
        store.append_rows(b, &[row(3, 4)]).unwrap();
        assert_eq!(store.read_scope(a).unwrap().rows[0].edges[0].target, 2);
        assert_eq!(store.read_scope(b).unwrap().rows[0].edges[0].target, 3);
        assert_eq!(segment_reads(), 0, "tail rows need no segment decode");

        // One seal, two scopes: the segment is decoded once for the directory.
        store.seal().unwrap();
        assert_eq!(store.read_scope(a).unwrap().rows[0].edges[0].target, 2);
        assert_eq!(segment_reads(), 1);
        assert_eq!(store.read_scope(b).unwrap().rows[0].edges[0].target, 3);
        assert_eq!(segment_reads(), 1, "a second scope reuses the segment");

        // A new tail row, including through a point read, needs no segment.
        store.append_rows(a, &[row(5, 1)]).unwrap();
        let rows = store.read_rows(a, &[(EdgeFamily::typed(3), 1)]).unwrap();
        assert_eq!(rows[0].as_ref().unwrap()[0].target, 5);
        assert_eq!(segment_reads(), 1);

        // A purge is a tail tombstone; it hides A without touching B.
        store.purge_scope(a).unwrap();
        assert!(store.read_scope(a).unwrap().rows.is_empty());
        assert_eq!(store.read_scope(b).unwrap().rows[0].edges[0].target, 3);
        assert_eq!(segment_reads(), 1);
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
