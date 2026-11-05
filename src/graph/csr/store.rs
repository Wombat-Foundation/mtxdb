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
//! decodes only bytes appended to the tail since the previous read; sealed
//! segments are decoded again only when the segment set changes. The cache is
//! in-process state, not a durability boundary: reopen starts from the files.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
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

/// One store directory's decoded state.
#[derive(Debug, Default)]
struct StoreCache {
    /// Segment numbers already folded into `rows`, oldest first.
    segment_ids: Vec<u64>,
    /// Newest visible row per scope, keyed by `(family, owner)`.
    rows: HashMap<[u8; 16], ScopeRows>,
    /// Bytes of `tail.log` already decoded into `rows`.
    tail_len: usize,
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
        let valid = self.cached_tail_len()?;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.tail_path())?;
        let len = file.metadata()?.len();
        if (valid as u64) < len {
            // Drop a torn final frame so new records follow a valid boundary.
            file.set_len(valid as u64)?;
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
        self.refresh_cached()?;
        Ok(())
    }

    /// Hide every row of `scope` with a tombstone and sync the tail.
    ///
    /// # Errors
    /// I/O errors.
    pub fn purge_scope(&self, scope: ScopeId) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.tail_path())?;
        let valid = self.cached_tail_len()?;
        let len = file.metadata()?.len();
        if (valid as u64) < len {
            file.set_len(valid as u64)?;
        }
        file.seek(SeekFrom::End(0))?;
        let payload = encode_row(scope, EdgeFamily::plain(0), 0, &[], true)?;
        let mut buffer = Vec::new();
        write_frame(&mut buffer, &payload);
        file.write_all(&buffer)?;
        file.sync_data()?;
        self.refresh_cached()?;
        Ok(())
    }

    /// Apply the tail as a new immutable segment, then truncate the tail.
    ///
    /// # Errors
    /// I/O errors or a corrupt frame that is not merely torn at the end.
    pub fn seal(&self) -> io::Result<()> {
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
        let mut payload = Vec::new();
        for record in &records {
            payload.extend_from_slice(
                &u32::try_from(record.payload.len())
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "csr row exceeds u32::MAX")
                    })?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(&record.payload);
        }
        let mut file_bytes =
            Vec::with_capacity(13usize.saturating_add(payload.len()).saturating_add(4));
        file_bytes.extend_from_slice(&SEGMENT_MAGIC);
        file_bytes.push(SEGMENT_VERSION);
        file_bytes.extend_from_slice(
            &u32::try_from(records.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many csr rows"))?
                .to_le_bytes(),
        );
        file_bytes.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "csr segment too large"))?
                .to_le_bytes(),
        );
        file_bytes.extend_from_slice(&payload);
        file_bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());

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
            self.seal()?;
        }
        Ok(())
    }

    /// Read every visible row of `scope`.
    ///
    /// # Errors
    /// I/O errors or a malformed sealed segment.
    pub fn read_scope(&self, scope: ScopeId) -> io::Result<ScopeData> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache)?;
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

    /// The valid prefix of the tail already decoded into the cache.
    fn cached_tail_len(&self) -> io::Result<usize> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache)?;
        Ok(cache.tail_len)
    }

    /// Bring the cache up to date with the files after a write.
    fn refresh_cached(&self) -> io::Result<()> {
        let cache = cache_for(&self.dir);
        let mut cache = lock(&cache);
        self.refresh_cache(&mut cache)
    }

    /// Fold new segments and new tail bytes into the cache.
    fn refresh_cache(&self, cache: &mut StoreCache) -> io::Result<()> {
        let segments = self.segments()?;
        let ids: Vec<u64> = segments.iter().map(|(seq, _)| *seq).collect();
        if ids != cache.segment_ids {
            if ids.starts_with(&cache.segment_ids) {
                for (seq, path) in &segments[cache.segment_ids.len()..] {
                    apply_segment(path, *seq, &mut cache.rows)?;
                    #[cfg(test)]
                    {
                        cache.segment_reads = cache.segment_reads.saturating_add(1);
                    }
                }
            } else {
                Self::rebuild_cache(cache, &segments)?;
            }
            cache.segment_ids = ids;
            // A new segment normally replaced the tail it was built from.
            // Re-decoding the tail is idempotent and restores newer rows.
            cache.tail_len = 0;
        }
        let len = self.tail_file_len()?;
        if len < cache.tail_len {
            // The tail was truncated, normally by a concurrent seal. Re-list
            // so the segment that replaced it is included, then replay from 0.
            let segments = self.segments()?;
            Self::rebuild_cache(cache, &segments)?;
        }
        let len = self.tail_file_len()?;
        if len > cache.tail_len {
            match fs::read(self.tail_path()) {
                Ok(data) => {
                    let start = cache.tail_len.min(data.len());
                    let (frames, valid) = decode_frames(&data[start..]);
                    for frame in &frames {
                        apply_payload(&frame.payload, &mut cache.rows)?;
                    }
                    cache.tail_len = start.saturating_add(valid);
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    let segments = self.segments()?;
                    Self::rebuild_cache(cache, &segments)?;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Drop the cached rows and replay every segment currently on disk.
    fn rebuild_cache(cache: &mut StoreCache, segments: &[(u64, PathBuf)]) -> io::Result<()> {
        let mut rows = HashMap::new();
        for (seq, path) in segments {
            apply_segment(path, *seq, &mut rows)?;
        }
        #[cfg(test)]
        {
            cache.segment_reads = cache
                .segment_reads
                .saturating_add(u64::try_from(segments.len()).unwrap_or(u64::MAX));
        }
        cache.rows = rows;
        cache.segment_ids = segments.iter().map(|(seq, _)| *seq).collect();
        cache.tail_len = 0;
        Ok(())
    }

    fn tail_file_len(&self) -> io::Result<usize> {
        match fs::metadata(self.tail_path()) {
            Ok(meta) => Ok(usize::try_from(meta.len()).unwrap_or(usize::MAX)),
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
        let path = self.tail_path();
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
            Err(error) => return Err(error),
        };
        Ok(decode_frames(&data))
    }
}

struct Frame {
    payload: Vec<u8>,
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

fn decode_frames(data: &[u8]) -> (Vec<Frame>, usize) {
    let mut frames = Vec::new();
    let mut at = 0usize;
    while let Some(rest) = data.get(at..) {
        let Some(header) = rest.get(..8) else {
            break;
        };
        let len = u32::from_le_bytes(header[..4].try_into().expect("frame length has four bytes"))
            as usize;
        let crc = u32::from_le_bytes(
            header[4..8]
                .try_into()
                .expect("frame checksum has four bytes"),
        );
        let Some(payload) = rest.get(8..).and_then(|tail| tail.get(..len)) else {
            break;
        };
        if crc32fast::hash(payload) != crc {
            break;
        }
        frames.push(Frame {
            payload: payload.to_vec(),
        });
        let Some(next) = at.checked_add(8).and_then(|n| n.checked_add(len)) else {
            break;
        };
        at = next;
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
        let edge = payload
            .get(at..)
            .and_then(|rest| rest.get(..EDGE_SIZE))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "csr row ended early"))?;
        let target = u32::from_le_bytes(edge[..4].try_into().expect("edge target has four bytes"));
        let kind = u16::from_le_bytes(edge[4..6].try_into().expect("edge kind has two bytes"));
        edges.push(Edge { target, kind });
        at = at
            .checked_add(EDGE_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "csr row overflows"))?;
    }
    Ok((
        ScopeId(scope),
        EdgeFamily { id: family, typed },
        owner,
        purge,
        edges,
    ))
}

fn apply_payload(payload: &[u8], rows: &mut HashMap<[u8; 16], ScopeRows>) -> io::Result<()> {
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

fn apply_segment(path: &Path, seq: u64, rows: &mut HashMap<[u8; 16], ScopeRows>) -> io::Result<()> {
    let data = fs::read(path)?;
    let corrupt = |what: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("csr segment {} ({what})", path.display()),
        )
    };
    if data.len() < 13 || data[..4] != SEGMENT_MAGIC {
        return Err(corrupt("bad magic"));
    }
    if data[4] != SEGMENT_VERSION {
        return Err(corrupt("unsupported version"));
    }
    let count = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
    let payload_len = u32::from_le_bytes([data[9], data[10], data[11], data[12]]) as usize;
    let payload_end = 13usize
        .checked_add(payload_len)
        .ok_or_else(|| corrupt("length overflows"))?;
    if payload_end.saturating_add(4) != data.len() {
        return Err(corrupt("length does not match file size"));
    }
    let payload = &data[13..payload_end];
    let checksum = data
        .get(payload_end..)
        .and_then(|rest| rest.get(..4))
        .ok_or_else(|| corrupt("missing checksum"))?;
    let expected_crc = u32::from_le_bytes(
        checksum
            .try_into()
            .map_err(|_| corrupt("missing checksum"))?,
    );
    if crc32fast::hash(payload) != expected_crc {
        return Err(corrupt("checksum mismatch"));
    }
    let mut at = 0usize;
    for _ in 0..count {
        let head = payload
            .get(at..)
            .and_then(|rest| rest.get(..4))
            .ok_or_else(|| corrupt("truncated row table"))?;
        let len = u32::from_le_bytes(head.try_into().expect("row length has four bytes")) as usize;
        at = at
            .checked_add(4)
            .ok_or_else(|| corrupt("row table overflows"))?;
        let end = at
            .checked_add(len)
            .ok_or_else(|| corrupt("row length overflows"))?;
        if end > payload.len() {
            return Err(corrupt("truncated row"));
        }
        apply_payload(&payload[at..end], rows).map_err(|_| corrupt("malformed row"))?;
        at = end;
    }
    if at != payload.len() {
        return Err(corrupt("trailing bytes after rows"));
    }
    let _ = seq;
    Ok(())
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
    fn cached_reads_follow_new_tail_rows_and_seals() {
        let layout = layout("cache");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([11; 16]);
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        assert_eq!(store.read_scope(scope).unwrap().rows[0].edges[0].target, 2);
        // A later append is visible from the newly decoded tail bytes only.
        store.append_rows(scope, &[row(6, 7)]).unwrap();
        let data = store.read_scope(scope).unwrap();
        assert_eq!(data.rows.len(), 1);
        assert_eq!(data.rows[0].edges[0].target, 6);
        // Sealing moves the same rows into a segment; the view must not change.
        store.seal().unwrap();
        let data = store.read_scope(scope).unwrap();
        assert_eq!(data.rows.len(), 1);
        assert_eq!(data.rows[0].edges[0].target, 6);
        let _ = fs::remove_dir_all(layout.root());
    }

    #[test]
    fn repeated_reads_do_not_redecode_sealed_segments() {
        let layout = layout("cache_segments");
        let store = CsrStore::open(&layout, ShardType::State);
        let scope = ScopeId([12; 16]);
        let segment_reads = || {
            let cache = cache_for(&store.dir);
            let cache = lock(&cache);
            cache.segment_reads
        };
        store.append_rows(scope, &[row(2, 1)]).unwrap();
        assert_eq!(segment_reads(), 0, "tail rows need no segment decode");
        store.read_scope(scope).unwrap();
        assert_eq!(segment_reads(), 0);
        store.seal().unwrap();
        store.read_scope(scope).unwrap();
        assert_eq!(segment_reads(), 1, "the new segment is decoded once");
        store.read_scope(scope).unwrap();
        assert_eq!(segment_reads(), 1, "a repeat read reuses the cache");
        store.append_rows(scope, &[row(3, 1)]).unwrap();
        store.read_scope(scope).unwrap();
        assert_eq!(segment_reads(), 1, "a new tail row needs no segment decode");
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
