//! Directory-backed CSR edge store: immutable segments plus a durable tail log.
//!
//! One store per pool. Rows are keyed by the engine's generic 16-byte scope
//! (collection id), family and owner, so any number of scopes share the same few
//! files; nothing is created per scope and nothing here knows what a scope is.
//!
//! Lives outside every pack and pool. Forward edge rows are immutable per
//! `(owner, family)`, so reads take the first hit from the tail, then segments
//! newest to oldest. `seal` writes the tail as a segment of [`EdgeChunk`]s
//! (temp file, fsync, rename, directory fsync) and then truncates the log. A
//! torn final log record is dropped on open.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::chunk::{chunk_of, ChunkError, EdgeChunk, Row};

const SEGMENT_MAGIC: [u8; 4] = *b"CSRS";
const SEGMENT_VERSION: u8 = 1;
const TAIL_FILE: &str = "tail.log";

/// Errors from the CSR store.
#[derive(Debug)]
pub enum CsrStoreError {
    /// Filesystem failure.
    Io(io::Error),
    /// A segment was malformed.
    Corrupt(String),
    /// A row for this owner and family already exists with different edges.
    Collision,
}

impl From<io::Error> for CsrStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ChunkError> for CsrStoreError {
    fn from(error: ChunkError) -> Self {
        Self::Corrupt(error.to_string())
    }
}

impl std::fmt::Display for CsrStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "csr store io: {error}"),
            Self::Corrupt(what) => write!(f, "csr store corrupt: {what}"),
            Self::Collision => f.write_str("csr store row differs from the recorded one"),
        }
    }
}

impl std::error::Error for CsrStoreError {}

/// Engine scope: the collection id the rows belong to.
pub type Scope = [u8; 16];

type Key = (Scope, u16, u32);

/// A directory of CSR segments plus a tail log.
#[derive(Debug)]
pub struct CsrStore {
    dir: PathBuf,
    segments: Vec<(u64, HashMap<Key, EdgeChunk>)>,
    tail: BTreeMap<(Scope, u16, u32), (bool, Row)>,
    log: File,
    next_segment: u64,
}

fn segment_path(dir: &Path, n: u64) -> PathBuf {
    dir.join(format!("seg-{n:010}.csr"))
}

fn crc(bytes: &[u8]) -> u32 {
    let mut hash = 0xcbf2_9ce4_u32;
    for &byte in bytes {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    hash
}

fn fsync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

impl CsrStore {
    /// Open (creating if needed) the store in `dir`.
    ///
    /// # Errors
    /// Filesystem failures or a corrupt segment.
    pub fn open(dir: &Path) -> Result<Self, CsrStoreError> {
        fs::create_dir_all(dir)?;
        let mut numbers = Vec::new();
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            let is_temp = Path::new(&name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"));
            if is_temp {
                let _ = fs::remove_file(dir.join(&name));
            } else if let Some(n) = name
                .strip_prefix("seg-")
                .and_then(|rest| rest.strip_suffix(".csr"))
                .and_then(|n| n.parse::<u64>().ok())
            {
                numbers.push(n);
            }
        }
        numbers.sort_unstable();
        let mut segments = Vec::new();
        for &n in &numbers {
            segments.push((n, read_segment(&segment_path(dir, n))?));
        }
        let next_segment = match numbers.last() {
            Some(&n) => n
                .checked_add(1)
                .ok_or_else(|| CsrStoreError::Corrupt("segment number exhausted".to_owned()))?,
            None => 1,
        };
        let mut store = Self {
            dir: dir.to_path_buf(),
            segments,
            tail: BTreeMap::new(),
            log: OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(dir.join(TAIL_FILE))?,
            next_segment,
        };
        store.replay()?;
        Ok(store)
    }

    fn replay(&mut self) -> Result<(), CsrStoreError> {
        let mut bytes = Vec::new();
        File::open(self.dir.join(TAIL_FILE))?.read_to_end(&mut bytes)?;
        let mut at = 0usize;
        let mut good = 0usize;
        while let Some(rest) = bytes.get(at..) {
            let Some(header) = rest.get(..8) else {
                break;
            };
            let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
            let sum = u32::from_be_bytes(header[4..8].try_into().unwrap());
            let Some(body) = rest.get(8..).and_then(|tail| tail.get(..len)) else {
                break;
            };
            if crc(body) != sum {
                break;
            }
            let Some((scope, family, typed, owner, row)) = decode_record(body) else {
                break;
            };
            self.tail.insert((scope, family, owner), (typed, row));
            let Some(next) = at.checked_add(8).and_then(|n| n.checked_add(len)) else {
                break;
            };
            at = next;
            good = at;
        }
        if good != bytes.len() {
            self.log.set_len(good as u64)?;
            self.log.sync_all()?;
        }
        Ok(())
    }

    /// The row of `owner` in `family`, if recorded.
    #[must_use]
    pub fn get(&self, scope: Scope, family: u16, owner: u32) -> Option<Row> {
        if let Some((_, row)) = self.tail.get(&(scope, family, owner)) {
            return Some(row.clone());
        }
        let key = (scope, family, chunk_of(owner));
        self.segments
            .iter()
            .rev()
            .find_map(|(_, chunks)| chunks.get(&key).and_then(|chunk| chunk.get(owner)))
    }

    /// Record `owner`'s sorted row. Re-recording identical edges is a no-op.
    ///
    /// # Errors
    /// [`CsrStoreError::Collision`] if a different row exists; I/O errors.
    pub fn append(
        &mut self,
        scope: Scope,
        family: u16,
        typed: bool,
        owner: u32,
        row: &[(u32, u16)],
    ) -> Result<(), CsrStoreError> {
        if let Some(existing) = self.get(scope, family, owner) {
            return if existing == row {
                Ok(())
            } else {
                Err(CsrStoreError::Collision)
            };
        }
        let body = encode_record(scope, family, typed, owner, row);
        let mut framed = Vec::with_capacity(body.len().saturating_add(8));
        framed.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
        framed.extend_from_slice(&crc(&body).to_be_bytes());
        framed.extend_from_slice(&body);
        self.log.write_all(&framed)?;
        self.tail
            .insert((scope, family, owner), (typed, row.to_vec()));
        Ok(())
    }

    /// Make appended rows durable.
    ///
    /// # Errors
    /// I/O errors.
    pub fn sync(&self) -> Result<(), CsrStoreError> {
        self.log.sync_data()?;
        Ok(())
    }

    /// Rows currently in the unsealed tail.
    #[must_use]
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// Number of sealed segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Seal the tail into a new immutable segment and truncate the log.
    ///
    /// # Errors
    /// I/O errors. On failure the tail and log are left intact.
    pub fn seal(&mut self) -> Result<(), CsrStoreError> {
        if self.tail.is_empty() {
            return Ok(());
        }
        let mut chunks: HashMap<Key, EdgeChunk> = HashMap::new();
        for (&(scope, family, owner), (typed, row)) in &self.tail {
            chunks
                .entry((scope, family, chunk_of(owner)))
                .or_insert_with(|| EdgeChunk::new(*typed))
                .insert(owner, row);
        }
        let n = self.next_segment;
        let path = segment_path(&self.dir, n);
        let tmp = path.with_extension("csr.tmp");
        write_segment(&tmp, &chunks)?;
        fs::rename(&tmp, &path)?;
        fsync_dir(&self.dir)?;
        self.log.set_len(0)?;
        self.log.sync_all()?;
        self.segments.push((n, chunks));
        self.next_segment = n
            .checked_add(1)
            .ok_or_else(|| CsrStoreError::Corrupt("segment number exhausted".to_owned()))?;
        self.tail.clear();
        Ok(())
    }
}

fn encode_record(
    scope: Scope,
    family: u16,
    typed: bool,
    owner: u32,
    row: &[(u32, u16)],
) -> Vec<u8> {
    let mut chunk = EdgeChunk::new(typed);
    chunk.insert(owner, row);
    let mut out = scope.to_vec();
    out.extend_from_slice(&family.to_be_bytes());
    out.extend_from_slice(&chunk.encode());
    out
}

fn decode_record(body: &[u8]) -> Option<(Scope, u16, bool, u32, Row)> {
    let scope: Scope = body.get(..16)?.try_into().ok()?;
    let family = u16::from_be_bytes(body.get(16..18)?.try_into().ok()?);
    let typed = *body.get(18)? == 1;
    let chunk = EdgeChunk::decode(&body[18..], typed).ok()?;
    let (owner, row) = chunk.iter().next()?;
    Some((scope, family, typed, owner, row))
}

fn write_segment(path: &Path, chunks: &HashMap<Key, EdgeChunk>) -> Result<(), CsrStoreError> {
    let mut keys: Vec<&Key> = chunks.keys().collect();
    keys.sort_unstable();
    let mut out = SEGMENT_MAGIC.to_vec();
    out.push(SEGMENT_VERSION);
    out.extend_from_slice(&u32::try_from(keys.len()).unwrap_or(u32::MAX).to_be_bytes());
    for key in keys {
        let body = chunks[key].encode();
        out.extend_from_slice(&key.0);
        out.extend_from_slice(&key.1.to_be_bytes());
        out.extend_from_slice(&key.2.to_be_bytes());
        out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
        out.extend_from_slice(&body);
    }
    let sum = crc(&out);
    out.extend_from_slice(&sum.to_be_bytes());
    let mut file = File::create(path)?;
    file.write_all(&out)?;
    file.sync_all()?;
    Ok(())
}

fn read_segment(path: &Path) -> Result<HashMap<Key, EdgeChunk>, CsrStoreError> {
    let bad = |what: &str| CsrStoreError::Corrupt(format!("{}: {what}", path.display()));
    let bytes = fs::read(path)?;
    if bytes.len() < 13 || bytes[..4] != SEGMENT_MAGIC || bytes[4] != SEGMENT_VERSION {
        return Err(bad("header"));
    }
    let split = bytes.len().checked_sub(4).ok_or_else(|| bad("header"))?;
    let (payload, sum) = bytes.split_at(split);
    if crc(payload) != u32::from_be_bytes(sum.try_into().unwrap()) {
        return Err(bad("checksum"));
    }
    let count = u32::from_be_bytes(payload[5..9].try_into().unwrap());
    let mut at = 9usize;
    let mut chunks = HashMap::new();
    for _ in 0..count {
        let head_end = at.checked_add(26).ok_or_else(|| bad("truncated"))?;
        let head = payload.get(at..head_end).ok_or_else(|| bad("truncated"))?;
        let scope: Scope = head[..16].try_into().unwrap();
        let family = u16::from_be_bytes(head[16..18].try_into().unwrap());
        let chunk_index = u32::from_be_bytes(head[18..22].try_into().unwrap());
        let len = u32::from_be_bytes(head[22..26].try_into().unwrap()) as usize;
        let body_end = head_end.checked_add(len).ok_or_else(|| bad("truncated"))?;
        let body = payload
            .get(head_end..body_end)
            .ok_or_else(|| bad("truncated"))?;
        let typed = body.first() == Some(&1);
        chunks.insert(
            (scope, family, chunk_index),
            EdgeChunk::decode(body, typed)?,
        );
        at = body_end;
    }
    if at != payload.len() {
        return Err(bad("trailing bytes"));
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Scope = [7; 16];
    const T: Scope = [8; 16];

    fn dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("mtxdb-csr-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn tail_survives_reopen_and_seal_moves_it_to_a_segment() {
        let d = dir("seal");
        let mut store = CsrStore::open(&d).unwrap();
        store.append(S, 1, false, 5, &[(2, 0), (3, 0)]).unwrap();
        store.append(S, 3, true, 2000, &[(5, 9)]).unwrap();
        store.sync().unwrap();
        drop(store);
        let mut store = CsrStore::open(&d).unwrap();
        assert_eq!(store.tail_len(), 2);
        assert_eq!(store.get(S, 1, 5), Some(vec![(2, 0), (3, 0)]));
        store.seal().unwrap();
        assert_eq!((store.tail_len(), store.segment_count()), (0, 1));
        drop(store);
        let store = CsrStore::open(&d).unwrap();
        assert_eq!(store.tail_len(), 0);
        assert_eq!(store.get(S, 1, 5), Some(vec![(2, 0), (3, 0)]));
        assert_eq!(store.get(S, 3, 2000), Some(vec![(5, 9)]));
        assert_eq!(store.get(S, 1, 6), None);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn rows_are_immutable_and_a_torn_tail_record_is_dropped() {
        let d = dir("torn");
        let mut store = CsrStore::open(&d).unwrap();
        store.append(S, 1, false, 7, &[(1, 0)]).unwrap();
        store.append(S, 1, false, 7, &[(1, 0)]).unwrap();
        assert!(matches!(
            store.append(S, 1, false, 7, &[(2, 0)]),
            Err(CsrStoreError::Collision)
        ));
        store.append(S, 1, false, 8, &[(1, 0)]).unwrap();
        store.sync().unwrap();
        drop(store);
        let log = d.join(TAIL_FILE);
        let len = fs::metadata(&log).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&log)
            .unwrap()
            .set_len(len - 3)
            .unwrap();
        let store = CsrStore::open(&d).unwrap();
        assert_eq!(store.get(S, 1, 7), Some(vec![(1, 0)]));
        assert_eq!(store.get(S, 1, 8), None);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn scopes_share_one_store_without_mixing_rows() {
        let d = dir("scopes");
        let mut store = CsrStore::open(&d).unwrap();
        store.append(S, 1, false, 5, &[(2, 0)]).unwrap();
        store.append(T, 1, false, 5, &[(3, 0)]).unwrap();
        store.seal().unwrap();
        let store = CsrStore::open(&d).unwrap();
        assert_eq!(store.get(S, 1, 5), Some(vec![(2, 0)]));
        assert_eq!(store.get(T, 1, 5), Some(vec![(3, 0)]));
        assert_eq!(
            fs::read_dir(&d).unwrap().count(),
            2,
            "tail.log + one segment"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn newer_segments_do_not_hide_older_rows() {
        let d = dir("segs");
        let mut store = CsrStore::open(&d).unwrap();
        store.append(S, 1, false, 1, &[(9, 0)]).unwrap();
        store.seal().unwrap();
        store.append(S, 1, false, 2, &[(8, 0)]).unwrap();
        store.seal().unwrap();
        let store = CsrStore::open(&d).unwrap();
        assert_eq!(store.segment_count(), 2);
        assert_eq!(store.get(S, 1, 1), Some(vec![(9, 0)]));
        assert_eq!(store.get(S, 1, 2), Some(vec![(8, 0)]));
        let _ = fs::remove_dir_all(&d);
    }
}
