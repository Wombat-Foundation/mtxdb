//! Prototype: a segmented, manifest-published point index.
//!
//! This is a *measurement prototype*, not production code. It exists to answer
//! one question with numbers: if the per-collection locator index were a small
//! mutable tail plus immutable sorted segments under an atomically published
//! manifest, instead of a resident open-addressed table plus a whole-image
//! checkpoint and a delta log, what would insert cost, lookup cost, reopen
//! cost, durable bytes and resident memory look like?
//!
//! Layout on disk:
//!
//! ```text
//!   manifest.bin          generation, tail sequence, segment table
//!   seg-<id>.bin          [64B header][bloom][sorted 16B keys][8B locators]
//!   tail.log              [seq u64][key 16B][locator u64] ...
//! ```
//!
//! Publication order is segment (fsync, rename) -> manifest (fsync, rename) ->
//! truncate the tail log. A crash before the manifest rename leaves an
//! unreferenced segment; a crash after it but before truncation replays only
//! tail records above `manifest.tail_seq`, so already-covered records are
//! skipped.
//!
//! Residency:
//!
//! - The tail is a `HashMap` bounded by the flush threshold.
//! - Segments are memory-mapped and never decoded wholesale.
//! - Reopen reads one small manifest and maps every segment; it never scans an
//!   entry.
//!
//! Deliberately out of scope (so the numbers are read for what they are):
//! removals/tombstones, multi-collection sharding, concurrent readers, crash
//! injection, external sort for a bounded-memory build, and a streaming k-way
//! merge for compaction (compaction here materializes the merged set).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};

/// Fixed segment header length.
pub const SEGMENT_HEADER_LEN: usize = 64;
/// Bytes per sorted key in a segment.
pub const SEGMENT_KEY_BYTES: usize = 16;
/// Bytes per locator in a segment.
pub const SEGMENT_LOCATOR_BYTES: usize = 8;
/// Bytes per entry in a segment, excluding the Bloom filter.
pub const SEGMENT_ENTRY_BYTES: usize = SEGMENT_KEY_BYTES + SEGMENT_LOCATOR_BYTES;

const SEGMENT_MAGIC: &[u8; 8] = b"MTXSEG01";
const MANIFEST_MAGIC: &[u8; 8] = b"MTXMAN01";
/// Fixed manifest header length.
const MANIFEST_HEADER_LEN: usize = 36;
const TAIL_RECORD_LEN: usize = 8 + SEGMENT_KEY_BYTES + SEGMENT_LOCATOR_BYTES;

/// Tunable policy for the prototype.
#[derive(Debug, Clone, Copy)]
pub struct SegmentedOptions {
    /// Tail entries accumulated before an automatic segment flush.
    pub flush_entries: usize,
    /// Bloom-filter bits per key. Zero disables the filter.
    pub bloom_bits_per_key: u32,
    /// Segment count above which [`SegmentedIndex::compact`] merges them all.
    pub compact_threshold: usize,
    /// Whether writes are fsynced. Disabling this measures CPU/layout only.
    pub fsync: bool,
}

impl Default for SegmentedOptions {
    fn default() -> Self {
        Self {
            flush_entries: 65_536,
            bloom_bits_per_key: 10,
            compact_threshold: 32,
            fsync: true,
        }
    }
}

/// Cumulative write accounting for one prototype instance.
#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentedStats {
    /// Segment bytes written (including headers and Bloom filters).
    pub segment_bytes: u64,
    /// Manifest bytes written.
    pub manifest_bytes: u64,
    /// Tail-log bytes appended.
    pub tail_log_bytes: u64,
    /// Segment publications.
    pub segment_writes: u64,
    /// Manifest publications.
    pub manifest_writes: u64,
    /// Tail flushes, including explicit syncs.
    pub flushes: u64,
}

/// One entry in the manifest's segment table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentMeta {
    /// Monotonic segment id; also the file suffix.
    pub id: u64,
    /// Entry count.
    pub count: u64,
    /// Exact file length.
    pub file_len: u64,
    /// CRC32 over Bloom + keys + locators.
    pub crc32: u32,
}

/// The atomically published root of the prototype index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    /// Monotonic publication generation.
    pub generation: u64,
    /// Highest tail sequence covered by the segment set.
    pub tail_seq: u64,
    /// Segments in creation order (oldest first).
    pub segments: Vec<SegmentMeta>,
}

impl Manifest {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MANIFEST_HEADER_LEN + self.segments.len() * 32);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&self.generation.to_le_bytes());
        out.extend_from_slice(&self.tail_seq.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.segments.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        out.extend_from_slice(&0_u32.to_le_bytes());
        for segment in &self.segments {
            out.extend_from_slice(&segment.id.to_le_bytes());
            out.extend_from_slice(&segment.count.to_le_bytes());
            out.extend_from_slice(&segment.file_len.to_le_bytes());
            out.extend_from_slice(&segment.crc32.to_le_bytes());
            out.extend_from_slice(&0_u32.to_le_bytes());
        }
        out
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "manifest is malformed");
        if bytes.len() < MANIFEST_HEADER_LEN || &bytes[..8] != MANIFEST_MAGIC {
            return Err(invalid());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| invalid())?);
        if version != 1 {
            return Err(invalid());
        }
        let generation = u64::from_le_bytes(bytes[12..20].try_into().map_err(|_| invalid())?);
        let tail_seq = u64::from_le_bytes(bytes[20..28].try_into().map_err(|_| invalid())?);
        let count = u32::from_le_bytes(bytes[28..32].try_into().map_err(|_| invalid())?) as usize;
        let expected = MANIFEST_HEADER_LEN
            .checked_add(count.saturating_mul(32))
            .ok_or_else(invalid)?;
        if bytes.len() != expected {
            return Err(invalid());
        }
        let mut segments = Vec::with_capacity(count);
        for i in 0..count {
            let at = MANIFEST_HEADER_LEN + i * 32;
            segments.push(SegmentMeta {
                id: u64::from_le_bytes(bytes[at..at + 8].try_into().map_err(|_| invalid())?),
                count: u64::from_le_bytes(
                    bytes[at + 8..at + 16].try_into().map_err(|_| invalid())?,
                ),
                file_len: u64::from_le_bytes(
                    bytes[at + 16..at + 24].try_into().map_err(|_| invalid())?,
                ),
                crc32: u32::from_le_bytes(
                    bytes[at + 24..at + 28].try_into().map_err(|_| invalid())?,
                ),
            });
        }
        Ok(Self {
            generation,
            tail_seq,
            segments,
        })
    }
}

/// A mapped immutable segment.
pub struct Segment {
    meta: SegmentMeta,
    mmap: Mmap,
    bloom_len: usize,
    keys_off: usize,
    locators_off: usize,
}

impl Segment {
    /// Open and sanity-check the header of an existing segment.
    pub fn open(dir: &Path, meta: SegmentMeta) -> io::Result<Self> {
        let path = segment_path(dir, meta.id);
        let file = File::open(path)?;
        if file.metadata()?.len() != meta.file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment length does not match manifest",
            ));
        }
        // SAFETY: the file is immutable for the lifetime of this mapping and
        // the prototype never truncates a published segment.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "segment is malformed");
        if mmap.len() < SEGMENT_HEADER_LEN || &mmap[..8] != SEGMENT_MAGIC {
            return Err(invalid());
        }
        let version = read_u32(&mmap, 8)?;
        if version != 1 {
            return Err(invalid());
        }
        if read_u64(&mmap, 16)? != meta.count {
            return Err(invalid());
        }
        let bloom_len = usize::try_from(read_u64(&mmap, 24)?).map_err(|_| invalid())?;
        let keys_off = usize::try_from(read_u64(&mmap, 32)?).map_err(|_| invalid())?;
        let locators_off = usize::try_from(read_u64(&mmap, 40)?).map_err(|_| invalid())?;
        let total_len = usize::try_from(read_u64(&mmap, 56)?).map_err(|_| invalid())?;
        let count = usize::try_from(meta.count).map_err(|_| invalid())?;
        if total_len != mmap.len()
            || keys_off != SEGMENT_HEADER_LEN.saturating_add(bloom_len)
            || locators_off != keys_off.saturating_add(count.saturating_mul(SEGMENT_KEY_BYTES))
            || locators_off.saturating_add(count.saturating_mul(SEGMENT_LOCATOR_BYTES))
                != mmap.len()
        {
            return Err(invalid());
        }
        Ok(Self {
            meta,
            mmap,
            bloom_len,
            keys_off,
            locators_off,
        })
    }

    /// Entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meta.count as usize
    }

    /// Whether the segment holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exact mapped file bytes.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        self.meta.file_len
    }

    /// Segment id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.meta.id
    }

    fn key(&self, index: usize) -> &[u8; SEGMENT_KEY_BYTES] {
        let start = self
            .keys_off
            .saturating_add(index.saturating_mul(SEGMENT_KEY_BYTES));
        self.mmap[start..start + SEGMENT_KEY_BYTES]
            .try_into()
            .expect("validated segment key range")
    }

    fn locator(&self, index: usize) -> u64 {
        let start = self
            .locators_off
            .saturating_add(index.saturating_mul(SEGMENT_LOCATOR_BYTES));
        u64::from_le_bytes(
            self.mmap[start..start + SEGMENT_LOCATOR_BYTES]
                .try_into()
                .expect("validated segment locator range"),
        )
    }

    /// Binary-search the segment. Returns the stored locator.
    #[must_use]
    pub fn lookup(&self, key: &[u8; SEGMENT_KEY_BYTES]) -> Option<u64> {
        if !self.might_contain(key) {
            return None;
        }
        let mut lo = 0_usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key(mid).cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(self.locator(mid)),
            }
        }
        None
    }

    fn might_contain(&self, key: &[u8; SEGMENT_KEY_BYTES]) -> bool {
        if self.bloom_len == 0 {
            return true;
        }
        let bits = self.bloom_len.saturating_mul(8);
        let (h1, h2) = bloom_hashes(key);
        let passes = (0..BLOOM_HASHES).all(|i| {
            let pos = h1.wrapping_add((i as u64).wrapping_mul(h2)) % bits as u64;
            let byte = self.mmap[SEGMENT_HEADER_LEN + (pos / 8) as usize];
            byte & (1 << (pos % 8)) != 0
        });
        passes
    }

    /// Visit every entry in key order.
    pub fn for_each(&self, mut visit: impl FnMut(&[u8; SEGMENT_KEY_BYTES], u64)) {
        for i in 0..self.len() {
            visit(self.key(i), self.locator(i));
        }
    }

    /// Validate the CRC over Bloom + keys + locators.
    pub fn verify(&self) -> io::Result<()> {
        let payload = &self.mmap[SEGMENT_HEADER_LEN..];
        let actual = crc32fast::hash(payload);
        if actual != self.meta.crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment CRC mismatch",
            ));
        }
        Ok(())
    }
}

const BLOOM_HASHES: usize = 7;

fn bloom_hashes(key: &[u8; SEGMENT_KEY_BYTES]) -> (u64, u64) {
    let h1 = u64::from_le_bytes(key[..8].try_into().expect("8-byte half"));
    let h2 = u64::from_le_bytes(key[8..].try_into().expect("8-byte half")) | 1;
    (h1, h2)
}

fn read_u32(bytes: &[u8], at: usize) -> io::Result<u32> {
    let slice = bytes
        .get(at..at + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated header"))?;
    Ok(u32::from_le_bytes(slice.try_into().expect("4-byte slice")))
}

fn read_u64(bytes: &[u8], at: usize) -> io::Result<u64> {
    let slice = bytes
        .get(at..at + 8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated header"))?;
    Ok(u64::from_le_bytes(slice.try_into().expect("8-byte slice")))
}

fn segment_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("seg-{id:016x}.bin"))
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

fn write_segment(
    dir: &Path,
    id: u64,
    entries: &[([u8; SEGMENT_KEY_BYTES], u64)],
    bloom_bits_per_key: u32,
    fsync: bool,
) -> io::Result<SegmentMeta> {
    let count = entries.len();
    let bloom_len = if bloom_bits_per_key == 0 {
        0
    } else {
        let bits = (count as u64)
            .saturating_mul(u64::from(bloom_bits_per_key))
            .saturating_add(7);
        usize::try_from(bits / 8).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "segment Bloom filter is too large",
            )
        })?
    };
    let keys_off = SEGMENT_HEADER_LEN + bloom_len;
    let locators_off = keys_off + count * SEGMENT_KEY_BYTES;
    let file_len = locators_off + count * SEGMENT_LOCATOR_BYTES;

    let mut bloom = vec![0_u8; bloom_len];
    if bloom_len > 0 {
        let bits = (bloom_len * 8) as u64;
        for (key, _) in entries {
            let (h1, h2) = bloom_hashes(key);
            for i in 0..BLOOM_HASHES {
                let pos = h1.wrapping_add((i as u64).wrapping_mul(h2)) % bits;
                bloom[(pos / 8) as usize] |= 1 << (pos % 8);
            }
        }
    }

    let mut keys = Vec::with_capacity(count * SEGMENT_KEY_BYTES);
    let mut locators = Vec::with_capacity(count * SEGMENT_LOCATOR_BYTES);
    for (key, locator) in entries {
        keys.extend_from_slice(key);
        locators.extend_from_slice(&locator.to_le_bytes());
    }
    let crc32 = {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bloom);
        hasher.update(&keys);
        hasher.update(&locators);
        hasher.finalize()
    };

    let mut header = [0_u8; SEGMENT_HEADER_LEN];
    header[..8].copy_from_slice(SEGMENT_MAGIC);
    header[8..12].copy_from_slice(&1_u32.to_le_bytes());
    header[16..24].copy_from_slice(&(count as u64).to_le_bytes());
    header[24..32].copy_from_slice(&(bloom_len as u64).to_le_bytes());
    header[32..40].copy_from_slice(&(keys_off as u64).to_le_bytes());
    header[40..48].copy_from_slice(&(locators_off as u64).to_le_bytes());
    header[48..52].copy_from_slice(&crc32.to_le_bytes());
    header[56..64].copy_from_slice(&(file_len as u64).to_le_bytes());

    let final_path = segment_path(dir, id);
    let tmp_path = dir.join(format!(".seg-{id:016x}.tmp"));
    {
        let mut file = BufWriter::new(File::create(&tmp_path)?);
        file.write_all(&header)?;
        file.write_all(&bloom)?;
        file.write_all(&keys)?;
        file.write_all(&locators)?;
        file.flush()?;
        if fsync {
            file.get_ref().sync_all()?;
        }
    }
    fs::rename(&tmp_path, &final_path)?;
    if fsync {
        sync_dir(dir)?;
    }
    Ok(SegmentMeta {
        id,
        count: count as u64,
        file_len: file_len as u64,
        crc32,
    })
}

/// The segmented index prototype.
pub struct SegmentedIndex {
    dir: PathBuf,
    options: SegmentedOptions,
    manifest: Manifest,
    segments: Vec<Segment>,
    tail: HashMap<[u8; SEGMENT_KEY_BYTES], u64>,
    tail_path: PathBuf,
    tail_log: BufWriter<File>,
    tail_seq: u64,
    next_segment_id: u64,
    stats: SegmentedStats,
}

impl SegmentedIndex {
    /// Create a new index in `dir`, which must be empty or absent.
    pub fn create(dir: PathBuf, options: SegmentedOptions) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        let manifest_path = dir.join("manifest.bin");
        if manifest_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "probe directory already holds an index",
            ));
        }
        let tail_path = dir.join("tail.log");
        let tail_log = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&tail_path)?,
        );
        let mut index = Self {
            dir,
            options,
            manifest: Manifest::default(),
            segments: Vec::new(),
            tail: HashMap::new(),
            tail_path,
            tail_log,
            tail_seq: 0,
            next_segment_id: 1,
            stats: SegmentedStats::default(),
        };
        index.publish_manifest()?;
        Ok(index)
    }

    /// Open an existing index from its manifest. Does not verify segment CRCs.
    pub fn open(dir: PathBuf, options: SegmentedOptions) -> io::Result<Self> {
        let manifest_path = dir.join("manifest.bin");
        let bytes = fs::read(&manifest_path)?;
        let manifest = Manifest::decode(&bytes)?;
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for meta in &manifest.segments {
            segments.push(Segment::open(&dir, *meta)?);
        }

        let tail_path = dir.join("tail.log");
        let mut tail_file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&tail_path)?;
        let mut tail = HashMap::new();
        let mut buf = Vec::new();
        tail_file.read_to_end(&mut buf)?;
        if !buf.is_empty() {
            let usable = buf.len() / TAIL_RECORD_LEN * TAIL_RECORD_LEN;
            for record in buf[..usable].chunks_exact(TAIL_RECORD_LEN) {
                let seq = u64::from_le_bytes(record[..8].try_into().expect("tail seq"));
                if seq <= manifest.tail_seq {
                    continue;
                }
                let key: [u8; SEGMENT_KEY_BYTES] = record[8..8 + SEGMENT_KEY_BYTES]
                    .try_into()
                    .expect("tail key");
                let locator = u64::from_le_bytes(
                    record[8 + SEGMENT_KEY_BYTES..]
                        .try_into()
                        .expect("tail locator"),
                );
                tail.insert(key, locator);
            }
            // A torn record is ignored, matching the write-once tail contract.
        }
        tail_file.seek(SeekFrom::End(0))?;

        let next_segment_id = manifest
            .segments
            .iter()
            .map(|meta| meta.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let tail_seq = manifest.tail_seq;
        Ok(Self {
            dir,
            options,
            manifest,
            segments,
            tail,
            tail_path,
            tail_log: BufWriter::new(tail_file),
            tail_seq,
            next_segment_id,
            stats: SegmentedStats::default(),
        })
    }

    /// Insert or overwrite one key. Flushes a segment when the tail is full.
    pub fn insert(&mut self, key: [u8; SEGMENT_KEY_BYTES], locator: u64) -> io::Result<()> {
        self.tail_seq = self.tail_seq.saturating_add(1);
        let mut record = [0_u8; TAIL_RECORD_LEN];
        record[..8].copy_from_slice(&self.tail_seq.to_le_bytes());
        record[8..8 + SEGMENT_KEY_BYTES].copy_from_slice(&key);
        record[8 + SEGMENT_KEY_BYTES..].copy_from_slice(&locator.to_le_bytes());
        self.tail_log.write_all(&record)?;
        self.stats.tail_log_bytes = self
            .stats
            .tail_log_bytes
            .saturating_add(TAIL_RECORD_LEN as u64);
        self.tail.insert(key, locator);
        if self.tail.len() >= self.options.flush_entries {
            self.flush_tail()?;
        }
        Ok(())
    }

    /// Point lookup: tail first, then segments newest to oldest.
    #[must_use]
    pub fn lookup(&self, key: &[u8; SEGMENT_KEY_BYTES]) -> Option<u64> {
        if let Some(locator) = self.tail.get(key) {
            return Some(*locator);
        }
        for segment in self.segments.iter().rev() {
            if let Some(locator) = segment.lookup(key) {
                return Some(locator);
            }
        }
        None
    }

    /// Flush the tail into a new immutable segment and publish it.
    pub fn flush_tail(&mut self) -> io::Result<()> {
        if self.tail.is_empty() {
            return Ok(());
        }
        self.tail_log.flush()?;
        if self.options.fsync {
            self.tail_log.get_ref().sync_all()?;
        }
        let covered_seq = self.tail_seq;
        let mut entries: Vec<([u8; SEGMENT_KEY_BYTES], u64)> = self
            .tail
            .drain()
            .map(|(key, locator)| (key, locator))
            .collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        entries.dedup_by_key(|entry| entry.0);

        let id = self.next_segment_id;
        self.next_segment_id = self.next_segment_id.saturating_add(1);
        let meta = write_segment(
            &self.dir,
            id,
            &entries,
            self.options.bloom_bits_per_key,
            self.options.fsync,
        )?;
        self.stats.segment_bytes = self.stats.segment_bytes.saturating_add(meta.file_len);
        self.stats.segment_writes = self.stats.segment_writes.saturating_add(1);
        self.manifest.segments.push(meta);
        self.manifest.tail_seq = covered_seq;
        self.publish_manifest()?;
        self.segments.push(Segment::open(&self.dir, meta)?);
        self.reset_tail_log()?;
        self.stats.flushes = self.stats.flushes.saturating_add(1);
        Ok(())
    }

    /// Durably flush the tail and rewrite the manifest, even when the tail is
    /// below its automatic flush threshold.
    pub fn sync(&mut self) -> io::Result<()> {
        self.top_up_tail_log()?;
        self.flush_tail()
    }

    fn top_up_tail_log(&mut self) -> io::Result<()> {
        self.tail_log.flush()?;
        if self.options.fsync {
            self.tail_log.get_ref().sync_all()?;
        }
        Ok(())
    }

    fn reset_tail_log(&mut self) -> io::Result<()> {
        self.tail_log.flush()?;
        let file = self.tail_log.get_ref();
        file.set_len(0)?;
        if self.options.fsync {
            file.sync_all()?;
        }
        let fresh = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.tail_path)?;
        self.tail_log = BufWriter::new(fresh);
        Ok(())
    }

    fn publish_manifest(&mut self) -> io::Result<()> {
        let bytes = self.manifest.encode();
        let tmp = self.dir.join(".manifest.tmp");
        {
            let mut file = BufWriter::new(File::create(&tmp)?);
            file.write_all(&bytes)?;
            file.flush()?;
            if self.options.fsync {
                file.get_ref().sync_all()?;
            }
        }
        fs::rename(&tmp, self.dir.join("manifest.bin"))?;
        if self.options.fsync {
            sync_dir(&self.dir)?;
        }
        self.stats.manifest_bytes = self.stats.manifest_bytes.saturating_add(bytes.len() as u64);
        self.stats.manifest_writes = self.stats.manifest_writes.saturating_add(1);
        Ok(())
    }

    /// Merge every segment into one. Newest locators win on duplicate keys.
    /// The tail is left in place; it is always newer than every segment.
    pub fn compact(&mut self) -> io::Result<u64> {
        if self.segments.len() <= 1 {
            return Ok(0);
        }
        self.top_up_tail_log()?;
        let mut merged: Vec<([u8; SEGMENT_KEY_BYTES], u64, usize)> = Vec::new();
        for (rank, segment) in self.segments.iter().rev().enumerate() {
            segment.for_each(|key, locator| {
                merged.push((*key, locator, rank));
            });
        }
        merged.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
        merged.dedup_by_key(|entry| entry.0);
        let entries: Vec<([u8; SEGMENT_KEY_BYTES], u64)> = merged
            .into_iter()
            .map(|(key, locator, _)| (key, locator))
            .collect();
        let id = self.next_segment_id;
        self.next_segment_id = self.next_segment_id.saturating_add(1);
        let meta = write_segment(
            &self.dir,
            id,
            &entries,
            self.options.bloom_bits_per_key,
            self.options.fsync,
        )?;
        self.stats.segment_bytes = self.stats.segment_bytes.saturating_add(meta.file_len);
        self.stats.segment_writes = self.stats.segment_writes.saturating_add(1);
        let old = std::mem::take(&mut self.segments);
        self.manifest.segments.clear();
        self.manifest.segments.push(meta);
        self.publish_manifest()?;
        self.segments.push(Segment::open(&self.dir, meta)?);
        for segment in old {
            let _ = fs::remove_file(segment_path(&self.dir, segment.id()));
        }
        Ok(meta.file_len)
    }

    /// Compact when the segment count is over the configured threshold.
    pub fn maybe_compact(&mut self) -> io::Result<Option<u64>> {
        if self.segments.len() > self.options.compact_threshold {
            Ok(Some(self.compact()?))
        } else {
            Ok(None)
        }
    }

    /// Verify every segment's CRC and ordering.
    pub fn verify_all(&self) -> io::Result<()> {
        for segment in &self.segments {
            segment.verify()?;
            let mut previous: Option<[u8; SEGMENT_KEY_BYTES]> = None;
            let mut ordered = true;
            segment.for_each(|key, _| {
                if let Some(prev) = previous {
                    if prev >= *key {
                        ordered = false;
                    }
                }
                previous = Some(*key);
            });
            if !ordered {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "segment keys are not strictly ordered",
                ));
            }
        }
        Ok(())
    }

    /// Tail entries currently resident.
    #[must_use]
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// Number of live segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Mapped segment bytes.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        self.segments.iter().map(Segment::mapped_bytes).sum()
    }

    /// Durable index bytes on disk: manifest + segments + tail log.
    #[must_use]
    pub fn disk_bytes(&self) -> u64 {
        let manifest = fs::metadata(self.dir.join("manifest.bin"))
            .map(|meta| meta.len())
            .unwrap_or(0);
        let tail = fs::metadata(&self.tail_path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        self.mapped_bytes()
            .saturating_add(manifest)
            .saturating_add(tail)
    }

    /// Cumulative write accounting.
    #[must_use]
    pub fn stats(&self) -> SegmentedStats {
        self.stats
    }

    /// The current manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

impl Drop for SegmentedIndex {
    fn drop(&mut self) {
        let _ = self.tail_log.flush();
    }
}
