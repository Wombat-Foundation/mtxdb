//! Prototype: a segmented, manifest-published point index.
//!
//! This is a *measurement prototype*, not production code. It answers one
//! question with numbers: if the per-collection locator index were a small
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
//! - Segments are loaded into safe `Vec<u8>` buffers and never decoded into
//!   per-entry objects. A production version would map them read-only; loading
//!   here is the *pessimistic* reopen bound because it reads every segment
//!   body instead of only the manifest.
//! - Reopen reads one small manifest and one segment body each; it never
//!   scans entries into a table.
//!
//! Deliberately out of scope (so the numbers are read for what they are):
//! removals/tombstones, multi-collection sharding, concurrent readers, crash
//! injection, external sort for a bounded-memory build, and a streaming k-way
//! merge for compaction (compaction here materializes the merged set).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Fixed segment header length.
pub(crate) const SEGMENT_HEADER_LEN: usize = 64;
/// Bytes per sorted key in a segment.
pub(crate) const SEGMENT_KEY_BYTES: usize = 16;
/// Bytes per locator in a segment.
pub(crate) const SEGMENT_LOCATOR_BYTES: usize = 8;

const SEGMENT_MAGIC: &[u8; 8] = b"MTXSEG01";
const MANIFEST_MAGIC: &[u8; 8] = b"MTXMAN01";
const MANIFEST_HEADER_LEN: usize = 36;
const MANIFEST_SEGMENT_LEN: usize = 32;
const TAIL_RECORD_LEN: usize = 8 + SEGMENT_KEY_BYTES + SEGMENT_LOCATOR_BYTES;
const BLOOM_HASHES: u64 = 7;
const BLOOM_SHIFT: u32 = 3;
const BLOOM_MASK: u64 = 7;

fn add_usize(left: usize, right: usize) -> io::Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "index size overflow"))
}

fn mul_usize(left: usize, right: usize) -> io::Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "index size overflow"))
}

fn add_u64(left: u64, right: u64) -> io::Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "index counter overflow"))
}

fn usize_of(value: u64) -> io::Result<usize> {
    usize::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "value does not fit usize"))
}

fn u64_of(value: usize) -> io::Result<u64> {
    u64::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "value does not fit u64"))
}

fn malformed(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// Tunable policy for the prototype.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentedOptions {
    /// Tail entries accumulated before an automatic segment flush.
    pub(crate) flush_entries: usize,
    /// Bloom-filter bits per key. Zero disables the filter.
    pub(crate) bloom_bits_per_key: u32,
    /// Whether writes are fsynced. Disabling this measures CPU/layout only.
    pub(crate) fsync: bool,
}

impl Default for SegmentedOptions {
    fn default() -> Self {
        Self {
            flush_entries: 65_536,
            bloom_bits_per_key: 10,
            fsync: true,
        }
    }
}

/// Cumulative write accounting for one prototype instance.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SegmentedStats {
    /// Segment bytes written (including headers and Bloom filters).
    pub(crate) segment_bytes: u64,
    /// Manifest bytes written.
    pub(crate) manifest_bytes: u64,
    /// Tail-log bytes appended.
    pub(crate) tail_log_bytes: u64,
    /// Segment publications.
    pub(crate) segment_writes: u64,
    /// Manifest publications.
    pub(crate) manifest_writes: u64,
    /// Tail flushes, including explicit syncs.
    pub(crate) flushes: u64,
}

/// One entry in the manifest's segment table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentMeta {
    /// Monotonic segment id; also the file suffix.
    pub(crate) id: u64,
    /// Entry count.
    pub(crate) count: u64,
    /// Exact file length.
    pub(crate) file_len: u64,
    /// CRC32 over Bloom + keys + locators.
    pub(crate) crc32: u32,
}

/// The atomically published root of the prototype index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Manifest {
    /// Monotonic publication generation.
    pub(crate) generation: u64,
    /// Highest tail sequence covered by the segment set.
    pub(crate) tail_seq: u64,
    /// Segments in creation order (oldest first).
    pub(crate) segments: Vec<SegmentMeta>,
}

impl Manifest {
    fn encode(&self) -> io::Result<Vec<u8>> {
        let segments_len = mul_usize(self.segments.len(), MANIFEST_SEGMENT_LEN)?;
        let capacity = add_usize(MANIFEST_HEADER_LEN, segments_len)?;
        let count =
            u32::try_from(self.segments.len()).map_err(|_| malformed("too many segments"))?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&self.generation.to_le_bytes());
        out.extend_from_slice(&self.tail_seq.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        for segment in &self.segments {
            out.extend_from_slice(&segment.id.to_le_bytes());
            out.extend_from_slice(&segment.count.to_le_bytes());
            out.extend_from_slice(&segment.file_len.to_le_bytes());
            out.extend_from_slice(&segment.crc32.to_le_bytes());
            out.extend_from_slice(&0_u32.to_le_bytes());
        }
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take(8)? != MANIFEST_MAGIC {
            return Err(malformed("manifest magic mismatch"));
        }
        if cursor.u32()? != 1 {
            return Err(malformed("unsupported manifest version"));
        }
        let generation = cursor.u64()?;
        let tail_seq = cursor.u64()?;
        let count = usize::try_from(cursor.u32()?).map_err(|_| malformed("manifest count"))?;
        let _reserved = cursor.u32()?;
        let table_len = mul_usize(count, MANIFEST_SEGMENT_LEN)?;
        let expected = add_usize(MANIFEST_HEADER_LEN, table_len)?;
        if bytes.len() != expected {
            return Err(malformed("manifest length mismatch"));
        }
        let mut segments = Vec::with_capacity(count);
        for _ in 0..count {
            segments.push(SegmentMeta {
                id: cursor.u64()?,
                count: cursor.u64()?,
                file_len: cursor.u64()?,
                crc32: cursor.u32()?,
                // Reserved word keeps each manifest row 32 bytes.
            });
            let _reserved = cursor.u32()?;
        }
        Ok(Self {
            generation,
            tail_seq,
            segments,
        })
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = add_usize(self.at, len)?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| malformed("truncated record"))?;
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| malformed("truncated u32"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| malformed("truncated u64"))?;
        Ok(u64::from_le_bytes(bytes))
    }
}

/// A loaded immutable segment.
pub(crate) struct Segment {
    meta: SegmentMeta,
    data: Vec<u8>,
    bloom_len: usize,
    keys_off: usize,
    locators_off: usize,
}

impl Segment {
    /// Open and sanity-check the header of an existing segment.
    pub(crate) fn open(dir: &Path, meta: SegmentMeta) -> io::Result<Self> {
        let path = segment_path(dir, meta.id);
        let data = fs::read(path)?;
        if u64_of(data.len())? != meta.file_len {
            return Err(malformed("segment length does not match manifest"));
        }
        if data.len() < SEGMENT_HEADER_LEN || data.get(..8) != Some(SEGMENT_MAGIC.as_slice()) {
            return Err(malformed("segment magic mismatch"));
        }
        let mut cursor = Cursor::new(&data);
        let _magic = cursor.take(8)?;
        if cursor.u32()? != 1 {
            return Err(malformed("unsupported segment version"));
        }
        let _flags = cursor.u32()?;
        if cursor.u64()? != meta.count {
            return Err(malformed("segment count does not match manifest"));
        }
        let bloom_len = usize_of(cursor.u64()?)?;
        let keys_off = usize_of(cursor.u64()?)?;
        let locators_off = usize_of(cursor.u64()?)?;
        let _crc = cursor.u32()?;
        let _reserved = cursor.u32()?;
        let total_len = usize_of(cursor.u64()?)?;
        let count = usize_of(meta.count)?;
        let expected_keys = add_usize(SEGMENT_HEADER_LEN, bloom_len)?;
        let expected_locators = add_usize(expected_keys, mul_usize(count, SEGMENT_KEY_BYTES)?)?;
        let expected_total =
            add_usize(expected_locators, mul_usize(count, SEGMENT_LOCATOR_BYTES)?)?;
        if total_len != data.len()
            || keys_off != expected_keys
            || locators_off != expected_locators
            || expected_total != data.len()
        {
            return Err(malformed("segment layout mismatch"));
        }
        Ok(Self {
            meta,
            data,
            bloom_len,
            keys_off,
            locators_off,
        })
    }

    /// Entry count.
    pub(crate) fn len(&self) -> usize {
        usize_of(self.meta.count).expect("validated segment count")
    }

    /// Exact loaded segment bytes.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.meta.file_len
    }

    /// Segment id.
    pub(crate) fn id(&self) -> u64 {
        self.meta.id
    }

    fn key(&self, index: usize) -> &[u8; SEGMENT_KEY_BYTES] {
        let offset = index
            .checked_mul(SEGMENT_KEY_BYTES)
            .and_then(|offset| self.keys_off.checked_add(offset))
            .expect("validated segment key range");
        let end = offset
            .checked_add(SEGMENT_KEY_BYTES)
            .expect("validated segment key end");
        self.data[offset..end]
            .try_into()
            .expect("validated segment key width")
    }

    fn locator(&self, index: usize) -> u64 {
        let offset = index
            .checked_mul(SEGMENT_LOCATOR_BYTES)
            .and_then(|offset| self.locators_off.checked_add(offset))
            .expect("validated segment locator range");
        let end = offset
            .checked_add(SEGMENT_LOCATOR_BYTES)
            .expect("validated segment locator end");
        u64::from_le_bytes(
            self.data[offset..end]
                .try_into()
                .expect("validated segment locator width"),
        )
    }

    /// Binary-search the segment. Returns the stored locator.
    pub(crate) fn lookup(&self, key: &[u8; SEGMENT_KEY_BYTES]) -> Option<u64> {
        if !self.might_contain(key) {
            return None;
        }
        let mut lo = 0_usize;
        let mut hi = self.len();
        while lo < hi {
            let span = hi.checked_sub(lo).expect("lo <= hi");
            let mid = lo.checked_add(span >> 1).expect("mid within segment");
            match self.key(mid).cmp(key) {
                std::cmp::Ordering::Less => lo = mid.checked_add(1).expect("mid below len"),
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
        let bits = u64_of(self.bloom_len)
            .ok()
            .and_then(|len| len.checked_mul(8))
            .expect("bloom length is bounded by the segment file");
        let (h1, h2) = bloom_hashes(key);
        (0..BLOOM_HASHES).all(|index| {
            let pos = h1
                .wrapping_add(index.wrapping_mul(h2))
                .checked_rem(bits)
                .expect("bloom bit count is nonzero");
            let byte_offset = usize_of(pos >> BLOOM_SHIFT)
                .ok()
                .and_then(|offset| SEGMENT_HEADER_LEN.checked_add(offset))
                .expect("bloom position is inside the segment");
            let byte = self.data[byte_offset];
            byte & (1_u8 << (pos & BLOOM_MASK)) != 0
        })
    }

    /// Visit every entry in key order.
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&[u8; SEGMENT_KEY_BYTES], u64)) {
        for index in 0..self.len() {
            visit(self.key(index), self.locator(index));
        }
    }

    /// Validate the CRC over Bloom + keys + locators.
    pub(crate) fn verify(&self) -> io::Result<()> {
        let payload = self
            .data
            .get(SEGMENT_HEADER_LEN..)
            .ok_or_else(|| malformed("segment header is truncated"))?;
        let actual = crc32fast::hash(payload);
        if actual != self.meta.crc32 {
            return Err(malformed("segment CRC mismatch"));
        }
        Ok(())
    }
}

fn bloom_hashes(key: &[u8; SEGMENT_KEY_BYTES]) -> (u64, u64) {
    let h1 = u64::from_le_bytes(key[..8].try_into().expect("8-byte half"));
    let h2 = u64::from_le_bytes(key[8..].try_into().expect("8-byte half")) | 1;
    (h1, h2)
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
        let bits = u64_of(count)?
            .checked_mul(u64::from(bloom_bits_per_key))
            .and_then(|bits| bits.checked_add(7))
            .ok_or_else(|| malformed("segment Bloom filter is too large"))?;
        usize_of(bits >> BLOOM_SHIFT)?
    };
    let keys_off = add_usize(SEGMENT_HEADER_LEN, bloom_len)?;
    let locators_off = add_usize(keys_off, mul_usize(count, SEGMENT_KEY_BYTES)?)?;
    let file_len = add_usize(locators_off, mul_usize(count, SEGMENT_LOCATOR_BYTES)?)?;

    let mut bloom = vec![0_u8; bloom_len];
    if bloom_len > 0 {
        let bits = u64_of(bloom_len)?
            .checked_mul(8)
            .ok_or_else(|| malformed("bloom size"))?;
        for (key, _) in entries {
            let (h1, h2) = bloom_hashes(key);
            for index in 0..BLOOM_HASHES {
                let pos = h1
                    .wrapping_add(index.wrapping_mul(h2))
                    .checked_rem(bits)
                    .ok_or_else(|| malformed("bloom remainder"))?;
                let byte = usize_of(pos >> BLOOM_SHIFT)?;
                bloom[byte] |= 1_u8 << (pos & BLOOM_MASK);
            }
        }
    }

    let mut keys = Vec::with_capacity(mul_usize(count, SEGMENT_KEY_BYTES)?);
    let mut locators = Vec::with_capacity(mul_usize(count, SEGMENT_LOCATOR_BYTES)?);
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
    header[16..24].copy_from_slice(&u64_of(count)?.to_le_bytes());
    header[24..32].copy_from_slice(&u64_of(bloom_len)?.to_le_bytes());
    header[32..40].copy_from_slice(&u64_of(keys_off)?.to_le_bytes());
    header[40..48].copy_from_slice(&u64_of(locators_off)?.to_le_bytes());
    header[48..52].copy_from_slice(&crc32.to_le_bytes());
    header[56..64].copy_from_slice(&u64_of(file_len)?.to_le_bytes());

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
    // The manifest publication's directory sync makes this rename durable.
    // Syncing the directory here would add a third fsync per flush.
    Ok(SegmentMeta {
        id,
        count: u64_of(count)?,
        file_len: u64_of(file_len)?,
        crc32,
    })
}

/// The segmented index prototype.
pub(crate) struct SegmentedIndex {
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
    pub(crate) fn create(dir: PathBuf, options: SegmentedOptions) -> io::Result<Self> {
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
    pub(crate) fn open(dir: PathBuf, options: SegmentedOptions) -> io::Result<Self> {
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
        let buf = fs::read(&tail_path)?;
        if !buf.is_empty() {
            let (records, _remainder) = buf.as_chunks::<TAIL_RECORD_LEN>();
            for record in records {
                let seq = u64::from_le_bytes(record[..8].try_into().expect("tail sequence"));
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
    pub(crate) fn insert(&mut self, key: [u8; SEGMENT_KEY_BYTES], locator: u64) -> io::Result<()> {
        self.tail_seq = add_u64(self.tail_seq, 1)?;
        let mut record = [0_u8; TAIL_RECORD_LEN];
        record[..8].copy_from_slice(&self.tail_seq.to_le_bytes());
        record[8..8 + SEGMENT_KEY_BYTES].copy_from_slice(&key);
        record[8 + SEGMENT_KEY_BYTES..].copy_from_slice(&locator.to_le_bytes());
        self.tail_log.write_all(&record)?;
        self.stats.tail_log_bytes = add_u64(self.stats.tail_log_bytes, u64_of(TAIL_RECORD_LEN)?)?;
        self.tail.insert(key, locator);
        if self.tail.len() >= self.options.flush_entries {
            self.flush_tail()?;
        }
        Ok(())
    }

    /// Point lookup: tail first, then segments newest to oldest.
    pub(crate) fn lookup(&self, key: &[u8; SEGMENT_KEY_BYTES]) -> Option<u64> {
        if let Some(locator) = self.tail.get(key) {
            return Some(*locator);
        }
        self.segments
            .iter()
            .rev()
            .find_map(|segment| segment.lookup(key))
    }

    /// Flush the tail into a new immutable segment and publish it.
    pub(crate) fn flush_tail(&mut self) -> io::Result<()> {
        if self.tail.is_empty() {
            return Ok(());
        }
        self.top_up_tail_log()?;
        let covered_seq = self.tail_seq;
        let mut entries: Vec<([u8; SEGMENT_KEY_BYTES], u64)> = self.tail.drain().collect();
        entries.sort_unstable_by_key(|entry| entry.0);
        entries.dedup_by_key(|entry| entry.0);

        let id = self.next_segment_id;
        self.next_segment_id = add_u64(self.next_segment_id, 1)?;
        let meta = write_segment(
            &self.dir,
            id,
            &entries,
            self.options.bloom_bits_per_key,
            self.options.fsync,
        )?;
        self.stats.segment_bytes = add_u64(self.stats.segment_bytes, meta.file_len)?;
        self.stats.segment_writes = add_u64(self.stats.segment_writes, 1)?;
        self.manifest.segments.push(meta);
        self.manifest.tail_seq = covered_seq;
        self.publish_manifest()?;
        self.segments.push(Segment::open(&self.dir, meta)?);
        self.reset_tail_log()?;
        self.stats.flushes = add_u64(self.stats.flushes, 1)?;
        Ok(())
    }

    /// Durably flush the tail and rewrite the manifest, even when the tail is
    /// below its automatic flush threshold.
    pub(crate) fn sync(&mut self) -> io::Result<()> {
        self.top_up_tail_log()?;
        self.flush_tail()
    }

    fn top_up_tail_log(&mut self) -> io::Result<()> {
        // The tail buffer is about to be replaced by a segment; a crash before
        // the manifest rename leaves the operation unacknowledged, so there is
        // nothing extra to fsync here.
        self.tail_log.flush()
    }

    fn reset_tail_log(&mut self) -> io::Result<()> {
        self.tail_log.flush()?;
        let file = self.tail_log.get_ref();
        // Truncation is not fsynced: a crash may resurrect covered tail
        // records, and replay skips them by `manifest.tail_seq`.
        file.set_len(0)?;
        let fresh = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.tail_path)?;
        self.tail_log = BufWriter::new(fresh);
        Ok(())
    }

    fn publish_manifest(&mut self) -> io::Result<()> {
        let bytes = self.manifest.encode()?;
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
        self.stats.manifest_bytes = add_u64(self.stats.manifest_bytes, u64_of(bytes.len())?)?;
        self.stats.manifest_writes = add_u64(self.stats.manifest_writes, 1)?;
        Ok(())
    }

    /// Merge every segment into one. Newest locators win on duplicate keys.
    /// The tail is left in place; it is always newer than every segment.
    pub(crate) fn compact(&mut self) -> io::Result<u64> {
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
        merged.sort_unstable_by_key(|entry| (entry.0, entry.2));
        merged.dedup_by_key(|entry| entry.0);
        let entries: Vec<([u8; SEGMENT_KEY_BYTES], u64)> = merged
            .into_iter()
            .map(|(key, locator, _)| (key, locator))
            .collect();
        let id = self.next_segment_id;
        self.next_segment_id = add_u64(self.next_segment_id, 1)?;
        let meta = write_segment(
            &self.dir,
            id,
            &entries,
            self.options.bloom_bits_per_key,
            self.options.fsync,
        )?;
        self.stats.segment_bytes = add_u64(self.stats.segment_bytes, meta.file_len)?;
        self.stats.segment_writes = add_u64(self.stats.segment_writes, 1)?;
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

    /// Verify every segment's CRC and ordering.
    pub(crate) fn verify_all(&self) -> io::Result<()> {
        for segment in &self.segments {
            segment.verify()?;
            let mut previous: Option<[u8; SEGMENT_KEY_BYTES]> = None;
            let mut ordered = true;
            segment.for_each(|key, _| {
                if previous.is_some_and(|prev| prev >= *key) {
                    ordered = false;
                }
                previous = Some(*key);
            });
            if !ordered {
                return Err(malformed("segment keys are not strictly ordered"));
            }
        }
        Ok(())
    }

    /// Tail entries currently resident.
    pub(crate) fn tail_len(&self) -> usize {
        self.tail.len()
    }

    /// Number of live segments.
    pub(crate) fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Loaded segment bytes.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.segments.iter().fold(0_u64, |total, segment| {
            total.saturating_add(segment.resident_bytes())
        })
    }

    /// Durable index bytes on disk: manifest + segments + tail log.
    pub(crate) fn disk_bytes(&self) -> u64 {
        let manifest = fs::metadata(self.dir.join("manifest.bin")).map_or(0, |meta| meta.len());
        let tail = fs::metadata(&self.tail_path).map_or(0, |meta| meta.len());
        self.resident_bytes()
            .saturating_add(manifest)
            .saturating_add(tail)
    }

    /// Cumulative write accounting.
    pub(crate) fn stats(&self) -> SegmentedStats {
        self.stats
    }
}

impl Drop for SegmentedIndex {
    fn drop(&mut self) {
        let _ = self.tail_log.flush();
    }
}
