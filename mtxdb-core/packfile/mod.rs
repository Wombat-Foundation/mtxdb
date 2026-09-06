pub mod storage;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use memmap2::Mmap;
use parking_lot::RwLock;

/// Magic bytes identifying an mdb packfile: "MDB1"
pub const MAGIC: [u8; 4] = *b"MDB1";

/// Maximum record size (64KB). Reject anything larger during recovery scan.
pub const MAX_RECORD_LEN: u32 = 64 * 1024;

/// A single record in the packfile.
///
/// Frame layout on disk:
/// ```text
/// [u32 len]       — byte length of (hash ++ node_bytes), little-endian
/// [16-byte hash]  — structural hash (index-rebuild metadata only, NOT for verification)
/// [node bytes]    — opaque node payload
/// [u32 crc32]     — CRC32 covering len + hash + node_bytes
/// ```
///
/// **Security invariant:** The framed hash is index-rebuild metadata only.
/// Verification always compares against the *caller-requested* hash, never
/// against the hash stored in the frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub hash: [u8; 16],
    pub data: Bytes,
}

impl Record {
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        4_usize
            .wrapping_add(16)
            .wrapping_add(self.data.len())
            .wrapping_add(4)
    }
}

/// A generation of a per-room packfile. Readers hold an `Arc` to this
/// and the repacker swaps in a new one atomically via `ArcSwap`.
#[derive(Debug)]
pub struct PackGeneration {
    pub room_id: [u8; 16],
    pub pack_id: u8,
    pub file: File,
    pub path: PathBuf,
    /// Lazily-created mmap of the packfile. May be remapped when the file
    /// grows (the active generation is appended to between repacks).
    pub(crate) mmap: RwLock<Option<Mmap>>,
    /// Serializes append + offset-capture for this generation's active
    /// write head. Scoped to one generation (not global): writes to a
    /// different room, or a different generation of this room after a
    /// repack swap, never contend on this lock. Without it, two threads
    /// racing `put()` on the same generation could both capture the same
    /// `seek(SeekFrom::End(0))` offset before either's `write_record`
    /// lands — `O_APPEND` places both writes correctly, but the `LossyIndex`
    /// would record the same offset for both, permanently orphaning one.
    pub(crate) append_lock: parking_lot::Mutex<()>,
    /// Whether this generation is still the active one for its room.
    /// Set to `false` when `RepackManager::swap_pack` replaces it.
    /// Drop only deletes the file when this is still `true`, preventing
    /// a wrapped `pack_id` from unlinking a newer generation's file.
    pub(crate) is_current: AtomicBool,
}

impl PackGeneration {
    /// Get the memory-mapped view of this packfile, creating it if absent.
    ///
    /// The mmap is lazily created on first access. Unlike a `OnceLock`, the
    /// map is *not* frozen: `read_at` may remap it to a larger size when the
    /// active generation accumulates appends after the initial mapping.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the packfile cannot be mapped or `mmap(2)`
    /// fails.
    pub fn mmap(&self) -> io::Result<parking_lot::RwLockReadGuard<'_, Option<Mmap>>> {
        let guard = self.mmap.read();
        if guard.is_some() {
            return Ok(guard);
        }
        drop(guard);
        let mut guard = self.mmap.write();
        if guard.is_none() {
            *guard = Some(map_pack(&self.file)?);
        }
        drop(guard);
        Ok(self.mmap.read())
    }
}

/// Scaffold `Mmap::map`, isolating the single unsafe call in this crate.
///
/// # Safety
/// This is the only `unsafe` block in the codebase. The mapping it creates
/// is sound for the following reasons:
///
/// 1. **Append-only file growth.** A packfile only ever grows by appends
///    that serialize through `put()`/`put_many()`. `read_at` remaps when a
///    read lands beyond the current mapping and the file has grown, so the
///    mapping is never stale relative to the data being read. Bytes already
///    written are never mutated, so existing mappings of those bytes stay
///    valid.
/// 2. **No truncation or size change.** A mapped `File` is never
///    `set_len`-shrunk or extended after the mapping is created. Reads
///    beyond a mapping's current end never fault because `read_at` bounds
///    against `map.len()` before touching the map, and remaps (rather than
///    raw faulting) when the file has grown. The only way to shrink is
///    generation unlink at `Drop`, which happens only while readers hold the
///    `Arc` and before any concurrent access could observe a map.
/// 3. **Stable file descriptor.** `PackGeneration` owns the only `File`
///    handle and is kept alive by `Arc`s held by the repack manager and any
///    in-flight reader, so the descriptor remains valid for the mapping's
///    lifetime. `Mmap` also keeps the mapping alive independently of the
///    `File` drop.
#[allow(unsafe_code)]
pub(crate) fn map_pack(file: &File) -> io::Result<Mmap> {
    // SAFETY: See the `map_pack` doc comment — append-only file, remapped on
    // growth, never truncated or shrunk, and a stable fd for the mapping's
    // lifetime. The only mutation mode is append, which does not invalidate
    // the existing mapped bytes.
    unsafe { Mmap::map(file) }
}

impl Drop for PackGeneration {
    fn drop(&mut self) {
        // Only delete the file if this generation is still the active one.
        // When a room is repacked, swap_pack marks the old generation as
        // no longer current — its file must survive until all readers drop
        // their Arc. If the pack_id has wrapped and this generation is no
        // longer current, deleting would unlink a newer generation's file.
        if self.is_current.load(Ordering::Acquire) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Write a single record into the packfile.
///
/// # Errors
/// Returns `io::Error` on write failure.
///
/// # Panics
/// Panics if the record payload exceeds `u32::MAX`.
pub fn write_record(writer: &mut impl Write, record: &Record) -> io::Result<()> {
    let payload_len = u32::try_from(16_usize.wrapping_add(record.data.len()))
        .expect("record payload exceeds u32::MAX");
    if payload_len > MAX_RECORD_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("record payload too large: {payload_len} > {MAX_RECORD_LEN}"),
        ));
    }
    writer.write_all(&payload_len.to_le_bytes())?;
    writer.write_all(&record.hash)?;
    writer.write_all(&record.data)?;

    // CRC covers len + hash + data
    let mut crc = crc32fast::Hasher::new();
    crc.update(&payload_len.to_le_bytes());
    crc.update(&record.hash);
    crc.update(&record.data);
    let checksum = crc.finalize();
    writer.write_all(&checksum.to_le_bytes())?;

    Ok(())
}

/// Read a single record from the packfile. Returns `None` on EOF.
///
/// # Errors
/// Returns `io::Error` on read failure or `io::ErrorKind::InvalidData`
/// if the record length is invalid or the CRC check fails.
///
/// # Panics
/// Panics if `payload_len` (a `u32`) cannot be converted to `usize`.
pub fn read_record(reader: &mut impl Read) -> io::Result<Option<Record>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let payload_len = u32::from_le_bytes(len_buf);
    if !(16..=MAX_RECORD_LEN).contains(&payload_len) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid record length: {payload_len}"),
        ));
    }

    let mut payload = vec![0u8; usize::try_from(payload_len).expect("u32 always fits in usize")];
    reader.read_exact(&mut payload)?;

    let mut crc_buf = [0u8; 4];
    reader.read_exact(&mut crc_buf)?;

    // Verify CRC
    let mut crc = crc32fast::Hasher::new();
    crc.update(&len_buf);
    crc.update(&payload);
    let expected = crc.finalize();
    let actual = u32::from_le_bytes(crc_buf);
    if expected != actual {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("CRC mismatch: expected {expected:08x}, got {actual:08x}"),
        ));
    }

    let mut hash = [0u8; 16];
    hash.copy_from_slice(&payload[..16]);
    let data = Bytes::copy_from_slice(&payload[16..]);

    Ok(Some(Record { hash, data }))
}

/// Write the packfile header (magic + version).
///
/// # Errors
/// Returns `io::Error` on write failure.
pub fn write_header(writer: &mut impl Write) -> io::Result<()> {
    writer.write_all(&MAGIC)?;
    writer.write_all(&[0x01])?; // version 1
    Ok(())
}

/// Read and validate the packfile header.
///
/// # Errors
/// Returns `io::Error` on read failure.
pub fn read_header(reader: &mut impl Read) -> io::Result<bool> {
    let mut buf = [0u8; 5];
    match reader.read_exact(&mut buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e),
    }
    Ok(buf[..4] == MAGIC && buf[4] == 0x01)
}

/// The on-disk path for a room's packfile generation: `<base_dir>/<32-hex
/// room_id>_<2-hex pack_id>.pack`. The single source of this naming
/// convention — `PackfileStorage` and `RepackManager` both need it and must
/// agree on it exactly, since `PackfileStorage::scan_existing` parses this
/// same format back out of existing filenames on startup.
#[must_use]
pub fn pack_path(base_dir: &Path, room_id: &[u8; 16], pack_id: u8) -> PathBuf {
    use std::fmt::Write as _;
    let hex: String = room_id
        .iter()
        .fold(String::with_capacity(32), |mut acc, &b| {
            let _ = write!(acc, "{b:02x}");
            acc
        });
    base_dir.join(format!("{hex}_{pack_id:02x}.pack"))
}

/// Open or create a packfile, writing the header if it's new.
///
/// # Errors
/// Returns `io::Error` on open/write failure or `io::ErrorKind::InvalidData`
/// if the existing header is invalid.
pub fn open_packfile(path: &Path, create: bool) -> io::Result<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(create)
        .open(path)?;

    if file.metadata()?.len() == 0 {
        write_header(&mut file)?;
        file.sync_all()?;
    } else {
        let mut reader = BufReader::new(&file);
        if !read_header(&mut reader)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid packfile header",
            ));
        }
    }

    Ok(file)
}

/// Scan an existing packfile and rebuild a (hash → offset) map.
/// Used for crash recovery of the in-memory index.
///
/// # Errors
/// Returns `io::Error` on read failure (but stops gracefully on
/// unexpected EOF or corrupt tail).
pub fn scan_packfile(path: &Path) -> io::Result<Vec<([u8; 16], u64)>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    if !read_header(&mut reader)? {
        return Ok(entries);
    }

    loop {
        let offset = reader.stream_position()?;
        match read_record(&mut reader) {
            Ok(Some(record)) => {
                entries.push((record.hash, offset));
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                // Torn tail — stop at last good record
                eprintln!("warning: packfile scan stopped at offset {offset}: {e}");
                break;
            }
        }
    }

    Ok(entries)
}

/// Scan a packfile and truncate any torn tail at the last valid record
/// boundary. Used during explicit recovery to repair a packfile before
/// reopening for append.
///
/// # Errors
/// Returns `io::Error` on read or truncate failure.
pub fn scan_and_recover_packfile(path: &Path) -> io::Result<Vec<([u8; 16], u64)>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut entries = Vec::new();

    if !read_header(&mut reader)? {
        return Ok(entries);
    }

    let mut last_valid_offset = reader.stream_position()?;
    loop {
        let offset = reader.stream_position()?;
        match read_record(&mut reader) {
            Ok(Some(record)) => {
                entries.push((record.hash, offset));
                last_valid_offset = reader.stream_position()?;
            }
            Ok(None) => break,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("warning: packfile scan stopped at offset {offset}: {e}");
                break;
            }
        }
    }

    // Truncate torn tail: if we stopped mid-record, rewind to the last
    // valid record boundary so future appends don't land after a corrupt frame.
    drop(reader);
    {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(last_valid_offset)?;
        file.sync_all()?;
    }

    Ok(entries)
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn test_record(hash: [u8; 16], data: &[u8]) -> Record {
        Record {
            hash,
            data: Bytes::copy_from_slice(data),
        }
    }

    fn test_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_pf_{name}_{id}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_write_read_roundtrip() {
        let record = test_record([0xaa; 16], b"hello world");
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        let mut cursor = Cursor::new(&buf);
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(record, read);
    }

    #[test]
    fn test_multiple_records() {
        let records = vec![
            test_record([0x01; 16], b"first"),
            test_record([0x02; 16], b"second record"),
            test_record([0x03; 16], &vec![0xff; 1024]),
        ];

        let mut buf = Vec::new();
        for r in &records {
            write_record(&mut buf, r).unwrap();
        }

        let mut cursor = Cursor::new(&buf);
        for expected in &records {
            let read = read_record(&mut cursor).unwrap().unwrap();
            assert_eq!(*expected, read);
        }
        assert!(read_record(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn test_crc_corruption_detected() {
        let record = test_record([0xaa; 16], b"test data");
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        // Flip a byte in the data payload
        buf[20] ^= 0xff;

        let mut cursor = Cursor::new(&buf);
        let result = read_record(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_length_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);

        let mut cursor = Cursor::new(&buf);
        let result = read_record(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_roundtrip() {
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        assert!(read_header(&mut cursor).unwrap());
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut buf = vec![0u8; 5];
        buf[0..4].copy_from_slice(b"BADC");

        let mut cursor = Cursor::new(&buf);
        assert!(!read_header(&mut cursor).unwrap());
    }

    #[test]
    fn test_header_empty_returns_false() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        assert!(!read_header(&mut cursor).unwrap());
    }

    #[test]
    fn test_record_serialized_len() {
        let r = Record {
            hash: [0u8; 16],
            data: Bytes::from_static(b"hello"),
        };
        assert_eq!(r.serialized_len(), 4 + 16 + 5 + 4);
    }

    #[test]
    fn test_write_record_payload_too_large() {
        let r = Record {
            hash: [0u8; 16],
            data: Bytes::from(vec![0u8; MAX_RECORD_LEN as usize]),
        };
        let mut buf = Vec::new();
        let err = write_record(&mut buf, &r).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_read_record_non_eof_io_error() {
        struct FailRead;
        impl Read for FailRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke"))
            }
        }
        let result = read_record(&mut FailRead);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn test_read_header_non_eof_io_error() {
        struct FailRead;
        impl Read for FailRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "nope"))
            }
        }
        let result = read_header(&mut FailRead);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn test_open_packfile_invalid_header() {
        let dir = test_dir("packfile_invalid_header");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        std::fs::write(&path, b"BADC\x01extra").unwrap();
        let result = open_packfile(&path, false);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_scan_packfile_empty_file() {
        let dir = test_dir("scan_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        std::fs::write(&path, b"").unwrap();
        let entries = scan_packfile(&path).unwrap();
        assert_eq!(entries, vec![]);
    }

    #[test]
    fn test_scan_packfile_torn_tail() {
        let dir = test_dir("scan_torn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        // Write header + one valid record + partial garbage
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        write_record(&mut buf, &test_record([0xaa; 16], b"data")).unwrap();
        buf.extend_from_slice(&[0xff; 10]); // torn trailing bytes
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_packfile(&path).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_scan_and_recover_truncates_torn_tail() {
        let dir = test_dir("recover_torn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        write_record(&mut buf, &test_record([0xaa; 16], b"good")).unwrap();
        let valid_len = buf.len();
        buf.extend_from_slice(&[0xff; 20]); // torn tail
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len as u64);
    }

    #[test]
    fn test_scan_and_recover_clean_file() {
        let dir = test_dir("recover_clean");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        let mut buf = Vec::new();
        write_header(&mut buf).unwrap();
        write_record(&mut buf, &test_record([0xbb; 16], b"ok")).unwrap();
        write_record(&mut buf, &test_record([0xcc; 16], b"ok2")).unwrap();
        let expected_len = buf.len();
        std::fs::write(&path, &buf).unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), expected_len as u64);
    }

    #[test]
    fn test_scan_and_recover_empty_header() {
        let dir = test_dir("recover_noheader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("00000000000000000000000000000000_00.pack");
        std::fs::write(&path, b"").unwrap();
        let entries = scan_and_recover_packfile(&path).unwrap();
        assert_eq!(entries, vec![]);
    }
}
