use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
        let _ = fs::remove_file(&self.path);
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
    if payload_len == 0 || payload_len > MAX_RECORD_LEN {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn test_record(hash: [u8; 16], data: &[u8]) -> Record {
        Record {
            hash,
            data: Bytes::copy_from_slice(data),
        }
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
}
