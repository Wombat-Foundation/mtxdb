//! On-disk collection-version baseline (`index.versions`).
//!
//! Collection logical versions are a journal concept, not a packfile one: a
//! checkpoint records them, and the retained WAL can seed versions for groups
//! it still holds. A reclaimed group has no frame-level version tag, so once
//! the checkpoint that covered it is deleted — as a writable paged open does —
//! the only durable copy of the version is gone.
//!
//! This file is that durable copy. It holds only `(collection_id, version)`
//! pairs: no index slots, no pack table, no fingerprint. A paged reopen reads
//! it in one sequential pass and merges it into the journal's version tables
//! with the same `max` rule as the checkpoint baseline. It is written before
//! the WAL coverage claim that permits reclaim, so a reclaimed group's version
//! always has a home.
//!
//! Layout (little-endian, fixed width):
//!
//! ```text
//!   [magic "MTXVERS1" 8B]
//!   [entry count u32 4B]
//!   [entry 0: collection_id 16B, version u64 8B]
//!   ...
//!   [crc32 u32 over header and entries]
//! ```
//!
//! A missing, truncated, or corrupt file is treated as absent. Callers merge
//! it with the checkpoint and the retained WAL using `max`, so a damaged file
//! can lower nothing below what those sources still hold, and a stale file
//! cannot raise a version past the frames the packs contain.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Magic identifying an `index.versions` file.
const MAGIC: &[u8; 8] = b"MTXVERS1";
/// Magic plus the `u32` entry count.
const HEADER_LEN: usize = 12;
/// One `(collection_id, version)` entry.
const ENTRY_LEN: usize = 24;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// File name of the collection-version baseline inside a store directory.
pub const INDEX_VERSIONS_FILE: &str = "index.versions";

/// Atomically write `versions`, sorted by collection id.
///
/// The write is temp-file + `sync_data` + rename, followed by a directory
/// sync so the rename itself is durable. On any failure the temporary file is
/// removed and the previous file is left untouched.
///
/// # Errors
/// Returns an I/O error if the temporary file cannot be created, written,
/// synced, renamed, or its directory cannot be synced.
pub fn write_versions(path: &Path, versions: &[([u8; 16], u64)]) -> std::io::Result<()> {
    let mut sorted = versions.to_vec();
    sorted.sort_unstable_by_key(|(collection_id, _)| *collection_id);

    let count = u32::try_from(sorted.len())
        .map_err(|_| std::io::Error::other("too many collection versions"))?;
    let capacity = HEADER_LEN
        .checked_add(
            sorted
                .len()
                .checked_mul(ENTRY_LEN)
                .ok_or_else(|| std::io::Error::other("collection versions overflow"))?,
        )
        .and_then(|len| len.checked_add(4))
        .ok_or_else(|| std::io::Error::other("collection versions overflow"))?;
    let mut buf = Vec::with_capacity(capacity);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&count.to_le_bytes());
    for (collection_id, version) in &sorted {
        buf.extend_from_slice(collection_id);
        buf.extend_from_slice(&version.to_le_bytes());
    }
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let unique = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = path.with_extension(format!("tmp.{}.{unique}", std::process::id()));
    let write_result = (|| -> std::io::Result<()> {
        let mut tmp = fs::File::create(&tmp_path)?;
        tmp.write_all(&buf)?;
        tmp.sync_data()
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(error);
    }
    fs::rename(&tmp_path, path)?;
    if let Some(dir) = path.parent() {
        crate::shard::sync_directory(dir)?;
    }
    Ok(())
}

/// Read and validate `versions`; `None` for a missing or malformed file.
#[must_use]
pub fn read_versions(path: &Path) -> Option<Vec<([u8; 16], u64)>> {
    let bytes = fs::read(path).ok()?;
    decode_versions(&bytes)
}

fn decode_versions(bytes: &[u8]) -> Option<Vec<([u8; 16], u64)>> {
    let header = bytes.get(..HEADER_LEN)?;
    if header.get(..8)? != MAGIC {
        return None;
    }
    let count = u32::from_le_bytes(header.get(8..12)?.try_into().ok()?) as usize;
    let entries_len = count.checked_mul(ENTRY_LEN)?;
    let entries_start = HEADER_LEN;
    let crc_start = entries_start.checked_add(entries_len)?;
    let crc_end = crc_start.checked_add(4)?;
    if bytes.len() != crc_end {
        return None;
    }
    let expected = u32::from_le_bytes(bytes.get(crc_start..crc_end)?.try_into().ok()?);
    if crc32fast::hash(&bytes[..crc_start]) != expected {
        return None;
    }
    let mut versions = Vec::with_capacity(count);
    for index in 0..count {
        let start = entries_start.checked_add(index.checked_mul(ENTRY_LEN)?)?;
        let end = start.checked_add(ENTRY_LEN)?;
        let entry = bytes.get(start..end)?;
        let collection_id: [u8; 16] = entry.get(..16)?.try_into().ok()?;
        let version = u64::from_le_bytes(entry.get(16..24)?.try_into().ok()?);
        versions.push((collection_id, version));
    }
    Some(versions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_sorted_versions() {
        let path = std::env::temp_dir().join(format!(
            "mtxdb-versions-{}-{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let versions = vec![([9u8; 16], 7), ([1u8; 16], 42)];
        write_versions(&path, &versions).unwrap();
        let mut read = read_versions(&path).unwrap();
        read.sort_unstable();
        let mut expected = versions.clone();
        expected.sort_unstable();
        assert_eq!(read, expected);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn corrupt_and_truncated_files_are_absent() {
        assert!(decode_versions(b"").is_none());
        let path = std::env::temp_dir().join(format!(
            "mtxdb-versions-corrupt-{}-{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        write_versions(&path, &[([3u8; 16], 5)]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(decode_versions(&bytes).is_none());
        assert!(decode_versions(&bytes[..bytes.len() - 1]).is_none());
        let _ = fs::remove_file(path);
    }
}
