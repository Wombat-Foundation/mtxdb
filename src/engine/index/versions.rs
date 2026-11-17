//! On-disk collection-version baseline (`index.versions`).
//!
//! Collection logical versions are a journal concept, not a packfile one: a
//! checkpoint records them, and the retained WAL can seed versions for groups
//! it still holds. A reclaimed group has no frame-level version tag, so once
//! the checkpoint that covered it is deleted — as a writable paged open does —
//! the only durable copy of the version is gone.
//!
//! This file is that durable copy. It holds only `(collection_id, version)`
//! pairs and the coverage LSN those versions are durable through: no index
//! slots, no pack table. A paged reopen reads it in one sequential pass and
//! merges it into the journal's version tables with the same `max` rule as the
//! checkpoint baseline. It is written before the WAL coverage claim that
//! permits reclaim, so a reclaimed group's version always has a home, and the
//! coverage it reports is never newer than the versions it carries.
//!
//! Layout (little-endian, fixed width):
//!
//! ```text
//!   [magic "MTXVERS2" 8B]
//!   [covered_lsn u64 8B]
//!   [entry count u32 4B]
//!   [entry 0: collection_id 16B, version u64 8B]
//!   ...
//!   [crc32 u32 over header and entries]
//! ```
//!
//! The first format (`MTXVERS1`) had no coverage field and is still accepted
//! with coverage 0, so a store written between the two commits does not lose
//! its versions.
//!
//! A missing, truncated, or corrupt file is treated as absent. Because the
//! file is a whole-file atomic snapshot (temp + fsync + rename + directory
//! sync), there is no partial record to replay: a torn write is never
//! published, and a damaged published file is rejected whole. Callers merge it
//! with the checkpoint and the retained WAL using `max`, so a stale file
//! cannot lower a version and a damaged one cannot raise it.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Magic identifying the current `index.versions` format.
const MAGIC_V2: &[u8; 8] = b"MTXVERS2";
/// Magic of the format without a coverage field (coverage read as 0).
const MAGIC_V1: &[u8; 8] = b"MTXVERS1";
/// Magic, `covered_lsn`, and the `u32` entry count.
const HEADER_V2_LEN: usize = 20;
/// Magic and the `u32` entry count.
const HEADER_V1_LEN: usize = 12;
/// One `(collection_id, version)` entry.
const ENTRY_LEN: usize = 24;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// File name of the collection-version baseline inside a store directory.
pub const INDEX_VERSIONS_FILE: &str = "index.versions";

/// A validated `index.versions` snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionsSnapshot {
    /// The LSN these versions are durable through.
    pub covered_lsn: u64,
    /// `(collection_id, version)`, in file (sorted) order.
    pub versions: Vec<([u8; 16], u64)>,
}

/// Atomically write `versions`, sorted by collection id, with `covered_lsn`.
///
/// The write is temp-file + `sync_data` + rename, followed by a directory
/// sync so the rename itself is durable. Failures before rename leave the
/// previous file untouched; a directory-sync failure is reported after the
/// new file has been published.
///
/// # Errors
/// Returns an I/O error if the temporary file cannot be created, written,
/// synced, renamed, or its directory cannot be synced.
pub fn write_versions(
    path: &Path,
    versions: &[([u8; 16], u64)],
    covered_lsn: u64,
) -> std::io::Result<()> {
    write_versions_deferring_directory_sync(path, versions, covered_lsn)?;
    if let Some(dir) = path.parent() {
        crate::shard::sync_directory(dir)?;
    }
    Ok(())
}

/// [`write_versions`] without the final directory sync: the file is synced and
/// renamed into place, but the rename is not yet durable.
///
/// For a caller that publishes several files in one directory and syncs the
/// directory once after the last rename (see `CheckpointTail::run`). It must
/// not acknowledge anything that depends on this file, such as a WAL coverage
/// claim, until that directory sync has succeeded.
///
/// # Errors
/// Returns an I/O error if the temporary file cannot be created, written,
/// synced or renamed.
pub fn write_versions_deferring_directory_sync(
    path: &Path,
    versions: &[([u8; 16], u64)],
    covered_lsn: u64,
) -> std::io::Result<()> {
    let mut sorted = versions.to_vec();
    sorted.sort_unstable_by_key(|(collection_id, _)| *collection_id);

    let count = u32::try_from(sorted.len())
        .map_err(|_| std::io::Error::other("too many collection versions"))?;
    let capacity = HEADER_V2_LEN
        .checked_add(
            sorted
                .len()
                .checked_mul(ENTRY_LEN)
                .ok_or_else(|| std::io::Error::other("collection versions overflow"))?,
        )
        .and_then(|len| len.checked_add(4))
        .ok_or_else(|| std::io::Error::other("collection versions overflow"))?;
    let mut buf = Vec::with_capacity(capacity);
    buf.extend_from_slice(MAGIC_V2);
    buf.extend_from_slice(&covered_lsn.to_le_bytes());
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
    if let Err(error) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(error);
    }
    Ok(())
}

/// Read and validate `versions`; `None` for a missing or malformed file.
#[must_use]
pub fn read_versions(path: &Path) -> Option<VersionsSnapshot> {
    let bytes = fs::read(path).ok()?;
    decode_versions(&bytes)
}

/// Read only the coverage field, without decoding the entries.
///
/// `None` for a missing or malformed file.
#[must_use]
pub fn read_coverage(path: &Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    decode_versions(&bytes).map(|snapshot| snapshot.covered_lsn)
}

fn decode_versions(bytes: &[u8]) -> Option<VersionsSnapshot> {
    let magic = bytes.get(..8)?;
    let (covered_lsn, header_len) = if magic == MAGIC_V2 {
        (
            u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?),
            HEADER_V2_LEN,
        )
    } else if magic == MAGIC_V1 {
        (0, HEADER_V1_LEN)
    } else {
        return None;
    };
    let count_offset = header_len.checked_sub(4)?;
    let count = u32::from_le_bytes(bytes.get(count_offset..header_len)?.try_into().ok()?) as usize;
    let entries_len = count.checked_mul(ENTRY_LEN)?;
    let entries_start = header_len;
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
    Some(VersionsSnapshot {
        covered_lsn,
        versions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mtxdb-versions-{label}-{}-{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn round_trips_sorted_versions_and_coverage() {
        let path = temp_path("roundtrip");
        let versions = vec![([9u8; 16], 7), ([1u8; 16], 42)];
        write_versions(&path, &versions, 1234).unwrap();
        let snapshot = read_versions(&path).unwrap();
        assert_eq!(snapshot.covered_lsn, 1234);
        assert_eq!(read_coverage(&path), Some(1234));
        let mut expected = versions.clone();
        expected.sort_unstable_by_key(|(collection_id, _)| *collection_id);
        assert_eq!(snapshot.versions, expected, "decoded in sorted file order");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn accepts_the_coverage_less_format_as_zero() {
        let path = temp_path("v1");
        // Build a v1 image by hand: the current writer always emits v2.
        let versions: [([u8; 16], u64); 1] = [([3u8; 16], 5)];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V1);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&versions[0].0);
        bytes.extend_from_slice(&versions[0].1.to_le_bytes());
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let snapshot = read_versions(&path).unwrap();
        assert_eq!(snapshot.covered_lsn, 0);
        assert_eq!(snapshot.versions, versions.to_vec());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn corrupt_and_truncated_files_are_absent() {
        assert!(decode_versions(b"").is_none());
        let path = temp_path("corrupt");
        write_versions(&path, &[([3u8; 16], 5)], 9).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(decode_versions(&bytes).is_none());
        assert!(decode_versions(&bytes[..bytes.len() - 1]).is_none());
        let _ = fs::remove_file(path);
    }
}
