//! Cross-process publish generation for the read-committed overlay.
//!
//! A read-only worker proves its journal overlay is still current by stat-ing
//! the writer's segment on every [`get_read_committed`] call. That
//! `fs::metadata` (plus the tail comparison it gates) dominates the per-call
//! cost — roughly 90-113 us, on par with or worse than an indexed SQL query —
//! even when nothing changed between back-to-back reads.
//!
//! The writer already knows the answer for free. It bumps a monotonic revision
//! both when a group's commit trailer lands (`publish_groups`) and whenever a
//! reclaim replaces the segment, and mirrors it into a tiny memory-mapped file
//! beside the segment so a worker can read it with a plain atomic load: no
//! syscall, no lock, and no staleness. Only when the sampled generation
//! differs does the worker fall back to the full stat-and-scan refresh.
//!
//! The reclaim bump is not optional. `Journal::reclaim_through` rewrites the
//! segment in place with a new base LSN and a replaced inode, but need not
//! publish a group. A reader that only watched publishes could skip the
//! `NeedsReload`/base-jump check on an unchanged signal and serve an index
//! older than the reclaimed prefix. Watching a revision that both events
//! advance closes that gap.
//!
//! # Why the durability mark is not reused
//!
//! A segment already carries a monotonic durability mark, but it advances on
//! `fsync` (`Journal::write_mark`), not at the read-committed visibility point.
//! A worker reading a *published-but-not-yet-durable* group is an explicitly
//! supported shape (see `tests/shared_wal_processes.rs`), so gating on the mark
//! would lag that visibility. This signal is published at the append boundary
//! instead.
//!
//! # Format
//!
//! Sixteen bytes: `epoch(8) | revision(8)`, little-endian on disk and read as
//! native-endian atomics. `epoch` is regenerated from OS entropy every time a
//! coordinator opens the segment, so a writer restart that discards a
//! visible-but-undurable group can never leave a worker holding a generation
//! that a later incarnation reissues: the epoch alone forces a resync. Within
//! one epoch `revision` only ever increases.
//!
//! The file is purely a cache-invalidation hint. Its absence (an older store,
//! or a worker that opened before the writer) disables the fast path and the
//! worker keeps the existing stat-based refresh, so correctness never depends
//! on it.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::entropy;

/// Name of the generation file, written beside the journal segment.
pub(crate) const PUBLISH_SIGNAL_FILE: &str = "read_committed.gen";

/// `epoch(8) | revision(8)`.
const SIGNAL_LEN: u64 = 16;
const EPOCH_OFFSET: usize = 0;
const REVISION_OFFSET: usize = 8;

/// A memory-mapped `(epoch, visible_lsn)` pair: the writer stores it as groups
/// become visible, a worker samples it to decide whether a refresh is needed.
pub(crate) struct PublishSignal {
    mapping: Mapping,
}

enum Mapping {
    Writable(memmap2::MmapMut),
    ReadOnly(memmap2::Mmap),
}

impl PublishSignal {
    /// The generation file that belongs to the journal segment at `segment`.
    ///
    /// It sits beside the segment, so a shared root segment
    /// (`<root>/wal.bin`) and a per-pool segment (`<base_dir>/wal.bin`) each
    /// get their own signal without the worker needing to know which layout it
    /// is in: it derives the path from the segment it was already handed.
    pub(crate) fn path_for(segment: &Path) -> PathBuf {
        segment.with_file_name(PUBLISH_SIGNAL_FILE)
    }

    /// Create the generation file if absent and map it writable, installing a
    /// fresh writer epoch.
    ///
    /// Regenerating the epoch on every open is what makes a restart safe: a
    /// worker that recorded the previous incarnation's epoch always observes a
    /// different value and resyncs, even if the recovered `visible_lsn` happens
    /// to equal the one it last saw.
    pub(crate) fn writer(segment: &Path) -> io::Result<Self> {
        let path = Self::path_for(segment);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        if file.metadata()?.len() < SIGNAL_LEN {
            file.set_len(SIGNAL_LEN)?;
        }
        let mapping = map_writable(&file)?;
        let signal = Self {
            mapping: Mapping::Writable(mapping),
        };
        let mut epoch = [0_u8; 8];
        entropy::fill(&mut epoch)?;
        // Zero is reserved: a never-written slot reads as epoch 0, which no
        // live writer ever installs, so a partially created file is never
        // mistaken for a valid generation.
        let epoch = u64::from_le_bytes(epoch) | 1;
        signal.atomic(EPOCH_OFFSET).store(epoch, Ordering::SeqCst);
        signal.atomic(REVISION_OFFSET).store(0, Ordering::SeqCst);
        Ok(signal)
    }

    /// Map the generation file read-only, or `Ok(None)` when it is absent or
    /// not yet fully sized (so the caller falls back to the stat path).
    pub(crate) fn reader(segment: &Path) -> io::Result<Option<Self>> {
        let path = Self::path_for(segment);
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if file.metadata()?.len() < SIGNAL_LEN {
            return Ok(None);
        }
        let mapping = map_readable(&file)?;
        Ok(Some(Self {
            mapping: Mapping::ReadOnly(mapping),
        }))
    }

    /// Sample `(epoch, revision)` as a consistent pair.
    ///
    /// `revision` only increases within an epoch and `epoch` only changes on
    /// writer restart, so reading the revision around the epoch and retrying on
    /// a mismatch can never return a pair that mixes two epochs.
    pub(crate) fn snapshot(&self) -> (u64, u64) {
        let epoch = self.atomic(EPOCH_OFFSET);
        let revision = self.atomic(REVISION_OFFSET);
        loop {
            let before = revision.load(Ordering::Acquire);
            let epoch_now = epoch.load(Ordering::Acquire);
            let after = revision.load(Ordering::Acquire);
            if before == after {
                return (epoch_now, before);
            }
        }
    }

    /// Advance the revision. Called by the writer whenever something a reader
    /// must notice happens: a group's commit trailer lands, or a reclaim
    /// replaces the segment.
    pub(crate) fn bump(&self) {
        self.atomic(REVISION_OFFSET).fetch_add(1, Ordering::Release);
    }

    fn base(&self) -> *const u8 {
        match &self.mapping {
            Mapping::Writable(mapping) => mapping.as_ptr(),
            Mapping::ReadOnly(mapping) => mapping.as_ptr(),
        }
    }

    #[allow(unsafe_code)]
    // An mmap base is page-aligned and `offset` is 0 or 8, so the cast is
    // 8-byte aligned; clippy cannot see that through the raw pointer.
    #[allow(clippy::cast_ptr_alignment)]
    fn atomic(&self, offset: usize) -> &AtomicU64 {
        // SAFETY: `writer`/`reader` only construct a signal after ensuring the
        // file is at least `SIGNAL_LEN` bytes, and the mapping covers the whole
        // file, so both `offset` slots are in bounds. An mmap base is
        // page-aligned, so a `+0`/`+8` offset is 8-byte aligned for `AtomicU64`.
        // The mapping is owned by `self` and never truncated or remapped, so
        // the reference cannot dangle. `AtomicU64` has the same layout as the
        // eight native-endian bytes at that offset, which is how both the
        // constructor stores and every accessor loads them.
        unsafe { &*self.base().add(offset).cast::<AtomicU64>() }
    }
}

/// Map the signal file writable, isolating the unsafe call (see `map_pack`).
///
/// # Safety
/// The caller must hold the file open for the mapping's lifetime and must not
/// shrink it while the mapping exists. Both hold: the returned `MmapMut` owns
/// the mapping and the file is only ever grown to `SIGNAL_LEN` at creation.
#[allow(unsafe_code)]
fn map_writable(file: &File) -> io::Result<memmap2::MmapMut> {
    // SAFETY: a `SIGNAL_LEN`-sized region of a read/write file, kept alive by
    // the returned mapping and never shrunk while mapped.
    unsafe { memmap2::MmapMut::map_mut(file) }
}

/// Map the signal file read-only, isolating the unsafe call.
///
/// # Safety
/// The caller must hold the file open for the mapping's lifetime and must not
/// shrink it while the mapping exists. Both hold: the returned `Mmap` owns the
/// mapping and a writer only ever grows the file to `SIGNAL_LEN`.
#[allow(unsafe_code)]
fn map_readable(file: &File) -> io::Result<memmap2::Mmap> {
    // SAFETY: a `SIGNAL_LEN`-sized region of a read-only file, kept alive by
    // the returned mapping and never shrunk while mapped.
    unsafe { memmap2::Mmap::map(file) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_segment(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mtxdb-publish-signal-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("wal.bin")
    }

    #[test]
    fn a_reader_sees_a_writers_bump_and_a_reopens_new_epoch() {
        let segment = temp_segment("bump");
        let writer = PublishSignal::writer(&segment).unwrap();
        let reader = PublishSignal::reader(&segment).unwrap().unwrap();

        let (epoch, revision) = reader.snapshot();
        assert_ne!(epoch, 0, "a live writer installs a nonzero epoch");
        assert_eq!(writer.snapshot(), (epoch, revision));

        writer.bump();
        let (epoch_after, revision_after) = reader.snapshot();
        assert_eq!(epoch_after, epoch, "a bump must not change the epoch");
        assert_eq!(
            revision_after,
            revision + 1,
            "a reader must observe exactly the writer's bumps"
        );

        // A re-opened writer is a new incarnation: fresh epoch, revision reset.
        let reopened = PublishSignal::writer(&segment).unwrap();
        let (epoch_new, revision_new) = reader.snapshot();
        assert_ne!(epoch_new, epoch, "a reopen must install a new epoch");
        assert_eq!(revision_new, 0);
        assert_eq!(reopened.snapshot(), (epoch_new, revision_new));
    }

    #[test]
    fn a_reader_for_an_absent_segment_is_none() {
        let segment = temp_segment("absent");
        assert!(PublishSignal::reader(&segment).unwrap().is_none());
    }
}
