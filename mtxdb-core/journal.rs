//! Append-only, checksummed commit journal used by the packfile WAL path.
//!
//! This module owns the journal's disk framing and durability boundary. It
//! also provides [`JournalCoordinator`](crate::journal::JournalCoordinator),
//! which captures each sync caller's
//! target LSN and releases it only after a durable group covers that target.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use crc32fast::Hasher;

const FILE_MAGIC: &[u8; 8] = b"MTXWAL01";
/// Bumped to 2 when the header gained the base sequence/LSN a rotated segment
/// needs to keep numbering global across a rewrite.
const FILE_VERSION: u32 = 2;
// magic(8) + version(4) + base_sequence(8) + base_lsn(8) + header CRC(4).
const FILE_HEADER_LEN: usize = 32;
const GROUP_MAGIC: &[u8; 4] = b"MWG1";
const GROUP_HEADER_LEN: usize = 48;
const GROUP_COMMIT_MAGIC: &[u8; 4] = b"CMIT";
const GROUP_TRAILER_LEN: usize = 16;
const FRAME_FIXED_LEN: usize = 48;
const FRAME_TRAILER_LEN: usize = 4;
/// Smallest possible encoded mutation frame: fixed fields plus its CRC.
const MIN_FRAME_LEN: usize = FRAME_FIXED_LEN + FRAME_TRAILER_LEN;
const MAX_GROUP_LEN: u64 = 256 << 20;
const MAX_SEGMENT_LEN: u64 = (256 << 20) + FILE_HEADER_LEN as u64;

/// A durable mutation represented in a journal group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mutation {
    /// Insert a content-addressed record.
    Put {
        /// Collection containing the record.
        collection_id: [u8; 16],
        /// Record identity.
        node_id: [u8; 16],
        /// Encoded record payload.
        payload: Vec<u8>,
    },
    /// Remove a collection. This is a journal operation because a replayed
    /// put must not resurrect a collection deleted by a later committed LSN.
    DeleteCollection {
        /// Collection to remove.
        collection_id: [u8; 16],
    },
}

/// One decoded mutation frame, with the byte range it occupied in the segment
/// so a reader can re-read and re-validate the frame it has already trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalEntry {
    /// Monotonic LSN of this mutation.
    pub lsn: u64,
    /// Offset of the frame's first byte within the segment file.
    pub offset: u64,
    /// Encoded frame length in bytes (fixed fields + payload + CRC).
    pub frame_len: u64,
    /// Decoded mutation.
    pub mutation: Mutation,
}

/// One recovered, complete, committed group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedGroup {
    /// Monotonic group sequence, starting at 1.
    pub sequence: u64,
    /// First mutation LSN in this group.
    pub first_lsn: u64,
    /// Last mutation LSN in this group.
    pub last_lsn: u64,
    /// Entries in LSN order.
    pub entries: Vec<JournalEntry>,
}

/// Receipt returned after a group and its commit trailer are durably synced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitReceipt {
    /// Monotonic group sequence.
    pub sequence: u64,
    /// First mutation LSN in the group.
    pub first_lsn: u64,
    /// Last mutation LSN in the group.
    pub last_lsn: u64,
}

/// Result of validating a journal file.
#[derive(Debug)]
pub struct Scan {
    /// Complete committed groups in sequence order.
    pub groups: Vec<CommittedGroup>,
    /// Byte offset immediately after the last complete committed group.
    pub valid_len: u64,
    /// True when bytes after `valid_len` were an incomplete final append.
    pub truncated_tail: bool,
}

/// Outcome of compacting a journal segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reclaim {
    /// Complete committed groups left in the segment because the checkpoint had
    /// not yet materialized them.
    pub retained_groups: u64,
    /// Bytes removed from the segment.
    pub reclaimed_bytes: u64,
}

/// A single-writer journal file. Calls to [`Self::commit_group`] are
/// serialized by the caller's coordinator (or by `&mut self`) and return only
/// after the group is durably synced.
pub struct Journal {
    path: PathBuf,
    file: File,
    next_sequence: u64,
    next_lsn: u64,
    poisoned: bool,
}

/// Serializes mutation publication and durable commits for one journal.
///
/// Mutations are assigned LSNs under a short queue lock. A sync caller captures
/// the published LSN, detaches the covered mutations, then commits them while
/// holding the journal lock. Publishers can queue later mutations during the
/// fsync, and concurrent sync callers recheck the committed LSN after taking
/// the journal lock.
pub struct JournalCoordinator {
    journal: Mutex<Journal>,
    pending: Mutex<Vec<(u64, Mutation)>>,
    /// Next LSN to assign. Advanced under `pending`, independently of journal
    /// I/O, so a publish never blocks behind a sync's fsync.
    next_lsn: AtomicU64,
    published_lsn: AtomicU64,
    committed_lsn: AtomicU64,
    /// Mirrors the journal's poison bit, so `publish` can reject without
    /// taking the `journal` mutex (which a sync holds across its fsync).
    poisoned: AtomicBool,
}

impl JournalCoordinator {
    /// Build a coordinator from an opened journal and its recovery scan.
    #[must_use]
    pub fn new(journal: Journal, scan: &Scan) -> Self {
        let committed_lsn = scan.groups.last().map_or(0, |group| group.last_lsn);
        let next_lsn = journal.next_lsn;
        Self {
            journal: Mutex::new(journal),
            pending: Mutex::new(Vec::new()),
            next_lsn: AtomicU64::new(next_lsn),
            published_lsn: AtomicU64::new(committed_lsn),
            committed_lsn: AtomicU64::new(committed_lsn),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Highest durably committed LSN. Everything at or below this is on disk.
    #[must_use]
    pub fn committed_lsn(&self) -> u64 {
        self.committed_lsn.load(Ordering::Acquire)
    }

    /// Highest published (assigned) LSN, committed or not.
    #[must_use]
    pub fn published_lsn(&self) -> u64 {
        self.published_lsn.load(Ordering::Acquire)
    }

    /// Assign an LSN, publish the mutation to the caller's live overlay, and
    /// queue it for the next covering durable group.
    ///
    /// The callback runs while publication is serialized and before the LSN
    /// becomes visible to sync callers. It must not call back into this
    /// coordinator. Keeping overlay publication in this critical section
    /// prevents a successful sync from racing ahead of a not-yet-visible put.
    ///
    /// # Errors
    /// Returns an error if the journal has been poisoned by a failed commit or
    /// if its LSN space is exhausted.
    pub fn publish(
        &self,
        mutation: Mutation,
        publish_overlay: impl FnOnce(u64),
    ) -> io::Result<u64> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        let mut pending = self.pending.lock();
        let lsn = self.next_lsn.load(Ordering::Relaxed);
        let next_lsn = lsn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
        publish_overlay(lsn);
        pending.push((lsn, mutation));
        self.next_lsn.store(next_lsn, Ordering::Relaxed);
        self.published_lsn.store(lsn, Ordering::Release);
        Ok(lsn)
    }

    /// Capture the latest fully published mutation as this sync caller's
    /// acknowledgement boundary.
    #[must_use]
    pub fn capture_sync_target(&self) -> u64 {
        self.published_lsn.load(Ordering::Acquire)
    }

    /// Durably commit pending mutations through `target_lsn`.
    ///
    /// Calls with the same or an earlier target are covered by an already
    /// completed group. Later published mutations are left for a later group.
    /// The journal mutex serializes the disk commit; concurrent sync callers
    /// wait for it and then recheck the committed LSN before deciding whether
    /// to write.
    ///
    /// # Errors
    /// Returns an error if the target was never published, the journal is
    /// poisoned, or writing/syncing the covering group fails.
    pub fn sync_through(&self, target_lsn: u64) -> io::Result<Option<CommitReceipt>> {
        if target_lsn == 0 {
            return Ok(None);
        }
        if target_lsn > self.published_lsn.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sync target has not been published",
            ));
        }
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            return Ok(None);
        }
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }

        let mut journal = self.journal.lock();
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            return Ok(None);
        }
        let batch = {
            let mut pending = self.pending.lock();
            let covered_count = pending
                .iter()
                .take_while(|(lsn, _)| *lsn <= target_lsn)
                .count();
            if covered_count == 0 {
                // Another sync may have committed and drained this target
                // after our first check but before we acquired the journal
                // lock. In that case its durable group covers this caller.
                if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
                    return Ok(None);
                }
                return Err(io::Error::other(
                    "published sync target has no pending journal mutations",
                ));
            }
            let first_lsn = pending[0].0;
            let last_lsn = covered_count
                .checked_sub(1)
                .and_then(|last_index| pending.get(last_index))
                .map(|(lsn, _)| *lsn)
                .ok_or_else(|| io::Error::other("pending journal batch is incomplete"))?;
            if first_lsn != journal.next_lsn || last_lsn != target_lsn {
                return Err(io::Error::other(
                    "journal pending LSN sequence does not cover sync target",
                ));
            }
            pending.drain(..covered_count).collect::<Vec<_>>()
        };
        let mutations: Vec<Mutation> = batch.iter().map(|(_, mutation)| mutation.clone()).collect();
        let receipt = match journal.commit_group(&mutations) {
            Ok(receipt) => receipt,
            Err(error) => {
                if journal.poisoned {
                    self.poisoned.store(true, Ordering::Release);
                } else {
                    let mut pending = self.pending.lock();
                    pending.splice(0..0, batch);
                }
                return Err(error);
            }
        };
        // The group is durable, so record its extent before checking the
        // receipt. This prevents a retry from re-appending committed entries.
        self.committed_lsn
            .store(receipt.last_lsn, Ordering::Release);
        let committed_count = usize::try_from(
            receipt
                .last_lsn
                .saturating_sub(receipt.first_lsn)
                .saturating_add(1),
        )
        .unwrap_or(batch.len())
        .min(batch.len());
        if committed_count < batch.len() {
            let mut pending = self.pending.lock();
            pending.splice(0..0, batch.into_iter().skip(committed_count));
        }
        if receipt.last_lsn != target_lsn || committed_count != mutations.len() {
            return Err(io::Error::other(
                "durable journal group did not cover requested sync target",
            ));
        }
        Ok(Some(receipt))
    }

    /// Capture the current boundary and wait for a durable group covering it.
    ///
    /// # Errors
    /// Returns an error if committing the captured boundary fails.
    pub fn sync(&self) -> io::Result<Option<CommitReceipt>> {
        let target = self.capture_sync_target();
        self.sync_through(target)
    }

    /// Compact the segment, dropping committed groups at or below
    /// `covered_lsn` while preserving any newer groups.
    ///
    /// Intended to run after a checkpoint durably records that LSN as applied,
    /// so it reclaims only data the packfiles already represent. Mutations that
    /// are published but not yet committed are untouched: they live in memory
    /// until the next [`Self::sync_through`]. Numbering is preserved across the
    /// rewrite.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned or the segment cannot be
    /// rewritten.
    pub fn reclaim_through(&self, covered_lsn: u64) -> io::Result<Reclaim> {
        let mut journal = self.journal.lock();
        journal.reclaim_through(covered_lsn)
    }
}

impl Journal {
    /// Open a journal and recover its committed groups.
    ///
    /// A segment shorter than the file header (a fresh file, or a crash while
    /// the header was being written) is reset and re-initialised: no group can
    /// exist without a fully synced header, so there is no acknowledged data
    /// to lose. An incomplete final group is truncated. A malformed
    /// *committed* group fails open and is never silently skipped.
    ///
    /// # Errors
    /// Returns `io::Error` if the segment cannot be created or read, if a
    /// data-bearing segment's header is invalid, if any committed group fails
    /// validation, or if the segment exceeds `MAX_SEGMENT_LEN`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<(Self, Scan)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        let len = file.metadata()?.len();
        if len < FILE_HEADER_LEN as u64 {
            // Empty (fresh) or a crash while writing the header: no committed
            // group can exist yet, so reset and re-initialise rather than
            // failing open forever on a partial header.
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            write_file_header(&mut file, 1, 1)?;
            file.sync_all()?;
            sync_parent_dir(&path)?;
        }

        let bytes = fs::read(&path)?;
        let (base_sequence, base_lsn) = validate_file_header(&bytes)?;
        if bytes.len() as u64 > MAX_SEGMENT_LEN {
            return Err(invalid_data("journal segment exceeds the 256 MiB limit"));
        }
        let scan = scan_bytes(&bytes, base_sequence, base_lsn)?;
        if scan.truncated_tail {
            file.set_len(scan.valid_len)?;
        }
        file.seek(SeekFrom::Start(scan.valid_len))?;
        let next_sequence = scan
            .groups
            .last()
            .map_or(base_sequence, |group| group.sequence.saturating_add(1));
        let next_lsn = scan
            .groups
            .last()
            .map_or(base_lsn, |group| group.last_lsn.saturating_add(1));

        Ok((
            Self {
                path,
                file,
                next_sequence,
                next_lsn,
                poisoned: false,
            },
            scan,
        ))
    }

    /// Append a non-empty group and durably sync it. Each mutation receives
    /// its own monotonic LSN. If writing or syncing fails, this handle is
    /// poisoned: the caller must reopen and rescan before attempting another
    /// commit, because the failed group's on-disk state is uncertain.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned, the group is empty or
    /// oversized, the LSN/sequence space is exhausted, the segment is full
    /// (`WouldBlock`; the caller must drain/rotate), or the write or durable
    /// sync fails.
    pub fn commit_group(&mut self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        if mutations.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot commit an empty journal group",
            ));
        }

        let record_count = u32::try_from(mutations.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "too many journal mutations")
        })?;
        let first_lsn = self.next_lsn;
        let last_lsn = first_lsn
            .checked_add(u64::from(record_count).saturating_sub(1))
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
        let sequence = self.next_sequence;

        let mut payload = Vec::new();
        for (index, mutation) in mutations.iter().enumerate() {
            let lsn = first_lsn
                .checked_add(u64::try_from(index).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mutation index exceeds u64")
                })?)
                .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
            encode_mutation(lsn, mutation, &mut payload)?;
            if u64::try_from(payload.len()).unwrap_or(u64::MAX) > MAX_GROUP_LEN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "journal group exceeds the 256 MiB format limit",
                ));
            }
        }
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "group too large"))?;
        if payload_len > MAX_GROUP_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal group exceeds the 256 MiB segment limit",
            ));
        }

        let header = encode_group_header(sequence, first_lsn, last_lsn, payload_len, record_count);
        let mut group_checksum = Hasher::new();
        group_checksum.update(&header);
        group_checksum.update(&payload);
        let trailer = encode_group_trailer(sequence, group_checksum.finalize());

        let group_size = u64::try_from(
            header
                .len()
                .saturating_add(payload.len())
                .saturating_add(trailer.len()),
        )
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "group too large"))?;
        let segment_size = self.file.metadata()?.len();
        if segment_size.saturating_add(group_size) > MAX_SEGMENT_LEN {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "journal segment is full; drain/rotate before accepting more writes",
            ));
        }
        let next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal group sequence exhausted"))?;
        let next_lsn = last_lsn
            .checked_add(1)
            .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;

        let write_result = (|| -> io::Result<()> {
            self.file.write_all(&header)?;
            self.file.write_all(&payload)?;
            self.file.write_all(&trailer)?;
            self.file.sync_all()
        })();
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(error);
        }

        self.next_sequence = next_sequence;
        self.next_lsn = next_lsn;

        Ok(CommitReceipt {
            sequence,
            first_lsn,
            last_lsn,
        })
    }

    /// Path of this journal file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Compact this segment, dropping committed groups the checkpoint has
    /// already materialized.
    ///
    /// Every group with `last_lsn <= covered_lsn` is removed; the survivors are
    /// re-encoded with their original sequence and LSNs into a temporary file,
    /// durably synced, and atomically renamed over the active path, after which
    /// the parent directory is synced. A crash before the rename leaves the old
    /// segment intact and the temp file inert. Because the header records the
    /// surviving base numbering, the next [`Self::commit_group`] continues past
    /// the last committed LSN rather than restarting at 1.
    ///
    /// A checkpoint that has not advanced past the current segment (or has
    /// already reclaimed it) is a no-op.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned, the segment cannot be
    /// read or re-encoded, or the replacement cannot be synced or renamed.
    pub fn reclaim_through(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        let bytes = fs::read(&self.path)?;
        let (base_sequence, base_lsn) = validate_file_header(&bytes)?;
        let scan = scan_bytes(&bytes, base_sequence, base_lsn)?;
        let retained = scan
            .groups
            .iter()
            .filter(|group| group.last_lsn > covered_lsn)
            .collect::<Vec<_>>();
        if retained.len() == scan.groups.len() {
            return Ok(Reclaim {
                retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
            });
        }

        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let mut rebuilt = file_header_bytes(new_base_sequence, new_base_lsn);
        for group in &retained {
            encode_group(group, &mut rebuilt)?;
        }

        let temp_path = self.path.with_extension("rotate");
        let write_result = (|| -> io::Result<()> {
            let mut temp = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)?;
            temp.write_all(&rebuilt)?;
            temp.sync_all()?;
            drop(temp);
            fs::rename(&temp_path, &self.path)?;
            sync_parent_dir(&self.path)
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }
        // Point the live handle at the replaced segment. A failure here means
        // the handle no longer tracks the durable file, so poison it rather
        // than append to a stale inode.
        match OpenOptions::new().read(true).write(true).open(&self.path) {
            Ok(file) => self.file = file,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        }
        self.file.seek(SeekFrom::End(0))?;
        Ok(Reclaim {
            retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
            reclaimed_bytes: u64::try_from(bytes.len().saturating_sub(rebuilt.len()))
                .unwrap_or(u64::MAX),
        })
    }
}

fn file_header_bytes(base_sequence: u64, base_lsn: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(FILE_HEADER_LEN);
    header.extend_from_slice(FILE_MAGIC);
    header.extend_from_slice(&FILE_VERSION.to_le_bytes());
    header.extend_from_slice(&base_sequence.to_le_bytes());
    header.extend_from_slice(&base_lsn.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header);
    header.extend_from_slice(&crc.finalize().to_le_bytes());
    header
}

fn write_file_header(file: &mut File, base_sequence: u64, base_lsn: u64) -> io::Result<()> {
    file.write_all(&file_header_bytes(base_sequence, base_lsn))
}

/// Validate the file header, returning the base `(sequence, lsn)` this segment
/// starts numbering at. A rotated segment records the first surviving group's
/// numbers here so scanning never has to assume the sequence begins at 1.
fn validate_file_header(bytes: &[u8]) -> io::Result<(u64, u64)> {
    let Some(header) = bytes.get(..FILE_HEADER_LEN) else {
        return Err(invalid_data("truncated journal file header"));
    };
    if &header[..8] != FILE_MAGIC {
        return Err(invalid_data("invalid journal magic"));
    }
    if u32::from_le_bytes(header[8..12].try_into().expect("fixed slice")) != FILE_VERSION {
        return Err(invalid_data("unsupported journal version"));
    }
    let mut crc = Hasher::new();
    crc.update(&header[..28]);
    if u32::from_le_bytes(header[28..32].try_into().expect("fixed slice")) != crc.finalize() {
        return Err(invalid_data("journal header checksum mismatch"));
    }
    Ok((
        u64::from_le_bytes(header[12..20].try_into().expect("fixed slice")),
        u64::from_le_bytes(header[20..28].try_into().expect("fixed slice")),
    ))
}

fn encode_group_header(
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
) -> [u8; GROUP_HEADER_LEN] {
    let mut header = [0_u8; GROUP_HEADER_LEN];
    let header_len = u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN must fit in a u32");
    header[..4].copy_from_slice(GROUP_MAGIC);
    header[4..8].copy_from_slice(&header_len.to_le_bytes());
    header[8..16].copy_from_slice(&sequence.to_le_bytes());
    header[16..24].copy_from_slice(&first_lsn.to_le_bytes());
    header[24..32].copy_from_slice(&last_lsn.to_le_bytes());
    header[32..40].copy_from_slice(&payload_len.to_le_bytes());
    header[40..44].copy_from_slice(&record_count.to_le_bytes());
    let mut crc = Hasher::new();
    crc.update(&header[..44]);
    header[44..48].copy_from_slice(&crc.finalize().to_le_bytes());
    header
}

fn encode_group_trailer(sequence: u64, group_crc: u32) -> [u8; GROUP_TRAILER_LEN] {
    let mut trailer = [0_u8; GROUP_TRAILER_LEN];
    trailer[..4].copy_from_slice(GROUP_COMMIT_MAGIC);
    trailer[4..12].copy_from_slice(&sequence.to_le_bytes());
    trailer[12..16].copy_from_slice(&group_crc.to_le_bytes());
    trailer
}

/// Re-encode a committed group with its original sequence and LSNs, for a
/// segment rewrite. Mirrors exactly the framing `commit_group` writes.
fn encode_group(group: &CommittedGroup, into: &mut Vec<u8>) -> io::Result<()> {
    let mut payload = Vec::new();
    for entry in &group.entries {
        encode_mutation(entry.lsn, &entry.mutation, &mut payload)?;
    }
    let record_count = u32::try_from(group.entries.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many journal mutations"))?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "journal group too large"))?;
    let header = encode_group_header(
        group.sequence,
        group.first_lsn,
        group.last_lsn,
        payload_len,
        record_count,
    );
    let mut group_crc = Hasher::new();
    group_crc.update(&header);
    group_crc.update(&payload);
    let trailer = encode_group_trailer(group.sequence, group_crc.finalize());
    into.extend_from_slice(&header);
    into.extend_from_slice(&payload);
    into.extend_from_slice(&trailer);
    Ok(())
}

fn encode_mutation(lsn: u64, mutation: &Mutation, into: &mut Vec<u8>) -> io::Result<()> {
    let start = into.len();
    match mutation {
        Mutation::Put {
            collection_id,
            node_id,
            payload,
        } => {
            into.push(1);
            into.push(0);
            into.extend_from_slice(&0_u16.to_le_bytes());
            into.extend_from_slice(&lsn.to_le_bytes());
            into.extend_from_slice(collection_id);
            into.extend_from_slice(node_id);
            let payload_len = u32::try_from(payload.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "mutation payload exceeds u32")
            })?;
            into.extend_from_slice(&payload_len.to_le_bytes());
            into.extend_from_slice(payload);
        }
        Mutation::DeleteCollection { collection_id } => {
            into.push(2);
            into.push(0);
            into.extend_from_slice(&0_u16.to_le_bytes());
            into.extend_from_slice(&lsn.to_le_bytes());
            into.extend_from_slice(collection_id);
            into.extend_from_slice(&[0_u8; 16]);
            into.extend_from_slice(&0_u32.to_le_bytes());
        }
    }
    let mut crc = Hasher::new();
    crc.update(&into[start..]);
    into.extend_from_slice(&crc.finalize().to_le_bytes());
    Ok(())
}

/// Validated fixed fields of one committed group header.
struct GroupHeader {
    sequence: u64,
    first_lsn: u64,
    last_lsn: u64,
    payload_len: u64,
    record_count: u32,
}

/// Validate a group header's magic, length, and CRC, returning its fields.
fn parse_group_header(header: &[u8]) -> io::Result<GroupHeader> {
    if &header[..4] != GROUP_MAGIC
        || u32::from_le_bytes(header[4..8].try_into().expect("fixed slice"))
            != u32::try_from(GROUP_HEADER_LEN).expect("GROUP_HEADER_LEN fits in u32")
    {
        return Err(invalid_data("invalid journal group header"));
    }
    let mut header_crc = Hasher::new();
    header_crc.update(&header[..44]);
    if u32::from_le_bytes(header[44..48].try_into().expect("fixed slice")) != header_crc.finalize()
    {
        return Err(invalid_data("journal group header checksum mismatch"));
    }
    Ok(GroupHeader {
        sequence: u64::from_le_bytes(header[8..16].try_into().expect("fixed slice")),
        first_lsn: u64::from_le_bytes(header[16..24].try_into().expect("fixed slice")),
        last_lsn: u64::from_le_bytes(header[24..32].try_into().expect("fixed slice")),
        payload_len: u64::from_le_bytes(header[32..40].try_into().expect("fixed slice")),
        record_count: u32::from_le_bytes(header[40..44].try_into().expect("fixed slice")),
    })
}

/// Verify a group's commit trailer and whole-group CRC, returning its payload.
fn verify_group_payload<'a>(
    bytes: &'a [u8],
    header: &[u8],
    group: &GroupHeader,
    payload_start: usize,
    payload_len: usize,
) -> io::Result<&'a [u8]> {
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer_end = payload_end
        .checked_add(GROUP_TRAILER_LEN)
        .ok_or_else(|| invalid_data("journal group length overflow"))?;
    let trailer = bytes
        .get(payload_end..trailer_end)
        .ok_or_else(|| invalid_data("truncated journal commit trailer"))?;
    if &trailer[..4] != GROUP_COMMIT_MAGIC
        || u64::from_le_bytes(trailer[4..12].try_into().expect("fixed slice")) != group.sequence
    {
        return Err(invalid_data("invalid journal commit trailer"));
    }
    let payload = bytes
        .get(payload_start..payload_end)
        .ok_or_else(|| invalid_data("truncated journal group payload"))?;
    let mut group_crc = Hasher::new();
    group_crc.update(header);
    group_crc.update(payload);
    if u32::from_le_bytes(trailer[12..16].try_into().expect("fixed slice")) != group_crc.finalize()
    {
        return Err(invalid_data("journal committed group checksum mismatch"));
    }
    Ok(payload)
}

fn scan_bytes(bytes: &[u8], base_sequence: u64, base_lsn: u64) -> io::Result<Scan> {
    if bytes.len() < FILE_HEADER_LEN {
        return Err(invalid_data("truncated journal file header"));
    }
    let mut cursor = FILE_HEADER_LEN;
    let mut valid_len = cursor;
    let mut groups = Vec::new();
    let mut expected_sequence = base_sequence;
    let mut expected_lsn = base_lsn;
    let mut truncated_tail = false;

    while cursor < bytes.len() {
        let remaining = bytes.len().saturating_sub(cursor);
        if remaining < GROUP_HEADER_LEN {
            truncated_tail = true;
            break;
        }
        let header = bytes
            .get(cursor..cursor.saturating_add(GROUP_HEADER_LEN))
            .ok_or_else(|| invalid_data("truncated journal group header"))?;
        let group = parse_group_header(header)?;
        if group.sequence != expected_sequence
            || group.first_lsn != expected_lsn
            || group.last_lsn < group.first_lsn
            || group
                .last_lsn
                .saturating_sub(group.first_lsn)
                .saturating_add(1)
                != u64::from(group.record_count)
            || group.record_count == 0
            || group.payload_len > MAX_GROUP_LEN
            || u64::from(group.record_count).saturating_mul(MIN_FRAME_LEN as u64)
                > group.payload_len
        {
            return Err(invalid_data("invalid journal sequence or group bounds"));
        }
        let payload_len = usize::try_from(group.payload_len)
            .map_err(|_| invalid_data("journal group length exceeds address space"))?;
        let total_len = GROUP_HEADER_LEN
            .checked_add(payload_len)
            .and_then(|len| len.checked_add(GROUP_TRAILER_LEN))
            .ok_or_else(|| invalid_data("journal group length overflow"))?;
        if remaining < total_len {
            truncated_tail = true;
            break;
        }
        let payload_start = cursor.saturating_add(GROUP_HEADER_LEN);
        let payload = verify_group_payload(bytes, header, &group, payload_start, payload_len)?;
        let entries =
            decode_mutations(payload, group.first_lsn, group.record_count, payload_start)?;
        groups.push(CommittedGroup {
            sequence: group.sequence,
            first_lsn: group.first_lsn,
            last_lsn: group.last_lsn,
            entries,
        });
        cursor = cursor.saturating_add(total_len);
        valid_len = cursor;
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal group sequence overflow"))?;
        expected_lsn = group
            .last_lsn
            .checked_add(1)
            .ok_or_else(|| invalid_data("journal LSN overflow"))?;
    }

    Ok(Scan {
        groups,
        valid_len: u64::try_from(valid_len).unwrap_or(u64::MAX),
        truncated_tail,
    })
}

fn decode_mutations(
    payload: &[u8],
    first_lsn: u64,
    record_count: u32,
    payload_offset: usize,
) -> io::Result<Vec<JournalEntry>> {
    let mut cursor = 0_usize;
    let mut entries = Vec::with_capacity(usize::try_from(record_count).unwrap_or(0));
    for index in 0..record_count {
        let start = cursor;
        if payload.len().saturating_sub(cursor) < MIN_FRAME_LEN {
            return Err(invalid_data("truncated journal mutation frame"));
        }
        let fixed_end = cursor
            .checked_add(FRAME_FIXED_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let frame = payload
            .get(cursor..fixed_end)
            .ok_or_else(|| invalid_data("truncated journal mutation frame"))?;
        let kind = frame[0];
        let flags = frame[1];
        if flags != 0 || frame[2..4] != [0, 0] {
            return Err(invalid_data("unsupported journal mutation flags"));
        }
        let lsn = u64::from_le_bytes(frame[4..12].try_into().expect("fixed slice"));
        let expected = first_lsn
            .checked_add(u64::from(index))
            .ok_or_else(|| invalid_data("journal mutation LSN overflow"))?;
        if lsn != expected {
            return Err(invalid_data("journal mutation LSN out of order"));
        }
        let collection_id: [u8; 16] = frame[12..28].try_into().expect("fixed slice");
        let node_id: [u8; 16] = frame[28..44].try_into().expect("fixed slice");
        let payload_len = usize::try_from(u32::from_le_bytes(
            frame[44..48].try_into().expect("fixed slice"),
        ))
        .map_err(|_| invalid_data("journal mutation payload length exceeds address space"))?;
        let frame_end = fixed_end
            .checked_add(payload_len)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let crc_end = frame_end
            .checked_add(FRAME_TRAILER_LEN)
            .ok_or_else(|| invalid_data("journal mutation length overflow"))?;
        let data = payload
            .get(fixed_end..frame_end)
            .ok_or_else(|| invalid_data("journal mutation payload exceeds group"))?;
        let crc_bytes = payload
            .get(frame_end..crc_end)
            .ok_or_else(|| invalid_data("journal mutation checksum exceeds group"))?;
        let mut crc = Hasher::new();
        crc.update(
            payload
                .get(start..frame_end)
                .ok_or_else(|| invalid_data("journal mutation frame exceeds group"))?,
        );
        if u32::from_le_bytes(crc_bytes.try_into().expect("fixed slice")) != crc.finalize() {
            return Err(invalid_data("journal mutation checksum mismatch"));
        }
        let mutation = match kind {
            1 => Mutation::Put {
                collection_id,
                node_id,
                payload: data.to_vec(),
            },
            2 if payload_len == 0 && node_id == [0; 16] => {
                Mutation::DeleteCollection { collection_id }
            }
            _ => return Err(invalid_data("unknown or malformed journal mutation kind")),
        };
        entries.push(JournalEntry {
            lsn,
            offset: u64::try_from(payload_offset.saturating_add(start)).unwrap_or(u64::MAX),
            frame_len: u64::try_from(crc_end.saturating_sub(start)).unwrap_or(u64::MAX),
            mutation,
        });
        cursor = crc_end;
    }
    if cursor != payload.len() {
        return Err(invalid_data("extra bytes after journal mutation frames"));
    }
    Ok(entries)
}

fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        // Rust's portable std API has no directory-handle durability
        // operation on non-Unix platforms. Refuse to enable the journal until
        // a platform-specific implementation can uphold create/rotation
        // durability; silently succeeding would weaken the contract.
        let _ = parent;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "durable journal creation requires platform directory sync support",
        ))
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{Journal, JournalCoordinator, Mutation};
    use std::fs;
    use std::io::Write as _;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mtxdb_journal_{label}_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    fn put(collection: u8, node: u8, payload: &[u8]) -> Mutation {
        Mutation::Put {
            collection_id: [collection; 16],
            node_id: [node; 16],
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn committed_groups_round_trip_with_contiguous_lsns() {
        let path = temp_path("round_trip");
        let _ = fs::remove_file(&path);
        let (mut journal, initial) = Journal::open(&path).unwrap();
        assert_eq!(initial.groups.len(), 0);
        let committed = journal
            .commit_group(&[
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [2; 16],
                    payload: b"alpha".to_vec(),
                },
                Mutation::DeleteCollection {
                    collection_id: [3; 16],
                },
            ])
            .unwrap();
        assert_eq!((committed.first_lsn, committed.last_lsn), (1, 2));
        drop(journal);

        let (reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].sequence, 1);
        assert_eq!(scan.groups[0].first_lsn, committed.first_lsn);
        assert_eq!(scan.groups[0].last_lsn, committed.last_lsn);
        let entries = &scan.groups[0].entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(
            entries[0].mutation,
            Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"alpha".to_vec(),
            }
        );
        assert_eq!(entries[1].lsn, 2);
        assert_eq!(
            entries[1].mutation,
            Mutation::DeleteCollection {
                collection_id: [3; 16]
            }
        );
        // The recovered byte ranges must let a reader re-read each frame from
        // the segment: the first frame starts right after the file + group
        // headers, and lengths are fixed + payload + CRC.
        let group_payload_start = super::FILE_HEADER_LEN + super::GROUP_HEADER_LEN;
        assert_eq!(entries[0].offset, group_payload_start as u64);
        assert_eq!(
            entries[0].frame_len,
            (super::FRAME_FIXED_LEN + 5 + super::FRAME_TRAILER_LEN) as u64
        );
        assert_eq!(
            entries[1].offset,
            entries[0].offset.saturating_add(entries[0].frame_len)
        );
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn incomplete_final_group_is_truncated_on_open() {
        let path = temp_path("torn_tail");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"complete".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let good_len = fs::metadata(&path).unwrap().len();
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"MWG1\x30\0")
            .unwrap();
        let (reopened, scan) = Journal::open(&path).unwrap();
        assert!(scan.truncated_tail);
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(fs::metadata(&path).unwrap().len(), good_len);
        drop(reopened);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn committed_group_corruption_fails_open() {
        let path = temp_path("corrupt");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"committed".to_vec(),
            }])
            .unwrap();
        drop(journal);

        let mut bytes = fs::read(&path).unwrap();
        let payload_byte = super::FILE_HEADER_LEN
            .saturating_add(super::GROUP_HEADER_LEN)
            .saturating_add(super::FRAME_FIXED_LEN);
        bytes[payload_byte] ^= 0x80;
        fs::write(&path, bytes).unwrap();
        assert!(Journal::open(&path).is_err());
        fs::remove_file(path).unwrap();
    }

    /// A partial header is a crash mid-creation, not corruption: no group can
    /// exist without a synced header, so open resets the segment and it
    /// remains usable.
    #[test]
    fn partial_file_header_is_reset_on_open() {
        let path = temp_path("partial_header");
        let _ = fs::remove_file(&path);
        fs::write(&path, b"MTX").unwrap();

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 0);
        assert!(!scan.truncated_tail);
        let committed = journal
            .commit_group(&[Mutation::Put {
                collection_id: [4; 16],
                node_id: [5; 16],
                payload: b"after-reset".to_vec(),
            }])
            .unwrap();
        assert_eq!(committed.sequence, 1);
        assert_eq!(committed.first_lsn, 1);
        drop(journal);

        let (_reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].entries.len(), 1);
        fs::remove_file(path).unwrap();
    }

    /// Reopen after a torn tail, truncate it, and confirm numbering continues
    /// from the last committed group rather than restarting.
    #[test]
    fn recovery_after_torn_tail_continues_numbering() {
        let path = temp_path("continue");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal
            .commit_group(&[Mutation::Put {
                collection_id: [1; 16],
                node_id: [1; 16],
                payload: b"one".to_vec(),
            }])
            .unwrap();
        drop(journal);

        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"MWG1\x30\0\x30\0")
            .unwrap();

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert!(scan.truncated_tail);
        assert_eq!(scan.groups.len(), 1);
        let committed = journal
            .commit_group(&[
                Mutation::Put {
                    collection_id: [2; 16],
                    node_id: [2; 16],
                    payload: b"two".to_vec(),
                },
                Mutation::DeleteCollection {
                    collection_id: [3; 16],
                },
            ])
            .unwrap();
        assert_eq!(committed.sequence, 2);
        assert_eq!(committed.first_lsn, 2);
        assert_eq!(committed.last_lsn, 3);
        drop(journal);

        let (_reopened, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups[0].last_lsn, 1);
        assert_eq!(scan.groups[1].first_lsn, 2);
        assert_eq!(scan.groups[1].last_lsn, 3);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn coordinator_commits_only_through_the_captured_sync_boundary() {
        let path = temp_path("coordinator_boundary");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);

        let first_lsn = coordinator
            .publish(
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [1; 16],
                    payload: b"first".to_vec(),
                },
                |_| {},
            )
            .unwrap();
        let first_target = coordinator.capture_sync_target();
        assert_eq!(first_target, first_lsn);

        let second_lsn = coordinator
            .publish(
                Mutation::Put {
                    collection_id: [1; 16],
                    node_id: [2; 16],
                    payload: b"second".to_vec(),
                },
                |_| {},
            )
            .unwrap();
        assert_eq!(second_lsn, first_lsn.saturating_add(1));

        let first_commit = coordinator.sync_through(first_target).unwrap().unwrap();
        assert_eq!(first_commit.first_lsn, first_lsn);
        assert_eq!(first_commit.last_lsn, first_target);
        assert!(coordinator.sync_through(first_target).unwrap().is_none());

        let second_commit = coordinator.sync().unwrap().unwrap();
        assert_eq!(second_commit.first_lsn, second_lsn);
        assert_eq!(second_commit.last_lsn, second_lsn);

        drop(coordinator);
        let (_journal, recovered) = Journal::open(&path).unwrap();
        assert_eq!(recovered.groups.len(), 2);
        assert_eq!(recovered.groups[0].last_lsn, first_target);
        assert_eq!(recovered.groups[1].first_lsn, second_lsn);
        fs::remove_file(path).unwrap();
    }

    /// A target that was never published is a caller bug, not a durable commit.
    #[test]
    fn sync_through_rejects_an_unpublished_target() {
        let path = temp_path("coordinator_unpublished");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);

        assert!(coordinator.sync_through(0).unwrap().is_none());
        let error = coordinator.sync_through(1).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        fs::remove_file(path).unwrap();
    }

    /// Many writers publishing and syncing concurrently must produce one
    /// contiguous, gap-free LSN sequence: a caller that loses the commit race
    /// is covered by another caller's group rather than acknowledged early.
    #[test]
    fn concurrent_callers_produce_contiguous_groups() {
        let path = temp_path("coordinator_concurrent");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = std::sync::Arc::new(JournalCoordinator::new(journal, &scan));

        let threads = 8_u8;
        let per_thread = 16_u8;
        let mut handles = Vec::new();
        for thread in 0..threads {
            let coordinator = std::sync::Arc::clone(&coordinator);
            handles.push(std::thread::spawn(move || {
                for item in 0..per_thread {
                    coordinator
                        .publish(
                            Mutation::Put {
                                collection_id: [thread; 16],
                                node_id: [item; 16],
                                payload: b"payload".to_vec(),
                            },
                            |_| {},
                        )
                        .unwrap();
                    // Either this caller commits the group or a concurrent
                    // caller already committed one covering this LSN.
                    coordinator.sync().unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let published = coordinator.capture_sync_target();
        assert_eq!(
            published,
            u64::from(u32::from(threads) * u32::from(per_thread))
        );
        coordinator.sync().unwrap();

        drop(coordinator);
        let (_journal, recovered) = Journal::open(&path).unwrap();
        let total: u64 = recovered
            .groups
            .iter()
            .map(|group| u64::try_from(group.entries.len()).unwrap())
            .sum();
        assert_eq!(total, published);
        let mut expected_first = 1;
        for group in &recovered.groups {
            assert_eq!(group.first_lsn, expected_first);
            expected_first = group.last_lsn.saturating_add(1);
        }
        fs::remove_file(path).unwrap();
    }

    /// Rotation drops covered groups but keeps the uncovered suffix, preserving
    /// each surviving group's original sequence and LSNs.
    #[test]
    fn reclaim_retains_uncovered_suffix_and_preserves_numbering() {
        let path = temp_path("reclaim_suffix");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        let first = journal.commit_group(&[put(1, 1, b"first")]).unwrap();
        let second = journal
            .commit_group(&[
                put(1, 2, b"second"),
                Mutation::DeleteCollection {
                    collection_id: [9; 16],
                },
            ])
            .unwrap();
        let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();

        let reclaim = journal.reclaim_through(first.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 2);
        assert!(reclaim.reclaimed_bytes > 0);
        assert!(!path.with_extension("rotate").exists());
        drop(journal);

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 2);
        assert_eq!(scan.groups[0].sequence, second.sequence);
        assert_eq!(scan.groups[0].first_lsn, second.first_lsn);
        assert_eq!(scan.groups[0].last_lsn, second.last_lsn);
        assert_eq!(scan.groups[1].sequence, third.sequence);
        assert_eq!(scan.groups[1].last_lsn, third.last_lsn);

        // Numbering continues from the global high-water mark, not the new base.
        let fourth = journal.commit_group(&[put(1, 4, b"fourth")]).unwrap();
        assert_eq!(fourth.sequence, third.sequence.saturating_add(1));
        assert_eq!(fourth.first_lsn, third.last_lsn.saturating_add(1));
        drop(journal);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 3);
        fs::remove_file(path).unwrap();
    }

    /// Reclaiming everything leaves an empty segment whose base still points
    /// past the last committed LSN, so numbering never restarts.
    #[test]
    fn reclaim_all_covered_keeps_numbering_for_the_next_group() {
        let path = temp_path("reclaim_all");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal.commit_group(&[put(1, 1, b"first")]).unwrap();
        let second = journal.commit_group(&[put(1, 2, b"second")]).unwrap();

        let reclaim = journal.reclaim_through(second.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 0);
        drop(journal);

        let (mut journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 0);
        let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();
        assert_eq!(third.sequence, second.sequence.saturating_add(1));
        assert_eq!(third.first_lsn, second.last_lsn.saturating_add(1));
        drop(journal);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].first_lsn, second.last_lsn.saturating_add(1));
        fs::remove_file(path).unwrap();
    }

    /// A covered LSN at or below the segment base reclaims nothing.
    #[test]
    fn reclaim_below_base_is_a_noop() {
        let path = temp_path("reclaim_noop");
        let _ = fs::remove_file(&path);
        let (mut journal, _) = Journal::open(&path).unwrap();
        journal.commit_group(&[put(1, 1, b"only")]).unwrap();

        let reclaim = journal.reclaim_through(0).unwrap();
        assert_eq!(reclaim.retained_groups, 1);
        assert_eq!(reclaim.reclaimed_bytes, 0);
        drop(journal);
        fs::remove_file(path).unwrap();
    }

    /// The coordinator exposes reclamation without disturbing its LSN counters:
    /// a publish after reclaim still continues the sequence.
    #[test]
    fn coordinator_reclaims_without_reusing_lsns() {
        let path = temp_path("coordinator_reclaim");
        let _ = fs::remove_file(&path);
        let (journal, scan) = Journal::open(&path).unwrap();
        let coordinator = JournalCoordinator::new(journal, &scan);
        let lsn = coordinator.publish(put(1, 1, b"first"), |_| {}).unwrap();
        let receipt = coordinator.sync().unwrap().unwrap();
        assert_eq!(receipt.last_lsn, lsn);

        let reclaim = coordinator.reclaim_through(receipt.last_lsn).unwrap();
        assert_eq!(reclaim.retained_groups, 0);
        assert_eq!(coordinator.committed_lsn(), receipt.last_lsn);

        let next = coordinator.publish(put(1, 2, b"second"), |_| {}).unwrap();
        assert_eq!(next, lsn.saturating_add(1));
        coordinator.sync().unwrap();
        drop(coordinator);

        let (_journal, scan) = Journal::open(&path).unwrap();
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.groups[0].first_lsn, lsn.saturating_add(1));
        fs::remove_file(path).unwrap();
    }
}
