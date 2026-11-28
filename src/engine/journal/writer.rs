//! The journal segment writer: open, append, sync, reclaim, and rewrite.
//!
//! The `Journal` struct, its tuning types, and the coordinator state machine
//! stay in the parent module.

use super::{
    boundary_through, emergency_len_for_cap, encode_group, encode_group_header,
    encode_group_trailer, encode_mark_slot, encode_mutation, file_header_bytes, fs, invalid_data,
    io, marks_from_scan, newest_mark, pool_bit, read_durable_len_from_header, read_file_header,
    read_range_into, scan_bytes, scan_groups_from, sync_parent_dir, validate_file_header,
    write_file_header, CommitReceipt, File, GroupDirectoryStats, GroupMark, HashMap, Hasher,
    Journal, MarkSlot, Mutation, OpenOptions, Path, Read, Reclaim, ReclaimedHistory, Scan, Seek,
    SeekFrom, ShardType, SharedBoundary, Write, FILE_HEADER_LEN, MARK_SLOT_LEN, MARK_SLOT_OFFSETS,
    MAX_GROUP_LEN, MAX_SEGMENT_LEN, POOL_SLOTS, UNPROFITABLE_REWRITE_MIN_COPY,
    UNPROFITABLE_REWRITE_RATIO,
};
#[cfg(test)]
use super::{Arc, RewritePhase, RewritePhaseGuard};

impl Journal {
    /// Scan an existing journal without opening it for writing or repairing a
    /// torn tail. A read-only worker uses this to observe only complete,
    /// committed groups while the writer may still be appending or reclaiming
    /// the segment.
    ///
    /// An incomplete final group is reported in [`Scan::truncated_tail`] and
    /// excluded from `groups`; the file is left byte-for-byte unchanged. A
    /// malformed committed group remains an error.
    ///
    /// A missing file or one shorter than the file header is treated as an
    /// empty segment. This permits a read-only worker to start before the
    /// writer has created the journal, without creating or repairing it.
    ///
    /// # Errors
    /// Returns `io::Error` if the file is unreadable, a complete header is
    /// invalid, the segment is oversized, or a committed group fails
    /// validation.
    pub fn scan_read_only(path: impl AsRef<Path>) -> io::Result<Scan> {
        let path = path.as_ref();
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Scan::empty()),
            Err(error) => return Err(error),
        };
        if bytes.len() < FILE_HEADER_LEN {
            return Ok(Scan::empty());
        }
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_SEGMENT_LEN {
            return Err(invalid_data("journal segment exceeds the 256 MiB limit"));
        }
        let (base_sequence, base_lsn) = validate_file_header(&bytes)?;
        let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        scan_bytes(&bytes, base_sequence, base_lsn, || {
            read_durable_len_from_header(&bytes, file_len)
        })
    }

    /// Read only the segment header's base LSN without scanning journal groups.
    /// A read-only worker uses this after a publish-signal change to detect a
    /// reclaim that rewrote the segment without changing its length.
    pub(crate) fn read_base_lsn_read_only(path: impl AsRef<Path>) -> io::Result<u64> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let len = file.metadata()?.len();
        if len < u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX) {
            return Ok(0);
        }
        let mut header = vec![0; FILE_HEADER_LEN];
        file.read_exact(&mut header)?;
        let (_, base_lsn) = validate_file_header(&header)?;
        Ok(base_lsn)
    }

    /// Scan only the committed groups appended at or after `start`, a group
    /// boundary returned by an earlier scan's `valid_len`.
    ///
    /// Reads only the tail bytes, so a reader that already applied the earlier
    /// groups does not re-decode them. `expected_lsn` is the next LSN the
    /// caller expects (its highest applied LSN plus one); a mismatch means the
    /// segment was replaced or rotated, and the caller should rebuild from a
    /// full [`Self::scan_read_only`]. Never repairs or creates the file.
    ///
    /// # Errors
    /// Returns `io::Error` if the tail is unreadable or a group fails
    /// validation. A missing file yields an empty scan.
    // This intentionally seeks to `start` and reads only the appended tail;
    // `fs::read` would decode the whole segment on every incremental refresh.
    #[allow(clippy::verbose_file_reads)]
    pub fn scan_read_only_from(
        path: impl AsRef<Path>,
        start: u64,
        expected_lsn: u64,
    ) -> io::Result<Scan> {
        Self::scan_read_only_range(path, start, u64::MAX, expected_lsn)
    }

    /// Scan complete groups only through `end`, an absolute group boundary.
    /// This keeps paged replay from reading and decoding later groups that do
    /// not belong to the requested page.
    #[allow(clippy::verbose_file_reads)]
    pub(super) fn scan_read_only_range(
        path: impl AsRef<Path>,
        start: u64,
        end: u64,
        expected_lsn: u64,
    ) -> io::Result<Scan> {
        let path = path.as_ref();
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Scan::empty()),
            Err(error) => return Err(error),
        };
        let len = file.metadata()?.len();
        if len < u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX) {
            return Ok(Scan::empty());
        }
        file.seek(SeekFrom::Start(0))?;
        let mut header = vec![0; FILE_HEADER_LEN];
        file.read_exact(&mut header)?;
        let (base_sequence, base_lsn) = validate_file_header(&header)?;
        let scan_end = end.min(len);
        if start >= scan_end {
            return Ok(Scan {
                groups: Vec::new(),
                valid_len: start.min(scan_end),
                truncated_tail: false,
                base_lsn,
                base_sequence,
                consumed_tail: Vec::new(),
                group_ends: Vec::new(),
            });
        }
        file.seek(SeekFrom::Start(start))?;
        let mut tail = Vec::new();
        file.take(scan_end.saturating_sub(start))
            .read_to_end(&mut tail)?;
        scan_groups_from(
            &tail,
            start,
            0,
            expected_lsn,
            base_lsn,
            base_sequence,
            || read_durable_len_from_header(&header, len),
        )
    }

    /// Read up to `max_len` bytes of the segment ending at `end_offset` from an
    /// already-open `file` into `window`, never reaching back into the file
    /// header. Returns `false` when the file is shorter than `end_offset` or
    /// has no group bytes before it, leaving `window` unspecified.
    ///
    /// A read-only overlay keeps this window from the end of the last group it
    /// consumed and compares it on later refreshes: file length alone cannot
    /// show that the bytes already consumed are still the same ones. The
    /// window is compared byte for byte instead of relying on the group
    /// trailer, because the trailer is not a content fingerprint: the group
    /// checksum covers `header || records` and every record already ends with
    /// its own CRC32, so for groups of the same shape it comes out identical
    /// whatever the payload. The window includes the last record's own CRC,
    /// which does depend on the record's full content.
    ///
    /// Reads through the caller's descriptor with a positioned read, so the
    /// refresh path pays one syscall and reuses `window`'s allocation instead
    /// of opening the file each time. The caller must notice a replaced file
    /// itself (its descriptor keeps the old inode). Never repairs or creates
    /// the file.
    ///
    /// # Errors
    /// Returns `io::Error` if the read fails.
    pub(crate) fn read_tail_ending_at(
        file: &File,
        end_offset: u64,
        max_len: u64,
        window: &mut Vec<u8>,
    ) -> io::Result<bool> {
        let header_len = u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX);
        if end_offset <= header_len {
            return Ok(false);
        }
        let start = end_offset.saturating_sub(max_len).max(header_len);
        let window_len = usize::try_from(end_offset.saturating_sub(start))
            .map_err(|_| invalid_data("tail window exceeds the address space"))?;
        window.clear();
        window.resize(window_len, 0);
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_exact_at(file, window, start);
        #[cfg(not(unix))]
        let read = {
            let mut handle = file;
            handle
                .seek(SeekFrom::Start(start))
                .and_then(|_| handle.read_exact(window))
        };
        match read {
            Ok(()) => Ok(true),
            // The file is shorter than the window's end: the consumed prefix is gone.
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
            Err(error) => Err(error),
        }
    }

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
        Self::open_shared(path)
    }

    /// Open a shared (multi-pool) journal segment, or create one if the file is
    /// absent. The segment is created as the v1 pool-tagged format, so
    /// every mutation frame appended through it must carry a pool tag.
    ///
    /// # Errors
    /// Same as [`Self::open`], plus `InvalidData` if an existing segment has an
    /// unsupported version.
    pub fn open_shared(path: impl AsRef<Path>) -> io::Result<(Self, Scan)> {
        Self::open_shared_with_base(path, 1)
    }

    /// Like [`Self::open_shared`], but when the segment is first created its
    /// LSN space begins at `base_lsn` rather than 1. An existing segment
    /// ignores `base_lsn` and keeps its own.
    ///
    /// Internal to the root-level open path: an arbitrary `base_lsn` would let
    /// a caller create a segment whose numbering does not line up with the
    /// pools' recorded coverage, so this is not public. Callers that open a
    /// database root should go through
    /// [`crate::database::Database::open`], which computes the seed from
    /// the pools' `journal.lsn` and passes it here.
    ///
    /// # Errors
    /// Same as [`Self::open_shared`].
    pub(crate) fn open_shared_with_base(
        path: impl AsRef<Path>,
        base_lsn: u64,
    ) -> io::Result<(Self, Scan)> {
        let base_lsn = base_lsn.max(1);
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
            write_file_header(&mut file, 1, base_lsn, [0; POOL_SLOTS])?;
            file.sync_all()?;
            sync_parent_dir(&path)?;
        }

        let bytes = fs::read(&path)?;
        let header = read_file_header(&bytes)?;
        let (base_sequence, base_lsn) = (header.base_sequence, header.base_lsn);
        if bytes.len() as u64 > MAX_SEGMENT_LEN {
            return Err(invalid_data("journal segment exceeds the 256 MiB limit"));
        }
        let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, || {
            read_durable_len_from_header(&bytes, file_len)
        })?;
        if scan.truncated_tail {
            file.set_len(scan.valid_len)?;
        }
        file.seek(SeekFrom::Start(scan.valid_len))?;
        if scan.truncated_tail || !scan.groups.is_empty() {
            // What recovery keeps may exist only in the page cache (a process
            // crash leaves it there), yet the coordinator will report it
            // committed. Make it durable now, and record that in the hint, so a
            // later sync never claims bytes that were not fsynced. The same
            // fsync makes a truncation durable, including one that dropped
            // every group, so a dropped tail cannot reappear.
            file.sync_all()?;
        }
        let next_sequence = scan
            .groups
            .last()
            .map_or(base_sequence, |group| group.sequence.saturating_add(1));
        let next_lsn = scan
            .groups
            .last()
            .map_or(base_lsn, |group| group.last_lsn.saturating_add(1));

        let mark_generation = newest_mark(&bytes).map_or(0, |slot| slot.generation);
        let mut journal = Self {
            path,
            file,
            file_len: scan.valid_len,
            base_sequence,
            base_lsn,
            reclaimed_history: header.reclaimed,
            mark_generation,
            next_sequence,
            next_lsn,
            poisoned: false,
            segment_cap: MAX_SEGMENT_LEN,
            groups: marks_from_scan(&scan),
            #[cfg(test)]
            rewrite_phase: Arc::new(RewritePhase::new()),
        };
        if !scan.groups.is_empty() {
            // Recovery fsynced what it kept above, so record that in the mark.
            journal.write_mark(scan.valid_len);
        }
        Ok((journal, scan))
    }

    /// Append a non-empty group and its commit trailer, without fsyncing.
    ///
    /// Each mutation receives its own monotonic LSN. The group is complete and
    /// readable by [`Self::scan_read_only`] as soon as this returns, so a
    /// caller can publish a visibility boundary before [`Self::make_durable`]
    /// fsyncs it. If writing fails, this handle is poisoned: the caller must
    /// reopen and rescan before attempting another commit.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned, the group is empty or
    /// oversized, the LSN/sequence space is exhausted, the segment is full
    /// (`WouldBlock`; the caller must drain/rotate), or the write fails.
    pub fn append_group(&mut self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        self.append_group_with_sequence(mutations, None)
    }

    /// Append a group using `sequence` when given, or this segment's own next
    /// sequence otherwise.
    ///
    /// A shared sequence lets several pools' segments be ordered by one global
    /// group sequence; a gap between a segment's consecutive groups is then
    /// expected, and recovery permits it (integrity still comes from the group
    /// header CRC and commit trailer, which both cover the sequence).
    ///
    /// # Errors
    /// Same as [`Self::append_group`], plus `InvalidInput` if `sequence` is
    /// behind this segment's next sequence, or if this is a pool-tagged segment
    /// (which requires the
    /// `append_group_tagged_with_sequence` method).
    pub fn append_group_with_sequence(
        &mut self,
        mutations: &[Mutation],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        let untagged = mutations
            .iter()
            .cloned()
            // A direct single-pool journal has no separate pool argument. It
            // is still encoded in the one supported wire format, using State
            // as the default pool for this low-level helper. Production
            // multi-pool callers use the tagged API below.
            .map(|mutation| (ShardType::State, mutation))
            .collect::<Vec<_>>();
        self.append_group_inner(&untagged, sequence)
    }

    /// Append a pool-tagged group to a shared, pool-tagged segment.
    ///
    /// # Errors
    /// Same as [`Self::append_group_with_sequence`], plus `InvalidInput` if
    /// this segment is not pool-tagged.
    pub fn append_group_tagged_with_sequence(
        &mut self,
        mutations: &[(ShardType, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        self.append_group_inner(mutations, sequence)
    }

    pub(super) fn append_group_for_current_mode(
        &mut self,
        mutations: &[(ShardType, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
        self.append_group_tagged_with_sequence(mutations, sequence)
    }

    /// Shared implementation for a group of mutations, each paired with its
    /// optional pool tag. The tag must match the segment's dialect (see
    /// the journal's single v1 format.
    #[allow(
        clippy::too_many_lines,
        reason = "the group framing, capacity check, LSN assignment and write must \
                  stay in one order-sensitive block"
    )]
    fn append_group_inner(
        &mut self,
        mutations: &[(ShardType, Mutation)],
        sequence: Option<u64>,
    ) -> io::Result<CommitReceipt> {
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
        let sequence = sequence.unwrap_or(self.next_sequence);
        if sequence < self.next_sequence {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal group sequence would move backwards",
            ));
        }

        let mut payload = Vec::new();
        for (index, (pool, mutation)) in mutations.iter().enumerate() {
            let lsn = first_lsn
                .checked_add(u64::try_from(index).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "mutation index exceeds u64")
                })?)
                .ok_or_else(|| io::Error::other("journal LSN exhausted"))?;
            encode_mutation(lsn, *pool, mutation, &mut payload)?;
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
        if self.file_len.saturating_add(group_size) > self.segment_cap {
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

        // One write per group: every autocommit publish lands here, so the
        // group goes out as one syscall. Unix uses `writev` so the large
        // payload is not copied into a second frame buffer; Windows keeps the
        // contiguous buffer because its `write_vectored` is not a true gather.
        // A torn write still leaves an incomplete tail that recovery truncates,
        // exactly as before.
        #[cfg(unix)]
        let write_result = {
            let bufs = [
                io::IoSlice::new(&header),
                io::IoSlice::new(&payload),
                io::IoSlice::new(&trailer),
            ];
            crate::io::write_all_vectored(&mut self.file, &bufs)
        };
        #[cfg(not(unix))]
        let write_result = {
            let mut frame = Vec::with_capacity(
                header
                    .len()
                    .saturating_add(payload.len())
                    .saturating_add(trailer.len()),
            );
            frame.extend_from_slice(&header);
            frame.extend_from_slice(&payload);
            frame.extend_from_slice(&trailer);
            self.file.write_all(&frame)
        };
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(error);
        }
        self.file_len = self.file_len.saturating_add(group_size);
        self.note_appended_group(sequence, first_lsn, last_lsn, mutations);

        self.next_sequence = next_sequence;
        self.next_lsn = next_lsn;

        Ok(CommitReceipt {
            sequence,
            first_lsn,
            last_lsn,
            bytes_written: group_size,
        })
    }

    /// Fsync the bytes [`Self::append_group`] wrote, making the group durable.
    ///
    /// Separate from [`Self::append_group`] so a caller can publish a
    /// visibility boundary after the group is complete but before the fsync.
    /// On failure this handle is poisoned: the appended group's on-disk state
    /// is uncertain, so the caller must reopen and rescan.
    ///
    /// # Errors
    /// Returns `io::Error` if the handle is poisoned or the sync fails.
    pub fn make_durable(&mut self) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        #[cfg(test)]
        let _trace = crate::test_trace::span("wal.make_durable", std::path::Path::new("wal"));
        if let Err(error) = self.file.sync_all() {
            self.poisoned = true;
            return Err(error);
        }
        // The fsync returned, so everything appended so far is durable: record
        // that in the durability mark, after the fact.
        self.write_mark(self.file_len);
        Ok(())
    }

    /// Record, in the slot not holding the newest mark, that `durable_len`
    /// bytes are durable. Call it only after an fsync
    /// that covered them, so the mark can only under-claim.
    ///
    /// Only pool-tagged segments keep a mark. The slot is written with a
    /// positioned write, so it never moves the append cursor. A failed write
    /// is ignored: the previous mark stays in the other slot, or recovery sees
    /// no mark and truncates conservatively.
    pub(super) fn write_mark(&mut self, durable_len: u64) {
        let generation = self.mark_generation.saturating_add(1);
        let bytes = encode_mark_slot(MarkSlot {
            generation,
            durable_len,
        });
        let slot = usize::try_from(generation % 2).unwrap_or(0);
        let offset = u64::try_from(MARK_SLOT_OFFSETS[slot]).unwrap_or(u64::MAX);
        #[cfg(unix)]
        let written = std::os::unix::fs::FileExt::write_all_at(&self.file, &bytes, offset);
        // Elsewhere a positioned write moves the shared cursor, so restore it
        // to the end of the segment, where appends continue. Appends take the
        // journal lock too, so nothing writes in between.
        #[cfg(not(unix))]
        let written = self
            .file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| self.file.write_all(&bytes))
            .and_then(|()| self.file.seek(SeekFrom::Start(self.file_len)).map(drop));
        if written.is_ok() {
            self.mark_generation = generation;
        }
    }

    /// Append a non-empty group and durably sync it.
    ///
    /// Convenience wrapper over [`Self::append_group`] followed by
    /// [`Self::make_durable`] for callers that do not need to publish a
    /// visibility boundary between the two.
    ///
    /// # Errors
    /// Propagates either step's failure. A failure leaves the handle poisoned.
    pub fn commit_group(&mut self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        let receipt = self.append_group(mutations)?;
        self.make_durable()?;
        Ok(receipt)
    }

    /// Path of this journal file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sequence the next group appended without an explicit sequence would use.
    ///
    /// Callers sharing one cross-pool sequence allocator initialize it above
    /// the maximum of these across segments. See
    /// `JournalCoordinator::with_shared_sequence`.
    #[must_use]
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// A no-op reclaim result for a rewrite that must wait until a newer group
    /// is durable and covered by the pre-sync hook.
    pub(super) fn skipped_reclaim(&self) -> Reclaim {
        Reclaim {
            retained_groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
            retained_bytes: self
                .file_len
                .saturating_sub(u64::try_from(FILE_HEADER_LEN).unwrap_or(u64::MAX)),
            ..Default::default()
        }
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
        self.reclaim_through_with_blocker(covered_lsn, None)
    }

    /// [`Self::reclaim_through`] with the pool whose missing coverage stopped
    /// the cut recorded on the returned [`Reclaim`], for diagnostics only.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::reclaim_through`].
    pub fn reclaim_through_with_blocker(
        &mut self,
        covered_lsn: u64,
        blocked_by: Option<ShardType>,
    ) -> io::Result<Reclaim> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        if covered_lsn >= self.next_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot reclaim an LSN that has not been committed",
            ));
        }
        let mut reclaimed = if self.directory_matches_file() {
            self.reclaim_through_directory(covered_lsn)?
        } else {
            self.reclaim_through_scan(covered_lsn)?
        };
        reclaimed.blocked_by = blocked_by;
        Ok(reclaimed)
    }

    /// Record a group just appended, ending at `file_len`, in the directory.
    fn note_appended_group(
        &mut self,
        sequence: u64,
        first_lsn: u64,
        last_lsn: u64,
        mutations: &[(ShardType, Mutation)],
    ) {
        self.groups.push(GroupMark {
            sequence,
            first_lsn,
            last_lsn,
            end_offset: self.file_len,
            pools: mutations
                .iter()
                .fold(0, |mask, (pool, _)| mask | pool_bit(*pool)),
        });
    }

    /// The pools that hold reclaim back: those with a frame in the oldest group
    /// still needed that have not reported coverage through it. Empty when the
    /// directory cannot be trusted.
    pub(super) fn blocking_pools(&self, covered: &HashMap<ShardType, u64>) -> Vec<ShardType> {
        if !self.directory_matches_file() {
            return Vec::new();
        }
        for group in &self.groups {
            let blockers: Vec<ShardType> = ShardType::ALL
                .into_iter()
                .enumerate()
                .filter(|(index, pool)| {
                    group.pools & (1_u8 << index) != 0
                        && covered.get(pool).is_none_or(|lsn| *lsn < group.last_lsn)
                })
                .map(|(_, pool)| pool)
                .collect();
            if !blockers.is_empty() {
                return blockers;
            }
        }
        Vec::new()
    }

    /// What the group directory currently holds.
    pub(super) fn directory_stats(&self) -> GroupDirectoryStats {
        GroupDirectoryStats {
            groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
            allocated_bytes: u64::try_from(
                self.groups
                    .capacity()
                    .saturating_mul(std::mem::size_of::<GroupMark>()),
            )
            .unwrap_or(u64::MAX),
        }
    }

    /// Whether the in-memory group directory describes the file exactly: its
    /// last group ends where the file ends.
    pub(super) fn directory_matches_file(&self) -> bool {
        let directory_end = self
            .groups
            .last()
            .map_or(FILE_HEADER_LEN as u64, |group| group.end_offset);
        directory_end == self.file_len
    }

    /// Absolute file offset where the group following `lsn` begins: the end of
    /// the last directory group at or before `lsn`, or the file header when no
    /// group precedes it. Used as a replay cursor's resume hint so a later page
    /// reads only the appended tail.
    pub(super) fn resume_offset_for(&self, lsn: u64) -> u64 {
        self.groups
            .partition_point(|group| group.last_lsn <= lsn)
            .checked_sub(1)
            .and_then(|index| self.groups.get(index))
            .map_or(FILE_HEADER_LEN as u64, |group| group.end_offset)
    }

    /// Bytes a replay cursor pinned at `lsn` still needs: the retained suffix
    /// beginning after the last group whose `last_lsn` is at or below `lsn`,
    /// through the current end of the segment. The file header is excluded, and
    /// a pin at or beyond the last group retains zero. Falls back to the whole
    /// segment length when the directory cannot be trusted.
    pub(super) fn retained_bytes_from(&self, lsn: u64) -> u64 {
        if !self.directory_matches_file() {
            return self.file_len;
        }
        self.file_len.saturating_sub(self.resume_offset_for(lsn))
    }

    /// The newest LSN through which a shared segment may be reclaimed, given
    /// each pool's reported coverage: the last group of the longest prefix in
    /// which every pool with a frame has covered the group. The directory being
    /// stale is reported apart from nothing being covered, so the caller can
    /// scan the file in that case.
    pub(super) fn shared_reclaim_boundary(
        &self,
        covered: &HashMap<ShardType, u64>,
    ) -> SharedBoundary {
        if !self.directory_matches_file() {
            return SharedBoundary::Untrusted;
        }
        boundary_through(&self.groups, covered)
            .map_or(SharedBoundary::NothingCovered, |(lsn, blocked_by)| {
                SharedBoundary::Through(lsn, blocked_by)
            })
    }

    /// Reclaim using the group directory: pick the cut from it and read only
    /// the retained suffix from disk. The dropped prefix is never read.
    fn reclaim_through_directory(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        let cut = self
            .groups
            .partition_point(|group| group.last_lsn <= covered_lsn);
        let Some(last_dropped) = cut.checked_sub(1).and_then(|index| self.groups.get(index)) else {
            return Ok(Reclaim {
                retained_groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
                ..Default::default()
            });
        };
        let copy_started = std::time::Instant::now();
        let header_len = FILE_HEADER_LEN as u64;
        let cut_offset = last_dropped.end_offset;
        // A rewrite copies the retained suffix. Dropping a sliver while copying
        // a large suffix costs more than it frees, and the pool whose coverage
        // completes the cut reports moments later and drops it with no copy.
        // The emergency zone stays an exception so the segment cannot fill
        // while this waits. Skipping only leaves disk in use; it cannot lose
        // data.
        let dropped_bytes = cut_offset.saturating_sub(header_len);
        let suffix_bytes = self.file_len.saturating_sub(cut_offset);
        if suffix_bytes >= UNPROFITABLE_REWRITE_MIN_COPY
            && suffix_bytes > dropped_bytes.saturating_mul(UNPROFITABLE_REWRITE_RATIO)
            && self.file_len < emergency_len_for_cap(self.segment_cap)
        {
            return Ok(Reclaim {
                retained_groups: u64::try_from(self.groups.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
                retained_bytes: suffix_bytes,
                ..Default::default()
            });
        }
        let retained = self.groups.get(cut..).unwrap_or_default();
        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let reclaimed_history = self.reclaimed_after(
            self.groups
                .get(..cut)
                .unwrap_or_default()
                .iter()
                .map(|group| (group.pools, group.last_lsn)),
        );
        let mut rebuilt = file_header_bytes(new_base_sequence, new_base_lsn, reclaimed_history);
        if let Some(first) = retained.first() {
            // Read the suffix straight into the buffer that becomes the new
            // segment, after its header, so it is allocated once.
            let header_len = rebuilt.len();
            read_range_into(
                &self.path,
                cut_offset,
                self.file_len.saturating_sub(cut_offset),
                &mut rebuilt,
            )?;
            // The suffix is copied byte for byte, so check it the way the full
            // scan would have: it must hold exactly the groups the directory
            // says, each valid. Anything else means the directory is stale, and
            // the scan path decides instead.
            let Ok(checked) = scan_groups_from(
                rebuilt.get(header_len..).unwrap_or_default(),
                cut_offset,
                first.sequence,
                first.first_lsn,
                new_base_lsn,
                first.sequence,
                || u64::MAX,
            ) else {
                return self.reclaim_through_scan(covered_lsn);
            };
            if checked.groups.len() != retained.len()
                || checked.valid_len != self.file_len
                || checked.truncated_tail
            {
                return self.reclaim_through_scan(covered_lsn);
            }
        }
        let reclaimed_bytes = self
            .file_len
            .saturating_sub(u64::try_from(rebuilt.len()).unwrap_or(u64::MAX));
        let retained_groups = u64::try_from(retained.len()).unwrap_or(u64::MAX);
        let shift = cut_offset.saturating_sub(header_len);
        let survivors = retained
            .iter()
            .map(|group| GroupMark {
                end_offset: group.end_offset.saturating_sub(shift),
                ..*group
            })
            .collect::<Vec<_>>();
        let copy = copy_started.elapsed();
        let rebuilt_len = rebuilt.len();
        let fsync_started = std::time::Instant::now();
        self.install_rebuilt(rebuilt, new_base_sequence, new_base_lsn)?;
        let fsync = fsync_started.elapsed();
        self.groups = survivors;
        self.reclaimed_history = reclaimed_history;
        Ok(Reclaim {
            retained_groups,
            reclaimed_bytes,
            retained_bytes: u64::try_from(rebuilt_len.saturating_sub(FILE_HEADER_LEN))
                .unwrap_or(u64::MAX),
            copy,
            fsync,
            ..Default::default()
        })
    }

    /// The reclaimed-history marks after a rewrite that drops `dropped` (each
    /// group's pool bitmask and last LSN): the current marks carried forward and
    /// raised to the newest dropped group of each pool.
    fn reclaimed_after(&self, dropped: impl Iterator<Item = (u8, u64)>) -> ReclaimedHistory {
        let mut marks = self.reclaimed_history;
        for (pools, last_lsn) in dropped {
            for (index, mark) in marks.iter_mut().enumerate() {
                if pools & (1_u8 << index) != 0 {
                    *mark = (*mark).max(last_lsn);
                }
            }
        }
        marks
    }

    /// Reclaim by reading and decoding the whole segment. Used when the group
    /// directory cannot be trusted, and it rebuilds the directory from what it
    /// finds.
    pub(super) fn reclaim_through_scan(&mut self, covered_lsn: u64) -> io::Result<Reclaim> {
        let scan_started = std::time::Instant::now();
        let bytes = fs::read(&self.path)?;
        let (base_sequence, base_lsn) = validate_file_header(&bytes)?;
        // Reclaim rewrites the segment from its own live file, so any invalid
        // group there is corruption, not a crash tail: treat every byte as
        // durable and let it fail.
        let scan = scan_bytes(&bytes, base_sequence, base_lsn, || u64::MAX)?;
        let boundary = scan_started.elapsed();
        let retained = scan
            .groups
            .iter()
            .filter(|group| group.last_lsn > covered_lsn)
            .collect::<Vec<_>>();
        if retained.len() == scan.groups.len() {
            self.groups = marks_from_scan(&scan);
            return Ok(Reclaim {
                retained_groups: u64::try_from(retained.len()).unwrap_or(u64::MAX),
                reclaimed_bytes: 0,
                boundary,
                ..Default::default()
            });
        }

        let copy_started = std::time::Instant::now();
        let (new_base_sequence, new_base_lsn) = retained
            .first()
            .map_or((self.next_sequence, self.next_lsn), |group| {
                (group.sequence, group.first_lsn)
            });
        let reclaimed_history = self.reclaimed_after(
            scan.groups
                .iter()
                .filter(|group| group.last_lsn <= covered_lsn)
                .map(|group| {
                    (
                        group
                            .entries
                            .iter()
                            .fold(0, |mask, entry| mask | pool_bit(entry.pool)),
                        group.last_lsn,
                    )
                }),
        );
        let mut rebuilt = file_header_bytes(new_base_sequence, new_base_lsn, reclaimed_history);
        let mut survivors = Vec::with_capacity(retained.len());
        for group in &retained {
            encode_group(group, &mut rebuilt)?;
            survivors.push(GroupMark {
                sequence: group.sequence,
                first_lsn: group.first_lsn,
                last_lsn: group.last_lsn,
                end_offset: u64::try_from(rebuilt.len()).unwrap_or(u64::MAX),
                pools: group
                    .entries
                    .iter()
                    .fold(0, |mask, entry| mask | pool_bit(entry.pool)),
            });
        }
        let copy = copy_started.elapsed();
        let retained_groups = u64::try_from(retained.len()).unwrap_or(u64::MAX);
        let rebuilt_len = rebuilt.len();
        let reclaimed_bytes =
            u64::try_from(bytes.len().saturating_sub(rebuilt.len())).unwrap_or(u64::MAX);
        let fsync_started = std::time::Instant::now();
        self.install_rebuilt(rebuilt, new_base_sequence, new_base_lsn)?;
        let fsync = fsync_started.elapsed();
        self.groups = survivors;
        self.reclaimed_history = reclaimed_history;
        Ok(Reclaim {
            retained_groups,
            reclaimed_bytes,
            retained_bytes: u64::try_from(rebuilt_len.saturating_sub(FILE_HEADER_LEN))
                .unwrap_or(u64::MAX),
            boundary,
            copy,
            fsync,
            ..Default::default()
        })
    }

    /// Durably replace the segment with `rebuilt` (a new header plus the
    /// retained groups) and point this handle at it.
    fn install_rebuilt(
        &mut self,
        mut rebuilt: Vec<u8>,
        new_base_sequence: u64,
        new_base_lsn: u64,
    ) -> io::Result<()> {
        #[cfg(test)]
        let _rewriting = RewritePhaseGuard::enter(&self.rewrite_phase);
        // The rebuilt file is fsynced before it replaces the segment, so all of
        // it is durable. Put its mark inside it, first generation, so the mark
        // is durable with the data and the old segment's marks, which described
        // offsets that no longer exist, are simply gone.
        let rebuilt_mark = Some(MarkSlot {
            generation: 1,
            durable_len: u64::try_from(rebuilt.len()).unwrap_or(u64::MAX),
        });
        if let Some(mark) = rebuilt_mark {
            let offset = MARK_SLOT_OFFSETS[1];
            let end = offset.saturating_add(MARK_SLOT_LEN);
            rebuilt[offset..end].copy_from_slice(&encode_mark_slot(mark));
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
            fs::rename(&temp_path, &self.path)
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }
        // The rename has replaced the path, so from here onward any failure
        // must poison the handle: continuing to append through the old file
        // descriptor would write an unlinked inode and could acknowledge data
        // that is absent from the journal path.
        let replacement = match OpenOptions::new().read(true).write(true).open(&self.path) {
            Ok(file) => file,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        self.file = replacement;
        self.file_len = u64::try_from(rebuilt.len()).unwrap_or(u64::MAX);
        self.base_sequence = new_base_sequence;
        self.base_lsn = new_base_lsn;
        self.mark_generation = u64::from(rebuilt_mark.is_some());
        if let Err(error) = sync_parent_dir(&self.path) {
            self.poisoned = true;
            return Err(error);
        }
        if let Err(error) = self.file.seek(SeekFrom::End(0)) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }
}
