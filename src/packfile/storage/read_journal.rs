//! Cross-process read-committed overlay, gated behind the `multi-reader`
//! feature (see `Cargo.toml` and `docs/TODO.txt`'s extraction plan).
//!
//! Lets a separate OS process observe a live writer's committed-but-not-yet-
//! checkpointed data by scanning the writer's journal segment read-only. A
//! single-process embedded deployment never calls any of this.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::journal::{Journal, Mutation as JournalMutation};
use crate::storage::{NodeData, NodeId, StorageError};

use super::{OpenTimings, PackfileStorage, ReloadMode};

/// How many bytes ending at the last consumed group the overlay remembers to
/// notice that the consumed prefix was rewritten.
const TAIL_WINDOW: u64 = crate::journal::CONSUMED_TAIL_LEN as u64;

/// A file's identity and change stamp: device, inode, length and change time.
/// Equal stamps on a quiet file mean nothing wrote to it in between, so the
/// consumed bytes need no re-read. The change time (ctime) is used because,
/// unlike the modification time, no caller can set it backwards.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    len: u64,
    ctime_secs: i64,
    ctime_nanos: i64,
}

impl FileStamp {
    /// Whether two stamps name the same file, ignoring length and time.
    fn same_file(self, other: Self) -> bool {
        self.dev == other.dev && self.ino == other.ino
    }

    /// Whether the last change is older than [`QUIET_AFTER`], so any later
    /// write is guaranteed a different change time whatever the filesystem's
    /// timestamp granularity. A change inside that window could share a tick
    /// with a later write, so the stamp is not trusted yet.
    ///
    /// This reads the wall clock. A backwards step makes the age negative and
    /// the stamp counts as not quiet, which is the safe direction. A forward
    /// step could make a just-changed file look quiet; that would also need a
    /// same-tick rewrite to go unnoticed, so it is accepted rather than guarded.
    fn is_quiet(self) -> bool {
        let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) else {
            return false;
        };
        let Ok(secs) = u64::try_from(self.ctime_secs) else {
            return false;
        };
        let changed = std::time::Duration::new(secs, u32::try_from(self.ctime_nanos).unwrap_or(0));
        now.checked_sub(changed)
            .is_some_and(|age| age > QUIET_AFTER)
    }
}

/// How long a file must have been unchanged before an equal [`FileStamp`] is
/// trusted to mean it was not rewritten in between.
const QUIET_AFTER: std::time::Duration = std::time::Duration::from_millis(100);

/// The stamp of `meta`, when the platform exposes a reliable file identity.
/// Inode 0 is not a real identity (some filesystems report it for every file),
/// so it yields no stamp.
#[cfg(unix)]
fn file_stamp(meta: &fs::Metadata) -> Option<FileStamp> {
    use std::os::unix::fs::MetadataExt;
    if meta.ino() == 0 {
        return None;
    }
    Some(FileStamp {
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
        ctime_secs: meta.ctime(),
        ctime_nanos: meta.ctime_nsec(),
    })
}

/// Without inode numbers and change times, the overlay compares the tail bytes
/// on every refresh instead of trusting a synthetic identity.
#[cfg(not(unix))]
fn file_stamp(_meta: &fs::Metadata) -> Option<FileStamp> {
    None
}

/// One committed value in the read-journal overlay: payload and its LSN.
type ReadJournalValue = (bytes::Bytes, u64);
/// Values for one collection, keyed by node ID.
type ReadJournalCollection = HashMap<[u8; 16], ReadJournalValue>;
/// Committed puts keyed `collection_id -> node_id`.
type ReadJournalPuts = HashMap<[u8; 16], ReadJournalCollection>;

/// Outcome of refreshing the overlay against the writer's segment.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadRefresh {
    /// The segment was scanned and applied; the overlay is current.
    Applied,
    /// The segment was reclaimed past the coverage this reader's index
    /// incorporated. The checkpoint-bound index must be reloaded before the
    /// overlay can be trusted again.
    NeedsReload,
}

/// In-memory overlay of committed journal groups, built by scanning a
/// writer's journal segment read-only. It is the read-committed source of
/// truth for [`PackfileStorage::get_read_committed`]: entries at or below
/// `observed_lsn` have a complete commit trailer, so they are committed even
/// though the writer may not have fsynced or advanced the durable index yet.
pub(super) struct ReadJournal {
    /// Pool whose tagged frames this overlay applies, or `None` for a per-pool
    /// (untagged, version-2) segment. A shared segment interleaves all pools'
    /// frames, so a pool's reader must apply only frames tagged for it.
    pool: Option<crate::layout::ShardType>,
    /// Segment file scanned for committed groups.
    path: PathBuf,
    /// The journal LSN covered by the durable index this reader actually
    /// loaded, fixed when the overlay was enabled.
    ///
    /// This is the *only* coverage entries may be filtered or pruned against.
    /// Reading a fresher `journal.lsn` from disk would prune a committed entry
    /// the reader's stale in-memory index does not yet contain, dropping the
    /// record from both places. It is deliberately not advanced by a live
    /// packfile rescan: only loading the corresponding checkpoint may advance
    /// the index/coverage pair.
    pub(super) covered: u64,
    /// Raw file length at the last scan. Skips a rescan when unchanged, so an
    /// unchanged partial tail is not re-probed on every read.
    observed_len: u64,
    /// Absolute offset after the last complete group consumed. A partial tail
    /// is re-probed from here once more bytes land.
    observed_valid_len: u64,
    /// Highest applied LSN. Groups at or below this are already in the overlay.
    observed_lsn: u64,
    /// The last bytes consumed (up to [`TAIL_WINDOW`], ending at
    /// `observed_valid_len`), or `None` before any group was applied.
    ///
    /// File length cannot show that the bytes already consumed are still the
    /// ones on disk: after a writer restart that discarded a visible-but-
    /// undurable group and reissued its LSN, or after a reclaim followed by
    /// enough appends, the file can be the same length (or longer) while its
    /// content differs. Comparing this window against the bytes now ending at
    /// `observed_valid_len` catches that. It covers the most recently consumed
    /// records only: a rewrite that leaves the whole window byte-identical is
    /// not detected. The group trailer is not used because it is not a content
    /// fingerprint (see [`Journal::read_tail_ending_at`]).
    observed_tail: Option<Vec<u8>>,
    /// Descriptor the tail window is read through, opened when the window was
    /// recorded, so a refresh pays one positioned read instead of an open.
    tail_file: Option<File>,
    /// Stamp of the file when the tail window was recorded. The descriptor
    /// keeps pointing at an old inode after the writer renames a replacement
    /// into place (reclaim), so a path whose device or inode differs means the
    /// segment was replaced and the overlay must be rebuilt. An equal stamp on
    /// a file that was already quiet when recorded (`tail_quiet`) proves the
    /// consumed bytes were not rewritten, so the window is not re-read.
    tail_stamp: Option<FileStamp>,
    /// Whether the file was quiet, per [`FileStamp::is_quiet`], when
    /// `tail_stamp` was taken.
    tail_quiet: bool,
    /// Test-only count of window reads, to show the fingerprint skips them.
    #[cfg(test)]
    pub(super) tail_reads: u64,
    /// Test-only switch that makes every stamp absent, as on a platform without
    /// inode numbers, to exercise the no-identity path on unix.
    #[cfg(test)]
    pub(super) force_no_identity: bool,
    /// Test-only count of descriptors opened to record a tail. It does not
    /// count the fresh open a no-identity comparison makes (see
    /// `observed_tail_changed`), so `0` means "no descriptor is kept", not
    /// "the file is never opened".
    #[cfg(test)]
    pub(super) tail_opens: u64,
    /// Test-only count of overlay rebuilds forced by a changed tail.
    #[cfg(test)]
    pub(super) tail_resets: u64,
    /// Scratch buffer the window is read into, reused across refreshes.
    tail_scratch: Vec<u8>,
    /// Committed puts as `(payload, lsn)`, keyed `collection_id -> node_id`.
    pub(super) puts: ReadJournalPuts,
    /// Highest committed delete LSN per collection. This is a persistent
    /// boundary: durable fallback stays suppressed for the collection until the
    /// checkpoint covers the delete, because the durable index may still hold
    /// pre-delete records. A later put does not clear it.
    pub(super) delete_lsn: HashMap<[u8; 16], u64>,
}

impl ReadJournal {
    pub(super) fn empty(
        path: PathBuf,
        covered: u64,
        pool: Option<crate::layout::ShardType>,
    ) -> Self {
        Self {
            pool,
            path,
            covered,
            observed_len: 0,
            observed_valid_len: 0,
            observed_lsn: 0,
            observed_tail: None,
            tail_file: None,
            tail_stamp: None,
            tail_quiet: false,
            #[cfg(test)]
            tail_reads: 0,
            #[cfg(test)]
            force_no_identity: false,
            #[cfg(test)]
            tail_opens: 0,
            #[cfg(test)]
            tail_resets: 0,
            tail_scratch: Vec::new(),
            puts: HashMap::new(),
            delete_lsn: HashMap::new(),
        }
    }

    /// Drop all applied state so the next refresh rebuilds from the segment.
    pub(super) fn reset_overlay(&mut self) {
        self.puts.clear();
        self.delete_lsn.clear();
        self.observed_lsn = 0;
        self.observed_valid_len = 0;
        self.observed_tail = None;
        self.tail_file = None;
        self.tail_stamp = None;
        self.tail_quiet = false;
        // Zero the length too, so an equal-length segment after a reload still
        // forces a full rescan instead of short-circuiting on `len == observed_len`.
        self.observed_len = 0;
    }

    /// Discard overlay state the durable index this reader loaded now covers.
    ///
    /// Reclaim is best-effort, so a journal group can outlive the checkpoint
    /// that covers it. Without this, an old journal put or delete could shadow
    /// newer checkpointed state. The bound is [`Self::covered`] — the coverage
    /// of the loaded index, not a detached disk read — so an entry is only
    /// dropped once the fallback index is known to contain it.
    fn prune_covered(&mut self, covered: u64) {
        if covered == 0 {
            return;
        }
        self.puts.retain(|_, entries| {
            entries.retain(|_, (_, lsn)| *lsn > covered);
            !entries.is_empty()
        });
        self.delete_lsn.retain(|_, lsn| *lsn > covered);
    }

    /// Whether the segment's base has advanced past the coverage this reader's
    /// index incorporated, so the prefix may hold frames this index lacks.
    ///
    /// The reclaimed prefix is already gone from disk, so its contents cannot
    /// be inferred from the surviving groups. A base jump is therefore treated
    /// as a possible gap unconditionally; whether it is a *real* gap for this
    /// pool is decided by the caller, which reloads the checkpoint-bound index
    /// and only accepts the jump once that index is known to be current (see
    /// [`PackfileStorage::refresh_read_journal`]). A `base_lsn` at or below
    /// `covered + 1` means nothing at or below `covered` was dropped.
    pub(super) fn reset_has_coverage_gap(&self, scan: &crate::journal::Scan) -> bool {
        scan.base_lsn > self.covered.saturating_add(1)
    }

    /// Whether the bytes that ended the consumed prefix are no longer the ones
    /// ending at `observed_valid_len` on disk. `stamp` is the segment path's
    /// stamp right now. A replaced file, or missing or unreadable bytes, count
    /// as changed: the consumed prefix is gone.
    fn observed_tail_changed(&mut self, stamp: Option<FileStamp>) -> bool {
        let Some(expected) = &self.observed_tail else {
            // Groups were applied but their tail could not be recorded (or the
            // recording was discarded), so nothing protects the overlay from a
            // rewritten prefix: rebuild it and try recording again.
            return self.observed_lsn > 0;
        };
        let mut unchanged_stamp = None;
        // With a real identity the descriptor held since recording still names
        // the file the overlay was built from. Without one (a platform that
        // reports no inode numbers) a replaced file cannot be told apart by
        // inode, so the path is opened afresh and what is there now is compared.
        let fresh;
        let file = match (stamp, self.tail_stamp) {
            (Some(now), Some(then)) => {
                if !now.same_file(then) {
                    return true;
                }
                // Nothing wrote to a file that was already quiet when the
                // window was recorded, so the consumed bytes are unchanged.
                if self.tail_quiet && now == then {
                    return false;
                }
                if now == then {
                    unchanged_stamp = Some(now);
                }
                let Some(held) = &self.tail_file else {
                    return true;
                };
                held
            }
            (None, None) => {
                let Ok(opened) = File::open(&self.path) else {
                    return true;
                };
                fresh = opened;
                &fresh
            }
            // An identity that appeared or vanished since recording means the
            // file is not the one the overlay was built from.
            _ => return true,
        };
        #[cfg(test)]
        {
            self.tail_reads = self.tail_reads.saturating_add(1);
        }
        match Journal::read_tail_ending_at(
            file,
            self.observed_valid_len,
            TAIL_WINDOW,
            &mut self.tail_scratch,
        ) {
            Ok(true) if &self.tail_scratch != expected => true,
            Ok(true) => {
                // The window matches and the stamp is unchanged. If the file
                // has by now been quiet longer than the timestamp granularity,
                // any later write is guaranteed a different change time, so
                // later refreshes can trust the stamp and skip this read. This
                // is what lets a file that was still warm when recorded reach
                // the fast path.
                if unchanged_stamp.is_some_and(FileStamp::is_quiet) {
                    self.tail_quiet = true;
                }
                false
            }
            Ok(false) | Err(_) => true,
        }
    }

    /// Remember the bytes that end the consumed prefix, and keep a descriptor
    /// for later refreshes to compare them through.
    ///
    /// The window is built from the bytes the scan actually consumed
    /// (`consumed`), appended to the window already remembered, never from a
    /// separate read of the file: a rewrite that lands after the scan can then
    /// never be folded into what is remembered, and a later comparison is
    /// always against what the overlay was built from. `scanned` is the
    /// segment path's stamp taken before the scan; a write after that stat
    /// changes the stamp, so the next refresh compares the window instead of
    /// trusting it.
    ///
    /// While the segment is still the same file the held descriptor is reused,
    /// so a refresh that only appended opens nothing. A new descriptor is
    /// opened only after a reset or a replaced file, and if it refers to a
    /// different file than `scanned` the segment was replaced mid-refresh and
    /// the recording is discarded. On any failure nothing is remembered, which
    /// [`Self::observed_tail_changed`] treats as untrusted, so the next refresh
    /// rescans from the start instead of trusting the prefix.
    fn record_observed_tail(&mut self, scanned: Option<FileStamp>, consumed: &[u8]) {
        let mut window = self.observed_tail.take().unwrap_or_default();
        // A stamp exists only where the platform reports a reliable file
        // identity, so a `Some` pair means the held descriptor can be trusted
        // to name the same file. With no identity there is no descriptor to
        // reuse and none is ever held.
        let reusable = match (scanned, self.tail_stamp) {
            (Some(now), Some(then)) => now.same_file(then),
            _ => false,
        };
        let held = if reusable {
            self.tail_file.take()
        } else {
            None
        };
        self.tail_file = None;
        self.tail_stamp = None;
        self.tail_quiet = false;
        if self.observed_valid_len == 0 {
            return;
        }
        window.extend_from_slice(consumed);
        if window.len() > crate::journal::CONSUMED_TAIL_LEN {
            let excess = window
                .len()
                .saturating_sub(crate::journal::CONSUMED_TAIL_LEN);
            window.drain(..excess);
        }
        // The remembered bytes must be exactly the window a later comparison
        // reads. If they are not (nothing consumed to build them from), leave
        // the overlay untrusted so it rebuilds.
        let expected_len = crate::journal::consumed_tail_len(self.observed_valid_len);
        // A segment with no group bytes yet (header only) has no window to
        // protect and none to compare, so nothing is recorded.
        if expected_len == 0 || window.len() != expected_len {
            return;
        }
        // Without an identity no descriptor is kept: later comparisons open the
        // path afresh (see `observed_tail_changed`).
        let file = if scanned.is_none() {
            None
        } else if let Some(file) = held {
            Some(file)
        } else {
            #[cfg(test)]
            {
                self.tail_opens = self.tail_opens.saturating_add(1);
            }
            let Ok(file) = File::open(&self.path) else {
                return;
            };
            let opened = file.metadata().ok().and_then(|meta| file_stamp(&meta));
            if let (Some(scanned), Some(opened)) = (scanned, opened) {
                if !scanned.same_file(opened) {
                    return;
                }
            }
            Some(file)
        };
        self.observed_tail = Some(window);
        self.tail_file = file;
        self.tail_quiet = scanned.is_some_and(FileStamp::is_quiet);
        self.tail_stamp = scanned;
    }

    /// Apply every committed group above `observed_lsn` to the overlay.
    ///
    /// A missing segment is treated as empty only before this reader has seen
    /// any journal bytes; if an observed segment disappears, the caller must
    /// reload the checkpoint. A segment that shrank since the last scan was
    /// reclaimed, so the overlay is rebuilt from scratch. Never repairs or
    /// creates the file, matching a read-only worker's constraints.
    ///
    /// `accept_reclaimed_prefix` is set by the caller once it has reloaded the
    /// checkpoint-bound index to the writer's latest durable coverage for this
    /// pool. On a shared segment a base jump beyond that coverage can only be
    /// other pools' reclaimed frames, which this pool does not need, so the
    /// jump is accepted instead of forcing another reload.
    ///
    /// Returns [`ReadRefresh::NeedsReload`] when a reset reveals the writer
    /// reclaimed past this reader's incorporated coverage; the caller must
    /// reload the checkpoint-bound index and rebind `covered` before retrying.
    fn refresh(&mut self, accept_reclaimed_prefix: bool) -> Result<ReadRefresh, StorageError> {
        // Coverage is fixed to the index this reader loaded. Reading
        // `journal.lsn` fresh would prune entries the reader's stale index has
        // not incorporated yet, dropping records from both sources.
        let covered = self.covered;
        let (len, stamp) = match fs::metadata(&self.path) {
            Ok(meta) => (meta.len(), file_stamp(&meta)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if self.observed_len != 0 || self.observed_lsn > covered {
                    return Ok(ReadRefresh::NeedsReload);
                }
                (0, None)
            }
            Err(error) => return Err(StorageError::Io(error)),
        };
        #[cfg(test)]
        let stamp = if self.force_no_identity { None } else { stamp };
        // Length alone cannot show that the consumed prefix is unchanged, so
        // check the bytes that ended it first. On a change, drop everything and
        // rebuild: a full rescan is gap-checked like any other reset below.
        if self.observed_tail_changed(stamp) {
            #[cfg(test)]
            {
                self.tail_resets = self.tail_resets.saturating_add(1);
            }
            self.reset_overlay();
        }
        if len != self.observed_len {
            let mut reset = false;
            if len < self.observed_len {
                // The segment was reclaimed/rotated; re-derive from the start.
                self.reset_overlay();
                reset = true;
            }
            let full_scan = self.observed_valid_len == 0;
            let scan = if full_scan {
                Journal::scan_read_only(&self.path).map_err(StorageError::Io)?
            } else {
                // Read only the tail past the last complete group. A failure
                // here means the segment was replaced under us (or a rotation
                // changed its base), so fall back to a full rescan.
                if let Ok(scan) = Journal::scan_read_only_from(
                    &self.path,
                    self.observed_valid_len,
                    self.observed_lsn.saturating_add(1),
                ) {
                    scan
                } else {
                    self.reset_overlay();
                    reset = true;
                    Journal::scan_read_only(&self.path).map_err(StorageError::Io)?
                }
            };
            // A full rescan has no continuity with the previous overlay, so it
            // must be gap-checked too — not only an explicit shrink. Without
            // this, a reload that rebinds `covered` below the segment base
            // would silently serve a hole.
            if (reset || full_scan)
                && !accept_reclaimed_prefix
                && self.reset_has_coverage_gap(&scan)
            {
                return Ok(ReadRefresh::NeedsReload);
            }
            for group in &scan.groups {
                for entry in &group.entries {
                    if entry.lsn <= self.observed_lsn || entry.lsn <= covered {
                        continue;
                    }
                    // A shared segment interleaves all pools' frames; apply
                    // only this pool's. `None` pool is a per-pool segment,
                    // whose frames are all this store's.
                    if self.pool.is_some_and(|pool| entry.pool != Some(pool)) {
                        continue;
                    }
                    match &entry.mutation {
                        JournalMutation::Put {
                            collection_id,
                            node_id,
                            payload,
                        } => {
                            self.puts.entry(*collection_id).or_default().insert(
                                *node_id,
                                (bytes::Bytes::copy_from_slice(payload), entry.lsn),
                            );
                        }
                        JournalMutation::DeleteCollection { collection_id } => {
                            self.puts.remove(collection_id);
                            self.delete_lsn
                                .entry(*collection_id)
                                .and_modify(|lsn| *lsn = (*lsn).max(entry.lsn))
                                .or_insert(entry.lsn);
                        }
                    }
                }
                self.observed_lsn = self.observed_lsn.max(group.last_lsn);
            }
            // Resume from the last complete group, not the raw file length, so
            // a partial tail is re-probed once its trailer lands.
            self.observed_valid_len = scan.valid_len;
            self.observed_len = len;
            self.record_observed_tail(stamp, &scan.consumed_tail);
        }
        // The checkpoint can advance without the segment changing (reclaim is
        // best-effort), so prune overlay state it now covers.
        self.prune_covered(covered);
        Ok(ReadRefresh::Applied)
    }
}

impl PackfileStorage {
    /// Reload the durable index from the current on-disk checkpoint and rebind
    /// [`Self::read_covered_lsn`] to it.
    ///
    /// Returns `false` when no checkpoint matching the current packs is on
    /// disk, so the caller can fail closed instead of serving a stale index.
    /// Coverage is captured before the checkpoint is read, so the new bound can
    /// never be newer than the index loaded below.
    pub(super) fn reload_index_from_checkpoint(&self) -> bool {
        // A writer may have created new packs since this handle opened, and the
        // checkpoint it wrote names that new pack set. Rediscover before
        // building the fingerprint, or the reload can never match.
        if self.shards.discover_shards().is_err() {
            self.read_reload_failures.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // Rebuild the fingerprint from files that still exist. This worker's
        // shard table can retain open handles to packs the writer retired and
        // unlinked during repack; including their cached lengths makes every
        // checkpoint fingerprint mismatch forever. New/grown live packs are
        // included at their current lengths, and the checkpoint delta log
        // validates the suffix from the checkpoint's original fingerprint.
        let mut open_shards: Vec<(u16, u64, PathBuf, u64)> = Vec::new();
        for (id, shard) in self.shards.all_shards() {
            match fs::metadata(&shard.path) {
                Ok(metadata) => {
                    open_shards.push((id, shard.pack_id, shard.path.clone(), metadata.len()));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {
                    self.read_reload_failures.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        }
        let deleted_collections = self.deleted_collections.lock().clone();
        let mut timings = OpenTimings::default();
        let Some((scan_out, collection_order, _delta_state, checkpoint_covered)) =
            Self::checkpoint_scan_out(
                &self.base_dir,
                self.cache_capacity,
                &self.shards,
                &open_shards,
                &deleted_collections,
                false,
                ReloadMode::JournalBound,
                self.index_config,
                &mut timings,
            )
        else {
            self.read_reload_failures.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        {
            let mut tables = self.index_tables.write();
            tables.collections = scan_out.collections;
            tables.collection_order = collection_order;
            tables.shard_collections = scan_out.shard_collections;
            tables.collection_shards = scan_out.collection_shards;
        }
        // Bind coverage to the checkpoint actually loaded, not to a
        // separately-read `journal.lsn` that a concurrent checkpoint may
        // already have advanced past this index.
        self.read_covered_lsn
            .store(checkpoint_covered, Ordering::Release);
        self.read_reloads.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Refresh the read-committed overlay, reloading the checkpoint-bound index
    /// when the writer reclaimed past this reader's incorporated coverage.
    ///
    /// A reclaim that outruns the reader's index is the one case the overlay
    /// cannot resolve from the segment alone; reloading the checkpoint advances
    /// the index/coverage pair together, after which the retry is consistent.
    /// Without a usable checkpoint the read fails closed.
    ///
    /// On success the held overlay guard is returned so the caller can read the
    /// applied overlay without releasing and reacquiring the mutex, which could
    /// otherwise expose an empty or partially rebuilt overlay to another reader.
    pub(super) fn refresh_read_journal(
        &self,
    ) -> Result<parking_lot::MutexGuard<'_, Option<ReadJournal>>, StorageError> {
        const RELOAD_ATTEMPTS: usize = 8;
        const MAX_BACKOFF_MS: u64 = 64;
        // Capture coverage once, before the first attempt. Comparing the final
        // value against a per-iteration snapshot would only detect coverage that
        // advanced on the *last* attempt, misreporting a moving checkpoint as a
        // genuine gap. Reloads only ever advance `read_covered_lsn`, so a strict
        // increase across the whole loop is exactly the retryable signal.
        let coverage_before_attempts = self.read_covered_lsn.load(Ordering::Acquire);
        // A failed reload attempt makes the outcome retryable even if coverage
        // never advanced: a concurrent writer may simply not have written the
        // checkpoint yet, so the same gap can resolve on a later read.
        // Deliberately sticky: any failed reload makes the whole refresh
        // retryable, even if a later attempt's reload succeeds.
        let mut reload_failed = false;
        for attempt in 0..RELOAD_ATTEMPTS {
            let mut guard = self.read_journal.lock();
            let Some(overlay) = guard.as_mut() else {
                return Ok(guard);
            };
            if overlay.refresh(false)? == ReadRefresh::Applied {
                return Ok(guard);
            }
            if !self.reload_index_from_checkpoint() {
                reload_failed = true;
                drop(guard);
                let delay_ms = 1_u64
                    .checked_shl(u32::try_from(attempt).unwrap_or(u32::MAX))
                    .unwrap_or(MAX_BACKOFF_MS)
                    .min(MAX_BACKOFF_MS);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                continue;
            }
            overlay.covered = self.read_covered_lsn.load(Ordering::Acquire);
            overlay.reset_overlay();
            // The reload bound this overlay to the newest checkpoint it could
            // load. On a shared segment a base jump beyond that checkpoint is
            // made up of other pools' reclaimed frames, which this pool never
            // needs, so accept it once the loaded checkpoint is at least this
            // pool's latest durable coverage. A per-pool segment holds only
            // this store's frames, so it stays fail-closed.
            let accept_reclaimed_prefix =
                overlay.pool.is_some() && overlay.covered >= Self::read_journal_lsn(&self.base_dir);
            // Rebuild immediately while still holding the guard: returning the
            // guard only once the overlay is applied keeps another reader from
            // observing the just-cleared state.
            if overlay.refresh(accept_reclaimed_prefix)? == ReadRefresh::Applied {
                return Ok(guard);
            }
            drop(guard);
            if attempt.saturating_add(1) < RELOAD_ATTEMPTS {
                let delay_ms = 1_u64
                    .checked_shl(u32::try_from(attempt).unwrap_or(u32::MAX))
                    .unwrap_or(MAX_BACKOFF_MS)
                    .min(MAX_BACKOFF_MS);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
        let advanced = self.read_covered_lsn.load(Ordering::Acquire) > coverage_before_attempts;
        if reload_failed || advanced {
            Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "read-committed reload failed or checkpoint coverage advanced; retry the read",
            )))
        } else {
            Err(StorageError::Corrupt(
                "read-committed journal segment skips LSNs not covered by the loaded checkpoint"
                    .to_owned(),
            ))
        }
    }

    /// Enable a read-only journal overlay for the read-committed API.
    ///
    /// `path` is a writer's journal segment. The overlay is built with
    /// [`Journal::scan_read_only`], which never creates, repairs, or locks the
    /// segment, so this is safe from a read-only worker process. The durable
    /// read API ([`crate::storage::StorageEngine::get_many`], [`Self::get_many_with_refresh`])
    /// is unchanged and continues to hide unflushed writes; only
    /// [`Self::get_read_committed`] consults the overlay.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the segment is unreadable, a committed
    /// group fails validation, or a coverage gap cannot be resolved by
    /// reloading the checkpoint.
    pub fn enable_read_journal(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), StorageError> {
        self.enable_read_journal_inner(path, None)
    }

    /// Like [`Self::enable_read_journal`], but for a shared, pool-tagged
    /// segment: the overlay applies only frames tagged with `pool`, so a
    /// worker reading one pool of a shared WAL does not observe the other
    /// pools' mutations.
    ///
    /// # Errors
    /// Same as [`Self::enable_read_journal`].
    pub fn enable_read_journal_shared(
        &self,
        path: impl AsRef<std::path::Path>,
        pool: crate::layout::ShardType,
    ) -> Result<(), StorageError> {
        self.enable_read_journal_inner(path, Some(pool))
    }

    /// Enable the in-process overlay a writer's own pool uses while a published
    /// transaction is being materialized.
    ///
    /// This is not the reader's setup. A read-only worker has an index loaded
    /// from a checkpoint, so a reclaimed journal prefix past that checkpoint is
    /// a gap it must reload to fill. A writer's live index already holds every
    /// mutation it has applied, so a reclaimed prefix can never hide data from
    /// it, and there is nothing to reload: its `read_covered_lsn` is never set
    /// and a reload from the checkpoint cannot succeed against its own open
    /// packs. Running the reader's gap check here made every commit fail with a
    /// retry-forever `WouldBlock` once a checkpoint had reclaimed the shared
    /// WAL past LSN 1.
    ///
    /// The overlay is bound to the pool's checkpoint coverage (`journal.lsn`):
    /// frames at or below it are in the durable index, and frames above it are
    /// served from the journal until the transaction is fully materialized.
    /// It is built directly and refreshed once with `accept_reclaimed_prefix`
    /// set, rather than going through `enable_read_journal_inner`, whose
    /// refresh-then-reload path is the reader's and cannot succeed here.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the segment is unreadable or a committed
    /// group fails validation.
    #[cfg(feature = "multi-reader")]
    pub(super) fn enable_transaction_read_journal(
        &self,
        path: &std::path::Path,
        pool: crate::layout::ShardType,
    ) -> Result<(), StorageError> {
        let covered = Self::read_journal_lsn(&self.base_dir);
        let mut overlay = ReadJournal::empty(path.to_path_buf(), covered, Some(pool));
        // The writer's index already holds everything it applied, so accept a
        // reclaimed prefix instead of treating it as a gap. A fresh overlay has
        // observed nothing and accepts the prefix, so the only outcome besides
        // an error is `Applied`; anything else is a broken assumption and is
        // surfaced instead of installing an overlay that was never built.
        match overlay.refresh(true)? {
            ReadRefresh::Applied => {}
            ReadRefresh::NeedsReload => {
                return Err(StorageError::Internal(
                    "the writer's transaction overlay unexpectedly needed a checkpoint reload"
                        .to_owned(),
                ));
            }
        }
        *self.read_journal.lock() = Some(overlay);
        Ok(())
    }

    fn enable_read_journal_inner(
        &self,
        path: impl AsRef<std::path::Path>,
        pool: Option<crate::layout::ShardType>,
    ) -> Result<(), StorageError> {
        // Bind the overlay to the coverage of the index this handle loaded, not
        // whatever `journal.lsn` says now: a concurrent checkpoint may already
        // have advanced past this handle's in-memory index.
        let covered = self.read_covered_lsn.load(Ordering::Acquire);
        *self.read_journal.lock() = Some(ReadJournal::empty(
            path.as_ref().to_path_buf(),
            covered,
            pool,
        ));
        if let Err(error) = self.refresh_read_journal() {
            *self.read_journal.lock() = None;
            return Err(error);
        }
        Ok(())
    }

    /// Read records at read-committed visibility: the durable index plus any
    /// committed-but-unflushed journal group.
    ///
    /// This is the cross-process freshness path for read-only workers, and it
    /// is deliberately separate from the durable API. A key absent from the
    /// durable index is filled from a complete journal group (commit trailer
    /// present) even before the writer fsyncs or advances the index
    /// checkpoint. A committed collection delete shadows durable records.
    ///
    /// Without [`Self::enable_read_journal`] this degrades to
    /// [`Self::get_many_with_refresh`].
    ///
    /// # Errors
    /// Propagates errors from the durable read or journal scan. An unresolved
    /// gap after journal reclamation returns [`StorageError::Corrupt`], or
    /// [`StorageError::Io`] with `WouldBlock` if checkpoint coverage advanced
    /// during the reload attempts.
    pub fn get_read_committed(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];
        let mut unresolved: Vec<usize> = Vec::new();

        // Refresh outside the read lock: this may reload the checkpoint-bound
        // index if the writer reclaimed past this reader's coverage. The
        // returned guard keeps the applied overlay locked for the lookup below,
        // so another reader cannot reset it in between.
        let guard = self.refresh_read_journal()?;

        {
            if let Some(overlay) = guard.as_ref() {
                match overlay.puts.get(collection_id) {
                    Some(committed) => {
                        for (index, id) in ids.iter().enumerate() {
                            match committed.get(id) {
                                Some((payload, _)) => {
                                    results[index] = Some(NodeData::new(payload.clone()));
                                }
                                None => unresolved.push(index),
                            }
                        }
                    }
                    None => unresolved.extend(0..ids.len()),
                }
                // A delete the checkpoint does not yet cover means the durable
                // index may still hold pre-delete records, so falling back for
                // missing keys would resurrect them. Return the overlay's view
                // (post-delete puts only) until the delete is covered.
                if overlay.delete_lsn.contains_key(collection_id) {
                    return Ok(results);
                }
            } else {
                unresolved.extend(0..ids.len());
            }
        }
        drop(guard);

        if !unresolved.is_empty() {
            let durable_ids: Vec<NodeId> = unresolved.iter().map(|&index| ids[index]).collect();
            let durable = self.get_many_with_refresh(collection_id, &durable_ids)?;
            for (index, value) in unresolved.into_iter().zip(durable) {
                results[index] = value;
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    const COLLECTION: [u8; 16] = [0x42; 16];
    const REUSED: [u8; 16] = [0x02; 16];

    fn put(node: [u8; 16], payload: &[u8]) -> JournalMutation {
        JournalMutation::Put {
            collection_id: COLLECTION,
            node_id: node,
            payload: payload.to_vec(),
        }
    }

    fn temp_wal(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mtxdb_read_journal_{label}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("wal.bin")
    }

    /// LSN 1 and LSN 2 (`stale-`), returning the length that keeps only LSN 1.
    fn write_two_groups(wal: &std::path::Path) -> u64 {
        let (mut journal, _) = Journal::open(wal).unwrap();
        journal.append_group(&[put([0x01; 16], b"stable")]).unwrap();
        let first_len = fs::metadata(wal).unwrap().len();
        journal.append_group(&[put(REUSED, b"stale-")]).unwrap();
        first_len
    }

    /// A restart that drops LSN 2 and reissues it with the same length.
    fn restart_with_reissued_lsn(wal: &std::path::Path, keep_len: u64) {
        fs::OpenOptions::new()
            .write(true)
            .open(wal)
            .unwrap()
            .set_len(keep_len)
            .unwrap();
        let (mut journal, _) = Journal::open(wal).unwrap();
        journal.append_group(&[put(REUSED, b"fresh-")]).unwrap();
    }

    fn value(overlay: &ReadJournal) -> Option<Vec<u8>> {
        overlay
            .puts
            .get(&COLLECTION)
            .and_then(|nodes| nodes.get(&REUSED))
            .map(|(bytes, _)| bytes.to_vec())
    }

    /// A file that has been quiet longer than the timestamp granularity is
    /// trusted on an unchanged fingerprint: repeated refreshes read nothing.
    #[test]
    fn an_unchanged_quiet_file_is_not_re_read() {
        let wal = temp_wal("quiet_skip");
        write_two_groups(&wal);
        std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        assert!(matches!(overlay.refresh(true), Ok(ReadRefresh::Applied)));
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        for _ in 0..50 {
            assert!(matches!(overlay.refresh(true), Ok(ReadRefresh::Applied)));
        }
        assert_eq!(
            overlay.tail_reads, 0,
            "an unchanged quiet file needs no read"
        );
        assert_eq!(overlay.tail_resets, 0);
    }

    /// Appending to the same segment reuses the held descriptor when the
    /// overlay records each new consumed tail; only the initial recording
    /// opens the WAL.
    #[test]
    fn same_file_appends_reuse_the_tail_descriptor() {
        let wal = temp_wal("reuse_tail_descriptor");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        assert_eq!(overlay.tail_opens, 1);

        for (node, payload) in [([0x03; 16], &b"third"[..]), ([0x04; 16], &b"fourth"[..])] {
            let (mut journal, _) = Journal::open(&wal).unwrap();
            journal.append_group(&[put(node, payload)]).unwrap();
            drop(journal);
            overlay.refresh(true).unwrap();
        }

        assert_eq!(
            overlay.tail_opens, 1,
            "same-file appends must reuse the WAL"
        );
    }

    /// A same-length rewrite of a quiet file changes its change time, so the
    /// fingerprint no longer matches, the window is compared, and the overlay
    /// is rebuilt. The file's length and inode are exactly what they were.
    #[test]
    fn a_same_length_rewrite_of_a_quiet_file_is_detected() {
        let wal = temp_wal("quiet_rewrite");
        let keep_len = write_two_groups(&wal);
        std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        let len_before = fs::metadata(&wal).unwrap().len();

        restart_with_reissued_lsn(&wal, keep_len);
        assert_eq!(fs::metadata(&wal).unwrap().len(), len_before);
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
        assert_eq!(overlay.tail_resets, 1);
    }

    /// Reclaim renames a replacement over the segment. Even a byte-identical
    /// replacement is a different file, and is caught by its inode before any
    /// bytes are compared.
    #[test]
    fn a_renamed_in_replacement_is_detected_by_identity_alone() {
        let wal = temp_wal("rename_replace");
        write_two_groups(&wal);
        std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        let reads_before = overlay.tail_reads;

        let copy = wal.with_extension("copy");
        fs::copy(&wal, &copy).unwrap();
        fs::rename(&copy, &wal).unwrap();

        overlay.refresh(true).unwrap();
        assert_eq!(
            overlay.tail_resets, 1,
            "a replaced file must rebuild the overlay"
        );
        assert_eq!(
            overlay.tail_reads, reads_before,
            "the replacement is caught before any window is read"
        );
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    }

    /// The real path: a stamp recorded right after a write is warm and not
    /// trusted, then promoted after one verifying read once the file has gone
    /// quiet. If the machine is so slow that the file was already quiet at the
    /// first refresh, there is no warm state to observe and the test returns.
    #[test]
    fn a_freshly_recorded_stamp_is_verified_then_promoted_once_quiet() {
        let wal = temp_wal("fresh_then_quiet");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        if overlay.tail_quiet {
            eprintln!(
                "skipped: the file was already quiet at the first refresh, so there \
                 is no warm stamp to observe on this machine"
            );
            return;
        }
        std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));
        for _ in 0..20 {
            overlay.refresh(true).unwrap();
        }
        assert_eq!(overlay.tail_reads, 1, "one verifying read, then trusted");
        assert!(overlay.tail_quiet);
        assert_eq!(overlay.tail_resets, 0);
    }

    /// On a platform without inode numbers a replaced file cannot be told apart
    /// by identity, and the persistent descriptor would keep naming the old
    /// file. The overlay must then compare what is at the path now: a
    /// same-length replacement on a new inode with different content is caught,
    /// and no descriptor is held. `force_no_identity` stands in for such a
    /// platform on unix.
    #[test]
    fn without_a_file_identity_a_replacement_is_caught_by_content() {
        let wal = temp_wal("no_identity_replace");
        let keep_len = write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.force_no_identity = true;
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        assert!(overlay.tail_stamp.is_none(), "no stamp without an identity");
        assert!(overlay.tail_file.is_none(), "no descriptor may be held");

        // Build the same-length replacement beside the segment and rename it
        // over the original, so the path now names a different inode.
        let replacement = wal.with_extension("replacement");
        fs::copy(&wal, &replacement).unwrap();
        restart_with_reissued_lsn(&replacement, keep_len);
        fs::rename(&replacement, &wal).unwrap();

        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
        assert_eq!(overlay.tail_resets, 1);
        // The counter only rises, so this also covers the first refresh: no
        // descriptor was opened to record a tail at any point.
        assert_eq!(
            overlay.tail_opens, 0,
            "no descriptor is opened to record a tail without an identity"
        );
    }

    /// Without a file identity an in-place same-length rewrite (same inode,
    /// new content) is caught by the same fresh-handle comparison.
    #[test]
    fn without_a_file_identity_an_in_place_rewrite_is_caught_by_content() {
        let wal = temp_wal("no_identity_rewrite");
        let keep_len = write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.force_no_identity = true;
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        let len_before = fs::metadata(&wal).unwrap().len();

        restart_with_reissued_lsn(&wal, keep_len);
        assert_eq!(fs::metadata(&wal).unwrap().len(), len_before);
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
        assert_eq!(overlay.tail_resets, 1);
        assert_eq!(overlay.tail_opens, 0);
    }

    /// An identity that disappears between recording and refresh (a stamp was
    /// recorded, none is available now) means the file is not known to be the
    /// one the overlay was built from, so the overlay is rebuilt.
    #[test]
    #[cfg(unix)]
    fn a_stamp_that_vanishes_between_refreshes_forces_a_rebuild() {
        let wal = temp_wal("identity_flip");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        assert!(overlay.tail_stamp.is_some());
        overlay.force_no_identity = true;
        overlay.refresh(true).unwrap();
        assert_eq!(overlay.tail_resets, 1);
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        assert!(
            overlay.tail_stamp.is_none(),
            "the rebuild records without an identity"
        );
        assert!(overlay.tail_file.is_none());
    }

    /// The reverse flip: a segment recorded without an identity that reports
    /// one on a later refresh is not known to be the file the overlay was built
    /// from, so it is rebuilt and recorded with the identity.
    #[test]
    #[cfg(unix)]
    fn an_identity_that_appears_between_refreshes_forces_a_rebuild() {
        let wal = temp_wal("identity_appears");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.force_no_identity = true;
        overlay.refresh(true).unwrap();
        assert!(overlay.tail_stamp.is_none());
        overlay.force_no_identity = false;
        overlay.refresh(true).unwrap();
        assert_eq!(overlay.tail_resets, 1);
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
        assert!(
            overlay.tail_stamp.is_some(),
            "the rebuild records the identity now available"
        );
        assert!(overlay.tail_file.is_some());
    }

    /// Without a file identity an unchanged segment is not rebuilt: the window
    /// is compared through a fresh handle each refresh and matches.
    #[test]
    fn without_a_file_identity_an_unchanged_segment_is_not_rebuilt() {
        let wal = temp_wal("no_identity_unchanged");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.force_no_identity = true;
        overlay.refresh(true).unwrap();
        for _ in 0..10 {
            overlay.refresh(true).unwrap();
        }
        assert_eq!(overlay.tail_resets, 0);
        assert_eq!(overlay.tail_reads, 10, "one comparison per refresh");
        assert_eq!(
            overlay.tail_opens, 0,
            "no descriptor is retained; fresh compare opens are not counted"
        );
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    }

    /// State-model test, not a real recording: the stamp is flipped to
    /// untrusted by hand after a genuine quiet period, to check the promotion
    /// rule in isolation from timing. The real path is covered by
    /// `a_freshly_recorded_stamp_is_verified_then_promoted_once_quiet`.
    #[test]
    fn a_warm_fingerprint_is_verified_then_promoted_once_quiet() {
        let wal = temp_wal("warm_then_quiet");
        write_two_groups(&wal);
        std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        // Model a stamp recorded while the file was still warm.
        overlay.tail_quiet = false;
        for _ in 0..20 {
            overlay.refresh(true).unwrap();
        }
        assert_eq!(overlay.tail_reads, 1, "one verifying read, then trusted");
        assert!(overlay.tail_quiet);
        assert_eq!(overlay.tail_resets, 0);
    }

    /// A stamp that is not yet trusted must keep validating the tail: a
    /// same-length rewrite is caught even though length and inode match.
    #[test]
    fn an_untrusted_fingerprint_still_validates_the_tail() {
        let wal = temp_wal("untrusted_rewrite");
        let keep_len = write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        overlay.tail_quiet = false;

        restart_with_reissued_lsn(&wal, keep_len);
        overlay.refresh(true).unwrap();
        assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
        assert_eq!(overlay.tail_resets, 1);
    }

    /// State-model test: a stamp recorded earlier but unavailable now cannot
    /// be compared, so it fails closed and rebuilds. Only platforms that
    /// report inode numbers and change times record a stamp at all.
    #[cfg(unix)]
    #[test]
    fn a_stamp_that_cannot_be_compared_fails_closed() {
        let wal = temp_wal("stamp_mismatch");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        assert!(
            overlay.tail_stamp.is_some(),
            "a unix platform must record a stamp"
        );
        overlay.tail_stamp = None;
        overlay.refresh(true).unwrap();
        assert_eq!(overlay.tail_resets, 1);
    }

    /// The remembered window is composed from consumed bytes across refreshes
    /// (the previous window plus what each scan consumed). The previous window
    /// ends exactly at the scan's start offset, `observed_valid_len`, and the
    /// scan's bytes begin there, so the two are contiguous. The composed window
    /// must always equal the window a later comparison reads from disk,
    /// including once the file outgrows the window, which is where the join and
    /// the trim in `record_observed_tail` matter.
    #[test]
    fn the_composed_window_matches_the_disk_after_every_refresh() {
        let wal = temp_wal("composed_window");
        let (mut journal, _) = Journal::open(&wal).unwrap();
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        let file = File::open(&wal).unwrap();
        for round in 0..12u8 {
            journal
                .append_group(&[put([round; 16], &[round; 300])])
                .unwrap();
            overlay.refresh(true).unwrap();
            let mut on_disk = Vec::new();
            assert!(Journal::read_tail_ending_at(
                &file,
                overlay.observed_valid_len,
                TAIL_WINDOW,
                &mut on_disk
            )
            .unwrap());
            assert_eq!(
                overlay.observed_tail.as_deref(),
                Some(on_disk.as_slice()),
                "round {round}: the remembered window must equal the disk"
            );
        }
        assert_eq!(overlay.tail_resets, 0);
    }

    /// If groups were applied but their tail could not be recorded, nothing
    /// protects the overlay, so the next refresh must rebuild it and try again
    /// instead of trusting the prefix.
    #[test]
    fn an_unrecorded_tail_forces_a_rebuild() {
        let wal = temp_wal("unrecorded_tail");
        write_two_groups(&wal);
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        overlay.refresh(true).unwrap();
        assert!(overlay.observed_tail.is_some());

        overlay.observed_tail = None;
        overlay.refresh(true).unwrap();
        assert_eq!(overlay.tail_resets, 1);
        assert!(
            overlay.observed_tail.is_some(),
            "the rebuild records the tail again"
        );
        assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    }

    /// An empty segment has no consumed bytes to protect; refreshing it must
    /// not rebuild forever.
    #[test]
    fn an_empty_segment_does_not_rebuild_on_every_refresh() {
        let wal = temp_wal("empty_segment");
        drop(Journal::open(&wal).unwrap());
        let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
        for _ in 0..5 {
            overlay.refresh(true).unwrap();
        }
        assert_eq!(overlay.tail_resets, 0);
    }
}
