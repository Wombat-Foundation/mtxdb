//! Cross-process read-committed overlay, gated behind the `multi-reader`
//! feature (see `Cargo.toml` and `docs/TODO.txt`'s extraction plan).
//!
//! Lets a separate OS process observe a live writer's committed-but-not-yet-
//! checkpointed data by scanning the writer's journal segment read-only. A
//! single-process embedded deployment never calls any of this.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::journal::{Journal, Mutation as JournalMutation};
use crate::storage::{NodeData, NodeId, StorageError};

use super::{OpenTimings, PackfileStorage, ReloadMode};

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
    /// Committed puts as `(payload, lsn)`, keyed `collection_id -> node_id`.
    pub(super) puts: ReadJournalPuts,
    /// Highest committed delete LSN per collection. This is a persistent
    /// boundary: durable fallback stays suppressed for the collection until the
    /// checkpoint covers the delete, because the durable index may still hold
    /// pre-delete records. A later put does not clear it.
    pub(super) delete_lsn: HashMap<[u8; 16], u64>,
}

impl ReadJournal {
    pub(super) fn empty(path: PathBuf, covered: u64) -> Self {
        Self {
            path,
            covered,
            observed_len: 0,
            observed_valid_len: 0,
            observed_lsn: 0,
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

    /// Whether a reset (reclaimed/rotated) segment now begins past the
    /// coverage this reader's durable index incorporated.
    ///
    /// After a reset the overlay is rebuilt from the segment alone. If the
    /// segment's base LSN is past `covered + 1`, the writer reclaimed groups
    /// this reader's index never absorbed: those records are in neither the
    /// overlay nor the index. This uses the header base LSN, not the first
    /// group, so a fully reclaimed (header-only) segment — where the segment
    /// has no groups at all — is still detected. A missing/short segment
    /// reports base 0 and is not a gap.
    pub(super) fn reset_has_coverage_gap(scan: &crate::journal::Scan, covered: u64) -> bool {
        scan.base_lsn > covered.saturating_add(1)
    }

    /// Apply every committed group above `observed_lsn` to the overlay.
    ///
    /// A missing segment is treated as empty only before this reader has seen
    /// any journal bytes; if an observed segment disappears, the caller must
    /// reload the checkpoint. A segment that shrank since the last scan was
    /// reclaimed, so the overlay is rebuilt from scratch. Never repairs or
    /// creates the file, matching a read-only worker's constraints.
    ///
    /// Returns [`ReadRefresh::NeedsReload`] when a reset reveals the writer
    /// reclaimed past this reader's incorporated coverage; the caller must
    /// reload the checkpoint-bound index and rebind `covered` before retrying.
    fn refresh(&mut self) -> Result<ReadRefresh, StorageError> {
        // Coverage is fixed to the index this reader loaded. Reading
        // `journal.lsn` fresh would prune entries the reader's stale index has
        // not incorporated yet, dropping records from both sources.
        let covered = self.covered;
        let len = match fs::metadata(&self.path) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if self.observed_len != 0 || self.observed_lsn > covered {
                    return Ok(ReadRefresh::NeedsReload);
                }
                0
            }
            Err(error) => return Err(StorageError::Io(error)),
        };
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
            if (reset || full_scan) && Self::reset_has_coverage_gap(&scan, covered) {
                return Ok(ReadRefresh::NeedsReload);
            }
            for group in &scan.groups {
                for entry in &group.entries {
                    if entry.lsn <= self.observed_lsn || entry.lsn <= covered {
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
    fn refresh_read_journal(&self) -> Result<(), StorageError> {
        const RELOAD_ATTEMPTS: usize = 8;
        const MAX_BACKOFF_MS: u64 = 64;
        // Capture coverage once, before the first attempt. Comparing the final
        // value against a per-iteration snapshot would only detect coverage that
        // advanced on the *last* attempt, misreporting a moving checkpoint as a
        // genuine gap. Reloads only ever advance `read_covered_lsn`, so a strict
        // increase across the whole loop is exactly the retryable signal.
        let coverage_before_attempts = self.read_covered_lsn.load(Ordering::Acquire);
        for attempt in 0..RELOAD_ATTEMPTS {
            let mut guard = self.read_journal.lock();
            let Some(overlay) = guard.as_mut() else {
                return Ok(());
            };
            if overlay.refresh()? == ReadRefresh::Applied {
                return Ok(());
            }
            if !self.reload_index_from_checkpoint() {
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
            drop(guard);
            if attempt.saturating_add(1) < RELOAD_ATTEMPTS {
                let delay_ms = 1_u64
                    .checked_shl(u32::try_from(attempt).unwrap_or(u32::MAX))
                    .unwrap_or(MAX_BACKOFF_MS)
                    .min(MAX_BACKOFF_MS);
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
        }
        if self.read_covered_lsn.load(Ordering::Acquire) > coverage_before_attempts {
            Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "read-committed checkpoint coverage advanced during reload; retry the read",
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
    /// read API ([`StorageEngine::get_many`], [`Self::get_many_with_refresh`])
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
        // Bind the overlay to the coverage of the index this handle loaded, not
        // whatever `journal.lsn` says now: a concurrent checkpoint may already
        // have advanced past this handle's in-memory index.
        let covered = self.read_covered_lsn.load(Ordering::Acquire);
        *self.read_journal.lock() = Some(ReadJournal::empty(path.as_ref().to_path_buf(), covered));
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
        // index if the writer reclaimed past this reader's coverage.
        self.refresh_read_journal()?;

        {
            let guard = self.read_journal.lock();
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
