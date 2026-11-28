use std::fs;
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::index::checkpoint::{CheckpointWrite, PreparedCheckpoint};
use crate::index::delta::{self, DeltaOperation, DELTA_LOG_HEADER_LEN};
use crate::index::redo::{RedoOp, RedoRecord};
use crate::index::BucketIndex;
use crate::journal::pool_from_tag;
use crate::packfile::PackId;
use crate::storage::StorageError;

#[cfg(test)]
use super::{retire_delta_epoch_file, TailHook};
use super::{
    CapturedCheckpoint, CheckpointBodySource, CheckpointBreakdown, CheckpointTail,
    CheckpointWorker, DeltaBatchTooLarge, DeltaLogState, PackfileStorage, PendingDelta,
    RecordLocator, RoomGeneration, V1PendingBatch, PACK_INDEX_OFFSET_LIMIT,
};

impl PackfileStorage {
    /// Record a successful index insertion as a logical redo record, if the delta
    /// log can legitimately continue through it.
    ///
    /// The record names the change by content identity (`hash`) and its
    /// locator (`pack_id`, `offset`, `record_len`), not by table position, and
    /// takes its `delta_seq` from this pool's counter under the delta-state lock,
    /// so sequence order is append order across collections. Recording stops
    /// once a snapshot or tombstone supersedes the collection's pending updates.
    /// A generation mismatch, or a locator a record cannot represent, promotes
    /// the collection to a snapshot instead.
    pub(super) fn record_redo(
        &self,
        collection_id: &[u8; 16],
        generation: u64,
        hash: &[u8; 16],
        locator: RecordLocator,
        removed: bool,
    ) {
        let RecordLocator {
            slot,
            offset,
            len: record_len,
        } = locator;
        // A slot with no open shard has no stable pack id to name. Falling back
        // to the slot number would name a different pack that happens to carry
        // that number, so such a record is not representable: the collection is
        // snapshotted instead.
        let pack_id = self.shards.get_shard(slot).map(|shard| shard.pack_id);
        let record_len = u32::try_from(record_len).unwrap_or(0);
        let representable =
            pack_id.is_some() && record_len != 0 && offset <= PACK_INDEX_OFFSET_LIMIT;
        let pack_id = pack_id.unwrap_or(PackId([0; crate::packfile::PACK_ID_LEN]));
        let mut state = self.delta_state.lock();
        let base_generation_matches =
            state.base_generations.get(collection_id) == Some(&generation);
        if !representable || !base_generation_matches {
            // Replaces pending records too: the snapshot taken at sync time
            // includes them.
            if !matches!(
                state.pending.get(collection_id),
                Some(PendingDelta::Delete { .. })
            ) {
                state.pending.insert(*collection_id, PendingDelta::Snapshot);
            }
            return;
        }
        if matches!(
            state.pending.get(collection_id),
            Some(PendingDelta::Snapshot | PendingDelta::Delete { .. })
        ) {
            return;
        }
        state.delta_seq_high = state.delta_seq_high.saturating_add(1);
        let record = RedoRecord {
            collection_id: *collection_id,
            op: RedoOp::Set {
                full_hash: *hash,
                pack_id,
                offset,
                record_len,
                removed,
            },
            delta_seq: state.delta_seq_high,
            base_generation: generation,
        };
        match state.pending.entry(*collection_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(PendingDelta::Redo(vec![record]));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if let PendingDelta::Redo(records) = entry.get_mut() {
                    records.push(record);
                }
            }
        }
    }

    /// Snapshot every collection's live generation for a checkpoint, encoded as
    /// `(collection_id, generation, room)`. Callers hold all collection put
    /// mutexes, so the snapshot is a consistent point in the index.
    pub(super) fn checkpoint_snapshots(&self) -> Vec<([u8; 16], u64, Arc<RoomGeneration>)> {
        let tables = self.index_tables.read();
        tables
            .collection_order
            .iter()
            .filter_map(|collection_id| {
                tables.collections.get(collection_id).map(|g| {
                    let generation = arc_swap::ArcSwapAny::load_full(g);
                    (*collection_id, generation.generation, generation)
                })
            })
            .collect()
    }

    /// The journal LSN this store's checkpoint covers: its own pool's committed
    /// watermark on a shared segment (one group can interleave other pools'
    /// frames), or the global committed LSN for a per-pool segment. `None` when
    /// no journal is enabled.
    ///
    /// The value is floored by the durable coverage already recorded beside the
    /// checkpoint. A pool whose frames have all been reclaimed has no committed
    /// group left to derive a watermark from, so the fresh coordinator reports
    /// zero for it; writing that zero would erase the durable coverage and make
    /// the next replay start from a stale bound. Coverage only ever advances.
    pub(super) fn checkpoint_covered_lsn(&self) -> Option<u64> {
        self.journal().map(|journal| {
            let committed = match pool_from_tag(self.journal_pool.load(Ordering::Acquire)) {
                Some(pool) => journal.committed_lsn_for_pool(pool),
                None => journal.committed_lsn(),
            };
            // The shared-WAL coverage batch and the full checkpoint both run
            // `sync_all` after capturing this and before recording it, so a
            // frame at or below `committed` that has been applied is covered.
            // `a_power_cut_image_never_loses_a_record_a_claim_covered` checks
            // it for autocommit writes and shared-WAL transactions, on records
            // made durable by a full sync of both pools (an inflated claim
            // fails it). A pool synced alone while another stays silent, and
            // per-pool coordinators, are not exercised. `max` only keeps the
            // value from moving backwards.
            committed.max(self.durable_coverage())
        })
    }

    /// Snapshot this pool's logical versions through the checkpoint coverage
    /// boundary. Versions above that boundary remain in the retained WAL and
    /// must not be claimed by the checkpoint.
    pub(super) fn checkpoint_logical_versions(
        &self,
        covered_lsn: Option<u64>,
    ) -> Vec<([u8; 16], u64)> {
        if let (Some(journal), Some(pool), Some(covered_lsn)) = (
            self.journal(),
            pool_from_tag(self.journal_pool.load(Ordering::Acquire)),
            covered_lsn,
        ) {
            return journal.collection_versions_through(pool, covered_lsn);
        }
        let _ = covered_lsn;
        Vec::new()
    }

    /// Persist the full per-collection index state to `index.checkpoint`, so
    /// the next open can load it instead of rescanning every packfile.
    ///
    /// Called after every sync barrier — packfile data first, checkpoint
    /// second. A crash in between leaves a fingerprint mismatch that the next
    /// open resolves with a rescan; a crash after leaves a valid checkpoint.
    ///
    /// The rewrite hands off the delta log across two epochs: the session
    /// keeps appending to the log that continues the *old* checkpoint (D0)
    /// right up to the rotation below, after which every `put` is recorded
    /// against a fresh epoch (D1) named for the new checkpoint's fingerprint.
    /// D0's file is untouched through the rotation and retired only once the
    /// new checkpoint (C1) is durably renamed, so a crash at any point pairs
    /// the loader with whichever log legitimately continues the checkpoint on
    /// disk: before C1's rename C0 + D0 is authoritative, after it C1 + D1.
    /// The log's fingerprint-based name is itself the safety gate — a stale
    /// predecessor (D0) or an orphaned successor (D1) can never match the
    /// checkpoint a reopen loads (see [`Self::delta_path`]).
    ///
    /// Cheap no-op when no collection data has changed since the last write
    /// (`index_checkpoint_dirty` cleared on success), so a writer that syncs
    /// repeatedly without writes doesn't rewrite the acceleration file. The
    /// flag is cleared only when nothing after this rewrite's snapshot is
    /// left unpersisted (see the lock re-acquisition below), so a transient
    /// failure or a racing write defers rather than drops the update.
    ///
    /// Returns `true` when a checkpoint was written, `false` when nothing was
    /// dirty to persist — callers that count writes must gate on this, since a
    /// concurrent sync can consume the dirty flag first.
    pub(super) fn persist_index_checkpoint(&self) -> Result<bool, StorageError> {
        // A synchronous checkpoint never overlaps a worker's.
        self.finish_checkpoint_worker(true);
        let Some(captured) = self.capture_checkpoint()? else {
            return Ok(false);
        };
        let CapturedCheckpoint { tail, breakdown } = captured;
        match tail.run() {
            Ok(done) => {
                self.record_checkpoint(breakdown, &done);
                Ok(true)
            }
            Err(error) => {
                self.abandon_checkpoint();
                Err(error)
            }
        }
    }

    /// The part of a checkpoint that needs the locks: everything up to the
    /// point where the image bytes and the delta-epoch rotation are captured.
    /// After it returns the returned tail owns everything it needs, so it can
    /// run on any thread. `None` when nothing is dirty.
    pub(super) fn capture_checkpoint(&self) -> Result<Option<CapturedCheckpoint>, StorageError> {
        let started = std::time::Instant::now();
        let mut breakdown = CheckpointBreakdown::default();
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // With a journal, this checkpoint must make the packs durable before it
        // records coverage, and it does so below with every collection locked,
        // which stalls every writer for as long as the fsync takes. Do the bulk
        // of that here first, with no lock held: the fsync under the locks then
        // only covers what was written since.
        if self.journal().is_some() {
            self.shards.sync_dirty()?;
        }
        breakdown.pre_sync = started.elapsed();
        // Open the temp file before taking any collection lock: creating it
        // touches the parent directory, which can contend with a concurrent
        // directory sync or dirty-page throttle. Foreground puts must never
        // wait on that while holding every collection's put_mutex.
        let checkpoint_path = Self::index_checkpoint_path(&self.base_dir);
        let prepared = PreparedCheckpoint::create(&checkpoint_path).map_err(StorageError::Io)?;
        let lock_started = std::time::Instant::now();
        // Epoch-handoff protocol: hold every collection's put_mutex only
        // across the snapshot + epoch rotation below (same sorted-lock
        // pattern as `retire_empty_shards_after_batch`), then release it
        // before the comparatively slow serialize + fsync + rename. This is
        // safe — where releasing right after the fingerprint snapshot alone
        // was *not* (see git history on this function) — because rotation
        // switches every future `put`'s delta frames onto a fresh epoch
        // (named for the fingerprint captured here) before the locks drop,
        // rather than leaving them targeting an epoch that's about to be
        // silently discarded:
        //
        //   - Every put after this point is recorded, if at all, against the
        //     *new* epoch (D1), never the old one (D0) — `rotate_delta_epoch`
        //     runs inside the same locked section as the fingerprint/index
        //     snapshot, so there is no window where a put's frame could still
        //     land on D0 after D0's corresponding checkpoint state has
        //     already been captured for D1's own checkpoint.
        //   - D0's on-disk file is left completely untouched by the rotation
        //     — only retired (deleted) after the new checkpoint (C1) is
        //     durably renamed, below. So a crash before C1 exists leaves the
        //     old checkpoint (C0) still correctly paired with D0.
        //   - D1 is only ever trusted by a reopen if it names the exact
        //     fingerprint of the checkpoint that reopen loads (see
        //     `delta_path`) — so a crash after rotation but before C1 is
        //     durable leaves D1 orphaned (no checkpoint names its
        //     fingerprint) and correctly ignored; the reopen uses C0 + D0.
        //
        // `RoomGeneration` is immutable once published (COW, swapped
        // atomically), so an owned `Arc` clone taken under the lock is as
        // good a snapshot as the live one for serialization purposes.
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        // The guards borrow these `Arc`s, so the handles must outlive the
        // guards; this vector is also re-used for the end-of-function
        // dirty-check re-lock so late-added collections stay pinned too.
        let mut lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();

        // Exclude new-collection publication from the fingerprint→snapshot
        // window. A put *to* an existing collection is already gated by that
        // collection's `put_mutex`, but a put that is about to *create* a
        // collection holds no mutex this function acquired, so its record
        // frame could be Eager-flushed into a pack between our `flush_all` and
        // our fingerprint read — the fingerprint would then name a record
        // whose collection never made it into the snapshot, and a reopen that
        // trusts C1 would silently drop it. Publishing puts hold this lock
        // shared across the entire put; acquiring it exclusive here (before
        // the fingerprint, held through the rotation) means no such put can
        // interleave its flush with our capture: it either finished
        // beforehand (its collection is in the snapshot) or runs entirely
        // afterwards (its bytes land post-fingerprint, and the fingerprint
        // mismatch on reopen routes to the full rescan).
        let create_guard = self.collection_creation.write();
        // A collection created between the initial scan above and this lock
        // wasn't in the locked set and must be now — otherwise a put to one
        // serially dispatched by the engine could still squeeze under the
        // fingerprint while its publication was snapshotted.
        let ids_now: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        for id in ids_now.iter().filter(|id| !collection_ids.contains(id)) {
            lock_arcs.push(self.put_mutex(id));
        }
        // Now build the lock guards: the arcs must outlive them (collected
        // in `lock_arcs` above and reused below for the dirty-check).
        let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();
        breakdown.lock_wait = lock_started.elapsed();
        let locked_started = std::time::Instant::now();

        // Commit any buffered frames first: the fingerprint below pins each
        // shard to its committed on-disk length, and the serialized index
        // offsets must be readable against exactly that length on the next
        // open. Writing the checkpoint while records are still buffered
        // would persist virtual offsets the fingerprint can't describe.
        //
        // With a journal enabled the checkpoint is also the point at which the
        // packfiles themselves must become durable — the WAL only covers the
        // post-checkpoint suffix — so fsync every shard here, not just flush.
        if self.journal().is_some() {
            self.shards.sync_all()?;
        } else {
            self.shards.flush_all()?;
        }
        breakdown.locked_sync = locked_started.elapsed();
        let snapshot_started = std::time::Instant::now();
        let live_shards = self.shards.all_shards();
        let packs: Vec<(PackId, u64)> = live_shards
            .iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        let fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);
        // Every pack live right now, keyed by this writer's own local slot —
        // lets a reader translate this checkpoint's slot-encoded index
        // entries to its own local slot for the same pack_id, rather than
        // trusting the writer's raw slot number (see the checkpoint pack-table
        // documentation for why that's unsafe after a retirement).
        let pack_table: Vec<(u16, PackId)> = live_shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();

        let snapshots = self.checkpoint_snapshots();
        let serialize_started = std::time::Instant::now();

        // Rotate the delta epoch while still locked. Every collection's
        // writers target the new fingerprint by the time the locks drop; the
        // tail preserves that epoch and retires all others after the checkpoint
        // is durable, including rejected epochs from a full rescan.
        self.rotate_delta_epoch(fingerprint, &snapshots);
        // Every put mutex is held, so no redo record is in flight: the sequence
        // reached so far is exactly what this checkpoint incorporates.
        let base_delta_seq = self.delta_state.lock().delta_seq_high;
        self.delta_state.lock().base_pack_ids =
            pack_table.iter().map(|&(_, pack_id)| pack_id).collect();
        // Every collection's put mutex is held here, so no `put` is mid-flight
        // and every mutation published so far has completed its index update.
        // The recorded LSN must be *committed*, not merely published: a
        // concurrent put can publish above the last WAL commit, and LSNs above
        // `committed_lsn` are discarded and reused after a crash. Recording one
        // as covered would make a reopen skip a future mutation that reuses
        // that LSN (the journal scan only knows the committed prefix). The
        // committed LSN is conservative -- the fsynced packs above may cover
        // more -- and replaying that suffix again is idempotent.
        //
        // On a shared segment the global committed LSN can include other pools'
        // frames, which this checkpoint does not materialize. Record this
        // pool's own committed watermark instead, so its coverage and the
        // reclaim floor both describe only frames this index holds.
        let wal_lsn = self.checkpoint_covered_lsn();
        let logical_versions = self.checkpoint_logical_versions(wal_lsn);
        let source = CheckpointBodySource {
            fingerprint,
            base_delta_seq,
            covered_lsn: wal_lsn,
            snapshots: &snapshots,
            pack_table: &pack_table,
            logical_versions: &logical_versions,
        };
        let prepared = self.write_checkpoint_body(prepared, &source)?;
        breakdown.serialize = serialize_started.elapsed();
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        drop(guards);
        breakdown.snapshot = snapshot_started
            .elapsed()
            .saturating_sub(breakdown.serialize);
        // A collection published after this point lands after `fingerprint`
        // (its frames change a pack length), so a reopen that trusts C1 either
        // finds its frames in the new epoch or fails the tail-fingerprint gate
        // and rescans; it can never be silently omitted, because the image was
        // captured with publication excluded. The creation lock therefore drops
        // with the put mutexes rather than riding out the tail: every in-flight
        // sync appends to the new epoch through `append_index_delta_v1`, which
        // also takes this lock, so holding it across the tail would stall every
        // sync for the tail's duration and defeat the background checkpoint.
        drop(create_guard);
        let tail = CheckpointTail {
            base_dir: self.base_dir.clone(),
            fingerprint,
            covered_lsn: wal_lsn,
            logical_versions,
            prepared,
            journal: self.journal(),
            pool_tag: self.journal_pool.load(Ordering::Acquire),
            durable_coverage: Arc::clone(&self.durable_coverage),
            #[cfg(test)]
            hook: self.checkpoint_tail_hook.lock().take(),
        };
        breakdown.total = started.elapsed();
        Ok(Some(CapturedCheckpoint { tail, breakdown }))
    }

    pub(super) fn write_checkpoint_body(
        &self,
        prepared: PreparedCheckpoint,
        source: &CheckpointBodySource<'_>,
    ) -> Result<PreparedCheckpoint, StorageError> {
        let result = (|| {
            // Snapshot dimensions are O(1) from each resident index, so the
            // directory can be sized before any snapshot byte is produced.
            let mut metadata = Vec::with_capacity(source.snapshots.len());
            for (collection_id, generation, room) in source.snapshots {
                let snapshot_bytes = u32::try_from(room.index.serialized_len()).map_err(|_| {
                    std::io::Error::other("bucket snapshot exceeds the checkpoint u32 length field")
                })?;
                let entry_count = u32::try_from(room.index.len()).map_err(|_| {
                    std::io::Error::other("bucket snapshot entry count exceeds u32")
                })?;
                metadata.push((*collection_id, *generation, snapshot_bytes, entry_count));
            }
            let input = CheckpointWrite {
                fingerprint: source.fingerprint,
                // The fixed-width header uses zero for absent coverage, as the
                // existing format does. Keep the Option in CheckpointTail for
                // the post-install journal decision.
                covered_lsn: source.covered_lsn.unwrap_or(0),
                base_delta_seq: source.base_delta_seq,
                collections: &metadata,
                pack_table: source.pack_table,
                logical_versions: source.logical_versions,
            };
            prepared.write_body(&input, |index, writer| {
                #[cfg(test)]
                if self.checkpoint_prepare_fail.swap(false, Ordering::Relaxed) {
                    return Err(std::io::Error::other("injected checkpoint prepare failure"));
                }
                source.snapshots[index]
                    .2
                    .index
                    .write_snapshot_to(metadata[index].3 as usize, writer)
            })
        })();
        result.map_err(|error| {
            self.abandon_checkpoint();
            StorageError::Io(error)
        })
    }

    /// Serialize snapshots and write their checkpoint, for the test mirrors of
    /// the production capture and [`CheckpointTail::run`].
    #[cfg(test)]
    pub(super) fn write_checkpoint_snapshot(
        path: &Path,
        fingerprint: u64,
        covered_lsn: u64,
        base_delta_seq: u64,
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
        pack_table: &[(u16, PackId)],
        logical_versions: &[([u8; 16], u64)],
    ) -> std::io::Result<std::time::Duration> {
        let serialize_started = std::time::Instant::now();
        let entries: Vec<([u8; 16], u64, Vec<u8>)> = snapshots
            .iter()
            .map(|(collection_id, generation, room)| {
                (*collection_id, *generation, room.index.serialize())
            })
            .collect();
        let blobs: Vec<([u8; 16], u64, &[u8])> = entries
            .iter()
            .map(|(collection_id, generation, blob)| (*collection_id, *generation, blob.as_slice()))
            .collect();
        let serialized = serialize_started.elapsed();
        crate::index::checkpoint::write_checkpoint(
            path,
            fingerprint,
            covered_lsn,
            base_delta_seq,
            &blobs,
            pack_table,
            logical_versions,
        )?;
        Ok(serialized)
    }

    /// Fold a finished tail into the capture's breakdown and publish it.
    pub(super) fn record_checkpoint(
        &self,
        mut breakdown: CheckpointBreakdown,
        done: &CheckpointBreakdown,
    ) {
        breakdown.write = done.write;
        breakdown.directory_sync = done.directory_sync;
        breakdown.journal_lsn = done.journal_lsn;
        breakdown.reclaim = done.reclaim;
        breakdown.retire = done.retire;
        breakdown.checkpoint_bytes = done.checkpoint_bytes;
        // The capture's own time plus the tail's, not the time until somebody
        // collected the worker.
        breakdown.total = breakdown.total.saturating_add(done.total);
        *self.last_checkpoint_breakdown.lock() = Some(breakdown);
    }

    /// A checkpoint whose tail failed did not become durable: the old
    /// checkpoint and its log still pair, but the epoch this session rotated onto
    /// names an image that does not exist. Forget it so the next sync rewrites
    /// the checkpoint rather than appending to an orphaned log.
    pub(super) fn abandon_checkpoint(&self) {
        self.delta_state.lock().base_fingerprint = None;
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
    }

    /// Collect the background checkpoint tail. Returns whether none is left
    /// running. With `wait` it blocks until the tail ends; without, it only
    /// collects one that already has. A failed or panicked tail is reported and
    /// abandoned, and the old checkpoint and log stay valid.
    pub(super) fn finish_checkpoint_worker(&self, wait: bool) -> bool {
        let mut slot = self.checkpoint_worker.lock();
        let Some(worker) = slot.take() else {
            return true;
        };
        if !wait && !worker.handle.is_finished() {
            *slot = Some(worker);
            return false;
        }
        drop(slot);
        match worker.handle.join() {
            Ok(Ok(done)) => self.record_checkpoint(worker.breakdown, &done),
            Ok(Err(error)) => {
                eprintln!("mtxdb: background checkpoint failed: {error}");
                self.abandon_checkpoint();
            }
            Err(_) => {
                eprintln!("mtxdb: background checkpoint panicked");
                self.abandon_checkpoint();
            }
        }
        true
    }

    /// Collect a finished checkpoint tail before a sync decides what to do.
    /// Returns `true` when none is still running.
    ///
    /// A tail still running holds the WAL: no coverage may be claimed and
    /// nothing reclaimed until its image is durable, and a second checkpoint
    /// must not start. It is waited for only when the WAL nears its cap or the
    /// delta log its rotation point (reclaim is never suppressed indefinitely).
    /// Otherwise this sync's operations go to the new epoch's log and the sync
    /// is done, which is why `false` means the caller returns.
    pub(super) fn settle_checkpoint_worker(&self) -> bool {
        let must_wait = self
            .journal()
            .is_some_and(|journal| journal.in_emergency_zone())
            || self.delta_state.lock().log_bytes >= self.delta_log_rotate_bytes();
        let had_tail = self.checkpoint_worker.lock().is_some();
        if self.finish_checkpoint_worker(must_wait) {
            return true;
        }
        if had_tail {
            self.syncs_with_tail_in_flight
                .fetch_add(1, Ordering::Relaxed);
            if must_wait {
                self.syncs_waited_for_tail.fetch_add(1, Ordering::Relaxed);
            }
        }
        if self.index_checkpoint_dirty.load(Ordering::Relaxed)
            && !self.delta_state.lock().pending.is_empty()
        {
            match self.append_index_delta() {
                Ok(()) => self.shard_collections_stale.store(true, Ordering::Relaxed),
                Err(error) => {
                    eprintln!("mtxdb: delta append during a checkpoint failed: {error}");
                }
            }
        }
        false
    }

    /// Test-only: hold the next background checkpoint tail until the returned
    /// sender is used or dropped; with `fail` the tail then errors instead of
    /// installing.
    #[cfg(test)]
    pub(crate) fn hold_next_checkpoint_tail(&self, fail: bool) -> std::sync::mpsc::Sender<()> {
        let (release, gate) = std::sync::mpsc::channel();
        *self.checkpoint_tail_hook.lock() = Some(TailHook { gate, fail });
        release
    }

    /// Test-only: fail the next checkpoint capture after its delta epoch has
    /// rotated, so the next sync must abandon and rewrite the image.
    #[cfg(test)]
    pub(crate) fn fail_next_checkpoint_prepare(&self) {
        self.checkpoint_prepare_fail.store(true, Ordering::Relaxed);
    }

    /// Whether a checkpoint tail is running on its worker thread.
    #[must_use]
    pub fn checkpoint_in_flight(&self) -> bool {
        self.checkpoint_worker.lock().is_some()
    }

    /// Block until any background checkpoint has finished.
    ///
    /// Holds `index_persist_lock` while the tail is joined: the worker slot is
    /// emptied before the join, so without the lock a sync could see no worker
    /// and capture a new checkpoint while this one's tail is still sweeping.
    /// Must not be called with that lock already held.
    pub fn wait_for_checkpoint(&self) {
        let _persist_guard = self.index_persist_lock.lock();
        self.finish_checkpoint_worker(true);
    }

    /// Let a sync that needs a full checkpoint hand its tail (write, fsync,
    /// install, coverage, reclaim, retire) to a worker thread. Off by default.
    pub fn set_background_checkpoint(&self, enabled: bool) {
        self.background_checkpoint.store(enabled, Ordering::Relaxed);
    }

    /// Full checkpoint for a sync: capture under the locks, then run the tail on
    /// a worker when enabled, inline otherwise. Logs and swallows a failure like
    /// [`Self::persist_index_checkpoint_best_effort`].
    pub(super) fn persist_index_checkpoint_for_sync(&self) {
        if !self.background_checkpoint.load(Ordering::Relaxed) {
            // `persist_index_checkpoint_or_delta` already holds
            // `index_persist_lock` across this call.
            self.persist_index_checkpoint_best_effort();
            return;
        }
        if let Err(error) = self.start_checkpoint_worker() {
            eprintln!("mtxdb: failed to persist index checkpoint: {error}");
        }
    }

    /// Capture a checkpoint under the locks and hand its tail to a worker
    /// thread, returning once the capture is done. A failed spawn abandons the
    /// capture (the next sync rewrites the checkpoint).
    pub(super) fn start_checkpoint_worker(&self) -> Result<(), StorageError> {
        let Some(CapturedCheckpoint { tail, breakdown }) = self.capture_checkpoint()? else {
            return Ok(());
        };
        let spawned = std::thread::Builder::new()
            .name("mtxdb-checkpoint".into())
            .spawn(move || tail.run());
        match spawned {
            Ok(handle) => {
                self.checkpoint_tails_started
                    .fetch_add(1, Ordering::Relaxed);
                *self.checkpoint_worker.lock() = Some(CheckpointWorker { handle, breakdown });
                Ok(())
            }
            Err(error) => {
                self.abandon_checkpoint();
                Err(StorageError::Io(error))
            }
        }
    }

    /// Switch the live delta-log state onto a fresh epoch named for
    /// `fingerprint`, the checkpoint about to be written. Must be called
    /// while every collection's `put_mutex` is held (see
    /// `persist_index_checkpoint`), so every `put` from here on is recorded,
    /// if at all, against the new epoch — never the one this call is
    /// replacing. Returns the *previous* base fingerprint (the epoch to
    /// retire once the new checkpoint is durable), or `None` if this session
    /// had no prior epoch (opened via a full rescan).
    pub(super) fn rotate_delta_epoch(
        &self,
        fingerprint: u64,
        snapshots: &[([u8; 16], u64, Arc<RoomGeneration>)],
    ) -> Option<u64> {
        let mut state = self.delta_state.lock();
        let old_base_fingerprint = state.base_fingerprint;
        state.base_fingerprint = Some(fingerprint);
        state.base_generations = snapshots
            .iter()
            .map(|(collection_id, generation, _)| (*collection_id, *generation))
            .collect();
        state.base_order = snapshots
            .iter()
            .enumerate()
            .map(|(order, (collection_id, _, _))| {
                (*collection_id, u64::try_from(order).unwrap_or(u64::MAX))
            })
            .collect();
        state.next_order_key = u64::try_from(snapshots.len()).unwrap_or(u64::MAX);
        state.pending.clear();
        state.log_bytes = 0;
        state.log_synced_bytes = 0;
        // The rotated epoch names a new file, so no directory sync has yet
        // completed for that name.
        state.log_dir_durable = false;
        state.log_version = 1;
        old_base_fingerprint
    }

    /// Test-only: a byte-for-byte mirror of `persist_index_checkpoint`,
    /// except it sleeps for `delay` right after releasing the locks (in
    /// place of, not in addition to, the production function's body — this
    /// is never called from non-test code and the production function is
    /// untouched). Lets a test make the unlocked window arbitrarily long
    /// without depending on a dataset large enough to make a real
    /// serialize+fsync slow relative to test-machine noise. See
    /// `test_put_does_not_block_on_slow_checkpoint_rewrite`.
    #[cfg(test)]
    pub(super) fn test_persist_index_checkpoint_with_delay(
        &self,
        delay: std::time::Duration,
        entered_unlocked_window: &AtomicBool,
    ) -> Result<(), StorageError> {
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            entered_unlocked_window.store(true, Ordering::Release);
            return Ok(());
        }
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mut lock_arcs: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let create_guard = self.collection_creation.write();
        let ids_now: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        for id in ids_now.iter().filter(|id| !collection_ids.contains(id)) {
            lock_arcs.push(self.put_mutex(id));
        }
        let guards: Vec<_> = lock_arcs.iter().map(|arc| arc.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let snapshots = self.checkpoint_snapshots();
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
        drop(guards);
        drop(create_guard);

        // The test waits on this: it now *knows* the rewrite is sleeping in
        // its unlocked window rather than guessing via a fixed sleep, so a
        // put issued from here provably overlaps the delay.
        entered_unlocked_window.store(true, Ordering::Release);

        std::thread::sleep(delay);

        let pack_table: Vec<(u16, PackId)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            self.delta_state.lock().delta_seq_high,
            &snapshots,
            &pack_table,
            &[],
        )
        .map_err(StorageError::Io)?;
        let _ = crate::shard::sync_directory(&self.base_dir);
        self.retire_delta_epoch(old_base_fingerprint, fingerprint);
        let guards = lock_arcs.iter().map(|m| m.lock()).collect::<Vec<_>>();
        let has_unfinished_work = {
            let state = self.delta_state.lock();
            !state.pending.is_empty()
        };
        if !has_unfinished_work {
            self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        }
        drop(guards);
        self.persist_shard_collections_best_effort();
        Ok(())
    }

    /// Test-only: turn a copy of this pool's directory into the disk a power
    /// cut would leave, by cutting every pack back to the length an fsync has
    /// covered. Bytes written but not yet fsynced are what a crash may lose;
    /// bytes still buffered in memory are not in the copy at all.
    #[cfg(test)]
    pub(crate) fn test_cut_packs_to_synced(&self, image_dir: &Path) {
        for (_, shard) in self.shards.all_shards() {
            // A pack that has not reached a durability barrier exists only
            // under its staging name in the image; cut that copy instead.
            let name = shard.data_path();
            let name = name.file_name().expect("a pack has a file name");
            let file = fs::OpenOptions::new()
                .write(true)
                .open(image_dir.join(name))
                .expect("the image holds every pack");
            file.set_len(shard.synced_len().min(shard.file_len()))
                .expect("cut the pack");
        }
    }

    /// Test-only: perform exactly the locked prefix of
    /// `persist_index_checkpoint` (flush, fingerprint, index snapshot, epoch
    /// rotation) and then stop, *never* writing or renaming a checkpoint.
    /// Simulates a crash between the D0→D1 rotation and C1's rename — the
    /// window where D1 exists only in memory and no on-disk checkpoint names
    /// its fingerprint yet.
    #[cfg(test)]
    pub(super) fn test_rotate_epoch_without_checkpoint(&self) -> (u64, Option<u64>) {
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all().unwrap();
        let fingerprint = self.current_pack_fingerprint();
        let snapshots = self.checkpoint_snapshots();
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        (fingerprint, old_base_fingerprint)
    }

    /// Test-only: perform everything `persist_index_checkpoint` does up
    /// through the checkpoint rename and its directory fsync, but stop
    /// *before* `retire_delta_epoch`. Simulates a crash between C1 becoming
    /// durable and D0's retirement — the window where both C1+D1 (current)
    /// and C0's now-stale D0 (orphaned) exist on disk simultaneously.
    #[cfg(test)]
    pub(super) fn test_write_checkpoint_without_retire(
        &self,
    ) -> Result<(u64, Option<u64>), StorageError> {
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();
        self.shards.flush_all()?;
        let fingerprint = self.current_pack_fingerprint();
        let snapshots = self.checkpoint_snapshots();
        let old_base_fingerprint = self.rotate_delta_epoch(fingerprint, &snapshots);
        let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
        drop(guards);
        let pack_table: Vec<(u16, PackId)> = self
            .shards
            .all_shards()
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        Self::write_checkpoint_snapshot(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            covered_lsn,
            self.delta_state.lock().delta_seq_high,
            &snapshots,
            &pack_table,
            &[],
        )
        .map_err(StorageError::Io)?;
        let _ = crate::shard::sync_directory(&self.base_dir);
        Ok((fingerprint, old_base_fingerprint))
    }

    /// Remove the now-superseded delta-log epoch, once the checkpoint that
    /// makes it safe to discard is durably renamed. A no-op if this session
    /// had no prior epoch. Best-effort: a failure here leaves the retired
    /// epoch's file on disk, which is always safe (see `delta_path`) — it
    /// simply names a fingerprint no future checkpoint will ever carry again.
    #[cfg(test)]
    pub(super) fn retire_delta_epoch(
        &self,
        old_base_fingerprint: Option<u64>,
        live_fingerprint: u64,
    ) {
        retire_delta_epoch_file(&self.base_dir, old_base_fingerprint, live_fingerprint);
    }

    /// Append the pending delta operations and clear the dirty flag. The
    /// on-disk log, if any, is
    /// continued; otherwise a fresh header pins `base_fingerprint` (the
    /// checkpoint the frames extend) for the reopen replay gate.
    pub(super) fn append_index_delta(&self) -> Result<(), StorageError> {
        self.append_index_delta_v1(false).map(|_| ())
    }

    /// The delta-log cap, the bytes still free under it for a batch, and the
    /// length of a batch before any operation is added to it.
    pub(super) fn delta_batch_budget(
        &self,
        state: &DeltaLogState,
        with_coverage: bool,
    ) -> (u64, u64, usize) {
        let cap = self.delta_log_cap();
        let header = if state.log_bytes == 0 {
            DELTA_LOG_HEADER_LEN as u64
        } else {
            0
        };
        let budget = cap.saturating_sub(state.log_bytes).saturating_sub(header);
        let mut projected = delta::v1_empty_batch_len();
        if with_coverage {
            projected = projected.saturating_add(delta::v1_coverage_frame_len());
        }
        (cap, budget, projected)
    }

    pub(super) fn serialize_snapshot_index(index: &BucketIndex) -> (usize, Vec<u8>) {
        (index.serialized_len(), index.serialize())
    }

    pub(super) fn build_v1_pending_batch(
        &self,
        state: &DeltaLogState,
        with_coverage: bool,
    ) -> Result<V1PendingBatch, StorageError> {
        // The batch's length is known from its parts without building them, so
        // one that cannot fit under the delta-log cap is refused before a
        // whole-index snapshot is serialized just to be thrown away.
        let (cap, budget, mut projected) = self.delta_batch_budget(state, with_coverage);
        let too_large = |projected: usize| {
            StorageError::Io(std::io::Error::other(DeltaBatchTooLarge {
                batch_bytes: projected,
                log_bytes: state.log_bytes,
                cap,
            }))
        };
        let collections = self.collections_read();
        let mut operations = Vec::new();
        let mut redo: Vec<RedoRecord> = Vec::new();
        let mut pending_ids: Vec<_> = state.pending.keys().copied().collect();
        pending_ids.sort_unstable();
        let mut generation_updates = Vec::new();
        let mut order_updates = Vec::new();
        let mut next_order_key = state.next_order_key;
        for collection_id in pending_ids {
            let pending = state
                .pending
                .get(&collection_id)
                .expect("pending id exists");
            match pending {
                PendingDelta::Redo(records) => {
                    let Some(base_generation) = state.base_generations.get(&collection_id) else {
                        return Err(StorageError::Io(std::io::Error::other(
                            "redo records target a collection without a base",
                        )));
                    };
                    if records
                        .iter()
                        .any(|record| record.base_generation != *base_generation)
                    {
                        return Err(StorageError::Io(std::io::Error::other(
                            "a redo record's generation differs from its base",
                        )));
                    }
                    projected = projected
                        .saturating_add(records.len().saturating_mul(delta::v1_redo_frame_len()));
                    redo.extend(records.iter().copied());
                }
                PendingDelta::Snapshot => {
                    if let Some(room) = collections.get(&collection_id) {
                        let generation = arc_swap::ArcSwapAny::load_full(room);
                        // The v1 payload is the canonical slots-only form.
                        let blob_len = generation.index.serialized_len();
                        let frame_len =
                            delta::v1_snapshot_frame_len(blob_len).ok_or_else(|| {
                                StorageError::Io(std::io::Error::other("v1 batch size overflow"))
                            })?;
                        projected = projected.saturating_add(frame_len);
                        if u64::try_from(projected).unwrap_or(u64::MAX) > budget {
                            return Err(too_large(projected));
                        }
                        let (_, index_blob) = Self::serialize_snapshot_index(&generation.index);
                        let order_key =
                            if let Some(order_key) = state.base_order.get(&collection_id) {
                                *order_key
                            } else {
                                let order_key = next_order_key;
                                next_order_key = next_order_key.saturating_add(1);
                                order_key
                            };
                        operations.push(DeltaOperation::CollectionSnapshot {
                            collection_id,
                            generation: generation.generation,
                            order_key,
                            index_blob,
                        });
                        generation_updates.push((collection_id, generation.generation));
                        order_updates.push((collection_id, order_key));
                    } else if let Some(&generation) = state.base_generations.get(&collection_id) {
                        projected = projected.saturating_add(delta::v1_tombstone_frame_len());
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
                PendingDelta::Delete { generation } => {
                    if state.base_generations.contains_key(&collection_id) {
                        projected = projected.saturating_add(delta::v1_tombstone_frame_len());
                        operations.push(DeltaOperation::CollectionTombstone {
                            collection_id,
                            generation: *generation,
                        });
                        generation_updates.push((collection_id, 0));
                    }
                }
            }
        }
        // Serialized in `delta_seq` order, whatever the collection order, so a
        // reader that validates strict monotonicity sees the append order.
        redo.sort_unstable_by_key(|record| record.delta_seq);
        operations.extend(redo.into_iter().map(DeltaOperation::Redo));
        Ok(V1PendingBatch {
            operations,
            generation_updates,
            order_updates,
            next_order_key,
        })
    }

    pub(super) fn write_v1_delta_batch(
        &self,
        state: &mut DeltaLogState,
        base_fingerprint: u64,
        operations: &[DeltaOperation],
        tail_fingerprint: u64,
        durable: bool,
    ) -> Result<(), StorageError> {
        let path = Self::delta_path(&self.base_dir, base_fingerprint);
        let on_disk_len = fs::metadata(&path)
            .map_or_else(
                |error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Ok(0)
                    } else {
                        Err(error)
                    }
                },
                |metadata| Ok(metadata.len()),
            )
            .map_err(StorageError::Io)?;
        if on_disk_len < state.log_bytes {
            return Err(StorageError::Io(std::io::Error::other(
                "v1 delta log shrank below its sealed frontier",
            )));
        }
        if on_disk_len > state.log_bytes {
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(state.log_bytes))
                .map_err(StorageError::Io)?;
        }
        let write_header = state.log_bytes == 0;
        let batch_bytes = delta::v1_batch_len(operations)
            .ok_or_else(|| StorageError::Io(std::io::Error::other("v1 batch size overflow")))?;
        let next_len = state
            .log_bytes
            .saturating_add(u64::try_from(batch_bytes).unwrap_or(u64::MAX))
            .saturating_add(if write_header {
                DELTA_LOG_HEADER_LEN as u64
            } else {
                0
            });
        let cap = self.delta_log_cap();
        if next_len > cap {
            return Err(StorageError::Io(std::io::Error::other(
                DeltaBatchTooLarge {
                    batch_bytes,
                    log_bytes: state.log_bytes,
                    cap,
                },
            )));
        }
        let bytes_written = delta::append_batch_with_durability_and_directory(
            &path,
            write_header,
            base_fingerprint,
            operations,
            tail_fingerprint,
            durable,
            state.log_dir_durable,
        )
        .map_err(StorageError::Io)?;
        state.log_bytes = state
            .log_bytes
            .saturating_add(u64::try_from(bytes_written).unwrap_or(u64::MAX));
        if durable {
            state.log_synced_bytes = state.log_bytes;
            // The append returned `Ok` only after syncing the entry (or
            // skipping it because it was already durable). A failed sync
            // returns `Err`, leaving the flag unset for the next attempt.
            state.log_dir_durable = true;
        } else {
            // Pay for the log's durability in slices as it grows. Best effort:
            // the log is acceleration, and a coverage batch fsyncs it again.
            let budget = self.delta_fsync_budget_bytes.load(Ordering::Relaxed);
            if budget != 0 && state.log_bytes.saturating_sub(state.log_synced_bytes) >= budget {
                match fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .and_then(|file| file.sync_data())
                {
                    Ok(()) => state.log_synced_bytes = state.log_bytes,
                    Err(error) => eprintln!("mtxdb: delta log fsync failed: {error}"),
                }
            }
        }
        Ok(())
    }

    /// Append v1 operations. Collection locks pin every live index
    /// and the collection-creation lock prevents an unrepresented collection
    /// from appearing between pack flush and snapshot capture.
    ///
    /// With `with_coverage` the batch also carries a claim that the packs
    /// durably cover this pool's journal frames through the returned LSN, so
    /// the journal can be reclaimed without a full checkpoint. The ordering that
    /// makes the claim true: with every collection locked (so no put is
    /// mid-flight and everything published for this pool has been applied) the
    /// LSN is captured and every pack is fsynced, and only then is the batch,
    /// claim last, appended and fsynced. A crash before the batch is durable
    /// leaves no claim; a torn batch fails its CRC and claims nothing.
    pub(super) fn append_index_delta_v1(
        &self,
        with_coverage: bool,
    ) -> Result<Option<u64>, StorageError> {
        if with_coverage {
            // Make the bulk of the pack bytes durable before any lock is taken;
            // the fsync under the locks below then only covers what was written
            // since.
            self.shards.sync_dirty()?;
        }
        let create_guard = self.collection_creation.write();
        let mut ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        {
            let state = self.delta_state.lock();
            ids.extend(state.pending.keys().copied());
        }
        ids.sort_unstable();
        ids.dedup();
        let lock_arcs: Vec<_> = ids.iter().map(|id| self.put_mutex(id)).collect();
        let guards: Vec<_> = lock_arcs.iter().map(|lock| lock.lock()).collect();

        // A put may have completed after the caller's initial dirty flush but
        // before these locks were acquired. Flush again while the snapshot set
        // is stable so the tail fingerprint covers every serialized slot.
        self.shards.flush_all()?;
        let covered = if with_coverage {
            let covered = self.checkpoint_covered_lsn();
            // Everything up to `covered` is applied and flushed; make it durable
            // before claiming it. `sync_all`, not `sync_dirty`: a claim must not
            // rest on a dirty bit some other syncer cleared.
            self.shards.sync_all()?;
            covered
        } else {
            None
        };
        let tail_fingerprint = self.current_pack_fingerprint();

        // Copy the small state-machine metadata, then release its mutex while
        // serializing whole-index snapshots. Collection put locks and the
        // creation lock keep this snapshot stable; the persistence lock keeps
        // another sync from advancing the on-disk frontier concurrently.
        let snapshot_state = self.delta_state.lock().clone();
        let Some(base_fingerprint) = snapshot_state.base_fingerprint else {
            return Err(StorageError::Io(std::io::Error::other(
                "v1 delta append with no base fingerprint",
            )));
        };
        if snapshot_state.log_version != 1 {
            return Err(StorageError::Io(std::io::Error::other(
                "v1 delta append without a clean v1 checkpoint base",
            )));
        }
        if snapshot_state.pending.is_empty() && covered.is_none() {
            return Err(StorageError::Io(std::io::Error::other(
                "v1 delta append with no pending operations",
            )));
        }

        let batch = self.build_v1_pending_batch(&snapshot_state, with_coverage)?;
        let V1PendingBatch {
            mut operations,
            generation_updates,
            order_updates,
            next_order_key,
        } = batch;

        let mut state = self.delta_state.lock();
        if state.pending != snapshot_state.pending
            || state.base_fingerprint != snapshot_state.base_fingerprint
            || state.log_bytes != snapshot_state.log_bytes
            || state.log_version != snapshot_state.log_version
        {
            return Err(StorageError::Io(std::io::Error::other(
                "v1 delta state changed while snapshot batch was prepared",
            )));
        }
        // The claim is the batch's last operation, directly after the version
        // table that keeps the pool's collection clocks once the reclaimed WAL
        // groups are gone. A reader applies the operations up to the claim and
        // holds exactly the index and versions it describes.
        if let Some(covered_lsn) = covered {
            let entries = self.checkpoint_logical_versions(Some(covered_lsn));
            operations.push(DeltaOperation::Versions {
                covered_lsn,
                entries,
            });
            operations.push(DeltaOperation::Coverage { covered_lsn });
        }
        self.write_v1_delta_batch(
            &mut state,
            base_fingerprint,
            &operations,
            tail_fingerprint,
            covered.is_some(),
        )?;
        for (collection_id, generation) in generation_updates {
            if generation == 0 {
                state.base_generations.remove(&collection_id);
                state.base_order.remove(&collection_id);
            } else {
                state.base_generations.insert(collection_id, generation);
            }
        }
        for (collection_id, order_key) in order_updates {
            state.base_order.insert(collection_id, order_key);
        }
        state.next_order_key = state.next_order_key.max(next_order_key);
        state.pending.clear();
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        drop(state);
        drop(guards);
        drop(create_guard);
        if let Some(covered_lsn) = covered {
            // The batch is durable, so the claim is too.
            self.durable_coverage
                .fetch_max(covered_lsn, Ordering::AcqRel);
        }
        Ok(covered)
    }

    /// Best-effort wrapper around [`Self::persist_index_checkpoint`] — logs
    /// and swallows a failure rather than turning it into a hard error. The
    /// checkpoint is an acceleration structure; packfiles remain
    /// authoritative, so a failed persist only costs the next open a rescan.
    ///
    /// The caller must hold `index_persist_lock`. Repack takes it before marking
    /// the index dirty; the sync path already holds it around the whole
    /// checkpoint/delta decision.
    pub(super) fn persist_index_checkpoint_best_effort(&self) {
        if let Err(e) = self.persist_index_checkpoint() {
            eprintln!("mtxdb: failed to persist index checkpoint: {e}");
        }
    }
}
