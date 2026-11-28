use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::cache::NodeCache;
use crate::index::delta::{self, DeltaOperation};
use crate::index::format::DeltaFrame;
use crate::index::redo::{RedoOp, RedoRecord};
use crate::index::{BucketIndex, InsertError};
use crate::packfile::PackId;
use crate::shard::ShardPool;

use super::{
    disk_bytes_from_physical, read_persisted_shard_collections, BookkeepingSource,
    CheckpointReject, DeltaLogState, DurableIndexState, OpenTimings, PackfileStorage, ReloadMode,
    RoomGeneration, RoomScanOutput,
};

impl PackfileStorage {
    /// Whether every pending redo set names a record that is really at its
    /// locator: the frame there belongs to the same collection and carries the
    /// same full hash. Any mismatch, or a frame that cannot be read, fails the
    /// whole log closed to the caller's rescan.
    fn redo_locators_verified(
        shards: &ShardPool,
        sets: &HashMap<[u8; 16], Vec<RedoRecord>>,
        pack_lookup: &HashMap<PackId, (u16, u64)>,
    ) -> Option<u64> {
        let mut identity_reads = 0_u64;
        for record in sets.values().flatten() {
            let RedoOp::Set {
                full_hash,
                pack_id,
                offset,
                ..
            } = record.op
            else {
                return None;
            };
            let shard = pack_lookup
                .get(&pack_id)
                .and_then(|(slot, _)| shards.get_shard(*slot))?;
            identity_reads = identity_reads.saturating_add(1);
            if !matches!(
                shards.record_identity_at(&shard, offset),
                Ok((collection, hash, _removed))
                    if collection == record.collection_id && hash == full_hash
            ) {
                return None;
            }
        }
        Some(identity_reads)
    }

    /// Apply logical redo sets, in `delta_seq` order, to a loaded index whose
    /// slots are already this process's. The index is pre-sized once for the whole
    /// batch and grown from its persisted identity tables when a growth boundary
    /// is reached; if it cannot grow (incomplete tables), the whole replay fails
    /// closed to the caller's rescan. Locators are checked for extent by the
    /// caller; identity is confirmed lazily by reads, which compare the record's
    /// hash.
    pub(super) fn apply_redo_sets(
        collection_id: &[u8; 16],
        index: BucketIndex,
        records: &[RedoRecord],
        pack_lookup: &HashMap<PackId, (u16, u64)>,
        mut identity_at: impl FnMut(u16, u64) -> Option<([u8; 16], [u8; 16])>,
    ) -> Option<BucketIndex> {
        let mut index = index;
        let target =
            BucketIndex::capacity_for_entries(index.len().saturating_add(records.len()), 0)?;
        while usize::try_from(index.capacity()).ok()? < target {
            index = index.grow()?;
        }
        for record in records {
            let RedoOp::Set {
                full_hash,
                pack_id,
                offset,
                removed,
                ..
            } = record.op
            else {
                return None;
            };
            let (slot, _) = *pack_lookup.get(&pack_id)?;
            loop {
                let inserted = index
                    .insert_with_identity(&full_hash, slot, offset, removed, |candidate| {
                        let (found_collection, found_hash) =
                            identity_at(candidate.shard_slot, candidate.offset).ok_or(())?;
                        if found_collection != *collection_id {
                            return Err(());
                        }
                        Ok(found_hash)
                    })
                    .ok()?;
                match inserted {
                    Ok(_) => break,
                    Err(InsertError::TableFull) => index = index.grow()?,
                }
            }
        }
        Some(index)
    }

    /// Fast-path index loading for `open_with_options`: build the
    /// per-collection state directly from the persisted-index checkpoint
    /// instead of rescanning every packfile.
    ///
    /// Returns `None` — falling through to the full rescan — for a missing,
    /// malformed, or stale checkpoint, one that doesn't describe the pack set
    /// currently on disk, or a delta log that can't be replayed onto the
    /// checkpoint. The fast path deliberately skips the torn-tail recovery
    /// scan, which is safe: a fingerprint match means the packs are the exact
    /// state the previous session's last sync fsynced, so there can be no torn
    /// tail to recover.
    ///
    /// On success, also returns the fresh [`DeltaLogState`] for the session: the
    /// delta log's base fingerprint (the checkpoint's) and each collection's
    /// checkpoint generation. A malformed or semantically inconsistent log
    /// makes this function return `None`, selecting the full rescan path.
    ///
    /// `reload_mode` selects the caller's stance (see [`ReloadMode`]).
    /// `ReloadMode::JournalBound` is for the read-committed overlay's reload
    /// (see `reload_index_from_checkpoint`): the caller is a read-only worker
    /// advancing its index/coverage pair to a writer's checkpoint, not a
    /// session opening the store. In that mode the exact-pack fingerprint gate
    /// and the delta-log replay are skipped. A writer that is still appending
    /// has already grown the live packs past the checkpoint, and the delta log
    /// may not yet cover that growth, so the gate would reject the checkpoint
    /// and fail the read closed — the failure mode that otherwise looks like a
    /// persistent miss. The overlay is authoritative for the post-checkpoint
    /// suffix, so the durable index only needs the checkpoint's base, and its
    /// coverage binds to exactly that base. The checkpoint itself must still be
    /// valid; only the "packs match exactly / log bridges to live" checks are
    /// dropped.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "one validation/materialization pipeline; splitting it would duplicate the fail-closed gates"
    )]
    pub(super) fn checkpoint_scan_out(
        base_dir: &std::path::Path,
        cache_capacity: usize,
        cache_budget: Option<&Arc<crate::cache::CacheBudget>>,
        shards: &ShardPool,
        open_shards: &[(u16, PackId, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
        writable: bool,
        reload_mode: ReloadMode,
        index_config: crate::index::IndexConfig,
        timings: &mut OpenTimings,
    ) -> Option<(RoomScanOutput, Vec<[u8; 16]>, DeltaLogState, u64)> {
        let decode_started = std::time::Instant::now();
        let checkpoint =
            crate::index::checkpoint::read_checkpoint(&Self::index_checkpoint_path(base_dir));
        timings.checkpoint_decode = decode_started.elapsed();
        let Some(checkpoint) = checkpoint else {
            timings.checkpoint_reject = Some(CheckpointReject::MissingOrMalformed);
            return None;
        };
        let covered_lsn = checkpoint.covered_lsn;
        let fingerprint_started = std::time::Instant::now();
        let packs: Vec<(PackId, u64)> = open_shards
            .iter()
            .map(|(_, pack_id, _, file_len)| (*pack_id, *file_len))
            .collect();
        let local_fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);
        timings.fingerprint = fingerprint_started.elapsed();

        // Generations at the checkpoint, per collection — used both to seed the
        // new session's `DeltaLogState` and to gate whether any delta frames may be
        // applied (a frame's stamp must equal the checkpoint generation it
        // claims to continue, which also rules out the never-recorded ancestors
        // of collections created or structurally changed since the checkpoint).
        let ckpt_generations: HashMap<[u8; 16], u64> = checkpoint
            .collections
            .iter()
            .map(|loaded| (loaded.collection_id, loaded.generation))
            .collect();

        let delta_path = Self::delta_path(base_dir, checkpoint.fingerprint);
        // A delta epoch rejected below is deliberately left on disk. A
        // shared-WAL caller runs `enable_shared_journal` only after this open
        // returns, and that check reads a matching epoch's version table as the
        // pool's durable baseline for history the WAL has already reclaimed.
        // Deleting a rejected epoch here would destroy that baseline before the
        // check can use it and turn a recoverable open into a permanent
        // refusal. A rejected epoch cannot be appended to later (a rescan
        // starts a fresh epoch), and the next successful checkpoint sweeps it.
        // Only the `!replay_needed` case below removes it eagerly: there the
        // packs still match the checkpoint, so there is no reclaimed history to
        // bridge, and the next append would reuse this exact path.
        // Only the epoch named for this exact checkpoint can ever be trusted
        // (see `delta_path`); anything else on disk — a stale predecessor
        // epoch left by a crash between checkpoint rename and retirement, or
        // an orphaned successor epoch from a rewrite that never completed —
        // is inert by construction, but sweep it now on a writable open so
        // it doesn't accumulate across sessions.
        if writable {
            Self::sweep_orphan_delta_logs(base_dir, Some(&delta_path));
        }
        // A delta log can only continue a checkpoint whose packs have advanced
        // (every append accompanies a flush that grows a pack, changing the
        // fingerprint). When the packs still match the checkpoint exactly,
        // any leftover log is stale — likely a crashed full rewrite that had
        // already committed the new checkpoint but not yet deleted its
        // predecessor log. Never replayed; removed on a writable open.
        // A read-journal reload deliberately skips the exact-pack gate and the
        // delta replay (see this function's doc): the overlay supplies the
        // committed suffix, and requiring the live pack set to match would fail
        // the read closed while a writer is still appending.
        let replay_needed =
            reload_mode == ReloadMode::Strict && local_fingerprint != checkpoint.fingerprint;
        if !replay_needed && writable {
            let _ = std::fs::remove_file(&delta_path);
        }
        // The new session continues whatever log is genuinely on disk — its
        // committed byte length must be inherited, not zeroed: a later append
        // decides whether to write a fresh base header from `log_bytes == 0`,
        // and re-headering an existing log would corrupt the batch framing the
        // next reader depends on. Zero only when the open removed the file.
        let mut log_bytes_on_disk = 0u64;

        // Gate the log (case B only): it must start at this checkpoint and end
        // at the exact pack set now on disk, and every committed frame must
        // target a checkpointed collection at its recorded checkpoint
        // generation. Any failure rejects the log wholesale and falls through
        // to the rescan — never a partial replay.
        let replay_frames: Vec<DeltaFrame> = Vec::new();
        let mut replay_operations = Vec::new();
        // The journal coverage the applied part of the delta log claims. A
        // coverage batch lets the journal be reclaimed past the checkpoint's own
        // coverage, so the index built here is only complete up to the claim if
        // the operations the claim describes were applied.
        let mut log_coverage: Option<u64> = None;
        // A checkpoint with no surviving delta log is a valid base for the
        // current v1 writer.
        let replay_log_version = 1;
        if replay_needed {
            let delta_started = std::time::Instant::now();
            if let Some(log) = delta::read_delta_log(&delta_path) {
                if log.base_fingerprint != checkpoint.fingerprint
                    || log.tail_fingerprint != local_fingerprint
                {
                    timings.delta_replay = delta_started.elapsed();
                    timings.checkpoint_reject = Some(CheckpointReject::DeltaFingerprintMismatch);
                    return None;
                }
                log_bytes_on_disk = log.file_len;
                log_coverage = log.coverage;
                replay_operations = log.operations;
            } else {
                timings.delta_replay = delta_started.elapsed();
                timings.checkpoint_reject = Some(CheckpointReject::DeltaLogMissing);
                return None;
            }
            timings.delta_replay = delta_started.elapsed();
            // Reachable only after a log validated against both fingerprints,
            // so a nonzero count means operations were actually replayed.
            timings.delta_replay_operations = u64::try_from(replay_operations.len())
                .unwrap_or(u64::MAX)
                .saturating_add(u64::try_from(replay_frames.len()).unwrap_or(u64::MAX));
        } else if reload_mode == ReloadMode::JournalBound {
            // A read-journal reload skips the exact-pack gate (the writer keeps
            // appending), but it must still see the index changes a coverage
            // batch describes, or a reclaim past the checkpoint's coverage would
            // leave records that are in neither the index nor the journal. Take
            // the delta operations up to the last coverage claim. Every one of
            // them was written after the packs they name were durable, and the
            // writer only appends coverage while every live pack is in this
            // checkpoint's pack table, so their slots translate below; if one
            // does not, the load fails closed like any other bad slot.
            let delta_started = std::time::Instant::now();
            if let Some(mut log) = delta::read_delta_log(&delta_path) {
                if log.base_fingerprint == checkpoint.fingerprint && log.coverage_prefix_ops > 0 {
                    log.operations.truncate(log.coverage_prefix_ops);
                    log_coverage = log.coverage;
                    replay_operations = log.operations;
                    timings.delta_replay_operations =
                        u64::try_from(replay_operations.len()).unwrap_or(u64::MAX);
                }
            }
            timings.delta_replay = delta_started.elapsed();
        }

        let prepare_started = std::time::Instant::now();
        // Group the (gated) frames by target collection once, so the
        // materialization loop below applies each collection's frames by hash
        // lookup rather than re-scanning the whole frame list per collection.
        let mut frames_by_collection: HashMap<[u8; 16], Vec<DeltaFrame>> = HashMap::new();
        for frame in &replay_frames {
            frames_by_collection
                .entry(frame.collection_id)
                .or_default()
                .push(*frame);
        }

        let mut live_generations: HashMap<[u8; 16], u64> = ckpt_generations
            .iter()
            .filter(|(collection_id, _)| !deleted_collections.contains(*collection_id))
            .map(|(collection_id, generation)| (*collection_id, *generation))
            .collect();
        let mut live_order: HashMap<[u8; 16], u64> = checkpoint
            .collections
            .iter()
            .filter(|loaded| !deleted_collections.contains(&loaded.collection_id))
            .enumerate()
            .map(|(order, loaded)| {
                (
                    loaded.collection_id,
                    u64::try_from(order).unwrap_or(u64::MAX),
                )
            })
            .collect();
        let mut snapshot_indexes: HashMap<[u8; 16], (u64, BucketIndex)> = HashMap::new();
        let mut v1_tombstones = HashSet::new();
        // Logical redo: every pack this log may name, by its stable id, with its
        // current length. The log's tail fingerprint pins those lengths, so a
        // record whose extent lies outside its pack is corruption.
        let pack_lookup: HashMap<PackId, (u16, u64)> = open_shards
            .iter()
            .map(|(slot, pack_id, _, file_len)| (*pack_id, (*slot, *file_len)))
            .collect();
        let mut redo_by_collection: HashMap<[u8; 16], Vec<RedoRecord>> = HashMap::new();
        let mut last_delta_seq = checkpoint.base_delta_seq;
        for operation in replay_operations {
            match operation {
                DeltaOperation::Redo(record) => {
                    // Strictly increasing above the checkpoint's base sequence.
                    if record.delta_seq <= last_delta_seq {
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaOperationInvalid);
                        return None;
                    }
                    last_delta_seq = record.delta_seq;
                    if deleted_collections.contains(&record.collection_id) {
                        continue;
                    }
                    let RedoOp::Set {
                        pack_id,
                        offset,
                        record_len,
                        ..
                    } = record.op
                    else {
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaOperationInvalid);
                        return None;
                    };
                    let extent_ok = pack_lookup.get(&pack_id).is_some_and(|(_, pack_len)| {
                        offset
                            .checked_add(u64::from(record_len))
                            .is_some_and(|end| end <= *pack_len)
                    });
                    if live_generations.get(&record.collection_id) != Some(&record.base_generation)
                        || !extent_ok
                    {
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaGenerationMismatch);
                        return None;
                    }
                    redo_by_collection
                        .entry(record.collection_id)
                        .or_default()
                        .push(record);
                }
                DeltaOperation::Incremental(frame) => {
                    if deleted_collections.contains(&frame.collection_id) {
                        continue;
                    }
                    if live_generations.get(&frame.collection_id) != Some(&frame.generation) {
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaGenerationMismatch);
                        return None;
                    }
                    // Physical-slot deltas name a legacy linear-probe position.
                    // They cannot be replayed into the bucket table; the
                    // caller falls back to the authoritative pack scan.
                    timings.checkpoint_reject = Some(CheckpointReject::DeltaApplyFailed);
                    return None;
                }
                DeltaOperation::CollectionSnapshot {
                    collection_id,
                    generation,
                    order_key,
                    index_blob,
                } => {
                    if deleted_collections.contains(&collection_id) {
                        continue;
                    }
                    let Ok(index) = BucketIndex::deserialize_with_config(&index_blob, index_config)
                    else {
                        // Legacy linear-probe snapshots are not interpreted as
                        // bucket tables. Their pack records remain the source
                        // of truth, so reopen safely migrates them by scanning.
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaApplyFailed);
                        return None;
                    };
                    frames_by_collection.remove(&collection_id);
                    redo_by_collection.remove(&collection_id);
                    snapshot_indexes.insert(collection_id, (generation, index));
                    live_generations.insert(collection_id, generation);
                    live_order.insert(collection_id, order_key);
                    v1_tombstones.remove(&collection_id);
                }
                DeltaOperation::CollectionTombstone {
                    collection_id,
                    generation,
                } => {
                    if deleted_collections.contains(&collection_id) {
                        continue;
                    }
                    if live_generations.get(&collection_id) != Some(&generation) {
                        timings.checkpoint_reject = Some(CheckpointReject::DeltaGenerationMismatch);
                        return None;
                    }
                    live_generations.remove(&collection_id);
                    live_order.remove(&collection_id);
                    frames_by_collection.remove(&collection_id);
                    redo_by_collection.remove(&collection_id);
                    snapshot_indexes.remove(&collection_id);
                    v1_tombstones.insert(collection_id);
                }
                // A coverage claim changes no index state; it is read
                // separately (see `durable_coverage_from_disk`).
                DeltaOperation::Coverage { .. } | DeltaOperation::Versions { .. } => {}
            }
        }

        // Every redo record's locator must lead to the record it names. A locator
        // that is in bounds but wrong would otherwise be installed, and the key
        // would read as missing (the read compares the frame's hash and skips a
        // mismatch). Offsets ascend within a pack in log order, since each set
        // names a freshly appended record, so this is a sequential pass over
        // pages the pack recovery has just read.
        let Some(mut index_identity_reads) =
            Self::redo_locators_verified(shards, &redo_by_collection, &pack_lookup)
        else {
            timings.checkpoint_reject = Some(CheckpointReject::RedoLocatorUnverified);
            return None;
        };

        let slot_to_pack_id: HashMap<u16, PackId> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*slot, *pack_id))
            .collect();

        // Translate every slot this checkpoint's index entries (and any
        // delta frame continuing it) encode — the writer's own local slot at
        // checkpoint-write time — to this reader's own local slot for the
        // same pack_id. Slot numbers are a process-local handle, not a
        // stable identity: a fresh reader's `discover_shards` reassigns
        // slots by first-free-in-pack_id-order, which does not reproduce
        // the writer's numbering once any shard has ever been retired (see
        // checkpoint pack-table documentation). A checkpoint_slot with no
        // entry here (its pack_id isn't among this reader's currently-open
        // packs) is left unmapped; any index entry that actually needs it
        // makes `remap_slots` fail closed below, forcing a full rescan
        // rather than serving an unresolvable slot.
        let pack_id_to_local_slot: HashMap<PackId, u16> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*pack_id, *slot))
            .collect();
        let slot_remap: HashMap<u16, u16> = checkpoint
            .pack_table
            .iter()
            .filter_map(|&(checkpoint_slot, pack_id)| {
                pack_id_to_local_slot
                    .get(&pack_id)
                    .map(|&local_slot| (checkpoint_slot, local_slot))
            })
            .collect();

        // The inspection sidecar is the same reduced per-(pack, collection)
        // bookkeeping the loop below would recover by walking every slot of
        // every collection index. Accept it only when it is gated to the pack
        // set now on disk and describes every non-deleted collection completely
        // (see `gated_sidecar_bookkeeping`); any mismatch falls through to the
        // faithful slot walk, because the sidecar is an acceleration and
        // correctness never depends on it.
        //
        // Gating needs each collection's post-replay length (the checkpoint
        // occupancy plus whatever fresh-slot writes the replayed frames add),
        // so the sidecar check runs after materialization below.
        let mut scan_out = RoomScanOutput {
            index_identity_reads,
            ..RoomScanOutput::default()
        };
        let mut collection_order = Vec::with_capacity(checkpoint.collections.len());
        let mut current_len: HashMap<[u8; 16], u32> =
            HashMap::with_capacity(checkpoint.collections.len());
        timings.replay_prepare = prepare_started.elapsed();
        let materialization_started = std::time::Instant::now();
        for loaded in &checkpoint.collections {
            if deleted_collections.contains(&loaded.collection_id)
                || v1_tombstones.contains(&loaded.collection_id)
            {
                // The checkpoint may predate the deletion marker; the
                // logical-delete set is authoritative.
                continue;
            }
            // The checkpoint reader has validated this range. Decode the
            // bucket snapshot directly into its owned table exactly once.
            let (mut index, generation) = if let Some((generation, index)) =
                snapshot_indexes.remove(&loaded.collection_id)
            {
                (index, generation)
            } else {
                let index = loaded
                    .snapshot_len
                    .checked_add(loaded.slots_offset)
                    .and_then(|end| checkpoint.mmap.get(loaded.slots_offset..end))
                    .and_then(|bytes| {
                        BucketIndex::deserialize_validated_checkpoint_snapshot(bytes, index_config)
                            .ok()
                    });
                let Some(index) = index else {
                    timings.checkpoint_reject = Some(CheckpointReject::IndexUpdateFailed);
                    return None;
                };
                (index, loaded.generation)
            };
            if !index.remap_slots(&slot_remap) {
                timings.checkpoint_reject = Some(CheckpointReject::SlotRemapFailed);
                return None;
            }
            let index = match redo_by_collection.remove(&loaded.collection_id) {
                Some(records) => {
                    if let Some(index) = Self::apply_redo_sets(
                        &loaded.collection_id,
                        index,
                        &records,
                        &pack_lookup,
                        |slot, offset| {
                            index_identity_reads = index_identity_reads.saturating_add(1);
                            let shard = shards.get_shard(slot)?;
                            let (collection, hash, _) =
                                shards.record_identity_at(&shard, offset).ok()?;
                            Some((collection, hash))
                        },
                    ) {
                        index
                    } else {
                        timings.checkpoint_reject = Some(CheckpointReject::IndexUpdateFailed);
                        return None;
                    }
                }
                None => index,
            };
            let Some(len) = u32::try_from(index.len()).ok() else {
                timings.checkpoint_reject = Some(CheckpointReject::IndexUpdateFailed);
                return None;
            };
            current_len.insert(loaded.collection_id, len);
            scan_out.collections.insert(
                loaded.collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::with_budget(
                        cache_capacity,
                        cache_budget.cloned(),
                    )),
                    generation,
                }),
            );
            collection_order.push(loaded.collection_id);
        }
        for (collection_id, (generation, mut index)) in snapshot_indexes {
            if deleted_collections.contains(&collection_id) {
                continue;
            }
            if !index.remap_slots(&slot_remap) {
                timings.checkpoint_reject = Some(CheckpointReject::SlotRemapFailed);
                return None;
            }
            let index = match redo_by_collection.remove(&collection_id) {
                Some(records) => {
                    if let Some(index) = Self::apply_redo_sets(
                        &collection_id,
                        index,
                        &records,
                        &pack_lookup,
                        |slot, offset| {
                            index_identity_reads = index_identity_reads.saturating_add(1);
                            let shard = shards.get_shard(slot)?;
                            let (collection, hash, _) =
                                shards.record_identity_at(&shard, offset).ok()?;
                            Some((collection, hash))
                        },
                    ) {
                        index
                    } else {
                        timings.checkpoint_reject = Some(CheckpointReject::IndexUpdateFailed);
                        return None;
                    }
                }
                None => index,
            };
            let Some(len) = u32::try_from(index.len()).ok() else {
                timings.checkpoint_reject = Some(CheckpointReject::IndexUpdateFailed);
                return None;
            };
            current_len.insert(collection_id, len);
            scan_out.collections.insert(
                collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::with_budget(
                        cache_capacity,
                        cache_budget.cloned(),
                    )),
                    generation,
                }),
            );
            collection_order.push(collection_id);
        }
        collection_order
            .sort_unstable_by_key(|id| (live_order.get(id).copied().unwrap_or(u64::MAX), *id));
        timings.index_materialization = materialization_started.elapsed();
        let bookkeeping_started = std::time::Instant::now();

        let mut all_deleted_collections = deleted_collections.clone();
        all_deleted_collections.extend(v1_tombstones.iter().copied());
        let (bookkeeping_source, sidecar_counts) = Self::gated_sidecar_bookkeeping(
            base_dir,
            local_fingerprint,
            &current_len,
            open_shards,
            &all_deleted_collections,
        );
        timings.bookkeeping_source = bookkeeping_source;
        if bookkeeping_source == BookkeepingSource::Sidecar {
            if let Some(directory) = read_persisted_shard_collections(base_dir) {
                for record in directory.records {
                    let total = scan_out
                        .collection_disk_bytes
                        .entry(record.pack_id)
                        .or_default()
                        .entry(record.collection_id)
                        .or_default();
                    *total = total.saturating_add(record.disk_bytes);
                }
            }
        } else if writable {
            // No valid sidecar to seed the physical byte totals from. Rebuild
            // them exactly (every appended frame, as `physical_layout`
            // counts) rather than persisting zeros, and make the next sync
            // rewrite the sidecar. This is a recovery path, so one pack scan
            // is acceptable.
            scan_out.sidecar_missing = true;
            scan_out.sidecar_recovery_failed = true;
            if let Ok(physical) = crate::packfile::layout::physical_layout(base_dir) {
                scan_out.collection_disk_bytes = disk_bytes_from_physical(&physical);
                scan_out.sidecar_recovery_failed = false;
            }
        }

        // Bookkeeping (home-shard seeding + shard directory) from the sidecar
        // where that passed its gates, otherwise a slot walk of the — possibly
        // replayed — live indexes.
        let counts_lookup: HashMap<[u8; 16], HashMap<u16, u64>> = match bookkeeping_source {
            BookkeepingSource::Sidecar => sidecar_counts,
            BookkeepingSource::SlotScan => {
                let mut out = HashMap::with_capacity(collection_order.len());
                for collection_id in &collection_order {
                    if let Some(gen) = scan_out.collections.get(collection_id) {
                        out.insert(*collection_id, gen.load().index.slot_counts());
                    }
                }
                out
            }
        };
        for collection_id in &collection_order {
            let Some(counts) = counts_lookup.get(collection_id) else {
                continue;
            };
            // Seed the home shard from the highest shard a slot references —
            // the nearest proxy for the scan path's "shard of the last
            // record", since slots were appended in shard-id order.
            if let Some(&home_shard) = counts.keys().max() {
                shards.set_collection_home(collection_id, home_shard);
            }
            for (&slot, &count) in counts {
                let Some(&pack_id) = slot_to_pack_id.get(&slot) else {
                    continue;
                };
                scan_out
                    .shard_collections
                    .entry(pack_id)
                    .or_default()
                    .insert(*collection_id, count);
                scan_out
                    .collection_shards
                    .entry(*collection_id)
                    .or_default()
                    .insert(pack_id);
            }
        }
        // `SlotScan` derives counts straight from the live indexes; the
        // sidecar's `counts` are consumed above either way.

        timings.bookkeeping = bookkeeping_started.elapsed();

        let delta_state = DeltaLogState {
            base_fingerprint: Some(checkpoint.fingerprint),
            base_generations: live_generations,
            next_order_key: live_order
                .values()
                .copied()
                .max()
                .map_or(0, |last| last.saturating_add(1)),
            base_order: live_order,
            log_bytes: log_bytes_on_disk,
            // What was on disk at open is treated as durable; only what this
            // session appends is owed an fsync.
            log_synced_bytes: log_bytes_on_disk,
            // A visible file does not prove its directory entry survived a
            // power loss (a non-durable batch may have created it), so the
            // first durable batch of this session syncs the parent once.
            log_dir_durable: false,
            delta_seq_high: last_delta_seq,
            log_version: replay_log_version,
            base_pack_ids: checkpoint
                .pack_table
                .iter()
                .map(|&(_, pack_id)| pack_id)
                .collect(),
            ..DeltaLogState::default()
        };

        // The index holds the checkpoint plus the operations of the log that were
        // applied, so it covers whatever those describe as well.
        let covered_lsn = covered_lsn.max(log_coverage.unwrap_or(0));
        scan_out.index_identity_reads = index_identity_reads;
        Some((scan_out, collection_order, delta_state, covered_lsn))
    }
    /// Persist the collection-version table a delta log carries into
    /// `index.versions` before a writable open can delete that log.
    ///
    /// A coverage batch lets the journal be reclaimed past the checkpoint's
    /// own coverage, and the version table it carries is then the only durable
    /// copy of the versions for the reclaimed groups. The open path deletes
    /// delta logs in several places (a log that no longer matches the pack set,
    /// a checkpoint rejected into a rescan), and every one
    /// of them runs before [`Self::enable_shared_journal`] checks that the
    /// baseline covers the reclaimed history. Deleting the log first leaves
    /// nothing for that check to find, and every retry fails the same way.
    /// Moving the table into its own file first makes the baseline independent
    /// of what happens to the log.
    ///
    /// Only a table the reader already trusts is moved (see
    /// [`Self::delta_version_table`]), and it is merged with `max` into any
    /// existing file so neither a newer coverage nor a collection that only the
    /// file knows is lost. A no-op when there is no such table or the file
    /// already covers it.
    ///
    /// # Errors
    /// Returns the write error. The caller must refuse the open rather than
    /// continue: the log may be the only copy of those versions.
    pub(super) fn promote_delta_versions(base_dir: &std::path::Path) -> std::io::Result<()> {
        let Some(table) = Self::delta_version_table(base_dir) else {
            return Ok(());
        };
        if table.covered_lsn == 0 {
            return Ok(());
        }
        let versions_path = Self::index_versions_path(base_dir);
        // A missing file means this is the first promotion. A file that exists
        // but cannot be read or decoded is not an empty baseline: writing the
        // older delta table over it would destroy whatever versions it held.
        let existing = crate::index::versions::read_versions_strict(&versions_path)?;
        if existing
            .as_ref()
            .is_some_and(|snapshot| snapshot.covered_lsn >= table.covered_lsn)
        {
            // A previous attempt may have renamed the file and then failed its
            // directory sync; make that rename durable before the caller
            // reclaims the log.
            if let Some(dir) = versions_path.parent() {
                crate::shard::sync_directory(dir)?;
            }
            return Ok(());
        }
        let mut merged: std::collections::BTreeMap<[u8; 16], u64> = existing
            .map(|snapshot| snapshot.versions.into_iter().collect())
            .unwrap_or_default();
        for (collection, version) in table.entries {
            let entry = merged.entry(collection).or_insert(0);
            *entry = (*entry).max(version);
        }
        let merged: Vec<([u8; 16], u64)> = merged.into_iter().collect();
        crate::index::versions::write_versions(&versions_path, &merged, table.covered_lsn)
    }

    /// What the disk records of the durable index state, in one pass over the
    /// checkpoint and its delta log: the coverage the checkpoint records or a
    /// committed batch of the log continuing it claims (whichever is newer; 0
    /// when there is neither), and the fingerprint of the durable pack state
    /// (0 when unknown). The log only counts if it names that checkpoint's
    /// fingerprint; a stale epoch left by a crash is inert.
    ///
    /// Reading the checkpoint's own record matters: a new checkpoint retires
    /// the delta epoch that held the newest claim, and `journal.lsn` is written
    /// only after the checkpoint is durable, so for a moment the checkpoint's
    /// header is the only place that coverage is written down.
    pub(super) fn durable_index_state(base_dir: &std::path::Path) -> DurableIndexState {
        // The versions file is the coverage record that survives checkpoint
        // retirement, so a crash between its fsync and `journal.lsn` must still
        // recover the coverage it names.
        let versions_coverage =
            crate::index::versions::read_coverage(&Self::index_versions_path(base_dir))
                .unwrap_or(0);
        let Ok(Some(summary)) = crate::index::checkpoint::read_checkpoint_summary(
            &Self::index_checkpoint_path(base_dir),
        ) else {
            return DurableIndexState {
                covered_lsn: versions_coverage,
                fingerprint: 0,
            };
        };
        // One pass over the delta log yields both what it claims and where it
        // ends; the log is only trusted if it names this checkpoint.
        let tail =
            delta::read_delta_tail_fingerprint(&Self::delta_path(base_dir, summary.fingerprint));
        let (log_coverage, fingerprint) = match tail {
            Ok(Some(tail)) if tail.base_fingerprint == summary.fingerprint => {
                (tail.coverage.unwrap_or(0), tail.tail_fingerprint)
            }
            Ok(Some(_) | None) => (0, summary.fingerprint),
            Err(_) => (0, 0),
        };
        DurableIndexState {
            covered_lsn: summary.covered_lsn.max(log_coverage).max(versions_coverage),
            fingerprint,
        }
    }
}
