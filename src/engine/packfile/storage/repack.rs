use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::csr::Csr;
use crate::packfile::{self, PackId};
use crate::shard::{self, Shard, ShardPool};
use crate::storage::{NodeId, StorageError};

use super::{
    AdjacencyResult, DagWalk, PackfileStorage, RepackIncrementalState, RepackOffsets, RepackPlan,
    RepackRecordMap, RepackScanResult, WalkLimits,
};

impl PackfileStorage {
    /// Builds this repack's deduped `hash → (slot, offset)` map,
    /// incrementally where possible, and returns the previous
    /// [`RepackIncrementalState`] (moved out, not cloned) so the caller can
    /// reuse its `adjacency` cache when deciding which nodes need fresh disk
    /// reads.
    ///
    /// On the incremental path a copy of the previous `live_map` is extended
    /// with only the newly-appended entries from each shard (O(delta) scan
    /// work per repack call rather than O(total)). The first call, or a change
    /// in a previously scanned slot's pack identity, triggers a full scan.
    ///
    /// Returns `(hash_to_shard_offset, Some(prev))` on the incremental path
    /// and `(hash_to_shard_offset, None)` on the cold-start path. The caller
    /// must pass `prev` to the adjacency helpers to unlock the incremental
    /// BFS optimization; when it is `None` the caller uses the full-scan
    /// adjacency helpers instead.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure scanning a shard.
    #[allow(clippy::too_many_lines)]
    pub(super) fn repack_scan_incremental(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<RepackScanResult, StorageError> {
        let generation_floor = self.collection_generation_floor(collection_id);
        // Check for existing state under a read lock, then move it out under a
        // write lock if present — O(1) ownership transfer, no clone.
        let has_prev = self.repack_incremental.read().contains_key(collection_id);
        if has_prev {
            let prev = self
                .repack_incremental
                .write()
                .remove(collection_id)
                .expect("entry was present under read lock and collection mutex is held");

            let shards = self.shards.all_shards();
            let current_pack_ids: HashMap<u16, PackId> = shards
                .iter()
                .map(|(slot, shard)| (*slot, shard.pack_id))
                .collect();
            // Slot IDs are recyclable.  A cursor (and every entry in the
            // carried live_map) is meaningful only for the pack incarnation
            // that produced it.  On any replacement or disappearance, start
            // over rather than mixing old offsets with the new pack.
            let pack_changed = prev
                .scan_offsets
                .iter()
                .any(|(slot, (pack_id, _))| current_pack_ids.get(slot) != Some(pack_id));
            if pack_changed {
                let scanned = self.scan_collection_repack_records(collection_id)?;
                let mut map = HashMap::new();
                let mut removed = HashMap::new();
                for (slot, entries) in scanned {
                    if generation_floor.is_some_and(|floor| {
                        self.shards
                            .get_shard(slot)
                            .is_some_and(|shard| shard.creation_seq < floor)
                    }) {
                        continue;
                    }
                    for (hash, offset, is_removed) in entries {
                        self.update_repack_record_map(
                            &mut map,
                            &mut removed,
                            hash,
                            slot,
                            offset,
                            is_removed,
                        )?;
                    }
                }
                self.finalize_repack_record_maps(&mut map, &mut removed)?;
                return Ok((map, removed, None));
            }

            // Scan only bytes appended since the last repack, merging into
            // the stolen live_map in-place — O(delta) work, not O(total).
            let mut map = prev.live_map.clone();
            let mut removed = HashMap::new();
            for (slot, shard) in shards {
                let start = prev
                    .scan_offsets
                    .get(&slot)
                    .filter(|(pack_id, _)| *pack_id == shard.pack_id)
                    .map_or(0, |(_, offset)| *offset);
                let entries = shard
                    .scan_data_path(|path| {
                        packfile::scan_packfile_from_with_removal(
                            path,
                            start,
                            self.shards.checksum_policy().verifies_reads(),
                        )
                    })
                    .map_err(StorageError::Io)?;
                for entry in entries {
                    if entry.collection_id == *collection_id
                        && generation_floor.is_none_or(|floor| shard.creation_seq >= floor)
                    {
                        self.update_repack_record_map(
                            &mut map,
                            &mut removed,
                            entry.hash,
                            slot,
                            entry.offset,
                            entry.removed,
                        )?;
                    }
                }
            }
            self.finalize_repack_record_maps(&mut map, &mut removed)?;
            Ok((map, removed, Some(prev)))
        } else {
            // First repack: full scan of every shard.
            let scanned = self.scan_collection_repack_records(collection_id)?;
            let mut map = HashMap::new();
            let mut removed = HashMap::new();
            for (slot, entries) in &scanned {
                if generation_floor.is_some_and(|floor| {
                    self.shards
                        .get_shard(*slot)
                        .is_some_and(|shard| shard.creation_seq < floor)
                }) {
                    continue;
                }
                for (hash, offset, is_removed) in entries {
                    self.update_repack_record_map(
                        &mut map,
                        &mut removed,
                        *hash,
                        *slot,
                        *offset,
                        *is_removed,
                    )?;
                }
            }
            self.finalize_repack_record_maps(&mut map, &mut removed)?;
            Ok((map, removed, None))
        }
    }

    /// Keep the newest scanned live frame and removal frame independently.
    pub(super) fn update_repack_record_map(
        &self,
        map: &mut RepackRecordMap,
        removed: &mut RepackRecordMap,
        hash: NodeId,
        slot: u16,
        offset: u64,
        is_removed: bool,
    ) -> Result<(), StorageError> {
        let Some(shard) = self.shards.get_shard(slot) else {
            return Err(StorageError::Corrupt(format!(
                "repack candidate references missing shard {slot}"
            )));
        };
        let target = if is_removed { removed } else { map };
        let replace = if let Some(&(old_slot, old_offset)) = target.get(&hash) {
            let old_shard = self.shards.get_shard(old_slot).ok_or_else(|| {
                StorageError::Corrupt(format!(
                    "repack candidate references missing shard {old_slot}"
                ))
            })?;
            Self::repack_record_is_newer(&shard, offset, &old_shard, old_offset)
        } else {
            true
        };
        if replace {
            target.insert(hash, (slot, offset));
        }
        Ok(())
    }

    /// Whether a frame at `(new_shard, new_offset)` was written no earlier than
    /// one at `(old_shard, old_offset)`: immutable pack creation order, then
    /// offset within the pack. Runtime slots can be recycled and journal LSNs
    /// are absent from unjournaled writes, so neither is consulted.
    pub(super) fn repack_record_is_newer(
        new_shard: &Shard,
        new_offset: u64,
        old_shard: &Shard,
        old_offset: u64,
    ) -> bool {
        (new_shard.creation_seq, new_offset) >= (old_shard.creation_seq, old_offset)
    }

    /// Resolve the latest live and removed candidates independent of shard
    /// slot order. A marker survives only when it is newer than the newest
    /// live frame; otherwise the key was written again after its removal.
    pub(super) fn finalize_repack_record_maps(
        &self,
        live: &mut RepackRecordMap,
        removed: &mut RepackRecordMap,
    ) -> Result<(), StorageError> {
        removed.retain(|hash, _| live.contains_key(hash));
        let hashes: Vec<_> = removed.keys().copied().collect();
        for hash in hashes {
            let (Some(&(live_slot, live_offset)), Some(&(removed_slot, removed_offset))) =
                (live.get(&hash), removed.get(&hash))
            else {
                continue;
            };
            let live_shard = self.shards.get_shard(live_slot).ok_or_else(|| {
                StorageError::Corrupt(format!(
                    "repack candidate references missing shard {live_slot}"
                ))
            })?;
            let removed_shard = self.shards.get_shard(removed_slot).ok_or_else(|| {
                StorageError::Corrupt(format!(
                    "repack candidate references missing shard {removed_slot}"
                ))
            })?;
            if Self::repack_record_is_newer(
                &live_shard,
                live_offset,
                &removed_shard,
                removed_offset,
            ) {
                removed.remove(&hash);
            } else {
                live.remove(&hash);
            }
        }
        Ok(())
    }

    /// Incremental BFS reachability walk.
    ///
    /// Uses the adjacency cache from `prev` to avoid re-reading disk for nodes
    /// that were already known-live after the last repack. Only newly-added
    /// nodes (those absent from `prev.adjacency`) require a disk read.
    ///
    /// **Root-removal invariant (critical):** This function must NOT be called
    /// when the current root set is a strict subset of `prev.prev_roots` (i.e.
    /// any root was removed since the last repack). In that case nodes
    /// exclusively reachable through the removed root would remain in the live
    /// set indefinitely, because `visited` in `prev.adjacency` only grows —
    /// nothing in this incremental path forces a re-check of reachability.
    /// The caller ([`Self::repack_collection_reachable`]) detects root removal
    /// and falls back to [`Self::bfs_live_set`] (cold-start, full BFS) for
    /// that repack, then saves a fresh cursor so subsequent repacks can be
    /// incremental again.
    ///
    /// **Memory:** `prev.adjacency` holds `O(live_nodes` × `avg_fanout`) hashes
    /// (≈ `live_nodes × avg_fanout × 16` bytes) in RAM per collection. This
    /// is the cost of eliminating the O(live) disk reads per repack call —
    /// callers should be aware for large collections.
    pub(super) fn bfs_live_set_incremental(
        pool: &ShardPool,
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        // Seed with roots that are not already in the cached live set.
        // Anything in prev.adjacency was reachable last time and — given no
        // root was removed — is still reachable now; no need to re-expand it.
        let mut visited: HashSet<[u8; 16]> = prev.adjacency.keys().copied().collect();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        // Retain the cached edges for already-live nodes.  Initialising these
        // entries with empty vectors loses prior edges when they are not put
        // back on the queue during an incremental rooted repack.
        let mut adjacency = prev.adjacency.clone();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            // Fast path: edge list already cached — no disk read needed.
            if let Some(edges) = prev.adjacency.get(&hash) {
                for edge in edges {
                    if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                        queue.push_back(*edge);
                    }
                }
                adjacency.insert(hash, edges.clone());
                continue;
            }

            // Slow path: new node, must read from disk.
            let Some(&(slot, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&slot) else {
                continue;
            };
            let record = pool.read_at(old_shard, offset, true)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        // Restrict adjacency to the actual visited set — entries inherited
        // from prev that are no longer in hash_to_shard_offset (e.g. they
        // were in a shard that was later dropped) must not survive.
        adjacency.retain(|h, _| visited.contains(h) && hash_to_shard_offset.contains_key(h));

        let mut live_hashes: Vec<[u8; 16]> = visited
            .into_iter()
            .filter(|h| hash_to_shard_offset.contains_key(h))
            .collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// Incremental full-adjacency scan (no-roots / preserve-everything path).
    ///
    /// For each hash in `hash_to_shard_offset`, reuses the cached edge list
    /// from `prev.adjacency` when available — avoiding a disk read for any
    /// node that survived the last repack. Only genuinely new hashes (not in
    /// the cache) require a disk read.
    ///
    /// All hashes in `hash_to_shard_offset` are live (no GC in this path),
    /// so the result is always the full map.
    ///
    /// **Memory:** same as [`Self::bfs_live_set_incremental`] — see that doc.
    pub(super) fn scan_full_adjacency_incremental(
        pool: &ShardPool,
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> =
            HashMap::with_capacity(hash_to_shard_offset.len());

        for (hash, &(slot, offset)) in hash_to_shard_offset {
            // Fast path: node was live last time, edge list is cached.
            if let Some(edges) = prev.adjacency.get(hash) {
                adjacency.insert(*hash, edges.clone());
                continue;
            }
            // Slow path: new node, must read from disk.
            if let Some(shard) = pinned.get(&slot) {
                let record = pool.read_at(shard, offset, true)?;
                let edges = extract_edges(hash, &record.data);
                adjacency.insert(*hash, edges);
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Saves the cursor state a future call to [`Self::repack_scan_incremental`]
    /// needs: each shard's current byte length (the next repack's scan start
    /// point), the deduped live map restricted to hashes that survived into
    /// `new_offsets`, the adjacency cache for those survivors (so the next
    /// repack's BFS/scan can skip re-reading disk for already-seen nodes), and
    /// the roots snapshot (so the next repack can detect whether any root was
    /// removed and fall back to a full BFS sweep if so).
    ///
    /// Anything this repack dropped must not resurface via the carried-forward
    /// `live_map` or adjacency cache next time.
    pub(super) fn repack_save_incremental_state(
        &self,
        collection_id: &[u8; 16],
        new_offsets: &[([u8; 16], u16, u64, bool)],
        adjacency: HashMap<[u8; 16], Vec<[u8; 16]>>,
        current_roots: &[[u8; 16]],
    ) {
        let surviving: HashSet<[u8; 16]> = new_offsets.iter().map(|&(h, _, _, _)| h).collect();

        let mut scan_offsets = HashMap::new();
        for (slot, shard) in self.shards.all_shards() {
            scan_offsets.insert(slot, (shard.pack_id, shard.file_len()));
        }
        let next_live_map: HashMap<[u8; 16], (u16, u64)> = new_offsets
            .iter()
            .map(|&(hash, slot, offset, _)| (hash, (slot, offset)))
            .collect();
        // Evict dropped nodes from the adjacency cache so they can't
        // re-enter the live set on a future incremental BFS pass.
        let next_adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = adjacency
            .into_iter()
            .filter(|(h, _)| surviving.contains(h))
            .collect();
        self.repack_incremental.write().insert(
            *collection_id,
            RepackIncrementalState {
                scan_offsets,
                live_map: next_live_map,
                adjacency: next_adjacency,
                prev_roots: current_roots.to_vec(),
            },
        );
    }

    /// Non-mutating preview of what [`Self::repack_collection_reachable`] would
    /// do for `collection_id`: runs the exact same scan + reachability
    /// computation (so kept/dropped counts are exact, not estimates), but
    /// performs no writes, no index swap, and no incremental-repack
    /// cursor update. Used to preview a repack's effect before committing
    /// to it — e.g. a shard-compaction preflight that needs "how many
    /// bytes would survive, and which shards do they currently live in"
    /// without actually rewriting anything.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    pub fn plan_collection_repack(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        // `repack_scan_incremental` moves the cursor out to avoid cloning on
        // real repacks.  A plan must leave that state untouched, and must not
        // race a real repack while temporarily doing so.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        // Make buffered frames visible to the packfile scan so the plan
        // reflects records that have actually been put (not yet flushed).
        self.shards.flush_all()?;
        let saved_state = self.repack_incremental.read().get(collection_id).cloned();
        let result = self
            .repack_scan_incremental(collection_id)
            .and_then(|(map, _, _)| {
                self.plan_collection_repack_from_map(collection_id, &map, &extract_edges)
            });
        let mut states = self.repack_incremental.write();
        if let Some(state) = saved_state {
            states.insert(*collection_id, state);
        } else {
            states.remove(collection_id);
        }
        result
    }

    pub(super) fn plan_collection_repack_from_map(
        &self,
        collection_id: &[u8; 16],
        hash_to_shard_offset: &RepackRecordMap,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, _adjacency) = match roots {
            Some(roots) if !roots.is_empty() => Self::bfs_live_set(
                &self.shards,
                &roots,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
            )?,
            _ => Self::scan_full_adjacency(
                &self.shards,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
            )?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
        let mut kept_bytes: u64 = 0;
        let mut dropped_bytes: u64 = 0;
        let mut shards_touched: HashSet<u16> = HashSet::new();
        for (hash, &(slot, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&slot) {
                // Actual on-disk bytes, not the uncompressed upper bound —
                // a repack preflight should report what will really be
                // reclaimed/kept, which is smaller than plaintext size for
                // any frame that compressed.
                if let Ok(bytes) = Self::record_disk_len_at(shard, offset) {
                    if live_set.contains(hash) {
                        shards_touched.insert(slot);
                        kept_bytes = kept_bytes.saturating_add(bytes);
                    } else {
                        dropped_bytes = dropped_bytes.saturating_add(bytes);
                    }
                }
            }
        }

        let mut shards_touched: Vec<u16> = shards_touched.into_iter().collect();
        shards_touched.sort_unstable();
        Ok(RepackPlan {
            kept: live_hashes.len(),
            dropped,
            kept_bytes,
            dropped_bytes,
            shards_touched,
        })
    }

    /// Plan several collection repacks from one physical scan of their shard
    /// closure. Unlike repeatedly calling [`Self::plan_collection_repack`], this
    /// is O(bytes scanned) rather than O(collections × bytes scanned).
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption while scanning or
    /// resolving a requested collection's records.
    pub fn plan_collections_repack(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<RepackPlan>, StorageError> {
        // Planning reads packfiles directly; flush first so buffered records
        // are part of the plan.
        self.shards.flush_all()?;
        let maps = self.scan_collection_record_maps(collection_ids)?;
        collection_ids
            .iter()
            .map(|collection_id| {
                let (map, _) = maps.get(collection_id).ok_or_else(|| {
                    StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
                })?;
                self.plan_collection_repack_from_map(collection_id, map, &extract_edges)
            })
            .collect()
    }

    /// Finds every collection and shard transitively reachable from
    /// `start_shard` via the collection↔shard reference relation: collections
    /// referencing this shard, the other shards those collections reference,
    /// the collections referencing *those* shards, and so on until nothing new
    /// is found. In practice this converges immediately in almost every
    /// case — a collection typically has a live footprint on only one or two
    /// shards (sticky home routing) — but the closure has to be computed
    /// rather than assumed, since a collection-scoped repack can't correctly
    /// judge liveness by looking at only one shard's slice of that collection's
    /// data (see [`Self::collections_referencing_shard`]'s doc).
    ///
    /// This is the basis for a shard-compaction preflight: you can't
    /// truthfully say "this touches N collections across K shards" without
    /// first finding the full closure, not just the one shard the
    /// operation was originally pointed at.
    ///
    /// # Errors
    /// Returns `StorageError` if `start_shard` isn't an open shard.
    pub fn repack_closure(
        &self,
        start_shard: u16,
    ) -> Result<(Vec<[u8; 16]>, Vec<u16>), StorageError> {
        let mut collections: HashSet<[u8; 16]> = HashSet::new();
        let mut shards: HashSet<u16> = HashSet::new();
        let mut shard_queue = vec![start_shard];
        let mut collection_queue: Vec<[u8; 16]> = Vec::new();

        loop {
            if let Some(slot) = shard_queue.pop() {
                if shards.insert(slot) {
                    for collection_id in self.collections_referencing_shard(slot)? {
                        if collections.insert(collection_id) {
                            collection_queue.push(collection_id);
                        }
                    }
                }
            } else if let Some(collection_id) = collection_queue.pop() {
                for slot in self.collection_referenced_shards(&collection_id) {
                    if !shards.contains(&slot) {
                        shard_queue.push(slot);
                    }
                }
            } else {
                break;
            }
        }

        let mut collections: Vec<[u8; 16]> = collections.into_iter().collect();
        collections.sort_unstable();
        let mut shards: Vec<u16> = shards.into_iter().collect();
        shards.sort_unstable();
        Ok((collections, shards))
    }

    /// Every shard id `collection_id`'s current index has at least one live
    /// entry pointing into. Empty if the collection doesn't exist.
    #[must_use]
    pub fn collection_referenced_shards(&self, collection_id: &[u8; 16]) -> Vec<u16> {
        let Some(gen) = self.generation(collection_id) else {
            return Vec::new();
        };
        gen.index
            .referenced_slot_ids()
            .iter()
            .enumerate()
            .filter_map(|(i, &referenced)| referenced.then(|| u16::try_from(i).ok()).flatten())
            .collect()
    }

    /// Like `collection_referenced_shards`, but returns the `pack_id`
    /// for each shard rather than the runtime slot index.
    pub fn collection_referenced_pack_ids(&self, collection_id: &[u8; 16]) -> Vec<PackId> {
        self.collection_referenced_shards(collection_id)
            .iter()
            .filter_map(|&slot| self.shards.get_shard(slot).map(|s| s.pack_id))
            .collect()
    }

    /// Current indexed-node count for every shard.
    ///
    /// Unlike a physical packfile scan, these counts include only index
    /// entries that are presently live and resolve through a collection's current
    /// generation. Rewritten or superseded append entries are excluded.
    #[must_use]
    pub fn shard_node_counts(&self) -> HashMap<PackId, u64> {
        self.index_tables
            .read()
            .shard_collections
            .iter()
            .map(|(&slot, collections)| {
                let count = collections
                    .values()
                    .copied()
                    .fold(0u64, u64::saturating_add);
                (slot, count)
            })
            .collect()
    }

    /// Rewrite a collection's records in topological order, optionally performing
    /// garbage collection.
    ///
    /// If live roots are configured for this collection (see
    /// [`Self::set_live_roots`]), traverses outward from them via
    /// `extract_edges` and drops anything not reached: only reachable
    /// records are read from disk during the traversal, so unreachable
    /// data is never even fetched, not just excluded from the output.
    ///
    /// If no live roots are configured (`set_live_roots` was never called,
    /// or was called with an empty list), nothing is known to be garbage,
    /// so every scanned record is preserved, still deduplicated and
    /// rewritten in topological order — rather than risk deleting live
    /// data on the assumption that "no roots" means "nothing is live".
    ///
    /// All shards this call will read from are pinned (held via an
    /// `Arc<Shard>` for the whole call) before any writes happen, so a
    /// rotation triggered by this call's own writes can never retire a
    /// shard this
    /// call still needs to read stale data from.
    ///
    /// Returns `(kept, dropped)`: the number of records written to the new
    /// generation, and the number of scanned records found unreachable and
    /// discarded.
    ///
    /// Shard writes are synced before success. Checkpoint and inspection
    /// sidecar persistence are best-effort.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption, including a cycle in the
    /// live graph.
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    // Four dispatch arms (roots/no-roots × incremental/cold-start) plus root-removal
    // detection make this function long; splitting it would just move the complexity
    // without reducing it.
    #[allow(clippy::too_many_lines)]
    pub fn repack_collection_reachable(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<(usize, usize), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let collection_guard = collection_arc.lock();

        // A deleted collection's old frames remain in packs until garbage
        // collection. Never let a repack scan those historical frames back
        // into the live index.
        if self
            .load_deleted_collections_cached()?
            .contains(collection_id)
        {
            return Ok((0, 0));
        }

        // The scan below reads packfiles directly, so any records still
        // sitting in a shard's append buffer would be invisible to it and
        // get silently dropped by this repack. Flush first so the whole
        // collection's committed bytes are scan-visible. Holding this
        // collection's put mutex means nothing can buffer behind it for this
        // collection between the flush and the scan.
        self.shards.flush_all()?;

        let (hash_to_shard_offset, removed_markers, prev_state) =
            self.repack_scan_incremental(collection_id)?;

        // Pin every shard this call could possibly read from before any
        // writes happen — see pin_shards' doc for why this must come
        // first.
        let pinned = self.pin_shards(
            hash_to_shard_offset
                .values()
                .chain(removed_markers.values())
                .map(|&(id, _)| id),
        );

        let roots = self.live_roots.read().get(collection_id).cloned();

        // Root-removal detection: if any root present in the previous repack's
        // snapshot is absent from the current root set, nodes exclusively
        // reachable through that removed root may be stuck in the cached live
        // set indefinitely (the incremental adjacency cache only grows — it has
        // no mechanism to re-derive reachability). Fall back to a full cold-start
        // BFS for this repack so those nodes can actually be collected. Once the
        // full sweep completes, the saved cursor is fresh and subsequent repacks
        // can be incremental again.
        let roots_shrunk = prev_state.as_ref().is_some_and(|prev| {
            let current_root_set: HashSet<[u8; 16]> =
                roots.as_deref().unwrap_or(&[]).iter().copied().collect();
            prev.prev_roots
                .iter()
                .any(|r| !current_root_set.contains(r))
        });
        // Use incremental adjacency only when prev state exists AND no root was removed.
        let use_incremental = prev_state.is_some() && !roots_shrunk;

        let (live_hashes, adjacency) = match (roots.as_deref(), use_incremental) {
            (Some(roots), true) if !roots.is_empty() => {
                // Incremental BFS: only expand from roots not already in the
                // cached live set; reuse cached edge lists for everything else.
                Self::bfs_live_set_incremental(
                    &self.shards,
                    roots,
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            (Some(roots), false) if !roots.is_empty() => {
                // Cold-start BFS: root set shrank or first repack — full walk.
                Self::bfs_live_set(
                    &self.shards,
                    roots,
                    &hash_to_shard_offset,
                    &pinned,
                    &extract_edges,
                )?
            }
            (_, true) => {
                // No live roots, incremental path: only read disk for new nodes.
                Self::scan_full_adjacency_incremental(
                    &self.shards,
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            _ => {
                // No live roots, cold-start: preserve everything, read all.
                Self::scan_full_adjacency(
                    &self.shards,
                    &hash_to_shard_offset,
                    &pinned,
                    &extract_edges,
                )?
            }
        };

        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        // Evict garbage-collected hashes from the collection's cache. Repack
        // carries the same cache forward into the new generation (below);
        // without this, a stale cache hit could still serve bytes for a
        // hash this pass just decided is unreachable, defeating GC.
        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        // Dependencies first, level by level, ties broken by node id.
        let (topo, cyclic) = csr.layered_or_id_order();
        if cyclic {
            eprintln!(
                "mtxdb: repack: dependency cycle among {} records in collection {}; writing them in id order",
                live_hashes.len(),
                crate::storage::hex16(collection_id),
            );
        }

        // Copy into a non-source destination shard, never back into the
        // collection's current generation. Once the new index is published below,
        // this lets old source shards retire when no other collection references
        // them.
        let source_shards: HashSet<u16> = hash_to_shard_offset
            .values()
            .chain(removed_markers.values())
            .map(|&(slot, _)| slot)
            .collect();
        self.shards
            .prepare_collection_repack(collection_id, &source_shards)?;

        let mut new_offsets: RepackOffsets = Vec::with_capacity(topo.len());
        let mut new_disk_bytes = HashMap::new();
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_slot, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, &pinned, old_slot, old_offset)?
            {
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
                new_offsets.push((hash, slot, offset, false));
            }
        }

        let kept = new_offsets.len();

        // Preserve a marker while a previous live version may remain in a
        // shard retained by another collection. On the next repack, the full
        // scan can drop it once that older copy has gone away.
        let mut removed_ids: Vec<_> = removed_markers.keys().copied().collect();
        removed_ids.sort_unstable();
        for hash in removed_ids {
            let (old_slot, old_offset) = removed_markers[&hash];
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, &pinned, old_slot, old_offset)?
            {
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
                new_offsets.push((hash, slot, offset, true));
            }
        }

        let index = Self::build_index(&new_offsets, self.index_config)?;
        // Repack writes additional physical frames. Keep accounting for old
        // frames in packs that remain live; retirement prunes contributions
        // for packs that are actually removed.
        self.add_collection_disk_bytes(collection_id, &new_disk_bytes);
        self.replace_collection_shard_counts(
            collection_id,
            &self.slot_counts_to_pack_id_counts(&index.slot_counts()),
        );
        // The repack index is rebuilt from current frames and is therefore
        // complete.
        let cache = self.generation(collection_id).map(|g| g.cache.clone());
        self.store_generation(collection_id, index, cache, true)?;
        self.retire_empty_shards(collection_id);

        self.repack_count.fetch_add(1, Ordering::Relaxed);
        self.repack_kept_total
            .fetch_add(kept as u64, Ordering::Relaxed);
        self.repack_dropped_total
            .fetch_add(dropped as u64, Ordering::Relaxed);
        self.repack_counts_by_collection
            .write()
            .entry(*collection_id)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);

        let current_roots: Vec<[u8; 16]> = roots.as_deref().unwrap_or(&[]).to_vec();
        if removed_markers.is_empty() {
            self.repack_save_incremental_state(
                collection_id,
                &new_offsets,
                adjacency,
                &current_roots,
            );
        } else {
            // The carried tombstone must be observed in a full shard scan next
            // time so it can be retired once no older copy remains.
            self.repack_incremental.write().remove(collection_id);
        }

        // Fsync the shards this repack just wrote into and persist the
        // updated stats snapshot as part of finishing the repack, rather
        // than leaving both to whatever the next unrelated sync_dirty()
        // call happens to be — a crash right after a repack should not
        // lose durability for data this repack itself just wrote, nor
        // leave the persisted stats stale relative to what's on disk.
        self.shards.sync_dirty()?;

        // Release this collection's put_mutex before persisting the
        // checkpoint: persist_index_checkpoint acquires every collection's
        // put_mutex itself (parking_lot's Mutex is non-reentrant), and this
        // repack still holds this collection's.
        drop(collection_guard);
        // retire_empty_shards may have unlinked packs this collection's old
        // index entries pointed at. A reader that reloads from the
        // checkpoint after this point must see the new shard layout, not
        // the stale one — otherwise reload_index_from_checkpoint rebuilds a
        // fingerprint that can never match a checkpoint naming retired
        // packs, and the read-journal reload path fails closed. Mark the
        // checkpoint dirty (repack itself never does, since it never runs
        // through the put/sync paths that do) and persist it now so the
        // on-disk checkpoint matches what this repack just wrote.
        //
        // Best-effort: the repack itself already succeeded and its shard
        // writes are durable (fsynced above). A checkpoint-write failure
        // here shouldn't be reported as a repack failure — the packfiles
        // stay authoritative and the dirty flag (left set on failure, see
        // `persist_index_checkpoint`) means the next sync retries it.
        let _persist_guard = self.index_persist_lock.lock();
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.persist_index_checkpoint_best_effort();
        self.persist_shard_collections_best_effort();

        Ok((kept, dropped))
    }

    /// Repack several collections as one physical shard-compaction batch.
    ///
    /// The input is scanned once, then all replacement records are written
    /// through one shared destination stream. This is O(bytes + nodes), not
    /// O(collections × bytes), and avoids leaving one partially-filled destination
    /// shard behind for each collection.
    ///
    /// # Errors
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        self.repack_collections_reachable_with_progress(
            collection_ids,
            extract_edges,
            |_collection, _from, _to, _nodes, _complete| {},
        )
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// This lets an interactive caller show meaningful physical-compaction
    /// progress without turning the shared shard batch back into a
    /// collection-at-a-time operation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub(super) fn report_repack_output_rotation(
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        active_output_shard: &mut u16,
        current_active_shard: u16,
        collection_id: &[u8; 16],
        copied_nodes: usize,
    ) {
        if current_active_shard != *active_output_shard {
            output_progress(
                Some(*collection_id),
                *active_output_shard,
                current_active_shard,
                copied_nodes,
                false,
            );
            *active_output_shard = current_active_shard;
        }
    }

    pub(super) fn copy_repack_batch_collection(
        &self,
        collection_id: &[u8; 16],
        record_maps: &(RepackRecordMap, RepackRecordMap),
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        output_state: &mut (u16, usize),
    ) -> Result<(RepackOffsets, usize, HashMap<PackId, u64>), StorageError> {
        let (hash_to_shard_offset, removed_markers) = record_maps;
        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, adjacency) = match roots {
            Some(roots) if !roots.is_empty() => Self::bfs_live_set(
                &self.shards,
                &roots,
                hash_to_shard_offset,
                pinned,
                extract_edges,
            )?,
            _ => Self::scan_full_adjacency(
                &self.shards,
                hash_to_shard_offset,
                pinned,
                extract_edges,
            )?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        // Dependencies first, level by level, ties broken by node id.
        let (topo, cyclic) = csr.layered_or_id_order();
        if cyclic {
            eprintln!(
                "mtxdb: repack: dependency cycle among {} records in collection {}; writing them in id order",
                live_hashes.len(),
                crate::storage::hex16(collection_id),
            );
        }

        let mut new_offsets = Vec::with_capacity(topo.len().saturating_add(removed_markers.len()));
        let mut new_disk_bytes = HashMap::new();
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_slot, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, pinned, old_slot, old_offset)?
            {
                output_state.1 = output_state.1.saturating_add(1);
                let current_active_shard = self.shards.active_shard().slot;
                Self::report_repack_output_rotation(
                    output_progress,
                    &mut output_state.0,
                    current_active_shard,
                    collection_id,
                    output_state.1,
                );
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
                new_offsets.push((hash, slot, offset, false));
            }
        }

        let mut removed_ids: Vec<_> = removed_markers.keys().copied().collect();
        removed_ids.sort_unstable();
        for hash in removed_ids {
            let (old_slot, old_offset) = removed_markers[&hash];
            if let Some((hash, slot, offset, disk_bytes)) =
                self.copy_record_to_shard(collection_id, pinned, old_slot, old_offset)?
            {
                output_state.1 = output_state.1.saturating_add(1);
                let current_active_shard = self.shards.active_shard().slot;
                Self::report_repack_output_rotation(
                    output_progress,
                    &mut output_state.0,
                    current_active_shard,
                    collection_id,
                    output_state.1,
                );
                if let Some(shard) = self.shards.get_shard(slot) {
                    let total = new_disk_bytes.entry(shard.pack_id).or_insert(0_u64);
                    *total = (*total).saturating_add(disk_bytes);
                }
                new_offsets.push((hash, slot, offset, true));
            }
        }

        Ok((new_offsets, dropped, new_disk_bytes))
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable_with_progress(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        mut output_progress: impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let mut collection_ids = collection_ids.to_vec();
        collection_ids.sort_unstable();
        collection_ids.dedup();
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        // A batch is one coherent generation change. Hold every selected
        // collection's writer mutex in a stable order so a concurrent put cannot
        // be missed by the scan or revive an old source after the swap.
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let collection_guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();

        // A deleted collection's frames are historical. Repacking them would
        // copy them into a new pack past the tombstone, and a later rebuild
        // would resurrect records the delete removed. Drop those IDs only after
        // the put mutexes are held, so a concurrent delete cannot slip between
        // this check and the scan.
        collection_ids.retain(|id| !self.is_deleted_collection(id));
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Same scan visibility requirement as the single-collection repack:
        // every selected collection's buffered frames must be on disk before
        // `scan_collection_record_maps` reads packfiles, or those records
        // would be dropped as if never written.
        self.shards.flush_all()?;

        let maps = self.scan_collection_record_maps(&collection_ids)?;
        let source_shards: HashSet<u16> = maps
            .values()
            .flat_map(|(live, removed)| {
                live.values().chain(removed.values()).map(|&(slot, _)| slot)
            })
            .collect();
        let pinned = self.pin_shards(source_shards.iter().copied());

        // Establish one common destination before copying the first frame.
        // `put_record` moves every collection to the same active successor when a
        // destination fills, so output remains densely packed across collections.
        self.shards
            .prepare_collections_repack(&collection_ids, &source_shards)?;

        let mut results = Vec::with_capacity(collection_ids.len());
        let mut output_state = (self.shards.active_shard().slot, 0usize);
        for collection_id in &collection_ids {
            let record_maps = maps.get(collection_id).ok_or_else(|| {
                StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
            })?;
            let (_, removed_markers) = record_maps;
            let (new_offsets, dropped, new_disk_bytes) = self.copy_repack_batch_collection(
                collection_id,
                record_maps,
                &pinned,
                &extract_edges,
                &mut output_progress,
                &mut output_state,
            )?;
            let kept = new_offsets.len();
            let index = Self::build_index(&new_offsets, self.index_config)?;
            self.add_collection_disk_bytes(collection_id, &new_disk_bytes);
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&index.slot_counts()),
            );
            // The repack index is rebuilt from current frames and is therefore
            // complete.
            let cache = self.generation(collection_id).map(|g| g.cache.clone());
            self.store_generation(collection_id, index, cache, true)?;
            // The batch path does a full cold-start scan (scan_collection_record_maps),
            // so we have no incremental adjacency to carry forward. Pass an empty
            // map so the next single-collection repack starts with a fresh cache
            // (it will populate it from the full BFS on that first pass).
            let batch_roots: Vec<[u8; 16]> = self
                .live_roots
                .read()
                .get(collection_id)
                .cloned()
                .unwrap_or_default();
            if removed_markers.is_empty() {
                self.repack_save_incremental_state(
                    collection_id,
                    &new_offsets,
                    HashMap::new(),
                    &batch_roots,
                );
            } else {
                self.repack_incremental.write().remove(collection_id);
            }

            self.repack_count.fetch_add(1, Ordering::Relaxed);
            self.repack_kept_total
                .fetch_add(kept as u64, Ordering::Relaxed);
            self.repack_dropped_total
                .fetch_add(dropped as u64, Ordering::Relaxed);
            self.repack_counts_by_collection
                .write()
                .entry(*collection_id)
                .and_modify(|count| *count = count.saturating_add(1))
                .or_insert(1);
            results.push((*collection_id, kept, dropped));
        }

        output_progress(None, output_state.0, output_state.0, output_state.1, true);

        // All target collection locks remain held while source references are
        // replaced. Drop them before the general retirement protocol, which
        // obtains every collection lock itself.
        drop(collection_guards);
        self.retire_empty_shards_after_batch();
        self.shards.sync_dirty()?;

        // See the matching comment in repack_collection_reachable: retirement
        // above may have unlinked packs this batch's old index entries
        // pointed at, so the on-disk checkpoint must be refreshed or a
        // reader's checkpoint reload can never match the live shard set.
        // Best-effort for the same reason: the batch's shard writes are
        // already durable, so a checkpoint-write failure here costs the
        // next sync a retry, not correctness.
        let _persist_guard = self.index_persist_lock.lock();
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        self.persist_index_checkpoint_best_effort();
        self.persist_shard_collections_best_effort();

        Ok(results)
    }

    /// Fetches the ancestors of `frontier`, walking `extract_edges`
    /// backward and stopping expansion at (and excluding) anything in
    /// `stop_at` — a bounded ancestor walk, e.g. for edges or
    /// prev-event traversal from a caller-supplied frontier back to
    /// caller-supplied already-known boundary nodes.
    ///
    /// This is **not** a from/to range or span: if a branch's history
    /// never crosses any `stop_at` node (a fork off the main line, say),
    /// that branch is walked all the way back to the collection's true roots,
    /// not truncated at an implied boundary. A genuine "everything between
    /// these two frontiers" query is `ancestors(frontier) ∩
    /// descendants(stop_at)`, which needs reverse adjacency this doesn't
    /// build — `NodeId`s are content hashes with no key order to slice a
    /// real range query over, and this walk only covers the ancestor half.
    /// `limits.max_nodes` bounds the runaway case above.
    ///
    /// Returns nodes lazily, ancestor-first (parents before children,
    /// topologically), as a [`DagWalk`] iterator over data already
    /// resolved during the walk itself (no second per-node fetch against
    /// whatever generation happens to be live when the iterator is
    /// drained) — every record was matched by content hash, not just
    /// index tag, so an index-tag collision can't surface the wrong one.
    ///
    /// Read-only: resolves against one frozen generation snapshot via its
    /// internal breadth-first traversal and takes no `put_mutex`. Cost is
    /// proportional to the number of ancestors actually walked — one
    /// index lookup per node — not collection size, since it doesn't pre-scan
    /// every shard the way `repack_collection_reachable` does.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption, or if the walk finds a
    /// cycle (an event graph must be acyclic; a cycle means the extracted
    /// edges are wrong or the data is corrupt).
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    pub fn walk_ancestors(
        &self,
        collection_id: &[u8; 16],
        frontier: &[[u8; 16]],
        stop_at: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        limits: WalkLimits,
    ) -> Result<DagWalk, StorageError> {
        let Some(gen) = self.generation(collection_id).map(|g| Arc::clone(&g)) else {
            return Ok(DagWalk {
                order: Vec::new().into_iter(),
                resolved: HashMap::new(),
            });
        };

        let ((live_hashes, adjacency), resolved) =
            self.bfs_ancestors(&gen, frontier, stop_at, &extract_edges, limits)?;

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let mut topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "walk_ancestors: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        // `extract_edges` points each node at its parents, so Kahn's
        // algorithm (which surfaces zero-incoming nodes first) naturally
        // yields children before parents. Reverse it: callers of an
        // ancestor walk want parents-before-children, the order data
        // could actually be replayed in.
        topo.reverse();

        let order: Vec<[u8; 16]> = topo
            .into_iter()
            .map(|local| {
                *csr.hash_of(local)
                    .expect("topo order contains valid local IDs")
            })
            .collect();

        Ok(DagWalk {
            order: order.into_iter(),
            resolved,
        })
    }

    /// After a repack, scan all collections' indexes and retire any shard that
    /// no collection references.
    ///
    /// Acquires every collection's `put_mutex` (in sorted order, skipping the
    /// caller's already-held lock) to prevent a concurrent `put()` from
    /// landing a write in a shard whose index entry hasn't been inserted
    /// yet — without that, the scan could see zero references to a shard
    /// that a writer just committed bytes to but hasn't index-updated yet,
    /// causing a live shard to be retired under it.
    pub(super) fn retire_empty_shards(&self, held_collection: &[u8; 16]) {
        // Collect collection IDs in sorted order for deadlock-free lock acquisition.
        // Skip the collection whose put_mutex the caller already holds.
        let mut collections_to_lock: Vec<[u8; 16]> = self
            .collections_read()
            .keys()
            .filter(|id| *id != held_collection)
            .copied()
            .collect();
        collections_to_lock.sort_unstable();

        // Acquire all other collections' put_mutexes. The sorted order prevents
        // deadlocks; the held collection is skipped (parking_lot is non-reentrant).
        let mutexes: Vec<_> = collections_to_lock
            .iter()
            .map(|id| self.put_mutex(id))
            .collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();

        self.retire_empty_shards_locked();
    }

    /// Retire unreferenced shards after a batch has released all collection locks.
    /// This takes every collection lock itself, unlike [`Self::retire_empty_shards`]
    /// which is called while one particular collection lock is already held.
    pub(super) fn retire_empty_shards_after_batch(&self) {
        let mut collection_ids: Vec<[u8; 16]> = self.collections_read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();
        self.retire_empty_shards_locked();
    }

    /// Every collection writer lock is held by the caller.
    pub(super) fn retire_empty_shards_locked(&self) {
        // Do not retain the map guard while acquiring collection locks: another
        // operation may need the map lock while it holds a collection lock.
        let collections = self.collections_read();

        // Build the union of shard IDs referenced across all collections.
        let mut referenced = [false; shard::MAX_SHARDS];
        for gen_swap in collections.values() {
            let gen = gen_swap.load();
            let ids = gen.index.referenced_slot_ids();
            for (i, &has_refs) in ids.iter().enumerate() {
                referenced[i] |= has_refs;
            }
        }

        // Retire any shard not in the union.
        for id in 0..u16::try_from(shard::MAX_SHARDS).unwrap() {
            if !referenced[id as usize] {
                self.shards.retire_slot(id);
            }
        }
        drop(collections);
        let mut tables = self.index_tables.write();
        let live_pack_ids: HashSet<PackId> = tables
            .collection_shards
            .values()
            .flat_map(|packs| packs.iter().copied())
            .collect();
        tables
            .collection_disk_bytes
            .retain(|pack_id, _| live_pack_ids.contains(pack_id));
    }
}
