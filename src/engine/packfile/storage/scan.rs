//! Consistent collection scans over pinned packfiles and the journal overlay.
//!
//! These are the multi-shard read paths used by point lookups, collection
//! iteration, and the shard-compaction preflight.

#[cfg(test)]
use super::TestScannedRecord;
use super::{
    packfile, Arc, CollectionGeneration, CollectionScan, CollectionScanPair, HashMap, HashSet,
    JournalCoordinator, JournalCursor, JournalReplayLease, MutexGuard, NodeData, NodeId, Ordering,
    PackfileStorage, PreparedCollectionScan, ScanWork, ScannedRepackShard, Shard, StorageError,
};

impl PackfileStorage {
    /// Stream every live record of one collection from a single consistent
    /// snapshot.
    ///
    /// The scan captures a generation and pack-file lengths, resolving keys
    /// found within those lengths through the generation's index. Candidate
    /// locators and open shard handles are retained for lazy payload reads, so
    /// a concurrent repack may retire those shards while they remain readable.
    ///
    /// When a read-committed journal overlay is installed, its committed puts
    /// are merged in and win over the durable index (matching
    /// `get_read_committed`), and a collection delete the durable index does not
    /// yet reflect suppresses durable records entirely, so a live multi-reader
    /// store is scanned completely.
    ///
    /// The pack scan covers the captured file lengths without flushing or
    /// syncing them. Writes still buffered in this process are excluded unless
    /// supplied by the journal overlay; file-visible bytes need not be fsynced.
    /// Records are yielded as `(NodeId, NodeData)` with no guaranteed ordering.
    ///
    /// # Errors
    /// Propagates pack metadata-scan and journal-refresh errors, including
    /// [`StorageError::Io`] with `WouldBlock` when a read-journal reload must
    /// be retried. Payload-read errors are yielded by the returned iterator.
    pub fn scan_collection(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<CollectionScan<'_>, StorageError> {
        // Puts update a non-mmap index in place without replacing its
        // generation. Serialize snapshot construction with those updates so
        // every locator we resolve belongs to the captured pack-length
        // boundary. Shard handles are opened before the guard is released;
        // the full metadata scan and lazy payload reads do not block later puts.
        let put_lock = self.put_mutex(collection_id);
        let mut put_guard = Some(put_lock.lock());
        self.scan_collection_locked(collection_id, &mut put_guard)
    }

    /// Take a collection scan paired with a durable journal cursor.
    ///
    /// The collection lock spans the pack flush, WAL durability barrier, and
    /// snapshot construction. Thus every mutation for this collection through
    /// the returned cursor is represented by the scan; later mutations receive
    /// larger LSNs and are available from [`JournalCoordinator::changes_since`].
    /// The returned [`crate::journal::JournalReplayLease`] pins that history
    /// until dropped, preventing either per-pool or shared-WAL reclaim from
    /// expiring the cursor while the caller catches up.
    ///
    /// In a shared-WAL `Database`, a transaction publishes its journal group before
    /// materializing its pack records. The lifecycle lock covers the active
    /// transaction check and WAL-boundary capture, then is released before the
    /// pack walk. A transaction that starts later cannot materialize this
    /// collection while its put mutex is held, and this method scans packs only
    /// (never the read-journal overlay), so its mutations remain strictly in
    /// replay. The lock can span the WAL sync, but not the collection scan or
    /// the preceding pack flush.
    ///
    /// # Errors
    /// Returns `Unsupported` when no journal is enabled, `WouldBlock` while a
    /// transaction is already being materialized, or propagates pack/WAL
    /// errors. It also returns an expired-cursor error if reclaim wins the race
    /// before the replay lease is installed; in that case retry the snapshot.
    pub fn scan_collection_at_snapshot(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<(CollectionScan<'_>, JournalCursor, JournalReplayLease), StorageError> {
        let put_lock = self.put_mutex(collection_id);
        let mut put_guard = Some(put_lock.lock());

        let journal = self.journal().ok_or_else(|| {
            StorageError::Unsupported(
                "replayable collection scans require an enabled journal".to_owned(),
            )
        })?;

        // Flush before taking the transaction lifecycle lock. A target-
        // collection put is excluded by `put_mutex`; unrelated writes may
        // proceed, but their records are irrelevant to this collection.
        self.shards.flush_all()?;

        let (cursor, lease) = self.capture_replay_boundary(&journal)?;

        #[cfg(test)]
        if let Some(hook) = self.replay_snapshot_hook.lock().take() {
            hook();
        }

        let scan = self.scan_collection_packs_locked(collection_id, &mut put_guard)?;
        Ok((scan, cursor, lease))
    }

    /// Sync the journal and pin a cursor at this pool's materialized watermark,
    /// floored below any published-but-not-yet-materialized transaction and by
    /// durable checkpoint coverage when the retained WAL prefix is empty.
    ///
    /// Refuses while a transaction overlay is materializing: the materialized
    /// packs are not yet a stable boundary, so a scan captured then could mix
    /// pre- and post-materialization state.
    fn capture_replay_boundary(
        &self,
        journal: &JournalCoordinator,
    ) -> Result<(JournalCursor, JournalReplayLease), StorageError> {
        {
            let _transaction_lifecycle = self.transaction_overlay_lifecycle.lock();
            if self.transaction_overlay_users.load(Ordering::Acquire) != 0 {
                return Err(StorageError::WouldBlock(
                    "cannot capture a replayable scan while a transaction is materializing"
                        .to_owned(),
                ));
            }
        }

        journal.sync().map_err(StorageError::Io)?;
        let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
        let cursor = journal
            .replay_cursor(covered_lsn)
            .map_err(StorageError::Io)?;
        let lease = journal
            .pin_replay_cursor(&cursor)
            .map_err(StorageError::Io)?;
        Ok((cursor, lease))
    }

    /// Take scans of multiple collections at one shared durable journal
    /// boundary.
    ///
    /// Collection IDs are sorted and deduplicated; returned scans follow that
    /// order. Their collection put mutexes are acquired in the same sorted
    /// order, then held through the shared WAL boundary capture and until shard
    /// handles and file-length bounds for every scan are open. This ensures a
    /// write to any requested collection is either represented by its scan at
    /// the shared cursor or left for replay after it. The mutexes are released
    /// before walking pack metadata and before the lazy scans are returned.
    ///
    /// The returned replay lease pins history after the shared cursor until the
    /// caller has caught up (or advances the lease) across all the collections.
    /// This API is scoped to one `PackfileStorage`/pool; cross-pool snapshots
    /// must use a shared-database boundary API.
    ///
    /// # Errors
    /// Returns `Unsupported` when no journal is enabled, `WouldBlock` while a
    /// transaction is already being materialized, or propagates pack/WAL
    /// errors. It also returns an expired-cursor error if reclaim wins the race
    /// before the replay lease is installed; in that case retry the snapshot.
    pub fn scan_collections_at_snapshot(
        &self,
        collection_ids: &[[u8; 16]],
    ) -> Result<
        (
            Vec<CollectionScanPair<'_>>,
            crate::journal::JournalCursor,
            JournalReplayLease,
        ),
        StorageError,
    > {
        let mut collection_ids = collection_ids.to_vec();
        collection_ids.sort_unstable();
        collection_ids.dedup();

        // Every caller acquires multiple collection locks in this canonical
        // order, avoiding deadlocks with other multi-collection operations.
        let put_locks: Vec<_> = collection_ids
            .iter()
            .map(|collection_id| self.put_mutex(collection_id))
            .collect();
        let put_guards: Vec<_> = put_locks.iter().map(|lock| lock.lock()).collect();

        let journal = self.journal().ok_or_else(|| {
            StorageError::Unsupported(
                "replayable collection scans require an enabled journal".to_owned(),
            )
        })?;

        // Flush before taking the transaction lifecycle lock, as in the
        // single-collection entry point. The requested collection locks are
        // held, so their buffered puts cannot cross the boundary.
        self.shards.flush_all()?;

        let capture_boundary =
            || -> Result<(crate::journal::JournalCursor, JournalReplayLease), StorageError> {
                // Reclaim can pass the covered LSN between reading it and
                // pinning the cursor. Shared reclaim may also have passed an
                // idle pool's recorded coverage, which never advances on its
                // own, so a re-read alone would rebuild the same expired
                // cursor. Every mutation in the reclaimed prefix is covered
                // by its contributing pool; if this pool's watermark is older
                // than that prefix, it contributed no mutations there. Its
                // pack scan is therefore complete through the retained floor.
                let mut attempts = 0_u32;
                loop {
                    journal.sync().map_err(StorageError::Io)?;
                    let covered_lsn = self.checkpoint_covered_lsn().unwrap_or(0);
                    let mut cursor = journal
                        .replay_cursor(covered_lsn)
                        .map_err(StorageError::Io)?;
                    if cursor.is_behind_retained_history() {
                        cursor = journal
                            .replay_cursor(cursor.retained_floor_lsn())
                            .map_err(StorageError::Io)?;
                    }
                    match journal.pin_replay_cursor(&cursor) {
                        Ok(lease) => return Ok((cursor, lease)),
                        Err(error)
                            if attempts < 64 && crate::journal::is_cursor_expired(&error) =>
                        {
                            attempts = attempts.saturating_add(1);
                            std::thread::yield_now();
                        }
                        Err(error) => return Err(StorageError::Io(error)),
                    }
                }
            };

        let (cursor, lease) = {
            let _transaction_lifecycle = self.transaction_overlay_lifecycle.lock();
            if self.transaction_overlay_users.load(Ordering::Acquire) != 0 {
                return Err(StorageError::WouldBlock(
                    "cannot capture a replayable scan while a transaction is materializing"
                        .to_owned(),
                ));
            }
            capture_boundary()?
        };

        // Open every scanner and capture every file-length boundary before
        // releasing any requested collection's put lock. The subsequent pack
        // walk uses retained handles, so repack may safely unlink old shards.
        let mut prepared: Vec<PreparedCollectionScan> = Vec::with_capacity(collection_ids.len());
        for collection_id in &collection_ids {
            // Own the generation `Arc` so it stays pinned while the pack walk
            // runs after every collection put lock has been released.
            let generation = self.generation(collection_id).map(|guard| {
                let generation = Arc::clone(&*guard);
                if generation.index.is_mmap_backed() {
                    generation
                } else {
                    Arc::new(CollectionGeneration {
                        index: generation.index.clone(),
                        cache: Arc::clone(&generation.cache),
                        generation: generation.generation,
                    })
                }
            });
            let shards = self.shards.all_shards();
            let scanners = if generation.is_some() {
                Self::open_collection_scan_shards(&shards)?
            } else {
                Vec::new()
            };
            prepared.push((*collection_id, generation, shards, scanners));
        }
        drop(put_guards);

        let mut scans = Vec::with_capacity(prepared.len());
        for (collection_id, generation, shards, scanners) in prepared {
            scans.push((
                collection_id,
                self.finish_collection_pack_scan(
                    generation.as_deref(),
                    &shards,
                    scanners,
                    &collection_id,
                )?,
            ));
        }
        Ok((scans, cursor, lease))
    }

    /// Finish a collection scan after its shard handles and lengths have been
    /// captured and the collection put lock has been released.
    fn finish_collection_pack_scan(
        &self,
        generation: Option<&CollectionGeneration>,
        shards: &[(u16, Arc<Shard>)],
        scanners: Vec<(u64, packfile::PackfileScanner)>,
        collection_id: &[u8; 16],
    ) -> Result<CollectionScan<'_>, StorageError> {
        let keys = Self::scan_collection_keys(scanners, collection_id)?;
        let mut pinned = HashMap::new();
        let mut work = Vec::new();
        if let Some(generation) = generation {
            Self::append_collection_locator_work(
                keys,
                generation,
                &HashMap::new(),
                &HashSet::new(),
                shards,
                &mut pinned,
                &mut work,
            );
        }
        Ok(CollectionScan {
            store: self,
            pinned,
            pending: work.into_iter(),
        })
    }

    /// Build the pinned scan while the caller's collection put mutex remains
    /// held. The shared scan helper releases it after opening the snapshot's
    /// shard handles.
    fn scan_collection_locked(
        &self,
        collection_id: &[u8; 16],
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        // Snapshot the read-committed overlay once: committed-but-unflushed
        // puts, plus the per-collection delete boundary a reader's stale index
        // may not yet cover.
        let (overlay_puts, overlay_removes, overlay_deletes) = {
            let guard = self.refresh_read_journal()?;
            match guard.as_ref() {
                Some(journal) => {
                    let deleted = journal.delete_lsn.contains_key(collection_id);
                    let puts: HashMap<NodeId, NodeData> = journal
                        .puts
                        .get(collection_id)
                        .map(|committed| {
                            committed
                                .iter()
                                .map(|(id, (payload, _lsn))| (*id, NodeData::new(payload.clone())))
                                .collect()
                        })
                        .unwrap_or_default();
                    let removes: HashSet<NodeId> = journal
                        .removes
                        .get(collection_id)
                        .map(|entries| entries.keys().copied().collect())
                        .unwrap_or_default();
                    (puts, removes, deleted)
                }
                None => (HashMap::new(), HashSet::new(), false),
            }
        };

        self.scan_collection_with_overlay(
            collection_id,
            &overlay_puts,
            &overlay_removes,
            overlay_deletes,
            put_guard,
        )
    }

    /// Snapshot only materialized pack records. The replayable scan uses this
    /// form so an overlay mutation beyond its cursor cannot be included both in
    /// the scan and in subsequent WAL replay.
    fn scan_collection_packs_locked(
        &self,
        collection_id: &[u8; 16],
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        let overlay_puts = HashMap::new();
        let overlay_removes = HashSet::new();
        self.scan_collection_with_overlay(
            collection_id,
            &overlay_puts,
            &overlay_removes,
            false,
            put_guard,
        )
    }

    /// Build lazy scan work, letting `overlay_puts` replace pack values.
    ///
    /// `overlay_deletes` suppresses all pack records for this collection. The
    /// caller holds its put mutex in `put_guard`; this releases the guard after
    /// capturing the generation, file lengths, and open handles. Pack-open and
    /// metadata-scan errors propagate.
    fn scan_collection_with_overlay(
        &self,
        collection_id: &[u8; 16],
        overlay_puts: &HashMap<NodeId, NodeData>,
        overlay_removes: &HashSet<NodeId>,
        overlay_deletes: bool,
        put_guard: &mut Option<MutexGuard<'_, ()>>,
    ) -> Result<CollectionScan<'_>, StorageError> {
        // ArcSwap keeps the generation itself stable, but a non-mmap index is
        // updated in place by puts. Copy that index while the put lock is held
        // so locator resolution after the unlock still uses this boundary.
        let generation = self.generation(collection_id).map(|guard| {
            let generation = Arc::clone(&*guard);
            if generation.index.is_mmap_backed() {
                generation
            } else {
                Arc::new(CollectionGeneration {
                    index: generation.index.clone(),
                    cache: Arc::clone(&generation.cache),
                    generation: generation.generation,
                })
            }
        });
        let shards = self.shards.all_shards();

        // A delete visible only in the overlay means the reader's durable
        // index may still hold pre-delete records, so it cannot be trusted
        // for this collection; the overlay puts are the whole answer.
        let scanners = if !overlay_deletes && generation.is_some() {
            Self::open_collection_scan_shards(&shards)?
        } else {
            Vec::new()
        };

        // The generation, file lengths, and open file handles now pin the
        // snapshot boundary. Releasing the collection lock here lets writers
        // proceed while the metadata scan walks every shard; retained handles
        // remain readable if repack unlinks a shard.
        drop(put_guard.take());

        let keys = Self::scan_collection_keys(scanners, collection_id)?;
        let mut pinned = HashMap::new();
        let mut work = Vec::new();
        if let Some(generation) = generation.as_deref() {
            Self::append_collection_locator_work(
                keys,
                generation,
                overlay_puts,
                overlay_removes,
                &shards,
                &mut pinned,
                &mut work,
            );
        }

        // Durable keys shadowed by the overlay were skipped above, so each
        // overlay put is appended once.
        work.extend(
            overlay_puts
                .iter()
                .map(|(id, data)| ScanWork::Data(*id, data.clone())),
        );

        Ok(CollectionScan {
            store: self,
            pinned,
            pending: work.into_iter(),
        })
    }

    /// Open metadata scanners paired with each shard's captured byte length.
    /// Open, metadata, and header-validation errors propagate.
    pub(super) fn open_collection_scan_shards(
        shards: &[(u16, Arc<Shard>)],
    ) -> Result<Vec<(u64, packfile::PackfileScanner)>, StorageError> {
        shards
            .iter()
            .map(|(_, shard)| {
                // Scan the physical file: a newly created pack may still be
                // staged until its first durability barrier publishes it.
                let file = shard.open_data_file()?;
                let end = shard.file_len();
                let scanner = packfile::scan_packfile_iter_from_file(file, false)?;
                Ok((end, scanner))
            })
            .collect()
    }

    /// Collect distinct hashes for `collection_id` before each captured byte
    /// length, in first-seen order. Scanner errors propagate.
    pub(super) fn scan_collection_keys(
        scanners: Vec<(u64, packfile::PackfileScanner)>,
        collection_id: &[u8; 16],
    ) -> Result<Vec<NodeId>, StorageError> {
        let mut keys = Vec::new();
        let mut seen = HashSet::new();
        for (end, scanner) in scanners {
            for entry in scanner {
                let (record_collection, hash, offset) = entry?;
                // The scanner streams in file order, so crossing the captured
                // length means every later entry is post-boundary.
                if offset >= end {
                    break;
                }
                if record_collection != *collection_id {
                    continue;
                }
                if seen.insert(hash) {
                    keys.push(hash);
                }
            }
        }
        Ok(keys)
    }

    /// Append indexed candidates for keys not shadowed by `overlay_puts`,
    /// pinning available shard handles. Keys without candidates are skipped.
    #[allow(
        clippy::too_many_arguments,
        reason = "one scan snapshot: collection, keys, generation, overlay, shards, and outputs"
    )]
    fn append_collection_locator_work(
        keys: Vec<NodeId>,
        generation: &CollectionGeneration,
        overlay_puts: &HashMap<NodeId, NodeData>,
        overlay_removes: &HashSet<NodeId>,
        shards: &[(u16, Arc<Shard>)],
        pinned: &mut HashMap<u16, Arc<Shard>>,
        work: &mut Vec<ScanWork>,
    ) {
        for id in keys {
            if overlay_puts.contains_key(&id) || overlay_removes.contains(&id) {
                // An overlay mutation is newer than the durable one.
                continue;
            }
            let candidates = generation.candidates(&id);
            if candidates.is_empty() {
                continue;
            }
            for &(slot, _) in &candidates {
                if let Some((_, shard)) = shards.iter().find(|(s, _)| *s == slot) {
                    pinned.entry(slot).or_insert_with(|| Arc::clone(shard));
                }
            }
            work.push(ScanWork::Locators(id, candidates));
        }
    }

    /// Materialized scan of one collection, kept for tests that assert on
    /// the raw frame list; `rebuild_index` streams instead.
    #[cfg(test)]
    pub(super) fn scan_collection_records(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Vec<TestScannedRecord>, StorageError> {
        let mut scanned = Vec::new();
        for (slot, shard) in self.shards.all_shards() {
            for entry in shard.scan_data_path(|path| {
                packfile::scan_packfile_with_removal(
                    path,
                    self.shards.checksum_policy().verifies_reads(),
                )
            })? {
                if entry.collection_id == *collection_id {
                    scanned.push((slot, entry.hash, entry.offset, shard.pack_id, entry.removed));
                }
            }
        }
        Ok(scanned)
    }

    pub(super) fn scan_collection_repack_records(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Vec<ScannedRepackShard>, StorageError> {
        let mut scanned = Vec::new();
        for (slot, shard) in self.shards.all_shards() {
            let entries = shard
                .scan_data_path(|path| {
                    packfile::scan_packfile_with_removal(
                        path,
                        self.shards.checksum_policy().verifies_reads(),
                    )
                })?
                .into_iter()
                .filter(|entry| entry.collection_id == *collection_id)
                .map(|entry| (entry.hash, entry.offset, entry.removed))
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                scanned.push((slot, entries));
            }
        }
        Ok(scanned)
    }
}
