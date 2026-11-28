//! Buffered transaction staging for journal-backed writes.
//!
//! [`TxnStage`] holds a transaction's mutations until its caller declares
//! commit, then hands them to the shared-WAL publisher. The state machine and
//! its supporting types are declared in the parent module.

use super::*;

impl Default for TxnStage {
    fn default() -> Self {
        Self::new()
    }
}

impl TxnStage {
    const ACTIVE: u8 = 0;
    const DISCARDED: u8 = 1;
    const PUBLISHED: u8 = 2;
    const MATERIALIZING: u8 = 3;
    const JOURNAL_PUBLISHED: u8 = 4;

    /// Create an empty stage for one transaction attempt.
    #[must_use]
    pub fn new() -> Self {
        Self::with_limit(MAX_TXN_STAGE_BYTES)
    }

    /// Create an empty stage with an explicit payload limit.
    #[must_use]
    pub(crate) fn with_limit(limit_bytes: usize) -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(Self::ACTIVE),
            data: Mutex::new(TxnStageData {
                pools: std::array::from_fn(|_| Vec::new()),
                index: std::array::from_fn(|_| StageIndex::default()),
                applied: std::array::from_fn(|_| Vec::new()),
                expectations: Vec::new(),
                record_expectations: Vec::new(),
                record_expectation_keys: HashSet::new(),
                bytes: 0,
                appended: [false; ShardType::ALL.len()],
                receipt: None,
            }),
            limit_bytes,
            #[cfg(test)]
            lookup_many_calls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> TxnStageState {
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED => TxnStageState::Discarded,
            Self::PUBLISHED => TxnStageState::Published,
            Self::MATERIALIZING => TxnStageState::Materializing,
            Self::JOURNAL_PUBLISHED => TxnStageState::JournalPublished,
            _ => TxnStageState::Active,
        }
    }

    /// Look one record up in the mutations staged for `pool`. See
    /// [`Self::lookup_many`], which this calls.
    #[must_use]
    pub fn lookup(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_id: &[u8; 16],
    ) -> StagedLookup {
        self.lookup_many(pool, collection_id, std::slice::from_ref(node_id))
            .pop()
            .unwrap_or(StagedLookup::Absent)
    }

    /// Look several records up in the mutations staged for `pool`, in the order
    /// of `node_ids`, taking the stage lock once.
    ///
    /// This is what lets a transaction read its own uncommitted writes. The
    /// staged mutations are in append order, so the newest one that names a
    /// record wins. A staged `DeleteCollection` for the record's collection
    /// hides everything older, both earlier staged puts and whatever the live
    /// pool holds, while a put staged after it is newer and is seen. Only an
    /// active stage answers: once the stage is discarded, or publication has
    /// begun, every record is `Absent` and the caller reads the live store,
    /// which by then serves the group through the transaction overlay.
    ///
    /// The stage keeps a per-pool `StageIndex` of the newest put or removal
    /// for each record and the last delete for each collection. A call folds in
    /// only the mutations staged since the previous one, so a transaction that
    /// stages and looks up records in turn costs each mutation once, not once
    /// per lookup. Payloads are cloned only for records that resolve to a put.
    #[must_use]
    pub fn lookup_many(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_ids: &[[u8; 16]],
    ) -> Vec<StagedLookup> {
        #[cfg(test)]
        self.lookup_many_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE || node_ids.is_empty() {
            return vec![StagedLookup::Absent; node_ids.len()];
        }
        let data = &mut *guard;
        let pool = pool_index(pool);
        data.index[pool].sync(&data.pools[pool]);
        let staged = &data.index[pool];
        let mutations = &data.pools[pool];
        let last_delete = staged.last_delete.get(collection_id).copied();
        node_ids
            .iter()
            .map(|node_id| {
                let newest = staged.newest.get(&(*collection_id, *node_id)).copied();
                match (newest, last_delete) {
                    // A put older than the delete is hidden by it.
                    (Some(position), Some(delete)) if position < delete => StagedLookup::Deleted,
                    (Some(position), _) => match mutations.get(position) {
                        Some(Mutation::Put { payload, .. }) => StagedLookup::Put(payload.clone()),
                        _ => StagedLookup::Deleted,
                    },
                    (None, Some(_)) => StagedLookup::Deleted,
                    (None, None) => StagedLookup::Absent,
                }
            })
            .collect()
    }

    /// Number of batch lookup passes, for complexity tests only.
    #[cfg(test)]
    #[must_use]
    pub fn lookup_many_calls(&self) -> u64 {
        self.lookup_many_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Discard an active attempt. Safe to call more than once.
    ///
    /// See the "Not a rollback" section on [`TxnStage`]: this drops only the
    /// buffered journal mutations and never touches packs or the live index.
    pub fn discard(&self) {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) == Self::ACTIVE {
            data.pools.iter_mut().for_each(Vec::clear);
            data.index.iter_mut().for_each(StageIndex::reset);
            data.applied.iter_mut().for_each(Vec::clear);
            data.expectations.clear();
            data.record_expectations.clear();
            data.record_expectation_keys.clear();
            data.bytes = 0;
            self.state.store(Self::DISCARDED, Ordering::Release);
        }
    }

    /// Snapshot staged mutations for application to storage at commit time.
    pub(crate) fn snapshot_mutations(&self) -> [Vec<Mutation>; ShardType::ALL.len()] {
        self.data.lock().pools.clone()
    }

    /// Receipt identifying this stage's published journal group.
    pub(crate) fn published_receipt(&self) -> Option<CommitReceipt> {
        self.data.lock().receipt
    }

    /// Which pools (in [`ShardType::ALL`] order) have at least one staged
    /// mutation.
    pub(crate) fn touched_pools(&self) -> [bool; ShardType::ALL.len()] {
        let data = self.data.lock();
        std::array::from_fn(|index| data.pools.get(index).is_some_and(|pool| !pool.is_empty()))
    }

    /// Whether this stage has no mutations to publish or materialize.
    pub(crate) fn is_empty(&self) -> bool {
        self.data.lock().pools.iter().all(Vec::is_empty)
    }

    /// Return whether one staged mutation has already been applied to storage.
    pub(crate) fn mutation_applied(&self, pool: ShardType, index: usize) -> bool {
        self.data.lock().applied[pool_index(pool)]
            .get(index)
            .copied()
            .unwrap_or(false)
    }

    /// Record successful application of one staged mutation. This makes a
    /// retry after a later mutation fails resume at the failed mutation.
    pub(crate) fn mark_mutation_applied(&self, pool: ShardType, index: usize) -> io::Result<()> {
        let mut data = self.data.lock();
        let applied = data.applied[pool_index(pool)]
            .get_mut(index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "staged mutation index out of bounds",
                )
            })?;
        *applied = true;
        Ok(())
    }

    /// Begin pack/index materialization after the journal group is published.
    pub(crate) fn begin_materialization(&self) -> io::Result<()> {
        let state = self.state.load(Ordering::Acquire);
        match state {
            Self::ACTIVE | Self::JOURNAL_PUBLISHED => {
                self.state.store(Self::MATERIALIZING, Ordering::Release);
                Ok(())
            }
            Self::MATERIALIZING | Self::PUBLISHED => Ok(()),
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other("invalid transaction stage state")),
        }
    }

    /// Mark the journal group published while storage application remains
    /// retryable.
    pub(crate) fn mark_journal_published(&self) -> io::Result<()> {
        match self.state.load(Ordering::Acquire) {
            Self::ACTIVE => {
                self.state.store(Self::JOURNAL_PUBLISHED, Ordering::Release);
                Ok(())
            }
            Self::JOURNAL_PUBLISHED | Self::MATERIALIZING | Self::PUBLISHED => Ok(()),
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other("invalid transaction stage state")),
        }
    }

    /// Complete an active no-op transaction without creating a journal group.
    pub(crate) fn mark_empty_published(&self) -> io::Result<()> {
        let data = self.data.lock();
        if data.pools.iter().any(|pool| !pool.is_empty()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot complete a transaction with staged mutations as empty",
            ));
        }
        match self.state.load(Ordering::Acquire) {
            Self::ACTIVE | Self::PUBLISHED => {
                self.state.store(Self::PUBLISHED, Ordering::Release);
                Ok(())
            }
            Self::DISCARDED => Err(io::Error::other("transaction stage was discarded")),
            _ => Err(io::Error::other(
                "transaction stage is not empty and active",
            )),
        }
    }

    /// Mark both journal publication and storage application complete.
    pub(crate) fn mark_published(&self) -> io::Result<()> {
        match self.state.load(Ordering::Acquire) {
            Self::MATERIALIZING | Self::PUBLISHED => {
                self.state.store(Self::PUBLISHED, Ordering::Release);
                Ok(())
            }
            _ => Err(io::Error::other("transaction storage has not been applied")),
        }
    }

    /// Ensure an estimated batch fits before buffering its journal mutations.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the stage is not active or the estimated
    /// total exceeds [`MAX_TXN_STAGE_BYTES`].
    pub fn ensure_capacity(&self, additional: usize) -> io::Result<()> {
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let data = self.data.lock();
        let total = data.bytes.checked_add(additional).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        Ok(())
    }

    /// Add an immutable put to this attempt's pool-ordered journal batch.
    ///
    /// Charges the payload length plus [`TXN_STAGE_PUT_OVERHEAD_BYTES`] against
    /// the staging budget.
    ///
    /// # Errors
    /// Returns `InvalidInput` if adding the payload would exceed the stage
    /// limit, or another error if the stage has already been discarded or
    /// published.
    pub fn stage_put(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        payload: Vec<u8>,
    ) -> io::Result<()> {
        let charge = payload.len().saturating_add(TXN_STAGE_PUT_OVERHEAD_BYTES);
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        data.pools[pool_index(pool)].push(Mutation::Put {
            collection_id,
            node_id,
            payload,
        });
        data.applied[pool_index(pool)].push(false);
        data.bytes = total;
        Ok(())
    }

    /// Add a per-record removal to this attempt.
    ///
    /// The caller must establish that the key is live (or was put earlier in
    /// this stage) before staging the removal. This keeps an absent-key
    /// removal from creating a journal entry or advancing collection clocks.
    ///
    /// # Errors
    /// Returns an error if the stage is inactive or its size limit would be
    /// exceeded.
    pub fn stage_remove_record(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        data.pools[pool_index(pool)].push(Mutation::RemoveRecord {
            collection_id,
            node_id,
        });
        data.applied[pool_index(pool)].push(false);
        data.bytes = total;
        Ok(())
    }

    /// Add per-record removals for many keys at once, charging the stage
    /// budget in one step. Same contract as [`Self::stage_remove_record`]: the
    /// caller has already established that each key is live.
    ///
    /// # Errors
    /// Returns an error if the stage is inactive or the batch would exceed its
    /// size limit; on error nothing is staged.
    pub fn stage_remove_records(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_ids: &[[u8; 16]],
    ) -> io::Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let charge = node_ids.len().checked_mul(64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        let total = data.bytes.checked_add(charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        let index = pool_index(pool);
        data.pools[index].extend(node_ids.iter().map(|node_id| Mutation::RemoveRecord {
            collection_id,
            node_id: *node_id,
        }));
        data.applied[index].extend(vec![false; node_ids.len()]);
        data.bytes = total;
        Ok(())
    }

    /// Stage a normalized removal batch all-or-nothing: drop this stage's earlier
    /// mutations for `pending`, stage removal markers for `live_ids`, and record
    /// each pending key's version precondition. The budget is checked before
    /// anything changes, so an `Err` leaves the stage exactly as it was.
    ///
    /// # Errors
    /// Returns an error if the stage is not active or the result would exceed
    /// the stage's byte limit.
    pub fn stage_removals_normalized(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        pending: &[([u8; 16], u64)],
        live_ids: &[[u8; 16]],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let forget: HashSet<[u8; 16]> = pending.iter().map(|(node_id, _)| *node_id).collect();
        let index = pool_index(pool);
        let mutation_charge = |mutation: &Mutation| match mutation {
            Mutation::Put {
                collection_id: c,
                node_id,
                payload,
            } if *c == collection_id && forget.contains(node_id) => {
                Some(payload.len().saturating_add(64))
            }
            Mutation::RemoveRecord {
                collection_id: c,
                node_id,
            } if *c == collection_id && forget.contains(node_id) => Some(64),
            _ => None,
        };
        let released = data.pools[index]
            .iter()
            .filter_map(mutation_charge)
            .fold(0_usize, usize::saturating_add);
        let overflow = || {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        };
        let charge = live_ids.len().checked_mul(64).ok_or_else(overflow)?;
        let total = data
            .bytes
            .saturating_sub(released)
            .checked_add(charge)
            .ok_or_else(overflow)?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }

        data.pools[index].retain(|mutation| mutation_charge(mutation).is_none());
        data.pools[index].extend(live_ids.iter().map(|node_id| Mutation::RemoveRecord {
            collection_id,
            node_id: *node_id,
        }));
        data.index[index].reset();
        data.applied[index] = vec![false; data.pools[index].len()];
        data.bytes = total;
        for (node_id, expected) in pending {
            data.record_expectation_keys
                .insert((index, collection_id, *node_id));
            data.record_expectations.push(RecordExpectation {
                pool,
                collection_id,
                node_id: *node_id,
                expected: *expected,
            });
        }
        Ok(())
    }

    /// Remove staged point mutations for a key whose live value is absent.
    /// Used to normalize `put` then `remove` into a true no-op before publish.
    ///
    /// # Errors
    /// Returns an error if the stage is inactive.
    pub fn forget_staged_record(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
    ) -> io::Result<()> {
        self.forget_staged_records(pool, collection_id, &HashSet::from([node_id]))
    }

    /// Drop every staged put or removal for the given keys in one pass over
    /// the pool's staged mutations, releasing exactly the budget they held.
    ///
    /// # Errors
    /// Returns an error if the stage is inactive.
    pub fn forget_staged_records(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_ids: &HashSet<[u8; 16]>,
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let index = pool_index(pool);
        let mut released = 0usize;
        data.pools[index].retain(|mutation| {
            let (c, node, charge) = match mutation {
                Mutation::Put {
                    collection_id: c,
                    node_id,
                    payload,
                } => (c, node_id, payload.len().saturating_add(64)),
                Mutation::RemoveRecord {
                    collection_id: c,
                    node_id,
                } => (c, node_id, 64),
                Mutation::DeleteCollection { .. } => return true,
            };
            if *c == collection_id && node_ids.contains(node) {
                released = released.saturating_add(charge);
                false
            } else {
                true
            }
        });
        // This helper only runs while the stage is active, before publication;
        // no mutation can have been applied yet.
        data.index[index].reset();
        data.applied[index] = vec![false; data.pools[index].len()];
        data.bytes = data.bytes.saturating_sub(released);
        Ok(())
    }

    /// Add a batch of immutable puts atomically.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the batch would exceed the stage limit, or
    /// another error if the stage has already been discarded or published.
    pub fn stage_puts(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        entries: &[([u8; 16], Vec<u8>)],
    ) -> io::Result<()> {
        self.stage_puts_inner(pool, collection_id, None, entries)
    }

    /// Atomically add record-version preconditions and puts. Used by
    /// transaction primitives that must not leave partial staged state when
    /// the staging budget or an expectation check fails.
    pub(crate) fn stage_puts_with_record_expectations(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        expectations: &[([u8; 16], u64)],
        entries: &[([u8; 16], Vec<u8>)],
    ) -> io::Result<()> {
        self.stage_puts_inner(pool, collection_id, Some(expectations), entries)
    }

    fn stage_puts_inner(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        expectations: Option<&[([u8; 16], u64)]>,
        entries: &[([u8; 16], Vec<u8>)],
    ) -> io::Result<()> {
        let charge = entries
            .iter()
            .try_fold(0usize, |total, (_, payload)| {
                total.checked_add(payload.len().saturating_add(64))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "transaction stage size overflow",
                )
            })?;
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(charge).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        let index = pool_index(pool);
        if let Some(expectations) = expectations {
            for (node_id, expected) in expectations {
                if data
                    .record_expectation_keys
                    .insert((index, collection_id, *node_id))
                {
                    data.record_expectations.push(RecordExpectation {
                        pool,
                        collection_id,
                        node_id: *node_id,
                        expected: *expected,
                    });
                }
            }
        }
        data.pools[index].extend(entries.iter().map(|(node_id, payload)| Mutation::Put {
            collection_id,
            node_id: *node_id,
            payload: payload.clone(),
        }));
        data.applied[index].extend(vec![false; entries.len()]);
        data.bytes = total;
        Ok(())
    }

    /// Add a collection delete to this attempt. Callers must not eagerly
    /// mutate the live index for a delete that can still roll back.
    ///
    /// # Errors
    /// Returns `InvalidInput` if adding the mutation would exceed the stage
    /// limit, or another error if the stage has already been discarded or
    /// published.
    pub fn stage_delete_collection(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        let total = data.bytes.checked_add(64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "transaction stage size overflow",
            )
        })?;
        if total > self.limit_bytes {
            return Err(stage_too_large_with_limit(total, self.limit_bytes));
        }
        data.pools[pool_index(pool)].push(Mutation::DeleteCollection { collection_id });
        data.applied[pool_index(pool)].push(false);
        data.bytes = total;
        Ok(())
    }

    /// Stage a collection-version precondition for validation at publication.
    ///
    /// # Errors
    /// Returns an error if the stage has already been discarded or published.
    pub fn stage_expectation(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        expected: u64,
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        if let Some(existing) = data
            .expectations
            .iter()
            .find(|existing| existing.pool == pool && existing.collection_id == collection_id)
        {
            if existing.expected != expected && expected != 0 {
                // Keep the first version observed before this transaction's own
                // staged writes, but reject a second distinct real
                // precondition: validating only the first would let the later
                // one silently pass at commit. A synthetic zero token from a
                // fully staged collection is not a real precondition and must
                // not replace the original.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "conflicting collection-version expectations for collection \
                         {collection_id:?}: already staged {}, then {}",
                        existing.expected, expected
                    ),
                ));
            }
        } else {
            data.expectations.push(CollectionExpectation {
                pool,
                collection_id,
                expected,
            });
        }
        Ok(())
    }

    /// Snapshot the staged collection-version preconditions.
    #[must_use]
    pub fn expectations(&self) -> Vec<CollectionExpectation> {
        self.data.lock().expectations.clone()
    }

    /// Stage a per-record version precondition for validation at publication.
    /// `expected` is the record's write LSN as observed by a read (the
    /// collection's delete LSN when the record was absent).
    ///
    /// # Errors
    /// Returns an error if the stage has already been discarded or published.
    pub fn stage_record_expectation(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        expected: u64,
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        if self.state.load(Ordering::Acquire) != Self::ACTIVE {
            return Err(io::Error::other("transaction stage is not active"));
        }
        data.record_expectation_keys
            .insert((pool_index(pool), collection_id, node_id));
        data.record_expectations.push(RecordExpectation {
            pool,
            collection_id,
            node_id,
            expected,
        });
        Ok(())
    }

    /// Whether a per-record version precondition is already staged for this
    /// key, regardless of whether the key also has a staged write.
    #[must_use]
    pub fn has_record_expectation(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_id: &[u8; 16],
    ) -> bool {
        self.data.lock().record_expectation_keys.contains(&(
            pool_index(pool),
            *collection_id,
            *node_id,
        ))
    }

    /// Snapshot the staged per-record version preconditions.
    #[must_use]
    pub fn record_expectations(&self) -> Vec<RecordExpectation> {
        self.data.lock().record_expectations.clone()
    }

    /// Append staged pool groups in dependency order: edges, event-DAG,
    /// then state. Does not fsync; the ordinary coalesced sync remains the
    /// durability boundary. Repeated calls are safe, including after a
    /// partial error.
    ///
    /// # Errors
    /// Returns an error if a staged pool has no coordinator or its journal
    /// cannot append the group's complete framing and trailer.
    pub fn publish(
        &self,
        coordinators: [Option<&JournalCoordinator>; ShardType::ALL.len()],
    ) -> io::Result<()> {
        let mut data = self.data.lock();
        match self.state.load(Ordering::Acquire) {
            Self::DISCARDED | Self::JOURNAL_PUBLISHED | Self::MATERIALIZING | Self::PUBLISHED => {
                return Ok(())
            }
            _ => {}
        }
        if coordinators.iter().all(Option::is_none) {
            // Journaling is disabled process-wide, so there is no journal to
            // publish into.
            self.state.store(Self::PUBLISHED, Ordering::Release);
            return Ok(());
        }
        // Preserve the historical per-pool append order for the existing
        // pools; ServerInfo is appended after them without changing the WAL
        // ordering of the other pools' frames.
        let pools = [ShardType::EventDag, ShardType::State, ShardType::ServerInfo];

        // A shared-WAL database gives every pool the same coordinator. Keep
        // the transaction as one journal group in that case, so readers never
        // observe only a prefix of a cross-pool commit. Standalone/per-pool
        // journals cannot provide that boundary and retain the ordered,
        // retry-safe fallback below.
        let active_pool_count = pools
            .iter()
            .filter(|pool| {
                let index = pool_index(**pool);
                !data.appended[index] && !data.pools[index].is_empty()
            })
            .count();
        if active_pool_count == 0 {
            self.state.store(Self::PUBLISHED, Ordering::Release);
            return Ok(());
        }
        let active_coordinators = pools
            .iter()
            .filter_map(|pool| {
                let index = pool_index(*pool);
                (!data.appended[index] && !data.pools[index].is_empty())
                    .then(|| coordinators[index])
            })
            .flatten()
            .collect::<Vec<_>>();
        if active_coordinators.len() == active_pool_count {
            if let Some(coordinator) = active_coordinators.first().copied() {
                if active_coordinators
                    .iter()
                    .all(|candidate| std::ptr::eq(*candidate, coordinator))
                {
                    let batches = pools
                        .iter()
                        .filter_map(|pool| {
                            let index = pool_index(*pool);
                            (!data.appended[index] && !data.pools[index].is_empty())
                                .then_some((*pool, data.pools[index].as_slice()))
                        })
                        .collect::<Vec<_>>();
                    let receipt = coordinator.publish_tagged_groups_checked(
                        &batches,
                        &data.expectations,
                        &data.record_expectations,
                    )?;
                    data.appended.fill(true);
                    data.receipt = Some(receipt);
                    self.mark_journal_published()?;
                    return Ok(());
                }
            }
        }

        if active_pool_count > 1 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-pool transactions require one shared journal coordinator",
            ));
        }

        for pool in pools {
            let index = pool_index(pool);
            if data.appended[index] || data.pools[index].is_empty() {
                data.appended[index] = true;
                continue;
            }
            let coordinator = coordinators[pool_index(pool)].ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotConnected, "staged pool has no journal")
            })?;
            let receipt = coordinator.publish_group_tagged_checked(
                pool,
                &data.pools[index],
                &data.expectations,
            )?;
            data.appended[index] = true;
            data.receipt = Some(receipt);
        }
        self.mark_journal_published()?;
        Ok(())
    }
}
