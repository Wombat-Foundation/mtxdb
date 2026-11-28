//! The journal coordinator: durability waits, replay leases, and background commits.
//!
//! The `JournalCoordinator` struct and its counters live in the parent module;
//! this file holds its inherent impl block.

use super::*;

impl JournalCoordinator {
    /// Seed logical collection versions and per-record write LSNs from the
    /// retained segment. Records whose group was reclaimed are absent and fall
    /// back to their frame's `last_write_lsn` on first read.
    fn seed_versions(scan: &Scan) -> SeededVersions {
        let mut collection_versions: HashMap<(ShardType, [u8; 16]), u64> = HashMap::new();
        let mut record_versions: HashMap<(ShardType, [u8; 16]), RecordVersions> = HashMap::new();
        for group in &scan.groups {
            for entry in &group.entries {
                let pool = entry.pool;
                match &entry.mutation {
                    Mutation::Put {
                        collection_id,
                        node_id,
                        ..
                    }
                    | Mutation::RemoveRecord {
                        collection_id,
                        node_id,
                    } => {
                        let version = collection_versions
                            .entry((pool, *collection_id))
                            .or_insert(0);
                        *version = (*version).max(group.last_lsn);
                        let collection = record_versions.entry((pool, *collection_id)).or_default();
                        let record = collection.nodes.entry(*node_id).or_insert(0);
                        *record = (*record).max(group.last_lsn);
                    }
                    Mutation::DeleteCollection { collection_id } => {
                        let version = collection_versions
                            .entry((pool, *collection_id))
                            .or_insert(0);
                        *version = (*version).max(group.last_lsn);
                        let collection = record_versions.entry((pool, *collection_id)).or_default();
                        collection.nodes.clear();
                        collection.deleted_at = Some(
                            collection
                                .deleted_at
                                .map_or(group.last_lsn, |deleted| deleted.max(group.last_lsn)),
                        );
                    }
                }
            }
        }
        (collection_versions, record_versions)
    }

    /// Build a coordinator from an opened journal and its recovery scan.
    #[must_use]
    pub fn new(journal: Journal, scan: &Scan) -> Self {
        let committed_lsn = scan.groups.last().map_or(0, |group| group.last_lsn);
        let path = journal.path.clone();
        let coverage = CoverageState::default();
        // Seed each pool's committed watermark from the recovered groups, so a
        // fresh coordinator over a pre-existing segment reports the same
        // coverage the segment already carries.
        let mut pool_committed: HashMap<ShardType, u64> = HashMap::new();
        for group in &scan.groups {
            for entry in &group.entries {
                let watermark = pool_committed.entry(entry.pool).or_insert(0);
                *watermark = (*watermark).max(group.last_lsn);
            }
        }
        // Seed each collection's logical version and each record's write LSN
        // from the retained segment. A version whose group was already
        // reclaimed is not recoverable here and reads as 0 until written again.
        let (collection_versions, record_versions) = Self::seed_versions(scan);
        Self {
            journal: Mutex::new(journal),
            path,
            publication: Mutex::new(()),
            published_lsn: AtomicU64::new(committed_lsn),
            visible_lsn: AtomicU64::new(committed_lsn),
            committed_lsn: AtomicU64::new(committed_lsn),
            reclaim_trigger_len: AtomicU64::new(RECLAIM_TRIGGER_LEN),
            replay_lease_cap: AtomicU64::new(u64::MAX),
            replay_leases_revoked: AtomicU64::new(0),
            reclaim_stall: Mutex::new(None),
            replay_lease_stall: Mutex::new(None),
            coverage_epoch: AtomicU64::new(0),
            reclaim_stalls: AtomicU64::new(0),
            #[cfg(test)]
            reclaim_hook: Mutex::new(None),
            blocker_remediation: Mutex::new(None),
            last_remediation: Mutex::new(None),
            sequence: None,
            coverage: Mutex::new(coverage),
            pool_committed: Mutex::new(pool_committed),
            collection_versions: Mutex::new(collection_versions),
            record_versions: Mutex::new(record_versions),
            version_floor: Mutex::new(HashMap::new()),
            unmaterialized: Mutex::new(std::collections::BTreeSet::new()),
            materialized_cv: Condvar::new(),
            materialized_lock: Mutex::new(()),
            pending_promotions: Mutex::new(Vec::new()),
            recovered: scan.groups.clone(),
            poisoned: AtomicBool::new(false),
            apply_failure: std::sync::OnceLock::new(),
            sync_calls: AtomicU64::new(0),
            journal_waiters: AtomicU64::new(0),
            coalesced_syncs: AtomicU64::new(0),
            sync_in_flight: AtomicU64::new(0),
            durable_requested: AtomicU64::new(committed_lsn),
            sync_lock: Mutex::new(()),
            #[cfg(test)]
            fsync_hook: Mutex::new(None),
            pre_sync_hook: Mutex::new(None),
            durable_lock: Mutex::new(()),
            durable_cv: Condvar::new(),
            background: Mutex::new(BackgroundState::Stopped),
            background_failure: Mutex::new(None),
            committer_wake_threshold: AtomicU64::new(0),
            commit_epoch: AtomicU64::new(0),
            threshold_notified_epoch: AtomicU64::new(u64::MAX),
            #[cfg(test)]
            committer_parks: AtomicU64::new(0),
            #[cfg(test)]
            threshold_wakes: AtomicU64::new(0),
            background_commits: AtomicU64::new(0),
            background_coalesced: AtomicU64::new(0),
            durable_requests: AtomicU64::new(0),
            durable_waits_already_durable: AtomicU64::new(0),
            durable_wait: DurableWaitTotals::default(),
            commits: AtomicU64::new(0),
            commit_records: AtomicU64::new(0),
            max_commit_records: AtomicU64::new(0),
            publish_signal: std::sync::OnceLock::new(),
            replay_leases: Arc::new(ReplayLeaseRegistry::default()),
            next_replay_lease: AtomicU64::new(1),
            replay_page_stats: ReplayPageStatsAtomic::default(),
        }
    }

    /// The committed groups recovered when this coordinator's segment was
    /// opened, in sequence order. A store attaching to a coordinator it did
    /// not construct (the shared-WAL path) uses this to seed its replay set.
    #[must_use]
    pub fn recovered_groups(&self) -> Vec<CommittedGroup> {
        self.recovered.clone()
    }

    /// The newest LSN of history that a reclaim removed from the shared WAL for
    /// `pool`, as the WAL header records it: 0 when none was, and `None` when
    /// the header cannot say (a journal written before it recorded this), which
    /// is not proof that nothing was reclaimed.
    ///
    /// A pool must restore its collection versions through this LSN from its
    /// own files, since the groups that carried them are gone.
    #[must_use]
    pub fn reclaimed_history_through(&self, pool: ShardType) -> u64 {
        self.journal.lock().reclaimed_history[pool.index()]
    }

    /// Snapshot cumulative replay paging path counters for this coordinator.
    #[must_use]
    pub fn replay_page_stats(&self) -> ReplayPageStats {
        ReplayPageStats {
            resumed_pages: self.replay_page_stats.resumed_pages.load(Ordering::Relaxed),
            full_scan_fallbacks: self
                .replay_page_stats
                .full_scan_fallbacks
                .load(Ordering::Relaxed),
        }
    }

    /// Create and map the cross-process publish signal beside this
    /// coordinator's segment. Idempotent; journal setup fails if the signal
    /// cannot be installed, since an old mapped signal could otherwise make
    /// workers skip refreshes after this writer starts publishing.
    ///
    /// The store that enables a journal calls this so read-only workers can
    /// sample `(epoch, revision)` with atomic loads instead of a
    /// per-call `fs::metadata`. A store that never calls it leaves the signal
    /// absent, and workers fall back to the stat-based refresh.
    ///
    /// # Errors
    /// Propagates signal-file and OS entropy errors. A failed attempt is
    /// cached, so later calls return the same error without retrying.
    pub(crate) fn enable_publish_signal(&self) -> io::Result<()> {
        let result = self.publish_signal.get_or_init(|| {
            PublishSignal::writer(&self.path)
                .map(Arc::new)
                .map_err(|error| (error.kind(), error.to_string()))
        });
        result
            .as_ref()
            .map(|_| ())
            .map_err(|(kind, message)| io::Error::new(*kind, message.clone()))
    }

    /// The publish signal, if [`Self::enable_publish_signal`] created one.
    #[must_use]
    pub(crate) fn publish_signal(&self) -> Option<&Arc<PublishSignal>> {
        self.publish_signal
            .get()
            .and_then(|result| result.as_ref().ok())
    }

    /// Run the test-only reclaim hook; a no-op outside tests.
    #[inline]
    #[cfg_attr(not(test), allow(clippy::unused_self))]
    fn run_reclaim_hook(&self) {
        #[cfg(test)]
        {
            let hook = self.reclaim_hook.lock().clone();
            if let Some(hook) = hook {
                hook();
            }
        }
    }

    /// Advance the publish signal so a gated reader rescans. Called on a group
    /// publication and on a reclaim, the two events a reader must notice.
    fn bump_publish_signal(&self) {
        if let Some(signal) = self.publish_signal() {
            signal.bump();
        }
    }

    /// Record that `pool`'s durable checkpoint has materialized every frame it
    /// owns through `lsn`. Coverage only advances. A pool with no reported
    /// coverage never blocks a group that does not carry its frames.
    pub fn report_pool_coverage(&self, pool: ShardType, lsn: u64) {
        let mut coverage = self.coverage.lock();
        let entry = coverage.covered.entry(pool).or_insert(0);
        if lsn > *entry {
            *entry = lsn;
            self.coverage_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Logical version of `collection_id` in `pool`: the global LSN of the last
    /// published mutation that touched it, or `0` when none is known (never
    /// written, or its last write predates the retained segment). See the
    /// tracker field docs for the reclaim/reopen caveat.
    #[must_use]
    pub fn collection_version(&self, pool: ShardType, collection_id: &[u8; 16]) -> u64 {
        self.collection_versions
            .lock()
            .get(&(pool, *collection_id))
            .copied()
            .unwrap_or(0)
    }

    /// Version of one record: the LSN of the group that last wrote it, or the
    /// collection's delete LSN when it was removed. `None` means the live map
    /// has no entry, so the caller must seed it from durable state (the
    /// record's frame metadata) before validating a conditional write.
    #[must_use]
    pub(crate) fn record_version(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_id: &[u8; 16],
    ) -> Option<u64> {
        let map = self.record_versions.lock();
        let entry = map.get(&(pool, *collection_id))?;
        entry.nodes.get(node_id).copied().or(entry.deleted_at)
    }

    /// Seed a cold record's version from durable state (its frame metadata).
    /// Never overwrites a live entry, and a collection that was deleted
    /// resolves an absent record to its delete LSN rather than the seed.
    pub(crate) fn seed_record_version(
        &self,
        pool: ShardType,
        collection_id: [u8; 16],
        node_id: [u8; 16],
        lsn: u64,
    ) -> u64 {
        let mut map = self.record_versions.lock();
        let entry = map.entry((pool, collection_id)).or_default();
        if let Some(existing) = entry.nodes.get(&node_id) {
            return *existing;
        }
        if let Some(deleted) = entry.deleted_at {
            return deleted;
        }
        entry.nodes.insert(node_id, lsn);
        lsn
    }

    /// Snapshot the logical versions this pool's checkpoint can safely claim.
    /// Versions above `covered_lsn` remain represented by retained WAL and
    /// must not be folded into the checkpoint baseline.
    pub(crate) fn collection_versions_through(
        &self,
        pool: ShardType,
        covered_lsn: u64,
    ) -> Vec<([u8; 16], u64)> {
        let mut versions = self
            .collection_versions
            .lock()
            .iter()
            .filter_map(|(&(version_pool, collection_id), &version)| {
                (version_pool == pool && version <= covered_lsn).then_some((collection_id, version))
            })
            .collect::<Vec<_>>();
        versions.sort_unstable_by_key(|(collection_id, _)| *collection_id);
        versions
    }

    /// Merge a durable pool-checkpoint baseline with versions recovered from
    /// the retained WAL. Newer WAL versions win; deleted collection IDs are
    /// retained as tombstone versions too.
    pub(crate) fn restore_collection_versions(
        &self,
        pool: ShardType,
        versions: &[([u8; 16], u64)],
    ) {
        let mut current = self.collection_versions.lock();
        let mut floor = self.version_floor.lock();
        for &(collection_id, version) in versions {
            let entry = current.entry((pool, collection_id)).or_insert(0);
            *entry = (*entry).max(version);
            let baseline = floor.entry((pool, collection_id)).or_insert(0);
            *baseline = (*baseline).max(version);
        }
    }

    /// Version of a record known to be absent: its live entry or delete LSN,
    /// else the collection's checkpoint baseline, else `0`.
    #[must_use]
    pub(crate) fn absent_record_version(
        &self,
        pool: ShardType,
        collection_id: &[u8; 16],
        node_id: &[u8; 16],
    ) -> u64 {
        self.record_version(pool, collection_id, node_id)
            .or_else(|| {
                self.version_floor
                    .lock()
                    .get(&(pool, *collection_id))
                    .copied()
            })
            .unwrap_or(0)
    }

    /// Highest committed group `last_lsn` that carried at least one frame for
    /// `pool` and is fully in the packs, or 0 when there is none.
    ///
    /// A pool checkpoint must record this, not [`Self::committed_lsn`]. One
    /// shared group can interleave several pools' frames, so the global
    /// committed LSN can run ahead of what this pool's index has materialized;
    /// recording it would claim coverage the checkpoint does not have and let
    /// reclaim drop another pool's uncovered frames.
    #[must_use]
    pub fn committed_lsn_for_pool(&self, pool: ShardType) -> u64 {
        let committed = self.pool_committed.lock().get(&pool).copied().unwrap_or(0);
        // A published group, including an autocommit group, whose writes are
        // not all in the packs yet holds coverage below its first frame.
        match self.unmaterialized.lock().first() {
            Some(first_lsn) => committed.min(first_lsn.saturating_sub(1)),
            None => committed,
        }
    }

    /// Record that the group whose first frame is `first_lsn` has been fully
    /// written to the packs, so coverage may pass it.
    pub fn transaction_materialized(&self, first_lsn: u64) {
        self.unmaterialized.lock().remove(&first_lsn);
        self.notify_materialized();
    }

    /// Wake versioned reads waiting for a pending group to lift the read
    /// boundary. Called when a pool's materialized watermark or the pending
    /// set changes.
    pub(crate) fn notify_materialized(&self) {
        self.materialized_cv.notify_all();
    }

    /// Block until a materialization may have advanced the read boundary, or
    /// `timeout` elapses. A spurious wake is fine: callers re-check the
    /// boundary and wait again.
    pub(crate) fn wait_for_materialized(&self, timeout: std::time::Duration) {
        let mut guard = self.materialized_lock.lock();
        self.materialized_cv.wait_for(&mut guard, timeout);
    }

    /// Highest LSN a reader may treat as applied by every pool's index: one
    /// below the earliest published-but-not-yet-materialized group,
    /// or `u64::MAX` when nothing is pending.
    ///
    /// A pool-wide *materialized* watermark is a maximum, so out-of-order
    /// materialization can push it past an earlier group that is still pending.
    /// Clamping a versioned read to this floor keeps it from pairing data that
    /// predates a publication with the version that publication already
    /// advanced.
    #[must_use]
    pub(crate) fn unmaterialized_floor(&self) -> u64 {
        self.unmaterialized
            .lock()
            .first()
            .map_or(u64::MAX, |first_lsn| first_lsn.saturating_sub(1))
    }

    /// Record a durable commit and advance the generation that re-arms the
    /// background committer's threshold wake. Centralised so every commit path
    /// — background, explicit `sync_through`, and fallback — re-arms alike.
    fn note_commit(&self, through_lsn: u64) {
        let previous = self.committed_lsn.fetch_max(through_lsn, Ordering::AcqRel);
        if through_lsn > previous {
            let covered = through_lsn.saturating_sub(previous);
            self.commits.fetch_add(1, Ordering::Relaxed);
            self.commit_records.fetch_add(covered, Ordering::Relaxed);
            self.max_commit_records
                .fetch_max(covered, Ordering::Relaxed);
            self.commit_epoch.fetch_add(1, Ordering::AcqRel);
            self.wake_committer_if_backlogged();
        }
    }

    /// Advance per-pool committed watermarks for every appended group whose
    /// `last_lsn` is at or below `through_lsn`.
    fn promote_pool_committed(&self, through_lsn: u64) {
        let mut promotions = self.pending_promotions.lock();
        let mut watermarks = self.pool_committed.lock();
        let (ready, pending): (Vec<PendingPromotion>, Vec<PendingPromotion>) = promotions
            .drain(..)
            .partition(|(last_lsn, _)| *last_lsn <= through_lsn);
        *promotions = pending;
        for (_, extents) in ready {
            for (pool, lsn) in extents {
                let entry = watermarks.entry(pool).or_insert(0);
                *entry = (*entry).max(lsn);
            }
        }
    }

    /// Reclaim the longest prefix of the shared segment whose every group is
    /// durably covered by all of that group's own pools, returning the reclaim
    /// result or `None` when nothing is reclaimable yet.
    ///
    /// A group is reclaimed atomically, so it may only be dropped once every
    /// pool that contributed a frame to it has reported coverage through the
    /// group's `last_lsn`. The pools in one group are independent of the pools
    /// in another, so the reclaimable prefix is **not** a single minimum across
    /// every pool: an idle pool whose last frame sits in an early group must not
    /// pin later groups that its frames never reached. Walk the live groups in
    /// order and stop at the first one any contributing pool has not covered.
    ///
    /// # Errors
    /// Propagates a scan or segment-rewrite failure from [`Self::reclaim_through`].
    pub fn reclaim_shared(&self) -> io::Result<Option<Reclaim>> {
        // The group directory answers this without reading the segment. The
        // lock order matches `reclaim_through`: the sync lock, then the journal,
        // and coverage last.
        {
            let _sync = self.sync_lock.lock();
            // Rewriting fsyncs the retained suffix, which can include groups
            // that were published but are not yet durable. Run the pre-sync
            // hook first, outside the journal lock (it can wait on derived-store
            // writers that are themselves waiting to publish), so their derived
            // rows are durable before the groups can be.
            let hooked_through = self.published_lsn();
            let had_hook = self.run_pre_sync_hook()?;
            let mut journal = self.journal.lock();
            self.revoke_over_cap_leases(&journal);
            let boundary = {
                let coverage = self.coverage.lock();
                journal.shared_reclaim_boundary(&coverage.covered)
            };
            match boundary {
                SharedBoundary::Through(covered_lsn, blocked_by) => {
                    if had_hook && journal.next_lsn.saturating_sub(1) > hooked_through {
                        // A group landed after the hook ran; rewriting would make
                        // it durable through an unhooked fsync.
                        return Ok(Some(journal.skipped_reclaim()));
                    }
                    // Shared reclaim computes its own per-group/per-pool
                    // boundary, so apply the replay floor after that decision
                    // and before the segment rewrite. The lease is global to
                    // this segment and therefore protects groups from every
                    // pool, including groups whose pool coverage is complete.
                    let replay_limit = self.replay_reclaim_limit(covered_lsn);
                    let lease_limited = replay_limit < covered_lsn;
                    let blocked_by = (!lease_limited).then_some(blocked_by).flatten();
                    let reclaimed = journal.reclaim_through_with_blocker(replay_limit, blocked_by);
                    if reclaimed.is_ok() {
                        self.bump_publish_signal();
                    }
                    let reclaimed = reclaimed.map(Some);
                    if lease_limited {
                        self.record_replay_lease_outcome(&journal);
                    } else {
                        // A cut held back only by an active replay lease is not a
                        // pool-coverage stall. Recording one would drive forced
                        // checkpoints and warning logs for as long as a rebuild
                        // pins its own window; the next reclaim after the lease
                        // is dropped records the real outcome.
                        let coverage = self.coverage.lock();
                        self.record_reclaim_outcome(&journal, &coverage.covered);
                        *self.replay_lease_stall.lock() = None;
                    }
                    return reclaimed;
                }
                SharedBoundary::NothingCovered => {
                    let coverage = self.coverage.lock();
                    self.record_reclaim_outcome(&journal, &coverage.covered);
                    return Ok(None);
                }
                SharedBoundary::Untrusted => {}
            }
        }
        // The directory did not match the file: decide from a scan of it.
        let scan = Journal::scan_read_only(&self.path)?;
        let marks = marks_from_scan(&scan);
        let boundary = {
            let coverage = self.coverage.lock();
            boundary_through(&marks, &coverage.covered)
        };
        match boundary {
            Some((covered_lsn, blocked_by)) => self
                .reclaim_through_with_blocker(covered_lsn, blocked_by)
                .map(Some),
            None => Ok(None),
        }
    }

    /// The pools whose missing coverage holds reclaim back right now, oldest
    /// needed group first. Empty when nothing is blocked or it cannot be told.
    #[must_use]
    pub fn reclaim_blockers(&self) -> Vec<ShardType> {
        let journal = self.journal.lock();
        let coverage = self.coverage.lock();
        journal.blocking_pools(&coverage.covered)
    }

    /// Whether the last reclaim left the segment over the trigger.
    #[must_use]
    pub fn is_reclaim_stalled(&self) -> bool {
        self.reclaim_stall.lock().is_some()
    }

    /// Times a reclaim has been found unable to shrink a segment over the
    /// trigger (each stall counts once, until a reclaim clears it).
    #[must_use]
    pub fn reclaim_stalls(&self) -> u64 {
        self.reclaim_stalls.load(Ordering::Relaxed)
    }

    /// Give the coordinator a way to make one named pool checkpoint (and so
    /// report coverage). It is used only in the emergency zone, to un-stall a
    /// reclaim held back by a pool nobody is syncing.
    pub fn set_blocker_remediation(&self, remediate: impl Fn(ShardType) + Send + Sync + 'static) {
        *self.blocker_remediation.lock() = Some(Arc::new(remediate));
    }

    /// Note how a reclaim ended, with the journal and sync locks held. A
    /// segment still over the trigger means the reclaim is stalled: remember
    /// it, and say so once, naming the pools it is waiting on.
    fn record_reclaim_outcome(&self, journal: &Journal, covered: &HashMap<ShardType, u64>) {
        let len = journal.file_len;
        if len <= self.reclaim_trigger_len() {
            *self.reclaim_stall.lock() = None;
            *self.last_remediation.lock() = None;
            return;
        }
        let epoch = self.coverage_epoch.load(Ordering::Acquire);
        let mut stall = self.reclaim_stall.lock();
        let previous = *stall;
        let emergency = len >= emergency_len_for_cap(journal.segment_cap);
        // The first reclaim to leave the segment over the trigger is normal:
        // pools sync one after another, so the first to cross it cannot yet
        // reclaim what the others have not reported. It is only worth saying
        // once the segment has grown by an eighth of the trigger without a
        // reclaim getting it back under the trigger, or it reaches the
        // emergency zone. The
        // back-off and remediation below act on the stall from the start.
        let grace_from = previous.map_or(len, |old| old.grace_from);
        let already_reported = previous.is_some_and(|old| old.reported);
        let report = !already_reported
            && (emergency || len >= grace_from.saturating_add(self.reclaim_retry_growth()));
        *stall = Some(ReclaimStall {
            at_len: len,
            coverage_epoch: epoch,
            grace_from,
            reported: already_reported || report,
        });
        let entered_emergency = emergency
            && previous.is_some_and(|old| old.at_len < emergency_len_for_cap(journal.segment_cap));
        if report {
            self.reclaim_stalls.fetch_add(1, Ordering::Relaxed);
        }
        if report || (already_reported && entered_emergency) {
            eprintln!(
                "warning: shared WAL reclaim is stalled at {len} of {} bytes (trigger {}); \
                 waiting on pools {:?} to report coverage",
                journal.segment_cap,
                self.reclaim_trigger_len(),
                journal.blocking_pools(covered)
            );
        }
    }

    fn record_replay_lease_outcome(&self, journal: &Journal) {
        let len = journal.file_len;
        if len <= self.reclaim_trigger_len() {
            *self.replay_lease_stall.lock() = None;
            return;
        }
        let emergency = len >= emergency_len_for_cap(journal.segment_cap);
        let mut stall = self.replay_lease_stall.lock();
        let previous = *stall;
        let grace_from = previous.map_or(len, |old| old.grace_from);
        *stall = Some(ReclaimStall {
            at_len: len,
            coverage_epoch: self.coverage_epoch.load(Ordering::Acquire),
            grace_from,
            reported: previous.is_some_and(|old| old.reported) || emergency,
        });
        if emergency && !previous.is_some_and(|old| old.reported) {
            eprintln!(
                "warning: shared WAL reclaim is held by an active replay lease at {len} of {} bytes",
                journal.segment_cap
            );
        }
    }

    /// If a stalled reclaim has reached the emergency zone, make the pools
    /// holding it back checkpoint, except `caller`, which is mid-sync. Runs
    /// with no journal or persistence lock held, at most once per further
    /// eighth of the trigger of growth unless coverage moved, and never from
    /// inside another remediation.
    pub fn remediate_blockers(&self, caller: Option<ShardType>) {
        if REMEDIATING.with(std::cell::Cell::get) {
            return;
        }
        let len = self.segment_len();
        if len < self.emergency_len() || !self.is_reclaim_stalled() {
            return;
        }
        let epoch = self.coverage_epoch.load(Ordering::Acquire);
        {
            let mut last = self.last_remediation.lock();
            if let Some((last_len, last_epoch)) = *last {
                if last_epoch == epoch && len < last_len.saturating_add(self.reclaim_retry_growth())
                {
                    return;
                }
            }
            *last = Some((len, epoch));
        }
        let Some(remediate) = self.blocker_remediation.lock().clone() else {
            return;
        };
        REMEDIATING.with(|flag| flag.set(true));
        for pool in self.reclaim_blockers() {
            if Some(pool) != caller {
                remediate(pool);
            }
        }
        REMEDIATING.with(|flag| flag.set(false));
    }

    /// The error for a commit refused because the segment is full, naming the
    /// pools reclaim is waiting on when that is known.
    fn explain_full_segment(&self, error: io::Error, journal: &Journal) -> io::Error {
        if error.kind() == io::ErrorKind::WouldBlock {
            let blockers = journal.blocking_pools(&self.coverage.lock().covered);
            if !blockers.is_empty() {
                return io::Error::new(
                    io::ErrorKind::WouldBlock,
                    format!("{error}; reclaim is waiting on pools {blockers:?} to report coverage"),
                );
            }
        }
        error
    }

    /// Build a coordinator whose groups draw their sequence from a shared,
    /// cross-pool counter instead of this segment's own numbering.
    ///
    /// The caller must initialize `sequence` above every participating
    /// segment's recovered next sequence (see [`Journal::next_sequence`]), so
    /// the first allocation is larger than any sequence already on disk. Gaps
    /// in a single segment's group sequence are then expected; recovery
    /// permits them.
    #[must_use]
    pub fn with_shared_sequence(journal: Journal, scan: &Scan, sequence: Arc<AtomicU64>) -> Self {
        let mut coordinator = Self::new(journal, scan);
        coordinator.sequence = Some(sequence);
        coordinator
    }

    /// Highest durably committed LSN. Everything at or below this is on disk.
    #[must_use]
    pub fn committed_lsn(&self) -> u64 {
        self.committed_lsn.load(Ordering::Acquire)
    }

    /// Create a replay cursor at a materialized LSN for this journal.
    ///
    /// The caller must establish that the snapshot contains the records through
    /// `lsn` before creating the cursor. The cursor is scoped to this journal
    /// path and writer incarnation. It must be reacquired after a writer
    /// restart; if reclaim has already removed required history,
    /// [`Self::changes_since`] reports an expired-cursor error.
    ///
    /// # Errors
    /// Returns `Unsupported` when no publish signal is installed.
    pub(crate) fn replay_cursor(&self, lsn: u64) -> io::Result<JournalCursor> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay cursors require an installed publish signal",
            )
        })?;
        let (base_lsn, resume_offset) = {
            let journal = self.journal.lock();
            (journal.base_lsn, journal.resume_offset_for(lsn))
        };
        Ok(JournalCursor {
            path: self.path.clone(),
            incarnation: signal.snapshot().0,
            lsn,
            base_lsn,
            resume_offset,
        })
    }

    /// Pin the journal history required by `cursor` until the returned lease
    /// is dropped. The pin and reclaim share the `sync_lock` -> `journal` lock
    /// order, so reclaim either observes this lease or completes first and
    /// causes an explicit expired-cursor error here.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a different segment, `Unsupported` without a
    /// publish signal, `InvalidData` for a previous writer incarnation or
    /// reclaimed history, and `Other` if lease IDs are exhausted.
    pub(crate) fn pin_replay_cursor(
        &self,
        cursor: &JournalCursor,
    ) -> io::Result<JournalReplayLease> {
        if cursor.path != self.path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal cursor belongs to a different segment",
            ));
        }
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay leases require an installed publish signal",
            )
        })?;

        let _sync = self.sync_lock.lock();
        let journal = self.journal.lock();
        if signal.snapshot().0 != cursor.incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal cursor belongs to a previous writer incarnation",
            ));
        }
        if cursor.lsn.saturating_add(1) < journal.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{CURSOR_EXPIRED_PREFIX}{}; retained history starts at {}",
                    cursor.lsn, journal.base_lsn
                ),
            ));
        }

        let mut next = self.next_replay_lease.load(Ordering::Relaxed);
        let id = loop {
            let following = next
                .checked_add(1)
                .ok_or_else(|| io::Error::other("journal replay lease IDs exhausted"))?;
            match self.next_replay_lease.compare_exchange_weak(
                next,
                following,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break next,
                Err(observed) => next = observed,
            }
        };
        self.replay_leases.pins.lock().insert(
            id,
            LeasePin {
                lsn: cursor.lsn,
                revoked: false,
                created: std::time::Instant::now(),
            },
        );
        Ok(JournalReplayLease {
            registry: Arc::clone(&self.replay_leases),
            id,
        })
    }

    /// Highest LSN reclaim may pass without expiring an active replay cursor.
    /// Revoked leases no longer constrain the cut.
    fn replay_reclaim_limit(&self, covered_lsn: u64) -> u64 {
        self.replay_leases
            .pins
            .lock()
            .values()
            .filter(|pin| !pin.revoked)
            .map(|pin| pin.lsn)
            .min()
            .map_or(covered_lsn, |pin| covered_lsn.min(pin))
    }

    /// Set the optional cap, in bytes, on the history one replay lease may
    /// retain. `None` keeps retention unlimited, matching `PostgreSQL`'s default
    /// for `max_slot_wal_keep_size`. See `retained_bytes_from` for the measured
    /// range and `revoke_over_cap_leases` for the selection.
    pub(crate) fn configure_replay_lease_cap(&self, cap: Option<u64>) {
        self.replay_lease_cap
            .store(cap.unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    /// Replay leases revoked by the retention cap.
    #[must_use]
    pub fn replay_leases_revoked(&self) -> u64 {
        self.replay_leases_revoked.load(Ordering::Relaxed)
    }

    /// Number of replay leases currently pinning history.
    #[must_use]
    pub fn replay_lease_count(&self) -> usize {
        self.replay_leases
            .pins
            .lock()
            .values()
            .filter(|pin| !pin.revoked)
            .count()
    }

    /// Oldest pinned LSN among active replay leases, or `None` when none are
    /// active.
    #[must_use]
    pub fn oldest_replay_lease_lsn(&self) -> Option<u64> {
        self.replay_leases
            .pins
            .lock()
            .values()
            .filter(|pin| !pin.revoked)
            .map(|pin| pin.lsn)
            .min()
    }

    /// Bytes an active replay lease currently prevents reclaiming, measured
    /// from the oldest pin to the end of the segment. `None` when no lease is
    /// active.
    #[must_use]
    pub fn replay_lease_lag_bytes(&self) -> Option<u64> {
        let journal = self.journal.lock();
        let oldest = self.oldest_replay_lease_lsn()?;
        Some(journal.retained_bytes_from(oldest))
    }

    /// Age of the oldest active replay lease, or `None` when none are active.
    #[must_use]
    pub fn oldest_replay_lease_age(&self) -> Option<Duration> {
        self.replay_leases
            .pins
            .lock()
            .values()
            .filter(|pin| !pin.revoked)
            .map(|pin| pin.created.elapsed())
            .max()
    }

    /// Revoke every active lease whose retained suffix exceeds `cap`. Each
    /// lease is measured independently from its own pin, so a newer lease can
    /// survive while an older one is revoked. A lease retaining exactly `cap`
    /// is kept. Runs with the journal lock held, so it sees a stable segment
    /// length and shares the reclaim lock order `journal -> pins`.
    fn revoke_over_cap_leases(&self, journal: &Journal) {
        let cap = self.replay_lease_cap.load(Ordering::Relaxed);
        if cap == u64::MAX {
            return;
        }
        let mut pins = self.replay_leases.pins.lock();
        for pin in pins.values_mut() {
            if pin.revoked || journal.retained_bytes_from(pin.lsn) <= cap {
                continue;
            }
            pin.revoked = true;
            self.replay_leases_revoked.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Read a bounded page of complete, durable groups after `cursor`.
    ///
    /// Pages preserve group boundaries and journal order. The durable horizon
    /// is sampled before the segment is read and returned as
    /// [`JournalChangesPage::horizon_lsn`]: `has_more` and the returned groups
    /// are complete with respect to that horizon, so a group that becomes
    /// durable after the sample is never silently skipped. Reclaim is allowed
    /// to remove history; if it has advanced past the cursor, this returns an
    /// `InvalidData` error so the caller can discard its partial rebuild and
    /// take a fresh snapshot. This method does not retain a lease across page
    /// calls. Hold a [`JournalReplayLease`] from the snapshot cursor while
    /// paging if reclaim must not expire it during a long replay.
    ///
    /// `limit` bounds the number of groups, not entries or bytes. Shared-journal
    /// pages include all pools and collections; callers filter the entries
    /// they need.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a foreign cursor or zero page size,
    /// `Unsupported` without a publish signal, and `InvalidData` when the writer
    /// incarnation differs, required history has been reclaimed, or reclaim
    /// changes the segment's base LSN during the read. A concurrent reclaim
    /// may allow retrying the same cursor if its history remains available.
    /// Also propagates segment I/O and validation errors.
    pub fn changes_since(
        &self,
        cursor: &JournalCursor,
        limit: usize,
    ) -> io::Result<JournalChangesPage> {
        self.validate_changes_request(cursor, limit)?;
        let incarnation = self.replay_incarnation(cursor)?;

        // Sample the durable high-water mark *before* reading the WAL. A group
        // at or below it was appended and fsynced before this read, so the tail
        // scan below cannot miss it. Sampling after the read could include a
        // group appended in between that the scan never saw, letting the page
        // report `has_more == false` with a durable group outstanding.
        let durable_lsn = self.committed_lsn();

        // Capture the directory boundary and the byte offset needed for at most
        // `limit + 1` durable groups. The extra group determines `has_more`.
        // Holding the journal lock makes these offsets consistent with reclaim
        // and append; the actual I/O happens after releasing it.
        let window = self.changes_window(cursor, limit, durable_lsn)?;

        // A test can force a commit after the horizon and byte range are fixed
        // but before the WAL is read. That group belongs to a later pass.
        #[cfg(test)]
        if let Some(hook) = take_changes_since_before_read_hook() {
            hook();
        }

        let (scan, resumed) = self.scan_changes(cursor, &window)?;
        self.validate_changes_scan(cursor, incarnation, &window, &scan)?;
        if resumed {
            self.replay_page_stats
                .resumed_pages
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(self.changes_page(cursor, incarnation, durable_lsn, limit, scan))
    }

    /// Validate a [`Self::changes_since`] request's cursor and page size.
    fn validate_changes_request(&self, cursor: &JournalCursor, limit: usize) -> io::Result<()> {
        if cursor.path != self.path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal cursor belongs to a different segment",
            ));
        }
        if limit == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "journal changes page size must be nonzero",
            ));
        }
        Ok(())
    }

    /// The writer incarnation a [`Self::changes_since`] cursor must match, or an
    /// error when no publish signal is installed or the cursor is from a
    /// previous writer.
    fn replay_incarnation(&self, cursor: &JournalCursor) -> io::Result<u64> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay requires an installed publish signal",
            )
        })?;
        let incarnation = signal.snapshot().0;
        if cursor.incarnation != incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal cursor belongs to a previous writer incarnation",
            ));
        }
        Ok(incarnation)
    }

    /// Capture the [`ChangesWindow`] for a page: the directory-derived byte
    /// range through at most `limit + 1` durable groups, the base LSN to guard
    /// against a concurrent reclaim, and whether the in-memory directory
    /// matched the file.
    fn changes_window(
        &self,
        cursor: &JournalCursor,
        limit: usize,
        durable_lsn: u64,
    ) -> io::Result<ChangesWindow> {
        let journal = self.journal.lock();
        if cursor.lsn.saturating_add(1) < journal.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{CURSOR_EXPIRED_PREFIX}{}; retained history starts at {}",
                    cursor.lsn, journal.base_lsn
                ),
            ));
        }
        let first = journal
            .groups
            .partition_point(|group| group.last_lsn <= cursor.lsn);
        let mut durable_groups = journal
            .groups
            .iter()
            .skip(first)
            .take_while(|group| group.last_lsn <= durable_lsn);
        let last_durable_group = durable_groups.by_ref().take(limit.saturating_add(1)).last();
        let end = last_durable_group
            .as_ref()
            .map_or(cursor.resume_offset, |group| group.end_offset);
        let expected_lsn = journal
            .groups
            .get(first)
            .map_or(journal.next_lsn, |group| group.first_lsn);
        Ok(ChangesWindow {
            base_lsn: journal.base_lsn,
            end,
            expected_lsn,
            has_durable_group: last_durable_group.is_some(),
            directory_trusted: journal.directory_matches_file() && !changes_since_force_untrusted(),
        })
    }

    /// Read the segment region a [`Self::changes_since`] page needs, resuming at
    /// the cursor's own byte offset so the retained prefix is never re-decoded.
    /// The directory only caps the read; when it is untrusted the read runs to
    /// the end of the segment.
    ///
    /// A resume offset is invalidated by a reclaim rewrite, a replacement, or a
    /// rotation. Any failure therefore falls back to a full scan, and
    /// [`Self::validate_changes_scan`] decides whether the cursor survived.
    fn scan_changes(
        &self,
        cursor: &JournalCursor,
        window: &ChangesWindow,
    ) -> io::Result<(Scan, bool)> {
        let end = if window.directory_trusted {
            window.end
        } else {
            u64::MAX
        };
        // A reclaim rewrites the retained prefix, moving every surviving group,
        // so the cursor's offset can point past the new end. That read succeeds
        // with no groups even though the window says groups follow; treat it as
        // invalid and rebuild. An empty read is otherwise trusted when the
        // window itself holds no group past the offset, or when the read stopped
        // exactly at the cursor's offset with the segment base unchanged — a
        // legitimately idle tail rather than a moved one.
        if let Ok(scan) = Journal::scan_read_only_range(
            &self.path,
            cursor.resume_offset,
            end,
            window.expected_lsn,
        ) {
            let idle_tail =
                scan.valid_len == cursor.resume_offset && scan.base_lsn == cursor.base_lsn;
            if !scan.groups.is_empty()
                || (!window.has_durable_group && end == cursor.resume_offset)
                || idle_tail
            {
                return Ok((scan, true));
            }
        }
        self.replay_page_stats
            .full_scan_fallbacks
            .fetch_add(1, Ordering::Relaxed);
        note_changes_since_full_scan();
        Journal::scan_read_only(&self.path).map(|scan| (scan, false))
    }

    /// Re-check the writer incarnation and the reclaim guard after the WAL read,
    /// so a restart or in-place reclaim between capture and read is not mistaken
    /// for stable history.
    fn validate_changes_scan(
        &self,
        cursor: &JournalCursor,
        incarnation: u64,
        window: &ChangesWindow,
        scan: &Scan,
    ) -> io::Result<()> {
        let signal = self.publish_signal().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "replay requires an installed publish signal",
            )
        })?;
        if signal.snapshot().0 != incarnation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "journal writer restarted while reading changes",
            ));
        }
        if scan.base_lsn != window.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                SEGMENT_RECLAIMED_MESSAGE,
            ));
        }
        if cursor.lsn.saturating_add(1) < scan.base_lsn {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{CURSOR_EXPIRED_PREFIX}{}; retained history starts at {}",
                    cursor.lsn, scan.base_lsn
                ),
            ));
        }
        Ok(())
    }

    /// Select the durable groups after `cursor` into a page capped at `limit`,
    /// and derive the next cursor and `has_more` from one extra group.
    fn changes_page(
        &self,
        cursor: &JournalCursor,
        incarnation: u64,
        durable_lsn: u64,
        limit: usize,
        scan: Scan,
    ) -> JournalChangesPage {
        let base_lsn = scan.base_lsn;
        let mut following = scan
            .groups
            .into_iter()
            .zip(scan.group_ends)
            .filter(|(group, _)| group.last_lsn > cursor.lsn && group.last_lsn <= durable_lsn);
        let mut selected = Vec::new();
        let mut resume_offset = cursor.resume_offset;
        while selected.len() < limit {
            match following.next() {
                Some((group, end)) => {
                    resume_offset = end;
                    selected.push(group);
                }
                None => break,
            }
        }
        let has_more = following.next().is_some();
        let through_lsn = selected.last().map_or(cursor.lsn, |group| group.last_lsn);
        JournalChangesPage {
            groups: selected,
            next_cursor: JournalCursor {
                path: self.path.clone(),
                incarnation,
                lsn: through_lsn,
                base_lsn,
                resume_offset,
            },
            through_lsn,
            has_more,
            horizon_lsn: durable_lsn,
        }
    }

    /// Path of the journal segment used by this coordinator.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// Highest LSN whose journal group is complete (commit trailer appended),
    /// whether or not it has been fsynced.
    ///
    /// A read-only overlay may observe everything at or below this. Durability
    /// is only promised by [`Self::committed_lsn`]: a crash before the fsync
    /// may lose a visible-but-uncommitted group.
    #[must_use]
    pub fn visible_lsn(&self) -> u64 {
        self.visible_lsn.load(Ordering::Acquire)
    }

    /// Highest published (assigned) LSN, committed or not.
    #[must_use]
    pub fn published_lsn(&self) -> u64 {
        self.published_lsn.load(Ordering::Acquire)
    }

    /// Install (or clear) the work to make durable before each WAL group's
    /// fsync. See [`PreSyncHook`].
    pub fn set_pre_sync_hook(&self, hook: Option<PreSyncHook>) {
        *self.pre_sync_hook.lock() = hook;
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
    /// `sync_lock` serializes the fsync (the journal mutex is held only briefly
    /// to capture the range, so publishers keep appending); concurrent sync
    /// callers wait for it and then recheck the committed LSN before deciding
    /// whether to flush.
    ///
    /// # Errors
    /// Returns an error if the target was never published, the journal is
    /// poisoned, or writing/syncing the covering group fails.
    pub fn sync_through(&self, target_lsn: u64) -> io::Result<Option<CommitReceipt>> {
        self.sync_through_timed(target_lsn)
            .map(|(receipt, _)| receipt)
    }

    /// Durably commit pending mutations and report where the time went.
    ///
    /// # Errors
    /// Returns an I/O error if the target is invalid, the journal is poisoned,
    /// or appending/fsyncing the group fails.
    pub fn sync_through_timed(
        &self,
        target_lsn: u64,
    ) -> io::Result<(Option<CommitReceipt>, JournalSyncTimings)> {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        let mut timings = JournalSyncTimings::default();
        let in_flight = self
            .sync_in_flight
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        timings.journal_in_flight = in_flight;
        let _in_flight_guard = SyncInFlightGuard(&self.sync_in_flight);
        if target_lsn > self.published_lsn.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sync target has not been published",
            ));
        }
        if self.covered_by_prior_commit(target_lsn, &mut timings) {
            return Ok((None, timings));
        }
        self.reject_if_poisoned()?;

        let _sync = self.acquire_sync_lock(&mut timings);
        self.reject_if_poisoned()?;
        // A concurrent fsync that finished while we waited usually covered us.
        if self.covered_by_prior_commit(target_lsn, &mut timings) {
            return Ok((None, timings));
        }
        let previously_committed = self.committed_lsn.load(Ordering::Acquire);

        let capture = self.capture_sync_range(target_lsn)?;
        self.flush_captured_file(&capture, &mut timings)?;
        let receipt = self.record_durable_range(previously_committed, &capture, &mut timings);
        Ok((Some(receipt), timings))
    }

    /// Whether an earlier commit already made `target_lsn` durable. A `0`
    /// target is always covered. Records the coalescing in `timings` and the
    /// coordinator's counter when it is.
    fn covered_by_prior_commit(&self, target_lsn: u64, timings: &mut JournalSyncTimings) -> bool {
        if target_lsn <= self.committed_lsn.load(Ordering::Acquire) {
            timings.journal_coalesced = true;
            self.coalesced_syncs.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Record that a published group could not be applied to the packs and
    /// index. The WAL holds it, so the data is not lost, but this process's
    /// index no longer matches the journal: new publications and versioned
    /// reads that would wait for the group fail with `cause` until the database
    /// is reopened and the WAL replayed. Only the first cause is kept.
    pub(crate) fn fail_apply(&self, cause: String) {
        let publication = self.publication.lock();
        let _ = self.apply_failure.set(cause);
        drop(publication);
        // Wake reads waiting for the group so they see the failure.
        self.notify_materialized();
    }

    /// The reason a published group failed to apply, if one did.
    pub(crate) fn apply_failure(&self) -> Option<&str> {
        self.apply_failure.get().map(String::as_str)
    }

    /// Fail if a published group could not be applied (see [`Self::fail_apply`]).
    fn reject_if_apply_failed(&self) -> io::Result<()> {
        match self.apply_failure() {
            Some(cause) => Err(io::Error::other(format!(
                "a published group failed to apply; reopen the database to \
                 replay the WAL: {cause}"
            ))),
            None => Ok(()),
        }
    }

    /// Fail if a failed commit or append poisoned the journal.
    fn reject_if_poisoned(&self) -> io::Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed commit",
            ));
        }
        Ok(())
    }

    /// Take [`Self::sync_lock`], counting this caller as a waiter when another
    /// fsync is in flight, and record how long the wait took.
    fn acquire_sync_lock(&self, timings: &mut JournalSyncTimings) -> MutexGuard<'_, ()> {
        let lock_started = std::time::Instant::now();
        let guard = if let Some(guard) = self.sync_lock.try_lock() {
            guard
        } else {
            timings.journal_waiter = true;
            self.journal_waiters.fetch_add(1, Ordering::Relaxed);
            self.sync_lock.lock()
        };
        timings.journal_lock_wait = lock_started.elapsed();
        guard
    }

    /// Capture what a sync needs, under the journal lock.
    ///
    /// Every published group is already complete in the segment, so the
    /// sync only needs to make its bytes durable. Under the journal lock,
    /// capture the newest visible LSN (everything at or below it has been
    /// fully written), the group sequence, and a handle to the current
    /// file; then release the lock so publishers run during the fsync.
    /// Syncing through the newest visible LSN, not just `target_lsn`, lets
    /// one fsync cover every caller that queued behind the previous one.
    fn capture_sync_range(&self, target_lsn: u64) -> io::Result<SyncCapture> {
        let mut journal = self.journal.lock();
        if journal.poisoned {
            return Err(io::Error::other(
                "journal handle is poisoned after an earlier failed commit",
            ));
        }
        let through_lsn = self.visible_lsn.load(Ordering::Acquire);
        if target_lsn > through_lsn {
            return Err(io::Error::other(
                "published sync target is not yet visible in the journal",
            ));
        }
        let file = match journal.file.try_clone() {
            Ok(file) => file,
            Err(error) => {
                self.poisoned.store(true, Ordering::Release);
                journal.poisoned = true;
                return Err(error);
            }
        };
        Ok(SyncCapture {
            file,
            through_lsn,
            sequence: journal.next_sequence.saturating_sub(1),
            path: journal.path.clone(),
            file_len: journal.file_len,
        })
    }

    /// Run the installed [`PreSyncHook`] before WAL durability work. Returns
    /// whether a hook was installed.
    ///
    /// Callers must not hold the journal lock: the hook syncs derived stores
    /// whose writers can be waiting on the journal while holding those stores'
    /// locks.
    fn run_pre_sync_hook(&self) -> io::Result<bool> {
        let hook = self.pre_sync_hook.lock().clone();
        if let Some(hook) = hook {
            hook()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Fsync the captured handle with no journal lock held, recording the
    /// fsync time in `timings`.
    fn flush_captured_file(
        &self,
        capture: &SyncCapture,
        timings: &mut JournalSyncTimings,
    ) -> io::Result<()> {
        // The boundary is already captured and no journal lock is held, so the
        // hook's work covers everything published up to that boundary while
        // publishers carry on past it.
        self.run_pre_sync_hook()?;
        let fsync_started = std::time::Instant::now();
        #[cfg(test)]
        let _trace = crate::test_trace::span("wal.group_fsync", std::path::Path::new("wal"));
        #[cfg(test)]
        let hooked = self.run_fsync_hook();
        #[cfg(not(test))]
        let hooked: io::Result<()> = Ok(());
        // A group commit only appends to an already-published segment, so
        // `sync_data` is sufficient: it persists the appended bytes and the
        // file size they live beyond, which is the metadata a later read needs.
        // The creation, truncation and replacement paths still use `sync_all`
        // (see `open_shared_with_base` and `install_rebuilt`), because those
        // change the inode's publication state. Skipping unrelated metadata
        // here removes work without changing the barrier count.
        if let Err(error) = hooked.and_then(|()| capture.file.sync_data()) {
            // Never retry a failed fsync and report success: the kernel may
            // have dropped the dirty pages. Poison until reopen and rescan.
            self.poisoned.store(true, Ordering::Release);
            self.journal.lock().poisoned = true;
            return Err(error);
        }
        timings.journal_fsync = fsync_started.elapsed();
        Ok(())
    }

    /// Run the test-only fsync hook, if one is set, so a test can park the
    /// sync before its fsync or make it fail.
    #[cfg(test)]
    fn run_fsync_hook(&self) -> io::Result<()> {
        let hook = self.fsync_hook.lock().clone();
        match hook {
            Some(hook) => hook(),
            None => Ok(()),
        }
    }

    /// Publish the durable boundary a successful fsync established and build
    /// the receipt for the range this call advanced.
    fn record_durable_range(
        &self,
        previously_committed: u64,
        capture: &SyncCapture,
        timings: &mut JournalSyncTimings,
    ) -> CommitReceipt {
        timings.journal_records = capture.through_lsn.saturating_sub(previously_committed);
        self.warn_if_slow_fsync(capture.path.as_path(), capture.through_lsn, timings);
        self.note_commit(capture.through_lsn);
        self.promote_pool_committed(capture.through_lsn);
        // The fsync has returned, so every byte up to the captured length is
        // durable. Record that, after the fact and without an fsync of its own,
        // so recovery can tell a crash tail above it from corruption below it.
        self.journal.lock().write_mark(capture.file_len);
        // The groups were appended at publication, so this fsync wrote no
        // bytes: the receipt reports the durable range this call advanced.
        CommitReceipt {
            sequence: capture.sequence,
            first_lsn: previously_committed.saturating_add(1),
            last_lsn: capture.through_lsn,
            bytes_written: 0,
        }
    }

    /// Report an fsync slower than [`SLOW_FSYNC_WARN`] as it happens, with the
    /// lock wait and waiter count needed to tell contention from disk latency.
    fn warn_if_slow_fsync(&self, path: &Path, target_lsn: u64, timings: &JournalSyncTimings) {
        if timings.journal_fsync >= SLOW_FSYNC_WARN {
            let _ = writeln!(
                io::stderr().lock(),
                "mtxdb: slow WAL fsync {}ms (through lsn {target_lsn}, lock wait {}ms, {} records, in-flight {}, waiters {}, {})",
                timings.journal_fsync.as_millis(),
                timings.journal_lock_wait.as_millis(),
                timings.journal_records,
                timings.journal_in_flight,
                self.journal_waiters.load(Ordering::Relaxed),
                path.display(),
            );
        }
    }

    /// Publish one autocommit mutation as its own complete group, without
    /// fsyncing it.
    ///
    /// The LSN is assigned under the publication lock and the group is
    /// visible to read-only overlays when this returns. Durability is
    /// separate: a later [`Self::sync_through`] or the background committer
    /// fsyncs every visible group at once, so many small groups still share
    /// one fsync.
    /// A caller applying the mutations directly must call
    /// [`Self::transaction_materialized`] with `receipt.first_lsn` after all
    /// writes reach the packs; until then this group holds back WAL reclaim.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned, its LSN space is
    /// exhausted, or the append fails. A partial append poisons the
    /// underlying journal; publication is rejected until reopen/recovery.
    pub fn publish_group(&self, mutations: &[Mutation]) -> io::Result<CommitReceipt> {
        self.publish_groups(&[(ShardType::State, mutations)], &[], &[])?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cannot publish an empty group")
            })
    }

    /// Like [`Self::publish_group`], tagging every mutation with `pool`.
    /// Required when the coordinator's segment is pool-tagged; the pool-less
    /// form would be rejected there.
    /// The direct applier must call [`Self::transaction_materialized`] with
    /// `receipt.first_lsn` after all writes reach the packs, or the group will
    /// continue to hold back WAL reclaim.
    ///
    /// # Errors
    /// Same as [`Self::publish_group`].
    pub fn publish_group_tagged(
        &self,
        pool: ShardType,
        mutations: &[Mutation],
    ) -> io::Result<CommitReceipt> {
        self.publish_groups(&[(pool, mutations)], &[], &[])?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cannot publish an empty group")
            })
    }

    /// Like [`Self::publish_group_tagged`], validating `expectations` inside the
    /// publication critical section. A mismatch rejects the commit with an
    /// [`io::ErrorKind::WouldBlock`] error wrapping a `StaleVersion` instead
    /// of appending, so no group is published.
    /// The direct applier must call [`Self::transaction_materialized`] with
    /// `receipt.first_lsn` after all writes reach the packs, or the group will
    /// continue to hold back WAL reclaim.
    ///
    /// # Errors
    /// Returns a stale-version error if a collection's logical version differs
    /// from its expectation, or the usual publication errors.
    pub fn publish_group_tagged_checked(
        &self,
        pool: ShardType,
        mutations: &[Mutation],
        expectations: &[CollectionExpectation],
    ) -> io::Result<CommitReceipt> {
        self.publish_groups(&[(pool, mutations)], expectations, &[])?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "cannot publish an empty group")
            })
    }

    /// Publish several pool-tagged transaction batches as one visible journal
    /// group.
    ///
    /// This is the atomic publication boundary for a transaction that writes
    /// multiple pools through a shared coordinator. The caller must provide
    /// each pool at most once; empty batches are ignored.
    /// The direct applier must call [`Self::transaction_materialized`] with
    /// `receipt.first_lsn` after every batch's writes reach the packs, or the
    /// group will continue to hold back WAL reclaim.
    ///
    /// # Errors
    /// Returns an error if the journal is poisoned or the group cannot be
    /// appended.
    pub fn publish_tagged_groups(
        &self,
        batches: &[(ShardType, &[Mutation])],
    ) -> io::Result<CommitReceipt> {
        self.publish_tagged_groups_checked(batches, &[], &[])
    }

    /// Like [`Self::publish_tagged_groups`], validating `expectations` inside
    /// the publication critical section before the group is appended.
    /// The direct applier must call [`Self::transaction_materialized`] with
    /// `receipt.first_lsn` after every batch's writes reach the packs, or the
    /// group will continue to hold back WAL reclaim.
    ///
    /// # Errors
    /// Returns a stale-version error if a collection's logical version differs
    /// from its expectation, or the usual publication errors.
    pub fn publish_tagged_groups_checked(
        &self,
        batches: &[(ShardType, &[Mutation])],
        expectations: &[CollectionExpectation],
        record_expectations: &[RecordExpectation],
    ) -> io::Result<CommitReceipt> {
        let staged = batches
            .iter()
            .map(|(pool, mutations)| (*pool, *mutations))
            .collect::<Vec<_>>();
        self.publish_groups(&staged, expectations, record_expectations)?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cannot publish an empty transaction group",
                )
            })
    }

    /// Validate `expectations` against the live collection versions. The caller
    /// must hold the publication lock, which also serializes the version bumps,
    /// so a passing check cannot race a competing publish.
    fn validate_expectations(
        &self,
        expectations: &[CollectionExpectation],
        record_expectations: &[RecordExpectation],
    ) -> io::Result<()> {
        if expectations.is_empty() && record_expectations.is_empty() {
            return Ok(());
        }
        let versions = self.collection_versions.lock();
        for expectation in expectations {
            let actual = versions
                .get(&(expectation.pool, expectation.collection_id))
                .copied()
                .unwrap_or(0);
            if actual != expectation.expected {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    StaleVersion {
                        pool: expectation.pool,
                        collection_id: expectation.collection_id,
                        expected: expectation.expected,
                        actual,
                    },
                ));
            }
        }
        for expectation in record_expectations {
            // A missing entry is a record whose version predates this process
            // or was never seeded. Use the pool/collection baseline for that
            // case; the read path seeds cold records from frame metadata before
            // staging the expectation.
            let actual = self.absent_record_version(
                expectation.pool,
                &expectation.collection_id,
                &expectation.node_id,
            );
            if actual != expectation.expected {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    StaleVersion {
                        pool: expectation.pool,
                        collection_id: expectation.collection_id,
                        expected: expectation.expected,
                        actual,
                    },
                ));
            }
        }
        Ok(())
    }

    /// Validate `expectations` under the publication lock without publishing.
    /// Used by an expectation-only commit that has no mutations to append.
    ///
    /// # Errors
    /// Returns a stale-version error if any collection's logical version
    /// differs from its expectation.
    pub(crate) fn check_expectations(
        &self,
        expectations: &[CollectionExpectation],
        record_expectations: &[RecordExpectation],
    ) -> io::Result<()> {
        let _publication = self.publication.lock();
        // An expectation-only commit must fail once a published group failed
        // to apply, just like a publishing commit does.
        self.reject_if_apply_failed()?;
        self.validate_expectations(expectations, record_expectations)
    }

    /// Advance the collection and per-record version trackers for a published
    /// group, under the publication lock, so a later conditional commit
    /// validates against the version this group published.
    fn note_published_versions(&self, staged: &[(ShardType, &[Mutation])], last_lsn: u64) {
        let mut versions = self.collection_versions.lock();
        let mut records = self.record_versions.lock();
        for (pool, mutations) in staged {
            for mutation in *mutations {
                match mutation {
                    Mutation::Put {
                        collection_id,
                        node_id,
                        ..
                    }
                    | Mutation::RemoveRecord {
                        collection_id,
                        node_id,
                    } => {
                        let version = versions.entry((*pool, *collection_id)).or_insert(0);
                        *version = (*version).max(last_lsn);
                        let record = records.entry((*pool, *collection_id)).or_default();
                        let seen = record.nodes.entry(*node_id).or_insert(0);
                        *seen = (*seen).max(last_lsn);
                    }
                    Mutation::DeleteCollection { collection_id } => {
                        let version = versions.entry((*pool, *collection_id)).or_insert(0);
                        *version = (*version).max(last_lsn);
                        let record = records.entry((*pool, *collection_id)).or_default();
                        record.nodes.clear();
                        record.deleted_at = Some(
                            record
                                .deleted_at
                                .map_or(last_lsn, |deleted| deleted.max(last_lsn)),
                        );
                    }
                }
            }
        }
    }

    /// Single publication path. Holds `publication` for the whole append so
    /// LSN assignment, the append and the visibility advance are one step.
    ///
    /// Every group is published before its writes are materialized into the
    /// packs. Until [`Self::transaction_materialized`] says otherwise, no pool
    /// may claim coverage at or past its first frame:
    /// the packs do not hold those frames yet, so a claim (and the reclaim it
    /// would license) could lose an acknowledged commit to a crash. It is
    /// registered here, under the journal lock, so neither a read nor a sync
    /// can advance past the group before it is on the floor list.
    fn publish_groups(
        &self,
        staged: &[(ShardType, &[Mutation])],
        expectations: &[CollectionExpectation],
        record_expectations: &[RecordExpectation],
    ) -> io::Result<Option<CommitReceipt>> {
        let _publication = self.publication.lock();
        self.validate_expectations(expectations, record_expectations)?;
        self.reject_if_apply_failed()?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed append",
            ));
        }
        let mut journal = self.journal.lock();
        if self.poisoned.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "journal is poisoned after a failed append",
            ));
        }
        let group_mutations: Vec<(ShardType, Mutation)> = staged
            .iter()
            .flat_map(|(pool, mutations)| mutations.iter().cloned().map(|m| (*pool, m)))
            .collect();
        if group_mutations.is_empty() {
            return Ok(None);
        }
        let expected_first_lsn = journal.next_lsn;
        let expected_count = u64::try_from(group_mutations.len()).unwrap_or(u64::MAX);
        let sequence = self
            .sequence
            .as_ref()
            .map(|counter| counter.fetch_add(1, Ordering::Relaxed));
        let receipt = match journal.append_group_for_current_mode(&group_mutations, sequence) {
            Ok(receipt) => receipt,
            Err(error) => {
                if journal.poisoned {
                    self.poisoned.store(true, Ordering::Release);
                }
                return Err(self.explain_full_segment(error, &journal));
            }
        };
        // Register before advancing the visible version below, so a reader
        // cannot observe this group without also seeing the floor it holds.
        self.unmaterialized.lock().insert(receipt.first_lsn);
        debug_assert_eq!(
            receipt.first_lsn, expected_first_lsn,
            "appended group must start at the journal tail"
        );
        debug_assert_eq!(
            receipt
                .last_lsn
                .saturating_sub(receipt.first_lsn)
                .saturating_add(1),
            expected_count,
            "appended group must cover every mutation"
        );
        // Queue this group's per-pool extents before it becomes visible, still
        // under the journal lock. A sync that captures the new visible LSN
        // then always finds the extents and promotes each pool's watermark;
        // queuing after the visibility advance would let a sync race past them.
        let extents = pool_extents(
            receipt.last_lsn,
            group_mutations.iter().map(|(pool, _)| *pool),
        );
        if !extents.is_empty() {
            self.pending_promotions
                .lock()
                .push((receipt.last_lsn, extents));
        }
        self.published_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        self.visible_lsn
            .fetch_max(receipt.last_lsn, Ordering::Release);
        // Advance every mutated collection's logical version to this group's
        // LSN, still under the publication lock, so a later conditional commit
        // validates against the version this group published.
        self.note_published_versions(staged, receipt.last_lsn);
        // Publish the read-committed change boundary for cross-process workers
        // at the exact point the group becomes visible, before any fsync: a
        // reader may observe a published-but-not-yet-durable group.
        self.bump_publish_signal();
        drop(journal);
        // Wake the committer only when a burst is at or above its early-flush
        // bound; this is the hot path, so ordinary publishes must not pay a
        // futex wake. `committer_wake_threshold` is zero when no committer is
        // running. Only the first publish at or above the bound wakes in a
        // commit epoch; any commit advances the epoch and re-arms the next
        // crossing.
        let threshold = self.committer_wake_threshold.load(Ordering::Relaxed);
        if threshold > 0 && self.pending_count() >= threshold && self.claim_threshold_wake() {
            #[cfg(test)]
            self.threshold_wakes.fetch_add(1, Ordering::Relaxed);
            self.wake_durable_waiters();
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

    /// Register `target_lsn` as a durability request without blocking.
    ///
    /// Returns a [`DurabilityToken`] to pass to [`Self::wait_durable`]. The
    /// request is additive: it never weakens [`Self::sync_through`] /
    /// [`Self::sync`], which remain immediate barriers. With a background
    /// committer running, `wait_durable` blocks until the committer's group
    /// covers the token; without one it performs the blocking commit itself.
    #[must_use]
    pub fn request_durable(&self, target_lsn: u64) -> DurabilityToken {
        self.durable_requests.fetch_add(1, Ordering::Relaxed);
        self.durable_requested
            .fetch_max(target_lsn, Ordering::AcqRel);
        // Wake the committer (for a threshold flush) and any waiters parked on
        // the old boundary.
        self.wake_durable_waiters();
        DurabilityToken { lsn: target_lsn }
    }

    /// Current length of the journal segment file, including its header.
    #[must_use]
    pub fn segment_len(&self) -> u64 {
        self.journal.lock().file_len
    }

    /// The segment's live LSN and sequence numbering without scanning it.
    ///
    /// A reclaim moves the base forward but never changes `next_lsn` or
    /// `next_sequence`: reclaim drops already-assigned groups, it does not
    /// assign new numbers. Compare two snapshots to see how far reclaim advanced
    /// the retained window.
    #[must_use]
    pub fn segment_position(&self) -> SegmentPosition {
        let journal = self.journal.lock();
        SegmentPosition {
            base_sequence: journal.base_sequence,
            next_sequence: journal.next_sequence,
            base_lsn: journal.base_lsn,
            next_lsn: journal.next_lsn,
            groups: journal.directory_stats().groups,
        }
    }

    /// Largest the segment may grow before commits are refused.
    #[must_use]
    pub fn segment_cap(&self) -> u64 {
        self.journal.lock().segment_cap
    }

    /// The segment length past which a stalled reclaim is treated as an
    /// emergency: three quarters of the cap.
    fn emergency_len(&self) -> u64 {
        emergency_len_for_cap(self.segment_cap())
    }

    /// Whether the segment has reached the emergency zone, where a checkpoint
    /// tail still in flight must be waited for rather than let run on.
    #[must_use]
    pub fn in_emergency_zone(&self) -> bool {
        self.segment_len() >= self.emergency_len()
    }

    /// How much a stalled segment must grow before the next forced checkpoint
    /// is worth trying: an eighth of the trigger.
    fn reclaim_retry_growth(&self) -> u64 {
        self.reclaim_trigger_len() / 8
    }

    /// Whether a sync of `pool` should force a reclaiming checkpoint now.
    ///
    /// True once the segment is over the trigger, except while a reclaim is
    /// known to be stalled. Then a pool that is not itself holding the reclaim
    /// back would only repeat a checkpoint that cannot help, so it waits until
    /// the segment has grown by an eighth of the trigger or some pool's
    /// coverage advanced. A pool that is holding it back always checkpoints:
    /// that is what un-stalls it. The emergency zone does not lift the wait:
    /// measured, it only repeated checkpoints that reclaimed nothing.
    #[must_use]
    pub fn should_force_reclaim_checkpoint(&self, pool: Option<ShardType>) -> bool {
        let len = self.segment_len();
        if len <= self.reclaim_trigger_len() {
            return false;
        }
        self.stall_permits_forcing(len, pool)
    }

    /// Whether a stalled reclaim (if there is one) still lets `pool`'s sync
    /// force a checkpoint at segment length `len`.
    fn stall_permits_forcing(&self, len: u64, pool: Option<ShardType>) -> bool {
        if let Some(stall) = *self.replay_lease_stall.lock() {
            return self.coverage_epoch.load(Ordering::Acquire) != stall.coverage_epoch
                || len >= stall.at_len.saturating_add(self.reclaim_retry_growth());
        }
        let Some(stall) = *self.reclaim_stall.lock() else {
            return true;
        };
        self.coverage_epoch.load(Ordering::Acquire) != stall.coverage_epoch
            || len >= stall.at_len.saturating_add(self.reclaim_retry_growth())
            || pool.is_some_and(|pool| self.reclaim_blockers().contains(&pool))
    }

    /// What the journal's in-memory group directory holds: its entry count and
    /// allocated bytes.
    #[must_use]
    pub fn group_directory_stats(&self) -> GroupDirectoryStats {
        self.journal.lock().directory_stats()
    }

    /// Segment length above which a sync forces a reclaiming checkpoint.
    #[must_use]
    pub fn reclaim_trigger_len(&self) -> u64 {
        self.reclaim_trigger_len.load(Ordering::Relaxed)
    }

    /// Test-only handle on the rewrite-window recorder; see
    /// `Journal::rewrite_phase`.
    #[cfg(test)]
    pub(crate) fn rewrite_phase(&self) -> Arc<RewritePhase> {
        Arc::clone(&self.journal.lock().rewrite_phase)
    }

    /// Set the segment length above which a sync forces a reclaiming
    /// checkpoint, from runtime configuration. Unlike
    /// [`Self::set_reclaim_trigger_len`] this is always available and rejects
    /// values outside `1..=MAX_RECLAIM_TRIGGER_LEN`, further limited by this
    /// coordinator's live segment cap so the threshold stays below its
    /// emergency-zone boundary.
    ///
    /// # Errors
    /// Returns `InvalidInput` for zero or a threshold at/above the emergency
    /// boundary.
    pub(crate) fn configure_reclaim_trigger_len(&self, len: u64) -> io::Result<()> {
        let max = max_reclaim_trigger_for_cap(self.segment_cap()).min(MAX_RECLAIM_TRIGGER_LEN);
        if len == 0 || len > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "reclaim trigger must be in 1..={max} bytes (below the emergency-zone boundary), got {len}"
                ),
            ));
        }
        self.reclaim_trigger_len.store(len, Ordering::Relaxed);
        Ok(())
    }

    /// Test-only: install a hook that runs inside a reclaim while the journal
    /// locks are held.
    #[cfg(test)]
    pub(crate) fn set_reclaim_hook(&self, hook: Option<std::sync::Arc<dyn Fn() + Send + Sync>>) {
        *self.reclaim_hook.lock() = hook;
    }

    /// Set the segment length above which a sync forces a reclaiming
    /// checkpoint. Tests use a small value to cross it without writing tens of
    /// MiB. This is available only to this crate's tests and downstream tests
    /// that enable the `test-util` feature. A value of zero forces a checkpoint
    /// on every sync; a value above the segment cap switches forced reclaim off.
    #[cfg(any(test, feature = "test-util"))]
    pub fn set_reclaim_trigger_len(&self, len: u64) {
        self.reclaim_trigger_len.store(len, Ordering::Relaxed);
    }

    /// Shrink the segment cap, and the trigger with it (a quarter of the cap),
    /// so a test can reach the hard limit with a few hundred KiB.
    ///
    /// Only built for this crate's tests and the `test-util` feature; a
    /// production caller has no reason to shrink the cap.
    #[cfg(any(test, feature = "test-util"))]
    pub fn set_segment_cap(&self, cap: u64) {
        self.journal.lock().segment_cap = cap;
        self.set_reclaim_trigger_len(cap / 4);
    }

    /// Highest LSN any caller has requested through [`Self::request_durable`].
    #[must_use]
    pub fn durable_requested(&self) -> u64 {
        self.durable_requested.load(Ordering::Acquire)
    }

    /// Lifetime durability accounting: requests versus real fsyncs, records
    /// per fsync, and blocked-wait latency.
    #[must_use]
    pub fn durability_stats(&self) -> DurabilityStats {
        DurabilityStats {
            durable_requests: self.durable_requests.load(Ordering::Relaxed),
            durable_waits_already_durable: self
                .durable_waits_already_durable
                .load(Ordering::Relaxed),
            durable_wait: self.durable_wait.snapshot(),
            sync_requests: self.sync_calls.load(Ordering::Relaxed),
            sync_waiters: self.journal_waiters.load(Ordering::Relaxed),
            sync_coalesced: self.coalesced_syncs.load(Ordering::Relaxed),
            commits: self.commits.load(Ordering::Relaxed),
            commit_records: self.commit_records.load(Ordering::Relaxed),
            max_commit_records: self.max_commit_records.load(Ordering::Relaxed),
        }
    }

    /// Number of published mutations not yet durably committed. Used as the
    /// background committer's early-flush bound.
    #[must_use]
    pub fn pending_count(&self) -> u64 {
        self.published_lsn
            .load(Ordering::Acquire)
            .saturating_sub(self.committed_lsn.load(Ordering::Acquire))
    }

    /// Whether a background committer is currently running.
    #[must_use]
    pub fn has_background_committer(&self) -> bool {
        matches!(*self.background.lock(), BackgroundState::Running(_))
    }

    /// Number of background group commits that appended and fsynced a group.
    #[must_use]
    pub fn background_commits(&self) -> u64 {
        self.background_commits.load(Ordering::Relaxed)
    }

    /// Number of background commit attempts already covered by a concurrent
    /// durable group. Idle timer ticks (nothing pending) are not counted.
    #[must_use]
    pub fn background_coalesced(&self) -> u64 {
        self.background_coalesced.load(Ordering::Relaxed)
    }

    fn background_failure_detail(&self) -> Option<BackgroundFailure> {
        self.background_failure.lock().clone()
    }

    /// Return the terminal background-committer failure, if any, without
    /// consuming it. Callers use this to retain dirty work and retry through
    /// their own scheduling layer; `wait_durable` remains the blocking API.
    #[must_use]
    pub fn background_failure_message(&self) -> Option<String> {
        self.background_failure
            .lock()
            .as_ref()
            .map(|failure| failure.message.clone())
    }

    /// Wake the committer and any `wait_durable` callers.
    ///
    /// Takes [`Self::durable_lock`] so the wake cannot be lost to the race
    /// where a waiter checks its predicate and then parks after the notifier
    /// has already notified: the notifier either holds the lock while the
    /// waiter is parked (waking it) or blocks until the waiter releases it.
    fn wake_durable_waiters(&self) {
        let _guard = self.durable_lock.lock();
        self.durable_cv.notify_all();
    }

    /// Claim the wake for the current commit epoch.
    ///
    /// Returns `true` only for the publish that wakes the committer for this
    /// epoch, so a burst costs one wake. The claim is retried because a commit
    /// can advance the epoch between the load and the CAS; re-reading the epoch
    /// after a successful claim ensures that newer commit also gets a wake
    /// instead of being suppressed by a claim that was valid a moment earlier.
    fn claim_threshold_wake(&self) -> bool {
        self.claim_threshold_wake_with(|| {})
    }

    /// [`Self::claim_threshold_wake`] with a hook run between a successful claim
    /// and the epoch recheck, so tests can force a commit into that window.
    fn claim_threshold_wake_with(&self, mut between_claim_and_recheck: impl FnMut()) -> bool {
        loop {
            let notified = self.threshold_notified_epoch.load(Ordering::Acquire);
            let epoch = self.commit_epoch.load(Ordering::Acquire);
            if notified == epoch {
                return false;
            }
            if self
                .threshold_notified_epoch
                .compare_exchange(notified, epoch, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                between_claim_and_recheck();
                if self.commit_epoch.load(Ordering::Acquire) == epoch {
                    return true;
                }
            }
        }
    }

    /// Wake the committer and durability waiters when a durable commit left the
    /// pending queue at or above the early-flush bound.
    ///
    /// Publishers that arrived during an fsync read the pre-commit epoch and
    /// suppressed their wake, so nothing would re-arm the parked committer
    /// until the next publish or timer tick. This closes that window from the
    /// commit side, after the epoch has advanced. It is a no-op when no
    /// committer is running (`committer_wake_threshold` is zero).
    fn wake_committer_if_backlogged(&self) {
        let threshold = self.committer_wake_threshold.load(Ordering::Relaxed);
        if threshold > 0 && self.pending_count() >= threshold {
            self.wake_durable_waiters();
        }
    }

    /// Block until the group covering `token` is durable.
    ///
    /// With a background committer running this waits for it to flush; the
    /// bounded poll interval lets the waiter notice a committer that stopped or
    /// a poisoned journal. A committer that failed terminally is reported
    /// rather than waited on. Without a committer, this is exactly
    /// [`Self::sync_through`] on the token's LSN — the historical blocking
    /// behavior, so a caller that never opts into background commits is
    /// unaffected.
    ///
    /// # Errors
    /// Returns an error if the target was never published, the journal is
    /// poisoned, the committer failed, or the committing sync fails.
    pub fn wait_durable(&self, token: DurabilityToken) -> io::Result<Option<CommitReceipt>> {
        let target = token.lsn;
        if target == 0 || target <= self.committed_lsn.load(Ordering::Acquire) {
            self.durable_waits_already_durable
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let result = self.wait_durable_blocking(target);
        self.durable_wait.observe(started.elapsed());
        result
    }

    fn wait_durable_blocking(&self, target: u64) -> io::Result<Option<CommitReceipt>> {
        loop {
            if target <= self.committed_lsn.load(Ordering::Acquire) {
                return Ok(None);
            }
            // Recheck every predicate under `durable_lock`, the same lock the
            // notifiers take, so a state change cannot slip between check and
            // park. The poll interval remains a backstop, not a requirement.
            let mut guard = self.durable_lock.lock();
            if target <= self.committed_lsn.load(Ordering::Acquire) {
                return Ok(None);
            }
            if self.poisoned.load(Ordering::Acquire) {
                return Err(io::Error::other(
                    "journal is poisoned after a failed commit",
                ));
            }
            // A stopped committer falls through to a direct commit, preserving
            // the historical no-committer behavior and error text, but a
            // committer that failed must not be silently papered over.
            if let Some(failure) = self.background_failure_detail() {
                return Err(failure.into_io());
            }
            if !self.has_background_committer() {
                drop(guard);
                return self.sync_through(target);
            }
            // Only with a live committer is an unpublished target validated
            // here; the no-committer path above reaches `sync_through` first.
            if target > self.published_lsn.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "sync target has not been published",
                ));
            }
            self.durable_cv.wait_for(&mut guard, DURABILITY_WAIT_POLL);
        }
    }

    /// Perform one bounded group commit over everything published so far.
    ///
    /// Used by the background committer's timer and by
    /// [`Self::stop_background_committer`]. A no-op returning `None` when
    /// nothing is pending (so an idle tick neither fsyncs nor inflates the
    /// coalescing counter). This is the journal-level durability boundary; a
    /// caller that also needs its packfile checkpoint rewritten should use the
    /// storage-level sync path instead.
    ///
    /// # Errors
    /// Propagates a durable-commit failure from [`Self::sync_through`].
    pub fn flush_durable(&self) -> io::Result<Option<CommitReceipt>> {
        let target = self.capture_sync_target();
        if target == 0 || target <= self.committed_lsn.load(Ordering::Acquire) {
            return Ok(None);
        }
        let receipt = self.sync_through(target)?;
        if receipt.is_some() {
            self.background_commits.fetch_add(1, Ordering::Relaxed);
        } else {
            // Another caller's group covered this target first.
            self.background_coalesced.fetch_add(1, Ordering::Relaxed);
        }
        // A group commit may cover waiters parked on a boundary below `target`.
        self.wake_durable_waiters();
        Ok(receipt)
    }

    /// Start a background thread that fsyncs the pending WAL group on a bounded
    /// interval, coalescing every mutation published in the window into one
    /// group. Flushes early once `max_pending` records are outstanding.
    /// Idempotent: a call while one is running (or stopping) is a no-op.
    ///
    /// The thread holds only a [`Weak`] reference between iterations, so
    /// dropping the last `Arc<JournalCoordinator>` lets it exit within one
    /// interval without an `Arc` cycle. That is a bounded delay, not immediate
    /// teardown — call [`Self::stop_background_committer`] for deterministic
    /// cleanup and a final flush.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] if `interval` is zero or
    /// `max_pending` is zero (both would busy-loop or flush continuously), or
    /// another error if the committer thread cannot be spawned.
    pub fn start_background_committer(
        self: &Arc<Self>,
        config: GroupCommitConfig,
    ) -> io::Result<()> {
        if config.interval.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "background committer interval must be greater than zero",
            ));
        }
        if config.max_pending == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "background committer max_pending must be greater than zero",
            ));
        }
        let mut slot = self.background.lock();
        // Running or Stopping: never spawn a second committer.
        if !matches!(*slot, BackgroundState::Stopped) {
            return Ok(());
        }
        *self.background_failure.lock() = None;
        // A failed run can leave the notification epoch equal to the current
        // commit epoch, which would suppress the first threshold wake after a
        // restart. Reset it so the first crossing always wakes.
        self.threshold_notified_epoch
            .store(u64::MAX, Ordering::Release);
        let stop = Arc::new(AtomicBool::new(false));
        let weak = Arc::downgrade(self);
        let stop_clone = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("mtxdb-journal-committer".to_owned())
            .spawn(move || {
                // Catch a panic so the terminal failure is recorded and waiters
                // are released instead of polling forever on a dead worker.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Self::background_committer_loop(&weak, config, &stop_clone)
                }));
                if let Some(coordinator) = weak.upgrade() {
                    let failure = match outcome {
                        Ok(Ok(())) => None,
                        Ok(Err(failure)) => Some(failure),
                        Err(_) => Some(BackgroundFailure {
                            kind: io::ErrorKind::Other,
                            message: "background committer panicked".to_owned(),
                        }),
                    };
                    coordinator.finish_background(failure);
                }
            })
            .map_err(|error| {
                io::Error::other(format!("failed to spawn journal committer: {error}"))
            })?;
        self.committer_wake_threshold
            .store(config.max_pending, Ordering::Release);
        *slot = BackgroundState::Running(BackgroundCommitter {
            stop,
            handle: Some(handle),
        });
        Ok(())
    }

    /// Signal the background committer to stop, join it, then flush any final
    /// pending group so a quiet stream's last write is durable before this
    /// returns. Safe to call when no committer is running (it still flushes).
    ///
    /// The committer slot is held in a `Stopping` state across the join, so a
    /// concurrent [`Self::start_background_committer`] cannot spawn a second
    /// worker mid-teardown.
    ///
    /// # Errors
    /// Propagates a failure from the final [`Self::flush_durable`], and surfaces
    /// a terminal committer failure recorded by the worker.
    pub fn stop_background_committer(&self) -> io::Result<()> {
        let handle = {
            let mut slot = self.background.lock();
            match std::mem::replace(&mut *slot, BackgroundState::Stopping) {
                BackgroundState::Running(mut committer) => {
                    committer.stop.store(true, Ordering::Release);
                    committer.handle.take()
                }
                other => {
                    *slot = other;
                    None
                }
            }
        };
        self.committer_wake_threshold.store(0, Ordering::Release);
        self.wake_durable_waiters();
        if let Some(handle) = handle {
            // A panic in the committer is caught and recorded by the worker's
            // wrapper; joining still succeeds and the failure is surfaced below.
            let _ = handle.join();
        }
        // Transition `Stopping -> Stopped` and take the recorded failure under
        // one `background` lock. Otherwise a `start_background_committer` that
        // sees `Stopped` can clear `background_failure` in the window between
        // the transition and the take, silently losing the worker's failure.
        let failure = {
            let mut slot = self.background.lock();
            if matches!(*slot, BackgroundState::Stopping) {
                *slot = BackgroundState::Stopped;
            }
            self.background_failure.lock().take()
        };
        self.flush_durable()?;
        if let Some(failure) = failure {
            return Err(failure.into_io());
        }
        Ok(())
    }

    /// Mark the committer slot stopped, publish any terminal failure, and wake
    /// waiters. Called by the worker itself as it exits.
    fn finish_background(&self, failure: Option<BackgroundFailure>) {
        // Transition, publish the failure, and clear the wake threshold under a
        // single `background` lock. A concurrent `start_background_committer`
        // (which holds that lock to clear `background_failure` and set its own
        // threshold) therefore either sees this worker still `Running` and does
        // nothing, or runs after the failure is already recorded; it cannot
        // erase the failure or have its threshold zeroed by this exiting worker.
        {
            let mut slot = self.background.lock();
            // Leave `Stopping` alone: `stop_background_committer` owns that
            // transition and takes the failure after joining this thread.
            if matches!(*slot, BackgroundState::Running(_)) {
                *slot = BackgroundState::Stopped;
                self.committer_wake_threshold.store(0, Ordering::Release);
            }
            if let Some(failure) = failure {
                *self.background_failure.lock() = Some(failure);
            }
        }
        self.wake_durable_waiters();
    }

    fn background_committer_loop(
        weak: &Weak<Self>,
        config: GroupCommitConfig,
        stop: &Arc<AtomicBool>,
    ) -> Result<(), BackgroundFailure> {
        let interval = config.interval;
        loop {
            let Some(coordinator) = weak.upgrade() else {
                return Ok(());
            };
            let now = std::time::Instant::now();
            let deadline = now.checked_add(interval).unwrap_or(now);
            {
                let mut guard = coordinator.durable_lock.lock();
                loop {
                    if stop.load(Ordering::Acquire) || coordinator.poisoned.load(Ordering::Acquire)
                    {
                        break;
                    }
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    if coordinator.pending_count() >= config.max_pending {
                        break;
                    }
                    let timeout = deadline.saturating_duration_since(now);
                    #[cfg(test)]
                    coordinator.committer_parks.fetch_add(1, Ordering::Release);
                    coordinator.durable_cv.wait_for(&mut guard, timeout);
                }
            }
            if stop.load(Ordering::Acquire) || coordinator.poisoned.load(Ordering::Acquire) {
                return Ok(());
            }
            // The final flush on stop is done by `stop_background_committer`
            // after join, so a stop that lands here does not double-fsync.
            if let Err(error) = coordinator.flush_durable() {
                if coordinator.poisoned.load(Ordering::Acquire) {
                    // Poison is already reported to waiters via the poison bit.
                    return Ok(());
                }
                return Err(BackgroundFailure::from_io(&error));
            }
        }
    }

    /// Compact the segment, dropping committed groups at or below
    /// `covered_lsn` while preserving any newer groups.
    ///
    /// Intended to run after a checkpoint durably records that LSN as applied,
    /// so it reclaims only data the packfiles already represent. Published
    /// groups are already appended; [`Self::sync_through`] makes them durable.
    /// Numbering is preserved across the rewrite. Active replay leases may
    /// reduce the reclaim boundary below
    /// `covered_lsn`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the effective reclaim boundary reaches an LSN
    /// that has not been appended. Also propagates poison, segment-read,
    /// validation, and rewrite errors.
    pub fn reclaim_through(&self, covered_lsn: u64) -> io::Result<Reclaim> {
        // Wait out any in-flight fsync: the rewrite replaces the segment file,
        // and a sync must not flush a handle to the replaced inode.
        let _sync = self.sync_lock.lock();
        // The rebuilt file fsyncs every retained group. Run the derived-store
        // hook first, without the journal lock, and remember the boundary it
        // covered. A group published after that boundary must not become
        // durable through this rewrite.
        let hooked_through = self.published_lsn();
        let had_hook = self.run_pre_sync_hook()?;
        let mut journal = self.journal.lock();
        self.revoke_over_cap_leases(&journal);
        self.run_reclaim_hook();
        if had_hook && journal.next_lsn.saturating_sub(1) > hooked_through {
            return Ok(journal.skipped_reclaim());
        }
        let replay_limit = self.replay_reclaim_limit(covered_lsn);
        let reclaim = journal.reclaim_through(replay_limit)?;
        // The rewrite replaced the segment inode and moved its base LSN without
        // necessarily publishing a group. Advance the signal so a gated reader
        // rescans and detects the base jump instead of serving an index older
        // than the reclaimed prefix.
        self.bump_publish_signal();
        Ok(reclaim)
    }

    /// [`Self::reclaim_through`] recording the pool whose missing coverage
    /// stopped the cut on the returned [`Reclaim`]. Diagnostics only.
    fn reclaim_through_with_blocker(
        &self,
        covered_lsn: u64,
        blocked_by: Option<ShardType>,
    ) -> io::Result<Reclaim> {
        let _sync = self.sync_lock.lock();
        let hooked_through = self.published_lsn();
        let had_hook = self.run_pre_sync_hook()?;
        let mut journal = self.journal.lock();
        self.revoke_over_cap_leases(&journal);
        self.run_reclaim_hook();
        if had_hook && journal.next_lsn.saturating_sub(1) > hooked_through {
            return Ok(journal.skipped_reclaim());
        }
        let replay_limit = self.replay_reclaim_limit(covered_lsn);
        let blocked_by = (replay_limit == covered_lsn)
            .then_some(blocked_by)
            .flatten();
        let reclaim = journal.reclaim_through_with_blocker(replay_limit, blocked_by)?;
        // See [`Self::reclaim_through`]: a reader gated on the signal must be
        // forced to rescan after the segment is rewritten in place.
        self.bump_publish_signal();
        Ok(reclaim)
    }
}

