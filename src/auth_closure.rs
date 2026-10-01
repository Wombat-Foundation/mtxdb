//! Transitive auth closures over a room's immutable `auth` edges.
//!
//! For a Matrix event `E`, the closure of `E` is the set of short ids of `E`'s
//! transitive **ancestors** through `auth_events`: every direct auth target, and
//! everything those target in turn. `E` itself is **not** included; a caller
//! that needs Synapse's `include_given` semantics uses
//! [`AuthGraph::union_of_with_given`] (or adds `E`'s own id). The only input is
//! the immutable `auth` adjacency ([`crate::matrix_adjacency::AUTH`]); `prev` and
//! relations are never read, so a redaction, rejection or relation change can
//! never move a closure.
//!
//! # Layers
//!
//! - [`AuthGraph`] and [`ClosureOutcome`] are **in-memory and pure**: they need
//!   only the `bitmaps` feature and no transaction engine. A caller with its own
//!   adjacency can compute [`AuthGraph::compute`], [`AuthGraph::union_of`] and
//!   [`AuthGraph::union_of_with_given`] in a default build.
//! - The persisted generation layer ([`AuthClosure`]) additionally needs
//!   `multi-reader`: it reads a room through
//!   [`MatrixAdjacency`](crate::matrix_adjacency::MatrixAdjacency), materializes
//!   an [`AuthGraph`], and publishes closure generations through a
//!   [`ClosureStore`] and its [`LogicalHead`](crate::logical_head::LogicalHead).
//!
//! Closures are sets of the room's event short ids, so they use the room's event
//! scope as their [`BitmapSet`] domain ([`DomainTag::for_scope`] over that
//! scope). Bitmaps from different rooms therefore cannot be combined by
//! accident.
//!
//! A walk is **complete** only when every reachable event has a recorded `auth`
//! adjacency (a leaf records a present but empty list, so "no auth events" is
//! complete while "never recorded" is not). A reachable event with no record
//! makes the walk incomplete and is reported by id; an `auth` cycle is
//! corruption and is rejected outright.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::bitmap_set::{BitmapSet, DomainTag};
use crate::storage::StorageError;

#[cfg(feature = "multi-reader")]
use crate::closure_store::{
    ClosureCoverageBuilder, ClosureHead, ClosureSnapshot, ClosureStore, GenerationBuilder,
};
#[cfg(feature = "multi-reader")]
use crate::database::SharedDatabase;
#[cfg(feature = "multi-reader")]
use crate::layout::ShardType;
#[cfg(feature = "multi-reader")]
use crate::matrix_adjacency::{MatrixAdjacency, AUTH};
#[cfg(feature = "multi-reader")]
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Outcome of an ancestors-only auth-closure walk.
#[derive(Debug, Clone, PartialEq)]
pub enum ClosureOutcome {
    /// Every reachable auth event was recorded; the set is authoritative and
    /// does not include the queried event.
    Complete(BitmapSet),
    /// A reachable event (or one of its ancestors) has no recorded `auth`
    /// adjacency, so no closure can be trusted.
    Incomplete {
        /// Event ids whose `auth` adjacency is absent.
        missing: Vec<String>,
    },
}

/// An in-memory `auth` adjacency for pure closure computation.
///
/// Only recorded events have an entry in the adjacency (an event with no auth
/// events has a present, empty list); an id with no entry is treated as
/// *missing* and makes every walk through it incomplete. Labels are optional
/// and used only to name missing ids in [`ClosureOutcome::Incomplete`].
#[derive(Debug, Clone)]
pub struct AuthGraph {
    domain: DomainTag,
    auth: HashMap<u32, Vec<u32>>,
    labels: HashMap<u32, String>,
    forward: HashMap<String, u32>,
}

impl AuthGraph {
    /// An empty graph in `domain`.
    #[must_use]
    pub fn new(domain: DomainTag) -> Self {
        Self {
            domain,
            auth: HashMap::new(),
            labels: HashMap::new(),
            forward: HashMap::new(),
        }
    }

    /// The bitmap domain closures over this graph are tagged with.
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Register an event's short id and its human-readable event id.
    pub fn insert_event(&mut self, event_id: impl Into<String>, short_id: u32) {
        let event_id = event_id.into();
        self.labels.insert(short_id, event_id.clone());
        self.forward.insert(event_id, short_id);
    }

    /// Record an event's `auth` targets (present, possibly empty).
    pub fn record_auth(&mut self, short_id: u32, targets: impl IntoIterator<Item = u32>) {
        self.auth.insert(short_id, targets.into_iter().collect());
    }

    /// The short id registered for `event_id`, if any.
    #[must_use]
    pub fn short_id(&self, event_id: &str) -> Option<u32> {
        self.forward.get(event_id).copied()
    }

    /// The event id registered for `short_id`, or a placeholder naming the id.
    #[must_use]
    pub fn label(&self, short_id: u32) -> String {
        self.labels
            .get(&short_id)
            .cloned()
            .unwrap_or_else(|| format!("<unresolved short id {short_id}>"))
    }

    /// The ancestors-only closure of `event_id`.
    ///
    /// # Errors
    /// Returns an error on an `auth` cycle; a domain mismatch cannot arise
    /// because every set is built in this graph's own domain.
    pub fn compute(&self, event_id: &str) -> Result<ClosureOutcome, StorageError> {
        match self.short_id(event_id) {
            Some(short_id) => self.compute_at(short_id),
            None => Ok(ClosureOutcome::Incomplete {
                missing: vec![event_id.to_owned()],
            }),
        }
    }

    /// The ancestors-only closure of a known event short id.
    ///
    /// # Errors
    /// As [`Self::compute`].
    pub fn compute_at(&self, short_id: u32) -> Result<ClosureOutcome, StorageError> {
        let mut walk = Walk::new(self);
        match walk.closure(short_id)? {
            WalkOutcome::Complete(set) => Ok(ClosureOutcome::Complete(set)),
            WalkOutcome::Incomplete => Ok(ClosureOutcome::Incomplete {
                missing: walk.missing_ids(),
            }),
        }
    }

    /// The union of several events' ancestors. The queried events themselves
    /// are not included.
    ///
    /// # Errors
    /// As [`Self::compute`].
    pub fn union_of(&self, event_ids: &[&str]) -> Result<ClosureOutcome, StorageError> {
        self.union_inner(event_ids, false)
    }

    /// As [`Self::union_of`], but also includes each queried event's own short
    /// id (Synapse's `include_given` semantics).
    ///
    /// # Errors
    /// As [`Self::compute`].
    pub fn union_of_with_given(&self, event_ids: &[&str]) -> Result<ClosureOutcome, StorageError> {
        self.union_inner(event_ids, true)
    }

    fn union_inner(
        &self,
        event_ids: &[&str],
        include_given: bool,
    ) -> Result<ClosureOutcome, StorageError> {
        let mut walk = Walk::new(self);
        let mut union = BitmapSet::new(self.domain);
        let mut missing: BTreeSet<String> = BTreeSet::new();
        let mut complete = true;
        for event_id in event_ids {
            if let Some(short_id) = self.short_id(event_id) {
                match walk.closure(short_id)? {
                    WalkOutcome::Complete(set) => union = union.union(&set)?,
                    WalkOutcome::Incomplete => complete = false,
                }
                if include_given {
                    union.insert(short_id);
                }
            } else {
                complete = false;
                missing.insert((*event_id).to_owned());
            }
        }
        if complete {
            Ok(ClosureOutcome::Complete(union))
        } else {
            missing.extend(walk.missing_ids());
            Ok(ClosureOutcome::Incomplete {
                missing: missing.into_iter().collect(),
            })
        }
    }
}

/// Private walk result; strings are resolved once, on demand.
enum WalkOutcome {
    Complete(BitmapSet),
    Incomplete,
}

enum Step {
    Enter(u32),
    Finish(u32),
}

/// A memoizing, iterative post-order walk over `auth` short-id edges.
struct Walk<'a> {
    graph: &'a AuthGraph,
    memo: HashMap<u32, BitmapSet>,
    incomplete: HashSet<u32>,
    missing: BTreeSet<u32>,
}

impl<'a> Walk<'a> {
    fn new(graph: &'a AuthGraph) -> Self {
        Self {
            graph,
            memo: HashMap::new(),
            incomplete: HashSet::new(),
            missing: BTreeSet::new(),
        }
    }

    fn closure(&mut self, root: u32) -> Result<WalkOutcome, StorageError> {
        if let Some(set) = self.memo.get(&root) {
            return Ok(WalkOutcome::Complete(set.clone()));
        }
        if self.incomplete.contains(&root) {
            return Ok(WalkOutcome::Incomplete);
        }
        // Ids on the current root-to-node path, for cycle detection.
        let mut on_path: HashSet<u32> = HashSet::new();
        let mut stack = vec![Step::Enter(root)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(short_id) => {
                    if self.memo.contains_key(&short_id) || self.incomplete.contains(&short_id) {
                        continue;
                    }
                    match self.graph.auth.get(&short_id) {
                        None => {
                            self.incomplete.insert(short_id);
                            self.missing.insert(short_id);
                        }
                        Some(targets) => {
                            let targets = targets.clone();
                            if !on_path.insert(short_id) {
                                return Err(StorageError::Corrupt("auth closure cycle".to_owned()));
                            }
                            stack.push(Step::Finish(short_id));
                            for target in targets {
                                stack.push(Step::Enter(target));
                            }
                        }
                    }
                }
                Step::Finish(short_id) => {
                    on_path.remove(&short_id);
                    // Present at Enter and immutable, so still present.
                    let Some(targets) = self.graph.auth.get(&short_id).cloned() else {
                        return Err(StorageError::Corrupt(
                            "auth edges vanished mid-walk".to_owned(),
                        ));
                    };
                    let mut set = BitmapSet::new(self.graph.domain);
                    let mut complete = true;
                    for target in targets {
                        set.insert(target);
                        match self.memo.get(&target) {
                            Some(child) => set = set.union(child)?,
                            None => complete = false,
                        }
                    }
                    if complete {
                        self.memo.insert(short_id, set);
                    } else {
                        self.incomplete.insert(short_id);
                    }
                }
            }
        }
        match self.memo.get(&root) {
            Some(set) => Ok(WalkOutcome::Complete(set.clone())),
            None => Ok(WalkOutcome::Incomplete),
        }
    }

    fn missing_ids(&self) -> Vec<String> {
        self.missing
            .iter()
            .map(|short_id| {
                self.graph
                    .labels
                    .get(short_id)
                    .cloned()
                    .unwrap_or_else(|| format!("<unresolved short id {short_id}>"))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Persisted generation layer (`multi-reader` + `bitmaps`)
// ---------------------------------------------------------------------------

/// Closure records written per committed batch while rebuilding.
#[cfg(feature = "multi-reader")]
const BATCH_SIZE: usize = 256;

/// Whether an error is a finding about the stored data rather than a failure to
/// read it.
///
/// Corruption is the answer, not the fault: a hand-truncated coverage record or a
/// generation whose head and coverage record disagree are both things verify
/// exists to notice. Everything else is a read that did not happen.
#[cfg(feature = "multi-reader")]
fn is_malformed(error: &StorageError) -> bool {
    matches!(error, StorageError::Corrupt(_))
}

/// A published closure generation.
#[cfg(feature = "multi-reader")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildReport {
    /// Generation number.
    pub generation: u64,
    /// Exclusive upper bound of the covered range: the generation covers short
    /// ids `1..source_next`. Named for the event counter this generation was
    /// built against, not for a count of events, so it reads the same as the
    /// head field it mirrors.
    pub source_next: u32,
    /// Number of closure records written.
    pub count: u32,
    /// Every covered event left without a closure record because a reachable
    /// `auth` parent was absent, in short-id order. One entry per event, not one
    /// per coverage run, so a skipped run of three events lists all three.
    /// Empty means the generation is complete.
    pub skipped: Vec<String>,
    /// Number of events in `skipped`, mirroring the head field it reports.
    pub skipped_count: u32,
    /// The root causes: events that were never recorded, so they have no `auth`
    /// of their own and cannot be walked. Every event in `skipped` reaches one of
    /// these, directly or through a dependent.
    pub missing: Vec<String>,
    /// The skipped ids as canonical inclusive `(start, end)` short-id runs, which
    /// is how the generation's coverage record encodes them. Adjacent skipped ids
    /// collapse into one run.
    pub runs: Vec<(u32, u32)>,
}

#[cfg(feature = "multi-reader")]
impl RebuildReport {
    /// Whether the generation covers every id in `1..source_next` with a real
    /// closure record. A strict caller that cannot tolerate an absent-by-design
    /// answer checks this and fails the rebuild instead of publishing.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.skipped_count == 0
    }
}

/// Result of [`AuthClosure::verify`].
#[cfg(feature = "multi-reader")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthClosureVerifyReport {
    /// Closures recomputed from direct auth edges.
    pub closures_checked: u32,
    /// Covered events confirmed incomplete and absent by design.
    pub skipped_checked: u32,
    /// Violations; empty means consistent.
    pub problems: Vec<String>,
}

#[cfg(feature = "multi-reader")]
impl AuthClosureVerifyReport {
    /// Whether every invariant held.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.problems.is_empty()
    }
}

#[cfg(feature = "multi-reader")]
fn closure_scope(events: [u8; 16]) -> [u8; 16] {
    let mut canonical = b"matrix-auth-closure:".to_vec();
    canonical.extend_from_slice(&events);
    derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
}

/// One room's persisted auth closures.
#[cfg(feature = "multi-reader")]
#[derive(Debug, Clone, Copy)]
pub struct AuthClosure {
    adjacency: MatrixAdjacency,
    store: ClosureStore,
    domain: DomainTag,
}

#[cfg(feature = "multi-reader")]
impl AuthClosure {
    /// Address `room_id`'s auth closures inside `pool`.
    #[must_use]
    pub fn new(pool: ShardType, room_id: &str) -> Self {
        Self::from_adjacency(MatrixAdjacency::new(pool, room_id))
    }

    /// Use an existing room adjacency, sharing its event scope (and therefore
    /// its short ids).
    #[must_use]
    pub fn from_adjacency(adjacency: MatrixAdjacency) -> Self {
        let (pool, events) = adjacency.events_index().scope();
        Self {
            adjacency,
            store: ClosureStore::new(pool, closure_scope(events)),
            domain: DomainTag::for_scope(pool, &events),
        }
    }

    /// The domain of this room's closure bitmaps (its event short-id scope).
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Materialize the room's current `auth` adjacency as a pure [`AuthGraph`].
    ///
    /// This reads every assigned short id, so it is O(room); hold the returned
    /// graph to compute several closures.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn graph(&self, db: &SharedDatabase) -> Result<AuthGraph, StorageError> {
        let events = self.adjacency.events_index();
        let next = events.counter(db)?;
        let ids: Vec<u32> = (1..next).collect();
        let labels = events.resolve(db, &ids)?;
        let mut graph = AuthGraph::new(self.domain);
        for (short_id, key) in ids.into_iter().zip(labels) {
            if let Some(bytes) = key {
                let event_id = String::from_utf8(bytes).map_err(|_| {
                    StorageError::Corrupt(format!(
                        "auth event id for short id {short_id} is not UTF-8"
                    ))
                })?;
                graph.insert_event(event_id, short_id);
            }
            if let Some(edges) = events.edges(db, short_id, AUTH)? {
                graph.record_auth(short_id, edges.iter().map(|edge| edge.target));
            }
        }
        Ok(graph)
    }

    /// The published generation, if any.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt head.
    pub fn head(&self, db: &SharedDatabase) -> Result<Option<ClosureHead>, StorageError> {
        self.store.head(db)
    }

    /// The published head plus the ids it skipped, as one consistent snapshot.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt head or coverage record.
    pub fn snapshot(&self, db: &SharedDatabase) -> Result<Option<ClosureSnapshot>, StorageError> {
        self.store.snapshot(db)
    }

    /// The ancestors-only closure of `event_id`.
    ///
    /// Materializes the room graph (see [`Self::graph`]).
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt record, or an `auth`
    /// cycle.
    pub fn compute(
        &self,
        db: &SharedDatabase,
        event_id: &str,
    ) -> Result<ClosureOutcome, StorageError> {
        self.graph(db)?.compute(event_id)
    }

    /// The ancestors-only closure of a known event short id.
    ///
    /// # Errors
    /// As [`Self::compute`].
    pub fn compute_at(
        &self,
        db: &SharedDatabase,
        short_id: u32,
    ) -> Result<ClosureOutcome, StorageError> {
        self.graph(db)?.compute_at(short_id)
    }

    /// The union of several events' ancestors, optionally including the queried
    /// events themselves.
    ///
    /// # Errors
    /// As [`Self::compute`].
    pub fn union_of(
        &self,
        db: &SharedDatabase,
        event_ids: &[&str],
        include_given: bool,
    ) -> Result<ClosureOutcome, StorageError> {
        let graph = self.graph(db)?;
        if include_given {
            graph.union_of_with_given(event_ids)
        } else {
            graph.union_of(event_ids)
        }
    }

    /// The persisted closure of `event_id` from the published generation.
    ///
    /// `None` means the event is unknown or not covered; a covered leaf is
    /// `Some` and empty.
    ///
    /// # Errors
    /// Returns an error on a read failure or a record stored for another domain.
    pub fn get(
        &self,
        db: &SharedDatabase,
        event_id: &str,
    ) -> Result<Option<BitmapSet>, StorageError> {
        let Some(short_id) = self.adjacency.short_id(db, event_id)? else {
            return Ok(None);
        };
        self.get_at(db, short_id)
    }

    /// The persisted closure of a known event short id.
    ///
    /// # Errors
    /// As [`Self::get`].
    pub fn get_at(
        &self,
        db: &SharedDatabase,
        short_id: u32,
    ) -> Result<Option<BitmapSet>, StorageError> {
        match self.store.get(db, short_id)? {
            Some(blob) => Ok(Some(BitmapSet::decode_in_domain(&blob, self.domain)?)),
            None => Ok(None),
        }
    }

    /// Recompute and publish a fresh generation covering every assigned event
    /// id.
    ///
    /// If the walk for a covered event is incomplete because some reachable `auth`
    /// parent is unknown, the event is recorded as `skipped` in the published
    /// generation and no closure record is written for it; no error is raised and
    /// the generation is still published. This makes "absent by design" explicit
    /// for readers (via [`crate::closure_store::ClosureSnapshot::coverage`]).
    ///
    /// An `auth` cycle is not a gap: the reserved generation is abandoned and the
    /// error propagates, leaving the previously published head in place.
    ///
    /// # Errors
    /// Returns an error on a read/stage/commit failure, a corrupt record, an
    /// `auth` cycle, or a lost publish race (`StorageError::StaleRead`).
    pub fn rebuild(&self, db: &SharedDatabase) -> Result<RebuildReport, StorageError> {
        let events = self.adjacency.events_index();
        let next = events.counter(db)?;
        let graph = self.graph(db)?;
        let mut builder = self.store.begin(db)?;
        let mut walk = Walk::new(&graph);
        let mut batch: Vec<(u32, Vec<u8>)> = Vec::new();
        // Skipped ids are accumulated twice, on purpose: the runs, which are what
        // the coverage record stores and stay compact however holey the room is,
        // and the ids themselves, which only the report needs and only one label
        // per event. A room with a single hole near the end costs one run plus one
        // id, not one id per covered event.
        let mut skipped_ids: Vec<u32> = Vec::new();
        let mut coverage = ClosureCoverageBuilder::new();
        // A walk error (a cycle) is corruption, not a publishable gap, so release
        // the reserved generation rather than leaving it orphaned for retire.
        let walked = (|| -> Result<(), StorageError> {
            for short_id in 1..next {
                match walk.closure(short_id)? {
                    WalkOutcome::Complete(set) => {
                        batch.push((short_id, set.encode()?));
                        if batch.len() >= BATCH_SIZE {
                            flush(&mut builder, db, &mut batch)?;
                        }
                    }
                    WalkOutcome::Incomplete => {
                        skipped_ids.push(short_id);
                        coverage.push(short_id);
                    }
                }
            }
            flush(&mut builder, db, &mut batch)
        })();
        if let Err(error) = walked {
            // The walk error is the real failure; a failed cleanup must not
            // replace it, or the caller loses the corrupt cycle that caused it.
            if let Err(cleanup) = builder.abandon(db) {
                eprintln!("closure rebuild: could not abandon the reserved generation: {cleanup}");
            }
            return Err(error);
        }
        // The walk accumulated the root causes across every closure, so a skipped
        // run and the never-recorded parent that caused it are both reportable.
        let coverage = coverage.build();
        let head = builder.publish_with_coverage(db, next, &coverage)?;
        Ok(RebuildReport {
            generation: head.generation,
            source_next: head.source_next,
            count: head.count,
            skipped: skipped_ids.iter().map(|&id| graph.label(id)).collect(),
            skipped_count: head.skipped_count,
            missing: walk.missing_ids(),
            runs: coverage.runs().to_vec(),
        })
    }

    /// Recompute every covered closure from the direct `auth` edges and compare
    /// it with the published generation.
    ///
    /// A walk that comes out incomplete must be one the coverage record lists as
    /// skipped, and a walk that comes out complete must have a record; coverage is
    /// checked against the walk both ways.
    ///
    /// # Errors
    /// Returns an error only when a record cannot be read; disagreements are
    /// reported in the returned report.
    pub fn verify(&self, db: &SharedDatabase) -> Result<AuthClosureVerifyReport, StorageError> {
        let mut report = AuthClosureVerifyReport::default();
        let storage = self.store.verify(db)?;
        report.problems.extend(
            storage
                .problems
                .into_iter()
                .map(|problem| format!("closure store: {problem}")),
        );
        // A malformed head or coverage record is exactly what verify exists to report,
        // so it becomes a problem rather than an error. Only a genuine read
        // failure aborts.
        let snapshot = match self.store.snapshot(db) {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => return Ok(report),
            Err(error) if is_malformed(&error) => {
                report.problems.push(format!("closure store: {error}"));
                return Ok(report);
            }
            Err(error) => return Err(error),
        };
        let graph = self.graph(db)?;
        let mut walk = Walk::new(&graph);
        for short_id in 1..snapshot.head.source_next {
            match walk.closure(short_id)? {
                WalkOutcome::Complete(expected) => {
                    if snapshot.skipped.contains(short_id) {
                        report.problems.push(format!(
                            "short id {short_id}: generation marks it skipped but its walk is \
                             complete"
                        ));
                        continue;
                    }
                    report.closures_checked = report.closures_checked.saturating_add(1);
                    let blob = match self.store.get_many_pinned(db, &snapshot, &[short_id]) {
                        Ok(blobs) => blobs.into_iter().next().flatten(),
                        // A lost generation or an undecodable record is a finding
                        // about the data, not a failure to check it.
                        Err(error) if is_malformed(&error) => {
                            report
                                .problems
                                .push(format!("short id {short_id}: {error}"));
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                    match blob {
                        Some(blob) => match BitmapSet::decode_in_domain(&blob, self.domain) {
                            Ok(actual) => {
                                if actual != expected {
                                    report.problems.push(format!(
                                        "short id {short_id}: stored closure disagrees with its \
                                         direct auth edges"
                                    ));
                                }
                            }
                            Err(error) => report
                                .problems
                                .push(format!("short id {short_id}: undecodable closure: {error}")),
                        },
                        None => report.problems.push(format!(
                            "short id {short_id}: covered by the head but has no closure record"
                        )),
                    }
                }
                WalkOutcome::Incomplete => {
                    if snapshot.skipped.contains(short_id) {
                        report.skipped_checked = report.skipped_checked.saturating_add(1);
                    } else {
                        report.problems.push(format!(
                            "short id {short_id}: closure walk is incomplete but the generation \
                             does not mark it skipped"
                        ));
                    }
                }
            }
        }
        Ok(report)
    }

    #[cfg(test)]
    #[allow(
        dead_code,
        reason = "test-only accessor kept for closure tests that are still landing"
    )]
    pub(crate) fn store_for_test(&self) -> ClosureStore {
        self.store
    }
}

#[cfg(feature = "multi-reader")]
fn flush(
    builder: &mut GenerationBuilder,
    db: &SharedDatabase,
    batch: &mut Vec<(u32, Vec<u8>)>,
) -> Result<(), StorageError> {
    if batch.is_empty() {
        return Ok(());
    }
    let refs: Vec<(u32, &[u8])> = batch
        .iter()
        .map(|(short_id, blob)| (*short_id, blob.as_slice()))
        .collect();
    builder.add(db, &refs)?;
    batch.clear();
    Ok(())
}
