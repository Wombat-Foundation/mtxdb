//! Public façade over a room's auth data: adjacency, closures and their
//! lifecycle behind one handle and one error type.
//!
//! Callers use [`RoomAuth`](crate::room_auth::RoomAuth) and never coordinate the short-id index, the
//! adjacency, the closure generations or the bitmaps by hand:
//!
//! - **Ingest** with [`RoomAuth::record_event`](crate::room_auth::RoomAuth::record_event) / [`RoomAuth::record_events`](crate::room_auth::RoomAuth::record_events).
//!   Ingestion commits the event's `prev`, `auth` and relation data; it does
//!   not touch closures. A query for a recent event is still correct, because
//!   [`RoomAuth::auth_chain`](crate::room_auth::RoomAuth::auth_chain) walks only the events newer than the published
//!   generation and stops at any parent the generation already covers.
//! - **Query** with [`RoomAuth::auth_edges`](crate::room_auth::RoomAuth::auth_edges), [`RoomAuth::auth_chain`](crate::room_auth::RoomAuth::auth_chain) and
//!   [`RoomAuth::is_in_auth_chain`](crate::room_auth::RoomAuth::is_in_auth_chain), or pin a generation with
//!   [`RoomAuth::snapshot`](crate::room_auth::RoomAuth::snapshot) to make several queries against one consistent
//!   generation.
//! - **Maintain** with [`RoomAuth::rebuild`](crate::room_auth::RoomAuth::rebuild), [`RoomAuth::retire_old_generations`](crate::room_auth::RoomAuth::retire_old_generations),
//!   [`RoomAuth::verify`](crate::room_auth::RoomAuth::verify) and [`RoomAuth::purge`](crate::room_auth::RoomAuth::purge). Each is an explicit call; none
//!   happens implicitly.
//!
//! # What the visibility filter does and does not do
//!
//! Redaction, rejection and soft-fail change what a *relationship* query may
//! show, so [`RoomAuth::relation_of`](crate::room_auth::RoomAuth::relation_of) takes an
//! [`EventVisibility`](crate::matrix_adjacency::EventVisibility). They never
//! change the auth chain: a redacted event keeps its `auth_events`, and an
//! auth chain computed against a different disposition would be wrong. So
//! `auth_edges`, `auth_chain` and `is_in_auth_chain` take no filter, by design.
//!
//! # Chains are ancestors-only
//!
//! The chain of an event excludes the event itself, so
//! `is_in_auth_chain(e, e)` is `false`.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::auth_closure::{AuthClosure, RebuildReport};
use crate::bitmap_set::BitmapSet;
use crate::closure_store::{ClosureCoverage, ClosureSnapshot};
use crate::database::Database;
use crate::layout::ShardType;
use crate::matrix_adjacency::{EventRecord, EventVisibility, Relation, AUTH};
use crate::storage::StorageError;

/// A room auth operation failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum RoomAuthError {
    /// A batch had more events than one transaction takes. Nothing was recorded.
    BatchTooLarge {
        /// The most events one batch may hold.
        limit: usize,
    },
    /// The batch needs more staging than one transaction allows (64 MiB), for
    /// example events with extreme fan-out. Nothing was recorded; split it into
    /// smaller batches. Distinct from [`Self::BatchTooLarge`], which is the event
    /// count limit.
    BatchExceedsTransactionLimit,
    /// The room's `u32` short-id space or its `u16` relation-kind dictionary is
    /// full. Ids are never reused, so no further event or relation type can be
    /// assigned one, and the operation published nothing.
    OrdinalExhausted,
    /// An event was never seen by this room: it was neither recorded nor
    /// referenced by a recorded event.
    UnknownEvent {
        /// The event id.
        event_id: String,
    },
    /// An event is known only because another event referenced it; it was never
    /// recorded itself, so it has no `auth` data.
    EventNotRecorded {
        /// The event id.
        event_id: String,
    },
    /// An event's `auth` chain reaches an event that was never recorded, so its
    /// chain cannot be computed. This names the first such event met.
    MissingParent {
        /// The absent event id.
        event_id: String,
    },
    /// A snapshot pinned a generation that a later publish has since retired.
    /// Take a fresh snapshot; retrying the same read cannot succeed.
    StaleGeneration {
        /// Generation the handle was pinned to.
        pinned: u64,
        /// Generation published now, or `None` if nothing is (for example the
        /// room was purged).
        current: Option<u64>,
    },
    /// Another rebuild published first, so this one's generation was discarded.
    /// The room is unchanged; rebuild again if a fresher generation is wanted.
    RebuildConflict,
    /// A closure record belongs to a different room or scope than the one
    /// addressed.
    WrongDomain,
    /// Stored data is malformed or fails an integrity check, including an `auth`
    /// cycle.
    Corruption(String),
    /// The closure walk was incomplete: a reachable event has no recorded
    /// `auth` adjacency.
    IncompleteClosure {
        /// Event ids whose `auth` adjacency is absent.
        missing: Vec<String>,
    },
    /// The underlying storage layer failed.
    Storage(StorageError),
}

impl fmt::Display for RoomAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchTooLarge { limit } => {
                write!(f, "a batch holds at most {limit} events")
            }
            Self::BatchExceedsTransactionLimit => write!(
                f,
                "the batch needs more than one transaction can stage; split it into smaller batches"
            ),
            Self::OrdinalExhausted => write!(f, "room id space is exhausted"),
            Self::UnknownEvent { event_id } => write!(f, "unknown event {event_id}"),
            Self::EventNotRecorded { event_id } => {
                write!(f, "event {event_id} is referenced but was never recorded")
            }
            Self::MissingParent { event_id } => {
                write!(f, "auth chain references unrecorded event {event_id}")
            }
            Self::StaleGeneration { pinned, current } => match current {
                Some(current) => write!(
                    f,
                    "pinned closure generation {pinned} is stale; current generation is {current}"
                ),
                None => write!(
                    f,
                    "pinned closure generation {pinned} is stale; no generation is published"
                ),
            },
            Self::RebuildConflict => write!(f, "another rebuild published first"),
            Self::WrongDomain => write!(f, "closure record belongs to a different domain"),
            Self::Corruption(message) => write!(f, "room auth data is corrupt: {message}"),
            Self::IncompleteClosure { missing } => {
                write!(f, "auth closure is incomplete; missing {missing:?}")
            }
            Self::Storage(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RoomAuthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<StorageError> for RoomAuthError {
    fn from(error: StorageError) -> Self {
        if error.is_stage_too_large() {
            return Self::BatchExceedsTransactionLimit;
        }
        match error {
            StorageError::Corrupt(message) => Self::Corruption(message),
            StorageError::Exhausted(_) => Self::OrdinalExhausted,
            StorageError::StaleGeneration {
                generation,
                current,
            } => Self::StaleGeneration {
                pinned: generation,
                current,
            },
            other => Self::Storage(other),
        }
    }
}

/// An event to record: its `prev` and `auth` events and, optionally, its
/// relation to another event.
pub type NewEvent<'a> = EventRecord<'a>;

/// A recorded event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedEvent {
    /// The event's room-local short id.
    pub short_id: u32,
}

/// What [`RoomAuth::verify`] found.
///
/// Problems are findings about stored data. The other fields describe the
/// room's state and are not problems: events newer than the generation need a
/// rebuild, and events skipped by design are the ones whose history is missing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoomAuthVerifyReport {
    /// The generation that was checked, if any is published.
    pub generation: Option<u64>,
    /// Problems in the stored `prev`, `auth` and relation adjacency.
    pub adjacency_problems: Vec<String>,
    /// Problems in the closure generation, including closures that disagree
    /// with the direct `auth` edges and malformed head or coverage records.
    pub closure_problems: Vec<String>,
    /// Closures recomputed from the direct `auth` edges and compared.
    pub closures_checked: u32,
    /// Covered events left without a closure because a parent was never
    /// recorded. Absent by design, not a problem.
    pub incomplete_events: u32,
    /// Events referenced as an `auth` parent but never recorded.
    pub missing_parents: Vec<String>,
    /// Events recorded after the published generation. They are queried through
    /// the lazy walk until the next [`RoomAuth::rebuild`].
    pub uncovered_events: u32,
}

impl RoomAuthVerifyReport {
    /// Whether no stored data disagrees with the direct `auth` edges.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.adjacency_problems.is_empty() && self.closure_problems.is_empty()
    }

    /// Whether a rebuild would cover more events than the published generation.
    #[must_use]
    pub const fn needs_rebuild(&self) -> bool {
        self.uncovered_events > 0
    }
}

/// The most events [`RoomAuth::record_events`] takes in one call. One
/// transaction can only stage a bounded amount, so larger imports are split.
pub const MAX_BATCH_EVENTS: usize = 1000;

/// How many times a convenience query retries after its fresh snapshot was
/// retired by a concurrent rebuild and retire.
const QUERY_ATTEMPTS: usize = 3;

/// A room's auth data.
#[derive(Debug, Clone, Copy)]
pub struct RoomAuth {
    closure: AuthClosure,
}

impl RoomAuth {
    /// Address `room_id`'s auth data inside `pool`.
    #[must_use]
    pub fn new(pool: ShardType, room_id: &str) -> Self {
        Self {
            closure: AuthClosure::new(pool, room_id),
        }
    }

    /// Record an event's `prev`, `auth` and relation in one transaction,
    /// allocating short ids for it and every event it references.
    ///
    /// Recording identical data again is a no-op; different `prev`, `auth` or
    /// relation for an already-recorded event is an error, because those never
    /// change. Closures are not touched: queries stay correct, and
    /// [`Self::rebuild`] covers the event.
    ///
    /// # Errors
    /// [`RoomAuthError::OrdinalExhausted`] if an id space is full; storage
    /// errors, including a lost race (`StorageError::StaleRead`), which a caller
    /// may retry.
    pub fn record_event(
        &self,
        db: &Database,
        event: &NewEvent<'_>,
    ) -> Result<RecordedEvent, RoomAuthError> {
        let short_id = self.closure.adjacency().record_event(
            db,
            event.event_id,
            event.prev,
            event.auth,
            event.relation,
        )?;
        Ok(RecordedEvent { short_id })
    }

    /// Record several events in **one** transaction, returning them in order.
    ///
    /// Either every event is recorded or none is, so a failed batch leaves the
    /// room as it was and can simply be run again. Events in the batch may
    /// reference each other. A batch is limited to [`MAX_BATCH_EVENTS`] events,
    /// because one transaction can only stage so much; split a larger import into
    /// batches.
    ///
    /// # Errors
    /// [`RoomAuthError::BatchTooLarge`] over the limit,
    /// [`RoomAuthError::OrdinalExhausted`] if an id space is full, otherwise as
    /// [`Self::record_event`]. In every case nothing from the batch is recorded.
    pub fn record_events(
        &self,
        db: &Database,
        events: &[NewEvent<'_>],
    ) -> Result<Vec<RecordedEvent>, RoomAuthError> {
        if events.len() > MAX_BATCH_EVENTS {
            return Err(RoomAuthError::BatchTooLarge {
                limit: MAX_BATCH_EVENTS,
            });
        }
        Ok(self
            .closure
            .adjacency()
            .record_events(db, events)?
            .into_iter()
            .map(|short_id| RecordedEvent { short_id })
            .collect())
    }

    /// The event's `auth_events`.
    ///
    /// An event recorded with an empty `auth` list returns an empty list. That
    /// differs from an event that was only referenced.
    ///
    /// # Errors
    /// [`RoomAuthError::UnknownEvent`] if the room never saw the event,
    /// [`RoomAuthError::EventNotRecorded`] if it was only referenced.
    pub fn auth_edges(&self, db: &Database, event_id: &str) -> Result<Vec<String>, RoomAuthError> {
        let adjacency = self.closure.adjacency();
        match adjacency.auth_of(db, event_id)? {
            Some(edges) => Ok(edges),
            None => Err(self.absent(db, event_id)?),
        }
    }

    /// The event's relation, if it has one and `visibility` shows it.
    ///
    /// This is the one query the visibility filter applies to; see the module
    /// documentation for why auth queries ignore it.
    ///
    /// # Errors
    /// Storage errors.
    pub fn relation_of(
        &self,
        db: &Database,
        event_id: &str,
        visibility: &dyn EventVisibility,
    ) -> Result<Option<Relation>, RoomAuthError> {
        Ok(self
            .closure
            .adjacency()
            .relation_of(db, event_id, visibility)?)
    }

    /// Pin the published generation (or none) so several queries read one
    /// consistent generation.
    ///
    /// # Errors
    /// Storage errors, or [`RoomAuthError::Corruption`] for a malformed head.
    pub fn snapshot(&self, db: &Database) -> Result<AuthSnapshot<'_>, RoomAuthError> {
        Ok(AuthSnapshot {
            room: self,
            snapshot: self.closure.snapshot(db)?,
        })
    }

    /// The event's auth chain: every ancestor reachable through `auth_events`,
    /// excluding the event itself, as event ids in short-id order.
    ///
    /// # Errors
    /// See [`AuthSnapshot::auth_chain`].
    pub fn auth_chain(&self, db: &Database, event_id: &str) -> Result<Vec<String>, RoomAuthError> {
        self.with_snapshot(db, |snapshot| snapshot.auth_chain(db, event_id))
    }

    /// The event's auth chain as a set of short ids, for set algebra.
    ///
    /// # Errors
    /// See [`AuthSnapshot::auth_chain`].
    pub fn auth_chain_ids(
        &self,
        db: &Database,
        event_id: &str,
    ) -> Result<BitmapSet, RoomAuthError> {
        self.with_snapshot(db, |snapshot| snapshot.auth_chain_ids(db, event_id))
    }

    /// Whether `ancestor` is in `event`'s auth chain. An event is not in its own
    /// chain, and an `ancestor` the room never saw is not in any chain.
    ///
    /// # Errors
    /// See [`AuthSnapshot::auth_chain`].
    pub fn is_in_auth_chain(
        &self,
        db: &Database,
        ancestor: &str,
        event: &str,
    ) -> Result<bool, RoomAuthError> {
        self.with_snapshot(db, |snapshot| {
            snapshot.is_in_auth_chain(db, ancestor, event)
        })
    }

    /// Recompute and publish a fresh closure generation covering every event.
    ///
    /// Events whose chain reaches an unrecorded parent are left without a
    /// closure and listed in the report; the generation is still published.
    /// Check [`RebuildReport::is_complete`] if a gap should be a failure. An
    /// `auth` cycle is corruption: the attempt is abandoned and the previous
    /// generation stays published.
    ///
    /// # Errors
    /// [`RoomAuthError::RebuildConflict`] if another rebuild published first,
    /// [`RoomAuthError::Corruption`] for a cycle, otherwise storage errors.
    pub fn rebuild(&self, db: &Database) -> Result<RebuildReport, RoomAuthError> {
        self.closure.rebuild(db).map_err(rebuild_error)
    }

    /// Delete generations older than the published one and its predecessor.
    /// Returns how many generations were removed. A snapshot pinned to a removed
    /// generation fails with [`RoomAuthError::StaleGeneration`].
    ///
    /// # Errors
    /// Storage errors.
    pub fn retire_old_generations(&self, db: &Database) -> Result<u64, RoomAuthError> {
        Ok(self.closure.store().retire_superseded(db)?)
    }

    /// Check the room's stored data against its direct `auth` edges.
    ///
    /// Malformed data is reported as findings; only a failure to read aborts.
    ///
    /// # Errors
    /// Storage errors that are not findings about the data.
    pub fn verify(&self, db: &Database) -> Result<RoomAuthVerifyReport, RoomAuthError> {
        let adjacency = self.closure.adjacency();
        let events = adjacency.events_index();
        let mut report = RoomAuthVerifyReport {
            adjacency_problems: adjacency.verify(db)?.problems,
            ..RoomAuthVerifyReport::default()
        };
        let closure = self.closure.verify(db)?;
        report.closures_checked = closure.closures_checked;
        report.closure_problems = closure.problems;

        let next = events.counter(db)?;
        for short_id in 1..next {
            if events.edges(db, short_id, AUTH)?.is_none() {
                report.missing_parents.push(self.label(db, short_id)?);
            }
        }
        match self.closure.snapshot(db) {
            Ok(Some(snapshot)) => {
                report.generation = Some(snapshot.head.generation);
                report.incomplete_events = snapshot.head.skipped_count;
                report.uncovered_events = next.saturating_sub(snapshot.head.source_next);
            }
            Ok(None) => report.uncovered_events = next.saturating_sub(1),
            // The closure verify above already reported a malformed snapshot.
            Err(StorageError::Corrupt(_)) => {}
            Err(error) => return Err(error.into()),
        }
        Ok(report)
    }

    /// Remove everything this room stored — adjacency, relation kinds and
    /// closure generations — in one transaction. A snapshot held across the
    /// purge fails with [`RoomAuthError::StaleGeneration`] and no current
    /// generation.
    ///
    /// # Errors
    /// Storage errors.
    pub fn purge(&self, db: &Database) -> Result<(), RoomAuthError> {
        let txn = db.begin_transaction();
        self.closure.adjacency().stage_purge(&txn)?;
        self.closure.stage_purge(&txn)?;
        Ok(txn.commit()?)
    }

    /// The closure layer, for tests that reach below the façade.
    #[cfg(test)]
    pub(crate) const fn closure_for_test(&self) -> &AuthClosure {
        &self.closure
    }

    /// Run `query` against a fresh snapshot, retrying with a new one if a
    /// concurrent rebuild and retire expires it mid-query.
    fn with_snapshot<T>(
        &self,
        db: &Database,
        query: impl Fn(&AuthSnapshot<'_>) -> Result<T, RoomAuthError>,
    ) -> Result<T, RoomAuthError> {
        let mut last = None;
        for _ in 0..QUERY_ATTEMPTS {
            let snapshot = self.snapshot(db)?;
            match query(&snapshot) {
                Err(error @ RoomAuthError::StaleGeneration { .. }) => last = Some(error),
                other => return other,
            }
        }
        Err(last.unwrap_or(RoomAuthError::RebuildConflict))
    }

    /// The error for an event with no `auth` data: unknown, or only referenced.
    fn absent(&self, db: &Database, event_id: &str) -> Result<RoomAuthError, RoomAuthError> {
        Ok(match self.closure.adjacency().short_id(db, event_id)? {
            Some(_) => RoomAuthError::EventNotRecorded {
                event_id: event_id.to_owned(),
            },
            None => RoomAuthError::UnknownEvent {
                event_id: event_id.to_owned(),
            },
        })
    }

    /// The event id for a short id.
    fn label(&self, db: &Database, short_id: u32) -> Result<String, RoomAuthError> {
        self.labels(db, &[short_id])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                RoomAuthError::Corruption(format!("short id {short_id} has no event id"))
            })
    }

    /// The event ids for several short ids, resolved in one batched read.
    fn labels(&self, db: &Database, short_ids: &[u32]) -> Result<Vec<String>, RoomAuthError> {
        let events = self.closure.adjacency().events_index();
        events
            .resolve(db, short_ids)?
            .into_iter()
            .zip(short_ids)
            .map(|(key, short_id)| {
                let key = key.ok_or_else(|| {
                    RoomAuthError::Corruption(format!("short id {short_id} has no event id"))
                })?;
                String::from_utf8(key).map_err(|_| {
                    RoomAuthError::Corruption(format!(
                        "event id for short id {short_id} is not UTF-8"
                    ))
                })
            })
            .collect()
    }
}

/// A rebuild's own errors: a lost publish race is its own case.
#[cfg(test)]
pub(crate) fn rebuild_error_for_test(error: StorageError) -> RoomAuthError {
    rebuild_error(error)
}

/// A rebuild's own errors: a lost publish race is its own case.
fn rebuild_error(error: StorageError) -> RoomAuthError {
    if error.is_stale_read() {
        RoomAuthError::RebuildConflict
    } else {
        error.into()
    }
}

/// A pinned view of a room's closure generation.
///
/// Every query through it reads the same generation, so several queries cannot
/// observe different generations while a rebuild runs. If the generation is
/// retired while the snapshot is held, a read that needs it fails with
/// [`RoomAuthError::StaleGeneration`]. A query answered only from the snapshot's
/// own coverage, or from adjacency, never touches the generation and does not
/// report staleness.
#[derive(Debug)]
pub struct AuthSnapshot<'a> {
    room: &'a RoomAuth,
    snapshot: Option<ClosureSnapshot>,
}

impl AuthSnapshot<'_> {
    /// The pinned generation, or `None` if no closures were published.
    #[must_use]
    pub fn generation(&self) -> Option<u64> {
        self.snapshot
            .as_ref()
            .map(|snapshot| snapshot.head.generation)
    }

    /// The event's auth chain as event ids in short-id order, excluding the
    /// event itself.
    ///
    /// # Errors
    /// - [`RoomAuthError::UnknownEvent`] / [`RoomAuthError::EventNotRecorded`]
    ///   for an event that was never seen / never recorded;
    /// - [`RoomAuthError::MissingParent`] naming the first ancestor that was
    ///   never recorded;
    /// - [`RoomAuthError::Corruption`] for an `auth` cycle;
    /// - [`RoomAuthError::StaleGeneration`] if the pinned generation was retired;
    /// - [`RoomAuthError::WrongDomain`] for a closure stored for another room.
    pub fn auth_chain(&self, db: &Database, event_id: &str) -> Result<Vec<String>, RoomAuthError> {
        let ids = self.auth_chain_ids(db, event_id)?;
        self.room.labels(db, &ids.iter().collect::<Vec<u32>>())
    }

    /// The event's auth chain as a set of short ids. See [`Self::auth_chain`].
    ///
    /// # Errors
    /// As [`Self::auth_chain`].
    pub fn auth_chain_ids(
        &self,
        db: &Database,
        event_id: &str,
    ) -> Result<BitmapSet, RoomAuthError> {
        let adjacency = self.room.closure.adjacency();
        let Some(root) = adjacency.short_id(db, event_id)? else {
            return Err(RoomAuthError::UnknownEvent {
                event_id: event_id.to_owned(),
            });
        };
        if adjacency.events_index().edges(db, root, AUTH)?.is_none() {
            return Err(RoomAuthError::EventNotRecorded {
                event_id: event_id.to_owned(),
            });
        }
        self.chain(db, root)
    }

    /// Whether `ancestor` is in `event`'s auth chain. An event is not in its own
    /// chain, and an `ancestor` the room never saw is not in any chain.
    ///
    /// # Errors
    /// As [`Self::auth_chain`], for `event`.
    pub fn is_in_auth_chain(
        &self,
        db: &Database,
        ancestor: &str,
        event: &str,
    ) -> Result<bool, RoomAuthError> {
        let chain = self.auth_chain_ids(db, event)?;
        Ok(self
            .room
            .closure
            .adjacency()
            .short_id(db, ancestor)?
            .is_some_and(|id| chain.contains(id)))
    }

    /// Whether the pinned generation holds a closure for the event. `false` for
    /// an event recorded after it, or one whose history is incomplete; queries
    /// for such events still work through the lazy walk, or fail with
    /// [`RoomAuthError::MissingParent`] when history is missing.
    ///
    /// # Errors
    /// Storage errors.
    pub fn is_covered(&self, db: &Database, event_id: &str) -> Result<bool, RoomAuthError> {
        let Some(short_id) = self.room.closure.adjacency().short_id(db, event_id)? else {
            return Ok(false);
        };
        Ok(self.coverage(short_id) == ClosureCoverage::Complete)
    }

    fn coverage(&self, short_id: u32) -> ClosureCoverage {
        self.snapshot
            .as_ref()
            .map_or(ClosureCoverage::Absent, |snapshot| {
                snapshot.coverage(short_id)
            })
    }

    /// The ancestors of `root`, reading a persisted closure wherever the
    /// snapshot has one and walking `auth` edges only through the rest.
    ///
    /// The walk is iterative, so a long uncovered chain cannot overflow the
    /// stack, and its cost is bounded by the events the generation does not
    /// cover, not by the size of the room.
    fn chain(&self, db: &Database, root: u32) -> Result<BitmapSet, RoomAuthError> {
        struct Frame {
            id: u32,
            targets: Vec<u32>,
            next: usize,
            acc: BitmapSet,
        }
        enum Opened {
            Done(BitmapSet),
            Frame(Frame),
        }

        let domain = self.room.closure.domain();
        let events = self.room.closure.adjacency().events_index();
        let mut memo: HashMap<u32, BitmapSet> = HashMap::new();
        let mut on_path: HashSet<u32> = HashSet::new();
        let mut stack: Vec<Frame> = Vec::new();

        let open = |id: u32, memo: &mut HashMap<u32, BitmapSet>| -> Result<Opened, RoomAuthError> {
            if let Some(set) = memo.get(&id) {
                return Ok(Opened::Done(set.clone()));
            }
            if self.coverage(id) == ClosureCoverage::Complete {
                let set = self.persisted(db, id)?;
                memo.insert(id, set.clone());
                return Ok(Opened::Done(set));
            }
            #[cfg(test)]
            tests::count_edge_read();
            match events.edges(db, id, AUTH)? {
                Some(edges) => Ok(Opened::Frame(Frame {
                    id,
                    targets: edges.iter().map(|edge| edge.target).collect(),
                    next: 0,
                    acc: BitmapSet::new(domain),
                })),
                None => Err(RoomAuthError::MissingParent {
                    event_id: self.room.label(db, id)?,
                }),
            }
        };

        match open(root, &mut memo)? {
            Opened::Done(set) => return Ok(set),
            Opened::Frame(frame) => {
                on_path.insert(frame.id);
                stack.push(frame);
            }
        }
        loop {
            let Some(top) = stack.last_mut() else {
                return Err(RoomAuthError::Corruption(
                    "auth walk finished without a result".to_owned(),
                ));
            };
            if let Some(&parent) = top.targets.get(top.next) {
                top.next = top.next.saturating_add(1);
                top.acc.insert(parent);
                if on_path.contains(&parent) {
                    return Err(RoomAuthError::Corruption(format!(
                        "auth cycle through {}",
                        self.room.label(db, parent)?
                    )));
                }
                match open(parent, &mut memo)? {
                    Opened::Done(set) => top.acc = top.acc.union(&set).map_err(domain_error)?,
                    Opened::Frame(frame) => {
                        on_path.insert(frame.id);
                        stack.push(frame);
                    }
                }
                continue;
            }
            let Some(done) = stack.pop() else {
                return Err(RoomAuthError::Corruption(
                    "auth walk lost its frame".to_owned(),
                ));
            };
            on_path.remove(&done.id);
            memo.insert(done.id, done.acc.clone());
            match stack.last_mut() {
                None => return Ok(done.acc),
                Some(parent) => {
                    parent.acc = parent.acc.union(&done.acc).map_err(domain_error)?;
                }
            }
        }
    }

    /// The persisted closure of a `Complete` id, from the pinned generation.
    fn persisted(&self, db: &Database, short_id: u32) -> Result<BitmapSet, RoomAuthError> {
        let Some(snapshot) = self.snapshot.as_ref() else {
            return Err(RoomAuthError::Corruption(
                "a closure was expected but no generation is pinned".to_owned(),
            ));
        };
        let blob = self
            .room
            .closure
            .store()
            .get_many_pinned(db, snapshot, &[short_id])?
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| {
                RoomAuthError::Corruption(format!(
                    "generation {} has no closure for complete id {short_id}",
                    snapshot.head.generation
                ))
            })?;
        BitmapSet::decode_in_domain(&blob, self.room.closure.domain()).map_err(domain_error)
    }
}

/// A bitmap whose domain is not this room's means a closure belongs elsewhere;
/// anything else about it is corruption.
fn domain_error(error: StorageError) -> RoomAuthError {
    match error {
        StorageError::Collision(_) => RoomAuthError::WrongDomain,
        other => other.into(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    thread_local! {
        static EDGE_READS: Cell<u64> = const { Cell::new(0) };
    }

    /// Count one `auth` edge read made by the chain walker.
    pub(crate) fn count_edge_read() {
        EDGE_READS.with(|reads| reads.set(reads.get().saturating_add(1)));
    }

    /// Reset and return this thread's count of `auth` edge reads.
    pub(crate) fn take_edge_reads() -> u64 {
        EDGE_READS.with(Cell::take)
    }
}
