//! Matrix event adjacency stored as per-family CSR segments.
//!
//! One room maps to two [`ShortIdIndex`](crate::short_id::ShortIdIndex) scopes
//! and one CSR snapshot:
//!
//! - **events**: event id -> room-local `u32` short id, with the event-id
//!   reverse records. This scope no longer stores per-event edge records;
//!   adjacency lives in the CSR snapshot.
//! - **relation kinds**: relation type string (`m.annotation`, ...) -> `u16`
//!   kind id.
//! - **CSR**: one versioned snapshot holding the `prev`, `auth` and typed
//!   `relations` families as compressed sparse rows keyed by short id, plus
//!   the maintained [`PREV_CHILDREN`](crate::csr_adjacency::PREV_CHILDREN)
//!   family, which stores each parent's child list so
//!   [`CsrAdjacency::children_of`](crate::csr_adjacency::CsrAdjacency::children_of)
//!   is a point read. Transposing a forward family on demand
//!   ([`CsrAdjacency::reverse_family`](crate::csr_adjacency::CsrAdjacency::reverse_family))
//!   is still available for whole-family batch queries.
//!
//! Kind ids are persisted in the dictionary and never regenerated: once a type
//! has an id it keeps it for the life of the room's data, so stored relation
//! edges stay meaningful. The four relation types defined by the spec get fixed
//! ids ([`KNOWN_RELATION_TYPES`](crate::csr_adjacency::KNOWN_RELATION_TYPES)); any other type string is preserved and gets
//! the next free id on first sight. Ids above `u16::MAX` are a hard error.
//!
//! What is *not* stored here: whether an event is still visible. A relation
//! disappears when its source event is redacted, rejected or soft-failed, but
//! those are facts about the source event held elsewhere (and rejection is
//! reversible). Reads therefore take an [`EventVisibility`](crate::csr_adjacency::EventVisibility) filter and apply it
//! to the relation at query time; `prev` and `auth` are signed event-core fields
//! that redaction never changes, so they are not filtered. The transitive
//! auth-closure layer must read only [`CsrAdjacency::auth_of`](crate::csr_adjacency::CsrAdjacency::auth_of), never relations.
//!
//! Rows live in the State pool's shared graph store,
//! `<db_root>/mtgr-state/`, keyed by `(collection id, family, owner)`.
//! Referenced but not-yet-observed parents are ordinary nodes: every target is
//! allocated a short id, so the row set never drops an edge to a missing body.

use std::collections::{BTreeSet, HashMap};

/// The distinct `(target event id, kind)` edges one family claims for an owner.
type ClaimedEdges = BTreeSet<(Vec<u8>, u16)>;

use crate::csr::store::{CsrStore, RowRecord, ScopeId};
use crate::csr::{Edge, EdgeFamily, EdgeKey, ReverseCsr};
use crate::database::{Database, DatabaseTransaction};
use crate::edge_log::{EdgeLog, LoggedEvent};
use crate::layout::ShardType;
use crate::short_id::{GraphReader, ShortIdIndex};
use crate::storage::StorageError;
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Whether an append waits for its rows to reach stable storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendDurability {
    /// Sync the tail before returning. The default for every caller.
    Durable,
    /// Leave the rows in the page cache. The caller owes a
    /// [`CsrAdjacency::sync_tail`] before it publishes any claim covering them.
    Deferred,
}

/// Make every adjacency tail in `db` durable, whichever pool holds it.
///
/// A run that publishes a completion claim calls this first, so the claim
/// never covers rows that exist only in the page cache. It is idempotent and
/// cheap when nothing is pending.
///
/// # Errors
/// I/O errors from any tail.
pub fn sync_all_tails(db: &Database) -> Result<(), StorageError> {
    for pool in ShardType::ALL {
        let store = CsrStore::open(db.layout(), pool);
        store
            .with_write_lock(|| store.sync_tail())
            .map_err(StorageError::Io)?;
    }
    Ok(())
}

/// Family of an event's `prev_events`.
pub const PREV: EdgeFamily = EdgeFamily::plain(1);
/// Throwaway prototype family for the maintained reverse `AUTH` benchmark.
#[cfg(test)]
pub(crate) const PROTO_AUTH_CHILDREN: EdgeFamily = EdgeFamily::plain(5);

/// Family of an event's `auth_events`.
pub const AUTH: EdgeFamily = EdgeFamily::plain(2);
/// Family of an event's single relation (`m.relates_to`), typed by relation kind.
pub const RELATIONS: EdgeFamily = EdgeFamily::typed(3);

const FAMILIES: [EdgeFamily; 3] = [PREV, AUTH, RELATIONS];

/// Reverse of [`PREV`]: the row owned by a parent lists its children.
///
/// Derived from `PREV` and written with it; it is not one of the three
/// forward families, so snapshots and verification skip it.
pub const PREV_CHILDREN: EdgeFamily = EdgeFamily::plain(4);

/// Owner of the `PREV_CHILDREN` row that marks a scope's reverse rows as
/// complete. Short ids start at 1, so 0 is never a real parent.
const CHILDREN_MARKER_OWNER: u32 = 0;

/// Relation types defined by the Matrix spec, with fixed kind ids (1-based, in
/// this order).
pub const KNOWN_RELATION_TYPES: [&str; 4] =
    ["m.annotation", "m.reference", "m.replace", "m.thread"];

/// Decides whether an event's relation is currently visible.
pub trait EventVisibility {
    /// Whether `event_id`'s relation should be shown.
    fn is_visible(&self, event_id: &str) -> bool;
}

/// A filter that shows everything.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlwaysVisible;

impl EventVisibility for AlwaysVisible {
    fn is_visible(&self, _event_id: &str) -> bool {
        true
    }
}

impl<F: Fn(&str) -> bool> EventVisibility for F {
    fn is_visible(&self, event_id: &str) -> bool {
        self(event_id)
    }
}

/// An event's relation to another event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    /// The event this one relates to.
    pub target: String,
    /// The relation type string (`m.annotation`, or any other type).
    pub rel_type: String,
}

/// Borrowed form of [`Relation`] for recording.
#[derive(Debug, Clone, Copy)]
pub struct RelationRef<'a> {
    /// The event this one relates to.
    pub target: &'a str,
    /// The relation type string.
    pub rel_type: &'a str,
}

/// An event to record: its `prev` and `auth` events and, optionally, its
/// relation to another event.
#[derive(Debug, Clone, Copy)]
pub struct EventRecord<'a> {
    /// The event id.
    pub event_id: &'a str,
    /// The event's `prev_events`.
    pub prev: &'a [&'a str],
    /// The event's `auth_events`.
    pub auth: &'a [&'a str],
    /// The event's relation, if it has one.
    pub relation: Option<RelationRef<'a>>,
}

/// Result of [`CsrAdjacency::verify`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdjacencyVerifyReport {
    /// Violations; empty means consistent.
    pub problems: Vec<String>,
}

impl AdjacencyVerifyReport {
    /// Whether every invariant held.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.problems.is_empty()
    }
}

/// One room's Matrix adjacency.
#[derive(Debug, Clone, Copy)]
pub struct CsrAdjacency {
    events: ShortIdIndex,
    kinds: ShortIdIndex,
    /// The room's event collection id, when this scope was addressed by it
    /// (see [`Self::for_collection`]); the edge log needs it to find the scope
    /// again after a crash.
    room_collection: Option<[u8; 16]>,
}

/// Preserve the room-id-derived scope used before collection-based scopes
/// were introduced. Existing databases keep their persisted short-id maps and
/// CSR rows at these addresses.
fn legacy_collection(prefix: &str, room_id: &str) -> [u8; 16] {
    let mut canonical = prefix.as_bytes().to_vec();
    canonical.extend_from_slice(room_id.as_bytes());
    derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
}

fn utf8(bytes: Vec<u8>, what: &str) -> Result<String, StorageError> {
    String::from_utf8(bytes).map_err(|_| StorageError::Corrupt(format!("{what} is not UTF-8")))
}

/// One family's mutable rows, keyed by short id (`rows[0]` is unused).
#[derive(Debug, Clone)]
struct FamilyRows {
    family: EdgeFamily,
    recorded: Vec<bool>,
    rows: Vec<Vec<Edge>>,
}

impl FamilyRows {
    fn new(family: EdgeFamily, ids: usize) -> Self {
        Self {
            family,
            recorded: vec![false; ids],
            rows: vec![Vec::new(); ids],
        }
    }

    fn resize(&mut self, ids: usize) {
        if ids > self.rows.len() {
            self.recorded.resize(ids, false);
            self.rows.resize(ids, Vec::new());
        }
    }
}

/// A room's three forward CSR families.
#[derive(Debug, Clone)]
struct Snapshot {
    families: [FamilyRows; 3],
}

impl Snapshot {
    fn empty(ids: usize) -> Self {
        Self {
            families: FAMILIES.map(|family| FamilyRows::new(family, ids)),
        }
    }

    fn resize(&mut self, ids: usize) {
        for family in &mut self.families {
            family.resize(ids);
        }
    }
}

/// The comparable form of one family's requested edges: key bytes plus kind.
fn event_key_set(edges: &[EdgeKey<'_>]) -> BTreeSet<(Vec<u8>, u16)> {
    edges
        .iter()
        .map(|edge| (edge.target.to_vec(), edge.kind))
        .collect()
}

impl CsrAdjacency {
    /// Address `room_id`'s adjacency.
    ///
    /// The event-id maps and relation-kind dictionary live in the
    /// [`ShardType::State`] pool. CSR rows live in that pool's shared graph
    /// store, `<db_root>/mtgr-state/`, outside the packs. Event bodies stay in
    /// `EventDag`. This keeps the original room-id-derived storage scopes so
    /// existing rooms retain their adjacency.
    #[must_use]
    pub fn new(room_id: &str) -> Self {
        Self::scoped(
            legacy_collection("matrix-adjacency:", room_id),
            legacy_collection("matrix-relation-kinds:", room_id),
        )
    }

    /// Address a fresh adjacency scope using the room's 16-byte event
    /// collection id. This scope is different from the legacy room-id-derived
    /// scope used by [`Self::new`]; existing data must be re-exported before
    /// switching to it. This explicit migration requirement avoids silently
    /// treating legacy adjacency as empty.
    ///
    /// For callers that resolve an event to its room through the locator and
    /// never see the room id string. The scope is derived from the collection
    /// id alone, so every caller using the same id reaches the same rows.
    #[must_use]
    pub fn for_collection(room_collection: &[u8; 16]) -> Self {
        let derive = |prefix: &[u8]| {
            let mut canonical = prefix.to_vec();
            canonical.extend_from_slice(room_collection);
            derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
        };
        let mut adjacency = Self::scoped(
            derive(b"matrix-adjacency-c:"),
            derive(b"matrix-relation-kinds-c:"),
        );
        adjacency.room_collection = Some(*room_collection);
        adjacency
    }

    fn scoped(events: [u8; 16], kinds: [u8; 16]) -> Self {
        let pool = ShardType::State;
        Self {
            events: ShortIdIndex::new(pool, events),
            kinds: ShortIdIndex::new(pool, kinds).with_max_id(u32::from(u16::MAX)),
            room_collection: None,
        }
    }

    fn csr_collection(&self) -> [u8; 16] {
        let (_, events) = self.events.scope();
        csr_scope_for(&events)
    }

    fn scope(&self) -> ScopeId {
        ScopeId(self.csr_collection())
    }

    fn store<R: GraphReader + ?Sized>(&self, db: &R) -> CsrStore {
        CsrStore::open(db.layout(), self.events.scope().0)
    }

    /// Load this scope's rows from the graph store into a mutable snapshot.
    fn load_snapshot(&self, db: &Database, ids: usize) -> Result<Snapshot, StorageError> {
        let data = self
            .store(db)
            .read_scope(self.scope())
            .map_err(StorageError::Io)?;
        // Do not size the snapshot from the counter: a parked or corrupt
        // counter can be near `u32::MAX`, and rows are the only thing that
        // actually needs storage. Callers grow to the allocated counter after
        // a successful allocation.
        let mut snapshot = Snapshot::empty(0);
        for row in data.rows {
            if row.family.id == PREV_CHILDREN.id || row.family.id == PURGE_APPLIED.id {
                continue;
            }
            let Some(index) = FAMILIES
                .iter()
                .position(|family| family.id == row.family.id && family.typed == row.family.typed)
            else {
                return Err(StorageError::Corrupt(format!(
                    "csr adjacency family {} is not part of this scope",
                    row.family.id
                )));
            };
            let owner = row.owner as usize;
            // The graph store is outside the WAL. A rolled-back id allocation
            // leaves rows at or above the committed counter, and a torn or
            // foreign record can name an arbitrary owner. Such rows are not
            // part of this scope's committed graph: drop them on read. They are
            // harmless leftovers, and a later commit overwrites the slot.
            if owner == 0
                || owner >= ids
                || row
                    .edges
                    .iter()
                    .any(|edge| edge.target == 0 || edge.target as usize >= ids)
            {
                continue;
            }
            if owner >= snapshot.families[index].rows.len() {
                snapshot.resize(owner.saturating_add(1));
            }
            snapshot.families[index].recorded[owner] = true;
            snapshot.families[index].rows[owner] = row.edges;
        }
        Ok(snapshot)
    }

    /// The room's event short-id scope. It is also the ordinal space that
    /// auth-closure bitmaps over this room are built in.
    #[must_use]
    pub const fn events_index(&self) -> ShortIdIndex {
        self.events
    }

    /// Record an event's `prev`, `auth` and (optional) relation in one
    /// transaction, allocating short ids for the event and every referenced
    /// event. Returns the event's short id.
    ///
    /// Re-recording identical data is a no-op; different `prev` or `auth` for an
    /// already-recorded event is a collision error (an event's core fields never
    /// change).
    ///
    /// A relation can be added but never changed or removed. Passing `None`
    /// therefore means "no relation known", not "no relation", so redaction
    /// never has to be stored.
    ///
    /// Recording also updates each parent's [`PREV_CHILDREN`] row in the same
    /// append, which is what makes [`Self::children_of`] a point read. A scope
    /// written before that family existed is backfilled from its stored `PREV`
    /// rows once, on its first write.
    ///
    /// # Errors
    /// Storage errors, a collision as described, or an error if the relation
    /// kind dictionary is out of `u16` ids.
    pub fn record_event(
        &self,
        db: &Database,
        event_id: &str,
        prev: &[&str],
        auth: &[&str],
        relation: Option<RelationRef<'_>>,
    ) -> Result<u32, StorageError> {
        let recorded = self.record_events(
            db,
            &[EventRecord {
                event_id,
                prev,
                auth,
                relation,
            }],
        )?;
        Ok(recorded[0])
    }

    /// Record several events in **one** transaction, returning each event's
    /// short id in order.
    ///
    /// Either every event is recorded or none is. Events may reference each
    /// other, and the same event twice with the same data is a no-op.
    ///
    /// Like [`Self::record_event`], this maintains the [`PREV_CHILDREN`] rows
    /// for every parent in the batch.
    ///
    /// # Errors
    /// As [`Self::record_event`], applied to the whole batch.
    pub fn record_events(
        &self,
        db: &Database,
        events: &[EventRecord<'_>],
    ) -> Result<Vec<u32>, StorageError> {
        self.record_events_with(db, events, AppendDurability::Durable)
    }

    /// [`Self::record_events`] with an explicit durability for the appended rows.
    ///
    /// Under [`AppendDurability::Deferred`] the rows are readable immediately
    /// but are not durable until [`Self::sync_tail`] returns; the caller must
    /// call it before publishing any claim that covers them.
    ///
    /// # Errors
    /// As [`Self::record_event`], applied to the whole batch.
    pub fn record_events_with(
        &self,
        db: &Database,
        events: &[EventRecord<'_>],
        durability: AppendDurability,
    ) -> Result<Vec<u32>, StorageError> {
        self.record_events_inner(db, events, durability, None)
    }

    /// [`Self::record_events_with`], also writing the events' edges to the
    /// journaled `log` before the rows reach the CSR tail.
    ///
    /// The log entry is durable with the next WAL group commit, so the tail does
    /// not have to be: after a crash [`Self::recover_edge_log`] rebuilds any
    /// rows the tail lost. The rows are followed, in the same append, by a
    /// watermark row naming the last log entry they cover, which is what makes
    /// recovery replay only what is missing.
    ///
    /// If the tail append fails after the log entry is published, the entry is
    /// left in the log rather than rolled back. A later recovery may therefore
    /// add edges for an event whose recording failed. Logging first is what
    /// makes recovery possible, so the entry is not removed.
    ///
    /// # Errors
    /// As [`Self::record_event`], plus an error if this adjacency was not
    /// addressed by a room collection (see [`Self::for_collection`]), since the
    /// log must be able to find the scope again.
    pub fn record_events_logged(
        &self,
        db: &Database,
        log: &EdgeLog,
        events: &[EventRecord<'_>],
        durability: AppendDurability,
    ) -> Result<Vec<u32>, StorageError> {
        self.record_events_inner(db, events, durability, Some(log))
    }

    fn record_events_inner(
        &self,
        db: &Database,
        events: &[EventRecord<'_>],
        durability: AppendDurability,
        log: Option<&EdgeLog>,
    ) -> Result<Vec<u32>, StorageError> {
        let store = self.store(db);
        if events.is_empty() {
            if durability == AppendDurability::Durable {
                store.sync_tail().map_err(StorageError::Io)?;
            }
            return Ok(Vec::new());
        }
        let logged_scope = match log {
            None => None,
            Some(_) => Some(self.room_collection.ok_or_else(|| {
                StorageError::Corrupt(
                    "edge logging needs an adjacency addressed by its room collection".to_owned(),
                )
            })?),
        };
        // The read, merge and append must be one critical section: two
        // recordings that both read the same stored row would otherwise each
        // validate against the old state, and the later append would silently
        // replace the earlier one.
        let owners = store.with_write_lock(|| -> Result<_, StorageError> {
            let prepared = self.prepare_events(db, events)?;

            // Validate everything that can collide before allocating any id,
            // so a rejected batch records nothing. Only the batch's own owners
            // are read; the rest of the collection is not touched.
            let (owner_ids, mut stored) = self.load_stored(db, &prepared)?;
            self.check_conflicts(db, &prepared, &owner_ids, &stored)?;

            // Allocate every id in one call, exactly as the short-id layer did.
            let (keys, starts) = batch_keys(&prepared);
            let ids = self.events.get_or_create(db, &keys)?;
            let (owners, mut changed_rows) = merge_batch(&mut stored, &prepared, &starts, &ids)?;
            // Every logged `prev` edge feeds the reverse rows, not just the
            // forward rows that changed: after a partial tail loss a forward
            // row can survive while its reverse row is gone, and replay must
            // restore it. Merging a child already listed changes nothing.
            let mut prev_pairs: Vec<(u32, u32)> = Vec::new();
            for (event, &start) in prepared.iter().zip(&starts) {
                let first = start.saturating_add(1);
                for offset in 0..event.prev.len() {
                    prev_pairs.push((ids[start], ids[first.saturating_add(offset)]));
                }
            }
            self.add_child_rows(db, &mut changed_rows, &prev_pairs)?;
            let appended = !changed_rows.is_empty();
            // Log the edges first, in a journaled transaction: if it fails
            // nothing has reached the tail, and if the rows are later lost the
            // log still has them. The log's sequence order is the order rows are
            // applied in, because this whole section holds the store's write lock.
            let log_seq = match (log, logged_scope) {
                (Some(log), Some(scope)) => {
                    let logged: Vec<LoggedEvent> = events
                        .iter()
                        .map(|event| LoggedEvent {
                            scope,
                            event_id: event.event_id.to_owned(),
                            prev: event.prev.iter().map(|p| (*p).to_owned()).collect(),
                            auth: event.auth.iter().map(|a| (*a).to_owned()).collect(),
                            relation: event
                                .relation
                                .map(|r| (r.target.to_owned(), r.rel_type.to_owned())),
                        })
                        .collect();
                    Some(log.append(db, &logged)?)
                }
                _ => None,
            };
            if appended {
                // Append under the lock, but never fsync under it. The sync is
                // the expensive part of a durable append, and holding the store
                // write lock across it stalls every overlapping reader for the
                // duration of the flush. The rows are readable as soon as they
                // are written; `sync_tail` below makes them durable after the
                // lock is released, and this call does not return before then.
                store
                    .writer(db)
                    .and_then(|writer| {
                        writer.append_rows_deferred_locked(self.scope(), &changed_rows)
                    })
                    .map_err(StorageError::Io)?;
            }
            if let Some(seq) = log_seq {
                // The watermark follows the rows in the same tail, so a
                // valid watermark means the rows before it are valid too.
                // It only advances from the entry just before this batch:
                // if an earlier batch's tail append failed, its entry is
                // still unapplied, and moving past it would hide it from
                // recovery. The watermark then stays put until
                // `recover_edge_log` replays the gap. An unchanged batch logs an
                // entry without appending rows and must advance it too, or that
                // entry would pin the watermark and block pruning.
                let before = seq.saturating_sub(u64::try_from(events.len()).unwrap_or(u64::MAX));
                if read_watermark_locked(&store).map_err(StorageError::Io)? == before {
                    write_watermark_locked(&store, db, seq).map_err(StorageError::Io)?;
                }
            }
            Ok(owners)
        })?;

        // Durability barrier: outside the write lock, so readers can proceed
        // while the tail is flushed. A concurrent seal may have folded these
        // rows into a segment (and synced it) and truncated the tail; syncing
        // an empty or unrelated tail is harmless, and the rows are durable
        // either way. A durable call syncs even when it appended nothing: its
        // rows may already be present from an earlier deferred call, and the
        // caller is told they are durable on return. The sync is cheap when
        // nothing is pending.
        if durability == AppendDurability::Durable {
            store.sync_tail().map_err(StorageError::Io)?;
        }
        Ok(owners)
    }

    /// Add the reverse rows for the batch's `(child, parent)` `prev` pairs: for each parent,
    /// the stored children plus the batch's new ones. Runs under the store's
    /// write lock, so the read-merge-append is atomic.
    ///
    /// A scope without the completeness marker (a fresh scope, or one written
    /// before the reverse family existed) is first backfilled from its stored
    /// `PREV` rows, once, and then marked.
    fn add_child_rows(
        &self,
        db: &Database,
        changed: &mut Vec<RowRecord>,
        prev_pairs: &[(u32, u32)],
    ) -> Result<(), StorageError> {
        let store = self.store(db);
        let mut added: std::collections::BTreeMap<u32, BTreeSet<Edge>> =
            std::collections::BTreeMap::new();
        let marked = store
            .read_rows_locked(self.scope(), &[(PREV_CHILDREN, CHILDREN_MARKER_OWNER)])
            .map_err(StorageError::Io)?
            .pop()
            .flatten()
            .is_some();
        if !marked {
            for (child, parents) in store
                .read_family_locked(self.scope(), PREV)
                .map_err(StorageError::Io)?
            {
                for edge in parents {
                    added.entry(edge.target).or_default().insert(Edge {
                        target: child,
                        kind: 0,
                    });
                }
            }
            changed.push(RowRecord {
                family: PREV_CHILDREN,
                owner: CHILDREN_MARKER_OWNER,
                edges: vec![Edge { target: 0, kind: 0 }],
            });
        }
        for &(child, parent) in prev_pairs {
            added.entry(parent).or_default().insert(Edge {
                target: child,
                kind: 0,
            });
        }
        if added.is_empty() {
            return Ok(());
        }
        let keys: Vec<(EdgeFamily, u32)> = added.keys().map(|&p| (PREV_CHILDREN, p)).collect();
        let existing = store
            .read_rows_locked(self.scope(), &keys)
            .map_err(StorageError::Io)?;
        for ((_, parent), old) in keys.iter().zip(existing) {
            let mut children: BTreeSet<Edge> = old.unwrap_or_default().into_iter().collect();
            let before = children.len();
            children.extend(added.remove(parent).unwrap_or_default());
            if children.len() != before {
                changed.push(RowRecord {
                    family: PREV_CHILDREN,
                    owner: *parent,
                    edges: children.into_iter().collect(),
                });
            }
        }
        Ok(())
    }

    /// The events that list `short_id` in their `prev_events`: one point read
    /// of the maintained reverse family, no whole-room scan. `None` when no
    /// child is recorded.
    ///
    /// A scope recorded before the reverse family existed has no completeness
    /// marker until its next write; for it this falls back to deriving the
    /// answer with [`Self::reverse_family`] (a whole-scope read), so the result
    /// is the same, only slower.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt segment.
    pub fn children_of<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        short_id: u32,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        let mut rows = self
            .store(db)
            .read_rows(
                self.scope(),
                &[
                    (PREV_CHILDREN, short_id),
                    (PREV_CHILDREN, CHILDREN_MARKER_OWNER),
                ],
            )
            .map_err(StorageError::Io)?;
        let marker = rows.pop().flatten();
        if let Some(edges) = rows.pop().flatten() {
            return Ok(Some(edges.into_iter().map(|edge| edge.target).collect()));
        }
        if marker.is_some() {
            return Ok(None);
        }
        Ok(self
            .reverse_family(db, PREV)?
            .children_of(short_id)
            .map(|edges| edges.iter().map(|edge| edge.target).collect()))
    }

    /// Throwaway prototype: maintain a reverse `AUTH` family (id 5) for a batch
    /// of `(owner, auth targets)` pairs, in its own transaction after the
    /// event write. Test-only; the production write path never calls it.
    #[cfg(test)]
    pub(crate) fn proto_maintain_auth_children(
        &self,
        db: &Database,
        batch: &[(u32, Vec<u32>)],
    ) -> Result<(), StorageError> {
        let store = self.store(db);
        store.with_write_lock(|| {
            let mut added: std::collections::BTreeMap<u32, BTreeSet<Edge>> =
                std::collections::BTreeMap::new();
            for (owner, parents) in batch {
                for &parent in parents {
                    added.entry(parent).or_default().insert(Edge {
                        target: *owner,
                        kind: 0,
                    });
                }
            }
            let keys: Vec<(EdgeFamily, u32)> =
                added.keys().map(|&p| (PROTO_AUTH_CHILDREN, p)).collect();
            let existing = store
                .read_rows_locked(self.scope(), &keys)
                .map_err(StorageError::Io)?;
            let mut changed = Vec::new();
            for ((_, parent), old) in keys.iter().zip(existing) {
                let mut children: BTreeSet<Edge> = old.unwrap_or_default().into_iter().collect();
                children.extend(added.remove(parent).unwrap_or_default());
                changed.push(RowRecord {
                    family: PROTO_AUTH_CHILDREN,
                    owner: *parent,
                    edges: children.into_iter().collect(),
                });
            }
            store
                .writer(db)
                .and_then(|writer| writer.append_rows_locked(self.scope(), &changed))
                .map_err(StorageError::Io)
        })
    }

    /// Throwaway prototype: one point read of the prototype reverse `AUTH` row.
    #[cfg(test)]
    pub(crate) fn proto_auth_children_of(
        &self,
        db: &Database,
        short_id: u32,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        let mut rows = self
            .store(db)
            .read_rows(self.scope(), &[(PROTO_AUTH_CHILDREN, short_id)])
            .map_err(StorageError::Io)?;
        Ok(rows
            .pop()
            .flatten()
            .map(|edges| edges.into_iter().map(|edge| edge.target).collect()))
    }

    /// Seal the scope's tail into an immutable segment (benchmark helper).
    #[cfg(test)]
    pub(crate) fn proto_seal(&self, db: &Database) -> Result<(), StorageError> {
        self.store(db).seal().map_err(StorageError::Io)
    }

    /// Write forward `PREV` rows the way a pre-reverse-family build did: no
    /// reverse rows and no completeness marker. Each pair is (child, parent).
    #[cfg(test)]
    pub(crate) fn record_legacy_prev(
        &self,
        db: &Database,
        pairs: &[(&str, &str)],
    ) -> Result<(), StorageError> {
        let keys: Vec<&[u8]> = pairs
            .iter()
            .flat_map(|(child, parent)| [child.as_bytes(), parent.as_bytes()])
            .collect();
        let ids = self.events.get_or_create(db, &keys)?;
        let rows: Vec<RowRecord> = ids
            .chunks(2)
            .map(|pair| RowRecord {
                family: PREV,
                owner: pair[0],
                edges: vec![Edge {
                    target: pair[1],
                    kind: 0,
                }],
            })
            .collect();
        let store = self.store(db);
        store
            .with_write_lock(|| {
                store
                    .writer(db)
                    .and_then(|writer| writer.append_rows_locked(self.scope(), &rows))
            })
            .map_err(StorageError::Io)
    }

    /// Make every row appended so far durable, including rows written under
    /// [`AppendDurability::Deferred`]. Idempotent, and cheap when nothing is
    /// pending.
    ///
    /// # Errors
    /// I/O errors.
    pub fn sync_tail(&self, db: &Database) -> Result<(), StorageError> {
        let store = self.store(db);
        store
            .with_write_lock(|| store.sync_tail())
            .map_err(StorageError::Io)
    }

    /// Turn each event into edge keys, allocating relation kind ids once per
    /// distinct type.
    fn prepare_events<'a>(
        &self,
        db: &Database,
        events: &[EventRecord<'a>],
    ) -> Result<Vec<PreparedEvent<'a>>, StorageError> {
        let mut kinds: HashMap<&str, u16> = HashMap::new();
        let mut prepared = Vec::with_capacity(events.len());
        for event in events {
            let relation = match event.relation {
                Some(relation) => {
                    let kind = if let Some(kind) = kinds.get(relation.rel_type) {
                        *kind
                    } else {
                        let kind = self.kind_id(db, relation.rel_type)?;
                        kinds.insert(relation.rel_type, kind);
                        kind
                    };
                    vec![EdgeKey {
                        target: relation.target.as_bytes(),
                        kind,
                    }]
                }
                None => Vec::new(),
            };
            prepared.push(PreparedEvent {
                owner: event.event_id.as_bytes(),
                prev: event
                    .prev
                    .iter()
                    .map(|id| EdgeKey::plain(id.as_bytes()))
                    .collect(),
                auth: event
                    .auth
                    .iter()
                    .map(|id| EdgeKey::plain(id.as_bytes()))
                    .collect(),
                relation,
            });
        }
        Ok(prepared)
    }

    /// The short ids of the batch's already-known owners and their stored rows.
    fn load_stored<'a>(
        &self,
        db: &Database,
        prepared: &[PreparedEvent<'a>],
    ) -> Result<(HashMap<&'a [u8], u32>, StoredRows), StorageError> {
        let mut owner_ids: HashMap<&[u8], u32> = HashMap::new();
        let mut stored_keys: Vec<(EdgeFamily, u32)> = Vec::new();
        for event in prepared {
            if let Some(owner) = self.events.lookup(db, event.owner)? {
                owner_ids.insert(event.owner, owner);
                for family in FAMILIES {
                    stored_keys.push((family, owner));
                }
            }
        }
        let rows = self
            .store(db)
            .read_rows_locked(self.scope(), &stored_keys)
            .map_err(StorageError::Io)?;
        let mut stored: StoredRows = HashMap::new();
        for ((family, owner), row) in stored_keys.iter().zip(rows) {
            if let Some(row) = row {
                stored.insert((family.id, *owner), row);
            }
        }
        Ok((owner_ids, stored))
    }

    /// Reject a batch whose edges contradict each other or what is stored.
    fn check_conflicts(
        &self,
        db: &Database,
        prepared: &[PreparedEvent<'_>],
        owner_ids: &HashMap<&[u8], u32>,
        stored: &StoredRows,
    ) -> Result<(), StorageError> {
        let mut claims: HashMap<&[u8], Vec<ClaimedEdges>> = HashMap::new();
        for event in prepared {
            let wanted = [
                event_key_set(&event.prev),
                event_key_set(&event.auth),
                event_key_set(&event.relation),
            ];
            if let Some(claim) = claims.get_mut(event.owner) {
                claim_again(claim, &wanted)?;
            } else {
                claims.insert(event.owner, wanted.to_vec());
            }
            if let Some(owner) = owner_ids.get(event.owner).copied() {
                self.check_stored(db, stored, owner, &wanted)?;
            }
        }
        Ok(())
    }

    /// Compare an owner's requested edges with the rows already stored for it.
    fn check_stored(
        &self,
        db: &Database,
        stored: &StoredRows,
        owner: u32,
        wanted: &[ClaimedEdges; 3],
    ) -> Result<(), StorageError> {
        for (index, family) in FAMILIES.iter().enumerate() {
            let Some(stored_edges) = stored.get(&(family.id, owner)) else {
                continue;
            };
            let stored_set = self.resolve_row(db, stored_edges)?;
            let requested = &wanted[index];
            // Relations are add-only and absent means "unknown".
            let conflicts = if family.typed {
                !requested.is_empty() && !stored_set.is_empty() && stored_set != *requested
            } else {
                stored_set != *requested
            };
            if conflicts {
                return Err(StorageError::Collision(DIFFERS_FROM_RECORDED.to_owned()));
            }
        }
        Ok(())
    }

    /// Resolve a stored row to `(event id bytes, kind)` so it can be compared
    /// with keys that have no short ids yet.
    fn resolve_row(
        &self,
        db: &Database,
        row: &[Edge],
    ) -> Result<BTreeSet<(Vec<u8>, u16)>, StorageError> {
        let ids: Vec<u32> = row.iter().map(|edge| edge.target).collect();
        let keys = self.events.resolve(db, &ids)?;
        row.iter()
            .zip(keys)
            .map(|(edge, key)| {
                key.map(|bytes| (bytes, edge.kind))
                    .ok_or_else(|| StorageError::Corrupt("edge target does not resolve".to_owned()))
            })
            .collect()
    }

    /// The room-local short id of `event_id`, if it has been recorded or
    /// referenced.
    ///
    /// # Errors
    /// Returns an error on a read failure.
    pub fn short_id<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        event_id: &str,
    ) -> Result<Option<u32>, StorageError> {
        self.events.lookup(db, event_id.as_bytes())
    }

    /// Resolve room-local short ids back to event ids; `None` for an unknown
    /// id.
    ///
    /// This is the reverse direction of [`Self::short_id`] for callers that
    /// walk a [`ReverseCsr`](crate::csr::ReverseCsr) built by
    /// [`Self::reverse_family`].
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt reverse record, or a
    /// stored key that is not UTF-8.
    pub fn resolve_event_ids<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        short_ids: &[u32],
    ) -> Result<Vec<Option<String>>, StorageError> {
        self.events
            .resolve(db, short_ids)?
            .into_iter()
            .map(|key| match key {
                None => Ok(None),
                Some(bytes) => utf8(bytes, "event id").map(Some),
            })
            .collect()
    }

    /// The event's `prev_events`, or `None` if the event was never recorded.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn prev_of<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        event_id: &str,
    ) -> Result<Option<Vec<String>>, StorageError> {
        self.targets_of(db, event_id, PREV)
    }

    /// The event's `auth_events`, or `None` if the event was never recorded.
    /// This is the only input an auth closure may use.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn auth_of<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        event_id: &str,
    ) -> Result<Option<Vec<String>>, StorageError> {
        self.targets_of(db, event_id, AUTH)
    }

    /// The raw edge list of `short_id` in `family`, as stored in the CSR.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt snapshot.
    pub fn family_edges<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        short_id: u32,
        family: EdgeFamily,
    ) -> Result<Option<Vec<Edge>>, StorageError> {
        let mut rows = self
            .store(db)
            .read_rows(self.scope(), &[(family, short_id)])
            .map_err(StorageError::Io)?;
        Ok(rows.pop().flatten())
    }

    /// Every recorded `AUTH` row as `(owner short id, target short ids)`.
    ///
    /// Includes owners whose row is present but empty. Neither side is resolved
    /// to an event id, so a missing reverse mapping is reported by the caller
    /// instead of aborting the read.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt segment.
    pub fn auth_target_ids<R: GraphReader + ?Sized>(
        &self,
        db: &R,
    ) -> Result<Vec<(u32, Vec<u32>)>, StorageError> {
        let rows = self
            .store(db)
            .read_family(self.scope(), AUTH)
            .map_err(StorageError::Io)?;
        Ok(rows
            .into_iter()
            .map(|(owner, edges)| (owner, edges.into_iter().map(|edge| edge.target).collect()))
            .collect())
    }

    /// Every recorded row of one family for this scope, in one store read.
    ///
    /// Use this instead of calling [`Self::family_edges`] in a loop: the store
    /// is append-only, so one family read is far cheaper than N point reads
    /// that each replay the tail. Sealed segments seek only this family's
    /// directory range.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt segment.
    pub(crate) fn family_rows(
        &self,
        db: &Database,
        family: EdgeFamily,
    ) -> Result<Vec<(u32, Vec<Edge>)>, StorageError> {
        self.store(db)
            .read_family(self.scope(), family)
            .map_err(StorageError::Io)
    }

    /// Every event id the room's `prev_events` and `auth_events` rows point
    /// at, whether or not that event has been stored.
    ///
    /// A caller that wants the *missing* parents checks this set against its
    /// event store: "referenced but not present" is derived, never stored.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt segment, or a target that
    /// does not resolve to an event id.
    pub fn referenced_event_ids(&self, db: &Database) -> Result<BTreeSet<String>, StorageError> {
        let (mut prev, auth) = self.referenced_event_ids_by_family(db)?;
        prev.extend(auth);
        Ok(prev)
    }

    /// Every event id the room's `prev_events` and `auth_events` point at,
    /// separated by family.
    ///
    /// A consumer that tracks "referenced but absent" work per family needs to
    /// know which edges a late arrival would satisfy: a `prev` arrival changes
    /// descendant state groups, an `auth` arrival changes verdicts.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt segment, or a target that
    /// does not resolve to an event id.
    pub fn referenced_event_ids_by_family(
        &self,
        db: &Database,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>), StorageError> {
        Ok((
            self.resolved_family_targets(db, PREV)?,
            self.resolved_family_targets(db, AUTH)?,
        ))
    }

    fn resolved_family_targets(
        &self,
        db: &Database,
        family: EdgeFamily,
    ) -> Result<BTreeSet<String>, StorageError> {
        let store = self.store(db);
        let mut targets = BTreeSet::new();
        for (_, edges) in store
            .read_family(self.scope(), family)
            .map_err(StorageError::Io)?
        {
            targets.extend(edges.iter().map(|edge| edge.target));
        }
        let ids: Vec<u32> = targets.into_iter().collect();
        self.events
            .resolve(db, &ids)?
            .into_iter()
            .map(|key| {
                key.ok_or_else(|| StorageError::Corrupt("edge target does not resolve".to_owned()))
                    .and_then(|bytes| utf8(bytes, "event id"))
            })
            .collect()
    }

    /// Build a sparse parent-to-children CSR view for one forward family.
    ///
    /// This reads the family's stored rows once (no event-visibility filter:
    /// for `RELATIONS` the caller must check each child's visibility) and is
    /// intended for a batch of reverse lookups; callers should reuse the
    /// returned view rather than rebuilding it per parent. Reverse rows are
    /// derived from the canonical forward rows, so purges and relation unions
    /// are reflected automatically.
    ///
    /// For one parent's children, prefer [`Self::children_of`]: it reads the
    /// maintained [`PREV_CHILDREN`] family in one point read. Build this view
    /// when a request has enough `PREV` parents that the single derive beats
    /// one point read each, or when another family such as `RELATIONS` is
    /// needed.
    ///
    /// # Errors
    /// Returns an error on a read failure, corrupt segment, or an index too
    /// large for `u32` offsets.
    pub fn reverse_family<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        family: EdgeFamily,
    ) -> Result<ReverseCsr, StorageError> {
        let rows = self
            .store(db)
            .read_family(self.scope(), family)
            .map_err(StorageError::Io)?;
        ReverseCsr::from_forward_rows(family, &rows)
            .map_err(|error| StorageError::Corrupt(format!("could not build reverse CSR: {error}")))
    }

    /// The event's relation if it has one and `visibility` shows it.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt record, or a relation kind
    /// missing from the dictionary.
    pub fn relation_of(
        &self,
        db: &Database,
        event_id: &str,
        visibility: &dyn EventVisibility,
    ) -> Result<Option<Relation>, StorageError> {
        if !visibility.is_visible(event_id) {
            return Ok(None);
        }
        let Some(id) = self.short_id(db, event_id)? else {
            return Ok(None);
        };
        let Some(stored) = self.family_edges(db, id, RELATIONS)? else {
            return Ok(None);
        };
        let Some(edge) = stored.first() else {
            return Ok(None);
        };
        let target = self
            .events
            .resolve(db, &[edge.target])?
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| StorageError::Corrupt("relation target does not resolve".to_owned()))?;
        let rel_type = self
            .relation_type(db, edge.kind)?
            .ok_or_else(|| StorageError::Corrupt("relation kind not in dictionary".to_owned()))?;
        Ok(Some(Relation {
            target: utf8(target, "relation target")?,
            rel_type,
        }))
    }

    /// The persisted kind id of `rel_type`, allocating one on first sight.
    ///
    /// # Errors
    /// An error (publishing nothing) once the dictionary would exceed `u16::MAX`
    /// ids, or a storage error.
    pub fn kind_id(&self, db: &Database, rel_type: &str) -> Result<u16, StorageError> {
        let mut keys: Vec<&[u8]> = KNOWN_RELATION_TYPES.iter().map(|t| t.as_bytes()).collect();
        keys.push(rel_type.as_bytes());
        let ids = self.kinds.get_or_create(db, &keys)?;
        ids.last()
            .and_then(|id| u16::try_from(*id).ok())
            .ok_or_else(|| StorageError::Corrupt("relation kind id exceeds u16".to_owned()))
    }

    /// The relation type string of a kind id.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn relation_type(&self, db: &Database, kind: u16) -> Result<Option<String>, StorageError> {
        self.kinds
            .resolve(db, &[u32::from(kind)])?
            .into_iter()
            .next()
            .flatten()
            .map(|bytes| utf8(bytes, "relation type"))
            .transpose()
    }

    /// Park the kind counter so tests can reach the `u16` ceiling.
    #[cfg(test)]
    pub(crate) fn set_kind_counter_for_test(
        &self,
        db: &Database,
        next: u32,
    ) -> Result<(), StorageError> {
        self.kinds.set_counter_for_test(db, next)
    }

    /// The dictionary id of `rel_type` if it has one, without allocating.
    #[cfg(test)]
    pub(crate) fn kinds_lookup_for_test(&self, db: &Database, rel_type: &str) -> Option<u32> {
        self.kinds.lookup(db, rel_type.as_bytes()).unwrap()
    }

    /// Record that the purge `(seq, token)` is applied to this scope, as a
    /// crash can leave it.
    #[cfg(test)]
    pub(crate) fn mark_purge_applied_for_test(&self, db: &Database, seq: u64, token: u64) {
        let store = self.store(db);
        store
            .with_write_lock(|| write_purge_applied_locked(&store, db, self.scope(), seq, token))
            .unwrap();
        store.sync_tail().unwrap();
    }

    /// The next relation-kind id this scope will allocate.
    #[cfg(test)]
    pub(crate) fn kind_counter_for_test(&self, db: &Database) -> Result<u32, StorageError> {
        self.kinds.counter(db)
    }

    /// Check the event and kind scopes and that every stored relation kind
    /// resolves.
    ///
    /// # Errors
    /// Returns an error only when a record cannot be read.
    pub fn verify(&self, db: &Database) -> Result<AdjacencyVerifyReport, StorageError> {
        let mut report = AdjacencyVerifyReport::default();
        report.problems.extend(
            self.events
                .verify(db)?
                .problems
                .into_iter()
                .map(|problem| format!("events: {problem}")),
        );
        report.problems.extend(
            self.kinds
                .verify(db)?
                .problems
                .into_iter()
                .map(|problem| format!("relation kinds: {problem}")),
        );
        let next = self.events.counter(db)?;
        let snapshot = self.load_snapshot(db, next as usize)?;
        for family in &snapshot.families {
            for (id, row) in family.rows.iter().enumerate() {
                for edge in row {
                    if edge.target == 0 || edge.target >= next {
                        report.problems.push(format!(
                            "csr id {id}: family {} target {} is out of range",
                            family.family.id, edge.target
                        ));
                    }
                }
                if family.family.typed {
                    for edge in row {
                        if self.relation_type(db, edge.kind)?.is_none() {
                            report.problems.push(format!(
                                "csr id {id}: relation kind missing from dictionary"
                            ));
                        }
                    }
                }
            }
        }
        Ok(report)
    }

    /// Remove the room's adjacency and relation-kind dictionary in one commit:
    /// the kind ids live and die with the edges that use them.
    ///
    /// Callers must serialize this with every other writer to the same scope.
    /// The State deletion commits before the graph tombstone is appended, so a
    /// concurrent `record_event` between those steps can be tombstoned without
    /// being covered by the purge marker.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged or committed.
    pub fn purge(&self, db: &Database) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        self.stage_purge(&txn)?;
        txn.commit()?;
        self.purge_graph(db)
    }

    /// Hide the scope's CSR rows in the shared graph store with a tombstone.
    ///
    /// This is separate from [`Self::stage_purge`] because the graph store is
    /// outside the transaction layer. Call it after the transaction that drops
    /// the event-id maps commits.
    ///
    /// Callers must serialize this with every other writer to the same scope: a
    /// write that lands between the State deletion and this call can be
    /// tombstoned without being covered by the purge marker.
    ///
    /// # Errors
    /// Returns an error if the tombstone cannot be appended.
    pub fn purge_graph(&self, db: &Database) -> Result<(), StorageError> {
        let store = self.store(db);
        let scope = self.scope();
        let log = EdgeLog::new();
        store.with_write_lock(|| -> Result<(), StorageError> {
            // The marker and tombstone share the store write lock with logged
            // appends, so later events receive larger log sequences and remain
            // eligible for recovery after this purge.
            let marker = match self.room_collection {
                Some(room_collection) => Some(log.append_scope_purge(db, room_collection)?),
                None => None,
            };
            store
                .writer(db)
                .and_then(|writer| writer.purge_scope_locked(scope))
                .map_err(StorageError::Io)?;
            if let Some((seq, token)) = marker {
                // Record that this purge is applied to the scope, in the same
                // tail right after the tombstone. Recovery then never replays
                // the tombstone over writes that came after it, whether or not
                // those writes were logged.
                write_purge_applied_locked(&store, db, scope, seq, token)
                    .map_err(StorageError::Io)?;
                // The tombstone is applied, so the marker is no longer a gap.
                // Move the watermark over it only when it was contiguous;
                // otherwise an earlier unapplied entry must stay visible.
                if read_watermark_locked(&store).map_err(StorageError::Io)? == seq.saturating_sub(1)
                {
                    write_watermark_locked(&store, db, seq).map_err(StorageError::Io)?;
                }
            }
            Ok(())
        })?;
        store.sync_tail().map_err(StorageError::Io)
    }

    /// Stage removal of the room's State-pool records into `txn`.
    ///
    /// The graph rows are tombstoned by [`Self::purge_graph`], not here.
    ///
    /// # Errors
    /// Returns an error if a delete cannot be staged.
    pub fn stage_purge(&self, txn: &DatabaseTransaction<'_>) -> Result<(), StorageError> {
        self.events.stage_purge(txn)?;
        self.kinds.stage_purge(txn)?;
        Ok(())
    }

    fn targets_of<R: GraphReader + ?Sized>(
        &self,
        db: &R,
        event_id: &str,
        family: EdgeFamily,
    ) -> Result<Option<Vec<String>>, StorageError> {
        let Some(id) = self.short_id(db, event_id)? else {
            return Ok(None);
        };
        let Some(stored) = self.family_edges(db, id, family)? else {
            return Ok(None);
        };
        let ids: Vec<u32> = stored.iter().map(|edge| edge.target).collect();
        self.events
            .resolve(db, &ids)?
            .into_iter()
            .map(|key| {
                key.ok_or_else(|| StorageError::Corrupt("edge target does not resolve".to_owned()))
                    .and_then(|bytes| utf8(bytes, "event id"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

struct PreparedEvent<'a> {
    owner: &'a [u8],
    prev: Vec<EdgeKey<'a>>,
    auth: Vec<EdgeKey<'a>>,
    relation: Vec<EdgeKey<'a>>,
}

/// Stored rows of the batch's owners, by `(family id, owner)`.
type StoredRows = HashMap<(u16, u32), Vec<Edge>>;

impl<'a> PreparedEvent<'a> {
    /// The edge keys of the family at `index` in [`FAMILIES`].
    fn family_keys(&self, index: usize) -> &[EdgeKey<'a>] {
        match index {
            0 => &self.prev,
            1 => &self.auth,
            _ => &self.relation,
        }
    }
}

const DIFFERS_FROM_RECORDED: &str = "short-id edge list differs from the recorded one";
const DIFFERS_IN_BATCH: &str = "short-id edge list differs from one already in this batch";

/// Fold a repeated owner's edges into what the batch already claimed for it.
fn claim_again(claim: &mut [ClaimedEdges], wanted: &[ClaimedEdges; 3]) -> Result<(), StorageError> {
    for (index, family) in FAMILIES.iter().enumerate() {
        if family.typed {
            // Relations are add-only: an absent relation (a redacted copy) does
            // not contradict an observed one.
            if !claim[index].is_empty()
                && !wanted[index].is_empty()
                && claim[index] != wanted[index]
            {
                return Err(StorageError::Collision(DIFFERS_IN_BATCH.to_owned()));
            }
            claim[index].extend(wanted[index].iter().cloned());
        } else if claim[index] != wanted[index] {
            return Err(StorageError::Collision(DIFFERS_IN_BATCH.to_owned()));
        }
    }
    Ok(())
}

/// Every key the batch references, owner first, and where each event starts.
fn batch_keys<'a>(prepared: &[PreparedEvent<'a>]) -> (Vec<&'a [u8]>, Vec<usize>) {
    let mut keys: Vec<&[u8]> = Vec::new();
    let mut starts = Vec::with_capacity(prepared.len());
    for event in prepared {
        starts.push(keys.len());
        keys.push(event.owner);
        for family in [&event.prev, &event.auth, &event.relation] {
            keys.extend(family.iter().map(|edge| edge.target));
        }
    }
    (keys, starts)
}

/// Merge the batch into the stored rows, returning each owner's short id and
/// the rows that changed.
fn merge_batch(
    stored: &mut StoredRows,
    prepared: &[PreparedEvent<'_>],
    starts: &[usize],
    ids: &[u32],
) -> Result<(Vec<u32>, Vec<RowRecord>), StorageError> {
    let mut owners = Vec::with_capacity(prepared.len());
    let mut changed_rows: Vec<RowRecord> = Vec::new();
    for (event, &start) in prepared.iter().zip(starts) {
        let owner = ids[start];
        owners.push(owner);
        let mut cursor = start.saturating_add(1);
        for (index, family) in FAMILIES.iter().enumerate() {
            let mut wanted: BTreeSet<Edge> = BTreeSet::new();
            for edge in event.family_keys(index) {
                wanted.insert(Edge {
                    target: ids[cursor],
                    kind: edge.kind,
                });
                cursor = cursor.saturating_add(1);
            }
            // No relation known means no relations row, so a relation can be
            // added later and a stored one is never contradicted.
            if family.id == RELATIONS.id && wanted.is_empty() {
                continue;
            }
            let list: Vec<Edge> = wanted.into_iter().collect();
            merge_row(stored, *family, owner, list, &mut changed_rows)?;
        }
    }
    Ok((owners, changed_rows))
}

/// Merge one owner's row into the stored rows, recording it in `changed` if it
/// is new or (for the add-only relations family) gained edges.
fn merge_row(
    stored: &mut StoredRows,
    family: EdgeFamily,
    owner: u32,
    list: Vec<Edge>,
    changed: &mut Vec<RowRecord>,
) -> Result<(), StorageError> {
    let key = (family.id, owner);
    if let Some(stored_edges) = stored.get(&key) {
        if family.typed {
            if !list.is_empty() && !stored_edges.is_empty() && *stored_edges != list {
                return Err(StorageError::Collision(DIFFERS_FROM_RECORDED.to_owned()));
            }
            if list.is_empty() || !stored_edges.is_empty() {
                return Ok(());
            }
        } else if *stored_edges != list {
            return Err(StorageError::Collision(DIFFERS_FROM_RECORDED.to_owned()));
        } else {
            return Ok(());
        }
    }
    stored.insert(key, list.clone());
    changed.push(RowRecord {
        family,
        owner,
        edges: list,
    });
    Ok(())
}

/// The reserved scope of the edge-log watermark row, outside any room's scope.
const WATERMARK_SCOPE: ScopeId = ScopeId([0xED; 16]);
/// Family of the watermark row: one row, owner 0, whose single edge target is
/// the last edge-log sequence number the CSR has applied.
const WATERMARK: EdgeFamily = EdgeFamily::plain(40);

/// Append the watermark row for log entry `seq`, under the store's write lock.
fn write_watermark_locked(store: &CsrStore, db: &Database, seq: u64) -> std::io::Result<()> {
    store.writer(db).and_then(|writer| {
        writer.append_rows_deferred_locked(
            WATERMARK_SCOPE,
            &[RowRecord {
                family: WATERMARK,
                owner: 0,
                // A u32 edge target holds half the sequence number; the kind
                // says which half.
                edges: vec![
                    Edge {
                        target: u32::try_from(seq >> 32).unwrap_or(u32::MAX),
                        kind: 0,
                    },
                    Edge {
                        target: u32::try_from(seq & u64::from(u32::MAX)).unwrap_or(u32::MAX),
                        kind: 1,
                    },
                ],
            }],
        )
    })
}

/// The last edge-log sequence number whose rows the CSR has (0 for none).
fn read_watermark(store: &CsrStore) -> std::io::Result<u64> {
    Ok(watermark_from(
        store.read_rows(WATERMARK_SCOPE, &[(WATERMARK, 0)])?,
    ))
}

/// [`read_watermark`] for a caller holding the store's write lock.
fn read_watermark_locked(store: &CsrStore) -> std::io::Result<u64> {
    Ok(watermark_from(
        store.read_rows_locked(WATERMARK_SCOPE, &[(WATERMARK, 0)])?,
    ))
}

/// Family of a scope's applied-purge row: owner 0, four edges holding the log
/// sequence number and the token of the last purge applied to the scope. Readers
/// of the scope's snapshot ignore it, like the owner-0 slot.
const PURGE_APPLIED: EdgeFamily = EdgeFamily::plain(41);

/// Split `values` into edges: value `n` is the pair of halves with kinds `2n`
/// (high) and `2n + 1` (low).
fn u64_edges(values: &[u64]) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(values.len().saturating_mul(2));
    for (index, value) in values.iter().enumerate() {
        let kind = u16::try_from(index.saturating_mul(2)).unwrap_or(u16::MAX);
        edges.push(Edge {
            target: u32::try_from(value >> 32).unwrap_or(u32::MAX),
            kind,
        });
        edges.push(Edge {
            target: u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX),
            kind: kind.saturating_add(1),
        });
    }
    edges
}

/// The value `index` encoded by [`u64_edges`].
fn u64_at(edges: &[Edge], index: usize) -> u64 {
    let half = |kind: usize| {
        edges
            .iter()
            .find(|edge| usize::from(edge.kind) == kind)
            .map_or(0, |edge| u64::from(edge.target))
    };
    let kind = index.saturating_mul(2);
    (half(kind) << 32) | half(kind.saturating_add(1))
}

/// The last purge each scope in `entries` has applied.
fn applied_purges(
    store: &CsrStore,
    entries: &[(u64, LoggedEvent)],
) -> Result<HashMap<[u8; 16], (u64, u64)>, StorageError> {
    let mut applied = HashMap::new();
    for (_, event) in entries {
        if applied.contains_key(&event.scope) {
            continue;
        }
        let csr = CsrAdjacency::for_collection(&event.scope).scope();
        if let Some(done) = read_purge_applied(store, csr).map_err(StorageError::Io)? {
            applied.insert(event.scope, done);
        }
    }
    Ok(applied)
}

/// Raise the log past `floor` (the highest applied purge) when it is above
/// `head`, and move the watermark with it. Entries up to `head` are pruned
/// first, so nothing unapplied is lost, and the raised head is made durable
/// before the watermark: a durable watermark above a rolled-back head would
/// send the next recovery back through the rollback repair.
fn raise_log_floor(
    db: &Database,
    store: &CsrStore,
    log: &EdgeLog,
    head: u64,
    floor: u64,
) -> Result<(), StorageError> {
    if floor <= head {
        return Ok(());
    }
    log.prune_through(db, head)?;
    log.raise_head_to(db, floor)?;
    db.coordinator().sync().map_err(StorageError::Io)?;
    store
        .with_write_lock(|| write_watermark_locked(store, db, floor))
        .map_err(StorageError::Io)?;
    store.sync_tail().map_err(StorageError::Io)
}

/// The highest log sequence number any scope records as an applied purge.
fn max_applied_purge(store: &CsrStore) -> std::io::Result<u64> {
    let mut highest = 0_u64;
    for scope in store.all_scopes()? {
        if let Some((seq, _)) = read_purge_applied(store, ScopeId(scope))? {
            highest = highest.max(seq);
        }
    }
    Ok(highest)
}

/// Record, under the store's write lock, that the purge `(seq, token)` is
/// applied to `scope`.
fn write_purge_applied_locked(
    store: &CsrStore,
    db: &Database,
    scope: ScopeId,
    seq: u64,
    token: u64,
) -> std::io::Result<()> {
    store.writer(db).and_then(|writer| {
        writer.append_rows_deferred_locked(
            scope,
            &[RowRecord {
                family: PURGE_APPLIED,
                owner: 0,
                edges: u64_edges(&[seq, token]),
            }],
        )
    })
}

/// The `(seq, token)` of the last purge applied to `scope`, if one is recorded.
fn read_purge_applied(store: &CsrStore, scope: ScopeId) -> std::io::Result<Option<(u64, u64)>> {
    Ok(store
        .read_rows(scope, &[(PURGE_APPLIED, 0)])?
        .into_iter()
        .next()
        .flatten()
        .filter(|edges| !edges.is_empty())
        .map(|edges| (u64_at(&edges, 0), u64_at(&edges, 1))))
}

fn watermark_from(rows: Vec<Option<Vec<Edge>>>) -> u64 {
    let edges = rows.into_iter().next().flatten().unwrap_or_default();
    let half = |kind: u16| {
        edges
            .iter()
            .find(|edge| edge.kind == kind)
            .map_or(0, |edge| u64::from(edge.target))
    };
    (half(0) << 32) | half(1)
}

impl CsrAdjacency {
    /// Rebuild the rows the CSR lost, from the edge log.
    ///
    /// Reads the watermark (the last log entry whose rows are in the CSR),
    /// replays every later entry into its scope, advances the watermark and
    /// makes the result durable. Replay is idempotent: recording an event whose
    /// rows survived changes nothing. Run it once after opening a writable
    /// store and before serving. Returns how many entries were replayed.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt log entry, or a failed
    /// recording or sync.
    pub fn recover_edge_log(db: &Database, log: &EdgeLog) -> Result<u32, StorageError> {
        let store = CsrStore::open(db.layout(), ShardType::State);
        let watermark = read_watermark(&store).map_err(StorageError::Io)?;
        let head = log.head(db)?;
        // A scope can record an applied purge past the recovered head: the log
        // was rolled back beyond it. Numbers up to it must not be handed out
        // again, or a later entry would look older than that purge.
        let purge_floor = max_applied_purge(&store).map_err(StorageError::Io)?;
        if watermark > head {
            // The pre-sync hook makes CSR rows durable before the WAL group
            // that covers them, so a crash can leave rows for events whose
            // short-id allocations rolled back. Those ids can be handed out
            // again, and their orphan rows would then be merged into an
            // unrelated event. Keep every row whose owner and targets are
            // still allocated, drop only the references to rolled-back ids,
            // and lower the watermark to the recovered log head.
            repair_rows_after_rollback(db, &store, head)?;
            // The repaired rows and the reset watermark are durable, so the
            // entries they cover are no longer needed.
            log.prune_through(db, head)?;
            raise_log_floor(db, &store, log, head, purge_floor)?;
            return Ok(0);
        }
        if head == watermark {
            // Nothing to replay, but the watermark and its rows may still be
            // only in the page cache; do not report recovery done before they
            // are durable.
            store.sync_tail().map_err(StorageError::Io)?;
            repair_rows_after_rollback(db, &store, watermark)?;
            raise_log_floor(db, &store, log, head, purge_floor)?;
            return Ok(0);
        }
        // Repair stale ids without advancing the watermark. If recovery crashes
        // before replay completes, the same entries must remain visible next time.
        repair_rows_after_rollback(db, &store, watermark)?;
        let entries = log.read_range(db, watermark, head)?;
        // Entries at or below the prune point are legitimately gone; any other
        // gap means the log lost an entry and the rows it covered, so advancing
        // the watermark would hide the loss.
        let floor = watermark.max(log.pruned(db)?);
        let expected = head.saturating_sub(floor);
        let present = entries.iter().filter(|(seq, _)| *seq > floor).count();
        if u64::try_from(present).ok() != Some(expected) {
            return Err(StorageError::Corrupt(format!(
                "edge log is missing entries above {floor}: expected {expected}, found {present}"
            )));
        }
        let latest_purge: HashMap<[u8; 16], u64> = entries
            .iter()
            .filter(|(_, event)| EdgeLog::is_scope_purge(event))
            .map(|(seq, event)| (event.scope, *seq))
            .collect();
        // The last purge each scope has applied. A scope applied through a purge
        // whose log marker was since lost must not replay the entries before it,
        // and a purge already applied must not be replayed over later writes.
        let mut applied = applied_purges(&store, &entries)?;
        let mut replayed = 0_u32;
        let mut pending_scope = None;
        let mut pending = Vec::new();
        for (seq, event) in &entries {
            if EdgeLog::is_scope_purge(event) {
                if let Some(scope) = pending_scope.take() {
                    replayed = replayed.saturating_add(replay_edge_log_run(db, scope, &pending)?);
                    pending.clear();
                }
                if latest_purge.get(&event.scope) != Some(seq) {
                    // A later tombstone subsumes this one. Reapplying an older
                    // marker could erase unlogged rows written after the latest
                    // purge, so only the latest boundary may clear the scope.
                    continue;
                }
                // Sequence numbers are reused after a crash rolls the log back,
                // so identity is the (sequence, token) pair. A boundary written
                // before tokens counts as token 0.
                let token = EdgeLog::purge_token(event).unwrap_or(0);
                let already_applied =
                    applied
                        .get(&event.scope)
                        .is_some_and(|(done_seq, done_token)| {
                            *done_seq > *seq || (*done_seq == *seq && *done_token == token)
                        });
                if already_applied {
                    continue;
                }
                let csr = CsrAdjacency::for_collection(&event.scope).scope();
                store
                    .with_write_lock(|| -> std::io::Result<()> {
                        store.writer(db)?.purge_scope_locked(csr)?;
                        write_purge_applied_locked(&store, db, csr, *seq, token)
                    })
                    .map_err(StorageError::Io)?;
                applied.insert(event.scope, (*seq, token));
                continue;
            }
            let skip_through = latest_purge
                .get(&event.scope)
                .copied()
                .max(applied.get(&event.scope).map(|(done_seq, _)| *done_seq));
            if skip_through.is_some_and(|through| *seq <= through) {
                continue;
            }
            if let Some(scope) = pending_scope {
                if scope != event.scope {
                    replayed = replayed.saturating_add(replay_edge_log_run(db, scope, &pending)?);
                    pending.clear();
                }
            }
            pending_scope = Some(event.scope);
            pending.push(event);
        }
        if let Some(scope) = pending_scope {
            replayed = replayed.saturating_add(replay_edge_log_run(db, scope, &pending)?);
        }
        store
            .with_write_lock(|| write_watermark_locked(&store, db, head))
            .map_err(StorageError::Io)?;
        store.sync_tail().map_err(StorageError::Io)?;
        // The rows and the watermark are durable, so the log no longer needs
        // these entries.
        log.prune_through(db, head)?;
        raise_log_floor(db, &store, log, head, purge_floor)?;
        Ok(replayed)
    }

    /// Make the applied rows durable and drop the log entries they cover.
    ///
    /// Reads the watermark first, then syncs the tail, so everything up to that
    /// watermark is durable when the log is pruned through it. Cheap when there
    /// is nothing new. This is the maintenance step that keeps the log short and
    /// recovery fast; it has no ordering relationship with WAL commits.
    ///
    /// # Errors
    /// Returns an error if the sync or the prune fails.
    pub fn checkpoint_edge_log(db: &Database, log: &EdgeLog) -> Result<(), StorageError> {
        let store = CsrStore::open(db.layout(), ShardType::State);
        let watermark = read_watermark(&store).map_err(StorageError::Io)?;
        // A tail that outlived the WAL can name a watermark past the log head;
        // pruning through it would hide every future entry up to that number.
        let watermark = watermark.min(log.head(db)?);
        store.sync_tail().map_err(StorageError::Io)?;
        log.prune_through(db, watermark)
    }
}

/// The CSR scope holding one event index's rows.
fn csr_scope_for(events: &[u8; 16]) -> [u8; 16] {
    let mut name = b"matrix-csr:".to_vec();
    name.extend_from_slice(events);
    derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &name)
}

/// Drop CSR rows that name short ids the recovered short-id counters no longer
/// allocate, then persist the supplied watermark.
///
/// The pre-sync hook syncs adjacency tails before the WAL group covering them,
/// so a crash can leave rows for events whose allocations rolled back. The
/// allocations and their rows are all derived data: a row belongs to a
/// recovered scope only while its owner and every target are below the scope's
/// recovered next id. Anything above it names an event that no longer exists,
/// and reusing that id for a new event must not inherit the orphan row.
fn repair_rows_after_rollback(
    db: &Database,
    store: &CsrStore,
    watermark: u64,
) -> Result<bool, StorageError> {
    let scopes = store.all_scopes().map_err(StorageError::Io)?;
    // A CSR scope is derived from its event collection, so enumerate the State
    // pool's collections to recover each scope's committed short-id counter.
    // A CSR scope with no matching collection lost every event it indexed.
    let mut counters: HashMap<[u8; 16], u32> = HashMap::new();
    for collection in db.pool(ShardType::State).collection_ids() {
        // `collection` is the event index scope itself, so hash it directly.
        // `for_collection` would treat it as the room collection and derive a
        // different scope.
        let csr = csr_scope_for(&collection);
        if scopes.contains(&csr) {
            let next = ShortIdIndex::new(ShardType::State, collection).counter(db)?;
            counters.insert(csr, next);
        }
    }
    // Relation kinds are allocated in a separate dictionary derived from the
    // room collection, not from the event index. EventDag's collections are
    // those room collections, so use them to find each scope's kind counter.
    // Scopes with no match (legacy or synthetic) keep their relation edges.
    let mut kind_counters: HashMap<[u8; 16], u32> = HashMap::new();
    for room_collection in db.pool(ShardType::EventDag).collection_ids() {
        let adjacency = CsrAdjacency::for_collection(&room_collection);
        let csr = adjacency.csr_collection();
        if scopes.contains(&csr) {
            kind_counters.insert(csr, adjacency.kinds.counter(db)?);
        }
    }
    let mut repaired: Vec<(ScopeId, Vec<RowRecord>)> = Vec::new();
    for scope in scopes {
        let scope = ScopeId(scope);
        if scope == WATERMARK_SCOPE {
            continue;
        }
        // Without the matching recovered event index, there is no evidence
        // that these IDs rolled back. Keep the scope intact rather than
        // treating an unknown counter as an empty index and deleting its rows.
        let Some(next) = counters.get(&scope.0).copied() else {
            continue;
        };
        let kind_limit = kind_counters.get(&scope.0).copied();
        let data = store.read_scope(scope).map_err(StorageError::Io)?;
        let mut changed = false;
        let mut rows = Vec::with_capacity(data.rows.len());
        for row in data.rows {
            if row.family.id == PURGE_APPLIED.id {
                // Its edges are log numbers, not short ids, so the filters
                // below do not apply. Keep the row, so a rewrite does not lose
                // the applied-purge state. A sequence beyond the recovered
                // head may be reused, but the token distinguishes a different
                // purge that takes the same number.
                rows.push(RowRecord {
                    family: row.family,
                    owner: row.owner,
                    edges: row.edges,
                });
                continue;
            }
            if row.owner >= next {
                changed = true;
                continue;
            }
            let before = row.edges.len();
            let edges: Vec<Edge> = row
                .edges
                .into_iter()
                .filter(|edge| {
                    // A scope whose kind dictionary cannot be found is not
                    // proof that its kinds rolled back, so only a known counter
                    // invalidates a typed edge.
                    let kind_valid = !row.family.typed
                        || kind_limit.is_none_or(|limit| u32::from(edge.kind) < limit);
                    edge.target < next && kind_valid
                })
                .collect();
            if edges.len() != before {
                changed = true;
                // A forward row is written with its owner and targets
                // together, so a lost target means the claim is gone: an empty
                // `prev` or `auth` row would read as a known leaf. Reverse rows
                // just keep the children that survive.
                if row.family == PREV || row.family == AUTH || row.family == RELATIONS {
                    continue;
                }
            }
            rows.push(RowRecord {
                family: row.family,
                owner: row.owner,
                edges,
            });
        }
        if changed {
            repaired.push((scope, rows));
        }
    }
    let changed = store.with_write_lock(|| -> Result<bool, StorageError> {
        let watermark_changed =
            read_watermark_locked(store).map_err(StorageError::Io)? != watermark;
        if !repaired.is_empty() {
            store
                .replace_scopes_locked(&repaired)
                .map_err(StorageError::Io)?;
        }
        if watermark_changed {
            write_watermark_locked(store, db, watermark).map_err(StorageError::Io)?;
        }
        Ok(watermark_changed || !repaired.is_empty())
    })?;
    if changed {
        store.sync_tail().map_err(StorageError::Io)?;
    }
    Ok(changed)
}

fn replay_edge_log_run(
    db: &Database,
    scope: [u8; 16],
    run: &[&LoggedEvent],
) -> Result<u32, StorageError> {
    if run.is_empty() {
        return Ok(0);
    }
    let prevs: Vec<Vec<&str>> = run
        .iter()
        .map(|event| event.prev.iter().map(String::as_str).collect())
        .collect();
    let auths: Vec<Vec<&str>> = run
        .iter()
        .map(|event| event.auth.iter().map(String::as_str).collect())
        .collect();
    let records: Vec<EventRecord<'_>> = run
        .iter()
        .zip(&prevs)
        .zip(&auths)
        .map(|((event, prev), auth)| EventRecord {
            event_id: &event.event_id,
            prev,
            auth,
            relation: event
                .relation
                .as_ref()
                .map(|(target, rel_type)| RelationRef { target, rel_type }),
        })
        .collect();
    CsrAdjacency::for_collection(&scope).record_events_inner(
        db,
        &records,
        AppendDurability::Deferred,
        None,
    )?;
    Ok(u32::try_from(run.len()).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod scope_compatibility_tests {
    use super::*;

    #[test]
    fn room_id_constructor_preserves_legacy_scopes() {
        let room = "!legacy:example.org";
        let adjacency = CsrAdjacency::new(room);
        assert_eq!(
            adjacency.events.scope().1,
            legacy_collection("matrix-adjacency:", room)
        );
        assert_eq!(
            adjacency.kinds.scope().1,
            legacy_collection("matrix-relation-kinds:", room)
        );
    }
}
