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
//!   `relations` families as compressed sparse rows keyed by short id. The
//!   reverse (child-of) view is the transpose of each forward row set.
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
//! `<db_root>/graph/state/`, keyed by `(collection id, family, owner)`.
//! Referenced but not-yet-observed parents are ordinary nodes: every target is
//! allocated a short id, so the row set never drops an edge to a missing body.

use std::collections::{BTreeSet, HashMap};

/// The distinct `(target event id, kind)` edges one family claims for an owner.
type ClaimedEdges = BTreeSet<(Vec<u8>, u16)>;

use crate::csr::store::{CsrStore, RowRecord, ScopeId};
use crate::csr::{Edge, EdgeFamily, EdgeKey, ReverseCsr};
use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::short_id::ShortIdIndex;
use crate::storage::StorageError;
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Family of an event's `prev_events`.
pub const PREV: EdgeFamily = EdgeFamily::plain(1);
/// Family of an event's `auth_events`.
pub const AUTH: EdgeFamily = EdgeFamily::plain(2);
/// Family of an event's single relation (`m.relates_to`), typed by relation kind.
pub const RELATIONS: EdgeFamily = EdgeFamily::typed(3);

const FAMILIES: [EdgeFamily; 3] = [PREV, AUTH, RELATIONS];

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
}

fn collection(prefix: &str, room_id: &str) -> [u8; 16] {
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
    /// store, `<db_root>/graph/state/`, outside the packs. Event bodies stay in
    /// `EventDag`.
    #[must_use]
    pub fn new(room_id: &str) -> Self {
        let pool = ShardType::State;
        Self {
            events: ShortIdIndex::new(pool, collection("matrix-adjacency:", room_id)),
            kinds: ShortIdIndex::new(pool, collection("matrix-relation-kinds:", room_id))
                .with_max_id(u32::from(u16::MAX)),
        }
    }

    fn csr_collection(&self) -> [u8; 16] {
        let (_, events) = self.events.scope();
        let mut name = b"matrix-csr:".to_vec();
        name.extend_from_slice(&events);
        derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &name)
    }

    fn scope(&self) -> ScopeId {
        ScopeId(self.csr_collection())
    }

    fn store(&self, db: &Database) -> CsrStore {
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
    /// # Errors
    /// As [`Self::record_event`], applied to the whole batch.
    pub fn record_events(
        &self,
        db: &Database,
        events: &[EventRecord<'_>],
    ) -> Result<Vec<u32>, StorageError> {
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let store = self.store(db);
        // The read, merge and append must be one critical section: two
        // recordings that both read the same stored row would otherwise each
        // validate against the old state, and the later append would silently
        // replace the earlier one.
        store.with_write_lock(|| {
            let prepared = self.prepare_events(db, events)?;

            // Validate everything that can collide before allocating any id,
            // so a rejected batch records nothing. Only the batch's own owners
            // are read; the rest of the collection is not touched.
            let (owner_ids, mut stored) = self.load_stored(db, &prepared)?;
            self.check_conflicts(db, &prepared, &owner_ids, &stored)?;

            // Allocate every id in one call, exactly as the short-id layer did.
            let (keys, starts) = batch_keys(&prepared);
            let ids = self.events.get_or_create(db, &keys)?;
            let (owners, changed_rows) = merge_batch(&mut stored, &prepared, &starts, &ids)?;
            if !changed_rows.is_empty() {
                store
                    .writer(db)
                    .and_then(|writer| writer.append_rows_locked(self.scope(), &changed_rows))
                    .map_err(StorageError::Io)?;
            }
            Ok(owners)
        })
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
    pub fn short_id(&self, db: &Database, event_id: &str) -> Result<Option<u32>, StorageError> {
        self.events.lookup(db, event_id.as_bytes())
    }

    /// The event's `prev_events`, or `None` if the event was never recorded.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn prev_of(
        &self,
        db: &Database,
        event_id: &str,
    ) -> Result<Option<Vec<String>>, StorageError> {
        self.targets_of(db, event_id, PREV)
    }

    /// The event's `auth_events`, or `None` if the event was never recorded.
    /// This is the only input an auth closure may use.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn auth_of(
        &self,
        db: &Database,
        event_id: &str,
    ) -> Result<Option<Vec<String>>, StorageError> {
        self.targets_of(db, event_id, AUTH)
    }

    /// The raw edge list of `short_id` in `family`, as stored in the CSR.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt snapshot.
    pub fn family_edges(
        &self,
        db: &Database,
        short_id: u32,
        family: EdgeFamily,
    ) -> Result<Option<Vec<Edge>>, StorageError> {
        let mut rows = self
            .store(db)
            .read_rows(self.scope(), &[(family, short_id)])
            .map_err(StorageError::Io)?;
        Ok(rows.pop().flatten())
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
    #[cfg(feature = "bitmaps")]
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
        let store = self.store(db);
        let mut targets = BTreeSet::new();
        for family in [PREV, AUTH] {
            for (_, edges) in store
                .read_family(self.scope(), family)
                .map_err(StorageError::Io)?
            {
                targets.extend(edges.iter().map(|edge| edge.target));
            }
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
    /// for `RELATIONS` the caller must check each child's visibility) and is intended for a batch
    /// of reverse lookups; callers should reuse the returned view rather than
    /// rebuilding it per parent. Reverse rows are derived from the canonical
    /// forward rows, so purges and relation unions are reflected automatically.
    ///
    /// # Errors
    /// Returns an error on a read failure, corrupt segment, or an index too
    /// large for `u32` offsets.
    pub fn reverse_family(
        &self,
        db: &Database,
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
    /// # Errors
    /// Returns an error if the tombstone cannot be appended.
    pub fn purge_graph(&self, db: &Database) -> Result<(), StorageError> {
        let store = self.store(db);
        store
            .writer(db)
            .and_then(|writer| writer.purge_scope(self.scope()))
            .map_err(StorageError::Io)
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

    fn targets_of(
        &self,
        db: &Database,
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
