//! Matrix event adjacency over the generic short-id primitives.
//!
//! One room maps to two [`ShortIdIndex`] scopes:
//!
//! - **events**: event id -> room-local `u32` short id, with three immutable
//!   adjacency families keyed by the event: `prev` (plain), `auth` (plain) and
//!   `relations` (typed, at most one edge per event, as in Synapse's unique
//!   `event_relations(event_id)` row);
//! - **relation kinds**: relation type string (`m.annotation`, ...) -> `u16`
//!   kind id.
//!
//! Kind ids are persisted in the dictionary and never regenerated: once a type
//! has an id it keeps it for the life of the room's data, so stored relation
//! edges stay meaningful. The four relation types defined by the spec get fixed
//! ids ([`KNOWN_RELATION_TYPES`]); any other type string is preserved (never
//! dropped) and gets the next free id on first sight. Ids above `u16::MAX` are a
//! hard error.
//!
//! What is *not* stored here: whether an event is still visible. A relation
//! disappears when its source event is redacted, rejected or soft-failed, but
//! those are facts about the source event held elsewhere (and rejection is
//! reversible). Reads therefore take an [`EventVisibility`] filter and apply it
//! to the relation at query time; `prev` and `auth` are signed event-core fields
//! that redaction never changes, so they are not filtered. The transitive
//! auth-closure layer must read only [`MatrixAdjacency::auth_of`], never
//! relations.
//!
//! Reverse lookups (who relates to this event) are not provided: fan-in is
//! unbounded and belongs in an ordered index, not a per-event edge record.
//!
//! JSON handling is the caller's: this module takes already-extracted event
//! ids and relation type strings. Built on the transaction layer in
//! [`crate::database`].

use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::short_id::{EdgeFamily, EdgeKey, FamilyEdges, ShortIdIndex};
use crate::storage::StorageError;
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Family of an event's `prev_events`.
pub const PREV: EdgeFamily = EdgeFamily::plain(1);
/// Family of an event's `auth_events`.
pub const AUTH: EdgeFamily = EdgeFamily::plain(2);
/// Family of an event's single relation (`m.relates_to`), typed by relation kind.
pub const RELATIONS: EdgeFamily = EdgeFamily::typed(3);

/// Relation types defined by the Matrix spec, with fixed kind ids (1-based, in
/// this order). They are seeded into every room's dictionary before any other
/// type, so their ids are identical across rooms and stores.
pub const KNOWN_RELATION_TYPES: [&str; 4] =
    ["m.annotation", "m.reference", "m.replace", "m.thread"];

/// Decides whether an event's relation is currently visible.
///
/// Implemented by the caller against its authoritative redaction and
/// disposition state (redacted, rejected, soft-failed). Must be cheap: it is
/// consulted once per relation read.
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

/// Result of [`MatrixAdjacency::verify`].
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
pub struct MatrixAdjacency {
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

impl MatrixAdjacency {
    /// Address `room_id`'s adjacency inside `pool`.
    #[must_use]
    pub fn new(pool: ShardType, room_id: &str) -> Self {
        Self {
            events: ShortIdIndex::new(pool, collection("matrix-adjacency:", room_id)),
            kinds: ShortIdIndex::new(pool, collection("matrix-relation-kinds:", room_id))
                .with_max_id(u32::from(u16::MAX)),
        }
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
    /// Re-recording identical data is a no-op; different `prev`, `auth` or
    /// relation for an already-recorded event is a collision error (an event's
    /// core fields and relation never change).
    ///
    /// The relation kind is allocated first in its own transaction. That step
    /// is idempotent and permanent, so a failure after it leaves only an unused
    /// dictionary entry.
    ///
    /// # Errors
    /// Storage errors; a collision as described; or an error if the relation
    /// kind dictionary is out of `u16` ids.
    pub fn record_event(
        &self,
        db: &Database,
        event_id: &str,
        prev: &[&str],
        auth: &[&str],
        relation: Option<RelationRef<'_>>,
    ) -> Result<u32, StorageError> {
        let kind = match relation {
            Some(relation) => Some(self.kind_id(db, relation.rel_type)?),
            None => None,
        };
        let prev_edges: Vec<EdgeKey<'_>> = prev
            .iter()
            .map(|id| EdgeKey::plain(id.as_bytes()))
            .collect();
        let auth_edges: Vec<EdgeKey<'_>> = auth
            .iter()
            .map(|id| EdgeKey::plain(id.as_bytes()))
            .collect();
        let relation_edges: Vec<EdgeKey<'_>> = match (relation, kind) {
            (Some(relation), Some(kind)) => vec![EdgeKey {
                target: relation.target.as_bytes(),
                kind,
            }],
            _ => Vec::new(),
        };
        let recorded = self.events.record_event(
            db,
            event_id.as_bytes(),
            &[
                FamilyEdges {
                    family: PREV,
                    edges: &prev_edges,
                },
                FamilyEdges {
                    family: AUTH,
                    edges: &auth_edges,
                },
                FamilyEdges {
                    family: RELATIONS,
                    edges: &relation_edges,
                },
            ],
        )?;
        Ok(recorded.id)
    }

    /// The room-local short id of `event_id`, if it has been recorded or
    /// referenced.
    ///
    /// # Errors
    /// Returns an error on a read failure.
    pub fn short_id(&self, db: &Database, event_id: &str) -> Result<Option<u32>, StorageError> {
        // `get_or_create` would allocate; look the key up without writing.
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
        let Some(stored) = self.events.edges(db, id, RELATIONS)? else {
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
    /// The spec-defined types are always part of the same allocation batch,
    /// ahead of `rel_type`, so on a fresh dictionary they take ids 1..=4 in one
    /// transaction and no other type can be allocated before them, however calls
    /// race.
    ///
    /// # Errors
    /// An error (publishing nothing) once the dictionary would exceed `u16::MAX`
    /// ids; or a storage error.
    pub fn kind_id(&self, db: &Database, rel_type: &str) -> Result<u16, StorageError> {
        let mut keys: Vec<&[u8]> = KNOWN_RELATION_TYPES.iter().map(|t| t.as_bytes()).collect();
        keys.push(rel_type.as_bytes());
        let ids = self.kinds.get_or_create(db, &keys)?;
        // The index refuses ids above `u16::MAX` inside the allocation
        // transaction, so this conversion cannot fail and nothing is published
        // past the limit.
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
                .verify(db, &[PREV, AUTH, RELATIONS])?
                .problems
                .into_iter()
                .map(|problem| format!("events: {problem}")),
        );
        report.problems.extend(
            self.kinds
                .verify(db, &[])?
                .problems
                .into_iter()
                .map(|problem| format!("relation kinds: {problem}")),
        );
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
        txn.commit()
    }

    /// Stage the room's removal into `txn`.
    ///
    /// # Errors
    /// Returns an error if a delete cannot be staged.
    pub fn stage_purge(&self, txn: &DatabaseTransaction<'_>) -> Result<(), StorageError> {
        self.events.stage_purge(txn)?;
        self.kinds.stage_purge(txn)
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
        let Some(stored) = self.events.edges(db, id, family)? else {
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
