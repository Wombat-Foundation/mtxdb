//! Room-scoped dense `u32` short ids and compact adjacency families.
//!
//! This is the generic substrate for graph workloads that need compact integer
//! ids (auth-chain closure bitmaps, relation walks). It knows nothing about
//! Matrix: a *scope* (one collection, e.g. a room) maps opaque key bytes to
//! sequentially allocated `u32` ids, and stores per-id **adjacency** as named
//! *families*. The schema-aware layer decides what keys and families mean.
//!
//! Records, all in one collection so a scope purges with one
//! `delete_collection`:
//!
//! - **counter** (`SIDC`): the next id to allocate;
//! - **forward** (`SIDF`): `hash(key) -> id + key bytes`. The node id is a
//!   truncated hash, so the stored key is compared on every hit and a mismatch
//!   is reported as a collision, never silently merged;
//! - **reverse** (`SIDR`): `id -> key bytes`;
//! - **edges** (`SIDE`): `(family, id) -> sorted [Edge]`, where an [`Edge`] is a
//!   `u32` target plus, for typed families, a `u16` kind. A record is never
//!   empty (it has a header), so a known owner with zero edges is distinguishable
//!   from an absent record.
//!
//! An [`EdgeFamily`] fixes the record's merge policy and whether edges are
//! typed. [`EdgeMerge::Immutable`] families (for example a graph's parent or dependency edges) treat a
//! different list for an existing owner as a collision. [`EdgeMerge::Union`]
//! families (for example typed cross-references that arrive over time) merge repeated writes by set union under a CAS
//! on the edge record. The merge policy and typing are stored in the record and
//! checked on every access, so a caller cannot reinterpret a family.
//!
//! Allocation and every family's edge write commit in one
//! [`DatabaseTransaction`] guarded by the engine's per-record compare-and-swap
//! (the counter, every forward record that was absent, and every edge record
//! that was read). A lost race surfaces as `StorageError::StaleRead` and the
//! operation re-reads and retries, finding the winner's state. There is no
//! partial state to repair: counter, forward, reverse and edge records publish
//! together or not at all.
//!
//! Ids start at 1 and never wrap; exhausting `u32` is a checked error. The
//! CAS is atomic within the single writer process; it is unavailable without the
//! `multi-reader` feature, like the transaction layer it uses.

use std::collections::{BTreeSet, HashMap};

use bytes::Bytes;

use crate::database::{DatabaseTransaction, SharedDatabase};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageError};

/// Wire version shared by every short-id record.
pub const SHORT_ID_FORMAT_VERSION: u8 = 2;
/// Largest id that may be allocated.
pub const SHORT_ID_MAX: u32 = u32::MAX - 1;

const COUNTER_MAGIC: [u8; 4] = *b"SIDC";
const FORWARD_MAGIC: [u8; 4] = *b"SIDF";
const REVERSE_MAGIC: [u8; 4] = *b"SIDR";
const EDGES_MAGIC: [u8; 4] = *b"SIDE";

const COUNTER_ID: NodeId = *b"MTXD-SID-CNTR-v1";
const REVERSE_PREFIX: [u8; 8] = *b"MTXSIDR\0";
const EDGES_PREFIX: [u8; 8] = *b"MTXSIDE\0";

const FLAG_TYPED: u8 = 0b01;
const FLAG_UNION: u8 = 0b10;

/// Maximum optimistic-retry rounds before a stale read is returned.
const MAX_ATTEMPTS: usize = 64;

/// How repeated writes of one owner's edge set within a family combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMerge {
    /// The first set wins; a different set later is a collision error.
    Immutable,
    /// Repeated writes merge by set union, under a CAS on the edge record.
    Union,
}

/// A named adjacency family within a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeFamily {
    /// Family number; part of the edge record's address.
    pub id: u16,
    /// Merge policy for repeated writes.
    pub merge: EdgeMerge,
    /// Whether each edge carries a `u16` kind (otherwise the kind is always 0).
    pub typed: bool,
}

impl EdgeFamily {
    /// An untyped family whose edge list never changes once written.
    #[must_use]
    pub const fn immutable(id: u16) -> Self {
        Self {
            id,
            merge: EdgeMerge::Immutable,
            typed: false,
        }
    }

    /// A typed family whose edge set grows by union.
    #[must_use]
    pub const fn typed_union(id: u16) -> Self {
        Self {
            id,
            merge: EdgeMerge::Union,
            typed: true,
        }
    }

    const fn flags(self) -> u8 {
        let typed = if self.typed { FLAG_TYPED } else { 0 };
        let union = match self.merge {
            EdgeMerge::Union => FLAG_UNION,
            EdgeMerge::Immutable => 0,
        };
        typed | union
    }

    const fn entry_len(self) -> usize {
        if self.typed {
            6
        } else {
            4
        }
    }
}

/// A resolved edge: target short id plus kind (0 for untyped families).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    /// Target short id.
    pub target: u32,
    /// Kind id (0 when the family is untyped).
    pub kind: u16,
}

/// An edge to a not-yet-resolved key, as supplied by the caller.
#[derive(Debug, Clone, Copy)]
pub struct EdgeKey<'a> {
    /// Target key bytes; allocated a short id if new.
    pub target: &'a [u8],
    /// Kind id (must be 0 for untyped families).
    pub kind: u16,
}

impl<'a> EdgeKey<'a> {
    /// An untyped edge to `target`.
    #[must_use]
    pub const fn plain(target: &'a [u8]) -> Self {
        Self { target, kind: 0 }
    }
}

/// The edges to record for one family of the owner.
#[derive(Debug, Clone, Copy)]
pub struct FamilyEdges<'a> {
    /// The family.
    pub family: EdgeFamily,
    /// Its edges; sorted and de-duplicated on write.
    pub edges: &'a [EdgeKey<'a>],
}

/// Outcome of [`ShortIdIndex::record_event`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvent {
    /// The owner's short id.
    pub id: u32,
    /// The stored (merged) edge list of each requested family, in request order.
    pub edges: Vec<Vec<Edge>>,
}

fn forward_id(key: &[u8]) -> NodeId {
    let digest = DigestAlgorithm::Blake3.digest(key);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn indexed_id(prefix: [u8; 8], short_id: u32) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&prefix);
    id[8..12].copy_from_slice(&short_id.to_be_bytes());
    id
}

fn edges_record_id(short_id: u32, family: u16) -> NodeId {
    let mut id = indexed_id(EDGES_PREFIX, short_id);
    id[12..14].copy_from_slice(&family.to_be_bytes());
    id
}

fn header(magic: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&magic);
    out.push(SHORT_ID_FORMAT_VERSION);
    out
}

fn check_header<'a>(bytes: &'a [u8], magic: [u8; 4], what: &str) -> Result<&'a [u8], StorageError> {
    if bytes.len() < 5 || bytes[..4] != magic || bytes[4] != SHORT_ID_FORMAT_VERSION {
        return Err(StorageError::Corrupt(format!("short-id {what} header")));
    }
    Ok(&bytes[5..])
}

fn read_u32(bytes: &[u8], what: &str) -> Result<u32, StorageError> {
    <[u8; 4]>::try_from(bytes)
        .map(u32::from_be_bytes)
        .map_err(|_| StorageError::Corrupt(format!("short-id {what} length")))
}

fn encode_counter(next: u32) -> Vec<u8> {
    let mut out = header(COUNTER_MAGIC);
    out.extend_from_slice(&next.to_be_bytes());
    out
}

fn decode_counter(bytes: &[u8]) -> Result<u32, StorageError> {
    read_u32(check_header(bytes, COUNTER_MAGIC, "counter")?, "counter")
}

fn encode_forward(short_id: u32, key: &[u8]) -> Vec<u8> {
    let mut out = header(FORWARD_MAGIC);
    out.extend_from_slice(&short_id.to_be_bytes());
    out.extend_from_slice(key);
    out
}

/// Decode a forward record into `(short_id, key)`.
fn decode_forward(bytes: &[u8]) -> Result<(u32, &[u8]), StorageError> {
    let body = check_header(bytes, FORWARD_MAGIC, "forward")?;
    if body.len() < 4 {
        return Err(StorageError::Corrupt("short-id forward length".to_owned()));
    }
    Ok((read_u32(&body[..4], "forward")?, &body[4..]))
}

fn encode_reverse(key: &[u8]) -> Vec<u8> {
    let mut out = header(REVERSE_MAGIC);
    out.extend_from_slice(key);
    out
}

fn encode_edges(family: EdgeFamily, edges: &[Edge]) -> Result<Vec<u8>, StorageError> {
    let count = u32::try_from(edges.len())
        .map_err(|_| StorageError::Internal("short-id edge list too long".to_owned()))?;
    let mut out = header(EDGES_MAGIC);
    out.push(family.flags());
    out.extend_from_slice(&count.to_be_bytes());
    for edge in edges {
        out.extend_from_slice(&edge.target.to_be_bytes());
        if family.typed {
            out.extend_from_slice(&edge.kind.to_be_bytes());
        }
    }
    Ok(out)
}

fn decode_edges(family: EdgeFamily, bytes: &[u8]) -> Result<Vec<Edge>, StorageError> {
    let body = check_header(bytes, EDGES_MAGIC, "edges")?;
    if body.len() < 5 {
        return Err(StorageError::Corrupt("short-id edges length".to_owned()));
    }
    if body[0] != family.flags() {
        return Err(StorageError::Collision(format!(
            "short-id family {} was stored with different merge/typing flags",
            family.id
        )));
    }
    let count = read_u32(&body[1..5], "edges")? as usize;
    let list = &body[5..];
    if count.checked_mul(family.entry_len()) != Some(list.len()) {
        return Err(StorageError::Corrupt("short-id edges length".to_owned()));
    }
    list.chunks_exact(family.entry_len())
        .map(|chunk| {
            Ok(Edge {
                target: read_u32(&chunk[..4], "edges")?,
                kind: if family.typed {
                    u16::from_be_bytes([chunk[4], chunk[5]])
                } else {
                    0
                },
            })
        })
        .collect()
}

/// Result of [`ShortIdIndex::verify`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortIdVerifyReport {
    /// Ids below the counter that have a reverse record.
    pub ids_checked: u32,
    /// Edge lists decoded.
    pub edge_lists_checked: u32,
    /// Human-readable invariant violations; empty means consistent.
    pub problems: Vec<String>,
}

impl ShortIdVerifyReport {
    /// Whether every invariant held.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.problems.is_empty()
    }
}

/// One short-id scope (a collection inside one pool).
#[derive(Debug, Clone, Copy)]
pub struct ShortIdIndex {
    pool: ShardType,
    collection_id: [u8; 16],
}

impl ShortIdIndex {
    /// Address the scope stored in `collection_id` of `pool`.
    #[must_use]
    pub fn new(pool: ShardType, collection_id: [u8; 16]) -> Self {
        Self {
            pool,
            collection_id,
        }
    }

    /// Look up or allocate the short id of every key, in order.
    ///
    /// # Errors
    /// Returns an error on a hash collision between different keys, on
    /// corruption, when the `u32` id space is exhausted, or when the retry
    /// budget is spent on a contended counter.
    pub fn get_or_create(
        &self,
        db: &SharedDatabase,
        keys: &[&[u8]],
    ) -> Result<Vec<u32>, StorageError> {
        self.write_with_retry(db, keys, None).map(|(ids, _)| ids)
    }

    /// Allocate ids for `owner` and every edge target, then store each family's
    /// edge list for `owner`, all in one transaction.
    ///
    /// Edges are sorted and de-duplicated. Per family: an
    /// [`EdgeMerge::Immutable`] list that matches the stored one is a no-op and
    /// a different one is a collision error; an [`EdgeMerge::Union`] list merges
    /// with the stored one. Returns the owner's id and each family's stored list.
    ///
    /// # Errors
    /// As [`Self::get_or_create`], plus a collision error for an immutable
    /// family whose list differs, a family reused with different flags, or a
    /// non-zero kind on an untyped family.
    pub fn record_event(
        &self,
        db: &SharedDatabase,
        owner: &[u8],
        families: &[FamilyEdges<'_>],
    ) -> Result<RecordedEvent, StorageError> {
        for family in families {
            if !family.family.typed && family.edges.iter().any(|edge| edge.kind != 0) {
                return Err(StorageError::Internal(
                    "untyped edge family given a non-zero kind".to_owned(),
                ));
            }
        }
        let mut keys: Vec<&[u8]> = vec![owner];
        for family in families {
            keys.extend(family.edges.iter().map(|edge| edge.target));
        }
        self.write_with_retry(db, &keys, Some(families))
            .map(|(_, recorded)| recorded)
    }

    /// Convenience for one untyped family: record `owner`'s edges to `targets`
    /// in `family`. Returns `(owner id, sorted target ids)`.
    ///
    /// # Errors
    /// As [`Self::record_event`].
    pub fn record_edges(
        &self,
        db: &SharedDatabase,
        owner: &[u8],
        family: EdgeFamily,
        targets: &[&[u8]],
    ) -> Result<(u32, Vec<u32>), StorageError> {
        let edges: Vec<EdgeKey<'_>> = targets.iter().map(|t| EdgeKey::plain(t)).collect();
        let recorded = self.record_event(
            db,
            owner,
            &[FamilyEdges {
                family,
                edges: &edges,
            }],
        )?;
        let targets = recorded.edges[0].iter().map(|edge| edge.target).collect();
        Ok((recorded.id, targets))
    }

    /// The edge list of `short_id` in `family`, or `None` if none was recorded.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt record, or a family stored
    /// with different flags.
    pub fn edges(
        &self,
        db: &SharedDatabase,
        short_id: u32,
        family: EdgeFamily,
    ) -> Result<Option<Vec<Edge>>, StorageError> {
        let txn = db.begin_transaction();
        let (records, _) = txn.get_with_record_versions(
            self.pool,
            &self.collection_id,
            &[edges_record_id(short_id, family.id)],
        )?;
        records
            .into_iter()
            .next()
            .flatten()
            .map(|record| decode_edges(family, &record.bytes))
            .transpose()
    }

    /// Resolve short ids back to their key bytes; `None` for an unknown id.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn resolve(
        &self,
        db: &SharedDatabase,
        short_ids: &[u32],
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let ids: Vec<NodeId> = short_ids
            .iter()
            .map(|id| indexed_id(REVERSE_PREFIX, *id))
            .collect();
        let txn = db.begin_transaction();
        let (records, _) = txn.get_with_record_versions(self.pool, &self.collection_id, &ids)?;
        records
            .into_iter()
            .map(|record| {
                record
                    .map(|data| {
                        check_header(&data.bytes, REVERSE_MAGIC, "reverse").map(<[u8]>::to_vec)
                    })
                    .transpose()
            })
            .collect()
    }

    /// Overwrite the counter so tests can exercise the allocation ceiling.
    #[cfg(test)]
    pub(crate) fn set_counter_for_test(
        &self,
        db: &SharedDatabase,
        next: u32,
    ) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        txn.put(
            self.pool,
            self.collection_id,
            COUNTER_ID,
            &NodeData::new(Bytes::from(encode_counter(next))),
        )?;
        txn.commit()
    }

    /// Remove the whole scope (counter, mappings and edges) in one commit.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged or committed.
    pub fn purge(&self, db: &SharedDatabase) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        self.stage_purge(&txn)?;
        txn.commit()
    }

    /// Stage the scope's removal into `txn`, so it commits atomically with
    /// other deletes (for example the scope's closure generations).
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged.
    pub fn stage_purge(&self, txn: &DatabaseTransaction<'_>) -> Result<(), StorageError> {
        txn.delete_collection(self.pool, self.collection_id)?;
        Ok(())
    }

    /// Check the scope's invariants: every id below the counter has a reverse
    /// record whose key maps forward to the same id, and every edge list in
    /// `families` decodes with targets that resolve.
    ///
    /// # Errors
    /// Returns an error only if a record cannot be read; invariant violations
    /// are reported in the returned report.
    pub fn verify(
        &self,
        db: &SharedDatabase,
        families: &[EdgeFamily],
    ) -> Result<ShortIdVerifyReport, StorageError> {
        let txn = db.begin_transaction();
        let (counter, _) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[COUNTER_ID])?;
        let mut report = ShortIdVerifyReport::default();
        let next = match counter.into_iter().next().flatten() {
            Some(record) => decode_counter(&record.bytes)?,
            None => return Ok(report),
        };
        let get = |id: NodeId| -> Result<Option<NodeData>, StorageError> {
            let (records, _) =
                txn.get_with_record_versions(self.pool, &self.collection_id, &[id])?;
            Ok(records.into_iter().next().flatten())
        };
        for short_id in 1..next {
            let Some(reverse) = get(indexed_id(REVERSE_PREFIX, short_id))? else {
                report
                    .problems
                    .push(format!("id {short_id} below counter has no reverse record"));
                continue;
            };
            report.ids_checked = report.ids_checked.saturating_add(1);
            let key = check_header(&reverse.bytes, REVERSE_MAGIC, "reverse")?;
            match get(forward_id(key))? {
                Some(forward) => {
                    let (forward_short, forward_key) = decode_forward(&forward.bytes)?;
                    if forward_short != short_id || forward_key != key {
                        report.problems.push(format!(
                            "id {short_id}: forward record disagrees (id {forward_short})"
                        ));
                    }
                }
                None => report
                    .problems
                    .push(format!("id {short_id}: reverse has no forward record")),
            }
            for family in families {
                let Some(edges) = get(edges_record_id(short_id, family.id))? else {
                    continue;
                };
                report.edge_lists_checked = report.edge_lists_checked.saturating_add(1);
                for edge in decode_edges(*family, &edges.bytes)? {
                    if edge.target == 0 || edge.target >= next {
                        report.problems.push(format!(
                            "id {short_id}: family {} target {} is out of range",
                            family.id, edge.target
                        ));
                    } else if get(indexed_id(REVERSE_PREFIX, edge.target))?.is_none() {
                        report.problems.push(format!(
                            "id {short_id}: family {} target {} unresolved",
                            family.id, edge.target
                        ));
                    }
                }
            }
        }
        Ok(report)
    }

    /// `families` is `Some` when `keys[0]` owns the edge lists built from the
    /// remaining keys (in family order), or `None` for a plain allocation.
    fn write_with_retry(
        &self,
        db: &SharedDatabase,
        keys: &[&[u8]],
        families: Option<&[FamilyEdges<'_>]>,
    ) -> Result<(Vec<u32>, RecordedEvent), StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            match self.attempt(&txn, keys, families) {
                Ok(outcome) if !outcome.commit_needed => {
                    return Ok((outcome.ids, outcome.recorded))
                }
                Ok(outcome) => match txn.commit() {
                    Ok(()) => return Ok((outcome.ids, outcome.recorded)),
                    Err(error) if error.is_stale_read() => last = Some(error),
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_stale_read() => last = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("short-id retry budget".to_owned())))
    }

    /// Resolve every key to an id, allocating for keys absent from the store.
    fn allocate(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
    ) -> Result<Allocation, StorageError> {
        // Read the counter and every forward record together, before staging
        // anything: staged records report token 0, and the tokens must match
        // the data they guard.
        let mut read_ids: Vec<NodeId> = Vec::with_capacity(keys.len().saturating_add(1));
        read_ids.push(COUNTER_ID);
        read_ids.extend(keys.iter().map(|key| forward_id(key)));
        let (records, tokens) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &read_ids)?;
        let mut next = match &records[0] {
            Some(record) => decode_counter(&record.bytes)?,
            None => 1,
        };
        let counter_token = tokens[0];
        let mut allocation = Allocation {
            ids: Vec::with_capacity(keys.len()),
            fresh: Vec::new(),
            staged: Vec::new(),
            counter: (counter_token, next),
        };
        let mut seen: HashMap<&[u8], u32> = HashMap::new();
        for (index, key) in keys.iter().enumerate() {
            let slot = index.saturating_add(1);
            if let Some(record) = &records[slot] {
                let (short_id, stored_key) = decode_forward(&record.bytes)?;
                if stored_key != *key {
                    return Err(StorageError::Collision(
                        "short-id key hash collision between different keys".to_owned(),
                    ));
                }
                allocation.ids.push(short_id);
            } else if let Some(short_id) = seen.get(key) {
                allocation.ids.push(*short_id);
            } else {
                if next > SHORT_ID_MAX {
                    return Err(StorageError::Internal(
                        "short-id space exhausted for this scope".to_owned(),
                    ));
                }
                let short_id = next;
                next = next.saturating_add(1);
                seen.insert(key, short_id);
                allocation.fresh.push(short_id);
                allocation.ids.push(short_id);
                allocation.staged.push((
                    forward_id(key),
                    encode_forward(short_id, key),
                    Some(tokens[slot]),
                ));
                allocation.staged.push((
                    indexed_id(REVERSE_PREFIX, short_id),
                    encode_reverse(key),
                    None,
                ));
            }
        }
        allocation.counter.1 = next;
        Ok(allocation)
    }

    /// Merge the requested edge lists with what is stored and stage the changes.
    fn stage_edges(
        &self,
        txn: &DatabaseTransaction<'_>,
        allocation: &mut Allocation,
        families: &[FamilyEdges<'_>],
    ) -> Result<Vec<Vec<Edge>>, StorageError> {
        let owner = allocation.ids[0];
        // Read every existing edge record of the owner in one call, before
        // staging, so each token matches the data it guards. A freshly
        // allocated owner has none.
        let existing: Vec<Option<(Option<NodeData>, u64)>> = if allocation.fresh.contains(&owner) {
            vec![None; families.len()]
        } else {
            let ids: Vec<NodeId> = families
                .iter()
                .map(|f| edges_record_id(owner, f.family.id))
                .collect();
            let (records, tokens) =
                txn.get_with_record_versions(self.pool, &self.collection_id, &ids)?;
            records.into_iter().zip(tokens).map(Some).collect()
        };
        let mut lists = Vec::with_capacity(families.len());
        let mut cursor = 1usize;
        for (family, existing) in families.iter().zip(existing) {
            let mut wanted: BTreeSet<Edge> = BTreeSet::new();
            for edge in family.edges {
                wanted.insert(Edge {
                    target: allocation.ids[cursor],
                    kind: edge.kind,
                });
                cursor = cursor.saturating_add(1);
            }
            let (stored, token) = match existing {
                Some((Some(record), token)) => (
                    Some(decode_edges(family.family, &record.bytes)?),
                    Some(token),
                ),
                Some((None, token)) => (None, Some(token)),
                None => (None, None),
            };
            let merged: Vec<Edge> = match (&stored, family.family.merge) {
                (Some(stored), EdgeMerge::Immutable) => {
                    let wanted: Vec<Edge> = wanted.into_iter().collect();
                    if *stored != wanted {
                        return Err(StorageError::Collision(
                            "short-id edge list differs from the recorded one".to_owned(),
                        ));
                    }
                    wanted
                }
                (Some(stored), EdgeMerge::Union) => {
                    wanted.extend(stored.iter().copied());
                    wanted.into_iter().collect()
                }
                (None, _) => wanted.into_iter().collect(),
            };
            if stored.as_ref() != Some(&merged) {
                allocation.staged.push((
                    edges_record_id(owner, family.family.id),
                    encode_edges(family.family, &merged)?,
                    token,
                ));
            }
            lists.push(merged);
        }
        Ok(lists)
    }

    fn attempt(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
        families: Option<&[FamilyEdges<'_>]>,
    ) -> Result<Attempt, StorageError> {
        let mut allocation = self.allocate(txn, keys)?;
        let edges = match families {
            Some(families) => self.stage_edges(txn, &mut allocation, families)?,
            None => Vec::new(),
        };
        let ids = allocation.ids.clone();
        let recorded = RecordedEvent { id: ids[0], edges };
        if allocation.staged.is_empty() {
            return Ok(Attempt {
                recorded,
                ids,
                commit_needed: false,
            });
        }
        if !allocation.fresh.is_empty() {
            let (token, next) = allocation.counter;
            txn.expect_record_version(self.pool, self.collection_id, COUNTER_ID, token)?;
            txn.put(
                self.pool,
                self.collection_id,
                COUNTER_ID,
                &NodeData::new(Bytes::from(encode_counter(next))),
            )?;
        }
        for (id, payload, token) in allocation.staged {
            if let Some(token) = token {
                txn.expect_record_version(self.pool, self.collection_id, id, token)?;
            }
            txn.put(
                self.pool,
                self.collection_id,
                id,
                &NodeData::new(Bytes::from(payload)),
            )?;
        }
        Ok(Attempt {
            recorded,
            ids,
            commit_needed: true,
        })
    }
}

struct Allocation {
    /// Short id of each input key, in order.
    ids: Vec<u32>,
    /// Ids allocated by this attempt.
    fresh: Vec<u32>,
    /// `(record id, payload, CAS token)`; a `None` token means a fresh record.
    staged: Vec<(NodeId, Vec<u8>, Option<u64>)>,
    /// `(counter token, next id after this attempt)`.
    counter: (u64, u32),
}

struct Attempt {
    recorded: RecordedEvent,
    ids: Vec<u32>,
    commit_needed: bool,
}
