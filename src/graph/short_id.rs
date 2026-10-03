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
//!   is reported as a collision, never silently list;
//! - **reverse** (`SIDR`): `id -> key bytes`;
//! - **edges** (`SIDE`): `(family, id) -> sorted [Edge]`, where an [`Edge`](crate::short_id::Edge) is a
//!   `u32` target plus, for typed families, a `u16` kind. A record is never
//!   empty (it has a header), so a known owner with zero edges is distinguishable
//!   from an absent record.
//!
//! An [`EdgeFamily`](crate::short_id::EdgeFamily) fixes whether edges are typed. Edge lists are **immutable
//! once written**: writing the same list again is a no-op and a different list
//! for an existing owner is a collision. Mutable visibility (for example
//! redaction or rejection of the owning record) is not modelled here; callers
//! filter at read time against their own authoritative state. The typing is
//! stored in each record and checked on every access, so a caller cannot
//! reinterpret a family.
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
//! CAS is atomic within the single writer process, through the transaction layer
//! in [`crate::database`].

use std::collections::{BTreeSet, HashMap, HashSet};

use bytes::Bytes;

use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Wire version shared by every short-id record.
pub const SHORT_ID_FORMAT_VERSION: u8 = 4;
/// Largest id that may be allocated.
pub const SHORT_ID_MAX: u32 = u32::MAX - 1;

const COUNTER_MAGIC: [u8; 4] = *b"SIDC";
const FORWARD_MAGIC: [u8; 4] = *b"SIDF";
const REVERSE_MAGIC: [u8; 4] = *b"SIDR";
const EDGES_MAGIC: [u8; 4] = *b"SIDE";
const OWNER_LOG_MAGIC: [u8; 4] = *b"OWNL";

const COUNTER_ID: NodeId = *b"MTXD-SID-CNTR-v1";
const REVERSE_PREFIX: [u8; 8] = *b"MTXSIDR\0";
const EDGES_PREFIX: [u8; 8] = *b"MTXSIDE\0";
const OWNER_LOG_PREFIX: [u8; 8] = *b"MTXOWNL\0";
const OWNER_LOG_SCOPE: &[u8] = b"short-id-owner-log";

/// Forward-record flag: the key has been recorded as an *owner* (an event the
/// caller holds), as opposed to merely allocated an id as an edge target.
const FORWARD_OWNER: u8 = 0b01;

const FLAG_TYPED: u8 = 0b01;

/// Maximum optimistic-retry rounds before a stale read is returned.
const MAX_ATTEMPTS: usize = 64;

/// A named adjacency family within a scope. Its edge lists are immutable once
/// written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeFamily {
    /// Family number; part of the edge record's address.
    pub id: u16,
    /// Whether each edge carries a `u16` kind (otherwise the kind is always 0).
    pub typed: bool,
}

impl EdgeFamily {
    /// An untyped family.
    #[must_use]
    pub const fn plain(id: u16) -> Self {
        Self { id, typed: false }
    }

    /// A family whose edges each carry a `u16` kind.
    #[must_use]
    pub const fn typed(id: u16) -> Self {
        Self { id, typed: true }
    }

    const fn flags(self) -> u8 {
        if self.typed {
            FLAG_TYPED
        } else {
            0
        }
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

/// One event of a [`ShortIdIndex::record_events`](crate::short_id::ShortIdIndex::record_events)
/// batch: an owner key and its edge lists.
#[derive(Debug, Clone, Copy)]
pub struct BatchEvent<'a> {
    /// The owner's key.
    pub owner: &'a [u8],
    /// The owner's edge lists, one per family.
    pub families: &'a [FamilyEdges<'a>],
    /// If set, this event is an *owner*: one the caller holds, as opposed to
    /// an id allocated only because something referenced it. The first time a
    /// key is recorded as an owner (its owner bit goes false -> true) the
    /// payload is appended to the owner log under the next sequence number;
    /// recording the same owner again, in this batch or later, appends
    /// nothing. The payload is opaque here (for populations, the element
    /// hashes). The same owner twice in a batch must carry the same payload.
    pub owner_payload: Option<&'a [u8]>,
}

/// Outcome of [`ShortIdIndex::record_event`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvent {
    /// The owner's short id.
    pub id: u32,
    /// The stored edge list of each requested family, in request order.
    pub edges: Vec<Vec<Edge>>,
}

fn forward_id(key: &[u8]) -> NodeId {
    let digest = DigestAlgorithm::Blake3.digest(key);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// A record id derived from a prefix, a short id and a discriminator.
///
/// The id is a hash, not a layout. The engine's hash index picks a record's
/// bucket from the first 8 bytes of its id and its tag from the next 4, and
/// assumes ids are uniformly distributed. An id with a constant prefix there
/// puts every such record in one bucket and gives them all one tag, which turns
/// each insert into a scan of the whole chain: ingesting a 60,000-event room took
/// about a minute that way. Hashing spreads both fields.
fn derived_id(prefix: [u8; 8], short_id: u32, discriminator: u16) -> NodeId {
    let mut input = [0u8; 14];
    input[..8].copy_from_slice(&prefix);
    input[8..12].copy_from_slice(&short_id.to_be_bytes());
    input[12..14].copy_from_slice(&discriminator.to_be_bytes());
    let digest = DigestAlgorithm::Blake3.digest(&input);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn indexed_id(prefix: [u8; 8], short_id: u32) -> NodeId {
    derived_id(prefix, short_id, 0)
}

fn edges_record_id(short_id: u32, family: u16) -> NodeId {
    derived_id(EDGES_PREFIX, short_id, family)
}

/// The id of the reverse record for `short_id`, exposed so a test can check the
/// ids the index will see are spread.
#[cfg(test)]
pub(crate) fn reverse_record_id_for_test(short_id: u32) -> NodeId {
    indexed_id(REVERSE_PREFIX, short_id)
}

/// The id of an owner-log entry, exposed so a test can corrupt one.
#[cfg(test)]
pub(crate) fn owner_log_record_id_for_test(seq: u32) -> NodeId {
    owner_log_record_id(seq)
}

/// The id of an edges record, exposed for the same reason.
#[cfg(test)]
pub(crate) fn edges_record_id_for_test(short_id: u32, family: u16) -> NodeId {
    edges_record_id(short_id, family)
}

fn header(magic: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&magic);
    out.push(SHORT_ID_FORMAT_VERSION);
    out
}

fn check_header<'a>(bytes: &'a [u8], magic: [u8; 4], what: &str) -> Result<&'a [u8], StorageError> {
    if bytes.len() < 5 || bytes[..4] != magic {
        return Err(StorageError::Corrupt(format!("short-id {what} header")));
    }
    if bytes[4] != SHORT_ID_FORMAT_VERSION {
        return Err(StorageError::Corrupt(format!(
            "unsupported short-id format v{} in {what} record (this build reads v{SHORT_ID_FORMAT_VERSION})",
            bytes[4]
        )));
    }
    Ok(&bytes[5..])
}

fn read_u32(bytes: &[u8], what: &str) -> Result<u32, StorageError> {
    <[u8; 4]>::try_from(bytes)
        .map(u32::from_be_bytes)
        .map_err(|_| StorageError::Corrupt(format!("short-id {what} length")))
}

/// The scope's allocation counters, all stored in one CAS-guarded record.
///
/// Ids and owner sequence numbers are dense and start at 1. The owner log is
/// written into one *generation* collection at a time, named by `log_epoch`;
/// `log_epoch_start_seq` is the first sequence number of the current
/// generation, so earlier entries live in an older generation or have been
/// folded into a compacted run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeCounters {
    /// The next short id to allocate.
    pub next_id: u32,
    /// The next owner-log sequence number to assign.
    pub next_owner_seq: u32,
    /// The generation the owner log is currently written to.
    pub log_epoch: u32,
    /// First owner-log sequence number of `log_epoch`.
    pub log_epoch_start_seq: u32,
}

impl Default for ScopeCounters {
    fn default() -> Self {
        Self {
            next_id: 1,
            next_owner_seq: 1,
            log_epoch: 0,
            log_epoch_start_seq: 1,
        }
    }
}

fn encode_counter(counters: ScopeCounters) -> Vec<u8> {
    let mut out = header(COUNTER_MAGIC);
    for value in [
        counters.next_id,
        counters.next_owner_seq,
        counters.log_epoch,
        counters.log_epoch_start_seq,
    ] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    out
}

fn decode_counter(bytes: &[u8]) -> Result<ScopeCounters, StorageError> {
    let body = check_header(bytes, COUNTER_MAGIC, "counter")?;
    if body.len() != 16 {
        return Err(StorageError::Corrupt("short-id counter length".to_owned()));
    }
    Ok(ScopeCounters {
        next_id: read_u32(&body[..4], "counter")?,
        next_owner_seq: read_u32(&body[4..8], "counter")?,
        log_epoch: read_u32(&body[8..12], "counter")?,
        log_epoch_start_seq: read_u32(&body[12..16], "counter")?,
    })
}

fn encode_forward(short_id: u32, owner: bool, key: &[u8]) -> Vec<u8> {
    let mut out = header(FORWARD_MAGIC);
    out.push(if owner { FORWARD_OWNER } else { 0 });
    out.extend_from_slice(&short_id.to_be_bytes());
    out.extend_from_slice(key);
    out
}

/// A decoded forward record.
struct Forward<'a> {
    short_id: u32,
    owner: bool,
    key: &'a [u8],
}

fn decode_forward(bytes: &[u8]) -> Result<Forward<'_>, StorageError> {
    let body = check_header(bytes, FORWARD_MAGIC, "forward")?;
    if body.len() < 5 || body[0] & !FORWARD_OWNER != 0 {
        return Err(StorageError::Corrupt("short-id forward length".to_owned()));
    }
    Ok(Forward {
        short_id: read_u32(&body[1..5], "forward")?,
        owner: body[0] & FORWARD_OWNER != 0,
        key: &body[5..],
    })
}

fn encode_owner_log(short_id: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = header(OWNER_LOG_MAGIC);
    out.extend_from_slice(&short_id.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// One owner-log entry: the event that became an owner at `seq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerLogEntry {
    /// Dense, never-reused sequence number.
    pub seq: u32,
    /// The owner's short id.
    pub short_id: u32,
    /// The caller's opaque payload (for populations, the element hashes).
    pub payload: Vec<u8>,
}

fn decode_owner_log(seq: u32, bytes: &[u8]) -> Result<OwnerLogEntry, StorageError> {
    let body = check_header(bytes, OWNER_LOG_MAGIC, "owner log")?;
    if body.len() < 4 {
        return Err(StorageError::Corrupt(
            "short-id owner log length".to_owned(),
        ));
    }
    Ok(OwnerLogEntry {
        seq,
        short_id: read_u32(&body[..4], "owner log")?,
        payload: body[4..].to_vec(),
    })
}

fn owner_log_record_id(seq: u32) -> NodeId {
    indexed_id(OWNER_LOG_PREFIX, seq)
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
            "short-id family {} was stored with different typing",
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
    /// Forward records carrying the owner bit.
    pub owners_checked: u32,
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
    max_id: u32,
}

impl ShortIdIndex {
    /// Address the scope stored in `collection_id` of `pool`.
    #[must_use]
    pub fn new(pool: ShardType, collection_id: [u8; 16]) -> Self {
        Self {
            pool,
            collection_id,
            max_id: SHORT_ID_MAX,
        }
    }

    /// Lower the largest id this scope may allocate (at most [`SHORT_ID_MAX`]),
    /// for consumers whose ids must fit a narrower type. Reaching it is a hard
    /// error inside the allocation transaction, so nothing is published.
    #[must_use]
    pub fn with_max_id(mut self, max_id: u32) -> Self {
        self.max_id = max_id.min(SHORT_ID_MAX);
        self
    }

    /// The pool and collection this scope addresses.
    #[must_use]
    pub const fn scope(&self) -> (ShardType, [u8; 16]) {
        (self.pool, self.collection_id)
    }

    /// The next id this scope will allocate; ids `1..next` are assigned.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt counter.
    pub fn counter(&self, db: &Database) -> Result<u32, StorageError> {
        self.counters(db).map(|counters| counters.next_id)
    }

    /// All of the scope's counters, read together.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt counter.
    pub fn counters(&self, db: &Database) -> Result<ScopeCounters, StorageError> {
        let txn = db.begin_transaction();
        self.read_counters(&txn).map(|(counters, _)| counters)
    }

    fn read_counters(
        &self,
        txn: &DatabaseTransaction<'_>,
    ) -> Result<(ScopeCounters, u64), StorageError> {
        let (records, tokens) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[COUNTER_ID])?;
        match records.into_iter().next().flatten() {
            Some(record) => Ok((decode_counter(&record.bytes)?, tokens[0])),
            None => Ok((ScopeCounters::default(), tokens[0])),
        }
    }

    /// The collection holding generation `epoch` of the owner log. Each
    /// generation is its own collection so a folded generation can be dropped
    /// with one `delete_collection`.
    #[must_use]
    pub fn owner_log_collection(&self, epoch: u32) -> [u8; 16] {
        let mut name = Vec::with_capacity(OWNER_LOG_SCOPE.len().saturating_add(20));
        name.extend_from_slice(OWNER_LOG_SCOPE);
        name.extend_from_slice(&self.collection_id);
        name.extend_from_slice(&epoch.to_be_bytes());
        derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &name)
    }

    /// The owner-log entries of generation `epoch` with `from <= seq < to`.
    ///
    /// The current generation is bounded below by `log_epoch_start_seq`. An
    /// older generation is readable only while it has not been dropped (a
    /// snapshot's grace period); which sequence numbers it holds is the
    /// caller's knowledge, so a missing entry is reported as corruption.
    ///
    /// # Errors
    /// Returns an error if `epoch` is in the future, the range reaches below
    /// the current generation (for `epoch == log_epoch`) or above the assigned
    /// sequence, or if an entry in range is missing or corrupt.
    pub fn owner_log(
        &self,
        db: &Database,
        epoch: u32,
        from: u32,
        to: u32,
    ) -> Result<Vec<OwnerLogEntry>, StorageError> {
        let txn = db.begin_transaction();
        let (counters, _) = self.read_counters(&txn)?;
        let below_current = epoch == counters.log_epoch && from < counters.log_epoch_start_seq;
        if epoch > counters.log_epoch || below_current || to > counters.next_owner_seq || from > to
        {
            return Err(StorageError::Internal(
                "owner-log range outside the requested generation".to_owned(),
            ));
        }
        let ids: Vec<NodeId> = (from..to).map(owner_log_record_id).collect();
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.owner_log_collection(epoch), &ids)?;
        (from..to)
            .zip(records)
            .map(|(seq, record)| {
                let record = record.ok_or_else(|| {
                    StorageError::Corrupt(format!("owner-log entry {seq} is missing"))
                })?;
                decode_owner_log(seq, &record.bytes)
            })
            .collect()
    }

    /// Whether `key` has been recorded as an owner; `None` if it has no id.
    ///
    /// # Errors
    /// As [`Self::lookup`].
    pub fn is_owner(&self, db: &Database, key: &[u8]) -> Result<Option<bool>, StorageError> {
        let txn = db.begin_transaction();
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[forward_id(key)])?;
        match records.into_iter().next().flatten() {
            None => Ok(None),
            Some(record) => {
                let forward = decode_forward(&record.bytes)?;
                if forward.key == key {
                    Ok(Some(forward.owner))
                } else {
                    Err(StorageError::Collision(
                        "short-id key hash collision between different keys".to_owned(),
                    ))
                }
            }
        }
    }

    /// Look up or allocate the short id of every key, in order.
    ///
    /// # Errors
    /// Returns an error on a hash collision between different keys, on
    /// corruption, when the `u32` id space is exhausted, or when the retry
    /// budget is spent on a contended counter.
    pub fn get_or_create(&self, db: &Database, keys: &[&[u8]]) -> Result<Vec<u32>, StorageError> {
        self.write_with_retry(db, keys, None).map(|(ids, _)| ids)
    }

    /// The short id of `key` if it has one; never allocates.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt record, or a hash collision
    /// with a different key.
    pub fn lookup(&self, db: &Database, key: &[u8]) -> Result<Option<u32>, StorageError> {
        let txn = db.begin_transaction();
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[forward_id(key)])?;
        match records.into_iter().next().flatten() {
            None => Ok(None),
            Some(record) => {
                let forward = decode_forward(&record.bytes)?;
                if forward.key == key {
                    Ok(Some(forward.short_id))
                } else {
                    Err(StorageError::Collision(
                        "short-id key hash collision between different keys".to_owned(),
                    ))
                }
            }
        }
    }

    /// Allocate ids for `owner` and every edge target, then store each family's
    /// edge list for `owner`, all in one transaction.
    ///
    /// Edges are sorted and de-duplicated. Per family, a list that matches the
    /// stored one is a no-op and a different one is a collision error. Returns
    /// the owner's id and each family's stored list.
    ///
    /// # Errors
    /// As [`Self::get_or_create`], plus a collision error for a family whose
    /// list differs from the stored one or that was stored with different
    /// typing, or an error for a non-zero kind on an untyped family.
    pub fn record_event(
        &self,
        db: &Database,
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

    /// Record several events' edge lists in **one** transaction.
    ///
    /// Ids are allocated once for every owner and target across the whole batch,
    /// so events that reference each other share ids, and either every event is
    /// recorded or none is. The same owner twice with the same lists is a no-op;
    /// with different lists it is a collision error and nothing is published.
    /// Returns one [`RecordedEvent`] per input, in order.
    ///
    /// The transaction layer bounds how much one transaction may stage, so a
    /// caller with a very large import should split it into batches.
    ///
    /// # Errors
    /// As [`Self::record_event`], applied to the whole batch.
    pub fn record_events(
        &self,
        db: &Database,
        events: &[BatchEvent<'_>],
    ) -> Result<Vec<RecordedEvent>, StorageError> {
        for event in events {
            for family in event.families {
                if !family.family.typed && family.edges.iter().any(|edge| edge.kind != 0) {
                    return Err(StorageError::Internal(
                        "untyped edge family given a non-zero kind".to_owned(),
                    ));
                }
            }
        }
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let mut keys: Vec<&[u8]> = Vec::new();
        let mut starts: Vec<usize> = Vec::with_capacity(events.len());
        for event in events {
            starts.push(keys.len());
            keys.push(event.owner);
            for family in event.families {
                keys.extend(family.edges.iter().map(|edge| edge.target));
            }
        }
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            match self.attempt_batch(&txn, &keys, &starts, events) {
                Ok((recorded, commit_needed)) => {
                    if !commit_needed {
                        return Ok(recorded);
                    }
                    match txn.commit() {
                        Ok(()) => return Ok(recorded),
                        Err(error) if error.is_stale_read() => last = Some(error),
                        Err(error) => return Err(error),
                    }
                }
                Err(error) if error.is_stale_read() => last = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("short-id retry budget".to_owned())))
    }

    /// Convenience for one untyped family: record `owner`'s edges to `targets`
    /// in `family`. Returns `(owner id, sorted target ids)`.
    ///
    /// # Errors
    /// As [`Self::record_event`].
    pub fn record_edges(
        &self,
        db: &Database,
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
        db: &Database,
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
        db: &Database,
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
        db: &Database,
        next: u32,
    ) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        txn.put(
            self.pool,
            self.collection_id,
            COUNTER_ID,
            &NodeData::new(Bytes::from(encode_counter(ScopeCounters {
                next_id: next,
                ..ScopeCounters::default()
            }))),
        )?;
        txn.commit()
    }

    /// Remove the whole scope (counter, mappings and edges) in one commit.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged or committed.
    pub fn purge(&self, db: &Database) -> Result<(), StorageError> {
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
        // The owner-log generations are separate collections; the counter
        // says how many there have been.
        let (counters, _) = self.read_counters(txn)?;
        for epoch in 0..=counters.log_epoch {
            txn.delete_collection(self.pool, self.owner_log_collection(epoch))?;
        }
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
        db: &Database,
        families: &[EdgeFamily],
    ) -> Result<ShortIdVerifyReport, StorageError> {
        let txn = db.begin_transaction();
        let (counter, _) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[COUNTER_ID])?;
        let mut report = ShortIdVerifyReport::default();
        let counters = match counter.into_iter().next().flatten() {
            Some(record) => decode_counter(&record.bytes)?,
            None => return Ok(report),
        };
        let next = counters.next_id;
        let mut owner_bits = 0_u32;
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
                    let stored = decode_forward(&forward.bytes)?;
                    if stored.short_id != short_id || stored.key != key {
                        report.problems.push(format!(
                            "id {short_id}: forward record disagrees (id {})",
                            stored.short_id
                        ));
                    }
                    if stored.owner {
                        owner_bits = owner_bits.saturating_add(1);
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
        self.verify_owner_log(db, &txn, counters, &mut report)?;
        report.owners_checked = owner_bits;
        if owner_bits != counters.next_owner_seq.saturating_sub(1) {
            report.problems.push(format!(
                "{owner_bits} owner bits but {} owner-log sequence numbers assigned",
                counters.next_owner_seq.saturating_sub(1)
            ));
        }
        Ok(report)
    }

    /// The current owner-log generation: every entry exists, decodes, and names
    /// a distinct id whose forward record carries the owner bit. Older
    /// generations were folded into runs and are checked against them.
    fn verify_owner_log(
        &self,
        db: &Database,
        txn: &DatabaseTransaction<'_>,
        counters: ScopeCounters,
        report: &mut ShortIdVerifyReport,
    ) -> Result<(), StorageError> {
        let get = |id: NodeId| -> Result<Option<NodeData>, StorageError> {
            let (records, _) =
                txn.get_with_record_versions(self.pool, &self.collection_id, &[id])?;
            Ok(records.into_iter().next().flatten())
        };
        match self.owner_log(
            db,
            counters.log_epoch,
            counters.log_epoch_start_seq,
            counters.next_owner_seq,
        ) {
            Ok(entries) => {
                let mut distinct: HashSet<u32> = HashSet::new();
                for entry in &entries {
                    if !distinct.insert(entry.short_id) {
                        report.problems.push(format!(
                            "owner-log entry {} repeats id {}",
                            entry.seq, entry.short_id
                        ));
                    }
                    let owned = get(indexed_id(REVERSE_PREFIX, entry.short_id))?
                        .map(|r| {
                            check_header(&r.bytes, REVERSE_MAGIC, "reverse").map(<[u8]>::to_vec)
                        })
                        .transpose()?
                        .and_then(|key| get(forward_id(&key)).transpose())
                        .transpose()?
                        .map(|f| decode_forward(&f.bytes).map(|f| f.owner))
                        .transpose()?;
                    if owned != Some(true) {
                        report.problems.push(format!(
                            "owner-log entry {} names id {} without an owner bit",
                            entry.seq, entry.short_id
                        ));
                    }
                }
            }
            Err(error) => report
                .problems
                .push(format!("owner log unreadable: {error}")),
        }
        Ok(())
    }

    /// `families` is `Some` when `keys[0]` owns the edge lists built from the
    /// remaining keys (in family order), or `None` for a plain allocation.
    fn write_with_retry(
        &self,
        db: &Database,
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
        owners: &HashMap<&[u8], &[u8]>,
    ) -> Result<Allocation, StorageError> {
        // Read the counter and every forward record together, before staging
        // anything: staged records report token 0, and the tokens must match
        // the data they guard.
        let mut read_ids: Vec<NodeId> = Vec::with_capacity(keys.len().saturating_add(1));
        read_ids.push(COUNTER_ID);
        read_ids.extend(keys.iter().map(|key| forward_id(key)));
        let (records, tokens) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &read_ids)?;
        let (mut counters, counter_token) = match &records[0] {
            Some(record) => (decode_counter(&record.bytes)?, tokens[0]),
            None => (ScopeCounters::default(), tokens[0]),
        };
        let mut allocation = Allocation {
            ids: Vec::with_capacity(keys.len()),
            fresh: Vec::new(),
            staged: Vec::new(),
            owner_log: Vec::new(),
            counter: (counter_token, counters),
        };
        let mut seen: HashMap<&[u8], u32> = HashMap::new();
        // Keys whose owner bit this attempt has already set, so a key that
        // appears more than once in the batch flips exactly once.
        let mut flipped: HashSet<u32> = HashSet::new();
        for (index, key) in keys.iter().enumerate() {
            let slot = index.saturating_add(1);
            let payload = owners.get(key).copied();
            if let Some(record) = &records[slot] {
                let forward = decode_forward(&record.bytes)?;
                if forward.key != *key {
                    return Err(StorageError::Collision(
                        "short-id key hash collision between different keys".to_owned(),
                    ));
                }
                allocation.ids.push(forward.short_id);
                // The exactly-once guard: the log entry is staged only when
                // the owner bit goes false -> true, in the same CAS as the
                // forward record that carries the bit. A second record of the
                // same owner (a replay, a re-fetch) finds the bit set and
                // stages nothing, so the population toggle cannot undo itself.
                if let Some(payload) = payload {
                    if !forward.owner && flipped.insert(forward.short_id) {
                        Self::stage_owner(
                            &mut allocation,
                            &mut counters,
                            forward.short_id,
                            payload,
                        )?;
                        allocation.staged.push((
                            forward_id(key),
                            encode_forward(forward.short_id, true, key),
                            Some(tokens[slot]),
                        ));
                    }
                }
            } else if let Some(short_id) = seen.get(key) {
                allocation.ids.push(*short_id);
            } else {
                if counters.next_id > self.max_id {
                    return Err(StorageError::Exhausted(
                        "short-id space exhausted for this scope".to_owned(),
                    ));
                }
                let short_id = counters.next_id;
                counters.next_id = counters.next_id.saturating_add(1);
                seen.insert(key, short_id);
                allocation.fresh.push(short_id);
                allocation.ids.push(short_id);
                if let Some(payload) = payload {
                    Self::stage_owner(&mut allocation, &mut counters, short_id, payload)?;
                }
                allocation.staged.push((
                    forward_id(key),
                    encode_forward(short_id, payload.is_some(), key),
                    Some(tokens[slot]),
                ));
                allocation.staged.push((
                    indexed_id(REVERSE_PREFIX, short_id),
                    encode_reverse(key),
                    None,
                ));
            }
        }
        allocation.counter.1 = counters;
        Ok(allocation)
    }

    /// Assign the next owner-log sequence number to `short_id` and queue its
    /// entry.
    fn stage_owner(
        allocation: &mut Allocation,
        counters: &mut ScopeCounters,
        short_id: u32,
        payload: &[u8],
    ) -> Result<(), StorageError> {
        if counters.next_owner_seq > SHORT_ID_MAX {
            return Err(StorageError::Exhausted(
                "owner-log sequence space exhausted for this scope".to_owned(),
            ));
        }
        let seq = counters.next_owner_seq;
        counters.next_owner_seq = seq.saturating_add(1);
        allocation
            .owner_log
            .push((seq, encode_owner_log(short_id, payload)));
        Ok(())
    }

    /// Merge the requested edge lists with what is stored and stage the changes.
    ///
    /// The owner is `allocation.ids[start]` and its targets follow it. `claimed`
    /// holds the lists already staged by earlier events of the same batch, so the
    /// same owner and family twice either matches or collides rather than staging
    /// two writes of one record.
    fn stage_edges(
        &self,
        txn: &DatabaseTransaction<'_>,
        allocation: &mut Allocation,
        start: usize,
        families: &[FamilyEdges<'_>],
        claimed: &mut HashMap<NodeId, Vec<Edge>>,
    ) -> Result<Vec<Vec<Edge>>, StorageError> {
        let owner = allocation.ids[start];
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
        let mut cursor = start.saturating_add(1);
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
            let list: Vec<Edge> = wanted.into_iter().collect();
            if let Some(stored) = &stored {
                if *stored != list {
                    return Err(StorageError::Collision(
                        "short-id edge list differs from the recorded one".to_owned(),
                    ));
                }
            }
            let record = edges_record_id(owner, family.family.id);
            if let Some(earlier) = claimed.get(&record) {
                // Already staged by an earlier event of this batch.
                if *earlier != list {
                    return Err(StorageError::Collision(
                        "short-id edge list differs from one already in this batch".to_owned(),
                    ));
                }
            } else if stored.is_none() {
                allocation
                    .staged
                    .push((record, encode_edges(family.family, &list)?, token));
                claimed.insert(record, list.clone());
            }
            lists.push(list);
        }
        Ok(lists)
    }

    /// One attempt at a whole batch: allocate every key once, stage each event's
    /// edges, then stage the counter and records. Returns the recorded events and
    /// whether anything needs committing.
    fn attempt_batch(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
        starts: &[usize],
        events: &[BatchEvent<'_>],
    ) -> Result<(Vec<RecordedEvent>, bool), StorageError> {
        let mut owners: HashMap<&[u8], &[u8]> = HashMap::new();
        for event in events {
            if let Some(payload) = event.owner_payload {
                if owners
                    .insert(event.owner, payload)
                    .is_some_and(|p| p != payload)
                {
                    return Err(StorageError::Collision(
                        "short-id owner payload differs from one already in this batch".to_owned(),
                    ));
                }
            }
        }
        let mut allocation = self.allocate(txn, keys, &owners)?;
        let mut claimed: HashMap<NodeId, Vec<Edge>> = HashMap::new();
        let mut recorded = Vec::with_capacity(events.len());
        for (event, &start) in events.iter().zip(starts) {
            let edges =
                self.stage_edges(txn, &mut allocation, start, event.families, &mut claimed)?;
            recorded.push(RecordedEvent {
                id: allocation.ids[start],
                edges,
            });
        }
        if allocation.staged.is_empty() {
            return Ok((recorded, false));
        }
        self.stage_allocation(txn, allocation)?;
        Ok((recorded, true))
    }

    /// Stage the counter bump (if ids were allocated) and every record.
    fn stage_allocation(
        &self,
        txn: &DatabaseTransaction<'_>,
        allocation: Allocation,
    ) -> Result<(), StorageError> {
        if !allocation.fresh.is_empty() || !allocation.owner_log.is_empty() {
            let (token, counters) = allocation.counter;
            txn.expect_record_version(self.pool, self.collection_id, COUNTER_ID, token)?;
            txn.put(
                self.pool,
                self.collection_id,
                COUNTER_ID,
                &NodeData::new(Bytes::from(encode_counter(counters))),
            )?;
            let log = self.owner_log_collection(counters.log_epoch);
            for (seq, payload) in allocation.owner_log {
                txn.put(
                    self.pool,
                    log,
                    owner_log_record_id(seq),
                    &NodeData::new(Bytes::from(payload)),
                )?;
            }
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
        Ok(())
    }

    fn attempt(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
        families: Option<&[FamilyEdges<'_>]>,
    ) -> Result<Attempt, StorageError> {
        let mut allocation = self.allocate(txn, keys, &HashMap::new())?;
        let edges = match families {
            Some(families) => {
                self.stage_edges(txn, &mut allocation, 0, families, &mut HashMap::new())?
            }
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
        self.stage_allocation(txn, allocation)?;
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
    /// Owner-log entries `(seq, payload)` to write into the current generation.
    owner_log: Vec<(u32, Vec<u8>)>,
    /// `(counter token, counters after this attempt)`.
    counter: (u64, ScopeCounters),
}

struct Attempt {
    recorded: RecordedEvent,
    ids: Vec<u32>,
    commit_needed: bool,
}
