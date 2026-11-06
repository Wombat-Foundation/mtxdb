//! Room-scoped dense `u32` short ids.
//!
//! This is the generic substrate for graph workloads that need compact integer
//! ids (auth-chain closure bitmaps, relation walks). It knows nothing about
//! Matrix: a *scope* (one collection, e.g. a room) maps opaque key bytes to
//! sequentially allocated `u32` ids. Adjacency is not stored here; it lives in
//! the CSR store (see [`crate::csr`]) keyed by these ids.
//!
//! Records, all in one collection so a scope purges with one
//! `delete_collection`:
//!
//! - **counter** (`SIDC`): the next id to allocate;
//! - **forward** (`SIDF`): `hash(key) -> id + key bytes`. The node id is a
//!   truncated hash, so the stored key is compared on every hit and a mismatch
//!   is reported as a collision, never silently list;
//! - **reverse** (`SIDR`): `id -> key bytes`;
//! - **owner log**: the keys recorded as owners, in order.
//!
//! Allocation commits in one [`DatabaseTransaction`] guarded by the engine's
//! per-record compare-and-swap (the counter and every forward record that was
//! absent). A lost race surfaces as `StorageError::StaleRead` and the
//! operation re-reads and retries, finding the winner's state. There is no
//! partial state to repair: counter, forward and reverse records publish
//! together or not at all.
//!
//! Ids start at 1 and never wrap; exhausting `u32` is a checked error. The
//! CAS is atomic within the single writer process, through the transaction layer
//! in [`crate::database`].

use std::collections::{HashMap, HashSet};

use bytes::Bytes;

use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Wire version shared by every short-id record. Version 2 intentionally
/// rejects the pre-CSR format, whose `SIDE` adjacency records are no longer
/// read or migrated.
pub const SHORT_ID_FORMAT_VERSION: u8 = 2;
/// Largest id that may be allocated.
pub const SHORT_ID_MAX: u32 = u32::MAX - 1;

const COUNTER_MAGIC: [u8; 4] = *b"SIDC";
const FORWARD_MAGIC: [u8; 4] = *b"SIDF";
const REVERSE_MAGIC: [u8; 4] = *b"SIDR";
const OWNER_LOG_MAGIC: [u8; 4] = *b"OWNL";

pub(crate) const COUNTER_ID: NodeId = *b"MTXD-SID-CNTR-v1";
const REVERSE_PREFIX: [u8; 8] = *b"MTXSIDR\0";
const OWNER_LOG_PREFIX: [u8; 8] = *b"MTXOWNL\0";
const OWNER_LOG_SCOPE: &[u8] = b"short-id-owner-log";

/// Forward-record flag: the key has been recorded as an *owner* (an event the
/// caller holds), as opposed to merely allocated an id as an edge target.
const FORWARD_OWNER: u8 = 0b01;

/// Maximum optimistic-retry rounds before a stale read is returned.
pub(crate) const MAX_ATTEMPTS: usize = 64;

/// One event of a [`ShortIdIndex::record_events`](crate::short_id::ShortIdIndex::record_events)
/// batch: an owner key and its optional owner-log payload.
#[derive(Debug, Clone, Copy)]
pub struct BatchEvent<'a> {
    /// The owner's key.
    pub owner: &'a [u8],
    /// If set, this event is an *owner*: one the caller holds, as opposed to
    /// an id allocated only because something referenced it. The first time a
    /// key is recorded as an owner (its owner bit goes false -> true) the
    /// payload is appended to the owner log under the next sequence number;
    /// recording the same owner again, in this batch or later, appends
    /// nothing. The payload is opaque here (for populations, the element
    /// hashes). The same owner twice in a batch must carry the same payload.
    pub owner_payload: Option<&'a [u8]>,
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
pub(crate) fn derived_id(prefix: [u8; 8], short_id: u32, discriminator: u16) -> NodeId {
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

pub(crate) fn header(magic: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&magic);
    out.push(SHORT_ID_FORMAT_VERSION);
    out
}

pub(crate) fn check_header<'a>(
    bytes: &'a [u8],
    magic: [u8; 4],
    what: &str,
) -> Result<&'a [u8], StorageError> {
    if bytes.len() < 5 || bytes[..4] != magic || bytes[4] != SHORT_ID_FORMAT_VERSION {
        return Err(StorageError::Corrupt(format!("short-id {what} header")));
    }
    Ok(&bytes[5..])
}

pub(crate) fn read_u32(bytes: &[u8], what: &str) -> Result<u32, StorageError> {
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
/// folded into a compacted run. `manifest_version` names the run manifest that
/// covers the folded entries (0: none); it lives here so a reader gets the
/// manifest and the log tail it pairs with from one record.
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
    /// The run manifest covering owner-log sequence numbers below
    /// `log_epoch_start_seq` (0: no runs).
    pub manifest_version: u32,
}

impl Default for ScopeCounters {
    fn default() -> Self {
        Self {
            next_id: 1,
            next_owner_seq: 1,
            log_epoch: 0,
            log_epoch_start_seq: 1,
            manifest_version: 0,
        }
    }
}

pub(crate) fn encode_counter(counters: ScopeCounters) -> Vec<u8> {
    let mut out = header(COUNTER_MAGIC);
    for value in [
        counters.next_id,
        counters.next_owner_seq,
        counters.log_epoch,
        counters.log_epoch_start_seq,
        counters.manifest_version,
    ] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    out
}

fn decode_counter(bytes: &[u8]) -> Result<ScopeCounters, StorageError> {
    let body = check_header(bytes, COUNTER_MAGIC, "counter")?;
    if body.len() != 20 {
        return Err(StorageError::Corrupt("short-id counter length".to_owned()));
    }
    let counters = ScopeCounters {
        next_id: read_u32(&body[..4], "counter")?,
        next_owner_seq: read_u32(&body[4..8], "counter")?,
        log_epoch: read_u32(&body[8..12], "counter")?,
        log_epoch_start_seq: read_u32(&body[12..16], "counter")?,
        manifest_version: read_u32(&body[16..20], "counter")?,
    };
    // Ids and owner sequence numbers start at 1, so a counter naming 0 would
    // allocate a number the scope never hands out and `verify` never visits.
    if counters.next_id == 0 || counters.next_owner_seq == 0 || counters.log_epoch_start_seq == 0 {
        return Err(StorageError::Corrupt(
            "short-id counter names a zero id or sequence".to_owned(),
        ));
    }
    Ok(counters)
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

pub(crate) fn encode_owner_log(short_id: u32, payload: &[u8]) -> Vec<u8> {
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

pub(crate) fn owner_log_record_id(seq: u32) -> NodeId {
    indexed_id(OWNER_LOG_PREFIX, seq)
}

fn encode_reverse(key: &[u8]) -> Vec<u8> {
    let mut out = header(REVERSE_MAGIC);
    out.extend_from_slice(key);
    out
}

/// Result of [`ShortIdIndex::verify`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortIdVerifyReport {
    /// Ids below the counter that have a reverse record.
    pub ids_checked: u32,
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
    pub(crate) pool: ShardType,
    /// The pool holding owner-log generations and compacted runs. Defaults to
    /// `pool`; a pool holding little else lets a dropped generation's shards
    /// retire (see [`Self::with_log_pool`]).
    pub(crate) log_pool: ShardType,
    pub(crate) collection_id: [u8; 16],
    max_id: u32,
}

impl ShortIdIndex {
    /// Address the scope stored in `collection_id` of `pool`.
    #[must_use]
    pub fn new(pool: ShardType, collection_id: [u8; 16]) -> Self {
        Self {
            pool,
            log_pool: pool,
            collection_id,
            max_id: SHORT_ID_MAX,
        }
    }

    /// Keep owner-log generations and compacted runs in `log_pool` instead of
    /// the scope's own pool.
    ///
    /// Dropping a collection is a tombstone; its bytes come back only when a
    /// repack retires every shard they sit in. In the scope's own pool the log
    /// shares shards with the id records, which stay live, so a dropped
    /// generation frees nothing (measured at 1M owners). In a pool that holds
    /// little else, a drop followed by `repack_live_collections` retires whole
    /// shards. Counter, manifests and the garbage list stay in the scope's
    /// pool, and one transaction may span both.
    ///
    /// Choose it once per scope, before any owner is recorded: entries
    /// already written stay where they were.
    #[must_use]
    pub fn with_log_pool(mut self, log_pool: ShardType) -> Self {
        self.log_pool = log_pool;
        self
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

    pub(crate) fn read_counters(
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
    /// sequence, or if an entry in range is corrupt. An entry that is missing
    /// (its generation was dropped) is [`StorageError::NotFound`].
    pub fn owner_log(
        &self,
        db: &Database,
        epoch: u32,
        from: u32,
        to: u32,
    ) -> Result<Vec<OwnerLogEntry>, StorageError> {
        let txn = db.begin_transaction();
        self.owner_log_in(&txn, epoch, from, to)
    }

    /// [`Self::owner_log`] inside the caller's transaction.
    pub(crate) fn owner_log_in(
        &self,
        txn: &DatabaseTransaction<'_>,
        epoch: u32,
        from: u32,
        to: u32,
    ) -> Result<Vec<OwnerLogEntry>, StorageError> {
        let (counters, _) = self.read_counters(txn)?;
        let below_current = epoch == counters.log_epoch && from < counters.log_epoch_start_seq;
        if epoch > counters.log_epoch || below_current || to > counters.next_owner_seq || from > to
        {
            return Err(StorageError::Internal(
                "owner-log range outside the requested generation".to_owned(),
            ));
        }
        let ids: Vec<NodeId> = (from..to).map(owner_log_record_id).collect();
        let (records, _) =
            txn.get_with_record_versions(self.log_pool, &self.owner_log_collection(epoch), &ids)?;
        (from..to)
            .zip(records)
            .map(|(seq, record)| {
                let record = record.ok_or(StorageError::NotFound(owner_log_record_id(seq)))?;
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
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.write_with_retry(db, keys, &HashMap::new())
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

    /// Record several owners in **one** transaction.
    ///
    /// Ids are allocated once for every owner across the whole batch, and
    /// either every owner is recorded or none is. An owner with a payload is
    /// appended to the owner log the first time it becomes an owner (see
    /// [`BatchEvent::owner_payload`]). Returns each owner's id, in order.
    ///
    /// The transaction layer bounds how much one transaction may stage, so a
    /// caller with a very large import should split it into batches.
    ///
    /// # Errors
    /// As [`Self::get_or_create`], plus a collision error when one owner
    /// carries two different payloads in the batch.
    pub fn record_events(
        &self,
        db: &Database,
        events: &[BatchEvent<'_>],
    ) -> Result<Vec<u32>, StorageError> {
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let keys: Vec<&[u8]> = events.iter().map(|event| event.owner).collect();
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
        self.write_with_retry(db, &keys, &owners)
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

    /// Remove the scope's counter, id mappings, and owner log in one commit.
    ///
    /// Adjacency is stored separately; callers that own the graph must also
    /// tombstone its rows with `CsrAdjacency::purge_graph`.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged or committed.
    pub fn purge(&self, db: &Database) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        self.stage_purge(&txn)?;
        txn.commit()
    }

    /// Stage removal of this index's counter, id mappings, and owner log into
    /// `txn`. CSR adjacency rows are tombstoned separately by
    /// `CsrAdjacency::purge_graph`.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be staged.
    pub fn stage_purge(&self, txn: &DatabaseTransaction<'_>) -> Result<(), StorageError> {
        // The owner-log generations are separate collections; the counter
        // says how many there have been.
        let (counters, _) = self.read_counters(txn)?;
        for epoch in 0..=counters.log_epoch {
            txn.delete_collection(self.log_pool, self.owner_log_collection(epoch))?;
        }
        self.stage_population_purge(txn, &counters)?;
        txn.delete_collection(self.pool, self.collection_id)?;
        Ok(())
    }

    /// Without `reconcile` nothing writes runs, so there is nothing to drop.
    #[cfg(not(feature = "reconcile"))]
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn stage_population_purge(
        &self,
        _txn: &DatabaseTransaction<'_>,
        _counters: &ScopeCounters,
    ) -> Result<(), StorageError> {
        Ok(())
    }

    /// Without `reconcile` nothing writes manifests, so there is nothing to
    /// check.
    #[cfg(not(feature = "reconcile"))]
    #[allow(clippy::unused_self)]
    fn verify_manifest(
        &self,
        _txn: &DatabaseTransaction<'_>,
        _counters: &ScopeCounters,
        _problems: &mut [String],
    ) {
    }

    /// Check the scope's invariants: every id below the counter has a reverse
    /// record whose key maps forward to the same id.
    ///
    /// # Errors
    /// Returns an error only if a record cannot be read; invariant violations
    /// are reported in the returned report.
    pub fn verify(&self, db: &Database) -> Result<ShortIdVerifyReport, StorageError> {
        // A transaction reads the live store, not a snapshot, so a writer
        // committing mid-pass can make a healthy store look inconsistent.
        // Every write moves the counter record (ids and owner sequence
        // numbers both live there), so a pass is trusted only if the counter's
        // version is unchanged when it ends.
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            let (_, before) = self.read_counters(&txn)?;
            let report = self.verify_pass(&txn)?;
            let (_, after) = self.read_counters(&txn)?;
            if before == after {
                return Ok(report);
            }
        }
        Err(StorageError::Internal(
            "short-id verify: store changed on every pass".to_owned(),
        ))
    }

    fn verify_pass(
        &self,
        txn: &DatabaseTransaction<'_>,
    ) -> Result<ShortIdVerifyReport, StorageError> {
        let (counters, _) = self.read_counters(txn)?;
        let mut report = ShortIdVerifyReport::default();
        if counters == ScopeCounters::default() {
            return Ok(report);
        }
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
        }
        self.verify_owner_log(txn, counters, &mut report)?;
        self.verify_manifest(txn, &counters, &mut report.problems);
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
        txn: &DatabaseTransaction<'_>,
        counters: ScopeCounters,
        report: &mut ShortIdVerifyReport,
    ) -> Result<(), StorageError> {
        let get = |id: NodeId| -> Result<Option<NodeData>, StorageError> {
            let (records, _) =
                txn.get_with_record_versions(self.pool, &self.collection_id, &[id])?;
            Ok(records.into_iter().next().flatten())
        };
        match self.owner_log_in(
            txn,
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

    /// Allocate every key in one transaction, retrying a stale read.
    fn write_with_retry(
        &self,
        db: &Database,
        keys: &[&[u8]],
        owners: &HashMap<&[u8], &[u8]>,
    ) -> Result<Vec<u32>, StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            match self.attempt(&txn, keys, owners) {
                Ok((ids, false)) => return Ok(ids),
                Ok((ids, true)) => match txn.commit() {
                    Ok(()) => return Ok(ids),
                    Err(error) if error.is_stale_read() => last = Some(error),
                    Err(error) => return Err(error),
                },
                Err(error)
                    if error.is_stale_read() || matches!(error, StorageError::WouldBlock(_)) =>
                {
                    last = Some(error);
                }
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
                    self.log_pool,
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

    /// One attempt: allocate every key, then stage the counter and records.
    /// Returns the ids and whether anything needs committing.
    fn attempt(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
        owners: &HashMap<&[u8], &[u8]>,
    ) -> Result<(Vec<u32>, bool), StorageError> {
        let allocation = self.allocate(txn, keys, owners)?;
        let ids = allocation.ids.clone();
        if allocation.staged.is_empty() {
            return Ok((ids, false));
        }
        self.stage_allocation(txn, allocation)?;
        Ok((ids, true))
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

#[cfg(test)]
mod counter_tests {
    use super::*;

    #[test]
    fn counters_that_name_zero_are_corrupt() {
        let valid = ScopeCounters::default();
        assert_eq!(decode_counter(&encode_counter(valid)).unwrap(), valid);
        for broken in [
            ScopeCounters {
                next_id: 0,
                ..valid
            },
            ScopeCounters {
                next_owner_seq: 0,
                ..valid
            },
            ScopeCounters {
                log_epoch_start_seq: 0,
                ..valid
            },
        ] {
            let error = decode_counter(&encode_counter(broken)).unwrap_err();
            assert!(
                matches!(error, StorageError::Corrupt(_)),
                "{broken:?}: {error:?}"
            );
        }
    }
}
