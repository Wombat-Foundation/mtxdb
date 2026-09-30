//! Room-scoped dense `u32` short ids and `u32` adjacency lists.
//!
//! This is the generic substrate for graph workloads that need compact integer
//! ids (auth-chain closure bitmaps, relation walks). It knows nothing about
//! Matrix: a *scope* (one collection, e.g. a room) maps opaque key bytes to
//! sequentially allocated `u32` ids, and stores an immutable sorted `u32` edge
//! list per id. The schema-aware layer decides what the keys and edges mean.
//!
//! Records, all in one collection so a scope purges with one
//! `delete_collection`:
//!
//! - **counter** (`SIDC`): the next id to allocate;
//! - **forward** (`SIDF`): `hash(key) -> id + key bytes`. The node id is a
//!   truncated hash, so the stored key is compared on every hit and a mismatch
//!   is reported as a collision, never silently merged;
//! - **reverse** (`SIDR`): `id -> key bytes`;
//! - **edges** (`SIDE`): `id -> [u32]`, always at least the 5-byte header, so a
//!   known leaf with zero edges is distinguishable from an absent record.
//!
//! Allocation and edge writes commit in one [`DatabaseTransaction`] guarded by
//! the engine's per-record compare-and-swap (the counter plus every forward
//! record that was absent when read). A lost race surfaces as
//! `StorageError::StaleRead` and the operation re-reads and retries, finding the
//! winner's mapping. There is no partial state to repair: the counter, forward,
//! reverse and edge records publish together or not at all.
//!
//! Ids start at 1 and never wrap; exhausting `u32` is a checked error. The
//! CAS is atomic within the single writer process; it is unavailable without the
//! `multi-reader` feature, like the transaction layer it uses.

use std::collections::HashMap;

use bytes::Bytes;

use crate::database::{DatabaseTransaction, SharedDatabase};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageError};

/// Wire version shared by every short-id record.
pub const SHORT_ID_FORMAT_VERSION: u8 = 1;
/// Largest id that may be allocated.
pub const SHORT_ID_MAX: u32 = u32::MAX - 1;

const COUNTER_MAGIC: [u8; 4] = *b"SIDC";
const FORWARD_MAGIC: [u8; 4] = *b"SIDF";
const REVERSE_MAGIC: [u8; 4] = *b"SIDR";
const EDGES_MAGIC: [u8; 4] = *b"SIDE";

const COUNTER_ID: NodeId = *b"MTXD-SID-CNTR-v1";
const REVERSE_PREFIX: [u8; 8] = *b"MTXSIDR\0";
const EDGES_PREFIX: [u8; 8] = *b"MTXSIDE\0";

/// Maximum optimistic-retry rounds before a stale read is returned.
const MAX_ATTEMPTS: usize = 64;

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

fn encode_edges(targets: &[u32]) -> Result<Vec<u8>, StorageError> {
    let count = u32::try_from(targets.len())
        .map_err(|_| StorageError::Internal("short-id edge list too long".to_owned()))?;
    let mut out = header(EDGES_MAGIC);
    out.extend_from_slice(&count.to_be_bytes());
    for target in targets {
        out.extend_from_slice(&target.to_be_bytes());
    }
    Ok(out)
}

fn decode_edges(bytes: &[u8]) -> Result<Vec<u32>, StorageError> {
    let body = check_header(bytes, EDGES_MAGIC, "edges")?;
    if body.len() < 4 {
        return Err(StorageError::Corrupt("short-id edges length".to_owned()));
    }
    let count = read_u32(&body[..4], "edges")? as usize;
    let list = &body[4..];
    if count.checked_mul(4) != Some(list.len()) {
        return Err(StorageError::Corrupt("short-id edges length".to_owned()));
    }
    list.chunks_exact(4)
        .map(|chunk| read_u32(chunk, "edges"))
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

    /// Allocate ids for `key` and every target, then store `key`'s immutable
    /// edge list, all in one transaction. Returns `(key id, sorted target ids)`.
    ///
    /// Targets are sorted and de-duplicated. A repeat with the same set is a
    /// no-op; a different set for an existing key is a collision error, because
    /// the edge list is immutable once written.
    ///
    /// # Errors
    /// As [`Self::get_or_create`], plus a collision error when `key` already has
    /// a different edge list.
    pub fn record_edges(
        &self,
        db: &SharedDatabase,
        key: &[u8],
        targets: &[&[u8]],
    ) -> Result<(u32, Vec<u32>), StorageError> {
        let mut keys: Vec<&[u8]> = Vec::with_capacity(targets.len().saturating_add(1));
        keys.push(key);
        keys.extend_from_slice(targets);
        let (ids, _) = self.write_with_retry(db, &keys, Some(0))?;
        let mut target_ids: Vec<u32> = ids[1..].to_vec();
        target_ids.sort_unstable();
        target_ids.dedup();
        Ok((ids[0], target_ids))
    }

    /// The edge list of `short_id`, or `None` if none was recorded.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    pub fn edges(
        &self,
        db: &SharedDatabase,
        short_id: u32,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        let txn = db.begin_transaction();
        let (records, _) = txn.get_with_record_versions(
            self.pool,
            &self.collection_id,
            &[indexed_id(EDGES_PREFIX, short_id)],
        )?;
        records
            .into_iter()
            .next()
            .flatten()
            .map(|record| decode_edges(&record.bytes))
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
    /// record whose key maps forward to the same id, and every edge list decodes
    /// with targets that resolve.
    ///
    /// # Errors
    /// Returns an error only if a record cannot be read; invariant violations
    /// are reported in the returned report.
    pub fn verify(&self, db: &SharedDatabase) -> Result<ShortIdVerifyReport, StorageError> {
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
            if let Some(edges) = get(indexed_id(EDGES_PREFIX, short_id))? {
                report.edge_lists_checked = report.edge_lists_checked.saturating_add(1);
                for target in decode_edges(&edges.bytes)? {
                    if target == 0 || target >= next {
                        report.problems.push(format!(
                            "id {short_id}: edge target {target} is out of range"
                        ));
                    } else if get(indexed_id(REVERSE_PREFIX, target))?.is_none() {
                        report
                            .problems
                            .push(format!("id {short_id}: edge target {target} unresolved"));
                    }
                }
            }
        }
        Ok(report)
    }

    /// `edge_owner` is the index in `keys` whose edge list is written from the
    /// remaining keys (`Some(0)`), or `None` for a plain allocation.
    fn write_with_retry(
        &self,
        db: &SharedDatabase,
        keys: &[&[u8]],
        edge_owner: Option<usize>,
    ) -> Result<(Vec<u32>, Option<Vec<u32>>), StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            match self.attempt(&txn, keys, edge_owner) {
                Ok(outcome) if !outcome.commit_needed => return Ok((outcome.ids, outcome.edges)),
                Ok(outcome) => match txn.commit() {
                    Ok(()) => return Ok((outcome.ids, outcome.edges)),
                    Err(error) if error.is_stale_read() => last = Some(error),
                    Err(error) => return Err(error),
                },
                Err(error) if error.is_stale_read() => last = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("short-id retry budget".to_owned())))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one read, validate and stage pass keeps the CAS token discipline in one place"
    )]
    fn attempt(
        &self,
        txn: &DatabaseTransaction<'_>,
        keys: &[&[u8]],
        edge_owner: Option<usize>,
    ) -> Result<Attempt, StorageError> {
        // Read the counter and every forward record (plus the owner's edge
        // record) together, before staging anything: staged records report
        // token 0, and the tokens must match the data they guard.
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

        let mut ids: Vec<u32> = Vec::with_capacity(keys.len());
        let mut fresh: HashMap<&[u8], u32> = HashMap::new();
        let mut staged: Vec<(NodeId, Vec<u8>, Option<u64>)> = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            let slot = index.saturating_add(1);
            if let Some(record) = &records[slot] {
                let (short_id, stored_key) = decode_forward(&record.bytes)?;
                if stored_key != *key {
                    return Err(StorageError::Collision(
                        "short-id key hash collision between different keys".to_owned(),
                    ));
                }
                ids.push(short_id);
            } else if let Some(short_id) = fresh.get(key) {
                ids.push(*short_id);
            } else {
                if next > SHORT_ID_MAX {
                    return Err(StorageError::Internal(
                        "short-id space exhausted for this scope".to_owned(),
                    ));
                }
                let short_id = next;
                next = next.saturating_add(1);
                fresh.insert(key, short_id);
                ids.push(short_id);
                let forward = forward_id(key);
                staged.push((forward, encode_forward(short_id, key), Some(tokens[slot])));
                staged.push((
                    indexed_id(REVERSE_PREFIX, short_id),
                    encode_reverse(key),
                    None,
                ));
            }
        }

        let mut edges = None;
        if let Some(owner) = edge_owner {
            let mut targets: Vec<u32> = ids
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != owner)
                .map(|(_, id)| *id)
                .collect();
            targets.sort_unstable();
            targets.dedup();
            let edge_id = indexed_id(EDGES_PREFIX, ids[owner]);
            let encoded = encode_edges(&targets)?;
            // Edge records are addressed by the owner's id, which may have been
            // allocated just now: only read an existing record when it was not.
            if fresh.contains_key(keys[owner]) {
                staged.push((edge_id, encoded, None));
            } else {
                let (existing, edge_tokens) =
                    txn.get_with_record_versions(self.pool, &self.collection_id, &[edge_id])?;
                match existing.into_iter().next().flatten() {
                    Some(record) if record.bytes.as_ref() == encoded.as_slice() => {}
                    Some(_) => {
                        return Err(StorageError::Collision(
                            "short-id edge list differs from the recorded one".to_owned(),
                        ));
                    }
                    None => staged.push((edge_id, encoded, Some(edge_tokens[0]))),
                }
            }
            edges = Some(targets);
        }

        if staged.is_empty() {
            return Ok(Attempt {
                ids,
                edges,
                commit_needed: false,
            });
        }
        if !fresh.is_empty() {
            txn.expect_record_version(self.pool, self.collection_id, COUNTER_ID, counter_token)?;
            txn.put(
                self.pool,
                self.collection_id,
                COUNTER_ID,
                &NodeData::new(Bytes::from(encode_counter(next))),
            )?;
        }
        for (id, payload, token) in staged {
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
            ids,
            edges,
            commit_needed: true,
        })
    }
}

struct Attempt {
    ids: Vec<u32>,
    edges: Option<Vec<u32>>,
    commit_needed: bool,
}
