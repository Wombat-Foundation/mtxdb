//! Persisted, immutable closure generations published through a
//! [`LogicalHead`].
//!
//! A *closure* is a derived per-id blob (for auth chains, a serialized roaring
//! bitmap of an event's transitive auth ancestors) keyed by a room-local `u32`
//! short id (see [`crate::short_id`]). The blob encoding is the caller's; this
//! module stores opaque bytes and needs no bitmap dependency.
//!
//! Layout, per scope (for example a room):
//!
//! - one **head collection** holding the [`LogicalHead`] (`MTXD-CLS-HEAD-v1`)
//!   and a generation counter (`CLSG`);
//! - one **generation collection** per generation holding `short_id -> blob`.
//!
//! A generation gets its own collection because the transaction layer has no
//! per-record delete: retiring a superseded generation is one
//! `delete_collection`, and a scope purge is the head collection plus every
//! generation collection, staged into the caller's transaction so it is atomic
//! with purging the short-id scope.
//!
//! Rebuild protocol: [`ClosureStore::begin`] reserves a never-reused generation
//! number (a CAS on the counter) and records the head token it started from;
//! [`GenerationBuilder::add`] writes closures in bounded batches under the new,
//! still unpublished generation; [`GenerationBuilder::publish`] swaps the head
//! by CAS. Readers never see a partial generation: until the swap they read the
//! old one. If another builder published first, `publish` fails with
//! `StorageError::StaleRead` and the caller discards its work with
//! [`GenerationBuilder::abandon`]. A crash before the swap leaves an orphan
//! generation collection that [`ClosureStore::retire_superseded`] reclaims.
//!
//! The head also records the short-id counter the generation was built against
//! (`source_next`): closures cover ids `1..source_next`. Verifying a closure's
//! *content* against the direct edges is the adapter's job; [`ClosureStore::verify`]
//! checks the storage invariants (head resolves, every covered id has a record,
//! counts match).
//!
//! Gated on `multi-reader` with the transaction layer and engine CAS it uses.

use bytes::Bytes;

use crate::database::{DatabaseTransaction, SharedDatabase};
use crate::layout::ShardType;
use crate::logical_head::{LogicalHead, LogicalHeadValue};
use crate::storage::{NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Wire version of closure records and the generation counter.
pub const CLOSURE_FORMAT_VERSION: u8 = 1;

const HEAD_LOGICAL_ID: NodeId = *b"MTXD-CLS-HEAD-v1";
const COUNTER_ID: NodeId = *b"MTXD-CLS-NEXTG-1";
const RECORD_MAGIC: [u8; 4] = *b"CLSR";
const COUNTER_MAGIC: [u8; 4] = *b"CLSG";
const RECORD_PREFIX: [u8; 8] = *b"MTXCLSR\0";
const HEAD_METADATA_LEN: usize = 8 + 8 + 4 + 4;
const MAX_ATTEMPTS: usize = 64;

/// Test-only crash injection: abort the process (no destructors, no flush) when
/// `MTXDB_CRASH_AT` names this point, so recovery tests can kill a real child
/// process at an exact step.
#[cfg(test)]
fn crash_point(name: &str) {
    if std::env::var("MTXDB_CRASH_AT").as_deref() == Ok(name) {
        std::process::abort();
    }
}

#[cfg(not(test))]
#[inline(always)]
fn crash_point(_name: &str) {}

/// The published generation and what it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClosureHead {
    /// Active generation number.
    pub generation: u64,
    /// The generation it superseded (`0` for the first). Kept readable so a
    /// reader that resolved the old head can finish.
    pub previous: u64,
    /// Short-id counter the generation was built against; covers `1..source_next`.
    pub source_next: u32,
    /// Number of closure records in the generation.
    pub count: u32,
}

/// Result of [`ClosureStore::verify`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureVerifyReport {
    /// Closure records found for covered ids.
    pub records_checked: u32,
    /// Violations; empty means consistent.
    pub problems: Vec<String>,
}

impl ClosureVerifyReport {
    /// Whether every invariant held.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.problems.is_empty()
    }
}

fn record_id(short_id: u32) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&RECORD_PREFIX);
    id[8..12].copy_from_slice(&short_id.to_be_bytes());
    id
}

fn encode_record(blob: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blob.len().saturating_add(5));
    out.extend_from_slice(&RECORD_MAGIC);
    out.push(CLOSURE_FORMAT_VERSION);
    out.extend_from_slice(blob);
    out
}

fn decode_record(bytes: &[u8]) -> Result<Vec<u8>, StorageError> {
    if bytes.len() < 5 || bytes[..4] != RECORD_MAGIC || bytes[4] != CLOSURE_FORMAT_VERSION {
        return Err(StorageError::Corrupt("closure record header".to_owned()));
    }
    Ok(bytes[5..].to_vec())
}

fn encode_counter(next: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(13);
    out.extend_from_slice(&COUNTER_MAGIC);
    out.push(CLOSURE_FORMAT_VERSION);
    out.extend_from_slice(&next.to_be_bytes());
    out
}

fn decode_counter(bytes: &[u8]) -> Result<u64, StorageError> {
    if bytes.len() != 13 || bytes[..4] != COUNTER_MAGIC || bytes[4] != CLOSURE_FORMAT_VERSION {
        return Err(StorageError::Corrupt(
            "closure generation counter".to_owned(),
        ));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[5..13]);
    Ok(u64::from_be_bytes(raw))
}

fn encode_head_metadata(head: &ClosureHead) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_METADATA_LEN);
    out.extend_from_slice(&head.generation.to_be_bytes());
    out.extend_from_slice(&head.previous.to_be_bytes());
    out.extend_from_slice(&head.source_next.to_be_bytes());
    out.extend_from_slice(&head.count.to_be_bytes());
    out
}

fn decode_head(
    value: &LogicalHeadValue,
    store: &ClosureStore,
) -> Result<ClosureHead, StorageError> {
    let bytes = &value.metadata;
    if bytes.len() != HEAD_METADATA_LEN {
        return Err(StorageError::Corrupt("closure head metadata".to_owned()));
    }
    let word = |range: std::ops::Range<usize>| -> Result<[u8; 8], StorageError> {
        <[u8; 8]>::try_from(&bytes[range])
            .map_err(|_| StorageError::Corrupt("closure head metadata".to_owned()))
    };
    let head = ClosureHead {
        generation: u64::from_be_bytes(word(0..8)?),
        previous: u64::from_be_bytes(word(8..16)?),
        source_next: u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        count: u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
    };
    if value.target != store.generation_collection(head.generation) {
        return Err(StorageError::Corrupt(
            "closure head target does not name its generation".to_owned(),
        ));
    }
    Ok(head)
}

/// Closure generations for one scope.
#[derive(Debug, Clone, Copy)]
pub struct ClosureStore {
    pool: ShardType,
    scope: [u8; 16],
}

impl ClosureStore {
    /// Address the closure store of `scope` (the same scope id used by the
    /// matching [`crate::short_id::ShortIdIndex`] collection is a good choice).
    #[must_use]
    pub fn new(pool: ShardType, scope: [u8; 16]) -> Self {
        Self { pool, scope }
    }

    fn head_collection(&self) -> [u8; 16] {
        let mut canonical = b"closure-head:".to_vec();
        canonical.extend_from_slice(&self.scope);
        derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
    }

    fn generation_collection(&self, generation: u64) -> [u8; 16] {
        let mut canonical = b"closure-gen:".to_vec();
        canonical.extend_from_slice(&self.scope);
        canonical.extend_from_slice(&generation.to_be_bytes());
        derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
    }

    fn heads(&self) -> LogicalHead {
        LogicalHead::new(self.pool, self.head_collection())
    }

    fn read_head(
        &self,
        txn: &DatabaseTransaction<'_>,
    ) -> Result<(Option<ClosureHead>, u64), StorageError> {
        let read = self.heads().read(txn, &HEAD_LOGICAL_ID)?;
        let head = read
            .value
            .map(|value| decode_head(&value, self))
            .transpose()?;
        Ok((head, read.token))
    }

    /// The published generation, if any.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt head.
    pub fn head(&self, db: &SharedDatabase) -> Result<Option<ClosureHead>, StorageError> {
        self.read_head(&db.begin_transaction())
            .map(|(head, _)| head)
    }

    /// The closure blob of `short_id` in the published generation.
    ///
    /// # Errors
    /// As [`Self::get_many`].
    pub fn get(&self, db: &SharedDatabase, short_id: u32) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self
            .get_many(db, &[short_id])?
            .1
            .into_iter()
            .next()
            .flatten())
    }

    /// Closure blobs for several ids, all read from **one** generation: the head
    /// is resolved once, so the result can never mix two generations. Returns the
    /// generation read (`None` if nothing is published) and the blobs in order.
    ///
    /// If the generation is retired between reading the head and its records,
    /// the head is re-read and the read retried, so a concurrent republish never
    /// surfaces as a false miss.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt record.
    #[allow(clippy::type_complexity, reason = "generation plus ordered blobs")]
    pub fn get_many(
        &self,
        db: &SharedDatabase,
        short_ids: &[u32],
    ) -> Result<(Option<u64>, Vec<Option<Vec<u8>>>), StorageError> {
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            let (Some(head), _) = self.read_head(&txn)? else {
                return Ok((None, vec![None; short_ids.len()]));
            };
            let ids: Vec<NodeId> = short_ids.iter().map(|id| record_id(*id)).collect();
            let (records, _) = txn.get_with_record_versions(
                self.pool,
                &self.generation_collection(head.generation),
                &ids,
            )?;
            let blobs = records
                .into_iter()
                .map(|record| record.map(|data| decode_record(&data.bytes)).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            if blobs.iter().all(Option::is_some) {
                return Ok((Some(head.generation), blobs));
            }
            // Some absent: only a true miss if the head did not move underneath us.
            let (again, _) = self.read_head(&db.begin_transaction())?;
            if again.map(|h| h.generation) == Some(head.generation) {
                return Ok((Some(head.generation), blobs));
            }
        }
        Err(StorageError::Internal(
            "closure head kept moving during a read".to_owned(),
        ))
    }

    /// Start building a new generation. Reserves its number so two builders
    /// can never write the same generation collection.
    ///
    /// # Errors
    /// Returns an error on a read or commit failure, or a corrupt counter.
    pub fn begin(&self, db: &SharedDatabase) -> Result<GenerationBuilder, StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            let (records, tokens) =
                txn.get_with_record_versions(self.pool, &self.head_collection(), &[COUNTER_ID])?;
            let generation = match records.into_iter().next().flatten() {
                Some(record) => decode_counter(&record.bytes)?,
                None => 1,
            };
            let next = generation
                .checked_add(1)
                .ok_or_else(|| StorageError::Internal("closure generation overflow".to_owned()))?;
            // Capture the head token before reserving, so a publish from a
            // builder that started earlier invalidates this one.
            let (base, head_token) = self.read_head(&txn)?;
            txn.expect_record_version(self.pool, self.head_collection(), COUNTER_ID, tokens[0])?;
            txn.put(
                self.pool,
                self.head_collection(),
                COUNTER_ID,
                &NodeData::new(Bytes::from(encode_counter(next))),
            )?;
            match txn.commit() {
                Ok(()) => {
                    return Ok(GenerationBuilder {
                        store: *self,
                        generation,
                        base,
                        head_token,
                        count: 0,
                    });
                }
                Err(error) if error.is_stale_read() => last = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("closure begin retries".to_owned())))
    }

    /// Check the storage invariants of the published generation.
    ///
    /// # Errors
    /// Returns an error only when a record cannot be read.
    pub fn verify(&self, db: &SharedDatabase) -> Result<ClosureVerifyReport, StorageError> {
        let mut report = ClosureVerifyReport::default();
        let Some(head) = self.head(db)? else {
            return Ok(report);
        };
        let txn = db.begin_transaction();
        let collection = self.generation_collection(head.generation);
        for short_id in 1..head.source_next {
            let (records, _) =
                txn.get_with_record_versions(self.pool, &collection, &[record_id(short_id)])?;
            match records.into_iter().next().flatten() {
                Some(record) => {
                    decode_record(&record.bytes)?;
                    report.records_checked = report.records_checked.saturating_add(1);
                }
                None => report
                    .problems
                    .push(format!("covered id {short_id} has no closure record")),
            }
        }
        if report.records_checked != head.count {
            report.problems.push(format!(
                "head counts {} closures but {} were found",
                head.count, report.records_checked
            ));
        }
        Ok(report)
    }

    /// Delete generations other than the published one and its predecessor, and
    /// any orphan left by a crashed or abandoned build. Returns how many
    /// generation collections were removed.
    ///
    /// # Errors
    /// Returns an error on a read or commit failure.
    pub fn retire_superseded(&self, db: &SharedDatabase) -> Result<u64, StorageError> {
        let txn = db.begin_transaction();
        let (head, _) = self.read_head(&txn)?;
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.head_collection(), &[COUNTER_ID])?;
        let Some(record) = records.into_iter().next().flatten() else {
            return Ok(0);
        };
        let next = decode_counter(&record.bytes)?;
        let keep = |generation: u64| {
            head.is_some_and(|head| generation == head.generation || generation == head.previous)
        };
        let mut removed = 0u64;
        for generation in 1..next {
            if !keep(generation) {
                txn.delete_collection(self.pool, self.generation_collection(generation))?;
                removed = removed.saturating_add(1);
            }
        }
        crash_point("retire-before-commit");
        txn.commit()?;
        crash_point("retire-after-commit");
        Ok(removed)
    }

    /// Read one record straight from a generation's collection, bypassing the
    /// head, so tests can prove exactly how far a crashed step got.
    #[cfg(test)]
    pub(crate) fn raw_generation_record(
        &self,
        db: &SharedDatabase,
        generation: u64,
        short_id: u32,
    ) -> Option<Vec<u8>> {
        let txn = db.begin_transaction();
        let (records, _) = txn
            .get_with_record_versions(
                self.pool,
                &self.generation_collection(generation),
                &[record_id(short_id)],
            )
            .unwrap();
        records
            .into_iter()
            .next()
            .flatten()
            .map(|record| decode_record(&record.bytes).unwrap())
    }

    /// Stage removal of the head and every generation collection of this scope
    /// into `txn`, so a scope purge can commit atomically with other deletes
    /// (for example the short-id collection).
    ///
    /// # Errors
    /// Returns an error if the counter cannot be read or a delete cannot be
    /// staged.
    pub fn stage_purge(&self, txn: &DatabaseTransaction<'_>) -> Result<(), StorageError> {
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.head_collection(), &[COUNTER_ID])?;
        if let Some(record) = records.into_iter().next().flatten() {
            for generation in 1..decode_counter(&record.bytes)? {
                txn.delete_collection(self.pool, self.generation_collection(generation))?;
            }
        }
        txn.delete_collection(self.pool, self.head_collection())?;
        Ok(())
    }
}

/// An unpublished generation under construction.
pub struct GenerationBuilder {
    store: ClosureStore,
    generation: u64,
    base: Option<ClosureHead>,
    head_token: u64,
    count: u32,
}

impl GenerationBuilder {
    /// The reserved generation number.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Write one batch of `(short_id, blob)` closures in a single transaction.
    /// Nothing is visible to readers until [`Self::publish`]. Re-adding an id
    /// overwrites it, so a batch can be retried.
    ///
    /// # Errors
    /// Returns an error on a stage or commit failure, or an empty blob (an
    /// empty value would be a tombstone).
    pub fn add(
        &mut self,
        db: &SharedDatabase,
        closures: &[(u32, &[u8])],
    ) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        let collection = self.store.generation_collection(self.generation);
        for (short_id, blob) in closures {
            if blob.is_empty() {
                return Err(StorageError::Internal(
                    "closure blob must not be empty".to_owned(),
                ));
            }
            txn.put(
                self.store.pool,
                collection,
                record_id(*short_id),
                &NodeData::new(Bytes::from(encode_record(blob))),
            )?;
        }
        txn.commit()?;
        crash_point("after-add");
        let added = u32::try_from(closures.len())
            .map_err(|_| StorageError::Internal("closure batch too large".to_owned()))?;
        self.count = self
            .count
            .checked_add(added)
            .ok_or_else(|| StorageError::Internal("closure count overflow".to_owned()))?;
        Ok(())
    }

    /// Atomically publish this generation as the head, covering ids
    /// `1..source_next`. Fails with `StorageError::StaleRead` if the head moved
    /// since [`ClosureStore::begin`].
    ///
    /// # Errors
    /// `StaleRead` on a lost race; otherwise a commit failure.
    pub fn publish(
        self,
        db: &SharedDatabase,
        source_next: u32,
    ) -> Result<ClosureHead, StorageError> {
        crash_point("before-publish");
        let head = ClosureHead {
            generation: self.generation,
            previous: self.base.map_or(0, |base| base.generation),
            source_next,
            count: self.count,
        };
        let value = LogicalHeadValue::new(
            self.store.generation_collection(self.generation),
            encode_head_metadata(&head),
        );
        let txn = db.begin_transaction();
        self.store
            .heads()
            .stage_replace(&txn, &HEAD_LOGICAL_ID, self.head_token, &value)?;
        crash_point("publish-before-commit");
        txn.commit()?;
        crash_point("publish-after-commit");
        Ok(head)
    }

    /// Discard an unpublished generation.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be committed.
    pub fn abandon(self, db: &SharedDatabase) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        txn.delete_collection(
            self.store.pool,
            self.store.generation_collection(self.generation),
        )?;
        txn.commit()
    }
}
