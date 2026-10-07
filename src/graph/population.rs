//! A room's reconciliation population, read from the owner log and its
//! compacted runs.
//!
//! The owner log holds one entry per event the room's holder owns, each with
//! the 24-byte payload from [`encode_owner_payload`]. Compaction folds the
//! log into immutable sorted *runs* and publishes them under a *manifest*; the
//! counter record carries `(log_epoch, log_epoch_start_seq, manifest_version)`
//! together, so one read yields a manifest and the log tail it pairs with.
//!
//! A [`PopulationSnapshot`] is a cached, shared base (the runs merged into a
//! sorted population) plus a small tail (the log entries since the last
//! compaction), pinned at an `owner_seq_ceiling`. It is materialized in
//! memory, so an exchange keeps answering from one population while writers
//! append and compaction runs.
//!
//! Layout, all in the scope's pool:
//! - manifest `v`: a record in the scope's collection, immutable once
//!   written, holding the kernel at the folded watermark and the run list;
//! - one collection per run, holding sorted [`RUN_CHUNK_ENTRIES`]-entry chunk
//!   records, so a retired run is dropped with one `delete_collection`;
//! - a mutable garbage list of collections awaiting deletion after a grace
//!   period, which [`ShortIdIndex::collect_garbage`] works through.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use bytes::Bytes;
use parking_lot::Mutex;
use rezzy_recon::triage::NodeSummary;
use rezzy_recon::{
    AlgebraicError, ElementHash, Population, ResidentKernel, SortedPopulation,
    RESIDENT_SERIALIZED_LEN,
};

use crate::database::{Database, DatabaseTransaction};
use crate::short_id::{
    check_header, derived_id, encode_counter, encode_owner_log, header, owner_log_record_id,
    read_u32, ScopeCounters, ShortIdIndex, COUNTER_ID, MAX_ATTEMPTS,
};
use crate::storage::{NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Length of an owner-log payload: `h64` then `h128`, both big-endian.
pub const OWNER_PAYLOAD_LEN: usize = 24;
/// Entries per run chunk record.
pub const RUN_CHUNK_ENTRIES: usize = 2000;
/// Log entries since the last compaction at which [`ShortIdIndex::compact`]
/// acts without being forced.
pub const COMPACT_TRIGGER: u32 = 2000;

/// Owner-log entries read per call: bounds the transient records held while
/// the snapshot materializes, so peak memory tracks the population and not the
/// log's record overhead.
const READ_CHUNK: u32 = 4096;
/// Chunk records written per transaction (24 MB, under the staging budget).
const CHUNKS_PER_TXN: usize = 500;
/// Chunk records read per call.
const CHUNKS_PER_READ: u32 = 50;

const MANIFEST_MAGIC: [u8; 4] = *b"SIDM";
const RUN_CHUNK_MAGIC: [u8; 4] = *b"SIDK";
const GC_MAGIC: [u8; 4] = *b"SIDG";
const MANIFEST_PREFIX: [u8; 8] = *b"MTXSIDM\0";
const RUN_CHUNK_PREFIX: [u8; 8] = *b"MTXSIDK\0";
const GC_ID: NodeId = *b"MTXD-SID-GCLS-v1";
const RUN_SCOPE: &[u8] = b"short-id-run";

/// One element as the runs store it, ordered `(h64, h128)`.
type Entry = (u64, u128);

/// The owner-log payload that adds `hash` to the population.
#[must_use]
pub fn encode_owner_payload(hash: ElementHash) -> [u8; OWNER_PAYLOAD_LEN] {
    let mut payload = [0_u8; OWNER_PAYLOAD_LEN];
    payload[..8].copy_from_slice(&hash.h64.to_be_bytes());
    payload[8..].copy_from_slice(&hash.h128.to_be_bytes());
    payload
}

fn decode_entry(what: &str, bytes: &[u8]) -> Result<Entry, StorageError> {
    let corrupt = || StorageError::Corrupt(format!("{what} entry is not 24 bytes"));
    let (h64, rest) = bytes.split_first_chunk::<8>().ok_or_else(corrupt)?;
    let h128 = <&[u8; 16]>::try_from(rest).map_err(|_| corrupt())?;
    Ok((u64::from_be_bytes(*h64), u128::from_be_bytes(*h128)))
}

fn element((h64, h128): Entry) -> ElementHash {
    ElementHash { h64, h128 }
}

fn algebraic(error: AlgebraicError) -> StorageError {
    StorageError::Internal(format!("population: {error:?}"))
}

#[cfg(test)]
pub(crate) fn run_chunk_id_for_test(index: u32) -> NodeId {
    derived_id(RUN_CHUNK_PREFIX, index, 0)
}

/// A run's identity. `token` is the counter record's version when the run was
/// built: two builds from the same counter state produce identical bytes, and
/// builds from different states cannot share a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunMeta {
    version: u32,
    token: u64,
    entries: u32,
    chunks: u32,
}

const RUN_META_LEN: usize = 20;

/// What manifest `version` publishes: the kernel as of `folded_through` and
/// the runs, oldest first, that hold every owner below it.
#[derive(Debug, Clone)]
struct Manifest {
    version: u32,
    folded_through: u32,
    kernel: ResidentKernel,
    runs: Vec<RunMeta>,
}

impl Manifest {
    fn empty() -> Self {
        Self {
            version: 0,
            folded_through: 1,
            kernel: ResidentKernel::new(),
            runs: Vec::new(),
        }
    }

    fn total_entries(&self) -> u64 {
        self.runs.iter().map(|run| u64::from(run.entries)).sum()
    }

    fn encode(&self) -> Result<Vec<u8>, StorageError> {
        let mut out = header(MANIFEST_MAGIC);
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.folded_through.to_be_bytes());
        out.extend_from_slice(&self.kernel.to_bytes());
        let count = u32::try_from(self.runs.len())
            .map_err(|_| StorageError::Internal("manifest run count".to_owned()))?;
        out.extend_from_slice(&count.to_be_bytes());
        for run in &self.runs {
            out.extend_from_slice(&run.version.to_be_bytes());
            out.extend_from_slice(&run.token.to_be_bytes());
            out.extend_from_slice(&run.entries.to_be_bytes());
            out.extend_from_slice(&run.chunks.to_be_bytes());
        }
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let corrupt = || StorageError::Corrupt("short-id manifest length".to_owned());
        let body = check_header(bytes, MANIFEST_MAGIC, "manifest")?;
        let kernel_end = 8_usize.saturating_add(RESIDENT_SERIALIZED_LEN);
        let fixed = kernel_end.saturating_add(4);
        if body.len() < fixed {
            return Err(corrupt());
        }
        let version = read_u32(&body[..4], "manifest")?;
        let folded_through = read_u32(&body[4..8], "manifest")?;
        let kernel = ResidentKernel::from_bytes(&body[8..kernel_end])
            .map_err(|_| StorageError::Corrupt("short-id manifest kernel".to_owned()))?;
        let count = read_u32(&body[kernel_end..fixed], "manifest")? as usize;
        let list = &body[fixed..];
        if count.checked_mul(RUN_META_LEN) != Some(list.len()) {
            return Err(corrupt());
        }
        let runs = list
            .as_chunks::<RUN_META_LEN>()
            .0
            .iter()
            .map(|c| {
                Ok(RunMeta {
                    version: read_u32(&c[..4], "manifest")?,
                    token: u64::from_be_bytes(c[4..12].try_into().map_err(|_| corrupt())?),
                    entries: read_u32(&c[12..16], "manifest")?,
                    chunks: read_u32(&c[16..20], "manifest")?,
                })
            })
            .collect::<Result<Vec<_>, StorageError>>()?;
        Ok(Self {
            version,
            folded_through,
            kernel,
            runs,
        })
    }
}

fn encode_chunk(entries: &[Entry]) -> Result<Vec<u8>, StorageError> {
    let mut out = header(RUN_CHUNK_MAGIC);
    let count = u32::try_from(entries.len())
        .map_err(|_| StorageError::Internal("run chunk length".to_owned()))?;
    out.extend_from_slice(&count.to_be_bytes());
    for &(h64, h128) in entries {
        out.extend_from_slice(&h64.to_be_bytes());
        out.extend_from_slice(&h128.to_be_bytes());
    }
    Ok(out)
}

fn decode_chunk(bytes: &[u8]) -> Result<Vec<Entry>, StorageError> {
    let body = check_header(bytes, RUN_CHUNK_MAGIC, "run chunk")?;
    if body.len() < 4 {
        return Err(StorageError::Corrupt(
            "short-id run chunk length".to_owned(),
        ));
    }
    let count = read_u32(&body[..4], "run chunk")? as usize;
    let entries = &body[4..];
    if count.checked_mul(OWNER_PAYLOAD_LEN) != Some(entries.len()) {
        return Err(StorageError::Corrupt(
            "short-id run chunk length".to_owned(),
        ));
    }
    entries
        .as_chunks::<OWNER_PAYLOAD_LEN>()
        .0
        .iter()
        .map(|entry| decode_entry("run chunk", entry))
        .collect()
}

fn encode_gc(list: &[([u8; 16], u64)]) -> Result<Vec<u8>, StorageError> {
    let mut out = header(GC_MAGIC);
    let count = u32::try_from(list.len())
        .map_err(|_| StorageError::Internal("garbage list length".to_owned()))?;
    out.extend_from_slice(&count.to_be_bytes());
    for (collection, deadline) in list {
        out.extend_from_slice(collection);
        out.extend_from_slice(&deadline.to_be_bytes());
    }
    Ok(out)
}

fn decode_gc(bytes: &[u8]) -> Result<Vec<([u8; 16], u64)>, StorageError> {
    let corrupt = || StorageError::Corrupt("short-id garbage list length".to_owned());
    let body = check_header(bytes, GC_MAGIC, "garbage list")?;
    if body.len() < 4 {
        return Err(corrupt());
    }
    let count = read_u32(&body[..4], "garbage list")? as usize;
    let list = &body[4..];
    if count.checked_mul(24) != Some(list.len()) {
        return Err(corrupt());
    }
    list.as_chunks::<24>()
        .0
        .iter()
        .map(|c| {
            Ok((
                <[u8; 16]>::try_from(&c[..16]).map_err(|_| corrupt())?,
                u64::from_be_bytes(c[16..].try_into().map_err(|_| corrupt())?),
            ))
        })
        .collect()
}

/// K-way merge of sorted sequences into one sorted sequence.
fn merge_sorted(sequences: &[&[Entry]]) -> Vec<Entry> {
    let total = sequences.iter().map(|sequence| sequence.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut heap: BinaryHeap<Reverse<(Entry, usize)>> = BinaryHeap::new();
    let mut positions = vec![0_usize; sequences.len()];
    for (index, sequence) in sequences.iter().enumerate() {
        if let Some(&first) = sequence.first() {
            heap.push(Reverse((first, index)));
        }
    }
    while let Some(Reverse((entry, index))) = heap.pop() {
        out.push(entry);
        positions[index] = positions[index].saturating_add(1);
        if let Some(&next) = sequences[index].get(positions[index]) {
            heap.push(Reverse((next, index)));
        }
    }
    out
}

/// The population folded into runs, shared between snapshots.
#[derive(Debug)]
pub struct PopulationBase {
    manifest_version: u32,
    kernel: ResidentKernel,
    population: SortedPopulation,
}

impl PopulationBase {
    /// The manifest this base was merged from (0: no runs).
    #[must_use]
    pub fn manifest_version(&self) -> u32 {
        self.manifest_version
    }

    /// Number of elements folded into runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.population.len()
    }

    /// Whether no element has been folded into a run.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.population.is_empty()
    }
}

/// A population pinned at one point in the owner log: a shared base plus the
/// tail of log entries since the base's compaction.
#[derive(Debug, Clone)]
pub struct PopulationSnapshot {
    base: Arc<PopulationBase>,
    tail: SortedPopulation,
    owner_seq_ceiling: u32,
}

impl PopulationSnapshot {
    /// The compacted-run manifest this snapshot covers (0: no runs).
    #[must_use]
    pub fn manifest_version(&self) -> u32 {
        self.base.manifest_version
    }

    /// The owner-log sequence number the snapshot stops before: it holds
    /// every owner with `seq < owner_seq_ceiling` and none after.
    #[must_use]
    pub fn owner_seq_ceiling(&self) -> u32 {
        self.owner_seq_ceiling
    }

    /// Number of elements in the population.
    #[must_use]
    pub fn len(&self) -> usize {
        self.base.len().saturating_add(self.tail.len())
    }

    /// Whether the population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The room kernel at the ceiling: the manifest's kernel plus the tail.
    ///
    /// # Errors
    /// Returns an error if the element count overflows.
    pub fn kernel(&self) -> Result<ResidentKernel, StorageError> {
        let mut kernel = self.base.kernel.clone();
        for (&h64, &h128) in self.tail.h64s().iter().zip(self.tail.h128s()) {
            kernel
                .insert(ElementHash { h64, h128 })
                .map_err(algebraic)?;
        }
        Ok(kernel)
    }
}

impl Population for PopulationSnapshot {
    fn node_summary(&self, node: (u8, u64)) -> Result<NodeSummary, AlgebraicError> {
        let base = self.base.population.node_summary(node)?;
        let tail = self.tail.node_summary(node)?;
        Ok(NodeSummary {
            count: base
                .count
                .checked_add(tail.count)
                .ok_or(AlgebraicError::CountOverflow)?,
            digest: base.digest ^ tail.digest,
        })
    }

    fn for_each_h64_in(&self, node: (u8, u64), f: &mut dyn FnMut(u64)) {
        self.base.population.for_each_h64_in(node, f);
        self.tail.for_each_h64_in(node, f);
    }

    fn candidates_into(&self, root: u64, out: &mut Vec<u128>) {
        self.base.population.candidates_into(root, out);
        self.tail.candidates_into(root, out);
    }
}

/// The counters a snapshot is built from, read at one instant.
#[derive(Debug, Clone, Copy)]
pub struct PopulationPin {
    counters: ScopeCounters,
}

impl PopulationPin {
    /// The counters the pin was read from.
    #[must_use]
    pub fn counters(&self) -> ScopeCounters {
        self.counters
    }
}

/// Every run collection a write was started for, as `(scope, run)`.
#[cfg(test)]
pub(crate) static WRITTEN_RUNS: Mutex<Vec<([u8; 16], [u8; 16])>> = Mutex::new(Vec::new());

/// Runs once, on the compacting thread, between writing a run and publishing it.
#[cfg(test)]
type PublishHook = Box<dyn FnMut()>;

#[cfg(test)]
thread_local! {
    pub(crate) static BEFORE_PUBLISH: std::cell::RefCell<Option<PublishHook>> =
        const { std::cell::RefCell::new(None) };
}

type CacheKey = ([u8; 16], u32);

/// A small LRU of merged bases, keyed by scope and manifest version, so a new
/// snapshot costs a tail read and not a merge.
#[derive(Debug)]
pub struct PopulationCache {
    capacity: usize,
    entries: Mutex<VecDeque<(CacheKey, Arc<PopulationBase>)>>,
}

impl PopulationCache {
    /// A cache holding at most `capacity` bases.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Mutex::new(VecDeque::new()),
        }
    }

    /// Number of cached bases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Whether nothing is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: CacheKey) -> Option<Arc<PopulationBase>> {
        let mut entries = self.entries.lock();
        let position = entries.iter().position(|(k, _)| *k == key)?;
        let entry = entries.remove(position)?;
        let base = Arc::clone(&entry.1);
        entries.push_front(entry);
        Some(base)
    }

    fn insert(&self, key: CacheKey, base: Arc<PopulationBase>) {
        let mut entries = self.entries.lock();
        entries.retain(|(k, _)| *k != key);
        entries.push_front((key, base));
        entries.truncate(self.capacity);
    }
}

/// What the final publish transaction of a compaction needs.
struct PublishPlan {
    counters: ScopeCounters,
    counter_token: u64,
    version: u32,
    run_collection: [u8; 16],
    /// The encoded manifest for `version`. It is written by the publish
    /// transaction itself: two compactions that prepared from different
    /// counter states would otherwise overwrite each other's record.
    manifest: Vec<u8>,
    retired: Vec<[u8; 16]>,
    now_ms: u64,
    grace_ms: u64,
}

/// How one publish transaction ended.
enum Published {
    Done(usize),
    /// Another compaction moved the counter's epoch, start or manifest; the
    /// prepared run is stale and the compaction starts over.
    Superseded,
}

/// What one compaction did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionReport {
    /// The manifest version published.
    pub manifest_version: u32,
    /// Owner-log entries folded into the new run.
    pub folded_entries: u32,
    /// Entries in the run written (larger than `folded_entries` when older
    /// runs were merged into it).
    pub run_entries: u32,
    /// Collections queued for deletion after the grace period.
    pub retired_collections: usize,
}

/// What one garbage-collection pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GcReport {
    /// Collections deleted.
    pub dropped: usize,
    /// Collections still inside their grace period.
    pub pending: usize,
}

impl ShortIdIndex {
    fn run_collection(&self, run: &RunMeta) -> [u8; 16] {
        let mut name = Vec::with_capacity(RUN_SCOPE.len().saturating_add(40));
        name.extend_from_slice(RUN_SCOPE);
        name.extend_from_slice(&self.collection_id);
        name.extend_from_slice(&run.version.to_be_bytes());
        name.extend_from_slice(&run.token.to_be_bytes());
        derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &name)
    }

    fn read_record(
        &self,
        txn: &DatabaseTransaction<'_>,
        collection: &[u8; 16],
        id: NodeId,
    ) -> Result<(Option<NodeData>, u64), StorageError> {
        Self::read_record_in(self.pool, txn, collection, id)
    }

    fn read_record_in(
        pool: crate::layout::ShardType,
        txn: &DatabaseTransaction<'_>,
        collection: &[u8; 16],
        id: NodeId,
    ) -> Result<(Option<NodeData>, u64), StorageError> {
        let (records, tokens) = txn.get_with_record_versions(pool, collection, &[id])?;
        Ok((
            records.into_iter().next().flatten(),
            tokens.first().copied().unwrap_or(0),
        ))
    }

    fn read_manifest(
        &self,
        txn: &DatabaseTransaction<'_>,
        version: u32,
    ) -> Result<Manifest, StorageError> {
        if version == 0 {
            return Ok(Manifest::empty());
        }
        let (record, _) = self.read_record(
            txn,
            &self.collection_id,
            derived_id(MANIFEST_PREFIX, version, 0),
        )?;
        let record = record.ok_or(StorageError::NotFound(derived_id(
            MANIFEST_PREFIX,
            version,
            0,
        )))?;
        let manifest = Manifest::decode(&record.bytes)?;
        if manifest.version != version {
            return Err(StorageError::Corrupt(
                "short-id manifest version mismatch".to_owned(),
            ));
        }
        Ok(manifest)
    }

    /// Every entry of `run`, sorted. A run whose collection has been dropped
    /// reports an expired generation, not corruption.
    fn load_run(
        &self,
        txn: &DatabaseTransaction<'_>,
        run: &RunMeta,
    ) -> Result<Vec<Entry>, StorageError> {
        let collection = self.run_collection(run);
        let mut out = Vec::with_capacity(run.entries as usize);
        let mut chunk = 0_u32;
        while chunk < run.chunks {
            let end = chunk.saturating_add(CHUNKS_PER_READ).min(run.chunks);
            let ids: Vec<NodeId> = (chunk..end)
                .map(|i| derived_id(RUN_CHUNK_PREFIX, i, 0))
                .collect();
            let (records, _) = txn.get_with_record_versions(self.log_pool, &collection, &ids)?;
            for record in records {
                let record = record.ok_or(StorageError::StaleGeneration {
                    generation: u64::from(run.version),
                    current: None,
                })?;
                out.extend(decode_chunk(&record.bytes)?);
            }
            chunk = end;
        }
        if out.len() != run.entries as usize {
            return Err(StorageError::Corrupt(
                "short-id run entry count disagrees with its manifest".to_owned(),
            ));
        }
        Ok(out)
    }

    /// The owner-log entries `[from, to)` of generation `epoch`, sorted.
    fn read_tail(
        &self,
        db: &Database,
        epoch: u32,
        from: u32,
        to: u32,
    ) -> Result<Vec<Entry>, StorageError> {
        // One transaction per chunk is fine: an entry below the ceiling was
        // committed with the counter that bounds it and is never rewritten, so
        // the chunks need no common snapshot.
        let mut entries = Vec::new();
        let mut cursor = from;
        while cursor < to {
            let end = cursor.saturating_add(READ_CHUNK).min(to);
            for entry in self.owner_log(db, epoch, cursor, end)? {
                entries.push(decode_entry("owner-log", &entry.payload)?);
            }
            cursor = end;
        }
        entries.sort_unstable();
        Ok(entries)
    }

    /// Read the counters once; a snapshot is built from this and nothing else.
    ///
    /// # Errors
    /// Returns an error if the counter is unreadable.
    pub fn pin_population(&self, db: &Database) -> Result<PopulationPin, StorageError> {
        Ok(PopulationPin {
            counters: self.counters(db)?,
        })
    }

    /// Build the population a pin names, from the cache's base if it holds one.
    ///
    /// # Errors
    /// Returns [`StorageError::StaleGeneration`] if compaction and garbage
    /// collection have since dropped what the pin names (take a fresh pin),
    /// or another error if a record is unreadable or corrupt.
    pub fn materialize_population(
        &self,
        db: &Database,
        pin: &PopulationPin,
        cache: Option<&PopulationCache>,
    ) -> Result<PopulationSnapshot, StorageError> {
        let counters = pin.counters;
        let version = counters.manifest_version;
        let key = (self.collection_id, version);
        let cached = cache.and_then(|cache| cache.get(key));
        let base = if let Some(base) = cached {
            base
        } else {
            let base = Arc::new(
                self.load_base(db, version, counters.log_epoch_start_seq)
                    .map_err(|error| self.expired_if_superseded(db, &counters, error))?,
            );
            if let Some(cache) = cache {
                cache.insert(key, Arc::clone(&base));
            }
            base
        };
        let tail = self
            .read_tail(
                db,
                counters.log_epoch,
                counters.log_epoch_start_seq,
                counters.next_owner_seq,
            )
            .map_err(|error| self.expired_if_superseded(db, &counters, error))?;
        Ok(PopulationSnapshot {
            base,
            tail: SortedPopulation::new(tail.into_iter().map(element).collect()),
            owner_seq_ceiling: counters.next_owner_seq,
        })
    }

    /// A missing record means corruption unless the scope has compacted past
    /// the pin, in which case the generation was legitimately dropped.
    fn expired_if_superseded(
        &self,
        db: &Database,
        pinned: &ScopeCounters,
        error: StorageError,
    ) -> StorageError {
        let dropped = matches!(
            error,
            StorageError::StaleGeneration { .. } | StorageError::NotFound(_)
        );
        match self.counters(db) {
            Ok(live) if dropped && live.manifest_version != pinned.manifest_version => {
                StorageError::StaleGeneration {
                    generation: u64::from(pinned.manifest_version),
                    current: Some(u64::from(live.manifest_version)),
                }
            }
            _ => error,
        }
    }

    fn load_base(
        &self,
        db: &Database,
        version: u32,
        folded_through: u32,
    ) -> Result<PopulationBase, StorageError> {
        let txn = db.begin_transaction();
        let manifest = self.read_manifest(&txn, version)?;
        if manifest.folded_through != folded_through {
            return Err(StorageError::Corrupt(
                "short-id manifest watermark disagrees with the counter".to_owned(),
            ));
        }
        let runs = manifest
            .runs
            .iter()
            .map(|run| self.load_run(&txn, run))
            .collect::<Result<Vec<_>, _>>()?;
        let slices: Vec<&[Entry]> = runs.iter().map(Vec::as_slice).collect();
        let merged = merge_sorted(&slices);
        Ok(PopulationBase {
            manifest_version: version,
            kernel: manifest.kernel,
            population: SortedPopulation::new(merged.into_iter().map(element).collect()),
        })
    }

    /// Pin the room's whole population at the current owner-log ceiling.
    ///
    /// # Errors
    /// As [`Self::materialize_population`].
    pub fn population_snapshot(&self, db: &Database) -> Result<PopulationSnapshot, StorageError> {
        let pin = self.pin_population(db)?;
        self.materialize_population(db, &pin, None)
    }

    /// As [`Self::population_snapshot`], reusing and filling `cache`.
    ///
    /// # Errors
    /// As [`Self::materialize_population`].
    pub fn population_snapshot_cached(
        &self,
        db: &Database,
        cache: &PopulationCache,
    ) -> Result<PopulationSnapshot, StorageError> {
        let pin = self.pin_population(db)?;
        self.materialize_population(db, &pin, Some(cache))
    }

    /// Fold the owner-log tail into a new sorted run and publish it.
    ///
    /// Acts when the tail holds at least [`COMPACT_TRIGGER`] entries, or any
    /// entry when `force` is set; returns `None` otherwise. Older runs of
    /// comparable size are merged into the new one (each run is under half
    /// its predecessor, so there are about `log2(n / 2000)` of them). The
    /// folded log generation and any merged runs are queued for deletion
    /// `grace_ms` after `now_ms`; [`Self::collect_garbage`] performs it.
    ///
    /// The run is written first, in transactions nothing references; one final
    /// transaction writes the manifest and moves the counter's epoch, start
    /// sequence and manifest version together, carrying any entries appended
    /// meanwhile into the new generation. A commit landing inside that short
    /// transaction makes it stale and the compaction is retried from a fresh
    /// read. A writer committing back to back with no idle gap can still beat
    /// every attempt; the error is then a `StaleRead` and the caller tries
    /// again later.
    ///
    /// # Errors
    /// Returns an error on a read or write failure, corruption, or when
    /// writers keep invalidating the compaction past the retry budget.
    pub fn compact(
        &self,
        db: &Database,
        now_ms: u64,
        grace_ms: u64,
        force: bool,
    ) -> Result<Option<CompactionReport>, StorageError> {
        let mut last = None;
        for attempt in 0..MAX_ATTEMPTS {
            match self.compact_once(db, now_ms, grace_ms, force) {
                Err(error) if error.is_stale_read() => last = Some(error),
                other => return other,
            }
            // Give the writer that beat us a gap to idle in.
            let backoff = u64::try_from(attempt.saturating_add(1).min(10)).unwrap_or(10);
            std::thread::sleep(std::time::Duration::from_micros(
                200_u64.saturating_mul(backoff),
            ));
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("compaction retry budget".to_owned())))
    }

    fn compact_once(
        &self,
        db: &Database,
        now_ms: u64,
        grace_ms: u64,
        force: bool,
    ) -> Result<Option<CompactionReport>, StorageError> {
        let txn = db.begin_transaction();
        let (counters, counter_token) = self.read_counters(&txn)?;
        let tail_len = counters
            .next_owner_seq
            .checked_sub(counters.log_epoch_start_seq)
            .ok_or_else(|| StorageError::Corrupt("owner sequence below its epoch".to_owned()))?;
        if tail_len == 0 || (!force && tail_len < COMPACT_TRIGGER) {
            return Ok(None);
        }
        let manifest = self.read_manifest(&txn, counters.manifest_version)?;
        let version = counters
            .manifest_version
            .checked_add(1)
            .ok_or_else(|| StorageError::Exhausted("short-id manifest versions".to_owned()))?;

        let mut carry = self.read_tail(
            db,
            counters.log_epoch,
            counters.log_epoch_start_seq,
            counters.next_owner_seq,
        )?;
        let mut kernel = manifest.kernel.clone();
        for &entry in &carry {
            kernel.insert(element(entry)).map_err(algebraic)?;
        }

        // Size tiering: absorb the newest kept run while it is under twice
        // the carry, so runs shrink at least by half toward the newest.
        let mut kept = manifest.runs.clone();
        let mut retired: Vec<[u8; 16]> = vec![self.owner_log_collection(counters.log_epoch)];
        while let Some(previous) = kept.last().copied() {
            let carried = u64::try_from(carry.len()).unwrap_or(u64::MAX);
            if carried.saturating_mul(2) < u64::from(previous.entries) {
                break;
            }
            let older = self.load_run(&txn, &previous)?;
            carry = merge_sorted(&[&older, &carry]);
            retired.push(self.run_collection(&previous));
            kept.pop();
        }

        let run = RunMeta {
            version,
            token: counter_token,
            entries: u32::try_from(carry.len())
                .map_err(|_| StorageError::Exhausted("short-id run entries".to_owned()))?,
            chunks: u32::try_from(carry.len().div_ceil(RUN_CHUNK_ENTRIES))
                .map_err(|_| StorageError::Exhausted("short-id run chunks".to_owned()))?,
        };
        // Queue the run for deletion before it exists: an attempt that is
        // abandoned (another compactor won, or a crash) leaves a queued
        // collection that garbage collection removes. The publish takes the
        // entry off the list.
        let run_collection = self.run_collection(&run);
        self.queue_garbage(db, &[(run_collection, now_ms.saturating_add(grace_ms))])?;
        self.write_run(db, &run, &carry)?;
        kept.push(run);
        let new_manifest = Manifest {
            version,
            folded_through: counters.next_owner_seq,
            kernel,
            runs: kept,
        };
        if new_manifest.total_entries() != u64::from(counters.next_owner_seq.saturating_sub(1)) {
            return Err(StorageError::Corrupt(
                "short-id runs do not cover the folded owner sequence".to_owned(),
            ));
        }
        #[cfg(test)]
        BEFORE_PUBLISH.with(|hook| {
            if let Some(mut hook) = hook.borrow_mut().take() {
                hook();
            }
        });
        let retired_collections = self.publish_compaction(
            db,
            &PublishPlan {
                counters,
                counter_token,
                version,
                run_collection,
                manifest: new_manifest.encode()?,
                retired,
                now_ms,
                grace_ms,
            },
        )?;
        Ok(Some(CompactionReport {
            manifest_version: version,
            folded_entries: tail_len,
            run_entries: run.entries,
            retired_collections,
        }))
    }

    /// Append `items` to the garbage list.
    fn queue_garbage(&self, db: &Database, items: &[([u8; 16], u64)]) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        let (record, token) = self.read_record(&txn, &self.collection_id, GC_ID)?;
        let mut garbage = match record {
            Some(record) => decode_gc(&record.bytes)?,
            None => Vec::new(),
        };
        garbage.extend_from_slice(items);
        txn.expect_record_version(self.pool, self.collection_id, GC_ID, token)?;
        txn.put(
            self.pool,
            self.collection_id,
            GC_ID,
            &NodeData::new(Bytes::from(encode_gc(&garbage)?)),
        )?;
        txn.commit()
    }

    /// The compaction's last step. The run and manifest stay valid for as
    /// long as the counter's epoch, start and manifest version are unchanged,
    /// so only the publish is retried when a writer commits first.
    fn publish_compaction(&self, db: &Database, plan: &PublishPlan) -> Result<usize, StorageError> {
        let mut last = None;
        for attempt in 0..MAX_ATTEMPTS {
            match self.publish_once(db, plan) {
                Ok(Published::Done(retired)) => return Ok(retired),
                Ok(Published::Superseded) => {
                    return Err(StorageError::StaleRead {
                        pool: self.pool,
                        collection_id: self.collection_id,
                        expected: plan.counter_token,
                        actual: 0,
                    })
                }
                Err(error) if error.is_stale_read() => last = Some(error),
                Err(error) => return Err(error),
            }
            // Give the writer that beat us a gap to idle in.
            let backoff = u64::try_from(attempt.saturating_add(1).min(10)).unwrap_or(10);
            std::thread::sleep(std::time::Duration::from_micros(
                100_u64.saturating_mul(backoff),
            ));
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("publish retry budget".to_owned())))
    }

    /// One transaction that moves the counter and queues the retired
    /// collections. Writers kept appending while the run was built, so it
    /// re-reads the counter itself and carries the few entries appended since
    /// into the new generation; only a commit landing between that read and
    /// this commit makes it stale.
    fn publish_once(&self, db: &Database, plan: &PublishPlan) -> Result<Published, StorageError> {
        let publish = db.begin_transaction();
        let (live, live_token) = self.read_counters(&publish)?;
        if live.log_epoch != plan.counters.log_epoch
            || live.manifest_version != plan.counters.manifest_version
            || live.log_epoch_start_seq != plan.counters.log_epoch_start_seq
        {
            return Ok(Published::Superseded);
        }
        let new_epoch = plan.counters.log_epoch.saturating_add(1);
        let new_generation = self.owner_log_collection(new_epoch);
        for entry in self.owner_log_in(
            &publish,
            plan.counters.log_epoch,
            plan.counters.next_owner_seq,
            live.next_owner_seq,
        )? {
            publish.put(
                self.log_pool,
                new_generation,
                owner_log_record_id(entry.seq),
                &NodeData::new(Bytes::from(encode_owner_log(
                    entry.short_id,
                    &entry.payload,
                ))),
            )?;
        }
        let (gc_record, gc_token) = self.read_record(&publish, &self.collection_id, GC_ID)?;
        let mut garbage = match gc_record {
            Some(record) => decode_gc(&record.bytes)?,
            None => Vec::new(),
        };
        // The run is installed: it is live, not garbage. If it is no longer
        // on the list, garbage collection took it (its grace period ended
        // while the run was being written) and the collection is gone, so
        // publishing would name a deleted run. The opposite order is safe:
        // collection would fail its compare-and-set on this list and retry.
        let queued = garbage.len();
        garbage.retain(|(collection, _)| *collection != plan.run_collection);
        if garbage.len() == queued {
            return Ok(Published::Superseded);
        }
        let deadline = plan.now_ms.saturating_add(plan.grace_ms);
        let retired_collections = plan.retired.len();
        garbage.extend(
            plan.retired
                .iter()
                .copied()
                .map(|collection| (collection, deadline)),
        );
        publish.expect_record_version(self.pool, self.collection_id, COUNTER_ID, live_token)?;
        publish.expect_record_version(self.pool, self.collection_id, GC_ID, gc_token)?;
        publish.put(
            self.pool,
            self.collection_id,
            COUNTER_ID,
            &NodeData::new(Bytes::from(encode_counter(ScopeCounters {
                log_epoch: new_epoch,
                log_epoch_start_seq: plan.counters.next_owner_seq,
                manifest_version: plan.version,
                ..live
            }))),
        )?;
        publish.put(
            self.pool,
            self.collection_id,
            GC_ID,
            &NodeData::new(Bytes::from(encode_gc(&garbage)?)),
        )?;
        publish.put(
            self.pool,
            self.collection_id,
            derived_id(MANIFEST_PREFIX, plan.version, 0),
            &NodeData::new(Bytes::from(plan.manifest.clone())),
        )?;
        publish.commit()?;
        Ok(Published::Done(retired_collections))
    }

    fn write_run(
        &self,
        db: &Database,
        run: &RunMeta,
        entries: &[Entry],
    ) -> Result<(), StorageError> {
        let collection = self.run_collection(run);
        #[cfg(test)]
        WRITTEN_RUNS.lock().push((self.collection_id, collection));
        let mut index = 0_u32;
        for group in entries.chunks(RUN_CHUNK_ENTRIES * CHUNKS_PER_TXN) {
            let txn = db.begin_transaction();
            for chunk in group.chunks(RUN_CHUNK_ENTRIES) {
                txn.put(
                    self.log_pool,
                    collection,
                    derived_id(RUN_CHUNK_PREFIX, index, 0),
                    &NodeData::new(Bytes::from(encode_chunk(chunk)?)),
                )?;
                index = index.saturating_add(1);
            }
            txn.commit()?;
        }
        Ok(())
    }

    /// Delete every queued collection whose grace period ended by `now_ms`.
    ///
    /// Dropping a collection is a tombstone; its bytes return only once every
    /// collection sharing the shard has been repacked, and the scope's id
    /// records share the log's shards (see [`Self::repack_live_collections`]).
    ///
    /// # Errors
    /// Returns an error on a read, write or repack failure.
    pub fn collect_garbage(&self, db: &Database, now_ms: u64) -> Result<GcReport, StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            match self.collect_garbage_once(db, now_ms) {
                Err(error) if error.is_stale_read() => last = Some(error),
                other => return other,
            }
        }
        Err(last.unwrap_or_else(|| StorageError::Internal("garbage retry budget".to_owned())))
    }

    fn collect_garbage_once(&self, db: &Database, now_ms: u64) -> Result<GcReport, StorageError> {
        let txn = db.begin_transaction();
        let (record, token) = self.read_record(&txn, &self.collection_id, GC_ID)?;
        let Some(record) = record else {
            return Ok(GcReport::default());
        };
        let (due, pending): (Vec<_>, Vec<_>) = decode_gc(&record.bytes)?
            .into_iter()
            .partition(|(_, deadline)| *deadline <= now_ms);
        if due.is_empty() {
            return Ok(GcReport {
                pending: pending.len(),
                ..GcReport::default()
            });
        }
        txn.expect_record_version(self.pool, self.collection_id, GC_ID, token)?;
        for (collection, _) in &due {
            txn.delete_collection(self.log_pool, *collection)?;
        }
        txn.put(
            self.pool,
            self.collection_id,
            GC_ID,
            &NodeData::new(Bytes::from(encode_gc(&pending)?)),
        )?;
        txn.commit()?;
        Ok(GcReport {
            dropped: due.len(),
            pending: pending.len(),
        })
    }

    /// Repack the live log generation and the live runs.
    ///
    /// Measured at 1M owners: after compaction and garbage collection this
    /// reclaimed nothing (0 shards retired) and grew the packs by the size of
    /// the repacked run, because the dropped log generation shares its shards
    /// with the scope's id records, which stay referenced. It is therefore not
    /// part of [`Self::collect_garbage`]; call it only if the shards in
    /// question hold nothing else.
    ///
    /// # Errors
    /// Returns an error on a read or repack failure.
    pub fn repack_live_collections(&self, db: &Database) -> Result<usize, StorageError> {
        let counters = self.counters(db)?;
        let txn = db.begin_transaction();
        let manifest = self.read_manifest(&txn, counters.manifest_version)?;
        let mut live = vec![self.owner_log_collection(counters.log_epoch)];
        live.extend(manifest.runs.iter().map(|run| self.run_collection(run)));
        let pool = db.pool(self.log_pool);
        for collection in &live {
            pool.repack_collection_reachable(collection, |_, _| Vec::new())?;
        }
        Ok(live.len())
    }

    /// Every collection the current manifest's runs and the garbage list name,
    /// and whether its first chunk record is still stored.
    #[cfg(test)]
    pub(crate) fn run_collections_for_test(
        &self,
        db: &Database,
    ) -> Result<Vec<([u8; 16], bool)>, StorageError> {
        let counters = self.counters(db)?;
        let txn = db.begin_transaction();
        let manifest = self.read_manifest(&txn, counters.manifest_version)?;
        let mut collections: Vec<[u8; 16]> = manifest
            .runs
            .iter()
            .map(|run| self.run_collection(run))
            .collect();
        let (record, _) = self.read_record(&txn, &self.collection_id, GC_ID)?;
        if let Some(record) = record {
            collections.extend(decode_gc(&record.bytes)?.into_iter().map(|(c, _)| c));
        }
        collections
            .into_iter()
            .map(|collection| {
                let (record, _) = Self::read_record_in(
                    self.log_pool,
                    &txn,
                    &collection,
                    derived_id(RUN_CHUNK_PREFIX, 0, 0),
                )?;
                Ok((collection, record.is_some()))
            })
            .collect()
    }

    /// Stage the drop of every run and queued collection into `txn`; the
    /// manifests and the garbage list live in the scope's own collection.
    pub(crate) fn stage_population_purge(
        &self,
        txn: &DatabaseTransaction<'_>,
        counters: &ScopeCounters,
    ) -> Result<(), StorageError> {
        // Purge is the recovery tool: a manifest that cannot be read must not
        // stop it, so the runs it names are skipped and the rest still dropped.
        if let Ok(manifest) = self.read_manifest(txn, counters.manifest_version) {
            for run in &manifest.runs {
                txn.delete_collection(self.log_pool, self.run_collection(run))?;
            }
        }
        let (record, _) = self.read_record(txn, &self.collection_id, GC_ID)?;
        if let Some(record) = record {
            for (collection, _) in decode_gc(&record.bytes)? {
                txn.delete_collection(self.log_pool, collection)?;
            }
        }
        Ok(())
    }

    /// Check the folded part of the scope: the manifest named by the counter
    /// exists and its runs hold exactly the owners below the log's start.
    pub(crate) fn verify_manifest(
        &self,
        txn: &DatabaseTransaction<'_>,
        counters: &ScopeCounters,
        problems: &mut Vec<String>,
    ) {
        let folded = u64::from(counters.log_epoch_start_seq.saturating_sub(1));
        match self.read_manifest(txn, counters.manifest_version) {
            Ok(manifest) => {
                if manifest.folded_through != counters.log_epoch_start_seq {
                    problems.push(format!(
                        "manifest {} folds through {} but the log starts at {}",
                        manifest.version, manifest.folded_through, counters.log_epoch_start_seq
                    ));
                }
                if manifest.total_entries() != folded {
                    problems.push(format!(
                        "manifest {} runs hold {} entries but {folded} owners are folded",
                        manifest.version,
                        manifest.total_entries()
                    ));
                }
            }
            Err(error) => problems.push(format!("manifest unreadable: {error}")),
        }
    }
}
