//! Persisted, immutable closure generations published through a
//! [`LogicalHead`].
//!
//! A *closure* is a derived per-id blob (for auth chains, a serialized rezzy
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
//! Rebuild protocol: [`ClosureStore::begin`](crate::closure_store::ClosureStore::begin) reserves a never-reused generation
//! number (a CAS on the counter) and records the head token it started from;
//! [`GenerationBuilder::add`](crate::closure_store::GenerationBuilder::add) writes closures in bounded batches under the new,
//! still unpublished generation; [`GenerationBuilder::publish`](crate::closure_store::GenerationBuilder::publish) swaps the head
//! by CAS. Readers never see a partial generation: until the swap they read the
//! old one. If another builder published first, `publish` fails with
//! `StorageError::StaleRead` and discards its generation. A crash before the
//! swap leaves an orphan generation collection that
//! [`ClosureStore::retire_superseded`](crate::closure_store::ClosureStore::retire_superseded) reclaims.
//!
//! The head also records the short-id counter the generation was built against
//! (`source_next`) and the coverage it achieved: ids in `1..source_next` that
//! are deliberately left without a record because their walk was incomplete are
//! listed in the head's `skipped` set, run-encoded. That makes the three states
//! a reader can observe explicit and distinguishable — [`ClosureCoverage::Complete`](crate::closure_store::ClosureCoverage::Complete)
//! (record present), [`ClosureCoverage::Incomplete`](crate::closure_store::ClosureCoverage::Incomplete) (covered, record absent by
//! design), and [`ClosureCoverage::Absent`](crate::closure_store::ClosureCoverage::Absent) (outside `1..source_next`) — instead
//! of "no record" being ambiguous between incomplete and out of range.
//! Verifying a closure's *content* against the direct edges is the adapter's
//! job; [`ClosureStore::verify`](crate::closure_store::ClosureStore::verify) checks the storage invariants (head resolves,
//! every non-skipped covered id has a record, every skipped id has none,
//! counts match).
//!
//! Built on the transaction layer and engine CAS in [`crate::database`].

use bytes::Bytes;

use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::logical_head::{LogicalHead, LogicalHeadValue};
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageError};
use crate::template::{derive_collection_id, MEMBER_NAMESPACE_INTL};

/// Wire version of closure records, the generation counter and the coverage
/// record.
pub const CLOSURE_FORMAT_VERSION: u8 = 4;

const HEAD_LOGICAL_ID: NodeId = *b"MTXD-CLS-HEAD-v1";
const COUNTER_ID: NodeId = *b"MTXD-CLS-NEXTG-1";
const RECORD_MAGIC: [u8; 4] = *b"CLSR";
const COUNTER_MAGIC: [u8; 4] = *b"CLSG";
const COVERAGE_MAGIC: [u8; 4] = *b"CLCG";
const RECORD_PREFIX: [u8; 8] = *b"MTXCLSR\0";
const COVERAGE_PREFIX: [u8; 8] = *b"MTXCLSC\0";
/// Head metadata length: `generation`, `previous`, `source_next`, `count`,
/// `skipped_count`. Fixed width, so reading a head never scales with the size of
/// the generation.
const HEAD_METADATA_LEN: usize = 8 + 8 + 4 + 4 + 4;
/// Bytes per encoded skipped run: `(start, span)`, where `span` is `end - start`,
/// so the stored span is one less than the run's length.
const RUN_LEN: usize = 8;
const MAX_ATTEMPTS: usize = 64;

// ---------------------------------------------------------------------------
// Test-only instrumentation: head reads are counted so tests can assert the
// read-side work a call actually does, and a hook can republish or retire at an
// exact point to make a race deterministic instead of timing-dependent.
//
// Both are thread-local. Cargo runs tests in parallel threads, so a process-wide
// atomic would let one test observe another's reads, and a process-wide hook slot
// would let one test's `snapshot` consume another's hook and publish a generation
// into the wrong database.
// ---------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    static HEAD_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Bump this thread's head-read counter.
#[cfg(test)]
pub(crate) fn head_reads() {
    HEAD_READS.with(|reads| {
        let bumped = reads.get().saturating_add(1);
        reads.set(bumped);
    });
}

/// Reset and return this thread's head-read count.
#[cfg(test)]
pub(crate) fn take_head_reads() -> u64 {
    HEAD_READS.with(std::cell::Cell::take)
}

/// Runs once, inside `snapshot`, after the coverage record has been read and
/// before it is checked against the head's count.
#[cfg(test)]
type SnapshotHook = Box<dyn Fn(&Database)>;

#[cfg(test)]
thread_local! {
    /// Thread-local because cargo runs tests in parallel: a process-global hook
    /// could be consumed by another test's `snapshot`, which would then publish a
    /// generation into the wrong database. Thread-local also means a panic
    /// before `disarm_snapshot_hook` cannot leak into an unrelated test.
    static SNAPSHOT_HOOK: std::cell::RefCell<Option<SnapshotHook>> =
        const { std::cell::RefCell::new(None) };
}

/// Arm a hook that runs inside [`ClosureStore::snapshot`] after the coverage
/// read. The hook gets the database so it can republish or retire, which is how
/// tests make the head-moved-mid-snapshot case deterministic.
///
/// # Panics
/// Panics if called outside a test build.
#[cfg(test)]
pub(crate) fn arm_snapshot_hook<F>(hook: F)
where
    F: Fn(&Database) + 'static,
{
    SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

/// Clear any armed [`arm_snapshot_hook`].
///
/// # Panics
/// Panics if called outside a test build.
#[cfg(test)]
pub(crate) fn disarm_snapshot_hook() {
    SNAPSHOT_HOOK.with(|slot| *slot.borrow_mut() = None);
}

/// Take the armed hook, if any, so it fires at most once.
#[cfg(test)]
fn take_snapshot_hook() -> Option<SnapshotHook> {
    SNAPSHOT_HOOK.with(|slot| slot.borrow_mut().take())
}

/// Exit code of a child stopped by [`crash_point`]. Distinct from 0 (clean run)
/// and 101 (panic), so a test can tell an injected crash from either.
#[cfg(test)]
pub(super) const CRASH_EXIT_CODE: i32 = 86;

/// Test-only crash injection: end the process (no destructors, no flush) when
/// `MTXDB_CRASH_AT` names this point, so recovery tests can kill a real child
/// process at an exact step.
///
/// Exits with [`CRASH_EXIT_CODE`] rather than `abort()`: `exit` skips the same
/// Rust destructors, but a `SIGABRT` would be dumped by systemd-coredump and
/// raise a desktop crash report for every injected crash.
#[cfg(test)]
fn crash_point(name: &str) {
    if std::env::var("MTXDB_CRASH_AT").as_deref() == Ok(name) {
        std::process::exit(CRASH_EXIT_CODE);
    }
}

#[cfg(not(test))]
#[inline(always)]
fn crash_point(_name: &str) {}

/// Whether a published generation holds a closure for an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureCoverage {
    /// Covered and complete: a closure record exists.
    Complete,
    /// Covered but incomplete: no record, by design, because a reachable `auth`
    /// parent was absent when the generation was built.
    Incomplete,
    /// Not covered by this generation: short id `0`, or at/after `source_next`.
    Absent,
}

/// The published generation, what it covers, and where it fell short.
///
/// Fixed width on purpose: the head is read on every `get`, `get_many`, `begin`,
/// `verify` and `publish`, so it must not grow with the size of a generation.
/// The skipped ids themselves live in one record inside the generation
/// collection (see [`ClosureCoverageSet`]), fetched once per reader snapshot.
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
    /// How many covered ids are incomplete. The ids themselves need
    /// [`ClosureSnapshot::coverage`] to resolve.
    pub skipped_count: u32,
}

impl ClosureHead {
    /// Whether this generation claims to cover every id in `1..source_next`.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.skipped_count == 0
    }
}

/// The skipped ids of one generation, as runs of consecutive short ids.
///
/// Kept run-encoded rather than expanded, so a room whose early event is missing
/// costs a single `(1, len)` run instead of one `u32` per event. A hole near the
/// start of a large room would otherwise allocate proportional to the room on
/// every head read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureCoverageSet {
    runs: Box<[(u32, u32)]>,
}

/// A pinned generation: its head plus the ids it skipped.
///
/// Everything a reader derives from this belongs to one generation, so a
/// multi-step auth query can hold it and never mix generations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureSnapshot {
    /// The generation this snapshot is pinned to.
    pub head: ClosureHead,
    /// Which covered ids that generation deliberately left unrecorded.
    pub skipped: ClosureCoverageSet,
}

impl ClosureSnapshot {
    /// What a reader will find for `short_id`.
    #[must_use]
    pub fn coverage(&self, short_id: u32) -> ClosureCoverage {
        self.skipped.coverage(&self.head, short_id)
    }

    /// Whether this generation covers every id in `1..source_next`.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.head.is_complete()
    }
}

impl ClosureCoverageSet {
    /// Build from an ascending, deduplicated id list, collapsing consecutive ids
    /// into runs.
    #[must_use]
    pub fn from_ids(ids: &[u32]) -> Self {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for &id in ids {
            match runs.last_mut() {
                // `checked_add` keeps a run from wrapping past u32::MAX.
                Some((_, end)) if end.checked_add(1) == Some(id) => *end = id,
                _ => runs.push((id, id)),
            }
        }
        Self {
            runs: runs.into_boxed_slice(),
        }
    }

    /// The runs, each `(start, end)` inclusive and ascending.
    #[must_use]
    pub fn runs(&self) -> &[(u32, u32)] {
        &self.runs
    }

    /// How many ids are skipped in total.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.runs
            .iter()
            .map(|&(start, end)| u64::from(end.saturating_sub(start)).saturating_add(1))
            .sum()
    }

    /// Whether no ids are skipped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Whether `short_id` is one of the skipped ids. `O(log runs)`.
    #[must_use]
    pub fn contains(&self, short_id: u32) -> bool {
        self.runs
            .binary_search_by(|&(start, end)| {
                if short_id < start {
                    std::cmp::Ordering::Greater
                } else if short_id > end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }

    /// What a reader will find for `short_id` in `head`.
    #[must_use]
    pub fn coverage(&self, head: &ClosureHead, short_id: u32) -> ClosureCoverage {
        if short_id == 0 || short_id >= head.source_next {
            ClosureCoverage::Absent
        } else if self.contains(short_id) {
            ClosureCoverage::Incomplete
        } else {
            ClosureCoverage::Complete
        }
    }
}

/// Accumulates skipped ids into a [`ClosureCoverageSet`] as they are
/// discovered, without holding every id and sorting afterwards.
///
/// [`ClosureCoverageSet::from_ids`] needs ascending, deduplicated input, so a
/// caller that discovers ids one at a time would otherwise keep a second
/// `Vec<u32>` and sort it, on top of the runs the set already stores. This keeps
/// only runs: an id that follows the open run extends it, anything else starts a
/// new run. A caller that pushes in ascending order, as the rebuild walk does,
/// gets the same set with one copy and no sort.
///
/// Out-of-order pushes are still correct: [`Self::build`] sorts and merges once,
/// so only the run-collapsing fast path depends on ascending input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureCoverageBuilder {
    runs: Vec<(u32, u32)>,
}

impl ClosureCoverageBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Record `id` as skipped, extending the open run when it is the next id.
    pub fn push(&mut self, id: u32) {
        match self.runs.last_mut() {
            // `checked_add` keeps a run from wrapping past u32::MAX.
            Some((_, end)) if end.checked_add(1) == Some(id) => *end = id,
            _ => self.runs.push((id, id)),
        }
    }

    /// How many ids have been pushed.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.runs
            .iter()
            .map(|&(start, end)| u64::from(end.saturating_sub(start)).saturating_add(1))
            .sum()
    }

    /// Whether no id has been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Finish the set, merging runs that were pushed out of order.
    #[must_use]
    pub fn build(mut self) -> ClosureCoverageSet {
        self.runs.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.runs.len());
        for &(start, end) in &self.runs {
            match merged.last_mut() {
                // `<=`, so overlapping or adjacent runs coalesce. Duplicate ids
                // arrive here as `(id, id)` twice.
                Some((_, last)) if last.saturating_add(1) >= start => {
                    if end > *last {
                        *last = end;
                    }
                }
                _ => merged.push((start, end)),
            }
        }
        ClosureCoverageSet {
            runs: merged.into_boxed_slice(),
        }
    }
}

/// The counts and coverage payload a publish would commit, or the reason it must
/// not.
///
/// Kept separate from the write path so every check is known to run before the
/// first byte is written, which is what lets a rejected publish roll its records
/// back without a partially staged generation.
fn validate_publish(
    stored: &[u32],
    set: &ClosureCoverageSet,
    source_next: u32,
) -> Result<(u32, u32, Vec<u8>), StorageError> {
    // Every check runs over the runs themselves, not an expanded id list: a
    // skipped run can cover thousands of ids, and the point of the run encoding
    // is to not touch them.
    let runs = set.runs();
    // The runs are ascending and merged, so the first start and the last end
    // bound every id in between.
    if let (Some(&(first, _)), Some(&(_, last))) = (runs.first(), runs.last()) {
        if first == 0 || last >= source_next {
            return Err(StorageError::Internal(format!(
                "skipped closure ids {first}..={last} lie outside 1..{source_next}"
            )));
        }
    }
    // Stored ids must themselves be covered. `stored` is sorted, so the ends bound
    // the whole vector. Without this, storing id 10 and publishing `source_next =
    // 5` yields a head that decodes fine but names a count no generation can
    // satisfy: the record is unreachable and `verify` reports the mismatch. Reject
    // it at the write path instead.
    if let (Some(&first), Some(&last)) = (stored.first(), stored.last()) {
        if first == 0 || last >= source_next {
            return Err(StorageError::Internal(format!(
                "stored closure ids {first}..={last} lie outside 1..{source_next}"
            )));
        }
    }
    // An id cannot be both stored and skipped: readers resolve coverage from the
    // skipped set first, so a stored closure for a skipped id would be unreachable
    // and `verify` would flag it. `stored` is sorted, so one binary search per run
    // finds any overlap.
    for &(start, end) in runs {
        let index = stored.partition_point(|&id| id < start);
        if let Some(&clash) = stored.get(index) {
            if clash <= end {
                return Err(StorageError::Internal(format!(
                    "closure id {clash} is stored and claimed as skipped"
                )));
            }
        }
    }
    let stored_count = u32::try_from(stored.len())
        .map_err(|_| StorageError::Internal("too many stored closures".to_owned()))?;
    let skipped_count = u32::try_from(set.len())
        .map_err(|_| StorageError::Internal("too many skipped closure ids".to_owned()))?;
    // Belt and braces: the two checks above already imply this, but the head
    // decoder enforces the partition, so reject it here too rather than committing
    // a generation that `decode_head` would refuse to read back.
    let covered = u64::from(source_next.saturating_sub(1));
    if u64::from(stored_count)
        .checked_add(set.len())
        .is_none_or(|total| total > covered)
    {
        return Err(StorageError::Internal(format!(
            "{stored_count} stored and {} skipped closures exceed the {covered} covered ids",
            set.len()
        )));
    }
    let payload = encode_coverage(set);
    Ok((stored_count, skipped_count, payload))
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

/// The id of a closure record. Hashed rather than laid out as prefix plus short
/// id: the engine's hash index buckets by the first 8 bytes of an id, so a
/// constant prefix there would put every closure record in one bucket (see
/// `short_id::derived_id`).
fn record_id(short_id: u32) -> NodeId {
    let mut input = [0u8; 12];
    input[..8].copy_from_slice(&RECORD_PREFIX);
    input[8..12].copy_from_slice(&short_id.to_be_bytes());
    let digest = DigestAlgorithm::Blake3.digest(&input);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// The id of a closure record, exposed so a test can check the ids the index
/// will see are spread.
#[cfg(test)]
pub(crate) fn record_id_for_test(short_id: u32) -> NodeId {
    record_id(short_id)
}

/// The one coverage record inside a generation collection. It sits beside the
/// closure records but under a distinct prefix, so it is never mistaken for a
/// closure for short id `0`.
fn coverage_id() -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&COVERAGE_PREFIX);
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

#[derive(Clone, Copy)]
struct GenerationCounter {
    next: u64,
    retired_through: u64,
}

fn encode_counter(counter: GenerationCounter) -> Vec<u8> {
    let mut out = Vec::with_capacity(21);
    out.extend_from_slice(&COUNTER_MAGIC);
    out.push(CLOSURE_FORMAT_VERSION);
    out.extend_from_slice(&counter.next.to_be_bytes());
    out.extend_from_slice(&counter.retired_through.to_be_bytes());
    out
}

fn decode_counter(bytes: &[u8]) -> Result<GenerationCounter, StorageError> {
    if (bytes.len() != 13 && bytes.len() != 21)
        || bytes[..4] != COUNTER_MAGIC
        || bytes[4] != CLOSURE_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt(
            "closure generation counter".to_owned(),
        ));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[5..13]);
    let next = u64::from_be_bytes(raw);
    let retired_through = if bytes.len() == 21 {
        raw.copy_from_slice(&bytes[13..21]);
        u64::from_be_bytes(raw)
    } else {
        0
    };
    if retired_through >= next && next != 0 {
        return Err(StorageError::Corrupt(
            "closure generation retirement cursor".to_owned(),
        ));
    }
    Ok(GenerationCounter {
        next,
        retired_through,
    })
}

fn encode_head_metadata(head: &ClosureHead) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_METADATA_LEN);
    out.extend_from_slice(&head.generation.to_be_bytes());
    out.extend_from_slice(&head.previous.to_be_bytes());
    out.extend_from_slice(&head.source_next.to_be_bytes());
    out.extend_from_slice(&head.count.to_be_bytes());
    out.extend_from_slice(&head.skipped_count.to_be_bytes());
    out
}

fn decode_head(
    value: &LogicalHeadValue,
    store: &ClosureStore,
) -> Result<ClosureHead, StorageError> {
    fn corrupt() -> StorageError {
        StorageError::Corrupt("closure head metadata".to_owned())
    }
    let bytes = &value.metadata;
    if bytes.len() != HEAD_METADATA_LEN {
        return Err(corrupt());
    }
    let word = |at: usize| -> Result<u64, StorageError> {
        let raw: [u8; 8] = bytes
            .get(at..)
            .and_then(|b| b.get(..8))
            .ok_or_else(corrupt)?
            .try_into()
            .map_err(|_| corrupt())?;
        Ok(u64::from_be_bytes(raw))
    };
    let short = |at: usize| -> Result<u32, StorageError> {
        let raw: [u8; 4] = bytes
            .get(at..)
            .and_then(|b| b.get(..4))
            .ok_or_else(corrupt)?
            .try_into()
            .map_err(|_| corrupt())?;
        Ok(u32::from_be_bytes(raw))
    };
    let generation = word(0)?;
    let previous = word(8)?;
    let source_next = short(16)?;
    let count = short(20)?;
    let skipped_count = short(24)?;

    // The head only counts skipped ids; it does not list them, so it cannot
    // under-claim the exact set. It can over-claim, which
    // `ClosureStore::verify` and the `get_many` corruption path both catch.
    if u64::from(skipped_count).saturating_add(u64::from(count))
        > u64::from(source_next.saturating_sub(1))
    {
        return Err(corrupt());
    }

    if value.target != store.generation_collection(generation) {
        return Err(StorageError::Corrupt(
            "closure head target does not name its generation".to_owned(),
        ));
    }
    Ok(ClosureHead {
        generation,
        previous,
        source_next,
        count,
        skipped_count,
    })
}

/// Whether an error describes on-disk inconsistency, which `verify` should
/// report as a finding, rather than a read failure it should propagate.
///
/// A stale generation is included: a head that moved under a verify run is not
/// a storage defect, and `verify` retries internally before giving up, so
/// reaching this arm means the head is genuinely thrashing.
fn is_malformed(error: &StorageError) -> bool {
    matches!(
        error,
        StorageError::Corrupt(_) | StorageError::StaleGeneration { .. }
    )
}

/// Coverage records hold one run per gap, `(start, span)`, ascending.
fn encode_coverage(set: &ClosureCoverageSet) -> Vec<u8> {
    let mut out = Vec::with_capacity(RUN_LEN.saturating_mul(set.runs().len()));
    out.extend_from_slice(&COVERAGE_MAGIC);
    out.push(CLOSURE_FORMAT_VERSION);
    for &(start, end) in set.runs() {
        out.extend_from_slice(&start.to_be_bytes());
        out.extend_from_slice(&end.saturating_sub(start).to_be_bytes());
    }
    out
}

fn decode_coverage(bytes: &[u8]) -> Result<ClosureCoverageSet, StorageError> {
    let corrupt = || StorageError::Corrupt("closure coverage record".to_owned());
    let Some(body) = bytes.get(5..) else {
        return Err(corrupt());
    };
    if bytes.len() < 5 || bytes[..4] != COVERAGE_MAGIC || bytes[4] != CLOSURE_FORMAT_VERSION {
        return Err(corrupt());
    }
    if body.len() % RUN_LEN != 0 {
        return Err(corrupt());
    }
    let mut runs: Vec<(u32, u32)> = Vec::with_capacity(body.len() / RUN_LEN);
    for chunk in body.as_chunks::<RUN_LEN>().0 {
        let start = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let stored = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let Some(end) = start.checked_add(stored) else {
            return Err(corrupt());
        };
        // Runs must be strictly ascending and non-adjacent, so the encoding is
        // canonical. Adjacent runs would decode to the same set as one merged
        // run, so accepting them would let two encodings mean the same thing.
        // Runs are not bounds-checked against `source_next` here because that
        // needs the head; `ClosureCoverageSet::coverage` checks `source_next`
        // first, and `verify` reports out-of-range runs.
        if start == 0
            || runs.last().is_some_and(|&(_, prev_end)| {
                start <= prev_end || start == prev_end.saturating_add(1)
            })
        {
            return Err(corrupt());
        }
        runs.push((start, end));
    }
    Ok(ClosureCoverageSet {
        runs: runs.into_boxed_slice(),
    })
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
        #[cfg(test)]
        head_reads();
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
    pub fn head(&self, db: &Database) -> Result<Option<ClosureHead>, StorageError> {
        self.read_head(&db.begin_transaction())
            .map(|(head, _)| head)
    }

    /// The published head plus the ids it skipped, as one consistent snapshot.
    ///
    /// One transaction is **not** enough on its own: the head and its coverage
    /// record are separate records, so a concurrent republish and retire can land
    /// between the two reads. Consistency comes from the count check instead. The
    /// head states how many ids are skipped; the coverage record is fetched and
    /// checked against that count, so a generation retired mid-read arrives as a
    /// mismatch rather than as a silently short skipped set. A mismatch means
    /// either a corrupt record or a moved head, and re-reading the head
    /// distinguishes the two: only an unmoved head is corruption.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt head or coverage record, a
    /// head whose skipped count disagrees with its coverage record, or a head
    /// that keeps moving for every attempt.
    pub fn snapshot(&self, db: &Database) -> Result<Option<ClosureSnapshot>, StorageError> {
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            let (Some(head), _) = self.read_head(&txn)? else {
                return Ok(None);
            };
            // Fires before the coverage read, which is the window a concurrent
            // republish and retire can invalidate.
            #[cfg(test)]
            if let Some(hook) = take_snapshot_hook() {
                hook(db);
            }
            // A complete generation writes no coverage record, so the common case
            // costs the head read alone.
            let set = if head.skipped_count == 0 {
                ClosureCoverageSet::default()
            } else {
                self.read_coverage(&txn, head.generation)?
            };
            if set.len() != u64::from(head.skipped_count) {
                // The count disagrees, so either the record is corrupt or this
                // generation was retired and replaced while we read. Re-reading
                // the head distinguishes the two.
                let (again, _) = self.read_head(&db.begin_transaction())?;
                if again.map(|fresh| fresh.generation) != Some(head.generation) {
                    continue;
                }
                return Err(StorageError::Corrupt(format!(
                    "closure generation {} head counts {} skipped ids but its coverage record \
                     has {}",
                    head.generation,
                    head.skipped_count,
                    set.len()
                )));
            }
            return Ok(Some(ClosureSnapshot { head, skipped: set }));
        }
        Err(StorageError::Internal(
            "closure head kept moving during a read".to_owned(),
        ))
    }

    /// Read the coverage record of `generation` inside an existing transaction.
    ///
    /// # Errors
    /// Returns an error on a read failure or a corrupt coverage record.
    fn read_coverage(
        &self,
        txn: &DatabaseTransaction<'_>,
        generation: u64,
    ) -> Result<ClosureCoverageSet, StorageError> {
        let (records, _) = txn.get_with_record_versions(
            self.pool,
            &self.generation_collection(generation),
            &[coverage_id()],
        )?;
        match records.into_iter().next().flatten() {
            Some(record) => decode_coverage(&record.bytes),
            None => Ok(ClosureCoverageSet::default()),
        }
    }

    /// The closure blob of `short_id` in the published generation.
    ///
    /// # Errors
    /// As [`Self::get_many`].
    pub fn get(&self, db: &Database, short_id: u32) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self
            .get_many(db, &[short_id])?
            .1
            .into_iter()
            .next()
            .flatten())
    }

    /// Closure blobs for several ids, all read from **one** generation: the head
    /// and its coverage record are resolved once, so the result can never mix two
    /// generations. Returns the generation read (`None` if nothing is published)
    /// and the blobs in order.
    ///
    /// Coverage is consulted first, so an id the generation records as
    /// [`ClosureCoverage::Incomplete`] or [`ClosureCoverage::Absent`] costs no
    /// record read and no re-check — those `None`s are final answers, not races.
    /// Only ids classified [`ClosureCoverage::Complete`] are fetched, and a
    /// missing one means the generation was retired mid-read: the head is
    /// re-read, and if it moved the whole call retries against the new
    /// generation. If it did not move, a complete id with no record is
    /// corruption, and is reported as such rather than as a silent miss.
    ///
    /// Prefer [`Self::snapshot`] plus [`Self::get_many_pinned`] when a caller
    /// makes several reads and wants them all from one generation.
    ///
    /// # Errors
    /// Returns an error on a read failure, a corrupt record, a corrupt head or
    /// coverage record, or a complete id whose record is missing from an
    /// unmoved head.
    #[allow(clippy::type_complexity, reason = "generation plus ordered blobs")]
    pub fn get_many(
        &self,
        db: &Database,
        short_ids: &[u32],
    ) -> Result<(Option<u64>, Vec<Option<Vec<u8>>>), StorageError> {
        for _ in 0..MAX_ATTEMPTS {
            let Some(snapshot) = self.snapshot(db)? else {
                return Ok((None, vec![None; short_ids.len()]));
            };
            let head = &snapshot.head;
            let wanted: Vec<(usize, u32)> = short_ids
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, id)| snapshot.coverage(*id) == ClosureCoverage::Complete)
                .collect();
            let mut blobs: Vec<Option<Vec<u8>>> = vec![None; short_ids.len()];
            if wanted.is_empty() {
                return Ok((Some(head.generation), blobs));
            }
            let txn = db.begin_transaction();
            let ids: Vec<NodeId> = wanted.iter().map(|(_, id)| record_id(*id)).collect();
            let (records, _) = txn.get_with_record_versions(
                self.pool,
                &self.generation_collection(head.generation),
                &ids,
            )?;
            for ((slot, _short_id), record) in wanted.iter().zip(records) {
                if let Some(record) = record {
                    blobs[*slot] = Some(decode_record(&record.bytes)?);
                }
            }
            if wanted.iter().all(|(slot, _)| blobs[*slot].is_some()) {
                return Ok((Some(head.generation), blobs));
            }
            // A complete id with no record: either this generation was retired
            // after the head read, or the head is lying about its own coverage.
            let (again, _) = self.read_head(&db.begin_transaction())?;
            if again.map(|h| h.generation) == Some(head.generation) {
                let absent: Vec<String> = wanted
                    .iter()
                    .filter(|(slot, _)| blobs[*slot].is_none())
                    .map(|(_, id)| id.to_string())
                    .collect();
                return Err(StorageError::Corrupt(format!(
                    "closure generation {} records ids {absent:?} as complete but stores no \
                     closure for them",
                    head.generation
                )));
            }
        }
        Err(StorageError::Internal(
            "closure head kept moving during a read".to_owned(),
        ))
    }

    /// Read closure blobs for several ids from the generation a [`ClosureSnapshot`]
    /// is pinned to, taking the coverage classification from that same snapshot.
    ///
    /// This is the fast path for a reader that has already resolved a snapshot:
    /// it performs no head re-check and no retry, because absent is a legitimate
    /// answer and the snapshot already says which ids are absent by design.
    ///
    /// A blob missing for an id the snapshot calls
    /// [`ClosureCoverage::Complete`] is the one case that is not an answer but a
    /// fault. The store keeps only the published generation and its predecessor,
    /// so a snapshot held across two republishes reaches a generation that was
    /// retired out from under it. That is not corruption: it is reported as
    /// [`StorageError::StaleGeneration`], which tells the caller the right
    /// response is a fresh snapshot rather than a retry of this read. Corruption
    /// is reserved for a head that has not moved, where no such explanation
    /// exists.
    ///
    /// # Errors
    /// [`StorageError::StaleGeneration`] if the pinned generation is no longer
    /// retained, [`StorageError::Corrupt`] for a complete id with no record under
    /// an unmoved head or an undecodable record, otherwise a read failure.
    pub fn get_many_pinned(
        &self,
        db: &Database,
        snapshot: &ClosureSnapshot,
        short_ids: &[u32],
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let generation = snapshot.head.generation;
        let txn = db.begin_transaction();
        let ids: Vec<NodeId> = short_ids.iter().map(|id| record_id(*id)).collect();
        let (records, _) =
            txn.get_with_record_versions(self.pool, &self.generation_collection(generation), &ids)?;
        let mut blobs = Vec::with_capacity(records.len());
        for (short_id, record) in short_ids.iter().copied().zip(records) {
            match record {
                Some(data) => blobs.push(Some(decode_record(&data.bytes)?)),
                None if snapshot.coverage(short_id) != ClosureCoverage::Complete => {
                    blobs.push(None);
                }
                None => {
                    // Complete but unrecorded. Re-read the head: if it moved, this
                    // generation was retired out from under the reader, which is a
                    // stale generation rather than corruption. `StaleGeneration` is
                    // distinct from the record-level `StaleRead` on purpose: the
                    // response is a fresh snapshot, not a retry of this read.
                    let (again, _) = self.read_head(&db.begin_transaction())?;
                    let current = again.map(|fresh| fresh.generation);
                    if current != Some(generation) {
                        return Err(StorageError::StaleGeneration {
                            generation,
                            current,
                        });
                    }
                    return Err(StorageError::Corrupt(format!(
                        "closure generation {generation} records id {short_id} as complete but \
                         stores no closure for it"
                    )));
                }
            }
        }
        Ok(blobs)
    }

    /// Start building a new generation. Reserves its number so two builders
    /// can never write the same generation collection.
    ///
    /// # Errors
    /// Returns an error on a read or commit failure, or a corrupt counter.
    pub fn begin(&self, db: &Database) -> Result<GenerationBuilder, StorageError> {
        let mut last = None;
        for _ in 0..MAX_ATTEMPTS {
            let txn = db.begin_transaction();
            let (records, tokens) =
                txn.get_with_record_versions(self.pool, &self.head_collection(), &[COUNTER_ID])?;
            let counter = match records.into_iter().next().flatten() {
                Some(record) => decode_counter(&record.bytes)?,
                None => GenerationCounter {
                    next: 1,
                    retired_through: 0,
                },
            };
            let generation = counter.next;
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
                &NodeData::new(Bytes::from(encode_counter(GenerationCounter {
                    next,
                    retired_through: counter.retired_through,
                }))),
            )?;
            match txn.commit() {
                Ok(()) => {
                    return Ok(GenerationBuilder {
                        store: *self,
                        generation,
                        base,
                        head_token,
                        stored: Vec::new(),
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
    /// The head's skipped count must match its coverage record, every skipped id
    /// must fall inside `1..source_next`, and every id in that range the coverage
    /// record does not list as skipped must have a decodable record. The reverse
    /// holds too: a skipped id must have no record. A head that under-claims, by
    /// listing fewer skipped ids than it counts or by omitting a real gap, shows
    /// up as a covered id with no record.
    ///
    /// # Errors
    /// Returns an error only when a record cannot be read.
    pub fn verify(&self, db: &Database) -> Result<ClosureVerifyReport, StorageError> {
        let mut report = ClosureVerifyReport::default();
        // A corrupt head or coverage record is a finding, not an abort: `verify`
        // exists to report exactly that, and callers want the structured report
        // rather than an error they would have to unwrap.
        let snapshot = match self.snapshot(db) {
            Ok(snapshot) => snapshot,
            Err(error) if is_malformed(&error) => {
                report.problems.push(format!("closure snapshot: {error}"));
                return Ok(report);
            }
            Err(error) => return Err(error),
        };
        let Some(snapshot) = snapshot else {
            return Ok(report);
        };
        let head = &snapshot.head;
        let txn = db.begin_transaction();
        let collection = self.generation_collection(head.generation);
        for &(start, end) in snapshot.skipped.runs() {
            if start == 0 || end < start || end >= head.source_next {
                report.problems.push(format!(
                    "skipped run {start}..={end} falls outside 1..{}",
                    head.source_next
                ));
            }
        }
        for short_id in 1..head.source_next {
            let (records, _) =
                txn.get_with_record_versions(self.pool, &collection, &[record_id(short_id)])?;
            match (
                snapshot.skipped.contains(short_id),
                records.into_iter().next().flatten(),
            ) {
                (false, Some(record)) => match decode_record(&record.bytes) {
                    Ok(_) => {
                        report.records_checked = report.records_checked.saturating_add(1);
                    }
                    Err(error) => report.problems.push(format!(
                        "closure record for id {short_id} is unreadable: {error}"
                    )),
                },
                (false, None) => report
                    .problems
                    .push(format!("covered id {short_id} has no closure record")),
                (true, Some(_)) => report.problems.push(format!(
                    "id {short_id} is recorded as skipped but has a closure record"
                )),
                (true, None) => {}
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
    pub fn retire_superseded(&self, db: &Database) -> Result<u64, StorageError> {
        let txn = db.begin_transaction();
        let (head, _) = self.read_head(&txn)?;
        let (records, tokens) =
            txn.get_with_record_versions(self.pool, &self.head_collection(), &[COUNTER_ID])?;
        let Some(record) = records.into_iter().next().flatten() else {
            return Ok(0);
        };
        let counter = decode_counter(&record.bytes)?;
        // Only generations below the head are safe to retire. One at or above
        // it may belong to a builder that `begin` reserved but has not yet
        // published; deleting that would leave the head pointing at a deleted
        // collection. With no head yet, retain the newest reservation and
        // reclaim only older reservations left by crashed builders.
        let (end, predecessor) = match head {
            Some(head) => (head.generation.saturating_sub(1), Some(head.previous)),
            None => (counter.next.saturating_sub(2), None),
        };
        // The cursor names the predecessor retained by the last pass. When a
        // newer head exists, that predecessor is now safe to delete and the
        // cursor can move to the new head's predecessor in the same CAS.
        let cursor_limit = predecessor.filter(|&previous| previous != 0).unwrap_or(end);
        if counter.retired_through == cursor_limit {
            return Ok(0);
        }
        if counter.retired_through > cursor_limit {
            return Err(StorageError::Corrupt(
                "closure generation retirement cursor".to_owned(),
            ));
        }
        let old_cursor = counter.retired_through;
        let mut removed = 0u64;
        if old_cursor != 0 && predecessor != Some(0) && predecessor != Some(old_cursor) {
            let collection = self.generation_collection(old_cursor);
            if db.pool(self.pool).try_collection_exists(&collection)? {
                txn.delete_collection(self.pool, collection)?;
                removed = removed.saturating_add(1);
            }
        }
        for generation in old_cursor.saturating_add(1)..=end {
            if predecessor == Some(generation) {
                continue;
            }
            let collection = self.generation_collection(generation);
            if db.pool(self.pool).try_collection_exists(&collection)? {
                txn.delete_collection(self.pool, collection)?;
                removed = removed.saturating_add(1);
            }
        }
        let retired_through =
            predecessor.map_or(end, |previous| if previous == 0 { end } else { previous });
        txn.expect_record_version(self.pool, self.head_collection(), COUNTER_ID, tokens[0])?;
        txn.put(
            self.pool,
            self.head_collection(),
            COUNTER_ID,
            &NodeData::new(Bytes::from(encode_counter(GenerationCounter {
                retired_through,
                ..counter
            }))),
        )?;
        crash_point("retire-before-commit");
        txn.commit()?;
        crash_point("retire-after-commit");
        Ok(removed)
    }

    /// Read one record straight from a generation's collection, bypassing the
    /// head, so tests can prove exactly how far a crashed step got.
    /// The raw coverage record of `generation`, bypassing the head, so tests can
    /// assert whether one was written at all.
    #[cfg(test)]
    pub(crate) fn raw_coverage_record(&self, db: &Database, generation: u64) -> Option<Vec<u8>> {
        let txn = db.begin_transaction();
        let (records, _) = txn
            .get_with_record_versions(
                self.pool,
                &self.generation_collection(generation),
                &[coverage_id()],
            )
            .unwrap();
        records
            .into_iter()
            .next()
            .flatten()
            .map(|record| record.bytes.to_vec())
    }

    #[cfg(test)]
    pub(crate) fn raw_generation_record(
        &self,
        db: &Database,
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
            for generation in 1..decode_counter(&record.bytes)?.next {
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
    /// Ids a closure was written for, ascending. Kept so [`Self::publish`] can
    /// reject an id that is claimed as both stored and skipped; a builder is
    /// short-lived, so this costs one `u32` per event only while publishing.
    stored: Vec<u32>,
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
    pub fn add(&mut self, db: &Database, closures: &[(u32, &[u8])]) -> Result<(), StorageError> {
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
        for (short_id, _) in closures {
            let at = self.stored.partition_point(|&seen| seen < *short_id);
            if self.stored.get(at) != Some(short_id) {
                self.stored.insert(at, *short_id);
            }
        }
        Ok(())
    }

    /// Number of closure records written so far.
    #[must_use]
    pub fn count(&self) -> u32 {
        u32::try_from(self.stored.len()).unwrap_or(u32::MAX)
    }

    /// Atomically publish this generation as the head, covering ids
    /// `1..source_next`.
    ///
    /// `skipped` names the covered ids left without a record because their walk
    /// was incomplete. It is sorted, deduplicated and stored run-encoded as one
    /// coverage record inside the generation collection, in the same transaction
    /// that swaps the head. A reader therefore never sees a head that points at
    /// a generation whose coverage record is absent or stale, and the head
    /// itself stays a fixed 28 bytes however holey the room is.
    ///
    /// Fails with `StorageError::StaleRead` if the head moved since
    /// [`ClosureStore::begin`].
    ///
    /// # Errors
    /// `StaleRead` on a lost race, `Internal` if `skipped` is out of range or
    /// overlaps a stored closure, otherwise a commit failure.
    pub fn publish(
        self,
        db: &Database,
        source_next: u32,
        skipped: &[u32],
    ) -> Result<ClosureHead, StorageError> {
        let mut sorted = skipped.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        self.publish_with_coverage(db, source_next, &ClosureCoverageSet::from_ids(&sorted))
    }

    /// As [`Self::publish`], but taking the coverage set directly.
    ///
    /// A caller that discovers skipped ids one at a time should push them into a
    /// [`ClosureCoverageBuilder`] as it goes and pass the finished set here,
    /// rather than collecting a `Vec<u32>` for [`Self::publish`] to copy and sort
    /// on top of the runs the set already stores. The set must already be sorted
    /// and merged, which [`ClosureCoverageBuilder::build`] guarantees.
    ///
    /// # Errors
    /// As [`Self::publish`].
    pub fn publish_with_coverage(
        self,
        db: &Database,
        source_next: u32,
        set: &ClosureCoverageSet,
    ) -> Result<ClosureHead, StorageError> {
        crash_point("before-publish");
        // Every validation runs over the runs themselves, not an expanded id list:
        // a skipped run can cover thousands of ids, and the point of the run
        // encoding is to not touch them.
        //
        // All of it happens before the first write, and any failure deletes the
        // records already staged by `add`. A refused publish would otherwise leave
        // a generation collection that no head names, holding the failed attempt
        // until `retire_superseded` happened to reclaim it.
        let validated = validate_publish(&self.stored, set, source_next);
        let (stored_count, skipped_count, payload) = match validated {
            Ok(validated) => validated,
            Err(error) => {
                if let Err(cleanup) = self.discard_generation(db) {
                    eprintln!(
                        "closure publish: could not discard the rejected generation: {cleanup}"
                    );
                }
                return Err(error);
            }
        };
        let head = ClosureHead {
            generation: self.generation,
            previous: self.base.map_or(0, |base| base.generation),
            source_next,
            count: stored_count,
            skipped_count,
        };
        let metadata = encode_head_metadata(&head);
        let value =
            LogicalHeadValue::new(self.store.generation_collection(self.generation), metadata);
        let txn = db.begin_transaction();
        let collection = self.store.generation_collection(self.generation);
        // Written before the head swap and committed with it, so the coverage
        // record and the head that names it become visible together. A
        // generation with no skipped ids writes no record at all; a head that
        // then claims `skipped_count > 0` is a mismatch `verify` reports.
        if !set.is_empty() {
            if let Err(error) = txn.put(
                self.store.pool,
                collection,
                coverage_id(),
                &NodeData::new(Bytes::from(payload)),
            ) {
                drop(txn);
                if let Err(cleanup) = self.discard_generation(db) {
                    eprintln!("closure publish: could not discard failed generation: {cleanup}");
                }
                return Err(StorageError::Io(error));
            }
        }
        self.store
            .heads()
            .stage_replace(&txn, &HEAD_LOGICAL_ID, self.head_token, &value)?;
        crash_point("publish-before-commit");
        if let Err(error) = txn.commit() {
            if error.is_stale_read() {
                if let Err(cleanup) = self.discard_generation(db) {
                    eprintln!("closure publish: could not discard stale generation: {cleanup}");
                }
            }
            return Err(error);
        }
        crash_point("publish-after-commit");
        Ok(head)
    }

    /// Delete this generation's collection, whether or not anything was staged.
    fn discard_generation(&self, db: &Database) -> Result<(), StorageError> {
        let txn = db.begin_transaction();
        txn.delete_collection(
            self.store.pool,
            self.store.generation_collection(self.generation),
        )?;
        txn.commit()
    }

    /// Discard an unpublished generation.
    ///
    /// # Errors
    /// Returns an error if the delete cannot be committed.
    pub fn abandon(self, db: &Database) -> Result<(), StorageError> {
        self.discard_generation(db)
    }
}

/// Test hook namespace, so tests reach these without importing every item.
#[cfg(test)]
pub(crate) mod test {
    pub(crate) use super::{arm_snapshot_hook, disarm_snapshot_hook, take_head_reads};

    pub(crate) fn coverage_id() -> super::NodeId {
        super::coverage_id()
    }

    pub(crate) fn generation_collection(store: &super::ClosureStore, generation: u64) -> [u8; 16] {
        store.generation_collection(generation)
    }

    pub(crate) fn encode_coverage_for_test(set: &super::ClosureCoverageSet) -> Vec<u8> {
        super::encode_coverage(set)
    }

    pub(crate) fn generation_exists(
        store: &super::ClosureStore,
        db: &super::Database,
        generation: u64,
        short_ids: &[u32],
    ) -> Result<bool, super::StorageError> {
        let txn = db.begin_transaction();
        let mut ids: Vec<super::NodeId> =
            short_ids.iter().map(|id| super::record_id(*id)).collect();
        ids.push(super::coverage_id());
        let (records, _) = txn.get_with_record_versions(
            store.pool,
            &store.generation_collection(generation),
            &ids,
        )?;
        Ok(records.into_iter().any(|record| record.is_some()))
    }

    #[cfg(feature = "bitmaps")]
    pub(crate) fn generation_record(
        store: &super::ClosureStore,
        db: &super::Database,
        generation: u64,
        short_id: u32,
    ) -> Result<Option<Vec<u8>>, super::StorageError> {
        let txn = db.begin_transaction();
        let (records, _) = txn.get_with_record_versions(
            store.pool,
            &store.generation_collection(generation),
            &[super::record_id(short_id)],
        )?;
        records
            .into_iter()
            .next()
            .flatten()
            .map(|record| super::decode_record(&record.bytes))
            .transpose()
    }
}
