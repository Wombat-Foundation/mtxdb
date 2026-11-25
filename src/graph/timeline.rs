//! Persisted, ordered timeline index with copy-on-write roots.
//!
//! The packfile index maps node id -> frame and has no key order, so it cannot
//! answer a range read over a room's timeline. This module adds a **derived**
//! ordered index built from ordinary records, in the same spirit as
//! [`crate::auxiliary::AuxiliaryIndex`]: it is rebuildable, packfiles stay
//! authoritative, and event payloads are never duplicated — a leaf stores only
//! the ordering key plus the event's node id; the caller fetches the payload.
//!
//! The persisted key is the total order
//!
//! ```text
//! (room_id, topological_ordering, stream_ordering, node_id, event_ref)
//! ```
//!
//! This mirrors Synapse's `events_order_room (room_id, topological_ordering,
//! stream_ordering)` history order.
//!
//! - `room_id` scopes the index; a timeline read is a room-prefix range.
//! - `topological_ordering` (Matrix: depth) and `stream_ordering` are
//!   caller-supplied, signed 64-bit and **persisted**; the index never
//!   recomputes them. `stream_ordering` is signed because Synapse gives
//!   backfilled events negative stream positions. The caller must define a
//!   deterministic fallback for events without a computable position (missing
//!   parents, cycles) before building, because the in-memory DAG order cannot
//!   be regenerated reliably after reopening.
//! - `node_id` is the operational 128-bit id of the stored event and the
//!   tie-break after the two orderings, so equal positions never tie.
//! - `event_ref` is a caller-supplied 32-byte event identity (reference hash),
//!   kept as the final disambiguator so two ids for the same content stay
//!   distinct and ordered deterministically.
//!
//! **Prototype.** This is a copy-on-write leaf index, not the production
//! ordered-run format (memtable, sorted runs, fences, blooms, compaction); see
//! `docs/docs/2026-09-30-ordered-index-reconciliation.md`. The `TML*` formats
//! are not stable.
//!
//! Layout (all content-addressed, immutable except the head):
//! - **leaf** (`TMLF`): up to [`TIMELINE_LEAF_CAP`] sorted entries;
//! - **directory** (`TMLD`): up to [`TIMELINE_DIRECTORY_FANOUT`] leaf
//!   descriptors with minimum, maximum, and entry count;
//! - **root** (`TMLR`): an ordered directory manifest plus total entry count;
//! - **head** (`TMLH`): a fixed-id record holding the current root id, the only
//!   mutable id (last write wins).
//!
//! A rebuild writes new leaves and a new root and then swaps the head; old
//! records survive until GC, so a [`TimelineCursor`] that pins the root it
//! started on keeps paginating that snapshot even after an update. Forward
//! paging returns keys strictly greater than the cursor, reverse strictly less,
//! so resuming is exact and idempotent in both directions.

use std::borrow::Cow;

use bytes::Bytes;

use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    MEMBER_NAMESPACE_INTL,
};

/// Derive a 16-byte scope for [`TimelineEntry`] with domain-separated BLAKE3 over the room ID.
#[must_use]
pub fn room_scope_id(room_id: &[u8]) -> [u8; 16] {
    crate::interop::derive_id(b"timeline-room-scope", room_id)
}

/// Hash full event ID into a 32-byte reference for [`TimelineEntry`].
#[must_use]
pub fn event_id_ref(event_id: &[u8]) -> EventRef {
    *blake3::hash(event_id).as_bytes()
}

/// Version byte for timeline records and cursors.
pub const TIMELINE_FORMAT_VERSION: u8 = 0x01;
/// Maximum entries per leaf record.
pub const TIMELINE_LEAF_CAP: usize = 128;
/// Maximum leaf descriptors per directory record.
pub const TIMELINE_DIRECTORY_FANOUT: usize = 256;

/// Magic prefix of a leaf record (`b"TMLF"`).
pub const TIMELINE_LEAF_MAGIC: [u8; 4] = *b"TMLF";
/// Magic prefix of a root manifest record (`b"TMLR"`).
pub const TIMELINE_ROOT_MAGIC: [u8; 4] = *b"TMLR";
/// Magic prefix of a directory record (`b"TMLD"`).
pub const TIMELINE_DIRECTORY_MAGIC: [u8; 4] = *b"TMLD";
/// Magic prefix of the head record (`b"TMLH"`).
pub const TIMELINE_HEAD_MAGIC: [u8; 4] = *b"TMLH";
/// Magic prefix of an opaque cursor (`b"TMLC"`).
pub const TIMELINE_CURSOR_MAGIC: [u8; 4] = *b"TMLC";

/// Reserved node id of the head record (the only mutable id in the index).
pub const TIMELINE_HEAD_ID: NodeId = *b"MTXD-TML-HEAD-v1";

/// A caller-supplied event identity (reference hash).
pub type EventRef = [u8; 32];

const ENTRY_LEN: usize = 16 + 8 + 8 + 16 + 32;
const LEAF_HEADER_LEN: usize = 7;
const SPAN_DESCRIPTOR_LEN: usize = ENTRY_LEN * 2 + 8 + 16;
const DIRECTORY_HEADER_LEN: usize = 8;
const ROOT_HEADER_LEN: usize = 18;
const ROOT_DIRECTORY_LEN: usize = SPAN_DESCRIPTOR_LEN;
const DIRECTORY_LEAF_LEN: usize = SPAN_DESCRIPTOR_LEN;
const LEAF_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const ROOT_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const DIRECTORY_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const HEAD_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const CURSOR_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const DIRECTORY_LEVEL: u8 = 1;
const ROOT_HEIGHT: u8 = 2;
const HEAD_LEN: usize = 5 + 16 + 8;
const CURSOR_LEN: usize = 5 + 16 + ENTRY_LEN;
const MAX_NODE_ID: NodeId = [0xff; 16];
const MAX_EVENT_REF: EventRef = [0xff; 32];

/// Order-preserving big-endian encoding of a signed position: flipping the
/// sign bit makes byte order equal numeric order, so the wire key is
/// byte-comparable.
fn encode_position(value: i64) -> [u8; 8] {
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    bytes
}

fn decode_position(mut bytes: [u8; 8]) -> i64 {
    bytes[0] ^= 0x80;
    i64::from_be_bytes(bytes)
}

/// One ordered entry: the full
/// `(room, topological_ordering, stream_ordering, node, event_ref)` key.
///
/// Field order is the sort order (`derive(Ord)` compares in declaration order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimelineEntry {
    /// Room scope; the leading key component.
    pub room_id: [u8; 16],
    /// Caller-supplied topological ordering (Matrix: depth).
    pub topological_ordering: i64,
    /// Caller-supplied stream ordering; negative for backfilled events.
    pub stream_ordering: i64,
    /// The event's operational 128-bit node id; tie-break after the orderings.
    pub node_id: NodeId,
    /// Caller-supplied 32-byte event identity; final disambiguator.
    pub event_ref: EventRef,
}

impl TimelineEntry {
    /// Construct an entry.
    #[must_use]
    pub const fn new(
        room_id: [u8; 16],
        topological_ordering: i64,
        stream_ordering: i64,
        node_id: NodeId,
        event_ref: EventRef,
    ) -> Self {
        Self {
            room_id,
            topological_ordering,
            stream_ordering,
            node_id,
            event_ref,
        }
    }

    fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.room_id);
        out.extend_from_slice(&encode_position(self.topological_ordering));
        out.extend_from_slice(&encode_position(self.stream_ordering));
        out.extend_from_slice(&self.node_id);
        out.extend_from_slice(&self.event_ref);
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        if bytes.len() != ENTRY_LEN {
            return Err(StorageError::Corrupt("timeline entry length".to_owned()));
        }
        let position = |range: std::ops::Range<usize>, what: &str| {
            <[u8; 8]>::try_from(&bytes[range])
                .map(decode_position)
                .map_err(|_| StorageError::Corrupt(format!("timeline entry {what}")))
        };
        let mut room_id = [0u8; 16];
        room_id.copy_from_slice(&bytes[..16]);
        let topological_ordering = position(16..24, "topological ordering")?;
        let stream_ordering = position(24..32, "stream ordering")?;
        let mut node_id = [0u8; 16];
        node_id.copy_from_slice(&bytes[32..48]);
        let mut event_ref = [0u8; 32];
        event_ref.copy_from_slice(&bytes[48..ENTRY_LEN]);
        Ok(Self {
            room_id,
            topological_ordering,
            stream_ordering,
            node_id,
            event_ref,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanDescriptor {
    min_entry: TimelineEntry,
    max_entry: TimelineEntry,
    entry_count: u64,
    child_id: NodeId,
}

type LeafDescriptor = SpanDescriptor;
type DirectoryDescriptor = SpanDescriptor;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimelineRoot {
    height: u8,
    entry_count: u64,
    directories: Vec<DirectoryDescriptor>,
}

#[derive(Default)]
struct StagedTree {
    leaf_records: Vec<(NodeId, Vec<u8>)>,
    directory_records: Vec<(NodeId, Vec<u8>)>,
    inserted: u64,
}

#[derive(Clone, Copy)]
enum SplitMode {
    Tail,
    Front,
    Interior,
}

fn balanced_chunk_ranges(
    len: usize,
    cap: usize,
) -> Result<Vec<std::ops::Range<usize>>, StorageError> {
    if len == 0 || cap == 0 {
        return Ok(Vec::new());
    }
    let chunk_count = len.div_ceil(cap);
    let base = len
        .checked_div(chunk_count)
        .ok_or_else(|| StorageError::Internal("timeline chunk count is zero".to_owned()))?;
    let extra = len
        .checked_rem(chunk_count)
        .ok_or_else(|| StorageError::Internal("timeline chunk count is zero".to_owned()))?;
    let mut ranges = Vec::with_capacity(chunk_count);
    let mut start = 0_usize;
    for index in 0..chunk_count {
        let size = base
            .checked_add(usize::from(index < extra))
            .ok_or_else(|| StorageError::Internal("timeline chunk size overflow".to_owned()))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| StorageError::Internal("timeline chunk end overflow".to_owned()))?;
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

fn directory_chunk_ranges(
    len: usize,
    mode: SplitMode,
) -> Result<Vec<std::ops::Range<usize>>, StorageError> {
    match mode {
        SplitMode::Interior => balanced_chunk_ranges(len, TIMELINE_DIRECTORY_FANOUT),
        SplitMode::Tail | SplitMode::Front => {
            let cap = TIMELINE_DIRECTORY_FANOUT;
            if len == 0 {
                return Ok(Vec::new());
            }
            let remainder = len
                .checked_rem(cap)
                .ok_or_else(|| StorageError::Internal("directory fanout is zero".to_owned()))?;
            let first_size = match mode {
                SplitMode::Tail => cap,
                SplitMode::Front if remainder == 0 => cap,
                SplitMode::Front => remainder,
                SplitMode::Interior => unreachable!(),
            };
            let mut ranges = Vec::with_capacity(len.div_ceil(cap));
            let mut start = 0_usize;
            while start < len {
                let size = if start == 0 {
                    first_size
                } else {
                    cap.min(len.saturating_sub(start))
                };
                let end = start.checked_add(size).ok_or_else(|| {
                    StorageError::Internal("directory range end overflow".to_owned())
                })?;
                ranges.push(start..end);
                start = end;
            }
            Ok(ranges)
        }
    }
}

fn split_mode(
    siblings: &[DirectoryDescriptor],
    index: usize,
    span: &DirectoryDescriptor,
    entries: &[TimelineEntry],
) -> SplitMode {
    let all_front = entries.iter().all(|entry| *entry < span.min_entry);
    if index == 0 && all_front {
        return SplitMode::Front;
    }
    let no_same_room_after = siblings
        .get(index.saturating_add(1))
        .is_none_or(|next| next.min_entry.room_id != span.max_entry.room_id);
    let all_tail = entries.iter().all(|entry| *entry > span.max_entry);
    if no_same_room_after && all_tail {
        return SplitMode::Tail;
    }
    SplitMode::Interior
}

/// An opaque, stable cursor naming the snapshot root plus the last entry read.
///
/// It carries the full ordering key, so both directions resume exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineCursor {
    root_id: NodeId,
    entry: TimelineEntry,
}

impl TimelineCursor {
    /// Build a cursor for `root_id` at `entry`.
    #[must_use]
    pub const fn new(root_id: NodeId, entry: TimelineEntry) -> Self {
        Self { root_id, entry }
    }

    /// The snapshot root this cursor pins.
    #[must_use]
    pub const fn root_id(&self) -> NodeId {
        self.root_id
    }

    /// The entry this cursor names.
    #[must_use]
    pub const fn entry(&self) -> TimelineEntry {
        self.entry
    }

    /// Encode the cursor as opaque, versioned bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CURSOR_LEN);
        out.extend_from_slice(&TIMELINE_CURSOR_MAGIC);
        out.push(CURSOR_FORMAT_VERSION);
        out.extend_from_slice(&self.root_id);
        self.entry.encode_into(&mut out);
        out
    }

    /// Decode cursor bytes produced by [`Self::encode`].
    ///
    /// Returns `None` for a wrong magic, version, or length.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != CURSOR_LEN
            || bytes[..4] != TIMELINE_CURSOR_MAGIC
            || bytes[4] != CURSOR_FORMAT_VERSION
        {
            return None;
        }
        let mut root_id = [0u8; 16];
        root_id.copy_from_slice(&bytes[5..21]);
        let entry = TimelineEntry::decode(&bytes[21..CURSOR_LEN]).ok()?;
        Some(Self { root_id, entry })
    }
}

/// One page of a timeline read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelinePage {
    /// The entries in traversal order.
    pub entries: Vec<TimelineEntry>,
    /// Resume token, present when the page filled `limit`.
    pub next_cursor: Option<TimelineCursor>,
}

/// The timeline index collection id for a caller-chosen index name.
#[must_use]
pub fn timeline_collection_id(name: &str) -> [u8; 16] {
    let mut canonical = Vec::with_capacity(name.len().saturating_add(9));
    canonical.extend_from_slice(b"timeline:");
    canonical.extend_from_slice(name.as_bytes());
    derive_collection_id(Some(MEMBER_NAMESPACE_INTL), &canonical)
}

fn content_id(payload: &[u8]) -> NodeId {
    let digest = DigestAlgorithm::Blake3.digest(payload);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn merge_and_stage_leaves(
    existing: &[TimelineEntry],
    incoming: &[TimelineEntry],
    manifest: &mut Vec<LeafDescriptor>,
    inserted: &mut u64,
    mut stage_record: impl FnMut(NodeId, Vec<u8>) -> Result<(), StorageError>,
) -> Result<(), StorageError> {
    let mut merged = Vec::with_capacity(existing.len().saturating_add(incoming.len()));
    let (mut left, mut right) = (0, 0);
    while left < existing.len() || right < incoming.len() {
        let take_existing =
            right == incoming.len() || (left < existing.len() && existing[left] <= incoming[right]);
        let entry = if take_existing {
            let entry = existing[left];
            left = left.checked_add(1).ok_or_else(|| {
                StorageError::Internal("timeline merge offset overflow".to_owned())
            })?;
            if right < incoming.len() && incoming[right] == entry {
                right = right.checked_add(1).ok_or_else(|| {
                    StorageError::Internal("timeline merge offset overflow".to_owned())
                })?;
            }
            entry
        } else {
            let entry = incoming[right];
            right = right.checked_add(1).ok_or_else(|| {
                StorageError::Internal("timeline merge offset overflow".to_owned())
            })?;
            *inserted = inserted.checked_add(1).ok_or_else(|| {
                StorageError::Internal("timeline entry count overflow".to_owned())
            })?;
            entry
        };
        merged.push(entry);
    }

    for chunk in merged.chunks(TIMELINE_LEAF_CAP) {
        let payload = encode_leaf(chunk)?;
        let id = content_id(&payload);
        let first = *chunk
            .first()
            .ok_or_else(|| StorageError::Internal("empty timeline leaf".to_owned()))?;
        stage_record(id, payload)?;
        let last = *chunk
            .last()
            .ok_or_else(|| StorageError::Internal("empty timeline leaf".to_owned()))?;
        manifest.push(LeafDescriptor {
            min_entry: first,
            max_entry: last,
            entry_count: u64::try_from(chunk.len())
                .map_err(|_| StorageError::Internal("timeline leaf exceeds u64".to_owned()))?,
            child_id: id,
        });
    }
    Ok(())
}

fn build_directory_blocks(
    leaves: &[LeafDescriptor],
    level: u8,
    mode: SplitMode,
    mut write_record: impl FnMut(NodeId, Vec<u8>) -> Result<(), StorageError>,
) -> Result<Vec<DirectoryDescriptor>, StorageError> {
    let mut directories = Vec::with_capacity(leaves.len().div_ceil(TIMELINE_DIRECTORY_FANOUT));
    for range in directory_chunk_ranges(leaves.len(), mode)? {
        let block = &leaves[range];
        let payload = encode_directory(block, level)?;
        let directory_id = content_id(&payload);
        let first = block
            .first()
            .ok_or_else(|| StorageError::Internal("empty timeline directory".to_owned()))?;
        let last = block
            .last()
            .ok_or_else(|| StorageError::Internal("empty timeline directory".to_owned()))?;
        let entry_count = block.iter().try_fold(0u64, |count, leaf| {
            count.checked_add(leaf.entry_count).ok_or_else(|| {
                StorageError::Internal("timeline directory count overflow".to_owned())
            })
        })?;
        write_record(directory_id, payload)?;
        directories.push(DirectoryDescriptor {
            min_entry: first.min_entry,
            max_entry: last.max_entry,
            entry_count,
            child_id: directory_id,
        });
    }
    Ok(directories)
}

fn room_start(room_id: &[u8; 16]) -> TimelineEntry {
    TimelineEntry::new(*room_id, i64::MIN, i64::MIN, [0u8; 16], [0u8; 32])
}

fn room_end(room_id: &[u8; 16]) -> TimelineEntry {
    TimelineEntry::new(*room_id, i64::MAX, i64::MAX, MAX_NODE_ID, MAX_EVENT_REF)
}

/// A persisted timeline index over an existing storage engine.
pub struct TimelineIndex<'a, S: StorageEngine + ?Sized> {
    engine: &'a S,
    name: Cow<'a, str>,
    collection_id: [u8; 16],
}

impl<'a, S: StorageEngine + ?Sized> TimelineIndex<'a, S> {
    /// Open (or create) the named timeline index.
    #[must_use]
    pub fn open(engine: &'a S, name: impl Into<Cow<'a, str>>) -> Self {
        let name = name.into();
        let collection_id = timeline_collection_id(name.as_ref());
        Self::with_collection_id(engine, name, collection_id)
    }

    /// Open with a pre-computed collection ID and pre-owned or borrowed name.
    #[must_use]
    pub fn with_collection_id(
        engine: &'a S,
        name: impl Into<Cow<'a, str>>,
        collection_id: [u8; 16],
    ) -> Self {
        Self {
            engine,
            name: name.into(),
            collection_id,
        }
    }

    /// This index's collection id.
    #[must_use]
    pub fn collection_id(&self) -> [u8; 16] {
        self.collection_id
    }

    /// The collection metadata describing this index.
    #[must_use]
    pub fn metadata(&self) -> CollectionMetadata {
        CollectionMetadata {
            member_namespace: Some(MEMBER_NAMESPACE_INTL),
            collection_canonical_id: format!("timeline:{}", self.name).into_bytes(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("timeline_index".to_owned()),
            schema: Some("mtxdb.timeline.v1".to_owned()),
        }
    }

    /// Ensure the index collection's metadata is established.
    ///
    /// # Errors
    /// Propagates the backend error, or [`StorageError::Internal`] on a
    /// metadata derivation mismatch.
    pub fn ensure_metadata(&self) -> Result<(), StorageError> {
        self.engine
            .create_or_put_established(&self.collection_id, &self.metadata(), &[])
    }

    /// Build (or rebuild) the index from `entries`, returning the new root id.
    ///
    /// Entries are sorted and de-duplicated by the full key. Identical inputs
    /// produce identical content-addressed records, so a replay is idempotent;
    /// only the head pointer changes between builds. Old roots stay readable,
    /// so cursors pinned to them keep working.
    ///
    /// # Errors
    /// Propagates the backend error, or [`StorageError::Internal`] if a count
    /// exceeds the record layout.
    pub fn build(&self, entries: &[TimelineEntry]) -> Result<NodeId, StorageError> {
        self.ensure_metadata()?;
        let mut sorted = entries.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut leaves = Vec::new();
        for chunk in sorted.chunks(TIMELINE_LEAF_CAP) {
            let payload = encode_leaf(chunk)?;
            let id = content_id(&payload);
            self.write_record(&id, payload)?;
            let first = *chunk
                .first()
                .ok_or_else(|| StorageError::Internal("empty timeline leaf".to_owned()))?;
            let last = *chunk
                .last()
                .ok_or_else(|| StorageError::Internal("empty timeline leaf".to_owned()))?;
            leaves.push(LeafDescriptor {
                min_entry: first,
                max_entry: last,
                entry_count: u64::try_from(chunk.len())
                    .map_err(|_| StorageError::Internal("timeline leaf exceeds u64".to_owned()))?,
                child_id: id,
            });
        }

        let count = u64::try_from(sorted.len()).unwrap_or(u64::MAX);
        let (height, directories) = if leaves.len() <= TIMELINE_DIRECTORY_FANOUT {
            (1, leaves)
        } else {
            (
                ROOT_HEIGHT,
                build_directory_blocks(
                    &leaves,
                    DIRECTORY_LEVEL,
                    SplitMode::Tail,
                    |id, payload| self.write_record(&id, payload),
                )?,
            )
        };
        let root_payload = encode_root(&directories, count, height)?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(&TIMELINE_HEAD_ID, encode_head(&root_id, count))?;
        Ok(root_id)
    }

    /// Append entries in one optimistic transaction. The caller supplies the
    /// database and pool that back this index. The collection's logical
    /// version is checked at commit, so concurrent appends cannot silently
    /// overwrite one another.
    ///
    /// Returns the resulting root id. Entries are sorted and deduplicated by
    /// the full key; an empty input or only duplicates leaves an existing head
    /// unchanged. An index without a head is initialized even for empty input.
    /// This convenience method opens and commits its own transaction. Use
    /// [`Self::stage_append`] to combine the index update with other writes.
    ///
    /// # Errors
    /// Propagates errors from [`Self::stage_append`] and transaction commit,
    /// including [`StorageError::StaleRead`] if another writer changes a head
    /// guarded by this transaction before commit.
    pub fn append(
        &self,
        database: &Database,
        pool: ShardType,
        new_entries: &[TimelineEntry],
    ) -> Result<NodeId, StorageError> {
        let transaction = database.begin_transaction();
        let root_id = self.stage_append(&transaction, pool, new_entries)?;
        transaction.commit()?;
        Ok(root_id)
    }

    /// Stage entries into the current index in `transaction`, preserving old
    /// roots for existing cursors. Only leaves whose key ranges are touched by
    /// `new_entries` are read and rewritten; untouched leaf records are reused.
    /// The root manifest is rewritten because it lists every leaf. `pool`
    /// must select the engine backing this index.
    ///
    /// Duplicate entries are ignored by the full key, both within the input
    /// and against the existing index. If every entry is a duplicate, the
    /// current root is returned without changing the head. Empty input also
    /// returns an existing root; without a head, an empty index is staged.
    /// Returns the resulting root id. Collection metadata is ensured through
    /// the index's storage engine before staging, so that idempotent
    /// metadata write is outside this transaction. The caller must commit the
    /// transaction to publish the leaves, new root, and head together.
    /// Repeated calls in one transaction compose: each observes the staged
    /// head and reuses the same collection-version precondition.
    ///
    /// # Errors
    /// Propagates backend errors, or returns [`StorageError::Corrupt`] for
    /// malformed existing records, [`StorageError::NotFound`] for a missing
    /// referenced root or touched leaf, and [`StorageError::Internal`] if the
    /// pool differs from this index's engine or a count exceeds the layout.
    /// A concurrent head change can cause [`StorageError::StaleRead`] when
    /// the caller commits the transaction.
    pub fn stage_append(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        new_entries: &[TimelineEntry],
    ) -> Result<NodeId, StorageError> {
        let engine_identity = std::ptr::from_ref(self.engine).cast::<()>() as usize;
        if engine_identity != transaction.pool_identity(pool) {
            return Err(StorageError::Internal(
                "timeline index and transaction use different storage pools".to_owned(),
            ));
        }
        self.ensure_metadata()?;
        let (head_records, collection_version) = transaction.get_with_collection_version(
            pool,
            &self.collection_id,
            &[TIMELINE_HEAD_ID],
        )?;
        let head_record = head_records.first().ok_or_else(|| {
            StorageError::Internal("timeline head read returned no slot".to_owned())
        })?;
        let (old_root_id, old_count, root) = if let Some(head) = head_record {
            let (root_id, count) = decode_head(&head.bytes)?;
            let root_records = transaction.get(pool, &self.collection_id, &[root_id])?;
            let root_record = root_records
                .first()
                .and_then(Option::as_ref)
                .ok_or(StorageError::NotFound(root_id))?;
            let root = decode_root(&root_record.bytes)?;
            if root.entry_count != count {
                return Err(StorageError::Corrupt(
                    "timeline head/root count mismatch".to_owned(),
                ));
            }
            (Some(root_id), count, root)
        } else {
            (
                None,
                0,
                TimelineRoot {
                    height: 0,
                    entry_count: 0,
                    directories: Vec::new(),
                },
            )
        };
        if new_entries.is_empty() {
            if let Some(root_id) = old_root_id {
                return Ok(root_id);
            }
        }

        let mut pending = new_entries.to_vec();
        pending.sort_unstable();
        pending.dedup();
        let mut staged = StagedTree::default();
        let next_directories =
            self.stage_directory_tree(transaction, pool, &root, &pending, &mut staged)?;
        if staged.inserted == 0 {
            if let Some(root_id) = old_root_id {
                return Ok(root_id);
            }
        }

        // The precondition is the collection's logical version rather than
        // the head's record token: a record token read through the
        // transaction is the synthetic `0` for a head an earlier
        // `stage_append` in this transaction already staged, while the
        // collection version is live and immune to this transaction's own
        // staged writes. Repeated calls therefore reuse the same
        // precondition, and a concurrent writer still makes the commit
        // stale.
        transaction
            .expect_collection_version(pool, self.collection_id, collection_version)
            .map_err(StorageError::Io)?;

        for (leaf_id, payload) in staged.leaf_records {
            self.stage_record(transaction, pool, &leaf_id, payload)?;
        }
        for (directory_id, payload) in staged.directory_records {
            self.stage_record(transaction, pool, &directory_id, payload)?;
        }

        let count = old_count
            .checked_add(staged.inserted)
            .ok_or_else(|| StorageError::Internal("timeline entry count overflow".to_owned()))?;
        let root_payload = encode_root(&next_directories, count, root_height(&next_directories))?;
        let root_id = content_id(&root_payload);
        self.stage_record(transaction, pool, &root_id, root_payload)?;
        self.stage_record(
            transaction,
            pool,
            &TIMELINE_HEAD_ID,
            encode_head(&root_id, count),
        )?;
        Ok(root_id)
    }

    fn stage_directory_tree(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        root: &TimelineRoot,
        pending: &[TimelineEntry],
        staged: &mut StagedTree,
    ) -> Result<Vec<DirectoryDescriptor>, StorageError> {
        if root.directories.is_empty() {
            let mut initial_leaves = Vec::new();
            Self::stage_merged_leaves(
                &[],
                pending,
                &mut initial_leaves,
                &mut staged.inserted,
                &mut staged.leaf_records,
            )?;
            return build_directory_blocks(
                &initial_leaves,
                DIRECTORY_LEVEL,
                SplitMode::Tail,
                |id, payload| {
                    staged.directory_records.push((id, payload));
                    Ok(())
                },
            );
        }

        let mut assigned = vec![Vec::new(); root.directories.len()];
        for entry in pending {
            let index = root
                .directories
                .partition_point(|directory| directory.min_entry <= *entry)
                .saturating_sub(1);
            assigned[index].push(*entry);
        }

        let mut next_directories = Vec::with_capacity(root.directories.len());
        for (directory_index, descriptor) in root.directories.iter().enumerate() {
            if assigned[directory_index].is_empty() {
                next_directories.push(*descriptor);
                continue;
            }
            let changed = self.stage_directory(
                transaction,
                pool,
                descriptor,
                split_mode(
                    &root.directories,
                    directory_index,
                    descriptor,
                    &assigned[directory_index],
                ),
                &assigned[directory_index],
                staged,
            )?;
            next_directories.extend(changed);
        }
        Ok(next_directories)
    }

    fn stage_directory(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        descriptor: &DirectoryDescriptor,
        mode: SplitMode,
        entries: &[TimelineEntry],
        staged: &mut StagedTree,
    ) -> Result<Vec<DirectoryDescriptor>, StorageError> {
        let directory_records =
            transaction.get(pool, &self.collection_id, &[descriptor.child_id])?;
        let directory_record = directory_records
            .first()
            .and_then(Option::as_ref)
            .ok_or(StorageError::NotFound(descriptor.child_id))?;
        let old_leaves = decode_directory(&directory_record.bytes, DIRECTORY_LEVEL)?;
        let mut assigned_leaves = vec![Vec::new(); old_leaves.len()];
        for entry in entries {
            let leaf_index = old_leaves
                .partition_point(|leaf| leaf.min_entry <= *entry)
                .saturating_sub(1);
            assigned_leaves[leaf_index].push(*entry);
        }
        let mut next_leaves = Vec::new();
        for (leaf_index, leaf_descriptor) in old_leaves.iter().enumerate() {
            let incoming = &assigned_leaves[leaf_index];
            if incoming.is_empty() {
                next_leaves.push(*leaf_descriptor);
                continue;
            }
            let all_before = incoming
                .iter()
                .all(|entry| *entry < leaf_descriptor.min_entry);
            let all_after = incoming
                .iter()
                .all(|entry| *entry > leaf_descriptor.max_entry);
            if leaf_descriptor.entry_count == TIMELINE_LEAF_CAP as u64 && (all_before || all_after)
            {
                let mut inserted_leaves = Vec::new();
                Self::stage_merged_leaves(
                    &[],
                    incoming,
                    &mut inserted_leaves,
                    &mut staged.inserted,
                    &mut staged.leaf_records,
                )?;
                if all_before {
                    next_leaves.extend(inserted_leaves);
                    next_leaves.push(*leaf_descriptor);
                } else {
                    next_leaves.push(*leaf_descriptor);
                    next_leaves.extend(inserted_leaves);
                }
                continue;
            }
            let leaf_records =
                transaction.get(pool, &self.collection_id, &[leaf_descriptor.child_id])?;
            let leaf_record = leaf_records
                .first()
                .and_then(Option::as_ref)
                .ok_or(StorageError::NotFound(leaf_descriptor.child_id))?;
            let old_leaf = decode_leaf(&leaf_record.bytes)?;
            Self::stage_merged_leaves(
                &old_leaf,
                incoming,
                &mut next_leaves,
                &mut staged.inserted,
                &mut staged.leaf_records,
            )?;
        }
        build_directory_blocks(&next_leaves, DIRECTORY_LEVEL, mode, |id, payload| {
            staged.directory_records.push((id, payload));
            Ok(())
        })
    }

    /// Unconditionally append through the raw storage engine. This utility is
    /// useful for single-writer maintenance and in-memory tests; concurrent
    /// writers should use [`Self::append`] or [`Self::stage_append`].
    ///
    /// Returns the resulting root id, sorting and deduplicating entries by the
    /// full key. Empty input preserves an existing head; without a head, this
    /// builds an index even for empty input. Writes are published as they occur,
    /// so an error can leave new records stored before the head is updated.
    ///
    /// # Errors
    /// Propagates backend errors, or returns [`StorageError::Corrupt`] for
    /// malformed existing records, [`StorageError::NotFound`] for a missing
    /// referenced root or touched leaf, and [`StorageError::Internal`] if the
    /// resulting entry count cannot be represented.
    pub fn append_untransactional(
        &self,
        new_entries: &[TimelineEntry],
    ) -> Result<NodeId, StorageError> {
        self.ensure_metadata()?;
        let Some((old_root_id, old_count)) = self.head()? else {
            return self.build(new_entries);
        };
        if new_entries.is_empty() {
            return Ok(old_root_id);
        }

        let root_bytes = self
            .engine
            .get(&self.collection_id, &old_root_id)?
            .ok_or(StorageError::NotFound(old_root_id))?;
        let root = decode_root(&root_bytes.bytes)?;
        if root.entry_count != old_count {
            return Err(StorageError::Corrupt(
                "timeline head/root count mismatch".to_owned(),
            ));
        }
        let mut pending = new_entries.to_vec();
        pending.sort_unstable();
        pending.dedup();

        let mut inserted = 0u64;
        let mut next_directories = Vec::new();
        if root.directories.is_empty() {
            let mut leaves = Vec::new();
            self.write_merged_leaves(&[], &pending, &mut leaves, &mut inserted)?;
            next_directories = if leaves.len() <= TIMELINE_DIRECTORY_FANOUT {
                leaves
            } else {
                build_directory_blocks(&leaves, DIRECTORY_LEVEL, SplitMode::Tail, |id, payload| {
                    self.write_record(&id, payload)
                })?
            };
        } else {
            let mut assigned = vec![Vec::new(); root.directories.len()];
            for entry in pending {
                let index = root
                    .directories
                    .partition_point(|directory| directory.min_entry <= entry)
                    .saturating_sub(1);
                assigned[index].push(entry);
            }

            for (directory_index, descriptor) in root.directories.iter().enumerate() {
                if assigned[directory_index].is_empty() {
                    next_directories.push(*descriptor);
                    continue;
                }
                let mode = split_mode(
                    &root.directories,
                    directory_index,
                    descriptor,
                    &assigned[directory_index],
                );
                let old_leaves = self.load_directory(descriptor.child_id, DIRECTORY_LEVEL)?;
                let mut assigned_leaves = vec![Vec::new(); old_leaves.len()];
                for entry in &assigned[directory_index] {
                    let leaf_index = old_leaves
                        .partition_point(|leaf| leaf.min_entry <= *entry)
                        .saturating_sub(1);
                    assigned_leaves[leaf_index].push(*entry);
                }
                let mut next_leaves = Vec::new();
                for (leaf_index, leaf_descriptor) in old_leaves.iter().enumerate() {
                    let incoming = &assigned_leaves[leaf_index];
                    if incoming.is_empty() {
                        next_leaves.push(*leaf_descriptor);
                        continue;
                    }
                    let all_before = incoming
                        .iter()
                        .all(|entry| *entry < leaf_descriptor.min_entry);
                    let all_after = incoming
                        .iter()
                        .all(|entry| *entry > leaf_descriptor.max_entry);
                    if leaf_descriptor.entry_count == TIMELINE_LEAF_CAP as u64
                        && (all_before || all_after)
                    {
                        let mut inserted_leaves = Vec::new();
                        self.write_merged_leaves(
                            &[],
                            incoming,
                            &mut inserted_leaves,
                            &mut inserted,
                        )?;
                        if all_before {
                            next_leaves.extend(inserted_leaves);
                            next_leaves.push(*leaf_descriptor);
                        } else {
                            next_leaves.push(*leaf_descriptor);
                            next_leaves.extend(inserted_leaves);
                        }
                        continue;
                    }
                    let old_leaf = self.load_leaf(&leaf_descriptor.child_id)?;
                    self.write_merged_leaves(&old_leaf, incoming, &mut next_leaves, &mut inserted)?;
                }
                let mut changed_directories =
                    build_directory_blocks(&next_leaves, DIRECTORY_LEVEL, mode, |id, payload| {
                        self.write_record(&id, payload)
                    })?;
                if inserted == 0 {
                    next_directories.push(*descriptor);
                } else {
                    next_directories.append(&mut changed_directories);
                }
            }
        }

        let count = old_count
            .checked_add(inserted)
            .ok_or_else(|| StorageError::Internal("timeline entry count overflow".to_owned()))?;
        let root_payload = encode_root(&next_directories, count, root_height(&next_directories))?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(&TIMELINE_HEAD_ID, encode_head(&root_id, count))?;
        Ok(root_id)
    }

    /// Merge sorted, unique inputs, write the resulting leaves immediately,
    /// and append their first keys and IDs to `manifest`. Adds the number of
    /// entries absent from `existing` to `inserted`.
    ///
    /// # Errors
    /// Propagates write errors or returns [`StorageError::Internal`] on count
    /// or offset overflow. Earlier writes and output updates remain on error.
    fn write_merged_leaves(
        &self,
        existing: &[TimelineEntry],
        incoming: &[TimelineEntry],
        manifest: &mut Vec<LeafDescriptor>,
        inserted: &mut u64,
    ) -> Result<(), StorageError> {
        merge_and_stage_leaves(existing, incoming, manifest, inserted, |id, payload| {
            self.write_record(&id, payload)
        })
    }

    /// Merge sorted, unique inputs into encoded leaves in `pending_records`
    /// and append their first keys and IDs to `manifest`, without writing to
    /// storage. Adds entries absent from `existing` to the `inserted` count.
    ///
    /// # Errors
    /// Returns [`StorageError::Internal`] on count or offset overflow;
    /// earlier output updates remain on error.
    fn stage_merged_leaves(
        existing: &[TimelineEntry],
        incoming: &[TimelineEntry],
        manifest: &mut Vec<LeafDescriptor>,
        inserted: &mut u64,
        pending_records: &mut Vec<(NodeId, Vec<u8>)>,
    ) -> Result<(), StorageError> {
        merge_and_stage_leaves(existing, incoming, manifest, inserted, |id, payload| {
            pending_records.push((id, payload));
            Ok(())
        })
    }

    /// Stage a record in this index's collection for publication on commit.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] if the transaction is inactive or its
    /// staging budget is exceeded.
    fn stage_record(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        id: &NodeId,
        payload: Vec<u8>,
    ) -> Result<(), StorageError> {
        transaction
            .put(
                pool,
                self.collection_id,
                *id,
                &NodeData::new(Bytes::from(payload)),
            )
            .map_err(StorageError::Io)
    }

    /// The current head's `(root_id, entry_count)`, if the index was built.
    ///
    /// # Errors
    /// [`StorageError::Corrupt`] if the head record does not decode.
    pub fn head(&self) -> Result<Option<(NodeId, u64)>, StorageError> {
        match self.engine.get(&self.collection_id, &TIMELINE_HEAD_ID)? {
            None => Ok(None),
            Some(data) => decode_head(&data.bytes).map(Some),
        }
    }

    /// Read one page of `room_id` in `forward` (ascending) or reverse order.
    ///
    /// Forward returns keys strictly greater than `cursor`; reverse strictly
    /// less. `None` starts at the room's first (forward) or last (reverse)
    /// entry and reads the current head; a `Some` cursor reads the root it
    /// pins, so pagination is stable across rebuilds. `next_cursor` is `Some`
    /// whenever the page filled `limit` (a following call may return empty).
    ///
    /// # Errors
    /// [`StorageError::Corrupt`] if the cursor names a different room or a
    /// record fails to decode; [`StorageError::NotFound`] if a referenced
    /// record is missing.
    pub fn page(
        &self,
        room_id: &[u8; 16],
        cursor: Option<&TimelineCursor>,
        forward: bool,
        limit: usize,
    ) -> Result<TimelinePage, StorageError> {
        let empty = TimelinePage {
            entries: Vec::new(),
            next_cursor: None,
        };
        if limit == 0 {
            return Ok(empty);
        }

        let (snapshot_root, key) = match cursor {
            Some(cursor) => {
                if cursor.entry.room_id != *room_id {
                    return Err(StorageError::Corrupt(
                        "timeline cursor names a different room".to_owned(),
                    ));
                }
                (cursor.root_id, Some(cursor.entry()))
            }
            None => match self.head()? {
                Some((snapshot_root, _count)) => {
                    let key = if forward {
                        room_start(room_id)
                    } else {
                        room_end(room_id)
                    };
                    (snapshot_root, Some(key))
                }
                None => return Ok(empty),
            },
        };

        let root_bytes = self
            .engine
            .get(&self.collection_id, &snapshot_root)?
            .ok_or(StorageError::NotFound(snapshot_root))?;
        let root = decode_root(&root_bytes.bytes)?;
        if root.directories.is_empty() {
            return Ok(empty);
        }

        if forward {
            self.forward_page(snapshot_root, &root.directories, room_id, key, limit)
        } else {
            self.reverse_page(snapshot_root, &root.directories, room_id, key, limit)
        }
    }

    fn forward_page(
        &self,
        snapshot_root: NodeId,
        directories: &[DirectoryDescriptor],
        room_scope: &[u8; 16],
        key: Option<TimelineEntry>,
        limit: usize,
    ) -> Result<TimelinePage, StorageError> {
        let mut out: Vec<TimelineEntry> = Vec::with_capacity(limit);
        let start = forward_directory_start(directories, key);
        for directory in directories.iter().skip(start) {
            if directory.min_entry.room_id > *room_scope {
                break;
            }
            if directory.max_entry.room_id < *room_scope {
                continue;
            }
            let leaves = self.load_directory(directory.child_id, DIRECTORY_LEVEL)?;
            let leaf_start = forward_leaf_start(&leaves, key);
            for leaf_descriptor in leaves.iter().skip(leaf_start) {
                if leaf_descriptor.min_entry.room_id > *room_scope {
                    return Ok(TimelinePage {
                        entries: out,
                        next_cursor: None,
                    });
                }
                if leaf_descriptor.max_entry.room_id < *room_scope {
                    continue;
                }
                let leaf = self.load_leaf(&leaf_descriptor.child_id)?;
                for entry in &leaf {
                    if entry.room_id < *room_scope {
                        continue;
                    }
                    if entry.room_id > *room_scope {
                        return Ok(TimelinePage {
                            entries: out,
                            next_cursor: None,
                        });
                    }
                    if key.is_some_and(|k| *entry <= k) {
                        continue;
                    }
                    out.push(*entry);
                    if out.len() == limit {
                        return Ok(TimelinePage {
                            next_cursor: Some(TimelineCursor::new(snapshot_root, *entry)),
                            entries: out,
                        });
                    }
                }
            }
        }
        Ok(TimelinePage {
            entries: out,
            next_cursor: None,
        })
    }

    fn reverse_page(
        &self,
        snapshot_root: NodeId,
        directories: &[DirectoryDescriptor],
        room_scope: &[u8; 16],
        key: Option<TimelineEntry>,
        limit: usize,
    ) -> Result<TimelinePage, StorageError> {
        let Some(start) = reverse_directory_start(directories, key) else {
            return Ok(TimelinePage {
                entries: Vec::new(),
                next_cursor: None,
            });
        };
        let mut out: Vec<TimelineEntry> = Vec::with_capacity(limit);
        for index in (0..=start).rev() {
            let directory = &directories[index];
            if directory.min_entry.room_id > *room_scope {
                continue;
            }
            if directory.max_entry.room_id < *room_scope {
                break;
            }
            let leaves = self.load_directory(directory.child_id, DIRECTORY_LEVEL)?;
            let Some(leaf_start) = reverse_leaf_start(&leaves, key) else {
                continue;
            };
            for leaf_index in (0..=leaf_start).rev() {
                let leaf_descriptor = &leaves[leaf_index];
                if leaf_descriptor.min_entry.room_id > *room_scope {
                    continue;
                }
                if leaf_descriptor.max_entry.room_id < *room_scope {
                    break;
                }
                let leaf = self.load_leaf(&leaf_descriptor.child_id)?;
                for entry in leaf.iter().rev() {
                    if entry.room_id > *room_scope {
                        continue;
                    }
                    if entry.room_id < *room_scope {
                        return Ok(TimelinePage {
                            entries: out,
                            next_cursor: None,
                        });
                    }
                    if key.is_some_and(|k| *entry >= k) {
                        continue;
                    }
                    out.push(*entry);
                    if out.len() == limit {
                        return Ok(TimelinePage {
                            next_cursor: Some(TimelineCursor::new(snapshot_root, *entry)),
                            entries: out,
                        });
                    }
                }
            }
        }
        Ok(TimelinePage {
            entries: out,
            next_cursor: None,
        })
    }

    fn write_record(&self, id: &NodeId, payload: Vec<u8>) -> Result<(), StorageError> {
        let data = NodeData::new(Bytes::from(payload));
        let mut validate = |_existing: Option<&NodeData>| Ok(());
        self.engine.create_or_upsert_established_validated(
            &self.collection_id,
            &self.metadata(),
            id,
            &data,
            &mut validate,
        )
    }

    fn load_leaf(&self, id: &NodeId) -> Result<Vec<TimelineEntry>, StorageError> {
        let bytes = self
            .engine
            .get(&self.collection_id, id)?
            .ok_or(StorageError::NotFound(*id))?;
        decode_leaf(&bytes.bytes)
    }

    fn load_directory(
        &self,
        id: NodeId,
        expected_level: u8,
    ) -> Result<Vec<SpanDescriptor>, StorageError> {
        let bytes = self
            .engine
            .get(&self.collection_id, &id)?
            .ok_or(StorageError::NotFound(id))?;
        decode_directory(&bytes.bytes, expected_level)
    }
}

fn forward_directory_start(
    directories: &[DirectoryDescriptor],
    key: Option<TimelineEntry>,
) -> usize {
    let Some(key) = key else {
        return 0;
    };
    directories
        .partition_point(|directory| directory.min_entry <= key)
        .saturating_sub(1)
}

fn reverse_directory_start(
    directories: &[DirectoryDescriptor],
    key: Option<TimelineEntry>,
) -> Option<usize> {
    let bound = key?;
    directories
        .partition_point(|directory| directory.min_entry < bound)
        .checked_sub(1)
}

fn forward_leaf_start(leaves: &[LeafDescriptor], key: Option<TimelineEntry>) -> usize {
    let Some(key) = key else {
        return 0;
    };
    leaves
        .partition_point(|leaf| leaf.min_entry <= key)
        .saturating_sub(1)
}

fn reverse_leaf_start(leaves: &[LeafDescriptor], key: Option<TimelineEntry>) -> Option<usize> {
    let bound = key?;
    leaves
        .partition_point(|leaf| leaf.min_entry < bound)
        .checked_sub(1)
}

fn encode_leaf(entries: &[TimelineEntry]) -> Result<Vec<u8>, StorageError> {
    let count = u16::try_from(entries.len())
        .map_err(|_| StorageError::Internal("timeline leaf exceeds u16 entries".to_owned()))?;
    let mut out = Vec::with_capacity(
        entries
            .len()
            .saturating_mul(ENTRY_LEN)
            .saturating_add(LEAF_HEADER_LEN),
    );
    out.extend_from_slice(&TIMELINE_LEAF_MAGIC);
    out.push(LEAF_FORMAT_VERSION);
    out.extend_from_slice(&count.to_be_bytes());
    for entry in entries {
        entry.encode_into(&mut out);
    }
    Ok(out)
}

fn decode_leaf(bytes: &[u8]) -> Result<Vec<TimelineEntry>, StorageError> {
    if bytes.len() < LEAF_HEADER_LEN
        || bytes[..4] != TIMELINE_LEAF_MAGIC
        || bytes[4] != LEAF_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline leaf header".to_owned()));
    }
    let count = u16::from_be_bytes([bytes[5], bytes[6]]) as usize;
    let expected = count
        .checked_mul(ENTRY_LEN)
        .and_then(|body| body.checked_add(LEAF_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline leaf length".to_owned()))?;
    if bytes.len() != expected {
        return Err(StorageError::Corrupt("timeline leaf length".to_owned()));
    }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let start = index
            .checked_mul(ENTRY_LEN)
            .and_then(|offset| offset.checked_add(LEAF_HEADER_LEN))
            .ok_or_else(|| StorageError::Corrupt("timeline leaf offset".to_owned()))?;
        let end = start
            .checked_add(ENTRY_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline leaf offset".to_owned()))?;
        entries.push(TimelineEntry::decode(&bytes[start..end])?);
    }
    Ok(entries)
}

fn encode_directory(spans: &[SpanDescriptor], level: u8) -> Result<Vec<u8>, StorageError> {
    if spans.is_empty() || spans.len() > TIMELINE_DIRECTORY_FANOUT {
        return Err(StorageError::Internal(
            "timeline directory occupancy out of range".to_owned(),
        ));
    }
    let span_count = u16::try_from(spans.len())
        .map_err(|_| StorageError::Internal("timeline directory exceeds u16 spans".to_owned()))?;
    let mut out = Vec::with_capacity(
        spans
            .len()
            .saturating_mul(DIRECTORY_LEAF_LEN)
            .saturating_add(DIRECTORY_HEADER_LEN),
    );
    out.extend_from_slice(&TIMELINE_DIRECTORY_MAGIC);
    out.push(DIRECTORY_FORMAT_VERSION);
    out.push(level);
    out.extend_from_slice(&span_count.to_be_bytes());
    for span in spans {
        encode_span(span, &mut out);
    }
    Ok(out)
}

fn decode_directory(bytes: &[u8], expected_level: u8) -> Result<Vec<SpanDescriptor>, StorageError> {
    if bytes.len() < DIRECTORY_HEADER_LEN
        || bytes[..4] != TIMELINE_DIRECTORY_MAGIC
        || bytes[4] != DIRECTORY_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt(
            "timeline directory header".to_owned(),
        ));
    }
    let level = bytes[5];
    if level != expected_level {
        return Err(StorageError::Corrupt("timeline directory level".to_owned()));
    }
    let span_count = u16::from_be_bytes([bytes[6], bytes[7]]) as usize;
    let expected = span_count
        .checked_mul(DIRECTORY_LEAF_LEN)
        .and_then(|body| body.checked_add(DIRECTORY_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline directory length".to_owned()))?;
    if span_count == 0 || span_count > TIMELINE_DIRECTORY_FANOUT || bytes.len() != expected {
        return Err(StorageError::Corrupt(
            "timeline directory length or occupancy".to_owned(),
        ));
    }
    let body = bytes
        .get(DIRECTORY_HEADER_LEN..)
        .ok_or_else(|| StorageError::Corrupt("timeline directory body".to_owned()))?;
    let mut chunks = body.chunks_exact(DIRECTORY_LEAF_LEN);
    if !chunks.remainder().is_empty() {
        return Err(StorageError::Corrupt(
            "timeline directory trailing bytes".to_owned(),
        ));
    }
    let mut spans = Vec::with_capacity(span_count);
    let mut previous_max = None;
    for chunk in &mut chunks {
        let span = decode_span(chunk)?;
        if span.min_entry > span.max_entry
            || span.entry_count == 0
            || previous_max.is_some_and(|previous| previous >= span.min_entry)
        {
            return Err(StorageError::Corrupt(
                "timeline directory span bounds or count".to_owned(),
            ));
        }
        previous_max = Some(span.max_entry);
        spans.push(span);
    }
    Ok(spans)
}

fn encode_span(span: &SpanDescriptor, out: &mut Vec<u8>) {
    span.min_entry.encode_into(out);
    span.max_entry.encode_into(out);
    out.extend_from_slice(&span.entry_count.to_be_bytes());
    out.extend_from_slice(&span.child_id);
}

fn decode_span(bytes: &[u8]) -> Result<SpanDescriptor, StorageError> {
    if bytes.len() != SPAN_DESCRIPTOR_LEN {
        return Err(StorageError::Corrupt("timeline span length".to_owned()));
    }
    let (min_bytes, rest) = bytes
        .split_at_checked(ENTRY_LEN)
        .ok_or_else(|| StorageError::Corrupt("timeline span min bound".to_owned()))?;
    let (max_bytes, rest) = rest
        .split_at_checked(ENTRY_LEN)
        .ok_or_else(|| StorageError::Corrupt("timeline span max bound".to_owned()))?;
    let (count_bytes, child_bytes) = rest
        .split_at_checked(8)
        .ok_or_else(|| StorageError::Corrupt("timeline span count".to_owned()))?;
    Ok(SpanDescriptor {
        min_entry: TimelineEntry::decode(min_bytes)?,
        max_entry: TimelineEntry::decode(max_bytes)?,
        entry_count: u64::from_be_bytes(
            <[u8; 8]>::try_from(count_bytes)
                .map_err(|_| StorageError::Corrupt("timeline span count".to_owned()))?,
        ),
        child_id: <[u8; 16]>::try_from(child_bytes)
            .map_err(|_| StorageError::Corrupt("timeline span child".to_owned()))?,
    })
}
fn root_height(directories: &[DirectoryDescriptor]) -> u8 {
    if directories.is_empty() {
        0
    } else {
        ROOT_HEIGHT
    }
}

fn encode_root(
    directories: &[DirectoryDescriptor],
    count: u64,
    height: u8,
) -> Result<Vec<u8>, StorageError> {
    let directory_count = u32::try_from(directories.len())
        .map_err(|_| StorageError::Internal("timeline root exceeds u32 directories".to_owned()))?;
    let counted = directories.iter().try_fold(0u64, |sum, directory| {
        sum.checked_add(directory.entry_count)
            .ok_or_else(|| StorageError::Internal("timeline root count overflow".to_owned()))
    })?;
    if counted != count {
        return Err(StorageError::Internal(
            "timeline root/directory count mismatch".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(
        directories
            .len()
            .saturating_mul(ROOT_DIRECTORY_LEN)
            .saturating_add(ROOT_HEADER_LEN),
    );
    out.extend_from_slice(&TIMELINE_ROOT_MAGIC);
    out.push(ROOT_FORMAT_VERSION);
    out.push(height);
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&directory_count.to_be_bytes());
    for directory in directories {
        directory.min_entry.encode_into(&mut out);
        directory.max_entry.encode_into(&mut out);
        out.extend_from_slice(&directory.entry_count.to_be_bytes());
        out.extend_from_slice(&directory.child_id);
    }
    Ok(out)
}

fn decode_root(bytes: &[u8]) -> Result<TimelineRoot, StorageError> {
    if bytes.len() < ROOT_HEADER_LEN
        || bytes[..4] != TIMELINE_ROOT_MAGIC
        || bytes[4] != ROOT_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline root header".to_owned()));
    }
    let height = bytes[5];
    let entry_count = u64::from_be_bytes(
        bytes[6..14]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline root count".to_owned()))?,
    );
    let directory_count = u32::from_be_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
    let expected = directory_count
        .checked_mul(ROOT_DIRECTORY_LEN)
        .and_then(|body| body.checked_add(ROOT_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline root length".to_owned()))?;
    if bytes.len() != expected {
        return Err(StorageError::Corrupt("timeline root length".to_owned()));
    }
    let body = bytes
        .get(ROOT_HEADER_LEN..)
        .ok_or_else(|| StorageError::Corrupt("timeline root body".to_owned()))?;
    let mut chunks = body.chunks_exact(ROOT_DIRECTORY_LEN);
    if !chunks.remainder().is_empty() {
        return Err(StorageError::Corrupt(
            "timeline root trailing bytes".to_owned(),
        ));
    }
    let mut directories = Vec::with_capacity(directory_count);
    let mut previous_max = None;
    let mut counted = 0_u64;
    for chunk in &mut chunks {
        let (min_bytes, rest) = chunk
            .split_at_checked(ENTRY_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline root directory bounds".to_owned()))?;
        let (max_bytes, rest) = rest
            .split_at_checked(ENTRY_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline root directory bounds".to_owned()))?;
        let (count_bytes, child_bytes) = rest
            .split_at_checked(8)
            .ok_or_else(|| StorageError::Corrupt("timeline root directory count".to_owned()))?;
        let min_entry = TimelineEntry::decode(min_bytes)?;
        let max_entry = TimelineEntry::decode(max_bytes)?;
        let directory_entry_count = u64::from_be_bytes(
            <[u8; 8]>::try_from(count_bytes)
                .map_err(|_| StorageError::Corrupt("timeline root directory count".to_owned()))?,
        );
        let child_id = <[u8; 16]>::try_from(child_bytes)
            .map_err(|_| StorageError::Corrupt("timeline root child".to_owned()))?;
        if min_entry > max_entry
            || directory_entry_count == 0
            || previous_max.is_some_and(|previous| previous >= min_entry)
        {
            return Err(StorageError::Corrupt(
                "timeline root directory bounds or count".to_owned(),
            ));
        }
        counted = counted
            .checked_add(directory_entry_count)
            .ok_or_else(|| StorageError::Corrupt("timeline root count overflow".to_owned()))?;
        previous_max = Some(max_entry);
        directories.push(DirectoryDescriptor {
            min_entry,
            max_entry,
            entry_count: directory_entry_count,
            child_id,
        });
    }
    if counted != entry_count || (entry_count == 0) != directories.is_empty() {
        return Err(StorageError::Corrupt(
            "timeline root count mismatch".to_owned(),
        ));
    }
    let expected_height = if directories.is_empty() {
        0
    } else {
        ROOT_HEIGHT
    };
    if height != expected_height {
        return Err(StorageError::Corrupt("timeline root height".to_owned()));
    }
    Ok(TimelineRoot {
        height,
        entry_count,
        directories,
    })
}
fn encode_head(root_id: &NodeId, count: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_LEN);
    out.extend_from_slice(&TIMELINE_HEAD_MAGIC);
    out.push(HEAD_FORMAT_VERSION);
    out.extend_from_slice(root_id);
    out.extend_from_slice(&count.to_be_bytes());
    out
}

fn decode_head(bytes: &[u8]) -> Result<(NodeId, u64), StorageError> {
    if bytes.len() != HEAD_LEN
        || bytes[..4] != TIMELINE_HEAD_MAGIC
        || bytes[4] != HEAD_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline head".to_owned()));
    }
    let mut root_id = [0u8; 16];
    root_id.copy_from_slice(&bytes[5..21]);
    let count = u64::from_be_bytes(
        bytes[21..HEAD_LEN]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline head count".to_owned()))?,
    );
    Ok((root_id, count))
}

#[cfg(test)]
#[path = "test_timeline.rs"]
mod tests;
