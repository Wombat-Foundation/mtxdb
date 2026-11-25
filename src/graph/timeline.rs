//! Persisted, scoped sequence index with copy-on-write roots.
//!
//! The packfile index maps node id -> frame and has no key order, so it cannot
//! answer a range read over an ordered scope. This module adds a **derived**
//! ordered index built from ordinary records, in the same spirit as
//! [`crate::auxiliary::AuxiliaryIndex`]: it is rebuildable, packfiles stay
//! authoritative, and event payloads are never duplicated — a leaf stores only
//! the ordering key plus the event's node id; the caller fetches the payload.
//!
//! The persisted key is the total order
//!
//! ```text
//! (scope, primary, secondary, id, tag)
//! ```
//!
//! The caller defines the meaning of `scope`, `primary`, `secondary`, and
//! `tag`. These fixed-width fields form a memcomparable key followed by a
//! 128-bit record id and a 32-byte final tie-break.
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

use crate::bitmap_set::DomainTag;
use crate::database::{Database, DatabaseTransaction};
use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    MEMBER_NAMESPACE_ORDX,
};

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

/// A fixed-width final tie-break component of an ordered key.
pub type KeyTag = [u8; 32];
/// Compatibility alias for timeline callers that used event references.
pub type EventRef = KeyTag;

const ENTRY_LEN: usize = 16 + 8 + 8 + 16 + 32;
const LEAF_HEADER_LEN: usize = 7;
const DIRECTORY_HEADER_LEN: usize = 8;
const ROOT_HEADER_LEN: usize = 46;
const ROOT_FIXED_LEN: usize = 57;
const CURSOR_FIXED_LEN: usize = 49;
const HEAD_LEN: usize = ROOT_FIXED_LEN;
#[cfg(test)]
const CURSOR_LEN: usize = CURSOR_FIXED_LEN + ENTRY_LEN;
const LEAF_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const ROOT_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const DIRECTORY_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const HEAD_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;
const CURSOR_FORMAT_VERSION: u8 = TIMELINE_FORMAT_VERSION;

/// Order-preserving big-endian encoding of a signed position: flipping the
/// sign bit makes byte order equal numeric order, so the wire key is
/// byte-comparable.
fn span_descriptor_len<K: OrderedKey>() -> Result<usize, StorageError> {
    K::WIRE_LEN
        .checked_mul(2)
        .and_then(|value| value.checked_add(8 + 16))
        .ok_or_else(|| StorageError::Internal("ordered span length overflow".to_owned()))
}

fn validate_geometry<K: OrderedKey>() -> Result<(), StorageError> {
    if K::WIRE_LEN == 0 || K::LEAF_CAP == 0 || K::DIR_FANOUT < 2 {
        return Err(StorageError::Internal(
            "ordered index geometry is zero".to_owned(),
        ));
    }
    if K::LEAF_CAP > usize::from(u16::MAX) || K::DIR_FANOUT > usize::from(u16::MAX) {
        return Err(StorageError::Internal(
            "ordered index geometry exceeds u16".to_owned(),
        ));
    }
    Ok(())
}

fn encode_position(value: i64) -> [u8; 8] {
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    bytes
}

fn decode_position(mut bytes: [u8; 8]) -> i64 {
    bytes[0] ^= 0x80;
    i64::from_be_bytes(bytes)
}

/// One fixed-width ordered key: `(scope, primary, secondary, id, tag)`.
///
/// Field order is the sort order (`derive(Ord)` compares in declaration order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScopedSeqKey {
    /// The leading scope component.
    pub scope: [u8; 16],
    /// Caller-supplied primary signed order component.
    pub primary: i64,
    /// Caller-supplied secondary signed order component.
    pub secondary: i64,
    /// The 128-bit record id; tie-break after the order components.
    pub id: NodeId,
    /// Caller-supplied 32-byte final disambiguator.
    pub tag: KeyTag,
}

impl ScopedSeqKey {
    /// Construct an entry.
    #[must_use]
    pub const fn new(
        scope: [u8; 16],
        primary: i64,
        secondary: i64,
        id: NodeId,
        tag: KeyTag,
    ) -> Self {
        Self {
            scope,
            primary,
            secondary,
            id,
            tag,
        }
    }

    fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.scope);
        out.extend_from_slice(&encode_position(self.primary));
        out.extend_from_slice(&encode_position(self.secondary));
        out.extend_from_slice(&self.id);
        out.extend_from_slice(&self.tag);
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
        let mut scope = [0u8; 16];
        scope.copy_from_slice(&bytes[..16]);
        let primary = position(16..24, "primary ordering")?;
        let secondary = position(24..32, "secondary ordering")?;
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes[32..48]);
        let mut tag = [0u8; 32];
        tag.copy_from_slice(&bytes[48..ENTRY_LEN]);
        Ok(Self {
            scope,
            primary,
            secondary,
            id,
            tag,
        })
    }
}

/// Compatibility alias for the original timeline-specific key name.
pub type TimelineEntry = ScopedSeqKey;

/// Contract for fixed-width keys whose byte encoding matches their order.
pub trait OrderedKey: Copy + Ord + Eq {
    /// Leading prefix used to scope a range scan.
    type Prefix: Copy + Ord + Eq;
    /// Exact encoded key length.
    const WIRE_LEN: usize;
    /// Four-byte schema discriminator mixed into the index domain.
    const SCHEMA_TAG: [u8; 4];
    /// Maximum entries in one leaf record.
    const LEAF_CAP: usize = TIMELINE_LEAF_CAP;
    /// Maximum child descriptors in one directory record.
    const DIR_FANOUT: usize = TIMELINE_DIRECTORY_FANOUT;

    /// Return the leftmost prefix of this key's sort order.
    fn prefix(&self) -> Self::Prefix;
    /// Encode this key in memcomparable order.
    fn encode_into(&self, out: &mut Vec<u8>);
    /// Decode an encoded key.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] if `bytes` is not exactly
    /// [`Self::WIRE_LEN`] bytes or does not decode into a valid key.
    fn decode(bytes: &[u8]) -> Result<Self, StorageError>;
}

impl OrderedKey for ScopedSeqKey {
    type Prefix = [u8; 16];

    const WIRE_LEN: usize = ENTRY_LEN;
    const SCHEMA_TAG: [u8; 4] = *b"SQK1";

    fn prefix(&self) -> Self::Prefix {
        self.scope
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        (*self).encode_into(out);
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        ScopedSeqKey::decode(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanDescriptor<K: OrderedKey> {
    min_entry: K,
    max_entry: K,
    entry_count: u64,
    child_id: NodeId,
}

type LeafDescriptor<K> = SpanDescriptor<K>;
type DirectoryDescriptor<K> = SpanDescriptor<K>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimelineRoot<K: OrderedKey> {
    domain: DomainTag,
    height: u8,
    entry_count: u64,
    directories: Vec<DirectoryDescriptor<K>>,
}

#[derive(Default)]
struct StagedTree {
    records: Vec<StagedRecord>,
    by_id: std::collections::HashMap<NodeId, usize>,
    inserted: u64,
    removed: u64,
}

impl StagedTree {
    fn push(&mut self, id: NodeId, record: StagedRecord) {
        let index = self.records.len();
        self.records.push(record);
        self.by_id.insert(id, index);
    }
}

enum StagedRecord {
    Leaf(NodeId, Vec<u8>),
    Directory(NodeId, Vec<u8>),
}

enum TreeMutation<K: OrderedKey> {
    Remove(Vec<K>),
    PurgePrefix(K::Prefix),
}

trait TimelineTreeIo<K: OrderedKey> {
    fn load_leaf(&mut self, span: &SpanDescriptor<K>) -> Result<Vec<K>, StorageError>;
    fn load_directory(
        &mut self,
        span: &SpanDescriptor<K>,
        level: u8,
    ) -> Result<Vec<SpanDescriptor<K>>, StorageError>;
    fn write_leaf(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError>;
    fn write_directory(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError>;
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
    fanout: usize,
) -> Result<Vec<std::ops::Range<usize>>, StorageError> {
    match mode {
        SplitMode::Interior => balanced_chunk_ranges(len, fanout),
        SplitMode::Tail | SplitMode::Front => {
            let cap = fanout;
            if len == 0 {
                return Ok(Vec::new());
            }
            let remainder = len
                .checked_rem(cap)
                .ok_or_else(|| StorageError::Internal("directory fanout is zero".to_owned()))?;
            let first_size = match mode {
                SplitMode::Tail => cap.min(len),
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

fn split_mode<K: OrderedKey>(
    siblings: &[DirectoryDescriptor<K>],
    index: usize,
    span: &DirectoryDescriptor<K>,
    entries: &[K],
) -> SplitMode {
    let all_front = entries.last().is_some_and(|entry| *entry < span.min_entry);
    if index == 0 && all_front {
        return SplitMode::Front;
    }
    let no_same_room_after = siblings
        .get(index.saturating_add(1))
        .is_none_or(|next| next.min_entry.prefix() != span.max_entry.prefix());
    let all_tail = entries.first().is_some_and(|entry| *entry > span.max_entry);
    if no_same_room_after && all_tail {
        return SplitMode::Tail;
    }
    SplitMode::Interior
}

fn root_split_mode<K: OrderedKey>(root: &TimelineRoot<K>, entries: &[K]) -> (u8, SplitMode) {
    if root.height == 0 {
        return (1, SplitMode::Tail);
    }
    let mode = if root
        .directories
        .first()
        .is_some_and(|span| entries.last().is_some_and(|entry| *entry < span.min_entry))
    {
        SplitMode::Front
    } else if root
        .directories
        .last()
        .is_some_and(|span| entries.first().is_some_and(|entry| *entry > span.max_entry))
    {
        SplitMode::Tail
    } else {
        SplitMode::Interior
    };
    (root.height, mode)
}

fn validate_span_children<K: OrderedKey>(
    parent: &SpanDescriptor<K>,
    children: &[SpanDescriptor<K>],
) -> Result<(), StorageError> {
    let first = children
        .first()
        .ok_or_else(|| StorageError::Corrupt("empty timeline directory".to_owned()))?;
    let last = children
        .last()
        .ok_or_else(|| StorageError::Corrupt("empty timeline directory".to_owned()))?;
    let entry_count = children.iter().try_fold(0_u64, |count, child| {
        count
            .checked_add(child.entry_count)
            .ok_or_else(|| StorageError::Corrupt("timeline span count overflow".to_owned()))
    })?;
    if first.min_entry != parent.min_entry
        || last.max_entry != parent.max_entry
        || entry_count != parent.entry_count
    {
        return Err(StorageError::Corrupt(
            "timeline directory span mismatch".to_owned(),
        ));
    }
    Ok(())
}

fn target_span_index<K: OrderedKey>(spans: &[SpanDescriptor<K>], level: u8, entry: &K) -> usize {
    let pos = spans.partition_point(|span| span.min_entry <= *entry);
    if pos == 0 || pos == spans.len() {
        return pos.saturating_sub(1);
    }
    let left = pos.saturating_sub(1);
    if *entry <= spans[left].max_entry {
        return left;
    }
    let left_span = &spans[left];
    let right_span = &spans[pos];
    if left_span.max_entry.prefix() != entry.prefix()
        && right_span.min_entry.prefix() == entry.prefix()
    {
        return pos;
    }
    if level == 0
        && left_span.max_entry.prefix() == entry.prefix()
        && right_span.min_entry.prefix() == entry.prefix()
        && left_span.entry_count == u64::try_from(K::LEAF_CAP).unwrap_or(u64::MAX)
        && right_span.entry_count < u64::try_from(K::LEAF_CAP).unwrap_or(u64::MAX)
    {
        return pos;
    }
    left
}

fn append_spans<K: OrderedKey>(
    spans: &[SpanDescriptor<K>],
    level: u8,
    entries: &[K],
    io: &mut dyn TimelineTreeIo<K>,
    inserted: &mut u64,
) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
    if spans.is_empty() {
        if level != 0 {
            return Err(StorageError::Corrupt(
                "empty non-leaf timeline directory".to_owned(),
            ));
        }
        let mut leaves = Vec::new();
        merge_and_stage_leaves(&[], entries, &mut leaves, inserted, |id, payload| {
            io.write_leaf(id, payload)
        })?;
        return Ok(leaves);
    }

    let mut assigned = vec![Vec::new(); spans.len()];
    for entry in entries {
        let index = target_span_index(spans, level, entry);
        let slot = assigned
            .get_mut(index)
            .ok_or_else(|| StorageError::Internal("timeline span route".to_owned()))?;
        slot.push(*entry);
    }

    let mut output = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().enumerate() {
        let incoming = &assigned[index];
        if incoming.is_empty() {
            output.push(*span);
            continue;
        }
        let before = *inserted;
        if level == 0 {
            let all_before = incoming.last().is_some_and(|entry| *entry < span.min_entry);
            let all_after = incoming
                .first()
                .is_some_and(|entry| *entry > span.max_entry);
            let leaf_cap = u64::try_from(K::LEAF_CAP).unwrap_or(u64::MAX);
            if span.entry_count == leaf_cap && (all_before || all_after) {
                let mut added = Vec::new();
                merge_and_stage_leaves(&[], incoming, &mut added, inserted, |id, payload| {
                    io.write_leaf(id, payload)
                })?;
                if all_before {
                    output.extend(added);
                    output.push(*span);
                } else {
                    output.push(*span);
                    output.extend(added);
                }
                continue;
            }
            let existing = io.load_leaf(span)?;
            validate_leaf_span(span, &existing)?;
            if incoming
                .iter()
                .all(|entry| existing.binary_search(entry).is_ok())
            {
                output.push(*span);
                continue;
            }
            merge_and_stage_leaves(&existing, incoming, &mut output, inserted, |id, payload| {
                io.write_leaf(id, payload)
            })?;
        } else {
            let children = io.load_directory(span, level)?;
            validate_span_children(span, &children)?;
            let child_level = level.checked_sub(1).ok_or_else(|| {
                StorageError::Corrupt("timeline directory level underflow".to_owned())
            })?;
            let changed = append_spans(&children, child_level, incoming, io, inserted)?;
            if *inserted == before {
                output.push(*span);
                continue;
            }
            let mode = split_mode(spans, index, span, incoming);
            output.extend(build_directory_blocks(
                &changed,
                level,
                mode,
                |id, payload| io.write_directory(id, payload),
            )?);
        }
    }
    Ok(output)
}

fn write_leaf_span<K: OrderedKey>(
    entries: &[K],
    io: &mut dyn TimelineTreeIo<K>,
) -> Result<SpanDescriptor<K>, StorageError> {
    let first = *entries
        .first()
        .ok_or_else(|| StorageError::Internal("cannot write an empty ordered leaf".to_owned()))?;
    let last = *entries
        .last()
        .ok_or_else(|| StorageError::Internal("cannot write an empty ordered leaf".to_owned()))?;
    let payload = encode_leaf::<K>(entries)?;
    let id = content_id(&payload);
    io.write_leaf(id, payload)?;
    Ok(SpanDescriptor {
        min_entry: first,
        max_entry: last,
        entry_count: u64::try_from(entries.len())
            .map_err(|_| StorageError::Internal("ordered leaf count overflow".to_owned()))?,
        child_id: id,
    })
}

fn remove_spans<K: OrderedKey>(
    spans: &[SpanDescriptor<K>],
    level: u8,
    to_remove: &[K],
    io: &mut dyn TimelineTreeIo<K>,
    removed: &mut u64,
) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
    if spans.is_empty() || to_remove.is_empty() {
        return Ok(spans.to_vec());
    }
    let mut assigned = vec![Vec::new(); spans.len()];
    for entry in to_remove {
        let index = spans.partition_point(|span| span.max_entry < *entry);
        let Some(span) = spans.get(index) else {
            continue;
        };
        if *entry < span.min_entry {
            continue;
        }
        assigned
            .get_mut(index)
            .ok_or_else(|| StorageError::Internal("ordered remove route".to_owned()))?
            .push(*entry);
    }

    let mut output = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().enumerate() {
        let incoming = assigned
            .get(index)
            .ok_or_else(|| StorageError::Internal("ordered remove assignment".to_owned()))?;
        if incoming.is_empty() {
            output.push(*span);
            continue;
        }
        if level == 0 {
            let existing = io.load_leaf(span)?;
            validate_leaf_span(span, &existing)?;
            let mut surviving = Vec::with_capacity(existing.len());
            let (mut source, mut requested) = (0_usize, 0_usize);
            let mut removed_here = 0_u64;
            while source < existing.len() {
                let entry = existing[source];
                while incoming
                    .get(requested)
                    .is_some_and(|candidate| *candidate < entry)
                {
                    requested = requested.checked_add(1).ok_or_else(|| {
                        StorageError::Internal("ordered remove offset overflow".to_owned())
                    })?;
                }
                if incoming
                    .get(requested)
                    .is_some_and(|candidate| *candidate == entry)
                {
                    requested = requested.checked_add(1).ok_or_else(|| {
                        StorageError::Internal("ordered remove offset overflow".to_owned())
                    })?;
                    removed_here = removed_here.checked_add(1).ok_or_else(|| {
                        StorageError::Internal("ordered remove count overflow".to_owned())
                    })?;
                } else {
                    surviving.push(entry);
                }
                source = source.checked_add(1).ok_or_else(|| {
                    StorageError::Internal("ordered remove offset overflow".to_owned())
                })?;
            }
            *removed = removed.checked_add(removed_here).ok_or_else(|| {
                StorageError::Internal("ordered remove count overflow".to_owned())
            })?;
            if removed_here == 0 {
                output.push(*span);
            } else if !surviving.is_empty() {
                output.push(write_leaf_span(&surviving, io)?);
            }
            continue;
        }

        let children = io.load_directory(span, level)?;
        validate_span_children(span, &children)?;
        let before = *removed;
        let child_level = level
            .checked_sub(1)
            .ok_or_else(|| StorageError::Corrupt("ordered remove level underflow".to_owned()))?;
        let changed = remove_spans(&children, child_level, incoming, io, removed)?;
        if *removed == before {
            output.push(*span);
        } else if !changed.is_empty() {
            output.extend(build_directory_blocks(
                &changed,
                level,
                SplitMode::Interior,
                |id, payload| io.write_directory(id, payload),
            )?);
        }
    }
    Ok(output)
}

fn purge_prefix_spans<K: OrderedKey>(
    spans: &[SpanDescriptor<K>],
    level: u8,
    prefix: &K::Prefix,
    io: &mut dyn TimelineTreeIo<K>,
    removed: &mut u64,
) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
    let mut output = Vec::with_capacity(spans.len());
    for span in spans {
        if span.max_entry.prefix() < *prefix || span.min_entry.prefix() > *prefix {
            output.push(*span);
            continue;
        }
        if span.min_entry.prefix() == *prefix && span.max_entry.prefix() == *prefix {
            *removed = removed.checked_add(span.entry_count).ok_or_else(|| {
                StorageError::Internal("ordered prefix purge count overflow".to_owned())
            })?;
            continue;
        }
        if level == 0 {
            let existing = io.load_leaf(span)?;
            validate_leaf_span(span, &existing)?;
            let surviving: Vec<_> = existing
                .iter()
                .copied()
                .filter(|entry| entry.prefix() != *prefix)
                .collect();
            let dropped = span
                .entry_count
                .checked_sub(u64::try_from(surviving.len()).map_err(|_| {
                    StorageError::Internal("ordered prefix purge count overflow".to_owned())
                })?)
                .ok_or_else(|| StorageError::Corrupt("ordered leaf count underflow".to_owned()))?;
            *removed = removed.checked_add(dropped).ok_or_else(|| {
                StorageError::Internal("ordered prefix purge count overflow".to_owned())
            })?;
            if !surviving.is_empty() {
                output.push(write_leaf_span(&surviving, io)?);
            }
            continue;
        }

        let children = io.load_directory(span, level)?;
        validate_span_children(span, &children)?;
        let before = *removed;
        let child_level = level.checked_sub(1).ok_or_else(|| {
            StorageError::Corrupt("ordered prefix purge level underflow".to_owned())
        })?;
        let changed = purge_prefix_spans(&children, child_level, prefix, io, removed)?;
        if *removed == before {
            output.push(*span);
        } else if !changed.is_empty() {
            output.extend(build_directory_blocks(
                &changed,
                level,
                SplitMode::Interior,
                |id, payload| io.write_directory(id, payload),
            )?);
        }
    }
    Ok(output)
}

fn demote_spans<K: OrderedKey>(
    mut spans: Vec<SpanDescriptor<K>>,
    mut height: u8,
    io: &mut dyn TimelineTreeIo<K>,
) -> Result<(u8, Vec<SpanDescriptor<K>>), StorageError> {
    if spans.is_empty() {
        return Ok((0, spans));
    }
    while height > 1 && spans.len() == 1 {
        let child_level = height
            .checked_sub(1)
            .ok_or_else(|| StorageError::Corrupt("ordered root height underflow".to_owned()))?;
        let children = io.load_directory(&spans[0], child_level)?;
        validate_span_children(&spans[0], &children)?;
        spans = children;
        height = child_level;
    }
    Ok((height, spans))
}

fn span_entry_count<K: OrderedKey>(spans: &[SpanDescriptor<K>]) -> Result<u64, StorageError> {
    spans.iter().try_fold(0_u64, |count, span| {
        count
            .checked_add(span.entry_count)
            .ok_or_else(|| StorageError::Internal("ordered entry count overflow".to_owned()))
    })
}

fn mutation_count<K: OrderedKey>(
    old_count: u64,
    removed: u64,
    directories: &[SpanDescriptor<K>],
) -> Result<u64, StorageError> {
    let count = old_count
        .checked_sub(removed)
        .ok_or_else(|| StorageError::Corrupt("ordered remove count underflow".to_owned()))?;
    if span_entry_count(directories)? != count {
        return Err(StorageError::Corrupt(
            "ordered mutation root count mismatch".to_owned(),
        ));
    }
    Ok(count)
}

fn validate_leaf_span<K: OrderedKey>(
    descriptor: &SpanDescriptor<K>,
    entries: &[K],
) -> Result<(), StorageError> {
    let first = entries
        .first()
        .ok_or_else(|| StorageError::Corrupt("empty timeline leaf".to_owned()))?;
    let last = entries
        .last()
        .ok_or_else(|| StorageError::Corrupt("empty timeline leaf".to_owned()))?;
    let count = u64::try_from(entries.len())
        .map_err(|_| StorageError::Corrupt("timeline leaf count overflow".to_owned()))?;
    if *first != descriptor.min_entry
        || *last != descriptor.max_entry
        || count != descriptor.entry_count
    {
        return Err(StorageError::Corrupt(
            "timeline leaf span mismatch".to_owned(),
        ));
    }
    Ok(())
}

fn promote_spans<K: OrderedKey>(
    mut spans: Vec<SpanDescriptor<K>>,
    mut height: u8,
    mode: SplitMode,
    io: &mut dyn TimelineTreeIo<K>,
) -> Result<(u8, Vec<SpanDescriptor<K>>), StorageError> {
    if spans.is_empty() {
        return Ok((0, spans));
    }
    while spans.len() > K::DIR_FANOUT {
        let next_height = height
            .checked_add(1)
            .ok_or_else(|| StorageError::Internal("timeline height overflow".to_owned()))?;
        spans = build_directory_blocks(&spans, height, mode, |id, payload| {
            io.write_directory(id, payload)
        })?;
        height = next_height;
    }
    Ok((height, spans))
}

/// An opaque, stable cursor naming the snapshot root plus the last entry read.
///
/// It carries the full ordering key, so both directions resume exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderedCursor<K: OrderedKey> {
    domain: DomainTag,
    schema_tag: [u8; 4],
    leaf_cap: u32,
    dir_fanout: u32,
    root_id: NodeId,
    entry: K,
}

/// Compatibility alias for the timeline-scoped ordered cursor.
pub type TimelineCursor = OrderedCursor<ScopedSeqKey>;

impl<K: OrderedKey> OrderedCursor<K> {
    /// Build a cursor for `root_id` at `entry`.
    ///
    /// # Errors
    /// Returns an error if the key geometry cannot be represented in the
    /// cursor's wire header.
    pub fn new(root_id: NodeId, entry: K, domain: DomainTag) -> Result<Self, StorageError> {
        let leaf_cap = u32::try_from(K::LEAF_CAP)
            .map_err(|_| StorageError::Internal("ordered leaf cap exceeds u32".to_owned()))?;
        let dir_fanout = u32::try_from(K::DIR_FANOUT)
            .map_err(|_| StorageError::Internal("ordered fanout exceeds u32".to_owned()))?;
        Ok(Self {
            domain,
            schema_tag: K::SCHEMA_TAG,
            leaf_cap,
            dir_fanout,
            root_id,
            entry,
        })
    }

    /// The snapshot root this cursor pins.
    #[must_use]
    pub const fn root_id(&self) -> NodeId {
        self.root_id
    }

    /// The entry this cursor names.
    #[must_use]
    pub const fn entry(&self) -> K {
        self.entry
    }

    /// The storage and key-schema domain this cursor belongs to.
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Encode the cursor as opaque, versioned bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let wire_len = CURSOR_FIXED_LEN.saturating_add(K::WIRE_LEN);
        let mut out = Vec::with_capacity(wire_len);
        out.extend_from_slice(&TIMELINE_CURSOR_MAGIC);
        out.push(CURSOR_FORMAT_VERSION);
        out.extend_from_slice(self.domain.as_bytes());
        out.extend_from_slice(&self.schema_tag);
        out.extend_from_slice(&self.leaf_cap.to_be_bytes());
        out.extend_from_slice(&self.dir_fanout.to_be_bytes());
        out.extend_from_slice(&self.root_id);
        self.entry.encode_into(&mut out);
        out
    }

    /// Decode cursor bytes produced by [`Self::encode`].
    ///
    /// Returns `None` for a wrong magic, version, or length.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let wire_len = CURSOR_FIXED_LEN.saturating_add(K::WIRE_LEN);
        if bytes.len() != wire_len
            || bytes[..4] != TIMELINE_CURSOR_MAGIC
            || bytes[4] != CURSOR_FORMAT_VERSION
        {
            return None;
        }
        let domain = DomainTag::new(bytes[5..21].try_into().ok()?);
        let schema_tag = bytes[21..25].try_into().ok()?;
        let leaf_cap = u32::from_be_bytes(bytes[25..29].try_into().ok()?);
        let dir_fanout = u32::from_be_bytes(bytes[29..33].try_into().ok()?);
        let mut root_id = [0u8; 16];
        root_id.copy_from_slice(&bytes[33..49]);
        let entry = K::decode(&bytes[CURSOR_FIXED_LEN..wire_len]).ok()?;
        Some(Self {
            domain,
            schema_tag,
            leaf_cap,
            dir_fanout,
            root_id,
            entry,
        })
    }
}

/// One page of a timeline read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedPage<K: OrderedKey> {
    /// The entries in traversal order.
    pub entries: Vec<K>,
    /// Resume token, present when the page filled `limit`.
    pub next_cursor: Option<OrderedCursor<K>>,
}

/// Compatibility alias for the timeline-scoped ordered page.
pub type TimelinePage = OrderedPage<ScopedSeqKey>;

/// Derive an ordered-index collection id for a caller-chosen index name.
#[must_use]
pub fn ordered_collection_id<K: OrderedKey>(name: &str) -> [u8; 16] {
    let mut canonical = Vec::with_capacity(name.len().saturating_add(13));
    canonical.extend_from_slice(b"ordered:");
    canonical.extend_from_slice(&K::SCHEMA_TAG);
    canonical.push(b':');
    canonical.extend_from_slice(name.as_bytes());
    derive_collection_id(Some(MEMBER_NAMESPACE_ORDX), &canonical)
}

/// The compatibility collection id for timeline callers.
#[must_use]
pub fn timeline_collection_id(name: &str) -> [u8; 16] {
    ordered_collection_id::<ScopedSeqKey>(name)
}

fn ordered_domain<K: OrderedKey>(collection_id: &[u8; 16]) -> DomainTag {
    let schema = K::SCHEMA_TAG;
    let mut label = Vec::with_capacity(schema.len().saturating_add(24));
    label.extend_from_slice(&schema);
    label.extend_from_slice(&u64::try_from(K::LEAF_CAP).unwrap_or(u64::MAX).to_be_bytes());
    label.extend_from_slice(
        &u64::try_from(K::DIR_FANOUT)
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    label.extend_from_slice(collection_id);
    DomainTag::derive(&label)
}

fn content_id(payload: &[u8]) -> NodeId {
    let digest = DigestAlgorithm::Blake3.digest(payload);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

fn merge_and_stage_leaves<K: OrderedKey>(
    existing: &[K],
    incoming: &[K],
    manifest: &mut Vec<LeafDescriptor<K>>,
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

    for chunk in merged.chunks(K::LEAF_CAP) {
        let payload = encode_leaf::<K>(chunk)?;
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
                .map_err(|_| StorageError::Internal("ordered leaf exceeds u64".to_owned()))?,
            child_id: id,
        });
    }
    Ok(())
}

fn build_directory_blocks<K: OrderedKey>(
    leaves: &[LeafDescriptor<K>],
    level: u8,
    mode: SplitMode,
    mut write_record: impl FnMut(NodeId, Vec<u8>) -> Result<(), StorageError>,
) -> Result<Vec<DirectoryDescriptor<K>>, StorageError> {
    let mut directories = Vec::with_capacity(leaves.len().div_ceil(K::DIR_FANOUT));
    for range in directory_chunk_ranges(leaves.len(), mode, K::DIR_FANOUT)? {
        let block = &leaves[range];
        let payload = encode_directory::<K>(block, level)?;
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

/// A persisted copy-on-write ordered index over fixed-width keys.
pub struct OrderedIndex<'a, S: StorageEngine + ?Sized, K: OrderedKey> {
    engine: &'a S,
    name: Cow<'a, str>,
    collection_id: [u8; 16],
    domain: DomainTag,
    _key: std::marker::PhantomData<K>,
}

/// Compatibility alias for the original ordered timeline index API.
pub type TimelineIndex<'a, S> = OrderedIndex<'a, S, ScopedSeqKey>;

impl<'a, S: StorageEngine + ?Sized, K: OrderedKey> OrderedIndex<'a, S, K> {
    /// Open (or create) the named ordered index.
    #[must_use]
    pub fn open(engine: &'a S, name: impl Into<Cow<'a, str>>) -> Self {
        let name = name.into();
        let collection_id = ordered_collection_id::<K>(name.as_ref());
        Self::with_collection_id(engine, name, collection_id)
    }

    /// Open with a pre-computed collection ID and pre-owned or borrowed name.
    #[must_use]
    pub fn with_collection_id(
        engine: &'a S,
        name: impl Into<Cow<'a, str>>,
        collection_id: [u8; 16],
    ) -> Self {
        let domain = ordered_domain::<K>(&collection_id);
        Self::with_collection_id_and_domain(engine, name, collection_id, domain)
    }

    /// Open with an explicit collection id and domain.
    #[must_use]
    pub fn with_collection_id_and_domain(
        engine: &'a S,
        name: impl Into<Cow<'a, str>>,
        collection_id: [u8; 16],
        domain: DomainTag,
    ) -> Self {
        Self {
            engine,
            name: name.into(),
            collection_id,
            domain,
            _key: std::marker::PhantomData,
        }
    }

    /// This index's collection id.
    #[must_use]
    pub fn collection_id(&self) -> [u8; 16] {
        self.collection_id
    }

    /// This index's key and storage-pool domain.
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Construct a cursor pinned to this index's domain.
    ///
    /// # Errors
    /// Returns an error if the key geometry cannot be represented in the
    /// cursor's wire header.
    pub fn cursor(&self, root_id: NodeId, entry: K) -> Result<OrderedCursor<K>, StorageError> {
        OrderedCursor::new(root_id, entry, self.domain)
    }

    /// The collection metadata describing this index.
    #[must_use]
    pub fn metadata(&self) -> CollectionMetadata {
        CollectionMetadata {
            member_namespace: Some(MEMBER_NAMESPACE_ORDX),
            collection_canonical_id: {
                let mut canonical = Vec::with_capacity(self.name.len().saturating_add(13));
                canonical.extend_from_slice(b"ordered:");
                canonical.extend_from_slice(&K::SCHEMA_TAG);
                canonical.push(b':');
                canonical.extend_from_slice(self.name.as_bytes());
                canonical
            },
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("ordered_index".to_owned()),
            schema: Some(format!(
                "mtxdb.ordered.{}",
                String::from_utf8_lossy(&K::SCHEMA_TAG)
            )),
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
    pub fn build(&self, entries: &[K]) -> Result<NodeId, StorageError> {
        validate_geometry::<K>()?;
        self.ensure_metadata()?;
        let mut sorted = entries.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut leaves: Vec<LeafDescriptor<K>> = Vec::new();
        for chunk in sorted.chunks(K::LEAF_CAP) {
            let payload = encode_leaf::<K>(chunk)?;
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

        let count = u64::try_from(sorted.len())
            .map_err(|_| StorageError::Internal("timeline entry count exceeds u64".to_owned()))?;
        let (height, directories) = {
            let mut io = DirectTimelineTreeIo { index: self };
            promote_spans(leaves, 1, SplitMode::Tail, &mut io)?
        };
        let root_payload = encode_root::<K>(&directories, count, height, self.domain)?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(
            &TIMELINE_HEAD_ID,
            encode_head::<K>(&root_id, count, self.domain)?,
        )?;
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
        new_entries: &[K],
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
        new_entries: &[K],
    ) -> Result<NodeId, StorageError> {
        validate_geometry::<K>()?;
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
            let (root_id, count) = decode_head::<K>(&head.bytes, self.domain)?;
            let root_records = transaction.get(pool, &self.collection_id, &[root_id])?;
            let root_record = root_records
                .first()
                .and_then(Option::as_ref)
                .ok_or(StorageError::NotFound(root_id))?;
            let root = decode_root::<K>(&root_record.bytes, self.domain)?;
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
                    domain: self.domain,
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
        let mut inserted = 0_u64;
        let mut staged = StagedTree::default();
        let (start_height, split_mode) = root_split_mode(&root, &pending);
        let (next_height, next_directories) = {
            let mut io = TransactionTimelineTreeIo {
                index: self,
                transaction,
                pool,
                staged: &mut staged,
            };
            let next_spans = append_spans(
                &root.directories,
                root.height.saturating_sub(1),
                &pending,
                &mut io,
                &mut inserted,
            )?;
            promote_spans(next_spans, start_height, split_mode, &mut io)?
        };
        staged.inserted = inserted;
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

        for record in staged.records {
            match record {
                StagedRecord::Leaf(id, payload) | StagedRecord::Directory(id, payload) => {
                    self.stage_record(transaction, pool, &id, payload)?;
                }
            }
        }

        let count = old_count
            .checked_add(staged.inserted)
            .ok_or_else(|| StorageError::Internal("timeline entry count overflow".to_owned()))?;
        let root_payload = encode_root::<K>(&next_directories, count, next_height, self.domain)?;
        let root_id = content_id(&root_payload);
        self.stage_record(transaction, pool, &root_id, root_payload)?;
        self.stage_record(
            transaction,
            pool,
            &TIMELINE_HEAD_ID,
            encode_head::<K>(&root_id, count, self.domain)?,
        )?;
        Ok(root_id)
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
    pub fn append_untransactional(&self, new_entries: &[K]) -> Result<NodeId, StorageError> {
        validate_geometry::<K>()?;
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
        let root = decode_root::<K>(&root_bytes.bytes, self.domain)?;
        if root.entry_count != old_count {
            return Err(StorageError::Corrupt(
                "timeline head/root count mismatch".to_owned(),
            ));
        }
        let mut pending = new_entries.to_vec();
        pending.sort_unstable();
        pending.dedup();
        let (start_height, mode) = root_split_mode(&root, &pending);
        let mut inserted = 0_u64;
        let next_spans = {
            let mut io = DirectTimelineTreeIo { index: self };
            append_spans(
                &root.directories,
                root.height.saturating_sub(1),
                &pending,
                &mut io,
                &mut inserted,
            )?
        };
        if inserted == 0 {
            return Ok(old_root_id);
        }
        let mut io = DirectTimelineTreeIo { index: self };
        let (height, directories) = promote_spans(next_spans, start_height, mode, &mut io)?;
        let count = old_count
            .checked_add(inserted)
            .ok_or_else(|| StorageError::Internal("timeline entry count overflow".to_owned()))?;
        let root_payload = encode_root::<K>(&directories, count, height, self.domain)?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(
            &TIMELINE_HEAD_ID,
            encode_head::<K>(&root_id, count, self.domain)?,
        )?;
        Ok(root_id)
    }

    /// Stage removal of the supplied keys and publish one replacement root.
    /// Missing and duplicate keys are ignored.
    ///
    /// # Errors
    /// Propagates transaction, storage, and record validation errors.
    pub fn stage_remove(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        to_remove: &[K],
    ) -> Result<(NodeId, u64), StorageError> {
        let mut keys = to_remove.to_vec();
        keys.sort_unstable();
        keys.dedup();
        self.stage_mutation(transaction, pool, TreeMutation::Remove(keys))
    }

    /// Stage removal of every key whose leading prefix equals `prefix`.
    /// Entire covered subtrees are removed from their descriptors without
    /// reading their directory or leaf records.
    ///
    /// # Errors
    /// Propagates transaction, storage, and record validation errors.
    pub fn stage_purge_prefix(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        prefix: K::Prefix,
    ) -> Result<(NodeId, u64), StorageError> {
        self.stage_mutation(transaction, pool, TreeMutation::PurgePrefix(prefix))
    }

    /// Remove keys through raw storage for single-writer maintenance.
    ///
    /// # Errors
    /// Propagates storage errors or reports corrupt/missing referenced nodes.
    pub fn remove_untransactional(&self, to_remove: &[K]) -> Result<(NodeId, u64), StorageError> {
        let mut keys = to_remove.to_vec();
        keys.sort_unstable();
        keys.dedup();
        self.mutate_untransactional(TreeMutation::Remove(keys))
    }

    /// Remove a complete leading-prefix range through raw storage.
    ///
    /// # Errors
    /// Propagates storage errors or reports corrupt/missing referenced nodes.
    pub fn purge_prefix_untransactional(
        &self,
        prefix: K::Prefix,
    ) -> Result<(NodeId, u64), StorageError> {
        self.mutate_untransactional(TreeMutation::PurgePrefix(prefix))
    }

    /// Stage a removal followed by an insertion in one transaction.
    ///
    /// Use this to repair or re-derive ordering keys: both passes publish a
    /// single new root and head. The removal set is applied first.
    ///
    /// # Errors
    /// Propagates the errors of [`Self::stage_remove`] and
    /// [`Self::stage_append`].
    pub fn stage_mutate(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        to_remove: &[K],
        to_insert: &[K],
    ) -> Result<NodeId, StorageError> {
        self.stage_remove(transaction, pool, to_remove)?;
        self.stage_append(transaction, pool, to_insert)
    }

    fn stage_mutation(
        &self,
        transaction: &DatabaseTransaction<'_>,
        pool: ShardType,
        mutation: TreeMutation<K>,
    ) -> Result<(NodeId, u64), StorageError> {
        validate_geometry::<K>()?;
        let engine_identity = std::ptr::from_ref(self.engine).cast::<()>() as usize;
        if engine_identity != transaction.pool_identity(pool) {
            return Err(StorageError::Internal(
                "ordered index and transaction use different storage pools".to_owned(),
            ));
        }
        self.ensure_metadata()?;
        let (head_records, collection_version) = transaction.get_with_collection_version(
            pool,
            &self.collection_id,
            &[TIMELINE_HEAD_ID],
        )?;
        let head = head_records.first().ok_or_else(|| {
            StorageError::Internal("ordered head read returned no slot".to_owned())
        })?;
        let (old_root_id, old_count, root) = if let Some(head) = head {
            let (root_id, count) = decode_head::<K>(&head.bytes, self.domain)?;
            let records = transaction.get(pool, &self.collection_id, &[root_id])?;
            let root_record = records
                .first()
                .and_then(Option::as_ref)
                .ok_or(StorageError::NotFound(root_id))?;
            let root = decode_root::<K>(&root_record.bytes, self.domain)?;
            if root.entry_count != count {
                return Err(StorageError::Corrupt(
                    "ordered head/root count mismatch".to_owned(),
                ));
            }
            (Some(root_id), count, root)
        } else {
            (
                None,
                0,
                TimelineRoot {
                    domain: self.domain,
                    height: 0,
                    entry_count: 0,
                    directories: Vec::new(),
                },
            )
        };

        let mut staged = StagedTree::default();
        let mut removed = 0_u64;
        let (height, directories) = {
            let mut io = TransactionTimelineTreeIo {
                index: self,
                transaction,
                pool,
                staged: &mut staged,
            };
            let changed = match mutation {
                TreeMutation::Remove(keys) => remove_spans(
                    &root.directories,
                    root.height.saturating_sub(1),
                    &keys,
                    &mut io,
                    &mut removed,
                )?,
                TreeMutation::PurgePrefix(prefix) => purge_prefix_spans(
                    &root.directories,
                    root.height.saturating_sub(1),
                    &prefix,
                    &mut io,
                    &mut removed,
                )?,
            };
            demote_spans(changed, root.height, &mut io)?
        };
        staged.removed = removed;
        if removed == 0 {
            if let Some(root_id) = old_root_id {
                return Ok((root_id, 0));
            }
        }
        let count = mutation_count(old_count, removed, &directories)?;
        transaction
            .expect_collection_version(pool, self.collection_id, collection_version)
            .map_err(StorageError::Io)?;
        for record in staged.records {
            match record {
                StagedRecord::Leaf(id, payload) | StagedRecord::Directory(id, payload) => {
                    self.stage_record(transaction, pool, &id, payload)?;
                }
            }
        }
        let root_payload = encode_root::<K>(&directories, count, height, self.domain)?;
        let root_id = content_id(&root_payload);
        self.stage_record(transaction, pool, &root_id, root_payload)?;
        self.stage_record(
            transaction,
            pool,
            &TIMELINE_HEAD_ID,
            encode_head::<K>(&root_id, count, self.domain)?,
        )?;
        Ok((root_id, removed))
    }

    fn mutate_untransactional(
        &self,
        mutation: TreeMutation<K>,
    ) -> Result<(NodeId, u64), StorageError> {
        validate_geometry::<K>()?;
        self.ensure_metadata()?;
        let Some((old_root_id, old_count)) = self.head()? else {
            let root_id = self.build(&[])?;
            return Ok((root_id, 0));
        };
        let bytes = self
            .engine
            .get(&self.collection_id, &old_root_id)?
            .ok_or(StorageError::NotFound(old_root_id))?;
        let root = decode_root::<K>(&bytes.bytes, self.domain)?;
        if root.entry_count != old_count {
            return Err(StorageError::Corrupt(
                "ordered head/root count mismatch".to_owned(),
            ));
        }
        let mut removed = 0_u64;
        let mut io = DirectTimelineTreeIo { index: self };
        let changed = match mutation {
            TreeMutation::Remove(keys) => remove_spans(
                &root.directories,
                root.height.saturating_sub(1),
                &keys,
                &mut io,
                &mut removed,
            )?,
            TreeMutation::PurgePrefix(prefix) => purge_prefix_spans(
                &root.directories,
                root.height.saturating_sub(1),
                &prefix,
                &mut io,
                &mut removed,
            )?,
        };
        if removed == 0 {
            return Ok((old_root_id, 0));
        }
        let (height, directories) = demote_spans(changed, root.height, &mut io)?;
        let count = mutation_count(old_count, removed, &directories)?;
        let root_payload = encode_root::<K>(&directories, count, height, self.domain)?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(
            &TIMELINE_HEAD_ID,
            encode_head::<K>(&root_id, count, self.domain)?,
        )?;
        Ok((root_id, removed))
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
            Some(data) => decode_head::<K>(&data.bytes, self.domain).map(Some),
        }
    }

    /// Read one page for a leading-key prefix in ascending or reverse order.
    ///
    /// Forward returns keys strictly greater than `cursor`; reverse strictly
    /// less. `None` starts at the prefix's first (forward) or last (reverse)
    /// entry and reads the current head; a `Some` cursor reads the root it
    /// pins, so pagination is stable across rebuilds. `next_cursor` is `Some`
    /// whenever the page filled `limit` (a following call may return empty).
    ///
    /// # Errors
    /// [`StorageError::Corrupt`] if the cursor has different key metadata or prefix, or a
    /// record fails to decode; [`StorageError::NotFound`] if a referenced
    /// record is missing.
    pub fn page(
        &self,
        prefix: &K::Prefix,
        cursor: Option<&OrderedCursor<K>>,
        forward: bool,
        limit: usize,
    ) -> Result<OrderedPage<K>, StorageError> {
        self.page_with_snapshot(prefix, cursor, forward, limit)
            .map(|(page, _snapshot_root)| page)
    }

    /// Read a page and return the exact root used for the read.
    pub(crate) fn page_with_snapshot(
        &self,
        prefix: &K::Prefix,
        cursor: Option<&OrderedCursor<K>>,
        forward: bool,
        limit: usize,
    ) -> Result<(OrderedPage<K>, Option<NodeId>), StorageError> {
        let empty = OrderedPage {
            entries: Vec::new(),
            next_cursor: None,
        };
        if limit == 0 {
            return Ok((empty, None));
        }
        validate_geometry::<K>()?;

        let (snapshot_root, key) = match cursor {
            Some(cursor) => {
                if cursor.domain != self.domain
                    || cursor.schema_tag != K::SCHEMA_TAG
                    || cursor.leaf_cap != u32::try_from(K::LEAF_CAP).unwrap_or(u32::MAX)
                    || cursor.dir_fanout != u32::try_from(K::DIR_FANOUT).unwrap_or(u32::MAX)
                {
                    return Err(StorageError::Corrupt(
                        "ordered cursor domain or key geometry mismatch".to_owned(),
                    ));
                }
                if cursor.entry.prefix() != *prefix {
                    return Err(StorageError::Corrupt(
                        "ordered cursor names a different prefix".to_owned(),
                    ));
                }
                (cursor.root_id, Some(cursor.entry()))
            }
            None => match self.head()? {
                Some((snapshot_root, _count)) => (snapshot_root, None),
                None => return Ok((empty, None)),
            },
        };

        let root_bytes = self
            .engine
            .get(&self.collection_id, &snapshot_root)?
            .ok_or(StorageError::NotFound(snapshot_root))?;
        let root = decode_root::<K>(&root_bytes.bytes, self.domain)?;
        if root.directories.is_empty() {
            return Ok((empty, Some(snapshot_root)));
        }

        let mut out: Vec<K> = Vec::with_capacity(limit);
        let level = root.height.saturating_sub(1);
        if forward {
            self.forward_page(&root.directories, level, prefix, key, limit, &mut out)?;
        } else {
            self.reverse_page(&root.directories, level, prefix, key, limit, &mut out)?;
        }
        let next_cursor = if out.len() == limit {
            out.last()
                .map(|entry| self.cursor(snapshot_root, *entry))
                .transpose()?
        } else {
            None
        };
        Ok((
            OrderedPage {
                entries: out,
                next_cursor,
            },
            Some(snapshot_root),
        ))
    }

    fn forward_page(
        &self,
        spans: &[SpanDescriptor<K>],
        level: u8,
        prefix: &K::Prefix,
        key: Option<K>,
        limit: usize,
        out: &mut Vec<K>,
    ) -> Result<(), StorageError> {
        let start = forward_leaf_start(spans, key, prefix);
        for span in spans.iter().skip(start) {
            if span.min_entry.prefix() > *prefix {
                break;
            }
            if span.max_entry.prefix() < *prefix {
                continue;
            }
            if level == 0 {
                let leaf = self.load_leaf(&span.child_id)?;
                for entry in &leaf {
                    if entry.prefix() < *prefix {
                        continue;
                    }
                    if entry.prefix() > *prefix {
                        return Ok(());
                    }
                    if key.is_some_and(|k| *entry <= k) {
                        continue;
                    }
                    out.push(*entry);
                    if out.len() == limit {
                        return Ok(());
                    }
                }
            } else {
                let children = self.load_directory(span.child_id, level)?;
                self.forward_page(&children, level.saturating_sub(1), prefix, key, limit, out)?;
                if out.len() == limit {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn reverse_page(
        &self,
        spans: &[SpanDescriptor<K>],
        level: u8,
        prefix: &K::Prefix,
        key: Option<K>,
        limit: usize,
        out: &mut Vec<K>,
    ) -> Result<(), StorageError> {
        let Some(start) = reverse_leaf_start(spans, key, prefix) else {
            return Ok(());
        };
        for index in (0..=start).rev() {
            let span = &spans[index];
            if span.min_entry.prefix() > *prefix {
                continue;
            }
            if span.max_entry.prefix() < *prefix {
                break;
            }
            if level == 0 {
                let leaf = self.load_leaf(&span.child_id)?;
                for entry in leaf.iter().rev() {
                    if entry.prefix() > *prefix {
                        continue;
                    }
                    if entry.prefix() < *prefix {
                        return Ok(());
                    }
                    if key.is_some_and(|k| *entry >= k) {
                        continue;
                    }
                    out.push(*entry);
                    if out.len() == limit {
                        return Ok(());
                    }
                }
            } else {
                let children = self.load_directory(span.child_id, level)?;
                self.reverse_page(&children, level.saturating_sub(1), prefix, key, limit, out)?;
                if out.len() == limit {
                    return Ok(());
                }
            }
        }
        Ok(())
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

    fn load_leaf(&self, id: &NodeId) -> Result<Vec<K>, StorageError> {
        let bytes = self
            .engine
            .get(&self.collection_id, id)?
            .ok_or(StorageError::NotFound(*id))?;
        decode_leaf::<K>(&bytes.bytes)
    }

    fn load_directory(
        &self,
        id: NodeId,
        expected_level: u8,
    ) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
        let bytes = self
            .engine
            .get(&self.collection_id, &id)?
            .ok_or(StorageError::NotFound(id))?;
        decode_directory::<K>(&bytes.bytes, expected_level)
    }
}

struct DirectTimelineTreeIo<'index, 'engine, S: StorageEngine + ?Sized, K: OrderedKey> {
    index: &'index OrderedIndex<'engine, S, K>,
}

impl<S: StorageEngine + ?Sized, K: OrderedKey> TimelineTreeIo<K>
    for DirectTimelineTreeIo<'_, '_, S, K>
{
    fn load_leaf(&mut self, span: &SpanDescriptor<K>) -> Result<Vec<K>, StorageError> {
        self.index.load_leaf(&span.child_id)
    }

    fn load_directory(
        &mut self,
        span: &SpanDescriptor<K>,
        level: u8,
    ) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
        self.index.load_directory(span.child_id, level)
    }

    fn write_leaf(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError> {
        self.index.write_record(&id, payload)
    }

    fn write_directory(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError> {
        self.index.write_record(&id, payload)
    }
}

struct TransactionTimelineTreeIo<
    'index,
    'engine,
    'tx,
    'db,
    S: StorageEngine + ?Sized,
    K: OrderedKey,
> {
    index: &'index OrderedIndex<'engine, S, K>,
    transaction: &'tx DatabaseTransaction<'db>,
    pool: ShardType,
    staged: &'tx mut StagedTree,
}

impl<S: StorageEngine + ?Sized, K: OrderedKey> TimelineTreeIo<K>
    for TransactionTimelineTreeIo<'_, '_, '_, '_, S, K>
{
    fn load_leaf(&mut self, span: &SpanDescriptor<K>) -> Result<Vec<K>, StorageError> {
        if let Some(index) = self.staged.by_id.get(&span.child_id).copied() {
            let record = self.staged.records.get(index).ok_or_else(|| {
                StorageError::Internal("staged ordered record index is invalid".to_owned())
            })?;
            return match record {
                StagedRecord::Leaf(_, payload) => decode_leaf::<K>(payload),
                StagedRecord::Directory(_, _) => Err(StorageError::Corrupt(
                    "ordered leaf id names a directory".to_owned(),
                )),
            };
        }
        let records =
            self.transaction
                .get(self.pool, &self.index.collection_id, &[span.child_id])?;
        let record = records
            .first()
            .and_then(Option::as_ref)
            .ok_or(StorageError::NotFound(span.child_id))?;
        decode_leaf::<K>(&record.bytes)
    }

    fn load_directory(
        &mut self,
        span: &SpanDescriptor<K>,
        level: u8,
    ) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
        if let Some(index) = self.staged.by_id.get(&span.child_id).copied() {
            let record = self.staged.records.get(index).ok_or_else(|| {
                StorageError::Internal("staged ordered record index is invalid".to_owned())
            })?;
            return match record {
                StagedRecord::Directory(_, payload) => decode_directory::<K>(payload, level),
                StagedRecord::Leaf(_, _) => Err(StorageError::Corrupt(
                    "ordered directory id names a leaf".to_owned(),
                )),
            };
        }
        let records =
            self.transaction
                .get(self.pool, &self.index.collection_id, &[span.child_id])?;
        let record = records
            .first()
            .and_then(Option::as_ref)
            .ok_or(StorageError::NotFound(span.child_id))?;
        decode_directory::<K>(&record.bytes, level)
    }

    fn write_leaf(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError> {
        self.staged.push(id, StagedRecord::Leaf(id, payload));
        Ok(())
    }

    fn write_directory(&mut self, id: NodeId, payload: Vec<u8>) -> Result<(), StorageError> {
        self.staged.push(id, StagedRecord::Directory(id, payload));
        Ok(())
    }
}

fn forward_leaf_start<K: OrderedKey>(
    leaves: &[LeafDescriptor<K>],
    key: Option<K>,
    prefix: &K::Prefix,
) -> usize {
    match key {
        Some(key) => leaves
            .partition_point(|leaf| leaf.min_entry <= key)
            .saturating_sub(1),
        None => leaves.partition_point(|leaf| leaf.max_entry.prefix() < *prefix),
    }
}

fn reverse_leaf_start<K: OrderedKey>(
    leaves: &[LeafDescriptor<K>],
    key: Option<K>,
    prefix: &K::Prefix,
) -> Option<usize> {
    match key {
        Some(bound) => leaves
            .partition_point(|leaf| leaf.min_entry < bound)
            .checked_sub(1),
        None => leaves
            .partition_point(|leaf| leaf.min_entry.prefix() <= *prefix)
            .checked_sub(1),
    }
}

fn encode_leaf<K: OrderedKey>(entries: &[K]) -> Result<Vec<u8>, StorageError> {
    validate_geometry::<K>()?;
    let count = u16::try_from(entries.len())
        .map_err(|_| StorageError::Internal("ordered leaf exceeds u16 entries".to_owned()))?;
    let capacity = entries
        .len()
        .checked_mul(K::WIRE_LEN)
        .and_then(|len| len.checked_add(LEAF_HEADER_LEN))
        .ok_or_else(|| StorageError::Internal("ordered leaf length overflow".to_owned()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&TIMELINE_LEAF_MAGIC);
    out.push(LEAF_FORMAT_VERSION);
    out.extend_from_slice(&count.to_be_bytes());
    for entry in entries {
        entry.encode_into(&mut out);
    }
    Ok(out)
}

fn decode_leaf<K: OrderedKey>(bytes: &[u8]) -> Result<Vec<K>, StorageError> {
    validate_geometry::<K>()?;
    if bytes.len() < LEAF_HEADER_LEN
        || bytes[..4] != TIMELINE_LEAF_MAGIC
        || bytes[4] != LEAF_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline leaf header".to_owned()));
    }
    let count = u16::from_be_bytes([bytes[5], bytes[6]]) as usize;
    let expected = count
        .checked_mul(K::WIRE_LEN)
        .and_then(|body| body.checked_add(LEAF_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline leaf length".to_owned()))?;
    if bytes.len() != expected {
        return Err(StorageError::Corrupt("timeline leaf length".to_owned()));
    }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let start = index
            .checked_mul(K::WIRE_LEN)
            .and_then(|offset| offset.checked_add(LEAF_HEADER_LEN))
            .ok_or_else(|| StorageError::Corrupt("timeline leaf offset".to_owned()))?;
        let end = start
            .checked_add(K::WIRE_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline leaf offset".to_owned()))?;
        entries.push(K::decode(&bytes[start..end])?);
    }
    Ok(entries)
}

fn encode_directory<K: OrderedKey>(
    spans: &[SpanDescriptor<K>],
    level: u8,
) -> Result<Vec<u8>, StorageError> {
    validate_geometry::<K>()?;
    if spans.is_empty() || spans.len() > K::DIR_FANOUT {
        return Err(StorageError::Internal(
            "timeline directory occupancy out of range".to_owned(),
        ));
    }
    let span_count = u16::try_from(spans.len())
        .map_err(|_| StorageError::Internal("timeline directory exceeds u16 spans".to_owned()))?;
    let descriptor_len = span_descriptor_len::<K>()?;
    let capacity = spans
        .len()
        .checked_mul(descriptor_len)
        .and_then(|len| len.checked_add(DIRECTORY_HEADER_LEN))
        .ok_or_else(|| StorageError::Internal("ordered directory length overflow".to_owned()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&TIMELINE_DIRECTORY_MAGIC);
    out.push(DIRECTORY_FORMAT_VERSION);
    out.push(level);
    out.extend_from_slice(&span_count.to_be_bytes());
    for span in spans {
        encode_span(span, &mut out);
    }
    Ok(out)
}

fn decode_directory<K: OrderedKey>(
    bytes: &[u8],
    expected_level: u8,
) -> Result<Vec<SpanDescriptor<K>>, StorageError> {
    validate_geometry::<K>()?;
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
    let descriptor_len = span_descriptor_len::<K>()?;
    let expected = span_count
        .checked_mul(descriptor_len)
        .and_then(|body| body.checked_add(DIRECTORY_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline directory length".to_owned()))?;
    if span_count == 0 || span_count > K::DIR_FANOUT || bytes.len() != expected {
        return Err(StorageError::Corrupt(
            "timeline directory length or occupancy".to_owned(),
        ));
    }
    let body = bytes
        .get(DIRECTORY_HEADER_LEN..)
        .ok_or_else(|| StorageError::Corrupt("timeline directory body".to_owned()))?;
    let mut chunks = body.chunks_exact(descriptor_len);
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

fn encode_span<K: OrderedKey>(span: &SpanDescriptor<K>, out: &mut Vec<u8>) {
    span.min_entry.encode_into(out);
    span.max_entry.encode_into(out);
    out.extend_from_slice(&span.entry_count.to_be_bytes());
    out.extend_from_slice(&span.child_id);
}

fn decode_span<K: OrderedKey>(bytes: &[u8]) -> Result<SpanDescriptor<K>, StorageError> {
    if bytes.len() != span_descriptor_len::<K>()? {
        return Err(StorageError::Corrupt("timeline span length".to_owned()));
    }
    let (min_bytes, rest) = bytes
        .split_at_checked(K::WIRE_LEN)
        .ok_or_else(|| StorageError::Corrupt("timeline span min bound".to_owned()))?;
    let (max_bytes, rest) = rest
        .split_at_checked(K::WIRE_LEN)
        .ok_or_else(|| StorageError::Corrupt("timeline span max bound".to_owned()))?;
    let (count_bytes, child_bytes) = rest
        .split_at_checked(8)
        .ok_or_else(|| StorageError::Corrupt("timeline span count".to_owned()))?;
    Ok(SpanDescriptor {
        min_entry: K::decode(min_bytes)?,
        max_entry: K::decode(max_bytes)?,
        entry_count: u64::from_be_bytes(
            <[u8; 8]>::try_from(count_bytes)
                .map_err(|_| StorageError::Corrupt("timeline span count".to_owned()))?,
        ),
        child_id: <[u8; 16]>::try_from(child_bytes)
            .map_err(|_| StorageError::Corrupt("timeline span child".to_owned()))?,
    })
}
fn encode_root<K: OrderedKey>(
    directories: &[DirectoryDescriptor<K>],
    count: u64,
    height: u8,
    domain: DomainTag,
) -> Result<Vec<u8>, StorageError> {
    validate_geometry::<K>()?;
    let descriptor_len = span_descriptor_len::<K>()?;
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
    let capacity = directories
        .len()
        .checked_mul(descriptor_len)
        .and_then(|len| len.checked_add(ROOT_HEADER_LEN))
        .ok_or_else(|| StorageError::Internal("ordered root length overflow".to_owned()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&TIMELINE_ROOT_MAGIC);
    out.push(ROOT_FORMAT_VERSION);
    out.push(height);
    out.extend_from_slice(domain.as_bytes());
    out.extend_from_slice(&K::SCHEMA_TAG);
    out.extend_from_slice(
        &u32::try_from(K::LEAF_CAP)
            .map_err(|_| StorageError::Internal("ordered leaf cap exceeds u32".to_owned()))?
            .to_be_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(K::DIR_FANOUT)
            .map_err(|_| StorageError::Internal("ordered fanout exceeds u32".to_owned()))?
            .to_be_bytes(),
    );
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&directory_count.to_be_bytes());
    for directory in directories {
        encode_span(directory, &mut out);
    }
    Ok(out)
}

fn decode_root_header<K: OrderedKey>(
    bytes: &[u8],
    expected_domain: DomainTag,
) -> Result<(u8, DomainTag, u64, usize), StorageError> {
    validate_geometry::<K>()?;
    if bytes.len() < ROOT_HEADER_LEN
        || bytes[..4] != TIMELINE_ROOT_MAGIC
        || bytes[4] != ROOT_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline root header".to_owned()));
    }
    let height = bytes[5];
    let domain = DomainTag::new(
        bytes[6..22]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline root domain".to_owned()))?,
    );
    if domain != expected_domain {
        return Err(StorageError::Corrupt(
            "timeline root domain mismatch".to_owned(),
        ));
    }
    let schema_tag = <[u8; 4]>::try_from(&bytes[22..26])
        .map_err(|_| StorageError::Corrupt("timeline root schema".to_owned()))?;
    let leaf_cap = u32::from_be_bytes(
        bytes[26..30]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline root leaf cap".to_owned()))?,
    );
    let dir_fanout = u32::from_be_bytes(
        bytes[30..34]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline root fanout".to_owned()))?,
    );
    if schema_tag != K::SCHEMA_TAG
        || leaf_cap != u32::try_from(K::LEAF_CAP).unwrap_or(u32::MAX)
        || dir_fanout != u32::try_from(K::DIR_FANOUT).unwrap_or(u32::MAX)
    {
        return Err(StorageError::Corrupt(
            "timeline root key geometry mismatch".to_owned(),
        ));
    }
    let entry_count = u64::from_be_bytes(
        bytes[34..42]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline root count".to_owned()))?,
    );
    let directory_count = usize::try_from(u32::from_be_bytes([
        bytes[42], bytes[43], bytes[44], bytes[45],
    ]))
    .map_err(|_| StorageError::Corrupt("ordered root directory count".to_owned()))?;
    Ok((height, domain, entry_count, directory_count))
}

fn decode_root_spans<K: OrderedKey>(
    body: &[u8],
    descriptor_count: usize,
    expected_count: u64,
) -> Result<Vec<DirectoryDescriptor<K>>, StorageError> {
    let descriptor_len = span_descriptor_len::<K>()?;
    let mut chunks = body.chunks_exact(descriptor_len);
    if !chunks.remainder().is_empty() || chunks.len() != descriptor_count {
        return Err(StorageError::Corrupt(
            "timeline root trailing bytes".to_owned(),
        ));
    }
    let mut directories: Vec<DirectoryDescriptor<K>> = Vec::with_capacity(descriptor_count);
    let mut previous_max = None;
    let mut counted = 0_u64;
    for chunk in &mut chunks {
        let descriptor = decode_span::<K>(chunk)?;
        if descriptor.min_entry > descriptor.max_entry
            || descriptor.entry_count == 0
            || previous_max.is_some_and(|previous| previous >= descriptor.min_entry)
        {
            return Err(StorageError::Corrupt(
                "timeline root directory bounds or count".to_owned(),
            ));
        }
        counted = counted
            .checked_add(descriptor.entry_count)
            .ok_or_else(|| StorageError::Corrupt("timeline root count overflow".to_owned()))?;
        previous_max = Some(descriptor.max_entry);
        directories.push(descriptor);
    }
    if counted != expected_count || (expected_count == 0) != directories.is_empty() {
        return Err(StorageError::Corrupt(
            "timeline root count mismatch".to_owned(),
        ));
    }
    Ok(directories)
}

fn decode_root<K: OrderedKey>(
    bytes: &[u8],
    expected_domain: DomainTag,
) -> Result<TimelineRoot<K>, StorageError> {
    let descriptor_len = span_descriptor_len::<K>()?;
    let (height, domain, entry_count, directory_count) =
        decode_root_header::<K>(bytes, expected_domain)?;
    let expected = directory_count
        .checked_mul(descriptor_len)
        .and_then(|body| body.checked_add(ROOT_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline root length".to_owned()))?;
    if bytes.len() != expected {
        return Err(StorageError::Corrupt("timeline root length".to_owned()));
    }
    let body = bytes
        .get(ROOT_HEADER_LEN..)
        .ok_or_else(|| StorageError::Corrupt("timeline root body".to_owned()))?;
    let directories = decode_root_spans::<K>(body, directory_count, entry_count)?;
    if (height == 0) != directories.is_empty() {
        return Err(StorageError::Corrupt("timeline root height".to_owned()));
    }
    Ok(TimelineRoot {
        domain,
        height,
        entry_count,
        directories,
    })
}
fn encode_head<K: OrderedKey>(
    root_id: &NodeId,
    count: u64,
    domain: DomainTag,
) -> Result<Vec<u8>, StorageError> {
    validate_geometry::<K>()?;
    let mut out = Vec::with_capacity(HEAD_LEN);
    out.extend_from_slice(&TIMELINE_HEAD_MAGIC);
    out.push(HEAD_FORMAT_VERSION);
    out.extend_from_slice(domain.as_bytes());
    out.extend_from_slice(&K::SCHEMA_TAG);
    out.extend_from_slice(
        &u32::try_from(K::LEAF_CAP)
            .map_err(|_| StorageError::Internal("ordered leaf cap exceeds u32".to_owned()))?
            .to_be_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(K::DIR_FANOUT)
            .map_err(|_| StorageError::Internal("ordered fanout exceeds u32".to_owned()))?
            .to_be_bytes(),
    );
    out.extend_from_slice(root_id);
    out.extend_from_slice(&count.to_be_bytes());
    Ok(out)
}

fn decode_head<K: OrderedKey>(
    bytes: &[u8],
    expected_domain: DomainTag,
) -> Result<(NodeId, u64), StorageError> {
    validate_geometry::<K>()?;
    if bytes.len() != HEAD_LEN
        || bytes[..4] != TIMELINE_HEAD_MAGIC
        || bytes[4] != HEAD_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline head".to_owned()));
    }
    let domain = DomainTag::new(
        bytes[5..21]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline head domain".to_owned()))?,
    );
    if domain != expected_domain {
        return Err(StorageError::Corrupt(
            "timeline head domain mismatch".to_owned(),
        ));
    }
    let schema_tag: [u8; 4] = bytes[21..25]
        .try_into()
        .map_err(|_| StorageError::Corrupt("ordered head schema".to_owned()))?;
    let leaf_cap = u32::from_be_bytes(
        bytes[25..29]
            .try_into()
            .map_err(|_| StorageError::Corrupt("ordered head leaf cap".to_owned()))?,
    );
    let dir_fanout = u32::from_be_bytes(
        bytes[29..33]
            .try_into()
            .map_err(|_| StorageError::Corrupt("ordered head fanout".to_owned()))?,
    );
    if schema_tag != K::SCHEMA_TAG
        || usize::try_from(leaf_cap).ok() != Some(K::LEAF_CAP)
        || usize::try_from(dir_fanout).ok() != Some(K::DIR_FANOUT)
    {
        return Err(StorageError::Corrupt(
            "ordered head key schema or geometry mismatch".to_owned(),
        ));
    }
    let mut root_id = [0u8; 16];
    root_id.copy_from_slice(&bytes[33..49]);
    let count = u64::from_be_bytes(
        bytes[49..HEAD_LEN]
            .try_into()
            .map_err(|_| StorageError::Corrupt("timeline head count".to_owned()))?,
    );
    Ok((root_id, count))
}

#[cfg(test)]
#[path = "test_timeline.rs"]
mod tests;
