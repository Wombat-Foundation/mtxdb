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
//! (room_id, timeline_order, node_id, event_ref)
//! ```
//!
//! - `room_id` scopes the index; a timeline read is a room-prefix range.
//! - `timeline_order` is caller-supplied (Matrix: topological depth). It is
//!   **persisted**; the index never recomputes it. The caller must define a
//!   deterministic fallback for events without a computable position (missing
//!   parents, cycles) before building, because the in-memory DAG order cannot be
//!   regenerated reliably after reopening.
//! - `node_id` is the operational 128-bit id of the stored event and the
//!   total-order tie-break, so equal `timeline_order` values never tie.
//! - `event_ref` is a caller-supplied 32-byte event identity (reference hash),
//!   kept as the final disambiguator so two ids for the same content stay
//!   distinct and ordered deterministically.
//!
//! Layout (all content-addressed, immutable except the head):
//! - **leaf** (`TMLF`): up to [`TIMELINE_LEAF_CAP`] sorted entries;
//! - **root** (`TMLR`): a manifest of `(first_key, leaf_id)` in order plus the
//!   total entry count;
//! - **head** (`TMLH`): a fixed-id record holding the current root id, the only
//!   mutable id (last write wins).
//!
//! A rebuild writes new leaves and a new root and then swaps the head; old
//! records survive until GC, so a [`TimelineCursor`] that pins the root it
//! started on keeps paginating that snapshot even after an update. Forward
//! paging returns keys strictly greater than the cursor, reverse strictly less,
//! so resuming is exact and idempotent in both directions.

use bytes::Bytes;

use crate::storage::{DigestAlgorithm, NodeData, NodeId, StorageEngine, StorageError};
use crate::template::{
    derive_collection_id, CollectionMetadata, FrameIdPolicy, PayloadPolicy, RecordIdentityRule,
    MEMBER_NAMESPACE_INTL,
};

/// Version byte of every timeline record and the cursor encoding.
pub const TIMELINE_FORMAT_VERSION: u8 = 0x01;
/// Maximum entries per leaf record.
pub const TIMELINE_LEAF_CAP: usize = 128;

/// Magic prefix of a leaf record (`b"TMLF"`).
pub const TIMELINE_LEAF_MAGIC: [u8; 4] = *b"TMLF";
/// Magic prefix of a root manifest record (`b"TMLR"`).
pub const TIMELINE_ROOT_MAGIC: [u8; 4] = *b"TMLR";
/// Magic prefix of the head record (`b"TMLH"`).
pub const TIMELINE_HEAD_MAGIC: [u8; 4] = *b"TMLH";
/// Magic prefix of an opaque cursor (`b"TMLC"`).
pub const TIMELINE_CURSOR_MAGIC: [u8; 4] = *b"TMLC";

/// Reserved node id of the head record (the only mutable id in the index).
pub const TIMELINE_HEAD_ID: NodeId = *b"MTXD-TML-HEAD-v1";

/// A caller-supplied event identity (reference hash).
pub type EventRef = [u8; 32];

const ENTRY_LEN: usize = 16 + 8 + 16 + 32;
const LEAF_HEADER_LEN: usize = 7;
const ROOT_HEADER_LEN: usize = 17;
const ROOT_LEAF_LEN: usize = ENTRY_LEN + 16;
const HEAD_LEN: usize = 5 + 16 + 8;
const CURSOR_LEN: usize = 5 + 16 + 16 + 8 + 16 + 32;
const MAX_NODE_ID: NodeId = [0xff; 16];
const MAX_EVENT_REF: EventRef = [0xff; 32];

/// One ordered entry: the full `(room, order, node, event_ref)` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimelineEntry {
    /// Room scope; the leading key component.
    pub room_id: [u8; 16],
    /// Caller-supplied timeline order (e.g. topological depth).
    pub order: u64,
    /// The event's operational 128-bit node id; total-order tie-break.
    pub node_id: NodeId,
    /// Caller-supplied 32-byte event identity; final disambiguator.
    pub event_ref: EventRef,
}

impl TimelineEntry {
    /// Construct an entry.
    #[must_use]
    pub const fn new(room_id: [u8; 16], order: u64, node_id: NodeId, event_ref: EventRef) -> Self {
        Self {
            room_id,
            order,
            node_id,
            event_ref,
        }
    }

    fn encode_into(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.room_id);
        out.extend_from_slice(&self.order.to_be_bytes());
        out.extend_from_slice(&self.node_id);
        out.extend_from_slice(&self.event_ref);
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        if bytes.len() != ENTRY_LEN {
            return Err(StorageError::Corrupt("timeline entry length".to_owned()));
        }
        let mut room_id = [0u8; 16];
        room_id.copy_from_slice(&bytes[..16]);
        let order = u64::from_be_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| StorageError::Corrupt("timeline entry order".to_owned()))?,
        );
        let mut node_id = [0u8; 16];
        node_id.copy_from_slice(&bytes[24..40]);
        let mut event_ref = [0u8; 32];
        event_ref.copy_from_slice(&bytes[40..ENTRY_LEN]);
        Ok(Self {
            room_id,
            order,
            node_id,
            event_ref,
        })
    }
}

/// An opaque, stable cursor naming the snapshot root plus the last entry read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineCursor {
    root_id: NodeId,
    room_id: [u8; 16],
    order: u64,
    node_id: NodeId,
    event_ref: EventRef,
}

impl TimelineCursor {
    /// Build a cursor for `root_id` at `entry`.
    #[must_use]
    pub const fn new(root_id: NodeId, entry: TimelineEntry) -> Self {
        Self {
            root_id,
            room_id: entry.room_id,
            order: entry.order,
            node_id: entry.node_id,
            event_ref: entry.event_ref,
        }
    }

    /// The snapshot root this cursor pins.
    #[must_use]
    pub const fn root_id(&self) -> NodeId {
        self.root_id
    }

    /// The entry this cursor names.
    #[must_use]
    pub const fn entry(&self) -> TimelineEntry {
        TimelineEntry {
            room_id: self.room_id,
            order: self.order,
            node_id: self.node_id,
            event_ref: self.event_ref,
        }
    }

    /// Encode the cursor as opaque, versioned bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CURSOR_LEN);
        out.extend_from_slice(&TIMELINE_CURSOR_MAGIC);
        out.push(TIMELINE_FORMAT_VERSION);
        out.extend_from_slice(&self.root_id);
        out.extend_from_slice(&self.room_id);
        out.extend_from_slice(&self.order.to_be_bytes());
        out.extend_from_slice(&self.node_id);
        out.extend_from_slice(&self.event_ref);
        out
    }

    /// Decode cursor bytes produced by [`Self::encode`].
    ///
    /// Returns `None` for a wrong magic, version, or length.
    #[must_use]
    #[allow(
        clippy::similar_names,
        reason = "root_id/room_id are the wire field names"
    )]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != CURSOR_LEN
            || bytes[..4] != TIMELINE_CURSOR_MAGIC
            || bytes[4] != TIMELINE_FORMAT_VERSION
        {
            return None;
        }
        let mut root_id = [0u8; 16];
        root_id.copy_from_slice(&bytes[5..21]);
        let mut room_id = [0u8; 16];
        room_id.copy_from_slice(&bytes[21..37]);
        let order = u64::from_be_bytes(bytes[37..45].try_into().ok()?);
        let mut node_id = [0u8; 16];
        node_id.copy_from_slice(&bytes[45..61]);
        let mut event_ref = [0u8; 32];
        event_ref.copy_from_slice(&bytes[61..CURSOR_LEN]);
        Some(Self {
            root_id,
            room_id,
            order,
            node_id,
            event_ref,
        })
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

fn room_start(room_id: &[u8; 16]) -> TimelineEntry {
    TimelineEntry::new(*room_id, 0, [0u8; 16], [0u8; 32])
}

fn room_end(room_id: &[u8; 16]) -> TimelineEntry {
    TimelineEntry::new(*room_id, u64::MAX, MAX_NODE_ID, MAX_EVENT_REF)
}

/// A persisted timeline index over an existing storage engine.
pub struct TimelineIndex<'a, S: StorageEngine + ?Sized> {
    engine: &'a S,
    name: String,
    collection_id: [u8; 16],
}

impl<'a, S: StorageEngine + ?Sized> TimelineIndex<'a, S> {
    /// Open (or create) the named timeline index.
    #[must_use]
    pub fn open(engine: &'a S, name: &str) -> Self {
        Self {
            engine,
            name: name.to_owned(),
            collection_id: timeline_collection_id(name),
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

        let mut manifest: Vec<(TimelineEntry, NodeId)> = Vec::new();
        for chunk in sorted.chunks(TIMELINE_LEAF_CAP) {
            let payload = encode_leaf(chunk)?;
            let id = content_id(&payload);
            self.write_record(&id, payload)?;
            let first = *chunk
                .first()
                .ok_or_else(|| StorageError::Internal("empty timeline leaf".to_owned()))?;
            manifest.push((first, id));
        }

        let count = u64::try_from(sorted.len()).unwrap_or(u64::MAX);
        let root_payload = encode_root(&manifest, count)?;
        let root_id = content_id(&root_payload);
        self.write_record(&root_id, root_payload)?;
        self.write_record(&TIMELINE_HEAD_ID, encode_head(&root_id, count))?;
        Ok(root_id)
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
    #[allow(clippy::similar_names, reason = "room_id/root_id are the domain names")]
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

        let (root_id, key) = match cursor {
            Some(cursor) => {
                if cursor.room_id != *room_id {
                    return Err(StorageError::Corrupt(
                        "timeline cursor names a different room".to_owned(),
                    ));
                }
                (cursor.root_id, Some(cursor.entry()))
            }
            None => match self.head()? {
                Some((root_id, _count)) => {
                    let key = if forward {
                        room_start(room_id)
                    } else {
                        room_end(room_id)
                    };
                    (root_id, Some(key))
                }
                None => return Ok(empty),
            },
        };

        let root_bytes = self
            .engine
            .get(&self.collection_id, &root_id)?
            .ok_or(StorageError::NotFound(root_id))?;
        let manifest = decode_root(&root_bytes.bytes)?;
        if manifest.is_empty() {
            return Ok(empty);
        }

        let mut out: Vec<TimelineEntry> = Vec::with_capacity(limit);
        if forward {
            let start = forward_start(&manifest, key);
            for (_, leaf_id) in manifest.iter().skip(start) {
                let leaf = self.load_leaf(leaf_id)?;
                for entry in &leaf {
                    if entry.room_id < *room_id {
                        continue;
                    }
                    if entry.room_id > *room_id {
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
                            next_cursor: Some(TimelineCursor::new(root_id, *entry)),
                            entries: out,
                        });
                    }
                }
            }
        } else {
            let Some(start) = reverse_start(&manifest, key) else {
                return Ok(empty);
            };
            for index in (0..=start).rev() {
                let leaf = self.load_leaf(&manifest[index].1)?;
                for entry in leaf.iter().rev() {
                    if entry.room_id > *room_id {
                        continue;
                    }
                    if entry.room_id < *room_id {
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
                            next_cursor: Some(TimelineCursor::new(root_id, *entry)),
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
}

fn forward_start(manifest: &[(TimelineEntry, NodeId)], key: Option<TimelineEntry>) -> usize {
    let Some(key) = key else {
        return 0;
    };
    // The first leaf that can hold a key greater than `key` is the last leaf
    // whose first key is <= `key`; if every leaf starts after `key`, leaf 0.
    manifest
        .partition_point(|entry| entry.0 <= key)
        .saturating_sub(1)
}

fn reverse_start(
    manifest: &[(TimelineEntry, NodeId)],
    key: Option<TimelineEntry>,
) -> Option<usize> {
    let bound = key?;
    // The last leaf that can hold a key less than `bound` starts strictly
    // before it; when none does, there is nothing to return.
    manifest
        .partition_point(|entry| entry.0 < bound)
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
    out.push(TIMELINE_FORMAT_VERSION);
    out.extend_from_slice(&count.to_be_bytes());
    for entry in entries {
        entry.encode_into(&mut out);
    }
    Ok(out)
}

fn decode_leaf(bytes: &[u8]) -> Result<Vec<TimelineEntry>, StorageError> {
    if bytes.len() < LEAF_HEADER_LEN
        || bytes[..4] != TIMELINE_LEAF_MAGIC
        || bytes[4] != TIMELINE_FORMAT_VERSION
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

fn encode_root(manifest: &[(TimelineEntry, NodeId)], count: u64) -> Result<Vec<u8>, StorageError> {
    let leaf_count = u32::try_from(manifest.len())
        .map_err(|_| StorageError::Internal("timeline root exceeds u32 leaves".to_owned()))?;
    let mut out = Vec::with_capacity(
        manifest
            .len()
            .saturating_mul(ROOT_LEAF_LEN)
            .saturating_add(ROOT_HEADER_LEN),
    );
    out.extend_from_slice(&TIMELINE_ROOT_MAGIC);
    out.push(TIMELINE_FORMAT_VERSION);
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&leaf_count.to_be_bytes());
    for (first, leaf_id) in manifest {
        first.encode_into(&mut out);
        out.extend_from_slice(leaf_id);
    }
    Ok(out)
}

fn decode_root(bytes: &[u8]) -> Result<Vec<(TimelineEntry, NodeId)>, StorageError> {
    if bytes.len() < ROOT_HEADER_LEN
        || bytes[..4] != TIMELINE_ROOT_MAGIC
        || bytes[4] != TIMELINE_FORMAT_VERSION
    {
        return Err(StorageError::Corrupt("timeline root header".to_owned()));
    }
    let leaf_count = u32::from_be_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]) as usize;
    let expected = leaf_count
        .checked_mul(ROOT_LEAF_LEN)
        .and_then(|body| body.checked_add(ROOT_HEADER_LEN))
        .ok_or_else(|| StorageError::Corrupt("timeline root length".to_owned()))?;
    if bytes.len() != expected {
        return Err(StorageError::Corrupt("timeline root length".to_owned()));
    }
    let mut manifest = Vec::with_capacity(leaf_count);
    for index in 0..leaf_count {
        let start = index
            .checked_mul(ROOT_LEAF_LEN)
            .and_then(|offset| offset.checked_add(ROOT_HEADER_LEN))
            .ok_or_else(|| StorageError::Corrupt("timeline root offset".to_owned()))?;
        let entry_end = start
            .checked_add(ENTRY_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline root offset".to_owned()))?;
        let next_leaf = start
            .checked_add(ROOT_LEAF_LEN)
            .ok_or_else(|| StorageError::Corrupt("timeline root offset".to_owned()))?;
        let first = TimelineEntry::decode(&bytes[start..entry_end])?;
        let mut leaf_id = [0u8; 16];
        leaf_id.copy_from_slice(&bytes[entry_end..next_leaf]);
        manifest.push((first, leaf_id));
    }
    Ok(manifest)
}

fn encode_head(root_id: &NodeId, count: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD_LEN);
    out.extend_from_slice(&TIMELINE_HEAD_MAGIC);
    out.push(TIMELINE_FORMAT_VERSION);
    out.extend_from_slice(root_id);
    out.extend_from_slice(&count.to_be_bytes());
    out
}

fn decode_head(bytes: &[u8]) -> Result<(NodeId, u64), StorageError> {
    if bytes.len() != HEAD_LEN
        || bytes[..4] != TIMELINE_HEAD_MAGIC
        || bytes[4] != TIMELINE_FORMAT_VERSION
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
