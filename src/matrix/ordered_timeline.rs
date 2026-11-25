//! Matrix event timeline adapters over the protocol-neutral ordered index.
//!
//! The two key types intentionally differ: topological paging orders by
//! `(room, depth, stream, ...)`, while stream paging orders by
//! `(room, stream, depth, ...)`. Their cursors are distinct Rust types and
//! their collections are shared across rooms.

use crate::database::DatabaseTransaction;
use crate::layout::ShardType;
use crate::storage::{NodeData, NodeId, StorageEngine, StorageError};
use crate::timeline::{OrderedCursor, OrderedIndex, OrderedKey, OrderedPage, ScopedSeqKey};

const EVENT_TIMELINE_INDEX_NAME: &str = "event-timeline-v1";
const EVENT_KEY_WIRE_LEN: usize = 16 + 8 + 8 + 16 + 32;
type DecodedEventKey = ([u8; 16], i64, i64, NodeId, [u8; 32]);
const EVENT_TOPO_SCHEMA: [u8; 4] = *b"ETOP";
const EVENT_STREAM_SCHEMA: [u8; 4] = *b"ESTR";
const MAX_NODE_ID: NodeId = [0xff; 16];
const MAX_EVENT_TAG: [u8; 32] = [0xff; 32];

/// Matrix event ordering and identity in domain terminology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventTimelineEntry {
    /// Stable room scope used as the leading key prefix.
    pub room_scope: [u8; 16],
    /// Matrix topological ordering, also called depth.
    pub topological_ordering: i64,
    /// Matrix stream ordering, signed to preserve backfill positions.
    pub stream_ordering: i64,
    /// Stored event node identifier.
    pub event_node_id: NodeId,
    /// Stable logical event reference, unchanged by payload redaction.
    pub event_ref: [u8; 32],
}

impl EventTimelineEntry {
    /// Create an event timeline entry.
    #[must_use]
    pub const fn new(
        room_scope: [u8; 16],
        topological_ordering: i64,
        stream_ordering: i64,
        event_node_id: NodeId,
        event_ref: [u8; 32],
    ) -> Self {
        Self {
            room_scope,
            topological_ordering,
            stream_ordering,
            event_node_id,
            event_ref,
        }
    }

    /// Project into topological-first key order.
    #[must_use]
    pub const fn topo_key(self) -> EventTopoKey {
        EventTopoKey {
            room_scope: self.room_scope,
            topological_ordering: self.topological_ordering,
            stream_ordering: self.stream_ordering,
            event_node_id: self.event_node_id,
            event_ref: self.event_ref,
        }
    }

    /// Project into stream-first key order.
    #[must_use]
    pub const fn stream_key(self) -> EventStreamKey {
        EventStreamKey {
            room_scope: self.room_scope,
            stream_ordering: self.stream_ordering,
            topological_ordering: self.topological_ordering,
            event_node_id: self.event_node_id,
            event_ref: self.event_ref,
        }
    }
}

impl From<ScopedSeqKey> for EventTimelineEntry {
    fn from(key: ScopedSeqKey) -> Self {
        Self::new(key.scope, key.primary, key.secondary, key.id, key.tag)
    }
}

/// `(room, topological ordering, stream ordering, node id, event ref)` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventTopoKey {
    /// Leading room scope.
    pub room_scope: [u8; 16],
    /// Topological ordering.
    pub topological_ordering: i64,
    /// Stream ordering.
    pub stream_ordering: i64,
    /// Event record node id.
    pub event_node_id: NodeId,
    /// Stable event reference.
    pub event_ref: [u8; 32],
}

impl OrderedKey for EventTopoKey {
    type Prefix = [u8; 16];

    const WIRE_LEN: usize = EVENT_KEY_WIRE_LEN;
    const SCHEMA_TAG: [u8; 4] = EVENT_TOPO_SCHEMA;

    fn prefix(&self) -> Self::Prefix {
        self.room_scope
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        encode_event_key(
            out,
            self.room_scope,
            self.topological_ordering,
            self.stream_ordering,
            self.event_node_id,
            self.event_ref,
        );
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let (room_scope, primary, secondary, event_node_id, event_ref) = decode_event_key(bytes)?;
        Ok(Self {
            room_scope,
            topological_ordering: primary,
            stream_ordering: secondary,
            event_node_id,
            event_ref,
        })
    }
}

/// `(room, stream ordering, topological ordering, node id, event ref)` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventStreamKey {
    /// Leading room scope.
    pub room_scope: [u8; 16],
    /// Stream ordering precedes depth in this key's ordering.
    pub stream_ordering: i64,
    /// Topological ordering.
    pub topological_ordering: i64,
    /// Event record node id.
    pub event_node_id: NodeId,
    /// Stable event reference.
    pub event_ref: [u8; 32],
}

impl OrderedKey for EventStreamKey {
    type Prefix = [u8; 16];

    const WIRE_LEN: usize = EVENT_KEY_WIRE_LEN;
    const SCHEMA_TAG: [u8; 4] = EVENT_STREAM_SCHEMA;

    fn prefix(&self) -> Self::Prefix {
        self.room_scope
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        encode_event_key(
            out,
            self.room_scope,
            self.stream_ordering,
            self.topological_ordering,
            self.event_node_id,
            self.event_ref,
        );
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let (room_scope, stream_ordering, topological_ordering, event_node_id, event_ref) =
            decode_event_key(bytes)?;
        Ok(Self {
            room_scope,
            stream_ordering,
            topological_ordering,
            event_node_id,
            event_ref,
        })
    }
}

/// Topological-order index; its cursor cannot be passed to [`EventStreamIndex`].
pub type EventTopoIndex<'a, S> = OrderedIndex<'a, S, EventTopoKey>;
/// Stream-order index; its cursor cannot be passed to [`EventTopoIndex`].
pub type EventStreamIndex<'a, S> = OrderedIndex<'a, S, EventStreamKey>;
/// Cursor for [`EventTopoIndex`].
pub type EventTopoCursor = OrderedCursor<EventTopoKey>;
/// Cursor for [`EventStreamIndex`].
pub type EventStreamCursor = OrderedCursor<EventStreamKey>;

/// Open the two shared per-schema indexes over one storage pool.
#[must_use]
pub fn open_event_timeline_indexes<S: StorageEngine + ?Sized>(
    engine: &S,
) -> (EventTopoIndex<'_, S>, EventStreamIndex<'_, S>) {
    (
        EventTopoIndex::open(engine, EVENT_TIMELINE_INDEX_NAME),
        EventStreamIndex::open(engine, EVENT_TIMELINE_INDEX_NAME),
    )
}

/// Stage payload records and both ordering roots in one caller transaction.
///
/// The payloads use their existing event collection; the two ordered indexes
/// use separate ORDX collections derived from their key schema tags. A commit
/// publishes all three sets of records atomically.
///
/// # Errors
/// Propagates payload staging errors, malformed index records, pool mismatch,
/// or a capacity/entry-count error from either ordered index.
pub fn stage_event_timeline_batch<S: StorageEngine + ?Sized>(
    transaction: &DatabaseTransaction<'_>,
    pool: ShardType,
    payload_collection: [u8; 16],
    payloads: &[(NodeId, NodeData)],
    topo_index: &EventTopoIndex<'_, S>,
    stream_index: &EventStreamIndex<'_, S>,
    entries: &[EventTimelineEntry],
) -> Result<(NodeId, NodeId), StorageError> {
    for (node_id, data) in payloads {
        transaction
            .put(pool, payload_collection, *node_id, data)
            .map_err(StorageError::Io)?;
    }
    let topo_entries: Vec<_> = entries
        .iter()
        .copied()
        .map(EventTimelineEntry::topo_key)
        .collect();
    let stream_entries: Vec<_> = entries
        .iter()
        .copied()
        .map(EventTimelineEntry::stream_key)
        .collect();
    let topo_root = topo_index.stage_append(transaction, pool, &topo_entries)?;
    let stream_root = stream_index.stage_append(transaction, pool, &stream_entries)?;
    Ok((topo_root, stream_root))
}

/// Read a stream-ordered page bounded by the contiguous visible watermark.
///
/// Reverse reads seek from a synthetic maximal key at the watermark. Forward
/// reads stop at the first greater stream position and rebuild the resume
/// cursor from the last retained entry so a later watermark advance cannot
/// skip entries.
///
/// # Errors
/// Propagates ordered-index read errors, including cursor domain/prefix
/// mismatches and missing referenced records.
pub fn page_stream_up_to<S: StorageEngine + ?Sized>(
    index: &EventStreamIndex<'_, S>,
    room_scope: &[u8; 16],
    cursor: Option<&EventStreamCursor>,
    forward: bool,
    limit: usize,
    persisted_up_to: i64,
) -> Result<OrderedPage<EventStreamKey>, StorageError> {
    let bounded_cursor = if forward {
        cursor.copied()
    } else {
        match cursor {
            Some(cursor)
                if cursor.domain() == index.domain()
                    && cursor.entry().stream_ordering > persisted_up_to =>
            {
                Some(index.cursor(
                    cursor.root_id(),
                    max_stream_key(*room_scope, persisted_up_to),
                )?)
            }
            Some(cursor) => Some(*cursor),
            None => match index.head()? {
                Some((root_id, _)) => {
                    Some(index.cursor(root_id, max_stream_key(*room_scope, persisted_up_to))?)
                }
                None => {
                    return Ok(OrderedPage {
                        entries: Vec::new(),
                        next_cursor: None,
                    })
                }
            },
        }
    };
    let mut page = index.page(room_scope, bounded_cursor.as_ref(), forward, limit)?;
    if !forward {
        return Ok(page);
    }
    let visible_len = page
        .entries
        .iter()
        .position(|entry| entry.stream_ordering > persisted_up_to)
        .unwrap_or(page.entries.len());
    if visible_len == page.entries.len() {
        return Ok(page);
    }
    page.entries.truncate(visible_len);
    page.next_cursor = match (page.entries.last().copied(), page.next_cursor.take()) {
        (Some(last_visible), Some(source_cursor)) => {
            Some(index.cursor(source_cursor.root_id(), last_visible)?)
        }
        _ => None,
    };
    Ok(page)
}

fn max_stream_key(room_scope: [u8; 16], persisted_up_to: i64) -> EventStreamKey {
    EventStreamKey {
        room_scope,
        stream_ordering: persisted_up_to,
        topological_ordering: i64::MAX,
        event_node_id: MAX_NODE_ID,
        event_ref: MAX_EVENT_TAG,
    }
}

fn encode_event_key(
    out: &mut Vec<u8>,
    room_scope: [u8; 16],
    primary: i64,
    secondary: i64,
    event_node_id: NodeId,
    event_ref: [u8; 32],
) {
    out.extend_from_slice(&room_scope);
    out.extend_from_slice(&encode_i64(primary));
    out.extend_from_slice(&encode_i64(secondary));
    out.extend_from_slice(&event_node_id);
    out.extend_from_slice(&event_ref);
}

fn decode_event_key(bytes: &[u8]) -> Result<DecodedEventKey, StorageError> {
    if bytes.len() != EVENT_KEY_WIRE_LEN {
        return Err(StorageError::Corrupt("event ordered key length".to_owned()));
    }
    let mut room_scope = [0; 16];
    room_scope.copy_from_slice(&bytes[..16]);
    let primary = decode_i64(&bytes[16..24])?;
    let secondary = decode_i64(&bytes[24..32])?;
    let mut event_node_id = [0; 16];
    event_node_id.copy_from_slice(&bytes[32..48]);
    let mut event_ref = [0; 32];
    event_ref.copy_from_slice(&bytes[48..80]);
    Ok((room_scope, primary, secondary, event_node_id, event_ref))
}

fn encode_i64(value: i64) -> [u8; 8] {
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    bytes
}

fn decode_i64(bytes: &[u8]) -> Result<i64, StorageError> {
    let mut bytes = <[u8; 8]>::try_from(bytes)
        .map_err(|_| StorageError::Corrupt("event ordered position".to_owned()))?;
    bytes[0] ^= 0x80;
    Ok(i64::from_be_bytes(bytes))
}

#[cfg(test)]
#[path = "test_ordered_timeline.rs"]
mod tests;
