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

/// Stable logical name shared by both event-order key schemas.
pub const EVENT_TIMELINE_INDEX_NAME: &str = "event-timeline-v1";
const EVENT_KEY_WIRE_LEN: usize = 16 + 8 + 8 + 16;
type DecodedEventKey = ([u8; 16], i64, i64, NodeId);
type TimelineMutationRoots = ((NodeId, u64), (NodeId, u64));
const EVENT_TOPO_SCHEMA: [u8; 4] = *b"ETOP";
const EVENT_STREAM_SCHEMA: [u8; 4] = *b"ESTR";
const MAX_NODE_ID: NodeId = [0xff; 16];

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
}

impl EventTimelineEntry {
    /// Create an event timeline entry.
    #[must_use]
    pub const fn new(
        room_scope: [u8; 16],
        topological_ordering: i64,
        stream_ordering: i64,
        event_node_id: NodeId,
    ) -> Self {
        Self {
            room_scope,
            topological_ordering,
            stream_ordering,
            event_node_id,
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
        }
    }
}

impl From<ScopedSeqKey> for EventTimelineEntry {
    fn from(key: ScopedSeqKey) -> Self {
        Self::new(key.scope, key.primary, key.secondary, key.id)
    }
}

/// `(room, topological ordering, stream ordering, node id)` key.
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
        );
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let (room_scope, primary, secondary, event_node_id) = decode_event_key(bytes)?;
        Ok(Self {
            room_scope,
            topological_ordering: primary,
            stream_ordering: secondary,
            event_node_id,
        })
    }
}

/// `(room, stream ordering, topological ordering, node id)` key.
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
        );
    }

    fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        let (room_scope, stream_ordering, topological_ordering, event_node_id) =
            decode_event_key(bytes)?;
        Ok(Self {
            room_scope,
            stream_ordering,
            topological_ordering,
            event_node_id,
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
) -> Result<Option<(NodeId, NodeId)>, StorageError> {
    for (node_id, data) in payloads {
        transaction
            .put(pool, payload_collection, *node_id, data)
            .map_err(StorageError::Io)?;
    }
    if entries.is_empty() {
        return Ok(None);
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
    Ok(Some((topo_root, stream_root)))
}

/// Read a stream-ordered page bounded by the contiguous visible watermark.
///
/// Reverse reads seek from a synthetic maximal key at the watermark. Forward
/// reads stop at the first greater stream position and rebuild the resume
/// cursor from the last retained entry so a later watermark advance cannot
/// skip entries. Returned cursors pin that read's root; callers resuming across
/// requests must rebind the last key to the current root.
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
                    && cursor.entry().room_scope == *room_scope
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
    let (mut page, snapshot_root) =
        index.page_with_snapshot(room_scope, bounded_cursor.as_ref(), forward, limit)?;
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
    page.next_cursor = match (page.entries.last().copied(), snapshot_root) {
        (Some(last_visible), Some(root_id)) => Some(index.cursor(root_id, last_visible)?),
        _ => None,
    };
    Ok(page)
}

/// Re-anchor a stream cursor to the current index head for a new request.
///
/// # Errors
/// Returns an error if the cursor belongs to another index domain or room.
pub fn rebind_stream_cursor_to_head<S: StorageEngine + ?Sized>(
    index: &EventStreamIndex<'_, S>,
    room_scope: &[u8; 16],
    cursor: &EventStreamCursor,
) -> Result<Option<EventStreamCursor>, StorageError> {
    if cursor.domain() != index.domain() || cursor.entry().room_scope != *room_scope {
        return Err(StorageError::Corrupt(
            "event stream cursor domain or room mismatch".to_owned(),
        ));
    }
    index
        .head()?
        .map(|(root_id, _count)| index.cursor(root_id, cursor.entry()))
        .transpose()
}

/// Stage removal of event keys from both ordering indexes.
///
/// Missing keys are ignored. Payload and locator removal, if desired, stays in
/// the same caller transaction.
///
/// # Errors
/// Propagates ordered-index removal and validation errors.
pub fn stage_remove_event_timeline_batch<S: StorageEngine + ?Sized>(
    transaction: &DatabaseTransaction<'_>,
    pool: ShardType,
    topo_index: &EventTopoIndex<'_, S>,
    stream_index: &EventStreamIndex<'_, S>,
    entries: &[EventTimelineEntry],
) -> Result<Option<TimelineMutationRoots>, StorageError> {
    if entries.is_empty() {
        return Ok(None);
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
    let topo = topo_index.stage_remove(transaction, pool, &topo_entries)?;
    let stream = stream_index.stage_remove(transaction, pool, &stream_entries)?;
    Ok(Some((topo, stream)))
}

/// Purge every event key in a room from both ordering indexes.
///
/// Covered subtrees are dropped from their span descriptors without reading
/// their records.
///
/// # Errors
/// Propagates ordered-index purge and validation errors.
pub fn stage_purge_room_timeline<S: StorageEngine + ?Sized>(
    transaction: &DatabaseTransaction<'_>,
    pool: ShardType,
    topo_index: &EventTopoIndex<'_, S>,
    stream_index: &EventStreamIndex<'_, S>,
    room_scope: &[u8; 16],
) -> Result<TimelineMutationRoots, StorageError> {
    let topo = topo_index.stage_purge_prefix(transaction, pool, *room_scope)?;
    let stream = stream_index.stage_purge_prefix(transaction, pool, *room_scope)?;
    Ok((topo, stream))
}

/// Stage the index half of an outlier-to-timeline promotion.
///
/// The caller must also update the room flags and any payload metadata in the
/// same transaction.
///
/// # Errors
/// Propagates ordered-index append and validation errors.
pub fn stage_promote_outlier_to_timeline<S: StorageEngine + ?Sized>(
    transaction: &DatabaseTransaction<'_>,
    pool: ShardType,
    payload_collection: [u8; 16],
    topo_index: &EventTopoIndex<'_, S>,
    stream_index: &EventStreamIndex<'_, S>,
    entries: &[EventTimelineEntry],
) -> Result<Option<(NodeId, NodeId)>, StorageError> {
    stage_event_timeline_batch(
        transaction,
        pool,
        payload_collection,
        &[],
        topo_index,
        stream_index,
        entries,
    )
}

/// Page a cursored read, re-anchoring an expired snapshot onto the head once.
///
/// Active cursors keep strict snapshot isolation. If the pinned root has been
/// reclaimed and the cursor's full key is still available, the read retries
/// from the current head at that key, degrading to keyset pagination instead
/// of surfacing a missing-root error.
///
/// # Errors
/// Propagates ordered-index read errors other than an expired cursor root, and
/// returns the retry error if re-anchored paging still fails.
pub fn page_with_rebind_fallback<S: StorageEngine + ?Sized, K: OrderedKey>(
    index: &OrderedIndex<'_, S, K>,
    prefix: &K::Prefix,
    cursor: Option<&OrderedCursor<K>>,
    forward: bool,
    limit: usize,
) -> Result<OrderedPage<K>, StorageError> {
    match index.page(prefix, cursor, forward, limit) {
        Ok(page) => Ok(page),
        Err(StorageError::NotFound(missing))
            if cursor.is_some_and(|cursor| cursor.root_id() == missing) =>
        {
            let Some(pinned) = cursor else {
                return Ok(OrderedPage {
                    entries: Vec::new(),
                    next_cursor: None,
                });
            };
            let Some((root_id, _count)) = index.head()? else {
                return Ok(OrderedPage {
                    entries: Vec::new(),
                    next_cursor: None,
                });
            };
            let rebased = index.cursor(root_id, pinned.entry())?;
            index.page(prefix, Some(&rebased), forward, limit)
        }
        Err(error) => Err(error),
    }
}

fn max_stream_key(room_scope: [u8; 16], persisted_up_to: i64) -> EventStreamKey {
    EventStreamKey {
        room_scope,
        stream_ordering: persisted_up_to,
        topological_ordering: i64::MAX,
        event_node_id: MAX_NODE_ID,
    }
}

fn encode_event_key(
    out: &mut Vec<u8>,
    room_scope: [u8; 16],
    primary: i64,
    secondary: i64,
    event_node_id: NodeId,
) {
    out.extend_from_slice(&room_scope);
    out.extend_from_slice(&encode_i64(primary));
    out.extend_from_slice(&encode_i64(secondary));
    out.extend_from_slice(&event_node_id);
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
    Ok((room_scope, primary, secondary, event_node_id))
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
