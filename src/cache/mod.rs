use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::storage::{NodeData, NodeId};

/// Maximum number of entries in the node cache.
const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// Index into the intrusive linked list Vec.
type LinkIdx = u32;

/// Sentinel index meaning "no node".
const NIL: LinkIdx = u32::MAX;

/// A node in the intrusive doubly-linked list for LRU ordering.
struct LruNode {
    data: Arc<NodeData>,
    key: NodeId,
    prev: LinkIdx,
    next: LinkIdx,
}

/// The mutable inner state of the LRU cache, protected by a single lock.
struct LruState {
    /// Maps `NodeId` → index into the linked list.
    map: HashMap<NodeId, LinkIdx>,
    /// Doubly-linked list nodes.
    nodes: Vec<LruNode>,
    /// Head of the list (most recently used). NIL if empty.
    head: LinkIdx,
    /// Tail of the list (least recently used). NIL if empty.
    tail: LinkIdx,
}

impl LruState {
    fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            nodes: Vec::with_capacity(cap),
            head: NIL,
            tail: NIL,
        }
    }

    fn move_to_head(&mut self, idx: LinkIdx) {
        if self.head == idx {
            return;
        }
        let prev = self.nodes[idx as usize].prev;
        let next = self.nodes[idx as usize].next;

        if prev != NIL {
            self.nodes[prev as usize].next = next;
        }
        if next != NIL {
            self.nodes[next as usize].prev = prev;
        }
        if self.tail == idx {
            self.tail = prev;
        }

        self.nodes[idx as usize].prev = NIL;
        self.nodes[idx as usize].next = self.head;
        if self.head != NIL {
            self.nodes[self.head as usize].prev = idx;
        }
        self.head = idx;
        if self.tail == NIL {
            self.tail = idx;
        }
    }

    fn evict_lru(&mut self) {
        let victim = self.tail;
        if victim == NIL {
            return;
        }
        let victim_prev = self.nodes[victim as usize].prev;
        let victim_key = self.nodes[victim as usize].key;

        if victim_prev != NIL {
            self.nodes[victim_prev as usize].next = NIL;
        }
        self.tail = victim_prev;
        if self.head == victim {
            self.head = NIL;
        }

        self.map.remove(&victim_key);
    }
}

/// A verify-once decoded-node cache with O(1) LRU eviction.
///
/// Uses an intrusive doubly-linked list (Vec-backed, index-based)
/// for O(1) access/eviction without external dependencies or unsafe.
pub struct NodeCache {
    state: RwLock<LruState>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl NodeCache {
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            state: RwLock::new(LruState::with_capacity(max_entries)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }

    /// Look up a node by its structural hash.
    /// Returns `None` if the node is not cached.
    pub fn get(&self, id: &NodeId) -> Option<Arc<NodeData>> {
        let mut state = self.state.write();
        let Some(&idx) = state.map.get(id) else {
            drop(state);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let data = state.nodes[idx as usize].data.clone();
        state.move_to_head(idx);
        drop(state);

        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(data)
    }

    /// Insert a node into the cache.
    /// If the cache is full, evicts the least recently accessed entry.
    ///
    /// # Panics
    /// Panics if the cache exceeds `u32::MAX` entries (impractical).
    pub fn insert(&self, id: NodeId, data: Arc<NodeData>) {
        let mut state = self.state.write();

        if let Some(&idx) = state.map.get(&id) {
            state.nodes[idx as usize].data = data;
            state.move_to_head(idx);
            return;
        }

        if state.map.len() >= self.max_entries {
            state.evict_lru();
        }

        let idx = LinkIdx::try_from(state.nodes.len()).expect("cache exceeds u32::MAX entries");
        let new_node = LruNode {
            data,
            key: id,
            prev: NIL,
            next: state.head,
        };
        state.nodes.push(new_node);
        if state.head != NIL {
            let head = state.head as usize;
            state.nodes[head].prev = idx;
        }
        state.head = idx;
        if state.tail == NIL {
            state.tail = idx;
        }
        state.map.insert(id, idx);
    }

    /// Remove a node from the cache.
    pub fn remove(&self, id: &NodeId) -> Option<Arc<NodeData>> {
        let mut state = self.state.write();
        let idx = state.map.remove(id)?;
        let data = state.nodes[idx as usize].data.clone();

        let prev = state.nodes[idx as usize].prev;
        let next = state.nodes[idx as usize].next;
        if prev != NIL {
            state.nodes[prev as usize].next = next;
        }
        if next != NIL {
            state.nodes[next as usize].prev = prev;
        }
        if state.head == idx {
            state.head = next;
        }
        if state.tail == idx {
            state.tail = prev;
        }

        Some(data)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.state.read().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.read().map.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Hit rate as a fraction [0.0, 1.0].
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        let total = h.saturating_add(m);
        h.checked_mul(10_000)
            .and_then(|v| v.checked_div(total))
            .map_or(0.0, |scaled| {
                let clamped = u32::try_from(scaled).unwrap_or(u32::MAX);
                f64::from(clamped) / 10_000.0
            })
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut state = self.state.write();
        state.map.clear();
        state.head = NIL;
        state.tail = NIL;
    }
}

/// A pinned set of nodes that are never evicted.
///
/// Represents the top two levels of a room's HAMT trie:
/// - Level 0: 1 node (the root)
/// - Level 1: up to 32 nodes
///
/// Total: ~33 nodes, ~17KB per room.
pub struct PinnedNodes {
    nodes: RwLock<HashMap<NodeId, Arc<NodeData>>>,
}

impl PinnedNodes {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::with_capacity(64)),
        }
    }

    /// Pin a node. Returns true if it was newly pinned.
    pub fn pin(&self, id: NodeId, data: Arc<NodeData>) -> bool {
        let mut map = self.nodes.write();
        let was_absent = !map.contains_key(&id);
        if was_absent {
            map.insert(id, data);
        }
        was_absent
    }

    /// Check if a node is pinned.
    pub fn is_pinned(&self, id: &NodeId) -> bool {
        self.nodes.read().contains_key(id)
    }

    /// Get a pinned node.
    pub fn get(&self, id: &NodeId) -> Option<Arc<NodeData>> {
        self.nodes.read().get(id).cloned()
    }

    /// Number of pinned nodes.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Clear all pinned nodes.
    pub fn clear(&self) {
        self.nodes.write().clear();
    }
}

impl Default for PinnedNodes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(s: &str) -> Arc<NodeData> {
        Arc::new(NodeData {
            bytes: bytes::Bytes::copy_from_slice(s.as_bytes()),
        })
    }

    #[test]
    fn test_insert_and_get() {
        let cache = NodeCache::new(100);
        let id = [0x42u8; 16];
        let data = test_data("hello");

        cache.insert(id, data.clone());
        let got = cache.get(&id).unwrap();
        assert_eq!(got.bytes, data.bytes);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = NodeCache::new(3);
        let ids: Vec<NodeId> = (0..4u8).map(|i| [i; 16]).collect();

        for (i, id) in ids.iter().enumerate() {
            cache.insert(*id, test_data(&format!("node {i}")));
        }

        assert!(cache.get(&ids[0]).is_none());
        assert!(cache.get(&ids[1]).is_some());
        assert!(cache.get(&ids[2]).is_some());
        assert!(cache.get(&ids[3]).is_some());
    }

    #[test]
    fn test_lru_access_refreshes() {
        let cache = NodeCache::new(3);
        let a = [1u8; 16];
        let b = [2u8; 16];
        let c = [3u8; 16];
        let d = [4u8; 16];

        cache.insert(a, test_data("a"));
        cache.insert(b, test_data("b"));
        cache.insert(c, test_data("c"));

        // Access a to make it recently used
        cache.get(&a);

        // Insert d — should evict b (least recently used), not a
        cache.insert(d, test_data("d"));
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&c).is_some());
        assert!(cache.get(&d).is_some());
    }

    #[test]
    fn test_hit_rate() {
        let cache = NodeCache::new(100);
        let id = [0x01u8; 16];

        cache.insert(id, test_data("data"));
        cache.get(&id); // hit
        cache.get(&id); // hit
        cache.get(&[0x02u8; 16]); // miss

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_pinned_nodes() {
        let pinned = PinnedNodes::new();
        let id = [0x01u8; 16];
        let data = test_data("pinned");

        assert!(pinned.pin(id, data.clone()));
        assert!(!pinned.pin(id, test_data("other")));
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned.get(&id).unwrap().bytes, data.bytes);
    }

    #[test]
    fn test_remove() {
        let cache = NodeCache::new(10);
        let id = [0x01u8; 16];
        cache.insert(id, test_data("data"));
        assert!(cache.remove(&id).is_some());
        assert!(cache.get(&id).is_none());
        assert!(cache.is_empty());
    }
}
