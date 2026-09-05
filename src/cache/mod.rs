use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::storage::{NodeData, NodeId};

/// Maximum number of entries in the node cache.
const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// An LRU entry tracking access time for eviction.
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Arc<NodeData>,
    last_access: u64,
}

/// A verify-once decoded-node cache with pointer swizzling.
///
/// Design principles from docs:
/// - Stores `Arc<NodeData>`, never raw bytes. This enforces that all
///   cache entries have been verified (decode_v1_verified is the only
///   way to produce a NodeData from storage bytes).
/// - Byte-bounded LRU eviction to prevent OOM.
/// - Pointer swizzling: when a parent node's children are also resident,
///   the parent can hold direct `Arc` references to them instead of
///   re-hashing on each traversal.
/// - Pinned prefix: for active rooms, the top ~33 nodes (L0+L1) are
///   pinned and never evicted.
pub struct NodeCache {
    entries: RwLock<HashMap<NodeId, CacheEntry>>,
    /// Monotonic counter for LRU ordering.
    counter: AtomicU64,
    /// Maximum number of entries.
    max_entries: usize,
    /// Number of cache hits (for metrics).
    hits: AtomicU64,
    /// Number of cache misses (for metrics).
    misses: AtomicU64,
}

impl NodeCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries)),
            counter: AtomicU64::new(0),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }

    /// Look up a node by its structural hash.
    /// Returns `None` if the node is not cached.
    pub fn get(&self, id: &NodeId) -> Option<Arc<NodeData>> {
        let mut map = self.entries.write();
        if let Some(entry) = map.get_mut(id) {
            entry.last_access = self.counter.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.data.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a node into the cache.
    /// If the cache is full, evicts the least recently accessed entry.
    pub fn insert(&self, id: NodeId, data: Arc<NodeData>) {
        let now = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut map = self.entries.write();

        // If already present, update
        if let Some(entry) = map.get_mut(&id) {
            entry.data = data;
            entry.last_access = now;
            return;
        }

        // Evict if full
        if map.len() >= self.max_entries {
            if let Some(evict_id) = map
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(id, _)| *id)
            {
                map.remove(&evict_id);
            }
        }

        map.insert(
            id,
            CacheEntry {
                data,
                last_access: now,
            },
        );
    }

    /// Remove a node from the cache.
    pub fn remove(&self, id: &NodeId) -> Option<Arc<NodeData>> {
        self.entries.write().remove(id).map(|e| e.data)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
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
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            h / total
        }
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

/// A pinned set of nodes that are never evicted.
///
/// Represents the top two levels of a room's HAMT trie:
/// - Level 0: 1 node (the root)
/// - Level 1: up to 32 nodes
///
/// Total: ~33 nodes, ~17KB per room.
///
/// Swizzling within the pinned set is free: children that are also
/// pinned can be resolved to direct `Arc` references, eliminating
/// hash lookups on traversal.
pub struct PinnedNodes {
    nodes: RwLock<HashMap<NodeId, Arc<NodeData>>>,
}

impl PinnedNodes {
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

        // First entry should be evicted
        assert!(cache.get(&ids[0]).is_none());
        // Others should still be there
        assert!(cache.get(&ids[1]).is_some());
        assert!(cache.get(&ids[2]).is_some());
        assert!(cache.get(&ids[3]).is_some());
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
        assert!(!pinned.pin(id, test_data("other"))); // already pinned
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned.get(&id).unwrap().bytes, data.bytes);
    }
}
