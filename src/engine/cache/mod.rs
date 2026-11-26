use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::storage::{NodeData, NodeId, NodeRef};

mod lru;

pub use lru::LruCache;

/// Maximum number of entries in the node cache.
const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// Default process-wide budget for decoded node bytes.
pub const DEFAULT_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Approximate resident bytes one decoded node occupies in a cache.
///
/// Counts the payload, the resolved-child vector's capacity, and the
/// bookkeeping an entry adds (the LRU slot and its map key). Fixed-size
/// `Bytes` handles whose payload lives elsewhere are not separate
/// allocations, so this is a close upper estimate rather than an exact
/// allocator measurement.
fn resident_bytes(data: &NodeData) -> usize {
    std::mem::size_of::<NodeData>()
        .saturating_add(data.bytes.len())
        .saturating_add(
            data.children
                .capacity()
                .saturating_mul(std::mem::size_of::<NodeRef>()),
        )
        .saturating_add(std::mem::size_of::<LruNode>())
        .saturating_add(std::mem::size_of::<NodeId>())
}

/// A byte budget shared by every decoded-node cache in one process.
///
/// Each collection still has its own LRU cache, but admission draws from this
/// shared counter, so a database with many per-room collections cannot hold a
/// full per-collection cache in each one. A cache that reaches the budget
/// refuses new entries instead of evicting another collection's; reads then
/// fall back to the pack.
pub struct CacheBudget {
    cap: usize,
    used: AtomicUsize,
}

impl CacheBudget {
    /// Create a budget that admits at most `cap` resident bytes.
    #[must_use]
    pub const fn new(cap: usize) -> Self {
        Self {
            cap,
            used: AtomicUsize::new(0),
        }
    }

    /// The configured byte cap.
    #[must_use]
    pub const fn cap(&self) -> usize {
        self.cap
    }

    /// Bytes currently admitted across every cache sharing this budget.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// Reserve `bytes` if the budget has room. Never exceeds `cap`.
    fn try_reserve(&self, bytes: usize) -> bool {
        if bytes == 0 {
            return true;
        }
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return false;
            };
            if next > self.cap {
                return false;
            }
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Return `bytes` to the budget. Saturates rather than wrapping if a
    /// caller ever over-releases.
    fn release(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(bytes);
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Index into the intrusive linked list Vec.
type LinkIdx = u32;

/// Sentinel index meaning "no node".
const NIL: LinkIdx = u32::MAX;

/// A node in the intrusive doubly-linked list for LRU ordering.
struct LruNode {
    data: Option<Arc<NodeData>>,
    key: NodeId,
    /// Approximate resident bytes charged to the budget for this entry.
    cost: usize,
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
    /// Head of the free list for recycled slots. NIL if empty.
    free_list: LinkIdx,
    /// Sum of every resident entry's `cost`.
    bytes: usize,
}

impl LruState {
    fn empty() -> Self {
        Self {
            map: HashMap::new(),
            nodes: Vec::new(),
            head: NIL,
            tail: NIL,
            free_list: NIL,
            bytes: 0,
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

    /// Unlink the entry at `idx`, recycle its slot, and return its data and
    /// charged cost. The caller returns the cost to the shared budget.
    fn take(&mut self, idx: LinkIdx) -> (Option<Arc<NodeData>>, usize) {
        let key = self.nodes[idx as usize].key;
        self.map.remove(&key);
        let data = self.nodes[idx as usize].data.take();
        let cost = self.nodes[idx as usize].cost;
        self.nodes[idx as usize].cost = 0;
        self.bytes = self.bytes.saturating_sub(cost);

        let prev = self.nodes[idx as usize].prev;
        let next = self.nodes[idx as usize].next;
        if prev == NIL {
            self.head = next;
        } else {
            self.nodes[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.nodes[next as usize].prev = prev;
        }
        self.nodes[idx as usize].prev = NIL;
        self.nodes[idx as usize].next = self.free_list;
        self.free_list = idx;
        (data, cost)
    }

    /// Evict the least recently used entry and return its charged cost.
    fn evict_lru(&mut self) -> Option<usize> {
        let victim = self.tail;
        if victim == NIL {
            return None;
        }
        let (_, cost) = self.take(victim);
        Some(cost)
    }
}

/// A verify-once decoded-node cache with O(1) LRU eviction.
///
/// Uses an intrusive doubly-linked list (Vec-backed, index-based)
/// for O(1) access/eviction without external dependencies or unsafe.
pub struct NodeCache {
    state: RwLock<LruState>,
    max_entries: usize,
    /// Shared byte budget, or `None` for an entry-bounded cache.
    budget: Option<Arc<CacheBudget>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl NodeCache {
    /// Create a cache holding at most `max_entries` decoded nodes.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self::with_budget(max_entries, None)
    }

    /// Create a cache that also draws admission from `budget`.
    ///
    /// `None` keeps the historical entry-only bound. `Some` makes every
    /// insertion pay its approximate resident bytes, and refuses the
    /// insertion when other caches have already consumed the budget.
    #[must_use]
    pub fn with_budget(max_entries: usize, budget: Option<Arc<CacheBudget>>) -> Self {
        Self {
            // Empty per-collection caches are the normal state immediately
            // after an mmap-backed open. Let their backing collections grow
            // amortized with admitted nodes: reserving the full cache budget
            // either here or on the first admission makes an untouched cache
            // expensive to open or a newly written collection expensive to
            // use for the first time.
            state: RwLock::new(LruState::empty()),
            max_entries,
            budget,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// The budget this cache draws from, if any.
    #[must_use]
    pub fn budget(&self) -> Option<&Arc<CacheBudget>> {
        self.budget.as_ref()
    }

    fn reserve(&self, bytes: usize) -> bool {
        self.budget
            .as_ref()
            .is_none_or(|budget| budget.try_reserve(bytes))
    }

    fn release(&self, bytes: usize) {
        if let Some(budget) = &self.budget {
            budget.release(bytes);
        }
    }

    /// Create a cache using the default capacity (100,000 entries).
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
        let data = state.nodes[idx as usize].data.as_ref()?.clone();
        state.move_to_head(idx);
        drop(state);

        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(data)
    }

    /// Insert a node into the cache.
    ///
    /// The least recently used entries are evicted from this cache to make
    /// room. If the node still cannot fit it is not cached, and any previous
    /// version under the same id is invalidated rather than left visible.
    ///
    /// # Panics
    /// Panics if the cache exceeds `u32::MAX` entries (impractical).
    pub fn insert(&self, id: NodeId, data: Arc<NodeData>) {
        if self.max_entries == 0 {
            return;
        }

        let cost = resident_bytes(&data);
        let mut state = self.state.write();

        // A replacement must not leave the previous version visible if the
        // new one cannot be admitted: drop it first, then treat the write as a
        // fresh admission. A miss is correct; stale data is not.
        if let Some(&idx) = state.map.get(&id) {
            let (_, previous) = state.take(idx);
            self.release(previous);
        }

        // Draw from the shared budget, sacrificing this cache's own LRU
        // entries when the budget is full. A node that cannot fit even after
        // this cache is empty is simply not cached; the caller keeps it.
        while !self.reserve(cost) {
            let Some(evicted) = state.evict_lru() else {
                return;
            };
            self.release(evicted);
        }

        // The byte budget is not the entry-count limit: still make one slot.
        if state.map.len() >= self.max_entries {
            if let Some(evicted) = state.evict_lru() {
                self.release(evicted);
            }
        }

        let idx = if state.free_list == NIL {
            let new_idx =
                LinkIdx::try_from(state.nodes.len()).expect("cache exceeds u32::MAX entries");
            state.nodes.push(LruNode {
                data: None,
                key: id,
                cost: 0,
                prev: NIL,
                next: NIL,
            });
            new_idx
        } else {
            let free_idx = state.free_list;
            state.free_list = state.nodes[free_idx as usize].next;
            free_idx
        };

        state.bytes = state.bytes.saturating_add(cost);
        state.nodes[idx as usize].data = Some(data);
        state.nodes[idx as usize].key = id;
        state.nodes[idx as usize].cost = cost;
        state.nodes[idx as usize].prev = NIL;
        state.nodes[idx as usize].next = state.head;
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
        let idx = *state.map.get(id)?;
        let (data, cost) = state.take(idx);
        self.release(cost);
        data
    }

    /// Check which hashes are already resident in the cache.
    ///
    /// Returns a `Vec<Option<Arc<NodeData>>>` of the same length as
    /// `hashes`, where `Some(data)` means the hash is cached with that
    /// data. Does NOT move entries in the LRU order (this is a read-only
    /// probe, not an access).
    ///
    /// Used by the swizzling layer to identify which child pointers can be
    /// rewritten from `Lazy(hash)` to `Resolved(Arc<NodeData>)` without
    /// touching disk, providing the actual cached data for the callback.
    pub fn resolve_hashes(&self, hashes: &[NodeId]) -> Vec<Option<Arc<NodeData>>> {
        let state = self.state.read();
        hashes
            .iter()
            .map(|h| {
                state
                    .map
                    .get(h)
                    .and_then(|&idx| state.nodes[idx as usize].data.clone())
            })
            .collect()
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.state.read().map.len()
    }

    /// Returns `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.state.read().map.is_empty()
    }

    /// Approximate resident bytes held by this cache's decoded nodes.
    pub fn bytes(&self) -> usize {
        self.state.read().bytes
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
    ///
    /// The relaxed atomic loads form an approximate snapshot while accesses
    /// are concurrent. Large counters are scaled down, so the result can
    /// remain approximate even after the counters stop changing.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed);
        let m = self.misses.load(Ordering::Relaxed);
        // Scale both counters down together until they fit in 32 bits so the
        // ratio needs no lossy casts. Below 2^32 lookups the counters are
        // converted exactly; above it, truncation can round a tiny hit count
        // down to zero (1 hit in 2^32 misses reports 0.0).
        let excess = u32::BITS.saturating_sub(h.max(m).leading_zeros());
        let narrow = |count: u64| f64::from(u32::try_from(count >> excess).unwrap_or(u32::MAX));
        let (h, m) = (narrow(h), narrow(m));
        if h + m == 0.0 {
            0.0
        } else {
            h / (h + m)
        }
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut state = self.state.write();
        let bytes = state.bytes;
        state.map.clear();
        state.nodes.clear();
        state.head = NIL;
        state.tail = NIL;
        state.free_list = NIL;
        state.bytes = 0;
        self.release(bytes);
    }
}

impl Drop for NodeCache {
    fn drop(&mut self) {
        // A deleted collection drops its cache without calling `clear`; return
        // whatever it still holds so the shared budget does not leak.
        let bytes = self.state.get_mut().bytes;
        self.release(bytes);
    }
}

/// A pinned set of nodes that are never evicted.
///
/// Represents the top two levels of a collection's HAMT trie:
/// - Level 0: 1 node (the root)
/// - Level 1: up to 32 nodes
///
/// Total: ~33 nodes, ~17KB per collection.
pub struct PinnedNodes {
    nodes: RwLock<HashMap<NodeId, Arc<NodeData>>>,
}

impl PinnedNodes {
    /// Create an empty pinned-node set.
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

    /// Returns `true` if no nodes are pinned.
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

    fn test_node(payload_size: usize) -> Arc<NodeData> {
        Arc::new(NodeData::new(bytes::Bytes::from(vec![7u8; payload_size])))
    }

    #[test]
    fn shared_budget_bounds_total_admission_across_caches() {
        let data = test_node(64);
        let cost = resident_bytes(&data);
        let budget = Arc::new(CacheBudget::new(cost.saturating_mul(2)));
        let first = NodeCache::with_budget(4, Some(Arc::clone(&budget)));
        let second = NodeCache::with_budget(4, Some(Arc::clone(&budget)));

        first.insert([1; 16], Arc::clone(&data));
        second.insert([2; 16], Arc::clone(&data));
        assert_eq!(budget.used(), cost.saturating_mul(2));

        // The budget is full, so the first cache gives up its own entry rather
        // than growing past the shared cap.
        first.insert([3; 16], Arc::clone(&data));
        assert_eq!(budget.used(), cost.saturating_mul(2));
        assert_eq!(first.len(), 1);
        assert!(first.get(&[3; 16]).is_some());
        assert!(second.get(&[2; 16]).is_some());
    }

    #[test]
    fn removal_and_clear_release_shared_budget() {
        let data = test_node(32);
        let cost = resident_bytes(&data);
        let budget = Arc::new(CacheBudget::new(cost.saturating_mul(3)));
        let cache = NodeCache::with_budget(8, Some(Arc::clone(&budget)));

        cache.insert([1; 16], Arc::clone(&data));
        cache.insert([2; 16], Arc::clone(&data));
        assert_eq!(budget.used(), cost.saturating_mul(2));

        assert!(cache.remove(&[1; 16]).is_some());
        assert_eq!(budget.used(), cost);

        cache.clear();
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_node_larger_than_the_budget_is_not_cached() {
        let data = test_node(4096);
        let cost = resident_bytes(&data);
        let budget = Arc::new(CacheBudget::new(cost.saturating_sub(1)));
        let cache = NodeCache::with_budget(8, Some(Arc::clone(&budget)));

        cache.insert([1; 16], data);
        assert_eq!(cache.len(), 0);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_rejected_larger_overwrite_invalidates_the_old_value() {
        let old = test_node(32);
        let larger = test_node(4096);
        let old_cost = resident_bytes(&old);
        let budget = Arc::new(CacheBudget::new(old_cost));
        let cache = NodeCache::with_budget(8, Some(Arc::clone(&budget)));
        let id = [1; 16];

        cache.insert(id, old);
        assert_eq!(cache.get(&id).map(|data| data.bytes.len()), Some(32));

        // The replacement cannot fit, but the older cached value must not
        // survive under an id that has already been overwritten.
        cache.insert(id, larger);
        assert!(cache.get(&id).is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_replacement_that_fits_stores_the_new_value() {
        let small = test_node(32);
        let larger = test_node(96);
        let budget = Arc::new(CacheBudget::new(resident_bytes(&larger).saturating_mul(2)));
        let cache = NodeCache::with_budget(8, Some(Arc::clone(&budget)));

        cache.insert([1; 16], small);
        cache.insert([1; 16], Arc::clone(&larger));
        assert_eq!(cache.get(&[1; 16]).map(|data| data.bytes.len()), Some(96));
        assert_eq!(cache.len(), 1);
        assert_eq!(budget.used(), resident_bytes(&larger));
    }

    #[test]
    fn entry_limit_eviction_releases_budget() {
        let data = test_node(16);
        let cost = resident_bytes(&data);
        let budget = Arc::new(CacheBudget::new(cost.saturating_mul(8)));
        let cache = NodeCache::with_budget(1, Some(Arc::clone(&budget)));

        cache.insert([1; 16], Arc::clone(&data));
        cache.insert([2; 16], Arc::clone(&data));
        assert_eq!(cache.len(), 1);
        assert_eq!(budget.used(), cost);
    }

    #[test]
    fn dropping_a_cache_returns_its_budget() {
        let data = test_node(24);
        let cost = resident_bytes(&data);
        let budget = Arc::new(CacheBudget::new(cost.saturating_mul(4)));
        {
            let cache = NodeCache::with_budget(8, Some(Arc::clone(&budget)));
            cache.insert([1; 16], Arc::clone(&data));
            cache.insert([2; 16], data);
            assert_eq!(budget.used(), cost.saturating_mul(2));
        }
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn hit_rate_scaling_boundary() {
        let cache = NodeCache::new(1);
        cache.hits.store(1, Ordering::Relaxed);
        cache
            .misses
            .store(u64::from(u32::MAX) - 1, Ordering::Relaxed);
        let exact = 1.0 / f64::from(u32::MAX);
        assert!((cache.hit_rate() - exact).abs() < exact * 1.0e-12);

        // Past 2^32 the counters are scaled; a lone hit is rounded away.
        cache.misses.store(1 << 32, Ordering::Relaxed);
        assert!(cache.hit_rate().abs() < f64::EPSILON);

        // Large balanced counters keep their ratio.
        cache.hits.store(1 << 40, Ordering::Relaxed);
        cache.misses.store(3 << 40, Ordering::Relaxed);
        assert!((cache.hit_rate() - 0.25).abs() < 1.0e-9);
    }
}
