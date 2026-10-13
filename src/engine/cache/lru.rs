//! A small, concurrent, generic least-recently-used cache.

use std::collections::HashMap;
use std::hash::Hash;

use parking_lot::{Mutex, RwLock};

type LinkIdx = u32;
const NIL: LinkIdx = u32::MAX;

struct Entry<K, V> {
    key: Option<K>,
    value: Option<V>,
    prev: LinkIdx,
    next: LinkIdx,
}

struct State<K, V> {
    map: HashMap<K, LinkIdx>,
    entries: Vec<Entry<K, V>>,
    head: LinkIdx,
    tail: LinkIdx,
    free: LinkIdx,
}

impl<K, V> State<K, V>
where
    K: Eq + Hash,
{
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            entries: Vec::new(),
            head: NIL,
            tail: NIL,
            free: NIL,
        }
    }

    fn move_to_head(&mut self, index: LinkIdx) {
        if self.head == index {
            return;
        }
        let entry = &self.entries[index as usize];
        let (prev, next) = (entry.prev, entry.next);
        if prev != NIL {
            self.entries[prev as usize].next = next;
        }
        if next != NIL {
            self.entries[next as usize].prev = prev;
        }
        if self.tail == index {
            self.tail = prev;
        }
        self.entries[index as usize].prev = NIL;
        self.entries[index as usize].next = self.head;
        if self.head != NIL {
            self.entries[self.head as usize].prev = index;
        }
        self.head = index;
        if self.tail == NIL {
            self.tail = index;
        }
    }

    fn evict_tail(&mut self) {
        if self.tail == NIL {
            return;
        }
        let index = self.tail;
        let prev = self.entries[index as usize].prev;
        self.map.retain(|_, &mut entry_index| entry_index != index);
        if prev != NIL {
            self.entries[prev as usize].next = NIL;
        }
        self.tail = prev;
        if self.head == index {
            self.head = NIL;
        }
        self.entries[index as usize].value = None;
        self.entries[index as usize].key = None;
        self.entries[index as usize].next = self.free;
        self.free = index;
    }
}

/// A thread-safe, bounded LRU cache with O(1) lookup and recency updates.
///
/// Eviction scans the key index when recycling a slot. That small tradeoff
/// keeps the commit path independent of caller-supplied key hashing/equality,
/// which is important for panic safety.
///
/// A successful [`Self::get`] refreshes the entry's recency. Values are cloned
/// on lookup, which permits callers to store cheap handles such as `Arc`s as
/// well as small owned values (for example a room's 2 KiB lattice).
pub struct LruCache<K, V> {
    state: RwLock<State<K, V>>,
    capacity: usize,
    stats: Mutex<Stats>,
}

#[derive(Default)]
struct Stats {
    hits: u64,
    misses: u64,
}

impl<K, V> LruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty cache with room for at most `capacity` entries.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            state: RwLock::new(State::new()),
            capacity,
            stats: Mutex::new(Stats::default()),
        }
    }

    /// Looks up `key`, refreshing its recency when present.
    ///
    /// A hit clones its value under a shared lock and only takes the exclusive
    /// lock for the recency refresh. Misses never take the cache write lock.
    /// Exact global LRU ordering still makes the refresh phase contend briefly;
    /// callers with highly parallel workloads can use multiple caches when
    /// approximate global recency is acceptable.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let state = self.state.read();
        let Some(&index) = state.map.get(key) else {
            drop(state);
            self.record_miss();
            return None;
        };
        let value = state.entries[index as usize].value.as_ref()?.clone();
        drop(state);

        let mut state = self.state.write();
        if state.map.get(key).copied() == Some(index) {
            state.move_to_head(index);
        }
        self.record_hit();
        Some(value)
    }

    fn record_hit(&self) {
        let mut stats = self.stats.lock();
        stats.hits = stats.hits.saturating_add(1);
    }

    fn record_miss(&self) {
        let mut stats = self.stats.lock();
        stats.misses = stats.misses.saturating_add(1);
    }

    /// Inserts or replaces an entry, making it the most recently used entry.
    ///
    /// # Panics
    ///
    /// Panics if the cache grows beyond `u32::MAX` entries, which is
    /// impractical for an in-process cache.
    pub fn insert(&self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }
        let mut state = self.state.write();
        if let Some(&index) = state.map.get(&key) {
            state.entries[index as usize].value = Some(value);
            state.move_to_head(index);
            return;
        }

        // Clone before touching the cache. Besides supplying the separate map
        // and entry keys, this is the preflight phase: a caller-controlled
        // Clone panic cannot evict an existing entry or leave a new entry
        // linked without its map index.
        let map_key = key.clone();
        let was_full = state.map.len() >= self.capacity;
        let free_index = state.free;
        let next_free = if free_index == NIL {
            NIL
        } else {
            state.entries[free_index as usize].next
        };

        // Install the map entry before linking the slot. Hash/Eq panics leave
        // the LRU list untouched; the standard HashMap operation remains
        // panic-safe and no eviction has happened yet.
        let index = if free_index == NIL {
            assert!(
                state.entries.len() < NIL as usize,
                "LRU cache cannot allocate the sentinel entry index"
            );
            LinkIdx::try_from(state.entries.len()).expect("LRU cache exceeds u32::MAX entries")
        } else {
            free_index
        };
        state.map.insert(map_key, index);

        if was_full {
            // The map temporarily contains the new entry, so remove the old
            // tail before publishing the new list links.
            let victim = state.tail;
            state.evict_tail();
            if index == free_index {
                // The candidate slot was on the free list before eviction.
                // Keep the newly freed victim slot while removing the
                // candidate from that list before reusing it.
                let victim_next = state.entries[victim as usize].next;
                state.entries[victim as usize].next = next_free;
                state.free = victim;
                debug_assert_eq!(
                    victim_next, free_index,
                    "evicted slot must precede the candidate free slot"
                );
            }
        }
        let reuse_slot = index == free_index && free_index != NIL;
        let head = state.head;
        let entry = Entry {
            key: Some(key),
            value: Some(value),
            prev: NIL,
            next: head,
        };
        if reuse_slot {
            // When the cache was already full, `evict_tail` prepended the
            // victim to the free list. The candidate was its next link, so
            // leave the victim as the free-list head and unlink only the
            // candidate being reused. Otherwise consume the ordinary free
            // list head.
            if !was_full {
                state.free = next_free;
            }
            state.entries[index as usize] = entry;
        } else {
            state.entries.push(entry);
        }
        if state.head != NIL {
            let head = state.head as usize;
            state.entries[head].prev = index;
        }
        state.head = index;
        if state.tail == NIL {
            state.tail = index;
        }
    }

    /// Removes `key` and returns its value if present.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut state = self.state.write();
        let index = state.map.remove(key)?;
        let (prev, next) = {
            let entry = &mut state.entries[index as usize];
            (entry.prev, entry.next)
        };
        if prev != NIL {
            state.entries[prev as usize].next = next;
        }
        if next != NIL {
            state.entries[next as usize].prev = prev;
        }
        if state.head == index {
            state.head = next;
        }
        if state.tail == index {
            state.tail = prev;
        }
        let value = state.entries[index as usize].value.take();
        state.entries[index as usize].key = None;
        state.entries[index as usize].next = state.free;
        state.free = index;
        value
    }

    /// Removes all entries. Hit/miss counters are retained.
    pub fn clear(&self) {
        let mut state = self.state.write();
        *state = State::new();
    }

    /// Number of resident entries.
    pub fn len(&self) -> usize {
        self.state.read().map.len()
    }

    /// Whether no entries are resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configured maximum number of entries.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of successful lookups.
    pub fn hits(&self) -> u64 {
        self.stats.lock().hits
    }

    /// Number of unsuccessful lookups.
    pub fn misses(&self) -> u64 {
        self.stats.lock().misses
    }

    /// Successful lookups divided by all lookups.
    ///
    /// The counters are read under the same lock used by lookup updates, so
    /// both values come from one coherent snapshot.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock();
        let hits = stats.hits as f64;
        let misses = stats.misses as f64;
        let total = hits + misses;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LruCache;

    #[test]
    fn full_cache_with_free_slot_reuses_slots() {
        let cache = LruCache::new(2);

        for key in 0..100u32 {
            cache.insert(key, key);
            assert!(cache.len() <= cache.capacity());
            assert!(cache.state.read().entries.len() <= cache.capacity() + 1);
        }

        assert_eq!(cache.state.read().entries.len(), cache.capacity() + 1);
    }
}
