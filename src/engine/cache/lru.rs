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
    /// Creates an empty state with no reusable slots.
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            entries: Vec::new(),
            head: NIL,
            tail: NIL,
            free: NIL,
        }
    }

    /// Marks the occupied slot at `index` as most recently used.
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

    /// Removes the least recently used entry and makes its slot reusable.
    /// Does nothing when empty. Panics from dropping keys or values propagate
    /// after the entry has been unlinked and its slot made reusable.
    fn evict_tail(&mut self) {
        if self.tail == NIL {
            return;
        }
        let index = self.tail;
        let prev = self.entries[index as usize].prev;
        if prev != NIL {
            self.entries[prev as usize].next = NIL;
        }
        self.tail = prev;
        if self.head == index {
            self.head = NIL;
        }

        // Remove the map entry while keeping its key alive until the cache is
        // structurally consistent; a panic in `K::drop` must not strand a
        // linked slot without a map entry.
        let victim_key = self
            .map
            .extract_if(|_, &mut entry_index| entry_index == index)
            .next()
            .map(|(key, _)| key);

        let entry = &mut self.entries[index as usize];
        let entry_key = entry.key.take();
        let entry_value = entry.value.take();
        entry.next = self.free;
        self.free = index;

        drop(victim_key);
        drop(entry_key);
        drop(entry_value);
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
    /// A capacity of zero disables storage; lookups record misses.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            state: RwLock::new(State::new()),
            capacity,
            stats: Mutex::new(Stats::default()),
        }
    }

    /// Returns a clone of the value for `key`, or `None` if absent.
    ///
    /// Records a hit when returning a value and a miss when the key is absent.
    /// Refreshes recency if the key still maps to the same slot after cloning.
    /// Concurrent removal, eviction, or replacement can occur before the
    /// refresh; the cloned value is still returned and counted as a hit.
    ///
    /// A hit clones its value under a shared lock and only takes the exclusive
    /// lock for the recency refresh. Misses never take the cache write lock.
    ///
    /// # Panics
    ///
    /// Panics from key hashing or equality, or from cloning the value, propagate.
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

    /// Increments the hit count, saturating at `u64::MAX`.
    fn record_hit(&self) {
        let mut stats = self.stats.lock();
        stats.hits = stats.hits.saturating_add(1);
    }

    /// Increments the miss count, saturating at `u64::MAX`.
    fn record_miss(&self) {
        let mut stats = self.stats.lock();
        stats.misses = stats.misses.saturating_add(1);
    }

    /// Inserts or replaces an entry, making it the most recently used entry.
    ///
    /// Inserting a new key into a full cache evicts the least recently used
    /// entry. A capacity of zero drops the supplied key and value without
    /// storing them. Hit/miss counters are unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a new slot is needed after `u32::MAX` slots have been allocated.
    /// Panics from cloning, hashing, or comparing keys, or from dropping keys
    /// or values, propagate. A key clone panic occurs before eviction; a hash
    /// or equality panic while publishing the new map entry occurs after
    /// eviction but before the new slot is linked. An evicted key or value is
    /// dropped only after the cache has been made structurally consistent.
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

        let index = if free_index == NIL {
            assert!(
                state.entries.len() < NIL as usize,
                "LRU cache cannot allocate the sentinel entry index"
            );
            LinkIdx::try_from(state.entries.len()).expect("LRU cache exceeds u32::MAX entries")
        } else {
            free_index
        };

        let victim = if was_full {
            let victim = state.tail;
            state.evict_tail();
            Some(victim)
        } else {
            None
        };

        // Publish the new map entry only after eviction is complete. If
        // dropping the evicted key or value panics, the map must not already
        // point at a slot that has not been linked into the LRU list.
        state.map.insert(map_key, index);

        if let Some(victim) = victim {
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
    /// Hit/miss counters are unchanged.
    ///
    /// # Panics
    ///
    /// Panics from key hashing, equality, or destruction propagate. A panic
    /// while dropping a key can leave the cache partially updated.
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
    ///
    /// # Panics
    ///
    /// Panics from dropping keys or values propagate.
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
    /// The count saturates at `u64::MAX` and is retained by [`Self::clear`].
    pub fn hits(&self) -> u64 {
        self.stats.lock().hits
    }

    /// Number of unsuccessful lookups.
    /// The count saturates at `u64::MAX` and is retained by [`Self::clear`].
    pub fn misses(&self) -> u64 {
        self.stats.lock().misses
    }

    /// Successful lookups divided by all lookups.
    ///
    /// Returns a fraction in `[0.0, 1.0]`, or `0.0` before any lookups.
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
            if key >= 1 {
                let state = cache.state.read();
                for expected in [key - 1, key] {
                    let index = state.map.get(&expected).expect("expected key to remain");
                    assert_eq!(
                        state.entries[*index as usize].value.as_ref(),
                        Some(&expected)
                    );
                }
            }
            if key >= 2 {
                assert!(!cache.state.read().map.contains_key(&(key - 2)));
            }
            assert!(cache.state.read().entries.len() <= cache.capacity() + 1);
        }

        assert_eq!(cache.state.read().entries.len(), cache.capacity() + 1);
    }
}
