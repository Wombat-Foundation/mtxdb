//! A small, concurrent, generic least-recently-used cache.

use std::collections::HashMap;
use std::hash::Hash;

use parking_lot::RwLock;

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
    hits: u64,
    misses: u64,
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
            hits: 0,
            misses: 0,
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
        }
    }

    /// Looks up `key`, refreshing its recency when present.
    ///
    /// Exact global LRU ordering requires an exclusive lock for hits, so
    /// concurrent hits serialize briefly. Callers with highly parallel,
    /// read-heavy workloads should use multiple caches (for example, one per
    /// shard) when approximate global recency is acceptable.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut state = self.state.write();
        let Some(&index) = state.map.get(key) else {
            state.misses = state.misses.saturating_add(1);
            return None;
        };
        let value = state.entries[index as usize].value.as_ref()?.clone();
        state.move_to_head(index);
        state.hits = state.hits.saturating_add(1);
        Some(value)
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
        let index = if was_full || free_index == NIL {
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
            state.free = next_free;
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
        let hits = state.hits;
        let misses = state.misses;
        *state = State::new();
        state.hits = hits;
        state.misses = misses;
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
        self.state.read().hits
    }

    /// Number of unsuccessful lookups.
    pub fn misses(&self) -> u64 {
        self.state.read().misses
    }

    /// Successful lookups divided by all lookups.
    ///
    /// The counters are read under the same lock used by lookups, so both
    /// values come from one coherent snapshot.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let state = self.state.read();
        let hits = state.hits as f64;
        let misses = state.misses as f64;
        let total = hits + misses;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }
}
