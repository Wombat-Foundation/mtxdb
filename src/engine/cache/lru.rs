//! A small, concurrent, generic least-recently-used cache.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

type LinkIdx = u32;
const NIL: LinkIdx = u32::MAX;

struct Entry<K, V> {
    key: K,
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
        self.map.remove(&self.entries[index as usize].key);
        if prev != NIL {
            self.entries[prev as usize].next = NIL;
        }
        self.tail = prev;
        if self.head == index {
            self.head = NIL;
        }
        self.entries[index as usize].value = None;
        self.entries[index as usize].next = self.free;
        self.free = index;
    }
}

/// A thread-safe, bounded LRU cache with O(1) lookup, insertion, and eviction.
///
/// A successful [`Self::get`] refreshes the entry's recency. Values are cloned
/// on lookup, which permits callers to store cheap handles such as `Arc`s as
/// well as small owned values (for example a room's 2 KiB lattice).
pub struct LruCache<K, V> {
    state: RwLock<State<K, V>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
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
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Looks up `key`, refreshing its recency when present.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut state = self.state.write();
        let Some(&index) = state.map.get(key) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let value = state.entries[index as usize].value.as_ref()?.clone();
        state.move_to_head(index);
        self.hits.fetch_add(1, Ordering::Relaxed);
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
        if state.map.len() >= self.capacity {
            state.evict_tail();
        }
        let index = if state.free == NIL {
            let index =
                LinkIdx::try_from(state.entries.len()).expect("LRU cache exceeds u32::MAX entries");
            let head = state.head;
            state.entries.push(Entry {
                key,
                value: Some(value),
                prev: NIL,
                next: head,
            });
            index
        } else {
            let index = state.free;
            state.free = state.entries[index as usize].next;
            let head = state.head;
            state.entries[index as usize] = Entry {
                key,
                value: Some(value),
                prev: NIL,
                next: head,
            };
            index
        };
        if state.head != NIL {
            let head = state.head as usize;
            state.entries[head].prev = index;
        }
        state.head = index;
        if state.tail == NIL {
            state.tail = index;
        }
        let key = state.entries[index as usize].key.clone();
        state.map.insert(key, index);
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
        self.hits.load(Ordering::Relaxed)
    }

    /// Number of unsuccessful lookups.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Successful lookups divided by all lookups.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits();
        let misses = self.misses();
        let total = hits.saturating_add(misses);
        hits.checked_mul(10_000)
            .and_then(|scaled| scaled.checked_div(total))
            .map_or(0.0, |scaled| {
                let scaled = u32::try_from(scaled).unwrap_or(u32::MAX);
                f64::from(scaled) / 10_000.0
            })
    }
}
