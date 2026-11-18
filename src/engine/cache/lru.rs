//! A small, concurrent, generic least-recently-used cache.

use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

type LinkIdx = u32;
const NIL: LinkIdx = u32::MAX;

fn hash_key<T>(key: &T) -> u64
where
    T: Hash + ?Sized,
{
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

struct Entry<K, V> {
    key: Option<K>,
    /// Hash of `key`, retained so eviction never has to call caller-supplied
    /// hashing or equality while holding the cache lock.
    key_hash: u64,
    value: Option<Arc<V>>,
    prev: LinkIdx,
    next: LinkIdx,
}

impl<K, V> Entry<K, V> {
    fn vacant() -> Self {
        Self {
            key: None,
            key_hash: 0,
            value: None,
            prev: NIL,
            next: NIL,
        }
    }
}

struct State<K, V> {
    /// Maps a key hash to the slot(s) containing a key with that hash.
    ///
    /// The key itself is owned only by its slot, so keys need not be `Clone`.
    index: HashMap<u64, Vec<LinkIdx>>,
    entries: Vec<Entry<K, V>>,
    len: usize,
    head: LinkIdx,
    tail: LinkIdx,
    free: LinkIdx,
}

/// The key and value removed by an eviction.
///
/// They are returned rather than dropped while the state is being mutated so
/// that a panicking destructor cannot leave the index, list, and slot disagreeing.
struct Evicted<K, V> {
    key: K,
    value: Arc<V>,
}

impl<K, V> State<K, V>
where
    K: Eq + Hash,
{
    /// Creates an empty state with no reusable slots.
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            entries: Vec::new(),
            len: 0,
            head: NIL,
            tail: NIL,
            free: NIL,
        }
    }

    /// Finds the slot holding an entry equal to `key`.
    fn find<Q>(&self, key_hash: u64, key: &Q) -> Option<LinkIdx>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.index.get(&key_hash)?.iter().copied().find(|&index| {
            self.entries[index as usize]
                .key
                .as_ref()
                .is_some_and(|resident| resident.borrow() == key)
        })
    }

    fn index_insert(&mut self, key_hash: u64, index: LinkIdx) {
        self.index.entry(key_hash).or_default().push(index);
    }

    fn index_remove(&mut self, key_hash: u64, index: LinkIdx) {
        let mut empty = false;
        if let Some(bucket) = self.index.get_mut(&key_hash) {
            if let Some(position) = bucket.iter().position(|&slot| slot == index) {
                bucket.swap_remove(position);
            }
            empty = bucket.is_empty();
        }
        if empty {
            self.index.remove(&key_hash);
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
    ///
    /// Does nothing when empty. The removed key and value are returned so the
    /// caller can drop them only after the cache is structurally consistent.
    /// The stored hash is used, so eviction does not invoke caller-supplied
    /// `Hash` or `Eq` implementations.
    fn evict_tail(&mut self) -> Option<Evicted<K, V>> {
        if self.tail == NIL {
            return None;
        }
        let index = self.tail;
        let key_hash = self.entries[index as usize].key_hash;
        self.index_remove(key_hash, index);

        let prev = self.entries[index as usize].prev;
        if prev != NIL {
            self.entries[prev as usize].next = NIL;
        }
        self.tail = prev;
        if self.head == index {
            self.head = NIL;
        }

        let free = self.free;
        let entry = &mut self.entries[index as usize];
        let key = entry.key.take().expect("occupied LRU slot must have a key");
        let value = entry
            .value
            .take()
            .expect("occupied LRU slot must have a value");
        entry.key_hash = 0;
        entry.prev = NIL;
        entry.next = free;
        self.free = index;
        self.len = self.len.saturating_sub(1);

        Some(Evicted { key, value })
    }
}

impl<K, V> Evicted<K, V> {
    /// Drops the removed key and value after the cache state is consistent.
    fn drop_removed(self) {
        let Self { key, value } = self;
        drop(key);
        drop(value);
    }
}

/// A thread-safe, bounded LRU cache with O(1) lookup and recency updates.
///
/// Values are stored behind an [`Arc`] so a lookup can clone the shared handle
/// under the lock and run the caller-supplied [`Clone`] implementation after the
/// lock is released. This keeps exact LRU ordering atomic with the lookup
/// without invoking caller code while a cache lock is held.
///
/// Keys are owned by their slots and indexed by a stored hash, so keys do not
/// need to be `Clone`. Key hashing for lookup happens before the cache lock is
/// taken; equality still runs under the lock (as it must for the index lookup),
/// so `Eq` implementations must not call back into the same cache or they will
/// deadlock.
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
    K: Eq + Hash,
{
    /// Creates an empty cache with room for at most `capacity` entries.
    /// A capacity of zero disables storage; lookups record misses.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            state: RwLock::new(State::new()),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Returns a clone of the value for `key`, or `None` if absent.
    ///
    /// Records a hit when returning a value and a miss when the key is absent.
    /// The lookup and recency update happen under one exclusive lock, so a
    /// concurrent insertion cannot evict the accessed entry between them. The
    /// value's `Clone` runs after that lock is released, which allows a
    /// `Clone` implementation to call back into the cache without deadlocking.
    ///
    /// Misses are detected under a shared lock and only the hit path takes the
    /// exclusive lock.
    ///
    /// # Panics
    ///
    /// Panics from key hashing or equality, or from cloning the value, propagate.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        V: Clone,
    {
        let key_hash = hash_key(key);
        {
            let state = self.state.read();
            if state.find(key_hash, key).is_none() {
                drop(state);
                self.record_miss();
                return None;
            }
        }

        let mut state = self.state.write();
        let Some(index) = state.find(key_hash, key) else {
            drop(state);
            self.record_miss();
            return None;
        };
        let value = state.entries[index as usize]
            .value
            .as_ref()
            .expect("resident key must have a value")
            .clone();
        state.move_to_head(index);
        self.record_hit();
        drop(state);

        Some((*value).clone())
    }

    /// Increments the hit count, saturating at `u64::MAX`.
    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the miss count, saturating at `u64::MAX`.
    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
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
    /// Panics from hashing or comparing keys, or from dropping keys or values,
    /// propagate. Evicted keys and values are dropped only after the
    /// replacement entry is fully installed, so a destructor panic leaves the
    /// cache consistent.
    pub fn insert(&self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }

        let key_hash = hash_key(&key);
        let mut state = self.state.write();
        if let Some(index) = state.find(key_hash, &key) {
            let old_value = state.entries[index as usize].value.replace(Arc::new(value));
            state.move_to_head(index);
            drop(state);
            drop(old_value);
            return;
        }

        // Evict before publishing the replacement, so a panic while removing
        // the victim leaves the cache without a dangling new index entry.
        let evicted = if state.len >= self.capacity {
            state.evict_tail()
        } else {
            None
        };

        let need_new_slot = state.free == NIL;
        let index = if need_new_slot {
            assert!(
                state.entries.len() < NIL as usize,
                "LRU cache cannot allocate the sentinel entry index"
            );
            let index =
                LinkIdx::try_from(state.entries.len()).expect("LRU cache exceeds u32::MAX entries");
            state.entries.push(Entry::vacant());
            index
        } else {
            state.free
        };

        // Publish only after eviction is structurally complete. A hash panic
        // here leaves the victim removed and this slot unused, which is a
        // consistent cache.
        state.index_insert(key_hash, index);

        if !need_new_slot {
            state.free = state.entries[index as usize].next;
        }
        let head = state.head;
        state.entries[index as usize] = Entry {
            key: Some(key),
            key_hash,
            value: Some(Arc::new(value)),
            prev: NIL,
            next: head,
        };
        if head != NIL {
            state.entries[head as usize].prev = index;
        }
        state.head = index;
        if state.tail == NIL {
            state.tail = index;
        }
        state.len = state.len.saturating_add(1);
        drop(state);

        if let Some(evicted) = evicted {
            evicted.drop_removed();
        }
    }

    /// Removes `key` and returns its value if present.
    /// Hit/miss counters are unchanged.
    ///
    /// # Panics
    ///
    /// Panics from key equality or destruction propagate. Removed keys are
    /// dropped only after the cache is structurally consistent.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        V: Clone,
    {
        let key_hash = hash_key(key);
        let mut state = self.state.write();
        let index = state.find(key_hash, key)?;
        state.index_remove(key_hash, index);

        let (prev, next) = {
            let entry = &state.entries[index as usize];
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
        let free = state.free;
        let entry = &mut state.entries[index as usize];
        let entry_key = entry.key.take();
        let value = entry.value.take();
        entry.key_hash = 0;
        entry.prev = NIL;
        entry.next = free;
        state.free = index;
        state.len = state.len.saturating_sub(1);
        drop(state);

        drop(entry_key);
        value.map(|value| match Arc::try_unwrap(value) {
            Ok(value) => value,
            // A concurrent lookup still holds a clone of the handle. The cache
            // no longer owns the value, so hand the caller its own copy.
            Err(shared) => (*shared).clone(),
        })
    }

    /// Removes all entries. Hit/miss counters are retained.
    ///
    /// # Panics
    ///
    /// Panics from dropping keys or values propagate after the cache has been
    /// emptied and the lock released.
    pub fn clear(&self) {
        let old = {
            let mut state = self.state.write();
            std::mem::replace(&mut *state, State::new())
        };
        drop(old);
    }

    /// Number of resident entries.
    pub fn len(&self) -> usize {
        self.state.read().len
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
        self.hits.load(Ordering::Relaxed)
    }

    /// Number of unsuccessful lookups.
    /// The count saturates at `u64::MAX` and is retained by [`Self::clear`].
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Successful lookups divided by all lookups.
    ///
    /// Returns a fraction in `[0.0, 1.0]`, or `0.0` before any lookups.
    ///
    /// The hit and miss counters are independent atomics, so concurrent lookups
    /// may observe a snapshot that never existed at a single instant. Individual
    /// updates are still exact and saturating.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let misses = self.misses() as f64;
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::LruCache;

    struct BlockingClone {
        value: u8,
        block: Arc<AtomicBool>,
        started: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl Clone for BlockingClone {
        fn clone(&self) -> Self {
            if self.block.swap(false, Ordering::SeqCst) {
                self.started.wait();
                self.release.wait();
            }
            Self {
                value: self.value,
                block: Arc::clone(&self.block),
                started: Arc::clone(&self.started),
                release: Arc::clone(&self.release),
            }
        }
    }

    #[test]
    fn hit_refresh_is_atomic_with_concurrent_insert() {
        let cache = Arc::new(LruCache::new(2));
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let blocked = Arc::new(AtomicBool::new(true));
        let make_value = |value, block| BlockingClone {
            value,
            block,
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        };

        cache.insert(1u8, make_value(1, Arc::clone(&blocked)));
        cache.insert(2u8, make_value(2, Arc::new(AtomicBool::new(false))));

        let lookup_cache = Arc::clone(&cache);
        let lookup = thread::spawn(move || lookup_cache.get(&1).map(|value| value.value));

        // `get` has already refreshed key 1's recency and is now blocked in the
        // caller-supplied `Clone`, outside the cache lock.
        started.wait();
        cache.insert(3u8, make_value(3, Arc::new(AtomicBool::new(false))));
        release.wait();

        assert_eq!(lookup.join().unwrap(), Some(1));
        assert_eq!(cache.get(&1).map(|value| value.value), Some(1));
        assert!(cache.get(&2).is_none());
        assert_eq!(cache.get(&3).map(|value| value.value), Some(3));
    }

    #[test]
    fn full_cache_with_free_slot_reuses_slots() {
        let cache = LruCache::new(2);

        for key in 0..100u32 {
            cache.insert(key, key);
            assert!(cache.len() <= cache.capacity());

            if key >= 1 {
                assert_eq!(cache.get(&(key - 1)), Some(key - 1));
                assert_eq!(cache.get(&key), Some(key));
            }
            if key >= 2 {
                assert_eq!(cache.get(&(key - 2)), None);
            }

            // Repeated insertion must not grow the slot vector without bound,
            // even if a slot is temporarily left allocated during a resize.
            assert!(cache.state.read().entries.len() <= cache.capacity() + 1);
        }

        assert!(cache.state.read().entries.len() <= cache.capacity() + 1);
    }
}
