#![allow(clippy::tests_outside_test_module)]

use mtxdb::cache::*;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Barrier};
use std::thread;

use mtxdb::storage::{NodeData, NodeId};

fn test_data(s: &str) -> Arc<NodeData> {
    Arc::new(NodeData::new(bytes::Bytes::copy_from_slice(s.as_bytes())))
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
    assert_eq!(cache.hit_rate().total_cmp(&0.6666), Ordering::Equal);
}

#[test]
fn test_pinned_nodes() {
    let pinned = PinnedNodes::new();
    let id = [0x01u8; 16];
    let data = test_data("pinned");

    assert!(pinned.pin(id, data.clone()));
    assert!(!pinned.pin(id, test_data("other")));
    assert_eq!(pinned.len(), 1);
    assert!(pinned.is_pinned(&id));
    assert!(!pinned.is_pinned(&[0x02; 16]));
    assert_eq!(pinned.get(&id).unwrap().bytes, data.bytes);
}

#[test]
fn test_pinned_default_is_empty_clear() {
    let pinned = PinnedNodes::default();
    assert!(pinned.is_empty());
    let id = [0x03u8; 16];
    pinned.pin(id, test_data("x"));
    assert!(!pinned.is_empty());
    pinned.clear();
    assert!(pinned.is_empty());
}

#[test]
fn test_cache_len() {
    let cache = NodeCache::new(100);
    assert_eq!(cache.len(), 0);
    cache.insert([0x01; 16], test_data("a"));
    assert_eq!(cache.len(), 1);
    cache.insert([0x02; 16], test_data("b"));
    assert_eq!(cache.len(), 2);
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

#[test]
fn test_generic_lru_eviction_and_refresh() {
    let cache = LruCache::new(2);
    cache.insert("a", 1);
    cache.insert("b", 2);
    assert_eq!(cache.get(&"a"), Some(1));
    cache.insert("c", 3);
    assert_eq!(cache.get(&"b"), None);
    assert_eq!(cache.get(&"a"), Some(1));
    assert_eq!(cache.get(&"c"), Some(3));
}

#[test]
fn test_generic_lru_zero_capacity_and_stats() {
    let cache = LruCache::<u8, u8>::new(0);
    cache.insert(1, 1);
    assert_eq!(cache.get(&1), None);
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hit_rate().total_cmp(&0.0), Ordering::Equal);
}

struct MaybePanicCloneKey {
    value: u8,
    panic_on_clone: bool,
}

impl Clone for MaybePanicCloneKey {
    fn clone(&self) -> Self {
        assert!(!self.panic_on_clone, "test clone panic");
        Self {
            value: self.value,
            panic_on_clone: self.panic_on_clone,
        }
    }
}

impl PartialEq for MaybePanicCloneKey {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for MaybePanicCloneKey {}

impl Hash for MaybePanicCloneKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

#[test]
fn test_generic_lru_panic_during_key_clone_is_transactional() {
    let cache = LruCache::new(1);
    let resident = MaybePanicCloneKey {
        value: 1,
        panic_on_clone: false,
    };
    cache.insert(resident, 10);

    let panicking = MaybePanicCloneKey {
        value: 2,
        panic_on_clone: true,
    };
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| cache.insert(panicking, 20)));
    assert!(result.is_err());
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.get(&MaybePanicCloneKey {
            value: 1,
            panic_on_clone: false
        }),
        Some(10)
    );
}

#[test]
fn test_generic_lru_hit_rate_preserves_small_positive_values() {
    let cache = LruCache::new(1);
    cache.insert(1, 1);
    assert_eq!(cache.get(&1), Some(1));
    for key in 2..=10_001 {
        assert_eq!(cache.get(&key), None);
    }
    assert!(cache.hit_rate() > 0.0);
}

#[test]
fn test_generic_lru_remove_and_reinsert() {
    let cache = LruCache::new(2);
    cache.insert("a", 1);
    cache.insert("b", 2);

    assert_eq!(cache.remove(&"a"), Some(1));
    assert_eq!(cache.get(&"a"), None);
    assert_eq!(cache.len(), 1);

    cache.insert("a", 3);
    assert_eq!(cache.get(&"a"), Some(3));
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_generic_lru_clear() {
    let cache = LruCache::new(2);
    cache.insert("a", 1);
    cache.insert("b", 2);
    cache.clear();

    assert!(cache.is_empty());
    assert_eq!(cache.get(&"a"), None);
    assert_eq!(cache.get(&"b"), None);
}

#[test]
fn test_generic_lru_replacement_refreshes_recency() {
    let cache = LruCache::new(2);
    cache.insert("a", 1);
    cache.insert("b", 2);
    cache.insert("a", 10);
    cache.insert("c", 3);

    assert_eq!(cache.get(&"a"), Some(10));
    assert_eq!(cache.get(&"b"), None);
    assert_eq!(cache.get(&"c"), Some(3));
}

#[test]
fn test_generic_lru_concurrent_access() {
    const THREADS: usize = 8;
    const ITEMS_PER_THREAD: usize = 64;
    let cache = Arc::new(LruCache::new(THREADS * ITEMS_PER_THREAD));
    let ready = Arc::new(Barrier::new(THREADS));

    thread::scope(|scope| {
        for thread_id in 0..THREADS {
            let cache = Arc::clone(&cache);
            let ready = Arc::clone(&ready);
            scope.spawn(move || {
                ready.wait();
                for offset in 0..ITEMS_PER_THREAD {
                    let key = thread_id * ITEMS_PER_THREAD + offset;
                    cache.insert(key, key * 2);
                    assert_eq!(cache.get(&key), Some(key * 2));
                    assert_eq!(cache.remove(&key), Some(key * 2));
                }
            });
        }
    });

    assert!(cache.is_empty());
}
