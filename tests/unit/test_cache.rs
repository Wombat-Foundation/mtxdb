#![allow(clippy::tests_outside_test_module)]

use mtxdb::cache::*;
use std::cmp::Ordering;
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
fn test_lru_reuses_slots_without_mixing_keys_and_values() {
    let cache = NodeCache::new(2);

    for i in 0..20u8 {
        let id = [i; 16];
        cache.insert(id, test_data(&format!("node {i}")));

        if i >= 1 {
            let previous_id = [i - 1; 16];
            let previous = cache.get(&previous_id).unwrap();
            assert_eq!(
                previous.bytes,
                bytes::Bytes::copy_from_slice(format!("node {}", i - 1).as_bytes())
            );
        }
        if i >= 2 {
            assert!(cache.get(&[i - 2; 16]).is_none());
        }
        let current = cache.get(&id).unwrap();
        assert_eq!(
            current.bytes,
            bytes::Bytes::copy_from_slice(format!("node {i}").as_bytes())
        );
        assert_eq!(cache.len(), 2.min(usize::from(i) + 1));
    }
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
    assert_eq!(cache.hit_rate().total_cmp(&(2.0 / 3.0)), Ordering::Equal);
}

#[test]
fn test_hit_rate_preserves_small_positive_values() {
    let cache = NodeCache::new(1);
    let id = [0x01u8; 16];
    cache.insert(id, test_data("data"));
    cache.get(&id);
    for i in 0..10_000u32 {
        let mut miss = [0u8; 16];
        miss[..4].copy_from_slice(&(i + 2).to_le_bytes());
        cache.get(&miss);
    }

    let expected = 1.0 / 10_001.0;
    assert!((cache.hit_rate() - expected).abs() < expected * 1.0e-12);
}

#[test]
fn test_hit_rate_is_zero_without_hits() {
    let cache = NodeCache::new(1);
    cache.get(&[0x02u8; 16]);

    assert_eq!(cache.hit_rate().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn test_insert_replaces_cached_node_data() {
    let cache = NodeCache::new(2);
    let id = [0x01u8; 16];
    cache.insert(id, test_data("old"));
    cache.insert(id, test_data("new"));
    assert_eq!(
        cache.get(&id).unwrap().bytes,
        bytes::Bytes::copy_from_slice(b"new")
    );
    assert_eq!(cache.len(), 1);
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

struct Bomb {
    armed: bool,
}

impl Clone for Bomb {
    fn clone(&self) -> Self {
        Self { armed: false }
    }
}

impl Drop for Bomb {
    fn drop(&mut self) {
        assert!(!self.armed || thread::panicking(), "test drop panic");
    }
}

#[test]
fn test_generic_lru_panicking_destructor_leaves_cache_consistent() {
    let cache = LruCache::new(1);
    cache.insert(1u8, Bomb { armed: true });
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        cache.insert(2u8, Bomb { armed: false });
    }));
    assert!(result.is_err());
    assert_eq!(cache.len(), 1);
    assert!(cache.get(&1).is_none());
    assert!(cache.get(&2).is_some());
    cache.insert(3u8, Bomb { armed: false });
    assert!(cache.get(&2).is_none());
    assert!(cache.get(&3).is_some());
}

#[test]
fn test_generic_lru_reentrant_clone_does_not_deadlock() {
    struct Reenter(Arc<LruCache<u8, Reenter>>);
    impl Clone for Reenter {
        fn clone(&self) -> Self {
            self.0.insert(99, Self(Arc::clone(&self.0)));
            Self(Arc::clone(&self.0))
        }
    }
    let cache = Arc::new(LruCache::new(4));
    cache.insert(1, Reenter(Arc::clone(&cache)));
    assert!(cache.get(&1).is_some());
    assert!(!cache.is_empty());
    cache.clear(); // break the Arc cycle
}

#[test]
fn test_generic_lru_non_clone_keys() {
    #[derive(PartialEq, Eq, Hash)]
    struct NoClone(u8);
    let cache = LruCache::new(1);
    cache.insert(NoClone(1), 1);
    cache.insert(NoClone(2), 2);
    assert_eq!(cache.get(&NoClone(1)), None);
    assert_eq!(cache.get(&NoClone(2)), Some(2));
}

#[test]
fn test_generic_lru_hit_rate_preserves_small_positive_values() {
    let cache = LruCache::new(1);
    cache.insert(1, 1);
    assert_eq!(cache.get(&1), Some(1));
    for key in 2..=10_001 {
        assert_eq!(cache.get(&key), None);
    }
    assert!((cache.hit_rate() - 1.0 / 10_001.0).abs() < 1e-12);
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
