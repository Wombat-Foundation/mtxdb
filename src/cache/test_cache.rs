use super::*;

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
