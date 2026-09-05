use std::collections::HashMap;

use crate::storage::{NodeData, NodeId, StorageEngine};

/// Represents a batch of independent node hashes to fetch at one BFS level.
///
/// This is the data structure that enables concurrent frontier submission:
/// resolve all hashes to pack offsets in RAM, then issue all reads
/// concurrently via `io_uring` or a thread pool.
#[derive(Debug, Clone)]
pub struct FrontierBatch {
    /// The hashes to fetch.
    pub hashes: Vec<NodeId>,
    /// Optional: pre-resolved (`pack_id`, offset) pairs from the index.
    pub resolved: Vec<Option<(u8, u64)>>,
}

impl FrontierBatch {
    #[must_use]
    pub fn new(hashes: Vec<NodeId>) -> Self {
        let resolved = vec![None; hashes.len()];
        Self { hashes, resolved }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

/// Concurrently fetch all nodes in a frontier batch.
///
/// This is the key optimization for HDD: issue all reads simultaneously
/// so the kernel's I/O scheduler (mq-deadline) can sort them into
/// a head sweep. On `NVMe` this is purely parallel; on HDD it reduces
/// N full seeks to one sweep across N sorted positions.
///
/// # Arguments
/// * `engine` - The storage engine to read from.
/// * `room_id` - The room whose index and packfiles to search.
/// * `batch` - The frontier batch with hashes to fetch.
///
/// # Returns
/// A vector of `(NodeId, Option<NodeData>)` in the same order as the input.
///
/// # Errors
/// Returns `StorageError::Io` on I/O failure from the storage engine.
pub fn fetch_frontier_concurrent<S: StorageEngine>(
    engine: &S,
    room_id: &[u8; 16],
    batch: &FrontierBatch,
) -> Result<Vec<(NodeId, Option<NodeData>)>, crate::storage::StorageError> {
    let results = engine.get_many(room_id, &batch.hashes)?;

    Ok(batch
        .hashes
        .iter()
        .zip(results)
        .map(|(&id, data)| (id, data))
        .collect())
}

/// A BFS layer of the HAMT trie traversal.
///
/// Represents one level of the trie: a set of child hashes at the same
/// depth, all to be fetched in parallel. Uses a `HashSet` for O(1)
/// dedup on insert.
#[derive(Debug)]
pub struct BfsLayer {
    /// The hashes at this level.
    pub hashes: Vec<NodeId>,
    /// For each hash, which target keys are waiting on it.
    pub dependents: Vec<Vec<NodeId>>,
    /// O(1) dedup: maps hash → index into `hashes`.
    seen: HashMap<NodeId, usize>,
}

impl BfsLayer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            hashes: Vec::new(),
            dependents: Vec::new(),
            seen: HashMap::new(),
        }
    }

    pub fn push(&mut self, hash: NodeId, dependent: NodeId) {
        if let Some(&pos) = self.seen.get(&hash) {
            self.dependents[pos].push(dependent);
        } else {
            let pos = self.hashes.len();
            self.hashes.push(hash);
            self.dependents.push(vec![dependent]);
            self.seen.insert(hash, pos);
        }
    }
}

impl Default for BfsLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontier_batch() {
        let hashes = vec![[1u8; 16], [2u8; 16], [3u8; 16]];
        let batch = FrontierBatch::new(hashes.clone());
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
        assert_eq!(batch.hashes, hashes);
    }

    #[test]
    fn test_bfs_layer_dedup() {
        let mut layer = BfsLayer::new();
        let h1 = [1u8; 16];
        let h2 = [2u8; 16];
        let target = [0xFFu8; 16];

        layer.push(h1, target);
        layer.push(h1, [0xEE; 16]); // same hash, different dependent
        layer.push(h2, target);

        assert_eq!(layer.hashes.len(), 2);
        assert_eq!(layer.dependents[0].len(), 2); // h1 has 2 dependents
        assert_eq!(layer.dependents[1].len(), 1); // h2 has 1 dependent
    }

    #[test]
    fn test_fetch_frontier_concurrent() {
        use crate::storage::InMemoryStorage;

        let engine = InMemoryStorage::new();
        let room = [0x01; 16];
        let id1 = [1u8; 16];
        let id2 = [2u8; 16];

        engine
            .put(
                &room,
                &id1,
                &NodeData::new(bytes::Bytes::from_static(b"node1")),
            )
            .unwrap();
        engine
            .put(
                &room,
                &id2,
                &NodeData::new(bytes::Bytes::from_static(b"node2")),
            )
            .unwrap();

        let batch = FrontierBatch::new(vec![id1, id2, [3u8; 16]]); // id3 not in store
        let results = fetch_frontier_concurrent(&engine, &room, &batch).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some());
        assert!(results[1].1.is_some());
        assert!(results[2].1.is_none());
    }
}
