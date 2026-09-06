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

/// Batch-fetch all nodes in a frontier batch.
///
/// Currently performs sequential reads via `get_many`. The `FrontierBatch`
/// data structure is designed for future concurrent I/O (pre-resolved
/// offsets enable `io_uring` or thread-pool dispatch), but this
/// implementation issues reads in sorted-offset order for sequential
/// disk access rather than true concurrency.
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
pub fn fetch_frontier_batch<S: StorageEngine>(
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

/// Concurrently fetch all nodes in a frontier batch.
///
/// **Note:** This is an alias for [`fetch_frontier_batch`] and currently
/// performs sequential I/O. A true concurrent implementation (`io_uring`,
/// thread pool) that issues all reads in parallel is planned but not
/// yet implemented.
///
/// # Errors
/// Returns `StorageError::Io` on I/O failure from the storage engine.
pub fn fetch_frontier_concurrent<S: StorageEngine>(
    engine: &S,
    room_id: &[u8; 16],
    batch: &FrontierBatch,
) -> Result<Vec<(NodeId, Option<NodeData>)>, crate::storage::StorageError> {
    fetch_frontier_batch(engine, room_id, batch)
}

/// A BFS layer of the HAMT trie traversal.
///
/// Represents one level of the trie: a set of child hashes at the same
/// depth, all to be fetched in parallel. Uses a `HashSet` for O(1)
/// dedup on insert.
#[derive(Debug)]
pub struct BfsLayer {
    /// The hashes at this level.
    hashes: Vec<NodeId>,
    /// For each hash, which target keys are waiting on it.
    dependents: Vec<Vec<NodeId>>,
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

    /// The hashes at this layer.
    #[must_use]
    pub fn hashes(&self) -> &[NodeId] {
        &self.hashes
    }

    /// For each hash, which target keys are waiting on it.
    #[must_use]
    pub fn dependents(&self) -> &[Vec<NodeId>] {
        &self.dependents
    }

    /// Number of unique hashes in this layer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether this layer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

impl Default for BfsLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
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

        assert_eq!(layer.len(), 2);
        assert_eq!(layer.dependents()[0].len(), 2); // h1 has 2 dependents
        assert_eq!(layer.dependents()[1].len(), 1); // h2 has 1 dependent
    }

    #[test]
    fn test_bfs_layer_hashes_and_is_empty() {
        let layer = BfsLayer::new();
        assert!(layer.is_empty());
        let empty: &[NodeId] = &[];
        assert_eq!(layer.hashes(), empty);

        let mut layer = BfsLayer::new();
        layer.push([1u8; 16], [0xFF; 16]);
        layer.push([2u8; 16], [0xFF; 16]);
        assert!(!layer.is_empty());
        assert_eq!(layer.hashes().len(), 2);
        assert_eq!(layer.hashes()[0], [1u8; 16]);
        assert_eq!(layer.hashes()[1], [2u8; 16]);
    }

    #[test]
    fn test_bfs_layer_default() {
        let layer = BfsLayer::default();
        assert!(layer.is_empty());
        assert_eq!(layer.len(), 0);
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
