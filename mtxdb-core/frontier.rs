use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::storage::{NodeData, NodeId, StorageEngine, StorageError};

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

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
    /// Create a batch for the given hashes with all offsets unresolved.
    #[must_use]
    pub fn new(hashes: Vec<NodeId>) -> Self {
        let resolved = vec![None; hashes.len()];
        Self { hashes, resolved }
    }

    /// Number of hashes in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Returns `true` if this batch has no hashes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

/// Batch-fetch all nodes in a frontier batch.
///
/// Uses a bounded worker set so independent frontier nodes can be fetched
/// concurrently. Results retain input order. Backends whose native batched
/// path can improve physical ordering may use [`fetch_frontier_sequential`]
/// instead.
///
/// # Arguments
/// * `engine` - The storage engine to read from.
/// * `collection_id` - The collection whose index and packfiles to search.
/// * `batch` - The frontier batch with hashes to fetch.
///
/// # Returns
/// A vector of `(NodeId, Option<NodeData>)` in the same order as the input.
///
/// # Errors
/// Returns `StorageError::Io` on I/O failure from the storage engine.
pub fn fetch_frontier_batch<S: StorageEngine>(
    engine: &S,
    collection_id: &[u8; 16],
    batch: &FrontierBatch,
) -> Result<Vec<(NodeId, Option<NodeData>)>, crate::storage::StorageError> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }

    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(batch.len());
    let pending = Mutex::new(
        batch
            .hashes
            .iter()
            .copied()
            .enumerate()
            .collect::<VecDeque<_>>(),
    );
    let results = Mutex::new(vec![None; batch.len()]);
    let failure = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                if lock_unpoisoned(&failure).is_some() {
                    return;
                }
                let Some((position, id)) = lock_unpoisoned(&pending).pop_front() else {
                    return;
                };
                match engine.get(collection_id, &id) {
                    Ok(data) => {
                        lock_unpoisoned(&results)[position] = Some(data);
                    }
                    Err(error) => {
                        let mut recorded = lock_unpoisoned(&failure);
                        if recorded.is_none() {
                            *recorded = Some(error);
                        }
                        return;
                    }
                }
            });
        }
    });

    if let Some(error) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(error);
    }
    let results = results.into_inner().unwrap_or_else(PoisonError::into_inner);

    let mut output = Vec::with_capacity(batch.len());
    for (&id, data) in batch.hashes.iter().zip(results) {
        let data = data.ok_or_else(|| {
            StorageError::Internal("frontier worker exited without producing a result".to_owned())
        })?;
        output.push((id, data));
    }
    Ok(output)
}

/// Fetch all nodes in a frontier batch.
///
/// Performs a backend-native sequential/batched read. The sorted physical
/// offset ordering is usually preferable for packfile/HDD reads.
///
/// # Errors
/// Returns `StorageError::Io` on I/O failure from the storage engine.
pub fn fetch_frontier_sequential<S: StorageEngine>(
    engine: &S,
    collection_id: &[u8; 16],
    batch: &FrontierBatch,
) -> Result<Vec<(NodeId, Option<NodeData>)>, crate::storage::StorageError> {
    let results = engine.get_many(collection_id, &batch.hashes)?;
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
    hashes: Vec<NodeId>,
    /// For each hash, which target keys are waiting on it.
    dependents: Vec<Vec<NodeId>>,
    /// O(1) dedup: maps hash → index into `hashes`.
    seen: HashMap<NodeId, usize>,
}

impl BfsLayer {
    /// Create an empty BFS layer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hashes: Vec::new(),
            dependents: Vec::new(),
            seen: HashMap::new(),
        }
    }

    /// Record that `dependent` is waiting on `hash`, deduping `hash` entries.
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
    fn test_fetch_frontier_sequential() {
        use crate::storage::InMemoryStorage;

        let engine = InMemoryStorage::new();
        let collection = [0x01; 16];
        let id1 = [1u8; 16];
        let id2 = [2u8; 16];

        engine
            .put(
                &collection,
                &id1,
                &NodeData::new(bytes::Bytes::from_static(b"node1")),
            )
            .unwrap();
        engine
            .put(
                &collection,
                &id2,
                &NodeData::new(bytes::Bytes::from_static(b"node2")),
            )
            .unwrap();

        let batch = FrontierBatch::new(vec![id1, id2, [3u8; 16]]); // id3 not in store
        let results = fetch_frontier_sequential(&engine, &collection, &batch).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some());
        assert!(results[1].1.is_some());
        assert!(results[2].1.is_none());
    }
}
