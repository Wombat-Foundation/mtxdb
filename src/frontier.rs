use std::collections::HashMap;

use crate::storage::{NodeData, NodeId, StorageEngine};

/// Represents a batch of independent node hashes to fetch at one BFS level.
///
/// This is the data structure that enables batched frontier submission:
/// resolve all hashes against one collection generation, then issue one
/// backend-native read operation for the whole level.
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
/// Uses the storage engine's native batch operation so all nodes observe one
/// collection generation. Results retain input order.
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

    // A single backend batch call pins one collection generation for engines
    // whose indexes are copy-on-write. Per-node worker calls can otherwise
    // observe different generations if deletion or repacking swaps the
    // collection while this frontier is being resolved.
    let results = engine.get_many(collection_id, &batch.hashes)?;
    Ok(batch.hashes.iter().copied().zip(results).collect())
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
#[path = "test_frontier.rs"]
mod tests;
