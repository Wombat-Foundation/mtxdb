use std::collections::HashMap;

/// Dense local ID for a room. No single room exceeds 2^31 events.
pub type LocalId = u32;

/// A compressed sparse row (CSR) arena for the active event DAG.
///
/// This is the on-disk-compatible form of the in-memory DAG frontier.
/// Events and their edges arrive sequentially, building contiguously.
///
/// Design from docs:
/// - Two heap allocations per room: `nodes` Vec and `edges` Vec.
/// - Edges of a node are contiguous in the `edges` array (cache line).
/// - O(1) drop: `nodes.clear()` + `edges.clear()`.
/// - No self-referential structs (compiles in safe Rust).
///
/// Usage:
/// - Resident edges use `GraphEdge::resident(slot)`.
/// - Disk edges use `GraphEdge::disk(short_id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphEdge(u32);

impl GraphEdge {
    const RESIDENT_BIT: u32 = 1 << 31;
    const PAYLOAD_MASK: u32 = !Self::RESIDENT_BIT;

    #[inline]
    pub fn is_resident(self) -> bool {
        (self.0 & Self::RESIDENT_BIT) != 0
    }

    #[inline]
    pub fn arena_index(self) -> usize {
        debug_assert!(self.is_resident());
        (self.0 & Self::PAYLOAD_MASK) as usize
    }

    #[inline]
    pub fn local_id(self) -> LocalId {
        debug_assert!(!self.is_resident());
        self.0 & Self::PAYLOAD_MASK
    }

    #[inline]
    pub fn resident(index: usize) -> Self {
        debug_assert!(index as u32 <= Self::PAYLOAD_MASK);
        Self(Self::RESIDENT_BIT | (index as u32 & Self::PAYLOAD_MASK))
    }

    #[inline]
    pub fn disk(id: LocalId) -> Self {
        debug_assert!(id <= Self::PAYLOAD_MASK);
        Self(id & Self::PAYLOAD_MASK)
    }
}

/// A node in the event DAG.
#[derive(Debug, Clone)]
pub struct EventNode {
    /// Global short event ID.
    pub short_id: u64,
    /// Dense local ID within this room.
    pub local_id: LocalId,
    /// Range into the edges array for prev_events: (start, len).
    pub prev: (u32, u32),
    /// Range into the edges array for auth_events: (start, len).
    pub auth: (u32, u32),
}

/// CSR arena for an active room's event DAG.
///
/// Contains all events loaded for an active room, with edges stored
/// contiguously in a single Vec. Graph algorithms like topological
/// sort are linear scans over this array.
#[derive(Debug)]
pub struct ActiveRoomFrontier {
    /// All event nodes in the room.
    pub nodes: Vec<EventNode>,
    /// Contiguous edge storage. Edges for a node are at `edges[start..start+len]`.
    pub edges: Vec<GraphEdge>,
    /// Maps global short_id → index into nodes Vec.
    pub resident: HashMap<u64, u32>,
    /// Maps global short_id → dense local_id for this room.
    pub id_remap: HashMap<u64, LocalId>,
    /// Next local_id to assign.
    next_local_id: LocalId,
}

impl ActiveRoomFrontier {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            resident: HashMap::new(),
            id_remap: HashMap::new(),
            next_local_id: 0,
        }
    }

    /// Register a global short_id and assign it a dense local_id.
    pub fn register_id(&mut self, short_id: u64) -> LocalId {
        if let Some(&local) = self.id_remap.get(&short_id) {
            return local;
        }
        let local = self.next_local_id;
        self.next_local_id += 1;
        self.id_remap.insert(short_id, local);
        local
    }

    /// Insert an event with its raw prev and auth edges.
    ///
    /// For each parent in `raw_prevs` and `raw_auths`:
    /// - If the parent is already in the arena, creates a `resident` edge.
    /// - Otherwise, creates a `disk` edge with the parent's local_id.
    pub fn insert_event(&mut self, short_id: u64, raw_prevs: &[u64], raw_auths: &[u64]) -> usize {
        let local_id = self.register_id(short_id);
        let node_idx = self.nodes.len() as u32;

        let prev_start = self.edges.len() as u32;
        for &parent_id in raw_prevs {
            let edge = self
                .id_remap
                .get(&parent_id)
                .map(|&local| {
                    if let Some(&slot) = self.resident.get(&parent_id) {
                        GraphEdge::resident(slot as usize)
                    } else {
                        GraphEdge::disk(local)
                    }
                })
                .unwrap_or_else(|| GraphEdge::disk(self.register_id(parent_id)));
            self.edges.push(edge);
        }
        let prev_len = self.edges.len() as u32 - prev_start;

        let auth_start = self.edges.len() as u32;
        for &parent_id in raw_auths {
            let edge = self
                .id_remap
                .get(&parent_id)
                .map(|&local| {
                    if let Some(&slot) = self.resident.get(&parent_id) {
                        GraphEdge::resident(slot as usize)
                    } else {
                        GraphEdge::disk(local)
                    }
                })
                .unwrap_or_else(|| GraphEdge::disk(self.register_id(parent_id)));
            self.edges.push(edge);
        }
        let auth_len = self.edges.len() as u32 - auth_start;

        self.nodes.push(EventNode {
            short_id,
            local_id,
            prev: (prev_start, prev_len),
            auth: (auth_start, auth_len),
        });
        self.resident.insert(short_id, node_idx);

        node_idx as usize
    }

    /// Get the prev_edges for a node by its arena index.
    pub fn prev_edges(&self, node_idx: usize) -> &[GraphEdge] {
        let node = &self.nodes[node_idx];
        &self.edges[node.prev.0 as usize..(node.prev.0 + node.prev.1) as usize]
    }

    /// Get the auth_edges for a node by its arena index.
    pub fn auth_edges(&self, node_idx: usize) -> &[GraphEdge] {
        let node = &self.nodes[node_idx];
        &self.edges[node.auth.0 as usize..(node.auth.0 + node.auth.1) as usize]
    }

    /// Drop all data for this room. O(1).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.resident.clear();
        self.id_remap.clear();
        self.next_local_id = 0;
    }

    /// Number of events in this room's frontier.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for ActiveRoomFrontier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_edge_packing() {
        let resident = GraphEdge::resident(42);
        assert!(resident.is_resident());
        assert_eq!(resident.arena_index(), 42);

        let disk = GraphEdge::disk(12345);
        assert!(!disk.is_resident());
        assert_eq!(disk.local_id(), 12345);
    }

    #[test]
    fn test_insert_event_basic() {
        let mut frontier = ActiveRoomFrontier::new();

        // Event A (no parents)
        let idx_a = frontier.insert_event(100, &[], &[]);
        assert_eq!(idx_a, 0);
        assert_eq!(frontier.len(), 1);

        // Event B (parent is A)
        let idx_b = frontier.insert_event(101, &[100], &[100]);
        assert_eq!(idx_b, 1);
        assert_eq!(frontier.len(), 2);

        // B's prev should be a resident edge pointing to A
        let prev = frontier.prev_edges(idx_b);
        assert_eq!(prev.len(), 1);
        assert!(prev[0].is_resident());
        assert_eq!(prev[0].arena_index(), 0); // A is at index 0
    }

    #[test]
    fn test_insert_event_with_disk_parent() {
        let mut frontier = ActiveRoomFrontier::new();

        // Event A with a parent that isn't in the arena
        let idx_a = frontier.insert_event(100, &[999], &[]);
        assert_eq!(idx_a, 0);

        let prev = frontier.prev_edges(idx_a);
        assert_eq!(prev.len(), 1);
        assert!(!prev[0].is_resident()); // parent not in arena
    }

    #[test]
    fn test_clear_is_o1() {
        let mut frontier = ActiveRoomFrontier::new();
        for i in 0..1000 {
            frontier.insert_event(i, &[], &[]);
        }
        assert_eq!(frontier.len(), 1000);
        frontier.clear();
        assert_eq!(frontier.len(), 0);
        assert!(frontier.is_empty());
    }

    #[test]
    fn test_id_remap() {
        let mut frontier = ActiveRoomFrontier::new();
        let local_a = frontier.register_id(100);
        let local_b = frontier.register_id(200);
        let local_a2 = frontier.register_id(100); // same id

        assert_eq!(local_a, 0);
        assert_eq!(local_b, 1);
        assert_eq!(local_a2, 0); // same as first
    }
}
