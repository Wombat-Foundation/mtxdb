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
    #[must_use]
    /// Returns whether this edge points to an arena-resident node.
    pub fn is_resident(self) -> bool {
        (self.0 & Self::RESIDENT_BIT) != 0
    }

    /// Returns the arena index for a resident edge.
    ///
    /// # Panics
    /// Panics if this is a disk-resident edge.
    #[inline]
    #[must_use]
    pub fn arena_index(self) -> usize {
        assert!(
            self.is_resident(),
            "arena_index called on a disk-resident node"
        );
        (self.0 & Self::PAYLOAD_MASK) as usize
    }

    /// Returns the local ID for a disk-resident edge.
    ///
    /// # Panics
    /// Panics if this is an arena-resident edge.
    #[inline]
    #[must_use]
    pub fn local_id(self) -> LocalId {
        assert!(
            !self.is_resident(),
            "local_id called on an arena-resident node"
        );
        self.0 & Self::PAYLOAD_MASK
    }

    /// Create a resident edge pointing to an arena index.
    ///
    /// # Panics
    /// Panics if `index` exceeds the 31-bit payload capacity.
    #[inline]
    #[must_use]
    pub fn resident(index: u32) -> Self {
        assert!(
            index <= Self::PAYLOAD_MASK,
            "resident index exceeds payload capacity: {index} > {}",
            Self::PAYLOAD_MASK
        );
        Self(Self::RESIDENT_BIT | (index & Self::PAYLOAD_MASK))
    }

    /// Create a disk-resident edge with a local ID.
    ///
    /// # Panics
    /// Panics if `id` exceeds the 31-bit payload capacity.
    #[inline]
    #[must_use]
    pub fn disk(id: LocalId) -> Self {
        assert!(
            id <= Self::PAYLOAD_MASK,
            "disk id exceeds payload capacity: {id} > {}",
            Self::PAYLOAD_MASK
        );
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
    /// Range into the edges array for `prev_events`: (start, len).
    pub prev: (u32, u32),
    /// Range into the edges array for `auth_events`: (start, len).
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
    /// Maps global `short_id` → index into nodes Vec.
    pub resident: HashMap<u64, u32>,
    /// Maps global `short_id` → dense `local_id` for this room.
    pub id_remap: HashMap<u64, LocalId>,
    /// Next `local_id` to assign.
    next_local_id: LocalId,
}

impl ActiveRoomFrontier {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            resident: HashMap::new(),
            id_remap: HashMap::new(),
            next_local_id: 0,
        }
    }

    /// Register a global `short_id` and assign it a dense `local_id`.
    pub fn register_id(&mut self, short_id: u64) -> LocalId {
        if let Some(&local) = self.id_remap.get(&short_id) {
            return local;
        }
        let local = self.next_local_id;
        self.next_local_id = self.next_local_id.saturating_add(1);
        self.id_remap.insert(short_id, local);
        local
    }

    /// Resolve a parent edge: if the parent is in the arena, use a resident
    /// edge; if it has a `local_id`, use a disk edge; otherwise assign a new
    /// `local_id` and use a disk edge.
    fn resolve_edge(&mut self, parent_id: u64) -> GraphEdge {
        let local = match self.id_remap.get(&parent_id) {
            Some(&local) => local,
            None => self.register_id(parent_id),
        };
        if let Some(&slot) = self.resident.get(&parent_id) {
            GraphEdge::resident(slot)
        } else {
            GraphEdge::disk(local)
        }
    }

    /// Insert an event with its raw prev and auth edges.
    ///
    /// For each parent in `raw_prevs` and `raw_auths`:
    /// - If the parent is already in the arena, creates a `resident` edge.
    /// - Otherwise, creates a `disk` edge with the parent's `local_id`.
    ///
    /// # Panics
    /// Panics if the arena index exceeds `u32::MAX`.
    pub fn insert_event(&mut self, short_id: u64, raw_prevs: &[u64], raw_auths: &[u64]) -> usize {
        let local_id = self.register_id(short_id);

        let node_idx: u32 =
            u32::try_from(self.nodes.len()).expect("too many events for u32 arena index");

        let prev_start: u32 =
            u32::try_from(self.edges.len()).expect("too many edges for u32 arena index");
        for &parent_id in raw_prevs {
            let edge = self.resolve_edge(parent_id);
            self.edges.push(edge);
        }
        let prev_end: u32 =
            u32::try_from(self.edges.len()).expect("too many edges for u32 arena index");
        let prev_len = prev_end
            .checked_sub(prev_start)
            .expect("prev_len underflow");

        let auth_start: u32 =
            u32::try_from(self.edges.len()).expect("too many edges for u32 arena index");
        for &parent_id in raw_auths {
            let edge = self.resolve_edge(parent_id);
            self.edges.push(edge);
        }
        let auth_end: u32 =
            u32::try_from(self.edges.len()).expect("too many edges for u32 arena index");
        let auth_len = auth_end
            .checked_sub(auth_start)
            .expect("auth_len underflow");

        self.nodes.push(EventNode {
            short_id,
            local_id,
            prev: (prev_start, prev_len),
            auth: (auth_start, auth_len),
        });
        self.resident.insert(short_id, node_idx);

        node_idx as usize
    }

    /// Get the `prev_edges` for a node by its arena index.
    ///
    /// # Panics
    /// Panics if `node_idx` is out of bounds.
    #[must_use]
    pub fn prev_edges(&self, node_idx: usize) -> &[GraphEdge] {
        let node = &self.nodes[node_idx];
        let start = node.prev.0 as usize;
        let end = start.wrapping_add(node.prev.1 as usize);
        &self.edges[start..end]
    }

    /// Get the `auth_edges` for a node by its arena index.
    ///
    /// # Panics
    /// Panics if `node_idx` is out of bounds.
    #[must_use]
    pub fn auth_edges(&self, node_idx: usize) -> &[GraphEdge] {
        let node = &self.nodes[node_idx];
        let start = node.auth.0 as usize;
        let end = start.wrapping_add(node.auth.1 as usize);
        &self.edges[start..end]
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
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
#[cfg_attr(coverage_nightly, coverage(off))]
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

        let idx_a = frontier.insert_event(100, &[], &[]);
        assert_eq!(idx_a, 0);
        assert_eq!(frontier.len(), 1);

        let idx_b = frontier.insert_event(101, &[100], &[100]);
        assert_eq!(idx_b, 1);
        assert_eq!(frontier.len(), 2);

        let prev = frontier.prev_edges(idx_b);
        assert_eq!(prev.len(), 1);
        assert!(prev[0].is_resident());
        assert_eq!(prev[0].arena_index(), 0);
    }

    #[test]
    fn test_insert_event_with_disk_parent() {
        let mut frontier = ActiveRoomFrontier::new();

        let idx_a = frontier.insert_event(100, &[999], &[]);
        assert_eq!(idx_a, 0);

        let prev = frontier.prev_edges(idx_a);
        assert_eq!(prev.len(), 1);
        assert!(!prev[0].is_resident());
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
    fn test_auth_edges_resident_and_disk() {
        let mut frontier = ActiveRoomFrontier::new();
        let a = frontier.insert_event(100, &[], &[]);
        let b = frontier.insert_event(101, &[], &[100]);
        let c = frontier.insert_event(102, &[], &[100, 999]);

        assert_eq!(frontier.auth_edges(a), &[]);

        let auth_b = frontier.auth_edges(b);
        assert_eq!(auth_b.len(), 1);
        assert!(auth_b[0].is_resident());
        assert_eq!(auth_b[0].arena_index(), a);

        let auth_c = frontier.auth_edges(c);
        assert_eq!(auth_c.len(), 2);
        assert!(auth_c[0].is_resident());
        assert_eq!(auth_c[0].arena_index(), a);
        assert!(!auth_c[1].is_resident());
    }

    #[test]
    fn test_active_room_frontier_default() {
        let frontier = ActiveRoomFrontier::default();
        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
    }

    #[test]
    fn test_id_remap() {
        let mut frontier = ActiveRoomFrontier::new();
        let local_a = frontier.register_id(100);
        let local_b = frontier.register_id(200);
        let local_a2 = frontier.register_id(100);

        assert_eq!(local_a, 0);
        assert_eq!(local_b, 1);
        assert_eq!(local_a2, 0);
    }
}
