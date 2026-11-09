//! Compressed sparse row graphs plus the pool-scoped store that persists them.
//!
//! [`CsrStore`](crate::csr::store::CsrStore) keeps derived graph rows outside the packs, one store per
//! source pool, addressed by `(collection id, family, owner)`.
pub mod store;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

const CSR_VERSION: u8 = 1;
const CSR_FAMILIES_MAGIC: &[u8; 4] = b"CSRF";
const CSR_FAMILIES_VERSION: u8 = 1;

/// A named adjacency family within a scope. Its edge lists are immutable once
/// written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeFamily {
    /// Family number; part of the edge record's address.
    pub id: u16,
    /// Whether each edge carries a `u16` kind (otherwise the kind is always 0).
    pub typed: bool,
}

impl EdgeFamily {
    /// An untyped family.
    #[must_use]
    pub const fn plain(id: u16) -> Self {
        Self { id, typed: false }
    }

    /// A family whose edges each carry a `u16` kind.
    #[must_use]
    pub const fn typed(id: u16) -> Self {
        Self { id, typed: true }
    }
}

/// A resolved edge: target short id plus kind (0 for untyped families).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    /// Target short id.
    pub target: u32,
    /// Kind id (0 when the family is untyped).
    pub kind: u16,
}

/// An edge to a not-yet-resolved key, as supplied by the caller.
#[derive(Debug, Clone, Copy)]
pub struct EdgeKey<'a> {
    /// Target key bytes; allocated a short id if new.
    pub target: &'a [u8],
    /// Kind id (must be 0 for untyped families).
    pub kind: u16,
}

/// Sparse reverse CSR for one edge family, built from its forward rows.
///
/// `parents` is a sorted sparse directory; `offsets` and `counts` identify
/// each parent's row in the flat `children` column. Typed families retain a
/// parallel `kind` value on each reverse edge. Build this once for a query
/// batch rather than once per parent lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReverseCsr {
    family: EdgeFamily,
    parents: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    children: Vec<Edge>,
}

impl ReverseCsr {
    /// Transpose forward rows `(child, parents)` into sparse parent rows.
    ///
    /// Child ids are sorted within each row and duplicate `(child, kind)`
    /// edges are collapsed. `counts` stores each row's cardinality so callers
    /// can answer count queries without decoding or scanning its children.
    ///
    /// # Errors
    /// Returns [`CsrError::Malformed`] if the reverse index exceeds `u32`
    /// address space.
    pub fn from_forward_rows(
        family: EdgeFamily,
        rows: &[(u32, Vec<Edge>)],
    ) -> Result<Self, CsrError> {
        let mut by_parent: BTreeMap<u32, BTreeSet<(u32, u16)>> = BTreeMap::new();
        for (child, parents) in rows {
            for edge in parents {
                by_parent
                    .entry(edge.target)
                    .or_default()
                    .insert((*child, edge.kind));
            }
        }

        let mut parents = Vec::with_capacity(by_parent.len());
        let mut offsets = Vec::with_capacity(by_parent.len().saturating_add(1));
        let mut counts = Vec::with_capacity(by_parent.len());
        let mut children = Vec::new();
        offsets.push(0);
        for (parent, row) in by_parent {
            let count = u32::try_from(row.len()).map_err(|_| CsrError::Malformed)?;
            parents.push(parent);
            counts.push(count);
            children.extend(row.into_iter().map(|(child, kind)| Edge {
                target: child,
                kind,
            }));
            offsets.push(u32::try_from(children.len()).map_err(|_| CsrError::Malformed)?);
        }

        Ok(Self {
            family,
            parents,
            offsets,
            counts,
            children,
        })
    }

    /// The forward family this index transposes.
    #[must_use]
    pub const fn family(&self) -> EdgeFamily {
        self.family
    }

    /// Sorted parent ids represented in this sparse directory.
    #[must_use]
    pub fn parents(&self) -> &[u32] {
        &self.parents
    }

    /// The parent row's children, sorted by child id and then edge kind.
    #[must_use]
    pub fn children_of(&self, parent: u32) -> Option<&[Edge]> {
        let index = self.parents.binary_search(&parent).ok()?;
        let start = *self.offsets.get(index)? as usize;
        let end = *self.offsets.get(index.saturating_add(1))? as usize;
        self.children.get(start..end)
    }

    /// O(1) cardinality after the sparse parent lookup.
    #[must_use]
    pub fn child_count(&self, parent: u32) -> Option<u32> {
        let index = self.parents.binary_search(&parent).ok()?;
        self.counts.get(index).copied()
    }
}

impl<'a> EdgeKey<'a> {
    /// An untyped edge to `target`.
    #[must_use]
    pub const fn plain(target: &'a [u8]) -> Self {
        Self { target, kind: 0 }
    }
}

/// Compressed sparse row graph for a collection's event DAG.
///
/// Stores adjacency information in two flat arrays (`offsets` and `targets`)
/// for cache-friendly traversal. O(1) neighbor lookup, O(V+E) topological sort.
#[derive(Debug, Clone)]
pub struct Csr {
    offsets: Vec<u32>,
    targets: Vec<u32>,
    hash_to_local: HashMap<[u8; 16], u32>,
    local_to_hash: Vec<[u8; 16]>,
}

/// Three forward CSR families sharing one node universe.
///
/// Referenced ids are included even when their event body is absent. The row
/// sets distinguish a known empty row from a referenced-but-missing event.
#[derive(Debug, Clone)]
pub struct CsrFamilies {
    /// Previous-event rows.
    pub prev: Csr,
    /// Auth-chain rows.
    pub auth: Csr,
    /// Relation rows.
    pub relations: Csr,
    nodes: Vec<[u8; 16]>,
    prev_rows: HashSet<[u8; 16]>,
    auth_rows: HashSet<[u8; 16]>,
    relation_rows: HashSet<[u8; 16]>,
    relation_types: HashMap<[u8; 16], Vec<u16>>,
}

impl CsrFamilies {
    /// Build the forward families, retaining every referenced event id.
    ///
    /// `relation_types` must contain one type id per target in each relation
    /// row. A missing type row is treated as an empty type list.
    #[must_use]
    pub fn build(
        event_ids: &[[u8; 16]],
        prev: &HashMap<[u8; 16], Vec<[u8; 16]>>,
        auth: &HashMap<[u8; 16], Vec<[u8; 16]>>,
        relations: &HashMap<[u8; 16], Vec<[u8; 16]>>,
        relation_types: HashMap<[u8; 16], Vec<u16>>,
    ) -> Self {
        let mut nodes = event_ids.to_vec();
        for rows in [prev, auth, relations] {
            for (source, targets) in rows {
                nodes.push(*source);
                nodes.extend(targets.iter().copied());
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        Self {
            prev: Csr::build_from_edges(&nodes, prev),
            auth: Csr::build_from_edges(&nodes, auth),
            relations: Csr::build_from_edges(&nodes, relations),
            nodes,
            prev_rows: prev.keys().copied().collect(),
            auth_rows: auth.keys().copied().collect(),
            relation_rows: relations.keys().copied().collect(),
            relation_types,
        }
    }

    /// All event ids in shared local-id order, including missing parents.
    #[must_use]
    pub fn nodes(&self) -> &[[u8; 16]] {
        &self.nodes
    }

    /// Whether a previous-event row was recorded (possibly empty).
    #[must_use]
    pub fn has_prev_row(&self, event_id: &[u8; 16]) -> bool {
        self.prev_rows.contains(event_id)
    }

    /// Whether an auth row was recorded (possibly empty).
    #[must_use]
    pub fn has_auth_row(&self, event_id: &[u8; 16]) -> bool {
        self.auth_rows.contains(event_id)
    }

    /// Whether a relation row was recorded.
    #[must_use]
    pub fn has_relation_row(&self, event_id: &[u8; 16]) -> bool {
        self.relation_rows.contains(event_id)
    }

    /// Relation type ids aligned with the source row's CSR targets.
    #[must_use]
    pub fn relation_types(&self, event_id: &[u8; 16]) -> &[u16] {
        self.relation_types.get(event_id).map_or(&[], Vec::as_slice)
    }

    /// Encode the three CSR families and their shared event-id directory.
    ///
    /// The segment is versioned and checksummed. Row-presence bytes preserve
    /// the distinction between an empty family and a missing event row.
    ///
    /// # Errors
    /// Returns [`CsrError::Malformed`] if relation type ids do not align with
    /// the relation targets or the segment exceeds its length fields.
    pub fn serialize(&self) -> Result<Vec<u8>, CsrError> {
        let relation_type_count = self
            .relation_types
            .values()
            .try_fold(0usize, |count, types| {
                count.checked_add(types.len()).ok_or(CsrError::Malformed)
            })?;
        if relation_type_count != self.relations.edge_count()
            || self.nodes.len() != self.prev.node_count()
            || self.nodes.len() != self.auth.node_count()
            || self.nodes.len() != self.relations.node_count()
        {
            return Err(CsrError::Malformed);
        }
        for (index, node) in self.nodes.iter().enumerate() {
            let local = u32::try_from(index).map_err(|_| CsrError::Malformed)?;
            if self.prev.hash_of(local) != Some(node)
                || self.auth.hash_of(local) != Some(node)
                || self.relations.hash_of(local) != Some(node)
            {
                return Err(CsrError::Malformed);
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(CSR_FAMILIES_MAGIC);
        out.push(CSR_FAMILIES_VERSION);
        out.extend_from_slice(
            &u32::try_from(self.nodes.len())
                .map_err(|_| CsrError::Malformed)?
                .to_le_bytes(),
        );
        for node in &self.nodes {
            out.extend_from_slice(node);
        }
        for (csr, rows) in [
            (&self.prev, &self.prev_rows),
            (&self.auth, &self.auth_rows),
            (&self.relations, &self.relation_rows),
        ] {
            for node in &self.nodes {
                out.push(u8::from(rows.contains(node)));
            }
            let encoded = csr.serialize();
            out.extend_from_slice(
                &u32::try_from(encoded.len())
                    .map_err(|_| CsrError::Malformed)?
                    .to_le_bytes(),
            );
            out.extend_from_slice(&encoded);
        }
        for node in &self.nodes {
            let local = self.relations.local_id(node).ok_or(CsrError::Malformed)?;
            let target_count = self.relations.neighbors(local).len();
            let types = self.relation_types.get(node).map_or(&[][..], Vec::as_slice);
            if target_count != types.len() {
                return Err(CsrError::Malformed);
            }
            for kind in types {
                out.extend_from_slice(&kind.to_le_bytes());
            }
        }
        let checksum = crc32fast::hash(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    /// Decode a versioned CSR family segment.
    ///
    /// # Errors
    /// Returns [`CsrError::TooShort`], [`CsrError::UnsupportedVersion`], or
    /// [`CsrError::Malformed`] for an invalid, truncated, or corrupt segment.
    pub fn deserialize(data: &[u8]) -> Result<Self, CsrError> {
        if data.len() < 13 {
            return Err(CsrError::TooShort);
        }
        let body_len = data.len().checked_sub(4).ok_or(CsrError::TooShort)?;
        let expected_crc = u32::from_le_bytes(
            data[body_len..]
                .try_into()
                .map_err(|_| CsrError::TooShort)?,
        );
        if crc32fast::hash(&data[..body_len]) != expected_crc {
            return Err(CsrError::Malformed);
        }
        if &data[..4] != CSR_FAMILIES_MAGIC {
            return Err(CsrError::Malformed);
        }
        if data[4] != CSR_FAMILIES_VERSION {
            return Err(CsrError::UnsupportedVersion(data[4]));
        }
        let count =
            u32::from_le_bytes(data[5..9].try_into().map_err(|_| CsrError::TooShort)?) as usize;
        let nodes_len = count.checked_mul(16).ok_or(CsrError::Malformed)?;
        let mut cursor = 9usize;
        let node_bytes = take(data, &mut cursor, nodes_len)?;
        let mut nodes = Vec::with_capacity(count);
        for chunk in node_bytes.as_chunks::<16>().0 {
            nodes.push(*chunk);
        }
        if nodes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CsrError::Malformed);
        }

        let mut row_sets = Vec::with_capacity(3);
        let mut decoded = Vec::with_capacity(3);
        for _ in 0..3 {
            let flags = take(data, &mut cursor, count)?;
            if flags.iter().any(|&flag| flag > 1) {
                return Err(CsrError::Malformed);
            }
            row_sets.push(
                nodes
                    .iter()
                    .zip(flags)
                    .filter_map(|(node, &flag)| (flag == 1).then_some(*node))
                    .collect::<HashSet<_>>(),
            );
            let length = read_u32(data, &mut cursor)? as usize;
            let bytes = take(data, &mut cursor, length)?;
            let csr = Csr::deserialize(bytes)?;
            if csr.node_count() != count {
                return Err(CsrError::Malformed);
            }
            decoded.push(csr);
        }

        validate_rows(&decoded, &row_sets, &nodes)?;

        let relation_edge_count = decoded[2].edge_count();
        let relation_bytes_len = relation_edge_count
            .checked_mul(2)
            .ok_or(CsrError::Malformed)?;
        let relation_bytes = take(data, &mut cursor, relation_bytes_len)?;
        if cursor != body_len {
            return Err(CsrError::Malformed);
        }
        let mut all_relation_types = Vec::with_capacity(relation_edge_count);
        for chunk in relation_bytes.as_chunks::<2>().0 {
            all_relation_types.push(u16::from_le_bytes(*chunk));
        }

        let prev = rebuild_with_nodes(&nodes, &decoded.remove(0))?;
        let auth = rebuild_with_nodes(&nodes, &decoded.remove(0))?;
        let relations = rebuild_with_nodes(&nodes, &decoded.remove(0))?;
        let mut relation_types = HashMap::new();
        let mut type_cursor = 0usize;
        for (local, node) in nodes.iter().enumerate() {
            let local = u32::try_from(local).map_err(|_| CsrError::Malformed)?;
            let edge_count = relations.neighbors(local).len();
            let end = type_cursor
                .checked_add(edge_count)
                .ok_or(CsrError::Malformed)?;
            if row_sets[2].contains(node) {
                relation_types.insert(*node, all_relation_types[type_cursor..end].to_vec());
            } else if edge_count != 0 {
                return Err(CsrError::Malformed);
            }
            type_cursor = end;
        }
        Ok(Self {
            prev,
            auth,
            relations,
            nodes,
            prev_rows: row_sets.remove(0),
            auth_rows: row_sets.remove(0),
            relation_rows: row_sets.remove(0),
            relation_types,
        })
    }
}

fn validate_rows(
    decoded: &[Csr],
    row_sets: &[HashSet<[u8; 16]>],
    nodes: &[[u8; 16]],
) -> Result<(), CsrError> {
    for (csr, rows) in decoded.iter().zip(row_sets) {
        for (local, node) in nodes.iter().enumerate() {
            let local = u32::try_from(local).map_err(|_| CsrError::Malformed)?;
            if !rows.contains(node) && !csr.neighbors(local).is_empty() {
                return Err(CsrError::Malformed);
            }
        }
    }
    Ok(())
}

fn take<'a>(data: &'a [u8], cursor: &mut usize, length: usize) -> Result<&'a [u8], CsrError> {
    let end = cursor.checked_add(length).ok_or(CsrError::Malformed)?;
    let bytes = data.get(*cursor..end).ok_or(CsrError::TooShort)?;
    *cursor = end;
    Ok(bytes)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, CsrError> {
    Ok(u32::from_le_bytes(
        take(data, cursor, 4)?
            .try_into()
            .map_err(|_| CsrError::TooShort)?,
    ))
}

fn rebuild_with_nodes(nodes: &[[u8; 16]], decoded: &Csr) -> Result<Csr, CsrError> {
    let mut adjacency = HashMap::new();
    for (local, source) in nodes.iter().enumerate() {
        let targets = decoded
            .neighbors(u32::try_from(local).map_err(|_| CsrError::Malformed)?)
            .iter()
            .map(|&target| {
                nodes
                    .get(target as usize)
                    .copied()
                    .ok_or(CsrError::Malformed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !targets.is_empty() {
            adjacency.insert(*source, targets);
        }
    }
    Ok(Csr::build_from_edges(nodes, &adjacency))
}

impl Csr {
    /// Build a CSR from an ordered node list and an adjacency map.
    ///
    /// `edges` defines node ordering (position = local ID).
    /// `adjacency` maps each node to its outgoing edge targets.
    /// Nodes in `adjacency` not present in `edges` are silently skipped.
    ///
    /// # Panics
    /// Panics if node count or edge count exceeds `u32::MAX`.
    #[must_use]
    pub fn build_from_edges(
        edges: &[[u8; 16]],
        adjacency: &HashMap<[u8; 16], Vec<[u8; 16]>>,
    ) -> Self {
        let mut hash_to_local = HashMap::new();
        let mut local_to_hash = Vec::with_capacity(edges.len());

        for (i, hash) in edges.iter().enumerate() {
            let local = u32::try_from(i).expect("node count exceeds u32::MAX");
            hash_to_local.insert(*hash, local);
            local_to_hash.push(*hash);
        }

        let n = edges.len();
        let mut offsets = Vec::with_capacity(n.wrapping_add(1));
        offsets.push(0u32);

        let mut targets = Vec::new();
        for hash in edges {
            let neighbors = adjacency.get(hash).map_or(&[][..], |v| v.as_slice());
            let start = u32::try_from(targets.len()).expect("edge count exceeds u32::MAX");
            for neighbor in neighbors {
                if let Some(&local) = hash_to_local.get(neighbor) {
                    targets.push(local);
                }
            }
            let end = u32::try_from(targets.len()).expect("edge count exceeds u32::MAX");
            offsets.push(end);
            let _ = start;
        }

        Self {
            offsets,
            targets,
            hash_to_local,
            local_to_hash,
        }
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.offsets.len().wrapping_sub(1)
    }

    /// Number of directed edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }

    /// Outgoing neighbors of a node (by local ID).
    #[must_use]
    pub fn neighbors(&self, local_id: u32) -> &[u32] {
        let start = self.offsets[local_id as usize] as usize;
        let next = (local_id as usize).saturating_add(1);
        let end = self.offsets[next] as usize;
        &self.targets[start..end]
    }

    /// Map a 16-byte hash to its dense local ID, if present.
    #[must_use]
    pub fn local_id(&self, hash: &[u8; 16]) -> Option<u32> {
        self.hash_to_local.get(hash).copied()
    }

    /// Map a dense local ID back to its 16-byte hash, if in range.
    #[must_use]
    pub fn hash_of(&self, local_id: u32) -> Option<&[u8; 16]> {
        self.local_to_hash.get(local_id as usize)
    }

    /// Kahn's algorithm: topological order over local IDs.
    ///
    /// Deterministic for a given graph: initially ready nodes are visited in
    /// local-ID order, and subsequently ready nodes are appended in CSR
    /// adjacency order.
    /// Returns fewer nodes than `node_count()` if cycles exist.
    #[must_use]
    pub fn topo_order(&self) -> Vec<u32> {
        let n = self.node_count();
        let mut in_degree = vec![0u32; n];

        let mut i: u32 = 0;
        while (i as usize) < n {
            for &target in self.neighbors(i) {
                in_degree[target as usize] = in_degree[target as usize].wrapping_add(1);
            }
            i = i.wrapping_add(1);
        }

        let mut queue: VecDeque<u32> = VecDeque::new();
        i = 0;
        while (i as usize) < n {
            if in_degree[i as usize] == 0 {
                queue.push_back(i);
            }
            i = i.wrapping_add(1);
        }

        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &target in self.neighbors(node) {
                in_degree[target as usize] = in_degree[target as usize].wrapping_sub(1);
                if in_degree[target as usize] == 0 {
                    queue.push_back(target);
                }
            }
        }

        order
    }

    /// Layered storage order, dependencies first.
    ///
    /// Edges are read as `node -> dependency` (an event points at its
    /// `prev_events`). A node's `topo_depth` is the length of the longest
    /// dependency chain beneath it, so nodes with no dependencies are level 0.
    /// Nodes are returned sorted by `(topo_depth, hash)`: every dependency
    /// precedes its dependents, and nodes within a level fall back to their
    /// id, which makes the order deterministic and independent of insertion
    /// or traversal order. Disconnected components interleave by level.
    ///
    /// Returns `None` if the graph has a cycle.
    #[must_use]
    pub fn layered_order(&self) -> Option<Vec<u32>> {
        let topo = self.topo_order();
        if topo.len() != self.node_count() {
            return None;
        }
        let mut depth = vec![0u32; self.node_count()];
        // `topo` lists a node before its targets, so walking it backwards
        // finalizes every dependency before its dependents.
        for &node in topo.iter().rev() {
            let mut level = 0u32;
            for &dep in self.neighbors(node) {
                level = level.max(depth[dep as usize].saturating_add(1));
            }
            depth[node as usize] = level;
        }
        let mut order = topo;
        order.sort_unstable_by(|&a, &b| {
            depth[a as usize]
                .cmp(&depth[b as usize])
                .then_with(|| self.local_to_hash[a as usize].cmp(&self.local_to_hash[b as usize]))
        });
        Some(order)
    }

    /// [`Self::layered_order`], or, if the graph has a cycle, plain id order.
    ///
    /// The flag is `true` when the fallback was taken. A cycle only means the
    /// layout cannot follow dependencies; callers writing records can still
    /// proceed.
    #[must_use]
    pub fn layered_or_id_order(&self) -> (Vec<u32>, bool) {
        if let Some(order) = self.layered_order() {
            return (order, false);
        }
        let mut order: Vec<u32> = (0..self.node_count())
            .filter_map(|i| u32::try_from(i).ok())
            .collect();
        order.sort_unstable_by(|&a, &b| {
            self.local_to_hash[a as usize].cmp(&self.local_to_hash[b as usize])
        });
        (order, true)
    }

    /// Serialize the CSR to a byte buffer (mmap-compatible).
    ///
    /// Format: `[version:u8=1][nodes:u32 LE][edges:u32 LE][offsets...][targets...]`
    ///
    /// # Panics
    /// Panics if node count or edge count exceeds `u32::MAX`.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let n = u32::try_from(self.node_count()).expect("node count exceeds u32::MAX");
        let e = u32::try_from(self.edge_count()).expect("edge count exceeds u32::MAX");
        let byte_len = 9usize
            .wrapping_add((n as usize).wrapping_add(1).wrapping_mul(4))
            .wrapping_add((e as usize).wrapping_mul(4));

        let mut buf = Vec::with_capacity(byte_len);
        buf.push(CSR_VERSION);
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&e.to_le_bytes());

        for &off in &self.offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        for &tgt in &self.targets {
            buf.extend_from_slice(&tgt.to_le_bytes());
        }

        buf
    }

    /// Deserialize a CSR from a byte buffer.
    ///
    /// The hash↔local mapping is not preserved (rebuild with `build_from_edges` if needed).
    ///
    /// # Errors
    /// Returns [`CsrError::TooShort`] if the buffer is too small,
    /// [`CsrError::UnsupportedVersion`] if the version byte is unrecognized,
    /// or [`CsrError::Malformed`] if the offsets or targets are inconsistent.
    ///
    /// # Panics
    /// Panics if the 8-byte header fields cannot be read (guaranteed by the length check).
    pub fn deserialize(data: &[u8]) -> Result<Self, CsrError> {
        if data.len() < 9 {
            return Err(CsrError::TooShort);
        }

        let version = data[0];
        if version != CSR_VERSION {
            return Err(CsrError::UnsupportedVersion(version));
        }

        let n_u32 = u32::from_le_bytes(data[1..5].try_into().unwrap());
        let e_u32 = u32::from_le_bytes(data[5..9].try_into().unwrap());
        let n = n_u32 as usize;
        let e = e_u32 as usize;

        let offsets_len = n.checked_add(1).ok_or(CsrError::Malformed)?;
        let offsets_bytes = offsets_len.checked_mul(4).ok_or(CsrError::Malformed)?;
        let targets_bytes = e.checked_mul(4).ok_or(CsrError::Malformed)?;
        let expected = 9usize
            .checked_add(offsets_bytes)
            .and_then(|size| size.checked_add(targets_bytes))
            .ok_or(CsrError::Malformed)?;
        if data.len() < expected {
            return Err(CsrError::TooShort);
        }

        let mut offsets = Vec::with_capacity(offsets_len);
        for i in 0..offsets_len {
            let off = 9usize.wrapping_add(i.wrapping_mul(4));
            offsets.push(u32::from_le_bytes(
                data[off..off.wrapping_add(4)].try_into().unwrap(),
            ));
        }

        let targets_start = 9usize.wrapping_add(offsets_bytes);
        let mut targets = Vec::with_capacity(e);
        for i in 0..e {
            let off = targets_start.wrapping_add(i.wrapping_mul(4));
            targets.push(u32::from_le_bytes(
                data[off..off.wrapping_add(4)].try_into().unwrap(),
            ));
        }

        if offsets.first() != Some(&0) {
            return Err(CsrError::Malformed);
        }
        if offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err(CsrError::Malformed);
        }
        if offsets.last().copied() != Some(e_u32) {
            return Err(CsrError::Malformed);
        }
        if targets.iter().any(|&t| t >= n_u32) {
            return Err(CsrError::Malformed);
        }

        Ok(Self {
            offsets,
            targets,
            hash_to_local: HashMap::new(),
            local_to_hash: Vec::new(),
        })
    }

    /// Estimated heap memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            .wrapping_add(self.offsets.capacity().wrapping_mul(4))
            .wrapping_add(self.targets.capacity().wrapping_mul(4))
            .wrapping_add(self.hash_to_local.capacity().wrapping_mul(
                std::mem::size_of::<([u8; 16], u32)>().wrapping_add(std::mem::size_of::<usize>()),
            ))
            .wrapping_add(self.local_to_hash.capacity().wrapping_mul(16))
    }
}

/// Errors from CSR serialization or deserialization.
#[derive(Debug)]
pub enum CsrError {
    /// The byte buffer is too short for a valid CSR.
    TooShort,
    /// The CSR version byte is not supported.
    UnsupportedVersion(u8),
    /// The offsets or targets are inconsistent (malformed data).
    Malformed,
}

impl std::fmt::Display for CsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "csr data too short"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported csr version: {v}"),
            Self::Malformed => write!(f, "csr data is malformed"),
        }
    }
}

impl std::error::Error for CsrError {}

#[cfg(test)]
#[path = "test_layered.rs"]
mod test_layered;
