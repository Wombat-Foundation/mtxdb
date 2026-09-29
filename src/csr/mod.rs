use std::collections::{HashMap, VecDeque};

const CSR_VERSION: u8 = 1;

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
