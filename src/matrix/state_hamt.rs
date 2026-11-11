//! Matrix room state HAMTs: the Matrix layer over [`crate::hamt_store`].
//!
//! Every distinct state set becomes an `MTHR\x01` root record whose internal
//! nodes are content-addressed `MTHN\x01` records built by `rezzy`'s CHAMP
//! trie. Nodes and roots live in the room's `STAT` collection: a node under
//! `structural_hash[..16]` ([`crate::state_hamt_node_id`]), a root under
//! [`state_hamt_root_node_id`](crate::state_hamt::state_hamt_root_node_id). The unkeyed `LtHash` over the logical
//! `(event_type, state_key, event_id)` entries is the state-group id, both
//! inside the root and in the `STGP` instance record.
//!
//! Node storage, verification and canonical routing come from the generic
//! [`crate::hamt_store`]; this module adds the Matrix key type, the room's
//! collection and structural key, the `MTHR` root record, and `STGP` lookups.
//!
//! This is the one implementation shared by the importer and any server
//! integration, so both write the same bytes.

use std::fmt;

use rezzy::hamt_root::{StateHamtKey, StateHamtRoot};
use rezzy::incremental::LtHash;

use crate::hamt_store::{self, HamtCollection};

use crate::storage::{DigestAlgorithm, StorageEngine, StorageError};
use crate::{derive_group_member_collection_id, PackfileStorage, MEMBER_NAMESPACE_STAT};

/// A logical state entry: `(event_type, state_key, event_id)`.
pub type StateEntry = (String, String, String);

/// Why a persisted state HAMT could not be read.
#[derive(Debug)]
pub enum StateHamtError {
    /// The store failed.
    Storage(StorageError),
    /// A HAMT node was missing, failed verification, or did not decode.
    Node(String),
    /// A record was not the kind its address promised.
    InvalidRecord(String),
}

impl fmt::Display for StateHamtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(error) => write!(f, "state HAMT storage error: {error}"),
            Self::Node(message) => write!(f, "could not read a state HAMT node: {message}"),
            Self::InvalidRecord(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for StateHamtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            Self::Node(_) | Self::InvalidRecord(_) => None,
        }
    }
}

impl From<StorageError> for StateHamtError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

/// A materialized room state HAMT: its encoded root record and every
/// content-addressed node reachable from it.
pub struct BuiltStateHamt {
    /// Structural hash of the root node, stored in the root record.
    pub root_hash: [u8; 32],
    /// Unkeyed `LtHash` digest of the state set, i.e. the state-group id.
    pub state_group_id: [u8; 32],
    /// The encoded `MTHR\x01` root record.
    pub root_record: Vec<u8>,
    /// `(structural_hash, MTHN\x01 payload)` for every node in the tree.
    pub nodes: Vec<([u8; 32], Vec<u8>)>,
}

/// Build the flat state HAMT for one state set.
///
/// Leaf keys use rezzy's typed, length-prefixed `(EventType, String)` codec.
///
/// # Errors
/// Returns a description if `rezzy` cannot build the trie (e.g. a hash
/// collision below the maximum depth).
pub fn build_state_hamt(
    room_id: &str,
    room_prefix: &[u8],
    entries: &[StateEntry],
) -> Result<BuiltStateHamt, String> {
    let structural_key = room_structural_key(room_id);
    let mut lattice = LtHash::default();
    for (event_type, state_key, event_id) in entries {
        lattice.insert(event_type, state_key, event_id);
    }

    let hamt_entries = entries.iter().map(|(event_type, state_key, event_id)| {
        (state_hamt_leaf_key(event_type, state_key), event_id.clone())
    });
    let root = rezzy::hamt::build_hamt(&structural_key, hamt_entries)
        .map_err(|error| format!("failed to build state HAMT for {room_id}: {error:?}"))?;
    let handle = rezzy::hamt::RootHandle::from_lthash(root.structural_hash, &lattice);

    let nodes = hamt_store::encode_nodes(&root);

    let root_record =
        encode_state_hamt_root(room_prefix, room_id, &handle.structural_hash, &lattice);
    Ok(BuiltStateHamt {
        root_hash: handle.structural_hash,
        state_group_id: handle.state_group_id,
        root_record,
        nodes,
    })
}

/// Encode an `MTHR\x01` root record:
///
/// ```text
/// "MTHR" (4)
/// version 0x01 (1)
/// room-prefix length (u16 BE)
/// room prefix
/// room-id length (u16 BE)
/// UTF-8 room id
/// root structural hash (32)
/// lattice (2048 bytes: 1024 u16 lanes, little-endian)
/// ```
///
/// Byte-compatible with `_encode_state_hamt_root` in Synapse and its Rust
/// mirror.
///
/// # Panics
/// Panics if the room prefix or room id exceeds a `u16` length field.
#[must_use]
pub fn encode_state_hamt_root(
    room_prefix: &[u8],
    room_id: &str,
    root_hash: &[u8; 32],
    lattice: &LtHash,
) -> Vec<u8> {
    rezzy::hamt_root::encode_state_hamt_root(room_prefix, room_id, root_hash, lattice)
        .expect("room prefix and room id fit the u16 length fields")
}

/// Decode a persisted `MTHR\x01` root record, or `None` when the bytes are not
/// one.
#[must_use]
pub fn decode_state_hamt_root(bytes: &[u8]) -> Option<StateHamtRoot> {
    rezzy::hamt_root::decode_state_hamt_root(bytes).ok()
}

/// The structural key for a room's state HAMT.
///
/// This is the same derivation congruent uses, so a tree built here has the
/// same routing and structural hashes as one built at runtime.
#[must_use]
pub fn room_structural_key(room_id: &str) -> [u8; 32] {
    let mut hasher = DigestAlgorithm::Sha256.hasher();
    hasher.update(b"congruent state hamt structural key\0");
    hasher.update(room_id.as_bytes());
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest[..32]);
    key
}

/// The room-scoped prefix stored in root records. MSC4291 room ids are the
/// sigil `!` followed by an unpadded base64url-encoded 32-byte hash; those use
/// the decoded hash prefix. Ordinary `!localpart:server` ids use the legacy
/// SHA-256-of-text fallback.
#[must_use]
pub fn room_hamt_prefix(room_id: &str) -> [u8; 8] {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    if let Some(encoded) = room_id.strip_prefix('!') {
        if let Ok(decoded) = URL_SAFE_NO_PAD.decode(encoded) {
            if decoded.len() == 32 {
                let mut prefix = [0u8; 8];
                prefix.copy_from_slice(&decoded[..8]);
                return prefix;
            }
        }
    }
    let digest = DigestAlgorithm::Sha256.digest(room_id.as_bytes());
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    prefix
}

/// The room's `STAT` member collection id, where its HAMT nodes and roots live.
#[must_use]
pub fn state_hamt_collection_id(room_id: &str) -> Option<[u8; 16]> {
    derive_group_member_collection_id(MEMBER_NAMESPACE_STAT, room_id.as_bytes())
}

/// Derive a root record's node id from the (namespace, state-group id), keeping
/// roots in a distinct id space from `structural_hash[..16]` nodes. Mirrors
/// sithnapse's `state_group_root_node_id`.
///
/// # Panics
/// Panics if `namespace` is longer than `u32::MAX` bytes.
#[must_use]
pub fn state_hamt_root_node_id(namespace: &str, state_group_id: &[u8; 32]) -> [u8; 16] {
    let namespace_len = u32::try_from(namespace.len()).expect("namespace length fits u32");
    let mut hasher = DigestAlgorithm::Sha256.hasher();
    hasher.update(b"hamt:root-id:v1:");
    hasher.update(&namespace_len.to_be_bytes());
    hasher.update(namespace.as_bytes());
    hasher.update(state_group_id);
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// Repack edges for the state-side record kinds, or `None` when `data` is not
/// one of them (so the caller can try it as an event).
///
/// Edges point from a record to what it depends on, so that dependencies are
/// written first:
/// - `STGP` state group -> its parent groups, then its HAMT root record;
/// - `MTHR` root record -> its root HAMT node;
/// - HAMT internal node -> its child nodes.
///
/// Dispatch is by record magic, so one extractor serves every collection.
#[must_use]
pub fn extract_state_edges(data: &[u8]) -> Option<Vec<[u8; 16]>> {
    if let Ok(instance) = crate::decode_state_group_record(data) {
        let mut edges = instance.parents;
        edges.push(instance.root_id);
        return Some(edges);
    }
    if let Some(edges) = hamt_store::node_edges::<StateHamtKey, String>(data) {
        return Some(edges);
    }
    let root = decode_state_hamt_root(data)?;
    Some(vec![crate::state_hamt_node_id(&root.root_hash)])
}

/// The typed state key encoded by rezzy's length-prefixed HAMT codec.
#[must_use]
pub fn state_hamt_leaf_key(event_type: &str, state_key: &str) -> StateHamtKey {
    (event_type.into(), state_key.to_owned())
}

/// On-demand reader for one room's persisted state HAMTs.
///
/// Resolves only the nodes a traversal touches and verifies each against the
/// full structural hash, since the store key carries only its first 16 bytes.
pub struct StateHamtReader<'a> {
    nodes: HamtCollection<'a>,
}

impl<'a> StateHamtReader<'a> {
    /// Open the reader for `room_id`. Returns `None` when the room's `STAT`
    /// collection cannot be derived.
    #[must_use]
    pub fn new(store: &'a PackfileStorage, room_id: &str) -> Option<Self> {
        Some(Self {
            nodes: HamtCollection::new(
                store,
                state_hamt_collection_id(room_id)?,
                room_structural_key(room_id).to_vec(),
            ),
        })
    }

    /// The room's `STAT` collection id.
    #[must_use]
    pub fn collection(&self) -> [u8; 16] {
        self.nodes.collection()
    }

    /// The generic node collection underneath.
    #[must_use]
    pub fn nodes(&self) -> &HamtCollection<'a> {
        &self.nodes
    }

    /// Collect every `(event_type, state_key, event_id)` entry under `root_hash`.
    ///
    /// # Errors
    /// Returns an error when a node is absent or corrupt.
    pub fn collect_entries(
        &mut self,
        root_hash: &[u8; 32],
    ) -> Result<Vec<StateEntry>, StateHamtError> {
        let raw = self
            .nodes
            .collect_entries::<StateHamtKey, String>(*root_hash)
            .map_err(|error| StateHamtError::Node(error.to_string()))?;
        Ok(raw
            .into_iter()
            .map(|((event_type, state_key), event_id)| {
                (event_type.as_str().to_owned(), state_key, event_id)
            })
            .collect())
    }

    /// Look up one `(event_type, state_key)` under `root_hash`, routing with
    /// the canonical key hash the tree was built with.
    ///
    /// # Errors
    /// Returns an error when a node on the path is absent or corrupt.
    pub fn get(
        &mut self,
        root_hash: &[u8; 32],
        event_type: &str,
        state_key: &str,
    ) -> Result<Option<String>, StateHamtError> {
        self.nodes
            .get(*root_hash, &state_hamt_leaf_key(event_type, state_key))
            .map_err(|error| StateHamtError::Node(error.to_string()))
    }
}

/// Load the full state set of a persisted state group.
///
/// Returns `None` when the group has no `STGP` record or no materialized root
/// (an unresolved group). A present group missing a node is an error: its root
/// claims content that is not durable.
///
/// A loaded group is *complete* by construction: the writer suppresses
/// publication whenever the state computation fell back or an event's base was
/// missing history, so no incomplete group is ever stored.
///
/// # Errors
/// Returns an error for an unreadable record or a root that references a
/// missing or corrupt node.
pub fn load_state_group_entries(
    store: &PackfileStorage,
    room_id: &str,
    state_group_id: &[u8; 16],
) -> Result<Option<Vec<StateEntry>>, StateHamtError> {
    let Some(stgp_collection) = crate::state_group_collection_id(room_id) else {
        return Ok(None);
    };
    let Some(group) = store.get(&stgp_collection, state_group_id)? else {
        return Ok(None);
    };
    let instance = crate::decode_state_group_record(&group.bytes).map_err(|error| {
        StateHamtError::InvalidRecord(format!(
            "invalid state-group record {}: {error}",
            hex(state_group_id)
        ))
    })?;
    let Some(mut reader) = StateHamtReader::new(store, room_id) else {
        return Ok(None);
    };
    let Some(root) = store.get(&reader.collection(), &instance.root_id)? else {
        return Ok(None);
    };
    let Some(decoded) = decode_state_hamt_root(&root.bytes) else {
        return Err(StateHamtError::InvalidRecord(format!(
            "state-group root {} is not an MTHR v1 record",
            hex(&instance.root_id)
        )));
    };
    if decoded.state_group_id != instance.lthash {
        return Err(StateHamtError::InvalidRecord(format!(
            "state-group root {} does not match its STGP lattice digest",
            hex(&instance.root_id)
        )));
    }
    Ok(Some(reader.collect_entries(&decoded.root_hash)?))
}

fn hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    bytes.iter().fold(
        String::with_capacity(bytes.len().saturating_mul(2)),
        |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        },
    )
}
