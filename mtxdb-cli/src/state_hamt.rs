//! Materialize Matrix room state HAMTs in the Synapse/sithnapse wire format.
//!
//! This mirrors the flat path of `sithnapse/rust/src/state_hamt.rs`: every
//! distinct state set becomes a `MTHR\x01` root record whose internal nodes are
//! content-addressed `MTHN\x01` records built by `rezzy`'s CHAMP trie. The
//! unkeyed [`LtHash`] over the logical `(event_type, state_key, event_id)`
//! entries is the state-group id, both in the aux index and inside the root.
//!
//! The record bytes are portable to Synapse; only the collection/node key
//! layout is CLI-owned (`sithnapse` additionally keeps an integer state-group
//! sequence, `room_index`, and reference-count nodes that this importer has no
//! need for).

use std::collections::HashSet;
use std::sync::Arc;

use mtxdb::{derive_group_member_collection_id, DigestAlgorithm, MEMBER_NAMESPACE_STAT};
use rezzy::hamt::{build_hamt_root_handle, HamtNode, NodeRef, PersistedInternalNode};
use rezzy::state::LtHash;

/// Little-endian byte length of the retained `LtHash` lattice (1024 `u16`
/// lanes) stored in every root record.
const LATTICE_BYTES: usize = 2048;

/// A materialized room state HAMT: its encoded root record and every
/// content-addressed node reachable from it.
pub(crate) struct BuiltStateHamt {
    /// Structural hash of the root node, stored in the root record.
    #[allow(dead_code, reason = "exposed for callers that address the root node")]
    pub(crate) root_hash: [u8; 32],
    /// Unkeyed `LtHash` digest of the state set, i.e. the state-group id.
    pub(crate) state_group_id: [u8; 32],
    /// The encoded `MTHR\x01` root record.
    pub(crate) root_record: Vec<u8>,
    /// `(structural_hash, MTHN\x01 payload)` for every node in the tree.
    pub(crate) nodes: Vec<([u8; 32], Vec<u8>)>,
}

/// Build the flat state HAMT for one state set.
///
/// `entry` triples are `(event_type, state_key, event_id)`. The leaf key is
/// `serde_json`'s encoding of `(event_type, state_key)` so the on-disk bytes
/// match Synapse exactly.
///
/// # Errors
/// Returns a description if `rezzy` cannot build the trie (e.g. a hash
/// collision below the maximum depth).
pub(crate) fn build_state_hamt(
    room_id: &str,
    room_prefix: &[u8],
    entries: &[(String, String, String)],
) -> Result<BuiltStateHamt, String> {
    let structural_key = room_id.as_bytes();
    let mut lattice = LtHash::default();
    for (event_type, state_key, event_id) in entries {
        lattice.insert(event_type, state_key, event_id);
    }

    let hamt_entries = entries
        .iter()
        .map(|(event_type, state_key, event_id)| {
            (state_hamt_leaf_key(event_type, state_key), event_id.clone())
        });
    let (handle, root) = build_hamt_root_handle(structural_key, &lattice, hamt_entries)
        .map_err(|error| format!("failed to build state HAMT for {room_id}: {error:?}"))?;

    let mut seen = HashSet::new();
    let mut nodes = Vec::new();
    collect_persisted_nodes(&root, &mut seen, &mut nodes);

    let root_record = encode_state_hamt_root(room_prefix, room_id, &handle.structural_hash, &lattice);
    Ok(BuiltStateHamt {
        root_hash: handle.structural_hash,
        state_group_id: handle.state_group_id,
        root_record,
        nodes,
    })
}

/// Post-order walk collecting `(structural_hash, encoded node)` exactly once.
/// Mirrors sithnapse's `collect_persisted_nodes`.
fn collect_persisted_nodes(
    node: &Arc<HamtNode<String, String>>,
    seen: &mut HashSet<[u8; 32]>,
    nodes: &mut Vec<([u8; 32], Vec<u8>)>,
) {
    if !seen.insert(node.structural_hash) {
        return;
    }
    for child in &node.children {
        if let NodeRef::Resolved(child_node) = child {
            collect_persisted_nodes(child_node, seen, nodes);
        }
    }
    let persisted: PersistedInternalNode<String, String> = node.as_ref().into();
    nodes.push((node.structural_hash, persisted.encode_v1()));
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
#[must_use]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "fixed-width record layout arithmetic cannot overflow"
)]
pub(crate) fn encode_state_hamt_root(
    room_prefix: &[u8],
    room_id: &str,
    root_hash: &[u8; 32],
    lattice: &LtHash,
) -> Vec<u8> {
    let prefix_len = u16::try_from(room_prefix.len()).expect("room prefix fits u16");
    let room_id_len = u16::try_from(room_id.len()).expect("room id fits u16");

    let mut out = Vec::new();
    out.extend_from_slice(b"MTHR");
    out.push(0x01);
    out.extend_from_slice(&prefix_len.to_be_bytes());
    out.extend_from_slice(room_prefix);
    out.extend_from_slice(&room_id_len.to_be_bytes());
    out.extend_from_slice(room_id.as_bytes());
    out.extend_from_slice(root_hash);
    for lane in &lattice.0 {
        out.extend_from_slice(&lane.to_le_bytes());
    }
    debug_assert_eq!(
        out.len(),
        7 + room_prefix.len() + 2 + room_id.len() + 32 + LATTICE_BYTES,
        "MTHR root record length"
    );
    out
}

/// The room-scoped prefix stored in root records: `SHA-256(room_id)[..8]` for
/// ordinary rooms. (MSC4291 hash-shaped room ids would decode their own
/// base64url hash instead; the CLI imports ordinary `!localpart:server` rooms.)
#[must_use]
pub(crate) fn room_hamt_prefix(room_id: &str) -> [u8; 8] {
    let digest = DigestAlgorithm::Sha256.digest(room_id.as_bytes());
    let mut prefix = [0u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    prefix
}

/// The room's `STAT` member collection id, where its HAMT nodes and roots live.
#[must_use]
pub(crate) fn state_hamt_collection_id(room_id: &str) -> Option<[u8; 16]> {
    derive_group_member_collection_id(MEMBER_NAMESPACE_STAT, room_id.as_bytes())
}

/// Derive a root record's node id from the (namespace, state-group id), keeping
/// roots in a distinct id space from `structural_hash[..16]` nodes. Mirrors
/// sithnapse's `state_group_root_node_id`.
#[must_use]
pub(crate) fn state_hamt_root_node_id(namespace: &str, state_group_id: &[u8; 32]) -> [u8; 16] {
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

/// Base64url-encode a state-group id for the `event_id -> state_group_id` aux
/// index.
#[must_use]
pub(crate) fn encode_state_group_id(id: &[u8; 32]) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    URL_SAFE_NO_PAD.encode(id)
}

/// The state-group id (unkeyed `LtHash` digest) for a set of logical entries.
#[must_use]
pub(crate) fn state_group_id(entries: &[(String, String, String)]) -> [u8; 32] {
    let mut lattice = LtHash::default();
    for (event_type, state_key, event_id) in entries {
        lattice.insert(event_type, state_key, event_id);
    }
    rezzy::hamt::state_group_id_from_lthash(&lattice)
}

/// `serde_json::to_string(&(event_type, state_key))`, i.e. a two-element JSON
/// array. Implemented directly so the CLI need not pull in `serde_json` and so
/// the escaping is pinned to `serde_json`'s default.
#[must_use]
pub(crate) fn state_hamt_leaf_key(event_type: &str, state_key: &str) -> String {
    let mut out = String::new();
    out.push('[');
    push_json_string(&mut out, event_type);
    out.push(',');
    push_json_string(&mut out, state_key);
    out.push(']');
    out
}

fn push_json_string(out: &mut String, value: &str) {
    use std::fmt::Write as _;

    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if u32::from(control) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(control));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<(String, String, String)> {
        vec![
            (
                "m.room.create".to_owned(),
                "".to_owned(),
                "$create:example.org".to_owned(),
            ),
            (
                "m.room.member".to_owned(),
                "@alice:example.org".to_owned(),
                "$member:example.org".to_owned(),
            ),
            (
                "m.room.name".to_owned(),
                "".to_owned(),
                "$name:example.org".to_owned(),
            ),
        ]
    }

    #[test]
    fn build_is_deterministic_and_uses_lthash_identity() {
        let input = entries();
        let first = build_state_hamt("!room:example.org", b"prefix", &input).unwrap();
        let second = build_state_hamt("!room:example.org", b"prefix", &input).unwrap();

        assert_eq!(first.state_group_id, state_group_id(&input));
        assert_eq!(first.root_record, second.root_record);
        assert_eq!(first.nodes, second.nodes);
        assert!(first.root_record.starts_with(b"MTHR\x01"));
        assert!(!first.nodes.is_empty());
        assert!(first
            .nodes
            .iter()
            .all(|(_, payload)| payload.starts_with(b"MTHN\x01")));
    }

    #[test]
    fn state_group_identity_is_order_independent() {
        let mut reversed = entries();
        reversed.reverse();
        assert_eq!(state_group_id(&entries()), state_group_id(&reversed));
        assert_eq!(
            build_state_hamt("!room:example.org", b"prefix", &entries())
                .unwrap()
                .root_record,
            build_state_hamt("!room:example.org", b"prefix", &reversed)
                .unwrap()
                .root_record
        );
    }

    #[test]
    fn leaf_keys_match_json_array_encoding() {
        assert_eq!(
            state_hamt_leaf_key("m.room\"name", "line\nkey"),
            r#"["m.room\"name","line\nkey"]"#
        );
        assert_eq!(room_hamt_prefix("!room:example.org").len(), 8);
    }
}
