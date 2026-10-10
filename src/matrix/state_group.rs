//! Matrix state-group instance identity (`STGP` namespace).
//!
//! A state group is identified at two independent layers:
//! - its **content** (the shared HAMT) is keyed by the state `LtHash` and
//!   deduplicates equal state sets;
//! - a **derivation** (one instance per predecessor set + resulting state) is
//!   keyed by an `STGP` instance id and preserves the derivation DAG.
//!
//! Instance ids are deterministic, so replaying the same derivation is
//! idempotent and no counter or random allocator is needed. Equal `LtHash`
//! values with different parents stay distinct.
//!
//! An [`StateGroupInstance`] record carries its parent ids, the full state
//! `LtHash`, and the shared HAMT root id; the instance id derives from the
//! first two (the root follows from the state content).

use crate::storage::DigestAlgorithm;
use crate::template::{
    derive_group_full_id, derive_group_member_collection_id, wrapping_add_le,
    MEMBER_NAMESPACE_STGP, NAMESPACE_BIAS_STGP,
};

/// Magic prefix of an encoded state-group instance record (`b"STGP"`).
pub const STATE_GROUP_RECORD_MAGIC: [u8; 4] = *b"STGP";
/// Version byte of the state-group instance record format.
pub const STATE_GROUP_RECORD_VERSION: u8 = 0x01;

/// Domain prefix for state-group instance identity hashing.
pub const STATE_GROUP_DOMAIN_PREFIX: &[u8] = b"mtxdb/state-group/v1";

/// A 128-bit state-group instance id (also a record/node id).
pub type StateGroupId = [u8; 16];

/// One state-group derivation: its predecessors, state identity, and shared
/// HAMT root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateGroupInstance {
    /// Predecessor instance ids; order is not significant.
    pub parents: Vec<StateGroupId>,
    /// The 256-bit state `LtHash` identifying the resulting state set.
    pub lthash: [u8; 32],
    /// The shared, content-addressed HAMT root id.
    pub root_id: StateGroupId,
}

impl StateGroupInstance {
    /// This instance's id for `room_id`.
    #[must_use]
    pub fn id(&self, room_id: &str) -> StateGroupId {
        state_group_instance_id(room_id, &self.parents, &self.lthash)
    }
}

/// Derive the 256-bit full instance id for a state-group derivation.
///
/// ```text
/// group_full_id    = BLAKE3-256("mtxdb/group/v1" || room_id)
/// stgp_full_id     = wrapping_add_le(group_full_id, NAMESPACE_BIAS_STGP)
/// instance_full_id = BLAKE3-256(domain || stgp_full_id || canonical(parents) || lthash)
/// ```
///
/// The `STGP` namespace bias enters through the member derivation
/// (`wrapping_add_le`), matching the engine's other member collections. Parents
/// are sorted and deduplicated first, so the id depends only on the *set* of
/// predecessors.
#[must_use]
pub fn state_group_instance_full_id(
    room_id: &str,
    parents: &[StateGroupId],
    lthash: &[u8; 32],
) -> [u8; 32] {
    let stgp_full_id = wrapping_add_le(
        derive_group_full_id(room_id.as_bytes()),
        NAMESPACE_BIAS_STGP,
    );

    let mut sorted = parents.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let parent_count = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    let mut hasher = DigestAlgorithm::Blake3.hasher();
    hasher.update(STATE_GROUP_DOMAIN_PREFIX);
    hasher.update(&stgp_full_id);
    hasher.update(&parent_count.to_le_bytes());
    for parent in &sorted {
        hasher.update(parent);
    }
    hasher.update(lthash);
    hasher.finalize()
}

/// Derive the 128-bit truncated instance id for a state-group derivation.
#[must_use]
pub fn state_group_instance_id(
    room_id: &str,
    parents: &[StateGroupId],
    lthash: &[u8; 32],
) -> StateGroupId {
    let full = state_group_instance_full_id(room_id, parents, lthash);
    let mut id = [0u8; 16];
    id.copy_from_slice(&full[..16]);
    id
}

/// The room's `STGP` member collection id, where its state-group instance
/// records live.
#[must_use]
pub fn state_group_collection_id(room_id: &str) -> Option<[u8; 16]> {
    derive_group_member_collection_id(MEMBER_NAMESPACE_STGP, room_id.as_bytes())
}

/// Encode an `STGP\x01` instance record:
///
/// ```text
/// "STGP" (4)
/// version 0x01 (1)
/// parent count (u16 BE)
/// parent instance ids (16 bytes each, sorted, deduplicated)
/// state LtHash digest (32)
/// STAT/HAMT root node id (16)
/// ```
///
/// Parents are canonicalized so identical derivations encode identical bytes.
///
/// # Panics
/// Panics if the instance has more than `u16::MAX` parent ids, which cannot be
/// reached by a bounded state-group derivation.
#[must_use]
#[allow(
    clippy::arithmetic_side_effects,
    reason = "fixed-width record layout arithmetic cannot overflow"
)]
pub fn encode_state_group_record(instance: &StateGroupInstance) -> Vec<u8> {
    let mut parents = instance.parents.clone();
    parents.sort_unstable();
    parents.dedup();
    let count = u16::try_from(parents.len()).expect("parent count fits u16");

    let mut out = Vec::with_capacity(7 + parents.len() * 16 + 32 + 16);
    out.extend_from_slice(&STATE_GROUP_RECORD_MAGIC);
    out.push(STATE_GROUP_RECORD_VERSION);
    out.extend_from_slice(&count.to_be_bytes());
    for parent in &parents {
        out.extend_from_slice(parent);
    }
    out.extend_from_slice(&instance.lthash);
    out.extend_from_slice(&instance.root_id);
    out
}

/// Decode an `STGP\x01` instance record produced by
/// [`encode_state_group_record`].
///
/// # Errors
/// Returns a static description if the magic, version, or length are wrong.
#[allow(
    clippy::arithmetic_side_effects,
    reason = "fixed-width record layout arithmetic cannot overflow"
)]
pub fn decode_state_group_record(bytes: &[u8]) -> Result<StateGroupInstance, &'static str> {
    if bytes.get(..4) != Some(STATE_GROUP_RECORD_MAGIC.as_slice()) {
        return Err("bad state-group record magic");
    }
    if bytes.get(4) != Some(&STATE_GROUP_RECORD_VERSION) {
        return Err("unsupported state-group record version");
    }
    let count = usize::from(u16::from_be_bytes([
        *bytes.get(5).ok_or("truncated state-group record")?,
        *bytes.get(6).ok_or("truncated state-group record")?,
    ]));
    if bytes.len() != 7 + count * 16 + 32 + 16 {
        return Err("state-group record length mismatch");
    }

    let mut offset = 7;
    let mut parents = Vec::with_capacity(count);
    for _ in 0..count {
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes[offset..offset + 16]);
        parents.push(id);
        offset += 16;
    }
    let mut lthash = [0u8; 32];
    lthash.copy_from_slice(&bytes[offset..offset + 32]);
    offset += 32;
    let mut root_id = [0u8; 16];
    root_id.copy_from_slice(&bytes[offset..offset + 16]);

    Ok(StateGroupInstance {
        parents,
        lthash,
        root_id,
    })
}

/// The access relations the state-group layer exposes.
///
/// The two views are the same records read differently: the primary relation
/// returns every derivation (`SELECT *`), while grouping on `lthash` collapses
/// identical state content (`SELECT DISTINCT`). Only [`Self::InstanceByEvent`]
/// must be stored; the rest derive from the STGP records themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateGroupRelation {
    /// `instance_id -> STGP instance` (primary record id).
    Instance,
    /// `lthash -> many instance ids` (group STGP by `lthash`).
    InstancesByLtHash,
    /// `parent instance id -> many child instances` (invert `parents`).
    ChildrenByParent,
    /// `event_id -> instance id` (the v3 auxiliary mapping).
    InstanceByEvent,
}

impl StateGroupRelation {
    /// Every relation, primary first.
    pub const ALL: [Self; 4] = [
        Self::Instance,
        Self::InstancesByLtHash,
        Self::ChildrenByParent,
        Self::InstanceByEvent,
    ];

    /// Whether one key resolves to many values.
    #[must_use]
    pub const fn is_many(self) -> bool {
        matches!(self, Self::InstancesByLtHash | Self::ChildrenByParent)
    }

    /// Whether the relation is derived from STGP records rather than stored.
    #[must_use]
    pub const fn is_derived(self) -> bool {
        matches!(self, Self::InstancesByLtHash | Self::ChildrenByParent)
    }
}

#[cfg(test)]
#[path = "test_state_group.rs"]
mod tests;
