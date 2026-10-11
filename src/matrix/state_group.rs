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
//! `LtHash`, the shared HAMT root id, and the [`StateGroupDerivation`] that
//! produced it; the instance id derives from the parents, the `LtHash`, and
//! that derivation (the root follows from the state content).
//!
//! The derivation is part of the identity on purpose: the same parents and the
//! same state reached through two different resolvers, or under two different
//! trust claims, are two different claims about the room and must never share
//! an id.

use crate::storage::DigestAlgorithm;
use crate::template::{
    derive_group_full_id, derive_group_member_collection_id, wrapping_add_le,
    MEMBER_NAMESPACE_STGP, NAMESPACE_BIAS_STGP,
};

/// Magic prefix of an encoded state-group instance record (`b"STGP"`).
pub const STATE_GROUP_RECORD_MAGIC: [u8; 4] = *b"STGP";
/// Version byte of the state-group instance record format.
///
/// `0x02` adds the [`StateGroupDerivation`] tags after the version byte;
/// `0x01` records predate resolver/trust-aware identity and are rejected.
pub const STATE_GROUP_RECORD_VERSION: u8 = 0x02;

/// Domain prefix for state-group instance identity hashing.
///
/// Bumped from `v1` when resolver and trust became part of the derivation, so
/// ids derived under the old first-parent rule cannot collide with ids derived
/// under a real resolver.
pub const STATE_GROUP_DOMAIN_PREFIX: &[u8] = b"mtxdb/state-group/v2";

/// A 128-bit state-group instance id (also a record/node id).
pub type StateGroupId = [u8; 16];

/// The state-resolution algorithm that produced a derivation.
///
/// The same parents and the same `LtHash` resolved by two algorithms are two
/// different derivations, so the resolver enters the instance id: a group
/// produced by the retired first-parent merge never collides with one produced
/// by Matrix State Resolution, and v1 never collides with v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateGroupResolver {
    /// The retired first-parent-wins merge, kept so that groups it wrote stay
    /// decodable and distinguishable from real resolutions.
    FirstParent,
    /// Matrix State Resolution v1 (room version 1).
    V1,
    /// Matrix State Resolution v2 (room versions 2-11).
    V2,
    /// Matrix State Resolution v2.1 (room version 12).
    V2_1,
    /// Matrix State Resolution v2.1.1 (room version 12.1): v2.1 plus the
    /// ban-evasion hardening rezzy gates behind this version.
    V2_1_1,
    /// Matrix State Resolution v2.2 (MSC4242 state DAGs).
    V2_2,
    /// Rezzy v3 certified causal governance (`tk.nutra.cdo.12`).
    V3,
}

impl StateGroupResolver {
    /// Stable wire tag stored in `STGP` records and hashed into instance ids.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::FirstParent => 0,
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V2_1 => 3,
            Self::V2_1_1 => 4,
            Self::V2_2 => 5,
            Self::V3 => 6,
        }
    }

    /// Parse a tag written by [`Self::tag`].
    ///
    /// # Errors
    /// Returns a static description for an unknown tag.
    pub const fn from_tag(tag: u8) -> Result<Self, &'static str> {
        match tag {
            0 => Ok(Self::FirstParent),
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V2_1),
            4 => Ok(Self::V2_1_1),
            5 => Ok(Self::V2_2),
            6 => Ok(Self::V3),
            _ => Err("unknown state-group resolver tag"),
        }
    }

    /// Short display name used in metadata and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FirstParent => "first-parent",
            Self::V1 => "state-res-v1",
            Self::V2 => "state-res-v2",
            Self::V2_1 => "state-res-v2.1",
            Self::V2_1_1 => "state-res-v2.1.1",
            Self::V2_2 => "state-res-v2.2",
            Self::V3 => "state-res-v3",
        }
    }
}

/// The validation claim satisfied before a derivation was computed.
///
/// Recorded in the identity so a group computed from unverified input cannot
/// be mistaken for one computed from input whose signatures and auth rules
/// mtxdb checked itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateGroupTrust {
    /// Inputs arrived with dispositions attached; no signature or auth-rule
    /// verification ran. The importer's default.
    TrustedImport,
    /// mtxdb verified event signatures, content hashes, and auth rules.
    ValidatedImport,
    /// Persisted validation results were replayed rather than recomputed.
    Replay,
}

impl StateGroupTrust {
    /// Stable wire tag stored in `STGP` records and hashed into instance ids.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::TrustedImport => 0,
            Self::ValidatedImport => 1,
            Self::Replay => 2,
        }
    }

    /// Parse a tag written by [`Self::tag`].
    ///
    /// # Errors
    /// Returns a static description for an unknown tag.
    pub const fn from_tag(tag: u8) -> Result<Self, &'static str> {
        match tag {
            0 => Ok(Self::TrustedImport),
            1 => Ok(Self::ValidatedImport),
            2 => Ok(Self::Replay),
            _ => Err("unknown state-group trust tag"),
        }
    }

    /// Short display name used in metadata and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrustedImport => "trusted-import",
            Self::ValidatedImport => "validated-import",
            Self::Replay => "replay",
        }
    }
}

/// How a state-group derivation was produced: which resolver ran, and what the
/// importer had proven about its inputs beforehand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateGroupDerivation {
    /// The state-resolution algorithm used.
    pub resolver: StateGroupResolver,
    /// The validation claim satisfied before resolving.
    pub trust: StateGroupTrust,
}

impl StateGroupDerivation {
    /// Bundles the two halves of a derivation.
    #[must_use]
    pub const fn new(resolver: StateGroupResolver, trust: StateGroupTrust) -> Self {
        Self { resolver, trust }
    }

    /// The two tags as hashed/serialized bytes, resolver first.
    #[must_use]
    pub const fn tag_bytes(self) -> [u8; 2] {
        [self.resolver.tag(), self.trust.tag()]
    }

    /// `"state-res-v2/trusted-import"` — for metadata and diagnostics.
    #[must_use]
    #[allow(clippy::string_slice, reason = "const str concat, not slicing")]
    pub fn as_str(&self) -> String {
        format!("{}/{}", self.resolver.as_str(), self.trust.as_str())
    }
}

/// One state-group derivation: its predecessors, state identity, shared HAMT
/// root, and the rule that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateGroupInstance {
    /// Predecessor instance ids; order is not significant.
    pub parents: Vec<StateGroupId>,
    /// The 256-bit state `LtHash` identifying the resulting state set.
    pub lthash: [u8; 32],
    /// The shared, content-addressed HAMT root id.
    pub root_id: StateGroupId,
    /// The resolver and trust claim this instance was derived under.
    pub derivation: StateGroupDerivation,
}

impl StateGroupInstance {
    /// This instance's id for `room_id`.
    #[must_use]
    pub fn id(&self, room_id: &str) -> StateGroupId {
        state_group_instance_id(room_id, &self.parents, &self.lthash, self.derivation)
    }
}

/// Derive the 256-bit full instance id for a state-group derivation.
///
/// ```text
/// group_full_id    = BLAKE3-256("mtxdb/group/v1" || room_id)
/// stgp_full_id     = wrapping_add_le(group_full_id, NAMESPACE_BIAS_STGP)
/// instance_full_id = BLAKE3-256(domain || stgp_full_id || resolver || trust
///                               || canonical(parents) || lthash)
/// ```
///
/// The `STGP` namespace bias enters through the member derivation
/// (`wrapping_add_le`), matching the engine's other member collections. Parents
/// are sorted and deduplicated first, so the id depends only on the *set* of
/// predecessors. The derivation tags follow the namespace bias, so the same
/// state reached under a different resolver or trust claim is a different id.
#[must_use]
pub fn state_group_instance_full_id(
    room_id: &str,
    parents: &[StateGroupId],
    lthash: &[u8; 32],
    derivation: StateGroupDerivation,
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
    hasher.update(&derivation.tag_bytes());
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
    derivation: StateGroupDerivation,
) -> StateGroupId {
    let full = state_group_instance_full_id(room_id, parents, lthash, derivation);
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

/// Encode an `STGP\x02` instance record:
///
/// ```text
/// "STGP" (4)
/// version 0x02 (1)
/// resolver tag (1)
/// trust tag (1)
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

    let mut out = Vec::with_capacity(8 + parents.len() * 16 + 32 + 16);
    out.extend_from_slice(&STATE_GROUP_RECORD_MAGIC);
    out.push(STATE_GROUP_RECORD_VERSION);
    out.push(instance.derivation.resolver.tag());
    out.push(instance.derivation.trust.tag());
    out.extend_from_slice(&count.to_be_bytes());
    for parent in &parents {
        out.extend_from_slice(parent);
    }
    out.extend_from_slice(&instance.lthash);
    out.extend_from_slice(&instance.root_id);
    out
}

/// Decode an `STGP\x02` instance record produced by
/// [`encode_state_group_record`].
///
/// # Errors
/// Returns a static description if the magic, version, derivation tags, or
/// length are wrong.
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
    let resolver =
        StateGroupResolver::from_tag(*bytes.get(5).ok_or("truncated state-group record")?)?;
    let trust = StateGroupTrust::from_tag(*bytes.get(6).ok_or("truncated state-group record")?)?;
    let count = usize::from(u16::from_be_bytes([
        *bytes.get(7).ok_or("truncated state-group record")?,
        *bytes.get(8).ok_or("truncated state-group record")?,
    ]));
    if bytes.len() != 9 + count * 16 + 32 + 16 {
        return Err("state-group record length mismatch");
    }

    let mut offset = 9;
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
        derivation: StateGroupDerivation::new(resolver, trust),
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
    /// `event_id -> instance id` (the state-group auxiliary mapping).
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
