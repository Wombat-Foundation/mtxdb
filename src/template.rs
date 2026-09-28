//! Executable policy primitives for format-specific collection templates.
//!
//! Storage remains format-agnostic. Importers bind a collection to one of
//! these policies and persist the resulting metadata alongside their index.

use std::borrow::Cow;

use crate::storage::{Digest32, DigestAlgorithm, NodeId};

/// Current generic collection-template format identifier.
pub const COLLECTION_TEMPLATE_FORMAT_V1: &str = "mtxdb.collection-template/v1";

/// How a record's stored payload is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadPolicy {
    /// Retain the complete received source record.
    Source,
    /// Retain only an explicit, derived projection of the source record.
    Projection {
        /// RFC 6901 pointers to fields that should be included in the projection.
        include: Vec<String>,
    },
}

/// Which bytes a frame's 256-bit logical identity (`logical_id`) is derived
/// from.
///
/// Protocol-neutral by construction: this selects *what* gets hashed, while
/// [`DigestAlgorithm`] selects *how*. Protocol-specific rules (Matrix reference
/// hashes, redaction, and the like) are inputs to [`FrameIdPolicy::Canonical`]
/// rather than a parallel family of enums.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameIdPolicy {
    /// Hash the value at an RFC 6901 pointer into the retained source record.
    Pointer {
        /// RFC 6901 pointer to a stable, source-level identity.
        pointer: String,
    },
    /// Hash the stored payload bytes.
    Payload,
    /// Hash an immutable header/descriptor — the same strategy a pack uses for
    /// its own address. `fields` names the descriptor fields to include, and is
    /// honored by whoever assembles [`FrameIdInput::descriptor`].
    HeaderDescriptor {
        /// Descriptor field names to include in the hashed bytes.
        fields: Vec<String>,
    },
    /// Hash a canonicalized form of the source record.
    ///
    /// `include` and `exclude_prefixes` are honored by the caller's
    /// canonicalizer (e.g. a protocol extension applying a redaction policy);
    /// the resulting canonical bytes are supplied in [`FrameIdInput::canonical`].
    Canonical {
        /// RFC 6901 pointers to fields retained in the canonical form.
        include: Vec<String>,
        /// Field-name prefixes stripped before canonicalization.
        exclude_prefixes: Vec<String>,
    },
    /// The full logical identity is supplied by the caller and only
    /// cross-checked, never derived.
    ExternalCanonicalIdToCrosscheck,
    /// The record ID is a caller-supplied key digest (e.g. BLAKE3(key)[..16]).
    /// Validation and mapping to the underlying key are owned by the application layer.
    Key,
}

/// The byte sources a [`FrameIdPolicy`] may draw from.
pub struct FrameIdInput<'a> {
    /// The stored payload bytes.
    pub payload: &'a [u8],
    /// The immutable header/descriptor bytes, if the frame has one.
    pub descriptor: &'a [u8],
    /// The canonicalized source bytes, if the caller produced them.
    pub canonical: Option<&'a [u8]>,
    /// Resolves an RFC 6901 pointer against the retained source record.
    pub resolve: &'a dyn Fn(&str) -> Option<Vec<u8>>,
}

/// Derive a frame's 256-bit logical identity under `policy`.
///
/// Returns `None` when the policy's input is unavailable — a missing pointer or
/// canonical form, or no descriptor — or when the identity is
/// [`FrameIdPolicy::ExternalCanonicalIdToCrosscheck`] or [`FrameIdPolicy::Key`]
/// and must be supplied by the caller instead.
#[must_use]
pub fn frame_digest(
    policy: &FrameIdPolicy,
    algorithm: DigestAlgorithm,
    input: &FrameIdInput<'_>,
) -> Option<Digest32> {
    let bytes: Cow<'_, [u8]> = match policy {
        FrameIdPolicy::Payload => Cow::Borrowed(input.payload),
        FrameIdPolicy::HeaderDescriptor { .. } => Cow::Borrowed(input.descriptor),
        FrameIdPolicy::Pointer { pointer } => Cow::Owned((input.resolve)(pointer)?),
        FrameIdPolicy::Canonical { .. } => Cow::Borrowed(input.canonical?),
        FrameIdPolicy::ExternalCanonicalIdToCrosscheck | FrameIdPolicy::Key => return None,
    };
    Some(algorithm.digest(&bytes))
}

/// Derive the 128-bit record routing key from a full identity digest.
///
/// The full digest remains the authoritative identity when it is available;
/// this fixed-width prefix is only the core index key. Keeping the truncation
/// here prevents adapters from independently choosing different slices.
///
/// # Panics
///
/// Never in practice: [`Digest32`] is exactly 32 bytes, so its first 16 bytes
/// always fit [`NodeId`].
#[must_use]
pub fn record_logical_id(digest: &Digest32) -> NodeId {
    digest[..16]
        .try_into()
        .expect("the first 16 bytes of Digest32 always fit NodeId")
}

/// 256-bit little-endian wrapping addition: `(a + b) mod 2^256`.
///
/// Unrolled 4-limb `u64` carry chain lowering to `ADD` / `ADC` in optimized codegen.
#[inline]
#[must_use]
pub const fn wrapping_add_le(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
    let a0 = u64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
    let a1 = u64::from_le_bytes([a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]]);
    let a2 = u64::from_le_bytes([a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23]]);
    let a3 = u64::from_le_bytes([a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]]);

    let b0 = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    let b1 = u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    let b2 = u64::from_le_bytes([b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]]);
    let b3 = u64::from_le_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]]);

    let (r0, carry0) = a0.overflowing_add(b0);
    let (r1, carry1) = add_with_carry(a1, b1, carry0);
    let (r2, carry2) = add_with_carry(a2, b2, carry1);
    let (r3, _carry3) = add_with_carry(a3, b3, carry2);

    let o0 = r0.to_le_bytes();
    let o1 = r1.to_le_bytes();
    let o2 = r2.to_le_bytes();
    let o3 = r3.to_le_bytes();

    [
        o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], o1[0], o1[1], o1[2], o1[3], o1[4],
        o1[5], o1[6], o1[7], o2[0], o2[1], o2[2], o2[3], o2[4], o2[5], o2[6], o2[7], o3[0], o3[1],
        o3[2], o3[3], o3[4], o3[5], o3[6], o3[7],
    ]
}

#[inline]
const fn add_with_carry(left: u64, right: u64, carry: bool) -> (u64, bool) {
    let (sum, left_right_carry) = left.overflowing_add(right);
    let (sum, carry_carry) = sum.overflowing_add(carry as u64);
    (sum, left_right_carry || carry_carry)
}

/// Domain prefix for collection group canonical identity hashing.
pub const GROUP_DOMAIN_PREFIX: &[u8] = b"mtxdb/group/v1";

/// Member namespace for event DAG collections (`b"EVNT"`).
pub const MEMBER_NAMESPACE_EVNT: [u8; 4] = *b"EVNT";
/// Member namespace for previous-event edge collections (`b"PREV"`).
pub const MEMBER_NAMESPACE_PREV: [u8; 4] = *b"PREV";
/// Member namespace for auth-chain edge collections (`b"AUTH"`).
pub const MEMBER_NAMESPACE_AUTH: [u8; 4] = *b"AUTH";
/// Member namespace for state collections (`b"STAT"`).
pub const MEMBER_NAMESPACE_STAT: [u8; 4] = *b"STAT";
/// Member namespace for internal system collections (`b"INTL"`).
pub const MEMBER_NAMESPACE_INTL: [u8; 4] = *b"INTL";

/// Fixed 32-byte namespace bias constant for `b"EVNT"`.
///
/// Computed as `BLAKE3-256("mtxdb/namespace/v1/EVNT")`.
pub const NAMESPACE_BIAS_EVNT: [u8; 32] = [
    0x67, 0xe8, 0xdb, 0xce, 0x5b, 0x34, 0x85, 0xb6, 0x95, 0x6a, 0x63, 0x66, 0x9c, 0xa0, 0x23, 0xe0,
    0x37, 0xce, 0x1e, 0x2f, 0xf4, 0x39, 0x96, 0xeb, 0xcf, 0x58, 0x88, 0x5b, 0xd5, 0x61, 0xb1, 0xd2,
];

/// Fixed 32-byte namespace bias constant for `b"PREV"`.
///
/// Computed as `BLAKE3-256("mtxdb/namespace/v1/PREV")`.
pub const NAMESPACE_BIAS_PREV: [u8; 32] = [
    0xd0, 0x8a, 0xc3, 0x18, 0xa2, 0x59, 0x2a, 0xaf, 0xe4, 0xf8, 0xaa, 0xfd, 0x91, 0xd9, 0x0d, 0x87,
    0x82, 0x34, 0x7e, 0x04, 0xa8, 0xdf, 0x07, 0xd7, 0xcc, 0x86, 0xfa, 0x8f, 0x77, 0x99, 0x86, 0xb2,
];

/// Fixed 32-byte namespace bias constant for `b"AUTH"`.
///
/// Computed as `BLAKE3-256("mtxdb/namespace/v1/AUTH")`.
pub const NAMESPACE_BIAS_AUTH: [u8; 32] = [
    0x1a, 0x23, 0xc0, 0x78, 0x8b, 0x71, 0x12, 0x9d, 0x05, 0xd8, 0x8a, 0xcd, 0xe4, 0x81, 0xe8, 0x8a,
    0x9a, 0x13, 0x34, 0x2e, 0x80, 0x18, 0x45, 0x0e, 0x4b, 0x95, 0xd1, 0x87, 0x25, 0xf4, 0xb6, 0x77,
];

/// Fixed 32-byte namespace bias constant for `b"STAT"`.
///
/// Computed as `BLAKE3-256("mtxdb/namespace/v1/STAT")`.
pub const NAMESPACE_BIAS_STAT: [u8; 32] = [
    0xb7, 0x41, 0xe5, 0x66, 0xc4, 0x8f, 0xa6, 0x55, 0x3b, 0xac, 0xf3, 0x3f, 0x03, 0xcb, 0x4a, 0xb3,
    0x45, 0x9d, 0x81, 0xae, 0xce, 0xde, 0x93, 0x9f, 0x16, 0xbe, 0xf2, 0x35, 0xf7, 0x89, 0xeb, 0x7d,
];

/// Fixed 32-byte namespace bias constant for `b"INTL"`.
///
/// Computed as `BLAKE3-256("mtxdb/namespace/v1/INTL")`.
pub const NAMESPACE_BIAS_INTL: [u8; 32] = [
    0x16, 0xd7, 0xdf, 0x62, 0xd4, 0xb3, 0x8e, 0x78, 0x19, 0x0b, 0x9d, 0x89, 0x55, 0x05, 0x52, 0x97,
    0xd4, 0x2e, 0xf5, 0x44, 0xe0, 0x21, 0xaf, 0x3c, 0x98, 0x44, 0xa4, 0xb6, 0x5c, 0x64, 0x38, 0xe3,
];

/// Map a 4-byte member namespace to its fixed 32-byte bias constant.
///
/// Returns `Some(bias)` for recognized member namespaces (`EVNT`, `PREV`, `AUTH`, `STAT`, `INTL`).
/// Returns `None` for unrecognized namespaces (such as physical pool tags like `b"EDGE"`).
#[must_use]
pub const fn namespace_bias(namespace: [u8; 4]) -> Option<[u8; 32]> {
    if namespace[0] == b'E' && namespace[1] == b'V' && namespace[2] == b'N' && namespace[3] == b'T'
    {
        Some(NAMESPACE_BIAS_EVNT)
    } else if namespace[0] == b'P'
        && namespace[1] == b'R'
        && namespace[2] == b'E'
        && namespace[3] == b'V'
    {
        Some(NAMESPACE_BIAS_PREV)
    } else if namespace[0] == b'A'
        && namespace[1] == b'U'
        && namespace[2] == b'T'
        && namespace[3] == b'H'
    {
        Some(NAMESPACE_BIAS_AUTH)
    } else if namespace[0] == b'S'
        && namespace[1] == b'T'
        && namespace[2] == b'A'
        && namespace[3] == b'T'
    {
        Some(NAMESPACE_BIAS_STAT)
    } else if namespace[0] == b'I'
        && namespace[1] == b'N'
        && namespace[2] == b'T'
        && namespace[3] == b'L'
    {
        Some(NAMESPACE_BIAS_INTL)
    } else {
        None
    }
}

/// Derive the 256-bit full logical identity for a collection group.
///
/// Formula: `BLAKE3-256("mtxdb/group/v1" || group_canonical_id)`
#[must_use]
pub fn derive_group_full_id(group_canonical_id: &[u8]) -> [u8; 32] {
    let mut hasher = DigestAlgorithm::Blake3.hasher();
    hasher.update(GROUP_DOMAIN_PREFIX);
    hasher.update(group_canonical_id);
    hasher.finalize()
}

/// Derive the 256-bit full logical identity for a group member directly from an
/// already-computed 256-bit group logical ID.
///
/// Formula: `wrapping_add_le(group_full_id, NAMESPACE_BIAS[namespace])`
///
/// This avoids re-hashing `group_canonical_id` when deriving multiple member IDs
/// for the same collection group (e.g. `EVNT`, `PREV`, `AUTH`, `STAT`).
#[must_use]
pub fn derive_member_full_id_from_group(
    member_namespace: [u8; 4],
    group_full_id: [u8; 32],
) -> Option<[u8; 32]> {
    let bias = namespace_bias(member_namespace)?;
    Some(wrapping_add_le(group_full_id, bias))
}

/// Derive the 128-bit truncated collection ID for a group member directly from an
/// already-computed 256-bit group logical ID.
///
/// Formula: `wrapping_add_le(group_full_id, NAMESPACE_BIAS[namespace])[..16]`
#[must_use]
pub fn derive_member_collection_id_from_group(
    member_namespace: [u8; 4],
    group_full_id: [u8; 32],
) -> Option<[u8; 16]> {
    let full_id = derive_member_full_id_from_group(member_namespace, group_full_id)?;
    let mut collection_id = [0u8; 16];
    collection_id.copy_from_slice(&full_id[..16]);
    Some(collection_id)
}

/// Derive the 256-bit full logical identity for a group member.
///
/// Formula:
/// ```text
/// group_full_id = BLAKE3-256("mtxdb/group/v1" || group_canonical_id)
/// member_full_id = wrapping_add_le(group_full_id, NAMESPACE_BIAS[namespace])
/// ```
///
/// Returns `None` if `member_namespace` is not a registered member namespace.
#[must_use]
pub fn derive_group_member_full_id(
    member_namespace: [u8; 4],
    group_canonical_id: &[u8],
) -> Option<[u8; 32]> {
    let group_full_id = derive_group_full_id(group_canonical_id);
    derive_member_full_id_from_group(member_namespace, group_full_id)
}

/// Derive the 128-bit truncated collection ID for a group member.
///
/// Formula:
/// `member_full_id[0..16]`
///
/// Returns `None` if `member_namespace` is not a registered member namespace.
#[must_use]
pub fn derive_group_member_collection_id(
    member_namespace: [u8; 4],
    group_canonical_id: &[u8],
) -> Option<[u8; 16]> {
    let full_id = derive_group_member_full_id(member_namespace, group_canonical_id)?;
    let mut collection_id = [0u8; 16];
    collection_id.copy_from_slice(&full_id[..16]);
    Some(collection_id)
}

/// Try to derive the 256-bit full logical identity for a collection from its optional
/// member namespace and group canonical ID.
///
/// When `member_namespace` is `Some(ns)`, delegates to [`derive_group_member_full_id`].
/// When `None`, computes [`derive_group_full_id`].
///
/// Returns `None` if an unrecognized member namespace was provided.
#[must_use]
pub fn try_derive_collection_full_id(
    member_namespace: Option<[u8; 4]>,
    collection_canonical_id: &[u8],
) -> Option<[u8; 32]> {
    if let Some(ns) = member_namespace {
        derive_group_member_full_id(ns, collection_canonical_id)
    } else {
        Some(derive_group_full_id(collection_canonical_id))
    }
}

/// Try to derive a collection's 128-bit logical ID from its optional member namespace
/// and group canonical ID.
///
/// Returns `None` if `member_namespace` is `Some(ns)` and `ns` is not a recognized
/// member namespace (e.g. if a physical pool tag like `b"EDGE"` was mistakenly passed).
#[must_use]
pub fn try_derive_collection_id(
    member_namespace: Option<[u8; 4]>,
    collection_canonical_id: &[u8],
) -> Option<[u8; 16]> {
    let full = try_derive_collection_full_id(member_namespace, collection_canonical_id)?;
    let mut id = [0u8; 16];
    id.copy_from_slice(&full[..16]);
    Some(id)
}

/// Derive a collection's 128-bit logical ID from its optional member namespace
/// and group canonical ID (e.g. `b"!room:server"`).
///
/// Convenience helper intended for static, compile-time registered namespaces
/// (such as `Some(MEMBER_NAMESPACE_EVNT)` or `None`).
///
/// # Panics
/// Panics if `member_namespace` is `Some(ns)` and `ns` is not a registered member
/// namespace (`EVNT`, `PREV`, `AUTH`, `STAT`, `INTL`).
/// Physical pool tags like `b"EDGE"` must **never** be passed here.
/// For fallible callers handling untrusted or dynamic input, use [`try_derive_collection_id`].
#[must_use]
pub fn derive_collection_id(
    member_namespace: Option<[u8; 4]>,
    collection_canonical_id: &[u8],
) -> [u8; 16] {
    try_derive_collection_id(member_namespace, collection_canonical_id)
        .expect("member_namespace must be a valid registered namespace (EVNT, PREV, AUTH, STAT, INTL) or None")
}

/// The reserved node id under which a collection's metadata (genesis) record
/// is stored.
///
/// Reserved by construction: it is the only frame ever written under this id,
/// so it needs no derived-hash ceremony — a value we control is exactly as
/// collision-safe as a hash of a reserved string.
pub const COLLECTION_METADATA_RECORD_ID: [u8; 16] = *b"mtxdb:metadata\0\0";

/// A collection's first-class, immutable definition, written once as its
/// metadata (genesis) record under [`COLLECTION_METADATA_RECORD_ID`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionMetadata {
    // --- Identity pre-image (hashed into `collection_logical_id`) ---
    /// Member namespace (4 bytes, e.g. `b"EVNT"`, `b"PREV"`, `b"AUTH"`, `b"STAT"`).
    /// Used with wrapping bias addition to derive `collection_logical_id`.
    ///
    /// Named `member_namespace` per contract, distinct from physical storage pool.
    pub member_namespace: Option<[u8; 4]>,
    /// The collection's **canonical** id — the caller-defined external key
    /// (e.g. `!room:server`), canonicalized.
    ///
    /// # Group Canonical ID
    /// Under the group derivation scheme, this represents the **group canonical ID**
    /// from which all member collections (events, state, prev-edges, auth-edges)
    /// deterministically derive their logical collection IDs via deterministic bias addition.
    pub collection_canonical_id: Vec<u8>,
    // --- Descriptive (NOT hashed; safe to change/extend) ---
    /// How frames in this collection calculate their `record_logical_id`.
    pub record_id_rule: RecordIdentityRule,
    /// Source payload retention rule.
    pub payload: PayloadPolicy,
    /// Optional, opaque application payload owned by the template/protocol
    /// layer — for example a self-describing Matrix room blob
    /// (`{"ext":"matrix.room","fmt":1,"room_version":"10",...}`). Core never
    /// interprets it; it is written once with the genesis metadata and handed
    /// back to the application on open, so a reader can learn protocol
    /// configuration (room version, creator) from the header without seeking
    /// to and parsing the establishment record.
    pub extension: Option<Vec<u8>>,
    /// Logical role of this collection (e.g. "`event_dag`", "`state_hamt`", "`system_auxiliary`").
    pub role: Option<String>,
    /// Optional wire schema / format version string (e.g. "matrix.event.v1").
    pub schema: Option<String>,
}

impl CollectionMetadata {
    /// The collection's group canonical ID.
    #[must_use]
    pub fn group_canonical_id(&self) -> &[u8] {
        &self.collection_canonical_id
    }
}

/// `CollectionMetadata` TLV tags.
const META_TAG_MEMBER_NAMESPACE: u8 = 0x01;
#[allow(dead_code)]
const META_TAG_POOL_DST: u8 = META_TAG_MEMBER_NAMESPACE;
const META_TAG_COLLECTION_CANONICAL_ID: u8 = 0x02;
const META_TAG_RECORD_ID_RULE: u8 = 0x03;
const META_TAG_PAYLOAD: u8 = 0x04;
const META_TAG_EXTENSION: u8 = 0x05;
const META_TAG_ROLE: u8 = 0x06;
const META_TAG_SCHEMA: u8 = 0x07;

/// `RecordIdentityRule` nested tags.
const IDENTITY_TAG_DIGEST_ALGORITHM: u8 = 0x01;
const IDENTITY_TAG_POLICY: u8 = 0x02;

/// [`FrameIdPolicy`] nested tags.
const POLICY_TAG_POINTER: u8 = 0x01;
const POLICY_TAG_PAYLOAD: u8 = 0x02;
const POLICY_TAG_HEADER_DESCRIPTOR: u8 = 0x03;
const POLICY_TAG_CANONICAL: u8 = 0x04;
const POLICY_TAG_EXTERNAL: u8 = 0x05;
const POLICY_TAG_KEY: u8 = 0x06;

/// [`FrameIdPolicy::Canonical`] nested tags.
const CANONICAL_TAG_INCLUDE: u8 = 0x01;
const CANONICAL_TAG_EXCLUDE_PREFIXES: u8 = 0x02;

/// Append one `[tag:1][len:4 LE][value]` record.
fn push_tlv(out: &mut Vec<u8>, tag: u8, value: &[u8]) {
    out.push(tag);
    out.extend_from_slice(
        &u32::try_from(value.len())
            .expect("collection-metadata field fits u32")
            .to_le_bytes(),
    );
    out.extend_from_slice(value);
}

/// Encode a list of strings as repeated `[len:4 LE][utf8]`.
fn encode_string_list(list: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in list {
        out.extend_from_slice(
            &u32::try_from(item.len())
                .expect("string fits u32")
                .to_le_bytes(),
        );
        out.extend_from_slice(item.as_bytes());
    }
    out
}

/// Cursor over a TLV block.
struct TlvReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> TlvReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn next(&mut self) -> Option<(u8, &'a [u8])> {
        let tag = *self.bytes.get(self.cursor)?;
        let len_start = self.cursor.checked_add(1)?;
        let len_end = self.cursor.checked_add(5)?;
        let len: [u8; 4] = self.bytes.get(len_start..len_end)?.try_into().ok()?;
        let value_end = len_end.checked_add(u32::from_le_bytes(len) as usize)?;
        let value = self.bytes.get(len_end..value_end)?;
        self.cursor = value_end;
        Some((tag, value))
    }

    fn is_exhausted(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

/// Decode repeated `[len:4 LE][utf8]` into a string list.
fn decode_string_list(mut bytes: &[u8]) -> Option<Vec<String>> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let len: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        let end = 4usize.checked_add(u32::from_le_bytes(len) as usize)?;
        out.push(std::str::from_utf8(bytes.get(4..end)?).ok()?.to_owned());
        bytes = bytes.get(end..)?;
    }
    Some(out)
}

fn encode_frame_id_policy(policy: &FrameIdPolicy) -> Vec<u8> {
    let mut out = Vec::new();
    match policy {
        FrameIdPolicy::Pointer { pointer } => {
            push_tlv(&mut out, POLICY_TAG_POINTER, pointer.as_bytes());
        }
        FrameIdPolicy::Payload => push_tlv(&mut out, POLICY_TAG_PAYLOAD, &[]),
        FrameIdPolicy::HeaderDescriptor { fields } => {
            push_tlv(
                &mut out,
                POLICY_TAG_HEADER_DESCRIPTOR,
                &encode_string_list(fields),
            );
        }
        FrameIdPolicy::Canonical {
            include,
            exclude_prefixes,
        } => {
            let mut inner = Vec::new();
            push_tlv(
                &mut inner,
                CANONICAL_TAG_INCLUDE,
                &encode_string_list(include),
            );
            push_tlv(
                &mut inner,
                CANONICAL_TAG_EXCLUDE_PREFIXES,
                &encode_string_list(exclude_prefixes),
            );
            push_tlv(&mut out, POLICY_TAG_CANONICAL, &inner);
        }
        FrameIdPolicy::ExternalCanonicalIdToCrosscheck => {
            push_tlv(&mut out, POLICY_TAG_EXTERNAL, &[]);
        }
        FrameIdPolicy::Key => {
            push_tlv(&mut out, POLICY_TAG_KEY, &[]);
        }
    }
    out
}

fn decode_frame_id_policy(bytes: &[u8]) -> Option<FrameIdPolicy> {
    let mut reader = TlvReader::new(bytes);
    let (tag, value) = reader.next()?;
    if !reader.is_exhausted() {
        return None;
    }
    Some(match tag {
        POLICY_TAG_POINTER => FrameIdPolicy::Pointer {
            pointer: std::str::from_utf8(value).ok()?.to_owned(),
        },
        POLICY_TAG_PAYLOAD => FrameIdPolicy::Payload,
        POLICY_TAG_HEADER_DESCRIPTOR => FrameIdPolicy::HeaderDescriptor {
            fields: decode_string_list(value)?,
        },
        POLICY_TAG_CANONICAL => {
            let mut include = Vec::new();
            let mut exclude_prefixes = Vec::new();
            let mut reader = TlvReader::new(value);
            while let Some((inner_tag, inner_value)) = reader.next() {
                match inner_tag {
                    CANONICAL_TAG_INCLUDE => include = decode_string_list(inner_value)?,
                    CANONICAL_TAG_EXCLUDE_PREFIXES => {
                        exclude_prefixes = decode_string_list(inner_value)?;
                    }
                    _ => {}
                }
            }
            if !reader.is_exhausted() {
                return None;
            }
            FrameIdPolicy::Canonical {
                include,
                exclude_prefixes,
            }
        }
        POLICY_TAG_EXTERNAL => FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
        POLICY_TAG_KEY => FrameIdPolicy::Key,
        _ => return None,
    })
}

fn encode_record_id_rule(rule: &RecordIdentityRule) -> Vec<u8> {
    let mut out = Vec::new();
    push_tlv(
        &mut out,
        IDENTITY_TAG_DIGEST_ALGORITHM,
        &[rule.digest_algorithm.id()],
    );
    push_tlv(
        &mut out,
        IDENTITY_TAG_POLICY,
        &encode_frame_id_policy(&rule.policy),
    );
    out
}

fn decode_record_id_rule(bytes: &[u8]) -> Option<RecordIdentityRule> {
    let mut rule = RecordIdentityRule {
        policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
        digest_algorithm: DigestAlgorithm::Sha256,
    };
    let mut has_digest_algorithm = false;
    let mut has_policy = false;
    let mut reader = TlvReader::new(bytes);
    while let Some((tag, value)) = reader.next() {
        match tag {
            IDENTITY_TAG_DIGEST_ALGORITHM => {
                rule.digest_algorithm = DigestAlgorithm::from_id(*value.first()?);
                has_digest_algorithm = true;
            }
            IDENTITY_TAG_POLICY => {
                rule.policy = decode_frame_id_policy(value)?;
                has_policy = true;
            }
            _ => {}
        }
    }
    (reader.is_exhausted() && has_digest_algorithm && has_policy).then_some(rule)
}

fn encode_payload(payload: &PayloadPolicy) -> Vec<u8> {
    match payload {
        PayloadPolicy::Source => vec![0x00],
        PayloadPolicy::Projection { include } => {
            let mut out = vec![0x01];
            out.extend_from_slice(&encode_string_list(include));
            out
        }
    }
}

fn decode_payload(bytes: &[u8]) -> Option<PayloadPolicy> {
    match bytes.split_first()? {
        (0x00, []) => Some(PayloadPolicy::Source),
        (0x01, rest) => Some(PayloadPolicy::Projection {
            include: decode_string_list(rest)?,
        }),
        _ => None,
    }
}

impl CollectionMetadata {
    /// Verify whether this metadata's canonical ID and member namespace reproduce the
    /// given collection ID under the canonical derivation contract.
    ///
    /// Recomputes the collection ID using [`try_derive_collection_id`]. If the
    /// metadata carries an unrecognized member namespace, this returns `false`.
    #[must_use]
    pub fn verify_collection_id(&self, collection_id: &[u8; 16]) -> bool {
        match try_derive_collection_id(self.member_namespace, &self.collection_canonical_id) {
            Some(derived) => derived == *collection_id,
            None => false,
        }
    }

    /// Check whether an existing stored metadata record conflicts in logical identity
    /// with `self` (i.e. a 128-bit collection truncation collision where the stored
    /// group canonical ID or member namespace disagrees).
    #[must_use]
    pub fn identity_collides_with(&self, other: &Self) -> bool {
        self.collection_canonical_id != other.collection_canonical_id
            || self.member_namespace != other.member_namespace
    }

    /// Validate collision between `self` (stored metadata) and `requested` metadata for a given
    /// `collection_id`.
    ///
    /// Compares:
    /// 1. `self.collection_canonical_id` (stored group canonical ID) vs `requested.collection_canonical_id`
    /// 2. `self.member_namespace` (stored member namespace) vs `requested.member_namespace`
    ///
    /// If either differs, this is a 128-bit collection truncation collision.
    /// It recomputes the full 256-bit group/member identity for both stored and requested
    /// identities and verifies that both reproduce the 128-bit `collection_id`, proving a true
    /// truncation collision.
    ///
    /// # Errors
    /// - Returns `StorageError::Internal` if requested or stored metadata does not reproduce `collection_id`.
    /// - Returns `StorageError::Collision` on true 128-bit truncation collision.
    pub fn validate_identity_collision(
        &self,
        requested: &Self,
        collection_id: &[u8; 16],
    ) -> Result<(), crate::storage::StorageError> {
        self.validate_identity_collision_with(
            requested,
            collection_id,
            try_derive_collection_full_id,
        )
    }

    pub(crate) fn validate_identity_collision_with(
        &self,
        requested: &Self,
        collection_id: &[u8; 16],
        derive: impl Fn(Option<[u8; 4]>, &[u8]) -> Option<[u8; 32]>,
    ) -> Result<(), crate::storage::StorageError> {
        if self.collection_canonical_id == requested.collection_canonical_id
            && self.member_namespace == requested.member_namespace
        {
            return Ok(());
        }

        let stored_full = derive(self.member_namespace, &self.collection_canonical_id);
        if stored_full.as_ref().map(|id| &id[..16]) != Some(&collection_id[..]) {
            return Err(crate::storage::StorageError::Internal(
                "stored collection metadata does not reproduce collection id".to_owned(),
            ));
        }

        let requested_full = derive(
            requested.member_namespace,
            &requested.collection_canonical_id,
        );
        if requested_full.as_ref().map(|id| &id[..16]) != Some(&collection_id[..]) {
            return Err(crate::storage::StorageError::Internal(
                "requested collection metadata does not reproduce collection id".to_owned(),
            ));
        }

        Err(crate::storage::StorageError::Collision(format!(
            "collection identity truncation collision: stored (group={:?}, ns={:?}) disagrees with requested (group={:?}, ns={:?})",
            String::from_utf8_lossy(&self.collection_canonical_id),
            self.member_namespace.map(|ns| String::from_utf8_lossy(&ns).into_owned()),
            String::from_utf8_lossy(&requested.collection_canonical_id),
            requested.member_namespace.map(|ns| String::from_utf8_lossy(&ns).into_owned()),
        )))
    }

    /// Encode this record as a TLV block.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(ns) = &self.member_namespace {
            push_tlv(&mut out, META_TAG_MEMBER_NAMESPACE, ns);
        }
        push_tlv(
            &mut out,
            META_TAG_COLLECTION_CANONICAL_ID,
            &self.collection_canonical_id,
        );
        push_tlv(
            &mut out,
            META_TAG_RECORD_ID_RULE,
            &encode_record_id_rule(&self.record_id_rule),
        );
        push_tlv(&mut out, META_TAG_PAYLOAD, &encode_payload(&self.payload));
        if let Some(extension) = &self.extension {
            push_tlv(&mut out, META_TAG_EXTENSION, extension);
        }
        if let Some(role) = &self.role {
            push_tlv(&mut out, META_TAG_ROLE, role.as_bytes());
        }
        if let Some(schema) = &self.schema {
            push_tlv(&mut out, META_TAG_SCHEMA, schema.as_bytes());
        }
        out
    }

    /// Decode a TLV block. Unrecognized tags are ignored: this record is
    /// write-once and never re-encoded, so there is nothing to preserve them
    /// for.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut meta = Self {
            member_namespace: None,
            collection_canonical_id: Vec::new(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: None,
            schema: None,
        };
        let mut has_canonical_id = false;
        let mut has_record_id_rule = false;
        let mut reader = TlvReader::new(bytes);
        while let Some((tag, value)) = reader.next() {
            match tag {
                META_TAG_MEMBER_NAMESPACE => {
                    let ns: [u8; 4] = value.try_into().ok()?;
                    namespace_bias(ns)?;
                    meta.member_namespace = Some(ns);
                }
                META_TAG_COLLECTION_CANONICAL_ID => {
                    meta.collection_canonical_id = value.to_vec();
                    has_canonical_id = true;
                }
                META_TAG_RECORD_ID_RULE => {
                    meta.record_id_rule = decode_record_id_rule(value)?;
                    has_record_id_rule = true;
                }
                META_TAG_PAYLOAD => meta.payload = decode_payload(value)?,
                META_TAG_EXTENSION => meta.extension = Some(value.to_vec()),
                META_TAG_ROLE => {
                    meta.role = Some(std::str::from_utf8(value).ok()?.to_owned());
                }
                META_TAG_SCHEMA => {
                    meta.schema = Some(std::str::from_utf8(value).ok()?.to_owned());
                }
                _ => {}
            }
        }
        (reader.is_exhausted()
            && has_canonical_id
            && !meta.collection_canonical_id.is_empty()
            && has_record_id_rule)
            .then_some(meta)
    }
}

/// Generic identity rule for an application record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordIdentityRule {
    /// Which bytes the frame's logical identity is derived from.
    pub policy: FrameIdPolicy,
    /// Hash function used to derive the 256-bit logical identity.
    pub digest_algorithm: DigestAlgorithm,
}

/// Generic rule used to assign a record to a collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionKeyRule {
    /// RFC 6901 pointer to the source-level collection key.
    pub pointer: String,
    /// Optional, template-opt-in 4-byte member namespace tag mixed into the
    /// collection-id derivation (see [`derive_collection_id`]).
    pub member_namespace: Option<[u8; 4]>,
    /// RFC 6901 pointer to the user-facing collection identifier.
    pub display_id_pointer: String,
}

impl CollectionKeyRule {}

/// The template's establishment (genesis) rule: which source record defines a
/// collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstablishmentRule {
    /// Application-defined selector for the establishment record, e.g. a
    /// Matrix `m.room.create` event. A collection has exactly one
    /// establishment; there is no optional or multi-record cardinality.
    pub selector: String,
}

/// Format-neutral, executable description of a collection template.
///
/// Protocol rules such as Matrix redaction and authorization deliberately do
/// not belong here. They are consumers of this base model and are represented
/// by a namespaced template extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionTemplate {
    /// Application-defined template name.
    pub name: String,
    /// Application-defined collection kind; opaque to storage.
    pub collection_kind: String,
    /// Stable record identity and internal node-key derivation rule.
    pub record_id_rule: RecordIdentityRule,
    /// Source payload retention rule.
    pub payload: PayloadPolicy,
    /// Collection membership and internal collection-key derivation rule.
    pub collection_key: CollectionKeyRule,
    /// Establishment (genesis) rule, when the collection has a defining record.
    pub establishment: Option<EstablishmentRule>,
}

#[cfg(test)]
#[path = "test_template.rs"]
mod tests;
