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

    let (r0, c0) = a0.overflowing_add(b0);

    let (t1, c1a) = a1.overflowing_add(b1);
    let (r1, c1b) = t1.overflowing_add(c0 as u64);

    let (t2, c2a) = a2.overflowing_add(b2);
    let (r2, c2b) = t2.overflowing_add((c1a as u64) | (c1b as u64));

    let (t3, _c3a) = a3.overflowing_add(b3);
    let (r3, _c3b) = t3.overflowing_add((c2a as u64) | (c2b as u64));

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

/// Deprecated compatibility alias for `MEMBER_NAMESPACE_EVNT`.
#[deprecated(note = "use MEMBER_NAMESPACE_EVNT")]
pub const POOL_DST_EVNT: [u8; 4] = MEMBER_NAMESPACE_EVNT;
/// Deprecated compatibility alias for `MEMBER_NAMESPACE_PREV`.
#[deprecated(note = "use MEMBER_NAMESPACE_PREV")]
pub const POOL_DST_PREV: [u8; 4] = MEMBER_NAMESPACE_PREV;
/// Deprecated compatibility alias for `MEMBER_NAMESPACE_AUTH`.
#[deprecated(note = "use MEMBER_NAMESPACE_AUTH")]
pub const POOL_DST_AUTH: [u8; 4] = MEMBER_NAMESPACE_AUTH;
/// Deprecated compatibility alias for `MEMBER_NAMESPACE_STAT`.
#[deprecated(note = "use MEMBER_NAMESPACE_STAT")]
pub const POOL_DST_STAT: [u8; 4] = MEMBER_NAMESPACE_STAT;
/// Deprecated compatibility alias for `MEMBER_NAMESPACE_INTL`.
#[deprecated(note = "use MEMBER_NAMESPACE_INTL")]
pub const POOL_DST_INTERNAL: [u8; 4] = MEMBER_NAMESPACE_INTL;

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
/// # Panics
/// Panics if `member_namespace` is `Some(ns)` and `ns` is not a registered member
/// namespace (`EVNT`, `PREV`, `AUTH`, `STAT`, `INTL`).
/// Physical pool tags like `b"EDGE"` must **never** be passed here.
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
    /// Deprecated compatibility alias for `member_namespace`.
    #[deprecated(note = "use member_namespace")]
    #[must_use]
    pub fn pool_dst(&self) -> Option<[u8; 4]> {
        self.member_namespace
    }

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
                META_TAG_MEMBER_NAMESPACE => meta.member_namespace = Some(value.try_into().ok()?),
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

impl CollectionKeyRule {
    /// Deprecated compatibility alias for `member_namespace`.
    #[deprecated(note = "use member_namespace")]
    #[must_use]
    pub fn pool_dst(&self) -> Option<[u8; 4]> {
        self.member_namespace
    }
}

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
mod tests {
    use super::*;

    fn sha256(data: &[u8]) -> Digest32 {
        DigestAlgorithm::Sha256.digest(data)
    }

    #[test]
    fn generic_template_does_not_require_a_protocol_extension() {
        let template = CollectionTemplate {
            name: "documents".into(),
            collection_kind: "notebook".into(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/uuid".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            collection_key: CollectionKeyRule {
                pointer: "/notebook".into(),
                member_namespace: Some(MEMBER_NAMESPACE_INTL),
                display_id_pointer: "/notebook".into(),
            },
            establishment: None,
        };

        assert_eq!(template.payload, PayloadPolicy::Source);
        assert_eq!(template.collection_key.display_id_pointer, "/notebook");
    }

    #[allow(
        clippy::arithmetic_side_effects,
        clippy::cast_lossless,
        clippy::cast_possible_truncation
    )]
    fn reference_wrapping_add_le(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut carry: u16 = 0;
        let mut i = 0;
        while i < 32 {
            let sum = (a[i] as u16) + (b[i] as u16) + carry;
            result[i] = sum as u8;
            carry = sum >> 8;
            i += 1;
        }
        result
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_wrapping_add_le() {
        let zero = [0u8; 32];
        let mut one = [0u8; 32];
        one[0] = 1;

        // 1. zero + zero == zero
        assert_eq!(wrapping_add_le(zero, zero), zero);

        // 2. zero + x == x and x + zero == x
        assert_eq!(wrapping_add_le(zero, one), one);
        assert_eq!(wrapping_add_le(one, zero), one);

        // 3. max + 1 -> 0 (ripple carry across all 32 bytes / all four limbs)
        let max_bytes = [0xffu8; 32];
        assert_eq!(wrapping_add_le(max_bytes, one), zero);

        // 4. max + max -> max - 1: (2^256 - 1) + (2^256 - 1) = 2^256 - 2 mod 2^256
        let mut max_minus_one = [0xffu8; 32];
        max_minus_one[0] = 0xfe;
        assert_eq!(wrapping_add_le(max_bytes, max_bytes), max_minus_one);

        // 5. Carry across each limb:
        // Limb 0 -> 1 carry
        let mut limb0_max = [0u8; 32];
        limb0_max[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        let sum_limb0 = wrapping_add_le(limb0_max, one);
        let mut expected_limb1 = [0u8; 32];
        expected_limb1[8..16].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(sum_limb0, expected_limb1);

        // Limb 1 -> 2 carry
        let mut limb1_max = [0u8; 32];
        limb1_max[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut limb1_one = [0u8; 32];
        limb1_one[8..16].copy_from_slice(&1u64.to_le_bytes());
        let sum_limb1 = wrapping_add_le(limb1_max, limb1_one);
        let mut expected_limb2 = [0u8; 32];
        expected_limb2[16..24].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(sum_limb1, expected_limb2);

        // Limb 2 -> 3 carry
        let mut limb2_max = [0u8; 32];
        limb2_max[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut limb2_one = [0u8; 32];
        limb2_one[16..24].copy_from_slice(&1u64.to_le_bytes());
        let sum_limb2 = wrapping_add_le(limb2_max, limb2_one);
        let mut expected_limb3 = [0u8; 32];
        expected_limb3[24..32].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(sum_limb2, expected_limb3);

        // Limb 3 -> wrap to 0
        let mut limb3_max = [0u8; 32];
        limb3_max[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut limb3_one = [0u8; 32];
        limb3_one[24..32].copy_from_slice(&1u64.to_le_bytes());
        let sum_limb3 = wrapping_add_le(limb3_max, limb3_one);
        assert_eq!(sum_limb3, zero);

        // 6. Test carry-in propagation (c1b, c2b, c3b paths where a_k + b_k does not overflow, but + carry does)
        // a has limb 0 = u64::MAX, limb 1 = u64::MAX, limb 2 = 0, limb 3 = 0
        // b has limb 0 = 1, limb 1 = 0, limb 2 = 0, limb 3 = 0
        let mut a_c1b = [0u8; 32];
        a_c1b[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        a_c1b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let sum_c1b = wrapping_add_le(a_c1b, one);
        let mut expected_c1b = [0u8; 32];
        expected_c1b[16..24].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(sum_c1b, expected_c1b);

        // 7. Ripple carry across all 4 limbs: limbs 0,1,2 = MAX, b = 1
        let mut a_cascade = [0u8; 32];
        a_cascade[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        a_cascade[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        a_cascade[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        let sum_cascade = wrapping_add_le(a_cascade, one);
        let mut expected_cascade = [0u8; 32];
        expected_cascade[24..32].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(sum_cascade, expected_cascade);

        // 8. Differential testing against reference byte-wise implementation
        // Deterministic PRNG (xorshift64) to generate 500 pseudo-random 32-byte pairs
        let mut state: u64 = 0x853c_49e6_748f_ea9b;
        let mut next_u64 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..500 {
            let mut a = [0u8; 32];
            let mut b = [0u8; 32];
            for i in 0..4 {
                a[i * 8..(i + 1) * 8].copy_from_slice(&next_u64().to_le_bytes());
                b[i * 8..(i + 1) * 8].copy_from_slice(&next_u64().to_le_bytes());
            }
            let fast = wrapping_add_le(a, b);
            let reference = reference_wrapping_add_le(a, b);
            assert_eq!(
                fast, reference,
                "mismatch between 4-limb and reference byte-wise add"
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn fixed_derivation_vectors() {
        let room = b"!room:example.com";
        let expected_group_full: [u8; 32] = [
            0x2e, 0x1f, 0xe2, 0x76, 0x68, 0x5e, 0x3f, 0x82, 0x98, 0x70, 0xb5, 0x95, 0x6b, 0xa3,
            0x53, 0xec, 0x31, 0x94, 0x4f, 0x48, 0x8d, 0xe6, 0x22, 0x38, 0xdc, 0xef, 0x2b, 0xf8,
            0x31, 0xc3, 0xef, 0xc9,
        ];
        assert_eq!(derive_group_full_id(room), expected_group_full);

        // EVNT
        let expected_evnt_full: [u8; 32] = [
            0x95, 0x07, 0xbe, 0x45, 0xc4, 0x92, 0xc4, 0x38, 0x2e, 0xdb, 0x18, 0xfc, 0x07, 0x44,
            0x77, 0xcc, 0x69, 0x62, 0x6e, 0x77, 0x81, 0x20, 0xb9, 0x23, 0xac, 0x48, 0xb4, 0x53,
            0x07, 0x25, 0xa1, 0x9c,
        ];
        let expected_evnt_id: [u8; 16] = [
            0x95, 0x07, 0xbe, 0x45, 0xc4, 0x92, 0xc4, 0x38, 0x2e, 0xdb, 0x18, 0xfc, 0x07, 0x44,
            0x77, 0xcc,
        ];
        assert_eq!(
            derive_group_member_full_id(MEMBER_NAMESPACE_EVNT, room),
            Some(expected_evnt_full)
        );
        assert_eq!(
            derive_group_member_collection_id(MEMBER_NAMESPACE_EVNT, room),
            Some(expected_evnt_id)
        );

        // PREV
        let expected_prev_full: [u8; 32] = [
            0xfe, 0xa9, 0xa5, 0x8f, 0x0a, 0xb8, 0x69, 0x31, 0x7d, 0x69, 0x60, 0x93, 0xfd, 0x7c,
            0x61, 0x73, 0xb4, 0xc8, 0xcd, 0x4c, 0x35, 0xc6, 0x2a, 0x0f, 0xa9, 0x76, 0x26, 0x88,
            0xa9, 0x5c, 0x76, 0x7c,
        ];
        let expected_prev_id: [u8; 16] = [
            0xfe, 0xa9, 0xa5, 0x8f, 0x0a, 0xb8, 0x69, 0x31, 0x7d, 0x69, 0x60, 0x93, 0xfd, 0x7c,
            0x61, 0x73,
        ];
        assert_eq!(
            derive_group_member_full_id(MEMBER_NAMESPACE_PREV, room),
            Some(expected_prev_full)
        );
        assert_eq!(
            derive_group_member_collection_id(MEMBER_NAMESPACE_PREV, room),
            Some(expected_prev_id)
        );

        // AUTH
        let expected_auth_full: [u8; 32] = [
            0x48, 0x42, 0xa2, 0xef, 0xf3, 0xcf, 0x51, 0x1f, 0x9e, 0x48, 0x40, 0x63, 0x50, 0x25,
            0x3c, 0x77, 0xcc, 0xa7, 0x83, 0x76, 0x0d, 0xff, 0x67, 0x46, 0x27, 0x85, 0xfd, 0x7f,
            0x57, 0xb7, 0xa6, 0x41,
        ];
        let expected_auth_id: [u8; 16] = [
            0x48, 0x42, 0xa2, 0xef, 0xf3, 0xcf, 0x51, 0x1f, 0x9e, 0x48, 0x40, 0x63, 0x50, 0x25,
            0x3c, 0x77,
        ];
        assert_eq!(
            derive_group_member_full_id(MEMBER_NAMESPACE_AUTH, room),
            Some(expected_auth_full)
        );
        assert_eq!(
            derive_group_member_collection_id(MEMBER_NAMESPACE_AUTH, room),
            Some(expected_auth_id)
        );

        // STAT
        let expected_stat_full: [u8; 32] = [
            0xe5, 0x60, 0xc7, 0xdd, 0x2c, 0xee, 0xe5, 0xd7, 0xd3, 0x1c, 0xa9, 0xd5, 0x6e, 0x6e,
            0x9e, 0x9f, 0x77, 0x31, 0xd1, 0xf6, 0x5b, 0xc5, 0xb6, 0xd7, 0xf2, 0xad, 0x1e, 0x2e,
            0x29, 0x4d, 0xdb, 0x47,
        ];
        let expected_stat_id: [u8; 16] = [
            0xe5, 0x60, 0xc7, 0xdd, 0x2c, 0xee, 0xe5, 0xd7, 0xd3, 0x1c, 0xa9, 0xd5, 0x6e, 0x6e,
            0x9e, 0x9f,
        ];
        assert_eq!(
            derive_group_member_full_id(MEMBER_NAMESPACE_STAT, room),
            Some(expected_stat_full)
        );
        assert_eq!(
            derive_group_member_collection_id(MEMBER_NAMESPACE_STAT, room),
            Some(expected_stat_id)
        );

        // INTL
        let expected_intl_full: [u8; 32] = [
            0x44, 0xf6, 0xc1, 0xd9, 0x3c, 0x12, 0xce, 0xfa, 0xb1, 0x7b, 0x52, 0x1f, 0xc1, 0xa8,
            0xa5, 0x83, 0x06, 0xc3, 0x44, 0x8d, 0x6d, 0x08, 0xd2, 0x74, 0x74, 0x34, 0xd0, 0xae,
            0x8e, 0x27, 0x28, 0xad,
        ];
        let expected_intl_id: [u8; 16] = [
            0x44, 0xf6, 0xc1, 0xd9, 0x3c, 0x12, 0xce, 0xfa, 0xb1, 0x7b, 0x52, 0x1f, 0xc1, 0xa8,
            0xa5, 0x83,
        ];
        assert_eq!(
            derive_group_member_full_id(MEMBER_NAMESPACE_INTL, room),
            Some(expected_intl_full)
        );
        assert_eq!(
            derive_group_member_collection_id(MEMBER_NAMESPACE_INTL, room),
            Some(expected_intl_id)
        );

        // Unknown namespaces (such as physical pool tags like EDGE) MUST be rejected
        assert_eq!(namespace_bias(*b"EDGE"), None);
        assert_eq!(namespace_bias(*b"XYZW"), None);
        assert_eq!(namespace_bias(*b"    "), None);
        assert_eq!(derive_group_member_full_id(*b"EDGE", room), None);
        assert_eq!(derive_group_member_collection_id(*b"EDGE", room), None);
        assert_eq!(try_derive_collection_id(Some(*b"EDGE"), room), None);

        // All member collection IDs are strictly separated
        let all_ids = [
            expected_evnt_id,
            expected_prev_id,
            expected_auth_id,
            expected_stat_id,
            expected_intl_id,
        ];
        for i in 0..all_ids.len() {
            for j in (i + 1)..all_ids.len() {
                assert_ne!(all_ids[i], all_ids[j]);
            }
        }

        // Direct derivation from group_full_id without re-hashing
        assert_eq!(
            derive_member_full_id_from_group(MEMBER_NAMESPACE_EVNT, expected_group_full),
            Some(expected_evnt_full)
        );
        assert_eq!(
            derive_member_collection_id_from_group(MEMBER_NAMESPACE_EVNT, expected_group_full),
            Some(expected_evnt_id)
        );
        assert_eq!(
            derive_member_collection_id_from_group(MEMBER_NAMESPACE_PREV, expected_group_full),
            Some(expected_prev_id)
        );
        assert_eq!(
            derive_member_collection_id_from_group(MEMBER_NAMESPACE_AUTH, expected_group_full),
            Some(expected_auth_id)
        );
        assert_eq!(
            derive_member_collection_id_from_group(MEMBER_NAMESPACE_STAT, expected_group_full),
            Some(expected_stat_id)
        );
        assert_eq!(
            derive_member_collection_id_from_group(MEMBER_NAMESPACE_INTL, expected_group_full),
            Some(expected_intl_id)
        );
    }

    #[test]
    fn frame_digest_selects_its_input_per_policy() {
        let resolve = |pointer: &str| (pointer == "/uuid").then(|| b"pointer-bytes".to_vec());
        let input = FrameIdInput {
            payload: b"payload-bytes",
            descriptor: b"descriptor-bytes",
            canonical: Some(b"canonical-bytes"),
            resolve: &resolve,
        };

        // Each policy hashes exactly the bytes it selects.
        assert_eq!(
            frame_digest(&FrameIdPolicy::Payload, DigestAlgorithm::Sha256, &input),
            Some(sha256(b"payload-bytes"))
        );
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::HeaderDescriptor { fields: vec![] },
                DigestAlgorithm::Sha256,
                &input
            ),
            Some(sha256(b"descriptor-bytes"))
        );
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::Canonical {
                    include: vec![],
                    exclude_prefixes: vec![],
                },
                DigestAlgorithm::Sha256,
                &input
            ),
            Some(sha256(b"canonical-bytes"))
        );
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::Pointer {
                    pointer: "/uuid".into(),
                },
                DigestAlgorithm::Sha256,
                &input
            ),
            Some(sha256(b"pointer-bytes"))
        );

        // Missing inputs and externally supplied identities yield `None`.
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::Pointer {
                    pointer: "/absent".into(),
                },
                DigestAlgorithm::Sha256,
                &input
            ),
            None
        );
        let no_canonical = FrameIdInput {
            canonical: None,
            ..input
        };
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::Canonical {
                    include: vec![],
                    exclude_prefixes: vec![],
                },
                DigestAlgorithm::Sha256,
                &no_canonical
            ),
            None
        );
        assert_eq!(
            frame_digest(
                &FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
                DigestAlgorithm::Sha256,
                &input
            ),
            None
        );
    }

    #[test]
    fn record_logical_id_uses_the_first_16_digest_bytes() {
        let mut digest = [0u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::try_from(index).expect("test index fits in u8");
        }

        assert_eq!(
            record_logical_id(&digest),
            [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f,
            ]
        );
    }

    #[test]
    fn collection_id_is_deterministic_and_pool_separated() {
        let room = b"!room:matrix.org";
        let event_dag = Some(*b"EVNT");
        let state = Some(*b"STAT");
        let prev = Some(*b"PREV");
        let auth = Some(*b"AUTH");
        assert_eq!(
            derive_collection_id(event_dag, room),
            derive_collection_id(event_dag, room)
        );
        // A different namespace tag yields a different id for the same key.
        assert_ne!(
            derive_collection_id(event_dag, room),
            derive_collection_id(state, room)
        );
        assert_ne!(
            derive_collection_id(prev, room),
            derive_collection_id(auth, room)
        );
        // A different key yields a different id for the same tag.
        assert_ne!(
            derive_collection_id(event_dag, room),
            derive_collection_id(event_dag, b"!other:matrix.org")
        );
        // No tag is a valid, deterministic derivation of its own.
        assert_eq!(
            derive_collection_id(None, room),
            derive_collection_id(None, room)
        );
        assert_ne!(
            derive_collection_id(None, room),
            derive_collection_id(event_dag, room)
        );

        // Fixed vector verification:
        let group_id = derive_group_full_id(room);
        let evnt_full = derive_group_member_full_id(*b"EVNT", room).unwrap();
        assert_eq!(evnt_full, wrapping_add_le(group_id, NAMESPACE_BIAS_EVNT));
        let evnt_col = derive_group_member_collection_id(*b"EVNT", room).unwrap();
        assert_eq!(evnt_col, evnt_full[..16]);
        assert_eq!(evnt_col, derive_collection_id(Some(*b"EVNT"), room));
    }

    #[test]
    fn collection_metadata_identity_collides_with_detects_mismatches() {
        let meta1 = CollectionMetadata {
            member_namespace: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: None,
            schema: None,
        };
        let mut meta2 = meta1.clone();
        assert!(!meta1.identity_collides_with(&meta2));

        // Different canonical ID -> collision
        meta2.collection_canonical_id = b"!other:matrix.org".to_vec();
        assert!(meta1.identity_collides_with(&meta2));

        // Different namespace -> collision
        let mut meta3 = meta1.clone();
        meta3.member_namespace = Some(*b"STAT");
        assert!(meta1.identity_collides_with(&meta3));

        // Different role/schema does NOT count as identity collision
        let mut meta4 = meta1.clone();
        meta4.role = Some("event_dag".to_owned());
        assert!(!meta1.identity_collides_with(&meta4));
    }

    #[test]
    fn collection_metadata_round_trips_through_tlv() {
        let meta = CollectionMetadata {
            member_namespace: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: Some(br#"{"ext":"matrix.room","fmt":1,"room_version":"10"}"#.to_vec()),
            role: Some("event_dag".to_owned()),
            schema: Some("matrix.event.v1".to_owned()),
        };
        assert_eq!(CollectionMetadata::decode(&meta.encode()).unwrap(), meta);
        let id = derive_collection_id(meta.member_namespace, &meta.collection_canonical_id);
        assert!(meta.verify_collection_id(&id));
        assert!(!meta.verify_collection_id(&[0u8; 16]));
    }

    #[test]
    fn frame_id_policy_key_round_trips() {
        let policy = FrameIdPolicy::Key;
        let encoded = encode_frame_id_policy(&policy);
        assert_eq!(decode_frame_id_policy(&encoded).unwrap(), policy);
    }

    fn sample_metadata() -> CollectionMetadata {
        CollectionMetadata {
            member_namespace: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: Some(b"ext".to_vec()),
            role: Some("event_dag".to_owned()),
            schema: Some("matrix.event.v1".to_owned()),
        }
    }

    #[test]
    fn collection_metadata_decode_rejects_truncation_and_trailing_bytes() {
        let encoded = sample_metadata().encode();

        // Dropping the final byte leaves the last TLV length prefix promising
        // more bytes than remain.
        assert!(CollectionMetadata::decode(&encoded[..encoded.len() - 1]).is_none());
        // Truncated inside the first length prefix.
        assert!(CollectionMetadata::decode(&encoded[..3]).is_none());
        // A complete record followed by trailing bytes is not accepted.
        let mut trailing = encoded;
        trailing.push(0xAA);
        assert!(CollectionMetadata::decode(&trailing).is_none());
    }

    #[test]
    fn collection_metadata_decode_requires_mandatory_fields() {
        let rule = sample_metadata().record_id_rule;

        // No fields at all.
        assert!(CollectionMetadata::decode(&[]).is_none());

        // Canonical id present, record-id rule absent.
        let mut missing_rule = Vec::new();
        push_tlv(
            &mut missing_rule,
            META_TAG_COLLECTION_CANONICAL_ID,
            b"!room:matrix.org",
        );
        assert!(CollectionMetadata::decode(&missing_rule).is_none());

        // Record-id rule present, canonical id absent.
        let mut missing_id = Vec::new();
        push_tlv(
            &mut missing_id,
            META_TAG_RECORD_ID_RULE,
            &encode_record_id_rule(&rule),
        );
        assert!(CollectionMetadata::decode(&missing_id).is_none());

        // An empty canonical id does not count as present.
        let mut empty_id = Vec::new();
        push_tlv(&mut empty_id, META_TAG_COLLECTION_CANONICAL_ID, &[]);
        push_tlv(
            &mut empty_id,
            META_TAG_RECORD_ID_RULE,
            &encode_record_id_rule(&rule),
        );
        assert!(CollectionMetadata::decode(&empty_id).is_none());
    }

    #[test]
    fn record_id_rule_decode_requires_both_fields_and_no_trailing_bytes() {
        let rule = sample_metadata().record_id_rule;
        assert_eq!(
            decode_record_id_rule(&encode_record_id_rule(&rule)).unwrap(),
            rule
        );

        let mut only_digest = Vec::new();
        push_tlv(
            &mut only_digest,
            IDENTITY_TAG_DIGEST_ALGORITHM,
            &[DigestAlgorithm::Sha256.id()],
        );
        assert!(decode_record_id_rule(&only_digest).is_none());

        let mut trailing = encode_record_id_rule(&rule);
        trailing.push(0x00);
        assert!(decode_record_id_rule(&trailing).is_none());
    }

    #[test]
    fn frame_id_policy_decode_rejects_trailing_bytes() {
        let policy = FrameIdPolicy::Pointer {
            pointer: "/event_id".into(),
        };
        let mut encoded = encode_frame_id_policy(&policy);
        assert_eq!(decode_frame_id_policy(&encoded).unwrap(), policy);
        encoded.push(0x7F);
        assert!(decode_frame_id_policy(&encoded).is_none());
    }

    #[test]
    fn payload_decode_rejects_trailing_bytes() {
        assert_eq!(
            decode_payload(&[0x00]).unwrap(),
            PayloadPolicy::Source,
            "a lone source marker is valid"
        );
        assert!(
            decode_payload(&[0x00, 0xAA]).is_none(),
            "trailing bytes after a source marker must be rejected"
        );
        assert_eq!(
            decode_payload(&[0x01]).unwrap(),
            PayloadPolicy::Projection { include: vec![] }
        );
    }

    #[test]
    fn collection_metadata_decode_rejects_invalid_utf8_role_or_schema() {
        let meta = CollectionMetadata {
            member_namespace: Some(*b"EVNT"),
            collection_canonical_id: b"!room:example.com".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("valid_role".to_owned()),
            schema: Some("valid.schema.v1".to_owned()),
        };
        let encoded = meta.encode();
        assert!(CollectionMetadata::decode(&encoded).is_some());

        // Corrupt role TLV (tag 0x06) with invalid UTF-8:
        let mut corrupted_role = Vec::new();
        push_tlv(&mut corrupted_role, 0x01, b"EVNT");
        push_tlv(&mut corrupted_role, 0x02, b"!room:example.com");
        push_tlv(&mut corrupted_role, 0x03, &[0x06, 0x01]); // Key policy, Blake3
        push_tlv(&mut corrupted_role, 0x04, &[0x00]); // Payload source
        push_tlv(&mut corrupted_role, 0x06, &[0xFF, 0xFE, 0xFD]); // Invalid UTF-8 role
        assert!(CollectionMetadata::decode(&corrupted_role).is_none());

        // Corrupt schema TLV (tag 0x07) with invalid UTF-8:
        let mut corrupted_schema = Vec::new();
        push_tlv(&mut corrupted_schema, 0x01, b"EVNT");
        push_tlv(&mut corrupted_schema, 0x02, b"!room:example.com");
        push_tlv(&mut corrupted_schema, 0x03, &[0x06, 0x01]);
        push_tlv(&mut corrupted_schema, 0x04, &[0x00]);
        push_tlv(&mut corrupted_schema, 0x07, &[0xFF, 0xFE, 0xFD]); // Invalid UTF-8 schema
        assert!(CollectionMetadata::decode(&corrupted_schema).is_none());
    }

    #[test]
    fn validate_identity_collision_and_namespace_rejection() {
        let meta1 = CollectionMetadata {
            member_namespace: Some(MEMBER_NAMESPACE_EVNT),
            collection_canonical_id: b"!room:example.com".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Key,
                digest_algorithm: DigestAlgorithm::Blake3,
            },
            payload: PayloadPolicy::Source,
            extension: None,
            role: Some("event_dag".to_owned()),
            schema: None,
        };
        let col1_id = derive_collection_id(meta1.member_namespace, &meta1.collection_canonical_id);

        // 1. Identical metadata -> Ok(())
        let meta1_clone = meta1.clone();
        assert!(meta1
            .validate_identity_collision(&meta1_clone, &col1_id)
            .is_ok());

        // 2. Different canonical id, but requested does not reproduce col1_id -> StorageError::Internal
        let mut meta_different_canonical = meta1.clone();
        meta_different_canonical.collection_canonical_id = b"!room2:example.com".to_vec();
        let err = meta1
            .validate_identity_collision(&meta_different_canonical, &col1_id)
            .unwrap_err();
        assert!(matches!(err, crate::storage::StorageError::Internal(_)));

        // 2b. Stored metadata does not reproduce collection_id -> StorageError::Internal
        let err_stored = meta_different_canonical
            .validate_identity_collision(&meta1, &col1_id)
            .unwrap_err();
        assert!(matches!(
            err_stored,
            crate::storage::StorageError::Internal(_)
        ));

        // 3. Simulated true collision: both stored and requested reproduce the same collection_id
        // despite differing in canonical ID or member namespace -> StorageError::Collision
        let mock_collision_full = [0x55u8; 32];
        let mock_collision_id = [0x55u8; 16];
        let mock_derive = |_ns: Option<[u8; 4]>, _canon: &[u8]| Some(mock_collision_full);

        let err_collision = meta1
            .validate_identity_collision_with(
                &meta_different_canonical,
                &mock_collision_id,
                mock_derive,
            )
            .unwrap_err();
        assert!(matches!(
            err_collision,
            crate::storage::StorageError::Collision(_)
        ));

        // 4. Different member namespace with collision
        let mut meta_stat = meta1.clone();
        meta_stat.member_namespace = Some(MEMBER_NAMESPACE_STAT);
        let err_ns_collision = meta1
            .validate_identity_collision_with(&meta_stat, &mock_collision_id, mock_derive)
            .unwrap_err();
        assert!(matches!(
            err_ns_collision,
            crate::storage::StorageError::Collision(_)
        ));

        // 5. Unregistered namespace (e.g. physical pool tag EDGE)
        let mut meta_edge = meta1.clone();
        meta_edge.member_namespace = Some(*b"EDGE");
        assert!(!meta_edge.verify_collection_id(&col1_id));
        assert_eq!(
            try_derive_collection_full_id(Some(*b"EDGE"), b"!room:example.com"),
            None
        );
        assert_eq!(
            try_derive_collection_id(Some(*b"EDGE"), b"!room:example.com"),
            None
        );
    }

    #[test]
    #[should_panic(expected = "member_namespace must be a valid registered namespace")]
    fn derive_collection_id_panics_on_unknown_namespace() {
        let _ = derive_collection_id(Some(*b"EDGE"), b"!room:example.com");
    }
}
