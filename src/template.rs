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
/// [`FrameIdPolicy::ExternalCanonicalIdToCrosscheck`] and must be supplied by the caller instead.
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
        FrameIdPolicy::ExternalCanonicalIdToCrosscheck => return None,
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

/// Pool namespace discriminator for core-internal collections (auxiliary
/// indexes) that are not owned by one of the protocol pools.
pub const POOL_DST_INTERNAL: [u8; 4] = *b"INTL";

/// Derive a collection's 128-bit **logical** id from its pool namespace
/// discriminator and **canonical** id (the caller-defined external key, e.g.
/// `b"!room:server"`).
///
/// Naming: a `*_canonical_id` is the caller-defined external identity; a
/// `*_logical_id` is the engine-derived, truncated routing hash. The result here
/// is a **router, not an identity**: it is a truncated digest, so two distinct
/// canonical ids can collide in 128 bits. The genesis record stores the full
/// `collection_canonical_id`, and a reader recomputes this before trusting the
/// id — a bare 128-bit id is never sufficient to resolve a collision.
///
/// `pool_dst` is an optional, template-opt-in domain-separation tag (4 bytes).
/// When `Some`, it is mixed into the pre-image so the same canonical id in
/// different namespaces (e.g. `!room` in the `EventDag` pool versus the State
/// pool) derives different logical ids. When `None` the derivation is exactly
/// `BLAKE3(collection_canonical_id)` with no tag, so callers must not assume
/// collection logical ids are globally unique across pools.
///
/// # Panics
/// Never in practice: a 16-byte prefix of a 32-byte digest always converts.
#[must_use]
pub fn derive_collection_id(pool_dst: Option<[u8; 4]>, collection_canonical_id: &[u8]) -> [u8; 16] {
    // Stream directly into the hasher — no temporary concatenation buffer on
    // this hot path.
    let mut hasher = DigestAlgorithm::Blake3.hasher();
    if let Some(dst) = pool_dst {
        hasher.update(&dst);
    }
    hasher.update(collection_canonical_id);
    hasher.finalize()[..16]
        .try_into()
        .expect("16-byte prefix of a 32-byte digest")
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
    /// Optional, template-opt-in domain-separation tag hashed into
    /// `collection_logical_id` (see [`derive_collection_id`]). `None` means the
    /// derivation is `BLAKE3(collection_canonical_id)`.
    pub pool_dst: Option<[u8; 4]>,
    /// The collection's **canonical** id — the caller-defined external key
    /// (e.g. `!room:server`), canonicalized. Retained so a reader can recompute
    /// [`derive_collection_id`] and verify the logical id rather than trusting a
    /// bare 128-bit routing hash.
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
}

/// `CollectionMetadata` TLV tags.
const META_TAG_POOL_DST: u8 = 0x01;
const META_TAG_COLLECTION_CANONICAL_ID: u8 = 0x02;
const META_TAG_RECORD_ID_RULE: u8 = 0x03;
const META_TAG_PAYLOAD: u8 = 0x04;
const META_TAG_EXTENSION: u8 = 0x05;

/// `RecordIdentityRule` nested tags.
const IDENTITY_TAG_DIGEST_ALGORITHM: u8 = 0x01;
const IDENTITY_TAG_POLICY: u8 = 0x02;

/// [`FrameIdPolicy`] nested tags.
const POLICY_TAG_POINTER: u8 = 0x01;
const POLICY_TAG_PAYLOAD: u8 = 0x02;
const POLICY_TAG_HEADER_DESCRIPTOR: u8 = 0x03;
const POLICY_TAG_CANONICAL: u8 = 0x04;
const POLICY_TAG_EXTERNAL: u8 = 0x05;

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
    /// Encode this record as a TLV block.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(dst) = &self.pool_dst {
            push_tlv(&mut out, META_TAG_POOL_DST, dst);
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
        out
    }

    /// Decode a TLV block. Unrecognized tags are ignored: this record is
    /// write-once and never re-encoded, so there is nothing to preserve them
    /// for.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut meta = Self {
            pool_dst: None,
            collection_canonical_id: Vec::new(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: None,
        };
        let mut has_canonical_id = false;
        let mut has_record_id_rule = false;
        let mut reader = TlvReader::new(bytes);
        while let Some((tag, value)) = reader.next() {
            match tag {
                META_TAG_POOL_DST => meta.pool_dst = Some(value.try_into().ok()?),
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
    /// Optional, template-opt-in 4-byte domain-separation tag mixed into the
    /// collection-id derivation (see [`derive_collection_id`]). `None` derives
    /// `BLAKE3(collection_canonical_id)`; a Matrix room event collection opts in
    /// with the `EventDag` pool's DST.
    pub pool_dst: Option<[u8; 4]>,
    /// RFC 6901 pointer to the user-facing collection identifier.
    pub display_id_pointer: String,
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
                pool_dst: Some(POOL_DST_INTERNAL),
                display_id_pointer: "/notebook".into(),
            },
            establishment: None,
        };

        assert_eq!(
            COLLECTION_TEMPLATE_FORMAT_V1,
            "mtxdb.collection-template/v1"
        );
        assert_eq!(template.payload, PayloadPolicy::Source);
        assert_eq!(template.collection_key.display_id_pointer, "/notebook");
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
        assert_eq!(
            derive_collection_id(event_dag, room),
            derive_collection_id(event_dag, room)
        );
        // A different namespace tag yields a different id for the same key.
        assert_ne!(
            derive_collection_id(event_dag, room),
            derive_collection_id(state, room)
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
    }

    #[test]
    fn collection_metadata_round_trips_through_tlv() {
        let meta = CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: Some(br#"{"ext":"matrix.room","fmt":1,"room_version":"10"}"#.to_vec()),
        };
        assert_eq!(CollectionMetadata::decode(&meta.encode()).unwrap(), meta);
    }

    fn sample_metadata() -> CollectionMetadata {
        CollectionMetadata {
            pool_dst: Some(*b"EVNT"),
            collection_canonical_id: b"!room:matrix.org".to_vec(),
            record_id_rule: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            extension: Some(b"ext".to_vec()),
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
}
