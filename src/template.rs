//! Executable policy primitives for format-specific collection templates.
//!
//! Storage remains format-agnostic. Importers bind a collection to one of
//! these policies and persist the resulting metadata alongside their index.

use std::borrow::Cow;

use crate::storage::{Digest32, DigestAlgorithm};

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

/// Domain separator for the v1 collection-id derivation.
pub const COLLECTION_ID_DOMAIN: &[u8] = b"mtxdb:collection:v1:";

/// Collection-type value meaning "unset"; never a valid collection.
pub const COLLECTION_TYPE_UNSET: u16 = 0x0000;
/// First collection-type value reserved for core-internal collections (e.g.
/// auxiliary indexes). The range `0x0001..=0x00FF` is core-owned.
pub const COLLECTION_TYPE_INTERNAL_BASE: u16 = 0x0001;
/// First collection-type value available to protocol extensions (e.g. Matrix
/// rooms). The range `0x0100..=0x7FFF` is protocol-owned.
pub const COLLECTION_TYPE_PROTOCOL_BASE: u16 = 0x0100;
/// First collection-type value available to applications. The range
/// `0x8000..=0xFFFF` is application-owned.
pub const COLLECTION_TYPE_APP_BASE: u16 = 0x8000;

/// Derive a collection's 128-bit id from its type discriminator and canonical
/// external key (e.g. `b"!room:server"`).
///
/// The result is a **router, not an identity**: it is a truncated digest, so two
/// distinct keys can collide in 128 bits. A collection's genesis record stores
/// the full `canonical_preimage`, and a reader verifies it before trusting the
/// id — a bare 128-bit id is never sufficient to resolve a collision.
///
/// # Panics
/// Never in practice: a 16-byte prefix of a 32-byte digest always converts.
#[must_use]
pub fn derive_collection_id(collection_type: u16, canonical_key: &[u8]) -> [u8; 16] {
    // Stream directly into the hasher — no temporary concatenation buffer on
    // this hot path.
    let mut hasher = DigestAlgorithm::Sha256.hasher();
    hasher.update(COLLECTION_ID_DOMAIN);
    hasher.update(&collection_type.to_be_bytes());
    hasher.update(canonical_key);
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

/// Current wire format of a [`CollectionMetadata`] genesis record.
pub const COLLECTION_METADATA_FORMAT_V1: u16 = 1;

/// A collection's first-class, immutable definition, written once as its
/// metadata (genesis) record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionMetadata {
    /// Wire format version of this record.
    pub format_version: u16,
    /// The type discriminator used in [`derive_collection_id`].
    pub collection_type: u16,
    /// The full canonical external key — the derivation pre-image — retained so
    /// a 128-bit collection-id collision can be detected and rejected.
    pub canonical_preimage: Vec<u8>,
    /// Digest of the canonical establishment (genesis) record, if any.
    pub establishment_digest: Option<Digest32>,
    /// Record identity rule for frames in this collection.
    pub record_identity: RecordIdentityRule,
    /// Source payload retention rule.
    pub payload: PayloadPolicy,
    /// Tags this build does not interpret, preserved verbatim so a read/rewrite
    /// cycle never drops a newer writer's fields.
    pub unknown: Vec<(u8, Vec<u8>)>,
}

/// `CollectionMetadata` TLV tags.
const META_TAG_FORMAT_VERSION: u8 = 0x01;
const META_TAG_COLLECTION_TYPE: u8 = 0x02;
const META_TAG_CANONICAL_PREIMAGE: u8 = 0x03;
const META_TAG_ESTABLISHMENT_DIGEST: u8 = 0x04;
const META_TAG_RECORD_IDENTITY: u8 = 0x05;
const META_TAG_PAYLOAD: u8 = 0x06;

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
    let (tag, value) = TlvReader::new(bytes).next()?;
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
            FrameIdPolicy::Canonical {
                include,
                exclude_prefixes,
            }
        }
        POLICY_TAG_EXTERNAL => FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
        _ => return None,
    })
}

fn encode_record_identity(rule: &RecordIdentityRule) -> Vec<u8> {
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

fn decode_record_identity(bytes: &[u8]) -> Option<RecordIdentityRule> {
    let mut rule = RecordIdentityRule {
        policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
        digest_algorithm: DigestAlgorithm::Sha256,
    };
    let mut reader = TlvReader::new(bytes);
    while let Some((tag, value)) = reader.next() {
        match tag {
            IDENTITY_TAG_DIGEST_ALGORITHM => {
                rule.digest_algorithm = DigestAlgorithm::from_id(*value.first()?);
            }
            IDENTITY_TAG_POLICY => rule.policy = decode_frame_id_policy(value)?,
            _ => {}
        }
    }
    Some(rule)
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
        (0x00, _) => Some(PayloadPolicy::Source),
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
        push_tlv(
            &mut out,
            META_TAG_FORMAT_VERSION,
            &self.format_version.to_le_bytes(),
        );
        push_tlv(
            &mut out,
            META_TAG_COLLECTION_TYPE,
            &self.collection_type.to_le_bytes(),
        );
        push_tlv(
            &mut out,
            META_TAG_CANONICAL_PREIMAGE,
            &self.canonical_preimage,
        );
        if let Some(digest) = &self.establishment_digest {
            push_tlv(&mut out, META_TAG_ESTABLISHMENT_DIGEST, digest);
        }
        push_tlv(
            &mut out,
            META_TAG_RECORD_IDENTITY,
            &encode_record_identity(&self.record_identity),
        );
        push_tlv(&mut out, META_TAG_PAYLOAD, &encode_payload(&self.payload));
        for (tag, value) in &self.unknown {
            push_tlv(&mut out, *tag, value);
        }
        out
    }

    /// Decode a TLV block, preserving unrecognized tags in [`Self::unknown`].
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut meta = Self {
            format_version: 0,
            collection_type: 0,
            canonical_preimage: Vec::new(),
            establishment_digest: None,
            record_identity: RecordIdentityRule {
                policy: FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            unknown: Vec::new(),
        };
        let mut reader = TlvReader::new(bytes);
        while let Some((tag, value)) = reader.next() {
            match tag {
                META_TAG_FORMAT_VERSION => {
                    meta.format_version = u16::from_le_bytes(value.try_into().ok()?);
                }
                META_TAG_COLLECTION_TYPE => {
                    meta.collection_type = u16::from_le_bytes(value.try_into().ok()?);
                }
                META_TAG_CANONICAL_PREIMAGE => meta.canonical_preimage = value.to_vec(),
                META_TAG_ESTABLISHMENT_DIGEST => {
                    meta.establishment_digest = Some(value.try_into().ok()?);
                }
                META_TAG_RECORD_IDENTITY => meta.record_identity = decode_record_identity(value)?,
                META_TAG_PAYLOAD => meta.payload = decode_payload(value)?,
                _ => meta.unknown.push((tag, value.to_vec())),
            }
        }
        Some(meta)
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
    /// Compact 2-byte type discriminator mixed into the collection-id
    /// derivation. Replaces a duplicated type string; the value space is owned
    /// by the protocol extension (e.g. `0x0001` for a Matrix room).
    pub collection_type: u16,
    /// RFC 6901 pointer to the user-facing collection identifier.
    pub display_id_pointer: String,
}

/// The template's establishment (genesis) rule: which source record defines a
/// collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EstablishmentRule {
    /// Application-defined selector for the establishment record, e.g. a
    /// Matrix `m.room.create` event.
    pub selector: String,
    /// Expected cardinality, e.g. `exactly-one`.
    pub cardinality: String,
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
    pub record_identity: RecordIdentityRule,
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
            record_identity: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/uuid".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            collection_key: CollectionKeyRule {
                pointer: "/notebook".into(),
                collection_type: 0x0001,
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
    fn collection_id_is_deterministic_and_type_separated() {
        let room = b"!room:matrix.org";
        assert_eq!(
            derive_collection_id(0x0001, room),
            derive_collection_id(0x0001, room)
        );
        // A different type discriminator yields a different id for the same key.
        assert_ne!(
            derive_collection_id(0x0001, room),
            derive_collection_id(0x0002, room)
        );
        // A different key yields a different id for the same type.
        assert_ne!(
            derive_collection_id(0x0001, room),
            derive_collection_id(0x0001, b"!other:matrix.org")
        );
    }

    #[test]
    fn collection_metadata_round_trips_through_tlv() {
        let meta = CollectionMetadata {
            format_version: COLLECTION_METADATA_FORMAT_V1,
            collection_type: 0x0100,
            canonical_preimage: b"!room:matrix.org".to_vec(),
            establishment_digest: Some([0xAB; 32]),
            record_identity: RecordIdentityRule {
                policy: FrameIdPolicy::Pointer {
                    pointer: "/event_id".into(),
                },
                digest_algorithm: DigestAlgorithm::Sha256,
            },
            payload: PayloadPolicy::Source,
            // An unrecognized tag must survive a decode/encode cycle.
            unknown: vec![(0x7F, vec![1, 2, 3])],
        };
        assert_eq!(CollectionMetadata::decode(&meta.encode()).unwrap(), meta);
    }
}
