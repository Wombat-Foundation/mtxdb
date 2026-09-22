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

/// Domain separator for the collection genesis (establishment) frame's
/// reserved node id.
pub const GENESIS_SENTINEL_DOMAIN: &[u8] = b"mtxdb:sentinel:genesis_frame";

/// The reserved node id under which a collection's genesis/establishment
/// metadata frame is stored: `SHA-256(GENESIS_SENTINEL_DOMAIN)[..16]`.
///
/// Domain-separated from every user record id, so a real frame can never
/// collide with it (asserted by a test).
pub const GENESIS_FRAME_NODE_ID: [u8; 16] = [
    0x79, 0xad, 0xa1, 0x33, 0x02, 0x69, 0x39, 0x76, 0xa1, 0x3f, 0x5c, 0xd9, 0x6c, 0xc2, 0x43, 0x22,
];

/// Current wire format of a [`CollectionMetadata`] genesis record.
pub const COLLECTION_METADATA_FORMAT_V1: u16 = 1;

/// A collection's first-class, immutable definition, written once as its
/// genesis/establishment record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionMetadata {
    /// Wire format version of this record.
    pub format_version: u16,
    /// The type discriminator used in [`derive_collection_id`].
    pub collection_type: u16,
    /// The full canonical external key — the derivation pre-image — retained so
    /// a 128-bit collection-id collision can be detected and rejected.
    pub canonical_preimage: Vec<u8>,
    /// Record identity rule for frames in this collection.
    pub record_identity: RecordIdentityRule,
    /// Source payload retention rule.
    pub payload: PayloadPolicy,
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
    fn genesis_sentinel_matches_its_domain_hash() {
        let mut hasher = DigestAlgorithm::Sha256.hasher();
        hasher.update(GENESIS_SENTINEL_DOMAIN);
        let digest = hasher.finalize();
        assert_eq!(&GENESIS_FRAME_NODE_ID[..], &digest[..16]);
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
}
