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
    /// Digest algorithm used to derive mtxdb's internal collection key.
    pub collection_id_algorithm: String,
    /// RFC 6901 pointer to the user-facing collection identifier.
    pub display_id_pointer: String,
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
                collection_id_algorithm: "sha2-256".into(),
                display_id_pointer: "/notebook".into(),
            },
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
}
