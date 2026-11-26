//! The ordinal-universe tag shared by bitmap sets and ordered indexes.

use crate::layout::ShardType;
use crate::storage::DigestAlgorithm;

pub(crate) const DOMAIN_LEN: usize = 16;
// Label hashed by [`DomainTag::for_scope`]; not part of any wire format.
const SCOPE_DOMAIN_LABEL: &[u8] = b"mtxdb/bitmap-set-domain/v1";

/// Names the ordinal universe a `BitmapSet`'s ids were allocated from.
///
/// Two sets may only be combined (union, intersection, difference, subset,
/// disjoint) when their tags are equal. Build one with [`DomainTag::new`] for a
/// caller-defined constant, [`DomainTag::derive`] to hash an arbitrary label, or
/// [`DomainTag::for_scope`] for the `(pool, collection)` scopes used elsewhere
/// in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DomainTag([u8; DOMAIN_LEN]);

impl DomainTag {
    /// Wrap 16 explicit domain bytes.
    #[must_use]
    pub const fn new(bytes: [u8; DOMAIN_LEN]) -> Self {
        Self(bytes)
    }

    /// The raw tag bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; DOMAIN_LEN] {
        &self.0
    }

    /// Derive a tag by hashing `label` (BLAKE3, first 16 bytes).
    #[must_use]
    pub fn derive(label: &[u8]) -> Self {
        Self::from_label(label)
    }

    /// Derive the tag of a `(pool, collection)` ordinal scope.
    ///
    /// Mixing the pool tag keeps two collections that happen to share a
    /// collection id in different pools from sharing a domain.
    #[must_use]
    pub fn for_scope(pool: ShardType, collection_id: &[u8; DOMAIN_LEN]) -> Self {
        let mut label = Vec::with_capacity(SCOPE_DOMAIN_LABEL.len().saturating_add(20));
        label.extend_from_slice(SCOPE_DOMAIN_LABEL);
        label.extend_from_slice(&pool.physical_pool_tag());
        label.extend_from_slice(collection_id);
        Self::from_label(&label)
    }

    fn from_label(label: &[u8]) -> Self {
        let digest = DigestAlgorithm::Blake3.digest(label);
        let mut bytes = [0u8; DOMAIN_LEN];
        bytes.copy_from_slice(&digest[..DOMAIN_LEN]);
        Self(bytes)
    }
}
