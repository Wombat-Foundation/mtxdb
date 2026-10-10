//! Domain-tagged `u32` sets with rezzy bitmaps and generic set algebra.
//!
//! A [`BitmapSet`] is a compact set of `u32` **ordinals** held entirely in
//! memory. The ordinals come from a caller-owned ordinal space — typically a
//! short-id scope, which maps opaque key bytes (event ids, `NodeId`s, user ids)
//! to dense `u32` ids. A [`DomainTag`] names that space, and is carried both
//! inside the set and in its serialized header: combining two sets whose tags
//! differ is a checked error, so bitmaps from different scopes can never be
//! merged by accident.
//!
//! Ordinal scopes are local by construction. A room-local scope (one room's
//! event ids) is the common case for auth-chain closures. Intersecting across
//! collections instead needs a *shared* ordinal space (for example one global
//! user or `NodeId` scope) so both sides allocate from the same numbers. The tag
//! does not choose that scope for you; it only makes a mismatch loud.
//!
//! Storage uses rezzy's dependency-free compressed bitmap
//! representation. A set that is small and rarely intersected is still
//! frequently better as an ordinary sorted `Vec<u32>`, and the caller decides
//! which representation to use.
//!
//! # Wire format
//!
//! ```text
//! [magic:4 = "BMPS"][version:1][domain:16][rezzy RBMP payload...]
//! ```
//!
//! The magic, version and domain tag are this crate's; the payload is
//! rezzy's canonical `RBMP` format. Nothing here does I/O or transactions:
//! persistence is the caller's, layered on the existing opaque-blob stores.

use rezzy_recon::bitmap::Bitmap as RezzyBitmap;

use crate::layout::ShardType;
use crate::storage::{DigestAlgorithm, StorageError};

/// Magic of an encoded [`BitmapSet`].
pub const BITMAP_SET_MAGIC: [u8; 4] = *b"BMPS";
/// Wire version of an encoded [`BitmapSet`].
pub const BITMAP_SET_FORMAT_VERSION: u8 = 1;

const DOMAIN_LEN: usize = 16;
// magic(4) + version(1) + domain(16).
const HEADER_LEN: usize = 21;
// Label hashed by [`DomainTag::for_scope`]; not part of the wire format.
const SCOPE_DOMAIN_LABEL: &[u8] = b"mtxdb/bitmap-set-domain/v1";

/// Names the ordinal universe a [`BitmapSet`]'s ids were allocated from.
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

/// A domain-tagged set of `u32` ordinals, backed by a rezzy bitmap.
#[derive(Debug, Clone, PartialEq)]
pub struct BitmapSet {
    domain: DomainTag,
    bitmap: RezzyBitmap,
}

impl BitmapSet {
    /// An empty set in `domain`.
    #[must_use]
    pub fn new(domain: DomainTag) -> Self {
        Self {
            domain,
            bitmap: RezzyBitmap::new(),
        }
    }

    /// Build a set from `values`, de-duplicating.
    #[must_use]
    pub fn from_values<I>(domain: DomainTag, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        Self {
            domain,
            bitmap: values.into_iter().collect(),
        }
    }

    /// The ordinal domain this set's ids belong to.
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Number of ids in the set.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.bitmap.len()
    }

    /// Whether the set holds no ids.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    /// Whether `value` is in the set.
    #[must_use]
    pub fn contains(&self, value: u32) -> bool {
        self.bitmap.contains(value)
    }

    /// Add `value`; returns whether it was newly inserted.
    pub fn insert(&mut self, value: u32) -> bool {
        self.bitmap.insert(value)
    }

    /// Remove `value`; returns whether it was present.
    pub fn remove(&mut self, value: u32) -> bool {
        self.bitmap.remove(value)
    }

    /// Ascending iteration over the ids.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bitmap.iter()
    }

    /// Whether every id of `self` is also in `other`.
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] if the domains differ.
    pub fn is_subset(&self, other: &Self) -> Result<bool, StorageError> {
        self.check_domain(other)?;
        Ok(self.bitmap.is_subset(&other.bitmap))
    }

    /// Whether `self` and `other` share no id.
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] if the domains differ.
    pub fn is_disjoint(&self, other: &Self) -> Result<bool, StorageError> {
        self.check_domain(other)?;
        Ok(self.bitmap.is_disjoint(&other.bitmap))
    }

    /// The union of `self` and `other`.
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] if the domains differ.
    pub fn union(&self, other: &Self) -> Result<Self, StorageError> {
        self.check_domain(other)?;
        Ok(Self {
            domain: self.domain,
            bitmap: &self.bitmap | &other.bitmap,
        })
    }

    /// The intersection of `self` and `other`.
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] if the domains differ.
    pub fn intersection(&self, other: &Self) -> Result<Self, StorageError> {
        self.check_domain(other)?;
        Ok(Self {
            domain: self.domain,
            bitmap: &self.bitmap & &other.bitmap,
        })
    }

    /// The ids of `self` that are not in `other`.
    ///
    /// # Errors
    /// Returns [`StorageError::Collision`] if the domains differ.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "`-` is bitmap set difference, not integer arithmetic"
    )]
    pub fn difference(&self, other: &Self) -> Result<Self, StorageError> {
        self.check_domain(other)?;
        let mut bitmap = self.bitmap.clone();
        bitmap -= &other.bitmap;
        Ok(Self {
            domain: self.domain,
            bitmap,
        })
    }

    /// Serialize this set (header plus rezzy's canonical bitmap payload).
    ///
    /// # Errors
    /// Returns the underlying bitmap encoding error if the payload cannot be
    /// serialized.
    pub fn encode(&self) -> Result<Vec<u8>, StorageError> {
        let payload = self.bitmap.encode();
        let capacity = HEADER_LEN.saturating_add(payload.len());
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(&BITMAP_SET_MAGIC);
        out.push(BITMAP_SET_FORMAT_VERSION);
        out.extend_from_slice(self.domain.as_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Parse a set produced by [`BitmapSet::encode`], keeping its domain tag.
    ///
    /// # Errors
    /// Returns [`StorageError::Corrupt`] if `bytes` is truncated, has a bad
    /// magic or version, or is not a valid rezzy bitmap.
    pub fn decode(bytes: &[u8]) -> Result<Self, StorageError> {
        if bytes.len() < HEADER_LEN
            || bytes[..BITMAP_SET_MAGIC.len()] != BITMAP_SET_MAGIC
            || bytes[BITMAP_SET_MAGIC.len()] != BITMAP_SET_FORMAT_VERSION
        {
            return Err(StorageError::Corrupt("bitmap-set header".to_owned()));
        }
        let mut domain = [0u8; DOMAIN_LEN];
        domain.copy_from_slice(&bytes[5..HEADER_LEN]);
        let bitmap = RezzyBitmap::decode(&bytes[HEADER_LEN..])
            .map_err(|error| StorageError::Corrupt(format!("bitmap-set payload: {error}")))?;
        Ok(Self {
            domain: DomainTag::new(domain),
            bitmap,
        })
    }

    /// Parse a set and require it to carry `domain`.
    ///
    /// Use this when reading a persisted set back for a known scope, so a blob
    /// written for some other scope is rejected rather than mixed in.
    ///
    /// # Errors
    /// As [`BitmapSet::decode`], plus [`StorageError::Collision`] if the stored
    /// domain differs from `domain`.
    pub fn decode_in_domain(bytes: &[u8], domain: DomainTag) -> Result<Self, StorageError> {
        let set = Self::decode(bytes)?;
        if set.domain == domain {
            Ok(set)
        } else {
            Err(StorageError::Collision(
                "bitmap-set domain mismatch on decode".to_owned(),
            ))
        }
    }

    fn check_domain(&self, other: &Self) -> Result<(), StorageError> {
        if self.domain == other.domain {
            Ok(())
        } else {
            Err(StorageError::Collision(
                "bitmap-set domain mismatch: ids from different ordinal scopes cannot be combined"
                    .to_owned(),
            ))
        }
    }
}
