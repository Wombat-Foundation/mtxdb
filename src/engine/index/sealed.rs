//! Paged primary-index locators for sealed packs.
//!
//! The resident [`crate::index::LossyIndex`] keeps every live record's locator
//! in a hash table. That is the dominant steady-state memory cost. A sealed
//! pack is immutable, so its locators never change and a [`MappedPackIndex`]
//! can answer point lookups directly from a memory-mapped, sorted sidecar.
//!
//! This module is the per-collection view over those sidecars: an ordered set
//! of sealed packs that resolves `(collection, logical_id)` to the newest
//! version recorded among them, *including* a removal version. It deliberately
//! does not touch the filesystem; callers supply an already-opened sidecar.
//!
//! # Ordering
//!
//! Packs are ordered by `creation_seq`, the same total order the open-time
//! frame scan uses. A lookup walks newest to oldest and stops at the first
//! pack that records the key: a removal in a newer pack shadows a live frame
//! in an older one. Within one pack the sidecar has already collapsed
//! duplicate ids to the greatest offset, so it is authoritative for that pack.
//!
//! # Not a replacement for the resident index
//!
//! Unsealed packs still accept appends, so their locators live in the resident
//! [`crate::index::LossyIndex`] until the pack is sealed and its sidecar is
//! published. A caller must consult the resident index first and this set
//! second. A miss in both is a true miss only when every pack is sealed and
//! accounted for; otherwise the caller must fall back to a pack scan.

use std::sync::Arc;

use crate::packfile::PackId;
use crate::storage::NodeId;

use super::packidx::{MappedPackIndex, PackIndexLocator};

/// A sidecar that can answer locator lookups for one sealed pack.
///
/// Implemented for both [`MappedPackIndex`] and the owned
/// [`super::packidx::PackIndex`] so callers can use the representation their
/// open path already built. Tests use a small in-memory implementation.
pub trait SealedIndex: Send + Sync {
    /// The newest version recorded for `logical_id`, or `None` if this pack
    /// does not hold the key at all.
    fn locator(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<PackIndexLocator>;

    /// Visit one collection's entries. Order is the sidecar's (logical-id
    /// order); callers that need newest-wins must walk packs newest first.
    fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        visit: &mut dyn FnMut(NodeId, PackIndexLocator),
    );

    /// File-backed bytes this sidecar maps.
    fn mapped_bytes(&self) -> u64;

    /// Unique locators recorded for one collection.
    fn collection_entry_count(&self, collection_id: &[u8; 16]) -> Option<u64>;
}

impl SealedIndex for MappedPackIndex {
    fn locator(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<PackIndexLocator> {
        self.get_locator(collection_id, logical_id)
    }

    fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        visit: &mut dyn FnMut(NodeId, PackIndexLocator),
    ) {
        self.for_each_entry_in_collection(collection_id, visit);
    }

    fn mapped_bytes(&self) -> u64 {
        MappedPackIndex::mapped_file_bytes(self)
    }

    fn collection_entry_count(&self, collection_id: &[u8; 16]) -> Option<u64> {
        MappedPackIndex::collection_entry_count(self, collection_id)
    }
}

impl SealedIndex for super::packidx::PackIndex {
    fn locator(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<PackIndexLocator> {
        self.get_locator(collection_id, logical_id)
    }

    fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        visit: &mut dyn FnMut(NodeId, PackIndexLocator),
    ) {
        self.for_each_entry_in_collection(collection_id, visit);
    }

    fn mapped_bytes(&self) -> u64 {
        super::packidx::PackIndex::mapped_file_bytes(self)
    }

    fn collection_entry_count(&self, collection_id: &[u8; 16]) -> Option<u64> {
        super::packidx::PackIndex::collection_entry_count(self, collection_id)
    }
}

/// One sealed pack's sidecar plus the shard slot its locators resolve in.
#[derive(Clone)]
pub struct SealedPack {
    /// Shard slot the pack occupied when sealed. A locator's `(slot, offset)`
    /// pair is resolved through the pool, exactly like an index candidate.
    pub slot: u16,
    /// Stable pack identity, used to replace or retire the right entry.
    pub pack_id: PackId,
    /// Creation order; strictly increasing for packs that can hold competing
    /// versions of the same key.
    pub creation_seq: u64,
    /// The mapped sorted locator index.
    pub index: Arc<dyn SealedIndex>,
}

/// A locator resolved from a sealed pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SealedHit {
    /// Shard slot holding the frame.
    pub slot: u16,
    /// Pack the locator came from.
    pub pack_id: PackId,
    /// Creation order of that pack.
    pub creation_seq: u64,
    /// The frame or removal version.
    pub locator: PackIndexLocator,
}

impl SealedHit {
    /// Byte offset of the frame named by the locator.
    #[must_use]
    pub fn offset(self) -> u64 {
        match self.locator {
            PackIndexLocator::Frame(offset) | PackIndexLocator::Removed(offset) => {
                u64::from(offset)
            }
        }
    }

    /// Whether this locator is a removal version.
    #[must_use]
    pub fn removed(self) -> bool {
        matches!(self.locator, PackIndexLocator::Removed(_))
    }
}

/// Per-collection ordered set of sealed-pack locator indexes.
///
/// Clone is cheap: packs are `Arc`s and the vector only holds metadata. The
/// set is immutable while published; callers build a new set and swap it in
/// with the rest of the collection generation.
#[derive(Clone, Default)]
pub struct SealedLocators {
    /// Sorted by `(creation_seq, pack_id)`, oldest first.
    packs: Vec<SealedPack>,
}

impl SealedLocators {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace one sealed pack's sidecar.
    ///
    /// Replacing the same `pack_id` is idempotent (for example when an open is
    /// retried); it keeps the set sorted and never duplicates a pack.
    pub fn insert(&mut self, pack: SealedPack) {
        let key = (pack.creation_seq, pack.pack_id);
        if let Some(existing) = self
            .packs
            .iter_mut()
            .find(|existing| existing.pack_id == pack.pack_id)
        {
            *existing = pack;
            // A replacement keeps the same pack identity in practice, but a
            // changed creation sequence must not leave the vector unsorted.
            self.packs
                .sort_by_key(|pack| (pack.creation_seq, pack.pack_id));
            return;
        }
        // Open scans insert packs oldest-first, so the common path is an
        // append. Avoid re-sorting the whole vector per pack: that turned a
        // 1,000-pack open into O(packs^2 log packs) comparisons.
        if self
            .packs
            .last()
            .is_none_or(|last| (last.creation_seq, last.pack_id) < key)
        {
            self.packs.push(pack);
            return;
        }
        let position = self
            .packs
            .partition_point(|existing| (existing.creation_seq, existing.pack_id) < key);
        self.packs.insert(position, pack);
    }

    /// Whether the set already serves `pack_id`.
    #[must_use]
    pub fn contains_pack(&self, pack_id: PackId) -> bool {
        self.packs.iter().any(|pack| pack.pack_id == pack_id)
    }

    /// Drop a pack that has been retired or repacked away.
    ///
    /// A stale locator must never answer a read, so retirement has to be
    /// explicit. Returns whether a pack was removed.
    pub fn remove_pack(&mut self, pack_id: PackId) -> bool {
        let before = self.packs.len();
        self.packs.retain(|pack| pack.pack_id != pack_id);
        self.packs.len() != before
    }

    /// Number of sealed packs in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.packs.len()
    }

    /// Whether the set holds no sealed packs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.packs.is_empty()
    }

    /// The sealed packs, oldest first.
    #[must_use]
    pub fn packs(&self) -> &[SealedPack] {
        &self.packs
    }

    /// Total bytes mapped by the sidecars.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        self.packs.iter().fold(0_u64, |total, pack| {
            total.saturating_add(pack.index.mapped_bytes())
        })
    }

    /// Whether any sealed pack in the set occupies `slot`.
    #[must_use]
    pub fn contains_slot(&self, slot: u16) -> bool {
        self.packs.iter().any(|pack| pack.slot == slot)
    }

    /// Unique locators recorded for one collection across every sealed pack.
    #[must_use]
    pub fn entry_count_in_collection(&self, collection_id: &[u8; 16]) -> u64 {
        self.packs.iter().fold(0_u64, |total, pack| {
            total.saturating_add(
                pack.index
                    .collection_entry_count(collection_id)
                    .unwrap_or(0),
            )
        })
    }

    /// Locator counts per shard slot for one collection, derived from each
    /// sidecar's directory entry count (the same number a slot walk would
    /// produce, because the sidecar has one entry per unique id).
    #[must_use]
    pub fn slot_counts_in_collection(
        &self,
        collection_id: &[u8; 16],
    ) -> std::collections::HashMap<u16, u64> {
        let mut counts = std::collections::HashMap::new();
        for pack in &self.packs {
            if let Some(count) = pack.index.collection_entry_count(collection_id) {
                if count > 0 {
                    let entry = counts.entry(pack.slot).or_insert(0u64);
                    *entry = entry.saturating_add(count);
                }
            }
        }
        counts
    }

    /// Resolve a key to the newest sealed version.
    ///
    /// Packs are walked newest to oldest and the first hit wins. A removal hit
    /// is returned as such; the caller must treat it as authoritative and stop
    /// falling back to older packs.
    #[must_use]
    pub fn lookup(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<SealedHit> {
        self.packs.iter().rev().find_map(|pack| {
            pack.index
                .locator(collection_id, logical_id)
                .map(|locator| SealedHit {
                    slot: pack.slot,
                    pack_id: pack.pack_id,
                    creation_seq: pack.creation_seq,
                    locator,
                })
        })
    }

    /// Visit one collection's entries, newest pack first.
    ///
    /// Entries from different packs are yielded with the newer pack's version
    /// first, so a caller that keeps the first occurrence of each id gets
    /// newest-wins semantics without a second sort. Duplicate ids across packs
    /// are yielded more than once; deduplication is the caller's job.
    pub fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        mut visit: impl FnMut(SealedHit, NodeId),
    ) {
        for pack in self.packs.iter().rev() {
            pack.index
                .for_each_in_collection(collection_id, &mut |logical_id, locator| {
                    visit(
                        SealedHit {
                            slot: pack.slot,
                            pack_id: pack.pack_id,
                            creation_seq: pack.creation_seq,
                            locator,
                        },
                        logical_id,
                    );
                });
        }
    }
}

#[cfg(test)]
#[path = "test_sealed.rs"]
mod tests;
