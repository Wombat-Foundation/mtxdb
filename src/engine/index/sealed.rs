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
//!
//! # Runs
//!
//! A [`SealedRun`] serves a contiguous range of sealed packs from one merged
//! run file instead of one sidecar per pack. Runs and unmerged sidecars are
//! walked together in pack order, newest first. A run that fails a block
//! check answers from its covered packs' sidecars instead, loaded on first
//! need; if those cannot be loaded either, the lookup fails rather than
//! reporting a miss.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::packfile::PackId;
use crate::storage::NodeId;

use super::packidx::{MappedPackIndex, PackIndexLocator};
use super::run::RunIndex;

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
        0
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

/// One pack a run covers, resolved to the pool slot it occupies now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunPack {
    /// Shard slot holding the pack.
    pub slot: u16,
    /// Stable pack identity.
    pub pack_id: PackId,
    /// Creation order of the pack.
    pub creation_seq: u64,
}

/// Loads the covered packs' sidecars when a run can no longer answer.
pub type RunFallback = Box<dyn Fn() -> Result<Vec<SealedPack>, String> + Send + Sync>;

/// A sealed lookup could not be answered: a run failed its check and the
/// covered packs' sidecars could not be loaded either. This is never a miss.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedUnavailable(pub String);

impl std::fmt::Display for SealedUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sealed locators unavailable: {}", self.0)
    }
}

impl std::error::Error for SealedUnavailable {}

/// A merged run serving a contiguous range of sealed packs.
///
/// One value is shared by every collection the run holds.
pub struct SealedRun {
    run: Arc<RunIndex>,
    /// The run's packs, indexed by run ordinal.
    packs: Vec<RunPack>,
    load_fallback: RunFallback,
    fallback: RwLock<Option<Arc<[SealedPack]>>>,
}

impl SealedRun {
    /// Wrap an opened run. `packs` must list the run's packs in ordinal order;
    /// `load_fallback` returns their sidecars, oldest first.
    ///
    /// # Errors
    /// Returns an error if `packs` does not match the run's pack table.
    pub fn new(
        run: Arc<RunIndex>,
        packs: Vec<RunPack>,
        load_fallback: RunFallback,
    ) -> Result<Self, SealedUnavailable> {
        let matches = run.packs().len() == packs.len()
            && run.packs().iter().zip(&packs).all(|(identity, pack)| {
                identity.pack_id == pack.pack_id && identity.creation_seq == pack.creation_seq
            });
        if !matches {
            return Err(SealedUnavailable(
                "run pack table does not match the pool".into(),
            ));
        }
        Ok(Self {
            run,
            packs,
            load_fallback,
            fallback: RwLock::new(None),
        })
    }

    /// The run's packs, oldest first.
    #[must_use]
    pub fn packs(&self) -> &[RunPack] {
        &self.packs
    }

    /// The underlying run.
    #[must_use]
    pub fn run(&self) -> &RunIndex {
        &self.run
    }

    fn newest_key(&self) -> (u64, PackId) {
        self.packs.last().map_or((0, PackId([0; 16])), |pack| {
            (pack.creation_seq, pack.pack_id)
        })
    }

    fn covers(&self, pack_id: PackId) -> bool {
        self.packs.iter().any(|pack| pack.pack_id == pack_id)
    }

    fn hit(&self, ordinal: u16, locator: PackIndexLocator) -> Result<SealedHit, SealedUnavailable> {
        let pack = self
            .packs
            .get(usize::from(ordinal))
            .ok_or_else(|| SealedUnavailable("run ordinal outside its pack table".into()))?;
        Ok(SealedHit {
            slot: pack.slot,
            pack_id: pack.pack_id,
            creation_seq: pack.creation_seq,
            locator,
        })
    }

    fn fallback(&self) -> Result<Arc<[SealedPack]>, SealedUnavailable> {
        if let Some(cached) = self.fallback.read().as_ref() {
            return Ok(Arc::clone(cached));
        }
        let mut guard = self.fallback.write();
        if let Some(cached) = guard.as_ref() {
            return Ok(Arc::clone(cached));
        }
        let loaded: Arc<[SealedPack]> = (self.load_fallback)().map_err(SealedUnavailable)?.into();
        *guard = Some(Arc::clone(&loaded));
        Ok(loaded)
    }

    fn lookup(
        &self,
        collection_id: &[u8; 16],
        logical_id: &NodeId,
    ) -> Result<Option<SealedHit>, SealedUnavailable> {
        match self.run.lookup(collection_id, logical_id) {
            Ok(Some(hit)) => self.hit(hit.ordinal, hit.locator).map(Some),
            Ok(None) => Ok(None),
            Err(_) => {
                let fallback = self.fallback()?;
                Ok(lookup_packs(&fallback, collection_id, logical_id))
            }
        }
    }

    fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        visit: &mut dyn FnMut(SealedHit, NodeId),
    ) -> Result<(), SealedUnavailable> {
        // Check the collection's blocks before visiting any entry, so the walk
        // streams instead of buffering against a later block failure.
        if self.run.verify_collection(collection_id).is_ok() {
            let mut failure = None;
            let walked = self
                .run
                .for_each_in_collection(collection_id, |hit, logical_id| {
                    if failure.is_none() {
                        match self.hit(hit.ordinal, hit.locator) {
                            Ok(hit) => visit(hit, logical_id),
                            Err(error) => failure = Some(error),
                        }
                    }
                });
            if let Some(error) = failure {
                return Err(error);
            }
            if walked.is_ok() {
                return Ok(());
            }
            // A concurrent reader found a damaged block elsewhere in this run
            // mid-walk. The entries already visited were verified and are the
            // newest in this range; the fallback repeats the range's versions
            // after them, which first-occurrence callers already ignore.
        }
        let fallback = self.fallback()?;
        for_each_in_packs(&fallback, collection_id, visit);
        Ok(())
    }

    fn collection_slot_counts(
        &self,
        collection_id: &[u8; 16],
    ) -> impl Iterator<Item = (u16, u64)> + '_ {
        let collection_id = *collection_id;
        let mut rows = Vec::new();
        for stats in self.run.pack_stats_in_collection(&collection_id) {
            if let Some(pack) = self.packs.get(usize::from(stats.ordinal)) {
                rows.push((pack.slot, stats.entry_count));
            }
        }
        rows.into_iter()
    }
}

fn lookup_packs(
    packs: &[SealedPack],
    collection_id: &[u8; 16],
    logical_id: &NodeId,
) -> Option<SealedHit> {
    packs
        .iter()
        .rev()
        .find_map(|pack| pack.lookup(collection_id, logical_id))
}

fn for_each_in_packs(
    packs: &[SealedPack],
    collection_id: &[u8; 16],
    visit: &mut dyn FnMut(SealedHit, NodeId),
) {
    for pack in packs.iter().rev() {
        pack.index
            .for_each_in_collection(collection_id, &mut |logical_id, locator| {
                visit(pack.hit(locator), logical_id);
            });
    }
}

impl SealedPack {
    fn key(&self) -> (u64, PackId) {
        (self.creation_seq, self.pack_id)
    }

    fn hit(&self, locator: PackIndexLocator) -> SealedHit {
        SealedHit {
            slot: self.slot,
            pack_id: self.pack_id,
            creation_seq: self.creation_seq,
            locator,
        }
    }

    fn lookup(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<SealedHit> {
        self.index
            .locator(collection_id, logical_id)
            .map(|locator| self.hit(locator))
    }
}

/// One source in newest-first order.
enum Source<'a> {
    Pack(&'a SealedPack),
    Run(&'a SealedRun),
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
    /// Unmerged sidecars, sorted by `(creation_seq, pack_id)`, oldest first.
    packs: Vec<SealedPack>,
    /// Runs holding this collection, oldest first. Their pack ranges never
    /// overlap each other or an unmerged pack.
    runs: Vec<Arc<SealedRun>>,
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
        let key = pack.key();
        if let Some(existing) = self
            .packs
            .iter_mut()
            .find(|existing| existing.pack_id == pack.pack_id)
        {
            *existing = pack;
            // A replacement keeps the same pack identity in practice, but a
            // changed creation sequence must not leave the vector unsorted.
            self.packs.sort_by_key(SealedPack::key);
            return;
        }
        // Open scans insert packs oldest-first, so the common path is an
        // append. Avoid re-sorting the whole vector per pack: that turned a
        // 1,000-pack open into O(packs^2 log packs) comparisons.
        if self.packs.last().is_none_or(|last| last.key() < key) {
            self.packs.push(pack);
            return;
        }
        let position = self.packs.partition_point(|existing| existing.key() < key);
        self.packs.insert(position, pack);
    }

    /// Add a run. Adding the same run again is a no-op. The caller guarantees
    /// that its packs overlap no other run or unmerged pack in the set.
    pub fn insert_run(&mut self, run: Arc<SealedRun>) {
        if self.runs.iter().any(|existing| Arc::ptr_eq(existing, &run)) {
            return;
        }
        let key = run.newest_key();
        let position = self
            .runs
            .partition_point(|existing| existing.newest_key() < key);
        self.runs.insert(position, run);
    }

    /// Whether the set already serves `pack_id`, from a sidecar or a run.
    #[must_use]
    pub fn contains_pack(&self, pack_id: PackId) -> bool {
        self.packs.iter().any(|pack| pack.pack_id == pack_id)
            || self.runs.iter().any(|run| run.covers(pack_id))
    }

    /// Slot ids referenced by sealed sidecars and runs. A run contributes
    /// every pack it covers, even ones that hold none of this collection.
    #[must_use]
    pub fn slot_ids(&self) -> std::collections::HashSet<u16> {
        self.packs
            .iter()
            .map(|pack| pack.slot)
            .chain(
                self.runs
                    .iter()
                    .flat_map(|run| run.packs().iter().map(|pack| pack.slot)),
            )
            .collect()
    }

    /// Shard slots referenced by this collection's sealed sidecars and run
    /// accounting rows. A pool-wide run contributes only packs whose original
    /// sidecar recorded entries for this collection.
    #[must_use]
    pub fn slot_ids_in_collection(
        &self,
        collection_id: &[u8; 16],
    ) -> std::collections::HashSet<u16> {
        self.packs
            .iter()
            .filter(|pack| pack.index.collection_entry_count(collection_id).is_some())
            .map(|pack| pack.slot)
            .chain(self.runs.iter().flat_map(|run| {
                run.collection_slot_counts(collection_id)
                    .map(|(slot, _)| slot)
            }))
            .collect()
    }

    /// Whether this set contains the exact published run instance.
    #[must_use]
    pub fn contains_run(&self, run: &Arc<SealedRun>) -> bool {
        self.runs.iter().any(|existing| Arc::ptr_eq(existing, run))
    }

    /// Remove the exact published run instance. Returns whether it was held.
    pub fn remove_run(&mut self, run: &Arc<SealedRun>) -> bool {
        let before = self.runs.len();
        self.runs.retain(|existing| !Arc::ptr_eq(existing, run));
        self.runs.len() != before
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

    /// Number of sources in the set: unmerged sidecars plus runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.packs.len().saturating_add(self.runs.len())
    }

    /// Whether the set holds no sidecars and no runs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.packs.is_empty() && self.runs.is_empty()
    }

    /// The unmerged sealed packs, oldest first. Packs served by runs are not
    /// listed; see [`Self::runs`].
    #[must_use]
    pub fn packs(&self) -> &[SealedPack] {
        &self.packs
    }

    /// The runs, oldest first.
    #[must_use]
    pub fn runs(&self) -> &[Arc<SealedRun>] {
        &self.runs
    }

    /// Total bytes mapped by the sidecars and runs.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        let packs = self.packs.iter().fold(0_u64, |total, pack| {
            total.saturating_add(pack.index.mapped_bytes())
        });
        self.runs.iter().fold(packs, |total, run| {
            total.saturating_add(run.run.mapped_bytes())
        })
    }

    /// Whether any sealed pack in the set occupies `slot`.
    #[must_use]
    pub fn contains_slot(&self, slot: u16) -> bool {
        self.packs.iter().any(|pack| pack.slot == slot)
            || self
                .runs
                .iter()
                .any(|run| run.packs().iter().any(|pack| pack.slot == slot))
    }

    /// Unique locators recorded for one collection across every sealed pack,
    /// counted per pack as its sidecar directory records them.
    #[must_use]
    pub fn entry_count_in_collection(&self, collection_id: &[u8; 16]) -> u64 {
        self.slot_counts_in_collection(collection_id)
            .values()
            .fold(0_u64, |total, count| total.saturating_add(*count))
    }

    /// Locator counts per shard slot for one collection, derived from each
    /// sidecar's directory entry count (the same number a slot walk would
    /// produce, because the sidecar has one entry per unique id). Packs served
    /// by a run report the counts their sidecars recorded, from the run's pack
    /// stats.
    #[must_use]
    pub fn slot_counts_in_collection(
        &self,
        collection_id: &[u8; 16],
    ) -> std::collections::HashMap<u16, u64> {
        let mut counts = std::collections::HashMap::new();
        let mut add = |slot: u16, count: u64| {
            if count > 0 {
                let entry = counts.entry(slot).or_insert(0u64);
                *entry = entry.saturating_add(count);
            }
        };
        for pack in &self.packs {
            if let Some(count) = pack.index.collection_entry_count(collection_id) {
                add(pack.slot, count);
            }
        }
        for run in &self.runs {
            for (slot, count) in run.collection_slot_counts(collection_id) {
                add(slot, count);
            }
        }
        counts
    }

    /// Sidecars and runs, newest first by pack order.
    fn newest_first(&self) -> impl Iterator<Item = Source<'_>> {
        let mut packs = self.packs.iter().rev().peekable();
        let mut runs = self.runs.iter().rev().peekable();
        std::iter::from_fn(move || match (packs.peek(), runs.peek()) {
            (Some(pack), Some(run)) if pack.key() > run.newest_key() => {
                packs.next().map(Source::Pack)
            }
            (Some(_), None) => packs.next().map(Source::Pack),
            (_, Some(_)) => runs.next().map(|run| Source::Run(run)),
            (None, None) => None,
        })
    }

    /// Resolve a key to the newest sealed version.
    ///
    /// Sources are walked newest to oldest and the first hit wins. A removal
    /// hit is returned as such; the caller must treat it as authoritative and
    /// stop falling back to older packs.
    ///
    /// # Errors
    /// Returns [`SealedUnavailable`] when a damaged run's fallback sidecars
    /// cannot be loaded. That is never a miss.
    pub fn lookup(
        &self,
        collection_id: &[u8; 16],
        logical_id: &NodeId,
    ) -> Result<Option<SealedHit>, SealedUnavailable> {
        for source in self.newest_first() {
            let hit = match source {
                Source::Pack(pack) => pack.lookup(collection_id, logical_id),
                Source::Run(run) => run.lookup(collection_id, logical_id)?,
            };
            if hit.is_some() {
                return Ok(hit);
            }
        }
        Ok(None)
    }

    /// Visit one collection's entries, newest source first.
    ///
    /// Entries from different packs are yielded with the newer pack's version
    /// first, so a caller that keeps the first occurrence of each id gets
    /// newest-wins semantics without a second sort. Duplicate ids across
    /// sources are yielded more than once; deduplication is the caller's job.
    ///
    /// # Errors
    /// Returns [`SealedUnavailable`] when a damaged run's fallback sidecars
    /// cannot be loaded; entries already visited must then be discarded.
    pub fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        mut visit: impl FnMut(SealedHit, NodeId),
    ) -> Result<(), SealedUnavailable> {
        for source in self.newest_first() {
            match source {
                Source::Pack(pack) => {
                    for_each_in_packs(std::slice::from_ref(pack), collection_id, &mut visit);
                }
                Source::Run(run) => run.for_each_in_collection(collection_id, &mut visit)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "test_sealed.rs"]
mod tests;
