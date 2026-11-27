//! Storage-facing index over [`super::bucket::RawBucketTable`].
//!
//! The table stores only a seeded fingerprint and locator. Full identity is
//! checked against the authoritative frame header at the storage boundary.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use super::bucket::{BucketCandidate, BucketSnapshotError, BucketUndo, RawBucketTable};
use super::{IndexConfig, InsertError};

/// Per-collection candidate index backed by cache-line buckets.
#[derive(Debug)]
pub struct BucketIndex {
    table: RawBucketTable,
    config: IndexConfig,
    max_candidates: AtomicU32,
}

impl Clone for BucketIndex {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone_direct(),
            config: self.config,
            max_candidates: AtomicU32::new(self.max_candidates.load(Ordering::Relaxed)),
        }
    }
}

impl BucketIndex {
    /// Largest pack offset representable in a bucket locator.
    pub const MAX_RECORD_OFFSET: u64 = (1_u64 << 48) - 1;

    /// Create an index with at least `min_entries` base slots.
    #[must_use]
    pub fn new(min_entries: usize) -> Self {
        Self::with_config(min_entries, IndexConfig::default())
    }

    /// Create an index using the pool's hash seed and load threshold.
    #[must_use]
    pub fn with_config(min_entries: usize, config: IndexConfig) -> Self {
        let min_entries = min_entries.max(usize::try_from(config.floor.max(16)).unwrap_or(16));
        Self {
            table: RawBucketTable::new(min_entries, config.seed),
            config,
            max_candidates: AtomicU32::new(0),
        }
    }

    /// Base capacity, excluding the bounded collision overflow area.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        u32::try_from(self.table.capacity()).unwrap_or(u32::MAX)
    }

    /// Occupied locators, including collision overflow slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether this index has no locators.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Insert or replace a key after checking every same-fingerprint locator
    /// against the authoritative record identity.
    ///
    /// # Errors
    /// Returns the identity callback's error if a candidate frame cannot be
    /// read. The inner result is [`InsertError::TableFull`] when the current
    /// table cannot accept a distinct key.
    pub fn insert_with_identity<E>(
        &self,
        hash: &[u8; 16],
        shard_slot: u16,
        offset: u64,
        removed: bool,
        mut identity_at: impl FnMut(&BucketCandidate) -> Result<[u8; 16], E>,
    ) -> Result<Result<(BucketUndo, bool), InsertError>, E> {
        let fingerprint = self.table.fingerprint(hash);
        for candidate in self.candidates_for_fingerprint(fingerprint) {
            if identity_at(&candidate)? == *hash {
                return Ok(self
                    .table
                    .replace(candidate.location, fingerprint, shard_slot, offset, removed)
                    .map(|undo| (undo, false))
                    .ok_or(InsertError::TableFull));
            }
        }
        let threshold = max_occupied(self.table.capacity(), self.config.load_factor_percent);
        if self.len() >= threshold {
            return Ok(Err(InsertError::TableFull));
        }
        Ok(self
            .table
            .insert_new(fingerprint, shard_slot, offset, removed)
            .map(|undo| (undo, true))
            .ok_or(InsertError::TableFull))
    }

    /// Undo a prior insert or replacement while holding the collection lock.
    pub fn rollback_slot(&self, undo: &BucketUndo) {
        self.table.rollback(*undo);
    }

    /// Candidates with this key's seeded fingerprint. Callers must verify
    /// each candidate against the frame's full identity before returning it.
    pub fn candidates(&self, hash: &[u8; 16]) -> impl Iterator<Item = BucketCandidate> + '_ {
        self.candidates_for_fingerprint(self.table.fingerprint(hash))
    }

    /// Iterate candidate `(shard, offset)` locators for a key.
    pub fn lookup_all(&self, hash: &[u8; 16]) -> impl Iterator<Item = (u16, u64)> + '_ {
        self.candidates(hash)
            .map(|candidate| (candidate.shard_slot, candidate.offset))
    }

    /// Return the first candidate locator for a key, if any.
    #[must_use]
    pub fn lookup(&self, hash: &[u8; 16]) -> Option<(u16, u64)> {
        self.candidates(hash)
            .next()
            .map(|candidate| (candidate.shard_slot, candidate.offset))
    }

    fn candidates_for_fingerprint(
        &self,
        fingerprint: u64,
    ) -> impl Iterator<Item = BucketCandidate> + '_ {
        let mut candidates = self.table.candidate_iter(fingerprint);
        let mut seen = 0_u32;
        std::iter::from_fn(move || {
            candidates.next().inspect(|_| {
                seen = seen.saturating_add(1);
                self.max_candidates.fetch_max(seen, Ordering::Relaxed);
            })
        })
    }

    /// Maximum candidate count seen in a fingerprint lookup.
    #[must_use]
    pub fn max_candidate_count(&self) -> u32 {
        self.max_candidates.load(Ordering::Relaxed)
    }

    /// Compatibility metric for callers still using the former probe-chain
    /// statistic. It now reports the maximum fingerprint candidate count.
    #[must_use]
    pub fn max_probe_len(&self) -> u32 {
        self.max_candidate_count()
    }

    /// Grow the table and rehash its stored fingerprints without reading packs.
    #[must_use]
    pub fn grow(&self) -> Option<Self> {
        Some(Self {
            table: self.table.grow()?,
            config: self.config,
            max_candidates: AtomicU32::new(self.max_candidate_count()),
        })
    }

    /// Count live (non-removal) locators.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.table.live_len()
    }

    /// Count removal locators.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.table.len().saturating_sub(self.table.live_len())
    }

    /// Iterate live `(shard, offset)` locators.
    pub fn live_entries(&self) -> impl Iterator<Item = (u16, u64)> + use<'_> {
        self.table
            .iter_entries()
            .filter(|entry| !entry.removed)
            .map(|entry| (entry.shard_slot, entry.offset))
    }

    /// Count occupied locators by shard slot.
    #[must_use]
    pub fn slot_counts(&self) -> HashMap<u16, u64> {
        self.slot_counts_excluding(|_| false)
    }

    /// Count occupied locators, excluding slots accounted for elsewhere.
    #[must_use]
    pub fn slot_counts_excluding(
        &self,
        mut excluded: impl FnMut(u16) -> bool,
    ) -> HashMap<u16, u64> {
        let mut counts = HashMap::new();
        self.table.for_each_entry(|entry| {
            if !excluded(entry.shard_slot) {
                let count = counts.entry(entry.shard_slot).or_insert(0_u64);
                *count = count.saturating_add(1);
            }
            true
        });
        counts
    }

    /// Count occupied locators excluding a set of shard slots.
    #[must_use]
    pub fn slot_counts_excluding_set(&self, excluded: &HashSet<u16>) -> HashMap<u16, u64> {
        self.slot_counts_excluding(|slot| excluded.contains(&slot))
    }

    /// Which shard slots have at least one occupied locator.
    #[must_use]
    pub fn referenced_slot_ids(&self) -> [bool; crate::shard::MAX_SHARDS] {
        let mut seen = [false; crate::shard::MAX_SHARDS];
        self.table.for_each_entry(|entry| {
            if let Some(slot) = seen.get_mut(usize::from(entry.shard_slot)) {
                *slot = true;
            }
            true
        });
        seen
    }

    /// Whether any locator references `slot`.
    #[must_use]
    pub fn references_slot(&self, slot: u16) -> bool {
        self.table.references_slot(slot)
    }

    /// Remap every locator's process-local shard slot.
    #[must_use]
    pub fn remap_slots(&mut self, remap: &HashMap<u16, u16>) -> bool {
        let mut all_mapped = true;
        let mut needs_rewrite = false;
        self.table.for_each_entry(|entry| {
            match remap.get(&entry.shard_slot) {
                Some(&new_slot) => needs_rewrite |= new_slot != entry.shard_slot,
                None => all_mapped = false,
            }
            all_mapped
        });
        if !all_mapped {
            return false;
        }
        if !needs_rewrite {
            return true;
        }
        let entries = self.table.entries();
        // Apply to a private copy so a failed `replace` cannot leave the live
        // table half-remapped.
        let table = self.table.clone_direct();
        for entry in entries {
            let slot = remap[&entry.shard_slot];
            if slot != entry.shard_slot
                && table
                    .replace(
                        entry.location,
                        entry.fingerprint,
                        slot,
                        entry.offset,
                        entry.removed,
                    )
                    .is_none()
            {
                return false;
            }
        }
        self.table = table;
        true
    }

    /// Encoded v1 bucket snapshot length.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.table.snapshot_len()
    }

    /// Encode the bucket snapshot.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        self.table.snapshot()
    }

    /// Decode a bucket snapshot using this pool's load-factor policy.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot is malformed, corrupt,
    /// or contains more entries than the configured load threshold permits.
    pub fn deserialize_with_config(
        data: &[u8],
        config: IndexConfig,
    ) -> Result<Self, BucketSnapshotError> {
        Self::deserialize(data, config)
    }

    /// Decode a bucket snapshot using this pool's load-factor policy.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot is malformed, corrupt,
    /// or contains more entries than the configured load threshold permits.
    pub fn deserialize(data: &[u8], config: IndexConfig) -> Result<Self, BucketSnapshotError> {
        let table = RawBucketTable::from_snapshot(data)?;
        if table.len() > max_occupied(table.capacity(), config.load_factor_percent) {
            return Err(BucketSnapshotError::InvalidDimensions);
        }
        Ok(Self {
            table,
            config,
            max_candidates: AtomicU32::new(0),
        })
    }

    /// The table is always resident; it is never backed by a checkpoint mmap.
    #[must_use]
    pub const fn is_mmap_backed(&self) -> bool {
        false
    }

    /// Bytes mapped from checkpoint storage. Bucket snapshots are decoded to
    /// owned cache-line buckets, so this is always zero.
    #[must_use]
    pub const fn mapped_bytes(&self) -> usize {
        0
    }

    /// Estimated total memory for the standard capacity policy.
    #[must_use]
    pub fn memory_usage_for_entries(entries: usize, min_capacity: usize) -> Option<usize> {
        Self::capacity_for_entries(entries, min_capacity)?
            .checked_mul(16)?
            .checked_add(std::mem::size_of::<Self>())
    }

    /// Table capacity after applying the standard two-times-entry allocation
    /// policy and the caller's starting floor.
    #[must_use]
    pub fn capacity_for_entries(entries: usize, min_capacity: usize) -> Option<usize> {
        entries
            .saturating_mul(2)
            .max(min_capacity.max(16))
            .checked_next_power_of_two()
    }

    /// Size a table for `entries` under the supplied maximum load factor.
    #[must_use]
    pub fn capacity_for_entries_with_config(
        entries: usize,
        min_capacity: usize,
        config: IndexConfig,
    ) -> Option<usize> {
        config.validate().ok()?;
        let percent = usize::from(config.load_factor_percent);
        if percent == 0 {
            return None;
        }
        let numerator = entries
            .checked_mul(100)?
            .checked_add(percent.saturating_sub(1))?;
        numerator
            .checked_div(percent)?
            .max(min_capacity.max(16))
            .checked_next_power_of_two()
    }

    /// Bytes occupied by the owned bucket allocation and overflow slots.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.table.resident_bytes()
    }

    /// Estimate index memory, including the table header and overflow slots.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.resident_bytes()
    }
}

fn max_occupied(capacity: usize, percent: u8) -> usize {
    capacity.saturating_mul(usize::from(percent)) / 100
}

#[cfg(test)]
mod tests {
    use super::BucketIndex;
    use crate::index::IndexConfig;

    #[test]
    fn exact_identity_overwrite_and_undo_preserve_count() {
        let index = BucketIndex::with_config(
            16,
            IndexConfig {
                seed: 77,
                floor: 16,
                load_factor_percent: 75,
            },
        );
        let hash = [0xA5; 16];
        let (insert, was_new) = index
            .insert_with_identity(&hash, 2, 100, false, |_| Ok::<_, ()>(hash))
            .unwrap()
            .unwrap();
        assert!(was_new);
        assert!(insert.was_empty());
        assert_eq!(index.len(), 1);

        let (overwrite, was_new) = index
            .insert_with_identity(&hash, 4, 200, true, |_| Ok::<_, ()>(hash))
            .unwrap()
            .unwrap();
        assert!(!was_new);
        assert!(!overwrite.was_empty());
        assert_eq!(index.len(), 1);
        let candidates: Vec<_> = index.candidates(&hash).collect();
        let [candidate] = candidates.as_slice() else {
            panic!("same identity must retain one locator")
        };
        assert_eq!(
            (candidate.shard_slot, candidate.offset, candidate.removed),
            (4, 200, true)
        );

        index.rollback_slot(&overwrite);
        let candidates: Vec<_> = index.candidates(&hash).collect();
        let [candidate] = candidates.as_slice() else {
            panic!("rollback must restore one locator")
        };
        assert_eq!(
            (candidate.shard_slot, candidate.offset, candidate.removed),
            (2, 100, false)
        );
        index.rollback_slot(&insert);
        assert!(index.is_empty());
    }

    #[test]
    fn bucket_snapshot_round_trip_and_growth_keep_candidate_identity() {
        let index = BucketIndex::with_config(
            16,
            IndexConfig {
                seed: 0xCAFE,
                floor: 16,
                load_factor_percent: 75,
            },
        );
        let hashes = [[0x11; 16], [0x22; 16], [0x33; 16]];
        for (offset, hash) in hashes.iter().enumerate() {
            index
                .insert_with_identity(
                    hash,
                    7,
                    u64::try_from(offset).unwrap() << 40,
                    offset == 2,
                    |_| Ok::<_, ()>(*hash),
                )
                .unwrap()
                .unwrap();
        }
        let restored = BucketIndex::deserialize(
            &index.serialize(),
            IndexConfig {
                seed: 0xCAFE,
                floor: 16,
                load_factor_percent: 75,
            },
        )
        .unwrap();
        assert_eq!(restored.len(), 3);
        let grown = restored.grow().unwrap();
        for (offset, hash) in hashes.iter().enumerate() {
            let candidates: Vec<_> = grown.candidates(hash).collect();
            let [candidate] = candidates.as_slice() else {
                panic!("snapshot and growth must retain one candidate per identity")
            };
            assert_eq!(candidate.offset, u64::try_from(offset).unwrap() << 40);
            assert_eq!(candidate.removed, offset == 2);
        }
    }

    #[test]
    fn configured_capacity_respects_load_factor() {
        let config = IndexConfig {
            seed: 7,
            floor: 16,
            load_factor_percent: 25,
        };
        assert_eq!(
            BucketIndex::capacity_for_entries_with_config(100, 16, config),
            Some(512)
        );
        let invalid = IndexConfig {
            load_factor_percent: 0,
            ..config
        };
        assert_eq!(
            BucketIndex::capacity_for_entries_with_config(100, 16, invalid),
            None
        );
    }

    #[test]
    fn standard_capacity_keeps_two_times_entry_policy() {
        assert_eq!(BucketIndex::capacity_for_entries(40, 64), Some(128));
        assert_eq!(BucketIndex::capacity_for_entries(48, 64), Some(128));
        assert_eq!(
            BucketIndex::capacity_for_entries_with_config(48, 64, IndexConfig::default()),
            Some(64)
        );
    }

    #[test]
    fn identity_remap_preserves_snapshot_and_missing_mapping_is_atomic() {
        let index = BucketIndex::new(16);
        let hashes = [[0x11; 16], [0x22; 16]];
        for (slot, hash) in hashes.iter().enumerate() {
            index
                .insert_with_identity(hash, u16::try_from(slot).unwrap(), 10, false, |_| {
                    Ok::<_, ()>(*hash)
                })
                .unwrap()
                .unwrap();
        }
        let before = index.serialize();
        let mut identity = index.clone();
        assert!(identity.remap_slots(&std::collections::HashMap::from([(0, 0), (1, 1)])));
        assert_eq!(identity.serialize(), before);

        let mut incomplete = index.clone();
        assert!(!incomplete.remap_slots(&std::collections::HashMap::from([(0, 9)])));
        assert_eq!(incomplete.serialize(), before);
    }
}
