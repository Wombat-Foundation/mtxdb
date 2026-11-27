//! Storage-facing index over [`super::bucket::RawBucketTable`].
//!
//! The table stores only a seeded fingerprint and locator. Full identity is
//! checked against the authoritative frame header at the storage boundary.

use std::collections::{HashMap, HashSet};

use super::bucket::{BucketCandidate, BucketSnapshotError, BucketUndo, RawBucketTable};
use super::{IndexConfig, InsertError};

/// Per-collection candidate index backed by cache-line buckets.
#[derive(Debug)]
pub struct BucketIndex {
    table: RawBucketTable,
    config: IndexConfig,
}

impl Clone for BucketIndex {
    fn clone(&self) -> Self {
        let table = RawBucketTable::from_snapshot(&self.table.snapshot())
            .expect("a table's own snapshot is valid");
        Self {
            table,
            config: self.config,
        }
    }
}

impl BucketIndex {
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
        for candidate in self.table.candidates(fingerprint) {
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
    #[must_use]
    pub fn candidates(&self, hash: &[u8; 16]) -> Vec<BucketCandidate> {
        self.table.candidates(self.table.fingerprint(hash))
    }

    /// Grow the table and rehash its stored fingerprints without reading packs.
    #[must_use]
    pub fn grow(&self) -> Option<Self> {
        Some(Self {
            table: self.table.grow()?,
            config: self.config,
        })
    }

    /// Count live (non-removal) locators.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.table
            .entries()
            .iter()
            .filter(|entry| !entry.removed)
            .count()
    }

    /// Count removal locators.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.table
            .entries()
            .iter()
            .filter(|entry| entry.removed)
            .count()
    }

    /// Iterate live `(shard, offset)` locators.
    pub fn live_entries(&self) -> impl Iterator<Item = (u16, u64)> {
        self.table
            .entries()
            .into_iter()
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
        for entry in self.table.entries() {
            if !excluded(entry.shard_slot) {
                let count = counts.entry(entry.shard_slot).or_insert(0_u64);
                *count = count.saturating_add(1);
            }
        }
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
        for entry in self.table.entries() {
            if let Some(slot) = seen.get_mut(usize::from(entry.shard_slot)) {
                *slot = true;
            }
        }
        seen
    }

    /// Whether any locator references `slot`.
    #[must_use]
    pub fn references_slot(&self, slot: u16) -> bool {
        self.table
            .entries()
            .iter()
            .any(|entry| entry.shard_slot == slot)
    }

    /// Remap every locator's process-local shard slot.
    #[must_use]
    pub fn remap_slots(&mut self, remap: &HashMap<u16, u16>) -> bool {
        let entries = self.table.entries();
        if entries
            .iter()
            .any(|entry| !remap.contains_key(&entry.shard_slot))
        {
            return false;
        }
        for entry in entries {
            let slot = remap[&entry.shard_slot];
            if slot != entry.shard_slot
                && self
                    .table
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
        true
    }

    /// Encoded v1 bucket snapshot length.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.table.snapshot().len()
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
    pub fn deserialize(data: &[u8], config: IndexConfig) -> Result<Self, BucketSnapshotError> {
        let table = RawBucketTable::from_snapshot(data)?;
        if table.len() > max_occupied(table.capacity(), config.load_factor_percent) {
            return Err(BucketSnapshotError::InvalidDimensions);
        }
        Ok(Self { table, config })
    }

    /// Bytes occupied by the owned bucket allocation and overflow slots.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.table.resident_bytes()
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
        let candidates = index.candidates(&hash);
        let [candidate] = candidates.as_slice() else {
            panic!("same identity must retain one locator")
        };
        assert_eq!(
            (candidate.shard_slot, candidate.offset, candidate.removed),
            (4, 200, true)
        );

        index.rollback_slot(&overwrite);
        let candidates = index.candidates(&hash);
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
            let candidates = grown.candidates(hash);
            let [candidate] = candidates.as_slice() else {
                panic!("snapshot and growth must retain one candidate per identity")
            };
            assert_eq!(candidate.offset, u64::try_from(offset).unwrap() << 40);
            assert_eq!(candidate.removed, offset == 2);
        }
    }
}
