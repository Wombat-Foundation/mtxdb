//! Removal inspection and a standalone removal for one pool.
//!
//! Two levels of inspection, so a cheap question never pays for an expensive
//! one:
//!
//! - [`PackfileStorage::removal_counts`] answers "how many keys are live and
//!   how many are removed" from the in-memory index alone (the removed bit in
//!   each entry). No pack is read.
//! - [`PackfileStorage::removal_stats`] and [`PackfileStorage::removed_keys`]
//!   read the packs: frame counts, superseded versions and the bytes a repack
//!   could give back, or the removed ids themselves. The index keeps neither
//!   frame sizes, nor superseded-version totals, nor complete 128-bit ids, so
//!   these cannot be derived from it. They use the same ordering repack uses
//!   (pack creation sequence, then offset), so they cannot drift from the data
//!   after a crash.
//!
//! [`PackfileStorage::remove_record`] is the pool-level counterpart of
//! `DatabaseTransaction::remove` for tooling that works on one pool directory.

use super::{
    packfile, HashMap, JournalMutation, NodeId, PackfileStorage, StorageEngine, StorageError,
};

/// What a collection's packs hold, split into live data and removals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemovalStats {
    /// Distinct keys whose newest version is a live value.
    pub live_keys: usize,
    /// Distinct keys whose newest version is a removal marker.
    pub removed_keys: usize,
    /// Every frame the collection has in its packs, all versions included.
    pub frames: usize,
    /// Frames that are removal markers, newest or not.
    pub marker_frames: usize,
    /// Frames replaced by a newer version of the same key.
    pub superseded_frames: usize,
    /// Bytes of every frame of the collection.
    pub total_bytes: u64,
    /// Bytes a repack could drop: superseded versions plus the newest marker
    /// of each removed key. An upper bound, because a shard another
    /// collection still references is retained, and a marker is carried
    /// forward while an older copy may survive there.
    pub reclaimable_bytes: u64,
}

/// Live and removed key counts taken from the index alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemovalCounts {
    /// Indexed keys whose newest version is a live value.
    pub live_keys: usize,
    /// Indexed keys whose newest version is a removal marker.
    pub removed_keys: usize,
}

struct Newest {
    position: (u32, u64),
    removed: bool,
    len: u64,
}

impl PackfileStorage {
    /// Count live keys, removed keys and reclaimable bytes for a collection.
    ///
    /// Reads every frame of the collection (one verified read each), so it is a
    /// diagnostic, not a hot-path call. It looks at the packs only: a mutation
    /// still waiting in the journal overlay is not reflected until it is
    /// materialized.
    ///
    /// # Errors
    /// Returns [`StorageError::Io`] or [`StorageError::Corrupt`] if a pack
    /// cannot be scanned or a frame cannot be read.
    pub fn removal_stats(&self, collection_id: &[u8; 16]) -> Result<RemovalStats, StorageError> {
        let (newest, mut stats) = self.newest_versions(collection_id)?;
        for version in newest.values() {
            if version.removed {
                stats.removed_keys = stats.removed_keys.saturating_add(1);
                stats.reclaimable_bytes = stats.reclaimable_bytes.saturating_add(version.len);
            } else {
                stats.live_keys = stats.live_keys.saturating_add(1);
            }
        }
        Ok(stats)
    }

    /// The keys whose newest version is a removal marker, sorted.
    ///
    /// # Errors
    /// As [`Self::removal_stats`].
    pub fn removed_keys(&self, collection_id: &[u8; 16]) -> Result<Vec<NodeId>, StorageError> {
        let (newest, _) = self.newest_versions(collection_id)?;
        let mut keys: Vec<NodeId> = newest
            .into_iter()
            .filter(|(_, version)| version.removed)
            .map(|(key, _)| key)
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Remove one record from this pool, returning whether it was live.
    ///
    /// Removing a key that is not live writes nothing. With a journal enabled
    /// the removal is published as a `RemoveRecord` mutation first, so replay
    /// and the change feed see it; without one it is written directly, like
    /// [`StorageEngine::put`]. Single-pool maintenance only: a transaction
    /// spanning pools uses `DatabaseTransaction::remove`.
    ///
    /// The liveness check, the publication and the application all run under
    /// the collection's put lock (and the snapshot gate), in the same lock
    /// order `put_internal` uses. Serializing with a concurrent put on
    /// the same collection is what keeps the live index and WAL replay in
    /// agreement: without it, a put could publish a later LSN and apply first,
    /// and this removal would then apply its older tombstone last, leaving the
    /// index saying removed while replay (LSN order) restored the record live.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the liveness read, the journal publish or
    /// the write fails.
    pub fn remove_record(
        &self,
        collection_id: &[u8; 16],
        node_id: &NodeId,
    ) -> Result<bool, StorageError> {
        let _snapshot_gate = self.read_snapshot_gate.lock();
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        if self.get(collection_id, node_id)?.is_none() {
            return Ok(false);
        }
        let mutation = JournalMutation::RemoveRecord {
            collection_id: *collection_id,
            node_id: *node_id,
        };
        let published_lsn = self.publish_mutation(|| mutation.clone())?;
        self.apply_remove_record_locked(collection_id, node_id, published_lsn.unwrap_or(0))?;
        self.note_published_materialized(published_lsn);
        Ok(true)
    }

    /// Live and removed key counts from the in-memory index, with no pack
    /// reads. `None` if the collection is not loaded.
    ///
    /// These count indexed ids, so they include any genesis metadata record,
    /// and they reflect what this handle's index holds (a read-only handle can
    /// lag a writer). [`Self::removal_stats`] gives the same two numbers from
    /// the packs, plus frame and byte totals.
    #[must_use]
    pub fn removal_counts(&self, collection_id: &[u8; 16]) -> Option<RemovalCounts> {
        self.collections_read()
            .get(collection_id)
            .map(|generation| {
                let loaded = generation.load();
                let removed = loaded.index.removed_count();
                RemovalCounts {
                    live_keys: loaded.index.len().saturating_sub(removed),
                    removed_keys: removed,
                }
            })
    }

    /// How many of the collection's index entries point at an explicit removal
    /// version, counted from the in-memory index alone (no pack reads). With
    /// the collection's `len`, this gives the live count: `len - removed`.
    /// `None` if the collection is not loaded.
    #[must_use]
    pub fn indexed_removed_count(&self, collection_id: &[u8; 16]) -> Option<usize> {
        self.collections_read()
            .get(collection_id)
            .map(|generation| generation.load().index.removed_count())
    }

    fn newest_versions(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<(HashMap<NodeId, Newest>, RemovalStats), StorageError> {
        self.shards.flush_all()?;
        let verify = self.shards.checksum_policy().verifies_reads();
        let mut stats = RemovalStats::default();
        let mut newest: HashMap<NodeId, Newest> = HashMap::new();
        for (_, shard) in self.shards.all_shards() {
            let file_len = std::fs::metadata(&shard.path)?.len();
            let mut entries = packfile::scan_packfile(&shard.path)?;
            entries.sort_unstable_by_key(|(_, _, offset)| *offset);
            for (index, (rid, hash, offset)) in entries.iter().enumerate() {
                if rid != collection_id {
                    continue;
                }
                let end = entries
                    .get(index.saturating_add(1))
                    .map_or(file_len, |(_, _, next)| *next);
                let len = end.saturating_sub(*offset);
                let removed = self
                    .read_at(&shard, *offset, verify)?
                    .metadata
                    .is_some_and(|metadata| metadata.removed);
                stats.frames = stats.frames.saturating_add(1);
                stats.total_bytes = stats.total_bytes.saturating_add(len);
                if removed {
                    stats.marker_frames = stats.marker_frames.saturating_add(1);
                }
                let position = (shard.creation_seq, *offset);
                let candidate = Newest {
                    position,
                    removed,
                    len,
                };
                match newest.get_mut(hash) {
                    Some(current) if current.position >= position => {
                        stats.superseded_frames = stats.superseded_frames.saturating_add(1);
                        stats.reclaimable_bytes = stats.reclaimable_bytes.saturating_add(len);
                    }
                    Some(current) => {
                        stats.superseded_frames = stats.superseded_frames.saturating_add(1);
                        stats.reclaimable_bytes =
                            stats.reclaimable_bytes.saturating_add(current.len);
                        *current = candidate;
                    }
                    None => {
                        newest.insert(*hash, candidate);
                    }
                }
            }
        }
        Ok((newest, stats))
    }
}
