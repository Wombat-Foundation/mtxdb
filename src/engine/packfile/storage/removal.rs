//! Removal inspection and a standalone removal for one pool.
//!
//! [`PackfileStorage::removal_stats`] reports, from the packs themselves, how
//! many keys a collection holds live, how many have been removed, and how many
//! bytes a repack could give back. It is computed on demand from the same
//! ordering repack uses (pack creation sequence, then offset), so it cannot
//! drift from the data after a crash. [`PackfileStorage::remove_record`] is the
//! pool-level counterpart of `DatabaseTransaction::remove` for tooling that
//! works on one pool directory.

use super::*;

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
    /// # Errors
    /// Returns [`StorageError`] if the liveness read, the journal publish or
    /// the write fails.
    pub fn remove_record(
        &self,
        collection_id: &[u8; 16],
        node_id: &NodeId,
    ) -> Result<bool, StorageError> {
        if self.get(collection_id, node_id)?.is_none() {
            return Ok(false);
        }
        let mutation = JournalMutation::RemoveRecord {
            collection_id: *collection_id,
            node_id: *node_id,
        };
        let lsn = self.publish_mutation(|| mutation.clone())?.unwrap_or(0);
        self.apply_transaction_mutation(&mutation, lsn)?;
        Ok(true)
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
