//! Fixed-width, little-endian wire records for the persisted index cache.
//!
//! The cache is deliberately a separate, rebuildable acceleration structure:
//! packfiles remain authoritative.  Keeping these records fixed-width makes a
//! torn delta tail unambiguous and lets loaders bounds-check before allocating.

#[cfg(not(target_endian = "little"))]
compile_error!("the persisted index cache is currently supported only on little-endian targets");

/// Bytes in one [`DeltaFrame`].
pub const DELTA_FRAME_LEN: usize = 36;
/// Bytes in the checkpoint header.
pub const CHECKPOINT_HEADER_LEN: usize = 80;
/// Bytes in one collection directory entry.
pub const COLLECTION_DIR_ENTRY_LEN: usize = 56;
/// Bytes in one [`PackTableEntry`].
pub const PACK_TABLE_ENTRY_LEN: usize = 12;

/// One slot overwrite after a checkpoint.
///
/// `bucket` is the frame's landing bucket — the actual table position the
/// slot was written at, after `insert_tracked` advanced past any occupied
/// slots during linear probing — not the hash's home bucket. `replay_frames`
/// writes directly to it, so a producer logging a frame at the home bucket
/// instead would place the delta at the wrong slot. `slot` is the packed
/// [`crate::index::IndexSlot`] representation stored there. `generation`
/// prevents a delta for a pre-resize table being applied to a resized or
/// repacked collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaFrame {
    /// The collection whose index changed.
    pub collection_id: [u8; 16],
    /// The affected table bucket.
    pub bucket: u32,
    /// The collection-index generation this frame applies to.
    pub generation: u64,
    /// The packed `IndexSlot` value to store.
    pub slot: u64,
}

impl DeltaFrame {
    #[must_use]
    /// Encodes this frame in its fixed-width little-endian representation.
    pub fn encode(self) -> [u8; DELTA_FRAME_LEN] {
        let mut bytes = [0; DELTA_FRAME_LEN];
        bytes[..16].copy_from_slice(&self.collection_id);
        bytes[16..20].copy_from_slice(&self.bucket.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.generation.to_le_bytes());
        bytes[28..36].copy_from_slice(&self.slot.to_le_bytes());
        bytes
    }

    #[must_use]
    /// Decodes exactly one fixed-width frame, rejecting any other length.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; DELTA_FRAME_LEN] = bytes.try_into().ok()?;
        Some(Self {
            collection_id: bytes[..16].try_into().ok()?,
            bucket: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
            generation: u64::from_le_bytes(bytes[20..28].try_into().ok()?),
            slot: u64::from_le_bytes(bytes[28..36].try_into().ok()?),
        })
    }
}

/// Checkpoint envelope.  The directory immediately follows this header,
/// followed by the concatenated raw index slot arrays, then homes and tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointHeader {
    /// Format-identifying magic bytes.
    pub magic: [u8; 8],
    /// Wire-format version.
    pub version: u32,
    /// Number of collection directory entries.
    pub collection_count: u32,
    /// Total byte length of the directory section.
    pub directory_bytes: u64,
    /// Total byte length of the raw-slot section.
    pub slots_bytes: u64,
    /// Fingerprint of the packfile set the checkpoint describes.
    pub pack_fingerprint: u64,
    /// CRC-32/IEEE (`crc32fast`, the same crate/polynomial already used for
    /// pack-frame checksums) of every byte after this header — the
    /// directory, raw slot, homes, and tails sections combined. Verified
    /// once, in full, at read time.
    pub content_crc32: u32,
    /// Total byte length of the homes section (all collections concatenated).
    /// Zero for pre-v4 checkpoints.
    pub homes_bytes: u64,
    /// Total byte length of the tails section (all collections concatenated).
    /// Zero for pre-v4 checkpoints.
    pub tails_bytes: u64,
    /// Journal LSN whose mutations this checkpoint's index snapshot already
    /// incorporates. Written atomically with the checkpoint, so a reader can
    /// bind its overlay coverage to exactly the index it loaded rather than to
    /// a separately-read `journal.lsn` that may have advanced past it. Zero
    /// when the checkpoint was written without a journal.
    pub covered_lsn: u64,
    /// Number of [`PackTableEntry`] records in the pack table.
    pub pack_table_count: u32,
    /// Total byte length of the pack table section
    /// (`pack_table_count * PACK_TABLE_ENTRY_LEN`).
    pub pack_table_bytes: u64,
}

impl CheckpointHeader {
    #[must_use]
    /// Encodes this header in its fixed-width little-endian representation.
    pub fn encode(self) -> [u8; CHECKPOINT_HEADER_LEN] {
        let mut bytes = [0; CHECKPOINT_HEADER_LEN];
        bytes[..8].copy_from_slice(&self.magic);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.collection_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.directory_bytes.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.slots_bytes.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.pack_fingerprint.to_le_bytes());
        bytes[40..44].copy_from_slice(&self.content_crc32.to_le_bytes());
        bytes[44..52].copy_from_slice(&self.homes_bytes.to_le_bytes());
        bytes[52..60].copy_from_slice(&self.tails_bytes.to_le_bytes());
        bytes[60..68].copy_from_slice(&self.covered_lsn.to_le_bytes());
        bytes
    }

    #[must_use]
    /// Decodes exactly one fixed-width header, rejecting any other length.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; CHECKPOINT_HEADER_LEN] = bytes.try_into().ok()?;
        Some(Self {
            magic: bytes[..8].try_into().ok()?,
            version: u32::from_le_bytes(bytes[8..12].try_into().ok()?),
            collection_count: u32::from_le_bytes(bytes[12..16].try_into().ok()?),
            directory_bytes: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
            slots_bytes: u64::from_le_bytes(bytes[24..32].try_into().ok()?),
            pack_fingerprint: u64::from_le_bytes(bytes[32..40].try_into().ok()?),
            content_crc32: u32::from_le_bytes(bytes[40..44].try_into().ok()?),
            homes_bytes: u64::from_le_bytes(bytes[44..52].try_into().ok()?),
            tails_bytes: u64::from_le_bytes(bytes[52..60].try_into().ok()?),
            covered_lsn: u64::from_le_bytes(bytes[60..68].try_into().ok()?),
        })
    }
}

/// Locates one collection's raw slot, homes, and tails arrays in a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionDirEntry {
    /// The collection whose raw slots this entry locates.
    pub collection_id: [u8; 16],
    /// The checkpoint generation for this collection.
    pub generation: u64,
    /// Byte offset of the collection's slot data in the checkpoint.
    pub slots_offset: u64,
    /// Byte offset of the collection's homes data in the checkpoint.
    /// Zero for pre-v4 checkpoints (homes not persisted).
    pub homes_offset: u64,
    /// Byte offset of the collection's tails data in the checkpoint.
    /// Zero for pre-v4 checkpoints (tails not persisted).
    pub tails_offset: u64,
    /// Number of slots allocated by the collection's index.
    pub capacity: u32,
    /// Number of occupied slots.
    pub slot_count: u32,
}

impl CollectionDirEntry {
    #[must_use]
    /// Encodes this entry in its fixed-width little-endian representation.
    pub fn encode(self) -> [u8; COLLECTION_DIR_ENTRY_LEN] {
        let mut bytes = [0; COLLECTION_DIR_ENTRY_LEN];
        bytes[..16].copy_from_slice(&self.collection_id);
        bytes[16..24].copy_from_slice(&self.generation.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.slots_offset.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.homes_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.tails_offset.to_le_bytes());
        bytes[48..52].copy_from_slice(&self.capacity.to_le_bytes());
        bytes[52..56].copy_from_slice(&self.slot_count.to_le_bytes());
        bytes
    }

    #[must_use]
    /// Decodes exactly one fixed-width directory entry, rejecting other lengths.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; COLLECTION_DIR_ENTRY_LEN] = bytes.try_into().ok()?;
        Some(Self {
            collection_id: bytes[..16].try_into().ok()?,
            generation: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
            slots_offset: u64::from_le_bytes(bytes[24..32].try_into().ok()?),
            homes_offset: u64::from_le_bytes(bytes[32..40].try_into().ok()?),
            tails_offset: u64::from_le_bytes(bytes[40..48].try_into().ok()?),
            capacity: u32::from_le_bytes(bytes[48..52].try_into().ok()?),
            slot_count: u32::from_le_bytes(bytes[52..56].try_into().ok()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_records_round_trip() {
        let delta = DeltaFrame {
            collection_id: [1; 16],
            bucket: 2,
            generation: 3,
            slot: 4,
        };
        assert_eq!(DeltaFrame::decode(&delta.encode()), Some(delta));

        let header = CheckpointHeader {
            magic: *b"MTXIDX01",
            version: 4,
            collection_count: 2,
            directory_bytes: 112,
            slots_bytes: 128,
            pack_fingerprint: 9,
            content_crc32: 0xDEAD_BEEF,
            homes_bytes: 0,
            tails_bytes: 0,
            covered_lsn: 11,
        };
        assert_eq!(CheckpointHeader::decode(&header.encode()), Some(header));

        let entry = CollectionDirEntry {
            collection_id: [5; 16],
            generation: 6,
            slots_offset: 7,
            homes_offset: 100,
            tails_offset: 200,
            capacity: 8,
            slot_count: 9,
        };
        assert_eq!(CollectionDirEntry::decode(&entry.encode()), Some(entry));
    }
}
