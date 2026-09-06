/// Per-slot entry in the lossy fanout index.
///
/// Layout: `[24-bit tag | 8-bit pack_id | 32-bit offset]` packed into a `u64`.
///
/// - **tag** (high 24 bits): truncated fingerprint for fast rejection.
/// - **`pack_id`** (next 8 bits): which packfile generation this record lives in.
/// - **offset** (low 32 bits): byte offset within the packfile, stored as
///   `offset + 1` so that the all-zeros encoding is reserved as the empty
///   sentinel. Actual offset 0 is stored as 1, and `offset()` subtracts 1
///   to recover the real value.
///
/// Empty slots are all-zeros. The tag serves double duty: an empty slot
/// (tag == 0) terminates a probe sequence, since hashes are uniformly random
/// and the probability of a legitimate hash mapping to tag 0 is 1/16M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSlot(u64);

impl IndexSlot {
    const EMPTY: Self = Self(0);

    const TAG_SHIFT: u64 = 40; // PACK_BITS + OFFSET_BITS
    const PACK_SHIFT: u64 = 32; // OFFSET_BITS
    const OFFSET_MASK: u64 = 0xFFFF_FFFF;

    /// Create a new slot from its components.
    ///
    /// The offset is stored as `offset + 1` so that the all-zeros `u64`
    /// encoding is reserved as the empty sentinel. Actual offset 0 is
    /// stored as 1 in the slot.
    ///
    /// # Panics
    /// Panics if `tag` exceeds 24 bits or `offset` exceeds `u32::MAX - 1`.
    #[must_use]
    pub fn new(tag: u32, pack_id: u8, offset: u64) -> Self {
        assert!(tag <= 0xFF_FFFF, "tag must fit in 24 bits");
        assert!(
            offset <= u64::from(u32::MAX - 1),
            "offset must fit in 32 bits minus 1 (reserved for empty sentinel)"
        );
        Self(
            (u64::from(tag) << Self::TAG_SHIFT)
                | (u64::from(pack_id) << Self::PACK_SHIFT)
                | ((offset.wrapping_add(1)) & Self::OFFSET_MASK),
        )
    }

    #[must_use]
    pub fn empty() -> Self {
        Self::EMPTY
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub fn tag(self) -> u32 {
        ((self.0 >> Self::TAG_SHIFT) & 0xFF_FFFF) as u32
    }

    #[must_use]
    pub fn pack_id(self) -> u8 {
        ((self.0 >> Self::PACK_SHIFT) & 0xFF) as u8
    }

    #[must_use]
    pub fn offset(self) -> u64 {
        (self.0 & Self::OFFSET_MASK).wrapping_sub(1)
    }
}

/// A per-room lossy fanout index for content-addressed records.
///
/// Uses open addressing with linear probing on a power-of-two sized table.
/// The index is mmap-able (flat `u64` array) and small enough to stay
/// resident in RAM for active rooms.
///
/// Design decisions (from docs):
/// - Partition by room: ~8KB per 1000-node room, 100 active rooms < 1MB.
/// - Empty slot terminates probe (write-once, no tombstones needed).
/// - Tag collisions surface as verification failures (`decode_v1_verified`).
/// - Power-of-two capacity: shift-and-mask bucket selection, cache-aligned probes.
#[derive(Debug)]
pub struct LossyIndex {
    /// Power-of-two capacity.
    capacity: u32,
    /// Bitmask for bucket selection: capacity - 1.
    mask: u32,
    /// Shift to extract top bits from hash for bucket index.
    shift: u32,
    /// The flat slot array.
    slots: Vec<IndexSlot>,
    /// Number of occupied slots.
    len: u32,
}

impl LossyIndex {
    /// Create a new index with at least `min_capacity` slots.
    /// Capacity is rounded up to the next power of two.
    ///
    /// # Panics
    /// Panics if `min_capacity` exceeds `u32::MAX` when rounded to a power of two.
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        let capacity_usize = min_capacity.next_power_of_two().max(16);
        let capacity = u32::try_from(capacity_usize).expect("index capacity exceeds u32::MAX");
        let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());
        Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift,
            slots: vec![IndexSlot::empty(); capacity_usize],
            len: 0,
        }
    }

    /// Extract the bucket index from a 16-byte hash.
    #[inline]
    fn bucket(&self, hash: &[u8; 16]) -> usize {
        let top_bytes = u64::from_be_bytes(hash[..8].try_into().unwrap());
        let masked = (top_bytes >> self.shift) & u64::from(self.mask);
        // Safety: masked is always <= mask < capacity which fits in usize on all platforms
        usize::try_from(masked).unwrap_or(usize::MAX)
    }

    /// Extract the 24-bit tag from a 16-byte hash.
    ///
    /// Uses bytes 8..12, disjoint from the bytes `bucket()` reads (0..8).
    /// If the tag were derived from bucket bits (or a superset of them),
    /// same-bucket entries would already agree on those bits, collapsing
    /// the tag's effective discriminating power from 2^-24 to roughly
    /// 2^-(24 - `bucket_bits`) and causing far more spurious "same tag"
    /// overwrites than the nominal 24-bit collision rate predicts.
    #[inline]
    fn tag(hash: &[u8; 16]) -> u32 {
        let top = u32::from_be_bytes(hash[8..12].try_into().unwrap());
        top >> 8 // top 24 bits
    }

    /// Insert a (hash → `pack_id`, offset) mapping.
    ///
    /// # Errors
    /// Returns `InsertError::TableFull` if the table has less than 25% free slots
    /// and the hash is not already present (overwrites are always allowed).
    pub fn insert(&mut self, hash: &[u8; 16], pack_id: u8, offset: u64) -> Result<(), InsertError> {
        let tag = Self::tag(hash);
        let mut bucket = self.bucket(hash);

        loop {
            let slot = self.slots[bucket];
            if slot.is_empty() {
                // Only reject when actually inserting into a new slot
                let threshold = self.capacity.wrapping_mul(3) / 4;
                if self.len >= threshold {
                    return Err(InsertError::TableFull);
                }
                self.slots[bucket] = IndexSlot::new(tag, pack_id, offset);
                self.len = self.len.wrapping_add(1);
                return Ok(());
            }
            // If same tag already exists at this bucket, overwrite
            // (same hash, different offset after repack)
            if slot.tag() == tag {
                self.slots[bucket] = IndexSlot::new(tag, pack_id, offset);
                return Ok(());
            }
            bucket = bucket.wrapping_add(1) & self.mask as usize;
        }
    }

    /// Look up a hash in the index.
    /// Returns `(pack_id, offset)` if found, `None` if absent.
    ///
    /// An empty slot terminates the probe — this is correct because:
    /// 1. Class A tables are write-once with no deletes (no tombstones needed).
    /// 2. An empty slot means the key was never inserted.
    #[inline]
    #[must_use]
    pub fn lookup(&self, hash: &[u8; 16]) -> Option<(u8, u64)> {
        self.lookup_all(hash).next()
    }

    /// Look up all candidate offsets for a hash, yielding tag collisions.
    ///
    /// The caller **must** verify each candidate against the caller-requested
    /// hash. Tag collisions (0.1% at 24-bit tags) surface as candidates here;
    /// only the one whose stored hash matches the request is valid.
    ///
    /// Yields `(pack_id, offset)` for each slot whose tag matches, then
    /// terminates at the first empty slot or after `capacity` probes.
    #[inline]
    #[must_use]
    pub fn lookup_all(&self, hash: &[u8; 16]) -> LookupIter<'_> {
        let tag = Self::tag(hash);
        let bucket = self.bucket(hash);
        LookupIter {
            slots: &self.slots,
            tag,
            bucket,
            mask: self.mask as usize,
            remaining: self.capacity as usize,
        }
    }

    /// Number of occupied slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        (self.capacity as usize)
            .wrapping_mul(8)
            .wrapping_add(std::mem::size_of::<Self>())
    }

    /// Serialize the index to bytes for persistence.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let byte_len = 8_usize.wrapping_add((self.capacity as usize).wrapping_mul(8));
        let mut buf = Vec::with_capacity(byte_len);
        buf.extend_from_slice(&u64::from(self.capacity).to_le_bytes());
        for slot in &self.slots {
            buf.extend_from_slice(&slot.0.to_le_bytes());
        }
        buf
    }

    /// Deserialize an index from bytes.
    ///
    /// # Errors
    /// Returns `DeserializationError::TooShort` if the data is too short,
    /// or `DeserializationError::InvalidCapacity` if the capacity is not
    /// a power of two or is less than 16.
    ///
    /// # Panics
    /// Panics if the 8-byte capacity header cannot be read (guaranteed by the length check).
    pub fn deserialize(data: &[u8]) -> Result<Self, DeserializationError> {
        if data.len() < 8 {
            return Err(DeserializationError::TooShort);
        }
        let capacity_wire = u64::from_le_bytes(data[..8].try_into().unwrap());
        let capacity = u32::try_from(capacity_wire).map_err(|_| DeserializationError::TooShort)?;
        let capacity_usize = capacity as usize;

        // Reject unsupported capacities: must be power of two and >= 16
        if capacity < 16 || !capacity.is_power_of_two() {
            return Err(DeserializationError::InvalidCapacity);
        }

        let expected_len = 8_usize.wrapping_add(capacity_usize.wrapping_mul(8));
        if data.len() < expected_len {
            return Err(DeserializationError::TooShort);
        }

        let mut slots = Vec::with_capacity(capacity_usize);
        let mut len: u32 = 0;
        for i in 0..capacity_usize {
            let offset = 8_usize.wrapping_add(i.wrapping_mul(8));
            let val = u64::from_le_bytes(data[offset..offset.wrapping_add(8)].try_into().unwrap());
            let slot = IndexSlot(val);
            if !slot.is_empty() {
                len = len.wrapping_add(1);
            }
            slots.push(slot);
        }

        let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());

        Ok(Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift,
            slots,
            len,
        })
    }
}

#[derive(Debug)]
pub enum InsertError {
    TableFull,
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableFull => write!(f, "index table too full"),
        }
    }
}

impl std::error::Error for InsertError {}

#[derive(Debug)]
pub enum DeserializationError {
    TooShort,
    InvalidCapacity,
}

impl std::fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "data too short"),
            Self::InvalidCapacity => {
                write!(f, "capacity must be a power of two and >= 16")
            }
        }
    }
}

impl std::error::Error for DeserializationError {}

/// Iterator over candidate offsets for a hash lookup.
///
/// Yields `(pack_id, offset)` for each slot whose 24-bit tag matches,
/// terminating at the first empty slot or after the full capacity is probed.
pub struct LookupIter<'a> {
    slots: &'a [IndexSlot],
    tag: u32,
    bucket: usize,
    mask: usize,
    remaining: usize,
}

impl Iterator for LookupIter<'_> {
    type Item = (u8, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            self.remaining = self.remaining.wrapping_sub(1);
            let slot = self.slots[self.bucket];
            if slot.is_empty() {
                return None;
            }
            let current = self.bucket;
            self.bucket = self.bucket.wrapping_add(1) & self.mask;
            if slot.tag() == self.tag {
                return Some((slot.pack_id(), slot.offset()));
            }
            let _ = current;
        }
        None
    }
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;

    fn test_hash(byte: u8) -> [u8; 16] {
        let mut h = [0u8; 16];
        h[0] = byte;
        h
    }

    #[test]
    fn test_slot_packing() {
        let slot = IndexSlot::new(0x00AB_CDEF, 42, 0x1234_5678);
        assert_eq!(slot.tag(), 0x00AB_CDEF);
        assert_eq!(slot.pack_id(), 42);
        assert_eq!(slot.offset(), 0x1234_5678);
        assert!(!slot.is_empty());
    }

    #[test]
    fn test_slot_empty() {
        let slot = IndexSlot::empty();
        assert!(slot.is_empty());
        assert_eq!(slot.tag(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut index = LossyIndex::new(128);
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);
        let h3 = test_hash(0xFF);

        index.insert(&h1, 0, 100).unwrap();
        index.insert(&h2, 1, 200).unwrap();
        index.insert(&h3, 0, 999).unwrap();

        assert_eq!(index.len(), 3);
        assert_eq!(index.lookup(&h1), Some((0, 100)));
        assert_eq!(index.lookup(&h2), Some((1, 200)));
        assert_eq!(index.lookup(&h3), Some((0, 999)));
        assert_eq!(index.lookup(&[0xFE; 16]), None);
    }

    #[test]
    fn test_linear_probing() {
        let mut index = LossyIndex::new(16); // small table
        for i in 0..10u8 {
            let mut h = [0u8; 16];
            h[0] = i;
            h[1] = 0xFF; // different second byte to avoid tag collisions
            index.insert(&h, 0, u64::from(i) * 100).unwrap();
        }
        for i in 0..10u8 {
            let mut h = [0u8; 16];
            h[0] = i;
            h[1] = 0xFF;
            assert!(index.lookup(&h).is_some());
        }
    }

    #[test]
    fn test_empty_terminates_probe() {
        let index = LossyIndex::new(16);
        let h = test_hash(0x42);
        assert_eq!(index.lookup(&h), None);
    }

    #[test]
    fn test_table_full_returns_error() {
        let mut index = LossyIndex::new(16); // capacity 16, threshold 12
        for i in 0..12u64 {
            let h = splitmix_hash(i);
            index.insert(&h, 0, i).unwrap();
        }
        assert_eq!(index.len(), 12);
        let h = splitmix_hash(999);
        assert!(matches!(
            index.insert(&h, 0, 999),
            Err(InsertError::TableFull)
        ));
    }

    #[test]
    fn test_is_empty_and_memory_usage() {
        let index = LossyIndex::new(128);
        assert!(index.is_empty());
        assert_eq!(
            index.memory_usage(),
            128 * 8 + std::mem::size_of::<LossyIndex>()
        );

        let mut index = LossyIndex::new(128);
        index.insert(&test_hash(0x01), 0, 1).unwrap();
        assert!(!index.is_empty());
    }

    #[test]
    fn test_overwrite_same_hash() {
        let mut index = LossyIndex::new(128);
        let h = test_hash(0x01);
        index.insert(&h, 0, 100).unwrap();
        index.insert(&h, 1, 200).unwrap(); // overwrite
        assert_eq!(index.len(), 1);
        assert_eq!(index.lookup(&h), Some((1, 200)));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut index = LossyIndex::new(128);
        for i in 0..50u8 {
            let mut h = [0u8; 16];
            h[0] = i;
            index.insert(&h, i % 3, u64::from(i) * 1000 + 1).unwrap();
        }

        let bytes = index.serialize();
        let restored = LossyIndex::deserialize(&bytes).unwrap();

        assert_eq!(restored.len(), index.len());
        for i in 0..50u8 {
            let mut h = [0u8; 16];
            h[0] = i;
            assert_eq!(restored.lookup(&h), index.lookup(&h));
        }
    }

    #[test]
    fn test_power_of_two_capacity() {
        let index = LossyIndex::new(100);
        assert_eq!(index.capacity, 128);
        let index = LossyIndex::new(128);
        assert_eq!(index.capacity, 128);
        let index = LossyIndex::new(129);
        assert_eq!(index.capacity, 256);
    }

    // jscpd:ignore-start
    // False-positive match against storage.rs's NodeRef::data() — token-shape
    // coincidence (both short, similar brace/operator density), not related
    // logic. Nothing to extract: one's a hash mixer, one's an enum accessor.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
    // jscpd:ignore-end

    fn splitmix_hash(i: u64) -> [u8; 16] {
        let a = splitmix64(i);
        let b = splitmix64(a ^ 0x7372_9A1E_4288_1F7D);
        let mut h = [0u8; 16];
        h[..8].copy_from_slice(&a.to_le_bytes());
        h[8..].copy_from_slice(&b.to_le_bytes());
        h
    }

    /// Regression test for a shift-formula bug: with `capacity` a `u32`,
    /// `64 - capacity.leading_zeros()` does not equal `64 - log2(capacity)`,
    /// so at larger capacities the bucket function only used a fraction of
    /// its intended hash bits, collapsing most of the table into a handful
    /// of buckets and causing O(n) probes per insert. `trailing_zeros` gives
    /// the correct shift for any power-of-two capacity.
    #[test]
    fn test_shift_uses_full_bucket_range() {
        for &capacity in &[16u32, 1024, 65536, 131_072, 262_144] {
            let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());
            assert_eq!(shift, 64 - capacity.ilog2());
        }
    }

    /// Regression test: at scale, insert must not silently drop distinct
    /// entries via spurious tag collisions caused by the tag and bucket
    /// being derived from overlapping hash bits.
    #[test]
    fn test_insert_preserves_distinct_entries_at_scale() {
        let n = 50_000usize;
        let mut index = LossyIndex::new(n * 2);
        for i in 0..n {
            let h = splitmix_hash(i as u64);
            index.insert(&h, 0, i as u64).unwrap();
        }
        // No distinct entry should have been silently overwritten.
        assert_eq!(index.len(), n);
        for i in 0..n {
            let h = splitmix_hash(i as u64);
            assert_eq!(index.lookup(&h), Some((0, i as u64)));
        }
    }

    #[test]
    fn test_deserialize_errors() {
        // TooShort: data shorter than 8 bytes
        assert!(matches!(
            LossyIndex::deserialize(&[0u8; 4]),
            Err(DeserializationError::TooShort)
        ));
        // TooShort: capacity header present but slots truncated
        let mut buf = vec![0u8; 16];
        buf[..8].copy_from_slice(&16u64.to_le_bytes());
        assert!(matches!(
            LossyIndex::deserialize(&buf),
            Err(DeserializationError::TooShort)
        ));
        // InvalidCapacity: capacity < 16
        let mut buf = vec![0u8; 8 + 8 * 8];
        buf[..8].copy_from_slice(&8u64.to_le_bytes());
        assert!(matches!(
            LossyIndex::deserialize(&buf),
            Err(DeserializationError::InvalidCapacity)
        ));
        // InvalidCapacity: capacity not power of two
        let mut buf = vec![0u8; 8 + 20 * 8];
        buf[..8].copy_from_slice(&20u64.to_le_bytes());
        assert!(matches!(
            LossyIndex::deserialize(&buf),
            Err(DeserializationError::InvalidCapacity)
        ));
    }

    #[test]
    fn test_error_display() {
        assert_eq!(InsertError::TableFull.to_string(), "index table too full");
        assert_eq!(DeserializationError::TooShort.to_string(), "data too short");
        assert_eq!(
            DeserializationError::InvalidCapacity.to_string(),
            "capacity must be a power of two and >= 16"
        );
    }

    #[test]
    fn test_lookup_iter_exhausts_remaining() {
        // Build a 16-slot index with all 16 slots occupied (no empty
        // terminator) by constructing the serialized form directly.
        let capacity: u64 = 16;
        let mut bytes = Vec::with_capacity(8 + 16 * 8);
        bytes.extend_from_slice(&capacity.to_le_bytes());
        for i in 0..16u64 {
            let h = splitmix_hash(i + 5000);
            let tag = LossyIndex::tag(&h);
            let slot = IndexSlot::new(tag, 0, i);
            bytes.extend_from_slice(&slot.0.to_le_bytes());
        }
        let restored = LossyIndex::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 16);
        // Query a hash whose tag does NOT match any slot — the iterator
        // must probe all 16 slots and return None.
        let mut query = [0xFFu8; 16];
        query[8..12].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        assert_eq!(restored.lookup(&query), None);
    }
}
