//! Cache-line bucket storage for lossy record locators.
//!
//! A bucket is one 64-byte cache line containing four 16-byte slots. Each slot
//! keeps a 62-bit fingerprint and a packed `(shard, offset)` locator. The
//! record header remains the authority for full 128-bit identity: callers
//! verify a candidate before returning it or replacing it.
//!
//! The table has two candidate buckets per fingerprint. It intentionally does
//! not perform record I/O or decide identity equality; that belongs to the
//! storage layer, which can verify a candidate against its authoritative pack.

use std::hint::spin_loop;
use std::ops::Deref;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

const SLOT_COUNT: usize = 4;
const MAX_OVERFLOW_PER_BUCKET: usize = 1;
const SLOT_STATE_MASK: u64 = 0b11;
const SLOT_EMPTY: u64 = 0;
const SLOT_LIVE: u64 = 1;
const SLOT_REMOVED: u64 = 2;
const SLOT_LOCKED: u64 = 3;
const FINGERPRINT_MASK: u64 = (1 << 62) - 1;
const OFFSET_MASK: u64 = (1 << 48) - 1;
const SNAPSHOT_MAGIC: [u8; 8] = *b"MTXRBK01";
const SNAPSHOT_HEADER_LEN: usize = 32;
const SNAPSHOT_ENTRY_LEN: usize = 24;

/// One locator returned by a fingerprint lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketCandidate {
    /// Fingerprint stored in the slot. This only narrows the candidate set.
    pub fingerprint: u64,
    /// Shard slot that owns the record.
    pub shard_slot: u16,
    /// Byte offset of the record frame.
    pub offset: u64,
    /// Whether the indexed frame is an explicit removal version.
    pub removed: bool,
    /// Stable address used to update or undo this candidate.
    pub location: SlotLocation,
}

/// Stable slot address inside a bucket table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotLocation {
    /// Slot inside one of the cache-line buckets.
    Bucket {
        /// Bucket number.
        bucket: u32,
        /// Slot number within the four-way bucket.
        slot: u8,
    },
    /// Slot in the small overflow area used when both choices are full.
    Overflow(u32),
}

/// Result of inserting or replacing one locator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketUndo {
    /// Location changed by the operation.
    pub location: SlotLocation,
    /// Previous raw fingerprint/state word.
    old_tag_state: u64,
    /// Previous packed locator.
    old_locator: u64,
    /// True when the insertion created a previously empty slot.
    pub was_empty: bool,
}

#[derive(Debug)]
struct Slot {
    tag_state: AtomicU64,
    locator: AtomicU64,
}

impl Slot {
    fn new() -> Self {
        Self {
            tag_state: AtomicU64::new(SLOT_EMPTY),
            locator: AtomicU64::new(0),
        }
    }

    fn snapshot(&self, location: SlotLocation) -> Option<BucketCandidate> {
        loop {
            let before = self.tag_state.load(Ordering::Acquire);
            let state = before & SLOT_STATE_MASK;
            if state == SLOT_LOCKED {
                spin_loop();
                continue;
            }
            if state == SLOT_EMPTY {
                return None;
            }
            let packed = self.locator.load(Ordering::Relaxed);
            if self.tag_state.load(Ordering::Acquire) != before {
                continue;
            }
            return Some(BucketCandidate {
                fingerprint: before >> 2,
                shard_slot: (packed >> 48) as u16,
                offset: packed & OFFSET_MASK,
                removed: state == SLOT_REMOVED,
                location,
            });
        }
    }

    fn write(&self, fingerprint: u64, packed: u64, removed: bool) -> (u64, u64) {
        let next = ((fingerprint & FINGERPRINT_MASK) << 2)
            | if removed { SLOT_REMOVED } else { SLOT_LIVE };
        loop {
            let old = self.tag_state.load(Ordering::Acquire);
            if old & SLOT_STATE_MASK == SLOT_LOCKED {
                spin_loop();
                continue;
            }
            let locked = (old & !SLOT_STATE_MASK) | SLOT_LOCKED;
            if self
                .tag_state
                .compare_exchange_weak(old, locked, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let old_locator = self.locator.load(Ordering::Relaxed);
            self.locator.store(packed, Ordering::Relaxed);
            self.tag_state.store(next, Ordering::Release);
            return (old, old_locator);
        }
    }

    fn write_if_empty(&self, fingerprint: u64, packed: u64, removed: bool) -> Option<(u64, u64)> {
        let locked = ((fingerprint & FINGERPRINT_MASK) << 2) | SLOT_LOCKED;
        if self
            .tag_state
            .compare_exchange(SLOT_EMPTY, locked, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        self.locator.store(packed, Ordering::Relaxed);
        self.tag_state.store(
            ((fingerprint & FINGERPRINT_MASK) << 2)
                | if removed { SLOT_REMOVED } else { SLOT_LIVE },
            Ordering::Release,
        );
        Some((SLOT_EMPTY, 0))
    }

    fn restore(&self, tag_state: u64, locator: u64) {
        loop {
            let old = self.tag_state.load(Ordering::Acquire);
            if old & SLOT_STATE_MASK == SLOT_LOCKED {
                spin_loop();
                continue;
            }
            let locked = (old & !SLOT_STATE_MASK) | SLOT_LOCKED;
            if self
                .tag_state
                .compare_exchange_weak(old, locked, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            self.locator.store(locator, Ordering::Relaxed);
            self.tag_state.store(tag_state, Ordering::Release);
            return;
        }
    }
}

/// Four packed slots aligned as one cache line.
#[repr(C, align(64))]
#[derive(Debug)]
struct Bucket {
    slots: [Slot; SLOT_COUNT],
}

enum SlotHandle<'a> {
    Bucket(&'a Slot),
    Overflow(Arc<Slot>),
}

impl Deref for SlotHandle<'_> {
    type Target = Slot;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Bucket(slot) => slot,
            Self::Overflow(slot) => slot,
        }
    }
}

impl Bucket {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| Slot::new()),
        }
    }
}

/// Two-choice cache-line table for lossy `(collection, node id)` locators.
#[derive(Debug)]
pub struct RawBucketTable {
    buckets: Vec<Bucket>,
    overflow: Mutex<Vec<Arc<Slot>>>,
    len: AtomicU32,
    seed: u64,
}

fn parse_snapshot_header(bytes: &[u8]) -> Result<(u32, u64, usize), BucketSnapshotError> {
    let header = bytes
        .get(..SNAPSHOT_HEADER_LEN)
        .ok_or(BucketSnapshotError::TooShort)?;
    if header.get(..8) != Some(SNAPSHOT_MAGIC.as_slice()) {
        return Err(BucketSnapshotError::BadMagic);
    }
    let version = u32::from_le_bytes(
        header
            .get(8..12)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::TooShort)?,
    );
    if version != 1 {
        return Err(BucketSnapshotError::UnsupportedVersion(version));
    }
    let bucket_count = u32::from_le_bytes(
        header
            .get(12..16)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::TooShort)?,
    );
    if bucket_count == 0 || !bucket_count.is_power_of_two() {
        return Err(BucketSnapshotError::InvalidDimensions);
    }
    let seed = u64::from_le_bytes(
        header
            .get(16..24)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::TooShort)?,
    );
    let count = u64::from_le_bytes(
        header
            .get(24..32)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::TooShort)?,
    );
    let count = usize::try_from(count).map_err(|_| BucketSnapshotError::InvalidDimensions)?;
    Ok((bucket_count, seed, count))
}

fn parse_snapshot_entry(entry: &[u8]) -> Result<(u64, u16, u64, bool), BucketSnapshotError> {
    let fingerprint = u64::from_le_bytes(
        entry
            .get(..8)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::InvalidEntry)?,
    );
    let shard_slot = u16::from_le_bytes(
        entry
            .get(8..10)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::InvalidEntry)?,
    );
    let removed = match entry.get(10) {
        Some(0) => false,
        Some(1) => true,
        _ => return Err(BucketSnapshotError::InvalidEntry),
    };
    if entry.get(11..16) != Some([0; 5].as_slice()) || fingerprint > FINGERPRINT_MASK {
        return Err(BucketSnapshotError::InvalidEntry);
    }
    let record_offset = u64::from_le_bytes(
        entry
            .get(16..24)
            .and_then(|slice| slice.try_into().ok())
            .ok_or(BucketSnapshotError::InvalidEntry)?,
    );
    if record_offset > OFFSET_MASK {
        return Err(BucketSnapshotError::InvalidEntry);
    }
    Ok((fingerprint, shard_slot, record_offset, removed))
}

impl RawBucketTable {
    /// Create a table with at least `min_entries` slots.
    #[must_use]
    pub fn new(min_entries: usize, seed: u64) -> Self {
        let needed_buckets = min_entries
            .max(SLOT_COUNT)
            .div_ceil(SLOT_COUNT)
            .next_power_of_two();
        Self {
            buckets: (0..needed_buckets).map(|_| Bucket::new()).collect(),
            overflow: Mutex::new(Vec::new()),
            len: AtomicU32::new(0),
            seed,
        }
    }

    /// Number of slots available in the two-choice buckets.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buckets.len().saturating_mul(SLOT_COUNT)
    }

    /// Number of occupied slots, including overflow slots.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(self.len.load(Ordering::Acquire)).unwrap_or(usize::MAX)
    }

    /// Whether the table has no occupied slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encode a deterministic, checksummed snapshot of the occupied locators.
    /// The snapshot stores fingerprints rather than full keys; pack frames
    /// remain authoritative and callers must verify candidates against them.
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        let mut entries = Vec::with_capacity(self.len());
        for (bucket_index, bucket) in self.buckets.iter().enumerate() {
            for slot in &bucket.slots {
                if let Some(candidate) = slot.snapshot(SlotLocation::Bucket {
                    bucket: u32::try_from(bucket_index).unwrap_or(u32::MAX),
                    slot: 0,
                }) {
                    entries.push(candidate);
                }
            }
        }
        let overflow = self.overflow.lock();
        for (index, slot) in overflow.iter().enumerate() {
            if let Some(candidate) = slot.snapshot(SlotLocation::Overflow(
                u32::try_from(index).unwrap_or(u32::MAX),
            )) {
                entries.push(candidate);
            }
        }
        drop(overflow);
        entries.sort_unstable_by_key(|entry| {
            (
                entry.fingerprint,
                entry.shard_slot,
                entry.offset,
                entry.removed,
            )
        });

        let mut bytes = Vec::with_capacity(
            SNAPSHOT_HEADER_LEN
                .saturating_add(entries.len().saturating_mul(SNAPSHOT_ENTRY_LEN))
                .saturating_add(4),
        );
        bytes.extend_from_slice(&SNAPSHOT_MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(
            &u32::try_from(self.buckets.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(entries.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for entry in entries {
            bytes.extend_from_slice(&entry.fingerprint.to_le_bytes());
            bytes.extend_from_slice(&entry.shard_slot.to_le_bytes());
            bytes.push(u8::from(entry.removed));
            bytes.extend_from_slice(&[0; 5]);
            bytes.extend_from_slice(&entry.offset.to_le_bytes());
        }
        let checksum = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes
    }

    /// Restore a table snapshot, rejecting malformed dimensions, entries, or
    /// checksums before publishing any state.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot is truncated, has a
    /// bad magic/version/checksum, has inconsistent dimensions, or contains an
    /// entry that cannot be installed.
    pub fn from_snapshot(bytes: &[u8]) -> Result<Self, BucketSnapshotError> {
        let (bucket_count, seed, count) = parse_snapshot_header(bytes)?;
        let expected_len = SNAPSHOT_HEADER_LEN
            .checked_add(
                count
                    .checked_mul(SNAPSHOT_ENTRY_LEN)
                    .ok_or(BucketSnapshotError::InvalidDimensions)?,
            )
            .and_then(|len| len.checked_add(4))
            .ok_or(BucketSnapshotError::InvalidDimensions)?;
        if bytes.len() != expected_len {
            return Err(BucketSnapshotError::InvalidDimensions);
        }
        let checksum_offset = bytes
            .len()
            .checked_sub(4)
            .ok_or(BucketSnapshotError::TooShort)?;
        let expected_crc = u32::from_le_bytes(
            bytes
                .get(checksum_offset..)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(BucketSnapshotError::TooShort)?,
        );
        if crc32fast::hash(&bytes[..checksum_offset]) != expected_crc {
            return Err(BucketSnapshotError::ChecksumMismatch);
        }
        let minimum_entries = usize::try_from(bucket_count)
            .ok()
            .and_then(|count| count.checked_mul(SLOT_COUNT))
            .ok_or(BucketSnapshotError::InvalidDimensions)?;
        let maximum_entries = minimum_entries
            .checked_add(
                usize::try_from(bucket_count)
                    .ok()
                    .and_then(|count| count.checked_mul(MAX_OVERFLOW_PER_BUCKET))
                    .ok_or(BucketSnapshotError::InvalidDimensions)?,
            )
            .ok_or(BucketSnapshotError::InvalidDimensions)?;
        if count > maximum_entries {
            return Err(BucketSnapshotError::InvalidDimensions);
        }
        let table = Self::new(minimum_entries, seed);
        if table.buckets.len() != usize::try_from(bucket_count).unwrap_or(0) {
            return Err(BucketSnapshotError::InvalidDimensions);
        }
        let mut offset = SNAPSHOT_HEADER_LEN;
        for _ in 0..count {
            let end = offset
                .checked_add(SNAPSHOT_ENTRY_LEN)
                .ok_or(BucketSnapshotError::InvalidDimensions)?;
            let entry = bytes
                .get(offset..end)
                .ok_or(BucketSnapshotError::InvalidEntry)?;
            offset = end;
            let (fingerprint, shard_slot, record_offset, removed) = parse_snapshot_entry(entry)?;
            if usize::from(shard_slot) >= crate::shard::MAX_SHARDS
                || table
                    .insert_new(fingerprint, shard_slot, record_offset, removed)
                    .is_none()
            {
                return Err(BucketSnapshotError::InvalidEntry);
            }
        }
        Ok(table)
    }

    /// Number of entries that could not fit in either candidate bucket.
    #[must_use]
    pub fn overflow_len(&self) -> usize {
        self.overflow
            .lock()
            .iter()
            .filter(|slot| slot.tag_state.load(Ordering::Acquire) & SLOT_STATE_MASK != SLOT_EMPTY)
            .count()
    }

    /// Produce a seeded 62-bit fingerprint from the full key hash.
    #[must_use]
    pub fn fingerprint(&self, hash: &[u8; 16]) -> u64 {
        let mut left = [0_u8; 8];
        left.copy_from_slice(&hash[..8]);
        let mut right = [0_u8; 8];
        right.copy_from_slice(&hash[8..]);
        let left = u64::from_le_bytes(left);
        let right = u64::from_le_bytes(right);
        mix64(left ^ mix64(right ^ self.seed) ^ self.seed.rotate_left(29)) & FINGERPRINT_MASK
    }

    /// Return every slot with this fingerprint. Callers must verify candidates
    /// against the full record identity before treating one as a key match.
    #[must_use]
    pub fn candidates(&self, fingerprint: u64) -> Vec<BucketCandidate> {
        let mut result = Vec::new();
        let (first, second) = self.choices(fingerprint);
        let bucket_count = if first == second { 1 } else { 2 };
        for bucket in [first, second].into_iter().take(bucket_count) {
            for (slot_index, slot) in self.buckets[bucket].slots.iter().enumerate() {
                let location = SlotLocation::Bucket {
                    bucket: u32::try_from(bucket).unwrap_or(u32::MAX),
                    slot: u8::try_from(slot_index).unwrap_or(u8::MAX),
                };
                if let Some(candidate) = slot.snapshot(location) {
                    if candidate.fingerprint == fingerprint {
                        result.push(candidate);
                    }
                }
            }
        }
        let overflow = self.overflow.lock();
        for (index, slot) in overflow.iter().enumerate() {
            let location = SlotLocation::Overflow(u32::try_from(index).unwrap_or(u32::MAX));
            if let Some(candidate) = slot.snapshot(location) {
                if candidate.fingerprint == fingerprint {
                    result.push(candidate);
                }
            }
        }
        result
    }

    /// Insert a new key candidate. Exact replacement is a separate operation
    /// so storage can verify full identity when fingerprints collide.
    ///
    /// # Errors
    /// Returns `None` if the locator cannot be represented or the overflow
    /// address space is exhausted.
    pub fn insert_new(
        &self,
        fingerprint: u64,
        shard_slot: u16,
        offset: u64,
        removed: bool,
    ) -> Option<BucketUndo> {
        let packed = pack_locator(shard_slot, offset)?;
        let fingerprint = fingerprint & FINGERPRINT_MASK;
        let (first, second) = self.choices(fingerprint);
        let first_empty = self.first_empty(first);
        let second_empty = if second == first {
            None
        } else {
            self.first_empty(second)
        };
        let location = match (first_empty, second_empty) {
            (Some(a), Some(b)) => {
                if self.bucket_occupancy(first) <= self.bucket_occupancy(second) {
                    a
                } else {
                    b
                }
            }
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => {
                let mut overflow = self.overflow.lock();
                if let Some((index, slot)) = overflow.iter().enumerate().find(|(_, slot)| {
                    slot.tag_state.load(Ordering::Acquire) & SLOT_STATE_MASK == SLOT_EMPTY
                }) {
                    let (old_tag_state, old_locator) =
                        slot.write_if_empty(fingerprint, packed, removed)?;
                    self.len.fetch_add(1, Ordering::Release);
                    return Some(BucketUndo {
                        location: SlotLocation::Overflow(u32::try_from(index).ok()?),
                        old_tag_state,
                        old_locator,
                        was_empty: true,
                    });
                }
                if overflow.len() >= self.buckets.len().saturating_mul(MAX_OVERFLOW_PER_BUCKET) {
                    return None;
                }
                let index = u32::try_from(overflow.len()).ok()?;
                let slot = Arc::new(Slot::new());
                let (old_tag_state, old_locator) =
                    slot.write_if_empty(fingerprint, packed, removed)?;
                overflow.push(slot);
                self.len.fetch_add(1, Ordering::Release);
                return Some(BucketUndo {
                    location: SlotLocation::Overflow(index),
                    old_tag_state,
                    old_locator,
                    was_empty: true,
                });
            }
        };
        let slot = self.slot(location)?;
        let (old_tag_state, old_locator) = slot.write_if_empty(fingerprint, packed, removed)?;
        self.len.fetch_add(1, Ordering::Release);
        Some(BucketUndo {
            location,
            old_tag_state,
            old_locator,
            was_empty: true,
        })
    }

    /// Insert a key or replace the candidate whose full identity matches it.
    ///
    /// `identity_at` reads the authoritative frame header for every matching
    /// fingerprint. Distinct full hashes with the same fingerprint coexist in
    /// different slots; they are never overwritten as if fingerprint equality
    /// proved key equality.
    ///
    /// # Errors
    /// Returns the closure's error if a candidate cannot be verified. Returns
    /// `Ok(None)` if the locator cannot be represented or the overflow address
    /// space is exhausted.
    pub fn insert_with_identity<E>(
        &self,
        hash: &[u8; 16],
        shard_slot: u16,
        offset: u64,
        removed: bool,
        mut identity_at: impl FnMut(&BucketCandidate) -> Result<[u8; 16], E>,
    ) -> Result<Option<BucketUndo>, E> {
        let fingerprint = self.fingerprint(hash);
        for candidate in self.candidates(fingerprint) {
            if identity_at(&candidate)? == *hash {
                return Ok(self.replace(
                    candidate.location,
                    fingerprint,
                    shard_slot,
                    offset,
                    removed,
                ));
            }
        }
        Ok(self.insert_new(fingerprint, shard_slot, offset, removed))
    }

    /// Replace a previously identity-verified candidate.
    ///
    /// # Errors
    /// Returns `None` if the locator cannot be represented or `location` is
    /// not an occupied slot.
    pub fn replace(
        &self,
        location: SlotLocation,
        fingerprint: u64,
        shard_slot: u16,
        offset: u64,
        removed: bool,
    ) -> Option<BucketUndo> {
        let slot = self.slot(location)?;
        let previous = slot.snapshot(location)?;
        if previous.fingerprint != (fingerprint & FINGERPRINT_MASK) {
            return None;
        }
        let packed = pack_locator(shard_slot, offset)?;
        let (old_tag_state, old_locator) = slot.write(fingerprint, packed, removed);
        Some(BucketUndo {
            location,
            old_tag_state,
            old_locator,
            was_empty: false,
        })
    }

    /// Undo one insert or replacement while the owning collection writer lock
    /// is held.
    pub fn rollback(&self, undo: BucketUndo) {
        let Some(slot) = self.slot(undo.location) else {
            return;
        };
        slot.restore(undo.old_tag_state, undo.old_locator);
        if undo.was_empty {
            self.len.fetch_sub(1, Ordering::Release);
        }
    }

    /// Grow to twice the current bucket count and rehash stored fingerprints.
    #[must_use]
    pub fn grow(&self) -> Option<Self> {
        let next_entries = self.capacity().checked_mul(2)?;
        let grown = Self::new(next_entries, self.seed);
        for bucket in &self.buckets {
            for slot in &bucket.slots {
                if let Some(candidate) = slot.snapshot(SlotLocation::Bucket { bucket: 0, slot: 0 })
                {
                    grown.insert_new(
                        candidate.fingerprint,
                        candidate.shard_slot,
                        candidate.offset,
                        candidate.removed,
                    )?;
                }
            }
        }
        let overflow = self.overflow.lock();
        for slot in overflow.iter() {
            if let Some(candidate) = slot.snapshot(SlotLocation::Overflow(0)) {
                grown.insert_new(
                    candidate.fingerprint,
                    candidate.shard_slot,
                    candidate.offset,
                    candidate.removed,
                )?;
            }
        }
        Some(grown)
    }

    fn choices(&self, fingerprint: u64) -> (usize, usize) {
        let mask = self.buckets.len().saturating_sub(1);
        let first = usize::try_from(mix64(fingerprint ^ self.seed)).unwrap_or(0) & mask;
        let mut second = usize::try_from(mix64(
            fingerprint ^ self.seed.rotate_left(17) ^ 0xA076_1D64_78BD_642F,
        ))
        .unwrap_or(0)
            & mask;
        if self.buckets.len() > 1 && second == first {
            second = second.wrapping_add(1) & mask;
        }
        (first, second)
    }

    fn first_empty(&self, bucket: usize) -> Option<SlotLocation> {
        self.buckets[bucket]
            .slots
            .iter()
            .position(|slot| slot.tag_state.load(Ordering::Acquire) & SLOT_STATE_MASK == SLOT_EMPTY)
            .map(|slot| SlotLocation::Bucket {
                bucket: u32::try_from(bucket).unwrap_or(u32::MAX),
                slot: u8::try_from(slot).unwrap_or(u8::MAX),
            })
    }

    fn bucket_occupancy(&self, bucket: usize) -> usize {
        self.buckets[bucket]
            .slots
            .iter()
            .filter(|slot| slot.tag_state.load(Ordering::Acquire) & SLOT_STATE_MASK != SLOT_EMPTY)
            .count()
    }

    fn slot(&self, location: SlotLocation) -> Option<SlotHandle<'_>> {
        match location {
            SlotLocation::Bucket { bucket, slot } => Some(SlotHandle::Bucket(
                self.buckets
                    .get(usize::try_from(bucket).ok()?)?
                    .slots
                    .get(usize::from(slot))?,
            )),
            SlotLocation::Overflow(index) => {
                let overflow = self.overflow.lock();
                Some(SlotHandle::Overflow(Arc::clone(
                    overflow.get(usize::try_from(index).ok()?)?,
                )))
            }
        }
    }
}

/// Snapshot decoding errors for [`RawBucketTable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BucketSnapshotError {
    /// Input is shorter than the fixed header and checksum.
    TooShort,
    /// Input does not contain the bucket snapshot magic.
    BadMagic,
    /// The snapshot version is not supported.
    UnsupportedVersion(u32),
    /// Capacity or encoded length is inconsistent.
    InvalidDimensions,
    /// An entry contains an invalid locator, fingerprint, or reserved field.
    InvalidEntry,
    /// The body checksum did not match.
    ChecksumMismatch,
}

impl std::fmt::Display for BucketSnapshotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => formatter.write_str("bucket snapshot is too short"),
            Self::BadMagic => formatter.write_str("invalid bucket snapshot magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported bucket snapshot version {version}")
            }
            Self::InvalidDimensions => formatter.write_str("invalid bucket snapshot dimensions"),
            Self::InvalidEntry => formatter.write_str("invalid bucket snapshot entry"),
            Self::ChecksumMismatch => formatter.write_str("bucket snapshot checksum mismatch"),
        }
    }
}

impl std::error::Error for BucketSnapshotError {}

fn pack_locator(shard_slot: u16, offset: u64) -> Option<u64> {
    (offset <= OFFSET_MASK).then_some((u64::from(shard_slot) << 48) | offset)
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{Bucket, RawBucketTable, SlotLocation, SNAPSHOT_HEADER_LEN};

    #[test]
    fn bucket_is_one_cache_line_of_four_slots() {
        assert_eq!(std::mem::size_of::<Bucket>(), 64);
        assert_eq!(std::mem::align_of::<Bucket>(), 64);
    }

    #[test]
    fn candidates_replace_and_rollback_keep_locator_state_coherent() {
        let table = RawBucketTable::new(8, 0x5A17);
        let fingerprint = table.fingerprint(&[0xA5; 16]);
        let inserted = table.insert_new(fingerprint, 17, 1 << 40, false).unwrap();
        assert!(inserted.was_empty);
        let candidates = table.candidates(fingerprint);
        let [candidate] = candidates.as_slice() else {
            panic!("expected one candidate")
        };
        assert_eq!((candidate.shard_slot, candidate.offset), (17, 1 << 40));

        let replaced = table
            .replace(inserted.location, fingerprint, 19, 1 << 41, true)
            .unwrap();
        let candidates = table.candidates(fingerprint);
        let [candidate] = candidates.as_slice() else {
            panic!("expected one replacement")
        };
        assert_eq!(
            (candidate.shard_slot, candidate.offset, candidate.removed),
            (19, 1 << 41, true)
        );

        table.rollback(replaced);
        let candidates = table.candidates(fingerprint);
        let [candidate] = candidates.as_slice() else {
            panic!("expected restored candidate")
        };
        assert_eq!(
            (candidate.shard_slot, candidate.offset, candidate.removed),
            (17, 1 << 40, false)
        );
        table.rollback(inserted);
        assert!(table.candidates(fingerprint).is_empty());
        assert!(table.is_empty());
    }

    #[test]
    fn overflow_slots_are_reused_after_rollback() {
        let table = RawBucketTable::new(4, 1);
        let mut entries = Vec::new();
        let mut by_bucket = std::collections::HashMap::<usize, Vec<u64>>::new();
        let mut collision_set = None;
        for seed in 0..10_000_u64 {
            let fingerprint = seed;
            let bucket = table.choices(fingerprint).0;
            let fingerprints = by_bucket.entry(bucket).or_default();
            fingerprints.push(fingerprint);
            if fingerprints.len() == 5 {
                collision_set = Some(fingerprints.clone());
                break;
            }
        }
        let fingerprints = collision_set.expect("five keys share one first-choice bucket");
        for (i, fingerprint) in fingerprints.iter().copied().enumerate() {
            entries.push(
                table
                    .insert_new(fingerprint, 0, u64::try_from(i).unwrap(), false)
                    .unwrap(),
            );
        }
        assert!(table.overflow_len() > 0);
        let overflow_undo = entries
            .into_iter()
            .find(|undo| matches!(undo.location, SlotLocation::Overflow(_)))
            .unwrap();
        table.rollback(overflow_undo);
        assert_eq!(table.overflow_len(), 0);
        let reused = table.insert_new(fingerprints[4], 0, 99, false).unwrap();
        assert!(matches!(reused.location, SlotLocation::Overflow(_)));
    }

    #[test]
    fn equal_fingerprints_replace_only_the_verified_identity() {
        let table = RawBucketTable::new(16, 0xBEEF);
        let first_hash = [0x11; 16];
        let second_hash = [0x22; 16];
        let fingerprint = table.fingerprint(&second_hash);
        table.insert_new(fingerprint, 1, 100, false).unwrap();
        table.insert_new(fingerprint, 2, 200, false).unwrap();

        let undo = table
            .insert_with_identity(&second_hash, 3, 300, true, |candidate| {
                Ok::<_, ()>(if candidate.offset == 100 {
                    first_hash
                } else {
                    second_hash
                })
            })
            .unwrap()
            .unwrap();
        let candidates = table.candidates(fingerprint);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|candidate| {
            candidate.shard_slot == 1 && candidate.offset == 100 && !candidate.removed
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.shard_slot == 3 && candidate.offset == 300 && candidate.removed
        }));
        table.rollback(undo);
        assert!(table
            .candidates(fingerprint)
            .iter()
            .any(|candidate| candidate.shard_slot == 2 && candidate.offset == 200));
    }

    #[test]
    fn snapshots_round_trip_and_reject_corruption() {
        let table = RawBucketTable::new(32, 0x1234);
        let first = table.fingerprint(&[0x11; 16]);
        let second = table.fingerprint(&[0x22; 16]);
        table.insert_new(first, 7, 1 << 40, false).unwrap();
        table.insert_new(second, 9, 1 << 41, true).unwrap();

        let snapshot = table.snapshot();
        let restored = RawBucketTable::from_snapshot(&snapshot).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.fingerprint(&[0x11; 16]), first);
        assert_eq!(restored.candidates(first)[0].offset, 1 << 40);
        assert!(restored.candidates(second)[0].removed);
        assert_eq!(restored.snapshot(), snapshot);

        let mut corrupt = snapshot;
        corrupt[SNAPSHOT_HEADER_LEN + 10] = 2;
        let checksum_offset = corrupt.len() - 4;
        let checksum = crc32fast::hash(&corrupt[..checksum_offset]);
        corrupt[checksum_offset..].copy_from_slice(&checksum.to_le_bytes());
        assert!(matches!(
            RawBucketTable::from_snapshot(&corrupt),
            Err(super::BucketSnapshotError::InvalidEntry)
        ));
    }
}
