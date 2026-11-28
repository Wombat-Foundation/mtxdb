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
use std::io::{self, Write};
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

fn write_snapshot_header(
    header: &mut [u8; SNAPSHOT_HEADER_LEN],
    bucket_count: usize,
    seed: u64,
    entry_count: usize,
) {
    header[..8].copy_from_slice(&SNAPSHOT_MAGIC);
    header[8..12].copy_from_slice(&1_u32.to_le_bytes());
    header[12..16].copy_from_slice(
        &u32::try_from(bucket_count)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    header[16..24].copy_from_slice(&seed.to_le_bytes());
    header[24..32].copy_from_slice(&u64::try_from(entry_count).unwrap_or(u64::MAX).to_le_bytes());
}

fn encode_snapshot_entry(encoded: &mut [u8; SNAPSHOT_ENTRY_LEN], entry: &BucketCandidate) {
    encoded[..8].copy_from_slice(&entry.fingerprint.to_le_bytes());
    encoded[8..10].copy_from_slice(&entry.shard_slot.to_le_bytes());
    encoded[10] = u8::from(entry.removed);
    encoded[11..16].fill(0);
    encoded[16..24].copy_from_slice(&entry.offset.to_le_bytes());
}

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

impl BucketUndo {
    /// Whether the operation occupied a previously empty slot.
    #[must_use]
    pub const fn was_empty(&self) -> bool {
        self.was_empty
    }
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

    /// Copy this slot's value without going through a serialized snapshot.
    /// Spins past a concurrent writer so the copied pair is internally
    /// consistent; callers clone under the collection writer lock, so this
    /// normally succeeds on the first read.
    fn clone_direct(&self) -> Self {
        loop {
            let tag_state = self.tag_state.load(Ordering::Acquire);
            if tag_state & SLOT_STATE_MASK == SLOT_LOCKED {
                spin_loop();
                continue;
            }
            let locator = self.locator.load(Ordering::Acquire);
            if self.tag_state.load(Ordering::Acquire) == tag_state {
                return Self {
                    tag_state: AtomicU64::new(tag_state),
                    locator: AtomicU64::new(locator),
                };
            }
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
            let packed = self.locator.load(Ordering::Acquire);
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
                .compare_exchange_weak(old, locked, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let old_locator = self.locator.load(Ordering::Relaxed);
            self.locator.store(packed, Ordering::Release);
            self.tag_state.store(next, Ordering::Release);
            return (old, old_locator);
        }
    }

    fn write_if_empty(&self, fingerprint: u64, packed: u64, removed: bool) -> Option<(u64, u64)> {
        let locked = ((fingerprint & FINGERPRINT_MASK) << 2) | SLOT_LOCKED;
        if self
            .tag_state
            .compare_exchange(SLOT_EMPTY, locked, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        self.locator.store(packed, Ordering::Release);
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
                .compare_exchange_weak(old, locked, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            self.locator.store(locator, Ordering::Release);
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

    fn clone_direct(&self) -> Self {
        Self {
            slots: std::array::from_fn(|index| self.slots[index].clone_direct()),
        }
    }
}

/// Two-choice cache-line table for lossy `(collection, node id)` locators.
#[derive(Debug)]
pub struct RawBucketTable {
    buckets: Vec<Bucket>,
    overflow: Mutex<Vec<Arc<Slot>>>,
    /// Number of occupied overflow locators; zero lets readers skip the mutex.
    overflow_len: AtomicU32,
    len: AtomicU32,
    /// Number of occupied live (non-removal) locators.
    live_len: AtomicU32,
    seed: u64,
}

/// Allocation-free iterator over occupied table slots.
pub struct BucketEntries<'a> {
    table: &'a RawBucketTable,
    bucket_index: usize,
    slot_index: usize,
    overflow_entries: Option<std::vec::IntoIter<BucketCandidate>>,
    overflow_scanned: bool,
}

impl Iterator for BucketEntries<'_> {
    type Item = BucketCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.bucket_index < self.table.buckets.len() {
                let bucket = self.bucket_index;
                let slot = self.slot_index;
                self.slot_index = self.slot_index.saturating_add(1);
                if self.slot_index == SLOT_COUNT {
                    self.slot_index = 0;
                    self.bucket_index = self.bucket_index.saturating_add(1);
                }
                let location = SlotLocation::Bucket {
                    bucket: u32::try_from(bucket).ok()?,
                    slot: u8::try_from(slot).ok()?,
                };
                if let Some(entry) = self.table.buckets[bucket].slots[slot].snapshot(location) {
                    return Some(entry);
                }
                continue;
            }
            if !self.overflow_scanned {
                self.overflow_entries = Some(self.table.overflow_entries(None));
                self.overflow_scanned = true;
            }
            return self.overflow_entries.as_mut()?.next();
        }
    }
}

/// Allocation-free iterator over a fingerprint's candidate slots.
pub struct BucketCandidates<'a> {
    table: &'a RawBucketTable,
    fingerprint: u64,
    choices: [usize; 2],
    choice_count: usize,
    choice_index: usize,
    slot_index: usize,
    overflow_entries: Option<std::vec::IntoIter<BucketCandidate>>,
    overflow_scanned: bool,
}

impl Iterator for BucketCandidates<'_> {
    type Item = BucketCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.choice_index < self.choice_count {
                let bucket = self.choices[self.choice_index];
                let slot_index = self.slot_index;
                self.slot_index = self.slot_index.saturating_add(1);
                if self.slot_index == SLOT_COUNT {
                    self.slot_index = 0;
                    self.choice_index = self.choice_index.saturating_add(1);
                }
                let location = SlotLocation::Bucket {
                    bucket: u32::try_from(bucket).ok()?,
                    slot: u8::try_from(slot_index).ok()?,
                };
                if let Some(candidate) =
                    self.table.buckets[bucket].slots[slot_index].snapshot(location)
                {
                    if candidate.fingerprint == self.fingerprint {
                        return Some(candidate);
                    }
                }
                continue;
            }
            if !self.overflow_scanned {
                self.overflow_entries = Some(self.table.overflow_entries(Some(self.fingerprint)));
                self.overflow_scanned = true;
            }
            return self.overflow_entries.as_mut()?.next();
        }
    }
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

fn validated_snapshot_layout(
    bytes: &[u8],
    verify_checksum: bool,
) -> Result<(usize, u64, usize, usize), BucketSnapshotError> {
    let (bucket_count, seed, count) = parse_snapshot_header(bytes)?;
    let checksum_offset = bytes
        .len()
        .checked_sub(4)
        .ok_or(BucketSnapshotError::TooShort)?;
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
    if verify_checksum {
        let expected_crc = u32::from_le_bytes(
            bytes
                .get(checksum_offset..)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(BucketSnapshotError::TooShort)?,
        );
        if crc32fast::hash(&bytes[..checksum_offset]) != expected_crc {
            return Err(BucketSnapshotError::ChecksumMismatch);
        }
    }
    let bucket_count =
        usize::try_from(bucket_count).map_err(|_| BucketSnapshotError::InvalidDimensions)?;
    let max_entries = bucket_count
        .checked_mul(SLOT_COUNT + MAX_OVERFLOW_PER_BUCKET)
        .ok_or(BucketSnapshotError::InvalidDimensions)?;
    if count > max_entries {
        return Err(BucketSnapshotError::InvalidDimensions);
    }
    Ok((bucket_count, seed, count, checksum_offset))
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
            overflow_len: AtomicU32::new(0),
            len: AtomicU32::new(0),
            live_len: AtomicU32::new(0),
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

    /// Number of occupied locators that are not removal markers.
    #[must_use]
    pub fn live_len(&self) -> usize {
        usize::try_from(self.live_len.load(Ordering::Acquire)).unwrap_or(usize::MAX)
    }

    /// Whether the table has no occupied slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy every occupied locator in table order.
    #[must_use]
    pub fn entries(&self) -> Vec<BucketCandidate> {
        self.iter_entries().collect()
    }

    /// Visit occupied entries without materializing a table-sized vector.
    /// Return `false` from the callback to stop early.
    pub fn for_each_entry(&self, mut visit: impl FnMut(BucketCandidate) -> bool) -> bool {
        for (bucket_index, bucket) in self.buckets.iter().enumerate() {
            for (slot_index, slot) in bucket.slots.iter().enumerate() {
                let location = SlotLocation::Bucket {
                    bucket: u32::try_from(bucket_index).unwrap_or(u32::MAX),
                    slot: u8::try_from(slot_index).unwrap_or(u8::MAX),
                };
                if let Some(candidate) = slot.snapshot(location) {
                    if !visit(candidate) {
                        return false;
                    }
                }
            }
        }
        if self.overflow_len.load(Ordering::Acquire) == 0 {
            return true;
        }
        let overflow = self.overflow.lock();
        for (index, slot) in overflow.iter().enumerate() {
            if let Some(candidate) = slot.snapshot(SlotLocation::Overflow(
                u32::try_from(index).unwrap_or(u32::MAX),
            )) {
                if !visit(candidate) {
                    return false;
                }
            }
        }
        true
    }

    /// Iterate occupied entries. The common case performs no heap allocation;
    /// the overflow handles are copied only when overflow entries exist.
    pub fn iter_entries(&self) -> BucketEntries<'_> {
        BucketEntries {
            table: self,
            bucket_index: 0,
            slot_index: 0,
            overflow_entries: None,
            overflow_scanned: false,
        }
    }

    /// Copy the table in place, preserving every slot's location.
    ///
    /// Unlike [`Self::snapshot`] followed by [`Self::from_snapshot`], this is
    /// infallible and insertion-order independent: overflow slots keep their
    /// `SlotLocation::Overflow` indices, which `remap_slots` depends on.
    #[must_use]
    pub fn clone_direct(&self) -> Self {
        Self {
            buckets: self.buckets.iter().map(Bucket::clone_direct).collect(),
            overflow: Mutex::new(
                self.overflow
                    .lock()
                    .iter()
                    .map(|slot| Arc::new(slot.clone_direct()))
                    .collect(),
            ),
            overflow_len: AtomicU32::new(self.overflow_len.load(Ordering::Acquire)),
            len: AtomicU32::new(self.len.load(Ordering::Acquire)),
            live_len: AtomicU32::new(self.live_len.load(Ordering::Acquire)),
            seed: self.seed,
        }
    }

    /// Encoded snapshot length without building the snapshot.
    #[must_use]
    pub fn snapshot_len(&self) -> usize {
        SNAPSHOT_HEADER_LEN
            .saturating_add(
                usize::try_from(self.len.load(Ordering::Acquire))
                    .unwrap_or(usize::MAX)
                    .saturating_mul(SNAPSHOT_ENTRY_LEN),
            )
            .saturating_add(4)
    }

    /// Whether any occupied slot points at `shard_slot`, stopping at the first
    /// match instead of copying the table.
    #[must_use]
    pub fn references_slot(&self, shard_slot: u16) -> bool {
        let mut found = false;
        self.for_each_entry(|candidate| {
            found = candidate.shard_slot == shard_slot;
            !found
        });
        found
    }

    /// Occupied entries in the deterministic snapshot order.
    fn sorted_entries(&self) -> Vec<BucketCandidate> {
        let mut entries = self.entries();
        entries.sort_unstable_by_key(|entry| {
            (
                entry.fingerprint,
                entry.shard_slot,
                entry.offset,
                entry.removed,
            )
        });
        entries
    }

    /// Encode a deterministic, checksummed snapshot of the occupied locators.
    /// The snapshot stores fingerprints rather than full keys; pack frames
    /// remain authoritative and callers must verify candidates against them.
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        let entries = self.sorted_entries();
        let mut bytes = Vec::with_capacity(self.snapshot_len());
        // Writing into a `Vec` cannot fail, and live entries are always in
        // range, so encoding a snapshot here is infallible.
        let _ = self.write_snapshot_entries(&entries, &mut bytes);
        bytes
    }

    /// Stream the same bytes [`Self::snapshot`] returns into `writer`, without
    /// building the encoded snapshot on the heap.
    ///
    /// `expected_entries` is the occupied-entry count the caller used to size
    /// its directory entry; a mismatch means the table changed under the
    /// capture, which must never happen while its writer lock is held.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if an occupied entry is out of
    /// range or the count disagrees with `expected_entries`, or any error from
    /// `writer`.
    pub fn write_snapshot_to<W: Write + ?Sized>(
        &self,
        expected_entries: usize,
        writer: &mut W,
    ) -> io::Result<()> {
        let entries = self.sorted_entries();
        if entries.len() != expected_entries {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bucket snapshot entry count changed during capture",
            ));
        }
        for entry in &entries {
            if entry.fingerprint > FINGERPRINT_MASK
                || entry.offset > OFFSET_MASK
                || usize::from(entry.shard_slot) >= crate::shard::MAX_SHARDS
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bucket snapshot entry is out of range",
                ));
            }
        }
        self.write_snapshot_entries(&entries, writer)
    }

    /// The single snapshot encoder shared by [`Self::snapshot`] and
    /// [`Self::write_snapshot_to`].
    fn write_snapshot_entries<W: Write + ?Sized>(
        &self,
        entries: &[BucketCandidate],
        writer: &mut W,
    ) -> io::Result<()> {
        let mut header = [0_u8; SNAPSHOT_HEADER_LEN];
        write_snapshot_header(&mut header, self.buckets.len(), self.seed, entries.len());
        let mut crc = crc32fast::Hasher::new();
        writer.write_all(&header)?;
        crc.update(&header);
        // Batch entries through a stack buffer so the writer and CRC see
        // 4 KiB at a time instead of one 24-byte call per entry.
        let mut chunk = [0_u8; 4096];
        let mut used: usize = 0;
        let mut encoded = [0_u8; SNAPSHOT_ENTRY_LEN];
        for entry in entries {
            let end = used
                .checked_add(SNAPSHOT_ENTRY_LEN)
                .expect("snapshot chunk offset is bounded by its capacity");
            if end > chunk.len() {
                writer.write_all(&chunk[..used])?;
                crc.update(&chunk[..used]);
                used = 0;
            }
            let end = used
                .checked_add(SNAPSHOT_ENTRY_LEN)
                .expect("snapshot chunk offset is bounded by its capacity");
            encode_snapshot_entry(&mut encoded, entry);
            chunk[used..end].copy_from_slice(&encoded);
            used = end;
        }
        if used > 0 {
            writer.write_all(&chunk[..used])?;
            crc.update(&chunk[..used]);
        }
        writer.write_all(&crc.finalize().to_le_bytes())
    }

    /// Restore a table snapshot, rejecting malformed dimensions, entries, or
    /// checksums before publishing any state.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot is truncated, has a
    /// bad magic/version/checksum, has inconsistent dimensions, or contains an
    /// entry that cannot be installed.
    pub fn from_snapshot(bytes: &[u8]) -> Result<Self, BucketSnapshotError> {
        Self::from_snapshot_with_checksum(bytes, true)
    }

    /// Restore a snapshot after an enclosing format has applied its integrity
    /// policy. Dimensions and entries are still validated; only the inner CRC
    /// is skipped because a checkpoint's body CRC already covers these bytes.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot is malformed or an
    /// entry cannot be installed.
    pub(crate) fn from_validated_snapshot(bytes: &[u8]) -> Result<Self, BucketSnapshotError> {
        Self::from_snapshot_with_checksum(bytes, false)
    }

    fn from_snapshot_with_checksum(
        bytes: &[u8],
        verify_checksum: bool,
    ) -> Result<Self, BucketSnapshotError> {
        let (bucket_count, seed, count, _) = validated_snapshot_layout(bytes, verify_checksum)?;
        let minimum_entries = bucket_count
            .checked_mul(SLOT_COUNT)
            .ok_or(BucketSnapshotError::InvalidDimensions)?;
        let table = Self::new(minimum_entries, seed);
        if table.buckets.len() != bucket_count {
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
                    .insert_snapshot_entry(fingerprint, shard_slot, record_offset, removed)
                    .is_none()
            {
                return Err(BucketSnapshotError::InvalidEntry);
            }
        }
        Ok(table)
    }

    /// Validate a snapshot and return its `(bucket_count, entry_count)` without
    /// allocating a bucket table. Checkpoint readers use this before the
    /// owning index is materialized exactly once.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] when the snapshot's dimensions, entries,
    /// or checksum are invalid.
    pub fn validate_snapshot(bytes: &[u8]) -> Result<(usize, usize), BucketSnapshotError> {
        Self::validate_snapshot_with_checksum(bytes, true)
    }

    /// Validate snapshot dimensions and entries, optionally checking its CRC.
    ///
    /// Checkpoint `WriteOnly` mode still validates every structural invariant
    /// while intentionally trusting both the outer body CRC and this snapshot
    /// CRC.
    ///
    /// # Errors
    /// Returns [`BucketSnapshotError`] for malformed dimensions or entries, or
    /// for a checksum mismatch when `verify_checksum` is true.
    pub fn validate_snapshot_with_checksum(
        bytes: &[u8],
        verify_checksum: bool,
    ) -> Result<(usize, usize), BucketSnapshotError> {
        let (bucket_count, _, count, _) = validated_snapshot_layout(bytes, verify_checksum)?;
        let mut offset = SNAPSHOT_HEADER_LEN;
        for _ in 0..count {
            let end = offset
                .checked_add(SNAPSHOT_ENTRY_LEN)
                .ok_or(BucketSnapshotError::InvalidDimensions)?;
            let entry = bytes
                .get(offset..end)
                .ok_or(BucketSnapshotError::InvalidEntry)?;
            offset = end;
            let (_, shard_slot, _, _) = parse_snapshot_entry(entry)?;
            if usize::from(shard_slot) >= crate::shard::MAX_SHARDS {
                return Err(BucketSnapshotError::InvalidEntry);
            }
        }
        Ok((bucket_count, count))
    }

    /// Number of entries that could not fit in either candidate bucket.
    #[must_use]
    pub fn overflow_len(&self) -> usize {
        usize::try_from(self.overflow_len.load(Ordering::Acquire)).unwrap_or(usize::MAX)
    }

    /// Approximate private heap used by the base buckets and overflow slots.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let overflow = self.overflow.lock();
        std::mem::size_of::<Self>()
            .saturating_add(
                self.buckets
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Bucket>()),
            )
            .saturating_add(
                overflow
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Arc<Slot>>()),
            )
            .saturating_add(overflow.len().saturating_mul(std::mem::size_of::<Slot>()))
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
        self.candidate_iter(fingerprint).collect()
    }

    /// Iterate matches directly from the two candidate buckets. The ordinary
    /// path is allocation-free; overflow handles are copied only when needed.
    pub fn candidate_iter(&self, fingerprint: u64) -> BucketCandidates<'_> {
        let (first, second) = self.choices(fingerprint);
        BucketCandidates {
            table: self,
            fingerprint,
            choices: [first, second],
            choice_count: if first == second { 1 } else { 2 },
            choice_index: 0,
            slot_index: 0,
            overflow_entries: None,
            overflow_scanned: false,
        }
    }

    /// Snapshot overflow matches while holding the overflow mutex once.
    /// Overflow is a rare slow path; readers then iterate without lock
    /// thrashing and writers are blocked only for this one scan.
    fn overflow_entries(&self, fingerprint: Option<u64>) -> std::vec::IntoIter<BucketCandidate> {
        if self.overflow_len.load(Ordering::Acquire) == 0 {
            return Vec::new().into_iter();
        }
        let overflow = self.overflow.lock();
        overflow
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let location = SlotLocation::Overflow(u32::try_from(index).ok()?);
                let candidate = slot.snapshot(location)?;
                fingerprint
                    .is_none_or(|expected| candidate.fingerprint == expected)
                    .then_some(candidate)
            })
            .collect::<Vec<_>>()
            .into_iter()
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
                    self.note_new_entry(removed);
                    self.overflow_len.fetch_add(1, Ordering::Release);
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
                self.note_new_entry(removed);
                self.overflow_len.fetch_add(1, Ordering::Release);
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
        self.note_new_entry(removed);
        Some(BucketUndo {
            location,
            old_tag_state,
            old_locator,
            was_empty: true,
        })
    }

    /// Restore a validated snapshot entry. Snapshot order is canonical and can
    /// differ from insertion order; preserve every entry by spilling beyond
    /// the normal overflow budget when that order has a worse greedy layout.
    fn insert_snapshot_entry(
        &self,
        fingerprint: u64,
        shard_slot: u16,
        offset: u64,
        removed: bool,
    ) -> Option<BucketUndo> {
        if let Some(undo) = self.insert_new(fingerprint, shard_slot, offset, removed) {
            return Some(undo);
        }
        let packed = pack_locator(shard_slot, offset)?;
        let mut overflow = self.overflow.lock();
        let index = u32::try_from(overflow.len()).ok()?;
        let slot = Arc::new(Slot::new());
        let (old_tag_state, old_locator) = slot.write_if_empty(fingerprint, packed, removed)?;
        overflow.push(slot);
        self.note_new_entry(removed);
        self.overflow_len.fetch_add(1, Ordering::Release);
        Some(BucketUndo {
            location: SlotLocation::Overflow(index),
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
        self.note_replacement(previous.removed, removed);
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
        let current_removed = slot
            .snapshot(undo.location)
            .is_some_and(|candidate| candidate.removed);
        let prior_state = undo.old_tag_state & SLOT_STATE_MASK;
        let prior_removed = prior_state == SLOT_REMOVED;
        slot.restore(undo.old_tag_state, undo.old_locator);
        if undo.was_empty {
            self.len.fetch_sub(1, Ordering::Release);
            if !current_removed {
                self.live_len.fetch_sub(1, Ordering::Relaxed);
            }
            if matches!(undo.location, SlotLocation::Overflow(_)) {
                self.overflow_len.fetch_sub(1, Ordering::Release);
            }
        } else {
            self.note_replacement(current_removed, prior_removed);
        }
    }

    /// Grow to twice the current bucket count and rehash stored fingerprints.
    #[must_use]
    pub fn grow(&self) -> Option<Self> {
        let next_entries = self.capacity().checked_mul(2)?;
        let grown = Self::new(next_entries, self.seed);
        for (bucket_index, bucket) in self.buckets.iter().enumerate() {
            for (slot_index, slot) in bucket.slots.iter().enumerate() {
                if let Some(candidate) = slot.snapshot(SlotLocation::Bucket {
                    bucket: u32::try_from(bucket_index).ok()?,
                    slot: u8::try_from(slot_index).ok()?,
                }) {
                    grown.insert_snapshot_entry(
                        candidate.fingerprint,
                        candidate.shard_slot,
                        candidate.offset,
                        candidate.removed,
                    )?;
                }
            }
        }
        if self.overflow_len.load(Ordering::Acquire) != 0 {
            let overflow = self.overflow.lock();
            for (index, slot) in overflow.iter().enumerate() {
                if let Some(candidate) = slot.snapshot(SlotLocation::Overflow(
                    u32::try_from(index).unwrap_or(u32::MAX),
                )) {
                    grown.insert_snapshot_entry(
                        candidate.fingerprint,
                        candidate.shard_slot,
                        candidate.offset,
                        candidate.removed,
                    )?;
                }
            }
        }
        Some(grown)
    }

    fn note_new_entry(&self, removed: bool) {
        self.len.fetch_add(1, Ordering::Release);
        if !removed {
            self.live_len.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn note_replacement(&self, old_removed: bool, new_removed: bool) {
        match (old_removed, new_removed) {
            (true, false) => {
                self.live_len.fetch_add(1, Ordering::Relaxed);
            }
            (false, true) => {
                self.live_len.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
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
    use super::{
        mix64, Bucket, RawBucketTable, SlotLocation, FINGERPRINT_MASK, SNAPSHOT_HEADER_LEN,
    };

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
    fn clone_direct_preserves_overflow_locations_without_reinserting() {
        let table = RawBucketTable::new(16, 0x9E37);
        // One fingerprint forces every entry through the same two choices, so
        // the last insertions spill into the overflow area.
        for offset in 0..12 {
            table.insert_new(42, 1, offset, false).unwrap();
        }
        assert!(table.overflow_len() > 0, "fixture must use overflow");

        let before = table.entries();
        let cloned = table.clone_direct();
        assert_eq!(cloned.entries(), before, "locations must be preserved");
        assert_eq!(cloned.len(), table.len());
        assert_eq!(cloned.snapshot(), table.snapshot());
        assert!(before
            .iter()
            .any(|entry| matches!(entry.location, SlotLocation::Overflow(_))));
        assert_eq!(table.snapshot_len(), table.snapshot().len());
    }

    #[test]
    fn grow_preserves_snapshot_overflow_above_live_insertion_limit() {
        let table = RawBucketTable::new(4, 0x19);
        for offset in 0..12 {
            table
                .insert_snapshot_entry(42, 1, offset, false)
                .expect("snapshot restore permits bounded-layout overflow");
        }
        assert!(table.overflow_len() > table.buckets.len());

        let grown = table
            .grow()
            .expect("rehash must preserve every restored entry");
        assert_eq!(grown.len(), table.len());
        assert_eq!(grown.live_len(), table.live_len());
        assert_eq!(grown.entries().len(), 12);
    }

    #[test]
    fn canonical_snapshot_order_restores_tables_at_overflow_limit() {
        let table = RawBucketTable::new(256, 0x6A09_E667);
        let mut inserted = 0_u64;
        // Deterministic pseudo-random fingerprints create varied two-choice
        // graphs; fill until the bounded live insertion policy is exhausted.
        for value in 0_u64..10_000 {
            let fingerprint = mix64(value.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & FINGERPRINT_MASK;
            if table
                .insert_new(fingerprint, 3, value, value % 7 == 0)
                .is_none()
            {
                break;
            }
            inserted += 1;
        }
        assert!(usize::try_from(inserted).unwrap() >= table.capacity());
        assert!(table.overflow_len() > 0);

        let encoded = table.snapshot();
        let restored = RawBucketTable::from_snapshot(&encoded)
            .expect("canonical order may require extra snapshot overflow slots");
        assert_eq!(restored.len(), table.len());
        assert_eq!(restored.live_len(), table.live_len());
        assert_eq!(restored.snapshot(), encoded);
    }

    #[test]
    fn live_count_tracks_insert_replace_rollback_and_snapshot() {
        let table = RawBucketTable::new(16, 9);
        let live = table.insert_new(1, 2, 10, false).unwrap();
        assert_eq!((table.len(), table.live_len()), (1, 1));
        let location = live.location;
        let removed = table.replace(location, 1, 2, 11, true).unwrap();
        assert_eq!((table.len(), table.live_len()), (1, 0));
        table.rollback(removed);
        assert_eq!((table.len(), table.live_len()), (1, 1));
        table.rollback(live);
        assert_eq!((table.len(), table.live_len()), (0, 0));

        table.insert_new(5, 2, 20, true).unwrap();
        let restored = RawBucketTable::from_snapshot(&table.snapshot()).unwrap();
        assert_eq!((restored.len(), restored.live_len()), (1, 0));
    }

    #[test]
    fn snapshots_round_trip_and_reject_corruption() {
        let table = RawBucketTable::new(32, 0x1234);
        let first = table.fingerprint(&[0x11; 16]);
        let second = table.fingerprint(&[0x22; 16]);
        table.insert_new(first, 7, 1 << 40, false).unwrap();
        table.insert_new(second, 9, 1 << 41, true).unwrap();

        let snapshot = table.snapshot();
        let mut streamed = Vec::new();
        table.write_snapshot_to(table.len(), &mut streamed).unwrap();
        assert_eq!(
            streamed, snapshot,
            "streaming must preserve the snapshot bytes"
        );
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
