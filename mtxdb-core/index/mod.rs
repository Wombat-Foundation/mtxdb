use memmap2::Mmap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

pub mod checkpoint;
pub mod delta;
pub mod format;

use crate::index::delta::DeltaReplayError;
use crate::index::format::DeltaFrame;

/// Per-database-instance tuning for [`LossyIndex`] construction.
///
/// `seed` is mixed into bucket *and* tag derivation to prevent adversarial
/// precomputation against a fixed, unsalted content hash. Every real
/// `PackfileStorage`-sourced config must carry the pool's actual persisted
/// seed; the `Default` impl uses `seed: 0` only for tests and transient
/// construction that never touches disk.
#[derive(Debug, Clone, Copy)]
pub struct IndexConfig {
    /// Mixed into bucket and tag derivation to prevent adversarial
    /// precomputation. See `index/mod.rs` seed docs.
    pub seed: u64,
    /// Minimum initial capacity for a newly created collection's index.
    /// Composes with the hard 16-slot minimum: `effective_min = floor.max(16)`.
    pub floor: u32,
    /// Load factor (percent, 1..=90) at which the index grows.
    pub load_factor_percent: u8,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            floor: 64,
            load_factor_percent: 75,
        }
    }
}

impl IndexConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns `Err(message)` if `load_factor_percent` is 0 or greater than
    /// 90, or if `floor` is 0.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.load_factor_percent == 0 || self.load_factor_percent > 90 {
            return Err("load_factor_percent must be in 1..=90");
        }
        if self.floor == 0 {
            return Err("floor must be > 0");
        }
        Ok(())
    }
}

/// Splitmix64-style avalanche finalizer. Cheap, well-distributed, no
/// dependencies. Used to mix the per-pool seed into bucket and tag
/// derivation so an attacker cannot precompute hash → bucket/tag mappings
/// without knowing the seed.
#[inline]
fn mix(x: u64) -> u64 {
    let mut v = x;
    v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// Per-slot entry in the lossy fanout index.
///
/// Layout: `[16-bit tag | 16-bit shard_id | 32-bit offset]` packed into a `u64`.
///
/// - **tag** (high 16 bits): truncated fingerprint for fast rejection.
/// - **`shard_id`** (next 16 bits): which shard file this record lives in.
/// - **offset** (low 32 bits): byte offset within the shard, stored as
///   `offset + 1` so that the all-zeros encoding is reserved as the empty
///   sentinel. Actual offset 0 is stored as 1, and `offset()` subtracts 1
///   to recover the real value.
///
/// Empty slots are all-zeros. This is safe regardless of tag value: the
/// `offset + 1` encoding guarantees a live slot's low 32 bits are never zero,
/// so the all-zeros pattern can never be produced by a real insert — the tag
/// bits play no role in reserving the sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSlot(u64);

impl IndexSlot {
    const EMPTY: Self = Self(0);

    const TAG_SHIFT: u64 = 48; // SHARD_BITS + OFFSET_BITS
    const SHARD_SHIFT: u64 = 32; // OFFSET_BITS
    const OFFSET_MASK: u64 = 0xFFFF_FFFF;

    /// Create a new slot from its components.
    ///
    /// The offset is stored as `offset + 1` so that the all-zeros `u64`
    /// encoding is reserved as the empty sentinel. Actual offset 0 is
    /// stored as 1 in the slot.
    ///
    /// # Panics
    /// Panics if `tag` exceeds 16 bits or `offset` exceeds `2^32 - 2`.
    #[must_use]
    pub fn new(tag: u32, shard_id: u16, offset: u64) -> Self {
        assert!(tag <= 0xFFFF, "tag must fit in 16 bits");
        assert!(
            offset <= (1u64 << 32) - 2,
            "offset must fit in 32 bits minus 1 (reserved for empty sentinel)"
        );
        Self(
            (u64::from(tag) << Self::TAG_SHIFT)
                | (u64::from(shard_id) << Self::SHARD_SHIFT)
                | ((offset.wrapping_add(1)) & Self::OFFSET_MASK),
        )
    }

    /// The sentinel value representing an empty slot.
    #[must_use]
    pub fn empty() -> Self {
        Self::EMPTY
    }

    /// Returns `true` if this slot is the empty sentinel.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The 16-bit tag stored in this slot.
    #[must_use]
    pub fn tag(self) -> u32 {
        ((self.0 >> Self::TAG_SHIFT) & 0xFFFF) as u32
    }

    /// The 16-bit shard ID stored in this slot.
    #[must_use]
    pub fn shard_id(self) -> u16 {
        ((self.0 >> Self::SHARD_SHIFT) & 0xFFFF) as u16
    }

    /// The byte offset within the shard stored in this slot.
    #[must_use]
    pub fn offset(self) -> u64 {
        (self.0 & Self::OFFSET_MASK).wrapping_sub(1)
    }
}

/// A per-collection lossy fanout index for content-addressed records.
///
/// Uses open addressing with linear probing on a power-of-two sized table.
/// The index is mmap-able (flat `u64` array) and small enough to stay
/// resident in RAM for active collections.
///
/// Design decisions (from docs):
/// - Partition by collection: ~8KB per 1000-node collection, 100 active collections < 1MB.
/// - Empty slot terminates probe (write-once, no tombstones needed).
/// - Tag collisions coexist in the probe chain; a tag is only a candidate
///   filter and never an overwrite/equality proof.
/// - Owned indexes keep `homes` and `tails` identity side tables: two
///   anonymous `u64`s (16 bytes) per slot beyond the packed slot array.
///   Checkpoints retain only packed slots; identities are hydrated lazily
///   from authoritative frame headers when a writer needs them.
/// - Power-of-two capacity: shift-and-mask bucket selection, cache-aligned probes.
#[derive(Debug)]
pub struct LossyIndex {
    /// Power-of-two capacity.
    capacity: u32,
    /// Bitmask for bucket selection: capacity - 1.
    mask: u32,
    /// Shift to extract top bits from hash for bucket index.
    shift: u32,
    /// Per-instance config: seed, floor, load factor.
    config: IndexConfig,
    /// The flat slot array. Checkpoint-loaded indexes borrow the immutable
    /// raw slots from their mmap until their first write, which clones them
    /// into the owned atomic representation.
    slots: SlotStorage,
    /// The first 64 bits of each slot's hash. Index slots intentionally pack
    /// only a tag, shard, and offset; retaining the home hash separately lets
    /// a live writer grow the table without rescanning the packfiles.
    homes: Mutex<Vec<u64>>,
    /// The remaining 40 bits of each live slot's hash, plus a high-bit
    /// marker that says the identity is known. Together with `homes` and the
    /// packed tag this makes overwrite equality exact without putting
    /// full hashes in the checkpoint.
    tails: Mutex<Vec<u64>>,
    /// `deserialize` restores slots without their source hashes. Such an
    /// index remains readable, but must be rebuilt from packfiles if it ever
    /// needs to grow.
    can_grow: bool,
    /// Number of occupied slots.
    len: AtomicU32,
}

#[derive(Debug)]
enum SlotStorage {
    Owned(Vec<AtomicU64>),
    Mmap { mmap: Arc<Mmap>, offset: usize },
}

impl SlotStorage {
    #[inline]
    fn get(&self, index: usize) -> u64 {
        match self {
            Self::Owned(slots) => slots[index].load(Ordering::Acquire),
            Self::Mmap { mmap, offset } => {
                let start = offset.saturating_add(index.saturating_mul(8));
                let bytes: [u8; 8] = mmap[start..start.saturating_add(8)]
                    .try_into()
                    .expect("validated checkpoint slot range");
                u64::from_le_bytes(bytes)
            }
        }
    }

    fn materialize(&self, capacity: usize) -> Vec<AtomicU64> {
        (0..capacity)
            .map(|index| AtomicU64::new(self.get(index)))
            .collect()
    }
}

impl Clone for LossyIndex {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            mask: self.mask,
            shift: self.shift,
            config: self.config,
            slots: SlotStorage::Owned(self.slots.materialize(self.capacity as usize)),
            // Checkpoint-backed indexes deliberately omit source hashes. On
            // their first write, clone their slots into owned storage but
            // keep an empty home table; `can_grow` remains false, so a later
            // capacity boundary correctly falls back to a pack rebuild.
            homes: Mutex::new(if self.is_mmap_backed() {
                vec![0; self.capacity as usize]
            } else {
                self.homes.lock().clone()
            }),
            tails: Mutex::new(if self.is_mmap_backed() {
                vec![0; self.capacity as usize]
            } else {
                self.tails.lock().clone()
            }),
            can_grow: self.can_grow,
            len: AtomicU32::new(self.len.load(Ordering::Acquire)),
        }
    }
}

/// The slot plus its retained home-hash used by a live index generation.
const LIVE_SLOT_BYTES: usize = std::mem::size_of::<IndexSlot>() + 2 * std::mem::size_of::<u64>();

impl LossyIndex {
    /// Create a new index with at least `min_capacity` slots using default config.
    /// Capacity is rounded up to the next power of two.
    /// The floor is set to 0 so `new()` preserves backward compatibility
    /// (real callers should use `with_config` with an explicit floor).
    ///
    /// # Panics
    /// Panics if `min_capacity` exceeds `u32::MAX` when rounded to a power of two.
    #[must_use]
    pub fn new(min_capacity: usize) -> Self {
        Self::with_config(
            min_capacity,
            IndexConfig {
                floor: 1,
                ..IndexConfig::default()
            },
        )
    }

    /// Create a new index with at least `min_capacity` slots using the given config.
    /// Capacity is rounded up to the next power of two. The configured `floor`
    /// composes with the hard 16-slot minimum: `effective_min = floor.max(16)`.
    ///
    /// # Panics
    /// Panics if `min_capacity` exceeds `u32::MAX` when rounded to a power of two,
    /// or if `config` fails validation.
    #[must_use]
    pub fn with_config(min_capacity: usize, config: IndexConfig) -> Self {
        config.validate().expect("invalid IndexConfig");
        let effective_min = (config.floor.max(16)) as usize;
        let capacity_usize = min_capacity.max(effective_min).next_power_of_two();
        let capacity = u32::try_from(capacity_usize).expect("index capacity exceeds u32::MAX");
        let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());
        Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift,
            config,
            slots: SlotStorage::Owned((0..capacity_usize).map(|_| AtomicU64::new(0)).collect()),
            homes: Mutex::new(vec![0; capacity_usize]),
            tails: Mutex::new(vec![0; capacity_usize]),
            can_grow: true,
            len: AtomicU32::new(0),
        }
    }

    /// This index's slot-table capacity (always a power of two).
    #[must_use]
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Extract the bucket index from a 16-byte hash.
    ///
    /// The seed is `XORed` into the raw home value before the avalanche mix,
    /// so the same content hash lands in different buckets across pools
    /// with different seeds. The mixed value is what gets stored in `homes`,
    /// so `grow()` (which rehashes from stored homes) sees the already-mixed
    /// value and doesn't need to know the seed at all.
    #[inline]
    fn bucket(&self, hash: &[u8; 16]) -> usize {
        let raw = u64::from_be_bytes(hash[..8].try_into().unwrap());
        let home = mix(raw ^ self.config.seed);
        self.bucket_for_home(home)
    }

    /// Mix a raw home value with this index's seed. Used by insert to store
    /// the post-mix value in `homes`, and by `hydrate_slot_identity` to
    /// reproduce the same home from a recovered hash.
    #[inline]
    fn mix_home(&self, raw_home: u64) -> u64 {
        mix(raw_home ^ self.config.seed)
    }

    #[inline]
    fn bucket_for_home(&self, home: u64) -> usize {
        let masked = (home >> self.shift) & u64::from(self.mask);
        // Safety: masked is always <= mask < capacity which fits in usize on all platforms
        usize::try_from(masked).unwrap_or(usize::MAX)
    }

    /// Extract the 16-bit tag from a 16-byte hash, seeded with this index's seed.
    ///
    /// Uses bytes 8..12, disjoint from the bytes `bucket()` reads (0..8).
    /// Seeding the tag prevents an attacker from grinding offline for
    /// tag collisions against a known target bucket — without the seed,
    /// a narrower 16-bit tag would make I/O amplification attacks feasible.
    #[inline]
    fn tag(&self, hash: &[u8; 16]) -> u32 {
        let raw = u32::from_be_bytes(hash[8..12].try_into().unwrap());
        let mixed = mix(u64::from(raw) ^ self.config.seed);
        (mixed >> 48) as u32
    }

    /// The packed-slot tag for `hash`, exposed to crate-local recovery code
    /// that must validate a persisted slot against its authoritative frame.
    ///
    /// Must use this index's own seed: every stored slot's tag was computed
    /// via `self.tag(hash)` at insert time, so a recovery comparison against
    /// an unseeded (or differently-seeded) tag would fail closed on every
    /// occupied slot, not just colliding ones.
    #[inline]
    pub(crate) fn tag_for_hash(&self, hash: &[u8; 16]) -> u32 {
        self.tag(hash)
    }

    #[inline]
    fn tail(hash: &[u8; 16]) -> u64 {
        const KNOWN: u64 = 1_u64 << 63;
        let mut bytes = [0; 8];
        bytes[3..].copy_from_slice(&hash[11..]);
        KNOWN | u64::from_be_bytes(bytes)
    }

    /// Insert a (hash → `shard_id`, offset) mapping.
    ///
    /// # Errors
    /// Returns `InsertError::TableFull` if the table has less than 25% free slots
    /// and the hash is not already present (overwrites are always allowed).
    pub fn insert(&self, hash: &[u8; 16], shard_id: u16, offset: u64) -> Result<(), InsertError> {
        self.insert_tracked(hash, shard_id, offset).map(|_| ())
    }

    /// Like [`Self::insert`], but also reports the write's exact landing spot —
    /// the affected bucket and the packed slot value stored there — so the
    /// delta persistence layer can append a replay frame without re-probing.
    ///
    /// # Errors
    /// Returns `InsertError::TableFull` under the same conditions as
    /// [`Self::insert`] (the table is left unmodified).
    // `bucket` is always below `self.capacity`, which is capped at `u32::MAX`
    // by construction, so the `as u32` narrowings below never truncate.
    #[allow(clippy::cast_possible_truncation)]
    pub fn insert_tracked(
        &self,
        hash: &[u8; 16],
        shard_id: u16,
        offset: u64,
    ) -> Result<(u32, u64), InsertError> {
        self.insert_undoable(hash, shard_id, offset)
            .map(|(bucket, value, _undo)| (bucket, value))
    }

    /// Like [`Self::insert_tracked`], but also returns a [`SlotUndo`]
    /// capturing the slot's prior contents, so a caller that needs to
    /// mutate the live (not cloned) index for a multi-record batch can
    /// undo this one write if a later entry in the batch fails.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::insert_tracked`] and leaves the
    /// table unmodified in every error case.
    ///
    /// # Panics
    /// Never panics in practice: `hash[..8]` is always exactly 8 bytes for a
    /// `&[u8; 16]` input, so the `try_into` this performs cannot fail.
    #[allow(clippy::cast_possible_truncation)]
    pub fn insert_undoable(
        &self,
        hash: &[u8; 16],
        shard_id: u16,
        offset: u64,
    ) -> Result<(u32, u64, SlotUndo), InsertError> {
        let tag = self.tag(hash);
        let raw_home = u64::from_be_bytes(hash[..8].try_into().unwrap());
        let home = self.mix_home(raw_home);
        let mut bucket = self.bucket_for_home(home);

        loop {
            let SlotStorage::Owned(slots) = &self.slots else {
                return Err(InsertError::TableFull);
            };
            let slot = IndexSlot(slots[bucket].load(Ordering::Acquire));
            if slot.is_empty() {
                // Only reject when actually inserting into a new slot
                let threshold = self
                    .capacity
                    .wrapping_mul(u32::from(self.config.load_factor_percent))
                    / 100;
                if self.len.load(Ordering::Relaxed) >= threshold {
                    return Err(InsertError::TableFull);
                }
                let old_home = self.homes.lock()[bucket];
                let old_tail = self.tails.lock()[bucket];
                // Writers are serialized by the collection put lock. Publish
                // the slot last so concurrent readers see either the prior
                // empty slot or a complete record location.
                self.homes.lock()[bucket] = home;
                self.tails.lock()[bucket] = Self::tail(hash);
                let value = IndexSlot::new(tag, shard_id, offset).0;
                slots[bucket].store(value, Ordering::Release);
                self.len.fetch_add(1, Ordering::Relaxed);
                let undo = SlotUndo {
                    bucket: bucket as u32,
                    old_slot: 0,
                    old_home,
                    old_tail,
                    was_empty: true,
                };
                return Ok((bucket as u32, value, undo));
            }
            // A tag only filters candidates; it never proves key equality.
            // Checkpoint-derived slots initially lack their 40-bit tail and
            // ask storage to hydrate it from the frame before deciding.
            if slot.tag() == tag {
                let tail = self.tails.lock()[bucket];
                if tail == 0 {
                    return Err(InsertError::NeedsIdentity {
                        bucket: bucket as u32,
                        shard_id: slot.shard_id(),
                        offset: slot.offset(),
                    });
                }
                if self.homes.lock()[bucket] == home && tail == Self::tail(hash) {
                    let old_slot = slot.0;
                    let old_home = self.homes.lock()[bucket];
                    let value = IndexSlot::new(tag, shard_id, offset).0;
                    slots[bucket].store(value, Ordering::Release);
                    let undo = SlotUndo {
                        bucket: bucket as u32,
                        old_slot,
                        old_home,
                        old_tail: tail,
                        was_empty: false,
                    };
                    return Ok((bucket as u32, value, undo));
                }
            }
            bucket = bucket.wrapping_add(1) & self.mask as usize;
        }
    }

    /// Reverse a write reported by [`Self::insert_undoable`]. Callers must
    /// replay a batch's undo entries in the reverse order they were
    /// produced, and only against the same live index that produced them
    /// (never a cloned/owned index materialized after the fact).
    ///
    /// # Panics
    /// Never panics; a no-op if called against a checkpoint-mmap-backed
    /// index, since [`Self::insert_undoable`] cannot have produced an undo
    /// entry against one (it errors with `TableFull` first).
    pub fn rollback_slot(&self, undo: &SlotUndo) {
        let SlotStorage::Owned(slots) = &self.slots else {
            return;
        };
        let bucket = undo.bucket as usize;
        self.homes.lock()[bucket] = undo.old_home;
        self.tails.lock()[bucket] = undo.old_tail;
        slots[bucket].store(undo.old_slot, Ordering::Release);
        if undo.was_empty {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Mark a checkpoint-derived slot with the full identity recovered from
    /// its authoritative frame. Returns `false` if the slot changed or the
    /// supplied hash cannot describe that slot.
    pub(crate) fn hydrate_slot_identity(&self, bucket: u32, hash: &[u8; 16]) -> bool {
        let bucket = bucket as usize;
        if bucket >= self.capacity as usize {
            return false;
        }
        let slot = IndexSlot(self.slot_at(bucket));
        if slot.is_empty() || slot.tag() != self.tag(hash) {
            return false;
        }
        let raw_home = u64::from_be_bytes(hash[..8].try_into().expect("16-byte hash"));
        let home = self.mix_home(raw_home);
        self.homes.lock()[bucket] = home;
        self.tails.lock()[bucket] = Self::tail(hash);
        true
    }

    /// Apply recorded delta frames on top of a checkpoint-backed index,
    /// replicating the live insert/overwrite sequence that produced them.
    ///
    /// Only valid on an owned (materialized) index — callers must clone an
    /// mmap-backed index first. Frames are applied exactly as the live
    /// session wrote them: an overwrite of a non-empty slot leaves the length
    /// unchanged, a write into an empty slot increments it. The index's `len`
    /// after replay therefore equals the checkpoint's occupancy plus the
    /// number of fresh-slot writes, which is what the sidecar integrity sum
    /// checks against.
    ///
    /// # Errors
    /// Returns `DeltaReplayError` if a frame targets a bucket outside this
    /// index's capacity, or carries the empty-slot sentinel (either is a
    /// structurally inconsistent log, rejected wholesale), or the index is
    /// still mmap-backed.
    pub fn replay_frames(&self, frames: &[DeltaFrame]) -> Result<(), DeltaReplayError> {
        let SlotStorage::Owned(slots) = &self.slots else {
            return Err(DeltaReplayError::RequiresOwnedIndex);
        };
        // Validate every frame before mutating any slot: the contract is
        // wholesale rejection of a structurally inconsistent log, so a
        // caller must never observe a partially replayed index.
        for frame in frames {
            let bucket = frame.bucket as usize;
            if bucket >= self.capacity as usize {
                return Err(DeltaReplayError::FrameOutOfBounds {
                    bucket: frame.bucket,
                    capacity: self.capacity,
                });
            }
            if frame.slot == 0 {
                // A legitimate insert never logs the empty sentinel (see
                // `record_delta`'s only caller). Storing it here would erase
                // whatever live entry currently occupies this bucket and cut
                // the probe chain short for anything beyond it — reject the
                // whole log instead of silently losing data.
                return Err(DeltaReplayError::EmptySlot {
                    bucket: frame.bucket,
                });
            }
        }
        for frame in frames {
            let bucket = frame.bucket as usize;
            let previous = slots[bucket].load(Ordering::Acquire);
            if previous == 0 {
                self.len.fetch_add(1, Ordering::Relaxed);
            }
            slots[bucket].store(frame.slot, Ordering::Release);
        }
        Ok(())
    }

    /// Double this live index's capacity without reading packfiles.
    ///
    /// Returns `false` for an index deserialized from the compact on-disk
    /// representation, whose original hashes are unavailable for rehashing.
    /// Callers should retain their existing packfile rebuild fallback in that
    /// case.
    pub fn grow(&self) -> Option<Self> {
        if !self.can_grow || self.capacity > u32::MAX / 2 {
            return None;
        }
        let grown = Self::with_config(self.capacity as usize * 2, self.config);
        let homes = self.homes.lock();
        let tails = self.tails.lock();
        for (index, home) in homes.iter().copied().enumerate() {
            let slot = IndexSlot(self.slots.get(index));
            if slot.is_empty() {
                continue;
            }
            let mut bucket = grown.bucket_for_home(home);
            while !IndexSlot(grown.slot_at(bucket)).is_empty() {
                bucket = bucket.wrapping_add(1) & grown.mask as usize;
            }
            grown.homes.lock()[bucket] = home;
            grown.tails.lock()[bucket] = tails[index];
            let SlotStorage::Owned(slots) = &grown.slots else {
                unreachable!("new index is owned")
            };
            slots[bucket].store(slot.0, Ordering::Relaxed);
            grown.len.fetch_add(1, Ordering::Relaxed);
        }
        Some(grown)
    }

    /// Double the table by recovering each occupied entry's full hash from
    /// its stored location.
    ///
    /// Checkpoint-backed indexes deliberately omit `homes`, so their first
    /// post-restart resize cannot use [`Self::grow`].  The packed slots still
    /// retain every `(shard_id, offset)`, however.  A caller can therefore
    /// recover only the `len` source hashes from the authoritative records
    /// and rehash without rescanning every pack in the store. Locations are
    /// visited in `(shard, offset)` order rather than hash-table order, so a
    /// cold mmap walk follows pack order and lets the kernel coalesce page
    /// faults/read-ahead. The callback must return the hash for precisely the
    /// supplied location.
    ///
    /// `Ok(None)` means the table cannot grow further; callers retain their
    /// full-rebuild fallback for that terminal case.
    pub(crate) fn grow_by_recovering_hashes<E>(
        &self,
        mut hash_at: impl FnMut(u16, u64, u32) -> Result<[u8; 16], E>,
    ) -> Result<Option<Self>, E> {
        if self.capacity > u32::MAX / 2 {
            return Ok(None);
        }

        let mut locations = Vec::with_capacity(self.len());
        for index in 0..self.capacity as usize {
            let slot = IndexSlot(self.slot_at(index));
            if slot.is_empty() {
                continue;
            }
            locations.push((slot.shard_id(), slot.offset(), slot.tag()));
        }
        locations.sort_unstable_by_key(|(shard_id, offset, _)| (*shard_id, *offset));

        let grown = Self::with_config(self.capacity as usize * 2, self.config);
        for (shard_id, offset, slot_tag) in locations {
            let hash = hash_at(shard_id, offset, slot_tag)?;
            // A doubled table is at most 37.5% full because insertion only
            // requests growth at 75%, so this cannot hit TableFull.
            let _ = grown.insert(&hash, shard_id, offset);
        }
        Ok(Some(grown))
    }

    /// Look up a hash in the index.
    /// Returns `(shard_id, offset)` if found, `None` if absent.
    ///
    /// An empty slot terminates the probe — this is correct because:
    /// 1. Class A tables are write-once with no deletes (no tombstones needed).
    /// 2. An empty slot means the key was never inserted.
    #[inline]
    #[must_use]
    pub fn lookup(&self, hash: &[u8; 16]) -> Option<(u16, u64)> {
        self.lookup_all(hash).next()
    }

    /// Look up all candidate offsets for a hash, yielding tag collisions.
    ///
    /// Candidates with hydrated in-memory identity (home + tail) are verified
    /// before yielding, so tag collisions in a live/warm index are filtered
    /// out here rather than forcing the caller to disk. For a slot whose
    /// identity is not yet hydrated (e.g. a fresh checkpoint restore), the
    /// caller **must** verify the candidate against the caller-requested
    /// hash — a raw tag collision (~1/65536 at 16-bit tags) can still surface
    /// unverified.
    ///
    /// Yields `(shard_id, offset)` for each slot whose tag matches and isn't
    /// ruled out, then terminates at the first empty slot or after
    /// `capacity` probes.
    ///
    /// # Panics
    /// Never panics in practice: `hash[..8]` is always exactly 8 bytes for a
    /// `&[u8; 16]` input, so the `try_into` this performs cannot fail.
    #[inline]
    #[must_use]
    pub fn lookup_all(&self, hash: &[u8; 16]) -> LookupIter<'_> {
        let tag = self.tag(hash);
        let bucket = self.bucket(hash);
        let raw_home = u64::from_be_bytes(hash[..8].try_into().unwrap());
        let mixed_home = self.mix_home(raw_home);
        let expected_tail = Self::tail(hash);
        LookupIter {
            index: self,
            tag,
            mixed_home,
            expected_tail,
            bucket,
            mask: self.mask as usize,
            remaining: self.capacity as usize,
        }
    }

    /// Number of occupied slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len.load(Ordering::Acquire) == 0
    }

    /// Estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        (self.capacity as usize)
            .wrapping_mul(LIVE_SLOT_BYTES)
            .wrapping_add(std::mem::size_of::<Self>())
    }

    /// Slot-table capacity an index holding `entries` records would have
    /// after the standard two-times-capacity allocation policy is applied —
    /// i.e. without opening the index, e.g. from a persisted node count.
    ///
    /// `min_capacity` is the caller's own actual starting floor (`Self::new`
    /// always rounds up to at least 16, but a caller that always creates
    /// indexes above that — e.g. `mtxdb-core`'s collections, which start at
    /// `NEW_COLLECTION_INDEX_FLOOR` — must pass its own floor here, or this
    /// underestimates a small collection's real capacity).
    #[must_use]
    pub fn capacity_for_entries(entries: usize, min_capacity: usize) -> usize {
        let minimum = entries.saturating_mul(2).max(min_capacity.max(16));
        minimum.next_power_of_two()
    }

    /// Memory that an index holding `entries` records would use after the
    /// standard two-times-capacity allocation policy is applied. See
    /// [`Self::capacity_for_entries`] for `min_capacity`.
    #[must_use]
    pub fn memory_usage_for_entries(entries: usize, min_capacity: usize) -> usize {
        Self::capacity_for_entries(entries, min_capacity)
            .wrapping_mul(LIVE_SLOT_BYTES)
            .wrapping_add(std::mem::size_of::<Self>())
    }

    /// Returns which shard IDs are referenced by at least one occupied slot.
    ///
    /// Used by shard retirement to determine which shards are still live
    /// across all collections before freeing a pool slot.
    #[must_use]
    pub fn referenced_shard_ids(&self) -> [bool; crate::shard::MAX_SHARDS] {
        let mut seen = [false; crate::shard::MAX_SHARDS];
        for index in 0..self.capacity as usize {
            let slot = IndexSlot(self.slot_at(index));
            if !slot.is_empty() {
                let id = slot.shard_id() as usize;
                if id < crate::shard::MAX_SHARDS {
                    seen[id] = true;
                }
            }
        }
        seen
    }

    /// Tally of how many occupied slots point into each shard.
    ///
    /// Used to maintain the persisted per-shard collection directory (see
    /// `PackfileStorage`'s `shard_collections` tracking): whenever a collection's
    /// index is rebuilt or swapped in, this gives the exact per-shard
    /// contribution to record against that collection, without a second scan
    /// of the packfile itself.
    #[must_use]
    pub fn shard_counts(&self) -> std::collections::HashMap<u16, u64> {
        let mut counts = std::collections::HashMap::new();
        for index in 0..self.capacity as usize {
            let slot = IndexSlot(self.slot_at(index));
            if !slot.is_empty() {
                let entry = counts.entry(slot.shard_id()).or_insert(0u64);
                *entry = entry.saturating_add(1);
            }
        }
        counts
    }

    /// Whether this collection's index currently has any live entry pointing
    /// into `shard_id`. Short-circuits on the first match — unlike
    /// `referenced_shard_ids`, which always builds a full `MAX_SHARDS`
    /// membership map, this is the cheap check for "does this one collection
    /// still reference this one shard," used to filter a shard-scan's
    /// candidate collection list down to collections that haven't already repacked
    /// past it.
    #[must_use]
    pub fn references_shard(&self, shard_id: u16) -> bool {
        (0..self.capacity as usize).any(|index| {
            let slot = IndexSlot(self.slot_at(index));
            !slot.is_empty() && slot.shard_id() == shard_id
        })
    }

    /// Serialize the index to bytes for persistence.
    ///
    /// Format (all little-endian):
    /// ```text
    ///   [capacity: u64]
    ///   [slot_0: u64] ... [slot_{cap-1}: u64]     ← capacity × 8 B
    ///   [home_0: u64] ... [home_{cap-1}: u64]     ← capacity × 8 B (0 if unhydrated)
    ///   [tail_0: u64] ... [tail_{cap-1}: u64]     ← capacity × 8 B (0 if unhydrated)
    /// ```
    /// Total: `8 + capacity * 24` bytes.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let cap = self.capacity as usize;
        let byte_len = 8_usize.wrapping_add(cap.wrapping_mul(24));
        let mut buf = Vec::with_capacity(byte_len);
        buf.extend_from_slice(&u64::from(self.capacity).to_le_bytes());
        for index in 0..cap {
            let slot = IndexSlot(self.slot_at(index));
            buf.extend_from_slice(&slot.0.to_le_bytes());
        }
        let homes = self.homes.lock();
        let tails = self.tails.lock();
        for index in 0..cap {
            buf.extend_from_slice(&homes[index].to_le_bytes());
        }
        for index in 0..cap {
            buf.extend_from_slice(&tails[index].to_le_bytes());
        }
        buf
    }

    /// Deserialize an index from bytes, using default config (no seed).
    ///
    /// # Errors
    /// Returns `DeserializationError::TooShort` if the data is too short,
    /// or `DeserializationError::InvalidCapacity` if the capacity is not
    /// a power of two or is less than 16.
    ///
    /// # Panics
    /// Panics if the 8-byte capacity header cannot be read (guaranteed by the length check).
    pub fn deserialize(data: &[u8]) -> Result<Self, DeserializationError> {
        Self::deserialize_with_config(data, IndexConfig::default())
    }

    /// Deserialize an index from bytes with a given config (providing seed).
    ///
    /// Accepts both the legacy slots-only format (`8 + cap*8` bytes) and the
    /// v4 slots+homes+tails format (`8 + cap*24` bytes). Homes and tails are
    /// populated from the blob when present; zeroed otherwise.
    ///
    /// # Errors
    /// Returns `DeserializationError::TooShort` if the data is too short,
    /// or `DeserializationError::InvalidCapacity` if the capacity is not
    /// a power of two or is less than 16.
    ///
    /// # Panics
    /// Panics if the 8-byte capacity header cannot be read (guaranteed by the length check).
    pub fn deserialize_with_config(
        data: &[u8],
        config: IndexConfig,
    ) -> Result<Self, DeserializationError> {
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

        let slots_len = 8_usize.wrapping_add(capacity_usize.wrapping_mul(8));
        let full_len = 8_usize.wrapping_add(capacity_usize.wrapping_mul(24));
        if data.len() < slots_len {
            return Err(DeserializationError::TooShort);
        }
        let has_homes_tails = data.len() >= full_len;

        let mut slots = Vec::with_capacity(capacity_usize);
        let mut len: u32 = 0;
        for i in 0..capacity_usize {
            let offset = 8_usize.wrapping_add(i.wrapping_mul(8));
            let val = u64::from_le_bytes(data[offset..offset.wrapping_add(8)].try_into().unwrap());
            let slot = IndexSlot(val);
            if !slot.is_empty() {
                len = len.wrapping_add(1);
            }
            slots.push(AtomicU64::new(slot.0));
        }

        let shift = 64_u32.wrapping_sub(capacity.trailing_zeros());

        let homes = if has_homes_tails {
            let base = slots_len;
            let mut h = Vec::with_capacity(capacity_usize);
            for i in 0..capacity_usize {
                let offset = base.wrapping_add(i.wrapping_mul(8));
                h.push(u64::from_le_bytes(
                    data[offset..offset.wrapping_add(8)].try_into().unwrap(),
                ));
            }
            h
        } else {
            vec![0; capacity_usize]
        };

        let tails = if has_homes_tails {
            let base = slots_len.wrapping_add(capacity_usize.wrapping_mul(8));
            let mut t = Vec::with_capacity(capacity_usize);
            for i in 0..capacity_usize {
                let offset = base.wrapping_add(i.wrapping_mul(8));
                t.push(u64::from_le_bytes(
                    data[offset..offset.wrapping_add(8)].try_into().unwrap(),
                ));
            }
            t
        } else {
            vec![0; capacity_usize]
        };

        Ok(Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift,
            config,
            slots: SlotStorage::Owned(slots),
            homes: Mutex::new(homes),
            tails: Mutex::new(tails),
            can_grow: false,
            len: AtomicU32::new(len),
        })
    }

    /// Build a read-only index directly over a validated checkpoint's raw
    /// slots, using default config. The first write clones it into the
    /// normal owned representation.
    #[must_use]
    pub fn from_mmap_slots(mmap: Arc<Mmap>, offset: usize, capacity: u32, len: u32) -> Self {
        Self::from_mmap_slots_with_config(mmap, offset, capacity, len, IndexConfig::default())
    }

    /// Build a read-only index directly over a validated checkpoint's raw
    /// slots with a given config. The first write clones it into the
    /// normal owned representation.
    #[must_use]
    pub fn from_mmap_slots_with_config(
        mmap: Arc<Mmap>,
        offset: usize,
        capacity: u32,
        len: u32,
        config: IndexConfig,
    ) -> Self {
        Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift: 64_u32.wrapping_sub(capacity.trailing_zeros()),
            config,
            slots: SlotStorage::Mmap { mmap, offset },
            homes: Mutex::new(Vec::new()),
            tails: Mutex::new(Vec::new()),
            can_grow: false,
            len: AtomicU32::new(len),
        }
    }

    /// Build a read-only index directly over a validated checkpoint's raw
    /// slots with a given config, loading pre-hydrated homes and tails from
    /// the checkpoint mmap. This eliminates cold-start packfile reads for
    /// tag-collision verification when the checkpoint carries identity data.
    ///
    /// The slots remain mmap-backed; homes and tails are copied into owned
    /// `Vec<u64>` so they can be updated by `hydrate_slot_identity` if needed.
    ///
    /// # Panics
    /// Panics if `[homes_offset, homes_offset + capacity*8)` or
    /// `[tails_offset, tails_offset + capacity*8)` fall outside `mmap` —
    /// callers must only pass offsets validated against a checkpoint's own
    /// declared body length (see `checkpoint::read_checkpoint`).
    #[must_use]
    pub fn from_mmap_slots_with_homes_tails(
        mmap: Arc<Mmap>,
        slots_offset: usize,
        homes_offset: usize,
        tails_offset: usize,
        capacity: u32,
        len: u32,
        config: IndexConfig,
    ) -> Self {
        let cap = usize::try_from(capacity).unwrap_or(usize::MAX);
        let mut homes = Vec::with_capacity(cap);
        let mut tails = Vec::with_capacity(cap);
        let buf: &[u8] = &mmap;
        for i in 0..cap {
            let h_off = homes_offset.wrapping_add(i.wrapping_mul(8));
            let t_off = tails_offset.wrapping_add(i.wrapping_mul(8));
            homes.push(u64::from_le_bytes(
                buf[h_off..h_off.wrapping_add(8)].try_into().unwrap(),
            ));
            tails.push(u64::from_le_bytes(
                buf[t_off..t_off.wrapping_add(8)].try_into().unwrap(),
            ));
        }
        Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift: 64_u32.wrapping_sub(capacity.trailing_zeros()),
            config,
            slots: SlotStorage::Mmap {
                mmap,
                offset: slots_offset,
            },
            homes: Mutex::new(homes),
            tails: Mutex::new(tails),
            can_grow: false,
            len: AtomicU32::new(len),
        }
    }

    /// True while the index borrows raw slots from a checkpoint mapping.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        matches!(self.slots, SlotStorage::Mmap { .. })
    }

    #[inline]
    fn slot_at(&self, index: usize) -> u64 {
        self.slots.get(index)
    }
}

/// Captures a single slot's prior contents so a live (uncloned) index write
/// made by [`LossyIndex::insert_undoable`] can be reversed by
/// [`LossyIndex::rollback_slot`] if a later entry in the same batch fails.
#[derive(Debug, Clone, Copy)]
pub struct SlotUndo {
    bucket: u32,
    old_slot: u64,
    old_home: u64,
    old_tail: u64,
    was_empty: bool,
}

/// Errors that can occur while inserting into a [`LossyIndex`].
#[derive(Debug)]
pub enum InsertError {
    /// The table has reached 75% occupancy.
    /// Returned when the table reaches 75% occupancy to keep probe sequences short.
    TableFull,
    /// A checkpoint-derived same-tag slot needs its frame identity restored
    /// before insertion can determine whether it is an overwrite.
    NeedsIdentity {
        /// Probe bucket holding the candidate slot.
        bucket: u32,
        /// Candidate frame's shard.
        shard_id: u16,
        /// Candidate frame's offset.
        offset: u64,
    },
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableFull => write!(f, "index table too full"),
            Self::NeedsIdentity { .. } => write!(f, "index slot needs frame identity"),
        }
    }
}

impl std::error::Error for InsertError {}

/// Errors that can occur while deserializing a [`LossyIndex`] from bytes.
#[derive(Debug)]
pub enum DeserializationError {
    /// The input buffer was too short to contain a valid header.
    TooShort,
    /// The header declared a capacity that is not a valid power of two.
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
/// Yields `(shard_id, offset)` for each slot whose tag matches and whose
/// in-memory identity (home + tail) matches the query hash, terminating at
/// the first empty slot or after the full capacity is probed.
///
/// When the index has hydrated identity (live/warm index), candidates are
/// verified in-memory before yielding — this eliminates disk I/O for tag
/// collisions on the normal path. For checkpoint-backed indexes without
/// hydrated identity (tail == 0), all tag-matching candidates are yielded
/// and the caller must verify against the packfile.
pub struct LookupIter<'a> {
    index: &'a LossyIndex,
    tag: u32,
    /// The mixed home for the query hash, for in-memory verification.
    mixed_home: u64,
    /// The expected tail for the query hash.
    expected_tail: u64,
    bucket: usize,
    mask: usize,
    remaining: usize,
}

impl Iterator for LookupIter<'_> {
    type Item = (u16, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            self.remaining = self.remaining.wrapping_sub(1);
            let slot = IndexSlot(self.index.slot_at(self.bucket));
            if slot.is_empty() {
                return None;
            }
            let current = self.bucket;
            self.bucket = self.bucket.wrapping_add(1) & self.mask;
            if slot.tag() != self.tag {
                continue;
            }
            // Tag matches. For a live index with hydrated identity, verify
            // home + tail in-memory to avoid yielding false positives that
            // would force the caller into an expensive packfile read.
            let tails = self.index.tails.lock();
            let tail = if tails.len() > current {
                tails[current]
            } else {
                0
            };
            drop(tails);
            if tail != 0 {
                // Identity is hydrated — check home and tail.
                let homes = self.index.homes.lock();
                let home = if homes.len() > current {
                    homes[current]
                } else {
                    0
                };
                drop(homes);
                if home == self.mixed_home && tail == self.expected_tail {
                    return Some((slot.shard_id(), slot.offset()));
                }
                // Tag matched but home/tail didn't — false positive, skip.
                continue;
            }
            // Checkpoint-backed without hydrated identity — yield and let
            // the caller verify against the packfile.
            return Some((slot.shard_id(), slot.offset()));
        }
        None
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_hash(byte: u8) -> [u8; 16] {
        let mut h = [0u8; 16];
        h[0] = byte;
        h
    }

    #[test]
    fn test_slot_packing() {
        let slot = IndexSlot::new(0xCDEF, 42, 0x0FFF_FFF0);
        assert_eq!(slot.tag(), 0xCDEF);
        assert_eq!(slot.shard_id(), 42);
        assert_eq!(slot.offset(), 0x0FFF_FFF0);
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
        let index = LossyIndex::new(128);
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
    fn grow_by_recovering_hashes_rehashes_only_occupied_locations() {
        let index = LossyIndex::new(16);
        let mut hashes = HashMap::new();
        for offset in 0..12u64 {
            let mut hash = [0u8; 16];
            hash[..8].copy_from_slice(&(offset.wrapping_mul(17).wrapping_add(1)).to_be_bytes());
            hash[8] = u8::try_from(offset.wrapping_add(1)).unwrap();
            index.insert(&hash, 3, offset).unwrap();
            hashes.insert((3, offset), hash);
        }

        let mut recovered = 0usize;
        let grown = index
            .grow_by_recovering_hashes(|shard, offset, _tag| {
                recovered = recovered.saturating_add(1);
                Ok::<_, ()>(hashes[&(shard, offset)])
            })
            .unwrap()
            .expect("a 16-slot table can double");

        assert_eq!(recovered, 12, "recover exactly one hash per occupied slot");
        assert_eq!(grown.capacity, 32);
        for ((shard, offset), hash) in hashes {
            assert_eq!(grown.lookup(&hash), Some((shard, offset)));
        }
    }

    #[test]
    fn test_linear_probing() {
        let index = LossyIndex::new(16); // small table
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
        let index = LossyIndex::new(16); // capacity 16, threshold at 75% = 12
                                         // Insert hashes until we get 12 occupied slots.
        let mut inserted = 0u64;
        for i in 0..1000u64 {
            let h = splitmix_hash(i);
            if index.insert(&h, 0, i).is_ok() {
                inserted += 1;
                if inserted >= 12 {
                    break;
                }
            }
        }
        assert_eq!(index.len(), 12);
        let h = splitmix_hash(9999);
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
            128 * LIVE_SLOT_BYTES + std::mem::size_of::<LossyIndex>()
        );

        let index = LossyIndex::new(128);
        index.insert(&test_hash(0x01), 0, 1).unwrap();
        assert!(!index.is_empty());
    }

    #[test]
    fn test_grow_preserves_existing_lookups() {
        let index = LossyIndex::new(16);
        let hashes: Vec<_> = (0..12).map(splitmix_hash).collect();
        for (offset, hash) in hashes.iter().enumerate() {
            index.insert(hash, 0, offset as u64).unwrap();
        }
        let index = index.grow().expect("live index can grow");
        assert_eq!(index.len(), hashes.len());
        for (offset, hash) in hashes.iter().enumerate() {
            assert_eq!(index.lookup(hash), Some((0, offset as u64)));
        }
        index.insert(&splitmix_hash(12), 0, 12).unwrap();
    }

    #[test]
    fn test_rollback_slot_undoes_fresh_insert() {
        let index = LossyIndex::new(128);
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);
        index.insert(&h1, 0, 100).unwrap();
        assert_eq!(index.len(), 1);

        let (_, _, undo) = index.insert_undoable(&h2, 1, 200).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.lookup(&h2), Some((1, 200)));

        index.rollback_slot(&undo);
        assert_eq!(
            index.len(),
            1,
            "rollback of a fresh-slot insert must restore len"
        );
        assert_eq!(
            index.lookup(&h2),
            None,
            "rolled-back entry must not be findable"
        );
        assert_eq!(
            index.lookup(&h1),
            Some((0, 100)),
            "rollback must not disturb an unrelated slot"
        );
    }

    #[test]
    fn test_rollback_slot_undoes_overwrite() {
        let index = LossyIndex::new(128);
        let h = test_hash(0x01);
        index.insert(&h, 0, 100).unwrap();
        assert_eq!(index.len(), 1);

        let (_, _, undo) = index.insert_undoable(&h, 1, 200).unwrap();
        assert_eq!(index.lookup(&h), Some((1, 200)));

        index.rollback_slot(&undo);
        assert_eq!(
            index.len(),
            1,
            "rollback of an overwrite must not change len"
        );
        assert_eq!(
            index.lookup(&h),
            Some((0, 100)),
            "rollback of an overwrite must restore the prior location"
        );
    }

    #[test]
    fn test_overwrite_same_hash() {
        let index = LossyIndex::new(128);
        let h = test_hash(0x01);
        index.insert(&h, 0, 100).unwrap();
        index.insert(&h, 1, 200).unwrap(); // overwrite
        assert_eq!(index.len(), 1);
        assert_eq!(index.lookup(&h), Some((1, 200)));
    }

    #[test]
    fn same_tag_and_home_but_distinct_hashes_both_remain_indexed() {
        let index = LossyIndex::new(16);
        let mut first = [0u8; 16];
        first[8] = 0x42;
        first[15] = 1;
        let mut second = first;
        second[15] = 2;

        index.insert(&first, 0, 100).unwrap();
        index.insert(&second, 0, 200).unwrap();

        assert_eq!(index.len(), 2, "a tag is not an overwrite proof");
        // Both coexist and are individually findable via their exact hash.
        assert_eq!(index.lookup(&first), Some((0, 100)));
        assert_eq!(index.lookup(&second), Some((0, 200)));
    }

    #[test]
    fn replay_preserves_a_distinct_same_tag_slot() {
        let mut first = [0u8; 16];
        first[8] = 0x42;
        first[15] = 1;
        let mut second = first;
        second[15] = 2;

        // Model the checkpoint before the second write. Serialization omits
        // identities, as the compact checkpoint format intentionally does.
        let checkpoint_source = LossyIndex::new(16);
        checkpoint_source.insert(&first, 0, 100).unwrap();
        let checkpoint = LossyIndex::deserialize(&checkpoint_source.serialize()).unwrap();

        // The live writer records the exact bucket/slot placement for the
        // second colliding identity. Replay must preserve that placement; it
        // must not re-run tag-only insertion and drop either record.
        let live = LossyIndex::new(16);
        live.insert(&first, 0, 100).unwrap();
        let (bucket, slot) = live.insert_tracked(&second, 0, 200).unwrap();
        checkpoint
            .replay_frames(&[DeltaFrame {
                collection_id: [0; 16],
                bucket,
                generation: 0,
                slot,
            }])
            .unwrap();

        assert_eq!(checkpoint.len(), 2);
        assert_eq!(
            checkpoint.lookup_all(&first).collect::<Vec<_>>(),
            vec![(0, 100), (0, 200)]
        );
    }

    #[test]
    fn replay_rejects_a_frame_carrying_the_empty_sentinel() {
        let mut hash = [0u8; 16];
        hash[15] = 1;

        let checkpoint_source = LossyIndex::new(16);
        let (bucket, _slot) = checkpoint_source.insert_tracked(&hash, 0, 100).unwrap();
        let checkpoint = LossyIndex::deserialize(&checkpoint_source.serialize()).unwrap();
        let occupied_before = checkpoint.len();

        // A structurally-framed but invalid delta frame claiming the empty
        // sentinel for an already-occupied bucket must be rejected wholesale,
        // not silently applied — applying it would erase the live entry and
        // truncate the probe chain past it.
        let err = checkpoint
            .replay_frames(&[DeltaFrame {
                collection_id: [0; 16],
                bucket,
                generation: 0,
                slot: 0,
            }])
            .unwrap_err();
        assert!(matches!(err, DeltaReplayError::EmptySlot { bucket: b } if b == bucket));

        // The index must be left untouched: an owned clone is made before
        // replay, so this checks the caller's rescan fallback has a live
        // checkpoint copy to fall back to, not a partially-erased one.
        assert_eq!(checkpoint.len(), occupied_before);
        assert_eq!(checkpoint.lookup(&hash), Some((0, 100)));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let index = LossyIndex::new(128);
        for i in 0..50u16 {
            let mut h = [0u8; 16];
            h[0] = (i & 0xFF) as u8;
            h[8] = (i & 0xFF) as u8; // distinct tag bytes
            h[9] = (i >> 8) as u8;
            index.insert(&h, i % 3, u64::from(i) * 1000 + 1).unwrap();
        }

        let bytes = index.serialize();
        let restored = LossyIndex::deserialize(&bytes).unwrap();

        assert_eq!(restored.len(), index.len());
        for i in 0..50u16 {
            let mut h = [0u8; 16];
            h[0] = (i & 0xFF) as u8;
            h[8] = (i & 0xFF) as u8;
            h[9] = (i >> 8) as u8;
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
        let index = LossyIndex::new(n * 2);
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
        let tmp = LossyIndex::new(16);
        for i in 0..16u64 {
            let h = splitmix_hash(i + 5000);
            let tag = tmp.tag(&h);
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
