use memmap2::Mmap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

pub mod checkpoint;
pub mod delta;
pub mod format;
pub mod redo;

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
    /// Composes with the hard 16-entry minimum: `effective_min = floor.max(16)`.
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

/// Per-entry entry in the lossy fanout index.
///
/// Layout: `[16-bit tag | 16-bit slot | 32-bit offset]` packed into a `u64`.
///
/// - **tag** (high 16 bits): truncated fingerprint for fast rejection.
/// - **`slot`** (next 16 bits): which shard file this record lives in. The
///   field is a full 16 bits, but live shard IDs only ever reach
///   `crate::shard::MAX_SHARDS` (4096), so the top 4 bits of this field are
///   reserved — `IndexEntry::new` rejects a slot that is not a valid live ID.
///   They are *not* reclaimed as extra offset bits: `slot()` still decodes
///   all 16, and widening the offset would change the on-disk encoding.
/// - **offset** (low 32 bits): byte offset within the shard, stored as
///   `offset + 1` so that the all-zeros encoding is reserved as the empty
///   sentinel. Actual offset 0 is stored as 1, and `offset()` subtracts 1
///   to recover the real value.
///
/// Empty slots are all-zeros. This is safe regardless of tag value: the
/// `offset + 1` encoding guarantees a live entry's low 32 bits are never zero,
/// so the all-zeros pattern can never be produced by a real insert — the tag
/// bits play no role in reserving the sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry(u64);

impl IndexEntry {
    const EMPTY: Self = Self(0);

    /// Largest byte offset representable by the 32-bit offset field while
    /// reserving the all-zeros encoding for the empty-entry sentinel.
    pub const MAX_OFFSET: u64 = (1u64 << 32) - 2;

    const TAG_SHIFT: u64 = 48; // SHARD_BITS + OFFSET_BITS
    const SHARD_SHIFT: u64 = 32; // OFFSET_BITS
    const OFFSET_MASK: u64 = 0xFFFF_FFFF;

    /// Create a new entry from its components.
    ///
    /// The offset is stored as `offset + 1` so that the all-zeros `u64`
    /// encoding is reserved as the empty sentinel. Actual offset 0 is
    /// stored as 1 in the entry.
    ///
    /// # Panics
    /// Panics if `tag` exceeds 16 bits, `slot` is not a valid live shard ID
    /// (`>= crate::shard::MAX_SHARDS`), or `offset` exceeds
    /// [`Self::MAX_OFFSET`].
    #[must_use]
    pub fn new(tag: u32, slot: u16, offset: u64) -> Self {
        assert!(tag <= 0xFFFF, "tag must fit in 16 bits");
        assert!(
            usize::from(slot) < crate::shard::MAX_SHARDS,
            "slot {slot} is not a valid live shard ID (must be < MAX_SHARDS)"
        );
        assert!(
            offset <= Self::MAX_OFFSET,
            "offset must fit in 32 bits minus 1 (reserved for empty sentinel)"
        );
        Self(
            (u64::from(tag) << Self::TAG_SHIFT)
                | (u64::from(slot) << Self::SHARD_SHIFT)
                | ((offset.wrapping_add(1)) & Self::OFFSET_MASK),
        )
    }

    /// The sentinel value representing an empty entry.
    #[must_use]
    pub fn empty() -> Self {
        Self::EMPTY
    }

    /// Returns `true` if this entry is the empty sentinel.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The 16-bit tag stored in this entry.
    #[must_use]
    pub fn tag(self) -> u32 {
        ((self.0 >> Self::TAG_SHIFT) & 0xFFFF) as u32
    }

    /// The 16-bit shard ID stored in this entry.
    #[must_use]
    pub fn slot(self) -> u16 {
        ((self.0 >> Self::SHARD_SHIFT) & 0xFFFF) as u16
    }

    /// The byte offset within the shard stored in this entry.
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
/// - Empty entry terminates probe (write-once, no tombstones needed).
/// - Tag collisions coexist in the probe chain; a tag is only a candidate
///   filter and never an overwrite/equality proof.
/// - Owned indexes keep `homes` and `tails` identity side tables: two
///   anonymous `u64`s (16 bytes) per entry beyond the packed entry array.
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
    /// The flat entry array. Checkpoint-loaded indexes borrow the immutable
    /// raw slots from their mmap until their first write, which clones them
    /// into the owned atomic representation.
    slots: EntryStorage,
    /// The first 64 bits of each entry's hash. Index slots intentionally pack
    /// only a tag, shard, and offset; retaining the home hash separately lets
    /// a live writer grow the table without rescanning the packfiles.
    homes: Mutex<Vec<u64>>,
    /// The remaining 40 bits of each live entry's hash, plus a high-bit
    /// marker that says the identity is known. Together with `homes` and the
    /// packed tag this makes overwrite equality exact without putting
    /// full hashes in the checkpoint.
    tails: Mutex<Vec<u64>>,
    /// An index with incomplete identity tables remains readable, but must be
    /// rebuilt from packfiles if it ever needs to grow. Hydrated checkpoint
    /// indexes are growable without that pack scan.
    can_grow: bool,
    /// Number of occupied slots.
    len: AtomicU32,
    /// Longest linear-probe chain walked by any insert or lookup against
    /// this index instance so far (never reset, monotonically
    /// non-decreasing). Pure observability — insert has no probe-length
    /// cap (see the doc on `InsertError::TableFull`: growth is what keeps
    /// chains short, not a cap), so this exists to let an operator notice
    /// a probe chain growing unexpectedly (e.g. an unseeded/misconfigured
    /// deployment under adversarial content) without changing behavior.
    max_probe_len: AtomicU32,
}

#[derive(Debug)]
enum EntryStorage {
    Owned(Vec<AtomicU64>),
    Mmap { mmap: Arc<Mmap>, offset: usize },
}

impl EntryStorage {
    #[inline]
    fn get(&self, index: usize) -> u64 {
        match self {
            Self::Owned(slots) => slots[index].load(Ordering::Acquire),
            Self::Mmap { mmap, offset } => {
                let start = offset.saturating_add(index.saturating_mul(8));
                let bytes: [u8; 8] = mmap[start..start.saturating_add(8)]
                    .try_into()
                    .expect("validated checkpoint entry range");
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

impl LossyIndex {
    /// Whether the persisted identity tables are sufficient to rehash every
    /// occupied slot without consulting the packs. The probe-chain check is
    /// intentional: a corrupted home value could otherwise make `grow()` lose
    /// an entry while still looking superficially hydrated.
    fn identity_tables_complete(
        capacity: u32,
        shift: u32,
        slots: &EntryStorage,
        homes: &[u64],
        tails: &[u64],
    ) -> bool {
        let cap = capacity as usize;
        if homes.len() != cap || tails.len() != cap {
            return false;
        }
        let mask = usize::try_from(capacity.wrapping_sub(1)).unwrap_or(usize::MAX);
        (0..cap).all(|bucket| {
            if IndexEntry(slots.get(bucket)).is_empty() {
                return true;
            }
            if tails[bucket] >> 63 == 0 {
                return false;
            }
            let mut probe =
                usize::try_from((homes[bucket] >> shift) & u64::from(capacity.wrapping_sub(1)))
                    .unwrap_or(usize::MAX);
            while probe != bucket {
                if IndexEntry(slots.get(probe)).is_empty() {
                    return false;
                }
                probe = probe.wrapping_add(1) & mask;
            }
            true
        })
    }

    /// [`Self::identity_tables_complete`] over this index's current tables.
    fn identities_now_complete(&self) -> bool {
        let homes = self.homes.lock();
        let tails = self.tails.lock();
        Self::identity_tables_complete(self.capacity, self.shift, &self.slots, &homes, &tails)
    }

    /// A copy of an identity side table, or an all-zero (unhydrated) one of
    /// `capacity` entries if the source has none.
    fn identity_table(table: &Mutex<Vec<u64>>, capacity: u32) -> Vec<u64> {
        let table = table.lock();
        if table.len() == capacity as usize {
            table.clone()
        } else {
            vec![0; capacity as usize]
        }
    }
}

impl Clone for LossyIndex {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            mask: self.mask,
            shift: self.shift,
            config: self.config,
            slots: EntryStorage::Owned(self.slots.materialize(self.capacity as usize)),
            // Keep whatever identity tables the source holds. A mapping loaded
            // with its persisted `homes`/`tails` carries hydrated identities, and
            // dropping them here made every entry lose its identity on the first
            // write after a reopen (the next checkpoint then persisted zeros). A
            // mapping loaded without tables has none, so it gets zeroed ones.
            // Preserve the source's completeness gate: a hydrated checkpoint
            // remains growable after materialization, while an incomplete one
            // still falls back to a pack rebuild at a capacity boundary.
            homes: Mutex::new(Self::identity_table(&self.homes, self.capacity)),
            tails: Mutex::new(Self::identity_table(&self.tails, self.capacity)),
            can_grow: self.can_grow,
            len: AtomicU32::new(self.len.load(Ordering::Acquire)),
            max_probe_len: AtomicU32::new(self.max_probe_len.load(Ordering::Relaxed)),
        }
    }
}

/// The entry plus its retained home-hash used by a live index generation.
const LIVE_SLOT_BYTES: usize = std::mem::size_of::<IndexEntry>() + 2 * std::mem::size_of::<u64>();

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
    /// composes with the hard 16-entry minimum: `effective_min = floor.max(16)`.
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
            slots: EntryStorage::Owned((0..capacity_usize).map(|_| AtomicU64::new(0)).collect()),
            homes: Mutex::new(vec![0; capacity_usize]),
            tails: Mutex::new(vec![0; capacity_usize]),
            can_grow: true,
            len: AtomicU32::new(0),
            max_probe_len: AtomicU32::new(0),
        }
    }

    /// This index's entry-table capacity (always a power of two).
    #[must_use]
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Longest linear-probe chain observed so far by any insert or lookup
    /// against this index instance. Pure observability, never reset, never
    /// influences control flow — see the field doc on `max_probe_len`.
    #[must_use]
    pub fn max_probe_len(&self) -> u32 {
        self.max_probe_len.load(Ordering::Relaxed)
    }

    /// Record a probe chain of `len` steps as the new high-water mark if it
    /// exceeds the current one. `Relaxed` is enough: this is an
    /// observability counter with no ordering dependency on any other
    /// field, and a lost update under concurrent writers only means a
    /// transient undercount, never a wrong value that persists.
    #[inline]
    fn bump_max_probe_len(&self, len: u32) {
        self.max_probe_len.fetch_max(len, Ordering::Relaxed);
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

    /// The packed-entry tag for `hash`, exposed to crate-local recovery code
    /// that must validate a persisted entry against its authoritative frame.
    ///
    /// Must use this index's own seed: every stored entry's tag was computed
    /// via `self.tag(hash)` at insert time, so a recovery comparison against
    /// an unseeded (or differently-seeded) tag would fail closed on every
    /// occupied entry, not just colliding ones.
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

    /// Insert a (hash → `slot`, offset) mapping.
    ///
    /// # Errors
    /// Returns `InsertError::TableFull` if the table has less than 25% free slots
    /// and the hash is not already present (overwrites are always allowed).
    pub fn insert(&self, hash: &[u8; 16], slot: u16, offset: u64) -> Result<(), InsertError> {
        self.insert_tracked(hash, slot, offset).map(|_| ())
    }

    /// Like [`Self::insert`], but also reports the write's exact landing spot —
    /// the affected bucket and the packed entry value stored there — so the
    /// delta persistence layer can append a replay frame without re-probing.
    ///
    /// # Errors
    /// Returns `InsertError::TableFull` under the same conditions as
    /// [`Self::insert`] (the table is left unmodified).
    // `bucket` is always below `self.capacity`, which is capped at `u32::MAX`
    // by construction, so the checked `u32::try_from` in `insert_undoable`
    // below can never fail.
    pub fn insert_tracked(
        &self,
        hash: &[u8; 16],
        slot: u16,
        offset: u64,
    ) -> Result<(u32, u64), InsertError> {
        self.insert_undoable(hash, slot, offset)
            .map(|(bucket, value, _undo)| (bucket, value))
    }

    /// Like [`Self::insert_tracked`], but also returns a [`EntryUndo`]
    /// capturing the entry's prior contents, so a caller that needs to
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
    pub fn insert_undoable(
        &self,
        hash: &[u8; 16],
        slot: u16,
        offset: u64,
    ) -> Result<(u32, u64, EntryUndo), InsertError> {
        let tag = self.tag(hash);
        let raw_home = u64::from_be_bytes(hash[..8].try_into().unwrap());
        let home = self.mix_home(raw_home);
        let mut bucket = self.bucket_for_home(home);
        let mut probe_len: u32 = 0;

        loop {
            // `bucket` is masked by `self.mask` (capacity - 1) and capacity is
            // a u32, so the conversion is lossless; it is done once here so the
            // return/undo structs below don't each need a checked cast.
            let bucket_wire = u32::try_from(bucket)
                .expect("bucket is masked by the u32 capacity, so it always fits");
            let EntryStorage::Owned(slots) = &self.slots else {
                self.bump_max_probe_len(probe_len);
                return Err(InsertError::TableFull);
            };
            let entry = IndexEntry(slots[bucket].load(Ordering::Acquire));
            if entry.is_empty() {
                // Only reject when actually inserting into a new entry
                let threshold = self
                    .capacity
                    .wrapping_mul(u32::from(self.config.load_factor_percent))
                    / 100;
                if self.len.load(Ordering::Relaxed) >= threshold {
                    self.bump_max_probe_len(probe_len);
                    return Err(InsertError::TableFull);
                }
                let old_home = self.homes.lock()[bucket];
                let old_tail = self.tails.lock()[bucket];
                // Writers are serialized by the collection put lock. Publish
                // the entry last so concurrent readers see either the prior
                // empty entry or a complete record location.
                self.homes.lock()[bucket] = home;
                self.tails.lock()[bucket] = Self::tail(hash);
                let value = IndexEntry::new(tag, slot, offset).0;
                slots[bucket].store(value, Ordering::Release);
                self.len.fetch_add(1, Ordering::Relaxed);
                let undo = EntryUndo {
                    bucket: bucket_wire,
                    old_entry: 0,
                    old_home,
                    old_tail,
                    was_empty: true,
                };
                self.bump_max_probe_len(probe_len);
                return Ok((bucket_wire, value, undo));
            }
            // A tag only filters candidates; it never proves key equality.
            // Checkpoint-derived slots initially lack their 40-bit tail and
            // ask storage to hydrate it from the frame before deciding.
            if entry.tag() == tag {
                let tail = self.tails.lock()[bucket];
                if tail == 0 {
                    self.bump_max_probe_len(probe_len);
                    return Err(InsertError::NeedsIdentity {
                        bucket: bucket_wire,
                        slot: entry.slot(),
                        offset: entry.offset(),
                    });
                }
                if self.homes.lock()[bucket] == home && tail == Self::tail(hash) {
                    let old_entry = entry.0;
                    let old_home = self.homes.lock()[bucket];
                    let value = IndexEntry::new(tag, slot, offset).0;
                    slots[bucket].store(value, Ordering::Release);
                    let undo = EntryUndo {
                        bucket: bucket_wire,
                        old_entry,
                        old_home,
                        old_tail: tail,
                        was_empty: false,
                    };
                    self.bump_max_probe_len(probe_len);
                    return Ok((bucket_wire, value, undo));
                }
            }
            probe_len = probe_len.saturating_add(1);
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
    pub fn rollback_slot(&self, undo: &EntryUndo) {
        let EntryStorage::Owned(slots) = &self.slots else {
            return;
        };
        let bucket = undo.bucket as usize;
        self.homes.lock()[bucket] = undo.old_home;
        self.tails.lock()[bucket] = undo.old_tail;
        slots[bucket].store(undo.old_entry, Ordering::Release);
        if undo.was_empty {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Mark a checkpoint-derived entry with the full identity recovered from
    /// its authoritative frame. Returns `false` if the entry changed or the
    /// supplied hash cannot describe that entry.
    pub(crate) fn hydrate_slot_identity(&self, bucket: u32, hash: &[u8; 16]) -> bool {
        let bucket = bucket as usize;
        if bucket >= self.capacity as usize {
            return false;
        }
        let entry = IndexEntry(self.entry_at(bucket));
        if entry.is_empty() || entry.tag() != self.tag(hash) {
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
    /// session wrote them: an overwrite of a non-empty entry leaves the length
    /// unchanged, a write into an empty entry increments it. The index's `len`
    /// after replay therefore equals the checkpoint's occupancy plus the
    /// number of fresh-entry writes, which is what the sidecar integrity sum
    /// checks against.
    ///
    /// # Errors
    /// Returns `DeltaReplayError` if a frame targets a bucket outside this
    /// index's capacity, or carries the empty-entry sentinel (either is a
    /// structurally inconsistent log, rejected wholesale), or the index is
    /// still mmap-backed.
    pub fn replay_frames(&self, frames: &[DeltaFrame]) -> Result<(), DeltaReplayError> {
        let EntryStorage::Owned(slots) = &self.slots else {
            return Err(DeltaReplayError::RequiresOwnedIndex);
        };
        // Validate every frame before mutating any entry: the contract is
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
            // The packed entry's slot field must name a live shard. A writer
            // only ever encodes a slot from its own `ShardPool` (bounded by
            // `MAX_SHARDS`; see `IndexEntry::new`), so a value at or above the
            // cap can only be log corruption or a structurally invalid frame.
            // Accepting it would install an entry no shard scan can resolve —
            // `referenced_slot_ids` and friends skip `id >= MAX_SHARDS` — so
            // the record would silently vanish from recovery.
            let decoded = IndexEntry(frame.slot);
            if usize::from(decoded.slot()) >= crate::shard::MAX_SHARDS {
                return Err(DeltaReplayError::SlotOutOfRange {
                    bucket: frame.bucket,
                    slot: decoded.slot(),
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
    /// Returns `false` when the identity side tables are incomplete or the
    /// index cannot grow further. Complete checkpoint identities can rehash
    /// without reading packfiles; older/incomplete checkpoints retain the
    /// packfile rebuild fallback.
    pub fn grow(&self) -> Option<Self> {
        // The gate is checked when growth is used, not only when the index was
        // loaded: delta replay stores slot values straight into the table without
        // identities, so a table that was complete at load can be incomplete by
        // the time it has to grow.
        if !self.can_grow || self.capacity > u32::MAX / 2 || !self.identities_now_complete() {
            return None;
        }
        let grown = Self::with_config((self.capacity as usize).saturating_mul(2), self.config);
        let homes = self.homes.lock();
        let tails = self.tails.lock();
        for (index, home) in homes.iter().copied().enumerate() {
            let entry = IndexEntry(self.slots.get(index));
            if entry.is_empty() {
                continue;
            }
            let mut bucket = grown.bucket_for_home(home);
            while !IndexEntry(grown.entry_at(bucket)).is_empty() {
                bucket = bucket.wrapping_add(1) & grown.mask as usize;
            }
            grown.homes.lock()[bucket] = home;
            grown.tails.lock()[bucket] = tails[index];
            let EntryStorage::Owned(slots) = &grown.slots else {
                unreachable!("new index is owned")
            };
            slots[bucket].store(entry.0, Ordering::Relaxed);
            grown.len.fetch_add(1, Ordering::Relaxed);
        }
        Some(grown)
    }

    /// Double the table by recovering each occupied entry's full hash from
    /// its stored location.
    ///
    /// Checkpoint-backed indexes deliberately omit `homes`, so their first
    /// post-restart resize cannot use [`Self::grow`].  The packed slots still
    /// retain every `(slot, offset)`, however.  A caller can therefore
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
            let entry = IndexEntry(self.entry_at(index));
            if entry.is_empty() {
                continue;
            }
            locations.push((entry.slot(), entry.offset(), entry.tag()));
        }
        locations.sort_unstable_by_key(|(slot, offset, _)| (*slot, *offset));

        let grown = Self::with_config((self.capacity as usize).saturating_mul(2), self.config);
        for (slot, offset, slot_tag) in locations {
            let hash = hash_at(slot, offset, slot_tag)?;
            // A doubled table is at most 37.5% full because insertion only
            // requests growth at 75%, so this cannot hit TableFull.
            let _ = grown.insert(&hash, slot, offset);
        }
        Ok(Some(grown))
    }

    /// Look up a hash in the index.
    /// Returns `(slot, offset)` if found, `None` if absent.
    ///
    /// An empty entry terminates the probe — this is correct because:
    /// 1. Class A tables are write-once with no deletes (no tombstones needed).
    /// 2. An empty entry means the key was never inserted.
    #[inline]
    #[must_use]
    pub fn lookup(&self, hash: &[u8; 16]) -> Option<(u16, u64)> {
        self.lookup_all(hash).next()
    }

    /// Look up all candidate offsets for a hash, yielding tag collisions.
    ///
    /// Candidates with hydrated in-memory identity (home + tail) are verified
    /// before yielding, so tag collisions in a live/warm index are filtered
    /// out here rather than forcing the caller to disk. For a entry whose
    /// identity is not yet hydrated (e.g. a fresh checkpoint restore), the
    /// caller **must** verify the candidate against the caller-requested
    /// hash — a raw tag collision (~1/65536 at 16-bit tags) can still surface
    /// unverified.
    ///
    /// Yields `(slot, offset)` for each entry whose tag matches and isn't
    /// ruled out, then terminates at the first empty entry or after
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
            steps: 0,
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
    /// indexes above that — e.g. `mtxdb`'s collections, which start at
    /// `NEW_COLLECTION_INDEX_FLOOR` — must pass its own floor here, or this
    /// underestimates a small collection's real capacity).
    ///
    /// Returns `None` when `entries` is large enough that the rounded capacity
    /// would overflow `usize`. Callers that derive `entries` from a persisted
    /// node count treat this as malformed input rather than panicking.
    #[must_use]
    pub fn capacity_for_entries(entries: usize, min_capacity: usize) -> Option<usize> {
        let minimum = entries.saturating_mul(2).max(min_capacity.max(16));
        minimum.checked_next_power_of_two()
    }

    /// Memory that an index holding `entries` records would use after the
    /// standard two-times-capacity allocation policy is applied. See
    /// [`Self::capacity_for_entries`] for `min_capacity`. Returns `None` on the
    /// same overflow conditions as [`Self::capacity_for_entries`], or when the
    /// byte estimate itself would overflow.
    #[must_use]
    pub fn memory_usage_for_entries(entries: usize, min_capacity: usize) -> Option<usize> {
        Self::capacity_for_entries(entries, min_capacity)?
            .checked_mul(LIVE_SLOT_BYTES)?
            .checked_add(std::mem::size_of::<Self>())
    }

    /// Returns which shard IDs are referenced by at least one occupied entry.
    ///
    /// Used by shard retirement to determine which shards are still live
    /// across all collections before freeing a pool entry.
    #[must_use]
    pub fn referenced_slot_ids(&self) -> [bool; crate::shard::MAX_SHARDS] {
        let mut seen = [false; crate::shard::MAX_SHARDS];
        for index in 0..self.capacity as usize {
            let entry = IndexEntry(self.entry_at(index));
            if !entry.is_empty() {
                let id = entry.slot() as usize;
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
    pub fn slot_counts(&self) -> std::collections::HashMap<u16, u64> {
        let mut counts = std::collections::HashMap::new();
        for index in 0..self.capacity as usize {
            let entry = IndexEntry(self.entry_at(index));
            if !entry.is_empty() {
                let entry = counts.entry(entry.slot()).or_insert(0u64);
                *entry = entry.saturating_add(1);
            }
        }
        counts
    }

    /// Whether this collection's index currently has any live entry pointing
    /// into `slot`. Short-circuits on the first match — unlike
    /// `referenced_slot_ids`, which always builds a full `MAX_SHARDS`
    /// membership map, this is the cheap check for "does this one collection
    /// still reference this one shard," used to filter a shard-scan's
    /// candidate collection list down to collections that haven't already repacked
    /// past it.
    #[must_use]
    pub fn references_slot(&self, slot: u16) -> bool {
        (0..self.capacity as usize).any(|index| {
            let entry = IndexEntry(self.entry_at(index));
            !entry.is_empty() && entry.slot() == slot
        })
    }

    /// Rewrites every occupied entry's `slot` through `remap`, in place.
    ///
    /// A checkpoint-loaded index's `slot`s were encoded by the writer's
    /// own local `ShardPool` slots at checkpoint-write time — a process-local
    /// handle, not a stable identity. `remap` (built by the caller from the
    /// checkpoint's persisted pack table, translated through this reader's
    /// own currently-open packs) reconciles that against this reader's own
    /// entry numbering, which can differ once any shard has ever been retired
    /// (see `CHECKPOINT_VERSION`'s v6 doc comment).
    ///
    /// Checks first without touching storage: the common case (no shard has
    /// ever been retired, so a fresh reader's entry numbering already agrees
    /// with the checkpoint's) needs no rewrite at all, and mmap-backed
    /// indexes must stay mmap-backed when nothing actually changes — callers
    /// and tests depend on an unmodified checkpoint-loaded index staying
    /// zero-copy until its first real write. Only materializes into an owned
    /// copy (writes are never applied to the mmap) when at least one
    /// occupied entry's `slot` actually needs to change.
    ///
    /// Returns `false`, leaving the index unmodified, if any occupied entry's
    /// `slot` has no entry in `remap` — the checkpoint's pack table
    /// should cover every `slot` any of its own index entries use, so a
    /// miss means the checkpoint is internally inconsistent and the caller
    /// should fall back to a full rescan rather than serve unresolvable
    /// slots.
    #[must_use]
    pub fn remap_slots(&mut self, remap: &std::collections::HashMap<u16, u16>) -> bool {
        let capacity = self.capacity as usize;
        let mut needs_rewrite = false;
        for index in 0..capacity {
            let entry = IndexEntry(self.slots.get(index));
            if entry.is_empty() {
                continue;
            }
            let Some(&new_slot) = remap.get(&entry.slot()) else {
                return false;
            };
            if new_slot != entry.slot() {
                needs_rewrite = true;
            }
        }
        if !needs_rewrite {
            return true;
        }
        if !matches!(self.slots, EntryStorage::Owned(_)) {
            self.slots = EntryStorage::Owned(self.slots.materialize(capacity));
        }
        let EntryStorage::Owned(slots) = &self.slots else {
            unreachable!("just materialized to Owned above");
        };
        for cell in slots {
            let raw = cell.load(Ordering::Acquire);
            let entry = IndexEntry(raw);
            if entry.is_empty() {
                continue;
            }
            let Some(&new_slot) = remap.get(&entry.slot()) else {
                return false;
            };
            if new_slot == entry.slot() {
                continue;
            }
            let remapped = IndexEntry::new(entry.tag(), new_slot, entry.offset()).0;
            cell.store(remapped, Ordering::Relaxed);
        }
        true
    }

    /// Length of [`Self::serialize`]'s output, `8 + capacity * 24` bytes, known
    /// without building it.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        (self.capacity as usize)
            .saturating_mul(24)
            .saturating_add(8)
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
            let entry = IndexEntry(self.entry_at(index));
            buf.extend_from_slice(&entry.0.to_le_bytes());
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
            let entry = IndexEntry(val);
            if !entry.is_empty() {
                len = len.wrapping_add(1);
            }
            slots.push(AtomicU64::new(entry.0));
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

        let slots = EntryStorage::Owned(slots);
        let can_grow = Self::identity_tables_complete(capacity, shift, &slots, &homes, &tails);
        Ok(Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift,
            config,
            slots,
            homes: Mutex::new(homes),
            tails: Mutex::new(tails),
            can_grow,
            len: AtomicU32::new(len),
            max_probe_len: AtomicU32::new(0),
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
            slots: EntryStorage::Mmap { mmap, offset },
            homes: Mutex::new(Vec::new()),
            tails: Mutex::new(Vec::new()),
            can_grow: false,
            len: AtomicU32::new(len),
            max_probe_len: AtomicU32::new(0),
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
        let slots = EntryStorage::Mmap {
            mmap,
            offset: slots_offset,
        };
        let can_grow = Self::identity_tables_complete(
            capacity,
            64_u32.wrapping_sub(capacity.trailing_zeros()),
            &slots,
            &homes,
            &tails,
        );
        Self {
            mask: capacity.wrapping_sub(1),
            capacity,
            shift: 64_u32.wrapping_sub(capacity.trailing_zeros()),
            config,
            slots,
            homes: Mutex::new(homes),
            tails: Mutex::new(tails),
            can_grow,
            len: AtomicU32::new(len),
            max_probe_len: AtomicU32::new(0),
        }
    }

    /// True while the index borrows raw slots from a checkpoint mapping.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        matches!(self.slots, EntryStorage::Mmap { .. })
    }

    #[inline]
    fn entry_at(&self, index: usize) -> u64 {
        self.slots.get(index)
    }
}

/// Captures a single entry's prior contents so a live (uncloned) index write
/// made by [`LossyIndex::insert_undoable`] can be reversed by
/// [`LossyIndex::rollback_slot`] if a later entry in the same batch fails.
#[derive(Debug, Clone, Copy)]
pub struct EntryUndo {
    bucket: u32,
    old_entry: u64,
    old_home: u64,
    old_tail: u64,
    was_empty: bool,
}

impl EntryUndo {
    /// Whether the insertion occupied an empty slot rather than overwriting
    /// an existing key.
    #[must_use]
    pub fn was_empty(&self) -> bool {
        self.was_empty
    }
}

/// Errors that can occur while inserting into a [`LossyIndex`].
#[derive(Debug)]
pub enum InsertError {
    /// The table has reached 75% occupancy.
    /// Returned when the table reaches 75% occupancy to keep probe sequences short.
    TableFull,
    /// A checkpoint-derived same-tag entry needs its frame identity restored
    /// before insertion can determine whether it is an overwrite.
    NeedsIdentity {
        /// Probe bucket holding the candidate entry.
        bucket: u32,
        /// Candidate frame's shard.
        slot: u16,
        /// Candidate frame's offset.
        offset: u64,
    },
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableFull => write!(f, "index table too full"),
            Self::NeedsIdentity { .. } => write!(f, "index entry needs frame identity"),
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
/// Yields `(slot, offset)` for each entry whose tag matches and whose
/// in-memory identity (home + tail) matches the query hash, terminating at
/// the first empty entry or after the full capacity is probed.
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
    /// Steps walked so far, for `LossyIndex::max_probe_len` observability.
    steps: u32,
}

impl Iterator for LookupIter<'_> {
    type Item = (u16, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            self.remaining = self.remaining.wrapping_sub(1);
            let entry = IndexEntry(self.index.entry_at(self.bucket));
            if entry.is_empty() {
                self.index.bump_max_probe_len(self.steps);
                return None;
            }
            self.steps = self.steps.saturating_add(1);
            let current = self.bucket;
            self.bucket = self.bucket.wrapping_add(1) & self.mask;
            if entry.tag() != self.tag {
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
                    self.index.bump_max_probe_len(self.steps);
                    return Some((entry.slot(), entry.offset()));
                }
                // Tag matched but home/tail didn't — false positive, skip.
                continue;
            }
            // Checkpoint-backed without hydrated identity — yield and let
            // the caller verify against the packfile.
            self.index.bump_max_probe_len(self.steps);
            return Some((entry.slot(), entry.offset()));
        }
        self.index.bump_max_probe_len(self.steps);
        None
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[path = "test_index.rs"]
mod tests;
