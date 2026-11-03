//! Room-scoped bitmap index over event flags and dispositions.
//!
//! Maps dense room `u32` short IDs (as assigned in `room_auth_collection_id`)
//! to flag sets compressed via [`BitmapSet`]. Chunks are partitioned into 64K
//! slices (`short_id >> 16`), providing bounded chunk sizes and fast set algebra
//! (AND/OR/NOT) for flag queries.

use std::collections::{HashMap, HashSet};

use bytes::Bytes;

use crate::bitmap_set::{BitmapSet, DomainTag};
use crate::database::{Database, DatabaseTransaction};
use crate::interop::{
    auth_counter_record_id, auth_reverse_record_id, decode_auth_counter, decode_verdict, derive_id,
    event_record_id, event_verdict_collection_id, room_auth_collection_id,
    room_event_collection_id, room_flag_collection_id,
};
use crate::layout::ShardType;
use crate::storage::{NodeData, NodeId, StorageEngine, StorageError};
use crate::timeline::{room_scope_id, TimelineIndex};

/// Magic prefix for the flag index metadata record (`b"FLGM"`).
pub const FLAG_META_MAGIC: [u8; 4] = *b"FLGM";
/// Wire format version for the flag index metadata record.
pub const FLAG_META_VERSION: u8 = 1;
/// Reserved node ID for the flag index metadata record.
pub const FLAG_META_ID: NodeId = *b"MTXD-FLG-META-v1";
/// Reserved node ID for the flag generation counter record.
pub const FLAG_COUNTER_ID: NodeId = *b"MTXD-FLG-NEXTG-1";

/// Maximum short IDs per chunk (64K).
pub const CHUNK_CAPACITY: u32 = 65536;

const REVERSE_BATCH_SIZE: usize = 1024;
const REBUILD_BATCH_SIZE: u32 = 512;

/// Header metadata for a room's flag index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagMeta {
    /// Active generation number.
    pub generation: u64,
    /// Maximum short ID indexed in this generation.
    pub max_short_id: u32,
}

/// Event properties and dispositions tracked by [`RoomFlagIndex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Flag {
    /// The event is physically stored in the database.
    Present,
    /// The event belongs to the room timeline.
    Timeline,
    /// The event is stored as an outlier.
    Outlier,
    /// The event was rejected by auth checks.
    Rejected,
    /// The event soft-failed.
    SoftFailed,
    /// The event was redacted.
    Redacted,
    /// Caller-defined flag (values 0..=191).
    Custom(u8),
}

impl Flag {
    /// Maximum index for a custom caller-defined flag (`191`).
    pub const MAX_CUSTOM: u8 = 191;

    /// Construct a custom flag with an index in `0..=191`.
    ///
    /// Returns `None` if `index > MAX_CUSTOM`.
    #[must_use]
    pub const fn custom(index: u8) -> Option<Self> {
        if index <= Self::MAX_CUSTOM {
            Some(Self::Custom(index))
        } else {
            None
        }
    }

    /// Convert flag to a stable byte identifier.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::Present => 0,
            Self::Timeline => 1,
            Self::Outlier => 2,
            Self::Rejected => 3,
            Self::SoftFailed => 4,
            Self::Redacted => 5,
            Self::Custom(b) => {
                let clamped = if b > Self::MAX_CUSTOM {
                    Self::MAX_CUSTOM
                } else {
                    b
                };
                64_u8.saturating_add(clamped)
            }
        }
    }

    /// Reconstruct flag from a byte identifier.
    ///
    /// Returns `None` for reserved unallocated bytes (`6..=63`).
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Present),
            1 => Some(Self::Timeline),
            2 => Some(Self::Outlier),
            3 => Some(Self::Rejected),
            4 => Some(Self::SoftFailed),
            5 => Some(Self::Redacted),
            6..=63 => None,
            64..=255 => Some(Self::Custom(b.saturating_sub(64))),
        }
    }

    /// Human-readable label for the flag.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Timeline => "timeline",
            Self::Outlier => "outlier",
            Self::Rejected => "rejected",
            Self::SoftFailed => "soft-failed",
            Self::Redacted => "redacted",
            Self::Custom(_) => "custom",
        }
    }
}

/// Derive the record ID for one `(flag, chunk_idx)` bitmap chunk.
#[must_use]
pub fn flag_chunk_node_id(flag: Flag, chunk_idx: u16) -> NodeId {
    let mut key = [0_u8; 3];
    key[0] = flag.to_byte();
    key[1..3].copy_from_slice(&chunk_idx.to_be_bytes());
    derive_id(b"room:flag:chunk", &key)
}

pub(crate) fn encode_meta(meta: FlagMeta) -> [u8; 17] {
    let mut out = [0_u8; 17];
    out[0..4].copy_from_slice(&FLAG_META_MAGIC);
    out[4] = FLAG_META_VERSION;
    out[5..13].copy_from_slice(&meta.generation.to_be_bytes());
    out[13..17].copy_from_slice(&meta.max_short_id.to_be_bytes());
    out
}

pub(crate) fn decode_meta(bytes: &[u8]) -> Result<FlagMeta, StorageError> {
    if bytes.len() != 17 || !bytes.starts_with(&FLAG_META_MAGIC) || bytes[4] != FLAG_META_VERSION {
        return Err(StorageError::Corrupt(
            "malformed flag meta record".to_owned(),
        ));
    }
    let mut gen_bytes = [0_u8; 8];
    gen_bytes.copy_from_slice(&bytes[5..13]);
    let mut short_bytes = [0_u8; 4];
    short_bytes.copy_from_slice(&bytes[13..17]);
    Ok(FlagMeta {
        generation: u64::from_be_bytes(gen_bytes),
        max_short_id: u32::from_be_bytes(short_bytes),
    })
}

fn encode_counter(next: u64) -> [u8; 12] {
    let mut out = [0_u8; 12];
    out[0..4].copy_from_slice(b"FLGC");
    out[4..12].copy_from_slice(&next.to_be_bytes());
    out
}

fn decode_counter(bytes: &[u8]) -> Result<u64, StorageError> {
    if bytes.len() != 12 || bytes[0..4] != *b"FLGC" {
        return Err(StorageError::Corrupt(
            "malformed flag generation counter".to_owned(),
        ));
    }
    let mut b = [0_u8; 8];
    b.copy_from_slice(&bytes[4..12]);
    Ok(u64::from_be_bytes(b))
}

/// A mutation to apply to an event's flag in the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlagChange {
    /// Room-scoped short id of the event.
    pub short_id: u32,
    /// The flag being modified.
    pub flag: Flag,
    /// Whether to set (`true`) or clear (`false`) the flag.
    pub set: bool,
}

impl FlagChange {
    /// Create a change that sets `flag` for `short_id`.
    #[must_use]
    pub const fn set(short_id: u32, flag: Flag) -> Self {
        Self {
            short_id,
            flag,
            set: true,
        }
    }

    /// Create a change that clears `flag` for `short_id`.
    #[must_use]
    pub const fn clear(short_id: u32, flag: Flag) -> Self {
        Self {
            short_id,
            flag,
            set: false,
        }
    }
}

/// Boolean expression over flags within a room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagQuery {
    /// Matches events that have `flag` set.
    Has(Flag),
    /// Matches events that do NOT have `flag` set (evaluated as `Present \ flag`).
    Not(Flag),
    /// Conjunction of queries (all must match).
    And(Vec<FlagQuery>),
    /// Disjunction of queries (any must match).
    Or(Vec<FlagQuery>),
}

impl FlagQuery {
    /// Query for events having `flag`.
    #[must_use]
    pub const fn has(flag: Flag) -> Self {
        Self::Has(flag)
    }

    /// Query for events not having `flag` (relative to `Present`).
    #[must_use]
    pub const fn not(flag: Flag) -> Self {
        Self::Not(flag)
    }

    /// Query for events satisfying all subqueries.
    #[must_use]
    pub fn and<I: IntoIterator<Item = Self>>(queries: I) -> Self {
        Self::And(queries.into_iter().collect())
    }

    /// Query for events satisfying at least one subquery.
    #[must_use]
    pub fn or<I: IntoIterator<Item = Self>>(queries: I) -> Self {
        Self::Or(queries.into_iter().collect())
    }
}

/// Room-scoped flag index providing popcount counts and matching event selections.
#[derive(Debug, Clone)]
pub struct RoomFlagIndex {
    pool: ShardType,
    collection: [u8; 16],
    auth_pool: ShardType,
    auth_collection: [u8; 16],
    domain: DomainTag,
    room_id: Vec<u8>,
}

impl RoomFlagIndex {
    /// Create a room flag index using default event pool for flags and edge pool for auth.
    #[must_use]
    pub fn new(room_id: &[u8]) -> Self {
        Self::with_pool(ShardType::EventDag, room_id)
    }

    /// Create a room flag index with a specific shard pool for the flags collection.
    #[must_use]
    pub fn with_pool(pool: ShardType, room_id: &[u8]) -> Self {
        Self::with_pools(pool, ShardType::Edges, room_id)
    }

    /// Create a room flag index specifying both the flags pool and the auth pool.
    #[must_use]
    pub fn with_pools(flag_pool: ShardType, auth_pool: ShardType, room_id: &[u8]) -> Self {
        let collection = room_flag_collection_id(room_id);
        let auth_collection = room_auth_collection_id(room_id);
        let domain = DomainTag::for_scope(auth_pool, &auth_collection);
        Self {
            pool: flag_pool,
            collection,
            auth_pool,
            auth_collection,
            domain,
            room_id: room_id.to_vec(),
        }
    }

    /// The shard pool holding the flags collection.
    #[must_use]
    pub const fn pool(&self) -> ShardType {
        self.pool
    }

    /// The collection ID for this room's flag chunks.
    #[must_use]
    pub const fn collection(&self) -> &[u8; 16] {
        &self.collection
    }

    /// The collection ID for this room's auth / short-id mappings.
    #[must_use]
    pub const fn auth_collection(&self) -> &[u8; 16] {
        &self.auth_collection
    }

    /// The domain tag used by this room's bitmap chunks.
    #[must_use]
    pub const fn domain(&self) -> DomainTag {
        self.domain
    }

    /// Room ID this index was built for.
    #[must_use]
    pub fn room_id(&self) -> &[u8] {
        &self.room_id
    }

    /// Derive the collection holding chunk records for a given generation.
    ///
    /// Every generation gets its own isolated collection so rebuilds do not
    /// mutate or delete live records until atomically published, and retired
    /// generations can be deleted cleanly.
    #[must_use]
    pub fn chunk_collection(&self, generation: u64) -> [u8; 16] {
        let mut key = Vec::with_capacity(self.room_id.len().saturating_add(8));
        key.extend_from_slice(&self.room_id);
        key.extend_from_slice(&generation.to_be_bytes());
        derive_id(b"room:flag:gen", &key)
    }

    /// Read the flag index metadata (generation, maximum short ID, and write count).
    ///
    /// # Errors
    /// Returns [`StorageError`] if database reads fail or metadata is corrupt.
    pub fn meta(&self, db: &Database) -> Result<FlagMeta, StorageError> {
        let meta_raw = db.pool(self.pool).get(&self.collection, &FLAG_META_ID)?;
        match meta_raw {
            Some(data) => decode_meta(&data.bytes),
            None => Ok(FlagMeta {
                generation: 1,
                max_short_id: 0,
            }),
        }
    }

    /// Read the maximum short ID recorded in the flag index metadata.
    ///
    /// # Errors
    /// Returns [`StorageError`] if database reads fail or metadata is corrupt.
    pub fn max_short_id(&self, db: &Database) -> Result<u32, StorageError> {
        self.meta(db).map(|m| m.max_short_id)
    }

    /// Whether one event currently has `flag` set.
    ///
    /// Reads only the chunk containing `short_id`. Use this to avoid staging
    /// a flag rewrite when the desired bit is already in that state.
    ///
    /// # Errors
    /// Returns [`StorageError`] on read error or corrupt index payload.
    pub fn has_flag(&self, db: &Database, short_id: u32, flag: Flag) -> Result<bool, StorageError> {
        let meta = self.meta(db)?;
        if short_id > meta.max_short_id {
            return Ok(false);
        }
        let chunk_col = self.chunk_collection(meta.generation);
        let chunk_idx = u16::try_from(short_id / CHUNK_CAPACITY)
            .map_err(|_| StorageError::Corrupt("flag chunk index out of range".to_owned()))?;
        let chunk = self.get_or_load_chunk(db, &chunk_col, flag, chunk_idx, &mut HashMap::new())?;
        Ok(chunk.contains(short_id % CHUNK_CAPACITY))
    }

    /// Read several flags for many short ids, decoding each chunk once.
    ///
    /// Returns one bitmask per short id. Bit `i` is set when `flags[i]` is
    /// present for that short id.
    ///
    /// # Errors
    /// Returns [`StorageError`] on read error or corrupt index payload.
    pub fn flags_for_shorts(
        &self,
        db: &Database,
        shorts: &[u32],
        flags: &[Flag],
    ) -> Result<Vec<u32>, StorageError> {
        let mut result = vec![0_u32; shorts.len()];
        let meta = self.meta(db)?;
        if meta.max_short_id == 0 || shorts.is_empty() {
            return Ok(result);
        }
        let chunk_col = self.chunk_collection(meta.generation);
        let mut cache: HashMap<(Flag, u16), BitmapSet> = HashMap::new();
        for (position, &short_id) in shorts.iter().enumerate() {
            if short_id > meta.max_short_id {
                continue;
            }
            let chunk_idx = u16::try_from(short_id / CHUNK_CAPACITY)
                .map_err(|_| StorageError::Corrupt("flag chunk index out of range".to_owned()))?;
            let offset = short_id % CHUNK_CAPACITY;
            for (bit, &flag) in flags.iter().enumerate() {
                let chunk = self.get_or_load_chunk(db, &chunk_col, flag, chunk_idx, &mut cache)?;
                if chunk.contains(offset) {
                    result[position] |= 1_u32 << bit;
                }
            }
        }
        Ok(result)
    }

    /// Stage a batch of flag changes directly into a specified chunk collection.
    ///
    /// Returns the maximum short ID encountered in `changes`.
    ///
    /// # Errors
    /// Returns [`StorageError`] on read error or payload encoding failure.
    pub fn stage_into_collection(
        &self,
        tx: &DatabaseTransaction<'_>,
        chunk_col: [u8; 16],
        changes: &[FlagChange],
    ) -> Result<u32, StorageError> {
        Ok(self.stage_into_collection_inner(tx, chunk_col, changes)?.0)
    }

    fn stage_into_collection_inner(
        &self,
        tx: &DatabaseTransaction<'_>,
        chunk_col: [u8; 16],
        changes: &[FlagChange],
    ) -> Result<(u32, bool), StorageError> {
        if changes.is_empty() {
            return Ok((0, false));
        }

        let mut grouped: HashMap<(Flag, u16), Vec<(u32, bool)>> = HashMap::new();
        let mut max_short_seen = 0_u32;
        let mut mutated = false;

        for change in changes {
            if change.short_id > max_short_seen {
                max_short_seen = change.short_id;
            }
            let chunk_idx = u16::try_from(change.short_id / CHUNK_CAPACITY)
                .map_err(|_| StorageError::Corrupt("short id exceeds chunk capacity".to_owned()))?;
            let offset = change.short_id % CHUNK_CAPACITY;
            grouped
                .entry((change.flag, chunk_idx))
                .or_default()
                .push((offset, change.set));
        }

        for ((flag, chunk_idx), chunk_changes) in grouped {
            let chunk_id = flag_chunk_node_id(flag, chunk_idx);
            let (existing_nodes, versions) =
                tx.get_with_record_versions(self.pool, &chunk_col, &[chunk_id])?;
            let token = versions.first().copied().unwrap_or(0);
            let existing_data = existing_nodes.into_iter().next().flatten();

            let mut bitmap = match existing_data.as_ref() {
                Some(data) => BitmapSet::decode_in_domain(&data.bytes, self.domain)?,
                None => BitmapSet::new(self.domain),
            };

            for (offset, set) in chunk_changes {
                if set {
                    bitmap.insert(offset);
                } else {
                    bitmap.remove(offset);
                }
            }

            let encoded = bitmap.encode()?;
            if existing_data
                .as_ref()
                .is_some_and(|data| data.bytes.as_ref() == encoded.as_slice())
                || (existing_data.is_none() && bitmap.is_empty())
            {
                continue;
            }
            tx.put(
                self.pool,
                chunk_col,
                chunk_id,
                &NodeData::new(Bytes::from(encoded)),
            )?;
            mutated = true;
            // If the record was already staged in this transaction, token is 0.
            // Avoid adding a bogus version 0 expectation that fails commit.
            if existing_data.is_none() || token != 0 {
                tx.expect_record_version(self.pool, chunk_col, chunk_id, token)?;
            }
        }

        Ok((max_short_seen, mutated))
    }

    /// Stage a batch of flag changes into the transaction for the active generation.
    ///
    /// Changes are grouped and applied chunk-by-chunk in memory, writing each
    /// touched chunk record exactly once. Bumps `max_short_id` and asserts
    /// `FLAG_META_ID` version so a concurrent rebuild or writer is detected
    /// rather than silently overwritten.
    ///
    /// # Errors
    /// Returns [`StorageError`] if reading existing chunks or staging writes fails.
    pub fn stage(
        &self,
        tx: &DatabaseTransaction<'_>,
        changes: &[FlagChange],
    ) -> Result<(), StorageError> {
        if changes.is_empty() {
            return Ok(());
        }

        let (meta_nodes, meta_versions) =
            tx.get_with_record_versions(self.pool, &self.collection, &[FLAG_META_ID])?;
        let meta_token = meta_versions.first().copied().unwrap_or(0);
        let meta_raw = meta_nodes.into_iter().next().flatten();
        let current_meta = match meta_raw.as_ref() {
            Some(data) => decode_meta(&data.bytes)?,
            None => FlagMeta {
                generation: 1,
                max_short_id: 0,
            },
        };

        let chunk_col = self.chunk_collection(current_meta.generation);
        let (max_short_seen, mutated) = self.stage_into_collection_inner(tx, chunk_col, changes)?;
        if !mutated {
            return Ok(());
        }

        if !(meta_raw.is_some() && meta_token == 0) {
            tx.expect_record_version(self.pool, self.collection, FLAG_META_ID, meta_token)?;
        }
        // Every writer updates this CAS record, including changes that do not
        // increase max_short_id. Rebuild publication checks the same record,
        // so a flag update to an existing event cannot race the generation swap.
        let new_meta = FlagMeta {
            generation: current_meta.generation,
            max_short_id: current_meta.max_short_id.max(max_short_seen),
        };
        let meta_encoded = encode_meta(new_meta);
        tx.put(
            self.pool,
            self.collection,
            FLAG_META_ID,
            &NodeData::new(Bytes::from(meta_encoded.to_vec())),
        )?;
        Ok(())
    }

    /// Maximum retries for [`Self::update`] on concurrent transaction conflict.
    pub const MAX_RETRIES: usize = 64;

    /// Apply flag changes atomically with a retry loop on CAS conflicts.
    ///
    /// # Errors
    /// Returns [`StorageError::StaleRead`] if retries are exhausted, or propagates backend errors.
    pub fn update(&self, db: &Database, changes: &[FlagChange]) -> Result<(), StorageError> {
        let mut last_error = None;
        for _ in 0..Self::MAX_RETRIES {
            let tx = db.begin_transaction();
            match self.stage(&tx, changes) {
                Ok(()) => match tx.commit() {
                    Ok(()) => return Ok(()),
                    Err(e) if e.is_stale_read() || matches!(e, StorageError::WouldBlock(_)) => {
                        last_error = Some(e);
                    }
                    Err(e) => return Err(e),
                },
                Err(e) if e.is_stale_read() || matches!(e, StorageError::WouldBlock(_)) => {
                    last_error = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_error.unwrap_or(StorageError::StaleRead {
            pool: self.pool,
            collection_id: self.collection,
            expected: 0,
            actual: 0,
        }))
    }

    /// Helper to stage multiple flags for a single event short ID.
    ///
    /// # Errors
    /// Returns [`StorageError`] on stage failure.
    pub fn stage_event_flags(
        &self,
        tx: &DatabaseTransaction<'_>,
        short_id: u32,
        flags: &[Flag],
    ) -> Result<(), StorageError> {
        let changes: Vec<FlagChange> = flags
            .iter()
            .map(|&flag| FlagChange::set(short_id, flag))
            .collect();
        self.stage(tx, &changes)
    }

    fn get_or_load_chunk(
        &self,
        db: &Database,
        chunk_col: &[u8; 16],
        flag: Flag,
        chunk_idx: u16,
        cache: &mut HashMap<(Flag, u16), BitmapSet>,
    ) -> Result<BitmapSet, StorageError> {
        if let Some(set) = cache.get(&(flag, chunk_idx)) {
            return Ok(set.clone());
        }
        let chunk_id = flag_chunk_node_id(flag, chunk_idx);
        let set = match db.pool(self.pool).get(chunk_col, &chunk_id)? {
            Some(data) => BitmapSet::decode_in_domain(&data.bytes, self.domain)?,
            None => BitmapSet::new(self.domain),
        };
        cache.insert((flag, chunk_idx), set.clone());
        Ok(set)
    }

    fn eval_chunk(
        &self,
        db: &Database,
        chunk_col: &[u8; 16],
        chunk_idx: u16,
        query: &FlagQuery,
        cache: &mut HashMap<(Flag, u16), BitmapSet>,
    ) -> Result<BitmapSet, StorageError> {
        match query {
            FlagQuery::Has(flag) => self.get_or_load_chunk(db, chunk_col, *flag, chunk_idx, cache),
            FlagQuery::Not(flag) => {
                let present =
                    self.get_or_load_chunk(db, chunk_col, Flag::Present, chunk_idx, cache)?;
                let target = self.get_or_load_chunk(db, chunk_col, *flag, chunk_idx, cache)?;
                present.difference(&target)
            }
            FlagQuery::And(sub) => {
                if sub.is_empty() {
                    return self.get_or_load_chunk(db, chunk_col, Flag::Present, chunk_idx, cache);
                }
                let mut iter = sub.iter();
                let first = iter.next().expect("non-empty");
                let mut result = self.eval_chunk(db, chunk_col, chunk_idx, first, cache)?;
                for next in iter {
                    if result.is_empty() {
                        return Ok(result);
                    }
                    let next_res = self.eval_chunk(db, chunk_col, chunk_idx, next, cache)?;
                    result = result.intersection(&next_res)?;
                }
                Ok(result)
            }
            FlagQuery::Or(sub) => {
                if sub.is_empty() {
                    return Ok(BitmapSet::new(self.domain));
                }
                let mut iter = sub.iter();
                let first = iter.next().expect("non-empty");
                let mut result = self.eval_chunk(db, chunk_col, chunk_idx, first, cache)?;
                for next in iter {
                    let next_res = self.eval_chunk(db, chunk_col, chunk_idx, next, cache)?;
                    result = result.union(&next_res)?;
                }
                Ok(result)
            }
        }
    }

    /// Count all records in the room matching `query`.
    ///
    /// Evaluates bitwise chunk operations in memory across chunk records.
    ///
    /// # Errors
    /// Returns [`StorageError`] on read error or corrupt index payload.
    pub fn count(&self, db: &Database, query: &FlagQuery) -> Result<u64, StorageError> {
        let meta = self.meta(db)?;
        if meta.max_short_id == 0 {
            return Ok(0);
        }
        let chunk_col = self.chunk_collection(meta.generation);
        let max_chunk = (meta.max_short_id / CHUNK_CAPACITY) as u16;
        let mut total = 0_u64;
        let mut cache = HashMap::new();

        for chunk_idx in 0..=max_chunk {
            cache.clear();
            let result = self.eval_chunk(db, &chunk_col, chunk_idx, query, &mut cache)?;
            total = total.saturating_add(result.len());
        }

        Ok(total)
    }

    /// Select all matching short IDs in the room.
    ///
    /// # Errors
    /// Returns [`StorageError`] on storage failure.
    pub fn select_shorts(
        &self,
        db: &Database,
        query: &FlagQuery,
    ) -> Result<Vec<u32>, StorageError> {
        self.select_shorts_bounded(db, query, usize::MAX)
    }

    /// Select matching short IDs up to `limit`.
    ///
    /// # Errors
    /// Returns [`StorageError`] on storage failure.
    pub fn select_shorts_bounded(
        &self,
        db: &Database,
        query: &FlagQuery,
        limit: usize,
    ) -> Result<Vec<u32>, StorageError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let meta = self.meta(db)?;
        if meta.max_short_id == 0 {
            return Ok(Vec::new());
        }
        let chunk_col = self.chunk_collection(meta.generation);
        let max_chunk = (meta.max_short_id / CHUNK_CAPACITY) as u16;
        let mut shorts = Vec::new();
        let mut cache = HashMap::new();

        for chunk_idx in 0..=max_chunk {
            cache.clear();
            let result = self.eval_chunk(db, &chunk_col, chunk_idx, query, &mut cache)?;
            let chunk_base = u32::from(chunk_idx).saturating_mul(CHUNK_CAPACITY);
            for offset in result.iter() {
                shorts.push(chunk_base.saturating_add(offset));
                if shorts.len() >= limit {
                    return Ok(shorts);
                }
            }
        }

        Ok(shorts)
    }

    /// Select all matching event IDs (bytes) in the room.
    ///
    /// Looks up each matching short ID in the room's auth reverse records.
    ///
    /// # Errors
    /// Returns [`StorageError`] on storage failure.
    pub fn select(&self, db: &Database, query: &FlagQuery) -> Result<Vec<Vec<u8>>, StorageError> {
        self.select_bounded(db, query, usize::MAX)
    }

    /// Select matching event IDs up to `limit`.
    ///
    /// # Errors
    /// Returns [`StorageError`] on storage failure.
    pub fn select_bounded(
        &self,
        db: &Database,
        query: &FlagQuery,
        limit: usize,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        let shorts = self.select_shorts_bounded(db, query, limit)?;
        if shorts.is_empty() {
            return Ok(Vec::new());
        }

        let mut event_ids = Vec::with_capacity(shorts.len());
        for batch in shorts.chunks(REVERSE_BATCH_SIZE) {
            let record_ids: Vec<NodeId> = batch
                .iter()
                .map(|&short| auth_reverse_record_id(short))
                .collect();
            let nodes = db
                .pool(self.auth_pool)
                .get_many(&self.auth_collection, &record_ids)?;
            for node in nodes.into_iter().flatten() {
                event_ids.push(node.bytes.to_vec());
            }
        }

        Ok(event_ids)
    }

    /// Reserve a never-reused generation number for rebuild.
    ///
    /// Reads the current active metadata and counter record, reserving a new
    /// generation `> current_meta.generation` by committing an increment to
    /// [`FLAG_COUNTER_ID`].
    ///
    /// Returns `(current_meta, new_generation, meta_token)`.
    ///
    /// # Errors
    /// Returns [`StorageError`] on read error, corrupt counter, or exhausted retries.
    pub fn reserve_rebuild_generation(
        &self,
        db: &Database,
    ) -> Result<(FlagMeta, u64, u64), StorageError> {
        let mut last = None;
        for _ in 0..Self::MAX_RETRIES {
            let tx = db.begin_transaction();
            let (meta_nodes, meta_versions) =
                tx.get_with_record_versions(self.pool, &self.collection, &[FLAG_META_ID])?;
            let meta_token = meta_versions.first().copied().unwrap_or(0);
            let meta_raw = meta_nodes.into_iter().next().flatten();
            let current_meta = match meta_raw.as_ref() {
                Some(data) => decode_meta(&data.bytes)?,
                None => FlagMeta {
                    generation: 1,
                    max_short_id: 0,
                },
            };

            let (counter_nodes, counter_versions) =
                tx.get_with_record_versions(self.pool, &self.collection, &[FLAG_COUNTER_ID])?;
            let counter_token = counter_versions.first().copied().unwrap_or(0);
            let counter_raw = counter_nodes.into_iter().next().flatten();
            let next_alloc = match counter_raw.as_ref() {
                Some(data) => decode_counter(&data.bytes)?,
                None => current_meta.generation.saturating_add(1).max(2),
            };

            let new_gen = next_alloc.max(current_meta.generation.saturating_add(1));
            let next_counter = new_gen
                .checked_add(1)
                .ok_or_else(|| StorageError::Internal("flag generation overflow".to_owned()))?;

            tx.put(
                self.pool,
                self.collection,
                FLAG_COUNTER_ID,
                &NodeData::new(Bytes::from(encode_counter(next_counter).to_vec())),
            )?;
            if !(counter_raw.is_some() && counter_token == 0) {
                tx.expect_record_version(
                    self.pool,
                    self.collection,
                    FLAG_COUNTER_ID,
                    counter_token,
                )?;
            }

            match tx.commit() {
                Ok(()) => return Ok((current_meta, new_gen, meta_token)),
                Err(e) if e.is_stale_read() || matches!(e, StorageError::WouldBlock(_)) => {
                    last = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or(StorageError::StaleRead {
            pool: self.pool,
            collection_id: self.collection,
            expected: 0,
            actual: 0,
        }))
    }

    /// Rebuild the flag index for this room from existing auth short IDs and verdicts.
    ///
    /// Rebuilds into a new generation collection without mutating or deleting
    /// the active index. Upon completion, swaps the active generation pointer atomically
    /// via CAS. If rebuild fails, active index is unchanged and the target collection is cleaned up.
    ///
    /// Returns the number of events indexed.
    ///
    /// # Errors
    /// Returns [`StorageError`] if database reads, staging, or CAS publication fails.
    #[allow(clippy::too_many_lines)]
    pub fn rebuild(&self, db: &Database) -> Result<u64, StorageError> {
        // Reserve the target generation before reading the auth counter. The
        // reservation keeps generation numbers from ever being reused, and
        // reading the counter afterwards keeps the scan bound consistent with
        // the snapshot the rebuilt index is published against.
        let (current_meta, new_gen, meta_token) = self.reserve_rebuild_generation(db)?;
        let counter_raw = db
            .pool(self.auth_pool)
            .get(&self.auth_collection, &auth_counter_record_id())?;
        let counter = match counter_raw {
            Some(data) => decode_auth_counter(&data.bytes)?,
            None => return Ok(0),
        };
        if counter == 0 {
            return Ok(0);
        }

        let target_col = self.chunk_collection(new_gen);

        // Pre-emptively clear target collection before writing. A failed delete
        // or commit must stop here: otherwise stale data from a crashed rebuild
        // could be published alongside this rebuild's chunks.
        let prep_tx = db.begin_transaction();
        prep_tx.delete_collection(self.pool, target_col)?;
        prep_tx.commit()?;

        match self.rebuild_generation(db, counter, current_meta, new_gen, target_col, meta_token) {
            Ok(total) => Ok(total),
            Err(e) => {
                let cleanup_tx = db.begin_transaction();
                if let Err(cleanup_err) = cleanup_tx.delete_collection(self.pool, target_col) {
                    eprintln!("room flag rebuild: failed to delete target collection {new_gen}: {cleanup_err}");
                } else if let Err(cleanup_err) = cleanup_tx.commit() {
                    eprintln!("room flag rebuild: failed to commit cleanup of target collection {new_gen}: {cleanup_err}");
                }
                Err(e)
            }
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn rebuild_generation(
        &self,
        db: &Database,
        counter: u32,
        current_meta: FlagMeta,
        new_gen: u64,
        target_col: [u8; 16],
        meta_token: u64,
    ) -> Result<u64, StorageError> {
        // Query timeline index for timeline events in this room
        let room_name = String::from_utf8_lossy(&self.room_id);
        let timeline_index =
            TimelineIndex::open(db.pool(ShardType::EventDag).as_ref(), room_name.as_ref());
        let room_scope = room_scope_id(&self.room_id);
        let mut cursor = None;
        let mut timeline_nodes = HashSet::new();
        loop {
            let page = timeline_index.page(&room_scope, cursor.as_ref(), true, 1024)?;
            if page.entries.is_empty() {
                break;
            }
            for entry in &page.entries {
                timeline_nodes.insert(entry.node_id);
            }
            if page.next_cursor.is_none() {
                break;
            }
            cursor = page.next_cursor;
        }

        let event_col = room_event_collection_id(&self.room_id);
        let mut total_indexed = 0_u64;
        let mut max_short_seen = 0_u32;

        let mut start_id = 1_u32;
        while start_id <= counter {
            let end_id = start_id
                .saturating_add(REBUILD_BATCH_SIZE)
                .saturating_sub(1)
                .min(counter);
            let mut changes = Vec::new();

            // 1. Fetch reverse records for this batch of shorts
            let shorts: Vec<u32> = (start_id..=end_id).collect();
            let reverse_keys: Vec<NodeId> =
                shorts.iter().map(|&s| auth_reverse_record_id(s)).collect();
            let reverse_nodes = db
                .pool(self.auth_pool)
                .get_many(&self.auth_collection, &reverse_keys)?;

            // 2. Map existing events and check event presence + verdicts
            let mut event_node_pairs = Vec::new();
            for (&short, node_opt) in shorts.iter().zip(reverse_nodes) {
                if let Some(data) = node_opt {
                    let event_id = data.bytes.to_vec();
                    let node_id = event_record_id(&event_id);
                    event_node_pairs.push((short, event_id, node_id));
                }
            }

            if !event_node_pairs.is_empty() {
                let event_nodes: Vec<NodeId> =
                    event_node_pairs.iter().map(|(_, _, node)| *node).collect();
                let event_data = db
                    .pool(ShardType::EventDag)
                    .get_many(&event_col, &event_nodes)?;

                let mut present_events = Vec::new();
                for ((short, event_id, node_id), event_data) in
                    event_node_pairs.into_iter().zip(event_data)
                {
                    if event_data.is_some() {
                        changes.push(FlagChange::set(short, Flag::Present));
                        if timeline_nodes.contains(&node_id) {
                            changes.push(FlagChange::set(short, Flag::Timeline));
                        }
                        present_events.push((short, event_id, node_id));
                        total_indexed = total_indexed.saturating_add(1);
                    }
                }

                // Check verdicts for present events
                for (short, _event_id, node_id) in present_events {
                    let verdict_col = event_verdict_collection_id(&node_id);
                    if let Some(v_data) =
                        db.pool(ShardType::EventDag).get(&verdict_col, &node_id)?
                    {
                        let (_, verdict) = decode_verdict(&v_data.bytes)?;
                        if verdict.outlier {
                            changes.push(FlagChange::set(short, Flag::Outlier));
                        }
                        if verdict.soft_fail.is_some() {
                            changes.push(FlagChange::set(short, Flag::SoftFailed));
                        }
                        if verdict.rejection.is_some() {
                            changes.push(FlagChange::set(short, Flag::Rejected));
                        }
                        if verdict.redacted_by.is_some() {
                            changes.push(FlagChange::set(short, Flag::Redacted));
                        }
                    }
                }
            }

            if !changes.is_empty() {
                let tx = db.begin_transaction();
                let batch_max = self.stage_into_collection(&tx, target_col, &changes)?;
                if batch_max > max_short_seen {
                    max_short_seen = batch_max;
                }
                tx.commit()?;
            }

            start_id = end_id.saturating_add(1);
        }

        // Atomically publish new generation via CAS
        let publish_tx = db.begin_transaction();
        let (meta_nodes, meta_versions) =
            publish_tx.get_with_record_versions(self.pool, &self.collection, &[FLAG_META_ID])?;
        let live_meta_token = meta_versions.first().copied().unwrap_or(0);
        let live_meta_raw = meta_nodes.into_iter().next().flatten();
        let live_meta = match live_meta_raw.as_ref() {
            Some(data) => decode_meta(&data.bytes)?,
            None => FlagMeta {
                generation: 1,
                max_short_id: 0,
            },
        };
        // Verify that no competing rebuild or writer published while this one
        // was running. Every writer CASes FLAG_META_ID, even when max_short_id
        // is unchanged, so this also detects updates to existing short IDs.
        if live_meta.generation != current_meta.generation {
            return Err(StorageError::StaleRead {
                pool: self.pool,
                collection_id: self.collection,
                expected: meta_token,
                actual: live_meta_token,
            });
        }
        let published_meta = FlagMeta {
            generation: new_gen,
            max_short_id: max_short_seen,
        };
        let meta_encoded = encode_meta(published_meta);
        publish_tx.put(
            self.pool,
            self.collection,
            FLAG_META_ID,
            &NodeData::new(Bytes::from(meta_encoded.to_vec())),
        )?;
        if !(live_meta_raw.is_some() && live_meta_token == 0) {
            publish_tx.expect_record_version(
                self.pool,
                self.collection,
                FLAG_META_ID,
                live_meta_token,
            )?;
        }
        publish_tx.commit()?;

        // Keep the old generation available to readers that fetched its meta
        // before publication. Database reads do not offer a cross-collection
        // snapshot or reader lease, so deleting it here could turn a valid
        // old-generation query into an empty/partial result.

        Ok(total_indexed)
    }
}
