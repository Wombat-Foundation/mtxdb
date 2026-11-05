//! Sparse CSR chunk: the on-disk row layout for forward edge families.
//!
//! One chunk holds the edge rows of every owner whose short id falls in the
//! same [`CHUNK_SHORTS`]-wide window. Rows are sparse (an owner appears only
//! once its edges were recorded; an empty row means "recorded, no edges"), and
//! laid out CSR-style: sorted owners, row offsets, flat targets and an optional
//! aligned kind column for typed families.
//!
//! Wire encoding: `[flags:u8][rows:varint]` then per row
//! `[owner gap:varint][count:varint][first target: zigzag(target - owner)]`
//! `[later targets: gap varints]` and, for typed families, a `u16` kind per
//! target. Targets within a row are sorted by `(target, kind)`; equal targets
//! with different kinds therefore encode a zero gap.

use std::fmt;

/// Owners per chunk. The chunk index of a short id is `id / CHUNK_SHORTS`.
pub const CHUNK_SHORTS: u32 = 1024;

/// A malformed chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    /// Input ended early or had trailing bytes.
    Truncated,
    /// A varint, id or count was out of range.
    Malformed,
    /// The chunk was written for a different typing than requested.
    TypingMismatch,
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("edge chunk truncated"),
            Self::Malformed => f.write_str("edge chunk malformed"),
            Self::TypingMismatch => f.write_str("edge chunk stored with different typing"),
        }
    }
}

impl std::error::Error for ChunkError {}

/// The chunk index holding `owner`.
#[must_use]
pub const fn chunk_of(owner: u32) -> u32 {
    owner / CHUNK_SHORTS
}

/// One row's edges: `(target, kind)` pairs, sorted.
pub type Row = Vec<(u32, u16)>;

/// Decoded sparse CSR chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EdgeChunk {
    typed: bool,
    owners: Vec<u32>,
    offsets: Vec<u32>,
    targets: Vec<u32>,
    kinds: Vec<u16>,
}

impl EdgeChunk {
    /// An empty chunk for a family with the given typing.
    #[must_use]
    pub fn new(typed: bool) -> Self {
        Self {
            typed,
            owners: Vec::new(),
            offsets: vec![0],
            targets: Vec::new(),
            kinds: Vec::new(),
        }
    }

    /// Whether rows carry a kind per target.
    #[must_use]
    pub const fn is_typed(&self) -> bool {
        self.typed
    }

    /// Number of owner rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.owners.len()
    }

    /// The row of `owner`, if recorded.
    #[must_use]
    pub fn get(&self, owner: u32) -> Option<Row> {
        let at = self.owners.binary_search(&owner).ok()?;
        let (start, end) = (self.offsets[at] as usize, self.offsets[at + 1] as usize);
        Some(
            (start..end)
                .map(|i| (self.targets[i], if self.typed { self.kinds[i] } else { 0 }))
                .collect(),
        )
    }

    /// Every `(owner, row)` in owner order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, Row)> + '_ {
        self.owners
            .iter()
            .map(|&owner| (owner, self.get(owner).unwrap_or_default()))
    }

    /// Insert `owner`'s row. The caller guarantees `owner` is not present and
    /// `row` is sorted.
    ///
    /// # Panics
    /// Panics if the edge count exceeds `u32::MAX`.
    pub fn insert(&mut self, owner: u32, row: &[(u32, u16)]) {
        let at = match self.owners.binary_search(&owner) {
            Ok(_) => return,
            Err(at) => at,
        };
        let split = self.offsets[at] as usize;
        let n = u32::try_from(row.len()).expect("row too long");
        self.owners.insert(at, owner);
        self.offsets.insert(at + 1, self.offsets[at]);
        for off in &mut self.offsets[at + 1..] {
            *off = off.checked_add(n).expect("edge count exceeds u32::MAX");
        }
        self.targets
            .splice(split..split, row.iter().map(|&(target, _)| target));
        if self.typed {
            self.kinds
                .splice(split..split, row.iter().map(|&(_, kind)| kind));
        }
    }

    /// Encode to the wire format.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![u8::from(self.typed)];
        put_varint(&mut out, self.owners.len() as u64);
        let mut prev_owner = 0u32;
        for (at, &owner) in self.owners.iter().enumerate() {
            put_varint(&mut out, u64::from(owner - prev_owner));
            prev_owner = owner;
            let (start, end) = (self.offsets[at] as usize, self.offsets[at + 1] as usize);
            put_varint(&mut out, (end - start) as u64);
            let mut prev = 0u32;
            for i in start..end {
                let target = self.targets[i];
                if i == start {
                    put_varint(&mut out, zigzag(i64::from(target) - i64::from(owner)));
                } else {
                    put_varint(&mut out, u64::from(target - prev));
                }
                prev = target;
                if self.typed {
                    out.extend_from_slice(&self.kinds[i].to_be_bytes());
                }
            }
        }
        out
    }

    /// Decode a chunk, requiring it to match `typed`.
    ///
    /// # Errors
    /// [`ChunkError`] on truncation, overflow, or a typing mismatch.
    pub fn decode(bytes: &[u8], typed: bool) -> Result<Self, ChunkError> {
        let (&flags, mut rest) = bytes.split_first().ok_or(ChunkError::Truncated)?;
        if flags > 1 {
            return Err(ChunkError::Malformed);
        }
        if (flags == 1) != typed {
            return Err(ChunkError::TypingMismatch);
        }
        let rows = take_varint(&mut rest)?;
        let mut chunk = Self::new(typed);
        let mut prev_owner = 0u32;
        for row_index in 0..rows {
            let gap = u32::try_from(take_varint(&mut rest)?).map_err(|_| ChunkError::Malformed)?;
            let owner = prev_owner.checked_add(gap).ok_or(ChunkError::Malformed)?;
            if row_index > 0 && gap == 0 {
                return Err(ChunkError::Malformed);
            }
            prev_owner = owner;
            let count = take_varint(&mut rest)?;
            let mut prev = 0u32;
            for i in 0..count {
                let target = if i == 0 {
                    let signed = i64::from(owner) + unzigzag(take_varint(&mut rest)?);
                    u32::try_from(signed).map_err(|_| ChunkError::Malformed)?
                } else {
                    let gap = u32::try_from(take_varint(&mut rest)?)
                        .map_err(|_| ChunkError::Malformed)?;
                    prev.checked_add(gap).ok_or(ChunkError::Malformed)?
                };
                prev = target;
                chunk.targets.push(target);
                if typed {
                    if rest.len() < 2 {
                        return Err(ChunkError::Truncated);
                    }
                    chunk.kinds.push(u16::from_be_bytes([rest[0], rest[1]]));
                    rest = &rest[2..];
                }
            }
            chunk.owners.push(owner);
            chunk
                .offsets
                .push(u32::try_from(chunk.targets.len()).map_err(|_| ChunkError::Malformed)?);
        }
        if rest.is_empty() {
            Ok(chunk)
        } else {
            Err(ChunkError::Malformed)
        }
    }
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_varint(rest: &mut &[u8]) -> Result<u64, ChunkError> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let (&byte, tail) = rest.split_first().ok_or(ChunkError::Truncated)?;
        *rest = tail;
        if shift == 63 && byte > 1 {
            return Err(ChunkError::Malformed);
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(ChunkError::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip_with_parents_below_and_above() {
        for typed in [false, true] {
            let kind = |k: u16| if typed { k } else { 0 };
            let mut chunk = EdgeChunk::new(typed);
            chunk.insert(1030, &[(5, kind(1)), (1029, kind(2)), (1029, kind(3))]);
            chunk.insert(1024, &[]);
            chunk.insert(1500, &[(1600, kind(0)), (u32::MAX, kind(0))]);
            let decoded = EdgeChunk::decode(&chunk.encode(), typed).unwrap();
            assert_eq!(decoded, chunk);
            assert_eq!(decoded.get(1024), Some(vec![]));
            assert_eq!(
                decoded.get(1030),
                Some(vec![(5, kind(1)), (1029, kind(2)), (1029, kind(3))])
            );
            assert_eq!(decoded.get(1031), None);
        }
    }

    #[test]
    fn typing_mismatch_and_garbage_are_rejected() {
        let mut chunk = EdgeChunk::new(true);
        chunk.insert(3, &[(1, 7)]);
        let bytes = chunk.encode();
        assert_eq!(
            EdgeChunk::decode(&bytes, false),
            Err(ChunkError::TypingMismatch)
        );
        for cut in 0..bytes.len() {
            assert!(EdgeChunk::decode(&bytes[..cut], true).is_err(), "cut {cut}");
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(EdgeChunk::decode(&extra, true).is_err());
    }
}
