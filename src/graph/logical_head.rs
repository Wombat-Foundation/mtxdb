//! Stable logical-ID to current-record pointers.
//!
//! A logical head is the mutable query-side view of an append-only record
//! store. The physical target stays immutable; replacing a head appends a new
//! frame under the same logical ID.
//!
//! Concurrency control is **not** implemented here. A head is an ordinary
//! record, so it uses the engine's per-record compare-and-swap: read the head
//! with [`LogicalHead::read`] (which returns the record's write-LSN token),
//! stage the replacement with [`LogicalHead::stage_replace`] (which records the
//! token as a precondition), and commit the [`DatabaseTransaction`]. A stale
//! token makes the commit fail with `StorageError::StaleRead`. Because the
//! transaction may also stage other records, a head swap can be published
//! atomically with the structures it points into (for example a current-state
//! map plus its change stream).

use bytes::Bytes;

use crate::database::DatabaseTransaction;
use crate::layout::ShardType;
use crate::storage::{NodeData, NodeId, StorageError};

/// Wire magic for a logical-head pointer value (`LHP1`).
pub const LOGICAL_HEAD_MAGIC: [u8; 4] = *b"LHP1";
/// Current logical-head wire version.
pub const LOGICAL_HEAD_VERSION: u8 = 1;

const HEADER_LEN: usize = 4 + 1 + 16 + 4;

/// The current value of one stable logical record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalHeadValue {
    /// Immutable physical record currently selected by this logical ID.
    pub target: NodeId,
    /// Optional opaque metadata owned by the caller. This is not a
    /// concurrency token; conflicts are detected by the engine's write-LSN.
    pub metadata: Bytes,
}

impl LogicalHeadValue {
    /// Construct a logical-head value.
    #[must_use]
    pub fn new(target: NodeId, metadata: impl Into<Bytes>) -> Self {
        Self {
            target,
            metadata: metadata.into(),
        }
    }
}

/// Encode a logical-head value for storage.
///
/// # Errors
/// Returns an error if the metadata exceeds `u32::MAX` bytes.
pub fn encode_logical_head(value: &LogicalHeadValue) -> Result<Bytes, StorageError> {
    let metadata_len = u32::try_from(value.metadata.len())
        .map_err(|_| StorageError::Internal("logical-head metadata too large".to_owned()))?;
    let mut encoded = Vec::with_capacity(HEADER_LEN.saturating_add(value.metadata.len()));
    encoded.extend_from_slice(&LOGICAL_HEAD_MAGIC);
    encoded.push(LOGICAL_HEAD_VERSION);
    encoded.extend_from_slice(&value.target);
    encoded.extend_from_slice(&metadata_len.to_be_bytes());
    encoded.extend_from_slice(&value.metadata);
    Ok(Bytes::from(encoded))
}

/// Decode a logical-head value from storage.
///
/// # Errors
/// Returns `StorageError::Corrupt` for a bad header, length or version.
pub fn decode_logical_head(bytes: &[u8]) -> Result<LogicalHeadValue, StorageError> {
    if bytes.len() < HEADER_LEN
        || bytes[..4] != LOGICAL_HEAD_MAGIC
        || bytes[4] != LOGICAL_HEAD_VERSION
    {
        return Err(StorageError::Corrupt("logical-head header".to_owned()));
    }
    let mut target = [0u8; 16];
    target.copy_from_slice(&bytes[5..21]);
    let metadata_len = u32::from_be_bytes(
        bytes[21..25]
            .try_into()
            .map_err(|_| StorageError::Corrupt("logical-head metadata length".to_owned()))?,
    ) as usize;
    let end = HEADER_LEN
        .checked_add(metadata_len)
        .ok_or_else(|| StorageError::Corrupt("logical-head length overflow".to_owned()))?;
    if bytes.len() != end {
        return Err(StorageError::Corrupt("logical-head length".to_owned()));
    }
    Ok(LogicalHeadValue::new(
        target,
        Bytes::copy_from_slice(&bytes[HEADER_LEN..]),
    ))
}

/// A head read together with the engine token that guards a later replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalHeadRead {
    /// The current value, or `None` if the logical ID has no head.
    pub value: Option<LogicalHeadValue>,
    /// The record's write LSN (or delete LSN when absent). Pass it to
    /// [`LogicalHead::stage_replace`].
    pub token: u64,
}

/// A logical-head collection inside one pool.
#[derive(Debug, Clone, Copy)]
pub struct LogicalHead {
    pool: ShardType,
    collection_id: [u8; 16],
}

impl LogicalHead {
    /// Address the logical-head collection `collection_id` in `pool`.
    #[must_use]
    pub fn new(pool: ShardType, collection_id: [u8; 16]) -> Self {
        Self {
            pool,
            collection_id,
        }
    }

    /// Read the current head and its CAS token through `txn`.
    ///
    /// Read before staging writes: a record this transaction has already
    /// staged reports token `0`.
    ///
    /// # Errors
    /// Returns an error if the pool cannot be read or the stored head is corrupt.
    pub fn read(
        &self,
        txn: &DatabaseTransaction<'_>,
        logical_id: &NodeId,
    ) -> Result<LogicalHeadRead, StorageError> {
        let (records, tokens) =
            txn.get_with_record_versions(self.pool, &self.collection_id, &[*logical_id])?;
        let value = records
            .into_iter()
            .next()
            .flatten()
            .map(|record| decode_logical_head(&record.bytes))
            .transpose()?;
        Ok(LogicalHeadRead {
            value,
            token: tokens.first().copied().unwrap_or(0),
        })
    }

    /// Stage a replacement that commits only if the head is unchanged since
    /// `token` was read. The caller commits the transaction; a conflict
    /// surfaces there as `StorageError::StaleRead`.
    ///
    /// # Errors
    /// Returns an error if the value cannot be encoded or the transaction is
    /// no longer active.
    pub fn stage_replace(
        &self,
        txn: &DatabaseTransaction<'_>,
        logical_id: &NodeId,
        token: u64,
        value: &LogicalHeadValue,
    ) -> Result<(), StorageError> {
        let encoded = encode_logical_head(value)?;
        txn.expect_record_version(self.pool, self.collection_id, *logical_id, token)?;
        txn.put(
            self.pool,
            self.collection_id,
            *logical_id,
            &NodeData::new(encoded),
        )?;
        Ok(())
    }
}
