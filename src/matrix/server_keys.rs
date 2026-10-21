//! Federation signing keys in the server-info pool.
//!
//! Each origin owns one collection holding one record per key id, so a key is
//! a direct lookup and an origin's keys are one collection scan. There are no
//! tier or reverse indexes: an origin has a handful of keys, and a tier is a
//! field of the key's own record, so changing it rewrites one record. The
//! Notary payload, the self-signed JSON the notary endpoint re-serves, is the
//! one exception: it lives beside the keys as verbatim bytes, since any
//! re-encoding would invalidate its signature.
//!
//! [`update_server_keys`] is the write path. It reads only the records of the
//! keys it was handed (their ids are derived from the key id), applies the
//! MSC4499 binding rules, and commits the changed records, and the Notary
//! payload when one is given, in one transaction guarded by per-record version
//! checks. A concurrent writer that touched any of those records makes the
//! commit fail with a stale read, and the whole read-resolve-write is retried.

use std::collections::BTreeMap;

use bytes::Bytes;

use crate::database::{Database, DatabaseTransaction};
use crate::interop::{
    decode_server_key, derive_id, encode_server_key, notary_record_id, server_keys_collection_id,
    SERVER_INFO_POOL,
};
use crate::packfile::storage::PackfileStorage;
use crate::storage::{NodeData, StorageEngine, StorageError};

/// Version byte of the per-key record encoding.
pub const SERVER_KEY_VERSION: u8 = 1;

/// How many times [`update_server_keys`] re-runs after losing a race before it
/// returns the stale-read error.
const UPDATE_ATTEMPTS: usize = 8;

/// How firmly a key is bound to its origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyTier {
    /// Seen only through a notary so far.
    Provisional,
    /// Observed by a direct fetch from the origin.
    Corroborated,
}

impl KeyTier {
    const fn tag(self) -> u8 {
        match self {
            Self::Provisional => 1,
            Self::Corroborated => 2,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Provisional),
            2 => Some(Self::Corroborated),
            _ => None,
        }
    }
}

/// How a key reached us.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeySighting {
    /// Fetched from the origin itself.
    Direct,
    /// Relayed by a notary.
    Notary,
}

/// One stored signing key of an origin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerKey {
    /// Matrix key identifier, for example `ed25519:abc`.
    pub key_id: Vec<u8>,
    /// Raw public key bytes.
    pub key: Vec<u8>,
    /// Millisecond timestamp the key is valid until.
    pub valid_until_ts: u64,
    /// Who first supplied this binding; kept when the same key is seen again.
    pub source: Vec<u8>,
    /// Millisecond timestamp the key was last seen.
    pub ts_added_ms: u64,
    /// Whether a direct fetch has confirmed the binding.
    pub tier: KeyTier,
}

/// A key as one response presented it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedKey {
    /// Matrix key identifier.
    pub key_id: Vec<u8>,
    /// Raw public key bytes.
    pub key: Vec<u8>,
    /// Millisecond timestamp the key is valid until.
    pub valid_until_ts: u64,
}

/// One batch of keys to merge into an origin's records.
#[derive(Clone, Copy, Debug)]
pub struct KeyUpdate<'a> {
    /// The server the keys belong to.
    pub origin: &'a [u8],
    /// Whether the keys came from the origin or from a notary.
    pub sighting: KeySighting,
    /// Where the keys came from, recorded when a binding is created or replaced.
    pub source: &'a [u8],
    /// The keys the response carried.
    pub observed: &'a [ObservedKey],
    /// The current time, written as `ts_added_ms`.
    pub now_ms: u64,
    /// The response's self-signed payload, stored verbatim when present.
    pub notary: Option<&'a [u8]>,
}

/// The collection holding one origin's key records.
#[must_use]
pub fn origin_keys_collection_id(origin: &[u8]) -> [u8; 16] {
    derive_id(b"server-key-origin", origin)
}

/// The record id of one key within its origin's collection.
#[must_use]
pub fn origin_key_record_id(key_id: &[u8]) -> [u8; 16] {
    derive_id(b"server-key-id", key_id)
}

fn corrupt(what: &str) -> StorageError {
    StorageError::Corrupt(format!("malformed signing-key record: {what}"))
}

fn push_field(out: &mut Vec<u8>, field: &[u8]) -> Result<(), StorageError> {
    let length = u32::try_from(field.len())
        .map_err(|_| StorageError::Corrupt("signing-key field is too large".to_owned()))?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(field);
    Ok(())
}

fn take<'a>(rest: &mut &'a [u8], length: usize, what: &str) -> Result<&'a [u8], StorageError> {
    let (head, tail) = rest.split_at_checked(length).ok_or_else(|| corrupt(what))?;
    *rest = tail;
    Ok(head)
}

fn take_u64(rest: &mut &[u8], what: &str) -> Result<u64, StorageError> {
    let bytes = take(rest, 8, what)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| corrupt(what))?,
    ))
}

fn take_field<'a>(rest: &mut &'a [u8], what: &str) -> Result<&'a [u8], StorageError> {
    let bytes = take(rest, 4, what)?;
    let length = u32::from_le_bytes(bytes.try_into().map_err(|_| corrupt(what))?);
    take(
        rest,
        usize::try_from(length).map_err(|_| corrupt(what))?,
        what,
    )
}

/// Encode one key record: version, tier, `valid_until_ts`, `ts_added_ms`, then
/// the length-prefixed origin, key id, key bytes and source.
///
/// The origin rides along so a read can reject a record that landed in the
/// wrong origin's collection, which a 128-bit collection id alone cannot rule
/// out.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if a field does not fit its length prefix.
pub fn encode_server_key_record(origin: &[u8], key: &ServerKey) -> Result<Vec<u8>, StorageError> {
    let mut out = vec![SERVER_KEY_VERSION, key.tier.tag()];
    out.extend_from_slice(&key.valid_until_ts.to_le_bytes());
    out.extend_from_slice(&key.ts_added_ms.to_le_bytes());
    for field in [origin, &key.key_id, &key.key, &key.source] {
        push_field(&mut out, field)?;
    }
    Ok(out)
}

/// Decode a key record into its origin and key.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record, an unknown
/// version or tier, or trailing bytes.
pub fn decode_server_key_record(bytes: &[u8]) -> Result<(Vec<u8>, ServerKey), StorageError> {
    let mut rest = bytes;
    if take(&mut rest, 1, "version")? != [SERVER_KEY_VERSION] {
        return Err(StorageError::Corrupt(
            "unknown signing-key record version".to_owned(),
        ));
    }
    let tier = KeyTier::from_tag(take(&mut rest, 1, "tier")?[0]).ok_or_else(|| corrupt("tier"))?;
    let valid_until_ts = take_u64(&mut rest, "valid_until_ts")?;
    let ts_added_ms = take_u64(&mut rest, "ts_added_ms")?;
    let origin = take_field(&mut rest, "origin")?.to_vec();
    let key_id = take_field(&mut rest, "key id")?.to_vec();
    let key = take_field(&mut rest, "key")?.to_vec();
    let source = take_field(&mut rest, "source")?.to_vec();
    if !rest.is_empty() {
        return Err(corrupt("trailing bytes"));
    }
    Ok((
        origin,
        ServerKey {
            key_id,
            key,
            valid_until_ts,
            source,
            ts_added_ms,
            tier,
        },
    ))
}

/// Decode `bytes` and check it is the record of `origin`'s `key_id`.
fn decode_for(
    origin: &[u8],
    expected_key_id: Option<&[u8]>,
    bytes: &[u8],
) -> Result<ServerKey, StorageError> {
    let (stored_origin, key) = decode_server_key_record(bytes)?;
    if stored_origin != origin || expected_key_id.is_some_and(|id| id != key.key_id) {
        return Err(StorageError::Collision(
            "signing-key record does not belong to its slot".to_owned(),
        ));
    }
    Ok(key)
}

/// Read one key of `origin` from `store`.
///
/// # Errors
/// Returns an error if the pool cannot be read or the record is corrupt.
pub fn read_server_key(
    store: &PackfileStorage,
    origin: &[u8],
    key_id: &[u8],
) -> Result<Option<ServerKey>, StorageError> {
    let found = store.get_many(
        &origin_keys_collection_id(origin),
        &[origin_key_record_id(key_id)],
    )?;
    found
        .into_iter()
        .next()
        .flatten()
        .map(|data| decode_for(origin, Some(key_id), &data.bytes))
        .transpose()
}

/// Read every key of `origin` from `store`, ordered by key id. Filter on
/// [`ServerKey::tier`] for one tier.
///
/// Reads the live pool, so a read-only worker holding a store with the
/// committed-journal overlay installed sees keys written since it opened.
///
/// # Errors
/// Returns an error if the pool cannot be scanned or a record is corrupt.
pub fn read_server_keys(
    store: &PackfileStorage,
    origin: &[u8],
) -> Result<Vec<ServerKey>, StorageError> {
    let mut keys = Vec::new();
    for item in store.scan_collection(&origin_keys_collection_id(origin))? {
        let (node_id, data) = item?;
        let key = decode_for(origin, None, &data.bytes)?;
        if origin_key_record_id(&key.key_id) != node_id {
            return Err(StorageError::Collision(
                "signing-key record does not belong to its slot".to_owned(),
            ));
        }
        keys.push(key);
    }
    keys.sort_by(|a, b| a.key_id.cmp(&b.key_id));
    Ok(keys)
}

/// Read the verbatim Notary payload stored for `origin`.
///
/// # Errors
/// Returns an error if the pool cannot be read or the record is corrupt.
pub fn read_notary_payload(
    store: &PackfileStorage,
    origin: &[u8],
) -> Result<Option<Vec<u8>>, StorageError> {
    let found = store.get_many(&server_keys_collection_id(), &[notary_record_id(origin)])?;
    let Some(data) = found.into_iter().next().flatten() else {
        return Ok(None);
    };
    let (stored_origin, payload) = decode_server_key(&data.bytes)?;
    if stored_origin != origin {
        return Err(StorageError::Collision(
            "notary record does not belong to its origin".to_owned(),
        ));
    }
    Ok(Some(payload.to_vec()))
}

/// Stage `key` as a record of `origin`, replacing any record of the same key
/// id. No merge rules apply; use [`update_server_keys`] for those.
///
/// # Errors
/// Returns an error if the record cannot be encoded or the transaction cannot
/// take another write.
pub fn put_server_key(
    tx: &DatabaseTransaction<'_>,
    origin: &[u8],
    key: &ServerKey,
) -> Result<(), StorageError> {
    let bytes = encode_server_key_record(origin, key)?;
    tx.put(
        SERVER_INFO_POOL,
        origin_keys_collection_id(origin),
        origin_key_record_id(&key.key_id),
        &NodeData::new(Bytes::from(bytes)),
    )?;
    Ok(())
}

/// Stage `payload` as `origin`'s Notary record, replacing any earlier one. The
/// bytes are stored exactly as given.
///
/// # Errors
/// Returns an error if the record cannot be encoded or the transaction cannot
/// take another write.
pub fn put_notary_payload(
    tx: &DatabaseTransaction<'_>,
    origin: &[u8],
    payload: &[u8],
) -> Result<(), StorageError> {
    let bytes = encode_server_key(origin, payload)?;
    tx.put(
        SERVER_INFO_POOL,
        server_keys_collection_id(),
        notary_record_id(origin),
        &NodeData::new(Bytes::from(bytes)),
    )?;
    Ok(())
}

/// Apply the MSC4499 binding rules to one observed key.
///
/// * A key id seen for the first time is bound: the first one seen wins. A
///   direct fetch binds it corroborated, a notary provisionally.
/// * The same key body again refreshes only the validity time and
///   `ts_added_ms`; the original source stays. The validity time is replaced
///   exactly, as Synapse's upsert does, not only moved forward. A direct fetch of a provisional
///   key also corroborates it.
/// * A different body replaces the stored one only when a direct fetch meets a
///   provisional binding; the new binding is corroborated and takes the new
///   source.
/// * Otherwise the stored key stays.
fn bind(stored: Option<&ServerKey>, observed: &ObservedKey, update: &KeyUpdate<'_>) -> ServerKey {
    let direct = update.sighting == KeySighting::Direct;
    let fresh = |tier| ServerKey {
        key_id: observed.key_id.clone(),
        key: observed.key.clone(),
        valid_until_ts: observed.valid_until_ts,
        source: update.source.to_vec(),
        ts_added_ms: update.now_ms,
        tier,
    };
    match stored {
        None => fresh(if direct {
            KeyTier::Corroborated
        } else {
            KeyTier::Provisional
        }),
        Some(stored) if stored.key == observed.key => {
            let mut key = stored.clone();
            key.valid_until_ts = observed.valid_until_ts;
            key.ts_added_ms = update.now_ms;
            if direct {
                key.tier = KeyTier::Corroborated;
            }
            key
        }
        Some(stored) if direct && stored.tier == KeyTier::Provisional => {
            fresh(KeyTier::Corroborated)
        }
        Some(stored) => stored.clone(),
    }
}

/// Merge `update` into its origin's records in one transaction and return the
/// authoritative key for each distinct key id it carried, in first-seen order.
///
/// Reads the stored records of the carried key ids, applies the binding rules
/// in order (a key id repeated in `observed` sees the earlier result), and
/// commits every changed record and the Notary payload together. Each stored
/// record is version-checked, so an overlapping writer fails the commit and the
/// whole read-resolve-write is re-run on fresh data.
///
/// # Errors
/// Returns an error if a stored record is corrupt, if the pool cannot be
/// read or written, or the stale-read error if every attempt lost a race.
pub fn update_server_keys(
    db: &Database,
    update: &KeyUpdate<'_>,
) -> Result<Vec<ServerKey>, StorageError> {
    validate(update)?;
    if update.observed.is_empty() && update.notary.is_none() {
        return Ok(Vec::new());
    }
    let collection = origin_keys_collection_id(update.origin);
    let mut attempt = 1;
    loop {
        match try_update(db, update, collection) {
            Err(error) if error.is_stale_read() && attempt < UPDATE_ATTEMPTS => {
                attempt = attempt.saturating_add(1);
            }
            result => return result,
        }
    }
}

/// Reject input that could not be stored faithfully, before anything is read
/// or staged: an empty origin, key id or key body, or two distinct key ids
/// that derive the same record id.
fn validate(update: &KeyUpdate<'_>) -> Result<(), StorageError> {
    let invalid = |what: &str| StorageError::Corrupt(format!("invalid server key update: {what}"));
    if update.origin.is_empty() {
        return Err(invalid("empty origin"));
    }
    let mut seen: BTreeMap<[u8; 16], &[u8]> = BTreeMap::new();
    for observed in update.observed {
        if observed.key_id.is_empty() {
            return Err(invalid("empty key id"));
        }
        if observed.key.is_empty() {
            return Err(invalid("empty key bytes"));
        }
        let id = origin_key_record_id(&observed.key_id);
        if *seen.entry(id).or_insert(&observed.key_id) != observed.key_id.as_slice() {
            return Err(invalid("key ids share a record id"));
        }
    }
    Ok(())
}

fn try_update(
    db: &Database,
    update: &KeyUpdate<'_>,
    collection: [u8; 16],
) -> Result<Vec<ServerKey>, StorageError> {
    let mut ids = Vec::new();
    for observed in update.observed {
        let id = origin_key_record_id(&observed.key_id);
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let tx = db.begin_transaction();
    let (records, versions) = tx.get_with_record_versions(SERVER_INFO_POOL, &collection, &ids)?;
    let mut current: BTreeMap<[u8; 16], ServerKey> = BTreeMap::new();
    for ((id, record), version) in ids.iter().zip(&records).zip(&versions) {
        tx.expect_record_version(SERVER_INFO_POOL, collection, *id, *version)?;
        if let Some(data) = record {
            current.insert(*id, decode_for(update.origin, None, &data.bytes)?);
        }
    }
    let stored = current.clone();
    let mut order = Vec::with_capacity(ids.len());
    for observed in update.observed {
        let id = origin_key_record_id(&observed.key_id);
        let bound = bind(current.get(&id), observed, update);
        current.insert(id, bound);
        if !order.contains(&id) {
            order.push(id);
        }
    }
    for id in &order {
        let key = &current[id];
        if stored.get(id) != Some(key) {
            put_server_key(&tx, update.origin, key)?;
        }
    }
    if let Some(payload) = update.notary {
        put_notary_payload(&tx, update.origin, payload)?;
    }
    tx.commit()?;
    Ok(order.iter().map(|id| current[id].clone()).collect())
}

#[cfg(test)]
#[path = "test_server_keys.rs"]
mod tests;
