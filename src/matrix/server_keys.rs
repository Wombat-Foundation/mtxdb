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
//! [`update_server_keys`](crate::server_keys::update_server_keys) is the write path. It reads the records of the keys
//! it was handed (their ids are derived from the key id), applies the MSC4499
//! binding rules, and commits the changed records, and the Notary payload when
//! one is given, in one transaction guarded by per-record version checks. A
//! snapshot, or a key id new to the origin, also reads the origin's other
//! records, since retiring omitted keys, the digest-cap buckets and the
//! retired-key ceiling are judged across them. A concurrent writer that touched
//! any record read makes the commit fail with a stale read, and the whole
//! read-resolve-write is retried.
//!
//! A key's record is also its digest binding: evicting a retired key clears
//! its bytes and keeps the record, so a later body for the key id still
//! collides.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use sha2::{Digest, Sha256};

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
const UPDATE_ATTEMPTS: usize = 32;

/// How firmly a key is bound to its origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyTier {
    /// Seen only through a notary so far.
    Provisional,
    /// Observed by a direct fetch from the origin. First Seen Wins from here.
    Permanent,
}

impl KeyTier {
    const fn tag(self) -> u8 {
        match self {
            Self::Provisional => 1,
            Self::Permanent => 2,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Provisional),
            2 => Some(Self::Permanent),
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
    /// Whether a direct fetch has confirmed the binding. The tier also names
    /// the digest-cap bucket the key counts against: permanent is the direct
    /// bucket, provisional the notary one.
    pub tier: KeyTier,
    /// SHA-256 of the key body. It is the digest binding: it outlives the key
    /// bytes when they are evicted, so a later body for the key id is still a
    /// collision.
    pub digest: [u8; 32],
    /// When this receiver first bound the key id.
    pub first_seen_ms: u64,
    /// The last time the key was seen active; `0` if it never was.
    pub last_active_ms: u64,
    /// Whether this receiver itself observed the key active. It changes the
    /// retired-key eviction order and nothing else.
    pub corroborated: bool,
    /// The key vanished from the origin's responses without an `expired_ts`, so
    /// it counts as retired at `last_active_ms`.
    pub inferred_retired: bool,
    /// When the key was retired: the first `expired_ts` observed for it, kept
    /// for good. `None` while no sane `expired_ts` has been seen.
    pub expired_ts: Option<u64>,
    /// The last millisecond a direct fetch may still override a provisional
    /// binding, fixed when the binding was made; `0` once it is permanent.
    pub promote_by_ms: u64,
}

impl ServerKey {
    /// Whether the key bytes were evicted, leaving only the digest binding.
    #[must_use]
    pub fn is_evicted(&self) -> bool {
        self.key.is_empty()
    }

    /// The effective retirement timestamp: the first `expired_ts`, else the
    /// last time an inferred-retired key was seen active, else `None` for a
    /// key still current.
    #[must_use]
    pub fn retirement_ts(&self) -> Option<u64> {
        self.expired_ts
            .or_else(|| self.inferred_retired.then_some(self.last_active_ms))
    }
}

/// The SHA-256 digest of a key body, as a [`ServerKey::digest`].
#[must_use]
pub fn key_digest(key: &[u8]) -> [u8; 32] {
    Sha256::digest(key).into()
}

/// The ceilings of MSC4499.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyLimits {
    /// Most active keys one response may carry; more rejects the response.
    pub active_per_response: usize,
    /// Most retired keys one response may carry; more rejects the response.
    pub retired_per_response: usize,
    /// Most retired keys, with key bytes, kept per origin. The rest are
    /// evicted to their digest binding.
    pub retired_retained: usize,
    /// Most digest bindings per origin and bucket; a new key id past it is
    /// refused, never made room for.
    pub digest_cap: usize,
}

impl Default for KeyLimits {
    fn default() -> Self {
        Self {
            active_per_response: 50,
            retired_per_response: 3_000,
            retired_retained: 3_000,
            digest_cap: 30_000,
        }
    }
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
    /// The `expired_ts` of an `old_verify_keys` entry, if it came from one.
    pub expired_ts: Option<u64>,
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
    /// The response's self-signed payload, stored verbatim when present and
    /// when no key of the batch was rejected.
    pub notary: Option<&'a [u8]>,
    /// How long a new provisional binding stays open to a direct override.
    pub promotion_window_ms: u64,
    /// Whether `observed` is the origin's whole key set. A direct snapshot
    /// retires every stored active key it leaves out.
    pub snapshot: bool,
    /// The ceilings to enforce.
    pub limits: KeyLimits,
}

/// How far ahead of `now_ms` an `expired_ts` may be before it is malformed.
pub const EXPIRED_TS_SKEW_MS: u64 = 5 * 60 * 1000;

/// What went wrong, or notable, for one key of a batch. The caller logs these.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyIssueKind {
    /// A different body for a bound key id arrived and was rejected.
    Collision,
    /// A direct fetch displaced a provisional binding with a different body.
    Displaced,
    /// A different `expired_ts` arrived for a key that already has one.
    ExpiredTsConflict,
    /// The entry's `expired_ts` was too far in the future; the entry was
    /// dropped and nothing about it stored.
    FutureExpiredTs,
    /// A new key id was refused because its bucket is at the digest cap.
    DigestCapReached,
    /// A retired key fell below the retention floor; its bytes were evicted.
    Evicted,
    /// Like [`Self::Evicted`], but the key was corroborated: the flood signal.
    EvictedCorroborated,
}

/// One [`KeyIssueKind`] and the key id it concerns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyIssue {
    /// The key id.
    pub key_id: Vec<u8>,
    /// What happened to it.
    pub kind: KeyIssueKind,
}

/// The result of [`update_server_keys`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyMerge {
    /// The authoritative key of each distinct key id kept, in first-seen order.
    pub keys: Vec<ServerKey>,
    /// Collisions and rejected values, for the caller to log.
    pub issues: Vec<KeyIssue>,
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

/// Encode one key record: version, tier, a flags byte (`expired_ts` present,
/// corroborated, inferred retired), the numbers `valid_until_ts`,
/// `ts_added_ms`, `promote_by_ms`, `expired_ts` (zero when absent),
/// `first_seen_ms` and `last_active_ms`, the key digest, then the
/// length-prefixed origin, key id, key bytes (empty once evicted) and source.
///
/// The origin rides along so a read can reject a record that landed in the
/// wrong origin's collection, which a 128-bit collection id alone cannot rule
/// out.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if a field does not fit its length prefix.
pub fn encode_server_key_record(origin: &[u8], key: &ServerKey) -> Result<Vec<u8>, StorageError> {
    let flags = u8::from(key.expired_ts.is_some())
        | u8::from(key.corroborated) << 1
        | u8::from(key.inferred_retired) << 2;
    let mut out = vec![SERVER_KEY_VERSION, key.tier.tag(), flags];
    for number in [
        key.valid_until_ts,
        key.ts_added_ms,
        key.promote_by_ms,
        key.expired_ts.unwrap_or(0),
        key.first_seen_ms,
        key.last_active_ms,
    ] {
        out.extend_from_slice(&number.to_le_bytes());
    }
    out.extend_from_slice(&key.digest);
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
    let flags = take(&mut rest, 1, "flags")?[0];
    if flags & !0b111 != 0 {
        return Err(corrupt("flags"));
    }
    let has_expired = flags & 1 != 0;
    let valid_until_ts = take_u64(&mut rest, "valid_until_ts")?;
    let ts_added_ms = take_u64(&mut rest, "ts_added_ms")?;
    let promote_by_ms = take_u64(&mut rest, "promote_by_ms")?;
    let expired = take_u64(&mut rest, "expired_ts")?;
    if !has_expired && expired != 0 {
        return Err(corrupt("expired_ts without its flag"));
    }
    let first_seen_ms = take_u64(&mut rest, "first_seen_ms")?;
    let last_active_ms = take_u64(&mut rest, "last_active_ms")?;
    let digest: [u8; 32] = take(&mut rest, 32, "digest")?
        .try_into()
        .map_err(|_| corrupt("digest"))?;
    let origin = take_field(&mut rest, "origin")?.to_vec();
    let key_id = take_field(&mut rest, "key id")?.to_vec();
    let key = take_field(&mut rest, "key")?.to_vec();
    let source = take_field(&mut rest, "source")?.to_vec();
    if !rest.is_empty() {
        return Err(corrupt("trailing bytes"));
    }
    if !key.is_empty() && key_digest(&key) != digest {
        return Err(corrupt("digest does not match the key"));
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
            expired_ts: has_expired.then_some(expired),
            promote_by_ms,
            digest,
            first_seen_ms,
            last_active_ms,
            corroborated: flags & 0b10 != 0,
            inferred_retired: flags & 0b100 != 0,
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

/// Whether a direct fetch may still displace the provisional binding `stored`:
/// its own promotion window is open, and it has neither expired nor retired.
fn overridable(stored: &ServerKey, now_ms: u64) -> bool {
    stored.tier == KeyTier::Provisional
        && now_ms <= stored.promote_by_ms
        && stored.valid_until_ts >= now_ms
        && stored
            .retirement_ts()
            .is_none_or(|retired| retired > now_ms)
}

/// Apply the MSC4499 binding rules to one observed key, returning the binding
/// that stands and anything notable that happened.
///
/// * A key id seen for the first time is bound: the first one seen wins. A
///   direct fetch binds it permanent, a notary provisionally with its own
///   promotion deadline.
/// * The same key body (compared by digest, so an evicted key matches too)
///   refreshes only the validity time and `ts_added_ms`; the original source
///   stays. The validity time is replaced exactly, as Synapse's upsert does,
///   not only moved forward. A direct fetch of a provisional key also makes it
///   permanent, moving it to the direct digest bucket. An evicted key whose
///   body returns gets its bytes back.
/// * Seeing a key active records the observation: it is corroborated, its
///   last-active time moves, and an inferred retirement is undone.
/// * `expired_ts` is first-assignment-wins: the first one seen is kept, and a
///   different later one is reported and ignored.
/// * A different body replaces the stored one only when a direct fetch meets
///   an [`overridable`] provisional binding; the new binding is permanent and
///   takes the new source. Any other different body is a collision.
fn bind(
    stored: Option<&ServerKey>,
    observed: &ObservedKey,
    update: &KeyUpdate<'_>,
) -> (ServerKey, Option<KeyIssueKind>) {
    let direct = update.sighting == KeySighting::Direct;
    let active = observed.expired_ts.is_none();
    let digest = key_digest(&observed.key);
    let fresh = |tier| ServerKey {
        key_id: observed.key_id.clone(),
        key: observed.key.clone(),
        valid_until_ts: observed.valid_until_ts,
        source: update.source.to_vec(),
        ts_added_ms: update.now_ms,
        tier,
        digest,
        first_seen_ms: update.now_ms,
        last_active_ms: if active { update.now_ms } else { 0 },
        corroborated: active,
        inferred_retired: false,
        expired_ts: observed.expired_ts,
        promote_by_ms: match tier {
            KeyTier::Provisional => update.now_ms.saturating_add(update.promotion_window_ms),
            KeyTier::Permanent => 0,
        },
    };
    match stored {
        None => (
            fresh(if direct {
                KeyTier::Permanent
            } else {
                KeyTier::Provisional
            }),
            None,
        ),
        Some(stored) if stored.digest == digest => {
            let mut key = stored.clone();
            key.key.clone_from(&observed.key);
            key.valid_until_ts = observed.valid_until_ts;
            key.ts_added_ms = update.now_ms;
            if active {
                key.corroborated = true;
                key.last_active_ms = update.now_ms;
                key.inferred_retired = false;
            }
            let mut issue = None;
            match (stored.expired_ts, observed.expired_ts) {
                (None, Some(expired)) => key.expired_ts = Some(expired),
                (Some(first), Some(later)) if first != later => {
                    issue = Some(KeyIssueKind::ExpiredTsConflict);
                }
                _ => {}
            }
            if direct {
                key.tier = KeyTier::Permanent;
                key.promote_by_ms = 0;
            }
            (key, issue)
        }
        Some(stored) if direct && overridable(stored, update.now_ms) => {
            (fresh(KeyTier::Permanent), Some(KeyIssueKind::Displaced))
        }
        Some(stored) => (stored.clone(), Some(KeyIssueKind::Collision)),
    }
}

/// Reject input that could not be stored faithfully, before anything is read
/// or staged: an empty origin, key id or key body, a key id past the 255-byte
/// digest-binding limit, more keys than a response may carry, or two distinct
/// key ids that derive the same record id.
fn validate(update: &KeyUpdate<'_>) -> Result<(), StorageError> {
    let invalid = |what: &str| StorageError::Corrupt(format!("invalid server key update: {what}"));
    if update.origin.is_empty() {
        return Err(invalid("empty origin"));
    }
    let retired = update
        .observed
        .iter()
        .filter(|observed| observed.expired_ts.is_some())
        .count();
    if retired > update.limits.retired_per_response
        || update.observed.len().saturating_sub(retired) > update.limits.active_per_response
    {
        return Err(invalid("too many keys in one response"));
    }
    let mut seen: BTreeMap<[u8; 16], &[u8]> = BTreeMap::new();
    for observed in update.observed {
        if observed.key_id.is_empty() {
            return Err(invalid("empty key id"));
        }
        if observed.key_id.len() > MAX_KEY_ID_BYTES {
            return Err(invalid("key id is too long"));
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

/// Longest key id a digest binding may carry.
pub const MAX_KEY_ID_BYTES: usize = 255;

/// Retire every stored key a direct snapshot left out, then evict retired keys
/// below the retention floor.
///
/// The order is corroborated before uncorroborated, then retirement timestamp
/// descending, then `algorithm:key_id` ascending. Evicting clears the key bytes
/// and keeps the digest binding. Returns the issues for the keys evicted now.
fn retire_and_evict(
    current: &mut BTreeMap<[u8; 16], ServerKey>,
    seen: &BTreeSet<[u8; 16]>,
    update: &KeyUpdate<'_>,
) -> Vec<KeyIssue> {
    if update.snapshot && update.sighting == KeySighting::Direct {
        for (id, key) in current.iter_mut() {
            if !seen.contains(id) && key.retirement_ts().is_none() {
                key.inferred_retired = true;
            }
        }
    }
    let mut retained: Vec<&ServerKey> = current
        .values()
        .filter(|key| key.retirement_ts().is_some() && !key.is_evicted())
        .collect();
    retained.sort_by(|a, b| {
        b.corroborated
            .cmp(&a.corroborated)
            .then(b.retirement_ts().cmp(&a.retirement_ts()))
            .then(a.key_id.cmp(&b.key_id))
    });
    let evicted: Vec<(Vec<u8>, bool)> = retained
        .into_iter()
        .skip(update.limits.retired_retained)
        .map(|key| (key.key_id.clone(), key.corroborated))
        .collect();
    let mut issues = Vec::new();
    for (key_id, corroborated) in evicted {
        if let Some(key) = current.get_mut(&origin_key_record_id(&key_id)) {
            key.key.clear();
        }
        issues.push(KeyIssue {
            key_id,
            kind: if corroborated {
                KeyIssueKind::EvictedCorroborated
            } else {
                KeyIssueKind::Evicted
            },
        });
    }
    issues
}

/// The record ids of every key stored for `origin`, from a scan.
fn stored_ids(db: &Database, collection: &[u8; 16]) -> Result<Vec<[u8; 16]>, StorageError> {
    let mut ids = Vec::new();
    for item in db.server_info().scan_collection(collection)? {
        ids.push(item?.0);
    }
    Ok(ids)
}

/// Merge `update` into its origin's records in one transaction and return the
/// authoritative key for each distinct key id it carried, in first-seen order,
/// with the issues the caller should log.
///
/// Reads the stored records of the carried key ids and applies the binding
/// rules in order (a key id repeated in `observed` sees the earlier result).
/// An entry whose `expired_ts` is more than [`EXPIRED_TS_SKEW_MS`] ahead is
/// dropped without touching its slot. A key id new to its digest bucket is
/// refused once the bucket is at [`KeyLimits::digest_cap`]. A snapshot, or any
/// update that meets a new key id, also reads the origin's remaining records:
/// that is what retires keys a direct snapshot omits, counts the digest
/// buckets, and enforces the retired-key ceiling by evicting key bytes while
/// keeping the digest binding. That read pins the origin's whole collection
/// version, and every other read is version-checked per record, so an
/// overlapping writer, including one that inserts a record the scan missed,
/// fails the commit and the whole read-resolve-write is re-run on fresh data.
///
/// When a body collision, a conflicting `expired_ts` or a refused key was
/// rejected, the Notary payload is not stored, so what is served stays
/// consistent with what is bound.
///
/// # Errors
/// Returns an error if the update fails validation, if a stored
/// record is corrupt, if the pool cannot be read or written, or the stale-read
/// stale-read or `WouldBlock` error if every attempt failed because of a race
/// or a full journal segment.
pub fn update_server_keys(db: &Database, update: &KeyUpdate<'_>) -> Result<KeyMerge, StorageError> {
    validate(update)?;
    if update.observed.is_empty() && update.notary.is_none() && !update.snapshot {
        return Ok(KeyMerge {
            keys: Vec::new(),
            issues: Vec::new(),
        });
    }
    let collection = origin_keys_collection_id(update.origin);
    retry_transient_update(|| try_update(db, update, collection))
}

fn retry_transient_update<T>(
    mut update: impl FnMut() -> Result<T, StorageError>,
) -> Result<T, StorageError> {
    let mut attempt = 1;
    loop {
        match update() {
            Err(error)
                if (error.is_stale_read() || error.is_would_block())
                    && attempt < UPDATE_ATTEMPTS =>
            {
                attempt = attempt.saturating_add(1);
                let backoff = std::time::Duration::from_micros(
                    u64::try_from(attempt.min(16))
                        .unwrap_or(16)
                        .saturating_mul(50),
                );
                std::thread::sleep(backoff);
            }
            result => return result,
        }
    }
}

/// Read `ids` through `tx`, version-checking each, into `current`.
fn load(
    tx: &DatabaseTransaction<'_>,
    collection: [u8; 16],
    origin: &[u8],
    ids: &[[u8; 16]],
    current: &mut BTreeMap<[u8; 16], ServerKey>,
) -> Result<(), StorageError> {
    let (records, versions) = tx.get_with_record_versions(SERVER_INFO_POOL, &collection, ids)?;
    for ((id, record), version) in ids.iter().zip(&records).zip(&versions) {
        tx.expect_record_version(SERVER_INFO_POOL, collection, *id, *version)?;
        if let Some(data) = record {
            current.insert(*id, decode_for(origin, None, &data.bytes)?);
        }
    }
    Ok(())
}

fn try_update(
    db: &Database,
    update: &KeyUpdate<'_>,
    collection: [u8; 16],
) -> Result<KeyMerge, StorageError> {
    let latest = update.now_ms.saturating_add(EXPIRED_TS_SKEW_MS);
    let mut ids = Vec::new();
    for observed in update.observed {
        let id = origin_key_record_id(&observed.key_id);
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let seen: BTreeSet<[u8; 16]> = ids.iter().copied().collect();
    let tx = db.begin_transaction();
    // Sampled before any data is read, so it can only be older than what the
    // scan below sees: a write that lands later makes the commit stale, and a
    // write the scan already saw only costs a retry.
    let (_, version) = tx.get_with_collection_version(SERVER_INFO_POOL, &collection, &[])?;
    let mut current: BTreeMap<[u8; 16], ServerKey> = BTreeMap::new();
    load(&tx, collection, update.origin, &ids, &mut current)?;
    // An evicted key whose body returns competes for a retention slot again,
    // which only a full read can rank.
    if update.snapshot
        || ids.iter().any(|id| !current.contains_key(id))
        || current.values().any(ServerKey::is_evicted)
    {
        let rest: Vec<[u8; 16]> = stored_ids(db, &collection)?
            .into_iter()
            .filter(|id| !seen.contains(id))
            .collect();
        load(&tx, collection, update.origin, &rest, &mut current)?;
        // The scan is not versioned record by record, so pin the whole
        // collection: any insert it could have missed fails the commit.
        tx.expect_collection_version(SERVER_INFO_POOL, collection, version)?;
    }
    let stored = current.clone();
    let bucket = |tier: KeyTier, records: &BTreeMap<[u8; 16], ServerKey>| {
        records.values().filter(|key| key.tier == tier).count()
    };
    let mut order = Vec::with_capacity(ids.len());
    let mut issues = Vec::new();
    let mut rejected = false;
    for observed in update.observed {
        let issue = |kind| KeyIssue {
            key_id: observed.key_id.clone(),
            kind,
        };
        if observed.expired_ts.is_some_and(|expired| expired > latest) {
            issues.push(issue(KeyIssueKind::FutureExpiredTs));
            rejected = true;
            continue;
        }
        let id = origin_key_record_id(&observed.key_id);
        if !current.contains_key(&id) {
            let tier = match update.sighting {
                KeySighting::Direct => KeyTier::Permanent,
                KeySighting::Notary => KeyTier::Provisional,
            };
            if bucket(tier, &current) >= update.limits.digest_cap {
                issues.push(issue(KeyIssueKind::DigestCapReached));
                rejected = true;
                continue;
            }
        }
        let (bound, kind) = bind(current.get(&id), observed, update);
        if let Some(kind) = kind {
            rejected |= kind != KeyIssueKind::Displaced;
            issues.push(issue(kind));
        }
        current.insert(id, bound);
        if !order.contains(&id) {
            order.push(id);
        }
    }
    issues.extend(retire_and_evict(&mut current, &seen, update));
    for (id, key) in &current {
        if stored.get(id) != Some(key) {
            put_server_key(&tx, update.origin, key)?;
        }
    }
    if let Some(payload) = update.notary.filter(|_| !rejected) {
        put_notary_payload(&tx, update.origin, payload)?;
    }
    tx.commit()?;
    Ok(KeyMerge {
        keys: order.iter().map(|id| current[id].clone()).collect(),
        issues,
    })
}

#[cfg(test)]
#[path = "test_server_keys.rs"]
mod tests;
