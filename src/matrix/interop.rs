//! Shared on-disk layouts for the server/importer boundary.
//!
//! mtxdb is the only store for events, verdicts, signing keys and state. A
//! server writes those records through its adapters; an `mtxdb import`
//! batch writes the same records from an export file. Both must agree on
//! every byte, or a room that imported cleanly reads as empty after a
//! restart. This module owns that agreement: the encodings, the collection
//! and node derivations, and the data-format version the import marker
//! vouches for.
//!
//! Identity note: room collection ids use mtxdb's BLAKE3 group derivation
//! plus a namespace bias. Other interop ids use BLAKE3-128; where multiple
//! logical ids share a collection, their derivation uses a NUL-separated
//! domain tag. Event record ids deliberately match the Matrix template's
//! BLAKE3-128 of the complete event id.
//!
//! State HAMT construction and state derivation are deliberately not
//! implemented here. The server and offline importer must use the same
//! `rezzy` HAMT builder and encoding; this module owns only stable mtxdb
//! collection/node identities and bounded publication helpers.

use crate::{
    derive_group_full_id, derive_member_collection_id_from_group, storage::StorageError, ShardType,
    MEMBER_NAMESPACE_AUTH, MEMBER_NAMESPACE_EVNT, MEMBER_NAMESPACE_PREV, MEMBER_NAMESPACE_STAT,
    MEMBER_NAMESPACE_STGP,
};

/// Serialized `rezzy::hamt::RootHandle` length used by the server's state
/// adapter. Field order is codec version, routing version, routing parameters,
/// structural hash, then state-group id.
pub const STATE_ROOT_HANDLE_V1_LEN: usize = 70;

/// The byte-level root handle shared by the importer and the Matrix server.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateRootHandleV1 {
    /// HAMT node codec version.
    pub codec_version: u8,
    /// HAMT routing version.
    pub routing_version: u8,
    /// Parameters for the selected routing version.
    pub routing_params: [u8; 4],
    /// Full content-addressed structural hash of the root node.
    pub structural_hash: [u8; 32],
    /// Cross-server state-group identity derived from the state's `LtHash`.
    pub state_group_id: [u8; 32],
}

/// Encode the server's fixed-width state root handle as 70 bytes, ordered as
/// codec version, routing version, routing parameters, structural hash, and
/// state-group ID. Field values are copied without validation.
#[must_use]
pub fn encode_state_root_handle_v1(handle: StateRootHandleV1) -> [u8; STATE_ROOT_HANDLE_V1_LEN] {
    let mut out = [0; STATE_ROOT_HANDLE_V1_LEN];
    out[0] = handle.codec_version;
    out[1] = handle.routing_version;
    out[2..6].copy_from_slice(&handle.routing_params);
    out[6..38].copy_from_slice(&handle.structural_hash);
    out[38..70].copy_from_slice(&handle.state_group_id);
    out
}

/// Decode the server's fixed-width state root handle.
///
/// Version and routing-parameter values are preserved without validation.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] unless `bytes` has the exact v1 length.
pub fn decode_state_root_handle_v1(bytes: &[u8]) -> Result<StateRootHandleV1, StorageError> {
    if bytes.len() != STATE_ROOT_HANDLE_V1_LEN {
        return Err(StorageError::Corrupt(format!(
            "malformed state root handle: expected {STATE_ROOT_HANDLE_V1_LEN} bytes, got {}",
            bytes.len()
        )));
    }
    let malformed = || StorageError::Corrupt("malformed state root handle".to_owned());
    let (codec_version, fields) = bytes.split_first().ok_or_else(malformed)?;
    let (routing_version, fields) = fields.split_first().ok_or_else(malformed)?;
    let (routing_params, fields) = fields
        .split_first_chunk::<4>()
        .ok_or_else(|| StorageError::Corrupt("malformed state root handle".to_owned()))?;
    let (structural_hash, fields) = fields
        .split_first_chunk::<32>()
        .ok_or_else(|| StorageError::Corrupt("malformed state root handle".to_owned()))?;
    let (state_group_id, _) = fields.split_first_chunk::<32>().ok_or_else(malformed)?;
    Ok(StateRootHandleV1 {
        codec_version: *codec_version,
        routing_version: *routing_version,
        routing_params: *routing_params,
        structural_hash: *structural_hash,
        state_group_id: *state_group_id,
    })
}

/// Derive the room's STAT collection for HAMT nodes.
///
/// # Panics
/// The fixed STAT namespace is valid, so this cannot panic unless the engine's
/// namespace validation changes without updating this function.
#[must_use]
pub fn room_state_collection_id(room_id: &[u8]) -> [u8; 16] {
    room_member_collection(MEMBER_NAMESPACE_STAT, room_id)
}

/// Derive the room's STGP collection for room and event root handles.
///
/// # Panics
/// The fixed STGP namespace is valid, so this cannot panic unless the engine's
/// namespace validation changes without updating this function.
#[must_use]
pub fn room_roots_collection_id(room_id: &[u8]) -> [u8; 16] {
    room_member_collection(MEMBER_NAMESPACE_STGP, room_id)
}

/// Return the first 16 bytes of a HAMT structural hash as its stored node id.
#[must_use]
pub fn state_hamt_node_id(structural_hash: &[u8; 32]) -> [u8; 16] {
    let mut node_id = [0; 16];
    node_id.copy_from_slice(&structural_hash[..16]);
    node_id
}

/// Derive the node id holding a room's current state root.
///
/// This fixed id is shared across rooms, each in its own roots collection.
#[must_use]
pub fn room_state_root_node_id() -> [u8; 16] {
    derive_id(b"root", b"current")
}

/// Derive the node id holding an event's resolved state root.
#[must_use]
pub fn event_state_root_node_id(event_id: &[u8]) -> [u8; 16] {
    derive_id(b"event-root", event_id)
}

/// Version of the export/import data contract the marker vouches for. Bump
/// it when a stored layout below changes incompatibly, so an older database
/// is refused instead of misread.
pub const DATA_FORMAT: u8 = 1;

/// Derive a 128-bit BLAKE3 identifier from a domain and logical identifier.
///
/// Hashes `domain`, a NUL separator, then `identifier`, taking the first
/// 16 digest bytes. Domains must exclude NUL bytes for unambiguous separation.
/// Callers that need collision detection must retain the original identifier
/// beside the derived id — 128 bits is not enough to rule out a collision by
/// itself.
#[must_use]
pub fn derive_id(domain: &[u8], identifier: &[u8]) -> [u8; 16] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(identifier);

    let mut id = [0_u8; 16];
    id.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    id
}

/// Number of bucketed verdict collections. Fixed for the life of a database:
/// the bucket is part of where a verdict lives, so changing it orphans every
/// existing entry.
pub const VERDICT_BUCKETS: u32 = 256;

/// Version byte of the verdict value encoding.
pub const VERDICT_VERSION: u8 = 1;
const HAS_REJECTION: u8 = 1;
const HAS_SOFT_FAIL: u8 = 1 << 1;
const HAS_REDACTION: u8 = 1 << 2;
const IS_OUTLIER: u8 = 1 << 3;

/// Everything recorded about an event beyond its body and edges.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Verdict {
    /// Why the event was rejected, as the source numbered it.
    pub rejection: Option<u8>,
    /// Why the event soft-failed, as the source numbered it.
    pub soft_fail: Option<u8>,
    /// Event id of the redaction that redacted this event.
    pub redacted_by: Option<Vec<u8>>,
    /// Whether the event is stored as an outlier. This is independent of its
    /// rejection and soft-fail verdicts.
    pub outlier: bool,
}

impl Verdict {
    /// Whether there is nothing worth recording.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rejection.is_none()
            && self.soft_fail.is_none()
            && self.redacted_by.is_none()
            && !self.outlier
    }
}

/// Derive an event record id as the first 16 bytes of the BLAKE3 hash of the
/// complete event id, including any leading `$` and server name.
#[must_use]
pub fn event_record_id(event_id: &[u8]) -> [u8; 16] {
    let digest = blake3::hash(event_id);
    let mut id = [0_u8; 16];
    id.copy_from_slice(&digest.as_bytes()[..16]);
    id
}

/// Derive the per-event verdict collection holding `node`.
///
/// Bucketed so one write's cost stays proportional to its bucket rather
/// than to the server's total event count: mtxdb clones a collection's
/// index on every write, so a single shared collection would make every
/// verdict write pay for every other.
#[must_use]
pub fn event_verdict_collection_id(node: &[u8; 16]) -> [u8; 16] {
    let bucket = u32::from_le_bytes([node[0], node[1], node[2], node[3]]) % VERDICT_BUCKETS;
    derive_id(b"sys", format!("event-verdict:{bucket}").as_bytes())
}

/// Encode one event's verdict for storage.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if a field does not fit its length prefix.
pub fn encode_verdict(event_id: &[u8], verdict: &Verdict) -> Result<Vec<u8>, StorageError> {
    let too_large = || StorageError::Corrupt("verdict field is too large".to_owned());
    let mut flags = 0_u8;
    if verdict.outlier {
        flags |= IS_OUTLIER;
    }
    if verdict.rejection.is_some() {
        flags |= HAS_REJECTION;
    }
    if verdict.soft_fail.is_some() {
        flags |= HAS_SOFT_FAIL;
    }
    if verdict.redacted_by.is_some() {
        flags |= HAS_REDACTION;
    }

    let mut out = vec![VERDICT_VERSION, flags];
    out.extend_from_slice(
        &u32::try_from(event_id.len())
            .map_err(|_| too_large())?
            .to_le_bytes(),
    );
    out.extend_from_slice(event_id);
    out.extend(verdict.rejection);
    out.extend(verdict.soft_fail);
    if let Some(redactor) = &verdict.redacted_by {
        out.extend_from_slice(
            &u32::try_from(redactor.len())
                .map_err(|_| too_large())?
                .to_le_bytes(),
        );
        out.extend_from_slice(redactor);
    }
    Ok(out)
}

/// Decode a stored verdict, returning the event id it was written for.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record or an unknown
/// version.
pub fn decode_verdict(bytes: &[u8]) -> Result<(&[u8], Verdict), StorageError> {
    let malformed = || StorageError::Corrupt("malformed event verdict".to_owned());
    let take_len = |rest: &[u8]| -> Result<(usize, usize), StorageError> {
        let (length, _) = rest.split_first_chunk::<4>().ok_or_else(malformed)?;
        Ok((
            usize::try_from(u32::from_le_bytes(*length)).map_err(|_| malformed())?,
            4,
        ))
    };

    let (&version, rest) = bytes.split_first().ok_or_else(malformed)?;
    if version != VERDICT_VERSION {
        return Err(StorageError::Corrupt(
            "unknown event verdict version".to_owned(),
        ));
    }
    let (&flags, rest) = rest.split_first().ok_or_else(malformed)?;
    let (length, header) = take_len(rest)?;
    let rest = rest.get(header..).ok_or_else(malformed)?;
    if rest.len() < length {
        return Err(malformed());
    }
    let (event_id, mut rest) = rest.split_at(length);

    let mut verdict = Verdict {
        outlier: flags & IS_OUTLIER != 0,
        ..Verdict::default()
    };
    if flags & HAS_REJECTION != 0 {
        let (&code, tail) = rest.split_first().ok_or_else(malformed)?;
        verdict.rejection = Some(code);
        rest = tail;
    }
    if flags & HAS_SOFT_FAIL != 0 {
        let (&code, tail) = rest.split_first().ok_or_else(malformed)?;
        verdict.soft_fail = Some(code);
        rest = tail;
    }
    if flags & HAS_REDACTION != 0 {
        let (length, header) = take_len(rest)?;
        let tail = rest.get(header..).ok_or_else(malformed)?;
        verdict.redacted_by = Some(tail.get(..length).ok_or_else(malformed)?.to_vec());
    }
    Ok((event_id, verdict))
}

/// Version byte of the signing-key record value encoding.
pub const KEY_RECORD_VERSION: u8 = 1;

/// Which of an origin's signing-key records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyRecordKind {
    /// Cumulative `verify_keys` and `old_verify_keys` ever seen.
    Historical,
    /// Key ids bound only through a notary so far.
    Provisional,
    /// Key ids observed as currently active by a direct fetch.
    Corroborated,
    /// The last verbatim self-signed payload, re-served by the notary.
    Notary,
}

impl KeyRecordKind {
    /// Every kind, for scan and summary paths.
    pub const ALL: [Self; 4] = [
        Self::Historical,
        Self::Provisional,
        Self::Corroborated,
        Self::Notary,
    ];

    /// The kind tag written into the record-id pre-image.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Historical => "historical",
            Self::Provisional => "provisional",
            Self::Corroborated => "corroborated",
            Self::Notary => "notary",
        }
    }

    /// Parse a kind tag, or `None` for an unknown tag.
    #[must_use]
    pub fn parse(tag: &str) -> Option<Self> {
        match tag {
            "historical" => Some(Self::Historical),
            "provisional" => Some(Self::Provisional),
            "corroborated" => Some(Self::Corroborated),
            "notary" => Some(Self::Notary),
            _ => None,
        }
    }

    fn as_bytes(self) -> &'static [u8] {
        self.as_str().as_bytes()
    }
}

/// Derive the collection holding every origin's signing-key records.
#[must_use]
pub fn server_keys_collection_id() -> [u8; 16] {
    derive_id(b"sys", b"server-keys")
}

/// Derive the record id of one signing-key record of `origin`.
#[must_use]
pub fn server_key_record_id(kind: KeyRecordKind, origin: &[u8]) -> [u8; 16] {
    let mut identifier = Vec::with_capacity(
        kind.as_bytes()
            .len()
            .saturating_add(origin.len())
            .saturating_add(1),
    );
    identifier.extend_from_slice(kind.as_bytes());
    identifier.push(0);
    identifier.extend_from_slice(origin);
    derive_id(b"server-key", &identifier)
}

/// Encode one signing-key record for storage.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if the origin does not fit its length
/// prefix.
pub fn encode_server_key(origin: &[u8], payload: &[u8]) -> Result<Vec<u8>, StorageError> {
    let length = u32::try_from(origin.len())
        .map_err(|_| StorageError::Corrupt("origin is too large".to_owned()))?;
    let mut out = Vec::with_capacity(
        5_usize
            .saturating_add(origin.len())
            .saturating_add(payload.len()),
    );
    out.push(KEY_RECORD_VERSION);
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(origin);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Decode a stored signing-key record into its origin and payload.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record or an unknown
/// version.
pub fn decode_server_key(bytes: &[u8]) -> Result<(&[u8], &[u8]), StorageError> {
    let malformed = || StorageError::Corrupt("malformed signing-key record".to_owned());
    let (&version, rest) = bytes.split_first().ok_or_else(malformed)?;
    if version != KEY_RECORD_VERSION {
        return Err(StorageError::Corrupt(
            "unknown signing-key record version".to_owned(),
        ));
    }
    let (length, rest) = rest.split_first_chunk::<4>().ok_or_else(malformed)?;
    let length = usize::try_from(u32::from_le_bytes(*length)).map_err(|_| malformed())?;
    if rest.len() < length {
        return Err(malformed());
    }
    Ok(rest.split_at(length))
}

/// Pool holding event bodies, event locators and verdicts.
pub const EVENT_POOL: ShardType = ShardType::EventDag;

/// Pool holding the prev and auth edge records.
pub const EDGE_POOL: ShardType = ShardType::Edges;

/// Pool holding signing keys and the import marker.
pub const SERVER_INFO_POOL: ShardType = ShardType::ServerInfo;

/// Number of bucketed event-locator collections. Fixed for the life of a
/// database, like [`VERDICT_BUCKETS`].
pub const EVENT_LOCATOR_BUCKETS: u32 = 256;

/// Version byte of the event-locator value encoding.
pub const LOCATOR_VERSION: u8 = 1;

/// Derive a room member collection from the room ID and namespace.
///
/// # Panics
/// Panics if `namespace` is not a registered member namespace.
fn room_member_collection(namespace: [u8; 4], room_id: &[u8]) -> [u8; 16] {
    derive_member_collection_id_from_group(namespace, derive_group_full_id(room_id))
        .expect("Matrix member namespace is registered")
}

/// Derive the collection holding a room's event bodies.
#[must_use]
pub fn room_event_collection_id(room_id: &[u8]) -> [u8; 16] {
    room_member_collection(MEMBER_NAMESPACE_EVNT, room_id)
}

/// Derive the collection holding a room's prev-event edge lists.
#[must_use]
pub fn room_prev_collection_id(room_id: &[u8]) -> [u8; 16] {
    room_member_collection(MEMBER_NAMESPACE_PREV, room_id)
}

/// Derive the collection holding a room's auth-event edge lists.
#[must_use]
pub fn room_auth_collection_id(room_id: &[u8]) -> [u8; 16] {
    room_member_collection(MEMBER_NAMESPACE_AUTH, room_id)
}

/// Derive the event-locator collection holding `node`.
///
/// The locator maps an event id to its room across all rooms, bucketed for
/// the same reason as [`event_verdict_collection_id`].
#[must_use]
pub fn event_locator_collection_id(node: &[u8; 16]) -> [u8; 16] {
    let bucket = u32::from_le_bytes([node[0], node[1], node[2], node[3]]) % EVENT_LOCATOR_BUCKETS;
    derive_id(b"sys", format!("event-locator:{bucket}").as_bytes())
}

/// Encode an event locator: `[version][u32 LE length][event id][room id]`.
///
/// The full event id rides along because the record id is only 128 bits.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if the event id does not fit its length
/// prefix.
pub fn encode_event_locator(event_id: &[u8], room_id: &[u8]) -> Result<Vec<u8>, StorageError> {
    let length = u32::try_from(event_id.len())
        .map_err(|_| StorageError::Corrupt("event id is too large".to_owned()))?;
    let mut out = Vec::with_capacity(
        1_usize
            .saturating_add(4)
            .saturating_add(event_id.len())
            .saturating_add(room_id.len()),
    );
    out.push(LOCATOR_VERSION);
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(event_id);
    out.extend_from_slice(room_id);
    Ok(out)
}

/// Decode an event locator into `(event id, room id)`.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record or an unknown
/// version.
pub fn decode_event_locator(bytes: &[u8]) -> Result<(&[u8], &[u8]), StorageError> {
    let malformed = || StorageError::Corrupt("malformed event locator".to_owned());
    let (&version, rest) = bytes.split_first().ok_or_else(malformed)?;
    if version != LOCATOR_VERSION {
        return Err(StorageError::Corrupt(
            "unknown event locator version".to_owned(),
        ));
    }
    let (length, rest) = rest.split_first_chunk::<4>().ok_or_else(malformed)?;
    let length = usize::try_from(u32::from_le_bytes(*length)).map_err(|_| malformed())?;
    if rest.len() < length {
        return Err(malformed());
    }
    Ok(rest.split_at(length))
}

/// Encode an edge list (prev or auth) as repeated `[u32 LE length][event id]`.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] if an id does not fit its length prefix.
pub fn encode_edges(edges: &[&[u8]]) -> Result<Vec<u8>, StorageError> {
    let too_large = || StorageError::Corrupt("event edge is too large".to_owned());
    let mut out = Vec::new();
    for edge in edges {
        let length = u32::try_from(edge.len()).map_err(|_| too_large())?;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(edge);
    }
    Ok(out)
}

/// Record id of the per-room auth short-id counter (big-endian `u32`).
#[must_use]
pub fn auth_counter_record_id() -> [u8; 16] {
    derive_id(b"authchain:counter", b"")
}

/// Record id of an event's forward short-id record in the room auth collection.
#[must_use]
pub fn auth_forward_record_id(event_id: &[u8]) -> [u8; 16] {
    derive_id(b"authchain:fwd", event_id)
}

/// Record id of a short id's reverse record (the event id).
#[must_use]
pub fn auth_reverse_record_id(short: u32) -> [u8; 16] {
    derive_id(b"authchain:rev", &short.to_be_bytes())
}

/// Record id of a short id's closure bitmap.
#[must_use]
pub fn auth_closure_record_id(short: u32) -> [u8; 16] {
    derive_id(b"authchain:closure", &short.to_be_bytes())
}

/// Forward short-id record: `[u32 BE short][event id]`. The event id lets a
/// reader reject a record that merely shares the 128-bit key.
#[must_use]
pub fn encode_auth_forward(short: u32, event_id: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(event_id.len().saturating_add(4));
    out.extend_from_slice(&short.to_be_bytes());
    out.extend_from_slice(event_id);
    out
}

/// Decode a forward short-id record.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a record shorter than its prefix.
pub fn decode_auth_forward(bytes: &[u8]) -> Result<(u32, &[u8]), StorageError> {
    let (short, event) = bytes
        .split_first_chunk::<4>()
        .ok_or_else(|| StorageError::Corrupt("malformed auth short-id record".to_owned()))?;
    Ok((u32::from_be_bytes(*short), event))
}

/// Decode the per-room short-id counter.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] unless the record is exactly four bytes.
pub fn decode_auth_counter(bytes: &[u8]) -> Result<u32, StorageError> {
    <[u8; 4]>::try_from(bytes)
        .map(u32::from_be_bytes)
        .map_err(|_| StorageError::Corrupt("malformed auth counter".to_owned()))
}

/// Encode the per-room short-id counter as big-endian `u32`.
#[must_use]
pub const fn encode_auth_counter(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

/// Decode an edge list written by [`encode_edges`].
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record.
pub fn decode_edges(bytes: &[u8]) -> Result<Vec<Vec<u8>>, StorageError> {
    let malformed = || StorageError::Corrupt("malformed event edge record".to_owned());
    let mut edges = Vec::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        let (length, tail) = rest.split_first_chunk::<4>().ok_or_else(malformed)?;
        let length = usize::try_from(u32::from_le_bytes(*length)).map_err(|_| malformed())?;
        let (edge, tail) = tail.split_at_checked(length).ok_or_else(malformed)?;
        edges.push(edge.to_vec());
        rest = tail;
    }
    Ok(edges)
}

/// Version byte of the import-marker value encoding.
pub const MARKER_VERSION: u8 = 1;

/// What an import (or a fresh install) recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportMarker {
    /// Data-format version this database's records conform to.
    pub data_format: u8,
    /// Events imported. Zero for a fresh install.
    pub events: u64,
    /// Auth closure bitmaps written. Zero for a fresh install.
    pub closures: u64,
}

/// Derive the collection holding the import-complete marker.
#[must_use]
pub fn import_marker_collection_id() -> [u8; 16] {
    derive_id(b"sys", b"import-marker")
}

/// Derive the marker's node id within its collection.
#[must_use]
pub fn import_marker_node_id() -> [u8; 16] {
    derive_id(b"marker", b"import")
}

/// Encode the import marker for storage.
#[must_use]
pub fn encode_import_marker(marker: ImportMarker) -> Vec<u8> {
    let mut out = vec![MARKER_VERSION, marker.data_format];
    out.extend_from_slice(&marker.events.to_le_bytes());
    out.extend_from_slice(&marker.closures.to_le_bytes());
    out
}

/// Decode a stored import marker.
///
/// # Errors
/// Returns [`StorageError::Corrupt`] for a truncated record or an unknown
/// version.
pub fn decode_import_marker(bytes: &[u8]) -> Result<ImportMarker, StorageError> {
    let malformed = || StorageError::Corrupt("malformed import marker".to_owned());
    let (&version, rest) = bytes.split_first().ok_or_else(malformed)?;
    if version != MARKER_VERSION {
        return Err(StorageError::Corrupt(
            "unknown import marker version".to_owned(),
        ));
    }
    let (&data_format, rest) = rest.split_first().ok_or_else(malformed)?;
    let (events, rest) = rest.split_first_chunk::<8>().ok_or_else(malformed)?;
    let (closures, _) = rest.split_first_chunk::<8>().ok_or_else(malformed)?;
    Ok(ImportMarker {
        data_format,
        events: u64::from_le_bytes(*events),
        closures: u64::from_le_bytes(*closures),
    })
}

#[cfg(test)]
#[path = "test_interop.rs"]
mod tests;

/// Pool holding HAMT nodes and state roots.
pub const STATE_POOL: ShardType = ShardType::State;

/// Default charged staging budget per state batch, including per-put framing.
pub const STATE_BATCH_BYTES: usize = 48 << 20;

/// Stage HAMT nodes, then roots, into bounded transactions of at most
/// `batch_bytes` charged bytes, committing each before the next begins.
///
/// All supplied nodes are committed before any root. The caller must supply
/// every node needed by the roots that is not already stored; references and
/// content hashes are not validated. An interrupted run can leave unreferenced
/// nodes and already-committed roots. `nodes` and `roots` are
/// `(collection, node id, bytes)` stored in [`STATE_POOL`]; the caller derives
/// ids with the functions above. The budget includes payload bytes plus
/// [`crate::journal::TXN_STAGE_PUT_OVERHEAD_BYTES`] per record. Returns the
/// number of transactions committed, or zero if both inputs are empty.
///
/// # Errors
/// Returns the first staging or commit error; earlier batches stay committed.
/// Returns [`StorageError::Io`] with [`std::io::ErrorKind::InvalidInput`] for
/// a zero or over-engine-limit budget, staging-size overflow, or a single
/// record that cannot fit in the requested budget.
///
/// This is not an atomic migration across batches. The caller must publish its
/// import-complete marker only after this function succeeds; on failure,
/// already-committed nodes/roots remain visible, with nodes always committed
/// before roots.
pub fn commit_state_batches<'a>(
    db: &crate::Database,
    nodes: impl IntoIterator<Item = ([u8; 16], [u8; 16], &'a [u8])>,
    roots: impl IntoIterator<Item = ([u8; 16], [u8; 16], &'a [u8])>,
    batch_bytes: usize,
) -> Result<usize, StorageError> {
    if batch_bytes == 0 || batch_bytes > crate::journal::MAX_TXN_STAGE_BYTES {
        return Err(StorageError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "state batch budget must be in 1..={} bytes",
                crate::journal::MAX_TXN_STAGE_BYTES
            ),
        )));
    }
    let mut committed = 0_usize;
    for records in [
        nodes.into_iter().collect::<Vec<_>>(),
        roots.into_iter().collect(),
    ] {
        let mut txn = db.begin_transaction();
        let mut staged = 0_usize;
        for (collection, node, bytes) in records {
            let charge = bytes
                .len()
                .checked_add(crate::journal::TXN_STAGE_PUT_OVERHEAD_BYTES)
                .ok_or_else(|| {
                    StorageError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "state record staging size overflow",
                    ))
                })?;
            if charge > batch_bytes {
                return Err(StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "one state record exceeds the transaction staging budget",
                )));
            }
            if staged > 0 && staged.saturating_add(charge) > batch_bytes {
                txn.commit()?;
                committed = committed.saturating_add(1);
                txn = db.begin_transaction();
                staged = 0;
            }
            txn.put(
                STATE_POOL,
                collection,
                node,
                &crate::storage::NodeData::new(bytes::Bytes::copy_from_slice(bytes)),
            )?;
            staged = staged.saturating_add(charge);
        }
        if staged > 0 {
            txn.commit()?;
            committed = committed.saturating_add(1);
        }
    }
    Ok(committed)
}
