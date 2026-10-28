//! Writing an `mtxdb-export/v1` file into the server's own layouts.
//!
//! Every record goes where `mtxdb::interop` says the server reads it: the
//! room's event, prev and auth collections, the bucketed locator and verdict
//! collections, and the signing-key collection. Nothing here invents a
//! layout. Each room's auth short ids and closure bitmaps are written here too,
//! by the same completeness rule the server applies when it writes an event: an
//! event gets a closure only when every auth event it cites has one. State is
//! not written.
//!
//! A run is complete only when the import marker says so. Before the first
//! record of a run the marker is cleared, data goes in capped batches, and the
//! marker is committed last, in a transaction of its own. A run that stops
//! anywhere earlier leaves the database visibly incomplete.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{anyhow, bail, Context};
use mtxdb::export_schema::RejectionCode;
use mtxdb::interop::{
    auth_closure_record_id, auth_counter_record_id, auth_forward_record_id, auth_reverse_record_id,
    decode_auth_counter, decode_auth_forward, decode_event_locator, decode_import_marker,
    encode_auth_forward, encode_edges, encode_event_locator, encode_import_marker,
    encode_room_version_record, encode_verdict, event_locator_collection_id, event_record_id,
    event_verdict_collection_id, import_marker_collection_id, import_marker_node_id,
    room_auth_collection_id, room_event_collection_id, room_prev_collection_id,
    room_version_collection_id, room_version_node_id, ImportMarker, Verdict, DATA_FORMAT,
    EDGE_POOL, EVENT_POOL, SERVER_INFO_POOL,
};
use mtxdb::server_keys;
use mtxdb::storage::StorageEngine as _;
use mtxdb::{Database, NodeData, PackfileStorage};
use rezzy::auth::{check_auth, StateProvider};
use rezzy::basespec::event_types::M_ROOM_CREATE;
use rezzy::bitmap::Bitmap;
use rezzy::json::Value as OwnedValue;
use rezzy::signing::{
    ed25519_zebra::{Signature, VerificationKey},
    verify_event_signatures_from_server, SignatureVerifier,
};
use rezzy::{LeanEvent, StateResVersion};

use crate::interchange::{EventMetadata, KeyRecord, RoomEvent, RoomExport};
use crate::state_resolve::lean_event;

/// Bytes one batch may stage. Half the engine's per-transaction limit, so the
/// estimate below can be rough.
pub(crate) const BATCH_BYTES: usize = 32 << 20;

/// Let the shared journal reclaim during a long import.
///
/// The journal is one 256 MiB segment shared by every pool, and it is only
/// reclaimed once every pool the writer touched has synced and so reported
/// coverage. An import that syncs only at the end fills the segment and every
/// later commit fails with "journal segment is full". Once the segment passes
/// `trigger` bytes, sync every pool, one after another; the last pool's
/// coverage lets the journal shrink. At or below `trigger` this does nothing.
pub(crate) fn reclaim_journal_with(
    segment_len: u64,
    trigger: u64,
    mut sync_pool: impl FnMut(mtxdb::ShardType) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    if segment_len <= trigger {
        return Ok(());
    }
    for pool in mtxdb::ShardType::ALL {
        sync_pool(pool)?;
    }
    Ok(())
}

/// [`reclaim_journal_with`] against the database's own journal and pools.
pub(crate) fn reclaim_journal(db: &Database) -> anyhow::Result<()> {
    let coordinator = db.coordinator();
    reclaim_journal_with(
        coordinator.segment_len(),
        coordinator.reclaim_trigger_len(),
        |pool| {
            db.pool(pool)
                .sync_all()
                .with_context(|| format!("syncing the {pool:?} pool to reclaim the journal"))
        },
    )
}

/// Allowance per staged record for keys and framing, on top of its value.
const RECORD_OVERHEAD: usize = 128;
fn node(bytes: Vec<u8>) -> NodeData {
    NodeData::new(bytes::Bytes::from(bytes))
}

fn room_version_record(room_id: &str, events: &[RoomEvent]) -> anyhow::Result<Vec<u8>> {
    let version = events
        .iter()
        .find(|event| {
            event.pdu.get("type").and_then(OwnedValue::as_str) == Some(M_ROOM_CREATE)
                && event.pdu.get("state_key").and_then(OwnedValue::as_str) == Some("")
        })
        .and_then(|event| event.pdu.get("content"))
        .and_then(|content| content.get("room_version"))
        .and_then(OwnedValue::as_str)
        .ok_or_else(|| anyhow!("room {room_id} has no create-event room_version"))?;
    encode_room_version_record(room_id, version).context("encoding room-version record")
}

/// Stage the room's immutable version record into `tx`, unless an identical
/// one is already stored. Returns the bytes the record reserves in the batch
/// budget.
fn stage_room_version(
    db: &Database,
    tx: &mut mtxdb::DatabaseTransaction<'_>,
    room_id: &str,
    events: &[RoomEvent],
    collection: [u8; 16],
    node_id: [u8; 16],
) -> anyhow::Result<usize> {
    let record = room_version_record(room_id, events)?;
    let (existing, collection_version) = db
        .pool(SERVER_INFO_POOL)
        .get_with_collection_version(&collection, &[node_id])
        .context("reading room-version record")?;
    match existing.into_iter().next().flatten() {
        Some(existing) if existing.bytes.as_ref() == record.as_slice() => Ok(0),
        Some(_) => bail!("room {room_id} has a conflicting immutable room-version record"),
        None => {
            tx.expect_collection_version(SERVER_INFO_POOL, collection, collection_version)
                .context("guarding room-version record")?;
            let staged = record.len().saturating_add(RECORD_OVERHEAD);
            tx.put(SERVER_INFO_POOL, collection, node_id, &node(record))
                .context("writing room-version record")?;
            Ok(staged)
        }
    }
}

/// Refuse to overwrite a stored locator that names a different event.
fn check_locator_collision(
    tx: &mtxdb::DatabaseTransaction<'_>,
    record: [u8; 16],
    event_id: &str,
) -> anyhow::Result<()> {
    let collection = event_locator_collection_id(&record);
    let existing = tx
        .get(EVENT_POOL, &collection, &[record])?
        .into_iter()
        .next()
        .flatten();
    if let Some(existing) = existing {
        let (stored, _) = decode_event_locator(&existing.bytes)?;
        if stored != event_id.as_bytes() {
            bail!("{event_id}: record id collides with a stored event; refusing to shadow it");
        }
    }
    Ok(())
}

/// Whether the database holds a marker, and what it says.
///
/// Production code only needs presence (see [`clear_marker`]); reading the
/// counts back is for tests and diagnostics.
#[allow(dead_code)] // read-side counterpart of `write_marker`, used by tests
pub(crate) fn read_marker(db: &Database) -> anyhow::Result<Option<ImportMarker>> {
    let record = db
        .pool(SERVER_INFO_POOL)
        .get(&import_marker_collection_id(), &import_marker_node_id())
        .context("reading the import marker")?;
    record
        .map(|data| decode_import_marker(&data.bytes))
        .transpose()
        .context("decoding the import marker")
}

/// Drop any marker left by an earlier run, in its own transaction.
///
/// Without this, a run interrupted on top of a complete import would still
/// look complete: the old marker would vouch for data it no longer describes.
///
/// Presence is checked on the raw record, not by decoding: a marker written by
/// an older data format must still be clearable, or the re-import that
/// migrates it could never start.
pub(crate) fn clear_marker(db: &Database) -> anyhow::Result<()> {
    let present = db
        .pool(SERVER_INFO_POOL)
        .get(&import_marker_collection_id(), &import_marker_node_id())
        .context("reading the import marker")?;
    if present.is_none() {
        return Ok(());
    }
    let tx = db.begin_transaction();
    tx.remove(
        SERVER_INFO_POOL,
        import_marker_collection_id(),
        import_marker_node_id(),
    )?;
    tx.commit().context("clearing the import marker")?;
    Ok(())
}

/// What a run wrote, per class.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ImportCounts {
    pub(crate) events: u64,
    pub(crate) closures: u64,
}

/// Commit the marker, last and alone, with the counts of this run's writes.
pub(crate) fn write_marker(db: &Database, counts: ImportCounts) -> anyhow::Result<()> {
    let tx = db.begin_transaction();
    tx.put(
        SERVER_INFO_POOL,
        import_marker_collection_id(),
        import_marker_node_id(),
        &node(encode_import_marker(ImportMarker {
            data_format: DATA_FORMAT,
            events: counts.events,
            closures: counts.closures,
        })),
    )?;
    tx.commit().context("committing the import marker")?;
    Ok(())
}

/// Make everything written so far durable. Called before the marker commits,
/// so the marker is never durable ahead of the data it vouches for.
pub(crate) fn sync_data(db: &Database) -> anyhow::Result<()> {
    for pool in [EVENT_POOL, EDGE_POOL, SERVER_INFO_POOL] {
        db.pool(pool)
            .sync_all()
            .with_context(|| format!("persisting the {pool:?} pool"))?;
    }
    Ok(())
}

fn id_list<'a>(event: &'a RoomEvent, field: &str) -> anyhow::Result<Vec<&'a [u8]>> {
    let value = if field == "prev_events" {
        &event.prev_events
    } else {
        &event.auth_events
    };
    let OwnedValue::Array(items) = value else {
        bail!("{}: `{field}` is not a list", event.event_id);
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::as_bytes)
                .ok_or_else(|| anyhow!("{}: `{field}` holds a non-string id", event.event_id))
        })
        .collect()
}

/// The verdict to store for an event: the rejection the import derived, and
/// nothing the export said about rejection, soft-fail or outlier status. Those
/// are derived from the stored events and server keys, not migrated.
/// `redacted_by` is a fact about the stored event, so it is kept.
fn verdict_of(metadata: &EventMetadata, forced: Option<u8>) -> Verdict {
    Verdict {
        rejection: forced,
        soft_fail: None,
        redacted_by: metadata
            .redacted_by
            .as_ref()
            .map(|redactor| redactor.as_bytes().to_vec()),
        // A derived rejection is stored as an outlier: the event cannot be
        // part of the room's accepted state.
        outlier: forced.is_some(),
    }
}

fn exported_rejection(metadata: &EventMetadata) -> bool {
    metadata.rejected || metadata.rejection_code.is_some()
}

/// The room's id and the hex ids of its event, auth and prev collections, as
/// `mtxdb info`/`get --collection` take them.
pub(crate) fn room_collections_line(room_id: &str) -> String {
    let room = room_id.as_bytes();
    format!(
        "room {room_id} (events 0x{}, auth 0x{}, prev 0x{})",
        hex::encode_upper(room_event_collection_id(room)),
        hex::encode_upper(room_auth_collection_id(room)),
        hex::encode_upper(room_prev_collection_id(room)),
    )
}

/// What a room's edges say about events the import cannot fully authorize.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct AuthFindings {
    /// The rejection code forced on each event, by position in the export.
    forced: Vec<Option<u8>>,
    /// Events with an `auth_events` id that is not in the export.
    pub(crate) missing_auth: Vec<String>,
    /// Events rejected because they cite a rejected auth event, in the order
    /// the rejection reached them.
    pub(crate) cascaded: Vec<String>,
    /// Events with a `prev_events` id that is not in the export. These keep
    /// their verdicts: a gap in history is not an authorization failure.
    pub(crate) missing_prev: Vec<String>,
    /// Events the source exported as rejected that pass the first auth check,
    /// by id. Their rejection is kept until the state-before-event check can
    /// decide whether the export had another reason to reject them.
    pub(crate) passes_but_exported_rejected: Vec<String>,
    /// Events the source exported as accepted that fail the auth check, with
    /// the rule that refused them.
    pub(crate) fails_but_exported_accepted: Vec<AuthFailure>,
    /// Events whose content hash does not match their contents. They are not
    /// rejected for it: the auth check ran on their redaction instead.
    pub(crate) hash_mismatched: Vec<String>,
    /// The redacted PDU to store for each event, by position in the export,
    /// when its content hash does not match. The export keeps the original.
    redacted_pdus: Vec<Option<Vec<u8>>>,
    /// One audit record per hash-mismatched event.
    pub(crate) hash_audit: Vec<HashAudit>,
    /// Key bindings corroborated by successfully verified historical events.
    pub(crate) verified_keys: Vec<VerifiedKeyObservation>,
    /// Events rejected because a usable, in-window origin key had a bad signature.
    pub(crate) signature_failures: Vec<AuthFailure>,
    /// One record per event whose signature did not verify, whatever the reason.
    pub(crate) signature_audit: Vec<SignatureAudit>,
    /// Events with no in-window origin key for the signer.
    pub(crate) signature_no_key: usize,
    /// Events without the origin, timestamp, or signature material to check.
    pub(crate) signature_no_material: usize,
    /// Events that verified against an unretired key after its cached
    /// `valid_until_ts`. That value is a refresh deadline, not the key's
    /// lifetime, so these are accepted but do not corroborate the key.
    pub(crate) signature_verified_late: usize,
    /// One record per event that verified late against an unretired key.
    pub(crate) signature_late: Vec<SignatureAudit>,
    /// Derived rejections by stored rejection code.
    pub(crate) rejection_counts: BTreeMap<u8, usize>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct KeyBinding {
    pub(crate) origin: Vec<u8>,
    pub(crate) key_id: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VerifiedKeyObservation {
    pub(crate) binding: KeyBinding,
    pub(crate) first_seen_ms: u64,
    pub(crate) last_active_ms: u64,
}

#[derive(Clone, Debug)]
struct VerificationKeyRecord {
    origin: String,
    key: mtxdb::server_keys::ServerKey,
    parsed_key: Option<VerificationKey>,
}

/// The key records visible after the keys-first phase of one import run.
#[derive(Debug, Default)]
pub(crate) struct SignatureKeyCatalog {
    records: HashMap<(String, Vec<u8>), VerificationKeyRecord>,
}

impl SignatureKeyCatalog {
    pub(crate) fn from_records(records: &[KeyRecord]) -> Self {
        let mut catalog = Self::default();
        catalog.extend_from_records(records);
        catalog
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.records.extend(other.records);
    }

    fn extend_from_records(&mut self, records: &[KeyRecord]) {
        for record in records {
            let KeyRecord::Key { origin, key } = record else {
                continue;
            };
            self.records.insert(
                (origin.to_ascii_lowercase(), key.key_id.clone()),
                VerificationKeyRecord {
                    origin: origin.clone(),
                    parsed_key: VerificationKey::try_from(key.key.as_slice()).ok(),
                    key: key.clone(),
                },
            );
        }
    }

    pub(crate) fn has_key_material(&self) -> bool {
        self.records
            .values()
            .any(|record| !record.key.key.is_empty())
    }

    pub(crate) fn read_for_room(
        store: &PackfileStorage,
        export: &RoomExport,
    ) -> anyhow::Result<Self> {
        let mut catalog = Self::default();
        let Some(room_version) = create_room_version(&export.events) else {
            return Ok(catalog);
        };
        let origins: HashSet<_> = export
            .events
            .iter()
            .filter_map(|event| event_origin(&event.pdu, room_version))
            .collect();
        for origin in origins {
            for key in server_keys::read_server_keys(store, origin.as_bytes())? {
                catalog.records.insert(
                    (origin.to_ascii_lowercase(), key.key_id.clone()),
                    VerificationKeyRecord {
                        origin: origin.to_owned(),
                        parsed_key: VerificationKey::try_from(key.key.as_slice()).ok(),
                        key,
                    },
                );
            }
        }
        Ok(catalog)
    }

    fn verify(&self, event: &RoomEvent, room_version: &str) -> SignatureOutcome {
        let Some(origin) = event_origin(&event.pdu, room_version) else {
            return SignatureOutcome::NoMaterial;
        };
        let Some(timestamp) = event
            .pdu
            .get("origin_server_ts")
            .and_then(OwnedValue::as_u64)
        else {
            return SignatureOutcome::NoMaterial;
        };
        let Some(signatures) = event.pdu.get("signatures").and_then(OwnedValue::as_object) else {
            return SignatureOutcome::NoMaterial;
        };
        let Some(origin_signatures) = signatures.get(origin).and_then(OwnedValue::as_object) else {
            return SignatureOutcome::NoMaterial;
        };

        let mut candidates = Vec::new();
        let mut unresolved = false;
        for (key_id, signature) in origin_signatures {
            let key_id_bytes = key_id.as_bytes();
            let Some(record) = self
                .records
                .get(&(origin.to_ascii_lowercase(), key_id_bytes.to_vec()))
            else {
                unresolved = true;
                continue;
            };
            let key = &record.key;
            // `expired_ts` is the inclusive hard retirement boundary. For v5+
            // rooms, `valid_until_ts` is a cache-refresh deadline: an
            // unretired key past it can still verify, but is reported late.
            // A retired key at or before `expired_ts` verifies normally. Past a
            // stale cached deadline, a failure stays NoKey rather than
            // becoming a rejection, because the snapshot cannot refresh.
            // v1-v4 ignore `valid_until_ts` entirely.
            let past_cached_deadline =
                room_version_uses_valid_until_ts(room_version) && timestamp > key.valid_until_ts;
            let late = key.expired_ts.is_none() && past_cached_deadline;
            let reject_failure = !past_cached_deadline;
            if key.key.is_empty() {
                unresolved = true;
                continue;
            }
            if key.expired_ts.is_some_and(|expired| timestamp > expired) {
                continue;
            }
            candidates.push((record, key_id.as_str(), signature, late, reject_failure));
        }
        if candidates.is_empty() {
            return SignatureOutcome::NoKey;
        }

        // Prefer an in-window signature if multiple keys are present. A
        // verified late signature is useful migration evidence, but must not
        // hide a usable in-window signature.
        candidates.sort_by_key(|(_, _, _, late, _)| *late);
        let mut rejectable_failure = None;
        let mut any_parsed_key = false;
        for (record, key_id, _, late, reject_failure) in &candidates {
            let Some(parsed_key) = record.parsed_key.as_ref() else {
                unresolved = true;
                continue;
            };
            any_parsed_key = true;
            let verifier = OneKeyVerifier {
                origin: &record.origin,
                key_id,
                key: parsed_key,
            };
            match verify_event_signatures_from_server(
                &event.pdu,
                room_version,
                &record.origin,
                &verifier,
            ) {
                Ok(()) => {
                    return if *late {
                        SignatureOutcome::VerifiedLate
                    } else {
                        SignatureOutcome::Verified(KeyBinding {
                            origin: record.origin.as_bytes().to_vec(),
                            key_id: record.key.key_id.clone(),
                        })
                    };
                }
                Err(error) if *reject_failure => rejectable_failure = Some(error),
                Err(_) => {}
            }
        }
        if !any_parsed_key {
            return if unresolved {
                SignatureOutcome::NoKey
            } else {
                SignatureOutcome::NoMaterial
            };
        }
        // A missing or malformed key is missing evidence, not proof that the
        // event is invalid: the absent key may still verify.
        if unresolved {
            return SignatureOutcome::NoKey;
        }
        if let Some(error) = rejectable_failure {
            return SignatureOutcome::Invalid(error);
        }
        // Late failures alone cannot distinguish a bad signature from a key
        // rotated under the same ID, so they do not reject the event.
        SignatureOutcome::NoKey
    }

    /// Why each of the origin's signature keys was or was not usable for the
    /// event, one line per key.
    fn explain(&self, event: &RoomEvent, room_version: &str) -> Vec<String> {
        let Some(origin) = event_origin(&event.pdu, room_version) else {
            return vec!["no origin server".to_owned()];
        };
        let timestamp = event
            .pdu
            .get("origin_server_ts")
            .and_then(OwnedValue::as_u64);
        let uses_valid_until = room_version_uses_valid_until_ts(room_version);
        let Some(origin_signatures) = event
            .pdu
            .get("signatures")
            .and_then(OwnedValue::as_object)
            .and_then(|signatures| signatures.get(origin))
            .and_then(OwnedValue::as_object)
        else {
            return vec![format!("no signatures from {origin}")];
        };
        origin_signatures
            .keys()
            .map(|key_id| {
                let Some(record) = self
                    .records
                    .get(&(origin.to_ascii_lowercase(), key_id.as_bytes().to_vec()))
                else {
                    return format!("{key_id}: not in key store");
                };
                let key = &record.key;
                if key.key.is_empty() {
                    return format!("{key_id}: no key bytes");
                }
                match timestamp {
                    Some(ts) if key.expired_ts.is_some_and(|expired| ts > expired) => format!(
                        "{key_id}: event after expired_ts {}",
                        key.expired_ts.unwrap_or_default()
                    ),
                    Some(ts)
                        if key.expired_ts.is_none()
                            && uses_valid_until
                            && ts > key.valid_until_ts =>
                    {
                        format!(
                            "{key_id}: event after valid_until_ts {}",
                            key.valid_until_ts
                        )
                    }
                    _ => format!("{key_id}: in window"),
                }
            })
            .collect()
    }
}

struct OneKeyVerifier<'a> {
    origin: &'a str,
    key_id: &'a str,
    key: &'a VerificationKey,
}

impl SignatureVerifier for OneKeyVerifier<'_> {
    fn has_key(&self, server_name: &str, key_id: &str) -> bool {
        self.origin.eq_ignore_ascii_case(server_name) && self.key_id == key_id
    }

    fn verify(
        &self,
        server_name: &str,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), String> {
        if !self.has_key(server_name, key_id) {
            return Err(format!("no public key for {server_name}/{key_id}"));
        }
        let signature: [u8; 64] = signature
            .try_into()
            .map_err(|_| String::from("signature must be 64 bytes"))?;
        self.key
            .verify(&Signature::from(signature), message)
            .map_err(|error| format!("signature verification failed: {error:?}"))
    }
}

enum SignatureOutcome {
    NoKey,
    NoMaterial,
    Verified(KeyBinding),
    /// Verified against an unretired key after its cached `valid_until_ts`.
    VerifiedLate,
    Invalid(String),
}

/// Whether a room version applies `valid_until_ts` to historical signatures.
/// Versions 1-4 ignore it; version 5 and later use it as a cache-refresh
/// deadline. Unknown version strings are treated as modern.
fn room_version_uses_valid_until_ts(room_version: &str) -> bool {
    match room_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
    {
        Some(version) => version >= 5,
        None => true,
    }
}

fn event_origin<'a>(pdu: &'a OwnedValue, room_version: &str) -> Option<&'a str> {
    if matches!(room_version, "1" | "2") {
        return pdu
            .get("event_id")?
            .as_str()?
            .strip_prefix('$')?
            .split_once(':')
            .map(|(_, server)| server);
    }
    pdu.get("sender")?
        .as_str()?
        .strip_prefix('@')?
        .split_once(':')
        .map(|(_, server)| server)
}

/// An accepted event the first auth check refuted, and why.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct AuthFailure {
    /// The event the source exported as accepted.
    pub(crate) event_id: String,
    /// Rezzy's reason for refusing it.
    pub(crate) reason: String,
}

/// How many event ids each import warning names. `MTXDB_IMPORT_WARNING_LIMIT`
/// overrides the default of 10; `all` names every id.
fn warning_id_limit() -> usize {
    const DEFAULT: usize = 10;
    match std::env::var("MTXDB_IMPORT_WARNING_LIMIT") {
        Ok(value) if value.eq_ignore_ascii_case("all") => usize::MAX,
        Ok(value) => value.parse().unwrap_or(DEFAULT),
        Err(_) => DEFAULT,
    }
}

impl AuthFindings {
    /// Human-readable warnings, empty for a clean room.
    pub(crate) fn warnings(&self) -> Vec<String> {
        let id_limit = warning_id_limit();
        let mut lines = Vec::new();
        for (count, ids, what) in [
            (
                self.missing_auth.len(),
                &self.missing_auth,
                "stored as rejected outliers: an auth event is not in the export",
            ),
            (
                self.cascaded.len(),
                &self.cascaded,
                "stored as rejected outliers: they cite a rejected auth event",
            ),
            (
                self.missing_prev.len(),
                &self.missing_prev,
                "have a prev event that is not in the export (verdicts unchanged)",
            ),
            (
                self.passes_but_exported_rejected.len(),
                &self.passes_but_exported_rejected,
                "were exported as rejected but pass the first auth check (stored accepted; the state-before-event check is not run yet)",
            ),
        ] {
            if count == 0 {
                continue;
            }
            let shown: Vec<&str> = ids.iter().take(id_limit).map(String::as_str).collect();
            let more = count.saturating_sub(shown.len());
            let tail = if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            };
            lines.push(format!("{count} events {what}: {}{tail}", shown.join(", ")));
        }
        // Most hash failures are already-redacted events; listing them is noise.
        // The audit sidecar has every row, so the log names only the ones whose
        // content the redaction would actually change.
        let differs: Vec<&str> = self
            .hash_audit
            .iter()
            .filter(|row| !row.content_matches_redaction)
            .map(|row| row.event_id.as_str())
            .collect();
        if !differs.is_empty() {
            lines.push(format!(
                "{} hash-mismatched events have content that differs from their redaction: {}",
                differs.len(),
                differs.join(", ")
            ));
        }
        if !self.fails_but_exported_accepted.is_empty() {
            let mut groups: Vec<(&str, Vec<&str>)> = Vec::new();
            for failure in &self.fails_but_exported_accepted {
                if let Some((_, ids)) = groups
                    .iter_mut()
                    .find(|(reason, _)| *reason == failure.reason)
                {
                    ids.push(&failure.event_id);
                } else {
                    groups.push((failure.reason.as_str(), vec![&failure.event_id]));
                }
            }
            let mut shown = 0;
            let mut parts = Vec::new();
            for (reason, ids) in groups {
                let take = id_limit.saturating_sub(shown).min(ids.len());
                let listed = ids
                    .iter()
                    .take(take)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ");
                let suffix = if take < ids.len() { ", ..." } else { "" };
                parts.push(format!("{reason} ({}): {listed}{suffix}", ids.len()));
                shown = shown.saturating_add(take);
            }
            let more = self.fails_but_exported_accepted.len().saturating_sub(shown);
            let tail = if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            };
            lines.push(format!(
                "{} events were exported as accepted but fail the auth check against their auth \
                 events (stored as rejected), grouped by reason: {}{tail}",
                self.fails_but_exported_accepted.len(),
                parts.join("; "),
            ));
        }
        if let Some(warning) = self.signature_failure_warning(id_limit) {
            lines.push(warning);
        }
        lines
    }

    fn signature_failure_warning(&self, listed_limit: usize) -> Option<String> {
        if self.signature_failures.is_empty() {
            return None;
        }
        let shown = self
            .signature_failures
            .iter()
            .take(listed_limit)
            .map(|failure| failure.event_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let more = self.signature_failures.len().saturating_sub(listed_limit);
        let tail = if more > 0 {
            format!(", and {more} more")
        } else {
            String::new()
        };
        Some(format!(
            "{} events have an invalid signature from an available in-window key (stored as rejected): {shown}{tail}",
            self.signature_failures.len()
        ))
    }
}

/// Work out which events cannot be authorized from the room's own edges.
///
/// An event with an auth event missing from the export is forced to
/// `MissingAuthEvent`; an event citing a forced-rejected auth event is forced
/// to `DependsOnRejectedAuthEvent`, transitively. The exported verdicts are not
/// read. The room fails as a whole when its `m.room.create` would be forced.
pub(crate) fn auth_findings(export: &RoomExport) -> anyhow::Result<AuthFindings> {
    auth_findings_seeded(export, &vec![false; export.events.len()])
}

/// [`auth_findings`] with the rejections that start the cascade supplied by
/// the caller, by position in the export.
fn auth_findings_seeded(export: &RoomExport, seeds: &[bool]) -> anyhow::Result<AuthFindings> {
    let events = &export.events;
    let index: HashMap<&[u8], usize> = events
        .iter()
        .enumerate()
        .map(|(position, event)| (event.event_id.as_bytes(), position))
        .collect();
    let mut findings = AuthFindings {
        forced: vec![None; events.len()],
        ..AuthFindings::default()
    };
    let mut rejected: Vec<bool> = seeds.to_vec();
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); events.len()];

    for (position, event) in events.iter().enumerate() {
        if id_list(event, "prev_events")?
            .iter()
            .any(|prev| !index.contains_key(prev))
        {
            findings.missing_prev.push(event.event_id.clone());
        }
        let mut missing = false;
        for auth in id_list(event, "auth_events")? {
            match index.get(auth) {
                Some(&parent) => dependents[parent].push(position),
                None => missing = true,
            }
        }
        if missing && !rejected[position] {
            findings.forced[position] = Some(RejectionCode::MissingAuthEvent.to_u8());
            rejected[position] = true;
            findings.missing_auth.push(event.event_id.clone());
        }
    }

    let mut pending: Vec<usize> = (0..events.len()).filter(|&i| rejected[i]).collect();
    while let Some(parent) = pending.pop() {
        for &child in &dependents[parent] {
            if !rejected[child] {
                findings.forced[child] = Some(RejectionCode::DependsOnRejectedAuthEvent.to_u8());
                rejected[child] = true;
                findings.cascaded.push(events[child].event_id.clone());
                pending.push(child);
            }
        }
    }

    for (position, event) in events.iter().enumerate() {
        let is_create = event.pdu.get("type").and_then(OwnedValue::as_str) == Some(M_ROOM_CREATE);
        if is_create && findings.forced[position].is_some() {
            bail!(
                "refusing to import events for {}: its m.room.create event {} cannot be \
                 authorized from this export",
                export.room_id,
                event.event_id
            );
        }
    }
    refresh_rejection_counts(&mut findings);
    Ok(findings)
}

fn refresh_rejection_counts(findings: &mut AuthFindings) {
    findings.rejection_counts.clear();
    for code in findings.forced.iter().flatten() {
        let count = findings.rejection_counts.entry(*code).or_insert(0);
        *count = count.saturating_add(1);
    }
}

/// The state an event's own `auth_events` describe, as rezzy's auth rules read
/// it: each cited event under its `(type, state_key)`.
struct AuthState<'a>(Vec<(&'a str, &'a str, &'a LeanEvent<String>)>);

impl StateProvider<String, OwnedValue, LeanEvent<String>> for AuthState<'_> {
    fn get_event(&self, event_type: &str, state_key: &str) -> Option<&LeanEvent<String>> {
        self.0
            .iter()
            .find(|(kind, key, _)| *kind == event_type && *key == state_key)
            .map(|(_, _, event)| *event)
    }
}

/// The position of the room's create event, when the export has one.
fn create_position(events: &[RoomEvent]) -> Option<usize> {
    events.iter().position(|event| {
        event.pdu.get("type").and_then(OwnedValue::as_str) == Some(M_ROOM_CREATE)
            && event.pdu.get("state_key").and_then(OwnedValue::as_str) == Some("")
    })
}

/// The room version the create event names, when the export has one.
fn create_room_version(events: &[RoomEvent]) -> Option<&str> {
    create_position(events)
        .and_then(|position| events[position].pdu.get("content"))
        .and_then(|content| content.get("room_version"))
        .and_then(OwnedValue::as_str)
}

/// The event's redaction when its content hash does not match its contents.
///
/// The spec redacts such an event before the auth rules run; it is not
/// rejected for the mismatch. `None` when the hash matches, and when the hash
/// cannot be judged: no `hashes.sha256`, a hash that cannot be computed, or a
/// redaction that cannot be produced. Those events are checked as they are.
fn redacted_for_hash(pdu: &OwnedValue, room_version: &str) -> Option<HashMismatch> {
    let expected = pdu
        .get("hashes")
        .and_then(|hashes| hashes.get("sha256"))
        .and_then(OwnedValue::as_str)?;
    // `verify_content_hash` reports a mismatch and a failed computation the
    // same way; only a computed hash that differs is a mismatch.
    let computed = rezzy::compute_content_hash(pdu, room_version).ok()?;
    if computed == expected {
        return None;
    }
    let redacted = rezzy::try_canonical_redacted_json(pdu, room_version).ok()?;
    let value = OwnedValue::parse_bytes(redacted.as_bytes()).ok()?;
    // Record whether canonical redaction leaves the PDU's content unchanged.
    let content_matches_redaction = pdu.get("content") == value.get("content");
    Some(HashMismatch {
        value,
        bytes: redacted.into_bytes(),
        stored: expected.to_owned(),
        computed,
        content_matches_redaction,
    })
}

/// An event whose content hash failed, and its canonical redaction.
struct HashMismatch {
    value: OwnedValue,
    bytes: Vec<u8>,
    stored: String,
    computed: String,
    /// Whether canonical redaction left the PDU's content unchanged.
    content_matches_redaction: bool,
}

/// Audit record for one hash-mismatched event, written to a JSONL sidecar.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct HashAudit {
    pub(crate) event_id: String,
    pub(crate) room_id: String,
    pub(crate) room_version: String,
    pub(crate) stored: String,
    pub(crate) computed: String,
    pub(crate) content_matches_redaction: bool,
    /// Signature check result when signing keys were available for this import.
    pub(crate) signature_status: Option<&'static str>,
}

impl HashAudit {
    /// Serialize one row with the CLI's JSON string escaping.
    pub(crate) fn json_line(&self) -> String {
        crate::cmd::json_compact_object(&[
            ("event_id", crate::cmd::json_string(&self.event_id)),
            ("room_id", crate::cmd::json_string(&self.room_id)),
            ("stored_hash", crate::cmd::json_string(&self.stored)),
            ("computed_hash", crate::cmd::json_string(&self.computed)),
            (
                "content_matches_redaction",
                self.content_matches_redaction.to_string(),
            ),
            ("room_version", crate::cmd::json_string(&self.room_version)),
            (
                "signature_status",
                self.signature_status
                    .map_or_else(|| "null".to_owned(), crate::cmd::json_string),
            ),
        ])
    }
}

/// Audit record for one non-plain-verified event, written to a JSONL sidecar.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SignatureAudit {
    pub(crate) event_id: String,
    pub(crate) room_id: String,
    pub(crate) origin: String,
    pub(crate) origin_server_ts: Option<u64>,
    /// `no_usable_key`, `no_signature_material`, `invalid` or `verified_late`.
    pub(crate) outcome: &'static str,
    /// The verifier's error, if any, then one line per signing key.
    pub(crate) detail: Vec<String>,
}

impl SignatureAudit {
    /// Serialize one row with the CLI's JSON string escaping.
    pub(crate) fn json_line(&self) -> String {
        let detail: Vec<String> = self
            .detail
            .iter()
            .map(|line| crate::cmd::json_string(line))
            .collect();
        crate::cmd::json_compact_object(&[
            ("event_id", crate::cmd::json_string(&self.event_id)),
            ("room_id", crate::cmd::json_string(&self.room_id)),
            ("origin", crate::cmd::json_string(&self.origin)),
            (
                "origin_server_ts",
                self.origin_server_ts
                    .map_or_else(|| "null".to_owned(), |ts| ts.to_string()),
            ),
            ("outcome", crate::cmd::json_string(self.outcome)),
            ("detail", format!("[{}]", detail.join(","))),
        ])
    }
}

/// [`auth_findings`] with the rejections derived from the events, not read
/// from the export.
///
/// Each event is checked with rezzy's auth rules against the state its own
/// `auth_events` describe (the spec's first check). The exported verdicts are
/// not an input and are never stored: they are stripped before the event is
/// built, and are compared with the result only to report disagreements.
///
/// An event that passes is stored accepted even if the export rejected it,
/// because the state-before-event check (the spec's second check) and the
/// signature checks are not run here yet. Those rejections will appear once
/// they are.
///
/// An event whose content hash does not match is checked in its redacted form
/// and listed in `hash_mismatched`; a bad hash never rejects on its own.
///
/// A v12+ event's `auth_events` no longer cite `m.room.create`, so the room's
/// create event is added to the auth state the rules read. An event the check
/// cannot judge (an unknown room version, or a PDU rezzy cannot read) is
/// stored accepted; one with an auth event missing from the export is stored
/// as a rejected outlier. Soft-fail is not derived and is not stored.
enum EventAuthOutcome {
    Unchecked,
    Passed,
    Refused { code: u8, reason: String },
}

fn judge_event_auth(
    event: &RoomEvent,
    subject: Option<&LeanEvent<String>>,
    index: &HashMap<&[u8], usize>,
    lean: &[Option<LeanEvent<String>>],
    create: Option<&LeanEvent<String>>,
    version: StateResVersion,
) -> anyhow::Result<EventAuthOutcome> {
    let Some(subject) = subject else {
        return Ok(EventAuthOutcome::Unchecked);
    };
    let is_create = event.pdu.get("type").and_then(OwnedValue::as_str) == Some(M_ROOM_CREATE)
        && event.pdu.get("state_key").and_then(OwnedValue::as_str) == Some("");
    let mut cited = Vec::new();
    for auth in id_list(event, "auth_events")? {
        match index.get(auth).and_then(|&parent| lean[parent].as_ref()) {
            Some(parent) => cited.push(parent),
            None => return Ok(EventAuthOutcome::Unchecked),
        }
    }
    let mut entries: Vec<(&str, &str, &LeanEvent<String>)> = Vec::new();
    let mut duplicate = false;
    for parent in cited {
        let Some(key) = parent.state_key.as_deref() else {
            continue;
        };
        if entries
            .iter()
            .any(|(kind, existing, _)| *kind == parent.event_type && *existing == key)
        {
            duplicate = true;
        }
        entries.push((parent.event_type.as_str(), key, parent));
    }
    if version.is_v2_1_plus() && !is_create {
        // v12+ makes the create event implicit by removing it from
        // `auth_events`. It is still part of the state the rules read, so
        // supply it for every other event. If the create event is
        // unreadable, leave the event unchecked so its exported verdict
        // stands.
        let Some(create) = create else {
            return Ok(EventAuthOutcome::Unchecked);
        };
        if !entries
            .iter()
            .any(|(kind, key, _)| *kind == M_ROOM_CREATE && key.is_empty())
        {
            entries.push((M_ROOM_CREATE, "", create));
        }
    }
    if duplicate {
        return Ok(EventAuthOutcome::Refused {
            code: RejectionCode::DuplicateAuthEventKey.to_u8(),
            reason: "auth_events contains a duplicate (type, state_key) pair".to_owned(),
        });
    }
    match check_auth(subject, &AuthState(entries), version, None) {
        Ok(()) => Ok(EventAuthOutcome::Passed),
        Err(error) => Ok(EventAuthOutcome::Refused {
            code: RejectionCode::AuthCheckFailed.to_u8(),
            reason: error.to_string(),
        }),
    }
}

/// The result of running the first auth check over every event.
struct AuthJudgement {
    /// Whether the check could judge the event.
    checked: Vec<bool>,
    /// The rejection code when the check judged and refused the event.
    refused: Vec<Option<u8>>,
    /// The rejection reason for a refused event.
    reason: Vec<Option<String>>,
}

/// The canonical redaction of every event whose content hash fails.
fn redactions_for_export(events: &[RoomEvent], room_version: &str) -> Vec<Option<HashMismatch>> {
    events
        .iter()
        .map(|event| redacted_for_hash(&event.pdu, room_version))
        .collect()
}

/// The auth-check input for each event, using its redacted form on mismatch.
fn lean_redacted_events(
    events: &[RoomEvent],
    redacted: &[Option<HashMismatch>],
    room_id: &str,
) -> Vec<Option<LeanEvent<String>>> {
    events
        .iter()
        .zip(redacted)
        .map(|(event, redacted)| {
            let value = match redacted {
                Some(mismatch) => event.unflagged_from(&mismatch.value, room_id),
                None => event.unflagged(room_id),
            };
            lean_event(&value, &event.event_id)
        })
        .collect()
}

fn judge_events(
    events: &[RoomEvent],
    lean: &[Option<LeanEvent<String>>],
    index: &HashMap<&[u8], usize>,
    create: Option<&LeanEvent<String>>,
    version: StateResVersion,
) -> anyhow::Result<AuthJudgement> {
    let mut checked = vec![false; events.len()];
    let mut refused: Vec<Option<u8>> = vec![None; events.len()];
    let mut reasons: Vec<Option<String>> = vec![None; events.len()];
    for (position, event) in events.iter().enumerate() {
        match judge_event_auth(event, lean[position].as_ref(), index, lean, create, version)? {
            EventAuthOutcome::Unchecked => {}
            EventAuthOutcome::Passed => checked[position] = true,
            EventAuthOutcome::Refused { code, reason } => {
                checked[position] = true;
                refused[position] = Some(code);
                reasons[position] = Some(reason);
            }
        }
    }
    Ok(AuthJudgement {
        checked,
        refused,
        reason: reasons,
    })
}

fn audit_hash_mismatches(
    events: &[RoomEvent],
    redacted: &[Option<HashMismatch>],
    room_id: &str,
    room_version: &str,
) -> Vec<HashAudit> {
    events
        .iter()
        .zip(redacted)
        .filter_map(|(event, mismatch)| {
            mismatch.as_ref().map(|m| HashAudit {
                event_id: event.event_id.clone(),
                room_id: room_id.to_owned(),
                room_version: room_version.to_owned(),
                stored: m.stored.clone(),
                computed: m.computed.clone(),
                content_matches_redaction: m.content_matches_redaction,
                signature_status: None,
            })
        })
        .collect()
}

fn apply_judgement(
    findings: &mut AuthFindings,
    events: &[RoomEvent],
    judgement: &AuthJudgement,
    exported: &[bool],
) {
    for (position, event) in events.iter().enumerate() {
        if !judgement.checked[position] {
            continue;
        }
        if let Some(code) = judgement.refused[position] {
            // The seed kept the cascade from assigning a code.
            findings.forced[position] = Some(code);
            if !exported[position] {
                findings.fails_but_exported_accepted.push(AuthFailure {
                    event_id: event.event_id.clone(),
                    reason: judgement.reason[position]
                        .clone()
                        .unwrap_or_else(|| "unknown auth-check failure".to_owned()),
                });
            }
        } else if exported[position] {
            // Passing the first check does not rule out a state-before-event
            // rejection, which this pass cannot see yet. The event is stored
            // accepted; the disagreement is reported.
            findings
                .passes_but_exported_rejected
                .push(event.event_id.clone());
        }
    }
}

fn ensure_create_authorized(
    export: &RoomExport,
    events: &[RoomEvent],
    findings: &AuthFindings,
) -> anyhow::Result<()> {
    for (position, event) in events.iter().enumerate() {
        let is_create = event.pdu.get("type").and_then(OwnedValue::as_str) == Some(M_ROOM_CREATE)
            && event.pdu.get("state_key").and_then(OwnedValue::as_str) == Some("");
        if is_create && findings.forced[position].is_some() {
            bail!(
                "refusing to import events for {}: its m.room.create event {} cannot be \
                 authorized from this export",
                export.room_id,
                event.event_id
            );
        }
    }
    Ok(())
}

fn apply_signature_outcomes(
    findings: &mut AuthFindings,
    events: &[RoomEvent],
    outcomes: Vec<SignatureOutcome>,
    keys: &SignatureKeyCatalog,
    room_id: &str,
    room_version: &str,
) {
    let mut verified: HashMap<KeyBinding, (u64, u64)> = HashMap::new();
    let mut no_key = 0_usize;
    let mut no_material = 0_usize;
    let mut verified_late = 0_usize;
    let audit_positions: HashMap<_, _> = findings
        .hash_audit
        .iter()
        .enumerate()
        .map(|(position, audit)| (audit.event_id.clone(), position))
        .collect();
    for (position, outcome) in outcomes.iter().enumerate() {
        let status = match outcome {
            SignatureOutcome::NoKey => "no_usable_key",
            SignatureOutcome::NoMaterial => "no_signature_material",
            SignatureOutcome::Verified(_) => "verified",
            SignatureOutcome::VerifiedLate => "verified_late",
            SignatureOutcome::Invalid(_) => "invalid",
        };
        if let Some(&audit_position) = audit_positions.get(&events[position].event_id) {
            findings.hash_audit[audit_position].signature_status = Some(status);
        }
        if !matches!(outcome, SignatureOutcome::Verified(_)) {
            let event = &events[position];
            let mut detail = keys.explain(event, room_version);
            if let SignatureOutcome::Invalid(reason) = outcome {
                detail.insert(0, reason.clone());
            }
            let row = SignatureAudit {
                event_id: event.event_id.clone(),
                room_id: room_id.to_owned(),
                origin: event_origin(&event.pdu, room_version)
                    .unwrap_or_default()
                    .to_owned(),
                origin_server_ts: event
                    .pdu
                    .get("origin_server_ts")
                    .and_then(OwnedValue::as_u64),
                outcome: status,
                detail,
            };
            if matches!(outcome, SignatureOutcome::VerifiedLate) {
                findings.signature_late.push(row);
            } else {
                findings.signature_audit.push(row);
            }
        }
    }
    for (position, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            SignatureOutcome::NoKey => no_key = no_key.saturating_add(1),
            SignatureOutcome::NoMaterial => no_material = no_material.saturating_add(1),
            SignatureOutcome::VerifiedLate => verified_late = verified_late.saturating_add(1),
            SignatureOutcome::Verified(binding) => {
                let timestamp = events[position]
                    .pdu
                    .get("origin_server_ts")
                    .and_then(OwnedValue::as_u64)
                    .expect("verified signatures require origin_server_ts");
                verified
                    .entry(binding)
                    .and_modify(|(first, last)| {
                        *first = (*first).min(timestamp);
                        *last = (*last).max(timestamp);
                    })
                    .or_insert((timestamp, timestamp));
            }
            SignatureOutcome::Invalid(reason) => {
                findings.forced[position] =
                    Some(RejectionCode::SignatureVerificationFailed.to_u8());
                findings.signature_failures.push(AuthFailure {
                    event_id: events[position].event_id.clone(),
                    reason,
                });
            }
        }
    }
    findings.signature_no_key = no_key;
    findings.signature_no_material = no_material;
    findings.signature_verified_late = verified_late;
    findings.verified_keys = verified
        .into_iter()
        .map(
            |(binding, (first_seen_ms, last_active_ms))| VerifiedKeyObservation {
                binding,
                first_seen_ms,
                last_active_ms,
            },
        )
        .collect();
}

#[cfg(test)]
pub(crate) fn derived_auth_findings(export: &RoomExport) -> anyhow::Result<AuthFindings> {
    derived_auth_findings_with_keys(export, None)
}

pub(crate) fn derived_auth_findings_with_keys(
    export: &RoomExport,
    keys: Option<&SignatureKeyCatalog>,
) -> anyhow::Result<AuthFindings> {
    let events = &export.events;
    let Some(room_version) = create_room_version(events) else {
        return auth_findings(export);
    };
    let Some(version) = StateResVersion::from_room_version(room_version) else {
        return auth_findings(export);
    };
    let index: HashMap<&[u8], usize> = events
        .iter()
        .enumerate()
        .map(|(position, event)| (event.event_id.as_bytes(), position))
        .collect();
    // An event whose content hash fails is judged in its redacted form.
    let redacted = redactions_for_export(events, room_version);
    let lean = lean_redacted_events(events, &redacted, &export.room_id);
    let create = create_position(events).and_then(|position| lean[position].as_ref());
    let exported: Vec<bool> = events
        .iter()
        .map(|event| exported_rejection(&event.metadata))
        .collect();
    let judgement = judge_events(events, &lean, &index, create, version)?;

    // Only a derived refusal seeds the cascade. The exported verdicts are
    // compared below for the report and are never stored.
    let signature_outcomes = keys.map(|keys| {
        events
            .iter()
            .map(|event| keys.verify(event, room_version))
            .collect::<Vec<_>>()
    });
    let seeds: Vec<bool> = judgement
        .refused
        .iter()
        .enumerate()
        .map(|(position, refusal)| {
            refusal.is_some()
                || matches!(
                    signature_outcomes
                        .as_ref()
                        .and_then(|outcomes| outcomes.get(position)),
                    Some(SignatureOutcome::Invalid(_))
                )
        })
        .collect();
    let mut findings = auth_findings_seeded(export, &seeds)?;
    findings.hash_mismatched = events
        .iter()
        .zip(&redacted)
        .filter(|(_, redacted)| redacted.is_some())
        .map(|(event, _)| event.event_id.clone())
        .collect();
    findings.hash_audit = audit_hash_mismatches(events, &redacted, &export.room_id, room_version);
    findings.redacted_pdus = redacted
        .into_iter()
        .map(|redacted| redacted.map(|m| m.bytes))
        .collect();
    apply_judgement(&mut findings, events, &judgement, &exported);
    if let (Some(outcomes), Some(keys)) = (signature_outcomes, keys) {
        apply_signature_outcomes(
            &mut findings,
            events,
            outcomes,
            keys,
            &export.room_id,
            room_version,
        );
    }
    refresh_rejection_counts(&mut findings);
    ensure_create_authorized(export, events, &findings)?;
    Ok(findings)
}

/// Bytes an event's records will stage, with margin.
fn estimate(
    pdu_len: usize,
    event: &RoomEvent,
    room_id: &str,
    verdict: &Verdict,
    edges: usize,
) -> usize {
    let verdict_bytes = if verdict.is_empty() {
        0
    } else {
        RECORD_OVERHEAD
            .saturating_add(event.event_id.len())
            .saturating_add(verdict.redacted_by.as_ref().map_or(0, Vec::len))
    };
    pdu_len
        .saturating_add(event.event_id.len().saturating_mul(2))
        .saturating_add(room_id.len())
        .saturating_add(edges)
        .saturating_add(RECORD_OVERHEAD.saturating_mul(4))
        .saturating_add(verdict_bytes)
}

/// Write a room's events, edges, locators and verdicts in capped batches,
/// also returning what the room's edges said about events that could not be
/// authorized, for the caller to report.
///
/// All of one event's records share a transaction. Returns how many event
/// records were written. The marker is the caller's to commit.
///
/// Each rejection is derived by running the auth rules over the room's events
/// (see [`derived_auth_findings`]); the export's rejections are not trusted.
#[cfg(test)]
pub(crate) fn write_room_with_findings(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
) -> anyhow::Result<(ImportCounts, AuthFindings)> {
    write_room_with_signature_keys(db, export, batch_bytes, None)
}

pub(crate) fn write_room_with_signature_keys(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
    keys: Option<&SignatureKeyCatalog>,
) -> anyhow::Result<(ImportCounts, AuthFindings)> {
    let findings = derived_auth_findings_with_keys(export, keys)?;
    let plan = closure_plan(export)?;
    let room = export.room_id.as_bytes();
    let event_collection = room_event_collection_id(room);
    let prev_collection = room_prev_collection_id(room);
    let auth_collection = room_auth_collection_id(room);
    let version_collection = room_version_collection_id(room);
    let version_node = room_version_node_id();

    let mut written = 0_u64;
    let mut staged_bytes = 0_usize;
    // Event ids staged by the open batch, by record id: two different ids
    // sharing a record id must not shadow each other. Committed batches are
    // covered by reading the locator back through the transaction.
    let mut batch_ids: HashMap<[u8; 16], &str> = HashMap::new();
    let mut tx = db.begin_transaction();
    staged_bytes = staged_bytes.saturating_add(stage_room_version(
        db,
        &mut tx,
        &export.room_id,
        &export.events,
        version_collection,
        version_node,
    )?);

    for (position, event) in export.events.iter().enumerate() {
        let event_id = event.event_id.as_bytes();
        let record = event_record_id(event_id);
        let prev = id_list(event, "prev_events")?;
        let auth = id_list(event, "auth_events")?;
        let prev_bytes = encode_edges(&prev)?;
        let auth_bytes = encode_edges(&auth)?;
        let verdict = verdict_of(&event.metadata, findings.forced[position]);
        // A mismatched content hash means the effective event is its redaction;
        // the export keeps the original bytes.
        let stored: &[u8] = findings
            .redacted_pdus
            .get(position)
            .and_then(Option::as_deref)
            .unwrap_or(&event.pdu_bytes);
        let size = estimate(
            stored.len(),
            event,
            &export.room_id,
            &verdict,
            prev_bytes.len().saturating_add(auth_bytes.len()),
        );
        if size > batch_bytes {
            bail!(
                "{}: about {size} bytes of records exceed the {batch_bytes}-byte batch limit",
                event.event_id
            );
        }
        if staged_bytes.saturating_add(size) > batch_bytes && staged_bytes > 0 {
            tx.commit().context("committing an import batch")?;
            reclaim_journal(db)?;
            tx = db.begin_transaction();
            staged_bytes = 0;
            batch_ids.clear();
        }

        if let Some(other) = batch_ids.insert(record, &event.event_id) {
            if other != event.event_id {
                bail!(
                    "events {other} and {} share a 128-bit record id; refusing to shadow one",
                    event.event_id
                );
            }
        }
        let locator_collection = event_locator_collection_id(&record);
        check_locator_collision(&tx, record, &event.event_id)?;

        tx.put(EVENT_POOL, event_collection, record, &node(stored.to_vec()))?;
        tx.put(
            EVENT_POOL,
            locator_collection,
            record,
            &node(encode_event_locator(event_id, room)?),
        )?;
        tx.put(EDGE_POOL, prev_collection, record, &node(prev_bytes))?;
        tx.put(EDGE_POOL, auth_collection, record, &node(auth_bytes))?;
        if !verdict.is_empty() {
            tx.put(
                EVENT_POOL,
                event_verdict_collection_id(&record),
                record,
                &node(encode_verdict(event_id, &verdict)?),
            )?;
        }
        staged_bytes = staged_bytes.saturating_add(size);
        written = written.saturating_add(1);
    }
    tx.commit().context("committing an import batch")?;
    reclaim_journal(db)?;
    let closures = write_auth_closures(db, export, plan, batch_bytes)?;
    Ok((
        ImportCounts {
            events: written,
            closures,
        },
        findings,
    ))
}

/// One room's auth short ids and closures, staged into capped batches.
struct ClosureWriter<'a> {
    db: &'a Database,
    room: &'a str,
    collection: [u8; 16],
    batch_bytes: usize,
    counter: u32,
}

impl ClosureWriter<'_> {
    /// The short id for `id`: an existing one is reused, a new one advances the
    /// counter and stages its forward and reverse records. Returns the bytes
    /// the new records stage.
    fn short_for(
        &mut self,
        tx: &mtxdb::DatabaseTransaction<'_>,
        id: &[u8],
    ) -> anyhow::Result<(u32, usize)> {
        let forward = tx
            .get(EDGE_POOL, &self.collection, &[auth_forward_record_id(id)])?
            .into_iter()
            .next()
            .flatten();
        let mut staged = 0_usize;
        let mut found = None;
        if let Some(data) = forward {
            let (short, stored) = decode_auth_forward(&data.bytes)?;
            found = (stored == id).then_some(short);
        }
        let short = if let Some(short) = found {
            short
        } else {
            self.counter = self
                .counter
                .checked_add(1)
                .ok_or_else(|| anyhow!("{}: room short ids exhausted", self.room))?;
            let short = self.counter;
            tx.put(
                EDGE_POOL,
                self.collection,
                auth_forward_record_id(id),
                &node(encode_auth_forward(short, id)),
            )?;
            tx.put(
                EDGE_POOL,
                self.collection,
                auth_reverse_record_id(short),
                &node(id.to_vec()),
            )?;
            staged = id
                .len()
                .saturating_mul(2)
                .saturating_add(RECORD_OVERHEAD.saturating_mul(2));
            short
        };
        Ok((short, staged))
    }

    /// Stage the short-id counter and commit the batch. The counter goes in
    /// every batch so it is never behind the shorts a committed batch holds.
    fn commit(&self, tx: &mtxdb::DatabaseTransaction<'_>) -> anyhow::Result<()> {
        tx.put(
            EDGE_POOL,
            self.collection,
            auth_counter_record_id(),
            &node(self.counter.to_be_bytes().to_vec()),
        )?;
        tx.commit().context("committing auth closures")?;
        reclaim_journal(self.db)?;
        Ok(())
    }

    /// Stage one event: its short id, its auth events' shorts, and its closure
    /// when every auth event already has one. Returns the closure's short id if
    /// it wrote one, and the bytes staged.
    fn stage_event(
        &mut self,
        tx: &mtxdb::DatabaseTransaction<'_>,
        event_id: &[u8],
        auth: &[(&[u8], bool)],
    ) -> anyhow::Result<(bool, usize)> {
        let (own, mut staged) = self.short_for(tx, event_id)?;
        let mut closure = Bitmap::new();
        let mut complete = true;
        for &(id, has_closure) in auth {
            let (short, bytes) = self.short_for(tx, id)?;
            staged = staged.saturating_add(bytes);
            closure.insert(short);
            if !has_closure {
                complete = false;
                continue;
            }
            let stored = tx
                .get(
                    EDGE_POOL,
                    &self.collection,
                    &[auth_closure_record_id(short)],
                )?
                .into_iter()
                .next()
                .flatten()
                .ok_or_else(|| anyhow!("{}: a closure is missing after staging", self.room))?;
            closure |= &Bitmap::decode(&stored.bytes)
                .map_err(|e| anyhow!("{}: auth closure bitmap: {e}", self.room))?;
        }
        if !complete {
            return Ok((false, staged));
        }
        let encoded = closure.encode();
        staged = staged
            .saturating_add(encoded.len())
            .saturating_add(RECORD_OVERHEAD);
        tx.put(
            EDGE_POOL,
            self.collection,
            auth_closure_record_id(own),
            &node(encoded),
        )?;
        Ok((true, staged))
    }
}

/// Auth events before the events citing them, over the edges the export
/// contains. `None` for a cycle, with the position of an event stuck on it.
fn parents_first(auth: &[Vec<&[u8]>], index: &HashMap<&[u8], usize>) -> Result<Vec<usize>, usize> {
    let mut waiting: Vec<usize> = auth
        .iter()
        .map(|ids| ids.iter().filter(|id| index.contains_key(**id)).count())
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); auth.len()];
    for (child, ids) in auth.iter().enumerate() {
        for id in ids {
            if let Some(&parent) = index.get(*id) {
                children[parent].push(child);
            }
        }
    }
    let mut order: Vec<usize> = (0..auth.len()).filter(|&i| waiting[i] == 0).collect();
    let mut next = 0_usize;
    while let Some(&current) = order.get(next) {
        next = next.saturating_add(1);
        for &child in &children[current] {
            waiting[child] = waiting[child].saturating_sub(1);
            if waiting[child] == 0 {
                order.push(child);
            }
        }
    }
    if order.len() == auth.len() {
        Ok(order)
    } else {
        Err((0..auth.len()).find(|&i| waiting[i] > 0).unwrap_or(0))
    }
}

/// Each event's distinct auth ids, the position of every event id, and the
/// parents-first visiting order.
struct ClosurePlan<'a> {
    index: HashMap<&'a [u8], usize>,
    auth: Vec<Vec<&'a [u8]>>,
    order: Vec<usize>,
}

/// Plan a room's closures, failing on an auth cycle or a repeated event id.
/// Run before any event is written, so a refused room leaves nothing behind.
fn closure_plan(export: &RoomExport) -> anyhow::Result<ClosurePlan<'_>> {
    let events = &export.events;
    let mut index: HashMap<&[u8], usize> = HashMap::with_capacity(events.len());
    for (position, event) in events.iter().enumerate() {
        let id = event.event_id.as_bytes();
        if index.insert(id, position).is_some() {
            return Err(anyhow!(
                "{}: event {} appears more than once; refusing to import the room",
                export.room_id,
                event.event_id
            ));
        }
    }
    let mut auth: Vec<Vec<&[u8]>> = Vec::with_capacity(events.len());
    for event in events {
        let mut ids = id_list(event, "auth_events")?;
        let mut seen = std::collections::HashSet::new();
        ids.retain(|id| seen.insert(*id));
        auth.push(ids);
    }
    let order = parents_first(&auth, &index).map_err(|stuck| {
        anyhow!(
            "{}: auth events form a cycle (through {}); refusing to import the room",
            export.room_id,
            events[stuck].event_id
        )
    })?;
    Ok(ClosurePlan { index, auth, order })
}

/// Write the room's auth short ids and closure bitmaps in the server's layout.
///
/// Events are visited parents-first. Each event and each cited auth event gets
/// a short id (existing ones are reused, so a re-import keeps them), and an
/// event's closure is the union of its auth events' shorts and closures, written
/// only when every auth event has a closure. An event citing an auth event
/// outside the export therefore gets none, and neither do its dependents; the
/// server falls back to walking edges for them. An auth cycle fails the room.
fn write_auth_closures(
    db: &Database,
    export: &RoomExport,
    plan: ClosurePlan<'_>,
    batch_bytes: usize,
) -> anyhow::Result<u64> {
    let events = &export.events;
    let ClosurePlan { index, auth, order } = plan;

    let collection = room_auth_collection_id(export.room_id.as_bytes());
    let mut tx = db.begin_transaction();
    let counter = tx
        .get(EDGE_POOL, &collection, &[auth_counter_record_id()])?
        .into_iter()
        .next()
        .flatten()
        .map_or(Ok(0), |data| decode_auth_counter(&data.bytes))?;
    let mut writer = ClosureWriter {
        db,
        room: &export.room_id,
        collection,
        batch_bytes,
        counter,
    };
    let mut has_closure = vec![false; events.len()];
    let mut staged_bytes = 0_usize;
    for &position in &order {
        let cited: Vec<(&[u8], bool)> = auth[position]
            .iter()
            .map(|id| {
                (
                    *id,
                    index.get(*id).is_some_and(|&parent| has_closure[parent]),
                )
            })
            .collect();
        let (wrote, staged) =
            writer.stage_event(&tx, events[position].event_id.as_bytes(), &cited)?;
        has_closure[position] = wrote;
        staged_bytes = staged_bytes.saturating_add(staged);
        if staged_bytes > writer.batch_bytes {
            writer.commit(&tx)?;
            tx = writer.db.begin_transaction();
            staged_bytes = 0;
        }
    }
    writer.commit(&tx)?;
    Ok(u64::try_from(has_closure.iter().filter(|wrote| **wrote).count()).unwrap_or(u64::MAX))
}

/// Write a keys export in one transaction: a handful of records, far under
/// the batch limit.
pub(crate) fn write_keys(db: &Database, records: &[KeyRecord]) -> anyhow::Result<()> {
    let tx = db.begin_transaction();
    for record in records {
        match record {
            KeyRecord::Key { origin, key } => {
                server_keys::put_server_key(&tx, origin.as_bytes(), key)?;
            }
            KeyRecord::Notary { origin, payload } => {
                server_keys::put_notary_payload(&tx, origin.as_bytes(), payload)?;
            }
        }
    }
    tx.commit().context("committing the signing keys")?;
    Ok(())
}

#[cfg(test)]
mod auth_findings_tests {
    use bytes::Bytes;

    use super::*;

    fn event(id: &str, kind: &str, prevs: &[&str], auths: &[&str]) -> RoomEvent {
        let ids = |ids: &[&str]| {
            OwnedValue::Array(
                ids.iter()
                    .map(|id| OwnedValue::String((*id).to_owned()))
                    .collect(),
            )
        };
        let mut fields = std::collections::BTreeMap::new();
        fields.insert("type".to_owned(), OwnedValue::String(kind.to_owned()));
        if kind == M_ROOM_CREATE {
            // The auth rules authorize a create event by its sender alone, so
            // it needs a valid one to pass the check.
            fields.insert(
                "sender".to_owned(),
                OwnedValue::String("@alice:example.org".to_owned()),
            );
            fields.insert("state_key".to_owned(), OwnedValue::String(String::new()));
            let mut content = std::collections::BTreeMap::new();
            content.insert(
                "room_version".to_owned(),
                OwnedValue::String("10".to_owned()),
            );
            fields.insert("content".to_owned(), OwnedValue::Object(content));
        }
        RoomEvent {
            event_id: id.to_owned(),
            metadata: EventMetadata::default(),
            prev_events: ids(prevs),
            auth_events: ids(auths),
            pdu: OwnedValue::Object(fields),
            pdu_bytes: Bytes::new(),
        }
    }

    fn export(events: Vec<RoomEvent>) -> RoomExport {
        RoomExport {
            server: "example.org".to_owned(),
            room_id: "!room:example.org".to_owned(),
            events,
        }
    }

    fn forced(findings: &AuthFindings, position: usize) -> Option<u8> {
        findings.forced[position]
    }

    const MISSING_AUTH: u8 = 4;
    const DEPENDS_ON_REJECTED: u8 = 2;

    #[test]
    fn a_complete_room_has_no_findings() {
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$a", "m.room.message", &["$create"], &["$create"]),
        ]))
        .unwrap();
        assert_eq!(findings.forced, vec![None, None]);
        assert!(findings.warnings().is_empty());
    }

    #[test]
    fn a_missing_auth_event_forces_a_retryable_rejection() {
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$a", "m.room.message", &["$create"], &["$create", "$gone"]),
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 0), None);
        assert_eq!(forced(&findings, 1), Some(MISSING_AUTH));
        assert_eq!(findings.missing_auth, ["$a"]);
        assert!(findings.cascaded.is_empty());
        assert_eq!(findings.rejection_counts.get(&MISSING_AUTH), Some(&1));
    }

    #[test]
    fn rejection_cascades_through_auth_edges_in_any_export_order() {
        // The chain is listed child-first: the cascade does not rely on
        // dependency order in the file.
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$grandchild", "m.room.message", &["$create"], &["$child"]),
            event("$child", "m.room.message", &["$create"], &["$bad"]),
            event("$bad", "m.room.message", &["$create"], &["$gone"]),
            event("$unrelated", "m.room.message", &["$create"], &["$create"]),
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 3), Some(MISSING_AUTH));
        assert_eq!(forced(&findings, 2), Some(DEPENDS_ON_REJECTED));
        assert_eq!(forced(&findings, 1), Some(DEPENDS_ON_REJECTED));
        assert_eq!(forced(&findings, 4), None);
        assert_eq!(findings.missing_auth, ["$bad"]);
        assert_eq!(findings.cascaded, ["$child", "$grandchild"]);
        assert_eq!(findings.rejection_counts.get(&MISSING_AUTH), Some(&1));
        assert_eq!(
            findings.rejection_counts.get(&DEPENDS_ON_REJECTED),
            Some(&2)
        );
    }

    #[test]
    fn an_exported_rejection_is_not_read_and_does_not_cascade() {
        let mut rejected = event("$bad", "m.room.message", &["$create"], &["$create"]);
        rejected.metadata.rejected = true;
        rejected.metadata.rejection_code = Some(7);
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            rejected,
            event("$child", "m.room.message", &["$create"], &["$bad"]),
        ]))
        .unwrap();
        assert_eq!(findings.forced, vec![None, None, None]);
        assert!(findings.cascaded.is_empty());
    }

    #[test]
    fn a_missing_prev_event_is_a_warning_only() {
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$a", "m.room.message", &["$gone"], &["$create"]),
        ]))
        .unwrap();
        assert_eq!(findings.forced, vec![None, None]);
        assert_eq!(findings.missing_prev, ["$a"]);
        assert_eq!(findings.warnings().len(), 1);
    }

    #[test]
    fn a_create_event_that_cannot_be_authorized_fails_the_room() {
        let error = auth_findings(&export(vec![event(
            "$create",
            "m.room.create",
            &[],
            &["$gone"],
        )]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("m.room.create"), "{error}");
    }

    fn with_field(mut event: RoomEvent, key: &str, value: &str) -> RoomEvent {
        if let OwnedValue::Object(fields) = &mut event.pdu {
            fields.insert(key.to_owned(), OwnedValue::String(value.to_owned()));
        }
        event
    }

    fn create_with_version(version: &str) -> RoomEvent {
        let mut create = event("$create", "m.room.create", &[], &[]);
        if let OwnedValue::Object(fields) = &mut create.pdu {
            let mut content = std::collections::BTreeMap::new();
            content.insert(
                "room_version".to_owned(),
                OwnedValue::String(version.to_owned()),
            );
            fields.insert("content".to_owned(), OwnedValue::Object(content));
        }
        create
    }

    fn member(id: &str, user: &str, auths: &[&str]) -> RoomEvent {
        let mut event = event(id, "m.room.member", &["$create"], auths);
        if let OwnedValue::Object(fields) = &mut event.pdu {
            fields.insert("sender".to_owned(), OwnedValue::String(user.to_owned()));
            fields.insert("state_key".to_owned(), OwnedValue::String(user.to_owned()));
            let mut content = std::collections::BTreeMap::new();
            content.insert(
                "membership".to_owned(),
                OwnedValue::String("join".to_owned()),
            );
            fields.insert("content".to_owned(), OwnedValue::Object(content));
        }
        event
    }

    fn message(auths: &[&str]) -> RoomEvent {
        with_field(
            with_field(
                event("$msg", "m.room.message", &["$join"], auths),
                "sender",
                "@alice:example.org",
            ),
            "room_id",
            "!room:example.org",
        )
    }

    #[test]
    fn an_exported_accept_that_fails_the_auth_check_is_stored_rejected() {
        // The sender never joined, so the message cannot be authorized by an
        // auth state that holds only the create event, whatever was exported.
        let message = with_field(
            with_field(
                event("$msg", "m.room.message", &["$create"], &["$create"]),
                "sender",
                "@stranger:example.org",
            ),
            "room_id",
            "!room:example.org",
        );
        let findings = derived_auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            message,
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 1), Some(7));
        assert_eq!(findings.rejection_counts.get(&7), Some(&1));
        assert_eq!(findings.fails_but_exported_accepted.len(), 1);
        let failure = &findings.fails_but_exported_accepted[0];
        assert_eq!(failure.event_id, "$msg");
        assert!(failure.reason.contains("not joined"), "{}", failure.reason);
        assert!(
            findings
                .warnings()
                .iter()
                .any(|warning| warning.contains("not joined")),
            "the warning must explain why the event failed"
        );
    }

    #[test]
    fn an_exported_rejection_that_passes_the_first_check_is_reported_not_stored() {
        let mut rejected = message(&["$create", "$join"]);
        rejected.metadata.rejected = true;
        rejected.metadata.rejection_code = Some(7);
        let findings = derived_auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            rejected,
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 2), None);
        assert_eq!(findings.passes_but_exported_rejected, ["$msg"]);
        assert!(findings.fails_but_exported_accepted.is_empty());
    }

    #[test]
    fn two_auth_events_for_one_state_key_are_a_duplicate_rejection() {
        let findings = derived_auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            member("$join2", "@alice:example.org", &["$create"]),
            message(&["$create", "$join", "$join2"]),
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 3), Some(5));
        assert_eq!(findings.fails_but_exported_accepted.len(), 1);
        let failure = &findings.fails_but_exported_accepted[0];
        assert_eq!(failure.event_id, "$msg");
        assert!(failure.reason.contains("duplicate"), "{}", failure.reason);
    }

    fn stored_rejection(db: &Database, id: &str) -> Option<u8> {
        let record = event_record_id(id.as_bytes());
        read_pool(db, EVENT_POOL, event_verdict_collection_id(&record), record)
            .and_then(|bytes| mtxdb::interop::decode_verdict(&bytes).unwrap().1.rejection)
    }

    #[test]
    fn writing_a_room_stores_derived_verdicts_not_exported_ones() {
        let (db, dir) = db();
        let stranger = with_field(
            with_field(
                event("$msg", "m.room.message", &["$create"], &["$create"]),
                "sender",
                "@stranger:example.org",
            ),
            "room_id",
            "!room:example.org",
        );
        // Exported as accepted, but its auth state is just the create event.
        let room = export(vec![event("$create", "m.room.create", &[], &[]), stranger]);
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        assert_eq!(stored_rejection(&db, "$msg"), Some(7));
        assert_eq!(stored_rejection(&db, "$create"), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn writing_does_not_store_an_exported_rejection() {
        let (db, dir) = db();
        let mut rejected = message(&["$create", "$join"]);
        rejected.metadata.rejected = true;
        rejected.metadata.rejection_code = Some(7);
        let room = export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            rejected,
        ]);
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        assert_eq!(stored_rejection(&db, "$msg"), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_v12_room_judges_with_its_implicit_create_event() {
        // v12 events do not cite m.room.create. Without supplying it from the
        // room's create event the check would report MissingCreate; with it,
        // this accepted-by-the-check event is judged and its exported
        // rejection is reported as a disagreement.
        let mut rejected = message(&["$join"]);
        rejected.metadata.rejected = true;
        rejected.metadata.rejection_code = Some(7);
        let findings = derived_auth_findings(&export(vec![
            create_with_version("12"),
            member("$join", "@alice:example.org", &[]),
            rejected,
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 2), None);
        assert_eq!(findings.passes_but_exported_rejected, ["$msg"]);
        assert!(findings.fails_but_exported_accepted.is_empty());
    }

    #[test]
    fn a_historical_uppercase_create_sender_is_not_rejected() {
        // Existing rooms may have uppercase localparts. The create rule must
        // accept them through rezzy's historical MXID validator.
        let create = with_field(create_with_version("10"), "sender", "@Milan:tchncs.de");
        let findings = derived_auth_findings(&export(vec![create])).unwrap();
        assert_eq!(forced(&findings, 0), None);
        assert!(findings.fails_but_exported_accepted.is_empty());
    }

    #[test]
    fn a_derived_create_failure_fails_the_room() {
        let create = with_field(create_with_version("10"), "sender", "not-an-mxid");
        let error = derived_auth_findings(&export(vec![create]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("m.room.create"), "{error}");
    }

    /// `event` with a `hashes.sha256` of `hash`, or its true content hash when
    /// `hash` is `None`.
    fn with_hash(mut event: RoomEvent, hash: Option<&str>) -> RoomEvent {
        let hash = match hash {
            Some(hash) => hash.to_owned(),
            None => rezzy::compute_content_hash(&event.pdu, "10").unwrap(),
        };
        if let OwnedValue::Object(fields) = &mut event.pdu {
            let mut hashes = std::collections::BTreeMap::new();
            hashes.insert("sha256".to_owned(), OwnedValue::String(hash));
            fields.insert("hashes".to_owned(), OwnedValue::Object(hashes));
        }
        event
    }

    #[test]
    fn a_matching_content_hash_is_not_reported() {
        let create = with_hash(event("$create", "m.room.create", &[], &[]), None);
        let findings = derived_auth_findings(&export(vec![create])).unwrap();
        assert!(findings.hash_mismatched.is_empty());
        assert_eq!(findings.redacted_pdus, [None]);
    }

    #[test]
    fn a_content_hash_mismatch_redacts_the_event_and_does_not_reject_it() {
        let create = with_hash(
            event("$create", "m.room.create", &[], &[]),
            Some("not-the-hash"),
        );
        let create_pdu = create.pdu.clone();
        let findings = derived_auth_findings(&export(vec![create])).unwrap();
        assert_eq!(findings.hash_mismatched, ["$create"]);
        let audit = &findings.hash_audit[0];
        assert_eq!(audit.event_id, "$create");
        assert_eq!(audit.stored, "not-the-hash");
        assert_ne!(audit.computed, audit.stored);
        assert!(audit.json_line().starts_with('{'));
        assert!(audit.json_line().ends_with('}'));
        let expected = rezzy::try_canonical_redacted_json(&create_pdu, "10").unwrap();
        assert_eq!(
            findings.redacted_pdus[0].as_deref(),
            Some(expected.as_bytes()),
            "the redacted form is what gets stored"
        );
        assert_eq!(forced(&findings, 0), None, "a bad hash is not a rejection");
        assert!(findings.warnings().iter().any(|line| line.contains("hash")));
    }

    #[test]
    fn the_audit_distinguishes_already_redacted_content() {
        let create = with_hash(event("$create", M_ROOM_CREATE, &[], &[]), None);
        let mut join = with_hash(
            event("$join", "m.room.member", &[], &["$create"]),
            Some("bad"),
        );
        let mut message = with_hash(
            event("$msg", "m.room.message", &[], &["$create"]),
            Some("bad"),
        );
        for event in [&mut join, &mut message] {
            if let OwnedValue::Object(fields) = &mut event.pdu {
                fields.insert(
                    "sender".to_owned(),
                    OwnedValue::String("@alice:example.org".to_owned()),
                );
            }
        }
        if let OwnedValue::Object(fields) = &mut join.pdu {
            fields.insert(
                "state_key".to_owned(),
                OwnedValue::String("@alice:example.org".to_owned()),
            );
            let mut content = std::collections::BTreeMap::new();
            content.insert(
                "membership".to_owned(),
                OwnedValue::String("join".to_owned()),
            );
            fields.insert("content".to_owned(), OwnedValue::Object(content));
        }
        if let OwnedValue::Object(fields) = &mut message.pdu {
            let mut content = std::collections::BTreeMap::new();
            content.insert("body".to_owned(), OwnedValue::String("hi".to_owned()));
            fields.insert("content".to_owned(), OwnedValue::Object(content));
        }

        let findings = derived_auth_findings(&export(vec![create, join, message])).unwrap();
        let audit = |id: &str| {
            findings
                .hash_audit
                .iter()
                .find(|audit| audit.event_id == id)
                .expect("audit row")
        };
        assert!(audit("$join").content_matches_redaction);
        assert!(audit("$join")
            .json_line()
            .contains("\"content_matches_redaction\":true"));
        assert!(!audit("$msg").content_matches_redaction);
        assert!(audit("$msg")
            .json_line()
            .contains("\"content_matches_redaction\":false"));
        let warnings = findings.warnings().join("\n");
        assert!(
            !warnings.contains("$join"),
            "already-redacted joins are not listed: {warnings}"
        );
        assert!(
            warnings.contains("$msg"),
            "content-differs are listed: {warnings}"
        );
    }

    #[test]
    fn an_event_without_a_hash_is_judged_as_it_is() {
        let findings =
            derived_auth_findings(&export(vec![event("$create", "m.room.create", &[], &[])]))
                .unwrap();
        assert!(findings.hash_mismatched.is_empty());
    }

    #[test]
    fn an_unknown_room_version_derives_only_the_missing_auth_rejections() {
        let mut rejected = event("$msg", "m.room.message", &["$create"], &["$create"]);
        rejected.metadata.rejected = true;
        let export = export(vec![create_with_version("not-a-version"), rejected]);
        assert_eq!(
            derived_auth_findings(&export).unwrap(),
            auth_findings(&export).unwrap()
        );
    }

    #[test]
    fn a_forced_rejection_stores_a_rejected_outlier_and_nothing_exported() {
        let verdict = verdict_of(&EventMetadata::default(), Some(MISSING_AUTH));
        assert_eq!(verdict.rejection, Some(MISSING_AUTH));
        assert!(verdict.outlier);

        let exported = EventMetadata {
            rejected: true,
            rejection_code: Some(7),
            soft_failed: true,
            soft_fail_code: Some(2),
            outlier: true,
            ..EventMetadata::default()
        };
        assert!(
            verdict_of(&exported, None).is_empty(),
            "exported rejection, soft-fail and outlier flags are not migrated"
        );
        assert_eq!(
            verdict_of(&exported, Some(MISSING_AUTH)).rejection,
            Some(MISSING_AUTH)
        );
        assert!(verdict_of(&EventMetadata::default(), None).is_empty());
    }

    fn db() -> (Database, std::path::PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mtxdb-closure-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        (Database::open(dir.clone()).unwrap(), dir)
    }

    fn read_pool(
        db: &Database,
        pool: mtxdb::ShardType,
        collection: [u8; 16],
        record: [u8; 16],
    ) -> Option<Vec<u8>> {
        db.pool(pool)
            .get(&collection, &record)
            .unwrap()
            .map(|data| data.bytes.to_vec())
    }

    fn read(db: &Database, collection: [u8; 16], record: [u8; 16]) -> Option<Vec<u8>> {
        db.pool(EDGE_POOL)
            .get(&collection, &record)
            .unwrap()
            .map(|data| data.bytes.to_vec())
    }

    /// The closure the server would serve for `id`: the persisted bitmap
    /// resolved through the reverse records, or `None` when none was written.
    fn persisted(
        db: &Database,
        room: &str,
        id: &str,
    ) -> Option<std::collections::BTreeSet<String>> {
        let collection = room_auth_collection_id(room.as_bytes());
        let forward = read(db, collection, auth_forward_record_id(id.as_bytes()))?;
        let (short, stored) = decode_auth_forward(&forward).unwrap();
        assert_eq!(stored, id.as_bytes());
        let closure =
            Bitmap::decode(&read(db, collection, auth_closure_record_id(short))?).unwrap();
        Some(
            closure
                .iter()
                .map(|short| {
                    let id = read(db, collection, auth_reverse_record_id(short)).unwrap();
                    String::from_utf8(id).unwrap()
                })
                .collect(),
        )
    }

    /// Reference: every event reachable through auth edges.
    fn walked(export: &RoomExport, id: &str) -> std::collections::BTreeSet<String> {
        let by_id: HashMap<&str, &RoomEvent> = export
            .events
            .iter()
            .map(|event| (event.event_id.as_str(), event))
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![id.to_owned()];
        while let Some(current) = stack.pop() {
            let Some(event) = by_id.get(current.as_str()) else {
                continue;
            };
            for auth in id_list(event, "auth_events").unwrap() {
                let auth = String::from_utf8(auth.to_vec()).unwrap();
                if seen.insert(auth.clone()) {
                    stack.push(auth);
                }
            }
        }
        seen
    }

    fn diamond() -> RoomExport {
        export(vec![
            event(
                "$member",
                "m.room.member",
                &[],
                &["$create", "$power", "$rules"],
            ),
            event("$rules", "m.room.join_rules", &[], &["$create", "$power"]),
            event("$power", "m.room.power_levels", &[], &["$create"]),
            event("$create", "m.room.create", &[], &[]),
        ])
    }

    fn counter(db: &Database, room: &str) -> u32 {
        let bytes = read(
            db,
            room_auth_collection_id(room.as_bytes()),
            auth_counter_record_id(),
        )
        .unwrap();
        decode_auth_counter(&bytes).unwrap()
    }

    #[test]
    fn closures_match_a_walk_of_the_auth_edges() {
        let (db, dir) = db();
        let room = diamond();
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        for event in &room.events {
            assert_eq!(
                persisted(&db, &room.room_id, &event.event_id),
                Some(walked(&room, &event.event_id)),
                "{}",
                event.event_id
            );
        }
        assert_eq!(counter(&db, &room.room_id), 4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn closures_survive_capped_batches_and_a_reimport_keeps_the_shorts() {
        let (db, dir) = db();
        let room = diamond();
        // About one event per batch.
        write_room_with_findings(&db, &room, 1200).unwrap();
        let collection = room_auth_collection_id(room.room_id.as_bytes());
        let before: Vec<_> = room
            .events
            .iter()
            .map(|e| {
                read(
                    &db,
                    collection,
                    auth_forward_record_id(e.event_id.as_bytes()),
                )
            })
            .collect();
        assert!(before.iter().all(Option::is_some));
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        let after: Vec<_> = room
            .events
            .iter()
            .map(|e| {
                read(
                    &db,
                    collection,
                    auth_forward_record_id(e.event_id.as_bytes()),
                )
            })
            .collect();
        assert_eq!(before, after, "re-import reuses the stored short ids");
        assert_eq!(counter(&db, &room.room_id), 4);
        for event in &room.events {
            assert_eq!(
                persisted(&db, &room.room_id, &event.event_id),
                Some(walked(&room, &event.event_id))
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_missing_auth_event_leaves_it_and_its_dependents_without_closures() {
        let (db, dir) = db();
        let room = export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$orphan", "m.room.member", &[], &["$create", "$gone"]),
            event("$child", "m.room.message", &[], &["$orphan"]),
            event("$fine", "m.room.message", &[], &["$create"]),
        ]);
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        assert_eq!(persisted(&db, &room.room_id, "$orphan"), None);
        assert_eq!(persisted(&db, &room.room_id, "$child"), None);
        assert_eq!(
            persisted(&db, &room.room_id, "$fine"),
            Some(walked(&room, "$fine"))
        );
        // The unknown auth event still has a short id, as it would on the server.
        let collection = room_auth_collection_id(room.room_id.as_bytes());
        assert!(read(&db, collection, auth_forward_record_id(b"$gone")).is_some());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn closure_counts_fall_short_of_event_counts_when_auth_is_missing() {
        let (db, dir) = db();
        let room = export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$orphan", "m.room.member", &[], &["$create", "$gone"]),
            event("$child", "m.room.message", &[], &["$orphan"]),
            event("$fine", "m.room.message", &[], &["$create"]),
        ]);
        let (counts, _) = write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        assert_eq!((counts.events, counts.closures), (4, 2));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_auth_cycle_fails_the_room() {
        let (db, dir) = db();
        let room = export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$a", "m.room.member", &[], &["$b"]),
            event("$b", "m.room.member", &[], &["$a"]),
        ]);
        let error = write_room_with_findings(&db, &room, BATCH_BYTES)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cycle"), "{error}");
        let record = event_record_id(b"$create");
        assert!(
            read_pool(
                &db,
                EVENT_POOL,
                room_event_collection_id(room.room_id.as_bytes()),
                record
            )
            .is_none(),
            "a refused room leaves no event behind"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_journal_is_left_alone_at_or_below_the_reclaim_trigger() {
        let mut synced = Vec::new();
        for segment_len in [10, 11] {
            reclaim_journal_with(segment_len, 11, |pool| {
                synced.push(pool);
                Ok(())
            })
            .unwrap();
        }
        assert!(synced.is_empty());
    }

    #[test]
    fn every_pool_syncs_once_the_journal_passes_the_trigger() {
        for segment_len in [12, 13, u64::MAX] {
            let mut synced = Vec::new();
            reclaim_journal_with(segment_len, 11, |pool| {
                synced.push(pool);
                Ok(())
            })
            .unwrap();
            assert_eq!(synced, mtxdb::ShardType::ALL, "segment {segment_len}");
        }
    }

    #[test]
    fn a_failed_pool_sync_stops_the_import() {
        let mut synced = Vec::new();
        let error = reclaim_journal_with(100, 1, |pool| {
            synced.push(pool);
            if synced.len() == 2 {
                anyhow::bail!("disk full");
            }
            Ok(())
        })
        .unwrap_err();
        assert!(error.to_string().contains("disk full"));
        assert_eq!(synced.len(), 2, "no pool is synced after the failure");
    }

    #[test]
    fn importing_a_small_room_is_below_the_reclaim_trigger() {
        let (db, dir) = db();
        let room = diamond();
        write_room_with_findings(&db, &room, 1200).unwrap();
        let coordinator = db.coordinator();
        let segment_len = coordinator.segment_len();
        assert!(segment_len <= coordinator.reclaim_trigger_len());
        reclaim_journal(&db).unwrap();
        assert_eq!(coordinator.segment_len(), segment_len);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn derived_rejections_are_tallied_by_code() {
        let findings = derived_auth_findings(&export(vec![
            create_with_version("10"),
            event("$msg", "m.room.message", &["$create"], &["$missing"]),
        ]))
        .unwrap();
        assert_eq!(
            findings
                .rejection_counts
                .get(&RejectionCode::MissingAuthEvent.to_u8()),
            Some(&1)
        );
    }
}

#[cfg(test)]
mod signature_window_tests {
    use bytes::Bytes;
    use mtxdb::server_keys::{key_digest, KeyTier, ServerKey};
    use rezzy::signing::ed25519_zebra::SigningKey;

    use super::*;

    const ORIGIN: &str = "origin.example";

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from([seed; 32])
    }

    /// A key record for `sk` under `key_id`, with the given cached deadline and
    /// optional retirement.
    fn key_record(
        sk: &SigningKey,
        key_id: &str,
        valid_until_ts: u64,
        expired_ts: Option<u64>,
    ) -> KeyRecord {
        let bytes: Vec<u8> = AsRef::<[u8]>::as_ref(&sk.verification_key()).to_vec();
        KeyRecord::Key {
            origin: ORIGIN.to_owned(),
            key: ServerKey {
                key_id: key_id.as_bytes().to_vec(),
                digest: key_digest(&bytes),
                key: bytes,
                valid_until_ts,
                source: b"test".to_vec(),
                ts_added_ms: 0,
                tier: KeyTier::Permanent,
                first_seen_ms: 0,
                last_active_ms: 0,
                corroborated: false,
                inferred_retired: false,
                expired_ts,
                promote_by_ms: 0,
            },
        }
    }

    /// A message event at `ts`, signed under each `(key_id, signing key)`.
    /// Signing a key that is not the registered one makes a bad signature.
    fn signed_event(ts: u64, signers: &[(&str, &SigningKey)]) -> RoomEvent {
        signed_event_for(ts, signers, "10")
    }

    fn signed_event_for(ts: u64, signers: &[(&str, &SigningKey)], room_version: &str) -> RoomEvent {
        let fields = format!(
            r#""type":"m.room.message","room_id":"!r:{ORIGIN}","sender":"@a:{ORIGIN}","origin_server_ts":{ts},"content":{{"body":"hi"}},"depth":1,"auth_events":[],"prev_events":[]"#
        );
        let unsigned = OwnedValue::parse_bytes(format!("{{{fields}}}").as_bytes()).unwrap();
        let canonical = rezzy::try_canonical_redacted_json(&unsigned, room_version).unwrap();
        let signatures: Vec<String> = signers
            .iter()
            .map(|(key_id, sk)| {
                use base64::Engine as _;
                let signature = sk.sign(canonical.as_bytes());
                let encoded =
                    base64::engine::general_purpose::STANDARD_NO_PAD.encode(signature.to_bytes());
                format!(r#""{key_id}":"{encoded}""#)
            })
            .collect();
        let signed = format!(
            r#"{{{fields},"signatures":{{"{ORIGIN}":{{{}}}}}}}"#,
            signatures.join(",")
        );
        RoomEvent {
            event_id: "$m".to_owned(),
            metadata: EventMetadata::default(),
            prev_events: OwnedValue::Array(Vec::new()),
            auth_events: OwnedValue::Array(Vec::new()),
            pdu: OwnedValue::parse_bytes(signed.as_bytes()).unwrap(),
            pdu_bytes: Bytes::new(),
        }
    }

    fn catalog(records: &[KeyRecord]) -> SignatureKeyCatalog {
        SignatureKeyCatalog::from_records(records)
    }

    #[test]
    fn a_signature_inside_the_window_verifies() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, None)]);
        let outcome = keys.verify(&signed_event(1_000, &[("ed25519:a", &sk)]), "10");
        assert!(matches!(outcome, SignatureOutcome::Verified(_)));
    }

    #[test]
    fn an_unretired_key_past_its_cached_deadline_verifies_late() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, None)]);
        let outcome = keys.verify(&signed_event(3_000, &[("ed25519:a", &sk)]), "10");
        assert!(matches!(outcome, SignatureOutcome::VerifiedLate));
    }

    #[test]
    fn equality_at_the_cached_deadline_is_in_window() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, None)]);
        assert!(
            matches!(
                keys.verify(&signed_event(2_000, &[("ed25519:a", &sk)]), "10"),
                SignatureOutcome::Verified(_)
            ),
            "valid_until_ts >= T covers equality"
        );
    }

    #[test]
    fn equality_at_the_cached_deadline_still_rejects_a_bad_retired_signature() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, Some(5_000))]);
        assert!(
            matches!(
                keys.verify(&signed_event(2_000, &[("ed25519:a", &signer)]), "10"),
                SignatureOutcome::Invalid(_)
            ),
            "equality is inside the freshness window, not the stale-cache gap"
        );
    }

    #[test]
    fn a_bad_signature_inside_the_window_is_invalid() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let outcome = keys.verify(&signed_event(1_000, &[("ed25519:a", &signer)]), "10");
        assert!(matches!(outcome, SignatureOutcome::Invalid(_)));
    }

    #[test]
    fn a_failure_with_another_missing_key_is_deferred() {
        let (bad_signer, registered, missing_signer) =
            (signing_key(1), signing_key(2), signing_key(3));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let event = signed_event(
            1_000,
            &[
                ("ed25519:a", &bad_signer),
                ("ed25519:missing", &missing_signer),
            ],
        );
        assert!(
            matches!(keys.verify(&event, "10"), SignatureOutcome::NoKey),
            "an unresolved key could still verify, so the event is not Invalid yet"
        );
    }

    #[test]
    fn a_bad_signature_past_the_cached_deadline_is_not_a_rejection() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let outcome = keys.verify(&signed_event(3_000, &[("ed25519:a", &signer)]), "10");
        assert!(matches!(outcome, SignatureOutcome::NoKey));
    }

    #[test]
    fn a_late_success_wins_over_another_late_failure() {
        let (good, wrong) = (signing_key(1), signing_key(2));
        let keys = catalog(&[
            key_record(&wrong, "ed25519:bad", 1_000, None),
            key_record(&good, "ed25519:good", 1_000, None),
        ]);
        let event = signed_event(3_000, &[("ed25519:bad", &good), ("ed25519:good", &good)]);

        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::VerifiedLate
        ));
    }

    #[test]
    fn an_in_window_success_beats_a_late_failure() {
        let (good, wrong) = (signing_key(1), signing_key(2));
        let keys = catalog(&[
            // Late and wrong: registered key differs from the signer.
            key_record(&wrong, "ed25519:late", 1_000, None),
            key_record(&good, "ed25519:live", 5_000, None),
        ]);
        let event = signed_event(3_000, &[("ed25519:late", &good), ("ed25519:live", &good)]);
        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::Verified(_)
        ));
    }

    #[test]
    fn an_in_window_success_beats_a_late_success() {
        let sk = signing_key(1);
        let keys = catalog(&[
            key_record(&sk, "ed25519:late", 1_000, None),
            key_record(&sk, "ed25519:live", 5_000, None),
        ]);
        let event = signed_event(3_000, &[("ed25519:late", &sk), ("ed25519:live", &sk)]);
        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::Verified(_)
        ));
    }

    #[test]
    fn signature_rejections_are_included_in_the_rejection_tally() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let event = signed_event(1_000, &[("ed25519:a", &signer)]);
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let mut findings = AuthFindings {
            forced: vec![None],
            ..AuthFindings::default()
        };
        let outcome = keys.verify(&event, "10");
        apply_signature_outcomes(
            &mut findings,
            std::slice::from_ref(&event),
            vec![outcome],
            &keys,
            "!r:origin.example",
            "10",
        );
        refresh_rejection_counts(&mut findings);

        assert_eq!(
            findings
                .rejection_counts
                .get(&RejectionCode::SignatureVerificationFailed.to_u8()),
            Some(&1)
        );
    }

    #[test]
    fn a_retired_key_past_its_cached_deadline_verifies_in_window_until_expiry() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, Some(5_000))]);
        let event = signed_event(3_000, &[("ed25519:a", &sk)]);
        let outcome = keys.verify(&event, "10");
        assert!(
            matches!(&outcome, SignatureOutcome::Verified(_)),
            "the cached deadline is not the retired key's signing boundary"
        );

        // A retired key still inside its expiry window corroborates normally.
        let mut findings = AuthFindings {
            forced: vec![None],
            ..AuthFindings::default()
        };
        apply_signature_outcomes(
            &mut findings,
            std::slice::from_ref(&event),
            vec![outcome],
            &keys,
            "!r:origin.example",
            "10",
        );
        assert_eq!(findings.verified_keys.len(), 1);
        assert!(findings.signature_late.is_empty());
    }

    #[test]
    fn a_retired_key_past_its_cached_deadline_does_not_reject_a_bad_signature() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, Some(5_000))]);
        let outcome = keys.verify(&signed_event(3_000, &[("ed25519:a", &signer)]), "10");
        assert!(
            matches!(outcome, SignatureOutcome::NoKey),
            "a stale cached deadline must not turn an unverifiable retired-key signature into a rejection"
        );
    }

    #[test]
    fn room_versions_before_five_ignore_the_cached_deadline() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, None)]);
        let event = signed_event_for(3_000, &[("ed25519:a", &sk)], "4");
        assert!(
            matches!(keys.verify(&event, "4"), SignatureOutcome::Verified(_)),
            "v1-v4 do not use valid_until_ts as a historical cutoff"
        );
    }

    #[test]
    fn room_versions_before_five_still_reject_a_bad_signature() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let event = signed_event_for(3_000, &[("ed25519:a", &signer)], "4");
        assert!(
            matches!(keys.verify(&event, "4"), SignatureOutcome::Invalid(_)),
            "without a cached-deadline rule, a bad v1-v4 signature is in window"
        );
    }

    #[test]
    fn only_v5_and_later_rooms_apply_the_cached_deadline() {
        for version in ["1", "2", "3", "4"] {
            assert!(!room_version_uses_valid_until_ts(version), "{version}");
        }
        for version in ["5", "6", "9", "10", "11", "12", "12.1"] {
            assert!(room_version_uses_valid_until_ts(version), "{version}");
        }
    }

    #[test]
    fn a_retired_key_before_both_timestamps_verifies_in_window() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, Some(5_000))]);
        let outcome = keys.verify(&signed_event(1_000, &[("ed25519:a", &sk)]), "10");
        assert!(matches!(outcome, SignatureOutcome::Verified(_)));
    }

    #[test]
    fn a_retired_key_verifies_through_its_expiry_inclusively() {
        let sk = signing_key(1);
        for version in ["4", "10"] {
            for cached_deadline in [2_000, 5_000, 9_000] {
                let keys = catalog(&[key_record(
                    &sk,
                    "ed25519:a",
                    cached_deadline,
                    Some(5_000),
                )]);
                for ts in [4_999, 5_000] {
                    let event = signed_event_for(ts, &[("ed25519:a", &sk)], version);
                    assert!(
                        matches!(keys.verify(&event, version), SignatureOutcome::Verified(_)),
                        "v{version}, cached deadline {cached_deadline}, ts {ts}"
                    );
                    assert_eq!(keys.explain(&event, version), vec!["ed25519:a: in window"]);
                }
            }
        }
    }

    #[test]
    fn a_retired_key_is_unusable_one_millisecond_after_its_expiry() {
        let sk = signing_key(1);
        for version in ["4", "10"] {
            for cached_deadline in [2_000, 5_000, 9_000] {
                let keys = catalog(&[key_record(
                    &sk,
                    "ed25519:a",
                    cached_deadline,
                    Some(5_000),
                )]);
                let event = signed_event_for(5_001, &[("ed25519:a", &sk)], version);
                assert!(
                    matches!(keys.verify(&event, version), SignatureOutcome::NoKey),
                    "v{version}, cached deadline {cached_deadline}"
                );
                assert_eq!(
                    keys.explain(&event, version),
                    vec!["ed25519:a: event after expired_ts 5000"]
                );
            }
        }
    }

    #[test]
    fn late_events_reach_the_late_sidecar_and_not_the_failure_one() {
        let sk = signing_key(1);
        let keys = catalog(&[key_record(&sk, "ed25519:a", 2_000, None)]);
        let events = vec![signed_event(3_000, &[("ed25519:a", &sk)])];
        let mut findings = AuthFindings {
            forced: vec![None; events.len()],
            hash_audit: vec![HashAudit {
                event_id: events[0].event_id.clone(),
                room_id: "!r:origin.example".to_owned(),
                room_version: "10".to_owned(),
                stored: "stored".to_owned(),
                computed: "computed".to_owned(),
                content_matches_redaction: false,
                signature_status: None,
            }],
            ..AuthFindings::default()
        };
        let outcomes = vec![SignatureOutcome::VerifiedLate];
        apply_signature_outcomes(
            &mut findings,
            &events,
            outcomes,
            &keys,
            "!r:origin.example",
            "10",
        );
        assert_eq!(findings.signature_verified_late, 1);
        assert_eq!(findings.signature_late.len(), 1);
        assert_eq!(findings.signature_late[0].outcome, "verified_late");
        assert!(findings.signature_audit.is_empty());
        assert_eq!(
            findings.hash_audit[0].signature_status,
            Some("verified_late")
        );
        assert!(
            findings.verified_keys.is_empty(),
            "late events do not corroborate"
        );
    }
}
