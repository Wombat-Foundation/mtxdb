//! Writing an `mtxdb-export/v1` file into the server's own layouts.
//!
//! Event bodies go to their declared event collections. Room adjacency and auth
//! closures use the library graph path; locators, verdicts, and signing keys use
//! their declared collections. The CLI import path then reconstructs state
//! groups and HAMT records from the room's imported PDUs.
//!
//! A run is complete only when the import marker says so. Before the first
//! record of a run the marker is cleared, data goes in capped batches, and the
//! marker is committed last, in a transaction of its own. A run that stops
//! anywhere earlier leaves the database visibly incomplete.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{anyhow, bail, Context};
use mtxdb::csr_adjacency::{AppendDurability, EventRecord};
use mtxdb::export_schema::{RejectionCode, SoftFailCode};
use mtxdb::interop::{
    decode_event_locator, decode_import_marker, encode_event_locator, encode_import_marker,
    encode_room_version_record, encode_verdict, event_locator_collection_id, event_record_id,
    event_verdict_collection_id, import_marker_collection_id, import_marker_node_id,
    room_event_collection_id, room_version_collection_id, room_version_node_id, ImportMarker,
    Verdict, DATA_FORMAT, EVENT_POOL, SERVER_INFO_POOL,
};
use mtxdb::room_auth::{RoomAuth, MAX_BATCH_EVENTS};
use mtxdb::room_flag::{Flag, FlagChange, RoomFlagIndex};
use mtxdb::server_keys;
use mtxdb::storage::StorageEngine as _;
use mtxdb::{Database, NodeData, PackfileStorage};
use rezzy::auth::{check_auth, StateProvider};
use rezzy::basespec::event_types::{M_ROOM_CREATE, M_ROOM_MEMBER};
use rezzy::json::Value as OwnedValue;
use rezzy::signing::{
    ed25519_zebra::{Signature, VerificationKey},
    verify_event_signatures_from_server, SignatureVerifier,
};
use rezzy::{LeanEvent, StateResVersion};

use crate::interchange::{EventMetadata, KeyRecord, RoomEvent, RoomExport};
use crate::state_resolve::lean_event;

/// The auth events selected from the state before one event:
/// `(event type, state key, event id)`.
pub(crate) type AuthEntries = Vec<(String, String, String)>;

/// Bytes one batch may stage. Half the engine's per-transaction limit, so the
/// estimate below can be rough.
pub(crate) const BATCH_BYTES: usize = 32 << 20;

/// Let the shared journal reclaim during a long import.
///
/// The journal is one 256 MiB segment shared by every pool, and it is only
/// reclaimed once every pool the writer touched has synced and so reported
/// coverage. An import that syncs only at the end fills the segment and every
/// later commit fails with "journal segment is full". Once the segment passes
/// `trigger` bytes, sync the pools the journal is waiting on, one after
/// another; the last pool's coverage lets the journal shrink. At or below
/// `trigger` this does nothing.
///
/// `should_sync` decides which pools are worth a sync. A pool whose frames the
/// segment does not need to drop for this cut would rewrite a full checkpoint
/// and report an LSN it already reported, so skipping it costs nothing. A pool
/// that is blocking the reclaim must answer `true`: its sync is what unblocks
/// it. The coordinator's own back-off (`should_force_reclaim_checkpoint`)
/// already encodes both rules, including the retry after the segment has grown
/// by an eighth of the trigger without coverage moving.
pub(crate) fn reclaim_journal_with(
    segment_len: u64,
    trigger: u64,
    mut should_sync: impl FnMut(mtxdb::ShardType) -> bool,
    mut sync_pool: impl FnMut(mtxdb::ShardType) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    if segment_len <= trigger {
        return Ok(());
    }
    for pool in mtxdb::ShardType::ALL {
        if !should_sync(pool) {
            continue;
        }
        sync_pool(pool)?;
    }
    Ok(())
}

/// [`reclaim_journal_with`] against the database's own journal and pools.
pub(crate) fn reclaim_journal(db: &Database) -> anyhow::Result<()> {
    let coordinator = db.coordinator();
    let diagnose = crate::cmd::import_timings_enabled();
    reclaim_journal_with(
        coordinator.segment_len(),
        coordinator.reclaim_trigger_len(),
        |pool| coordinator.should_force_reclaim_checkpoint(Some(pool)),
        |pool| {
            let segment_before = coordinator.segment_len();
            let started = std::time::Instant::now();
            db.pool(pool)
                .sync_all()
                .with_context(|| format!("syncing the {pool:?} pool to reclaim the journal"))?;
            if diagnose {
                report_reclaim_sync(db, pool, segment_before, started.elapsed());
            }
            Ok(())
        },
    )
}

/// One line per pool sync a journal reclaim ran: where its time went, and how
/// far the segment shrank. Only printed under `MTXDB_IMPORT_TIMINGS`.
fn report_reclaim_sync(
    db: &Database,
    pool: mtxdb::ShardType,
    segment_before: u64,
    wall: std::time::Duration,
) {
    let segment_after = db.coordinator().segment_len();
    let Some(t) = db.pool(pool).stats().last_sync_timings else {
        eprintln!("  timing: reclaim sync {pool:?} {wall:?}, no sync timings recorded");
        return;
    };
    eprintln!(
        "  timing: reclaim sync {pool:?} wall {wall:?}: pack flush {:?} fsync {:?}, \
         sidecar {:?}, delta log {:?}, checkpoint {:?}, reclaim {:?} (copy {:?}, fsync {:?}, \
         retained {} B, blocked by {:?}), remediation {:?}, wal {:?} (fsync {:?}, {} sync calls, \
         {} records); segment {segment_before} -> {segment_after} B",
        t.pack_flush,
        t.pack_fsync,
        t.sidecar,
        t.delta_log,
        t.checkpoint,
        t.reclaim,
        t.reclaim_copy,
        t.reclaim_fsync,
        t.reclaim_retained_bytes,
        t.reclaim_blocked_by,
        t.remediation,
        t.wal,
        t.journal_fsync,
        t.journal_sync_calls,
        t.journal_records,
    );
    // A full checkpoint is the largest single phase of a reclaim and the
    // engine already records where its time went. Print it only when this
    // sync actually wrote one, so a stale breakdown is never attributed to
    // the current reclaim.
    if t.checkpoint.is_zero() {
        return;
    }
    let Some(c) = db.pool(pool).checkpoint_breakdown() else {
        return;
    };
    eprintln!(
        "  timing: reclaim checkpoint {pool:?}: pre sync {:?}, lock wait {:?}, locked sync {:?}, \
         snapshot {:?}, serialize {:?}, write {:?}, dir sync {:?}, journal lsn {:?}, \
         reclaim {:?}, retire {:?}, unaccounted {:?}, {} bytes",
        c.pre_sync,
        c.lock_wait,
        c.locked_sync,
        c.snapshot,
        c.serialize,
        c.write,
        c.directory_sync,
        c.journal_lsn,
        c.reclaim,
        c.retire,
        c.unaccounted(),
        c.checkpoint_bytes,
    );
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

/// Test-only seam for the pre-marker barrier. A test can force the adjacency
/// sync to fail without relying on file modes, which a privileged user (or a
/// process with `CAP_DAC_OVERRIDE`) would ignore.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        static FAIL_ADJACENCY_SYNC: Cell<bool> = const { Cell::new(false) };
    }

    /// Make the next barrier on this thread report a failed adjacency sync.
    pub(crate) fn fail_adjacency_sync() {
        FAIL_ADJACENCY_SYNC.with(|flag| flag.set(true));
    }

    pub(crate) fn take_adjacency_sync_failure() -> bool {
        FAIL_ADJACENCY_SYNC.with(|flag| flag.replace(false))
    }
}

/// Make everything written so far durable. Called before the marker commits,
/// so the marker is never durable ahead of the data it vouches for.
pub(crate) fn sync_data(db: &Database) -> anyhow::Result<()> {
    sync_adjacency_tails(db)?;
    for pool in [EVENT_POOL, SERVER_INFO_POOL] {
        db.pool(pool)
            .sync_all()
            .with_context(|| format!("persisting the {pool:?} pool"))?;
    }
    Ok(())
}

/// Sync every graph tail. This is a separate call so the test seam at the tail
/// sync also guards the call: remove `sync_adjacency_tails` from `sync_data`
/// and the injected failure never fires, so the ordering test fails.
fn sync_adjacency_tails(db: &Database) -> anyhow::Result<()> {
    // Adjacency rows live in graph stores outside the packs. Rooms may defer
    // their syncs, so this barrier, not the room writer, is what guarantees
    // the marker never covers a row that is only in the page cache.
    //
    // Bind the sync result before consulting the test seam: if the call is
    // deleted, `synced` is undefined and the CLI no longer compiles. The
    // injected failure therefore cannot outlive the call it guards.
    let synced = mtxdb::csr_adjacency::sync_all_tails(db);
    #[cfg(test)]
    if test_hooks::take_adjacency_sync_failure() {
        anyhow::bail!("persisting the adjacency tails: injected failure");
    }
    synced.context("persisting the adjacency tails")
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

/// The verdict to store for an event: the rejection the import derived, the
/// soft-fail the export recorded, and `redacted_by`. Rejection is derived from
/// the stored events and server keys, not migrated. Soft-fail is the narrow
/// exception to that derivation rule: check 6 depends on the state at arrival
/// time, which the import cannot rebuild, so the export's value is kept as a
/// historical fact like `arrival_ordering`. A derived rejection still takes
/// precedence. `redacted_by` is a fact about the stored event, so it is kept.
fn verdict_of(metadata: &EventMetadata, forced: Option<u8>) -> Verdict {
    Verdict {
        rejection: forced,
        soft_fail: (forced.is_none() && metadata.soft_failed == Some(true)).then(|| {
            metadata
                .soft_fail_code
                .unwrap_or(SoftFailCode::Imported.to_u8())
        }),
        arrival_ordering: None,
        redacted_by: metadata
            .redacted_by
            .as_ref()
            .map(|redactor| redactor.as_bytes().to_vec()),
        // A derived rejection is stored as an outlier: the event cannot be
        // part of the room's accepted state.
        outlier: forced.is_some(),
    }
}

/// Determine the topological ordering (depth) and stream ordering for a timeline event.
///
/// Returns `None` if the event is an outlier or lacks required ordering fields.
pub(crate) fn timeline_orderings(
    export: &RoomExport,
    event: &RoomEvent,
    rejected: bool,
) -> Option<(i64, i64)> {
    if rejected || event.metadata.outlier {
        return None;
    }
    let stream_opt = if export.timeline {
        // When the export declares timeline contract, presence of
        // stream_ordering is load-bearing: events without it are not
        // timeline events (e.g. outliers or rejected events).
        event.stream_ordering
    } else {
        // Legacy export fallback without explicit timeline metadata.
        event.stream_ordering.or_else(|| {
            event.pdu.get("origin_server_ts").and_then(|v| {
                v.as_i64()
                    .or_else(|| v.as_u64().and_then(|d| i64::try_from(d).ok()))
            })
        })
    };

    let depth_opt = event.depth.or_else(|| {
        event.pdu.get("depth").and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_u64().and_then(|d| i64::try_from(d).ok()))
        })
    });

    match (depth_opt, stream_opt) {
        (Some(topological_ordering), Some(stream_ordering)) => {
            Some((topological_ordering, stream_ordering))
        }
        _ => None,
    }
}

/// The rejection the source recorded, or `None` when the export carried no
/// verdict at all. An absent verdict is not a claim that the event was
/// accepted, so disagreement reporting must skip it.
fn exported_rejection(metadata: &EventMetadata) -> Option<bool> {
    if metadata.rejected == Some(true) || metadata.rejection_code.is_some() {
        Some(true)
    } else {
        metadata.rejected.map(|_| false)
    }
}

/// The room's id and the hex ids of its event, auth and prev collections, as
/// `mtxdb info`/`get --collection` take them.
pub(crate) fn room_collections_line(room_id: &str) -> String {
    let room = room_id.as_bytes();
    format!(
        "room {room_id} (events 0x{})",
        hex::encode_upper(room_event_collection_id(room)),
    )
}

/// What a room's edges say about events the import cannot fully authorize.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AuthFindings {
    /// The rejection code forced on each event, by position in the export.
    pub(crate) forced: Vec<Option<u8>>,
    /// Events with an `auth_events` id that is not in the export.
    pub(crate) missing_auth: Vec<String>,
    /// Events rejected because they cite a rejected auth event, in the order
    /// the rejection reached them.
    pub(crate) cascaded: Vec<String>,
    /// Events with a `prev_events` id that is not in the export. These keep
    /// their verdicts: a gap in history is not an authorization failure.
    pub(crate) missing_prev: Vec<String>,
    /// Events the source exported as rejected that pass the state-before-event
    /// auth check, by id. The derived pass wins; the disagreement is reported.
    pub(crate) passes_but_exported_rejected: Vec<String>,
    /// Events the source exported as accepted that fail the state-before-event
    /// auth check, with the rule that refused them.
    pub(crate) fails_but_exported_accepted: Vec<AuthFailure>,
    /// Events whose content hash does not match their contents. They are not
    /// rejected for it: the auth check ran on their redaction instead.
    pub(crate) hash_mismatched: Vec<String>,
    /// The redacted PDU to store for each event, by position in the export,
    /// when its content hash does not match. The export keeps the original.
    redacted_pdus: Vec<Option<Vec<u8>>>,
    /// Events with a computed content-hash mismatch whose canonical redaction
    /// could not be produced; imports containing them fail closed.
    redactions_failed: HashSet<String>,
    /// One audit record per hash-mismatched event.
    pub(crate) hash_audit: Vec<HashAudit>,
    /// Key bindings corroborated by successfully verified historical events.
    pub(crate) verified_keys: Vec<VerifiedKeyObservation>,
    /// Events rejected because a usable, in-window required key had a bad
    /// signature.
    pub(crate) signature_failures: Vec<AuthFailure>,
    /// One record per event whose signature did not verify, whatever the reason.
    pub(crate) signature_audit: Vec<SignatureAudit>,
    /// Events with no in-window key for a required signer.
    pub(crate) signature_no_key: usize,
    /// Events without the origin, timestamp, or signature material to check.
    pub(crate) signature_no_material: usize,
    /// Events that verified against an unretired key after its cached
    /// `valid_until_ts`. That value is a refresh deadline, not the key's
    /// lifetime, so these are accepted but do not corroborate the key.
    pub(crate) signature_verified_late: usize,
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
#[derive(Clone, Debug, Default)]
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

    /// Merge another catalog's records into this one without consuming it.
    /// A key already stored by an earlier import must remain available when a
    /// run also brings its own keys file.
    pub(crate) fn extend_from(&mut self, other: &Self) {
        self.records.extend(
            other
                .records
                .iter()
                .map(|(id, record)| (id.clone(), record.clone())),
        );
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

    /// Stored signing keys named by one room's required signatures.
    ///
    /// Keys are fetched by indexed `(origin, key_id)` lookups rather than by
    /// scanning the `ServerInfo` packs. `stored_keys` remembers loaded key ids
    /// for the rest of the run. The caller drops the memo whenever keys are
    /// imported.
    pub(crate) fn read_for_room(
        store: &PackfileStorage,
        export: &RoomExport,
        stored_keys: &mut HashMap<String, Vec<mtxdb::server_keys::ServerKey>>,
    ) -> anyhow::Result<Self> {
        let mut catalog = Self::default();
        let Some(room_version) = create_room_version(&export.events) else {
            return Ok(catalog);
        };
        // Collect only the key ids present for required signers. The origin
        // and, for a restricted join, the authoriser's homeserver are the
        // only signatures consulted by `SignatureKeyCatalog::verify`.
        let mut servers: HashMap<String, (String, HashSet<Vec<u8>>)> = HashMap::new();
        for event in &export.events {
            let required = [
                event_origin(&event.pdu, room_version),
                restricted_join_authoriser_server(&event.pdu),
            ];
            let Some(signatures) = event.pdu.get("signatures").and_then(OwnedValue::as_object)
            else {
                continue;
            };
            for server in required.into_iter().flatten() {
                let entry = servers
                    .entry(server.to_ascii_lowercase())
                    .or_insert_with(|| (server.to_owned(), HashSet::new()));
                for (signature_server, signature_map) in signatures {
                    if !signature_server.eq_ignore_ascii_case(server) {
                        continue;
                    }
                    if let Some(signature_map) = signature_map.as_object() {
                        entry.1.extend(
                            signature_map
                                .keys()
                                .map(|key_id| key_id.as_bytes().to_vec()),
                        );
                    }
                }
            }
        }
        for (_, (server, key_ids)) in servers {
            if key_ids.is_empty() {
                continue;
            }
            let mut key_ids: Vec<_> = key_ids.into_iter().collect();
            key_ids.sort_unstable();
            let keys = memoized_server_keys(stored_keys, &server, &key_ids, |missing| {
                Ok(server_keys::read_server_keys_by_ids(
                    store,
                    server.as_bytes(),
                    missing,
                )?)
            })?;
            for key in keys {
                catalog.records.insert(
                    (server.to_ascii_lowercase(), key.key_id.clone()),
                    VerificationKeyRecord {
                        origin: server.clone(),
                        parsed_key: VerificationKey::try_from(key.key.as_slice()).ok(),
                        key: key.clone(),
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
        let origin_outcome = self.verify_server(event, room_version, origin, timestamp);
        let Some(authoriser) = restricted_join_authoriser(&event.pdu) else {
            return origin_outcome;
        };
        let Some(authoriser_server) = authoriser_server(authoriser) else {
            // A malformed authorising user id leaves a required signer
            // unresolved; it is missing evidence, not a bad signature.
            return SignatureOutcome::NoKey;
        };
        if authoriser_server.eq_ignore_ascii_case(origin) {
            return origin_outcome;
        }
        let authoriser_outcome =
            self.verify_server(event, room_version, authoriser_server, timestamp);

        // All required signers must have usable evidence before a failure can
        // be called Invalid. Otherwise one unavailable signer could hide a
        // valid signature and manufacture a rejection.
        if matches!(
            &origin_outcome,
            SignatureOutcome::NoKey | SignatureOutcome::NoMaterial
        ) || matches!(
            &authoriser_outcome,
            SignatureOutcome::NoKey | SignatureOutcome::NoMaterial
        ) {
            return SignatureOutcome::NoKey;
        }
        if let SignatureOutcome::Invalid(reason) = origin_outcome {
            return SignatureOutcome::Invalid(reason);
        }
        if let SignatureOutcome::Invalid(reason) = authoriser_outcome {
            return SignatureOutcome::Invalid(format!("authoriser {authoriser_server}: {reason}"));
        }
        let mut bindings = Vec::new();
        let mut late = false;
        for outcome in [origin_outcome, authoriser_outcome] {
            match outcome {
                SignatureOutcome::Verified(binding) => bindings.extend(binding),
                SignatureOutcome::VerifiedLate(binding) => {
                    bindings.extend(binding);
                    late = true;
                }
                SignatureOutcome::NoKey
                | SignatureOutcome::NoMaterial
                | SignatureOutcome::Invalid(_) => unreachable!(),
            }
        }
        if late {
            SignatureOutcome::VerifiedLate(bindings)
        } else {
            SignatureOutcome::Verified(bindings)
        }
    }

    fn verify_server(
        &self,
        event: &RoomEvent,
        room_version: &str,
        server: &str,
        timestamp: u64,
    ) -> SignatureOutcome {
        let Some(signatures) = event.pdu.get("signatures").and_then(OwnedValue::as_object) else {
            return SignatureOutcome::NoMaterial;
        };
        let Some(server_signatures) = signatures.get(server).and_then(OwnedValue::as_object) else {
            return SignatureOutcome::NoMaterial;
        };

        let mut candidates = Vec::new();
        let mut unresolved = false;
        for (key_id, signature) in server_signatures {
            let key_id_bytes = key_id.as_bytes();
            let Some(record) = self
                .records
                .get(&(server.to_ascii_lowercase(), key_id_bytes.to_vec()))
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
                    let binding = KeyBinding {
                        origin: record.origin.as_bytes().to_vec(),
                        key_id: record.key.key_id.clone(),
                    };
                    return if *late {
                        SignatureOutcome::VerifiedLate(vec![binding])
                    } else {
                        SignatureOutcome::Verified(vec![binding])
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

    /// Why each required signer's keys were or were not usable for the event,
    /// one line per key.
    fn explain(&self, event: &RoomEvent, room_version: &str) -> Vec<String> {
        let Some(origin) = event_origin(&event.pdu, room_version) else {
            return vec!["no origin server".to_owned()];
        };
        let mut lines = self.explain_server(&event.pdu, room_version, origin);
        if let Some(authoriser) = restricted_join_authoriser(&event.pdu) {
            let Some(server) = authoriser_server(authoriser) else {
                lines.push(format!("authoriser {authoriser}: invalid user id"));
                return lines;
            };
            if !server.eq_ignore_ascii_case(origin) {
                lines.extend(
                    self.explain_server(&event.pdu, room_version, server)
                        .into_iter()
                        .map(|line| format!("authoriser {line}")),
                );
            }
        }
        lines
    }

    fn explain_server(&self, pdu: &OwnedValue, room_version: &str, server: &str) -> Vec<String> {
        let timestamp = pdu.get("origin_server_ts").and_then(OwnedValue::as_u64);
        let uses_valid_until = room_version_uses_valid_until_ts(room_version);
        let Some(server_signatures) = pdu
            .get("signatures")
            .and_then(OwnedValue::as_object)
            .and_then(|signatures| signatures.get(server))
            .and_then(OwnedValue::as_object)
        else {
            return vec![format!("no signatures from {server}")];
        };
        server_signatures
            .keys()
            .map(|key_id| {
                let Some(record) = self
                    .records
                    .get(&(server.to_ascii_lowercase(), key_id.as_bytes().to_vec()))
                else {
                    return format!("{key_id}: not in key store");
                };
                let key = &record.key;
                if timestamp.is_none() {
                    return format!("{key_id}: missing origin_server_ts");
                }
                if key.key.is_empty() {
                    return format!("{key_id}: no key bytes");
                }
                if record.parsed_key.is_none() {
                    return format!("{key_id}: not a valid Ed25519 public key");
                }
                match timestamp {
                    Some(ts) if key.expired_ts.is_some_and(|expired| ts > expired) => format!(
                        "{key_id}: event after expired_ts {}",
                        key.expired_ts.unwrap_or_default()
                    ),
                    Some(ts) if uses_valid_until && ts > key.valid_until_ts => {
                        if key.expired_ts.is_some() {
                            format!(
                                "{key_id}: event after valid_until_ts {} (retired key remains valid through expired_ts {})",
                                key.valid_until_ts,
                                key.expired_ts.unwrap_or_default()
                            )
                        } else {
                            format!(
                                "{key_id}: event after valid_until_ts {}",
                                key.valid_until_ts
                            )
                        }
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
    Verified(Vec<KeyBinding>),
    /// Verified against an unretired key after its cached `valid_until_ts`.
    VerifiedLate(Vec<KeyBinding>),
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

/// The user named by a restricted join's
/// `join_authorised_via_users_server`, if this event is one.
fn restricted_join_authoriser(pdu: &OwnedValue) -> Option<&str> {
    if pdu.get("type")?.as_str()? != M_ROOM_MEMBER {
        return None;
    }
    let content = pdu.get("content")?.as_object()?;
    if content.get("membership")?.as_str()? != "join" {
        return None;
    }
    content.get("join_authorised_via_users_server")?.as_str()
}

/// The homeserver whose signature a restricted join requires.
fn restricted_join_authoriser_server(pdu: &OwnedValue) -> Option<&str> {
    restricted_join_authoriser(pdu).and_then(authoriser_server)
}

fn authoriser_server(user_id: &str) -> Option<&str> {
    user_id
        .strip_prefix('@')?
        .split_once(':')
        .map(|(_, server)| server)
        .filter(|server| !server.is_empty())
}

/// An accepted event the first auth check refuted, and why.
#[derive(Clone, Debug, Eq, PartialEq)]
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

fn visible_warning_ids<'a>(ids: &'a [String], current: Option<&HashSet<String>>) -> Vec<&'a str> {
    ids.iter()
        .filter(|id| current.is_none_or(|set| set.contains(id.as_str())))
        .map(String::as_str)
        .collect()
}

impl AuthFindings {
    /// Warning lines for events in the current file only.
    ///
    /// Stored events are redacted, so their source `rejected`/`soft_failed`
    /// claims are not available on a later import. Reporting those earlier
    /// events again would compare the derived verdict with a false default.
    pub(crate) fn current_warnings(&self, current: &HashSet<String>) -> Vec<String> {
        self.warnings_for(Some(current))
    }

    /// Human-readable warnings, empty for a clean room.
    pub(crate) fn warnings(&self) -> Vec<String> {
        self.warnings_for(None)
    }

    #[allow(clippy::too_many_lines)]
    fn warnings_for(&self, current: Option<&HashSet<String>>) -> Vec<String> {
        let id_limit = warning_id_limit();
        let mut lines = Vec::new();
        for (ids, what) in [
            (
                visible_warning_ids(&self.missing_auth, current),
                "stored as rejected outliers: an auth event is not in the export",
            ),
            (
                visible_warning_ids(&self.cascaded, current),
                "stored as rejected outliers: they cite a rejected auth event",
            ),
            (
                visible_warning_ids(&self.missing_prev, current),
                "have a prev event that is not in the export (verdicts unchanged)",
            ),
            (
                visible_warning_ids(&self.passes_but_exported_rejected, current),
                "were exported as rejected but pass the state-before-event auth check (stored accepted)",
            ),
        ] {
            let count = ids.len();
            if count == 0 {
                continue;
            }
            let shown: Vec<&str> = ids.iter().take(id_limit).copied().collect();
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
            .filter(|row| current.is_none_or(|set| set.contains(&row.event_id)))
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
        let failures: Vec<_> = self
            .fails_but_exported_accepted
            .iter()
            .filter(|failure| current.is_none_or(|set| set.contains(&failure.event_id)))
            .collect();
        if !failures.is_empty() {
            let mut groups: Vec<(&str, Vec<&str>)> = Vec::new();
            for failure in &failures {
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
                if take == 0 {
                    continue;
                }
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
            let more = failures.len().saturating_sub(shown);
            let tail = if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            };
            lines.push(format!(
                "{} events were exported as accepted but fail the state-before-event auth check \
                 (stored as rejected), grouped by reason: {}{tail}",
                failures.len(),
                parts.join("; "),
            ));
        }
        let signature_failures: Vec<_> = self
            .signature_failures
            .iter()
            .filter(|failure| current.is_none_or(|set| set.contains(&failure.event_id)))
            .collect();
        if !signature_failures.is_empty() {
            let shown = signature_failures
                .iter()
                .take(id_limit)
                .map(|failure| failure.event_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let more = signature_failures.len().saturating_sub(id_limit);
            let tail = if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            };
            let warning = format!(
                "{} events have an invalid signature from an available in-window key (stored as rejected): {shown}{tail}",
                signature_failures.len()
            );
            lines.push(warning);
        }
        lines
    }
}

/// Work out which events cannot be authorized from the room's own edges.
///
/// An event with an auth event missing from the export is forced to
/// `MissingAuthEvent`; an event citing a forced-rejected auth event is forced
/// to `DependsOnRejectedAuthEvent`, transitively. The exported verdicts are not
/// read. The room fails as a whole when its `m.room.create` would be forced.
pub(crate) fn auth_findings(export: &RoomExport) -> anyhow::Result<AuthFindings> {
    auth_findings_seeded(export, &vec![false; export.events.len()], None)
}

/// [`auth_findings`] with the rejections that start the cascade supplied by
/// the caller, by position in the export.
fn auth_findings_seeded(
    export: &RoomExport,
    seeds: &[bool],
    judge_only: Option<&HashSet<String>>,
) -> anyhow::Result<AuthFindings> {
    let events = &export.events;
    let index: HashMap<&[u8], usize> = events
        .iter()
        .enumerate()
        .map(|(position, event)| (event.event_id.as_bytes(), position))
        .collect();
    let in_scope =
        |position: usize| judge_only.is_none_or(|only| only.contains(&events[position].event_id));
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
        // A judged event whose own auth cites an id the export lacks really is
        // missing auth. An event outside the judged set may only look missing
        // because the caller supplied a partial event universe (the direct
        // path loads just this file's citations); that must not seed a cascade.
        if missing && !rejected[position] && in_scope(position) {
            findings.forced[position] = Some(RejectionCode::MissingAuthEvent.to_u8());
            rejected[position] = true;
            findings.missing_auth.push(event.event_id.clone());
        }
    }

    let mut pending: Vec<usize> = (0..events.len()).filter(|&i| rejected[i]).collect();
    while let Some(parent) = pending.pop() {
        for &child in &dependents[parent] {
            if !rejected[child] {
                rejected[child] = true;
                // Propagate through out-of-scope events so a rejection can
                // still reach a judged descendant, but only record verdicts
                // for events this pass was asked to judge.
                if in_scope(child) {
                    findings.forced[child] =
                        Some(RejectionCode::DependsOnRejectedAuthEvent.to_u8());
                    findings.cascaded.push(events[child].event_id.clone());
                }
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

/// Event ids the initial findings pass rejected: check 4, missing auth,
/// signature failure, or a cascade from one of those.
pub(crate) fn rejected_event_ids(export: &RoomExport, findings: &AuthFindings) -> HashSet<String> {
    export
        .events
        .iter()
        .zip(&findings.forced)
        .filter_map(|(event, forced)| forced.map(|_| event.event_id.clone()))
        .collect()
}

/// Apply the rejections the state walk derived (check 5 and rejected-auth
/// cascades): record the code, add a report row when the export disagreed, and
/// refresh the counts.
pub(crate) fn apply_walk_rejections(
    export: &RoomExport,
    findings: &mut AuthFindings,
    rejections: &HashMap<String, (u8, String)>,
) {
    let positions: HashMap<&str, usize> = export
        .events
        .iter()
        .enumerate()
        .map(|(position, event)| (event.event_id.as_str(), position))
        .collect();
    let exported: Vec<Option<bool>> = export
        .events
        .iter()
        .map(|event| exported_rejection(&event.metadata))
        .collect();
    for (event_id, (code, reason)) in rejections {
        let Some(&position) = positions.get(event_id.as_str()) else {
            continue;
        };
        findings.forced[position] = Some(*code);
        findings
            .passes_but_exported_rejected
            .retain(|id| id != event_id);
        if *code == RejectionCode::DependsOnRejectedAuthEvent.to_u8() {
            findings.cascaded.push(event_id.clone());
        } else if exported[position] == Some(false) {
            findings.fails_but_exported_accepted.push(AuthFailure {
                event_id: event_id.clone(),
                reason: reason.clone(),
            });
        }
    }

    refresh_rejection_counts(findings);
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
/// rejected for the mismatch. `None` when the hash matches, or cannot be
/// judged because it is absent or cannot be computed. A detected mismatch is
/// retained even if canonical redaction fails, in which case import fails closed.
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
    let Some((redacted, value)) = rezzy::try_canonical_redacted_json(pdu, room_version)
        .ok()
        .and_then(|redacted| {
            OwnedValue::parse_bytes(redacted.as_bytes())
                .ok()
                .map(|value| (redacted, value))
        })
    else {
        // Preserve the detected mismatch for reporting even if this PDU
        // cannot be canonically redacted. The original remains the only
        // storable representation, and the audit marks that fact.
        return Some(HashMismatch {
            value: pdu.clone(),
            bytes: Vec::new(),
            stored: expected.to_owned(),
            computed,
            content_matches_redaction: false,
            redaction_failed: true,
        });
    };
    // Record whether canonical redaction leaves the PDU's content unchanged.
    let content_matches_redaction = pdu.get("content") == value.get("content");
    Some(HashMismatch {
        value,
        bytes: redacted.into_bytes(),
        stored: expected.to_owned(),
        computed,
        content_matches_redaction,
        redaction_failed: false,
    })
}

/// An event whose content hash failed, and its canonical redaction.
pub(crate) struct HashMismatch {
    value: OwnedValue,
    bytes: Vec<u8>,
    stored: String,
    computed: String,
    /// Whether canonical redaction left the PDU's content unchanged.
    content_matches_redaction: bool,
    redaction_failed: bool,
}

/// Audit record for one hash-mismatched event, written to a JSONL sidecar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HashAudit {
    pub(crate) event_id: String,
    pub(crate) room_id: String,
    pub(crate) room_version: String,
    pub(crate) stored: String,
    pub(crate) computed: String,
    pub(crate) content_matches_redaction: bool,
    pub(crate) redaction_failed: bool,
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
            ("redaction_failed", self.redaction_failed.to_string()),
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
#[derive(Clone, Debug, Eq, PartialEq)]
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

/// Outcome of one event's authorization check.
pub(crate) enum EventAuthOutcome {
    Unchecked,
    Passed,
    Refused { code: u8, reason: String },
}

/// Judge an event against its auth events and the state before it.
pub(crate) fn judge_event_with_state(
    event: &RoomEvent,
    subject: Option<&LeanEvent<String>>,
    index: &HashMap<&[u8], usize>,
    lean: &[Option<LeanEvent<String>>],
    create: Option<&LeanEvent<String>>,
    version: StateResVersion,
    base: &AuthEntries,
) -> anyhow::Result<EventAuthOutcome> {
    judge_event_auth(event, subject, index, lean, create, version, Some(base))
}

fn judge_event_auth(
    event: &RoomEvent,
    subject: Option<&LeanEvent<String>>,
    index: &HashMap<&[u8], usize>,
    lean: &[Option<LeanEvent<String>>],
    create: Option<&LeanEvent<String>>,
    version: StateResVersion,
    base: Option<&AuthEntries>,
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
    // Check 4: authorization against the event's own auth_events. This runs
    // first and is not skipped when the state before the event is available.
    match check_auth(subject, &AuthState(entries), version, None) {
        Ok(()) => {}
        Err(error) => {
            return Ok(EventAuthOutcome::Refused {
                code: RejectionCode::AuthCheckFailed.to_u8(),
                reason: error.to_string(),
            });
        }
    }

    // Check 5: authorization against the state before the event.
    let Some(base) = base else {
        return Ok(EventAuthOutcome::Passed);
    };
    let mut entries: Vec<(&str, &str, &LeanEvent<String>)> = Vec::with_capacity(base.len());
    for (event_type, state_key, event_id) in base {
        let Some(&parent_position) = index.get(event_id.as_bytes()) else {
            return Ok(EventAuthOutcome::Unchecked);
        };
        let Some(parent) = lean[parent_position].as_ref() else {
            return Ok(EventAuthOutcome::Unchecked);
        };
        entries.push((event_type.as_str(), state_key.as_str(), parent));
    }
    if version.is_v2_1_plus()
        && !is_create
        && !entries
            .iter()
            .any(|(kind, key, _)| *kind == M_ROOM_CREATE && key.is_empty())
    {
        let Some(create) = create else {
            return Ok(EventAuthOutcome::Unchecked);
        };
        entries.push((M_ROOM_CREATE, "", create));
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
///
/// Events loaded from this store are already redacted; re-redacting them would
/// strip the auth projection laid over their stored content.
pub(crate) fn redactions_for_export(
    events: &[RoomEvent],
    room_version: &str,
    already_redacted: &HashSet<String>,
) -> Vec<Option<HashMismatch>> {
    par_map(events, |event| {
        if already_redacted.contains(&event.event_id) {
            None
        } else {
            redacted_for_hash(&event.pdu, room_version)
        }
    })
}

/// Below this many items the thread start-up costs more than it saves.
const PAR_MIN_ITEMS: usize = 512;

/// Map `work` over `items` on every available core, keeping input order.
///
/// Used for the per-event checks that read only the event and shared,
/// immutable state (content hashes, signatures). Small inputs run inline.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    if threads < 2 || items.len() < PAR_MIN_ITEMS {
        return items.iter().map(&work).collect();
    }
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(work).collect::<Vec<R>>()))
            .collect();
        let mut out = Vec::with_capacity(items.len());
        for handle in handles {
            out.extend(handle.join().expect("per-event worker panicked"));
        }
        out
    })
}

/// The auth-check input for each event, using its redacted form on mismatch.
pub(crate) fn lean_redacted_events(
    events: &[RoomEvent],
    redacted: &[Option<HashMismatch>],
    room_id: &str,
) -> Vec<Option<LeanEvent<String>>> {
    lean_redacted_events_for(events, redacted, room_id, |_| true)
}

/// Build lean events only for the ids in `needed`.
///
/// An incremental import judges this file's events and reads their auth
/// events; it does not need a second full-room `LeanEvent` copy for events
/// that keep their stored verdicts. Positions outside `needed` stay `None`.
pub(crate) fn lean_redacted_events_for(
    events: &[RoomEvent],
    redacted: &[Option<HashMismatch>],
    room_id: &str,
    mut needed: impl FnMut(&str) -> bool,
) -> Vec<Option<LeanEvent<String>>> {
    events
        .iter()
        .zip(redacted)
        .map(|(event, redacted)| {
            if !needed(&event.event_id) {
                return None;
            }
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
    judge_only: Option<&HashSet<String>>,
) -> anyhow::Result<AuthJudgement> {
    let mut checked = vec![false; events.len()];
    let mut refused: Vec<Option<u8>> = vec![None; events.len()];
    let mut reasons: Vec<Option<String>> = vec![None; events.len()];
    for (position, event) in events.iter().enumerate() {
        // Incremental imports only need to judge the events whose inputs can
        // have changed. Earlier events keep their stored verdicts, so running
        // the auth rules over the whole room again is pure duplicate work.
        if judge_only.is_some_and(|only| !only.contains(&event.event_id)) {
            continue;
        }
        match judge_event_auth(
            event,
            lean[position].as_ref(),
            index,
            lean,
            create,
            version,
            None,
        )? {
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
                redaction_failed: m.redaction_failed,
                signature_status: None,
            })
        })
        .collect()
}

fn apply_judgement(
    findings: &mut AuthFindings,
    events: &[RoomEvent],
    judgement: &AuthJudgement,
    exported: &[Option<bool>],
) {
    for (position, event) in events.iter().enumerate() {
        if !judgement.checked[position] {
            continue;
        }
        if let Some(code) = judgement.refused[position] {
            // The seed kept the cascade from assigning a code.
            findings.forced[position] = Some(code);
            if exported[position] == Some(false) {
                findings.fails_but_exported_accepted.push(AuthFailure {
                    event_id: event.event_id.clone(),
                    reason: judgement.reason[position]
                        .clone()
                        .unwrap_or_else(|| "unknown auth-check failure".to_owned()),
                });
            }
        } else if exported[position] == Some(true) && findings.forced[position].is_none() {
            // The event passes auth against the state before it, so the
            // export's rejection is not reproduced. Cascade-rejected events
            // remain rejected, matching the verdict that will be stored.
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
            SignatureOutcome::VerifiedLate(_) => "verified_late",
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
            if !matches!(outcome, SignatureOutcome::VerifiedLate(_)) {
                findings.signature_audit.push(row);
            }
        }
    }
    for (position, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            SignatureOutcome::NoKey => no_key = no_key.saturating_add(1),
            SignatureOutcome::NoMaterial => no_material = no_material.saturating_add(1),
            SignatureOutcome::VerifiedLate(_) => verified_late = verified_late.saturating_add(1),
            SignatureOutcome::Verified(bindings) => {
                let timestamp = events[position]
                    .pdu
                    .get("origin_server_ts")
                    .and_then(OwnedValue::as_u64)
                    .expect("verified signatures require origin_server_ts");
                for binding in bindings {
                    verified
                        .entry(binding)
                        .and_modify(|(first, last)| {
                            *first = (*first).min(timestamp);
                            *last = (*last).max(timestamp);
                        })
                        .or_insert((timestamp, timestamp));
                }
            }
            SignatureOutcome::Invalid(reason) => {
                if findings.forced[position].is_none() {
                    findings.forced[position] =
                        Some(RejectionCode::SignatureVerificationFailed.to_u8());
                }
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

/// Derive authorization findings from event contents and graph relationships.
/// Exported verdict annotations are used only to report disagreements.
/// Check 4 evaluates each event's auth events before check 5 evaluates the
/// state before it.
pub(crate) fn derived_auth_findings_with_keys(
    export: &RoomExport,
    keys: Option<&SignatureKeyCatalog>,
) -> anyhow::Result<AuthFindings> {
    derived_auth_findings_inner(
        export,
        keys,
        true,
        None,
        &HashSet::new(),
        None,
        &HashSet::new(),
        None,
    )
}

/// [`derived_auth_findings_with_keys`] for a partial raw import: a batch may
/// not contain enough history to authorize the create event, and that is not
/// a reason to refuse the batch.
///
/// `room_version` is passed explicitly because the stored create event is
/// redacted and may no longer carry `content.room_version`. `already_redacted`
/// names events loaded from this store: their content hash intentionally no
/// longer matches because the stored form is redacted, so they must not be
/// redacted again or their auth projection would be lost.
#[cfg(test)]
pub(crate) fn derived_auth_findings_for_raw(
    export: &RoomExport,
    room_version: &str,
    already_redacted: &HashSet<String>,
    judge_only: Option<&HashSet<String>>,
    prior_rejected: &HashSet<String>,
) -> anyhow::Result<AuthFindings> {
    derived_auth_findings_inner(
        export,
        None,
        false,
        Some(room_version),
        already_redacted,
        judge_only,
        prior_rejected,
        None,
    )
}

/// [`derived_auth_findings_for_raw_with_inputs`] with the redaction and lean-event inputs
/// the caller already computed. The incremental import also needs them for the
/// state walk, and rebuilding one `LeanEvent` per stored event twice per file
/// is a large duplicate cost.
pub(crate) fn derived_auth_findings_for_raw_with_inputs(
    export: &RoomExport,
    room_version: &str,
    already_redacted: &HashSet<String>,
    judge_only: Option<&HashSet<String>>,
    prior_rejected: &HashSet<String>,
    redacted: &[Option<HashMismatch>],
    lean: &[Option<LeanEvent<String>>],
) -> anyhow::Result<AuthFindings> {
    derived_auth_findings_inner(
        export,
        None,
        false,
        Some(room_version),
        already_redacted,
        judge_only,
        prior_rejected,
        Some((redacted, lean)),
    )
}

/// Precomputed redaction and lean-event inputs shared by the findings pass and
/// the state walk.
type JudgeInputs<'a> = (&'a [Option<HashMismatch>], &'a [Option<LeanEvent<String>>]);

#[allow(clippy::too_many_arguments)]
fn derived_auth_findings_inner(
    export: &RoomExport,
    keys: Option<&SignatureKeyCatalog>,
    require_create_authorized: bool,
    room_version_override: Option<&str>,
    already_redacted: &HashSet<String>,
    judge_only: Option<&HashSet<String>>,
    prior_rejected: &HashSet<String>,
    precomputed: Option<JudgeInputs<'_>>,
) -> anyhow::Result<AuthFindings> {
    let events = &export.events;
    let Some(room_version) = room_version_override.or_else(|| create_room_version(events)) else {
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
    // An event whose content hash fails is judged in its redacted form, unless
    // it is a stored event that was redacted on import.
    let redacted_buf;
    let lean_buf;
    let (redacted, lean): JudgeInputs<'_> = if let Some((redacted, lean)) = precomputed {
        (redacted, lean)
    } else {
        redacted_buf = redactions_for_export(events, room_version, already_redacted);
        lean_buf = lean_redacted_events(events, &redacted_buf, &export.room_id);
        (&redacted_buf, &lean_buf)
    };
    let create = create_position(events).and_then(|position| lean[position].as_ref());
    let exported: Vec<Option<bool>> = events
        .iter()
        .map(|event| exported_rejection(&event.metadata))
        .collect();
    let judgement = judge_events(events, lean, &index, create, version, judge_only)?;

    // Only a derived refusal seeds the cascade. The exported verdicts are
    // compared below for the report and are never stored.
    let signature_outcomes =
        keys.map(|keys| par_map(events, |event| keys.verify(event, room_version)));
    let seeds: Vec<bool> = judgement
        .refused
        .iter()
        .enumerate()
        .map(|(position, refusal)| {
            refusal.is_some()
                || prior_rejected.contains(&events[position].event_id)
                || matches!(
                    signature_outcomes
                        .as_ref()
                        .and_then(|outcomes| outcomes.get(position)),
                    Some(SignatureOutcome::Invalid(_))
                )
        })
        .collect();
    let mut findings = auth_findings_seeded(export, &seeds, judge_only)?;
    findings.hash_mismatched = events
        .iter()
        .zip(redacted)
        .filter(|(_, redacted)| redacted.is_some())
        .map(|(event, _)| event.event_id.clone())
        .collect();
    findings.hash_audit = audit_hash_mismatches(events, redacted, &export.room_id, room_version);
    findings.redactions_failed = events
        .iter()
        .zip(redacted)
        .filter(|(_, mismatch)| {
            mismatch
                .as_ref()
                .is_some_and(|mismatch| mismatch.redaction_failed)
        })
        .map(|(event, _)| event.event_id.clone())
        .collect();
    findings.redacted_pdus = redacted
        .iter()
        .map(|redacted| redacted.as_ref().map(|m| m.bytes.clone()))
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
    if require_create_authorized {
        ensure_create_authorized(export, events, &findings)?;
    }
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
/// (see [`derived_auth_findings_with_keys`]); the export's rejections are not trusted.
/// Load the missing key ids for `server` and return the requested cached keys.
fn memoized_server_keys(
    memo: &mut HashMap<String, Vec<mtxdb::server_keys::ServerKey>>,
    server: &str,
    key_ids: &[Vec<u8>],
    load: impl FnOnce(&[Vec<u8>]) -> anyhow::Result<Vec<mtxdb::server_keys::ServerKey>>,
) -> anyhow::Result<Vec<mtxdb::server_keys::ServerKey>> {
    let normalized_server = server.to_ascii_lowercase();
    let missing: Vec<_> = key_ids
        .iter()
        .filter(|key_id| {
            !memo.get(&normalized_server).is_some_and(|cached| {
                cached
                    .iter()
                    .any(|key| key.key_id.as_slice() == key_id.as_slice())
            })
        })
        .cloned()
        .collect();
    if !missing.is_empty() {
        let loaded = load(&missing)?;
        let cached = memo.entry(normalized_server.clone()).or_default();
        for key in loaded {
            if !cached.iter().any(|known| known.key_id == key.key_id) {
                cached.push(key);
            }
        }
    }
    let Some(cached) = memo.get(&normalized_server) else {
        return Ok(Vec::new());
    };
    Ok(cached
        .iter()
        .filter(|key| {
            key_ids
                .iter()
                .any(|key_id| key.key_id.as_slice() == key_id.as_slice())
        })
        .cloned()
        .collect())
}

#[cfg(test)]
mod memoized_server_keys_tests {
    use super::*;

    #[test]
    fn each_server_key_id_is_loaded_once_per_memo() {
        let mut memo = HashMap::new();
        let mut loaded_ids = Vec::new();
        let a1 = b"ed25519:a1".to_vec();
        let a2 = b"ed25519:a2".to_vec();
        let a3 = b"ed25519:a3".to_vec();
        let b1 = b"ed25519:b1".to_vec();
        let first = load_and_count(
            &mut memo,
            "a.example",
            &[a1.clone(), a2.clone()],
            &mut loaded_ids,
        );
        assert_eq!(first.len(), 2);
        let second = load_and_count(
            &mut memo,
            "A.EXAMPLE",
            std::slice::from_ref(&a1),
            &mut loaded_ids,
        );
        assert_eq!(second.len(), 1);
        load_and_count(
            &mut memo,
            "a.example",
            &[a1.clone(), a3.clone()],
            &mut loaded_ids,
        );
        load_and_count(
            &mut memo,
            "b.example",
            std::slice::from_ref(&b1),
            &mut loaded_ids,
        );
        assert_eq!(loaded_ids, vec![a1.clone(), a2, a3, b1.clone()]);

        memo.clear();
        load_and_count(
            &mut memo,
            "a.example",
            std::slice::from_ref(&a1),
            &mut loaded_ids,
        );
        assert_eq!(
            loaded_ids.last(),
            Some(&a1),
            "clearing memo reloads the key"
        );
    }

    fn test_server_key(key_id: &[u8]) -> mtxdb::server_keys::ServerKey {
        mtxdb::server_keys::ServerKey {
            key_id: key_id.to_vec(),
            key: Vec::new(),
            valid_until_ts: 0,
            source: Vec::new(),
            ts_added_ms: 0,
            tier: mtxdb::server_keys::KeyTier::Permanent,
            digest: [0; 32],
            first_seen_ms: 0,
            last_active_ms: 0,
            corroborated: false,
            promote_by_ms: 0,
            expired_ts: None,
            inferred_retired: false,
        }
    }

    fn load_and_count(
        memo: &mut HashMap<String, Vec<mtxdb::server_keys::ServerKey>>,
        server: &str,
        key_ids: &[Vec<u8>],
        loaded_ids: &mut Vec<Vec<u8>>,
    ) -> Vec<mtxdb::server_keys::ServerKey> {
        memoized_server_keys(memo, server, key_ids, |missing| {
            loaded_ids.extend_from_slice(missing);
            Ok(missing
                .iter()
                .map(|key_id| test_server_key(key_id))
                .collect())
        })
        .unwrap()
    }
}

#[cfg(test)]
pub(crate) fn write_room_with_findings(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
) -> anyhow::Result<(ImportCounts, AuthFindings)> {
    write_room_with_signature_keys(db, export, batch_bytes, None)
}

#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
#[cfg(test)]
pub(crate) fn write_room_with_signature_keys(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
    keys: Option<&SignatureKeyCatalog>,
) -> anyhow::Result<(ImportCounts, AuthFindings)> {
    let findings = derived_auth_findings_with_keys(export, keys)?;
    let written = write_room_with_resolved_findings(db, export, batch_bytes, &findings)?;
    Ok((written, findings))
}

/// Write a room using verdicts the caller has already finalized, including
/// check-5 rejections from the state walk.
#[allow(
    clippy::too_many_lines,
    reason = "the batch writer keeps one event's records together"
)]
pub(crate) fn write_room_with_resolved_findings(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
    findings: &AuthFindings,
) -> anyhow::Result<ImportCounts> {
    precheck_room_auth(export)?;
    let events_started = std::time::Instant::now();
    let room = export.room_id.as_bytes();
    let event_collection = room_event_collection_id(room);
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

    let room_scope = mtxdb::timeline::room_scope_id(room);
    let timeline_index =
        mtxdb::timeline::TimelineIndex::open(db.pool(EVENT_POOL).as_ref(), &export.room_id);
    let mut batch_timeline_entries = Vec::new();

    for (position, event) in export.events.iter().enumerate() {
        let event_id = event.event_id.as_bytes();
        let record = event_record_id(event_id);
        let prev = id_list(event, "prev_events")?;
        let auth = id_list(event, "auth_events")?;
        let edge_bytes: usize = prev.iter().chain(&auth).map(|id| id.len()).sum();
        let mut verdict = verdict_of(&event.metadata, findings.forced[position]);
        // Arrival ordering is a source-recorded fact that cannot be recomputed,
        // like `redacted_by`, so keep it in the verdict record.
        if let Some(arrival) = event.arrival_ordering {
            let arrival = u64::try_from(arrival).map_err(|_| {
                anyhow!(
                    "{}: `arrival_ordering` must be a non-negative integer",
                    event.event_id
                )
            })?;
            verdict.arrival_ordering = Some(arrival);
        }
        // A mismatched content hash means the effective event is its redaction;
        // the export keeps the original bytes.
        if findings.redactions_failed.contains(&event.event_id) {
            bail!(
                "{}: content hash mismatched but canonical redaction failed",
                event.event_id
            );
        }
        let stored: &[u8] = findings
            .redacted_pdus
            .get(position)
            .and_then(Option::as_deref)
            .unwrap_or(&event.pdu_bytes);
        let size = estimate(stored.len(), event, &export.room_id, &verdict, edge_bytes);
        if size > batch_bytes {
            bail!(
                "{}: about {size} bytes of records exceed the {batch_bytes}-byte batch limit",
                event.event_id
            );
        }
        if staged_bytes.saturating_add(size) > batch_bytes && staged_bytes > 0 {
            if written == 0 {
                bail!(
                    "room {} version record and first event need more than the {batch_bytes}-byte batch limit",
                    export.room_id
                );
            }
            if !batch_timeline_entries.is_empty() {
                timeline_index
                    .stage_append(&tx, EVENT_POOL, &batch_timeline_entries)
                    .context("staging room timeline index")?;
                batch_timeline_entries.clear();
            }
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
        // A verdict record is written for derived rejections and for the
        // facts kept from the source. An exported outlier keeps a record with
        // the annotation cleared, so a rebuild can see it was checked without
        // migrating the source's outlier claim.
        let verdict_collection = event_verdict_collection_id(&record);
        if !verdict.is_empty()
            || (event.metadata.outlier && exported_rejection(&event.metadata) != Some(true))
        {
            tx.put(
                EVENT_POOL,
                verdict_collection,
                record,
                &node(encode_verdict(event_id, &verdict)?),
            )?;
        } else if tx
            .get(EVENT_POOL, &verdict_collection, &[record])?
            .into_iter()
            .next()
            .flatten()
            .is_some()
        {
            // A re-import that now derives no verdict must not leave the
            // previous run's verdict for flag rebuilds to read.
            tx.remove(EVENT_POOL, verdict_collection, record)?;
        }

        let rejected = findings.forced.get(position).copied().flatten().is_some();
        if let Some((topological_ordering, stream_ordering)) =
            timeline_orderings(export, event, rejected)
        {
            let event_ref = mtxdb::timeline::event_id_ref(event_id);
            batch_timeline_entries.push(mtxdb::timeline::TimelineEntry::new(
                room_scope,
                topological_ordering,
                stream_ordering,
                record,
                event_ref,
            ));
        }

        staged_bytes = staged_bytes.saturating_add(size);
        written = written.saturating_add(1);
    }

    if !batch_timeline_entries.is_empty() {
        timeline_index
            .stage_append(&tx, EVENT_POOL, &batch_timeline_entries)
            .context("staging room timeline index")?;
    }
    tx.commit().context("committing an import batch")?;
    reclaim_journal(db)?;
    crate::cmd::report_import_phase(
        "write: events, locators, verdicts, timeline",
        events_started.elapsed(),
    );
    let (closures, shorts) = record_room_auth(db, export)?;
    let flags_started = std::time::Instant::now();
    write_room_flags(db, export, findings, &shorts)?;
    crate::cmd::report_import_phase("write: room flags", flags_started.elapsed());

    Ok(ImportCounts {
        events: written,
        closures,
    })
}

/// Populate the room's flag bitmap index for all imported events.
fn write_room_flags(
    db: &Database,
    export: &RoomExport,
    findings: &AuthFindings,
    shorts: &[u32],
) -> anyhow::Result<()> {
    const FORWARD_BATCH_SIZE: usize = 512;
    // Changes accumulated before a flush. Every flush rewrites each touched
    // bitmap chunk and commits through the journal, so the threshold is in
    // changes (six per event), not events: 1,024 flushed after every 512-event
    // chunk, one commit per 170 events' worth of work at 16k events.
    const FLUSH_CHANGES: usize = 1 << 16;

    let flag_index = RoomFlagIndex::new(export.room_id.as_bytes());
    let mut tx = db.begin_transaction();
    let mut batch_changes = Vec::new();
    let mut stage_time = std::time::Duration::ZERO;
    let mut commit_time = std::time::Duration::ZERO;
    let mut reclaim_time = std::time::Duration::ZERO;
    let mut flushes = 0_u32;

    for (chunk_start, chunk) in export.events.chunks(FORWARD_BATCH_SIZE).enumerate() {
        let base_pos = chunk_start.saturating_mul(FORWARD_BATCH_SIZE);
        for (offset, event) in chunk.iter().enumerate() {
            let position = base_pos.saturating_add(offset);
            let short = shorts[position];

            batch_changes.push(FlagChange::set(short, Flag::Present));

            let forced = findings.forced.get(position).copied().flatten();
            let verdict = verdict_of(&event.metadata, forced);
            // Re-importing an event must also clear derived flags the current
            // record no longer carries; a set-only update would leave a stale
            // verdict visible forever.
            for (flag, set) in [
                (Flag::Outlier, verdict.outlier),
                (Flag::SoftFailed, verdict.soft_fail.is_some()),
                (Flag::Rejected, verdict.rejection.is_some()),
                (Flag::Redacted, event.metadata.redacted_by.is_some()),
                (
                    Flag::Timeline,
                    timeline_orderings(export, event, forced.is_some()).is_some(),
                ),
            ] {
                batch_changes.push(if set {
                    FlagChange::set(short, flag)
                } else {
                    FlagChange::clear(short, flag)
                });
            }
        }

        if batch_changes.len() >= FLUSH_CHANGES {
            let started = std::time::Instant::now();
            flag_index.stage(&tx, &batch_changes)?;
            batch_changes.clear();
            stage_time = stage_time.saturating_add(started.elapsed());
            let started = std::time::Instant::now();
            tx.commit().context("committing room flag batch")?;
            commit_time = commit_time.saturating_add(started.elapsed());
            let started = std::time::Instant::now();
            reclaim_journal(db)?;
            reclaim_time = reclaim_time.saturating_add(started.elapsed());
            flushes = flushes.saturating_add(1);
            tx = db.begin_transaction();
        }
    }

    let started = std::time::Instant::now();
    if !batch_changes.is_empty() {
        flag_index.stage(&tx, &batch_changes)?;
    }
    stage_time = stage_time.saturating_add(started.elapsed());
    let started = std::time::Instant::now();
    tx.commit().context("committing final room flag batch")?;
    commit_time = commit_time.saturating_add(started.elapsed());
    let started = std::time::Instant::now();
    reclaim_journal(db)?;
    reclaim_time = reclaim_time.saturating_add(started.elapsed());
    flushes = flushes.saturating_add(1);
    crate::cmd::report_import_phase(
        &format!("write: room flags stage ({flushes} commits)"),
        stage_time,
    );
    crate::cmd::report_import_phase("write: room flags commit", commit_time);
    crate::cmd::report_import_phase("write: room flags journal reclaim", reclaim_time);
    Ok(())
}

/// Refuse a room with a repeated event id or an auth cycle before anything is
/// written, so a refused room leaves nothing behind.
fn precheck_room_auth(export: &RoomExport) -> anyhow::Result<()> {
    let events = &export.events;
    let mut index: HashMap<&str, usize> = HashMap::with_capacity(events.len());
    for (position, event) in events.iter().enumerate() {
        if index.insert(event.event_id.as_str(), position).is_some() {
            bail!(
                "{}: event {} appears more than once; refusing to import the room",
                export.room_id,
                event.event_id
            );
        }
    }
    let mut waiting = vec![0_usize; events.len()];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); events.len()];
    for (child, event) in events.iter().enumerate() {
        let mut seen = HashSet::new();
        for id in id_list(event, "auth_events")? {
            let Ok(id) = std::str::from_utf8(id) else {
                continue;
            };
            if let Some(&parent) = index.get(id) {
                if seen.insert(parent) {
                    waiting[child] = waiting[child].saturating_add(1);
                    children[parent].push(child);
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..events.len()).filter(|&i| waiting[i] == 0).collect();
    let mut next = 0;
    while let Some(&current) = order.get(next) {
        next = next.saturating_add(1);
        for &child in &children[current] {
            waiting[child] = waiting[child].saturating_sub(1);
            if waiting[child] == 0 {
                order.push(child);
            }
        }
    }
    if order.len() != events.len() {
        let stuck = (0..events.len()).find(|&i| waiting[i] > 0).unwrap_or(0);
        bail!(
            "{}: auth events form a cycle (through {}); refusing to import the room",
            export.room_id,
            events[stuck].event_id
        );
    }
    Ok(())
}

/// Record the room's `prev`/`auth` edges and build its auth closures through the
/// library's `RoomAuth`, the same path the raw importer uses. Returns how many
/// closures were written.
fn record_room_auth(db: &Database, export: &RoomExport) -> anyhow::Result<(u64, Vec<u32>)> {
    let room = RoomAuth::new(&export.room_id);
    let lists: Vec<(Vec<&str>, Vec<&str>)> = export
        .events
        .iter()
        .map(|event| {
            let ids = |field| -> anyhow::Result<Vec<&str>> {
                id_list(event, field)?
                    .into_iter()
                    .map(|id| std::str::from_utf8(id).map_err(Into::into))
                    .collect()
            };
            Ok((ids("prev_events")?, ids("auth_events")?))
        })
        .collect::<anyhow::Result<_>>()?;
    let record_started = std::time::Instant::now();
    let mut shorts = Vec::with_capacity(export.events.len());
    for (events, lists) in export
        .events
        .chunks(MAX_BATCH_EVENTS)
        .zip(lists.chunks(MAX_BATCH_EVENTS))
    {
        let records: Vec<EventRecord<'_>> = events
            .iter()
            .zip(lists)
            .map(|(event, (prev, auth))| EventRecord {
                event_id: &event.event_id,
                prev,
                auth,
                relation: None,
            })
            .collect();
        // Every chunk is deferred; the one sync below makes the room's whole
        // adjacency durable before this function returns, so a room never
        // pays one fsync per chunk. The import marker is committed only after
        // every room has returned, so it cannot precede these rows.
        let recorded = room
            .record_events_with(db, &records, AppendDurability::Deferred)
            .with_context(|| format!("recording auth adjacency for room {}", export.room_id))?;
        shorts.extend(recorded.iter().map(|r| r.short_id));
    }
    crate::cmd::report_import_phase("write: auth adjacency record", record_started.elapsed());
    let sync_started = std::time::Instant::now();
    room.sync_adjacency(db)
        .with_context(|| format!("syncing auth adjacency for room {}", export.room_id))?;
    crate::cmd::report_import_phase("write: auth adjacency sync", sync_started.elapsed());
    let rebuild_started = std::time::Instant::now();
    let report = room.rebuild(db)?;
    crate::cmd::report_import_phase("write: auth closure rebuild", rebuild_started.elapsed());
    Ok((u64::from(report.count), shorts))
}

/// Write a keys export in one transaction: a handful of records, far under
/// the batch limit.
pub(crate) fn write_keys(db: &Database, records: &[KeyRecord]) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for record in records {
        if let KeyRecord::Key { origin, key } = record {
            let binding = (origin.to_ascii_lowercase(), key.key_id.clone());
            anyhow::ensure!(
                seen.insert(binding),
                "duplicate signing key for origin {} and key id {}",
                origin,
                String::from_utf8_lossy(&key.key_id)
            );
        }
    }
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
    use mtxdb::server_keys::{key_digest, KeyTier, ServerKey};
    use rezzy::json::Object;
    use rezzy::signing::ed25519_zebra::SigningKey;

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
            stream_ordering: None,
            arrival_ordering: None,
            depth: None,
        }
    }

    fn export(events: Vec<RoomEvent>) -> RoomExport {
        RoomExport {
            server: "example.org".to_owned(),
            room_id: "!room:example.org".to_owned(),
            events,
            timeline: false,
        }
    }

    fn forced(findings: &AuthFindings, position: usize) -> Option<u8> {
        findings.forced[position]
    }

    fn signing_key_record(origin: &str, key_id: &str) -> KeyRecord {
        let key = SigningKey::from([17; 32]);
        let bytes = AsRef::<[u8]>::as_ref(&key.verification_key()).to_vec();
        KeyRecord::Key {
            origin: origin.to_owned(),
            key: ServerKey {
                key_id: key_id.as_bytes().to_vec(),
                digest: key_digest(&bytes),
                key: bytes,
                valid_until_ts: 100,
                source: b"test".to_vec(),
                ts_added_ms: 0,
                tier: KeyTier::Permanent,
                first_seen_ms: 0,
                last_active_ms: 0,
                corroborated: false,
                promote_by_ms: 0,
                expired_ts: None,
                inferred_retired: false,
            },
        }
    }

    const MISSING_AUTH: u8 = 4;
    const DEPENDS_ON_REJECTED: u8 = 2;

    fn room_event(json: &str) -> RoomEvent {
        let value = OwnedValue::parse_bytes(json.as_bytes()).expect("valid event JSON");
        RoomEvent::from_pdu(&value, "!room:test").expect("event has an id")
    }

    #[test]
    fn stored_projection_survives_content_hash_redaction() {
        let mut create = room_event(
            r#"{"event_id":"$create","sender":"@alice:test","room_id":"!room:test","type":"m.room.create","state_key":"","content":{"room_version":"10","m.federate":false},"prev_events":[],"auth_events":[]}"#,
        );
        let stored_hash = rezzy::compute_content_hash(&create.pdu, "10").expect("content hash");
        if let OwnedValue::Object(fields) = &mut create.pdu {
            // The stored create is already redacted: it kept only the
            // projected m.federate field, not room_version or other content.
            let mut content = Object::new();
            content.insert("m.federate".to_owned(), OwnedValue::Bool(false));
            fields.insert("content".to_owned(), OwnedValue::Object(content));
            let mut hashes = Object::new();
            hashes.insert("sha256".to_owned(), OwnedValue::String(stored_hash));
            fields.insert("hashes".to_owned(), OwnedValue::Object(hashes));
        }
        let rules = room_event(
            r#"{"event_id":"$rules","sender":"@alice:test","room_id":"!room:test","type":"m.room.join_rules","state_key":"","content":{"join_rule":"public"},"prev_events":["$create"],"auth_events":["$create"]}"#,
        );
        let join_bob = room_event(
            r#"{"event_id":"$join_bob","sender":"@bob:other","room_id":"!room:test","type":"m.room.member","state_key":"@bob:other","content":{"membership":"join"},"prev_events":["$rules"],"auth_events":["$create","$rules"]}"#,
        );
        let export = export(vec![create, rules, join_bob]);
        let already_redacted = HashSet::from(["$create".to_owned()]);

        let findings =
            derived_auth_findings_for_raw(&export, "10", &already_redacted, None, &HashSet::new())
                .unwrap();
        assert_eq!(
            forced(&findings, 2),
            Some(RejectionCode::AuthCheckFailed.to_u8()),
            "the projected m.federate must survive the stored-event hash check"
        );
    }

    #[test]
    fn a_complete_room_has_no_findings() {
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            event("$a", "m.room.message", &["$create"], &["$create"]),
        ]))
        .unwrap();
        assert_eq!(findings.forced, vec![None, None]);
        assert_eq!(findings.warnings(), Vec::<String>::new());
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
        assert_eq!(findings.cascaded, Vec::<String>::new());
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
        rejected.metadata.rejected = Some(true);
        rejected.metadata.rejection_code = Some(7);
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            rejected,
            event("$child", "m.room.message", &["$create"], &["$bad"]),
        ]))
        .unwrap();
        assert_eq!(findings.forced, vec![None, None, None]);
        assert_eq!(findings.cascaded, Vec::<String>::new());
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
        let mut message = with_field(
            with_field(
                event("$msg", "m.room.message", &["$create"], &["$create"]),
                "sender",
                "@stranger:example.org",
            ),
            "room_id",
            "!room:example.org",
        );
        message.metadata.rejected = Some(false);
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
        rejected.metadata.rejected = Some(true);
        rejected.metadata.rejection_code = Some(7);
        let findings = derived_auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            rejected,
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 2), None);
        assert_eq!(findings.passes_but_exported_rejected, ["$msg"]);
        assert_eq!(
            findings.fails_but_exported_accepted,
            Vec::<AuthFailure>::new()
        );
    }

    #[test]
    fn a_walk_rejection_removes_the_earlier_pass_disagreement() {
        let mut rejected = message(&["$create", "$join"]);
        rejected.metadata.rejected = Some(true);
        rejected.metadata.rejection_code = Some(7);
        let export = export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            rejected,
        ]);
        let mut findings = derived_auth_findings(&export).unwrap();
        assert_eq!(findings.passes_but_exported_rejected, ["$msg"]);

        let mut rejections = HashMap::new();
        rejections.insert(
            "$msg".to_owned(),
            (
                RejectionCode::AuthCheckFailed.to_u8(),
                "state before the event refused it".to_owned(),
            ),
        );
        apply_walk_rejections(&export, &mut findings, &rejections);

        assert_eq!(
            forced(&findings, 2),
            Some(RejectionCode::AuthCheckFailed.to_u8())
        );
        assert!(
            findings.passes_but_exported_rejected.is_empty(),
            "a stored rejection must not still be reported as stored accepted"
        );
    }

    #[test]
    fn two_auth_events_for_one_state_key_are_a_duplicate_rejection() {
        let mut msg = message(&["$create", "$join", "$join2"]);
        msg.metadata.rejected = Some(false);
        let findings = derived_auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            member("$join", "@alice:example.org", &["$create"]),
            member("$join2", "@alice:example.org", &["$create"]),
            msg,
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
        rejected.metadata.rejected = Some(true);
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
        rejected.metadata.rejected = Some(true);
        rejected.metadata.rejection_code = Some(7);
        let findings = derived_auth_findings(&export(vec![
            create_with_version("12"),
            member("$join", "@alice:example.org", &[]),
            rejected,
        ]))
        .unwrap();
        assert_eq!(forced(&findings, 2), None);
        assert_eq!(findings.passes_but_exported_rejected, ["$msg"]);
        assert_eq!(
            findings.fails_but_exported_accepted,
            Vec::<AuthFailure>::new()
        );
    }

    #[test]
    fn a_historical_uppercase_create_sender_is_not_rejected() {
        // Existing rooms may have uppercase localparts. The create rule must
        // accept them through rezzy's historical MXID validator.
        let create = with_field(create_with_version("10"), "sender", "@Milan:tchncs.de");
        let findings = derived_auth_findings(&export(vec![create])).unwrap();
        assert_eq!(forced(&findings, 0), None);
        assert_eq!(
            findings.fails_but_exported_accepted,
            Vec::<AuthFailure>::new()
        );
    }

    #[test]
    fn a_derived_create_failure_fails_the_room() {
        let create = with_field(create_with_version("10"), "sender", "not-an-mxid");
        let error = derived_auth_findings(&export(vec![create]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("m.room.create"), "{error}");
    }

    #[test]
    fn a_derived_create_without_a_sender_fails_the_room() {
        let mut create = create_with_version("10");
        if let OwnedValue::Object(fields) = &mut create.pdu {
            fields.remove("sender");
        }
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
        assert_eq!(findings.hash_mismatched, Vec::<String>::new());
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
        assert_eq!(findings.hash_mismatched, Vec::<String>::new());
    }

    #[test]
    fn an_unknown_room_version_derives_only_the_missing_auth_rejections() {
        let mut rejected = event("$msg", "m.room.message", &["$create"], &["$create"]);
        rejected.metadata.rejected = Some(true);
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
            rejected: Some(true),
            rejection_code: Some(7),
            soft_failed: Some(true),
            soft_fail_code: Some(2),
            outlier: true,
            ..EventMetadata::default()
        };
        let v = verdict_of(&exported, None);
        assert_eq!(v.rejection, None, "exported rejection is not migrated");
        assert!(!v.outlier, "exported outlier is not migrated");
        assert_eq!(v.soft_fail, Some(2), "exported soft-fail is kept");

        let v_forced = verdict_of(&exported, Some(MISSING_AUTH));
        assert_eq!(
            v_forced.rejection,
            Some(MISSING_AUTH),
            "derived rejection is stored"
        );
        assert!(v_forced.outlier, "derived rejection is stored as outlier");
        assert_eq!(
            v_forced.soft_fail, None,
            "a derived rejection suppresses the exported soft-fail"
        );
        assert!(verdict_of(&EventMetadata::default(), None).is_empty());

        let uncoded = EventMetadata {
            soft_failed: Some(true),
            ..EventMetadata::default()
        };
        assert_eq!(
            verdict_of(&uncoded, None).soft_fail,
            Some(SoftFailCode::Imported.to_u8()),
            "a soft-fail without a reason is stored as imported"
        );
    }

    fn db() -> (Database, std::path::PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        crate::sweep_stale_test_dirs();
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

    /// The chain exposed by the same library API used by the server.
    fn persisted(
        db: &Database,
        room: &str,
        id: &str,
    ) -> Option<std::collections::BTreeSet<String>> {
        RoomAuth::new(room)
            .auth_chain(db, id)
            .ok()
            .map(|chain| chain.into_iter().collect())
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
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn closures_survive_capped_batches_and_a_reimport_keeps_the_shorts() {
        let (db, dir) = db();
        let room = diamond();
        // About one event per batch.
        write_room_with_findings(&db, &room, 1200).unwrap();
        let before: Vec<_> = room
            .events
            .iter()
            .map(|e| persisted(&db, &room.room_id, &e.event_id))
            .collect();
        write_room_with_findings(&db, &room, BATCH_BYTES).unwrap();
        let after: Vec<_> = room
            .events
            .iter()
            .map(|e| persisted(&db, &room.room_id, &e.event_id))
            .collect();
        assert_eq!(before, after, "re-import preserves auth-chain results");
        for event in &room.events {
            assert_eq!(
                persisted(&db, &room.room_id, &event.event_id),
                Some(walked(&room, &event.event_id))
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_first_event_that_does_not_fit_with_room_version_writes_nothing() {
        let (db, dir) = db();
        let room = diamond();
        // 710 bytes fits the first event, but not the immutable room version
        // record plus that event. The failed import must not publish the
        // version record in a separate transaction.
        let result = write_room_with_findings(&db, &room, 710);
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("version record and first event"),
            "the first event should not fit beside its version record: {error}"
        );
        assert!(
            read_pool(
                &db,
                SERVER_INFO_POOL,
                room_version_collection_id(room.room_id.as_bytes()),
                room_version_node_id()
            )
            .is_none(),
            "the version record must remain uncommitted"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn duplicate_signing_key_records_are_rejected_before_writing() {
        let (db, dir) = db();
        let records = [
            signing_key_record("example.org", "ed25519:a"),
            signing_key_record("EXAMPLE.ORG", "ed25519:a"),
        ];
        let error = write_keys(&db, &records).unwrap_err().to_string();
        assert!(error.contains("duplicate signing key"), "{error}");
        assert!(
            server_keys::read_server_key(db.pool(SERVER_INFO_POOL), b"example.org", b"ed25519:a")
                .unwrap()
                .is_none(),
            "prevalidation must leave the transaction untouched"
        );
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
            reclaim_journal_with(
                segment_len,
                11,
                |_| true,
                |pool| {
                    synced.push(pool);
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(synced, Vec::<mtxdb::ShardType>::new());
    }

    #[test]
    fn every_pool_syncs_once_the_journal_passes_the_trigger() {
        for segment_len in [12, 13, u64::MAX] {
            let mut synced = Vec::new();
            reclaim_journal_with(
                segment_len,
                11,
                |_| true,
                |pool| {
                    synced.push(pool);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(synced, mtxdb::ShardType::ALL, "segment {segment_len}");
        }
    }

    #[test]
    fn only_the_pools_the_journal_is_waiting_on_are_synced() {
        let skipped = mtxdb::ShardType::ALL[0];
        let mut synced = Vec::new();
        reclaim_journal_with(
            100,
            11,
            |pool| pool != skipped,
            |pool| {
                synced.push(pool);
                Ok(())
            },
        )
        .unwrap();
        assert!(
            !synced.contains(&skipped),
            "the skipped pool was synced anyway"
        );
        assert_eq!(
            synced.len(),
            mtxdb::ShardType::ALL.len() - 1,
            "every other pool still syncs"
        );
    }

    #[test]
    fn a_failed_pool_sync_stops_the_import() {
        let mut synced = Vec::new();
        let error = reclaim_journal_with(
            100,
            1,
            |_| true,
            |pool| {
                synced.push(pool);
                if synced.len() == 2 {
                    anyhow::bail!("disk full");
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("disk full"));
        assert_eq!(synced.len(), 2, "no pool is synced after the failure");
    }

    /// A reclaim must do what it claims: shrink the segment back under the
    /// trigger and leave no pool blocking it. The reopen test proves the data
    /// survives, which is a different property and passed while the coverage
    /// path was broken.
    #[test]
    fn a_reclaim_shrinks_the_segment_and_clears_its_blockers() {
        crate::sweep_stale_test_dirs();
        let dir =
            std::env::temp_dir().join(format!("mtxdb-reclaim-outcome-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The trigger has to sit above an empty segment's header, or the
        // segment could never come back under it.
        let empty = {
            let db = Database::open(dir.clone()).unwrap();
            db.coordinator().segment_len().max(1)
        };
        let config = mtxdb::RuntimeConfig {
            reclaim_trigger_bytes: Some(empty),
            ..mtxdb::RuntimeConfig::default()
        };
        let db = Database::open_with_config(dir.clone(), config).unwrap();
        let collection = [0x5Au8; 16];
        let tx = db.begin_transaction();
        for index in 0..64u8 {
            let mut id = [0u8; 16];
            id[0] = index;
            tx.put(
                mtxdb::ShardType::EventDag,
                collection,
                id,
                &node(vec![7u8; 1024]),
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let before = db.coordinator().segment_len();
        assert!(
            before > db.coordinator().reclaim_trigger_len(),
            "records did not cross the reclaim trigger: {before}"
        );
        reclaim_journal(&db).unwrap();
        let after = db.coordinator().segment_len();
        assert!(
            after < before,
            "reclaim did not shrink the segment: {before} -> {after}"
        );
        assert!(
            after <= db.coordinator().reclaim_trigger_len(),
            "reclaim left the segment at {after}, still over the trigger"
        );
        assert!(
            db.coordinator().reclaim_blockers().is_empty(),
            "reclaim left pools blocking: {:?}",
            db.coordinator().reclaim_blockers()
        );
        drop(db);
        let _ = std::fs::remove_dir_all(dir);
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

    /// A room written past the reclaim trigger, reclaimed, survives a reopen with every event readable,
    /// both with and without the pre-marker barrier in between.
    #[test]
    fn events_survive_a_reopen_after_a_reclaim() {
        for barrier in [false, true] {
            crate::sweep_stale_test_dirs();
            let dir = std::env::temp_dir().join(format!(
                "mtxdb-reclaim-test-{}-{barrier}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            let config = mtxdb::RuntimeConfig {
                reclaim_trigger_bytes: Some(1),
                ..mtxdb::RuntimeConfig::default()
            };
            let room = diamond();
            {
                let db = Database::open_with_config(dir.clone(), config).unwrap();
                write_room_with_findings(&db, &room, 1200).unwrap();
                assert!(db.coordinator().segment_len() > db.coordinator().reclaim_trigger_len());
                reclaim_journal(&db).unwrap();
                if barrier {
                    sync_data(&db).unwrap();
                }
            }
            let db = Database::open(dir.clone()).unwrap();
            for id in ["$member", "$rules", "$power", "$create"] {
                let record = event_record_id(id.as_bytes());
                assert!(
                    read_pool(
                        &db,
                        EVENT_POOL,
                        room_event_collection_id(room.room_id.as_bytes()),
                        record
                    )
                    .is_some(),
                    "{id} is readable after reopen (barrier: {barrier})"
                );
            }
            let _ = std::fs::remove_dir_all(dir);
        }
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
        key_record_for(ORIGIN, sk, key_id, valid_until_ts, expired_ts)
    }

    fn key_record_for(
        origin: &str,
        sk: &SigningKey,
        key_id: &str,
        valid_until_ts: u64,
        expired_ts: Option<u64>,
    ) -> KeyRecord {
        let bytes: Vec<u8> = AsRef::<[u8]>::as_ref(&sk.verification_key()).to_vec();
        key_record_with_bytes(origin, key_id, bytes, valid_until_ts, expired_ts)
    }

    fn key_record_with_bytes(
        origin: &str,
        key_id: &str,
        bytes: Vec<u8>,
        valid_until_ts: u64,
        expired_ts: Option<u64>,
    ) -> KeyRecord {
        KeyRecord::Key {
            origin: origin.to_owned(),
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
            stream_ordering: None,
            arrival_ordering: None,
            depth: None,
        }
    }

    /// A create event at `ts`, signed under each `(key_id, signing key)`.
    /// Signing a key that is not the registered one makes a bad signature.
    fn signed_create(ts: u64, signers: &[(&str, &SigningKey)]) -> RoomEvent {
        let fields = format!(
            r#""type":"m.room.create","room_id":"!r:{ORIGIN}","sender":"@a:{ORIGIN}","state_key":"","origin_server_ts":{ts},"content":{{"room_version":"10"}},"depth":1,"auth_events":[],"prev_events":[]"#
        );
        let unsigned = OwnedValue::parse_bytes(format!("{{{fields}}}").as_bytes()).unwrap();
        let canonical = rezzy::try_canonical_redacted_json(&unsigned, "10").unwrap();
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
            event_id: "$create".to_owned(),
            metadata: EventMetadata::default(),
            prev_events: OwnedValue::Array(Vec::new()),
            auth_events: OwnedValue::Array(Vec::new()),
            pdu: OwnedValue::parse_bytes(signed.as_bytes()).unwrap(),
            pdu_bytes: Bytes::new(),
            stream_ordering: None,
            arrival_ordering: None,
            depth: None,
        }
    }

    /// A restricted-join event with signatures from arbitrary servers. Each
    /// signer is `(server, key_id, signing key)`.
    fn restricted_join(
        ts: u64,
        authoriser: &str,
        signers: &[(&str, &str, &SigningKey)],
    ) -> RoomEvent {
        let fields = format!(
            r#""type":"m.room.member","room_id":"!r:{ORIGIN}","sender":"@a:{ORIGIN}","state_key":"@a:{ORIGIN}","origin_server_ts":{ts},"content":{{"membership":"join","join_authorised_via_users_server":"{authoriser}"}},"depth":1,"auth_events":[],"prev_events":[]"#
        );
        let unsigned = OwnedValue::parse_bytes(format!("{{{fields}}}").as_bytes()).unwrap();
        let canonical = rezzy::try_canonical_redacted_json(&unsigned, "10").unwrap();
        let mut by_server: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for (server, key_id, sk) in signers {
            use base64::Engine as _;
            let signature = sk.sign(canonical.as_bytes());
            let encoded =
                base64::engine::general_purpose::STANDARD_NO_PAD.encode(signature.to_bytes());
            by_server
                .entry(server)
                .or_default()
                .push(format!(r#""{key_id}":"{encoded}""#));
        }
        let signatures = by_server
            .into_iter()
            .map(|(server, entries)| format!(r#""{server}":{{{}}}"#, entries.join(",")))
            .collect::<Vec<_>>()
            .join(",");
        let signed = format!(r#"{{{fields},"signatures":{{{signatures}}}}}"#);
        RoomEvent {
            event_id: "$m".to_owned(),
            metadata: EventMetadata::default(),
            prev_events: OwnedValue::Array(Vec::new()),
            auth_events: OwnedValue::Array(Vec::new()),
            pdu: OwnedValue::parse_bytes(signed.as_bytes()).unwrap(),
            pdu_bytes: Bytes::new(),
            stream_ordering: None,
            arrival_ordering: None,
            depth: None,
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
        assert!(matches!(outcome, SignatureOutcome::VerifiedLate(_)));
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
    fn a_bad_create_signature_forces_the_room_to_fail_closed() {
        let (signer, registered) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&registered, "ed25519:a", 2_000, None)]);
        let export = RoomExport {
            server: ORIGIN.to_owned(),
            room_id: format!("!r:{ORIGIN}"),
            events: vec![signed_create(1_000, &[("ed25519:a", &signer)])],
            timeline: false,
        };
        let error = derived_auth_findings_with_keys(&export, Some(&keys))
            .expect_err("a create event with an invalid signature must refuse the room");
        assert!(
            format!("{error:#}").contains("cannot be authorized"),
            "unexpected error: {error:#}"
        );
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
            SignatureOutcome::VerifiedLate(_)
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
                let keys = catalog(&[key_record(&sk, "ed25519:a", cached_deadline, Some(5_000))]);
                for ts in [4_999, 5_000] {
                    let event = signed_event_for(ts, &[("ed25519:a", &sk)], version);
                    assert!(
                        matches!(keys.verify(&event, version), SignatureOutcome::Verified(_)),
                        "v{version}, cached deadline {cached_deadline}, ts {ts}"
                    );
                    let expected = if version == "4" || ts <= cached_deadline {
                        "ed25519:a: in window".to_owned()
                    } else {
                        format!(
                            "ed25519:a: event after valid_until_ts {cached_deadline} (retired key remains valid through expired_ts 5000)"
                        )
                    };
                    assert_eq!(keys.explain(&event, version), vec![expected]);
                }
            }
        }
    }

    #[test]
    fn a_retired_key_is_unusable_one_millisecond_after_its_expiry() {
        let sk = signing_key(1);
        for version in ["4", "10"] {
            for cached_deadline in [2_000, 5_000, 9_000] {
                let keys = catalog(&[key_record(&sk, "ed25519:a", cached_deadline, Some(5_000))]);
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
                redaction_failed: false,
                signature_status: None,
            }],
            ..AuthFindings::default()
        };
        let outcomes = vec![SignatureOutcome::VerifiedLate(Vec::new())];
        apply_signature_outcomes(
            &mut findings,
            &events,
            outcomes,
            &keys,
            "!r:origin.example",
            "10",
        );
        assert_eq!(findings.signature_verified_late, 1);
        assert_eq!(findings.signature_verified_late, 1);
        assert_eq!(findings.signature_audit, Vec::<SignatureAudit>::new());
        assert_eq!(
            findings.hash_audit[0].signature_status,
            Some("verified_late")
        );
        assert!(
            findings.verified_keys.is_empty(),
            "late events do not corroborate"
        );
    }

    #[test]
    fn a_restricted_join_requires_and_accepts_the_authoriser_signature() {
        let (origin_sk, authoriser_sk) = (signing_key(1), signing_key(2));
        let keys = catalog(&[
            key_record(&origin_sk, "ed25519:a", 2_000, None),
            key_record_for("auth.example", &authoriser_sk, "ed25519:b", 2_000, None),
        ]);
        let event = restricted_join(
            1_000,
            "@admin:auth.example",
            &[
                ("origin.example", "ed25519:a", &origin_sk),
                ("auth.example", "ed25519:b", &authoriser_sk),
            ],
        );
        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::Verified(_)
        ));
    }

    #[test]
    fn a_bad_authoriser_signature_rejects_a_restricted_join() {
        let (origin_sk, authoriser_sk, wrong) = (signing_key(1), signing_key(2), signing_key(3));
        let keys = catalog(&[
            key_record(&origin_sk, "ed25519:a", 2_000, None),
            key_record_for("auth.example", &authoriser_sk, "ed25519:b", 2_000, None),
        ]);
        let event = restricted_join(
            1_000,
            "@admin:auth.example",
            &[
                ("origin.example", "ed25519:a", &origin_sk),
                ("auth.example", "ed25519:b", &wrong),
            ],
        );
        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::Invalid(_)
        ));
    }

    #[test]
    fn a_restricted_join_without_the_authoriser_key_is_deferred() {
        let (origin_sk, authoriser_sk) = (signing_key(1), signing_key(2));
        let keys = catalog(&[key_record(&origin_sk, "ed25519:a", 2_000, None)]);
        let event = restricted_join(
            1_000,
            "@admin:auth.example",
            &[
                ("origin.example", "ed25519:a", &origin_sk),
                ("auth.example", "ed25519:b", &authoriser_sk),
            ],
        );
        assert!(matches!(keys.verify(&event, "10"), SignatureOutcome::NoKey));
    }

    #[test]
    fn missing_authoriser_evidence_defers_an_origin_signature_failure() {
        let (origin_sk, wrong_origin_sk, authoriser_sk) =
            (signing_key(1), signing_key(3), signing_key(2));
        let keys = catalog(&[key_record(&origin_sk, "ed25519:a", 2_000, None)]);
        let event = restricted_join(
            1_000,
            "@admin:auth.example",
            &[
                ("origin.example", "ed25519:a", &wrong_origin_sk),
                ("auth.example", "ed25519:b", &authoriser_sk),
            ],
        );
        assert!(matches!(keys.verify(&event, "10"), SignatureOutcome::NoKey));
    }

    #[test]
    fn a_restricted_join_without_the_authoriser_signature_is_deferred() {
        let origin_sk = signing_key(1);
        let keys = catalog(&[key_record(&origin_sk, "ed25519:a", 2_000, None)]);
        let event = restricted_join(
            1_000,
            "@admin:auth.example",
            &[("origin.example", "ed25519:a", &origin_sk)],
        );
        assert!(matches!(keys.verify(&event, "10"), SignatureOutcome::NoKey));
    }

    #[test]
    fn an_authoriser_on_the_origin_does_not_need_a_second_signature() {
        let origin_sk = signing_key(1);
        let keys = catalog(&[key_record(&origin_sk, "ed25519:a", 2_000, None)]);
        let event = restricted_join(
            1_000,
            "@admin:origin.example",
            &[("origin.example", "ed25519:a", &origin_sk)],
        );
        assert!(matches!(
            keys.verify(&event, "10"),
            SignatureOutcome::Verified(_)
        ));
    }

    #[test]
    fn an_unparsable_key_is_explained_as_invalid_public_key_material() {
        use base64::Engine as _;
        let sk = signing_key(1);
        // A 32-byte value that is not a valid compressed Edwards point.
        let bytes = base64::engine::general_purpose::STANDARD
            .decode("LDxGK5kSPwLsvnrg9BS1wewISp91k+eG8U3Dq8GwVIs=")
            .unwrap();
        let keys = catalog(&[key_record_with_bytes(
            ORIGIN,
            "ed25519:a",
            bytes,
            2_000,
            None,
        )]);
        let event = signed_event(1_000, &[("ed25519:a", &sk)]);
        assert!(matches!(keys.verify(&event, "10"), SignatureOutcome::NoKey));
        assert_eq!(
            keys.explain(&event, "10"),
            vec!["ed25519:a: not a valid Ed25519 public key"]
        );
    }
}
