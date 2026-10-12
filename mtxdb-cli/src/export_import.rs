//! Writing an `mtxdb-export/v1` file into the server's own layouts.
//!
//! Every record goes where `mtxdb::interop` says the server reads it: the
//! room's event, prev and auth collections, the bucketed locator and verdict
//! collections, and the signing-key collection. Nothing here invents a
//! layout. State, short ids and auth closures are derived by the server and
//! are not written.
//!
//! A run is complete only when the import marker says so. Before the first
//! record of a run the marker is cleared, data goes in capped batches, and the
//! marker is committed last, in a transaction of its own. A run that stops
//! anywhere earlier leaves the database visibly incomplete.

use std::collections::HashMap;

use anyhow::{Context, anyhow, bail};
use mtxdb::export_schema::{RejectionCode, SoftFailCode};
use mtxdb::interop::{
    DATA_FORMAT, EDGE_POOL, EVENT_POOL, ImportMarker, SERVER_INFO_POOL, Verdict,
    decode_event_locator, decode_import_marker, encode_edges, encode_event_locator,
    encode_import_marker, encode_server_key, encode_verdict, event_locator_collection_id,
    event_record_id, event_verdict_collection_id, import_marker_collection_id,
    import_marker_node_id, room_auth_collection_id, room_event_collection_id,
    room_prev_collection_id, server_key_record_id, server_keys_collection_id,
};
use mtxdb::storage::StorageEngine as _;
use mtxdb::{Database, NodeData};
use rezzy::json::Value as OwnedValue;

use crate::interchange::{EventMetadata, KeyRecord, RoomEvent, RoomExport};

/// Bytes one batch may stage. Half the engine's per-transaction limit, so the
/// estimate below can be rough.
pub(crate) const BATCH_BYTES: usize = 32 << 20;

/// Allowance per staged record for keys and framing, on top of its value.
const RECORD_OVERHEAD: usize = 128;

fn node(bytes: Vec<u8>) -> NodeData {
    NodeData::new(bytes::Bytes::from(bytes))
}

/// Whether the database holds a marker, and what it says.
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
pub(crate) fn clear_marker(db: &Database) -> anyhow::Result<()> {
    if read_marker(db)?.is_none() {
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

/// Commit the marker, last and alone. `events` counts the event records this
/// run wrote.
pub(crate) fn write_marker(db: &Database, events: u64) -> anyhow::Result<()> {
    let tx = db.begin_transaction();
    tx.put(
        SERVER_INFO_POOL,
        import_marker_collection_id(),
        import_marker_node_id(),
        &node(encode_import_marker(ImportMarker {
            data_format: DATA_FORMAT,
            events,
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

fn verdict_of(metadata: &EventMetadata, forced: Option<u8>) -> Verdict {
    let mut verdict = Verdict {
        rejection: (metadata.rejected || metadata.rejection_code.is_some()).then(|| {
            metadata
                .rejection_code
                .unwrap_or(RejectionCode::Unknown.to_u8())
        }),
        soft_fail: (metadata.soft_failed || metadata.soft_fail_code.is_some()).then(|| {
            metadata
                .soft_fail_code
                .unwrap_or(SoftFailCode::Unknown.to_u8())
        }),
        redacted_by: metadata
            .redacted_by
            .as_ref()
            .map(|redactor| redactor.as_bytes().to_vec()),
        // Rejection and soft-fail status do not imply outlier status. Preserve
        // the export's explicit outlier flag; only importer-forced rejection
        // below changes it.
        outlier: metadata.outlier || metadata.rejected || metadata.rejection_code.is_some(),
    };
    // An event whose auth chain the import cannot establish is stored as a
    // rejected outlier. An exported rejection is never overwritten.
    if let Some(code) = forced {
        if verdict.rejection.is_none() {
            verdict.rejection = Some(code);
        }
        verdict.outlier = true;
    }
    verdict
}

fn exported_rejection(metadata: &EventMetadata) -> bool {
    metadata.rejected || metadata.rejection_code.is_some()
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
}

impl AuthFindings {
    /// Human-readable warnings, empty for a clean room.
    pub(crate) fn warnings(&self) -> Vec<String> {
        const LISTED: usize = 10;
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
        ] {
            if count == 0 {
                continue;
            }
            let shown: Vec<&str> = ids.iter().take(LISTED).map(String::as_str).collect();
            let more = count.saturating_sub(shown.len());
            let tail = if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            };
            lines.push(format!("{count} events {what}: {}{tail}", shown.join(", ")));
        }
        lines
    }
}

/// Work out which events cannot be authorized from the room's own edges.
///
/// An event with an auth event missing from the export is forced to
/// `MissingAuthEvent`; an event citing a rejected auth event (exported or
/// forced) is forced to `DependsOnRejectedAuthEvent`, transitively. An event
/// that already carries a rejection keeps it. The room fails as a whole when
/// its `m.room.create` would be forced.
pub(crate) fn auth_findings(export: &RoomExport) -> anyhow::Result<AuthFindings> {
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
    let mut rejected: Vec<bool> = events
        .iter()
        .map(|event| exported_rejection(&event.metadata))
        .collect();
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
        let is_create = event.pdu.get("type").and_then(OwnedValue::as_str) == Some("m.room.create");
        if is_create && findings.forced[position].is_some() {
            bail!(
                "refusing to import events for {}: its m.room.create event {} cannot be \
                 authorized from this export",
                export.room_id,
                event.event_id
            );
        }
    }
    Ok(findings)
}

/// Bytes an event's records will stage, with margin.
fn estimate(event: &RoomEvent, room_id: &str, verdict: &Verdict, edges: usize) -> usize {
    let verdict_bytes = if verdict.is_empty() {
        0
    } else {
        RECORD_OVERHEAD
            .saturating_add(event.event_id.len())
            .saturating_add(verdict.redacted_by.as_ref().map_or(0, Vec::len))
    };
    event
        .pdu_bytes
        .len()
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
pub(crate) fn write_room_with_findings(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
) -> anyhow::Result<(u64, AuthFindings)> {
    let findings = auth_findings(export)?;
    let room = export.room_id.as_bytes();
    let event_collection = room_event_collection_id(room);
    let prev_collection = room_prev_collection_id(room);
    let auth_collection = room_auth_collection_id(room);

    let mut written = 0_u64;
    let mut staged_bytes = 0_usize;
    // Event ids staged by the open batch, by record id: two different ids
    // sharing a record id must not shadow each other. Committed batches are
    // covered by reading the locator back through the transaction.
    let mut batch_ids: HashMap<[u8; 16], &str> = HashMap::new();
    let mut tx = db.begin_transaction();

    for (position, event) in export.events.iter().enumerate() {
        let event_id = event.event_id.as_bytes();
        let record = event_record_id(event_id);
        let prev = id_list(event, "prev_events")?;
        let auth = id_list(event, "auth_events")?;
        let prev_bytes = encode_edges(&prev)?;
        let auth_bytes = encode_edges(&auth)?;
        let verdict = verdict_of(&event.metadata, findings.forced[position]);
        let size = estimate(
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
        let existing = tx
            .get(EVENT_POOL, &locator_collection, &[record])?
            .into_iter()
            .next()
            .flatten();
        if let Some(existing) = existing {
            let (stored, _) = decode_event_locator(&existing.bytes)?;
            if stored != event_id {
                bail!(
                    "{}: record id collides with a stored event; refusing to shadow it",
                    event.event_id
                );
            }
        }

        tx.put(
            EVENT_POOL,
            event_collection,
            record,
            &node(event.pdu_bytes.to_vec()),
        )?;
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
    Ok((written, findings))
}

/// Write a keys export in one transaction: a handful of records, far under
/// the batch limit.
pub(crate) fn write_keys(db: &Database, records: &[KeyRecord]) -> anyhow::Result<()> {
    let tx = db.begin_transaction();
    for record in records {
        let origin = record.origin.as_bytes();
        tx.put(
            SERVER_INFO_POOL,
            server_keys_collection_id(),
            server_key_record_id(record.kind, origin),
            &node(encode_server_key(origin, &record.payload)?),
        )?;
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
    }

    #[test]
    fn an_exported_rejection_keeps_its_code_and_still_cascades() {
        let mut rejected = event("$bad", "m.room.message", &["$create"], &["$gone"]);
        rejected.metadata.rejected = true;
        rejected.metadata.rejection_code = Some(7);
        let findings = auth_findings(&export(vec![
            event("$create", "m.room.create", &[], &[]),
            rejected,
            event("$child", "m.room.message", &["$create"], &["$bad"]),
        ]))
        .unwrap();
        assert_eq!(
            forced(&findings, 1),
            None,
            "an exported rejection is not overwritten"
        );
        assert_eq!(forced(&findings, 2), Some(DEPENDS_ON_REJECTED));
        assert!(findings.missing_auth.is_empty());
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

    #[test]
    fn a_forced_rejection_stores_a_rejected_outlier_and_keeps_other_verdicts() {
        let verdict = verdict_of(&EventMetadata::default(), Some(MISSING_AUTH));
        assert_eq!(verdict.rejection, Some(MISSING_AUTH));
        assert!(verdict.outlier);

        let soft = EventMetadata {
            soft_failed: true,
            soft_fail_code: Some(2),
            ..EventMetadata::default()
        };
        let verdict = verdict_of(&soft, Some(MISSING_AUTH));
        assert_eq!(verdict.soft_fail, Some(2), "a soft-fail verdict is kept");
        assert_eq!(verdict.rejection, Some(MISSING_AUTH));

        let rejected = EventMetadata {
            rejected: true,
            rejection_code: Some(7),
            ..EventMetadata::default()
        };
        assert_eq!(verdict_of(&rejected, Some(MISSING_AUTH)).rejection, Some(7));
        assert!(verdict_of(&EventMetadata::default(), None).is_empty());
    }
}
