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

use anyhow::{anyhow, bail, Context};
use mtxdb::interop::{
    decode_event_locator, decode_import_marker, encode_edges, encode_event_locator,
    encode_import_marker, encode_server_key, encode_verdict, event_locator_collection_id,
    event_record_id, event_verdict_collection_id, import_marker_collection_id,
    import_marker_node_id, room_auth_collection_id, room_event_collection_id,
    room_prev_collection_id, server_key_record_id, server_keys_collection_id, ImportMarker,
    Verdict, DATA_FORMAT, EDGE_POOL, EVENT_POOL, SERVER_INFO_POOL,
};
use mtxdb::export_schema::{RejectionCode, SoftFailCode};
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

fn verdict_of(metadata: &EventMetadata) -> Verdict {
    Verdict {
        rejection: (metadata.rejected || metadata.rejection_code.is_some())
            .then(|| metadata.rejection_code.unwrap_or(RejectionCode::Unknown.to_u8())),
        soft_fail: (metadata.soft_failed || metadata.soft_fail_code.is_some())
            .then(|| metadata.soft_fail_code.unwrap_or(SoftFailCode::Unknown.to_u8())),
        redacted_by: metadata
            .redacted_by
            .as_ref()
            .map(|redactor| redactor.as_bytes().to_vec()),
    }
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

/// Write a room's events, edges, locators and verdicts in capped batches.
///
/// All of one event's records share a transaction. Returns how many event
/// records were written. The marker is the caller's to commit.
pub(crate) fn write_room(
    db: &Database,
    export: &RoomExport,
    batch_bytes: usize,
) -> anyhow::Result<u64> {
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

    for event in &export.events {
        let event_id = event.event_id.as_bytes();
        let record = event_record_id(event_id);
        let prev = id_list(event, "prev_events")?;
        let auth = id_list(event, "auth_events")?;
        let prev_bytes = encode_edges(&prev)?;
        let auth_bytes = encode_edges(&auth)?;
        let verdict = verdict_of(&event.metadata);
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

        tx.put(EVENT_POOL, event_collection, record, &node(event.pdu_bytes.to_vec()))?;
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
    Ok(written)
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
