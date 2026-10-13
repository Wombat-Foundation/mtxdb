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

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context};
use mtxdb::export_schema::{RejectionCode, SoftFailCode};
use mtxdb::interop::{
    auth_closure_record_id, auth_counter_record_id, auth_forward_record_id, auth_reverse_record_id,
    decode_auth_counter, decode_auth_forward, decode_event_locator, decode_import_marker,
    encode_auth_forward, encode_edges, encode_event_locator, encode_import_marker,
    encode_server_key, encode_verdict, event_locator_collection_id, event_record_id,
    event_verdict_collection_id, import_marker_collection_id, import_marker_node_id,
    room_auth_collection_id, room_event_collection_id, room_prev_collection_id,
    server_key_record_id, server_keys_collection_id, ImportMarker, Verdict, DATA_FORMAT, EDGE_POOL,
    EVENT_POOL, SERVER_INFO_POOL,
};
use mtxdb::storage::StorageEngine as _;
use mtxdb::{Database, NodeData};
use rezzy::bitmap::Bitmap;
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
) -> anyhow::Result<(ImportCounts, AuthFindings)> {
    let findings = auth_findings(export)?;
    let plan = closure_plan(export)?;
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
}
