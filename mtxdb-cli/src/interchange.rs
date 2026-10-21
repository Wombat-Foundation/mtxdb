//! `mtxdb-export/v1`: the tagged JSONL an exporting server hands to `mtxdb import`.
//!
//! Each line is one JSON object tagged by `"t"`, written in dependency order:
//!
//! * `header` — the format tag, the exporting server, and the room id. It is
//!   the first line, so the importer can tell this input apart from a plain
//!   event stream before it reads anything else.
//! * `event` — the `event_id`, the `pdu` as the source stored it, the
//!   `prev_events` / `auth_events` lists, the `outlier` / `soft_failed` /
//!   `rejected` verdicts, and verdict codes or `redacted_by` when the source
//!   recorded them.
//! * `end` — the event count. A file without it was truncated.
//!
//! A keys export drops the framing: every line is a `server_keys` record with
//! an `origin` and a `kind`. A `provisional` or `permanent` line is one
//! signing key (`key_id`, base64 `key`, `valid_until_ts`, `source`,
//! `ts_added_ms`, and optionally `expired_ts` and, for a provisional key,
//! `promote_by_ms`); a `notary` line is the origin's self-signed payload, base64
//! of its exact bytes.
//!
//! # What the importer derives from this
//!
//! Locators, short ids, and auth-chain closures are deliberately absent: they
//! are derived data, rebuilt through the ordinary import path from each PDU's
//! own `prev_events` / `auth_events`. Two kinds of information have no place in
//! that path and are stored under the auxiliary names this module owns:
//!
//! * Event metadata — the verdicts and redaction bookkeeping above — goes to
//!   one index *per room*, keyed by event id, in the pool that holds the
//!   events. A room-scoped index is the whole point: an index shared across
//!   rooms makes every write carry everyone else's records (see the
//!   `put_state_hamt_roots` note in sithnapse's `mtxdb_syn.rs`), while per-room
//!   routing keeps one write's cost proportional to one room. Sparse verdicts
//!   also rule out sorted runs — they are point lookups on tiny values, so the
//!   engine's in-memory index is the right shape.
//! * Federation signing keys go to the server-info pool, one collection per
//!   origin holding one record per key id; the tier is a field of the key's
//!   record. An origin has a handful of keys and reads them as a unit, so a
//!   collection scan is the whole query and no index is kept. The notary
//!   payload is stored beside them, byte for byte: re-encoding it would
//!   invalidate its signature.
//!
//! The exported `pdu` bytes are stored exactly as written: redacting them again
//! would lose whatever the source already redacted, and re-encoding the parsed
//! JSON would not promise the same bytes back.

use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use bytes::Bytes;
use mtxdb::export_schema::{field, key_kind, tag};
use mtxdb::server_keys::{KeyTier, ServerKey};
use rezzy::json::{self, Object, Value as OwnedValue};

/// The export format this importer accepts, as written on the `header` line.
pub(crate) use mtxdb::export_schema::EXPORT_FORMAT;

/// What an input's first tagged line declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputFormat {
    /// A room export: `header`, `event`*, `end`.
    Room,
    /// A keys export: `server_keys` records only.
    Keys,
}


/// Everything an `event` record says about an event beyond the PDU itself.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct EventMetadata {
    /// The event is an outlier: outside the room's forward extremities.
    pub(crate) outlier: bool,
    /// The event failed auth and the server refused it.
    pub(crate) rejected: bool,
    /// The event failed auth checks against the current room state.
    pub(crate) soft_failed: bool,
    /// Why the event was rejected, as the source numbered it.
    pub(crate) rejection_code: Option<u8>,
    /// Why the event soft-failed, as the source numbered it.
    pub(crate) soft_fail_code: Option<u8>,
    /// The event that redacted this one, when the source recorded one.
    pub(crate) redacted_by: Option<String>,
}

/// One `event` record: the PDU as the source stored it, plus what the record
/// says around it.
#[derive(Debug)]
pub(crate) struct RoomEvent {
    /// The event's id as the source knows it.
    pub(crate) event_id: String,
    /// Verdict and redaction bookkeeping the PDU does not carry itself.
    pub(crate) metadata: EventMetadata,
    /// The record's `prev_events`, authoritative when the PDU carries none.
    pub(crate) prev_events: OwnedValue,
    /// The record's `auth_events`, authoritative when the PDU carries none.
    pub(crate) auth_events: OwnedValue,
    /// The PDU exactly as the export line wrote it.
    pub(crate) pdu: OwnedValue,
    /// The `pdu` value's exact bytes, stored verbatim.
    pub(crate) pdu_bytes: Bytes,
}

impl RoomEvent {
    /// The event as this importer's analysis path should see it: the PDU with
    /// the record's verdicts merged in and the identity fields supplied when
    /// the PDU carries none.
    ///
    /// Only the returned value is analyzed. [`Self::pdu_bytes`] is what gets
    /// stored, so these merged flags never reach the store.
    pub(crate) fn annotated(&self, room_id: &str) -> OwnedValue {
        let mut event = self.pdu.clone();
        let OwnedValue::Object(fields) = &mut event else {
            return event;
        };
        set_flag(fields, "rejected", self.metadata.rejected);
        set_flag(fields, "soft_failed", self.metadata.soft_failed);
        fields.insert(
            "event_id".to_owned(),
            OwnedValue::String(self.event_id.clone()),
        );
        fields
            .entry("room_id".to_owned())
            .or_insert_with(|| OwnedValue::String(room_id.to_owned()));
        if !matches!(fields.get("prev_events"), Some(OwnedValue::Array(_))) {
            fields.insert("prev_events".to_owned(), self.prev_events.clone());
        }
        if !matches!(fields.get("auth_events"), Some(OwnedValue::Array(_))) {
            fields.insert("auth_events".to_owned(), self.auth_events.clone());
        }
        event
    }
}

/// A parsed room export: its header and every `event` record.
#[derive(Debug)]
pub(crate) struct RoomExport {
    /// The server that produced the export.
    pub(crate) server: String,
    /// The room every record in the file belongs to.
    pub(crate) room_id: String,
    /// Every `event` record, in export order.
    pub(crate) events: Vec<RoomEvent>,
}

/// One `server_keys` record.
#[derive(Debug)]
pub(crate) enum KeyRecord {
    /// One signing key of an origin.
    Key {
        /// The server the key belongs to.
        origin: String,
        /// The key with its tier, source and timestamps.
        key: ServerKey,
    },
    /// The origin's self-signed payload, exactly as exported.
    Notary {
        /// The server the payload belongs to.
        origin: String,
        /// The payload bytes the export base64-encoded, decoded.
        payload: Bytes,
    },
}

/// Classify an input by its first tagged line, without reading the rest.
///
/// Returns `None` for anything that is not a tagged export, which leaves the
/// caller's other input formats to explain themselves.
pub(crate) fn detect(content: &[u8]) -> Option<InputFormat> {
    let text = std::str::from_utf8(content).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = OwnedValue::parse_bytes(line.as_bytes()).ok()?;
        return match value.get(field::T).and_then(OwnedValue::as_str) {
            Some(tag::HEADER) => Some(InputFormat::Room),
            Some(tag::SERVER_KEYS) => Some(InputFormat::Keys),
            _ => None,
        };
    }
    None
}

/// Parse a room export in full, validating its framing as it reads.
///
/// # Errors
/// Returns an error for a non-`mtxdb-export/v1` header, a record that is not a
/// JSON object or lacks its required fields, a verdict code outside `0..=255`,
/// a `pdu` whose identity disagrees with its record, a `prev_events` /
/// `auth_events` list that disagrees with the PDU's own, an `end` count that
/// does not match the records read, a record after `end`, or a missing `end`
/// (a truncated file).
pub(crate) fn parse_room(content: &[u8]) -> anyhow::Result<RoomExport> {
    let lines = tagged_lines(content)?;
    let mut lines = lines.into_iter();

    let (header_line, header_value, _) = lines
        .next()
        .ok_or_else(|| anyhow!("empty input; a room export starts with a `header` record"))?;
    let header = record_object(header_line, &header_value)?;
    if record_tag(header_line, header)? != tag::HEADER {
        bail!("line {header_line}: a room export starts with a `header` record");
    }
    let format = require_string(header_line, header, field::FORMAT)?;
    if format != EXPORT_FORMAT {
        bail!(
            "line {header_line}: unsupported export format `{format}`; expected `{EXPORT_FORMAT}`"
        );
    }
    let server = require_string(header_line, header, field::SERVER)?.to_owned();
    let room_id = require_string(header_line, header, field::ROOM_ID)?.to_owned();

    let mut events = Vec::new();
    let mut end = None;
    for (line, value, raw) in lines {
        let object = record_object(line, &value)?;
        if end.is_some() {
            bail!("line {line}: no record may follow the `end` record");
        }
        match record_tag(line, object)? {
            tag::EVENT => events.push(parse_event(line, object, raw, &room_id)?),
            tag::END => end = Some((line, require_number(line, object, field::EVENTS)?)),
            tag::HEADER => bail!("line {line}: a room export carries exactly one `header` record"),
            other => bail!("line {line}: unknown record tag `{other}`"),
        }
    }
    let Some((end_line, declared)) = end else {
        bail!("export has no `end` record; the file was truncated");
    };
    let read = u64::try_from(events.len()).context("event count overflow")?;
    if declared != read {
        bail!("line {end_line}: `end` declares {declared} events, but {read} were read");
    }
    Ok(RoomExport {
        server,
        room_id,
        events,
    })
}

/// Parse a keys export: every line a `server_keys` record, no framing.
///
/// # Errors
/// Returns an error for a line that is not a JSON object, is not tagged
/// `server_keys`, lacks a field its kind requires, names an unknown kind, or
/// carries key or payload bytes that are not base64 or are empty.
pub(crate) fn parse_keys(content: &[u8]) -> anyhow::Result<Vec<KeyRecord>> {
    let mut records = Vec::new();
    for (line, value, _) in tagged_lines(content)? {
        let object = record_object(line, &value)?;
        if record_tag(line, object)? != tag::SERVER_KEYS {
            bail!("line {line}: a keys export carries only `server_keys` records");
        }
        let origin = require_string(line, object, field::ORIGIN)?.to_owned();
        if origin.is_empty() {
            bail!("line {line}: `origin` is empty");
        }
        let record = match require_string(line, object, field::KIND)? {
            key_kind::NOTARY => KeyRecord::Notary {
                origin,
                payload: Bytes::from(require_base64(line, object, field::PAYLOAD)?),
            },
            kind @ (key_kind::PROVISIONAL | key_kind::PERMANENT) => {
                let key_id = require_string(line, object, field::KEY_ID)?;
                if key_id.is_empty() {
                    bail!("line {line}: `key_id` is empty");
                }
                let tier = if kind == key_kind::PROVISIONAL {
                    KeyTier::Provisional
                } else {
                    KeyTier::Permanent
                };
                KeyRecord::Key {
                    origin,
                    key: ServerKey {
                        key_id: key_id.as_bytes().to_vec(),
                        key: require_base64(line, object, field::KEY)?,
                        valid_until_ts: require_number(line, object, field::VALID_UNTIL_TS)?,
                        source: require_string(line, object, field::SOURCE)?
                            .as_bytes()
                            .to_vec(),
                        ts_added_ms: require_number(line, object, field::TS_ADDED_MS)?,
                        tier,
                        expired_ts: optional_number(line, object, field::EXPIRED_TS)?,
                        promote_by_ms: optional_number(line, object, field::PROMOTE_BY_MS)?
                            .unwrap_or(0),
                    },
                }
            }
            other => bail!("line {line}: unknown signing-key kind `{other}`"),
        };
        records.push(record);
    }
    if records.is_empty() {
        bail!("export has no `server_keys` records");
    }
    Ok(records)
}

/// A required, non-empty base64 field of a record, decoded.
fn require_base64(line: usize, object: &Object, field: &str) -> anyhow::Result<Vec<u8>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(require_string(line, object, field)?)
        .map_err(|error| anyhow!("line {line}: `{field}` is not base64: {error}"))?;
    if bytes.is_empty() {
        bail!("line {line}: `{field}` is empty");
    }
    Ok(bytes)
}

/// Parse every non-empty line of an export as a tagged record.
///
/// Returns the line number, the parsed object, and the line's own bytes (the
/// source a `pdu` value's exact bytes are cut from).
fn tagged_lines(content: &[u8]) -> anyhow::Result<Vec<(usize, OwnedValue, &[u8])>> {
    let text = std::str::from_utf8(content).context("input is not valid UTF-8")?;
    let mut lines = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line_number = index
            .checked_add(1)
            .context("export has more lines than this platform can number")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = OwnedValue::parse_bytes(line.as_bytes())
            .with_context(|| format!("line {line_number}: invalid JSON"))?;
        lines.push((line_number, value, line.as_bytes()));
    }
    Ok(lines)
}

/// Parse one `event` record against the header's room id.
fn parse_event(
    line: usize,
    object: &Object,
    raw: &[u8],
    room_id: &str,
) -> anyhow::Result<RoomEvent> {
    let event_id = require_string(line, object, field::EVENT_ID)?.to_owned();
    if event_id.is_empty() {
        bail!("line {line}: `event_id` is empty");
    }
    let prev_events = require_array(line, object, field::PREV_EVENTS)?.clone();
    let auth_events = require_array(line, object, field::AUTH_EVENTS)?.clone();
    let metadata = EventMetadata {
        outlier: require_bool(line, object, field::OUTLIER)?,
        rejected: require_bool(line, object, field::REJECTED)?,
        soft_failed: require_bool(line, object, field::SOFT_FAILED)?,
        rejection_code: optional_code(line, object, field::REJECTION_CODE)?,
        soft_fail_code: optional_code(line, object, field::SOFT_FAIL_CODE)?,
        redacted_by: match object.get(field::REDACTED_BY) {
            None => None,
            Some(value) => {
                let text = value
                    .as_str()
                    .ok_or_else(|| anyhow!("line {line}: `redacted_by` must be a string"))?;
                if text.is_empty() {
                    bail!("line {line}: `redacted_by` is empty");
                }
                Some(text.to_owned())
            }
        },
    };
    let pdu = match object.get(field::PDU) {
        Some(OwnedValue::Object(pdu)) => pdu,
        Some(_) => bail!("line {line}: `pdu` must be a JSON object"),
        None => bail!("line {line}: record is missing `pdu`"),
    };
    match pdu.get("event_id") {
        None => {}
        Some(OwnedValue::String(id)) if *id == event_id => {}
        Some(OwnedValue::String(_)) => {
            bail!("line {line}: `pdu.event_id` disagrees with the record's `event_id`");
        }
        Some(_) => bail!("line {line}: `pdu.event_id` must be a string"),
    }
    match pdu.get("room_id") {
        None => {}
        Some(OwnedValue::String(id)) if id == room_id => {}
        Some(OwnedValue::String(_)) => {
            bail!("line {line}: `pdu.room_id` does not match the exported room {room_id}");
        }
        Some(_) => bail!("line {line}: `pdu.room_id` must be a string"),
    }
    for field in ["prev_events", "auth_events"] {
        if let Some(OwnedValue::Array(from_pdu)) = pdu.get(field) {
            if let Some(OwnedValue::Array(from_record)) = object.get(field) {
                if from_pdu != from_record {
                    bail!("line {line}: `{field}` disagrees with the PDU's own list");
                }
            }
        }
    }
    Ok(RoomEvent {
        event_id,
        metadata,
        prev_events,
        auth_events,
        pdu: OwnedValue::Object(pdu.clone()),
        pdu_bytes: raw_pdu(line, raw)?,
    })
}

/// Cut the exact bytes of a record's `pdu` value out of its line.
///
/// The parsed `pdu` is kept for analysis and this copy is what gets stored, so
/// the store receives the export's bytes rather than a re-encoding of them.
fn raw_pdu(line_number: usize, raw: &[u8]) -> anyhow::Result<Bytes> {
    let mut tokenizer = json::Tokenizer::new(raw);
    let members = tokenizer
        .object_members()
        .map_err(|error| anyhow!("line {line_number}: reading record fields: {error}"))?;
    // A repeated key resolves to the last value, as JSON parsing does.
    members
        .into_iter()
        .rev()
        .find(|member| member.key.as_ref() == field::PDU)
        .map(|member| Bytes::copy_from_slice(member.raw_value))
        .ok_or_else(|| anyhow!("line {line_number}: record is missing `pdu`"))
}

/// Set a top-level disposition flag, replacing whatever spelling the PDU kept.
///
/// The record's verdict wins, because the policy table reads these flags off
/// the event value under several spellings (`rejected`, `soft_failed`,
/// `__soft-failed`) and only `true` reads as a verdict: a stale `true` in the
/// PDU must not survive a `false` record.
fn set_flag(fields: &mut Object, flag: &str, present: bool) {
    fields.retain(|key, _| normalize_flag(key) != flag);
    if present {
        fields.insert(flag.to_owned(), OwnedValue::Bool(true));
    }
}

/// The verdict an event field names: `__soft-failed` and `soft_failed` name
/// the same verdict to the policy table.
fn normalize_flag(key: &str) -> String {
    key.trim_start_matches('_').replace('-', "_")
}

/// A record object, or an error naming the line that was not one.
fn record_object(line: usize, value: &OwnedValue) -> anyhow::Result<&Object> {
    value
        .as_object()
        .ok_or_else(|| anyhow!("line {line}: record is not a JSON object"))
}

/// A record's `t` tag.
fn record_tag(line: usize, object: &Object) -> anyhow::Result<&str> {
    match object.get(field::T) {
        Some(OwnedValue::String(tag)) => Ok(tag),
        Some(_) => bail!("line {line}: `t` must be a string"),
        None => bail!("line {line}: record is missing its `t` tag"),
    }
}

/// A required string field of a record.
fn require_string<'a>(line: usize, object: &'a Object, field: &str) -> anyhow::Result<&'a str> {
    match object.get(field) {
        Some(OwnedValue::String(text)) => Ok(text),
        Some(_) => bail!("line {line}: `{field}` must be a string"),
        None => bail!("line {line}: record is missing `{field}`"),
    }
}

/// A required boolean field of a record.
fn require_bool(line: usize, object: &Object, field: &str) -> anyhow::Result<bool> {
    match object.get(field) {
        Some(OwnedValue::Bool(value)) => Ok(*value),
        Some(_) => bail!("line {line}: `{field}` must be a boolean"),
        None => bail!("line {line}: record is missing `{field}`"),
    }
}

/// A required array field of a record.
fn require_array<'a>(
    line: usize,
    object: &'a Object,
    field: &str,
) -> anyhow::Result<&'a OwnedValue> {
    match object.get(field) {
        Some(value @ OwnedValue::Array(_)) => Ok(value),
        Some(_) => bail!("line {line}: `{field}` must be an array"),
        None => bail!("line {line}: record is missing `{field}`"),
    }
}

/// A required non-negative integer field of a record.
fn require_number(line: usize, object: &Object, field: &str) -> anyhow::Result<u64> {
    match object.get(field) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| anyhow!("line {line}: `{field}` must be a non-negative integer")),
        None => bail!("line {line}: record is missing `{field}`"),
    }
}

/// An optional non-negative integer field of a record.
fn optional_number(line: usize, object: &Object, field: &str) -> anyhow::Result<Option<u64>> {
    object
        .get(field)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| anyhow!("line {line}: `{field}` must be a non-negative integer"))
        })
        .transpose()
}

/// An optional verdict code: absent, or an integer from 0 to 255.
fn optional_code(line: usize, object: &Object, field: &str) -> anyhow::Result<Option<u8>> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let number = value
        .as_u64()
        .ok_or_else(|| anyhow!("line {line}: `{field}` must be an integer from 0 to 255"))?;
    let code = u8::try_from(number)
        .map_err(|_| anyhow!("line {line}: `{field}` must be an integer from 0 to 255"))?;
    Ok(Some(code))
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    /// A two-event room export with a verdict and a redaction on one of them.
    fn room_export() -> String {
        let mut export = String::new();
        writeln!(
            export,
            "{{\"t\":\"header\",\"format\":\"{EXPORT_FORMAT}\",\"server\":\"example.org\",\"room_id\":\"!room:example.org\"}}"
        )
        .expect("writing to a String cannot fail");
        export.push_str(
            "{\"t\":\"event\",\"event_id\":\"$create\",\"prev_events\":[],\"auth_events\":[],\
             \"outlier\":true,\"soft_failed\":false,\"rejected\":false,\
             \"pdu\":{\"room_id\":\"!room:example.org\",\"event_id\":\"$create\",\
             \"type\":\"m.room.create\",\"sender\":\"@alice:example.org\",\"state_key\":\"\",\
             \"prev_events\":[],\"auth_events\":[],\"content\":{\"room_version\":\"11\"},\
             \"signatures\":{\"example.org\":{\"sha256\":\"abc\"}}}}\n",
        );
        export.push_str(
            "{\"t\":\"event\",\"event_id\":\"$msg\",\"prev_events\":[\"$create\"],\
             \"auth_events\":[\"$create\"],\"outlier\":false,\"soft_failed\":true,\
             \"soft_fail_code\":7,\"rejected\":true,\"rejection_code\":3,\
             \"redacted_by\":\"$redaction\",\"pdu\":{\"room_id\":\"!room:example.org\",\
             \"event_id\":\"$msg\",\"type\":\"m.room.message\",\"sender\":\"@bob:example.org\",\
             \"prev_events\":[\"$create\"],\"auth_events\":[\"$create\"],\
             \"content\":{\"body\":\"hello\"}}}\n",
        );
        export.push_str("{\"t\":\"end\",\"events\":2}\n");
        export
    }

    fn parse(content: &str) -> RoomExport {
        parse_room(content.as_bytes()).expect("export must parse")
    }

    /// The `header` line every malformed-export case starts from.
    fn header_line() -> String {
        format!(
            "{{\"t\":\"header\",\"format\":\"{EXPORT_FORMAT}\",\"server\":\"example.org\",\
             \"room_id\":\"!room:example.org\"}}\n"
        )
    }

    /// A complete `event` record: the required fields, the record's verdicts
    /// in `verdicts`, and `pdu` as the record's PDU.
    fn event_line(event_id: &str, verdicts: &str, pdu: &str) -> String {
        format!(
            "{{\"t\":\"event\",\"event_id\":\"{event_id}\",\"prev_events\":[],\
             \"auth_events\":[],\"outlier\":false,\"soft_failed\":false{verdicts},\
             \"pdu\":{pdu}}}\n"
        )
    }

    #[test]
    fn detect_reads_only_the_first_tagged_line() {
        assert_eq!(
            detect(room_export().as_bytes()),
            Some(InputFormat::Room),
            "a header line is a room export"
        );
        let keys = "{\"t\":\"server_keys\",\"origin\":\"example.org\",\"kind\":\"notary\",\
                    \"payload\":\"e30=\"}\n";
        assert_eq!(detect(keys.as_bytes()), Some(InputFormat::Keys));
        assert_eq!(detect(b"  \n{\"event_id\":\"$a\"}\n"), None);
        assert_eq!(detect(b"{}"), None, "a record without a tag is no export");
        assert_eq!(detect(b""), None);
    }

    #[test]
    fn parse_room_reads_header_and_records() {
        let export = parse(&room_export());
        assert_eq!(export.server, "example.org");
        assert_eq!(export.room_id, "!room:example.org");
        assert_eq!(export.events.len(), 2);

        let create = &export.events[0];
        assert_eq!(create.event_id, "$create");
        assert!(create.metadata.outlier);
        assert!(!create.metadata.rejected);
        assert!(create.metadata.outlier, "the outlier flag is metadata");

        let message = &export.events[1];
        assert_eq!(message.metadata.rejection_code, Some(3));
        assert_eq!(message.metadata.soft_fail_code, Some(7));
        assert_eq!(
            message.metadata.redacted_by.as_deref(),
            Some("$redaction"),
            "verdict codes and redaction bookkeeping must survive the round trip"
        );
        assert_eq!(
            message.pdu_bytes,
            Bytes::from_static(
                b"{\"room_id\":\"!room:example.org\",\"event_id\":\"$msg\",\
                  \"type\":\"m.room.message\",\"sender\":\"@bob:example.org\",\
                  \"prev_events\":[\"$create\"],\"auth_events\":[\"$create\"],\
                  \"content\":{\"body\":\"hello\"}}"
            ),
            "the PDU's exact bytes are what the store receives"
        );
    }

    #[test]
    fn annotated_merges_the_records_disposition_onto_the_pdu() {
        let export = parse(&room_export());
        let message = export.events[1].annotated(&export.room_id);
        assert!(matches!(
            message.get("rejected"),
            Some(OwnedValue::Bool(true))
        ));
        assert!(matches!(
            message.get("soft_failed"),
            Some(OwnedValue::Bool(true))
        ));
        let export = parse(&room_export());
        let create = export.events[0].annotated(&export.room_id);
        assert!(
            create.get("rejected").is_none(),
            "a clear verdict adds nothing"
        );
        assert_eq!(
            create.get("event_id"),
            Some(&OwnedValue::String("$create".to_owned()))
        );
    }

    #[test]
    fn annotated_replaces_a_stale_verdict_in_the_pdu() {
        let export = parse(&room_export());
        let mut fields = match &export.events[0].pdu {
            OwnedValue::Object(fields) => fields.clone(),
            _ => panic!("fixture pdu must be an object"),
        };
        fields.insert("rejected".to_owned(), OwnedValue::Bool(true));
        let mut event = RoomEvent {
            event_id: "$create".to_owned(),
            metadata: EventMetadata::default(),
            prev_events: OwnedValue::Array(Vec::new()),
            auth_events: OwnedValue::Array(Vec::new()),
            pdu: OwnedValue::Object(fields),
            pdu_bytes: Bytes::new(),
        };
        event.metadata.rejected = false;
        let annotated = event.annotated("!room:example.org");
        assert!(
            annotated.get("rejected").is_none(),
            "a false record must clear a stale `true` in the PDU"
        );
    }

    #[test]
    fn parse_room_rejects_a_malformed_export() {
        let header = header_line();
        let event = event_line("$a", ",\"rejected\":false", "{\"event_id\":\"$a\"}");
        let cases: [(&str, &str); 9] = [
            ("", "starts with a `header` record"),
            (
                "{\"t\":\"end\",\"events\":0}\n",
                "starts with a `header` record",
            ),
            (
                "{\"t\":\"header\",\"format\":\"other/v2\",\"server\":\"example.org\",\
                 \"room_id\":\"!room:example.org\"}\n",
                "unsupported export format",
            ),
            (
                &format!("{header}{{\"t\":\"end\",\"events\":0}}\n{event}"),
                "no record may follow",
            ),
            (
                &format!("{header}{{\"t\":\"event\"}}\n"),
                "record is missing",
            ),
            (
                &format!("{header}{event}{{\"t\":\"end\",\"events\":9}}\n"),
                "declares 9",
            ),
            (
                &format!("{header}{{\"t\":\"wat\"}}\n{{\"t\":\"end\",\"events\":0}}\n"),
                "unknown record tag",
            ),
            (&header, "was truncated"),
            (
                &format!(
                    "{header}{}",
                    event_line(
                        "$a",
                        ",\"rejected\":true,\"rejection_code\":900",
                        "{\"event_id\":\"$a\"}",
                    )
                ),
                "integer from 0 to 255",
            ),
        ];
        for (input, expected) in cases {
            let error =
                parse_room(input.as_bytes()).expect_err("malformed export must be rejected");
            let message = format!("{error:#}");
            assert!(
                message.contains(expected),
                "error {message:?} must mention {expected:?}"
            );
        }
    }

    #[test]
    fn parse_room_rejects_a_pdu_that_disagrees_with_its_record() {
        let header = header_line();
        let from_id = event_line("$a", ",\"rejected\":false", "{\"event_id\":\"$other\"}");
        let from_room = event_line(
            "$a",
            ",\"rejected\":false",
            "{\"event_id\":\"$a\",\"room_id\":\"!other:example.org\"}",
        );
        for (input, expected) in [
            (
                format!("{header}{from_id}"),
                "disagrees with the record's `event_id`",
            ),
            (
                format!("{header}{from_room}"),
                "does not match the exported room",
            ),
        ] {
            let error = parse_room(input.as_bytes()).expect_err("must reject the mismatch");
            assert!(
                format!("{error:#}").contains(expected),
                "error must mention {expected:?}"
            );
        }
    }

    #[test]
    fn parse_room_accepts_an_event_whose_pdu_omits_its_identity() {
        let mut export = String::new();
        writeln!(
            export,
            "{{\"t\":\"header\",\"format\":\"{EXPORT_FORMAT}\",\"server\":\"example.org\",\
             \"room_id\":\"!room:example.org\"}}"
        )
        .expect("writing to a String cannot fail");
        export.push_str(
            "{\"t\":\"event\",\"event_id\":\"$a\",\"prev_events\":[\"$b\"],\
             \"auth_events\":[],\"outlier\":false,\"soft_failed\":false,\"rejected\":false,\
             \"pdu\":{\"type\":\"m.room.message\"}}\n{\"t\":\"end\",\"events\":1}\n",
        );
        let export = parse(&export);
        let annotated = export.events[0].annotated(&export.room_id);
        assert_eq!(
            annotated.get("event_id"),
            Some(&OwnedValue::String("$a".to_owned()))
        );
        assert_eq!(
            annotated.get("room_id"),
            Some(&OwnedValue::String("!room:example.org".to_owned()))
        );
        assert_eq!(
            annotated.get("prev_events"),
            Some(&OwnedValue::Array(vec![OwnedValue::String(
                "$b".to_owned()
            )])),
            "the record's list is authoritative when the PDU carries none"
        );
    }

    const KEY_LINE: &str = "{\"t\":\"server_keys\",\"origin\":\"example.org\",\"kind\":\"provisional\",\
        \"key_id\":\"ed25519:A\",\"key\":\"AQID\",\"valid_until_ts\":99,\
        \"source\":\"notary.example\",\"ts_added_ms\":7}\n";

    #[test]
    fn parse_keys_reads_keys_and_the_verbatim_notary_payload() {
        let notary = "{\"t\":\"server_keys\",\"origin\":\"example.org\",\"kind\":\"notary\",\
                      \"payload\":\"eyJrZXlzIjp7fX0=\"}\n";
        let permanent = KEY_LINE.replace("provisional", "permanent");
        let export = format!("{KEY_LINE}{permanent}{notary}");
        let records = parse_keys(export.as_bytes()).expect("keys export must parse");
        assert_eq!(records.len(), 3);
        let KeyRecord::Key { origin, key } = &records[0] else {
            panic!("the first line is a key");
        };
        assert_eq!(origin, "example.org");
        assert_eq!(key.key_id, b"ed25519:A");
        assert_eq!(key.key, [1, 2, 3], "the key is base64-decoded");
        assert_eq!(
            (key.valid_until_ts, key.ts_added_ms, key.tier),
            (99, 7, KeyTier::Provisional)
        );
        assert_eq!(key.source, b"notary.example");
        let KeyRecord::Key { key, .. } = &records[1] else {
            panic!("the second line is a key");
        };
        assert_eq!(key.tier, KeyTier::Permanent);
        let KeyRecord::Notary { payload, .. } = &records[2] else {
            panic!("the third line is the notary payload");
        };
        assert_eq!(
            *payload,
            Bytes::from_static(b"{\"keys\":{}}"),
            "the base64 payload must decode to its original bytes"
        );
    }

    #[test]
    fn parse_keys_rejects_a_bad_record() {
        let without = |field: &str| {
            let members = [
                ("origin", "\"example.org\""),
                ("kind", "\"provisional\""),
                ("key_id", "\"ed25519:A\""),
                ("key", "\"AQID\""),
                ("valid_until_ts", "99"),
                ("source", "\"notary.example\""),
                ("ts_added_ms", "7"),
            ];
            let body: Vec<String> = members
                .iter()
                .filter(|(name, _)| *name != field)
                .map(|(name, value)| format!("\"{name}\":{value}"))
                .collect();
            format!("{{\"t\":\"server_keys\",{}}}\n", body.join(","))
        };
        let cases: Vec<(String, &str)> = vec![
            (String::new(), "no `server_keys` records"),
            ("{\"t\":\"header\"}\n".to_owned(), "carries only `server_keys`"),
            (
                KEY_LINE.replace("provisional", "historical"),
                "unknown signing-key kind",
            ),
            (KEY_LINE.replace("AQID", "!!"), "not base64"),
            (KEY_LINE.replace("\"AQID\"", "\"\""), "`key` is empty"),
            (
                KEY_LINE.replace("example.org", ""),
                "`origin` is empty",
            ),
            (KEY_LINE.replace("ed25519:A", ""), "`key_id` is empty"),
            (without("key"), "missing `key`"),
            (without("valid_until_ts"), "missing `valid_until_ts`"),
            (without("source"), "missing `source`"),
            (without("ts_added_ms"), "missing `ts_added_ms`"),
            (
                "{\"t\":\"server_keys\",\"origin\":\"o\",\"kind\":\"notary\",\"payload\":\"!!\"}\n"
                    .to_owned(),
                "not base64",
            ),
        ];
        for (input, expected) in cases {
            let error = parse_keys(input.as_bytes()).expect_err("bad keys export");
            let message = format!("{error:#}");
            assert!(
                message.contains(expected),
                "error {message:?} must mention {expected:?}"
            );
        }
    }
}
