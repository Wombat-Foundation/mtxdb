//! `mtxdb auth-check`: find auth targets that have no outbound edge record.
//!
//! A homeserver that keeps a room's auth graph in the state pool (sithnapse's
//! embedded engine) stores, under one `AUTH` collection per room:
//!
//! - a `u64` big-endian short-id counter,
//! - `event_id -> short_id` and `short_id -> event_id` maps, and
//! - one outbound edge record per *persisted* state event:
//!   `[u8 version = 1][u32 count][u32 auth_short_id; count]`, all big-endian.
//!
//! Every auth id an event cites is allocated a short id, even when that event
//! was never stored (for example, a PDU dropped for a bad signature). Such a
//! target has a short id but no edge record. A closure walk must treat that as
//! an incomplete graph, so this command lists them, together with the events
//! that cite them.
//!
//! The edge table is derived data. The authoritative graph is each persisted
//! event's own `auth_events`, kept in the room's `EventDag` collection, so the
//! check also re-derives every event's auth edges from its record, iterating
//! the real event ids from the reverse map, and compares them with the edge
//! table: a truncated or extra edge list is reported. A second scan lists
//! persisted event records that no short id names.
//!
//! The record layout belongs to the homeserver, not to this crate: the key
//! derivations and encoding below mirror sithnapse's
//! `rust/src/database/mtxdb_syn.rs` (`short_id_*_node_id`,
//! `encode_auth_edges`, `_encode_event_json_record`). If that layout changes, this check must change with
//! it; a version mismatch is reported rather than guessed at.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

const EDGE_ENCODING_VERSION: u8 = 1;

/// The event format tags the homeserver stores ahead of each event record: its
/// fixed `EventFormatVersions` table (1 room v1/v2, 2 v3, 3 v4+, 4 v11 hydra+,
/// 5 MSC4242), plus -1 for a record stored with no format version.
const EVENT_RECORD_FORMAT_VERSIONS: [i32; 6] = [-1, 1, 2, 3, 4, 5];

/// The MSC4242 event format tag (sithnapse's `ROOM_VMSC4242`). MSC4242 removes
/// `auth_events` from the PDU, so the record's own auth list is only in its
/// internal metadata.
pub(crate) const MSC4242_FORMAT_VERSION: i32 = 5;

type NodeId = [u8; 16];

fn node_id(parts: &[&[u8]]) -> NodeId {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

pub(crate) fn counter_node_id() -> NodeId {
    node_id(&[b"authchain:counter"])
}

pub(crate) fn edge_node_id(short_id: u32) -> NodeId {
    node_id(&[b"authchain:edges:", &short_id.to_be_bytes()])
}

pub(crate) fn reverse_node_id(short_id: u32) -> NodeId {
    node_id(&[b"authchain:rev:", &short_id.to_be_bytes()])
}

/// The node id sithnapse stores an event's auth short id under in the room's
/// `AUTH` collection.
pub(crate) fn forward_node_id(event_id: &str) -> NodeId {
    node_id(&[b"authchain:fwd:", event_id.as_bytes()])
}

/// The JSON bytes of an event record:
/// `[i32 format_version][u32 metadata_len][internal_metadata][json]`, big-endian.
pub(crate) fn event_record_json(bytes: &[u8]) -> Result<&[u8], String> {
    // Imported event collections can store a bare JSON object. Recognize
    // that complete shape before interpreting the first four bytes as a
    // binary format tag; malformed JSON still falls through to the envelope
    // checks, which retain their version and bounds errors.
    if matches!(
        rezzy::json::Value::parse_bytes(bytes),
        Ok(rezzy::json::Value::Object(_))
    ) {
        return Ok(bytes);
    }
    Ok(event_record_parts(bytes)?.2)
}

/// Split an event record into its format tag, internal metadata, and JSON body.
pub(crate) fn event_record_parts(bytes: &[u8]) -> Result<(i32, &[u8], &[u8]), String> {
    let Some(header) = bytes.get(..8) else {
        return Err("event record too short".to_owned());
    };
    let format_version = i32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if !EVENT_RECORD_FORMAT_VERSIONS.contains(&format_version) {
        return Err(format!("unsupported event format version {format_version}"));
    }
    let metadata_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let Some(metadata_end) = 8usize.checked_add(metadata_len) else {
        return Err("event record metadata length exceeds the record".to_owned());
    };
    let Some(metadata) = bytes.get(8..metadata_end) else {
        return Err("event record metadata length exceeds the record".to_owned());
    };
    Ok((format_version, metadata, &bytes[metadata_end..]))
}

/// The auth event ids a format-5 record's internal metadata records.
///
/// MSC4242 stores the ids selected for the event under
/// `calculated_auth_event_ids` in the metadata, beside the PDU. That copy is
/// written with the event and is independent of the derived AUTH edge table,
/// so it is the authoritative side of the audit. Missing or malformed metadata
/// is an error: the record is then uncomparable, not empty.
pub(crate) fn metadata_auth_ids(metadata: &[u8]) -> Result<Vec<String>, String> {
    let value = rezzy::json::Value::parse_bytes(metadata)
        .map_err(|error| format!("internal metadata is not valid JSON: {error}"))?;
    let Some(ids) = value.get("calculated_auth_event_ids") else {
        return Err(
            "MSC4242 record has no calculated_auth_event_ids in internal metadata".to_owned(),
        );
    };
    let rezzy::json::Value::Array(ids) = ids else {
        return Err("MSC4242 calculated_auth_event_ids is not an array".to_owned());
    };
    ids.iter()
        .map(|id| {
            id.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "MSC4242 calculated_auth_event_ids contains a non-string".to_owned())
        })
        .collect()
}

/// Decode an edge record, or say why it cannot be trusted.
pub(crate) fn decode_edges(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.len() < 5 {
        return Err("edge record too short".to_owned());
    }
    if bytes[0] != EDGE_ENCODING_VERSION {
        return Err(format!("unsupported edge record version {}", bytes[0]));
    }
    let count = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    if count.checked_mul(4).and_then(|n| n.checked_add(5)) != Some(bytes.len()) {
        return Err("edge record length mismatch".to_owned());
    }
    let (chunks, _) = bytes[5..].as_chunks::<4>();
    Ok(chunks.iter().map(|c| u32::from_be_bytes(*c)).collect())
}

/// Encode the edge payload consumed by [`check`].
pub(crate) fn encode_edges(targets: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(EDGE_ENCODING_VERSION);
    bytes.extend_from_slice(
        &u32::try_from(targets.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    for target in targets {
        bytes.extend_from_slice(&target.to_be_bytes());
    }
    bytes
}

/// One short id with no outbound edge record.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Dangling {
    pub(crate) short_id: u32,
    pub(crate) event_id: Option<String>,
    /// Short ids of the events that cite this one as an auth event.
    pub(crate) cited_by: Vec<u32>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Report {
    /// Highest short id allocated for the room.
    pub(crate) allocated: u32,
    /// Short ids with a valid outbound edge record.
    pub(crate) with_edges: usize,
    pub(crate) dangling: Vec<Dangling>,
    /// Edge records that could not be decoded, as `(short_id, reason)`.
    pub(crate) corrupt: Vec<(u32, String)>,
    /// `short_id -> event_id` for every id the report mentions.
    pub(crate) names: BTreeMap<u32, String>,
    /// Events whose embedded edges differ from their own `auth_events`.
    pub(crate) mismatched: Vec<Mismatch>,
    /// Events with an edge record but no event record to re-derive it from.
    pub(crate) no_record: Vec<String>,
    /// Short ids with an edge record but no reverse (short id -> event id) entry.
    pub(crate) unnamed: Vec<u32>,
    /// Event records that cannot be parsed, as `(event_id, reason)`.
    pub(crate) bad_records: Vec<(String, String)>,
}

/// An event whose embedded auth edges disagree with its own `auth_events`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Mismatch {
    pub(crate) event_id: String,
    /// In `auth_events` but absent from the embedded edges (truncation).
    pub(crate) missing: Vec<String>,
    /// In the embedded edges but absent from `auth_events`.
    pub(crate) extra: Vec<String>,
}

/// A state-HAMT member whose event has no auth short id.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MissingStateMember {
    pub(crate) event_id: String,
    /// State groups that hold this member, sorted.
    pub(crate) groups: Vec<[u8; 16]>,
}

/// Compare state-HAMT membership with the AUTH forward map.
///
/// `groups` is `(state group id, member event ids)`. An event that appears in
/// several groups is reported once, with every group that names it. Each event
/// is looked up once, so repeated memberships cost only the group bookkeeping.
pub(crate) fn missing_state_members<E>(
    groups: impl IntoIterator<Item = ([u8; 16], Vec<String>)>,
    mut forward_present: impl FnMut(&str) -> Result<bool, E>,
) -> Result<Vec<MissingStateMember>, E> {
    let mut absent: BTreeMap<String, BTreeSet<[u8; 16]>> = BTreeMap::new();
    let mut checked: BTreeMap<String, bool> = BTreeMap::new();
    for (group_id, event_ids) in groups {
        for event_id in event_ids {
            let present = if let Some(present) = checked.get(&event_id).copied() {
                present
            } else {
                let present = forward_present(&event_id)?;
                checked.insert(event_id.clone(), present);
                present
            };
            if !present {
                absent.entry(event_id).or_default().insert(group_id);
            }
        }
    }
    Ok(absent
        .into_iter()
        .map(|(event_id, groups)| MissingStateMember {
            event_id,
            groups: groups.into_iter().collect(),
        })
        .collect())
}

/// The auth event ids a stored event record implies.
///
/// Room v12 derives the room ID from the create event ID and omits that event
/// from ordinary events' `auth_events`. For those rooms, append the implicit
/// create event ID derived from the room ID. Earlier room versions, including
/// v11, list their auth events explicitly.
pub(crate) fn authoritative_auth_ids(
    value: &rezzy::json::Value,
    room_version: &str,
) -> Vec<String> {
    let mut ids = crate::state_resolve::reference_list(value, "auth_events");
    let is_create = value.get("type").and_then(rezzy::json::Value::as_str) == Some("m.room.create")
        && value.get("state_key").and_then(rezzy::json::Value::as_str) == Some("");
    if !is_create
        && mtxdb::MatrixRoomVersion::parse(room_version) == Some(mtxdb::MatrixRoomVersion::V12)
    {
        if let Some(room_id) = value.get("room_id").and_then(rezzy::json::Value::as_str) {
            if let Some(local) = room_id.strip_prefix('!') {
                let create_event_id = format!("${local}");
                if !ids.contains(&create_event_id) {
                    ids.push(create_event_id);
                }
            }
        }
    }
    ids
}

/// What an event record says about an event's auth events.
#[derive(Debug)]
pub(crate) enum EventAuth {
    /// No record for this event.
    Absent,
    /// The record could not be parsed.
    Unreadable(String),
    /// The event's own `auth_events`.
    Ids(Vec<String>),
}

/// Walk short ids `1..=counter`, reading through `read`, which returns the
/// payload of a node in the room's `AUTH` collection (`None` if absent).
/// `authoritative` returns what an event's own record says its auth events are.
pub(crate) fn check<E>(
    mut read: impl FnMut(&NodeId) -> Result<Option<Vec<u8>>, E>,
    mut authoritative: impl FnMut(&str) -> Result<EventAuth, E>,
) -> Result<Report, E> {
    let mut read_present = |id: &NodeId| read(id);

    let mut report = Report::default();
    let Some(counter) = read_present(&counter_node_id())? else {
        return Ok(report);
    };
    let Ok(counter_bytes) = <[u8; 8]>::try_from(counter.as_slice()) else {
        report
            .corrupt
            .push((0, "short-id counter is not 8 bytes".to_owned()));
        return Ok(report);
    };
    let Ok(allocated) = u32::try_from(u64::from_be_bytes(counter_bytes)) else {
        report
            .corrupt
            .push((0, "short-id counter exceeds u32".to_owned()));
        return Ok(report);
    };
    report.allocated = allocated;

    let mut missing: BTreeSet<u32> = BTreeSet::new();
    let mut cited_by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut edge_targets: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for short_id in 1..=allocated {
        match read_present(&edge_node_id(short_id))? {
            None => {
                missing.insert(short_id);
            }
            Some(bytes) => match decode_edges(&bytes) {
                Ok(targets) => {
                    report.with_edges = report.with_edges.saturating_add(1);
                    for &target in &targets {
                        cited_by.entry(target).or_default().push(short_id);
                    }
                    edge_targets.insert(short_id, targets);
                }
                Err(reason) => report.corrupt.push((short_id, reason)),
            },
        }
    }

    // Every allocated id: the authoritative comparison below names each event.
    for short_id in 1..=allocated {
        if let Some(bytes) = read_present(&reverse_node_id(short_id))? {
            if bytes.is_empty() {
                report
                    .corrupt
                    .push((short_id, "reverse map record is empty".to_owned()));
            } else if let Ok(event_id) = String::from_utf8(bytes) {
                report.names.insert(short_id, event_id);
            }
        }
    }

    for (&short_id, targets) in &edge_targets {
        let Some(event_id) = report.names.get(&short_id).cloned() else {
            report.unnamed.push(short_id);
            continue;
        };
        match authoritative(&event_id)? {
            EventAuth::Absent => report.no_record.push(event_id),
            EventAuth::Unreadable(reason) => report.bad_records.push((event_id, reason)),
            EventAuth::Ids(auth_events) => {
                let embedded: BTreeSet<String> = targets
                    .iter()
                    .map(|t| {
                        report
                            .names
                            .get(t)
                            .cloned()
                            .unwrap_or_else(|| format!("<short id {t}>"))
                    })
                    .collect();
                let own: BTreeSet<String> = auth_events.into_iter().collect();
                if embedded != own {
                    report.mismatched.push(Mismatch {
                        event_id,
                        missing: own.difference(&embedded).cloned().collect(),
                        extra: embedded.difference(&own).cloned().collect(),
                    });
                }
            }
        }
    }

    // A cited target outside `1..=allocated` cannot name an allocated short
    // id. Report it like an allocated target with no edge record, with its
    // citers, instead of letting the inconsistency vanish.
    for &target in cited_by.keys() {
        if target == 0 || target > allocated {
            missing.insert(target);
        }
    }

    for short_id in missing {
        report.dangling.push(Dangling {
            short_id,
            event_id: report.names.get(&short_id).cloned(),
            cited_by: cited_by.remove(&short_id).unwrap_or_default(),
        });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::convert::Infallible;

    fn edges(targets: &[u32]) -> Vec<u8> {
        let mut buf = vec![EDGE_ENCODING_VERSION];
        buf.extend_from_slice(
            &u32::try_from(targets.len())
                .expect("test edge list fits in u32")
                .to_be_bytes(),
        );
        for t in targets {
            buf.extend_from_slice(&t.to_be_bytes());
        }
        buf
    }

    fn room(
        allocated: u64,
        records: &[(u32, Vec<u8>)],
        names: &[(u32, &str)],
    ) -> HashMap<NodeId, Vec<u8>> {
        let mut map = HashMap::new();
        map.insert(counter_node_id(), allocated.to_be_bytes().to_vec());
        for (short_id, bytes) in records {
            map.insert(edge_node_id(*short_id), bytes.clone());
        }
        for (short_id, name) in names {
            map.insert(reverse_node_id(*short_id), name.as_bytes().to_vec());
        }
        map
    }

    fn run(map: &HashMap<NodeId, Vec<u8>>) -> Report {
        run_with(map, &HashMap::new())
    }

    /// `own` maps an event id to the `auth_events` its record carries.
    fn run_with(map: &HashMap<NodeId, Vec<u8>>, own: &HashMap<&str, Vec<&str>>) -> Report {
        check(
            |id| Ok::<_, Infallible>(map.get(id).cloned()),
            |event_id| {
                Ok(match own.get(event_id) {
                    Some(ids) => EventAuth::Ids(ids.iter().map(|s| (*s).to_owned()).collect()),
                    None => EventAuth::Absent,
                })
            },
        )
        .unwrap()
    }

    #[test]
    fn complete_graph_has_no_dangling_targets() {
        // 1 is a leaf (a present, empty edge list), 2 cites 1.
        let map = room(2, &[(1, edges(&[])), (2, edges(&[1]))], &[]);
        let report = run(&map);
        assert_eq!(report.allocated, 2);
        assert_eq!(report.with_edges, 2);
        assert_eq!(report.dangling, Vec::<Dangling>::new());
    }

    #[test]
    fn unpersisted_auth_target_is_reported_with_its_citers() {
        // 3 was allocated as an auth target but never persisted.
        let map = room(
            3,
            &[(1, edges(&[])), (2, edges(&[1, 3]))],
            &[(2, "$child"), (3, "$dropped")],
        );
        let report = run(&map);
        assert_eq!(
            report.dangling,
            vec![Dangling {
                short_id: 3,
                event_id: Some("$dropped".to_owned()),
                cited_by: vec![2],
            }]
        );
        assert_eq!(report.names.get(&2).map(String::as_str), Some("$child"));
    }

    #[test]
    fn present_empty_edge_payload_is_corrupt_not_absent() {
        let map = room(1, &[(1, Vec::new())], &[]);
        let report = run(&map);
        assert_eq!(report.with_edges, 0);
        assert_eq!(report.corrupt.len(), 1);
        assert_eq!(report.corrupt[0].0, 1);
    }

    #[test]
    fn present_empty_reverse_record_is_corrupt_not_a_name() {
        // A live zero-byte payload is an ordinary value, not a removal.
        let map = room(1, &[(1, edges(&[]))], &[(1, "")]);
        let report = run(&map);
        assert_eq!(report.names.get(&1), None);
        assert_eq!(
            report.corrupt,
            vec![(1, "reverse map record is empty".to_owned())]
        );
        assert_eq!(report.unnamed, vec![1]);
    }

    #[test]
    fn room_without_a_counter_is_empty() {
        let report = run(&HashMap::new());
        assert_eq!(report, Report::default());
    }

    #[test]
    fn unknown_edge_version_is_corrupt_not_dangling() {
        let mut bad = edges(&[]);
        bad[0] = 9;
        let map = room(1, &[(1, bad)], &[]);
        let report = run(&map);
        assert_eq!(report.dangling, Vec::<Dangling>::new());
        assert_eq!(report.corrupt.len(), 1);
    }

    #[test]
    fn matching_edges_and_records_are_consistent() {
        let map = room(
            2,
            &[(1, edges(&[])), (2, edges(&[1]))],
            &[(1, "$create"), (2, "$child")],
        );
        let own = HashMap::from([("$create", vec![]), ("$child", vec!["$create"])]);
        let report = run_with(&map, &own);
        assert_eq!(report.mismatched, Vec::<Mismatch>::new());
        assert_eq!(report.no_record, Vec::<String>::new());
        assert_eq!(report.unnamed, Vec::<u32>::new());
    }

    #[test]
    fn truncated_edge_list_is_reported_against_the_event_record() {
        // The record cites two auth events; the derived table kept only one.
        let map = room(
            3,
            &[(1, edges(&[])), (2, edges(&[])), (3, edges(&[1]))],
            &[(1, "$a"), (2, "$b"), (3, "$child")],
        );
        let own = HashMap::from([("$a", vec![]), ("$b", vec![]), ("$child", vec!["$a", "$b"])]);
        let report = run_with(&map, &own);
        assert_eq!(
            report.mismatched,
            vec![Mismatch {
                event_id: "$child".to_owned(),
                missing: vec!["$b".to_owned()],
                extra: vec![],
            }]
        );
    }

    #[test]
    fn extra_embedded_edge_is_reported() {
        let map = room(
            2,
            &[(1, edges(&[])), (2, edges(&[1]))],
            &[(1, "$a"), (2, "$child")],
        );
        let own = HashMap::from([("$a", vec![]), ("$child", vec![])]);
        let report = run_with(&map, &own);
        assert_eq!(report.mismatched.len(), 1);
        assert_eq!(report.mismatched[0].extra, vec!["$a".to_owned()]);
    }

    #[test]
    fn edge_record_without_an_event_record_is_reported() {
        let map = room(1, &[(1, edges(&[]))], &[(1, "$lonely")]);
        let report = run(&map);
        assert_eq!(report.no_record, vec!["$lonely".to_owned()]);
    }

    #[test]
    fn edge_record_without_a_reverse_entry_is_unnamed() {
        let map = room(1, &[(1, edges(&[]))], &[]);
        assert_eq!(run(&map).unnamed, vec![1]);
    }

    #[test]
    fn event_record_json_skips_the_metadata_header() {
        let plain = b"{\"auth_events\":[]}";
        assert_eq!(event_record_json(plain).unwrap(), plain);
        assert!(event_record_json(b"{not json}").is_err());
        let mut record = Vec::new();
        record.extend_from_slice(&3i32.to_be_bytes());
        record.extend_from_slice(&3u32.to_be_bytes());
        record.extend_from_slice(b"{}x");
        record.extend_from_slice(b"{\"auth_events\":[]}");
        assert_eq!(event_record_json(&record).unwrap(), b"{\"auth_events\":[]}");
        assert!(event_record_json(&record[..6]).is_err());
        for version in [-1_i32, 1, 2, 3, 4, 5] {
            let mut supported = record.clone();
            supported[..4].copy_from_slice(&version.to_be_bytes());
            assert_eq!(
                event_record_json(&supported).unwrap(),
                b"{\"auth_events\":[]}"
            );
        }
        for version in [0_i32, 6, 99] {
            let mut unsupported = record.clone();
            unsupported[..4].copy_from_slice(&version.to_be_bytes());
            assert!(event_record_json(&unsupported)
                .unwrap_err()
                .contains(&format!("unsupported event format version {version}")));
        }
        let mut bad = record.clone();
        bad[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(event_record_json(&bad).is_err());
        let mut unsupported = record;
        unsupported[..4].copy_from_slice(&99i32.to_be_bytes());
        assert!(event_record_json(&unsupported)
            .unwrap_err()
            .contains("unsupported event format version 99"));
    }

    #[test]
    fn metadata_auth_ids_reads_the_recorded_list() {
        assert_eq!(
            metadata_auth_ids(br#"{"calculated_auth_event_ids":["$a","$b"],"device_id":"DEV"}"#)
                .unwrap(),
            vec!["$a".to_owned(), "$b".to_owned()]
        );
        assert_eq!(
            metadata_auth_ids(br#"{"calculated_auth_event_ids":[]}"#).unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn metadata_auth_ids_rejects_missing_or_malformed_lists() {
        let cases: [(&[u8], &str); 4] = [
            (b"{}", "has no calculated_auth_event_ids"),
            (br#"{"calculated_auth_event_ids":"$a"}"#, "is not an array"),
            (
                br#"{"calculated_auth_event_ids":["$a",1]}"#,
                "contains a non-string",
            ),
            (b"not json", "not valid JSON"),
        ];
        for (metadata, expected) in cases {
            let error = metadata_auth_ids(metadata).unwrap_err();
            assert!(
                error.contains(expected),
                "{error} does not contain {expected}"
            );
        }
    }

    #[test]
    fn authoritative_auth_ids_do_not_add_create_for_v11_server_assigned_rooms() {
        let value = rezzy::json::Value::parse_bytes(
            br#"{"type":"m.room.member","state_key":"@a:x","room_id":"!room:x",
                 "auth_events":["$a","$b"]}"#,
        )
        .unwrap();
        assert_eq!(
            authoritative_auth_ids(&value, "11"),
            vec!["$a".to_owned(), "$b".to_owned()]
        );
    }

    #[test]
    fn authoritative_auth_ids_adds_implicit_create_for_v12_hash_derived_rooms() {
        let value = rezzy::json::Value::parse_bytes(
            br#"{"type":"m.room.member","state_key":"@a:x","room_id":"!room:x",
                 "auth_events":["$a","$b"]}"#,
        )
        .unwrap();
        assert_eq!(
            authoritative_auth_ids(&value, "12"),
            vec!["$a".to_owned(), "$b".to_owned(), "$room:x".to_owned()]
        );
    }

    #[test]
    fn authoritative_auth_ids_leaves_the_create_event_alone() {
        let value = rezzy::json::Value::parse_bytes(
            br#"{"type":"m.room.create","state_key":"","room_id":"!room:x",
                 "auth_events":[]}"#,
        )
        .unwrap();
        assert_eq!(authoritative_auth_ids(&value, "12"), Vec::<String>::new());
    }

    #[test]
    fn authoritative_auth_ids_keeps_v1_events_as_stored() {
        let value = rezzy::json::Value::parse_bytes(
            br#"{"type":"m.room.member","state_key":"@a:x",
                 "auth_events":[["$create",{}],["$member",{}]]}"#,
        )
        .unwrap();
        assert_eq!(
            authoritative_auth_ids(&value, "1"),
            vec!["$create".to_owned(), "$member".to_owned()]
        );
    }

    #[test]
    fn authoritative_auth_ids_does_not_duplicate_an_explicit_create_event() {
        let value = rezzy::json::Value::parse_bytes(
            br#"{"type":"m.room.member","state_key":"@a:x","room_id":"!room:x",
                 "auth_events":["$room:x"]}"#,
        )
        .unwrap();
        assert_eq!(
            authoritative_auth_ids(&value, "12"),
            vec!["$room:x".to_owned()]
        );
    }

    #[test]
    fn state_members_without_a_forward_id_are_grouped_and_deduplicated() {
        let groups = vec![
            ([1u8; 16], vec!["$a".to_owned(), "$b".to_owned()]),
            ([2u8; 16], vec!["$b".to_owned()]),
        ];
        let report =
            missing_state_members(groups, |event_id| Ok::<_, Infallible>(event_id == "$a"))
                .unwrap();
        assert_eq!(
            report,
            vec![MissingStateMember {
                event_id: "$b".to_owned(),
                groups: vec![[1u8; 16], [2u8; 16]],
            }]
        );
    }

    #[test]
    fn state_members_with_forward_ids_are_clean() {
        let groups = vec![([1u8; 16], vec!["$a".to_owned()])];
        assert_eq!(
            missing_state_members(groups, |_| Ok::<_, Infallible>(true)).unwrap(),
            Vec::<MissingStateMember>::new()
        );
    }

    #[test]
    fn state_members_are_looked_up_once() {
        let groups = vec![
            ([1u8; 16], vec!["$a".to_owned()]),
            ([2u8; 16], vec!["$a".to_owned()]),
        ];
        let lookups = std::cell::Cell::new(0u32);
        let report = missing_state_members(groups, |_| {
            lookups.set(lookups.get() + 1);
            Ok::<_, Infallible>(false)
        })
        .unwrap();
        assert_eq!(lookups.get(), 1);
        assert_eq!(report[0].groups, vec![[1u8; 16], [2u8; 16]]);
    }

    #[test]
    fn forwarding_errors_propagate() {
        let groups = vec![([1u8; 16], vec!["$a".to_owned()])];
        let result: Result<_, &str> = missing_state_members(groups, |_| Err("boom"));
        assert_eq!(result.unwrap_err(), "boom");
    }

    #[test]
    fn counter_above_u32_is_corrupt_not_truncated() {
        let mut map = HashMap::new();
        map.insert(
            counter_node_id(),
            (u64::from(u32::MAX) + 1).to_be_bytes().to_vec(),
        );
        let report = run(&map);
        assert_eq!(report.allocated, 0);
        assert_eq!(
            report.corrupt,
            vec![(0, "short-id counter exceeds u32".to_owned())]
        );
    }

    #[test]
    fn cited_target_outside_the_counter_is_reported_as_dangling() {
        // 2 cites 9, but the counter says only 1..=2 were allocated.
        let map = room(2, &[(1, edges(&[])), (2, edges(&[1, 9]))], &[(2, "$child")]);
        let report = run(&map);
        assert_eq!(
            report.dangling,
            vec![Dangling {
                short_id: 9,
                event_id: None,
                cited_by: vec![2],
            }]
        );
    }
}
