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
//! The record layout belongs to the homeserver, not to this crate: the key
//! derivations and encoding below mirror sithnapse's
//! `rust/src/database/mtxdb_syn.rs` (`short_id_*_node_id`,
//! `encode_auth_edges`). If that layout changes, this check must change with
//! it; a version mismatch is reported rather than guessed at.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

const EDGE_ENCODING_VERSION: u8 = 1;

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
    Ok(bytes[5..]
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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
}

/// Walk short ids `1..=counter`, reading through `read`, which returns the
/// payload of a node in the room's `AUTH` collection (`None` if absent).
/// Empty payloads count as absent: the engine treats them as removal markers.
pub(crate) fn check<E>(
    mut read: impl FnMut(&NodeId) -> Result<Option<Vec<u8>>, E>,
) -> Result<Report, E> {
    let mut read_present = |id: &NodeId| -> Result<Option<Vec<u8>>, E> {
        Ok(read(id)?.filter(|bytes| !bytes.is_empty()))
    };

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
    let allocated = u32::try_from(u64::from_be_bytes(counter_bytes)).unwrap_or(u32::MAX);
    report.allocated = allocated;

    let mut missing: BTreeSet<u32> = BTreeSet::new();
    let mut cited_by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for short_id in 1..=allocated {
        match read_present(&edge_node_id(short_id))? {
            None => {
                missing.insert(short_id);
            }
            Some(bytes) => match decode_edges(&bytes) {
                Ok(targets) => {
                    report.with_edges = report.with_edges.saturating_add(1);
                    for target in targets {
                        cited_by.entry(target).or_default().push(short_id);
                    }
                }
                Err(reason) => report.corrupt.push((short_id, reason)),
            },
        }
    }

    let mut wanted: BTreeSet<u32> = missing.clone();
    for citers in cited_by.values() {
        wanted.extend(citers.iter().copied());
    }
    for &short_id in &wanted {
        if let Some(bytes) = read_present(&reverse_node_id(short_id))? {
            if let Ok(event_id) = String::from_utf8(bytes) {
                report.names.insert(short_id, event_id);
            }
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
        check(|id| Ok::<_, Infallible>(map.get(id).cloned())).unwrap()
    }

    #[test]
    fn complete_graph_has_no_dangling_targets() {
        // 1 is a leaf (a present, empty edge list), 2 cites 1.
        let map = room(2, &[(1, edges(&[])), (2, edges(&[1]))], &[]);
        let report = run(&map);
        assert_eq!(report.allocated, 2);
        assert_eq!(report.with_edges, 2);
        assert!(report.dangling.is_empty());
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
    fn empty_edge_payload_is_a_removal_marker_not_a_leaf() {
        let map = room(1, &[(1, Vec::new())], &[]);
        let report = run(&map);
        assert_eq!(report.with_edges, 0);
        assert_eq!(report.dangling.len(), 1);
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
        assert!(report.dangling.is_empty());
        assert_eq!(report.corrupt.len(), 1);
    }
}
