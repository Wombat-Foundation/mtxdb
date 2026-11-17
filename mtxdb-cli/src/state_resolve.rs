//! Adapter from mtxdb's forward DAG walk to rezzy's state resolver.
//!
//! mtxdb walks the room's event DAG itself and keeps one state set per
//! frontier event. When a node has more than one resident parent those sets
//! have to be merged, and this module owns that merge: it converts mtxdb's
//! state sets into rezzy's persistent `SharedState` maps, feeds rezzy the
//! events its auth rules read, and converts the resolved map back.
//!
//! Two properties are load-bearing.
//!
//! **No panics.** mtxdb's binary sets `panic = "abort"`, so an event missing
//! inside rezzy would kill the process instead of unwinding into a fallback.
//! [`LeanEventContext::ensure`] therefore materializes everything rezzy will
//! read *before* calling it, and [`resolve_parents`] returns
//! [`IncompleteResolution`] instead of panicking when the batch is short.
//!
//! **Identity.** The resolver version and the importer's trust claim come from
//! a [`ResolutionPolicy`] and are hashed into the state-group id, so a group
//! produced here can never be mistaken for one produced by another algorithm
//! or by an unverified import.
//!
//! What rezzy reads: every event id reachable from a parent state map (that is
//! what lands in the unconflicted base and the conflicted set), plus the
//! transitive `auth_events` closure of all of them, which is where mainline
//! ordering and local auth look. `prev_events` is never read — `dag_edges`
//! resolves to `auth_events` for every state-resolution version, and for a
//! `LeanEvent` MSC4242's `prev_state_events` aliases `auth_events` anyway.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use rezzy::json::{Object, Value as OwnedValue};
use rezzy::{EventType, LeanEvent, SharedState as RezzyState};

use crate::cmd::{event_id, event_string_field};
use crate::state_policy::{event_disposition, EventDisposition, ResolutionPolicy};

/// mtxdb's state set, keyed by `(type, state_key)` and valued by event id.
///
/// This is the shape `crate::cmd::StateSet::entries` stores; it is spelled
/// out here so the adapter can be written and tested against mtxdb's own type
/// without widening that module's API.
///
/// A persistent map: cloning is O(1) and a write copies only its root-to-leaf
/// path, so a state event costs O(log n) instead of a copy of the whole state.
pub(crate) type StateEntries = rezzy::PersistentOrdMap<(String, String), String>;

/// A merge that could not be resolved because the import batch does not carry
/// everything the resolver reads.
///
/// This is not an error: the caller falls back to the first-parent
/// merge and deliberately skips publishing the result so that a later, fuller
/// import recomputes it under the real resolver.
#[derive(Debug, Default)]
pub(crate) struct IncompleteResolution {
    /// The referenced event ids this batch does not contain, sorted and
    /// deduplicated. Empty only when the context has never seen a gap.
    pub(crate) missing: Vec<String>,
}

/// Lazily materialized rezzy events for one state-group walk.
///
/// The index is built once over the batch; [`LeanEvent`]s are allocated only
/// for the auth closure the resolver actually walks, and each id is walked at
/// most once for the whole run, so a long walk pays for its context once
/// rather than once per merge.
pub(crate) struct LeanEventContext<'a> {
    values: HashMap<&'a str, &'a OwnedValue>,
    events: HashMap<String, LeanEvent<String>>,
    /// Ids whose full auth closure is known to resolve. Monotone: an id is
    /// never walked twice, and never re-reported once it resolves.
    verified: HashSet<String>,
    /// Ids that resolve to nothing in this batch. Monotone, so reporting the
    /// whole set on every call stays correct.
    missing: BTreeSet<String>,
    /// The room-wide reachability index and the resolver's memoization,
    /// built on the first real merge and reused by every later one. Rebuilding
    /// them per merge made each merge cost O(room).
    resolve_state: Option<(
        rezzy::resolve::reachability::RangePrefilterReachability<String>,
        rezzy::ForkResolveCaches<String, rezzy::JsonValue>,
    )>,
    /// Per-event auth-chain bitmaps over every event in the batch, built with
    /// the reachability index. Present only when the batch's auth graph is
    /// complete: the difference it yields is exact only then, so a gapped
    /// batch falls back to the resolver's own walk.
    auth_graph: Option<rezzy::auth::roaring::AuthGraph<String>>,
}

impl<'a> LeanEventContext<'a> {
    /// Index `values` by event id without allocating any [`LeanEvent`] yet.
    pub(crate) fn new(values: impl Iterator<Item = &'a OwnedValue>) -> Self {
        let mut index: HashMap<&'a str, &'a OwnedValue> = HashMap::new();
        for value in values {
            if let Some(id) = event_id(value) {
                index.insert(id, value);
            }
        }
        Self {
            values: index,
            events: HashMap::new(),
            verified: HashSet::new(),
            missing: BTreeSet::new(),
            resolve_state: None,
            auth_graph: None,
        }
    }

    /// Resolve `maps` with rezzy, sharing one reachability index and one set
    /// of caches across every merge in the walk.
    ///
    /// The first call materializes every batch event so the index covers the
    /// whole room; later [`Self::ensure`] calls then find them already built.
    /// Dangling `auth_events` references are skipped by the index, so a batch
    /// with absent ancestors cannot make it panic.
    pub(crate) fn resolve(
        &mut self,
        maps: &[RezzyState],
        version: rezzy::StateResVersion,
    ) -> RezzyState {
        if self.resolve_state.is_none() {
            let ids: Vec<&'a str> = self.values.keys().copied().collect();
            for id in ids {
                if !self.events.contains_key(id) {
                    if let Some(event) = self.values.get(id).and_then(|v| lean_event(v, id)) {
                        self.events.insert(id.to_owned(), event);
                    }
                }
            }
            let index = rezzy::resolve::reachability::RangePrefilterReachability::<String>::build(
                &self.events,
            );
            self.resolve_state = Some((index, rezzy::ForkResolveCaches::new(version)));
            self.auth_graph = rezzy::auth::roaring::AuthGraph::try_build(&self.events)
                .ok()
                .filter(rezzy::auth::roaring::AuthGraph::is_complete);
        }
        // The merge's auth difference from the bitmaps, when every state event
        // is in a complete graph; otherwise the resolver walks the auth lists.
        let auth_diff = self
            .auth_graph
            .as_ref()
            .and_then(|graph| graph.try_state_auth_difference(maps).ok());
        match self.resolve_state.as_mut() {
            Some((index, caches)) => match auth_diff {
                Some(diff) => rezzy::resolve_state_maps_cached_with_auth_diff(
                    maps,
                    &self.events,
                    version,
                    index,
                    caches,
                    &diff,
                ),
                None => {
                    rezzy::resolve_state_maps_cached(maps, &self.events, version, index, caches)
                }
            },
            None => rezzy::resolve_state_maps(maps, &self.events, version),
        }
    }

    /// Materialize `roots` and their transitive `auth_events` closure.
    ///
    /// Returns every id this context knows to be absent from the batch, sorted.
    /// An empty result means the materialized event map is complete enough
    /// to hand to `rezzy::resolve_state_maps`.
    pub(crate) fn ensure(&mut self, roots: &[String]) -> Vec<String> {
        let mut queue: VecDeque<String> = roots
            .iter()
            .filter(|id| !self.verified.contains(*id) && !self.missing.contains(*id))
            .cloned()
            .collect();
        while let Some(id) = queue.pop_front() {
            if self.verified.contains(&id) || self.missing.contains(&id) {
                continue;
            }
            let auth_events = if let Some(event) = self.events.get(&id) {
                event.auth_events.clone()
            } else {
                let Some(value) = self.values.get(id.as_str()).copied() else {
                    self.missing.insert(id);
                    continue;
                };
                let Some(event) = lean_event(value, &id) else {
                    self.missing.insert(id);
                    continue;
                };
                let auth_events = event.auth_events.clone();
                self.events.insert(id.clone(), event);
                auth_events
            };
            for reference in auth_events {
                if !self.verified.contains(&reference) && !self.missing.contains(&reference) {
                    queue.push_back(reference);
                }
            }
            self.verified.insert(id);
        }
        if self.missing.is_empty() {
            Vec::new()
        } else {
            self.missing.iter().cloned().collect()
        }
    }

    /// The materialized events, ready to pass as `event_context`.
    #[cfg(test)]
    pub(crate) fn map(&self) -> &HashMap<String, LeanEvent<String>> {
        &self.events
    }
}

/// Convert mtxdb's state set into a rezzy state map.
pub(crate) fn to_rezzy_state(entries: &StateEntries) -> RezzyState {
    entries
        .iter()
        .map(|((event_type, state_key), event_id)| {
            (
                (EventType::from(event_type.as_str()), state_key.clone()),
                event_id.clone(),
            )
        })
        .collect()
}

/// Convert a resolved rezzy state map back into mtxdb's state set.
pub(crate) fn from_rezzy_state(state: &RezzyState) -> StateEntries {
    state
        .iter()
        .map(|((event_type, state_key), event_id)| {
            (
                (event_type.as_str().to_owned(), state_key.clone()),
                event_id.clone(),
            )
        })
        .collect()
}

/// Time spent inside each part of one merge, so an import can report where a
/// slow gapped room actually spends its state-walk seconds.
#[derive(Debug, Default)]
pub(crate) struct MergePhaseTimes {
    /// Converting mtxdb's state maps into rezzy's key type.
    pub(crate) convert: std::time::Duration,
    /// Materializing the auth closure the resolver reads.
    pub(crate) ensure: std::time::Duration,
    /// `rezzy::resolve_state_maps`.
    pub(crate) resolve: std::time::Duration,
    /// Converting the resolved map back into mtxdb's type.
    pub(crate) back: std::time::Duration,
}

/// Resolve one multi-parent merge with rezzy.
///
/// # Errors
/// Returns [`IncompleteResolution`] when the batch is missing any event the
/// resolver would read, or when `policy` names no resolver at all. The caller
/// falls back to the first-parent merge in both cases.
pub(crate) fn resolve_parents(
    context: &mut LeanEventContext<'_>,
    parent_entries: &[&StateEntries],
    policy: ResolutionPolicy,
    phases: &mut MergePhaseTimes,
) -> Result<StateEntries, IncompleteResolution> {
    let Some(version) = policy.rezzy_version else {
        // No recognized room version means no auth rules; the derivation
        // already records `FirstParent`, so the caller uses the first-parent merge.
        return Err(IncompleteResolution::default());
    };
    let convert_started = std::time::Instant::now();
    let maps: Vec<RezzyState> = parent_entries.iter().map(|e| to_rezzy_state(e)).collect();
    let roots = referenced_ids(&maps);
    phases.convert = phases.convert.saturating_add(convert_started.elapsed());
    let ensure_started = std::time::Instant::now();
    let missing = context.ensure(&roots);
    phases.ensure = phases.ensure.saturating_add(ensure_started.elapsed());
    if !missing.is_empty() {
        return Err(IncompleteResolution { missing });
    }
    let resolve_started = std::time::Instant::now();
    let resolved = context.resolve(&maps, version);
    phases.resolve = phases.resolve.saturating_add(resolve_started.elapsed());
    let back_started = std::time::Instant::now();
    let entries = from_rezzy_state(&resolved);
    phases.back = phases.back.saturating_add(back_started.elapsed());
    Ok(entries)
}

/// Every event id any parent state map points at, deduplicated in first-seen
/// order. These are exactly the ids that land in the resolved state, so their
/// auth closures are what the resolver has to be able to walk.
fn referenced_ids(maps: &[RezzyState]) -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut ids = Vec::new();
    for map in maps {
        for (_key, id) in map {
            if seen.insert(id.as_str()) {
                ids.push(id.clone());
            }
        }
    }
    ids
}

/// Build the [`LeanEvent`] rezzy reads for one event.
///
/// Returns `None` when the value is not a usable event — no `type`, or not an
/// object at all. The caller records that as a missing id rather than handing
/// rezzy something it would unwrap.
pub(crate) fn lean_event(value: &OwnedValue, event_id: &str) -> Option<LeanEvent<String>> {
    let event_type = event_string_field(value, "type")?;
    let disposition = event_disposition(value);
    Some(LeanEvent {
        event_id: event_id.to_owned(),
        event_type: event_type.to_owned(),
        state_key: event_string_field(value, "state_key").map(str::to_owned),
        // Only a precomputed sort hint. The authoritative power level comes
        // from the auth chain during resolution, so leaving it zero is not a
        // soundness gap.
        power_level: 0,
        origin_server_ts: u64_field(value, "origin_server_ts"),
        sender: event_string_field(value, "sender")
            .unwrap_or_default()
            .to_owned(),
        content: value
            .get("content")
            .cloned()
            .unwrap_or_else(|| OwnedValue::Object(Object::default())),
        prev_events: reference_list(value, "prev_events"),
        auth_events: reference_list(value, "auth_events"),
        depth: u64_field(value, "depth"),
        rejected: disposition == EventDisposition::Rejected,
        soft_fail: disposition == EventDisposition::SoftFailed,
        // mtxdb filters to one room before the walk, and `LeanEvent`'s room
        // field only arms `check_auth_chain`, which resolution does not call.
        // Leaving it unset keeps rezzy on its default path.
        room_id: None,
    })
}

/// Read an unsigned integer field, defaulting to zero.
fn u64_field(value: &OwnedValue, field: &str) -> u64 {
    value
        .get(field)
        .and_then(OwnedValue::as_u64)
        .unwrap_or_default()
}

/// Read an array of event references. Matrix rooms v1/v2 encode references as
/// `[event_id, hashes]` pairs while later room versions use plain strings;
/// both forms name the id in the same place.
pub(crate) fn reference_list(value: &OwnedValue, field: &str) -> Vec<String> {
    let Some(OwnedValue::Array(items)) = value.get(field) else {
        return Vec::new();
    };
    items.iter().filter_map(reference_id).collect()
}

/// The event id one array element of a reference list names.
fn reference_id(item: &OwnedValue) -> Option<String> {
    match item {
        OwnedValue::String(text) => Some(text.clone()),
        OwnedValue::Array(pair) => pair.first().and_then(|id| id.as_str()).map(str::to_owned),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(json: &str) -> OwnedValue {
        OwnedValue::parse_bytes(json.as_bytes()).expect("valid JSON fixture")
    }

    fn event(event_id: &str, auth_events: &[&str]) -> OwnedValue {
        let auth = auth_events
            .iter()
            .map(|id| format!("\"{id}\""))
            .collect::<Vec<_>>()
            .join(",");
        value(&format!(
            r#"{{"event_id":"{event_id}","type":"m.room.member","sender":"@a:x",
                 "state_key":"@a:x","depth":1,"origin_server_ts":1,
                 "content":{{"membership":"join"}},"auth_events":[{auth}],"prev_events":[]}}"#
        ))
    }

    #[test]
    fn context_materializes_only_the_auth_closure_it_is_asked_for() {
        let events = [
            event("$root", &[]),
            event("$child", &["$root"]),
            event("$unrelated", &[]),
        ];
        let mut context = LeanEventContext::new(events.iter());
        assert_eq!(context.ensure(&["$child".to_owned()]), Vec::<String>::new());
        assert!(context.map().contains_key("$child"));
        assert!(context.map().contains_key("$root"));
        assert!(!context.map().contains_key("$unrelated"));
    }

    #[test]
    fn context_reports_ids_absent_from_the_batch() {
        let events = [event("$present", &["$absent"])];
        let mut context = LeanEventContext::new(events.iter());
        assert_eq!(
            context.ensure(&["$present".to_owned()]),
            vec!["$absent".to_owned()]
        );
        // A second call with a resolvable root still reports the known gap,
        // because the gap is what makes any merge unsafe.
        assert_eq!(
            context.ensure(&["$present".to_owned()]),
            vec!["$absent".to_owned()]
        );
    }

    fn room_event(
        id: &str,
        kind: &str,
        state_key: &str,
        sender: &str,
        depth: u32,
        content: &str,
        auth: &[&str],
    ) -> OwnedValue {
        let auth = auth
            .iter()
            .map(|a| format!("\"{a}\""))
            .collect::<Vec<_>>()
            .join(",");
        value(&format!(
            r#"{{"event_id":"{id}","type":"{kind}","state_key":"{state_key}","sender":"{sender}",
                 "depth":{depth},"origin_server_ts":{depth},"content":{content},
                 "auth_events":[{auth}],"prev_events":[]}}"#
        ))
    }

    fn state(rows: &[(&str, &str, &str)]) -> RezzyState {
        let mut entries = StateEntries::new();
        for (kind, key, id) in rows {
            entries.insert(((*kind).to_owned(), (*key).to_owned()), (*id).to_owned());
        }
        to_rezzy_state(&entries)
    }

    /// A merge resolved with the room's auth bitmaps must equal the same merge
    /// resolved by the resolver's own auth-list walk.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the fixture lists each event of the two forks"
    )]
    fn a_merge_resolved_from_the_bitmaps_equals_the_walked_one() {
        let events = [
            room_event(
                "$create",
                "m.room.create",
                "",
                "@a:x",
                1,
                r#"{"creator":"@a:x"}"#,
                &[],
            ),
            room_event(
                "$alice",
                "m.room.member",
                "@a:x",
                "@a:x",
                2,
                r#"{"membership":"join"}"#,
                &["$create"],
            ),
            room_event(
                "$pl",
                "m.room.power_levels",
                "",
                "@a:x",
                3,
                r#"{"users":{"@a:x":100}}"#,
                &["$create", "$alice"],
            ),
            room_event(
                "$jr",
                "m.room.join_rules",
                "",
                "@a:x",
                4,
                r#"{"join_rule":"public"}"#,
                &["$create", "$alice", "$pl"],
            ),
            room_event(
                "$topic_a",
                "m.room.topic",
                "",
                "@a:x",
                5,
                r#"{"topic":"a"}"#,
                &["$create", "$alice", "$pl"],
            ),
            room_event(
                "$bob_a",
                "m.room.member",
                "@b:x",
                "@b:x",
                5,
                r#"{"membership":"join"}"#,
                &["$create", "$jr", "$pl"],
            ),
            room_event(
                "$topic_b",
                "m.room.topic",
                "",
                "@a:x",
                6,
                r#"{"topic":"b"}"#,
                &["$create", "$alice", "$pl"],
            ),
            room_event(
                "$carol_b",
                "m.room.member",
                "@c:x",
                "@c:x",
                6,
                r#"{"membership":"join"}"#,
                &["$create", "$jr", "$pl"],
            ),
        ];
        let shared = [
            ("m.room.create", "", "$create"),
            ("m.room.member", "@a:x", "$alice"),
            ("m.room.power_levels", "", "$pl"),
            ("m.room.join_rules", "", "$jr"),
        ];
        let mut fork_a = shared.to_vec();
        fork_a.extend([
            ("m.room.topic", "", "$topic_a"),
            ("m.room.member", "@b:x", "$bob_a"),
        ]);
        let mut fork_b = shared.to_vec();
        fork_b.extend([
            ("m.room.topic", "", "$topic_b"),
            ("m.room.member", "@c:x", "$carol_b"),
        ]);
        let maps = [state(&fork_a), state(&fork_b)];

        for version in [rezzy::StateResVersion::V2, rezzy::StateResVersion::V2_1] {
            let mut context = LeanEventContext::new(events.iter());
            let resolved = context.resolve(&maps, version);
            assert!(
                context.auth_graph.is_some(),
                "a complete batch must build the bitmap graph"
            );
            let walked = rezzy::resolve_state_maps(&maps, context.map(), version);
            assert_eq!(resolved, walked, "{version:?}");
        }
    }

    /// A batch with a dangling auth reference has no bitmap graph, so the
    /// merge falls back to the walk instead of using an inexact difference.
    /// The gap sits on an event outside both forks, as `ensure` has already
    /// refused any merge whose own auth closure is gapped.
    #[test]
    fn a_gapped_batch_builds_no_bitmap_graph_and_still_resolves() {
        let events = [
            room_event("$create", "m.room.create", "", "@a:x", 1, "{}", &[]),
            room_event(
                "$t1",
                "m.room.topic",
                "",
                "@a:x",
                2,
                r#"{"topic":"1"}"#,
                &["$create"],
            ),
            room_event(
                "$t2",
                "m.room.topic",
                "",
                "@a:x",
                2,
                r#"{"topic":"2"}"#,
                &["$create"],
            ),
            room_event(
                "$orphan",
                "m.room.name",
                "",
                "@a:x",
                2,
                r#"{"name":"n"}"#,
                &["$missing"],
            ),
        ];
        let maps = [
            state(&[
                ("m.room.create", "", "$create"),
                ("m.room.topic", "", "$t1"),
            ]),
            state(&[
                ("m.room.create", "", "$create"),
                ("m.room.topic", "", "$t2"),
            ]),
        ];
        let mut context = LeanEventContext::new(events.iter());
        let resolved = context.resolve(&maps, rezzy::StateResVersion::V2);
        assert!(context.auth_graph.is_none());
        let walked = rezzy::resolve_state_maps(&maps, context.map(), rezzy::StateResVersion::V2);
        assert_eq!(resolved, walked);
    }

    #[test]
    fn state_round_trips_between_mtxdb_and_rezzy_shapes() {
        let mut entries = StateEntries::new();
        entries.insert(
            ("m.room.create".to_owned(), String::new()),
            "$create".to_owned(),
        );
        entries.insert(
            ("m.room.member".to_owned(), "@a:x".to_owned()),
            "$join".to_owned(),
        );
        let state = to_rezzy_state(&entries);
        assert_eq!(state.len(), 2);
        assert_eq!(from_rezzy_state(&state), entries);
    }

    #[test]
    fn rejected_flags_reach_rezzy() {
        let flagged = value(r#"{"event_id":"$x","type":"m.room.member","rejected":true}"#);
        let event = lean_event(&flagged, "$x").expect("event materializes");
        assert!(event.rejected);
        assert!(!event.soft_fail);
        assert!(event.accepted_state_key().is_none());
    }

    #[test]
    fn unusable_values_do_not_materialize() {
        assert!(lean_event(&value(r#"{"event_id":"$x"}"#), "$x").is_none());
        assert!(lean_event(&value(r#""just a string""#), "$x").is_none());
    }
}
