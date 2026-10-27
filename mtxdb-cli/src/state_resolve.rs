//! Adapter from mtxdb's forward DAG walk to rezzy's state resolver.
//!
//! mtxdb walks the room's event DAG itself and keeps one state set per
//! frontier event. When a node has more than one resident parent those sets
//! have to be merged, and this module owns that merge: it converts mtxdb's
//! state sets into rezzy's persistent [`SharedState`] maps, feeds rezzy the
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
/// This is the shape [`crate::cmd::StateSet::entries`] stores; it is spelled
/// out here so the adapter can be written and tested against mtxdb's own type
/// without widening that module's API.
pub(crate) type StateEntries = HashMap<(String, String), String>;

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
        }
    }

    /// Materialize `roots` and their transitive `auth_events` closure.
    ///
    /// Returns every id this context knows to be absent from the batch, sorted.
    /// An empty result means [`Self::map`] is complete enough to hand to
    /// `rezzy::resolve_state_maps`.
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
) -> Result<StateEntries, IncompleteResolution> {
    let Some(version) = policy.rezzy_version else {
        // No recognized room version means no auth rules; the derivation
        // already records `FirstParent`, so the caller uses the first-parent merge.
        return Err(IncompleteResolution::default());
    };
    let maps: Vec<RezzyState> = parent_entries.iter().map(|e| to_rezzy_state(e)).collect();
    let roots = referenced_ids(&maps);
    let missing = context.ensure(&roots);
    if !missing.is_empty() {
        return Err(IncompleteResolution { missing });
    }
    let resolved = rezzy::resolve_state_maps(&maps, context.map(), version);
    Ok(from_rezzy_state(&resolved))
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
fn reference_list(value: &OwnedValue, field: &str) -> Vec<String> {
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
        assert!(context.ensure(&["$child".to_owned()]).is_empty());
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
