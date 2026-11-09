//! Small deterministic state-walk fixtures for the gapped-import optimization.
//!
//! These exercise the state computation directly: no CLI import, source
//! database, or external fixture is involved. The timing line is diagnostic;
//! only merge counts and coverage are asserted.

use super::{compute_state_groups_partial, OwnedValue};

fn event(json: &str) -> OwnedValue {
    OwnedValue::parse_bytes(json.as_bytes()).expect("valid fixture event")
}

fn report(fixture: &str, elapsed: std::time::Duration, result: &super::PartialStateComputation) {
    eprintln!(
        "gapped-state fixture={fixture} total_us={} merges={} convert_us={} ensure_us={} resolve_us={} back_us={}",
        elapsed.as_micros(),
        result.merges,
        result.merge_phases.convert.as_micros(),
        result.merge_phases.ensure.as_micros(),
        result.merge_phases.resolve.as_micros(),
        result.merge_phases.back.as_micros(),
    );
}

#[test]
fn gapped_merge_skips_resolution_and_stays_unmapped() {
    let events = vec![
        event(
            r#"{"event_id":"$create","room_id":"!gapped:test","type":"m.room.create","state_key":"","sender":"@admin:test","depth":1,"prev_events":[],"auth_events":[],"content":{"room_version":"10","creator":"@admin:test"}}"#,
        ),
        event(
            r#"{"event_id":"$join","room_id":"!gapped:test","type":"m.room.member","state_key":"@admin:test","sender":"@admin:test","depth":2,"prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
        ),
        event(
            r#"{"event_id":"$power","room_id":"!gapped:test","type":"m.room.power_levels","state_key":"","sender":"@admin:test","depth":3,"prev_events":["$join"],"auth_events":["$create","$join"],"content":{"users_default":0,"state_default":50,"events_default":0,"users":{"@admin:test":100}}}"#,
        ),
        event(
            r#"{"event_id":"$chain","room_id":"!gapped:test","type":"m.room.message","sender":"@admin:test","depth":4,"prev_events":["$power"],"auth_events":["$create","$join","$power"],"content":{"body":"complete chain"}}"#,
        ),
        event(
            r#"{"event_id":"$topic","room_id":"!gapped:test","type":"m.room.topic","state_key":"","sender":"@admin:test","depth":5,"prev_events":["$chain"],"auth_events":["$create","$join","$power"],"content":{"topic":"complete branch"}}"#,
        ),
        // The body for `$missing` is absent. `$gap` is therefore incomplete;
        // `$child` is a genuinely divergent two-parent merge across that gap.
        event(
            r#"{"event_id":"$gap","room_id":"!gapped:test","type":"m.room.message","sender":"@admin:test","depth":5,"prev_events":["$missing"],"auth_events":["$create","$join","$power"],"content":{"body":"gapped branch"}}"#,
        ),
        event(
            r#"{"event_id":"$child","room_id":"!gapped:test","type":"m.room.message","sender":"@admin:test","depth":6,"prev_events":["$topic","$gap"],"auth_events":["$create","$join","$power"],"content":{"body":"fork across gap"}}"#,
        ),
        event(
            r#"{"event_id":"$grandchild","room_id":"!gapped:test","type":"m.room.message","sender":"@admin:test","depth":7,"prev_events":["$child"],"auth_events":["$create","$join","$power"],"content":{"body":"gapped descendant"}}"#,
        ),
    ];

    let started = std::time::Instant::now();
    let result = compute_state_groups_partial(&events, &[], "!gapped:test");
    report("gapped", started.elapsed(), &result);

    assert_eq!(result.merges, 0, "do not resolve a fork past a gap");
    for id in ["$gap", "$child", "$grandchild"] {
        assert!(result.gapped.iter().any(|gapped| gapped == id), "{id}");
        assert!(!result.groups.contains_key(id), "{id} must remain unmapped");
    }
}

#[test]
fn fork_merge_still_runs_and_publishes() {
    let events = vec![
        event(
            r#"{"event_id":"$create","room_id":"!fork:test","type":"m.room.create","state_key":"","sender":"@admin:test","depth":1,"prev_events":[],"auth_events":[],"content":{"room_version":"10","creator":"@admin:test"}}"#,
        ),
        event(
            r#"{"event_id":"$join","room_id":"!fork:test","type":"m.room.member","state_key":"@admin:test","sender":"@admin:test","depth":2,"prev_events":["$create"],"auth_events":["$create"],"content":{"membership":"join"}}"#,
        ),
        event(
            r#"{"event_id":"$power","room_id":"!fork:test","type":"m.room.power_levels","state_key":"","sender":"@admin:test","depth":3,"prev_events":["$join"],"auth_events":["$create","$join"],"content":{"users_default":0,"state_default":50,"events_default":0,"users":{"@admin:test":100}}}"#,
        ),
        event(
            r#"{"event_id":"$left","room_id":"!fork:test","type":"m.room.topic","state_key":"","sender":"@admin:test","depth":4,"prev_events":["$power"],"auth_events":["$create","$join","$power"],"content":{"topic":"left"}}"#,
        ),
        event(
            r#"{"event_id":"$right","room_id":"!fork:test","type":"m.room.topic","state_key":"","sender":"@admin:test","depth":4,"prev_events":["$power"],"auth_events":["$create","$join","$power"],"content":{"topic":"right"}}"#,
        ),
        event(
            r#"{"event_id":"$merge","room_id":"!fork:test","type":"m.room.message","sender":"@admin:test","depth":5,"prev_events":["$left","$right"],"auth_events":["$create","$join","$power"],"content":{"body":"complete fork"}}"#,
        ),
    ];

    let started = std::time::Instant::now();
    let result = compute_state_groups_partial(&events, &[], "!fork:test");
    report("complete-fork", started.elapsed(), &result);

    assert!(
        result.merges > 0,
        "complete divergent parents still resolve"
    );
    assert!(
        result.incomplete.is_empty(),
        "the fixture includes auth closure"
    );
    assert!(result.gapped.is_empty());
    assert!(
        result.groups.contains_key("$merge"),
        "merged event is mapped"
    );
}
