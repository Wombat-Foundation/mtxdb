use super::*;

const fn class(
    retention: Retention,
    durability: Durability,
    ordering: OrderingPolicy,
) -> RecordClass {
    RecordClass::new(retention, durability, ordering)
}

#[test]
fn immutable_records_do_not_supersede() {
    let events = class(
        Retention::Immutable,
        Durability::Synchronous,
        OrderingPolicy::TopologicalAndStream,
    );
    assert!(!events.supersedes());
    assert!(!events.retains_prior_versions());
    assert_eq!(events.version_limit(), Some(1));
    assert!(events.is_synchronous());
    assert!(!events.is_volatile());
    assert!(events.is_topological());
    assert!(events.is_streamed());
}

#[test]
fn ordering_policies_compose() {
    assert!(!OrderingPolicy::None.is_topological());
    assert!(!OrderingPolicy::None.is_streamed());

    assert!(OrderingPolicy::Topological.is_topological());
    assert!(!OrderingPolicy::Topological.is_streamed());

    assert!(!OrderingPolicy::Stream.is_topological());
    assert!(OrderingPolicy::Stream.is_streamed());

    assert!(OrderingPolicy::TopologicalAndStream.is_topological());
    assert!(OrderingPolicy::TopologicalAndStream.is_streamed());
}

#[test]
fn ephemeral_latest_is_volatile_and_unversioned() {
    let presence = class(
        Retention::EphemeralLatest,
        Durability::Volatile,
        OrderingPolicy::None,
    );
    assert!(presence.supersedes());
    assert!(!presence.retains_prior_versions());
    assert_eq!(presence.version_limit(), Some(1));
    assert!(presence.is_volatile());
    assert!(!presence.is_synchronous());
    assert!(!presence.is_streamed());
}

#[test]
fn versioned_records_are_unbounded_and_durable() {
    let state = class(
        Retention::Versioned,
        Durability::Synchronous,
        OrderingPolicy::None,
    );
    assert!(state.retains_prior_versions());
    assert_eq!(state.version_limit(), None);
    assert!(state.supersedes());
}

#[test]
fn bounded_retention_reports_its_limit() {
    let bounded = class(
        Retention::Bounded(4),
        Durability::GroupCommit,
        OrderingPolicy::None,
    );
    assert!(bounded.retains_prior_versions());
    assert_eq!(bounded.version_limit(), Some(4));

    let single = class(
        Retention::Bounded(1),
        Durability::GroupCommit,
        OrderingPolicy::None,
    );
    assert!(!single.retains_prior_versions());
}
