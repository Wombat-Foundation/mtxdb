use super::*;

#[test]
fn presence_is_ephemeral_and_volatile() {
    let class = MatrixRecordClass::Presence.record_class();
    assert_eq!(class.retention, Retention::EphemeralLatest);
    assert_eq!(class.durability, Durability::Volatile);
    assert_eq!(class.ordering, OrderingPolicy::None);
    assert!(class.is_volatile());
    assert!(class.supersedes());
    assert!(!class.retains_prior_versions());
}

#[test]
fn events_are_immutable_durable_and_streamed() {
    let class = MatrixRecordClass::Events.record_class();
    assert_eq!(class.retention, Retention::Immutable);
    assert_eq!(class.durability, Durability::Synchronous);
    assert_eq!(class.ordering, OrderingPolicy::TopologicalAndStream);
    assert!(!class.supersedes());
    assert!(class.is_topological());
    assert!(class.is_streamed());
}

#[test]
fn current_state_is_versioned_and_synchronous() {
    let class = MatrixRecordClass::CurrentState.record_class();
    assert_eq!(class.retention, Retention::Versioned);
    assert!(class.retains_prior_versions());
    assert!(class.is_synchronous());
    assert_eq!(class.version_limit(), None);
}

#[test]
fn receipts_account_data_and_state_group_links_have_distinct_policies() {
    let receipts = MatrixRecordClass::Receipts.record_class();
    assert_eq!(receipts.retention, Retention::LatestOnly);
    assert_eq!(receipts.durability, Durability::GroupCommit);
    // Stream-ordered for pagination, but not a causal DAG.
    assert_eq!(receipts.ordering, OrderingPolicy::Stream);
    assert!(receipts.is_streamed());
    assert!(!receipts.is_topological());

    let account = MatrixRecordClass::AccountData.record_class();
    assert_eq!(account.retention, Retention::Versioned);
    assert_eq!(account.durability, Durability::GroupCommit);

    // Compact link records are write-once, so a mapping is never overwritten.
    let link = MatrixRecordClass::StateGroupMapping.record_class();
    assert_eq!(link.retention, Retention::Immutable);
    assert!(!link.supersedes());
}

#[test]
fn state_groups_are_topological_but_the_mapping_is_not() {
    let group = MatrixRecordClass::StateGroup.record_class();
    assert_eq!(group.retention, Retention::Immutable);
    assert_eq!(group.durability, Durability::Synchronous);
    assert_eq!(group.ordering, OrderingPolicy::Topological);
    assert!(group.is_topological());
    assert!(!group.is_streamed());

    // The event -> state-group edge is a lookup, so it carries no ordering.
    let mapping = MatrixRecordClass::StateGroupMapping.record_class();
    assert_eq!(mapping.ordering, OrderingPolicy::None);
    assert!(!mapping.is_topological());
    assert!(!mapping.is_streamed());
}

#[test]
fn hamt_content_is_shared_and_unordered() {
    let content = MatrixRecordClass::StateHamtContent.record_class();
    assert_eq!(content.retention, Retention::Immutable);
    assert_eq!(content.durability, Durability::Synchronous);
    assert_eq!(content.ordering, OrderingPolicy::None);
    assert!(!content.is_topological());
    assert!(!content.is_streamed());
}
