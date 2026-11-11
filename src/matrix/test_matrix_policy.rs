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
fn server_info_pool_keeps_full_checksum_verification() {
    let policies = matrix_pool_policies();
    let policy = policies.for_shard(crate::layout::ShardType::ServerInfo);
    assert_eq!(
        policy.checksum_policy,
        crate::packfile::ChecksumPolicy::Full
    );
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

#[test]
fn matrix_pools_declare_which_layout_repack_gives_them() {
    use crate::layout::ShardType;
    use crate::record_class::OrderingPolicy;
    let policies = super::matrix_pool_policies();
    assert!(policies
        .for_shard(ShardType::State)
        .ordering
        .is_topological());
    assert!(policies
        .for_shard(ShardType::EventDag)
        .ordering
        .is_topological());
    assert_eq!(
        policies.for_shard(ShardType::ServerInfo).ordering,
        OrderingPolicy::None
    );
}

const ALL_VERSIONS: [(&str, MatrixRoomVersion); 12] = [
    ("1", MatrixRoomVersion::V1),
    ("2", MatrixRoomVersion::V2),
    ("3", MatrixRoomVersion::V3),
    ("4", MatrixRoomVersion::V4),
    ("5", MatrixRoomVersion::V5),
    ("6", MatrixRoomVersion::V6),
    ("7", MatrixRoomVersion::V7),
    ("8", MatrixRoomVersion::V8),
    ("9", MatrixRoomVersion::V9),
    ("10", MatrixRoomVersion::V10),
    ("11", MatrixRoomVersion::V11),
    ("12", MatrixRoomVersion::V12),
];

#[test]
fn room_version_strings_parse_and_unknown_ones_do_not() {
    for (text, version) in ALL_VERSIONS {
        assert_eq!(MatrixRoomVersion::parse(text), Some(version), "{text}");
    }
    assert_eq!(
        MatrixRoomVersion::parse("12.1"),
        Some(MatrixRoomVersion::V12)
    );
    for bad in [
        "",
        "0",
        "unknown_roomversion_99",
        "v4",
        "04",
        "4 ",
        "org.example.custom",
    ] {
        assert_eq!(MatrixRoomVersion::parse(bad), None, "{bad:?}");
    }
}

#[test]
fn event_identity_rules_switch_at_the_documented_versions() {
    use MatrixRoomVersion as V;
    for (_, v) in ALL_VERSIONS {
        let server_assigned = matches!(v, V::V1 | V::V2);
        assert_eq!(
            v.event_id_policy() == EventIdPolicy::ServerAssigned,
            server_assigned
        );
        assert_eq!(
            v.reference_hash_input_policy() == ReferenceHashInputPolicy::NotApplicable,
            server_assigned
        );
        assert_eq!(
            v.reference_hash_encoding() == ReferenceHashEncoding::NotApplicable,
            server_assigned
        );
    }
    assert_eq!(
        V::V3.reference_hash_encoding(),
        ReferenceHashEncoding::StandardBase64NoPad
    );
    assert_eq!(
        V::V4.reference_hash_encoding(),
        ReferenceHashEncoding::UrlSafeBase64NoPad
    );
}

#[test]
fn reference_hash_input_tracks_the_redaction_policy_exactly() {
    use MatrixRoomVersion as V;
    for (_, v) in ALL_VERSIONS {
        if v.event_id_policy() == EventIdPolicy::ServerAssigned {
            continue;
        }
        let same = matches!(
            (v.redaction_policy(), v.reference_hash_input_policy()),
            (RedactionPolicy::V1ToV5, ReferenceHashInputPolicy::V1ToV5)
                | (RedactionPolicy::V6ToV8, ReferenceHashInputPolicy::V6ToV8)
                | (RedactionPolicy::V9ToV10, ReferenceHashInputPolicy::V9ToV10)
                | (RedactionPolicy::V11Plus, ReferenceHashInputPolicy::V11Plus)
        );
        assert!(same, "{v:?} hashes a different shape than it redacts");
    }
    assert_eq!(V::V5.redaction_policy(), RedactionPolicy::V1ToV5);
    assert_eq!(V::V6.redaction_policy(), RedactionPolicy::V6ToV8);
    assert_eq!(V::V9.redaction_policy(), RedactionPolicy::V9ToV10);
    assert_eq!(V::V11.redaction_policy(), RedactionPolicy::V11Plus);
}

#[test]
fn state_resolution_room_id_and_feature_gates_follow_the_spec() {
    use MatrixRoomVersion as V;
    assert_eq!(V::V1.state_resolution_policy(), StateResolutionPolicy::V1);
    assert_eq!(V::V11.state_resolution_policy(), StateResolutionPolicy::V2);
    assert_eq!(
        V::V12.state_resolution_policy(),
        StateResolutionPolicy::V2_1
    );

    for (_, v) in ALL_VERSIONS {
        let v12 = v == V::V12;
        assert_eq!(v.room_id_policy() == RoomIdPolicy::CreateEventId, v12);
        assert_eq!(
            v.collection_key_pointer(),
            if v12 { "/event_id" } else { "/room_id" }
        );
        let n = ALL_VERSIONS.iter().position(|(_, x)| *x == v).unwrap() + 1;
        assert_eq!(v.is_supported(), n >= 4, "{v:?}");
        assert_eq!(v.requires_strict_canonical_numbers(), n >= 6);
        assert_eq!(v.supports_knock(), n >= 7);
        assert_eq!(v.supports_restricted_join(), n >= 8);
        assert_eq!(v.supports_knock_restricted_join(), n >= 10);
    }
    assert_eq!(V::MIN_SUPPORTED, V::V4);
}

#[test]
fn collection_identity_is_normalized_only_for_room_version_12() {
    use MatrixRoomVersion as V;
    assert_eq!(V::V12.normalize_collection_identity("$abc"), "!abc");
    assert_eq!(
        V::V12.normalize_collection_identity("!abc"),
        "!abc",
        "an already-normalized id is left alone"
    );
    assert_eq!(V::V11.normalize_collection_identity("$abc"), "$abc");
    assert_eq!(V::V4.normalize_collection_identity("!r:x"), "!r:x");
}
