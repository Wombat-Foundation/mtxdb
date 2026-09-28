use super::*;

#[test]
#[cfg(feature = "multi-reader")]
fn matrix_pool_policies_skips_state_compression() {
    let policies = matrix_pool_policies();
    assert!(!policies.state.compress);
    assert!(policies.event_dag.compress);
    assert!(policies.edges.compress);
    assert_eq!(
        policies.state.checksum_policy,
        crate::packfile::ChecksumPolicy::Full
    );
}

#[test]
fn v1_v2_v3_identity_boundaries_are_explicit() {
    assert_eq!(
        MatrixRoomVersion::V1.event_id_policy(),
        EventIdPolicy::ServerAssigned
    );
    assert_eq!(
        MatrixRoomVersion::V2.state_resolution_policy(),
        StateResolutionPolicy::V2
    );
    assert_eq!(
        MatrixRoomVersion::V1.reference_hash_input_policy(),
        ReferenceHashInputPolicy::NotApplicable
    );
    assert_eq!(
        MatrixRoomVersion::V3.event_id_policy(),
        EventIdPolicy::ReferenceHash
    );
    assert!(!MatrixRoomVersion::V3.requires_strict_canonical_numbers());
    assert_eq!(
        MatrixRoomVersion::V4.reference_hash_encoding(),
        ReferenceHashEncoding::UrlSafeBase64NoPad
    );
    assert_eq!(
        MatrixRoomVersion::V6.redaction_policy(),
        RedactionPolicy::V6ToV8
    );
    assert!(MatrixRoomVersion::V8.supports_restricted_join());
    assert!(MatrixRoomVersion::V10.supports_knock_restricted_join());
    assert_eq!(
        MatrixRoomVersion::V12.room_id_policy(),
        RoomIdPolicy::CreateEventId
    );
    assert_eq!(
        MatrixRoomVersion::V12.state_resolution_policy(),
        StateResolutionPolicy::V2_1
    );
}

/// A reference hash is over the redacted event, so its input policy must
/// split at the same v11 boundary the redaction algorithm does. v9/v10
/// and v11/v12 must not collapse to one variant.
#[test]
fn reference_hash_input_policy_splits_at_v11() {
    for version in [MatrixRoomVersion::V9, MatrixRoomVersion::V10] {
        assert_eq!(version.redaction_policy(), RedactionPolicy::V9ToV10);
        assert_eq!(
            version.reference_hash_input_policy(),
            ReferenceHashInputPolicy::V9ToV10
        );
    }
    for version in [MatrixRoomVersion::V11, MatrixRoomVersion::V12] {
        assert_eq!(version.redaction_policy(), RedactionPolicy::V11Plus);
        assert_eq!(
            version.reference_hash_input_policy(),
            ReferenceHashInputPolicy::V11Plus
        );
    }
}

/// The support floor is v4: v1/v2 are server-assigned and v3 uses
/// non-URL-safe base64. The base64url shift is exactly the v3/v4 boundary.
#[test]
fn support_floor_is_v4() {
    assert_eq!(MatrixRoomVersion::MIN_SUPPORTED, MatrixRoomVersion::V4);
    for version in [
        MatrixRoomVersion::V1,
        MatrixRoomVersion::V2,
        MatrixRoomVersion::V3,
    ] {
        assert!(!version.is_supported());
    }
    for version in [
        MatrixRoomVersion::V4,
        MatrixRoomVersion::V11,
        MatrixRoomVersion::V12,
    ] {
        assert!(version.is_supported());
    }
    assert_eq!(
        MatrixRoomVersion::V3.reference_hash_encoding(),
        ReferenceHashEncoding::StandardBase64NoPad
    );
    assert_eq!(
        MatrixRoomVersion::V4.reference_hash_encoding(),
        ReferenceHashEncoding::UrlSafeBase64NoPad
    );
}

/// The collection key source switches exactly where the room-id policy
/// does: v12 reads the create event's `/event_id`, earlier versions read
/// its server-assigned `/room_id`.
#[test]
fn collection_key_source_switches_at_v12() {
    for version in [MatrixRoomVersion::V4, MatrixRoomVersion::V11] {
        assert_eq!(version.room_id_policy(), RoomIdPolicy::ServerAssigned);
        assert_eq!(version.collection_key_pointer(), "/room_id");
    }
    assert_eq!(
        MatrixRoomVersion::V12.room_id_policy(),
        RoomIdPolicy::CreateEventId
    );
    assert_eq!(MatrixRoomVersion::V12.collection_key_pointer(), "/event_id");
}

/// A v12 create event id and the `room_id` ordinary events carry name the
/// same collection; the sigil swap is what makes the follow-up batch route.
#[test]
fn v12_normalizes_the_create_event_id_into_the_room_id_form() {
    assert_eq!(
        MatrixRoomVersion::V12.normalize_collection_identity("$DGMOhash"),
        "!DGMOhash"
    );
    // A server-assigned room id is already the referenced form.
    assert_eq!(
        MatrixRoomVersion::V10.normalize_collection_identity("!room:server"),
        "!room:server"
    );
    // Already-normalized or unexpected input is left untouched.
    assert_eq!(
        MatrixRoomVersion::V12.normalize_collection_identity("!already"),
        "!already"
    );
}
