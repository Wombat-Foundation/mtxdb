use super::*;

const ROOM: &str = "!room:example.org";

const DERIVATION: StateGroupDerivation =
    StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport);

fn lthash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn parent(byte: u8) -> StateGroupId {
    [byte; 16]
}

#[test]
fn derivation_is_replay_idempotent() {
    let a = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9), DERIVATION);
    let b = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9), DERIVATION);
    assert_eq!(a, b);
}

#[test]
fn parent_order_and_duplicates_do_not_matter() {
    let a = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9), DERIVATION);
    let b = state_group_instance_id(ROOM, &[parent(2), parent(1)], &lthash(9), DERIVATION);
    let c = state_group_instance_id(
        ROOM,
        &[parent(1), parent(1), parent(2)],
        &lthash(9),
        DERIVATION,
    );
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn equal_lthash_with_different_parents_stays_distinct() {
    let a = state_group_instance_id(ROOM, &[parent(1)], &lthash(9), DERIVATION);
    let b = state_group_instance_id(ROOM, &[parent(2)], &lthash(9), DERIVATION);
    assert_ne!(a, b);
}

#[test]
fn different_rooms_stay_distinct() {
    let a = state_group_instance_id(ROOM, &[parent(1)], &lthash(9), DERIVATION);
    let b = state_group_instance_id("!other:example.org", &[parent(1)], &lthash(9), DERIVATION);
    assert_ne!(a, b);
}

#[test]
fn instance_id_is_the_truncated_full_id() {
    let parents = [parent(1), parent(2)];
    let full = state_group_instance_full_id(ROOM, &parents, &lthash(9), DERIVATION);
    let id = state_group_instance_id(ROOM, &parents, &lthash(9), DERIVATION);
    assert_eq!(&id[..], &full[..16]);
}

#[test]
fn record_round_trips_through_encode_decode() {
    let instance = StateGroupInstance {
        parents: vec![parent(2), parent(1)],
        lthash: lthash(9),
        root_id: parent(7),
        derivation: DERIVATION,
    };
    let bytes = encode_state_group_record(&instance);
    assert_eq!(&bytes[..5], b"STGP\x01");
    let decoded = decode_state_group_record(&bytes).unwrap();
    assert_eq!(decoded.lthash, instance.lthash);
    assert_eq!(decoded.root_id, instance.root_id);
    assert_eq!(decoded.parents, vec![parent(1), parent(2)]);
    assert_eq!(decoded.id(ROOM), instance.id(ROOM));
}

#[test]
fn record_encoding_canonicalizes_parent_order_and_duplicates() {
    let a = StateGroupInstance {
        parents: vec![parent(1), parent(2)],
        lthash: lthash(9),
        root_id: parent(7),
        derivation: DERIVATION,
    };
    let b = StateGroupInstance {
        parents: vec![parent(2), parent(1), parent(1)],
        lthash: lthash(9),
        root_id: parent(7),
        derivation: DERIVATION,
    };
    assert_eq!(encode_state_group_record(&a), encode_state_group_record(&b));
}

#[test]
fn record_decode_rejects_malformed_input() {
    assert!(decode_state_group_record(b"").is_err());
    assert!(decode_state_group_record(b"MTHR\x01").is_err());
    let mut wrong_version = encode_state_group_record(&StateGroupInstance {
        parents: vec![],
        lthash: lthash(1),
        root_id: parent(2),
        derivation: DERIVATION,
    });
    wrong_version[4] = 0x00;
    assert!(decode_state_group_record(&wrong_version).is_err());
    let mut bad_resolver = encode_state_group_record(&StateGroupInstance {
        parents: vec![],
        lthash: lthash(1),
        root_id: parent(2),
        derivation: DERIVATION,
    });
    bad_resolver[5] = 0xff;
    assert!(decode_state_group_record(&bad_resolver).is_err());
    let mut bad_trust = encode_state_group_record(&StateGroupInstance {
        parents: vec![],
        lthash: lthash(1),
        root_id: parent(2),
        derivation: DERIVATION,
    });
    bad_trust[6] = 0xff;
    assert!(decode_state_group_record(&bad_trust).is_err());
    let truncated = &encode_state_group_record(&StateGroupInstance {
        parents: vec![parent(1)],
        lthash: lthash(1),
        root_id: parent(2),
        derivation: DERIVATION,
    })[..20];
    assert!(decode_state_group_record(truncated).is_err());
}

#[test]
fn collection_id_is_per_room_and_namespaced() {
    let a = state_group_collection_id(ROOM).unwrap();
    let b = state_group_collection_id("!other:example.org").unwrap();
    assert_ne!(a, b);
    assert_eq!(
        a,
        derive_group_member_collection_id(MEMBER_NAMESPACE_STGP, ROOM.as_bytes()).unwrap()
    );
}

#[test]
fn relation_arity_and_derivation_are_explicit() {
    assert!(!StateGroupRelation::Instance.is_many());
    assert!(!StateGroupRelation::Instance.is_derived());
    assert!(!StateGroupRelation::InstanceByEvent.is_many());
    assert!(!StateGroupRelation::InstanceByEvent.is_derived());

    assert!(StateGroupRelation::InstancesByLtHash.is_many());
    assert!(StateGroupRelation::InstancesByLtHash.is_derived());
    assert!(StateGroupRelation::ChildrenByParent.is_many());
    assert!(StateGroupRelation::ChildrenByParent.is_derived());

    assert_eq!(StateGroupRelation::ALL.len(), 4);
}

#[test]
fn instance_struct_id_matches_free_function() {
    let instance = StateGroupInstance {
        parents: vec![parent(1), parent(2)],
        lthash: lthash(9),
        root_id: parent(7),
        derivation: DERIVATION,
    };
    assert_eq!(
        instance.id(ROOM),
        state_group_instance_id(
            ROOM,
            &instance.parents,
            &instance.lthash,
            instance.derivation
        )
    );
}

#[test]
fn resolver_version_separates_instance_ids() {
    let parents = [parent(1), parent(2)];
    let hash = lthash(9);
    let v1 = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(StateGroupResolver::V1, StateGroupTrust::TrustedImport),
    );
    let v2 = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport),
    );
    let v2_1 = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(StateGroupResolver::V2_1, StateGroupTrust::TrustedImport),
    );
    let first_parent = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(
            StateGroupResolver::FirstParent,
            StateGroupTrust::TrustedImport,
        ),
    );
    assert_ne!(v1, v2);
    assert_ne!(v2, v2_1);
    assert_ne!(first_parent, v2);
}

#[test]
fn trust_policy_separates_instance_ids() {
    let parents = [parent(1), parent(2)];
    let hash = lthash(9);
    let trusted = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport),
    );
    let validated = state_group_instance_id(
        ROOM,
        &parents,
        &hash,
        StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::ValidatedImport),
    );
    assert_ne!(trusted, validated);
}

#[test]
fn derivation_tags_round_trip_through_the_record() {
    let instance = StateGroupInstance {
        parents: vec![parent(3)],
        lthash: lthash(4),
        root_id: parent(5),
        derivation: StateGroupDerivation::new(
            StateGroupResolver::V2_1,
            StateGroupTrust::ValidatedImport,
        ),
    };
    let decoded = decode_state_group_record(&encode_state_group_record(&instance)).unwrap();
    assert_eq!(decoded.derivation, instance.derivation);
    assert_eq!(
        decoded.derivation.resolver.as_str(),
        StateGroupResolver::V2_1.as_str()
    );
    assert_eq!(
        decoded.derivation.trust.as_str(),
        StateGroupTrust::ValidatedImport.as_str()
    );
}

#[test]
fn derivation_strings_are_stable() {
    assert_eq!(
        StateGroupDerivation::new(StateGroupResolver::V2, StateGroupTrust::TrustedImport).as_str(),
        "state-res-v2/trusted-import"
    );
    assert_eq!(StateGroupResolver::FirstParent.as_str(), "first-parent");
    assert_eq!(StateGroupTrust::Replay.as_str(), "replay");
}
