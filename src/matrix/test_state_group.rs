use super::*;

const ROOM: &str = "!room:example.org";

fn lthash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn parent(byte: u8) -> StateGroupId {
    [byte; 16]
}

#[test]
fn derivation_is_replay_idempotent() {
    let a = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9));
    let b = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9));
    assert_eq!(a, b);
}

#[test]
fn parent_order_and_duplicates_do_not_matter() {
    let a = state_group_instance_id(ROOM, &[parent(1), parent(2)], &lthash(9));
    let b = state_group_instance_id(ROOM, &[parent(2), parent(1)], &lthash(9));
    let c = state_group_instance_id(ROOM, &[parent(1), parent(1), parent(2)], &lthash(9));
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn equal_lthash_with_different_parents_stays_distinct() {
    let a = state_group_instance_id(ROOM, &[parent(1)], &lthash(9));
    let b = state_group_instance_id(ROOM, &[parent(2)], &lthash(9));
    assert_ne!(a, b);
}

#[test]
fn different_rooms_stay_distinct() {
    let a = state_group_instance_id(ROOM, &[parent(1)], &lthash(9));
    let b = state_group_instance_id("!other:example.org", &[parent(1)], &lthash(9));
    assert_ne!(a, b);
}

#[test]
fn instance_id_is_the_truncated_full_id() {
    let parents = [parent(1), parent(2)];
    let full = state_group_instance_full_id(ROOM, &parents, &lthash(9));
    let id = state_group_instance_id(ROOM, &parents, &lthash(9));
    assert_eq!(&id[..], &full[..16]);
}

#[test]
fn record_round_trips_through_encode_decode() {
    let instance = StateGroupInstance {
        parents: vec![parent(2), parent(1)],
        lthash: lthash(9),
        root_id: parent(7),
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
    };
    let b = StateGroupInstance {
        parents: vec![parent(2), parent(1), parent(1)],
        lthash: lthash(9),
        root_id: parent(7),
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
    });
    wrong_version[4] = 0x02;
    assert!(decode_state_group_record(&wrong_version).is_err());
    let truncated = &encode_state_group_record(&StateGroupInstance {
        parents: vec![parent(1)],
        lthash: lthash(1),
        root_id: parent(2),
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
    };
    assert_eq!(
        instance.id(ROOM),
        state_group_instance_id(ROOM, &instance.parents, &instance.lthash)
    );
}
