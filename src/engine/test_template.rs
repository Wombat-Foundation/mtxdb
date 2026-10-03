use super::*;

fn sha256(data: &[u8]) -> Digest32 {
    DigestAlgorithm::Sha256.digest(data)
}

#[test]
fn generic_template_does_not_require_a_protocol_extension() {
    let template = CollectionTemplate {
        name: "documents".into(),
        collection_kind: "notebook".into(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/uuid".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        collection_key: CollectionKeyRule {
            pointer: "/notebook".into(),
            member_namespace: Some(MEMBER_NAMESPACE_INTL),
            display_id_pointer: "/notebook".into(),
        },
        establishment: None,
    };

    assert_eq!(template.payload, PayloadPolicy::Source);
    assert_eq!(template.collection_key.display_id_pointer, "/notebook");
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::cast_possible_truncation
)]
fn reference_wrapping_add_le(a: [u8; 32], b: [u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry: u16 = 0;
    let mut i = 0;
    while i < 32 {
        let sum = (a[i] as u16) + (b[i] as u16) + carry;
        result[i] = sum as u8;
        carry = sum >> 8;
        i += 1;
    }
    result
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_wrapping_add_le() {
    let zero = [0u8; 32];
    let mut one = [0u8; 32];
    one[0] = 1;

    // 1. zero + zero == zero
    assert_eq!(wrapping_add_le(zero, zero), zero);

    // 2. zero + x == x and x + zero == x
    assert_eq!(wrapping_add_le(zero, one), one);
    assert_eq!(wrapping_add_le(one, zero), one);

    // 3. max + 1 -> 0 (ripple carry across all 32 bytes / all four limbs)
    let max_bytes = [0xffu8; 32];
    assert_eq!(wrapping_add_le(max_bytes, one), zero);

    // 4. max + max -> max - 1: (2^256 - 1) + (2^256 - 1) = 2^256 - 2 mod 2^256
    let mut max_minus_one = [0xffu8; 32];
    max_minus_one[0] = 0xfe;
    assert_eq!(wrapping_add_le(max_bytes, max_bytes), max_minus_one);

    // 5. Carry across each limb:
    // Limb 0 -> 1 carry
    let mut limb0_max = [0u8; 32];
    limb0_max[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let sum_limb0 = wrapping_add_le(limb0_max, one);
    let mut expected_limb1 = [0u8; 32];
    expected_limb1[8..16].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(sum_limb0, expected_limb1);

    // Limb 1 -> 2 carry
    let mut limb1_max = [0u8; 32];
    limb1_max[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut limb1_one = [0u8; 32];
    limb1_one[8..16].copy_from_slice(&1u64.to_le_bytes());
    let sum_limb1 = wrapping_add_le(limb1_max, limb1_one);
    let mut expected_limb2 = [0u8; 32];
    expected_limb2[16..24].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(sum_limb1, expected_limb2);

    // Limb 2 -> 3 carry
    let mut limb2_max = [0u8; 32];
    limb2_max[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut limb2_one = [0u8; 32];
    limb2_one[16..24].copy_from_slice(&1u64.to_le_bytes());
    let sum_limb2 = wrapping_add_le(limb2_max, limb2_one);
    let mut expected_limb3 = [0u8; 32];
    expected_limb3[24..32].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(sum_limb2, expected_limb3);

    // Limb 3 -> wrap to 0
    let mut limb3_max = [0u8; 32];
    limb3_max[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut limb3_one = [0u8; 32];
    limb3_one[24..32].copy_from_slice(&1u64.to_le_bytes());
    let sum_limb3 = wrapping_add_le(limb3_max, limb3_one);
    assert_eq!(sum_limb3, zero);

    // 6. Test carry-in propagation (c1b, c2b, c3b paths where a_k + b_k does not overflow, but + carry does)
    // a has limb 0 = u64::MAX, limb 1 = u64::MAX, limb 2 = 0, limb 3 = 0
    // b has limb 0 = 1, limb 1 = 0, limb 2 = 0, limb 3 = 0
    let mut a_c1b = [0u8; 32];
    a_c1b[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    a_c1b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let sum_c1b = wrapping_add_le(a_c1b, one);
    let mut expected_c1b = [0u8; 32];
    expected_c1b[16..24].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(sum_c1b, expected_c1b);

    // 7. Ripple carry across all 4 limbs: limbs 0,1,2 = MAX, b = 1
    let mut a_cascade = [0u8; 32];
    a_cascade[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    a_cascade[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    a_cascade[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let sum_cascade = wrapping_add_le(a_cascade, one);
    let mut expected_cascade = [0u8; 32];
    expected_cascade[24..32].copy_from_slice(&1u64.to_le_bytes());
    assert_eq!(sum_cascade, expected_cascade);

    // 8. Differential testing against reference byte-wise implementation
    // Deterministic PRNG (xorshift64) to generate 500 pseudo-random 32-byte pairs
    let mut state: u64 = 0x853c_49e6_748f_ea9b;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for _ in 0..500 {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        for i in 0..4 {
            a[i * 8..(i + 1) * 8].copy_from_slice(&next_u64().to_le_bytes());
            b[i * 8..(i + 1) * 8].copy_from_slice(&next_u64().to_le_bytes());
        }
        let fast = wrapping_add_le(a, b);
        let reference = reference_wrapping_add_le(a, b);
        assert_eq!(
            fast, reference,
            "mismatch between 4-limb and reference byte-wise add"
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn fixed_derivation_vectors() {
    let room = b"!room:example.com";
    let expected_group_full: [u8; 32] = [
        0x2e, 0x1f, 0xe2, 0x76, 0x68, 0x5e, 0x3f, 0x82, 0x98, 0x70, 0xb5, 0x95, 0x6b, 0xa3, 0x53,
        0xec, 0x31, 0x94, 0x4f, 0x48, 0x8d, 0xe6, 0x22, 0x38, 0xdc, 0xef, 0x2b, 0xf8, 0x31, 0xc3,
        0xef, 0xc9,
    ];
    assert_eq!(derive_group_full_id(room), expected_group_full);

    // EVNT
    let expected_evnt_full: [u8; 32] = [
        0x95, 0x07, 0xbe, 0x45, 0xc4, 0x92, 0xc4, 0x38, 0x2e, 0xdb, 0x18, 0xfc, 0x07, 0x44, 0x77,
        0xcc, 0x69, 0x62, 0x6e, 0x77, 0x81, 0x20, 0xb9, 0x23, 0xac, 0x48, 0xb4, 0x53, 0x07, 0x25,
        0xa1, 0x9c,
    ];
    let expected_evnt_id: [u8; 16] = [
        0x95, 0x07, 0xbe, 0x45, 0xc4, 0x92, 0xc4, 0x38, 0x2e, 0xdb, 0x18, 0xfc, 0x07, 0x44, 0x77,
        0xcc,
    ];
    assert_eq!(
        derive_group_member_full_id(MEMBER_NAMESPACE_EVNT, room),
        Some(expected_evnt_full)
    );
    assert_eq!(
        derive_group_member_collection_id(MEMBER_NAMESPACE_EVNT, room),
        Some(expected_evnt_id)
    );

    // PREV
    let expected_prev_full: [u8; 32] = [
        0xfe, 0xa9, 0xa5, 0x8f, 0x0a, 0xb8, 0x69, 0x31, 0x7d, 0x69, 0x60, 0x93, 0xfd, 0x7c, 0x61,
        0x73, 0xb4, 0xc8, 0xcd, 0x4c, 0x35, 0xc6, 0x2a, 0x0f, 0xa9, 0x76, 0x26, 0x88, 0xa9, 0x5c,
        0x76, 0x7c,
    ];
    let expected_prev_id: [u8; 16] = [
        0xfe, 0xa9, 0xa5, 0x8f, 0x0a, 0xb8, 0x69, 0x31, 0x7d, 0x69, 0x60, 0x93, 0xfd, 0x7c, 0x61,
        0x73,
    ];
    assert_eq!(
        derive_group_member_full_id(MEMBER_NAMESPACE_PREV, room),
        Some(expected_prev_full)
    );
    assert_eq!(
        derive_group_member_collection_id(MEMBER_NAMESPACE_PREV, room),
        Some(expected_prev_id)
    );

    // AUTH
    let expected_auth_full: [u8; 32] = [
        0x48, 0x42, 0xa2, 0xef, 0xf3, 0xcf, 0x51, 0x1f, 0x9e, 0x48, 0x40, 0x63, 0x50, 0x25, 0x3c,
        0x77, 0xcc, 0xa7, 0x83, 0x76, 0x0d, 0xff, 0x67, 0x46, 0x27, 0x85, 0xfd, 0x7f, 0x57, 0xb7,
        0xa6, 0x41,
    ];
    let expected_auth_id: [u8; 16] = [
        0x48, 0x42, 0xa2, 0xef, 0xf3, 0xcf, 0x51, 0x1f, 0x9e, 0x48, 0x40, 0x63, 0x50, 0x25, 0x3c,
        0x77,
    ];
    assert_eq!(
        derive_group_member_full_id(MEMBER_NAMESPACE_AUTH, room),
        Some(expected_auth_full)
    );
    assert_eq!(
        derive_group_member_collection_id(MEMBER_NAMESPACE_AUTH, room),
        Some(expected_auth_id)
    );

    // STAT
    let expected_stat_full: [u8; 32] = [
        0xe5, 0x60, 0xc7, 0xdd, 0x2c, 0xee, 0xe5, 0xd7, 0xd3, 0x1c, 0xa9, 0xd5, 0x6e, 0x6e, 0x9e,
        0x9f, 0x77, 0x31, 0xd1, 0xf6, 0x5b, 0xc5, 0xb6, 0xd7, 0xf2, 0xad, 0x1e, 0x2e, 0x29, 0x4d,
        0xdb, 0x47,
    ];
    let expected_stat_id: [u8; 16] = [
        0xe5, 0x60, 0xc7, 0xdd, 0x2c, 0xee, 0xe5, 0xd7, 0xd3, 0x1c, 0xa9, 0xd5, 0x6e, 0x6e, 0x9e,
        0x9f,
    ];
    assert_eq!(
        derive_group_member_full_id(MEMBER_NAMESPACE_STAT, room),
        Some(expected_stat_full)
    );
    assert_eq!(
        derive_group_member_collection_id(MEMBER_NAMESPACE_STAT, room),
        Some(expected_stat_id)
    );

    // INTL
    let expected_intl_full: [u8; 32] = [
        0x44, 0xf6, 0xc1, 0xd9, 0x3c, 0x12, 0xce, 0xfa, 0xb1, 0x7b, 0x52, 0x1f, 0xc1, 0xa8, 0xa5,
        0x83, 0x06, 0xc3, 0x44, 0x8d, 0x6d, 0x08, 0xd2, 0x74, 0x74, 0x34, 0xd0, 0xae, 0x8e, 0x27,
        0x28, 0xad,
    ];
    let expected_intl_id: [u8; 16] = [
        0x44, 0xf6, 0xc1, 0xd9, 0x3c, 0x12, 0xce, 0xfa, 0xb1, 0x7b, 0x52, 0x1f, 0xc1, 0xa8, 0xa5,
        0x83,
    ];
    assert_eq!(
        derive_group_member_full_id(MEMBER_NAMESPACE_INTL, room),
        Some(expected_intl_full)
    );
    assert_eq!(
        derive_group_member_collection_id(MEMBER_NAMESPACE_INTL, room),
        Some(expected_intl_id)
    );

    // Unknown namespaces (such as physical pool tags like EDGE) MUST be rejected
    assert_eq!(namespace_bias(*b"EDGE"), None);
    assert_eq!(namespace_bias(*b"XYZW"), None);
    assert_eq!(namespace_bias(*b"    "), None);
    assert_eq!(derive_group_member_full_id(*b"EDGE", room), None);
    assert_eq!(derive_group_member_collection_id(*b"EDGE", room), None);
    assert_eq!(try_derive_collection_id(Some(*b"EDGE"), room), None);

    // All member collection IDs are strictly separated
    let all_ids = [
        expected_evnt_id,
        expected_prev_id,
        expected_auth_id,
        expected_stat_id,
        expected_intl_id,
    ];
    for i in 0..all_ids.len() {
        for j in (i + 1)..all_ids.len() {
            assert_ne!(all_ids[i], all_ids[j]);
        }
    }

    // Direct derivation from group_full_id without re-hashing
    assert_eq!(
        derive_member_full_id_from_group(MEMBER_NAMESPACE_EVNT, expected_group_full),
        Some(expected_evnt_full)
    );
    assert_eq!(
        derive_member_collection_id_from_group(MEMBER_NAMESPACE_EVNT, expected_group_full),
        Some(expected_evnt_id)
    );
    assert_eq!(
        derive_member_collection_id_from_group(MEMBER_NAMESPACE_PREV, expected_group_full),
        Some(expected_prev_id)
    );
    assert_eq!(
        derive_member_collection_id_from_group(MEMBER_NAMESPACE_AUTH, expected_group_full),
        Some(expected_auth_id)
    );
    assert_eq!(
        derive_member_collection_id_from_group(MEMBER_NAMESPACE_STAT, expected_group_full),
        Some(expected_stat_id)
    );
    assert_eq!(
        derive_member_collection_id_from_group(MEMBER_NAMESPACE_INTL, expected_group_full),
        Some(expected_intl_id)
    );
}

#[test]
fn frame_digest_selects_its_input_per_policy() {
    let resolve = |pointer: &str| (pointer == "/uuid").then(|| b"pointer-bytes".to_vec());
    let input = FrameIdInput {
        payload: b"payload-bytes",
        descriptor: b"descriptor-bytes",
        canonical: Some(b"canonical-bytes"),
        resolve: &resolve,
    };

    // Each policy hashes exactly the bytes it selects.
    assert_eq!(
        frame_digest(&FrameIdPolicy::Payload, DigestAlgorithm::Sha256, &input),
        Some(sha256(b"payload-bytes"))
    );
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::HeaderDescriptor { fields: vec![] },
            DigestAlgorithm::Sha256,
            &input
        ),
        Some(sha256(b"descriptor-bytes"))
    );
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::Canonical {
                include: vec![],
                exclude_prefixes: vec![],
            },
            DigestAlgorithm::Sha256,
            &input
        ),
        Some(sha256(b"canonical-bytes"))
    );
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::Pointer {
                pointer: "/uuid".into(),
            },
            DigestAlgorithm::Sha256,
            &input
        ),
        Some(sha256(b"pointer-bytes"))
    );

    // Missing inputs and externally supplied identities yield `None`.
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::Pointer {
                pointer: "/absent".into(),
            },
            DigestAlgorithm::Sha256,
            &input
        ),
        None
    );
    let no_canonical = FrameIdInput {
        canonical: None,
        ..input
    };
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::Canonical {
                include: vec![],
                exclude_prefixes: vec![],
            },
            DigestAlgorithm::Sha256,
            &no_canonical
        ),
        None
    );
    assert_eq!(
        frame_digest(
            &FrameIdPolicy::ExternalCanonicalIdToCrosscheck,
            DigestAlgorithm::Sha256,
            &input
        ),
        None
    );
}

#[test]
fn record_logical_id_uses_the_first_16_digest_bytes() {
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::try_from(index).expect("test index fits in u8");
    }

    assert_eq!(
        record_logical_id(&digest),
        [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]
    );
}

#[test]
fn collection_id_is_deterministic_and_pool_separated() {
    let room = b"!room:matrix.org";
    let event_dag = Some(*b"EVNT");
    let state = Some(*b"STAT");
    let prev = Some(*b"PREV");
    let auth = Some(*b"AUTH");
    let forward = Some(*b"FWD ");
    assert_eq!(
        derive_collection_id(event_dag, room),
        derive_collection_id(event_dag, room)
    );
    // A different namespace tag yields a different id for the same key.
    assert_ne!(
        derive_collection_id(event_dag, room),
        derive_collection_id(state, room)
    );
    assert_ne!(
        derive_collection_id(prev, room),
        derive_collection_id(auth, room)
    );
    assert_ne!(
        derive_collection_id(forward, room),
        derive_collection_id(prev, room)
    );
    // A different key yields a different id for the same tag.
    assert_ne!(
        derive_collection_id(event_dag, room),
        derive_collection_id(event_dag, b"!other:matrix.org")
    );
    // No tag is a valid, deterministic derivation of its own.
    assert_eq!(
        derive_collection_id(None, room),
        derive_collection_id(None, room)
    );
    assert_ne!(
        derive_collection_id(None, room),
        derive_collection_id(event_dag, room)
    );

    // Fixed vector verification:
    let group_id = derive_group_full_id(room);
    let evnt_full = derive_group_member_full_id(*b"EVNT", room).unwrap();
    assert_eq!(evnt_full, wrapping_add_le(group_id, NAMESPACE_BIAS_EVNT));
    let evnt_col = derive_group_member_collection_id(*b"EVNT", room).unwrap();
    assert_eq!(evnt_col, evnt_full[..16]);
    assert_eq!(evnt_col, derive_collection_id(Some(*b"EVNT"), room));
}

#[test]
fn collection_metadata_identity_collides_with_detects_mismatches() {
    let meta1 = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: None,
        schema: None,
    };
    let mut meta2 = meta1.clone();
    assert!(!meta1.identity_collides_with(&meta2));

    // Different canonical ID -> collision
    meta2.collection_canonical_id = b"!other:matrix.org".to_vec();
    assert!(meta1.identity_collides_with(&meta2));

    // Different namespace -> collision
    let mut meta3 = meta1.clone();
    meta3.member_namespace = Some(*b"STAT");
    assert!(meta1.identity_collides_with(&meta3));

    // Different role/schema does NOT count as identity collision
    let mut meta4 = meta1.clone();
    meta4.role = Some("event_dag".to_owned());
    assert!(!meta1.identity_collides_with(&meta4));
}

#[test]
fn collection_metadata_round_trips_through_tlv() {
    let meta = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: Some(br#"{"ext":"matrix.room","fmt":1,"room_version":"10"}"#.to_vec()),
        role: Some("event_dag".to_owned()),
        schema: Some("matrix.event.v1".to_owned()),
    };
    assert_eq!(CollectionMetadata::decode(&meta.encode()).unwrap(), meta);
    let id = derive_collection_id(meta.member_namespace, &meta.collection_canonical_id);
    assert!(meta.verify_collection_id(&id));
    assert!(!meta.verify_collection_id(&[0u8; 16]));
}

#[test]
fn frame_id_policy_key_round_trips() {
    let policy = FrameIdPolicy::Key;
    let encoded = encode_frame_id_policy(&policy);
    assert_eq!(decode_frame_id_policy(&encoded).unwrap(), policy);
}

fn sample_metadata() -> CollectionMetadata {
    CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:matrix.org".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Pointer {
                pointer: "/event_id".into(),
            },
            digest_algorithm: DigestAlgorithm::Sha256,
        },
        payload: PayloadPolicy::Source,
        extension: Some(b"ext".to_vec()),
        role: Some("event_dag".to_owned()),
        schema: Some("matrix.event.v1".to_owned()),
    }
}

#[test]
fn collection_metadata_decode_rejects_truncation_and_trailing_bytes() {
    let encoded = sample_metadata().encode();

    // Dropping the final byte leaves the last TLV length prefix promising
    // more bytes than remain.
    assert!(CollectionMetadata::decode(&encoded[..encoded.len() - 1]).is_none());
    // Truncated inside the first length prefix.
    assert!(CollectionMetadata::decode(&encoded[..3]).is_none());
    // A complete record followed by trailing bytes is not accepted.
    let mut trailing = encoded;
    trailing.push(0xAA);
    assert!(CollectionMetadata::decode(&trailing).is_none());
}

#[test]
fn collection_metadata_decode_requires_mandatory_fields() {
    let rule = sample_metadata().record_id_rule;

    // No fields at all.
    assert!(CollectionMetadata::decode(&[]).is_none());

    // Canonical id present, record-id rule absent.
    let mut missing_rule = Vec::new();
    push_tlv(
        &mut missing_rule,
        META_TAG_COLLECTION_CANONICAL_ID,
        b"!room:matrix.org",
    );
    assert!(CollectionMetadata::decode(&missing_rule).is_none());

    // Record-id rule present, canonical id absent.
    let mut missing_id = Vec::new();
    push_tlv(
        &mut missing_id,
        META_TAG_RECORD_ID_RULE,
        &encode_record_id_rule(&rule),
    );
    assert!(CollectionMetadata::decode(&missing_id).is_none());

    // An empty canonical id does not count as present.
    let mut empty_id = Vec::new();
    push_tlv(&mut empty_id, META_TAG_COLLECTION_CANONICAL_ID, &[]);
    push_tlv(
        &mut empty_id,
        META_TAG_RECORD_ID_RULE,
        &encode_record_id_rule(&rule),
    );
    assert!(CollectionMetadata::decode(&empty_id).is_none());
}

#[test]
fn record_id_rule_decode_requires_both_fields_and_no_trailing_bytes() {
    let rule = sample_metadata().record_id_rule;
    assert_eq!(
        decode_record_id_rule(&encode_record_id_rule(&rule)).unwrap(),
        rule
    );

    let mut only_digest = Vec::new();
    push_tlv(
        &mut only_digest,
        IDENTITY_TAG_DIGEST_ALGORITHM,
        &[DigestAlgorithm::Sha256.id()],
    );
    assert!(decode_record_id_rule(&only_digest).is_none());

    let mut trailing = encode_record_id_rule(&rule);
    trailing.push(0x00);
    assert!(decode_record_id_rule(&trailing).is_none());
}

#[test]
fn frame_id_policy_decode_rejects_trailing_bytes() {
    let policy = FrameIdPolicy::Pointer {
        pointer: "/event_id".into(),
    };
    let mut encoded = encode_frame_id_policy(&policy);
    assert_eq!(decode_frame_id_policy(&encoded).unwrap(), policy);
    encoded.push(0x7F);
    assert!(decode_frame_id_policy(&encoded).is_none());
}

#[test]
fn payload_decode_rejects_trailing_bytes() {
    assert_eq!(
        decode_payload(&[0x00]).unwrap(),
        PayloadPolicy::Source,
        "a lone source marker is valid"
    );
    assert!(
        decode_payload(&[0x00, 0xAA]).is_none(),
        "trailing bytes after a source marker must be rejected"
    );
    assert_eq!(
        decode_payload(&[0x01]).unwrap(),
        PayloadPolicy::Projection { include: vec![] }
    );
}

#[test]
fn collection_metadata_decode_rejects_invalid_utf8_role_or_schema() {
    let meta = CollectionMetadata {
        member_namespace: Some(*b"EVNT"),
        collection_canonical_id: b"!room:example.com".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("valid_role".to_owned()),
        schema: Some("valid.schema.v1".to_owned()),
    };
    let encoded = meta.encode();
    assert!(CollectionMetadata::decode(&encoded).is_some());

    // Corrupt role TLV (tag 0x06) with invalid UTF-8:
    let mut corrupted_role = Vec::new();
    push_tlv(&mut corrupted_role, 0x01, b"EVNT");
    push_tlv(&mut corrupted_role, 0x02, b"!room:example.com");
    push_tlv(&mut corrupted_role, 0x03, &[0x06, 0x01]); // Key policy, Blake3
    push_tlv(&mut corrupted_role, 0x04, &[0x00]); // Payload source
    push_tlv(&mut corrupted_role, 0x06, &[0xFF, 0xFE, 0xFD]); // Invalid UTF-8 role
    assert!(CollectionMetadata::decode(&corrupted_role).is_none());

    // Corrupt schema TLV (tag 0x07) with invalid UTF-8:
    let mut corrupted_schema = Vec::new();
    push_tlv(&mut corrupted_schema, 0x01, b"EVNT");
    push_tlv(&mut corrupted_schema, 0x02, b"!room:example.com");
    push_tlv(&mut corrupted_schema, 0x03, &[0x06, 0x01]);
    push_tlv(&mut corrupted_schema, 0x04, &[0x00]);
    push_tlv(&mut corrupted_schema, 0x07, &[0xFF, 0xFE, 0xFD]); // Invalid UTF-8 schema
    assert!(CollectionMetadata::decode(&corrupted_schema).is_none());
}

#[test]
fn validate_identity_collision_and_namespace_rejection() {
    let meta1 = CollectionMetadata {
        member_namespace: Some(MEMBER_NAMESPACE_EVNT),
        collection_canonical_id: b"!room:example.com".to_vec(),
        record_id_rule: RecordIdentityRule {
            policy: FrameIdPolicy::Key,
            digest_algorithm: DigestAlgorithm::Blake3,
        },
        payload: PayloadPolicy::Source,
        extension: None,
        role: Some("event_dag".to_owned()),
        schema: None,
    };
    let col1_id = derive_collection_id(meta1.member_namespace, &meta1.collection_canonical_id);

    // 1. Identical metadata -> Ok(())
    let meta1_clone = meta1.clone();
    assert!(meta1
        .validate_identity_collision(&meta1_clone, &col1_id)
        .is_ok());

    // 2. Different canonical id, but requested does not reproduce col1_id -> StorageError::Internal
    let mut meta_different_canonical = meta1.clone();
    meta_different_canonical.collection_canonical_id = b"!room2:example.com".to_vec();
    let err = meta1
        .validate_identity_collision(&meta_different_canonical, &col1_id)
        .unwrap_err();
    assert!(matches!(err, crate::storage::StorageError::Internal(_)));

    // 2b. Stored metadata does not reproduce collection_id -> StorageError::Internal
    let err_stored = meta_different_canonical
        .validate_identity_collision(&meta1, &col1_id)
        .unwrap_err();
    assert!(matches!(
        err_stored,
        crate::storage::StorageError::Internal(_)
    ));

    // 3. Simulated true collision: both stored and requested reproduce the same collection_id
    // despite differing in canonical ID or member namespace -> StorageError::Collision
    let mock_collision_full = [0x55u8; 32];
    let mock_collision_id = [0x55u8; 16];
    let mock_derive = |_ns: Option<[u8; 4]>, _canon: &[u8]| Some(mock_collision_full);

    let err_collision = meta1
        .validate_identity_collision_with(
            &meta_different_canonical,
            &mock_collision_id,
            mock_derive,
        )
        .unwrap_err();
    assert!(matches!(
        err_collision,
        crate::storage::StorageError::Collision(_)
    ));

    // 4. Different member namespace with collision
    let mut meta_stat = meta1.clone();
    meta_stat.member_namespace = Some(MEMBER_NAMESPACE_STAT);
    let err_ns_collision = meta1
        .validate_identity_collision_with(&meta_stat, &mock_collision_id, mock_derive)
        .unwrap_err();
    assert!(matches!(
        err_ns_collision,
        crate::storage::StorageError::Collision(_)
    ));

    // 5. Unregistered namespace (e.g. physical pool tag EDGE)
    let mut meta_edge = meta1.clone();
    meta_edge.member_namespace = Some(*b"EDGE");
    assert!(!meta_edge.verify_collection_id(&col1_id));
    assert_eq!(
        try_derive_collection_full_id(Some(*b"EDGE"), b"!room:example.com"),
        None
    );
    assert_eq!(
        try_derive_collection_id(Some(*b"EDGE"), b"!room:example.com"),
        None
    );
    assert!(CollectionMetadata::decode(&meta_edge.encode()).is_none());
}

#[test]
#[should_panic(expected = "member_namespace must be a valid registered namespace")]
fn derive_collection_id_panics_on_unknown_namespace() {
    let _ = derive_collection_id(Some(*b"EDGE"), b"!room:example.com");
}

#[test]
fn namespace_bias_constants_match_blake3() {
    for (namespace, bias) in [
        (b"EVNT", NAMESPACE_BIAS_EVNT),
        (b"PREV", NAMESPACE_BIAS_PREV),
        (b"AUTH", NAMESPACE_BIAS_AUTH),
        (b"STAT", NAMESPACE_BIAS_STAT),
        (b"STGP", NAMESPACE_BIAS_STGP),
        (b"FWD ", NAMESPACE_BIAS_FWD),
        (b"INTL", NAMESPACE_BIAS_INTL),
    ] {
        let mut input = Vec::from(&b"mtxdb/namespace/v1/"[..]);
        input.extend_from_slice(namespace);
        assert_eq!(
            *blake3::hash(&input).as_bytes(),
            bias,
            "bias for {:?}",
            core::str::from_utf8(namespace)
        );
        assert_eq!(namespace_bias(*namespace), Some(bias));
    }
}
