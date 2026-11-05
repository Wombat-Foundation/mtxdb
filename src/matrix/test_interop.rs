use super::*;

#[test]
fn state_batch_budget_respects_the_configured_stage_limit() {
    let dir =
        std::env::temp_dir().join(format!("mtxdb-interop-stage-limit-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let config = crate::RuntimeConfig {
        txn_stage_limit_bytes: 512,
        ..crate::RuntimeConfig::default()
    };
    let db = crate::Database::open_with_config(dir.clone(), config).unwrap();
    let room = b"!room:example.com";
    let stat = room_state_collection_id(room);
    let payload = [0x11_u8; 100];
    let node = state_hamt_node_id(&[0x22_u8; 32]);

    let error = commit_state_batches(&db, [(stat, node, payload.as_slice())], [], 1024)
        .expect_err("the requested batch exceeds the configured stage limit");
    assert!(
        error.to_string().contains("1..=512"),
        "unexpected error: {error}"
    );
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derive_id_separates_domains_and_is_stable() {
    assert_eq!(derive_id(b"event", b"x"), derive_id(b"event", b"x"));
    assert_ne!(derive_id(b"event", b"x"), derive_id(b"root", b"x"));
    assert_ne!(derive_id(b"event", b"x"), derive_id(b"event", b"y"));
    let event_digest = crate::storage::DigestAlgorithm::Blake3.digest(b"$event:example.com");
    assert_eq!(
        &event_record_id(b"$event:example.com")[..],
        &event_digest[..16]
    );
}

#[test]
fn room_version_record_round_trips_and_rejects_malformed_values() {
    let encoded = encode_room_version_record("!room:example.com", "12").unwrap();
    assert_eq!(
        decode_room_version_record(&encoded).unwrap(),
        ("!room:example.com", "12")
    );
    assert!(decode_room_version_record(&encoded[..7]).is_err());
    assert!(decode_room_version_record(&encoded[..13]).is_err());
    assert!(decode_room_version_record(b"bad!").is_err());
    assert_ne!(
        room_version_collection_id(b"!a:example.com"),
        room_version_collection_id(b"!b:example.com")
    );
}

#[test]
fn state_root_handle_v1_has_the_servers_pinned_layout() {
    let handle = StateRootHandleV1 {
        codec_version: 1,
        routing_version: 2,
        routing_params: [3, 4, 5, 6],
        structural_hash: [7; 32],
        state_group_id: [8; 32],
    };
    let bytes = encode_state_root_handle_v1(handle);
    assert_eq!(bytes.len(), STATE_ROOT_HANDLE_V1_LEN);
    assert_eq!(&bytes[..6], &[1, 2, 3, 4, 5, 6]);
    assert_eq!(&bytes[6..38], &[7; 32]);
    assert_eq!(&bytes[38..70], &[8; 32]);
    assert_eq!(decode_state_root_handle_v1(&bytes).unwrap(), handle);
    assert!(decode_state_root_handle_v1(&bytes[..69]).is_err());
    assert!(decode_state_root_handle_v1(&[&bytes[..], &[0]].concat()).is_err());
}

#[test]
fn state_layout_ids_match_the_server_pinned_layout() {
    let room = b"!room:example.com";
    let event = b"$event:example.com";
    let mut structural_hash = [0_u8; 32];
    structural_hash[..16].fill(0x11);
    structural_hash[16..].fill(0x22);
    assert_eq!(
        hex(&room_state_collection_id(room)),
        "e560c7dd2ceee5d7d31ca9d56e6e9e9f"
    );
    assert_eq!(
        hex(&room_roots_collection_id(room)),
        "7abdfe381bda1791ba9c6ca5e67117ad"
    );
    assert_eq!(state_hamt_node_id(&structural_hash), [0x11; 16]);
    assert_eq!(
        hex(&room_state_root_node_id()),
        "3fa8ef6f8d891872b0e4a05165972dfe"
    );
    assert_eq!(
        hex(&event_state_root_node_id(event)),
        "97ef9503cbbf4df387c4dd8eb3d7ccfa"
    );
}

#[test]
fn verdicts_round_trip_and_reject_truncation() {
    let verdict = Verdict {
        rejection: Some(3),
        soft_fail: Some(7),
        redacted_by: Some(b"$redact:example.com".to_vec()),
        outlier: true,
        arrival_ordering: Some(42),
    };
    let bytes = encode_verdict(b"$event:example.com", &verdict).unwrap();
    let (event_id, decoded) = decode_verdict(&bytes).unwrap();
    assert_eq!(event_id, b"$event:example.com");
    assert_eq!(decoded, verdict);

    for cut in 0..bytes.len() {
        assert!(decode_verdict(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    assert!(decode_verdict(&[]).is_err());
    assert!(decode_verdict(&[9, 0, 0, 0, 0]).is_err());
}

#[test]
fn arrival_ordering_follows_redaction_in_the_wire_format() {
    let arrival = 0x1122_3344_5566_7788_u64;
    let verdict = Verdict {
        redacted_by: Some(b"$r:x".to_vec()),
        arrival_ordering: Some(arrival),
        ..Verdict::default()
    };
    let bytes = encode_verdict(b"$e:x", &verdict).unwrap();
    // version, flags, u32 event-id length, event id, redactor length, redactor,
    // then the arrival ordering. An older decoder stops after `redacted_by`.
    let redactor = 6 + b"$e:x".len();
    assert_eq!(&bytes[redactor..redactor + 4], &4_u32.to_le_bytes());
    assert_eq!(&bytes[redactor + 4..redactor + 8], b"$r:x");
    assert_eq!(&bytes[redactor + 8..], &arrival.to_le_bytes());
    assert_eq!(decode_verdict(&bytes).unwrap().1, verdict);
}

#[test]
fn verdict_collections_bucket_within_the_bound() {
    let mut seen = std::collections::HashSet::new();
    for i in 0_u32..4096 {
        let node = event_record_id(format!("$event{i}:example.com").as_bytes());
        seen.insert(event_verdict_collection_id(&node));
    }
    let buckets = usize::try_from(VERDICT_BUCKETS).unwrap();
    assert!(seen.len() <= buckets, "more collections than buckets");
    assert!(
        seen.len() > buckets / 2,
        "verdict ids should spread across buckets"
    );
}

#[test]
fn server_keys_round_trip_and_reject_truncation() {
    let payload = br#"{"server_name":"a.example","verify_keys":{}}"#;
    let bytes = encode_server_key(b"a.example", payload).unwrap();
    let (origin, decoded) = decode_server_key(&bytes).unwrap();
    assert_eq!(origin, b"a.example");
    assert_eq!(decoded, payload);

    // Incomplete header or a length that overruns the remaining bytes.
    assert!(decode_server_key(&[]).is_err());
    for cut in 1..5 {
        assert!(
            decode_server_key(&bytes[..cut]).is_err(),
            "header cut at {cut}"
        );
    }
    let mid_origin = 5_usize.saturating_add("a.exam".len());
    assert!(
        decode_server_key(&bytes[..mid_origin]).is_err(),
        "origin cut mid-string"
    );

    // A cut that leaves a complete origin but drops the payload is still a
    // valid (empty-payload) record; only a length that overruns is corrupt.
    let mut oversize = vec![KEY_RECORD_VERSION];
    oversize.extend_from_slice(&u32::MAX.to_le_bytes());
    oversize.extend_from_slice(b"a.example");
    assert!(
        decode_server_key(&oversize).is_err(),
        "length overruns the buffer"
    );
}

#[test]
fn notary_record_ids_are_distinct_per_origin() {
    let one = notary_record_id(b"a.example");
    assert_ne!(one, notary_record_id(b"b.example"));
    assert_eq!(one, notary_record_id(b"a.example"));
    // Pin the derivation so changing the domain or hash silently rewrites
    // every existing Notary slot id is caught here.
    assert_eq!(
        one,
        [
            0xE2, 0x68, 0xE8, 0x1F, 0xF2, 0x0A, 0xFA, 0x72, 0xE0, 0x04, 0x45, 0xB0, 0x00, 0xCE,
            0x60, 0xA0,
        ]
    );
}

#[test]
fn import_marker_round_trips_and_rejects_truncation() {
    let marker = ImportMarker {
        data_format: DATA_FORMAT,
        events: 42,
        closures: 40,
    };
    let bytes = encode_import_marker(marker);
    assert_eq!(decode_import_marker(&bytes).unwrap(), marker);

    assert!(decode_import_marker(&[]).is_err());
    assert!(decode_import_marker(&[MARKER_VERSION, DATA_FORMAT, 0, 0]).is_err());
    // Events present, closures count cut short.
    assert!(decode_import_marker(&bytes[..bytes.len() - 1]).is_err());
    assert!(decode_import_marker(&[9, 1, 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
    // A well-formed marker from an older data contract is refused, not misread.
    let mut legacy = vec![MARKER_VERSION, 1];
    legacy.extend_from_slice(&42u64.to_le_bytes());
    legacy.extend_from_slice(&40u64.to_le_bytes());
    assert!(decode_import_marker(&legacy).is_err());
}

#[test]
fn marker_collections_are_stable() {
    assert_eq!(import_marker_collection_id(), import_marker_collection_id());
    assert_ne!(import_marker_collection_id(), server_keys_collection_id());
    assert_ne!(import_marker_node_id(), event_record_id(b"marker"));
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// Every id below is read by a server adapter; the golden values pin them so
/// a refactor cannot silently orphan stored records. The server's own tests
/// check the same values against its key derivations.
#[test]
fn event_collection_ids_are_pinned() {
    let room = b"!room:example.com";
    let node = event_record_id(b"$event:example.com");
    assert_eq!(hex(&node), "d3930850aae2d72b92a1d07108dcaa7f");
    assert_eq!(
        hex(&room_event_collection_id(room)),
        "9507be45c492c4382edb18fc074477cc"
    );
    assert_eq!(
        hex(&event_locator_collection_id(&node)),
        "8630e25f0c7e1a94558485e7176a1bda"
    );
    assert_ne!(
        room_event_collection_id(room),
        room_event_collection_id(b"!other:example.com")
    );
}

#[test]
fn locator_encoding_is_pinned_and_round_trips() {
    let bytes = encode_event_locator(b"$e:x", b"!r:x").unwrap();
    assert_eq!(bytes, [&[1, 4, 0, 0, 0][..], b"$e:x", b"!r:x"].concat());
    assert_eq!(
        decode_event_locator(&bytes).unwrap(),
        (&b"$e:x"[..], &b"!r:x"[..])
    );
    for cut in 0..5 {
        assert!(decode_event_locator(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    assert!(decode_event_locator(&[1, 255, 0, 0, 0, 1]).is_err());
    assert!(decode_event_locator(&[9, 0, 0, 0, 0]).is_err());
    // The unversioned legacy value, a bare room id, is refused.
    assert!(decode_event_locator(b"!room:example.com").is_err());
}

#[test]
fn outlier_flag_is_part_of_the_verdict_record() {
    let outlier = Verdict {
        outlier: true,
        ..Verdict::default()
    };
    assert!(!outlier.is_empty(), "an outlier needs a record");
    let bytes = encode_verdict(b"$e:x", &outlier).unwrap();
    // version, flags (outlier = bit 3), u32 length, event id
    assert_eq!(bytes, [&[1, 0b1000, 4, 0, 0, 0][..], b"$e:x"].concat());
    assert_eq!(decode_verdict(&bytes).unwrap().1, outlier);
    let timeline = Verdict::default();
    assert!(timeline.is_empty());
    assert!(
        !decode_verdict(&encode_verdict(b"$e:x", &timeline).unwrap())
            .unwrap()
            .1
            .outlier
    );
}

#[test]
fn state_batches_split_under_the_budget_and_read_back() {
    let dir = std::env::temp_dir().join(format!("mtxdb-interop-state-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = crate::Database::open(dir.clone()).unwrap();
    let room = b"!room:example.com";
    let (stat, roots) = (
        room_state_collection_id(room),
        room_roots_collection_id(room),
    );

    let payloads: Vec<Vec<u8>> = (0..10_u8).map(|i| vec![i; 100]).collect();
    let nodes: Vec<_> = payloads
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let mut hash = [0_u8; 32];
            hash[0] = u8::try_from(i).unwrap();
            (stat, state_hamt_node_id(&hash), bytes.as_slice())
        })
        .collect();
    let root_bytes = [0xAA_u8; 50];
    let event_root = (
        roots,
        event_state_root_node_id(b"$e:example.com"),
        root_bytes.as_slice(),
    );
    let room_root = (roots, room_state_root_node_id(), root_bytes.as_slice());

    // 10 nodes charge 164 bytes each: 250-byte batches hold one, so 10 batches, then 1 for roots.
    let committed =
        commit_state_batches(&db, nodes.iter().copied(), [event_root, room_root], 250).unwrap();
    assert_eq!(committed, 11);

    for (collection, node, bytes) in nodes.iter().copied().chain([event_root, room_root]) {
        let got = db
            .begin_transaction()
            .get(STATE_POOL, &collection, &[node])
            .unwrap();
        assert_eq!(got[0].as_ref().unwrap().bytes.as_ref(), bytes);
    }
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
