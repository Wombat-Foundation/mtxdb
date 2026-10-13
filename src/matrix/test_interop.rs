use super::*;

#[test]
fn derive_id_separates_domains_and_is_stable() {
    assert_eq!(derive_id(b"event", b"x"), derive_id(b"event", b"x"));
    assert_ne!(derive_id(b"event", b"x"), derive_id(b"root", b"x"));
    assert_ne!(derive_id(b"event", b"x"), derive_id(b"event", b"y"));
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
        "0463880c2405d3407f4c3e12f7858b65"
    );
    assert_eq!(
        hex(&event_state_root_node_id(event)),
        "161f86168f7cce258d2bdb7c6eade04f"
    );
}

#[test]
fn verdicts_round_trip_and_reject_truncation() {
    let verdict = Verdict {
        rejection: Some(3),
        soft_fail: Some(7),
        redacted_by: Some(b"$redact:example.com".to_vec()),
        outlier: true,
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
fn server_key_record_ids_are_distinct_per_kind_and_origin() {
    let historical = server_key_record_id(KeyRecordKind::Historical, b"a.example");
    let provisional = server_key_record_id(KeyRecordKind::Provisional, b"a.example");
    let other = server_key_record_id(KeyRecordKind::Historical, b"b.example");
    assert_ne!(historical, provisional);
    assert_ne!(historical, other);
    assert_eq!(
        historical,
        server_key_record_id(KeyRecordKind::Historical, b"a.example")
    );
}

#[test]
fn key_record_kinds_parse_their_tags() {
    for kind in KeyRecordKind::ALL {
        assert_eq!(KeyRecordKind::parse(kind.as_str()), Some(kind));
    }
    assert_eq!(KeyRecordKind::parse("wat"), None);
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
    assert!(decode_import_marker(&[MARKER_VERSION, 1, 0, 0]).is_err());
    // Events present, closures count cut short.
    assert!(decode_import_marker(&bytes[..bytes.len() - 1]).is_err());
    assert!(decode_import_marker(&[9, 1, 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
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
    assert_eq!(hex(&node), "f4f315ab8b948ccb0355c85e9d609ccd");
    assert_eq!(
        hex(&room_event_collection_id(room)),
        "9507be45c492c4382edb18fc074477cc"
    );
    assert_eq!(
        hex(&room_prev_collection_id(room)),
        "fea9a58f0ab869317d696093fd7c6173"
    );
    assert_eq!(
        hex(&room_auth_collection_id(room)),
        "4842a2eff3cf511f9e48406350253c77"
    );
    assert_eq!(
        hex(&event_locator_collection_id(&node)),
        "9d34360b21fa7f1ad961f46fb7b25857"
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
fn edge_encoding_is_pinned_and_round_trips() {
    let bytes = encode_edges(&[b"$a", b"$bc"]).unwrap();
    assert_eq!(
        bytes,
        [&[2, 0, 0, 0][..], b"$a", &[3, 0, 0, 0], b"$bc"].concat()
    );
    assert_eq!(
        decode_edges(&bytes).unwrap(),
        vec![b"$a".to_vec(), b"$bc".to_vec()]
    );
    assert_eq!(encode_edges(&[]).unwrap(), Vec::<u8>::new());
    assert_eq!(decode_edges(&[]).unwrap(), Vec::<Vec<u8>>::new());
    for cut in [1, 3, 5, 8] {
        assert!(decode_edges(&bytes[..cut]).is_err(), "cut at {cut}");
    }
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
fn auth_chain_records_are_pinned_and_round_trip() {
    assert_eq!(
        hex(&auth_counter_record_id()),
        "107eed6ef642cc6e219edd8a43cafaad"
    );
    assert_eq!(
        hex(&auth_forward_record_id(b"$e:x")),
        "16a53f37238f9d0c2adbba80aa84661f"
    );
    assert_eq!(
        hex(&auth_reverse_record_id(7)),
        "a80ce9b45f60f10d32afb99f79d8cafe"
    );
    assert_eq!(
        hex(&auth_closure_record_id(7)),
        "4d225a7b56f203a482578ff4b59b82b6"
    );
    assert_ne!(auth_reverse_record_id(7), auth_closure_record_id(7));
    assert_ne!(auth_reverse_record_id(7), auth_reverse_record_id(8));

    let forward = encode_auth_forward(0x0102_0304, b"$e:x");
    assert_eq!(forward, [&[1, 2, 3, 4][..], b"$e:x"].concat());
    assert_eq!(
        decode_auth_forward(&forward).unwrap(),
        (0x0102_0304, &b"$e:x"[..])
    );
    for cut in 0..4 {
        assert!(
            decode_auth_forward(&forward[..cut]).is_err(),
            "cut at {cut}"
        );
    }
    assert_eq!(decode_auth_counter(&9_u32.to_be_bytes()).unwrap(), 9);
    assert!(decode_auth_counter(&[0, 0, 9]).is_err());
    assert!(decode_auth_counter(&[0, 0, 0, 0, 9]).is_err());
}
