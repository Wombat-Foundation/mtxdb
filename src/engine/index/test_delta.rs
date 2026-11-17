use super::*;

fn test_frame(bucket: u32, generation: u64) -> DeltaFrame {
    DeltaFrame {
        collection_id: [0x44, 0x4C, 0x54, 0x52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bucket,
        generation,
        slot: u64::from(bucket).wrapping_mul(7).wrapping_add(1),
    }
}

fn v1_batch_bytes(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
    let mut frames = Vec::new();
    for operation in operations {
        frames.extend_from_slice(&encode_frame(operation).unwrap());
    }
    let crc = batch_crc(&frames, tail_fingerprint);
    let mut batch = Vec::from(encode_batch_header(
        u32::try_from(operations.len()).unwrap(),
    ));
    batch.extend_from_slice(&frames);
    batch.extend_from_slice(&encode_trailer(tail_fingerprint, crc));
    batch
}

fn sample_v1_operations() -> Vec<DeltaOperation> {
    vec![
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: [7; 16],
            bucket: 1,
            generation: 4,
            slot: 99,
        }),
        DeltaOperation::CollectionSnapshot {
            collection_id: [8; 16],
            generation: 5,
            order_key: 1,
            index_blob: vec![1, 2, 3, 4],
        },
        DeltaOperation::CollectionTombstone {
            collection_id: [9; 16],
            generation: 6,
        },
    ]
}

fn v1_log(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
    let mut log = Vec::from(encode_header(0xABC));
    log[4] = DELTA_LOG_VERSION;
    log.extend_from_slice(&v1_batch_bytes(operations, tail_fingerprint));
    log
}

fn redo_set(collection_id: [u8; 16], full_hash: [u8; 16], delta_seq: u64) -> DeltaOperation {
    DeltaOperation::Redo(RedoRecord {
        collection_id,
        op: RedoOp::Set {
            full_hash,
            pack_id: crate::packfile::PackId([1; 16]),
            offset: 4096,
            record_len: 128,
            removed: false,
        },
        delta_seq,
        base_generation: 1,
    })
}

/// An empty version table for a coverage claim at `covered_lsn`: the shape the
/// writer produces is the table, then the claim.
fn table(covered_lsn: u64) -> DeltaOperation {
    DeltaOperation::Versions {
        covered_lsn,
        entries: Vec::new(),
    }
}

fn replay_incremental_snapshot_model(
    operations: &[DeltaOperation],
) -> HashMap<[u8; 16], Vec<DeltaOperation>> {
    let mut state = HashMap::<[u8; 16], Vec<DeltaOperation>>::new();
    for operation in operations {
        match operation {
            DeltaOperation::Incremental(frame) => {
                let entries = state.entry(frame.collection_id).or_default();
                entries.retain(|existing| {
                    !matches!(existing, DeltaOperation::Incremental(old) if old.bucket == frame.bucket)
                });
                entries.push(operation.clone());
            }
            DeltaOperation::CollectionSnapshot { collection_id, .. } => {
                state.insert(*collection_id, vec![operation.clone()]);
            }
            DeltaOperation::CollectionTombstone { collection_id, .. } => {
                state.remove(collection_id);
            }
            DeltaOperation::Coverage { .. }
            | DeltaOperation::Versions { .. }
            | DeltaOperation::Redo(_) => {}
        }
    }
    state
}

#[test]
fn compaction_keeps_the_newest_redo_and_the_final_coverage() {
    let collection = [3; 16];
    let redo_a = redo_set(collection, [1; 16], 1);
    let redo_b = redo_set(collection, [2; 16], 2);
    let redo_a_newer = redo_set(collection, [1; 16], 3);
    let versions = table(42);
    let coverage = DeltaOperation::Coverage { covered_lsn: 42 };
    let operations = vec![
        redo_a,
        redo_b,
        redo_a_newer,
        versions.clone(),
        coverage.clone(),
    ];

    let compacted = compact_operations(&operations);
    assert_eq!(compacted.len(), 4);
    assert_eq!(compacted[2], versions);
    assert_eq!(compacted[3], coverage);
    match &compacted[0] {
        DeltaOperation::Redo(record) => {
            assert_eq!(record.delta_seq, 2, "the untouched hash keeps its set");
            let RedoOp::Set { full_hash, .. } = record.op else {
                panic!("expected a set");
            };
            assert_eq!(full_hash, [2; 16]);
        }
        other => panic!("expected redo, got {other:?}"),
    }
    match &compacted[1] {
        DeltaOperation::Redo(record) => {
            assert_eq!(
                record.delta_seq, 3,
                "only the newest set for a hash survives"
            );
            let RedoOp::Set { full_hash, .. } = record.op else {
                panic!("expected a set");
            };
            assert_eq!(full_hash, [1; 16]);
        }
        other => panic!("expected redo, got {other:?}"),
    }
}

#[test]
fn compaction_drops_everything_before_a_collection_cut() {
    let collection = [4; 16];
    let operations = vec![
        redo_set(collection, [1; 16], 1),
        DeltaOperation::CollectionTombstone {
            collection_id: collection,
            generation: 2,
        },
        redo_set(collection, [2; 16], 2),
        table(9),
        DeltaOperation::Coverage { covered_lsn: 9 },
    ];

    let compacted = compact_operations(&operations);
    assert_eq!(compacted.len(), 4);
    assert!(matches!(
        compacted[0],
        DeltaOperation::CollectionTombstone { .. }
    ));
    assert!(matches!(compacted[1], DeltaOperation::Redo(_)));
    assert!(matches!(compacted[2], DeltaOperation::Versions { .. }));
    assert!(matches!(compacted[3], DeltaOperation::Coverage { .. }));
}

#[test]
fn compact_delta_log_rewrites_in_place_and_preserves_the_claim() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_compact_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let collection = [5; 16];
    let operations = vec![
        redo_set(collection, [1; 16], 1),
        redo_set(collection, [2; 16], 2),
        redo_set(collection, [1; 16], 3),
        table(77),
        DeltaOperation::Coverage { covered_lsn: 77 },
    ];
    std::fs::write(&path, v1_log(&operations, 0x5151)).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let after = compact_delta_log(&path)
        .expect("compaction succeeds")
        .expect("operations were redundant");
    assert!(after < before, "compaction must shorten the log");

    let decoded = read_delta_log(&path).expect("compacted log reads");
    assert_eq!(decoded.base_fingerprint, 0xABC);
    assert_eq!(decoded.tail_fingerprint, 0x5151);
    assert_eq!(decoded.coverage, Some(77));
    assert_eq!(
        decoded.operations.len(),
        4,
        "two redos, the table, the claim"
    );
    assert!(matches!(
        decoded.operations.last(),
        Some(DeltaOperation::Coverage { covered_lsn: 77 })
    ));
    assert_eq!(
        decoded
            .version_table
            .as_ref()
            .map(|table| table.covered_lsn),
        Some(77)
    );

    // A second pass has nothing left to remove.
    assert_eq!(compact_delta_log(&path).unwrap(), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_delta_log_skips_a_log_with_marginal_savings() {
    let dir = std::env::temp_dir().join(format!(
        "mtxdb_delta_compact_marginal_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let collection = [6; 16];
    let mut operations = Vec::new();
    for index in 0..8u8 {
        operations.push(redo_set(collection, [index; 16], u64::from(index)));
    }
    operations.push(redo_set(collection, [0; 16], 99));
    operations.push(table(100));
    operations.push(DeltaOperation::Coverage { covered_lsn: 100 });
    std::fs::write(&path, v1_log(&operations, 0x6161)).unwrap();
    let before = std::fs::read(&path).unwrap();

    assert_eq!(
        compact_delta_log(&path).unwrap(),
        None,
        "a rewrite that frees less than the minimum margin is not published"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "the log is unchanged"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_delta_log_reaps_a_crash_leftover_temp() {
    let dir =
        std::env::temp_dir().join(format!("mtxdb_delta_compact_stale_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    let stale = dir.join(format!(".{INDEX_DELTA_FILE}.compact"));
    let collection = [0x61; 16];
    let operations = vec![
        redo_set(collection, [1; 16], 1),
        redo_set(collection, [2; 16], 2),
        redo_set(collection, [1; 16], 3),
        table(77),
        DeltaOperation::Coverage { covered_lsn: 77 },
    ];
    std::fs::write(&path, v1_log(&operations, 0x7171)).unwrap();
    std::fs::write(&stale, b"leftover").unwrap();

    compact_delta_log(&path)
        .expect("compaction succeeds")
        .expect("operations were redundant");

    let compacted = read_delta_log(&path).expect("live log remains readable");
    assert_eq!(compacted.coverage, Some(77));
    assert_eq!(compacted.operations.len(), 4);
    assert!(!stale.exists(), "a crash leftover is removed before reuse");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compact_operations_preserve_incremental_and_snapshot_replay_result() {
    let collection_a = [0x71; 16];
    let collection_b = [0x72; 16];
    let operations = vec![
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: collection_a,
            bucket: 1,
            generation: 1,
            slot: 11,
        }),
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: collection_b,
            bucket: 9,
            generation: 1,
            slot: 19,
        }),
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: collection_b,
            bucket: 9,
            generation: 1,
            slot: 29,
        }),
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: collection_a,
            bucket: 1,
            generation: 1,
            slot: 12,
        }),
        DeltaOperation::CollectionSnapshot {
            collection_id: collection_a,
            generation: 3,
            order_key: 4,
            index_blob: vec![3, 3, 3],
        },
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: collection_a,
            bucket: 7,
            generation: 3,
            slot: 17,
        }),
        DeltaOperation::Coverage { covered_lsn: 88 },
    ];

    let compacted = compact_operations(&operations);
    assert_eq!(
        replay_incremental_snapshot_model(&operations),
        replay_incremental_snapshot_model(&compacted)
    );
    assert_eq!(
        compacted
            .iter()
            .filter(|operation| matches!(operation, DeltaOperation::Incremental(frame) if frame.collection_id == collection_b && frame.bucket == 9))
            .count(),
        1,
        "only the latest incremental frame for a bucket survives"
    );
    assert!(
        compacted.iter().any(|operation| matches!(
            operation,
            DeltaOperation::CollectionSnapshot { collection_id, generation: 3, .. }
                if *collection_id == collection_a
        )),
        "the latest snapshot must survive the cut"
    );
    assert!(
        !compacted.iter().any(|operation| matches!(
            operation,
            DeltaOperation::Incremental(frame)
                if frame.collection_id == collection_a && frame.generation == 1
        )),
        "the snapshot must cut earlier incrementals"
    );
}

#[test]
fn v1_log_round_trips_operations() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v1_round_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v1_operations();
    let mut log = v1_log(&operations[..2], 0x111);
    log.extend_from_slice(&v1_batch_bytes(&operations[2..], 0x222));
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log(&path).expect("valid v1 log reads");
    assert_eq!(decoded.base_fingerprint, 0xABC);
    assert_eq!(decoded.tail_fingerprint, 0x222);
    assert_eq!(decoded.operations, operations);
    assert!(!decoded.torn_tail);
    assert_eq!(decoded.file_len, u64::try_from(log.len()).unwrap());
    let tail = read_delta_tail_fingerprint(&path)
        .unwrap()
        .expect("v1 tail scan succeeds");
    assert_eq!(tail.base_fingerprint, decoded.base_fingerprint);
    assert_eq!(tail.tail_fingerprint, decoded.tail_fingerprint);
    assert_eq!(tail.torn_tail, decoded.torn_tail);
    // An unsupported version must not be accepted as a v1 epoch.
    let mut unsupported = encode_header(1);
    unsupported[4] = DELTA_LOG_VERSION.wrapping_add(1);
    std::fs::write(&path, unsupported).unwrap();
    assert!(read_delta_log(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v1_torn_tail_keeps_the_committed_prefix() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v1_torn_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v1_operations();
    let mut log = v1_log(&operations[..1], 0x111);
    let committed_len = u64::try_from(log.len()).unwrap();
    // A second batch with a header and one frame but no trailer.
    log.extend_from_slice(&encode_batch_header(2));
    log.extend_from_slice(&encode_frame(&operations[1]).unwrap());
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log(&path).expect("committed prefix is trusted");
    assert_eq!(decoded.operations, vec![operations[0].clone()]);
    assert_eq!(decoded.tail_fingerprint, 0x111);
    assert_eq!(decoded.file_len, committed_len);
    assert!(decoded.torn_tail);
    let tail = read_delta_tail_fingerprint(&path)
        .unwrap()
        .expect("tail scan keeps the committed prefix");
    assert_eq!(tail.tail_fingerprint, decoded.tail_fingerprint);
    assert!(tail.torn_tail);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v1_corrupt_batch_drops_that_batch_and_later() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v1_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v1_operations();
    let mut log = v1_log(&operations[..1], 0x111);
    let first_batch_end = log.len();
    log.extend_from_slice(&v1_batch_bytes(&operations[1..], 0x222));
    // Corrupt one byte inside the second batch's frame region.
    let corrupt_at = first_batch_end
        .saturating_add(DELTA_BATCH_HEADER_LEN)
        .saturating_add(4);
    log[corrupt_at] ^= 0x20;
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log(&path).expect("first batch is trusted");
    assert_eq!(decoded.operations, vec![operations[0].clone()]);
    assert_eq!(decoded.tail_fingerprint, 0x111);
    assert!(decoded.torn_tail);
    assert_eq!(decoded.file_len, u64::try_from(first_batch_end).unwrap());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v1_reader_rejects_unsupported_and_empty_logs() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v1_reject_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    // An unsupported header version is not a v1 log.
    let mut unsupported = encode_header(1);
    unsupported[4] = DELTA_LOG_VERSION.wrapping_add(1);
    std::fs::write(&path, unsupported).unwrap();
    assert!(read_delta_log(&path).is_none());

    // A v1 header with no committed batch is not replayable.
    let mut header_only = Vec::from(encode_header(1));
    header_only[4] = DELTA_LOG_VERSION;
    std::fs::write(&path, header_only).unwrap();
    assert!(read_delta_log(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v1_frames_round_trip_every_variant() {
    // Incremental, CollectionSnapshot, and CollectionTombstone: each must
    // survive encode -> decode byte-for-byte.
    for operation in sample_v1_operations() {
        let frame = encode_frame(&operation).expect("frame encodes");
        let (decoded, consumed) = decode_frame(&frame).expect("frame decodes");
        assert_eq!(decoded, operation);
        assert_eq!(consumed, frame.len());
    }
}

#[test]
fn v1_frame_decode_consumes_exactly_one_frame() {
    let operations = sample_v1_operations();
    let first = encode_frame(&operations[0]).unwrap();
    let second = encode_frame(&operations[1]).unwrap();
    let mut concatenated = first.clone();
    concatenated.extend_from_slice(&second);

    let (decoded, consumed) = decode_frame(&concatenated).expect("first frame decodes");
    assert_eq!(decoded, operations[0]);
    assert_eq!(consumed, first.len());

    let (decoded_second, second_consumed) =
        decode_frame(&concatenated[consumed..]).expect("second frame decodes");
    assert_eq!(decoded_second, operations[1]);
    assert_eq!(second_consumed, second.len());
}

#[test]
fn v1_frame_decode_rejects_corruption_and_truncation() {
    let operation = DeltaOperation::CollectionSnapshot {
        collection_id: [5; 16],
        generation: 2,
        order_key: 1,
        index_blob: vec![9, 8, 7, 6],
    };
    let frame = encode_frame(&operation).unwrap();

    // Every truncation short of the complete frame must fail, not panic.
    for shorter in [0, 1, V1_FRAME_HEADER_LEN, frame.len().saturating_sub(1)] {
        assert!(decode_frame(&frame[..shorter]).is_none());
    }

    // A flipped payload byte must be caught by the frame CRC.
    let mut corrupt_payload = frame.clone();
    corrupt_payload[V1_FRAME_HEADER_LEN] ^= 0x40;
    assert!(decode_frame(&corrupt_payload).is_none());

    // ...as must a flipped CRC byte.
    let mut corrupt_crc = frame.clone();
    let crc_byte = corrupt_crc.len().saturating_sub(1);
    corrupt_crc[crc_byte] ^= 0x01;
    assert!(decode_frame(&corrupt_crc).is_none());

    // A length field reaching past the buffer must fail, not allocate.
    let mut bad_len = frame.clone();
    bad_len[1..5].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_frame(&bad_len).is_none());

    // A plausible, shorter length has enough bytes for framing, so this
    // specifically proves the CRC covers the length field. The decoder
    // must reject it at checksum validation rather than only at bounds
    // checking.
    let mut bad_len = frame.clone();
    let payload_len = u32::from_le_bytes(frame[1..5].try_into().unwrap());
    bad_len[1..5].copy_from_slice(&payload_len.checked_sub(1).unwrap().to_le_bytes());
    assert!(decode_frame(&bad_len).is_none());

    // An unknown operation kind with a recomputed valid CRC is rejected.
    let mut unknown = frame.clone();
    unknown[0] = 0x7F;
    let crc_len = unknown.len().saturating_sub(V1_FRAME_TRAILER_LEN);
    let crc = crc32fast::hash(&unknown[..crc_len]).to_le_bytes();
    unknown[crc_len..].copy_from_slice(&crc);
    assert!(decode_frame(&unknown).is_none());
}

#[test]
fn v1_frame_encoded_sizes_are_exact() {
    let incremental = DeltaOperation::Incremental(test_frame(3, 5));
    assert_eq!(
        encode_frame(&incremental).unwrap().len(),
        V1_FRAME_HEADER_LEN + 36 + V1_FRAME_TRAILER_LEN,
        "incremental framing is header + fixed payload + crc"
    );

    let snapshot = DeltaOperation::CollectionSnapshot {
        collection_id: [0x11; 16],
        generation: 7,
        order_key: 42,
        index_blob: vec![0xAB; 128],
    };
    assert_eq!(
        encode_frame(&snapshot).unwrap().len(),
        V1_FRAME_HEADER_LEN + V1_SNAPSHOT_FIXED_LEN + 128 + V1_FRAME_TRAILER_LEN,
        "snapshot framing is header + fixed fields + blob + crc"
    );

    let tombstone = DeltaOperation::CollectionTombstone {
        collection_id: [0x22; 16],
        generation: 9,
    };
    assert_eq!(
        encode_frame(&tombstone).unwrap().len(),
        V1_FRAME_HEADER_LEN + V1_TOMBSTONE_LEN + V1_FRAME_TRAILER_LEN,
        "tombstone framing is header + fixed fields + crc"
    );
}

#[test]
fn v1_snapshot_rejects_blob_length_mismatch() {
    // Valid framing and CRC whose advertised index_blob length runs past
    // the payload. The existing corruption test covers a bad *framing*
    // length; this covers the separate blob length field inside a snapshot.
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0x11; 16]);
    payload.extend_from_slice(&7_u64.to_le_bytes());
    payload.extend_from_slice(&42_u64.to_le_bytes());
    payload.extend_from_slice(&128_u32.to_le_bytes());
    payload.extend_from_slice(&[0u8; 8]);
    let mut frame = Vec::new();
    frame.push(V1_COLLECTION_SNAPSHOT);
    frame.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    assert!(
        decode_frame(&frame).is_none(),
        "declared blob length must match the payload exactly"
    );
}

#[test]
fn v1_append_batch_round_trips_through_the_reader() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v1_append_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let first = sample_v1_operations();
    let mut expected_len = DELTA_LOG_HEADER_LEN
        .saturating_add(DELTA_BATCH_HEADER_LEN)
        .saturating_add(DELTA_LOG_TRAILER_LEN);
    for operation in &first {
        expected_len = expected_len.saturating_add(encode_frame(operation).unwrap().len());
    }
    let appended = append_batch(&path, true, 0xABC, &first, 0x111).expect("append v1 batch");
    assert_eq!(appended, expected_len, "append reports the bytes it wrote");

    let second = [DeltaOperation::CollectionTombstone {
        collection_id: [3; 16],
        generation: 4,
    }];
    append_batch(&path, false, 0xABC, &second, 0x222).expect("continue the v1 log");

    let decoded = read_delta_log(&path).expect("appended v1 log reads");
    assert_eq!(decoded.base_fingerprint, 0xABC);
    assert_eq!(decoded.tail_fingerprint, 0x222);
    let mut expected = first;
    expected.extend_from_slice(&second);
    assert_eq!(decoded.operations, expected);
    assert!(!decoded.torn_tail);
    assert_eq!(decoded.file_len, fs::metadata(&path).unwrap().len());

    std::fs::remove_dir_all(&dir).unwrap();
}
fn coverage_log_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_cov_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Coverage is an ordinary v1 frame: it round-trips, is sized exactly, and
/// its payload must be exactly one LSN.
#[test]
fn v1_coverage_frame_round_trips() {
    let coverage = DeltaOperation::Coverage {
        covered_lsn: 0xDEAD_BEEF,
    };
    let encoded = encode_frame(&coverage).unwrap();
    assert_eq!(
        encoded.len(),
        V1_FRAME_HEADER_LEN + V1_COVERAGE_LEN + V1_FRAME_TRAILER_LEN
    );
    assert_eq!(
        decode_frame(&encoded),
        Some((coverage.clone(), encoded.len()))
    );
    assert_eq!(
        v1_batch_len(&[coverage]).unwrap(),
        DELTA_BATCH_HEADER_LEN + encoded.len() + DELTA_LOG_TRAILER_LEN
    );
    // A coverage frame with the wrong payload width is rejected.
    let mut frame = vec![V1_COVERAGE];
    frame.extend_from_slice(&4_u32.to_le_bytes());
    frame.extend_from_slice(&[1, 2, 3, 4]);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    assert!(decode_frame(&frame).is_none());
}

/// Both readers report the newest coverage a committed batch claims, and
/// the full reader also says which operations that coverage describes.
#[test]
fn readers_report_the_newest_committed_coverage() {
    let dir = coverage_log_dir("readers");
    let path = dir.join(INDEX_DELTA_FILE);
    let incremental = |bucket: u32| {
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: [1; 16],
            bucket,
            generation: 1,
            slot: 5,
        })
    };
    // batch 1: two ops, no coverage
    append_batch(&path, true, 0xABC, &[incremental(1), incremental(2)], 0x111).unwrap();
    let log = read_delta_log(&path).unwrap();
    assert_eq!(log.coverage, None);
    assert_eq!(log.coverage_prefix_ops, 0);
    assert_eq!(
        read_delta_tail_fingerprint(&path)
            .unwrap()
            .unwrap()
            .coverage,
        None
    );
    // batch 2: an op, then coverage
    append_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental(3),
            table(40),
            DeltaOperation::Coverage { covered_lsn: 40 },
        ],
        0x222,
        true,
    )
    .unwrap();
    // batch 3: more ops with no coverage: a later claim is not implied
    append_batch(&path, false, 0xABC, &[incremental(4)], 0x333).unwrap();
    let log = read_delta_log(&path).unwrap();
    assert_eq!(log.coverage, Some(40));
    assert_eq!(
        log.coverage_prefix_ops, 5,
        "two + one + the table + the coverage op itself"
    );
    assert_eq!(log.operations.len(), 6);
    assert_eq!(
        read_delta_tail_fingerprint(&path)
            .unwrap()
            .unwrap()
            .coverage,
        Some(40)
    );
    // batch 4: a lower claim never lowers the coverage
    append_batch(
        &path,
        false,
        0xABC,
        &[table(10), DeltaOperation::Coverage { covered_lsn: 10 }],
        0x444,
    )
    .unwrap();
    let log = read_delta_log(&path).unwrap();
    assert_eq!(log.coverage, Some(40));
    assert_eq!(log.coverage_prefix_ops, 8);
    assert_eq!(
        log.version_table.as_ref().map(|table| table.covered_lsn),
        Some(40),
        "a lower claim's table never replaces the newer one"
    );
    assert_eq!(
        read_delta_tail_fingerprint(&path)
            .unwrap()
            .unwrap()
            .coverage,
        Some(40)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The lengths computed without building a batch equal the built batch's.
#[test]
fn projected_frame_lengths_match_the_built_batch() {
    let blob = vec![7u8; 1000];
    let operations = [
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: [1; 16],
            bucket: 1,
            generation: 1,
            slot: 1,
        }),
        DeltaOperation::CollectionSnapshot {
            collection_id: [2; 16],
            generation: 2,
            order_key: 3,
            index_blob: blob.clone(),
        },
        DeltaOperation::CollectionTombstone {
            collection_id: [3; 16],
            generation: 4,
        },
        table(5),
        DeltaOperation::Coverage { covered_lsn: 5 },
    ];
    let projected = v1_empty_batch_len()
        + v1_incremental_frame_len()
        + v1_snapshot_frame_len(blob.len()).unwrap()
        + v1_tombstone_frame_len()
        + v1_versions_frame_len(0).unwrap()
        + v1_coverage_frame_len();
    assert_eq!(v1_batch_len(&operations), Some(projected));
    let dir = coverage_log_dir("projected_len");
    let path = dir.join(INDEX_DELTA_FILE);
    let written =
        append_batch_with_durability(&path, false, 0xABC, &operations, 0x1, false).unwrap();
    assert_eq!(written, projected);
    fs::remove_dir_all(&dir).unwrap();
}

/// The tail scanner reads through a buffer, so a torn log has to end its scan
/// exactly where the full reader does. Every truncation point of a log with
/// several batches gives the same tail fingerprint and coverage from both.
#[test]
fn the_buffered_tail_scanner_agrees_with_the_full_reader_at_every_cut() {
    let dir = coverage_log_dir("buffered_cuts");
    let path = dir.join(INDEX_DELTA_FILE);
    let incremental = |slot| {
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: [2; 16],
            bucket: 1,
            generation: 1,
            slot,
        })
    };
    append_batch_with_durability(&path, true, 0xABC, &[incremental(1)], 0x111, false).unwrap();
    append_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental(2),
            table(9),
            DeltaOperation::Coverage { covered_lsn: 9 },
        ],
        0x222,
        false,
    )
    .unwrap();
    append_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental(3),
            incremental(4),
            table(20),
            DeltaOperation::Coverage { covered_lsn: 20 },
        ],
        0x333,
        false,
    )
    .unwrap();
    let full = fs::read(&path).unwrap();
    for cut in DELTA_LOG_HEADER_LEN..=full.len() {
        fs::write(&path, &full[..cut]).unwrap();
        let log = read_delta_log(&path);
        let tail = read_delta_tail_fingerprint(&path);
        match (log, tail) {
            (Some(log), Ok(Some(tail))) => {
                assert_eq!(log.tail_fingerprint, tail.tail_fingerprint, "cut {cut}");
                assert_eq!(log.coverage, tail.coverage, "cut {cut}");
                assert_eq!(log.torn_tail, tail.torn_tail, "cut {cut}");
            }
            (None, Err(_) | Ok(None)) => {}
            (log, tail) => panic!(
                "cut {cut}: full reader {:?}, tail scanner {:?}",
                log.map(|log| log.tail_fingerprint),
                tail.map(|tail| tail.map(|tail| tail.tail_fingerprint)).ok()
            ),
        }
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// A batch whose claim is not its last operation, or that holds two, was
/// not written by this writer; it is dropped along with everything after it.
#[test]
fn a_misplaced_or_repeated_coverage_claim_is_rejected() {
    let incremental = DeltaOperation::Incremental(DeltaFrame {
        collection_id: [2; 16],
        bucket: 1,
        generation: 1,
        slot: 5,
    });
    let claim = |covered_lsn| DeltaOperation::Coverage { covered_lsn };
    let bad_batches: [Vec<DeltaOperation>; 7] = [
        // A claim that is not last.
        vec![table(50), claim(50), incremental.clone()],
        // Two claims.
        vec![table(50), claim(50), claim(60)],
        // A claim with no table.
        vec![incremental.clone(), claim(50)],
        // A table with no claim.
        vec![incremental.clone(), table(50)],
        // A table that disagrees with its claim.
        vec![table(40), claim(50)],
        // Two tables.
        vec![table(50), table(50), claim(50)],
        // A table that is not directly before the claim.
        vec![table(50), incremental.clone(), claim(50)],
    ];
    for (index, bad) in bad_batches.iter().enumerate() {
        let dir = coverage_log_dir(&format!("misplaced-{index}"));
        let path = dir.join(INDEX_DELTA_FILE);
        append_batch_with_durability(
            &path,
            true,
            0xABC,
            &[
                incremental.clone(),
                table(7),
                DeltaOperation::Coverage { covered_lsn: 7 },
            ],
            0x111,
            true,
        )
        .unwrap();
        append_batch_with_durability(&path, false, 0xABC, bad, 0x222, true).unwrap();
        let log = read_delta_log(&path).unwrap();
        assert_eq!(log.coverage, Some(7), "case {index}");
        assert_eq!(log.coverage_prefix_ops, 3, "case {index}");
        assert_eq!(log.operations.len(), 3, "case {index}");
        assert_eq!(
            log.version_table.as_ref().map(|table| table.covered_lsn),
            Some(7),
            "case {index}: a damaged batch's table is never exposed"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// The two readers make the same durability decision from the same bytes,
/// including for batches whose claim is misplaced or repeated.
/// Before the shape rule reached the tail scanner, a claim placed first in
/// its batch was honored by it (50 here) and rejected by the full reader (7).
#[test]
fn the_tail_scanner_and_the_full_reader_agree_on_coverage() {
    let incremental = DeltaOperation::Incremental(DeltaFrame {
        collection_id: [2; 16],
        bucket: 1,
        generation: 1,
        slot: 5,
    });
    let claim = |covered_lsn| DeltaOperation::Coverage { covered_lsn };
    let cases: [Vec<DeltaOperation>; 8] = [
        vec![incremental.clone(), table(50), claim(50)],
        vec![incremental.clone(), claim(50)],
        vec![table(50), claim(50), incremental.clone()],
        vec![table(50), claim(50), claim(60)],
        vec![incremental.clone()],
        vec![incremental.clone(), table(50)],
        vec![table(40), claim(50)],
        vec![table(50), incremental.clone(), claim(50)],
    ];
    for (index, batch) in cases.iter().enumerate() {
        let dir = coverage_log_dir(&format!("agree-{index}"));
        let path = dir.join(INDEX_DELTA_FILE);
        append_batch_with_durability(
            &path,
            true,
            0xABC,
            &[incremental.clone(), table(7), claim(7)],
            0x111,
            true,
        )
        .unwrap();
        append_batch_with_durability(&path, false, 0xABC, batch, 0x222, true).unwrap();
        let full = read_delta_log(&path).unwrap();
        let tail = read_delta_tail_fingerprint(&path).unwrap().unwrap();
        assert_eq!(full.coverage, tail.coverage, "case {index}");
        assert_eq!(full.tail_fingerprint, tail.tail_fingerprint, "case {index}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// What a reader that predates the `Coverage` kind does: it cannot decode the
/// claim frame, so it rejects that whole batch and everything after it. The
/// frame's kind byte is rewritten to an unknown one with every checksum
/// recomputed, so only the kind is unfamiliar. It keeps the earlier batches
/// and loses the ops in the batch that carries the claim (the writer then
/// recovers them by replaying more, never by skipping).
#[test]
fn a_reader_that_does_not_know_the_coverage_kind_stops_at_that_batch() {
    let dir = coverage_log_dir("old_reader");
    let path = dir.join(INDEX_DELTA_FILE);
    let incremental = |slot| {
        DeltaOperation::Incremental(DeltaFrame {
            collection_id: [2; 16],
            bucket: 1,
            generation: 1,
            slot,
        })
    };
    append_batch_with_durability(&path, true, 0xABC, &[incremental(1)], 0x111, true).unwrap();
    let first_len = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
    append_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental(2),
            table(9),
            DeltaOperation::Coverage { covered_lsn: 9 },
        ],
        0x222,
        true,
    )
    .unwrap();
    let intact = read_delta_log(&path).unwrap();
    assert_eq!(intact.coverage, Some(9));
    assert_eq!(intact.operations.len(), 4);

    let mut bytes = fs::read(&path).unwrap();
    let frame_len = V1_FRAME_HEADER_LEN + V1_COVERAGE_LEN + V1_FRAME_TRAILER_LEN;
    let trailer_start = bytes.len() - DELTA_LOG_TRAILER_LEN;
    let frame_start = trailer_start - frame_len;
    bytes[frame_start] = 0x7F;
    let frame_crc_at = trailer_start - V1_FRAME_TRAILER_LEN;
    let crc = crc32fast::hash(&bytes[frame_start..frame_crc_at]).to_le_bytes();
    bytes[frame_crc_at..trailer_start].copy_from_slice(&crc);
    let batch_body_start = first_len + DELTA_BATCH_HEADER_LEN;
    let batch_fingerprint = u64::from_le_bytes(
        bytes[trailer_start + TRAILER_FINGERPRINT_OFFSET..trailer_start + 16]
            .try_into()
            .unwrap(),
    );
    let batch = batch_crc(&bytes[batch_body_start..trailer_start], batch_fingerprint);
    bytes[trailer_start + TRAILER_CRC_OFFSET..trailer_start + 8]
        .copy_from_slice(&batch.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    let old = read_delta_log(&path).unwrap();
    assert_eq!(old.coverage, None, "an unknown frame claims nothing");
    assert_eq!(old.operations.len(), 1, "only the first batch survives");
    assert!(old.torn_tail, "the rest is treated as an unreadable tail");
    fs::remove_dir_all(&dir).unwrap();
}

/// A torn batch, or one whose bytes were damaged, claims nothing: coverage
/// is whatever the last intact batch proved. This is the conservative
/// outcome, since a lower coverage only makes recovery replay more.
#[test]
fn a_torn_or_damaged_coverage_batch_claims_nothing() {
    let dir = coverage_log_dir("torn");
    let path = dir.join(INDEX_DELTA_FILE);
    let incremental = DeltaOperation::Incremental(DeltaFrame {
        collection_id: [2; 16],
        bucket: 1,
        generation: 1,
        slot: 5,
    });
    append_batch_with_durability(
        &path,
        true,
        0xABC,
        &[
            incremental.clone(),
            table(7),
            DeltaOperation::Coverage { covered_lsn: 7 },
        ],
        0x111,
        true,
    )
    .unwrap();
    let intact_len = fs::metadata(&path).unwrap().len();
    append_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental,
            table(99),
            DeltaOperation::Coverage { covered_lsn: 99 },
        ],
        0x222,
        true,
    )
    .unwrap();
    let full = fs::read(&path).unwrap();
    let intact_len = usize::try_from(intact_len).unwrap();

    // Every prefix that stops inside the second batch loses its claim.
    for cut in (intact_len + 1)..full.len() {
        fs::write(&path, &full[..cut]).unwrap();
        let log = read_delta_log(&path).unwrap();
        assert_eq!(log.coverage, Some(7), "cut at {cut}");
        assert!(log.torn_tail, "cut at {cut}");
        assert_eq!(
            read_delta_tail_fingerprint(&path)
                .unwrap()
                .unwrap()
                .coverage,
            Some(7),
            "light reader, cut at {cut}"
        );
    }
    // A single flipped bit anywhere in the second batch does the same.
    for offset in intact_len..full.len() {
        let mut damaged = full.clone();
        damaged[offset] ^= 0x01;
        fs::write(&path, &damaged).unwrap();
        let claimed = read_delta_log(&path).unwrap().coverage;
        assert_eq!(claimed, Some(7), "bit flip at {offset}");
        let light = read_delta_tail_fingerprint(&path)
            .ok()
            .flatten()
            .and_then(|log| log.coverage);
        assert_eq!(light, Some(7), "light reader, bit flip at {offset}");
    }
    // A log whose first batch is damaged claims nothing at all.
    let mut damaged = full.clone();
    damaged[DELTA_LOG_HEADER_LEN + DELTA_BATCH_HEADER_LEN + 3] ^= 0xFF;
    fs::write(&path, &damaged).unwrap();
    assert!(read_delta_log(&path).is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn list_epochs_returns_sorted_well_formed_names_only() {
    let dir = scratch_dir("epochs");
    for name in [
        "index.delta.00000000000000ff",
        "index.delta.0000000000000001",
        "index.delta.short",
        "index.delta.zzzzzzzzzzzzzzzz",
        "index.delta",
        "unrelated.file",
    ] {
        std::fs::write(dir.join(name), b"x").unwrap();
    }
    let epochs = list_epochs(&dir).unwrap();
    let fingerprints: Vec<u64> = epochs.iter().map(|(fp, _)| *fp).collect();
    assert_eq!(fingerprints, vec![1, 0xff]);
    assert!(list_epochs(&dir.join("missing")).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fingerprint_errors_render_their_context() {
    let path = Path::new("/x/index.delta");
    let invalid = invalid(path, "bad thing");
    assert_eq!(
        invalid.to_string(),
        "invalid delta fingerprint /x/index.delta: bad thing"
    );
    let io = io_error(path, "open delta log", io::Error::other("boom"));
    assert_eq!(io.to_string(), "open delta log /x/index.delta: boom");
}

#[test]
fn replay_errors_render_their_context() {
    let cases = [
        (
            DeltaReplayError::FrameOutOfBounds {
                bucket: 9,
                capacity: 4,
            },
            "delta frame bucket 9 exceeds index capacity 4",
        ),
        (
            DeltaReplayError::RequiresOwnedIndex,
            "delta replay requires an owned index",
        ),
        (
            DeltaReplayError::EmptySlot { bucket: 3 },
            "delta frame at bucket 3 carries the empty-slot sentinel",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    let slot = DeltaReplayError::SlotOutOfRange { bucket: 1, slot: 7 }.to_string();
    assert!(slot.contains("bucket 1") && slot.contains("slot 7"));
}

#[test]
fn tail_fingerprint_reader_handles_each_log_shape() {
    let dir = scratch_dir("tailfp");
    let path = dir.join(INDEX_DELTA_FILE);

    assert!(read_delta_tail_fingerprint(&path).unwrap().is_none());

    std::fs::write(&path, [0u8; 4]).unwrap();
    assert!(read_delta_tail_fingerprint(&path).is_err());

    let mut bad_magic = v1_log(&sample_v1_operations(), 0x77);
    bad_magic[0] = b'X';
    std::fs::write(&path, &bad_magic).unwrap();
    assert!(read_delta_tail_fingerprint(&path).is_err());

    let mut bad_version = v1_log(&sample_v1_operations(), 0x77);
    bad_version[4] = 9;
    std::fs::write(&path, &bad_version).unwrap();
    assert!(read_delta_tail_fingerprint(&path).is_err());

    let good = v1_log(&sample_v1_operations(), 0x77);
    std::fs::write(&path, &good).unwrap();
    let fp = read_delta_tail_fingerprint(&path).unwrap().unwrap();
    assert_eq!((fp.base_fingerprint, fp.tail_fingerprint), (0xABC, 0x77));
    assert!(!fp.torn_tail);

    let mut torn = good.clone();
    torn.extend_from_slice(&[1, 2, 3]);
    std::fs::write(&path, &torn).unwrap();
    assert!(
        read_delta_tail_fingerprint(&path)
            .unwrap()
            .unwrap()
            .torn_tail
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn length_helpers_agree_with_encoded_sizes() {
    assert_eq!(
        batch_len(2),
        Some(DELTA_BATCH_HEADER_LEN + 2 * DELTA_FRAME_LEN + DELTA_LOG_TRAILER_LEN)
    );
    assert_eq!(batch_len(usize::MAX), None);
    assert_eq!(v1_frame_len(usize::MAX), None);
    let ops = sample_v1_operations();
    assert_eq!(v1_batch_len(&ops), Some(v1_batch_bytes(&ops, 1).len()),);
    assert_eq!(v1_empty_batch_len(), v1_batch_bytes(&[], 1).len());
    for (op, len) in [
        (
            DeltaOperation::Coverage { covered_lsn: 1 },
            v1_coverage_frame_len(),
        ),
        (
            DeltaOperation::CollectionTombstone {
                collection_id: [1; 16],
                generation: 1,
            },
            v1_tombstone_frame_len(),
        ),
    ] {
        assert_eq!(encode_frame(&op).unwrap().len(), len);
    }
    assert_eq!(
        encode_frame(&DeltaOperation::Incremental(test_frame(1, 1)))
            .unwrap()
            .len(),
        v1_incremental_frame_len()
    );
    assert_eq!(
        encode_frame(&redo_set([1; 16], [2; 16], 1)).unwrap().len(),
        v1_redo_frame_len()
    );
    assert_eq!(
        v1_snapshot_frame_len(4),
        Some(
            encode_frame(&DeltaOperation::CollectionSnapshot {
                collection_id: [1; 16],
                generation: 1,
                order_key: 1,
                index_blob: vec![0; 4],
            })
            .unwrap()
            .len()
        )
    );
}

/// A raw version-table frame around `payload`, with a valid frame checksum, so a
/// test can hand the decoder a payload the encoder would never produce.
fn raw_versions_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = vec![V1_VERSIONS];
    frame.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    frame
}

fn versions_payload(revision: u8, covered_lsn: u64, count: u32, ids: &[[u8; 16]]) -> Vec<u8> {
    let mut payload = vec![revision];
    payload.extend_from_slice(&covered_lsn.to_le_bytes());
    payload.extend_from_slice(&count.to_le_bytes());
    for (index, id) in ids.iter().enumerate() {
        payload.extend_from_slice(id);
        let version = u64::try_from(index).unwrap().saturating_add(1);
        payload.extend_from_slice(&version.to_le_bytes());
    }
    payload
}

#[test]
fn a_version_table_frame_round_trips_and_rejects_every_malformed_table() {
    let entries = vec![([1; 16], 3_u64), ([2; 16], 9), ([9; 16], 12)];
    let operation = DeltaOperation::Versions {
        covered_lsn: 40,
        entries: entries.clone(),
    };
    let encoded = encode_frame(&operation).unwrap();
    assert_eq!(encoded.len(), v1_versions_frame_len(3).unwrap());
    assert_eq!(decode_frame(&encoded), Some((operation, encoded.len())));
    // An empty table is valid: a pool with no collections still claims coverage.
    let empty = DeltaOperation::Versions {
        covered_lsn: 1,
        entries: Vec::new(),
    };
    let encoded = encode_frame(&empty).unwrap();
    assert_eq!(decode_frame(&encoded), Some((empty, encoded.len())));

    // The encoder refuses a table it could not decode.
    for bad in [
        vec![([2; 16], 1_u64), ([1; 16], 2)],
        vec![([1; 16], 1), ([1; 16], 2)],
    ] {
        assert!(
            encode_frame(&DeltaOperation::Versions {
                covered_lsn: 5,
                entries: bad,
            })
            .is_err(),
            "an unsorted or duplicated table must not encode"
        );
    }

    // The decoder rejects what the encoder never writes.
    let ascending = [[1; 16], [2; 16], [3; 16]];
    let good = versions_payload(1, 7, 3, &ascending);
    assert!(decode_frame(&raw_versions_frame(&good)).is_some());
    let cases: [(&str, Vec<u8>); 7] = [
        ("an unknown revision", versions_payload(2, 7, 3, &ascending)),
        (
            "a count that disagrees with the length",
            versions_payload(1, 7, 4, &ascending),
        ),
        (
            "a count below the entries present",
            versions_payload(1, 7, 2, &ascending),
        ),
        (
            "descending ids",
            versions_payload(1, 7, 3, &[[3; 16], [2; 16], [1; 16]]),
        ),
        (
            "a duplicate id",
            versions_payload(1, 7, 3, &[[1; 16], [2; 16], [2; 16]]),
        ),
        ("a truncated header", good[..9].to_vec()),
        ("an empty payload", Vec::new()),
    ];
    for (what, payload) in cases {
        assert!(
            decode_frame(&raw_versions_frame(&payload)).is_none(),
            "{what} must not decode"
        );
    }
}

/// A coverage claim and the version table it carries are one thing: whatever a
/// reader sees of the second batch, torn at any byte or damaged by any flipped
/// bit, it never exposes the claim without its table or the table without its
/// claim, and both readers decide the same.
#[test]
fn a_version_table_and_its_claim_are_exposed_together_or_not_at_all() {
    let dir = coverage_log_dir("table-atomic");
    let path = dir.join(INDEX_DELTA_FILE);
    let incremental = DeltaOperation::Incremental(DeltaFrame {
        collection_id: [2; 16],
        bucket: 1,
        generation: 1,
        slot: 5,
    });
    let first = vec![
        DeltaOperation::Versions {
            covered_lsn: 7,
            entries: vec![([2; 16], 4)],
        },
        DeltaOperation::Coverage { covered_lsn: 7 },
    ];
    let second = vec![
        incremental.clone(),
        DeltaOperation::Versions {
            covered_lsn: 99,
            entries: vec![([2; 16], 4), ([7; 16], 98)],
        },
        DeltaOperation::Coverage { covered_lsn: 99 },
    ];
    append_batch_with_durability(&path, true, 0xABC, &first, 0x111, true).unwrap();
    let intact_len = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
    append_batch_with_durability(&path, false, 0xABC, &second, 0x222, true).unwrap();
    let full = fs::read(&path).unwrap();

    let check = |bytes: &[u8], expect_second: bool, what: &str| {
        fs::write(&path, bytes).unwrap();
        let log = read_delta_log(&path).unwrap();
        let tail = read_delta_tail_fingerprint(&path).unwrap().unwrap();
        let (coverage, table_lsn, entries) = if expect_second {
            (99, 99, 2)
        } else {
            (7, 7, 1)
        };
        assert_eq!(log.coverage, Some(coverage), "{what}");
        assert_eq!(tail.coverage, Some(coverage), "{what}: light reader");
        let table = log
            .version_table
            .as_ref()
            .expect("a claim carries its table");
        assert_eq!(table.covered_lsn, table_lsn, "{what}");
        assert_eq!(table.entries.len(), entries, "{what}");
    };
    check(&full, true, "intact");
    for cut in (intact_len + 1)..full.len() {
        check(&full[..cut], false, &format!("cut at {cut}"));
    }
    // Two bit positions per byte (the lowest and the highest): the batch checksum
    // covers every byte, and each flip is a file write and two reads.
    for at in intact_len..full.len() {
        for bit in [0, 7] {
            let mut damaged = full.clone();
            damaged[at] ^= 1 << bit;
            check(&damaged, false, &format!("bit {bit} of byte {at}"));
        }
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// Each table is a complete snapshot, so compaction keeps only the newest,
/// directly before the claim it belongs to.
#[test]
fn compaction_keeps_only_the_newest_table() {
    let dir = coverage_log_dir("table-compaction");
    let path = dir.join(INDEX_DELTA_FILE);
    let collection = [0x42; 16];
    let first = vec![
        redo_set(collection, [1; 16], 1),
        redo_set(collection, [2; 16], 2),
        DeltaOperation::Versions {
            covered_lsn: 10,
            entries: vec![([1; 16], 1)],
        },
        DeltaOperation::Coverage { covered_lsn: 10 },
    ];
    let second = vec![
        redo_set(collection, [1; 16], 3),
        redo_set(collection, [2; 16], 4),
        DeltaOperation::Versions {
            covered_lsn: 20,
            entries: vec![([1; 16], 2), ([5; 16], 18)],
        },
        DeltaOperation::Coverage { covered_lsn: 20 },
    ];
    append_batch_with_durability(&path, true, 0xABC, &first, 0x111, true).unwrap();
    append_batch_with_durability(&path, false, 0xABC, &second, 0x222, true).unwrap();
    let before = read_delta_log(&path).unwrap();
    assert_eq!(before.version_table.as_ref().unwrap().covered_lsn, 20);

    compact_delta_log(&path)
        .expect("compaction succeeds")
        .expect("the superseded operations were released");
    let after = read_delta_log(&path).expect("the compacted log reads");
    assert_eq!(after.coverage, Some(20));
    assert_eq!(after.version_table, before.version_table);
    assert_eq!(
        after
            .operations
            .iter()
            .filter(|operation| matches!(operation, DeltaOperation::Versions { .. }))
            .count(),
        1,
        "only the newest table survives"
    );
    let n = after.operations.len();
    assert!(matches!(
        after.operations[n - 2],
        DeltaOperation::Versions {
            covered_lsn: 20,
            ..
        }
    ));
    assert!(matches!(
        after.operations[n - 1],
        DeltaOperation::Coverage { covered_lsn: 20 }
    ));
    fs::remove_dir_all(&dir).unwrap();
}

/// A durable append syncs the parent directory unless the caller has already
/// established that the log's current directory entry is durable. A caller
/// that is unsure must still pay for it, and a failed sync surfaces as an
/// error rather than being swallowed.
#[test]
fn a_durable_append_skips_the_parent_sync_only_when_the_entry_is_known_durable() {
    let dir = coverage_log_dir("dir-durable");
    let path = dir.join(INDEX_DELTA_FILE);
    let operations = [DeltaOperation::Incremental(test_frame(1, 1))];
    append_batch_with_durability(&path, true, 0xABC, &operations, 0x111, true).unwrap();

    crate::shard::fail_directory_sync_after(&dir, 0);
    // The wrapper does not claim the entry is durable, so it must sync and
    // the injected failure must reach the caller.
    assert!(
        append_batch_with_durability(&path, false, 0xABC, &operations, 0x222, true).is_err(),
        "an unproven directory entry must still be synced"
    );
    // With the entry known durable, the bytes are still synced but the parent
    // is not, so the injected failure does not apply.
    append_batch_with_durability_and_directory(&path, false, 0xABC, &operations, 0x333, true, true)
        .unwrap();
    crate::shard::clear_directory_sync_failure(&dir);

    let log = read_delta_log(&path).unwrap();
    assert_eq!(log.tail_fingerprint, 0x333);
    assert_eq!(log.operations.len(), 3);
    fs::remove_dir_all(&dir).unwrap();
}
