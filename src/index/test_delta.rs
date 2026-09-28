use super::*;

fn test_frame(bucket: u32, generation: u64) -> DeltaFrame {
    DeltaFrame {
        collection_id: [0x44, 0x4C, 0x54, 0x52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bucket,
        generation,
        slot: u64::from(bucket).wrapping_mul(7).wrapping_add(1),
    }
}

#[test]
fn header_and_batch_header_and_trailer_round_trip() {
    let header = encode_header(0xDEAD_BEEF);
    assert_eq!(&header[..4], DELTA_LOG_MAGIC);
    assert_eq!(header[4], DELTA_LOG_VERSION);
    assert_eq!(
        u64::from_le_bytes(header[BASE_FINGERPRINT_OFFSET..16].try_into().unwrap()),
        0xDEAD_BEEF
    );

    let batch = encode_batch_header(12);
    assert_eq!(&batch[..4], DELTA_BATCH_MAGIC);
    assert_eq!(u32::from_le_bytes(batch[4..8].try_into().unwrap()), 12);

    let trailer = encode_trailer(0xCAFE_F00D, 0x1234_5678);
    assert_eq!(&trailer[..4], DELTA_LOG_TRAILER_MAGIC);
    assert_eq!(
        u32::from_le_bytes(trailer[TRAILER_CRC_OFFSET..8].try_into().unwrap()),
        0x1234_5678
    );
    assert_eq!(
        u64::from_le_bytes(trailer[TRAILER_FINGERPRINT_OFFSET..16].try_into().unwrap()),
        0xCAFE_F00D
    );
}

#[test]
fn single_batch_round_trips_as_committed() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_single_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    let frames = vec![test_frame(0, 3), test_frame(1, 3), test_frame(2, 3)];
    append_batch(&path, true, 7, &frames, 99).unwrap();

    let log = read_delta_log(&path).expect("valid log reads");
    assert_eq!(log.base_fingerprint, 7);
    assert_eq!(log.tail_fingerprint, 99);
    assert_eq!(log.frames, frames);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn multiple_batches_keep_every_committed_frame() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_multi_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    append_batch(&path, true, 7, &[test_frame(0, 3), test_frame(1, 3)], 11).unwrap();
    append_batch(
        &path,
        false,
        7,
        &[test_frame(2, 3), test_frame(3, 3), test_frame(4, 3)],
        22,
    )
    .unwrap();

    let log = read_delta_log(&path).expect("multi-batch log reads");
    assert_eq!(log.tail_fingerprint, 22, "the final trailer wins");
    assert_eq!(log.frames.len(), 5);
    assert_eq!(log.frames[4], test_frame(4, 3));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn torn_final_batch_is_dropped() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_torn_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    // Two committed batches, then a third batch with a torn trailer.
    append_batch(&path, true, 7, &[test_frame(0, 3), test_frame(1, 3)], 11).unwrap();
    append_batch(&path, false, 7, &[test_frame(2, 3), test_frame(3, 3)], 22).unwrap();
    append_batch(&path, false, 7, &[test_frame(4, 3), test_frame(5, 3)], 33).unwrap();
    let full = std::fs::read(&path).unwrap();
    // Cut 4 bytes off the shortest possible third batch (frames fully
    // written, trailer partial).
    let torn_len = full
        .len()
        .saturating_sub(DELTA_LOG_TRAILER_LEN.saturating_sub(4));
    std::fs::write(&path, &full[..torn_len]).unwrap();

    let log = read_delta_log(&path).expect("torn tail must not fail the committed prefix");
    assert_eq!(log.tail_fingerprint, 22);
    assert_eq!(
        log.frames.len(),
        4,
        "frames of the unfinalized batch are dropped"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unfinalized_frames_are_never_stranded() {
    // A complete batch followed by a batch whose frames are fully written
    // but whose trailer is missing entirely must also drop that batch.
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_no_trailer_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();

    // Append a second batch's header + frames but no trailer.
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(&encode_batch_header(2)).unwrap();
    file.write_all(&test_frame(1, 3).encode()).unwrap();
    file.write_all(&test_frame(2, 3).encode()).unwrap();
    let _ = file.sync_all();

    let log = read_delta_log(&path).expect("frame-without-trailer batch is dropped, not fatal");
    assert_eq!(log.tail_fingerprint, 11);
    assert_eq!(log.frames.len(), 1);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn corrupted_frame_bytes_are_rejected_by_crc() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_crc_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    // A committed batch, then a second batch whose frame bytes are
    // corrupted after being fully written — the length framing alone
    // can't catch this; only the CRC can.
    append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();
    append_batch(&path, false, 7, &[test_frame(1, 3), test_frame(2, 3)], 22).unwrap();
    let mut full = std::fs::read(&path).unwrap();
    // Flip a byte inside the second batch's frame region (well past the
    // first batch's header+frame+trailer).
    let corrupt_at = full.len() - DELTA_LOG_TRAILER_LEN - 1;
    full[corrupt_at] ^= 0xFF;
    std::fs::write(&path, &full).unwrap();

    let log = read_delta_log(&path).expect("first committed batch must still be trusted");
    assert_eq!(log.tail_fingerprint, 11, "corrupted batch must be dropped");
    assert_eq!(log.frames.len(), 1);
    assert_eq!(log.frames[0], test_frame(0, 3));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn corrupted_tail_fingerprint_is_rejected_by_crc() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_fp_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);
    // A committed batch, then a second batch whose frame bytes are
    // untouched but whose trailer fingerprint is corrupted after being
    // written. The CRC covers the fingerprint too, so this must be
    // caught even though every frame decodes fine.
    append_batch(&path, true, 7, &[test_frame(0, 3)], 11).unwrap();
    append_batch(&path, false, 7, &[test_frame(1, 3), test_frame(2, 3)], 22).unwrap();
    let mut full = std::fs::read(&path).unwrap();
    // Flip the last byte of the file: the trailer's tail_fingerprint
    // field, not any frame byte.
    let last = full.len() - 1;
    full[last] ^= 0xFF;
    std::fs::write(&path, &full).unwrap();

    let log = read_delta_log(&path).expect("first committed batch must still be trusted");
    assert_eq!(
        log.tail_fingerprint, 11,
        "batch with corrupted fingerprint must be dropped"
    );
    assert_eq!(log.frames.len(), 1);
    assert_eq!(log.frames[0], test_frame(0, 3));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn entirely_torn_or_missing_log_is_none() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_torn_all_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    // Missing file.
    assert!(read_delta_log(&path).is_none());

    // Header but no complete committed batch (header + frames, no trailer).
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&encode_header(7)).unwrap();
    file.write_all(&encode_batch_header(1)).unwrap();
    file.write_all(&test_frame(0, 3).encode()).unwrap();
    let _ = file.sync_all();
    assert!(read_delta_log(&path).is_none());

    // Bad magic.
    std::fs::write(&path, vec![0xAB; 64]).unwrap();
    assert!(read_delta_log(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

fn v3_batch_bytes(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
    let mut frames = Vec::new();
    for operation in operations {
        frames.extend_from_slice(&encode_v3_frame(operation).unwrap());
    }
    let crc = batch_crc(&frames, tail_fingerprint);
    let mut batch = Vec::from(encode_batch_header(
        u32::try_from(operations.len()).unwrap(),
    ));
    batch.extend_from_slice(&frames);
    batch.extend_from_slice(&encode_trailer(tail_fingerprint, crc));
    batch
}

fn sample_v3_operations() -> Vec<DeltaOperation> {
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

fn v3_log(operations: &[DeltaOperation], tail_fingerprint: u64) -> Vec<u8> {
    let mut log = Vec::from(encode_header(0xABC));
    log[4] = DELTA_LOG_VERSION_V3;
    log.extend_from_slice(&v3_batch_bytes(operations, tail_fingerprint));
    log
}

#[test]
fn v3_log_round_trips_operations() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_round_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v3_operations();
    let mut log = v3_log(&operations[..2], 0x111);
    log.extend_from_slice(&v3_batch_bytes(&operations[2..], 0x222));
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log_v3(&path).expect("valid v3 log reads");
    assert_eq!(decoded.base_fingerprint, 0xABC);
    assert_eq!(decoded.tail_fingerprint, 0x222);
    assert_eq!(decoded.operations, operations);
    assert!(!decoded.torn_tail);
    assert_eq!(decoded.file_len, u64::try_from(log.len()).unwrap());
    let tail = read_delta_tail_fingerprint(&path)
        .unwrap()
        .expect("v3 tail scan succeeds");
    assert_eq!(tail.base_fingerprint, decoded.base_fingerprint);
    assert_eq!(tail.tail_fingerprint, decoded.tail_fingerprint);
    assert_eq!(tail.torn_tail, decoded.torn_tail);
    // The v2 reader must not misread a v3 epoch.
    assert!(read_delta_log(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v3_torn_tail_keeps_the_committed_prefix() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_torn_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v3_operations();
    let mut log = v3_log(&operations[..1], 0x111);
    let committed_len = u64::try_from(log.len()).unwrap();
    // A second batch with a header and one frame but no trailer.
    log.extend_from_slice(&encode_batch_header(2));
    log.extend_from_slice(&encode_v3_frame(&operations[1]).unwrap());
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log_v3(&path).expect("committed prefix is trusted");
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
fn v3_corrupt_batch_drops_that_batch_and_later() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let operations = sample_v3_operations();
    let mut log = v3_log(&operations[..1], 0x111);
    let first_batch_end = log.len();
    log.extend_from_slice(&v3_batch_bytes(&operations[1..], 0x222));
    // Corrupt one byte inside the second batch's frame region.
    let corrupt_at = first_batch_end
        .saturating_add(DELTA_BATCH_HEADER_LEN)
        .saturating_add(4);
    log[corrupt_at] ^= 0x20;
    std::fs::write(&path, &log).unwrap();

    let decoded = read_delta_log_v3(&path).expect("first batch is trusted");
    assert_eq!(decoded.operations, vec![operations[0].clone()]);
    assert_eq!(decoded.tail_fingerprint, 0x111);
    assert!(decoded.torn_tail);
    assert_eq!(decoded.file_len, u64::try_from(first_batch_end).unwrap());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v3_reader_rejects_v2_and_empty_logs() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_reject_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    // A v2 header (the default `encode_header` version) is not a v3 log.
    std::fs::write(&path, encode_header(1)).unwrap();
    assert!(read_delta_log_v3(&path).is_none());

    // A v3 header with no committed batch is not replayable.
    let mut header_only = Vec::from(encode_header(1));
    header_only[4] = DELTA_LOG_VERSION_V3;
    std::fs::write(&path, header_only).unwrap();
    assert!(read_delta_log_v3(&path).is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn v3_frames_round_trip_every_variant() {
    // Incremental, CollectionSnapshot, and CollectionTombstone: each must
    // survive encode -> decode byte-for-byte.
    for operation in sample_v3_operations() {
        let frame = encode_v3_frame(&operation).expect("frame encodes");
        let (decoded, consumed) = decode_v3_frame(&frame).expect("frame decodes");
        assert_eq!(decoded, operation);
        assert_eq!(consumed, frame.len());
    }
}

#[test]
fn v3_frame_decode_consumes_exactly_one_frame() {
    let operations = sample_v3_operations();
    let first = encode_v3_frame(&operations[0]).unwrap();
    let second = encode_v3_frame(&operations[1]).unwrap();
    let mut concatenated = first.clone();
    concatenated.extend_from_slice(&second);

    let (decoded, consumed) = decode_v3_frame(&concatenated).expect("first frame decodes");
    assert_eq!(decoded, operations[0]);
    assert_eq!(consumed, first.len());

    let (decoded_second, second_consumed) =
        decode_v3_frame(&concatenated[consumed..]).expect("second frame decodes");
    assert_eq!(decoded_second, operations[1]);
    assert_eq!(second_consumed, second.len());
}

#[test]
fn v3_frame_decode_rejects_corruption_and_truncation() {
    let operation = DeltaOperation::CollectionSnapshot {
        collection_id: [5; 16],
        generation: 2,
        order_key: 1,
        index_blob: vec![9, 8, 7, 6],
    };
    let frame = encode_v3_frame(&operation).unwrap();

    // Every truncation short of the complete frame must fail, not panic.
    for shorter in [0, 1, V3_FRAME_HEADER_LEN, frame.len().saturating_sub(1)] {
        assert!(decode_v3_frame(&frame[..shorter]).is_none());
    }

    // A flipped payload byte must be caught by the frame CRC.
    let mut corrupt_payload = frame.clone();
    corrupt_payload[V3_FRAME_HEADER_LEN] ^= 0x40;
    assert!(decode_v3_frame(&corrupt_payload).is_none());

    // ...as must a flipped CRC byte.
    let mut corrupt_crc = frame.clone();
    let crc_byte = corrupt_crc.len().saturating_sub(1);
    corrupt_crc[crc_byte] ^= 0x01;
    assert!(decode_v3_frame(&corrupt_crc).is_none());

    // A length field reaching past the buffer must fail, not allocate.
    let mut bad_len = frame.clone();
    bad_len[1..5].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_v3_frame(&bad_len).is_none());

    // A plausible, shorter length has enough bytes for framing, so this
    // specifically proves the CRC covers the length field. The decoder
    // must reject it at checksum validation rather than only at bounds
    // checking.
    let mut bad_len = frame.clone();
    let payload_len = u32::from_le_bytes(frame[1..5].try_into().unwrap());
    bad_len[1..5].copy_from_slice(&payload_len.checked_sub(1).unwrap().to_le_bytes());
    assert!(decode_v3_frame(&bad_len).is_none());

    // An unknown operation kind with a recomputed valid CRC is rejected.
    let mut unknown = frame.clone();
    unknown[0] = 0x7F;
    let crc_len = unknown.len().saturating_sub(V3_FRAME_TRAILER_LEN);
    let crc = crc32fast::hash(&unknown[..crc_len]).to_le_bytes();
    unknown[crc_len..].copy_from_slice(&crc);
    assert!(decode_v3_frame(&unknown).is_none());
}

#[test]
fn v3_frame_encoded_sizes_are_exact() {
    let incremental = DeltaOperation::Incremental(test_frame(3, 5));
    assert_eq!(
        encode_v3_frame(&incremental).unwrap().len(),
        V3_FRAME_HEADER_LEN + 36 + V3_FRAME_TRAILER_LEN,
        "incremental framing is header + fixed payload + crc"
    );

    let snapshot = DeltaOperation::CollectionSnapshot {
        collection_id: [0x11; 16],
        generation: 7,
        order_key: 42,
        index_blob: vec![0xAB; 128],
    };
    assert_eq!(
        encode_v3_frame(&snapshot).unwrap().len(),
        V3_FRAME_HEADER_LEN + V3_SNAPSHOT_FIXED_LEN + 128 + V3_FRAME_TRAILER_LEN,
        "snapshot framing is header + fixed fields + blob + crc"
    );

    let tombstone = DeltaOperation::CollectionTombstone {
        collection_id: [0x22; 16],
        generation: 9,
    };
    assert_eq!(
        encode_v3_frame(&tombstone).unwrap().len(),
        V3_FRAME_HEADER_LEN + V3_TOMBSTONE_LEN + V3_FRAME_TRAILER_LEN,
        "tombstone framing is header + fixed fields + crc"
    );
}

#[test]
fn v3_snapshot_rejects_blob_length_mismatch() {
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
    frame.push(V3_COLLECTION_SNAPSHOT);
    frame.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    frame.extend_from_slice(&payload);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    assert!(
        decode_v3_frame(&frame).is_none(),
        "declared blob length must match the payload exactly"
    );
}

#[test]
fn v3_append_batch_round_trips_through_the_reader() {
    let dir = std::env::temp_dir().join(format!("mtxdb_delta_v3_append_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(INDEX_DELTA_FILE);

    let first = sample_v3_operations();
    let mut expected_len = DELTA_LOG_HEADER_LEN
        .saturating_add(DELTA_BATCH_HEADER_LEN)
        .saturating_add(DELTA_LOG_TRAILER_LEN);
    for operation in &first {
        expected_len = expected_len.saturating_add(encode_v3_frame(operation).unwrap().len());
    }
    let appended = append_v3_batch(&path, true, 0xABC, &first, 0x111).expect("append v3 batch");
    assert_eq!(appended, expected_len, "append reports the bytes it wrote");

    let second = [DeltaOperation::CollectionTombstone {
        collection_id: [3; 16],
        generation: 4,
    }];
    append_v3_batch(&path, false, 0xABC, &second, 0x222).expect("continue the v3 log");

    let decoded = read_delta_log_v3(&path).expect("appended v3 log reads");
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

/// Coverage is an ordinary v3 frame: it round-trips, is sized exactly, and
/// its payload must be exactly one LSN.
#[test]
fn v3_coverage_frame_round_trips() {
    let coverage = DeltaOperation::Coverage {
        covered_lsn: 0xDEAD_BEEF,
    };
    let encoded = encode_v3_frame(&coverage).unwrap();
    assert_eq!(
        encoded.len(),
        V3_FRAME_HEADER_LEN + V3_COVERAGE_LEN + V3_FRAME_TRAILER_LEN
    );
    assert_eq!(
        decode_v3_frame(&encoded),
        Some((coverage.clone(), encoded.len()))
    );
    assert_eq!(
        v3_batch_len(&[coverage]).unwrap(),
        DELTA_BATCH_HEADER_LEN + encoded.len() + DELTA_LOG_TRAILER_LEN
    );
    // A coverage frame with the wrong payload width is rejected.
    let mut frame = vec![V3_COVERAGE];
    frame.extend_from_slice(&4_u32.to_le_bytes());
    frame.extend_from_slice(&[1, 2, 3, 4]);
    frame.extend_from_slice(&crc32fast::hash(&frame).to_le_bytes());
    assert!(decode_v3_frame(&frame).is_none());
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
    append_v3_batch(&path, true, 0xABC, &[incremental(1), incremental(2)], 0x111).unwrap();
    let log = read_delta_log_v3(&path).unwrap();
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
    append_v3_batch_with_durability(
        &path,
        false,
        0xABC,
        &[incremental(3), DeltaOperation::Coverage { covered_lsn: 40 }],
        0x222,
        true,
    )
    .unwrap();
    // batch 3: more ops with no coverage: a later claim is not implied
    append_v3_batch(&path, false, 0xABC, &[incremental(4)], 0x333).unwrap();
    let log = read_delta_log_v3(&path).unwrap();
    assert_eq!(log.coverage, Some(40));
    assert_eq!(
        log.coverage_prefix_ops, 4,
        "two + one + the coverage op itself"
    );
    assert_eq!(log.operations.len(), 5);
    assert_eq!(
        read_delta_tail_fingerprint(&path)
            .unwrap()
            .unwrap()
            .coverage,
        Some(40)
    );
    // batch 4: a lower claim never lowers the coverage
    append_v3_batch(
        &path,
        false,
        0xABC,
        &[DeltaOperation::Coverage { covered_lsn: 10 }],
        0x444,
    )
    .unwrap();
    let log = read_delta_log_v3(&path).unwrap();
    assert_eq!(log.coverage, Some(40));
    assert_eq!(log.coverage_prefix_ops, 6);
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
        DeltaOperation::Coverage { covered_lsn: 5 },
    ];
    let projected = v3_empty_batch_len()
        + v3_incremental_frame_len()
        + v3_snapshot_frame_len(blob.len()).unwrap()
        + v3_tombstone_frame_len()
        + v3_coverage_frame_len();
    assert_eq!(v3_batch_len(&operations), Some(projected));
    let dir = coverage_log_dir("projected_len");
    let path = dir.join(INDEX_DELTA_FILE);
    let written =
        append_v3_batch_with_durability(&path, false, 0xABC, &operations, 0x1, false).unwrap();
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
    append_v3_batch_with_durability(&path, true, 0xABC, &[incremental(1)], 0x111, false).unwrap();
    append_v3_batch_with_durability(
        &path,
        false,
        0xABC,
        &[incremental(2), DeltaOperation::Coverage { covered_lsn: 9 }],
        0x222,
        false,
    )
    .unwrap();
    append_v3_batch_with_durability(
        &path,
        false,
        0xABC,
        &[
            incremental(3),
            incremental(4),
            DeltaOperation::Coverage { covered_lsn: 20 },
        ],
        0x333,
        false,
    )
    .unwrap();
    let full = fs::read(&path).unwrap();
    for cut in DELTA_LOG_HEADER_LEN..=full.len() {
        fs::write(&path, &full[..cut]).unwrap();
        let log = read_delta_log_v3(&path);
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
    let bad_batches: [Vec<DeltaOperation>; 2] = [
        vec![
            DeltaOperation::Coverage { covered_lsn: 50 },
            incremental.clone(),
        ],
        vec![
            DeltaOperation::Coverage { covered_lsn: 50 },
            DeltaOperation::Coverage { covered_lsn: 60 },
        ],
    ];
    for (index, bad) in bad_batches.iter().enumerate() {
        let dir = coverage_log_dir(&format!("misplaced-{index}"));
        let path = dir.join(INDEX_DELTA_FILE);
        append_v3_batch_with_durability(
            &path,
            true,
            0xABC,
            &[
                incremental.clone(),
                DeltaOperation::Coverage { covered_lsn: 7 },
            ],
            0x111,
            true,
        )
        .unwrap();
        append_v3_batch_with_durability(&path, false, 0xABC, bad, 0x222, true).unwrap();
        let log = read_delta_log_v3(&path).unwrap();
        assert_eq!(log.coverage, Some(7), "case {index}");
        assert_eq!(log.coverage_prefix_ops, 2, "case {index}");
        assert_eq!(log.operations.len(), 2, "case {index}");
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
    let cases: [Vec<DeltaOperation>; 4] = [
        vec![incremental.clone(), claim(50)],
        vec![claim(50), incremental.clone()],
        vec![claim(50), claim(60)],
        vec![incremental.clone()],
    ];
    for (index, batch) in cases.iter().enumerate() {
        let dir = coverage_log_dir(&format!("agree-{index}"));
        let path = dir.join(INDEX_DELTA_FILE);
        append_v3_batch_with_durability(
            &path,
            true,
            0xABC,
            &[incremental.clone(), claim(7)],
            0x111,
            true,
        )
        .unwrap();
        append_v3_batch_with_durability(&path, false, 0xABC, batch, 0x222, true).unwrap();
        let full = read_delta_log_v3(&path).unwrap();
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
    append_v3_batch_with_durability(&path, true, 0xABC, &[incremental(1)], 0x111, true).unwrap();
    let first_len = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
    append_v3_batch_with_durability(
        &path,
        false,
        0xABC,
        &[incremental(2), DeltaOperation::Coverage { covered_lsn: 9 }],
        0x222,
        true,
    )
    .unwrap();
    let intact = read_delta_log_v3(&path).unwrap();
    assert_eq!(intact.coverage, Some(9));
    assert_eq!(intact.operations.len(), 3);

    let mut bytes = fs::read(&path).unwrap();
    let frame_len = V3_FRAME_HEADER_LEN + V3_COVERAGE_LEN + V3_FRAME_TRAILER_LEN;
    let trailer_start = bytes.len() - DELTA_LOG_TRAILER_LEN;
    let frame_start = trailer_start - frame_len;
    bytes[frame_start] = 0x7F;
    let frame_crc_at = trailer_start - V3_FRAME_TRAILER_LEN;
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

    let old = read_delta_log_v3(&path).unwrap();
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
    append_v3_batch_with_durability(
        &path,
        true,
        0xABC,
        &[
            incremental.clone(),
            DeltaOperation::Coverage { covered_lsn: 7 },
        ],
        0x111,
        true,
    )
    .unwrap();
    let intact_len = fs::metadata(&path).unwrap().len();
    append_v3_batch_with_durability(
        &path,
        false,
        0xABC,
        &[incremental, DeltaOperation::Coverage { covered_lsn: 99 }],
        0x222,
        true,
    )
    .unwrap();
    let full = fs::read(&path).unwrap();
    let intact_len = usize::try_from(intact_len).unwrap();

    // Every prefix that stops inside the second batch loses its claim.
    for cut in (intact_len + 1)..full.len() {
        fs::write(&path, &full[..cut]).unwrap();
        let log = read_delta_log_v3(&path).unwrap();
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
        let claimed = read_delta_log_v3(&path).unwrap().coverage;
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
    assert!(read_delta_log_v3(&path).is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}
