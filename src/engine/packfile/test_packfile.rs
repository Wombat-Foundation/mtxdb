use super::*;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn test_record(collection_id: [u8; 16], hash: [u8; 16], data: &[u8]) -> Record {
    Record {
        collection_id,
        hash,
        data: Bytes::copy_from_slice(data),
        metadata: None,
    }
}

fn test_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mdb_test_pf_{name}_{}_{id}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn test_record_raw(hash: [u8; 16], data: &[u8]) -> Record {
    Record {
        collection_id: [0xAA; 16],
        hash,
        data: Bytes::copy_from_slice(data),
        metadata: None,
    }
}

/// Deterministic [`PackId`] fixture: the integer's big-endian bytes in the
/// first 8 of 32, so its hex display leads with the integer (matching the
/// filenames used in these tests).
use super::test_support::pack_id_for as test_pack_id;

#[test]
fn test_write_read_roundtrip() {
    let record = test_record_raw([0xaa; 16], b"hello world");
    let mut buf = Vec::new();
    write_record(&mut buf, &record).unwrap();

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(record, read);
}

/// Highly-compressible, well-above-threshold data should be stored
/// with the compressed flag set and round-trip exactly.
#[cfg(feature = "zstd")]
#[test]
fn test_write_read_roundtrip_compressed() {
    let data = vec![0x42u8; 8192];
    let record = test_record_raw([0xaa; 16], &data);
    let mut buf = Vec::new();
    let written = write_record(&mut buf, &record).unwrap();
    assert_eq!(written, buf.len() as u64);
    assert!(
        (buf.len() as u64) < record.serialized_len() as u64,
        "highly compressible data should shrink on disk: {} vs uncompressed upper bound {}",
        buf.len(),
        record.serialized_len()
    );

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(record, read);
}

/// `read_record_metadata` must agree with `read_record` on `collection_id`/`hash`
/// for both a compressed and a raw-fallback frame, without touching
/// (or needing to decompress) the payload.
#[test]
fn test_read_record_metadata_matches_read_record() {
    for data in [vec![0x42u8; 8192], b"hi".to_vec()] {
        let record = test_record_raw([0x77; 16], &data);
        let mut buf = Vec::new();
        write_record(&mut buf, &record).unwrap();

        let mut cursor = Cursor::new(&buf);
        let full = read_record(&mut cursor).unwrap().unwrap();

        let mut cursor = Cursor::new(&buf);
        let meta = read_record_metadata(&mut cursor).unwrap().unwrap();

        assert_eq!(meta.collection_id, full.collection_id);
        assert_eq!(meta.hash, full.hash);
    }
}

/// A metadata-bearing frame must round-trip its metadata through both the
/// full `read_record` path and the metadata-only scan path, including
/// zero-copy-preserving structural fields.
#[test]
fn test_metadata_roundtrip() {
    let metadata = FrameMetadata {
        logical_id: Some([0x11; 32]),
        content_digest: Some([0x22; 32]),
        digest_algorithm: DigestAlgorithm::Sha256,
        role: Some(b"canonical_event".to_vec()),
        last_write_lsn: Some(0x0102_0304_0506_0708),
        unknown: vec![(0x7f, vec![1, 2, 3])],
    };
    let record = Record {
        collection_id: [0xAA; 16],
        hash: [0xBB; 16],
        data: Bytes::from_static(b"hello metadata"),
        metadata: Some(metadata.clone()),
    };
    let mut buf = Vec::new();
    write_record(&mut buf, &record).unwrap();

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(read.data, record.data);
    assert_eq!(read.metadata.as_ref(), Some(&metadata));

    let mut cursor = Cursor::new(&buf);
    let meta = read_record_metadata(&mut cursor).unwrap().unwrap();
    assert_eq!(meta.metadata.as_ref(), Some(&metadata));

    let mut cursor = Cursor::new(&buf);
    let skipped = read_record_metadata_skip_payload(&mut cursor)
        .unwrap()
        .unwrap();
    assert_eq!(skipped.metadata.as_ref(), Some(&metadata));
}

/// The per-record write LSN is a forward-compatible optional tag: absent
/// means a legacy v4 frame (version 0), it round-trips exactly, and a value
/// that is not eight bytes is rejected rather than silently ignored.
#[test]
fn test_last_write_lsn_metadata_tag() {
    assert_eq!(FrameMetadata::default().last_write_lsn, None);

    let metadata = FrameMetadata {
        last_write_lsn: Some(0xDEAD_BEEF_0102_0304),
        ..FrameMetadata::default()
    };
    assert!(!metadata.is_empty());
    let encoded = metadata.encode().unwrap();
    let (decoded, consumed) = FrameMetadata::decode(&encoded).unwrap();
    assert_eq!(consumed, encoded.len());
    assert_eq!(decoded.last_write_lsn, Some(0xDEAD_BEEF_0102_0304));

    let mut malformed = vec![METADATA_VERSION];
    let mut tlv = vec![META_TAG_LAST_WRITE_LSN];
    tlv.extend_from_slice(&4u32.to_le_bytes());
    tlv.extend_from_slice(&[1, 2, 3, 4]);
    malformed.extend_from_slice(&u32::try_from(tlv.len()).unwrap().to_le_bytes());
    malformed.extend_from_slice(&tlv);
    let err = FrameMetadata::decode(&malformed).unwrap_err();
    assert!(err.to_string().contains("last_write_lsn"));
}

/// The metadata flag must be set only when metadata is present; a record
/// with `None` (or empty) metadata stays byte-identical to a v4 frame with
/// no metadata area, so generic records pay nothing.
#[test]
fn test_no_metadata_frame_is_unchanged() {
    let plain = test_record_raw([0xCC; 16], b"payload");
    let mut plain_buf = Vec::new();
    write_record(&mut plain_buf, &plain).unwrap();
    assert_eq!(plain_buf[4] & FLAG_METADATA, 0);

    let empty = Record {
        metadata: Some(FrameMetadata::default()),
        ..plain.clone()
    };
    let mut empty_buf = Vec::new();
    write_record(&mut empty_buf, &empty).unwrap();
    assert_eq!(empty_buf, plain_buf);
}

/// A metadata-bearing frame with a CRC-disabled policy must still write,
/// parse, and skip correctly (the metadata block is inside the frame's
/// CRC region, but CRC-disabled frames simply carry no checksum).
#[test]
fn test_metadata_with_crc_disabled() {
    let record = Record {
        collection_id: [0x01; 16],
        hash: [0x02; 16],
        data: Bytes::from_static(b"body"),
        metadata: Some(FrameMetadata {
            logical_id: Some([0x09; 32]),
            ..FrameMetadata::default()
        }),
    };
    let mut buf = Vec::new();
    encode_record_with_options(&record, true, false)
        .map(|bytes| buf.extend_from_slice(&bytes))
        .unwrap();

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(read.metadata, record.metadata);
    assert_eq!(read.data, record.data);
}

/// A frame spanning multiple `SCAN_DISCARD_BUF_LEN`-sized chunks must
/// still stream correctly (payload larger than one discard buffer).
#[test]
fn test_read_record_metadata_spans_multiple_discard_chunks() {
    // Incompressible so it stays large on disk and forces several
    // discard-buffer iterations through the streaming loop.
    let data: Vec<u8> = (0..(SCAN_DISCARD_BUF_LEN * 3 + 500))
        .scan(0x9E37_79B9_7F4A_7C15u64, |state, _| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            Some(u8::try_from(*state & 0xff).expect("low byte fits in u8"))
        })
        .collect();
    let record = test_record_raw([0x99; 16], &data);
    let mut buf = Vec::new();
    write_record(&mut buf, &record).unwrap();

    let mut cursor = Cursor::new(&buf);
    let meta = read_record_metadata(&mut cursor).unwrap().unwrap();
    assert_eq!(meta.collection_id, record.collection_id);
    assert_eq!(meta.hash, record.hash);
}

/// Corruption inside the node-payload region must still be caught by
/// `read_record_metadata` — it streams payload bytes through the CRC
/// as it discards them rather than skipping them outright, so this
/// must fail exactly like `read_record` does on the same corrupted
/// buffer, not silently succeed because the payload was never
/// "read" for its own sake.
#[test]
fn test_read_record_metadata_detects_payload_corruption() {
    let record = test_record_raw([0xbb; 16], b"payload region corruption test");
    let mut buf = Vec::new();
    write_record(&mut buf, &record).unwrap();

    // Flip a byte inside the node-bytes region (right after the fixed
    // 37-byte header: len(4) + flags(1) + uncompressed_len(4) +
    // collection_id(16) + hash(16)).
    let corrupt_at = 4 + FRAME_FIXED_LEN as usize + 3;
    buf[corrupt_at] ^= 0xff;

    let mut cursor = Cursor::new(&buf);
    let err = read_record_metadata(&mut cursor).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("CRC mismatch"));
}

/// A frame whose metadata block declares more bytes than the frame can
/// hold must be rejected *before* the reader allocates a buffer sized by
/// that disk-supplied length, rather than requesting an allocation far
/// larger than the frame (and the `u32::MAX` cap) permits.
#[test]
fn test_metadata_length_cannot_exceed_frame() {
    let frame_len = FRAME_FIXED_LEN + 5;
    let mut buf = Vec::new();
    buf.extend_from_slice(&frame_len.to_le_bytes());
    let mut fixed = [0u8; FRAME_FIXED_LEN as usize];
    fixed[0] = FLAG_METADATA;
    buf.extend_from_slice(&fixed);
    // Metadata prefix: a valid version, then a `tlv_len` claiming the
    // largest possible metadata block (a ~4 GiB allocation without the
    // bound), far beyond the five bytes of metadata space the frame has.
    buf.push(METADATA_VERSION);
    buf.extend_from_slice(&(u32::MAX - 5).to_le_bytes());

    let mut cursor = Cursor::new(&buf);
    let Err(err) = read_frame_header(&mut cursor) else {
        panic!("oversized metadata must be rejected");
    };
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("metadata"),
        "unexpected error: {err}"
    );
}

/// `checked_metadata_block_len` is inclusive at the frame body and rejects
/// one byte past it, so the guard is verified independently of allocation.
#[test]
fn test_checked_metadata_block_len_boundary() {
    let frame_len = FRAME_FIXED_LEN + 20;
    // block_len = 5 + tlv_len; body = 20, so tlv_len = 15 is exactly full.
    assert_eq!(checked_metadata_block_len(15, frame_len).unwrap(), 20);
    // One byte past the body is rejected.
    assert_eq!(
        checked_metadata_block_len(16, frame_len)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    // A `tlv_len` that overflows the 5-byte prefix is rejected too.
    assert_eq!(
        checked_metadata_block_len(u32::MAX, frame_len)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    // A frame shorter than the fixed header cannot hold any metadata.
    // Production callers reject it earlier via the `FRAME_FIXED_LEN
    // ..= u32::MAX` range check; this pins the helper's own behavior.
    assert_eq!(
        checked_metadata_block_len(0, FRAME_FIXED_LEN - 1)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
}

/// The bound is inclusive end to end: a metadata block that exactly fills
/// the frame's post-header body (zero node bytes) is a valid maximum-size
/// frame and must still decode, so the fix rejects only overrun.
#[test]
fn test_metadata_at_frame_body_boundary_decodes() {
    // One unknown TLV entry sized so the block exactly fills the frame
    // body: block = 5 (prefix) + 5 (tag + value_len) + value.
    let value_len = 64usize;
    let tlv_len = u32::try_from(5 + value_len).expect("test TLV length fits u32");
    let block_len = 5 + tlv_len;
    let frame_len = FRAME_FIXED_LEN + block_len;

    let mut buf = Vec::new();
    buf.extend_from_slice(&frame_len.to_le_bytes());
    let mut fixed = [0u8; FRAME_FIXED_LEN as usize];
    // `uncompressed_len` stays 0, matching the zero-length node region.
    fixed[0] = FLAG_METADATA | FLAG_CRC_DISABLED;
    buf.extend_from_slice(&fixed);
    buf.push(METADATA_VERSION);
    buf.extend_from_slice(&tlv_len.to_le_bytes());
    buf.push(0x7f); // unknown tag, preserved verbatim
    buf.extend_from_slice(
        &u32::try_from(value_len)
            .expect("test value length fits u32")
            .to_le_bytes(),
    );
    buf.extend_from_slice(&vec![0xAB; value_len]);
    // Zero node bytes, then the (unused) 4-byte checksum field.
    buf.extend_from_slice(&[0u8; 4]);

    let mut cursor = Cursor::new(&buf);
    let meta = read_record_metadata(&mut cursor)
        .expect("boundary-size frame must not error")
        .expect("boundary-size frame must decode");
    assert_eq!(
        meta.metadata.expect("metadata present").unknown,
        vec![(0x7f, vec![0xAB; value_len])]
    );
}

/// Without the `zstd` feature, a frame flagged compressed must be rejected
/// with a clear error rather than silently misread as raw.
#[cfg(not(feature = "zstd"))]
#[test]
fn test_compressed_frame_rejected_without_zstd_feature() {
    let record = test_record_raw([0xaa; 16], b"payload");
    // Raw, checksum-less frame; flip the compressed flag on (the flags
    // byte sits right after the u32 frame length). Checksum off so the
    // flag flip doesn't trip the CRC check first.
    let mut buf = encode_record_with_options(&record, false, false).unwrap();
    buf[4] |= FLAG_COMPRESSED;

    let mut cursor = Cursor::new(&buf);
    let err = read_record(&mut cursor).unwrap_err();
    assert!(
        err.to_string().contains("without the `zstd` feature"),
        "unexpected error: {err}"
    );
}

/// Data that doesn't shrink under zstd (tiny/incompressible) must fall
/// back to being stored raw, not expanded on disk.
#[cfg(feature = "zstd")]
#[test]
fn test_write_record_falls_back_to_raw_when_incompressible() {
    let record = test_record_raw([0xaa; 16], b"hi");
    let mut buf = Vec::new();
    let written = write_record(&mut buf, &record).unwrap();
    assert_eq!(written, buf.len() as u64);
    assert_eq!(buf.len(), record.serialized_len());

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(record, read);
}

#[test]
fn test_multiple_records() {
    let records = vec![
        test_record_raw([0x01; 16], b"first"),
        test_record_raw([0x02; 16], b"second record"),
        test_record_raw([0x03; 16], &vec![0xff; 1024]),
    ];

    let mut buf = Vec::new();
    for r in &records {
        write_record(&mut buf, r).unwrap();
    }

    let mut cursor = Cursor::new(&buf);
    for expected in &records {
        let read = read_record(&mut cursor).unwrap().unwrap();
        assert_eq!(*expected, read);
    }
    assert!(read_record(&mut cursor).unwrap().is_none());
}

#[test]
fn test_crc_corruption_detected() {
    let record = test_record_raw([0xaa; 16], b"test data");
    let mut buf = Vec::new();
    write_record(&mut buf, &record).unwrap();

    // Flip a byte in the node payload (after the fixed frame header:
    // len(4) + flags(1) + uncompressed_len(4) + collection_id(16) + hash(16))
    buf[41] ^= 0xff;

    let mut cursor = Cursor::new(&buf);
    let result = read_record(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_encode_without_checksum_sets_flag_and_zeroes_checksum() {
    let record = test_record_raw([0xbb; 16], b"unverified payload");
    let buf = encode_record_with_options(&record, false, false).unwrap();

    // flags byte sits right after the u32 length prefix.
    let flags = buf[4];
    assert_eq!(flags & FLAG_CRC_DISABLED, FLAG_CRC_DISABLED);
    assert_eq!(flags & FLAG_COMPRESSED, 0);

    // The 4 trailing bytes are the checksum field — must be zero.
    let checksum = u32::from_le_bytes(buf[buf.len() - 4..].try_into().unwrap());
    assert_eq!(checksum, 0);
}

#[test]
fn test_read_skips_crc_for_crc_disabled_frames() {
    let record = test_record_raw([0xcc; 16], b"unverified payload");
    let mut buf = encode_record_with_options(&record, false, false).unwrap();

    // Corrupt the payload region of a CRC-disabled frame: the reader must
    // not compare anything (the checksum field is zero), so the record
    // still decodes — with the tampered bytes surfacing as data.
    let node_bytes_start = 4 + FRAME_FIXED_LEN as usize;
    let corrupt_at = node_bytes_start + 2;
    let mut expected = b"unverified payload".to_vec();
    expected[2] ^= 0xff;
    buf[corrupt_at] ^= 0xff;

    let mut cursor = Cursor::new(&buf);
    let got = read_record(&mut cursor).unwrap().expect("must decode");
    assert_eq!(got.data.as_ref(), expected.as_slice());
}

#[test]
fn test_read_still_verifies_when_only_the_checksum_flag_is_set_on_a_valid_frame() {
    // A frame written WITHOUT the disabled flag must keep failing reads
    // when its payload is still tampered with — verification must not be
    // accidentally skipped just because `FLAG_CRC_DISABLED` exists.
    let record = test_record_raw([0xdd; 16], b"verified payload");
    let mut buf = encode_record_with_options(&record, false, true).unwrap();
    buf[41] ^= 0xff;

    let mut cursor = Cursor::new(&buf);
    let result = read_record(&mut cursor);
    assert!(result.is_err(), "valid frames must still be verified");
}

#[test]
fn test_write_record_with_options_keeps_full_checksums() {
    // The public write path stays on Full policy: frames carry the flag
    // clear and a real CRC, which `read_record` accepts.
    let record = test_record_raw([0xee; 16], b"still verified");
    let mut buf = Vec::new();
    write_record_with_options(&mut buf, &record, false).unwrap();

    let mut cursor = Cursor::new(&buf);
    let got = read_record(&mut cursor).unwrap().expect("must decode");
    assert_eq!(got.data.as_ref(), b"still verified");
    assert_eq!(buf[4] & FLAG_CRC_DISABLED, 0);
}

#[test]
fn test_zero_length_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);

    let mut cursor = Cursor::new(&buf);
    let result = read_record(&mut cursor);
    assert!(result.is_err());
}

#[test]
fn test_header_roundtrip() {
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(42), 1).unwrap();
    assert_eq!(buf.len(), HEADER_LEN);

    let mut cursor = Cursor::new(&buf);
    let header = read_header(&mut cursor).unwrap().expect("valid header");
    assert_eq!(header.pack_id, test_pack_id(42));
    assert!(header.created_at > 0);
}

#[test]
fn test_creation_sequence_header_roundtrip() {
    let mut buf = Vec::new();
    write_header_with_created_at(&mut buf, &test_pack_id(42), 123, 17).unwrap();

    let header = read_header(&mut Cursor::new(&buf))
        .unwrap()
        .expect("valid sequenced header");
    assert_eq!(header.created_at, 123);
    assert_eq!(header.creation_seq, 17);
}

#[test]
fn extract_inherits_the_source_ordering_key() {
    let dir = test_dir("extract_key");
    let (src, dst) = (dir.join("src.pack"), dir.join("dst.pack"));
    let mut buf = Vec::new();
    write_header_with_created_at(&mut buf, &test_pack_id(1), 77, 5).unwrap();
    std::fs::write(&src, buf).unwrap();

    extract_packfile_collection(&src, &dst, &[1u8; 16], &test_pack_id(2)).unwrap();
    let header = read_header(&mut std::fs::File::open(&dst).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!((header.created_at, header.creation_seq), (77, 5));
}

#[test]
fn zero_pack_id_is_rejected_by_hex_parse() {
    // The all-zero address is reserved and never a valid identity.
    assert!(PackId::from_hex(&"0".repeat(PACK_ID_LEN * 2)).is_none());
    // A nonzero address of the same length still parses.
    assert!(PackId::from_hex(&"a".repeat(PACK_ID_LEN * 2)).is_some());
    // `random` never returns the reserved sentinel.
    assert!(!PackId::random().is_zero());
}

#[test]
fn zero_pack_id_is_rejected_by_write_header() {
    let mut buf = Vec::new();
    let err = write_header_with_creation_seq(&mut buf, &PackId([0u8; PACK_ID_LEN]), 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(buf.is_empty(), "no header bytes may be emitted");
}

#[test]
fn zero_pack_id_is_rejected_by_read_header() {
    // Hand-build a header whose pack id field is all zeros but whose CRC
    // is otherwise valid, so the rejection is specifically about identity,
    // not a CRC failure.
    let mut buf = [0u8; HEADER_LEN];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    let header_len = u32::try_from(HEADER_LEN).expect("HEADER_LEN fits u32");
    buf[5..9].copy_from_slice(&header_len.to_le_bytes());
    // pack id bytes 9..(9 + PACK_ID_LEN) stay zero.
    let crc = crc32fast::hash(&buf[..CRC_COVERED_LEN]);
    buf[CRC_COVERED_LEN..CRC_COVERED_LEN + 4].copy_from_slice(&crc.to_le_bytes());

    let err = read_header(&mut Cursor::new(&buf)).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn full_all_zero_filename_is_rejected_but_truncated_prefix_is_accepted() {
    // The full 32-hex all-zero address is reserved.
    assert!(
        PackId::parse_filename_prefix(&format!("pack_{}", "0".repeat(PACK_ID_LEN * 2))).is_none()
    );
    // A truncated all-zero prefix is still a legal lookup key: it can
    // belong to a nonzero address whose leading bits are zero.
    let (bytes, digits) =
        PackId::parse_filename_prefix("pack_0000000000000000").expect("truncated prefix ok");
    assert_eq!(digits, PACK_FILENAME_PREFIX_HEX);
    assert!(bytes.iter().all(|&b| b == 0));
}

#[test]
fn filename_prefix_rejects_more_than_full_address_width() {
    assert!(PackId::parse_filename_prefix("pack_0000000000000000_0000000000000001").is_some());
    assert!(PackId::parse_filename_prefix(
        "pack_0000000000000000_0000000000000001_0000000000000002"
    )
    .is_none());
}

#[test]
fn test_header_invalid_magic() {
    let mut buf = vec![0u8; HEADER_LEN];
    buf[0..4].copy_from_slice(b"BADC");

    let mut cursor = Cursor::new(&buf);
    assert!(read_header(&mut cursor).unwrap().is_none());
}

#[test]
fn test_header_empty_returns_none() {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    assert!(read_header(&mut cursor).unwrap().is_none());
}

#[test]
fn test_header_crc_mismatch_is_an_error_not_none() {
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(1), 1).unwrap();
    // Corrupt a byte inside the CRC-covered region (the pack_id
    // field) without touching magic/version/header_len — this must
    // surface as corruption, not as "not a packfile".
    buf[12] ^= 0xFF;

    let mut cursor = Cursor::new(&buf);
    let err = read_header(&mut cursor).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn test_record_serialized_len() {
    let r = Record {
        collection_id: [0u8; 16],
        hash: [0u8; 16],
        data: Bytes::from_static(b"hello"),
        metadata: None,
    };
    assert_eq!(r.serialized_len(), 4 + FRAME_FIXED_LEN as usize + 5 + 4);
}

#[test]
fn test_write_record_larger_than_historical_pdu_limit() {
    let r = Record {
        collection_id: [0u8; 16],
        hash: [0u8; 16],
        data: Bytes::from(vec![0u8; 64 * 1024 + 1]),
        metadata: None,
    };
    let mut buf = Vec::new();
    write_record(&mut buf, &r).unwrap();
}

/// A record larger than the historical PDU limit still round-trips through
/// the generic packfile codec. PDU-specific limits belong to the template.
#[test]
fn test_write_read_roundtrip_above_pdu_limit() {
    let data_len = 128 * 1024;
    // Pseudorandom, not zeros/repeats, so zstd can't shrink it below
    // the raw size — this must take the uncompressed-fallback path.
    let data: Vec<u8> = (0..data_len)
        .map(|i| {
            u64::try_from(i)
                .expect("range index fits in u64")
                .wrapping_mul(2_654_435_761)
                .to_le_bytes()[0]
        })
        .collect();
    let record = test_record_raw([0xaa; 16], &data);
    let mut buf = Vec::new();
    let written = write_record(&mut buf, &record).unwrap();
    assert_eq!(written, buf.len() as u64);

    let mut cursor = Cursor::new(&buf);
    let read = read_record(&mut cursor).unwrap().unwrap();
    assert_eq!(record, read);
}

#[test]
fn test_read_record_non_eof_io_error() {
    struct FailRead;
    impl Read for FailRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke"))
        }
    }
    let result = read_record(&mut FailRead);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
}

#[test]
fn test_read_header_non_eof_io_error() {
    struct FailRead;
    impl Read for FailRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "nope"))
        }
    }
    let result = read_header(&mut FailRead);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
}

#[test]
fn test_open_packfile_invalid_header() {
    let dir = test_dir("packfile_invalid_header");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    std::fs::write(&path, b"BADC\x02extra").unwrap();
    let result = open_packfile(&path, false, &test_pack_id(0));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
}

/// Direct test of the identity cross-check: a file whose *header* is
/// perfectly valid (right magic, version, CRC) but whose embedded
/// `pack_id` doesn't match what the caller expects (i.e. what the
/// filename says) must be rejected — this is the actual detection
/// claim, independent of any test fixture happening to already agree
/// with its own filename.
#[test]
fn test_open_packfile_rejects_identity_mismatch() {
    let dir = test_dir("packfile_identity_mismatch");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pack_0000000000000002.pack");

    // Header genuinely says pack_id 0 — a valid current-format header on its
    // own terms, just not what this filename claims.
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    std::fs::write(&path, &buf).unwrap();

    let err = open_packfile(&path, false, &test_pack_id(2)).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    let msg = err.to_string();
    assert!(
        msg.contains(&test_pack_id(0).to_string()) && msg.contains(&test_pack_id(2).to_string()),
        "error must name both the header's actual pack_id and the filename's expected one, got: {msg}"
    );

    // The identical header opened under a matching expectation must
    // succeed — the check is about the mismatch, not the file itself.
    let ok_path = dir.join("pack_0000000000000000.pack");
    std::fs::write(&ok_path, &buf).unwrap();
    open_packfile(&ok_path, false, &test_pack_id(0)).unwrap();
}

/// A recognizable unsupported-version file must be refused with a specific,
/// identifiable error — not silently treated as absent/empty data.
#[test]
fn test_open_packfile_refuses_unknown_version_with_specific_error() {
    let dir = test_dir("packfile_unknown_version_refused");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");

    // A valid file with an unsupported version: 5-byte header followed by a
    // real, well-formed record, not merely a corrupt/truncated input.
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.push(VERSION.wrapping_add(1));
    write_record(&mut buf, &test_record_raw([0x11; 16], b"old data")).unwrap();
    std::fs::write(&path, &buf).unwrap();

    let err = open_packfile(&path, false, &test_pack_id(0)).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("0x{:02x}", VERSION.wrapping_add(1)))
            && msg.to_lowercase().contains("reset or migrate"),
        "error must identify the offending version and advise resetting/migrating, got: {msg}"
    );

    // The same file must not be silently readable as "0 entries" via
    // the standalone scan helpers either — this is the actual "too
    // quiet" failure mode: a v1 store must error, not look empty.
    assert_eq!(
        scan_packfile(&path).unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(
        scan_and_recover_packfile(&path).unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
    assert_eq!(
        scan_packfile_from(&path, 0).unwrap_err().kind(),
        io::ErrorKind::Unsupported
    );
}

/// The same rejection must also hold for an unsupported-version file too
/// short to fill a `HEADER_LEN` buffer — version mismatch must be caught by
/// the 5-byte prefix check before ever attempting to read a full header.
#[test]
fn test_read_header_refuses_short_unknown_version_not_silently_empty() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.push(VERSION.wrapping_add(1));
    let mut cursor = Cursor::new(&buf);
    let err = read_header(&mut cursor).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Unsupported);
}

#[test]
fn test_read_version_magic_without_version_is_not_an_invalid_header() {
    let mut cursor = Cursor::new(MAGIC);
    assert_eq!(read_version(&mut cursor).unwrap(), None);
}

#[test]
fn test_scan_packfile_empty_file() {
    let dir = test_dir("scan_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    std::fs::write(&path, b"").unwrap();
    let entries = scan_packfile(&path).unwrap();
    assert_eq!(entries, vec![]);
}

#[test]
fn test_scan_packfile_torn_tail() {
    let dir = test_dir("scan_torn");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    // Write header + one valid record + a torn tail: fewer than 4
    // bytes, so read_record can't even complete reading the length
    // prefix and hits UnexpectedEof.
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record_raw([0xaa; 16], b"data")).unwrap();
    buf.extend_from_slice(&[0xff; 3]); // torn trailing bytes
    std::fs::write(&path, &buf).unwrap();
    let entries = scan_packfile(&path).unwrap();
    assert_eq!(entries.len(), 1);
}

#[test]
fn test_scan_records_iter_reports_torn_tail() {
    let dir = test_dir("records_iter_torn");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pack_0000000000000000.pack");
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record_raw([0xaa; 16], b"data")).unwrap();
    buf.extend_from_slice(&[0xff; 3]); // torn trailing bytes
    std::fs::write(&path, &buf).unwrap();

    let mut scanner = scan_records_iter(&path).unwrap();
    let first = scanner.next().unwrap().unwrap();
    assert_eq!(first.1.data.as_ref(), b"data");
    assert!(scanner.next().is_none());
    assert!(scanner.torn_tail(), "torn trailing bytes should be flagged");
}

#[test]
fn test_scan_records_iter_rejects_a_non_packfile() {
    let dir = test_dir("records_iter_not_a_pack");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pack_0000000000000000.pack");
    std::fs::write(&path, b"not an mdb pack").unwrap();
    let error = scan_records_iter(&path).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn test_scan_and_recover_truncates_torn_tail() {
    let dir = test_dir("recover_torn");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record_raw([0xaa; 16], b"good")).unwrap();
    let valid_len = buf.len();
    // Simulate a realistic torn tail: valid length prefix declaring a
    // full frame, but fewer payload bytes actually present (missing
    // tail of the node bytes and the CRC). This mimics a crash mid-write.
    let declared_frame_len = FRAME_FIXED_LEN + 4; // fixed header + 4 bytes of data
    buf.extend_from_slice(&declared_frame_len.to_le_bytes());
    buf.push(0); // flags: uncompressed
    buf.extend_from_slice(&4u32.to_le_bytes()); // uncompressed_len
    buf.extend_from_slice(&[0xdd; 16]); // collection_id
    buf.extend_from_slice(&[0xbb; 16]); // hash
    buf.extend_from_slice(&[0xcc; 2]); // partial data (short of declared len; CRC never written)
    std::fs::write(&path, &buf).unwrap();
    let entries = scan_and_recover_packfile(&path).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len as u64);
}

/// A partial length prefix is a torn frame, not clean EOF. Recovery must
/// remove it so subsequent appends start at a valid record boundary.
#[test]
fn test_scan_and_recover_truncates_partial_length_prefix() {
    let dir = test_dir("recover_partial_prefix");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pack_0000000000000000.pack");
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record_raw([0xaa; 16], b"good")).unwrap();
    let valid_len = buf.len();
    buf.extend_from_slice(&[0x12, 0x34, 0x56]);
    std::fs::write(&path, &buf).unwrap();

    let entries = scan_and_recover_packfile(&path).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), valid_len as u64);
}

#[test]
fn test_scan_and_recover_clean_file() {
    let dir = test_dir("recover_clean");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record_raw([0xbb; 16], b"ok")).unwrap();
    write_record(&mut buf, &test_record_raw([0xcc; 16], b"ok2")).unwrap();
    let expected_len = buf.len();
    std::fs::write(&path, &buf).unwrap();
    let entries = scan_and_recover_packfile(&path).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), expected_len as u64);
}

/// The counting recovery does what the collecting one does: at every
/// truncation point of a multi-record pack it finds the same records, keeps
/// the same length and reports the truncation the same way; a corrupt frame
/// is the same error and leaves the file alone.
#[test]
fn test_recover_packfile_matches_scan_and_recover_at_every_cut() {
    let dir = test_dir("recover_matches");
    std::fs::create_dir_all(&dir).unwrap();
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    let header_len = buf.len();
    for (index, byte) in [0xa1u8, 0xa2, 0xa3, 0xa4].into_iter().enumerate() {
        let payload = vec![byte; 7 + index * 40];
        write_record(&mut buf, &test_record_raw([byte; 16], &payload)).unwrap();
    }
    for cut in 0..=buf.len() {
        let collecting = dir.join("collecting.pack");
        let counting = dir.join("counting.pack");
        std::fs::write(&collecting, &buf[..cut]).unwrap();
        std::fs::write(&counting, &buf[..cut]).unwrap();
        let entries = scan_and_recover_packfile(&collecting).map(|entries| entries.len());
        let recovery = recover_packfile(&counting);
        match (&entries, &recovery) {
            (Ok(records), Ok(found)) => {
                assert_eq!(u64::try_from(*records).unwrap(), found.records, "cut {cut}");
                assert_eq!(
                    std::fs::metadata(&collecting).unwrap().len(),
                    std::fs::metadata(&counting).unwrap().len(),
                    "cut {cut}: the two left different lengths"
                );
                if cut >= header_len {
                    assert_eq!(
                        found.valid_len,
                        std::fs::metadata(&counting).unwrap().len(),
                        "cut {cut}"
                    );
                }
            }
            (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "cut {cut}"),
            _ => panic!("cut {cut}: {entries:?} vs {recovery:?}"),
        }
    }
    // A flipped payload byte is corruption: the same error, and no truncation.
    let mut corrupt = buf.clone();
    let flip = header_len + 60;
    corrupt[flip] ^= 0x40;
    let collecting = dir.join("collecting.pack");
    let counting = dir.join("counting.pack");
    std::fs::write(&collecting, &corrupt).unwrap();
    std::fs::write(&counting, &corrupt).unwrap();
    let a = scan_and_recover_packfile(&collecting).unwrap_err();
    let b = recover_packfile(&counting).unwrap_err();
    assert_eq!(a.kind(), b.kind());
    assert_eq!(
        std::fs::metadata(&counting).unwrap().len(),
        corrupt.len() as u64
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_scan_and_recover_empty_header() {
    let dir = test_dir("recover_noheader");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    std::fs::write(&path, b"").unwrap();
    let entries = scan_and_recover_packfile(&path).unwrap();
    assert_eq!(entries, vec![]);
}

#[test]
fn test_scan_packfile_returns_collection_id() {
    let dir = test_dir("scan_collection_id");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    let collection1 = [0x01; 16];
    let collection2 = [0x02; 16];
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(
        &mut buf,
        &test_record(collection1, [0xAA; 16], b"collection1 msg"),
    )
    .unwrap();
    write_record(
        &mut buf,
        &test_record(collection2, [0xBB; 16], b"collection2 msg"),
    )
    .unwrap();
    std::fs::write(&path, &buf).unwrap();
    let entries = scan_packfile(&path).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].0, collection1);
    assert_eq!(entries[0].1, [0xAA; 16]);
    assert_eq!(entries[1].0, collection2);
    assert_eq!(entries[1].1, [0xBB; 16]);
}

/// `scan_packfile_skip_payload` must agree with `scan_packfile` on every
/// `(collection_id, hash, offset)` triple, for both a compressed and a
/// raw-fallback frame, without reading the payload it seeks over.
#[test]
fn test_scan_packfile_skip_payload_matches_scan_packfile() {
    let dir = test_dir("scan_skip_payload");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    let collection1 = [0x01; 16];
    let collection2 = [0x02; 16];
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(
        &mut buf,
        &test_record(collection1, [0xAA; 16], b"short raw payload"),
    )
    .unwrap();
    write_record(
        &mut buf,
        &test_record(collection2, [0xBB; 16], &vec![0x42u8; 8192]),
    )
    .unwrap();
    std::fs::write(&path, &buf).unwrap();

    let full = scan_packfile(&path).unwrap();
    let skipped = scan_packfile_skip_payload(&path).unwrap();
    assert_eq!(full, skipped);
    assert_eq!(skipped.len(), 2);
}

#[test]
fn test_scan_packfile_skip_payload_empty_file() {
    let dir = test_dir("scan_skip_payload_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    std::fs::write(&path, b"").unwrap();
    let entries = scan_packfile_skip_payload(&path).unwrap();
    assert_eq!(entries, vec![]);
}

#[test]
fn test_scan_packfile_skip_payload_stops_at_truncated_crc() {
    let dir = test_dir("scan_skip_payload_truncated");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard_00.pack");
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(&mut buf, &test_record([0x01; 16], [0xAA; 16], b"complete")).unwrap();
    let complete_len = buf.len();
    write_record(&mut buf, &test_record([0x02; 16], [0xBB; 16], b"torn")).unwrap();
    buf.truncate(buf.len() - 2);
    std::fs::write(&path, &buf).unwrap();

    let entries = scan_packfile_skip_payload(&path).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].2, HEADER_LEN as u64);
    assert!(buf.len() > complete_len);
}

/// The metadata bound must also hold on the on-disk scan paths: a packfile
/// whose frame declares a ~4 GiB metadata block is reported as corrupt,
/// not used to size an allocation, by both the full and skip-payload scans.
#[test]
fn test_scan_rejects_oversized_frame_metadata() {
    let dir = test_dir("scan_oversized_metadata");
    let path = dir.join("shard_00.pack");

    let record = Record {
        collection_id: [0x01; 16],
        hash: [0xAA; 16],
        data: Bytes::from_static(b"payload"),
        metadata: Some(FrameMetadata {
            logical_id: Some([0x09; 32]),
            ..FrameMetadata::default()
        }),
    };
    let mut frame = encode_record_with_options(&record, false, false).unwrap();
    // The metadata block starts right after the fixed header; its
    // `tlv_len` field is the u32 following the version byte.
    let tlv_len_at = 4 + FRAME_FIXED_LEN as usize + 1;
    frame[tlv_len_at..tlv_len_at + 4].copy_from_slice(&(u32::MAX - 5).to_le_bytes());

    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    buf.extend_from_slice(&frame);
    std::fs::write(&path, &buf).unwrap();

    for err in [
        scan_packfile(&path).unwrap_err(),
        scan_packfile_skip_payload(&path).unwrap_err(),
    ] {
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("metadata"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn test_extract_packfile_collection_slices_target_collection_verbatim() {
    let dir = test_dir("extract_collection_verbatim");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("source.pack");
    let dst_path = dir.join("dest.pack");

    let collection_a = [0x11; 16];
    let collection_b = [0x22; 16];

    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0x100), 1).unwrap();

    let target_record_first = Record {
        collection_id: collection_a,
        hash: [0xA1; 16],
        data: Bytes::from_static(b"room A record 1"),
        metadata: Some(FrameMetadata {
            logical_id: Some([0xAA; 32]),
            content_digest: Some([0xD1; 32]),
            digest_algorithm: DigestAlgorithm::Blake3,
            role: Some(b"event".to_vec()),
            last_write_lsn: Some(0x99),
            unknown: vec![(0x7f, vec![0xDE, 0xAD])],
        }),
    };
    write_record(&mut buf, &target_record_first).unwrap();

    let other_collection_record = Record {
        collection_id: collection_b,
        hash: [0xB1; 16],
        data: Bytes::from_static(b"room B record 1"),
        metadata: Some(FrameMetadata {
            role: Some(b"state".to_vec()),
            ..FrameMetadata::default()
        }),
    };
    write_record(&mut buf, &other_collection_record).unwrap();

    let target_record_second = Record {
        collection_id: collection_a,
        hash: [0xA2; 16],
        data: Bytes::from_static(b" { \"whitespace\" : true } "),
        metadata: None,
    };
    write_record(&mut buf, &target_record_second).unwrap();

    std::fs::write(&src_path, &buf).unwrap();

    let stats =
        extract_packfile_collection(&src_path, &dst_path, &collection_a, &test_pack_id(0x200))
            .unwrap();
    assert_eq!(stats.frames_extracted, 2);
    assert_eq!(stats.frames_scanned, 3);
    assert!(!stats.torn_tail);
    assert!(stats.frame_bytes_written > 0);

    // Verify the destination packfile
    let mut scanner = scan_records_iter(&dst_path).unwrap();
    let first = scanner.next().unwrap().unwrap().1;
    assert_eq!(first.collection_id, collection_a);
    assert_eq!(first.hash, [0xA1; 16]);
    assert_eq!(first.data.as_ref(), b"room A record 1");
    assert_eq!(first.metadata, target_record_first.metadata);

    let second = scanner.next().unwrap().unwrap().1;
    assert_eq!(second.collection_id, collection_a);
    assert_eq!(second.hash, [0xA2; 16]);
    assert_eq!(second.data.as_ref(), b" { \"whitespace\" : true } ");
    assert_eq!(second.metadata, None);

    assert!(scanner.next().is_none());

    // Verify header pack_id in dest file
    let mut dst_file = File::open(&dst_path).unwrap();
    let header = read_header(&mut dst_file).unwrap().unwrap();
    assert_eq!(header.pack_id, test_pack_id(0x200));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_extract_packfile_collection_handles_torn_tail() {
    let dir = test_dir("extract_collection_torn");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("source.pack");
    let dst_path = dir.join("dest.pack");

    let collection_a = [0x11; 16];
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(
        &mut buf,
        &test_record(collection_a, [0x01; 16], b"valid record"),
    )
    .unwrap();
    buf.extend_from_slice(&[0xff; 3]); // torn trailing bytes (< 4 prefix bytes)
    std::fs::write(&src_path, &buf).unwrap();

    let stats =
        extract_packfile_collection(&src_path, &dst_path, &collection_a, &test_pack_id(1)).unwrap();
    assert_eq!(stats.frames_extracted, 1);
    assert_eq!(stats.frames_scanned, 1);
    assert!(stats.torn_tail);

    let mut scanner = scan_records_iter(&dst_path).unwrap();
    let record = scanner.next().unwrap().unwrap().1;
    assert_eq!(record.data.as_ref(), b"valid record");
    assert!(scanner.next().is_none());
    assert!(
        !scanner.torn_tail(),
        "extracted destination must have clean EOF"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_extract_packfile_collection_handles_torn_payload_body() {
    let dir = test_dir("extract_collection_torn_body");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("source.pack");
    let dst_path = dir.join("dest.pack");

    let collection_a = [0x11; 16];
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(
        &mut buf,
        &test_record(collection_a, [0x01; 16], b"valid record"),
    )
    .unwrap();
    // Valid frame len prefix, but truncated payload body
    buf.extend_from_slice(&(FRAME_FIXED_LEN + 10).to_le_bytes());
    buf.extend_from_slice(&[0x00; 5]); // only 5 bytes of payload written before crash
    std::fs::write(&src_path, &buf).unwrap();

    let stats =
        extract_packfile_collection(&src_path, &dst_path, &collection_a, &test_pack_id(1)).unwrap();
    assert_eq!(stats.frames_extracted, 1);
    assert_eq!(stats.frames_scanned, 1);
    assert!(stats.torn_tail);

    let mut scanner = scan_records_iter(&dst_path).unwrap();
    let record = scanner.next().unwrap().unwrap().1;
    assert_eq!(record.data.as_ref(), b"valid record");
    assert!(scanner.next().is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_extract_packfile_collection_rejects_corrupted_crc() {
    let dir = test_dir("extract_collection_corrupt");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("source.pack");
    let dst_path = dir.join("dest.pack");

    let collection_a = [0x11; 16];
    let mut buf = Vec::new();
    write_header_with_creation_seq(&mut buf, &test_pack_id(0), 1).unwrap();
    write_record(
        &mut buf,
        &test_record(collection_a, [0x01; 16], b"valid record"),
    )
    .unwrap();
    // Corrupt a byte in the frame
    let corrupt_idx = HEADER_LEN + 10;
    buf[corrupt_idx] ^= 0x55;
    std::fs::write(&src_path, &buf).unwrap();

    let err = extract_packfile_collection(&src_path, &dst_path, &collection_a, &test_pack_id(1))
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A corrupt length prefix far beyond the data actually present must fail
/// cleanly (and, past the shard ceiling, immediately) rather than allocate.
#[test]
fn test_read_record_corrupt_huge_prefix_fails_without_allocating() {
    for prefix in [u32::MAX - 5, MAX_FRAME_LEN] {
        let mut buf = prefix.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 64]);
        let err = read_record(&mut buf.as_slice()).unwrap_err();
        assert!(matches!(
            err.kind(),
            io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
        ));
    }
}

#[test]
fn test_write_header_rejects_zero_creation_sequence() {
    let mut buf = Vec::new();
    let err = write_header_with_creation_seq(&mut buf, &test_pack_id(1), 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    let err = write_header_with_created_at(&mut buf, &test_pack_id(1), 5, 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(buf.is_empty());
}

#[test]
fn test_open_packfile_never_creates_or_initializes() {
    let dir = std::env::temp_dir().join(format!("mtxdb_open_no_create_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let missing = dir.join("missing.pack");
    assert!(open_packfile(&missing, true, &test_pack_id(1)).is_err());
    assert!(!missing.exists());

    let empty = dir.join("empty.pack");
    std::fs::write(&empty, b"").unwrap();
    assert!(open_packfile(&empty, true, &test_pack_id(1)).is_err());
    assert_eq!(std::fs::metadata(&empty).unwrap().len(), 0);

    let zero_seq = dir.join("zero.pack");
    assert_eq!(
        open_packfile_with_creation_seq(&zero_seq, &test_pack_id(1), 0)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
