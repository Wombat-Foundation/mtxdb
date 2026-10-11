//! Synthetic-data tests for the prototype's id -> key chunk format.

#[path = "../edge_layout_format.rs"]
mod format;

#[cfg(test)]
mod tests {
    use super::format::{
        chunk_bounds, encode_key_chunk, key_at, locate, FormatError, KEY_HEADER, MAX_CHUNK, MAX_KEY,
    };

    #[test]
    fn locate_is_correct_around_chunk_boundaries() {
        for size in [16_usize, 32, 64, 128, 256] {
            let cases = [
                (0, (0, 0)),
                (size - 1, (0, size - 1)),
                (size, (1, 0)),
                (size * 2 - 1, (1, size - 1)),
                (size * 2, (2, 0)),
            ];
            for (id, want) in cases {
                assert_eq!(locate(id, size), want, "id {id} at chunk size {size}");
            }
        }
    }

    #[test]
    fn chunk_bounds_clip_the_tail_chunk() {
        assert_eq!(chunk_bounds(0, 32, 70), (0, 32));
        assert_eq!(chunk_bounds(2, 32, 70), (64, 70));
        assert_eq!(chunk_bounds(1, 32, 70), (32, 64));
    }

    #[test]
    fn round_trips_first_last_and_partial_chunks() {
        let keys: Vec<String> = (0..5).map(|i| format!("${}", "k".repeat(i * 7))).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let bytes = encode_key_chunk(&refs, 9).expect("encode");
        for (pos, key) in refs.iter().enumerate() {
            assert_eq!(key_at(&bytes, pos), Ok(key.as_bytes()), "position {pos}");
        }
        assert_eq!(key_at(&bytes, 5), Err(FormatError::PositionOutOfRange));
        let empty = encode_key_chunk(&[""], 0).expect("empty key");
        assert_eq!(key_at(&empty, 0), Ok(&b""[..]));
    }

    #[test]
    fn the_maximum_chunk_fits_u16_offsets_and_decodes() {
        let keys: Vec<String> = (0..MAX_CHUNK)
            .map(|i| format!("{i:03}{}", "x".repeat(MAX_KEY - 3)))
            .collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let bytes = encode_key_chunk(&refs, 1).expect("encode the maximum chunk");
        let last = KEY_HEADER + (MAX_CHUNK - 1) * 2;
        let last_offset = u16::from_be_bytes([bytes[last], bytes[last + 1]]);
        assert_eq!(usize::from(last_offset), (MAX_CHUNK - 1) * (MAX_KEY + 1));
        for (pos, key) in refs.iter().enumerate() {
            assert_eq!(key_at(&bytes, pos), Ok(key.as_bytes()), "position {pos}");
        }
    }

    #[test]
    fn encoding_rejects_oversized_input() {
        let many = vec!["a"; MAX_CHUNK + 1];
        assert_eq!(encode_key_chunk(&many, 0), Err(FormatError::TooManyKeys));
        let long = "x".repeat(MAX_KEY + 1);
        assert_eq!(encode_key_chunk(&[&long], 0), Err(FormatError::KeyTooLong));
    }

    #[test]
    fn malformed_chunks_return_errors_without_panicking() {
        let good = encode_key_chunk(&["alpha", "beta"], 3).expect("encode");
        let first_entry_end = KEY_HEADER + 2 * 2 + 1 + "alpha".len();
        for cut in 0..first_entry_end {
            assert!(key_at(&good[..cut], 0).is_err(), "truncated at {cut}");
        }
        let mut magic = good.clone();
        magic[0] = b'X';
        assert_eq!(key_at(&magic, 0), Err(FormatError::BadMagic));
        let mut count = good.clone();
        count[8..10].copy_from_slice(&u16::try_from(MAX_CHUNK + 1).unwrap().to_be_bytes());
        assert_eq!(key_at(&count, 0), Err(FormatError::BadCount));
        let mut lying = good.clone();
        lying[8..10].copy_from_slice(&200_u16.to_be_bytes());
        assert_eq!(key_at(&lying, 150), Err(FormatError::Truncated));
        let mut far = good.clone();
        far[KEY_HEADER..KEY_HEADER + 2].copy_from_slice(&u16::MAX.to_be_bytes());
        assert_eq!(key_at(&far, 0), Err(FormatError::Truncated));
        assert_eq!(key_at(&good, 2), Err(FormatError::PositionOutOfRange));
    }
}
