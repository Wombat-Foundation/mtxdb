//! Synthetic-data tests for the prototype's id -> key chunk format.

#[path = "../edge_layout_format.rs"]
mod format;

#[cfg(test)]
mod tests {
    use super::format::{
        chunk_bounds, confirm, encode_key_chunk, encode_run, key_at, locate, merged_candidates,
        run_candidates, FormatError, KeyHash, LookupError, KEY_HEADER, MAX_CHUNK, MAX_KEY,
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

    /// Injected hash: every key starting with `$c` collides; others are distinct.
    fn hash(key: &str) -> KeyHash {
        if key.starts_with("$c") {
            [7; 16]
        } else {
            let mut h = [0_u8; 16];
            for (slot, byte) in h.iter_mut().zip(key.bytes()) {
                *slot = byte;
            }
            h
        }
    }

    /// Ids are positions in `keys`; key chunks hold four ids; runs are the
    /// batches the keys arrived in.
    struct Index {
        chunks: Vec<Option<Vec<u8>>>,
        runs: Vec<Vec<u8>>,
    }

    impl Index {
        fn build(batches: &[&[&str]]) -> Self {
            let keys: Vec<String> = batches
                .iter()
                .flat_map(|b| b.iter())
                .map(|k| (*k).to_owned())
                .collect();
            let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
            let chunks = refs
                .chunks(4)
                .enumerate()
                .map(|(n, c)| Some(encode_key_chunk(c, u32::try_from(n).unwrap()).unwrap()))
                .collect();
            let mut next = 0_u32;
            let runs = batches
                .iter()
                .map(|batch| {
                    let mut entries: Vec<(KeyHash, u32)> = batch
                        .iter()
                        .map(|k| {
                            let id = next;
                            next = next.saturating_add(1);
                            (hash(k), id)
                        })
                        .collect();
                    let mut out = encode_run(&mut entries);
                    out.drain(..8);
                    out
                })
                .collect();
            Self { chunks, runs }
        }

        fn candidates(&self, key: &str) -> Vec<u32> {
            self.runs
                .iter()
                .flat_map(|r| run_candidates(r, &hash(key)))
                .collect()
        }

        fn key_of(&self, id: u32) -> Option<String> {
            let (chunk, pos) = locate(id as usize, 4);
            let record = self.chunks.get(chunk)?.as_ref()?;
            key_at(record, pos)
                .ok()
                .map(|k| String::from_utf8_lossy(k).into_owned())
        }

        fn lookup(&self, key: &str) -> Result<Option<u32>, LookupError> {
            confirm(&self.candidates(key), key, &mut |id| self.key_of(id))
        }
    }

    #[test]
    fn colliding_keys_in_one_run_resolve_to_their_own_ids() {
        let index = Index::build(&[&["$a", "$c1", "$c2", "$b"]]);
        assert_eq!(index.candidates("$c1"), vec![1, 2]);
        assert_eq!(index.lookup("$c1"), Ok(Some(1)));
        assert_eq!(index.lookup("$c2"), Ok(Some(2)));
        assert_eq!(index.lookup("$a"), Ok(Some(0)));
    }

    #[test]
    fn colliding_keys_in_different_runs_both_resolve() {
        let index = Index::build(&[&["$c1", "$a"], &["$b", "$c2"], &["$c3"]]);
        assert_eq!(index.candidates("$c2").len(), 3);
        for (key, id) in [("$c1", 0), ("$c2", 3), ("$c3", 4)] {
            assert_eq!(index.lookup(key), Ok(Some(id)), "{key}");
        }
    }

    #[test]
    fn an_uninserted_key_in_a_full_bucket_is_a_miss_after_checking_every_candidate() {
        let index = Index::build(&[&["$c1", "$c2"], &["$c3"]]);
        assert_eq!(index.candidates("$cmissing").len(), 3);
        assert_eq!(index.lookup("$cmissing"), Ok(None));
    }

    #[test]
    fn a_hash_absent_from_every_run_is_a_miss() {
        let index = Index::build(&[&["$c1", "$a"]]);
        assert!(index.candidates("$zzz").is_empty());
        assert_eq!(index.lookup("$zzz"), Ok(None));
    }

    #[test]
    fn merging_the_runs_gives_the_same_candidates() {
        let index = Index::build(&[&["$c1", "$a"], &["$b", "$c2"]]);
        let mut merged: Vec<(KeyHash, u32)> = Vec::new();
        for run in &index.runs {
            for entry in run.chunks_exact(20) {
                let id = u32::from_be_bytes(entry[16..20].try_into().unwrap());
                merged.push((entry[..16].try_into().unwrap(), id));
            }
        }
        merged.sort_unstable();
        for key in ["$c1", "$c2", "$a", "$b", "$zzz"] {
            let mut want = index.candidates(key);
            want.sort_unstable();
            assert_eq!(merged_candidates(&merged, &hash(key)), want, "{key}");
        }
    }

    #[test]
    fn an_unreadable_candidate_is_an_error_never_a_miss_or_another_keys_id() {
        let mut index = Index::build(&[&["$c1", "$c2", "$c3", "$c4", "$c5"]]);
        // Lose the chunk holding ids 4..: "$c5" cannot be ruled out.
        index.chunks[1] = None;
        assert_eq!(index.lookup("$c1"), Err(LookupError::KeyUnreadable(4)));
        assert_eq!(
            index.lookup("$cmissing"),
            Err(LookupError::KeyUnreadable(4))
        );
        // A corrupt chunk fails the same way.
        index.chunks[1] = Some(vec![0; 3]);
        assert_eq!(index.lookup("$c1"), Err(LookupError::KeyUnreadable(4)));
    }

    #[test]
    fn the_same_full_key_under_two_ids_is_corruption() {
        let index = Index::build(&[&["$c1", "$c2"], &["$c1"]]);
        assert_eq!(index.lookup("$c1"), Err(LookupError::DuplicateKey));
        assert_eq!(index.lookup("$c2"), Ok(Some(1)));
    }
}
