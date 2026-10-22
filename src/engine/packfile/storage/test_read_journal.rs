use std::time::Duration;

use super::*;

const COLLECTION: [u8; 16] = [0x42; 16];
const REUSED: [u8; 16] = [0x02; 16];

fn put(node: [u8; 16], payload: &[u8]) -> JournalMutation {
    JournalMutation::Put {
        collection_id: COLLECTION,
        node_id: node,
        payload: payload.to_vec(),
    }
}

fn temp_wal(label: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mtxdb_read_journal_{label}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir.join("wal.bin")
}

/// LSN 1 and LSN 2 (`stale-`), returning the length that keeps only LSN 1.
fn write_two_groups(wal: &std::path::Path) -> u64 {
    let (mut journal, _) = Journal::open(wal).unwrap();
    journal.append_group(&[put([0x01; 16], b"stable")]).unwrap();
    let first_len = fs::metadata(wal).unwrap().len();
    journal.append_group(&[put(REUSED, b"stale-")]).unwrap();
    first_len
}

/// A restart that drops LSN 2 and reissues it with the same length.
fn restart_with_reissued_lsn(wal: &std::path::Path, keep_len: u64) {
    fs::OpenOptions::new()
        .write(true)
        .open(wal)
        .unwrap()
        .set_len(keep_len)
        .unwrap();
    let (mut journal, _) = Journal::open(wal).unwrap();
    journal.append_group(&[put(REUSED, b"fresh-")]).unwrap();
}

fn value(overlay: &ReadJournal) -> Option<Vec<u8>> {
    overlay
        .puts
        .get(&COLLECTION)
        .and_then(|nodes| nodes.get(&REUSED))
        .map(|(bytes, _)| bytes.to_vec())
}

/// The quiet margin is what the fast path rests on: a change younger than it
/// is never trusted, so a coarse-granularity filesystem cannot let a rewrite
/// share the recorded change time and go unnoticed.
#[test]
fn a_stamp_is_quiet_only_once_older_than_the_margin() {
    let stamp_aged = |age: Duration| {
        let changed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .checked_sub(age)
            .unwrap();
        FileStamp {
            dev: 1,
            ino: 1,
            len: 1,
            ctime_secs: i64::try_from(changed.as_secs()).unwrap(),
            ctime_nanos: i64::from(changed.subsec_nanos()),
        }
    };
    assert!(!stamp_aged(Duration::ZERO).is_quiet());
    assert!(
        !stamp_aged(QUIET_AFTER / 2).is_quiet(),
        "half the margin is not enough"
    );
    assert!(
        !stamp_aged(QUIET_AFTER * 3 / 4).is_quiet(),
        "a change inside the margin could still share a timestamp tick"
    );
    assert!(stamp_aged(QUIET_AFTER + Duration::from_millis(100)).is_quiet());
}

/// A file that has been quiet longer than the timestamp granularity is
/// trusted on an unchanged fingerprint. Platforms without file identity
/// exercise the content-comparison fallback on every refresh.
#[test]
fn an_unchanged_quiet_file_is_not_re_read() {
    let wal = temp_wal("quiet_skip");
    write_two_groups(&wal);
    std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    assert!(matches!(overlay.refresh(true), Ok(ReadRefresh::Applied)));
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    for _ in 0..50 {
        assert!(matches!(overlay.refresh(true), Ok(ReadRefresh::Applied)));
    }
    #[cfg(unix)]
    assert_eq!(overlay.tail_reads, 0, "a quiet identity stamp skips reads");
    #[cfg(not(unix))]
    assert_eq!(overlay.tail_reads, 50, "fallback compares on every refresh");
    assert_eq!(overlay.tail_resets, 0);
}

/// Appending to the same segment reuses the held descriptor when the
/// overlay records each new consumed tail. Identity platforms reuse the
/// held descriptor; the fallback opens and compares the path each time.
#[test]
fn same_file_appends_reuse_the_tail_descriptor() {
    let wal = temp_wal("reuse_tail_descriptor");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    #[cfg(unix)]
    assert_eq!(overlay.tail_opens, 1);
    #[cfg(not(unix))]
    assert_eq!(overlay.tail_opens, 0);

    for (node, payload) in [([0x03; 16], &b"third"[..]), ([0x04; 16], &b"fourth"[..])] {
        let (mut journal, _) = Journal::open(&wal).unwrap();
        journal.append_group(&[put(node, payload)]).unwrap();
        drop(journal);
        overlay.refresh(true).unwrap();
    }

    #[cfg(unix)]
    assert_eq!(overlay.tail_opens, 1, "same-file appends reuse the WAL");
    #[cfg(not(unix))]
    assert_eq!(overlay.tail_reads, 2, "fallback compares both appends");
    assert_eq!(
        overlay
            .puts
            .get(&COLLECTION)
            .and_then(|nodes| nodes.get(&[0x04; 16]))
            .map(|(bytes, _)| bytes.to_vec()),
        Some(b"fourth".to_vec())
    );
}

/// A same-length rewrite of a quiet file changes its change time, so the
/// fingerprint no longer matches, the window is compared, and the overlay
/// is rebuilt. The file's length and inode are exactly what they were.
#[test]
fn a_same_length_rewrite_of_a_quiet_file_is_detected() {
    let wal = temp_wal("quiet_rewrite");
    let keep_len = write_two_groups(&wal);
    std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    let len_before = fs::metadata(&wal).unwrap().len();

    restart_with_reissued_lsn(&wal, keep_len);
    assert_eq!(fs::metadata(&wal).unwrap().len(), len_before);
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
    assert_eq!(overlay.tail_resets, 1);
}

/// Reclaim renames a changed replacement over the segment. Identity
/// platforms detect the new file before reading; the fallback detects its
/// changed tail by comparing content at the path.
#[test]
fn a_renamed_replacement_is_detected_by_identity_or_content() {
    let wal = temp_wal("rename_replace");
    let keep_len = write_two_groups(&wal);
    std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    let reads_before = overlay.tail_reads;

    let copy = wal.with_extension("copy");
    fs::copy(&wal, &copy).unwrap();
    restart_with_reissued_lsn(&copy, keep_len);
    fs::rename(&copy, &wal).unwrap();

    overlay.refresh(true).unwrap();
    assert_eq!(
        overlay.tail_resets, 1,
        "a replaced file must rebuild the overlay"
    );
    #[cfg(unix)]
    assert_eq!(
        overlay.tail_reads, reads_before,
        "identity catches replacement"
    );
    #[cfg(not(unix))]
    assert!(
        overlay.tail_reads > reads_before,
        "fallback compares replacement content"
    );
    assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
}

/// The real path: a stamp recorded right after a write is warm and not
/// trusted, then promoted after one verifying read once the file has gone
/// quiet. If the machine is so slow that the file was already quiet at the
/// first refresh, there is no warm state to observe and the test returns.
#[test]
fn a_freshly_recorded_stamp_is_verified_then_promoted_once_quiet() {
    let wal = temp_wal("fresh_then_quiet");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    #[cfg(unix)]
    if overlay.tail_quiet {
        eprintln!(
            "skipped: the file was already quiet at the first refresh, so there \
                 is no warm stamp to observe on this machine"
        );
        return;
    }
    std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));
    for _ in 0..20 {
        overlay.refresh(true).unwrap();
    }
    #[cfg(unix)]
    {
        assert_eq!(overlay.tail_reads, 1, "one verifying read, then trusted");
        assert!(overlay.tail_quiet);
    }
    #[cfg(not(unix))]
    assert_eq!(overlay.tail_reads, 20, "fallback compares on every refresh");
    assert_eq!(overlay.tail_resets, 0);
}

/// On a platform without inode numbers a replaced file cannot be told apart
/// by identity, and the persistent descriptor would keep naming the old
/// file. The overlay must then compare what is at the path now: a
/// same-length replacement on a new inode with different content is caught,
/// and no descriptor is held. `force_no_identity` stands in for such a
/// platform on unix.
#[test]
fn without_a_file_identity_a_replacement_is_caught_by_content() {
    let wal = temp_wal("no_identity_replace");
    let keep_len = write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.force_no_identity = true;
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    assert!(overlay.tail_stamp.is_none(), "no stamp without an identity");
    assert!(overlay.tail_file.is_none(), "no descriptor may be held");

    // Build the same-length replacement beside the segment and rename it
    // over the original, so the path now names a different inode.
    let replacement = wal.with_extension("replacement");
    fs::copy(&wal, &replacement).unwrap();
    restart_with_reissued_lsn(&replacement, keep_len);
    fs::rename(&replacement, &wal).unwrap();

    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
    assert_eq!(overlay.tail_resets, 1);
    // The counter only rises, so this also covers the first refresh: no
    // descriptor was opened to record a tail at any point.
    assert_eq!(
        overlay.tail_opens, 0,
        "no descriptor is opened to record a tail without an identity"
    );
}

/// Without a file identity an in-place same-length rewrite (same inode,
/// new content) is caught by the same fresh-handle comparison.
#[test]
fn without_a_file_identity_an_in_place_rewrite_is_caught_by_content() {
    let wal = temp_wal("no_identity_rewrite");
    let keep_len = write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.force_no_identity = true;
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    let len_before = fs::metadata(&wal).unwrap().len();

    restart_with_reissued_lsn(&wal, keep_len);
    assert_eq!(fs::metadata(&wal).unwrap().len(), len_before);
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
    assert_eq!(overlay.tail_resets, 1);
    assert_eq!(overlay.tail_opens, 0);
}

/// An identity that disappears between recording and refresh (a stamp was
/// recorded, none is available now) means the file is not known to be the
/// one the overlay was built from, so the overlay is rebuilt.
#[test]
#[cfg(unix)]
fn a_stamp_that_vanishes_between_refreshes_forces_a_rebuild() {
    let wal = temp_wal("identity_flip");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    assert!(overlay.tail_stamp.is_some());
    overlay.force_no_identity = true;
    overlay.refresh(true).unwrap();
    assert_eq!(overlay.tail_resets, 1);
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    assert!(
        overlay.tail_stamp.is_none(),
        "the rebuild records without an identity"
    );
    assert!(overlay.tail_file.is_none());
}

/// The reverse flip: a segment recorded without an identity that reports
/// one on a later refresh is not known to be the file the overlay was built
/// from, so it is rebuilt and recorded with the identity.
#[test]
#[cfg(unix)]
fn an_identity_that_appears_between_refreshes_forces_a_rebuild() {
    let wal = temp_wal("identity_appears");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.force_no_identity = true;
    overlay.refresh(true).unwrap();
    assert!(overlay.tail_stamp.is_none());
    overlay.force_no_identity = false;
    overlay.refresh(true).unwrap();
    assert_eq!(overlay.tail_resets, 1);
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
    assert!(
        overlay.tail_stamp.is_some(),
        "the rebuild records the identity now available"
    );
    assert!(overlay.tail_file.is_some());
}

/// Without a file identity an unchanged segment is not rebuilt: the window
/// is compared through a fresh handle each refresh and matches.
#[test]
fn without_a_file_identity_an_unchanged_segment_is_not_rebuilt() {
    let wal = temp_wal("no_identity_unchanged");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.force_no_identity = true;
    overlay.refresh(true).unwrap();
    for _ in 0..10 {
        overlay.refresh(true).unwrap();
    }
    assert_eq!(overlay.tail_resets, 0);
    assert_eq!(overlay.tail_reads, 10, "one comparison per refresh");
    assert_eq!(
        overlay.tail_opens, 0,
        "no descriptor is retained; fresh compare opens are not counted"
    );
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
}

/// State-model test, not a real recording: the stamp is flipped to
/// untrusted by hand after a genuine quiet period, to check the promotion
/// rule in isolation from timing. The fallback path instead keeps
/// comparing the unchanged tail. The real path is covered by
/// `a_freshly_recorded_stamp_is_verified_then_promoted_once_quiet`.
#[test]
fn a_warm_fingerprint_is_verified_then_promoted_once_quiet() {
    let wal = temp_wal("warm_then_quiet");
    write_two_groups(&wal);
    std::thread::sleep(QUIET_AFTER + Duration::from_millis(60));

    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    #[cfg(unix)]
    {
        // Model a stamp recorded while the file was still warm.
        overlay.tail_quiet = false;
    }
    for _ in 0..20 {
        overlay.refresh(true).unwrap();
    }
    #[cfg(unix)]
    {
        assert_eq!(overlay.tail_reads, 1, "one verifying read, then trusted");
        assert!(overlay.tail_quiet);
    }
    #[cfg(not(unix))]
    assert_eq!(overlay.tail_reads, 20, "fallback compares on every refresh");
    assert_eq!(overlay.tail_resets, 0);
}

/// A stamp that is not yet trusted must keep validating the tail: a
/// same-length rewrite is caught even though length and inode match.
#[test]
fn an_untrusted_fingerprint_still_validates_the_tail() {
    let wal = temp_wal("untrusted_rewrite");
    let keep_len = write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    overlay.tail_quiet = false;

    restart_with_reissued_lsn(&wal, keep_len);
    overlay.refresh(true).unwrap();
    assert_eq!(value(&overlay), Some(b"fresh-".to_vec()));
    assert_eq!(overlay.tail_resets, 1);
}

/// State-model test: a stamp recorded earlier but unavailable now cannot
/// be compared, so it fails closed and rebuilds. Only platforms that
/// report inode numbers and change times record a stamp at all.
#[cfg(unix)]
#[test]
fn a_stamp_that_cannot_be_compared_fails_closed() {
    let wal = temp_wal("stamp_mismatch");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    assert!(
        overlay.tail_stamp.is_some(),
        "a unix platform must record a stamp"
    );
    overlay.tail_stamp = None;
    overlay.refresh(true).unwrap();
    assert_eq!(overlay.tail_resets, 1);
}

/// The remembered window is composed from consumed bytes across refreshes
/// (the previous window plus what each scan consumed). The previous window
/// ends exactly at the scan's start offset, `observed_valid_len`, and the
/// scan's bytes begin there, so the two are contiguous. The composed window
/// must always equal the window a later comparison reads from disk,
/// including once the file outgrows the window, which is where the join and
/// the trim in `record_observed_tail` matter.
#[test]
fn the_composed_window_matches_the_disk_after_every_refresh() {
    let wal = temp_wal("composed_window");
    let (mut journal, _) = Journal::open(&wal).unwrap();
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    let file = File::open(&wal).unwrap();
    for round in 0..12u8 {
        journal
            .append_group(&[put([round; 16], &[round; 300])])
            .unwrap();
        overlay.refresh(true).unwrap();
        let mut on_disk = Vec::new();
        assert!(Journal::read_tail_ending_at(
            &file,
            overlay.observed_valid_len,
            TAIL_WINDOW,
            &mut on_disk
        )
        .unwrap());
        assert_eq!(
            overlay.observed_tail.as_deref(),
            Some(on_disk.as_slice()),
            "round {round}: the remembered window must equal the disk"
        );
    }
    assert_eq!(overlay.tail_resets, 0);
}

/// If groups were applied but their tail could not be recorded, nothing
/// protects the overlay, so the next refresh must rebuild it and try again
/// instead of trusting the prefix.
#[test]
fn an_unrecorded_tail_forces_a_rebuild() {
    let wal = temp_wal("unrecorded_tail");
    write_two_groups(&wal);
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    overlay.refresh(true).unwrap();
    assert!(overlay.observed_tail.is_some());

    overlay.observed_tail = None;
    overlay.refresh(true).unwrap();
    assert_eq!(overlay.tail_resets, 1);
    assert!(
        overlay.observed_tail.is_some(),
        "the rebuild records the tail again"
    );
    assert_eq!(value(&overlay), Some(b"stale-".to_vec()));
}

/// An empty segment has no consumed bytes to protect; refreshing it must
/// not rebuild forever.
#[test]
fn an_empty_segment_does_not_rebuild_on_every_refresh() {
    let wal = temp_wal("empty_segment");
    drop(Journal::open(&wal).unwrap());
    let mut overlay = ReadJournal::empty(wal.clone(), 0, None);
    for _ in 0..5 {
        overlay.refresh(true).unwrap();
    }
    assert_eq!(overlay.tail_resets, 0);
}
