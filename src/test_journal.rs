use super::{
    BackgroundFailure, BackgroundState, GroupCommitConfig, Journal, JournalCoordinator, Mutation,
};
#[cfg(feature = "multi-reader")]
use super::{TxnStage, TxnStageState};
use std::fs;
use std::io::Write as _;
#[cfg(feature = "multi-reader")]
use std::path::Path;
#[cfg(feature = "multi-reader")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

fn temp_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mtxdb_journal_{label}_{}_{}",
        std::process::id(),
        std::thread::current()
            .name()
            .unwrap_or("test")
            .replace(':', "_")
    ))
}

#[test]
fn read_only_scan_treats_missing_and_short_segments_as_empty() {
    let missing = temp_path("read_only_missing");
    let _ = fs::remove_file(&missing);
    let scan = Journal::scan_read_only(&missing).unwrap();
    assert_eq!(scan.groups.len(), 0);
    assert_eq!(scan.valid_len, 0);
    assert!(!scan.truncated_tail);
    assert!(
        !missing.exists(),
        "read-only scan must not create a segment"
    );

    let short = temp_path("read_only_short");
    fs::write(&short, b"MTXWAL").unwrap();
    let before = fs::read(&short).unwrap();
    let scan = Journal::scan_read_only(&short).unwrap();
    assert_eq!(scan.groups.len(), 0);
    assert_eq!(scan.valid_len, 0);
    assert!(!scan.truncated_tail);
    assert_eq!(
        fs::read(&short).unwrap(),
        before,
        "scan must not repair bytes"
    );
    fs::remove_file(short).unwrap();
}

fn put(collection: u8, node: u8, payload: &[u8]) -> Mutation {
    Mutation::Put {
        collection_id: [collection; 16],
        node_id: [node; 16],
        payload: payload.to_vec(),
    }
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_segment_round_trips_pool_tags() {
    use crate::layout::ShardType;
    let path = temp_path("shared_pool_tags");
    let _ = fs::remove_file(&path);

    let (mut journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 0);
    let tagged = [
        (Some(ShardType::State), put(1, 1, b"state")),
        (Some(ShardType::EventDag), put(2, 2, b"event")),
        (Some(ShardType::Edges), put(3, 3, b"edges")),
    ];
    journal
        .append_group_tagged_with_sequence(&tagged, None)
        .unwrap();
    journal.make_durable().unwrap();

    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    let pools: Vec<_> = scan.groups[0]
        .entries
        .iter()
        .map(|entry| entry.pool)
        .collect();
    assert_eq!(
        pools,
        vec![
            Some(ShardType::State),
            Some(ShardType::EventDag),
            Some(ShardType::Edges),
        ]
    );

    // Recovery must preserve the tags so each frame can be routed.
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups[0].entries[1].pool, Some(ShardType::EventDag));
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_segment_rejects_untagged_frames() {
    let path = temp_path("shared_untagged");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open_shared(&path).unwrap();
    assert!(
        journal.append_group(&[put(1, 1, b"x")]).is_err(),
        "a pool-tagged segment must reject an untagged frame"
    );
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_committed_watermark_is_per_pool() {
    use crate::layout::ShardType;
    let path = temp_path("shared_pool_watermark");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    // Group one: state only. Only state's watermark may advance past zero.
    coordinator
        .publish_group_tagged(ShardType::State, &[put(1, 1, b"state")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    let state_only = coordinator.committed_lsn();
    assert_eq!(
        coordinator.committed_lsn_for_pool(ShardType::State),
        state_only,
        "the state frame's group advances state's watermark"
    );
    assert_eq!(
        coordinator.committed_lsn_for_pool(ShardType::EventDag),
        0,
        "another pool's group must not advance event-DAG's watermark"
    );
    assert_eq!(coordinator.committed_lsn_for_pool(ShardType::Edges), 0);

    // Group two: event-DAG only. State's watermark must stay put.
    coordinator
        .publish_group_tagged(ShardType::EventDag, &[put(2, 2, b"event")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    assert_eq!(
        coordinator.committed_lsn_for_pool(ShardType::State),
        state_only
    );
    assert!(coordinator.committed_lsn_for_pool(ShardType::EventDag) > state_only);

    // A group carrying both pools advances both to the group's own end, once
    // its writes are in the packs. Until then a transaction group holds the
    // watermarks below its first frame: the packs do not have it yet.
    let receipt = coordinator
        .publish_tagged_groups(&[
            (ShardType::State, &[put(3, 3, b"s2")]),
            (ShardType::EventDag, &[put(4, 4, b"e2")]),
        ])
        .unwrap();
    coordinator.sync().unwrap();
    assert!(coordinator.committed_lsn_for_pool(ShardType::State) < receipt.first_lsn);
    assert!(coordinator.committed_lsn_for_pool(ShardType::EventDag) < receipt.first_lsn);
    coordinator.transaction_materialized(receipt.first_lsn);
    let shared = coordinator.committed_lsn();
    assert_eq!(coordinator.committed_lsn_for_pool(ShardType::State), shared);
    assert_eq!(
        coordinator.committed_lsn_for_pool(ShardType::EventDag),
        shared
    );
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_reclaim_waits_for_a_staged_pools_frame() {
    use crate::layout::ShardType;
    let path = temp_path("shared_staged_required");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    // A transaction stages an event-DAG frame without going through
    // `publish`; it must still block reclaim until event-DAG checkpoints.
    coordinator
        .publish_group_tagged(ShardType::EventDag, &[put(2, 2, b"event")])
        .unwrap();
    coordinator
        .publish_group_tagged(ShardType::State, &[put(1, 1, b"state")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    let committed = coordinator.committed_lsn();
    assert!(coordinator.committed_lsn_for_pool(ShardType::EventDag) > 0);

    // State checkpoints, event-DAG does not: the staged frame's group must
    // survive, because event-DAG has no durable coverage for it yet.
    coordinator.report_pool_coverage(ShardType::State, committed);
    assert!(
        coordinator.reclaim_shared().unwrap().is_none(),
        "a staged pool without coverage must block reclaim"
    );

    // Once every pool reports, the prefix can go.
    coordinator.report_pool_coverage(ShardType::EventDag, committed);
    assert!(coordinator.reclaim_shared().unwrap().is_some());
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_reclaim_does_not_wait_for_an_idle_pool() {
    use crate::layout::ShardType;
    let path = temp_path("shared_idle_pool_reclaim");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    // State contributes only the first group, then goes idle.
    coordinator
        .publish_group_tagged(ShardType::State, &[put(1, 1, b"state")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    let state_lsn = coordinator.committed_lsn_for_pool(ShardType::State);

    // Event-DAG writes several later groups that never carry a state frame.
    let mut event_lsn = 0;
    for id in 0..4u8 {
        coordinator
            .publish_group_tagged(ShardType::EventDag, &[put(2, id, b"event")])
            .map(|receipt| receipt.last_lsn)
            .unwrap();
        coordinator.sync().unwrap();
        event_lsn = coordinator.committed_lsn_for_pool(ShardType::EventDag);
    }
    assert!(event_lsn > state_lsn);

    // State covers only its own group. Reclaim must advance through that
    // group and stop at the first uncovered event-DAG group — not refuse to
    // advance at all because some pool is behind.
    coordinator.report_pool_coverage(ShardType::State, state_lsn);
    assert!(coordinator.reclaim_shared().unwrap().is_some());
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 4, "uncovered event groups must survive");
    assert_eq!(
        scan.base_lsn,
        state_lsn + 1,
        "the covered state group is reclaimed independently"
    );

    // Once event-DAG covers its frames, the idle state pool must not pin
    // the later groups its frames never reached.
    coordinator.report_pool_coverage(ShardType::EventDag, event_lsn);
    assert!(coordinator.reclaim_shared().unwrap().is_some());
    let scan = Journal::scan_read_only(&path).unwrap();
    assert!(
        scan.groups.is_empty(),
        "an idle pool must not block later groups it never contributed to"
    );
    assert_eq!(scan.base_lsn, event_lsn + 1);
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn per_pool_segment_rejects_tagged_frames() {
    use crate::layout::ShardType;
    let path = temp_path("perpool_tagged");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    let tagged = [(Some(ShardType::State), put(1, 1, b"x"))];
    assert!(
        journal
            .append_group_tagged_with_sequence(&tagged, None)
            .is_err(),
        "a per-pool segment must reject a tagged frame"
    );
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn open_mode_must_match_segment_version() {
    let path = temp_path("version_mismatch");
    let _ = fs::remove_file(&path);
    Journal::open(&path).unwrap();
    assert!(
        Journal::open_shared(&path).is_err(),
        "a per-pool segment must not open as shared"
    );
    fs::remove_file(&path).unwrap();

    Journal::open_shared(&path).unwrap();
    assert!(
        Journal::open(&path).is_err(),
        "a shared segment must not open as per-pool"
    );
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn coordinator_publishes_tagged_frames() {
    use crate::layout::ShardType;
    let path = temp_path("coordinator_tagged");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    coordinator
        .publish_tagged_groups(&[
            (ShardType::State, &[put(1, 1, b"a")]),
            (ShardType::Edges, &[put(3, 3, b"c")]),
        ])
        .unwrap();
    coordinator.sync().unwrap();

    let scan = Journal::scan_read_only(&path).unwrap();
    let pools: Vec<_> = scan.groups[0]
        .entries
        .iter()
        .map(|entry| entry.pool)
        .collect();
    assert_eq!(pools, vec![Some(ShardType::State), Some(ShardType::Edges)]);
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn transaction_stage_publishes_all_pools_as_one_shared_group() {
    use crate::layout::ShardType;

    let path = temp_path("txn_stage_shared_group");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    let stage = TxnStage::new();
    stage
        .stage_put(ShardType::State, [1; 16], [1; 16], b"state".to_vec())
        .unwrap();
    stage
        .stage_put(ShardType::EventDag, [2; 16], [2; 16], b"event".to_vec())
        .unwrap();
    stage
        .stage_put(ShardType::Edges, [3; 16], [3; 16], b"edge".to_vec())
        .unwrap();

    stage
        .publish(Some(&coordinator), Some(&coordinator), Some(&coordinator))
        .unwrap();
    stage
        .publish(Some(&coordinator), Some(&coordinator), Some(&coordinator))
        .unwrap();

    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].entries.len(), 3);
    assert_eq!(
        scan.groups[0]
            .entries
            .iter()
            .map(|entry| entry.pool)
            .collect::<Vec<_>>(),
        vec![
            Some(ShardType::Edges),
            Some(ShardType::EventDag),
            Some(ShardType::State)
        ]
    );
    assert_eq!(stage.state(), TxnStageState::JournalPublished);
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn transaction_stage_publishes_after_autocommit_groups_in_lsn_order() {
    use crate::layout::ShardType;

    let path = temp_path("txn_stage_after_autocommit");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    let autocommit = coordinator
        .publish_group_tagged(ShardType::State, &[put(9, 9, b"autocommit")])
        .unwrap();
    assert_eq!((autocommit.first_lsn, autocommit.last_lsn), (1, 1));

    let stage = TxnStage::new();
    stage
        .stage_put(
            ShardType::EventDag,
            [2; 16],
            [2; 16],
            b"transaction".to_vec(),
        )
        .unwrap();
    stage
        .publish(Some(&coordinator), Some(&coordinator), Some(&coordinator))
        .unwrap();

    // The autocommit write keeps its own complete group ahead of the
    // transaction's group; the transaction never absorbs it.
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 2);
    assert_eq!(scan.groups[0].entries.len(), 1);
    assert_eq!(scan.groups[0].entries[0].pool, Some(ShardType::State));
    assert_eq!(scan.groups[1].entries.len(), 1);
    assert_eq!(scan.groups[1].entries[0].pool, Some(ShardType::EventDag));
    assert_eq!(coordinator.visible_lsn(), 2);
    fs::remove_file(path).unwrap();
}

#[test]
fn append_group_is_visible_before_make_durable() {
    let path = temp_path("append_visible");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    let receipt = journal.append_group(&[put(1, 1, b"pending")]).unwrap();
    assert_eq!((receipt.first_lsn, receipt.last_lsn), (1, 1));

    // The complete group, trailer included, is readable by a read-only
    // scanner before any fsync: this is the committed-but-unflushed state
    // the read overlay observes.
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].last_lsn, 1);
    assert!(!scan.truncated_tail);

    // `make_durable` is idempotent and must not duplicate the group.
    journal.make_durable().unwrap();
    journal.make_durable().unwrap();
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);

    drop(journal);
    fs::remove_file(path).unwrap();
}

#[test]
fn commit_group_appends_then_syncs() {
    let path = temp_path("commit_appends_then_syncs");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    let receipt = journal.commit_group(&[put(1, 1, b"durable")]).unwrap();
    assert_eq!(receipt.last_lsn, 1);
    // The wrapper left the group complete and readable.
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    // A later explicit make_durable is a harmless no-op.
    journal.make_durable().unwrap();
    assert_eq!(Journal::scan_read_only(&path).unwrap().groups.len(), 1);
    drop(journal);
    fs::remove_file(path).unwrap();
}

#[test]
fn coordinator_visible_lsn_matches_committed_after_sync() {
    let path = temp_path("coordinator_visible");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    assert_eq!(coordinator.visible_lsn(), 0);
    let lsn = coordinator
        .publish_group(&[put(1, 1, b"first")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    assert_eq!(coordinator.visible_lsn(), lsn);
    assert_eq!(coordinator.committed_lsn(), lsn);
    drop(coordinator);
    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "multi-reader")]
fn published_groups_are_visible_not_durable_until_one_fsync_covers_them() {
    use crate::layout::ShardType;
    let path = temp_path("published_groups_one_fsync");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    let first = coordinator
        .publish_group_tagged(ShardType::State, &[put(1, 1, b"state")])
        .unwrap();
    let second = coordinator
        .publish_group_tagged(ShardType::EventDag, &[put(2, 2, b"event")])
        .unwrap();
    assert_eq!((first.last_lsn, second.last_lsn), (1, 2));
    // Complete and visible to a read-only overlay, but not yet durable.
    assert_eq!(coordinator.visible_lsn(), 2);
    assert_eq!(coordinator.committed_lsn(), 0);

    // One sync makes both groups durable and promotes each pool.
    coordinator.sync_through(2).unwrap();
    assert_eq!(coordinator.committed_lsn(), 2);
    assert_eq!(coordinator.committed_lsn_for_pool(ShardType::State), 1);
    assert_eq!(coordinator.committed_lsn_for_pool(ShardType::EventDag), 2);
    assert_eq!(coordinator.durability_stats().commits, 1);
    drop(coordinator);
    fs::remove_file(path).unwrap();
}

#[test]
fn empty_group_publication_is_rejected() {
    let path = temp_path("publish_group_empty");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    let error = coordinator.publish_group(&[]).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    #[cfg(feature = "multi-reader")]
    {
        use crate::layout::ShardType;
        let error = coordinator
            .publish_group_tagged(ShardType::State, &[])
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }
    assert_eq!(coordinator.visible_lsn(), 0);
    fs::remove_file(path).unwrap();
}

#[test]
fn committed_groups_round_trip_with_contiguous_lsns() {
    let path = temp_path("round_trip");
    let _ = fs::remove_file(&path);
    let (mut journal, initial) = Journal::open(&path).unwrap();
    assert_eq!(initial.groups.len(), 0);
    let committed = journal
        .commit_group(&[
            Mutation::Put {
                collection_id: [1; 16],
                node_id: [2; 16],
                payload: b"alpha".to_vec(),
            },
            Mutation::DeleteCollection {
                collection_id: [3; 16],
            },
        ])
        .unwrap();
    assert_eq!((committed.first_lsn, committed.last_lsn), (1, 2));
    drop(journal);

    let (reopened, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].sequence, 1);
    assert_eq!(scan.groups[0].first_lsn, committed.first_lsn);
    assert_eq!(scan.groups[0].last_lsn, committed.last_lsn);
    let entries = &scan.groups[0].entries;
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].lsn, 1);
    assert_eq!(
        entries[0].mutation,
        Mutation::Put {
            collection_id: [1; 16],
            node_id: [2; 16],
            payload: b"alpha".to_vec(),
        }
    );
    assert_eq!(entries[1].lsn, 2);
    assert_eq!(
        entries[1].mutation,
        Mutation::DeleteCollection {
            collection_id: [3; 16]
        }
    );
    // The recovered byte ranges must let a reader re-read each frame from
    // the segment: the first frame starts right after the file + group
    // headers, and lengths are fixed + payload + CRC.
    let group_payload_start = super::FILE_HEADER_LEN + super::GROUP_HEADER_LEN;
    assert_eq!(entries[0].offset, group_payload_start as u64);
    assert_eq!(
        entries[0].frame_len,
        (super::FRAME_FIXED_LEN + 5 + super::FRAME_TRAILER_LEN) as u64
    );
    assert_eq!(
        entries[1].offset,
        entries[0].offset.saturating_add(entries[0].frame_len)
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn incomplete_final_group_is_truncated_on_open() {
    let path = temp_path("torn_tail");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal
        .commit_group(&[Mutation::Put {
            collection_id: [1; 16],
            node_id: [2; 16],
            payload: b"complete".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let good_len = fs::metadata(&path).unwrap().len();
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"MWG1\x30\0")
        .unwrap();
    let (reopened, scan) = Journal::open(&path).unwrap();
    assert!(scan.truncated_tail);
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(fs::metadata(&path).unwrap().len(), good_len);
    drop(reopened);
    fs::remove_file(path).unwrap();
}

#[test]
fn read_only_scan_keeps_complete_groups_and_leaves_torn_tail_untouched() {
    let path = temp_path("read_only_torn_tail");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal.commit_group(&[put(1, 2, b"complete")]).unwrap();
    drop(journal);

    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"MWG1\x30\0")
        .unwrap();
    let before = fs::read(&path).unwrap();
    let scan = Journal::scan_read_only(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert!(scan.truncated_tail);
    assert_eq!(fs::read(&path).unwrap(), before, "scan must not truncate");
    fs::remove_file(path).unwrap();
}

#[test]
fn committed_group_corruption_fails_open() {
    let path = temp_path("corrupt");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal
        .commit_group(&[Mutation::Put {
            collection_id: [1; 16],
            node_id: [2; 16],
            payload: b"committed".to_vec(),
        }])
        .unwrap();
    drop(journal);

    let mut bytes = fs::read(&path).unwrap();
    let payload_byte = super::FILE_HEADER_LEN
        .saturating_add(super::GROUP_HEADER_LEN)
        .saturating_add(super::FRAME_FIXED_LEN);
    bytes[payload_byte] ^= 0x80;
    fs::write(&path, bytes).unwrap();
    assert!(Journal::open(&path).is_err());
    fs::remove_file(path).unwrap();
}

/// A partial header is a crash mid-creation, not corruption: no group can
/// exist without a synced header, so open resets the segment and it
/// remains usable.
#[test]
fn partial_file_header_is_reset_on_open() {
    let path = temp_path("partial_header");
    let _ = fs::remove_file(&path);
    fs::write(&path, b"MTX").unwrap();

    let (mut journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 0);
    assert!(!scan.truncated_tail);
    let committed = journal
        .commit_group(&[Mutation::Put {
            collection_id: [4; 16],
            node_id: [5; 16],
            payload: b"after-reset".to_vec(),
        }])
        .unwrap();
    assert_eq!(committed.sequence, 1);
    assert_eq!(committed.first_lsn, 1);
    drop(journal);

    let (_reopened, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].entries.len(), 1);
    fs::remove_file(path).unwrap();
}

/// Reopen after a torn tail, truncate it, and confirm numbering continues
/// from the last committed group rather than restarting.
#[test]
fn recovery_after_torn_tail_continues_numbering() {
    let path = temp_path("continue");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal
        .commit_group(&[Mutation::Put {
            collection_id: [1; 16],
            node_id: [1; 16],
            payload: b"one".to_vec(),
        }])
        .unwrap();
    drop(journal);

    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"MWG1\x30\0\x30\0")
        .unwrap();

    let (mut journal, scan) = Journal::open(&path).unwrap();
    assert!(scan.truncated_tail);
    assert_eq!(scan.groups.len(), 1);
    let committed = journal
        .commit_group(&[
            Mutation::Put {
                collection_id: [2; 16],
                node_id: [2; 16],
                payload: b"two".to_vec(),
            },
            Mutation::DeleteCollection {
                collection_id: [3; 16],
            },
        ])
        .unwrap();
    assert_eq!(committed.sequence, 2);
    assert_eq!(committed.first_lsn, 2);
    assert_eq!(committed.last_lsn, 3);
    drop(journal);

    let (_reopened, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 2);
    assert_eq!(scan.groups[0].last_lsn, 1);
    assert_eq!(scan.groups[1].first_lsn, 2);
    assert_eq!(scan.groups[1].last_lsn, 3);
    fs::remove_file(path).unwrap();
}

#[test]
fn sync_commits_through_the_newest_visible_lsn() {
    let path = temp_path("coordinator_boundary");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    let first_lsn = coordinator
        .publish_group(&[Mutation::Put {
            collection_id: [1; 16],
            node_id: [1; 16],
            payload: b"first".to_vec(),
        }])
        .unwrap()
        .last_lsn;
    let first_target = coordinator.capture_sync_target();
    assert_eq!(first_target, first_lsn);

    let second_lsn = coordinator
        .publish_group(&[Mutation::Put {
            collection_id: [1; 16],
            node_id: [2; 16],
            payload: b"second".to_vec(),
        }])
        .unwrap()
        .last_lsn;
    assert_eq!(second_lsn, first_lsn.saturating_add(1));

    // A sync commits through the newest visible LSN, not just the
    // caller's target: one fsync covers everything already appended, and
    // later callers find their target already durable.
    let first_commit = coordinator.sync_through(first_target).unwrap().unwrap();
    assert_eq!(first_commit.first_lsn, first_lsn);
    assert_eq!(first_commit.last_lsn, second_lsn);
    assert_eq!(coordinator.committed_lsn(), second_lsn);
    assert!(coordinator.sync_through(first_target).unwrap().is_none());
    assert!(coordinator.sync().unwrap().is_none());

    drop(coordinator);
    let (_journal, recovered) = Journal::open(&path).unwrap();
    assert_eq!(recovered.groups.len(), 2);
    assert_eq!(recovered.groups[0].last_lsn, first_target);
    assert_eq!(recovered.groups[1].first_lsn, second_lsn);
    fs::remove_file(path).unwrap();
}

/// Park the next sync between its handle clone and its fsync, with the
/// journal lock released. Returns a receiver that fires once a sync is
/// parked and a sender that releases it.
fn park_next_fsync(
    coordinator: &JournalCoordinator,
) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
    let (parked_tx, parked_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let parked_tx = std::sync::Mutex::new(parked_tx);
    let release_rx = std::sync::Mutex::new(release_rx);
    *coordinator.fsync_hook.lock() = Some(Arc::new(move || {
        parked_tx.lock().unwrap().send(()).ok();
        release_rx.lock().unwrap().recv().ok();
        Ok(())
    }));
    (parked_rx, release_tx)
}

/// The regression this change exists for: a publisher must not stall
/// behind an fsync in flight. The sync is parked mid-fsync and a publish
/// must still complete; that later group is then correctly left for the
/// next sync because it was appended after the range was captured.
#[test]
fn publish_makes_progress_while_an_fsync_is_in_flight() {
    let coordinator = open_arc("publish_during_fsync");
    let first = coordinator
        .publish_group(&[put(1, 1, b"first")])
        .unwrap()
        .last_lsn;
    let (parked, release) = park_next_fsync(&coordinator);

    let syncer = {
        let coordinator = coordinator.clone();
        std::thread::spawn(move || coordinator.sync().unwrap().unwrap())
    };
    parked
        .recv_timeout(Duration::from_secs(5))
        .expect("the sync must reach its fsync");

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let publisher = {
        let coordinator = coordinator.clone();
        std::thread::spawn(move || {
            let receipt = coordinator.publish_group(&[put(1, 2, b"second")]).unwrap();
            done_tx.send(receipt.last_lsn).unwrap();
        })
    };
    let second = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("a publish must not block behind an in-flight fsync");
    assert_eq!(second, first + 1);
    assert_eq!(coordinator.visible_lsn(), second);
    assert_eq!(coordinator.committed_lsn(), 0, "the fsync is still parked");

    release.send(()).unwrap();
    let parked_receipt = syncer.join().unwrap();
    publisher.join().unwrap();
    assert_eq!(
        (parked_receipt.first_lsn, parked_receipt.last_lsn),
        (first, first),
        "the parked sync's receipt covers only the range it captured"
    );
    assert_eq!(
        coordinator.committed_lsn(),
        first,
        "the group published during the fsync is visible but not committed"
    );

    *coordinator.fsync_hook.lock() = None;
    let later = coordinator.sync().unwrap().unwrap();
    assert_eq!((later.first_lsn, later.last_lsn), (second, second));
    assert_eq!(coordinator.committed_lsn(), second);
    fs::remove_file(coordinator.path()).unwrap();
}

/// An fsync error must never be reported as durable, and must poison the
/// journal so no later publish or sync can succeed on top of it: the
/// kernel may have dropped the dirty pages, so a retry could falsely pass.
/// The error is injected through the fsync hook, which stands in for
/// `sync_all` failing; it exercises the production poison path, not the
/// kernel's behavior.
#[test]
fn an_injected_fsync_error_poisons_and_is_never_reported_durable() {
    let coordinator = open_arc("failed_fsync_poison");
    coordinator.publish_group(&[put(1, 1, b"first")]).unwrap();
    *coordinator.fsync_hook.lock() = Some(Arc::new(|| Err(std::io::Error::other("injected"))));

    assert!(coordinator.sync().is_err());
    assert_eq!(
        coordinator.committed_lsn(),
        0,
        "a failed fsync is not marked durable"
    );
    assert!(
        coordinator.publish_group(&[put(1, 2, b"second")]).is_err(),
        "publication must be rejected after a failed fsync"
    );

    // Even with the fault gone, the poison persists: no retry may report
    // the earlier data durable.
    *coordinator.fsync_hook.lock() = None;
    assert!(coordinator.sync().is_err());
    assert_eq!(coordinator.committed_lsn(), 0);
    fs::remove_file(coordinator.path()).unwrap();
}

/// Torn-tail recovery on hand-built truncated images. It writes two
/// groups, cuts a copy at the boundary before the second group (and again
/// inside it, as a torn write), and checks that recovery keeps exactly the
/// earlier prefix and continues numbering from it. It says nothing about
/// what an actual crash would leave on disk after an in-flight fsync.
#[test]
fn recovery_discards_a_clean_or_torn_tail_group() {
    let coordinator = open_arc("torn_tail_recovery");
    let path = coordinator.path().clone();
    let first = coordinator.publish_group(&[put(1, 1, b"first")]).unwrap();
    let second = coordinator.publish_group(&[put(1, 2, b"second")]).unwrap();
    coordinator.sync().unwrap();
    drop(coordinator);

    let prefix_len = super::FILE_HEADER_LEN as u64 + first.bytes_written;
    let full_len = prefix_len + second.bytes_written;
    assert_eq!(fs::metadata(&path).unwrap().len(), full_len);
    for cut in [prefix_len, prefix_len + second.bytes_written / 2] {
        let copy = temp_path("torn_tail_recovery_copy");
        fs::copy(&path, &copy).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&copy)
            .unwrap()
            .set_len(cut)
            .unwrap();
        let (journal, scan) = Journal::open(&copy).unwrap();
        assert_eq!(scan.groups.len(), 1, "only the intact prefix is recovered");
        assert_eq!(scan.groups[0].last_lsn, first.last_lsn);
        let recovered = JournalCoordinator::new(journal, &scan);
        let next = recovered.publish_group(&[put(1, 3, b"third")]).unwrap();
        assert_eq!(next.first_lsn, first.last_lsn + 1, "numbering continues");
        drop(recovered);
        fs::remove_file(copy).unwrap();
    }
    fs::remove_file(path).unwrap();
}

/// A crash can persist a later page while an earlier one never reached the
/// disk, leaving a hole with intact groups after it. That is
/// indistinguishable from bit rot in acknowledged data, so the journal
/// fails closed: the read-only scan and recovery both refuse the segment
/// with `InvalidData` and neither repairs it, instead of silently dropping
/// the later groups. A hole with nothing valid after it is an ordinary torn
/// tail and is covered elsewhere. The image is built by zeroing the middle
/// group of a finished file; it says nothing about which pages a real crash
/// would leave behind.
#[test]
fn a_hole_before_a_later_intact_group_fails_closed() {
    let coordinator = open_arc("hole_before_later_group");
    let path = coordinator.path().clone();
    let first = coordinator.publish_group(&[put(1, 1, b"first")]).unwrap();
    let middle = coordinator.publish_group(&[put(1, 2, b"middle")]).unwrap();
    coordinator.publish_group(&[put(1, 3, b"last")]).unwrap();
    coordinator.sync().unwrap();
    drop(coordinator);

    let hole_start = super::FILE_HEADER_LEN as u64 + first.bytes_written;
    let hole_len = usize::try_from(middle.bytes_written).unwrap();
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(hole_start)).unwrap();
        file.write_all(&vec![0; hole_len]).unwrap();
    }
    let before = fs::read(&path).unwrap();

    let scan_error = Journal::scan_read_only(&path).unwrap_err();
    assert_eq!(scan_error.kind(), std::io::ErrorKind::InvalidData);
    let open_error = Journal::open(&path)
        .err()
        .expect("recovery must refuse a hole");
    assert_eq!(open_error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(
        fs::read(&path).unwrap(),
        before,
        "neither the scan nor a failed recovery may modify the segment"
    );
    fs::remove_file(path).unwrap();
}

/// A scan reports the last bytes of the groups it consumed, never reaching
/// into the file header, whether it read the whole segment or only a tail.
#[test]
fn a_scan_reports_the_bytes_it_consumed() {
    let path = temp_path("consumed_tail");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal.append_group(&[put(1, 1, b"small")]).unwrap();
    let bytes = fs::read(&path).unwrap();

    let scan = Journal::scan_read_only(&path).unwrap();
    assert!(bytes.len() < super::FILE_HEADER_LEN + super::CONSUMED_TAIL_LEN);
    assert_eq!(
        scan.consumed_tail,
        &bytes[super::FILE_HEADER_LEN..],
        "a short segment reports everything after the header"
    );
    assert_eq!(
        scan.consumed_tail.len(),
        super::consumed_tail_len(scan.valid_len)
    );

    for node in 2..12u8 {
        journal.append_group(&[put(1, node, &[node; 300])]).unwrap();
    }
    let bytes = fs::read(&path).unwrap();
    let full = Journal::scan_read_only(&path).unwrap();
    let valid = usize::try_from(full.valid_len).unwrap();
    assert_eq!(
        full.consumed_tail,
        &bytes[valid - super::CONSUMED_TAIL_LEN..valid]
    );

    // An incremental scan reports only what it consumed from its start, so
    // the window is the last `CONSUMED_TAIL_LEN` bytes of `start..valid`,
    // or all of it when that range is shorter. The ten 300-byte groups
    // above give a range longer than the window.
    let start = scan.valid_len;
    let tail = Journal::scan_read_only_from(&path, start, scan.groups[0].last_lsn + 1).unwrap();
    let start = usize::try_from(start).unwrap();
    let expected_start = valid.saturating_sub(super::CONSUMED_TAIL_LEN).max(start);
    assert_eq!(
        tail.consumed_tail,
        &bytes[expected_start..valid],
        "incremental scans report the exact consumed suffix"
    );

    // A small incremental scan consumes fewer bytes than the window, so it
    // reports exactly those bytes and nothing from before its start.
    let before_small = full.valid_len;
    journal.append_group(&[put(1, 99, b"tiny")]).unwrap();
    let bytes = fs::read(&path).unwrap();
    let small = Journal::scan_read_only_from(
        &path,
        before_small,
        full.groups.last().unwrap().last_lsn + 1,
    )
    .unwrap();
    let from = usize::try_from(before_small).unwrap();
    let end = usize::try_from(small.valid_len).unwrap();
    assert!(end - from < super::CONSUMED_TAIL_LEN);
    assert_eq!(
        small.consumed_tail,
        &bytes[from..end],
        "a scan shorter than the window reports exactly the bytes it consumed"
    );
    fs::remove_file(path).unwrap();
}

/// Zero `len` bytes of the file at `start`, standing in for a page that
/// never reached the disk.
#[cfg(feature = "multi-reader")]
fn punch_hole(path: &Path, start: u64, len: u64) {
    use std::io::{Seek, SeekFrom, Write};
    let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(start)).unwrap();
    file.write_all(&vec![0; usize::try_from(len).unwrap()])
        .unwrap();
}

/// The embedded mark a shared segment currently claims.
#[cfg(feature = "multi-reader")]
fn durable_mark(path: &Path) -> u64 {
    let bytes = fs::read(path).unwrap();
    let (version, _base_sequence, _base_lsn) = super::validate_file_header(&bytes).unwrap();
    super::read_durable_len_from_header(&bytes, version, u64::try_from(bytes.len()).unwrap())
}

#[cfg(feature = "multi-reader")]
fn open_shared_arc(label: &str) -> Arc<JournalCoordinator> {
    let path = temp_path(label);
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    Arc::new(JournalCoordinator::new(journal, &scan))
}

#[cfg(feature = "multi-reader")]
fn publish(coordinator: &JournalCoordinator, node: u8, payload: &[u8]) -> super::CommitReceipt {
    coordinator
        .publish_group_tagged(crate::layout::ShardType::State, &[put(1, node, payload)])
        .unwrap()
}

/// The embedded mark advances only after the data fsync succeeds.
#[cfg(feature = "multi-reader")]
#[test]
fn a_sync_records_the_embedded_durable_mark_after_its_fsync() {
    let coordinator = open_shared_arc("embedded_mark_after_sync");
    let path = coordinator.path();
    assert_eq!(durable_mark(&path), 0);
    let first = publish(&coordinator, 1, b"first");
    assert_eq!(durable_mark(&path), 0);
    coordinator.sync().unwrap();
    assert_eq!(
        durable_mark(&path),
        super::FILE_HEADER_LEN as u64 + first.bytes_written
    );
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "multi-reader")]
#[test]
fn a_failed_fsync_does_not_advance_the_embedded_mark() {
    let coordinator = open_shared_arc("embedded_mark_failed_fsync");
    let path = coordinator.path();
    publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    let mark = durable_mark(&path);
    publish(&coordinator, 2, b"second");
    *coordinator.fsync_hook.lock() = Some(Arc::new(|| Err(std::io::Error::other("injected"))));
    assert!(coordinator.sync().is_err());
    assert_eq!(durable_mark(&path), mark);
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "multi-reader")]
#[test]
fn a_hole_above_the_embedded_mark_is_truncated() {
    let coordinator = open_shared_arc("embedded_hole_above_mark");
    let path = coordinator.path();
    let first = publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    let middle = publish(&coordinator, 2, b"middle");
    publish(&coordinator, 3, b"last");
    drop(coordinator);
    let hole_start = super::FILE_HEADER_LEN as u64 + first.bytes_written;
    punch_hole(&path, hole_start, middle.bytes_written);
    let (journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(fs::metadata(&path).unwrap().len(), hole_start);
    drop(journal);
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "multi-reader")]
#[test]
fn a_hole_below_the_embedded_mark_still_fails_closed() {
    let coordinator = open_shared_arc("embedded_hole_below_mark");
    let path = coordinator.path();
    let first = publish(&coordinator, 1, b"first");
    let middle = publish(&coordinator, 2, b"middle");
    publish(&coordinator, 3, b"last");
    coordinator.sync().unwrap();
    drop(coordinator);
    punch_hole(
        &path,
        super::FILE_HEADER_LEN as u64 + first.bytes_written,
        middle.bytes_written,
    );
    let error = Journal::open_shared(&path).err().expect("must refuse");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "multi-reader")]
#[test]
fn recovery_marks_groups_it_keeps() {
    let path = temp_path("embedded_recovery_mark");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open_shared(&path).unwrap();
    journal
        .append_group_for_current_mode(
            &[(Some(crate::layout::ShardType::State), put(1, 1, b"cached"))],
            None,
        )
        .unwrap();
    drop(journal);
    assert_eq!(durable_mark(&path), 0);
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert!(durable_mark(&path) > super::FILE_HEADER_LEN as u64);
    fs::remove_file(path).unwrap();
}

/// A torn newest slot leaves the previous valid slot usable. The immutable
/// base header is separate: corrupting it still fails header validation
/// rather than being interpreted as a durability-mark failure.
#[cfg(feature = "multi-reader")]
#[test]
fn a_torn_newest_mark_falls_back_without_trusting_the_tail() {
    let coordinator = open_shared_arc("embedded_mark_slots");
    let path = coordinator.path();
    let first = publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    let second = publish(&coordinator, 2, b"second");
    coordinator.sync().unwrap();
    drop(coordinator);

    let mut bytes = fs::read(&path).unwrap();
    // Generation two occupies slot zero; damage its CRC/data while the
    // generation-one slot remains intact.
    bytes[super::MARK_SLOT_OFFSETS[0]] ^= 0x80;
    fs::write(&path, &bytes).unwrap();
    punch_hole(
        &path,
        super::FILE_HEADER_LEN as u64 + first.bytes_written,
        second.bytes_written,
    );
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 1, "the older mark bounds recovery");
    fs::remove_file(path).unwrap();
}

/// Overwrite `bytes` at `offset`, standing in for a torn or garbled write.
#[cfg(feature = "multi-reader")]
fn overwrite(path: &Path, offset: usize, bytes: &[u8]) {
    use std::io::{Seek, SeekFrom};
    let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(u64::try_from(offset).unwrap()))
        .unwrap();
    file.write_all(bytes).unwrap();
}

/// Both mark slots as currently decoded from the file's header region.
#[cfg(feature = "multi-reader")]
fn mark_slots(path: &Path) -> [Option<(u64, u64)>; 2] {
    let bytes = fs::read(path).unwrap();
    super::MARK_SLOT_OFFSETS.map(|offset| {
        super::decode_mark_slot(&bytes[offset..]).map(|slot| (slot.generation, slot.durable_len))
    })
}

/// Groups begin after the base header's sector and both mark sectors, and
/// each region sits in a sector of its own, so a torn write to one cannot
/// reach another.
#[cfg(feature = "multi-reader")]
#[test]
fn groups_start_after_a_reserved_sector_aligned_mark_region() {
    assert_eq!(super::FILE_HEADER_LEN % super::MARK_SECTOR_LEN, 0);
    const { assert!(super::FH_BASE_CRC.end <= super::MARK_SECTOR_LEN) };
    for offset in super::MARK_SLOT_OFFSETS {
        assert_eq!(offset % super::MARK_SECTOR_LEN, 0, "slot starts a sector");
        assert!(offset >= super::MARK_SECTOR_LEN, "not in the header sector");
        assert!(offset + super::MARK_SLOT_LEN <= offset + super::MARK_SECTOR_LEN);
        assert!(offset + super::MARK_SECTOR_LEN <= super::FILE_HEADER_LEN);
    }
    assert_ne!(super::MARK_SLOT_OFFSETS[0], super::MARK_SLOT_OFFSETS[1]);

    let coordinator = open_shared_arc("mark_region_layout");
    let path = coordinator.path().clone();
    let first = publish(&coordinator, 1, b"first");
    drop(coordinator);
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        super::FILE_HEADER_LEN as u64 + first.bytes_written,
        "the first group starts right after the reserved region"
    );
    fs::remove_file(path).unwrap();
}

/// A fresh segment has written no mark: both slots are unwritten and claim
/// nothing.
#[cfg(feature = "multi-reader")]
#[test]
fn an_unwritten_mark_region_claims_nothing() {
    let coordinator = open_shared_arc("mark_unwritten");
    let path = coordinator.path().clone();
    publish(&coordinator, 1, b"first");
    drop(coordinator);
    assert_eq!(mark_slots(&path), [None, None]);
    assert_eq!(durable_mark(&path), 0);
    fs::remove_file(path).unwrap();
}

/// Marks alternate between the two slots by generation, so a write always
/// goes to the slot that does not hold the newest mark.
#[cfg(feature = "multi-reader")]
#[test]
fn marks_alternate_between_the_two_slots() {
    let coordinator = open_shared_arc("mark_alternate");
    let path = coordinator.path().clone();
    let mut file_sizes = Vec::new();
    for node in 1..=3u8 {
        publish(&coordinator, node, b"x");
        coordinator.sync().unwrap();
        file_sizes.push(fs::metadata(&path).unwrap().len());
    }
    drop(coordinator);

    let [slot0, slot1] = mark_slots(&path);
    let (generation0, len0) = slot0.expect("slot 0 holds a mark");
    let (generation1, len1) = slot1.expect("slot 1 holds a mark");
    // Generations 1 and 3 land in slot 1, generation 2 in slot 0, and the
    // third overwrote the first: the two slots hold the two newest.
    assert_eq!((generation0, len0), (2, file_sizes[1]));
    assert_eq!((generation1, len1), (3, file_sizes[2]));
    assert_eq!(
        durable_mark(&path),
        file_sizes[2],
        "recovery takes the newest"
    );
    fs::remove_file(path).unwrap();
}

/// Fault injection: the newest slot is torn, so recovery falls back to the
/// older slot's mark instead of losing the mark altogether.
#[cfg(feature = "multi-reader")]
#[test]
fn a_torn_newest_slot_falls_back_to_the_older_one() {
    let coordinator = open_shared_arc("mark_torn_newest");
    let path = coordinator.path().clone();
    publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    let first_len = fs::metadata(&path).unwrap().len();
    publish(&coordinator, 2, b"second");
    coordinator.sync().unwrap();
    drop(coordinator);
    assert_eq!(mark_slots(&path)[0].unwrap().0, 2, "generation 2 is newest");

    // Tear generation 2, in slot 0: half of it never reached the disk.
    overwrite(&path, super::MARK_SLOT_OFFSETS[0] + 12, &[0xAB; 16]);
    let [torn, older] = mark_slots(&path);
    assert!(torn.is_none(), "the torn slot fails its checksum");
    assert_eq!(older.unwrap().0, 1);
    assert_eq!(durable_mark(&path), first_len);
    fs::remove_file(path).unwrap();
}

/// Fault injection: both slots are invalid. Durability is then unknown,
/// not proven, so recovery treats damage as an unacknowledged tail and
/// truncates instead of refusing to open.
#[cfg(feature = "multi-reader")]
#[test]
fn two_invalid_slots_mean_unknown_durability() {
    let coordinator = open_shared_arc("mark_both_invalid");
    let path = coordinator.path().clone();
    let first = publish(&coordinator, 1, b"first");
    let middle = publish(&coordinator, 2, b"middle");
    publish(&coordinator, 3, b"last");
    coordinator.sync().unwrap();
    coordinator.sync().unwrap();
    drop(coordinator);
    // Garble every slot (both are nonzero, so neither reads as unwritten).
    for offset in super::MARK_SLOT_OFFSETS {
        overwrite(&path, offset, &[0xCD; super::MARK_SLOT_LEN]);
    }
    assert_eq!(mark_slots(&path), [None, None]);
    assert_eq!(durable_mark(&path), 0);

    // A hole that would be corruption under a valid mark is now a tail.
    let hole_start = super::FILE_HEADER_LEN as u64 + first.bytes_written;
    punch_hole(&path, hole_start, middle.bytes_written);
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(fs::metadata(&path).unwrap().len(), hole_start);
    fs::remove_file(path).unwrap();
}

/// The base header and the mark slots are checked independently: garbling
/// every mark leaves the base header valid and the segment openable, while
/// garbling the base header is detected even with valid marks.
#[cfg(feature = "multi-reader")]
#[test]
fn base_header_and_mark_slots_are_corrupted_independently() {
    let coordinator = open_shared_arc("mark_vs_base");
    let path = coordinator.path().clone();
    publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    drop(coordinator);
    let pristine = fs::read(&path).unwrap();

    // Mark sectors garbled: the base header is untouched and still valid.
    for offset in super::MARK_SLOT_OFFSETS {
        overwrite(&path, offset, &vec![0xEE; super::MARK_SECTOR_LEN]);
    }
    assert!(super::validate_file_header(&fs::read(&path).unwrap()).is_ok());
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 1, "the data is still there");

    // Base header garbled, marks restored: detected on its own.
    fs::write(&path, &pristine).unwrap();
    overwrite(&path, super::FH_BASE_SEQUENCE.start, &[0x77; 4]);
    assert_eq!(
        Journal::open_shared(&path)
            .err()
            .expect("must refuse")
            .kind(),
        std::io::ErrorKind::InvalidData
    );
    assert_eq!(
        Journal::scan_read_only(&path).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
    fs::remove_file(path).unwrap();
}

/// Writes the damage directly, modelling the post-crash disk image rather
/// than proving power-loss behavior. The reason the slots have sectors of their own: a
/// mark write torn across its whole sector leaves the base header and the
/// first group readable.
#[cfg(feature = "multi-reader")]
#[test]
fn a_torn_mark_sector_cannot_reach_the_base_header() {
    let coordinator = open_shared_arc("mark_sector_torn");
    let path = coordinator.path().clone();
    publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    let before = fs::read(&path).unwrap();
    drop(coordinator);
    overwrite(
        &path,
        super::MARK_SLOT_OFFSETS[1],
        &vec![0x00; super::MARK_SECTOR_LEN],
    );
    let after = fs::read(&path).unwrap();
    assert_eq!(
        after[..super::MARK_SECTOR_LEN],
        before[..super::MARK_SECTOR_LEN],
        "the base header's sector is untouched"
    );
    assert!(super::validate_file_header(&after).is_ok());
    assert_eq!(Journal::scan_read_only(&path).unwrap().groups.len(), 1);
    fs::remove_file(path).unwrap();
}

/// The mark is written with a positioned write, so it never moves the
/// append cursor: groups published after a sync land at the end, not at
/// the mark's offset.
#[cfg(feature = "multi-reader")]
#[test]
fn a_mark_write_leaves_the_append_cursor_alone() {
    let coordinator = open_shared_arc("mark_cursor");
    let path = coordinator.path().clone();
    let mut total = super::FILE_HEADER_LEN as u64;
    for node in 1..=4u8 {
        total += publish(&coordinator, node, b"data").bytes_written;
        coordinator.sync().unwrap();
    }
    drop(coordinator);
    assert_eq!(fs::metadata(&path).unwrap().len(), total);
    let (_journal, scan) = Journal::open_shared(&path).unwrap();
    assert_eq!(scan.groups.len(), 4, "no group was overwritten by a mark");
    fs::remove_file(path).unwrap();
}

/// Reclaim puts a first-generation mark into the rebuilt file before that
/// file is fsynced, so the mark is durable with the data and covers all of
/// it; the old segment's marks are gone. A hole in what it retained is
/// corruption and still fails closed.
#[cfg(feature = "multi-reader")]
#[test]
fn reclaim_writes_a_first_generation_mark_with_the_rebuilt_data() {
    let coordinator = open_shared_arc("mark_reclaim");
    let path = coordinator.path().clone();
    let first = publish(&coordinator, 1, b"first");
    let second = publish(&coordinator, 2, b"second");
    publish(&coordinator, 3, b"third");
    coordinator.sync().unwrap();
    coordinator.reclaim_through(first.last_lsn).unwrap();
    drop(coordinator);

    let len = fs::metadata(&path).unwrap().len();
    let [slot0, slot1] = mark_slots(&path);
    assert_eq!(slot0, None);
    assert_eq!(slot1, Some((1, len)));

    punch_hole(&path, super::FILE_HEADER_LEN as u64, second.bytes_written);
    assert_eq!(
        Journal::open_shared(&path)
            .err()
            .expect("must refuse")
            .kind(),
        std::io::ErrorKind::InvalidData
    );
    fs::remove_file(path).unwrap();
}

/// Only damage a crash can explain is truncated, and only above the mark. A
/// group whose header checksum verifies but whose LSN skips ahead cannot be
/// produced by a crash on an append-only file, so it stays fatal even above
/// the mark: dropping it would hide a writer bug.
#[cfg(feature = "multi-reader")]
#[test]
fn a_checksum_valid_group_with_a_wrong_lsn_is_fatal_even_above_the_mark() {
    let coordinator = open_shared_arc("mark_fatal_class");
    let path = coordinator.path().clone();
    let first = publish(&coordinator, 1, b"first");
    coordinator.sync().unwrap();
    drop(coordinator);

    let bad_lsn = first.last_lsn + 5;
    let group = super::CommittedGroup {
        sequence: first.sequence + 1,
        first_lsn: bad_lsn,
        last_lsn: bad_lsn,
        entries: vec![super::JournalEntry {
            lsn: bad_lsn,
            offset: 0,
            frame_len: 0,
            pool: Some(crate::layout::ShardType::State),
            mutation: put(1, 2, b"skipped"),
        }],
    };
    let mut bytes = Vec::new();
    super::encode_group(&group, super::JournalVersion::V5PoolTagged, &mut bytes).unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
    assert!(fs::metadata(&path).unwrap().len() > durable_mark(&path));
    assert_eq!(
        Journal::open_shared(&path)
            .err()
            .expect("must refuse")
            .kind(),
        std::io::ErrorKind::InvalidData
    );
    fs::remove_file(path).unwrap();
}

/// A per-pool segment keeps no mark, so its mark region stays unwritten and
/// it fails closed on any hole, as it always did.
#[test]
fn a_per_pool_segment_keeps_no_mark_and_fails_closed() {
    let coordinator = open_arc("per_pool_no_mark");
    let path = coordinator.path().clone();
    let first = coordinator.publish_group(&[put(1, 1, b"first")]).unwrap();
    let middle = coordinator.publish_group(&[put(1, 2, b"middle")]).unwrap();
    coordinator.publish_group(&[put(1, 3, b"last")]).unwrap();
    coordinator.sync().unwrap();
    drop(coordinator);
    let bytes = fs::read(&path).unwrap();
    assert!(
        bytes[super::MARK_SECTOR_LEN..super::FILE_HEADER_LEN]
            .iter()
            .all(|&byte| byte == 0),
        "a per-pool segment must not write a mark"
    );

    {
        use std::io::{Seek, SeekFrom};
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(
            super::FILE_HEADER_LEN as u64 + first.bytes_written,
        ))
        .unwrap();
        file.write_all(&vec![0; usize::try_from(middle.bytes_written).unwrap()])
            .unwrap();
    }
    assert_eq!(
        Journal::open(&path).err().expect("must refuse").kind(),
        std::io::ErrorKind::InvalidData
    );
    fs::remove_file(path).unwrap();
}

/// Reclaim replaces the segment file, so it must wait for an in-flight
/// fsync instead of swapping the inode under it.
#[test]
fn reclaim_blocks_behind_an_in_flight_fsync() {
    let coordinator = open_arc("reclaim_behind_fsync");
    let first = coordinator
        .publish_group(&[put(1, 1, b"first")])
        .unwrap()
        .last_lsn;
    let (parked, release) = park_next_fsync(&coordinator);

    let syncer = {
        let coordinator = coordinator.clone();
        std::thread::spawn(move || coordinator.sync().unwrap())
    };
    parked
        .recv_timeout(Duration::from_secs(5))
        .expect("the sync must reach its fsync");

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let reclaimer = {
        let coordinator = coordinator.clone();
        std::thread::spawn(move || {
            coordinator.reclaim_through(first).unwrap();
            done_tx.send(()).unwrap();
        })
    };
    assert!(
        done_rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "reclaim must wait for the in-flight fsync"
    );

    release.send(()).unwrap();
    syncer.join().unwrap();
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("reclaim must run once the fsync finishes");
    reclaimer.join().unwrap();
    assert_eq!(coordinator.committed_lsn(), first);
    fs::remove_file(coordinator.path()).unwrap();
}

/// Concurrent callers that queue behind an in-flight fsync must share the
/// next one instead of each starting their own. The first fsync is parked
/// until every other writer has published and is waiting on `sync_lock`,
/// so the outcome is exact: the first fsync covers only the first group,
/// and one more fsync covers the other seven.
#[test]
fn concurrent_syncs_coalesce_into_one_follow_up_fsync() {
    const WRITERS: u8 = 8;
    let coordinator = open_arc("concurrent_sync_coalesce");
    let (parked, release) = park_next_fsync(&coordinator);

    let leader = {
        let coordinator = coordinator.clone();
        std::thread::spawn(move || {
            coordinator.publish_group(&[put(2, 0, b"w")]).unwrap();
            coordinator.sync().unwrap();
        })
    };
    parked
        .recv_timeout(Duration::from_secs(5))
        .expect("the first sync must reach its fsync");

    let mut followers = Vec::new();
    for node in 1..WRITERS {
        let coordinator = coordinator.clone();
        followers.push(std::thread::spawn(move || {
            coordinator.publish_group(&[put(2, node, b"w")]).unwrap();
            coordinator.sync().unwrap();
        }));
    }
    // Each follower counts itself as a waiter before blocking on
    // `sync_lock`, which the parked leader holds.
    let waiting = coordinator.clone();
    wait_until(
        move || waiting.journal_waiters.load(Ordering::Relaxed) == u64::from(WRITERS - 1),
        "every follower to queue behind the parked fsync",
    );
    assert_eq!(coordinator.committed_lsn(), 0);

    // Let the parked fsync finish, and every later fsync run unparked
    // (a dropped sender makes the hook's wait return at once).
    release.send(()).unwrap();
    drop(release);
    leader.join().unwrap();
    for follower in followers {
        follower.join().unwrap();
    }

    assert_eq!(coordinator.committed_lsn(), u64::from(WRITERS));
    let stats = coordinator.durability_stats();
    assert_eq!(
        stats.commits, 2,
        "the leader's fsync plus one shared follow-up"
    );
    assert_eq!(
        stats.sync_coalesced,
        u64::from(WRITERS - 2),
        "all but the follow-up's leader found their target already marked durable"
    );
    *coordinator.fsync_hook.lock() = None;
    fs::remove_file(coordinator.path()).unwrap();
}

/// A target that was never published is a caller bug, not a durable commit.
#[test]
fn sync_through_rejects_an_unpublished_target() {
    let path = temp_path("coordinator_unpublished");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    assert!(coordinator.sync_through(0).unwrap().is_none());
    let error = coordinator.sync_through(1).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    fs::remove_file(path).unwrap();
}

/// Many writers publishing and syncing concurrently must produce one
/// contiguous, gap-free LSN sequence: a caller that loses the commit race
/// is covered by another caller's group rather than acknowledged early.
#[test]
fn concurrent_callers_produce_contiguous_groups() {
    let path = temp_path("coordinator_concurrent");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = std::sync::Arc::new(JournalCoordinator::new(journal, &scan));

    let threads = 8_u8;
    let per_thread = 16_u8;
    let mut handles = Vec::new();
    for thread in 0..threads {
        let coordinator = std::sync::Arc::clone(&coordinator);
        handles.push(std::thread::spawn(move || {
            for item in 0..per_thread {
                coordinator
                    .publish_group(&[Mutation::Put {
                        collection_id: [thread; 16],
                        node_id: [item; 16],
                        payload: b"payload".to_vec(),
                    }])
                    .unwrap();
                // Either this caller commits the group or a concurrent
                // caller already committed one covering this LSN.
                coordinator.sync().unwrap();
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let published = coordinator.capture_sync_target();
    assert_eq!(
        published,
        u64::from(u32::from(threads) * u32::from(per_thread))
    );
    coordinator.sync().unwrap();

    drop(coordinator);
    let (_journal, recovered) = Journal::open(&path).unwrap();
    let total: u64 = recovered
        .groups
        .iter()
        .map(|group| u64::try_from(group.entries.len()).unwrap())
        .sum();
    assert_eq!(total, published);
    let mut expected_first = 1;
    for group in &recovered.groups {
        assert_eq!(group.first_lsn, expected_first);
        expected_first = group.last_lsn.saturating_add(1);
    }
    fs::remove_file(path).unwrap();
}

/// Rotation drops covered groups but keeps the uncovered suffix, preserving
/// each surviving group's original sequence and LSNs.
#[test]
fn reclaim_retains_uncovered_suffix_and_preserves_numbering() {
    let path = temp_path("reclaim_suffix");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    let first = journal.commit_group(&[put(1, 1, b"first")]).unwrap();
    let second = journal
        .commit_group(&[
            put(1, 2, b"second"),
            Mutation::DeleteCollection {
                collection_id: [9; 16],
            },
        ])
        .unwrap();
    let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();

    let reclaim = journal.reclaim_through(first.last_lsn).unwrap();
    assert_eq!(reclaim.retained_groups, 2);
    assert!(reclaim.reclaimed_bytes > 0);
    assert!(!path.with_extension("rotate").exists());
    drop(journal);

    let (mut journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 2);
    assert_eq!(scan.groups[0].sequence, second.sequence);
    assert_eq!(scan.groups[0].first_lsn, second.first_lsn);
    assert_eq!(scan.groups[0].last_lsn, second.last_lsn);
    assert_eq!(scan.groups[1].sequence, third.sequence);
    assert_eq!(scan.groups[1].last_lsn, third.last_lsn);

    // Numbering continues from the global high-water mark, not the new base.
    let fourth = journal.commit_group(&[put(1, 4, b"fourth")]).unwrap();
    assert_eq!(fourth.sequence, third.sequence.saturating_add(1));
    assert_eq!(fourth.first_lsn, third.last_lsn.saturating_add(1));
    drop(journal);

    let (_journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 3);
    fs::remove_file(path).unwrap();
}

/// Reclaiming everything leaves an empty segment whose base still points
/// past the last committed LSN, so numbering never restarts.
#[test]
fn reclaim_all_covered_keeps_numbering_for_the_next_group() {
    let path = temp_path("reclaim_all");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal.commit_group(&[put(1, 1, b"first")]).unwrap();
    let second = journal.commit_group(&[put(1, 2, b"second")]).unwrap();

    let reclaim = journal.reclaim_through(second.last_lsn).unwrap();
    assert_eq!(reclaim.retained_groups, 0);
    drop(journal);

    let (mut journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 0);
    let third = journal.commit_group(&[put(1, 3, b"third")]).unwrap();
    assert_eq!(third.sequence, second.sequence.saturating_add(1));
    assert_eq!(third.first_lsn, second.last_lsn.saturating_add(1));
    drop(journal);

    let (_journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].first_lsn, second.last_lsn.saturating_add(1));
    fs::remove_file(path).unwrap();
}

/// A covered LSN at or below the segment base reclaims nothing.
#[test]
fn reclaim_below_base_is_a_noop() {
    let path = temp_path("reclaim_noop");
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open(&path).unwrap();
    journal.commit_group(&[put(1, 1, b"only")]).unwrap();

    let reclaim = journal.reclaim_through(0).unwrap();
    assert_eq!(reclaim.retained_groups, 1);
    assert_eq!(reclaim.reclaimed_bytes, 0);
    drop(journal);
    fs::remove_file(path).unwrap();
}

/// The coordinator exposes reclamation without disturbing its LSN counters:
/// a publish after reclaim still continues the sequence.
#[test]
fn coordinator_reclaims_without_reusing_lsns() {
    let path = temp_path("coordinator_reclaim");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    let lsn = coordinator
        .publish_group(&[put(1, 1, b"first")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    let receipt = coordinator.sync().unwrap().unwrap();
    assert_eq!(receipt.last_lsn, lsn);

    let reclaim = coordinator.reclaim_through(receipt.last_lsn).unwrap();
    assert_eq!(reclaim.retained_groups, 0);
    assert_eq!(coordinator.committed_lsn(), receipt.last_lsn);

    let next = coordinator
        .publish_group(&[put(1, 2, b"second")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    assert_eq!(next, lsn.saturating_add(1));
    coordinator.sync().unwrap();
    drop(coordinator);

    let (_journal, scan) = Journal::open(&path).unwrap();
    assert_eq!(scan.groups.len(), 1);
    assert_eq!(scan.groups[0].first_lsn, lsn.saturating_add(1));
    fs::remove_file(path).unwrap();
}

/// A reclaim publishes no group, but it does replace the segment and move
/// its base LSN. A gated reader watching only publishes would skip the
/// base-jump check, so the reclaim must advance the publish signal too.
#[test]
fn a_reclaim_advances_the_publish_signal() {
    let path = temp_path("reclaim_publish_signal");
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);
    coordinator.enable_publish_signal().unwrap();
    let signal = coordinator.publish_signal().expect("signal created");
    let (epoch, revision) = signal.snapshot();

    let receipt = coordinator
        .publish_group(&[put(1, 1, b"first")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    assert_eq!(
        signal.snapshot(),
        (epoch, revision.saturating_add(1)),
        "a publish advances the revision"
    );

    coordinator.sync().unwrap();
    coordinator.reclaim_through(receipt).unwrap();
    assert_eq!(
        signal.snapshot(),
        (epoch, revision.saturating_add(2)),
        "a reclaim advances the revision even though it publishes nothing"
    );

    drop(coordinator);
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(crate::packfile::publish_signal::PublishSignal::path_for(
        &path,
    ));
}

#[test]
fn journal_signal_setup_failure_is_reported_before_attachment() {
    let path = temp_path("publish_signal_setup_failure");
    let signal_path = crate::packfile::publish_signal::PublishSignal::path_for(&path);
    let _ = fs::remove_file(&signal_path);
    let (journal, scan) = Journal::open(&path).unwrap();
    fs::create_dir(&signal_path).unwrap();
    let coordinator = JournalCoordinator::new(journal, &scan);

    assert!(coordinator.enable_publish_signal().is_err());
    assert!(coordinator.publish_signal().is_none());

    fs::remove_dir(&signal_path).unwrap();
    let _ = fs::remove_file(&path);
}

#[test]
#[cfg(feature = "multi-reader")]
fn shared_sequence_orders_groups_across_segments_and_allows_gaps() {
    let dir = temp_path("shared_sequence");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let counter = Arc::new(AtomicU64::new(1));

    let path_a = dir.join("a.wal");
    let path_b = dir.join("b.wal");
    let (journal_a, scan_a) = Journal::open(&path_a).unwrap();
    let (journal_b, scan_b) = Journal::open(&path_b).unwrap();
    let a = JournalCoordinator::with_shared_sequence(journal_a, &scan_a, Arc::clone(&counter));
    let b = JournalCoordinator::with_shared_sequence(journal_b, &scan_b, Arc::clone(&counter));

    // Interleave commits; the shared counter hands out 1, 2, 3 so each
    // segment skips the values the other consumed.
    a.publish_group(&[put(1, 1, b"a1")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    a.sync().unwrap();
    b.publish_group(&[put(2, 1, b"b1")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    b.sync().unwrap();
    a.publish_group(&[put(1, 2, b"a2")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    a.sync().unwrap();

    drop(a);
    drop(b);
    // Segment A holds sequences 1 and 3; recovery must accept the gap.
    let (_journal, scan) = Journal::open(&path_a).unwrap();
    assert_eq!(
        scan.groups
            .iter()
            .map(|group| group.sequence)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    let (_journal, scan) = Journal::open(&path_b).unwrap();
    assert_eq!(
        scan.groups
            .iter()
            .map(|group| group.sequence)
            .collect::<Vec<_>>(),
        vec![2]
    );
    fs::remove_dir_all(dir).unwrap();
}

/// Block until the committer has parked at least once since `parks_before`,
/// then until it has released `durable_lock` into its wait, so a following
/// publish cannot race the committer's entry-time pending check.
fn wait_parked(coordinator: &Arc<JournalCoordinator>, parks_before: u64) {
    wait_until(
        {
            let coordinator = coordinator.clone();
            move || coordinator.committer_parks.load(Ordering::Acquire) > parks_before
        },
        "committer to park",
    );
    // The park counter is bumped under `durable_lock`; taking the lock here
    // returns only once the committer is inside its condvar wait.
    drop(coordinator.durable_lock.lock());
}

fn open_arc(label: &str) -> Arc<JournalCoordinator> {
    let path = temp_path(label);
    let _ = fs::remove_file(&path);
    let (journal, scan) = Journal::open(&path).unwrap();
    Arc::new(JournalCoordinator::new(journal, &scan))
}

fn wait_until(mut condition: impl FnMut() -> bool, label: &str) {
    let start = std::time::Instant::now();
    let deadline = start.checked_add(Duration::from_secs(5)).unwrap_or(start);
    while !condition() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {label}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn durability_stats_show_requests_sharing_few_fsyncs() {
    const WRITERS: u8 = 16;
    let coordinator = open_arc("dur_stats_shared");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_millis(200),
            max_pending: u64::MAX,
        })
        .unwrap();

    let mut handles = Vec::new();
    for node in 0..WRITERS {
        let coordinator = coordinator.clone();
        handles.push(std::thread::spawn(move || {
            let lsn = coordinator
                .publish_group(&[put(3, node, b"shared")])
                .map(|receipt| receipt.last_lsn)
                .unwrap();
            let token = coordinator.request_durable(lsn);
            coordinator.wait_durable(token).unwrap();
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let stats = coordinator.durability_stats();
    assert_eq!(stats.durable_requests, u64::from(WRITERS));
    assert_eq!(
        stats.durable_wait.calls + stats.durable_waits_already_durable,
        u64::from(WRITERS),
        "every wait is either blocked or already durable"
    );
    assert_eq!(
        stats.durable_wait.buckets.iter().sum::<u64>(),
        stats.durable_wait.calls
    );
    assert_eq!(stats.commit_records, u64::from(WRITERS));
    assert!(
        stats.commits >= 1 && stats.commits < u64::from(WRITERS),
        "{WRITERS} requests must share fewer fsyncs, saw {}",
        stats.commits
    );
    assert!(stats.max_commit_records > 1);
    assert!(stats.records_per_commit() > 1.0);
    assert!(
        stats.durable_wait.max
            >= stats.durable_wait.total / u32::try_from(stats.durable_wait.calls.max(1)).unwrap()
    );
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn durability_stats_count_already_durable_and_explicit_sync() {
    let coordinator = open_arc("dur_stats_explicit");
    let lsn = coordinator
        .publish_group(&[put(4, 1, b"one")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    let before = coordinator.durability_stats();
    assert_eq!(before.commits, 0);

    coordinator.sync_through(lsn).unwrap();
    let token = coordinator.request_durable(lsn);
    coordinator.wait_durable(token).unwrap();
    // A second barrier on the same LSN is covered, not a new fsync.
    coordinator.sync_through(lsn).unwrap();

    let stats = coordinator.durability_stats();
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.commit_records, 1);
    assert_eq!(stats.durable_requests, 1);
    assert_eq!(stats.durable_waits_already_durable, 1);
    assert_eq!(stats.durable_wait.calls, 0);
    assert_eq!(stats.sync_requests, 2);
    assert_eq!(stats.sync_coalesced, 1);
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn background_committer_batches_sequential_requests_into_one_commit() {
    let coordinator = open_arc("bg_batch");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_millis(750),
            max_pending: u64::MAX,
        })
        .unwrap();

    let mut token = None;
    for node in 0..8u8 {
        let lsn = coordinator
            .publish_group(&[put(1, node, b"batch")])
            .map(|receipt| receipt.last_lsn)
            .unwrap();
        token = Some(coordinator.request_durable(lsn));
    }
    let token = token.unwrap();
    coordinator.wait_durable(token).unwrap();

    assert!(token.is_satisfied_by(coordinator.committed_lsn()));
    assert_eq!(coordinator.pending_count(), 0);
    // `background_commits` is bumped after the committer's sync returns,
    // which is after the durable boundary a waiter observes. Wait for the
    // counter instead of racing it.
    let counted = coordinator.clone();
    wait_until(
        move || counted.background_commits() >= 1,
        "background commit to be counted",
    );
    assert_eq!(
        coordinator.background_commits(),
        1,
        "sequential requests inside one window must coalesce into one group"
    );
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn background_committer_flushes_a_quiet_stream_on_the_interval() {
    let coordinator = open_arc("bg_quiet");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_millis(100),
            max_pending: u64::MAX,
        })
        .unwrap();

    let lsn = coordinator
        .publish_group(&[put(2, 1, b"quiet")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    // No wait, no further request: the interval timer alone must flush it.
    let committed = coordinator.clone();
    wait_until(
        move || committed.committed_lsn() >= lsn,
        "quiet-stream interval flush",
    );
    // The committer bumps `background_commits` after its sync returns,
    // which is after the durable boundary observed above.
    let counted = coordinator.clone();
    wait_until(
        move || counted.background_commits() >= 1,
        "background commit to be counted",
    );
    // Idle timer ticks after the flush must not register as coalesced work.
    assert_eq!(coordinator.background_coalesced(), 0);
    let parks_before = coordinator.committer_parks.load(Ordering::Acquire);
    wait_until(
        {
            let coordinator = coordinator.clone();
            move || coordinator.committer_parks.load(Ordering::Acquire) >= parks_before + 2
        },
        "two idle interval ticks",
    );
    assert_eq!(
        coordinator.background_coalesced(),
        0,
        "idle ticks must not inflate the coalescing counter"
    );
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn background_committer_flushes_early_at_max_pending() {
    let coordinator = open_arc("bg_threshold");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: 4,
        })
        .unwrap();

    // Publish only: no request_durable call, so the threshold crossing in
    // `publish` itself must be what wakes the committer.
    for node in 0..6u8 {
        coordinator
            .publish_group(&[put(3, node, b"burst")])
            .map(|receipt| receipt.last_lsn)
            .unwrap();
    }
    // The 60s interval cannot be the trigger; the pending bound must be.
    // The wake fires on the fourth publish and the committer flushes
    // through whatever is published by then, so later publishes may wait
    // for the next crossing or the interval; the bound's worth must not.
    let committed = coordinator.clone();
    wait_until(
        move || committed.committed_lsn() >= 4,
        "max-pending early flush",
    );
    // The committer bumps `background_commits` after its sync returns,
    // which is after the durable boundary observed above.
    let counted = coordinator.clone();
    wait_until(
        move || counted.background_commits() >= 1,
        "background commit to be counted",
    );
    assert_eq!(coordinator.background_coalesced(), 0);
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn background_committer_rejects_degenerate_config() {
    let coordinator = open_arc("bg_bad_config");
    let zero_interval = coordinator.start_background_committer(GroupCommitConfig {
        interval: Duration::ZERO,
        max_pending: 1,
    });
    assert_eq!(
        zero_interval.unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
    let zero_pending = coordinator.start_background_committer(GroupCommitConfig {
        interval: Duration::from_secs(1),
        max_pending: 0,
    });
    assert_eq!(
        zero_pending.unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
    assert!(!coordinator.has_background_committer());
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn stop_background_committer_flushes_pending_before_returning() {
    let coordinator = open_arc("bg_stop_flush");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: u64::MAX,
        })
        .unwrap();

    let lsn = coordinator
        .publish_group(&[put(4, 1, b"final")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    let _ = coordinator.request_durable(lsn);
    coordinator.stop_background_committer().unwrap();

    assert!(
        coordinator.committed_lsn() >= lsn,
        "stop must flush the final pending group"
    );
    assert!(!coordinator.has_background_committer());
    wait_until(
        || coordinator.background_commits() >= 1,
        "background commit to be counted after stop",
    );
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn explicit_sync_through_remains_immediate_with_committer_running() {
    let coordinator = open_arc("bg_explicit_sync");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: u64::MAX,
        })
        .unwrap();

    let lsn = coordinator
        .publish_group(&[put(5, 1, b"explicit")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    // The explicit barrier must not wait for the committer's 60s interval:
    // the committer cannot fire early here (max_pending is unbounded), so a
    // barrier that returns promptly and satisfies `lsn` can only have done
    // the commit itself.
    let started = std::time::Instant::now();
    coordinator.sync_through(lsn).unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "explicit barrier must not block on the background interval"
    );
    assert!(coordinator.committed_lsn() >= lsn);
    assert_eq!(
        coordinator.background_commits(),
        0,
        "an explicit barrier is not a background commit"
    );
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn wait_durable_without_committer_commits_directly() {
    let coordinator = open_arc("no_committer_wait");
    assert!(!coordinator.has_background_committer());
    let lsn = coordinator
        .publish_group(&[put(6, 1, b"direct")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    let token = coordinator.request_durable(lsn);
    coordinator.wait_durable(token).unwrap();
    assert!(coordinator.committed_lsn() >= lsn);
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn durability_token_reports_an_already_committed_boundary() {
    let coordinator = open_arc("token_committed");
    let lsn = coordinator
        .publish_group(&[put(7, 1, b"done")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync().unwrap();
    let token = coordinator.request_durable(lsn);
    assert!(token.is_satisfied_by(coordinator.committed_lsn()));
    assert_eq!(token.lsn(), lsn);
}

#[test]
fn explicit_sync_advances_the_epoch_and_rearms_the_wake() {
    use std::sync::atomic::Ordering;

    let coordinator = open_arc("bg_epoch_explicit");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: 4,
        })
        .unwrap();
    // Park first: on entry the committer checks pending itself, which would
    // flush the burst below and mask a missing epoch bump.
    wait_parked(&coordinator, 0);

    // Pretend a wake was already claimed for the current epoch. From here
    // only a commit that advances the epoch can re-arm the crossing; a
    // burst alone must not wake the committer.
    coordinator.threshold_notified_epoch.store(
        coordinator.commit_epoch.load(Ordering::Acquire),
        Ordering::Release,
    );

    let lsn = coordinator
        .publish_group(&[put(1, 10, b"mid")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    coordinator.sync_through(lsn).unwrap();
    assert_eq!(coordinator.background_commits(), 0);
    assert_eq!(coordinator.threshold_wakes.load(Ordering::Relaxed), 0);

    // The explicit commit above advanced the epoch, so this burst wakes it.
    for node in 20..24u8 {
        coordinator
            .publish_group(&[put(1, node, b"second")])
            .map(|receipt| receipt.last_lsn)
            .unwrap();
    }
    wait_until(
        {
            let committed = coordinator.clone();
            move || committed.background_commits() >= 1
        },
        "background flush after an explicit commit re-armed the wake",
    );

    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn threshold_wakes_again_after_a_background_flush() {
    let coordinator = open_arc("bg_epoch_background");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: 4,
        })
        .unwrap();

    for (burst, expected) in (0u64..2).enumerate() {
        let base = u8::try_from(burst).unwrap() * 10;
        for node in 0..4u8 {
            coordinator
                .publish_group(&[put(2, base + node, b"burst")])
                .map(|receipt| receipt.last_lsn)
                .unwrap();
        }
        wait_until(
            {
                let committed = coordinator.clone();
                let want = expected + 1;
                move || committed.background_commits() >= want
            },
            "background burst flush",
        );
    }
    assert_eq!(coordinator.background_commits(), 2);

    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn commit_leaving_a_backlog_wakes_a_parked_committer() {
    use std::sync::atomic::Ordering;

    let coordinator = open_arc("bg_epoch_backlog");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: 4,
        })
        .unwrap();
    // Let the committer park so that only an explicit wake can move it.
    wait_parked(&coordinator, 0);

    // Suppress the publish wake for this epoch, modelling publishers that
    // arrived during an fsync and read the pre-commit epoch.
    coordinator.threshold_notified_epoch.store(
        coordinator.commit_epoch.load(Ordering::Acquire),
        Ordering::Release,
    );
    let mut lsns = Vec::new();
    for node in 0..4u8 {
        lsns.push(
            coordinator
                .publish_group(&[put(3, node, b"window")])
                .map(|receipt| receipt.last_lsn)
                .unwrap(),
        );
    }
    // `publish` computes its wake inline, so this is exact: the suppressed
    // burst selected none. (`background_commits` alone could not prove that, as
    // a wrongly woken committer needs time to flush.)
    assert_eq!(
        coordinator.threshold_wakes.load(Ordering::Relaxed),
        0,
        "a suppressed burst must not select a threshold wake"
    );
    assert_eq!(
        coordinator.background_commits(),
        0,
        "a suppressed burst must not flush before the explicit commit"
    );

    // Commit the first half explicitly while publishers add four more
    // during the fsync (the journal lock is released for it). Those four
    // arrive after the sync captured its range, so four records remain
    // pending at the bound; the post-commit recheck, not a publish, must
    // wake the committer.
    let publisher = coordinator.clone();
    let during_fsync = std::sync::Once::new();
    *coordinator.fsync_hook.lock() = Some(Arc::new(move || {
        during_fsync.call_once(|| {
            for node in 4..8u8 {
                publisher.publish_group(&[put(3, node, b"window")]).unwrap();
            }
        });
        Ok(())
    }));
    coordinator.sync_through(lsns[3]).unwrap();
    *coordinator.fsync_hook.lock() = None;
    wait_until(
        {
            let committed = coordinator.clone();
            move || committed.background_commits() >= 1
        },
        "post-commit backlog wake",
    );
    assert_eq!(coordinator.pending_count(), 0);

    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn start_during_stopping_is_a_noop() {
    let coordinator = open_arc("bg_start_stopping");
    // Simulate the join window: the slot is `Stopping` while the old worker
    // exits, before `stop_background_committer` has joined it.
    *coordinator.background.lock() = BackgroundState::Stopping;

    // The exiting worker must leave the transition to the stopper.
    coordinator.finish_background(None);
    assert!(matches!(
        *coordinator.background.lock(),
        BackgroundState::Stopping
    ));

    // A concurrent start must not spawn a second worker.
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: u64::MAX,
        })
        .unwrap();
    assert!(matches!(
        *coordinator.background.lock(),
        BackgroundState::Stopping
    ));
    assert!(!coordinator.has_background_committer());
    assert_eq!(coordinator.background_commits(), 0);

    // The stop path owns `Stopping -> Stopped` and still completes.
    coordinator.stop_background_committer().unwrap();
    assert!(matches!(
        *coordinator.background.lock(),
        BackgroundState::Stopped
    ));
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn worker_failure_is_surfaced_and_cleared_by_restart() {
    use std::sync::atomic::Ordering;

    let coordinator = open_arc("bg_failure_restart");
    coordinator.finish_background(Some(BackgroundFailure {
        kind: std::io::ErrorKind::Other,
        message: "boom".to_owned(),
    }));
    assert_eq!(
        coordinator.background_failure_message().as_deref(),
        Some("boom")
    );

    // A waiter must observe the terminal failure rather than block. The
    // failure check precedes publication validation by contract, so even a
    // never-published target reports the worker's failure, not InvalidInput.
    let error = coordinator
        .wait_durable(coordinator.request_durable(999))
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert!(error.to_string().contains("boom"));

    // A worker that died before committing can leave the notification epoch
    // equal to the current commit epoch. Without the start reset, the first
    // threshold wake after restart would be suppressed until the timer.
    coordinator.threshold_notified_epoch.store(
        coordinator.commit_epoch.load(Ordering::Acquire),
        Ordering::Release,
    );
    let parks_before = coordinator.committer_parks.load(Ordering::Acquire);
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: 1,
        })
        .unwrap();
    // Let the committer park before publishing: its entry-time pending check
    // would otherwise flush the burst and mask a suppressed wake.
    wait_parked(&coordinator, parks_before);
    assert!(coordinator.background_failure_detail().is_none());
    let lsn = coordinator
        .publish_group(&[put(5, 1, b"restart")])
        .map(|receipt| receipt.last_lsn)
        .unwrap();
    wait_until(
        {
            let committed = coordinator.clone();
            move || committed.committed_lsn() >= lsn
        },
        "first threshold wake after a restart",
    );
    coordinator
        .wait_durable(coordinator.request_durable(lsn))
        .unwrap();
    assert!(coordinator.committed_lsn() >= lsn);

    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn wait_durable_unpublished_target_has_a_stable_error() {
    // Without a committer the call falls through to `sync_through`.
    let coordinator = open_arc("bg_unpublished_none");
    let error = coordinator
        .wait_durable(coordinator.request_durable(1234))
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("has not been published"));
    fs::remove_file(coordinator.path()).unwrap();

    // With a committer the unpublished target is rejected under the lock.
    let coordinator = open_arc("bg_unpublished_committer");
    coordinator
        .start_background_committer(GroupCommitConfig {
            interval: Duration::from_secs(60),
            max_pending: u64::MAX,
        })
        .unwrap();
    let error = coordinator
        .wait_durable(coordinator.request_durable(1234))
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("has not been published"));
    coordinator.stop_background_committer().unwrap();
    fs::remove_file(coordinator.path()).unwrap();
}

#[test]
fn claim_retries_when_a_commit_lands_between_claim_and_recheck() {
    let coordinator = open_arc("bg_claim_race");
    let mut fired = false;
    let won = coordinator.claim_threshold_wake_with(|| {
        if !fired {
            fired = true;
            coordinator.commit_epoch.fetch_add(1, Ordering::AcqRel);
        }
    });
    // The commit invalidated the first claim; the retry claims the new
    // epoch, so the wake is not suppressed.
    assert!(won);
    assert_eq!(
        coordinator.threshold_notified_epoch.load(Ordering::Acquire),
        coordinator.commit_epoch.load(Ordering::Acquire)
    );
    // Same epoch again: exactly one wake per epoch.
    assert!(!coordinator.claim_threshold_wake());
    fs::remove_file(coordinator.path()).unwrap();
}
/// A shared segment with tagged groups spread across the three pools.
#[cfg(feature = "multi-reader")]
fn shared_journal_with_groups(label: &str, groups: u8) -> (Journal, std::path::PathBuf) {
    use crate::layout::ShardType;
    let path = temp_path(label);
    let _ = fs::remove_file(&path);
    let (mut journal, _) = Journal::open_shared(&path).unwrap();
    for i in 0..groups {
        let first = ShardType::ALL[usize::from(i) % 3];
        let second =
            ShardType::ALL[usize::from(i).checked_add(1).expect("index fits in usize") % 3];
        journal
            .append_group_tagged_with_sequence(
                &[
                    (Some(first), put(1, i, b"first-frame")),
                    (Some(second), put(2, i, b"second-frame")),
                ],
                None,
            )
            .unwrap();
    }
    journal.make_durable().unwrap();
    (journal, path)
}

#[cfg(feature = "multi-reader")]
fn directory_of_file(path: &std::path::Path) -> Vec<super::GroupMark> {
    super::marks_from_scan(&Journal::scan_read_only(path).unwrap())
}

/// The directory built on append, and the one built when reopening, must
/// both equal what a scan of the file finds.
#[cfg(feature = "multi-reader")]
#[test]
fn group_directory_matches_a_scan_of_the_file() {
    let (journal, path) = shared_journal_with_groups("dir_matches_scan", 9);
    assert!(journal.directory_matches_file());
    assert_eq!(journal.groups, directory_of_file(&path));
    drop(journal);
    let (reopened, _) = Journal::open_shared(&path).unwrap();
    assert!(reopened.directory_matches_file());
    assert_eq!(reopened.groups, directory_of_file(&path));
    fs::remove_file(&path).unwrap();
}

/// Reclaiming through the directory must leave exactly the bytes the
/// full-scan reclaim leaves, report the same result, and keep appends and a
/// later reclaim working.
#[cfg(feature = "multi-reader")]
#[test]
fn directory_reclaim_writes_the_same_file_as_the_scan_reclaim() {
    // Groups carry two LSNs each, so LSN 7 ends the fourth group.
    for covered in [0_u64, 1, 6, 7, 8, 17, 18] {
        let (mut by_directory, dir_path) =
            shared_journal_with_groups(&format!("reclaim_dir_{covered}"), 9);
        let (mut by_scan, scan_path) =
            shared_journal_with_groups(&format!("reclaim_scan_{covered}"), 9);
        let expected = by_scan.reclaim_through_scan(covered).unwrap();
        let actual = by_directory.reclaim_through(covered).unwrap();
        assert_eq!(
            (actual.retained_groups, actual.reclaimed_bytes),
            (expected.retained_groups, expected.reclaimed_bytes),
            "covered {covered}"
        );
        assert_eq!(
            fs::read(&dir_path).unwrap(),
            fs::read(&scan_path).unwrap(),
            "covered {covered}: reclaimed file"
        );
        assert_eq!(by_directory.groups, by_scan.groups, "covered {covered}");
        assert!(by_directory.directory_matches_file());
        assert_eq!(by_directory.groups, directory_of_file(&dir_path));

        // The handle keeps working: append to both, then reclaim again.
        for journal in [&mut by_directory, &mut by_scan] {
            journal
                .append_group_tagged_with_sequence(
                    &[(Some(crate::layout::ShardType::State), put(3, 9, b"after"))],
                    None,
                )
                .unwrap();
            journal.make_durable().unwrap();
        }
        assert_eq!(
            fs::read(&dir_path).unwrap(),
            fs::read(&scan_path).unwrap(),
            "covered {covered}: after append"
        );
        let next_covered = by_directory.next_lsn.saturating_sub(2);
        let again_expected = by_scan.reclaim_through_scan(next_covered).unwrap();
        let again_actual = by_directory.reclaim_through(next_covered).unwrap();
        assert_eq!(
            (again_actual.retained_groups, again_actual.reclaimed_bytes),
            (
                again_expected.retained_groups,
                again_expected.reclaimed_bytes
            ),
            "covered {covered}: second"
        );
        assert_eq!(
            fs::read(&dir_path).unwrap(),
            fs::read(&scan_path).unwrap(),
            "covered {covered}: second reclaim"
        );
        drop((by_directory, by_scan));
        // A reopen after the reclaim recovers the same groups.
        let (reopened, _) = Journal::open_shared(&dir_path).unwrap();
        assert_eq!(reopened.groups, directory_of_file(&dir_path));
        fs::remove_file(&dir_path).unwrap();
        fs::remove_file(&scan_path).unwrap();
    }
}

/// A directory that disagrees with the file must not be trusted: reclaim
/// falls back to the scan, gets the right answer, and repairs the directory.
#[cfg(feature = "multi-reader")]
#[test]
fn a_stale_directory_falls_back_to_the_scan() {
    let (mut journal, path) = shared_journal_with_groups("dir_stale", 6);
    journal.groups.pop();
    assert!(!journal.directory_matches_file());
    let reclaimed = journal.reclaim_through(5).unwrap();
    assert_eq!(reclaimed.retained_groups, 4);
    assert!(journal.directory_matches_file());
    assert_eq!(journal.groups, directory_of_file(&path));
    fs::remove_file(&path).unwrap();
}

/// The boundary the directory yields must equal the one a scan of the file
/// yields, for every coverage combination.
#[cfg(feature = "multi-reader")]
#[test]
fn directory_boundary_equals_the_scan_boundary() {
    use crate::layout::ShardType;
    use std::collections::HashMap;
    let (journal, path) = shared_journal_with_groups("dir_boundary", 12);
    let scan = Journal::scan_read_only(&path).unwrap();
    let by_scan = |covered: &HashMap<ShardType, u64>| {
        let mut boundary = None;
        'groups: for group in &scan.groups {
            for entry in &group.entries {
                match entry.pool {
                    Some(pool) if covered.get(&pool).is_some_and(|lsn| *lsn >= group.last_lsn) => {}
                    _ => break 'groups,
                }
            }
            boundary = Some(group.last_lsn);
        }
        boundary
    };
    let levels = [None, Some(0_u64), Some(6), Some(13), Some(24)];
    for state in levels {
        for event_dag in levels {
            for edges in levels {
                let mut covered = HashMap::new();
                for (pool, level) in [
                    (ShardType::State, state),
                    (ShardType::EventDag, event_dag),
                    (ShardType::Edges, edges),
                ] {
                    if let Some(lsn) = level {
                        covered.insert(pool, lsn);
                    }
                }
                let from_directory = match journal.shared_reclaim_boundary(&covered) {
                    super::SharedBoundary::Through(lsn, _) => Some(lsn),
                    super::SharedBoundary::NothingCovered | super::SharedBoundary::Untrusted => {
                        None
                    }
                };
                assert_eq!(from_directory, by_scan(&covered), "coverage {covered:?}");
            }
        }
    }
    fs::remove_file(&path).unwrap();
}
/// A frame with no pool tag cannot be attributed to any pool's coverage, so
/// the boundary must stop before its group even when every tagged pool has
/// reported past it.
#[cfg(feature = "multi-reader")]
#[test]
fn an_untagged_frame_stops_the_boundary() {
    use super::{boundary_through, GroupMark, UNATTRIBUTED_POOL_BIT};
    use crate::layout::ShardType;
    use std::collections::HashMap;
    let mark = |last_lsn: u64, pools: u8| GroupMark {
        sequence: last_lsn,
        first_lsn: last_lsn,
        last_lsn,
        end_offset: last_lsn * 100,
        pools,
    };
    let state = 1_u8;
    let groups = [
        mark(1, state),
        mark(2, state | UNATTRIBUTED_POOL_BIT),
        mark(3, state),
    ];
    let covered = HashMap::from([(ShardType::State, 99_u64)]);
    assert_eq!(boundary_through(&groups, &covered), Some((1, None)));
    assert_eq!(boundary_through(&groups[1..], &covered), None);
    assert_eq!(boundary_through(&groups[2..], &covered), Some((3, None)));
    // A tagged pool that has not reported blocks its group.
    assert_eq!(boundary_through(&groups[..1], &HashMap::new()), None);
    // With a prefix covered, the blocking pool is named: group 2 carries a
    // State frame whose coverage (1) does not reach it, so State blocks.
    let partial = HashMap::from([(ShardType::State, 1_u64)]);
    assert_eq!(
        boundary_through(&groups, &partial),
        Some((1, Some(ShardType::State)))
    );
    // With State covered through group 2, only the untagged frame blocks it,
    // and there is no pool to name.
    let through_two = HashMap::from([(ShardType::State, 2_u64)]);
    assert_eq!(boundary_through(&groups, &through_two), Some((1, None)));
}

/// A directory whose offsets make the retained suffix start mid-group must
/// not fail the reclaim: the suffix check errors, and the scan path decides.
#[cfg(feature = "multi-reader")]
#[test]
fn a_misaligned_directory_falls_back_to_the_scan_instead_of_failing() {
    let (mut broken, broken_path) = shared_journal_with_groups("dir_misaligned", 6);
    let (mut reference, reference_path) = shared_journal_with_groups("dir_reference", 6);
    // Keep the last group ending at the file length (so the directory is
    // trusted) but point an earlier group's end into the middle of a group.
    let true_end = broken.groups[1].end_offset;
    assert!(
        directory_of_file(&broken_path)
            .iter()
            .any(|mark| mark.end_offset == true_end),
        "groups[1] must end on a real group boundary before it is corrupted"
    );
    broken.groups[1].end_offset += 1;
    assert!(
        !directory_of_file(&broken_path)
            .iter()
            .any(|mark| mark.end_offset == broken.groups[1].end_offset),
        "the corrupted end must not be a real group boundary"
    );
    assert!(broken.directory_matches_file());
    let actual = broken.reclaim_through(5).unwrap();
    let expected = reference.reclaim_through_scan(5).unwrap();
    assert_eq!(
        (actual.retained_groups, actual.reclaimed_bytes),
        (expected.retained_groups, expected.reclaimed_bytes)
    );
    assert_eq!(
        fs::read(&broken_path).unwrap(),
        fs::read(&reference_path).unwrap()
    );
    assert!(broken.directory_matches_file());
    assert_eq!(broken.groups, directory_of_file(&broken_path));
    fs::remove_file(&broken_path).unwrap();
    fs::remove_file(&reference_path).unwrap();
}
/// The directory's accounting follows appends and reclaims, and its worst
/// case is what the format allows.
#[cfg(feature = "multi-reader")]
#[test]
fn group_directory_accounting_follows_appends_and_reclaims() {
    use super::{GroupDirectoryStats, GroupMark};
    let (journal, path) = shared_journal_with_groups("dir_accounting", 9);
    let stats = journal.directory_stats();
    assert_eq!(stats.groups, 9);
    assert!(stats.allocated_bytes >= 9 * std::mem::size_of::<GroupMark>() as u64);
    drop(journal);
    let (mut journal, _) = Journal::open_shared(&path).unwrap();
    assert_eq!(journal.directory_stats().groups, 9);
    journal.reclaim_through(10).unwrap();
    assert_eq!(journal.directory_stats().groups, 4);
    // A segment full of the smallest possible groups: a bounded, modest
    // number of entries.
    assert!(GroupDirectoryStats::worst_case_groups() > 0);
    assert!(GroupDirectoryStats::worst_case_bytes() < (1 << 30));
    fs::remove_file(&path).unwrap();
}
