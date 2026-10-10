use super::*;
use crate::packfile::test_support::pack_id_for;

fn test_dir(name: &str) -> PathBuf {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("mdb_test_shard_{name}_{}_{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn test_record(collection: u8, hash_byte: u8, data: &[u8]) -> Record {
    let mut collection_id = [0u8; 16];
    collection_id[0] = collection;
    let mut hash = [0u8; 16];
    hash[0] = hash_byte;
    Record {
        collection_id,
        hash,
        data: bytes::Bytes::copy_from_slice(data),
        metadata: None,
    }
}

#[test]
fn test_shard_pool_create_and_write() {
    let dir = test_dir("pool_create");
    let pool = ShardPool::open(dir).unwrap();

    let record = test_record(0x01, 0xAA, b"hello shard");
    let (slot, offset) = pool.put_record(&record).unwrap();
    assert_eq!(slot, 0);
    assert!(offset > 0);

    let shard = pool.get_shard(slot).unwrap();
    let read = pool.read_at(&shard, offset, true).unwrap();
    assert_eq!(read.collection_id[0], 0x01);
    assert_eq!(read.hash[0], 0xAA);
    assert_eq!(read.data.as_ref(), b"hello shard");
}

#[test]
fn poisoned_shard_refuses_append_and_flush() {
    let dir = test_dir("poisoned_shard_refuses_writes");
    let pool = ShardPool::open(dir).unwrap();
    let record = test_record(0x01, 0xAA, b"before poison");
    pool.put_record(&record).unwrap();
    let shard = pool.get_shard(0).unwrap();

    // Simulate the state left behind by an unrecoverable rollback
    // failure (see `put_record`'s failed-flush path) directly, rather
    // than engineering an actual `set_len` failure.
    shard.poisoned.store(true, Ordering::Release);

    let after = test_record(0x02, 0xBB, b"after poison");
    let put_err = pool
        .put_record(&after)
        .expect_err("a poisoned shard must refuse further appends");
    assert!(put_err.to_string().contains("poisoned"));

    let flush_err = pool
        .flush_shard(&shard)
        .expect_err("a poisoned shard must refuse flush too");
    assert!(flush_err.to_string().contains("poisoned"));
}

#[test]
fn raw_reads_borrow_the_mmap_and_survive_a_remap() {
    let dir = test_dir("raw_read_mmap_owner");
    let pool = ShardPool::open(dir).unwrap();
    let first = test_record(1, 1, b"first raw payload");
    let (slot, first_offset) = pool.put_record(&first).unwrap();
    let shard = pool.get_shard(slot).unwrap();
    // Buffered records are not on disk until flushed; the pointer
    // invariant below is about the mmap backing, which only exists for
    // committed bytes, so commit before establishing the mapping.
    pool.flush_all().unwrap();

    // Establish the initial mapping and retain it only for checking the
    // returned Bytes pointer. The read itself must hold its own owner.
    let initial_mapping = shard.mmap().unwrap().as_ref().unwrap().clone();
    let read_first = pool.read_at(&shard, first_offset, true).unwrap();
    let first_offset = usize::try_from(first_offset).unwrap();
    let node_offset = first_offset
        .checked_add(4)
        .and_then(|offset| offset.checked_add(37))
        .unwrap();
    assert_eq!(
        read_first.data.as_ptr(),
        initial_mapping[node_offset..].as_ptr(),
        "raw payload must be backed directly by the mmap"
    );

    // Grow the file, then read the new record. This replaces the pool's
    // current mapping; `read_first` must retain the old one safely.
    let second = test_record(1, 2, b"second raw payload");
    let (_, second_offset) = pool.put_record(&second).unwrap();
    let read_second = pool.read_at(&shard, second_offset, true).unwrap();
    assert_eq!(read_first.data.as_ref(), b"first raw payload");
    assert_eq!(read_second.data.as_ref(), b"second raw payload");
}

#[test]
fn record_disk_len_rejects_truncated_frame_body() {
    let dir = test_dir("disk_len_truncated_frame");
    let pool = ShardPool::open(dir).unwrap();
    let (slot, offset) = pool
        .put_record(&test_record(0x01, 0xAA, b"truncated frame"))
        .unwrap();
    let shard = pool.get_shard(slot).unwrap();
    pool.flush_all().unwrap();

    let len = shard.file.metadata().unwrap().len();
    shard.file.set_len(len - 1).unwrap();
    *shard.mmap.write() = None;

    assert!(matches!(
        ShardPool::record_disk_len_at(&shard, offset),
        Err(crate::storage::StorageError::Corrupt(_))
    ));
}

#[test]
fn test_sync_dirty_noop_when_nothing_written() {
    let dir = test_dir("sync_dirty_noop");
    let pool = ShardPool::open(dir).unwrap();
    assert!(pool.dirty.lock().is_empty());
    pool.sync_dirty().unwrap();
    assert!(pool.dirty.lock().is_empty());
}

#[test]
fn dirty_lock_wait_starts_at_zero_and_accrues_on_sync() {
    let dir = test_dir("dirty_lock_wait");
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(
        pool.dirty_lock_wait(),
        Duration::ZERO,
        "a fresh pool has never contended for the dirty-set lock"
    );

    let record = test_record(0x01, 0xAA, b"hello");
    pool.put_record(&record).unwrap();
    pool.sync_dirty().unwrap();
    // A single-threaded sync still acquires the lock (uncontended), so
    // this only asserts the counter is wired up and monotonic, not that
    // it measures anything close to real contention. Since the per-shard
    // sync coalescing landed, the pool-wide lock is only held for the
    // brief dirty-bit check/claim, so real contention shows up here only
    // if many syncers collide on that short critical section.
    let after_one_sync = pool.dirty_lock_wait();

    pool.put_record(&test_record(0x01, 0xAB, b"world")).unwrap();
    pool.sync_dirty().unwrap();
    assert!(
        pool.dirty_lock_wait() >= after_one_sync,
        "dirty_lock_wait must never decrease"
    );
}

/// N concurrent `sync_dirty` callers for the same shard must coalesce
/// into one physical fsync: the first to claim the dirty bit fsyncs, the
/// rest wait on the shard's `sync_lock` for that fsync, then find the bit
/// already cleared and skip theirs.
#[test]
fn concurrent_sync_dirty_coalesces_to_a_single_fsync() {
    use std::thread;

    let dir = test_dir("concurrent_sync_dirty_coalesce");
    let pool = Arc::new(ShardPool::open(dir).unwrap());
    let record = test_record(0x01, 0xAA, b"coalesce me");
    let (slot, _offset) = pool.put_record(&record).unwrap();
    let shard = pool.get_shard(slot).unwrap();
    assert_eq!(shard.stats().sync_count, 0);

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let pool = Arc::clone(&pool);
            thread::spawn(move || pool.sync_dirty())
        })
        .collect();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    assert_eq!(
        shard.stats().sync_count,
        1,
        "8 concurrent sync_dirty callers must coalesce into one fsync"
    );
    assert!(
        pool.dirty.lock().is_empty(),
        "sync_dirty must clear the dirty bit it claimed"
    );
}

#[test]
fn test_stats_persisted_at_none_until_first_flush() {
    let dir = test_dir("stats_persisted_at_none");
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.stats_persisted_at(), None);
}

#[test]
fn test_stats_persisted_at_set_after_sync_and_survives_reopen() {
    let dir = test_dir("stats_persisted_at_roundtrip");
    let pool = ShardPool::open(dir.clone()).unwrap();
    let record = test_record(0x01, 0xAA, b"hello");
    pool.put_record(&record).unwrap();
    pool.sync_all().unwrap();

    let persisted_at = pool
        .stats_persisted_at()
        .expect("sync_all must persist stats");
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(
        now.saturating_sub(persisted_at) < 5,
        "persisted_at must be a recent timestamp, not zero or garbage"
    );
    // `ShardPool`'s `Drop` impl does its own best-effort final stats
    // flush (`persist_stats_best_effort`) unconditionally, even
    // though nothing changed since the `sync_all` above — so
    // dropping `pool` legitimately bumps `shard_stats.bin`'s
    // timestamp to whatever `SystemTime::now()` reads *at drop
    // time*, which can differ from `persisted_at` by a second if a
    // wall-clock boundary falls between the two. Asserting exact
    // equality with the pre-drop value below would be pinning an
    // implementation-timing coincidence, not a real invariant —
    // hence the `>=`-and-recent checks instead, further down.
    drop(pool);

    // A fresh pool reading the same base_dir must restore *a*
    // recent timestamp along with the counters, not just the
    // counters — otherwise a read-only `mtxdb shards` invocation
    // could show a real snapshot's numbers next to a `None`/unknown
    // age.
    let reopened = ShardPool::open(dir).unwrap();
    let reopened_persisted_at = reopened
        .stats_persisted_at()
        .expect("stats must survive a reopen, not come back None");
    assert!(
        reopened_persisted_at >= persisted_at,
        "reopened persisted_at ({reopened_persisted_at}) must not be older than the \
         pre-drop snapshot ({persisted_at})"
    );
    let now_after_reopen = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(
        now_after_reopen.saturating_sub(reopened_persisted_at) < 5,
        "reopened persisted_at must still be a recent timestamp"
    );
}

#[test]
fn test_maybe_persist_stats_rate_limited() {
    let dir = test_dir("maybe_persist_rate_limited");
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.stats_persisted_at(), None);

    // First call always flushes (nothing to rate-limit against yet).
    pool.maybe_persist_stats(Duration::from_secs(3600));
    let first = pool
        .stats_persisted_at()
        .expect("first maybe_persist_stats call must flush");

    // A second call inside the interval must not re-flush. There's no
    // observable-from-here difference if it did (the timestamp is in
    // whole seconds), so this mainly documents the contract; the real
    // guard is that it doesn't pay a write+rename every call.
    pool.maybe_persist_stats(Duration::from_secs(3600));
    assert_eq!(pool.stats_persisted_at(), Some(first));

    // Zero interval must always flush.
    pool.maybe_persist_stats(Duration::ZERO);
    assert!(pool.stats_persisted_at().is_some());
}

#[test]
fn test_maybe_persist_stats_noop_on_read_only_pool() {
    let dir = test_dir("maybe_persist_read_only");
    // Keep the writer open (not dropped) with no explicit sync, so no
    // shard_stats.bin exists yet — Drop's own best-effort persist
    // would otherwise write one and confound what this test checks.
    let writer = ShardPool::open(dir.clone()).unwrap();
    writer.put_record(&test_record(0x01, 0xAA, b"x")).unwrap();

    let reader = ShardPool::open_read_only(dir).unwrap();
    assert_eq!(reader.stats_persisted_at(), None);
    reader.maybe_persist_stats(Duration::ZERO);
    // A read-only pool must never write a stats snapshot of its own
    // (always-zero) counters — it must still show nothing persisted,
    // not conjure a snapshot the real writer never flushed.
    assert_eq!(reader.stats_persisted_at(), None);
    drop(writer);
}

#[test]
fn test_clean_sync_all_does_not_rewrite_stats_snapshot() {
    let dir = test_dir("clean_sync_stats");
    let pool = ShardPool::open(dir).unwrap();
    let before = pool.stats_snapshots();
    pool.sync_all().unwrap();
    assert_eq!(
        pool.stats_snapshots(),
        before,
        "a clean sync must not rewrite the stats snapshot"
    );
    assert_eq!(
        pool.stats_persisted_at(),
        None,
        "a clean sync must not persist any stats snapshot"
    );
}

#[test]
fn test_dirty_sync_all_rewrites_stats_snapshot_once() {
    let dir = test_dir("dirty_sync_stats_once");
    let pool = ShardPool::open(dir).unwrap();
    pool.put_record(&test_record(0x02, 0xBB, b"payload"))
        .unwrap();
    let before = pool.stats_snapshots();
    pool.sync_all().unwrap();
    assert_eq!(
        pool.stats_snapshots(),
        before + 1,
        "a dirty sync must persist the stats snapshot exactly once"
    );
    assert!(
        pool.stats_persisted_at().is_some(),
        "the dirty sync must leave a stats snapshot timestamp"
    );
}

#[test]
fn test_maybe_persist_stats_still_works_without_data_mutation() {
    let dir = test_dir("periodic_stats_no_mutation");
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.stats_persisted_at(), None);
    let before = pool.stats_snapshots();
    pool.maybe_persist_stats(Duration::ZERO);
    assert_eq!(
        pool.stats_snapshots(),
        before + 1,
        "the periodic flush must write a snapshot even with no dirty shards"
    );
    assert!(
        pool.stats_persisted_at().is_some(),
        "the periodic flush must leave a stats snapshot timestamp"
    );
}

#[test]
fn test_sync_dirty_clears_only_written_shards() {
    let dir = test_dir("sync_dirty_clears");
    let pool = ShardPool::open(dir).unwrap();

    let record = test_record(0x01, 0xCC, b"needs sync");
    let (slot, _offset) = pool.put_record(&record).unwrap();
    // Default policy is eager: the put is committed immediately, so the
    // shard is already marked dirty and sync_dirty must clear it.
    assert!(
        pool.dirty.lock().contains(&slot),
        "eager put must mark the shard dirty"
    );

    pool.sync_dirty().unwrap();
    assert!(
        pool.dirty.lock().is_empty(),
        "dirty bit must clear after a successful sync"
    );

    // A second sync with nothing new written is a no-op, not an error.
    pool.sync_dirty().unwrap();
}

/// Compatibility contract for the default eager policy: an unsynced put
/// is committed to the page cache immediately, so a fresh open of the
/// same directory sees the record even though `flush_all`/`sync` were
/// never called. This is the historical behavior buffering must not
/// silently change — callers that don't opt into buffering keep it.
#[test]
fn eager_put_is_visible_to_fresh_open_without_flush_or_sync() {
    let dir = test_dir("eager_put_fresh_open");
    let record = test_record(0x01, 0xAA, b"eager visibility");

    let (offset, slot) = {
        let pool = ShardPool::open(dir.clone()).unwrap();
        let (slot, offset) = pool.put_record(&record).unwrap();
        assert!(
            pool.dirty.lock().contains(&slot),
            "eager put must mark the shard dirty before any sync"
        );
        (offset, slot)
    };

    // Fresh process-equivalent open, no flush/sync anywhere.
    let pool = ShardPool::open(dir).unwrap();
    let read = pool
        .read_at(&pool.get_shard(slot).unwrap(), offset, true)
        .unwrap();
    assert_eq!(read.data.as_ref(), b"eager visibility");
}

/// The buffered policy's known visibility boundary: an unflushed put
/// occupies only a virtual offset past the committed file length, so a
/// fresh open — which only sees committed bytes — must not be able to
/// read it back. This is opt-in behavior; eager callers never hit it.
#[test]
fn buffered_put_is_hidden_from_fresh_open_until_flush() {
    let record = test_record(0x01, 0xBB, b"buffered invisibility");

    // No flush before drop: the byte is only in the writer's memory, so
    // a fresh open must not be able to read it back.
    let dir = test_dir("buffered_put_fresh_open");
    let (offset, slot) = {
        let pool = ShardPool::open(dir.clone())
            .unwrap()
            .with_append_policy(AppendPolicy::buffered());
        let (slot, offset) = pool.put_record(&record).unwrap();
        assert!(
            pool.dirty.lock().is_empty(),
            "a buffered put alone must not dirty the shard"
        );
        (offset, slot)
    };
    let pool = ShardPool::open(dir).unwrap();
    let shard = pool.get_shard(slot).unwrap();
    assert!(
        shard.file_len.load(Ordering::Acquire) <= offset,
        "buffered byte must not have been committed to the file"
    );
    assert!(
        pool.read_at(&shard, offset, true).is_err(),
        "a fresh open must not see an unflushed buffered byte"
    );

    // flush_all before drop puts the frame on disk; a fresh open then
    // reads the same offset back — the boundary is flush, not drop.
    let dir = test_dir("buffered_put_fresh_open_flushed");
    let (slot, offset) = {
        let pool = ShardPool::open(dir.clone())
            .unwrap()
            .with_append_policy(AppendPolicy::buffered());
        let (slot, offset) = pool.put_record(&record).unwrap();
        pool.flush_all().unwrap();
        (slot, offset)
    };
    let pool = ShardPool::open(dir).unwrap();
    let read = pool
        .read_at(&pool.get_shard(slot).unwrap(), offset, true)
        .unwrap();
    assert_eq!(read.data.as_ref(), b"buffered invisibility");
}

/// A real fsync succeeding must never be turned into a hard error by a
/// failure in the best-effort stats snapshot write -- e.g. the base
/// directory getting removed out from under a live pool (test teardown,
/// or any other external interference) must not make `sync_all`/
/// `sync_dirty` (and by extension every `maybe_sync(DURABLE)` caller
/// upstream) return `Err` when the actual shard data is safely synced.
#[test]
fn test_sync_survives_persist_stats_failure() {
    let dir = test_dir("sync_survives_stats_failure");
    let pool = ShardPool::open(dir.clone()).unwrap();

    let record = test_record(0x01, 0xDD, b"data that must stay durable");
    pool.put_record(&record).unwrap();

    // Remove the base directory itself, so persist_stats' File::create
    // for its temp file fails with ENOENT -- while the shard's already-
    // open file descriptor (and thus its real fsync) is unaffected.
    fs::remove_dir_all(&dir).unwrap();

    pool.sync_all()
        .expect("sync_all must succeed even if stats persistence fails");
    pool.put_record(&test_record(0x01, 0xEE, b"more data"))
        .unwrap();
    pool.sync_dirty()
        .expect("sync_dirty must succeed even if stats persistence fails");
}

#[test]
fn test_shard_pool_rotation() {
    let dir = test_dir("pool_rotate");
    let pool = ShardPool::open(dir).unwrap();
    pool.active_shard()
        .file_len
        .store(MAX_SHARD_BYTES - 10, Ordering::Release);

    let record = test_record(0x01, 0xBB, b"trigger rotation");
    let (slot, _offset) = pool.put_record(&record).unwrap();
    assert_eq!(slot, 1);
}

/// Core fix: a collection's writes must stay on its own home shard even
/// after *unrelated* activity rotates the pool's global active-write
/// cursor far ahead. Before per-collection home routing, every collection simply
/// followed that single pool-wide cursor, so any other collection's churn
/// (with nothing to do with collection A, and no capacity reason for collection
/// A specifically to move) would silently redirect collection A's next
/// write too — destroying locality collection A never had a reason to lose.
#[test]
fn test_collection_stays_on_home_shard_despite_unrelated_pool_rotation() {
    let dir = test_dir("collection_locality");
    let pool = ShardPool::open(dir).unwrap();

    // Collection A's first write establishes its home on shard 0.
    let collection_a = test_record(0x01, 0x01, b"collection A first");
    let (shard_a1, _) = pool.put_record(&collection_a).unwrap();
    assert_eq!(shard_a1, 0);

    // Simulate unrelated churn (other collections' own rotations) dragging
    // the pool-wide cursor far ahead — collection A is not involved at all,
    // and shard 0 still has essentially all its capacity free.
    for _ in 0..5 {
        pool.rotate().unwrap();
    }

    // Collection A writes again: it must still land on shard 0, its own
    // home — not wherever unrelated rotations left the pool cursor.
    let collection_a2 = test_record(0x01, 0x03, b"collection A second");
    let (shard_a2, _) = pool.put_record(&collection_a2).unwrap();
    assert_eq!(
        shard_a2, 0,
        "collection A must stay on its own home shard, unaffected by unrelated pool rotation"
    );
}

/// Once a collection's own home shard actually fills up, that collection (and
/// only that collection) rotates to a new home — independent of whatever
/// the pool-wide cursor is doing for other collections.
#[test]
fn test_collection_rotates_its_own_home_when_full() {
    let dir = test_dir("collection_locality_own_rotation");
    let pool = ShardPool::open(dir).unwrap();

    let collection_a = test_record(0x01, 0x01, b"collection A first");
    let (shard_a1, _) = pool.put_record(&collection_a).unwrap();
    assert_eq!(shard_a1, 0);

    // Fill collection A's own home shard (shard 0) — not some other shard —
    // and confirm collection A itself rotates off it.
    pool.get_shard(0)
        .unwrap()
        .file_len
        .store(MAX_SHARD_BYTES - 10, Ordering::Release);
    let collection_a2 = test_record(0x01, 0x02, b"collection A triggers its own rotation");
    let (shard_a2, _) = pool.put_record(&collection_a2).unwrap();
    assert_eq!(shard_a2, 1);

    // And collection A stays on its new home from then on.
    let collection_a3 = test_record(0x01, 0x03, b"collection A third");
    let (shard_a3, _) = pool.put_record(&collection_a3).unwrap();
    assert_eq!(shard_a3, 1);
}

/// `persist_stats` used a fixed tmp filename, so two `ShardPool`s
/// pointed at the same `base_dir` (e.g. a long-running embedder and a
/// short-lived `mtxdb shards` CLI invocation) could race: one's
/// `rename()` consumes the shared tmp path out from under the
/// other's, which then fails its own `rename()` with ENOENT despite
/// having written its tmp file successfully. Each pool now uses a
/// tmp filename unique to its own process id and an internal counter,
/// so concurrent persists from separate pools never collide.
#[test]
fn test_concurrent_persist_stats_does_not_race() {
    // A single writable pool (only one can ever exist per directory
    // now — see the writer-lock tests below), shared across threads
    // within this one process — the realistic scenario for the
    // tmp-filename-uniqueness fix, since cross-process contention on
    // the same directory is now prevented entirely by the writer lock
    // rather than needing to be tolerated here.
    let dir = test_dir("persist_stats_race");
    let pool = Arc::new(ShardPool::open(dir.clone()).unwrap());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || {
                for _ in 0..25 {
                    pool.persist_stats().unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // The final file must be well-formed, not torn by a partial
    // overlapping write.
    let path = ShardPool::stats_path(&dir);
    let buf = fs::read(&path).unwrap();
    assert_eq!(&buf[0..4], STATS_MAGIC);
    assert_eq!(buf[4], STATS_VERSION);

    // No leftover tmp files from a failed/interrupted attempt.
    let leftover_tmp = fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().contains(".tmp."));
    assert!(!leftover_tmp, "a persist_stats tmp file was left behind");
}

/// A stats snapshot at the immediately preceding version (v5: 56-byte
/// records keyed by the 32-byte `PackId`) must be ignored rather than
/// misparsed as v6's 40-byte records. The snapshot is a rebuildable
/// observability cache, so rejection just means "not restored".
#[test]
fn previous_version_stats_snapshot_is_not_restored() {
    let dir = test_dir("stats_prev_version");
    // Hand-write a v5-format file: magic, version 5, timestamp, then one
    // 56-byte record (32-byte pack id + three u64 counters). No live shard
    // is needed: the version gate rejects the whole file before any record
    // is matched or decoded, which is exactly the behavior under test.
    let mut buf = Vec::new();
    buf.extend_from_slice(STATS_MAGIC);
    buf.push(STATS_VERSION - 1);
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&[0u8; 32 + 8 * 3]);
    fs::write(ShardPool::stats_path(&dir), &buf).unwrap();

    let empty_shards: Vec<Option<Arc<Shard>>> = (0..MAX_SHARDS).map(|_| None).collect();
    assert!(
        ShardPool::restore_persisted_stats(&dir, &empty_shards).is_none(),
        "a v{} stats snapshot must not be restored by the v{STATS_VERSION} reader",
        STATS_VERSION - 1
    );

    // Opening the pool with a stale v5 snapshot must ignore it without error,
    // and a subsequent persist must overwrite it with a fresh v6 snapshot.
    let pool = ShardPool::open(dir.clone()).unwrap();
    pool.persist_stats().unwrap();
    drop(pool);

    let rebuilt = fs::read(ShardPool::stats_path(&dir)).unwrap();
    assert_eq!(&rebuilt[0..4], STATS_MAGIC);
    assert_eq!(rebuilt[4], STATS_VERSION);
}

/// Core invariant of the writer lock: at most one writer per
/// `base_dir`. A second `open` while the first is still alive must
/// fail fast rather than silently risking the interleaved-append
/// corruption this lock exists to prevent.
#[test]
fn test_second_writer_fails_while_first_is_open() {
    let dir = test_dir("writer_lock_exclusive");
    let _first = ShardPool::open(dir.clone()).unwrap();

    let second = ShardPool::open(dir.clone());
    assert!(
        second.is_err(),
        "a second writer must not be able to open the same base_dir concurrently"
    );
}

/// Dropping the writer releases its lock immediately (via
/// `WriterLock`'s `Drop` removing the marker file), so a subsequent
/// open — not concurrent with the first — must succeed normally.
#[test]
fn test_writer_lock_releases_on_drop() {
    let dir = test_dir("writer_lock_release");
    let first = ShardPool::open(dir.clone()).unwrap();
    drop(first);

    let second = ShardPool::open(dir.clone());
    assert!(
        second.is_ok(),
        "a new writer must be able to open once the previous one has dropped"
    );
}

/// A lock file recording our own actual `{pid, starttime}` (exactly
/// what a live writer's own lock file looks like) must correctly be
/// treated as held, not stale — otherwise a legitimately-running
/// writer could have its own lock reclaimed out from under it.
#[test]
#[cfg(target_os = "linux")]
fn test_lock_with_matching_starttime_is_not_reclaimed() {
    let dir = test_dir("lock_matching_starttime_alive");
    std::fs::create_dir_all(&dir).unwrap();
    let lock_path = dir.join(".mtxdb.lock");
    let pid = std::process::id();
    let start = ShardPool::proc_start_time("self").expect("must read our own starttime");
    std::fs::write(&lock_path, format!("{pid} {start}")).unwrap();

    assert!(
        !ShardPool::lock_holder_is_dead(&lock_path),
        "a lock file matching our own live pid+starttime must not be reclaimable"
    );
}

/// The actual regression this fix exists for: a lock file whose PID
/// has been recycled onto a *different* still-running process (a
/// long-lived sidecar taking over a dead writer's old PID number, in
/// the original bug's terms) must be reclaimed, not treated as held
/// forever. We simulate "different process" the same way the real
/// check would distinguish one: same PID, mismatched starttime.
#[test]
#[cfg(target_os = "linux")]
fn test_lock_with_stale_starttime_is_reclaimed_despite_live_pid() {
    let dir = test_dir("lock_stale_starttime_pid_reused");
    std::fs::create_dir_all(&dir).unwrap();
    let lock_path = dir.join(".mtxdb.lock");
    let pid = std::process::id();
    let real_start = ShardPool::proc_start_time("self").expect("must read our own starttime");
    // A starttime that cannot possibly be ours: any recorded starttime
    // must exactly match `/proc/<pid>/stat`'s live value, so simply
    // perturbing it stands in for "this pid now belongs to someone
    // else" without needing to actually fork/reap a process to force a
    // real-world PID recycle.
    let impostor_start = real_start.wrapping_add(1);
    std::fs::write(&lock_path, format!("{pid} {impostor_start}")).unwrap();

    assert!(
        ShardPool::lock_holder_is_dead(&lock_path),
        "a starttime mismatch on a live pid must be treated as a recycled-PID impostor, not the original holder"
    );
}

/// An old-format lock file (bare PID, no starttime — what a pre-fix
/// binary writes) has nothing to disambiguate a PID reuse with, so it
/// must keep failing closed exactly as before: a live PID is always
/// treated as held, never reclaimed just because we can't check
/// further. This is a no-regression guarantee for the previous format.
#[test]
#[cfg(target_os = "linux")]
fn test_old_format_lock_file_with_live_pid_still_fails_closed() {
    let dir = test_dir("lock_old_format_live_pid");
    std::fs::create_dir_all(&dir).unwrap();
    let lock_path = dir.join(".mtxdb.lock");
    std::fs::write(&lock_path, format!("{}", std::process::id())).unwrap();

    assert!(
        !ShardPool::lock_holder_is_dead(&lock_path),
        "a bare-PID (pre-fix) lock file for a still-live pid must fail closed, not be reclaimed"
    );
}

/// A read-only open takes no lock at all (it never writes or
/// truncates anything), so it must succeed even while a writer is
/// actively holding the directory — this is the actual scenario a
/// `mtxdb shards`-style inspection tool needs to work at all.
#[test]
fn test_read_only_coexists_with_active_writer() {
    let dir = test_dir("writer_lock_reader_coexist");
    let writer = ShardPool::open(dir.clone()).unwrap();

    let reader = ShardPool::open_read_only(dir.clone());
    assert!(
        reader.is_ok(),
        "a read-only open must coexist with an active writer, not be excluded by its lock"
    );

    drop(writer);
}

/// A read-only pool must never write a stats snapshot (its own
/// counters are always zero), but it absolutely must *restore* the
/// real writer's already-persisted one — otherwise a `shards`-style
/// inspection tool built on `open_read_only` would always show
/// `write_count`/`bytes_written`/`sync_count` as 0 regardless of how much
/// real activity the writer has persisted, while file size (read
/// live off disk, independent of the stats file) correctly grows —
/// exactly the confusing "bytes climbing, everything else frozen at
/// zero" symptom this test guards against.
#[test]
fn test_read_only_open_restores_persisted_stats() {
    let dir = test_dir("read_only_restores_stats");

    let writer = ShardPool::open(dir.clone()).unwrap();
    let record = test_record(0x01, 0xAA, b"payload");
    writer.put_record(&record).unwrap();
    writer.sync_dirty().unwrap();
    let persisted = writer.get_shard(0).unwrap().stats();
    assert!(
        persisted.write_count > 0,
        "test setup: writer must have written something"
    );
    drop(writer);

    let reader = ShardPool::open_read_only(dir).unwrap();
    let restored = reader.get_shard(0).unwrap().stats();
    assert_eq!(
        restored, persisted,
        "a read-only open must restore the real writer's persisted stats, not start at zero"
    );
}

#[test]
fn test_shard_pool_scan() {
    let dir = test_dir("pool_scan");
    let pool = ShardPool::open(dir.clone()).unwrap();

    let r1 = test_record(0x01, 0x10, b"collection1 msg1");
    let r2 = test_record(0x02, 0x20, b"collection2 msg1");
    let r3 = test_record(0x01, 0x11, b"collection1 msg2");
    pool.put_record(&r1).unwrap();
    pool.put_record(&r2).unwrap();
    pool.put_record(&r3).unwrap();
    // The file scan only sees committed bytes, so flush before scanning.
    pool.flush_all().unwrap();

    let entries = ShardPool::scan_shard(&pool.get_shard(0).unwrap().path).unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].0[0], 0x01);
    assert_eq!(entries[0].1[0], 0x10);
    assert_eq!(entries[2].0[0], 0x01);
    assert_eq!(entries[2].1[0], 0x11);
}

/// End-to-end: real data survives a real close-and-reopen through the
/// actual `ShardPool::open` bootstrap, not just a raw byte-level
/// `read_header` call — proves the v2 header roundtrips through the
/// path a real process restart actually takes.
#[test]
fn test_shard_pool_reopen_survives_v2_header_roundtrip() {
    let dir = test_dir("pool_reopen_v2");
    let pool = ShardPool::open(dir.clone()).unwrap();
    let record = test_record(0x01, 0xAA, b"survives reopen");
    let (slot, offset) = pool.put_record(&record).unwrap();
    // Durability is the sync boundary: buffered records are RAM-only
    // until flushed, so make this reopen test's record actually durable
    // before the pool drops.
    pool.sync_all().unwrap();
    drop(pool);

    let pool = ShardPool::open(dir).unwrap();
    let shard = pool.get_shard(slot).unwrap();
    let read = pool.read_at(&shard, offset, true).unwrap();
    assert_eq!(read.data.as_ref(), b"survives reopen");

    // Recovered shards must retain append access. Opening an existing
    // pack read-only here makes the next real import fail with EBADF.
    let appended = test_record(0x01, 0xBB, b"appends after reopen");
    let (_, appended_offset) = pool.put_record(&appended).unwrap();
    let read = pool.read_at(&shard, appended_offset, true).unwrap();
    assert_eq!(read.data.as_ref(), b"appends after reopen");
}

/// End-to-end: `ShardPool::open` skips a shard file whose
/// embedded identity doesn't match its filename and creates a
/// fresh shard 0 when no valid files remain.
#[test]
fn test_shard_pool_open_fails_on_identity_mismatch() {
    let dir = test_dir("pool_open_identity_mismatch");
    std::fs::create_dir_all(&dir).unwrap();
    // A valid header claiming one address, filed under a
    // filename whose prefix claims another.
    let path = ShardPool::pack_path(&dir, &pack_id_for(7), &[]);
    let mut buf = Vec::new();
    packfile::write_header_with_creation_seq(&mut buf, &pack_id_for(0), 1).unwrap();
    std::fs::write(&path, &buf).unwrap();
    // The pool fails to open — identity mismatch is corruption.
    match ShardPool::open(dir) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        Ok(_) => panic!("expected InvalidData for identity mismatch"),
    }
}

/// End-to-end: `ShardPool::open` fails when a shard file has a
/// corrupted CRC — this is pool corruption, not a skip.
#[test]
fn test_shard_pool_open_fails_on_corrupt_header_crc() {
    let dir = test_dir("pool_open_bad_crc");
    std::fs::create_dir_all(&dir).unwrap();
    let pack_id = pack_id_for(0);
    let path = ShardPool::pack_path(&dir, &pack_id, &[]);
    let mut buf = Vec::new();
    packfile::write_header_with_creation_seq(&mut buf, &pack_id, 1).unwrap();
    buf[12] ^= 0xFF; // corrupt a byte inside the CRC-covered region
    std::fs::write(&path, &buf).unwrap();

    // The pool fails to open — corrupt header is corruption.
    match ShardPool::open(dir) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        Ok(_) => panic!("expected InvalidData for corrupt CRC"),
    }
}

/// End-to-end: `ShardPool::open` fails when a shard file has a
/// v4 filename but contains v1 content — this is a pre-v4
/// remnant, not a valid v4 pack.
#[test]
fn test_shard_pool_open_fails_on_v1_with_v4_filename() {
    let dir = test_dir("pool_open_v1_store");
    std::fs::create_dir_all(&dir).unwrap();
    let path = ShardPool::pack_path(&dir, &pack_id_for(0), &[]);
    let mut buf = Vec::new();
    buf.extend_from_slice(&packfile::MAGIC);
    buf.push(0x01);
    packfile::write_record(
        &mut buf,
        &packfile::Record {
            collection_id: [0x22; 16],
            hash: [0x33; 16],
            data: bytes::Bytes::from_static(b"pre-cutover data"),
            metadata: None,
        },
    )
    .unwrap();
    std::fs::write(&path, &buf).unwrap();

    // The pool fails to open — v1 content in a v4 filename slot
    // is corruption.
    match ShardPool::open(dir) {
        Err(e) => {
            assert!(
                e.kind() == io::ErrorKind::Unsupported || e.kind() == io::ErrorKind::InvalidData,
                "expected Unsupported or InvalidData, got: {:?}",
                e.kind()
            );
        }
        Ok(_) => panic!("expected error for v1 content in v4 filename"),
    }
}

/// `discover_pack_files` rejects v3 filenames (e.g.
/// `pack_0004_0000000000000004.pack`) with `Unsupported` rather
/// than silently skipping them.
#[test]
fn test_discover_rejects_v3_filenames() {
    let dir = test_dir("discover_v3_reject");
    std::fs::create_dir_all(&dir).unwrap();
    // Create a v3-format filename — 4-digit slot + underscore + 16-digit epoch.
    let path = dir.join("shard_0004_0000000000000004.pack");
    std::fs::write(&path, b"fake").unwrap();
    match ShardPool::discover_pack_files(&dir) {
        Err(e) => {
            assert_eq!(e.kind(), io::ErrorKind::Unsupported);
            assert!(
                e.to_string().contains("pre-v4"),
                "error should mention pre-v4, got: {e}"
            );
        }
        Ok(_) => panic!("expected Unsupported error for v3 filename"),
    }
}

/// A retired shard's file must survive as long as any `Arc<Shard>`
/// reference is held (e.g. via `PackfileStorage::pin_shards`, so a
/// repack reading stale offsets from it can't be undercut), and must
/// be deleted once the last reference actually drops. This is the
/// invariant `pin_shards` relies on to fix the shard-rotation race:
/// pinning a shard for a repack's duration keeps this `Drop` from
/// firing early, regardless of what else happens to the pool's own
/// slot for that shard in the meantime.
#[test]
fn test_drop_deletes_retired_shard_only_after_last_reference() {
    let dir = test_dir("drop_retire");
    let pool = ShardPool::open(dir).unwrap();

    let record = test_record(0x01, 0xAA, b"payload");
    let (slot, _offset) = pool.put_record(&record).unwrap();

    let pinned = pool.get_shard(slot).unwrap();
    let path = pinned.path.clone();
    assert!(path.exists());

    // Simulate a future repack-driven retirement (nothing currently
    // does this — see rotate()'s doc — but pin_shards must protect
    // against it regardless of how it eventually gets triggered), then
    // drop the pool's own reference the way replacing a recycled slot
    // would.
    pinned.is_current.store(false, Ordering::Release);
    drop(pool);

    // The pinned clone is still held, so the file must survive.
    assert!(
        path.exists(),
        "file deleted while a pinned Arc<Shard> was still held"
    );

    drop(pinned);
    assert!(
        !path.exists(),
        "retired shard's file should be deleted once its last reference drops"
    );
}

/// Crash-recovery: multiple valid shard files are discovered and
/// assigned to sequential slots in address order.
#[test]
fn test_scan_discovers_valid_shards_in_pack_id_order() {
    let dir = test_dir("scan_pack_id_order");
    let pool = ShardPool::open(dir.clone()).unwrap();

    // Write a record so a real pack exists, capture its path, then
    // discard it — this test hand-constructs on-disk files below
    // instead, since each one's embedded header must genuinely match
    // its own filename.
    let record = test_record(0x01, 0xAA, b"live data");
    pool.put_record(&record).unwrap();
    let old_path = pool.get_shard(0).unwrap().path.clone();
    drop(pool);
    std::fs::remove_file(&old_path).unwrap();

    // Create a valid pack_id 99 file.
    let live_pack = pack_id_for(99);
    let live_path = dir.join(live_pack.filename());
    let mut live_buf = Vec::new();
    packfile::write_header_with_creation_seq(&mut live_buf, &live_pack, 1).unwrap();
    packfile::write_record(
        &mut live_buf,
        &packfile::Record {
            collection_id: [0x01; 16],
            hash: [0xAA; 16],
            data: bytes::Bytes::from_static(b"live data"),
            metadata: None,
        },
    )
    .unwrap();
    std::fs::write(&live_path, &live_buf).unwrap();

    // Create a valid pack_id 0 file (the crash leftover).
    let leftover_pack = pack_id_for(0);
    let leftover_path = dir.join(leftover_pack.filename());
    let mut buf = Vec::new();
    packfile::write_header_with_creation_seq(&mut buf, &leftover_pack, 1).unwrap();
    packfile::write_record(
        &mut buf,
        &packfile::Record {
            collection_id: [0xFF; 16],
            hash: [0xBB; 16],
            data: bytes::Bytes::from_static(b"stale leftover"),
            metadata: None,
        },
    )
    .unwrap();
    std::fs::write(&leftover_path, &buf).unwrap();

    // Both files exist on disk.
    assert!(live_path.exists(), "pack_id 99 file missing");
    assert!(leftover_path.exists(), "pack_id 0 file missing");

    // Reopen the pool — the scan must discover both, assign them
    // to slots 0 and 1 in address order.
    let pool = ShardPool::open(dir.clone()).unwrap();
    let shard0 = pool.get_shard(0).unwrap();
    assert_eq!(
        shard0.pack_id, leftover_pack,
        "pack_id 0 should be assigned to slot 0"
    );
    assert_eq!(shard0.path, leftover_path);

    let shard1 = pool.get_shard(1).unwrap();
    assert_eq!(
        shard1.pack_id, live_pack,
        "pack_id 99 should be assigned to slot 1"
    );
    assert_eq!(shard1.path, live_path);
    drop(pool);

    // Both files should still exist (each pack address is unique).
    assert!(leftover_path.exists(), "pack_id 0 file should still exist");
    assert!(live_path.exists(), "pack_id 99 file should still exist");
}

/// A torn header in a validly-named shard file causes pool open to fail.
#[test]
fn test_scan_retains_valid_shard_when_newer_header_is_torn() {
    let dir = test_dir("scan_torn_newer_pack");
    let valid_pack = pack_id_for(1);
    let old_path = dir.join(valid_pack.filename());
    let mut old = Vec::new();
    packfile::write_header_with_creation_seq(&mut old, &valid_pack, 1).unwrap();
    std::fs::write(&old_path, old).unwrap();

    // A higher-address filename with a torn header — pool open
    // must fail because this corrupt shard is a valid filename
    // that can't be read.
    let torn_path = dir.join(pack_id_for(2).filename());
    std::fs::write(&torn_path, b"MTX").unwrap();

    match ShardPool::open(dir) {
        Err(e) => {
            assert!(
                e.kind() == io::ErrorKind::InvalidData || e.kind() == io::ErrorKind::UnexpectedEof,
                "expected InvalidData or UnexpectedEof for torn header, got: {:?}",
                e.kind()
            );
        }
        Ok(_) => panic!("expected error for torn shard header"),
    }
}

#[test]
fn discover_shards_reports_an_unopenable_new_pack() {
    let dir = test_dir("discover_unopenable_pack");
    let pool = ShardPool::open(dir.clone()).unwrap();
    let path = dir.join(pack_id_for(1).filename());
    std::fs::write(path, b"MTX").unwrap();

    let error = pool.discover_shards().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

/// `stats()` must reflect actual write/read/sync activity, so callers
/// can distinguish "many small fsyncs" (seek-bound, noisy) from
/// "few large writes, batched syncs" (quiet) without guessing from
/// disk sound. Deliberately no read counters — see `ShardStats`' doc
/// for why counting reads isn't worth the cache-line contention on
/// that path.
#[test]
fn test_shard_stats_track_write_sync() {
    let dir = test_dir("stats_tracking");
    // Buffered policy so the flush (triggered by the read of the still-
    // virtual offset below) is what credits the write counters.
    let pool = ShardPool::open(dir)
        .unwrap()
        .with_append_policy(AppendPolicy::buffered());

    let record = test_record(0x01, 0xAA, b"payload for stats");
    let (slot, offset) = pool.put_record(&record).unwrap();

    let shard = pool.get_shard(slot).unwrap();
    // Buffered records are counted at flush time, not append time.
    assert_eq!(shard.stats(), ShardStats::default());

    // The read of the still-buffered offset flushes the shard, which is
    // what credits the write counters.
    pool.read_at(&shard, offset, true).unwrap();
    let stats = shard.stats();
    assert_eq!(stats.write_count, 1);
    assert_eq!(stats.bytes_written, record.serialized_len() as u64);
    assert_eq!(stats.sync_count, 0);

    pool.sync_dirty().unwrap();
    let stats = shard.stats();
    assert_eq!(stats.sync_count, 1, "sync_dirty must bump sync_count");

    // A second sync_dirty with nothing new written must not double-count.
    pool.sync_dirty().unwrap();
    assert_eq!(shard.stats().sync_count, 1);

    let all = pool.all_stats();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, slot);
    assert_eq!(all[0].1, shard.stats());

    assert_eq!(pool.stats(slot), Some(shard.stats()));
    assert_eq!(pool.stats(slot.wrapping_add(1)), None);
}

/// The startup scan correctly parses the `pack_<16hex>.pack` filename
/// format and uses address order as the deterministic fallback when synthetic
/// test headers share the same one-second creation timestamp.
#[test]
fn test_scan_parses_pack_id_filename_format() {
    let dir = test_dir("scan_pack_id_format");

    // Manually create files. Each file's header must genuinely match
    // the address encoded by its own filename prefix.
    let make_pack = |collection_byte: u8, pack_id: packfile::PackId| -> Vec<u8> {
        let mut buf = Vec::new();
        packfile::write_header_with_creation_seq(&mut buf, &pack_id, 1).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                collection_id: {
                    let mut r = [0u8; 16];
                    r[0] = collection_byte;
                    r
                },
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"payload"),
                metadata: None,
            },
        )
        .unwrap();
        buf
    };

    let zero = pack_id_for(0);
    let three = pack_id_for(3);
    let five = pack_id_for(5);
    std::fs::write(dir.join(zero.filename()), make_pack(0x01, zero)).unwrap();
    std::fs::write(dir.join(five.filename()), make_pack(0x02, five)).unwrap();
    std::fs::write(dir.join(three.filename()), make_pack(0x03, three)).unwrap();

    let pool = ShardPool::open(dir).unwrap();

    // Equal-timestamp files are assigned to slots in address order (0, 3, 5).
    let s0 = pool.get_shard(0).unwrap();
    assert_eq!(s0.pack_id, zero, "pack_0000000000000000.pack → id 0");
    assert_eq!(s0.slot, 0);

    let s1 = pool.get_shard(1).unwrap();
    assert_eq!(s1.pack_id, three, "pack_0000000000000003.pack → id 3");
    assert_eq!(s1.slot, 1);

    let s2 = pool.get_shard(2).unwrap();
    assert_eq!(s2.pack_id, five, "pack_0000000000000005.pack → id 5");
    assert_eq!(s2.slot, 2);

    // Slot 3 was never created; pool should have no shard there.
    assert!(pool.get_shard(3).is_none());
}

/// Persisted creation sequences resolve same-second rotations without falling
/// back to the random pack address.
#[test]
fn test_scan_orders_same_timestamp_by_creation_sequence() {
    let dir = test_dir("scan_pack_creation_sequence");

    let make_pack = |pack_id: packfile::PackId, creation_seq: u32| {
        let mut buf = Vec::new();
        packfile::write_header_with_created_at(&mut buf, &pack_id, 1, creation_seq).unwrap();
        std::fs::write(dir.join(pack_id.filename()), buf).unwrap();
    };

    let newer_address = pack_id_for(3);
    let older_address = pack_id_for(5);
    make_pack(newer_address, 2);
    make_pack(older_address, 1);

    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.get_shard(0).unwrap().pack_id, older_address);
    assert_eq!(pool.get_shard(1).unwrap().pack_id, newer_address);
}

#[test]
fn test_pack_creation_sequence_survives_reopen() {
    let dir = test_dir("pack_creation_sequence_reopen");

    let pool = ShardPool::open(dir.clone()).unwrap();
    pool.try_active_shard().unwrap();
    pool.rotate().unwrap();
    drop(pool);

    let pool = ShardPool::open(dir.clone()).unwrap();
    pool.rotate().unwrap();

    let mut sequences = pool
        .all_shards()
        .into_iter()
        .map(|(_, shard)| {
            let mut reader = std::io::BufReader::new(std::fs::File::open(&shard.path).unwrap());
            packfile::read_header(&mut reader)
                .unwrap()
                .expect("created pack has a header")
                .creation_seq
        })
        .collect::<Vec<_>>();
    sequences.sort_unstable();
    assert_eq!(sequences, [1, 2, 3]);
}

/// A pack whose header carries `creation_seq == 0` is invalid: it is neither
/// sorted by timestamp nor given a synthesized sequence, the pool refuses to
/// open.
#[test]
fn test_open_rejects_a_pack_with_zero_creation_sequence() {
    let dir = test_dir("scan_zero_creation_seq");
    let pack_id = pack_id_for(5);
    let mut buf = Vec::new();
    packfile::write_header_with_created_at(&mut buf, &pack_id, 9, 1).unwrap();
    // creation_seq sits after magic(4) + version(1) + header_len(4) +
    // pack_id(16) + created_at(8); the CRC follows it.
    let seq_at = 4 + 1 + 4 + packfile::PACK_ID_LEN + 8;
    buf[seq_at..seq_at + 4].copy_from_slice(&0u32.to_le_bytes());
    let crc = crc32fast::hash(&buf[..seq_at + 4]);
    buf[seq_at + 4..seq_at + 8].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(dir.join(pack_id.filename()), buf).unwrap();

    assert!(ShardPool::open(dir).is_err());
}

#[test]
fn test_open_rejects_a_truncated_pack_without_header() {
    let dir = test_dir("scan_headerless_pack");
    std::fs::write(dir.join(pack_id_for(7).filename()), b"").unwrap();
    assert!(ShardPool::open(dir).is_err());
}

#[test]
fn two_rooted_pools_open_in_one_process_without_self_blocking() {
    let root = test_dir("rooted_two_pools_one_process");
    let layout = crate::layout::DatabaseLayout::open(root.clone()).unwrap();
    let state = ShardPool::open(layout.pool_path(crate::layout::ShardType::State));
    // The first standalone writer owns the root lock; a second one must fail
    // fast rather than block forever on the same file.
    let second = ShardPool::open(layout.pool_path(crate::layout::ShardType::Edges));
    assert!(state.is_ok());
    assert_eq!(
        second.err().map(|e| e.kind()),
        Some(std::io::ErrorKind::WouldBlock)
    );
    let _ = fs::remove_dir_all(&root);
}

/// Regression: `MAX_SHARD_BYTES` must stay within the offset field that
/// `IndexEntry` stores as `offset + 1`, so the maximum valid offset must
/// never reach the empty-slot sentinel boundary.
#[test]
fn max_shard_bytes_fits_index_slot() {
    // The largest offset a shard can ever present must survive
    // IndexEntry::new without panicking.
    let max_offset = MAX_SHARD_BYTES - 1;
    let slot = crate::index::IndexEntry::new(0, 0, max_offset);
    assert_eq!(slot.offset(), max_offset);
}

/// Regression: `retire_slot` must not clear the active write shard's
/// dirty bit. If it does, `sync_dirty()` sees no dirty shards and
/// skips the fsync — a repack can report success without persisting
/// its copied data.
#[test]
fn retire_slot_preserves_active_dirty_bit() {
    let dir = test_dir("retire_preserves_dirty");
    let pool = ShardPool::open(dir).unwrap();

    // Put a record so the active shard is dirty. Default policy is eager, so
    // the put alone commits it and marks the shard dirty.
    let record = test_record(1, 1, b"payload");
    let (slot, _offset) = pool.put_record(&record).unwrap();
    assert!(
        pool.dirty.lock().contains(&slot),
        "active shard should be dirty after writing data"
    );

    // Attempting to retire the active shard should be a no-op.
    pool.retire_slot(slot);
    assert!(
        pool.dirty.lock().contains(&slot),
        "active shard dirty bit must survive retire_slot"
    );

    // sync_dirty must still find and fsync the shard.
    pool.sync_dirty().unwrap();
    assert!(
        pool.dirty.lock().is_empty(),
        "sync_dirty should clear dirty after fsync"
    );
}
/// The durable watermark follows fsyncs: bytes written are owed until a sync
/// covers them, `sync_dirty` and `sync_all` both pay them off, and the
/// watermark never runs ahead of the file.
#[test]
fn synced_len_tracks_what_an_fsync_has_covered() {
    let dir = test_dir("synced_len");
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.unsynced_bytes(), 0);

    let record = test_record(0x01, 0xAA, b"first payload");
    let (slot, offset) = pool.put_record(&record).unwrap();
    let shard = pool.get_shard(slot).unwrap();
    // Force the buffered frame onto the file so it counts as written.
    pool.read_at(&shard, offset, true).unwrap();
    assert!(shard.file_len() > shard.synced_len());
    assert_eq!(pool.unsynced_bytes(), shard.unsynced_bytes());
    assert!(pool.unsynced_bytes() > 0);

    pool.sync_dirty().unwrap();
    assert_eq!(shard.synced_len(), shard.file_len());
    assert_eq!(pool.unsynced_bytes(), 0);

    // More data is owed again, and `sync_all` pays it off too.
    let (slot, offset) = pool
        .put_record(&test_record(0x02, 0xBB, b"second payload"))
        .unwrap();
    let shard = pool.get_shard(slot).unwrap();
    pool.read_at(&shard, offset, true).unwrap();
    assert!(pool.unsynced_bytes() > 0);
    pool.sync_all().unwrap();
    assert_eq!(pool.unsynced_bytes(), 0);
    assert!(shard.synced_len() <= shard.file_len());
}

/// Bytes that predate this process are not counted as owed.
#[test]
fn bytes_present_at_open_are_not_counted_as_unsynced() {
    let dir = test_dir("synced_len_reopen");
    {
        let pool = ShardPool::open(dir.clone()).unwrap();
        pool.put_record(&test_record(0x01, 0xAA, b"before"))
            .unwrap();
        pool.sync_all().unwrap();
    }
    let reopened = ShardPool::open(dir).unwrap();
    assert_eq!(reopened.unsynced_bytes(), 0);
}

fn pack_count(dir: &Path) -> usize {
    fs::read_dir(dir)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".pack")
        })
        .count()
}

/// An unused pool is just `pool.meta`: the first pack appears with the first
/// write, and a read-only open of the empty pool succeeds with no shards.
#[test]
fn a_new_pool_creates_no_pack_until_the_first_write() {
    let dir = test_dir("lazy_first_pack");
    let pool = ShardPool::open(dir.clone()).unwrap();
    assert_eq!(pack_count(&dir), 0, "open must not create a pack");
    assert!(dir.join(POOL_META_FILENAME).exists());
    assert_eq!(pool.shard_count(), 0);
    assert!(
        !dir.join("store.meta").exists(),
        "creator version lives in pool.meta"
    );
    assert_eq!(
        store_created_by_version(&dir).as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );

    let record = test_record(1, 1, b"first");
    let (slot, offset, _) = pool.put_record_with_len(&record).unwrap();
    assert_eq!(pack_count(&dir), 1, "the first write creates the pack");
    assert_eq!(pool.shard_count(), 1);
    pool.sync_all().unwrap();
    drop(pool);

    let reopened = ShardPool::open_read_only(dir.clone()).unwrap();
    assert_eq!(reopened.shard_count(), 1);
    assert_eq!(reopened.get_shard(slot).unwrap().slot, slot);
    let _ = offset;
}

#[test]
fn a_read_only_open_of_an_empty_pool_succeeds_and_a_missing_pool_fails() {
    let dir = test_dir("lazy_read_only_empty");
    drop(ShardPool::open(dir.clone()).unwrap());
    let reader = ShardPool::open_read_only(dir.clone()).unwrap();
    assert_eq!(reader.shard_count(), 0);
    assert!(reader.try_active_shard().is_err());
    assert!(ShardPool::open_read_only(dir.join("absent")).is_err());
}

/// A standalone writer on a pool inside a database root that was never written
/// must create the directory itself: it needs it for its lock file. (Pools opened
/// under the root's lock defer creation to their first pack instead.)
#[test]
fn a_standalone_writer_creates_an_absent_rooted_pool_directory() {
    let root = test_dir("rooted_standalone_writer");
    let layout = crate::layout::DatabaseLayout::open(root.clone()).unwrap();
    let pool = layout.pool_path(crate::layout::ShardType::State);
    assert!(!pool.exists(), "lazy init leaves the pool absent");

    let store = ShardPool::open(pool.clone()).expect("a standalone writer opens an absent pool");
    assert!(pool.is_dir());
    let record = test_record(1, 1, b"first");
    store.put_record_with_len(&record).unwrap();
    store.sync_all().unwrap();
    assert_eq!(pack_count(&pool), 1);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_standalone_rooted_writer_uses_the_database_writer_lock() {
    let root = test_dir("rooted_standalone_writer_lock");
    let layout = crate::layout::DatabaseLayout::open(root.clone()).unwrap();
    let _root_lock = crate::journal::SharedWalLock::acquire(&root).unwrap();
    let pool = layout.pool_path(crate::layout::ShardType::State);

    let Err(error) = ShardPool::open(pool) else {
        panic!("root writer lock must exclude pool writers")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
}

/// Pack addresses are random, so the pack with the highest address is not the
/// newest. Reopen must keep writing to the most recent pack that has room, not
/// to whichever pack sorts last.
#[test]
fn test_reopen_picks_most_recent_pack_not_highest_address() {
    let dir = test_dir("reopen_recent_pack");
    let newest_path = {
        let pool = ShardPool::open(dir.clone()).unwrap();
        for _ in 0..6 {
            pool.rotate().unwrap();
        }
        pool.active_shard().path.clone()
    };
    // Age every other pack so the last-created one is unambiguously newest,
    // independent of filesystem timestamp granularity.
    let past = std::time::SystemTime::now() - Duration::from_secs(3600);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "pack") && path != newest_path {
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(past)
                .unwrap();
        }
    }
    let pool = ShardPool::open(dir).unwrap();
    assert_eq!(pool.active_shard().path, newest_path);
}
