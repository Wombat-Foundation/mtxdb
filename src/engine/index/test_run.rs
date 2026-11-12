use super::*;

use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;

use crate::index::packidx::{PackIndex, PackIndexInput};

static DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A fresh directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mtxdb-run-test-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const PACK_LEN: u64 = 1 << 20;
/// Disk bytes each test sidecar records per entry.
const FRAME_BYTES: u64 = 100;

fn header_len() -> u32 {
    u32::try_from(HEADER_LEN).unwrap()
}

fn offset(delta: u32) -> u32 {
    header_len().checked_add(delta).unwrap()
}

fn identity(seq: u64) -> PackIndexIdentity {
    let mut id = [0_u8; 16];
    id[..8].copy_from_slice(&seq.to_le_bytes());
    id[15] = 0xA5;
    PackIndexIdentity {
        pack_id: PackId(id),
        creation_seq: seq,
        pack_file_len: u64::from(header_len()).checked_add(PACK_LEN).unwrap(),
    }
}

/// Deterministic splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next().checked_rem(bound).unwrap()
    }
}

/// A spread, never-zero id, so `[0; 16]` always sorts below every key.
fn logical_id(value: u64) -> [u8; 16] {
    let value = value.wrapping_add(1);
    let mut id = [0_u8; 16];
    id[..8].copy_from_slice(&value.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_be_bytes());
    id[8..].copy_from_slice(&value.to_be_bytes());
    id
}

fn collection(value: u8) -> [u8; 16] {
    [value; 16]
}

type Key = ([u8; 16], [u8; 16]);

/// One pack's contents: key -> (offset, removed).
type PackModel = BTreeMap<Key, (u32, bool)>;

fn write_sidecar(dir: &TestDir, seq: u64, model: &PackModel) -> MappedPackIndex {
    let pack = identity(seq);
    let inputs = model
        .iter()
        .map(
            |(&(collection_id, logical_id), &(at, removed))| PackIndexInput {
                collection_id,
                logical_id,
                offset: u64::from(at),
                removed,
                disk_bytes: FRAME_BYTES,
            },
        )
        .collect();
    let path = dir.join(&format!("pack-{seq}.idx"));
    let index = PackIndex::build(pack, inputs).unwrap();
    // These are merge inputs: they only have to be readable through the same
    // mmap reader. An atomic publish (file fsync, rename, directory fsync) is
    // covered by the packidx tests, and cost about 0.14 s per input on a
    // spinning disk, hundreds of times per run of the model tests.
    fs::write(&path, index.as_bytes()).unwrap();
    PackIndex::open_mmap(&path, pack).unwrap()
}

fn random_pack(rng: &mut Rng, keys: u64) -> PackModel {
    let mut model = PackModel::new();
    let count = rng.below(40).checked_add(1).unwrap();
    for _ in 0..count {
        let key = (
            collection(u8::try_from(rng.below(3)).unwrap()),
            logical_id(rng.below(keys)),
        );
        let at = offset(u32::try_from(rng.below(PACK_LEN)).unwrap());
        model.insert(key, (at, rng.below(4) == 0));
    }
    model
}

/// The newest version of `key` across `packs` (oldest first), as
/// `(pack index, locator)`.
fn expected(packs: &[PackModel], key: &Key) -> Option<(usize, PackIndexLocator)> {
    packs.iter().enumerate().rev().find_map(|(index, pack)| {
        pack.get(key).map(|&(at, removed)| {
            let locator = if removed {
                PackIndexLocator::Removed(at)
            } else {
                PackIndexLocator::Frame(at)
            };
            (index, locator)
        })
    })
}

fn entry(collection_id: u8, id: u64, ordinal: u16, delta: u32) -> RunEntry {
    RunEntry {
        collection_id: collection(collection_id),
        logical_id: logical_id(id),
        ordinal,
        locator: PackIndexLocator::Frame(offset(delta)),
    }
}

fn sorted(mut entries: Vec<RunEntry>) -> Vec<RunEntry> {
    entries.sort_by_key(|entry| (entry.collection_id, entry.logical_id));
    entries
}

fn fixture_pack_stats(entries: &[RunEntry]) -> Vec<RunPackStats> {
    let mut counts = BTreeMap::new();
    for entry in entries {
        let count = counts
            .entry((entry.collection_id, entry.ordinal))
            .or_insert(0_u64);
        *count = count
            .checked_add(1)
            .expect("pack stats entry count overflow");
    }
    counts
        .into_iter()
        .map(|((collection_id, ordinal), entry_count)| RunPackStats {
            ordinal,
            collection_id,
            entry_count,
            disk_bytes: 0,
        })
        .collect()
}

/// Write `entries` (sorted) as a run over packs `1..=pack_count` with small
/// blocks so tests cross block boundaries.
fn write_run(path: &Path, pack_count: u64, entries: &[RunEntry]) -> Option<RunSummary> {
    let packs = (1..=pack_count).map(identity).collect();
    let mut writer = RunWriter::create(path, packs, 4).unwrap();
    for entry in entries {
        writer.push(*entry).unwrap();
    }
    writer.set_pack_stats(fixture_pack_stats(entries)).unwrap();
    writer.finish().unwrap()
}

#[test]
fn written_run_answers_lookups_scans_and_misses() {
    let dir = TestDir::new();
    let path = dir.join("a.run");
    let entries = sorted(
        (0..30_u64)
            .map(|id| {
                let mut entry = entry(
                    u8::try_from(id % 3).unwrap(),
                    id,
                    u16::try_from(id % 2).unwrap(),
                    u32::try_from(id).unwrap(),
                );
                if id % 5 == 0 {
                    entry.locator = PackIndexLocator::Removed(offset(u32::try_from(id).unwrap()));
                }
                entry
            })
            .collect(),
    );
    let summary = write_run(&path, 2, &entries).unwrap();
    assert_eq!(summary.entry_count, 30);
    assert_eq!(summary.collection_count, 3);
    assert_eq!(
        summary.block_count, 9,
        "10 entries per collection in blocks of 4"
    );
    assert_eq!(summary.file_len, fs::metadata(&path).unwrap().len());

    let run = RunIndex::open(&path).unwrap();
    assert_eq!(run.entry_count(), 30);
    assert_eq!(run.packs(), &[identity(1), identity(2)]);
    for expected in &entries {
        let hit = run
            .lookup(&expected.collection_id, &expected.logical_id)
            .unwrap()
            .unwrap();
        assert_eq!(hit.ordinal, expected.ordinal);
        assert_eq!(
            hit.pack,
            identity(u64::from(expected.ordinal).checked_add(1).unwrap())
        );
        assert_eq!(hit.locator, expected.locator);
    }
    for id in 30..60 {
        assert_eq!(run.lookup(&collection(0), &logical_id(id)), Ok(None));
    }
    assert_eq!(run.lookup(&[0xFF; 16], &logical_id(1)), Ok(None));
    assert_eq!(
        run.lookup(&collection(0), &[0; 16]),
        Ok(None),
        "below the first fence"
    );

    let streamed: Vec<RunEntry> = run.entries().map(Result::unwrap).collect();
    assert_eq!(streamed, entries);
    let mut scanned = Vec::new();
    run.for_each_in_collection(&collection(1), |hit, id| scanned.push((id, hit.locator)))
        .unwrap();
    let expected_scan: Vec<_> = entries
        .iter()
        .filter(|entry| entry.collection_id == collection(1))
        .map(|entry| (entry.logical_id, entry.locator))
        .collect();
    assert_eq!(scanned, expected_scan);
    run.verify_all().unwrap();
    assert!(!run.is_invalid());
}

#[test]
fn writer_rejects_input_the_reader_would_reject() {
    let dir = TestDir::new();
    let path = dir.join("bad.run");
    let packs = || vec![identity(1), identity(2)];

    // Ids are hashed, so pick the larger of two by byte order.
    let (low, high) = if logical_id(1) < logical_id(2) {
        (1, 2)
    } else {
        (2, 1)
    };
    let mut writer = RunWriter::create(&path, packs(), 4).unwrap();
    writer.push(entry(0, high, 0, 0)).unwrap();
    assert!(matches!(
        writer.push(entry(0, high, 0, 0)),
        Err(RunError::OutOfOrder)
    ));
    assert!(matches!(
        writer.push(entry(0, low, 0, 0)),
        Err(RunError::OutOfOrder)
    ));

    let mut writer = RunWriter::create(&path, packs(), 4).unwrap();
    assert!(matches!(
        writer.push(entry(0, 1, 2, 0)),
        Err(RunError::BadLocator)
    ));
    let mut below_data = entry(0, 1, 0, 0);
    below_data.locator = PackIndexLocator::Frame(header_len().checked_sub(1).unwrap());
    assert!(matches!(writer.push(below_data), Err(RunError::BadLocator)));
    let mut past_end = entry(0, 1, 0, 0);
    past_end.locator = PackIndexLocator::Frame(u32::try_from(identity(1).pack_file_len).unwrap());
    assert!(matches!(writer.push(past_end), Err(RunError::BadLocator)));

    assert!(matches!(
        RunWriter::create(&path, vec![identity(2), identity(1)], 4),
        Err(RunError::BadPackTable)
    ));
    assert!(matches!(
        RunWriter::create(&path, Vec::new(), 4),
        Err(RunError::BadPackTable)
    ));
}

#[test]
fn an_empty_run_is_not_written() {
    let dir = TestDir::new();
    let path = dir.join("empty.run");
    let writer = RunWriter::create(&path, vec![identity(1)], 4).unwrap();
    assert_eq!(writer.finish().unwrap(), None);
    assert!(!path.exists());
}

#[test]
fn open_rejects_corrupt_metadata_and_header() {
    let dir = TestDir::new();
    let path = dir.join("meta.run");
    let entries = sorted((0..10).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&path, 1, &entries).unwrap();
    let good = fs::read(&path).unwrap();

    let header_at = 16; // entry count
    let meta_at = good.len().checked_sub(1).unwrap(); // last block-table byte
    for at in [header_at, meta_at] {
        let mut bytes = good.clone();
        bytes[at] ^= 0x01;
        let corrupt = dir.join(&format!("meta-{at}.run"));
        fs::write(&corrupt, &bytes).unwrap();
        assert!(
            matches!(RunIndex::open(&corrupt), Err(RunError::Invalid)),
            "flipped byte {at} must fail open"
        );
    }
    let truncated = dir.join("truncated.run");
    fs::write(&truncated, &good[..good.len().checked_sub(1).unwrap()]).unwrap();
    assert!(matches!(RunIndex::open(&truncated), Err(RunError::Invalid)));
}

#[test]
fn a_corrupt_block_fails_lookups_instead_of_missing() {
    let dir = TestDir::new();
    let path = dir.join("block.run");
    let entries = sorted((0..12).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&path, 1, &entries).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // The first block's first id: its CRC no longer matches.
    bytes[RUN_HEADER_LEN] ^= 0x80;
    fs::write(&path, &bytes).unwrap();

    let run = RunIndex::open(&path).expect("blocks are verified lazily, not at open");
    let first = entries.first().unwrap();
    assert_eq!(
        run.lookup(&first.collection_id, &first.logical_id),
        Err(RunCorrupt),
        "a damaged block must never read as a miss"
    );
    assert!(run.is_invalid());
    let last = entries.last().unwrap();
    assert_eq!(
        run.lookup(&last.collection_id, &last.logical_id),
        Err(RunCorrupt),
        "once invalid, intact blocks stop answering too"
    );
    assert_eq!(run.verify_all(), Err(RunCorrupt));
    assert!(run.entries().any(|entry| entry.is_err()));
}

/// Recompute `meta_crc32` after a test edits the header or metadata.
fn reseal_meta(bytes: &mut [u8]) {
    let meta_at = usize::try_from(read_u64(bytes, 40).unwrap()).unwrap();
    bytes[CRC_AT..CRC_AT.checked_add(4).unwrap()].fill(0);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..RUN_HEADER_LEN]);
    hasher.update(&bytes[meta_at..]);
    let meta_crc = hasher.finalize();
    bytes[CRC_AT..CRC_AT.checked_add(4).unwrap()].copy_from_slice(&meta_crc.to_le_bytes());
}

/// Byte offset of block `index`'s table entry.
fn block_entry_at(bytes: &[u8], index: usize) -> usize {
    let meta_at = usize::try_from(read_u64(bytes, 40).unwrap()).unwrap();
    let packs = usize::try_from(read_u32(bytes, 24).unwrap()).unwrap();
    let collections = usize::try_from(read_u32(bytes, 28).unwrap()).unwrap();
    meta_at
        .checked_add(packs.checked_mul(PACK_ENTRY_LEN).unwrap())
        .and_then(|at| at.checked_add(collections.checked_mul(COLLECTION_ENTRY_LEN).unwrap()))
        .and_then(|at| at.checked_add(index.checked_mul(BLOCK_ENTRY_LEN).unwrap()))
        .unwrap()
}

#[test]
fn open_rejects_overlapping_block_ranges() {
    let dir = TestDir::new();
    let path = dir.join("overlap.run");
    let entries = sorted((0..8).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&path, 1, &entries).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // Claim block 0 ends at block 1's fence, with a valid metadata CRC.
    let second_fence = entries[4].logical_id;
    let last_at = block_entry_at(&bytes, 0).checked_add(ID_LEN).unwrap();
    bytes[last_at..last_at.checked_add(ID_LEN).unwrap()].copy_from_slice(&second_fence);
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(matches!(RunIndex::open(&path), Err(RunError::Invalid)));
}

#[test]
fn a_block_whose_data_runs_past_its_recorded_last_id_is_corrupt() {
    let dir = TestDir::new();
    let path = dir.join("tail.run");
    let entries = sorted((0..8).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&path, 1, &entries).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // Rewrite block 0's last id in the data to sit beyond block 1's fence and
    // fix that block's CRC, leaving the recorded last id alone.
    let beyond = entries[7].logical_id;
    let last_id_at = RUN_HEADER_LEN
        .checked_add(ID_LEN.checked_mul(3).unwrap())
        .unwrap();
    bytes[last_id_at..last_id_at.checked_add(ID_LEN).unwrap()].copy_from_slice(&beyond);
    let block_len = ENTRY_LEN.checked_mul(4).unwrap();
    let crc =
        crc32fast::hash(&bytes[RUN_HEADER_LEN..RUN_HEADER_LEN.checked_add(block_len).unwrap()]);
    let crc_at = block_entry_at(&bytes, 0).checked_add(44).unwrap();
    bytes[crc_at..crc_at.checked_add(4).unwrap()].copy_from_slice(&crc.to_le_bytes());
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();

    let run = RunIndex::open(&path).unwrap();
    assert_eq!(
        run.lookup(&entries[0].collection_id, &entries[0].logical_id),
        Err(RunCorrupt)
    );
}

#[test]
fn a_pack_table_with_one_id_twice_is_rejected() {
    let dir = TestDir::new();
    let mut again = identity(1);
    again.creation_seq = 2;
    assert!(matches!(
        RunWriter::create(&dir.join("dup.run"), vec![identity(1), again], 4),
        Err(RunError::BadPackTable)
    ));
}

#[test]
fn an_empty_merge_removes_a_stale_run_at_its_path() {
    let dir = TestDir::new();
    let removal: PackModel = [((collection(0), logical_id(1)), (offset(0), true))]
        .into_iter()
        .collect();
    let sidecar = write_sidecar(&dir, 1, &removal);
    let path = dir.join("range.run");
    let sources = [MergeSource::Sidecar(&sidecar)];
    merge_to_path(&sources, &MergeOptions::default(), &path)
        .unwrap()
        .unwrap();
    assert!(path.exists(), "removals are kept by default");
    let options = MergeOptions {
        drop_removals: true,
        ..MergeOptions::default()
    };
    assert_eq!(merge_to_path(&sources, &options, &path).unwrap(), None);
    assert!(
        !path.exists(),
        "the earlier run for this range must not survive"
    );
}

#[test]
fn a_block_whose_crc_matches_but_whose_order_is_wrong_is_rejected() {
    let dir = TestDir::new();
    let path = dir.join("order.run");
    let entries = sorted((0..4).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&path, 1, &entries).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    // Swap the first two ids, then fix up the block CRC and metadata CRC so
    // only the content check can catch it.
    let first = RUN_HEADER_LEN;
    let second = first.checked_add(ID_LEN).unwrap();
    let (a, b) = (
        bytes[first..second].to_vec(),
        bytes[second..second.checked_add(ID_LEN).unwrap()].to_vec(),
    );
    bytes[first..second].copy_from_slice(&b);
    bytes[second..second.checked_add(ID_LEN).unwrap()].copy_from_slice(&a);
    let block_len = ENTRY_LEN.checked_mul(4).unwrap();
    let block_crc = crc32fast::hash(&bytes[first..first.checked_add(block_len).unwrap()]);
    let header = RunHeader::parse(&bytes).unwrap();
    let block_at = header.block_table_at().unwrap();
    let crc_at = block_at
        .checked_add(BLOCK_ENTRY_LEN)
        .unwrap()
        .checked_sub(4)
        .unwrap();
    bytes[crc_at..crc_at.checked_add(4).unwrap()].copy_from_slice(&block_crc.to_le_bytes());
    // The fence must still match the (new) first id for the swap alone to be
    // the defect.
    let fence_at = block_at;
    bytes[fence_at..fence_at.checked_add(ID_LEN).unwrap()].copy_from_slice(&b);
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();

    let run = RunIndex::open(&path).unwrap();
    assert_eq!(run.verify_all(), Err(RunCorrupt));
}

#[test]
fn merge_keeps_the_newest_version_against_a_model() {
    for seed in 0..60_u64 {
        let dir = TestDir::new();
        let mut rng = Rng(seed);
        let pack_count = usize::try_from(rng.below(9))
            .unwrap()
            .checked_add(1)
            .unwrap();
        let models: Vec<PackModel> = (0..pack_count).map(|_| random_pack(&mut rng, 60)).collect();
        let sidecars: Vec<MappedPackIndex> = models
            .iter()
            .zip(1_u64..)
            .map(|(model, seq)| write_sidecar(&dir, seq, model))
            .collect();

        // Cut the packs into up to three contiguous groups. A group of two or
        // more becomes its own run, so newer runs land at output ordinals
        // above their own and the final merge must remap them.
        let mut cuts: Vec<usize> = (0..2)
            .map(|_| usize::try_from(rng.below(u64::try_from(pack_count).unwrap())).unwrap())
            .chain([0, pack_count])
            .collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut runs = Vec::new();
        let mut loose: Vec<&MappedPackIndex> = Vec::new();
        for (group, bounds) in cuts.windows(2).enumerate() {
            let members = &sidecars[bounds[0]..bounds[1]];
            if members.len() < 2 {
                loose.extend(members);
                continue;
            }
            let sources: Vec<MergeSource<'_>> = members.iter().map(MergeSource::Sidecar).collect();
            let path = dir.join(&format!("group-{group}.run"));
            if merge_to_path(&sources, &MergeOptions::default(), &path)
                .unwrap()
                .is_some()
            {
                runs.push(RunIndex::open(&path).unwrap());
            }
        }
        let mut sources: Vec<MergeSource<'_>> = runs
            .iter()
            .map(MergeSource::Run)
            .chain(loose.into_iter().map(MergeSource::Sidecar))
            .collect();
        // The output must not depend on input order.
        sources.reverse();
        let options = MergeOptions {
            drop_removals: false,
            block_entries: 3,
            ..MergeOptions::default()
        };
        let path = dir.join("merged.run");
        let merged = merge_to_path(&sources, &options, &path).unwrap().unwrap();
        let run = RunIndex::open(&path).unwrap();
        run.verify_all().unwrap();
        let mut keys: Vec<Key> = models
            .iter()
            .flat_map(|model| model.keys().copied())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            merged.entry_count,
            u64::try_from(keys.len()).unwrap(),
            "seed {seed}"
        );
        for key in &keys {
            let (pack_index, locator) = expected(&models, key).unwrap();
            let hit = run.lookup(&key.0, &key.1).unwrap().unwrap();
            assert_eq!(hit.locator, locator, "seed {seed}");
            assert_eq!(
                hit.pack,
                identity(u64::try_from(pack_index).unwrap().checked_add(1).unwrap()),
                "seed {seed}: ordinals must resolve to the right pack"
            );
        }
        for miss in 60..80 {
            assert_eq!(run.lookup(&collection(0), &logical_id(miss)), Ok(None));
        }

        // Accounting is each pack's own sidecar directory, not a recount of
        // the merged winners.
        let mut expected_stats: Vec<RunPackStats> = Vec::new();
        for (index, model) in models.iter().enumerate() {
            let mut per_collection: BTreeMap<[u8; 16], u64> = BTreeMap::new();
            for (collection_id, _) in model.keys() {
                let count = per_collection.entry(*collection_id).or_default();
                *count = count.checked_add(1).unwrap();
            }
            for (collection_id, entry_count) in per_collection {
                expected_stats.push(RunPackStats {
                    ordinal: u16::try_from(index).unwrap(),
                    collection_id,
                    entry_count,
                    disk_bytes: entry_count.checked_mul(FRAME_BYTES).unwrap(),
                });
            }
        }
        expected_stats.sort_unstable_by_key(|row| (row.collection_id, row.ordinal));
        assert_eq!(
            run.pack_stats().collect::<Vec<_>>(),
            expected_stats,
            "seed {seed}"
        );
    }
}

#[test]
fn writer_rejects_invalid_pack_stats() {
    let dir = TestDir::new();
    let mut writer = RunWriter::create(&dir.join("stats.run"), vec![identity(1)], 4).unwrap();
    let row = RunPackStats {
        ordinal: 0,
        collection_id: collection(0),
        entry_count: 1,
        disk_bytes: 10,
    };
    let unknown_pack = RunPackStats { ordinal: 1, ..row };
    let empty = RunPackStats {
        entry_count: 0,
        ..row
    };
    assert!(matches!(
        writer.set_pack_stats(vec![unknown_pack]),
        Err(RunError::BadLocator)
    ));
    assert!(matches!(
        writer.set_pack_stats(vec![empty]),
        Err(RunError::BadLocator)
    ));
    assert!(matches!(
        writer.set_pack_stats(vec![row, row]),
        Err(RunError::BadLocator)
    ));
    writer.set_pack_stats(vec![row]).unwrap();
}

#[test]
fn finish_rejects_explicit_stats_that_miss_a_pushed_pair() {
    let dir = TestDir::new();
    let path = dir.join("missing-pair.run");
    let mut writer = RunWriter::create(&path, vec![identity(1), identity(2)], 4).unwrap();
    writer.push(entry(0, 1, 0, 10)).unwrap();
    writer.push(entry(0, 2, 1, 20)).unwrap();
    writer
        .set_pack_stats(vec![RunPackStats {
            ordinal: 0,
            collection_id: collection(0),
            entry_count: 1,
            disk_bytes: 10,
        }])
        .unwrap();
    assert!(matches!(writer.finish(), Err(RunError::BadLocator)));
}

#[test]
fn open_rejects_corrupt_pack_stats() {
    let dir = TestDir::new();
    let path = dir.join("stats.run");
    let packs = vec![identity(1), identity(2)];
    let mut writer = RunWriter::create(&path, packs, 4).unwrap();
    writer.push(entry(0, 1, 0, 0)).unwrap();
    let row = |ordinal| RunPackStats {
        ordinal,
        collection_id: collection(0),
        entry_count: 1,
        disk_bytes: 10,
    };
    writer.set_pack_stats(vec![row(0), row(1)]).unwrap();
    writer.finish().unwrap().unwrap();
    let good = fs::read(&path).unwrap();
    assert_eq!(RunIndex::open(&path).unwrap().pack_stats().count(), 2);

    // Point the last row at a pack outside the table, with a valid CRC.
    let mut bytes = good.clone();
    let ordinal_at = bytes.len().checked_sub(STATS_ENTRY_LEN).unwrap();
    bytes[ordinal_at..ordinal_at.checked_add(2).unwrap()].copy_from_slice(&2_u16.to_le_bytes());
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(matches!(RunIndex::open(&path), Err(RunError::Invalid)));

    // Swap the rows so they are out of order.
    let mut bytes = good;
    bytes[ordinal_at..ordinal_at.checked_add(2).unwrap()].copy_from_slice(&0_u16.to_le_bytes());
    let first_at = ordinal_at.checked_sub(STATS_ENTRY_LEN).unwrap();
    bytes[first_at..first_at.checked_add(2).unwrap()].copy_from_slice(&1_u16.to_le_bytes());
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(matches!(RunIndex::open(&path), Err(RunError::Invalid)));
}

#[test]
fn removals_are_dropped_only_when_asked() {
    let dir = TestDir::new();
    let older: PackModel = [
        ((collection(0), logical_id(1)), (offset(10), false)),
        ((collection(0), logical_id(2)), (offset(20), false)),
    ]
    .into_iter()
    .collect();
    let newer: PackModel = [
        ((collection(0), logical_id(1)), (offset(30), true)),
        ((collection(0), logical_id(3)), (offset(40), true)),
    ]
    .into_iter()
    .collect();
    let sidecars = [
        write_sidecar(&dir, 1, &older),
        write_sidecar(&dir, 2, &newer),
    ];
    let sources: Vec<MergeSource<'_>> = sidecars.iter().map(MergeSource::Sidecar).collect();

    let kept = dir.join("kept.run");
    merge_to_path(&sources, &MergeOptions::default(), &kept)
        .unwrap()
        .unwrap();
    let kept = RunIndex::open(&kept).unwrap();
    assert_eq!(
        kept.lookup(&collection(0), &logical_id(1))
            .unwrap()
            .unwrap()
            .locator,
        PackIndexLocator::Removed(offset(30)),
        "a newer removal shadows the older live frame"
    );
    assert_eq!(kept.entry_count(), 3);

    let dropped = dir.join("dropped.run");
    let options = MergeOptions {
        drop_removals: true,
        ..MergeOptions::default()
    };
    merge_to_path(&sources, &options, &dropped)
        .unwrap()
        .unwrap();
    let dropped = RunIndex::open(&dropped).unwrap();
    assert_eq!(dropped.lookup(&collection(0), &logical_id(1)), Ok(None));
    assert_eq!(dropped.lookup(&collection(0), &logical_id(3)), Ok(None));
    assert_eq!(
        dropped
            .lookup(&collection(0), &logical_id(2))
            .unwrap()
            .unwrap()
            .locator,
        PackIndexLocator::Frame(offset(20))
    );
    assert_eq!(dropped.entry_count(), 1);
}

#[test]
fn a_merge_of_only_dropped_removals_writes_nothing() {
    let dir = TestDir::new();
    let only_removals: PackModel = [((collection(0), logical_id(1)), (offset(0), true))]
        .into_iter()
        .collect();
    let sidecar = write_sidecar(&dir, 1, &only_removals);
    let path = dir.join("none.run");
    let options = MergeOptions {
        drop_removals: true,
        ..MergeOptions::default()
    };
    assert_eq!(
        merge_to_path(&[MergeSource::Sidecar(&sidecar)], &options, &path).unwrap(),
        None
    );
    assert!(!path.exists());
    let mut temp = path.into_os_string();
    temp.push(".tmp");
    assert!(!PathBuf::from(temp).exists());
}

#[test]
fn merge_rejects_overlapping_and_interleaved_inputs() {
    let dir = TestDir::new();
    let model: PackModel = [((collection(0), logical_id(1)), (offset(0), false))]
        .into_iter()
        .collect();
    let sidecars: Vec<MappedPackIndex> = (1..=3)
        .map(|seq| write_sidecar(&dir, seq, &model))
        .collect();
    let path = dir.join("bad.run");

    let twice = [
        MergeSource::Sidecar(&sidecars[0]),
        MergeSource::Sidecar(&sidecars[0]),
    ];
    assert!(matches!(
        merge_to_path(&twice, &MergeOptions::default(), &path),
        Err(RunError::OverlappingInputs)
    ));

    // A run over packs 1 and 3 cannot be merged with pack 2.
    let outer = dir.join("outer.run");
    merge_to_path(
        &[
            MergeSource::Sidecar(&sidecars[0]),
            MergeSource::Sidecar(&sidecars[2]),
        ],
        &MergeOptions::default(),
        &outer,
    )
    .unwrap()
    .unwrap();
    let outer = RunIndex::open(&outer).unwrap();
    assert!(matches!(
        merge_to_path(
            &[MergeSource::Run(&outer), MergeSource::Sidecar(&sidecars[1])],
            &MergeOptions::default(),
            &path,
        ),
        Err(RunError::OverlappingInputs)
    ));
    assert!(!path.exists());
}

#[test]
fn merging_a_corrupt_run_fails_and_leaves_no_output() {
    let dir = TestDir::new();
    let input = dir.join("input.run");
    let entries = sorted((0..8).map(|id| entry(0, id, 0, 0)).collect());
    write_run(&input, 1, &entries).unwrap();
    let mut bytes = fs::read(&input).unwrap();
    bytes[RUN_HEADER_LEN.checked_add(ID_LEN).unwrap()] ^= 0x01;
    fs::write(&input, &bytes).unwrap();
    let run = RunIndex::open(&input).unwrap();

    let path = dir.join("out.run");
    assert!(matches!(
        merge_to_path(&[MergeSource::Run(&run)], &MergeOptions::default(), &path),
        Err(RunError::Corrupt(RunCorrupt))
    ));
    assert!(!path.exists());
}

/// Write an empty-stats run over exactly `packs`.
fn write_run_over(path: &Path, packs: Vec<PackIndexIdentity>) {
    let mut writer = RunWriter::create(path, packs, 4).unwrap();
    let entry = entry(1, 1, 0, 0);
    writer.push(entry).unwrap();
    writer.set_pack_stats(fixture_pack_stats(&[entry])).unwrap();
    writer.finish().unwrap().unwrap();
}

#[test]
fn load_pool_runs_keeps_only_runs_that_match_the_pool() {
    let dir = TestDir::new();
    let pool: Vec<PackIndexIdentity> = (1..=12).map(identity).collect();
    let valid = dir.join("valid.run");
    write_run_over(&valid, vec![identity(1), identity(2)]);
    let mut changed = identity(3);
    changed.pack_file_len = changed.pack_file_len.checked_add(1).unwrap();
    let stale = dir.join("stale.run");
    write_run_over(&stale, vec![changed]);
    // Pack 5 sits inside this range without being covered.
    let gapped = dir.join("gapped.run");
    write_run_over(&gapped, vec![identity(4), identity(6)]);
    let missing = dir.join("missing.run");
    write_run_over(&missing, vec![identity(13)]);
    // A merged run published before its input run was deleted: the
    // containing run serves both ranges.
    let containing = dir.join("containing.run");
    write_run_over(&containing, vec![identity(7), identity(8)]);
    let contained = dir.join("contained.run");
    write_run_over(&contained, vec![identity(8)]);
    // Partial overlaps are ambiguous; both fall back to sidecars.
    let overlap_a = dir.join("overlap-a.run");
    write_run_over(&overlap_a, vec![identity(9), identity(10)]);
    let overlap_b = dir.join("overlap-b.run");
    write_run_over(&overlap_b, vec![identity(10), identity(11)]);
    let damaged = dir.join("damaged.run");
    fs::write(&damaged, b"not a run").unwrap();
    let temporary = dir.join("x.run.tmp");
    fs::write(&temporary, b"partial").unwrap();
    let unrelated = dir.join("notes.txt");
    fs::write(&unrelated, b"keep").unwrap();

    let rejected = [
        &stale, &gapped, &missing, &contained, &overlap_a, &overlap_b, &damaged, &temporary,
    ];

    // A read-only open accepts the same runs but deletes nothing.
    let read_only = load_pool_runs(&dir.0, &pool, false);
    assert_eq!(
        read_only
            .iter()
            .map(|run| run.path.clone())
            .collect::<Vec<_>>(),
        vec![valid.clone(), containing.clone()]
    );
    for path in rejected {
        assert!(path.exists(), "read-only open kept {}", path.display());
    }

    let writable = load_pool_runs(&dir.0, &pool, true);
    assert_eq!(
        writable
            .iter()
            .map(|run| run.path.clone())
            .collect::<Vec<_>>(),
        vec![valid.clone(), containing.clone()]
    );
    for path in rejected {
        assert!(!path.exists(), "writable open removed {}", path.display());
    }
    assert!(valid.exists());
    assert!(containing.exists());
    assert!(unrelated.exists(), "only run files are touched");
}

#[test]
fn pack_stats_in_collection_matches_a_filtered_scan() {
    let dir = TestDir::new();
    let mut rng = Rng(0x5EED_57A7);
    let sidecars: Vec<MappedPackIndex> = (1..=6)
        .map(|seq| write_sidecar(&dir, seq, &random_pack(&mut rng, 64)))
        .collect();
    let sources: Vec<MergeSource<'_>> = sidecars.iter().map(MergeSource::Sidecar).collect();
    let path = dir.join("stats.run");
    merge_to_path(&sources, &MergeOptions::default(), &path)
        .unwrap()
        .unwrap();
    let run = RunIndex::open(&path).unwrap();
    for value in 0..=3 {
        let collection_id = collection(value);
        let expected: Vec<RunPackStats> = run
            .pack_stats()
            .filter(|row| row.collection_id == collection_id)
            .collect();
        let found: Vec<RunPackStats> = run.pack_stats_in_collection(&collection_id).collect();
        assert_eq!(found, expected, "collection {value}");
    }
}

#[test]
fn verify_collection_checks_only_that_collections_blocks() {
    let dir = TestDir::new();
    let path = dir.join("verify.run");
    let mut entries: Vec<RunEntry> = (0..12).map(|id| entry(1, id, 0, 10)).collect();
    entries.extend((0..12).map(|id| entry(2, id, 0, 20)));
    write_run(&path, 1, &sorted(entries)).unwrap();

    let run = RunIndex::open(&path).unwrap();
    run.verify_collection(&collection(1)).unwrap();
    run.verify_collection(&collection(9)).unwrap();
    assert!(!run.is_invalid());

    // Damage collection 2's first block; collection 1 uses blocks 0 to 2.
    let mut bytes = fs::read(&path).unwrap();
    let meta = block_entry_at(&bytes, 3);
    let data = usize::try_from(read_u64(&bytes, meta.checked_add(32).unwrap()).unwrap()).unwrap();
    bytes[data] ^= 0xFF;
    let damaged = dir.join("damaged.run");
    fs::write(&damaged, &bytes).unwrap();

    let run = RunIndex::open(&damaged).unwrap();
    run.verify_collection(&collection(1))
        .expect("collection 1's blocks are intact");
    assert_eq!(run.verify_collection(&collection(2)), Err(RunCorrupt));
    assert!(run.is_invalid(), "a failed block invalidates the run");
    assert_eq!(run.verify_collection(&collection(1)), Err(RunCorrupt));
}

#[test]
fn writer_emits_default_pack_stats_when_omitted() {
    let dir = TestDir::new();
    let path = dir.join("default_stats.run");
    let mut writer = RunWriter::create(&path, vec![identity(1), identity(2)], 4).unwrap();
    writer.push(entry(0, 1, 0, 10)).unwrap();
    writer.push(entry(0, 2, 1, 20)).unwrap();
    writer.finish().unwrap().unwrap();

    let run = RunIndex::open(&path).unwrap();
    assert!(run.has_pack_stats());
    assert_eq!(run.stats_count(), 2);
    let stats: Vec<_> = run.pack_stats().collect();
    assert_eq!(stats.len(), 2);
    assert_eq!(stats[0].ordinal, 0);
    assert_eq!(stats[0].entry_count, 1);
    assert_eq!(stats[1].ordinal, 1);
    assert_eq!(stats[1].entry_count, 1);
}

#[test]
fn load_pool_runs_rejects_nonempty_run_without_pack_stats() {
    let dir = TestDir::new();
    let path = dir.join("no_stats.run");
    let mut writer = RunWriter::create(&path, vec![identity(1)], 4).unwrap();
    let entry = entry(0, 1, 0, 10);
    writer.push(entry).unwrap();
    writer.set_pack_stats(fixture_pack_stats(&[entry])).unwrap();
    writer.finish().unwrap().unwrap();

    let mut bytes = fs::read(&path).unwrap();
    // Drop the stats row and shrink meta_len too, so the run parses and only
    // the missing-stats condition remains.
    let old_meta_len = usize::try_from(read_u64(&bytes, 48).unwrap()).unwrap();
    let new_meta_len = old_meta_len.checked_sub(STATS_ENTRY_LEN).unwrap();
    bytes.truncate(bytes.len().checked_sub(STATS_ENTRY_LEN).unwrap());
    bytes[36..40].copy_from_slice(&0_u32.to_le_bytes());
    bytes[48..56].copy_from_slice(&u64::try_from(new_meta_len).unwrap().to_le_bytes());
    reseal_meta(&mut bytes);
    fs::write(&path, &bytes).unwrap();
    assert!(
        RunIndex::open(&path).is_ok(),
        "the fixture must parse so the missing-stats gate is what rejects it"
    );

    let pool = vec![identity(1)];
    let kept = load_pool_runs(&dir.0, &pool, true);
    assert!(
        kept.is_empty(),
        "nonempty run without pack stats must be rejected"
    );
    assert!(
        !path.exists(),
        "rejected run should be removed when writable"
    );
}

#[test]
fn merge_aborts_when_stop_signalled() {
    let dir = TestDir::new();
    let models = [random_pack(&mut Rng(1), 50), random_pack(&mut Rng(2), 50)];
    let sidecars: Vec<MappedPackIndex> = models
        .iter()
        .zip(1_u64..)
        .map(|(model, seq)| write_sidecar(&dir, seq, model))
        .collect();
    let sources: Vec<MergeSource<'_>> = sidecars.iter().map(MergeSource::Sidecar).collect();
    let path = dir.join("cancelled.run");
    let stop = Arc::new(AtomicBool::new(true));
    let options = MergeOptions {
        stop: Some(stop),
        ..MergeOptions::default()
    };
    let result = merge_to_path(&sources, &options, &path);
    assert!(matches!(result, Err(RunError::Cancelled)));
    assert!(!path.exists());
    let mut temp = path.into_os_string();
    temp.push(".tmp");
    assert!(!PathBuf::from(temp).exists());
}
