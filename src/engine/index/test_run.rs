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
                disk_bytes: 0,
            },
        )
        .collect();
    let path = dir.join(&format!("pack-{seq}.idx"));
    PackIndex::build(pack, inputs)
        .unwrap()
        .write_atomic(&path)
        .unwrap();
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

/// Write `entries` (sorted) as a run over packs `1..=pack_count` with small
/// blocks so tests cross block boundaries.
fn write_run(path: &Path, pack_count: u64, entries: &[RunEntry]) -> Option<RunSummary> {
    let packs = (1..=pack_count).map(identity).collect();
    let mut writer = RunWriter::create(path, packs, 4).unwrap();
    for entry in entries {
        writer.push(*entry).unwrap();
    }
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
    let crc_at = bytes.len().checked_sub(4).unwrap();
    bytes[crc_at..].copy_from_slice(&block_crc.to_le_bytes());
    // The fence must still match the (new) first id for the swap alone to be
    // the defect.
    let fence_at = bytes.len().checked_sub(BLOCK_ENTRY_LEN).unwrap();
    bytes[fence_at..fence_at.checked_add(ID_LEN).unwrap()].copy_from_slice(&b);
    let meta_at = usize::try_from(read_u64(&bytes, 40).unwrap()).unwrap();
    bytes[CRC_AT..CRC_AT.checked_add(4).unwrap()].fill(0);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..RUN_HEADER_LEN]);
    hasher.update(&bytes[meta_at..]);
    let meta_crc = hasher.finalize();
    bytes[CRC_AT..CRC_AT.checked_add(4).unwrap()].copy_from_slice(&meta_crc.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    let run = RunIndex::open(&path).unwrap();
    assert_eq!(run.verify_all(), Err(RunCorrupt));
}

#[test]
fn merge_keeps_the_newest_version_against_a_model() {
    for seed in 0..40_u64 {
        let dir = TestDir::new();
        let mut rng = Rng(seed);
        let pack_count = rng.below(7).checked_add(1).unwrap();
        let models: Vec<PackModel> = (0..pack_count).map(|_| random_pack(&mut rng, 60)).collect();
        let sidecars: Vec<MappedPackIndex> = models
            .iter()
            .zip(1_u64..)
            .map(|(model, seq)| write_sidecar(&dir, seq, model))
            .collect();

        // Merge the older half into a run first, so the final merge mixes a
        // run with sidecars and has to remap the run's pack ordinals.
        let split = usize::try_from(rng.below(pack_count)).unwrap();
        let older: Vec<MergeSource<'_>> =
            sidecars[..split].iter().map(MergeSource::Sidecar).collect();
        let older_run = if older.is_empty() {
            None
        } else {
            let path = dir.join("older.run");
            merge_to_path(&older, MergeOptions::default(), &path)
                .unwrap()
                .map(|_| RunIndex::open(&path).unwrap())
        };
        let mut sources: Vec<MergeSource<'_>> =
            sidecars[split..].iter().map(MergeSource::Sidecar).collect();
        if let Some(run) = &older_run {
            sources.push(MergeSource::Run(run));
        }
        let options = MergeOptions {
            drop_removals: false,
            block_entries: 3,
        };
        let path = dir.join("merged.run");
        if older_run.is_none() && split > 0 {
            continue;
        }
        let merged = merge_to_path(&sources, options, &path).unwrap().unwrap();
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
    }
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
    merge_to_path(&sources, MergeOptions::default(), &kept)
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
    merge_to_path(&sources, options, &dropped).unwrap().unwrap();
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
        merge_to_path(&[MergeSource::Sidecar(&sidecar)], options, &path).unwrap(),
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
        merge_to_path(&twice, MergeOptions::default(), &path),
        Err(RunError::OverlappingInputs)
    ));

    // A run over packs 1 and 3 cannot be merged with pack 2.
    let outer = dir.join("outer.run");
    merge_to_path(
        &[
            MergeSource::Sidecar(&sidecars[0]),
            MergeSource::Sidecar(&sidecars[2]),
        ],
        MergeOptions::default(),
        &outer,
    )
    .unwrap()
    .unwrap();
    let outer = RunIndex::open(&outer).unwrap();
    assert!(matches!(
        merge_to_path(
            &[MergeSource::Run(&outer), MergeSource::Sidecar(&sidecars[1])],
            MergeOptions::default(),
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
        merge_to_path(&[MergeSource::Run(&run)], MergeOptions::default(), &path),
        Err(RunError::Corrupt(RunCorrupt))
    ));
    assert!(!path.exists());
}
