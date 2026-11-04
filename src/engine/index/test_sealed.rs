use std::collections::HashMap;
use std::sync::Arc;

use crate::index::packidx::PackIndexLocator;
use crate::packfile::PackId;

use super::*;

const COLLECTION: [u8; 16] = [0xC1; 16];
const OTHER_COLLECTION: [u8; 16] = [0xC2; 16];
const KEY: NodeId = [0x11; 16];

struct FakeIndex {
    entries: HashMap<([u8; 16], NodeId), PackIndexLocator>,
    bytes: u64,
}

impl FakeIndex {
    fn new(entries: &[([u8; 16], NodeId, PackIndexLocator)], bytes: u64) -> Self {
        Self {
            entries: entries
                .iter()
                .map(|(collection, key, locator)| ((*collection, *key), *locator))
                .collect(),
            bytes,
        }
    }
}

impl SealedIndex for FakeIndex {
    fn locator(&self, collection_id: &[u8; 16], logical_id: &NodeId) -> Option<PackIndexLocator> {
        self.entries.get(&(*collection_id, *logical_id)).copied()
    }

    fn for_each_in_collection(
        &self,
        collection_id: &[u8; 16],
        visit: &mut dyn FnMut(NodeId, PackIndexLocator),
    ) {
        for ((collection, key), locator) in &self.entries {
            if collection == collection_id {
                visit(*key, *locator);
            }
        }
    }

    fn mapped_bytes(&self) -> u64 {
        self.bytes
    }

    fn collection_entry_count(&self, collection_id: &[u8; 16]) -> Option<u64> {
        let count = self
            .entries
            .keys()
            .filter(|(collection, _)| collection == collection_id)
            .count();
        (count > 0).then(|| u64::try_from(count).unwrap_or(u64::MAX))
    }
}

fn pack(seq: u64, slot: u16, index: FakeIndex) -> SealedPack {
    SealedPack {
        slot,
        pack_id: pack_id(seq),
        creation_seq: seq,
        index: Arc::new(index),
    }
}

fn pack_id(seq: u64) -> PackId {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&seq.to_le_bytes());
    PackId(bytes)
}

fn locators(packs: Vec<SealedPack>) -> SealedLocators {
    let mut set = SealedLocators::new();
    for pack in packs {
        set.insert(pack);
    }
    set
}

#[test]
fn lookup_prefers_the_newest_pack() {
    let set = locators(vec![
        pack(
            1,
            1,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
        ),
        pack(
            2,
            2,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(20))], 0),
        ),
    ]);

    let hit = set
        .lookup(&COLLECTION, &KEY)
        .unwrap()
        .expect("newest locator");
    assert_eq!(hit.creation_seq, 2);
    assert_eq!(hit.slot, 2);
    assert_eq!(hit.locator, PackIndexLocator::Frame(20));
}

#[test]
fn a_removal_shadows_an_older_live_frame() {
    let set = locators(vec![
        pack(
            1,
            1,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
        ),
        pack(
            2,
            2,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Removed(20))], 0),
        ),
    ]);

    let hit = set
        .lookup(&COLLECTION, &KEY)
        .unwrap()
        .expect("removal locator");
    assert_eq!(hit.creation_seq, 2);
    assert!(hit.removed());
}

#[test]
fn an_older_removal_does_not_shadow_a_newer_live_frame() {
    let set = locators(vec![
        pack(
            1,
            1,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Removed(10))], 0),
        ),
        pack(
            2,
            2,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(20))], 0),
        ),
    ]);

    let hit = set
        .lookup(&COLLECTION, &KEY)
        .unwrap()
        .expect("live locator");
    assert_eq!(hit.creation_seq, 2);
    assert!(!hit.removed());
}

#[test]
fn lookup_is_isolated_by_collection() {
    let set = locators(vec![pack(
        1,
        1,
        FakeIndex::new(&[(OTHER_COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
    )]);

    assert!(set.lookup(&COLLECTION, &KEY).unwrap().is_none());
    assert!(set.lookup(&OTHER_COLLECTION, &KEY).unwrap().is_some());
}

#[test]
fn a_missing_key_is_none() {
    let set = locators(vec![pack(1, 1, FakeIndex::new(&[], 0))]);
    assert!(set.lookup(&COLLECTION, &KEY).unwrap().is_none());
}

#[test]
fn replacing_the_same_pack_is_idempotent() {
    let mut set = locators(vec![pack(
        1,
        1,
        FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
    )]);
    set.insert(pack(
        1,
        7,
        FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(99))], 0),
    ));

    assert_eq!(set.len(), 1);
    let hit = set.lookup(&COLLECTION, &KEY).unwrap().expect("replacement");
    assert_eq!(hit.slot, 7);
    assert_eq!(hit.locator, PackIndexLocator::Frame(99));
}

#[test]
fn removing_a_pack_drops_its_locators() {
    let mut set = locators(vec![pack(
        1,
        1,
        FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
    )]);
    assert!(set.remove_pack(pack_id(1)));
    assert!(set.lookup(&COLLECTION, &KEY).unwrap().is_none());
    assert!(!set.remove_pack(pack_id(1)));
}

#[test]
fn mapped_bytes_sums_each_sidecar() {
    let set = locators(vec![
        pack(1, 1, FakeIndex::new(&[], 100)),
        pack(2, 2, FakeIndex::new(&[], 250)),
    ]);
    assert_eq!(set.mapped_bytes(), 350);
}

#[test]
fn iteration_yields_newest_pack_first() {
    let set = locators(vec![
        pack(
            1,
            1,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(10))], 0),
        ),
        pack(
            2,
            2,
            FakeIndex::new(&[(COLLECTION, KEY, PackIndexLocator::Frame(20))], 0),
        ),
    ]);

    let mut seen = Vec::new();
    set.for_each_in_collection(&COLLECTION, |hit, _key| seen.push(hit.creation_seq))
        .unwrap();
    assert_eq!(seen, vec![2, 1]);
}

#[test]
fn empty_set_has_no_packs() {
    let set = SealedLocators::new();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert_eq!(set.mapped_bytes(), 0);
    assert!(set.lookup(&COLLECTION, &KEY).unwrap().is_none());
}

const KEY_2: NodeId = [0x22; 16];
const KEY_3: NodeId = [0x33; 16];

fn run_offset(delta: u32) -> u32 {
    u32::try_from(crate::packfile::HEADER_LEN)
        .unwrap()
        .checked_add(delta)
        .unwrap()
}

fn run_identity(seq: u64) -> crate::index::packidx::PackIndexIdentity {
    crate::index::packidx::PackIndexIdentity {
        pack_id: pack_id(seq),
        creation_seq: seq,
        pack_file_len: 1 << 20,
    }
}

/// A run over packs 2 and 3 holding `KEY` (pack 3) and `KEY_3` (pack 2),
/// served from slots 20 and 30.
fn sample_run(
    dir: &std::path::Path,
    fallback: crate::index::sealed::RunFallback,
) -> Arc<SealedRun> {
    use crate::index::run::{RunEntry, RunIndex, RunPackStats, RunWriter};
    let path = dir.join("sample.run");
    let mut writer = RunWriter::create(&path, vec![run_identity(2), run_identity(3)], 4).unwrap();
    let mut entries = [
        RunEntry {
            collection_id: COLLECTION,
            logical_id: KEY,
            ordinal: 1,
            locator: PackIndexLocator::Frame(run_offset(30)),
        },
        RunEntry {
            collection_id: COLLECTION,
            logical_id: KEY_3,
            ordinal: 0,
            locator: PackIndexLocator::Frame(run_offset(20)),
        },
    ];
    entries.sort_by_key(|entry| entry.logical_id);
    for entry in entries {
        writer.push(entry).unwrap();
    }
    writer
        .set_pack_stats(vec![
            RunPackStats {
                ordinal: 0,
                collection_id: COLLECTION,
                entry_count: 1,
                disk_bytes: 0,
            },
            RunPackStats {
                ordinal: 1,
                collection_id: COLLECTION,
                entry_count: 1,
                disk_bytes: 0,
            },
        ])
        .unwrap();
    writer.finish().unwrap().unwrap();
    let packs = vec![
        RunPack {
            slot: 20,
            pack_id: pack_id(2),
            creation_seq: 2,
        },
        RunPack {
            slot: 30,
            pack_id: pack_id(3),
            creation_seq: 3,
        },
    ];
    Arc::new(SealedRun::new(Arc::new(RunIndex::open(&path).unwrap()), packs, fallback).unwrap())
}

fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mtxdb-sealed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn runs_and_unmerged_packs_are_walked_newest_first() {
    let dir = scratch_dir("order");
    let mut set = locators(vec![
        // Older than the run: shadowed by it for KEY, sole source of KEY_2.
        pack(
            1,
            10,
            FakeIndex::new(
                &[
                    (COLLECTION, KEY, PackIndexLocator::Frame(10)),
                    (COLLECTION, KEY_2, PackIndexLocator::Frame(11)),
                ],
                0,
            ),
        ),
        // Newer than the run: shadows it for KEY_3.
        pack(
            4,
            40,
            FakeIndex::new(&[(COLLECTION, KEY_3, PackIndexLocator::Removed(40))], 0),
        ),
    ]);
    set.insert_run(sample_run(&dir, Box::new(|| Err("unused".into()))));

    let hit = set.lookup(&COLLECTION, &KEY).unwrap().unwrap();
    assert_eq!((hit.slot, hit.creation_seq), (30, 3), "run shadows pack 1");
    let hit = set.lookup(&COLLECTION, &KEY_2).unwrap().unwrap();
    assert_eq!((hit.slot, hit.creation_seq), (10, 1));
    let hit = set.lookup(&COLLECTION, &KEY_3).unwrap().unwrap();
    assert_eq!((hit.slot, hit.creation_seq), (40, 4), "pack 4 shadows run");
    assert!(hit.removed());

    let mut order = Vec::new();
    set.for_each_in_collection(&COLLECTION, |hit, _| order.push(hit.creation_seq))
        .unwrap();
    let mut newest_first = order.clone();
    newest_first.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(order, newest_first, "scan yields newer sources first");
    assert!(set.contains_pack(pack_id(2)) && set.contains_slot(30));
    assert_eq!(set.len(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_damaged_run_without_fallback_fails_instead_of_missing() {
    let dir = scratch_dir("no_fallback");
    let run = sample_run(&dir, Box::new(|| Err("sidecars unavailable".into())));
    // Damage the run's only block after open.
    let path = dir.join("sample.run");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[crate::index::run::RUN_HEADER_LEN] ^= 0xFF;
    // Replace the file rather than writing through the live mapping.
    let damaged = dir.join("damaged.run");
    std::fs::write(&damaged, &bytes).unwrap();
    let packs = run.packs().to_vec();
    let damaged_run = SealedRun::new(
        Arc::new(crate::index::run::RunIndex::open(&damaged).unwrap()),
        packs,
        Box::new(|| Err("sidecars unavailable".into())),
    )
    .unwrap();
    let mut set = SealedLocators::new();
    set.insert_run(Arc::new(damaged_run));
    assert!(
        set.lookup(&COLLECTION, &KEY).is_err(),
        "a damaged run must never report a miss"
    );
    assert!(set.for_each_in_collection(&COLLECTION, |_, _| {}).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn damaged_run_fallback_retries_after_transient_failure() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let dir = scratch_dir("transient_fallback");
    let run = sample_run(&dir, Box::new(|| Err("unused".into())));
    let path = dir.join("sample.run");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[crate::index::run::RUN_HEADER_LEN] ^= 0xFF;
    let damaged = dir.join("damaged.run");
    std::fs::write(&damaged, &bytes).unwrap();
    let packs = run.packs().to_vec();

    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_clone = Arc::clone(&attempts);

    let damaged_run = SealedRun::new(
        Arc::new(crate::index::run::RunIndex::open(&damaged).unwrap()),
        packs,
        Box::new(move || {
            let attempt = attempts_clone.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                Err("transient load failure".into())
            } else {
                Ok(vec![
                    pack(
                        2,
                        20,
                        FakeIndex::new(
                            &[(COLLECTION, KEY_3, PackIndexLocator::Frame(run_offset(20)))],
                            0,
                        ),
                    ),
                    pack(
                        3,
                        30,
                        FakeIndex::new(
                            &[(COLLECTION, KEY, PackIndexLocator::Frame(run_offset(30)))],
                            0,
                        ),
                    ),
                ])
            }
        }),
    )
    .unwrap();

    let mut set = SealedLocators::new();
    set.insert_run(Arc::new(damaged_run));

    // First attempt fails due to transient error and must not permanently cache failure
    assert!(set.lookup(&COLLECTION, &KEY).is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);

    // Second attempt retries and succeeds
    let hit = set.lookup(&COLLECTION, &KEY).unwrap().unwrap();
    assert_eq!((hit.slot, hit.creation_seq), (30, 3));
    assert_eq!(attempts.load(Ordering::SeqCst), 2);

    // Subsequent lookups reuse the cached successful fallback
    let hit = set.lookup(&COLLECTION, &KEY_3).unwrap().unwrap();
    assert_eq!((hit.slot, hit.creation_seq), (20, 2));
    assert_eq!(attempts.load(Ordering::SeqCst), 2);

    let _ = std::fs::remove_dir_all(&dir);
}
