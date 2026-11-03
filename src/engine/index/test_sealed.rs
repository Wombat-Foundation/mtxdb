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

    let hit = set.lookup(&COLLECTION, &KEY).expect("newest locator");
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

    let hit = set.lookup(&COLLECTION, &KEY).expect("removal locator");
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

    let hit = set.lookup(&COLLECTION, &KEY).expect("live locator");
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

    assert!(set.lookup(&COLLECTION, &KEY).is_none());
    assert!(set.lookup(&OTHER_COLLECTION, &KEY).is_some());
}

#[test]
fn a_missing_key_is_none() {
    let set = locators(vec![pack(1, 1, FakeIndex::new(&[], 0))]);
    assert!(set.lookup(&COLLECTION, &KEY).is_none());
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
    let hit = set.lookup(&COLLECTION, &KEY).expect("replacement");
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
    assert!(set.lookup(&COLLECTION, &KEY).is_none());
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
    set.for_each_in_collection(&COLLECTION, |hit, _key| seen.push(hit.creation_seq));
    assert_eq!(seen, vec![2, 1]);
}

#[test]
fn empty_set_has_no_packs() {
    let set = SealedLocators::new();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert_eq!(set.mapped_bytes(), 0);
    assert!(set.lookup(&COLLECTION, &KEY).is_none());
}
