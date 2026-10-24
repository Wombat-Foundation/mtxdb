//! Reopen strategies for one sealed pack, side by side.
//!
//! Builds one store holding a single pack of roughly `MTXDB_IA_MIB` MiB and
//! compares, warm and cold page cache:
//!
//! 1. `current`: `PackfileStorage::open` (checkpoint decode + pack recovery).
//! 2. `scan`: `scan_and_recover_packfile` (what a rebuild without a checkpoint
//!    pays), alone and followed by the sort an index build needs.
//! 3. `.idx`: a sorted, dense, immutable index (header, fanout[256], sorted
//!    16-byte ids, u32 offsets, CRC32 trailer) built from the scan: build time,
//!    size, reopen (read + CRC), point lookups, sorted batch lookups.
//! 4. `trusted`: reopen that stats the pack, compares its length with a
//!    manifest, and loads the `.idx`, never reading the pack. `digest` shows what
//!    verifying the pack's bytes at reopen would cost instead.
//! 5. `sidecar`: the real [`mtxdb::index::packidx`] v1 index built from the same
//!    scan, with its exact size, lower-bound size projections for truncated-id
//!    and sparse variants, frame order in this synthetic arrival-order pack,
//!    and mapped RSS after a small random-access probe using `MADV_RANDOM`.
//!
//! The `.idx` here is a prototype of the layout, one collection only (ids are the
//! frame hashes); it is not crash-ordering-safe, it measures cost. The question it
//! answers: can a trusted sealed pack reopen without reading the whole pack?
//!
//! ```text
//! MTXDB_BENCH_ROOT=/run/media/shane/shane4tb-ent/bench-scratch \
//!   cargo bench --manifest-path benches/Cargo.toml --bench index_artifacts
//! ```
//!
//! Env knobs: `MTXDB_IA_MIB` (pack size, default 256), `MTXDB_IA_FRAMES`
//! (`tiny`: 16-byte payloads, or `mixed`: 1-64 KiB; default both, comma list),
//! `MTXDB_IA_LOOKUPS` (default 100000), `MTXDB_BENCH_ROOT`. Cold runs use
//! `vmtouch -e` after a sync and report whether the eviction took.
#[path = "support/bench_evict.rs"]
mod bench_evict;

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mtxdb::index::packidx::{
    PackIndex, PackIndexIdentity, PackIndexInput, PACK_INDEX_COLLECTION_ENTRY_LEN,
    PACK_INDEX_FANOUT_LEN, PACK_INDEX_HEADER_LEN, PACK_INDEX_SECTION_ENTRY_LEN,
};
use mtxdb::index::{IndexConfig, InsertError, LossyIndex};
use mtxdb::packfile::{read_header, scan_and_recover_packfile};
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xE7; 16];
const MAGIC: &[u8; 8] = b"MTXIDX01";
/// A truncated id keeps only the 8 bytes after the fanout byte, plus the offset.
const TRUNCATED_ENTRY_BYTES: u64 = 8 + 4;
/// Sample one entry in ten in the sparse lower-bound projection.
const SPARSE_STRIDE: u64 = 10;
/// Keep the RSS probe small relative to the number of pages in the sidecar.
const RSS_PROBES: u64 = 64;

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn add_usize(left: usize, right: usize) -> usize {
    left.checked_add(right).expect("benchmark usize addition")
}

fn mul_usize(left: usize, right: usize) -> usize {
    left.checked_mul(right)
        .expect("benchmark usize multiplication")
}

fn add_u64(left: u64, right: u64) -> u64 {
    left.checked_add(right).expect("benchmark u64 addition")
}

fn sub_u64(left: u64, right: u64) -> u64 {
    left.checked_sub(right).expect("benchmark u64 subtraction")
}

fn to_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark value fits in u32"))
}

fn usize_f64(value: usize) -> f64 {
    to_f64(u64::try_from(value).expect("benchmark usize fits in u64"))
}

fn mib_f64(bytes: u64) -> f64 {
    to_f64(bytes) / to_f64(1_u64 << 20)
}

fn sample_index(seed: u64, count: u64) -> u64 {
    mix(seed)
        .checked_rem(count)
        .expect("nonempty benchmark index")
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Well-spread id: both halves are hashed, so no constant prefix.
fn node_id(seq: u64) -> NodeId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&mix(seq).to_le_bytes());
    id[8..].copy_from_slice(&mix(seq ^ 0xabcd_ef01).to_le_bytes());
    id
}

fn timed<T>(work: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let out = work();
    (out, started.elapsed())
}

fn pack_path(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("read store dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "pack"))
        .expect("the store has a pack")
}

fn dir_pack_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .expect("read store dir")
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "pack"))
        .map(|e| e.metadata().map_or(0, |m| m.len()))
        .sum()
}

/// Sizes of the `index.*` files, itemized: the checkpoint, each delta log, anything else.
fn index_files(dir: &Path) -> String {
    let mut parts: Vec<String> = std::fs::read_dir(dir)
        .expect("read store dir")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("index."))
        .map(|e| {
            format!(
                "{} {:.2} MiB",
                e.file_name().to_string_lossy(),
                mib_f64(e.metadata().map_or(0, |m| m.len()))
            )
        })
        .collect();
    parts.sort();
    parts.join(", ")
}

fn checkpoint_bytes(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("index.checkpoint")).map_or(0, |m| m.len())
}

/// Resident bytes for one file mapping from Linux `/proc/self/smaps`.
/// This is this process's mapped RSS, not total system page-cache residency.
fn mapped_rss_bytes(path: &Path) -> Option<u64> {
    fn is_mapping_header(line: &str) -> bool {
        let Some(address) = line.split_ascii_whitespace().next() else {
            return false;
        };
        let Some((start, end)) = address.split_once('-') else {
            return false;
        };
        u64::from_str_radix(start, 16).is_ok() && u64::from_str_radix(end, 16).is_ok()
    }

    let path = path.canonicalize().ok()?;
    let path = path.to_string_lossy();
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut in_mapping = false;
    let mut found = false;
    let mut rss_kib = 0_u64;
    for line in smaps.lines() {
        if is_mapping_header(line) {
            in_mapping = line.ends_with(path.as_ref());
            found |= in_mapping;
        } else if in_mapping {
            if let Some(value) = line.strip_prefix("Rss:") {
                rss_kib =
                    rss_kib.saturating_add(value.split_ascii_whitespace().next()?.parse().ok()?);
            }
        }
    }
    found.then(|| rss_kib.saturating_mul(1024))
}

/// `(RssAnon, RssFile)` in bytes from `/proc/self/status` (Linux only).
/// Anonymous pages are private heap; file pages include checkpoint mappings.
fn process_rss_split() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| -> Option<u64> {
        let line = status.lines().find(|l| l.starts_with(name))?;
        let kib: u64 = line.split_ascii_whitespace().nth(1)?.parse().ok()?;
        Some(kib.saturating_mul(1024))
    };
    Some((field("RssAnon:")?, field("RssFile:")?))
}

fn pack_index_identity(pack: &Path, pack_len: u64) -> PackIndexIdentity {
    let mut file = std::fs::File::open(pack).expect("open pack header");
    let header = read_header(&mut file)
        .expect("read pack header")
        .expect("pack header exists");
    PackIndexIdentity {
        pack_id: header.pack_id,
        creation_seq: u64::from(header.creation_seq),
        pack_file_len: pack_len,
    }
}

fn build_pack_index(
    dir: &Path,
    pack: &Path,
    pack_len: u64,
    entries: &[([u8; 16], [u8; 16], u64)],
) -> (PathBuf, PackIndexIdentity, u64, Duration, Duration) {
    let inputs = entries
        .iter()
        .map(|(collection_id, logical_id, offset)| PackIndexInput {
            collection_id: *collection_id,
            logical_id: *logical_id,
            offset: *offset,
            removed: false,
        })
        .collect();
    let identity = pack_index_identity(pack, pack_len);
    let (index, build_t) = timed(|| PackIndex::build(identity, inputs).expect("build PackIndex"));
    let bytes = u64::try_from(index.as_bytes().len()).expect("PackIndex length fits u64");
    let path = dir.join("pack.mtxidx");
    let ((), write_t) = timed(|| {
        index.write_atomic(&path).expect("write PackIndex");
    });
    (path, identity, bytes, build_t, write_t)
}

fn report_pack_index(
    path: &Path,
    identity: PackIndexIdentity,
    entries: &[([u8; 16], [u8; 16], u64)],
    bytes: u64,
    build: Duration,
    write: Duration,
) {
    let count = u64::try_from(entries.len()).expect("entry count fits u64");
    assert!(count > 0, "sidecar report needs at least one record");
    let mut collection_ids: Vec<[u8; 16]> = entries.iter().map(|entry| entry.0).collect();
    collection_ids.sort_unstable();
    collection_ids.dedup();
    let collections = collection_ids.len();
    let overhead = u64::try_from(add_usize(
        add_usize(
            PACK_INDEX_HEADER_LEN,
            mul_usize(PACK_INDEX_COLLECTION_ENTRY_LEN, collections),
        ),
        mul_usize(PACK_INDEX_FANOUT_LEN, collections),
    ))
    .expect("PackIndex overhead fits u64");
    let truncated = overhead.saturating_add(count.saturating_mul(TRUNCATED_ENTRY_BYTES));
    let sparse_count = count.saturating_add(SPARSE_STRIDE - 1) / SPARSE_STRIDE;
    let sparse = overhead.saturating_add(sparse_count.saturating_mul(TRUNCATED_ENTRY_BYTES));
    let dense_projection =
        overhead.saturating_add(count.saturating_mul(PACK_INDEX_SECTION_ENTRY_LEN as u64));
    assert_eq!(
        bytes, dense_projection,
        "v1 layout projection matches encoding"
    );
    assert!(
        truncated < bytes,
        "truncated-id projection should be smaller"
    );
    assert!(sparse < truncated, "sparse projection should be smaller");
    println!(
        "  PackIndex sidecar {:.2} MiB ({:.2} B/record over {count} records, {collections} collection(s)); \
         build {:.0} ms, write+fsync {:.0} ms",
        mib_f64(bytes),
        to_f64(bytes) / to_f64(count),
        ms(build),
        ms(write),
    );
    println!(
        "    layout: {PACK_INDEX_HEADER_LEN} B header + {PACK_INDEX_COLLECTION_ENTRY_LEN} B/collection \
         directory + {PACK_INDEX_FANOUT_LEN} B/collection fanout + {PACK_INDEX_SECTION_ENTRY_LEN} B/record \
         (CRC is in the header)"
    );
    println!(
        "    truncated id (8 B): {:.2} MiB ({:.2} B/record, {:.0}% of dense)",
        mib_f64(truncated),
        to_f64(truncated) / to_f64(count),
        100.0 * to_f64(truncated) / to_f64(bytes),
    );
    println!(
        "    sparse + truncated (1 in {SPARSE_STRIDE}, lower bound; omits scan-bound metadata): \
         {:.2} MiB ({:.2} B/record, {:.0}% of dense)",
        mib_f64(sparse),
        to_f64(sparse) / to_f64(count),
        100.0 * to_f64(sparse) / to_f64(bytes),
    );

    // A sparse index is only valid if frames sit in id order in the pack. Sort
    // by (collection, id) and check the offsets increase along that order.
    let mut ordered = entries.to_vec();
    ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut backwards = 0usize;
    let mut worst_jump = 0u64;
    for pair in ordered.windows(2) {
        if pair[1].2 <= pair[0].2 {
            backwards = add_usize(backwards, 1);
            worst_jump = worst_jump.max(pair[0].2.saturating_sub(pair[1].2));
        }
    }
    let pairs = ordered.len().saturating_sub(1);
    println!(
        "  synthetic arrival-order pack frames in logical-id order: {} ({backwards} of {pairs} \
         adjacent pairs go backwards, worst back-jump {worst_jump} B)",
        if backwards == 0 { "yes" } else { "no" },
    );

    // `open_mmap` validates metadata and the directory without reading the
    // complete body. Random advice limits sequential readahead, though RSS can
    // still include kernel fault-around and shared page-cache pages.
    let mapped = PackIndex::open_mmap(path, identity).expect("mmap PackIndex");
    #[cfg(unix)]
    mapped.advise_random().expect("advise random mmap access");
    let before = mapped_rss_bytes(path);
    let probes = RSS_PROBES.min(count);
    for i in 0..probes {
        let seq = sample_index(i ^ 0x5041_434b, count);
        assert!(
            mapped.get(&entries[0].0, &node_id(seq)).is_some(),
            "PackIndex lookup hits"
        );
    }
    let after = mapped_rss_bytes(path);
    let show = |value: Option<u64>| {
        value.map_or_else(
            || "unavailable".to_owned(),
            |n| format!("{:.2} MiB", mib_f64(n)),
        )
    };
    println!(
        "  PackIndex mmap: mapped RSS before {probes} random lookups {}, after {} \
         (Linux smaps; shared/reclaimable pages; probe result, not a memory-cost estimate)",
        show(before),
        show(after),
    );
}

fn report_lossy_index(
    label: &str,
    entries: usize,
    capacity: u32,
    heap_bytes: usize,
    mapped_bytes: usize,
) {
    let load = 100.0 * usize_f64(entries) / to_f64(u64::from(capacity));
    let per_entry = |bytes: usize| usize_f64(bytes) / usize_f64(entries);
    println!(
        "  LossyIndex {label}: {entries} records / {capacity} slots ({load:.1}% load); \
         private heap {:.2} MiB ({:.1} B/record), mapped slots {:.2} MiB ({:.1} B/record)",
        mib_f64(u64::try_from(heap_bytes).expect("heap bytes fit u64")),
        per_entry(heap_bytes),
        mib_f64(u64::try_from(mapped_bytes).expect("mapped bytes fit u64")),
        per_entry(mapped_bytes),
    );
}

fn insert_or_grow(index: &mut LossyIndex, key: &[u8; 16], offset: u64) {
    match index.insert(key, 0, offset) {
        Ok(()) => {}
        Err(InsertError::TableFull) => {
            *index = index.grow().expect("complete owned index grows");
            index.insert(key, 0, offset).expect("grown index has room");
        }
        Err(error) => panic!("unexpected index insert failure: {error}"),
    }
}

/// Build an owned index with the same 64-slot floor and 75% growth threshold
/// as `PackfileStorage`'s default collection index.
fn build_lossy_index(entries: &[([u8; 16], [u8; 16], u64)]) -> LossyIndex {
    let mut index = LossyIndex::with_config(64, IndexConfig::default());
    for (_, id, offset) in entries {
        insert_or_grow(&mut index, id, *offset);
    }
    index
}

/// Exercise the first insertion past a full table's load threshold so the
/// resulting owned index is measured immediately after its doubling.
fn build_post_grow_lossy_index(
    entries: &[([u8; 16], [u8; 16], u64)],
    final_capacity: u32,
) -> LossyIndex {
    let old_capacity = final_capacity / 2;
    let threshold = usize::try_from(old_capacity)
        .expect("capacity fits usize")
        .checked_mul(75)
        .and_then(|value| value.checked_div(100))
        .expect("load threshold fits usize");
    assert!(
        threshold < entries.len(),
        "workload has an entry to trigger growth"
    );
    let mut index = LossyIndex::with_config(
        usize::try_from(old_capacity).expect("capacity fits usize"),
        IndexConfig::default(),
    );
    for (_, id, offset) in &entries[..threshold] {
        index
            .insert(id, 0, *offset)
            .expect("pre-growth index has room");
    }
    let (_, id, offset) = entries[threshold];
    assert!(
        matches!(index.insert(&id, 0, offset), Err(InsertError::TableFull)),
        "the next insert crosses the old table's load threshold"
    );
    index = index.grow().expect("complete index grows at threshold");
    index.insert(&id, 0, offset).expect("grown index has room");
    assert_eq!(
        index.len(),
        threshold.checked_add(1).expect("entry count fits usize"),
        "the triggering insert is included after growth"
    );
    assert_eq!(
        index.capacity(),
        final_capacity,
        "growth doubles the old capacity"
    );
    index
}

fn sync_and_evict(dir: &Path) -> bool {
    let _ = std::process::Command::new("sync").status();
    bench_evict::evict_dir(dir)
}

/// Fill one store until its pack reaches `target` bytes; returns the record count.
fn build(dir: &Path, target: u64, mixed: bool) -> u64 {
    let _ = std::fs::remove_dir_all(dir);
    let store = PackfileStorage::open(dir.to_path_buf()).expect("open new store");
    let tiny = NodeData::from_slice(&[0x5a; 16]);
    // Incompressible filler: the pack compresses frames, constant bytes would vanish.
    let big: Vec<u8> = (0..(64u64 << 10) / 8)
        .flat_map(|i| mix(i).to_le_bytes())
        .collect();
    let mut seq = 0u64;
    let mut written = 0u64;
    while written < target {
        let len = if mixed {
            add_usize(
                usize::try_from(mix(seq ^ 77) % (64_u64 << 10)).expect("len"),
                1,
            )
        } else {
            16
        };
        let data = if mixed {
            let start = mul_usize(usize::try_from(mix(seq) % 1024).expect("start"), 8);
            let available = big.len().checked_sub(start).expect("start within filler");
            let end = add_usize(start, len.min(available));
            NodeData::from_slice(&big[start..end])
        } else {
            tiny.clone()
        };
        store
            .put(&COLLECTION, &node_id(seq), &data)
            .expect("put record");
        seq = add_u64(seq, 1);
        let frame_bytes = add_u64(u64::try_from(len).expect("length fits in u64"), 61);
        written = add_u64(written, sub_u64(frame_bytes, 16));
        if seq.is_multiple_of(100_000) {
            store.sync_all().expect("sync_all");
            written = dir_pack_bytes(dir);
        }
    }
    store.sync_all().expect("sync_all");
    seq
}

enum Backing {
    Heap(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Heap(v) => v,
            Self::Mapped(m) => m,
        }
    }
}

struct Idx {
    bytes: Backing,
    count: usize,
}

impl Idx {
    const FAN: usize = 16;
    fn ids_at() -> usize {
        add_usize(Self::FAN, mul_usize(256, 4))
    }
    fn offs_at(&self) -> usize {
        add_usize(Self::ids_at(), mul_usize(self.count, 16))
    }
    fn fan(&self, i: usize) -> usize {
        let at = add_usize(Self::FAN, mul_usize(i, 4));
        let end = add_usize(at, 4);
        usize::try_from(u32::from_le_bytes(
            self.bytes[at..end].try_into().expect("fan"),
        ))
        .expect("fan fits in usize")
    }
    fn id(&self, i: usize) -> &[u8] {
        let at = add_usize(Self::ids_at(), mul_usize(i, 16));
        let end = add_usize(at, 16);
        &self.bytes[at..end]
    }
    fn off(&self, i: usize) -> u64 {
        let at = add_usize(self.offs_at(), mul_usize(i, 4));
        let end = add_usize(at, 4);
        u64::from(u32::from_le_bytes(
            self.bytes[at..end].try_into().expect("off"),
        ))
    }
    fn get(&self, key: &[u8; 16]) -> Option<u64> {
        let b = usize::from(key[0]);
        let mut lo = if b == 0 {
            0
        } else {
            self.fan(b.checked_sub(1).expect("nonzero fanout byte"))
        };
        let mut hi = self.fan(b);
        while lo < hi {
            let span = hi.checked_sub(lo).expect("ordered binary-search bounds");
            let mid = add_usize(lo, span / 2);
            match self.id(mid).cmp(&key[..]) {
                std::cmp::Ordering::Less => lo = add_usize(mid, 1),
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(self.off(mid)),
            }
        }
        None
    }
    /// Merge-walk sorted `keys` against the index; returns the number found.
    fn batch(&self, keys: &[[u8; 16]]) -> usize {
        let mut pos = 0usize;
        let mut found = 0;
        for key in keys {
            while pos < self.count && self.id(pos) < &key[..] {
                pos = add_usize(pos, 1);
            }
            if pos < self.count && self.id(pos) == &key[..] {
                found = add_usize(found, 1);
            }
        }
        found
    }
}

fn build_idx(entries: &[([u8; 16], [u8; 16], u64)]) -> Vec<u8> {
    let mut sorted: Vec<([u8; 16], u32)> = entries
        .iter()
        .map(|e| (e.1, u32::try_from(e.2).expect("offset fits u32")))
        .collect();
    sorted.sort_unstable();
    let entry_bytes = mul_usize(sorted.len(), 20);
    let capacity = add_usize(add_usize(add_usize(16, 1024), entry_bytes), 4);
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    let mut fan = [0u32; 256];
    for (id, _) in &sorted {
        let slot = usize::from(id[0]);
        fan[slot] = fan[slot].checked_add(1).expect("fanout count");
    }
    let mut run = 0u32;
    for f in &mut fan {
        run = run.checked_add(*f).expect("fanout total");
        *f = run;
        out.extend_from_slice(&run.to_le_bytes());
    }
    for (id, _) in &sorted {
        out.extend_from_slice(id);
    }
    for (_, off) in &sorted {
        out.extend_from_slice(&off.to_le_bytes());
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// Header-only validation: magic, count, and that the file length matches the layout.
fn check_header(bytes: &[u8]) -> usize {
    assert_eq!(&bytes[..8], MAGIC, "idx magic");
    let count = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().expect("count")))
        .expect("count");
    assert_eq!(
        bytes.len(),
        add_usize(add_usize(add_usize(16, 1024), mul_usize(count, 20)), 4,),
        "idx length matches its layout"
    );
    count
}

/// Map the index and check only its header (no CRC, no full read).
fn map_idx(path: &Path) -> Idx {
    let file = std::fs::File::open(path).expect("open idx");
    // SAFETY: the bench owns the file and nothing truncates it while mapped.
    #[allow(unsafe_code)]
    let map = unsafe { memmap2::Mmap::map(&file) }.expect("mmap idx");
    let count = check_header(&map);
    Idx {
        bytes: Backing::Mapped(map),
        count,
    }
}

/// Read only the header and fanout (1040 bytes): the metadata-only reopen.
fn header_only(path: &Path) -> usize {
    use std::io::Read;
    let mut head = [0u8; 16 + 1024];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .expect("read idx header");
    assert_eq!(&head[..8], MAGIC, "idx magic");
    usize::try_from(u64::from_le_bytes(head[8..16].try_into().expect("count"))).expect("count")
}

fn load_idx(path: &Path) -> Idx {
    let mut bytes = std::fs::read(path).expect("read idx");
    let tail = bytes.len().checked_sub(4).expect("idx crc trailer");
    let crc = u32::from_le_bytes(bytes[tail..].try_into().expect("crc"));
    assert_eq!(crc, crc32fast::hash(&bytes[..tail]), "idx crc");
    assert_eq!(&bytes[..8], MAGIC, "idx magic");
    bytes.truncate(tail);
    let count = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().expect("count")))
        .expect("count");
    Idx {
        bytes: Backing::Heap(bytes),
        count,
    }
}

fn prepare(dir: &Path, cold: bool) {
    if cold && !sync_and_evict(dir) {
        println!("  (eviction failed; cold numbers are really warm)");
    }
}

fn measure_reopen(dir: &Path, pack: &Path, idx_path: &Path, manifest_len: u64, cold: bool) {
    let state = if cold { "cold" } else { "warm" };
    prepare(dir, cold);
    let (store, t_open) = timed(|| PackfileStorage::open(dir.to_path_buf()).expect("open"));
    let first = timed(|| store.get(&COLLECTION, &node_id(0)).expect("get").is_some()).1;
    let timings = store.open_timings().expect("timings");
    drop(store);
    println!(
        "  [{state}] current open via {:?} {:.0} ms (pack recovery {:.0}, checkpoint decode {:.0}, \
         index materialization {:.0}), first read {:.2} ms{}",
        timings.path,
        ms(t_open),
        ms(timings.packfile_recovery),
        ms(timings.checkpoint_decode),
        ms(timings.index_materialization),
        ms(first),
        timings
            .checkpoint_reject
            .map(|r| format!(" [checkpoint rejected: {r:?}]"))
            .unwrap_or_default(),
    );

    prepare(dir, cold);
    let (n, t) = timed(|| scan_and_recover_packfile(pack).expect("scan").len());
    println!("  [{state}] scan rebuild {:.0} ms ({n} entries)", ms(t));

    prepare(dir, cold);
    let (_, t) = timed(|| load_idx(idx_path));
    println!("  [{state}] .idx reopen (read + CRC) {:.1} ms", ms(t));

    prepare(dir, cold);
    let (_, t) = timed(|| {
        assert_eq!(
            std::fs::metadata(pack).expect("stat").len(),
            manifest_len,
            "manifest length"
        );
        black_box(header_only(idx_path))
    });
    println!(
        "  [{state}] trusted, metadata-only (stat pack + idx header/fanout) {:.2} ms",
        ms(t)
    );

    prepare(dir, cold);
    let (_, t) = timed(|| {
        assert_eq!(
            std::fs::metadata(pack).expect("stat").len(),
            manifest_len,
            "manifest length"
        );
        black_box(map_idx(idx_path).count)
    });
    println!(
        "  [{state}] trusted, mmap .idx (header check, no CRC) {:.2} ms",
        ms(t)
    );

    prepare(dir, cold);
    let (_, t) = timed(|| {
        assert_eq!(
            std::fs::metadata(pack).expect("stat").len(),
            manifest_len,
            "manifest length"
        );
        black_box(load_idx(idx_path).count)
    });
    println!("  [{state}] trusted, read + CRC .idx {:.1} ms", ms(t));

    prepare(dir, cold);
    let (_, t) = timed(|| black_box(crc32fast::hash(&std::fs::read(pack).expect("read"))));
    println!(
        "  [{state}] digest-verify reopen (read whole pack + CRC) {:.0} ms",
        ms(t)
    );
}

fn measure_cold_lookups(dir: &Path, idx_path: &Path, records: u64) {
    let probe: Vec<[u8; 16]> = (0..1000_u64)
        .map(|i| node_id(sample_index(i ^ 5, records)))
        .collect();
    prepare(dir, true);
    let fresh = map_idx(idx_path);
    let (found, t) = timed(|| probe.iter().filter(|k| fresh.get(k).is_some()).count());
    assert_eq!(found, probe.len(), "cold mmap lookups hit");
    println!(
        "  [cold] mmap: first 1000 random lookups {:.1} ms total",
        ms(t)
    );
}

fn measure_warm_lookups(
    dir: &Path,
    pack: &Path,
    idx_path: &Path,
    idx: &Idx,
    records: u64,
    lookups: usize,
) {
    let keys: Vec<[u8; 16]> = (0..u64::try_from(lookups).expect("lookup count fits in u64"))
        .map(|i| node_id(sample_index(i, records)))
        .collect();
    let (found, t) = timed(|| keys.iter().filter(|k| idx.get(k).is_some()).count());
    assert_eq!(found, lookups, "every point lookup hits");
    println!(
        "  point lookups: {:.0} ns each over {lookups} random keys",
        t.as_secs_f64() * 1e9 / usize_f64(lookups)
    );

    let mapped = map_idx(idx_path);
    let (found, t) = timed(|| keys.iter().filter(|k| mapped.get(k).is_some()).count());
    assert_eq!(found, lookups, "every mmap lookup hits");
    println!(
        "  point lookups (mmap, resident): {:.0} ns each",
        t.as_secs_f64() * 1e9 / usize_f64(lookups)
    );

    measure_end_to_end(pack, idx, &keys, lookups);

    let mut sorted = keys.clone();
    sorted.sort_unstable();
    let (found, t) = timed(|| idx.batch(&sorted));
    assert_eq!(found, lookups, "every batch key hits");
    println!(
        "  sorted batch ({lookups} keys, merge-walk over the whole index): {:.1} ms",
        ms(t)
    );

    let store = PackfileStorage::open(dir.to_path_buf()).expect("open");
    println!(
        "  store.get lookups run on an open via {:?} (reject {:?})",
        store.open_timings().map(|t| t.path),
        store.open_timings().and_then(|t| t.checkpoint_reject),
    );
    let (found, t) = timed(|| {
        keys.iter()
            .filter(|k| store.get(&COLLECTION, k).expect("get").is_some())
            .count()
    });
    if found != lookups {
        // Diagnose before failing: which sequence numbers does the store miss?
        let mut missed = Vec::new();
        for i in 0..lookups as u64 {
            let seq = sample_index(i, records);
            if store
                .get(&COLLECTION, &node_id(seq))
                .expect("get")
                .is_none()
            {
                missed.push(seq);
            }
        }
        missed.sort_unstable();
        println!(
            "  STORE MISSES {} of {lookups}; records {records}; min {:?} max {:?}; deciles of misses by seq/records: {:?}",
            missed.len(),
            missed.first(),
            missed.last(),
            (0..10u64)
                .map(|d| {
                    missed
                        .iter()
                        .filter(|&&q| {
                            q.checked_mul(10)
                                .expect("decile multiplication")
                                .checked_div(records)
                                .expect("nonempty benchmark index")
                                == d
                        })
                        .count()
                })
                .collect::<Vec<_>>()
        );
        println!(
            "  store collection index stats: {:?} (reject {:?})",
            store.open_timings().map(|t| t.path),
            store.open_timings().and_then(|t| t.checkpoint_reject),
        );
    }
    assert_eq!(found, lookups, "store lookups hit");
    println!(
        "  end-to-end via store.get (hash locator + frame read): {:.0} ns each",
        t.as_secs_f64() * 1e9 / usize_f64(lookups)
    );
}

fn measure_end_to_end(pack: &Path, idx: &Idx, keys: &[[u8; 16]], lookups: usize) {
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::open(pack).expect("open pack");
    let mut buf = vec![0u8; 72 << 10];
    let (found, t) = timed(|| {
        let mut ok = 0;
        for key in keys {
            let off = idx.get(key).expect("hit");
            let n = file.read_at(&mut buf[..4096], off).expect("pread");
            let rec = if let Ok(rec) = mtxdb::packfile::read_record(&mut &buf[..n]) {
                rec
            } else {
                let n = file.read_at(&mut buf, off).expect("pread");
                mtxdb::packfile::read_record(&mut &buf[..n]).expect("read frame")
            };
            black_box(&rec);
            ok = add_usize(ok, 1);
        }
        ok
    });
    assert_eq!(found, lookups, "end-to-end hits");
    println!(
        "  end-to-end via .idx (locate + pread + read_record; naive read path, not tuned): {:.0} ns each",
        t.as_secs_f64() * 1e9 / usize_f64(lookups)
    );
}

fn report_lossy_index_cases(dir: &Path, entries: &[([u8; 16], [u8; 16], u64)]) {
    let (fresh_index, fresh_build) = timed(|| build_lossy_index(entries));
    println!(
        "  LossyIndex owned build for memory report: {:.0} ms",
        ms(fresh_build)
    );
    report_lossy_index(
        "owned after build",
        fresh_index.len(),
        fresh_index.capacity(),
        fresh_index.resident_bytes(),
        fresh_index.mapped_bytes(),
    );
    let (post_grow_index, post_grow_build) =
        timed(|| build_post_grow_lossy_index(entries, fresh_index.capacity()));
    println!(
        "  LossyIndex post-grow setup: {:.0} ms",
        ms(post_grow_build)
    );
    report_lossy_index(
        "owned immediately after doubling",
        post_grow_index.len(),
        post_grow_index.capacity(),
        post_grow_index.resident_bytes(),
        post_grow_index.mapped_bytes(),
    );
    drop((fresh_index, post_grow_index));

    let checkpoint_store = PackfileStorage::open(PathBuf::from(dir))
        .expect("open store before writing a clean checkpoint");
    checkpoint_store
        .force_index_checkpoint()
        .expect("write clean index checkpoint");
    drop(checkpoint_store);

    let reopened = PackfileStorage::open(PathBuf::from(dir)).expect("reopen checkpoint for memory");
    let (reopened_records, heap_bytes, mapped_bytes, capacity) = reopened
        .collection_index_residency(&COLLECTION)
        .expect("reopened collection index");
    println!(
        "  checkpoint reopen path: {:?}",
        reopened.open_timings().map(|timings| timings.path)
    );
    report_lossy_index(
        "after store reopen",
        reopened_records,
        capacity,
        heap_bytes,
        mapped_bytes,
    );

    // The first mutation materializes mmap-backed slots and initializes the
    // identity tables. Updating an existing id keeps the pack's record count
    // unchanged while measuring that representation transition.
    let first_id = entries.first().expect("benchmark pack has records").1;
    let rss_before_write = process_rss_split();
    let first_write = [(first_id, NodeData::from_slice(&[0xA5; 16]))];
    let ((), write_time) = timed(|| {
        reopened
            .put_many(&COLLECTION, &first_write)
            .expect("first write after checkpoint reopen");
    });
    let (written_records, heap_bytes, mapped_bytes, capacity) = reopened
        .collection_index_residency(&COLLECTION)
        .expect("written collection index");
    let rss_after_write = process_rss_split();
    println!("  first write after reopen: {:.1} ms", ms(write_time));
    // Allocated heap above counts zeroed identity tables in full. Anonymous
    // RSS shows how much of it, plus the materialized slot clone, became
    // resident; file RSS shows the released checkpoint mapping.
    match (rss_before_write, rss_after_write) {
        (Some((anon_before, file_before)), Some((anon_after, file_after))) => {
            let delta = |after: u64, before: u64| (to_f64(after) - to_f64(before)) / 1_048_576.0;
            println!(
                "    RSS across the first write: anonymous {:+.2} MiB, file-backed {:+.2} MiB \
                 (allocated private heap {:.2} MiB)",
                delta(anon_after, anon_before),
                delta(file_after, file_before),
                mib_f64(u64::try_from(heap_bytes).expect("heap fits u64")),
            );
        }
        _ => println!("    process RSS unavailable"),
    }
    report_lossy_index(
        "after reopen + one overwrite",
        written_records,
        capacity,
        heap_bytes,
        mapped_bytes,
    );
}

fn run(base: &Path, mib: u64, mixed: bool, lookups: usize) {
    let label = if mixed { "mixed 1-64 KiB" } else { "tiny 16 B" };
    let dir = base.join(format!("mtxdb_bench_idxart_{}_{mixed}", std::process::id()));
    let (records, build_t) = timed(|| build(&dir, mib << 20, mixed));
    if std::env::var_os("MTXDB_IA_LS").is_some() {
        for e in std::fs::read_dir(&dir).expect("ls").flatten() {
            println!(
                "  {} {}",
                e.file_name().to_string_lossy(),
                e.metadata().map_or(0, |m| m.len())
            );
        }
    }
    let pack_count = std::fs::read_dir(&dir)
        .expect("ls")
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "pack"))
        .count();
    assert_eq!(
        pack_count, 1,
        "this bench measures exactly one pack; got {pack_count}"
    );
    let pack = pack_path(&dir);
    let pack_len = std::fs::metadata(&pack).expect("stat").len();
    println!(
        "\n== {label}: {records} records, pack {} MiB ({:.0} B/frame), built in {:.1} s ==",
        pack_len >> 20,
        to_f64(pack_len) / to_f64(records),
        build_t.as_secs_f64(),
    );
    println!("  index files: {}", index_files(&dir));

    let entries = scan_and_recover_packfile(&pack).expect("scan");
    assert_eq!(entries.len() as u64, records, "scan saw every record");
    let (
        pack_index_path,
        pack_index_identity,
        pack_index_bytes,
        pack_index_build,
        pack_index_write,
    ) = build_pack_index(&dir, &pack, pack_len, &entries);
    report_pack_index(
        &pack_index_path,
        pack_index_identity,
        &entries,
        pack_index_bytes,
        pack_index_build,
        pack_index_write,
    );
    let idx_path = dir.join("pack.idx");
    let (idx_bytes, idx_build) = timed(|| {
        let bytes = build_idx(&entries);
        std::fs::write(&idx_path, &bytes).expect("write idx");
        std::fs::File::open(&idx_path)
            .and_then(|f| f.sync_all())
            .expect("fsync idx");
        bytes
    });
    let (_, sort_only) = timed(|| {
        let mut v: Vec<_> = entries.iter().map(|e| (e.1, e.2)).collect();
        v.sort_unstable();
        black_box(v.len())
    });
    println!(
        "  .idx size {:.2} MiB ({:.1} B/entry) vs checkpoint {:.2} MiB; build+write+fsync {:.0} ms \
         (sort alone {:.0} ms)",
        mib_f64(u64::try_from(idx_bytes.len()).expect("index length fits in u64")),
        to_f64(u64::try_from(idx_bytes.len()).expect("index length fits in u64"))
            / to_f64(records),
        mib_f64(checkpoint_bytes(&dir)),
        ms(idx_build),
        ms(sort_only),
    );
    drop(idx_bytes);
    let manifest_len = pack_len;
    let idx = load_idx(&idx_path);

    for cold in [false, true] {
        measure_reopen(&dir, &pack, &idx_path, manifest_len, cold);
        if cold {
            measure_cold_lookups(&dir, &idx_path, records);
        } else {
            measure_warm_lookups(&dir, &pack, &idx_path, &idx, records, lookups);
        }
    }
    report_lossy_index_cases(&dir, &entries);
    let _ = std::fs::remove_dir_all(&dir);
}

fn main() {
    let mib = env_u64("MTXDB_IA_MIB", 256);
    let lookups = usize::try_from(env_u64("MTXDB_IA_LOOKUPS", 100_000)).expect("lookups");
    let frames = std::env::var("MTXDB_IA_FRAMES").unwrap_or_else(|_| "tiny,mixed".to_owned());
    let base = std::env::var_os("MTXDB_BENCH_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&base).expect("create bench root");
    for kind in frames.split(',') {
        run(&base, mib, kind.trim() == "mixed", lookups);
    }
}
