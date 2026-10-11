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

use mtxdb::packfile::scan_and_recover_packfile;
use mtxdb::storage::{NodeData, NodeId, StorageEngine};
use mtxdb::PackfileStorage;

const COLLECTION: [u8; 16] = [0xE7; 16];
const MAGIC: &[u8; 8] = b"MTXIDX01";

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
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
                e.metadata().map_or(0, |m| m.len()) as f64 / f64::from(1 << 20)
            )
        })
        .collect();
    parts.sort();
    parts.join(", ")
}

fn checkpoint_bytes(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("index.checkpoint")).map_or(0, |m| m.len())
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
            usize::try_from(1 + mix(seq ^ 77) % (64 << 10)).expect("len")
        } else {
            16
        };
        let data = if mixed {
            let start = usize::try_from(mix(seq) % 1024).expect("start") * 8;
            NodeData::from_slice(&big[start..start + len.min(big.len() - start)])
        } else {
            tiny.clone()
        };
        store
            .put(&COLLECTION, &node_id(seq), &data)
            .expect("put record");
        seq += 1;
        written += len as u64 + 61 - 16;
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
        Self::FAN + 256 * 4
    }
    fn offs_at(&self) -> usize {
        Self::ids_at() + self.count * 16
    }
    fn fan(&self, i: usize) -> usize {
        let at = Self::FAN + i * 4;
        u32::from_le_bytes(self.bytes[at..at + 4].try_into().expect("fan")) as usize
    }
    fn id(&self, i: usize) -> &[u8] {
        let at = Self::ids_at() + i * 16;
        &self.bytes[at..at + 16]
    }
    fn off(&self, i: usize) -> u64 {
        let at = self.offs_at() + i * 4;
        u64::from(u32::from_le_bytes(
            self.bytes[at..at + 4].try_into().expect("off"),
        ))
    }
    fn get(&self, key: &[u8; 16]) -> Option<u64> {
        let b = usize::from(key[0]);
        let mut lo = if b == 0 { 0 } else { self.fan(b - 1) };
        let mut hi = self.fan(b);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.id(mid).cmp(&key[..]) {
                std::cmp::Ordering::Less => lo = mid + 1,
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
                pos += 1;
            }
            if pos < self.count && self.id(pos) == &key[..] {
                found += 1;
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
    let mut out = Vec::with_capacity(16 + 1024 + sorted.len() * 20 + 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
    let mut fan = [0u32; 256];
    for (id, _) in &sorted {
        fan[usize::from(id[0])] += 1;
    }
    let mut run = 0u32;
    for f in &mut fan {
        run += *f;
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
        16 + 1024 + count * 20 + 4,
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
    let tail = bytes.len() - 4;
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
        pack_len as f64 / records as f64,
        build_t.as_secs_f64(),
    );
    println!("  index files: {}", index_files(&dir));

    // Scan once to learn the entries (also the rebuild cost, measured per cache state below).
    let entries = scan_and_recover_packfile(&pack).expect("scan");
    assert_eq!(entries.len() as u64, records, "scan saw every record");
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
        idx_bytes.len() as f64 / f64::from(1 << 20),
        idx_bytes.len() as f64 / records as f64,
        checkpoint_bytes(&dir) as f64 / f64::from(1 << 20),
        ms(idx_build),
        ms(sort_only),
    );
    drop(idx_bytes);
    let manifest_len = pack_len;
    drop(entries);

    for cold in [false, true] {
        let state = if cold { "cold" } else { "warm" };
        let prep = |d: &Path| {
            if cold && !sync_and_evict(d) {
                println!("  (eviction failed; cold numbers are really warm)");
            }
        };
        prep(&dir);
        let (store, t) = timed(|| PackfileStorage::open(dir.clone()).expect("open"));
        let t_open = t;
        let first = timed(|| store.get(&COLLECTION, &node_id(0)).expect("get").is_some()).1;
        let timings = store.open_timings().expect("timings");
        drop(store);
        println!(
            "  [{state}] current open {:.0} ms (pack recovery {:.0}, checkpoint decode {:.0}, \
             index materialization {:.0}), first read {:.2} ms",
            ms(t_open),
            ms(timings.packfile_recovery),
            ms(timings.checkpoint_decode),
            ms(timings.index_materialization),
            ms(first),
        );

        prep(&dir);
        let (n, t) = timed(|| scan_and_recover_packfile(&pack).expect("scan").len());
        println!("  [{state}] scan rebuild {:.0} ms ({n} entries)", ms(t));

        prep(&dir);
        let (idx, t) = timed(|| load_idx(&idx_path));
        println!("  [{state}] .idx reopen (read + CRC) {:.1} ms", ms(t));

        prep(&dir);
        let (_, t) = timed(|| {
            assert_eq!(
                std::fs::metadata(&pack).expect("stat").len(),
                manifest_len,
                "manifest length"
            );
            black_box(header_only(&idx_path))
        });
        println!(
            "  [{state}] trusted, metadata-only (stat pack + idx header/fanout) {:.2} ms",
            ms(t)
        );

        prep(&dir);
        let (mapped, t) = timed(|| {
            assert_eq!(
                std::fs::metadata(&pack).expect("stat").len(),
                manifest_len,
                "manifest length"
            );
            map_idx(&idx_path)
        });
        println!(
            "  [{state}] trusted, mmap .idx (header check, no CRC) {:.2} ms",
            ms(t)
        );

        prep(&dir);
        let (_, t) = timed(|| {
            assert_eq!(
                std::fs::metadata(&pack).expect("stat").len(),
                manifest_len,
                "manifest length"
            );
            black_box(load_idx(&idx_path).count)
        });
        println!("  [{state}] trusted, read + CRC .idx {:.1} ms", ms(t));

        prep(&dir);
        let (_, t) = timed(|| black_box(crc32fast::hash(&std::fs::read(&pack).expect("read"))));
        println!(
            "  [{state}] digest-verify reopen (read whole pack + CRC) {:.0} ms",
            ms(t)
        );

        if cold {
            // Page-fault cost of the first lookups on a freshly mapped, evicted index.
            let probe: Vec<[u8; 16]> = (0..1000u64)
                .map(|i| node_id(mix(i ^ 5) % records))
                .collect();
            prep(&dir);
            let fresh = map_idx(&idx_path);
            let (found, t) = timed(|| probe.iter().filter(|k| fresh.get(k).is_some()).count());
            assert_eq!(found, probe.len(), "cold mmap lookups hit");
            println!(
                "  [cold] mmap: first 1000 random lookups {:.1} ms total",
                ms(t)
            );
        }
        drop(mapped);
        if !cold {
            let keys: Vec<[u8; 16]> = (0..lookups as u64)
                .map(|i| node_id(mix(i) % records))
                .collect();
            let (found, t) = timed(|| keys.iter().filter(|k| idx.get(k).is_some()).count());
            assert_eq!(found, lookups, "every point lookup hits");
            println!(
                "  point lookups: {:.0} ns each over {lookups} random keys",
                t.as_secs_f64() * 1e9 / lookups as f64
            );
            let mapped = map_idx(&idx_path);
            let (found, t) = timed(|| keys.iter().filter(|k| mapped.get(k).is_some()).count());
            assert_eq!(found, lookups, "every mmap lookup hits");
            println!(
                "  point lookups (mmap, resident): {:.0} ns each",
                t.as_secs_f64() * 1e9 / lookups as f64
            );
            {
                use std::os::unix::fs::FileExt;
                let file = std::fs::File::open(&pack).expect("open pack");
                // pread into a reused buffer, then decode and validate the frame.
                let mut buf = vec![0u8; 72 << 10];
                let (found, t) = timed(|| {
                    let mut ok = 0;
                    for k in &keys {
                        let off = idx.get(k).expect("hit");
                        // 4 KiB first; a longer frame fails with EOF and is re-read whole.
                        let n = file.read_at(&mut buf[..4096], off).expect("pread");
                        let rec = if let Ok(rec) = mtxdb::packfile::read_record(&mut &buf[..n]) {
                            rec
                        } else {
                            let n = file.read_at(&mut buf, off).expect("pread");
                            mtxdb::packfile::read_record(&mut &buf[..n]).expect("read frame")
                        };
                        black_box(&rec);
                        ok += 1;
                    }
                    ok
                });
                assert_eq!(found, lookups, "end-to-end hits");
                println!(
                    "  end-to-end via .idx (locate + pread + read_record; naive read path, not tuned): {:.0} ns each",
                    t.as_secs_f64() * 1e9 / lookups as f64
                );
            }
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            let (found, t) = timed(|| idx.batch(&sorted));
            assert_eq!(found, lookups, "every batch key hits");
            println!(
                "  sorted batch ({lookups} keys, merge-walk over the whole index): {:.1} ms",
                ms(t)
            );
            let store = PackfileStorage::open(dir.clone()).expect("open");
            let (found, t) = timed(|| {
                keys.iter()
                    .filter(|k| store.get(&COLLECTION, k).expect("get").is_some())
                    .count()
            });
            assert_eq!(found, lookups, "store lookups hit");
            println!(
                "  end-to-end via store.get (hash locator + frame read): {:.0} ns each",
                t.as_secs_f64() * 1e9 / lookups as f64
            );
            drop(store);
        }
    }
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
