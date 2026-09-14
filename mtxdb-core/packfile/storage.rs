use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::cache::{NodeCache, PinnedNodes};
use crate::csr::Csr;
use crate::index::LossyIndex;
use crate::packfile::{self, Record};
use crate::shard;
use crate::shard::{Shard, ShardPool};
use crate::storage::{NodeData, NodeId, NodeRef, StorageEngine, StorageError};

/// Callback that rewrites a node's child references given resolved child data,
/// used to inline already-cached children in place of lazy hash pointers.
pub type SwizzleFn = fn(&NodeData, &[NodeId], &[Option<Arc<NodeData>>]) -> NodeData;

/// Result of [`PackfileStorage::plan_collection_repack`] — what a real repack
/// of this collection would do, computed exactly (same scan + reachability
/// pass) but without performing any writes.
#[derive(Debug, Clone, Default)]
pub struct RepackPlan {
    /// Records that would survive (be kept) by this repack.
    pub kept: usize,
    /// Records that would be dropped (found unreachable) by this repack.
    pub dropped: usize,
    /// Total on-disk frame bytes of the records that would be kept.
    pub kept_bytes: u64,
    /// Total on-disk frame bytes of records that would be pruned.
    pub dropped_bytes: u64,
    /// Every shard id at least one kept record currently lives in.
    pub shards_touched: Vec<u16>,
}

/// Bounds on a [`PackfileStorage::walk_ancestors`] call.
///
/// Without a cap, a walk whose `stop_at` set is never reached (e.g. the
/// caller passed a stale or wrong boundary) silently degrades into "walk
/// the entire history of this branch" — `max_nodes` is the safety valve
/// for that case.
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkLimits {
    /// Stop discovering new ancestors once this many nodes have been
    /// visited. `None` means unbounded (walk until `stop_at` or the
    /// collection's true roots are reached).
    pub max_nodes: Option<usize>,
}

/// Lazy, ancestor-first iterator over the result of
/// [`PackfileStorage::walk_ancestors`].
///
/// Every node was already resolved (hash-verified, same as [`PackfileStorage::get`])
/// against one frozen generation snapshot while [`PackfileStorage::walk_ancestors`]
/// built the walk — this iterator just replays that in ancestor-first
/// order. No further store access happens here, so a repack that runs
/// after the walk was built can't cause a node to go missing partway
/// through iteration.
pub struct DagWalk {
    order: std::vec::IntoIter<[u8; 16]>,
    resolved: HashMap<[u8; 16], NodeData>,
}

impl Iterator for DagWalk {
    type Item = Result<(NodeId, NodeData), StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        let hash = self.order.next()?;
        let data = self
            .resolved
            .remove(&hash)
            .expect("order only contains hashes this walk itself resolved");
        Some(Ok((hash, data)))
    }
}

/// A live hash set plus the adjacency discovered while determining it, as
/// returned by the reachability repacker's live-set computation.
type AdjacencyResult = (Vec<[u8; 16]>, HashMap<[u8; 16], Vec<[u8; 16]>>);

/// One shard's scanned `(hash, offset)` entries for a single collection, as
/// produced by a repack's initial shard scan.
type ScannedShard = (u16, Vec<([u8; 16], u64)>);

/// Deduplicated live-location map for one collection during a repack scan.
type RepackRecordMap = HashMap<[u8; 16], (u16, u64)>;
/// Replacement index entries produced while copying one repack collection.
type RepackOffsets = Vec<([u8; 16], u16, u64)>;
/// Return type of [`PackfileStorage::repack_scan_incremental`]: the merged
/// live-location map and (on the incremental path) the previous cursor state
/// that the adjacency helpers need to skip re-reading already-seen nodes.
type RepackScanResult = (RepackRecordMap, Option<RepackIncrementalState>);

/// One scanned record's `(shard_id, hash, offset)`, as accumulated per
/// collection during `PackfileStorage::open_with_options`'s initial scan.
type ShardRecord = (u16, [u8; 16], u64, u64);

/// Accumulator for `PackfileStorage::open_with_options`'s phase 2 —
/// bundles the three maps `init_collection_from_scan` fills in per collection, so
/// that function takes one out-parameter instead of three.
#[derive(Default)]
struct RoomScanOutput {
    collections: HashMap<[u8; 16], ArcSwap<RoomGeneration>>,
    shard_collections: HashMap<u64, HashMap<[u8; 16], u64>>,
    collection_shards: HashMap<[u8; 16], HashSet<u64>>,
}

/// Magic bytes + version identifying the persisted shard→collection directory
/// format (see `PackfileStorage::persist_shard_collections`).
///
/// Pre-release, no compatibility fallback: an unrecognized version is
/// treated exactly like a missing/corrupt file (see
/// `read_persisted_shard_collections`) — reset or let the next sync
/// regenerate it, not a format this reader tries to still understand.
/// This sidecar has changed shape twice already (adding insertion order,
/// then switching `shard_id` for `pack_id`) and briefly kept read
/// support for every prior version each time; none of that carries
/// forward, since nothing depends on reading a store from before the
/// current format existed.
const SHARD_ROOMS_MAGIC: &[u8; 4] = b"MSRM";
const SHARD_ROOMS_VERSION: u8 = 3;
/// Header size: magic(4) + version(1) + `persisted_at`(8).
const SHARD_ROOMS_HEADER_LEN: usize = 4 + 1 + 8;
/// One entry: `pack_id`(8) + `collection_id`(16) + count(8) + the
/// collection's stable insertion ordinal(8).
const SHARD_ROOMS_RECORD_LEN: usize = 8 + 16 + 8 + 8;

/// The largest record offset `IndexSlot` can represent: its 28-bit offset
/// field stores `offset + 1`, reserving the all-zeros encoding for the empty
/// sentinel. Offsets beyond this can only come from legacy or externally
/// created oversized packs — this engine's own writes rotate long before
/// reaching it (`MAX_SHARD_BYTES` caps each shard's file size).
const PACK_INDEX_OFFSET_LIMIT: u64 = (1u64 << 28) - 2;

/// Reject a record whose in-shard offset the 28-bit `IndexSlot` field cannot
/// represent, so the caller surfaces a `StorageError::Corrupt` instead of
/// silently dropping the record (or panicking in `IndexSlot::new`).
fn check_index_offset(shard_id: u16, hash: &[u8; 16], offset: u64) -> Result<(), StorageError> {
    if offset > PACK_INDEX_OFFSET_LIMIT {
        return Err(StorageError::Corrupt(format!(
            "shard {shard_id} holds record {hash:?} at offset {offset}, beyond the index offset limit {PACK_INDEX_OFFSET_LIMIT}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PersistedShardRoom {
    pack_id: u64,
    collection_id: [u8; 16],
    count: u64,
    insertion_order: u64,
}

/// Decode the small inspection sidecar. This is deliberately shared by all
/// read-only CLI summary helpers so they agree on validation and format
/// compatibility.
fn read_persisted_shard_collections(
    base_dir: &std::path::Path,
) -> Option<(u64, Vec<PersistedShardRoom>)> {
    let buf = fs::read(base_dir.join("shard_collections.bin")).ok()?;
    if buf.len() < SHARD_ROOMS_HEADER_LEN
        || &buf[0..4] != SHARD_ROOMS_MAGIC
        || buf[4] != SHARD_ROOMS_VERSION
    {
        return None;
    }
    let body = &buf[SHARD_ROOMS_HEADER_LEN..];
    if body.len().checked_rem(SHARD_ROOMS_RECORD_LEN) != Some(0) {
        return None;
    }
    let persisted_at = u64::from_le_bytes(buf[5..13].try_into().ok()?);
    let records = body
        .chunks_exact(SHARD_ROOMS_RECORD_LEN)
        .map(|chunk| {
            let pack_id = u64::from_le_bytes(chunk[0..8].try_into().ok()?);
            let mut collection_id = [0u8; 16];
            collection_id.copy_from_slice(&chunk[8..24]);
            let count = u64::from_le_bytes(chunk[24..32].try_into().ok()?);
            let insertion_order = u64::from_le_bytes(chunk[32..40].try_into().ok()?);
            Some(PersistedShardRoom {
                pack_id,
                collection_id,
                count,
                insertion_order,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some((persisted_at, records))
}

/// The durable insertion order supplied by the inspection directory.
fn persisted_collection_order(base_dir: &std::path::Path) -> Option<Vec<[u8; 16]>> {
    let (_, records) = read_persisted_shard_collections(base_dir)?;
    let mut orders: HashMap<[u8; 16], u64> = HashMap::new();
    for record in records {
        let order = record.insertion_order;
        orders
            .entry(record.collection_id)
            .and_modify(|current| *current = (*current).min(order))
            .or_insert(order);
    }
    let mut collections: Vec<([u8; 16], u64)> = orders.into_iter().collect();
    collections.sort_unstable_by_key(|(collection_id, order)| (*order, *collection_id));
    Some(
        collections
            .into_iter()
            .map(|(collection_id, _)| collection_id)
            .collect(),
    )
}

/// Disambiguates concurrent `persist_shard_collections` tmp filenames within
/// this process, paired with the process id for uniqueness across
/// processes — same rationale as `shard::STATS_TMP_COUNTER`.
static SHARD_ROOMS_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Immutable snapshot of a collection's in-memory state.
///
/// Index and cache are bundled so readers see a consistent triple
/// via a single `ArcSwap::load()` — no three-lock coordination.
#[derive(Clone)]
struct RoomGeneration {
    index: LossyIndex,
    cache: Arc<NodeCache>,
}

/// A content-addressed packfile storage engine backed by a global shard pool.
///
/// All collections share a small pool of shard files (~4, each ~2GB), keeping
/// file descriptor usage constant regardless of collection count.
///
/// Per-collection state (index + cache) is bundled in an immutable
/// `RoomGeneration` and swapped atomically via `ArcSwap`:
///
/// - **Reads**: `load_full()` once, see a consistent snapshot.
/// - **Writes**: build new generation under put lock, swap atomically.
/// - **Delete**: drop the generation — cache disappears with it.
pub struct PackfileStorage {
    shards: ShardPool,
    collections: RwLock<HashMap<[u8; 16], ArcSwap<RoomGeneration>>>,
    collection_order: RwLock<Vec<[u8; 16]>>,
    pinned: PinnedNodes,
    base_dir: PathBuf,
    swizzle: Option<SwizzleFn>,
    put_locks: parking_lot::Mutex<HashMap<[u8; 16], Arc<parking_lot::Mutex<()>>>>,
    /// In-memory cache of the `deleted.collections` file, seeded once at
    /// `open()` from disk. Guards both the set and the read-modify-write of
    /// the backing file, so a plain membership check (the common case: most
    /// collections were never deleted) never touches disk. Collection locks
    /// protect a collection's lifecycle, but distinct collections may be
    /// recreated or deleted concurrently, hence the separate lock here.
    deleted_collections: parking_lot::Mutex<HashSet<[u8; 16]>>,
    live_roots: RwLock<HashMap<[u8; 16], Vec<NodeId>>>,
    repack_threshold_entries: AtomicU64,
    cache_capacity: usize,
    /// Total number of `repack_collection_reachable` calls across all collections.
    repack_count: AtomicU64,
    /// Total records kept (rewritten into the new generation) across all repacks.
    repack_kept_total: AtomicU64,
    /// Total records dropped (found unreachable) across all repacks.
    repack_dropped_total: AtomicU64,
    /// Per-collection repack counts, so a hot collection's churn is visible individually.
    repack_counts_by_collection: RwLock<HashMap<[u8; 16], u64>>,
    /// Per-collection incremental repack state. Each entry tracks the byte
    /// offset up to which each shard has been scanned for this collection and
    /// the deduplicated hash → (`shard_id`, offset) map from the last repack.
    /// On the next repack, only bytes after these offsets are scanned and
    /// merged into the existing map — turning O(n²) full rescan into
    /// O(n) total work across all repack calls.
    repack_incremental: RwLock<HashMap<[u8; 16], RepackIncrementalState>>,
    /// Persisted directory: which collections have live records in each shard,
    /// and how many. Maintained incrementally (a plain `put` just
    /// increments one counter; a full index rebuild/repack/initial scan
    /// replaces one collection's contribution wholesale via
    /// `LossyIndex::shard_counts`) rather than ever re-derived by
    /// scanning a shard file, which is what made `collections_referencing_shard`
    /// and a `collections`-style listing expensive before this existed.
    shard_collections: RwLock<HashMap<u64, HashMap<[u8; 16], u64>>>,
    /// Reverse index of `shard_collections`: which shards a given collection currently
    /// contributes a nonzero count to. Lets a collection's full-index-rebuild
    /// path (`replace_collection_shard_counts`) clear exactly the shard entries
    /// it used to occupy without scanning every shard in `shard_collections`.
    collection_shards: RwLock<HashMap<[u8; 16], HashSet<u64>>>,
    /// Wall-clock instant of the last `maybe_persist_shard_collections` flush,
    /// used to rate-limit that timer-driven path.
    last_shard_collections_flush: RwLock<Option<std::time::Instant>>,
    /// Whether any collection data changed since the last `index.checkpoint`
    /// write. Set by every generation swap (`put`/`put_many`/`repack`/`refresh`);
    /// cleared only by a successful [`Self::persist_index_checkpoint`], so a
    /// failed write is retried on the next sync. Lets `sync()`/`sync_all()`
    /// skip rewriting the checkpoint when nothing has changed since the last
    /// one — a steady-state writer that syncs between writes pays no checkpoint
    /// cost, while a crash left a stale checkpoint is still always resolved by
    /// the fingerprint → rescan fallback.
    index_checkpoint_dirty: AtomicBool,
}

/// Per-collection state for incremental repack.
#[derive(Clone)]
struct RepackIncrementalState {
    /// Per-shard byte offset: next scan starts here (file length at end
    /// of last scan). Shards not in this map haven't been scanned yet.
    // A slot can be retired and reused for a different pack.  Keep the
    // pack_id beside the cursor so an offset from the old incarnation is
    // never applied to the replacement file.
    scan_offsets: HashMap<u16, (u64, u64)>,
    /// The deduplicated hash → (`shard_id`, offset) map from the last repack.
    /// The next repack merges newly-scanned entries into this map.
    live_map: HashMap<[u8; 16], (u16, u64)>,
    /// Cached edge lists for every node that survived the last repack.
    /// Lets the BFS and full-scan adjacency walks skip re-reading disk for
    /// nodes that were already seen — turning O(total) disk reads per repack
    /// call into O(delta) reads for only nodes added since the last repack.
    adjacency: HashMap<[u8; 16], Vec<[u8; 16]>>,
    /// The live-roots snapshot from the last repack. Used to identify which
    /// roots are genuinely new so the incremental BFS can seed only those,
    /// rather than re-walking the entire reachable set from scratch.
    prev_roots: Vec<[u8; 16]>,
}

const DEFAULT_REPACK_THRESHOLD_ENTRIES: u64 = 2048;
const DEFAULT_CACHE_CAPACITY: usize = 100_000;

impl PackfileStorage {
    /// Open a packfile storage with default settings.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage with explicit compression and checksum
    /// policies — the advanced durability/performance entry point. See
    /// [`crate::packfile::ChecksumPolicy`] for what giving up read-time
    /// verification costs.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_policies(
        base_dir: PathBuf,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            compress,
            checksum_policy,
        )
    }

    /// Open a packfile storage with `compress` controlling whether records
    /// are zstd-attempted on write (see
    /// [`crate::packfile::write_record_with_options`]) — pass `false` for a
    /// pool whose payloads (e.g. HAMT nodes/roots) never benefit, to skip
    /// paying the compressor's cost on every put.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_compression(
        base_dir: PathBuf,
        compress: bool,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            None,
            compress,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage that rotates shards at `max_shard_bytes`
    /// instead of the default `~256 MB` ceiling. Intended for benchmarks
    /// and tests that want many small packs without writing gigabytes of
    /// data to trigger rotation — see
    /// [`crate::shard::ShardPool::open_with_max_shard_bytes`].
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or
    /// read, or if `max_shard_bytes` is zero or exceeds
    /// [`crate::shard::MAX_SHARD_BYTES`].
    pub fn open_with_max_shard_bytes(
        base_dir: PathBuf,
        max_shard_bytes: u64,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            true,
            Some(max_shard_bytes),
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage as a read-only observer, coexisting with a
    /// concurrent writer on the same directory (see
    /// `ShardPool::open_read_only`'s doc for the locking contract).
    ///
    /// Intended for inspection tooling (e.g. a `shards`/`collections`/`info` CLI
    /// command) that needs to run alongside a live writer process without
    /// either racing its on-disk state or being mistaken for a second
    /// writer and rejected.
    ///
    /// # Errors
    /// Returns `io::Error` if the directory can't be read, has no shards
    /// yet, or a writer already holds the exclusive lock.
    pub fn open_read_only(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            false,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage as a read-only observer with an explicit
    /// checksum policy, gating per-lookup CRC verification in `read_at`.
    /// See [`crate::packfile::ChecksumPolicy`]. Matches a writer opened with
    /// the same policy.
    ///
    /// # Errors
    /// Same as [`Self::open_read_only`].
    pub fn open_read_only_with_policies(
        base_dir: PathBuf,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            DEFAULT_CACHE_CAPACITY,
            None,
            false,
            None,
            true,
            checksum_policy,
        )
    }

    /// Open a packfile storage with a custom per-collection cache capacity.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_cache(
        base_dir: PathBuf,
        cache_capacity: usize,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            None,
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    /// Open a packfile storage with a swizzle callback for in-cache pointer resolution.
    ///
    /// # Errors
    /// Returns `io::Error` if the base directory cannot be created or read.
    pub fn open_with_swizzle(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: SwizzleFn,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            Some(swizzle),
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn open_with_options(
        base_dir: PathBuf,
        cache_capacity: usize,
        swizzle: Option<SwizzleFn>,
        writable: bool,
        max_shard_bytes: Option<u64>,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        fs::create_dir_all(&base_dir)?;

        let shards = if writable {
            match max_shard_bytes {
                // Custom rotation thresholds are only used by benchmarks/
                // tests today, none of which also need to disable
                // compression, so this combination just keeps the default.
                Some(max_shard_bytes) => ShardPool::open_with_policies(
                    base_dir.clone(),
                    compress,
                    checksum_policy,
                    Some(max_shard_bytes),
                )?,
                None => ShardPool::open_with_policies(
                    base_dir.clone(),
                    compress,
                    checksum_policy,
                    None,
                )?,
            }
        } else {
            ShardPool::open_read_only_with_policies(base_dir.clone(), checksum_policy)?
        };

        // Phase 1: accumulate all records per collection across every shard so
        // the index can be sized once for the true total.
        let mut collection_entries: HashMap<[u8; 16], Vec<ShardRecord>> = HashMap::new();
        // Seed logical insertion order before scanning. A newly written collection
        // cannot appear in the sidecar until the next flush, so append it at
        // first physical sight below. Legacy v1 sidecars yield no order and
        // therefore use the same first-seen fallback for every collection.
        let mut collection_order = persisted_collection_order(&base_dir).unwrap_or_default();
        let mut known_collections: HashSet<[u8; 16]> = collection_order.iter().copied().collect();

        // Collect open shard info (slot, pack_id, path, length) before the scan
        // loop so we don't hold the shards read-lock across the I/O-heavy scan.
        // The lengths feed the checkpoint fingerprint (see `index::checkpoint`).
        let open_shards: Vec<(u16, u64, PathBuf, u64)> = shards
            .all_shards()
            .into_iter()
            .map(|(id, shard)| (id, shard.pack_id, shard.path.clone(), shard.file_len()))
            .collect();

        // A checkpoint from the previous session's last sync lets us skip the
        // rescan and index rebuild entirely. It is only trusted when every
        // pack on disk still matches the fingerprint it was written against.
        let deleted_collections = Self::load_deleted_collections(&base_dir);
        if let Some((scan_out, collection_order)) = Self::checkpoint_scan_out(
            &base_dir,
            cache_capacity,
            &shards,
            &open_shards,
            &deleted_collections,
        ) {
            return Ok(Self::assemble(
                shards,
                scan_out,
                collection_order,
                deleted_collections,
                base_dir,
                swizzle,
                cache_capacity,
            ));
        }

        for (shard_id, shard_pack_id, path, _file_len) in open_shards {
            // A read-only open must never touch the file at all — the
            // truncating recovery scan below is only safe when we're the
            // sole writer (guaranteed by the writer lock); a read-only
            // opener can run concurrently with an active writer (no lock
            // taken at all — see ShardPool::open_read_only), so it must
            // use the non-mutating scan_packfile, which just stops
            // cleanly at a torn tail (indistinguishable from a writer's
            // in-flight append) instead of truncating it away.
            let entries = if writable {
                // Recovery can safely truncate only a torn final frame. Any
                // other failure (CRC mismatch, invalid framing, permissions)
                // means this pack cannot be indexed faithfully; fail open
                // rather than publish a store that silently omitted it.
                packfile::scan_and_recover_packfile(&path).map_err(|error| {
                    std::io::Error::new(
                        error.kind(),
                        format!("recovery scan failed for pack {}: {error}", path.display()),
                    )
                })?
            } else {
                match packfile::scan_packfile(&path) {
                    Ok(e) => e,
                    Err(scan_err) => {
                        eprintln!(
                            "warning: read-only scan of shard {shard_id:02x} skipped: {scan_err}"
                        );
                        continue;
                    }
                }
            };
            for (collection_id, hash, offset) in entries {
                if known_collections.insert(collection_id) {
                    collection_order.push(collection_id);
                }
                collection_entries.entry(collection_id).or_default().push((
                    shard_id,
                    hash,
                    offset,
                    shard_pack_id,
                ));
            }
        }
        collection_order.retain(|collection_id| collection_entries.contains_key(collection_id));

        let mut scan_out = RoomScanOutput::default();

        // Phase 2: build per-collection indexes sized to the true total.
        for collection_id in &collection_order {
            if deleted_collections.contains(collection_id) {
                continue;
            }
            Self::init_collection_from_scan(
                collection_id,
                &collection_entries[collection_id],
                &shards,
                cache_capacity,
                &mut scan_out,
            )?;
        }

        Ok(Self::assemble(
            shards,
            scan_out,
            collection_order,
            deleted_collections,
            base_dir,
            swizzle,
            cache_capacity,
        ))
    }

    /// Builds one collection's index, cache, and shard-directory contribution
    /// from its scanned records, and inserts all three into `out` — the
    /// per-collection body of `open_with_options`'s phase 2, factored out to
    /// keep that function under the line-count lint rather than
    /// suppressing it.
    fn init_collection_from_scan(
        collection_id: &[u8; 16],
        records: &[ShardRecord],
        shards: &ShardPool,
        cache_capacity: usize,
        out: &mut RoomScanOutput,
    ) -> Result<(), StorageError> {
        // Seed each collection's home shard from the scan: the shard of its
        // last-scanned record is a best-effort proxy for "most recent"
        // (shards are scanned in ascending ID order, and IDs generally
        // increase over time via rotation) — not exact chronology across
        // shards, but enough to keep a resumed collection's writes landing near
        // its existing data instead of restarting at whatever the pool's
        // active shard happens to be.
        if let Some(&(last_shard_id, _, _, _)) = records.last() {
            shards.set_collection_home(collection_id, last_shard_id);
        }
        let mut index = LossyIndex::new(records.len().saturating_mul(2).max(16));
        for (shard_id, hash, offset, _pack_id) in records {
            check_index_offset(*shard_id, hash, *offset)?;
            let _ = index.insert(hash, *shard_id, *offset);
        }
        let counts = index.shard_counts();

        // Build slot→pack_id lookup from the records for this collection.
        let slot_to_pack_id: HashMap<u16, u64> = records
            .iter()
            .map(|(slot, _, _, pack_id)| (*slot, *pack_id))
            .collect();

        for (&shard_id, &count) in &counts {
            let pack_id = slot_to_pack_id
                .get(&shard_id)
                .copied()
                .unwrap_or(u64::from(shard_id));
            out.shard_collections
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, count);
            out.collection_shards
                .entry(*collection_id)
                .or_default()
                .insert(pack_id);
        }
        out.collections.insert(
            *collection_id,
            ArcSwap::from_pointee(RoomGeneration {
                index,
                cache: Arc::new(NodeCache::new(cache_capacity)),
            }),
        );
        Ok(())
    }

    /// Assemble a fully constructed store from the per-collection state both
    /// the rescan path and the checkpoint fast path produce, so the two share
    /// one field-for-field constructor.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        shards: ShardPool,
        scan_out: RoomScanOutput,
        collection_order: Vec<[u8; 16]>,
        deleted_collections: HashSet<[u8; 16]>,
        base_dir: PathBuf,
        swizzle: Option<SwizzleFn>,
        cache_capacity: usize,
    ) -> Self {
        Self {
            shards,
            collections: RwLock::new(scan_out.collections),
            collection_order: RwLock::new(collection_order),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            deleted_collections: parking_lot::Mutex::new(deleted_collections),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_collection: RwLock::new(HashMap::new()),
            repack_incremental: RwLock::new(HashMap::new()),
            shard_collections: RwLock::new(scan_out.shard_collections),
            collection_shards: RwLock::new(scan_out.collection_shards),
            last_shard_collections_flush: RwLock::new(None),
            index_checkpoint_dirty: AtomicBool::new(false),
        }
    }

    /// Path of this store's persisted-index checkpoint.
    fn index_checkpoint_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join(crate::index::checkpoint::INDEX_CHECKPOINT_FILE)
    }

    /// Fast-path index loading for `open_with_options`: build the
    /// per-collection state directly from the persisted-index checkpoint
    /// instead of rescanning every packfile.
    ///
    /// Returns `None` — falling through to the full rescan — for a missing,
    /// malformed, or stale checkpoint, or one that doesn't describe the pack
    /// set currently on disk. The fast path deliberately skips the torn-tail
    /// recovery scan, which is safe: a fingerprint match means the packs are
    /// the exact state the previous session's last sync fsynced, so there can
    /// be no torn tail to recover.
    fn checkpoint_scan_out(
        base_dir: &std::path::Path,
        cache_capacity: usize,
        shards: &ShardPool,
        open_shards: &[(u16, u64, PathBuf, u64)],
        deleted_collections: &HashSet<[u8; 16]>,
    ) -> Option<(RoomScanOutput, Vec<[u8; 16]>)> {
        let checkpoint =
            crate::index::checkpoint::read_checkpoint(&Self::index_checkpoint_path(base_dir))?;
        let packs: Vec<(u64, u64)> = open_shards
            .iter()
            .map(|(_, pack_id, _, file_len)| (*pack_id, *file_len))
            .collect();
        if checkpoint.fingerprint != crate::index::checkpoint::pack_fingerprint(&packs) {
            return None;
        }

        let slot_to_pack_id: HashMap<u16, u64> = open_shards
            .iter()
            .map(|(slot, pack_id, _, _)| (*slot, *pack_id))
            .collect();

        let mut scan_out = RoomScanOutput::default();
        let mut collection_order = Vec::with_capacity(checkpoint.collections.len());
        for loaded in &checkpoint.collections {
            if deleted_collections.contains(&loaded.collection_id) {
                // The checkpoint may predate the deletion marker; the
                // logical-delete set is authoritative.
                continue;
            }
            // Malformed slots mean the checkpoint can't be trusted; rescan is
            // the only faithful path.
            let index = LossyIndex::deserialize(&loaded.blob).ok()?;
            let counts = index.shard_counts();
            // Seed the home shard from the highest shard a slot references —
            // the nearest proxy for the scan path's "shard of the last
            // record", since slots were appended in shard-id order.
            if let Some(&home_shard) = counts.keys().max() {
                shards.set_collection_home(&loaded.collection_id, home_shard);
            }
            for (&shard_id, &count) in &counts {
                let pack_id = slot_to_pack_id
                    .get(&shard_id)
                    .copied()
                    .unwrap_or(u64::from(shard_id));
                scan_out
                    .shard_collections
                    .entry(pack_id)
                    .or_default()
                    .insert(loaded.collection_id, count);
                scan_out
                    .collection_shards
                    .entry(loaded.collection_id)
                    .or_default()
                    .insert(pack_id);
            }
            scan_out.collections.insert(
                loaded.collection_id,
                ArcSwap::from_pointee(RoomGeneration {
                    index,
                    cache: Arc::new(NodeCache::new(cache_capacity)),
                }),
            );
            collection_order.push(loaded.collection_id);
        }

        Some((scan_out, collection_order))
    }

    fn generation(&self, collection_id: &[u8; 16]) -> Option<arc_swap::Guard<Arc<RoomGeneration>>> {
        self.collections
            .read()
            .get(collection_id)
            .map(arc_swap::ArcSwapAny::load)
    }

    /// Collection IDs currently known to this engine, sorted for deterministic output.
    pub fn collection_ids(&self) -> Vec<[u8; 16]> {
        let mut ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Collection IDs whose live index currently references at least one record
    /// physically stored in `shard_id`.
    ///
    /// Shards are shared, so this is normally more than one collection; it's the
    /// set `repack_shard` needs to touch before that shard can retire.
    ///
    /// Sweeps the shard's own file content (bounded by `MAX_SHARD_BYTES`,
    /// not by how many unrelated collections happen to be loaded) rather than
    /// scanning every collection's index to ask "does this touch shard N" — a
    /// candidate set from the raw scan is then filtered against each
    /// candidate collection's *current* live index, since the scan alone can't
    /// tell a still-live reference from a collection that already repacked past
    /// this shard (its old bytes just haven't been overwritten — they
    /// never are, shards are append-only).
    ///
    /// # Errors
    /// Returns `StorageError` if the shard's file can't be read, or if
    /// `shard_id` doesn't correspond to a currently-open shard.
    pub fn collections_referencing_shard(
        &self,
        shard_id: u16,
    ) -> Result<Vec<[u8; 16]>, StorageError> {
        let shard = self.shards.get_shard(shard_id).ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no open shard with id {shard_id}"),
            ))
        })?;
        // O(1) against the incrementally-maintained shard→collection directory
        // instead of scanning the shard file — this used to be the
        // dominant cost of shard retirement/evacuation-style operations
        // on a large shard.
        Ok(self
            .shard_collections
            .read()
            .get(&shard.pack_id)
            .map(|collections| collections.keys().copied().collect())
            .unwrap_or_default())
    }

    /// Repack every collection that still references `shard_id`.
    ///
    /// A shard only ever retires once *every* collection referencing it has
    /// repacked past its data there (see `retire_empty_shards`) — there's
    /// no per-shard compaction primitive, since reachability (what's
    /// live vs. garbage) is inherently a per-collection concept, not a shard
    /// one. This is the targeted way to reclaim one specific shard: find
    /// every collection still pinning it live and repack each of them, instead
    /// of waiting for each collection to independently cross its own repack
    /// threshold. Live records get moved to whatever's currently that
    /// collection's home shard (`ShardPool::collection_home`) — not necessarily the
    /// pool's single "main" shard, since homes are per-collection, not global.
    ///
    /// Returns `(collection_id, kept, dropped)` for each collection repacked, in the
    /// order `collections_referencing_shard` returned them. Does not itself
    /// guarantee the shard retires — a collection with `set_live_roots` never
    /// called preserves everything (nothing is provably garbage) and its
    /// repack is a no-op for this purpose.
    ///
    /// # Errors
    /// Returns `StorageError` if any collection's repack fails; already-repacked
    /// collections in this call are not rolled back.
    pub fn repack_shard(
        &self,
        shard_id: u16,
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let collections = self.collections_referencing_shard(shard_id)?;
        let mut results = Vec::with_capacity(collections.len());
        for collection_id in collections {
            let (kept, dropped) = self
                .repack_collection_reachable(&collection_id, |hash, data| {
                    extract_edges(hash, data)
                })?;
            results.push((collection_id, kept, dropped));
        }
        Ok(results)
    }

    /// A collection's `(entry count, memory usage in bytes)`, if the collection exists.
    pub fn collection_index_info(&self, collection_id: &[u8; 16]) -> Option<(usize, usize)> {
        self.collections.read().get(collection_id).map(|gen| {
            let g = gen.load();
            (g.index.len(), g.index.memory_usage())
        })
    }

    /// `(collection_id, entry count, memory usage in bytes)` for every known collection,
    /// sorted by collection ID, in a single pass over the collection map.
    pub fn collection_summaries(&self) -> Vec<([u8; 16], usize, usize)> {
        let collections = self.collections.read();
        self.collection_order
            .read()
            .iter()
            .filter_map(|id| {
                collections.get(id).map(|gen| {
                    let g = gen.load();
                    (*id, g.index.len(), g.index.memory_usage())
                })
            })
            .collect()
    }

    /// Set the live roots to preserve for a collection on its next repack.
    pub fn set_live_roots(&self, collection_id: &[u8; 16], roots: Vec<NodeId>) {
        self.live_roots.write().insert(*collection_id, roots);
    }

    /// Set the repack trigger threshold directly, in index entries.
    pub fn set_repack_threshold_entries(&self, entries: u64) {
        self.repack_threshold_entries
            .store(entries, std::sync::atomic::Ordering::Relaxed);
    }

    /// Returns `true` if a collection's index has reached the configured repack
    /// threshold.
    ///
    /// Repacking is entirely caller-driven — nothing in this engine polls
    /// this on its own. A background GC worker is expected to call this
    /// periodically and issue `repack_collection_reachable` itself; nothing in
    /// `mtxdb-core` currently does so.
    #[must_use]
    pub fn needs_repack(&self, collection_id: &[u8; 16]) -> bool {
        let count = self
            .collection_index_info(collection_id)
            .map_or(0, |(len, _)| len as u64);
        count
            >= self
                .repack_threshold_entries
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The collection-scoped node cache for `collection_id`, or a fresh empty one if the collection is unknown.
    pub fn collection_cache(&self, collection_id: &[u8; 16]) -> Arc<NodeCache> {
        self.generation(collection_id).map_or_else(
            || Arc::new(NodeCache::new(self.cache_capacity)),
            |g| g.cache.clone(),
        )
    }

    fn put_mutex(&self, collection_id: &[u8; 16]) -> Arc<parking_lot::Mutex<()>> {
        let mut locks = self.put_locks.lock();
        locks.entry(*collection_id).or_default().clone()
    }

    fn read_at(shard: &Shard, offset: u64, verify: bool) -> Result<Record, StorageError> {
        ShardPool::read_at(shard, offset, verify)
    }

    /// The record's actual on-disk byte length (see
    /// [`ShardPool::record_disk_len_at`]) — the true disk-usage figure,
    /// unlike `Record::serialized_len()` which is an uncompressed upper
    /// bound.
    fn record_disk_len_at(shard: &Shard, offset: u64) -> Result<u64, StorageError> {
        ShardPool::record_disk_len_at(shard, offset)
    }

    fn scan_collection_records(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<Vec<ScannedShard>, StorageError> {
        let mut scanned: Vec<ScannedShard> = Vec::new();
        for (shard_id, shard) in self.shards.all_shards() {
            match packfile::scan_packfile(&shard.path) {
                Ok(entries) => {
                    let collection_entries: Vec<([u8; 16], u64)> = entries
                        .into_iter()
                        .filter(|(rid, _, _)| rid == collection_id)
                        .map(|(_, hash, offset)| (hash, offset))
                        .collect();
                    if !collection_entries.is_empty() {
                        scanned.push((shard_id, collection_entries));
                    }
                }
                Err(e) => {
                    return Err(StorageError::Io(e));
                }
            }
        }
        Ok(scanned)
    }

    /// Scan every open shard once and build deduplicated record maps for the
    /// requested collections. This is the batch counterpart to `scan_collection_records`:
    /// a shard-compaction preflight must not reread the same packfile once per
    /// collection in its closure.
    fn scan_collection_record_maps(
        &self,
        collection_ids: &[[u8; 16]],
    ) -> Result<HashMap<[u8; 16], RepackRecordMap>, StorageError> {
        let wanted: HashSet<[u8; 16]> = collection_ids.iter().copied().collect();
        let mut maps: HashMap<[u8; 16], RepackRecordMap> = wanted
            .iter()
            .copied()
            .map(|collection_id| (collection_id, HashMap::new()))
            .collect();
        for (shard_id, shard) in self.shards.all_shards() {
            for (collection_id, hash, offset) in packfile::scan_packfile(&shard.path)? {
                if wanted.contains(&collection_id) {
                    maps.entry(collection_id)
                        .or_default()
                        .insert(hash, (shard_id, offset));
                }
            }
        }
        Ok(maps)
    }

    fn build_index(offsets: &[([u8; 16], u16, u64)]) -> Result<LossyIndex, StorageError> {
        let mut index = LossyIndex::new(offsets.len().saturating_mul(2).max(16));
        for (hash, shard_id, offset) in offsets {
            check_index_offset(*shard_id, hash, *offset)?;
            let _ = index.insert(hash, *shard_id, *offset);
        }
        Ok(index)
    }

    /// Convert a slot-keyed `shard_counts` from `LossyIndex` to a pack_id-keyed
    /// `HashMap<u64, u64>` for use with `replace_collection_shard_counts`.
    fn slot_counts_to_pack_id_counts(&self, counts: &HashMap<u16, u64>) -> HashMap<u64, u64> {
        let shards = self.shards.all_shards();
        let slot_to_pack_id: HashMap<u16, u64> = shards
            .iter()
            .map(|(slot, shard)| (*slot, shard.pack_id))
            .collect();
        counts
            .iter()
            .map(|(&slot, &count)| {
                let pack_id = slot_to_pack_id
                    .get(&slot)
                    .copied()
                    .unwrap_or(u64::from(slot));
                (pack_id, count)
            })
            .collect()
    }

    fn deleted_collections_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("deleted.collections")
    }

    fn load_deleted_collections(base_dir: &std::path::Path) -> HashSet<[u8; 16]> {
        let path = Self::deleted_collections_path(base_dir);
        let Ok(bytes) = fs::read(&path) else {
            return HashSet::new();
        };
        bytes
            .chunks_exact(16)
            .map(|chunk| {
                let mut id = [0u8; 16];
                id.copy_from_slice(chunk);
                id
            })
            .collect()
    }

    fn persist_deleted_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_collections_path(&self.base_dir);
        let mut set = self.deleted_collections.lock();
        set.insert(*collection_id);
        let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
        fs::write(&path, &bytes).map_err(StorageError::Io)?;
        Ok(())
    }

    /// Fast path for `store_generation`'s new-collection case: most
    /// collections were never deleted, so check the in-memory cache before
    /// paying for a lock + no-op write. Avoids re-reading `deleted.collections`
    /// from disk on every first write to a new collection — previously this
    /// reloaded the whole file from disk unconditionally, which under a cold
    /// page cache turned a batch of first-writes across many collections into
    /// one serialized disk read per collection.
    fn is_deleted_collection(&self, collection_id: &[u8; 16]) -> bool {
        self.deleted_collections.lock().contains(collection_id)
    }

    fn clear_deleted_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let path = Self::deleted_collections_path(&self.base_dir);
        let mut set = self.deleted_collections.lock();
        if set.remove(collection_id) {
            let bytes: Vec<u8> = set.iter().flat_map(|id| id.iter().copied()).collect();
            fs::write(&path, &bytes).map_err(StorageError::Io)?;
        }
        Ok(())
    }

    /// Records one new record landing in `shard_id` for `collection_id` — the
    /// cheap, O(1) path for a plain `put` that only appends, never moves
    /// or drops anything. Kept separate from
    /// [`Self::replace_collection_shard_counts`], which pays for a full
    /// per-shard recount and is reserved for the cases that actually
    /// change a collection's existing distribution (an index rebuild or a
    /// repack), so a normal write never regresses to O(collection size).
    fn record_new_shard_collection(&self, pack_id: u64, collection_id: &[u8; 16]) {
        let mut shard_collections = self.shard_collections.write();
        let count = shard_collections
            .entry(pack_id)
            .or_default()
            .entry(*collection_id)
            .or_insert(0);
        *count = count.saturating_add(1);
        drop(shard_collections);
        self.collection_shards
            .write()
            .entry(*collection_id)
            .or_default()
            .insert(pack_id);
    }

    /// Replaces `collection_id`'s entire contribution to `shard_collections` with
    /// `counts` (typically `LossyIndex::shard_counts()` converted to `pack_id` keys)
    /// — clears it out of any shard it no longer occupies and installs the fresh
    /// per-shard counts. Used wherever a collection's index is replaced wholesale
    /// rather than incrementally appended to, since only then can its distribution
    /// across shards actually change.
    fn replace_collection_shard_counts(
        &self,
        collection_id: &[u8; 16],
        counts: &HashMap<u64, u64>,
    ) {
        let old_shards = self
            .collection_shards
            .write()
            .insert(*collection_id, counts.keys().copied().collect());
        let mut shard_collections = self.shard_collections.write();
        if let Some(old_shards) = old_shards {
            for pack_id in &old_shards {
                if !counts.contains_key(pack_id) {
                    if let Some(m) = shard_collections.get_mut(pack_id) {
                        m.remove(collection_id);
                        if m.is_empty() {
                            shard_collections.remove(pack_id);
                        }
                    }
                }
            }
        }
        for (&pack_id, &count) in counts {
            shard_collections
                .entry(pack_id)
                .or_default()
                .insert(*collection_id, count);
        }
    }

    /// Removes `collection_id` from `shard_collections`/`collection_shards` entirely —
    /// used on collection deletion, where nothing of the collection survives in any
    /// shard.
    fn remove_collection_shard_counts(&self, collection_id: &[u8; 16]) {
        let Some(old_shards) = self.collection_shards.write().remove(collection_id) else {
            return;
        };
        let mut shard_collections = self.shard_collections.write();
        for pack_id in old_shards {
            if let Some(m) = shard_collections.get_mut(&pack_id) {
                m.remove(collection_id);
                if m.is_empty() {
                    shard_collections.remove(&pack_id);
                }
            }
        }
    }

    /// Path to the persisted shard→collection directory for a base directory.
    fn shard_collections_path(base_dir: &std::path::Path) -> PathBuf {
        base_dir.join("shard_collections.bin")
    }

    /// Persist the current `shard_collections` directory to disk: which collections
    /// have live records in each shard, and how many. Read by
    /// [`Self::collection_directory_from_disk`] — a free function a separate,
    /// short-lived process (a `collections`-style CLI listing) can call without
    /// opening a full `PackfileStorage` (which would otherwise mean
    /// scanning and index-building every shard just to answer "what collections
    /// exist and how big are they").
    ///
    /// Writes to a temp file and renames into place, same crash-safety
    /// pattern as `ShardPool::persist_stats`.
    ///
    /// # Errors
    /// Returns `StorageError` on write or rename failure.
    pub fn persist_shard_collections(&self) -> Result<(), StorageError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(SHARD_ROOMS_MAGIC);
        buf.push(SHARD_ROOMS_VERSION);
        let persisted_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        buf.extend_from_slice(&persisted_at.to_le_bytes());
        let collection_order: HashMap<[u8; 16], u64> = self
            .collection_order
            .read()
            .iter()
            .enumerate()
            .map(|(index, collection_id)| {
                u64::try_from(index)
                    .map(|index| (*collection_id, index))
                    .map_err(|_| {
                        StorageError::Io(std::io::Error::other("collection order exceeds u64"))
                    })
            })
            .collect::<Result<_, _>>()?;
        for (pack_id, collections) in self.shard_collections.read().iter() {
            for (collection_id, count) in collections {
                buf.extend_from_slice(&pack_id.to_le_bytes());
                buf.extend_from_slice(collection_id);
                buf.extend_from_slice(&count.to_le_bytes());
                let order = collection_order
                    .get(collection_id)
                    .copied()
                    .unwrap_or(u64::MAX);
                buf.extend_from_slice(&order.to_le_bytes());
            }
        }

        let unique = SHARD_ROOMS_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = Self::shard_collections_path(&self.base_dir)
            .with_extension(format!("bin.tmp.{}.{unique}", std::process::id()));
        let final_path = Self::shard_collections_path(&self.base_dir);
        let write_result = (|| -> std::io::Result<()> {
            let mut tmp = fs::File::create(&tmp_path)?;
            std::io::Write::write_all(&mut tmp, &buf)?;
            tmp.sync_all()
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(StorageError::Io(e));
        }
        fs::rename(&tmp_path, &final_path).map_err(StorageError::Io)?;
        Ok(())
    }

    /// Best-effort wrapper around [`Self::persist_shard_collections`] — logs and
    /// swallows a failure rather than turning it into a hard error, same
    /// contract as `ShardPool`'s stats persistence: this is observability
    /// data, not something worth failing an otherwise-successful sync
    /// over.
    fn persist_shard_collections_best_effort(&self) {
        if let Err(e) = self.persist_shard_collections() {
            eprintln!("mtxdb: failed to persist shard→collection directory: {e}");
        }
    }

    /// Persist the full per-collection index state to `index.checkpoint`, so
    /// the next open can load it instead of rescanning every packfile.
    ///
    /// Called after every sync barrier — packfile data first, checkpoint
    /// second. A crash in between leaves a fingerprint mismatch that the next
    /// open resolves with a rescan; a crash after leaves a valid checkpoint.
    ///
    /// Cheap no-op when no collection data has changed since the last write
    /// (`index_checkpoint_dirty` cleared on success), so a writer that syncs
    /// repeatedly without writes doesn't rewrite the acceleration file. The
    /// flag is only cleared after a successful write, so a transient failure
    /// defers rather than drops the update.
    fn persist_index_checkpoint(&self) -> Result<(), StorageError> {
        if !self.index_checkpoint_dirty.load(Ordering::Relaxed) {
            return Ok(());
        }
        let packs: Vec<(u64, u64)> = self
            .shards
            .all_shards()
            .into_iter()
            .map(|(_, shard)| (shard.pack_id, shard.file_len()))
            .collect();
        let fingerprint = crate::index::checkpoint::pack_fingerprint(&packs);

        let order = self.collection_order.read().clone();
        let collections = self.collections.read();
        let mut entries: Vec<([u8; 16], Vec<u8>)> = Vec::with_capacity(order.len());
        for collection_id in &order {
            let Some(generation) = collections
                .get(collection_id)
                .map(arc_swap::ArcSwapAny::load)
            else {
                continue;
            };
            entries.push((*collection_id, generation.index.serialize()));
        }
        let blobs: Vec<([u8; 16], &[u8])> = entries
            .iter()
            .map(|(collection_id, blob)| (*collection_id, blob.as_slice()))
            .collect();
        crate::index::checkpoint::write_checkpoint(
            &Self::index_checkpoint_path(&self.base_dir),
            fingerprint,
            &blobs,
        )
        .map_err(StorageError::Io)?;
        self.index_checkpoint_dirty.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Best-effort wrapper around [`Self::persist_index_checkpoint`] — logs
    /// and swallows a failure rather than turning it into a hard error. The
    /// checkpoint is an acceleration structure; packfiles remain
    /// authoritative, so a failed persist only costs the next open a rescan.
    fn persist_index_checkpoint_best_effort(&self) {
        if let Err(e) = self.persist_index_checkpoint() {
            eprintln!("mtxdb: failed to persist index checkpoint: {e}");
        }
    }

    /// Reads a persisted shard→collection directory directly off disk, with no
    /// `ShardPool`/`PackfileStorage` construction at all — the fast path
    /// for a `collections`-style CLI listing. Returns each collection's total record
    /// count summed across every shard it appears in, in insertion order.
    ///
    /// Best-effort: a missing, truncated, or corrupt file just yields an
    /// empty result rather than an error — the caller decides whether to
    /// fall back to a full scan (e.g. because this store predates the
    /// feature, or a writer hasn't flushed it yet).
    #[must_use]
    pub fn collection_directory_from_disk(base_dir: &std::path::Path) -> Vec<([u8; 16], u64)> {
        let Some((_, records)) = read_persisted_shard_collections(base_dir) else {
            return Vec::new();
        };
        let mut totals: HashMap<[u8; 16], (u64, u64)> = HashMap::new();
        for record in records {
            let entry = totals
                .entry(record.collection_id)
                .or_insert((0, record.insertion_order));
            entry.0 = entry.0.saturating_add(record.count);
            entry.1 = entry.1.min(record.insertion_order);
        }
        let mut result: Vec<([u8; 16], u64, u64)> = totals
            .into_iter()
            .map(|(collection_id, (count, order))| (collection_id, count, order))
            .collect();
        result.sort_unstable_by_key(|(collection_id, _, order)| (*order, *collection_id));

        result
            .into_iter()
            .map(|(collection_id, count, _)| (collection_id, count))
            .collect()
    }

    /// Reads collection summaries from the persisted shard→collection directory without
    /// opening packfiles. The node count and index-RAM figure match the
    /// index rebuilt by [`Self::open`], provided the sidecar is current.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn collection_summaries_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<Vec<([u8; 16], usize, usize)>> {
        Self::collection_directory_persisted_at(base_dir)?;
        Self::collection_directory_from_disk(base_dir)
            .into_iter()
            .map(|(collection_id, nodes)| {
                let nodes = usize::try_from(nodes).ok()?;
                Some((
                    collection_id,
                    nodes,
                    LossyIndex::memory_usage_for_entries(nodes),
                ))
            })
            .collect()
    }

    /// Current live shard `pack_id`s per collection from the persisted shard→collection
    /// directory, without opening packfiles or rebuilding collection indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn collection_shards_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<HashMap<[u8; 16], Vec<u64>>> {
        let (_, records) = read_persisted_shard_collections(base_dir)?;
        let mut shards: HashMap<[u8; 16], Vec<u64>> = HashMap::new();
        for record in records {
            shards
                .entry(record.collection_id)
                .or_default()
                .push(record.pack_id);
        }
        for collection_shards in shards.values_mut() {
            collection_shards.sort_unstable();
            collection_shards.dedup();
        }
        Some(shards)
    }

    /// Current live-node count per shard from the persisted shard→collection
    /// directory, without opening packfiles or rebuilding collection indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed. Callers that need an authoritative fresh count can then
    /// explicitly open the store and rebuild its indexes instead.
    #[must_use]
    pub fn shard_node_counts_from_disk(base_dir: &std::path::Path) -> Option<HashMap<u64, u64>> {
        let (_, records) = read_persisted_shard_collections(base_dir)?;
        let mut totals = HashMap::new();
        for record in records {
            let total = totals.entry(record.pack_id).or_insert(0_u64);
            *total = total.saturating_add(record.count);
        }
        Some(totals)
    }

    /// Current number of collections with live nodes per shard from the persisted
    /// shard→collection directory, without opening packfiles or rebuilding collection
    /// indexes.
    ///
    /// Returns `None` when the directory has not been persisted yet or is
    /// malformed.
    #[must_use]
    pub fn shard_collection_counts_from_disk(
        base_dir: &std::path::Path,
    ) -> Option<HashMap<u64, u64>> {
        let (_, records) = read_persisted_shard_collections(base_dir)?;
        let mut totals = HashMap::new();
        for record in records {
            let count = totals.entry(record.pack_id).or_insert(0_u64);
            *count = count.saturating_add(1);
        }
        Some(totals)
    }

    /// Unix-seconds timestamp of the persisted shard→collection directory, if
    /// one exists — lets a reader (e.g. a `collections` CLI listing) label how
    /// stale the counts it's showing are, the same way
    /// `ShardPool::stats_persisted_at` does for shard IO stats.
    #[must_use]
    pub fn collection_directory_persisted_at(base_dir: &std::path::Path) -> Option<u64> {
        read_persisted_shard_collections(base_dir).map(|(persisted_at, _)| persisted_at)
    }

    /// Pin one `Arc<Shard>` per distinct shard id, keeping each shard's file
    /// handle (and thus its `Drop`-triggered deletion, see `Shard::drop`)
    /// alive for as long as the returned map is held — even if a later
    /// write in the same operation triggers `ShardPool::rotate` and retires
    /// that shard id's pool slot.
    ///
    /// This matters specifically for a scan-then-rewrite repack: without
    /// it, re-querying `ShardPool::get_shard` on every loop iteration can
    /// observe a shard that a *rotation triggered by the same repack's own
    /// writes* just retired-and-recreated at the same path, turning a
    /// stale-but-valid offset into a read into unrelated, freshly-written
    /// bytes.
    fn pin_shards(&self, shard_ids: impl Iterator<Item = u16>) -> HashMap<u16, Arc<Shard>> {
        let unique: HashSet<u16> = shard_ids.collect();
        unique
            .into_iter()
            .filter_map(|id| self.shards.get_shard(id).map(|shard| (id, shard)))
            .collect()
    }

    /// Copy a record from an old (pinned) shard to the active shard.
    /// Returns the new `(hash, shard_id, offset)` or None if `old_shard_id`
    /// isn't in `pinned`.
    fn copy_record_to_shard(
        &self,
        collection_id: &[u8; 16],
        pinned: &HashMap<u16, Arc<Shard>>,
        old_shard_id: u16,
        old_offset: u64,
    ) -> Result<Option<([u8; 16], u16, u64)>, StorageError> {
        let Some(old_shard) = pinned.get(&old_shard_id) else {
            return Ok(None);
        };
        let record = Self::read_at(old_shard, old_offset, true)?;
        let (new_shard_id, new_offset) = self.shards.put_record(&Record {
            collection_id: *collection_id,
            hash: record.hash,
            data: record.data,
        })?;
        Ok(Some((record.hash, new_shard_id, new_offset)))
    }
    fn swap_generation(
        &self,
        collection_id: &[u8; 16],
        index: LossyIndex,
    ) -> Result<(), StorageError> {
        let cache = self.generation(collection_id).map(|g| g.cache.clone());
        self.store_generation(collection_id, index, cache)
    }

    /// Store a new generation for a collection, reusing the existing cache if present.
    ///
    /// If the collection is new (not yet in the `collections` map), it is added to
    /// `collection_order` so that [`Self::collection_summaries`] will include it, and
    /// any prior tombstone in `deleted.collections` is cleared so the collection
    /// survives a restart.
    fn store_generation(
        &self,
        collection_id: &[u8; 16],
        index: LossyIndex,
        cache: Option<Arc<NodeCache>>,
    ) -> Result<(), StorageError> {
        let cache = cache.unwrap_or_else(|| Arc::new(NodeCache::new(self.cache_capacity)));
        let new_gen = Arc::new(RoomGeneration { index, cache });

        let is_new = self.collections.read().get(collection_id).is_none();

        // Clear a prior deletion marker before publishing the recreated
        // generation. A failure must fail the write: otherwise it appears to
        // succeed but disappears on the next startup scan.
        if is_new {
            if self.is_deleted_collection(collection_id) {
                self.clear_deleted_collection(collection_id)?;
            }
            self.collections
                .write()
                .entry(*collection_id)
                .or_insert_with(|| {
                    ArcSwap::from_pointee(RoomGeneration {
                        index: LossyIndex::new(0),
                        cache: Arc::new(NodeCache::new(self.cache_capacity)),
                    })
                })
                .store(new_gen);

            if !self.collection_order.read().contains(collection_id) {
                self.collection_order.write().push(*collection_id);
            }
        } else {
            // Fast path: just update the ArcSwap using a read lock on the map.
            // This avoids a global write lock on every single put() call.
            let read_guard = self.collections.read();
            if let Some(arc_swap) = read_guard.get(collection_id) {
                arc_swap.store(new_gen);
            } else {
                // Fallback in case of a race condition with a deletion
                drop(read_guard);
                self.collections
                    .write()
                    .entry(*collection_id)
                    .or_insert_with(|| {
                        ArcSwap::from_pointee(RoomGeneration {
                            index: LossyIndex::new(0),
                            cache: Arc::new(NodeCache::new(self.cache_capacity)),
                        })
                    })
                    .store(new_gen);
            }
        }
        self.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }
    /// Forces a re-scan of a collection's shards from disk and atomically swaps in the new index.
    /// This is designed for multi-worker environments to pull in external appends on demand.
    ///
    /// # Errors
    /// Returns `StorageError` if reading or parsing the underlying shards fails.
    pub fn refresh_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        self.shards.discover_shards()?;
        let new_index = self.rebuild_index(collection_id)?;
        let existing_cache = self.generation(collection_id).map(|g| g.cache.clone());
        self.store_generation(collection_id, new_index, existing_cache)
    }

    fn rebuild_index(&self, collection_id: &[u8; 16]) -> Result<LossyIndex, StorageError> {
        let scanned = self.scan_collection_records(collection_id)?;
        let total: usize = scanned.iter().map(|(_, e)| e.len()).sum();
        let mut index = LossyIndex::new(total.saturating_mul(2).max(16));
        for (shard_id, entries) in scanned {
            for (hash, offset) in entries {
                // `IndexSlot` can only represent offsets up to
                // `PACK_INDEX_OFFSET_LIMIT`. Offsets a legacy or externally
                // created oversized shard can no longer fit are rejected
                // rather than silently skipped: writes are already capped at
                // `MAX_SHARD_BYTES`, so a record beyond the limit means the
                // shard did not come from this engine, and silently indexing
                // around it would make its data unreachable while reporting a
                // successful rebuild.
                check_index_offset(shard_id, &hash, offset)?;
                let _ = index.insert(&hash, shard_id, offset);
            }
        }
        Ok(index)
    }

    /// Fetch a node and return it as a `NodeRef` with swizzled children.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure or CRC mismatch.
    pub fn get_swizzled(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        extract_children: impl Fn(&NodeData) -> Vec<NodeId>,
    ) -> Result<Option<NodeRef>, StorageError> {
        let Some(data) = self.get(collection_id, id)? else {
            return Ok(None);
        };

        if let Some(swizzle_fn) = self.swizzle {
            let children = extract_children(&data);
            if !children.is_empty() {
                let cached = if let Some(gen) = self.generation(collection_id) {
                    gen.cache.resolve_hashes(&children)
                } else {
                    vec![None; children.len()]
                };
                let swizzled = swizzle_fn(&data, &children, &cached);
                if let Some(gen) = self.generation(collection_id) {
                    gen.cache.insert(*id, Arc::new(swizzled.clone()));
                }
                return Ok(Some(NodeRef::Resolved(*id, Arc::new(swizzled))));
            }
        }

        Ok(Some(NodeRef::Resolved(*id, Arc::new(data))))
    }

    fn resolve_from_candidates(
        &self,
        id: &NodeId,
        candidates: impl IntoIterator<Item = (u16, u64)>,
    ) -> Result<Option<NodeData>, StorageError> {
        let mut last_err: Option<StorageError> = None;
        for (shard_id, offset) in candidates {
            let Some(shard) = self.shards.get_shard(shard_id) else {
                continue;
            };

            match Self::read_at(
                &shard,
                offset,
                self.shards.checksum_policy().verifies_reads(),
            ) {
                Ok(record) => {
                    if record.hash != *id {
                        continue;
                    }

                    let data = NodeData {
                        bytes: record.data,
                        children: Vec::new(),
                    };
                    // A raw mmap-backed `Bytes` is already cheap to retain
                    // in the caller. Do not turn every cold lookup into an
                    // LRU write (hash probe, lock, and eviction bookkeeping).
                    // The cache remains populated by writes and swizzling,
                    // where it stores work that cannot be recovered by a
                    // simple mmap range.
                    return Ok(Some(data));
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        if let Some(err) = last_err {
            return Err(err);
        }
        Ok(None)
    }

    // repack_collection_rewrite and repack_collection_topo were retired: both scanned
    // for records without deduplicating physical duplicates (each prior
    // repack's rewritten copies included), giving an exponential blowup on
    // repeated calls against a growing collection, and both re-fetched
    // Arc<Shard> per loop iteration rather than pinning shards for the
    // call's duration — vulnerable to the rotation race documented on
    // `pin_shards`. `repack_collection_reachable`'s no-live-roots fallback
    // subsumes both: same topological ordering, correct dedup via
    // `hash_to_shard_offset`, and pinned shards throughout.

    /// BFS outward from `roots` over `extract_edges`, reading only records
    /// actually reached. Returns the sorted, deduplicated live hash set and
    /// the adjacency discovered along the way.
    fn bfs_live_set(
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut visited: HashSet<[u8; 16]> = HashSet::new();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(&(shard_id, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&shard_id) else {
                continue;
            };
            let record = Self::read_at(old_shard, offset, true)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        let mut live_hashes: Vec<[u8; 16]> = visited.into_iter().collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// BFS backward (toward ancestors) from `frontier` over `extract_edges`,
    /// stopping expansion (and excluding from the result) anything in
    /// `stop_at`. Unlike [`Self::bfs_live_set`]/[`Self::bfs_bounded`]'s
    /// repack-side sibling, this resolves each node through `gen`'s index
    /// — a single frozen generation snapshot taken once before the walk
    /// starts — one hash lookup at a time, rather than pre-scanning every
    /// shard for the whole collection up front. Cost is proportional to the
    /// number of nodes actually walked, not collection size.
    ///
    /// Pinning `gen` for the whole walk (instead of re-resolving through
    /// `self.get()` per node against whatever generation happens to be
    /// live at call time) is also what makes this concurrency-safe: a
    /// repack that runs mid-walk and GCs a hash this walk already needs
    /// swaps in a *new* generation rather than mutating this one, so a
    /// node this walk has already committed to visiting can't vanish out
    /// from under it and get silently skipped.
    ///
    /// Returns the resolved `NodeData` for each visited node alongside the
    /// adjacency, so the caller has everything it needs without a second
    /// per-node fetch.
    ///
    /// This is an **ancestor walk with stop markers**, not a "span between
    /// two frontiers": a branch whose history never crosses any `stop_at`
    /// node is walked all the way back to the collection's true roots, not
    /// truncated at some implied boundary. A real from/to span would need
    /// `ancestors(frontier) ∩ descendants(stop_at)`, which requires
    /// reverse adjacency this call doesn't have — out of scope here.
    ///
    /// `limits.max_nodes`, if set, caps how many nodes this walk will visit
    /// before giving up on expanding further (existing queued nodes still
    /// finish resolving their own edges into `adjacency`, but no new nodes
    /// are enqueued past the cap) — the safety valve for exactly the
    /// runaway case above, where `stop_at` never gets hit.
    fn bfs_ancestors(
        &self,
        gen: &Arc<RoomGeneration>,
        frontier: &[[u8; 16]],
        stop_at: &[[u8; 16]],
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        limits: WalkLimits,
    ) -> Result<(AdjacencyResult, HashMap<[u8; 16], NodeData>), StorageError> {
        let boundary: HashSet<[u8; 16]> = stop_at.iter().copied().collect();
        let mut visited: HashSet<[u8; 16]> = HashSet::new();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        let mut resolved: HashMap<[u8; 16], NodeData> = HashMap::new();
        let at_cap =
            |visited: &HashSet<[u8; 16]>| limits.max_nodes.is_some_and(|max| visited.len() >= max);

        for root in frontier {
            if !boundary.contains(root) && !at_cap(&visited) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            let Some(data) = self.resolve_pinned(gen, &hash)? else {
                // Not actually present under this generation — nothing to
                // expand from and nothing to yield.
                continue;
            };
            let edges = extract_edges(&hash, &data.bytes);
            for edge in &edges {
                if !boundary.contains(edge) && !at_cap(&visited) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
            resolved.insert(hash, data);
        }

        // live_hashes tracks only nodes actually resolved, not everything
        // `visited` touched — a hash that turned out absent under `gen`
        // was marked visited to prevent re-queueing, but never belongs in
        // the CSR or the output.
        let mut live_hashes: Vec<[u8; 16]> = resolved.keys().copied().collect();
        live_hashes.sort_unstable();
        Ok(((live_hashes, adjacency), resolved))
    }

    /// Resolves `id` within a specific, already-loaded generation snapshot
    /// rather than whatever generation happens to be live when called —
    /// the building block [`Self::bfs_ancestors`] uses to keep an entire
    /// walk consistent against one point-in-time view of the collection, immune
    /// to a repack swapping in a new generation partway through.
    fn resolve_pinned(
        &self,
        gen: &Arc<RoomGeneration>,
        id: &NodeId,
    ) -> Result<Option<NodeData>, StorageError> {
        if let Some(data) = gen.cache.get(id) {
            return Ok(Some((*data).clone()));
        }
        self.resolve_from_candidates(id, gen.index.lookup_all(id))
    }

    /// Read every record's edges, with no reachability filtering.
    /// Used when a collection has no configured live roots, so nothing is known
    /// to be garbage. Iterates `hash_to_shard_offset` directly rather
    /// than the raw scan output, so it works for both full and incremental
    /// repack (where the map was built by merging new entries into a
    /// previous state rather than scanning every packfile from byte zero).
    fn scan_full_adjacency(
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = HashMap::new();
        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&shard_id) {
                let record = Self::read_at(shard, offset, true)?;
                let edges = extract_edges(hash, &record.data);
                adjacency.insert(*hash, edges);
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Rewrite a collection's records in topological order, optionally performing
    /// garbage collection — this is the engine's one repack entry point.
    ///
    /// If live roots are configured for this collection (see
    /// [`Self::set_live_roots`]), traverses outward from them via
    /// `extract_edges` and drops anything not reached: only reachable
    /// records are read from disk during the traversal, so unreachable
    /// data is never even fetched, not just excluded from the output.
    ///
    /// If no live roots are configured (`set_live_roots` was never called,
    /// or was called with an empty list), nothing is known to be garbage,
    /// so every scanned record is preserved, still deduplicated and
    /// rewritten in topological order — rather than risk deleting live
    /// data on the assumption that "no roots" means "nothing is live".
    ///
    /// All shards this call will read from are pinned (held via an
    /// `Arc<Shard>` for the whole call) before any writes happen, so a
    /// rotation triggered by this call's own writes can never retire a
    /// shard this
    /// call still needs to read stale data from.
    ///
    /// Returns `(kept, dropped)`: the number of records written to the new
    /// generation, and the number of scanned records found unreachable and
    /// discarded.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    /// Builds this repack's deduped `hash → (shard_id, offset)` map,
    /// incrementally where possible, and returns the previous
    /// [`RepackIncrementalState`] (moved out, not cloned) so the caller can
    /// reuse its `adjacency` cache when deciding which nodes need fresh disk
    /// reads.
    ///
    /// On the incremental path the previous `live_map` is extended in-place
    /// with only the newly-appended entries from each shard (O(delta) scan
    /// work per repack call rather than O(total)). On the cold-start path
    /// (first repack for this collection) every shard is scanned from byte
    /// zero.
    ///
    /// Returns `(hash_to_shard_offset, Some(prev))` on the incremental path
    /// and `(hash_to_shard_offset, None)` on the cold-start path. The caller
    /// must pass `prev` to the adjacency helpers to unlock the incremental
    /// BFS optimisation; when it is `None` the caller uses the full-scan
    /// adjacency helpers instead.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure scanning a shard.
    fn repack_scan_incremental(
        &self,
        collection_id: &[u8; 16],
    ) -> Result<RepackScanResult, StorageError> {
        // Check for existing state under a read lock, then move it out under a
        // write lock if present — O(1) ownership transfer, no clone.
        let has_prev = self.repack_incremental.read().contains_key(collection_id);
        if has_prev {
            let prev = self
                .repack_incremental
                .write()
                .remove(collection_id)
                .expect("entry was present under read lock and collection mutex is held");

            let shards = self.shards.all_shards();
            let current_pack_ids: HashMap<u16, u64> = shards
                .iter()
                .map(|(shard_id, shard)| (*shard_id, shard.pack_id))
                .collect();
            // Slot IDs are recyclable.  A cursor (and every entry in the
            // carried live_map) is meaningful only for the pack incarnation
            // that produced it.  On any replacement or disappearance, start
            // over rather than mixing old offsets with the new pack.
            let pack_changed = prev
                .scan_offsets
                .iter()
                .any(|(shard_id, (pack_id, _))| current_pack_ids.get(shard_id) != Some(pack_id));
            if pack_changed {
                let scanned = self.scan_collection_records(collection_id)?;
                let mut map = HashMap::new();
                for (shard_id, entries) in scanned {
                    for (hash, offset) in entries {
                        map.insert(hash, (shard_id, offset));
                    }
                }
                return Ok((map, None));
            }

            // Scan only bytes appended since the last repack, merging into
            // the stolen live_map in-place — O(delta) work, not O(total).
            let mut map = prev.live_map.clone();
            for (shard_id, shard) in shards {
                let start = prev
                    .scan_offsets
                    .get(&shard_id)
                    .filter(|(pack_id, _)| *pack_id == shard.pack_id)
                    .map_or(0, |(_, offset)| *offset);
                let entries =
                    packfile::scan_packfile_from(&shard.path, start).map_err(StorageError::Io)?;
                for (rid, hash, offset) in entries {
                    if rid == *collection_id {
                        map.insert(hash, (shard_id, offset));
                    }
                }
            }
            Ok((map, Some(prev)))
        } else {
            // First repack: full scan of every shard.
            let scanned = self.scan_collection_records(collection_id)?;
            let mut map = HashMap::new();
            for (shard_id, entries) in &scanned {
                for (hash, offset) in entries {
                    map.insert(*hash, (*shard_id, *offset));
                }
            }
            Ok((map, None))
        }
    }

    /// Incremental BFS reachability walk.
    ///
    /// Uses the adjacency cache from `prev` to avoid re-reading disk for nodes
    /// that were already known-live after the last repack. Only newly-added
    /// nodes (those absent from `prev.adjacency`) require a disk read.
    ///
    /// **Root-removal invariant (critical):** This function must NOT be called
    /// when the current root set is a strict subset of `prev.prev_roots` (i.e.
    /// any root was removed since the last repack). In that case nodes
    /// exclusively reachable through the removed root would remain in the live
    /// set indefinitely, because `visited` in `prev.adjacency` only grows —
    /// nothing in this incremental path forces a re-check of reachability.
    /// The caller ([`Self::repack_collection_reachable`]) detects root removal
    /// and falls back to [`Self::bfs_live_set`] (cold-start, full BFS) for
    /// that repack, then saves a fresh cursor so subsequent repacks can be
    /// incremental again.
    ///
    /// **Memory:** `prev.adjacency` holds `O(live_nodes` × `avg_fanout`) hashes
    /// (≈ `live_nodes × avg_fanout × 16` bytes) in RAM per collection. This
    /// is the cost of eliminating the O(live) disk reads per repack call —
    /// callers should be aware for large collections.
    fn bfs_live_set_incremental(
        roots: &[[u8; 16]],
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        // Seed with roots that are not already in the cached live set.
        // Anything in prev.adjacency was reachable last time and — given no
        // root was removed — is still reachable now; no need to re-expand it.
        let mut visited: HashSet<[u8; 16]> = prev.adjacency.keys().copied().collect();
        let mut queue: VecDeque<[u8; 16]> = VecDeque::new();
        // Retain the cached edges for already-live nodes.  Initialising these
        // entries with empty vectors loses prior edges when they are not put
        // back on the queue during an incremental rooted repack.
        let mut adjacency = prev.adjacency.clone();

        for root in roots {
            if hash_to_shard_offset.contains_key(root) && visited.insert(*root) {
                queue.push_back(*root);
            }
        }

        while let Some(hash) = queue.pop_front() {
            // Fast path: edge list already cached — no disk read needed.
            if let Some(edges) = prev.adjacency.get(&hash) {
                for edge in edges {
                    if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                        queue.push_back(*edge);
                    }
                }
                adjacency.insert(hash, edges.clone());
                continue;
            }

            // Slow path: new node, must read from disk.
            let Some(&(shard_id, offset)) = hash_to_shard_offset.get(&hash) else {
                continue;
            };
            let Some(old_shard) = pinned.get(&shard_id) else {
                continue;
            };
            let record = Self::read_at(old_shard, offset, true)?;
            let edges = extract_edges(&hash, &record.data);
            for edge in &edges {
                if hash_to_shard_offset.contains_key(edge) && visited.insert(*edge) {
                    queue.push_back(*edge);
                }
            }
            adjacency.insert(hash, edges);
        }

        // Restrict adjacency to the actual visited set — entries inherited
        // from prev that are no longer in hash_to_shard_offset (e.g. they
        // were in a shard that was later dropped) must not survive.
        adjacency.retain(|h, _| visited.contains(h) && hash_to_shard_offset.contains_key(h));

        let mut live_hashes: Vec<[u8; 16]> = visited
            .into_iter()
            .filter(|h| hash_to_shard_offset.contains_key(h))
            .collect();
        live_hashes.sort_unstable();
        Ok((live_hashes, adjacency))
    }

    /// Incremental full-adjacency scan (no-roots / preserve-everything path).
    ///
    /// For each hash in `hash_to_shard_offset`, reuses the cached edge list
    /// from `prev.adjacency` when available — avoiding a disk read for any
    /// node that survived the last repack. Only genuinely new hashes (not in
    /// the cache) require a disk read.
    ///
    /// All hashes in `hash_to_shard_offset` are live (no GC in this path),
    /// so the result is always the full map.
    ///
    /// **Memory:** same as [`Self::bfs_live_set_incremental`] — see that doc.
    fn scan_full_adjacency_incremental(
        hash_to_shard_offset: &HashMap<[u8; 16], (u16, u64)>,
        prev: &RepackIncrementalState,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<AdjacencyResult, StorageError> {
        let mut all_hashes: Vec<[u8; 16]> = hash_to_shard_offset.keys().copied().collect();
        all_hashes.sort_unstable();
        let mut adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> =
            HashMap::with_capacity(hash_to_shard_offset.len());

        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            // Fast path: node was live last time, edge list is cached.
            if let Some(edges) = prev.adjacency.get(hash) {
                adjacency.insert(*hash, edges.clone());
                continue;
            }
            // Slow path: new node, must read from disk.
            if let Some(shard) = pinned.get(&shard_id) {
                let record = Self::read_at(shard, offset, true)?;
                let edges = extract_edges(hash, &record.data);
                adjacency.insert(*hash, edges);
            }
        }
        Ok((all_hashes, adjacency))
    }

    /// Saves the cursor state a future call to [`Self::repack_scan_incremental`]
    /// needs: each shard's current byte length (the next repack's scan start
    /// point), the deduped live map restricted to hashes that survived into
    /// `new_offsets`, the adjacency cache for those survivors (so the next
    /// repack's BFS/scan can skip re-reading disk for already-seen nodes), and
    /// the roots snapshot (so the next repack can detect whether any root was
    /// removed and fall back to a full BFS sweep if so).
    ///
    /// Anything this repack dropped must not resurface via the carried-forward
    /// `live_map` or adjacency cache next time.
    fn repack_save_incremental_state(
        &self,
        collection_id: &[u8; 16],
        new_offsets: &[([u8; 16], u16, u64)],
        adjacency: HashMap<[u8; 16], Vec<[u8; 16]>>,
        current_roots: &[[u8; 16]],
    ) {
        let surviving: HashSet<[u8; 16]> = new_offsets.iter().map(|&(h, _, _)| h).collect();

        let mut scan_offsets = HashMap::new();
        for (shard_id, shard) in self.shards.all_shards() {
            scan_offsets.insert(shard_id, (shard.pack_id, shard.file_len()));
        }
        let next_live_map: HashMap<[u8; 16], (u16, u64)> = new_offsets
            .iter()
            .map(|&(hash, shard_id, offset)| (hash, (shard_id, offset)))
            .collect();
        // Evict dropped nodes from the adjacency cache so they can't
        // re-enter the live set on a future incremental BFS pass.
        let next_adjacency: HashMap<[u8; 16], Vec<[u8; 16]>> = adjacency
            .into_iter()
            .filter(|(h, _)| surviving.contains(h))
            .collect();
        self.repack_incremental.write().insert(
            *collection_id,
            RepackIncrementalState {
                scan_offsets,
                live_map: next_live_map,
                adjacency: next_adjacency,
                prev_roots: current_roots.to_vec(),
            },
        );
    }

    /// Non-mutating preview of what [`Self::repack_collection_reachable`] would
    /// do for `collection_id`: runs the exact same scan + reachability
    /// computation (so kept/dropped counts are exact, not estimates), but
    /// performs no writes, no index swap, and no incremental-repack
    /// cursor update. Used to preview a repack's effect before committing
    /// to it — e.g. a shard-compaction preflight that needs "how many
    /// bytes would survive, and which shards do they currently live in"
    /// without actually rewriting anything.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    pub fn plan_collection_repack(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        // `repack_scan_incremental` moves the cursor out to avoid cloning on
        // real repacks.  A plan must leave that state untouched, and must not
        // race a real repack while temporarily doing so.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();
        let saved_state = self.repack_incremental.read().get(collection_id).cloned();
        let result = self
            .repack_scan_incremental(collection_id)
            .and_then(|(map, _)| {
                self.plan_collection_repack_from_map(collection_id, &map, &extract_edges)
            });
        let mut states = self.repack_incremental.write();
        if let Some(state) = saved_state {
            states.insert(*collection_id, state);
        } else {
            states.remove(collection_id);
        }
        result
    }

    fn plan_collection_repack_from_map(
        &self,
        collection_id: &[u8; 16],
        hash_to_shard_offset: &RepackRecordMap,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<RepackPlan, StorageError> {
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, _adjacency) = match roots {
            Some(roots) if !roots.is_empty() => {
                Self::bfs_live_set(&roots, hash_to_shard_offset, &pinned, &extract_edges)?
            }
            _ => Self::scan_full_adjacency(hash_to_shard_offset, &pinned, &extract_edges)?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
        let mut kept_bytes: u64 = 0;
        let mut dropped_bytes: u64 = 0;
        let mut shards_touched: HashSet<u16> = HashSet::new();
        for (hash, &(shard_id, offset)) in hash_to_shard_offset {
            if let Some(shard) = pinned.get(&shard_id) {
                // Actual on-disk bytes, not the uncompressed upper bound —
                // a repack preflight should report what will really be
                // reclaimed/kept, which is smaller than plaintext size for
                // any frame that compressed.
                if let Ok(bytes) = Self::record_disk_len_at(shard, offset) {
                    if live_set.contains(hash) {
                        shards_touched.insert(shard_id);
                        kept_bytes = kept_bytes.saturating_add(bytes);
                    } else {
                        dropped_bytes = dropped_bytes.saturating_add(bytes);
                    }
                }
            }
        }

        let mut shards_touched: Vec<u16> = shards_touched.into_iter().collect();
        shards_touched.sort_unstable();
        Ok(RepackPlan {
            kept: live_hashes.len(),
            dropped,
            kept_bytes,
            dropped_bytes,
            shards_touched,
        })
    }

    /// Plan several collection repacks from one physical scan of their shard
    /// closure. Unlike repeatedly calling [`Self::plan_collection_repack`], this
    /// is O(bytes scanned) rather than O(collections × bytes scanned).
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption while scanning or
    /// resolving a requested collection's records.
    pub fn plan_collections_repack(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<RepackPlan>, StorageError> {
        let maps = self.scan_collection_record_maps(collection_ids)?;
        collection_ids
            .iter()
            .map(|collection_id| {
                let map = maps.get(collection_id).ok_or_else(|| {
                    StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
                })?;
                self.plan_collection_repack_from_map(collection_id, map, &extract_edges)
            })
            .collect()
    }

    /// Finds every collection and shard transitively reachable from
    /// `start_shard` via the collection↔shard reference relation: collections
    /// referencing this shard, the other shards those collections reference,
    /// the collections referencing *those* shards, and so on until nothing new
    /// is found. In practice this converges immediately in almost every
    /// case — a collection typically has a live footprint on only one or two
    /// shards (sticky home routing) — but the closure has to be computed
    /// rather than assumed, since a collection-scoped repack can't correctly
    /// judge liveness by looking at only one shard's slice of that collection's
    /// data (see [`Self::collections_referencing_shard`]'s doc).
    ///
    /// This is the basis for a shard-compaction preflight: you can't
    /// truthfully say "this touches N collections across K shards" without
    /// first finding the full closure, not just the one shard the
    /// operation was originally pointed at.
    ///
    /// # Errors
    /// Returns `StorageError` if `start_shard` isn't an open shard.
    pub fn repack_closure(
        &self,
        start_shard: u16,
    ) -> Result<(Vec<[u8; 16]>, Vec<u16>), StorageError> {
        let mut collections: HashSet<[u8; 16]> = HashSet::new();
        let mut shards: HashSet<u16> = HashSet::new();
        let mut shard_queue = vec![start_shard];
        let mut collection_queue: Vec<[u8; 16]> = Vec::new();

        loop {
            if let Some(shard_id) = shard_queue.pop() {
                if shards.insert(shard_id) {
                    for collection_id in self.collections_referencing_shard(shard_id)? {
                        if collections.insert(collection_id) {
                            collection_queue.push(collection_id);
                        }
                    }
                }
            } else if let Some(collection_id) = collection_queue.pop() {
                for shard_id in self.collection_referenced_shards(&collection_id) {
                    if !shards.contains(&shard_id) {
                        shard_queue.push(shard_id);
                    }
                }
            } else {
                break;
            }
        }

        let mut collections: Vec<[u8; 16]> = collections.into_iter().collect();
        collections.sort_unstable();
        let mut shards: Vec<u16> = shards.into_iter().collect();
        shards.sort_unstable();
        Ok((collections, shards))
    }

    /// Every shard id `collection_id`'s current index has at least one live
    /// entry pointing into. Empty if the collection doesn't exist.
    #[must_use]
    pub fn collection_referenced_shards(&self, collection_id: &[u8; 16]) -> Vec<u16> {
        let Some(gen) = self.generation(collection_id) else {
            return Vec::new();
        };
        gen.index
            .referenced_shard_ids()
            .iter()
            .enumerate()
            .filter_map(|(i, &referenced)| referenced.then(|| u16::try_from(i).ok()).flatten())
            .collect()
    }

    /// Like `collection_referenced_shards`, but returns the `pack_id`
    /// for each shard rather than the runtime slot index.
    pub fn collection_referenced_pack_ids(&self, collection_id: &[u8; 16]) -> Vec<u64> {
        self.collection_referenced_shards(collection_id)
            .iter()
            .filter_map(|&slot| self.shards.get_shard(slot).map(|s| s.pack_id))
            .collect()
    }

    /// Current indexed-node count for every shard.
    ///
    /// Unlike a physical packfile scan, these counts include only index
    /// entries that are presently live and resolve through a collection's current
    /// generation. Rewritten or superseded append entries are excluded.
    #[must_use]
    pub fn shard_node_counts(&self) -> HashMap<u64, u64> {
        self.shard_collections
            .read()
            .iter()
            .map(|(&shard_id, collections)| {
                let count = collections
                    .values()
                    .copied()
                    .fold(0u64, u64::saturating_add);
                (shard_id, count)
            })
            .collect()
    }

    /// # Errors
    /// Returns `StorageError` on I/O or corruption.
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    // Four dispatch arms (roots/no-roots × incremental/cold-start) plus root-removal
    // detection make this function long; splitting it would just move the complexity
    // without reducing it.
    #[allow(clippy::too_many_lines)]
    pub fn repack_collection_reachable(
        &self,
        collection_id: &[u8; 16],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<(usize, usize), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        let (hash_to_shard_offset, prev_state) = self.repack_scan_incremental(collection_id)?;

        // Pin every shard this call could possibly read from before any
        // writes happen — see pin_shards' doc for why this must come
        // first.
        let pinned = self.pin_shards(hash_to_shard_offset.values().map(|&(id, _)| id));

        let roots = self.live_roots.read().get(collection_id).cloned();

        // Root-removal detection: if any root present in the previous repack's
        // snapshot is absent from the current root set, nodes exclusively
        // reachable through that removed root may be stuck in the cached live
        // set indefinitely (the incremental adjacency cache only grows — it has
        // no mechanism to re-derive reachability). Fall back to a full cold-start
        // BFS for this repack so those nodes can actually be collected. Once the
        // full sweep completes, the saved cursor is fresh and subsequent repacks
        // can be incremental again.
        let roots_shrunk = prev_state.as_ref().is_some_and(|prev| {
            let current_root_set: HashSet<[u8; 16]> =
                roots.as_deref().unwrap_or(&[]).iter().copied().collect();
            prev.prev_roots
                .iter()
                .any(|r| !current_root_set.contains(r))
        });
        // Use incremental adjacency only when prev state exists AND no root was removed.
        let use_incremental = prev_state.is_some() && !roots_shrunk;

        let (live_hashes, adjacency) = match (roots.as_deref(), use_incremental) {
            (Some(roots), true) if !roots.is_empty() => {
                // Incremental BFS: only expand from roots not already in the
                // cached live set; reuse cached edge lists for everything else.
                Self::bfs_live_set_incremental(
                    roots,
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            (Some(roots), false) if !roots.is_empty() => {
                // Cold-start BFS: root set shrank or first repack — full walk.
                Self::bfs_live_set(roots, &hash_to_shard_offset, &pinned, &extract_edges)?
            }
            (_, true) => {
                // No live roots, incremental path: only read disk for new nodes.
                Self::scan_full_adjacency_incremental(
                    &hash_to_shard_offset,
                    prev_state.as_ref().expect("use_incremental implies Some"),
                    &pinned,
                    &extract_edges,
                )?
            }
            _ => {
                // No live roots, cold-start: preserve everything, read all.
                Self::scan_full_adjacency(&hash_to_shard_offset, &pinned, &extract_edges)?
            }
        };

        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        // Evict garbage-collected hashes from the collection's cache. Repack
        // carries the same cache forward into the new generation (below);
        // without this, a stale cache hit could still serve bytes for a
        // hash this pass just decided is unreachable, defeating GC.
        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        let mut hex = String::with_capacity(32);
                        for b in hash {
                            let _ = write!(hex, "{b:02x}");
                        }
                        eprintln!("  dropped: {hex}");
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "repack: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        // Copy into a non-source destination shard, never back into the
        // collection's current generation. Once the new index is published below,
        // this lets old source shards retire when no other collection references
        // them.
        let source_shards: HashSet<u16> = hash_to_shard_offset
            .values()
            .map(|&(shard_id, _)| shard_id)
            .collect();
        self.shards
            .prepare_collection_repack(collection_id, &source_shards)?;

        let mut new_offsets: Vec<([u8; 16], u16, u64)> = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some(entry) =
                self.copy_record_to_shard(collection_id, &pinned, old_shard_id, old_offset)?
            {
                new_offsets.push(entry);
            }
        }

        let kept = new_offsets.len();

        let index = Self::build_index(&new_offsets)?;
        self.replace_collection_shard_counts(
            collection_id,
            &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
        );
        self.swap_generation(collection_id, index)?;
        self.retire_empty_shards(collection_id);

        self.repack_count.fetch_add(1, Ordering::Relaxed);
        self.repack_kept_total
            .fetch_add(kept as u64, Ordering::Relaxed);
        self.repack_dropped_total
            .fetch_add(dropped as u64, Ordering::Relaxed);
        self.repack_counts_by_collection
            .write()
            .entry(*collection_id)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);

        let current_roots: Vec<[u8; 16]> = roots.as_deref().unwrap_or(&[]).to_vec();
        self.repack_save_incremental_state(collection_id, &new_offsets, adjacency, &current_roots);

        // Fsync the shards this repack just wrote into and persist the
        // updated stats snapshot as part of finishing the repack, rather
        // than leaving both to whatever the next unrelated sync_dirty()
        // call happens to be — a crash right after a repack should not
        // lose durability for data this repack itself just wrote, nor
        // leave the persisted stats stale relative to what's on disk.
        self.shards.sync_dirty()?;

        Ok((kept, dropped))
    }

    /// Repack several collections as one physical shard-compaction batch.
    ///
    /// The input is scanned once, then all replacement records are written
    /// through one shared destination stream. This is O(bytes + nodes), not
    /// O(collections × bytes), and avoids leaving one partially-filled destination
    /// shard behind for each collection.
    ///
    /// # Errors
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        self.repack_collections_reachable_with_progress(
            collection_ids,
            extract_edges,
            |_collection, _from, _to, _nodes, _complete| {},
        )
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// This lets an interactive caller show meaningful physical-compaction
    /// progress without turning the shared shard batch back into a
    /// collection-at-a-time operation.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    fn report_repack_output_rotation(
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        active_output_shard: &mut u16,
        current_active_shard: u16,
        collection_id: &[u8; 16],
        copied_nodes: usize,
    ) {
        if current_active_shard != *active_output_shard {
            output_progress(
                Some(*collection_id),
                *active_output_shard,
                current_active_shard,
                copied_nodes,
                false,
            );
            *active_output_shard = current_active_shard;
        }
    }

    fn copy_repack_batch_collection(
        &self,
        collection_id: &[u8; 16],
        hash_to_shard_offset: &RepackRecordMap,
        pinned: &HashMap<u16, Arc<Shard>>,
        extract_edges: &impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        output_progress: &mut impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
        output_state: &mut (u16, usize),
    ) -> Result<(RepackOffsets, usize), StorageError> {
        let roots = self.live_roots.read().get(collection_id).cloned();
        let (live_hashes, adjacency) = match roots {
            Some(roots) if !roots.is_empty() => {
                Self::bfs_live_set(&roots, hash_to_shard_offset, pinned, extract_edges)?
            }
            _ => Self::scan_full_adjacency(hash_to_shard_offset, pinned, extract_edges)?,
        };
        let dropped = hash_to_shard_offset.len().saturating_sub(live_hashes.len());

        if dropped > 0 {
            let live_set: HashSet<[u8; 16]> = live_hashes.iter().copied().collect();
            if let Some(gen) = self.generation(collection_id) {
                for hash in hash_to_shard_offset.keys() {
                    if !live_set.contains(hash) {
                        gen.cache.remove(hash);
                    }
                }
            }
        }

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let topo = csr.topo_order();
        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "repack: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        let mut new_offsets = Vec::with_capacity(topo.len());
        for &local in &topo {
            let hash = csr
                .hash_of(local)
                .expect("topo order contains valid local IDs");
            let Some(&(old_shard_id, old_offset)) = hash_to_shard_offset.get(hash) else {
                continue;
            };
            if let Some(entry) =
                self.copy_record_to_shard(collection_id, pinned, old_shard_id, old_offset)?
            {
                output_state.1 = output_state.1.saturating_add(1);
                let current_active_shard = self.shards.active_shard().slot;
                Self::report_repack_output_rotation(
                    output_progress,
                    &mut output_state.0,
                    current_active_shard,
                    collection_id,
                    output_state.1,
                );
                new_offsets.push(entry);
            }
        }

        Ok((new_offsets, dropped))
    }

    /// As [`Self::repack_collections_reachable`], with a callback whenever the
    /// shared output stream rotates. The callback receives the collection whose
    /// copy crossed the boundary (or `None` at completion), the slot it
    /// rotated from, the replacement slot, the cumulative number of copied
    /// nodes, and whether the copy is complete.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] on I/O, corruption, or an unavailable staging
    /// shard slot.
    ///
    /// # Panics
    ///
    /// Panics if the CSR implementation returns a local ID not present in its
    /// own hash table, which would violate its internal ordering invariant.
    pub fn repack_collections_reachable_with_progress(
        &self,
        collection_ids: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        mut output_progress: impl FnMut(Option<[u8; 16]>, u16, u16, usize, bool),
    ) -> Result<Vec<([u8; 16], usize, usize)>, StorageError> {
        let mut collection_ids = collection_ids.to_vec();
        collection_ids.sort_unstable();
        collection_ids.dedup();
        if collection_ids.is_empty() {
            return Ok(Vec::new());
        }

        // A batch is one coherent generation change. Hold every selected
        // collection's writer mutex in a stable order so a concurrent put cannot
        // be missed by the scan or revive an old source after the swap.
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let collection_guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();

        let maps = self.scan_collection_record_maps(&collection_ids)?;
        let source_shards: HashSet<u16> = maps
            .values()
            .flat_map(|map| map.values().map(|&(shard_id, _)| shard_id))
            .collect();
        let pinned = self.pin_shards(source_shards.iter().copied());

        // Establish one common destination before copying the first frame.
        // `put_record` moves every collection to the same active successor when a
        // destination fills, so output remains densely packed across collections.
        self.shards
            .prepare_collections_repack(&collection_ids, &source_shards)?;

        let mut results = Vec::with_capacity(collection_ids.len());
        let mut output_state = (self.shards.active_shard().slot, 0usize);
        for collection_id in &collection_ids {
            let hash_to_shard_offset = maps.get(collection_id).ok_or_else(|| {
                StorageError::Corrupt("repack batch scan lost requested collection".to_owned())
            })?;
            let (new_offsets, dropped) = self.copy_repack_batch_collection(
                collection_id,
                hash_to_shard_offset,
                &pinned,
                &extract_edges,
                &mut output_progress,
                &mut output_state,
            )?;
            let kept = new_offsets.len();
            let index = Self::build_index(&new_offsets)?;
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
            );
            self.swap_generation(collection_id, index)?;
            // The batch path does a full cold-start scan (scan_collection_record_maps),
            // so we have no incremental adjacency to carry forward. Pass an empty
            // map so the next single-collection repack starts with a fresh cache
            // (it will populate it from the full BFS on that first pass).
            let batch_roots: Vec<[u8; 16]> = self
                .live_roots
                .read()
                .get(collection_id)
                .cloned()
                .unwrap_or_default();
            self.repack_save_incremental_state(
                collection_id,
                &new_offsets,
                HashMap::new(),
                &batch_roots,
            );

            self.repack_count.fetch_add(1, Ordering::Relaxed);
            self.repack_kept_total
                .fetch_add(kept as u64, Ordering::Relaxed);
            self.repack_dropped_total
                .fetch_add(dropped as u64, Ordering::Relaxed);
            self.repack_counts_by_collection
                .write()
                .entry(*collection_id)
                .and_modify(|count| *count = count.saturating_add(1))
                .or_insert(1);
            results.push((*collection_id, kept, dropped));
        }

        output_progress(None, output_state.0, output_state.0, output_state.1, true);

        // All target collection locks remain held while source references are
        // replaced. Drop them before the general retirement protocol, which
        // obtains every collection lock itself.
        drop(collection_guards);
        self.retire_empty_shards_after_batch();
        self.shards.sync_dirty()?;
        Ok(results)
    }

    /// Fetches the ancestors of `frontier`, walking `extract_edges`
    /// backward and stopping expansion at (and excluding) anything in
    /// `stop_at` — a bounded ancestor walk, e.g. for auth-chain or
    /// prev-event traversal from a caller-supplied frontier back to
    /// caller-supplied already-known boundary nodes.
    ///
    /// This is **not** a from/to range or span: if a branch's history
    /// never crosses any `stop_at` node (a fork off the main line, say),
    /// that branch is walked all the way back to the collection's true roots,
    /// not truncated at an implied boundary. A genuine "everything between
    /// these two frontiers" query is `ancestors(frontier) ∩
    /// descendants(stop_at)`, which needs reverse adjacency this doesn't
    /// build — `NodeId`s are content hashes with no key order to slice a
    /// real range query over, and this walk only covers the ancestor half.
    /// `limits.max_nodes` bounds the runaway case above.
    ///
    /// Returns nodes lazily, ancestor-first (parents before children,
    /// topologically), as a [`DagWalk`] iterator over data already
    /// resolved during the walk itself (no second per-node fetch against
    /// whatever generation happens to be live when the iterator is
    /// drained) — every record was matched by content hash, not just
    /// index tag, so an index-tag collision can't surface the wrong one.
    ///
    /// Read-only: resolves against one frozen generation snapshot via its
    /// internal breadth-first traversal and takes no `put_mutex`. Cost is
    /// proportional to the number of ancestors actually walked — one
    /// index lookup per node — not collection size, since it doesn't pre-scan
    /// every shard the way `repack_collection_reachable` does.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O or corruption, or if the walk finds a
    /// cycle (an event graph must be acyclic; a cycle means the extracted
    /// edges are wrong or the data is corrupt).
    ///
    /// # Panics
    /// Panics if any hash in the CSR exceeds `u32::MAX` local ID space.
    pub fn walk_ancestors(
        &self,
        collection_id: &[u8; 16],
        frontier: &[[u8; 16]],
        stop_at: &[[u8; 16]],
        extract_edges: impl Fn(&[u8; 16], &[u8]) -> Vec<[u8; 16]>,
        limits: WalkLimits,
    ) -> Result<DagWalk, StorageError> {
        let Some(gen) = self.generation(collection_id).map(|g| Arc::clone(&g)) else {
            return Ok(DagWalk {
                order: Vec::new().into_iter(),
                resolved: HashMap::new(),
            });
        };

        let ((live_hashes, adjacency), resolved) =
            self.bfs_ancestors(&gen, frontier, stop_at, &extract_edges, limits)?;

        let csr = Csr::build_from_edges(&live_hashes, &adjacency);
        let mut topo = csr.topo_order();

        if topo.len() != live_hashes.len() {
            return Err(StorageError::Corrupt(format!(
                "walk_ancestors: cyclic graph detected — topo_order produced {} nodes from {} live hashes",
                topo.len(),
                live_hashes.len(),
            )));
        }

        // `extract_edges` points each node at its parents, so Kahn's
        // algorithm (which surfaces zero-incoming nodes first) naturally
        // yields children before parents. Reverse it: callers of an
        // ancestor walk want parents-before-children, the order data
        // could actually be replayed in.
        topo.reverse();

        let order: Vec<[u8; 16]> = topo
            .into_iter()
            .map(|local| {
                *csr.hash_of(local)
                    .expect("topo order contains valid local IDs")
            })
            .collect();

        Ok(DagWalk {
            order: order.into_iter(),
            resolved,
        })
    }

    /// After a repack, scan all collections' indexes and retire any shard that
    /// no collection references.
    ///
    /// Acquires every collection's `put_mutex` (in sorted order, skipping the
    /// caller's already-held lock) to prevent a concurrent `put()` from
    /// landing a write in a shard whose index entry hasn't been inserted
    /// yet — without that, the scan could see zero references to a shard
    /// that a writer just committed bytes to but hasn't index-updated yet,
    /// causing a live shard to be retired under it.
    fn retire_empty_shards(&self, held_collection: &[u8; 16]) {
        // Collect collection IDs in sorted order for deadlock-free lock acquisition.
        // Skip the collection whose put_mutex the caller already holds.
        let mut collections_to_lock: Vec<[u8; 16]> = self
            .collections
            .read()
            .keys()
            .filter(|id| *id != held_collection)
            .copied()
            .collect();
        collections_to_lock.sort_unstable();

        // Acquire all other collections' put_mutexes. The sorted order prevents
        // deadlocks; the held collection is skipped (parking_lot is non-reentrant).
        let mutexes: Vec<_> = collections_to_lock
            .iter()
            .map(|id| self.put_mutex(id))
            .collect();
        let _guards: Vec<_> = mutexes.iter().map(|m| m.lock()).collect();

        self.retire_empty_shards_locked();
    }

    /// Retire unreferenced shards after a batch has released all collection locks.
    /// This takes every collection lock itself, unlike [`Self::retire_empty_shards`]
    /// which is called while one particular collection lock is already held.
    fn retire_empty_shards_after_batch(&self) {
        let mut collection_ids: Vec<[u8; 16]> = self.collections.read().keys().copied().collect();
        collection_ids.sort_unstable();
        let mutexes: Vec<_> = collection_ids.iter().map(|id| self.put_mutex(id)).collect();
        let _guards: Vec<_> = mutexes.iter().map(|mutex| mutex.lock()).collect();
        self.retire_empty_shards_locked();
    }

    /// Every collection writer lock is held by the caller.
    fn retire_empty_shards_locked(&self) {
        // Do not retain the map guard while acquiring collection locks: another
        // operation may need the map lock while it holds a collection lock.
        let collections = self.collections.read();

        // Build the union of shard IDs referenced across all collections.
        let mut referenced = [false; shard::MAX_SHARDS];
        for gen_swap in collections.values() {
            let gen = gen_swap.load();
            let ids = gen.index.referenced_shard_ids();
            for (i, &has_refs) in ids.iter().enumerate() {
                referenced[i] |= has_refs;
            }
        }

        // Retire any shard not in the union.
        for id in 0..u16::try_from(shard::MAX_SHARDS).unwrap() {
            if !referenced[id as usize] {
                self.shards.retire_slot(id);
            }
        }
    }
}

impl StorageEngine for PackfileStorage {
    fn get(&self, collection_id: &[u8; 16], id: &NodeId) -> Result<Option<NodeData>, StorageError> {
        let gen_guard = self.generation(collection_id);
        let gen = gen_guard.as_deref();

        let Some(gen) = gen else {
            return Ok(None);
        };
        if let Some(data) = gen.cache.get(id) {
            return Ok(Some((*data).clone()));
        }
        self.resolve_from_candidates(id, gen.index.lookup_all(id))
    }

    fn get_many(
        &self,
        collection_id: &[u8; 16],
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeData>>, StorageError> {
        let mut results: Vec<Option<NodeData>> = vec![None; ids.len()];

        let gen_guard = self.generation(collection_id);
        let gen = gen_guard.as_deref();

        let mut to_fetch: Vec<(usize, Vec<(u16, u64)>)> = Vec::new();
        if let Some(g) = gen {
            for (i, id) in ids.iter().enumerate() {
                // The generation-local cache only contains records from this
                // generation, so it is authoritative for hits. Avoid an
                // otherwise redundant index probe for the common warm case.
                if let Some(data) = g.cache.get(id) {
                    results[i] = Some((*data).clone());
                    continue;
                }
                let candidates: Vec<(u16, u64)> = g.index.lookup_all(id).collect();
                if !candidates.is_empty() {
                    to_fetch.push((i, candidates));
                }
            }
        }

        to_fetch.sort_unstable_by_key(|(_, candidates)| candidates[0]);

        for (i, candidates) in &to_fetch {
            results[*i] = self.resolve_from_candidates(&ids[*i], candidates.iter().copied())?;
        }

        Ok(results)
    }

    fn put(
        &self,
        collection_id: &[u8; 16],
        id: &NodeId,
        data: &NodeData,
    ) -> Result<(), StorageError> {
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        let record = Record {
            collection_id: *collection_id,
            hash: *id,
            data: data.bytes.clone(),
        };
        let (shard_id, offset) = self.shards.put_record(&record)?;
        let (index, cache) = {
            let old_gen = self.generation(collection_id);
            let mut index = match &old_gen {
                Some(g) => g.index.clone(),
                None => LossyIndex::new(4096),
            };
            let index_full = index.insert(id, shard_id, offset).is_err();
            if index_full {
                if index.grow() {
                    // The failed insert did not mutate the table, so retry it
                    // after the in-memory rehash. This is the normal capacity
                    // path and must not turn into a full-pack scan.
                    let _ = index.insert(id, shard_id, offset);
                } else {
                    index = self.rebuild_index(collection_id)?;
                    let _ = index.insert(id, shard_id, offset);
                }
                // The rebuild re-derived the collection's entire live set from
                // scratch, so its shard distribution needs a full
                // recompute too, not just crediting this one record.
                self.replace_collection_shard_counts(
                    collection_id,
                    &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
                );
            } else {
                let pack_id = self
                    .shards
                    .get_shard(shard_id)
                    .map_or(u64::from(shard_id), |s| s.pack_id);
                self.record_new_shard_collection(pack_id, collection_id);
            }
            let cache = match &old_gen {
                Some(g) => g.cache.clone(),
                None => Arc::new(NodeCache::new(self.cache_capacity)),
            };

            let mut data_to_cache = data.clone();
            for child in &mut data_to_cache.children {
                if let NodeRef::Lazy(child_id) = child {
                    if let Some(child_data) = self.pinned.get(child_id) {
                        *child = NodeRef::Resolved(*child_id, child_data);
                    }
                }
            }
            cache.insert(*id, Arc::new(data_to_cache));

            (index, cache)
        };

        self.store_generation(collection_id, index, Some(cache))?;

        Ok(())
    }

    fn put_many(
        &self,
        collection_id: &[u8; 16],
        entries: &[(NodeId, NodeData)],
    ) -> Result<(), StorageError> {
        if entries.is_empty() {
            return Ok(());
        }

        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        let old_gen = self.generation(collection_id);
        let mut index = match &old_gen {
            Some(g) => g.index.clone(),
            None => LossyIndex::new(4096),
        };
        let cache = match &old_gen {
            Some(g) => g.cache.clone(),
            None => Arc::new(NodeCache::new(self.cache_capacity)),
        };

        let mut index_needs_rebuild = false;

        for (id, data) in entries {
            let record = Record {
                collection_id: *collection_id,
                hash: *id,
                data: data.bytes.clone(),
            };

            let (shard_id, offset) = self.shards.put_record(&record)?;

            if !index_needs_rebuild {
                let inserted = if index.insert(id, shard_id, offset).is_ok() {
                    true
                } else if index.grow() {
                    // `insert` leaves the table unchanged on TableFull, so
                    // retrying with the record's still-available location is
                    // sufficient; no pack scan is needed for pure growth.
                    index.insert(id, shard_id, offset).is_ok()
                } else {
                    index_needs_rebuild = true;
                    false
                };
                if inserted {
                    let pack_id = self
                        .shards
                        .get_shard(shard_id)
                        .map_or(u64::from(shard_id), |s| s.pack_id);
                    self.record_new_shard_collection(pack_id, collection_id);
                }
            }
        }

        if index_needs_rebuild {
            index = self.rebuild_index(collection_id)?;
            // rebuild_index automatically discovers all the records we just appended
            self.replace_collection_shard_counts(
                collection_id,
                &self.slot_counts_to_pack_id_counts(&index.shard_counts()),
            );
        }

        // Apply cache mutations only after all disk writes succeed, so a
        // failed batch does not leak partial state into the shared cache.
        // Resolve and insert one entry at a time: retaining a prepared clone
        // of the entire batch here would defeat the cache's size bound.
        for (id, data) in entries {
            let mut data_to_cache = data.clone();
            for child in &mut data_to_cache.children {
                if let NodeRef::Lazy(child_id) = child {
                    if let Some(child_data) = self.pinned.get(child_id) {
                        *child = NodeRef::Resolved(*child_id, child_data);
                    }
                }
            }
            cache.insert(*id, Arc::new(data_to_cache));
        }

        self.store_generation(collection_id, index, Some(cache))?;

        Ok(())
    }

    fn delete_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        // Acquire the collection's put mutex to serialize with any in-flight put,
        // preventing a concurrent put from resurrecting the collection after we
        // remove it from the generation map.
        let collection_arc = self.put_mutex(collection_id);
        let _collection_guard = collection_arc.lock();

        self.collections.write().remove(collection_id);
        self.live_roots.write().remove(collection_id);
        // Keep lock entries for the storage lifetime. Removing an entry while
        // a caller still owns its Arc permits a later put to obtain a second
        // mutex and bypass this deletion's serialization.
        self.remove_collection_shard_counts(collection_id);
        self.persist_deleted_collection(collection_id)?;
        Ok(())
    }

    fn sync(&self) -> Result<(), StorageError> {
        self.shards.sync_dirty()?;
        self.persist_index_checkpoint_best_effort();
        Ok(())
    }

    fn refresh_collection(&self, collection_id: &[u8; 16]) -> Result<(), StorageError> {
        Self::refresh_collection(self, collection_id)
    }
}

impl PackfileStorage {
    /// Sync all open shards to disk (full pool, not just dirty).
    ///
    /// Also persists the shard→collection directory as a side effect, same as
    /// `ShardPool::sync_all` persists shard IO stats — an explicit sync is
    /// a natural point to flush this observability data too.
    ///
    /// # Errors
    /// Returns `StorageError` on I/O failure.
    pub fn sync_all(&self) -> Result<(), StorageError> {
        self.shards.sync_all()?;
        self.persist_shard_collections_best_effort();
        self.persist_index_checkpoint_best_effort();
        Ok(())
    }

    /// Best-effort, rate-limited flush of the shard→collection directory for a
    /// writer's own periodic tick — same contract as
    /// `ShardPool::maybe_persist_stats`: a no-op if called again before
    /// `min_interval` has passed since the last flush from here.
    pub fn maybe_persist_shard_collections(&self, min_interval: std::time::Duration) {
        let now = std::time::Instant::now();
        {
            let mut last = self.last_shard_collections_flush.write();
            if last.is_some_and(|prev| now.duration_since(prev) < min_interval) {
                return;
            }
            *last = Some(now);
        }
        self.persist_shard_collections_best_effort();
    }

    /// Snapshot IO/sync stats for every currently-open shard.
    ///
    /// Shards are shared across collections, so this is per-shard, not per-collection.
    #[must_use]
    pub fn shard_stats(&self) -> Vec<(u16, crate::shard::ShardStats)> {
        self.shards.all_stats()
    }

    /// Get IO/sync stats for a single shard by ID.
    #[must_use]
    pub fn shard_stats_for(&self, shard_id: u16) -> Option<crate::shard::ShardStats> {
        self.shards.stats(shard_id)
    }

    /// Total number of shards retired (garbage-collected after a repack)
    /// over this storage's lifetime.
    #[must_use]
    pub fn shards_retired(&self) -> u64 {
        self.shards.retired_count()
    }

    /// List every currently-open shard with basic size, generation, and
    /// IO/sync stats — the data behind a `shards` CLI listing.
    ///
    /// This needs no collection data at all; a caller that only wants shard-level
    /// info (e.g. a `shards`-only inspection tool that must coexist with a
    /// live writer) should call `ShardPool::summaries` directly instead of
    /// opening a full `PackfileStorage`, which rebuilds every collection's index.
    #[must_use]
    pub fn shard_summaries(&self) -> Vec<shard::ShardSummary> {
        self.shards.summaries()
    }

    /// Snapshot global repack stats across all collections.
    #[must_use]
    pub fn repack_stats(&self) -> RepackStats {
        RepackStats {
            repack_count: self.repack_count.load(Ordering::Relaxed),
            kept_total: self.repack_kept_total.load(Ordering::Relaxed),
            dropped_total: self.repack_dropped_total.load(Ordering::Relaxed),
        }
    }

    /// Number of times a specific collection has been repacked. 0 if it has
    /// never been repacked (or doesn't exist).
    #[must_use]
    pub fn repack_count_for_collection(&self, collection_id: &[u8; 16]) -> u64 {
        self.repack_counts_by_collection
            .read()
            .get(collection_id)
            .copied()
            .unwrap_or(0)
    }

    /// Cache hit/miss stats for a collection's decoded-node cache, if the collection
    /// currently has one loaded.
    #[must_use]
    pub fn cache_stats_for(&self, collection_id: &[u8; 16]) -> Option<CacheStats> {
        let gen = self.generation(collection_id)?;
        Some(CacheStats {
            hits: gen.cache.hits(),
            misses: gen.cache.misses(),
            hit_rate: gen.cache.hit_rate(),
        })
    }
}

/// Snapshot of global repack activity across all collections.
#[derive(Debug, Clone, Copy, Default)]
pub struct RepackStats {
    /// Total number of `repack_collection_reachable` calls across all collections.
    pub repack_count: u64,
    /// Total records kept (rewritten into a new generation) across all repacks.
    pub kept_total: u64,
    /// Total records dropped (found unreachable) across all repacks.
    pub dropped_total: u64,
}

/// Snapshot of a collection's decoded-node cache hit/miss stats.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub hit_rate: f64,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const TEST_COLLECTION: [u8; 16] = [0x01; 16];

    /// A node id distinct across both the bucket bytes (0..8) and tag bytes
    /// (8..12) the lossy index actually reads — an id that only varies byte
    /// 0 collapses every entry onto the same 24-bit tag, which is not a
    /// realistic content hash and defeats the index in ways unrelated to
    /// whatever a test using it is meant to check.
    fn distinct_id(byte: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[0] = byte;
        id[9] = byte.wrapping_mul(37).wrapping_add(11);
        id[15] = byte.wrapping_mul(7);
        id
    }
    const OTHER_COLLECTION: [u8; 16] = [0x02; 16];

    fn test_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mdb_test_pfs_{name}_{id}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ten_record_fixture() -> Vec<(NodeId, NodeData)> {
        (0..10u8)
            .map(|i| {
                let mut id = [0u8; 16];
                id[0] = i;
                id[8..12].copy_from_slice(&(u32::from(i) + 1).to_le_bytes());
                (id, NodeData::new(bytes::Bytes::from(format!("node {i}"))))
            })
            .collect()
    }

    #[test]
    fn test_put_and_get() {
        let dir = test_dir("putget");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x42u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"hello world"));

        store.put(&TEST_COLLECTION, &id, &data).unwrap();
        let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
        assert_eq!(got.bytes, data.bytes);
    }

    /// A writable open must not silently skip a corrupt pack and publish a
    /// partial index. The caller needs an error so it can repair or restore
    /// the pack before accepting writes.
    #[test]
    fn test_writable_open_rejects_corrupt_pack_instead_of_skipping_it() {
        let dir = test_dir("open_corrupt_pack");
        let store = PackfileStorage::open(dir.clone()).unwrap();
        let id = distinct_id(0x42);
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"payload")),
            )
            .unwrap();
        let path = store.shards.active_shard().path.clone();
        drop(store);

        let mut bytes = fs::read(&path).unwrap();
        let payload_byte =
            crate::packfile::HEADER_LEN + 4 + crate::packfile::FRAME_FIXED_LEN as usize;
        bytes[payload_byte] ^= 0xff;
        fs::write(path, bytes).unwrap();

        assert!(PackfileStorage::open(dir).is_err());
    }

    /// Full and `WriteOnly` keep writing real CRCs, so a reopen scan still
    /// verifies them: a physically tampered pack must fail to reopen.
    /// `Disabled` frames carry no checksum, so a tampered pack reopens and
    /// the engine serves the corrupted bytes as data.
    #[test]
    fn test_checksum_policy_gates_verification_on_reopen_scan() {
        for (policy, expect_reopen_ok) in [
            (crate::packfile::ChecksumPolicy::Full, false),
            (crate::packfile::ChecksumPolicy::WriteOnly, false),
            (crate::packfile::ChecksumPolicy::Disabled, true),
        ] {
            let dir = test_dir(&format!("checksum_policy_reopen_{policy:?}"));
            {
                let store = PackfileStorage::open_with_policies(dir.clone(), true, policy).unwrap();
                let id = distinct_id(0x52);
                store
                    .put(
                        &TEST_COLLECTION,
                        &id,
                        &NodeData::new(bytes::Bytes::from_static(b"tamper target")),
                    )
                    .unwrap();
                let (shard_id, offset) = store
                    .generation(&TEST_COLLECTION)
                    .unwrap()
                    .index
                    .lookup(&id)
                    .expect("just-written record must be indexed");
                let path = store.shards.get_shard(shard_id).unwrap().path.clone();
                drop(store);

                let mut bytes = fs::read(&path).unwrap();
                let payload_byte = usize::try_from(offset)
                    .expect("u64 offset fits usize on any supported host")
                    .saturating_add(4)
                    .saturating_add(crate::packfile::FRAME_FIXED_LEN as usize);
                assert!(payload_byte < bytes.len(), "tamper site must be in-file");
                bytes[payload_byte] ^= 0xff;
                bytes[payload_byte + 1] ^= 0x0f;
                fs::write(path, bytes).unwrap();
            }

            if expect_reopen_ok {
                let store = PackfileStorage::open_with_policies(dir, true, policy).unwrap();
                let got = store
                    .get(&TEST_COLLECTION, &distinct_id(0x52))
                    .unwrap()
                    .expect("Disabled reopen must serve the record");
                assert_eq!(
                    got.bytes.as_ref().len(),
                    b"tamper target".len(),
                    "payload frame must still decode"
                );
            } else {
                let reopened = PackfileStorage::open_with_policies(dir, true, policy);
                match reopened {
                    Ok(_) => panic!("reopen scan must reject a tampered pack"),
                    Err(err) => {
                        assert!(
                            matches!(err, std::io::Error { .. }),
                            "expected I/O error from scan_and_recover, got {err:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_read_survives_append_after_mmap_established() {
        let dir = test_dir("stale_mmap_regression");
        let store = PackfileStorage::open(dir).unwrap();

        let a = [0xAAu8; 16];
        let b = [0xBBu8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"aaaa"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"bbbb"));

        store.put(&TEST_COLLECTION, &a, &data_a).unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        let got_a = store.get(&TEST_COLLECTION, &a).unwrap();
        assert_eq!(got_a.unwrap().bytes, data_a.bytes);

        store.put(&TEST_COLLECTION, &b, &data_b).unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        let got_b = store.get(&TEST_COLLECTION, &b).unwrap();
        assert_eq!(got_b.expect("B must be found").bytes, data_b.bytes);
    }

    #[test]
    fn test_get_not_found() {
        let dir = test_dir("notfound");
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&TEST_COLLECTION, &[0x00; 16]).unwrap().is_none());
    }

    #[test]
    fn test_cache_hit() {
        let dir = test_dir("cachehit");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"cached"));

        store.put(&TEST_COLLECTION, &id, &data).unwrap();

        let _ = store.get(&TEST_COLLECTION, &id).unwrap();
        let gen = store.generation(&TEST_COLLECTION).unwrap();
        assert_eq!(gen.cache.hits(), 1);

        let _ = store.get(&TEST_COLLECTION, &id).unwrap();
        assert_eq!(gen.cache.hits(), 2);
    }

    #[test]
    fn test_cold_get_does_not_populate_lru() {
        let dir = test_dir("cold_get_no_lru_insert");
        let store = PackfileStorage::open(dir).unwrap();
        let id = [0x01u8; 16];
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"mmap-backed")),
            )
            .unwrap();

        let gen = store.generation(&TEST_COLLECTION).unwrap();
        gen.cache.clear();
        assert!(store.get(&TEST_COLLECTION, &id).unwrap().is_some());
        assert_eq!(gen.cache.len(), 0);
    }

    #[test]
    fn test_delete_collection() {
        let dir = test_dir("delete");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

        store.put(&OTHER_COLLECTION, &id, &data).unwrap();

        store.delete_collection(&OTHER_COLLECTION).unwrap();
        assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
        assert!(store.generation(&OTHER_COLLECTION).is_none());
    }

    #[test]
    fn test_delete_collection_does_not_resurrect_on_reopen() {
        let dir = test_dir("delete_no_resurrect");

        let id = [0x01u8; 16];
        let data = NodeData::new(bytes::Bytes::from_static(b"collection data"));

        {
            let store = PackfileStorage::open(dir.clone()).unwrap();
            store.put(&OTHER_COLLECTION, &id, &data).unwrap();
            store.delete_collection(&OTHER_COLLECTION).unwrap();
            store.sync_all().unwrap();
        }

        // Reopen from scratch: the on-disk deleted.collections marker must make
        // the startup scan skip OTHER_COLLECTION's leftover packfile records,
        // rather than resurrecting them into a fresh index.
        let store = PackfileStorage::open(dir).unwrap();
        assert!(store.get(&OTHER_COLLECTION, &id).unwrap().is_none());
        assert!(store.generation(&OTHER_COLLECTION).is_none());
    }

    #[test]
    fn test_delete_collection_clears_live_roots() {
        let dir = test_dir("delete_live_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x01u8; 16];
        store
            .put(
                &OTHER_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();
        store.set_live_roots(&OTHER_COLLECTION, vec![id]);
        assert!(store.live_roots.read().contains_key(&OTHER_COLLECTION));

        store.delete_collection(&OTHER_COLLECTION).unwrap();
        assert!(!store.live_roots.read().contains_key(&OTHER_COLLECTION));
    }

    #[test]
    fn test_batch_put_get() {
        let dir = test_dir("batch");
        let store = PackfileStorage::open(dir).unwrap();

        let entries = ten_record_fixture();

        store.put_many(&TEST_COLLECTION, &entries).unwrap();

        let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        let results = store.get_many(&TEST_COLLECTION, &ids).unwrap();
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_some());
            assert_eq!(result.as_ref().unwrap().bytes, entries[i].1.bytes);
        }
    }

    #[test]
    fn test_get_many_preserves_caller_order_despite_sorted_reads() {
        let dir = test_dir("batch_order");
        let store = PackfileStorage::open(dir).unwrap();

        let entries = ten_record_fixture();
        store.put_many(&TEST_COLLECTION, &entries).unwrap();
        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }

        let mut reversed_ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
        reversed_ids.reverse();

        let results = store.get_many(&TEST_COLLECTION, &reversed_ids).unwrap();
        assert_eq!(results.len(), 10);
        for (i, id) in reversed_ids.iter().enumerate() {
            let expected = &entries.iter().find(|(eid, _)| eid == id).unwrap().1;
            assert_eq!(
                results[i].as_ref().expect("record must be found").bytes,
                expected.bytes,
                "result at position {i} must match the id requested at that position"
            );
        }
    }

    #[test]
    fn test_walk_ancestors_stops_at_boundary() {
        let dir = test_dir("walk_basic");
        let store = PackfileStorage::open(dir).unwrap();

        // A -> B -> C -> D -> E, linear chain, edges point to predecessors.
        let ids: Vec<NodeId> = (0..5u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (ids[1], vec![ids[0]]),
            (ids[2], vec![ids[1]]),
            (ids[3], vec![ids[2]]),
            (ids[4], vec![ids[3]]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // frontier=[D], stop_at=[B]: should yield C, D in ancestor-first
        // order — stopping at and excluding B, including the D frontier.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[3]],
                &[ids[1]],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        assert_eq!(
            bytes,
            vec![b'C', b'D'],
            "walk must be ancestor-first, excluding the stop_at boundary"
        );

        // Empty stop_at: walks all the way back to the collection's true roots.
        let full: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[4]],
                &[],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let full_bytes: Vec<u8> = full.iter().map(|(_, d)| d.bytes[0]).collect();
        assert_eq!(full_bytes, vec![b'A', b'B', b'C', b'D', b'E']);
    }

    #[test]
    fn test_walk_ancestors_branching_dag() {
        let dir = test_dir("walk_branch");
        let store = PackfileStorage::open(dir).unwrap();

        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let ids: Vec<NodeId> = (0..4u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'A' + byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (ids[1], vec![ids[0]]),
            (ids[2], vec![ids[0]]),
            (ids[3], vec![ids[1], ids[2]]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // frontier=[D], stop_at=[A]: should yield exactly {B, C, D} — A
        // excluded as the boundary, both branches merge back in correctly.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[3]],
                &[ids[0]],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        // D must come last (ancestor-first order); B/C order between them
        // is unconstrained since they're siblings.
        assert_eq!(*bytes.last().unwrap(), b'D');
        bytes.sort_unstable();
        assert_eq!(bytes, vec![b'B', b'C', b'D']);
    }

    #[test]
    fn test_walk_ancestors_never_reaching_stop_at_walks_to_roots() {
        // A fork that never crosses the supplied stop_at boundary must not
        // be silently truncated there — it's an ancestor walk with stop
        // markers, not a from/to span, so it walks to the collection's true
        // roots instead.
        let dir = test_dir("walk_fork_misses_boundary");
        let store = PackfileStorage::open(dir).unwrap();

        //   ROOT
        //   /  \
        // MAIN  FORK
        //  |
        // TIP
        let root = distinct_id(0);
        let main = distinct_id(1);
        let fork = distinct_id(2);
        let tip = distinct_id(3);
        for (id, byte) in [(root, b'R'), (main, b'M'), (fork, b'F'), (tip, b'T')] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from(vec![byte])),
                )
                .unwrap();
        }
        let edges = std::collections::HashMap::from([
            (main, vec![root]),
            (fork, vec![root]),
            (tip, vec![main]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // stop_at names `fork`, which TIP's history never crosses — the
        // walk from TIP must still reach ROOT rather than stopping short.
        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[tip],
                &[fork],
                extract,
                WalkLimits::default(),
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let mut bytes: Vec<u8> = walked.iter().map(|(_, d)| d.bytes[0]).collect();
        bytes.sort_unstable();
        assert_eq!(bytes, vec![b'M', b'R', b'T']);
    }

    #[test]
    fn test_walk_ancestors_max_nodes_bounds_a_runaway_walk() {
        let dir = test_dir("walk_max_nodes");
        let store = PackfileStorage::open(dir).unwrap();

        // A long chain with a stop_at that never gets hit — max_nodes is
        // the only thing that keeps this walk from covering the chain.
        let ids: Vec<NodeId> = (0..20u8).map(distinct_id).collect();
        for (i, id) in ids.iter().enumerate() {
            let byte = u8::try_from(i % 26).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    id,
                    &NodeData::new(bytes::Bytes::from(vec![b'a' + byte])),
                )
                .unwrap();
        }
        let mut edges = std::collections::HashMap::new();
        for i in 1..ids.len() {
            edges.insert(ids[i], vec![ids[i - 1]]);
        }
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        let walked: Vec<(NodeId, NodeData)> = store
            .walk_ancestors(
                &TEST_COLLECTION,
                &[ids[19]],
                &[],
                extract,
                WalkLimits { max_nodes: Some(5) },
            )
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            walked.len() <= 5,
            "max_nodes must bound the walk even though stop_at was never reached"
        );
    }

    #[test]
    fn test_walk_ancestors_survives_concurrent_repack_gc() {
        // Regression test for the exact concurrency bug flagged in review:
        // an earlier draft resolved each node lazily, at drain time,
        // against whatever generation happened to be live *then* — so a
        // repack that GC'd a node between building the walk and draining
        // it would silently drop that node from the results. Resolving
        // eagerly against one frozen generation snapshot up front (what
        // walk_ancestors does now) must not exhibit that.
        let dir = test_dir("walk_survives_repack");
        let store = PackfileStorage::open(dir).unwrap();

        let a = distinct_id(0);
        let b = distinct_id(1);
        store
            .put(
                &TEST_COLLECTION,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();
        let edges = std::collections::HashMap::from([(b, vec![a])]);
        let extract = |hash: &[u8; 16], _data: &[u8]| edges.get(hash).cloned().unwrap_or_default();

        // Build the walk (resolves and captures A and B eagerly)...
        let walk = store
            .walk_ancestors(&TEST_COLLECTION, &[b], &[], extract, WalkLimits::default())
            .unwrap();

        // ...then, before draining it, run a repack that only keeps B as
        // a live root — A becomes unreachable and gets GC'd out of the
        // collection's index/cache entirely.
        store.set_live_roots(&TEST_COLLECTION, vec![b]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!((kept, dropped), (1, 1), "repack must have GC'd A");

        // The already-built walk must still yield both A and B: it
        // captured its data before the repack ran, so it isn't affected
        // by the generation swap that just happened.
        let results: Vec<(NodeId, NodeData)> = walk.collect::<Result<_, _>>().unwrap();
        let mut bytes: Vec<u8> = results.iter().map(|(_, d)| d.bytes[0]).collect();
        bytes.sort_unstable();
        assert_eq!(
            bytes,
            vec![b'A', b'B'],
            "a walk built before a concurrent repack must not lose nodes that repack GC'd afterward"
        );
    }

    #[test]
    fn test_multiple_records_same_collection() {
        let dir = test_dir("multi");
        let store = PackfileStorage::open(dir).unwrap();

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let data = NodeData::new(bytes::Bytes::from(format!("record {i}")));
            store.put(&TEST_COLLECTION, &id, &data).unwrap();
        }

        for i in 0..5u8 {
            let mut id = [0u8; 16];
            id[0] = i;
            let got = store.get(&TEST_COLLECTION, &id).unwrap().unwrap();
            assert_eq!(got.bytes, bytes::Bytes::from(format!("record {i}")));
        }
    }

    #[test]
    fn test_collection_isolation() {
        let dir = test_dir("isolation");
        let store = PackfileStorage::open(dir).unwrap();

        let id_a = [0x42u8; 16];
        let id_b = [0x43u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

        store.put(&TEST_COLLECTION, &id_a, &data_a).unwrap();
        store.put(&OTHER_COLLECTION, &id_b, &data_b).unwrap();

        let got_a = store.get(&TEST_COLLECTION, &id_a).unwrap().unwrap();
        let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
        assert_eq!(got_a.bytes, data_a.bytes);
        assert_eq!(got_b.bytes, data_b.bytes);

        assert!(store.get(&OTHER_COLLECTION, &id_a).unwrap().is_none());
        assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_none());

        store.delete_collection(&TEST_COLLECTION).unwrap();
        assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_none());
        let got_b = store.get(&OTHER_COLLECTION, &id_b).unwrap().unwrap();
        assert_eq!(got_b.bytes, data_b.bytes);
    }

    #[test]
    fn test_concurrent_federation_swarm() {
        use std::sync::Mutex;
        use std::thread;

        const NUM_WRITERS: usize = 8;
        const EVENTS_PER_WRITER: usize = 500;
        const NUM_READERS: usize = 8;
        const READS_PER_READER: usize = 2000;

        let dir = test_dir("concurrent_federation");
        let store = PackfileStorage::open(dir).unwrap();

        let collection = [0x77u8; 16];
        let written: Mutex<Vec<(NodeId, bytes::Bytes)>> = Mutex::new(Vec::new());
        let read_ok = std::sync::atomic::AtomicUsize::new(0);
        let read_not_found = std::sync::atomic::AtomicUsize::new(0);

        thread::scope(|scope| {
            for w in 0..NUM_WRITERS {
                let store = &store;
                let written = &written;
                scope.spawn(move || {
                    for i in 0..EVENTS_PER_WRITER {
                        let mut id = [0u8; 16];
                        id[0] = u8::try_from(w).unwrap();
                        id[1..9].copy_from_slice(&(i as u64).to_le_bytes());
                        let bytes = bytes::Bytes::from(format!("writer {w} event {i}"));
                        let data = NodeData::new(bytes.clone());
                        store.put(&collection, &id, &data).unwrap();
                        written.lock().unwrap().push((id, bytes));
                    }
                });
            }

            for _ in 0..NUM_READERS {
                let store = &store;
                let written = &written;
                let read_ok = &read_ok;
                let read_not_found = &read_not_found;
                scope.spawn(move || {
                    for i in 0..READS_PER_READER {
                        let snapshot_len = written.lock().unwrap().len();
                        if snapshot_len == 0 {
                            continue;
                        }
                        let idx = i % snapshot_len;
                        let (id, expected_bytes) = written.lock().unwrap()[idx].clone();
                        match store.get(&collection, &id).unwrap() {
                            Some(data) => {
                                assert_eq!(data.bytes, expected_bytes);
                                read_ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            None => {
                                read_not_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });

        let written = written.into_inner().unwrap();
        assert_eq!(written.len(), NUM_WRITERS * EVENTS_PER_WRITER);

        let mut lost = Vec::new();
        for (id, expected_bytes) in &written {
            match store.get(&collection, id).unwrap() {
                Some(data) => assert_eq!(data.bytes, *expected_bytes),
                None => lost.push(*id),
            }
        }

        eprintln!(
            "concurrent federation swarm: {} writes, {} reads ok, {} reads not-found, {} lost",
            written.len(),
            read_ok.load(std::sync::atomic::Ordering::Relaxed),
            read_not_found.load(std::sync::atomic::Ordering::Relaxed),
            lost.len()
        );

        assert!(
            lost.is_empty(),
            "{} of {} records lost: {lost:?}",
            lost.len(),
            written.len()
        );
    }

    #[test]
    fn test_swizzle_callback() {
        static SWIZZLE_CALLS: AtomicU64 = AtomicU64::new(0);
        static CACHED_CHILDREN_FOUND: AtomicU64 = AtomicU64::new(0);

        fn test_swizzle(
            data: &NodeData,
            children: &[NodeId],
            cached: &[Option<Arc<NodeData>>],
        ) -> NodeData {
            SWIZZLE_CALLS.fetch_add(1, Ordering::Relaxed);
            assert_eq!(children.len(), 2);
            for entry in cached {
                if entry.is_some() {
                    CACHED_CHILDREN_FOUND.fetch_add(1, Ordering::Relaxed);
                }
            }
            data.clone()
        }

        let dir = test_dir("swizzle");
        let store = PackfileStorage::open_with_swizzle(dir, 100, test_swizzle).unwrap();

        let parent_id = [0x10u8; 16];
        let child_a = [0x20u8; 16];
        let child_b = [0x30u8; 16];

        let data_a = NodeData::new(bytes::Bytes::from_static(b"child A"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"child B"));
        let parent_data = NodeData::new(bytes::Bytes::from_static(b"parent with children"));

        store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
        store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();
        store
            .put(&TEST_COLLECTION, &parent_id, &parent_data)
            .unwrap();

        if let Some(gen) = store.generation(&TEST_COLLECTION) {
            gen.cache.clear();
        }
        store.put(&TEST_COLLECTION, &child_a, &data_a).unwrap();
        store.put(&TEST_COLLECTION, &child_b, &data_b).unwrap();

        let extract = |_data: &NodeData| -> Vec<NodeId> { vec![child_a, child_b] };

        let result = store
            .get_swizzled(&TEST_COLLECTION, &parent_id, extract)
            .unwrap();
        assert!(result.is_some());
        let node_ref = result.unwrap();
        assert!(node_ref.is_resolved());
        assert_eq!(node_ref.structural_hash(), &parent_id);

        assert_eq!(SWIZZLE_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(CACHED_CHILDREN_FOUND.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_sync() {
        let dir = test_dir("sync");
        let store = PackfileStorage::open(dir).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &[0xAA; 16],
                &NodeData::new(bytes::Bytes::from_static(b"hi")),
            )
            .unwrap();
        store.sync().unwrap();
    }

    #[test]
    fn test_rebuild_index_triggers_on_full_table() {
        let dir = test_dir("rebuild_index_full");
        let store = PackfileStorage::open(dir).unwrap();

        let threshold = 3073u32;
        for i in 0..threshold {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            id[8..12].copy_from_slice(&(i + 1).to_le_bytes());
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(b"x")),
                )
                .unwrap();
        }

        let mut extra = [0u8; 16];
        extra[0..4].copy_from_slice(&threshold.to_le_bytes());
        extra[8..12].copy_from_slice(&(threshold + 1).to_le_bytes());
        store
            .put(
                &TEST_COLLECTION,
                &extra,
                &NodeData::new(bytes::Bytes::from_static(b"y")),
            )
            .unwrap();

        let got = store.get(&TEST_COLLECTION, &extra).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"y"));
    }

    #[test]
    fn test_index_offset_gate_rejects_unrepresentable_offsets() {
        let hash = [0x5A; 16];
        assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT).is_ok());
        assert!(check_index_offset(0, &hash, PACK_INDEX_OFFSET_LIMIT + 1).is_err());
        assert!(check_index_offset(0, &hash, u64::MAX).is_err());
    }

    #[test]
    fn test_refresh_collection_multi_worker_visibility() {
        let dir = test_dir("refresh_collection_multi_worker");
        let writer = PackfileStorage::open(dir.clone()).unwrap();

        // Open a read-only instance BEFORE the writer writes the data.
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();

        let id = [0xAA; 16];
        writer
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"multi_worker_test")),
            )
            .unwrap();
        writer.sync().unwrap();

        // Reader's in-memory index should be completely unaware of the new write.
        assert!(reader.get(&TEST_COLLECTION, &id).unwrap().is_none());

        // Trigger the external refresh, simulating a cache invalidation signal.
        reader.refresh_collection(&TEST_COLLECTION).unwrap();

        // Reader should now have correctly loaded the delta and rebuilt its index.
        let got = reader.get(&TEST_COLLECTION, &id).unwrap().unwrap();
        assert_eq!(got.bytes, bytes::Bytes::from_static(b"multi_worker_test"));
    }

    #[test]
    fn test_scan_existing_skips_malformed_filenames() {
        let dir = test_dir("scan_existing_junk");
        std::fs::write(dir.join("nounderscore.pack"), b"").unwrap();
        std::fs::write(dir.join("aabb_00.pack"), b"").unwrap();
        std::fs::write(dir.join("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_00.pack"), b"").unwrap();
        std::fs::write(dir.join("00000000000000000000000000000000_gg.pack"), b"").unwrap();
        {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt;
            let path = dir.join(OsStr::from_bytes(b"\xff\xfe.pack"));
            std::fs::write(path, b"").unwrap();
        }

        let valid_path = dir.join("pack_0000000000000000.pack");
        let mut buf = Vec::new();
        packfile::write_header(&mut buf, 0).unwrap();
        packfile::write_record(
            &mut buf,
            &packfile::Record {
                collection_id: [0x01; 16],
                hash: [0xAA; 16],
                data: bytes::Bytes::from_static(b"hello"),
            },
        )
        .unwrap();
        std::fs::write(&valid_path, &buf).unwrap();

        let store = PackfileStorage::open(dir).unwrap();
        let gen = store.generation(&[0x01u8; 16]).unwrap();
        assert_eq!(gen.index.len(), 1);
    }

    #[test]
    fn test_delete_collection_preserves_other_collection_cache() {
        let dir = test_dir("delete_collection_cache");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let collection_a = TEST_COLLECTION;
        let collection_b = OTHER_COLLECTION;
        let id_a = [0x10u8; 16];
        let id_b = [0x20u8; 16];
        let data_a = NodeData::new(bytes::Bytes::from_static(b"collection A data"));
        let data_b = NodeData::new(bytes::Bytes::from_static(b"collection B data"));

        store.put(&collection_a, &id_a, &data_a).unwrap();
        store.put(&collection_b, &id_b, &data_b).unwrap();

        assert!(store.get(&collection_a, &id_a).unwrap().is_some());
        assert!(store.get(&collection_b, &id_b).unwrap().is_some());

        let gen_b_before = store.generation(&collection_b).unwrap();
        let hits_b_before = gen_b_before.cache.hits();

        store.delete_collection(&collection_a).unwrap();

        assert!(store.get(&collection_a, &id_a).unwrap().is_none());
        assert!(store.generation(&collection_a).is_none());
        assert!(store.generation(&collection_b).is_some());
        assert!(store.get(&collection_b, &id_b).unwrap().is_some());

        let gen_b_after = store.generation(&collection_b).unwrap();
        assert!(gen_b_after.cache.hits() > hits_b_before);
    }

    #[test]
    fn test_repack_collection_reachable_no_roots_preserves_diamond_dag_in_topo_order() {
        let dir = test_dir("repack_topo");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;
        let mut id_c = [0u8; 16];
        id_c[0] = 3;
        let mut id_d = [0u8; 16];
        id_d[0] = 4;

        store
            .put(
                &TEST_COLLECTION,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_c,
                &NodeData::new(bytes::Bytes::from_static(b"C")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_d,
                &NodeData::new(bytes::Bytes::from_static(b"D")),
            )
            .unwrap();

        let edges = std::collections::HashMap::from([
            (id_b, vec![id_a]),
            (id_c, vec![id_a]),
            (id_d, vec![id_b, id_c]),
        ]);

        // No live roots configured for TEST_COLLECTION: preserves everything,
        // still deduplicated and topologically ordered.
        let result = store.repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
            edges.get(hash).cloned().unwrap_or_default()
        });
        let (kept, dropped) = result.unwrap();
        assert_eq!(kept, 4);
        assert_eq!(dropped, 0);

        for (id, expected) in [
            (id_a, b"A".as_slice()),
            (id_b, b"B".as_slice()),
            (id_c, b"C".as_slice()),
            (id_d, b"D".as_slice()),
        ] {
            let got = store
                .get(&TEST_COLLECTION, &id)
                .unwrap()
                .expect("record missing after topo repack");
            assert_eq!(got.bytes.as_ref(), expected);
        }
    }

    #[test]
    fn test_repack_collection_reachable_drops_unreachable_records() {
        let dir = test_dir("repack_reachable_gc");
        let store = PackfileStorage::open(dir).unwrap();

        // Live chain: root -> p1 -> p0 (p0 has no further dependencies).
        let root = distinct_id(1);
        let p1 = distinct_id(2);
        let p0 = distinct_id(3);

        // Garbage island, unreachable from root: garbage -> garbage_dep.
        let garbage = distinct_id(4);
        let garbage_dep = distinct_id(5);

        for (id, bytes) in [
            (root, b"root".as_slice()),
            (p1, b"p1".as_slice()),
            (p0, b"p0".as_slice()),
            (garbage, b"garbage".as_slice()),
            (garbage_dep, b"garbage_dep".as_slice()),
        ] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(bytes)),
                )
                .unwrap();
        }

        let edges = std::collections::HashMap::from([
            (root, vec![p1]),
            (p1, vec![p0]),
            (garbage, vec![garbage_dep]),
        ]);

        store.set_live_roots(&TEST_COLLECTION, vec![root]);

        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();

        assert_eq!(kept, 3, "root, p1, p0 should survive");
        assert_eq!(dropped, 2, "garbage and garbage_dep should be collected");

        for id in [root, p1, p0] {
            assert!(
                store.get(&TEST_COLLECTION, &id).unwrap().is_some(),
                "live record missing after reachable repack"
            );
        }
        for id in [garbage, garbage_dep] {
            assert!(
                store.get(&TEST_COLLECTION, &id).unwrap().is_none(),
                "garbage record survived reachable repack"
            );
        }

        let stats = store.repack_stats();
        assert_eq!(stats.repack_count, 1);
        assert_eq!(stats.kept_total, 3);
        assert_eq!(stats.dropped_total, 2);
        assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 1);

        // A second repack uses the incremental path: it clones the
        // previous live_map (3 entries from the first repack's output)
        // and scans only newly-appended bytes (none here).  The garbage
        // entries in the old shard are never re-scanned — the live_map
        // already excludes them, so dropped is 0.  This is the O(n)
        // behavior: each repack does bounded work proportional to new
        // bytes, not proportional to total shard size.
        let (kept2, dropped2) = store
            .repack_collection_reachable(&TEST_COLLECTION, |hash, _data| {
                edges.get(hash).cloned().unwrap_or_default()
            })
            .unwrap();
        assert_eq!((kept2, dropped2), (3, 0));

        let stats2 = store.repack_stats();
        assert_eq!(stats2.repack_count, 2);
        assert_eq!(stats2.kept_total, stats.kept_total + kept2 as u64);
        assert_eq!(stats2.dropped_total, stats.dropped_total + dropped2 as u64);
        assert_eq!(store.repack_count_for_collection(&TEST_COLLECTION), 2);
    }

    /// Verify the root-removal fallback: if a live root is removed between
    /// repacks, nodes exclusively reachable through it must not stay in the
    /// cached live set indefinitely. Without the fallback, the incremental
    /// adjacency cache only grows and those nodes would never be collected.
    #[test]
    fn test_repack_incremental_root_removal_forces_full_sweep() {
        let dir = test_dir("repack_root_removal_fallback");
        let store = PackfileStorage::open(dir).unwrap();

        // Graph:
        //   root_a -> shared -> base
        //   root_b -> shared -> base
        //   orphan_a -> orphan_dep    (reachable only via root_a)
        //
        // First repack: both root_a and root_b are live. Everything survives.
        // Second repack: only root_b is live. orphan_a and orphan_dep must be
        // collected. Without the root-removal fallback they would remain in the
        // incremental live set permanently.
        let root_a = distinct_id(1);
        let root_b = distinct_id(2);
        let shared = distinct_id(3);
        let base = distinct_id(4);
        let orphan_a = distinct_id(5);
        let orphan_dep = distinct_id(6);

        for (id, bytes) in [
            (root_a, b"root_a".as_slice()),
            (root_b, b"root_b".as_slice()),
            (shared, b"shared".as_slice()),
            (base, b"base".as_slice()),
            (orphan_a, b"orphan_a".as_slice()),
            (orphan_dep, b"orphan_dep".as_slice()),
        ] {
            store
                .put(
                    &TEST_COLLECTION,
                    &id,
                    &NodeData::new(bytes::Bytes::from_static(bytes)),
                )
                .unwrap();
        }

        let edges = std::collections::HashMap::from([
            (root_a, vec![shared, orphan_a]),
            (root_b, vec![shared]),
            (shared, vec![base]),
            (orphan_a, vec![orphan_dep]),
        ]);
        let extract = |hash: &[u8; 16], _data: &[u8]| -> Vec<[u8; 16]> {
            edges.get(hash).cloned().unwrap_or_default()
        };

        // First repack: both roots live — everything reachable from either root
        // survives and the incremental adjacency cache is populated.
        store.set_live_roots(&TEST_COLLECTION, vec![root_a, root_b]);
        let (kept1, dropped1) = store
            .repack_collection_reachable(&TEST_COLLECTION, extract)
            .unwrap();
        assert_eq!(kept1, 6, "all nodes should survive with both roots live");
        assert_eq!(dropped1, 0);

        // Now remove root_a — switch to root_b only.
        // orphan_a and orphan_dep are now exclusively reachable through root_a,
        // which was removed. The root-removal fallback must trigger a full BFS
        // sweep so they are actually collected, rather than remaining in the
        // stale incremental live set forever.
        store.set_live_roots(&TEST_COLLECTION, vec![root_b]);
        let (kept2, dropped2) = store
            .repack_collection_reachable(&TEST_COLLECTION, extract)
            .unwrap();
        assert_eq!(
            kept2, 3,
            "root_b, shared, base should survive; root_a, orphan_a, orphan_dep must be GC'd"
        );
        assert_eq!(
            dropped2, 3,
            "root_a, orphan_a, and orphan_dep must be collected once root_a is removed from live roots"
        );

        assert!(
            store.get(&TEST_COLLECTION, &orphan_a).unwrap().is_none(),
            "orphan_a must be gone after root_a was removed"
        );
        assert!(
            store.get(&TEST_COLLECTION, &orphan_dep).unwrap().is_none(),
            "orphan_dep must be gone after root_a was removed"
        );
        assert!(
            store.get(&TEST_COLLECTION, &root_a).unwrap().is_none(),
            "root_a itself has no incoming edges so it is unreachable from root_b"
        );
        assert!(
            store.get(&TEST_COLLECTION, &root_b).unwrap().is_some(),
            "root_b must still be present"
        );
        assert!(
            store.get(&TEST_COLLECTION, &shared).unwrap().is_some(),
            "shared must still be present"
        );
        assert!(
            store.get(&TEST_COLLECTION, &base).unwrap().is_some(),
            "base must still be present"
        );
    }

    /// Repack drops unreachable *records* from the index (`dropped` count),
    /// but today nothing ever retires the *shard file* they lived in: no
    /// code path sets `Shard::is_current` to `false` or clears the pool's
    /// slot for it (see `Shard::drop`'s doc and
    /// `test_drop_deletes_retired_shard_only_after_last_reference` in
    /// shard.rs, which has to simulate retirement manually because nothing
    /// production triggers it). `rotate()`'s error message says "repack to
    /// reclaim", but repack currently reclaims nothing at the shard level.
    ///
    /// This fills a small fixed number of pool slots (4), repacks away
    /// everything in the non-active ones (100% garbage, nothing live left),
    /// and then expects one more rotation to succeed by reusing a
    /// now-empty slot.
    #[test]
    fn test_repack_reclaims_shard_slots_for_rotation() {
        // Use a fixed small count rather than MAX_SHARDS — with 4096 slots
        // and 256MB each that would be a 1TB test.
        const TEST_SHARDS: usize = 4;

        let dir = test_dir("repack_reclaims_slots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root = [0u8; 16];
        root[0] = 0xFF;
        store
            .put(
                &TEST_COLLECTION,
                &root,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();

        // `root` already occupies the first slot; force rotation through
        // the remaining TEST_SHARDS - 1 slots, dumping garbage into each so
        // every shard but the last ends up fully unreachable once we
        // repack with `root` as the only live node.
        for i in 0..TEST_SHARDS - 1 {
            store.shards.active_shard().file_len.store(
                shard::MAX_SHARD_BYTES - 10,
                std::sync::atomic::Ordering::Release,
            );
            let mut garbage = [0u8; 16];
            garbage[0] = u8::try_from(i + 1).unwrap();
            store
                .put(
                    &TEST_COLLECTION,
                    &garbage,
                    &NodeData::new(bytes::Bytes::from_static(b"garbage")),
                )
                .unwrap();
        }

        // Every slot should now be occupied.
        let occupied = (0..u16::try_from(TEST_SHARDS).unwrap())
            .filter(|&id| store.shards.get_shard(id).is_some())
            .count();
        assert_eq!(
            occupied, TEST_SHARDS,
            "test setup should have filled every shard slot"
        );

        store.set_live_roots(&TEST_COLLECTION, vec![root]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1, "only root should survive");
        assert!(dropped > 0, "the garbage records should have been dropped");
        assert!(
            store.shards_retired() > 0,
            "retire_empty_shards should have retired at least one now-garbage-only shard"
        );

        // Every shard except the current active one held nothing but
        // garbage, and that garbage is now unreachable -- those slots
        // should be reclaimable. Forcing one more rotation should reuse
        // one of them rather than failing.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut one_more = [0u8; 16];
        one_more[0] = 0xEE;
        store
            .put(
                &TEST_COLLECTION,
                &one_more,
                &NodeData::new(bytes::Bytes::from_static(b"one more")),
            )
            .expect(
                "rotation should reclaim a garbage-only shard slot after repack, \
                 not report the pool as permanently full",
            );
    }

    #[test]
    fn test_repack_moves_live_data_out_of_its_source_shard() {
        let dir = test_dir("repack_moves_from_source");
        let store = PackfileStorage::open(dir).unwrap();
        let node = [0xA5; 16];
        store
            .put(
                &TEST_COLLECTION,
                &node,
                &NodeData::new(bytes::Bytes::from_static(b"live")),
            )
            .unwrap();

        let source = store
            .collection_referenced_shards(&TEST_COLLECTION)
            .into_iter()
            .next()
            .unwrap();
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        assert_eq!((kept, dropped), (1, 0));
        let destination = store
            .collection_referenced_shards(&TEST_COLLECTION)
            .into_iter()
            .next()
            .unwrap();
        assert_ne!(destination, source);
        assert!(store.shards.get_shard(source).is_none());
        assert_eq!(
            store
                .get(&TEST_COLLECTION, &node)
                .unwrap()
                .unwrap()
                .bytes
                .as_ref(),
            b"live"
        );
    }

    /// Repack must not retire a shard that another collection's index still
    /// references.  This test puts live records from two different collections
    /// into the same shard, then repacks only collection A — the shared shard
    /// must survive because collection B's index still points into it.
    #[test]
    fn test_repack_does_not_retire_shard_referenced_by_other_collection() {
        let dir = test_dir("repack_cross_collection_safety");
        let store = PackfileStorage::open(dir).unwrap();

        // Put a record for collection A — lands on shard 0.
        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        store
            .put(
                &TEST_COLLECTION,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
            )
            .unwrap();

        // Force a rotation so the next write goes to a different shard.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );

        // Put a record for collection B — lands on shard 1.
        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
            )
            .unwrap();

        // Force another rotation and put more collection A garbage on shard 2.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = 0xCC;
        store
            .put(
                &TEST_COLLECTION,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();

        // Repack collection A with only root_a as live.  The garbage record
        // should be dropped, but shard 1 (which holds collection B's root)
        // must NOT be retired.
        store.set_live_roots(&TEST_COLLECTION, vec![root_a]);
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, 1);
        assert!(dropped > 0);

        // Collection B's root must still be retrievable — shard 1 was not retired.
        let got = store.get(&OTHER_COLLECTION, &root_b).unwrap();
        assert!(
            got.is_some(),
            "collection B root must survive repack of collection A — shared shard must not be retired",
        );
    }

    /// `open_read_only` must actually coexist with a live writer end to
    /// end — not just take no lock, but also not touch/truncate the
    /// writer's files via the collection-index rebuild scan (the real risk this
    /// whole design exists to avoid). Also confirms it sees the writer's
    /// already-durable data and that the writer is unaffected afterward.
    #[test]
    fn test_open_read_only_coexists_with_active_writer() {
        let dir = test_dir("open_read_only_coexist");
        let writer = PackfileStorage::open(dir.clone()).unwrap();

        let mut id = [0u8; 16];
        id[0] = 0xAA;
        writer
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"hello")),
            )
            .unwrap();

        // Open read-only while the writer is still alive — must succeed
        // (no lock conflict) and see the write above.
        let reader = PackfileStorage::open_read_only(dir.clone()).unwrap();
        assert_eq!(
            reader.get(&TEST_COLLECTION, &id).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(b"hello"))
        );
        drop(reader);

        // The writer must be completely unaffected — still open, still
        // able to write more data afterward.
        let mut id2 = [0u8; 16];
        id2[0] = 0xBB;
        writer
            .put(
                &TEST_COLLECTION,
                &id2,
                &NodeData::new(bytes::Bytes::from_static(b"world")),
            )
            .unwrap();
        assert_eq!(
            writer.get(&TEST_COLLECTION, &id2).unwrap().map(|d| d.bytes),
            Some(bytes::Bytes::from_static(b"world"))
        );
    }

    /// `repack_shard` must find and repack every collection sharing a shard —
    /// the targeted way to reclaim one shard without waiting for each
    /// collection to independently cross its own repack threshold.
    #[test]
    fn test_repack_shard_repacks_every_referencing_collection() {
        let dir = test_dir("repack_shard");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root_a = [0u8; 16];
        root_a[0] = 0xAA;
        let mut garbage_a = [0u8; 16];
        garbage_a[0] = 0xA1;
        store
            .put(
                &TEST_COLLECTION,
                &root_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A root")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &garbage_a,
                &NodeData::new(bytes::Bytes::from_static(b"collection A garbage")),
            )
            .unwrap();
        store.set_live_roots(&TEST_COLLECTION, vec![root_a]);

        let mut root_b = [0u8; 16];
        root_b[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &root_b,
                &NodeData::new(bytes::Bytes::from_static(b"collection B root")),
            )
            .unwrap();
        // No set_live_roots for collection B: everything must survive its repack.

        // Fresh pool, both collections' first writes land on shard 0 (shared).
        let referencing = store.collections_referencing_shard(0).unwrap();
        assert_eq!(referencing.len(), 2);
        assert!(referencing.contains(&TEST_COLLECTION));
        assert!(referencing.contains(&OTHER_COLLECTION));

        let mut results = store.repack_shard(0, |_hash, _data| Vec::new()).unwrap();
        results.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);

        let mut expected = vec![
            (TEST_COLLECTION, 1usize, 1usize),
            (OTHER_COLLECTION, 1usize, 0usize),
        ];
        expected.sort_unstable_by_key(|(collection_id, _, _)| *collection_id);
        assert_eq!(
            results, expected,
            "repack_shard must repack both collections sharing shard 0"
        );

        assert!(store.get(&TEST_COLLECTION, &root_a).unwrap().is_some());
        assert!(store.get(&TEST_COLLECTION, &garbage_a).unwrap().is_none());
        assert!(store.get(&OTHER_COLLECTION, &root_b).unwrap().is_some());
    }

    /// `collections_referencing_shard` must filter the shard-scan's candidate
    /// set against each collection's *current* index, not just report every collection
    /// whose bytes ever physically touched the shard. A collection that has
    /// since repacked its live data onto a different shard leaves its old
    /// bytes sitting there untouched (shards are append-only, never
    /// rewritten in place) — that collection must NOT show up as still
    /// referencing the shard its data moved away from.
    #[test]
    fn test_collections_referencing_shard_excludes_collections_already_repacked_away() {
        let dir = test_dir("collections_referencing_shard_stale");
        let store = PackfileStorage::open(dir).unwrap();

        let mut root = [0u8; 16];
        root[0] = 0xAA;
        store
            .put(
                &TEST_COLLECTION,
                &root,
                &NodeData::new(bytes::Bytes::from_static(b"root")),
            )
            .unwrap();

        // OTHER_COLLECTION also lands on shard 0 (shared, still the pool's
        // active shard) and stays live there — this is what keeps shard 0
        // from being retired outright once TEST_COLLECTION moves off it, so the
        // "stale reference" case below is actually reachable to query.
        let mut other_root = [0u8; 16];
        other_root[0] = 0xBB;
        store
            .put(
                &OTHER_COLLECTION,
                &other_root,
                &NodeData::new(bytes::Bytes::from_static(b"other collection root")),
            )
            .unwrap();

        // Force TEST_COLLECTION's home (shard 0) to look full, so its next write
        // rotates *its own* home to shard 1 — root stays physically on
        // shard 0, but TEST_COLLECTION's home (and this next record, `garbage`)
        // moves to shard 1.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        let mut garbage = [0u8; 16];
        garbage[0] = 0xA1;
        store
            .put(
                &TEST_COLLECTION,
                &garbage,
                &NodeData::new(bytes::Bytes::from_static(b"garbage")),
            )
            .unwrap();

        // Repack with only root live: TEST_COLLECTION's records now span *two*
        // source shards (root on 0, garbage on 1), and a repack must never
        // rewrite live data back into one of its own sources (see
        // `ShardPool::prepare_collection_repack`'s doc) — so the kept copy lands
        // on a *third* shard (2), not shard 1. Shard 1, left holding only
        // the now-dropped `garbage` record with nothing live pointing at
        // it, gets retired and closed during this same repack.
        store.set_live_roots(&TEST_COLLECTION, vec![root]);
        store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        // Shard 0's raw bytes still physically contain TEST_COLLECTION's
        // original root record — a naive scan-only approach would still
        // report it. The current index must exclude it.
        assert!(
            !store.collections_referencing_shard(0).unwrap().contains(&TEST_COLLECTION),
            "a collection whose live data has moved off a shard must not be reported as still referencing it"
        );

        // The real invariant here isn't "the destination is shard 2" —
        // that's just where a deterministic, empty-pool allocator happens
        // to land today. What must hold is: exactly one destination, and
        // it's neither of the two source shards (0 and 1) a repack must
        // never write live data back into.
        let destinations = store.collection_referenced_shards(&TEST_COLLECTION);
        assert_eq!(
            destinations.len(),
            1,
            "TEST_COLLECTION's live data must land on exactly one shard after repack"
        );
        let destination = destinations[0];
        assert!(
            ![0, 1].contains(&destination),
            "the repack destination must not be one of its own source shards (0: root, 1: garbage)"
        );
        assert!(
            store
                .collections_referencing_shard(destination)
                .unwrap()
                .contains(&TEST_COLLECTION),
            "the collection's current shard must still be reported"
        );
    }

    #[test]
    fn test_collection_directory_from_disk_matches_collection_summaries_after_sync() {
        let dir = test_dir("collection_directory_roundtrip");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        for i in 0..5u8 {
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        for i in 0..3u8 {
            store
                .put(
                    &OTHER_COLLECTION,
                    &distinct_id(100 + i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        store.sync_all().unwrap();

        let mut from_disk = PackfileStorage::collection_directory_from_disk(&dir);
        from_disk.sort_unstable_by_key(|(collection_id, _)| *collection_id);

        let mut expected: Vec<([u8; 16], u64)> = store
            .collection_summaries()
            .into_iter()
            .map(|(collection_id, count, _mem)| (collection_id, count as u64))
            .collect();
        expected.sort_unstable_by_key(|(collection_id, _)| *collection_id);

        assert_eq!(
            from_disk, expected,
            "the persisted directory's per-collection totals must match the live index's own counts"
        );
        assert!(PackfileStorage::collection_directory_persisted_at(&dir).is_some());
    }

    #[test]
    fn test_collection_directory_from_disk_empty_when_never_persisted() {
        let dir = test_dir("collection_directory_never_persisted");
        let _store = PackfileStorage::open(dir.clone()).unwrap();
        // No sync_all call: nothing has been persisted yet.
        assert_eq!(
            PackfileStorage::collection_directory_from_disk(&dir),
            Vec::new()
        );
        assert_eq!(
            PackfileStorage::collection_directory_persisted_at(&dir),
            None
        );
    }

    #[test]
    fn test_collection_directory_reflects_delete_and_repack() {
        let dir = test_dir("collection_directory_delete_repack");
        let store = PackfileStorage::open(dir.clone()).unwrap();

        let a = distinct_id(0);
        let b = distinct_id(1);
        store
            .put(
                &TEST_COLLECTION,
                &a,
                &NodeData::new(bytes::Bytes::from_static(b"a")),
            )
            .unwrap();
        store
            .put(
                &OTHER_COLLECTION,
                &b,
                &NodeData::new(bytes::Bytes::from_static(b"b")),
            )
            .unwrap();
        store.sync_all().unwrap();

        let directory = PackfileStorage::collection_directory_from_disk(&dir);
        assert_eq!(
            directory.len(),
            2,
            "both collections must appear before deletion"
        );

        store.delete_collection(&TEST_COLLECTION).unwrap();
        store.sync_all().unwrap();

        let directory = PackfileStorage::collection_directory_from_disk(&dir);
        assert_eq!(
            directory,
            vec![(OTHER_COLLECTION, 1)],
            "a deleted collection must not linger in the persisted directory"
        );
    }

    #[test]
    fn test_plan_collection_repack_matches_real_repack_without_mutating() {
        let dir = test_dir("plan_matches_real");
        let store = PackfileStorage::open(dir).unwrap();

        for i in 0..5u8 {
            store
                .put(
                    &TEST_COLLECTION,
                    &distinct_id(i),
                    &NodeData::new(bytes::Bytes::from(vec![i])),
                )
                .unwrap();
        }
        // Duplicate a hash's payload under a fresh id to create something
        // dedup would drop... actually LossyIndex already dedups by hash
        // on insert, so instead exercise the "no live roots: keep
        // everything" path, which is what --root-less usage always hits.
        let plan = store
            .plan_collection_repack(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(plan.kept, 5);
        assert_eq!(plan.dropped, 0);
        assert!(plan.kept_bytes > 0);
        assert_eq!(plan.shards_touched, vec![0]);

        // The dry run must not have mutated anything: a real repack run
        // right after must see the exact same collection state and produce the
        // exact same result.
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();
        assert_eq!(kept, plan.kept);
        assert_eq!(dropped, plan.dropped);
    }

    #[test]
    fn test_repack_closure_single_collection_single_shard() {
        let dir = test_dir("closure_trivial");
        let store = PackfileStorage::open(dir).unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"x")),
            )
            .unwrap();

        let (collections, shards) = store.repack_closure(0).unwrap();
        assert_eq!(collections, vec![TEST_COLLECTION]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_repack_closure_pulls_in_collections_second_shard_transitively() {
        // Collection A lives on shards {0, 1} (spans a rotation). Collection B lives
        // only on shard 1. Starting the closure from shard 0 must still
        // discover shard 1 (because collection A references it) and, through
        // shard 1, collection B — even though collection B never touched shard 0 at
        // all.
        let dir = test_dir("closure_transitive");
        let store = PackfileStorage::open(dir).unwrap();

        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(0),
                &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-0")),
            )
            .unwrap();

        // Force rotation so TEST_COLLECTION's next write lands on a new shard.
        store.shards.active_shard().file_len.store(
            shard::MAX_SHARD_BYTES - 10,
            std::sync::atomic::Ordering::Release,
        );
        store
            .put(
                &TEST_COLLECTION,
                &distinct_id(1),
                &NodeData::new(bytes::Bytes::from_static(b"a-on-shard-1")),
            )
            .unwrap();

        store
            .put(
                &OTHER_COLLECTION,
                &distinct_id(2),
                &NodeData::new(bytes::Bytes::from_static(b"b-on-shard-1")),
            )
            .unwrap();

        let (mut collections, mut shards) = store.repack_closure(0).unwrap();
        collections.sort_unstable();
        shards.sort_unstable();
        let mut expected_collections = vec![TEST_COLLECTION, OTHER_COLLECTION];
        expected_collections.sort_unstable();
        assert_eq!(collections, expected_collections);
        assert_eq!(shards, vec![0, 1]);
    }

    #[test]
    fn test_collection_referenced_shards_empty_for_unknown_collection() {
        let dir = test_dir("referenced_shards_unknown");
        let store = PackfileStorage::open(dir).unwrap();
        assert_eq!(
            store.collection_referenced_shards(&TEST_COLLECTION),
            Vec::<u16>::new()
        );
    }

    #[test]
    fn test_repack_collection_reachable_without_live_roots_preserves_everything() {
        let dir = test_dir("repack_reachable_no_roots");
        let store = PackfileStorage::open(dir).unwrap();

        let mut id_a = [0u8; 16];
        id_a[0] = 1;
        let mut id_b = [0u8; 16];
        id_b[0] = 2;

        store
            .put(
                &TEST_COLLECTION,
                &id_a,
                &NodeData::new(bytes::Bytes::from_static(b"A")),
            )
            .unwrap();
        store
            .put(
                &TEST_COLLECTION,
                &id_b,
                &NodeData::new(bytes::Bytes::from_static(b"B")),
            )
            .unwrap();

        // No set_live_roots call for this collection: nothing is known to be
        // garbage, so everything must survive.
        let (kept, dropped) = store
            .repack_collection_reachable(&TEST_COLLECTION, |_hash, _data| Vec::new())
            .unwrap();

        assert_eq!(kept, 2);
        assert_eq!(dropped, 0);
        assert!(store.get(&TEST_COLLECTION, &id_a).unwrap().is_some());
        assert!(store.get(&TEST_COLLECTION, &id_b).unwrap().is_some());
    }

    /// `pin_shards` must return a distinct `Arc<Shard>` per unique id, and
    /// the returned handle must still resolve real, previously-written
    /// data — i.e. it's the live pool's own shard, not a stand-in.
    /// The deeper guarantee (pinning survives a shard being retired and
    /// its slot recycled) is `Shard::drop`'s contract, tested directly in
    /// `shard.rs` where `is_current` and the pool's internals are
    /// accessible without going through `ShardPool::rotate`, which no
    /// longer has any path to actually recycle a slot (see its doc).
    #[test]
    fn test_pin_shards_returns_readable_deduped_handles() {
        let dir = test_dir("pin_shards_basic");
        let store = PackfileStorage::open(dir).unwrap();

        let id = [0x77u8; 16];
        store
            .put(
                &TEST_COLLECTION,
                &id,
                &NodeData::new(bytes::Bytes::from_static(b"pin me")),
            )
            .unwrap();
        let (shard_id, offset) = store
            .generation(&TEST_COLLECTION)
            .unwrap()
            .index
            .lookup(&id)
            .expect("just-written record must be indexed");

        // Duplicate ids in the input must collapse to one pinned entry.
        let pinned = store.pin_shards([shard_id, shard_id, shard_id].into_iter());
        assert_eq!(pinned.len(), 1);

        let record = PackfileStorage::read_at(&pinned[&shard_id], offset, true)
            .expect("pinned shard must resolve the real on-disk record");
        assert_eq!(record.data.as_ref(), b"pin me");
    }

    #[test]
    fn test_concurrent_put_repack_reachable_no_lost_writes() {
        use std::thread;

        let dir = test_dir("concurrent_put_repack_reachable");
        let store = PackfileStorage::open(dir).unwrap();

        let collection = [0x66u8; 16];
        let written_count = std::sync::atomic::AtomicU32::new(0);

        // No live roots configured, so the reachable repacker must fall
        // back to "preserve everything" — this proves that fallback holds
        // even when interleaved with concurrent writes.
        thread::scope(|scope| {
            let writer = scope.spawn(|| {
                for i in 0..200u32 {
                    let mut id = [0u8; 16];
                    id[0..4].copy_from_slice(&i.to_le_bytes());
                    let data = NodeData::new(bytes::Bytes::from(format!("entry {i}")));
                    store.put(&collection, &id, &data).unwrap();
                    written_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

            let repacker = scope.spawn(|| {
                for _ in 0..5 {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    let _ =
                        store.repack_collection_reachable(&collection, |_hash, _data| Vec::new());
                }
            });

            writer.join().unwrap();
            repacker.join().unwrap();
        });

        let total = written_count.load(std::sync::atomic::Ordering::Relaxed);
        let mut found = 0u32;
        for i in 0..total {
            let mut id = [0u8; 16];
            id[0..4].copy_from_slice(&i.to_le_bytes());
            if store.get(&collection, &id).unwrap().is_some() {
                found += 1;
            }
        }

        assert_eq!(
            found, total,
            "lost writes during concurrent put+reachable-repack"
        );
    }
}
