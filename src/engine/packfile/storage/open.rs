use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::cache::PinnedNodes;
use crate::config::RuntimeConfig;
use crate::packfile::{self, PackId};
use crate::shard::ShardPool;
use crate::storage::StorageError;

use super::{
    persisted_collection_order, CollectionScanOutput, DeltaLogState, IndexTables, OpenPath,
    OpenTimings, OperationTimings, PackfileStorage, ReadPlanPolicy, ReloadMode, ScanInsertConfig,
    StreamingOpenScan, SwizzleFn, SyncDiagnostics, SyncTotals, DEFAULT_CACHE_CAPACITY,
    DEFAULT_DELTA_FSYNC_BUDGET_BYTES, DEFAULT_PACK_FSYNC_BUDGET_BYTES,
    DEFAULT_REPACK_THRESHOLD_ENTRIES,
};

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

    /// Open a writable packfile storage with explicit runtime settings.
    ///
    /// Callers that want environment defaults pass
    /// [`RuntimeConfig::from_env`] here.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an invalid [`RuntimeConfig`], or the same
    /// directory errors as [`Self::open`].
    pub fn open_with_config(
        base_dir: PathBuf,
        config: &RuntimeConfig,
    ) -> Result<Self, std::io::Error> {
        config.validate()?;
        let store = Self::open_with_options_locked(
            base_dir,
            config.cache_capacity,
            config
                .cache_bytes
                .map(|bytes| Arc::new(crate::cache::CacheBudget::new(bytes))),
            None,
            true,
            None,
            true,
            packfile::ChecksumPolicy::Full,
            true,
        )?;
        store.set_stats_enabled(config.stats_enabled);
        store.set_point_random_mmap(config.point_random_mmap);
        store.set_point_random_mmap_index(config.ordered_index_point_random_mmap);
        Ok(store)
    }

    /// Open a read-only packfile storage with explicit runtime settings.
    ///
    /// See [`Self::open_with_config`].
    ///
    /// # Errors
    /// Same as [`Self::open_with_config`], with the read-only contract of
    /// [`Self::open_read_only`].
    pub fn open_read_only_with_config(
        base_dir: PathBuf,
        config: &RuntimeConfig,
    ) -> Result<Self, std::io::Error> {
        config.validate()?;
        let store = Self::open_with_options_locked(
            base_dir,
            config.cache_capacity,
            config
                .cache_bytes
                .map(|bytes| Arc::new(crate::cache::CacheBudget::new(bytes))),
            None,
            false,
            None,
            true,
            packfile::ChecksumPolicy::Full,
            true,
        )?;
        store.set_stats_enabled(config.stats_enabled);
        store.set_point_random_mmap(config.point_random_mmap);
        store.set_point_random_mmap_index(config.ordered_index_point_random_mmap);
        Ok(store)
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

    /// Open a writable packfile storage with explicit cache, compression, and
    /// checksum policies.
    ///
    /// `cache_capacity` applies independently to each collection; pass zero
    /// to disable decoded-node caching entirely.
    ///
    /// # Errors
    /// Same as [`Self::open`].
    pub fn open_with_cache_and_policies(
        base_dir: PathBuf,
        cache_capacity: usize,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
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

    /// Open a read-only observer with explicit decoded-node cache and
    /// checksum policies.
    ///
    /// `cache_capacity` applies independently to each collection; pass zero
    /// to disable decoded-node caching entirely.
    ///
    /// # Errors
    /// Same as [`Self::open_read_only`].
    pub fn open_read_only_with_cache_and_policies(
        base_dir: PathBuf,
        cache_capacity: usize,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options(
            base_dir,
            cache_capacity,
            None,
            false,
            None,
            true,
            checksum_policy,
        )
    }

    /// Open a read-only observer with the read-committed journal overlay
    /// enabled, in one call.
    ///
    /// This is the intended constructor for an authoritative read-only worker
    /// handle. The durable read APIs ([`crate::storage::StorageEngine::get_many`],
    /// [`Self::get_many_with_refresh`]) deliberately hide a writer's
    /// unflushed and not-yet-checkpointed mutations, so a worker that must
    /// observe another process's committed writes has to read through
    /// [`Self::get_read_committed`] — and that only consults an overlay once
    /// [`Self::enable_read_journal`] has attached a writer's segment. Pairing
    /// the two here makes this the only supported way to obtain such a handle,
    /// so a caller cannot open one that silently falls back to the durable
    /// API.
    ///
    /// `wal_path` is the writer's journal segment. See
    /// [`Self::enable_read_journal`] for the read-only safety contract and
    /// [`Self::open_read_only`] for the writer-coexistence contract.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the store cannot be opened read-only (same
    /// as [`Self::open_read_only`]), or if the segment is unreadable, a
    /// committed group fails validation, or a coverage gap cannot be resolved
    /// by reloading the checkpoint (same as [`Self::enable_read_journal`]).
    pub fn open_read_committed(
        base_dir: PathBuf,
        wal_path: impl AsRef<Path>,
    ) -> Result<Self, StorageError> {
        let store = Self::open_read_only(base_dir)?;
        store.enable_read_journal(wal_path)?;
        Ok(store)
    }

    /// Open a read-only worker on one pool of a **shared** WAL.
    ///
    /// Identical to [`Self::open_read_committed`] except the overlay applies
    /// only frames tagged with `pool`, so a worker reading the state pool does
    /// not observe event-DAG or edges mutations that share the same
    /// segment. `wal_path` is the database root's shared `wal.bin`, not a
    /// per-pool segment.
    ///
    /// # Errors
    /// Returns [`StorageError`] if the store or journal cannot be opened.
    #[cfg(feature = "multi-reader")]
    pub fn open_read_committed_shared(
        base_dir: PathBuf,
        wal_path: impl AsRef<Path>,
        pool: crate::layout::ShardType,
    ) -> Result<Self, StorageError> {
        let store = Self::open_read_only(base_dir)?;
        store.enable_read_journal_shared(wal_path, pool)?;
        Ok(store)
    }

    /// Open a read-only worker on one pool of a shared WAL, using explicit
    /// runtime settings.
    ///
    /// # Errors
    /// Returns [`StorageError`] if opening the read-only store fails, runtime
    /// settings are invalid, or shared-WAL recovery fails.
    #[cfg(feature = "multi-reader")]
    pub fn open_read_committed_shared_with_config(
        base_dir: PathBuf,
        wal_path: impl AsRef<Path>,
        pool: crate::layout::ShardType,
        config: &RuntimeConfig,
    ) -> Result<Self, StorageError> {
        let store = Self::open_read_only_with_config(base_dir, config)?;
        store.enable_read_journal_shared(wal_path, pool)?;
        Ok(store)
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
        Self::open_with_options_locked(
            base_dir,
            cache_capacity,
            None,
            swizzle,
            writable,
            max_shard_bytes,
            compress,
            checksum_policy,
            true,
        )
    }

    /// Open a writable pool that belongs to a shared database. The database
    /// root's writer lock covers it, so the pool creates no `.mtxdb.lock`.
    pub(crate) fn open_shared_member(
        base_dir: PathBuf,
        cache_capacity: usize,
        cache_budget: Option<Arc<crate::cache::CacheBudget>>,
        max_shard_bytes: Option<u64>,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
    ) -> Result<Self, std::io::Error> {
        Self::open_with_options_locked(
            base_dir,
            cache_capacity,
            cache_budget,
            None,
            true,
            max_shard_bytes,
            compress,
            checksum_policy,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn open_with_options_locked(
        base_dir: PathBuf,
        cache_capacity: usize,
        cache_budget: Option<Arc<crate::cache::CacheBudget>>,
        swizzle: Option<SwizzleFn>,
        writable: bool,
        max_shard_bytes: Option<u64>,
        compress: bool,
        checksum_policy: packfile::ChecksumPolicy,
        take_writer_lock: bool,
    ) -> Result<Self, std::io::Error> {
        // A standalone writer creates its directory here. Shared members
        // create theirs with their first pack; readers never create anything.
        if writable && take_writer_lock {
            fs::create_dir_all(&base_dir)?;
        }
        let started = std::time::Instant::now();
        let mut timings = OpenTimings::default();

        let shard_open_started = std::time::Instant::now();
        let shards = if writable && !take_writer_lock {
            ShardPool::open_shared_member(
                base_dir.clone(),
                max_shard_bytes,
                compress,
                checksum_policy,
            )?
        } else if writable {
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
        let shard_open_time = shard_open_started.elapsed();
        timings.shard_open = shard_open_time;
        let index_config = crate::index::IndexConfig {
            seed: shards.bucket_seed(),
            ..Default::default()
        };
        if let Some(shard_timings) = shards.open_timings() {
            timings.shard_open = shard_timings.total;
            timings.shard_discovery = shard_timings.discovery;
            timings.writer_lock = shard_timings.writer_lock;
            timings.packfile_recovery = shard_timings.packfile_recovery;
            timings.packfile_recovery_calls = shard_timings.packfile_recovery_calls;
            timings.packfile_open = shard_timings.packfile_open;
            timings.packfile_open_calls = shard_timings.packfile_open_calls;
            timings.metadata_restore = shard_timings.metadata_restore;
            timings.pool_meta_restore = shard_timings.pool_meta_restore;
            timings.persisted_stats_restore = shard_timings.persisted_stats_restore;
            timings.pool_meta_persist = shard_timings.pool_meta_persist;
            timings.metadata_unattributed = shard_timings.metadata_unattributed;
            timings.shard_open_unattributed = shard_timings.unattributed;
        }

        // Keep the persisted order before scanning. New collections are
        // appended on first physical sight in the chronological scan below.
        let metadata_started = std::time::Instant::now();
        // Seed logical insertion order before scanning. A newly written collection
        // cannot appear in the sidecar until the next flush, so append it at
        // first physical sight below. A missing or unreadable sidecar yields no order and
        // therefore uses the same first-seen fallback for every collection.
        let collection_order = persisted_collection_order(&base_dir).unwrap_or_default();

        // Collect open shard info (slot, pack_id, path, length) before the scan
        // loop so we don't hold the shards read-lock across the I/O-heavy scan.
        // The lengths feed the checkpoint fingerprint (see `index::checkpoint`).
        let open_shards: Vec<(u16, PackId, PathBuf, u64)> = shards
            .all_shards()
            .into_iter()
            .map(|(id, shard)| (id, shard.pack_id, shard.path.clone(), shard.file_len()))
            .collect();

        // A checkpoint from the previous session's last sync lets us skip the
        // rescan and index rebuild entirely. It is only trusted when every
        // pack on disk still matches the fingerprint it was written against.
        let deleted_collections = Self::load_deleted_collections(&base_dir);
        let collection_generation_fences = Self::load_collection_generation_fences(&base_dir)?;
        // Every writable open that can delete a delta log (the pack-set and
        // fingerprint gates, and the rescan below) must first make the log's
        // version table durable elsewhere. Do it once, up front, so no
        // deletion site can run ahead of it.
        if writable {
            Self::promote_delta_versions(&base_dir)?;
        }
        // Capture the journal coverage *before* loading the index, so the bound
        // can never be newer than the index this handle builds. A writer writes
        // `journal.lsn` only after its covering checkpoint, so reading it first
        // always yields a bound <= the coverage of the checkpoint/scan below.
        let durable_state = Self::durable_index_state(&base_dir);
        let read_covered = Self::journal_lsn_with(&base_dir, durable_state.covered_lsn);
        // A writable open restores collection versions from the versions file
        // or a checkpoint's logical-version section. If neither exists while
        // the WAL still claims coverage, opening would report version 0 for
        // reclaimed groups, so a conditional commit expecting 0 could pass
        // against non-empty data. Fail closed instead.
        if writable
            && read_covered > 0
            // With no pack data there is nothing whose version could be lost.
            && !open_shards.is_empty()
            && crate::index::versions::read_versions(&Self::index_versions_path(&base_dir))
                .is_none()
            && crate::index::checkpoint::read_checkpoint(&Self::index_checkpoint_path(&base_dir))
                .is_none()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "no readable version baseline ({} or a checkpoint) while durable WAL coverage is {read_covered}; refusing to open a writable store with an empty version baseline",
                    Self::index_versions_path(&base_dir).display()
                ),
            ));
        }
        timings.metadata_load = metadata_started.elapsed();
        let checkpoint = Self::checkpoint_scan_out(
            &base_dir,
            cache_capacity,
            cache_budget.as_ref(),
            &shards,
            &open_shards,
            &deleted_collections,
            writable,
            ReloadMode::Strict,
            index_config,
            &mut timings,
        );
        if let Some((scan_out, collection_order, delta_state, checkpoint_covered)) = checkpoint {
            timings.path = OpenPath::Checkpoint;
            let assemble_started = std::time::Instant::now();
            let store = Self::assemble(
                shards,
                scan_out,
                collection_order,
                deleted_collections,
                collection_generation_fences,
                base_dir,
                swizzle,
                cache_capacity,
                cache_budget,
                delta_state,
                checkpoint_covered,
                read_covered,
                durable_state.fingerprint,
            );
            timings.assemble = assemble_started.elapsed();
            timings.total = started.elapsed();
            *store.last_open_timings.lock() = Some(timings);
            return Ok(store);
        }
        timings.path = OpenPath::FullScan;
        // The checkpoint/log gates rejected the persisted index state, but do
        // not sweep the delta epochs here: this open returns to the caller,
        // and a shared-WAL caller runs `enable_shared_journal` only afterwards.
        // That check reads a matching delta epoch as the pool's durable
        // collection-version baseline for history the WAL has already
        // reclaimed. Deleting it first would destroy the only bridge between
        // an old checkpoint and the reclaimed LSN and turn a recoverable open
        // into a permanent refusal, even though the epoch was valid. The
        // epoch is inert for the rescanned index; the first successful
        // checkpoint rotates a fresh one and sweeps the old epochs.

        let scan_started = std::time::Instant::now();
        let mut scan_state = StreamingOpenScan::new(collection_order);
        // The index must see physical versions in order so inserts naturally
        // replace older values and removal markers. The open_shards snapshot
        // remains slot ordered for fingerprinting; sort a separate list here.
        let mut scan_shards = shards.all_shards();
        scan_shards.sort_by_key(|(_, shard)| shard.creation_seq);
        for (slot, shard) in scan_shards {
            let tail_started = std::time::Instant::now();
            if writable {
                // Recovery can safely truncate only a torn final frame. The
                // visitor keeps no per-shard entry vector while preserving
                // that repair behavior. A semantic insert error stops this
                // scan immediately.
                packfile::try_recover_packfile_with::<StorageError, _>(&shard.path, |entry| {
                    scan_state.insert(
                        entry,
                        &shards,
                        slot,
                        shard.pack_id,
                        shard.creation_seq,
                        &ScanInsertConfig {
                            deleted_collections: &deleted_collections,
                            generation_fences: &collection_generation_fences,
                            index_config,
                        },
                    )
                })
                .map_err(|error| {
                    std::io::Error::other(format!(
                        "recovery scan failed for pack {}: {error}",
                        shard.path.display()
                    ))
                })?;
            } else {
                // Read-only mode must not truncate a possibly in-flight tail.
                let mut scan_error = None;
                {
                    let mut visit = |entry| {
                        if scan_error.is_none() {
                            if let Err(error) = scan_state.insert(
                                entry,
                                &shards,
                                slot,
                                shard.pack_id,
                                shard.creation_seq,
                                &ScanInsertConfig {
                                    deleted_collections: &deleted_collections,
                                    generation_fences: &collection_generation_fences,
                                    index_config,
                                },
                            ) {
                                scan_error = Some(error);
                            }
                        }
                    };
                    packfile::scan_packfile_from_with_removal_visit(
                        &shard.path,
                        0,
                        true,
                        &mut visit,
                    )
                    .map_err(|error| {
                        std::io::Error::new(
                            error.kind(),
                            format!(
                                "read-only scan failed for shard {slot:02x} ({}): {error}",
                                shard.path.display()
                            ),
                        )
                    })?;
                }
                if let Some(error) = scan_error {
                    return Err(error.into());
                }
            }
            timings.tail_scan = timings.tail_scan.saturating_add(tail_started.elapsed());
        }
        timings.full_scan = scan_started.elapsed();

        // The frames themselves built the indexes; this final pass only turns
        // them into the shared generation and shard-accounting structures.
        let materialization_started = std::time::Instant::now();
        let (scan_out, collection_order) =
            scan_state.finish(&shards, cache_capacity, cache_budget.as_ref());
        timings.index_materialization = materialization_started.elapsed();

        let store = Self::assemble(
            shards,
            scan_out,
            collection_order,
            deleted_collections,
            collection_generation_fences,
            base_dir,
            swizzle,
            cache_capacity,
            cache_budget,
            // A full rescan re-cooks every index from scratch, so no delta log
            // can continue anything the checkpoint recorded; force the next
            // dirty sync into a full checkpoint rewrite that re-bases the log.
            DeltaLogState::default(),
            read_covered,
            read_covered,
            durable_state.fingerprint,
        );
        // A writer that had to rescan has no checkpoint describing this pack
        // set. Mark it dirty so the next sync persists one; otherwise a store
        // opened and synced without further writes (e.g. after an aborted
        // import) would rescan every packfile on every open forever.
        if writable {
            store.index_checkpoint_dirty.store(true, Ordering::Relaxed);
        }
        timings.total = started.elapsed();
        *store.last_open_timings.lock() = Some(timings);
        Ok(store)
    }

    /// Assemble a fully constructed store from the per-collection state both
    /// the rescan path and the checkpoint fast path produce, so the two share
    /// one field-for-field constructor.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "one field-for-field constructor shared by both open paths"
    )]
    fn assemble(
        shards: ShardPool,
        scan_out: CollectionScanOutput,
        collection_order: Vec<[u8; 16]>,
        deleted_collections: HashSet<[u8; 16]>,
        collection_generation_fences: HashMap<[u8; 16], u32>,
        base_dir: PathBuf,
        swizzle: Option<SwizzleFn>,
        cache_capacity: usize,
        cache_budget: Option<Arc<crate::cache::CacheBudget>>,
        delta_state: DeltaLogState,
        read_covered: u64,
        durable_coverage: u64,
        initial_durable_fp: u64,
    ) -> Self {
        let initial_index_identity_reads = scan_out.index_identity_reads;
        let index_config = crate::index::IndexConfig {
            seed: shards.bucket_seed(),
            ..Default::default()
        };
        // Seed the per-collection refresh fingerprint from the persisted
        // checkpoint/delta state so the first miss for each collection can
        // skip refresh when nothing durable changed.
        let initial_fingerprints: HashMap<[u8; 16], u64> = collection_order
            .iter()
            .map(|&cid| (cid, initial_durable_fp))
            .collect();
        // The checkpoint coverage this open's index corresponds to, captured
        // before the index was loaded. Bound to the loaded index, never
        // re-read from disk by the overlay.
        let read_covered_lsn = AtomicU64::new(read_covered);
        Self {
            shards,
            index_tables: RwLock::new(IndexTables {
                collections: scan_out.collections,
                collection_order,
                shard_collections: scan_out.shard_collections,
                collection_shards: scan_out.collection_shards,
                collection_disk_bytes: scan_out.collection_disk_bytes,
            }),
            pinned: PinnedNodes::new(),
            base_dir,
            swizzle,
            put_locks: parking_lot::Mutex::new(HashMap::new()),
            refresh_locks: parking_lot::Mutex::new(HashMap::new()),
            last_refresh_fingerprint: parking_lot::Mutex::new(initial_fingerprints),
            initial_durable_fingerprint: initial_durable_fp,
            collection_creation: parking_lot::RwLock::new(()),
            deleted_collections: parking_lot::Mutex::new(deleted_collections),
            collection_generation_fences: parking_lot::Mutex::new(collection_generation_fences),
            deleted_collections_cache: parking_lot::Mutex::new(None),
            live_roots: RwLock::new(HashMap::new()),
            repack_threshold_entries: AtomicU64::new(DEFAULT_REPACK_THRESHOLD_ENTRIES),
            cache_capacity,
            cache_budget,
            index_config,
            read_plan: RwLock::new(ReadPlanPolicy::default()),
            index_collections: RwLock::new(std::collections::HashSet::new()),
            repack_count: AtomicU64::new(0),
            repack_kept_total: AtomicU64::new(0),
            repack_dropped_total: AtomicU64::new(0),
            repack_counts_by_collection: RwLock::new(HashMap::new()),
            repack_incremental: RwLock::new(HashMap::new()),
            last_shard_collections_flush: RwLock::new(None),
            index_checkpoint_dirty: AtomicBool::new(false),
            shard_collections_dirty: AtomicBool::new(scan_out.sidecar_missing),
            shard_collections_stale: AtomicBool::new(false),
            shard_collections_recovery_failed: AtomicBool::new(scan_out.sidecar_recovery_failed),
            shard_collections_failure_logged: AtomicBool::new(false),
            #[cfg(test)]
            recovery_pause_hook: parking_lot::Mutex::new(None),
            #[cfg(test)]
            replay_snapshot_hook: parking_lot::Mutex::new(None),
            journal: parking_lot::Mutex::new(None),
            journal_pool: std::sync::atomic::AtomicU8::new(0),
            journal_recovery: parking_lot::Mutex::new(Vec::new()),
            replaying: AtomicBool::new(false),
            read_journal: Arc::new(parking_lot::Mutex::new(None)),
            read_snapshot_gate: Arc::new(parking_lot::Mutex::new(())),
            parked_transaction_overlay: parking_lot::Mutex::new(None),
            transaction_overlay_lifecycle: parking_lot::Mutex::new(()),
            transaction_overlay_users: AtomicU64::new(0),
            #[cfg(test)]
            transaction_overlay_scanned: AtomicU64::new(0),
            read_covered_lsn,
            materialized_lsn: AtomicU64::new(read_covered),
            durable_coverage: Arc::new(AtomicU64::new(durable_coverage)),
            checkpoint_worker: parking_lot::Mutex::new(None),
            background_checkpoint: AtomicBool::new(false),
            #[cfg(test)]
            checkpoint_tail_hook: parking_lot::Mutex::new(None),
            #[cfg(test)]
            checkpoint_prepare_fail: AtomicBool::new(false),
            read_reloads: AtomicU64::new(0),
            read_reload_failures: AtomicU64::new(0),
            read_refreshes: AtomicU64::new(0),
            read_refresh_bytes: AtomicU64::new(0),
            read_refresh_sleeps: AtomicU64::new(0),
            read_refresh_sleep_micros: AtomicU64::new(0),
            last_open_timings: parking_lot::Mutex::new(None),
            last_sync_timings: parking_lot::Mutex::new(None),
            #[cfg(test)]
            delta_log_cap_override: AtomicU64::new(0),
            last_checkpoint_breakdown: parking_lot::Mutex::new(None),
            sync_totals: SyncTotals::default(),
            sync_diagnostics: parking_lot::Mutex::new(SyncDiagnostics::default()),
            publish_calls: AtomicU64::new(0),
            publish_time_ns: AtomicU64::new(0),
            pending_publish_since: parking_lot::Mutex::new(None),
            publish_generation: AtomicU64::new(0),
            checkpoint_rewrite_min_interval_ns: AtomicU64::new(0),
            pack_fsync_budget_bytes: AtomicU64::new(DEFAULT_PACK_FSYNC_BUDGET_BYTES),
            delta_fsync_budget_bytes: AtomicU64::new(DEFAULT_DELTA_FSYNC_BUDGET_BYTES),
            checkpoint_rewrite_max_bytes: AtomicU64::new(0),
            last_checkpoint_rewrite_at: parking_lot::Mutex::new(None),
            checkpoint_bytes_at_last_rewrite: AtomicU64::new(0),
            checkpoint_skips: AtomicU64::new(0),
            stats_enabled: AtomicBool::new(false),
            operation_timings: OperationTimings::default(),
            refresh_on_miss: AtomicBool::new(true),
            open_count: AtomicU64::new(1),
            get_calls: AtomicU64::new(0),
            get_misses: AtomicU64::new(0),
            get_many_calls: AtomicU64::new(0),
            get_many_records: AtomicU64::new(0),
            get_many_misses: AtomicU64::new(0),
            miss_refreshes: AtomicU64::new(0),
            miss_refresh_skips: AtomicU64::new(0),
            miss_refresh_recovered: AtomicU64::new(0),
            miss_refresh_retry_ids: AtomicU64::new(0),
            miss_refresh_fp_errors: AtomicU64::new(0),
            index_candidates: AtomicU64::new(0),
            candidate_reads: AtomicU64::new(0),
            candidate_hash_mismatches: AtomicU64::new(0),
            get_many_shards_touched: AtomicU64::new(0),
            candidate_frame_bytes: AtomicU64::new(0),
            read_many_runs: AtomicU64::new(0),
            read_many_span_bytes: AtomicU64::new(0),
            read_plan_extents: AtomicU64::new(0),
            read_plan_prefetch_bytes: AtomicU64::new(0),
            read_plan_skipped_extents: AtomicU64::new(0),
            put_calls: AtomicU64::new(0),
            put_bytes: AtomicU64::new(0),
            put_many_calls: AtomicU64::new(0),
            put_many_records: AtomicU64::new(0),
            put_many_bytes: AtomicU64::new(0),
            put_many_fast_path_calls: AtomicU64::new(0),
            put_many_clone_path_calls: AtomicU64::new(0),
            index_clone_time_ns: AtomicU64::new(0),
            index_grow_count: AtomicU64::new(0),
            index_rebuild_count: AtomicU64::new(0),
            index_identity_reads: AtomicU64::new(initial_index_identity_reads),
            delta_invalidations: AtomicU64::new(0),
            sync_calls: AtomicU64::new(0),
            checkpoint_writes: AtomicU64::new(0),
            checkpoint_tails_started: AtomicU64::new(0),
            syncs_with_tail_in_flight: AtomicU64::new(0),
            syncs_waited_for_tail: AtomicU64::new(0),
            delta_appends: AtomicU64::new(0),
            sidecar_writes: AtomicU64::new(0),
            delta_state: parking_lot::Mutex::new(delta_state),
            index_persist_lock: parking_lot::Mutex::new(()),
        }
    }
}
