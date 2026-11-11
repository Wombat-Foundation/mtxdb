//! mtxdb: a write-once, content-addressed packfile storage engine for Matrix.
//!
//! Records are append-only versions, including explicit removal versions;
//! physical reclamation happens during background repack, which rewrites only
//! reachable data in traversal order.
//!
//! # Removal
//!
//! Removing a key (`DatabaseTransaction::remove` / `remove_many`) appends a new
//! version of the record: an empty frame carrying `FLAG_REMOVED`. It is not a
//! tombstone in the index or the redo/delta logs; the index entry simply points
//! at the newest version, and every read, scan and count of live records treats
//! a removed version as absent. An empty payload alone is an ordinary live
//! value. Removing a key that is not live writes nothing and leaves the
//! collection version unchanged, and a later put of the same key supersedes the
//! removal.
//!
//! Space comes back only at repack. A repack drops a removal marker once no
//! older copy of the key can survive, and carries it forward while a retained
//! shard (one another collection still references) may still hold the older
//! value, so a rebuild from the shards never brings the key back. The order of
//! versions is the pack's creation sequence and the frame's offset, never the
//! runtime shard slot or a journal LSN.
//!
//! # Compatibility
//!
//! Removal is a one-way format addition. A pack that holds a removal frame sets
//! a frame flag that builds without removal support reject (an unknown frame
//! flag is an error, never skipped), and a journal group that holds a removal
//! uses a mutation kind those builds reject as malformed. Either way an older
//! build refuses to open the store rather than ignore a removal or read the
//! older value; it does not truncate the journal. A store that has never
//! recorded a removal remains readable by older builds.
//!
//! Licensed under either of MIT or Apache-2.0, at your option. See
//! `LICENSE-MIT` and `LICENSE-APACHE`.
//!
//! Copyright (c) 2026 Shane Jaroch <chown_tee@proton.me>

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(clippy::module_name_repetitions)]
// Stable rustdoc needs the explicit targets (module docs render at the re-export
// site); nightly resolves the bare labels and flags them as redundant.
#![allow(rustdoc::redundant_explicit_links)]

// On disk the source is grouped by layer, with each module's tests beside it:
//   engine/  the storage engine: packfiles, shards, journal, database, layout,
//            the hash index, templates and record classes;
//   graph/   ordered, adjacency and closure primitives built on the engine:
//            short ids, closure generations, logical heads, bitmap sets, the
//            timeline, auxiliary indexes and the graph helpers;
//   matrix/  the Matrix-specific adapters: event adjacency, room-version
//            policy, state groups, auth closures and the room auth facade.
// The module tree stays flat: each module below names its file with `#[path]`,
// so every `crate::journal::...` and `mtxdb::journal::...` path is unchanged.

/// Transitive auth closures: pure in-memory computation plus the persisted
/// generation layer, both under `bitmaps`.
#[cfg(feature = "bitmaps")]
#[path = "matrix/auth_closure.rs"]
pub mod auth_closure;
/// Named application-owned lookup indexes sharing the normal packfile engine.
#[path = "graph/auxiliary.rs"]
pub mod auxiliary;
/// Domain-tagged `u32` bitmap sets with generic set algebra.
#[cfg(feature = "bitmaps")]
#[path = "graph/bitmap_set.rs"]
pub mod bitmap_set;
/// Verify-once decoded-node cache with O(1) LRU eviction, plus a pinned-node set.
#[path = "engine/cache/mod.rs"]
pub mod cache;
/// Persisted closure generations published through a logical head.
#[path = "graph/closure_store.rs"]
pub mod closure_store;
/// Explicit runtime configuration and the opt-in `MTXDB_*` environment reader.
#[path = "engine/config.rs"]
pub mod config;
/// Compressed sparse row graph for deterministic topological ordering.
#[path = "graph/csr/mod.rs"]
pub mod csr;
/// Matrix event adjacency (`prev`, `auth`, relations) over the short-id primitives.
#[path = "matrix/csr_adjacency.rs"]
pub mod csr_adjacency;
/// In-memory dependency DAG used to track unresolved node references.
#[path = "graph/dag/mod.rs"]
pub mod dag;
/// Root-level handle that opens every pool behind one shared WAL fence.
#[path = "engine/database.rs"]
pub mod database;
#[path = "matrix/export_schema.rs"]
pub mod export_schema;
/// Frontier tracking for nodes awaiting their dependencies before being writable.
#[path = "graph/frontier.rs"]
pub mod frontier;
/// Persisted, content-addressed HAMTs in an mtxdb collection, generic over
/// key and value types; nodes are verified on read and routed canonically.
#[path = "graph/hamt_store.rs"]
pub mod hamt_store;
/// Lossy, append-only index mapping content hashes to packfile locations.
#[path = "engine/index/mod.rs"]
pub mod index;
/// Shared on-disk layouts for the server/importer boundary: verdicts,
/// signing keys, and the import marker.
#[path = "matrix/interop.rs"]
pub mod interop;
/// Platform-specific positioned, vectored, and advisory I/O primitives.
#[path = "engine/io.rs"]
mod io;
/// Checksummed append-only journal primitives for durable group commits.
///
/// This module itself stays always-on: the write-ahead journal
/// (`enable_journal`/`replay_journal`) is a same-process durability/group-
/// commit accelerator any embedded deployment can opt into. The cross-process
/// *read-committed overlay* built on top of it — letting a separate OS process
/// observe a live writer's committed-but-not-yet-checkpointed data — is always
/// available too; see `packfile::storage::read_journal`.
#[path = "engine/journal.rs"]
pub mod journal;
/// Content-addressed state lattice checkpoint/chain codec.
#[cfg(feature = "state-lattice")]
#[path = "matrix/lattice.rs"]
pub mod lattice;
/// Database-root layout and named independent packfile pools.
#[path = "engine/layout.rs"]
pub mod layout;
/// Stable logical-ID to current physical-record pointers.
#[path = "graph/logical_head.rs"]
pub mod logical_head;
/// Matrix-specific room-version policy.
#[path = "matrix/matrix_policy.rs"]
pub mod matrix_policy;
/// On-disk packfile format and the storage engine built on top of it.
#[path = "engine/packfile/mod.rs"]
pub mod packfile;
/// A room's reconciliation population, pinned from the owner log.
#[cfg(feature = "reconcile")]
#[path = "graph/population.rs"]
pub mod population;
/// Per-record retention, durability, and ordering policy.
#[path = "engine/record_class.rs"]
pub mod record_class;
/// Public façade and error surface over a room's auth closures.
#[cfg(feature = "bitmaps")]
#[path = "matrix/room_auth.rs"]
pub mod room_auth;
/// Room-scoped flag and disposition bitmap index.
#[cfg(feature = "bitmaps")]
#[path = "matrix/room_flag.rs"]
pub mod room_flag;
/// Rebuildable full-text and field secondary indexes.
#[path = "search.rs"]
pub mod search;
/// Federation signing keys: per-key records, the MSC4499 binding rules, and
/// the verbatim Notary payload.
#[path = "matrix/server_keys.rs"]
pub mod server_keys;
/// Fixed-size shard file pool that packfiles are written into.
#[path = "engine/shard.rs"]
pub mod shard;
/// Room-scoped dense `u32` short ids and compact adjacency lists.
#[path = "graph/short_id.rs"]
pub mod short_id;
/// Matrix state-group instance identity and record layout.
#[path = "matrix/state_group.rs"]
pub mod state_group;
/// Matrix room state HAMTs built on [`hamt_store`]: the `MTHR`/`MTHN` records
/// in a room's `STAT` collection, shared by the importer and server
/// integrations.
#[path = "matrix/state_hamt.rs"]
pub mod state_hamt;
/// Core node/storage types and the top-level `StorageEngine`.
#[path = "engine/storage.rs"]
pub mod storage;
/// Executable policy primitives for application collection templates.
#[path = "engine/template.rs"]
pub mod template;
/// Persisted, ordered timeline index with copy-on-write roots.
#[path = "graph/timeline.rs"]
pub mod timeline;

#[cfg(all(test, feature = "bitmaps"))]
#[path = "matrix/test_auth_closure.rs"]
mod test_auth_closure;
#[cfg(all(test, feature = "bitmaps"))]
#[path = "graph/test_bitmap_set.rs"]
mod test_bitmap_set;
#[cfg(test)]
#[path = "graph/test_closure_store.rs"]
mod test_closure_store;
#[cfg(test)]
#[path = "matrix/test_csr_adjacency.rs"]
mod test_csr_adjacency;
#[cfg(test)]
#[path = "graph/test_hamt_store.rs"]
mod test_hamt_store;
#[cfg(test)]
#[path = "graph/test_logical_head.rs"]
mod test_logical_head;
#[cfg(test)]
#[cfg(feature = "reconcile")]
#[path = "graph/test_population.rs"]
mod test_population;
#[cfg(all(test, feature = "bitmaps"))]
#[path = "matrix/test_room_auth.rs"]
mod test_room_auth;
#[cfg(all(test, feature = "bitmaps"))]
#[path = "matrix/test_room_flag.rs"]
mod test_room_flag;
#[cfg(test)]
mod test_scratch;
#[cfg(test)]
#[path = "test_search.rs"]
mod test_search;
#[cfg(test)]
#[path = "graph/test_short_id.rs"]
mod test_short_id;

pub use auxiliary::{
    auxiliary_collection_id, auxiliary_key_digest, AuxiliaryIndex, AuxiliaryKeyDigest,
};
#[cfg(feature = "bitmaps")]
pub use bitmap_set::{BitmapSet, DomainTag, BITMAP_SET_FORMAT_VERSION, BITMAP_SET_MAGIC};
pub use cache::NodeCache;
pub use config::{ConfigError, RuntimeConfig};
pub use database::{
    CommitPhaseStats, Database, DatabaseTransaction, PhaseTiming, PoolPolicies, PoolPolicy,
};
pub use index::LossyIndex;
pub use interop::{
    commit_state_batches, decode_import_marker, decode_server_key, decode_state_root_handle_v1,
    decode_verdict, encode_import_marker, encode_server_key, encode_state_root_handle_v1,
    encode_verdict, event_record_id, event_state_root_node_id, event_verdict_collection_id,
    import_marker_collection_id, import_marker_node_id, notary_record_id, room_flag_collection_id,
    room_roots_collection_id, room_state_collection_id, room_state_root_node_id,
    server_keys_collection_id, state_hamt_node_id, ImportMarker, StateRootHandleV1, Verdict,
    DATA_FORMAT, KEY_RECORD_VERSION, MARKER_VERSION, STATE_BATCH_BYTES, STATE_POOL,
    STATE_ROOT_HANDLE_V1_LEN, VERDICT_BUCKETS, VERDICT_VERSION,
};
pub use journal::SharedWalLock;
pub use journal::{
    CollectionExpectation, ReplayLeaseRevoked, StagedLookup, TxnStage, TxnStageState,
};
pub use journal::{
    DurabilityStats, DurabilityToken, DurableWaitLatency, GroupCommitConfig, GroupDirectoryStats,
};
pub use layout::{enclosing_root, is_database_root, DatabaseLayout, ShardType};
pub use logical_head::{
    decode_logical_head, encode_logical_head, LogicalHead, LogicalHeadRead, LogicalHeadValue,
    LOGICAL_HEAD_MAGIC, LOGICAL_HEAD_VERSION,
};
pub use matrix_policy::matrix_pool_policies;
pub use matrix_policy::{
    EventIdPolicy, MatrixRecordClass, MatrixRoomVersion, RedactionPolicy, ReferenceHashEncoding,
    ReferenceHashInputPolicy, RoomIdPolicy, RoomMetadata, StateResolutionPolicy,
};
pub use packfile::{
    storage::{OperationLatency, PackfileStorage, ReadPlanPolicy},
    FrameMetadata, Record,
};
pub use record_class::{Durability, OrderingPolicy, RecordClass, Retention};
/// Generic, content-addressed persistent HAMT (CHAMP) from `rezzy`.
///
/// Keys and values are any types with a [`hamt::HamtCodec`] encoding. Nodes are
/// immutable records addressed by a keyed BLAKE3 structural hash, so they store
/// directly as mtxdb records; path-copying writes share every unchanged
/// subtree, [`hamt::PersistedHamtReader`] verifies each node it loads, and
/// [`hamt::diff_hamt_nodes`] skips identical subtrees by hash.
/// [`hamt_store`] persists these trees in mtxdb collections, and
/// [`state_hamt`] is the Matrix room-state layer on top.
pub use rezzy::hamt;
#[cfg(feature = "reconcile")]
pub use rezzy_recon as recon;
#[cfg(feature = "bitmaps")]
pub use room_flag::{
    flag_chunk_node_id, Flag, FlagChange, FlagQuery, RoomFlagIndex, CHUNK_CAPACITY, FLAG_META_ID,
    FLAG_META_MAGIC, FLAG_META_VERSION,
};
pub use search::{SearchDocument, SearchIndexes, SearchQuery};
pub use shard::{LockHolderInfo, ShardPool};
pub use state_group::{
    decode_state_group_record, encode_state_group_record, state_group_collection_id,
    state_group_instance_full_id, state_group_instance_id, StateGroupDerivation, StateGroupId,
    StateGroupInstance, StateGroupRelation, StateGroupResolver, StateGroupTrust,
    STATE_GROUP_DOMAIN_PREFIX, STATE_GROUP_RECORD_MAGIC, STATE_GROUP_RECORD_VERSION,
    STATE_LATTICE_KEYFRAME_HOPS,
};
pub use storage::{
    content_digest, Digest32, DigestAlgorithm, DigestHasher, NodeData, NodeId, StorageEngine,
};
pub use template::{
    derive_collection_id, derive_group_full_id, derive_group_member_collection_id,
    derive_group_member_full_id, derive_member_collection_id_from_group,
    derive_member_full_id_from_group, frame_digest, namespace_bias, record_logical_id,
    try_derive_collection_full_id, try_derive_collection_id, wrapping_add_le, CollectionKeyRule,
    CollectionMetadata, CollectionTemplate, EstablishmentRule, FrameIdInput, FrameIdPolicy,
    PayloadPolicy, RecordIdentityRule, COLLECTION_METADATA_RECORD_ID,
    COLLECTION_TEMPLATE_FORMAT_V1, MEMBER_NAMESPACE_AUTH, MEMBER_NAMESPACE_EVNT,
    MEMBER_NAMESPACE_FLAG, MEMBER_NAMESPACE_FWD, MEMBER_NAMESPACE_INTL, MEMBER_NAMESPACE_PREV,
    MEMBER_NAMESPACE_STAT, MEMBER_NAMESPACE_STGP, NAMESPACE_BIAS_AUTH, NAMESPACE_BIAS_EVNT,
    NAMESPACE_BIAS_FLAG, NAMESPACE_BIAS_FWD, NAMESPACE_BIAS_INTL, NAMESPACE_BIAS_PREV,
    NAMESPACE_BIAS_STAT, NAMESPACE_BIAS_STGP,
};
pub use timeline::{
    event_id_ref, room_scope_id, timeline_collection_id, EventRef, TimelineCursor, TimelineEntry,
    TimelineIndex, TimelinePage, TIMELINE_FORMAT_VERSION, TIMELINE_LEAF_CAP,
};

#[cfg(test)]
mod test_trace;
