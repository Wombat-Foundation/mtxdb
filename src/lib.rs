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
/// Compressed sparse row graph for deterministic topological ordering.
#[path = "graph/csr/mod.rs"]
pub mod csr;
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
/// Lossy, append-only index mapping content hashes to packfile locations.
#[path = "engine/index/mod.rs"]
pub mod index;
/// Shared on-disk layouts for the server/importer boundary: verdicts,
/// signing keys, and the import marker.
#[path = "matrix/interop.rs"]
pub mod interop;
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
/// Database-root layout and named independent packfile pools.
#[path = "engine/layout.rs"]
pub mod layout;
/// Stable logical-ID to current physical-record pointers.
#[path = "graph/logical_head.rs"]
pub mod logical_head;
/// Matrix event adjacency (`prev`, `auth`, relations) over the short-id primitives.
#[path = "matrix/matrix_adjacency.rs"]
pub mod matrix_adjacency;
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
/// Rebuildable full-text and field secondary indexes.
#[path = "search.rs"]
pub mod search;
/// Fixed-size shard file pool that packfiles are written into.
#[path = "engine/shard.rs"]
pub mod shard;
/// Room-scoped dense `u32` short ids and compact adjacency lists.
#[path = "graph/short_id.rs"]
pub mod short_id;
/// Matrix state-group instance identity and record layout.
#[path = "matrix/state_group.rs"]
pub mod state_group;
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
#[path = "graph/test_logical_head.rs"]
mod test_logical_head;
#[cfg(test)]
#[path = "matrix/test_matrix_adjacency.rs"]
mod test_matrix_adjacency;
#[cfg(test)]
#[cfg(feature = "reconcile")]
#[path = "graph/test_population.rs"]
mod test_population;
#[cfg(all(test, feature = "bitmaps"))]
#[path = "matrix/test_room_auth.rs"]
mod test_room_auth;
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
#[allow(deprecated)]
pub use database::SharedDatabase;
pub use database::{
    CommitPhaseStats, Database, DatabaseTransaction, PhaseTiming, PoolPolicies, PoolPolicy,
};
pub use index::LossyIndex;
pub use interop::{
    auth_closure_record_id, auth_counter_record_id, auth_forward_record_id, auth_reverse_record_id,
    commit_state_batches, decode_auth_counter, decode_auth_forward, decode_import_marker,
    decode_server_key, decode_state_root_handle_v1, decode_verdict, encode_auth_counter,
    encode_auth_forward, encode_import_marker, encode_server_key, encode_state_root_handle_v1,
    encode_verdict, event_record_id, event_state_root_node_id, event_verdict_collection_id,
    import_marker_collection_id, import_marker_node_id, room_roots_collection_id,
    room_state_collection_id, room_state_root_node_id, server_key_record_id,
    server_keys_collection_id, state_hamt_node_id, ImportMarker, KeyRecordKind, StateRootHandleV1,
    Verdict, DATA_FORMAT, KEY_RECORD_VERSION, MARKER_VERSION, STATE_BATCH_BYTES, STATE_POOL,
    STATE_ROOT_HANDLE_V1_LEN, VERDICT_BUCKETS, VERDICT_VERSION,
};
pub use journal::SharedWalLock;
pub use journal::{CollectionExpectation, StagedLookup, TxnStage, TxnStageState};
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
#[cfg(feature = "reconcile")]
pub use rezzy_recon as recon;
pub use search::{SearchDocument, SearchIndexes, SearchQuery};
pub use shard::{LockHolderInfo, ShardPool};
pub use state_group::{
    decode_state_group_record, encode_state_group_record, state_group_collection_id,
    state_group_instance_full_id, state_group_instance_id, StateGroupDerivation, StateGroupId,
    StateGroupInstance, StateGroupRelation, StateGroupResolver, StateGroupTrust,
    STATE_GROUP_DOMAIN_PREFIX, STATE_GROUP_RECORD_MAGIC, STATE_GROUP_RECORD_VERSION,
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
    MEMBER_NAMESPACE_FWD, MEMBER_NAMESPACE_INTL, MEMBER_NAMESPACE_PREV, MEMBER_NAMESPACE_STAT,
    MEMBER_NAMESPACE_STGP, NAMESPACE_BIAS_AUTH, NAMESPACE_BIAS_EVNT, NAMESPACE_BIAS_FWD,
    NAMESPACE_BIAS_INTL, NAMESPACE_BIAS_PREV, NAMESPACE_BIAS_STAT, NAMESPACE_BIAS_STGP,
};
pub use timeline::{
    timeline_collection_id, EventRef, TimelineCursor, TimelineEntry, TimelineIndex, TimelinePage,
    TIMELINE_FORMAT_VERSION, TIMELINE_LEAF_CAP,
};
