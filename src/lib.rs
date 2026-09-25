//! mtxdb: a write-once, content-addressed packfile storage engine for Matrix.
//!
//! Never mutate, never delete, never tombstone. Every node is
//! content-addressed and append-only; garbage collection is a background
//! repack that rewrites only reachable data in traversal order.
//!
//! Licensed under either of MIT or Apache-2.0, at your option. See
//! `LICENSE-MIT` and `LICENSE-APACHE`.
//!
//! Copyright (c) 2026 Shane Jaroch <chown_tee@proton.me>

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(clippy::module_name_repetitions)]

/// Named application-owned lookup indexes sharing the normal packfile engine.
pub mod auxiliary;
/// Verify-once decoded-node cache with O(1) LRU eviction, plus a pinned-node set.
pub mod cache;
/// Compressed sparse row graph for deterministic topological ordering.
pub mod csr;
/// In-memory dependency DAG used to track unresolved node references.
pub mod dag;
/// Root-level handle that opens every pool behind one shared WAL fence.
#[cfg(feature = "multi-reader")]
pub mod database;
/// Frontier tracking for nodes awaiting their dependencies before being writable.
pub mod frontier;
/// Lossy, append-only index mapping content hashes to packfile locations.
pub mod index;
/// Checksummed append-only journal primitives for durable group commits.
///
/// This module itself stays always-on: the write-ahead journal
/// (`enable_journal`/`replay_journal`) is a same-process durability/group-
/// commit accelerator any embedded deployment can opt into. Only the
/// cross-process *read-committed overlay* built on top of it — letting a
/// separate OS process observe a live writer's committed-but-not-yet-
/// checkpointed data — is gated behind the `multi-reader` feature; see
/// `packfile::storage::read_journal` and Cargo.toml's `multi-reader` doc
/// comment.
pub mod journal;
/// Database-root layout and named independent packfile pools.
pub mod layout;
/// Matrix-specific room-version policy.
pub mod matrix_policy;
/// On-disk packfile format and the storage engine built on top of it.
pub mod packfile;
/// Fixed-size shard file pool that packfiles are written into.
pub mod shard;
/// Core node/storage types and the top-level `StorageEngine`.
pub mod storage;
/// Executable policy primitives for application collection templates.
pub mod template;

pub use auxiliary::{
    auxiliary_collection_id, auxiliary_key_digest, AuxiliaryIndex, AuxiliaryKeyDigest,
};
pub use cache::NodeCache;
#[cfg(feature = "multi-reader")]
pub use database::{DatabaseTransaction, PoolPolicies, PoolPolicy, SharedDatabase};
pub use index::LossyIndex;
#[cfg(feature = "multi-reader")]
pub use journal::SharedWalLock;
pub use journal::{DurabilityStats, DurabilityToken, DurableWaitLatency, GroupCommitConfig};
#[cfg(feature = "multi-reader")]
pub use journal::{TxnStage, TxnStageState};
pub use layout::{enclosing_root, read_wal_layout, DatabaseLayout, ShardType, WalLayout};
#[cfg(feature = "multi-reader")]
pub use matrix_policy::matrix_pool_policies;
pub use matrix_policy::{
    EventIdPolicy, MatrixRoomVersion, RedactionPolicy, ReferenceHashEncoding,
    ReferenceHashInputPolicy, RoomIdPolicy, RoomMetadata, StateResolutionPolicy,
};
pub use packfile::{
    storage::{OperationLatency, PackfileStorage, ReadPlanPolicy},
    FrameMetadata, Record,
};
pub use shard::{LockHolderInfo, ShardPool};
pub use storage::{
    content_digest, Digest32, DigestAlgorithm, DigestHasher, NodeData, NodeId, StorageEngine,
};
#[allow(deprecated)]
pub use template::{
    derive_collection_id, derive_group_full_id, derive_group_member_collection_id,
    derive_group_member_full_id, derive_member_collection_id_from_group,
    derive_member_full_id_from_group, frame_digest, namespace_bias, record_logical_id,
    try_derive_collection_full_id, try_derive_collection_id, wrapping_add_le, CollectionKeyRule,
    CollectionMetadata, CollectionTemplate, EstablishmentRule, FrameIdInput, FrameIdPolicy,
    PayloadPolicy, RecordIdentityRule, COLLECTION_METADATA_RECORD_ID,
    COLLECTION_TEMPLATE_FORMAT_V1, MEMBER_NAMESPACE_AUTH, MEMBER_NAMESPACE_EVNT,
    MEMBER_NAMESPACE_INTL, MEMBER_NAMESPACE_PREV, MEMBER_NAMESPACE_STAT, NAMESPACE_BIAS_AUTH,
    NAMESPACE_BIAS_EVNT, NAMESPACE_BIAS_INTL, NAMESPACE_BIAS_PREV, NAMESPACE_BIAS_STAT,
    POOL_DST_AUTH, POOL_DST_EVNT, POOL_DST_INTERNAL, POOL_DST_PREV, POOL_DST_STAT,
};
