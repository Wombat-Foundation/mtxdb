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

/// Verify-once decoded-node cache with O(1) LRU eviction, plus a pinned-node set.
pub mod cache;
/// Compressed sparse row graph for deterministic topological ordering.
pub mod csr;
/// In-memory dependency DAG used to track unresolved node references.
pub mod dag;
/// Frontier tracking for nodes awaiting their dependencies before being writable.
pub mod frontier;
/// Lossy, append-only index mapping content hashes to packfile locations.
pub mod index;
/// On-disk packfile format and the storage engine built on top of it.
pub mod packfile;
/// Fixed-size shard file pool that packfiles are written into.
pub mod shard;
/// Core node/storage types and the top-level `StorageEngine`.
pub mod storage;

pub use cache::NodeCache;
pub use index::LossyIndex;
pub use packfile::{storage::PackfileStorage, Record};
pub use shard::ShardPool;
pub use storage::{
    is_known_tag, tag_name, NodeData, NodeId, StorageEngine, ENTRY_TYPE_AUTH_CHAIN_LINKS,
    ENTRY_TYPE_EVENT_JSON, ENTRY_TYPE_EVENT_STATE_GROUP, ENTRY_TYPE_GENERIC_KV,
    ENTRY_TYPE_HAMT_NODE, ENTRY_TYPE_HAMT_ROOT_TYPED, ENTRY_TYPE_PREV_EVENT_EDGES,
    ENTRY_TYPE_STATE_GROUP_REFCOUNT,
};
