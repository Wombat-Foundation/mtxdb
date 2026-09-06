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

pub mod cache;
pub mod dag;
pub mod frontier;
pub mod index;
pub mod packfile;
pub mod repack;
pub mod shard;
pub mod storage;

pub use cache::NodeCache;
pub use index::LossyIndex;
pub use packfile::{storage::PackfileStorage, Record};
pub use shard::ShardPool;
pub use storage::{NodeData, NodeId, StorageEngine};
