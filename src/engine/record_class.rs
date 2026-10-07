//! Per-record storage policy: version retention, write durability, and stream
//! ordering.
//!
//! The engine has one physical append mechanism plus atomic index swaps. A
//! [`RecordClass`] selects the *policy* layered on top of it, so write-once
//! content-addressed data (events, DAG edges, HAMT nodes) and update-heavy data
//! (presence, receipts, account data) can share the same engine without a
//! second physical pool format. Only retention, durability, and ordering differ
//! per class; mutation is always a logical version swap, never a physical frame
//! overwrite.
//!
//! This module is protocol-neutral. The mapping from Matrix storage workloads to
//! record classes lives in [`crate::matrix_policy`].

/// How many versions of a record are retained after an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// The newest value only, not retained across restarts (presence heartbeats).
    EphemeralLatest,
    /// The newest value only, retained across restarts.
    LatestOnly,
    /// The newest `n` versions.
    Bounded(u32),
    /// The complete version history.
    Versioned,
    /// Write-once; distinct content is a distinct record and is never superseded.
    Immutable,
}

/// When a write is guaranteed durable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// No synchronous flush; losing the most recent writes on crash is acceptable.
    Volatile,
    /// Flush batched with other writers before acknowledging.
    GroupCommit,
    /// Flush before acknowledging this write.
    Synchronous,
}

/// Which orderings a record participates in.
///
/// Orderings compose: events are both causally ordered (for DAG/state
/// resolution) and stream-ordered (for `/sync` pagination), while receipts are
/// stream-ordered without forming a causal DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingPolicy {
    /// No ordering; records are addressed by logical id only.
    None,
    /// Topologically ordered for DAG/state-resolution traversal.
    Topological,
    /// Assigned a monotonic stream position for ordered range scans.
    Stream,
    /// Both topologically ordered and stream-ordered.
    TopologicalAndStream,
}

impl OrderingPolicy {
    /// Whether this policy includes topological ordering.
    #[must_use]
    pub const fn is_topological(self) -> bool {
        matches!(self, Self::Topological | Self::TopologicalAndStream)
    }

    /// Whether this policy includes stream ordering.
    #[must_use]
    pub const fn is_streamed(self) -> bool {
        matches!(self, Self::Stream | Self::TopologicalAndStream)
    }
}

/// The retention, durability, and ordering policy for one class of record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordClass {
    /// How many versions to retain.
    pub retention: Retention,
    /// When writes become durable.
    pub durability: Durability,
    /// Whether the record is ordered in a stream.
    pub ordering: OrderingPolicy,
}

impl RecordClass {
    /// Construct a record class from its three policy axes.
    #[must_use]
    pub const fn new(
        retention: Retention,
        durability: Durability,
        ordering: OrderingPolicy,
    ) -> Self {
        Self {
            retention,
            durability,
            ordering,
        }
    }

    /// Whether an update to an existing record supersedes the previous version
    /// rather than adding a distinct record.
    #[must_use]
    pub const fn supersedes(&self) -> bool {
        !matches!(self.retention, Retention::Immutable)
    }

    /// Whether more than the newest version of a key may be retained.
    #[must_use]
    pub const fn retains_prior_versions(&self) -> bool {
        match self.retention {
            Retention::Bounded(n) => n > 1,
            Retention::Versioned => true,
            Retention::EphemeralLatest | Retention::LatestOnly | Retention::Immutable => false,
        }
    }

    /// The maximum number of live versions for a key, if bounded.
    #[must_use]
    pub const fn version_limit(&self) -> Option<u32> {
        match self.retention {
            Retention::EphemeralLatest | Retention::LatestOnly | Retention::Immutable => Some(1),
            Retention::Bounded(n) => Some(n),
            Retention::Versioned => None,
        }
    }

    /// Whether a write is flushed before it is acknowledged.
    #[must_use]
    pub const fn is_synchronous(&self) -> bool {
matches!(self.durability, Durability::GroupCommit | Durability::Synchronous)
    }

    /// Whether the most recent writes may be lost on a crash.
    #[must_use]
    pub const fn is_volatile(&self) -> bool {
        matches!(self.durability, Durability::Volatile)
    }

    /// Whether the record is topologically ordered.
    #[must_use]
    pub const fn is_topological(&self) -> bool {
        self.ordering.is_topological()
    }

    /// Whether the record is assigned a monotonic stream position.
    #[must_use]
    pub const fn is_streamed(&self) -> bool {
        self.ordering.is_streamed()
    }
}

#[cfg(test)]
#[path = "test_record_class.rs"]
mod tests;
