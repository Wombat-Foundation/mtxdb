//! Matrix-specific room-version policy used by the Matrix import adapter.
//!
//! This is deliberately separate from the protocol-neutral storage engine:
//! packs and collection templates carry no Matrix semantics, while redaction,
//! authorization, and room-version rules are Matrix wire semantics.
//!
//! The current archive importer does not yet evaluate every policy below. They
//! remain together here because a Matrix adapter must use one coherent,
//! versioned policy table when verification and state handling are added.

#![allow(
    dead_code,
    unreachable_pub,
    reason = "the Matrix adapter keeps its complete version-policy table ahead of optional verification/state features"
)]

use crate::record_class::{Durability, OrderingPolicy, RecordClass, Retention};

/// The validation claim made for imported Matrix source records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationProfile {
    /// Parse, identify, namespace, and retain the source object.
    Archive,
    /// Additionally verify the protocol identity and signatures.
    Verified,
    /// Additionally evaluate authorization and resolve state.
    StateResolution,
}

/// How a Matrix room version assigns event identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventIdPolicy {
    /// A homeserver selected the event ID; it cannot be recomputed from JSON.
    ServerAssigned,
    /// The event ID is a SHA-256 reference hash of canonical redacted JSON.
    ReferenceHash,
}

/// Wire encoding of a reference-hash event ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceHashEncoding {
    /// Event IDs are assigned by a server rather than encoded from a hash.
    NotApplicable,
    /// Version 3: unpadded standard Base64.
    StandardBase64NoPad,
    /// Version 4 and later: unpadded URL-safe Base64.
    UrlSafeBase64NoPad,
}

/// What fields are stripped before calculating a reference-hash event ID.
///
/// A reference hash is taken over the *redacted* event — the server-server
/// API's "Calculating the reference hash for an event" puts the event through
/// the redaction algorithm as its first step — so this boundary must track
/// [`RedactionPolicy`] exactly: whenever the redaction algorithm changes, the
/// reference hash's input changes with it.
///
/// Room version 11 changed the redaction algorithm (see `rooms/v11`: the
/// top-level `origin`, `membership`, and `prev_state` properties are no longer
/// protected, `m.room.create` keeps its entire `content`, `m.room.redaction`
/// keeps `redacts` under `content`, and `m.room.power_levels` keeps `invite`).
/// Version 12 inherits it. That is why v9/v10 and v11+ cannot share one
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceHashInputPolicy {
    /// Event IDs are server-assigned, so no reference-hash input exists.
    NotApplicable,
    /// Room version 3 to 5 (V1 and V2 use server-assigned event IDs).
    V1ToV5,
    /// Room version 6 to 8.
    V6ToV8,
    /// Room version 9 to 10.
    V9ToV10,
    /// Room version 11 and later.
    V11Plus,
}

/// Versioned Matrix redaction content table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactionPolicy {
    /// Original room-version-1 through -5 rules.
    V1ToV5,
    /// Version 6 through 8 rules.
    V6ToV8,
    /// Version 9 through 10 rules.
    V9ToV10,
    /// Version 11 and later rules.
    V11Plus,
}

/// How a Matrix room derives its friendly room ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomIdPolicy {
    /// `!localpart:server-name` is assigned by a homeserver.
    ServerAssigned,
    /// Version 12 derives the room ID from the accepted create event ID.
    CreateEventId,
}

/// The state-resolution algorithm required by a Matrix room version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateResolutionPolicy {
    /// Matrix State Resolution v1.
    V1,
    /// Matrix State Resolution v2.
    V2,
    /// Matrix State Resolution v2.1.
    V2_1,
}

/// Matrix room versions whose event-format boundary matters to the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixRoomVersion {
    /// Room version 1.
    V1,
    /// Room version 2.
    V2,
    /// Room version 3.
    V3,
    /// Room version 4.
    V4,
    /// Room version 5.
    V5,
    /// Room version 6.
    V6,
    /// Room version 7.
    V7,
    /// Room version 8.
    V8,
    /// Room version 9.
    V9,
    /// Room version 10.
    V10,
    /// Room version 11.
    V11,
    /// Room version 12.
    V12,
}

impl MatrixRoomVersion {
    /// Parse a supported room-version string.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "1" => Some(Self::V1),
            "2" => Some(Self::V2),
            "3" => Some(Self::V3),
            "4" => Some(Self::V4),
            "5" => Some(Self::V5),
            "6" => Some(Self::V6),
            "7" => Some(Self::V7),
            "8" => Some(Self::V8),
            "9" => Some(Self::V9),
            "10" => Some(Self::V10),
            "11" => Some(Self::V11),
            "12" | "12.1" => Some(Self::V12),
            _ => None,
        }
    }

    /// Event identity rules for this version.
    #[must_use]
    pub const fn event_id_policy(self) -> EventIdPolicy {
        match self {
            Self::V1 | Self::V2 => EventIdPolicy::ServerAssigned,
            _ => EventIdPolicy::ReferenceHash,
        }
    }

    /// Reference-hash event-ID encoding for this version.
    #[must_use]
    pub const fn reference_hash_encoding(self) -> ReferenceHashEncoding {
        match self {
            Self::V1 | Self::V2 => ReferenceHashEncoding::NotApplicable,
            Self::V3 => ReferenceHashEncoding::StandardBase64NoPad,
            _ => ReferenceHashEncoding::UrlSafeBase64NoPad,
        }
    }

    /// State-resolution rules for this version.
    #[must_use]
    pub const fn state_resolution_policy(self) -> StateResolutionPolicy {
        match self {
            Self::V1 => StateResolutionPolicy::V1,
            Self::V12 => StateResolutionPolicy::V2_1,
            _ => StateResolutionPolicy::V2,
        }
    }

    /// Versioned redaction content rules.
    #[must_use]
    pub const fn redaction_policy(self) -> RedactionPolicy {
        match self {
            Self::V1 | Self::V2 | Self::V3 | Self::V4 | Self::V5 => RedactionPolicy::V1ToV5,
            Self::V6 | Self::V7 | Self::V8 => RedactionPolicy::V6ToV8,
            Self::V9 | Self::V10 => RedactionPolicy::V9ToV10,
            Self::V11 | Self::V12 => RedactionPolicy::V11Plus,
        }
    }

    /// Versioned reference hash input rules.
    #[must_use]
    pub const fn reference_hash_input_policy(self) -> ReferenceHashInputPolicy {
        match self {
            Self::V1 | Self::V2 => ReferenceHashInputPolicy::NotApplicable,
            Self::V3 | Self::V4 | Self::V5 => ReferenceHashInputPolicy::V1ToV5,
            Self::V6 | Self::V7 | Self::V8 => ReferenceHashInputPolicy::V6ToV8,
            Self::V9 | Self::V10 => ReferenceHashInputPolicy::V9ToV10,
            Self::V11 | Self::V12 => ReferenceHashInputPolicy::V11Plus,
        }
    }

    /// Friendly room-ID derivation for this version.
    #[must_use]
    pub const fn room_id_policy(self) -> RoomIdPolicy {
        match self {
            Self::V12 => RoomIdPolicy::CreateEventId,
            _ => RoomIdPolicy::ServerAssigned,
        }
    }

    /// The oldest room version mtxdb will import.
    ///
    /// v1/v2 event IDs are server-assigned, so a logical id cannot be
    /// re-derived from the payload and two distinct payloads may claim the
    /// same id. v3 is content-addressed but encodes with standard
    /// (non-URL-safe) base64, whose `/` and `+` break request paths and
    /// reverse proxies. v4+ is content-addressed with URL-safe unpadded
    /// base64, so the event id is always recomputable and collision-free.
    pub const MIN_SUPPORTED: Self = Self::V4;

    /// Whether this version meets [`Self::MIN_SUPPORTED`].
    ///
    /// Fail-open for future versions: only the three known-unsuitable
    /// versions are rejected.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        !matches!(self, Self::V1 | Self::V2 | Self::V3)
    }

    /// RFC 6901 pointer to the field of a room's `m.room.create` event that
    /// carries the collection's canonical (external) id.
    ///
    /// v12 derives the room id from the accepted create event's event id, so
    /// the source is `/event_id`; every earlier version stores a server-
    /// assigned `/room_id` on the create event itself.
    #[must_use]
    pub const fn collection_key_pointer(self) -> &'static str {
        match self {
            Self::V12 => "/event_id",
            _ => "/room_id",
        }
    }

    /// Normalize an establishment record's collection-key value into the form
    /// ordinary events reference it.
    ///
    /// Pre-v12 the create event carries its server-assigned `room_id`, which
    /// later events reference verbatim. Room version 12 derives the room id
    /// from the create event's id by replacing the `$` event sigil with `!`
    /// (MSC4291), so the raw `/event_id` value must be normalized before a
    /// later batch of ordinary events — whose `room_id` is `!<hash>` —
    /// derives the same collection identity.
    #[must_use]
    pub fn normalize_collection_identity(self, source: &str) -> String {
        match self {
            Self::V12 => match source.strip_prefix('$') {
                Some(rest) => format!("!{rest}"),
                None => source.to_owned(),
            },
            _ => source.to_owned(),
        }
    }

    /// Whether verification must use strict canonical-number validation.
    #[must_use]
    pub const fn requires_strict_canonical_numbers(self) -> bool {
        matches!(
            self,
            Self::V6 | Self::V7 | Self::V8 | Self::V9 | Self::V10 | Self::V11 | Self::V12
        )
    }

    /// Whether the room version supports the `knock` join rule.
    #[must_use]
    pub const fn supports_knock(self) -> bool {
        matches!(
            self,
            Self::V7 | Self::V8 | Self::V9 | Self::V10 | Self::V11 | Self::V12
        )
    }

    /// Whether the room version supports `restricted` joins.
    #[must_use]
    pub const fn supports_restricted_join(self) -> bool {
        matches!(
            self,
            Self::V8 | Self::V9 | Self::V10 | Self::V11 | Self::V12
        )
    }

    /// Whether the room version supports `knock_restricted` joins.
    #[must_use]
    pub const fn supports_knock_restricted_join(self) -> bool {
        matches!(self, Self::V10 | Self::V11 | Self::V12)
    }
}

/// Metadata extracted from a room's primordial `m.room.create` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomMetadata {
    /// Human-facing Matrix room ID; distinct from mtxdb's internal digest key.
    pub room_id: String,
    /// Server-name component of `room_id` for pre-v12 rooms.
    pub origin_domain: String,
    /// The immutable room-version binding.
    pub version: MatrixRoomVersion,
    /// Sender of the primordial create event.
    pub creator: String,
    /// Additional v12 creators; empty for earlier versions.
    pub additional_creators: Vec<String>,
}

/// Matrix storage profile defaults for a shared-WAL database.
///
/// Under the Matrix storage model:
/// - `State`: HAMT nodes, roots, and state-group sidecars are dense hashes that
///   do not benefit from zstd; compression is disabled to save CPU cycles on write and replay.
/// - `EventDag`: Event JSON benefits significantly from zstd; compression is enabled.
/// - `Edges`: Edge records retain standard defaults.
#[cfg(feature = "multi-reader")]
#[must_use]
pub fn matrix_pool_policies() -> crate::database::PoolPolicies {
    crate::database::PoolPolicies {
        state: crate::database::PoolPolicy {
            compress: false,
            checksum_policy: crate::packfile::ChecksumPolicy::Full,
        },
        event_dag: crate::database::PoolPolicy::default(),
        edges: crate::database::PoolPolicy::default(),
        server_info: crate::database::PoolPolicy {
            compress: false,
            checksum_policy: crate::packfile::ChecksumPolicy::Full,
        },
    }
}

/// A Matrix storage workload with its own retention/durability/ordering policy.
///
/// These are the per-record classes the Matrix adapter assigns to each hot
/// store. Immutable history (events, DAG edges, state groups) shares the engine
/// with update-heavy projections; only the [`RecordClass`] differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixRecordClass {
    /// Presence heartbeats; latest value only and disposable on crash.
    Presence,
    /// Read receipts; the latest value is normally sufficient.
    Receipts,
    /// Per-user/per-room account data; keeps version history.
    AccountData,
    /// Current room state and membership projections.
    CurrentState,
    /// Events and DAG edges; immutable topologically and stream ordered history.
    Events,
    /// Per-room state-group instances: one per derivation.
    ///
    /// The layers stay distinct:
    /// - the instance id is unique per derivation/version;
    /// - stored parent links form the DAG topology;
    /// - the state-group `LtHash` is the state identity;
    /// - the HAMT root id references shared, content-addressed materialization.
    ///
    /// Parents must precede descendants during materialization/replay. A merge
    /// may have several predecessors, so this is a partial order, not a linear
    /// sequence. Successors need not exist when a group is written; the stored
    /// parent links are enough to order it later.
    StateGroup,
    /// Shared, content-addressed HAMT roots and nodes.
    ///
    /// Equal state sets resolve to the same root, so this materialization is
    /// deduplicated across state-group instances and carries no ordering of its
    /// own.
    StateHamtContent,
    /// `event → state-group` mappings, stored as compact root-id links.
    ///
    /// This is a pure lookup index; the referenced state group carries the
    /// ordering, not the mapping.
    StateGroupMapping,
}

impl MatrixRecordClass {
    /// The engine-level record class for this Matrix workload.
    #[must_use]
    pub const fn record_class(self) -> RecordClass {
        match self {
            Self::Presence => RecordClass::new(
                Retention::EphemeralLatest,
                Durability::Volatile,
                OrderingPolicy::None,
            ),
            Self::Receipts => RecordClass::new(
                Retention::LatestOnly,
                Durability::GroupCommit,
                OrderingPolicy::Stream,
            ),
            Self::AccountData => RecordClass::new(
                Retention::Versioned,
                Durability::GroupCommit,
                OrderingPolicy::None,
            ),
            Self::CurrentState => RecordClass::new(
                Retention::Versioned,
                Durability::Synchronous,
                OrderingPolicy::None,
            ),
            Self::Events => RecordClass::new(
                Retention::Immutable,
                Durability::Synchronous,
                OrderingPolicy::TopologicalAndStream,
            ),
            Self::StateGroup => RecordClass::new(
                Retention::Immutable,
                Durability::Synchronous,
                OrderingPolicy::Topological,
            ),
            // Both are write-once: shared HAMT content dedupes by root, and the
            // event mapping is a single link per key.
            Self::StateHamtContent | Self::StateGroupMapping => RecordClass::new(
                Retention::Immutable,
                Durability::Synchronous,
                OrderingPolicy::None,
            ),
        }
    }
}

#[cfg(test)]
#[path = "test_matrix_policy.rs"]
mod tests;
