//! Matrix-specific room-version policy used by the Matrix import adapter.
//!
//! This is deliberately outside `mtxdb-core`: packs and collection templates
//! are protocol-neutral, while redaction, authorization, and room-version
//! rules are Matrix wire semantics.
//!
//! The current archive importer does not yet evaluate every policy below. They
//! remain together here because a Matrix adapter must use one coherent,
//! versioned policy table when verification and state handling are added.

#![allow(
    dead_code,
    unreachable_pub,
    reason = "the Matrix adapter keeps its complete version-policy table ahead of optional verification/state features"
)]

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceHashInputPolicy {
    /// Event IDs are server-assigned, so no reference-hash input exists.
    NotApplicable,
    /// Room version 3 to 5 (V1 and V2 use server-assigned event IDs).
    V1ToV5,
    /// Room version 6 to 8.
    V6ToV8,
    /// Room version 9 and later.
    V9Plus,
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
            _ => ReferenceHashInputPolicy::V9Plus,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_v2_v3_identity_boundaries_are_explicit() {
        assert_eq!(
            MatrixRoomVersion::V1.event_id_policy(),
            EventIdPolicy::ServerAssigned
        );
        assert_eq!(
            MatrixRoomVersion::V2.state_resolution_policy(),
            StateResolutionPolicy::V2
        );
        assert_eq!(
            MatrixRoomVersion::V1.reference_hash_input_policy(),
            ReferenceHashInputPolicy::NotApplicable
        );
        assert_eq!(
            MatrixRoomVersion::V3.event_id_policy(),
            EventIdPolicy::ReferenceHash
        );
        assert!(!MatrixRoomVersion::V3.requires_strict_canonical_numbers());
        assert_eq!(
            MatrixRoomVersion::V4.reference_hash_encoding(),
            ReferenceHashEncoding::UrlSafeBase64NoPad
        );
        assert_eq!(
            MatrixRoomVersion::V6.redaction_policy(),
            RedactionPolicy::V6ToV8
        );
        assert!(MatrixRoomVersion::V8.supports_restricted_join());
        assert!(MatrixRoomVersion::V10.supports_knock_restricted_join());
        assert_eq!(
            MatrixRoomVersion::V12.room_id_policy(),
            RoomIdPolicy::CreateEventId
        );
        assert_eq!(
            MatrixRoomVersion::V12.state_resolution_policy(),
            StateResolutionPolicy::V2_1
        );
    }
}
